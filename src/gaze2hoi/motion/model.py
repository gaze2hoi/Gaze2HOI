"""Factorized diffusion transformer with explicit contact conditioning."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from . import config as C
from .assets import NormStats, load_template_bank
from .config import ModelConfig
from .encoders import (
    N_INTENT_TOKENS,
    FurnitureEncoder,
    IntentEncoder,
    ObjectEncoder,
    _modulate,
    mlp,
    timestep_embedding,
)
from .geometry import _rot6d_to_matrix

N_ENTITIES = 4
ENTITY_SLICES = (C.LEFT_HAND_SLICE, C.RIGHT_HAND_SLICE, C.LEFT_OBJ_SLICE, C.RIGHT_OBJ_SLICE)
N_GOAL_TOKENS = 2


def _fps_indices(points: torch.Tensor, g: int) -> torch.Tensor:
    """Deterministic farthest point sampling: (P, 3) -> (g,) indices."""
    p = points.shape[0]
    sel = torch.zeros(g, dtype=torch.long)
    dist = torch.full((p,), float("inf"), dtype=points.dtype)
    cur = 0
    for k in range(g):
        sel[k] = cur
        d = ((points - points[cur]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        cur = int(dist.argmax())
    return sel


def build_surface_groups(
    points_local: torch.Tensor, n_groups: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FPS partition of every object's template surface."""
    n, p, _ = points_local.shape
    group_index = torch.zeros(n, p, dtype=torch.long)
    centroid = torch.zeros(n, n_groups, 3)
    radius = torch.zeros(n, n_groups)
    for i in range(n):
        pts = points_local[i]
        centers = pts[_fps_indices(pts, n_groups)]
        gid = torch.cdist(pts, centers).argmin(dim=-1)
        group_index[i] = gid
        for gk in range(n_groups):
            member = pts[gid == gk]
            if member.numel() == 0:
                centroid[i, gk] = centers[gk]
                continue
            c = member.mean(dim=0)
            centroid[i, gk] = c
            radius[i, gk] = ((member - c) ** 2).sum(-1).mean().sqrt()
    return (group_index, centroid, radius)


class SurfaceEncoder(nn.Module):
    """Per-hand bound-object surface tokens: ``n_groups`` tokens per hand."""

    def __init__(self, d_model: int, stats: NormStats, n_groups: int, asset_dir: Path) -> None:
        super().__init__()
        self.n_groups = n_groups
        bank = load_template_bank(asset_dir)
        points_local = bank["points"] * bank["scale"][:, None, None] + bank["offset"][:, None, :]
        gidx, cent, rad = build_surface_groups(points_local, n_groups)
        self.register_buffer("group_index", gidx, persistent=False)
        self.register_buffer("centroid", cent, persistent=False)
        self.register_buffer("radius", rad, persistent=False)
        self.register_buffer(
            "scene_mean", torch.from_numpy(stats.scene_transl_mean).float(), persistent=False
        )
        self.register_buffer(
            "scene_std", torch.from_numpy(stats.scene_transl_std).float(), persistent=False
        )

        def obj_stats(base: slice) -> tuple[torch.Tensor, torch.Tensor]:
            sl = slice(base.start + C.OBJ_TRANSL_SLICE.start, base.start + C.OBJ_TRANSL_SLICE.stop)
            return (
                torch.from_numpy(stats.state_mean[sl]).float(),
                torch.from_numpy(stats.state_std[sl]).float(),
            )

        lm, ls = obj_stats(C.LEFT_OBJ_SLICE)
        rm, rs = obj_stats(C.RIGHT_OBJ_SLICE)
        self.register_buffer("reach_mean", torch.stack([lm, rm]), persistent=False)
        self.register_buffer("reach_std", torch.stack([ls, rs]), persistent=False)
        self.id_emb = nn.Embedding(C.NUM_OBJECT_SLOTS, d_model)
        self.hand_emb = nn.Embedding(2, d_model)
        self.fuse = mlp(12, 2 * d_model, d_model)

    def _pool_pdf(self, pdf: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """(B, 2, P) point pdf -> (B, 2, G) per-group mass."""
        gidx = self.group_index[slots]
        mass = pdf.new_zeros(*pdf.shape[:2], self.n_groups)
        mass.scatter_add_(2, gidx, pdf)
        return mass

    def forward(
        self, batch: dict[str, torch.Tensor], use_uncond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (tokens (B, 2*G, d), keep mask (B, 2*G) bool)."""
        slots_raw = torch.stack([batch["left_obj_slot"], batch["right_obj_slot"]], dim=1)
        slots = slots_raw.clamp_min(0)
        bound = slots_raw >= 0
        b = slots.shape[0]
        contact = torch.where(
            use_uncond[:, None, None], batch["acg_contact_uncond"], batch["acg_contact_cond"]
        )
        acg_ok = batch["acg_hand_valid"] > 0.5
        bottom_ok = batch["acg_bottom_valid"] > 0.5
        g = self.n_groups
        c_mass = torch.log1p(g * self._pool_pdf(contact, slots))
        c_mass = c_mass * acg_ok[..., None]
        b_mass = torch.log1p(g * self._pool_pdf(batch["acg_bottom"], slots))
        b_mass = b_mass * bottom_ok[..., None]
        t_norm = torch.gather(batch["scene_transl_last"], 1, slots[..., None].expand(-1, -1, 3))
        r6 = torch.gather(batch["scene_rot6d_last"], 1, slots[..., None].expand(-1, -1, 6))
        present = torch.gather(batch["scene_present"], 1, slots) > 0.5
        t_metric = t_norm * self.scene_std + self.scene_mean
        rot = _rot6d_to_matrix(r6)
        cent = self.centroid[slots]
        reach = torch.einsum("bhij,bhgj->bhgi", rot, cent) + t_metric[:, :, None]
        reach = (reach - self.reach_mean[None, :, None]) / self.reach_std[None, :, None]
        pose_ok = bound & present
        reach = reach * pose_ok[..., None, None]
        feat = torch.cat(
            [
                cent,
                self.radius[slots][..., None],
                c_mass[..., None],
                b_mass[..., None],
                reach,
                (acg_ok & bound)[..., None, None].float().expand(-1, -1, g, 1),
                (bottom_ok & bound)[..., None, None].float().expand(-1, -1, g, 1),
                pose_ok[..., None, None].float().expand(-1, -1, g, 1),
            ],
            dim=-1,
        )
        tok = self.fuse(feat) + self.id_emb(slots)[:, :, None] + self.hand_emb.weight[None, :, None]
        keep = bound[..., None].expand(-1, -1, g)
        tok = tok * keep[..., None]
        return (tok.reshape(b, 2 * g, -1), keep.reshape(b, 2 * g))


class GoalEncoder(nn.Module):
    """Per-hand navigation-goal token + pooled global embedding."""

    def __init__(self, d_model: int, d_cond: int, d_feat: int = 128) -> None:
        super().__init__()
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 16, d_feat),
        )
        self.fuse = mlp(3 + 2 + d_feat + 2 + 1, 2 * d_model, d_model)
        self.hand_emb = nn.Embedding(2, d_model)
        self.null_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.null_token, std=0.02)
        self.pool_proj = mlp(d_model, d_model, d_cond)

    def forward(
        self,
        place_target: torch.Tensor,
        place_valid: torch.Tensor,
        move_target: torch.Tensor,
        move_valid: torch.Tensor,
        landing: torch.Tensor,
        landing_valid: torch.Tensor,
        landing_yaw_sincos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, gn, _ = landing.shape
        gfeat = self.grid_cnn(landing.reshape(b * 2, 1, gn, gn) * (gn * gn) / 50.0).view(b, 2, -1)
        gfeat = gfeat * (landing_valid > 0.5)[..., None]
        is_place = (place_valid > 0.5).float()[..., None]
        is_move = (move_valid > 0.5).float()[..., None] * (1.0 - is_place)
        goal_pos = is_place * place_target + is_move * move_target
        tok = self.fuse(
            torch.cat(
                [goal_pos, is_place, is_move, gfeat, landing_yaw_sincos, landing_valid[..., None]],
                dim=-1,
            )
        )
        tok = tok + self.hand_emb.weight[None]
        valid = torch.clamp(is_place + is_move, max=1.0)
        tok = valid * tok + (1.0 - valid) * self.null_token
        pooled = self.pool_proj(tok.mean(dim=1))
        return (tok, pooled)


class FactorizedBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_cond: int, mlp_ratio: float, dropout: float
    ) -> None:
        super().__init__()
        self.norm_t = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn_t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_e = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn_e = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_x = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn_x = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_f = nn.LayerNorm(d_model, elementwise_affine=False)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model)
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 12 * d_model))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, memory: torch.Tensor, memory_pad: torch.Tensor
    ) -> torch.Tensor:
        """x: (B, T, E, d)."""
        b, t, e, d = x.shape
        mods = self.ada(cond).chunk(12, dim=-1)
        h = _modulate(self.norm_t(x).view(b, t * e, d), mods[0], mods[1])
        h = h.view(b, t, e, d).permute(0, 2, 1, 3).reshape(b * e, t, d)
        h, _ = self.attn_t(h, h, h, need_weights=False)
        h = h.view(b, e, t, d).permute(0, 2, 1, 3)
        x = x + mods[2].view(b, 1, 1, d) * h
        h = _modulate(self.norm_e(x).view(b, t * e, d), mods[3], mods[4])
        h = h.view(b * t, e, d)
        h, _ = self.attn_e(h, h, h, need_weights=False)
        x = x + mods[5].view(b, 1, 1, d) * h.view(b, t, e, d)
        h = _modulate(self.norm_x(x).view(b, t * e, d), mods[6], mods[7])
        h, _ = self.attn_x(h, memory, memory, key_padding_mask=memory_pad, need_weights=False)
        x = x + mods[8].view(b, 1, 1, d) * h.view(b, t, e, d)
        h = _modulate(self.norm_f(x).view(b, t * e, d), mods[9], mods[10])
        x = x + mods[11].view(b, 1, 1, d) * self.ffn(h).view(b, t, e, d)
        return x


class OursDiT(nn.Module):
    """Frame x entity factorized DiT with surface-group conditioning."""

    uses_acg = True

    def __init__(
        self,
        cfg: ModelConfig,
        intent_vocab_sizes: dict[str, int],
        n_furniture_types: int,
        stats: NormStats,
        asset_dir: Path,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        d, dc = (cfg.d_model, cfg.d_cond)
        self.n_surface = 2 * cfg.n_surface_groups
        self.hand_proj = nn.Linear(C.HAND_DIM, d)
        self.obj_proj = nn.Linear(C.OBJ_DIM, d)
        self.time_pos_emb = nn.Embedding(C.WINDOW_LEN, d)
        self.entity_emb = nn.Embedding(N_ENTITIES, d)
        self.observed_emb = nn.Embedding(2, d)
        self.valid_emb = nn.Embedding(2, d)
        self.intent_enc = IntentEncoder(intent_vocab_sizes, d, dc)
        self.object_enc = ObjectEncoder(d, cfg.d_pointnet, asset_dir)
        self.furniture_enc = FurnitureEncoder(n_furniture_types, d)
        self.goal_enc = GoalEncoder(d, dc)
        self.surface_enc = SurfaceEncoder(d, stats, cfg.n_surface_groups, asset_dir)
        self.segment_emb = nn.Embedding(5, d)
        self.t_proj = mlp(cfg.d_timestep, dc, dc)
        self.betas_proj = mlp(C.MANO_BETAS_DIM, dc, dc)
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dc, dc))
        self.blocks = nn.ModuleList(
            [
                FactorizedBlock(d, cfg.n_heads, dc, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dc, 2 * d))
        self.hand_head = nn.Linear(d, C.HAND_DIM)
        self.obj_head = nn.Linear(d, C.OBJ_DIM)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        for head in (self.hand_head, self.obj_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def build_memory(
        self, batch: dict[str, torch.Tensor], intent_ids: torch.Tensor, place_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        intent_tok, intent_pooled = self.intent_enc(intent_ids)
        obj_tok, obj_mask = self.object_enc(
            batch["scene_present"],
            batch["scene_transl_last"],
            batch["scene_rot6d_last"],
            batch["scene_transl_mean"],
            batch["left_obj_slot"],
            batch["right_obj_slot"],
        )
        furn_tok, furn_mask = self.furniture_enc(
            batch["furn_present"],
            batch["furn_type_id"],
            batch["furn_shape_id"],
            batch["furn_params"],
            batch["furn_center"],
            batch["furn_yaw_sincos"],
        )
        tgt_slot = C.INTENT_SLOTS.index("target")
        move_valid = batch["move_target_valid"] * (intent_ids[:, :, tgt_slot] != 0).float()
        goal_tok, goal_pooled = self.goal_enc(
            batch["place_target"],
            place_valid,
            batch["move_target"],
            move_valid,
            batch["acg_landing"],
            batch["acg_landing_valid"],
            batch["acg_landing_yaw_sincos"],
        )
        use_uncond = (intent_ids == 0).all(dim=2).all(dim=1)
        surf_tok, surf_keep = self.surface_enc(batch, use_uncond)
        seg = self.segment_emb.weight
        memory = torch.cat(
            [
                intent_tok + seg[0],
                obj_tok + seg[1],
                furn_tok + seg[2],
                goal_tok + seg[3],
                surf_tok + seg[4],
            ],
            dim=1,
        )
        b = memory.shape[0]
        always = torch.ones(b, dtype=torch.bool, device=memory.device)
        keep = torch.cat(
            [
                always[:, None].expand(b, N_INTENT_TOKENS),
                obj_mask,
                furn_mask,
                always[:, None].expand(b, N_GOAL_TOKENS),
                surf_keep,
            ],
            dim=1,
        )
        return (memory, ~keep, intent_pooled, goal_pooled)

    def forward(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
        observed: torch.Tensor,
        batch: dict[str, torch.Tensor],
        intent_ids: torch.Tensor | None = None,
        place_target_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intent_ids is None:
            intent_ids = batch["intent_ids"]
        if place_target_valid is None:
            place_target_valid = batch["place_target_valid"]
        b, n, _ = state.shape
        memory, memory_pad, intent_pooled, goal_pooled = self.build_memory(
            batch, intent_ids, place_target_valid
        )
        t_emb = self.t_proj(timestep_embedding(t, self.cfg.d_timestep))
        betas_emb = self.betas_proj(batch["mano_betas"])
        cond = self.cond_mlp(t_emb + intent_pooled + betas_emb + goal_pooled)
        toks = torch.stack(
            [
                self.hand_proj(state[..., ENTITY_SLICES[0]]),
                self.hand_proj(state[..., ENTITY_SLICES[1]]),
                self.obj_proj(state[..., ENTITY_SLICES[2]]),
                self.obj_proj(state[..., ENTITY_SLICES[3]]),
            ],
            dim=2,
        )
        pos = torch.arange(n, device=state.device)
        ent = torch.arange(N_ENTITIES, device=state.device)
        valid = torch.stack(
            [(batch["state_mask"][..., sl.start] > 0).long() for sl in ENTITY_SLICES], dim=2
        )
        x = (
            toks
            + self.time_pos_emb(pos)[None, :, None, :]
            + self.entity_emb(ent)[None, None, :, :]
            + self.observed_emb(observed.long())[:, :, None, :]
            + self.valid_emb(valid)
        )
        for block in self.blocks:
            x = block(x, cond, memory, memory_pad)
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x).view(b, n * N_ENTITIES, -1), shift, scale).view(
            b, n, N_ENTITIES, -1
        )
        out = state.new_zeros(b, n, C.STATE_DIM)
        out[..., ENTITY_SLICES[0]] = self.hand_head(x[:, :, 0])
        out[..., ENTITY_SLICES[1]] = self.hand_head(x[:, :, 1])
        out[..., ENTITY_SLICES[2]] = self.obj_head(x[:, :, 2])
        out[..., ENTITY_SLICES[3]] = self.obj_head(x[:, :, 3])
        return out
