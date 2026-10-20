"""Scene and intention encoders for motion forecasting."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from . import config as C
from .assets import load_template_bank

N_INTENT_TOKENS = 2 * C.NUM_INTENT_SLOTS


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of diffusion timesteps ``t`` (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, emb[:, :1] * 0], dim=-1)
    return emb


def mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
    )


class PointNet(nn.Module):
    """Canonical object point-cloud encoder."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, out_dim)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points).max(dim=1).values


class IntentEncoder(nn.Module):
    """(B, 2, 5) slot ids -> 10 memory tokens + pooled global embedding."""

    def __init__(self, vocab_sizes: dict[str, int], d_model: int, d_cond: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_sizes[slot], d_model) for slot in C.INTENT_SLOTS]
        )
        self.hand_emb = nn.Embedding(2, d_model)
        self.slot_emb = nn.Embedding(C.NUM_INTENT_SLOTS, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.pool_proj = mlp(d_model, d_model, d_cond)

    def forward(self, intent_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = intent_ids.shape[0]
        toks = []
        for hand in range(2):
            for k, emb in enumerate(self.embeddings):
                tok = emb(intent_ids[:, hand, k])
                tok = tok + self.hand_emb.weight[hand] + self.slot_emb.weight[k]
                toks.append(tok)
        tokens = self.out_proj(torch.stack(toks, dim=1))
        pooled = self.pool_proj(tokens.mean(dim=1))
        assert tokens.shape == (b, N_INTENT_TOKENS, tokens.shape[-1])
        return (tokens, pooled)


class ObjectEncoder(nn.Module):
    """33 global object slots -> 33 memory tokens (+ presence mask)."""

    def __init__(self, d_model: int, d_pointnet: int, asset_dir: Path) -> None:
        super().__init__()
        bank = load_template_bank(asset_dir)
        self.register_buffer("template_points", bank["points"], persistent=False)
        self.register_buffer("template_scale", bank["scale"], persistent=False)
        self.id_emb = nn.Embedding(C.NUM_OBJECT_SLOTS, d_model)
        self.pointnet = PointNet(d_pointnet)
        in_dim = d_model + d_pointnet + 12 + 3 + 1
        self.fuse = mlp(in_dim, 2 * d_model, d_model)

    def forward(
        self,
        scene_present: torch.Tensor,
        scene_transl_last: torch.Tensor,
        scene_rot6d_last: torch.Tensor,
        scene_transl_mean: torch.Tensor,
        left_obj_slot: torch.Tensor,
        right_obj_slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = scene_present.shape[0]
        device = scene_present.device
        slots = torch.arange(C.NUM_OBJECT_SLOTS, device=device)
        geo_feat = self.pointnet(self.template_points)
        id_feat = self.id_emb(slots)
        static = (
            torch.cat([id_feat, geo_feat, self.template_scale[:, None]], dim=-1)
            .unsqueeze(0)
            .expand(b, -1, -1)
        )
        pose = torch.cat([scene_transl_last, scene_rot6d_last, scene_transl_mean], dim=-1)
        bound_l = (slots[None] == left_obj_slot[:, None]).float()
        bound_r = (slots[None] == right_obj_slot[:, None]).float()
        flags = torch.stack([scene_present, bound_l, bound_r], dim=-1)
        tokens = self.fuse(torch.cat([static, pose, flags], dim=-1))
        return (tokens, scene_present > 0.5)


class FurnitureEncoder(nn.Module):
    """Up to MAX_FURNITURE placed surfaces -> memory tokens (+ mask)."""

    def __init__(self, n_types: int, d_model: int) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(n_types, d_model)
        self.shape_emb = nn.Embedding(len(C.FURNITURE_SHAPES), d_model)
        self.fuse = mlp(d_model + 7, 2 * d_model, d_model)

    def forward(
        self,
        furn_present: torch.Tensor,
        furn_type_id: torch.Tensor,
        furn_shape_id: torch.Tensor,
        furn_params: torch.Tensor,
        furn_center: torch.Tensor,
        furn_yaw_sincos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.type_emb(furn_type_id) + self.shape_emb(furn_shape_id)
        feat = torch.cat([emb, furn_params, furn_center, furn_yaw_sincos], dim=-1)
        return (self.fuse(feat), furn_present > 0.5)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None]) + shift[:, None]
