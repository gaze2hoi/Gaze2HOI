"""Anticipatory contact grounding network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .assets import N_LEVELS
from .config import ACGConfig

TRAJ_LEN = 10
DIST_SCALE = 3.0
LOG_EPS = 1e-6
DENSITY_CLIP = 20.0


def _mlp(dims, dropout: float = 0.0, out_act: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or out_act:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _density_feats(pdf: torch.Tensor) -> torch.Tensor:
    """(B, N) pdf -> (B, N, 2) [log relative density | tanh density]."""
    dens = pdf * pdf.shape[-1]
    return torch.stack((torch.log(dens + 0.001), torch.tanh(dens / DENSITY_CLIP)), dim=-1)


class ACGNet(nn.Module):
    def __init__(
        self, cfg: ACGConfig, points_local: torch.Tensor, bottom_prior: torch.Tensor
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("points", points_local, persistent=False)
        self.register_buffer("bottom_prior", bottom_prior, persistent=False)
        self.point_mlp = _mlp([8, cfg.d_point, cfg.d_point], cfg.dropout, out_act=True)
        self.traj_enc = _mlp([TRAJ_LEN * 7, 128, cfg.d_traj], cfg.dropout, out_act=True)
        n_scalar = 4 + N_LEVELS + 3 + 1 + 2
        g_in = 2 * cfg.d_point + cfg.d_traj + n_scalar
        self.global_norm = nn.LayerNorm(g_in)
        self.global_mlp = _mlp([g_in, 256, cfg.d_global], cfg.dropout, out_act=True)
        self.contact_gate = nn.Linear(cfg.d_global, 2)
        self.contact_delta = _mlp([cfg.d_point + cfg.d_global, cfg.d_point, 1], cfg.dropout)
        self.bottom_mlp = _mlp([8, cfg.d_point, cfg.d_point], cfg.dropout, out_act=True)
        self.bottom_gate = nn.Linear(cfg.d_global, 1)
        self.bottom_delta = _mlp([cfg.d_point + cfg.d_global, cfg.d_point, 1], cfg.dropout)
        ch = cfg.landing_ch
        land_in = cfg.d_global + 2 * cfg.d_point + 2 + 1
        self.landing_fc = _mlp([land_in, 256, ch * 4 * 4], cfg.dropout, out_act=True)
        self.landing_deconv = nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(ch // 2, ch // 4, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(ch // 4, ch // 8, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(ch // 8, 1, 3, 1, 1),
        )
        for head in (self.contact_delta, self.bottom_delta):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.landing_deconv[-1].weight)
        nn.init.zeros_(self.landing_deconv[-1].bias)
        nn.init.zeros_(self.contact_gate.weight)
        nn.init.zeros_(self.contact_gate.bias)
        nn.init.zeros_(self.bottom_gate.weight)
        nn.init.constant_(self.bottom_gate.bias, 0.55)

    def forward(
        self, batch: dict[str, torch.Tensor], prior: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """``prior`` is passed explicitly so the caller controls intent drop."""
        pts = self.points[batch["slot"].clamp_min(0)]
        b, n, _ = pts.shape
        log_prior = torch.log(prior + LOG_EPS)
        log_obs = torch.log(batch["obs"] + LOG_EPS)
        d_wrist = torch.linalg.norm(pts - batch["wrist_last"][:, None, :], dim=-1) * DIST_SCALE
        d_wrist = d_wrist * batch["wrist_valid"][:, None]
        pfeat = torch.cat(
            [pts * 3.0, _density_feats(prior), _density_feats(batch["obs"]), d_wrist[..., None]],
            dim=-1,
        )
        pfeat = self.point_mlp(pfeat)
        scalars = torch.cat(
            [
                batch["scalars_c"],
                F.one_hot(batch["level_c"].clamp(0, N_LEVELS - 1), N_LEVELS).float(),
                batch["obs_valid"][:, None],
                batch["obs_frac"][:, None],
                batch["obs_area"][:, None] * 10.0,
                batch["wrist_valid"][:, None],
                F.one_hot(batch["side"], 2).float(),
            ],
            dim=-1,
        )
        g = torch.cat(
            [pfeat.max(dim=1).values, pfeat.mean(dim=1), self.traj_enc(batch["traj"]), scalars],
            dim=-1,
        )
        g = self.global_mlp(self.global_norm(g))
        gates = F.softplus(self.contact_gate(g))
        delta = self.contact_delta(
            torch.cat([pfeat, g[:, None, :].expand(-1, n, -1)], dim=-1)
        ).squeeze(-1)
        contact_logits = gates[:, :1] * log_prior + gates[:, 1:] * log_obs + delta
        bprior = self.bottom_prior[batch["slot"].clamp_min(0)]
        log_bprior = torch.log(bprior + LOG_EPS)
        down_h = (pts * batch["down_local"][:, None, :]).sum(dim=-1)
        bfeat = torch.cat(
            [
                pts * 3.0,
                (down_h * 3.0)[..., None],
                _density_feats(bprior),
                log_prior.clamp_min(-14.0)[..., None] / 7.0,
                log_obs.clamp_min(-14.0)[..., None] / 7.0,
            ],
            dim=-1,
        )
        bfeat = self.bottom_mlp(bfeat)
        bgate = F.softplus(self.bottom_gate(g))
        bdelta = self.bottom_delta(
            torch.cat([bfeat, g[:, None, :].expand(-1, n, -1)], dim=-1)
        ).squeeze(-1)
        bottom_logits = bgate * log_bprior + bdelta
        land_in = torch.cat(
            [
                g,
                bfeat.max(dim=1).values,
                bfeat.mean(dim=1),
                batch["heading_rel"],
                batch["goal_dist"][:, None] * 2.0,
            ],
            dim=-1,
        )
        h = self.landing_fc(land_in).view(b, self.cfg.landing_ch, 4, 4)
        landing_logits = self.landing_deconv(h).squeeze(1)
        return {
            "contact_logits": contact_logits,
            "bottom_logits": bottom_logits,
            "landing_logits": landing_logits,
            "gates": gates,
            "bottom_gate": bgate,
        }
