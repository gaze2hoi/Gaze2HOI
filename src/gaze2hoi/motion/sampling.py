"""Deterministic DDIM sampling with explicit observation preservation."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn

from .assets import NormStats


def draw_seed(base_seed: int, sample_id: str, draw: int) -> int:
    """Map a sample and zero-based draw index to the released noise seed."""
    payload = f"gaze2hoi-table2-v1\0{base_seed}\0{sample_id}\0{draw}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def initial_noise(sample_id: str, draw: int, seed: int = 999) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(draw_seed(seed, sample_id, draw))
    return torch.randn((1, 60, 66), generator=generator)


class DDIMSampler(nn.Module):
    """Cosine schedule with velocity prediction, 50 steps, and eta zero."""

    def __init__(self, stats: NormStats, timesteps: int = 1000) -> None:
        super().__init__()
        self.timesteps = timesteps
        steps = np.arange(timesteps + 1, dtype=np.float64)
        f = np.cos((steps / timesteps + 0.008) / 1.008 * math.pi / 2.0) ** 2
        alpha_bar = f / f[0]
        alphas = np.clip(alpha_bar[1:] / alpha_bar[:-1], 0.001, 0.9999)
        alpha_bar = np.cumprod(alphas)
        self.register_buffer(
            "sqrt_alpha_bar", torch.from_numpy(np.sqrt(alpha_bar)).float(), persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_bar",
            torch.from_numpy(np.sqrt(1.0 - alpha_bar)).float(),
            persistent=False,
        )
        self.register_buffer("state_mean", stats.state_mean_t().float(), persistent=False)
        self.register_buffer("state_std", stats.state_std_t().float(), persistent=False)

    def _ab(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sqrt_alpha_bar[t][:, None, None], self.sqrt_one_minus_alpha_bar[t][
            :, None, None
        ]

    def denormalize(self, state: torch.Tensor) -> torch.Tensor:
        return state * self.state_std + self.state_mean

    @torch.inference_mode()
    def forward(
        self,
        model: nn.Module,
        batch: dict,
        noise: torch.Tensor,
        steps: int = 50,
        guidance: float = 1.5,
    ) -> torch.Tensor:
        keep = batch["observed"][..., None]
        observation = batch["state"]
        x = keep * observation + (1.0 - keep) * noise.to(observation.device)
        timesteps = torch.linspace(self.timesteps - 1, 0, steps, device=x.device).round().long()
        null_intent = torch.zeros_like(batch["intent_ids"])
        null_place = torch.zeros_like(batch["place_target_valid"])
        for index, timestep in enumerate(timesteps):
            t = timestep.expand(len(x))
            conditional = model(x, t, batch["observed"], batch)
            unconditional = model(
                x,
                t,
                batch["observed"],
                batch,
                intent_ids=null_intent,
                place_target_valid=null_place,
            )
            prediction = unconditional + guidance * (conditional - unconditional)
            alpha, sigma = self._ab(t)
            clean = alpha * x - sigma * prediction
            epsilon = sigma * x + alpha * prediction
            if index + 1 < len(timesteps):
                alpha, sigma = self._ab(timesteps[index + 1].expand(len(x)))
                x = alpha * clean + sigma * epsilon
            else:
                x = clean
            x = keep * observation + (1.0 - keep) * x
        return self.denormalize(x)
