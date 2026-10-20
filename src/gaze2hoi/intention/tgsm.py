"""Temporal gaze saliency modulation for Qwen3-VL visual embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .task import FRAME_COUNT, GAZE_CHANNELS, GAZE_GRID_SIZE

PROJECTION_WIDTH = 64
SCALE_LIMIT = 0.1


class GazeFeatureFiLM(nn.Module):
    """Gated additive and multiplicative modulation with checkpoint-stable names."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.bias_projection = nn.Sequential(
            nn.Linear(len(GAZE_CHANNELS), PROJECTION_WIDTH, bias=False),
            nn.GELU(),
            nn.Linear(PROJECTION_WIDTH, hidden_size, bias=False),
        )
        self.scale_projection = nn.Sequential(
            nn.Linear(len(GAZE_CHANNELS), PROJECTION_WIDTH, bias=False),
            nn.GELU(),
            nn.Linear(PROJECTION_WIDTH, hidden_size, bias=False),
        )
        self.final_bias_gate = nn.Parameter(torch.full((), 0.1))
        self.deepstack_bias_gate = nn.Parameter(torch.full((), 0.1))
        self.final_scale_gate = nn.Parameter(torch.full((), 0.1))
        self.deepstack_scale_gate = nn.Parameter(torch.full((), 0.1))

    def forward(self, embeddings: torch.Tensor, weights: torch.Tensor, path: str) -> torch.Tensor:
        if weights.shape != (embeddings.shape[0], len(GAZE_CHANNELS)):
            raise ValueError(
                f"Gaze shape {tuple(weights.shape)} does not match embeddings {tuple(embeddings.shape)}"
            )
        if path == "final":
            bias_gate, scale_gate = self.final_bias_gate, self.final_scale_gate
        elif path == "deepstack":
            bias_gate, scale_gate = self.deepstack_bias_gate, self.deepstack_scale_gate
        else:
            raise ValueError(f"Unknown gaze modulation path: {path}")
        weights = weights.to(device=embeddings.device, dtype=self.bias_projection[0].weight.dtype)
        bias = torch.tanh(bias_gate) * self.bias_projection(weights)
        scale = SCALE_LIMIT * torch.tanh(scale_gate) * torch.tanh(self.scale_projection(weights))
        return embeddings * (1.0 + scale.to(embeddings.dtype)) + bias.to(embeddings.dtype)


def pool_gaze_features(
    heatmaps: np.ndarray, image_grid_thw: torch.Tensor, merge_size: int
) -> torch.Tensor:
    """Pool gaze onto observation tokens and assign zero features to object references."""
    expected = (FRAME_COUNT, len(GAZE_CHANNELS), GAZE_GRID_SIZE, GAZE_GRID_SIZE)
    if heatmaps.shape != expected:
        raise ValueError(f"Expected gaze shape {expected}, received {heatmaps.shape}")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [number of images, 3]")
    if image_grid_thw.shape[0] < FRAME_COUNT or merge_size < 1:
        raise ValueError("Expected at least five image grids and a positive merge size")
    heat = torch.from_numpy(np.asarray(heatmaps, dtype=np.float32))
    features = []
    for index, (temporal, height, width) in enumerate(image_grid_thw.tolist()):
        if temporal != 1 or height % merge_size or width % merge_size:
            raise ValueError("Expected still-image grids divisible by the visual merge size")
        grid_height, grid_width = int(height) // merge_size, int(width) // merge_size
        token_count = grid_height * grid_width
        if index < FRAME_COUNT:
            pooled = F.adaptive_avg_pool2d(heat[index][None], (grid_height, grid_width))[0]
            features.append(pooled.permute(1, 2, 0).reshape(token_count, len(GAZE_CHANNELS)))
        else:
            features.append(torch.zeros(token_count, len(GAZE_CHANNELS), dtype=torch.float32))
    return torch.cat(features, dim=0)


class GazeFeatureAdapter:
    """Attach TGSM before loading the LoRA adapter that contains its learned parameters."""

    module_name = "gaze_feature_film"

    def __init__(self) -> None:
        self._pending: torch.Tensor | None = None

    def set_pending(self, features: torch.Tensor | None) -> None:
        self._pending = features

    def attach(self, model: Any) -> None:
        if hasattr(model, self.module_name):
            raise ValueError("TGSM has already been attached to this model")
        model.gaze_feature_film = GazeFeatureFiLM(model.config.get_text_config().hidden_size).to(
            device=model.device, dtype=torch.float32
        )
        inner = model.model
        original = inner.get_image_features

        def get_image_features(pixel_values, image_grid_thw=None):
            image_embeddings, deepstack_embeddings = original(pixel_values, image_grid_thw)
            weights = self._pending
            if weights is None:
                raise RuntimeError("TGSM requires gaze features for the current sample")
            total_tokens = sum(item.shape[0] for item in image_embeddings)
            if weights.shape != (total_tokens, len(GAZE_CHANNELS)):
                raise ValueError(
                    f"Expected {total_tokens} gaze token features, received {tuple(weights.shape)}"
                )
            modulated = []
            offset = 0
            for embeddings in image_embeddings:
                count = embeddings.shape[0]
                modulated.append(
                    model.gaze_feature_film(
                        embeddings, weights[offset : offset + count], path="final"
                    )
                )
                offset += count
            deepstack = [
                model.gaze_feature_film(embeddings, weights, path="deepstack")
                for embeddings in deepstack_embeddings
            ]
            self._pending = None
            return tuple(modulated), deepstack

        inner.get_image_features = get_image_features
