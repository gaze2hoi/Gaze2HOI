"""Portable normalization, geometry, and contact-prior assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config as C

N_LEVELS = 6
LEVEL_INVALID = 5
SIDE_MERGED = 2
SCALAR_FIELDS = ("dispersion_m", "area_frac", "entropy_nats", "consistency")


class NormStats:
    """Frozen normalization statistics used by the released motion model."""

    def __init__(self, path: Path) -> None:
        fields = ("state", "scene_transl", "furn_center", "furn_params", "place_target")
        with np.load(path, allow_pickle=False) as data:
            for field in fields:
                for suffix in ("mean", "std"):
                    key = f"{field}_{suffix}"
                    value = np.asarray(data[key], dtype=np.float32)
                    if not np.isfinite(value).all():
                        raise ValueError(f"Non-finite normalization asset: {key}")
                    if suffix == "std" and np.any(value <= 0):
                        raise ValueError(f"Non-positive normalization scale: {key}")
                    setattr(self, key, value)
        if self.state_mean.shape != (C.STATE_DIM,):
            raise ValueError("Expected 66 motion-state normalization channels")

    def normalize(self, values: np.ndarray, field: str) -> np.ndarray:
        return (values - getattr(self, f"{field}_mean")) / getattr(self, f"{field}_std")

    def denormalize(self, values: np.ndarray, field: str) -> np.ndarray:
        return values * getattr(self, f"{field}_std") + getattr(self, f"{field}_mean")

    def state_mean_t(self, device=None) -> torch.Tensor:
        return torch.from_numpy(self.state_mean).to(device)

    def state_std_t(self, device=None) -> torch.Tensor:
        return torch.from_numpy(self.state_std).to(device)


def load_template_bank(asset_dir: Path, with_sdf: bool = False) -> dict:
    asset_dir = Path(asset_dir)
    vocab = json.loads((asset_dir / "object_vocab.json").read_text(encoding="utf-8"))
    uids = [str(row["uid"]) for row in vocab["slots"]]
    with np.load(asset_dir / "object_templates.npz", allow_pickle=False) as data:
        raw_uids = list(map(str, data["obj_uids"]))
        order = [raw_uids.index(uid) for uid in uids]
        bank = {"uids": uids, "names": [row["name"] for row in vocab["slots"]]}
        for output, source in (
            ("points", "points_canonical"),
            ("scale", "scale"),
            ("offset", "offset"),
        ):
            bank[output] = torch.from_numpy(np.asarray(data[source], dtype=np.float32)[order])
        if with_sdf:
            bank["sdf_grid"] = torch.from_numpy(
                np.asarray(data["sdf_grid"], dtype=np.float32)[order]
            )
    if bank["points"].shape != (C.NUM_OBJECT_SLOTS, C.NUM_TEMPLATE_POINTS, 3):
        raise ValueError(f"Unexpected template geometry shape: {bank['points'].shape}")
    return bank


class AffordanceBank(nn.Module):
    """Frozen per-object contact priors with a reliability-aware fallback."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        bool_names = {"valid", "reliable", "any_valid", "any_reliable"}
        names = (
            "heat",
            "valid",
            "reliable",
            "centroid",
            "scalars",
            "any_heat",
            "any_valid",
            "any_reliable",
            "any_centroid",
            "any_scalars",
            "points_local",
        )
        with np.load(path, allow_pickle=False) as data:
            for name in names:
                dtype = torch.bool if name in bool_names else torch.float32
                self.register_buffer(name, torch.from_numpy(data[name]).to(dtype), persistent=False)
        self.n_parts = self.heat.shape[1]

    @torch.no_grad()
    def lookup(
        self, obj_slots: torch.Tensor, part_ids: torch.Tensor, side: int
    ) -> dict[str, torch.Tensor]:
        """Resolve the fallback chain for one hand across the batch."""
        b = obj_slots.shape[0]
        device = obj_slots.device
        slot = obj_slots.clamp_min(0)
        pid = part_ids.clamp(0, self.n_parts - 1)
        bound = obj_slots >= 0
        cand = (
            (
                self.valid[slot, pid, side] & self.reliable[slot, pid, side],
                self.heat[slot, pid, side],
                self.centroid[slot, pid, side],
                self.scalars[slot, pid, side],
            ),
            (
                self.valid[slot, pid, SIDE_MERGED] & self.reliable[slot, pid, SIDE_MERGED],
                self.heat[slot, pid, SIDE_MERGED],
                self.centroid[slot, pid, SIDE_MERGED],
                self.scalars[slot, pid, SIDE_MERGED],
            ),
            (
                self.any_valid[slot, side] & self.any_reliable[slot, side],
                self.any_heat[slot, side],
                self.any_centroid[slot, side],
                self.any_scalars[slot, side],
            ),
            (
                self.any_valid[slot, SIDE_MERGED],
                self.any_heat[slot, SIDE_MERGED],
                self.any_centroid[slot, SIDE_MERGED],
                self.any_scalars[slot, SIDE_MERGED],
            ),
            (
                self.valid[slot, pid, side],
                self.heat[slot, pid, side],
                self.centroid[slot, pid, side],
                self.scalars[slot, pid, side],
            ),
        )
        heat = torch.zeros(b, self.heat.shape[-1], device=device)
        centroid = torch.zeros(b, 3, device=device)
        scalars = torch.zeros(b, len(SCALAR_FIELDS), device=device)
        level = torch.full((b,), LEVEL_INVALID, dtype=torch.long, device=device)
        taken = torch.zeros(b, dtype=torch.bool, device=device)
        for lv, (ok, h, c, s) in enumerate(cand):
            pick = bound & ok & ~taken
            if bool(pick.any()):
                heat[pick] = h[pick]
                centroid[pick] = c[pick]
                scalars[pick] = s[pick]
                level[pick] = lv
                taken |= pick
        return {
            "heat": heat,
            "centroid": centroid,
            "scalars": scalars,
            "level": level,
            "valid": taken,
        }
