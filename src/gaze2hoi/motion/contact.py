"""Online contact grounding from observations and predicted intentions."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from . import config as DC
from .acg_model import ACGNet
from .assets import AffordanceBank, load_template_bank
from .config import ACGConfig
from .geometry import (
    ContactExtractor,
    TemplateBank,
    _rot6d_to_matrix,
    _wrist_local_series,
    find_support_furniture,
)

POS_SCALE = 3.0
TRAJ_LEN = 10
_SIDES = ("left", "right")
_HAND_SLICES = (DC.LEFT_HAND_SLICE, DC.RIGHT_HAND_SLICE)
_OBJ_SLICES = (DC.LEFT_OBJ_SLICE, DC.RIGHT_OBJ_SLICE)
_N = DC.NUM_TEMPLATE_POINTS
_GN = DC.LANDING_GRID_N


@dataclass(frozen=True)
class OnlineSample:
    """One already-rebound shard and its parsed VLM intent."""

    sample_id: str
    shard: Mapping[str, np.ndarray]
    intent: Mapping[str, Mapping[str, Any]] | None


@dataclass
class GeometryCorpus:
    """CPU geometry tensors plus row-to-(sample, hand) scatter indices."""

    sample_ids: list[str]
    row_sample: np.ndarray
    row_side: np.ndarray
    landing_yaw: np.ndarray
    bottom_valid: np.ndarray
    landing_valid: np.ndarray
    tensors: dict[str, torch.Tensor]

    @property
    def n_rows(self) -> int:
        return len(self.row_sample)


def _normalise_intent(intent: Mapping[str, Any] | None) -> Mapping[str, Mapping[str, Any]]:
    """Accept either a parsed intent or a prediction-jsonl record."""
    if intent is None:
        return {}
    if "intent" in intent and ("parse_ok" in intent or not any(side in intent for side in _SIDES)):
        if not bool(intent.get("parse_ok", True)):
            return {}
        nested = intent.get("intent")
        return nested if isinstance(nested, Mapping) else {}
    return intent


def _summary(pdf: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, ...]:
    centroid = (pdf[..., None] * points).sum(dim=1)
    d2 = ((points - centroid[:, None]) ** 2).sum(dim=-1)
    dispersion = torch.sqrt((pdf * d2).sum(dim=1).clamp_min(0))
    entropy = -(pdf * torch.log(pdf.clamp_min(1e-12))).sum(dim=1)
    entropy = entropy / float(np.log(pdf.shape[1]))
    return (centroid, dispersion, entropy)


class OnlineACG:
    """Frozen contact model with observation-only geometric feature extraction."""

    def __init__(self, checkpoint_dir: Path, mano_dir: Path, device: torch.device):
        self.device = torch.device(device)
        assets = Path(checkpoint_dir) / "assets"
        bank = load_template_bank(assets, with_sdf=True)
        points = bank["points"] * bank["scale"][:, None, None] + bank["offset"][:, None, :]
        raw_config = json.loads((Path(checkpoint_dir) / "acg_config.json").read_text())
        keys = {field.name for field in dataclasses.fields(ACGConfig)}
        self.model_config = ACGConfig(
            **{key: value for key, value in raw_config.items() if key in keys}
        )
        bottom = torch.from_numpy(np.load(assets / "bottom_prior.npy", allow_pickle=False)).float()
        self.model = ACGNet(self.model_config, points, bottom)
        state = torch.load(Path(checkpoint_dir) / "acg.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).requires_grad_(False).eval()
        self.bank = AffordanceBank(assets / "contact_priors.npz").to(self.device)
        geometry = TemplateBank(
            bank["uids"], points, bank["sdf_grid"], bank["scale"], bank["offset"]
        )
        self.extractor = ContactExtractor(geometry, mano_dir, self.device)
        vocab = json.loads((assets / "intent_vocab.json").read_text())
        self.part_to_id = {str(value): index for index, value in enumerate(vocab["part"])}

    def _part_id(self, intent: Mapping[str, Mapping[str, Any]], side: str) -> int:
        hand = intent.get(side) or {}
        part = hand.get("part")
        if part is None:
            return 0
        value = str(part)
        if value not in self.part_to_id:
            raise ValueError(f"unknown VLM part for {side} hand: {value!r}")
        return self.part_to_id[value]

    @torch.no_grad()
    def extract_geometry(self, samples: Iterable[OnlineSample]) -> GeometryCorpus:
        """Phase 1: extract all deployment-legal geometric features."""
        sample_ids: list[str] = []
        row_sample: list[int] = []
        row_side: list[int] = []
        yaw_rows: list[float] = []
        bottom_ok_rows: list[bool] = []
        landing_ok_rows: list[bool] = []
        columns: dict[str, list[Any]] = {
            key: []
            for key in (
                "slot",
                "side",
                "prior_c",
                "prior_u",
                "level_c",
                "scalars_c",
                "obs",
                "obs_valid",
                "obs_frac",
                "obs_area",
                "traj",
                "wrist_last",
                "wrist_valid",
                "down_local",
                "heading_rel",
                "goal_dist",
            )
        }
        for sample_index, sample in enumerate(samples):
            sample_ids.append(str(sample.sample_id))
            shard = sample.shard
            intent = _normalise_intent(sample.intent)
            state_np = np.asarray(shard["state"], dtype=np.float32)
            state = torch.from_numpy(state_np).to(self.device)
            betas = torch.from_numpy(np.asarray(shard["mano_betas"], dtype=np.float32)).to(
                self.device
            )
            for side_index, side in enumerate(_SIDES):
                slot = int(np.asarray(shard[f"{side}_obj_slot"]).item())
                if slot < 0:
                    continue
                hand_sl, obj_sl = (_HAND_SLICES[side_index], _OBJ_SLICES[side_index])
                hand_valid = np.asarray(shard[f"{side}_hand_valid"]).astype(bool)
                obj_valid = np.asarray(shard[f"{side}_obj_valid"]).astype(bool)
                valid = hand_valid & obj_valid
                frames = np.flatnonzero(valid[: DC.PAST_LEN]).astype(np.int64)
                obs, n_touch, obs_area = self.extractor._contact_pdf(
                    state, betas, frames, side, hand_sl, obj_sl, slot
                )
                traj = _wrist_local_series(
                    state_np.astype(np.float64),
                    hand_valid,
                    obj_valid,
                    hand_sl,
                    obj_sl,
                    np.asarray(shard[f"{side}_j0"], dtype=np.float64),
                )
                traj_ok = traj[:, 6] > 0
                wrist = (
                    traj[np.flatnonzero(traj_ok)[-1], :3] / POS_SCALE
                    if traj_ok.any()
                    else np.zeros(3, np.float32)
                )
                down = np.zeros(3, np.float32)
                heading_rel = np.zeros(2, np.float32)
                goal_dist = 0.0
                yaw = 0.0
                support = -1
                predicted_hand = intent.get(side) or {}
                place_valid = predicted_hand.get("deep_action") == DC.PLACE_DEEP_ACTION and bool(
                    np.asarray(shard["place_target_valid"])[side_index]
                )
                anchor = DC.ANCHOR_OFFSET
                if obj_valid[anchor]:
                    block = state_np[anchor, obj_sl]
                    rotation = (
                        _rot6d_to_matrix(torch.from_numpy(block[DC.OBJ_ROT6D_SLICE][None]))[0]
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    down = (rotation.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)
                    if place_valid:
                        goal = np.asarray(shard["place_target"][side_index], dtype=np.float64)
                        support, yaw = find_support_furniture(
                            goal,
                            np.asarray(shard["furn_present"]),
                            np.asarray(shard["furn_center"]),
                            np.asarray(shard["furn_params"]),
                            np.asarray(shard["furn_shape_id"]),
                            np.asarray(shard["furn_yaw"]),
                        )
                        heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                        rel = heading - yaw
                        heading_rel[:] = (np.sin(rel), np.cos(rel))
                        goal_dist = float(np.linalg.norm(goal[:2] - block[DC.OBJ_TRANSL_SLICE][:2]))
                pid = self._part_id(intent, side)
                slots = torch.tensor([slot], device=self.device)
                parts = torch.tensor([pid], device=self.device)
                cond = self.bank.lookup(slots, parts, side_index)
                uncond = self.bank.lookup(slots, torch.zeros_like(parts), side_index)
                row_sample.append(sample_index)
                row_side.append(side_index)
                yaw_rows.append(yaw)
                bottom_ok_rows.append(place_valid)
                landing_ok_rows.append(place_valid and support >= 0)
                values = {
                    "slot": slot,
                    "side": side_index,
                    "prior_c": cond["heat"][0].cpu(),
                    "prior_u": uncond["heat"][0].cpu(),
                    "level_c": cond["level"][0].cpu(),
                    "scalars_c": cond["scalars"][0].cpu(),
                    "obs": torch.from_numpy(obs.astype(np.float32)),
                    "obs_valid": float(n_touch > 0),
                    "obs_frac": float(n_touch / max(len(frames), 1)),
                    "obs_area": float(obs_area),
                    "traj": torch.from_numpy(traj.reshape(TRAJ_LEN * 7)),
                    "wrist_last": torch.from_numpy(wrist.astype(np.float32)),
                    "wrist_valid": float(traj_ok.any()),
                    "down_local": torch.from_numpy(down),
                    "heading_rel": torch.from_numpy(heading_rel),
                    "goal_dist": goal_dist,
                }
                for key, value in values.items():
                    columns[key].append(value)
        tensors: dict[str, torch.Tensor] = {}
        int_keys = {"slot", "side", "level_c"}
        for key, values in columns.items():
            if values and torch.is_tensor(values[0]):
                tensors[key] = torch.stack(values)
            else:
                tensors[key] = torch.tensor(
                    values, dtype=torch.long if key in int_keys else torch.float32
                )
        return GeometryCorpus(
            sample_ids=sample_ids,
            row_sample=np.asarray(row_sample, dtype=np.int64),
            row_side=np.asarray(row_side, dtype=np.int64),
            landing_yaw=np.asarray(yaw_rows, dtype=np.float32),
            bottom_valid=np.asarray(bottom_ok_rows, dtype=bool),
            landing_valid=np.asarray(landing_ok_rows, dtype=bool),
            tensors=tensors,
        )

    @staticmethod
    def _empty_output(n_samples: int) -> dict[str, torch.Tensor]:
        z = {
            "acg_hand_valid": torch.zeros(n_samples, 2),
            "acg_contact_cond": torch.zeros(n_samples, 2, _N),
            "acg_contact_uncond": torch.zeros(n_samples, 2, _N),
            "acg_gates": torch.zeros(n_samples, 2, 2),
            "acg_scalars_cond": torch.zeros(n_samples, 2, 2),
            "acg_scalars_uncond": torch.zeros(n_samples, 2, 2),
            "acg_centroid_cond": torch.zeros(n_samples, 2, 3),
            "acg_centroid_uncond": torch.zeros(n_samples, 2, 3),
            "acg_bottom": torch.zeros(n_samples, 2, _N),
            "acg_bottom_valid": torch.zeros(n_samples, 2),
            "acg_landing": torch.zeros(n_samples, 2, _GN, _GN),
            "acg_landing_valid": torch.zeros(n_samples, 2),
            "acg_landing_yaw_sincos": torch.zeros(n_samples, 2, 2),
        }
        return z

    @torch.inference_mode()
    def predict_geometry(
        self, corpus: GeometryCorpus, batch_size: int = 2048
    ) -> dict[str, torch.Tensor]:
        """Phase 2: batch ACG forward and scatter into the DiT contract."""
        result = self._empty_output(len(corpus.sample_ids))
        if corpus.n_rows == 0:
            return result
        rows = torch.from_numpy(corpus.row_sample)
        sides = torch.from_numpy(corpus.row_side)
        result["acg_hand_valid"][rows, sides] = 1.0
        bottom_ok = torch.from_numpy(corpus.bottom_valid)
        landing_ok = torch.from_numpy(corpus.landing_valid)
        result["acg_bottom_valid"][rows[bottom_ok], sides[bottom_ok]] = 1.0
        result["acg_landing_valid"][rows[landing_ok], sides[landing_ok]] = 1.0
        yaw = torch.from_numpy(corpus.landing_yaw)
        yaw_sc = torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)
        yaw_sc *= landing_ok[:, None]
        result["acg_landing_yaw_sincos"][rows, sides] = yaw_sc
        for lo in range(0, corpus.n_rows, batch_size):
            hi = min(lo + batch_size, corpus.n_rows)
            batch = {
                key: value[lo:hi].to(self.device, non_blocking=True)
                for key, value in corpus.tensors.items()
            }
            cond = self.model(batch, prior=batch["prior_c"])
            uncond = self.model(batch, prior=batch["prior_u"])
            pc = F.softmax(cond["contact_logits"], dim=-1)
            pu = F.softmax(uncond["contact_logits"], dim=-1)
            bottom = F.softmax(cond["bottom_logits"], dim=-1)
            landing = F.softmax(cond["landing_logits"].flatten(1), dim=-1).view(-1, _GN, _GN)
            points = self.model.points[batch["slot"]]
            cc, dc, ec = _summary(pc, points)
            cu, du, eu = _summary(pu, points)
            rr, ss = (rows[lo:hi], sides[lo:hi])
            values = {
                "acg_contact_cond": pc,
                "acg_contact_uncond": pu,
                "acg_gates": cond["gates"],
                "acg_scalars_cond": torch.stack((dc, ec), dim=-1),
                "acg_scalars_uncond": torch.stack((du, eu), dim=-1),
                "acg_centroid_cond": cc,
                "acg_centroid_uncond": cu,
            }
            for key, value in values.items():
                result[key][rr, ss] = value.cpu()
            local_bottom_ok = bottom_ok[lo:hi]
            local_landing_ok = landing_ok[lo:hi]
            result["acg_bottom"][rr[local_bottom_ok], ss[local_bottom_ok]] = bottom[
                local_bottom_ok.to(self.device)
            ].cpu()
            result["acg_landing"][rr[local_landing_ok], ss[local_landing_ok]] = landing[
                local_landing_ok.to(self.device)
            ].cpu()
        return result

    def predict(
        self,
        shard: Mapping[str, np.ndarray],
        intent: Mapping[str, Any] | None,
        sample_id: str = "online",
    ) -> dict[str, torch.Tensor]:
        """Single-sample API; returned tensors have no leading batch axis."""
        geometry = self.extract_geometry([OnlineSample(sample_id, shard, intent)])
        return {key: value[0] for key, value in self.predict_geometry(geometry).items()}
