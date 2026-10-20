"""Bind predicted intentions to observed object tracks and placement planes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .assets import NormStats


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def parse_intention(record: dict) -> dict:
    """Validate the structured prediction before applying any conditioning."""
    if not isinstance(record, dict):
        raise ValueError("Expected a structured intention object")
    if "intent" in record and not any(side in record for side in ("left", "right")):
        if record.get("parse_ok", True) is not True:
            raise ValueError("Cannot run motion inference from an unparsed intention")
        record = record["intent"]
    if not isinstance(record, dict) or set(record) != {"left", "right"}:
        raise ValueError("An intention must contain exactly the left and right hand records")
    fields = (*C.INTENT_SLOTS, "point")
    result = {}
    for side in ("left", "right"):
        source = record[side]
        if source is None:
            source = {}
        if not isinstance(source, dict):
            raise ValueError(f"The {side} intention must be an object or null")
        unknown = set(source) - set(fields)
        if unknown:
            raise ValueError(f"Unknown {side} intention fields: {sorted(unknown)}")
        hand = {key: source.get(key) for key in fields}
        for key in C.INTENT_SLOTS:
            value = hand[key]
            if value is not None and (not isinstance(value, str) or not value or value == "<none>"):
                raise ValueError(f"The {side} intention {key} must be a nonempty string or null")
        if hand["action"] is None:
            if any(value is not None for value in hand.values()):
                raise ValueError(f"The {side} idle hand must have null intention fields")
        elif hand["object"] is None:
            raise ValueError(f"The {side} active hand requires an object")
        point = hand["point"]
        if hand["deep_action"] == C.PLACE_DEEP_ACTION:
            if not (
                isinstance(point, (list, tuple))
                and len(point) == 2
                and all(
                    type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1000
                    for value in point
                )
            ):
                raise ValueError(f"The {side} placement requires a finite norm1000 point")
        elif point is not None:
            raise ValueError(f"The {side} point requires a placement intention")
        result[side] = hand
    return result


def lift_placement(
    point, target: str | None, camera: dict, shard: dict
) -> tuple[np.ndarray | None, str]:
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all() or np.any((point < 0) | (point > 1000)):
        return None, "invalid_point"
    size = int(camera.get("display_size", 1408))
    display = point / 1000 * (size - 1)
    rectified = np.array([[display[1], (size - 1.0) - display[0]]], dtype=np.float64)
    k_inv = np.linalg.inv(np.asarray(camera["k_linear"], dtype=np.float64))
    camera_r = np.asarray(camera["head_rotation"], dtype=np.float64)
    direction = (np.hstack([rectified, np.ones((1, 1))]) @ k_inv.T) @ camera_r.T
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    align = np.asarray(camera["r_align"], dtype=np.float64)
    origin = align @ np.asarray(camera["head_translation"], dtype=np.float64)
    direction = align @ direction[0]
    planes = camera["planes"]

    def intersect(height):
        if abs(direction[2]) <= 1e-9:
            return None
        distance = (float(height) - origin[2]) / direction[2]
        if not np.isfinite(distance) or not 0 < distance <= 20.0:
            return None
        return origin + distance * direction

    hit, kind = None, "no_hit"
    if target in planes:
        hit = intersect(planes[target])
        if hit is not None:
            kind = "target_plane"
    if hit is None:
        hits = [intersect(height) for height in sorted(set(planes.values()))]
        hits = [value for value in hits if value is not None]
        if hits:
            hit = min(hits, key=lambda value: float(np.linalg.norm(value - origin)))
            kind = "fallback_first_hit"
    if hit is None:
        return None, kind
    anchor_r = np.asarray(shard["anchor_rotation"], dtype=np.float64)
    anchor_origin = np.asarray(shard["anchor_origin"], dtype=np.float64)
    anchor_point = (hit[None] - align @ anchor_origin) @ (anchor_r @ align.T).T
    return anchor_point[0].astype(np.float32), kind


def bind_observation(
    sample_dir: Path, intention: dict, vocab: dict, object_vocab: dict
) -> tuple[dict, dict, dict]:
    sample_dir = Path(sample_dir)
    metadata = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
    shard = read_arrays(sample_dir / "inputs" / "observation.npz")
    candidates = read_arrays(sample_dir / "inputs" / "objects.npz")
    if isinstance(intention, dict) and "sample_id" in intention:
        if intention["sample_id"] != metadata["sample_id"]:
            raise ValueError("The intention sample ID does not match the observation")
    intention = parse_intention(intention)
    count = len(candidates["slots"])
    slots = np.asarray(candidates["slots"])
    validity = np.asarray(candidates["valid"])
    if slots.shape != (count,) or slots.dtype.kind not in "iu" or len(set(slots.tolist())) != count:
        raise ValueError("Candidate object slots must be a unique integer vector")
    if np.any((slots < 0) | (slots >= C.NUM_OBJECT_SLOTS)):
        raise ValueError("Candidate object slots must lie in the released object vocabulary")
    if candidates["state"].shape != (count, C.PAST_LEN, C.OBJ_DIM):
        raise ValueError("Candidate object states must have shape (N, 30, 9)")
    if validity.shape != (count, C.PAST_LEN) or not np.isin(validity, (0, 1)).all():
        raise ValueError("Candidate object validity must contain binary flags with shape (N, 30)")
    if shard["state"].shape != (C.PAST_LEN, C.STATE_DIM):
        raise ValueError("Observation state must have shape (30, 66)")
    if not np.isfinite(shard["state"]).all():
        raise ValueError("Observation state contains non-finite values")
    slots_by_name = {str(row["name"]): int(row["slot"]) for row in object_vocab["slots"]}
    candidate_by_slot = {int(slot): index for index, slot in enumerate(candidates["slots"])}
    vocab_ids = {
        key: {value: index for index, value in enumerate(values)} for key, values in vocab.items()
    }
    camera_path = sample_dir / "inputs" / "placement.json"
    camera = json.loads(camera_path.read_text(encoding="utf-8")) if camera_path.exists() else None
    state = np.zeros((C.WINDOW_LEN, C.STATE_DIM), dtype=np.float32)
    state[: C.PAST_LEN, :48] = shard["state"][:, :48]
    shard["intent_ids"] = np.zeros((2, C.NUM_INTENT_SLOTS), dtype=np.int64)
    shard["place_target"] = np.zeros((2, 3), dtype=np.float32)
    shard["place_target_valid"] = np.zeros(2, dtype=np.float32)
    shard["target_slots"] = np.full(2, -1, dtype=np.int64)
    details = {}
    for index, side in enumerate(("left", "right")):
        hand = intention[side]
        for column, key in enumerate(C.INTENT_SLOTS):
            value = hand.get(key)
            if value is not None:
                if value not in vocab_ids[key]:
                    if key != "target" or value not in (camera or {}).get("planes", {}):
                        raise ValueError(f"Unknown {side} intention {key}: {value!r}")
                else:
                    shard["intent_ids"][index, column] = vocab_ids[key][value]
        active = hand.get("action") is not None
        name = hand.get("object") if active else metadata.get(f"contact_{side}")
        if name is not None and name not in slots_by_name:
            raise ValueError(f"Unknown {side} bound object: {name!r}")
        slot = slots_by_name.get(name, -1)
        row = candidate_by_slot.get(slot)
        if active and row is None:
            raise ValueError(f"The predicted {side} object is not an observed candidate: {name!r}")
        valid = np.zeros(C.WINDOW_LEN, dtype=np.uint8)
        reason = "no_observed_candidate"
        if row is not None and np.asarray(candidates["valid"][row], dtype=bool).mean() >= 0.5:
            block = np.asarray(candidates["state"][row], dtype=np.float32)
            if block.shape != (C.PAST_LEN, C.OBJ_DIM) or not np.isfinite(block).all():
                raise ValueError(f"Invalid observed object track: {name}")
            sl = C.LEFT_OBJ_SLICE if index == 0 else C.RIGHT_OBJ_SLICE
            state[: C.PAST_LEN, sl] = block
            valid[: C.PAST_LEN] = candidates["valid"][row]
            reason = "bound"
        else:
            slot = -1
        shard[f"{side}_obj_slot"] = np.asarray(slot, dtype=np.int64)
        shard[f"{side}_obj_valid"] = valid
        hand_valid = np.asarray(shard[f"{side}_hand_valid"])
        if hand_valid.shape != (C.PAST_LEN,) or not np.isin(hand_valid, (0, 1)).all():
            raise ValueError(f"Expected 30 binary observed validity flags for {side} hand")
        hand_valid = hand_valid.astype(np.uint8)
        shard[f"{side}_hand_valid"] = np.pad(hand_valid, (0, C.FUTURE_LEN))
        details[side] = {
            "object": name if slot >= 0 else None,
            "slot": slot,
            "source": "intent" if active else "contact",
            "reason": reason,
        }
        target = hand.get("target")
        if active and target in slots_by_name:
            shard["target_slots"][index] = slots_by_name[target]
        if hand.get("deep_action") == C.PLACE_DEEP_ACTION and hand.get("point") is not None:
            if camera is None:
                raise FileNotFoundError("A placement intention requires inputs/placement.json")
            point, status = lift_placement(hand["point"], target, camera, shard)
            details[side]["placement"] = status
            if point is not None:
                shard["place_target"][index] = point
                shard["place_target_valid"][index] = 1.0
    shard["state"] = state
    return shard, details, metadata


def model_inputs(shard: dict, stats: NormStats) -> dict[str, torch.Tensor]:
    """Normalize observed conditions and derive future masks from past presence."""
    state_mask = np.zeros((C.WINDOW_LEN, C.STATE_DIM), dtype=np.float32)
    for key, sl in (
        ("left_hand_valid", C.LEFT_HAND_SLICE),
        ("right_hand_valid", C.RIGHT_HAND_SLICE),
        ("left_obj_valid", C.LEFT_OBJ_SLICE),
        ("right_obj_valid", C.RIGHT_OBJ_SLICE),
    ):
        state_mask[:, sl] = np.asarray(shard[key], dtype=np.float32)[:, None]
    state = stats.normalize(shard["state"].astype(np.float32), "state") * state_mask
    state[C.PAST_LEN :] = 0.0
    state_mask[C.PAST_LEN :] = np.any(state_mask[: C.PAST_LEN] > 0, axis=0)
    result = {
        "state": state,
        "state_mask": state_mask,
        "observed": np.r_[np.ones(C.PAST_LEN), np.zeros(C.FUTURE_LEN)].astype(np.float32),
    }
    for key in (
        "intent_ids",
        "left_obj_slot",
        "right_obj_slot",
        "mano_betas",
        "scene_present",
        "scene_rot6d_last",
        "furn_present",
        "furn_type_id",
        "furn_shape_id",
        "place_target_valid",
    ):
        result[key] = np.asarray(shard[key])
    scene_mask = np.asarray(shard["scene_present"], dtype=np.float32)[:, None]
    for key in ("scene_transl_last", "scene_transl_mean"):
        result[key] = stats.normalize(shard[key].astype(np.float32), "scene_transl") * scene_mask
    furn_mask = np.asarray(shard["furn_present"], dtype=np.float32)[:, None]
    for key in ("furn_center", "furn_params"):
        result[key] = stats.normalize(shard[key].astype(np.float32), key) * furn_mask
    yaw = np.asarray(shard["furn_yaw"], dtype=np.float32)
    result["furn_yaw_sincos"] = np.stack((np.sin(yaw), np.cos(yaw)), axis=-1) * furn_mask
    result["place_target"] = (
        stats.normalize(shard["place_target"], "place_target")
        * result["place_target_valid"][:, None]
    )
    move = np.zeros((2, 3), dtype=np.float32)
    move_valid = np.zeros(2, dtype=np.float32)
    for index, slot in enumerate(shard["target_slots"]):
        if slot >= 0 and shard["scene_present"][slot] > 0:
            move[index] = shard["scene_transl_last"][slot]
            move_valid[index] = 1.0
    result["move_target"] = stats.normalize(move, "place_target") * move_valid[:, None]
    result["move_target_valid"] = move_valid
    integer_keys = {
        "intent_ids",
        "left_obj_slot",
        "right_obj_slot",
        "furn_type_id",
        "furn_shape_id",
    }
    return {
        key: torch.from_numpy(
            np.array(value, dtype=np.int64 if key in integer_keys else np.float32, copy=True)
        )
        for key, value in result.items()
    }
