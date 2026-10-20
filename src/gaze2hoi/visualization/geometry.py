"""Convert prediction arrays into explicit hand states and full-scene object poses."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

FUTURE_FRAMES = 30


def rotations_from_6d(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    first, second = value[..., :3], value[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def poses_from_state(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 9:
        raise ValueError(f"Expected object states ending in 9 values, received {value.shape}")
    poses = np.broadcast_to(np.eye(4), value.shape[:-1] + (4, 4)).copy()
    poses[..., :3, :3] = rotations_from_6d(value[..., 3:9])
    poses[..., :3, 3] = value[..., :3]
    return poses


def _future(value: Any, name: str, dimensions: int, draw: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == dimensions + 1:
        if not 0 <= draw < len(array):
            raise ValueError(
                f"Draw position {draw} is outside {name}, which has {len(array)} draws"
            )
        array = array[draw]
    if array.ndim != dimensions or len(array) not in (30, 60):
        raise ValueError(
            f"{name} must contain 30 future or 60 total frames; received {array.shape}"
        )
    return array[-FUTURE_FRAMES:]


def prepare_prediction(
    observation: Mapping[str, Any], prediction: Mapping[str, Any], draw: int = 0
) -> dict[str, np.ndarray]:
    """Expand only predicted object slots; all other objects retain observed poses."""
    if "hand_state" in prediction:
        hands = _future(prediction["hand_state"], "hand_state", 3, draw)
    elif "state" in prediction:
        state = _future(prediction["state"], "state", 2, draw)
        hands = state[:, :48].reshape(FUTURE_FRAMES, 2, 24)
    else:
        raise ValueError("The prediction requires hand_state or state arrays")
    if hands.shape != (FUTURE_FRAMES, 2, 24):
        raise ValueError(f"Expected hand_state shape [30, 2, 24], received {hands.shape}")
    if "hand_valid" in prediction:
        hand_valid = _future(prediction["hand_valid"], "hand_valid", 2, draw).astype(bool)
    elif all(f"{side}_hand_valid" in prediction for side in ("left", "right")):
        hand_valid = np.stack(
            [
                _future(prediction[f"{side}_hand_valid"], f"{side}_hand_valid", 1, draw)
                for side in ("left", "right")
            ],
            axis=-1,
        ).astype(bool)
    else:
        observed = [
            np.asarray(observation[f"{side}_hand_valid"]).any() for side in ("left", "right")
        ]
        hand_valid = np.broadcast_to(observed, (FUTURE_FRAMES, 2)).copy()
    if hand_valid.shape != (FUTURE_FRAMES, 2):
        raise ValueError("hand_valid must have two columns, ordered left then right")
    hand_valid &= np.isfinite(hands).all(axis=-1)

    scene_state = np.concatenate(
        (observation["scene_transl_last"], observation["scene_rot6d_last"]), axis=-1
    )
    present = np.asarray(observation["scene_present"], dtype=bool).copy()
    present &= np.isfinite(scene_state).all(axis=-1)
    present &= np.linalg.norm(scene_state[:, 3:6], axis=-1) > 1e-7
    present &= np.linalg.norm(np.cross(scene_state[:, 3:6], scene_state[:, 6:9]), axis=-1) > 1e-7
    poses = np.repeat(poses_from_state(scene_state)[None], FUTURE_FRAMES, axis=0)
    object_valid = np.repeat(present[None], FUTURE_FRAMES, axis=0)

    name = next(
        (
            key
            for key in ("object_poses", "object_state", "bound_object_state")
            if key in prediction
        ),
        None,
    )
    if name is None and "state" in prediction:
        source = _future(prediction["state"], "state", 2, draw)[:, 48:].reshape(FUTURE_FRAMES, 2, 9)
        slots = prediction.get("bound_object_slots", prediction.get("object_slots"))
    elif name is not None:
        source = np.asarray(prediction[name])
        dimensions = 4 if source.shape[-2:] == (4, 4) else 3
        source = _future(source, name, dimensions, draw)
        if name == "bound_object_state":
            slots = prediction.get("bound_object_slots", prediction.get("object_slots"))
        else:
            slots = prediction.get("object_slots", np.arange(source.shape[1]))
    else:
        return {
            "hands": hands,
            "hand_valid": hand_valid,
            "object_poses": poses,
            "object_valid": object_valid,
        }
    if slots is None and all(f"{side}_obj_slot" in prediction for side in ("left", "right")):
        slots = [prediction[f"{side}_obj_slot"] for side in ("left", "right")]
    if slots is None:
        raise ValueError("Bound object predictions require object_slots")
    slots = np.asarray(slots, dtype=np.int64).reshape(-1)
    if len(slots) != source.shape[1]:
        raise ValueError("object_slots must match the number of predicted object columns")
    supplied = source if source.shape[-2:] == (4, 4) else poses_from_state(source)
    mask = np.isfinite(supplied).all(axis=(-2, -1))
    with np.errstate(invalid="ignore"):
        mask &= np.abs(np.linalg.det(supplied[..., :3, :3])) > 1e-7
    if "object_valid" in prediction:
        provided = _future(prediction["object_valid"], "object_valid", 2, draw).astype(bool)
        if provided.shape != mask.shape:
            raise ValueError("object_valid must match the supplied object trajectory columns")
        mask &= provided
    elif name in (None, "bound_object_state") and all(
        f"{side}_obj_valid" in prediction for side in ("left", "right")
    ):
        provided = np.stack(
            [
                _future(prediction[f"{side}_obj_valid"], f"{side}_obj_valid", 1, draw)
                for side in ("left", "right")
            ],
            axis=-1,
        ).astype(bool)
        if provided.shape != mask.shape:
            raise ValueError("Per-hand object validity must match the bound object columns")
        mask &= provided
    for slot in dict.fromkeys(slots.tolist()):
        if slot == -1:
            continue
        if not 0 <= slot < len(present):
            raise ValueError(f"Object slot {slot} is outside the scene vocabulary")
        columns = np.flatnonzero(slots == slot)
        if name in (None, "bound_object_state"):
            values = supplied[:, columns]
            mean_rotation = values[..., :3, :3].mean(axis=1)
            u, _, vh = np.linalg.svd(mean_rotation)
            correction = np.broadcast_to(np.eye(3), mean_rotation.shape).copy()
            correction[..., 2, 2] = np.linalg.det(u @ vh)
            poses[:, slot, :3, :3] = u @ correction @ vh
            poses[:, slot, :3, 3] = values[..., :3, 3].mean(axis=1)
        elif len(columns) != 1:
            raise ValueError("Full-scene object slots must be unique")
        else:
            poses[:, slot] = supplied[:, columns[0]]
        object_valid[:, slot] = mask[:, columns].all(axis=1) & present[slot]
    return {
        "hands": hands,
        "hand_valid": hand_valid,
        "object_poses": poses,
        "object_valid": object_valid,
    }
