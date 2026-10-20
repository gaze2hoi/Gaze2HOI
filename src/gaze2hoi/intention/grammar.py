"""Per-sample JSON Schema for constrained decoding."""

from __future__ import annotations

from typing import Any

from . import task
from .serialization import SampleContext


def _null_hand() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {slot: {"type": "null"} for slot in task.HAND_SLOT_ORDER},
        "required": list(task.HAND_SLOT_ORDER),
        "additionalProperties": False,
    }


def _enum_or_null(values: list[str]) -> dict[str, Any]:
    return {"anyOf": [{"enum": sorted(values)}, {"type": "null"}]}


def _active_hand(context: SampleContext, object_name: str, is_place: bool) -> dict[str, Any]:
    parts = sorted(context.parts_by_object.get(object_name, frozenset()))
    if is_place:
        deep_action: dict[str, Any] = {"enum": [task.PLACE_DEEP_ACTION]}
        point: dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 0,
                "maximum": task.COORDINATE_MAX,
            },
            "minItems": 2,
            "maxItems": 2,
        }
    else:
        deep_action = _enum_or_null(sorted(context.deep_actions - {task.PLACE_DEEP_ACTION}))
        point = {"type": "null"}
    return {
        "type": "object",
        "properties": {
            "action": {"enum": sorted(context.actions)},
            "object": {"enum": [object_name]},
            "part": _enum_or_null(parts) if parts else {"type": "null"},
            "deep_action": deep_action,
            "target": _enum_or_null(sorted(context.target_names)),
            "point": point,
        },
        "required": list(task.HAND_SLOT_ORDER),
        "additionalProperties": False,
    }


def build_json_schema(context: SampleContext) -> dict[str, Any]:
    branches = [_null_hand()]
    for object_name in sorted(context.object_names):
        branches.append(_active_hand(context, object_name, False))
        branches.append(_active_hand(context, object_name, True))
    hand = {"anyOf": branches}
    return {
        "type": "object",
        "properties": {side: hand for side in task.HAND_ORDER},
        "required": list(task.HAND_ORDER),
        "additionalProperties": False,
    }
