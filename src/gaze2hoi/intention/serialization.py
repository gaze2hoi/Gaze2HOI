"""Canonical target serializer and strict, sample-aware prediction parser."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from . import task

Intent = dict[str, dict[str, Any]]


def empty_hand() -> dict[str, Any]:
    return {slot: None for slot in task.HAND_SLOT_ORDER}


def intent_to_text(intent: Intent) -> str:
    ordered = {
        side: {slot: intent[side][slot] for slot in task.HAND_SLOT_ORDER}
        for side in task.HAND_ORDER
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))


@dataclass(frozen=True)
class SampleContext:
    object_names: frozenset[str]
    parts_by_object: dict[str, frozenset[str]]
    target_names: frozenset[str]
    actions: frozenset[str]
    deep_actions: frozenset[str]


@dataclass
class ParseResult:
    intent: Intent | None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.intent is not None and not self.errors


def _extract_json(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
        if value.startswith("json"):
            value = value[4:].strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_prediction(text: str, ctx: SampleContext) -> ParseResult:
    raw = _extract_json(text)
    if raw is None:
        return ParseResult(None, ["not valid JSON"])
    errors: list[str] = []
    unknown_top = set(raw) - set(task.HAND_ORDER)
    if unknown_top:
        errors.append(f"unknown top-level keys {sorted(unknown_top)}")
    intent: Intent = {}
    for side in task.HAND_ORDER:
        source = raw.get(side)
        if not isinstance(source, dict):
            errors.append(f"{side}: missing or not an object")
            intent[side] = empty_hand()
            continue
        unknown = set(source) - set(task.HAND_SLOT_ORDER)
        if unknown:
            errors.append(f"{side}: unknown keys {sorted(unknown)}")
        missing = set(task.HAND_SLOT_ORDER) - set(source)
        if missing:
            errors.append(f"{side}: missing keys {sorted(missing)}")
        out = empty_hand()
        action = source.get("action")
        if action is None:
            non_null = [s for s in task.HAND_SLOT_ORDER[1:] if source.get(s) is not None]
            if non_null:
                errors.append(f"{side}: action=null but non-null {non_null}")
            intent[side] = out
            continue
        if not isinstance(action, str) or action not in ctx.actions:
            errors.append(f"{side}: unknown action {action!r}")
        out["action"] = action
        obj = source.get("object")
        if not isinstance(obj, str) or obj not in ctx.object_names:
            errors.append(f"{side}: object {obj!r} not a candidate")
        out["object"] = obj
        part = source.get("part")
        allowed_parts = (
            ctx.parts_by_object.get(obj, frozenset()) if isinstance(obj, str) else frozenset()
        )
        if part is not None and (not isinstance(part, str) or part not in allowed_parts):
            errors.append(f"{side}: part {part!r} not a part of {obj!r}")
        out["part"] = part
        deep_action = source.get("deep_action")
        if deep_action is not None and (
            not isinstance(deep_action, str) or deep_action not in ctx.deep_actions
        ):
            errors.append(f"{side}: unknown deep_action {deep_action!r}")
        out["deep_action"] = deep_action
        target = source.get("target")
        if target is not None and (not isinstance(target, str) or target not in ctx.target_names):
            errors.append(f"{side}: target {target!r} not in scene targets")
        out["target"] = target
        point = source.get("point")
        if deep_action == task.PLACE_DEEP_ACTION:
            valid = (
                isinstance(point, (list, tuple))
                and len(point) == 2
                and all(type(v) in (int, float) and math.isfinite(v) for v in point)
                and all(0 <= v <= task.COORDINATE_MAX for v in point)
            )
            if valid:
                out["point"] = [int(round(v)) for v in point]
            else:
                errors.append(f"{side}: place requires norm1000 point")
        elif point is not None:
            errors.append(f"{side}: point given but deep_action != place")
        intent[side] = out
    return ParseResult(intent, errors)
