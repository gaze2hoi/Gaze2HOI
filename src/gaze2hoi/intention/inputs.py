"""Validate portable sample files and preserve the published prompt layout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serialization import SampleContext
from .task import FRAME_COUNT, GAZE_CHANNELS, GAZE_GRID_SIZE


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def resolve_asset(sample_dir: Path, value: str, dataset_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Asset paths must be nonempty relative paths")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"Expected a relative asset path, received {value!r}")
    path = (sample_dir / relative).resolve()
    if not path.is_relative_to(dataset_root):
        raise ValueError(f"Asset path leaves the dataset directory: {value!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing sample asset: {path}")
    return path


def _string_set(value: Any, name: str) -> frozenset[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return frozenset(value)


def sample_context(metadata: dict[str, Any], schema: dict[str, Any]) -> SampleContext:
    """Build the vocabulary used by both the grammar and output validator."""
    try:
        parts = metadata["parts_by_object"]
        if not isinstance(parts, dict):
            raise ValueError("context.parts_by_object must be an object")
        actions = [entry["value"] for entry in schema["actions"]]
        deep_actions = [entry["value"] for entry in schema["deep_actions"]]
        context = SampleContext(
            object_names=_string_set(metadata["legal_objects"], "context.legal_objects"),
            parts_by_object={
                name: _string_set(values, f"context.parts_by_object.{name}")
                for name, values in parts.items()
            },
            target_names=_string_set(metadata["target_names"], "context.target_names"),
            actions=_string_set(actions, "schema.actions"),
            deep_actions=_string_set(deep_actions, "schema.deep_actions"),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete intention vocabulary: {exc}") from exc
    if not context.actions:
        raise ValueError("The intention action vocabulary must not be empty")
    return context


@dataclass(frozen=True)
class IntentionInput:
    sample_id: str
    system_prompt: str
    user_prompt: str
    image_paths: tuple[Path, ...]
    gaze_path: Path
    context: SampleContext

    @classmethod
    def load(
        cls,
        sample_dir: str | Path,
        *,
        schema: dict[str, Any] | Path | str | None = None,
        dataset_root: str | Path | None = None,
    ) -> IntentionInput:
        sample_dir = Path(sample_dir).resolve()
        root = Path(dataset_root).resolve() if dataset_root else sample_dir.parent.parent
        if not sample_dir.is_relative_to(root):
            raise ValueError("The sample directory must be inside dataset_root")
        payload = read_json(resolve_asset(sample_dir, "inputs/intention.json", root))
        sample_id = payload.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("intention.json requires a nonempty sample_id")
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"{sample_id}: provide exactly one system and one user message")
        for message, role in zip(messages, ("system", "user")):
            if (
                not isinstance(message, dict)
                or message.get("role") != role
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(f"{sample_id}: expected a {role} message with string content")
        images = payload.get("images")
        if not isinstance(images, list) or len(images) < FRAME_COUNT:
            raise ValueError(
                f"{sample_id}: expected five observation images followed by object references"
            )
        if messages[1]["content"].count("<image>") != len(images):
            raise ValueError(f"{sample_id}: image placeholder count does not match image paths")
        image_paths = tuple(resolve_asset(sample_dir, name, root) for name in images)
        gaze_path = resolve_asset(sample_dir, payload.get("gaze", "inputs/gaze.npz"), root)
        if schema is None:
            schema = read_json(resolve_asset(sample_dir, payload.get("schema"), root))
        elif not isinstance(schema, dict):
            schema = read_json(Path(schema))
        metadata = payload.get("context")
        if not isinstance(metadata, dict):
            raise ValueError(f"{sample_id}: context must be an object")
        return cls(
            sample_id=sample_id,
            system_prompt=messages[0]["content"],
            user_prompt=messages[1]["content"],
            image_paths=image_paths,
            gaze_path=gaze_path,
            context=sample_context(metadata, schema),
        )

    def load_gaze(self):
        import numpy as np

        with np.load(self.gaze_path, allow_pickle=False) as data:
            if "heatmaps" not in data:
                raise ValueError(f"{self.gaze_path}: expected an array named 'heatmaps'")
            heatmaps = np.asarray(data["heatmaps"], dtype=np.float32)
        expected = (FRAME_COUNT, len(GAZE_CHANNELS), GAZE_GRID_SIZE, GAZE_GRID_SIZE)
        if heatmaps.shape != expected:
            raise ValueError(
                f"{self.sample_id}: expected gaze shape {expected}, received {heatmaps.shape}"
            )
        if not np.isfinite(heatmaps).all():
            raise ValueError(f"{self.sample_id}: gaze heatmaps contain non-finite values")
        return heatmaps

    def open_chat(self) -> tuple[list[dict[str, Any]], list[Any]]:
        """Open RGB images while preserving every text chunk between image tags."""
        from PIL import Image

        content: list[dict[str, Any]] = []
        images: list[Any] = []
        try:
            for index, text in enumerate(self.user_prompt.split("<image>")):
                if text:
                    content.append({"type": "text", "text": text})
                if index < len(self.image_paths):
                    with Image.open(self.image_paths[index]) as source:
                        images.append(source.convert("RGB"))
                    content.append({"type": "image"})
        except Exception:
            for image in images:
                image.close()
            raise
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ], images
