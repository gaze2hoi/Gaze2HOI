"""Read the portable EgoAIM example manifest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import read_json


@dataclass(frozen=True)
class Example:
    sample_id: str
    directory: Path
    metadata: dict[str, Any]

    @property
    def homepage(self) -> dict[str, Any]:
        return self.metadata["homepage"]

    @property
    def group(self) -> str:
        return self.homepage["group"]

    def resolve(self, relative: str) -> Path:
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError(f"Sample path escapes its directory: {relative}")
        return path


class ExampleDataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = read_json(self.root / "manifest.json")
        self.examples: list[Example] = []
        seen = set()
        for record in self.manifest.get("samples", []):
            sample_id = record["sample_id"]
            if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
                raise ValueError(f"Invalid sample ID: {sample_id!r}")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            seen.add(sample_id)
            directory = (self.root / record["path"]).resolve()
            if not directory.is_relative_to(self.root):
                raise ValueError(f"Sample path escapes the dataset: {record['path']}")
            metadata = read_json(directory / "sample.json")
            if metadata.get("sample_id") != sample_id:
                raise ValueError(f"Mismatched sample ID in {directory / 'sample.json'}")
            self.examples.append(Example(sample_id, directory, metadata))
        if not self.examples:
            raise ValueError(f"No examples found in {self.root / 'manifest.json'}")

    def select(self, selectors: Iterable[str] | None = None) -> list[Example]:
        selectors = list(selectors or ["homepage"])
        default_groups = set(self.manifest.get("default_groups", ["qualitative", "failure"]))
        available = {item.sample_id for item in self.examples} | {
            item.group for item in self.examples
        }
        unknown = set(selectors) - available - {"all", "homepage"}
        if unknown:
            raise ValueError(
                f"Unknown example selector(s): {', '.join(sorted(unknown))}. Run gaze2hoi list."
            )
        return [
            item
            for item in self.examples
            if "all" in selectors
            or item.sample_id in selectors
            or item.group in selectors
            or ("homepage" in selectors and item.group in default_groups)
        ]


def check_examples(
    dataset: ExampleDataset, examples: list[Example], *, include_motion: bool = True
) -> list[str]:
    import numpy as np

    from .intention.inputs import IntentionInput

    errors = []
    for example in examples:
        try:
            inputs = IntentionInput.load(example.directory, dataset_root=dataset.root)
            inputs.load_gaze()
            for path in inputs.image_paths:
                if not path.is_file():
                    raise FileNotFoundError(f"Missing image: {path}")
            if not include_motion:
                continue
            for key in ("observation", "objects", "placement"):
                path = example.resolve(example.metadata["inputs"][key])
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {key}: {path}")
                if path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as arrays:
                        for name in arrays.files:
                            value = arrays[name]
                            if value.dtype.hasobject:
                                raise ValueError(f"Object array in {path.name}: {name}")
            with np.load(
                example.resolve(example.metadata["inputs"]["observation"]), allow_pickle=False
            ) as obs:
                if "state" not in obs:
                    raise ValueError("Observation is missing state")
                state = obs["state"]
                if state.shape != (30, 66) or not np.isfinite(state).all():
                    raise ValueError(
                        f"Expected finite observation state with shape (30, 66), received {state.shape}"
                    )
            with np.load(
                example.resolve(example.metadata["inputs"]["objects"]), allow_pickle=False
            ) as objects:
                state, valid, slots = objects["state"], objects["valid"], objects["slots"]
                if state.shape != (len(slots), 30, 9) or not np.isfinite(state).all():
                    raise ValueError(
                        "Expected finite candidate object states with shape (N, 30, 9)"
                    )
                if valid.shape != (len(slots), 30) or not np.isin(valid, [0, 1]).all():
                    raise ValueError("Expected binary candidate validity with shape (N, 30)")
                if slots.ndim != 1 or len(np.unique(slots)) != len(slots):
                    raise ValueError(
                        "Candidate object slots must be a unique one-dimensional array"
                    )
                if slots.dtype.kind not in "iu" or ((slots < 0) | (slots >= 33)).any():
                    raise ValueError("Candidate object slots must be integers from 0 through 32")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(f"{example.sample_id}: {exc}")
    return errors
