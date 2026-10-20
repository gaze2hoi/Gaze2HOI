"""Portable JSON, array, and file-integrity utilities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root: Path, records: list[dict[str, Any]]) -> list[str]:
    errors = []
    root = root.resolve()
    for record in records:
        relative = record.get("path", "")
        path = (root / relative).resolve()
        if not relative or not path.is_relative_to(root):
            errors.append(f"Invalid manifest path: {relative!r}")
        elif not path.is_file():
            errors.append(f"Missing file: {relative}")
        elif path.stat().st_size != record["bytes"]:
            errors.append(f"Size mismatch: {relative}")
        elif file_hash(path) != record["sha256"]:
            errors.append(f"Checksum mismatch: {relative}")
    return errors


def verify_checksum_file(root: Path, path: Path) -> list[str]:
    errors = []
    root = root.resolve()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"Malformed checksum line in {path.name}")
            continue
        digest, relative = parts
        relative = relative.lstrip("*")
        file = (root / relative).resolve()
        if not file.is_relative_to(root):
            errors.append(f"Invalid checksum path: {relative}")
        elif not file.is_file():
            errors.append(f"Missing file: {relative}")
        elif file_hash(file) != digest:
            errors.append(f"Checksum mismatch: {relative}")
    return errors


def write_arrays(path: str | Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.hasobject for value in converted.values()):
        raise ValueError("Object arrays are not supported in forecast files")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **converted)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
