"""Convert locally obtained MANO files to NumPy-only pickles for smplx."""

from __future__ import annotations

import argparse
import inspect
import pickle
from pathlib import Path

import numpy as np


def _convert(value):
    if isinstance(value, dict):
        return {key: _convert(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_convert(item) for item in value)
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if type(value).__module__.startswith("chumpy"):
        return np.asarray(value.r)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing officially obtained MANO_LEFT.pkl and MANO_RIGHT.pkl",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Use a separate output directory to preserve the original files")
    for side in ("LEFT", "RIGHT"):
        if not (args.source / f"MANO_{side}.pkl").is_file():
            parser.error(f"Missing MANO_{side}.pkl in {args.source}")
        if (args.output / f"MANO_{side}.pkl").exists():
            parser.error(f"Output already exists: MANO_{side}.pkl")
    # Chumpy's unpickler uses aliases removed by modern NumPy and Python.
    for name, dtype in {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
        "unicode": np.str_,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, dtype)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    try:
        import chumpy  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Install the conversion dependency: pip install --no-build-isolation chumpy==0.70"
        ) from exc
    converted = {}
    for side in ("LEFT", "RIGHT"):
        with (args.source / f"MANO_{side}.pkl").open("rb") as handle:
            converted[side] = _convert(pickle.load(handle, encoding="latin1"))
    args.output.mkdir(parents=True, exist_ok=True)
    for side, value in converted.items():
        with (args.output / f"MANO_{side}.pkl").open("wb") as handle:
            pickle.dump(value, handle, protocol=4)
    print(f"Prepared two local MANO files in {args.output}. Their original license still applies.")


if __name__ == "__main__":
    main()
