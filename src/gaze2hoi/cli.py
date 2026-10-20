"""Command-line entry points for the Gaze2HOI inference release."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

from . import __version__
from .data import ExampleDataset, check_examples
from .io import read_json, verify_checksum_file, verify_files, write_arrays, write_json

LOGGER = logging.getLogger("gaze2hoi")


def _dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data", type=Path, default=Path("../egoaim-demo"), help="Example dataset directory"
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["homepage"],
        metavar="ID_OR_GROUP",
        help="Sample IDs, qualitative, failure, overview, homepage, or all",
    )


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoints", type=Path, default=Path("../gaze2hoi-checkpoints"))


def _intention_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--intention-device", default="auto", help="auto, cpu, or a CUDA device")
    parser.add_argument(
        "--max-memory",
        action="append",
        default=[],
        metavar="DEVICE=LIMIT",
        help="Accelerate placement limit, e.g. --max-memory 0=10GiB (repeatable)",
    )
    parser.add_argument(
        "--local-files-only", action="store_true", help="Load the base model without network access"
    )


def _output_arguments(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--output", type=Path, default=Path(default))
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace results for the requested samples"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gaze-aware intention and hand-object motion inference"
    )
    parser.add_argument("--version", action="version", version=f"gaze2hoi {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Show exception tracebacks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="List packaged examples and homepage draw numbers")
    _dataset_arguments(listing)
    listing.set_defaults(samples=["all"])

    check = subparsers.add_parser("check", help="Validate packaged inputs and model assets")
    _dataset_arguments(check)
    _model_arguments(check)
    check.add_argument("--mano-dir", type=Path)
    check.add_argument("--hashes", action="store_true", help="Verify available file checksums")

    intention = subparsers.add_parser("intention", help="Predict structured two-hand intentions")
    _dataset_arguments(intention)
    _model_arguments(intention)
    _intention_arguments(intention)
    _output_arguments(intention, "outputs/intentions")

    forecast = subparsers.add_parser(
        "forecast", help="Run intention, contact, and motion inference"
    )
    _dataset_arguments(forecast)
    _model_arguments(forecast)
    _intention_arguments(forecast)
    _output_arguments(forecast, "outputs/homepage")
    forecast.add_argument("--mano-dir", type=Path, required=True)
    forecast.add_argument("--motion-device", default="cuda:0")
    forecast.add_argument("--threads", type=int, default=4)
    forecast.add_argument("--seed", type=int, default=999)
    draw = forecast.add_mutually_exclusive_group()
    draw.add_argument(
        "--draw", type=int, help="One-based draw number; defaults to each homepage example"
    )
    draw.add_argument("--num-draws", type=int, help="Generate draw numbers 1 through N")
    forecast.add_argument(
        "--intention-file",
        type=Path,
        help="Explicitly reuse predictions.jsonl from an earlier intention run",
    )

    render = subparsers.add_parser("render", help="Render actual generated forecasts")
    _dataset_arguments(render)
    render.add_argument("--results", type=Path, default=Path("outputs/homepage"))
    render.add_argument("--mano-dir", type=Path, required=True)
    render.add_argument("--output", type=Path, default=Path("outputs/gallery"))
    render.add_argument("--size", type=int, default=512)
    render.add_argument(
        "--draw", type=int, help="One-based draw number; defaults to the homepage draw"
    )

    download = subparsers.add_parser(
        "download", help="Download published assets from explicit Hub repositories"
    )
    download.add_argument("--model-repo", help="Published Hugging Face model repository ID")
    download.add_argument("--data-repo", help="Published Hugging Face dataset repository ID")
    download.add_argument("--model-revision", default="main")
    download.add_argument("--data-revision", default="main")
    _model_arguments(download)
    download.add_argument("--data", type=Path, default=Path("../egoaim-demo"))
    return parser


def _max_memory(entries: list[str]) -> dict[int | str, str] | None:
    memory: dict[int | str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected DEVICE=LIMIT, received {entry!r}")
        device, limit = entry.split("=", 1)
        if not limit or (not device.isdigit() and device != "cpu"):
            raise ValueError(f"Invalid memory limit: {entry!r}")
        memory[int(device) if device.isdigit() else device] = limit
    return memory or None


def _versions() -> dict[str, str]:
    versions = {"gaze2hoi": __version__, "python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "xgrammar", "numpy", "smplx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _ensure_output(args: argparse.Namespace, examples: list) -> None:
    if args.output.exists() and not args.overwrite:
        if any(args.output.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {args.output}; use another --output or pass --overwrite"
            )
    args.output.mkdir(parents=True, exist_ok=True)


def _read_intentions(path: Path, sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("sample_id"), str):
            raise ValueError("Each intention line must be an object with a string sample_id")
        if ("parse_ok" in record and record["parse_ok"] is not True) or not isinstance(
            record.get("intent"), dict
        ):
            raise ValueError(f"Invalid intention record for {record['sample_id']}")
        sample_id = record["sample_id"]
        if sample_id in records:
            raise ValueError(f"Duplicate intention record: {sample_id}")
        records[sample_id] = record
    missing = sample_ids - records.keys()
    if missing:
        raise ValueError(f"Missing intention records for: {', '.join(sorted(missing))}")
    return {key: value for key, value in records.items() if key in sample_ids}


def _predict_intentions(args: argparse.Namespace, dataset: ExampleDataset, examples: list) -> dict:
    from .intention import IntentionPredictor

    records = {}
    with IntentionPredictor(
        args.base_model,
        args.checkpoints / "intention",
        device=args.intention_device,
        max_memory=_max_memory(args.max_memory),
        local_files_only=args.local_files_only,
        dataset_root=dataset.root,
    ) as predictor:
        for index, example in enumerate(examples, 1):
            LOGGER.info("Intention %d/%d: %s", index, len(examples), example.sample_id)
            record = predictor.predict(example.directory)
            write_json(args.output / example.sample_id / "intention.json", record)
            records[example.sample_id] = record
            if not record["parse_ok"]:
                raise ValueError(
                    f"Invalid intention for {example.sample_id}: {record['parse_errors']}"
                )
    return records


def _save_intentions(output: Path, records: dict) -> None:
    path = output / "predictions.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n"
            for record in records.values()
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_check(args: argparse.Namespace, dataset: ExampleDataset, examples: list) -> int:
    errors = check_examples(dataset, examples)
    required = [
        "motion.pt",
        "motion_config.json",
        "acg.pt",
        "acg_config.json",
        "intention/adapter_model.safetensors",
        "intention/adapter_config.json",
        "assets/norm_stats.npz",
        "assets/object_templates.npz",
        "assets/contact_priors.npz",
        "assets/bottom_prior.npy",
        "assets/object_vocab.json",
        "assets/intent_vocab.json",
        "assets/furniture_vocab.json",
    ]
    errors.extend(
        f"Missing model asset: {name}"
        for name in required
        if not (args.checkpoints / name).is_file()
    )
    if args.mano_dir:
        errors.extend(
            f"Missing MANO model: MANO_{side}.pkl"
            for side in ("LEFT", "RIGHT")
            if not (args.mano_dir / f"MANO_{side}.pkl").is_file()
        )
    if args.hashes:
        for root in (args.checkpoints, dataset.root):
            manifest = read_json(root / "manifest.json")
            if manifest.get("files"):
                errors.extend(verify_files(root, manifest["files"]))
            elif (root / "checksums.json").exists():
                checksums = read_json(root / "checksums.json")
                errors.extend(verify_files(root, checksums["files"]))
            elif (root / "checksums.sha256").exists():
                errors.extend(verify_checksum_file(root, root / "checksums.sha256"))
            else:
                errors.append(f"No file checksums are available in {root}")
    for error in errors:
        LOGGER.error("%s", error)
    if not errors:
        print(f"Validated {len(examples)} examples and the inference model assets.")
    return int(bool(errors))


def _run_inference(args: argparse.Namespace, dataset: ExampleDataset, examples: list) -> int:
    if args.command == "forecast":
        if args.threads < 1 or args.seed < 0:
            raise ValueError("threads must be positive and seed must be non-negative")
        if (args.draw is not None and args.draw < 1) or (
            args.num_draws is not None and args.num_draws < 1
        ):
            raise ValueError("Draw numbers and counts must be positive")
        for side in ("LEFT", "RIGHT"):
            if not (args.mano_dir / f"MANO_{side}.pkl").is_file():
                raise FileNotFoundError(
                    f"Obtain MANO_{side}.pkl from the MANO website and place it in --mano-dir"
                )
    errors = check_examples(dataset, examples, include_motion=args.command == "forecast")
    if errors:
        raise ValueError("Invalid example inputs:\n" + "\n".join(errors))
    _ensure_output(args, examples)
    started = time.monotonic()
    reuse = getattr(args, "intention_file", None)
    if reuse:
        records = _read_intentions(reuse, {item.sample_id for item in examples})
        from .intention.inputs import IntentionInput
        from .intention.serialization import parse_prediction

        for example in examples:
            context = IntentionInput.load(example.directory, dataset_root=dataset.root).context
            record = records[example.sample_id]
            parsed = parse_prediction(json.dumps(record["intent"]), context)
            if not parsed.ok:
                raise ValueError(f"Invalid intention for {example.sample_id}: {parsed.errors}")
        for sample_id, record in records.items():
            write_json(args.output / sample_id / "intention.json", record)
    else:
        records = _predict_intentions(args, dataset, examples)
    _save_intentions(args.output, records)
    if args.command == "forecast":
        import torch

        from .motion import MotionPredictor

        torch.set_num_threads(args.threads)
        predictor = MotionPredictor(args.checkpoints, args.mano_dir, device=args.motion_device)
        for index, example in enumerate(examples, 1):
            draws = (
                list(range(args.num_draws))
                if args.num_draws
                else [args.draw - 1 if args.draw else example.homepage["draw_index"]]
            )
            LOGGER.info(
                "Forecast %d/%d: %s, draws %s",
                index,
                len(examples),
                example.sample_id,
                ", ".join(str(draw + 1) for draw in draws),
            )
            result = predictor.predict(
                example.directory, records[example.sample_id], draws, seed=args.seed
            )
            metadata = result.pop("metadata")
            metadata["versions"] = _versions()
            metadata["intention_source"] = (
                "provided_predictions" if reuse else "generated_in_this_run"
            )
            write_arrays(args.output / example.sample_id / "forecast.npz", result)
            write_json(args.output / example.sample_id / "metadata.json", metadata)
    write_json(
        args.output / "run.json",
        {
            "command": args.command,
            "samples": [item.sample_id for item in examples],
            "versions": _versions(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "intention_source": "provided_predictions" if reuse else "generated_in_this_run",
        },
    )
    print(f"Saved results to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "download":
            if not args.model_repo and not args.data_repo:
                raise ValueError("Provide at least one published --model-repo or --data-repo")
            from huggingface_hub import snapshot_download

            for repo, kind, revision, target in (
                (args.model_repo, "model", args.model_revision, args.checkpoints),
                (args.data_repo, "dataset", args.data_revision, args.data),
            ):
                if repo:
                    snapshot_download(repo, repo_type=kind, revision=revision, local_dir=target)
            return 0
        dataset = ExampleDataset(args.data)
        examples = dataset.select(args.samples)
        if args.command == "list":
            print(f"{'Sample':<36} {'Group':<12} {'Phase':<20} Draw")
            for example in examples:
                print(
                    f"{example.sample_id:<36} {example.group:<12} {example.metadata['phase']:<20} "
                    f"{example.homepage['draw_number']}"
                )
            return 0
        if args.command == "check":
            return _run_check(args, dataset, examples)
        if args.command in {"intention", "forecast"}:
            return _run_inference(args, dataset, examples)
        if args.command == "render":
            from .gallery import render_gallery

            render_gallery(
                dataset,
                examples,
                args.results,
                args.mano_dir,
                args.output,
                size=args.size,
                draw_number=args.draw,
            )
            print(f"Gallery: {args.output / 'index.html'}")
            return 0
        return 1
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
