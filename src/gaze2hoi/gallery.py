"""Build a self-contained gallery from actual inference outputs."""

from __future__ import annotations

import json
import logging
import shutil
from importlib.resources import files
from pathlib import Path

from .data import ExampleDataset
from .io import read_json, write_json

LOGGER = logging.getLogger("gaze2hoi")


def render_gallery(
    dataset: ExampleDataset,
    examples: list,
    results: Path,
    mano_dir: Path,
    output: Path,
    *,
    size: int = 512,
    draw_number: int | None = None,
) -> None:
    import numpy as np

    from .visualization import render_forecast

    if size < 128 or size % 2:
        raise ValueError("Render size must be an even integer of at least 128")
    if draw_number is not None and draw_number < 1:
        raise ValueError("Draw numbers must be positive")
    jobs = []
    for example in examples:
        forecast = results / example.sample_id / "forecast.npz"
        intention = read_json(results / example.sample_id / "intention.json")
        with np.load(forecast, allow_pickle=False) as arrays:
            indices = arrays["draw_indices"].astype(int).tolist()
        wanted = draw_number - 1 if draw_number else example.homepage["draw_index"]
        if wanted not in indices:
            raise ValueError(
                f"{example.sample_id}: draw {wanted + 1} is not in these results; use --draw"
            )
        jobs.append((example, forecast, intention, indices.index(wanted), wanted + 1))
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for position, (example, forecast, intention, draw_index, draw) in enumerate(jobs, 1):
        LOGGER.info("Rendering %d/%d: %s", position, len(jobs), example.sample_id)
        destination = output / example.sample_id
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(example.directory / "inputs/observation.mp4", destination / "observation.mp4")
        render_forecast(
            example.directory,
            forecast,
            mano_dir,
            destination / "prediction.mp4",
            size=size,
            draw=draw_index,
        )
        render_forecast(
            example.directory,
            example.directory / "ground_truth/scene_motion.npz",
            mano_dir,
            destination / "ground_truth.mp4",
            size=size,
        )
        archive = example.directory / "reference_outputs/website_prediction.mp4"
        if archive.is_file():
            shutil.copy2(archive, destination / "website_reference.mp4")
        metadata = read_json(results / example.sample_id / "metadata.json")
        write_json(destination / "metadata.json", metadata)
        truth = read_json(example.directory / "ground_truth/intention.json")
        entries.append(
            {
                "sample_id": example.sample_id,
                "group": example.group,
                "phase": example.metadata["phase"],
                "draw_number": draw,
                "intention": intention["intent"],
                "ground_truth_intention": truth.get("intent", truth),
                "observation": f"{example.sample_id}/observation.mp4",
                "prediction": f"{example.sample_id}/prediction.mp4",
                "ground_truth": f"{example.sample_id}/ground_truth.mp4",
                "metadata": f"{example.sample_id}/metadata.json",
                "website_reference": f"{example.sample_id}/website_reference.mp4"
                if archive.is_file()
                else None,
            }
        )
    payload = {"title": "Gaze2HOI", "samples": entries}
    write_json(output / "gallery.json", payload)
    assets = files("gaze2hoi").joinpath("web")
    encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    template = assets.joinpath("index.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(
        template.replace("__GALLERY_DATA__", encoded), encoding="utf-8"
    )
    for name in ("style.css", "player.js"):
        (output / name).write_text(
            assets.joinpath(name).read_text(encoding="utf-8"), encoding="utf-8"
        )
