"""Portable release safeguards for manifests, files, and command-line inputs."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gaze2hoi.cli import _ensure_output, _read_intentions, build_parser
from gaze2hoi.data import ExampleDataset, check_examples
from gaze2hoi.io import (
    file_hash,
    read_json,
    verify_checksum_file,
    verify_files,
    write_arrays,
    write_json,
)


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    root = tmp_path / "examples"
    records = []
    for sample_id, group in (
        ("sample_a", "qualitative"),
        ("sample_b", "failure"),
        ("sample_c", "overview"),
    ):
        directory = root / "samples" / sample_id
        write_json(
            directory / "sample.json",
            {
                "sample_id": sample_id,
                "phase": "move",
                "homepage": {"group": group, "draw_index": 0, "draw_number": 1},
                "inputs": {
                    "observation": "inputs/observation.npz",
                    "objects": "inputs/objects.npz",
                    "placement": "inputs/placement.json",
                },
            },
        )
        records.append({"sample_id": sample_id, "path": f"samples/{sample_id}"})
    write_json(
        root / "manifest.json", {"samples": records, "default_groups": ["qualitative", "failure"]}
    )
    return root


def test_selection_preserves_manifest_order_and_removes_overlap(dataset_path):
    dataset = ExampleDataset(dataset_path)
    assert [item.sample_id for item in dataset.select()] == ["sample_a", "sample_b"]
    assert [item.sample_id for item in dataset.select(["sample_c", "qualitative", "sample_a"])] == [
        "sample_a",
        "sample_c",
    ]
    assert len(dataset.select(["all"])) == 3
    with pytest.raises(ValueError, match="Unknown example"):
        dataset.select(["missing"])


@pytest.mark.parametrize("sample_id", ["../escape", "/absolute", "a/b", "..", "", 42])
def test_manifest_ids_cannot_escape_output_directories(dataset_path, sample_id):
    manifest = read_json(dataset_path / "manifest.json")
    manifest["samples"][0]["sample_id"] = sample_id
    write_json(dataset_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Invalid sample ID"):
        ExampleDataset(dataset_path)


def test_duplicate_sample_id_is_rejected(dataset_path):
    manifest = read_json(dataset_path / "manifest.json")
    manifest["samples"].append(manifest["samples"][0])
    write_json(dataset_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Duplicate sample ID"):
        ExampleDataset(dataset_path)


def test_sample_metadata_id_must_match_manifest(dataset_path):
    path = dataset_path / "samples/sample_a/sample.json"
    metadata = read_json(path)
    metadata["sample_id"] = "different_id"
    write_json(path, metadata)
    with pytest.raises(ValueError, match="Mismatched sample ID"):
        ExampleDataset(dataset_path)


def test_manifest_directory_symlink_escape_is_rejected(dataset_path, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (dataset_path / "linked").symlink_to(outside, target_is_directory=True)
    manifest = read_json(dataset_path / "manifest.json")
    manifest["samples"][0]["path"] = "linked"
    write_json(dataset_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="escapes the dataset"):
        ExampleDataset(dataset_path)


def test_sample_local_paths_cannot_escape_the_sample(dataset_path):
    example = ExampleDataset(dataset_path).examples[0]
    assert example.resolve("inputs/observation.npz").parent.name == "inputs"
    with pytest.raises(ValueError, match="escapes its directory"):
        example.resolve("../../outside.npz")


def test_hash_checks_detect_equal_length_corruption_and_path_escape(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abcd")
    record = {"path": path.name, "bytes": 4, "sha256": file_hash(path)}
    assert verify_files(tmp_path, [record]) == []
    path.write_bytes(b"abce")
    assert "Checksum mismatch" in verify_files(tmp_path, [record])[0]
    escaped = {**record, "path": "../outside.bin"}
    assert "Invalid manifest path" in verify_files(tmp_path, [escaped])[0]


def test_text_checksums_support_standard_format_and_reject_escape(tmp_path):
    payload = tmp_path / "model.bin"
    payload.write_bytes(b"model")
    checksums = tmp_path / "checksums.sha256"
    checksums.write_text(f"{file_hash(payload)}  model.bin\n", encoding="utf-8")
    assert verify_checksum_file(tmp_path, checksums) == []
    checksums.write_text(f"{file_hash(payload)}  ../model.bin\n", encoding="utf-8")
    assert "Invalid checksum path" in verify_checksum_file(tmp_path, checksums)[0]
    checksums.write_text("invalid checksum\n", encoding="utf-8")
    assert "Malformed checksum line" in verify_checksum_file(tmp_path, checksums)[0]


def test_invalid_json_and_object_arrays_do_not_replace_existing_files(tmp_path):
    document = tmp_path / "record.json"
    write_json(document, {"value": 1})
    with pytest.raises(ValueError):
        write_json(document, {"value": float("nan")})
    assert read_json(document) == {"value": 1}
    arrays = tmp_path / "forecast.npz"
    write_arrays(arrays, {"state": np.zeros((1, 30, 66), dtype=np.float32)})
    before = arrays.read_bytes()
    with pytest.raises(ValueError, match="Object arrays"):
        write_arrays(arrays, {"metadata": {"sample_id": "example"}})
    assert arrays.read_bytes() == before
    with np.load(arrays, allow_pickle=False) as archive:
        assert archive["state"].shape == (1, 30, 66)
    assert not list(tmp_path.glob(".*"))


def test_json_reader_rejects_nonobject_documents(tmp_path):
    path = tmp_path / "record.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_json(path)


def test_existing_output_requires_explicit_overwrite(tmp_path):
    output = tmp_path / "outputs"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing results", encoding="utf-8")
    args = argparse.Namespace(output=output, overwrite=False)
    with pytest.raises(FileExistsError, match="not empty"):
        _ensure_output(args, [])
    assert sentinel.read_text() == "existing results"
    args.overwrite = True
    _ensure_output(args, [])
    assert sentinel.read_text() == "existing results"


def test_draw_selection_options_are_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["forecast", "--mano-dir", "mano", "--draw", "2", "--num-draws", "5"])


def _prediction(sample_id: str) -> dict:
    return {"sample_id": sample_id, "parse_ok": True, "intent": {"left": {}, "right": {}}}


def test_external_intentions_require_coverage_and_unique_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(_prediction("sample_a")) + "\n", encoding="utf-8")
    assert set(_read_intentions(path, {"sample_a"})) == {"sample_a"}
    with pytest.raises(ValueError, match="Missing intention records"):
        _read_intentions(path, {"sample_a", "sample_b"})
    path.write_text((json.dumps(_prediction("sample_a")) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate intention record"):
        _read_intentions(path, {"sample_a"})


@pytest.mark.parametrize(
    "record",
    [
        [],
        {"sample_id": 5},
        {"sample_id": "sample_a", "parse_ok": False, "intent": {}},
        {"sample_id": "sample_a", "parse_ok": 0, "intent": {}},
        {"sample_id": "sample_a", "parse_ok": "false", "intent": {}},
        {"sample_id": "sample_a", "intent": None},
    ],
)
def test_invalid_external_intention_records_are_rejected(tmp_path, record):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _read_intentions(path, {"sample_a"})


@pytest.mark.parametrize("state", [None, np.full((30, 66), np.nan, dtype=np.float32)])
def test_input_check_rejects_missing_or_nonfinite_observed_state(dataset_path, monkeypatch, state):
    from gaze2hoi.intention.inputs import IntentionInput

    monkeypatch.setattr(
        IntentionInput,
        "load",
        lambda *args, **kwargs: SimpleNamespace(image_paths=(), load_gaze=lambda: None),
    )
    dataset = ExampleDataset(dataset_path)
    example = dataset.examples[0]
    inputs = example.directory / "inputs"
    inputs.mkdir()
    observation = {} if state is None else {"state": state}
    np.savez(inputs / "observation.npz", **observation)
    np.savez(
        inputs / "objects.npz",
        state=np.zeros((1, 30, 9)),
        valid=np.ones((1, 30)),
        slots=np.zeros(1, dtype=np.int64),
    )
    (inputs / "placement.json").write_text("{}", encoding="utf-8")
    assert check_examples(dataset, [example])


@pytest.mark.parametrize("slots", [np.array([0.5]), np.array([-1]), np.array([33])])
def test_input_check_rejects_invalid_global_object_slots(dataset_path, monkeypatch, slots):
    from gaze2hoi.intention.inputs import IntentionInput

    monkeypatch.setattr(
        IntentionInput,
        "load",
        lambda *args, **kwargs: SimpleNamespace(image_paths=(), load_gaze=lambda: None),
    )
    dataset = ExampleDataset(dataset_path)
    example = dataset.examples[0]
    inputs = example.directory / "inputs"
    inputs.mkdir()
    np.savez(inputs / "observation.npz", state=np.zeros((30, 66), dtype=np.float32))
    np.savez(
        inputs / "objects.npz", state=np.zeros((1, 30, 9)), valid=np.ones((1, 30)), slots=slots
    )
    (inputs / "placement.json").write_text("{}", encoding="utf-8")
    assert check_examples(dataset, [example])


def test_intention_only_inputs_do_not_require_motion_assets(dataset_path, monkeypatch):
    from gaze2hoi.intention.inputs import IntentionInput

    monkeypatch.setattr(
        IntentionInput,
        "load",
        lambda *args, **kwargs: SimpleNamespace(image_paths=(), load_gaze=lambda: None),
    )
    dataset = ExampleDataset(dataset_path)
    assert check_examples(dataset, dataset.examples, include_motion=False) == []
    assert check_examples(dataset, dataset.examples)
