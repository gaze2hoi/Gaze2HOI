"""Portable intention inputs and structured output contracts."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gaze2hoi.intention.grammar import build_json_schema
from gaze2hoi.intention.inputs import IntentionInput, sample_context
from gaze2hoi.intention.serialization import empty_hand, intent_to_text, parse_prediction


@pytest.fixture
def vocabulary():
    metadata = {
        "legal_objects": ["bowl", "mug"],
        "parts_by_object": {"bowl": ["rim"], "mug": ["body", "handle"]},
        "target_names": ["bowl", "mug", "table"],
    }
    schema = {
        "actions": [{"value": "grasp"}, {"value": "reach and grasp"}],
        "deep_actions": [{"value": "place"}, {"value": "shake to mix"}],
    }
    return metadata, schema


@pytest.fixture
def context(vocabulary):
    return sample_context(*vocabulary)


@pytest.fixture
def intention():
    return {
        "left": {
            "action": "grasp",
            "object": "bowl",
            "part": "rim",
            "deep_action": "place",
            "target": "table",
            "point": [250, 750],
        },
        "right": empty_hand(),
    }


@pytest.fixture
def sample(tmp_path, vocabulary):
    root = tmp_path / "dataset"
    directory = root / "samples" / "example"
    inputs = directory / "inputs"
    inputs.mkdir(parents=True)
    shared = root / "shared"
    shared.mkdir()
    metadata, schema = vocabulary
    (shared / "intention_schema.json").write_text(json.dumps(schema))
    images = []
    for index in range(5):
        path = inputs / f"frame-{index}.png"
        Image.new("RGB", (8, 8), (index, 20, 40)).save(path)
        images.append(f"inputs/{path.name}")
    Image.new("RGB", (8, 8), (90, 100, 110)).save(shared / "reference.png")
    images.append("../../shared/reference.png")
    prompt = "Observed: " + "<image>" * 5 + "\nCandidate: <image> bowl\nReturn JSON."
    payload = {
        "sample_id": "example",
        "messages": [
            {"role": "system", "content": "Predict the next interaction."},
            {"role": "user", "content": prompt},
        ],
        "images": images,
        "context": metadata,
        "gaze": "inputs/gaze.npz",
        "schema": "../../shared/intention_schema.json",
    }
    (inputs / "intention.json").write_text(json.dumps(payload))
    np.savez(inputs / "gaze.npz", heatmaps=np.zeros((5, 2, 42, 42), dtype=np.float32))
    return directory


def rewrite_payload(sample, change):
    path = sample / "inputs/intention.json"
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))


def test_input_preserves_prompt_interleaving_and_image_order(sample):
    source = json.loads((sample / "inputs/intention.json").read_text())
    item = IntentionInput.load(sample)
    assert item.sample_id == "example"
    assert item.context.object_names == {"bowl", "mug"}
    assert item.load_gaze().shape == (5, 2, 42, 42)
    messages, images = item.open_chat()
    try:
        assert [message["role"] for message in messages] == ["system", "user"]
        assert messages[0] == source["messages"][0]
        reconstructed = "".join(
            "<image>" if block["type"] == "image" else block["text"]
            for block in messages[1]["content"]
        )
        assert reconstructed == source["messages"][1]["content"]
        assert [image.getpixel((0, 0)) for image in images] == [
            (index, 20, 40) for index in range(5)
        ] + [(90, 100, 110)]
    finally:
        for image in images:
            image.close()


@pytest.mark.parametrize(
    "change",
    [
        lambda payload: payload["messages"].append({"role": "assistant", "content": "answer"}),
        lambda payload: payload["messages"][1].update(role="assistant"),
        lambda payload: payload["messages"].reverse(),
    ],
)
def test_assistant_labels_cannot_enter_inference_messages(sample, change):
    rewrite_payload(sample, change)
    with pytest.raises(ValueError, match="system|user"):
        IntentionInput.load(sample)


@pytest.mark.parametrize("field", ["images", "gaze", "schema"])
@pytest.mark.parametrize("path", ["../../../outside", "/absolute/path"])
def test_sample_asset_paths_are_confined_to_dataset(sample, field, path):
    def change(payload):
        if field == "images":
            payload[field][0] = path
        else:
            payload[field] = path

    rewrite_payload(sample, change)
    with pytest.raises(ValueError, match="relative asset path|leaves the dataset"):
        IntentionInput.load(sample)


@pytest.mark.parametrize("filename", ["intention.json", "frame-0.png", "gaze.npz"])
def test_sample_asset_symlinks_cannot_escape_dataset(sample, tmp_path, filename):
    original = sample / "inputs" / filename
    outside = tmp_path / filename
    original.replace(outside)
    original.symlink_to(outside)
    with pytest.raises(ValueError, match="leaves the dataset"):
        IntentionInput.load(sample)


def test_sample_directory_must_belong_to_explicit_dataset_root(sample, tmp_path):
    with pytest.raises(ValueError, match="inside dataset_root"):
        IntentionInput.load(sample, dataset_root=tmp_path / "different-dataset")


def test_image_tag_count_is_validated(sample):
    rewrite_payload(sample, lambda payload: payload["images"].pop())
    with pytest.raises(ValueError, match="placeholder count"):
        IntentionInput.load(sample)


@pytest.mark.parametrize("shape", [(5, 1, 42, 42), (4, 2, 42, 42), (5, 2, 41, 42)])
def test_gaze_shape_is_validated(sample, shape):
    np.savez(sample / "inputs/gaze.npz", heatmaps=np.zeros(shape))
    with pytest.raises(ValueError, match="expected gaze shape"):
        IntentionInput.load(sample).load_gaze()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_gaze_values_must_be_finite(sample, value):
    heatmaps = np.zeros((5, 2, 42, 42), dtype=np.float32)
    heatmaps[0, 0, 0, 0] = value
    np.savez(sample / "inputs/gaze.npz", heatmaps=heatmaps)
    with pytest.raises(ValueError, match="non-finite"):
        IntentionInput.load(sample).load_gaze()


def test_gaze_archive_requires_named_heatmaps(sample):
    np.savez(sample / "inputs/gaze.npz", unrelated=np.zeros((5, 2, 42, 42)))
    with pytest.raises(ValueError, match="array named 'heatmaps'"):
        IntentionInput.load(sample).load_gaze()


def test_vocabulary_rejects_missing_or_nonstring_actions(vocabulary):
    metadata, schema = vocabulary
    for invalid in ({}, {**schema, "actions": []}, {**schema, "actions": [{"value": 12}]}):
        with pytest.raises(ValueError, match="vocabulary|list of strings"):
            sample_context(metadata, invalid)


def test_schema_requires_complete_hands_and_object_specific_parts(context):
    schema = build_json_schema(context)
    assert schema["required"] == ["left", "right"]
    assert schema["additionalProperties"] is False
    branches = schema["properties"]["left"]["anyOf"]
    assert all(branch["additionalProperties"] is False for branch in branches)
    for branch in branches:
        assert set(branch["required"]) == set(empty_hand())
        properties = branch["properties"]
        if properties["object"].get("type") == "null":
            assert all(field == {"type": "null"} for field in properties.values())
            continue
        (name,) = properties["object"]["enum"]
        parts = properties["part"]["anyOf"][0]["enum"]
        assert set(parts) == context.parts_by_object[name]
        if properties["deep_action"] == {"enum": ["place"]}:
            point = properties["point"]
            assert point["minItems"] == point["maxItems"] == 2
            assert point["items"] == {"type": "integer", "minimum": 0, "maximum": 1000}
        else:
            assert properties["point"] == {"type": "null"}
            assert "place" not in properties["deep_action"]["anyOf"][0]["enum"]


def test_valid_intention_roundtrip(context, intention):
    raw = intent_to_text(intention)
    parsed = parse_prediction(raw, context)
    assert parsed.ok
    assert parsed.intent == intention
    assert parse_prediction(f"```json\n{raw}\n```", context).intent == intention
    assert parse_prediction(json.dumps({"left": empty_hand(), "right": empty_hand()}), context).ok


@pytest.mark.parametrize(
    "slot,value",
    [
        ("action", "unknown"),
        ("action", []),
        ("object", "unknown"),
        ("object", {}),
        ("part", "handle"),
        ("part", []),
        ("deep_action", "unknown"),
        ("deep_action", []),
        ("target", "unknown"),
        ("target", {}),
        ("point", None),
        ("point", [250]),
        ("point", [True, 750]),
        ("point", [-1, 750]),
        ("point", [250, 1001]),
        ("point", [float("inf"), 750]),
    ],
)
def test_invalid_slot_values_return_parse_errors(context, intention, slot, value):
    intention["left"][slot] = value
    parsed = parse_prediction(json.dumps(intention), context)
    assert not parsed.ok
    assert parsed.errors


def test_parser_rejects_incomplete_unknown_and_inconsistent_fields(context, intention):
    candidates = []
    missing_slot = copy.deepcopy(intention)
    del missing_slot["left"]["part"]
    candidates.append(missing_slot)
    missing_hand = {"left": intention["left"]}
    candidates.append(missing_hand)
    unknown_slot = copy.deepcopy(intention)
    unknown_slot["right"]["confidence"] = 1.0
    candidates.append(unknown_slot)
    candidates.append({**intention, "explanation": "text"})
    null_action = copy.deepcopy(intention)
    null_action["left"]["action"] = None
    candidates.append(null_action)
    other_action = copy.deepcopy(intention)
    other_action["left"]["deep_action"] = "shake to mix"
    candidates.append(other_action)
    for candidate in candidates:
        parsed = parse_prediction(json.dumps(candidate), context)
        assert not parsed.ok
        assert parsed.errors


@pytest.mark.parametrize(
    "wrapper", ["[{value}]", "Prefix {value}", "{value} trailing", "{value}{value}"]
)
def test_parser_requires_one_json_object(context, intention, wrapper):
    parsed = parse_prediction(wrapper.format(value=json.dumps(intention)), context)
    assert not parsed.ok
    assert parsed.intent is None


def test_intention_import_does_not_load_model_dependencies():
    package_root = str(Path(__file__).resolve().parents[1] / "src")
    environment = dict(os.environ, PYTHONPATH=package_root)
    code = (
        "import sys; import gaze2hoi.intention; "
        "assert not ({'torch', 'transformers', 'peft', 'xgrammar', 'pyrender'} & sys.modules.keys())"
    )
    subprocess.run([sys.executable, "-c", code], env=environment, check=True)
