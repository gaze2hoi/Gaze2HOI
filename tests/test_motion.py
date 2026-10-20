"""Inference contracts that do not require released weights or MANO files."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from gaze2hoi.motion.assets import NormStats
from gaze2hoi.motion.binding import bind_observation, model_inputs, parse_intention
from gaze2hoi.motion.sampling import DDIMSampler, draw_seed, initial_noise


def test_sample_keyed_noise_is_independent_of_draw_order():
    expected = initial_noise("sample-a", 2)
    initial_noise("sample-b", 0)
    initial_noise("sample-a", 4)
    assert torch.equal(expected, initial_noise("sample-a", 2))
    assert not torch.equal(expected, initial_noise("sample-a", 1))
    assert not torch.equal(expected, initial_noise("sample-a", 2, seed=1000))
    assert expected.shape == (1, 60, 66)
    assert expected.dtype == torch.float32
    assert expected.device.type == "cpu"


def test_homepage_noise_seed_is_preserved():
    assert draw_seed(999, "P0009_8c0caf6c_000215_000274", 0) == 103772317741487717


@pytest.fixture
def sample_contract(tmp_path: Path):
    sample = tmp_path / "sample"
    inputs = sample / "inputs"
    inputs.mkdir(parents=True)
    (sample / "sample.json").write_text(
        json.dumps({"sample_id": "example", "contact_left": "cup", "contact_right": None})
    )
    observation = {
        "state": np.ones((30, 66), dtype=np.float32),
        "left_hand_valid": np.ones(30, dtype=np.uint8),
        "right_hand_valid": np.ones(30, dtype=np.uint8),
        "mano_betas": np.zeros(10, dtype=np.float32),
        "left_j0": np.zeros(3, dtype=np.float32),
        "right_j0": np.zeros(3, dtype=np.float32),
        "scene_present": np.r_[np.ones(2), np.zeros(31)].astype(np.float32),
        "scene_transl_last": np.zeros((33, 3), dtype=np.float32),
        "scene_transl_mean": np.zeros((33, 3), dtype=np.float32),
        "scene_rot6d_last": np.zeros((33, 6), dtype=np.float32),
        "furn_present": np.zeros(6, dtype=np.float32),
        "furn_type_id": np.zeros(6, dtype=np.int64),
        "furn_shape_id": np.zeros(6, dtype=np.int64),
        "furn_params": np.zeros((6, 2), dtype=np.float32),
        "furn_center": np.zeros((6, 3), dtype=np.float32),
        "furn_yaw": np.zeros(6, dtype=np.float32),
        "anchor_rotation": np.eye(3),
        "anchor_origin": np.zeros(3),
    }
    np.savez(inputs / "observation.npz", **observation)
    tracks = np.stack((np.ones((30, 9)), np.full((30, 9), 2))).astype(np.float32)
    valid = np.ones((2, 30), dtype=np.uint8)
    np.savez(inputs / "objects.npz", state=tracks, valid=valid, slots=np.array([0, 1]))
    vocab = {
        "action": ["<none>", "grasp"],
        "object": ["<none>", "cup", "bowl"],
        "part": ["<none>", "rim"],
        "deep_action": ["<none>", "move"],
        "target": ["<none>", "cup"],
    }
    objects = {
        "slots": [{"name": "cup", "uid": "1", "slot": 0}, {"name": "bowl", "uid": "2", "slot": 1}]
    }
    intention = {
        "left": {"action": None},
        "right": {
            "action": "grasp",
            "object": "bowl",
            "part": "rim",
            "deep_action": "move",
            "target": "cup",
        },
    }
    stats = {}
    for name, shape in (
        ("state", (66,)),
        ("scene_transl", (3,)),
        ("furn_center", (3,)),
        ("furn_params", (2,)),
        ("place_target", (3,)),
    ):
        stats[f"{name}_mean"] = np.full(shape, 0.2, dtype=np.float32)
        stats[f"{name}_std"] = np.full(shape, 2.0, dtype=np.float32)
    np.savez(tmp_path / "norm_stats.npz", **stats)
    return sample, vocab, objects, intention, NormStats(tmp_path / "norm_stats.npz")


def test_predicted_object_and_observed_contact_determine_binding(sample_contract):
    sample, vocab, objects, intention, _ = sample_contract
    shard, bindings, _ = bind_observation(sample, intention, vocab, objects)
    assert bindings["left"]["source"] == "contact"
    assert int(shard["left_obj_slot"]) == 0
    assert int(shard["right_obj_slot"]) == 1
    np.testing.assert_array_equal(shard["state"][:30, 48:57], 1.0)
    np.testing.assert_array_equal(shard["state"][:30, 57:66], 2.0)
    np.testing.assert_array_equal(shard["state"][30:], 0.0)
    assert shard["target_slots"].tolist() == [-1, 0]


def test_inference_ignores_reference_files_and_future_labels(sample_contract):
    sample, vocab, objects, intention, stats = sample_contract
    shard, _, _ = bind_observation(sample, intention, vocab, objects)
    baseline = model_inputs(shard, stats)
    for name in ("ground_truth", "reference_outputs"):
        directory = sample / name
        directory.mkdir()
        (directory / "intention.json").write_text("not valid JSON")
        np.savez(directory / "motion.npz", state=np.full((60, 66), np.nan))
    repeated, _, _ = bind_observation(sample, intention, vocab, objects)
    repeated["state"][30:] = 12345.0
    for key in ("left_hand_valid", "right_hand_valid", "left_obj_valid", "right_obj_valid"):
        repeated[key][30:] = 1 - repeated[key][30:]
    actual = model_inputs(repeated, stats)
    for key, expected in baseline.items():
        assert torch.equal(actual[key], expected), key
    assert torch.count_nonzero(actual["state"][30:]) == 0
    assert torch.all(actual["state_mask"][30:] == 1)


def test_insufficient_observed_object_coverage_leaves_hand_unbound(sample_contract):
    sample, vocab, objects, intention, _ = sample_contract
    path = sample / "inputs" / "objects.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["valid"][1, :16] = 0
    np.savez(path, **arrays)
    shard, _, _ = bind_observation(sample, intention, vocab, objects)
    assert int(shard["right_obj_slot"]) == -1
    assert not shard["right_obj_valid"].any()
    assert not shard["state"][:, 57:].any()


def test_sampler_preserves_all_observed_states(sample_contract):
    sample, vocab, objects, intention, stats = sample_contract
    shard, _, _ = bind_observation(sample, intention, vocab, objects)
    batch = {key: value[None] for key, value in model_inputs(shard, stats).items()}

    class ConstantVelocity(nn.Module):
        def forward(self, state, timestep, observed, conditions, **overrides):
            return torch.ones_like(state)

    sampler = DDIMSampler(stats)
    output = sampler(ConstantVelocity(), batch, initial_noise("example", 0), steps=3)
    expected = sampler.denormalize(batch["state"])
    torch.testing.assert_close(output[:, :30], expected[:, :30], rtol=0, atol=0)
    assert torch.isfinite(output).all()
    assert not torch.equal(output[:, 30:], expected[:, 30:])


def test_parse_failure_is_not_silently_treated_as_an_intention():
    with pytest.raises(ValueError, match="unparsed"):
        parse_intention({"parse_ok": False, "intent": None})


@pytest.mark.parametrize("record", [None, [], {}, {"left": None}, {"left": [], "right": None}])
def test_malformed_intention_shape_is_rejected(record):
    with pytest.raises(ValueError):
        parse_intention(record)


def test_idle_hand_cannot_silently_ignore_an_object():
    with pytest.raises(ValueError, match="idle hand"):
        parse_intention({"left": {"action": None, "object": "cup"}, "right": None})


def test_external_intention_sample_id_must_match(sample_contract):
    sample, vocab, objects, intention, _ = sample_contract
    with pytest.raises(ValueError, match="sample ID"):
        bind_observation(
            sample,
            {"sample_id": "another-example", "intent": intention, "parse_ok": True},
            vocab,
            objects,
        )


def test_unknown_predicted_object_is_rejected(sample_contract):
    sample, vocab, objects, intention, _ = sample_contract
    intention["right"]["object"] = "unknown object"
    with pytest.raises(ValueError, match="Unknown right intention object"):
        bind_observation(sample, intention, vocab, objects)


def test_known_object_outside_sample_candidates_is_rejected(sample_contract):
    sample, vocab, objects, intention, _ = sample_contract
    vocab["object"].append("spoon")
    objects["slots"].append({"name": "spoon", "uid": "3", "slot": 2})
    intention["right"]["object"] = "spoon"
    with pytest.raises(ValueError, match="not an observed candidate"):
        bind_observation(sample, intention, vocab, objects)


def test_nonbinary_observed_validity_is_rejected(sample_contract):
    sample, vocab, objects, intention, _ = sample_contract
    path = sample / "inputs" / "objects.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["valid"] = arrays["valid"].astype(np.float32)
    arrays["valid"][0, 0] = 0.5
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="binary flags"):
        bind_observation(sample, intention, vocab, objects)
