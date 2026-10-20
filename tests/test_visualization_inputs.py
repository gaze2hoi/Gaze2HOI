"""Rendering preflight failures must not require an OpenGL context."""

import json

import pytest

from gaze2hoi.visualization.renderer import ForecastRenderer, _resolve_asset, render_forecast
from gaze2hoi.visualization.video import encode_video


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "dataset"
    directory = root / "samples" / "example"
    (directory / "visualization").mkdir(parents=True)
    (directory / "inputs").mkdir()
    (directory / "inputs/observation.npz").touch()
    (root / "shared/objects").mkdir(parents=True)
    (root / "shared/vocabulary.json").write_text('{"slots": []}')
    (directory / "visualization/furniture.npz").touch()
    config = {
        "native_size": 512,
        "object_vocabulary": "../../shared/vocabulary.json",
        "object_directory": "../../shared/objects",
        "furniture": "visualization/furniture.npz",
        "colored_object_slots": [],
        "gray_object_slots": [],
    }
    (directory / "visualization/config.json").write_text(json.dumps(config))
    return directory


def test_shared_asset_paths_remain_inside_dataset(sample):
    expected = sample.parent.parent / "shared/vocabulary.json"
    assert _resolve_asset(sample, "../../shared/vocabulary.json") == expected
    assert _resolve_asset(sample, "../../shared/objects", directory=True).is_dir()


@pytest.mark.parametrize("field", ["object_vocabulary", "object_directory", "furniture"])
@pytest.mark.parametrize("relative", ["../../../outside", "/absolute/path"])
def test_config_asset_paths_cannot_escape_dataset(sample, field, relative):
    path = sample / "visualization/config.json"
    config = json.loads(path.read_text())
    config[field] = relative
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="relative paths|leaves the dataset"):
        ForecastRenderer(sample, "unused-mano")


def test_asset_symlink_cannot_escape_dataset(sample, tmp_path):
    external = tmp_path / "outside.json"
    external.write_text("{}")
    link = sample / "visualization/external.json"
    link.symlink_to(external)
    with pytest.raises(ValueError, match="leaves the dataset"):
        _resolve_asset(sample, "visualization/external.json")


@pytest.mark.parametrize("size", [-128, 0, 127, 129, 256.5, True])
def test_invalid_output_size_fails_before_loading_assets(size):
    with pytest.raises(ValueError, match="even integer"):
        ForecastRenderer("missing-sample", "missing-mano", size=size)


def test_missing_ffmpeg_fails_before_loading_assets(monkeypatch, tmp_path):
    monkeypatch.setattr("gaze2hoi.visualization.video.shutil.which", lambda _: None)
    output = tmp_path / "result.mp4"
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        render_forecast("missing-sample", "missing-prediction", "missing-mano", output)
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        encode_video([], output, 256)
    assert not output.exists()


def test_raw_motion_state_uses_explicit_object_slots_and_validity():
    import numpy as np

    from gaze2hoi.visualization.geometry import prepare_prediction

    observation = {
        "left_hand_valid": np.ones(30, dtype=bool),
        "right_hand_valid": np.ones(30, dtype=bool),
        "scene_present": np.ones(3, dtype=bool),
        "scene_transl_last": np.zeros((3, 3)),
        "scene_rot6d_last": np.tile([1, 0, 0, 0, 1, 0], (3, 1)),
    }
    state = np.zeros((30, 66))
    state[:, 48:57] = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0]
    state[:, 57:66] = [0.4, 0.5, 0.6, 1, 0, 0, 0, 1, 0]
    payload = {
        "state": state,
        "left_obj_slot": np.array(2),
        "right_obj_slot": np.array(0),
        "left_hand_valid": np.zeros(30, dtype=bool),
        "right_hand_valid": np.ones(30, dtype=bool),
        "left_obj_valid": np.ones(30, dtype=bool),
        "right_obj_valid": np.zeros(30, dtype=bool),
    }
    prepared = prepare_prediction(observation, payload)
    np.testing.assert_allclose(prepared["object_poses"][:, 2, :3, 3], state[:, 48:51])
    assert not prepared["hand_valid"][:, 0].any()
    assert prepared["hand_valid"][:, 1].all()
    assert prepared["object_valid"][:, 2].all()
    assert not prepared["object_valid"][:, 0].any()
    assert prepared["object_valid"][:, 1].all()
