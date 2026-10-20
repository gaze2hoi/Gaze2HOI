"""Offscreen trajectory rendering with sample-specific camera and scene styling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def _resolve_asset(sample_dir: Path, relative: str, *, directory: bool = False) -> Path:
    if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
        raise ValueError(f"Rendering asset paths must be nonempty relative paths: {relative!r}")
    dataset_root = sample_dir.parent.parent
    path = (sample_dir / relative).resolve()
    if not path.is_relative_to(dataset_root):
        raise ValueError(f"Rendering asset path leaves the dataset directory: {relative!r}")
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        raise FileNotFoundError(f"Missing rendering {'directory' if directory else 'file'}: {path}")
    return path


def _load_arrays(path: str | Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _configure_surfaces(renderer, furniture) -> None:
    import ctypes

    import pyrender
    from OpenGL import GL

    primitives = {id(primitive) for mesh in furniture for primitive in mesh.primitives}
    get_program = renderer._renderer._get_primitive_program

    def program(primitive, flags, program_flags):
        if id(primitive) in primitives and not flags & pyrender.RenderFlags.SEG:
            flags |= pyrender.RenderFlags.FLAT
        return get_program(primitive, flags, program_flags)

    renderer._renderer._get_primitive_program = program
    original_draw = renderer._renderer._bind_and_draw_primitive

    def draw(primitive, pose, program, flags):
        if not (flags & pyrender.RenderFlags.SEG and primitive.material.doubleSided):
            return original_draw(primitive, pose, program, flags)
        program.set_uniform("M", pose)
        primitive._bind()
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ZERO)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)
        count = 1 if primitive.poses is None else len(primitive.poses)
        if primitive.indices is not None:
            GL.glDrawElementsInstanced(
                primitive.mode,
                primitive.indices.size,
                GL.GL_UNSIGNED_INT,
                ctypes.c_void_p(0),
                count,
            )
        else:
            GL.glDrawArraysInstanced(primitive.mode, 0, len(primitive.positions), count)
        primitive._unbind()

    renderer._renderer._bind_and_draw_primitive = draw


def _style_objects(rgb, labels, neutral, render_size: int):
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    rgb = rgb.copy()
    mask = np.isin(labels, neutral)
    if mask.any():
        luminance = rgb[mask, :3].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722]) / 255
        rgb[mask, :3] = np.rint(luminance[:, None] * np.array([224, 228, 232])).astype(np.uint8)
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = labels == label
        y, x = np.nonzero(mask)
        region = (
            slice(max(0, y.min() - 4), min(render_size, y.max() + 5)),
            slice(max(0, x.min() - 4), min(render_size, x.max() + 5)),
        )
        local = mask[region]
        weight = (
            np.clip(2 * render_size / 2816 + 0.5 - distance_transform_edt(local), 0, 1)
            * local
            * 0.65
        )
        selected = weight > 0
        alpha = weight[selected, None]
        pixels = rgb[region]
        pixels[selected, :3] = np.rint(
            pixels[selected, :3] * (1 - alpha) + np.array([95, 105, 115]) * alpha
        ).astype(np.uint8)
    return rgb


class ForecastRenderer:
    """Create and use on one thread; frames are rendered directly from supplied states."""

    def __init__(
        self,
        sample_dir: str | Path,
        mano_dir: str | Path,
        *,
        size: int = 512,
        render_size: int | None = None,
        platform: str | None = None,
    ) -> None:
        from .video import validate_video_size

        validate_video_size(size)
        self.sample_dir = Path(sample_dir).resolve()
        config_path = _resolve_asset(self.sample_dir, "visualization/config.json")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.size = size
        self.render_size = self.config["native_size"] if render_size is None else render_size
        if type(self.render_size) is not int or self.render_size < 128:
            raise ValueError("Render size must be an integer of at least 128 pixels")
        observation_path = _resolve_asset(self.sample_dir, "inputs/observation.npz")
        vocabulary_path = _resolve_asset(self.sample_dir, self.config["object_vocabulary"])
        furniture_path = _resolve_asset(self.sample_dir, self.config["furniture"])
        _resolve_asset(self.sample_dir, self.config["object_directory"], directory=True)
        vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        object_paths = {}
        for slot in self.config["colored_object_slots"] + self.config["gray_object_slots"]:
            uid = str(vocabulary["slots"][slot]["uid"])
            if Path(uid).name != uid or "\\" in uid:
                raise ValueError(f"Invalid object mesh identifier: {uid!r}")
            relative = str(Path(self.config["object_directory"]) / f"{uid}.glb")
            object_paths[slot] = _resolve_asset(self.sample_dir, relative)
        if platform is not None:
            os.environ["PYOPENGL_PLATFORM"] = platform
        else:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("LP_NUM_THREADS", "4")
        import numpy as np
        import pyrender

        from .assets import ManoMesher, load_furniture, load_object

        self.observation = _load_arrays(observation_path)
        self.background = tuple(self.config["background_rgb"])
        self.camera_pose = np.asarray(self.config["camera_to_anchor_gl"], dtype=np.float64)
        k = np.asarray(self.config["render_linear_k"], dtype=np.float64).copy()
        scale = self.render_size / self.config["native_size"]
        k[:2] *= scale
        self.camera = pyrender.IntrinsicsCamera(
            fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2], znear=0.05, zfar=20.0
        )
        self.crop = tuple(int(round(value * scale)) for value in self.config["crop_native_xyxy"])
        self.hidden_hands = set(self.config.get("hidden_hands", []))
        self.gray_slots = set(self.config["gray_object_slots"])
        self.objects = {
            slot: load_object(path, slot not in self.gray_slots)
            for slot, path in object_paths.items()
        }
        self.furniture = load_furniture(furniture_path, self.config.get("furniture_alpha", 0.2))
        self.mano = ManoMesher(mano_dir)
        self.renderer = pyrender.OffscreenRenderer(self.render_size, self.render_size)
        _configure_surfaces(self.renderer, self.furniture)

    def prepare(self, prediction: str | Path | Mapping[str, Any], draw: int = 0):
        from .geometry import prepare_prediction

        if isinstance(prediction, (str, Path)):
            prediction = _load_arrays(prediction)
        return prepare_prediction(self.observation, prediction, draw)

    def frame(self, prepared: Mapping[str, Any], frame: int = 0, *, native: bool = False):
        import numpy as np
        import pyrender
        from PIL import Image

        if not 0 <= frame < 30:
            raise ValueError("Frame index must lie between 0 and 29")
        scene = pyrender.Scene(
            bg_color=np.array([*(value / 255 for value in self.background), 1.0]),
            ambient_light=np.array([0.45, 0.45, 0.48]),
        )
        scene.add(self.camera, pose=self.camera_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.5), pose=self.camera_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=np.eye(4))
        for mesh in self.furniture:
            scene.add(mesh)
        nodes, neutral = [], []
        for slot, mesh in self.objects.items():
            if prepared["object_valid"][frame, slot]:
                nodes.append(scene.add(mesh, pose=prepared["object_poses"][frame, slot]))
                if slot in self.gray_slots:
                    neutral.append(len(nodes))
        for hand, side in enumerate(("left", "right")):
            if side in self.hidden_hands or not prepared["hand_valid"][frame, hand]:
                continue
            betas = self.observation["mano_betas"]
            if betas.ndim == 2:
                betas = betas[hand]
            mesh = self.mano.mesh(side, betas, prepared["hands"][frame, hand])
            if mesh is not None:
                scene.add(mesh)
        color, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        segmentation = {node: (0, 0, 0) for node in scene.mesh_nodes}
        segmentation.update({node: (index + 1, 0, 0) for index, node in enumerate(nodes)})
        labels, _ = self.renderer.render(
            scene, flags=pyrender.RenderFlags.SEG, seg_node_map=segmentation
        )
        color = _style_objects(color, labels[..., 0], neutral, self.render_size)
        rgba = np.dstack((color[..., :3], (depth > 0).astype(np.uint8) * 255))
        rotation = int(self.config.get("display_rotation_degrees_clockwise", 90))
        image = Image.fromarray(np.rot90(rgba, k=-(rotation // 90)).copy())
        if native:
            return image
        image = (
            image.crop(self.crop)
            .convert("RGBa")
            .resize((self.size, self.size), Image.Resampling.LANCZOS)
            .convert("RGBA")
        )
        background = Image.new("RGBA", image.size, (*self.background, 255))
        return Image.alpha_composite(background, image).convert("RGB")

    def render(
        self, prediction: str | Path | Mapping[str, Any], output: str | Path, *, draw: int = 0
    ) -> Path:
        from .video import encode_video

        prepared = self.prepare(prediction, draw)
        frames = (self.frame(prepared, index) for index in range(30))
        return encode_video(frames, output, self.size)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None

    def __enter__(self) -> ForecastRenderer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def render_forecast(
    sample_dir: str | Path,
    prediction_npz: str | Path,
    mano_dir: str | Path,
    output: str | Path,
    *,
    size: int = 512,
    draw: int = 0,
    render_size: int | None = None,
    platform: str | None = None,
) -> Path:
    """Render one supplied draw to MP4 and a first-frame WebP poster."""
    from .video import require_ffmpeg, validate_video_size

    validate_video_size(size)
    if Path(output).suffix.lower() != ".mp4":
        raise ValueError("Video output must use the .mp4 extension")
    require_ffmpeg()
    with ForecastRenderer(
        sample_dir, mano_dir, size=size, render_size=render_size, platform=platform
    ) as renderer:
        return renderer.render(prediction_npz, output, draw=draw)
