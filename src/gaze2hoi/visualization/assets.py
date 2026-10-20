"""MANO and object meshes, loaded only by the rendering entry point."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import rotations_from_6d

LEFT_HAND_COLOR = (60, 120, 255, 255)
RIGHT_HAND_COLOR = (255, 70, 70, 255)
SCENE_OBJECT_COLOR = (200, 200, 205, 255)


class ManoMesher:
    def __init__(self, model_dir: str | Path) -> None:
        import smplx
        import torch

        model_dir = Path(model_dir)
        self.layers = {}
        for side in ("left", "right"):
            path = model_dir / f"MANO_{side.upper()}.pkl"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path.name}; obtain MANO model files and set mano_dir to their directory"
                )
            self.layers[side] = smplx.create(
                str(path),
                "mano",
                use_pca=True,
                is_rhand=side == "right",
                num_pca_comps=15,
                num_betas=10,
            )
        left, right = self.layers["left"], self.layers["right"]
        if torch.sum(torch.abs(left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])) < 1:
            left.shapedirs[:, 0, :] *= -1

    def mesh(self, side: str, betas: np.ndarray, state: np.ndarray):
        import pyrender
        import torch
        import trimesh
        from scipy.spatial.transform import Rotation

        if not np.isfinite(state).all():
            return None
        rotation = rotations_from_6d(state[3:9][None])[0]
        axis_angle = Rotation.from_matrix(rotation).as_rotvec()
        layer = self.layers[side]
        with torch.inference_mode():
            output = layer(
                betas=torch.from_numpy(np.asarray(betas, dtype=np.float32))[None],
                global_orient=torch.from_numpy(axis_angle.astype(np.float32))[None],
                hand_pose=torch.from_numpy(np.asarray(state[9:24], dtype=np.float32))[None],
                transl=torch.from_numpy(np.asarray(state[:3], dtype=np.float32))[None],
                return_verts=True,
            )
        mesh = trimesh.Trimesh(
            vertices=output.vertices[0].numpy(), faces=layer.faces, process=False
        )
        mesh.visual.vertex_colors = LEFT_HAND_COLOR if side == "left" else RIGHT_HAND_COLOR
        return pyrender.Mesh.from_trimesh(mesh, smooth=True)


def load_object(path: Path, textured: bool):
    import pyrender
    import trimesh

    if not path.is_file():
        raise FileNotFoundError(f"Missing object mesh: {path}")
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh in {path}")
    if not textured:
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=SCENE_OBJECT_COLOR)
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def load_furniture(path: Path, alpha: float):
    import pyrender

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(97 / 255, 101 / 255, 99 / 255, alpha),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode="BLEND",
        doubleSided=True,
    )
    meshes = []
    with np.load(path, allow_pickle=False) as archive:
        for index in range(int(archive["primitive_count"])):
            prefix = f"primitive_{index}_"
            primitive = pyrender.Primitive(
                positions=archive[prefix + "positions"],
                normals=archive[prefix + "normals"],
                indices=archive[prefix + "indices"],
                material=material,
            )
            meshes.append(pyrender.Mesh(primitives=[primitive]))
    return meshes
