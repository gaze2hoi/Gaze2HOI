"""Observed hand geometry and object-local contact features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import config as C

_EPS = 1e-8
TRAJ_LEN = 10
POS_SCALE = 3.0
VEL_SCALE = 30.0


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Differentiable Gram-Schmidt: (..., 6) -> (..., 3, 3)."""
    a0, a1 = (rot6d[..., 0:3], rot6d[..., 3:6])
    b0 = torch.nn.functional.normalize(a0, dim=-1, eps=_EPS)
    a1p = a1 - (b0 * a1).sum(-1, keepdim=True) * b0
    b1 = torch.nn.functional.normalize(a1p, dim=-1, eps=_EPS)
    b2 = torch.cross(b0, b1, dim=-1)
    return torch.stack([b0, b1, b2], dim=-1)


def _matrix_to_axis_angle(rot: torch.Tensor) -> torch.Tensor:
    """Differentiable rotmat -> axis-angle via quaternion (singularity-guarded)."""
    m = rot
    w = torch.sqrt((1.0 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]).clamp_min(_EPS)) / 2.0
    w = w.clamp_min(0.0001)
    x = (m[..., 2, 1] - m[..., 1, 2]) / (4.0 * w)
    y = (m[..., 0, 2] - m[..., 2, 0]) / (4.0 * w)
    z = (m[..., 1, 0] - m[..., 0, 1]) / (4.0 * w)
    quat = torch.stack([w, x, y, z], dim=-1)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    sin_half = quat[..., 1:].norm(dim=-1).clamp(min=_EPS, max=1.0 - 1e-07)
    angle = 2.0 * torch.asin(sin_half)
    axis = quat[..., 1:] / sin_half[..., None]
    return axis * angle[..., None]


def _build_mano_layers(mano_model_dir: Path) -> tuple[nn.Module, nn.Module]:
    """Create frozen left/right MANO layers with the released convention."""
    import smplx

    mano_model_dir = Path(mano_model_dir)
    for side in ("LEFT", "RIGHT"):
        path = mano_model_dir / f"MANO_{side}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing licensed MANO model: {path}")
    layers = {}
    for side, is_right in (("left", False), ("right", True)):
        layers[side] = smplx.create(
            str(mano_model_dir / f"MANO_{side.upper()}.pkl"),
            "mano",
            use_pca=True,
            is_rhand=is_right,
            num_pca_comps=C.HAND_POSE_DIM,
            num_betas=C.MANO_BETAS_DIM,
        )
    if (
        torch.sum(torch.abs(layers["left"].shapedirs[:, 0, :] - layers["right"].shapedirs[:, 0, :]))
        < 1
    ):
        layers["left"].shapedirs[:, 0, :] *= -1
    return (layers["left"], layers["right"])


def mano_fk_vertices(
    layer: nn.Module, betas: torch.Tensor, hand_block: torch.Tensor
) -> torch.Tensor:
    """hand_block (M, 24) metric -> MANO vertices (M, 778, 3) in the anchor frame."""
    transl = hand_block[:, C.HAND_TRANSL_SLICE]
    rot = _rot6d_to_matrix(hand_block[:, C.HAND_ROT6D_SLICE])
    aa = _matrix_to_axis_angle(rot)
    pose = hand_block[:, C.HAND_POSE_SLICE]
    out = layer(betas=betas, global_orient=aa, hand_pose=pose, transl=transl, return_verts=True)
    return out.vertices


@dataclass
class TemplateBank:
    """Per-object canonical geometry, indexed by the global object slot."""

    uids: list[str]
    points_local: torch.Tensor
    sdf_grid: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor

    @property
    def n_objects(self) -> int:
        return len(self.uids)

    def slot_of(self, uid: str) -> int:
        return self.uids.index(str(uid))

    def to(self, device: torch.device) -> TemplateBank:
        return TemplateBank(
            uids=self.uids,
            points_local=self.points_local.to(device),
            sdf_grid=self.sdf_grid.to(device),
            scale=self.scale.to(device),
            offset=self.offset.to(device),
        )


class ManoHands:
    """Frozen left/right MANO PCA layers, matching the DiT loss convention."""

    def __init__(self, model_dir: Path | None = None, device: torch.device | None = None) -> None:
        self.device = torch.device(device or "cpu")
        left, right = _build_mano_layers(Path(model_dir))
        self.layers = {"left": left.to(self.device).eval(), "right": right.to(self.device).eval()}
        for layer in self.layers.values():
            for p in layer.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def vertices(self, side: str, betas: torch.Tensor, hand_blocks: torch.Tensor) -> torch.Tensor:
        """(M, 24) metric hand state -> (M, 778, 3) vertices in the same frame."""
        m = hand_blocks.shape[0]
        betas_m = betas.view(1, -1).expand(m, -1).contiguous()
        return mano_fk_vertices(self.layers[side], betas_m, hand_blocks)


def to_object_local(vertices: torch.Tensor, obj_blocks: torch.Tensor) -> torch.Tensor:
    """(M, V, 3) anchor-frame points + (M, 9) object pose -> object-local."""
    t_obj = obj_blocks[:, C.OBJ_TRANSL_SLICE]
    r_obj = _rot6d_to_matrix(obj_blocks[:, C.OBJ_ROT6D_SLICE])
    rel = vertices - t_obj[:, None, :]
    return torch.einsum("mji,mpj->mpi", r_obj, rel)


def signed_distance(points_local: torch.Tensor, slot: int, bank: TemplateBank) -> torch.Tensor:
    """(M, V, 3) object-local points -> (M, V) metric signed distance."""
    m, v, _ = points_local.shape
    canonical = (points_local - bank.offset[slot].view(1, 1, 3)) / bank.scale[slot]
    grid = bank.sdf_grid[slot].view(1, 1, 64, 64, 64).expand(m, -1, -1, -1, -1)
    gp = canonical[:, :, [2, 1, 0]].view(m, 1, 1, v, 3)
    sdf = F.grid_sample(grid, gp, mode="bilinear", align_corners=True, padding_mode="border").view(
        m, v
    )
    return sdf * bank.scale[slot]


def contact_heat(
    points_local: torch.Tensor,
    slot: int,
    bank: TemplateBank,
    sigma_m: float = C.SIGMA_M,
    truncate_m: float = C.TRUNCATE_M,
    penetration_tol_m: float = C.PENETRATION_TOL_M,
    area_thresh_m: float = C.AREA_THRESH_M,
    chunk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame contact heat over the object's 1024 template points."""
    template = bank.points_local[slot]
    heats: list[torch.Tensor] = []
    areas: list[torch.Tensor] = []
    two_sigma_sq = 2.0 * sigma_m * sigma_m
    for lo in range(0, points_local.shape[0], chunk):
        block = points_local[lo : lo + chunk]
        m = block.shape[0]
        inside = signed_distance(block, slot, bank) < -penetration_tol_m
        dist = torch.cdist(template.view(1, -1, 3).expand(m, -1, -1), block)
        dist = dist.masked_fill(inside[:, None, :], float("inf"))
        dmin = dist.amin(dim=2)
        heat = torch.exp(-(dmin**2) / two_sigma_sq)
        heat = torch.where(
            dmin <= truncate_m, heat, torch.zeros((), device=heat.device, dtype=heat.dtype)
        )
        heats.append(heat)
        areas.append((dmin <= area_thresh_m).to(heat.dtype).mean(dim=1))
    return (torch.cat(heats, dim=0), torch.cat(areas, dim=0))


def normalized(pdf: np.ndarray) -> np.ndarray:
    """L1-normalize a non-negative vector; all-zero stays all-zero."""
    total = float(pdf.sum())
    return pdf / total if total > 0.0 else pdf


def find_support_furniture(
    goal: np.ndarray,
    furn_present: np.ndarray,
    furn_center: np.ndarray,
    furn_params: np.ndarray,
    furn_shape: np.ndarray,
    furn_yaw: np.ndarray,
    z_tol: float = C.SUPPORT_Z_TOL_M,
    xy_margin: float = C.SUPPORT_XY_MARGIN_M,
) -> tuple[int, float]:
    """Return ``(furniture_index, yaw)`` of the surface the goal sits on."""
    rect_id = C.FURNITURE_SHAPE_TO_ID["rectangle"]
    circ_id = C.FURNITURE_SHAPE_TO_ID["circle"]
    best_idx, best_dz = (-1, float("inf"))
    for k in range(len(furn_present)):
        if furn_present[k] < 0.5:
            continue
        dz = abs(float(furn_center[k, 2]) - float(goal[2]))
        if dz > z_tol:
            continue
        d = goal[:2] - furn_center[k, :2]
        yaw = float(furn_yaw[k])
        cos, sin = (np.cos(yaw), np.sin(yaw))
        shape = int(furn_shape[k])
        if shape == rect_id:
            proj_l = d[0] * cos + d[1] * sin
            proj_w = -d[0] * sin + d[1] * cos
            inside = (
                abs(proj_l) <= furn_params[k, 1] * 0.5 + xy_margin
                and abs(proj_w) <= furn_params[k, 0] * 0.5 + xy_margin
            )
        elif shape == circ_id:
            inside = float(np.hypot(d[0], d[1])) <= furn_params[k, 0] + xy_margin
        else:
            dist = float(np.hypot(d[0], d[1]))
            ang = (np.arctan2(d[1], d[0]) - yaw) % (2.0 * np.pi)
            inside = dist <= furn_params[k, 0] + xy_margin and (
                ang <= furn_params[k, 1] or dist <= xy_margin
            )
        if inside and dz < best_dz:
            best_idx, best_dz = (k, dz)
    if best_idx < 0:
        return (-1, 0.0)
    return (best_idx, float(furn_yaw[best_idx]))


def _wrist_local_series(
    state: np.ndarray,
    hand_valid: np.ndarray,
    obj_valid: np.ndarray,
    hand_sl: slice,
    obj_sl: slice,
    j0: np.ndarray,
) -> np.ndarray:
    """(TRAJ_LEN, 7) object-local wrist [pos(3) | vel(3) | valid(1)]."""
    frames = np.arange(C.PAST_LEN - TRAJ_LEN, C.PAST_LEN)
    out = np.zeros((TRAJ_LEN, 7), dtype=np.float32)
    valid = hand_valid & obj_valid
    pos = np.zeros((TRAJ_LEN, 3), dtype=np.float64)
    ok = np.zeros(TRAJ_LEN, dtype=bool)
    for i, f in enumerate(frames):
        if not valid[f]:
            continue
        hand_t = state[f, hand_sl][C.HAND_TRANSL_SLICE] + j0
        blk = state[f, obj_sl]
        r_obj = (
            _rot6d_to_matrix(torch.from_numpy(blk[C.OBJ_ROT6D_SLICE][None].astype(np.float32)))[0]
            .numpy()
            .astype(np.float64)
        )
        rel = hand_t - blk[C.OBJ_TRANSL_SLICE]
        pos[i] = r_obj.T @ rel
        ok[i] = True
    out[ok, :3] = pos[ok] * POS_SCALE
    both = ok & np.roll(ok, 1)
    both[0] = False
    out[both, 3:6] = (pos[both] - pos[np.roll(both, -1)]) * VEL_SCALE
    out[ok, 6] = 1.0
    return out


class ContactExtractor:
    def __init__(self, bank: TemplateBank, mano_dir: Path, device: torch.device):
        self.device = device
        self.bank = bank.to(device)
        self.mano = ManoHands(mano_dir, device)

    @torch.no_grad()
    def _contact_pdf(
        self,
        state: torch.Tensor,
        betas: torch.Tensor,
        frames: np.ndarray,
        side: str,
        hand_sl: slice,
        obj_sl: slice,
        slot: int,
    ) -> tuple[np.ndarray, int, float]:
        """Mean per-frame contact pdf over ``frames``; mirrors the dictionary."""
        if frames.size == 0:
            return (np.zeros(C.NUM_TEMPLATE_POINTS, np.float64), 0, 0.0)
        idx = torch.from_numpy(frames.astype(np.int64)).to(self.device)
        hand_blocks = state[idx][:, hand_sl]
        obj_blocks = state[idx][:, obj_sl]
        verts = self.mano.vertices(side, betas, hand_blocks)
        local = to_object_local(verts, obj_blocks)
        heat, area = contact_heat(local, slot, self.bank)
        mass = heat.sum(dim=1)
        touching = mass > 0
        n_touch = int(touching.sum())
        if n_touch == 0:
            return (np.zeros(C.NUM_TEMPLATE_POINTS, np.float64), 0, 0.0)
        pdf = heat[touching] / mass[touching][:, None]
        mean_pdf = pdf.mean(dim=0).double().cpu().numpy()
        mean_area = float(area[touching].mean())
        return (normalized(mean_pdf), n_touch, mean_area)
