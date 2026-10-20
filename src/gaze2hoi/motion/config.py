"""Fixed state layout and released network architecture."""

from dataclasses import dataclass

PAST_LEN = 30
FUTURE_LEN = 30
WINDOW_LEN = 60
ANCHOR_OFFSET = 29
HAND_POSE_DIM = 15
HAND_DIM = 24
OBJ_DIM = 9
STATE_DIM = 66
MANO_BETAS_DIM = 10
NUM_OBJECT_SLOTS = 33
NUM_TEMPLATE_POINTS = 1024
MAX_FURNITURE = 6
LEFT_HAND_SLICE = slice(0, 24)
RIGHT_HAND_SLICE = slice(24, 48)
LEFT_OBJ_SLICE = slice(48, 57)
RIGHT_OBJ_SLICE = slice(57, 66)
HAND_TRANSL_SLICE = OBJ_TRANSL_SLICE = slice(0, 3)
HAND_ROT6D_SLICE = OBJ_ROT6D_SLICE = slice(3, 9)
HAND_POSE_SLICE = slice(9, 24)
INTENT_SLOTS = ("action", "object", "part", "deep_action", "target")
NUM_INTENT_SLOTS = 5
FURNITURE_SHAPES = ("rectangle", "circle", "rack_arc_lines")
FURNITURE_SHAPE_TO_ID = {name: index for index, name in enumerate(FURNITURE_SHAPES)}
PLACE_DEEP_ACTION = "place"
SIGMA_M = 0.010
TRUNCATE_M = 0.030
PENETRATION_TOL_M = 0.010
AREA_THRESH_M = 0.020
SUPPORT_Z_TOL_M = 0.03
SUPPORT_XY_MARGIN_M = 0.05
LANDING_GRID_N = 32


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    d_cond: int = 384
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    d_pointnet: int = 256
    d_timestep: int = 256
    n_surface_groups: int = 32


@dataclass(frozen=True)
class ACGConfig:
    d_point: int = 128
    d_global: int = 128
    d_traj: int = 64
    dropout: float = 0.1
    landing_ch: int = 128
