# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Insertion geometry and physical parameters in SI units."""

import math
import os

from isaaclab.utils import configclass


OPENING_OFFSET_Y = -0.225

FIXTURE_POS = (0.0, OPENING_OFFSET_Y, 0.755)

OPENING_ENTRANCE_POS = FIXTURE_POS

PLATE_HALF_EXTENTS = (0.250, 0.300)
PLATE_TOP_Z = FIXTURE_POS[2]

ROBOT_BASE_OFFSET_Y = 0.190

HOME_STANDOFF_Z = 0.400

BASE_TO_PLATE_EDGE = PLATE_HALF_EXTENTS[1] - ROBOT_BASE_OFFSET_Y
OPENING_TO_BASE_DISTANCE = ROBOT_BASE_OFFSET_Y - OPENING_OFFSET_Y


def _pocket_from_env(default_mm: float = 32.0) -> float:
    """Pocket side length [m], overridable per run via INSERTION_POCKET_SIDE_MM."""
    raw = os.environ.get("INSERTION_POCKET_SIDE_MM")
    if raw is None:
        return default_mm / 1000.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"INSERTION_POCKET_SIDE_MM={raw!r} is not a number") from exc
    if not 30.0 < value <= 60.0:
        raise ValueError(
            f"INSERTION_POCKET_SIDE_MM={value} mm is outside (30, 60] mm: at or below the "
            "30 mm peg nothing can insert, far above it the plate pieces around the pocket vanish"
        )
    return value / 1000.0


POCKET_SIDE = _pocket_from_env()
POCKET_DEPTH = 0.035
SUCCESS_DEPTH = 0.025


PEG_LENGTH = 0.050
PEG_DENSITY = 0.03202 / (math.pi * (0.028 / 2.0) ** 2 * PEG_LENGTH)


def peg_inertials(side: float, length: float = PEG_LENGTH, density: float = PEG_DENSITY) -> dict:
    """Mass and inertias of a solid square prism, in the form the USD needs."""
    mass = density * side * side * length
    i_axial = mass * side * side / 6.0
    i_trans_com = mass * (side * side + length * length) / 12.0
    return {
        "mass": mass,
        "inertia_axial": i_axial,
        "inertia_transverse_com": i_trans_com,
        "inertia_transverse_flange": i_trans_com + mass * (length / 2.0) ** 2,
    }


def _side_from_env(default_mm: float = 30.0) -> float:
    """Peg side length [m], overridable per run via INSERTION_PEG_SIDE_MM."""
    if os.environ.get("INSERTION_PEG_DIAMETER_MM") is not None:
        raise ValueError(
            "INSERTION_PEG_DIAMETER_MM is set, but this branch (square-peg-insertion) "
            "has no cylindrical peg. Unset it and use INSERTION_PEG_SIDE_MM instead."
        )
    raw = os.environ.get("INSERTION_PEG_SIDE_MM")
    if raw is None:
        return default_mm / 1000.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"INSERTION_PEG_SIDE_MM={raw!r} is not a number") from exc
    if not 5.0 <= value < POCKET_SIDE * 1000.0:
        raise ValueError(
            f"INSERTION_PEG_SIDE_MM={value} mm is outside (5, {POCKET_SIDE * 1000.0}) mm; "
            "a peg at or above the pocket width cannot enter it"
        )
    return value / 1000.0


PEG_SIDE = _side_from_env()
_PEG_INERTIALS = peg_inertials(PEG_SIDE)
PEG_MASS = _PEG_INERTIALS["mass"]
PEG_I_AXIAL = _PEG_INERTIALS["inertia_axial"]
PEG_I_TRANS_COM = _PEG_INERTIALS["inertia_transverse_com"]

SIDE_CLEARANCE = (POCKET_SIDE - PEG_SIDE) / 2.0

if SIDE_CLEARANCE <= 0.0:
    raise ValueError(
        f"peg side {PEG_SIDE * 1000.0:g} mm does not fit pocket side "
        f"{POCKET_SIDE * 1000.0:g} mm (clearance {SIDE_CLEARANCE * 1000.0:g} mm per axis). "
        "Check INSERTION_PEG_SIDE_MM and INSERTION_POCKET_SIDE_MM against each other."
    )

_PEG_DIAGONAL = PEG_SIDE * math.sqrt(2.0)
YAW_WINDOW_RAD = (
    math.pi / 4.0
    if _PEG_DIAGONAL <= POCKET_SIDE
    else math.asin(POCKET_SIDE / _PEG_DIAGONAL) - math.pi / 4.0
)


def default_peg_robot_usd_path() -> str:
    """Where the authoring script writes, and where the resolver looks last."""
    name = f"ur10e_peg_s{PEG_SIDE * 1000.0:g}.usd"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Robot", name)


def default_fixture_usd_name(side: float | None = None) -> str:
    """Filename of the generated table for a pocket size, ``b`` for the bore."""
    return f"tisch_square_b{(POCKET_SIDE if side is None else side) * 1000.0:g}.usd"


def resolve_peg_robot_usd_path() -> str:
    """Return the path to the peg-welded UR10e USD, or raise with a hint."""
    env_override = os.environ.get("INSERTION_UR10E_PEG_USD")
    candidates = []
    if env_override:
        candidates.append(env_override)
    candidates.append(default_peg_robot_usd_path())
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Peg-welded UR10e USD not found. Tried: "
        + "; ".join(candidates)
        + ". Run scripts/author_peg_ur10e.py on the training machine, "
        "or set INSERTION_UR10E_PEG_USD."
    )


TOOL_USD_NAME = "greifer_bauteil_asm.usd"
TOOL_ROBOT_USD_NAME = "ur5e_tool.usd"


def default_tool_usd_path() -> str:
    """Where the CAD import of gripper + part lives (RT-8 .. RT-12)."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "Werkzeug", TOOL_USD_NAME
    )


def default_tool_robot_usd_path() -> str:
    """Where the tool-welded UR5e is written, and looked for last."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "Robot", TOOL_ROBOT_USD_NAME
    )


def resolve_tool_usd_path() -> str:
    """Path to the imported gripper+part assembly, or raise with a hint."""
    env_override = os.environ.get("INSERTION_TOOL_USD")
    candidates = [p for p in (env_override, default_tool_usd_path()) if p]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Tool assembly USD not found. Tried: "
        + "; ".join(candidates)
        + ". Import CAD/Step/Greifer_Bauteil/greifer_bauteil_asm.stp in Isaac Sim "
        "and repair the units, or set INSERTION_TOOL_USD."
    )


POCKET_USD_NAME = "Aufnahme_real_v1_mm.usd"


def default_pocket_usd_path() -> str:
    """Where the imported fixture cut-out lives (RT-27 .. RT-36)."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "Werkzeug", POCKET_USD_NAME
    )


def resolve_pocket_usd_path() -> str:
    """Path to the real fixture cut-out USD, or raise with a hint."""
    env_override = os.environ.get("INSERTION_POCKET_USD")
    candidates = [p for p in (env_override, default_pocket_usd_path()) if p]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Fixture cut-out USD not found. Tried: "
        + "; ".join(candidates)
        + ". Convert CAD/Aufnahme_real_v1_mm.stp (NOT Aufnahme_real_v1.stp -- it "
        "declares INCH), repair it with scripts/fix_stage_units.py and "
        "scripts/fix_fixture_asset.py, give it a collider with "
        "scripts/add_fixture_collision.py, or set INSERTION_POCKET_USD."
    )


def resolve_tool_robot_usd_path() -> str:
    """Path to the tool-welded UR5e USD, or raise with a hint."""
    env_override = os.environ.get("INSERTION_UR5E_TOOL_USD")
    candidates = [p for p in (env_override, default_tool_robot_usd_path()) if p]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Tool-welded UR5e USD not found. Tried: "
        + "; ".join(candidates)
        + ". Run scripts/author_tool_ur5e.py on the training machine, "
        "or set INSERTION_UR5E_TOOL_USD."
    )


from .insertion_paths import (  # noqa: F401 - re-export, one home is insertion_paths
    PART_OBJ_NAME,
    POCKET_OBJ_NAME,
    default_part_obj_path,
    default_pocket_obj_path,
    resolve_part_obj_path,
    resolve_pocket_obj_path,
)


@configclass
class FixedAssetCfg:
    """The world-anchored asset carrying the pocket: the generated Tisch."""

    usd_path: str = ""
    side: float = POCKET_SIDE
    depth: float = POCKET_DEPTH
    """Blind-pocket depth [m]; the bottom is a physical stop.

    The DEFAULT is the proxy's. The env overrides it with the real fixture's
    POCKET_SEAT_DEPTH at the one place this class is instantiated
    (``insertion_env_cfg.InsertionEnvCfg.fixed_asset``). The default cannot be
    derived here: POCKET_SEAT_DEPTH is defined further down this module,
    after the measured inputs it comes from.
    """
    base_height: float = PLATE_TOP_Z
    """Env-local height of the opening plane the pocket is cut into [m]."""
    success_depth: float = SUCCESS_DEPTH
    """Insertion depth counted as success [m].

    The DEFAULT is the proxy's 25 mm. The env overrides it with
    SEATED_SUCCESS_DEPTH at the one place this class is
    instantiated, for the same ordering reason as ``depth`` above.
    """
    friction: float = 0.75


@configclass
class HeldAssetCfg:
    """The peg."""

    usd_path: str = ""
    side: float = PEG_SIDE
    """Square cross-section edge length [m]."""
    length: float = PEG_LENGTH
    mass: float = PEG_MASS
    inertia_axial: float = PEG_I_AXIAL
    inertia_transverse_com: float = PEG_I_TRANS_COM
    """About the centre of mass, as USD ``MassAPI:diagonalInertia`` expects."""
    friction: float | None = None


WORKCELL_PENDING: tuple[str, ...] = ()
"""Named unknowns. The scene builds and is internally consistent without them,
but is NOT absolutely placed until they are filled in. The startup report
prints this tuple so a run can never quietly look finished."""

ROBOT_BASE_SQUARE = 0.153
MOUNT_PLATE_T = 0.024

T1_SIZE_X, T1_SIZE_Y = 1.000, 0.600
T2_SIZE_X, T2_SIZE_Y = 0.750, 1.505
TABLE_PLATE_T = 0.020

PLATE_TOP_DROP = 0.0958
TABLE_GAP_X = 0.0067

BASE_EDGE_TO_T1_FRONT = 0.104
BASE_EDGE_TO_T1_RIGHT = 0.245

T2_OVERHANG_RIGHT = 0.447
T2_OVERHANG_LEFT = 0.443

WORKCELL_Y_SIGN = +1.0

RECT_SIZE_X, RECT_SIZE_Y = 0.400, 0.880
RECT_TO_T2_REAR = 0.054
RECT_TO_T2_FRONT = 0.296
RECT_TO_T2_LEFT = 0.163
RECT_TO_T2_RIGHT = 0.4615

BLOCK_SIZE_X, BLOCK_SIZE_Y = 0.210, 0.260
BLOCK_NEAR_EDGE_FROM_RECT = 0.105
GUIDE_EDGE_ABOVE_T2 = 0.1973
STAGE1_DEPTH = 0.015

POCKET_WALL_X = 0.0452938
POCKET_WALL_Y = 0.07255
POCKET_FLOOR_Z = -0.036
POCKET_ASSET_BOTTOM_Z = -0.046
POCKET_LOCAL_X_RANGE = (-0.0847938, 0.0652062)
POCKET_LOCAL_Y_RANGE = (-0.08865, 0.10045)

POCKET_STAGE1_X_RANGE = (-0.0747938, 0.0552938)
POCKET_STAGE1_Y_RANGE = (-0.07255, 0.08045)
POCKET_NOTCH_X = 0.0470876
POCKET_NOTCH_Y_RANGE = (-0.07865, 0.08655)

POCKET_BODY_Y_MAX = 0.09045
POCKET_TAB_X_RANGE = (-0.0327938, 0.0342062)

BLOCK_FRONT_TO_STAGE1 = 0.080
BLOCK_REAR_WALL_T = 0.015
BLOCK_REAR_WALL_H = 0.100
BLOCK_PLUSY_TO_STAGE1 = 0.050
BLOCK_MINUSY_TO_STAGE1_B12 = 0.058
BLOCK_MINUSY_TO_STAGE1 = 0.057


T1_TOP_Z = -MOUNT_PLATE_T
T2_TOP_Z = T1_TOP_Z - PLATE_TOP_DROP
BLOCK_TOP_Z = T2_TOP_Z + GUIDE_EDGE_ABOVE_T2

T1_NEAR_EDGE_X = BASE_EDGE_TO_T1_FRONT + ROBOT_BASE_SQUARE / 2.0
T2_NEAR_EDGE_X = T1_NEAR_EDGE_X + TABLE_GAP_X
RECT_NEAR_EDGE_X = T2_NEAR_EDGE_X + RECT_TO_T2_REAR
BLOCK_NEAR_EDGE_X = RECT_NEAR_EDGE_X + BLOCK_NEAR_EDGE_FROM_RECT
BLOCK_CENTRE_X = BLOCK_NEAR_EDGE_X + BLOCK_SIZE_X / 2.0

T1_CENTRE_X = T1_NEAR_EDGE_X - T1_SIZE_X / 2.0
T2_CENTRE_X = T2_NEAR_EDGE_X + T2_SIZE_X / 2.0

_T1_RIGHT_EDGE_Y = WORKCELL_Y_SIGN * (BASE_EDGE_TO_T1_RIGHT + ROBOT_BASE_SQUARE / 2.0)
T1_CENTRE_Y = _T1_RIGHT_EDGE_Y - WORKCELL_Y_SIGN * T1_SIZE_Y / 2.0
_T2_RIGHT_EDGE_Y = _T1_RIGHT_EDGE_Y + WORKCELL_Y_SIGN * T2_OVERHANG_RIGHT
T2_CENTRE_Y = _T2_RIGHT_EDGE_Y - WORKCELL_Y_SIGN * T2_SIZE_Y / 2.0
RECT_CENTRE_Y = _T2_RIGHT_EDGE_Y - WORKCELL_Y_SIGN * (RECT_TO_T2_RIGHT + RECT_SIZE_Y / 2.0)

BLOCK_CENTRE_Y = RECT_CENTRE_Y

BLOCK_HEIGHT = GUIDE_EDGE_ABOVE_T2

WORKCELL_Y_RESIDUAL = (T2_SIZE_Y - T2_OVERHANG_RIGHT - T1_SIZE_Y) - T2_OVERHANG_LEFT

TABLE1_POS = (T1_CENTRE_X, T1_CENTRE_Y, T1_TOP_Z - TABLE_PLATE_T / 2.0)
TABLE2_POS = (T2_CENTRE_X, T2_CENTRE_Y, T2_TOP_Z - TABLE_PLATE_T / 2.0)
BLOCK_POS = (BLOCK_CENTRE_X, BLOCK_CENTRE_Y, BLOCK_TOP_Z - BLOCK_HEIGHT / 2.0)

POCKET_RIM_Z = STAGE1_DEPTH
POCKET_LOCAL_Z_RANGE = (POCKET_ASSET_BOTTOM_Z, POCKET_RIM_Z)

POCKET_SEAT_DEPTH = -POCKET_FLOOR_Z

SUCCESS_DEPTH_BAND = 0.003

SEATED_SUCCESS_DEPTH = POCKET_SEAT_DEPTH - SUCCESS_DEPTH_BAND

POCKET_OPENING_X = 2.0 * POCKET_WALL_X
POCKET_OPENING_Y = 2.0 * POCKET_WALL_Y

POCKET_STAGE1_SIZE_X = POCKET_STAGE1_X_RANGE[1] - POCKET_STAGE1_X_RANGE[0]
POCKET_STAGE1_SIZE_Y = POCKET_STAGE1_Y_RANGE[1] - POCKET_STAGE1_Y_RANGE[0]

POCKET_ASSET_BBOX_M = (
    POCKET_LOCAL_X_RANGE[1] - POCKET_LOCAL_X_RANGE[0],
    POCKET_LOCAL_Y_RANGE[1] - POCKET_LOCAL_Y_RANGE[0],
    POCKET_LOCAL_Z_RANGE[1] - POCKET_LOCAL_Z_RANGE[0],
)

POCKET_ORIGIN_X = (
    BLOCK_NEAR_EDGE_X + BLOCK_FRONT_TO_STAGE1 - POCKET_STAGE1_X_RANGE[0]
)
POCKET_ORIGIN_Y = (
    BLOCK_CENTRE_Y + BLOCK_SIZE_Y / 2.0
    - BLOCK_PLUSY_TO_STAGE1 - POCKET_STAGE1_Y_RANGE[1]
)

BLOCK_MODEL_SIZE_X = BLOCK_FRONT_TO_STAGE1 + POCKET_STAGE1_SIZE_X + BLOCK_REAR_WALL_T

BLOCK_BEHIND_POCKET_X = (
    BLOCK_MODEL_SIZE_X - BLOCK_FRONT_TO_STAGE1
    - (POCKET_LOCAL_X_RANGE[1] - POCKET_STAGE1_X_RANGE[0])
)

WORKCELL_POCKET_Y_RESIDUAL = (
    BLOCK_SIZE_Y - BLOCK_PLUSY_TO_STAGE1 - POCKET_STAGE1_SIZE_Y
    - BLOCK_MINUSY_TO_STAGE1
)

WORKCELL_POCKET_MINUSY_VS_B12 = BLOCK_MINUSY_TO_STAGE1_B12 - BLOCK_MINUSY_TO_STAGE1


def _box_from_ranges(name: str, x_range, y_range, z_range) -> dict:
    """(min, max) per axis -> the {name, centre, size} the authoring script wants."""
    return {
        "name": name,
        "centre": tuple((lo + hi) / 2.0 for lo, hi in (x_range, y_range, z_range)),
        "size": tuple(hi - lo for lo, hi in (x_range, y_range, z_range)),
    }


def block_recess_boxes() -> list[dict]:
    """The block AROUND the CAD cut-out, asset-local: eight boxes."""
    rx0, rx1 = POCKET_LOCAL_X_RANGE
    ry0, ry1 = POCKET_LOCAL_Y_RANGE
    tab_x0, tab_x1 = POCKET_TAB_X_RANGE
    x_front = BLOCK_NEAR_EDGE_X - POCKET_ORIGIN_X
    x_rear = x_front + BLOCK_MODEL_SIZE_X
    y_lo = (BLOCK_CENTRE_Y - BLOCK_SIZE_Y / 2.0) - POCKET_ORIGIN_Y
    y_hi = (BLOCK_CENTRE_Y + BLOCK_SIZE_Y / 2.0) - POCKET_ORIGIN_Y
    top = STAGE1_DEPTH
    bottom = STAGE1_DEPTH - BLOCK_HEIGHT
    full_z = (bottom, top)
    slot_z = (POCKET_ASSET_BOTTOM_Z, top)
    return [
        _box_from_ranges("block_wand_x_minus", (x_front, rx0), (y_lo, y_hi), full_z),
        _box_from_ranges("block_wand_x_plus", (rx1, x_rear), (y_lo, y_hi), full_z),
        _box_from_ranges("block_wand_y_minus", (rx0, rx1), (y_lo, ry0), full_z),
        _box_from_ranges("block_wand_y_plus", (rx0, rx1), (ry1, y_hi), full_z),
        _box_from_ranges("block_boden", (rx0, rx1), (ry0, ry1), (bottom, POCKET_ASSET_BOTTOM_Z)),
        _box_from_ranges(
            "block_spalt_y_plus_minus_x", (rx0, tab_x0), (POCKET_BODY_Y_MAX, ry1), slot_z),
        _box_from_ranges(
            "block_spalt_y_plus_plus_x", (tab_x1, rx1), (POCKET_BODY_Y_MAX, ry1), slot_z),
        _box_from_ranges(
            "block_rueckwand",
            (POCKET_STAGE1_X_RANGE[1], x_rear), (y_lo, y_hi),
            (top, top + BLOCK_REAR_WALL_H),
        ),
    ]


WORKCELL_TABLES_USD_NAME = "arbeitszelle_tische_v2.usd"
WORKCELL_BLOCK_USD_NAME = "aufnahme_block_v4.usd"

INSERTION_PLANE_Z = BLOCK_TOP_Z - STAGE1_DEPTH

WORKCELL_HOME_TIP_ABOVE_ENTRANCE = HOME_STANDOFF_Z + STAGE1_DEPTH

WORKCELL_TABLES_POS = (0.0, 0.0, 0.0)
WORKCELL_BLOCK_POS = (POCKET_ORIGIN_X, POCKET_ORIGIN_Y, INSERTION_PLANE_Z)


def default_workcell_usd_path(name: str) -> str:
    """Where ``scripts/author_workcell.py`` writes, and the resolver looks last."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Arbeitszelle", name)


def resolve_workcell_usd_path(name: str, env_var: str) -> str:
    """Resolve a workcell asset at SPAWN time, never at module import."""
    candidates = []
    override = os.environ.get(env_var)
    if override:
        candidates.append(override)
    candidates.append(default_workcell_usd_path(name))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Workcell asset {name} not found. Tried: " + "; ".join(candidates)
        + ". Run scripts/author_workcell.py on the training machine, "
        f"or set {env_var}."
    )


def resolve_workcell_tables_usd_path() -> str:
    return resolve_workcell_usd_path(WORKCELL_TABLES_USD_NAME, "INSERTION_WORKCELL_TABLES_USD")


def resolve_workcell_block_usd_path() -> str:
    return resolve_workcell_usd_path(WORKCELL_BLOCK_USD_NAME, "INSERTION_WORKCELL_BLOCK_USD")


T1_HEIGHT_ABOVE_FLOOR = 0.884
FLOOR_Z = T1_TOP_Z - T1_HEIGHT_ABOVE_FLOOR

WORKCELL_PLATE_TOP_Z = T2_TOP_Z
WORKCELL_PLATE_BOTTOM_Z = T2_TOP_Z - TABLE_PLATE_T
WORKCELL_PLATE_HALF_EXTENTS = (T2_SIZE_X / 2.0, T2_SIZE_Y / 2.0)

WORKCELL_ENTRANCE_POS = WORKCELL_BLOCK_POS
WORKCELL_ROBOT_BASE_POS = (0.0, 0.0, 0.0)


FLANGE_TO_SUCTION_TIP_FREE = 0.1486
FLANGE_TO_SUCTION_FACE = 0.1420
FLANGE_TO_PART_BOTTOM = 0.1520

BELLOWS_COMPRESSION = FLANGE_TO_SUCTION_TIP_FREE - FLANGE_TO_SUCTION_FACE
PART_FLOOR_T_AT_CUP = FLANGE_TO_PART_BOTTOM - FLANGE_TO_SUCTION_FACE

SEATED_TOOL_QUAT_LOCAL = (0.0, 0.0, 1.0, 0.0)

FLANGE_TO_PART_CENTRE = 0.126747

PART_WELD_ROT_COLUMNS = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
GRIPPER_WELD_ROT_COLUMNS = ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))


PART_MARGIN_END_TOOL_FRAME = "+Y"

PART_HEIGHT_Z = 2.0 * (FLANGE_TO_PART_BOTTOM - FLANGE_TO_PART_CENTRE)

PART_BODY_X = 0.0900
"""Part body width across the short axis [m], between the lugs (C: part CAD).

The bounding box PART_BBOX_M[0] is WIDER (lugs included) and is a different
fact; the lugs enter the pocket's wall notches and never touch the opening."""


GRASP_POS_NOISE: tuple[float, float, float] | None = None

CONTACT_FRICTION = 0.14

TOOL_LINK_NAME = "tool_link"
TOOL_JOINT_NAME = "tool_weld"
TOOL_REF_NAME = "tool"
TOOL_PART_PRIM_NAME = "FUEGETEIL"
TOOL_GRIPPER_PRIM_NAME = "GREIFER_V1"

TOOL_WELD_YAW_RAD = 0.0

TOOL_HOME_YAW_RAD = math.pi

PART_CENTROID_FLANGE = (-0.001363926, -0.000587380, 0.128521115)
GRIPPER_CENTROID_FLANGE = (-0.000001749, -0.000059183, 0.055313338)


def tool_com_flange(part_mass: float, gripper_mass: float) -> tuple[float, float, float]:
    """Mass-weighted centre of mass of the welded tool, in the flange frame."""
    total = float(part_mass) + float(gripper_mass)
    if total <= 0.0:
        raise ValueError(f"tool mass must be positive, got {part_mass} + {gripper_mass}")
    return tuple(
        (float(part_mass) * PART_CENTROID_FLANGE[i]
         + float(gripper_mass) * GRIPPER_CENTROID_FLANGE[i]) / total
        for i in range(3)
    )


PART_BBOX_M = (0.096409, 0.143500, 0.050514)

PLAY_X = POCKET_OPENING_X - PART_BODY_X
PLAY_Y = POCKET_OPENING_Y - PART_BBOX_M[1]

POCKET_HOLE_MIN_Y = 0.0796347
HOLE_REACH_OFFSET_Y = POCKET_HOLE_MIN_Y - PART_BBOX_M[1] / 2.0


INTERPEN_THRESH = 0.5 * PLAY_X

KERNEL_MARGIN_COARSE = 0.17321
KERNEL_A_COARSE = math.acosh(10.0) / KERNEL_MARGIN_COARSE
KERNEL_A_MID = math.acosh(10.0) / SUCCESS_DEPTH_BAND
KERNEL_A_FINE = math.acosh(10.0) / PLAY_X
KERNEL_B_COARSE = 2.0
KERNEL_B_MID = 2.0
KERNEL_B_FINE = 0.0

TILT_MARGIN_RAD = math.radians(15.34)
KERNEL_A_TILT = math.acosh(10.0) / TILT_MARGIN_RAD

RUNG0_START_TIP_ABOVE_ENTRANCE = 0.020

ENGAGED_DEPTH_FRACTION = 0.30
ENGAGED_DEPTH = ENGAGED_DEPTH_FRACTION * POCKET_SEAT_DEPTH


TOOL_SPAN_LOCAL_Z_M = (0.0, 0.152004)

TOOL_CHAIN_PENDING: tuple[str, ...] = ("GRASP_POS_NOISE", "TOOL_DIAGONAL_INERTIA")
"""Tool-chain assumptions reported at startup. Grasp is deterministic until
GRASP_POS_NOISE is set. PhysX derives inertia from part-only collision
geometry: gripper mass/COM are included, but its inertia is approximate.
Check orientation using geometric features, not an invariant bounding box
(RT-21). The cylindrical lug span is 96.410 mm in STEP vs 96.409 mm in USD;
vertices alone do not capture its mid-arc extremum.
"""
