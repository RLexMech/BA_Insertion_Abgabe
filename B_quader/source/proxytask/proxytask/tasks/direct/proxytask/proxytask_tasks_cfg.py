# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Square-peg and pocket geometry in metres, relative to each environment origin."""

import math
import os

from isaaclab.utils import configclass


OPENING_OFFSET_Y = -0.225

FIXTURE_POS = (0.0, OPENING_OFFSET_Y, 0.755)

OPENING_ENTRANCE_POS = FIXTURE_POS

PLATE_THICKNESS = 0.055
PLATE_HALF_EXTENTS = (0.250, 0.300)
PLATE_TOP_Z = FIXTURE_POS[2]
PLATE_BOTTOM_Z = PLATE_TOP_Z - PLATE_THICKNESS

ROBOT_BASE_OFFSET_Y = 0.190
ROBOT_BASE_POS = (0.0, ROBOT_BASE_OFFSET_Y, PLATE_TOP_Z)

HOME_STANDOFF_Z = 0.150

BASE_TO_PLATE_EDGE = PLATE_HALF_EXTENTS[1] - ROBOT_BASE_OFFSET_Y
OPENING_TO_BASE_DISTANCE = ROBOT_BASE_OFFSET_Y - OPENING_OFFSET_Y


def _pocket_from_env(default_mm: float = 32.0) -> float:
    """Pocket side length [m], overridable per run via PROXYTASK_POCKET_SIDE_MM."""
    raw = os.environ.get("PROXYTASK_POCKET_SIDE_MM")
    if raw is None:
        return default_mm / 1000.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"PROXYTASK_POCKET_SIDE_MM={raw!r} is not a number") from exc
    if not 30.0 < value <= 60.0:
        raise ValueError(
            f"PROXYTASK_POCKET_SIDE_MM={value} mm is outside (30, 60] mm: at or below the "
            "30 mm peg nothing can insert, far above it the plate pieces around the pocket vanish"
        )
    return value / 1000.0


# Default: 32 mm; override with PROXYTASK_POCKET_SIDE_MM.
POCKET_SIDE = _pocket_from_env()
"""Square pocket edge length [m] -- the curriculum variable (D-033)."""
POCKET_DEPTH = 0.035
SUCCESS_DEPTH = 0.025


PEG_LENGTH = 0.050
PEG_DENSITY = 0.03202 / (math.pi * (0.028 / 2.0) ** 2 * PEG_LENGTH)


def peg_inertials(side: float, length: float = PEG_LENGTH, density: float = PEG_DENSITY) -> dict:
    """Return square-prism mass and inertias; USD requires inertia about the centre of mass."""
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
    """Peg side length [m], overridable per run via PROXYTASK_PEG_SIDE_MM."""
    if os.environ.get("PROXYTASK_PEG_DIAMETER_MM") is not None:
        raise ValueError(
            "PROXYTASK_PEG_DIAMETER_MM is set, but this branch (square-peg-insertion) "
            "has no cylindrical peg. Unset it and use PROXYTASK_PEG_SIDE_MM instead."
        )
    raw = os.environ.get("PROXYTASK_PEG_SIDE_MM")
    if raw is None:
        return default_mm / 1000.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"PROXYTASK_PEG_SIDE_MM={raw!r} is not a number") from exc
    if not 5.0 <= value < POCKET_SIDE * 1000.0:
        raise ValueError(
            f"PROXYTASK_PEG_SIDE_MM={value} mm is outside (5, {POCKET_SIDE * 1000.0}) mm; "
            "a peg at or above the pocket width cannot enter it"
        )
    return value / 1000.0


# Default: 30 mm; override with PROXYTASK_PEG_SIDE_MM.
PEG_SIDE = _side_from_env()
_PEG_INERTIALS = peg_inertials(PEG_SIDE)
PEG_MASS = _PEG_INERTIALS["mass"]
PEG_I_AXIAL = _PEG_INERTIALS["inertia_axial"]
PEG_I_TRANS_COM = _PEG_INERTIALS["inertia_transverse_com"]
PEG_I_TRANS_FLANGE = _PEG_INERTIALS["inertia_transverse_flange"]

SIDE_CLEARANCE = (POCKET_SIDE - PEG_SIDE) / 2.0

if SIDE_CLEARANCE <= 0.0:
    raise ValueError(
        f"peg side {PEG_SIDE * 1000.0:g} mm does not fit pocket side "
        f"{POCKET_SIDE * 1000.0:g} mm (clearance {SIDE_CLEARANCE * 1000.0:g} mm per axis). "
        "Check PROXYTASK_PEG_SIDE_MM and PROXYTASK_POCKET_SIDE_MM against each other."
    )

_PEG_DIAGONAL = PEG_SIDE * math.sqrt(2.0)
YAW_WINDOW_RAD = (
    math.pi / 4.0
    if _PEG_DIAGONAL <= POCKET_SIDE
    else math.asin(POCKET_SIDE / _PEG_DIAGONAL) - math.pi / 4.0
)

PEG_CENTRE_OFFSET = (0.0, 0.0, PEG_LENGTH / 2.0)
PEG_TIP_OFFSET = (0.0, 0.0, PEG_LENGTH)


def default_peg_robot_usd_path() -> str:
    """Where the authoring script writes, and where the resolver looks last."""
    name = f"ur10e_peg_s{PEG_SIDE * 1000.0:g}.usd"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Robot", name)


def default_fixture_usd_name(side: float | None = None) -> str:
    """Filename of the generated table for a pocket size, ``b`` for the bore."""
    return f"tisch_square_b{(POCKET_SIDE if side is None else side) * 1000.0:g}.usd"


def resolve_peg_robot_usd_path() -> str:
    """Return the path to the peg-welded UR10e USD, or raise with a hint."""
    env_override = os.environ.get("PROXYTASK_UR10E_PEG_USD")
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
        "or set PROXYTASK_UR10E_PEG_USD."
    )


@configclass
class FixedAssetCfg:
    """Geometry and material settings for the fixed pocket."""

    usd_path: str = ""
    side: float = POCKET_SIDE
    """Square pocket edge length [m]."""
    depth: float = POCKET_DEPTH
    """Blind-pocket depth [m]; the bottom is a physical stop."""
    base_height: float = PLATE_TOP_Z
    """Env-local height of the opening plane the pocket is cut into [m]."""
    success_depth: float = SUCCESS_DEPTH
    """Insertion depth counted as success [m] (task specification)."""
    friction: float = 0.75


@configclass
class HeldAssetCfg:
    """Square peg rigidly attached to the robot flange."""

    usd_path: str = ""
    side: float = PEG_SIDE
    """Square cross-section edge length [m]."""
    length: float = PEG_LENGTH
    mass: float = PEG_MASS
    inertia_axial: float = PEG_I_AXIAL
    inertia_transverse_com: float = PEG_I_TRANS_COM
    """About the centre of mass, as USD ``MassAPI:diagonalInertia`` expects."""
    friction: float | None = None
