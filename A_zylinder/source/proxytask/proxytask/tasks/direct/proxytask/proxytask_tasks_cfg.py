# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task geometry in metres, relative to each environment origin."""

import math
import os

from isaaclab.utils import configclass


FIXTURE_POS = (0.0, 0.0, 0.700)

PLATE_THICKNESS = 0.055
PLATE_HALF_EXTENTS = (0.250, 0.300)
PLATE_BOTTOM_Z = FIXTURE_POS[2]
PLATE_TOP_Z = FIXTURE_POS[2] + PLATE_THICKNESS

BORE_OFFSET_Y = -0.225
BORE_ENTRANCE_POS = (0.0, BORE_OFFSET_Y, PLATE_TOP_Z)

ROBOT_BASE_OFFSET_Y = 0.190
ROBOT_BASE_POS = (0.0, ROBOT_BASE_OFFSET_Y, PLATE_TOP_Z)

HOME_STANDOFF_Z = 0.150

BASE_TO_PLATE_EDGE = PLATE_HALF_EXTENTS[1] - ROBOT_BASE_OFFSET_Y
BORE_TO_BASE_DISTANCE = ROBOT_BASE_OFFSET_Y - BORE_OFFSET_Y


BORE_DIAMETER = 0.030
BORE_DEPTH = 0.030
SUCCESS_DEPTH = 0.025


PEG_LENGTH = 0.050
PEG_DENSITY = 0.03202 / (math.pi * (0.028 / 2.0) ** 2 * PEG_LENGTH)


def peg_inertials(diameter: float, length: float = PEG_LENGTH, density: float = PEG_DENSITY) -> dict:
    """Return cylinder mass and inertias; USD requires inertia about the centre of mass."""
    radius = diameter / 2.0
    mass = density * math.pi * radius**2 * length
    i_axial = 0.5 * mass * radius**2
    i_trans_com = mass * (3.0 * radius**2 + length**2) / 12.0
    return {
        "mass": mass,
        "inertia_axial": i_axial,
        "inertia_transverse_com": i_trans_com,
        "inertia_transverse_flange": i_trans_com + mass * (length / 2.0) ** 2,
    }


def _diameter_from_env(default_mm: float = 28.0) -> float:
    """Peg diameter [m], overridable per run via PROXYTASK_PEG_DIAMETER_MM."""
    raw = os.environ.get("PROXYTASK_PEG_DIAMETER_MM")
    if raw is None:
        return default_mm / 1000.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"PROXYTASK_PEG_DIAMETER_MM={raw!r} is not a number") from exc
    if not 5.0 <= value < BORE_DIAMETER * 1000.0:
        raise ValueError(
            f"PROXYTASK_PEG_DIAMETER_MM={value} mm is outside (5, {BORE_DIAMETER * 1000.0}) mm; "
            "a peg at or above the bore diameter cannot enter it"
        )
    return value / 1000.0


# Defaults to 28 mm; override with PROXYTASK_PEG_DIAMETER_MM.
PEG_DIAMETER = _diameter_from_env()
PEG_RADIUS = PEG_DIAMETER / 2.0
_PEG_INERTIALS = peg_inertials(PEG_DIAMETER)
PEG_MASS = _PEG_INERTIALS["mass"]
PEG_I_AXIAL = _PEG_INERTIALS["inertia_axial"]
PEG_I_TRANS_COM = _PEG_INERTIALS["inertia_transverse_com"]
PEG_I_TRANS_FLANGE = _PEG_INERTIALS["inertia_transverse_flange"]

RADIAL_CLEARANCE = (BORE_DIAMETER - PEG_DIAMETER) / 2.0

PEG_CENTRE_OFFSET = (0.0, 0.0, PEG_LENGTH / 2.0)
PEG_TIP_OFFSET = (0.0, 0.0, PEG_LENGTH)


def default_peg_robot_usd_path() -> str:
    """Where the authoring script writes, and where the resolver looks last."""
    name = f"ur10e_peg_d{round(PEG_DIAMETER * 1000)}.usd"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Robot", name)


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
    """Geometry and material settings for the fixed bore."""

    usd_path: str = ""
    diameter: float = BORE_DIAMETER
    """Bore diameter [m]."""
    depth: float = BORE_DEPTH
    """Blind-bore depth [m]; the bottom is a physical stop."""
    base_height: float = PLATE_TOP_Z
    """Env-local height of the surface the bore opens into [m]."""
    success_depth: float = SUCCESS_DEPTH
    """Insertion depth counted as success [m] (task specification)."""
    friction: float = 0.75


@configclass
class HeldAssetCfg:
    """Cylinder rigidly attached to the robot flange."""

    usd_path: str = ""
    diameter: float = PEG_DIAMETER
    length: float = PEG_LENGTH
    mass: float = PEG_MASS
    inertia_axial: float = PEG_I_AXIAL
    inertia_transverse_com: float = PEG_I_TRANS_COM
    """About the centre of mass, as USD ``MassAPI:diagonalInertia`` expects."""
    friction: float | None = None
