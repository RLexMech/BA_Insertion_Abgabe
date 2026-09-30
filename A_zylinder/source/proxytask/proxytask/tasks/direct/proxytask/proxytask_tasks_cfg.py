# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task geometry and asset configuration (D-022, D-023, D-018/D-019).

Renamed from ``task_geometry.py`` in the peg increment, following Factory's
``factory_tasks_cfg.py`` split (docs/factory_mapping.md: adopt the structure,
keep our own values). The plain constants are kept alongside the config classes
because they are what the startup report and the offline tools read; the config
classes carry the fixed/held asset split that the peg introduces.

All positions are relative to the **environment origin**, never absolute world
coordinates: Isaac Lab offsets each parallel environment by
``scene.env_origins``, so treating them as absolute would make the bore pose
correct only for environment 0.

Values are derived rather than repeated wherever one follows from another. The
robot base height *is* the plate top, and the radial clearance *is* half the
diameter difference; before this, 0.755 appeared in three places, and changing
the fixture height would have left the robot floating -- the same class of
silent mismatch that cost two debugging rounds on 2026-07-26.
"""

import math
import os

from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Fixture, plate and robot placement (D-022, D-023; verified 2026-07-26)
# ---------------------------------------------------------------------------

# Fixture spawn. The asset origin sits at the plate underside, so the legs reach
# from here down to the ground plane (verified: bbox z range -0.700 .. +0.055 m).
FIXTURE_POS = (0.0, 0.0, 0.700)

# Plate, read from the Creo source and confirmed against the asset bbox.
PLATE_THICKNESS = 0.055
PLATE_HALF_EXTENTS = (0.250, 0.300)
PLATE_BOTTOM_Z = FIXTURE_POS[2]
PLATE_TOP_Z = FIXTURE_POS[2] + PLATE_THICKNESS  # 0.755

# Bore entrance = task frame origin (D-022). On the -Y half of the plate.
BORE_OFFSET_Y = -0.225
BORE_ENTRANCE_POS = (0.0, BORE_OFFSET_Y, PLATE_TOP_Z)

# UR10e base (D-023): on the plate, on the x axis, 415 mm from the bore along +Y.
ROBOT_BASE_OFFSET_Y = 0.190
ROBOT_BASE_POS = (0.0, ROBOT_BASE_OFFSET_Y, PLATE_TOP_Z)

# Flange standoff above the plate top at the home pose.
HOME_STANDOFF_Z = 0.150

# Derived, for reporting rather than for use: distance from the base centre to
# the near plate edge. The UR10e base cylinder is nominally 190 mm across
# (published figure, not measured from the USD), so roughly 15 mm of rim
# clearance remains.
BASE_TO_PLATE_EDGE = PLATE_HALF_EXTENTS[1] - ROBOT_BASE_OFFSET_Y  # 0.110 m
BORE_TO_BASE_DISTANCE = ROBOT_BASE_OFFSET_Y - BORE_OFFSET_Y  # 0.415 m

# ---------------------------------------------------------------------------
# Bore (fixed asset) -- docs/task_specification.md, confirmed against the Creo
# source. The bore is blind: its bottom is a physical stop 5 mm beyond the
# success threshold, so the peg cannot fall through.
# ---------------------------------------------------------------------------

BORE_DIAMETER = 0.030
BORE_DEPTH = 0.030
SUCCESS_DEPTH = 0.025

# ---------------------------------------------------------------------------
# Peg (held asset), as a curriculum rung rather than a fixed geometry.
#
# The diameter is the ONE number that sets a rung, and everything else follows
# from it. It was four hand-maintained constants until 2026-07-27; changing a
# rung meant recomputing mass and three inertias consistently, and getting one
# wrong would have produced a physically wrong peg that every self-consistency
# check in the project would have passed. Deriving them removes that failure
# mode, and makes a rung change a single number.
#
# Rungs (hole fixed at 30 mm; D-006 stage 1 is the task specification's
# easiest rung, D-032 added stage 0 below it, the demo sprint added 25 mm):
#
#     28 mm -> 1.0 mm radial clearance   (D-006 stage 1, the specified task)
#     25 mm -> 2.5 mm                    (demo sprint, first policy that
#                                         inserted, tag demo-insertion-working)
#     24 mm -> 3.0 mm                    (D-032 stage 0, increment 3)
#
# Set it per run without editing this file:
#
#     $env:PROXYTASK_PEG_DIAMETER_MM = "28"
#
# The USD filename follows the diameter too, so rungs cannot overwrite each
# other's assets, and the startup report's asset check compares the authored
# mesh against these values -- a mismatch between a rung and the USD on disk
# is reported, not assumed away.
#
# Density is the ABS value the task specification implies: back-solved from
# its 0.03202 kg at 28 mm and confirmed consistent to three digits there.
# USD ``MassAPI:diagonalInertia`` is specified **about the centre of mass**, so
# the authoring script must use PEG_I_TRANS_COM. PEG_I_TRANS_FLANGE is carried
# only as the cross-check that the two agree via Steiner's theorem; using it as
# the diagonal inertia would overstate the peg's transverse inertia 3.4-fold.
# ---------------------------------------------------------------------------

PEG_LENGTH = 0.050
PEG_DENSITY = 0.03202 / (math.pi * (0.028 / 2.0) ** 2 * PEG_LENGTH)  # ~1040 kg/m^3, ABS


def peg_inertials(diameter: float, length: float = PEG_LENGTH, density: float = PEG_DENSITY) -> dict:
    """Mass and inertias of a solid cylinder, in the form the USD needs.

    ``inertia_transverse_com`` is about the centre of mass, which is what
    ``MassAPI:diagonalInertia`` means; ``inertia_transverse_flange`` is the
    Steiner-shifted value the authoring script asserts against, and is not
    written into the asset.
    """
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
    """Peg diameter [m], overridable per run via PROXYTASK_PEG_DIAMETER_MM.

    Default is the task specification's stage 1. A malformed value is a hard
    error rather than a silent fallback: training a rung other than the one
    intended would invalidate the comparison it exists for.
    """
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


PEG_DIAMETER = _diameter_from_env()
PEG_RADIUS = PEG_DIAMETER / 2.0
_PEG_INERTIALS = peg_inertials(PEG_DIAMETER)
PEG_MASS = _PEG_INERTIALS["mass"]
PEG_I_AXIAL = _PEG_INERTIALS["inertia_axial"]
PEG_I_TRANS_COM = _PEG_INERTIALS["inertia_transverse_com"]
PEG_I_TRANS_FLANGE = _PEG_INERTIALS["inertia_transverse_flange"]

# Derived, not repeated: 1.0 mm at the 28 mm stage-1 peg. If either diameter
# changes, this follows automatically instead of silently disagreeing.
RADIAL_CLEARANCE = (BORE_DIAMETER - PEG_DIAMETER) / 2.0

# Peg geometry in the flange frame (``wrist_3_link``). The flange frame is the
# body frame: D-020 addendum 1 measured ``ee_joint`` at local (0, 1.3e-8, 0), so
# there is no offset to account for. Flange-local +z points along the tool
# direction, which the increment-1 report measured as world-down at the home
# pose (z . (0,0,-1) = 0.999997, quoted in D-023). The peg therefore spans
# flange-local z in [0, PEG_LENGTH], with its centre -- and centre of mass -- at
# half that.
PEG_CENTRE_OFFSET = (0.0, 0.0, PEG_LENGTH / 2.0)
PEG_TIP_OFFSET = (0.0, 0.0, PEG_LENGTH)


# ---------------------------------------------------------------------------
# Asset path resolution
# ---------------------------------------------------------------------------


def default_peg_robot_usd_path() -> str:
    """Where the authoring script writes, and where the resolver looks last.

    Exists so the authoring script can name its own output without having to
    exist first. It previously recovered the path by parsing the resolver's
    exception message, which broke the moment that message was reworded.

    The filename carries the diameter (``ur10e_peg_d28.usd``), so curriculum
    rungs cannot overwrite each other's assets, and none of them collides
    with increment 3's ``ur10e_peg.usd`` on the training machine's shared
    executable copy. Switching rungs therefore needs no re-authoring of a
    rung already built.
    """
    name = f"ur10e_peg_d{round(PEG_DIAMETER * 1000)}.usd"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Robot", name)


def resolve_peg_robot_usd_path() -> str:
    """Return the path to the peg-welded UR10e USD, or raise with a hint.

    The asset is produced by ``scripts/author_peg_ur10e.py`` and, like every
    USD in this project, is stored outside git. Resolution happens at spawn
    time rather than at module import, so ``list_envs.py`` keeps working on a
    machine that does not have the asset.

    The matching fixture resolver deliberately stays in ``proxytask_env_cfg``:
    ``scripts/verify_fixture_spawn.py`` imports it from there, and moving it
    would break a verified script for cosmetic symmetry.
    """
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


# ---------------------------------------------------------------------------
# Fixed / held asset split (Factory structure, our values)
# ---------------------------------------------------------------------------


@configclass
class FixedAssetCfg:
    """The world-anchored asset carrying the hole: the Tisch fixture.

    ``usd_path`` stays empty and is resolved at spawn time. ``friction`` is the
    documented aluminium/steel pairing from docs/task_specification.md; no
    physics material is authored in the peg increment, so the field is carried
    as data rather than applied.
    """

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
    """The peg. Held in the D-005 sense -- rigidly attached, not grasped.

    Realized as a welded link in the robot USD (D-019), not as a separately
    spawned body: a runtime fixed joint breaks the articulation, and the
    Robot Assembler strips the articulation root (D-018 addendum). These values
    are what ``scripts/author_peg_ur10e.py`` writes into that USD.

    ``friction`` is deliberately ``None``: no measured value exists for the ABS
    peg, and CLAUDE.md forbids guessing physical parameters. It is measured in
    the contact increment, where it is observable.
    """

    usd_path: str = ""
    diameter: float = PEG_DIAMETER
    length: float = PEG_LENGTH
    mass: float = PEG_MASS
    inertia_axial: float = PEG_I_AXIAL
    inertia_transverse_com: float = PEG_I_TRANS_COM
    """About the centre of mass, as USD ``MassAPI:diagonalInertia`` expects."""
    friction: float | None = None
