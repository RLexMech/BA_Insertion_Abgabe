# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task geometry and asset configuration (D-022, D-023, D-018/D-019, D-029).

SQUARE-PEG PIVOT (branch square-peg-insertion, 2026-07-28, UNVERIFIED): the
held asset is a 30 x 30 x 50 mm square prism and the fixed asset is a table
plate carrying a square blind pocket, 35 mm deep, centred where the round bore
used to be. The cylinder constants and the round-bore vocabulary are deleted on
this branch; the increment and demo-sprint branches preserve them.

D-033 replaced the CAD import of that plate with a generator,
``scripts/author_tisch_square.py``, which composes it from nine box colliders.
Two consequences reach this file: the pocket side length is the **curriculum
variable** (the peg is welded into the robot USD and fixed at 30 mm, the table
is regenerated per rung), and the constants below are the generator's *input*
rather than measurements copied out of Creo.

The table asset is authored with its **origin on the opening plane at the
pocket centre** (docs/asset_contract_tisch.md addendum), which flips the
derivation chain: the spawn translation IS the task-frame origin, and the
plate top/bottom are derived from it instead of the other way round.

Renamed from ``task_geometry.py`` in the peg increment, following Factory's
``factory_tasks_cfg.py`` split (docs/factory_mapping.md: adopt the structure,
keep our own values). The plain constants are kept alongside the config classes
because they are what the startup report and the offline tools read; the config
classes carry the fixed/held asset split that the peg introduced.

All positions are relative to the **environment origin**, never absolute world
coordinates: Isaac Lab offsets each parallel environment by
``scene.env_origins``, so treating them as absolute would make the pocket pose
correct only for environment 0.

Values are derived rather than repeated wherever one follows from another --
the per-axis clearance *is* half the width difference, the yaw window *is* a
function of the two widths; before this pattern, 0.755 appeared in three
places, and changing the fixture height would have left the robot floating,
the same class of silent mismatch that cost two debugging rounds on
2026-07-26.
"""

import math
import os

from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Fixture, plate and robot placement (D-022, D-023 distances verified
# 2026-07-26; the asset origin convention changed with the square re-import,
# D-029, UNVERIFIED)
# ---------------------------------------------------------------------------

# Task-frame XY (D-022): the pocket centre, on the -Y half of the plate --
# the same point the round bore occupied, so the D-023 base distances and the
# home pose stay valid.
OPENING_OFFSET_Y = -0.225

# Fixture spawn. The square-pocket re-import is authored with its origin ON
# the opening plane AT the pocket centre, so the spawn translation is the
# task-frame origin itself: (0, -0.225, 0.755). The old asset's origin sat at
# the plate underside centre with the legs below it; this asset hangs
# entirely below and around its origin (bbox z range -0.755 .. 0.000).
FIXTURE_POS = (0.0, OPENING_OFFSET_Y, 0.755)

# Opening entrance = task frame origin = the asset origin by construction.
OPENING_ENTRANCE_POS = FIXTURE_POS

# Plate. The figures originated in the Creo source; since D-033 they are the
# generator's input instead, and ``author_tisch_square.py`` reads them from
# here. Derived downward from the opening plane now that the asset origin sits
# on it.
PLATE_THICKNESS = 0.055
PLATE_HALF_EXTENTS = (0.250, 0.300)
PLATE_TOP_Z = FIXTURE_POS[2]  # 0.755, the opening plane
PLATE_BOTTOM_Z = PLATE_TOP_Z - PLATE_THICKNESS  # 0.700

# UR10e base (D-023): on the plate, on the x axis, 415 mm from the pocket
# along +Y.
ROBOT_BASE_OFFSET_Y = 0.190
ROBOT_BASE_POS = (0.0, ROBOT_BASE_OFFSET_Y, PLATE_TOP_Z)

# Flange standoff above the plate top at the home pose.
HOME_STANDOFF_Z = 0.150

# Derived, for reporting rather than for use: distance from the base centre to
# the near plate edge. The UR10e base cylinder is nominally 190 mm across
# (published figure, not measured from the USD), so roughly 15 mm of rim
# clearance remains.
BASE_TO_PLATE_EDGE = PLATE_HALF_EXTENTS[1] - ROBOT_BASE_OFFSET_Y  # 0.110 m
OPENING_TO_BASE_DISTANCE = ROBOT_BASE_OFFSET_Y - OPENING_OFFSET_Y  # 0.415 m

# ---------------------------------------------------------------------------
# Pocket (fixed asset) -- docs/task_specification.md square addendum. A
# square blind pocket cut into the plate; its bottom is a physical stop
# 10 mm beyond the success threshold, so the peg cannot fall through, and
# 20 mm of plate material remain below it (55 - 35). The pocket ceiling at
# 35 mm against the plate underside at 55 mm keeps the dive-under-the-plate
# exploit of 2026-07-27 closed.
# ---------------------------------------------------------------------------

def _pocket_from_env(default_mm: float = 32.0) -> float:
    """Pocket side length [m], overridable per run via PROXYTASK_POCKET_SIDE_MM.

    The pocket is the curriculum variable since D-033: the peg is welded into
    the robot USD and cannot change per run, the pocket is a generated asset
    (scripts/author_tisch_square.py) and can. Default is the D-029 target
    pocket. A malformed value is a hard error rather than a silent fallback,
    for the same reason as in ``_side_from_env`` below.
    """
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


POCKET_SIDE = _pocket_from_env()
"""Square pocket edge length [m] -- the curriculum variable (D-033)."""
POCKET_DEPTH = 0.035
SUCCESS_DEPTH = 0.025

# ---------------------------------------------------------------------------
# Peg (held asset): a square prism.
#
# CURRICULUM (D-033): the POCKET is the curriculum variable, not the peg.
# The peg is welded into the robot USD and cannot change without a new
# asset; the pocket is generated by scripts/author_tisch_square.py in
# seconds. One rung = one pocket size against the fixed 30 mm peg, and a
# rung sets TWO difficulties at once: the lateral tolerance (per-axis
# clearance at zero yaw) and the free yaw window, because a square of side a
# turned by phi occupies a(|cos phi| + |sin phi|) laterally. Rungs:
#
#     45 mm -> 7.5 mm, yaw free (+-45 deg)  (stage 0: the 42.43 mm diagonal
#                                            fits at ANY yaw -- C4 inactive,
#                                            isolates "reward/obs works"
#                                            from "yaw is hard")
#     40 mm -> 5.0 mm, +-25.5 deg
#     36 mm -> 3.0 mm, +-13.1 deg
#     34 mm -> 2.0 mm, +-8.3 deg
#     32 mm -> 1.0 mm, +-3.96 deg           (D-029 target)
#
# Set per run without editing this file, one generated table per rung:
#
#     $env:PROXYTASK_POCKET_SIDE_MM = "45"
#
# PROXYTASK_PEG_SIDE_MM below still works (special experiments; it needs its
# own authored robot USD per value), but the default path leaves the peg at
# 30 mm and moves the pocket. USD filenames carry their size (``s`` prefix
# for pegs, ``b`` for pockets), so rungs cannot overwrite each other's
# assets in the training machine's shared folder.
#
# Density is unchanged from the round task: the ABS value back-solved from
# the task specification's 0.03202 kg at the Ø28 cylinder -- the material did
# not change, only the cross-section. USD ``MassAPI:diagonalInertia`` is
# specified **about the centre of mass**, so the authoring script must use
# PEG_I_TRANS_COM. PEG_I_TRANS_FLANGE is carried only as the cross-check that
# the two agree via Steiner's theorem.
# ---------------------------------------------------------------------------

PEG_LENGTH = 0.050
PEG_DENSITY = 0.03202 / (math.pi * (0.028 / 2.0) ** 2 * PEG_LENGTH)  # ~1040 kg/m^3, ABS


def peg_inertials(side: float, length: float = PEG_LENGTH, density: float = PEG_DENSITY) -> dict:
    """Mass and inertias of a solid square prism, in the form the USD needs.

    Axis convention matches the authored asset: the prism spans z in
    [0, length] with an a x a cross-section, faces normal to x and y. About
    the long axis, Izz = m (a^2 + a^2) / 12 = m a^2 / 6; transverse about the
    centre of mass, Ixx = Iyy = m (a^2 + L^2) / 12. At the 30 mm rung:
    m = 46.80 g, I_axial = 7.02e-6, I_trans_com = 1.326e-5 kg m^2.
    ``inertia_transverse_com`` is what ``MassAPI:diagonalInertia`` means;
    ``inertia_transverse_flange`` is the Steiner-shifted value the authoring
    script asserts against, and is not written into the asset.
    """
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
    """Peg side length [m], overridable per run via PROXYTASK_PEG_SIDE_MM.

    Default is the D-029 target rung. A malformed value is a hard error
    rather than a silent fallback: training a rung other than the one
    intended would invalidate the comparison it exists for. The round task's
    PROXYTASK_PEG_DIAMETER_MM is likewise a hard error if it is set at all:
    a stale launch script from before the pivot must fail loudly, not
    silently train the default square rung while claiming a diameter.
    """
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


PEG_SIDE = _side_from_env()
_PEG_INERTIALS = peg_inertials(PEG_SIDE)
PEG_MASS = _PEG_INERTIALS["mass"]
PEG_I_AXIAL = _PEG_INERTIALS["inertia_axial"]
PEG_I_TRANS_COM = _PEG_INERTIALS["inertia_transverse_com"]
PEG_I_TRANS_FLANGE = _PEG_INERTIALS["inertia_transverse_flange"]

# Passability is a PAIR now, not a single number (this replaces the round
# task's RADIAL_CLEARANCE): the lateral clearance per axis at zero yaw, and
# the yaw window inside which the turned square still fits the pocket at
# all. A square of side a turned by phi occupies a(|cos phi| + |sin phi|),
# so fitting requires |phi| <= asin(b / (a sqrt 2)) - 45 deg. Above
# b = a sqrt 2 (42.43 mm at the 30 mm peg) the diagonal itself fits and the
# window is the full +-45 deg -- yaw stops binding entirely, which is what
# makes the 45 mm pocket the curriculum's stage 0.
SIDE_CLEARANCE = (POCKET_SIDE - PEG_SIDE) / 2.0

# Belt and braces on the DERIVED quantity, after both reads. No current input
# reaches it: ``_side_from_env`` already rejects a peg at or above the pocket,
# and ``_pocket_from_env``'s lower bound is the default peg side. That is the
# point -- the two bounds are stated in two places, in millimetres, against
# each other's assumptions, and this line is what fails loudly if one of them
# is ever relaxed alone. A non-positive clearance means a peg that cannot
# enter its pocket at any yaw, which every downstream reward term would then
# quietly divide by.
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
)  # +-3.96 deg at the 32 mm pocket

# Peg geometry in the flange frame (``wrist_3_link``). The flange frame is the
# body frame: D-020 addendum 1 measured ``ee_joint`` at local (0, 1.3e-8, 0), so
# there is no offset to account for. Flange-local +z points along the tool
# direction, which the increment-1 report measured as world-down at the home
# pose (z . (0,0,-1) = 0.999997, quoted in D-023). The peg therefore spans
# flange-local z in [0, PEG_LENGTH], with its centre -- and centre of mass -- at
# half that. The prism's side faces are normal to flange-local x and y: with
# the identity weld, the peg's yaw about the tool axis IS the flange's yaw,
# which the startup report measures against the pocket walls (D-029).
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

    The filename carries the side length with an ``s`` prefix
    (``ur10e_peg_s30.usd``; ``%g`` keeps fractional rungs readable, e.g.
    ``s22.6``), so square rungs cannot overwrite each other's assets, and
    none of them collides with the round task's ``ur10e_peg_d*.usd`` on the
    training machine's shared executable copy.
    """
    name = f"ur10e_peg_s{PEG_SIDE * 1000.0:g}.usd"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Robot", name)


def default_fixture_usd_name(side: float | None = None) -> str:
    """Filename of the generated table for a pocket size, ``b`` for the bore.

    The same contract as the peg's ``s`` prefix one level up, for the same
    reason: the size is IN the name, so two rungs cannot overwrite each other
    in the training machine's shared asset folder, and the resolver can refuse
    a size-less ``tisch_square.usd`` outright instead of loading a table whose
    pocket is anyone's guess (D-033, name-is-contract).

    ``side`` defaults to the configured pocket; ``author_tisch_square.py``
    passes its ``--pocket-side-mm`` so the generator and the resolver cannot
    drift into two different spellings of the same filename.
    """
    return f"tisch_square_b{(POCKET_SIDE if side is None else side) * 1000.0:g}.usd"


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
    """The world-anchored asset carrying the pocket: the generated Tisch.

    ``usd_path`` stays empty and is resolved at spawn time. ``side`` is the
    curriculum variable (D-033), and the resolved USD must be the table
    generated for exactly this size -- see ``default_fixture_usd_name``.
    ``friction`` is the documented aluminium/steel pairing from
    docs/task_specification.md; the generator authors no physics material, so
    the field is carried as data rather than applied.
    """

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
    side: float = PEG_SIDE
    """Square cross-section edge length [m]."""
    length: float = PEG_LENGTH
    mass: float = PEG_MASS
    inertia_axial: float = PEG_I_AXIAL
    inertia_transverse_com: float = PEG_I_TRANS_COM
    """About the centre of mass, as USD ``MassAPI:diagonalInertia`` expects."""
    friction: float | None = None
