# NOTE (Phase B copy, 2026-08-18): copied from the verified proxy task
# (square peg, old repo). Geometry constants and square-peg-specific
# derivations (C4 yaw window, per-axis clearance, analytic inertias) are
# PROXY-SPECIFIC and get replaced in later phases from measured real-task
# values. Do not treat any number in this file as a real-task value yet.
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

# Plate. The figures originated in the Creo source; since D-033 they were the
# generator's input instead, and the proxy's ``author_tisch_square.py`` read
# them from here. That generator was deleted on 2026-08-28 (S6).
#
# PLATE_THICKNESS and PLATE_BOTTOM_Z went with it (S6): they fed only each
# other, and the env's own ``cfg.plate_bottom_z`` comes from
# WORKCELL_PLATE_BOTTOM_Z -- table 2's underside in the REAL cell, which is a
# different surface. PLATE_HALF_EXTENTS stays: BASE_TO_PLATE_EDGE below and
# scripts/tools/screen_home_pose_branches.py both read it.
PLATE_HALF_EXTENTS = (0.250, 0.300)
PLATE_TOP_Z = FIXTURE_POS[2]  # 0.755, the opening plane

# UR10e base (D-023): on the plate, on the x axis, 415 mm from the pocket
# along +Y.
ROBOT_BASE_OFFSET_Y = 0.190
# ROBOT_BASE_POS = (0.0, ROBOT_BASE_OFFSET_Y, PLATE_TOP_Z) stood here and was
# DELETED on 2026-08-28 (S6) together with its only reader, ur10e_cfg.py. The
# robot the env spawns is the UR5e and its base pose is
# WORKCELL_ROBOT_BASE_POS (the environment origin), read by ur5e_cfg.py.
# ROBOT_BASE_OFFSET_Y stays: BASE_TO_PLATE_EDGE and OPENING_TO_BASE_DISTANCE
# below are derived from it.

# Standoff of the LEADING TOOL POINT above the plate top at the home pose.
#
# It used to mean the FLANGE standoff, and did so correctly while nothing hung
# below the flange. Once the gripper and part are welded on (D-064) the flange
# is no longer what touches first, and the number that has to be controlled is
# the height of the part bottom. The screener adds FLANGE_TO_PART_BOTTOM back
# on to place the flange (scripts/tools/screen_home_pose_branches.py), so the
# value here is unchanged at 150 mm -- only what it is measured to has moved.
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
    """Pocket side length [m], overridable per run via INSERTION_POCKET_SIDE_MM.

    The pocket is the curriculum variable since D-033: the peg is welded into
    the robot USD and cannot change per run, the pocket is a generated asset
    (scripts/author_tisch_square.py) and can. Default is the D-029 target
    pocket. A malformed value is a hard error rather than a silent fallback,
    for the same reason as in ``_side_from_env`` below.
    """
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
"""Square pocket edge length [m] -- the curriculum variable (D-033)."""
# [proxy] the SQUARE proxy pocket really is 35 mm deep, so the number stays
# what it always was. It is NOT the real fixture's depth and the env no longer
# reads it: the live gate depth is POCKET_SEAT_DEPTH, derived below from the
# measured POCKET_FLOOR_Z. The only readers left are the dead proxy scripts
# (author_tisch_square.py, check_tisch_geometry.py, verify_peg_passability.py);
# this constant dies with them (S6).
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
#     $env:INSERTION_POCKET_SIDE_MM = "45"
#
# INSERTION_PEG_SIDE_MM below still works (special experiments; it needs its
# own authored robot USD per value), but the default path leaves the peg at
# 30 mm and moves the pocket. USD filenames carry their size (``s`` prefix
# for pegs, ``b`` for pockets), so rungs cannot overwrite each other's
# assets in the training machine's shared folder.
#
# Density is unchanged from the round task: the ABS value back-solved from
# the task specification's 0.03202 kg at the Ø28 cylinder -- the material did
# not change, only the cross-section. USD ``MassAPI:diagonalInertia`` is
# specified **about the centre of mass**, so the authoring script must use
# PEG_I_TRANS_COM. PEG_I_TRANS_FLANGE was carried as the cross-check that the
# two agree via Steiner's theorem -- deleted on 2026-08-28 (S6), because the
# script that would have run that cross-check is gone and nothing else read it.
# ``peg_inertials`` still RETURNS the flange value, so the cross-check can be
# taken again from the function without restoring a module constant.
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
    """Peg side length [m], overridable per run via INSERTION_PEG_SIDE_MM.

    Default is the D-029 target rung. A malformed value is a hard error
    rather than a silent fallback: training a rung other than the one
    intended would invalidate the comparison it exists for. The round task's
    INSERTION_PEG_DIAMETER_MM is likewise a hard error if it is set at all:
    a stale launch script from before the pivot must fail loudly, not
    silently train the default square rung while claiming a diameter.
    """
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
        "Check INSERTION_PEG_SIDE_MM and INSERTION_POCKET_SIDE_MM against each other."
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
# PEG_CENTRE_OFFSET and PEG_TIP_OFFSET stood here and were DELETED on
# 2026-08-28 (S6). Nothing read them, and they sat one name away from the
# offset the env really uses: ``peg_tip_offset = FLANGE_TO_PART_BOTTOM``
# (insertion_env_cfg.py), which is measured on the real tool chain.


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

    The matching fixture resolver deliberately stays in ``insertion_env_cfg``:
    ``scripts/verify_fixture_spawn.py`` imports it from there, and moving it
    would break a verified script for cosmetic symmetry.
    """
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


# --- the real tool chain: imported assembly, and the robot it is welded into --
# Two assets, two resolvers, same contract as the peg pair above. The first is
# the CAD import (gripper + part, one file, D-067); the second is what
# scripts/author_tool_ur5e.py produces from it and the shipped UR5e.

TOOL_USD_NAME = "greifer_bauteil_asm.usd"
TOOL_ROBOT_USD_NAME = "ur5e_tool.usd"


def default_tool_usd_path() -> str:
    """Where the CAD import of gripper + part lives (D-067, RT-8 .. RT-12)."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "Werkzeug", TOOL_USD_NAME
    )


def default_tool_robot_usd_path() -> str:
    """Where the tool-welded UR5e is written, and looked for last.

    No size suffix here, unlike the peg's ``s{mm}``: there is exactly one real
    part and no curriculum of rungs to keep apart. If a second grasp or a
    second part ever appears, the name has to carry the difference before two
    assets can share this folder.
    """
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
        "and repair the units (D-067), or set INSERTION_TOOL_USD."
    )


# --- the real fixture: the CAD cut-out the part is inserted into ------------
# Third asset of the same shape as the two above. It is a CAD IMPORT, not a
# generated asset: the STEP goes through the D-067 chain (convert, unit repair,
# instanceable repair, collision) and the result is dropped in by hand. So the
# resolver can only look it up and say what to run -- it can never produce it.
#
# The ``_mm`` in the name is the scar of the D-083 stopgap (the STEP declares
# INCH while its numbers are millimetres); it names the SOURCE that was
# repaired, not a unit of this USD, which reads metersPerUnit 1.0 (RT-36).
POCKET_USD_NAME = "Aufnahme_real_v1_mm.usd"


def default_pocket_usd_path() -> str:
    """Where the imported fixture cut-out lives (D-067, RT-27 .. RT-36)."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "Werkzeug", POCKET_USD_NAME
    )


def resolve_pocket_usd_path() -> str:
    """Path to the real fixture cut-out USD, or raise with a hint.

    Resolved at spawn time, never at import: this is the asset that does not
    exist on the dev laptop, and raising at import would take task registration
    down with it.
    """
    env_override = os.environ.get("INSERTION_POCKET_USD")
    candidates = [p for p in (env_override, default_pocket_usd_path()) if p]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Fixture cut-out USD not found. Tried: "
        + "; ".join(candidates)
        + ". Convert CAD/Aufnahme_real_v1_mm.stp (NOT Aufnahme_real_v1.stp -- it "
        "declares INCH, D-083), repair it with scripts/fix_stage_units.py and "
        "scripts/fix_fixture_asset.py, give it a collider with "
        "scripts/add_fixture_collision.py, or set INSERTION_POCKET_USD."
    )


def resolve_tool_robot_usd_path() -> str:
    """Path to the tool-welded UR5e USD, or raise with a hint.

    Resolved at spawn time, not at import, for the same reason as the peg
    resolver: a machine without the asset must still be able to import this
    module.
    """
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


# --- the Warp meshes: one OBJ per asset, beside its own USD -----------------
# MOVED 2026-08-29 to `insertion_paths.py` and re-exported here, so the names
# below still resolve for every existing caller. The reason is RT-88: this file
# imports `isaaclab.utils.configclass`, so asking it for a FILE PATH costs a
# SimulationApp. A path needs `os` and nothing else. The block's own reasoning
# -- why the OBJ sits beside its USD, and why the part's is in the `tool_link`
# frame -- moved with it and is NOT repeated here.
from .insertion_paths import (  # noqa: F401 - re-export, one home is insertion_paths
    PART_OBJ_NAME,
    POCKET_OBJ_NAME,
    default_part_obj_path,
    default_pocket_obj_path,
    resolve_part_obj_path,
    resolve_pocket_obj_path,
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
    the field is carried as data rather than applied. PROXY data only -- the
    real task's applied value is ``CONTACT_FRICTION`` (D-111), not this field.
    """

    usd_path: str = ""
    side: float = POCKET_SIDE
    """Square pocket edge length [m]."""
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
    SEATED_SUCCESS_DEPTH (D-106 (1)) at the one place this class is
    instantiated, for the same ordering reason as ``depth`` above.
    """
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


# ===========================================================================
# WORKCELL (real task) -- everything above this line is PROXY geometry.
#
# Frame, ROTATED 2026-08-19 (user decision, grill session). The first version
# (2026-08-18) put the fixture in +Y and claimed Factory does the same -- that
# claim was WRONG: Factory spawns its fixed asset at (0.6, 0.0, 0.05), i.e. in
# +X (isaaclab_tasks/direct/factory/factory_tasks_cfg.py:313, PegInsert). What
# we do share with Factory is the anchor: the robot base at the environment
# origin (factory_env_cfg.py: pos=(0.0, 0.0, 0.0)) with the task frame derived
# at runtime from the fixed asset's actual pose (factory_env.py:
# fixed_pos_obs_frame). Including z: the UR5e base_link origin IS (0, 0, 0),
# so the table tops are NEGATIVE.
#
#   +X  from the robot towards the fixture (= towards the adapter's rear
#       wall). Matches Factory AND the user's fixture-CAD frame, which spawns
#       with identity rotation.
#   +Y  to the robot's LEFT (right-handed with +Z up). The fixture stands to
#       the robot's RIGHT, so its lateral centre is NEGATIVE. The block's
#       7.9 mm-margin end points to +Y (M, user 2026-08-19).
#   +Z  up.
#
# Every value carries its provenance:
#   M = measured on site 2026-08-17   R = derived here
#   A = assumed, stated               P = PENDING, see WORKCELL_PENDING
#
# Sources: docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.tex section
# "Arbeitszelle" (the .tex beats the PDF), D-058, D-060; frame rotation
# recorded in docs/decisions_inbox.md (2026-08-19).
# ===========================================================================

WORKCELL_PENDING: tuple[str, ...] = ()
"""Named unknowns. The scene builds and is internally consistent without them,
but is NOT absolutely placed until they are filled in. The startup report
prints this tuple so a run can never quietly look finished."""

# --- Inputs: robot mounting plate (M, measured 2026-08-18) -----------------
# The plate is 24 mm thick and the robot sits centred on it. What the distances
# below are measured TO is the plate's central SQUARE, 153 mm across, whose
# edge is where the robot base begins -- so a distance to that edge plus half
# of 153 is the base centre. That 153 mm is the real footprint and replaces the
# 149 mm datasheet figure.
#
# Why this one number mattered more than its size suggests: MOUNT_PLATE_T sets
# T1_TOP_Z, and every other z in the cell is derived from it, so the whole cell
# -- including the insertion target -- shifts 1:1 with it against the robot.
# The earlier 20 mm placeholder put the block top 4 mm too high.
ROBOT_BASE_SQUARE = 0.153        # M: central square of the plate = base footprint
MOUNT_PLATE_T = 0.024            # M

# --- Inputs: tables (M) ----------------------------------------------------
# X = depth (robot -> fixture), Y = lateral. Same measured plates as before
# the 2026-08-19 frame rotation, values swapped into the new axis roles.
T1_SIZE_X, T1_SIZE_Y = 1.000, 0.600    # M: table 1 is 1000 deep x 600 wide
T2_SIZE_X, T2_SIZE_Y = 0.750, 1.505    # M: table 2 is 750 deep x 1505 wide
TABLE_PLATE_T = 0.020            # A: not measured; only the top face matters
                                 # physically, the thickness just has to be
                                 # thick enough not to be tunnelled through.

PLATE_TOP_DROP = 0.0958          # M: direct measurement, D-058 says it LEADS
                                 # over the 884/787 absolute pair, which would
                                 # give 97.0 mm. Consequence carried openly:
                                 # table 2 lands 1.2 mm above its own reading.
TABLE_GAP_X = 0.0067             # M: edge to edge, table 1 -> table 2 (depth)

# Both are BASE-edge to table-edge, i.e. to the 153 mm square, not to the
# plate's outer contour. The front reading works out because the base begins
# exactly at the plate edge there (user, 2026-08-18).
BASE_EDGE_TO_T1_FRONT = 0.104    # M: -> table-1 edge facing table 2
BASE_EDGE_TO_T1_RIGHT = 0.245    # M: -> table-1 right edge. Supersedes the
                                 # 220 mm in the .tex, which was taken to the
                                 # plate's outer edge; the plate reaches ~25 mm
                                 # further than the square on that side.

T2_OVERHANG_RIGHT = 0.447        # M: LEADS (user decision 2026-08-18)
T2_OVERHANG_LEFT = 0.443         # M: control only -- see WORKCELL_Y_RESIDUAL

# The tape measurements name left and right as seen by a person standing IN
# FRONT of table 2, facing the robot -- the same viewpoint D-058 uses when it
# says the adapter's rail back faces that person. That is the robot's view
# mirrored, so every "right" in the input block is the ROBOT'S LEFT.
#
# In this frame the robot's left IS +Y, so the mirror is absorbed by the axis
# direction and the sign is +1. Anchor (user, 2026-08-18, from the photos):
# standing at the robot and looking towards table 2, the fixture is to the
# RIGHT -- so the lateral chain must land the block at NEGATIVE y. With +1 it
# lands at y = -133 mm (robot's right, correct); -1 would mirror it to +133.
WORKCELL_Y_SIGN = +1.0           # M: derived from the user's left/right call

# --- Inputs: adapter base plate = the marked black rectangle (M) ------------
RECT_SIZE_X, RECT_SIZE_Y = 0.400, 0.880    # M: 400 deep x 880 wide
RECT_TO_T2_REAR = 0.054          # M: rectangle -> the table-2 edge near table 1
RECT_TO_T2_FRONT = 0.296
RECT_TO_T2_LEFT = 0.163
RECT_TO_T2_RIGHT = 0.4615

# --- Inputs: fixture block, OUTER contour only (M) -------------------------
# No pocket. The pocket geometry comes from the user's CAD as a separate file
# and is NOT modelled here -- the lugs are part of the insertion task, so a
# plain rectangular pocket would be a different task, not a coarse one.
# The block's LONG axis (260, the pocket's long axis) runs laterally; the
# 7.9 mm-margin end of the pocket points to +Y (M, user 2026-08-19).
BLOCK_SIZE_X, BLOCK_SIZE_Y = 0.210, 0.260    # M: 210 deep x 260 wide
BLOCK_NEAR_EDGE_FROM_RECT = 0.105   # M: rectangle edge nearest the robot ->
                                    # block edge nearest the robot
GUIDE_EDGE_ABOVE_T2 = 0.1973        # M: 0.01128 + 0.186. Clarified by the
                                    # user 2026-08-19: 11.28 is the wooden
                                    # base plate the adapter stands on, 186
                                    # goes up to the STAGE-1 TOP RIM = the
                                    # fixture's top face. NOT the insertion
                                    # plane -- that sits STAGE1_DEPTH lower.
STAGE1_DEPTH = 0.015                # M: from the user's fixture CAD
                                    # (Aufnahme_real_v1.stp, 2026-08-19):
                                    # stage-1 rim (z=+15 there) down to the
                                    # stage-2 opening plane (z=0 there).

# --- Inputs: the fixture CAD itself (M) ------------------------------------
# Measured on 2026-08-24 by reading CAD/Aufnahme_real_v1.stp directly: all 532
# CARTESIAN_POINTs, SI_UNIT(.MILLI.,.METRE.). Not read off a drawing and not
# taken from the USD -- the STEP is the authority (inbox "Fixture CAD
# authority") and the USD does not exist on this machine.
#
# What the CAD IS, and this decides how the block is modelled: the user
# measured the real pocket 1:1 and left roughly 1 cm of wall standing around
# it. The file is therefore a CUT-OUT OF THE BLOCK, not a separate insert that
# drops in. Block and cut-out are the same material and their faces touch --
# no clearance belongs between them (user, 2026-08-24, with photograph).
#
# Asset frame = the spawn frame: origin in the centre of the stage-2 opening,
# +Z out of the opening. The origin is centred in the OPENING but NOT in the
# outer body -- the ranges below are asymmetric, which is the 7.9 mm ledge.
POCKET_WALL_X = 0.0452938           # M: pocket wall, short axis (+-)
POCKET_WALL_Y = 0.07255             # M: pocket wall, long axis (+-)
POCKET_FLOOR_Z = -0.036             # M: pocket floor below the opening plane
POCKET_ASSET_BOTTOM_Z = -0.046      # M: underside of the cut-out
POCKET_LOCAL_X_RANGE = (-0.0847938, 0.0652062)   # M: outer body, short axis
POCKET_LOCAL_Y_RANGE = (-0.08865, 0.10045)       # M: outer body, long axis

# Stage 1 is the opening the part meets FIRST, one step above stage 2. Read
# off the top face (z=+15) of the STEP, not off a drawing:
POCKET_STAGE1_X_RANGE = (-0.0747938, 0.0552938)  # M: 130.0876 mm across
POCKET_STAGE1_Y_RANGE = (-0.07255, 0.08045)      # M: 153.0 mm along
# The 7.9 mm step sits at +y only: at -y stage 1 and stage 2 share the edge
# (-72.55 both). That asymmetry is the direction anchor for the whole fixture
# (see POCKET_ORIGIN_Y) -- the user placed that end towards +Y on 2026-08-19.
#
# The 47 x 12 notch reaches PAST the stage-1 outline by 6.1 mm at both ends.
# It is inside POCKET_LOCAL_Y_RANGE, so the block recess already clears it;
# it is written down because the outline alone would misdescribe the part.
POCKET_NOTCH_X = 0.0470876          # M: notch width
POCKET_NOTCH_Y_RANGE = (-0.07865, 0.08655)       # M: notch ends

# THE OUTER BODY IS NOT A RECTANGLE, and the first scene render showed it:
# two open slots along the +Y edge (user, 2026-08-24 evening, screenshot).
# Measured from the STEP the same evening, at both z = -46 and z = +15, so the
# contour is constant over the full height:
#
#   the body runs to y = +90.45 everywhere EXCEPT for a TAB in the middle,
#   x -32.7938 ... +34.2062, which reaches the +100.45 that sets the bbox.
#
# So POCKET_LOCAL_Y_RANGE[1] describes the TAB, not the body. A recess cut to
# the bounding box therefore leaves 10 mm of air on both sides of the tab --
# 52.0 mm long at -x and 31.0 mm long at +x. Those are the two holes. The rest
# of the outline IS the full rectangle; this is the only place it is inset.
POCKET_BODY_Y_MAX = 0.09045         # M: outer body edge at +Y, off the tab
POCKET_TAB_X_RANGE = (-0.0327938, 0.0342062)     # M: the tab that reaches +100.45

# --- Inputs: where the fixture sits INSIDE the block (M) --------------------
# Source: docs/Geometrie/CAD_Massblatt (1).pdf, row B12 ("links 80 / rechts 10
# / oben 50 / unten 58"), each value re-derived from the STEP on 2026-08-24:
# left step 29.5, right rib zone 10.0, top step 7.9, bottom step 0.0. The
# 80 mm was measured again on the real block by the user the same day.
#
# NOT centred, in either axis. Before this the fixture was spawned at the
# block centre, which was never measured -- it moves 49.8 mm in x.
BLOCK_FRONT_TO_STAGE1 = 0.080       # M: block face nearest the robot -> stage 1
BLOCK_REAR_WALL_T = 0.015           # M: the real wall behind the pocket,
                                    # measured from the stage-1 outline to the
                                    # outside -- which is what a calliper on the
                                    # real block reads.
                                    #
                                    # 15, NOT 10, since 2026-08-24 (evening):
                                    # the user re-measured the wall on the real
                                    # block. This SUPERSEDES both the Massblatt's
                                    # W2 ("Differenz 10") and the earlier 10 mm
                                    # reading from the same day. A direct
                                    # re-measurement outranks a drawing, the
                                    # same precedent as the 80 mm.
                                    #
                                    # CONSEQUENCE, and it is the reason the
                                    # earlier wording had to go: the fixture's
                                    # OWN rear wall is 9.9124 mm (STEP: stage-1
                                    # outline 55.2938 to body 65.2062). It is
                                    # therefore NOT the block's rear wall any
                                    # more -- 5.0876 mm of block stands behind
                                    # it, and that is why the back face reads as
                                    # one material instead of two (user).
BLOCK_REAR_WALL_H = 0.100           # M(approx): "so 10cm hoch" (user). The
                                    # high wall above it -- it is why the part
                                    # can only go in from the front, so it gets
                                    # collision, not just a visual (user,
                                    # 2026-08-24).
BLOCK_PLUSY_TO_STAGE1 = 0.050       # M: B12 "oben 50". LEADS.
BLOCK_MINUSY_TO_STAGE1_B12 = 0.058  # M: B12 "unten 58", the DRAWING reading.
                                    # Documentary only -- kept so the number
                                    # the model departs from stays visible.
BLOCK_MINUSY_TO_STAGE1 = 0.057      # MODELLED, not measured, and the two are
                                    # 1.0 mm apart on purpose.
                                    #
                                    # 2026-08-24 (evening) the user re-measured
                                    # stage 1 on the real block and read 153.0,
                                    # which is what the CAD carries. That closes
                                    # the Massblatt's W1 against its computed
                                    # 151.7 and takes the pocket length OUT of
                                    # the list of suspects for the 1.0 mm that
                                    # did not close. The millimetre must
                                    # therefore sit in 50, 58 or the block's
                                    # 260, and WHICH ONE IS UNKNOWN.
                                    #
                                    # It is put entirely here because the +Y
                                    # reading LEADS (same precedent as
                                    # T2_OVERHANG_RIGHT) and nothing in the task
                                    # touches this wall. NOT split 0.5/0.5:
                                    # that would move the fixture on a number
                                    # nobody measured.
                                    #
                                    # CONSEQUENCE, and it must not be read as
                                    # progress: WORKCELL_POCKET_Y_RESIDUAL is
                                    # now ZERO BY CONSTRUCTION. The
                                    # disagreement did not go away, it moved to
                                    # WORKCELL_POCKET_MINUSY_VS_B12.

# --- Derived (R) -----------------------------------------------------------
# Read strictly outwards from the robot foot, so exactly one input (the
# mounting plate) has to change when the photo arrives.

T1_TOP_Z = -MOUNT_PLATE_T
T2_TOP_Z = T1_TOP_Z - PLATE_TOP_DROP
BLOCK_TOP_Z = T2_TOP_Z + GUIDE_EDGE_ABOVE_T2

# X (depth): robot foot -> table-1 near edge -> gap -> table 2 -> rectangle
# -> block. Table 1 extends BACK under the robot (minus), table 2 extends
# FORWARD past its near edge (plus) -- the polarity carried over unchanged
# from the pre-rotation Y chain.
T1_NEAR_EDGE_X = BASE_EDGE_TO_T1_FRONT + ROBOT_BASE_SQUARE / 2.0
T2_NEAR_EDGE_X = T1_NEAR_EDGE_X + TABLE_GAP_X
RECT_NEAR_EDGE_X = T2_NEAR_EDGE_X + RECT_TO_T2_REAR
BLOCK_NEAR_EDGE_X = RECT_NEAR_EDGE_X + BLOCK_NEAR_EDGE_FROM_RECT
BLOCK_CENTRE_X = BLOCK_NEAR_EDGE_X + BLOCK_SIZE_X / 2.0

T1_CENTRE_X = T1_NEAR_EDGE_X - T1_SIZE_X / 2.0
T2_CENTRE_X = T2_NEAR_EDGE_X + T2_SIZE_X / 2.0

# Y (lateral): the robot foot sits BASE_EDGE_TO_T1_RIGHT + half the base
# square from table 1's right edge; table 1's right edge sits
# T2_OVERHANG_RIGHT inside table 2's. "Right" is the table-2 viewer's right =
# the robot's left = +Y, see WORKCELL_Y_SIGN.
_T1_RIGHT_EDGE_Y = WORKCELL_Y_SIGN * (BASE_EDGE_TO_T1_RIGHT + ROBOT_BASE_SQUARE / 2.0)
T1_CENTRE_Y = _T1_RIGHT_EDGE_Y - WORKCELL_Y_SIGN * T1_SIZE_Y / 2.0
_T2_RIGHT_EDGE_Y = _T1_RIGHT_EDGE_Y + WORKCELL_Y_SIGN * T2_OVERHANG_RIGHT
T2_CENTRE_Y = _T2_RIGHT_EDGE_Y - WORKCELL_Y_SIGN * T2_SIZE_Y / 2.0
RECT_CENTRE_Y = _T2_RIGHT_EDGE_Y - WORKCELL_Y_SIGN * (RECT_TO_T2_RIGHT + RECT_SIZE_Y / 2.0)

# A: the block is taken laterally centred on the rectangle. The on-site chain
# "100 = 10 + 40 + 50" describes the POCKET, not the block, and the pocket is
# not modelled here. Nothing touches the block in this build, so the error is
# cosmetic -- but it is an assumption, so it is labelled as one.
BLOCK_CENTRE_Y = RECT_CENTRE_Y

# The block spans from the table-2 top up to the guide edge in one piece,
# rather than as base plate + intermediate level + block. Deliberate
# simplification: nothing reaches underneath, and it removes the assumed
# 116 mm block height from the chain entirely.
#
# It is not one BOX, though: the CAD cut-out occupies part of that volume, so
# the block is authored as four walls plus a floor around it (see
# POCKET_RECESS_* below and scripts/author_workcell.py).
BLOCK_HEIGHT = GUIDE_EDGE_ABOVE_T2

# Residual of the over-determined lateral (Y) chain: the left overhang is also
# measured, so the two readings disagree. Kept as a number, not a comment, so
# the check script and the startup report can print it instead of it being
# folded away.
WORKCELL_Y_RESIDUAL = (T2_SIZE_Y - T2_OVERHANG_RIGHT - T1_SIZE_Y) - T2_OVERHANG_LEFT

TABLE1_POS = (T1_CENTRE_X, T1_CENTRE_Y, T1_TOP_Z - TABLE_PLATE_T / 2.0)
TABLE2_POS = (T2_CENTRE_X, T2_CENTRE_Y, T2_TOP_Z - TABLE_PLATE_T / 2.0)
BLOCK_POS = (BLOCK_CENTRE_X, BLOCK_CENTRE_Y, BLOCK_TOP_Z - BLOCK_HEIGHT / 2.0)

# --- Derived: the fixture CAD (R) ------------------------------------------
# The stage-1 rim in the ASSET's own frame. It is the same number as
# STAGE1_DEPTH by construction -- STAGE1_DEPTH is defined as the drop from
# that rim to the opening plane, and the opening plane is the asset origin --
# so it is derived here rather than written a second time.
POCKET_RIM_Z = STAGE1_DEPTH
POCKET_LOCAL_Z_RANGE = (POCKET_ASSET_BOTTOM_Z, POCKET_RIM_Z)

# FULL SEAT: how far the part bottom sits below the opening plane when it rests
# on the stage-2 floor. Same number as POCKET_FLOOR_Z with the sign flipped --
# depth counts DOWNWARD from the opening, the floor coordinate counts upward --
# so it is derived here and never written a second time. POCKET_FLOOR_Z is the
# home of the fact (M, read from the STEP).
#
# This is the env's live gate depth (``cfg.fixed_asset.depth``). The proxy's
# POCKET_DEPTH = 0.035 is a DIFFERENT pocket and is not read by the env.
POCKET_SEAT_DEPTH = -POCKET_FLOOR_Z

# D-106 (1): success requires the part at most 3 mm above full seat. DECIDED,
# not measured -- the band adopts IndustReal's, the only absolute-band
# precedent in the Block-2 literature check, and the user chose it over the
# 2 mm alternative ("3mm passt", 2026-08-25). D-106 is the home of the
# reasoning; this line is only the number's one place in code.
#
# D-121 superseded D-106's ALIGNMENT inputs, not its depth band: the band is
# an absolute 3 mm and does not divide by the cross play.
D106_DEPTH_BAND = 0.003

# The env's live success depth (``cfg.fixed_asset.success_depth``, read by the
# success predicate in ``InsertionEnv._get_dones``). Derived, so it follows both the
# measured floor and the decided band. The proxy's SUCCESS_DEPTH = 0.025 is a
# different pocket's and is not read by the env.
SEATED_SUCCESS_DEPTH = POCKET_SEAT_DEPTH - D106_DEPTH_BAND

# The stage-2 opening, i.e. what the part has to pass through. Long axis runs
# in Y (lateral), matching the block.
#
# ACROSS: the play is PLAY_X, derived below from PART_BODY_X. It is NOT
# written here as a literal -- one home per fact, and this line used to hold
# a wrong one: "the play is 0.2876 mm against a 90.3 mm body and nothing
# disputes that" (CORRECTED 2026-08-28). The 90.3 was a caliper reading whose
# two siblings had already been superseded by CAD; the part body is 90.00 mm
# and the play is 0.5876 mm. See PART_BODY_X.
#
# LENGTHWAYS: settled by D-088 (2026-08-25). The part is 143.50 mm (part CAD,
# D-075; user-confirmed), so the play against the 145.1 opening is 1.60 mm.
# The competing 143.34 (CAD_Massblatt 18-08) and 144 (Massblatt "final 21.08")
# are superseded -- see D-088 for the history. The REAL rig's per-side
# clearance (0.9 mm total lengthways, 0.4 mm across) lives in D-087 and is
# deliberately not modelled here: the fixture is a simplification, the CAD
# governs the simulation (user, 2026-08-24).
POCKET_OPENING_X = 2.0 * POCKET_WALL_X
POCKET_OPENING_Y = 2.0 * POCKET_WALL_Y

POCKET_STAGE1_SIZE_X = POCKET_STAGE1_X_RANGE[1] - POCKET_STAGE1_X_RANGE[0]
POCKET_STAGE1_SIZE_Y = POCKET_STAGE1_Y_RANGE[1] - POCKET_STAGE1_Y_RANGE[0]

# Outer bounding box of the CAD cut-out. The block's recess is exactly this,
# with no clearance: same material, faces touching.
POCKET_ASSET_BBOX_M = (
    POCKET_LOCAL_X_RANGE[1] - POCKET_LOCAL_X_RANGE[0],
    POCKET_LOCAL_Y_RANGE[1] - POCKET_LOCAL_Y_RANGE[0],
    POCKET_LOCAL_Z_RANGE[1] - POCKET_LOCAL_Z_RANGE[0],
)

# --- Derived: where the fixture spawns (R) ---------------------------------
# Read from the block face nearest the robot, which is the anchored end of the
# measured chain, inwards to the asset origin.
POCKET_ORIGIN_X = (
    BLOCK_NEAR_EDGE_X + BLOCK_FRONT_TO_STAGE1 - POCKET_STAGE1_X_RANGE[0]
)
# Read from the +Y block edge, because that is the end the 7.9 mm step sits at
# and the end B12 measures as 50 mm. The -Y reading is the control below.
POCKET_ORIGIN_Y = (
    BLOCK_CENTRE_Y + BLOCK_SIZE_Y / 2.0
    - BLOCK_PLUSY_TO_STAGE1 - POCKET_STAGE1_Y_RANGE[1]
)

# Read as the depth CHAIN rather than as an addition to BLOCK_SIZE_X. The two
# gave the same answer while the wall was 9.9124 mm, because that number was
# itself derived as 220 - 210.0876. With a measured 15 mm wall they no longer
# agree, and the chain is the one that means something: front gap, pocket,
# wall. 80 + 130.0876 + 15 = 225.0876 mm.
BLOCK_MODEL_SIZE_X = BLOCK_FRONT_TO_STAGE1 + POCKET_STAGE1_SIZE_X + BLOCK_REAR_WALL_T

# What the BLOCK adds behind the cut-out's own rear wall. Positive means the
# block stands proud of the fixture at +X, which is the visual the user asked
# for; zero would mean they end flush, negative would mean the cut-out pokes
# out of the block and something is wrong.
BLOCK_BEHIND_POCKET_X = (
    BLOCK_MODEL_SIZE_X - BLOCK_FRONT_TO_STAGE1
    - (POCKET_LOCAL_X_RANGE[1] - POCKET_STAGE1_X_RANGE[0])
)

# Residual of the over-determined lateral chain, same treatment as
# WORKCELL_Y_RESIDUAL: kept as a number so the checker can print it. It is
# -1.0 mm.
#
# WHERE IT IS NOT, since 2026-08-24 (evening): in POCKET_STAGE1_SIZE_Y. The
# Massblatt computed the stage-1 length as 151.7 and flagged the 1.3 mm gap to
# its own measured 153 as W1. The user re-measured the pocket on the real
# block and read 153. CAD and part agree, so W1 is DECIDED for 153.0 and this
# residual is not it.
#
# WHERE IT THEN IS: in one of 50 / 58 / 260, and which one is UNKNOWN --
# 50 + 153.0 + 58 = 261.0 against a 260 mm block. BLOCK_PLUSY_TO_STAGE1 leads,
# so the whole millimetre lands on the -Y wall: 40.9 mm in the model, i.e.
# 57.0 mm from the stage-1 edge instead of the 58 the Massblatt gives. Nothing
# in the task touches that wall. Splitting it 0.5/0.5 would move the fixture
# on a number nobody measured, so it is not split.
# ZERO BY CONSTRUCTION since 2026-08-24 (evening), and that is the whole point
# of the comment: BLOCK_MINUSY_TO_STAGE1 was SET to the value that closes this
# sum, so a zero here is arithmetic, NOT agreement between measurements. It is
# kept as a construction check -- if it ever leaves zero, one of the other
# three numbers moved and the modelled -Y wall was not updated with it.
WORKCELL_POCKET_Y_RESIDUAL = (
    BLOCK_SIZE_Y - BLOCK_PLUSY_TO_STAGE1 - POCKET_STAGE1_SIZE_Y
    - BLOCK_MINUSY_TO_STAGE1
)

# ... and THIS is where the disagreement actually lives now. The drawing says
# 58, the model uses 57. One millimetre, in a number nobody has re-measured on
# the real block. A calliper on that wall closes it; arithmetic cannot, because
# the model was fitted to the other three.
WORKCELL_POCKET_MINUSY_VS_B12 = BLOCK_MINUSY_TO_STAGE1_B12 - BLOCK_MINUSY_TO_STAGE1


def _box_from_ranges(name: str, x_range, y_range, z_range) -> dict:
    """(min, max) per axis -> the {name, centre, size} the authoring script wants."""
    return {
        "name": name,
        "centre": tuple((lo + hi) / 2.0 for lo, hi in (x_range, y_range, z_range)),
        "size": tuple(hi - lo for lo, hi in (x_range, y_range, z_range)),
    }


def block_recess_boxes() -> list[dict]:
    """The block AROUND the CAD cut-out, asset-local: eight boxes.

    Lives here rather than in the authoring script because the checker and the
    startup report need the same boxes, and a second derivation is a second
    thing that can drift.

    The recess is EXACTLY the cut-out's bounding box -- no clearance. The two
    are the same block of material (the user measured the pocket 1:1 and left
    a rim of wall around it), so a gap would be an invention. Touching faces
    between two kinematic bodies generate no forces
    (docs/reference/literature_check_collision_modelling_2026-08-17.md, C2).

    BUT A BOUNDING BOX IS NOT THE PART, and the first render proved it. Two of
    the eight boxes exist only to repair that:

    * ``block_spalt_y_plus_*`` fill the 10 mm strip beside the +Y tab. The body
      stops at POCKET_BODY_Y_MAX there while the bbox runs to the tab, so the
      recess opened two slots straight through the wall -- 52.0 mm at -x and
      31.0 mm at +x (user screenshot, 2026-08-24 evening; contour measured off
      the STEP, constant over the full height). Everywhere else the outline IS
      the rectangle, which is why two boxes are enough and not six.
    * ``block_wand_x_plus`` is new for the same session's second correction:
      the rear wall was re-measured at 15 mm, so the block no longer ends at
      the cut-out's rear face. BLOCK_BEHIND_POCKET_X of block stands behind it.

    ``block_rueckwand`` is the high wall on top of that rear wall -- what makes
    the part enterable only from the front. It spans the FULL 15 mm now, not
    the cut-out's 9.9124.

    Asset-local means: origin on the stage-2 opening plane at the stage-2
    centre, exactly as for the cut-out itself, so both spawn at
    WORKCELL_BLOCK_POS with no arithmetic.
    """
    rx0, rx1 = POCKET_LOCAL_X_RANGE
    ry0, ry1 = POCKET_LOCAL_Y_RANGE
    tab_x0, tab_x1 = POCKET_TAB_X_RANGE
    x_front = BLOCK_NEAR_EDGE_X - POCKET_ORIGIN_X
    x_rear = x_front + BLOCK_MODEL_SIZE_X
    y_lo = (BLOCK_CENTRE_Y - BLOCK_SIZE_Y / 2.0) - POCKET_ORIGIN_Y
    y_hi = (BLOCK_CENTRE_Y + BLOCK_SIZE_Y / 2.0) - POCKET_ORIGIN_Y
    top = STAGE1_DEPTH                      # block top == cut-out top (user)
    bottom = STAGE1_DEPTH - BLOCK_HEIGHT    # meets table 2
    full_z = (bottom, top)
    # The slots run only over the cut-out's own height; below it the floor box
    # already fills the strip, and overlapping colliders would be sloppy.
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


# --- Workcell assets -------------------------------------------------------
# TWO files, not one. The tables are world-anchored scenery; the fixture block
# has to be a separate kinematic body because D-034 writes its pose per reset,
# and a pose write needs its own root.
#
# Origins, chosen so the later CAD swap is a drop-in:
#   tables -> the robot foot, i.e. the environment origin. Spawns at (0,0,0)
#             and every box already carries its final place.
#   block  -> the STAGE-2 OPENING-PLANE CENTRE (user decision 2026-08-19,
#             amends the D-060 "top face" wording -- in the proxy both planes
#             were the same surface, here stage 1 separates them by
#             STAGE1_DEPTH). This keeps the proxy semantics "asset origin =
#             opening plane": part-tip z relative to the origin reads directly
#             as insertion depth, and the D-044 tilt/yaw randomisation rotates
#             the pocket about its insertion plane. The asset therefore pokes
#             STAGE1_DEPTH ABOVE its own origin (stage-1 rim) and hangs below
#             otherwise. The user's CAD (Aufnahme_real_v1.stp) is exported
#             with exactly this origin and spawns with no offset arithmetic.

# _v2 with the 2026-08-19 frame rotation: a stale _v1 USD on the training
# machine bakes the OLD axis roles and would load silently -- the resolver
# looks the file up by exact name, so bumping the name is the guard.
WORKCELL_TABLES_USD_NAME = "arbeitszelle_tische_v2.usd"
# _v3 (2026-08-24): the block stopped being one solid box. It became walls, a
# floor and a rear wall AROUND the CAD cut-out (block_recess_boxes()), and its
# origin moved from the block centre to the pocket origin. A _v2 file on the
# training machine would load without a word and put a wall where the pocket is.
#
# _v4 (2026-08-24, evening), after the FIRST RENDER was looked at: two open
# slots beside the +Y tab are filled, and the rear wall went from 10 to a
# re-measured 15 mm, so the block now stands proud of the cut-out. Eight boxes
# instead of five. A _v3 file still has the two holes in it, and a hole is
# exactly the kind of defect that loads without complaint.
WORKCELL_BLOCK_USD_NAME = "aufnahme_block_v4.usd"

# The insertion plane: stage-2 opening, where depth counting starts. This is
# where the fixture ASSET ORIGIN spawns (see the origin note above).
INSERTION_PLANE_Z = BLOCK_TOP_Z - STAGE1_DEPTH

# Height of the LEADING TOOL POINT above the INSERTION PLANE at the home pose.
#
# Two reference planes are in play and they are STAGE1_DEPTH apart, which is
# why this cannot be HOME_STANDOFF_Z: that one is quoted above the BLOCK TOP,
# because the block top is what the branch screener has to clear the arm
# against. Every depth the env reports is measured from the stage-2 opening,
# 15 mm lower. Using the wrong one of the two is a 15 mm error in a report line
# that reads as authoritative, so the two are given separate names.
WORKCELL_HOME_TIP_ABOVE_ENTRANCE = HOME_STANDOFF_Z + STAGE1_DEPTH   # 0.165 m

WORKCELL_TABLES_POS = (0.0, 0.0, 0.0)
WORKCELL_BLOCK_POS = (POCKET_ORIGIN_X, POCKET_ORIGIN_Y, INSERTION_PLANE_Z)


def default_workcell_usd_path(name: str) -> str:
    """Where ``scripts/author_workcell.py`` writes, and the resolver looks last."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Arbeitszelle", name)


def resolve_workcell_usd_path(name: str, env_var: str) -> str:
    """Resolve a workcell asset at SPAWN time, never at module import.

    Same contract as ``resolve_peg_robot_usd_path``: USDs live outside git, so
    raising at import would take task registration down on any machine that has
    not generated them yet -- including this dev laptop, where ``list_envs.py``
    still has to work.
    """
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


# --- Floor -----------------------------------------------------------------
# With the robot foot at the origin the ground plane is no longer at z = 0. It
# is the only place the ABSOLUTE table height enters: everything else in this
# section is a relative measurement, which is why the 1.2 mm quarrel between
# the 884/787 pair and the direct 95.8 mm drop reaches the floor and nothing
# else.
T1_HEIGHT_ABOVE_FLOOR = 0.884    # M
FLOOR_Z = T1_TOP_Z - T1_HEIGHT_ABOVE_FLOOR

# --- Names the env cfg mirrors ---------------------------------------------
# Mapped onto the field names the environment and the verify scripts already
# use, so the switch does not rename a dozen call sites. "plate" now means
# TABLE 2 -- the surface the fixture stands on -- because that is the role the
# proxy plate played.
WORKCELL_PLATE_TOP_Z = T2_TOP_Z
WORKCELL_PLATE_BOTTOM_Z = T2_TOP_Z - TABLE_PLATE_T
WORKCELL_PLATE_HALF_EXTENTS = (T2_SIZE_X / 2.0, T2_SIZE_Y / 2.0)

# The point the observation frame hangs off: the stage-2 opening-plane centre
# (2026-08-19 origin convention). Part-tip z below this point IS insertion
# depth, which is exactly what "entrance" promises.
WORKCELL_ENTRANCE_POS = WORKCELL_BLOCK_POS
WORKCELL_ROBOT_BASE_POS = (0.0, 0.0, 0.0)


# --- Tool chain: flange -> gripper -> suction -> part (D-068, D-069) --------
# Measured on the real cell with a calliper, 2026-08-21. The chain is a
# STRAIGHT LINE IN Z: the two suction cups sit symmetrically about the gripper
# axis (user-confirmed from the grasp photographs), so their midpoint is on the
# axis and no lateral term appears.
#
# Three surfaces are involved and they are NOT the same surface. Naming them
# apart is the whole point of this block -- an earlier draft collapsed
# "part top" and "suction face" into one name and produced an ambiguous number:
#
#   flange face  ->  the surface the gripper bolts to. This is z = 0.
#   part top     ->  the part's OUTER upper edge.
#   suction face ->  the INNER floor of the terminal-block chambers. The cups
#                    reach down INTO the part, so this lies BELOW the part top.
#   part bottom  ->  the closed outer underside. This enters the pocket first,
#                    so it is the task frame.
#
# Ordering is forced by geometry: part top < suction face < part bottom.
FLANGE_TO_SUCTION_TIP_FREE = 0.1486   # M: cup tip, UNLOADED, no part held.
                                      # Documentary only -- never weld with it.
FLANGE_TO_SUCTION_FACE = 0.1420       # M: GRIPPED. Tool TCP (industry sense).
FLANGE_TO_PART_BOTTOM = 0.1520        # M: GRIPPED. Task frame (Factory
                                      # `held_base` sense). Reward and every
                                      # distance term use THIS one.

# Derived, both documentary. Kept because each one names a physical quantity a
# reader will otherwise try to recompute from the three numbers above.
BELLOWS_COMPRESSION = FLANGE_TO_SUCTION_TIP_FREE - FLANGE_TO_SUCTION_FACE
PART_FLOOR_T_AT_CUP = FLANGE_TO_PART_BOTTOM - FLANGE_TO_SUCTION_FACE

# --- The seated tool orientation in the POCKET frame (RT-102, 2026-08-30) ---
# wxyz, a 180 deg rotation about the pocket y-axis: the tool_link frame hangs
# INVERTED in the cell -- its +z runs down the tool chain into the pocket
# (peg_tip_offset is +FLANGE_TO_PART_BOTTOM along tool z), while the pocket
# z-axis points OUT of the pocket. MEASURED, not derived: RT-102 read the
# tool_link quaternion at the seated pose as (0.0010, 0.0000, -1.0000,
# -0.0000) in the env frame (= the pocket frame at zero fixture noise), and
# RT-81 proved the physical seat ACCEPTS exactly this orientation (33 mm at
# 8.08 N, lateral 0.023 mm). D-107 (2) says the part fits its pocket in ONE
# rotational position, so fixture_quat composed with THIS constant is the
# seated orientation.
#
# WHY IT EXISTS: the first goal-pose build (63e9d24) took the fixture
# quaternion ALONE as the goal orientation -- the reading of "one rotational
# fit" that forgets the tool frame is flipped. RT-102 falsified it exactly as
# the written expectation predicted: sdf mean-outside 25.76 mm at a
# physically seated pose, and the goal FLANGE 125 mm UNDER the entrance.
SEATED_TOOL_QUAT_LOCAL = (0.0, 0.0, 1.0, 0.0)

# --- Part pose in the flange frame (CAD assembly, 2026-08-22) ---------------
# Provenance marker C = read from the user's Creo assembly. CAD is a source in
# its own right here, distinct from the on-site caliper (M): D-057 ranks CAD
# ABOVE measurement for the PART (and measurement above CAD for the fixture,
# which has no CAD at all).
#
# The assembly puts its origin on the flange mating face with +Z away from the
# robot (D-068). Read off in Creo, the flange face to the part centre is
#
#   dx = 0.05 mm      dy = -0.07 mm     dz = 126.747 mm
#
# and the STEP export confirms all three to 10 digits, including the SIGN of
# dy, which the hand reading did not carry. The rotation is NOT the identity;
# see the weld-transform block further down -- that distinction cost one
# wrong entry in this file and is the reason it is spelled out twice.
#
# The lateral pair is a CHECKED zero, not an assumed one -- that distinction is
# the reason this paragraph exists. 0.05 / 0.07 mm sits three orders of
# magnitude below the tightest real clearance (0.2 mm on the short axis,
# D-057) and below the grasp scatter GRASP_POS_NOISE will deliberately inject
# (D-070), so the lateral offset is carried as exactly zero. It gets no
# constant of its own: by the rule stated above for documentary values, a
# number nobody would ever recompute does not earn a name, and standing there
# as a named constant it would look like a modelled quantity and invite
# someone to wire it into a pose.
#
# It was worth measuring because a top view of the assembly on 2026-08-22
# appeared to show a large lateral offset. Those were non-central planes, not
# the mid-planes -- the offset was an artefact of the view. With that settled,
# the "straight line in Z" claim of the block above now holds for the PART as
# well, where until now it was argued only for the suction cups.
FLANGE_TO_PART_CENTRE = 0.126747   # C: flange face -> part centre, tool axis.

# --- Weld transform, read out of the exported STEP (C, 2026-08-22) ----------
# CORRECTION of an earlier entry in this file, kept visible because the wrong
# version was plausible: it said "the weld rotation is the IDENTITY, verified",
# inferred from a Creo reading of 180.000 deg between the flange face and a
# part datum plane. That reading is real, and the datum coordinate system
# FUEGETEIL_MITTE inside the assembly does carry identity orientation -- but a
# datum is not the geometry. The exported assembly places the part's GEOMETRY
# with a 180 deg rotation about Y. Spawning the part file under the flange
# with an identity rotation puts the part UPSIDE DOWN.
#
# Source: CAD/Step/greifer_bauteil_asm.stp, export run 18:27. Its origin is
# GREIFER_MITTE at (0, 0, 0) with identity orientation, which is what proves
# the flange coordinate system was used for the export. Structure survived the
# export as two components, PRODUCT('FUEGETEIL') and PRODUCT('GREIFER_V1'),
# each with an explicit ITEM_DEFINED_TRANSFORMATION.
#
# Verified, not just read: pushing each component's centroid (the STEP carries
# it as a geometric validation property, in the PART's own frame) through the
# transform below reproduces the centroid the assembly states in the FLANGE
# frame, to 12 significant digits, for both components. Centroid and placement
# are independent entries in the file, so this is a real cross-check.
#
#   FUEGETEIL   X -> (-1, 0, 0)   Y -> (0, +1, 0)   Z -> (0, 0, -1)
#               t = (0.05, -0.07, 126.746951) mm      = 180 deg about Y
#   GREIFER_V1  X -> ( 0, 0, 1)   Y -> (0, -1, 0)   Z -> (1, 0,  0)
#               t = (0, 0, 0)                          = 180 deg about (1,0,1)
#
# The rotation is a property of the PART FILE, not of the assembly: it says
# how that file's internal frame sits relative to the flange. It is valid for
# CAD/Step/fuegeteil_prt.stp and CAD/Step/greifer_v1_prt.stp from the 18:27
# run and for NO other part file in CAD/ -- the others were exported from
# different coordinate systems (see CAD/README.md).
#
# DO NOT APPLY THESE WHEN IMPORTING THE ASSEMBLY. The authoritative source is
# now the single-file assembly export CAD/Step/Greifer_Bauteil/
# greifer_bauteil_asm.stp (19:45), which carries BOTH solids already placed --
# the transforms below are baked into its geometry. Re-applying them on top
# would flip the part a second time and put it back upside down. They are kept
# here as the documented content of that placement, and for the case where
# someone converts a single part file on its own. The 19:45 export was checked
# against the file: 2 solids, structure preserved, GEOMETRIC_SET 0 (the datum
# wireframe is gone), and both transforms identical to the 18:27 run with the
# centroid cross-check matching.
#
# PART_MARGIN_END_TOOL_FRAME below is unaffected and stays "+Y": a rotation
# about Y leaves Y alone. Stated explicitly so it does not read as an
# oversight next to a 180 deg rotation.
PART_WELD_ROT_COLUMNS = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
GRIPPER_WELD_ROT_COLUMNS = ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))

# One angle is deliberately NOT a constant. An earlier reading of 179.978 deg
# -- 0.022 deg off antiparallel -- was taken against an UNDERSIDE FACE, not
# against the datum plane that reads 180.000. Different faces, no
# contradiction. It never enters the weld: it is part geometry and travels
# with the mesh through the STEP -> USD conversion for free.
#
# It is worth a sentence anyway, because it is NOT below the radar the way the
# lateral pair is, IF it is real. An angle grows with the lever it acts over:
# 0.022 deg is 0.019 mm over the 50.5 mm insertion depth and 0.055 mm over the
# part's 143 mm length. Against the 0.2 mm short-axis clearance that is a
# factor of 4 to 10, not three orders of magnitude. The consequence:
# FLANGE_TO_PART_BOTTOM above is only defined to about +/- 0.03 mm, since the
# face it measures to is tilted and the answer depends on where on the
# underside one lands. The success-depth threshold rests on that number, so
# the tolerance is stated rather than implied.

# Which way the part's asymmetry points, in the ASSEMBLY (tool) frame.
# Needed together with the cell-frame statement in the workcell block above
# ("the 7.9 mm-margin end of the pocket points to +Y", D-060): the tool axis
# points DOWN at insertion while the cell +Z points UP, so exactly one lateral
# axis is antiparallel between the two frames and "+Y" does NOT denote the
# same direction in both. Given both statements the insertion orientation is
# fully determined -- it is not a convention anyone gets to choose.
PART_MARGIN_END_TOOL_FRAME = "+Y"   # C: user, 2026-08-22.

# Part height along the insertion axis, derived from CAD alone -- the measured
# 152 mm cancels, since 152 - 126.747 is half the part height straight out of
# the part geometry.
#
# RESOLVED 2026-08-22 (user), was a three-way spread:
#   official manufacturer CAD (this line)  50.506 mm   <- leads
#   product datasheet, 43.5 + 7            50.5   mm
#   caliper on site 2026-08-17 (D-057)     50.7   mm   <- SUPERSEDED
# `CAD/fuegeteil.stp` is DEHN's own CAD of the part, not a redraw, and it
# agrees with the product datasheet to 6 um. D-057 already ranked CAD above
# measurement for the part; the 50.7 mm caliper reading is a measurement
# error and is no longer used anywhere. The gap mattered -- 0.194 mm is the
# same order as the 0.2 mm short-axis clearance the success criterion rests
# on -- which is why it is recorded here instead of quietly rounded.
PART_HEIGHT_Z = 2.0 * (FLANGE_TO_PART_BOTTOM - FLANGE_TO_PART_CENTRE)

# Body width across the short axis -- the number the lateral play is measured
# against. C: read out of CAD/Step/fuegeteil_prt.stp on 2026-08-28, the same
# way POCKET_WALL_X was read (all CARTESIAN_POINTs, SI_UNIT(.MILLI.,.METRE.)).
# Two dense point planes at -45.0 (138 points) and +45.0 (94 points).
# Fuegeteil_Neuer.stp and CAD/fuegeteil.stp give the same spans.
#
# The read carries its own CONTROL: on the long axis the same method returns
# -71.75 / +71.75, i.e. 143.50 mm, which is D-088's independently settled
# value. A method that reproduces the known axis is trusted on the unknown one.
#
# SUPERSEDES the 90.3 mm caliper reading of 2026-08-17 (D-057), and it is the
# THIRD number from that one measuring session to be overturned by CAD:
#   length  143.34 -> 143.50 (D-088)
#   height   50.7  ->  50.506 (D-073)
#   width    90.3  ->  90.00  (here)
# D-075 already had the evidence and did not use it: it recorded the mesh
# spanning x = -51.359 .. +45.050 about a part centre at +0.05, which puts the
# body half-width at 45.00 and the body at 90.00 -- then wrote "the plus side
# matches the 90.3 body" anyway.
#
# NOT derived as POCKET_OPENING_X - PLAY_X. That would be circular: the play
# is what you get by subtracting this width, not an input to it.
PART_BODY_X = 0.0900
"""Part body width across the short axis [m], between the lugs (C: part CAD).

The bounding box PART_BBOX_M[0] is WIDER (lugs included) and is a different
fact; the lugs enter the pocket's wall notches and never touch the opening."""

# PLAY_X / PLAY_Y are derived further down, after PART_BBOX_M supplies the
# long-axis width. They are not written here because PART_BBOX_M is defined
# below this point.

# The bellows compress under load and the cups deform; the free length is 6.6 mm
# longer than the gripped one. That difference is NOT modelled (D-070) -- the
# gripper is welded rigid and grasp variation enters as reset noise instead.
# Amplitude still open, hence the pending marker below.
GRASP_POS_NOISE: tuple[float, float, float] | None = None   # OPEN (D-070)

# Contact friction of the part/fixture pairing, D-181 (corrects D-111 (2)).
# PAIRING, an ASSUMPTION and not a measurement (D-181 (1)): PA6 for the part
# (the DEHNvap housing thermoplast -- datasheet 900 360 names no polymer)
# against POM for the fixture (the test-adapter station). Until D-181 the repo
# assumed ONE same thermoplast for both bodies. VALUE: the midpoint of the
# span of every DuPont PA66-on-POM primary value the research found
# (Zytel/Minlon Design Guide Module II, Tab. 39, p. 101; Delrin Design Guide,
# Tab. 9, p. 24). The band itself is `autodr.DR_DIMS` friction; this constant
# is its centre (D-181 (2)-(3)). NAMED GAP: every primary value is PA66, none
# is PA6 (D-181, Named gaps). Static = kinetic, one value per env. Applied at
# runtime to every material of the robot (the part is a welded link of it)
# and the kinematic fixture, Factory pattern (factory_utils.set_friction,
# Isaac Lab 2.3.2) -- no USD authors a material, so this constant is the
# single home of the nominal value, and it applies to every dr_mode='off' run
# too. Tables and block are bare static colliders without a PhysX view; the
# part never rests on them, so they keep the default material.
# The proxy-era FixedAssetCfg.friction = 0.75 (aluminium/steel) below is
# proxy DATA, never applied; it is not this fact.
CONTACT_FRICTION = 0.14

# The part's underside is NOT one plane (user, CAD inspection 2026-08-22): a
# SLOPED strip, a recess, and a strip PARALLEL to the seating plane. All three
# run ALONG THE LONG AXIS, so the underside is asymmetric across the SHORT
# axis -- not between the two ends of the long axis.
#
# Which of the two strips leads was open for one round and is now settled
# (user, 2026-08-22): FLANGE_TO_PART_BOTTOM = 0.1520 was measured to the
# PARALLEL strip, taken at its lowest edge. That strip is therefore the
# leading face and D-069's task frame is well defined after all. Two
# independent checks agree: a datasheet total height is quoted to the lowest
# point, and PART_HEIGHT_Z derived from the 152 mm matches it to 6 um.
#
# Worth carrying into D-071: the short axis is both the tighter one (0.2 mm
# clearance vs 0.5 mm, D-057) AND the axis the underside is asymmetric across.
# A tilt about the long axis therefore engages either the flat strip or the
# sloped one depending on its sign -- the two directions are not equivalent,
# which the tilted entry is free to exploit but must not be assumed to know.
# --- what the weld needs that the chain above does not carry ----------------
# Names, not prim paths (D-067: prim paths are never hard-coded). The authoring
# script finds these by name below the link it creates, the way
# author_peg_ur10e.py already finds the flange.
TOOL_LINK_NAME = "tool_link"
TOOL_JOINT_NAME = "tool_weld"
TOOL_REF_NAME = "tool"
TOOL_PART_PRIM_NAME = "FUEGETEIL"
TOOL_GRIPPER_PRIM_NAME = "GREIFER_V1"

# Rotation of the tool about the tool axis, at the weld. ZERO BY DEFINITION,
# not by measurement: the real mounting angle of the gripper on the flange is a
# statement about the CELL and is in no CAD file. Fixing it at zero here makes
# the tool frame and the flange frame share their yaw, so the real mounting
# angle appears as a constant offset in the wrist_3 home angle, where it can be
# measured against the cell and corrected in ONE number. Choosing the zero
# point of an angle is a convention; inventing its value would not be.
TOOL_WELD_YAW_RAD = 0.0

# ... and here is where that "ONE number" is now carried. This is the tool's
# yaw about its own axis AT THE HOME POSE, i.e. how the part is turned relative
# to the pocket before the policy does anything.
#
# 180 DEGREES since 2026-08-24 (evening), from LOOKING at the render: the
# part's lugs pointed away from the pocket features they have to enter (user
# screenshot, lugs circled on the tool and their mating cut-outs circled in the
# pocket). The part fits one way round, so this is not cosmetic.
#
# It is applied in the SCREENER's target orientation, not typed into the joint
# table, so the home pose stays a derived number and the next screener run
# reproduces it instead of undoing it.
#
# STILL A CONVENTION, not a measurement: D-078 defines the weld yaw as zero and
# says the real mounting angle of the gripper on the flange lives HERE. That
# angle is unmeasured. What is now known is that it is 180 degrees away from
# where it was assumed -- read off the geometry the part has to enter, which is
# better evidence than the zero it replaces, and worse than a measurement.
TOOL_HOME_YAW_RAD = math.pi

# Body centroids in the FLANGE frame, read out of the assembly STEP
# (`geometric validation property`, entities #41921 and #47542) on 2026-08-23.
# Millimetres there, metres here.
#
# These are centroids of VOLUME. Per body that is also the centre of mass --
# one body, one material -- so each may be used as a COM. The assembly-level
# centroid the same file states (#47583) may NOT: it weights the two bodies by
# volume, and they are not the same material. The combined COM has to be
# weighted by the two masses, which is what tool_com_flange() below does.
#
# Cross-check that made them trustworthy rather than merely present: pushing
# each body's own-frame centroid through its weld transform (D-074) reproduces
# the flange-frame value the file states, on all three axes, to 12 digits --
# 126.746951 + 1.774164 = 128.521115 for the part. That is the D-074
# transform confirmed a second time, through a quantity it was not derived
# from.
PART_CENTROID_FLANGE = (-0.001363926, -0.000587380, 0.128521115)
GRIPPER_CENTROID_FLANGE = (-0.000001749, -0.000059183, 0.055313338)


def tool_com_flange(part_mass: float, gripper_mass: float) -> tuple[float, float, float]:
    """Mass-weighted centre of mass of the welded tool, in the flange frame.

    The two masses are the caller's to supply and are not defaulted anywhere:
    the suction cups are missing from the CAD, so no density times volume can
    stand in for a weighing.
    """
    total = float(part_mass) + float(gripper_mass)
    if total <= 0.0:
        raise ValueError(f"tool mass must be positive, got {part_mass} + {gripper_mass}")
    return tuple(
        (float(part_mass) * PART_CENTROID_FLANGE[i]
         + float(gripper_mass) * GRIPPER_CENTROID_FLANGE[i]) / total
        for i in range(3)
    )


# Measured edges of the part prim in the imported USD (RT-12, 2026-08-23), the
# nominal the authoring script and the env's asset check compare against.
# Axis order is the flange frame: x across the lugs, y along the part, z the
# insertion axis. The z here is the MEASURED 50.514 mm, 8 um above the CAD
# PART_HEIGHT_Z of 50.506 -- that difference is the tessellation, and the two
# are kept apart on purpose rather than reconciled.
PART_BBOX_M = (0.096409, 0.143500, 0.050514)

# Lateral play, both axes: opening minus part. DERIVED, never written twice.
# Every D-106 bound divides by PLAY_X, so it gets exactly one home; if one of
# the inputs moves, these move with it instead of disagreeing with it.
#
# X uses PART_BODY_X (the body between the lugs), NOT PART_BBOX_M[0] -- the
# bounding box includes the lugs, which enter the pocket's wall notches and
# never pass through the opening. Y uses the bounding box, which on the long
# axis IS the body.
PLAY_X = POCKET_OPENING_X - PART_BODY_X
PLAY_Y = POCKET_OPENING_Y - PART_BBOX_M[1]

# --- The D-153 hole, and the offset at which the part reaches it ---
# MEASURED, not derived: RT-94 (git db72bae, judged in rt_logs/VERDICTS.md)
# located the fixture mesh's three unpaired edges as one ~1.5 mm triangle on
# the stage-1 shoulder at x 35.9863..37.4512, y 79.6347..79.7500, z 0.0000 mm.
# Only the NEAREST y edge matters for reachability, so that is the only one
# that gets a name here. D-153 owns the finding; this is its numeric home so
# the instrument below does not carry a second copy of it.
POCKET_HOLE_MIN_Y = 0.0796347       # M: RT-94, nearest y edge of the hole
# The part centre offset at which the part's +y face first touches the hole's
# y range. D-153 computes 7.8847 mm from exactly these two numbers; it is
# DERIVED here so it moves if either input moves.
HOLE_REACH_OFFSET_Y = POCKET_HOLE_MIN_Y - PART_BBOX_M[1] / 2.0
"""Lateral +y offset [m] at which the part reaches the D-153 hole: 0.0078847.

Read against what each pocket stage allows the part centre to do:
stage 2 allows PLAY_Y / 2 = 0.8 mm, so the seated state is clear by 7.0847 mm;
stage 1 allows POCKET_STAGE1_Y_RANGE[1] - PART_BBOX_M[1] / 2 = 8.7000 mm,
which is MORE than this offset. That gap is the open exposure D-153 names,
and `insertion_env._max_stage1_lat_y` is the instrument that measures it."""
"""Total lateral play [m]: X 0.0005876 (0.5876 mm), Y 0.00160 (1.60 mm).

X CHANGED on 2026-08-28, from 0.2876 mm, when PART_BODY_X was read out of the
part CAD -- see there. Everything derived from the cross play doubles with it:
the SAPU threshold, the fine kernel width, and the D-106 yaw and across-tilt
bounds.

REDONE 2026-08-30 by the concept stream. The sentence that stood here --
"those derivations live in D-106 / D-109 and are the concept stream's to redo;
this module only supplies the input" -- is no longer true of the first two: the
SAPU threshold and the fine kernel width are DECIDED and are derived from this
line, immediately below. The yaw and across-tilt bounds stay documentation of
what the walls enforce and are checked nowhere, so they get no constant.

Y is unaffected: 145.1 - 143.50, settled by D-088."""


# ===========================================================================
#  REWARD CONSTANTS DERIVED FROM THE PLAY AND THE BAND (D-106, D-109)
# ===========================================================================
#
# DECIDED 2026-08-30 by the concept stream, branch p1-konzept-messung, commit
# 77f5b3c, entries "The D-121 recomputation: a_fine = 5094, SAPU threshold =
# 0.2938 mm, sweep updated, the 2048 justification re-anchored" and
# "engaged_depth_m = 10.8 mm: 30 % of the stage-2 depth, Factory form, user-set
# fraction" in that branch's ``docs/decisions_inbox.md``. That file is the home
# of the reasoning; these lines are the numbers' one place in code.
#
# WHY DERIVED AND NOT TYPED. Each rule below is a RATIO or a FRACTION, and the
# decision says so in as many words -- "the rationale is a ratio, not a number".
# A typed 0.0002938 is right today and stops tracking the play the day
# PART_BODY_X or the opening is re-measured. Same reason POCKET_SEAT_DEPTH and
# SEATED_SUCCESS_DEPTH are derived rather than written out.
#
# ``scripts/check_insertion_math.py`` keeps its OWN literals for these (A_FINE,
# INTERPEN_THRESH, and the independent margin 0.0005876). That is deliberate:
# a check that imported these constants would assert them against themselves.

# D-106 (3): the SAPU threshold is HALF the play. Deeper than half the play is
# impossible from any regular pose even at perfect centering, so this is the
# largest bound that rejects only impossible poses. Feasibility was checked,
# not assumed: the Warp query has no minimum threshold, and the PhysX collider
# voxel pitch is cleared with a factor 3.2 (fixture) and 2.1 (part) at D-128's
# resolutions.
# [2026-09-15, inline CORRECTION, user] "Rejects only impossible poses" is an
# argument for WHERE the filter sits, not a licence: this number is the
# code's ALGORITHMIC acceptance threshold. Half the play does not make 0.2938 mm
# a physically admissible penetration -- the real pair admits none. Name it
# that way in every report (episodes_meta.json `t1_meaning`).
INTERPEN_THRESH = 0.5 * PLAY_X

# D-109 (9), the dm_control tolerance() form (Tassa et al. 2018):
# a = arccosh(1/0.1) / margin, so the kernel has fallen to 0.1 at the margin.
#
# WHAT A MARGIN IS, corrected 2026-09-01. dm_control's ``tolerance()`` takes
# ``bounds`` for the tolerance zone and ``margin`` for the DECAY REACH measured
# outward from that zone -- read off the source, recorded as
# [ADDENDUM 2026-09-01] in
# ``docs/reference/literature_check_reward_2026-08-26.md`` § 5. The COARSE
# kernel is the one that carries the reach of the whole approach; the two
# narrow ones stay tolerances.
#
#   coarse  173.21 mm  THE TRAVEL REACH, not a tolerance. It is the SDF
#                      distance MEASURED at the workcell home pose (RT-119,
#                      173.21 mm), so the shaping term still has a magnitude
#                      where an episode actually starts. a_coarse =
#                      acosh(10)/0.17321 = 17.28 /m. Typed, because no
#                      geometry constant carries a measured SDF distance.
#   mid       3 mm     the success band, D106_DEPTH_BAND.
#   fine      0.5876 mm the cross play, PLAY_X. Confirmed 2026-08-30 as still
#                      the TIGHTEST tolerance of the task (the long play is
#                      1.60 mm), which is what the D-109 (9) rule asks for.
#
# THE OLD 10 mm IS SUPERSEDED, and the story that went with it was wrong in
# its QUANTITY, not only in its size: it called the coarse margin "the start
# scatter", i.e. a tolerance. The start scatter is +-5 mm and the travel at
# rung 0 is 39.13 mm, so the margin was carrying neither. With 10 mm a start
# episode earned 0.0021 in total against a -1.0 abort -- factor 478 -- which
# is the degenerate "touch nothing" policy both failed PPO runs produced.
# Decided by the concept stream: inbox entry "Reward-Ueberarbeitung"
# 2026-09-01 (p1-konzept-messung), point (1). Mid and fine are untouched.
KERNEL_MARGIN_COARSE = 0.17321
KERNEL_A_COARSE = math.acosh(10.0) / KERNEL_MARGIN_COARSE
KERNEL_A_MID = math.acosh(10.0) / D106_DEPTH_BAND
KERNEL_A_FINE = math.acosh(10.0) / PLAY_X
KERNEL_B_COARSE = 2.0
KERNEL_B_MID = 2.0
KERNEL_B_FINE = 0.0

# THE TILT KERNEL of the alignment term (RT-171, 2026-09-06). The term is
# potential-based shaping, ``F = gamma * Phi(s') - Phi(s)`` with
# ``Phi = w_tilt * sech(a * theta)`` and ``theta`` the angle between the
# part's +z and the pocket's -z (seated = antiparallel, the minus sign of
# ``insertion_env._peg_geometry``'s ``alignment``). Same D-109 (9) rule as
# the three distance kernels: ``a = arccosh(10) / margin``, value 0.1 at the
# margin. ``sech(a x)`` is ``2 * squash(x, a, 0)``, so no new curve enters.
#
# THE MARGIN IS A MEASUREMENT, NOT CAD. 15.34 deg is RT-158's mouth-mode
# median of ``rot_beyond_yaw`` at episode end (``rt_logs/VERDICTS.md``
# 2026-09-06 15:05:29), i.e. the tilt the policy actually fails in. The CAD
# cone ``osc_tilt_clamp_rad`` = 8.52 deg was REJECTED as the margin (user,
# 2026-09-06): at 15.34 deg it leaves a pull of 0.018 w per 0.097 rad step,
# below the 0.08/step contact penalty measured there. At this margin the pull
# is 0.108 w at 15.34 deg and 0.370 w at 8.52 deg. One run, one median --
# the thesis states that (``InBachelorErwähnen.md``, 2026-09-06).
TILT_MARGIN_RAD = math.radians(15.34)
KERNEL_A_TILT = math.acosh(10.0) / TILT_MARGIN_RAD

# RUNG 0 OF THE D-110 LADDER: where the leading tool point starts.
#
# Signed like every other tip height in this module: measured from the STAGE-2
# OPENING PLANE, positive UP. WORKCELL_HOME_TIP_ABOVE_ENTRANCE = +0.165 m is
# the home pose; this one is NEGATIVE, i.e. the part starts already inside the
# pocket.
#
# WHY IT EXISTS (D-161, RT-119, measured): at the home pose the whole D-109
# reward is 3.05e-23. PPO cannot tell that from zero, so the policy trains on
# a flat landscape and learns only to avoid the -1.0 force abort. The kernel
# widths are NOT the fix -- D-109 (9) solves them from measured tolerances --
# so the START moves instead.
#
# USER-SET, 2026-09-01. NOT derived, and the reason it is not derived is the
# point: the number answers a question about the TASK, not about a tolerance.
# The part has to start clear of the fixture, because the strategy the user
# wants learned -- approach tilted, touch an edge, align on it, then push --
# happens entirely ABOVE the opening plane. A start inside the pocket skips it.
#
# 30 mm is 15 mm above the stage-1 top rim (STAGE1_DEPTH), i.e. the whole part
# hangs in free air with the fixture below it.
#
# SUPERSEDED ON THE SAME DAY, and the correction is the useful half:
# -(POCKET_SEAT_DEPTH - KERNEL_MARGIN_COARSE) = -26 mm, "one coarse kernel
# margin above the seat". (READ AS HISTORY: that arithmetic used the OLD
# coarse margin of 10 mm. Since 2026-09-01 the constant is 173.21 mm, so the
# expression no longer yields -26 mm -- it is quoted here for what was done,
# not as a live formula.) It was a literal reading of D-161's sentence and it
# put the part 26 mm INSIDE a 36 mm pocket. RT-120 measured what that does:
# force_abort_rate 1.0, force p50 93.5 N against a 60 N limit, every episode
# over after ~3.6 of 256 steps. The part was teleported into contact. Nothing
# could be learned, and the phases the task is about were skipped.
#
# WHAT THE REWARD IS WORTH HERE IS NOT MEASURED. RT-119's sweep read
# kernel_sum 1.46e-04 at +20 mm and 2.29e-08 at +50 mm; +30 mm lies between
# and no run has read it. Against the -1.0 abort payment that is small, which
# is D-161's mechanism all over again -- see the note in HANDOFF-RL § Open.
#
# 0.030 -> 0.020 ON 2026-09-14 (M): H_min MEASURED per D-179 (4). In the full
# AutoDR corner (lateral radius 30 mm, tilt 10 deg, yaw 5 deg, zero_agent.py,
# 256 envs) 20 mm is the lowest clean start: solver 0 dropped / 0 unconverged
# over 4096 episodes (RT-197a), reset force 0.000 N in every env (RT-197b).
# At 10 and 5 mm the tilted fixture pushes the part UP by 3 resp. 7 mm at the
# reset (RT-198b, RT-199b); at the centre 5 mm is clean (RT-199c), so the
# push is the corner's. Under dr_mode='off' this is also the UPPER edge of
# the start band, so a No-DR run without an override now starts at +20 mm
# (RT-191..194 ran with +30 mm). Home of the measurement:
# docs/decisions_inbox.md, entry "H_min is MEASURED" (2026-09-14).
RUNG0_START_TIP_ABOVE_ENTRANCE = 0.020

# D-109 (5): the depth at which the "engaged" display bonus starts paying.
#
# The FORM is the Isaac Lab factory task's -- engaged when the part has
# penetrated a FRACTION of the hole depth (``factory_env.py:343-373,466``,
# ``factory_tasks_cfg.py:84/132/395``), where Factory itself varies the
# fraction per task (0.9 for peg_insert, 0.5 for nut_thread).
#
# THE FRACTION IS A PLACEHOLDER AGAIN, 2026-09-01. It was set ad hoc by the
# user on 2026-08-30 ("from there the part can only fall in cleanly; before
# that it is not yet certain") and the concept stream has now named the RULE
# that has to replace it: the engaged depth is the MEASURED CAPTURE DEPTH of
# the tilted scripted insertion -- the depth at which the tilted part is
# upright and laterally captured, which is exactly the claim the 0.30 was
# making without a measurement behind it. Factory's own 0.9 was rejected
# (32.4 mm sits inside the success band). Inbox entry "Reward-Ueberarbeitung"
# 2026-09-01 (p1-konzept-messung), point (5); it closes D-109 (5)'s
# ``[CAD pending]`` by replacing it with ``[measure]``.
#
# THE VALUE BELOW STAYS 0.30 until that measurement exists, and it is a
# PLACEHOLDER while it does: 10.8 mm is a number nobody measured, and every
# run has to say so. Factory's FORM (a fraction of the hole depth,
# ``factory_env.py:343-373,466``, ``factory_tasks_cfg.py:84/132/395``) is what
# survives unchanged.
#
# Depth-only, no XY check (Factory additionally tests centering < 2.5 mm): at
# 0.5876 mm play the pocket walls enforce centering from this depth by
# themselves -- the implied-bounds argument of D-106 (2).
ENGAGED_DEPTH_FRACTION = 0.30
ENGAGED_DEPTH = ENGAGED_DEPTH_FRACTION * POCKET_SEAT_DEPTH

# The lug side DOES close. PART_BBOX_M[0] = 96.409 mm (imported USD, RT-12) is
# the span and stays the only home of it.
#
# D-123 opened a ~1.8 mm disagreement here because the part STEP's dense lug
# point cloud ends near +49.5 mm against a body edge at -45.0. It ends there
# because the lug is a CYLINDER -- R 4.203 mm about x -47.207 mm, measured
# 2026-08-29 in the Creo assembly STEP -- so its outermost point sits mid-arc
# and is neither a vertex nor a point in the cloud. Analytically the tip is at
# -51.410 mm and the span is 51.410 + 45.000 = 96.410 mm, which is 0.001 mm
# from the USD. There was no disagreement, only a reading method that cannot
# see the extreme point of a curved face. The remaining 0.051 mm between the
# analytic tip and the USD's -51.359 is the tessellation of that face.
#
# PART_LUG_SPAN_VS_BBOX stood here as a named unknown from 2026-08-28 and is
# CLOSED 2026-08-29 (user). No constant replaces it: the span already has a
# home in PART_BBOX_M[0], and a second number for the same edge is exactly what
# the one-home rule forbids. The lesson stays: a curved face's extreme point is
# invisible to any vertex or point-cloud query, so a span across a round
# feature is READ FROM THE SURFACE, never from points.

# The wrong-side test for the weld, in TOOL-LINK-LOCAL coordinates: the welded
# geometry starts at the flange face and runs away from the robot. Authored
# upside down it would span -0.152004 .. 0, which every size check would still
# call correct.
TOOL_SPAN_LOCAL_Z_M = (0.0, 0.152004)

TOOL_CHAIN_PENDING: tuple[str, ...] = ("GRASP_POS_NOISE", "TOOL_DIAGONAL_INERTIA")
"""Named unknowns of the tool chain, same contract as WORKCELL_PENDING: the
chain is internally consistent without them, but the grasp is deterministic
until GRASP_POS_NOISE is filled in. The startup report prints this tuple.

TOOL_DIAGONAL_INERTIA, added 2026-08-23: the weld writes mass and centre of
mass but leaves ``physics:diagonalInertia`` unauthored, so PhysX derives the
inertia from the collision geometry -- which is the PART only, since the
gripper deliberately carries no collider. The gripper's mass is therefore
distributed over the part's shape. Mass and COM are right; the inertia tensor
is an approximation, and it is named here rather than left to be discovered
from behaviour during training.

PART_FLIPPED_IN_PLACE was here from 2026-08-23 and is CLOSED 2026-08-28
(user): the part is welded the right way round -- the flange was turned by
180 degrees and the code has carried that correctly for some time. The blind
spot it named was real (RT-21 rotated the part about its own centre and none
of the twenty-one authoring checks moved, because an axis-aligned bounding
box is the same box either way), so the lesson stays even though the entry
goes: a wrong orientation is caught by a geometric FEATURE, never by another
bound.

PART_LUG_SPAN_VS_BBOX was here from 2026-08-28 and is CLOSED 2026-08-29
(user): the USD and the part STEP never disagreed. The lug is a cylinder, so
its outermost point lies mid-arc and no vertex or point-cloud query returns
it; read from the surface the span is 96.410 mm against the USD's 96.409 mm.
The note above PART_BBOX_M carries the numbers and the lesson."""
