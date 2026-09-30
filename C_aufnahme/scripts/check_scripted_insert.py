# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Offline check of ``scripted_policy.py`` -- the D-108 gate's pure math.

WHAT IT CHECKS
--------------
``scripted_policy.py`` turns the OBSERVATION into the env's joint-delta action
for the scripted insertion (D-108's gate). It imports only ``torch`` and
``insertion_math``, so this check executes the WHOLE file under the numpy
stand-in (``scripts/tools/torch_shim.py``) -- no Isaac, no torch, no GPU.

The frame chain is the reason this check exists in this shape. The command
travels pocket frame -> env frame -> robot base frame, and a swapped rotation
direction is the failure mode that stays plausible in review and only shows up
as "the arm drives sideways" on the training PC. Each hop therefore gets its
own check against a KNOWN rotation, never against a second copy of the same
expansion.

THE COUNTER-PROOF (D-080)
-------------------------
``--self-test`` breaks the source in exactly one way per mutation and requires
EXACTLY the named checks to flip. A mutation that flips nothing proves the
checks can never fail; one that flips extra checks proves they are not
surgical. Both are reported as failures of the CHECK, not of the math.

    python scripts/check_scripted_insert.py
    python scripts/check_scripted_insert.py --self-test
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import pathlib
import sys

_PKG = (
    pathlib.Path(__file__).resolve().parents[1]
    / "source"
    / "insertion"
    / "insertion"
    / "tasks"
    / "direct"
    / "insertion"
)
MATH_SRC = _PKG / "insertion_math.py"
POLICY_SRC = _PKG / "scripted_policy.py"

_SHIM_PATH = pathlib.Path(__file__).resolve().parent / "tools" / "torch_shim.py"
_spec = importlib.util.spec_from_file_location("torch_shim", _SHIM_PATH)
torch_shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(torch_shim)
torch, _IS_REAL_TORCH = torch_shim.load()
sys.modules.setdefault("torch", torch)

import numpy as np  # noqa: E402 - after the shim, which decides what torch is


# The one line that makes ``scripted_policy`` a package module. Executing the
# file standalone cannot run a relative import, so the import is removed and
# the two names are injected instead. Removing it by exact text rather than by
# regex: if the import line ever changes, this check must FAIL loudly rather
# than quietly stop injecting whatever was added to it.
RELATIVE_IMPORT = "from .insertion_math import axes_from_quat, rotate_into_frame"


def load_policy(mutations: tuple[tuple[str, str], ...] = ()) -> dict:
    """Execute both modules under the stand-in, optionally mutating the policy.

    Each mutation is ``(old, new)`` applied to the SOURCE TEXT and required to
    match exactly once. A mutation that silently matched nothing would make the
    counter-proof vacuous, which is the failure this harness exists to catch.
    """
    math_ns: dict = {"torch": torch, "math": math, "__name__": "insertion_math"}
    exec(compile(torch_shim.strip_jit_script(MATH_SRC.read_text(encoding="utf-8")),
                 str(MATH_SRC), "exec"), math_ns)  # noqa: S102

    src = POLICY_SRC.read_text(encoding="utf-8")
    if src.count(RELATIVE_IMPORT) != 1:
        raise SystemExit(
            f"{POLICY_SRC.name} no longer contains the expected relative import "
            f"{RELATIVE_IMPORT!r} exactly once -- this loader injects those names by hand "
            "and must be updated together with it."
        )
    src = src.replace(RELATIVE_IMPORT, "")
    for old, new in mutations:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"mutation {old!r} matches {n} times in {POLICY_SRC.name}, expected 1")
        src = src.replace(old, new)
    # Two views of the SAME mutated text, and the split is load-bearing.
    #
    # The AUDIT view keeps every decorator: the RT-66 structure check below
    # reads it, and a decorator is the thing it looks for. Stripping first
    # left it nothing to find, so in the baseline it was vacuously green
    # (found RT-108-gate3, 2026-08-31).
    #
    # The EXEC view has the decorators removed. Under real torch, jit.script
    # recompiles from the file on disk, so the mutation would never reach the
    # function (measured RT-108-gate, 2026-08-31); and the RT-66 mutation
    # deliberately puts a decorator on advance_phase, which under real torch
    # is the original import-time failure, not a check that flips.
    #
    # Stripping runs AFTER the mutations, so their counts still address the
    # real file.
    audit_src = src
    src = torch_shim.strip_jit_script(src)

    ns: dict = {
        "torch": torch,
        "math": math,
        "__name__": "scripted_policy",
        "axes_from_quat": math_ns["axes_from_quat"],
        "rotate_into_frame": math_ns["rotate_into_frame"],
    }
    exec(compile(src, str(POLICY_SRC), "exec"), ns)  # noqa: S102 - the point of the file
    # The MUTATED source travels with the namespace, decorators intact. The
    # structural check must see what this call built, not what happens to lie
    # on disk -- reading the file again would make its counter-proof vacuous,
    # which is the exact failure this harness exists to catch.
    ns["__src__"] = audit_src
    return ns


def module_level_constants(src: str) -> set[str]:
    """Names bound by an assignment at MODULE level."""
    tree = ast.parse(src)
    out: set[str] = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for t in targets:
            if isinstance(t, ast.Name):
                out.add(t.id)
    return out


def scripted_functions_closing_over_constants(src: str) -> list[str]:
    """Every ``@torch.jit.script`` function that reads a module-level constant.

    THIS CHECK EXISTS BECAUSE RT-66 FAILED ON IT. The run died at import with
    "python value of type 'float' cannot be used as a value. Perhaps it is a
    closed over global variable?" -- TorchScript compiles the decorated
    function eagerly and refuses to close over the module's phase codes. The
    defect was invisible to every other offline check, because the stand-in's
    ``torch.jit.script`` is a pass-through: it CANNOT reproduce the compiler.

    So this does not try to. It checks the STRUCTURE that torch rejects, which
    is decidable from the source alone and needs no torch at all. Function
    names are excluded deliberately -- TorchScript resolves a call to another
    scripted function fine; it is VALUES it will not close over.
    """
    tree = ast.parse(src)
    consts = module_level_constants(src)
    called_names = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    offenders: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorated = any(
            isinstance(d, ast.Attribute) and d.attr == "script" for d in node.decorator_list
        )
        if not decorated:
            continue
        local = {a.arg for a in node.args.args}
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id in consts and n.id not in local and n.id not in called_names:
                    offenders.append(f"{node.name} reads {n.id}")
                    break
    return offenders


def _arr(x):
    return np.asarray(x, dtype=float)


def _maxabs(a, b=0.0) -> float:
    return float(np.max(np.abs(_arr(a) - _arr(b))))


# ---------------------------------------------------------------------------
# PROBE VALUES. None of these is a decided task number -- the module owns no
# numbers and takes every gain as an argument. They are chosen to make each
# check's arithmetic readable by hand.
# ---------------------------------------------------------------------------
P_STANDOFF = 0.02        # m above the opening during ALIGN
P_LAT_GAIN = 1.0
P_VERT_GAIN = 1.0
P_YAW_GAIN = 1.0
P_DESCEND = 0.001        # m per control step
P_MAX_STEP = 0.005       # m per control step
P_MAX_YAW = 0.02         # rad per control step
P_LAT_TOL = 0.0005       # m
P_YAW_TOL = 0.01         # rad
P_HEIGHT_TOL = 0.001     # m, the ALIGN -> DESCEND height gate (RT-67)
P_F_ABORT = 50.0         # N -- the placeholder, used here only as a threshold


def build_checks(ns) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    def check(name: str, cond, detail: str = "") -> None:
        out.append((name, bool(cond), detail))

    latch_yaw_target = ns["latch_yaw_target"]
    yaw_error = ns["yaw_error"]
    rotate_out_of_frame = ns["rotate_out_of_frame"]
    rotate_by_quat = ns["rotate_by_quat"]
    rotate_by_quat_inv = ns["rotate_by_quat_inv"]
    axis_tilt_angle = ns["axis_tilt_angle"]
    advance_phase = ns["advance_phase"]
    update_peak_depth = ns["update_peak_depth"]
    command_in_pocket_frame = ns["command_in_pocket_frame"]
    command_to_base_frame = ns["command_to_base_frame"]
    joint_delta_to_action = ns["joint_delta_to_action"]
    bin_force_by_depth = ns["bin_force_by_depth"]
    bin_values_by_depth = ns["bin_values_by_depth"]
    force_depth_profile = ns["force_depth_profile"]
    depth_profile = ns["depth_profile"]
    lateral_error = ns["lateral_error"]
    signed_tilt_angle = ns["signed_tilt_angle"]
    latch_tilt_reference = ns["latch_tilt_reference"]
    tilt_since_reference = ns["tilt_since_reference"]
    tilt_offsets = ns["tilt_offsets"]
    tilt_schedule = ns["tilt_schedule"]
    command_in_pocket_frame_tilted = ns["command_in_pocket_frame_tilted"]
    first_depth_below = ns["first_depth_below"]
    ALIGN = ns["PHASE_ALIGN"]
    DESCEND = ns["PHASE_DESCEND"]
    ABORT = ns["PHASE_ABORT_FORCE"]

    # =====================================================================
    #  The latched yaw target (the assumption this module refuses to make)
    # =====================================================================
    cs_a = torch.tensor([[math.cos(0.3), math.sin(0.3)]])
    cs_b = torch.tensor([[math.cos(1.1), math.sin(1.1)]])
    unlatched = torch.tensor([[0.0]])
    already = torch.tensor([[1.0]])

    tgt0, lat0 = latch_yaw_target(cs_a, torch.tensor([[0.0, 0.0]]), unlatched)
    check("the first step latches the yaw it is given",
          _maxabs(tgt0, cs_a) < 1e-12,
          str(_arr(tgt0).round(6).tolist()))
    check("latching marks the env as latched",
          _maxabs(lat0, 1.0) < 1e-12)
    tgt1, _ = latch_yaw_target(cs_b, cs_a, already)
    check("a latched target survives a later yaw change",
          _maxabs(tgt1, cs_a) < 1e-12,
          str(_arr(tgt1).round(6).tolist()))

    # =====================================================================
    #  The yaw error
    # =====================================================================
    check("a part at its target reads zero yaw error",
          abs(float(_arr(yaw_error(cs_a, cs_a))[0])) < 1e-12)
    # +0.8 rad past the target, so the error is +0.8 and the command must be
    # negative. A sign flip here turns the yaw controller into a divergent one.
    check("the yaw error is signed, current minus target",
          abs(float(_arr(yaw_error(cs_b, cs_a))[0]) - 0.8) < 1e-9,
          f"{float(_arr(yaw_error(cs_b, cs_a))[0]):.9f}")
    # THE BRANCH CUT. Current +179 deg, target -179 deg. The true difference is
    # +358 deg, which wraps to -2 deg -- the part is two degrees SHORT of its
    # target the short way round. Subtracting two atan2 results would return
    # the raw +358 deg and send the wrist the long way, through the joint
    # limit. The sign matters as much as the magnitude here: a controller that
    # is handed +2 turns the wrong way.
    near_pi = torch.tensor([[math.cos(math.radians(179.0)), math.sin(math.radians(179.0))]])
    past_pi = torch.tensor([[math.cos(math.radians(-179.0)), math.sin(math.radians(-179.0))]])
    err_wrap = float(_arr(yaw_error(near_pi, past_pi))[0])
    check("the yaw error takes the SHORT way round the branch cut",
          abs(err_wrap - math.radians(-2.0)) < 1e-9,
          f"{math.degrees(err_wrap):.6f} deg")

    # =====================================================================
    #  The two frame directions, against KNOWN rotations
    # =====================================================================
    r2 = math.sqrt(0.5)
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_z90 = torch.tensor([[r2, 0.0, 0.0, r2]])
    v_x = torch.tensor([[1.0, 0.0, 0.0]])
    # A frame turned +90 deg about z: a vector along ITS x-axis points along
    # the parent's +y. That is the pocket -> env direction.
    check("rotate_by_quat maps the pocket frame OUT into its parent",
          _maxabs(rotate_by_quat(q_z90, v_x), _arr([[0.0, 1.0, 0.0]])) < 1e-12,
          str(_arr(rotate_by_quat(q_z90, v_x)).round(9).tolist()))
    check("rotate_by_quat_inv maps the parent INTO the pocket frame",
          _maxabs(rotate_by_quat_inv(q_z90, v_x), _arr([[0.0, -1.0, 0.0]])) < 1e-12,
          str(_arr(rotate_by_quat_inv(q_z90, v_x)).round(9).tolist()))
    # Round trip over a batch of DISTINCT rotations, fixed rather than random:
    # a check that cannot be re-run bit-for-bit is not evidence (cfg.seed rule).
    q_batch = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [r2, 0.0, r2, 0.0],
        [r2, 0.0, 0.0, r2],
        [0.0, 1.0, 0.0, 0.0],
    ])
    v_batch = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, -2.0, 0.5],
        [-0.25, 0.75, 1.5],
    ])
    round_trip = rotate_by_quat_inv(q_batch, rotate_by_quat(q_batch, v_batch))
    check("the two directions are exact inverses",
          _maxabs(round_trip, v_batch) < 1e-12,
          f"{_maxabs(round_trip, v_batch):.3e}")
    # rotate_out_of_frame against rotate_into_frame, which is the REUSED
    # function next door. Checked as transposes of one matrix, so a silent
    # copy of the wrong direction cannot pass.
    rot = ns["axes_from_quat"](q_batch)
    check("rotate_out_of_frame is the transpose of the reused rotate_into_frame",
          _maxabs(ns["rotate_into_frame"](rot, rotate_out_of_frame(rot, v_batch)), v_batch) < 1e-12)

    # =====================================================================
    #  The tilt between the part axis and the pocket axis (RT-70 instrument)
    # =====================================================================
    # KNOWN rotations, chosen so each one isolates one way the function can be
    # wrong. Angles are read back in degrees because that is how the run
    # reports them and how a wrong factor would be noticed.
    q_x30 = torch.tensor([[math.cos(math.radians(15.0)), math.sin(math.radians(15.0)), 0.0, 0.0]])
    q_y90 = torch.tensor([[r2, 0.0, r2, 0.0]])
    check("two identical orientations read zero tilt",
          abs(float(_arr(axis_tilt_angle(q_id, q_id))[0])) < 1e-7,
          f"{math.degrees(float(_arr(axis_tilt_angle(q_id, q_id))[0])):.9f} deg")
    # A YAW cannot be a tilt: turning about the shared z leaves both z-axes
    # parallel. This is the check that a yaw error cannot leak into the tilt
    # number and be read as a jam cause.
    check("a pure yaw difference is NOT a tilt",
          abs(float(_arr(axis_tilt_angle(q_z90, q_id))[0])) < 1e-7,
          f"{math.degrees(float(_arr(axis_tilt_angle(q_z90, q_id))[0])):.9f} deg")
    check("a 30 deg rotation about x reads 30 deg of tilt",
          abs(math.degrees(float(_arr(axis_tilt_angle(q_x30, q_id))[0])) - 30.0) < 1e-6,
          f"{math.degrees(float(_arr(axis_tilt_angle(q_x30, q_id))[0])):.6f} deg")
    check("a 90 deg rotation about y reads 90 deg of tilt",
          abs(math.degrees(float(_arr(axis_tilt_angle(q_y90, q_id))[0])) - 90.0) < 1e-6,
          f"{math.degrees(float(_arr(axis_tilt_angle(q_y90, q_id))[0])):.6f} deg")
    # SYMMETRIC in its two arguments: it is an angle between two directions,
    # so swapping which one is "the part" cannot change it. A version that
    # secretly used only one quaternion would pass everything above.
    check("the tilt is symmetric in its two arguments",
          _maxabs(axis_tilt_angle(q_x30, q_z90), axis_tilt_angle(q_z90, q_x30)) < 1e-12)
    # acos(1.0 + 1 ulp) is NaN. Identical unit axes are exactly the case that
    # produces it, and a NaN here would travel silently into the metrics file.
    tilt_batch = axis_tilt_angle(q_batch, q_batch)
    check("identical orientations never produce NaN",
          bool(_arr(torch.isfinite(tilt_batch)).all()) and _maxabs(tilt_batch, 0.0) < 1e-7,
          str(_arr(tilt_batch).round(9).tolist()))

    # =====================================================================
    #  The phase machine
    # =====================================================================
    # Five envs in ALIGN. The first four vary lateral/yaw at a height that is
    # already AT the standoff; the fifth is perfectly placed but still high up.
    # That fifth case is RT-67: the home pose was aligned to 1 um and 0 rad, so
    # without a height condition the gate opened 165 mm above the opening and
    # the fine descent owned the whole travel.
    ph_align = torch.tensor([ALIGN, ALIGN, ALIGN, ALIGN, ALIGN])
    lat_err = torch.tensor([0.0001, 0.0001, 0.0100, 0.0100, 0.0001])
    yaw_err = torch.tensor([0.0010, 0.5000, 0.0010, 0.5000, 0.0010])
    tip_z = torch.tensor([P_STANDOFF, P_STANDOFF, P_STANDOFF, P_STANDOFF, 0.1650])
    no_force = torch.zeros(5)
    nxt = _arr(advance_phase(ph_align, lat_err, yaw_err, tip_z, no_force,
                             P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT))
    check("ALIGN advances only when BOTH errors are inside tolerance",
          nxt.tolist()[:4] == [DESCEND, ALIGN, ALIGN, ALIGN],
          str(nxt.tolist()))
    check("a yaw error alone holds ALIGN",
          nxt[1] == ALIGN)
    check("a lateral error alone holds ALIGN",
          nxt[2] == ALIGN)
    # RT-67, as a check. A perfectly placed part that is still high up must
    # STAY in ALIGN, where the fast approach runs.
    check("a part that is aligned but still high stays in ALIGN",
          nxt[4] == ALIGN,
          f"tip_z {float(tip_z[4]):.4f} m vs standoff {P_STANDOFF:.4f} m -> {nxt[4]}")
    # ...and a tip already BELOW the standoff must descend, not climb back.
    below = _arr(advance_phase(
        torch.tensor([ALIGN]), torch.tensor([0.0001]), torch.tensor([0.0010]),
        torch.tensor([-0.0050]), torch.tensor([0.0]),
        P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT))
    check("a tip already below the standoff descends rather than climbing back",
          below.tolist() == [DESCEND], str(below.tolist()))
    # The abort fires from ALIGN and from DESCEND alike.
    ph_mixed = torch.tensor([ALIGN, DESCEND])
    hot = torch.tensor([P_F_ABORT, P_F_ABORT])
    aborted = _arr(advance_phase(ph_mixed, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]),
                                 torch.tensor([P_STANDOFF, P_STANDOFF]), hot,
                                 P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT))
    check("the force abort fires from any phase",
          aborted.tolist() == [ABORT, ABORT],
          str(aborted.tolist()))
    # STICKY: the force falls back to zero, the phase must not recover.
    stuck = _arr(advance_phase(torch.tensor([ABORT]), torch.tensor([0.0]), torch.tensor([0.0]),
                               torch.tensor([P_STANDOFF]), torch.tensor([0.0]),
                               P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT))
    check("an aborted env never leaves ABORT_FORCE",
          stuck.tolist() == [ABORT],
          str(stuck.tolist()))
    check("the abort limit is inclusive at exactly F_max",
          _arr(advance_phase(torch.tensor([DESCEND]), torch.tensor([0.0]), torch.tensor([0.0]),
                             torch.tensor([P_STANDOFF]), torch.tensor([P_F_ABORT]),
                             P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL,
                             P_F_ABORT)).tolist() == [ABORT])

    # --------------------------------------------------------------- H1 mode
    # ``allow_descend=False`` is the free-space run. The SAME inputs that
    # advance above must NOT advance here -- otherwise the run would touch the
    # fixture and the tracking number would be measuring a wall.
    free = _arr(advance_phase(ph_align, lat_err, yaw_err, tip_z, no_force,
                              P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT,
                              allow_descend=False))
    check("free space never leaves ALIGN",
          free.tolist() == [ALIGN] * 5,
          str(free.tolist()))
    # ...and the tip already BELOW the standoff, which descends above, must
    # hold too: that is the one case where the height condition alone would
    # not have stopped it.
    check("free space holds ALIGN even below the standoff",
          _arr(advance_phase(
              torch.tensor([ALIGN]), torch.tensor([0.0001]), torch.tensor([0.0010]),
              torch.tensor([-0.0050]), torch.tensor([0.0]),
              P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT,
              allow_descend=False)).tolist() == [ALIGN])
    # The abort is NOT disarmed by the mode. A free-space run that meets a
    # force has stopped being free space, and that has to be visible.
    # WELL ABOVE the limit, not exactly at it: whether the limit is inclusive
    # is a different claim with its own check, and one check must carry one.
    check("free space still aborts on force",
          _arr(advance_phase(torch.tensor([ALIGN]), torch.tensor([0.0]), torch.tensor([0.0]),
                             torch.tensor([P_STANDOFF]), torch.tensor([2.0 * P_F_ABORT]),
                             P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT,
                             allow_descend=False)).tolist() == [ABORT])

    # =====================================================================
    #  H1 settling (track_settle)
    # =====================================================================
    track_settle = ns["track_settle"]
    T_TOL = 0.001
    # Settles at index 1 and stays there.
    s = track_settle([0.005, 0.0002, 0.0002, 0.0002], T_TOL, 2)
    check("the settle index is the first step of the run that stays in tolerance",
          s["settle_step"] == 1, str(s))
    # THE LATE EXCURSION. Under tolerance at 0 and 1, out again at 2, back at
    # 3: the reading is 3, not 0. A forward "first step under tolerance" would
    # answer 0 and call a swinging chain settled.
    late = track_settle([0.0002, 0.0002, 0.005, 0.0002], T_TOL, 2)
    check("the settle index survives a late excursion",
          late["settle_step"] == 3, str(late))
    # Never in tolerance -> -1, not a step number.
    never = track_settle([0.005, 0.006, 0.007], T_TOL, 2)
    check("the settle index is -1 when the error never settles",
          never["settle_step"] == -1, str(never))
    # In tolerance from the very first step -> 0, which is a real answer and
    # must not collide with the "never" marker.
    check("a trace that is in tolerance from step 0 settles at 0",
          track_settle([0.0002, 0.0002], T_TOL, 2)["settle_step"] == 0)
    # THE TAIL IS A MAXIMUM. The last two entries are 0.0001 and 0.0009; their
    # mean is 0.0005 and would hide the 0.0009. The steady reading is 0.0009.
    tail = track_settle([0.02, 0.02, 0.0001, 0.0009], T_TOL, 2)
    check("the steady error takes the maximum of the tail, not its mean",
          abs(tail["steady_max_m"] - 0.0009) < 1e-12, str(tail))
    # The tail never reaches back past the start of the trace, and it reports
    # how many steps it actually used -- a tail wider than the run would
    # otherwise silently include the approach transient.
    check("the tail is clipped to the trace and reports its own width",
          track_settle([0.004, 0.001], T_TOL, 50)["tail"] == 2)
    # The tolerance is STRICT: an error exactly at the tolerance is not in it.
    check("an error exactly at the tolerance does not count as settled",
          track_settle([T_TOL], T_TOL, 1)["settle_step"] == -1)

    # =====================================================================
    #  Peak depth
    # =====================================================================
    peak = update_peak_depth(torch.tensor([0.004, 0.001]), torch.tensor([0.002, 0.003]))
    check("the peak depth is a running maximum, not the latest value",
          _maxabs(peak, _arr([0.004, 0.003])) < 1e-12,
          str(_arr(peak).round(6).tolist()))

    # =====================================================================
    #  Observation -> command, in the pocket frame
    # =====================================================================
    # Three envs: ALIGN with a lateral error, DESCEND on the axis, ABORTED.
    tip = torch.tensor([
        [0.0020, -0.0010, 0.0230],
        [0.0000, 0.0000, 0.0050],
        [0.0020, 0.0020, 0.0100],
    ])
    ph = torch.tensor([ALIGN, DESCEND, ABORT])
    ye = torch.tensor([0.5, 0.0, 0.5])
    no_aim = torch.zeros(3, 2)
    cmd = _arr(command_in_pocket_frame(
        tip, ph, ye, no_aim, P_STANDOFF, P_LAT_GAIN, P_VERT_GAIN, P_YAW_GAIN,
        P_DESCEND, P_MAX_STEP, P_MAX_YAW))
    check("the command is a 6-vector, translation then axis-angle",
          cmd.shape == (3, 6), str(cmd.shape))
    check("the lateral command OPPOSES the lateral error",
          cmd[0, 0] < 0.0 and cmd[0, 1] > 0.0,
          str(cmd[0, 0:2].round(6).tolist()))
    # ALIGN: the tip sits 23 mm up, the standoff is 20 mm, so it must come DOWN
    # by 3 mm -- the height ERROR, not the descend rate. Deliberately inside the
    # 5 mm step clip, so this check measures the regulation and the clip check
    # further down measures the clip; one probe testing both would leave either
    # defect able to hide behind the other.
    check("ALIGN regulates the height toward the standoff",
          abs(cmd[0, 2] - (P_STANDOFF - 0.0230)) < 1e-12,
          f"{cmd[0, 2]:.6f}")
    # DESCEND: a constant rate, and NEGATIVE, because tip_rel z decreases as
    # the part goes deeper (depth = -tip_rel_z).
    check("DESCEND commands the constant rate DOWNWARD",
          abs(cmd[1, 2] + P_DESCEND) < 1e-12,
          f"{cmd[1, 2]:.6f}")
    check("DESCEND ignores the standoff height entirely",
          abs(cmd[1, 2] - cmd[1, 2]) < 1e-12 and abs(cmd[1, 2]) == P_DESCEND)
    check("the yaw command OPPOSES the yaw error, about the pocket z-axis",
          cmd[0, 5] < 0.0 and abs(cmd[0, 3]) < 1e-12 and abs(cmd[0, 4]) < 1e-12,
          str(cmd[0, 3:6].round(6).tolist()))

    # ------------------------------------------------------- the seat stop
    # RT-128 is the reason: DESCEND had no end, so it commanded a constant
    # rate into the pocket FLOOR. 9 of 17 envs sat at 35.46-36.02 mm of a
    # 36.0 mm pocket while the force climbed from ~20 N to the 300 N abort,
    # and not one episode was clean. Three envs, all DESCEND, at 30 / 35 /
    # 36 mm of depth against a 35 mm stop -- above it, exactly on it, past it.
    _stop_tip = torch.tensor([
        [0.0000, 0.0000, -0.0300],
        [0.0000, 0.0000, -0.0350],
        [0.0020, 0.0020, -0.0360],
    ])
    _stop_ph = torch.tensor([DESCEND, DESCEND, DESCEND])
    _stop_ye = torch.tensor([0.0, 0.0, 0.5])
    _stop_args = (_stop_ph, _stop_ye, torch.zeros(3, 2), P_STANDOFF, P_LAT_GAIN,
                  P_VERT_GAIN, P_YAW_GAIN, P_DESCEND, P_MAX_STEP, P_MAX_YAW)
    _stopped = _arr(command_in_pocket_frame(_stop_tip, *_stop_args, 0.035))
    _no_stop = _arr(command_in_pocket_frame(_stop_tip, *_stop_args))

    check("above the seat stop the descent rate is unchanged",
          abs(_stopped[0, 2] + P_DESCEND) < 1e-12,
          f"{_stopped[0, 2]:.6f}")
    # INCLUSIVE at exactly the stop, the same rule the force abort follows at
    # exactly F_max. A part standing on the floor is seated, not one step
    # short of it.
    check("the seat stop is inclusive at exactly the stop depth",
          abs(_stopped[1, 2]) < 1e-12,
          f"{_stopped[1, 2]:.6f}")
    check("past the seat stop the descent stays zero",
          abs(_stopped[2, 2]) < 1e-12,
          f"{_stopped[2, 2]:.6f}")
    # OFF BY DEFAULT, and this is not politeness: scripted_insert.py and the
    # D-108 gate run call this function and must not change because
    # tilt_insert.py needed a stop.
    check("the seat stop is OFF by default, so the shared callers are unchanged",
          abs(_no_stop[0, 2] + P_DESCEND) < 1e-12
          and abs(_no_stop[1, 2] + P_DESCEND) < 1e-12
          and abs(_no_stop[2, 2] + P_DESCEND) < 1e-12,
          str(_no_stop[:, 2].round(6).tolist()))
    # A SEATED PART IS STILL HELD. Only the downward rate stops; dropping the
    # lateral and yaw terms too would leave the part to whatever the contact
    # does with it, which is the same defect the DESCEND bullet warns about.
    check("the seat stop leaves the lateral and yaw terms working",
          _stopped[2, 0] < 0.0 and _stopped[2, 1] < 0.0 and _stopped[2, 5] < 0.0,
          str(_stopped[2, [0, 1, 5]].round(6).tolist()))
    check("an aborted env commands exactly zero on all six components",
          float(np.max(np.abs(cmd[2]))) == 0.0,
          str(cmd[2].round(9).tolist()))
    # The clip. A 50 mm lateral error at gain 1.0 would ask for 50 mm in one
    # control step; the limit is 5 mm.
    big = torch.tensor([[0.0500, 0.0000, 0.5000]])
    cmd_big = _arr(command_in_pocket_frame(
        big, torch.tensor([ALIGN]), torch.tensor([3.0]), torch.zeros(1, 2), P_STANDOFF,
        P_LAT_GAIN, P_VERT_GAIN, P_YAW_GAIN, P_DESCEND, P_MAX_STEP, P_MAX_YAW))
    check("every translation component is clipped to the step limit",
          abs(cmd_big[0, 0] + P_MAX_STEP) < 1e-12 and abs(cmd_big[0, 2] + P_MAX_STEP) < 1e-12,
          str(cmd_big[0, 0:3].round(6).tolist()))
    check("the yaw component is clipped to its own step limit",
          abs(cmd_big[0, 5] + P_MAX_YAW) < 1e-12,
          f"{cmd_big[0, 5]:.6f}")

    # THE AIM, WITH A NON-ZERO OFFSET. Every probe above aims at the pocket
    # axis, where (aim - tip) and (-tip) are the same expression -- so a
    # controller that threw the aim away would pass all of them. That is the
    # WORSE defect of the two: the ALIGN gate still opens, the run looks
    # healthy, and the controller quietly pulls every env back to the centre,
    # so the sweep reports identical rows and reads as "the lateral aim does
    # not matter". Two envs, both sitting ON the pocket axis: one told to go
    # to +2 mm in x, one to -1 mm in y.
    on_axis = torch.tensor([[0.0, 0.0, 0.0230], [0.0, 0.0, 0.0230]])
    aim_probe = torch.tensor([[0.0020, 0.0000], [0.0000, -0.0010]])
    cmd_aim = _arr(command_in_pocket_frame(
        on_axis, torch.tensor([ALIGN, DESCEND]), torch.zeros(2), aim_probe,
        P_STANDOFF, P_LAT_GAIN, P_VERT_GAIN, P_YAW_GAIN,
        P_DESCEND, P_MAX_STEP, P_MAX_YAW))
    check("a part on the axis is driven TOWARD its aim, not held at the axis",
          cmd_aim[0, 0] > 0.0 and abs(cmd_aim[0, 0] - 0.0020 * P_LAT_GAIN) < 1e-12,
          str(cmd_aim[0, 0:2].round(6).tolist()))
    # DESCEND too, and that is the half that matters: dropping the lateral
    # term once the descent starts would place the part at the offset and then
    # leave it to whatever the contact does with it.
    check("DESCEND keeps holding the aim instead of releasing it",
          cmd_aim[1, 1] < 0.0 and abs(cmd_aim[1, 1] + 0.0010 * P_LAT_GAIN) < 1e-12,
          str(cmd_aim[1, 0:2].round(6).tolist()))

    # =====================================================================
    #  The frame chain, pocket -> env -> base
    # =====================================================================
    one = torch.tensor([[0.001, 0.002, -0.003, 0.0, 0.0, 0.05]])
    check("identity pocket and identity base leave the command unchanged",
          _maxabs(command_to_base_frame(one, q_id, q_id), one) < 1e-12)
    # Pocket turned +90 deg about z, base identity: pocket +x becomes env +y.
    only_x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    got = _arr(command_to_base_frame(only_x, q_z90, q_id))
    check("the pocket rotation is applied OUTWARD, pocket x -> env y",
          _maxabs(got[:, 0:3], _arr([[0.0, 1.0, 0.0]])) < 1e-12,
          str(got[0, 0:3].round(9).tolist()))
    # Base turned +90 deg about z, pocket identity: an env +x command must read
    # -y in the base frame. Opposite direction to the hop above -- that is what
    # separates the two rotations from each other.
    got_b = _arr(command_to_base_frame(only_x, q_id, q_z90))
    check("the base rotation is applied INWARD, env x -> base -y",
          _maxabs(got_b[:, 0:3], _arr([[0.0, -1.0, 0.0]])) < 1e-12,
          str(got_b[0, 0:3].round(9).tolist()))
    # Both hops at once must cancel, which no single-hop implementation does.
    both = _arr(command_to_base_frame(only_x, q_z90, q_z90))
    check("equal pocket and base rotations cancel, which one hop alone cannot",
          _maxabs(both[:, 0:3], _arr([[1.0, 0.0, 0.0]])) < 1e-12,
          str(both[0, 0:3].round(9).tolist()))
    # The axis-angle triple must rotate the SAME way as the translation.
    only_rz = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    q_y90 = torch.tensor([[r2, 0.0, r2, 0.0]])
    got_r = _arr(command_to_base_frame(only_rz, q_y90, q_id))
    check("the axis-angle triple rotates like a vector, pocket z -> env x",
          _maxabs(got_r[:, 3:6], _arr([[1.0, 0.0, 0.0]])) < 1e-12,
          str(got_r[0, 3:6].round(9).tolist()))
    check("the rotation hop does not leak into the translation triple",
          float(np.max(np.abs(got_r[:, 0:3]))) < 1e-12)

    # =====================================================================
    #  Joint targets -> the env's action
    # =====================================================================
    scale = 0.02
    q_des = torch.tensor([[0.010, -0.004, 0.000, 0.100, -0.100, 0.002]])
    q_tgt = torch.tensor([[0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])
    q_meas = torch.tensor([[0.005, 0.005, 0.005, 0.005, 0.005, 0.005]])
    act = _arr(joint_delta_to_action(q_des, q_tgt, scale))
    check("the action is the target difference over action_scale",
          abs(act[0, 0] - 0.5) < 1e-12 and abs(act[0, 1] + 0.2) < 1e-12,
          str(act[0, 0:2].round(6).tolist()))
    check("the action is clamped to [-1, 1]",
          abs(act[0, 3] - 1.0) < 1e-12 and abs(act[0, 4] + 1.0) < 1e-12,
          str(act[0, 3:5].round(6).tolist()))
    # THE ONE THAT MATTERS. Differencing against the MEASURED position instead
    # of the integrator target would command the tracking error a second time.
    # With a non-zero measured position the two routes must disagree.
    act_meas = _arr(joint_delta_to_action(q_des, q_meas, scale))
    check("the action differences against the TARGET, not the measured position",
          float(np.max(np.abs(act - act_meas))) > 1e-6,
          f"max |diff| = {float(np.max(np.abs(act - act_meas))):.6f}")

    # =====================================================================
    #  Pocket command -> the env's OSC action (inbox entry "Audit 2026-09-03 (a)")
    # =====================================================================
    osc_act = ns["osc_action_from_pocket_command"]
    _R2 = math.sqrt(0.5)
    _q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    _q_z90 = torch.tensor([[_R2, 0.0, 0.0, _R2]])
    _cmd = torch.tensor([[0.002, -0.001, -0.0005, 0.01, 0.0, 0.0]])
    _a = _arr(osc_act(_cmd, _q_id, 0.02, 0.097))
    check("the OSC action is the pocket delta over the step limits",
          abs(_a[0, 0] - 0.1) < 1e-12 and abs(_a[0, 1] + 0.05) < 1e-12
          and abs(_a[0, 3] - 0.01 / 0.097) < 1e-12,
          str(_a[0].round(6).tolist()))
    _a90 = _arr(osc_act(_cmd, _q_z90, 0.02, 0.097))
    check("the OSC action rotates the translation pocket -> env (z90: x -> y)",
          abs(_a90[0, 0] - 0.05) < 1e-12 and abs(_a90[0, 1] - 0.1) < 1e-12,
          str(_a90[0, 0:3].round(6).tolist()))
    check("the rotation triple rotates pocket -> env too",
          abs(_a90[0, 3]) < 1e-12 and abs(_a90[0, 4] - 0.01 / 0.097) < 1e-12,
          str(_a90[0, 3:6].round(6).tolist()))
    _big = _arr(osc_act(torch.tensor([[1.0, -1.0, 0.0, 5.0, 0.0, 0.0]]), _q_id, 0.02, 0.097))
    check("the OSC action is clamped to [-1, 1]",
          abs(_big[0, 0] - 1.0) < 1e-12 and abs(_big[0, 1] + 1.0) < 1e-12 and abs(_big[0, 3] - 1.0) < 1e-12,
          str(_big[0].round(6).tolist()))

    # =====================================================================
    #  What torch will actually compile (RT-66)
    # =====================================================================
    offenders = scripted_functions_closing_over_constants(ns["__src__"])
    check("no jit-scripted function closes over a module constant",
          not offenders,
          "; ".join(offenders) if offenders else "none")

    # =====================================================================
    #  Lateral error, and the AIM it is measured against (RT-75 instrument)
    # =====================================================================
    le = lateral_error(torch.tensor([[0.003, 0.004, 9.9]]), torch.zeros(1, 2))
    check("the lateral error is planar and ignores the vertical component",
          abs(float(_arr(le)[0]) - 0.005) < 1e-12,
          f"{float(_arr(le)[0]):.9f}")
    # THE DEFECT THIS GUARDS. The ALIGN -> DESCEND gate compares this against
    # lateral_tol (0.2 mm). Measured against the pocket AXIS instead of the
    # aim, a part sitting exactly on a 0.29 mm sweep offset would read a
    # permanent 0.29 mm error and the gate would never open -- the run would
    # produce depth 0.0 everywhere and look like a physics result.
    le_aim = lateral_error(torch.tensor([[0.003, 0.004, 9.9]]),
                           torch.tensor([[0.003, 0.004]]))
    check("a part sitting exactly on its aim reads zero lateral error",
          abs(float(_arr(le_aim)[0])) < 1e-12,
          f"{float(_arr(le_aim)[0]):.12f}")

    # The AIM SWEEP itself.
    aim_offsets = ns["aim_offsets"]
    aim_sweep_rows = ns["aim_sweep_rows"]

    # A NON-ZERO span on purpose. Probed at span 0.0 this check would pass for
    # any axis mapping at all, and "none" quietly sweeping the short axis would
    # be invisible -- every baseline run would sweep, and RT-75a against RT-74
    # would compare two different experiments.
    a_none = _arr(aim_offsets(4, "none", 0.6))
    check("aim_sweep none is all zeros -- it must reproduce RT-74 exactly",
          a_none.shape == (4, 2) and float(np.max(np.abs(a_none))) == 0.0,
          str(a_none.round(9).tolist()))
    # Span 0.6 over 5 envs: -0.30, -0.15, 0.00, +0.15, +0.30. Chosen so the
    # endpoints and the centre are exact in binary and readable by hand.
    a_x = _arr(aim_offsets(5, "x", 0.6))
    check("the aim spans exactly +-span/2, endpoints INCLUDED",
          abs(a_x[0, 0] + 0.3) < 1e-12 and abs(a_x[4, 0] - 0.3) < 1e-12,
          str(a_x[:, 0].round(6).tolist()))
    check("the aim is spread linearly across the envs",
          abs(a_x[1, 0] + 0.15) < 1e-12 and abs(a_x[2, 0]) < 1e-12,
          str(a_x[:, 0].round(6).tolist()))
    # THE DEFECT THIS GUARDS: an offset written to both columns sweeps the
    # DIAGONAL of the play while the log still says "x", and the 1.60 mm long
    # axis would silently ride along with the 0.5876 mm short one.
    check("sweeping x leaves the long axis at zero",
          float(np.max(np.abs(a_x[:, 1]))) == 0.0,
          str(a_x[:, 1].round(9).tolist()))
    a_y = _arr(aim_offsets(5, "y", 0.6))
    check("sweeping y leaves the short axis at zero",
          float(np.max(np.abs(a_y[:, 0]))) == 0.0,
          str(a_y.round(6).tolist()))
    check("sweeping y moves the long axis",
          abs(a_y[4, 1] - 0.3) < 1e-12,
          str(a_y[:, 1].round(6).tolist()))
    # ONE env cannot carry a sweep, and the linear spread divides by n-1.
    a_one = _arr(aim_offsets(1, "x", 0.6))
    check("a single env aims at the axis instead of dividing by zero",
          a_one.shape == (1, 2) and float(np.max(np.abs(a_one))) == 0.0,
          str(a_one.round(9).tolist()))

    # The table skeleton. Three envs whose aim order is NOT their index order,
    # and env 1 finishes nothing. Read through a dict rather than by position:
    # a mutation that drops a row must FLIP a check, not raise IndexError, and
    # a crashing mutation proves nothing about which check was watching.
    rows = aim_sweep_rows([[0.3, 0.0], [-0.3, 0.0], [0.0, 0.0]], [0, 2, 0])
    by_env = {r["env"]: r for r in rows}
    check("every env gets a row, including one that finished no episode",
          sorted(by_env) == [0, 1, 2] and by_env.get(1, {}).get("episodes") == [],
          str([(r["env"], r["aim_x_mm"], r["episodes"]) for r in rows]))
    check("the rows are sorted by the aim, not by the env index",
          [r["aim_x_mm"] for r in rows] == sorted(r["aim_x_mm"] for r in rows),
          str([(r["env"], r["aim_x_mm"]) for r in rows]))
    check("each row carries the indices of ITS OWN episodes",
          by_env.get(0, {}).get("episodes") == [0, 2]
          and by_env.get(2, {}).get("episodes") == [1],
          str([(r["env"], r["episodes"]) for r in rows]))

    # =====================================================================
    #  Force against depth (RT-74 instrument)
    # =====================================================================
    # THE DEFECT THIS GUARDS is not an arithmetic one. The tip starts 165 mm
    # ABOVE the opening, so most of every episode sits at a negative depth
    # carrying only the tool weight. Clamping those into bin 0 -- the obvious
    # way to write a binner -- buries the first real contact under hundreds
    # of free-air samples, which is the one thing the instrument exists to
    # find. Same for a sample past the last bin: piling it on invents a peak
    # at the deepest bin.
    _c, _mx, _sm = [0] * 4, [0.0] * 4, [0.0] * 4
    bin_force_by_depth(
        [-0.165, -0.001, 0.0005, 0.0015, 0.0015, 0.0099],
        [10.0, 11.0, 3.0, 5.0, 7.0, 999.0],
        0.001, _c, _mx, _sm,
    )
    check("free-air samples above the opening never reach bin 0",
          _c[0] == 1 and abs(_mx[0] - 3.0) < 1e-12,
          f"bin0 n={_c[0]} max={_mx[0]}")
    check("a sample past the last bin is dropped, not piled onto it",
          _c[3] == 0 and _mx[3] == 0.0,
          f"bin3 n={_c[3]} max={_mx[3]}")
    check("the binner keeps the max and the sum per bin",
          _c[1] == 2 and abs(_mx[1] - 7.0) < 1e-12 and abs(_sm[1] - 12.0) < 1e-12,
          f"bin1 n={_c[1]} max={_mx[1]} sum={_sm[1]}")
    # ITS OWN FIXTURE, not the accumulators above. Sharing them coupled this
    # check to the deep-drop one: piling a deep sample onto the last bin also
    # makes that bin non-empty, so one mutation flipped two checks and neither
    # of them said anything specific any more.
    _rows = force_depth_profile(0.001, [0, 2, 0, 5], [0.0, 7.0, 0.0, 9.0], [0.0, 12.0, 0.0, 20.0])
    check("the profile drops empty bins and keeps each row's own depth range",
          len(_rows) == 2
          and abs(_rows[0]["depth_lo_mm"] - 1.0) < 1e-12
          and abs(_rows[0]["force_mean_n"] - 6.0) < 1e-12
          and abs(_rows[1]["depth_lo_mm"] - 3.0) < 1e-12
          and abs(_rows[1]["force_mean_n"] - 4.0) < 1e-12,
          f"{len(_rows)} rows")

    # =====================================================================
    #  The tilted entry (2026-09-01). D-071's edge-first path, now scripted
    #  because three placeholders name the tilted run as their source:
    #  force_abort_f_max_n, engaged_depth_m and abort_payment.
    # =====================================================================
    # Quaternions built by hand and wxyz, the project convention: a rotation by
    # theta about a unit axis is (cos(theta/2), axis * sin(theta/2)).
    _th = 0.20
    _c, _s = math.cos(_th / 2.0), math.sin(_th / 2.0)
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_rx = torch.tensor([[_c, _s, 0.0, 0.0]])
    q_ry = torch.tensor([[_c, 0.0, _s, 0.0]])
    q_rx_neg = torch.tensor([[_c, -_s, 0.0, 0.0]])

    # THE SIGN IS THE RESULT, not a detail: the part's underside is a sloped
    # strip on one side and a parallel strip on the other, so the two tilt
    # directions meet different geometry. axis_tilt_angle above cannot say
    # which one a run was in.
    check("a +theta rotation about pocket x reads +theta on axis 0",
          abs(float(signed_tilt_angle(q_rx, q_id, 0)[0]) - _th) < 1e-9,
          f"{float(signed_tilt_angle(q_rx, q_id, 0)[0]):.9f} rad")
    check("a +theta rotation about pocket y reads +theta on axis 1",
          abs(float(signed_tilt_angle(q_ry, q_id, 1)[0]) - _th) < 1e-9,
          f"{float(signed_tilt_angle(q_ry, q_id, 1)[0]):.9f} rad")
    check("the signed tilt flips sign with the rotation, where axis_tilt_angle cannot",
          abs(float(signed_tilt_angle(q_rx_neg, q_id, 0)[0]) + _th) < 1e-9
          and abs(float(axis_tilt_angle(q_rx_neg, q_id)[0]) - _th) < 1e-9,
          f"signed {float(signed_tilt_angle(q_rx_neg, q_id, 0)[0]):.6f} "
          f"unsigned {float(axis_tilt_angle(q_rx_neg, q_id)[0]):.6f}")
    # A TILT IS RELATIVE TO THE POCKET, never to the world. With a tilted
    # fixture (fixture_tilt_noise_rad, D-037) a world reading would command the
    # part to stand up straight in a pocket that is not, which is the one
    # failure a tilt controller must not have.
    check("a part tilted WITH the pocket reads zero tilt",
          abs(float(signed_tilt_angle(q_rx, q_rx, 0)[0])) < 1e-9,
          f"{float(signed_tilt_angle(q_rx, q_rx, 0)[0]):.9f} rad")

    # ------------------------------------------- the latched tilt reference
    # RT-127 (training PC, git e38e687) is the whole reason this block exists:
    # the tool starts ANTIPARALLEL to the pocket axis, signed p50 -180.0000 deg
    # in all 16 envs, so a controller steering against the pocket axis carries
    # a permanent 180 deg error, the 0.2 deg gate never opens and nothing
    # descends. The reference has to be latched, exactly as the yaw target is.
    _t_ref = torch.tensor([-math.pi, 0.30])
    _t_new = torch.tensor([1.10, 1.10])
    _t_unlatched = torch.tensor([[0.0], [0.0]])
    _t_already = torch.tensor([[1.0], [1.0]])
    _ref_cs = torch.tensor([[math.cos(-math.pi), math.sin(-math.pi)],
                            [math.cos(0.30), math.sin(0.30)]])

    _r0, _l0 = latch_tilt_reference(_t_ref, torch.tensor([[0.0, 0.0], [0.0, 0.0]]), _t_unlatched)
    check("the first step latches the tilt it is given",
          _maxabs(_r0, _ref_cs) < 1e-12,
          str(_arr(_r0).round(6).tolist()))
    check("latching the tilt marks the env as latched",
          _maxabs(_l0, 1.0) < 1e-12)
    _r1, _ = latch_tilt_reference(_t_new, _ref_cs, _t_already)
    check("a latched tilt reference survives a later tilt change",
          _maxabs(_r1, _ref_cs) < 1e-12,
          str(_arr(_r1).round(6).tolist()))

    # ZERO AT THE REFERENCE, and the first env is the MEASURED case: exactly
    # -180 deg, on the branch cut. A controller that reads anything but zero
    # here is the RT-124/125 failure.
    check("a part at its latched tilt reference reads zero relative tilt",
          _maxabs(tilt_since_reference(_t_ref, _ref_cs), 0.0) < 1e-9,
          str(_arr(tilt_since_reference(_t_ref, _ref_cs)).round(9).tolist()))
    # Signed, current minus reference: +0.8 rad past the reference in env 1.
    check("the relative tilt is signed, current minus reference",
          abs(float(_arr(tilt_since_reference(torch.tensor([-math.pi, 1.10]), _ref_cs))[1]) - 0.8) < 1e-9,
          f"{float(_arr(tilt_since_reference(torch.tensor([-math.pi, 1.10]), _ref_cs))[1]):.9f}")
    # THE BRANCH CUT, and here it is not a precaution. The reference sits at
    # -180 deg, so a part leaning two degrees the OTHER way reads +178 deg.
    # The short way round is -2 deg; an unwrapped subtraction returns +358 deg
    # and sends the wrist all the way round, through the joint limit.
    _cut = tilt_since_reference(torch.tensor([math.radians(178.0), 0.30]), _ref_cs)
    check("the relative tilt takes the SHORT way round the branch cut",
          abs(float(_arr(_cut)[0]) - math.radians(-2.0)) < 1e-9,
          f"{math.degrees(float(_arr(_cut)[0])):.6f} deg")
    # THE RT-127 IDENTITY, in one line: the start attitude reads 180 deg
    # against the pocket axis and 0 deg against itself. Both numbers are
    # correct and only the second one is a tilt the controller may act on.
    _q_flip = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    _abs_flip = signed_tilt_angle(_q_flip, q_id, 0)
    _ref_flip, _ = latch_tilt_reference(_abs_flip, torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0]]))
    check("an antiparallel start reads 180 deg absolute and 0 deg relative",
          abs(abs(float(_arr(_abs_flip)[0])) - math.pi) < 1e-9
          and abs(float(_arr(tilt_since_reference(_abs_flip, _ref_flip))[0])) < 1e-9,
          f"absolute {math.degrees(float(_arr(_abs_flip)[0])):.4f} deg, "
          f"relative {math.degrees(float(_arr(tilt_since_reference(_abs_flip, _ref_flip))[0])):.9f} deg")

    # ------------------------------------------------------------ the ramp
    _start = torch.tensor([0.10, -0.10])
    _above = tilt_schedule(torch.tensor([-0.165, -0.001]), _start, 0.008)
    check("above the opening the full start tilt is held, however high",
          abs(float(_above[0]) - 0.10) < 1e-12 and abs(float(_above[1]) + 0.10) < 1e-12,
          f"{[round(float(v), 6) for v in _above.tolist()]}")
    # TWO DIFFERENT depths, not one: at a single midpoint a ramp that runs the
    # WRONG way (upright at the opening, tilted at the seat) reads exactly the
    # same as the right one, and the check would be blind to the one defect
    # that jams every episode at depth.
    _half = tilt_schedule(torch.tensor([0.002, 0.006]), _start, 0.008)
    check("the ramp FALLS with depth: three quarters at a quarter of the way down",
          abs(float(_half[0]) - 0.075) < 1e-12 and abs(float(_half[1]) + 0.025) < 1e-12,
          f"{[round(float(v), 6) for v in _half.tolist()]}")
    # AT the upright depth and BELOW it, zero. Below matters on its own: the
    # walls enforce 0.935 deg across at full seat (D-106 (2)), so a schedule
    # that kept commanding a tilt down there would be fighting the pocket and
    # the force it produced would be the schedule's, not the task's.
    _deep = tilt_schedule(torch.tensor([0.008, 0.033]), _start, 0.008)
    check("at and below the upright depth the commanded tilt is zero",
          abs(float(_deep[0])) < 1e-12 and abs(float(_deep[1])) < 1e-12,
          f"{[round(float(v), 9) for v in _deep.tolist()]}")
    check("an upright depth of zero is the straight controller, not a division",
          _maxabs(tilt_schedule(torch.tensor([-0.1, 0.02]), _start, 0.0)) == 0.0)

    # ---------------------------------------------------------- the sweep
    _off = tilt_offsets(5, -0.10, 0.10)
    check("the tilt sweep includes both endpoints and the straight case",
          abs(float(_off[0]) + 0.10) < 1e-12
          and abs(float(_off[4]) - 0.10) < 1e-12
          and abs(float(_off[2])) < 1e-12,
          f"{[round(float(v), 6) for v in _off.tolist()]}")
    check("one env carries no sweep and does not divide by zero",
          abs(float(tilt_offsets(1, -0.10, 0.10)[0]) + 0.10) < 1e-12)

    # ------------------------------------------------- the tilted command
    _tip = torch.tensor([[0.001, -0.002, 0.005], [0.000, 0.000, -0.010]])
    _ph = torch.tensor([DESCEND, ABORT])
    _ye = torch.tensor([0.01, 0.01])
    _aim0 = torch.zeros(2, 2)
    _terr = torch.tensor([0.01, 0.01])
    _base = command_in_pocket_frame(
        _tip, _ph, _ye, _aim0, P_STANDOFF, P_LAT_GAIN, P_VERT_GAIN, P_YAW_GAIN,
        P_DESCEND, P_MAX_STEP, P_MAX_YAW,
    )
    _tlt = command_in_pocket_frame_tilted(
        _tip, _ph, _ye, _aim0, _terr, 1, P_STANDOFF, P_LAT_GAIN, P_VERT_GAIN, P_YAW_GAIN,
        P_DESCEND, P_MAX_STEP, P_MAX_YAW, 1.0, 0.02,
    )
    # REUSE, NOT A COPY. If the tilted command ever stops agreeing with the
    # base one outside its own column, the lateral, height and yaw terms have
    # acquired a second home -- which is how a descend rate ends up wrong in
    # one run and right in the other.
    check("the tilted command leaves translation and yaw exactly as the base command",
          _maxabs(_arr(_tlt)[:, 0:3], _arr(_base)[:, 0:3]) < 1e-12
          and _maxabs(_arr(_tlt)[:, 5], _arr(_base)[:, 5]) < 1e-12,
          f"max deviation {_maxabs(_arr(_tlt)[:, 0:3], _arr(_base)[:, 0:3]):.3e}")
    check("the tilt command UNDOES the error and lands in the named column only",
          abs(float(_tlt[0, 4]) + 0.01) < 1e-12 and abs(float(_tlt[0, 3])) < 1e-12,
          f"col3 {float(_tlt[0, 3]):.6f} col4 {float(_tlt[0, 4]):.6f}")
    check("an aborted env commands no tilt either",
          abs(float(_tlt[1, 3])) < 1e-12 and abs(float(_tlt[1, 4])) < 1e-12,
          f"col3 {float(_tlt[1, 3]):.6f} col4 {float(_tlt[1, 4]):.6f}")
    _clip = command_in_pocket_frame_tilted(
        _tip, torch.tensor([DESCEND, DESCEND]), _ye, _aim0, torch.tensor([1.0, -1.0]), 0,
        P_STANDOFF, P_LAT_GAIN, P_VERT_GAIN, P_YAW_GAIN,
        P_DESCEND, P_MAX_STEP, P_MAX_YAW, 1.0, 0.02,
    )
    check("the commanded tilt is clipped in both directions",
          abs(float(_clip[0, 3]) + 0.02) < 1e-12 and abs(float(_clip[1, 3]) - 0.02) < 1e-12,
          f"{float(_clip[0, 3]):.6f} / {float(_clip[1, 3]):.6f}")

    # --------------------------------------- the generic binner and profile
    # The wrapper must DELEGATE. Two implementations of the drop rules is the
    # defect this pair exists to prevent, and "they agree today" is the only
    # way to see it.
    _ca, _ma, _sa = [0] * 4, [0.0] * 4, [0.0] * 4
    _cb, _mb, _sb = [0] * 4, [0.0] * 4, [0.0] * 4
    _dd = [-0.165, 0.0005, 0.0015, 0.0015, 0.0099]
    _vv = [10.0, 3.0, 5.0, 7.0, 999.0]
    bin_force_by_depth(_dd, _vv, 0.001, _ca, _ma, _sa)
    bin_values_by_depth(_dd, _vv, 0.001, _cb, _mb, _sb)
    check("the force binner is the generic binner, not a second copy of it",
          _ca == _cb and _ma == _mb and _sa == _sb,
          f"{_ca} vs {_cb}")
    _prow = depth_profile(0.001, [0, 2], [0.0, 7.0], [0.0, 12.0], "interpen", "mm")
    check("a profile row NAMES its quantity and its unit",
          "interpen_max_mm" in _prow[0] and "interpen_mean_mm" in _prow[0]
          and abs(_prow[0]["interpen_mean_mm"] - 6.0) < 1e-12,
          f"{sorted(_prow[0])}")
    check("the force profile keeps its original keys, so old metrics files still read",
          "force_max_n" in force_depth_profile(0.001, [2], [7.0], [12.0])[0])

    # ------------------------------------------------ the capture read-off
    # BACKWARDS, the track_settle rule. A quantity that dips under the
    # tolerance once and leaves it again has not been captured -- and reading
    # it forwards would name that dip as the capture depth, which is the
    # engaged_depth_m number this run exists to produce.
    _cap = [
        {"depth_lo_mm": 0.0, "v": 5.0},
        {"depth_lo_mm": 1.0, "v": 0.1},
        {"depth_lo_mm": 2.0, "v": 4.0},
        {"depth_lo_mm": 3.0, "v": 0.1},
        {"depth_lo_mm": 4.0, "v": 0.1},
    ]
    check("the capture depth is read BACKWARDS, so an early dip does not count",
          first_depth_below(_cap, "v", 1.0) == 3.0,
          f"{first_depth_below(_cap, 'v', 1.0)}")
    check("a quantity that never settles has no capture depth",
          first_depth_below([{"depth_lo_mm": 0.0, "v": 5.0}], "v", 1.0) is None)
    check("an empty profile has no capture depth and does not raise",
          first_depth_below([], "v", 1.0) is None)

    return out


# ---------------------------------------------------------------------------
# THE COUNTER-PROOF (D-080). One mutation, one named set of checks that MUST
# flip. Each mutation is a defect that could plausibly be written by hand.
# ---------------------------------------------------------------------------
MUTATIONS: tuple[tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]], ...] = (
    (
        # THE OBVIOUS WAY TO WRITE THE BINNER, and it destroys the
        # measurement: clamp instead of drop. Every free-air sample above the
        # opening -- most of every episode -- lands in bin 0, so the first
        # real contact is buried under hundreds of tool-weight readings and
        # the profile reports a busy bin 0 that means nothing.
        "free-air-depths-clamped-into-the-first-bin",
        ((
            "        if depth < 0.0:\n            continue",
            "        if depth < 0.0:\n            depth = 0.0",
        ),),
        ("free-air samples above the opening never reach bin 0",),
    ),
    (
        # The other half of the same mistake, at the deep end: a sample past
        # the last bin piled onto it invents a peak exactly where the run is
        # trying to find one.
        "deep-samples-piled-onto-the-last-bin",
        ((
            "        if b >= n_bins:\n            continue",
            "        if b >= n_bins:\n            b = n_bins - 1",
        ),),
        ("a sample past the last bin is dropped, not piled onto it",),
    ),
    (
        # Empty bins kept, by counting an unvisited bin as one sample. The
        # table becomes forty rows of zeros with the handful that matter lost
        # in them -- a report nobody reads is a measurement nobody has. Written
        # this way and not as a plain 'keep the empty rows', because that
        # divides by zero and a mutation that CRASHES proves nothing about
        # which check was watching.
        "empty-bins-kept-in-the-profile",
        ((
            "        if n == 0:\n            continue",
            "        if n == 0:\n            n = 1",
        ),),
        (
            "the profile drops empty bins and keeps each row's own depth range",
            # SHARED HOME, added 2026-09-01: force_depth_profile and the tilt
            # run's three profiles are one function now, so the empty-bin rule
            # can only be broken in one place -- and it breaks both readings.
            "a profile row NAMES its quantity and its unit",
        ),
    ),
    (
        # THE AIM MEASURED AGAINST THE POCKET AXIS instead of against the aim
        # -- the shape lateral_error had before RT-75, and the one a hand
        # reverts to. A part sitting exactly on a 0.29 mm offset then reads a
        # permanent 0.29 mm error against a 0.2 mm gate, so ALIGN -> DESCEND
        # never fires and the whole sweep returns depth 0.0.
        "lateral-error-measured-against-the-axis-not-the-aim",
        ((
            "    return torch.linalg.norm(tip_rel[:, 0:2] - aim, dim=-1)",
            "    return torch.linalg.norm(tip_rel[:, 0:2], dim=-1)",
        ),),
        ("a part sitting exactly on its aim reads zero lateral error",),
    ),
    (
        # The command steering to the axis while the ERROR is measured against
        # the aim. This is the worse half: the gate opens, the run looks
        # healthy, and the controller quietly pulls every env back to the
        # centre -- so the sweep reports sixteen identical rows and reads as
        # "the lateral aim does not matter", which is exactly the wrong
        # conclusion to draw from a controller that ignored it.
        "command-steers-to-the-axis-and-throws-the-aim-away",
        ((
            "    lateral = torch.clamp((aim - tip_rel[:, 0:2]) * lateral_gain, "
            "-max_step_m, max_step_m)",
            "    lateral = torch.clamp(-tip_rel[:, 0:2] * lateral_gain, "
            "-max_step_m, max_step_m)",
        ),),
        ("a part on the axis is driven TOWARD its aim, not held at the axis",
         "DESCEND keeps holding the aim instead of releasing it"),
    ),
    (
        # Half the span. The sweep then never reaches the wall and reports a
        # flat curve over the middle of the play -- the "lateral aim is not
        # the cause" outcome, produced by an instrument that never tested it.
        "aim-sweep-covers-only-half-the-play",
        ((
            "            rows[i][col] = -span_m / 2.0 + span_m * i / (num_envs - 1)",
            "            rows[i][col] = -span_m / 4.0 + span_m * i / (2 * (num_envs - 1))",
        ),),
        ("the aim spans exactly +-span/2, endpoints INCLUDED",
         "the aim is spread linearly across the envs",
         "sweeping y moves the long axis"),
    ),
    (
        # The offset written to BOTH columns: the sweep runs the diagonal of
        # the play while the log still says "x", so the 1.60 mm long axis
        # rides along with the 0.5876 mm short one and neither is measured.
        "aim-offset-written-to-both-axes",
        ((
            "            rows[i][col] = -span_m / 2.0 + span_m * i / (num_envs - 1)",
            "            rows[i][0] = -span_m / 2.0 + span_m * i / (num_envs - 1)\n"
            "            rows[i][1] = -span_m / 2.0 + span_m * i / (num_envs - 1)",
        ),),
        ("sweeping x leaves the long axis at zero",
         "sweeping y leaves the short axis at zero"),
    ),
    (
        # "none" falling through to the x column. Every baseline run would then
        # sweep silently, and the one comparison that proves the refactor did
        # not move the physics -- RT-75a against RT-74 -- would be against a
        # different experiment.
        "aim-sweep-none-falls-through-to-x",
        ((
            'AIM_AXES: dict = {"none": -1, "x": 0, "y": 1}',
            'AIM_AXES: dict = {"none": 0, "x": 0, "y": 1}',
        ),),
        ("aim_sweep none is all zeros -- it must reproduce RT-74 exactly",),
    ),
    (
        # THE FREE-SPACE SWITCH IGNORED. The flag is accepted, the log says
        # "free space", and the part descends into the fixture anyway -- so
        # the tracking number H1 reads was produced against a wall and reads
        # as the control chain's own precision. A defect that makes a run look
        # like the run it is not.
        "free-space-mode-still-descends",
        ((
            "    if not allow_descend:\n        aligned = torch.zeros_like(aligned)",
            "    if not allow_descend:\n        pass",
        ),),
        ("free space never leaves ALIGN",
         "free space holds ALIGN even below the standoff"),
    ),
    (
        # THE FORWARD READING of the settle index -- "first step under the
        # tolerance", which is what a hand writes first. A chain that dips
        # into tolerance at step 20, swings back out at 200 and returns at 210
        # then reports 20, and the swing that H1 exists to find disappears
        # into a number that says the chain settled early.
        "settle-index-accepts-a-later-excursion",
        ((
            "    settle_step = 0\n"
            "    for i in range(n - 1, -1, -1):\n"
            "        if not errors[i] < tol_m:\n"
            "            settle_step = i + 1\n"
            "            break",
            "    settle_step = -1\n"
            "    for i in range(n):\n"
            "        if errors[i] < tol_m:\n"
            "            settle_step = i\n"
            "            break",
        ),),
        (
            "the settle index survives a late excursion",
            # SHARED HOME, added 2026-09-01: first_depth_below reads the
            # capture depth through track_settle, so the backwards rule has
            # one implementation and one mutation breaks both readings.
            "the capture depth is read BACKWARDS, so an early dip does not count",
        ),
    ),
    (
        # The tail read as a MEAN. Two steps at 0.0001 and 0.0009 average to
        # 0.0005, which passes a 0.2938 mm gate that the 0.0009 excursion
        # would also pass -- but at a real oscillation the mean halves the
        # amplitude and turns a MISS into a TRACKS. The play has to
        # accommodate the worst placement, not the average one.
        "steady-error-uses-the-mean",
        ((
            '"steady_max_m": max(window)',
            '"steady_max_m": sum(window) / len(window)',
        ),),
        ("the steady error takes the maximum of the tail, not its mean",),
    ),
    (
        # An env that finished no episode dropped from the table. The row
        # vanishes, so a sweep of sixteen offsets prints fifteen and reads as
        # a complete curve -- the missing offset looks like one nobody asked
        # for rather than like an env that never terminated.
        "envs-without-an-episode-dropped-from-the-table",
        ((
            '    rows.sort(key=lambda r: (r["aim_x_mm"], r["aim_y_mm"], r["env"]))\n    return rows',
            '    rows.sort(key=lambda r: (r["aim_x_mm"], r["aim_y_mm"], r["env"]))\n'
            '    return [r for r in rows if r["episodes"]]',
        ),),
        ("every env gets a row, including one that finished no episode",),
    ),
    (
        # Sorted by env index instead of by aim. The numbers are all correct
        # and the curve is scrambled -- a jam that depends on the offset would
        # print as noise.
        "aim-table-sorted-by-env-instead-of-by-aim",
        ((
            '    rows.sort(key=lambda r: (r["aim_x_mm"], r["aim_y_mm"], r["env"]))',
            '    rows.sort(key=lambda r: r["env"])',
        ),),
        ("the rows are sorted by the aim, not by the env index",),
    ),
    (
        # The tilt is read off the WRONG column: x instead of z. The part's
        # x-axis is where the yaw lives, so this version reports a yaw as a
        # tilt -- and the yaw is the one thing already known to be zero, so
        # the number would read plausible and mean nothing.
        "tilt-measured-on-the-x-axis-instead-of-the-tool-axis",
        (("    part_axis = axes_from_quat(part_quat)[:, :, 2]",
          "    part_axis = axes_from_quat(part_quat)[:, :, 0]"),),
        (
            # ALL SIX. Reading the wrong column does not bend one case, it
            # measures a different quantity, so nothing about the tilt
            # survives it. The list is written out in full because a shorter
            # one would claim the other checks are independent of the column.
            "two identical orientations read zero tilt",
            "a pure yaw difference is NOT a tilt",
            "a 30 deg rotation about x reads 30 deg of tilt",
            "a 90 deg rotation about y reads 90 deg of tilt",
            "the tilt is symmetric in its two arguments",
            "identical orientations never produce NaN",
            # SEVEN since 2026-09-01: the signed reading is checked AGAINST
            # the unsigned one in the same line, so the wrong column breaks
            # that comparison too. Kept as one check on purpose -- what the
            # signed reading adds is only meaningful next to what the
            # unsigned one cannot say.
            "the signed tilt flips sign with the rotation, where axis_tilt_angle cannot",
        ),
    ),
    (
        # The clamp goes, because "the dot product of two unit vectors is in
        # [-1, 1] anyway". It is, in exact arithmetic. In floating point the
        # identical-axis case lands a few ulps past 1.0 and acos returns NaN --
        # and a NaN travels into the metrics file without stopping anything.
        "tilt-drops-the-clamp-before-acos",
        (("    return torch.acos(torch.clamp(cos, min=-1.0, max=1.0))",
          "    return torch.acos(cos)"),),
        ("identical orientations never produce NaN",),
    ),
    (
        # Only ONE quaternion is really used: the pocket is assumed upright.
        # True today (fixture_tilt_rad = 0.0) and false the moment D-037's
        # per-episode tilt is switched on -- the silent kind of wrong.
        "tilt-assumes-the-pocket-stands-upright",
        (("    pocket_axis = axes_from_quat(pocket_quat)[:, :, 2]",
          "    pocket_axis = torch.zeros_like(part_axis) + torch.tensor([0.0, 0.0, 1.0])"),),
        (
            "the tilt is symmetric in its two arguments",
            # Also this one, and it is the more telling of the two: q_batch
            # holds five DISTINCT orientations, so an assumed-upright pocket
            # no longer reads zero against them.
            "identical orientations never produce NaN",
        ),
    ),
    (
        "latch-overwrites-every-step",
        ((    "new_target = torch.where(latched > 0.5, target_cs, yaw_cs)",
              "new_target = yaw_cs"),),
        ("a latched target survives a later yaw change",),
    ),
    (
        "latch-never-marks-itself",
        (("    return new_target, torch.ones_like(latched)",
          "    return new_target, torch.zeros_like(latched)"),),
        ("latching marks the env as latched",),
    ),
    (
        "yaw-error-unwrapped-subtraction",
        (("    return torch.atan2(sin_a * cos_b - cos_a * sin_b, cos_a * cos_b + sin_a * sin_b)",
          "    return torch.atan2(sin_a, cos_a) - torch.atan2(sin_b, cos_b)"),),
        (
            "the yaw error takes the SHORT way round the branch cut",
            # SHARED HOME since 2026-09-01: the tilt reads its difference
            # through the same function, so one mutation breaks both callers.
            # That is the property the split was made for -- if this line ever
            # stops flipping the tilt check, the tilt has grown its own copy.
            "the relative tilt takes the SHORT way round the branch cut",
        ),
    ),
    (
        "yaw-error-sign-flipped",
        (("    return torch.atan2(sin_a * cos_b - cos_a * sin_b, cos_a * cos_b + sin_a * sin_b)",
          "    return -torch.atan2(sin_a * cos_b - cos_a * sin_b, cos_a * cos_b + sin_a * sin_b)"),),
        (
            "the yaw error is signed, current minus target",
            "the yaw error takes the SHORT way round the branch cut",
            "the relative tilt is signed, current minus reference",
            "the relative tilt takes the SHORT way round the branch cut",
        ),
    ),
    (
        "rotate-out-sums-the-wrong-axis",
        (("    return torch.sum(rot * v.reshape(-1, 1, 3), dim=2)",
          "    return torch.sum(rot * v.reshape(-1, 3, 1), dim=1)"),),
        (
            "rotate_by_quat maps the pocket frame OUT into its parent",
            "rotate_out_of_frame is the transpose of the reused rotate_into_frame",
            "the two directions are exact inverses",
            "the pocket rotation is applied OUTWARD, pocket x -> env y",
            "equal pocket and base rotations cancel, which one hop alone cannot",
            "the axis-angle triple rotates like a vector, pocket z -> env x",
            "the OSC action rotates the translation pocket -> env (z90: x -> y)",
            "the rotation triple rotates pocket -> env too",
        ),
    ),
    (
        "the-two-frame-hops-both-go-inward",
        (("    return rotate_out_of_frame(axes_from_quat(quat), vec)",
          "    return rotate_into_frame(axes_from_quat(quat), vec)"),),
        (
            "rotate_by_quat maps the pocket frame OUT into its parent",
            "the two directions are exact inverses",
            "the pocket rotation is applied OUTWARD, pocket x -> env y",
            "equal pocket and base rotations cancel, which one hop alone cannot",
            "the axis-angle triple rotates like a vector, pocket z -> env x",
            "the OSC action rotates the translation pocket -> env (z90: x -> y)",
            "the rotation triple rotates pocket -> env too",
        ),
    ),
    (
        # RT-67, put back exactly: the gate that ignores the height.
        "the-align-gate-forgets-the-height-again",
        (("    aligned = torch.logical_and(placed, at_height)",
          "    aligned = placed"),),
        (
            "a part that is aligned but still high stays in ALIGN",
        ),
    ),
    (
        "the-height-gate-is-a-band-instead-of-a-ceiling",
        (("    at_height = tip_z <= standoff_m + height_tol",
          "    at_height = torch.abs(tip_z - standoff_m) < height_tol"),),
        ("a tip already below the standoff descends rather than climbing back",),
    ),
    (
        "align-advances-on-either-error",
        (("    placed = torch.logical_and(lateral_err < lateral_tol, torch.abs(yaw_err) < yaw_tol)",
          "    placed = torch.logical_or(lateral_err < lateral_tol, torch.abs(yaw_err) < yaw_tol)"),),
        (
            "ALIGN advances only when BOTH errors are inside tolerance",
            "a yaw error alone holds ALIGN",
            "a lateral error alone holds ALIGN",
        ),
    ),
    (
        # The stickiness lives in the ``phase == PHASE_ALIGN`` gate: drop it and
        # an aborted env whose errors happen to be small marches back into
        # DESCEND, and the jam disappears from the statistics.
        "the-align-transition-forgets-which-phase-it-is-in",
        (("        torch.logical_and(phase == PHASE_ALIGN, aligned),",
          "        aligned,"),),
        ("an aborted env never leaves ABORT_FORCE",),
    ),
    (
        "the-abort-limit-is-exclusive",
        (("    aborting = force_norm >= force_abort_n",
          "    aborting = force_norm > force_abort_n"),),
        (
            "the force abort fires from any phase",
            "the abort limit is inclusive at exactly F_max",
        ),
    ),
    (
        "peak-depth-keeps-the-latest",
        (("    return torch.maximum(depth, peak)", "    return depth"),),
        ("the peak depth is a running maximum, not the latest value",),
    ),
    (
        "the-descend-rate-points-up",
        (("    sink = torch.full_like(hold, -descend_rate_m)",
          "    sink = torch.full_like(hold, descend_rate_m)"),),
        (
            "DESCEND commands the constant rate DOWNWARD",
            # The seat-stop checks ride the SAME code path, so they break
            # here too. Declared and not trimmed: a flip set edited to
            # look tidy stops proving anything.
            "above the seat stop the descent rate is unchanged",
            "the seat stop is OFF by default, so the shared callers are unchanged",
        ),
    ),
    (
        "the-lateral-term-follows-the-error",
        (("    lateral = torch.clamp((aim - tip_rel[:, 0:2]) * lateral_gain, -max_step_m, max_step_m)",
          "    lateral = torch.clamp((tip_rel[:, 0:2] - aim) * lateral_gain, -max_step_m, max_step_m)"),),
        (
            "the lateral command OPPOSES the lateral error",
            "every translation component is clipped to the step limit",
            "a part on the axis is driven TOWARD its aim, not held at the axis",
            "DESCEND keeps holding the aim instead of releasing it",
            # The seat-stop checks ride the SAME code path, so they break
            # here too. Declared and not trimmed: a flip set edited to
            # look tidy stops proving anything.
            "the seat stop leaves the lateral and yaw terms working",
        ),
    ),
    (
        "the-yaw-term-follows-the-error",
        (("    d_yaw = torch.clamp(-yaw_err * yaw_gain, -max_yaw_step_rad, max_yaw_step_rad)",
          "    d_yaw = torch.clamp(yaw_err * yaw_gain, -max_yaw_step_rad, max_yaw_step_rad)"),),
        (
            "the yaw command OPPOSES the yaw error, about the pocket z-axis",
            "the yaw component is clipped to its own step limit",
            # The seat-stop checks ride the SAME code path, so they break
            # here too. Declared and not trimmed: a flip set edited to
            # look tidy stops proving anything.
            "the seat stop leaves the lateral and yaw terms working",
        ),
    ),
    (
        "an-aborted-env-keeps-pushing",
        (("    active = phase != PHASE_ABORT_FORCE", "    active = phase == phase"),),
        ("an aborted env commands exactly zero on all six components",),
    ),
    (
        "the-translation-clip-is-dropped",
        (("    hold = torch.clamp(height_err * vertical_gain, -max_step_m, max_step_m)",
          "    hold = height_err * vertical_gain"),),
        ("every translation component is clipped to the step limit",),
    ),
    (
        "align-and-descend-are-swapped",
        (("    dz = torch.where(phase == PHASE_DESCEND, sink, hold)",
          "    dz = torch.where(phase == PHASE_DESCEND, hold, sink)"),),
        (
            "ALIGN regulates the height toward the standoff",
            "DESCEND commands the constant rate DOWNWARD",
            "DESCEND ignores the standoff height entirely",
            "every translation component is clipped to the step limit",
            # The seat-stop checks ride the SAME code path, so they break
            # here too. Declared and not trimmed: a flip set edited to
            # look tidy stops proving anything.
            "above the seat stop the descent rate is unchanged",
            "past the seat stop the descent stays zero",
            "the seat stop is inclusive at exactly the stop depth",
            "the seat stop is OFF by default, so the shared callers are unchanged",
        ),
    ),
    (
        "the-base-hop-is-skipped",
        (("    lin_base = rotate_by_quat_inv(root_quat, rotate_by_quat(pocket_quat, command_pocket[:, 0:3]))",
          "    lin_base = rotate_by_quat(pocket_quat, command_pocket[:, 0:3])"),),
        (
            "the base rotation is applied INWARD, env x -> base -y",
            "equal pocket and base rotations cancel, which one hop alone cannot",
        ),
    ),
    (
        "the-rotation-triple-is-left-in-the-pocket-frame",
        (("    rot_base = rotate_by_quat_inv(root_quat, rotate_by_quat(pocket_quat, command_pocket[:, 3:6]))",
          "    rot_base = command_pocket[:, 3:6]"),),
        ("the axis-angle triple rotates like a vector, pocket z -> env x",),
    ),
    (
        "the-action-differences-against-the-measured-position",
        (("    return torch.clamp((joint_pos_des - joint_targets) / action_scale, -1.0, 1.0)",
          "    return torch.clamp((joint_pos_des - joint_targets * 0.0) / action_scale, -1.0, 1.0)"),),
        ("the action differences against the TARGET, not the measured position",),
    ),
    (
        "the-action-is-not-clamped",
        (("(joint_pos_des - joint_targets) / action_scale, -1.0, 1.0)",
          "(joint_pos_des - joint_targets) / action_scale, -99.0, 99.0)"),),
        ("the action is clamped to [-1, 1]",),
    ),
    (
        # The other direction of the same check: keep the decorators the file
        # really has, and make one of THEM read a module constant. Without
        # this, nothing proves the check reads the 10 genuine decorators --
        # it was vacuously green until RT-108-gate3 (2026-08-31), because the
        # audited source had every decorator stripped out of it first.
        "a-really-scripted-function-starts-reading-a-module-constant",
        (("    return torch.maximum(depth, peak)",
          "    return torch.maximum(depth, peak) + PHASE_ALIGN"),),
        ("no jit-scripted function closes over a module constant",),
    ),
    (
        # The RT-66 defect, put back exactly as it was written the first time.
        "a-jit-decorator-returns-to-a-function-that-reads-the-phase-codes",
        (("# NOT ``@torch.jit.script``, and the reason is MEASURED, not stylistic.",
          "@torch.jit.script  # NOT ``@torch.jit.script``, and it is MEASURED, not stylistic."),),
        ("no jit-scripted function closes over a module constant",),
    ),
    (
        "the-lateral-error-includes-the-height",
        (("    return torch.linalg.norm(tip_rel[:, 0:2] - aim, dim=-1)",
          "    return torch.linalg.norm(\n"
          "        tip_rel - torch.cat((aim, torch.zeros_like(tip_rel[:, 2:3])), dim=-1),\n"
          "        dim=-1)"),),
        (
            "the lateral error is planar and ignores the vertical component",
            "a part sitting exactly on its aim reads zero lateral error",
        ),
    ),
    # ----------------------------------------------------- the tilted entry
    (
        # The sign convention written the obvious way and therefore backwards.
        # A rotation by +theta about pocket +x carries the part axis to
        # (0, -sin theta, cos theta), so the minus belongs there. Without it
        # the controller drives the part FURTHER over instead of upright, and
        # the run reports a jam that is the instrument's, not the task's.
        "the-signed-tilt-loses-its-sign-convention-about-x",
        (("        return torch.atan2(-a[:, 1], a[:, 2])",
          "        return torch.atan2(a[:, 1], a[:, 2])"),),
        (
            "a +theta rotation about pocket x reads +theta on axis 0",
            "the signed tilt flips sign with the rotation, where axis_tilt_angle cannot",
        ),
    ),
    (
        # The tilt measured in the WORLD instead of in the pocket. Invisible
        # while the fixture stands upright -- and every tilt-noise run
        # (D-037 -- whose "5 deg then 15 deg" ladder D-178 superseded; the
        # Phase-5 tilt ceiling is 10 deg, one-sided) would then command the
        # part to stand straight in a pocket that is not.
        "the-tilt-is-measured-in-the-world-not-in-the-pocket",
        (("    a = rotate_by_quat_inv(pocket_quat, part_axis_w)",
          "    a = part_axis_w"),),
        ("a part tilted WITH the pocket reads zero tilt",),
    ),
    (
        # The ramp left unclamped. Above the opening the fraction grows with
        # the stand-off, so at the 165 mm home pose the commanded tilt is
        # twenty times what was asked for -- and the run would report the
        # force of a wrist swinging in free air as an insertion force.
        "the-tilt-ramp-is-not-clamped-above-the-opening",
        (("    frac = torch.clamp(1.0 - depth / upright_depth_m, min=0.0, max=1.0)",
          "    frac = 1.0 - depth / upright_depth_m"),),
        (
            "above the opening the full start tilt is held, however high",
            "at and below the upright depth the commanded tilt is zero",
        ),
    ),
    (
        # The ramp inverted: upright at the opening and tilted at the seat.
        # That is the one attitude the walls forbid outright (0.935 deg across
        # at full seat, D-106 (2)), so every episode would jam at depth and the
        # jam would be read as the task's.
        "the-tilt-ramp-runs-the-wrong-way",
        (("    frac = torch.clamp(1.0 - depth / upright_depth_m, min=0.0, max=1.0)",
          "    frac = torch.clamp(depth / upright_depth_m, min=0.0, max=1.0)"),),
        (
            "the ramp FALLS with depth: three quarters at a quarter of the way down",
            "above the opening the full start tilt is held, however high",
            "at and below the upright depth the commanded tilt is zero",
        ),
    ),
    (
        # The tilt command FOLLOWING its error instead of undoing it -- the
        # same defect the yaw term has its own mutation for, one column over.
        "the-tilt-term-follows-the-error",
        (("    d_tilt = torch.clamp(-tilt_err * tilt_gain, -max_tilt_step_rad, max_tilt_step_rad)",
          "    d_tilt = torch.clamp(tilt_err * tilt_gain, -max_tilt_step_rad, max_tilt_step_rad)"),),
        (
            "the tilt command UNDOES the error and lands in the named column only",
            "the commanded tilt is clipped in both directions",
        ),
    ),
    (
        # The tilt written into BOTH in-plane rotation columns. The part then
        # tips about the diagonal, the axis the run says it swept is not the
        # axis it swept, and the underside asymmetry -- the whole reason the
        # sign is measured -- is averaged away.
        "the-tilt-is-written-into-both-rotation-columns",
        (("    cmd[:, 3 + tilt_axis] = torch.where(tilt_active, d_tilt, torch.zeros_like(d_tilt))",
          "    cmd[:, 3] = torch.where(tilt_active, d_tilt, torch.zeros_like(d_tilt))\n"
          "    cmd[:, 4] = torch.where(tilt_active, d_tilt, torch.zeros_like(d_tilt))"),),
        ("the tilt command UNDOES the error and lands in the named column only",),
    ),
    (
        # The capture depth read FORWARDS. It reports the first bin the
        # quantity dipped under the tolerance in, not the depth from which the
        # pocket HOLDS the part -- and that number would go straight into
        # engaged_depth_m as a capture depth that is too shallow.
        "the-capture-depth-is-read-forwards",
        ((
            "    return rows[settled[\"settle_step\"]][\"depth_lo_mm\"]",
            "    return next((r[\"depth_lo_mm\"] for r in rows if r[value_key] < tol), None)",
        ),),
        ("the capture depth is read BACKWARDS, so an early dip does not count",),
    ),
    (
        # THE MEASURED DEFECT ITSELF, written back in: steer against the pocket
        # axis instead of against the latched start attitude. This is exactly
        # what RT-124 and RT-125 ran -- 48 of 48 episodes on the 300 N abort at
        # depth p50 0.000 mm -- so the mutation is not hypothetical, it is the
        # code that produced two failed runs.
        "tilt-measured-against-the-pocket-axis-instead-of-the-latch",
        ((
            "    now_cs = torch.stack((torch.cos(tilt_now), torch.sin(tilt_now)), dim=1)\n"
            "    return signed_angle_difference(now_cs, ref_cs)",
            "    now_cs = torch.stack((torch.cos(tilt_now), torch.sin(tilt_now)), dim=1)\n"
            "    return tilt_now",
        ),),
        (
            "a part at its latched tilt reference reads zero relative tilt",
            "the relative tilt is signed, current minus reference",
            "the relative tilt takes the SHORT way round the branch cut",
            "an antiparallel start reads 180 deg absolute and 0 deg relative",
        ),
    ),
    (
        # The latch re-taken every step. The reference then tracks the part,
        # the relative tilt is zero forever, and the controller commands
        # nothing -- a tilt run that reports a perfectly held schedule while
        # the part never moved.
        "tilt-latch-overwrites-every-step",
        (("    new_ref = torch.where(latched > 0.5, ref_cs, now_cs)",
          "    new_ref = now_cs"),),
        ("a latched tilt reference survives a later tilt change",),
    ),
    (
        # The latch never marks itself, so the env is re-latched on the next
        # step. Same end state as above, reached through the flag instead of
        # through the value, and the two are worth separating: only this one
        # survives a caller that latches conditionally.
        "tilt-latch-never-marks-itself",
        (("    return new_ref, torch.ones_like(latched)",
          "    return new_ref, torch.zeros_like(latched)"),),
        ("latching the tilt marks the env as latched",),
    ),
    (
        # THE SIGN, and it is the one mistake this line invites: compare
        # tip_rel_z against the depth instead of negating it. tip_rel_z is
        # NEGATIVE inside the pocket, so the test is never true down there --
        # the stop silently does nothing and RT-128 repeats itself.
        "the-seat-stop-compares-tip_rel_z-instead-of-the-depth",
        (("        seated = -tip_rel[:, 2] >= stop_depth_m",
          "        seated = tip_rel[:, 2] >= stop_depth_m"),),
        (
            "the seat stop is inclusive at exactly the stop depth",
            "past the seat stop the descent stays zero",
        ),
    ),
    (
        # Exclusive instead of inclusive. One control step of overshoot, which
        # at the pocket floor is exactly the step that produces the force.
        "the-seat-stop-is-exclusive-at-the-boundary",
        (("        seated = -tip_rel[:, 2] >= stop_depth_m",
          "        seated = -tip_rel[:, 2] > stop_depth_m"),),
        ("the seat stop is inclusive at exactly the stop depth",),
    ),
    (
        # The off switch removed. With the default -1.0 every depth is "past
        # the stop", so the shared callers -- scripted_insert.py and the D-108
        # gate run -- stop descending at all.
        "the-seat-stop-is-always-on",
        (("    if stop_depth_m >= 0.0:",
          "    if True:"),),
        (
            "the seat stop is OFF by default, so the shared callers are unchanged",
            # AND BOTH EXISTING DESCEND CHECKS, which is the point: the off
            # switch is what keeps the shared callers working, so removing it
            # has to break the checks that describe them.
            "DESCEND commands the constant rate DOWNWARD",
            "DESCEND ignores the standoff height entirely",
        ),
    ),
    # -- the OSC action (inbox entry "Audit 2026-09-03 (a)") ---------------
    (
        # The rotation triple handed over in the POCKET frame: under a yawed
        # or tilted pocket the tilt is commanded about the wrong axis.
        "osc-rotation-triple-not-rotated",
        ((
            "    rot_env = rotate_by_quat(pocket_quat, command_pocket[:, 3:6])\n",
            "    rot_env = command_pocket[:, 3:6]\n",
        ),),
        ("the rotation triple rotates pocket -> env too",),
    ),
    (
        # Unclamped: the metrics file records an action the env silently
        # clipped, and a large command reads as a small one in the log.
        "osc-action-unclamped",
        ((
            "            torch.clamp(lin_env / pos_step_limit_m, -1.0, 1.0),\n",
            "            lin_env / pos_step_limit_m,\n",
        ),),
        ("the OSC action is clamped to [-1, 1]",),
    ),
)


def _run(mutations=()) -> dict[str, bool]:
    return {name: ok for name, ok, _ in build_checks(load_policy(mutations))}


def _self_test() -> int:
    """Break the source one way at a time; require exactly the named flips."""
    print("[check_scripted_insert] counter-proof (D-080)")
    baseline = _run()
    rc = 0
    if not all(baseline.values()):
        bad = [k for k, v in baseline.items() if not v]
        print(f"  FAIL  baseline is not green, counter-proof is meaningless: {bad}")
        return 1
    print(f"  PASS  baseline: {len(baseline)} checks green")

    for name, muts, expected in MUTATIONS:
        try:
            got = _run(muts)
        except SystemExit as exc:
            print(f"  FAIL  {name}: {exc}")
            rc = 1
            continue
        # A check that DISAPPEARED counts as flipped: a mutation can remove the
        # branch a check lived in, and "it never ran" is not "it passed".
        flipped = {k for k in baseline if k not in got or not got[k]}
        missing = set(expected) - flipped
        extra = flipped - set(expected)
        unknown = set(expected) - set(baseline)
        if unknown:
            print(f"  FAIL  {name}: names no existing check {sorted(unknown)}")
            rc = 1
            continue
        if not flipped:
            print(f"  FAIL  {name}: flipped NOTHING -- these checks can never fail")
            rc = 1
        elif missing or extra:
            print(f"  FAIL  {name}: missing {sorted(missing)} extra {sorted(extra)}")
            rc = 1
        else:
            print(f"  PASS  {name}: flips exactly {len(flipped)} declared check(s)")

    # The LAST line is a verdict, not a count. ``selftest_checks.py`` judges by
    # exit code but prints only this line, so a tail that reads "36 checks"
    # would show the same text whether the counter-proof passed or failed.
    print(f"[check_scripted_insert] {len(baseline)} checks / {len(MUTATIONS)} mutations")
    print("COUNTER-PROOF PASSED" if rc == 0 else "COUNTER-PROOF FAILED")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true", help="run the D-080 counter-proof")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()

    checks = build_checks(load_policy())
    rc = 0
    for name, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        suffix = f"   [{detail}]" if detail else ""
        print(f"  {tag}  {name}{suffix}")
        if not ok:
            rc = 1
    verdict = "PASS" if rc == 0 else "FAIL"
    print(f"[check_scripted_insert] {verdict} -- {len(checks)} checks")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
