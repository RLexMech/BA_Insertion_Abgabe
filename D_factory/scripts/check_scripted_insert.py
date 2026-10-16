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


RELATIVE_IMPORT = "from .insertion_math import axes_from_quat, rotate_into_frame"


def load_policy(mutations: tuple[tuple[str, str], ...] = ()) -> dict:
    """Execute both modules under the stand-in, optionally mutating the policy."""
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
    """Every ``@torch.jit.script`` function that reads a module-level constant."""
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


P_STANDOFF = 0.02
P_LAT_GAIN = 1.0
P_VERT_GAIN = 1.0
P_YAW_GAIN = 1.0
P_DESCEND = 0.001
P_MAX_STEP = 0.005
P_MAX_YAW = 0.02
P_LAT_TOL = 0.0005
P_YAW_TOL = 0.01
P_HEIGHT_TOL = 0.001
P_F_ABORT = 50.0


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

    check("a part at its target reads zero yaw error",
          abs(float(_arr(yaw_error(cs_a, cs_a))[0])) < 1e-12)
    check("the yaw error is signed, current minus target",
          abs(float(_arr(yaw_error(cs_b, cs_a))[0]) - 0.8) < 1e-9,
          f"{float(_arr(yaw_error(cs_b, cs_a))[0]):.9f}")
    near_pi = torch.tensor([[math.cos(math.radians(179.0)), math.sin(math.radians(179.0))]])
    past_pi = torch.tensor([[math.cos(math.radians(-179.0)), math.sin(math.radians(-179.0))]])
    err_wrap = float(_arr(yaw_error(near_pi, past_pi))[0])
    check("the yaw error takes the SHORT way round the branch cut",
          abs(err_wrap - math.radians(-2.0)) < 1e-9,
          f"{math.degrees(err_wrap):.6f} deg")

    r2 = math.sqrt(0.5)
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_z90 = torch.tensor([[r2, 0.0, 0.0, r2]])
    v_x = torch.tensor([[1.0, 0.0, 0.0]])
    check("rotate_by_quat maps the pocket frame OUT into its parent",
          _maxabs(rotate_by_quat(q_z90, v_x), _arr([[0.0, 1.0, 0.0]])) < 1e-12,
          str(_arr(rotate_by_quat(q_z90, v_x)).round(9).tolist()))
    check("rotate_by_quat_inv maps the parent INTO the pocket frame",
          _maxabs(rotate_by_quat_inv(q_z90, v_x), _arr([[0.0, -1.0, 0.0]])) < 1e-12,
          str(_arr(rotate_by_quat_inv(q_z90, v_x)).round(9).tolist()))
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
    rot = ns["axes_from_quat"](q_batch)
    check("rotate_out_of_frame is the transpose of the reused rotate_into_frame",
          _maxabs(ns["rotate_into_frame"](rot, rotate_out_of_frame(rot, v_batch)), v_batch) < 1e-12)

    q_x30 = torch.tensor([[math.cos(math.radians(15.0)), math.sin(math.radians(15.0)), 0.0, 0.0]])
    q_y90 = torch.tensor([[r2, 0.0, r2, 0.0]])
    check("two identical orientations read zero tilt",
          abs(float(_arr(axis_tilt_angle(q_id, q_id))[0])) < 1e-7,
          f"{math.degrees(float(_arr(axis_tilt_angle(q_id, q_id))[0])):.9f} deg")
    check("a pure yaw difference is NOT a tilt",
          abs(float(_arr(axis_tilt_angle(q_z90, q_id))[0])) < 1e-7,
          f"{math.degrees(float(_arr(axis_tilt_angle(q_z90, q_id))[0])):.9f} deg")
    check("a 30 deg rotation about x reads 30 deg of tilt",
          abs(math.degrees(float(_arr(axis_tilt_angle(q_x30, q_id))[0])) - 30.0) < 1e-6,
          f"{math.degrees(float(_arr(axis_tilt_angle(q_x30, q_id))[0])):.6f} deg")
    check("a 90 deg rotation about y reads 90 deg of tilt",
          abs(math.degrees(float(_arr(axis_tilt_angle(q_y90, q_id))[0])) - 90.0) < 1e-6,
          f"{math.degrees(float(_arr(axis_tilt_angle(q_y90, q_id))[0])):.6f} deg")
    check("the tilt is symmetric in its two arguments",
          _maxabs(axis_tilt_angle(q_x30, q_z90), axis_tilt_angle(q_z90, q_x30)) < 1e-12)
    tilt_batch = axis_tilt_angle(q_batch, q_batch)
    check("identical orientations never produce NaN",
          bool(_arr(torch.isfinite(tilt_batch)).all()) and _maxabs(tilt_batch, 0.0) < 1e-7,
          str(_arr(tilt_batch).round(9).tolist()))

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
    check("a part that is aligned but still high stays in ALIGN",
          nxt[4] == ALIGN,
          f"tip_z {float(tip_z[4]):.4f} m vs standoff {P_STANDOFF:.4f} m -> {nxt[4]}")
    below = _arr(advance_phase(
        torch.tensor([ALIGN]), torch.tensor([0.0001]), torch.tensor([0.0010]),
        torch.tensor([-0.0050]), torch.tensor([0.0]),
        P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT))
    check("a tip already below the standoff descends rather than climbing back",
          below.tolist() == [DESCEND], str(below.tolist()))
    ph_mixed = torch.tensor([ALIGN, DESCEND])
    hot = torch.tensor([P_F_ABORT, P_F_ABORT])
    aborted = _arr(advance_phase(ph_mixed, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]),
                                 torch.tensor([P_STANDOFF, P_STANDOFF]), hot,
                                 P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT))
    check("the force abort fires from any phase",
          aborted.tolist() == [ABORT, ABORT],
          str(aborted.tolist()))
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

    free = _arr(advance_phase(ph_align, lat_err, yaw_err, tip_z, no_force,
                              P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT,
                              allow_descend=False))
    check("free space never leaves ALIGN",
          free.tolist() == [ALIGN] * 5,
          str(free.tolist()))
    check("free space holds ALIGN even below the standoff",
          _arr(advance_phase(
              torch.tensor([ALIGN]), torch.tensor([0.0001]), torch.tensor([0.0010]),
              torch.tensor([-0.0050]), torch.tensor([0.0]),
              P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT,
              allow_descend=False)).tolist() == [ALIGN])
    check("free space still aborts on force",
          _arr(advance_phase(torch.tensor([ALIGN]), torch.tensor([0.0]), torch.tensor([0.0]),
                             torch.tensor([P_STANDOFF]), torch.tensor([2.0 * P_F_ABORT]),
                             P_LAT_TOL, P_YAW_TOL, P_STANDOFF, P_HEIGHT_TOL, P_F_ABORT,
                             allow_descend=False)).tolist() == [ABORT])

    track_settle = ns["track_settle"]
    T_TOL = 0.001
    s = track_settle([0.005, 0.0002, 0.0002, 0.0002], T_TOL, 2)
    check("the settle index is the first step of the run that stays in tolerance",
          s["settle_step"] == 1, str(s))
    late = track_settle([0.0002, 0.0002, 0.005, 0.0002], T_TOL, 2)
    check("the settle index survives a late excursion",
          late["settle_step"] == 3, str(late))
    never = track_settle([0.005, 0.006, 0.007], T_TOL, 2)
    check("the settle index is -1 when the error never settles",
          never["settle_step"] == -1, str(never))
    check("a trace that is in tolerance from step 0 settles at 0",
          track_settle([0.0002, 0.0002], T_TOL, 2)["settle_step"] == 0)
    tail = track_settle([0.02, 0.02, 0.0001, 0.0009], T_TOL, 2)
    check("the steady error takes the maximum of the tail, not its mean",
          abs(tail["steady_max_m"] - 0.0009) < 1e-12, str(tail))
    check("the tail is clipped to the trace and reports its own width",
          track_settle([0.004, 0.001], T_TOL, 50)["tail"] == 2)
    check("an error exactly at the tolerance does not count as settled",
          track_settle([T_TOL], T_TOL, 1)["settle_step"] == -1)

    peak = update_peak_depth(torch.tensor([0.004, 0.001]), torch.tensor([0.002, 0.003]))
    check("the peak depth is a running maximum, not the latest value",
          _maxabs(peak, _arr([0.004, 0.003])) < 1e-12,
          str(_arr(peak).round(6).tolist()))

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
    check("ALIGN regulates the height toward the standoff",
          abs(cmd[0, 2] - (P_STANDOFF - 0.0230)) < 1e-12,
          f"{cmd[0, 2]:.6f}")
    check("DESCEND commands the constant rate DOWNWARD",
          abs(cmd[1, 2] + P_DESCEND) < 1e-12,
          f"{cmd[1, 2]:.6f}")
    check("DESCEND ignores the standoff height entirely",
          abs(cmd[1, 2] - cmd[1, 2]) < 1e-12 and abs(cmd[1, 2]) == P_DESCEND)
    check("the yaw command OPPOSES the yaw error, about the pocket z-axis",
          cmd[0, 5] < 0.0 and abs(cmd[0, 3]) < 1e-12 and abs(cmd[0, 4]) < 1e-12,
          str(cmd[0, 3:6].round(6).tolist()))

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
    check("the seat stop is inclusive at exactly the stop depth",
          abs(_stopped[1, 2]) < 1e-12,
          f"{_stopped[1, 2]:.6f}")
    check("past the seat stop the descent stays zero",
          abs(_stopped[2, 2]) < 1e-12,
          f"{_stopped[2, 2]:.6f}")
    check("the seat stop is OFF by default, so the shared callers are unchanged",
          abs(_no_stop[0, 2] + P_DESCEND) < 1e-12
          and abs(_no_stop[1, 2] + P_DESCEND) < 1e-12
          and abs(_no_stop[2, 2] + P_DESCEND) < 1e-12,
          str(_no_stop[:, 2].round(6).tolist()))
    check("the seat stop leaves the lateral and yaw terms working",
          _stopped[2, 0] < 0.0 and _stopped[2, 1] < 0.0 and _stopped[2, 5] < 0.0,
          str(_stopped[2, [0, 1, 5]].round(6).tolist()))
    check("an aborted env commands exactly zero on all six components",
          float(np.max(np.abs(cmd[2]))) == 0.0,
          str(cmd[2].round(9).tolist()))
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

    on_axis = torch.tensor([[0.0, 0.0, 0.0230], [0.0, 0.0, 0.0230]])
    aim_probe = torch.tensor([[0.0020, 0.0000], [0.0000, -0.0010]])
    cmd_aim = _arr(command_in_pocket_frame(
        on_axis, torch.tensor([ALIGN, DESCEND]), torch.zeros(2), aim_probe,
        P_STANDOFF, P_LAT_GAIN, P_VERT_GAIN, P_YAW_GAIN,
        P_DESCEND, P_MAX_STEP, P_MAX_YAW))
    check("a part on the axis is driven TOWARD its aim, not held at the axis",
          cmd_aim[0, 0] > 0.0 and abs(cmd_aim[0, 0] - 0.0020 * P_LAT_GAIN) < 1e-12,
          str(cmd_aim[0, 0:2].round(6).tolist()))
    check("DESCEND keeps holding the aim instead of releasing it",
          cmd_aim[1, 1] < 0.0 and abs(cmd_aim[1, 1] + 0.0010 * P_LAT_GAIN) < 1e-12,
          str(cmd_aim[1, 0:2].round(6).tolist()))

    one = torch.tensor([[0.001, 0.002, -0.003, 0.0, 0.0, 0.05]])
    check("identity pocket and identity base leave the command unchanged",
          _maxabs(command_to_base_frame(one, q_id, q_id), one) < 1e-12)
    only_x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    got = _arr(command_to_base_frame(only_x, q_z90, q_id))
    check("the pocket rotation is applied OUTWARD, pocket x -> env y",
          _maxabs(got[:, 0:3], _arr([[0.0, 1.0, 0.0]])) < 1e-12,
          str(got[0, 0:3].round(9).tolist()))
    got_b = _arr(command_to_base_frame(only_x, q_id, q_z90))
    check("the base rotation is applied INWARD, env x -> base -y",
          _maxabs(got_b[:, 0:3], _arr([[0.0, -1.0, 0.0]])) < 1e-12,
          str(got_b[0, 0:3].round(9).tolist()))
    both = _arr(command_to_base_frame(only_x, q_z90, q_z90))
    check("equal pocket and base rotations cancel, which one hop alone cannot",
          _maxabs(both[:, 0:3], _arr([[1.0, 0.0, 0.0]])) < 1e-12,
          str(both[0, 0:3].round(9).tolist()))
    only_rz = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    q_y90 = torch.tensor([[r2, 0.0, r2, 0.0]])
    got_r = _arr(command_to_base_frame(only_rz, q_y90, q_id))
    check("the axis-angle triple rotates like a vector, pocket z -> env x",
          _maxabs(got_r[:, 3:6], _arr([[1.0, 0.0, 0.0]])) < 1e-12,
          str(got_r[0, 3:6].round(9).tolist()))
    check("the rotation hop does not leak into the translation triple",
          float(np.max(np.abs(got_r[:, 0:3]))) < 1e-12)

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
    act_meas = _arr(joint_delta_to_action(q_des, q_meas, scale))
    check("the action differences against the TARGET, not the measured position",
          float(np.max(np.abs(act - act_meas))) > 1e-6,
          f"max |diff| = {float(np.max(np.abs(act - act_meas))):.6f}")

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

    offenders = scripted_functions_closing_over_constants(ns["__src__"])
    check("no jit-scripted function closes over a module constant",
          not offenders,
          "; ".join(offenders) if offenders else "none")

    le = lateral_error(torch.tensor([[0.003, 0.004, 9.9]]), torch.zeros(1, 2))
    check("the lateral error is planar and ignores the vertical component",
          abs(float(_arr(le)[0]) - 0.005) < 1e-12,
          f"{float(_arr(le)[0]):.9f}")
    le_aim = lateral_error(torch.tensor([[0.003, 0.004, 9.9]]),
                           torch.tensor([[0.003, 0.004]]))
    check("a part sitting exactly on its aim reads zero lateral error",
          abs(float(_arr(le_aim)[0])) < 1e-12,
          f"{float(_arr(le_aim)[0]):.12f}")

    aim_offsets = ns["aim_offsets"]
    aim_sweep_rows = ns["aim_sweep_rows"]

    a_none = _arr(aim_offsets(4, "none", 0.6))
    check("aim_sweep none is all zeros -- it must reproduce RT-74 exactly",
          a_none.shape == (4, 2) and float(np.max(np.abs(a_none))) == 0.0,
          str(a_none.round(9).tolist()))
    a_x = _arr(aim_offsets(5, "x", 0.6))
    check("the aim spans exactly +-span/2, endpoints INCLUDED",
          abs(a_x[0, 0] + 0.3) < 1e-12 and abs(a_x[4, 0] - 0.3) < 1e-12,
          str(a_x[:, 0].round(6).tolist()))
    check("the aim is spread linearly across the envs",
          abs(a_x[1, 0] + 0.15) < 1e-12 and abs(a_x[2, 0]) < 1e-12,
          str(a_x[:, 0].round(6).tolist()))
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
    a_one = _arr(aim_offsets(1, "x", 0.6))
    check("a single env aims at the axis instead of dividing by zero",
          a_one.shape == (1, 2) and float(np.max(np.abs(a_one))) == 0.0,
          str(a_one.round(9).tolist()))

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
    _rows = force_depth_profile(0.001, [0, 2, 0, 5], [0.0, 7.0, 0.0, 9.0], [0.0, 12.0, 0.0, 20.0])
    check("the profile drops empty bins and keeps each row's own depth range",
          len(_rows) == 2
          and abs(_rows[0]["depth_lo_mm"] - 1.0) < 1e-12
          and abs(_rows[0]["force_mean_n"] - 6.0) < 1e-12
          and abs(_rows[1]["depth_lo_mm"] - 3.0) < 1e-12
          and abs(_rows[1]["force_mean_n"] - 4.0) < 1e-12,
          f"{len(_rows)} rows")

    _th = 0.20
    _c, _s = math.cos(_th / 2.0), math.sin(_th / 2.0)
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_rx = torch.tensor([[_c, _s, 0.0, 0.0]])
    q_ry = torch.tensor([[_c, 0.0, _s, 0.0]])
    q_rx_neg = torch.tensor([[_c, -_s, 0.0, 0.0]])

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
    check("a part tilted WITH the pocket reads zero tilt",
          abs(float(signed_tilt_angle(q_rx, q_rx, 0)[0])) < 1e-9,
          f"{float(signed_tilt_angle(q_rx, q_rx, 0)[0]):.9f} rad")

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

    check("a part at its latched tilt reference reads zero relative tilt",
          _maxabs(tilt_since_reference(_t_ref, _ref_cs), 0.0) < 1e-9,
          str(_arr(tilt_since_reference(_t_ref, _ref_cs)).round(9).tolist()))
    check("the relative tilt is signed, current minus reference",
          abs(float(_arr(tilt_since_reference(torch.tensor([-math.pi, 1.10]), _ref_cs))[1]) - 0.8) < 1e-9,
          f"{float(_arr(tilt_since_reference(torch.tensor([-math.pi, 1.10]), _ref_cs))[1]):.9f}")
    _cut = tilt_since_reference(torch.tensor([math.radians(178.0), 0.30]), _ref_cs)
    check("the relative tilt takes the SHORT way round the branch cut",
          abs(float(_arr(_cut)[0]) - math.radians(-2.0)) < 1e-9,
          f"{math.degrees(float(_arr(_cut)[0])):.6f} deg")
    _q_flip = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    _abs_flip = signed_tilt_angle(_q_flip, q_id, 0)
    _ref_flip, _ = latch_tilt_reference(_abs_flip, torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0]]))
    check("an antiparallel start reads 180 deg absolute and 0 deg relative",
          abs(abs(float(_arr(_abs_flip)[0])) - math.pi) < 1e-9
          and abs(float(_arr(tilt_since_reference(_abs_flip, _ref_flip))[0])) < 1e-9,
          f"absolute {math.degrees(float(_arr(_abs_flip)[0])):.4f} deg, "
          f"relative {math.degrees(float(_arr(tilt_since_reference(_abs_flip, _ref_flip))[0])):.9f} deg")

    _start = torch.tensor([0.10, -0.10])
    _above = tilt_schedule(torch.tensor([-0.165, -0.001]), _start, 0.008)
    check("above the opening the full start tilt is held, however high",
          abs(float(_above[0]) - 0.10) < 1e-12 and abs(float(_above[1]) + 0.10) < 1e-12,
          f"{[round(float(v), 6) for v in _above.tolist()]}")
    _half = tilt_schedule(torch.tensor([0.002, 0.006]), _start, 0.008)
    check("the ramp FALLS with depth: three quarters at a quarter of the way down",
          abs(float(_half[0]) - 0.075) < 1e-12 and abs(float(_half[1]) + 0.025) < 1e-12,
          f"{[round(float(v), 6) for v in _half.tolist()]}")
    _deep = tilt_schedule(torch.tensor([0.008, 0.033]), _start, 0.008)
    check("at and below the upright depth the commanded tilt is zero",
          abs(float(_deep[0])) < 1e-12 and abs(float(_deep[1])) < 1e-12,
          f"{[round(float(v), 9) for v in _deep.tolist()]}")
    check("an upright depth of zero is the straight controller, not a division",
          _maxabs(tilt_schedule(torch.tensor([-0.1, 0.02]), _start, 0.0)) == 0.0)

    _off = tilt_offsets(5, -0.10, 0.10)
    check("the tilt sweep includes both endpoints and the straight case",
          abs(float(_off[0]) + 0.10) < 1e-12
          and abs(float(_off[4]) - 0.10) < 1e-12
          and abs(float(_off[2])) < 1e-12,
          f"{[round(float(v), 6) for v in _off.tolist()]}")
    check("one env carries no sweep and does not divide by zero",
          abs(float(tilt_offsets(1, -0.10, 0.10)[0]) + 0.10) < 1e-12)

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


MUTATIONS: tuple[tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]], ...] = (
    (
        "free-air-depths-clamped-into-the-first-bin",
        ((
            "        if depth < 0.0:\n            continue",
            "        if depth < 0.0:\n            depth = 0.0",
        ),),
        ("free-air samples above the opening never reach bin 0",),
    ),
    (
        "deep-samples-piled-onto-the-last-bin",
        ((
            "        if b >= n_bins:\n            continue",
            "        if b >= n_bins:\n            b = n_bins - 1",
        ),),
        ("a sample past the last bin is dropped, not piled onto it",),
    ),
    (
        "empty-bins-kept-in-the-profile",
        ((
            "        if n == 0:\n            continue",
            "        if n == 0:\n            n = 1",
        ),),
        (
            "the profile drops empty bins and keeps each row's own depth range",
            "a profile row NAMES its quantity and its unit",
        ),
    ),
    (
        "lateral-error-measured-against-the-axis-not-the-aim",
        ((
            "    return torch.linalg.norm(tip_rel[:, 0:2] - aim, dim=-1)",
            "    return torch.linalg.norm(tip_rel[:, 0:2], dim=-1)",
        ),),
        ("a part sitting exactly on its aim reads zero lateral error",),
    ),
    (
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
        "aim-sweep-none-falls-through-to-x",
        ((
            'AIM_AXES: dict = {"none": -1, "x": 0, "y": 1}',
            'AIM_AXES: dict = {"none": 0, "x": 0, "y": 1}',
        ),),
        ("aim_sweep none is all zeros -- it must reproduce RT-74 exactly",),
    ),
    (
        "free-space-mode-still-descends",
        ((
            "    if not allow_descend:\n        aligned = torch.zeros_like(aligned)",
            "    if not allow_descend:\n        pass",
        ),),
        ("free space never leaves ALIGN",
         "free space holds ALIGN even below the standoff"),
    ),
    (
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
            "the capture depth is read BACKWARDS, so an early dip does not count",
        ),
    ),
    (
        "steady-error-uses-the-mean",
        ((
            '"steady_max_m": max(window)',
            '"steady_max_m": sum(window) / len(window)',
        ),),
        ("the steady error takes the maximum of the tail, not its mean",),
    ),
    (
        "envs-without-an-episode-dropped-from-the-table",
        ((
            '    rows.sort(key=lambda r: (r["aim_x_mm"], r["aim_y_mm"], r["env"]))\n    return rows',
            '    rows.sort(key=lambda r: (r["aim_x_mm"], r["aim_y_mm"], r["env"]))\n'
            '    return [r for r in rows if r["episodes"]]',
        ),),
        ("every env gets a row, including one that finished no episode",),
    ),
    (
        "aim-table-sorted-by-env-instead-of-by-aim",
        ((
            '    rows.sort(key=lambda r: (r["aim_x_mm"], r["aim_y_mm"], r["env"]))',
            '    rows.sort(key=lambda r: r["env"])',
        ),),
        ("the rows are sorted by the aim, not by the env index",),
    ),
    (
        "tilt-measured-on-the-x-axis-instead-of-the-tool-axis",
        (("    part_axis = axes_from_quat(part_quat)[:, :, 2]",
          "    part_axis = axes_from_quat(part_quat)[:, :, 0]"),),
        (
            "two identical orientations read zero tilt",
            "a pure yaw difference is NOT a tilt",
            "a 30 deg rotation about x reads 30 deg of tilt",
            "a 90 deg rotation about y reads 90 deg of tilt",
            "the tilt is symmetric in its two arguments",
            "identical orientations never produce NaN",
            "the signed tilt flips sign with the rotation, where axis_tilt_angle cannot",
        ),
    ),
    (
        "tilt-drops-the-clamp-before-acos",
        (("    return torch.acos(torch.clamp(cos, min=-1.0, max=1.0))",
          "    return torch.acos(cos)"),),
        ("identical orientations never produce NaN",),
    ),
    (
        "tilt-assumes-the-pocket-stands-upright",
        (("    pocket_axis = axes_from_quat(pocket_quat)[:, :, 2]",
          "    pocket_axis = torch.zeros_like(part_axis) + torch.tensor([0.0, 0.0, 1.0])"),),
        (
            "the tilt is symmetric in its two arguments",
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
        "a-really-scripted-function-starts-reading-a-module-constant",
        (("    return torch.maximum(depth, peak)",
          "    return torch.maximum(depth, peak) + PHASE_ALIGN"),),
        ("no jit-scripted function closes over a module constant",),
    ),
    (
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
    (
        "the-signed-tilt-loses-its-sign-convention-about-x",
        (("        return torch.atan2(-a[:, 1], a[:, 2])",
          "        return torch.atan2(a[:, 1], a[:, 2])"),),
        (
            "a +theta rotation about pocket x reads +theta on axis 0",
            "the signed tilt flips sign with the rotation, where axis_tilt_angle cannot",
        ),
    ),
    (
        "the-tilt-is-measured-in-the-world-not-in-the-pocket",
        (("    a = rotate_by_quat_inv(pocket_quat, part_axis_w)",
          "    a = part_axis_w"),),
        ("a part tilted WITH the pocket reads zero tilt",),
    ),
    (
        "the-tilt-ramp-is-not-clamped-above-the-opening",
        (("    frac = torch.clamp(1.0 - depth / upright_depth_m, min=0.0, max=1.0)",
          "    frac = 1.0 - depth / upright_depth_m"),),
        (
            "above the opening the full start tilt is held, however high",
            "at and below the upright depth the commanded tilt is zero",
        ),
    ),
    (
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
        "the-tilt-term-follows-the-error",
        (("    d_tilt = torch.clamp(-tilt_err * tilt_gain, -max_tilt_step_rad, max_tilt_step_rad)",
          "    d_tilt = torch.clamp(tilt_err * tilt_gain, -max_tilt_step_rad, max_tilt_step_rad)"),),
        (
            "the tilt command UNDOES the error and lands in the named column only",
            "the commanded tilt is clipped in both directions",
        ),
    ),
    (
        "the-tilt-is-written-into-both-rotation-columns",
        (("    cmd[:, 3 + tilt_axis] = torch.where(tilt_active, d_tilt, torch.zeros_like(d_tilt))",
          "    cmd[:, 3] = torch.where(tilt_active, d_tilt, torch.zeros_like(d_tilt))\n"
          "    cmd[:, 4] = torch.where(tilt_active, d_tilt, torch.zeros_like(d_tilt))"),),
        ("the tilt command UNDOES the error and lands in the named column only",),
    ),
    (
        "the-capture-depth-is-read-forwards",
        ((
            "    return rows[settled[\"settle_step\"]][\"depth_lo_mm\"]",
            "    return next((r[\"depth_lo_mm\"] for r in rows if r[value_key] < tol), None)",
        ),),
        ("the capture depth is read BACKWARDS, so an early dip does not count",),
    ),
    (
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
        "tilt-latch-overwrites-every-step",
        (("    new_ref = torch.where(latched > 0.5, ref_cs, now_cs)",
          "    new_ref = now_cs"),),
        ("a latched tilt reference survives a later tilt change",),
    ),
    (
        "tilt-latch-never-marks-itself",
        (("    return new_ref, torch.ones_like(latched)",
          "    return new_ref, torch.zeros_like(latched)"),),
        ("latching the tilt marks the env as latched",),
    ),
    (
        "the-seat-stop-compares-tip_rel_z-instead-of-the-depth",
        (("        seated = -tip_rel[:, 2] >= stop_depth_m",
          "        seated = tip_rel[:, 2] >= stop_depth_m"),),
        (
            "the seat stop is inclusive at exactly the stop depth",
            "past the seat stop the descent stays zero",
        ),
    ),
    (
        "the-seat-stop-is-exclusive-at-the-boundary",
        (("        seated = -tip_rel[:, 2] >= stop_depth_m",
          "        seated = -tip_rel[:, 2] > stop_depth_m"),),
        ("the seat stop is inclusive at exactly the stop depth",),
    ),
    (
        "the-seat-stop-is-always-on",
        (("    if stop_depth_m >= 0.0:",
          "    if True:"),),
        (
            "the seat stop is OFF by default, so the shared callers are unchanged",
            "DESCEND commands the constant rate DOWNWARD",
            "DESCEND ignores the standoff height entirely",
        ),
    ),
    (
        "osc-rotation-triple-not-rotated",
        ((
            "    rot_env = rotate_by_quat(pocket_quat, command_pocket[:, 3:6])\n",
            "    rot_env = command_pocket[:, 3:6]\n",
        ),),
        ("the rotation triple rotates pocket -> env too",),
    ),
    (
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
