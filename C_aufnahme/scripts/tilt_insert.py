# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""THE KIPP INSTRUMENT: a scripted insertion that goes in STRAIGHT and TILTED,
and reads the peak force and the SAPU interpenetration AGAINST the depth.

STATUS: SPENT (2026-09-01). It ran six times, RT-124 .. RT-129
(``rt_logs/VERDICTS.md``), and RT-129 answered the question it was built
for AGAINST the instrument: the "capture depth" it reads is the commanded
``--upright-depth-mm`` ramp (the depth profile's ``tilt max`` column falls
linearly from the start tilt to zero exactly over that distance), and the
forces it reads are this controller's, not the task's. The two measurement
rules quoted below (F_max lower bound, engaged capture depth) are WITHDRAWN
and the suction-cup datasheet anchor with them -- home: the inbox entry "The tilted scripted insertion is SPENT ..." (2026-09-01).
No further runs are planned; the text below is kept as the record of what
the run was meant to pay and how it commands the tilt.

WHY THIS FILE EXISTS (as written 2026-09-01, before RT-129)
--------------------
Three numbers in ``insertion_env_cfg.py`` are marked PLACEHOLDER -- invented,
no source -- and all three name the SAME missing measurement in their marks:

* ``force_abort_f_max_n`` (``:940-952``). The selection RULE is decided and the
  number is not: the lower bound is "the largest measured peak force of a CLEAN
  scripted insertion, straight AND tilted (edge-first), several repeats, across
  the fixture noise"; the upper anchor is the suction cup's holding force from
  its datasheet, ``[Datenblatt pending]``. The datasheet limit must be VERIFIED
  above the measured peak -- a limit below it is a conflict and the decision
  stops. Today's 60.0 N came down from 100.0 N after RT-107 and has no source
  either way.
* ``engaged_depth_m`` (``:953-963``). The RULE is "the MEASURED CAPTURE DEPTH
  of the tilted scripted insertion". Today's 10.8 mm is 30 % of the seat depth,
  set by hand on 2026-08-30 and downgraded again on 2026-09-01.
* ``abort_payment`` (``:627``). Its magnitude was left coupled to ``F_max`` at
  the same measurement (D-114, "decided together with ``F_max``").

None of them can be read off a run that cannot tilt. D-071 records that the
real part goes in EDGE FIRST and, in the same entry, that nothing in the code
encodes the tilted entry. This run encodes it -- as a commanded SCHEDULE, not
as a search.

WHAT IT MEASURES, AND WHAT IT REFUSES TO DECIDE
-----------------------------------------------
Four quantities, all against the same depth axis, in the same bins:

    force N | interpenetration mm (SAPU) | tilt deg | lateral offset mm

and per episode the peaks plus whether the episode was CLEAN.

IT SETS NO THRESHOLD, the same refusal ``seat_probe.py`` makes. What counts as
``F_max``, what counts as the capture depth, and what the abort should pay are
DECISIONS. This run reports the distributions those decisions need, marks the
``[Datenblatt pending]`` half as still missing, and stops there.

"CLEAN" IS NOT A JUDGEMENT EITHER. It is the rule the ``force_abort_f_max_n``
mark already fixed: an episode is clean when the part reached the success depth
``SEATED_SUCCESS_DEPTH`` INSIDE the pocket cross-section, and did not end in
``ABORT_FORCE``. Both halves matter -- RT-107 is the standing proof that a
depth reading without the lateral gate can be 109 mm beside the fixture and
still read 0.0 mm of depth.

HOW THE TILT IS COMMANDED
-------------------------
``scripted_policy.tilt_offsets`` gives ONE fixed start tilt per env, spanning
the requested range with the endpoints included. A span that crosses zero
therefore measures the STRAIGHT insertion and both tilt SIGNS in one run, under
the same controller, the same gains and the same noise draw. Two separate runs
would differ in all three and the comparison would carry those differences.

The sign is a result, not a detail: the part's underside is a sloped strip on
one side and a parallel strip on the other
(``insertion_tasks_cfg.py:1393-1409``), so the two tilt directions meet
different geometry.

WHAT "TILT" MEANS HERE, and it is not the pocket axis. RT-127 (training PC,
git e38e687) measured the start attitude at ``signed p50 -180.0000 deg`` in all
16 envs: the tool axis is held ANTIPARALLEL to the pocket axis. So the run
LATCHES the step-1 attitude per env (``scripted_policy.latch_tilt_reference``,
the same refusal-to-assume ``latch_yaw_target`` makes for the yaw) and both
commands and reports the tilt as the difference from it
(``tilt_since_reference``). A commanded +8 deg is eight degrees away from how
the part is held, which is what "edge-first" names. The ABSOLUTE angle to the
pocket axis stays in the log next to it, as ``peak_axis_tilt_rad`` and in the
step-1 line; the two are not interchangeable.

``scripted_policy.tilt_schedule`` holds the full tilt above the opening plane
and ramps it LINEARLY to zero by ``--upright-depth-mm``. A ramp and not a step,
because the walls forbid the tilt long before the seat (D-106 (2): 0.935 deg
across, 1.905 / 2.821 deg along at full seat). The ramp shape is a measurement
setting; the run reports the force and the interpenetration THAT ramp produced.

THE ALIGN GATE GAINS ONE CONDITION, and it lives here rather than in
``advance_phase``: an env that is otherwise ready to descend but has not
reached its commanded tilt is held in ALIGN. Without it the descent starts
while the wrist is still turning, and the force recorded would be the
schedule's, not the task's. ``advance_phase`` is left untouched because the
D-108 gate run shares it.

WHERE THIS RUN STANDS ON THE D-108 INFORMATION BOUNDARY
-------------------------------------------------------
This is a PROBE, like ``seat_probe.py``, not the gate. The split is deliberate
and it is drawn INSIDE the loop:

* THE CONTROLLER reads observation channels only -- ``tip_rel``,
  ``pocket_quat``, ``yaw_cos_sin``, ``ee_quat``, ``force`` -- exactly what the
  policy will get. Every command in this file is computed from those.
* THE INSTRUMENT reads env internals: ``_entrance_pos`` and ``_fixture_quat``
  for the SAPU pose, and the peg body index. It has to: interpenetration is not
  an observation channel and never will be.

So the measurement is outside the boundary and the control is inside it. Said
here because ``check_env_wiring.py`` enforces the boundary on
``scripted_insert.py`` only, and a reader must not have to infer which rule
applies to which file.

THE SAPU QUERY IS BUILT HERE, NOT BORROWED FROM THE ENV
-------------------------------------------------------
The env builds its two Warp queries only when ``rl_terms_enabled`` is True
(``insertion_env.py:508-528``), and this run needs the terms OFF: with them on,
the force abort and the success termination end episodes mid-descent and the
depth profile stops exactly where it gets interesting.

So this script constructs the SAME query the env constructs -- part OBJ
sampled, fixture OBJ queried,
``insertion_sdf.SdfDistanceQuery(..., mesh_obj_path=...)`` -- and calls it with
the SAME transform (``insertion_env.py:1167-1171``, copied line for line,
including the env-local part position). One class, one kernel, no second
implementation.

NAMED, UNRESOLVED: the fixture mesh is not watertight (9 unpaired edges,
RT-64) and the SDF sign has never been measured AT the surface. Every
interpenetration number this run prints inherits that flag. It is a comparison
between tilt values under one query, not an absolute penetration depth.

THE FORCE ABORT IS A MEASUREMENT SETTING HERE
----------------------------------------------
``--f-abort`` writes the limit into the cfg BEFORE ``gym.make``, the same way
``scripted_insert.py`` does and for the same reason (the startup banner is the
only proof the override arrived). A run that aborts is a TRUNCATED curve: the
peak force it reports is the limit, not the task's. The run says so in one loud
line and the metrics file carries ``abort_truncated``.

INVOCATION
----------
Laptop: nothing here runs without Isaac. The offline arithmetic underneath it
is checked by ``scripts/check_scripted_insert.py --self-test``.

Training PC, the straight-and-tilted sweep this file was written for::

    .\scripts\rt_log.ps1 RT-N python scripts/tilt_insert.py
        --task Ur5e-Insertion-Direct-v0 --num_envs 16 --headless
        --tilt-axis y --tilt-sweep-deg -8 8 --upright-depth-mm 8
        --f-abort 300 --max-steps 900

``--f-abort 300`` is not a proposal for ``F_max``. It is how the curve is kept
from being truncated by the invented limit of the day while the clean peak is being
found -- the same move RT-69 made with 200 N.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import math
import pathlib
import sys
import traceback

from isaaclab.app import AppLauncher

_TOOLS = pathlib.Path(__file__).resolve().parent / "tools"
_spec = importlib.util.spec_from_file_location("isaac_exit", _TOOLS / "isaac_exit.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["isaac_exit"] = _mod
_spec.loader.exec_module(_mod)
exit_with = _mod.exit_with

SCRIPT_MARKER = "tilt_insert-2026-09-03a"

parser = argparse.ArgumentParser(
    description="Scripted insertion, straight AND tilted: peak force and SAPU interpenetration over depth."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--max-steps", type=int, default=900, help="Control steps before the run stops.")
parser.add_argument(
    "--out",
    type=str,
    default="tilt_insert_metrics.json",
    help="Metrics file. Numbers are read from files, never from prose.",
)

ctrl = parser.add_argument_group("controller (measurement settings, not task decisions)")
ctrl.add_argument("--standoff", type=float, default=0.010, help="Height above the opening held during ALIGN, m.")
ctrl.add_argument("--lateral-gain", type=float, default=0.5)
ctrl.add_argument("--vertical-gain", type=float, default=0.5)
ctrl.add_argument("--yaw-gain", type=float, default=0.5)
ctrl.add_argument("--descend-rate", type=float, default=0.0005, help="Commanded descent per control step, m.")
ctrl.add_argument("--max-step", type=float, default=0.002, help="Clip on every commanded translation, m.")
ctrl.add_argument("--max-yaw-step", type=float, default=0.010, help="Clip on the commanded yaw, rad.")
ctrl.add_argument("--lateral-tol", type=float, default=0.0002, help="ALIGN -> DESCEND lateral gate, m.")
ctrl.add_argument("--yaw-tol", type=float, default=0.005, help="ALIGN -> DESCEND yaw gate, rad.")
ctrl.add_argument("--height-tol", type=float, default=0.001, help="ALIGN -> DESCEND height gate, m (RT-67).")
ctrl.add_argument("--ik-lambda", type=float, default=0.05, help="dls damping (Isaac Lab default is 0.01).")
ctrl.add_argument(
    "--f-abort",
    type=float,
    default=None,
    help=(
        "Override the cfg force-abort limit FOR THIS RUN. The cfg value is a "
        "placeholder; raising it is how a truncated curve is opened up so the "
        "CLEAN peak can be found. Not a proposal for F_max."
    ),
)

ctrl.add_argument(
    "--control-mode",
    type=str,
    default=None,
    choices=["osc", "joint_pd"],
    help=(
        "Override env cfg control_mode FOR THIS RUN (inbox entry 'Audit 2026-09-03 (a)'). "
        "Default None = the cfg default (osc). 'joint_pd' is the D-108 chain, the PD half "
        "of that entry's discriminating measurement."
    ),
)
ctrl.add_argument(
    "--osc-kp-pos",
    type=float,
    default=None,
    help="Override env cfg osc_kp_pos (N/m) FOR THIS RUN; the two placeholder runs are 100 and 500.",
)
ctrl.add_argument(
    "--osc-kp-rot",
    type=float,
    default=None,
    help="Override env cfg osc_kp_rot (Nm/rad) FOR THIS RUN.",
)
ctrl.add_argument(
    "--decimation",
    type=int,
    default=None,
    help=(
        "Override the physics steps per control step FOR THIS RUN, in BOTH modes (it is "
        "written into cfg.decimation and cfg.osc_decimation, so resolve_control_mode yields "
        "it either way). A measurement setting: the two modes default to different policy "
        "rates (joint_pd 60 Hz, osc 15 Hz placeholder). ALONE it does NOT hold the per-episode "
        "wall time equal (RT-146, 2026-09-04): resolve_control_mode derives "
        "episode_length_s = episode_steps * sim.dt * decimation, so changing decimation alone "
        "changes how many seconds one episode covers. Pair it with --episode-steps to hold the "
        "seconds fixed while the rate moves."
    ),
)
ctrl.add_argument(
    "--episode-steps",
    type=int,
    default=None,
    help=(
        "Override cfg.episode_steps FOR THIS RUN, set before gym.make so resolve_control_mode "
        "derives episode_length_s from IT (episode_steps * sim.dt * decimation). D-113's 256 is "
        "control steps at the DEFAULT decimation; holding it fixed while --decimation changes "
        "changes the per-episode SECONDS, which is what made RT-144d and RT-146 compare a "
        "4.267 s episode against a 17.07 s one instead of the same wall time at two rates. Safe "
        "only because this script sets rl_terms_enabled = False, which skips validate_rl_config's "
        "episode_steps/time_penalty_per_step identity check."
    ),
)

tilt = parser.add_argument_group("tilt (measurement settings, not task decisions)")
tilt.add_argument(
    "--tilt-axis",
    type=str,
    default="y",
    choices=("x", "y"),
    help=(
        "Which pocket in-plane axis the part rotates about. 'x' is the SHORT "
        "axis (PLAY_X 0.5876 mm), 'y' the LONG one (PLAY_Y 1.60 mm). Which one "
        "the real insertion uses is NOT decided -- D-071 says edge-first and "
        "names no axis, so both are run and both are reported."
    ),
)
tilt.add_argument(
    "--tilt-sweep-deg",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help=(
        "Per-env start tilt from LO to HI degrees, endpoints included. A span "
        "that CROSSES ZERO measures the straight insertion and both signs in "
        "one run, which is what the F_max rule asks for. Overrides --tilt-deg."
    ),
)
tilt.add_argument(
    "--tilt-deg",
    type=float,
    default=0.0,
    help="Single start tilt for every env, degrees. The default 0.0 is the straight insertion.",
)
tilt.add_argument(
    "--upright-depth-mm",
    type=float,
    default=8.0,
    help=(
        "Depth by which the commanded tilt has ramped to zero, mm. The ramp "
        "starts at the opening plane. 0 means no tilt at any depth."
    ),
)
tilt.add_argument("--tilt-gain", type=float, default=0.5)
tilt.add_argument("--max-tilt-step-deg", type=float, default=0.6, help="Clip on the commanded tilt per step, deg.")
tilt.add_argument(
    "--tilt-tol-deg",
    type=float,
    default=0.2,
    help="ALIGN -> DESCEND tilt gate, deg: an env not yet at its commanded tilt stays in ALIGN.",
)

tilt.add_argument(
    "--stop-depth-mm",
    type=float,
    default=None,
    help=(
        "Depth at which the DESCENT STOPS, mm. Default: the CAD seat depth "
        "(POCKET_FLOOR_Z, read from CAD/Aufnahme_real_v1.stp on 2026-08-24). "
        "A SMALLER value stops short of the floor and is a measurement "
        "setting, not a new seat depth -- the geometry constant is not "
        "touched by this flag."
    ),
)

prof = parser.add_argument_group("profile (measurement settings)")
prof.add_argument("--depth-bin-mm", type=float, default=1.0, help="Depth bin width, mm.")
prof.add_argument("--depth-bins", type=int, default=40, help="Number of depth bins.")
prof.add_argument(
    "--capture-tilt-deg",
    type=float,
    nargs="*",
    default=[],
    help=(
        "OPTIONAL extra capture read-offs: for each angle, the shallowest depth "
        "from which the measured tilt stays below it. EMPTY BY DEFAULT on "
        "purpose -- D-106 (2)'s bounds (2.821 / 1.905 / 0.935 deg) are "
        "documentation of what the walls enforce, carried in DECISIONS.md and "
        "deliberately given no constant, so they are passed in or not used. The "
        "capture read-off against half the cross play always runs and needs no "
        "argument."
    ),
)

sapu = parser.add_argument_group("SAPU (measurement settings)")
sapu.add_argument(
    "--sdf-points",
    type=int,
    default=None,
    help="Sample points for the SAPU query. Default: the task cfg's own sdf_num_sample_points.",
)
sapu.add_argument(
    "--no-interpen",
    action="store_true",
    default=False,
    help="Skip the SAPU query entirely (no Warp, no trimesh, no OBJ). The force half still runs.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# GUARDED for the reason D-081 exists: RT-66 died on an import-time TorchScript
# failure whose traceback printed while the wrapper still reported exit 0.
try:
    import json  # noqa: E402

    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402

    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
    from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

    import isaaclab_tasks  # noqa: F401, E402
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

    import insertion.tasks  # noqa: F401, E402
    from insertion.tasks.direct.insertion import (  # noqa: E402
        insertion_math,
        insertion_paths,
        insertion_sdf,
        insertion_tasks_cfg,
        scripted_policy as sp,
    )
except BaseException:
    traceback.print_exc()
    exit_with(simulation_app, 1, tag="tilt_insert-import")


def _slice(obs, name: str):
    """Read one observation block by NAME, from the one table that owns it.

    Same helper and same reason as ``scripted_insert.py`` and ``seat_probe.py``:
    ``OBS_SLICES`` is the single home of the channel layout, and indexing by
    name keeps this script correct when a block moves.
    """
    lo, hi = insertion_math.OBS_SLICES[name]
    return obs[:, lo:hi]


def _percentiles(values: list) -> dict:
    """p50 / p90 / p95 / max over a list, or an explicit empty marker.

    NEAREST-RANK, no interpolation -- the same reading ``scripted_insert.py``
    uses, so two metrics files can be compared row by row. An empty list
    returns ``None``s, which land in the JSON as ``null`` rather than as NaN.
    """
    if not values:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "max": None}
    s = sorted(values)

    def at(p: float) -> float:
        return s[max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))]

    return {"n": len(s), "p50": at(0.5), "p90": at(0.9), "p95": at(0.95), "max": s[-1]}


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # A MEASUREMENT env, and here it is not only "a scripted controller never
    # learns". With the RL terms ON the force abort and the success termination
    # end episodes mid-descent, and the depth profile this run exists to
    # produce would stop exactly where it gets interesting. The abort that
    # stays armed is the CONTROLLER's own (advance_phase), which stops
    # commanding without ending the episode.
    env_cfg.rl_terms_enabled = False
    # PINNED (2026-09-12). The three Phase-5 scatter fields default ON, and
    # `obs_noise.resolve_obs_noise_model` does NOT consult `rl_terms_enabled`
    # -- a measurement env builds the noise model as readily as a training env
    # does, and Isaac Lab applies it to `obs_buf["policy"]` in `step()`. This
    # script reads its pose AND its force straight out of that buffer
    # (`obs = obs_dict["policy"]`, then `_slice(obs, "tip_rel" / "force" /
    # "yaw_cos_sin" / "ee_quat")`), so 2.5 mm of per-episode pocket bias would
    # move the depth column of the profile, 3.5 N of per-step jitter would
    # move the peak force this run exists to pay, and the D-183 belief error
    # would move `tip_rel` a third time. A measurement needs the observation
    # to BE the reading.
    # THE SCRIPT IS SPENT (2026-09-01, after RT-124..RT-129: both its rules
    # were withdrawn) AND IT STILL RUNS. That is exactly why the pins are
    # here: a spent instrument that anyone can still start must not quietly
    # measure a noised observation.
    env_cfg.obs_noise_pocket_pos_std_m = 0.0
    env_cfg.force_obs_noise_std_n = 0.0
    env_cfg.grasp_obs_offset_x_m = 0.0
    # THE START POSE STAYS THE HOME POSE (D-161). The cfg default is rung 0,
    # which starts the tool point inside the pocket -- a part already in the
    # hole cannot be inserted edge-first. Pinned here, not assumed.
    env_cfg.start_tip_above_entrance = None
    # Into the CFG and BEFORE gym.make, the same two reasons as
    # scripted_insert.py: one home for the limit, and the startup banner prints
    # during __init__.
    if args_cli.f_abort is not None:
        env_cfg.force_abort_f_max_n = float(args_cli.f_abort)
    # THE CONTROLLER OVERRIDES, into the cfg and before gym.make, for the same
    # two reasons as --f-abort: one home, and the startup banner is the proof.
    if args_cli.control_mode is not None:
        env_cfg.control_mode = args_cli.control_mode
    if args_cli.osc_kp_pos is not None:
        env_cfg.osc_kp_pos = float(args_cli.osc_kp_pos)
    if args_cli.osc_kp_rot is not None:
        env_cfg.osc_kp_rot = float(args_cli.osc_kp_rot)
    if args_cli.decimation is not None:
        env_cfg.decimation = int(args_cli.decimation)
        env_cfg.osc_decimation = int(args_cli.decimation)
    # BEFORE gym.make: resolve_control_mode (insertion_env.py:128, inside
    # __init__) reads cfg.episode_steps to derive episode_length_s. Setting
    # it here, not after, is what lets a run choose the per-episode SECONDS
    # independently of decimation (RT-146 fix, 2026-09-04).
    if args_cli.episode_steps is not None:
        env_cfg.episode_steps = int(args_cli.episode_steps)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.robot
    num_envs = unwrapped.num_envs
    # READ from the env, not from the flag: the env resolved it (and refused
    # an unknown mode) in its own constructor.
    control_mode = str(unwrapped._control_mode)

    # ---------------------------------------------------------------- fail fast
    peg_body_idx = unwrapped._peg_body_idx
    if peg_body_idx is None:
        raise SystemExit(
            f"[tilt_insert] the robot carries no body '{unwrapped.cfg.peg_body_name}', so there is no part "
            "to insert. Author it with scripts/author_tool_ur5e.py first. Bodies present: "
            + str(robot.body_names)
        )
    n_joints = len(robot.joint_names)
    if control_mode == "joint_pd" and n_joints != unwrapped.cfg.action_space:
        raise SystemExit(
            f"[tilt_insert] the articulation has {n_joints} joints {robot.joint_names} but the action "
            f"space is {unwrapped.cfg.action_space}. The action-to-joint mapping here is one-to-one "
            "and would silently address the wrong column."
        )
    jacobi_idx = peg_body_idx - 1 if robot.is_fixed_base else peg_body_idx

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": args_cli.ik_lambda},
        ),
        num_envs=num_envs,
        device=device,
    )

    # ---------------------------------------------------- geometry, all READ
    seat_depth_m = float(insertion_tasks_cfg.POCKET_SEAT_DEPTH)
    success_depth_m = float(insertion_tasks_cfg.SEATED_SUCCESS_DEPTH)
    play_x_m = float(insertion_tasks_cfg.PLAY_X)
    play_y_m = float(insertion_tasks_cfg.PLAY_Y)
    wall_x = float(insertion_tasks_cfg.POCKET_WALL_X)
    wall_y = float(insertion_tasks_cfg.POCKET_WALL_Y)

    tilt_axis_idx = sp.TILT_AXES[args_cli.tilt_axis]
    if args_cli.tilt_sweep_deg is not None:
        lo_deg, hi_deg = float(args_cli.tilt_sweep_deg[0]), float(args_cli.tilt_sweep_deg[1])
    else:
        lo_deg = hi_deg = float(args_cli.tilt_deg)
    tilt_start = sp.tilt_offsets(num_envs, math.radians(lo_deg), math.radians(hi_deg), device)
    _tilt_start_deg = [round(math.degrees(v), 4) for v in tilt_start.tolist()]
    upright_depth_m = float(args_cli.upright_depth_mm) / 1000.0
    max_tilt_step_rad = math.radians(float(args_cli.max_tilt_step_deg))
    # THE DESCENT NEEDS AN END. RT-128: DESCEND commanded a constant rate
    # into the pocket floor, 2072 samples piled into the last millimetre
    # against ~50 per bin elsewhere, force ~20 N -> the 300 N abort, 0 clean.
    # The default is the CAD floor; a smaller --stop-depth-mm stops short of
    # it and is a setting of THIS RUN, not a redefinition of the seat.
    stop_depth_m = (seat_depth_m if args_cli.stop_depth_mm is None
                    else float(args_cli.stop_depth_mm) / 1000.0)
    tilt_tol_rad = math.radians(float(args_cli.tilt_tol_deg))

    print("[tilt_insert] ---- the Kipp instrument: straight AND tilted insertion ----")
    print(f"[tilt_insert] marker: {SCRIPT_MARKER}")
    print(f"[tilt_insert] tool body: {unwrapped.cfg.peg_body_name} (index {peg_body_idx}), "
          f"jacobian row block {jacobi_idx}, fixed base {robot.is_fixed_base}")
    print(f"[tilt_insert] seat {seat_depth_m * 1000.0:.1f} mm, success at {success_depth_m * 1000.0:.1f} mm, "
          f"play {play_x_m * 1000.0:.4f} mm across / {play_y_m * 1000.0:.4f} mm along")
    _f_src = "cfg default" if args_cli.f_abort is None else "--f-abort, THIS RUN ONLY"
    if control_mode == "osc":
        print(f"[tilt_insert] control_mode osc: pose-delta action, OSC kp pos {unwrapped.cfg.osc_kp_pos} "
              f"rot {unwrapped.cfg.osc_kp_rot}, step limits {unwrapped.cfg.osc_pos_step_limit_m} m / "
              f"{unwrapped.cfg.osc_rot_step_limit_rad} rad, decimation {unwrapped.cfg.decimation} "
              f"({1.0 / (unwrapped.cfg.sim.dt * unwrapped.cfg.decimation):.1f} Hz control steps); "
              "the scripted command goes pocket -> env -> action, NO IK")
    else:
        print(f"[tilt_insert] control_mode joint_pd: D-108 joint-delta chain via dls IK, decimation "
              f"{unwrapped.cfg.decimation} ({1.0 / (unwrapped.cfg.sim.dt * unwrapped.cfg.decimation):.1f} Hz)")
    print(f"[tilt_insert] action_scale {unwrapped.cfg.action_scale} rad, "
          f"F_abort {unwrapped.cfg.force_abort_f_max_n} N (PLACEHOLDER, invented; from {_f_src})")
    print(f"[tilt_insert] tilt about the pocket {args_cli.tilt_axis}-axis "
          f"({'short, PLAY_X' if tilt_axis_idx == 0 else 'long, PLAY_Y'}), "
          f"start {lo_deg:+.3f} .. {hi_deg:+.3f} deg over {num_envs} envs, "
          f"ramped to zero by {args_cli.upright_depth_mm:.2f} mm of depth")
    print(f"[tilt_insert] descent stops at {stop_depth_m * 1000.0:.2f} mm of depth "
          f"({'CAD seat depth, the default' if args_cli.stop_depth_mm is None else '--stop-depth-mm'}); "
          f"below it the downward rate is zero and the arm holds its last target.")
    print("[tilt_insert] NOTE: this run sets NO threshold. F_max, the capture depth and the abort "
          "payment are DECISIONS; what follows is the distribution they need.")
    print("[tilt_insert] NOTE: this script is SPENT (2026-09-01). The F_max and capture-depth rules "
          "it was written for are withdrawn and the datasheet anchor with them; see the inbox entry "
          "'The tilted scripted insertion is SPENT ...'. The numbers below describe THIS controller.")

    # ------------------------------------------------------------ SAPU query
    # The env's own query, built here because the env only builds it with the
    # RL terms on (insertion_env.py:508-528) and this run needs them off.
    sapu_query = None
    sapu_note = "disabled by --no-interpen"
    sdf_points = int(args_cli.sdf_points) if args_cli.sdf_points is not None else int(
        unwrapped.cfg.sdf_num_sample_points
    )
    if not args_cli.no_interpen:
        try:
            sapu_query = insertion_sdf.SdfDistanceQuery(
                insertion_paths.resolve_part_obj_path(),
                sdf_points,
                float(unwrapped.cfg.sdf_max_dist),
                str(device),
                int(unwrapped.cfg.sdf_seed),
                mesh_obj_path=insertion_paths.resolve_pocket_obj_path(),
            )
            print(sapu_query.describe())
            sapu_note = (
                "part OBJ sampled, fixture OBJ queried -- the env's own SAPU pair. UNVERIFIED: the "
                "fixture mesh has 9 unpaired edges (RT-64) and the sign was never measured at the "
                "surface, so these are comparisons between tilt values, not absolute depths."
            )
        except BaseException:
            traceback.print_exc()
            print("[tilt_insert] FAIL: the SAPU query could not be built. The interpenetration half of "
                  "this run is half the reason it exists, so the run stops rather than reporting the "
                  "force half as if it were the whole measurement. Use --no-interpen to ask for the "
                  "force half deliberately.")
            env.close()
            return 1

    # ------------------------------------------------------------ script state
    phase = torch.full((num_envs,), sp.PHASE_ALIGN, device=device)
    yaw_target = torch.zeros(num_envs, 2, device=device)
    latched = torch.zeros(num_envs, 1, device=device)
    # THE TILT REFERENCE, latched exactly like the yaw target and for the
    # measured reason in `scripted_policy.latch_tilt_reference`: RT-127 read
    # the start attitude at -180 deg against the pocket axis, so steering
    # against that axis is steering against a constant of the setup.
    tilt_ref_cs = torch.zeros(num_envs, 2, device=device)
    tilt_latched = torch.zeros(num_envs, 1, device=device)
    # NO LATERAL SWEEP HERE. The aim stays the pocket axis: this run varies the
    # TILT, and varying two things at once would make neither column readable.
    # The lateral sweep is scripted_insert.py's --aim-sweep and stays there.
    aim = torch.zeros(num_envs, 2, device=device)
    peak_depth = torch.zeros(num_envs, device=device)
    peak_force = torch.zeros(num_envs, device=device)
    peak_interpen = torch.zeros(num_envs, device=device)
    peak_tilt = torch.zeros(num_envs, device=device)
    peak_axis_tilt = torch.zeros(num_envs, device=device)
    aborted = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # JOINT TORQUES, all six joints, every control step (inbox entry 'Audit
    # 2026-09-03 (a)', discriminating measurement). Read off
    # ``robot.data.applied_torque`` -- the actuator model's applied effort
    # after its own clip: under joint_pd that is stiffness*err +
    # damping*err_vel (the implicit drive's formula, an ESTIMATE of what
    # PhysX applied, not a read-back), under osc the OSC's effort target
    # after the maxForce clip. Saturation = |tau| >= 0.99 * maxForce, per
    # joint, counted in control steps. maxForce is READ
    # (``joint_effort_limits``, the USD's 150 / 28).
    tau_max_force = robot.data.joint_effort_limits[0].clone()
    peak_tau = torch.zeros(num_envs, n_joints, device=device)
    sat_steps = torch.zeros(num_envs, n_joints, device=device)
    torque_trace: list = []

    _n_bins = int(args_cli.depth_bins)
    _bin_width_m = float(args_cli.depth_bin_mm) / 1000.0
    bin_tau_counts = [[0] * _n_bins for _ in range(n_joints)]
    bin_tau_max = [[0.0] * _n_bins for _ in range(n_joints)]
    bin_tau_sum = [[0.0] * _n_bins for _ in range(n_joints)]
    bin_counts = [0] * _n_bins
    bin_force_max, bin_force_sum = [0.0] * _n_bins, [0.0] * _n_bins
    bin_ipen_counts = [0] * _n_bins
    bin_ipen_max, bin_ipen_sum = [0.0] * _n_bins, [0.0] * _n_bins
    bin_tilt_counts = [0] * _n_bins
    bin_tilt_max, bin_tilt_sum = [0.0] * _n_bins, [0.0] * _n_bins
    bin_lat_counts = [0] * _n_bins
    bin_lat_max, bin_lat_sum = [0.0] * _n_bins, [0.0] * _n_bins

    episodes: list = []
    first_step: dict | None = None

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    steps = 0
    while simulation_app.is_running() and steps < args_cli.max_steps:
        with torch.inference_mode():
            tip_rel = _slice(obs, "tip_rel")
            pocket_quat = _slice(obs, "pocket_quat")
            yaw_cs = _slice(obs, "yaw_cos_sin")
            force = _slice(obs, "force")
            ee_quat = _slice(obs, "ee_quat")

            yaw_target, latched = sp.latch_yaw_target(yaw_cs, yaw_target, latched)
            yaw_err = sp.yaw_error(yaw_cs, yaw_target)
            lat_err = sp.lateral_error(tip_rel, aim)
            depth = -tip_rel[:, 2]
            force_norm = insertion_math.force_magnitude(force)

            # THE LATERAL GATE, and it is not decoration: RT-107 is the standing
            # proof that a depth without it reads 0.0 mm while the part is
            # 109 mm beside the fixture. Computed from the OBSERVATION with the
            # env's own predicate, so no env internal enters the control path.
            in_pocket = insertion_math.in_pocket_cross_section(tip_rel, wall_x, wall_y)
            gated_depth = torch.where(depth > 0.0, depth, torch.zeros_like(depth))
            gated_depth = torch.where(in_pocket, gated_depth, torch.zeros_like(gated_depth))
            peak_depth = sp.update_peak_depth(gated_depth, peak_depth)

            # THE TILT, signed and against the schedule -- but measured FROM THE
            # LATCHED START ATTITUDE, not from the pocket axis. `tilt_now` is the
            # absolute reading and stays printed; `tilt_rel` is what the
            # controller and the report use, and it is zero at step 1 by
            # construction. `axis_tilt_angle` next to them is the unsigned
            # absolute magnitude the other runs report, kept so the two metrics
            # files compare.
            tilt_now = sp.signed_tilt_angle(ee_quat, pocket_quat, tilt_axis_idx)
            tilt_ref_cs, tilt_latched = sp.latch_tilt_reference(tilt_now, tilt_ref_cs, tilt_latched)
            tilt_rel = sp.tilt_since_reference(tilt_now, tilt_ref_cs)
            tilt_cmd = sp.tilt_schedule(depth, tilt_start, upright_depth_m)
            tilt_err = tilt_rel - tilt_cmd
            tilt_abs = sp.axis_tilt_angle(ee_quat, pocket_quat)
            peak_tilt = torch.maximum(tilt_rel.abs(), peak_tilt)
            peak_axis_tilt = torch.maximum(tilt_abs, peak_axis_tilt)
            peak_force = torch.maximum(force_norm, peak_force)

            # ------------------------------------------------ the instrument
            # Env internals ENTER HERE and nowhere above: the SAPU pose needs
            # the fixture pose, which is not an observation channel. Copied
            # line for line from insertion_env.py:1167-1171, including the
            # env-local part position -- _entrance_pos lives in that frame.
            if sapu_query is not None:
                part_pos = robot.data.body_pos_w[:, peg_body_idx] - unwrapped.scene.env_origins
                body_quat = robot.data.body_quat_w[:, peg_body_idx]
                sapu_pos, sapu_quat = insertion_sdf.goal_relative_transform(
                    part_pos, body_quat, unwrapped._entrance_pos, unwrapped._fixture_quat
                )
                interpen_max = insertion_math.max_interpen_dist(
                    sapu_query.interpen_distances(sapu_pos, sapu_quat)
                )
            else:
                interpen_max = torch.zeros_like(depth)
            peak_interpen = torch.maximum(interpen_max, peak_interpen)

            _d = depth.tolist()
            sp.bin_values_by_depth(_d, force_norm.tolist(), _bin_width_m,
                                   bin_counts, bin_force_max, bin_force_sum)
            sp.bin_values_by_depth(_d, [v * 1000.0 for v in interpen_max.tolist()], _bin_width_m,
                                   bin_ipen_counts, bin_ipen_max, bin_ipen_sum)
            # THE PROFILE REPORTS THE RELATIVE TILT, the same quantity the
            # controller regulates. The absolute one would print ~180 deg in
            # every bin (RT-127) and say nothing about how far the part leans.
            sp.bin_values_by_depth(_d, [abs(math.degrees(v)) for v in tilt_rel.tolist()], _bin_width_m,
                                   bin_tilt_counts, bin_tilt_max, bin_tilt_sum)
            # Lateral offset from the pocket AXIS, not from the aim: the capture
            # question is how far the part can still stray, and the aim is zero
            # in this run anyway.
            sp.bin_values_by_depth(
                _d,
                [v * 1000.0 for v in torch.linalg.norm(tip_rel[:, 0:2], dim=-1).tolist()],
                _bin_width_m, bin_lat_counts, bin_lat_max, bin_lat_sum,
            )

            # THE FIRST STEP, printed. Added 2026-09-01 after RT-124/125/126:
            # with the tilt controller live every env stopped at 0.000 mm, and
            # with it silenced (--tilt-gain 0 --tilt-tol-deg 200) the same run
            # reached 19.07-21.12 mm. That located the cause in the tilt
            # REFERENCE, and there was no way to read the reference off the log
            # -- this run printed no step-1 residual at all, so the one number
            # that decides the question existed only as a guess.
            #
            # THE SIGNED TILT IS THE ONE THAT MATTERS and it is printed next to
            # the unsigned one on purpose: `axis_tilt_angle` cannot tell an
            # aligned part from an ANTIPARALLEL one at a glance, and the
            # controller drives on the signed value.
            if first_step is None:
                first_step = {
                    "lateral_error_m": _percentiles([float(v) for v in lat_err.tolist()]),
                    "yaw_error_rad": _percentiles([abs(float(v)) for v in yaw_err.tolist()]),
                    "depth_m": _percentiles([float(v) for v in depth.tolist()]),
                    "force_norm_n": _percentiles([float(v) for v in force_norm.tolist()]),
                    "signed_tilt_rad": _percentiles([float(v) for v in tilt_now.tolist()]),
                    "tilt_since_reference_rad": _percentiles([float(v) for v in tilt_rel.tolist()]),
                    "commanded_tilt_rad": _percentiles([float(v) for v in tilt_cmd.tolist()]),
                    "tilt_error_rad": _percentiles([float(v) for v in tilt_err.tolist()]),
                    "axis_tilt_rad": _percentiles([float(v) for v in tilt_abs.tolist()]),
                }
                print(f"[tilt_insert] step 1 residuals: "
                      f"lateral p50 {first_step['lateral_error_m']['p50']:.6f} m, "
                      f"|yaw| p50 {first_step['yaw_error_rad']['p50']:.6f} rad, "
                      f"depth p50 {first_step['depth_m']['p50']:.6f} m, "
                      f"force p50 {first_step['force_norm_n']['p50']:.3f} N")
                print(f"[tilt_insert] step 1 TILT REFERENCE: "
                      f"signed p50 {math.degrees(first_step['signed_tilt_rad']['p50']):+.4f} deg, "
                      f"since latched reference p50 "
                      f"{math.degrees(first_step['tilt_since_reference_rad']['p50']):+.4f} deg, "
                      f"commanded p50 {math.degrees(first_step['commanded_tilt_rad']['p50']):+.4f} deg, "
                      f"error p50 {math.degrees(first_step['tilt_error_rad']['p50']):+.4f} deg, "
                      f"unsigned axis tilt p50 "
                      f"{math.degrees(first_step['axis_tilt_rad']['p50']):.4f} deg")
                print(f"[tilt_insert] step 1 tilt gate: {int((tilt_err.abs() >= tilt_tol_rad).sum())} "
                      f"of {num_envs} envs are OUTSIDE the {args_cli.tilt_tol_deg} deg gate and are "
                      "therefore held in ALIGN. All of them means nothing ever descends.")

            # ------------------------------------------------ the phase machine
            new_phase = sp.advance_phase(
                phase, lat_err, yaw_err, tip_rel[:, 2], force_norm,
                args_cli.lateral_tol, args_cli.yaw_tol,
                args_cli.standoff, args_cli.height_tol,
                float(unwrapped.cfg.force_abort_f_max_n),
            )
            # THE TILT GATE, held HERE and not inside advance_phase: that
            # function is shared with the D-108 gate run, and a fourth condition
            # there would put this run's schedule into that run's contract. An
            # env that would start descending before it has reached its
            # commanded tilt stays in ALIGN; everything else passes through.
            hold = (
                (phase == sp.PHASE_ALIGN)
                & (new_phase == sp.PHASE_DESCEND)
                & (tilt_err.abs() >= tilt_tol_rad)
            )
            phase = torch.where(hold, torch.full_like(new_phase, sp.PHASE_ALIGN), new_phase)
            aborted = aborted | (phase == sp.PHASE_ABORT_FORCE)

            cmd_pocket = sp.command_in_pocket_frame_tilted(
                tip_rel, phase, yaw_err, aim, tilt_err, tilt_axis_idx,
                args_cli.standoff, args_cli.lateral_gain, args_cli.vertical_gain, args_cli.yaw_gain,
                args_cli.descend_rate, args_cli.max_step, args_cli.max_yaw_step,
                args_cli.tilt_gain, max_tilt_step_rad, stop_depth_m,
            )

            if control_mode == "osc":
                # ONE hop: pocket -> env, then the per-step limits. The env's
                # OSC tracks the delta every physics step; no IK here.
                actions = sp.osc_action_from_pocket_command(
                    cmd_pocket, pocket_quat,
                    float(unwrapped.cfg.osc_pos_step_limit_m), float(unwrapped.cfg.osc_rot_step_limit_rad),
                )
            else:
                root_pose_w = robot.data.root_pose_w
                cmd_base = sp.command_to_base_frame(cmd_pocket, pocket_quat, root_pose_w[:, 3:7])

                ee_pose_w = robot.data.body_pose_w[:, peg_body_idx]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, :n_joints]
                ik.set_command(cmd_base, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
                joint_pos_des = ik.compute(ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos)

                actions = sp.joint_delta_to_action(
                    joint_pos_des, unwrapped._joint_targets, float(unwrapped.cfg.action_scale)
                )
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            # THE TORQUE READING, after the step so it is the effort of the
            # last physics substep this action produced. Per-joint peak per
            # episode, saturation count, and the depth profile per joint --
            # the depth is the PRE-step depth above, i.e. where the arm was
            # when the torque was demanded.
            tau = robot.data.applied_torque[:, :n_joints].abs()
            peak_tau = torch.maximum(tau, peak_tau)
            sat_steps = sat_steps + (tau >= 0.99 * tau_max_force.reshape(1, -1)).to(sat_steps.dtype)
            for _j in range(n_joints):
                sp.bin_values_by_depth(_d, tau[:, _j].tolist(), _bin_width_m,
                                       bin_tau_counts[_j], bin_tau_max[_j], bin_tau_sum[_j])
            torque_trace.append({
                "step": steps + 1,
                "depth_p50_mm": float(depth.median()) * 1000.0,
                "force_max_n": float(force_norm.max()),
                "abs_tau_max_nm": [round(float(v), 4) for v in tau.amax(dim=0).tolist()],
                "abs_tau_p50_nm": [round(float(v), 4) for v in tau.median(dim=0).values.tolist()],
                "envs_saturated": [int(v) for v in (tau >= 0.99 * tau_max_force.reshape(1, -1)).sum(dim=0).tolist()],
            })
            obs = obs_dict["policy"]
            done = torch.logical_or(terminated, truncated)

        steps += 1

        if bool(done.any()):
            # From the LAST PRE-RESET observation this loop saw. DirectRLEnv
            # resets inside step() and hands back the post-reset observation, so
            # the terminal one is not available here -- the same reset-step trap
            # scripted_insert.py records, and taking the post-reset numbers
            # would be strictly worse.
            for i in torch.nonzero(done).flatten().tolist():
                _clean = bool(peak_depth[i] >= success_depth_m) and not bool(aborted[i])
                episodes.append({
                    "env": int(i),
                    "tilt_start_deg": _tilt_start_deg[int(i)],
                    "peak_depth_m": float(peak_depth[i]),
                    "peak_force_n": float(peak_force[i]),
                    "peak_interpen_m": float(peak_interpen[i]),
                    "peak_tilt_since_reference_rad": float(peak_tilt[i]),
                    "peak_axis_tilt_rad": float(peak_axis_tilt[i]),
                    "final_phase": sp.PHASE_NAMES[float(phase[i])],
                    "force_aborted": bool(aborted[i]),
                    # THE RULE, not a judgement: the F_max mark's own words --
                    # a CLEAN insertion reached the success depth inside the
                    # pocket and did not abort on force.
                    "clean": _clean,
                    "terminated": bool(terminated[i]),
                    "truncated": bool(truncated[i]),
                    "peak_abs_torque_nm": [round(float(v), 4) for v in peak_tau[i].tolist()],
                    "saturated_steps": [int(v) for v in sat_steps[i].tolist()],
                })
            keep = torch.logical_not(done)
            phase = torch.where(keep, phase, torch.full_like(phase, sp.PHASE_ALIGN))
            latched = latched * keep.reshape(-1, 1).to(latched.dtype)
            # THE TILT REFERENCE IS RE-LATCHED PER EPISODE, exactly as the yaw
            # target above is. Without this line episode 2 steers against
            # episode 1's start attitude, and the reset draws wrist_3 yaw noise
            # (D-038), so the two are not the same pose. Missed on the first
            # write of the latch and found while reading RT-128.
            tilt_latched = tilt_latched * keep.reshape(-1, 1).to(tilt_latched.dtype)
            peak_depth = peak_depth * keep.to(peak_depth.dtype)
            peak_force = peak_force * keep.to(peak_force.dtype)
            peak_interpen = peak_interpen * keep.to(peak_interpen.dtype)
            peak_tilt = peak_tilt * keep.to(peak_tilt.dtype)
            peak_axis_tilt = peak_axis_tilt * keep.to(peak_axis_tilt.dtype)
            aborted = aborted & keep
            peak_tau = peak_tau * keep.reshape(-1, 1).to(peak_tau.dtype)
            sat_steps = sat_steps * keep.reshape(-1, 1).to(sat_steps.dtype)

    # -------------------------------------------------------------- the numbers
    force_rows = sp.depth_profile(_bin_width_m, bin_counts, bin_force_max, bin_force_sum, "force", "n")
    ipen_rows = sp.depth_profile(_bin_width_m, bin_ipen_counts, bin_ipen_max, bin_ipen_sum, "interpen", "mm")
    tilt_rows = sp.depth_profile(_bin_width_m, bin_tilt_counts, bin_tilt_max, bin_tilt_sum, "tilt", "deg")
    lat_rows = sp.depth_profile(_bin_width_m, bin_lat_counts, bin_lat_max, bin_lat_sum, "lateral", "mm")

    clean = [e for e in episodes if e["clean"]]
    _abort_n = sum(1 for e in episodes if e["force_aborted"])

    # THE CAPTURE READ-OFF. Half the cross play is the only tolerance used by
    # default, and it is READ from the config -- it is the distance the walls
    # leave the part once they hold it. The shallowest depth from which the
    # lateral offset never again exceeds it is what "captured" means
    # mechanically. It is offered as an INPUT to the engaged_depth_m decision,
    # not as the decision.
    capture = [{
        "quantity": "lateral_max_mm",
        "tolerance": play_x_m * 500.0,
        "tolerance_source": "PLAY_X / 2, insertion_tasks_cfg.PLAY_X",
        "first_depth_mm": sp.first_depth_below(lat_rows, "lateral_max_mm", play_x_m * 500.0),
    }]
    for a in args_cli.capture_tilt_deg:
        capture.append({
            "quantity": "tilt_max_deg",
            "tolerance": float(a),
            "tolerance_source": "--capture-tilt-deg, THIS RUN ONLY (D-106 (2) bounds are documentation); tilt_max_deg is |tilt since the latched start attitude|, not the angle to the pocket axis",
            "first_depth_mm": sp.first_depth_below(tilt_rows, "tilt_max_deg", float(a)),
        })

    by_tilt = []
    for i in range(num_envs):
        _eps = [e for e in episodes if e["env"] == i]
        by_tilt.append({
            "env": i,
            "tilt_start_deg": _tilt_start_deg[i],
            "episodes": len(_eps),
            "clean": sum(1 for e in _eps if e["clean"]),
            "peak_depth_mm": _percentiles([e["peak_depth_m"] * 1000.0 for e in _eps]),
            "peak_force_n": _percentiles([e["peak_force_n"] for e in _eps]),
            "peak_interpen_mm": _percentiles([e["peak_interpen_m"] * 1000.0 for e in _eps]),
        })
    by_tilt.sort(key=lambda r: (r["tilt_start_deg"], r["env"]))

    metrics = {
        "run": "tilt_insert (straight AND tilted scripted insertion)",
        "marker": SCRIPT_MARKER,
        "task": args_cli.task,
        "num_envs": num_envs,
        "steps": steps,
        "episodes": len(episodes),
        "geometry": {
            "seat_depth_mm": seat_depth_m * 1000.0,
            "success_depth_mm": success_depth_m * 1000.0,
            "play_x_mm": play_x_m * 1000.0,
            "play_y_mm": play_y_m * 1000.0,
        },
        "tilt": {
            "axis": args_cli.tilt_axis,
            "axis_index": tilt_axis_idx,
            "start_deg_lo_hi": [lo_deg, hi_deg],
            "start_deg_per_env": _tilt_start_deg,
            "upright_depth_mm": float(args_cli.upright_depth_mm),
            "gain": float(args_cli.tilt_gain),
            "max_step_deg": float(args_cli.max_tilt_step_deg),
            "align_gate_tol_deg": float(args_cli.tilt_tol_deg),
        },
        "controller": {
            "standoff_m": args_cli.standoff,
            "lateral_gain": args_cli.lateral_gain,
            "vertical_gain": args_cli.vertical_gain,
            "yaw_gain": args_cli.yaw_gain,
            "descend_rate_m": args_cli.descend_rate,
            "max_step_m": args_cli.max_step,
            "max_yaw_step_rad": args_cli.max_yaw_step,
            "lateral_tol_m": args_cli.lateral_tol,
            "yaw_tol_rad": args_cli.yaw_tol,
            "height_tol_m": args_cli.height_tol,
            "ik_method": "dls",
            "ik_lambda": args_cli.ik_lambda,
            "f_abort_override_n": args_cli.f_abort,
            "stop_depth_mm": stop_depth_m * 1000.0,
        },
        "env": {
            "control_mode": control_mode,
            "decimation": int(unwrapped.cfg.decimation),
            "control_rate_hz": 1.0 / (float(unwrapped.cfg.sim.dt) * int(unwrapped.cfg.decimation)),
            "osc_kp_pos": float(unwrapped.cfg.osc_kp_pos),
            "osc_kp_rot": float(unwrapped.cfg.osc_kp_rot),
            "osc_pos_step_limit_m": float(unwrapped.cfg.osc_pos_step_limit_m),
            "osc_rot_step_limit_rad": float(unwrapped.cfg.osc_rot_step_limit_rad),
            "osc_pos_clamp_m": float(unwrapped.cfg.osc_pos_clamp_m),
            "osc_tilt_clamp_deg": math.degrees(float(unwrapped.cfg.osc_tilt_clamp_rad)),
            "action_scale_rad": float(unwrapped.cfg.action_scale),
            "rl_terms_enabled": bool(unwrapped.cfg.rl_terms_enabled),
            "force_abort_f_max_n": float(unwrapped.cfg.force_abort_f_max_n),
            "fixture_pos_noise_xy": float(unwrapped.cfg.fixture_pos_noise_xy),
            "fixture_yaw_noise_rad": float(unwrapped.cfg.fixture_yaw_noise_rad),
            "fixture_tilt_noise_rad": float(unwrapped.cfg.fixture_tilt_noise_rad),
        },
        "sapu": {
            "enabled": sapu_query is not None,
            "num_sample_points_requested": sdf_points,
            "num_points": None if sapu_query is None else int(sapu_query.num_points),
            "max_dist_m": float(unwrapped.cfg.sdf_max_dist),
            "seed": int(unwrapped.cfg.sdf_seed),
            "note": sapu_note,
        },
        "peak_depth_mm": _percentiles([e["peak_depth_m"] * 1000.0 for e in episodes]),
        "peak_force_n": _percentiles([e["peak_force_n"] for e in episodes]),
        "peak_interpen_mm": _percentiles([e["peak_interpen_m"] * 1000.0 for e in episodes]),
        "peak_tilt_since_reference_rad": _percentiles([e["peak_tilt_since_reference_rad"] for e in episodes]),
        "peak_axis_tilt_rad": _percentiles([e["peak_axis_tilt_rad"] for e in episodes]),
        "clean_insertions": {
            "rule": (
                "peak depth INSIDE the pocket cross-section reached SEATED_SUCCESS_DEPTH and the "
                "episode did not end in ABORT_FORCE -- the wording of the force_abort_f_max_n "
                "placeholder mark, insertion_env_cfg.py:940-952"
            ),
            "n": len(clean),
            "n_episodes": len(episodes),
            "peak_force_n": _percentiles([e["peak_force_n"] for e in clean]),
            "peak_interpen_mm": _percentiles([e["peak_interpen_m"] * 1000.0 for e in clean]),
            "f_max_lower_bound_n": max([e["peak_force_n"] for e in clean], default=None),
            "f_max_upper_anchor_n": None,
            "f_max_upper_anchor_source": (
                "WITHDRAWN 2026-09-01 -- the suction cup is not modelled (rigid chain, no sim-to-real); "
                "see the inbox entry 'The tilted scripted insertion is SPENT ...'."
            ),
        },
        "abort_truncated": {
            "n_episodes_aborted": _abort_n,
            "limit_n": float(unwrapped.cfg.force_abort_f_max_n),
            "note": (
                "an aborted episode's peak force IS the limit, not the task's. A lower bound for "
                "F_max read from a run with aborts is a lower bound on the LIMIT, not on the task."
            ),
        },
        "first_step": first_step,
        "capture_candidates": capture,
        "by_tilt": by_tilt,
        "profiles": {
            "force": force_rows,
            "interpen": ipen_rows,
            "tilt": tilt_rows,
            "lateral": lat_rows,
        },
        "per_episode": episodes,
        "joint_torques": {
            "source": (
                "robot.data.applied_torque -- the actuator model's applied effort after its clip: "
                "under joint_pd the implicit drive formula stiffness*err + damping*err_vel (an "
                "estimate, not a PhysX read-back), under osc the OSC effort target after the maxForce clip"
            ),
            "joint_names": list(robot.joint_names),
            "max_force_nm": [float(v) for v in tau_max_force.tolist()],
            "saturation_rule": "|tau| >= 0.99 * maxForce, counted in control steps",
            "peak_abs_per_joint_nm": [
                _percentiles([e["peak_abs_torque_nm"][_j] for e in episodes]) for _j in range(n_joints)
            ],
            "peak_abs_per_joint_clean_nm": [
                _percentiles([e["peak_abs_torque_nm"][_j] for e in clean]) for _j in range(n_joints)
            ],
            "saturated_steps_per_joint_total": [
                int(sum(e["saturated_steps"][_j] for e in episodes)) for _j in range(n_joints)
            ],
            "profile_per_joint": [
                sp.depth_profile(_bin_width_m, bin_tau_counts[_j], bin_tau_max[_j], bin_tau_sum[_j], "tau", "nm")
                for _j in range(n_joints)
            ],
        },
        "torque_trace": torque_trace,
    }
    out = pathlib.Path(args_cli.out)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # PRINTED AS WELL AS WRITTEN (RT-70): a metrics file whose path is printed
    # but whose contents are not proves nothing in a log, and the log is what
    # /rt-check reads.
    print(f"[tilt_insert] {len(episodes)} episodes, {len(clean)} clean, {_abort_n} force-aborted")
    print("[tilt_insert] start tilt -> peak depth / peak force / peak interpenetration:")
    for r in by_tilt:
        _pd = r["peak_depth_mm"]["p50"]
        _pf = r["peak_force_n"]["p50"]
        _pi = r["peak_interpen_mm"]["p50"]
        print(f"[tilt_insert]   tilt {r['tilt_start_deg']:+7.3f} deg  env {r['env']:3d}  "
              f"episodes {r['episodes']:3d}  clean {r['clean']:3d}  "
              f"depth p50 {float('nan') if _pd is None else _pd:8.3f} mm  "
              f"force p50 {float('nan') if _pf is None else _pf:8.2f} N  "
              f"interpen p50 {float('nan') if _pi is None else _pi:8.4f} mm")
    print("[tilt_insert] profile over depth (force N / interpenetration mm / tilt deg / lateral mm):")
    _ipen = {r["depth_lo_mm"]: r for r in ipen_rows}
    _tlt = {r["depth_lo_mm"]: r for r in tilt_rows}
    _lat = {r["depth_lo_mm"]: r for r in lat_rows}
    for r in force_rows:
        _k = r["depth_lo_mm"]
        print(f"[tilt_insert]   {_k:6.1f}-{r['depth_hi_mm']:6.1f} mm  n {r['samples']:6d}  "
              f"force max {r['force_max_n']:8.2f} mean {r['force_mean_n']:8.2f}  "
              f"interpen max {_ipen.get(_k, {}).get('interpen_max_mm', float('nan')):7.4f}  "
              f"tilt max {_tlt.get(_k, {}).get('tilt_max_deg', float('nan')):7.3f}  "
              f"lateral max {_lat.get(_k, {}).get('lateral_max_mm', float('nan')):7.4f}")
    print("[tilt_insert] joint torques (|tau| in Nm; per-episode peaks over all episodes; saturation "
          "= steps at >= 0.99 maxForce):")
    for _j, _jn in enumerate(robot.joint_names):
        _pk = metrics["joint_torques"]["peak_abs_per_joint_nm"][_j]
        _pc = metrics["joint_torques"]["peak_abs_per_joint_clean_nm"][_j]
        _pk_max = _pk["max"] if _pk["max"] is not None else float("nan")
        _pc_max = _pc["max"] if _pc["max"] is not None else float("nan")
        print(f"[tilt_insert]   {_jn:<20} maxForce {float(tau_max_force[_j]):7.2f}  peak {_pk_max:8.3f} "
              f"({_pk_max / float(tau_max_force[_j]) * 100.0:5.1f} %)  peak clean {_pc_max:8.3f}  "
              f"saturated steps {metrics['joint_torques']['saturated_steps_per_joint_total'][_j]:6d}")
    print("[tilt_insert] torque profile over depth (max |tau| per joint, Nm):")
    _tau_rows = metrics["joint_torques"]["profile_per_joint"]
    for _r in _tau_rows[0]:
        _k = _r["depth_lo_mm"]
        _cols = []
        for _j in range(n_joints):
            _row = next((x for x in _tau_rows[_j] if x["depth_lo_mm"] == _k), None)
            _cols.append(float("nan") if _row is None else _row["tau_max_nm"])
        print(f"[tilt_insert]   {_k:6.1f}-{_r['depth_hi_mm']:6.1f} mm  n {_r['samples']:6d}  "
              + "  ".join(f"{v:8.3f}" for v in _cols))
    for c in capture:
        print(f"[tilt_insert] capture read-off: {c['quantity']} stays below {c['tolerance']:.4f} "
              f"from depth {c['first_depth_mm']} mm  (tolerance: {c['tolerance_source']})")
    _lb = metrics["clean_insertions"]["f_max_lower_bound_n"]
    print(f"[tilt_insert] F_max LOWER BOUND from clean insertions: {_lb} N "
          f"(over {len(clean)} clean episodes) -- a bound on THIS controller. UPPER ANCHOR: withdrawn.")
    if _abort_n:
        print(f"[tilt_insert] WARNING: {_abort_n} episodes hit the {unwrapped.cfg.force_abort_f_max_n} N "
              "abort. The force curve is TRUNCATED and its peak is the limit, not the task. "
              "Re-run with a higher --f-abort before reading a lower bound off it.")
    print(f"[tilt_insert] metrics -> {out.resolve()}")
    print("[tilt_insert] READ IT LIKE THIS: every number above is a property of THIS scripted "
          "controller (its gains, its tilt ramp), not of the task. The capture read-off follows "
          "--upright-depth-mm. The interpenetration column is a comparison between tilt values, not an "
          "absolute penetration depth (RT-64).")

    env.close()
    return 0


if __name__ == "__main__":
    # D-081: set the exit status BEFORE the shutdown. close() does not return
    # (RT-22), so anything after it never decides anything.
    _code = 1
    try:
        _code = main()
    except BaseException:
        traceback.print_exc()
        exit_with(simulation_app, 1, tag="tilt_insert-main")
    exit_with(simulation_app, _code, tag="tilt_insert")
