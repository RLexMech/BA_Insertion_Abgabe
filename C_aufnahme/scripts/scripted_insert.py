# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""The scripted controller. Two modes. Runs on the training PC only.

WHAT THIS RUN DECIDES
---------------------
**THE DESCENT IS NO LONGER A GATE (user, 2026-08-29).** D-108 used to attach
one: before the first training run a SCRIPTED insertion had to place the part.
That gate is SUPERSEDED -- RT-81 measured the seated state directly (32.965 mm
at 8.08 N against ``SEATED_SUCCESS_DEPTH`` 33.000 mm), so "is the seat
reachable" is answered by measurement instead of by a controller, and
hand-building the search motion a 0.2-0.8 mm clearance needs is
classical-control work of its own size. Entry: ``docs/decisions_inbox.md``,
"The D-108 gate drops the scripted descent". D-108 carries an inline
correction. Do not put the gate back without reading that entry.

What the two modes are for now:

* DEFAULT (the descent) -- still the instrument for the numbers below. It
  measures; it gates nothing.
* ``--free-space`` (H1) -- hold the part above the opening, command a lateral
  offset of the play's order, and read the RESIDUAL. This is what the same
  decision assigns to D-108's H1, and ``action_scale = 0.02`` stays
  UNCONFIRMED until it runs. H2 needs no run: RT-74 already climbs to 122.2 N
  at 19 mm and stalls at 227.2 N.

The run also PAYS for open numbers that block the RL path today:

* ``force_abort_f_max_n`` -- currently an INVENTED number (RL_PLACEHOLDERS).
  This run measures the force distribution a successful insertion produces.
* ``engaged_depth_m`` -- UNSET (RL_PENDING). This run measures the reached
  depth as a distribution and sets no threshold.
* ``rung_step_sizes`` -- UNSET. The per-rung offsets come from sweeping the
  reset noise with ``env.fixture_pos_noise_xy=...`` and friends.

WHAT IT MAY LOOK AT
-------------------
Target information comes ONLY from the OBSERVATION -- the same 28 channels the
policy will get (user decision, 2026-08-28). This script never reads
``_entrance_pos``, ``_tilt_rot``, ``_fixture_quat`` or ``_peg_geometry()``.

The ROBOT model is not target information and stays allowed: the Jacobian, the
joint positions and the base pose are what any real controller has. The one
env internal it does read is ``_joint_targets`` -- the integrator its own
action feeds -- and the reason is in ``scripted_policy.joint_delta_to_action``.

THE CONTROL CHAIN
-----------------
observation -> pose delta in the POCKET frame -> env frame -> robot BASE frame
-> ``DifferentialIKController`` (``dls``) -> desired joint position -> the
env's joint-delta action. Every hop is a checked function in
``scripted_policy.py``; this file wires them to Isaac and owns no algebra.

The IK controller and the Jacobian come from Isaac Lab, not from us. The frame
convention is the one its own example uses
(``scripts/tutorials/05_controllers/run_diff_ik.py:162-171``): the Jacobian is
expressed in the ROOT frame, so the EE pose is converted there with
``subtract_frame_transforms`` before ``compute``.

The Jacobian is taken at the TOOL LINK, not at the part tip. For the
translation that is the same command. For the yaw it is also the same command,
because ``peg_tip_offset`` is a pure ``+z`` offset in the tool frame
(``insertion_env_cfg.py:398``), so the tip lies ON the tool axis and a rotation
about that axis leaves it where it is.

    .\scripts\rt_log.ps1 RT-66 python scripts\scripted_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 16 --headless --max-steps 600

H1, free space, ONE episode. ``--max-steps`` must stay below the 256-step
episode, or the reset teleports the part home and the settling trace crosses
that jump:

    .\scripts\rt_log.ps1 RT-84 python scripts\scripted_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 16 --headless --free-space --aim-sweep x --max-steps 250
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import math
import pathlib
import sys
import traceback

from isaaclab.app import AppLauncher

# Loaded by path because scripts/tools is not a package -- the idiom
# zero_agent.py and author_tool_ur5e.py already use.
_TOOLS = pathlib.Path(__file__).resolve().parent / "tools"
_spec = importlib.util.spec_from_file_location("isaac_exit", _TOOLS / "isaac_exit.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["isaac_exit"] = _mod
_spec.loader.exec_module(_mod)
exit_with = _mod.exit_with

# Printed at startup as `[scripted_insert] marker: <value>`, the pattern
# scripts/add_fixture_collision.py already uses. ADDED for RT-75: until then
# this script had no marker at all, so the only line that could prove the
# training PC was on the right commit was insertion_env's CODE_MARKER -- which
# says nothing about THIS file. A stale scripted_insert.py would have run its
# old controller under a fresh env and no log could have shown it.
SCRIPT_MARKER = "scripted_insert-2026-09-03a-h1"

parser = argparse.ArgumentParser(description="Scripted controller: descent (measurement) or --free-space (D-108 H1).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--max-steps",
    type=int,
    default=600,
    help=(
        "Stop after this many env steps and shut down normally. Without a cap the loop "
        "can only end by Ctrl+C, which kills the rt_log wrapper before it writes its "
        "exit line (RT-46, RT-50)."
    ),
)
parser.add_argument(
    "--out",
    type=str,
    default="scripted_insert_metrics.json",
    help="Where to write the metrics file. Numbers are read from THIS file, never from the console.",
)

# --- The controller's knobs. NONE of these is a decided task value: this is a
# --- measurement instrument, and every one of them is written into the metrics
# --- file so a number can never be reported without the settings that made it.
tune = parser.add_argument_group("controller (measurement settings, not task decisions)")
tune.add_argument("--standoff", type=float, default=0.010, help="m above the opening held during ALIGN.")
tune.add_argument("--lateral-gain", type=float, default=0.5, help="Proportional gain on the lateral error.")
tune.add_argument("--vertical-gain", type=float, default=0.5, help="Proportional gain on the standoff error.")
tune.add_argument("--yaw-gain", type=float, default=0.5, help="Proportional gain on the yaw error.")
tune.add_argument("--descend-rate", type=float, default=0.0005, help="m per control step during DESCEND.")
tune.add_argument("--max-step", type=float, default=0.002, help="m, cap on any commanded translation per step.")
tune.add_argument("--max-yaw-step", type=float, default=0.010, help="rad, cap on the commanded yaw per step.")
tune.add_argument("--lateral-tol", type=float, default=0.0002, help="m, ALIGN -> DESCEND lateral gate.")
tune.add_argument("--yaw-tol", type=float, default=0.005, help="rad, ALIGN -> DESCEND yaw gate.")
tune.add_argument(
    "--height-tol",
    type=float,
    default=0.001,
    help=(
        "m, ALIGN -> DESCEND height gate: the tip must be at or below "
        "standoff + this before the fine descent takes over. RT-67 ran without "
        "it and every episode crawled the full 165 mm at the descend rate."
    ),
)
tune.add_argument("--ik-lambda", type=float, default=0.05, help="dls damping (Isaac Lab default is 0.01).")
tune.add_argument(
    "--aim-sweep",
    choices=("none", "x", "y"),
    default="none",
    help=(
        "sweep the LATERAL AIM across the play, one fixed offset per env, and "
        "report peak depth against it. 'none' aims every env at the pocket "
        "axis and reproduces RT-74 exactly. 'x' spans PLAY_X (the 0.5876 mm "
        "short axis), 'y' spans PLAY_Y (the 1.60 mm long axis), endpoints "
        "included. RT-74 measured the force rising SMOOTHLY from ~8 mm with no "
        "jump at any bin, which points at friction over the travel rather than "
        "at an obstacle -- and every run so far aimed at the ONE point at the "
        "centre. This asks which lateral positions let the part go down at all."
    ),
)
tune.add_argument(
    "--depth-bin-mm",
    type=float,
    default=1.0,
    help=(
        "mm, width of one depth bin in the force-against-depth profile. The "
        "peak numbers say WHAT the force reached and never WHEN; this profile "
        "separates a force that rises from first contact (friction over the "
        "whole travel) from one that jumps at a depth (something engages "
        "there, and the depth names it)."
    ),
)
tune.add_argument(
    "--depth-bins",
    type=int,
    default=40,
    help=(
        "number of depth bins. 40 at 1 mm covers 0..40 mm, i.e. past the 36 mm "
        "seat. Samples beyond the last bin are DROPPED, never piled onto it."
    ),
)
tune.add_argument(
    "--free-space",
    action="store_true",
    default=False,
    help=(
        "H1 MODE. Block the ALIGN -> DESCEND transition, so the part regulates "
        "its lateral aim and its yaw at --standoff above the opening and never "
        "touches anything. What it measures is the RESIDUAL of that placement: "
        "D-108's H1 says terminal accuracy is set by PD tracking rather than by "
        "the action scale, and action_scale = 0.02 stays UNCONFIRMED until this "
        "runs. Pair it with --aim-sweep x so the commanded offset is of the "
        "play's order instead of zero -- an aim of zero commands no move at "
        "all, and RT-67 already measured the home pose as aligned to 1 um."
    ),
)
tune.add_argument(
    "--settle-tail",
    type=int,
    default=50,
    help=(
        "steps, the window at the END of the run whose WORST lateral error is "
        "reported as the settled placement. The whole-run mean is not the H1 "
        "number: it is dominated by the approach transient. Only used with "
        "--free-space."
    ),
)
tune.add_argument(
    "--f-abort",
    type=float,
    default=None,
    help=(
        "N, override force_abort_f_max_n for THIS run. Default None = use the cfg, so "
        "RT-66/67/68 stay reproducible. It is a flag and not a hydra override because "
        "this entry point has no hydra: parse_env_cfg reads only task/device/num_envs/"
        "use_fabric and never touches sys.argv, and parse_args() below rejects a bare "
        "'env.x=y' token with exit code 2 before Isaac even starts. RT-69 needs it "
        "because RT-68 ended every episode on the invented 50 N -- the placeholder "
        "truncates the very force curve this gate exists to measure."
    ),
)

tune.add_argument(
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
tune.add_argument(
    "--osc-kp-pos",
    type=float,
    default=None,
    help="Override env cfg osc_kp_pos (N/m) FOR THIS RUN; the two placeholder runs are 100 and 500.",
)
tune.add_argument(
    "--osc-kp-rot",
    type=float,
    default=None,
    help="Override env cfg osc_kp_rot (Nm/rad) FOR THIS RUN.",
)
tune.add_argument(
    "--decimation",
    type=int,
    default=None,
    help=(
        "Override the physics steps per control step FOR THIS RUN, in BOTH modes (it is "
        "written into cfg.decimation and cfg.osc_decimation, so resolve_control_mode yields "
        "it either way). A measurement setting: the two modes default to different policy "
        "rates (joint_pd 60 Hz, osc 15 Hz placeholder) and this flag is what holds the rate "
        "equal when the rate itself must be ruled out."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# GUARDED, and RT-66 is why. That run died HERE -- TorchScript refused to
# compile scripted_policy on import -- and the traceback printed while the
# wrapper still reported `exit code: 0`, because this block runs BEFORE
# `if __name__ == "__main__"` and the exit handler down there never saw it.
# A failing run that reports success is the one outcome D-081 exists to
# prevent, so the import path now leaves through the same door as main().
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
        insertion_tasks_cfg,
        scripted_policy as sp,
    )
except BaseException:
    traceback.print_exc()
    exit_with(simulation_app, 1, tag="scripted_insert-import")


def _slice(obs: torch.Tensor, name: str) -> torch.Tensor:
    """Read one observation block by NAME, from the one table that owns it.

    ``insertion_math.OBS_SLICES`` is the single home of the channel layout
    (``insertion_env.py`` builds the observation against the same table and
    ``check_env_wiring.py`` pins the two together). Indexing by name rather
    than by number is what keeps this script correct when a block moves.
    """
    lo, hi = insertion_math.OBS_SLICES[name]
    return obs[:, lo:hi]


def _percentiles(values: list[float]) -> dict:
    """p50 / p90 / p95 / max over a list, or an explicit empty marker.

    Written out rather than pulled from numpy so an empty list produces the
    word ``null`` in the metrics file instead of a NaN that later reads as a
    number.
    """
    if not values:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "max": None}
    s = sorted(values)

    def q(p: float) -> float:
        # Nearest-rank, the definition that needs no interpolation policy.
        k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return s[k]

    return {"n": len(s), "p50": q(0.50), "p90": q(0.90), "p95": q(0.95), "max": s[-1]}


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # A scripted controller never learns, so this is a MEASUREMENT env by
    # definition -- the same state zero_agent.py runs in. It is also the only
    # state that BUILDS while three RL_PENDING values are unset, and measuring
    # two of them is exactly what this run is for. The startup report says so
    # loudly, and so does the metrics file below.
    env_cfg.rl_terms_enabled = False
    # PINNED (2026-09-12). The three Phase-5 scatter fields default ON, and
    # `obs_noise.resolve_obs_noise_model` does NOT consult `rl_terms_enabled`
    # -- the noise model is built in a measurement env too, and Isaac Lab
    # applies it to `obs_buf["policy"]` in `step()`. This gate takes ALL its
    # target information from that buffer (the D-108 information boundary,
    # held in place by `check_env_wiring.py`), so the scatter would not be a
    # disturbance on the measurement, it would BE the controller's target: a
    # 2.5 mm pocket bias aims the descent 2.5 mm off, and the D-183 belief
    # error moves the same three channels again. The gate measures the
    # geometry, not the noise model.
    env_cfg.obs_noise_pocket_pos_std_m = 0.0
    env_cfg.force_obs_noise_std_n = 0.0
    env_cfg.grasp_obs_offset_x_m = 0.0
    # THE START POSE STAYS THE HOME POSE (D-161, 2026-08-31). The cfg default
    # is now rung 0 -- the env solves an IK at every reset and starts the tool
    # point inside the pocket. This run does its OWN teleport and reports
    # against the 165 mm home stand-off, so a rung-0 reset would move the pose
    # this script claims to command. Pinned here, not assumed.
    env_cfg.start_tip_above_entrance = None

    # THE ABORT LIMIT, WRITTEN INTO THE CFG AND BEFORE gym.make. Both parts
    # matter and neither is style.
    #   Into the CFG, not into a local: the limit has one home, and the env's
    #   own startup banner and demo_metrics.json both read that home. A second
    #   variable here would let the console report 50 while the run used 200.
    #   BEFORE gym.make: the banner prints during __init__. Assigning after the
    #   env is built would run the raised limit and print the old one -- and
    #   the printed line is the only proof RT-69 has that the override arrived.
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

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.robot
    num_envs = unwrapped.num_envs
    # READ from the env, not from the flag: the env resolved it (and refused
    # an unknown mode) in its own constructor.
    control_mode = str(unwrapped._control_mode)

    # ---------------------------------------------------------------- fail fast
    # No welded tool means no part, and a "scripted insertion" without a part
    # would report a success rate for a task that was never attempted. The env
    # already resolves this index and leaves it None when the tool is missing.
    peg_body_idx = unwrapped._peg_body_idx
    if peg_body_idx is None:
        raise SystemExit(
            f"[scripted_insert] the robot carries no body '{unwrapped.cfg.peg_body_name}', so there is no "
            "part to insert. This gate is meaningless without the welded tool -- author it with "
            "scripts/author_tool_ur5e.py first. Bodies present: " + str(robot.body_names)
        )
    n_joints = len(robot.joint_names)
    if n_joints != unwrapped.cfg.action_space:
        raise SystemExit(
            f"[scripted_insert] the articulation has {n_joints} joints {robot.joint_names} but the action "
            f"space is {unwrapped.cfg.action_space}. The action-to-joint mapping in this script is "
            "one-to-one and would silently address the wrong column."
        )

    # The Jacobian row block for a body. Isaac Lab's own example drops one for a
    # FIXED base, because a fixed base contributes no rows
    # (run_diff_ik.py:133-136). Read from the articulation, never assumed.
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

    # ------------------------------------------------ H1 free-space fail fast
    # ONE EPISODE, and the reason is the trace. A reset teleports the part back
    # to the home pose, the lateral error jumps, and a settling reading taken
    # across that jump measures the reset instead of the controller. The
    # episode is `episode_length_s * 60` control steps; the env computes the
    # same number itself, so it is READ here, never retyped.
    max_episode_length = int(unwrapped.max_episode_length)
    if args_cli.free_space and args_cli.max_steps >= max_episode_length:
        raise SystemExit(
            f"[scripted_insert] --free-space needs --max-steps below the episode length "
            f"({args_cli.max_steps} >= {max_episode_length}). One reset inside the run "
            "makes the settling trace unreadable, so the run is refused rather than "
            f"reported. Try --max-steps {max_episode_length - 6}."
        )
    if args_cli.free_space and args_cli.settle_tail > args_cli.max_steps:
        raise SystemExit(
            f"[scripted_insert] --settle-tail {args_cli.settle_tail} exceeds --max-steps "
            f"{args_cli.max_steps}; the tail would be the whole run including the "
            "approach transient, which is the one thing it exists to exclude."
        )

    print("[scripted_insert] ---- the D-108 gate ----" if not args_cli.free_space
          else "[scripted_insert] ---- H1: free-space lateral tracking ----")
    print(f"[scripted_insert] marker: {SCRIPT_MARKER}")
    print(f"[scripted_insert] tool body: {unwrapped.cfg.peg_body_name} (index {peg_body_idx}), "
          f"jacobian row block {jacobi_idx}, fixed base {robot.is_fixed_base}")
    print(f"[scripted_insert] joints: {robot.joint_names}")
    # The provenance is DERIVED from the flag, not typed out. The old line said
    # "(PLACEHOLDER, invented)" unconditionally, which stops being the whole
    # truth the moment --f-abort moves the number for one run.
    _f_src = "cfg default" if args_cli.f_abort is None else "--f-abort, THIS RUN ONLY"
    if control_mode == "osc":
        print(f"[scripted_insert] control_mode osc: pose-delta action, OSC kp pos {unwrapped.cfg.osc_kp_pos} "
              f"rot {unwrapped.cfg.osc_kp_rot}, step limits {unwrapped.cfg.osc_pos_step_limit_m} m / "
              f"{unwrapped.cfg.osc_rot_step_limit_rad} rad, decimation {unwrapped.cfg.decimation} "
              f"({1.0 / (unwrapped.cfg.sim.dt * unwrapped.cfg.decimation):.1f} Hz control steps); "
              "the scripted command goes pocket -> env -> action, NO IK")
    else:
        print(f"[scripted_insert] control_mode joint_pd: D-108 joint-delta chain via dls IK, decimation "
              f"{unwrapped.cfg.decimation} ({1.0 / (unwrapped.cfg.sim.dt * unwrapped.cfg.decimation):.1f} Hz)")
    print(f"[scripted_insert] action_scale {unwrapped.cfg.action_scale} rad, "
          f"F_abort {unwrapped.cfg.force_abort_f_max_n} N "
          f"(PLACEHOLDER, invented; from {_f_src})")

    # ------------------------------------ the lateral aim, one offset per env
    # The span is READ from the task config, never retyped here: PLAY_X and
    # PLAY_Y are derived from the measured opening minus the measured part
    # (insertion_tasks_cfg.py), and a hand-typed copy would stop tracking them
    # the day either input is re-measured.
    _aim_span_m = {
        "none": 0.0,
        "x": float(insertion_tasks_cfg.PLAY_X),
        "y": float(insertion_tasks_cfg.PLAY_Y),
    }[args_cli.aim_sweep]
    aim = sp.aim_offsets(num_envs, args_cli.aim_sweep, _aim_span_m, device)
    _aim_mm = [[round(v * 1000.0, 4) for v in row] for row in aim.tolist()]
    if args_cli.aim_sweep == "none":
        print("[scripted_insert] aim sweep: none -- every env aims at the pocket axis "
              "(reproduces RT-74)")
    else:
        print(f"[scripted_insert] aim sweep: {args_cli.aim_sweep} over "
              f"{_aim_span_m * 1000.0:.4f} mm of play, {num_envs} envs, "
              f"{-_aim_span_m * 500.0:+.4f} .. {_aim_span_m * 500.0:+.4f} mm "
              f"(pocket frame; col 0 = short axis, col 1 = long axis)")

    # ------------------------------------------------------------ script state
    # Per-env, carried by THIS script and not by the env: the env knows nothing
    # about phases and must not, or the gate would be measuring itself.
    phase = torch.full((num_envs,), sp.PHASE_ALIGN, device=device)
    yaw_target = torch.zeros(num_envs, 2, device=device)
    latched = torch.zeros(num_envs, 1, device=device)
    peak_depth = torch.zeros(num_envs, device=device)
    peak_force = torch.zeros(num_envs, device=device)
    peak_tilt = torch.zeros(num_envs, device=device)
    # H1: the lateral error of EVERY env at EVERY step, in order. Plain Python
    # lists so the settling reading is the same offline-testable arithmetic the
    # checks run (sp.track_settle). At 16 envs and 250 steps that is 4000
    # floats -- a measurement run, not a training loop. Filled only in
    # --free-space; an empty list is what says the mode was off.
    lat_err_trace: list[list[float]] = [[] for _ in range(num_envs)]
    # Force AGAINST depth, the RT-74 instrument. Plain Python lists, not
    # tensors: they are indexed by DEPTH and not by env, so they are shared
    # across all envs and across resets -- the profile is the run's, not an
    # episode's. Nothing steers on them.
    _n_bins = int(args_cli.depth_bins)
    _bin_width_m = float(args_cli.depth_bin_mm) / 1000.0
    depth_bin_counts = [0] * _n_bins
    depth_bin_force_max = [0.0] * _n_bins
    depth_bin_force_sum = [0.0] * _n_bins

    episodes: list[dict] = []
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
            # MEASUREMENT ONLY, added for RT-70. Nothing below steers on it.
            # It stays inside the information boundary: ee_quat is an
            # observation channel the policy also gets, not env target truth.
            ee_quat = _slice(obs, "ee_quat")

            # The yaw target is LATCHED, never assumed. See the module header:
            # the seated orientation of the real part is written down nowhere,
            # and at the default (all reset-noise ranges 0.0) this captures the
            # home pose.
            yaw_target, latched = sp.latch_yaw_target(yaw_cs, yaw_target, latched)
            yaw_err = sp.yaw_error(yaw_cs, yaw_target)
            lat_err = sp.lateral_error(tip_rel, aim)
            if args_cli.free_space:
                for _i, _e in enumerate(lat_err.tolist()):
                    lat_err_trace[_i].append(float(_e))
            depth = -tip_rel[:, 2]
            force_norm = insertion_math.force_magnitude(force)

            peak_depth = sp.update_peak_depth(depth, peak_depth)
            peak_force = torch.maximum(force_norm, peak_force)
            # Every step of every env, as plain numbers. num_envs is 4..16, so
            # this is a handful of floats per step and the run is a measurement
            # run, not a training loop.
            sp.bin_force_by_depth(
                depth.tolist(), force_norm.tolist(), _bin_width_m,
                depth_bin_counts, depth_bin_force_max, depth_bin_force_sum,
            )
            # The WORST tilt of the episode, not the tilt at the end: a part
            # that goes in crooked and is straightened by the pocket wall
            # still jammed on the way, and an end-of-episode reading would
            # show none of it.
            tilt = sp.axis_tilt_angle(ee_quat, pocket_quat)
            peak_tilt = torch.maximum(tilt, peak_tilt)

            # THE FIRST STEP IS THE REPORT THE USER ASKED FOR. With every reset
            # noise at 0.0 the part should already stand aligned over the
            # opening; these three numbers say whether it does. Recorded rather
            # than asserted -- a wrong home pose must show up as a finding, not
            # as a broken assumption further down.
            if first_step is None:
                first_step = {
                    "lateral_error_m": _percentiles([float(v) for v in lat_err.tolist()]),
                    "yaw_error_rad": _percentiles([abs(float(v)) for v in yaw_err.tolist()]),
                    "depth_m": _percentiles([float(v) for v in depth.tolist()]),
                    "force_norm_n": _percentiles([float(v) for v in force_norm.tolist()]),
                    # Radians in the file, degrees only in the printed line.
                    # One unit per number, and the file is the one that counts.
                    "axis_tilt_rad": _percentiles([float(v) for v in tilt.tolist()]),
                }
                print(f"[scripted_insert] step 1 residuals (all reset noise at its cfg value): "
                      f"lateral p50 {first_step['lateral_error_m']['p50']:.6f} m, "
                      f"|yaw| p50 {first_step['yaw_error_rad']['p50']:.6f} rad, "
                      f"depth p50 {first_step['depth_m']['p50']:.6f} m, "
                      f"axis tilt p50 {math.degrees(first_step['axis_tilt_rad']['p50']):.4f} deg")

            phase = sp.advance_phase(
                phase, lat_err, yaw_err, tip_rel[:, 2], force_norm,
                args_cli.lateral_tol, args_cli.yaw_tol,
                args_cli.standoff, args_cli.height_tol,
                float(unwrapped.cfg.force_abort_f_max_n),
                allow_descend=not args_cli.free_space,
            )

            cmd_pocket = sp.command_in_pocket_frame(
                tip_rel, phase, yaw_err, aim,
                args_cli.standoff, args_cli.lateral_gain, args_cli.vertical_gain, args_cli.yaw_gain,
                args_cli.descend_rate, args_cli.max_step, args_cli.max_yaw_step,
            )

            if control_mode == "osc":
                # ONE hop: pocket -> env, then the per-step limits. The env's
                # OSC tracks the delta every physics step; no IK here.
                actions = sp.osc_action_from_pocket_command(
                    cmd_pocket, pocket_quat,
                    float(unwrapped.cfg.osc_pos_step_limit_m), float(unwrapped.cfg.osc_rot_step_limit_rad),
                )
            else:
                # pocket -> env -> base. The root quaternion is READ, never assumed
                # to be identity.
                root_pose_w = robot.data.root_pose_w
                cmd_base = sp.command_to_base_frame(cmd_pocket, pocket_quat, root_pose_w[:, 3:7])

                # The IK hop, Isaac Lab's own controller and Isaac Lab's own frame
                # convention (run_diff_ik.py:162-171).
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
            obs = obs_dict["policy"]
            done = torch.logical_or(terminated, truncated)

        steps += 1

        if bool(done.any()):
            # Recorded from the LAST PRE-RESET observation this loop saw, i.e.
            # one control step (1/60 s) before the terminal one. DirectRLEnv
            # resets inside step() and hands back the post-reset observation, so
            # the terminal one is not available here. Stated rather than hidden:
            # it is the same reset-step trap the project rules name, and taking
            # the post-reset numbers would be strictly worse.
            for i in torch.nonzero(done).flatten().tolist():
                episodes.append({
                    "env": int(i),
                    # The env's fixed aim, carried on every episode rather than
                    # looked up later: the table below groups by it, and a
                    # grouping that re-derives the offset from the env index
                    # would silently agree with itself if either side moved.
                    "aim_offset_mm": _aim_mm[int(i)],
                    "peak_depth_m": float(peak_depth[i]),
                    "peak_force_n": float(peak_force[i]),
                    "peak_axis_tilt_rad": float(peak_tilt[i]),
                    "final_phase": sp.PHASE_NAMES[float(phase[i])],
                    "terminated": bool(terminated[i]),
                    "truncated": bool(truncated[i]),
                })
            keep = torch.logical_not(done)
            phase = torch.where(keep, phase, torch.full_like(phase, sp.PHASE_ALIGN))
            latched = latched * keep.reshape(-1, 1).to(latched.dtype)
            peak_depth = peak_depth * keep.to(peak_depth.dtype)
            peak_force = peak_force * keep.to(peak_force.dtype)
            peak_tilt = peak_tilt * keep.to(peak_tilt.dtype)

    # -------------------------------------------------------------- the numbers
    depths = [e["peak_depth_m"] for e in episodes]
    forces = [e["peak_force_n"] for e in episodes]
    tilts = [e["peak_axis_tilt_rad"] for e in episodes]
    phases: dict = {}
    for e in episodes:
        phases[e["final_phase"]] = phases.get(e["final_phase"], 0) + 1

    # PEAK DEPTH AGAINST THE LATERAL AIM -- the RT-75 measurement. The skeleton
    # (which env, which offset, which episodes) comes from the policy module so
    # it is checked offline; the statistics use _percentiles, which already
    # exists, rather than a second median written here.
    aim_rows = []
    for _r in sp.aim_sweep_rows(_aim_mm, [e["env"] for e in episodes]):
        _eps = [episodes[k] for k in _r["episodes"]]
        _ph: dict = {}
        for _e in _eps:
            _ph[_e["final_phase"]] = _ph.get(_e["final_phase"], 0) + 1
        aim_rows.append({
            "env": _r["env"],
            "aim_x_mm": _r["aim_x_mm"],
            "aim_y_mm": _r["aim_y_mm"],
            "episodes": len(_eps),
            "peak_depth_m": _percentiles([_e["peak_depth_m"] for _e in _eps]),
            "peak_force_n": _percentiles([_e["peak_force_n"] for _e in _eps]),
            "final_phase_counts": _ph,
        })

    # ------------------------------------------------------------ H1: settling
    # The comparison value is HALF the cross play: the part is held at an aim
    # inside the play, and a residual larger than half the play can put it
    # outside on the other side. It is DERIVED from PLAY_X (the measured
    # opening minus the measured part body) and never typed here -- the same
    # rule the aim span already follows.
    h1_tol_m = float(insertion_tasks_cfg.PLAY_X) / 2.0
    h1_rows = []
    if args_cli.free_space:
        for _i in range(num_envs):
            _s = sp.track_settle(lat_err_trace[_i], h1_tol_m, int(args_cli.settle_tail))
            h1_rows.append({
                "env": _i,
                "aim_x_mm": _aim_mm[_i][0],
                "aim_y_mm": _aim_mm[_i][1],
                "settle_step": _s["settle_step"],
                "steady_max_m": _s["steady_max_m"],
                "tail": _s["tail"],
            })
        h1_rows.sort(key=lambda r: (r["aim_x_mm"], r["aim_y_mm"], r["env"]))
    _h1_steady = [r["steady_max_m"] for r in h1_rows]
    _h1_pct = _percentiles(_h1_steady)
    # THREE OUTCOMES, not two. A reset inside a free-space run makes the trace
    # unreadable (the part teleports home and the error jumps), so it gets its
    # own verdict instead of being folded into a miss -- those are different
    # findings with different fixes.
    if not args_cli.free_space:
        h1_verdict = "NOT_RUN"
    elif episodes:
        h1_verdict = "H1_RESET_FIRED"
    elif _h1_pct["p95"] is not None and _h1_pct["p95"] < h1_tol_m:
        h1_verdict = "H1_TRACKS"
    else:
        h1_verdict = "H1_MISSES"

    metrics = {
        "run": ("scripted_insert (H1 free-space tracking)" if args_cli.free_space
                else "scripted_insert (descent, no longer a gate)"),
        "task": args_cli.task,
        "num_envs": num_envs,
        "steps": steps,
        "episodes": len(episodes),
        # WHAT THIS RUN PAYS FOR. Distributions, not thresholds: engaged_depth_m
        # is what the numbers below are supposed to produce, so asserting one
        # here would be circular.
        "peak_depth_m": _percentiles(depths),
        "peak_force_n": _percentiles(forces),
        # Radians. The angle between the part's tool axis and the pocket axis,
        # worst value per episode. It answers ONE question: did the part go in
        # crooked? The controller never commands either tilt axis, so this is
        # measured, never regulated.
        "peak_axis_tilt_rad": _percentiles(tilts),
        # FORCE AGAINST DEPTH, the one thing the peaks structurally cannot
        # say. Every run so far stopped at ~19.5 mm of the 36 mm seat at
        # ~204 N with the axis tilt exactly 0 -- straight, centred, and still
        # stopped -- and RT-73 watched it and saw nothing. What separates the
        # two candidate causes is WHEN the force arrives: smoothly from first
        # contact means friction over the whole travel, a jump at one depth
        # means something engages there and the depth names it.
        #
        # Every sample of every env of the whole run, not per episode: the
        # question is about depth, not about an episode.
        "force_by_depth": sp.force_depth_profile(
            _bin_width_m, depth_bin_counts, depth_bin_force_max, depth_bin_force_sum,
        ),
        # PEAK DEPTH AGAINST THE LATERAL AIM. RT-74 showed the force rising
        # smoothly from ~8 mm with no jump at any bin -- friction over the
        # travel, not an obstacle at a depth -- and every run up to it aimed at
        # the ONE point at the pocket axis. This asks the next question: does
        # the depth reached depend on WHERE inside the play the part is held?
        # A depth that varies with the offset means lateral binding and names
        # its direction; a depth flat at ~19.5 mm across the sweep takes the
        # lateral aim off the list of causes for that axis.
        "aim_sweep": {
            "axis": args_cli.aim_sweep,
            "span_m": _aim_span_m,
            "rows": aim_rows,
        },
        # H1 (D-108): can the joint-delta chain HOLD a commanded lateral offset
        # of the play's order? Free space, no contact, no insertion -- so the
        # number is the control chain's own precision and nothing else. Until
        # this reads, action_scale = 0.02 is UNCONFIRMED.
        #
        # `rows` is empty and `verdict` is NOT_RUN without --free-space. The
        # key is written either way, so a reader never has to tell "the mode
        # was off" from "the block is missing".
        "h1_free_space_tracking": {
            "mode": bool(args_cli.free_space),
            "verdict": h1_verdict,
            # PLAY_X / 2, derived. The provenance travels with the number.
            "tol_m": h1_tol_m,
            "tol_source": "insertion_tasks_cfg.PLAY_X / 2 (half the measured cross play)",
            "settle_tail_steps": int(args_cli.settle_tail),
            "steady_max_m": _h1_pct,
            "rows": h1_rows,
        },
        "final_phase_counts": phases,
        "first_step_residuals": first_step,
        # The settings that produced the numbers above. A measurement without
        # its instrument settings is not a measurement.
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
            "depth_bin_mm": args_cli.depth_bin_mm,
            "depth_bins": args_cli.depth_bins,
            # None when the run took the cfg default. The VALUE used is in
            # rl_placeholders below, read from the cfg; this field says only
            # whether a flag moved it, which the value alone cannot tell you.
            "f_abort_override_n": args_cli.f_abort,
        },
        "env": {
            "action_scale_rad": float(unwrapped.cfg.action_scale),
            "rl_terms_enabled": False,
            "reset_joint_noise": float(unwrapped.cfg.reset_joint_noise),
            "reset_yaw_noise": float(unwrapped.cfg.reset_yaw_noise),
            "fixture_pos_noise_xy": float(unwrapped.cfg.fixture_pos_noise_xy),
            "fixture_yaw_noise_rad": float(unwrapped.cfg.fixture_yaw_noise_rad),
            "fixture_tilt_noise_rad": float(unwrapped.cfg.fixture_tilt_noise_rad),
        },
        # Carried into the file for the same reason _write_metrics carries it:
        # no number may be reported without the invented values behind it.
        "rl_placeholders": {
            name: getattr(unwrapped.cfg, name) for name in ("force_abort_f_max_n",)
        },
        "per_episode": episodes,
    }
    out = pathlib.Path(args_cli.out)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[scripted_insert] {len(episodes)} episodes over {steps} steps")
    print(f"[scripted_insert] peak depth  p50 {metrics['peak_depth_m']['p50']} m  "
          f"max {metrics['peak_depth_m']['max']} m")
    print(f"[scripted_insert] peak force  p50 {metrics['peak_force_n']['p50']} N  "
          f"max {metrics['peak_force_n']['max']} N")
    _t = metrics["peak_axis_tilt_rad"]
    if _t["p50"] is None:
        print("[scripted_insert] peak axis tilt: no episode finished, nothing to report")
    else:
        # The DISCRIMINATING number for RT-70. The controller commands zero on
        # both tilt axes, so whatever stands here was never corrected. Printed
        # in degrees, stored in radians.
        print(f"[scripted_insert] peak axis tilt p50 {math.degrees(_t['p50']):.4f} deg  "
              f"max {math.degrees(_t['max']):.4f} deg  "
              "(NEVER commanded -- the controller holds both tilt axes at zero)")
    # THE PROFILE, PRINTED AS WELL AS WRITTEN. RT-70 taught this: a metrics
    # file whose path is printed but whose contents are not proves nothing in
    # a log, and the log is what /rt-check reads. One line per VISITED bin,
    # so the table is a handful of rows and not forty.
    _profile = metrics["force_by_depth"]
    if not _profile:
        print("[scripted_insert] force vs depth: no sample at a non-negative depth")
    else:
        print(f"[scripted_insert] force vs depth "
              f"({args_cli.depth_bin_mm:g} mm bins, {len(_profile)} visited):")
        for _row in _profile:
            print(f"[scripted_insert]   {_row['depth_lo_mm']:6.1f}..{_row['depth_hi_mm']:5.1f} mm  "
                  f"n {_row['samples']:6d}  "
                  f"force max {_row['force_max_n']:8.2f} N  mean {_row['force_mean_n']:8.2f} N")
    # THE AIM SWEEP TABLE, printed for the same reason the profile is: the log
    # is what /rt-check reads. Printed even at --aim-sweep none, where every
    # row carries the same 0.0000 offset -- a table that only appears when the
    # sweep is on would leave no baseline to compare the swept run against.
    print(f"[scripted_insert] peak depth vs lateral aim (sweep {args_cli.aim_sweep}, "
          f"span {_aim_span_m * 1000.0:.4f} mm, {len(aim_rows)} envs):")
    for _row in aim_rows:
        _d = _row["peak_depth_m"]["p50"]
        _fN = _row["peak_force_n"]["p50"]
        if _d is None:
            print(f"[scripted_insert]   aim ({_row['aim_x_mm']:+7.4f}, {_row['aim_y_mm']:+7.4f}) mm  "
                  f"env {_row['env']:2d}  n      0  -- no episode finished")
            continue
        print(f"[scripted_insert]   aim ({_row['aim_x_mm']:+7.4f}, {_row['aim_y_mm']:+7.4f}) mm  "
              f"env {_row['env']:2d}  n {_row['episodes']:6d}  "
              f"depth p50 {_d * 1000.0:7.3f} mm  force p50 {_fN:8.2f} N  "
              f"{_row['final_phase_counts']}")
    print(f"[scripted_insert] final phases {phases}")
    # THE H1 TABLE, printed for the same reason the profile and the aim sweep
    # are: /rt-check reads the log, and a metrics path proves nothing in it.
    if args_cli.free_space:
        print(f"[scripted_insert] H1 lateral tracking, free space, "
              f"tol {h1_tol_m * 1000.0:.4f} mm (= PLAY_X / 2), "
              f"tail {args_cli.settle_tail} steps:")
        for _row in h1_rows:
            _ss = _row["settle_step"]
            _ss_txt = "NEVER" if _ss < 0 else f"{_ss:5d}"
            print(f"[scripted_insert]   aim ({_row['aim_x_mm']:+7.4f}, {_row['aim_y_mm']:+7.4f}) mm  "
                  f"env {_row['env']:2d}  settle {_ss_txt}  "
                  f"steady max {_row['steady_max_m'] * 1000.0:8.4f} mm")
        if _h1_pct["p95"] is None:
            print("[scripted_insert] H1 steady error: no row, nothing to report")
        else:
            print(f"[scripted_insert] H1 steady error  p50 {_h1_pct['p50'] * 1000.0:.4f} mm  "
                  f"p95 {_h1_pct['p95'] * 1000.0:.4f} mm  "
                  f"max {_h1_pct['max'] * 1000.0:.4f} mm")
        print(f"[scripted_insert] H1 VERDICT: {h1_verdict}")
        if h1_verdict == "H1_RESET_FIRED":
            print(f"[scripted_insert]   {len(episodes)} episode(s) ended inside the run. "
                  "The part teleported home and the settling trace crosses that jump, "
                  "so no tracking number here can be read. Lower --max-steps.")
        elif h1_verdict == "H1_MISSES":
            print("[scripted_insert]   The p95 steady error is NOT below half the cross "
                  "play. That is D-108's H1 reading, and it is a FINDING, not a crash: "
                  "the recorded fallback is task-space deltas + impedance control. "
                  "Whether to take it is a decision nobody has taken.")
    print(f"[scripted_insert] metrics -> {out.resolve()}")
    print("[scripted_insert] NOTE: engaged_depth_m stays UNSET. This run reports a "
          "distribution; choosing the threshold is a decision, not a measurement.")

    env.close()

    # D-081 / the probe_wrench_bodies pattern: a verdict other than the target
    # one exits non-zero. A run that MEASURES a miss and reports success would
    # let the miss pass unread, and that is the one outcome the fail-fast rule
    # exists to prevent. Raised AFTER the file is written and the env is
    # closed, so the numbers survive the non-zero exit.
    if args_cli.free_space and h1_verdict != "H1_TRACKS":
        raise SystemExit(1)


if __name__ == "__main__":
    # D-081: set the exit status BEFORE the shutdown. close() does not return
    # (RT-22), so anything after it never decides anything, and RT-40 measured
    # this exact shape reporting `exit code: 0` on a traceback. The traceback is
    # printed explicitly so the failure stays visible in the rt_log dump.
    _code = 1
    try:
        main()
        _code = 0
    except SystemExit as _exc:
        _code = int(_exc.code or 0)
    except BaseException:
        traceback.print_exc()
        _code = 1
    exit_with(simulation_app, _code, tag="scripted_insert")
