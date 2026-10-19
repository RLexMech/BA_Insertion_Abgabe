# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""The scripted controller."""

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
        "Override env cfg control_mode for this run. "
        "Default None uses the cfg default (osc). 'joint_pd' uses the D-108 "
        "joint-space PD controller."
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

try:
    import json  # noqa: E402

    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402

    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
    from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

    import isaaclab_tasks  # noqa: F401, E402
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "source" / "insertion"))
    import insertion  # noqa: E402
    print(f"[scripted_insert] insertion package imported from: {pathlib.Path(insertion.__file__).resolve().parent}")
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
    """Read one observation block by NAME, from the one table that owns it."""
    lo, hi = insertion_math.OBS_SLICES[name]
    return obs[:, lo:hi]


def _percentiles(values: list[float]) -> dict:
    """p50 / p90 / p95 / max over a list, or an explicit empty marker."""
    if not values:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "max": None}
    s = sorted(values)

    def q(p: float) -> float:
        k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return s[k]

    return {"n": len(s), "p50": q(0.50), "p90": q(0.90), "p95": q(0.95), "max": s[-1]}


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.rl_terms_enabled = False
    # Keep diagnostic measurements independent of training defaults.
    env_cfg.obs_wrench_mode = "force"
    env_cfg.start_floor_m = 0.020
    env_cfg.autodr_stall_buffers = 0
    env_cfg.pocket_obs_yaw_rad = 0.0
    env_cfg.obs_noise_pocket_pos_std_m = 0.0
    env_cfg.force_obs_noise_std_n = 0.0
    env_cfg.grasp_obs_offset_x_m = 0.0
    env_cfg.start_tip_above_entrance = None

    if args_cli.f_abort is not None:
        env_cfg.force_abort_f_max_n = float(args_cli.f_abort)
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
    control_mode = str(unwrapped._control_mode)

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

    phase = torch.full((num_envs,), sp.PHASE_ALIGN, device=device)
    yaw_target = torch.zeros(num_envs, 2, device=device)
    latched = torch.zeros(num_envs, 1, device=device)
    peak_depth = torch.zeros(num_envs, device=device)
    peak_force = torch.zeros(num_envs, device=device)
    peak_tilt = torch.zeros(num_envs, device=device)
    lat_err_trace: list[list[float]] = [[] for _ in range(num_envs)]
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
            ee_quat = _slice(obs, "ee_quat")

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
            sp.bin_force_by_depth(
                depth.tolist(), force_norm.tolist(), _bin_width_m,
                depth_bin_counts, depth_bin_force_max, depth_bin_force_sum,
            )
            tilt = sp.axis_tilt_angle(ee_quat, pocket_quat)
            peak_tilt = torch.maximum(tilt, peak_tilt)

            if first_step is None:
                first_step = {
                    "lateral_error_m": _percentiles([float(v) for v in lat_err.tolist()]),
                    "yaw_error_rad": _percentiles([abs(float(v)) for v in yaw_err.tolist()]),
                    "depth_m": _percentiles([float(v) for v in depth.tolist()]),
                    "force_norm_n": _percentiles([float(v) for v in force_norm.tolist()]),
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
            obs = obs_dict["policy"]
            done = torch.logical_or(terminated, truncated)

        steps += 1

        if bool(done.any()):
            for i in torch.nonzero(done).flatten().tolist():
                episodes.append({
                    "env": int(i),
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

    depths = [e["peak_depth_m"] for e in episodes]
    forces = [e["peak_force_n"] for e in episodes]
    tilts = [e["peak_axis_tilt_rad"] for e in episodes]
    phases: dict = {}
    for e in episodes:
        phases[e["final_phase"]] = phases.get(e["final_phase"], 0) + 1

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
        "peak_depth_m": _percentiles(depths),
        "peak_force_n": _percentiles(forces),
        "peak_axis_tilt_rad": _percentiles(tilts),
        "force_by_depth": sp.force_depth_profile(
            _bin_width_m, depth_bin_counts, depth_bin_force_max, depth_bin_force_sum,
        ),
        "aim_sweep": {
            "axis": args_cli.aim_sweep,
            "span_m": _aim_span_m,
            "rows": aim_rows,
        },
        "h1_free_space_tracking": {
            "mode": bool(args_cli.free_space),
            "verdict": h1_verdict,
            "tol_m": h1_tol_m,
            "tol_source": "insertion_tasks_cfg.PLAY_X / 2 (half the measured cross play)",
            "settle_tail_steps": int(args_cli.settle_tail),
            "steady_max_m": _h1_pct,
            "rows": h1_rows,
        },
        "final_phase_counts": phases,
        "first_step_residuals": first_step,
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
        print(f"[scripted_insert] peak axis tilt p50 {math.degrees(_t['p50']):.4f} deg  "
              f"max {math.degrees(_t['max']):.4f} deg  "
              "(NEVER commanded -- the controller holds both tilt axes at zero)")
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

    if args_cli.free_space and h1_verdict != "H1_TRACKS":
        raise SystemExit(1)


if __name__ == "__main__":
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
