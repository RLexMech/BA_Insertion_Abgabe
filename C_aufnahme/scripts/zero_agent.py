# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

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

parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help=(
        "Stop after this many env steps and shut down normally. Without it the "
        "loop runs until the app is closed, which means the run can only end by "
        "Ctrl+C -- and that kills the rt_log wrapper before it writes its exit "
        "line (RT-46, RT-50). Pass a number to make the exit code observable."
    ),
)
parser.add_argument(
    "--start-tip-above-entrance-mm",
    type=float,
    default=None,
    help=(
        "Override env_cfg.start_tip_above_entrance with this height in mm "
        "(positive above the stage-2 opening plane, negative inside the "
        "pocket; same sign convention as the cfg field). Default None keeps "
        "the cfg value. Unlike the teleport in check_seated_success.py "
        "--reward-curve, this goes through the env's OWN reset path "
        "(D-162 dls-IK) -- which is exactly the difference the RT-120 vs "
        "RT-121 force conflict is about."
    ),
)
parser.add_argument(
    "--home-pose",
    action="store_true",
    default=False,
    help=(
        "Set env_cfg.start_tip_above_entrance = None, i.e. start at the bare "
        "home pose (165 mm above the opening) instead of the cfg's rung-0 "
        "height. This is the pose RT-46 measured the droop at, so a droop "
        "reading under a new controller (inbox entry 'Audit 2026-09-03 (a)') "
        "is comparable only with it. --start-tip-above-entrance-mm cannot "
        "express None, hence a flag."
    ),
)
parser.add_argument(
    "--osc-kp-pos",
    type=float,
    default=None,
    help=(
        "Override env cfg osc_kp_pos (N/m) FOR THIS RUN; the two placeholder runs are "
        "100 and 500. Under zero actions the OSC re-anchors on the current pose ONLY "
        "INSIDE the tip box; there kp acts through the critical damping 2*sqrt(kp) "
        "alone -- the drift SPEED, not a restoring force. OUTSIDE the box "
        "clamp_tip_in_box pins the target to osc_pos_clamp_m in x/y and "
        "osc_pos_clamp_z_m in z around the entrance and "
        "kp IS a restoring force (VERDICTS NACHTRAG 2026-09-04, RT-143). "
        "Ignored under control_mode joint_pd."
    ),
)
parser.add_argument(
    "--print-force",
    action="store_true",
    default=False,
    help=(
        "Print one line per step: tip depth and the tared contact-force "
        "norm across envs (p50 / max / envs above 1 N), read from the "
        "observation's force block. Zero actions plus this line is the "
        "discriminating measurement between 'the reset itself lands in "
        "contact' and 'the untrained policy's exploration causes the "
        "contact' (RT-120 vs RT-121)."
    ),
)
parser.add_argument(
    "--start-lateral-offset-mm",
    type=float,
    default=None,
    metavar="MM",
    help=(
        "Override env cfg start_lateral_offset FOR THIS RUN, as the RADIUS of "
        "the start disk in mm (the cfg field is one radius in metres since "
        "D-178 (3)). Without it the run sits at the cfg default 0.0, i.e. on "
        "the pocket axis, which is NOT the corner the H_min ladder is "
        "supposed to probe (plan 4.2: 0.030 m)."
    ),
)
parser.add_argument(
    "--fixture-tilt",
    type=float,
    default=None,
    metavar="DEG",
    help=(
        "Override env cfg fixture_tilt_rad FOR THIS RUN: the DETERMINISTIC "
        "static pocket tilt in degrees, same field and same name play.py "
        "uses (play.py:48, :269). Not the per-episode noise range "
        "fixture_tilt_noise_rad -- the cfg rejects mixing the two."
    ),
)
parser.add_argument(
    "--fixture-yaw",
    type=float,
    default=None,
    metavar="DEG",
    help=(
        "Override env cfg fixture_yaw_rad FOR THIS RUN: the DETERMINISTIC "
        "static pocket yaw in degrees, same field and same name play.py uses "
        "(play.py:60). Composes with --fixture-tilt as R_z(yaw) @ R_y(tilt)."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import insertion.tasks  # noqa: F401
from insertion.tasks.direct.insertion import insertion_math  # noqa: E402


def main():
    """Zero actions agent with Isaac Lab environment."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.rl_terms_enabled = False
    env_cfg.obs_noise_pocket_pos_std_m = 0.0
    env_cfg.force_obs_noise_std_n = 0.0
    env_cfg.grasp_obs_offset_x_m = 0.0

    if args_cli.home_pose:
        env_cfg.start_tip_above_entrance = None
        print("[zero_agent] --home-pose: start_tip_above_entrance = None (bare home pose, RT-46 form)")
    if args_cli.osc_kp_pos is not None:
        env_cfg.osc_kp_pos = float(args_cli.osc_kp_pos)
        print(f"[zero_agent] osc_kp_pos overridden: {env_cfg.osc_kp_pos} (THIS RUN ONLY)")
    if args_cli.start_tip_above_entrance_mm is not None:
        env_cfg.start_tip_above_entrance = args_cli.start_tip_above_entrance_mm / 1000.0
        env_cfg.start_tip_above_entrance_low = env_cfg.start_tip_above_entrance
        print(f"[zero_agent] start_tip_above_entrance overridden: "
              f"{args_cli.start_tip_above_entrance_mm:+.1f} mm above the "
              f"stage-2 opening plane (low edge set equal: fixed start).")
    if args_cli.start_lateral_offset_mm is not None:
        env_cfg.start_lateral_offset = args_cli.start_lateral_offset_mm / 1000.0
        print(f"[zero_agent] start_lateral_offset overridden: "
              f"{args_cli.start_lateral_offset_mm:.3f} mm start-disk radius "
              f"(THIS RUN ONLY).")
    if args_cli.fixture_tilt is not None:
        if abs(args_cli.fixture_tilt) > 15.0:
            raise SystemExit(
                f"[zero_agent] --fixture-tilt {args_cli.fixture_tilt} exceeds 15 deg. "
                "Beyond that the plate edge sweep and the reachable alignment "
                "envelope have not been checked (play.py:266-268)."
            )
        env_cfg.fixture_tilt_rad = math.radians(args_cli.fixture_tilt)
        print(f"[zero_agent] fixture_tilt_rad overridden: "
              f"{args_cli.fixture_tilt:.4f} deg = {env_cfg.fixture_tilt_rad:.6f} rad "
              f"(static probe tilt, THIS RUN ONLY).")
    if args_cli.fixture_yaw is not None:
        env_cfg.fixture_yaw_rad = math.radians(args_cli.fixture_yaw)
        print(f"[zero_agent] fixture_yaw_rad overridden: "
              f"{args_cli.fixture_yaw:.4f} deg = {env_cfg.fixture_yaw_rad:.6f} rad "
              f"(static probe yaw, THIS RUN ONLY).")
    if args_cli.fixture_tilt or args_cli.fixture_yaw:
        print(f"[zero_agent] probe angle set: fixture_pos_noise_xy "
              f"{env_cfg.fixture_pos_noise_xy} -> 0.0, fixture_tilt_noise_rad "
              f"{env_cfg.fixture_tilt_noise_rad} -> 0.0, fixture_yaw_noise_rad "
              f"{env_cfg.fixture_yaw_noise_rad} -> 0.0 (deterministic probe, THIS RUN ONLY).")
        env_cfg.fixture_pos_noise_xy = 0.0
        env_cfg.fixture_tilt_noise_rad = 0.0
        env_cfg.fixture_yaw_noise_rad = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()
    steps = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs_dict, _rew, terminated, truncated, _info = env.step(actions)
        steps += 1
        if args_cli.print_force:
            with torch.inference_mode():
                obs = obs_dict["policy"]
                f_lo, f_hi = insertion_math.OBS_SLICES["force"]
                t_lo, t_hi = insertion_math.OBS_SLICES["tip_rel"]
                force_n = insertion_math.force_magnitude(obs[:, f_lo:f_hi])
                depth_mm = -obs[:, t_lo:t_hi][:, 2] * 1000.0
                n_contact = int((force_n > 1.0).sum())
                reset_any = bool(terminated.any()) or bool(truncated.any())
            print(f"[zero_agent] step {steps:4d} | depth p50 "
                  f"{float(depth_mm.median()):+9.3f} mm | force p50 "
                  f"{float(force_n.median()):8.3f} N | max "
                  f"{float(force_n.max()):8.3f} N | >1N "
                  f"{n_contact}/{int(force_n.numel())}"
                  + ("  RESET this step -- later lines measure the NEXT episode"
                     if reset_any else ""))
        if args_cli.max_steps is not None and steps >= args_cli.max_steps:
            print(f"[zero_agent] --max-steps {args_cli.max_steps} reached after {steps} steps; stopping.")
            break

    env.close()


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
    exit_with(simulation_app, _code, tag="zero_agent")
