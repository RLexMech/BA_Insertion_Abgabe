# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay an RSL-RL checkpoint."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Replay an RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--fixture-offset",
    type=float,
    nargs=2,
    default=None,
    metavar=("DX", "DY"),
    help="Translate the fixture (and with it the pocket opening) by (DX, DY) metres "
    "in the env xy plane, robot base unmoved. Generalisation probe: evaluates a "
    "trained policy at a pocket position it never trained on. Max 0.15 m per axis.",
)
parser.add_argument(
    "--fixture-tilt",
    type=float,
    default=None,
    metavar="DEG",
    help="Tilt the fixture (and with it the pocket) by DEG degrees about the env y "
    "axis through the pocket centre; positive tips the +x plate side down (D-036). "
    "Generalisation probe: the pocket opening is no longer parallel to the ground. "
    "Gate, depth and observation are measured in the pocket frame. Max 15 deg.",
)
parser.add_argument(
    "--fixture-yaw",
    type=float,
    default=None,
    metavar="DEG",
    help="Rotate the fixture (and with it the pocket) by DEG degrees about the env z "
    "axis through the pocket centre (D-038). Generalisation probe: the square "
    "pocket's walls are no longer axis-aligned; phi, gate and observation are "
    "measured in the yawed pocket frame. The pocket repeats every 90 deg (C4), so "
    "+-45 deg already covers every distinct pose -- larger values are rejected. "
    "Composes with --fixture-offset and --fixture-tilt.",
)
parser.add_argument("--export-policy", action="store_true", help="Also export JIT and ONNX files (optional).")
parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many control steps; 0 runs until the window closes.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.max_steps < 0:
    parser.error("--max-steps must be non-negative")
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import math
import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(agent_cfg, _installed_version):
        return agent_cfg

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_checkpoint
except ImportError:
    def handle_deprecated_rsl_rl_checkpoint(resume_path, _installed_version):
        return resume_path

try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*_args, **_kwargs):
        raise RuntimeError(
            "--use_pretrained_checkpoint is unavailable: this Isaac Lab installation ships no "
            "isaaclab_rl.utils.pretrained_checkpoint. Pass --checkpoint with a local path instead."
        )

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import proxytask.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.fixture_offset is not None:
        dx, dy = args_cli.fixture_offset
        if max(abs(dx), abs(dy)) > 0.15:
            raise ValueError(
                f"--fixture-offset ({dx}, {dy}) exceeds 0.15 m per axis. Beyond that the "
                "pocket approaches the robot base's immediate surroundings (base sits "
                "0.415 m from the pocket at zero offset) and has not been checked."
            )
        fx, fy, fz = env_cfg.fixture_pos
        env_cfg.fixture_pos = (fx + dx, fy + dy, fz)
        env_cfg.opening_entrance_pos = env_cfg.fixture_pos
        if getattr(env_cfg, "fixture_pos_noise_xy", 0.0) or getattr(env_cfg, "fixture_yaw_noise_rad", 0.0):
            print("[INFO] --fixture-offset: fixture pose noise forced to 0 for this deterministic probe.")
        env_cfg.fixture_pos_noise_xy = 0.0
        env_cfg.fixture_yaw_noise_rad = 0.0
        print(
            f"[INFO] GENERALISATION PROBE: fixture translated by ({dx:+.3f}, {dy:+.3f}) m "
            f"to {env_cfg.fixture_pos}; robot base unmoved."
        )

    if args_cli.fixture_tilt is not None:
        tilt_deg = args_cli.fixture_tilt
        if abs(tilt_deg) > 15.0:
            raise ValueError(
                f"--fixture-tilt {tilt_deg} exceeds 15 deg. Beyond that the plate edge "
                "sweep and the reachable alignment envelope have not been checked."
            )
        env_cfg.fixture_tilt_rad = math.radians(tilt_deg)
        if getattr(env_cfg, "fixture_pos_noise_xy", 0.0) or getattr(env_cfg, "fixture_yaw_noise_rad", 0.0):
            print("[INFO] --fixture-tilt: fixture pose noise forced to 0 for this deterministic probe.")
        env_cfg.fixture_pos_noise_xy = 0.0
        env_cfg.fixture_yaw_noise_rad = 0.0
        print(
            f"[INFO] GENERALISATION PROBE: fixture tilted by {tilt_deg:+.2f} deg about env y "
            f"through the pocket centre; gate/depth/obs in the pocket frame."
        )

    if args_cli.fixture_yaw is not None:
        yaw_deg = args_cli.fixture_yaw
        if abs(yaw_deg) > 45.0:
            raise ValueError(
                f"--fixture-yaw {yaw_deg} exceeds 45 deg. The square pocket repeats every "
                "90 deg (C4), so +-45 deg already covers every distinct pose; use the "
                "equivalent angle inside +-45."
            )
        env_cfg.fixture_yaw_rad = math.radians(yaw_deg)
        if (
            getattr(env_cfg, "fixture_pos_noise_xy", 0.0)
            or getattr(env_cfg, "fixture_yaw_noise_rad", 0.0)
            or getattr(env_cfg, "fixture_tilt_noise_rad", 0.0)
        ):
            print("[INFO] --fixture-yaw: fixture pose noise forced to 0 for this deterministic probe.")
        env_cfg.fixture_pos_noise_xy = 0.0
        env_cfg.fixture_yaw_noise_rad = 0.0
        env_cfg.fixture_tilt_noise_rad = 0.0
        print(
            f"[INFO] GENERALISATION PROBE: fixture yawed by {yaw_deg:+.2f} deg about env z "
            f"through the pocket centre; phi/gate/obs in the yawed pocket frame."
        )

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    replay_dirname = "replay"
    if args_cli.fixture_offset is not None:
        dx, dy = args_cli.fixture_offset
        replay_dirname += f"_dx{dx * 1000:+05.0f}mm_dy{dy * 1000:+05.0f}mm"
    if args_cli.fixture_tilt is not None:
        replay_dirname += f"_tilt{args_cli.fixture_tilt:+05.1f}deg"
    if args_cli.fixture_yaw is not None:
        replay_dirname += f"_yaw{args_cli.fixture_yaw:+05.1f}deg"
    env_cfg.log_dir = os.path.join(log_dir, replay_dirname)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", replay_dirname),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    policy_nn = None
    if version.parse(installed_version) < version.parse("4.0.0"):
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

    if args_cli.export_policy:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        else:
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    replay_env = env.unwrapped
    last_reported_ep = 0

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        ep_count = getattr(replay_env, "_ep_count", 0)
        if ep_count - last_reported_ep >= 100:
            last_reported_ep = ep_count
            recent = getattr(replay_env, "_recent_successes", None)
            rate = (sum(recent) / len(recent)) if recent else float("nan")
            depths = getattr(replay_env, "_recent_depths", None)
            depth_mm = (sum(depths) / len(depths) * 1000.0) if depths else float("nan")
            print(f"[replay] episodes {ep_count}, recent success rate {rate:.4f}, "
                  f"mean max depth {depth_mm:.1f} mm")

        timestep += 1
        if args_cli.max_steps and timestep >= args_cli.max_steps:
            break
        if args_cli.video and timestep >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
