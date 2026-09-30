# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--stop-at-success-rate",
    type=float,
    default=None,
    help="Stop training once the trailing-window success rate reaches this value (e.g. 0.99) "
    "instead of always running --max_iterations. Checked between blocks of "
    "--stop-check-every iterations; --max_iterations stays the backstop. Note under a "
    "randomised pocket tilt (D-037) this is the MEAN over the whole angle range and can be "
    "reached while the largest angles lag -- read success_by_tilt_bin in demo_metrics.json "
    "before calling a rung done.",
)
parser.add_argument(
    "--stop-check-every",
    type=int,
    default=50,
    help="Iterations per block between checks of --stop-at-success-rate (default 50).",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import math
import os
import time
from datetime import datetime

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
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import proxytask.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    # agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    # Demo sprint: stamp the curriculum rung into the run name automatically,
    # so every run directory and every TensorBoard curve says which geometry it
    # was trained on. Relying on --run_name for this works until it is forgotten
    # once, and a run whose geometry is unknown cannot be compared to anything.
    # Guarded: this is the shared template script, and only this project's env
    # carries held_asset/fixed_asset.
    held = getattr(env_cfg, "held_asset", None)
    fixed = getattr(env_cfg, "fixed_asset", None)
    side = getattr(held, "side", None) if held is not None else None
    pocket = getattr(fixed, "side", None) if fixed is not None else None
    # Both sizes, because since D-033 the rung is the PAIR: the peg is fixed at
    # 30 mm and the pocket moves, so a tag naming only the peg would read the
    # same for every rung in the curriculum.
    #
    # Written out with units since 2026-08-16, replacing the compact "s30b32t15"
    # form. The tag is not only a folder name: it is the legend of every
    # TensorBoard curve and therefore of the figures in the thesis, where a
    # reader who does not know the project has to be able to tell the lines
    # apart without a key. Mixed units made that impossible in the compact form
    # -- s and b were millimetres, t was degrees, and nothing said so.
    #
    # Older runs keep their compact names; nothing renames history.
    parts = []
    if isinstance(side, (int, float)):
        parts.append(f"peg{side * 1000:g}mm")
    if isinstance(pocket, (int, float)):
        parts.append(f"pocket{pocket * 1000:g}mm")
    # D-034: the fixture-pose randomisation range is part of the run identity
    # for the same reason the pocket size is -- two runs at the same rung but
    # different ranges train different tasks, and a tag that cannot tell them
    # apart is how metrics get compared wrongly.
    noise_xy = getattr(env_cfg, "fixture_pos_noise_xy", None)
    if isinstance(noise_xy, (int, float)) and noise_xy > 0.0:
        parts.append(f"offset{noise_xy * 1000:g}mm")
    # D-037: the per-episode tilt range, same reason again. Written as the RANGE
    # it actually is (0 to the maximum, resampled per episode) rather than as
    # one number, which read like a fixed angle. Rounded to 0.1 deg before
    # formatting: the value is configured in radians, so 0.0873 rad came out as
    # "t5.00192" and 0.1745 rad as "t9.99807" -- three rungs of the same ladder
    # that looked like three unrelated experiments.
    noise_tilt = getattr(env_cfg, "fixture_tilt_noise_rad", None)
    if isinstance(noise_tilt, (int, float)) and noise_tilt > 0.0:
        parts.append(f"tilt0-{round(math.degrees(noise_tilt), 1):g}deg")
    # Slot for the yaw half; inert while the range is 0.
    noise_yaw = getattr(env_cfg, "fixture_yaw_noise_rad", None)
    if isinstance(noise_yaw, (int, float)) and noise_yaw > 0.0:
        parts.append(f"yaw+-{round(math.degrees(noise_yaw), 1):g}deg")
    if agent_cfg.run_name:
        parts.append(agent_cfg.run_name)
    suffix = "_".join(parts)
    if suffix:
        log_dir += f"_{suffix}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    if args_cli.stop_at_success_rate is None:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    else:
        # Early stop (D-037). rsl_rl's learn() runs to completion with no
        # callback, so the training is split into blocks and the criterion is
        # checked between them; learn() continues from current_learning_iteration,
        # so N blocks of k iterations train exactly as N*k in one call would.
        # init_at_random_ep_len only on the first block -- it staggers episode
        # lengths at the start of training, and re-applying it mid-run would
        # truncate episodes for no reason.
        #
        # The trailing window must be reasonably full before the criterion can
        # fire, or an early lucky block ends the run: same quarter-window guard
        # the env's episodes_to_threshold latch uses, for the same reason.
        target = float(args_cli.stop_at_success_rate)
        block = max(1, int(args_cli.stop_check_every))
        trained = 0
        stopped_early = False
        while trained < agent_cfg.max_iterations:
            chunk = min(block, agent_cfg.max_iterations - trained)
            runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=(trained == 0))
            trained += chunk
            recent = getattr(env.unwrapped, "_recent_successes", None)
            if recent is None:
                print("[stop-criterion] env exposes no success window; running to max_iterations.")
                break
            rate = (sum(recent) / len(recent)) if recent else 0.0
            filled = len(recent) >= (recent.maxlen // 4)
            print(
                f"[stop-criterion] iteration {trained}/{agent_cfg.max_iterations}: "
                f"recent success {rate:.4f} over {len(recent)} episodes "
                f"(target {target:.4f}, window {'full enough' if filled else 'too short to judge'})"
            )
            if filled and rate >= target:
                stopped_early = True
                print(f"[stop-criterion] target reached at iteration {trained}; stopping.")
                break
        if stopped_early:
            # learn() saves on its own schedule, so the block that triggered the
            # stop may not have written one. Save explicitly: the whole point of
            # the run is the policy it ended with.
            final_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
            runner.save(final_path)
            print(f"[stop-criterion] saved {final_path}")
        print(
            "[stop-criterion] NOTE: this criterion is the MEAN over all tilt angles. "
            "Read success_by_tilt_bin in demo_metrics.json before calling the rung done."
        )

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
