# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import math
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source", "insertion")))
import insertion  # noqa: E402
import insertion.tasks  # noqa: F401,E402
from insertion.tasks.direct.insertion.insertion_env_cfg import resolve_start_lateral_offset  # noqa: E402

print(f"[rl_games] insertion package imported from: {os.path.dirname(os.path.abspath(insertion.__file__))}")

logger = logging.getLogger(__name__)


def _run_folder_name(env_cfg, seed) -> str:
    """``<mm-dd_HH-MM-SS>__<RT_NAME>_<tags>``, the name ``scripts/rsl_rl/train.py`` gives (its lines 288-462, code copied without the comments; the reasons live there)."""
    parts = []
    if hasattr(env_cfg, "dr_mode") and str(env_cfg.dr_mode) != "off":
        parts.append(str(env_cfg.dr_mode))
    noise_xy = getattr(env_cfg, "fixture_pos_noise_xy", None)
    if isinstance(noise_xy, (int, float)) and noise_xy > 0.0:
        parts.append(f"offset{noise_xy * 1000:g}mm")
    noise_tilt = getattr(env_cfg, "fixture_tilt_noise_rad", None)
    if isinstance(noise_tilt, (int, float)) and noise_tilt > 0.0:
        parts.append(f"tilt0-{round(math.degrees(noise_tilt), 1):g}deg")
    noise_yaw = getattr(env_cfg, "fixture_yaw_noise_rad", None)
    if isinstance(noise_yaw, (int, float)) and noise_yaw > 0.0:
        parts.append(f"yaw+-{round(math.degrees(noise_yaw), 1):g}deg")
    if hasattr(env_cfg, "curriculum_enabled") and not bool(env_cfg.curriculum_enabled):
        parts.append("currOFF")
    if hasattr(env_cfg, "start_tip_above_entrance"):
        _start_h = env_cfg.start_tip_above_entrance
        if _start_h is None:
            parts.append("startHOME")
        else:
            _start_low = getattr(env_cfg, "start_tip_above_entrance_low", None)
            if _start_low is None or _start_low >= _start_h:
                parts.append(f"start{_start_h * 1000:+g}mm")
            else:
                parts.append(f"start{_start_low * 1000:+g}..{_start_h * 1000:+g}mm")
    if hasattr(env_cfg, "start_floor_m") and getattr(env_cfg, "start_tip_above_entrance", None) is not None:
        _floor_v = float(env_cfg.start_floor_m)
        if _floor_v < float(env_cfg.start_tip_above_entrance):
            parts.append(f"floor{_floor_v * 1000:+g}mm")
    _lat = resolve_start_lateral_offset(getattr(env_cfg, "start_lateral_offset", 0.0))
    if _lat > 0.0:
        parts.append(f"lat{_lat * 1000:g}mm")
    if hasattr(env_cfg, "spawn_workcell_block") and bool(env_cfg.spawn_workcell_block):
        parts.append("blockON")
    if isinstance(seed, int):
        parts.append(f"seed{seed}")
    name = datetime.now().strftime("%m-%d_%H-%M-%S")
    rt_name = os.environ.get("RT_NAME", "").strip()
    if rt_name:
        name += f"__{rt_name}"
    if parts:
        name += "_" + "_".join(parts)
    return name


ENV_STATE_FORMAT = "autodr-envstate-1"


class RlGamesGpuEnvAutoDR(RlGamesGpuEnv):
    def __init__(self, config_name: str, num_actors: int, seed=None, run_name: str = "", **kwargs):
        super().__init__(config_name, num_actors, **kwargs)
        self._seed = seed
        self._run_name = run_name

    def get_env_state(self):
        dr = getattr(self.env.unwrapped, "_dr", None)
        if dr is None:
            return None
        print(f"[autodr-state] saved into the checkpoint: bounds_version {getattr(dr, 'bounds_version', '?')}")
        return {"format": ENV_STATE_FORMAT, "run": self._run_name, "seed": self._seed, "autodr": dr.state_dict()}

    def set_env_state(self, env_state):
        dr = getattr(self.env.unwrapped, "_dr", None)
        if dr is None:
            return
        if env_state is None:
            raise SystemExit(
                "--checkpoint under an AutoDR provider, but the checkpoint carries no AutoDR "
                "state. Starting the boundaries at width 0 under a trained policy is a WARM "
                "START, its own condition."
            )
        if env_state.get("format") != ENV_STATE_FORMAT:
            raise SystemExit(f"checkpoint env_state format {env_state.get('format')!r} is not {ENV_STATE_FORMAT!r}")
        if env_state.get("seed") != self._seed:
            raise SystemExit(
                f"--checkpoint from a run with seed {env_state.get('seed')!r} while this run has seed "
                f"{self._seed!r} (run {env_state.get('run')!r}): a resume is the SAME run."
            )
        dr.load_state_dict(env_state["autodr"])
        print(
            f"[autodr-state] loaded from the checkpoint (run {env_state.get('run')!r}, seed "
            f"{env_state.get('seed')!r}): bounds_version {getattr(dr, 'bounds_version', '?')}"
        )


_WRITE_STATS_ARGS = (
    "total_time", "epoch_num", "step_time", "play_time", "update_time", "a_losses", "c_losses",
    "entropies", "kls", "last_lr", "lr_mul", "frame", "scaled_time", "scaled_play_time", "curr_frames",
)


def _mean(values) -> float:
    return float(torch.stack([torch.as_tensor(v, dtype=torch.float32).reshape(()) for v in values]).mean())


class ConsoleObserver(IsaacAlgoObserver):
    def after_init(self, algo):
        super().after_init(algo)
        self._printed_epoch = -1
        self._stats = {}
        write_stats = algo.write_stats

        def _write_stats(*args, **kwargs):
            self._stats = dict(zip(_WRITE_STATS_ARGS, args), **kwargs)
            return write_stats(*args, **kwargs)

        algo.write_stats = _write_stats

    def after_print_stats(self, frame, epoch_num, total_time):
        if epoch_num != self._printed_epoch:
            self._printed_epoch = epoch_num
            try:
                self._print_block(epoch_num)
            except Exception as exc:  # noqa: BLE001 -- a missing block must not cost the run
                print(f"[rl_games-console] block NOT printed ({type(exc).__name__}: {exc})")
        super().after_print_stats(frame, epoch_num, total_time)

    def _print_block(self, epoch_num):
        pad = 35
        algo, st = self.algo, self._stats
        lines = ["#" * 80, f"Learning iteration {epoch_num}/{algo.max_epochs}".center(80), ""]

        def row(label, value):
            lines.append(f"{label + ':':>{pad}} {value:.4f}")

        if algo.game_rewards.current_size > 0:
            row("Mean reward", float(algo.game_rewards.get_mean()[0]))
            row("Mean episode length", float(algo.game_lengths.get_mean()))
        sigmas = getattr(getattr(algo, "experience_buffer", None), "tensor_dict", {}).get("sigmas")
        if sigmas is not None:
            row("Mean rollout sigma", float(sigmas.float().mean()))
        for label, key in (("rl_games a_loss", "a_losses"), ("rl_games c_loss", "c_losses"),
                           ("rl_games entropy", "entropies"), ("rl_games kl", "kls")):
            if st.get(key):
                row(label, _mean(st[key]))
        if "last_lr" in st:
            row("rl_games lr", float(st["last_lr"]) * float(st.get("lr_mul", 1.0)))
        if self.ep_infos:
            for key in self.ep_infos[0]:
                vals = []
                for ep_info in self.ep_infos:
                    v = ep_info[key]
                    v = v if isinstance(v, torch.Tensor) else torch.tensor([float(v)])
                    vals.append(v.reshape(-1).float().to(algo.device))
                value = float(torch.cat(vals).mean())
                row(key if "/" in key else f"Mean episode {key}", value)
        lines.append("-" * 80)
        if "scaled_time" in st:
            lines.append(f"{'Iteration time:':>{pad}} {float(st['scaled_time']):.2f}s")
        print("\n".join(lines), flush=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    env_cfg.seed = agent_cfg["params"]["seed"]

    env_cfg.shaping_gamma = float(agent_cfg["params"]["config"]["gamma"])
    print(f"[rl_games] env.shaping_gamma = {env_cfg.shaping_gamma} (the yaml's gamma)")

    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    if "pbt" in agent_cfg and agent_cfg["pbt"]["directory"] != ".":
        log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)
    else:
        log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = _run_folder_name(env_cfg, agent_cfg["params"]["seed"])
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"Exact experiment name requested from command line: {os.path.join(log_root_path, log_dir)}")

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env_cfg.log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.rollout_steps_per_iteration = int(agent_cfg["params"]["config"]["horizon_length"])

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    print(env)

    _seed, _run = agent_cfg["params"]["seed"], log_dir
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnvAutoDR(
            config_name, num_actors, seed=_seed, run_name=_run, **kwargs
        ),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    console_observer = ConsoleObserver()
    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        observers = MultiObserver([console_observer, PbtAlgoObserver(agent_cfg, args_cli)])
        runner = Runner(observers)
    else:
        runner = Runner(console_observer)

    runner.load(agent_cfg)

    runner.reset()

    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        if not wandb.run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    try:
        import importlib.util
        import pathlib

        writer = getattr(getattr(console_observer, "algo", None), "writer", None)
        if writer is not None and hasattr(writer, "flush"):
            writer.flush()
        _spec = importlib.util.spec_from_file_location(
            "export_tb_scalars", pathlib.Path(__file__).resolve().parents[1] / "export_tb_scalars.py")
        _exp = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_exp)
        _run_dir = pathlib.Path(log_root_path) / log_dir
        _tags, _rows = _exp.read_scalars(_run_dir / "summaries")
        _n = _exp.write_csv(_run_dir / "scalars.csv", _tags, _rows)
        print(f"[scalars] wrote {_n} rows x {len(_tags)} tags to {_run_dir / 'scalars.csv'}")
    except Exception as exc:  # noqa: BLE001 -- a missing CSV must not cost the checkpoint
        print(f"[scalars] NOT written ({type(exc).__name__}: {exc}); run export_tb_scalars.py by hand")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
