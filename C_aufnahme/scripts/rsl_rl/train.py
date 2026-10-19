# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

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
    "--stop-when-dr-max",
    type=float,
    default=None,
    metavar="RATE",
    help="AutoDR stop rule (Phase 5 plan, section 4): stop once ALL AutoDR boundaries sit "
    "at their maximum AND the FRESH regular-episode success rate reaches RATE. The bar is "
    "owned by Laufplan_Phase5.md (stop bar); it is deliberately NOT defaulted or restated "
    "here -- the run must pass it, so a changed bar can never hide in this file. Checked between "
    "blocks of --stop-check-every iterations; --max_iterations stays the backstop. "
    "Requires env.dr_mode=autodr. Cannot be combined with --stop-at-success-rate: the two "
    "read different windows and reporting one as the other would misstate the stop. "
    "The value is a RATE in [0, 1]; 80 is refused rather than read as 80 percent.",
)
parser.add_argument(
    "--stop-check-every",
    type=int,
    default=50,
    help="Iterations per block between checks of --stop-at-success-rate OR "
    "--stop-when-dr-max, whichever is given (default 50). NOTE the block loop repeats "
    "one iteration INDEX per block boundary: rsl_rl 3.0.1 sets "
    "current_learning_iteration = it (not it+1) and the next learn() starts at that "
    "same index. The number of gradient updates is still blocks x this; the final "
    "checkpoint is named model_<updates - blocks + 1>.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

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

import json
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


def _checkpoint_width_refusal(path: str, env_unwrapped) -> str | None:
    """D-188: the one line that refuses a checkpoint of the other observation width, or ``None``."""
    mode = getattr(getattr(env_unwrapped, "cfg", None), "obs_wrench_mode", None)
    if mode is None:
        return None
    import importlib.util
    import pathlib

    from insertion.tasks.direct.insertion import insertion_math as _im

    _tools = pathlib.Path(__file__).resolve().parents[1] / "tools" / "checkpoint_width.py"
    _spec = importlib.util.spec_from_file_location("checkpoint_width", _tools)
    _cw = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cw)
    return _cw.refusal_for_checkpoint(
        path, str(mode), int(env_unwrapped.cfg.observation_space),
        {m: _im.obs_dim(m) for m in _im.OBS_MODES},
        loader=lambda p: torch.load(p, map_location="cpu", weights_only=False),
    )
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

import insertion.tasks  # noqa: F401
from insertion.tasks.direct.insertion.insertion_env_cfg import resolve_start_lateral_offset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


SIDECAR_FORMAT = "autodr-sidecar-1"


def _autodr_sidecar_path(model_path: str) -> str:
    """``.../model_1500.pt`` -> ``.../autodr_1500.json`` (plan section 5)."""
    head, name = os.path.split(model_path)
    stem = os.path.splitext(name)[0]
    if stem.startswith("model_"):
        return os.path.join(head, "autodr_" + stem[len("model_"):] + ".json")
    out = os.path.join(head, "autodr_" + stem + ".json")
    print(f"[autodr-state] checkpoint {name!r} is not 'model_<it>.pt'; sidecar named {out!r}")
    return out


def _save_with_autodr(save_fn, provider, seed, run_name: str):
    """Wrap ``runner.save`` so every checkpoint gets its AutoDR sidecar."""
    def _save(path, *args, **kwargs):
        out = save_fn(path, *args, **kwargs)
        side = _autodr_sidecar_path(path)
        with open(side, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "format": SIDECAR_FORMAT,
                    "seed": seed,
                    "run": run_name,
                    "autodr": provider.state_dict(),
                },
                fh,
                indent=2,
                sort_keys=True,
            )
        return out

    return _save


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )


    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
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
    if isinstance(getattr(agent_cfg, "seed", None), int):
        parts.append(f"seed{agent_cfg.seed}")
    if agent_cfg.run_name:
        parts.append(agent_cfg.run_name)
    suffix = "_".join(parts)
    rt_name = os.environ.get("RT_NAME", "").strip()
    if rt_name:
        log_dir += f"__{rt_name}"
    if suffix:
        log_dir += f"_{suffix}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env_cfg.log_dir = log_dir
    env_cfg.rollout_steps_per_iteration = agent_cfg.num_steps_per_env

    if not getattr(env_cfg, "rl_terms_enabled", True):
        raise SystemExit(
            "rl_terms_enabled is False: this env computes no real reward, no force "
            "abort and no curriculum. Enable rl_terms_enabled for training."
        )

    if args_cli.stop_when_dr_max is not None and args_cli.stop_at_success_rate is not None:
        raise SystemExit(
            "--stop-when-dr-max and --stop-at-success-rate are two different stop "
            "rules over two different episode windows. Pass one."
        )
    for _flag, _val in (("--stop-when-dr-max", args_cli.stop_when_dr_max),
                        ("--stop-at-success-rate", args_cli.stop_at_success_rate)):
        if _val is not None and not (0.0 <= float(_val) <= 1.0):
            raise SystemExit(
                f"{_flag} is {float(_val)!r}. It is a success RATE and must lie in "
                "[0, 1]; 0.80 is eighty percent, 80 is nothing the criterion can ever "
                "reach."
            )
    _floor_on = (
        hasattr(env_cfg, "start_floor_m")
        and getattr(env_cfg, "start_tip_above_entrance", None) is not None
        and float(env_cfg.start_floor_m) < float(env_cfg.start_tip_above_entrance)
    )
    if args_cli.stop_when_dr_max is not None and (
        str(getattr(env_cfg, "dr_mode", "off")) != "autodr" and not _floor_on
    ):
        raise SystemExit(
            f"--stop-when-dr-max needs a bounds provider: env.dr_mode=autodr or a start "
            f"floor (env.start_floor_m below start_tip_above_entrance); this run has "
            f"dr_mode={getattr(env_cfg, 'dr_mode', None)!r} and no floor. Without a "
            "provider there are no boundaries to sit at their maximum and no fresh "
            "window to read."
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, sort_alpha=False
        )

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

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    _dr = getattr(env.unwrapped, "_dr", None)
    if _dr is not None:
        runner.save = _save_with_autodr(
            runner.save, _dr, agent_cfg.seed, os.path.basename(log_dir)
        )
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        _refusal = _checkpoint_width_refusal(resume_path, env.unwrapped)
        if _refusal is not None:
            print(f"[train] REFUSED: {_refusal}")
            raise SystemExit(2)
        runner.load(resume_path)
        if _dr is not None and agent_cfg.algorithm.class_name == "Distillation":
            raise SystemExit(
                f"algorithm Distillation together with dr_mode={env_cfg.dr_mode!r}: the "
                "checkpoint being loaded is a TEACHER, not this run's own earlier self, "
                "so its AutoDR state is not this run's state. Plan section 5 calls this "
                "a warm start and gives it its own condition. Run distillation with "
                "dr_mode='off'."
            )
        if _dr is not None:
            _side = _autodr_sidecar_path(resume_path)
            if not os.path.isfile(_side):
                raise SystemExit(
                    f"--resume under dr_mode={env_cfg.dr_mode!r}, but the AutoDR state "
                    f"beside the checkpoint is missing: {_side} . Plan section 5: a "
                    "resume is the same run -- same seed, same fixture, same AutoDR "
                    "state. Starting the boundaries at width 0 under a trained policy "
                    "is a WARM START and is its own condition, not this one."
                )
            with open(_side, encoding="utf-8") as _fh:
                _doc = json.load(_fh)
            if _doc.get("format") != SIDECAR_FORMAT:
                raise SystemExit(
                    f"{_side} is not an {SIDECAR_FORMAT} file (format "
                    f"{_doc.get('format')!r}). Refusing rather than reading a shape "
                    "this script does not know."
                )
            if _doc.get("seed") != agent_cfg.seed:
                raise SystemExit(
                    f"--resume from a run with seed {_doc.get('seed')!r} while this run "
                    f"has seed {agent_cfg.seed!r} (state file {_side}, written by run "
                    f"{_doc.get('run')!r}). Plan section 5: a resume is the SAME run -- "
                    "same seed, same fixture, same AutoDR state. Initialising from "
                    "ANOTHER run's policy is a WARM START and is its own condition "
                    "with its own five seeds."
                )
            _dr.load_state_dict(_doc["autodr"])
            print(
                f"[autodr-state] loaded {_side} (run {_doc.get('run')!r}, seed "
                f"{_doc.get('seed')!r}): bounds_version {_dr.bounds_version}, "
                f"all_at_max {_dr.all_at_max()}, phase {getattr(_dr, 'phase', 'dr')}"
            )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    _dr_bar = args_cli.stop_when_dr_max

    def _criterion_success_rate():
        """D-037's rule, unchanged: the trailing window over EVERY episode."""
        recent = getattr(env.unwrapped, "_recent_successes", None)
        if recent is None:
            return "abandon", "[stop-criterion] env exposes no success window; running to max_iterations."
        rate = (sum(recent) / len(recent)) if recent else 0.0
        filled = len(recent) >= (recent.maxlen // 4)
        line = (
            f"[stop-criterion] iteration {trained}/{agent_cfg.max_iterations}: "
            f"recent success {rate:.4f} over {len(recent)} episodes "
            f"(target {target:.4f}, window {'full enough' if filled else 'too short to judge'})"
        )
        return ("fire" if (filled and rate >= target) else "wait"), line

    def _criterion_dr_max():
        """The AutoDR rule (plan section 4)."""
        base = env.unwrapped
        dr = getattr(base, "_dr", None)
        fresh = getattr(base, "_fresh_successes", None)
        if dr is None or fresh is None:
            return "abandon", "[stop-criterion] env exposes no AutoDR state; running to max_iterations."
        at_max = bool(dr.all_at_max())
        n_fresh = len(fresh)
        floor = int(getattr(base, "_fresh_min", 0))
        rate = (sum(fresh) / n_fresh) if n_fresh else 0.0
        enough = n_fresh >= floor
        shown = f"{rate:.4f}" if enough else "--"
        line = (
            f"[stop-criterion] iteration {trained}/{agent_cfg.max_iterations}: "
            f"all boundaries at max {at_max}, fresh regular episodes "
            f"{n_fresh}/{floor}, fresh success {shown} (bar {target:.4f}, "
            f"bounds_version {dr.bounds_version})"
        )
        return ("fire" if (at_max and enough and rate >= target) else "wait"), line

    if args_cli.stop_at_success_rate is None and _dr_bar is None:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    else:
        if _dr_bar is None:
            target = float(args_cli.stop_at_success_rate)
            criterion = _criterion_success_rate
            note = (
                "[stop-criterion] NOTE: this criterion is the MEAN over all tilt angles. "
                "Read success_by_tilt_bin in demo_metrics.json before calling the rung done."
            )
        else:
            target = float(_dr_bar)
            criterion = _criterion_dr_max
            note = (
                "[stop-criterion] NOTE: dr/train_success_regular_fresh is a TRAINING rate "
                "under PPO exploration noise, over the regular episodes of the current "
                "bounds_version. It is NOT the test rate -- that one comes from the "
                "evaluation table (plan section 3) and never from a training log."
            )
        block = max(1, int(args_cli.stop_check_every))
        trained = 0
        stopped_early = False
        while trained < agent_cfg.max_iterations:
            chunk = min(block, agent_cfg.max_iterations - trained)
            runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=(trained == 0))
            trained += chunk
            state, line = criterion()
            print(line)
            if state == "abandon":
                _rest = agent_cfg.max_iterations - trained
                if _rest > 0:
                    runner.learn(num_learning_iterations=_rest, init_at_random_ep_len=False)
                    trained += _rest
                break
            if state == "fire":
                stopped_early = True
                print(f"[stop-criterion] target reached at iteration {trained}; stopping.")
                break
        if stopped_early:
            final_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
            runner.save(final_path)
            print(f"[stop-criterion] saved {final_path}")
        print(note)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    try:
        writer = getattr(runner, "writer", None)
        if writer is not None and hasattr(writer, "flush"):
            writer.flush()
        import importlib.util
        import pathlib
        _spec = importlib.util.spec_from_file_location(
            "export_tb_scalars", pathlib.Path(__file__).resolve().parents[1] / "export_tb_scalars.py")
        _exp = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_exp)
        _tags, _rows = _exp.read_scalars(pathlib.Path(log_dir))
        _n = _exp.write_csv(pathlib.Path(log_dir) / "scalars.csv", _tags, _rows)
        print(f"[scalars] wrote {_n} rows x {len(_tags)} tags to {os.path.join(log_dir, 'scalars.csv')}")
    except Exception as exc:  # noqa: BLE001 -- a missing CSV must not cost the checkpoint
        print(f"[scalars] NOT written ({type(exc).__name__}: {exc}); run export_tb_scalars.py by hand")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
