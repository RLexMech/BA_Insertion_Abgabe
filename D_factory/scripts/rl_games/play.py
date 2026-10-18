# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause

'Replay and evaluate RL-Games checkpoints; optionally record observation traces.'

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument("--stochastic", action="store_true", default=False, help="Sample the action (yaml player default).")
parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many control steps.")
parser.add_argument(
    "--dr-table",
    type=str,
    default=None,
    metavar="CSV",
    help="Watch the policy on a pre-drawn table (env.dr_mode=table, width target, env i = row i, "
    "num_envs = rows). A VIEW, not an evaluation: no manifest, no stop after the first episode.",
)
parser.add_argument(
    "--autodr-state",
    action="store_true",
    default=False,
    help="Load the checkpoint's AutoDR state, so the env draws every episode from the boundaries the "
    "training reached. Needs the training's env.dr_mode=autodr flags. A view, not an evaluation.",
)

parser.add_argument(
    "--memory-reset-every-step",
    action="store_true",
    default=False,
    help="D-195 test condition: clear the recurrent policy's memory for EVERY env "
    "after every step (default: only for envs whose episode ended). Shows whether "
    "an LSTM policy uses its memory, with no extra training. No effect on an MLP. "
    "An evaluation names it in eval_manifest.json (action_mode); give it its own --eval-tag.",
)
parser.add_argument(
    "--eval-table",
    type=str,
    default=None,
    metavar="CSV",
    help="Evaluation run (eval_plan.md step C): play this pre-drawn table, env i "
    "= row i, under env.dr_mode=table. Sets num_envs to the table's row count, "
    "writes into <run folder>/eval_<width>_<sha8>[_<tag>], stops once "
    "every env has finished its FIRST episode and writes eval_manifest.json. "
    "Needs --seed and --checkpoint.",
)
parser.add_argument(
    "--eval-width",
    type=str,
    default="target",
    choices=("target", "nominal"),
    help="Width the table is mapped onto (env.dr_table_width). Only with --eval-table.",
)
parser.add_argument(
    "--eval-tag",
    type=str,
    default="",
    help="Suffix of the evaluation folder, for a REPEAT of the same policy on "
    "the same table (G6). An evaluation folder that already holds episodes.csv "
    "is refused, never appended to.",
)
parser.add_argument(
    "--eval-ood",
    action="store_true",
    default=False,
    help="Out-of-distribution table (D-206 (3), D-055/D-191): lets the env "
    "accept table values outside [0, 1] (env.dr_table_allow_outside). Only with "
    "--eval-table; where a value may leave [0, 1] is checked afterwards against "
    "the table's sidecar (check_eval_conditions.py G3e).",
)

parser.add_argument(
    "--trace-obs",
    type=str,
    default=None,
    metavar="PATH",
    help="Replay trace: write the raw policy observation slices tip_rel, ee_quat, yaw_cos_sin "
    "and force, the action and done per step to JSON, plus the truth block read before the "
    "observation noise (force, true tip, tool pose, OSC target, interpenetration). A view, no "
    "env change, no reward change. Stops after --trace-steps steps.",
)
parser.add_argument(
    "--trace-steps",
    type=int,
    default=512,
    help="Steps to record for --trace-obs (default 512 = two 256-step episodes).",
)

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True


sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import math
import os
import random
import time
from importlib import metadata

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
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

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source", "insertion")))
import insertion
import insertion.tasks

print(f"[rl_games] insertion package imported from: {os.path.dirname(os.path.abspath(insertion.__file__))}")


def _eval_manifest_tool():
    '``scripts/tools/eval_manifest.py``, loaded by path as ``scripts/rsl_rl/play.py``'
    import importlib.util
    import pathlib

    _path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "eval_manifest.py"
    _spec = importlib.util.spec_from_file_location("eval_manifest", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


def _normalizer_sha256(agent) -> str | None:
    "G5 of eval_plan.md, rl_games form of ``scripts/rsl_rl/play.py``'s: one hash"
    import hashlib

    norm = getattr(getattr(agent, "model", None), "running_mean_std", None)
    state = norm.state_dict() if hasattr(norm, "state_dict") else {}
    if not state:
        return None
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _versions() -> dict:
    'Package versions for the manifest. A package pip does not know is'
    out = {"python": sys.version.split()[0], "torch": str(torch.__version__)}
    for key, dist in (("rl_games", "rl-games"), ("isaaclab", "isaaclab"), ("isaacsim", "isaacsim")):
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError as exc:
            out[key] = f"UNKNOWN: {exc}"
    return out


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    'Play with RL-Games agent.'

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")


    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device


    _em = None
    if args_cli.eval_table is not None:
        _em = _eval_manifest_tool()
        _eval_refusals = []
        if args_cli.seed is None:
            _eval_refusals.append("--seed is not set (the yaml's seed would run, silently)")
        if args_cli.seed == -1:
            _eval_refusals.append("--seed -1 draws a random seed")
        if not args_cli.checkpoint:
            _eval_refusals.append("--checkpoint is not set (the manifest hashes ONE named file)")
        for _flag in ("dr_table", "max_steps", "trace_obs"):
            if getattr(args_cli, _flag) is not None:
                _eval_refusals.append(f"--{_flag.replace('_', '-')} does not combine with --eval-table")
        for _flag in ("use_last_checkpoint", "use_pretrained_checkpoint", "autodr_state", "video", "stochastic"):
            if getattr(args_cli, _flag):
                _eval_refusals.append(f"--{_flag.replace('_', '-')} does not combine with --eval-table")
        _table_path = os.path.abspath(args_cli.eval_table)
        if not os.path.isfile(_table_path):
            _eval_refusals.append(f"no table at {_table_path}")
        if _eval_refusals:
            for _line in _eval_refusals:
                print(f"[rl_games-play] REFUSED: {_line}")
            raise SystemExit(2)
        _table_sha = _em.sha256_file(_table_path)
        _table_rows = _em.count_table_rows(_table_path)
        if args_cli.num_envs is not None and int(args_cli.num_envs) != _table_rows:
            print(f"[rl_games-play] REFUSED: --num_envs {args_cli.num_envs} != {_table_rows} table rows")
            raise SystemExit(2)
        env_cfg.scene.num_envs = _table_rows
        env_cfg.dr_mode = "table"
        env_cfg.dr_table_path = _table_path
        env_cfg.dr_table_width = args_cli.eval_width
        env_cfg.dr_table_allow_outside = bool(args_cli.eval_ood)
    elif args_cli.eval_tag or args_cli.eval_width != "target" or args_cli.eval_ood:
        print("[rl_games-play] REFUSED: --eval-width / --eval-tag / --eval-ood without --eval-table")
        raise SystemExit(2)


    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]


    env_cfg.seed = agent_cfg["params"]["seed"]


    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:

        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")

        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:

            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"

        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))


    _h = env_cfg.start_tip_above_entrance
    replay_dirname = "replay" + ("_hhome" if _h is None else f"_h{float(_h) * 1000:+04.0f}mm")
    if _em is not None:


        replay_dirname = _em.eval_dirname(args_cli.eval_width, _table_sha, args_cli.eval_tag)
        if os.path.exists(os.path.join(log_dir, replay_dirname, "episodes.csv")):
            print(f"[rl_games-play] REFUSED: {os.path.join(log_dir, replay_dirname)} already holds "
                  "episodes.csv. Pass --eval-tag for a repeat, or move the old folder away.")
            raise SystemExit(2)
    env_cfg.log_dir = os.path.join(log_dir, replay_dirname)
    print(f"[rl_games-play] replay record folder: {env_cfg.log_dir}")


    if args_cli.dr_table is not None:
        _table_path = os.path.abspath(args_cli.dr_table)
        with open(_table_path, encoding="utf-8") as _f:
            _rows = sum(1 for _line in _f if _line.strip()) - 1
        env_cfg.scene.num_envs = _rows
        env_cfg.dr_mode = "table"
        env_cfg.dr_table_path = _table_path
        env_cfg.dr_table_width = "target"
        print(f"[rl_games-play] view table: {_table_path}, {_rows} rows = envs, width target (NOT an evaluation)")


    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)


    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)


    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)


    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)


    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})


    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")


    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)

    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    agent.is_deterministic = not args_cli.stochastic
    print(f"[rl_games-play] deterministic action: {agent.is_deterministic}, rnn: {agent.is_rnn}")

    _norm_before = _normalizer_sha256(agent) if _em is not None else None


    if args_cli.autodr_state:
        _es = torch.load(resume_path, map_location="cpu", weights_only=False).get("env_state")
        _dr = getattr(env.unwrapped, "_dr", None)
        if args_cli.dr_table is not None or _dr is None or not _es or "autodr" not in _es:
            raise SystemExit(
                "--autodr-state needs env.dr_mode=autodr (the training's flags), no --dr-table, and a "
                "checkpoint that carries the AutoDR env_state"
            )
        _dr.load_state_dict(_es["autodr"])
        print(
            f"[rl_games-play] AutoDR state loaded (run {_es.get('run')!r}, seed {_es.get('seed')!r}): "
            f"bounds_version {getattr(_dr, 'bounds_version', '?')}"
        )

    dt = env.unwrapped.step_dt


    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    steps = 0
    episodes = 0

    _ = agent.get_batch_size(obs, 1)

    if agent.is_rnn:
        agent.init_rnn()

    _eval_first_done = (
        torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=rl_device)
        if _em is not None else None
    )
    _eval_t0 = time.time()


    trace = None
    if args_cli.trace_obs is not None:
        replay_env = env.unwrapped

        def _scatter(name):
            v = getattr(replay_env.cfg, name, None)
            return None if v is None else float(v)

        from insertion.tasks.direct.insertion import insertion_math as _im
        _obs_mode = str(getattr(replay_env.cfg, "obs_wrench_mode", "force"))
        _sl = _im.obs_slices(_obs_mode)
        trace = {"tip_rel_m": [], "ee_quat": [], "yaw_cos_sin": [], "force_n": [],
                 "action": [], "done": [],
                 "slices": {"tip_rel": list(_sl["tip_rel"]), "ee_quat": list(_sl["ee_quat"]),
                            "yaw_cos_sin": list(_sl["yaw_cos_sin"]), "force": list(_sl["force"])},
                 "obs_wrench_mode": _obs_mode, "obs_version": _im.obs_version(_obs_mode),
                 "num_envs": int(replay_env.num_envs), "step_dt_s": float(dt),
                 "obs_noise_pocket_pos_std_m": _scatter("obs_noise_pocket_pos_std_m"),
                 "force_obs_noise_std_n": _scatter("force_obs_noise_std_n"),
                 "grasp_obs_offset_x_m": _scatter("grasp_obs_offset_x_m"),
                 "fixture_pos_noise_xy_m": _scatter("fixture_pos_noise_xy"),
                 "torque_obs_noise_std_nm": _scatter("torque_obs_noise_std_nm"),
                 "row0_is_unnoised": True}
        if "torque" in _sl:
            trace["torque_nm"] = []
            trace["slices"]["torque"] = list(_sl["torque"])
        truth = None
        _peg = getattr(replay_env, "_peg_body_idx", None)
        if _peg is not None and hasattr(replay_env, "_force_tared_raw"):
            truth = {"force_tared_raw_n": [], "force_ema_n": [], "tip_true_m": [],
                     "body_pose_w": [], "osc_delta": [], "osc_target_pose_w": [],
                     "interpen_max_m": [],
                     "torque_tared_raw_nm": [], "torque_ema_nm": [], "wrench_valid": []}
            trace["truth"] = truth
            trace["truth_frames"] = {
                "force_tared_raw_n": "force link parent-body frame, tared, BEFORE the EMA and the obs noise",
                "force_ema_n": "same frame, the EMA buffer _force_smooth, BEFORE the obs noise",
                "torque_tared_raw_nm": "same frame, about the PARENT link origin (assumed, RT-192 measures it), weight moment tared, BEFORE the EMA",
                "torque_ema_nm": "same frame, the EMA buffer _torque_smooth, BEFORE the obs noise; = obs 28:31 in wrench mode",
                "wrench_valid": "False on the step _reset_idx touched the env (the wrench read there is the OLD episode's), True otherwise",
                "tip_true_m": "leading tool point from the pocket entrance, pocket frame, WITHOUT grasp offset or pocket noise",
                "body_pose_w": "tool_link = the OSC body, world, pos xyz + quat wxyz",
                "osc_delta": "held pose delta of the last action AFTER the step limits, BEFORE box/cone clamp (m, rad axis-angle)",
                "osc_target_pose_w": "OSC target AFTER cone and box clamp, world, from the LAST physics substep of the previous step; body_pose_w is read after that substep",
                "interpen_max_m": "SAPU max interpenetration, Warp mesh query; > 0 means touching, 0 does NOT prove free",
            }
        trace_steps = 0


    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():

            obs = agent.obs_to_torch(obs)

            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            if trace is not None:
                o = obs["obs"] if isinstance(obs, dict) else obs
                trace["tip_rel_m"].append(o[:, _sl["tip_rel"][0]:_sl["tip_rel"][1]].cpu().tolist())
                trace["ee_quat"].append(o[:, _sl["ee_quat"][0]:_sl["ee_quat"][1]].cpu().tolist())
                trace["yaw_cos_sin"].append(o[:, _sl["yaw_cos_sin"][0]:_sl["yaw_cos_sin"][1]].cpu().tolist())
                trace["force_n"].append(o[:, _sl["force"][0]:_sl["force"][1]].cpu().tolist())
                if "torque_nm" in trace:
                    trace["torque_nm"].append(o[:, _sl["torque"][0]:_sl["torque"][1]].cpu().tolist())
                trace["action"].append(actions.cpu().tolist())
                if truth is not None:
                    _r = replay_env
                    _n = _r.num_envs
                    _rot = _r._tilt_rot if _r._tilt_rot is not None else _r._pocket_rot_identity
                    _bp = _r.robot.data.body_pos_w[:, _peg]
                    _bq = _r.robot.data.body_quat_w[:, _peg]
                    _tip, _, _ = _im.part_tip_pose(
                        _bp, _bq, _r._tip_offset_local.unsqueeze(0).repeat(_n, 1),
                        _r.scene.env_origins, _r._entrance_pos, _rot,
                    )
                    truth["force_tared_raw_n"].append(_r._force_tared_raw.cpu().tolist())
                    truth["force_ema_n"].append(_r._force_smooth.cpu().tolist())
                    truth["tip_true_m"].append(_tip.cpu().tolist())
                    truth["body_pose_w"].append(torch.cat((_bp, _bq), dim=-1).cpu().tolist())
                    truth["osc_delta"].append(_r._osc_delta.cpu().tolist())
                    truth["osc_target_pose_w"].append(_r._osc_target_pose_w.cpu().tolist())
                    truth["interpen_max_m"].append(_r._last_interpen_max.cpu().tolist())
                    truth["torque_tared_raw_nm"].append(_r._torque_tared_raw.cpu().tolist())
                    truth["torque_ema_nm"].append(_r._torque_smooth.cpu().tolist())
                    truth["wrench_valid"].append(_r._wrench_valid.cpu().tolist())

            obs, _, dones, _ = env.step(actions)
            steps += 1
            if trace is not None:
                trace["done"].append(dones.cpu().tolist())
                trace_steps += 1
                if trace_steps >= args_cli.trace_steps:
                    break


            if _eval_first_done is not None:
                _eval_first_done |= dones.to(torch.bool)
                if bool(_eval_first_done.all()):
                    print(f"[rl_games-play] evaluation complete: all {env.unwrapped.num_envs} envs finished "
                          f"their first episode after {steps} steps")
                    break


            _mem_reset = torch.ones_like(dones) if args_cli.memory_reset_every_step else dones
            if len(dones) > 0:

                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, _mem_reset, :] = 0.0
            episodes += int(dones.sum())
        if args_cli.video:
            timestep += 1

            if timestep == args_cli.video_length:
                break

        if args_cli.max_steps is not None and steps >= args_cli.max_steps:
            break


        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)


    if trace is not None:
        import json
        trace_path = os.path.abspath(args_cli.trace_obs)
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as fh:
            json.dump(trace, fh)
        print(f"[trace] wrote {trace_steps} steps x {trace['num_envs']} envs to {trace_path}")


    _log = env.unwrapped.extras.get("log", {}) or {}
    _succ = {
        k: round(float(torch.as_tensor(v, dtype=torch.float32).mean()), 4)
        for k, v in _log.items()
        if "success" in k.lower()
    }
    print(f"[rl_games-play] done: {steps} steps x {env.unwrapped.num_envs} envs, {episodes} episodes ended, success keys {_succ}")

    if _em is not None:


        _cfg = env.unwrapped.cfg
        _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _manifest = _em.build_manifest(
            **_em.git_commits(_repo),
            checkpoint_path=os.path.abspath(resume_path),
            checkpoint_sha256=_em.sha256_file(resume_path),
            table_path=_table_path,
            table_sha256=_table_sha,
            table_rows=_table_rows,
            width=args_cli.eval_width,
            tag=args_cli.eval_tag,
            task=args_cli.task,
            seed=int(agent_cfg["params"]["seed"]),
            num_envs=int(env.unwrapped.num_envs),
            episode_cap_steps=int(env.unwrapped.max_episode_length),
            step_dt_s=float(dt),
            action_mode="rl_games player.get_action(is_deterministic=True): the actor mean mu, no sampling "
            "(players.py:45-70 @6b3534f2); "
            + ("recurrent memory cleared EVERY step (--memory-reset-every-step)"
               if args_cli.memory_reset_every_step else "recurrent memory reset on done"),
            control_mode=str(getattr(_cfg, "control_mode", "UNKNOWN")),
            obs_wrench_mode=str(getattr(_cfg, "obs_wrench_mode", "UNKNOWN")),
            noise={
                _k: getattr(_cfg, _k, None)
                for _k in ("obs_noise_pocket_pos_std_m", "force_obs_noise_std_n",
                           "torque_obs_noise_std_nm", "grasp_obs_offset_x_m",
                           "pocket_obs_yaw_rad", "fixture_pos_noise_xy",
                           "fixture_pos_noise_z", "reset_joint_noise")
            },
            versions=_versions(),
            normalizer_sha256_before=_norm_before,
            normalizer_sha256_after=_normalizer_sha256(agent),
            policy_steps_run=int(steps),
            envs_finished_first_episode=int(_eval_first_done.sum().item()),
            wall_time_s=float(time.time() - _eval_t0),
            env_cfg=_cfg.to_dict(),
        )
        _manifest_path = _em.write_manifest(env_cfg.log_dir, _manifest)
        print(f"[rl_games-play] eval manifest: {_manifest_path}")


        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


    env.close()


if __name__ == "__main__":

    main()

    simulation_app.close()
