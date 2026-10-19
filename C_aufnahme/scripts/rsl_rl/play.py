# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
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
    "axis through the pocket centre; positive tips the +x plate side down. "
    "Since the 2026-08-19 frame rotation env y is the LATERAL axis, so this pitches "
    "the back-wall side of the fixture down. "
    "Generalisation probe: the pocket opening is no longer parallel to the ground. "
    "Gate, depth and observation are measured in the pocket frame. Max 15 deg.",
)
parser.add_argument(
    "--fixture-yaw",
    type=float,
    default=None,
    metavar="DEG",
    help="Rotate the fixture (and with it the pocket) by DEG degrees about the env z "
    "axis through the pocket centre. Generalisation probe: the square "
    "pocket's walls are no longer axis-aligned; phi, gate and observation are "
    "measured in the yawed pocket frame. The pocket repeats every 90 deg (C4), so "
    "+-45 deg already covers every distinct pose -- larger values are rejected. "
    "Composes with --fixture-offset and --fixture-tilt.",
)
parser.add_argument(
    "--trace-obs",
    type=str,
    default=None,
    metavar="PATH",
    help="RT-149 replay trace: write the raw policy "
    "observation slices tip_rel (12:15), ee_quat (15:19, added 2026-09-06 so "
    "the trace carries TILT and not only yaw), yaw_cos_sin (19:21) and force (25:28), "
    "the sampled action and the done flag PER STEP for every env to this JSON. "
    "Reads only channels the policy already sees (insertion_math.OBS_SLICES); "
    "no env change, no reward change. Stops after --trace-steps steps.",
)
parser.add_argument(
    "--trace-steps",
    type=int,
    default=512,
    help="Steps to record for --trace-obs (default 512 = two 256-step episodes).",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    metavar="N",
    help="Stop after N policy steps and exit. Without it (and without "
    "--trace-obs or --video) play.py runs until it is killed, so a probe "
    "that only wants demo_metrics.json had to write a trace it never read.",
)
parser.add_argument(
    "--eval-table",
    type=str,
    default=None,
    metavar="CSV",
    help="Evaluation run (eval_plan.md step C): play this pre-drawn table, env i "
    "= row i, under env.dr_mode=table. Sets num_envs to the table's row count, "
    "writes into <checkpoint folder>/eval_<width>_<sha8>[_<tag>], stops once "
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
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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

import insertion.tasks  # noqa: F401


def _checkpoint_width_refusal(path: str, env_unwrapped) -> str | None:
    """the one line that refuses a checkpoint of the other observation width, or ``None``."""
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


def _eval_manifest_tool():
    """``scripts/tools/eval_manifest.py``, loaded by path like the width guard: the folder name, the hashes and the manifest's field list live there (pure, self-tested)."""
    import importlib.util
    import pathlib

    _path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "eval_manifest.py"
    _spec = importlib.util.spec_from_file_location("eval_manifest", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


def _normalizer_sha256(runner) -> str | None:
    """Hash all observation-normalizer state buffers in sorted key order."""
    import hashlib

    norm = getattr(getattr(runner.alg, "policy", None), "actor_obs_normalizer", None)
    state = norm.state_dict() if hasattr(norm, "state_dict") else {}
    if not state:
        return None
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _versions() -> dict:
    """Package versions for the manifest."""
    out = {"python": sys.version.split()[0], "torch": str(torch.__version__)}
    for key, dist in (("rsl_rl", "rsl-rl-lib"), ("isaaclab", "isaaclab"), ("isaacsim", "isaacsim")):
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError as exc:
            out[key] = f"UNKNOWN: {exc}"
    return out


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    _em = None
    if args_cli.eval_table is not None:
        _em = _eval_manifest_tool()
        _eval_refusals = []
        if args_cli.seed is None:
            _eval_refusals.append("--seed is not set (the agent cfg would run its default, silently)")
        if not args_cli.checkpoint:
            _eval_refusals.append("--checkpoint is not set (the manifest hashes ONE named file)")
        for _flag in ("fixture_offset", "fixture_tilt", "fixture_yaw", "max_steps", "trace_obs"):
            if getattr(args_cli, _flag) is not None:
                _eval_refusals.append(f"--{_flag.replace('_', '-')} does not combine with --eval-table")
        if args_cli.video:
            _eval_refusals.append("--video does not combine with --eval-table (it ends the loop itself)")
        _table_path = os.path.abspath(args_cli.eval_table)
        if not os.path.isfile(_table_path):
            _eval_refusals.append(f"no table at {_table_path}")
        if _eval_refusals:
            for _line in _eval_refusals:
                print(f"[play] REFUSED: {_line}")
            raise SystemExit(2)
        _table_sha = _em.sha256_file(_table_path)
        _table_rows = _em.count_table_rows(_table_path)
        if args_cli.num_envs is not None and int(args_cli.num_envs) != _table_rows:
            print(f"[play] REFUSED: --num_envs {args_cli.num_envs} != {_table_rows} table rows")
            raise SystemExit(2)
        env_cfg.scene.num_envs = _table_rows
        env_cfg.dr_mode = "table"
        env_cfg.dr_table_path = _table_path
        env_cfg.dr_table_width = args_cli.eval_width
    elif args_cli.eval_tag or args_cli.eval_width != "target":
        print("[play] REFUSED: --eval-width / --eval-tag without --eval-table")
        raise SystemExit(2)

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
            f"(lateral axis) through the pocket centre; gate/depth/obs in the pocket frame."
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
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, sort_alpha=False
        )

    log_dir = os.path.dirname(resume_path)

    replay_dirname = "replay"
    _h = env_cfg.start_tip_above_entrance
    replay_dirname += "_hhome" if _h is None else f"_h{float(_h) * 1000:+04.0f}mm"
    if args_cli.fixture_offset is not None:
        dx, dy = args_cli.fixture_offset
        replay_dirname += f"_dx{dx * 1000:+05.0f}mm_dy{dy * 1000:+05.0f}mm"
    if args_cli.fixture_tilt is not None:
        replay_dirname += f"_tilt{args_cli.fixture_tilt:+05.1f}deg"
    if args_cli.fixture_yaw is not None:
        replay_dirname += f"_yaw{args_cli.fixture_yaw:+05.1f}deg"
    if _em is not None:
        replay_dirname = _em.eval_dirname(args_cli.eval_width, _table_sha, args_cli.eval_tag)
        if os.path.exists(os.path.join(log_dir, replay_dirname, "episodes.csv")):
            print(f"[play] REFUSED: {os.path.join(log_dir, replay_dirname)} already holds "
                  "episodes.csv. Pass --eval-tag for a repeat, or move the old folder away.")
            raise SystemExit(2)
    env_cfg.log_dir = os.path.join(log_dir, replay_dirname)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_root = os.path.abspath(
            os.path.join("videos", os.path.basename(log_dir), replay_dirname)
        )
        video_kwargs = {
            "video_folder": video_root,
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
    _refusal = _checkpoint_width_refusal(resume_path, env.unwrapped)
    if _refusal is not None:
        print(f"[play] REFUSED: {_refusal}")
        raise SystemExit(2)
    runner.load(resume_path)
    _norm_before = _normalizer_sha256(runner) if _em is not None else None

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

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
    steps_done = 0
    trace = None
    if args_cli.trace_obs is not None:
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
                 "num_envs": int(env.num_envs), "step_dt_s": float(dt),
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
    _eval_first_done = (
        torch.zeros(env.num_envs, dtype=torch.bool, device=replay_env.device)
        if _em is not None else None
    )
    _eval_t0 = time.time()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            if trace is not None:
                o = obs["policy"] if not torch.is_tensor(obs) else obs
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
            if trace is not None:
                trace["done"].append(dones.cpu().tolist())
                trace_steps += 1
                if trace_steps >= args_cli.trace_steps:
                    break
            steps_done += 1
            if args_cli.max_steps is not None and steps_done >= args_cli.max_steps:
                print(f"[play] stopping after {steps_done} steps (--max-steps)")
                break
            if _eval_first_done is not None:
                _eval_first_done |= dones.to(torch.bool)
                if bool(_eval_first_done.all()):
                    print(f"[play] evaluation complete: all {env.num_envs} envs finished "
                          f"their first episode after {steps_done} steps")
                    break
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

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
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

    if _em is not None:
        _cfg = replay_env.cfg
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
            seed=int(agent_cfg.seed),
            num_envs=int(env.num_envs),
            episode_cap_steps=int(replay_env.max_episode_length),
            step_dt_s=float(dt),
            action_mode="runner.get_inference_policy (rsl_rl): eval mode, act_inference = actor output, no sampling",
            control_mode=str(getattr(_cfg, "control_mode", "UNKNOWN")),
            obs_wrench_mode=str(getattr(_cfg, "obs_wrench_mode", "UNKNOWN")),
            noise={
                _k: getattr(_cfg, _k, None)
                for _k in ("obs_noise_pocket_pos_std_m", "force_obs_noise_std_n",
                           "torque_obs_noise_std_nm", "grasp_obs_offset_x_m",
                           "fixture_pos_noise_xy", "reset_joint_noise")
            },
            versions=_versions(),
            normalizer_sha256_before=_norm_before,
            normalizer_sha256_after=_normalizer_sha256(runner),
            policy_steps_run=int(steps_done),
            envs_finished_first_episode=int(_eval_first_done.sum().item()),
            wall_time_s=float(time.time() - _eval_t0),
            env_cfg=_cfg.to_dict(),
        )
        _manifest_path = _em.write_manifest(env_cfg.log_dir, _manifest)
        print(f"[play] eval manifest: {_manifest_path}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
