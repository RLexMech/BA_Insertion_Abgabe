# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
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

# The deprecation helpers and the pretrained-checkpoint utility are absent from
# the Isaac Lab installed on the training machine (2.3.2), which made play.py
# unusable while train.py ran fine -- train.py imports neither and has its own
# call commented out. Importing them unconditionally turned a shipped-template
# artefact into a hard failure at the one moment a trained policy was to be
# replayed. Degrade instead: the helpers become identities, and the pretrained
# lookup fails only if it is actually requested.
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
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Generalisation probe: shift fixture_pos and opening_entrance_pos together
    # (they are the same point by construction -- asset origin on the opening
    # plane at the pocket centre), z untouched. The robot base and its home
    # joint configuration stay where they are, so the policy has to reach a
    # joint configuration it never trained on while observation, gate and
    # reward stay pocket-relative and therefore correct. play.py-only on
    # purpose: the env is evaluated exactly as trained, nothing in its
    # semantics changes.
    if args_cli.fixture_offset is not None:
        dx, dy = args_cli.fixture_offset
        # Limit raised 0.05 -> 0.15 (2026-08-06) after checking the two things
        # the old message named as unchecked. (1) Peg spawn: the reset is pure
        # joint-space (home pose + noise), the peg tip starts HOME_STANDOFF_Z
        # = 150 mm ABOVE the opening plane, which equals the plate top
        # everywhere -- an xy translation of the table cannot create a spawn
        # collision at any offset. (2) Plate geometry: the pocket is cut into
        # the moving asset, so it stays inside the plate by construction; what
        # actually bounds the offset is the pocket-to-robot-base distance,
        # which at +-0.15 m stays within [0.265, 0.585] m from the base
        # (OPENING_TO_BASE_DISTANCE 0.415 m at zero offset). The inner edge
        # (+0.15 m in y) is kinematically the tightest case -- a failure there
        # may be reach/configuration, not generalisation; read the replay's
        # video, not only its success rate. Beyond 0.15 m the pocket enters
        # the base's immediate surroundings and nothing has been checked.
        if max(abs(dx), abs(dy)) > 0.15:
            raise ValueError(
                f"--fixture-offset ({dx}, {dy}) exceeds 0.15 m per axis. Beyond that the "
                "pocket approaches the robot base's immediate surroundings (base sits "
                "0.415 m from the pocket at zero offset) and has not been checked."
            )
        fx, fy, fz = env_cfg.fixture_pos
        env_cfg.fixture_pos = (fx + dx, fy + dy, fz)
        env_cfg.opening_entrance_pos = env_cfg.fixture_pos
        # Since D-034 the fixture also has a per-episode pose randomisation.
        # A probe measures ONE deterministic pose, so the noise is forced off
        # -- the offset shifts the base pose, randomisation would smear it.
        # This keeps --fixture-offset the single way to place the fixture in
        # a replay rather than a second mechanism competing with the noise.
        if getattr(env_cfg, "fixture_pos_noise_xy", 0.0) or getattr(env_cfg, "fixture_yaw_noise_rad", 0.0):
            print("[INFO] --fixture-offset: fixture pose noise forced to 0 for this deterministic probe.")
        env_cfg.fixture_pos_noise_xy = 0.0
        env_cfg.fixture_yaw_noise_rad = 0.0
        print(
            f"[INFO] GENERALISATION PROBE: fixture translated by ({dx:+.3f}, {dy:+.3f}) m "
            f"to {env_cfg.fixture_pos}; robot base unmoved."
        )

    # Generalisation probe, orientation half (D-036): tilt the fixture about
    # the env y axis through the pocket centre. The env measures gate, depth,
    # alignment and the goal-relative observation in the pocket frame, so the
    # metrics stay correct under tilt (a world-frame gate would be off by
    # ~depth*sin(tilt), i.e. more than the 1 mm clearance from ~2 deg).
    # Capped at 15 deg: beyond that the plate's far edge sweeps > 65 mm
    # vertically, the alignment the pocket demands leaves the region the
    # reset pose can reach without re-checking the home standoff, and none
    # of it has been checked. Composes with --fixture-offset (the rotation
    # is about the shifted origin). Noise forced off for the same reason as
    # the offset probe: one deterministic pose per run.
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

    # Generalisation probe, yaw half (D-038): rotate the fixture about the env
    # z axis through the pocket centre. Since D-037 the pocket-frame transform
    # is built from the full per-env fixture quaternion, so phi, the corner
    # containment test and obs 12:15/21:25 all follow the yawed frame with no
    # further code. Capped at 45 deg: the square pocket repeats every 90 deg
    # (C4 symmetry), so +-45 deg is the largest range of physically distinct
    # poses -- beyond it the probe would silently measure a duplicate of a
    # smaller angle. Composes with --fixture-offset and --fixture-tilt
    # (spawn orientation is R_z(yaw) @ R_y(tilt)). Noise forced off for the
    # same reason as the other probes: one deterministic pose per run.
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

    # specify directory for logging experiments
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

    # set the log directory for the environment (works for all environment types)
    #
    # NOT log_dir itself: the env writes demo_metrics.json into whatever it is
    # given, and log_dir here is the TRAINING run's directory. A replay would
    # therefore overwrite the metrics of the run whose checkpoint it is
    # replaying -- with its own handful of episodes at --num_envs 16, silently
    # destroying the result on disk. Observed on the training machine
    # 2026-08-06, where a b = 36 training run's metrics were replaced by a
    # 751-episode replay. It is the same failure the env's _write_metrics
    # docstring describes for concurrent training runs, returning through the
    # replay path. Replay metrics are worth keeping -- they measure the final
    # policy without exploration noise -- so they go one level down rather than
    # being switched off.
    #
    # One directory PER PROBE POSE (offset and/or tilt): successive probe runs
    # on the same checkpoint would otherwise overwrite each other's
    # demo_metrics.json inside replay/, and the probe is exactly a series of
    # such runs.
    replay_dirname = "replay"
    if args_cli.fixture_offset is not None:
        dx, dy = args_cli.fixture_offset
        replay_dirname += f"_dx{dx * 1000:+05.0f}mm_dy{dy * 1000:+05.0f}mm"
    if args_cli.fixture_tilt is not None:
        replay_dirname += f"_tilt{args_cli.fixture_tilt:+05.1f}deg"
    if args_cli.fixture_yaw is not None:
        replay_dirname += f"_yaw{args_cli.fixture_yaw:+05.1f}deg"
    env_cfg.log_dir = os.path.join(log_dir, replay_dirname)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        # Same per-probe-pose separation as the metrics directory above:
        # RecordVideo names its files by step, so two probe runs on the same
        # checkpoint write the identical filename and the second silently
        # replaces the first. The probe is a series of such runs, and its
        # videos are the evidence the success rate alone cannot carry.
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", replay_dirname),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # convert pre-5.0 published checkpoints to the layout expected by rsl-rl >= 5.0 (no-op otherwise)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Replay progress on the console. The env counts finished episodes and
    # dumps metrics to disk, but during a replay nothing prints -- so "have
    # the ~750 episodes run yet?" was unanswerable without opening the JSON.
    # Guarded getattr: the counters are proxytask-specific and this script
    # must keep working for other tasks.
    replay_env = env.unwrapped
    last_reported_ep = 0

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
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
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
