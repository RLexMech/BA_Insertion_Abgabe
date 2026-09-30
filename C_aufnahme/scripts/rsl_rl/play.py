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
    "axis through the pocket centre (D-038). Generalisation probe: the square "
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
    help="RT-149 replay trace (Plan-Merge 2026-09-05): write the raw policy "
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

import insertion.tasks  # noqa: F401


def _checkpoint_width_refusal(path: str, env_unwrapped) -> str | None:
    """D-188: the one line that refuses a checkpoint of the other observation
    width, or ``None``. The rule lives in ``scripts/tools/checkpoint_width.py``
    (pure, self-tested); this only binds torch.load and the env's mode. An env
    without ``obs_wrench_mode`` (another task) is never refused here."""
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
            f"(lateral axis) through the pocket centre; gate/depth/obs in the pocket frame."
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
        # sort_alpha=False -- pick the NEWEST run by mtime, not the
        # alphabetically last one. train.py's resume path was changed to this
        # on 2026-08-30 and play.py was not, which made the two disagree the
        # moment the run folder lost its year prefix: "08-30_23-55-48_..."
        # sorts BEFORE "2026-08-30_23-13-05_...", so the default alphabetical
        # pick silently replayed an older run while reporting no error at all.
        # Observed 2026-08-31, when RT-107 was to be watched and a different
        # policy appeared in the viewport.
        #
        # The one case where mtime is NOT the truth: a run folder COPIED
        # between machines gets a fresh mtime. Pass --load_run explicitly when
        # replaying from a copied folder.
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, sort_alpha=False
        )

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
    #
    # THE START HEIGHT IS PART OF THE PROBE POSE, added 2026-09-05. It is not
    # a play.py flag but a hydra override (env.start_tip_above_entrance), so
    # it never reached this name and a height series -- +40, +50, +60 mm on
    # one checkpoint -- silently wrote three runs into the same replay/. It
    # is therefore read off the cfg, not off args_cli, and it is appended
    # ALWAYS: a metrics file is only readable together with the height it was
    # measured at. Nothing else in the repo reads this directory name
    # (checked 2026-09-05), so the plain "replay" case changing to
    # "replay_h+30mm" breaks no caller.
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
        # The videos live at the WORKING-DIRECTORY root, not inside the run
        # folder: they are the evidence a reward hack is watched in, and they
        # are collected across runs and carried off the training machine by
        # hand. One flat place beats hunting through logs/rsl_rl/<exp>/<run>/.
        # The run folder name is kept as the sub-directory, so every video
        # still carries its date, rung and seed.
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
    # THE WIDTH GUARD (D-188). Two observation layouts exist now (force 28,
    # wrench 31) and rsl_rl stores no width of its own: `runner.load` would
    # die with `size mismatch for actor.0.weight`, correct but nameless. One
    # readable line instead, naming the obs_wrench_mode that fits. The
    # checkpoint is read once more here (CPU, no grad); no padding, no
    # weight transfer -- a 28-wide policy is not a 31-wide policy.
    _refusal = _checkpoint_width_refusal(resume_path, env.unwrapped)
    if _refusal is not None:
        print(f"[play] REFUSED: {_refusal}")
        raise SystemExit(2)
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
    # Guarded getattr: the counters are insertion-specific and this script
    # must keep working for other tasks.
    replay_env = env.unwrapped
    last_reported_ep = 0

    # reset environment
    obs = env.get_observations()
    timestep = 0
    steps_done = 0
    # RT-149 trace: raw obs BEFORE the policy's own normaliser (get_observations
    # returns the env's TensorDict; the policy normalises internally). Keyed
    # "policy" -- vecenv_wrapper.get_observations returns a TensorDict.
    trace = None
    if args_cli.trace_obs is not None:
        # ``ee_quat`` added 2026-09-06 (the RT-149b line of Plaene/Plan-Merge.md).
        # Without it the trace carries yaw but no TILT, and RT-157's parked
        # failure pose -- x +7.79 / y -15.9 / z +4.36 mm at both start heights
        # -- cannot be told apart from a part standing flat beside the pocket.
        # Four floats per env per step; a 1024-env run already weighs ~90 MB,
        # so probe with fewer envs rather than fewer steps.
        # THE TRACE IS A POLICY VIEW, NOT A POSE LOG -- and since D-182 those
        # are two different things. Before the observation noise model there
        # was no difference: the observation WAS the pose, so every reader
        # treated a trace row as a measured pose and was right. Now
        # `tip_rel` carries a per-episode bias and `force` a per-step
        # Gaussian, and row 0 is the odd one out on top of that: it comes
        # from `env.get_observations()`, which the rsl_rl wrapper answers by
        # calling `_get_observations()` itself, so the hook never runs --
        # every LATER row comes from `env.step()` and is noised.
        # THE SIGMAS ARE RECORDED HERE so the FILE says which it is. A reader
        # that finds them non-zero is holding a policy view and must not
        # report its numbers as poses; at 0.0 the two coincide again. Pinning
        # them stays a per-run choice, not a decision nailed into this file
        # (user, 2026-09-12).
        # Guarded the same way `replay_env`'s counters are, and for the same
        # reason: these four fields are insertion-specific and this script
        # must keep working for other tasks. A missing field records `None`,
        # which reads as "this task has no such width", not as zero.
        def _scatter(name):
            v = getattr(replay_env.cfg, name, None)
            return None if v is None else float(v)

        # THE SLICES COME FROM THE TABLE (D-188): `obs_wrench_mode` decides
        # whether channels 28:31 exist, and the trace records which layout it
        # read so a reader never guesses. Literals stood here until 2026-09-13.
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
        # TRUTH CHANNELS (RT-189s1pf, 2026-09-13). The rows above are the POLICY
        # VIEW and carry the observation noise; these are read from the env's
        # own buffers BEFORE the noise model, at the same moment as the
        # observation row, and change nothing the policy sees. Guarded: another
        # task, or an env without a welded tool, records no truth block.
        truth = None
        _peg = getattr(replay_env, "_peg_body_idx", None)
        if _peg is not None and hasattr(replay_env, "_force_tared_raw"):
            truth = {"force_tared_raw_n": [], "force_ema_n": [], "tip_true_m": [],
                     "body_pose_w": [], "osc_delta": [], "osc_target_pose_w": [],
                     "interpen_max_m": [],
                     # D-188: the torque buffers are filled in EVERY mode, so a
                     # force-mode replay still shows what the policy did not see.
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
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
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
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if trace is not None:
                trace["done"].append(dones.cpu().tolist())
                trace_steps += 1
                if trace_steps >= args_cli.trace_steps:
                    break
            # THE PLAIN STOP, added 2026-09-06. Until it existed the ONLY way
            # to make play.py end by itself was --trace-obs, so a probe that
            # wanted nothing but demo_metrics.json had to write a trace it
            # never read: at 1024 envs x 288 steps that is ~84 MB of JSON per
            # run, and four runs of it. Counted for every step, traced or not.
            steps_done += 1
            if args_cli.max_steps is not None and steps_done >= args_cli.max_steps:
                print(f"[play] stopping after {steps_done} steps (--max-steps)")
                break
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

    if trace is not None:
        import json
        trace_path = os.path.abspath(args_cli.trace_obs)
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as fh:
            json.dump(trace, fh)
        print(f"[trace] wrote {trace_steps} steps x {trace['num_envs']} envs to {trace_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
