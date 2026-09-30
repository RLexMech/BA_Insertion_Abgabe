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
    """D-188: the one line that refuses a checkpoint of the other observation
    width, or ``None``. Same binding as play.py's; the rule lives in
    ``scripts/tools/checkpoint_width.py``. An env without ``obs_wrench_mode``
    (another task) is never refused here."""
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

# import logger
logger = logging.getLogger(__name__)

import insertion.tasks  # noqa: F401
from insertion.tasks.direct.insertion.insertion_env_cfg import resolve_start_lateral_offset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# The sidecar's OWN format tag, not the AutoDR state's. `autodr.state_dict`
# carries "autodr-1" inside the envelope and keeps owning its own shape; this
# tag versions the WRAPPER -- the run identity a resume is checked against.
SIDECAR_FORMAT = "autodr-sidecar-1"


def _autodr_sidecar_path(model_path: str) -> str:
    """``.../model_1500.pt`` -> ``.../autodr_1500.json`` (plan section 5).

    rsl_rl names every checkpoint ``model_{it}.pt`` (rsl_rl 3.0.1,
    ``OnPolicyRunner.learn``: ``self.save(os.path.join(self.log_dir,
    f"model_{it}.pt"))``), so the AutoDR state can be the same iteration under
    a different stem and a resume finds it from the checkpoint path alone --
    no second flag, no folder scan.

    An unexpected stem is NOT quietly forced into the pattern. It still gets a
    findable name, and the mismatch is PRINTED, because a sidecar written
    under a name the resume cannot derive is a silent loss of the run state.
    """
    head, name = os.path.split(model_path)
    stem = os.path.splitext(name)[0]
    if stem.startswith("model_"):
        return os.path.join(head, "autodr_" + stem[len("model_"):] + ".json")
    out = os.path.join(head, "autodr_" + stem + ".json")
    print(f"[autodr-state] checkpoint {name!r} is not 'model_<it>.pt'; sidecar named {out!r}")
    return out


def _save_with_autodr(save_fn, provider, seed, run_name: str):
    """Wrap ``runner.save`` so every checkpoint gets its AutoDR sidecar.

    WHY A WRAPPER AND NOT A CALL SITE: ``learn()`` saves on its OWN schedule
    (``save_interval``) and offers no callback, so the only place that sees
    every checkpoint is ``save`` itself. rsl_rl 3.0.1 calls it as
    ``self.save(...)`` -- an INSTANCE attribute lookup -- so replacing the
    attribute on the runner catches the interval saves, learn()'s final save
    and this script's own early-stop save with one hook.

    The model is written FIRST. If the state dump raises, the checkpoint still
    exists and the run can be inspected; the reverse would leave a state file
    describing a policy nobody has.
    """
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
    # No year (user, 2026-08-30): the whole project runs inside one year, so
    # "2026-" was four characters of nothing on the front of every folder name
    # and every TensorBoard legend. The sorting consequence that creates is
    # FIXED at the resume call below (sort_alpha=False), not left as a rule to
    # remember. Nothing renames history; old folders keep their names.
    log_dir = datetime.now().strftime("%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    # The run name is stamped automatically, so every run directory and every
    # TensorBoard curve says which task it was trained on. Relying on
    # --run_name for that works until it is forgotten once, and a run whose
    # configuration is unknown cannot be compared to anything.
    #
    # THE PEG AND POCKET SIZES ARE DELIBERATELY NOT IN THE NAME (user,
    # 2026-08-30). They were, as `peg30mm_pocket32mm`, from the demo sprint
    # until today. Two reasons they had to go, and the second is the one that
    # matters:
    #
    #  1. They are no longer the rung. D-033 made the pocket side the
    #     curriculum variable of the SQUARE PROXY. The real task's ladder is
    #     the randomisation ranges below, and the pocket does not move at all.
    #  2. `cfg.held_asset.side` and `cfg.fixed_asset.side` are marked `[proxy]`
    #     in `insertion_env_cfg.PROXY_TASK_VALUES` -- the real part is not a
    #     square and has no "side". Printing them first in the folder name
    #     asserted the proxy's geometry as this run's, in the exact place a
    #     thesis figure takes its legend from.
    #
    # NOTHING IS LOST FROM THE RECORD: both values are still written to
    # demo_metrics.json (`peg_side_m`, `pocket_side_m`, labelled proxy there)
    # and to params/env.yaml, and `compare_runs.py` reads them from the JSON,
    # never from the folder name. Older runs keep their names; nothing renames
    # history.
    parts = []
    # WHICH DISTRIBUTION THE RUN TRAINED ON (Phase 5, plan section 10), and it
    # goes FIRST because it changes how every tag below reads. Under
    # `dr_mode='autodr'` THREE of the four static noise tags below are inert
    # BY REFUSAL -- `InsertionEnv.__init__` refuses autodr together with a
    # non-zero `fixture_yaw_noise_rad`, `fixture_tilt_noise_rad` or
    # `start_lateral_offset`, because a boundary already owns each of those.
    # The FOURTH, `offset{n}mm` from `fixture_pos_noise_xy`, IS still appended:
    # no boundary tracks it and it stays live under AutoDR on purpose (plan
    # section 1, `insertion_env.py` `_static_dr`). The round-1 comment here
    # said all four were inert and that was wrong.
    # Without this tag an AutoDR run and a run with no randomisation at all
    # would share a folder name while training on opposite distributions --
    # the exact failure the "currOFF" tag was added for, one mechanism over.
    #
    # THE MODE NAME IS THE TAG, not a hand-written "autodr": a mode this file
    # has never heard of still gets labelled, instead of being tagged as the
    # one mode somebody typed here.
    #
    # `hasattr` first, no default, like every tag below: a cfg without
    # `dr_mode` has no DR concept and must not be tagged either way. 'off'
    # stays untagged -- it is the state every run before Phase 5 was in, so a
    # tag there would read the same on every old line and distinguish nothing.
    if hasattr(env_cfg, "dr_mode") and str(env_cfg.dr_mode) != "off":
        parts.append(str(env_cfg.dr_mode))
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
    # D-110 (3): a run without the ladder and a run that climbed one are
    # different experiments, and the folder name is the legend of every curve.
    # Unlike the three ranges above this tag is NOT inert when off -- "off" is
    # exactly the state that has to be visible, because a fixed-rung success
    # rate read as a curriculum result would overstate the task.
    # ``hasattr`` first, no default: a cfg without the field has no ladder
    # concept at all, and tagging it "currOFF" would assert a fact about a
    # config that cannot carry it.
    if hasattr(env_cfg, "curriculum_enabled") and not bool(env_cfg.curriculum_enabled):
        parts.append("currOFF")
    # WHERE THE EPISODE STARTED (D-161), and it belongs beside "currOFF" for
    # the same reason: with the ladder off, the start height IS the rung, and
    # it is the one thing the 2026-08-31 run got wrong. The tag says "start"
    # plus the signed millimetres above the stage-2 opening plane, so a rung-0
    # run and a home-pose run cannot end up as two unlabelled lines in one
    # figure. ``hasattr`` first, no default: a cfg without the field has no
    # start-height concept and must not be tagged either way. ``None`` means
    # the bare home pose and gets its own word rather than a number, because
    # a number would look like a decision somebody took.
    if hasattr(env_cfg, "start_tip_above_entrance"):
        _start_h = env_cfg.start_tip_above_entrance
        if _start_h is None:
            parts.append("startHOME")
        else:
            # Plan step C (SBC): a sampled range reads "start-30..+30mm"; the
            # fixed start keeps its one number. Same hasattr rule as above.
            # A low at or above the high is the zero-width range = the fixed
            # start (the cfg default), and is tagged as such.
            _start_low = getattr(env_cfg, "start_tip_above_entrance_low", None)
            if _start_low is None or _start_low >= _start_h:
                parts.append(f"start{_start_h * 1000:+g}mm")
            else:
                parts.append(f"start{_start_low * 1000:+g}..{_start_h * 1000:+g}mm")
    # THE START FLOOR (SBC step 0), its own `if` beside the start tag: a run
    # whose lower edge CLIMBS must not share a legend with a fixed band. The
    # tag names the floor's start value; the high is the start tag above.
    if hasattr(env_cfg, "start_floor_m") and getattr(env_cfg, "start_tip_above_entrance", None) is not None:
        _floor_v = float(env_cfg.start_floor_m)
        if _floor_v < float(env_cfg.start_tip_above_entrance):
            parts.append(f"floor{_floor_v * 1000:+g}mm")
    # WHERE THE EPISODE STARTED, SIDEWAYS (D-170). Same argument as the start
    # height above, one axis over: with the offset at 0 every episode starts on
    # the pocket axis, and a run that trained on the rim must not share a
    # legend with one that never left the centre. NOT called "offset" -- that
    # word is already taken by fixture_pos_noise_xy above, and the two are
    # different quantities (that one moves the pocket AND the start together).
    # Inert at 0.0, like the three ranges at the top.
    # ONE RADIUS since D-178 (3), so the tag is `lat{r}mm`. D-176 (Phase 5
    # step B3) tagged the (x, y) pair as `lat{x}x{y}mm`; those folder names
    # keep their meaning and nothing renames history. The resolver is the one
    # place that reads the field's shape, and it refuses a pair by name.
    _lat = resolve_start_lateral_offset(getattr(env_cfg, "start_lateral_offset", 0.0))
    if _lat > 0.0:
        parts.append(f"lat{_lat * 1000:g}mm")
    # The scene, INVERTED 2026-08-30 (user): the tag names the block only when
    # it is ON. Through this phase the block is off on every run (D-156), so
    # "blockOFF" read the same on every line and told two runs apart never.
    # When the rear wall comes back it is the next phase, and THEN the tag is
    # the exception that has to be visible.
    #
    # THE FACT DID NOT GO AWAY WITH THE TAG. It is written to
    # demo_metrics.json as `spawn_workcell_block` (added in the same change)
    # and to params/env.yaml, so the qualifier D-156 attaches to every number
    # -- no rear wall, so the task is easier than the real cell -- still
    # travels with the numbers. A folder with no block tag before this change
    # does not exist: older runs carry "blockOFF" explicitly.
    #
    # ``hasattr`` first, no default: a cfg without the field has no block
    # concept at all and must not be tagged either way.
    if hasattr(env_cfg, "spawn_workcell_block") and bool(env_cfg.spawn_workcell_block):
        parts.append("blockON")
    # SEED, and ONLY the seed. CLAUDE.md names it as part of every run folder,
    # and it is the one thing here that genuinely varies between two runs of
    # the same configuration (D-053's protocol is several seeds per rung).
    #
    # THE ENV COUNT IS DELIBERATELY NOT HERE. It was, for a few hours on
    # 2026-08-30. D-117 (c) forbids the hyperparameter study from varying the
    # env count, so it is the SAME on every run being compared -- a tag that
    # reads the same on every line distinguishes nothing and only makes the
    # legend longer. It is still recorded where it belongs: `num_envs` in
    # demo_metrics.json and params/env.yaml.
    #
    # Appended AFTER the task tags and BEFORE --run_name, so the task reads
    # first and a hand-given name still has the last word.
    #
    # This does NOT touch the two logging print lines above (D-119): the Ray
    # workflow parses the bare timestamp, printed before any suffix is joined.
    if isinstance(getattr(agent_cfg, "seed", None), int):
        parts.append(f"seed{agent_cfg.seed}")
    if agent_cfg.run_name:
        parts.append(agent_cfg.run_name)
    suffix = "_".join(parts)
    # THE RT NAME IN THE FOLDER (user, 2026-09-13): rt_log.ps1 exports the
    # run name it was given as RT_NAME, and it goes right after the timestamp
    # with a double underscore, e.g. "09-13_15-24-11__RT-193_offset5mm_...".
    # Without rt_log.ps1 nothing is added and the old form stays. The bare
    # timestamp print above is untouched (D-119, the Ray workflow parses it).
    rt_name = os.environ.get("RT_NAME", "").strip()
    if rt_name:
        log_dir += f"__{rt_name}"
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
    # The rollout length, for the episode record's `iteration` column
    # (insertion_env._append_episode_rows, 2026-09-15). Same channel as
    # log_dir: an attribute on the cfg instance, not a cfg field.
    env_cfg.rollout_steps_per_iteration = agent_cfg.num_steps_per_env

    # A training run with the RL terms switched off would learn from a reward
    # of zero and still produce a full set of plausible curves. The switch
    # exists for MEASUREMENT envs (zero_agent, random_agent, the scripted
    # gate); it is never an experiment here. Refuse before the env is built,
    # so nothing is written to the log dir.
    if not getattr(env_cfg, "rl_terms_enabled", True):
        raise SystemExit(
            "rl_terms_enabled is False: this env computes no real reward, no force "
            "abort and no curriculum. That is the measurement configuration -- use "
            "scripts/zero_agent.py or scripts/random_agent.py for it. Training with "
            "it is always a mistake."
        )

    # THE TWO STOP CRITERIA ARE REFUSED TOGETHER, and before the env is built
    # so nothing is written to the log dir. They read DIFFERENT windows:
    # --stop-at-success-rate reads `_recent_successes` (every finished
    # episode, any `bounds_version`, boundary episodes included), while
    # --stop-when-dr-max reads `_fresh_successes` (regular episodes at the
    # current `bounds_version` only, plan section 4). Running both and
    # stopping on whichever fires first would put a number in the run report
    # that the other criterion never measured.
    if args_cli.stop_when_dr_max is not None and args_cli.stop_at_success_rate is not None:
        raise SystemExit(
            "--stop-when-dr-max and --stop-at-success-rate are two different stop "
            "rules over two different episode windows. Pass one."
        )
    # THE BAR IS A RATE. `--stop-when-dr-max 80`, meaning 80 percent, would
    # compare a number in [0, 1] against 80.0, never fire, and train to
    # --max_iterations while every block printed "bar 80.0000". That is
    # exactly the silent-non-firing failure the next refusal exists to
    # prevent, reached by a typo instead of by a config.
    for _flag, _val in (("--stop-when-dr-max", args_cli.stop_when_dr_max),
                        ("--stop-at-success-rate", args_cli.stop_at_success_rate)):
        if _val is not None and not (0.0 <= float(_val) <= 1.0):
            raise SystemExit(
                f"{_flag} is {float(_val)!r}. It is a success RATE and must lie in "
                "[0, 1]; 0.80 is eighty percent, 80 is nothing the criterion can ever "
                "reach."
            )
    # And the AutoDR rule needs the AutoDR provider. Refused here rather than
    # ignored later: a run given the flag under `dr_mode='off'` would train to
    # --max_iterations and look like a run whose criterion simply never fired.
    # A provider exists under dr_mode=autodr OR under a start floor (SBC step
    # 0: the No-DR branch runs the floor and the start-height ceiling on the
    # same machine, so its stop rule is the same all_at_max).
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        # sort_alpha=False: "latest run" means NEWEST BY TIME, not
        # alphabetically last. Isaac Lab's default is alphabetical
        # (isaaclab_tasks/utils/parse_cfg.py), which silently equates "latest"
        # with "last name in the list" -- true only while every folder carries
        # the same fixed-width date prefix. It stopped being true the moment
        # the year came out of the name: "08-31_..." sorts BEFORE
        # "2026-08-30_...", so a bare --resume would have reached for an OLD
        # run and trained on the wrong checkpoint without saying anything.
        # Time is what "latest" was always supposed to mean.
        #
        # The one case where mtime is NOT the truth: a run folder COPIED
        # between machines gets a fresh mtime. Pass --load_run explicitly when
        # resuming from a copied folder.
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, sort_alpha=False
        )

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
    # -- the AutoDR state travels with the checkpoint (plan section 5) -------
    # A resume is the SAME run: same seed, same fixture, same AutoDR state
    # (boundaries, buffers, `bounds_version`). The policy rides in
    # `model_<it>.pt`; the provider's state has no home in that file, so it
    # gets a sibling `autodr_<it>.json` written by the same call.
    #
    # Read through `env.unwrapped`, the same private-attribute route the stop
    # criterion below already uses for `_recent_successes`. `None` under
    # `dr_mode='off'`, and then none of this runs.
    _dr = getattr(env.unwrapped, "_dr", None)
    if _dr is not None:
        runner.save = _save_with_autodr(
            runner.save, _dr, agent_cfg.seed, os.path.basename(log_dir)
        )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # THE WIDTH GUARD (D-188): a checkpoint of the other observation
        # layout (force 28 / wrench 31) is refused with one readable line
        # naming the obs_wrench_mode that fits, before runner.load's nameless
        # `size mismatch`. The rule lives in scripts/tools/checkpoint_width.py.
        _refusal = _checkpoint_width_refusal(resume_path, env.unwrapped)
        if _refusal is not None:
            print(f"[train] REFUSED: {_refusal}")
            raise SystemExit(2)
        # load previously trained model
        runner.load(resume_path)
        if _dr is not None and agent_cfg.algorithm.class_name == "Distillation":
            # A DISTILLATION RUN LOADS A TEACHER, and a teacher is by
            # definition another run's policy. Plan section 5 calls that a
            # WARM START, its own condition. Reaching the resume path with it
            # would either load the teacher's boundaries as if they were this
            # run's, or refuse a run that never passed --resume with a message
            # about --resume. Refused instead of guessed.
            raise SystemExit(
                f"algorithm Distillation together with dr_mode={env_cfg.dr_mode!r}: the "
                "checkpoint being loaded is a TEACHER, not this run's own earlier self, "
                "so its AutoDR state is not this run's state. Plan section 5 calls this "
                "a warm start and gives it its own condition. Run distillation with "
                "dr_mode='off'."
            )
        if _dr is not None:
            # REFUSED, not warned. Continuing without the saved state would
            # put a TRAINED policy in front of boundaries reset to width 0.
            # Plan section 5 has a name for that and it is not a resume: it is
            # a WARM START, a different condition, which must never share a
            # seed group with runs from zero. Letting it run would hide that
            # difference inside a folder named like a continuation.
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
            # THE SEED IS THE RUN IDENTITY, and this is the check that makes
            # the refusal above mean what plan section 5 says. `--load_run`
            # names ANY run folder, so a weit -> eng warm start finds a
            # perfectly valid sidecar: same DR_DIMS, same centres, same
            # maxima, so `load_state_dict` accepts it and the run would carry
            # another seed's boundaries under a continuation's folder name.
            # Existence of a sidecar was never evidence of identity.
            #
            # NOT A COMPLETE IDENTITY CHECK, and the message says so: the
            # fixture variant is not in the envelope yet (it arrives with
            # `fixture_variant`, plan step 3).
            if _doc.get("seed") != agent_cfg.seed:
                raise SystemExit(
                    f"--resume from a run with seed {_doc.get('seed')!r} while this run "
                    f"has seed {agent_cfg.seed!r} (state file {_side}, written by run "
                    f"{_doc.get('run')!r}). Plan section 5: a resume is the SAME run -- "
                    "same seed, same fixture, same AutoDR state. Initialising from "
                    "ANOTHER run's policy is a WARM START and is its own condition "
                    "with its own five seeds."
                )
            # `load_state_dict` refuses a different boundary set, a different
            # centre and a different maximum on its own -- this script does
            # not re-check what autodr.py owns.
            _dr.load_state_dict(_doc["autodr"])
            print(
                f"[autodr-state] loaded {_side} (run {_doc.get('run')!r}, seed "
                f"{_doc.get('seed')!r}): bounds_version {_dr.bounds_version}, "
                f"all_at_max {_dr.all_at_max()}, phase {getattr(_dr, 'phase', 'dr')}"
            )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    #
    # TWO STOP CRITERIA, ONE BLOCK LOOP (Phase 5 step B6). The loop below is
    # D-037's and is unchanged in shape: rsl_rl's learn() runs to completion
    # with no callback, so the training is split into blocks and the criterion
    # is checked between them. init_at_random_ep_len only on the first block --
    # it staggers episode lengths at the start of training, and re-applying it
    # mid-run would truncate episodes for no reason.
    #
    # WHAT "N BLOCKS OF k TRAIN AS N*k WOULD" GETS RIGHT AND WRONG, corrected
    # in round 2 after reading rsl_rl 3.0.1 rather than assuming: the UPDATE
    # COUNT is exactly N*k -- every block runs k rollout-and-update passes.
    # The ITERATION INDEX is not. `learn()` does
    # `start_iter = self.current_learning_iteration`, loops
    # `for it in range(start_iter, start_iter + k)` and sets
    # `self.current_learning_iteration = it`, i.e. to the LAST index, not
    # past it. So every block boundary REPEATS one index: block 2 starts at
    # the index block 1 ended on. Consequences, all of them cosmetic-looking
    # and none of them harmless in a study: the final checkpoint of N blocks
    # is `model_{N*k - N}.pt`, one TensorBoard step is written twice per
    # boundary, and a checkpoint written at a boundary index is overwritten
    # (together with its sidecar, which is at least consistent).
    # THIS IS D-037's BLOCK LOOP, not step B6's, and it is NOT fixed here --
    # changing it would renumber every run this project has. It is written
    # down so the numbers can be read.
    #
    # What step B6 added is the SECOND criterion, not a second loop. Each one
    # is a closure returning ("fire" | "wait" | "abandon", one printed line).
    # "abandon" means the env does not expose the window this criterion reads.
    # The run then TRAINS THE REST IN ONE CALL and stops checking -- it does
    # not end early. Until round 2 this branch was a bare `break`, so a run
    # whose env exposed no window trained one block (50 iterations by default)
    # and exited with status 0 while both abandon lines printed "running to
    # max_iterations". That is the silent short run this file refuses
    # configurations to prevent, and it was the printed promise that was
    # false, not the intent.
    _dr_bar = args_cli.stop_when_dr_max

    def _criterion_success_rate():
        """D-037's rule, unchanged: the trailing window over EVERY episode.

        The window must be reasonably full before the criterion can fire, or
        an early lucky block ends the run -- the same quarter-window guard the
        env's episodes_to_threshold latch uses, for the same reason.
        """
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
        """The AutoDR rule (plan section 4). THREE conditions, all of them.

        (1) every AutoDR boundary sits at its maximum;
        (2) the fresh window holds at least ``_fresh_min`` episodes;
        (3) that window's rate is at or above the bar.

        THE WINDOW IS THE POINT. ``_fresh_successes`` holds REGULAR episodes
        (boundary episodes are nailed to an edge and have another
        distribution -- their rate is logged separately as
        ``dr/success_rate_boundary``) that STARTED under the current
        ``bounds_version``; the env clears it whenever a boundary actually
        moves. So a stop can never be bought with episodes from an easier
        distribution, which is the whole reason the stamp exists.

        THE RATE IS NOT A TEST RATE and the plan forbids calling it one:
        these episodes carry PPO's exploration noise. The test rate comes from
        the evaluation table alone (plan section 3).

        An under-full window prints its rate as "--" rather than a number.
        A rate over 40 episodes formatted to four decimals looks exactly like
        a rate over 2000, and the one thing this criterion must never do is
        let a reader take the first for the second.
        """
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
                    # init_at_random_ep_len is False here whatever `trained`
                    # is: the episodes are already staggered by the blocks
                    # that ran, and re-applying it would truncate them.
                    runner.learn(num_learning_iterations=_rest, init_at_random_ep_len=False)
                    trained += _rest
                break
            if state == "fire":
                stopped_early = True
                print(f"[stop-criterion] target reached at iteration {trained}; stopping.")
                break
        if stopped_early:
            # learn() saves on its own schedule, so the block that triggered the
            # stop may not have written one. Save explicitly: the whole point of
            # the run is the policy it ended with. Under AutoDR this call also
            # writes the sidecar, because `runner.save` is the wrapped one.
            final_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
            runner.save(final_path)
            print(f"[stop-criterion] saved {final_path}")
        print(note)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # THE RUN FOLDER CARRIES ITS OWN CURVES (user, 2026-09-13): the scalar CSV
    # export_tb_scalars.py makes by hand is written into the run folder at
    # the end, next to demo_metrics.json and the checkpoints, so ONE folder
    # holds everything that travels to the laptop. Never fails the run.
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

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
