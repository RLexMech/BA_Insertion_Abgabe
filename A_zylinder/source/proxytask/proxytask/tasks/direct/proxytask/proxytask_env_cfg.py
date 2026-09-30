# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene cfg: static Tisch fixture + UR10e with the welded peg, home pose.

DEMO SPRINT (branch demo-insertion-sprint, 2026-07-26, UNVERIFIED): this is a
separate, radically simplified increment cut from increment-2-verified
(90621a8), not a continuation of increment 3. Actions now command joint-
position deltas, the observation is a 19-dim vector (D-030's layout without
the wrench/contact-sensor channels, since there is no force reward tonight),
and there is a dense reward. None of increment 1-3's acceptance baselines
apply to this branch's Ø25 mm peg.

Whether the peg actually fits the bore is deliberately *not* tested here.
That measurement lives in ``scripts/verify_peg_passability.py``, so a motion
sequence never enters the env's semantics.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from . import proxytask_tasks_cfg
from .proxytask_tasks_cfg import FixedAssetCfg, HeldAssetCfg
from .ur10e_cfg import UR10E_HOME_CFG


def resolve_fixture_usd_path() -> str:
    """Return the path to the Tisch USD, or raise with a actionable message.

    Called from ``_setup_scene``, not at module import, so ``list_envs.py``
    keeps working on a machine without the gitignored asset. The file is
    lowercase ``tisch.usd`` on disk while the docs say ``Tisch.usd``; Windows
    hides the difference, Linux would not, so both spellings are tried.
    """
    env_override = os.environ.get("PROXYTASK_TISCH_USD")
    candidates = []
    if env_override:
        candidates.append(env_override)
    asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Tisch")
    candidates.append(os.path.join(asset_dir, "tisch.usd"))
    candidates.append(os.path.join(asset_dir, "Tisch.usd"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Tisch fixture USD not found. Tried: "
        + "; ".join(candidates)
        + ". Set PROXYTASK_TISCH_USD or place the asset per docs/asset_contract_tisch.md."
    )


@configclass
class ProxytaskEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2  # 60 Hz control rate (D-024)
    # 4.0 s = 240 control steps. Shortened from 5.0 s on 2026-07-27: a run
    # showed the arm spending the tail of each episode searching, and the
    # descent it has to make is ~125 mm (100 mm standoff plus 25 mm of
    # insertion), which 240 steps cover with room to spare.
    episode_length_s = 4.0
    # - spaces definition. action_space stays interim (joint deltas, no OSC
    #   this branch). observation_space is the demo-sprint 19-dim layout:
    #   0:6 joint_pos, 6:12 joint_vel (finite-difference, not the raw solver
    #   channel -- see D-030 addendum 2), 12:15 peg-tip-relative-to-bore,
    #   15:19 EE quat (wxyz). No wrench/contact-sensor channels tonight.
    action_space = 6
    observation_space = 19
    state_space = 0

    # Demo-sprint action/reward knobs. Starting values; the escalation ladder
    # in the plan adjusts these one at a time if the first curve is flat.
    action_scale = 0.02  # rad of joint-target delta per unit action, per step
    reset_joint_noise = 0.01  # rad, uniform +-, applied to the home pose at reset
    reward_w_approach = 2.0
    reward_w_depth = 100.0
    reward_w_success = 10.0
    reward_w_action = 0.01
    # Charged per metre of depth below the plate top that is not inside the
    # bore. At the ~55 mm read under the plate this costs 1.1 per step against
    # an approach term of ~0.13 there, so the space under the table is clearly
    # worse than the home pose (~0.17) rather than better, which is what it
    # was before 2026-07-27.
    reward_w_misplaced = 20.0
    # Charged for the peg's tool axis departing from world-down: 0 when it
    # points straight into the plate, this value when it lies flat. Added
    # 2026-07-27 after every environment converged on reaching the bore
    # sideways -- a distance-only approach term is maximised from any
    # direction, so that pose scored about zero, the best value reachable
    # without inserting, and nothing pulled the arm out of it.
    reward_w_align = 2.0
    # Minimum tool-axis alignment before depth counts at all. 0.99 is 8.1
    # degrees; the geometry itself allows at most 5.7 degrees inside a 2.5 mm
    # clearance over the 50 mm peg, so this gates on physical impossibility
    # rather than on a preferred style. The home pose reads 0.9998.
    gate_min_alignment = 0.99

    # How much training the trailing success-rate window should span, in PPO
    # iterations. The window is derived from this and the env count, so runs
    # at 1024 and 4096 envs are ranked over comparable amounts of training
    # rather than over a fixed episode count that means different things at
    # each scale. A floor of 2000 episodes applies regardless.
    metrics_window_iterations = 20

    # Success rate that counts as "solved" for the sample-efficiency figure
    # (episodes_to_threshold). Deliberately below 1.0: the episode count at
    # which a run first holds this is what separates configurations once
    # several of them all finish at 100 %.
    metrics_success_threshold = 0.9

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = UR10E_HOME_CFG
    ee_body_name = "wrist_3_link"

    # fixture: no rigid_props → static collider (D-018). usd_path is resolved in
    # _setup_scene via resolve_fixture_usd_path(); the placeholder here is never
    # spawned.
    fixture_cfg: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path="",
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )

    # Task geometry (D-022, D-023). Defined once in proxytask_tasks_cfg.py and
    # mirrored here as cfg fields, so the robot base height is derived from the
    # plate top rather than repeated as a matching literal.
    fixture_pos = proxytask_tasks_cfg.FIXTURE_POS
    plate_top_z = proxytask_tasks_cfg.PLATE_TOP_Z
    plate_bottom_z = proxytask_tasks_cfg.PLATE_BOTTOM_Z
    plate_half_extents = proxytask_tasks_cfg.PLATE_HALF_EXTENTS
    bore_entrance_pos = proxytask_tasks_cfg.BORE_ENTRANCE_POS
    robot_base_pos = proxytask_tasks_cfg.ROBOT_BASE_POS
    home_standoff_z = proxytask_tasks_cfg.HOME_STANDOFF_Z

    # Peg (D-005, D-019). The peg is a welded link of the robot articulation,
    # not a separately spawned body, so there is no spawn cfg for it here --
    # only the geometry the env needs in order to report where its tip is.
    fixed_asset: FixedAssetCfg = FixedAssetCfg()
    held_asset: HeldAssetCfg = HeldAssetCfg()
    peg_body_name = "peg_link"
    peg_tip_offset = proxytask_tasks_cfg.PEG_TIP_OFFSET
    peg_length = proxytask_tasks_cfg.PEG_LENGTH

    # The startup report is printed at these step counts after reset. The early
    # one only shows the buffers are filled; the late ones decide whether the
    # pose is held (acceptance criterion 2: unchanged after 5 s). At 60 Hz these
    # are 0.03 s, 1 s and 3 s.
    report_at_steps = (2, 60, 180)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=3.0, replicate_physics=True)
