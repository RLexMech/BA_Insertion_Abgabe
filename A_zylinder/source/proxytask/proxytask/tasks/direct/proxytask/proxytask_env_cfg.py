# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene, control and reward settings for cylindrical peg insertion."""

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
    """Return the path to the Tisch USD, or raise with a actionable message."""
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
    decimation = 2
    episode_length_s = 4.0
    action_space = 6
    observation_space = 19
    state_space = 0

    # Joint-target change in radians per control step.
    action_scale = 0.02
    reset_joint_noise = 0.01
    reward_w_approach = 2.0
    reward_w_depth = 100.0
    reward_w_success = 10.0
    reward_w_action = 0.01
    reward_w_misplaced = 20.0
    reward_w_align = 2.0
    gate_min_alignment = 0.99

    metrics_window_iterations = 20

    metrics_success_threshold = 0.9

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    robot_cfg: ArticulationCfg = UR10E_HOME_CFG
    ee_body_name = "wrist_3_link"

    fixture_cfg: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path="",
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )

    fixture_pos = proxytask_tasks_cfg.FIXTURE_POS
    plate_top_z = proxytask_tasks_cfg.PLATE_TOP_Z
    plate_bottom_z = proxytask_tasks_cfg.PLATE_BOTTOM_Z
    plate_half_extents = proxytask_tasks_cfg.PLATE_HALF_EXTENTS
    bore_entrance_pos = proxytask_tasks_cfg.BORE_ENTRANCE_POS
    robot_base_pos = proxytask_tasks_cfg.ROBOT_BASE_POS
    home_standoff_z = proxytask_tasks_cfg.HOME_STANDOFF_Z

    fixed_asset: FixedAssetCfg = FixedAssetCfg()
    held_asset: HeldAssetCfg = HeldAssetCfg()
    peg_body_name = "peg_link"
    peg_tip_offset = proxytask_tasks_cfg.PEG_TIP_OFFSET
    peg_length = proxytask_tasks_cfg.PEG_LENGTH

    report_at_steps = (2, 60, 180)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=3.0, replicate_physics=True)
