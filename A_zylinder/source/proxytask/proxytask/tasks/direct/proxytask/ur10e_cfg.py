# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e articulation settings and insertion home pose."""

from isaaclab_assets.robots.universal_robots import UR10e_CFG

from .proxytask_tasks_cfg import ROBOT_BASE_POS

UR10E_HOME_CFG = UR10e_CFG.copy()

UR10E_HOME_CFG.prim_path = "/World/envs/env_.*/Robot"

UR10E_HOME_CFG.init_state.pos = ROBOT_BASE_POS
UR10E_HOME_CFG.init_state.rot = (1.0, 0.0, 0.0, 0.0)

UR10E_HOME_CFG.init_state.joint_pos = {
    "shoulder_pan_joint": 2.003843,
    "shoulder_lift_joint": -1.911842,
    "elbow_joint": -2.265118,
    "wrist_1_joint": 2.606164,
    "wrist_2_joint": -1.570796,
    "wrist_3_joint": 0.433047,
}

UR10E_HOME_CFG.spawn.rigid_props.disable_gravity = False

UR10E_HOME_CFG.spawn.articulation_props.fix_root_link = True

UR10E_HOME_CFG.spawn.activate_contact_sensors = True
