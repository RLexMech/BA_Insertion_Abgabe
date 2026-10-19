# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e articulation settings and insertion home pose."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .insertion_tasks_cfg import WORKCELL_ROBOT_BASE_POS

UR5E_HOME_JOINT_POS = {
    "shoulder_pan_joint": 3.141293,
    "shoulder_lift_joint": -1.627801,
    "elbow_joint": -1.791996,
    "wrist_1_joint": -1.292592,
    "wrist_2_joint": 1.570796,
    "wrist_3_joint": 1.570497,
}

UR5E_HOME_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=UR5E_HOME_JOINT_POS,
        pos=WORKCELL_ROBOT_BASE_POS,
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=None,
            damping=None,
            friction=None,
            armature=None,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=None,
            damping=None,
            friction=None,
            armature=None,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=None,
            damping=None,
            friction=None,
            armature=None,
        ),
    },
)
"""UR5e articulation, bare arm, home pose from the IK screener, drives from the USD.

Asset structure:

  - 7 rigid bodies, ``base_link`` .. ``wrist_3_link``. There is NO ``ee_link``,
    so the flange body to weld against is ``wrist_3_link``.
  - the asset already carries ``/ur5e/root_joint``, a fixed joint with no
    body0, which is why the articulation root sits there.
  - it also carries ``/ur5e/joints/robot_gripper_joint``, a fixed joint from
    ``wrist_3_link`` to nothing. That dangling joint is the asset's own
    attachment point for a tool and is the obvious place to hang the suction
    chain, rather than authoring a new joint.
"""
