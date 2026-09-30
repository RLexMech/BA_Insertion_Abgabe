# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e articulation cfg (D-020 option b, D-023, D-025, D-019).

Increment 1 spawned the bare arm. The peg increment keeps every value below
unchanged and only points the spawn at the peg-welded USD at scene-build time.

Derived from the shipped ``UR10e_CFG`` via ``.copy()`` with targeted overrides
only, so no nested cfg field is silently reset to a class default. Actuator
gains stay at the shipped values (shoulder 1320/72.66, elbow 600/34.64,
wrist 216/29.39) — they are measured later, not guessed now.

Note: the articulation root API of the shipped asset sits on
``/ur10e/root_joint``, not on ``/ur10e``. ``articulation_root_prim_path`` is
left at ``None`` and only set if a run complains about the root.
"""

from isaaclab_assets.robots.universal_robots import UR10e_CFG

from .proxytask_tasks_cfg import ROBOT_BASE_POS

UR10E_HOME_CFG = UR10e_CFG.copy()

# Shipped value is MISSING and must be set.
UR10E_HOME_CFG.prim_path = "/World/envs/env_.*/Robot"

# D-023: base centre on the plate top, on the x axis, 415 mm from the bore along
# +Y. Imported rather than repeated: the base height *is* the plate top, so the
# robot stands on the fixture instead of coincidentally sharing its number.
UR10E_HOME_CFG.init_state.pos = ROBOT_BASE_POS
UR10E_HOME_CFG.init_state.rot = (1.0, 0.0, 0.0, 0.0)

# Home pose, keyed by joint name (asset joint names verified by probe).
#
# IK branch 3, not the branch D-025 originally selected. Two corrections, both
# measured on the training machine 2026-07-26 rather than derived:
#
# 1. The USD base_link frame is rotated by pi about z relative to the DH base
#    frame compute_home_pose.py solves in -- the caveat D-025 recorded in
#    advance. With the original values the tool pointed down correctly
#    (z . (0,0,-1) = 0.999997) and the standoff was right, but the flange sat at
#    y = +0.41494 relative to the base where the bore is at y = -0.415.
# 2. D-025's branch put the upper arm through the tabletop. Measured: forearm_link
#    at (-0.118, -0.167, 0.452), inside the plate footprint and 0.25 m below the
#    plate underside, with shoulder_pan held 6.6 deg off target by contact and not
#    converging over 3 s. D-025's criteria never included plate clearance because
#    the plate carried no collision at the time.
#
# scripts/tools/screen_home_pose_branches.py screens all eight branches against
# the plate volume, after validating the kinematic chain against the measured
# body positions to 0.06 mm. Four branches keep the whole arm at z >= 0.905, so
# the flange is the lowest point. Among those, D-025's own remaining criteria
# (all |q| < pi, q3 < 0, max |sin q5|, min ||q||) select branch 3.
UR10E_HOME_CFG.init_state.joint_pos = {
    "shoulder_pan_joint": 2.003843,
    "shoulder_lift_joint": -1.911842,
    "elbow_joint": -2.265118,
    "wrist_1_joint": 2.606164,
    "wrist_2_joint": -1.570796,
    "wrist_3_joint": 0.433047,
}

# Deliberate deviation from D-020 for commissioning only: with no action
# applied, the implicit actuators must hold the home pose against gravity —
# that is the observable increment 1 wants, and an implicit gain test.
UR10E_HOME_CFG.spawn.rigid_props.disable_gravity = False

UR10E_HOME_CFG.spawn.articulation_props.fix_root_link = True

# Peg increment: the spawn-time prerequisite for reading contact forces off the
# peg link (D-020). No ContactSensor is created yet -- that is a subsystem, and
# this increment tests one hypothesis (does the peg enter the bore). The flag
# alone costs nothing and keeps the sensor a later one-liner rather than a
# re-authoring of the asset.
#
# A configclass accepts an unknown attribute silently, so a wrong field name
# here would be a no-op nobody notices. The startup report prints the value
# back for exactly that reason; if it reports "absent", the field name is wrong.
UR10E_HOME_CFG.spawn.activate_contact_sensors = True

# The peg-welded USD (D-019) is NOT set here. Resolving it at module import
# would raise on any machine without the asset and take task registration down
# with it, which is the failure mode resolve_fixture_usd_path() was written to
# avoid. The env sets spawn.usd_path in _setup_scene instead.
