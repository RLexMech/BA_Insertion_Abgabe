# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e articulation cfg — scene-build phase C, step C1.

Asset facts below are VERIFIED on the training machine 2026-08-18 by
``scripts/probe_assets.py`` (Isaac Sim 5.1, conda env ``env_isaaclab``):

  - the USD path resolves and the stage opens; ``defaultPrim`` is ``/ur5e``;
  - the six revolute joints carry exactly the UR10e names, so the
    ``shoulder_.*`` / ``wrist_.*`` actuator regexes below do match;
  - ``PhysicsArticulationRootAPI`` sits on ``/ur5e/root_joint``, not on
    ``/ur5e`` — same as the UR10e asset, so ``articulation_root_prim_path``
    stays ``None`` here and is only set if a run complains;
  - every config field used below exists on its cfg class in this Isaac Lab
    build (``fix_root_link``, ``activate_contact_sensors``, the
    ``ImplicitActuatorCfg`` gain fields).

Spawned and stepped on the training machine 2026-08-18 (workcell run): the
articulation loads, the actuators hold a pose against gravity, and every joint
velocity is 0.000 rad/s at t = 3.0 s, so the arm settles rather than creeping.

The drive gains are SETTLED since 2026-08-25: they come from the USD, not from
this file — see the actuator block. Under them RT-46 measured a max joint
deviation of 0.000029 rad (0.002 deg) at t = 3.000 s. The 1.62 deg elbow error
recorded here on 2026-08-18, and the 1.810 deg of RT-18, both belong to the
UR10e placeholder gains that this file no longer sets.

Not shipped: ``ur5e_instanceable.usd`` does not exist on the 5.1 asset server
(probed, layer fails to open). Only the plain ``ur5e.usd`` is available.

Why this file is written out in full instead of ``UR5e_CFG.copy()``: Isaac Lab
2.3 ships no UR5e config at all. ``isaaclab_assets.robots.universal_robots``
defines UR10, UR10e and three UR10 gripper/suction variants — checked against
the v2.3.0 tag of that file. The USD asset itself does exist on the Nucleus
server (Isaac Sim 5.1 robot-asset list), so only the config is missing.

The proxy's ``ur10e_cfg.py`` derived from the shipped ``UR10e_CFG`` via
``.copy()`` so that no nested field of *that same robot* was silently reset.
(That file was deleted on 2026-08-28, S6 -- it is in git history.) The
argument does not transfer here: copying a different robot's config would
carry its link-specific tuning across a link-length change of ~200 mm per
segment, which is worse than stating every field.

The base pose is the ENVIRONMENT ORIGIN, ``(0, 0, 0)``, and that is a decision
rather than a convenience: Isaac Lab's Factory -- the reference implementation
for this task class -- puts the robot base there (``factory_env_cfg.py``) and
derives the task frame at runtime from the fixed asset's actual pose
(``factory_env.py``, ``fixed_pos_obs_frame``) instead of baking the target into
the scene layout. Imported from ``insertion_tasks_cfg`` rather than written as
a literal, so the cell geometry keeps exactly one owner.

Consequence worth knowing before reading any z in this project: the table tops
are NEGATIVE. Table 1 sits at minus the mounting-plate thickness.

What this file does NOT own:
  - the welded gripper/part chain (D-064). ``spawn.usd_path`` stays pointed at
    the bare arm until that asset exists; the env may override it in
    ``_setup_scene``, the way the proxy task already does for the peg.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .insertion_tasks_cfg import WORKCELL_ROBOT_BASE_POS

# Home pose: the LEADING TOOL POINT 150 mm above the block top face, tool axis
# down. That is the part's underside, FLANGE_TO_PART_BOTTOM = 152.0 mm below
# the flange, so the flange itself stands 302 mm up.
#
# Computed offline by scripts/tools/screen_home_pose_branches.py and NOT a
# placeholder any more. The chain it uses is validated against the simulator:
# driven with the joint angles the training machine reported at t = 3.0 s
# (2026-08-18, velocities all zero, so the pose was settled), the model
# reproduces all five measured body positions to 0.07 mm. The same test run
# with the identity frame convention instead of Rz(pi) is off by up to 1030 mm,
# so the agreement discriminates rather than merely tolerates.
#
# Selection: of the eight IK branches, two put some part of the arm BELOW the
# leading tool point (119 and 135 mm below it, over the block) and are
# rejected. That gate is stricter than the proxy screening, which only asked
# for positive clearance -- with a single plate the flange happened to be the
# lowest point anyway, so the distinction never surfaced. Six branches keep the
# tool point lowest; among them D-025's criteria (all |q| < pi, then q3 < 0,
# then max |sin q5|, then min ||q||) select branch 1, at +0.1865 m clearance
# over table 1.
#
# ONE reason this may still need recomputing: the pose has not been spawned
# WITH the tool at the POCKET target yet. Screened, not verified.
#
# Recomputed three times. 2026-08-18 twice: after the mounting plate was
# measured (24 mm thick, 153 mm base square), and after the lateral sign was
# resolved (fixture to the robot's RIGHT). 2026-08-19: after the frame
# rotation to +X-towards-fixture (see insertion_tasks_cfg.py WORKCELL header);
# the block moved from (0.133, 0.4512) to (0.4512, -0.133) and the screener
# again selects branch 1 at the full 150 mm -- the same physical pose relative
# to the block, rotated with the cell. shoulder_pan lands at -3.1409, which is
# 0.0007 rad INSIDE the |q| < pi gate of D-025; if the cell geometry ever
# shifts it past -pi, the screener will exclude the branch and must be re-read,
# not overridden.
#
# Recomputed a fourth time on 2026-08-23, for the tool chain: the standoff
# reference moved from the flange down to the part underside, so the flange
# rose by FLANGE_TO_PART_BOTTOM and three of the six angles changed. Branch 1
# survives that move, and shoulder_pan is unchanged at -3.140928 -- the
# standoff is a change in REACH, and the pan angle is set by where the block
# lies, not by how high the arm holds the tool. Superseded values, for anyone
# reading an older log: shoulder_lift -1.700567, elbow -2.154990,
# wrist_1 -0.856832.
# Recomputed a FIFTH time on 2026-08-24, and this is the one that moved the
# arm rather than the reference: the target stopped being the block CENTRE and
# became the POCKET. The block centre was never a place the part had to go --
# it was the centre of a solid box that has since become four walls around the
# user's CAD cut-out. The pocket sits 49.8 mm farther out in x and 0.5 mm over
# in y (docs: check_workcell_geometry.py section 3c prints both).
#
# Screened 2026-08-24 on the dev laptop, four obstacles now (table 1, table 2,
# block, and the 100 mm REAR WALL, which is new): 8 branches, 6 clear, branch 1
# selected again at +0.1865 m over table 1. The rear wall never becomes the
# limiting obstacle -- the two rejected branches (5 and 7) hang over the BLOCK,
# as before.
#
# READ THIS BEFORE "FIXING" THE SIGN: shoulder_pan flipped from -3.140928 to
# +3.141293. That is NOT a 360-degree turn and not an error. Wrapped to the
# same side, the old value is +3.142257, so the arm turned 0.000964 rad
# = 0.055 deg and wrap_to_pi picked the other representative. Both are inside
# the joint limits [-6.283185, +6.283185].
#
# STILL TIGHTER THAN BEFORE, and it is the number to watch: |q| = 3.141293 is
# 0.000300 rad inside D-025's |q| < pi gate, where the old pose had 0.000665.
# If the cell geometry moves this past pi, the screener EXCLUDES the branch.
# Then the screener output gets read, not overridden.
#
# Recomputed a SIXTH time on 2026-08-24 (evening), and this one moved exactly
# one joint. The user looked at the render and saw the part's lugs pointing
# away from the pocket features they have to enter. The part fits one way
# round, so the tool is turned 180 degrees about its own axis at the home pose:
# TOOL_HOME_YAW_RAD in insertion_tasks_cfg.py, applied in the SCREENER's target
# orientation rather than typed in here, so the next screener run reproduces
# this table instead of undoing it.
#
# THE SELF-CHECK THAT MAKES IT BELIEVABLE: a pure spin about the tool axis must
# move wrist_3 and NOTHING else, because wrist_3 rotates about that same axis.
# The re-run changed wrist_3 from -1.571096 to +1.570497, a difference of
# 3.141593 rad, and left the other five bit-identical. Clearance unchanged at
# +0.1865 m over table 1, branch 1 still selected.
#
# Superseded values, for anyone reading an older log: shoulder_pan -3.140928,
# shoulder_lift -1.510209, elbow -1.909351, wrist_1 -1.292829,
# wrist_3 -1.570131 and then -1.571096.
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
            # Supervisor decision 2026-08-25: gravity stays ON for training.
            # With the USD drives below there is barely a droop left to accept:
            # RT-46 measured the flange standoff at 0.317026 m against 0.3170
            # nominal (t = 3.000 s), i.e. 0.000026 m. Supersedes p1-gains D-082
            # (disable_gravity=True).
            disable_gravity=False,
            # max_depenetration_velocity stays UNSET (PhysX default), which is
            # the pre-D-159 state. D-159 set it to 5.0 and RT-113 did not show
            # the expected drop, so the user reverted it on 2026-08-31. NOTE
            # for whoever tries this again: unset is NOT the same as 0.0 --
            # 0.0 would forbid the solver to separate overlapping bodies at
            # all, a third setting nobody has measured.
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
        # Spawn-time prerequisite for reading contact forces off a link later
        # (D-020). No ContactSensor is created here; the flag alone keeps that
        # a one-liner instead of a re-authoring of the asset.
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=UR5E_HOME_JOINT_POS,
        pos=WORKCELL_ROBOT_BASE_POS,
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # ALL drive parameters come from the USD (None -> "value from the USD
        # joint prim", Isaac Lab 2.3 actuator docs). Supervisor decision
        # 2026-08-25: the shipped UR5e asset already carries tuned per-joint
        # drive values, so the earlier UR10e placeholder gains (1320/600/216)
        # are gone.
        #
        # VERIFIED on the training machine, not assumed. RT-45 probed the asset
        # the env actually spawns -- the WELDED robot+tool+peg USD -- and found
        # it carries the same authored drives as the bare arm: shoulder_pan
        # stiffness 9400.5009765625, damping 0.37800323963165283, maxForce
        # 150.0. RT-46 then measured the arm holding its home pose under them:
        # max joint deviation 0.000029 rad (0.002 deg) at t = 3.000 s.
        #
        # Measurement history of the placeholder gains: decisions_inbox.md
        # (2026-08-25 entry) and HANDOFF-SZENE.md.
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

Asset structure as probed, for whoever wires the gripper next (D-064):

  - 7 rigid bodies, ``base_link`` .. ``wrist_3_link``. There is NO ``ee_link``,
    so the flange body to weld against is ``wrist_3_link``.
  - the asset already carries ``/ur5e/root_joint``, a fixed joint with no
    body0, which is why the articulation root sits there.
  - it also carries ``/ur5e/joints/robot_gripper_joint``, a fixed joint from
    ``wrist_3_link`` to nothing. That dangling joint is the asset's own
    attachment point for a tool and is the obvious place to hang the suction
    chain, rather than authoring a new joint.
"""
