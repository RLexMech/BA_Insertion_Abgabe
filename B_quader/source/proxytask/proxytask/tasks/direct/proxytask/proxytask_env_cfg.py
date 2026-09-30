# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene cfg: static Tisch fixture + UR10e with the welded square peg, home pose.

SQUARE-PEG PIVOT (branch square-peg-insertion, 2026-07-28, D-029, UNVERIFIED):
cut from the demo sprint (a7c7ebb), keeping its joint-delta actions and dense
reward. The peg is a 30 x 30 x 50 mm square prism, the hole a 35 mm deep blind
pocket in a generated table (D-033: scripts/author_tisch_square.py) whose asset
origin sits ON the opening plane at the pocket centre. The pocket side length
is the curriculum variable and therefore part of the asset filename, so the
resolver below asks for one specific size rather than for "the square table".
The observation is 25-dim (the demo layout plus
(cos 4 phi, sin 4 phi) of the peg yaw and the pocket quaternion of D-037), the
reward gains a C4-invariant yaw
term, and reset noise is per-joint: +-45 deg on wrist_3, +-0.01 rad elsewhere.
None of the round-peg acceptance baselines apply here.

Whether the peg actually fits the pocket is deliberately *not* tested here.
That measurement lives in ``scripts/verify_peg_passability.py``, so a motion
sequence never enters the env's semantics.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from . import proxytask_tasks_cfg
from .proxytask_tasks_cfg import FixedAssetCfg, HeldAssetCfg
from .ur10e_cfg import UR10E_HOME_CFG


def resolve_fixture_usd_path() -> str:
    """Return the path to the square-pocket Tisch USD, or raise.

    Called from ``_setup_scene``, not at module import, so ``list_envs.py``
    keeps working on a machine without the gitignored asset.

    The name carries the pocket size (``tisch_square_b45.usd``) and the
    resolver asks for the size configured for THIS run -- name is contract
    (D-033). A size-less ``tisch_square.usd`` is deliberately not accepted:
    since the pocket became the curriculum variable, a file under that name
    could hold any rung's geometry, and loading it would mean training one
    rung while the metrics, the run tag and the yaw window all describe
    another. That is the stale-asset failure mode that motivated the startup
    asset check (D-028 context) -- everything runs, and every result is void.

    The round-bore ``tisch.usd`` is not a fallback either, for the same reason
    plus a second one: the square assets carry a different origin convention
    (origin on the opening plane at the pocket centre, not at the plate
    underside), so the two files are not even placed the same way.
    """
    env_override = os.environ.get("PROXYTASK_TISCH_USD")
    candidates = []
    if env_override:
        candidates.append(env_override)
    asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Tisch")
    candidates.append(os.path.join(asset_dir, proxytask_tasks_cfg.default_fixture_usd_name()))
    for path in candidates:
        if os.path.isfile(path):
            return path
    pocket_mm = proxytask_tasks_cfg.POCKET_SIDE * 1000.0
    raise FileNotFoundError(
        "Square-pocket Tisch USD not found. Tried: "
        + "; ".join(candidates)
        + f". Generate it with: python scripts/author_tisch_square.py --pocket-side-mm {pocket_mm:g} "
        "(or set PROXYTASK_TISCH_USD to a table whose pocket really is "
        f"{pocket_mm:g} mm). A size-less tisch_square.usd and the round tisch.usd are NOT "
        "accepted; the pocket size is part of the filename because it is the curriculum "
        "variable (D-033)."
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
    #   this branch). observation_space is the square-pivot layout (D-029)
    #   extended by the pocket orientation (D-037): 0:6 joint_pos, 6:12
    #   joint_vel (finite-difference, not the raw solver channel -- see D-030
    #   addendum 2), 12:15 peg-tip-relative-to-pocket-entrance (pocket frame
    #   under tilt), 15:19 EE quat (wxyz, env frame), 19:21 (cos 4 phi,
    #   sin 4 phi) of the peg yaw -- C4-invariant, so the four insertable
    #   orientations are one point in observation space -- and 21:25 the
    #   POCKET quaternion in the env frame (wxyz, sign canonicalised to
    #   w >= 0). No wrench/contact-sensor channels on this branch.
    #
    #   Channels 21:25 are what makes a randomly oriented pocket learnable at
    #   all: everything before them is either robot state or pocket-RELATIVE,
    #   so two episodes whose pockets are tilted differently look identical
    #   while needing different joint motions. The quaternion (rather than a
    #   6D rotation representation) follows GenPiH, which trains exactly this
    #   task on a UR10e in Isaac Lab with the hole orientation as a quaternion
    #   over RPY +-25 deg; the discontinuity result of Zhou et al. that argues
    #   for 6D concerns rotations produced as network OUTPUTS, and the q ~ -q
    #   ambiguity that remains for an INPUT is removed by canonicalising the
    #   sign. See D-037. Appended at the END so that no existing channel index
    #   moves and a checkpoint trained on 21 channels can be zero-padded into
    #   this layout.
    action_space = 6
    observation_space = 25
    state_space = 0

    # Action/reward knobs, carried from the demo sprint where noted.
    action_scale = 0.02  # rad of joint-target delta per unit action, per step
    # Reset noise, per group: reset_joint_noise on five joints,
    # reset_yaw_noise on yaw_joint_name alone. +-45 deg (pi/4) on wrist_3
    # spans a full C4 fundamental domain of start orientations; without it
    # every episode would start inside the +-3.96 deg free-yaw window, the
    # policy would never need to turn, and a 100 % success rate would say
    # nothing about the square task. wrist_3 is a pure tool-axis yaw actuator
    # at the home pose, so the wide noise moves phi and nothing else.
    reset_joint_noise = 0.01  # rad, uniform +-, on the five non-yaw joints
    reset_yaw_noise = 0.7854  # rad (+-45 deg), uniform, on the yaw joint
    yaw_joint_name = "wrist_3_joint"
    reward_w_approach = 2.0
    reward_w_depth = 100.0
    reward_w_success = 10.0
    reward_w_action = 0.01
    # Charged per metre of depth below the plate top that is not inside the
    # pocket. At the ~55 mm read under the plate this costs 1.1 per step against
    # an approach term of ~0.13 there, so the space under the table is clearly
    # worse than the home pose (~0.17) rather than better, which is what it
    # was before 2026-07-27.
    reward_w_misplaced = 20.0
    # Charged for the peg's tool axis departing from world-down: 0 when it
    # points straight into the plate, this value when it lies flat. Added
    # 2026-07-27 after every environment converged on reaching the opening
    # sideways -- a distance-only approach term is maximised from any
    # direction, so that pose scored about zero, the best value reachable
    # without inserting, and nothing pulled the arm out of it.
    reward_w_align = 2.0
    # Charged (1 - cos 4 phi) / 2 for the peg yaw leaving the four insertable
    # orientations: 0 there, this value at 45 deg. Derived, not guessed:
    # unwinding a 45 deg yaw error over ~40 steps yields about 0.04 * w_yaw
    # per step, against a worst-case action cost of ~0.01 per step -- parity
    # at w_yaw = 0.25, so 2.0 carries a safety factor of 8 while staying an
    # order of magnitude under the depth term. Closes the yaw-parking
    # attractor (vertical, centred, 45 deg turned: approach ~0, align ~0,
    # insertion geometrically impossible).
    reward_w_yaw = 2.0
    # Minimum tool-axis alignment before depth counts at all. 0.99 is 8.1
    # degrees; the square geometry allows at most ~2.3 degrees of tilt inside
    # 1.0 mm of per-axis clearance over the 50 mm peg, so this is a loose
    # anti-exploit gate on physical impossibility, not a style preference --
    # the four-corner containment test does the exact geometric gating. The
    # home pose reads 0.9998.
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

    # fixture: KINEMATIC rigid object since D-034 (supersedes D-018's static
    # collider). D-035 measured that a policy trained at one fixed fixture pose
    # presses the peg onto the trained location under a 2 cm fixture offset
    # (0 % success), so the pose must vary per episode -- and a static collider
    # has no runtime pose to write. Kinematic keeps the body immovable by
    # contact forces (PhysX drives it, the arm cannot push it away) while its
    # root pose becomes writable at reset, which is the Factory pattern for the
    # fixed asset. usd_path is resolved in _setup_scene via
    # resolve_fixture_usd_path(); the spawn here is a placeholder.
    # No rigid_props in the spawn cfg ON PURPOSE: the spawner's rigid_props
    # path is modify-only and a silent no-op on the API-less generated table
    # (it printed a "Could not perform 'modify_rigid_body_properties'" warning
    # on the training machine, 2026-08-06). The RigidBodyAPI incl. kinematic
    # flag is applied explicitly in _setup_scene via
    # define_rigid_body_properties, which is the call that actually works.
    fixture_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Fixture",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=proxytask_tasks_cfg.FIXTURE_POS),
    )

    # Per-episode fixture-pose randomisation (D-034). Uniform +- range in
    # metres applied to x and y of the fixture (and with it the pocket
    # opening) at every reset; z stays fixed -- the table height is a known
    # quantity in the real task. 0.0 reproduces the pre-D-034 behaviour
    # exactly (no pose write happens at all), which is the regression case the
    # commissioning chain replays the b = 32 checkpoint against. Factory's
    # PegInsert trains at fixed_asset_init_pos_noise = 0.05 m per axis
    # (isaaclab_tasks v2.3.2, factory_tasks_cfg.py), which is the target range
    # here. The trained default is 0 so that a run must OPT IN via hydra
    # (env.fixture_pos_noise_xy=0.05) and the run tag records it (train.py).
    fixture_pos_noise_xy = 0.0
    # Pocket-yaw randomisation (D-038): per-episode rotation about the env z
    # axis through the pocket centre, uniform in +-this range. Since D-037 the
    # per-env fixture quaternion carries the FULL orientation, so yaw rides the
    # same machinery as tilt: the reset write, the pocket-frame measurement
    # (phi and the corner test are evaluated in the yawed frame) and
    # observation channels 21:25. The square pocket repeats every 90 deg (C4),
    # so +-pi/4 (0.7854) is the largest distinct range that exists. The
    # trained default is 0 so that a run must OPT IN via hydra
    # (env.fixture_yaw_noise_rad=0.7854) and the run tag records it (train.py).
    fixture_yaw_noise_rad = 0.0
    # Static pocket tilt (D-036): rotation of the WHOLE fixture about the env
    # y axis through the pocket centre (the asset origin), in radians.
    # Positive tips the +x plate side down. Applied at SPAWN time and held for
    # the whole run in EVERY env -- this is the deterministic PROBE, set via
    # play.py --fixture-tilt, and it is what the 100 % / 50 % baseline of
    # D-036 was measured with. Training randomisation lives in the field
    # below; setting both at once is rejected in __init__, since one run
    # cannot be both a deterministic probe and a randomised curriculum rung.
    fixture_tilt_rad = 0.0
    # Static pocket yaw (D-038): one fixed rotation about the env z axis
    # through the pocket centre, every env, whole run -- the deterministic
    # PROBE, set via play.py --fixture-yaw. Composes with fixture_tilt_rad as
    # R_z(yaw) @ R_y(tilt), the same order the randomised reset uses. Mixing
    # any deterministic angle with any noise range is rejected in __init__.
    fixture_yaw_rad = 0.0
    # Per-episode pocket tilt (D-037), in radians: the MAXIMUM tilt magnitude.
    # At every reset each env draws a magnitude uniformly from [0, this] and a
    # tilt DIRECTION uniformly over 360 deg in the env xy plane, so the pocket
    # tips in an arbitrary direction rather than about y only. The resulting
    # orientation is written to the kinematic fixture, drives the pocket-frame
    # measurement (per-env rotation matrices) and is published to the policy
    # as observation channels 21:25 -- without that observation two episodes
    # at different angles would be indistinguishable to the policy while
    # requiring different motions, and PPO could only learn their average.
    # 0.0 reproduces the pre-D-037 behaviour exactly (no per-env sampling, no
    # pose write from this term). The trained default is 0 so that a run must
    # OPT IN via hydra (env.fixture_tilt_noise_rad=0.0873 for 5 deg) and the
    # run tag records it. Ladder of D-037: 5 deg, then 15 deg.
    fixture_tilt_noise_rad = 0.0

    # Task geometry (D-022, D-023, D-029). Defined once in
    # proxytask_tasks_cfg.py and mirrored here as cfg fields, so the robot
    # base height is derived from the plate top rather than repeated as a
    # matching literal. Note the square re-import's origin convention:
    # fixture_pos and opening_entrance_pos are the SAME point now (asset
    # origin on the opening plane at the pocket centre).
    fixture_pos = proxytask_tasks_cfg.FIXTURE_POS
    plate_top_z = proxytask_tasks_cfg.PLATE_TOP_Z
    plate_bottom_z = proxytask_tasks_cfg.PLATE_BOTTOM_Z
    plate_half_extents = proxytask_tasks_cfg.PLATE_HALF_EXTENTS
    opening_entrance_pos = proxytask_tasks_cfg.OPENING_ENTRANCE_POS
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
