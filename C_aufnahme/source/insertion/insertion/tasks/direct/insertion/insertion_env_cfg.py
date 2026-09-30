# NOTE (Phase B copy, 2026-08-18): copied from the verified proxy task
# (square peg, old repo). Geometry constants and square-peg-specific
# derivations (C4 yaw window, per-axis clearance, analytic inertias) are
# PROXY-SPECIFIC and get replaced in later phases from measured real-task
# values. Do not treat any number in this file as a real-task value yet.
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene cfg: static Tisch fixture + UR10e with the welded square peg, home pose.

SQUARE-PEG PIVOT (branch square-peg-insertion, 2026-07-28, D-029, UNVERIFIED):
cut from the demo sprint (a7c7ebb), keeping its joint-delta actions and dense
reward. The peg is a 30 x 30 x 50 mm square prism, the hole a 35 mm deep blind
pocket in a generated table (D-033: scripts/author_tisch_square.py, deleted
2026-08-28 with the dead proxy code, S6) whose asset origin sits ON the
opening plane at the pocket centre. The pocket side length
is the curriculum variable and therefore part of the asset filename, so the
resolver below asks for one specific size rather than for "the square table".
The observation is 25-dim (the demo layout plus
(cos 4 phi, sin 4 phi) of the peg yaw and the pocket quaternion of D-037), the
reward gains a C4-invariant yaw
term, and reset noise is per-joint: +-45 deg on wrist_3, +-0.01 rad elsewhere.
None of the round-peg acceptance baselines apply here.

Whether the peg actually fits the pocket is deliberately *not* tested here.
That measurement lived in ``scripts/verify_peg_passability.py``, so a motion
sequence never entered the env's semantics. That script was deleted on
2026-08-28 (S6) together with the proxy geometry it measured; the real task's
equivalent question is answered by the D-108 scripted-insertion gate
(``scripts/scripted_insert.py``).
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from . import insertion_math, insertion_tasks_cfg
from .agents.rsl_rl_ppo_cfg import PPORunnerCfg
from .insertion_tasks_cfg import FixedAssetCfg, HeldAssetCfg
from .ur5e_cfg import UR5E_HOME_CFG


def resolve_fixture_usd_path() -> str:
    """Return the path to the fixture USD the ENV should spawn, or raise.

    Since 2026-08-24 this returns the REAL CAD CUT-OUT, not the generated block.
    That is the whole point of the scene rebuild: the thing the part is inserted
    into is the user's measured pocket with its chamfer, its lugs and its
    stage-1/stage-2 step, and it is the thing whose pose D-034 randomises.

    History, so the two earlier meanings of "fixture" are not confused with each
    other: it was the proxy square Tisch until 2026-08-18, then the generated
    WORKCELL BLOCK (one solid box, no pocket), and now the CAD asset. The block
    did not disappear -- it became scenery and is spawned separately in
    ``_setup_scene`` via ``resolve_workcell_block_usd_path()``.

    The function is kept rather than replaced at the call sites because
    ``scripts/verify_fixture_spawn.py`` imports it from here.

    The proxy resolver survives as ``resolve_proxy_tisch_usd_path`` below --
    unused by the env, kept because its docstring records why the size is in
    the filename, which the real fixture will need again once the pocket
    becomes a curriculum variable.
    """
    return insertion_tasks_cfg.resolve_pocket_usd_path()


def resolve_proxy_tisch_usd_path() -> str:
    """Return the path to the square-pocket Tisch USD, or raise. PROXY ONLY.

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
    env_override = os.environ.get("INSERTION_TISCH_USD")
    candidates = []
    if env_override:
        candidates.append(env_override)
    asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Tisch")
    candidates.append(os.path.join(asset_dir, insertion_tasks_cfg.default_fixture_usd_name()))
    for path in candidates:
        if os.path.isfile(path):
            return path
    pocket_mm = insertion_tasks_cfg.POCKET_SIDE * 1000.0
    raise FileNotFoundError(
        "Square-pocket Tisch USD not found. Tried: "
        + "; ".join(candidates)
        + ". Its generator (scripts/author_tisch_square.py) was DELETED on "
        "2026-08-28 with the dead proxy code (S6), so the table cannot be "
        "rebuilt on this branch -- recover the script from git history, or set "
        f"INSERTION_TISCH_USD to a table whose pocket really is {pocket_mm:g} mm. "
        "A size-less tisch_square.usd and the round tisch.usd are NOT "
        "accepted; the pocket size is part of the filename because it is the curriculum "
        "variable (D-033)."
    )


@configclass
class InsertionEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2  # 60 Hz control rate (D-024)
    # D-113 (3): 256 steps at the 60 Hz control rate. IndustReal's value; the
    # proxy's 240 steps (4.0 s) sat next to it and is what stood here until
    # 2026-08-28, together with a justification that argued from the PROXY
    # travel ("100 mm standoff plus 25 mm of insertion").
    #
    # WRITTEN AS A RATIO ON PURPOSE. Isaac Lab derives the step count as
    # ceil(episode_length_s / (sim.dt * decimation)) (direct_rl_env.py:286),
    # and the rounded decimal 4.2667 yields 257 steps, not 256. Verified
    # numerically on the dev laptop, 2026-08-28. A check pins both the count
    # and the ratio form; see check_env_wiring.py.
    #
    # CHECK OBLIGATION, D-113 (3): the 256 is ADOPTED, not derived -- the
    # start height over the opening was [CAD pending] when it was decided.
    # Re-measure it against the scripted-insertion gate and correct HERE if
    # the travel needs more steps.
    episode_length_s = 256 / 60
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
    #
    # [REPLACED 2026-08-28, M2.4.] The paragraph above describes the PROXY
    # layout and is kept only because the channel INDICES it explains are
    # still the indices in force. What the env now builds is the real task's
    # layout, whose one home is ``insertion_math.OBS_SLICES``: 19:21 carries
    # (cos phi, sin phi), NOT (cos 4 phi, sin 4 phi) -- the real part fits its
    # pocket in exactly one rotational position (D-107 point (2)) -- and 12:15
    # is measured from the PART's own task frame, not the flange (D-107 point
    # (4)). Read the layout there; do not re-derive it here.
    #
    # WHY 28. It was 25 until 2026-08-28: the last block of ``OBS_SLICES`` is
    # the three force components D-114 adds at 25:28, and that block was held
    # back because nobody knew WHERE to read the wrench. RT-59 measured it --
    # ``tool_link``, ratio 1.000 against its own weight -- so the block is
    # built and the width is the FULL table width now. The number below is not
    # "28" as a value but ``insertion_math.OBS_DIM``, and
    # ``scripts/check_env_wiring.py`` pins it to that so the two cannot drift
    # apart.
    action_space = 6
    # THE FORCE LAYOUT'S WIDTH, read from its one home; ``resolve_obs_layout``
    # overwrites it with ``obs_dim(obs_wrench_mode)`` before the base class
    # reads it (D-188), so under ``wrench`` this reads 31 at run time.
    observation_space = insertion_math.OBS_DIM
    state_space = 0

    # Action/reward knobs, carried from the demo sprint where noted.
    action_scale = 0.02  # rad of joint-target delta per unit action, per step
    # ------------------------------------------------------- CONTROL MODE
    # Inbox entry "Audit 2026-09-03 (a): D-108 wird neu geoeffnet", Decision
    # (1)-(5), user 2026-09-03. Two modes, one switch, no third:
    #
    #   "osc"       the DECIDED controller. Action = 6-D pose delta (metres,
    #               axis-angle rad) in the env frame; target = current pose +
    #               clamped delta, re-anchored EVERY physics step (Factory
    #               form, factory_env.py:268); tracked by Isaac Lab's
    #               OperationalSpaceController in effort mode with gravity
    #               compensation, gravity ON. The PhysX drives must be inert
    #               for that: stiffness and damping of all three actuator
    #               blocks go to 0 at spawn (run_osc.py:101-104, Factory
    #               factory_env_cfg.py:160-172); maxForce, limits, armature
    #               and friction stay USD-authored. That breaks the LETTER of
    #               D-105 for two values -- SUPERVISOR QUESTION OPEN
    #               (Decision (2)); nothing in the concept is fixed until it
    #               is answered. D-105 itself is not edited.
    #   "joint_pd"  the D-108 controller this replaces: joint-target
    #               integrator on the USD drives. Kept ONLY as the PD half
    #               of the discriminating measurement (RT-129 form, once per
    #               mode) and as the reference every earlier run was made
    #               under. Not a training option.
    #
    # Hydra per run:  env.control_mode=joint_pd
    control_mode: str = "osc"
    # ------------------------------------------------------- OBSERVATION MODE
    # D-188 (2026-09-13). Two layouts, one switch, appended not moved:
    #
    #   "force"     the D-114 layout, 28 channels -- the default every run
    #               before D-188 trained under. The wrench's three TORQUE
    #               components are read and tared in this mode too (log,
    #               trace truth block), but NOT handed to the policy.
    #   "wrench"    the same 28 plus `torque` 28:31: the gravity-tared,
    #               EMA-smoothed torque of the SAME joint wrench, same
    #               frame, about the PARENT link origin (the Isaac test's
    #               assumption; RT-192 measures it). Unscaled -- rsl_rl's
    #               actor_obs_normalization owns the scaling.
    #
    # One home for the derivation: insertion_math.obs_slices / obs_dim;
    # ``resolve_obs_layout`` below writes ``observation_space`` from it
    # BEFORE the base class reads the cfg. A checkpoint of the other width
    # is refused with a readable line (scripts/tools/checkpoint_width.py).
    # Hydra per run:  env.obs_wrench_mode=wrench
    obs_wrench_mode: str = "force"
    # Task-space stiffness (N/m, Nm/rad) and the critical damping ratio.
    # kp position is NOT adopted from anywhere: Decision (4) DERIVES it,
    #     kp <= F_search / (Lambda_max * step_limit),
    # from F_search = 20 N (literature search band), Lambda_max = the
    # task-space inertia the startup report prints, and the step limit below.
    # Until Lambda is measured the field holds a labelled placeholder and the
    # measurement runs it at 100 AND 500 (two runs, user 2026-09-03). The
    # "100" precedents (OSC cfg default, reach example) are contact-free and
    # do NOT count as a source. Rotation stiffness has no source of ours:
    # Factory's 30 as a labelled placeholder. Damping: critical, 2*sqrt(kp),
    # the identical rule in Factory (factory_utils.py:21) and in the OSC
    # (operational_space.py:98-101, ratio 1.0) -- a rule, not a number.
    # Hydra per run:  env.osc_kp_pos=500
    osc_kp_pos: float = 100.0
    osc_kp_rot: float = 30.0
    osc_damping_ratio: float = 1.0
    # Per-step limits on the pose delta: the action in [-1, 1] is scaled by
    # these. Decision (4): "policy rate x allowed tool speed" -- the
    # tool-speed source is MISSING, so both are Factory's
    # (pos_action_threshold / rot_action_threshold, factory_env_cfg.py:53)
    # as labelled placeholders.
    osc_pos_step_limit_m: float = 0.02
    osc_rot_step_limit_rad: float = 0.097
    # The box the LEADING TOOL POINT may not leave, half-width around this
    # episode's pocket entrance. ONE HALF-WIDTH PER AXIS since Phase 5 step
    # B7; the box moves WITH the entrance, so fixture_pos_noise_xy does not
    # eat into either of them.
    #
    # BOTH HALF-WIDTHS ARE SET BY D-180 from the DISK formula plus set air, at
    # the D-178/D-179 ceilings (radius, tilt, start height) and the step limit
    # above. Neither is Factory's number any more; the required values and
    # the air are written in D-180, not here. Yaw drops out of both reaches
    # (D-178, Rationale).
    #
    # x and y: the disk reach r + h*sin(tilt) plus one action step, plus air
    # (D-180 (2)). Until D-180 this was Factory's pos_action_bounds, 0.05 m
    # (factory_env_cfg.py:53, CtrlCfg); the disk reach at the D-178/D-179
    # ceilings does not fit in that. TWO RULES in check_env_wiring guard it:
    # sufficiency (at least reach + step, from autodr.bounds_max() in the
    # WORLD frame the box is aligned to) and a ceiling with SET air (at most
    # reach + step + XY_CLAMP_AIR_MAX_M, D-180 (5)).
    osc_pos_clamp_m: float = 0.08
    # z: the disk reach h*cos(tilt) + r*sin(tilt) plus one action step, plus
    # air (D-180 (3)).
    #
    # TWO SEPARATE RULES GUARD IT, both in check_env_wiring, neither one a
    # restatement of this number:
    #   sufficiency -- the box must hold the outermost legal start plus one
    #     full action step, measured in WORLD z (the box is world-axis
    #     aligned; start_height is a POCKET-frame quantity, so the tilt and
    #     the lateral band both enter). That requirement comes from
    #     autodr.bounds_max() and moves when the band moves.
    #   upper -- the box must stay BELOW the home tip standoff
    #     (WORKCELL_HOME_TIP_ABOVE_ENTRANCE, and the RT-143 measurement in
    #     check_env_wiring). A box that swallows the home pose ends the reset
    #     pull-down RT-143 measured, and it is what catches a
    #     metre/millimetre slip that keeps the field name.
    #
    # Factory has no rule to borrow. Its own pos_action_bounds does not
    # contain the reset band of peg_insert -- the task this one is a variant
    # of (factory_tasks_cfg.py:112-113, hand_init_pos 0.047 + noise 0.010
    # against a bound of 0.05). Its clamp is not even per-axis:
    # factory_env.py:274 clips all three components with one scalar pair.
    osc_pos_clamp_z_m: float = 0.1435
    # The cone the tool axis may not leave, half-angle from the vertical.
    # DERIVED from CAD, not a placeholder: 8.52 deg is the maximum tilt with
    # only the front edge in the pocket (Geometrie_Fuegeteil_Aufnahme.tex:264,
    # "Einfaedeln"). Decision (3). Yaw stays free.
    osc_tilt_clamp_rad: float = math.radians(8.52)
    # Policy rate under OSC: Factory's decimation 8 = 15 Hz at dt 1/120 as a
    # labelled placeholder (Decision (4)); the controller itself runs every
    # physics step. ``decimation = 2`` above stays D-024's 60 Hz and is what
    # joint_pd runs at. ``resolve_control_mode`` below writes this into
    # ``decimation`` when the mode is osc and keeps the EPISODE STEP COUNT
    # (``episode_steps``, D-113 (3): 256 control steps) -- so the episode
    # becomes 256/15 = 17.07 s of sim time. That re-derivation is OWED
    # (audit section 6 (5)) and not done here.
    osc_decimation: int = 8
    # Reset noise, per group: reset_joint_noise on five joints,
    # reset_yaw_noise on yaw_joint_name alone. +-45 deg (pi/4) on wrist_3
    # spans a full C4 fundamental domain of start orientations; without it
    # every episode would start inside the +-3.96 deg free-yaw window, the
    # policy would never need to turn, and a 100 % success rate would say
    # nothing about the square task. wrist_3 is a pure tool-axis yaw actuator
    # at the home pose, so the wide noise moves phi and nothing else.
    #
    # BOTH ARE TEMPORARILY 0 since 2026-08-23, for the scene rebuild.
    # SUPERSEDED PLAN NOTE (2026-08-27): the old restore values 0.01 / 0.7854
    # are the PROXY's -- the 0.7854 (+-45 deg) comes from the C4 symmetry of
    # the square peg, which the one-orientation real part does not have.
    # D-110 (5) rejects adopting them; the real first-rung ranges are set by
    # the curriculum build (stream p2-rl-code) after the scripted-insertion
    # gate, small and symmetric around the target pose. Do NOT write 0.01 /
    # 0.7854 back for training. Hydra overrides per run still work:
    #     env.reset_joint_noise=<x> env.reset_yaw_noise=<y>
    # The config itself has to hold the zeros because zero_agent.py does NOT
    # take hydra overrides -- it is plain argparse plus parse_env_cfg.
    #
    # Why off: while the cell is still being built, the noise makes every run
    # unrepeatable and every deviation ambiguous. RT-17 cost two rounds of
    # arithmetic to separate noise from geometry in a flange standoff that read
    # 0.309950 m against 0.317000 -- and the droop half of that gap has since
    # gone away with the USD drives (RT-46: 0.317026 m). With both noises at 0
    # the four envs are identical, so any difference BETWEEN them is a real
    # finding.
    #
    # Why it must go back on before training: see the paragraph above. At
    # yaw noise 0 every episode starts inside the +-3.96 deg free-yaw window
    # and a 100 % success rate would be a non-result (D-038). Nothing in the
    # code guards this -- the reminder lives in HANDOFF-SZENE.md.
    #
    # RUNG 0, SETTLED 2026-08-30 (user): BOTH STAY 0.0. The start pose is not
    # scattered at all yet -- rung 0 varies the FIXTURE in x and y and nothing
    # else. The user watched a run and asked for the start noise off ("das
    # stellen wir spaeter ein, erstmal nur x und y versatz"). One thing moves
    # at a time.
    #
    # This block previously carried 0.005 / 0.02 for a few hours on
    # 2026-08-30. Those values are SUPERSEDED, and the reasoning below is kept
    # ONLY because it is the derivation the next rung argues against -- it is
    # not a description of what runs now.
    #
    # WHEN reset_joint_noise GOES BACK ON, 0.005 rad (+-0.29 deg) was the
    # derived ceiling, and THE DERIVATION IS DEAD as of 2026-09-01. It anchored
    # on the coarse kernel margin as "the start scatter" -- a reading the
    # concept stream has since refuted: dm_control's margin is the decay REACH,
    # not a tolerance, and KERNEL_MARGIN_COARSE now carries the measured travel
    # (173.21 mm, inbox entry "Reward-Ueberarbeitung" 2026-09-01,
    # p1-konzept-messung). The forward-kinematics numbers stay true as
    # MEASUREMENTS -- 0.005 rad is a lateral part-tip scatter of p95 3.43 mm,
    # max 4.65 mm, and 0.010 rad is p95 6.86 mm -- but no kernel width bounds
    # them any more. A start-scatter ceiling has to be argued from the task
    # (reachability of the pocket at the rung's start pose), not from the
    # reward, and nobody has argued it yet.
    #
    # WHEN reset_yaw_noise GOES BACK ON, the FLOOR is the real part's free-yaw
    # window, 0.0041 rad (0.235 deg), computed from PLAY_X and the part
    # length -- the angle at which the part still drops in without being
    # turned. D-038: a scatter inside that window makes a 100 % success rate a
    # non-result. AT 0.0 THAT IS EXACTLY THE STATE WE ARE IN, deliberately and
    # temporarily: this rung's success rate says nothing about yaw and must
    # not be reported as if it did. Nothing in the code guards this; the
    # reminder is here and in the handoff.
    #
    # zero_agent.py does NOT take hydra overrides (plain argparse plus
    # parse_env_cfg), so the config itself has to hold the zeros. A training
    # run can still opt in per run:
    #     env.reset_joint_noise=<x> env.reset_yaw_noise=<y>
    reset_joint_noise = 0.0  # rad, uniform +-, on the five non-yaw joints
    reset_yaw_noise = 0.0  # rad, uniform, on the yaw joint

    # ---------------------------------------------------------------- RUNG 0
    # Where the leading tool point starts, as a HEIGHT above the stage-2
    # opening plane (positive up), NOT as a joint table. ``None`` keeps the
    # bare home pose, which is the pre-D-161 behaviour and the only thing the
    # measurement scripts want.
    #
    # D-161 (RT-119, measured): at the home pose (+0.165 m) the whole D-109
    # reward is 3.05e-23. The kernel widths stay untouched -- the START moves.
    # The default is one coarse kernel margin above the seat; the constant
    # owns the derivation and the sign convention.
    #
    # TWO READINGS, by dr_mode (D-179 (3)):
    #   dr_mode='off'    -- the UPPER edge of the start band
    #                       [start_tip_above_entrance_low, this] (D-163, step C).
    #   dr_mode='autodr' -- the CENTRE of the one-sided start height, i.e. its
    #                       lower edge H_min: the lowest clean start, MEASURED
    #                       later (D-179 (4)); the ceiling is in
    #                       autodr.DR_DIMS. This field is H_min's one home.
    # [TESTWERT] in the autodr reading until that measurement: the value
    # below is the rung-0 height, not a measured H_min.
    #
    # Hydra per run:  env.start_tip_above_entrance=<metres>  (or =null)
    start_tip_above_entrance: float | None = (
        insertion_tasks_cfg.RUNG0_START_TIP_ABOVE_ENTRANCE
    )
    # START-HEIGHT SAMPLING (SBC, IndustReal sec. IV.G; plan step C,
    # 2026-09-02). Each resetting env draws its start height uniformly from
    # [low, start_tip_above_entrance]. The DEFAULT is low == high, i.e. a
    # zero-width range, which is bit-identical to the fixed start (rand * 0).
    #
    # WHY NOT ``None`` AS THE OFF VALUE: Isaac Lab's hydra merge
    # (``isaaclab/utils/dict.py`` ``update_class_from_dict``, step 5) refuses
    # a float over a ``None`` default -- "Incorrect type ... Expected
    # NoneType, Received float". RT-138's first attempt died on exactly that
    # (PROBLEMS.md 2026-09-02). A float default can be overridden by a float
    # AND by ``null``; a ``None`` default can be overridden by nothing.
    # ``start_tip_above_entrance`` above works for the same reason: its
    # default is the float constant.
    #
    # WHY: RT-134 / RT-137 measured the fixed +30 mm start parking the part
    # on the fixture's outer rim at depth 0; RT-135 measured the reward
    # landscape flat across the lateral error (5.82e-5 per step over
    # 4.88 mm). IndustReal reports the single fixed start above the hole as
    # the "naive curriculum" that FAILED and samples Uniform[z_low, z_high]
    # instead; the D-165 progress term pays only once a corner is in, so a
    # start distribution that sometimes begins inside is what lets it pay.
    # This OVERRIDES D-163's task argument for those episodes (a start
    # inside the pocket skips the tilted approach) -- recorded in
    # docs/decisions_inbox.md, entry "Step C ..." (2026-09-02), user decision.
    #
    # The value of ``low`` and its schedule are NOT decided; the field ships
    # equal to the upper bound (fixed start) and a run opts in per hydra.
    # With ``start_tip_above_entrance`` = None (home pose, the measurement
    # scripts) the lower bound is ignored -- there is no solver to read it.
    # Otherwise the env refuses (a) ``low`` above ``start_tip_above_entrance``,
    # (b) a ``low`` at or below ``-depth_min`` (a start inside the 33-36 mm
    # success band is a one-step success with the full lump), and (c) a
    # negative ``low`` together with ANY orientation noise or static tilt/yaw
    # -- the start IK commands position only, an un-commanded orientation
    # inside the pocket is a wall contact at reset (RT-120). A ``null``
    # override is read as "fixed start" too.
    #
    # Hydra per run:  env.start_tip_above_entrance_low=<metres>  (e.g. -0.030)
    start_tip_above_entrance_low: float | None = (
        insertion_tasks_cfg.RUNG0_START_TIP_ABOVE_ENTRANCE
    )
    # THE START FLOOR (SBC step 0, IndustReal sec. IV.G `z_low` raised with
    # success; Pläne/SBC_Schritt0_Entwurf.md, user decisions 2026-09-13/14).
    # Metres, the LOWEST start of the leading tool point at the beginning of
    # a run. Every run with a floor begins with the start height drawn
    # uniformly from [floor, start_tip_above_entrance]; the floor is an
    # eighth AutoDR boundary (`autodr.FLOOR_KEY`) that moves UP one rung per
    # full buffer at the 0.80/0.10 rule, until it stands on
    # start_tip_above_entrance (= H_min, 0.020 m measured RT-197a/b). Only
    # then do the other boundaries open (dr_mode='autodr') or, under
    # dr_mode='off', the start-height ceiling alone (up to autodr.DR_DIMS
    # start_height hi_max) -- the No-DR branch keeps the same curriculum so
    # the two conditions stay comparable (user, 2026-09-14).
    #
    # OFF VALUE: equal to start_tip_above_entrance (the default) -- a floor
    # AT H_min is no floor, and the env then builds exactly the pre-floor
    # provider (or none). NOT ``None``: Isaac Lab's hydra merge refuses a
    # float over a ``None`` default (RT-138, PROBLEMS.md 2026-09-02), the
    # same reason ``start_tip_above_entrance_low`` above is a float.
    #
    # REFUSED (insertion_env.__init__): a floor above start_tip_above_entrance;
    # at or below -depth_min (a start inside the success band); together
    # with a start_tip_above_entrance_low that differs from the high (two
    # lower edges for one band); with reset_joint_noise / reset_yaw_noise;
    # and under dr_mode='off' with any fixture angle (static or noise) or a
    # start_lateral_offset -- those would be LIVE while the start is inside
    # the pocket, which is the RT-120 wall contact. Under 'autodr' the
    # provider itself holds every other quantity on its centre while the
    # floor is below H_min (`autodr.AutoDR.bounds`, phase "floor").
    # fixture_pos_noise_xy stays live, as in RT-191..194.
    #
    # Hydra per run:  env.start_floor_m=-0.030  (user's f0, RT-191's band)
    start_floor_m: float = insertion_tasks_cfg.RUNG0_START_TIP_ABOVE_ENTRANCE
    # Rungs from the floor to H_min. 5 = 10 mm per rung for -0.030 -> 0.020,
    # one rung per edge of the height bins (user, 2026-09-14: ten are too
    # many). `[TESTWERT]` like DELTA_STEPS: decided, not derived.
    # Hydra per run:  env.start_floor_steps=5
    start_floor_steps: int = 5
    # AUTODR STALL REVIEW (user, 2026-09-14). Full boundary buffers IN A ROW
    # without a real move (holds and clamped_zero count, clamped_max does
    # not) after which the `[autodr]` event line carries a STALL note. The
    # note is a review trigger for the reader; the 80/10 rule is untouched.
    # 0 = no note ever. Logged per key as dr/<key>_last_rate and
    # dr/<key>_holds regardless of this value.
    # Hydra per run:  env.autodr_stall_buffers=3
    autodr_stall_buffers: int = 0
    # LATERAL START OFFSET (D-170, 2026-09-06; a DISK since D-178). Metres:
    # the RADIUS of a disk around the pocket axis on which x and y of the
    # start-pose GOAL are drawn, per env at every reset, uniform over the AREA
    # (`insertion_math.disk_offset`, D-178 (1)). 0.0 is the pre-D-170
    # behaviour: the buffer stays zero, and the goal reduces to 0.0 - tip_rel.
    # NOT the word
    # "bit-identical" -- that claim was written here and then FALSIFIED
    # offline over 100008 doubles: 0.0 - x equals -x in every bit EXCEPT
    # x = +0.0, where the old form gave -0.0 and the new one gives +0.0.
    # The two compare equal, carry the same norm and drive the same IK
    # command, so the BEHAVIOUR is unchanged -- the sign bit of an exact
    # zero is not.
    #
    # WHY IT EXISTS. D-034 randomises the fixture pose so the part starts off
    # the pocket. D-162 then added the start-pose solver and aimed it at the
    # pocket AXIS (tip_rel x = y = 0). The solver runs LAST in _reset_idx, so
    # it CANCELS the lateral part of D-034 and of reset_joint_noise: measured
    # on RT-160, start_pose_solve_worst_residual_mm = 0.0354 over 17780
    # episodes. Every episode starts perfectly centred no matter what those
    # two are set to. D-162 decided the HEIGHT and never mentioned the side;
    # the cancellation was never written down. This field is the side.
    #
    # WHAT IT IS FOR. The lateral misalignment the policy must recover from --
    # the part starts over the RIM, not over the opening. Read against the
    # plays: PLAY_X 0.588 mm and PLAY_Y 1.600 mm (RT-160 demo_metrics), so
    # anything above ~1 mm already lands on an edge.
    #
    # CEILING. D-153's shoulder-hole clause applies here exactly as it does to
    # fixture_pos_noise_xy: the part's +y face reaches the mesh hole at
    # HOLE_REACH_OFFSET_Y = 7.8847 mm. Below that the exposure cannot occur.
    # D-178 (5) used to ban any radius above it until the mesh was repaired.
    # LIFTED 2026-09-12 by D-187: RT-183 re-measured the hole (still there,
    # 0.0845 mm2) and RT-184 measured its effect (none the reward can see --
    # 64000 sample points over the part are 0.901 mm apart, so 0.10 points
    # are expected over the hole per pose). A run may draw lat_r up to the
    # 0.030 m ceiling. stage1_lateral_y_mm.over_reach stays worth reading but
    # is NOT a gate: its condition is the z band alone, so above the reach
    # offset it fires on the first step of every episode. The AutoDR ceiling
    # is `lat_r` in autodr.DR_DIMS. A run opts in per hydra.
    #
    # ONE NUMBER, the radius, since D-178 (3). D-176 (Phase 5 step B3) made
    # this an (x, y) half-width pair so the x question could be asked apart
    # from the y question; D-178 (4) WITHDREW that question, because the only
    # lateral readout is a radius. `resolve_start_lateral_offset` refuses a
    # list or a tuple by name.
    #
    # THE DEFAULT STAYS A BARE NUMBER, and that is load-bearing, not laziness.
    # Isaac Lab type-checks a SCALAR override against the type of the DEFAULT
    # (`isaaclab/utils/dict.py`, branch 4 `isinstance(value, type(obj_mem))`,
    # branch 5 raises). So with a TUPLE default `env.start_lateral_offset=0.006`,
    # the command form RT-174 and RT-175 ran, dies at startup with "Incorrect
    # type ... Expected: <class 'tuple'>, Received: <class 'float'>". With 0.0
    # it merges. `resolve_start_lateral_offset` is the one place that reads
    # the value's shape.
    #
    # Hydra per run:  env.start_lateral_offset=<radius in metres>  (e.g. 0.006)
    start_lateral_offset: float = 0.0
    # The reset-time IK that puts the tip there. THE SAME SOLVER AND THE SAME
    # DEFAULTS as the verified teleport in ``scripts/check_seated_success.py``
    # (--solve-steps 120, --solve-tol-mm 0.05, --ik-lambda 0.05), which RT-119
    # drove to eight different heights with a residual below 0.001 mm. They
    # are the solver's own knobs, not task values: nothing in the reward, the
    # observation or the termination reads them.
    start_pose_solve_steps: int = 120
    start_pose_solve_tol_m: float = 0.00005
    start_pose_ik_lambda: float = 0.05
    yaw_joint_name = "wrist_3_joint"
    # The seven proxy reward weights (reward_w_approach/depth/success/action/
    # misplaced/align/yaw) stood here until M2.4b step 3 (2026-08-30). They
    # left WITH the proxy reward function, in the same commit as the
    # termination swap -- removing one side alone would have left the proxy
    # reward paying its success bonus every step of a non-terminating seat.
    # The real reward's numbers are the kernel/bonus block further down.
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
    #
    # Switched from the UR10e to the UR5e with the workcell (scene build,
    # 2026-08-18). The UR5e asset carries NO welded peg, so ``peg_body_name``
    # below resolves to nothing and the env's ``_peg_body_idx is None`` path
    # takes over -- observation and reward run in their null branch. That is
    # the intended intermediate state: the scene stands, the task is not yet
    # solvable.
    #
    # This comment used to end "``ur10e_cfg.py`` stays in the tree as the
    # proxy record". OVERTURNED by the user on 2026-08-28: that file was
    # imported by nothing, and ``ur5e_cfg.py`` is the config this task runs
    # on, so the proxy record was carrying a second robot's numbers for no
    # reader. Both it and ``ROBOT_BASE_POS`` are deleted (S6); git history
    # keeps them.
    robot_cfg: ArticulationCfg = UR5E_HOME_CFG
    ee_body_name = "wrist_3_link"

    # Table plates (D-058). Static scenery, spawned once in _setup_scene
    # before cloning; no RigidObject wrapper, because nothing ever writes
    # their pose. The asset's own origin is the robot foot, so the spawn
    # translation is the environment origin itself.
    workcell_tables_pos = insertion_tasks_cfg.WORKCELL_TABLES_POS

    # The block AROUND the pocket (2026-08-24): four walls, a floor and the rear
    # wall, authored by scripts/author_workcell.py from
    # insertion_tasks_cfg.block_recess_boxes(). Static scenery like the tables --
    # no RigidObject, no pose write. It is asset-local to the POCKET ORIGIN, not
    # to its own centre, so it spawns at the same translation as the fixture and
    # the two need no arithmetic to line up.
    #
    # OFF BY DEFAULT since 2026-08-30, and the reason is the fixture noise the
    # robustness tests need (D-044). The recess is the cut-out's bounding box
    # with ZERO clearance, and only the cut-out has a runtime pose, so any
    # fixture offset drives the moving cut-out into standing walls (D-125).
    # Turning the block OFF removes the coupling instead of solving it: the
    # pocket walls belong to the cut-out itself (-x 39.5, +x 19.9, -y 16.1,
    # +y 27.9 mm, floor -36.0, rim +15.0), so the block contributes NO pocket
    # wall and the free-standing fixture is the complete pocket. Isaac Lab's
    # Factory spawns its fixed asset free-standing and randomises its position
    # and yaw the same way.
    #
    # THE COST, and it is real: the block carries `block_rueckwand`, 100 mm
    # above the rim, which is what made the pocket reachable ONLY FROM THE
    # FRONT. Without it the part may approach from any direction, so the task
    # is EASIER and less like the real cell. Logged in InBachelorErwaehnen.md.
    #
    # Nothing is deleted. Every block constant stays valid (they are geometry,
    # not prim state) and the flag turns the prim back on. D-125 is PARKED, not
    # closed: it governs again the day this is True.
    spawn_workcell_block: bool = False
    workcell_block_pos = insertion_tasks_cfg.WORKCELL_BLOCK_POS
    floor_z = insertion_tasks_cfg.FLOOR_Z

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
        # Must match ``fixture_pos`` below: _setup_scene spawns the prim with
        # that translation, while RigidObject seeds its buffers from this
        # init_state. If the two ever disagree the first reset teleports the
        # block. Both read the same constant so they cannot.
        init_state=RigidObjectCfg.InitialStateCfg(pos=insertion_tasks_cfg.WORKCELL_BLOCK_POS),
    )

    # WHICH BOUNDS PROVIDER DRIVES THE RESET (Phase 5, plan section 10).
    #
    #   "off"    -- no provider. The static fields below are the only source
    #               of randomisation; every run before Phase 5 is this mode
    #               and its behaviour is unchanged.
    #   "autodr" -- `autodr.AutoDR` over the seven boundaries. It STARTS AT
    #               WIDTH 0 and opens during the run, so the once-per-run
    #               branches in the env size themselves from the provider's
    #               MAXIMA (`bounds_max`), never from the width-0 bounds.
    #               `fixture_pos_noise_xy` is the ONE static field they still
    #               read: no boundary tracks it, it is not refused below, and
    #               it moves the fixture on its own.
    #
    # "table" (the evaluation of plan section 3) is NOT accepted yet -- it
    # arrives with `play.py --eval-table` in Phase C. The env refuses it
    # rather than falling back to "off", which would score a policy against
    # a distribution nobody asked for and say nothing.
    #
    # "autodr" and the FIVE BOUNDARY-OWNED static fields are MUTUALLY
    # EXCLUSIVE -- the env refuses that combination: with both live the pocket
    # pose would be drawn from one source and bounded by another. Which five,
    # and why, has one home: `InsertionEnv.__init__` (`_static_dr`). The
    # remaining static fields -- `fixture_pos_noise_xy`, the joint noise, the
    # tilt azimuth -- are NOT in that set and stay live under "autodr".
    #
    # Hydra per run:  env.dr_mode=autodr
    dr_mode: str = "off"

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
    #
    # RUNG 0, SET 2026-08-30 (user decision, D-110 (5)): 0.005, NOT Factory's
    # 0.05. THE CEILING HAS A REASON AND IT IS D-153. That decision keeps the
    # fixture mesh's 1.5 mm shoulder hole and reopens by its own clause the
    # moment this field goes above zero. The part's +y face reaches the hole
    # at a centre offset of HOLE_REACH_OFFSET_Y = 7.8847 mm, and inside
    # stage 1 the part may slide 8.7000 mm, so the exposure is possible in
    # principle. 0.005 m stays BELOW 7.8847 mm on either axis, so it cannot
    # occur -- and _stage1_lateral_stats() now MEASURES that instead of
    # asserting it (demo_metrics.json -> stage1_lateral_y_mm.over_reach).
    # Raising this toward 0.05 is not a knob turn: it needs that measurement
    # read first. It also ran clean in RT-106 at exactly this value.
    fixture_pos_noise_xy = 0.005
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
    # Positive tips the +x plate side down. Since the 2026-08-19 frame
    # rotation env y is the LATERAL axis, so this probe now PITCHES the
    # fixture: +x = the back-wall side dips away from the robot. The code axis
    # is kept on purpose (the D-036 baseline was a proxy measurement and does
    # not carry over anyway); which physical axis the REAL task's probe needs
    # is a concept-block-2 decision, recorded in docs/decisions_inbox.md.
    # Applied at SPAWN time and held for the whole run in EVERY env -- this is
    # the deterministic PROBE, set via play.py --fixture-tilt, and it is what
    # the 100 % / 50 % baseline of D-036 was measured with. Training
    # randomisation lives in the field below; setting both at once is rejected
    # in __init__, since one run cannot be both a deterministic probe and a
    # randomised curriculum rung.
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
    # run tag records it. SUPERSEDED LADDER (D-178): D-037's "5 deg, then
    # 15 deg" is no longer the plan. The Phase-5 tilt ceiling is 10 deg and
    # one-sided; it lives in autodr.DR_DIMS (`tilt`, D-178 (5)), not here.
    # Under `dr_mode='autodr'` THIS field is refused non-zero by the
    # `_static_dr` guard in insertion_env.py, so the tilt then comes from
    # the boundary. The 15 deg belonged to the superseded ladder only.
    fixture_tilt_noise_rad = 0.0

    # Task geometry (D-022, D-023, D-029). Defined once in
    # insertion_tasks_cfg.py and mirrored here as cfg fields, so the robot
    # base height is derived from the plate top rather than repeated as a
    # matching literal. Note the square re-import's origin convention:
    # fixture_pos and opening_entrance_pos are the SAME point now (asset
    # origin on the opening plane at the pocket centre).
    # Repointed at the WORKCELL constants with the scene build (2026-08-18).
    # The field NAMES are unchanged on purpose: the env, the startup report
    # and the verify scripts all read them, and renaming a dozen call sites
    # would be a second change riding along with this one. "plate" now means
    # TABLE 2 -- the surface the fixture stands on -- which is the role the
    # proxy plate played.
    fixture_pos = insertion_tasks_cfg.WORKCELL_BLOCK_POS
    plate_top_z = insertion_tasks_cfg.WORKCELL_PLATE_TOP_Z
    plate_bottom_z = insertion_tasks_cfg.WORKCELL_PLATE_BOTTOM_Z
    plate_half_extents = insertion_tasks_cfg.WORKCELL_PLATE_HALF_EXTENTS
    opening_entrance_pos = insertion_tasks_cfg.WORKCELL_ENTRANCE_POS
    robot_base_pos = insertion_tasks_cfg.WORKCELL_ROBOT_BASE_POS
    home_standoff_z = insertion_tasks_cfg.HOME_STANDOFF_Z

    # Peg (D-005, D-019). The peg is a welded link of the robot articulation,
    # not a separately spawned body, so there is no spawn cfg for it here --
    # only the geometry the env needs in order to report where its tip is.
    # THE TWO DEPTHS ARE OVERRIDDEN HERE, and this is their only override.
    # FixedAssetCfg's own defaults are the square proxy pocket's (35 / 25 mm);
    # the real values are derived in ``insertion_tasks_cfg`` from the measured
    # POCKET_FLOOR_Z, which is defined below that class and therefore cannot
    # be the class default. Overriding at the single instantiation site keeps
    # one home per fact and leaves the proxy constants for the proxy scripts
    # that still import them (S6 deletes both together).
    #   depth         reads the four-corner gate (``_peg_geometry``)
    #   success_depth reads the success predicate (``_get_dones``)
    # NAMED, not numbered, since 2026-08-30: the line numbers here read 837 and
    # 949 and the code had moved to 844 and 956. A pointer that goes stale on
    # every edit above it is worse than no pointer.
    fixed_asset: FixedAssetCfg = FixedAssetCfg(
        depth=insertion_tasks_cfg.POCKET_SEAT_DEPTH,
        success_depth=insertion_tasks_cfg.SEATED_SUCCESS_DEPTH,
    )
    held_asset: HeldAssetCfg = HeldAssetCfg()
    # The welded body is the TOOL now, not a box peg: gripper and part in one
    # link (author_tool_ur5e.py). The three fields below keep their peg names
    # because every reader in the env uses them, but their meaning moved with
    # the asset:
    #   peg_body_name   the link the tool chain hangs on
    #   peg_tip_offset  flange -> the LEADING tool point, i.e. the part
    #                   underside. This is D-069's task frame.
    #   peg_length      the length of that offset, which is what the report
    #                   compares "tip along tool axis" against.
    peg_body_name = insertion_tasks_cfg.TOOL_LINK_NAME
    peg_tip_offset = (0.0, 0.0, insertion_tasks_cfg.FLANGE_TO_PART_BOTTOM)
    peg_length = insertion_tasks_cfg.FLANGE_TO_PART_BOTTOM

    # Measured edges of the part in the imported USD, for the sidecar check.
    part_bbox_m = insertion_tasks_cfg.PART_BBOX_M

    # WEIGHED 2026-08-23 by the user, on a scale, both numbers: the part
    # 0.539 kg and the gripper 0.285 kg WITH the suction cups mounted (user
    # confirmed) -- which is the point, since the cups are missing from the CAD
    # and no density times volume stands in for them. The sum is what
    # author_tool_ur5e.py writes onto the link, so this value and the two
    # --part-mass / --gripper-mass arguments must be kept consistent by hand;
    # the asset check compares them and fails on a mismatch above 1e-5 kg.
    tool_mass_kg: float | None = 0.824

    # The startup report is printed at these ``_obs_calls`` counts. That counter
    # is set to 0 in ``InsertionEnv.__init__`` and only ever incremented in
    # ``_get_observations``, never reset, so it counts the WHOLE RUN -- these
    # are positions in the run, not in an episode. The first one only shows the
    # buffers are filled; the middle two decide whether the pose is held
    # (acceptance criterion 2: unchanged after 5 s).
    #
    # WHERE THE COUNTS ACTUALLY SIT UNDER ``train.py`` (corrected 2026-09-12,
    # critic round 2 (2) -- the earlier text here got this wrong twice).
    # Counts 1 and 2 are the two observation reads that happen BEFORE any step:
    # ``RslRlVecEnvWrapper.__init__``'s own ``env.reset()``
    # (vecenv_wrapper.py:66) and the runner constructor's
    # ``get_observations()`` (on_policy_runner.py:40). So count 2 is the RUNNER
    # CONSTRUCTOR, not a settle offset and not a time. ``learn()`` reads once
    # more before its first step (on_policy_runner.py:72), which makes it
    # count = 3 + control steps: at 60 Hz counts 60 and 180 are 0.95 s and
    # 2.95 s of episode time.
    #
    # THE FOURTH ONE IS THE OBSERVATION-BIAS INSTRUMENT (2026-09-12), and it is
    # DERIVED rather than picked. The per-episode bias is written in the noise
    # model's ``reset()`` and nowhere else, so two printouts with no reset
    # between them are identical whatever the bias does, and "is the pocket
    # bias re-drawn or accumulated" has no instrument. What the fourth count
    # has to be is the first one that lies past EVERY env's first reset:
    #   * Isaac Lab caps an episode at ceil(episode_length_s / (sim.dt *
    #     decimation)) (direct_rl_env.py:286), here ceil((256/60) / (1/60))
    #     = 256 control steps, and the timeout fires at
    #     ``episode_length_buf >= max_episode_length - 1`` (insertion_env.py),
    #     i.e. 255 steps after that counter stood at 0.
    #   * ``train.py:725`` passes ``init_at_random_ep_len=True``, so rsl_rl
    #     draws every env's counter from ``randint(0, 256)`` before the first
    #     step (on_policy_runner.py:66-69). The LATEST first reset therefore
    #     belongs to the env that drew 0: it resets on step 255, which is
    #     count 3 + 255 = 258. The reset runs BEFORE ``_get_observations`` in
    #     the same ``step`` (direct_rl_env.py:398-410), so count 258 already
    #     shows that env's NEW bias -- past every first reset with MARGIN ZERO.
    #   * A resume run (``init_at_random_ep_len=False``, train.py:748/758) puts
    #     every env at 0 and lands on the same 258.
    #
    # WHAT THIS COSTS, AND IT IS THE POINT: 2, 60 and 180 are NOT "identical by
    # construction". Randomised counters mean the envs reset at different
    # steps, and the report prints envs 0..3, so the chance that at least one
    # of those four has already reset by count 60 is 1 - (198/256)^4 = 0.64.
    # A LINE THAT CHANGES BEFORE COUNT 258 IS AN EARLY RESET, NOT A FINDING;
    # the only pair that answers ``operation="abs"`` against the library
    # default ``"add"`` is count 2 against count 258.
    # At 60 Hz, count 258 is 4.25 s. Pinned by ``check_env_wiring.py``, which
    # recomputes the cap from the three cfg fields rather than reading 256, and
    # which also pins the two write sites of ``_obs_calls`` -- a third one
    # would turn it into a per-episode counter and 258 would never arrive.
    report_at_steps = (2, 60, 180, 258)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=3.0, replicate_physics=True)

    # =======================================================================
    #  REAL-TASK values that are NOT yet known (D-109, D-113, D-114, D-110)
    # =======================================================================
    #
    # These are the numbers the concept phase deliberately left open. They are
    # named here so they exist as configuration, and they are UNSET so nothing
    # can quietly train on an invented value. ``validate_rl_config`` below
    # turns a missing one into a hard stop.
    #
    # This is the WORKCELL_PENDING contract (insertion_tasks_cfg.py) with
    # teeth: that tuple REPORTS its unknowns in the startup report, because
    # the scene is internally consistent without them. These are different --
    # a reward or a termination built on a placeholder number trains a task
    # nobody decided, and the run would look finished.
    #
    # THE RING, and how it is broken (decided 2026-08-28). Some of these
    # values can only be READ OFF A RUN: ``rung_step_sizes`` comes from this
    # project's own learning curves, ``abort_payment`` must fit a reward scale
    # nobody has seen yet. An env that refuses to build without them would make
    # its own measurement impossible. ``rl_terms_enabled`` below is the way
    # out: with it OFF the env computes no real reward, no force abort and no
    # curriculum, reads none of these values, and the guard does not run. That
    # is the measurement env. With it ON -- the default -- every value here
    # must be set.
    #
    # TWO KINDS OF "NOT KNOWN", kept apart since 2026-08-28. ``RL_PENDING`` is
    # UNSET: the env refuses to build. ``RL_PLACEHOLDERS`` IS SET TO A NUMBER
    # NOBODY MEASURED ON THIS SETUP: the env builds, and every run says so in
    # its startup report and in its metrics file. The second is strictly worse
    # than the first and only exists because a human decided to trade honesty
    # of the value against progress; making it loud is what keeps that trade
    # visible.
    #
    # "NOT MEASURED HERE" IS NOT THE SAME AS "INVENTED" (corrected 2026-09-12,
    # round-2 critic). Both sentences used to read "an INVENTED number", which
    # is true of some entries and false of others -- the three scatter widths
    # of D-182 / D-183 cite the TacSL table (since 2026-09-15), the UR5e datasheet and Factory's
    # own held-asset noise (``Belege_Streuwerte.md`` B4a, B4b, B5), and four
    # OSC entries carry Factory's values. What every entry DOES share is that
    # the step from whatever source exists to THIS env's number is ours and
    # unmeasured. Which kind an entry is stands in its own mark below, and
    # nowhere else.
    #
    # WHERE THE GUARD IS CALLED: ``InsertionEnv.__init__``, and only there.
    # Four scripts build an env (train, play, zero_agent, random_agent) and
    # all four pass through that constructor, so a guard in any one script
    # would protect one of four and a script added later would skip it
    # silently.

    # [placeholder] D-114. THIS NUMBER IS INVENTED. The user set it on
    # 2026-08-28 so that the value exists and the code can go on: first 50 N
    # ("das muss ueberhaupt mal laufen"), then 100 N the same day, after RT-68
    # measured a scripted peak of 50.96 N and aborted all 32 episodes on the
    # 50. The raise is explicitly NOT a measurement and NOT the final value --
    # it only stops the placeholder from ending every run while the task is
    # still being analysed and set up. It came back DOWN on 2026-08-31
    # (D-158), after RT-107 showed the policy pressing against the 100 as a
    # ceiling rather than avoiding it (force_norm_n p95 87.6 N). Still not a
    # measurement -- a lower invented number, on the grounds that an
    # unmeasured safety bound should err low. It went UP again on 2026-09-01
    # to 300 N, USER-SET as an observation window ("dann haben wir mehr zum
    # schauen"): the first PPO run under the reworked reward is to show the
    # policy's own force distribution instead of the limit. 300 N is the
    # ``--f-abort`` RT-129 ran under, so every clean scripted episode of that
    # run lies inside the window. Same status as before: not a measurement.
    # D-158's concern (RT-107 pressed against the 100 N ceiling; nothing but
    # the abort prices force) is accepted knowingly; ``force_norm_n`` p95 and
    # ``force_abort_rate`` in the next ``demo_metrics.json`` show whether it
    # repeats. Home of the reasoning: the inbox entry "The tilted scripted
    # insertion is SPENT ..." (2026-09-01). It has NO
    # source: not D-114, not a measurement of ours, and not the force-criterion
    # literature check, which states outright that the limit comes from measured
    # distributions rather than from the literature. What D-114 actually asks
    # for is unchanged -- the force distribution measured at the
    # scripted-insertion gate, together with the abort payment -- and the whole
    # force topic sits under a named counter-evidence (Factory Table IV, where
    # adding force cost ~35 % relative success).
    #
    # It is therefore NOT in ``RL_PENDING`` (it has a value, so the guard has
    # nothing to refuse) but in ``RL_PLACEHOLDERS`` below. The env's startup
    # report and ``demo_metrics.json`` both carry that dict, so no run can
    # quietly report a success rate produced under a guessed abort limit.
    #
    # 2026-09-06, D-169: 300.0 -> 50.0 N, and for the first time the number
    # comes off a MEASUREMENT rather than off a round guess. RT-157 replayed
    # the RT-156 policy from four fixed start heights and the traces give the
    # per-episode maximum of the same EMA-smoothed force the abort reads
    # (``rt_logs/VERDICTS.md``, 2026-09-06; ``scripts/trace_outcome_split.py``).
    # Over 1546 SUCCESSFUL first episodes at +40 and +60 mm, a 50 N limit would
    # have aborted NONE; 40 N would have aborted 2, 30 N about 3 %, and 20 N
    # HALF of them. The window therefore closes to the smallest round number
    # that still costs no known solution.
    #
    # It is NOT the number D-167 anticipated. D-167 (c) made the lowering
    # conditional on the abort never firing AND ``force_norm_n`` p95 sitting
    # far under 20 N. The first half held (``force_abort_rate`` 0.0 at all four
    # heights); the second did not -- p95 read 25.79 / 27.70 / 28.01 / 29.07 N,
    # i.e. ABOVE 20 N. D-166's 20 N is the SEARCH force, not the joining force.
    # Still a bound on our own simulation, still not a part-damage limit.
    #
    # 2026-09-15, USER DECISION: 50.0 -> 30.0 N for the restarted seed study
    # (inbox entry "Seed study restart: 1000 fixed iterations, a 30 N force
    # abort, every seed from scratch"). NOT off a measurement. The RT-201
    # family, trained under 50 N, read force_norm_n p95 30.68 / 30.89 /
    # 30.85 N (RT-201, RT-201s2, RT-201s3 demo_metrics.json), so about one
    # episode in twenty of such a policy would now end in the abort. Whether
    # training under the lower limit lowers the force or the success rate is
    # what the restarted runs show. Still a bound on our own simulation.
    force_abort_f_max_n: float = 30.0

    # [placeholder] D-114. THE MAGNITUDE IS STILL INVENTED. D-114 decided the
    # FORM -- a fixed negative payment on abort -- and left the magnitude
    # ``[open]``, "decided together with F_max at the measurement". On
    # 2026-08-30 the user set a value so the env can build, and chose the
    # smallest round number on OUR OWN reward scale: one kernel peak. That is
    # the same relative-to-scale logic Beltran-Hernandez used, whose -50 is on
    # a different scale and transfers as a form, never as a number.
    #
    # SMALL ON PURPOSE, and that is an argument rather than a shrug: the abort
    # already forfeits the entire remaining episode return, which is always
    # positive (D-109 (11)), so losing it IS the punishment and this payment is
    # a marker on top. The documented risk points the other way -- Factory
    # Tab. IV loses ~35 % relative success when a force cost is added, the
    # named counter-evidence D-114 carries unresolved.
    #
    # NOT the final number. The FORM of the final one is fixed now (a multiple
    # of the per-step kernel peak); the factor comes from the force
    # distribution measured at the scripted-insertion gate, and BOTH loud
    # placeholders -- this one and ``force_abort_f_max_n`` -- leave
    # ``RL_PLACEHOLDERS`` in that same decision.
    #
    # Home of the reasoning: branch p1-konzept-messung, commit 77f5b3c,
    # ``docs/decisions_inbox.md`` entry "abort_payment = -1.0 as a loud
    # placeholder; the real magnitude stays coupled to the F_max measurement".
    abort_payment: float = -1.0

    # D-109 point (5). The depth at which the "engaged" display bonus starts
    # paying. DECIDED 2026-08-30 -- it left RL_PENDING that day, and the
    # ``[CAD pending]`` mark with it.
    #
    # Derived, not typed: ``insertion_tasks_cfg.ENGAGED_DEPTH`` is
    # 30 % of POCKET_SEAT_DEPTH and owns both the fraction and the reasoning.
    # The fraction is USER-SET and ad hoc; the FORM is the Isaac Lab factory
    # task's. Read the note there before quoting the number anywhere.
    engaged_depth_m: float = insertion_tasks_cfg.ENGAGED_DEPTH

    # THE LADDER SWITCH (D-110 (3)), added 2026-08-30. OFF by default.
    #
    # ON  -- the manual multi-axis ladder runs, and ``rung_step_sizes`` must
    #        carry a real value.
    # OFF -- one FIXED rung, no advance, no retreat. ``rung_step_sizes`` must
    #        stay ``None``; a value set while the ladder is off would be a
    #        number nobody reads, and the guard refuses it.
    #
    # WHY THE SWITCH EXISTS AT ALL, said out loud: D-110 (3) sources the step
    # sizes from THIS PROJECT'S OWN LEARNING CURVES. Learning curves come from
    # a training run. The run cannot start while the step sizes are unset. The
    # only way out of that ring that invents no number is to run the first
    # training WITHOUT a ladder -- and to say so in every log line, so no
    # result is ever read as if a curriculum had been active.
    #
    # THE SWITCH IS A PUBLISHED PATTERN, not own construction. AutoMate ships
    # the same construct in the same framework and env base class:
    # ``if_sbc: bool = True`` (Isaac Lab 2.3.2,
    # ``isaaclab_tasks/direct/automate/assembly_tasks_cfg.py:135``), read in
    # ``assembly_env.py:189-193`` and set to False for evaluation. One
    # semantic difference, so a reader who knows ``if_sbc`` is not misled:
    # AutoMate's OFF pins the difficulty at the HARD end of its ladder (the
    # part starts fully outside); our OFF pins rung 0. The precedent for
    # training WITHOUT any ladder at full fixed scatter is Factory and FORGE,
    # which have no curriculum at all. IndustReal's released code has no
    # ladder-off mode; its step size is a constant in the source, so the ring
    # never arises there. OWN ADDITION, and only this: the guard REFUSES a
    # ``rung_step_sizes`` that is set while the ladder is off (AutoMate keeps
    # reading its curriculum fields either way).
    curriculum_enabled: bool = False

    # [open] D-110 point (3). Per-axis widening per curriculum rung. Comes from
    # this project's own learning curves, not from IndustReal's 5 mm, which is
    # a different ladder variable. ``curriculum.Ladder`` refuses to build
    # without it as well. Required only while ``curriculum_enabled`` is True --
    # see ``CURRICULUM_PENDING`` below.
    rung_step_sizes: dict | None = None

    # MEASURED 2026-08-28 (RT-59), and it left RL_PENDING that day. The link
    # whose incoming joint wrench is the force signal (D-114).
    #
    # It was the stream's blocker 1: the assumption was that a dedicated
    # force-sensor link had to be authored on the UR5e, the way FORGE authors
    # ``/panda/force_sensor``. It did not. Our tool is ALREADY its own link --
    # ``tool_link``, welded to ``wrist_3_link`` via ``tool_weld`` -- so the
    # wrench arriving there IS the flange-to-tool wrench a wrist sensor would
    # read. FORGE needs an extra link because its part is not welded on; ours
    # is.
    #
    # That is an api-shape claim (D-080) and it was paid for with a probe, not
    # an argument: ``scripts/probe_wrench_bodies.py`` on branch
    # p2-szene-umbau, run as RT-59. ``tool_link`` is the LAST of 8 bodies, so
    # at rest its incoming force must be its own authored ``m*g`` and nothing
    # else -- the probe's own control. It read |f| = 8.0861 N against
    # 8.0834 N (mass 0.82400 kg), ratio 1.000, verdict
    # ``TOOL_LINK_CARRIES_THE_WRENCH``, exit 0. Home of the fact:
    # ``HANDOFF-SZENE.md`` on branch p2-szene-umbau, "RESULT of RT-59 / RT-60".
    #
    # Consequence: the S2 asset job named in HANDOFF-SZENE.md does not have to
    # happen, and this is a DECIDED value like ``ft_smoothing_factor`` -- not
    # pending (it has a value) and not a placeholder (the value was measured).
    force_sensor_body_name: str = "tool_link"

    # THE GRAVITY TARE (2026-08-31). Subtract the welded tool's own weight
    # from the joint wrench before it becomes the force channel.
    #
    # THE CONFLICT THIS RESOLVES. D-114 imports FORGE's force channel
    # wholesale. FORGE's joint wrench IS the contact force, because Factory
    # and FORGE spawn robot and held asset with ``disable_gravity=True``
    # (``factory_env_cfg.py:127``, ``factory_tasks_cfg.py:165``). This repo
    # keeps gravity ON at the robot (supervisor, 2026-08-25, ``ur5e_cfg.py``,
    # superseding p1-gains D-082). The same code therefore reads a different
    # quantity here than at its source: contact PLUS a constant 8.083 N of
    # tool weight. RT-59 measured exactly that (8.0861 N at rest, no contact,
    # ratio 1.000 against m*g); RT-114 read the same 8.08 N with the part in
    # free air, 121.6 mm from anything and zero interpenetration.
    #
    # Neither decision is overturned: gravity stays on, and the channel now
    # carries what D-114 imported it to carry. What the tare does NOT touch is
    # the FRAME -- the wrench stays raw in the parent body frame, per the
    # 2026-08-28 inbox entry; only the known weight vector is rotated, because
    # it is fixed in the world while the sensor frame turns with the wrist.
    #
    # UNVERIFIED until a training-PC run reads ~0 N in free air.
    force_gravity_tare: bool = True

    # THE OBSERVATION NOISE (D-182), ON by default. Two scalars and nothing
    # else: NO TENSOR MAY ENTER THE CFG, or hydra's ``to_dict`` and the
    # ``params/env.yaml`` every run writes would have to carry one.
    # ``obs_noise.py`` turns the two into the (28,) masks at env-build time.
    #
    # Both at 0.0 build NO model at all (``resolve_obs_noise_model``), which
    # is the un-noised env bit for bit rather than a zero-width Gaussian that
    # still draws -- a draw per step would shift the global RNG stream and the
    # "no noise" arm would stop reproducing the run it is compared against.
    #
    # THIS REPLACES the D-111 pair ``obs_noise_enabled`` /
    # ``obs_noise_amplitude`` and the own form ``add_obs_offset``, all three
    # deleted. The library hook applies AFTER ``_get_observations`` and only
    # to ``obs_buf["policy"]`` (``direct_rl_env.py:414-415``), so reward,
    # termination and success read the CLEAN channel by construction rather
    # than by hand. Activation no longer waits on supervisor question F1;
    # F1 stays open as a question, the switch is the user's call (D-182).
    #
    # THE POCKET NUMBER IS A UNIFORM HALF WIDTH, NOT A SIGMA (user,
    # 2026-09-15; p1-thesis docs/decisions_inbox.md, entry "Pocket-position
    # observation noise becomes uniform +-5 mm"). The field keeps its old
    # name ``_std_m`` by user decision; the value is the half width of a
    # uniform draw in [-0.005, 0.005] m per tip_rel axis, once per episode.
    # Source of the value: TacSL, arXiv:2408.06506v2, App. C, Tab. V
    # (socket observation noise [-0.005, 0.005] m). TacSL does not state the
    # draw time; "once per episode" is kept from D-182 / FORGE App. A. The
    # uniform's own sigma is 0.005/sqrt(3) = 2.89 mm, above the 2.5 mm at
    # which FORGE (Sec. V-A) reports unstable training -- named, unresolved.
    # Replaces D-182's Gaussian sigma 0.0025 (FORGE Tab. II).
    obs_noise_pocket_pos_std_m: float = 0.005
    force_obs_noise_std_n: float = 3.5
    # THE TORQUE SIGMA (D-188), its OWN number in N m -- the 3.5 N are NOT
    # carried over. 0.0 and OPEN (user 2026-09-13: observation noise is a
    # suspect for the parking, so no provisional value; a later widening
    # starts from 0). Only meaningful under obs_wrench_mode="wrench"; a
    # non-zero value in "force" mode is REFUSED by insertion_math.noise_masks
    # at env build, because it would land on no channel. Belege B4c.
    torque_obs_noise_std_nm: float = 0.0
    # THE MOMENT'S REFERENCE POINT, metres in the PARENT (wrist_3_link)
    # frame from the parent link origin; the env's lever is COM minus this
    # point. Measured by `probe_wrench_bodies.py --tare-check` point 2 with
    # the joints PINNED (RT-192 expectation, Nachtraege). Home of the
    # number: rt_logs/RT-192_expectation.md.
    # MEASURED RT-192a5 (2026-09-13, joint_pd, static pose, RAW deltas): a
    # 5 N load at the tip (0.152 m) reads a lever of 0.2551 m, at tip - 0.1 m
    # a lever of 0.1552 m (difference 0.0999: the application point is
    # honoured), a y load at the tip 0.2552 m (no sideways offset). The
    # moment is therefore taken about a point 0.1032 m UP the arm along the
    # tool axis from the link origin, NOT the origin the Isaac test assumes.
    # NOTED, NOT EXPLAINED: that is the mirror image of the tool's authored
    # COM (+0.1032 m), i.e. algebraically "moment about the origin plus
    # (COM - origin) x F". With one asset the two readings cannot be told
    # apart; the constant is what was measured. a2 read the same number
    # under the drifting OSC; a3/a4 were test artefacts (expectation
    # Nachtraege). Home of the number: rt_logs/RT-192_expectation.md.
    torque_ref_offset_parent_m: tuple[float, float, float] = (0.0, 0.0, -0.1032)

    # THE GRASP OBSERVATION OFFSET (D-183), ON by default. Uniform
    # +-``grasp_obs_offset_x_m`` along the part's SHORT axis, drawn once per
    # episode, applied ONLY in ``_get_observations``. The PHYSICS STAYS
    # NOMINAL -- mass, centre of mass, lever arms and the contact body all sit
    # at the nominal weld -- which is the named gap in D-183 and the reason
    # this is a belief error rather than a grasp error.
    grasp_obs_offset_x_m: float = 0.003

    # ALSO not in RL_PENDING: a decided number, so it has a value.
    # D-114 puts the force channel through an EMA at 0.25. The source of the
    # number is FORGE (``forge_env_cfg.py:102`` ``ft_smoothing_factor = 0.25``)
    # and the source of the CONVENTION -- which side the 0.25 sits on -- is
    # ``forge_env.py:99-100``, read 2026-08-28: it weights the NEW sample.
    # ``insertion_math.ema_update`` owns the convention; this owns the number.
    ft_smoothing_factor: float = 0.25

    # =======================================================================
    #  THE REAL REWARD'S NUMBERS (D-106, D-109). LIVE since M2.4b step 3
    #  (2026-08-30): ``InsertionEnv._get_dones`` / ``_get_rewards`` read
    #  every field below through ``insertion_math`` / ``insertion_sdf``.
    # =======================================================================
    #
    # Every one of these is a POINTER, not a number. The reasoning and the
    # derivation live in ``insertion_tasks_cfg.py``; a literal typed here
    # would be a second home for the same fact and would stop tracking the
    # geometry the day the fixture is re-measured.

    # The three-kernel sum, D-109 (9). ``a`` is the width, ``b`` the squash
    # exponent; both come in pairs.
    kernel_a_coarse: float = insertion_tasks_cfg.KERNEL_A_COARSE
    kernel_b_coarse: float = insertion_tasks_cfg.KERNEL_B_COARSE
    kernel_a_mid: float = insertion_tasks_cfg.KERNEL_A_MID
    kernel_b_mid: float = insertion_tasks_cfg.KERNEL_B_MID
    kernel_a_fine: float = insertion_tasks_cfg.KERNEL_A_FINE
    kernel_b_fine: float = insertion_tasks_cfg.KERNEL_B_FINE
    # The tilt kernel of the alignment term (RT-171): the same
    # ``arccosh(10)/margin`` rule, margin = the RT-158 mouth-mode tilt. Home
    # of the number and of the reasoning: ``insertion_tasks_cfg.KERNEL_A_TILT``.
    kernel_a_tilt: float = insertion_tasks_cfg.KERNEL_A_TILT

    # D-109 (5): both display bonuses weigh 1.0 and are paid per timestep
    # without a latch (Isaac Lab factory pattern).
    #
    # NOT the proxy's ``reward_w_success``. That field was the PROXY's 10.0
    # and DIED with the proxy reward (M2.4b step 3); reusing its name would
    # have silently given the real success bonus ten times the weight D-109
    # decided.
    w_engaged: float = 1.0
    w_success: float = 1.0

    # D-165 (2026-09-02): the proxy's depth-progress term, per metre of NEW
    # gated max depth (``insertion_env._last_depth_progress``, read from the
    # same buffer as ``mean_max_depth_mm``). RT-135 measured the D-109
    # kernels flat beside the stage-2 opening (5.82e-5 per step across
    # 4.88 mm of lateral error), and RT-134 showed the policy parking there;
    # this term pays for every millimetre of entry once a corner is in. The
    # value is the square-peg proxy's ``reward_w_depth`` = 100.0
    # (``proxytask_env_cfg.py:130``), carried as ``[proxy]`` in
    # PROXY_TASK_VALUES: no measurement on this task sizes it yet (full seat
    # = 3.6 against ~59 of kernel income per episode). Zero reproduces the
    # pre-D-165 reward bit for bit -- the regression switch,
    # ``env.w_depth_progress=0`` per hydra.
    w_depth_progress: float = 100.0

    # THE ALIGNMENT TERM (RT-171, 2026-09-06): potential-based shaping,
    # ``F = shaping_gamma * Phi(s') - Phi(s)``, ``Phi = w_tilt * sech(a * theta)``
    # with ``theta`` the part's tilt against the POCKET axis
    # (``insertion_math.tilt_cos_theta`` / ``tilt_potential``). Why: RT-170
    # BEFUND 2 measured the kernel nearly blind to tilt (1.02e-5 per degree
    # against 2.35e-4 per mm), and RT-158 measured the failing part TILTED
    # (mouth mode 15.34 deg, rim mode 5.42 deg). The term pays for righting
    # and charges for tilting, and nothing else: discounted over an episode
    # it telescopes to ``gamma^T Phi(s_T) - Phi(s_0)``, so it cannot change
    # which policy is optimal (Ng, Harada & Russell 1999). Success/failure
    # ordering stays with the lump.
    #
    # THE WEIGHT IS DERIVED FROM A PER-EPISODE BUDGET, not typed. The time
    # penalty is ``-1/T`` per step so that a whole episode costs exactly
    # 1.0; ``w_tilt = 1.0`` is that same unit and says "from the cone edge
    # to upright is worth one episode of time". Lower bound, from the only
    # extra cost righting has over sitting still -- the action-rate penalty:
    # ``0.0034 * sqrt(6) * 38`` control steps (RT-168, kp_rot 30) = 0.3164
    # over the term's total pull ``(1 - 0.1) w`` gives ``w >= 0.3516``. 1.0
    # is 2.84x that bound and below the progress budget (3.6 for a full
    # seat). Sensitivity decade {0.1, 1.0, 10.0} on the SUCCESS RATE (D-109
    # (9)(iii)); 0.1 is below the bound and is the predicted FAIL arm. Zero
    # switches the term off and reproduces the pre-RT-171 reward bit for
    # bit -- the regression switch, ``env.w_tilt=0`` per hydra.
    # DEFAULT 0.0 SINCE D-172 (2026-09-07): the alignment term is dropped.
    # RT-172 solved a 0..5 deg pocket under the unchanged reward, so the
    # term has no motivation left; the code stays as the FALLBACK for a
    # rung that fails without it. At 0.0 the reward is the pre-RT-171
    # reward bit for bit. To revive the term, set ``env.w_tilt=1.0`` per
    # hydra -- 1.0 is the derived value above, not a new choice.
    w_tilt: float = 0.0
    # gamma of the shaping row. ONE HOME: ``agents/rsl_rl_ppo_cfg.py``,
    # ``PPORunnerCfg.algorithm.gamma`` -- read from there, never typed here.
    # NAMED GAP: a hydra override of ``agent.algorithm.gamma`` does NOT reach
    # this field (the env never sees the agent cfg at run time); such a run
    # must set ``env.shaping_gamma`` to the same number, and the startup
    # report prints this value so the log shows what the row used.
    shaping_gamma: float = PPORunnerCfg().algorithm.gamma

    # =======================================================================
    #  THE TWO PER-STEP PENALTIES, adopted 2026-09-01. Home of the decision:
    #  inbox entry "Reward-Ueberarbeitung" 2026-09-01 (p1-konzept-messung),
    #  points (6) and (8). Both are NEGATIVE, which is a deliberate revision
    #  of D-109 (11)'s "every step pays strictly positive" -- the decision
    #  states the two inequalities that keep the suicide exploit closed
    #  anyway, and they are not restated here.
    # =======================================================================

    # THE HORIZON, DERIVED AND NEVER TYPED A SECOND TIME. Isaac Lab computes
    # the step count as ceil(episode_length_s / (sim.dt * decimation))
    # (``direct_rl_env.py:286``), and this is that same expression over this
    # same config, so ``episode_steps`` cannot drift away from the
    # ``max_episode_length`` the env actually runs. The RATIO form of
    # ``episode_length_s`` above is what makes it land on 256 rather than 257;
    # that field stays the one home of the horizon, this is its step count.
    episode_steps: int = math.ceil(episode_length_s / (sim.dt * decimation))

    # R_time = -1/T per step, paid on EVERY step, terminal steps included.
    # Source: Brahmbhatt, Deka, Spielberg, Mueller, "Zero-Shot Transfer of
    # Haptics-Based Object Insertion Policies", ICRA 2023, arXiv:2301.12587,
    # Sec. III -- their -0.001 slide value IS -1/T at their horizon, which is
    # why the form and not the number is imported. Over a full episode the
    # term sums to exactly -1.0, i.e. one abort payment.
    time_penalty_per_step: float = -1.0 / episode_steps

    # -action_rate_scale * ||a_t - a_(t-1)||_2 per step, the FORGE form
    # (``forge_tasks_cfg.py:15``); Brahmbhatt has the same term as R_delta-a.
    # It EXECUTES the fallback D-109 (8) named for itself.
    #
    # THE SCALE IS DERIVED HERE, NOT IMPORTED (D-168, 2026-09-05). It was
    # FORGE's 0.1 and the caveat below said so; RT-151c/RT-153 turned that
    # caveat into a measured failure. The term reads the SAMPLED action
    # (``insertion_env.py:870,883``), so its expected value is a tax on the
    # policy's own exploration noise sigma and on nothing else:
    #
    #   a_t - a_(t-1) ~ N(0, 2*sigma^2 * I_6)  =>  E||.|| = sigma*sqrt(2)*E[chi_6]
    #   E[chi_6] = 2.3501, so per step  E||.|| = 3.3235 * sigma
    #   per episode = sigma * (1 * 2.3501 + 255 * 3.3235) = 849.8 * sigma
    #                          ^ first step: _prev_actions is 0 after reset
    #
    # At 0.1 and the rsl_rl start sigma of 1.0 (``rsl_rl_ppo_cfg.py:31``,
    # init_noise_std) that is 85 per episode against a TOTAL kernel income of
    # 57 (RT-151c: 0.2235/step * 256; logged Episode_Reward/kernels 56.4-57.1).
    # The kernel is flat near the axis, so the largest reliable gradient in
    # the whole reward pointed at "shrink sigma". RT-148b: sigma 0.99 -> 0.01
    # by iteration 500, 97 % of the apparent return gain coming from this term
    # alone. rsl_rl has no sigma floor, so nothing stopped it.
    #
    # THE BUDGET, written down before the run that tests it:
    #   S = 1.0  -- the target sigma to keep alive, i.e. init_noise_std itself
    #   X = 5 %  -- share of the 57 kernel income the noise tax may cost at S
    #   scale = 0.05 * 57 / 849.8 = 0.00335 -> 0.0034 (5.07 % at S)
    #
    # CROSS-CHECK against the source this number left: FORGE prices the same
    # norm over an EMA-SMOOTHED action buffer (``factory_env.py:213``,
    # ema_factor_range [0.025, 0.1] in ``forge_env_cfg.py:23``), which this env
    # does not have. Referred to the raw draw, FORGE's 0.1 is worth 0.0018 to
    # 0.0073 depending on ema_factor. 0.0034 lands inside that band: the form
    # stays FORGE's, the number is this task's.
    #
    # WHAT THIS BUYS AND WHAT IT COSTS: it stops the sigma tax. It does NOT
    # make the task solvable -- the kernel is still flat, so success stays 0
    # (RT-154 expectation). The risk it opens is raw-action drift: this term
    # is the only price on the UNCLAMPED command, and RT-149 already measured
    # |a| up to 3.5. Instrumented, not assumed: action_abs_max_mean and
    # action_sat_frac on the curve (``insertion_env.py``, extras["log"]).
    action_rate_scale: float = 0.0034

    # D-106 (3): the SAPU threshold, half the cross play.
    interpen_thresh: float = insertion_tasks_cfg.INTERPEN_THRESH

    # D-106 (1): the success band, in the env's depth coordinate (0 = the
    # stage-2 opening plane, depth counts downward). A BAND, not a threshold --
    # past full seat the part is through the fixture.
    #
    # These are the same two constants the four-corner gate already reads as
    # ``fixed_asset.success_depth`` / ``fixed_asset.depth``. They are named
    # again here because ``insertion_math.in_success_region`` takes them as
    # ``depth_min`` / ``depth_max`` and the pairing has to be unambiguous at
    # the call site; both still point at the one home.
    depth_min: float = insertion_tasks_cfg.SEATED_SUCCESS_DEPTH
    depth_max: float = insertion_tasks_cfg.POCKET_SEAT_DEPTH

    # =======================================================================
    #  THE SDF / SAPU QUERIES (D-109 (3), D-109 (10), D-154)
    # =======================================================================
    #
    # Two queries per RL step, both built once in ``InsertionEnv.__init__``:
    # the reward query samples the part and asks the PART's own surface at the
    # goal pose; the SAPU query samples the same part and asks the FIXTURE
    # mesh. The OBJ paths come from ``insertion_paths.py`` and are not
    # repeated here.

    # D-154: 64,000, chosen off the cost curve RT-97 measured. Point spacing
    # 1.047 mm, 1.78x the cross play, 174 ms per CALL at 128 envs. D-154 is the
    # home of the trade; note it priced ONE call and the step makes TWO.
    sdf_num_sample_points: int = 64000

    # The source's own 1.5 (``industreal_algo_utils`` ``mesh_sdf`` L299,
    # ``get_batch_sdf`` L323) in METRES: the radius beyond which the Warp query
    # gives up and reports this value instead of a distance. It has to exceed
    # any distance the reward should still see; the coarse kernel's margin is
    # 173.21 mm since 2026-09-01 (KERNEL_MARGIN_COARSE, the measured travel
    # reach), so 1.5 m is still an order out and nothing the reward has to see
    # is clipped.
    sdf_max_dist: float = 1.5

    # The surface sampling is random and the sample set is fixed for the life
    # of the env, so the seed is part of the reward definition, not a nuisance
    # parameter. Same value as ``cfg.seed`` would be misleading -- that one
    # seeds the policy and the resets.
    sdf_seed: int = 0

    # The switch that breaks the ring described above. Default ON, so the
    # guarded path is the one nobody has to remember.
    #
    # ON  -- the real reward, the force abort and the curriculum run, and
    #        ``validate_rl_config`` must pass.
    # OFF -- the measurement env: reward 0, no force abort, no curriculum,
    #        and the guard is skipped because nothing reads the open values.
    #        ``train.py`` refuses to start in this state; training on a zero
    #        reward is always a mistake, never an experiment.
    #
    # NAMED AS OURS: this is own construction. The proxy has a zero-reward
    # stub as a one-off state of the code, not a switch, and no source in the
    # Isaac Lab / IsaacGymEnvs lineage carries an equivalent -- their open
    # numbers do not exist, so they never needed a way to run without them.
    rl_terms_enabled: bool = True


# The names above that are still UNSET and must be filled before an env may be
# built. One place, so a fifth open value is added here and nowhere else.
# TWO names left this tuple on 2026-08-28, for opposite reasons.
# ``force_abort_f_max_n`` left because it holds an INVENTED number, so it moved
# to RL_PLACEHOLDERS below. ``force_sensor_body_name`` left because RT-59
# MEASURED it (``tool_link``, ratio 1.000 against its own weight), so it is in
# neither table -- it is simply a decided value now.
#
# TWO MORE NAMES LEFT ON 2026-08-30, again for opposite reasons, when the
# concept stream decided the five numbers this stream was blocked on (branch
# p1-konzept-messung, commit 77f5b3c). ``engaged_depth_m`` left because it is
# now a DERIVED value with a decision behind it, so it is in neither table.
# ``abort_payment`` left because it holds an INVENTED number, so it moved to
# RL_PLACEHOLDERS below and sits next to ``force_abort_f_max_n``.
#
# ONE OF THOSE TWO MOVES IS REVERSED ON 2026-09-01. ``engaged_depth_m`` is a
# PLACEHOLDER again (RL_PLACEHOLDERS below): the concept stream replaced the
# ad-hoc 0.30 fraction with a measurement rule -- the capture depth of the
# TILTED scripted insertion -- and until that measurement exists the field
# holds a number nobody measured. It goes into the placeholder table and not
# back into RL_PENDING on purpose: RL_PENDING stops the run, and a run that
# cannot start also cannot produce the scripted-insertion measurement the
# value is waiting for. ``abort_payment`` moved the other way in the same
# decision -- its magnitude is now decided and only its lack of a measurement
# keeps it in the table -- so the two no longer fall together.
#
# ONE NAME IS LEFT, and it is the one that needs a RUN rather than a decision.
RL_PENDING: tuple[str, ...] = (
    "rung_step_sizes",
)

RL_PENDING_MARKS: dict = {
    "rung_step_sizes": "[open] D-110 (3) -- from our own learning curves",
}

# The subset of RL_PENDING that is pending ONLY while the ladder runs, added
# 2026-08-30 with ``curriculum_enabled``. These names stay in RL_PENDING -- a
# rename still has to be caught as a code fault, and the day the ladder is
# switched on they are demanded again without a second edit. What the subset
# changes is one thing: with ``curriculum_enabled = False`` an unset value here
# does NOT stop the run, and a SET value does.
#
# The asymmetry is the point. Skipping the demand would be a silent hole; the
# reverse demand (must be None) closes it, because the only two states left are
# "ladder on, value decided" and "ladder off, value absent". There is no third
# state in which a step size exists that nothing reads.
CURRICULUM_PENDING: tuple[str, ...] = (
    "rung_step_sizes",
)

# Values that ARE set but whose number nobody measured. The env builds on them;
# the price is that every run has to SAY so. Two readers carry this dict, and
# both read it mechanically rather than typing the value out:
# ``InsertionEnv._print_startup_report`` prints it, and
# ``InsertionEnv._write_metrics`` writes it into ``demo_metrics.json``, next to
# the success rate it qualifies.
#
# The distinction from RL_PENDING is the whole point. An unset value stops the
# run. An invented value does not, so the only defence left is that it cannot
# be read off a result without being seen.
#
# A MARK NEVER RESTATES THE NUMBER. It used to read "50.0 N -- INVENTED",
# which was true only as long as nothing moved the value. Both readers print
# the live value right next to the mark, so a number written into the text is
# a second home for the same fact -- and the first override makes the two
# contradict each other. That is the RT-65 defect exactly: a hardcoded label
# outliving what it describes. The mark says WHAT KIND of number it is and
# WHERE the real one has to come from; the field says which number.
RL_PLACEHOLDERS: dict = {
    "force_abort_f_max_n": (
        "[placeholder] no longer INVENTED -- that was 2026-08-28..2026-09-01 "
        "(user, no source). D-169 on 2026-09-06 set it off RT-157: across "
        "every SUCCESSFUL first episode of the two probed start heights that "
        "limit aborted none, while the search force of D-166 would abort about "
        "half of them. On 2026-09-15 the user LOWERED it for the restarted "
        "seed study, below the force p95 the RT-201 family measured, so the "
        "abort now also ends a share of episodes that would have succeeded "
        "(inbox entry 'Seed study restart'). NOT a measurement. Still only an "
        "upper bound on OUR OWN simulation, not a "
        "damage limit of the real part, and it describes ONE policy at TWO "
        "heights. In the simulation this limit is only the brake against "
        "ramming (D-114) and must sit ABOVE the force a clean insertion "
        "needs, or it forbids the solution. The suction-cup datasheet anchor "
        "is WITHDRAWN (rigid chain, no sim-to-real). D-167's condition for "
        "lowering did NOT hold: it asked for p95 far under the D-166 search "
        "force, and RT-157 read above it. Read force_norm_n p95 and "
        "force_abort_rate off every run. Home: DECISIONS.md, D-169."
    ),
    "engaged_depth_m": (
        "[placeholder] the fraction of the seat depth behind it is AD HOC "
        "(user, 2026-08-30) and was downgraded again on 2026-09-01. The rule "
        "'measured capture depth of the tilted scripted insertion' (inbox "
        "entry Reward-Ueberarbeitung) was WITHDRAWN the same day: RT-129 "
        "showed that depth to be the controller's own tilt ramp. Proposed "
        "replacement, not decided: read the depth at which the lateral offset "
        "stays inside half the cross play off the TRAINED policy's "
        "trajectories (D-071: the tilt is learned). Only consumer: the engaged "
        "display bonus. Factory's own fraction was rejected: at this pocket it "
        "lands inside the success band. Home: docs/decisions_inbox.md, entry "
        "'The tilted scripted insertion is SPENT ...' (2026-09-01)."
    ),
    "abort_payment": (
        "[placeholder] the FORM is Beltran-Hernandez via D-114 (a fixed "
        "negative on abort) and the MAGNITUDE was decided 2026-09-01 (grill; "
        "inbox entry Reward-Ueberarbeitung, the abort-payment point), which "
        "closes D-114's [open] and un-couples it from the F_max measurement: "
        "one kernel peak, i.e. the per-step maximum of the shaping term. That "
        "is a SCALE ANCHOR, our own choice, openly labelled, with an optional "
        "decade sensitivity row per the D-109 width template. It stays in this "
        "table because it is still a magnitude nobody measured; the dominant "
        "abort price is anyway the forfeited remaining return."
    ),
    "osc_kp_pos": (
        "[placeholder] task-space position stiffness, N/m. NOT adopted: "
        "Decision (four) of the inbox entry 'Audit 2026-09-03 (a)' DERIVES it "
        "as kp <= F_search / (Lambda_max * step_limit), with F_search the "
        "literature search-band force and Lambda_max the task-space inertia "
        "the startup report prints at the reset pose. Until that is done the "
        "measurement runs two values (the OSC cfg default and five times it). "
        "The OSC cfg default and the reach example are contact-free and are "
        "NOT a source."
    ),
    "osc_kp_rot": (
        "[placeholder] task-space rotation stiffness, Nm/rad. No source of "
        "ours; Factory's default_task_prop_gains rotation value "
        "(factory_env_cfg.py, CtrlCfg). Decision (four) of the same entry."
    ),
    "osc_pos_step_limit_m": (
        "[placeholder] metres of tool travel one action step may command. "
        "Decision (four): policy rate x allowed tool speed, and the tool-speed "
        "source is MISSING. Factory's pos_action_threshold (factory_env_cfg.py, "
        "CtrlCfg) until then."
    ),
    "osc_rot_step_limit_rad": (
        "[placeholder] radians of tool rotation one action step may command. "
        "Same gap as osc_pos_step_limit_m; Factory's rot_action_threshold."
    ),
    "osc_pos_clamp_m": (
        "[placeholder] half-width of the box around the pocket entrance the "
        "leading tool point may not leave, along x and y. SET by D-180 from "
        "the disk reach (the lateral radius plus the tip's swing at maximum "
        "tilt, from autodr.bounds_max()) plus one action step plus set air; "
        "no longer Factory's pos_action_bounds. check_env_wiring pins it from "
        "both sides: at least that requirement, at most the set air above it."
    ),
    "osc_pos_clamp_z_m": (
        "[placeholder] half-width of the same box along z, SET by D-180 from "
        "the disk reach (the highest legal start at maximum tilt plus the "
        "lateral radius tipped up, from autodr.bounds_max()) plus one action "
        "step plus set air. check_env_wiring proves sufficiency in the WORLD "
        "frame the box is aligned to, and a second rule keeps it below the "
        "home tip standoff. Factory offers no rule to borrow: its "
        "pos_action_bounds does not contain the reset band of peg_insert, and "
        "its own clamp is not per-axis."
    ),
    "osc_decimation": (
        "[placeholder] physics steps per policy step under OSC (policy rate = "
        "physics rate / this). No source of ours; Factory's decimation "
        "(factory_env_cfg.py, FactoryEnvCfg). D-024's rate is the controller "
        "rate now. The episode is kept as a STEP count (D-113) and its seconds "
        "follow; re-deriving the length is owed (audit, section six)."
    ),
    # THE THREE SCATTER WIDTHS (D-182, D-183). Each has a published SOURCE,
    # which is more than the entries above have -- but the step FROM that
    # source TO this env's number is ours in all three cases, so they are
    # held here for the one thing this table does: a number nobody measured
    # on THIS setup cannot leave the machine unannounced.
    "obs_noise_pocket_pos_std_m": (
        "[placeholder] the HALF WIDTH of the per-episode uniform bias on the "
        "pocket-position channels, in metres (the name still says std; user "
        "2026-09-15). TacSL's Appendix C Table V names a uniform socket "
        "observation noise and that is the source; reading it as THIS env's "
        "per-episode bias is our own step, because TacSL does not state when "
        "it is drawn. NAMED TENSION: the uniform's sigma lies above the one "
        "at which FORGE reports unstable training. Home: p1-thesis "
        "docs/decisions_inbox.md 2026-09-15, replacing D-182's value."
    ),
    "force_obs_noise_std_n": (
        "[placeholder] the per-step Gaussian on the force channels, in "
        "newtons. The UR5e datasheet gives a force-sensing PRECISION for the "
        "tool flange; reading that figure as a standard deviation is an "
        "ASSUMPTION, because the datasheet does not define it as one. FORGE "
        "uses a smaller per-timestep sigma. Home: DECISIONS.md, D-182."
    ),
    "torque_obs_noise_std_nm": (
        "[placeholder] the per-step Gaussian on the torque channels "
        "(obs_wrench_mode='wrench' only), in newton metres. OFF and OPEN "
        "(user decision, observation noise is a parking suspect); the UR5e "
        "datasheet's torque precision is the candidate when it is widened, "
        "read as a sigma by the same assumption as the force. Home: "
        "DECISIONS.md, the wrench-mode entry; Belege_Streuwerte.md, row B-four-c."
    ),
    "grasp_obs_offset_x_m": (
        "[placeholder] the half width of the uniform grasp offset along the "
        "part's short axis, in metres, seen ONLY by the observation. "
        "Factory's held-asset position noise and FORGE's Table II carry the "
        "same magnitude for their peg; that the gripper on THIS part slips "
        "by as much is the user's estimate from a CAD side view, not a "
        "measurement. Home: DECISIONS.md, D-183."
    ),
}

# THE THIRD KIND OF "NOT OUR NUMBER", added 2026-08-28. It is neither of the
# two above and mixing it into either would state something false.
#
#   RL_PENDING        the number does not exist. The env refuses to build.
#   RL_PLACEHOLDERS   NOBODY MEASURED THIS NUMBER ON THIS SETUP. The env
#                     builds and every run announces it. The entries are not
#                     all the same kind, and each one's own mark says which:
#                     invented outright (``engaged_depth_m``, ad hoc, user),
#                     an own scale anchor (``abort_payment``), borrowed from
#                     Factory (the OSC gains and step limits), read off a
#                     published figure (``obs_noise_pocket_pos_std_m`` from
#                     the TacSL table, ``force_obs_noise_std_n`` from the
#                     UR5e datasheet, ``grasp_obs_offset_x_m`` from Factory's
#                     held-asset noise -- ``Belege_Streuwerte.md`` B4a, B4b,
#                     B5), or derived from a decided ceiling (the two OSC
#                     clamps, D-180). What they share is the last step: from
#                     whatever source there is to THIS env's number, and that
#                     step is ours and unmeasured.
#                     [Corrected 2026-09-12, round-2 critic: this legend read
#                     "the number was INVENTED. No source at all.", which is
#                     false for the three scatter widths of D-182 / D-183 and
#                     for the Factory-borrowed OSC entries. It stays true of
#                     the entries whose marks say so.]
#   PROXY_TASK_VALUES the number was MEASURED -- on the SQUARE PEG. It has a
#                     real source, for a task that is not this one.
#
# WHY THE TABLE EXISTS. RT-70's report printed
# "pocket: side ... clearance ... yaw window ... success at ..." as if those
# described the user's part. They describe the 30 mm peg in its 32 mm pocket.
# The line one row above it already carried a derived warning
# ("GATE MATH IS STILL SQUARE"); the numbers below it carried none, so the
# gap was labelled in one place and unlabelled in the next.
#
# WHY THESE FIVE AND NOT MORE. A proxy number that the code no longer prints
# and no longer computes on cannot mislead anyone, so it is deleted rather
# than labelled (S6). A proxy number whose REAL value exists and is measured
# is replaced rather than labelled -- PLAY_X / PLAY_Y took over the report's
# clearance line, and the depth pair moved to POCKET_FLOOR_Z / D-106. What is
# left is exactly the set that is still READ and has no real value to move to:
# the square gate's two sides, its alignment floor, and the two constants
# whose metrics keys ``compare_runs.py`` reads and which therefore cannot be
# dropped from the file.
#
# A MARK NEVER RESTATES THE NUMBER -- the RT-65 rule, stated in full above
# RL_PLACEHOLDERS. Both readers print the live value next to the mark.
#
# ROOTS. "cfg." resolves against the env config, "tasks." against
# ``insertion_tasks_cfg``. Two roots because two of the five are module
# constants and never became config fields.
PROXY_TASK_VALUES: dict = {
    "cfg.w_depth_progress": (
        "[proxy] the square-peg proxy's reward_w_depth, paid per metre of "
        "NEW gated max depth (D-165). The proxy solved 1 mm clearance with it "
        "at 100; nothing on this task has sized it. Zero switches the term "
        "off and reproduces the pre-D-165 reward."
    ),
    "cfg.held_asset.side": (
        "[proxy] the held body modelled as a SQUARE of this side. The real "
        "part is rectangular with lugs (PART_BODY_X / PART_BBOX_M). Read by "
        "the four-corner gate; moves with the SDF rework (D-109), not alone."
    ),
    "cfg.fixed_asset.side": (
        "[proxy] the square pocket edge the gate tests against. The real "
        "opening is POCKET_OPENING_X x POCKET_OPENING_Y and is not square. "
        "Read by the four-corner gate; moves with the SDF rework (D-109)."
    ),
    "cfg.gate_min_alignment": (
        "[proxy] tool-axis alignment floor, calibrated on the square peg. "
        "The real bound is D-106's across-tilt limit, which D-121 reopened "
        "and the concept stream has not recomputed."
    ),
    "tasks.SIDE_CLEARANCE": (
        "[proxy] half-clearance per axis of the square pair. The real task "
        "has TWO different total plays, PLAY_X and PLAY_Y, which the report "
        "now prints instead. Kept only because compare_runs.py reads the "
        "metrics key side_clearance_mm."
    ),
    "tasks.YAW_WINDOW_RAD": (
        "[proxy] the C4 free-yaw window of a square in a square. The real "
        "part fits ONE way round, so there is no equivalent; D-106's yaw "
        "bound was reopened by D-121 and is not recomputed. The report no "
        "longer prints it. Kept only because compare_runs.py reads the "
        "metrics key yaw_window_deg."
    ),
}

# Which keys of ``demo_metrics.json`` carry a proxy number, and WHICH ENTRY
# ABOVE OWNS THE EXPLANATION. A value, never a second mark: repeating the mark
# here would give one fact two homes, and the first edit would make the two
# contradict each other -- the exact defect this whole block exists to stop.
#
# The keys stay in the file rather than being removed, because
# ``compare_runs.py`` reads all four (`:199-203`) and a reader that silently
# loses a column is worse than one that reads a labelled proxy number.
PROXY_METRICS_KEYS: dict = {
    "peg_side_m": "cfg.held_asset.side",
    "pocket_side_m": "cfg.fixed_asset.side",
    "side_clearance_mm": "tasks.SIDE_CLEARANCE",
    "yaw_window_deg": "tasks.YAW_WINDOW_RAD",
}


class ProxyTableBroken(RuntimeError):
    """A path in ``PROXY_TASK_VALUES`` does not resolve. Code fault, not data."""


def resolve_proxy_task_values(cfg) -> list[tuple[str, object, str]]:
    """Return ``(path, live value, mark)`` for every proxy value still in use.

    Both readers go through this rather than through the dict, so neither can
    print a mark next to a number it did not read. Unresolvable paths raise
    instead of being skipped: a skipped entry is a proxy number that quietly
    stops being labelled, which is the failure this table exists to prevent.
    """
    roots = {"cfg": cfg, "tasks": insertion_tasks_cfg}
    out: list[tuple[str, object, str]] = []
    for path, mark in PROXY_TASK_VALUES.items():
        head, _, rest = path.partition(".")
        if head not in roots or not rest:
            raise ProxyTableBroken(
                f"PROXY_TASK_VALUES path {path!r} does not start with a known "
                f"root; expected one of {sorted(roots)}."
            )
        obj = roots[head]
        for part in rest.split("."):
            if not hasattr(obj, part):
                raise ProxyTableBroken(
                    f"PROXY_TASK_VALUES path {path!r} does not resolve: "
                    f"{type(obj).__name__} has no attribute {part!r}. That is "
                    "a rename or a typo in this file, not a missing value."
                )
            obj = getattr(obj, part)
        out.append((path, obj, mark))
    unknown = [k for k, p in PROXY_METRICS_KEYS.items() if p not in PROXY_TASK_VALUES]
    if unknown:
        raise ProxyTableBroken(
            "PROXY_METRICS_KEYS points at entries that do not exist in "
            f"PROXY_TASK_VALUES: {unknown}."
        )
    return out


# The two control modes, one home. ``resolve_control_mode`` is the ONE place
# the mode is turned into numbers the base class reads (decimation, episode
# seconds, render interval); the env calls it before ``super().__init__``, so
# a hydra override of ``control_mode`` arrives before anything is derived.
CONTROL_MODES: tuple[str, ...] = ("osc", "joint_pd")


def resolve_start_lateral_offset(value) -> float:
    """``cfg.start_lateral_offset`` as the RADIUS of the start disk, in metres.

    ONE NUMBER since D-178 (3). The lateral start offset is drawn on a disk
    (``insertion_math.disk_offset``), and this field is its radius: the cfg
    default ``0.0``, the hydra form ``env.start_lateral_offset=<radius>`` and
    a Python number from gym kwargs or a test all arrive here as an ``int``
    or a ``float``. The default must stay a bare number; the field comment
    says what a tuple default breaks.

    Refused, never repaired: a LIST or a TUPLE -- the (x, y) half-width pair
    of D-176 (Phase 5 step B3), withdrawn by D-178 -- a string, a bool,
    anything else that is not a number, and a negative radius. A pair is
    refused BY NAME rather than read as one of its entries: an old pair
    command would otherwise run a disk nobody asked for and say nothing.
    """
    if isinstance(value, (list, tuple)):
        raise ValueError(
            f"start_lateral_offset={value!r}: the (x, y) half-width pair of D-176 is "
            "withdrawn. Since D-178 (3) the lateral start offset is a disk and this "
            "field is ONE radius in metres: env.start_lateral_offset=<radius>."
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"start_lateral_offset={value!r} ({type(value).__name__}) is not a number: "
            "expected ONE radius in metres (D-178 (3))."
        )
    radius = float(value)
    if radius < 0.0:
        raise ValueError(
            f"start_lateral_offset={value!r}: a radius cannot be negative. Use 0.0 "
            "to start on the pocket axis."
        )
    return radius


def resolve_control_mode(cfg) -> str:
    """Derive the rate fields from ``cfg.control_mode`` and return the mode.

    * ``joint_pd``: nothing changes -- ``decimation`` stays D-024's 60 Hz and
      ``episode_length_s`` its ratio form.
    * ``osc``: ``decimation`` becomes ``osc_decimation`` (the 15 Hz
      placeholder), ``sim.render_interval`` follows it (DirectRLEnv warns
      when it is smaller), and ``episode_length_s`` is rewritten so that the
      EPISODE STEP COUNT stays ``episode_steps`` (D-113 (3): 256 control
      steps): seconds = steps * dt * decimation. ``episode_steps`` and the
      time penalty -1/T are therefore untouched, which is what keeps
      ``validate_rl_config``'s step-count identity true in both modes.

    Refuses an unknown mode: a typo in a hydra override must not silently
    run the other controller.
    """
    mode = str(cfg.control_mode)
    if mode not in CONTROL_MODES:
        raise RlConfigIncomplete(
            f"control_mode is {mode!r}; the only modes are {CONTROL_MODES} "
            "(inbox entry 'Audit 2026-09-03 (a)', Decision (2))."
        )
    if mode == "osc":
        dec = int(cfg.osc_decimation)
        if dec < 1:
            raise RlConfigIncomplete(f"osc_decimation must be >= 1, got {dec}")
        cfg.decimation = dec
        cfg.sim.render_interval = dec
        cfg.episode_length_s = int(cfg.episode_steps) * float(cfg.sim.dt) * dec
    return mode


def resolve_obs_layout(cfg) -> str:
    """Derive ``observation_space`` from ``cfg.obs_wrench_mode`` and return
    the mode (D-188). Same contract as ``resolve_control_mode``: the env
    calls it BEFORE ``super().__init__``, because ``DirectRLEnv`` builds its
    spaces from ``observation_space`` inside its own constructor.

    One home for the width: ``insertion_math.obs_dim``. Refuses an unknown
    mode through ``insertion_math.obs_slices``' own check, so a typo in
    ``env.obs_wrench_mode=`` never silently runs the other layout.
    """
    mode = str(cfg.obs_wrench_mode)
    try:
        width = insertion_math.obs_dim(mode)
    except ValueError as exc:
        raise RlConfigIncomplete(
            f"obs_wrench_mode is {mode!r}; the only modes are "
            f"{insertion_math.OBS_MODES} (D-188)."
        ) from exc
    cfg.observation_space = width
    return mode


class RlConfigIncomplete(RuntimeError):
    """An env that cannot be configured is a hard stop, never a default."""


def validate_rl_config(cfg) -> None:
    """Refuse to build an env while a decided-but-unmeasured number is unset.

    Fail fast, per the project's error-handling rule: no check reports PASS
    without evidence, and a training run that started on a placeholder
    ``F_max`` would produce a full set of plausible metrics for a task nobody
    decided. Raises with every missing name and its mark at once, so one run
    of the guard names all of the remaining work.

    Three failures, kept apart on purpose. A field that EXISTS and is ``None``
    is a missing measurement: someone goes and measures it. A field named in
    ``RL_PENDING`` that does not exist on the cfg at all is a code fault in
    this file -- a rename or a typo -- and no measurement can ever clear it.
    Reading them as one (``getattr(cfg, name, None)``) makes the second look
    like the first and sends the reader off to measure a number that already
    has a home. The third arrived with ``RL_PLACEHOLDERS`` (2026-08-28): a
    field that holds an INVENTED number was emptied. Nothing to measure and no
    typo either -- somebody removed a value the env was built to read.

    A FOURTH failure arrived 2026-08-30 with ``curriculum_enabled``: a
    ``CURRICULUM_PENDING`` value that is SET while the ladder is OFF. Nothing
    reads it in that state, so it is a number that looks decided and is not.
    The check_seated_success test sentinel is exactly this shape, and it is
    now refused rather than merely printed.
    """
    # The absent-field sweep covers every name the guard reads: both tables
    # AND the ladder switch itself. The switch is neither pending nor a
    # placeholder, so the message must not blame a table it is not in.
    named = RL_PENDING + tuple(RL_PLACEHOLDERS) + (
        "curriculum_enabled", "episode_steps", "time_penalty_per_step",
        "action_rate_scale", "w_depth_progress",
        "w_tilt", "shaping_gamma", "kernel_a_tilt",
        "dr_mode",
    )
    absent = [name for name in named if not hasattr(cfg, name)]
    if absent:
        raise RlConfigIncomplete(
            "the RL config guard reads config fields that do not exist on "
            f"{type(cfg).__name__}: {', '.join(absent)}. That is a code fault "
            "in this file (a rename or a typo), not a missing measurement -- "
            "no number can clear it."
        )
    emptied = [name for name in RL_PLACEHOLDERS if getattr(cfg, name) is None]
    if emptied:
        lines = [f"  {name}: {RL_PLACEHOLDERS[name]}" for name in emptied]
        raise RlConfigIncomplete(
            "a PLACEHOLDER value was emptied. It never carried a measurement, but "
            "the env was built to read it:\n" + "\n".join(lines)
        )
    ladder_on = bool(cfg.curriculum_enabled)
    # A FIFTH failure, Phase 5 step B6: THE LADDER AND AutoDR OVERLAP ON TWO
    # AXES. A rung is one half-width each on start height, FIXTURE OFFSET and
    # tilt (`curriculum.py`, "WHAT A RUNG IS", D-110 (2);
    # `check_curriculum_ladder.AXES`); AutoDR's five quantities are
    # `start_height`, `lat_r`, `yaw`, `tilt` and `friction`
    # (`autodr.DR_DIMS`). The overlap is `start_height` and `tilt` -- TWO
    # axes, not three. The ladder's `fixture_offset` is `fixture_pos_noise_xy`
    # and AutoDR does NOT track it (`insertion_env.py` `_static_dr`), while
    # AutoDR's `lat_r` is `start_lateral_offset`, a different
    # quantity (train.py's own lateral-tag comment says so). The round-1
    # comment here counted three and was wrong.
    #
    # TWO IS ENOUGH TO REFUSE. On those two, both mechanisms would write the
    # same reset quantity at the same reset from their own state, and the
    # applied range would be whichever ran last while each tracked only its
    # own half -- the run would randomise something no buffer measures. This
    # is the SAME collision `InsertionEnv.__init__` refuses for the static
    # randomisation fields, one level up: that one catches a config value,
    # this one catches a second mechanism.
    #
    # NOT LIVE TODAY, and the comment must not pretend otherwise: `Ladder` is
    # not wired into the env yet (milestone M2.5, HANDOFF-RL "NOT STARTED"),
    # so today the switch alone cannot move a range. The guard is here so the
    # combination cannot appear on the day it is wired -- which is exactly the
    # day nobody will re-read this file.
    #
    # `cfg.dr_mode` is read directly, no `getattr` default: the name is in the
    # absent-field sweep above, so a rename raises there with the code-fault
    # message instead of silently reading "off" here.
    if ladder_on and str(cfg.dr_mode) != "off":
        raise RlConfigIncomplete(
            f"curriculum_enabled is True together with dr_mode={cfg.dr_mode!r}. "
            "The ladder and AutoDR both widen START HEIGHT and TILT, so on "
            "those two the range an episode actually gets would be whichever "
            "mechanism wrote last, while each one tracked only its own half. "
            "Run the ladder (dr_mode='off') or run AutoDR "
            "(curriculum_enabled=False) -- never both."
        )
    if not ladder_on:
        set_anyway = [name for name in CURRICULUM_PENDING if getattr(cfg, name) is not None]
        if set_anyway:
            lines = [f"  {name} = {getattr(cfg, name)!r}" for name in set_anyway]
            raise RlConfigIncomplete(
                "curriculum_enabled is False, so the ladder never advances and "
                "nothing reads a step size -- but one is set:\n"
                + "\n".join(lines)
                + "\nEither switch the ladder on, or leave the value None. A step "
                "size that nothing reads is a number that looks decided and is not."
            )
    demanded = RL_PENDING if ladder_on else tuple(n for n in RL_PENDING if n not in CURRICULUM_PENDING)
    missing = [name for name in demanded if getattr(cfg, name) is None]
    if missing:
        lines = [f"  {name}: {RL_PENDING_MARKS[name]}" for name in missing]
        raise RlConfigIncomplete(
            "the following real-task values are still UNSET and must be measured "
            "or decided before an env may be built:\n" + "\n".join(lines)
        )

    # THE TWO REWARD TERMS ADOPTED 2026-09-01. Neither is pending and neither
    # is a placeholder -- both have a published source -- so what the guard
    # checks is the one thing a config edit can silently break: the RELATION
    # each of them is defined by. Home of the decision: inbox entry
    # "Reward-Ueberarbeitung" 2026-09-01 (p1-konzept-messung), points (6), (8).
    #
    # (6) is a RULE, not a value: R_time = -1/T. Brahmbhatt's own -0.001 typed
    # in here would pass every other check in this file and quietly price time
    # at a quarter of the decided rate, because -0.001 is -1/T at THEIR
    # horizon, not at ours. The comparison is therefore against the derived
    # quotient, never against a literal.
    steps = int(cfg.episode_steps)
    if steps <= 0:
        raise RlConfigIncomplete(
            f"episode_steps is {steps}. It is "
            "ceil(episode_length_s / (sim.dt * decimation)) and must be "
            "positive -- the time penalty divides by it."
        )
    want_time = -1.0 / steps
    if abs(float(cfg.time_penalty_per_step) - want_time) > 1e-12:
        raise RlConfigIncomplete(
            f"time_penalty_per_step is {float(cfg.time_penalty_per_step)!r}, but "
            f"the decided rule is R_time = -1/T = {want_time!r} at T = {steps} "
            "steps (Brahmbhatt et al., ICRA 2023, arXiv:2301.12587 Sec. III). A "
            "time penalty that is not -1/T no longer sums to one abort payment "
            "over an episode, and the decision's two safety inequalities are "
            "stated for -1/T alone."
        )
    # (8) fixes the SIGN in the reward function -- the term is SUBTRACTED
    # there -- so a negative scale here would pay for the jerk it exists to
    # price. Zero switches the term off, which is a legal experiment; below
    # zero is never intended.
    if float(cfg.action_rate_scale) < 0.0:
        raise RlConfigIncomplete(
            f"action_rate_scale is {float(cfg.action_rate_scale)!r}. The reward "
            "subtracts scale * ||a_t - a_(t-1)||, so a negative scale rewards "
            "the jerk the term exists to price."
        )
    # D-165: the progress term pays w * metres of NEW max depth; a negative
    # weight would charge for entering. Zero is the regression switch.
    if float(cfg.w_depth_progress) < 0.0:
        raise RlConfigIncomplete(
            f"w_depth_progress is {float(cfg.w_depth_progress)!r}. The reward "
            "pays w * metres of new gated max depth, so a negative weight "
            "charges the part for entering the pocket."
        )
    # RT-171: the alignment potential is w * sech(a * theta); a negative
    # weight would pay for tilting, a non-positive width has no margin. Zero
    # weight is the regression switch. The shaping gamma must be a discount:
    # outside (0, 1] the row is no longer the Ng-form and the telescoping
    # identity the offline check proves does not hold.
    if float(cfg.w_tilt) < 0.0:
        raise RlConfigIncomplete(
            f"w_tilt is {float(cfg.w_tilt)!r}. The alignment potential is "
            "w * sech(a * theta), so a negative weight pays for tilting."
        )
    if not float(cfg.kernel_a_tilt) > 0.0:
        raise RlConfigIncomplete(
            f"kernel_a_tilt is {float(cfg.kernel_a_tilt)!r}; the tilt kernel "
            "width must be positive (arccosh(10) / margin)."
        )
    if not 0.0 < float(cfg.shaping_gamma) <= 1.0:
        raise RlConfigIncomplete(
            f"shaping_gamma is {float(cfg.shaping_gamma)!r}; the shaping row "
            "gamma * Phi(s') - Phi(s) needs PPO's discount in (0, 1]."
        )
    # D-182 / D-183: four SCATTER WIDTHS -- two Gaussian sigmas (force,
    # torque) and two uniform half widths (pocket since 2026-09-15, grasp).
    # A negative one is not a smaller scatter, it is a
    # sign error nothing downstream would report: ``std * randn`` is
    # symmetric, so a negative sigma draws exactly the same distribution and
    # the run looks right, while the uniform grasp draw simply turns inside
    # out. Zero is the legal switch-off in all three, so the refusal is on
    # the sign alone. The names are in RL_PLACEHOLDERS, so the absent-field
    # sweep above already covers a rename.
    for _name in ("obs_noise_pocket_pos_std_m", "force_obs_noise_std_n",
                  "torque_obs_noise_std_nm", "grasp_obs_offset_x_m"):
        if float(getattr(cfg, _name)) < 0.0:
            raise RlConfigIncomplete(
                f"{_name} is {float(getattr(cfg, _name))!r}. It is a scatter "
                "width -- a Gaussian sigma or a uniform half width -- so it "
                "may be zero to switch the scatter off, but never negative."
            )
