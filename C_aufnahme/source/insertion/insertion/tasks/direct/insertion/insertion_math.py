# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Pure reward / observation / termination math for the UR5e insertion.

WHAT LIVES HERE, AND WHY IT IS ITS OWN FILE
-------------------------------------------
Free ``@torch.jit.script`` functions, the project convention for hot reward
and observation arithmetic (Isaac Lab Direct tutorial pattern; the proxy does
the same in ``insertion_env.py``). The one deviation from the proxy is the
FILE: the proxy keeps them at the top of a 1781-line env that imports
``isaaclab``, which no laptop can import, so its offline check cherry-picks
three functions by AST name and silently covers nothing that is added later.

This module imports ``torch`` and ``math`` and NOTHING else. No ``isaaclab``,
no config classes, no env. That is what lets
``scripts/check_insertion_math.py`` execute the WHOLE file under the numpy
stand-in and check every function in it.

IT OWNS NO NUMBERS
------------------
Every constant arrives as an argument. Kernel widths, weights, thresholds and
the abort payment live in ``insertion_env_cfg.py`` (D-109, D-114); the
geometry constants come from ``insertion_tasks_cfg.py``, which the scene
stream owns. A number written here would be a second home for a fact.

Decisions implemented: D-107 (observation), D-109 (reward), D-113
(termination), D-114 (force channel + abort payment), and the reward
revision of 2026-09-01 -- inbox entry "Reward-Ueberarbeitung" 2026-09-01
(p1-konzept-messung), which revises D-109 (5)/(7)/(8)/(11) and D-113 (1)/(2):
a per-step time penalty, an action-rate penalty, and success as a TERMINATING
absorbing state paid out with its remaining return.
"""

from __future__ import annotations

import math

import torch

# ===========================================================================
#  Pose algebra (D-107 (4), D-036/D-037 frame handling)
# ===========================================================================
#
# PROVENANCE, stated rather than implied: this block is OWN CONSTRUCTION.
# It is text-ported from the proxy env, and the proxy's own origin is
# `ur10e-peg-insertion-rl` commit 8fe0d6d, whose message records that the
# code first used ``isaaclab.utils.math.matrix_from_quat``, that this import
# came from model memory rather than from a checked interface, and that it
# was therefore replaced by hand-expanded arithmetic. No documentation and no
# study stands behind the expansion, so under this stream's source hierarchy
# (docs > published work > own construction, named) it counts as ours.
#
# The library route is not available here anyway: this module may import
# ``torch`` and ``math`` and nothing else, which is what lets the offline
# check execute all of it. Binding ``isaaclab.utils.math`` would move the algebra back into the
# 1781-line env, where no laptop can reach it.
#
# What DOES stand behind it is a measurement: the proxy verified the same
# expansion against identity, 180 deg about x, 90 deg about y and z, and found
# it orthonormal with det +1 over 2000 random quaternions (worst
# |R^T R - I| = 1.7e-15). The offline check repeats that here against known
# rotations rather than against a second copy of the same formula -- two
# copies agreeing proves only that they were copied.


@torch.jit.script
def axes_from_quat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) wxyz quaternion batch -> (N, 3, 3), the body axes as COLUMNS.

    Convention ``wxyz``, the one Isaac Lab and Isaac Sim both adopt (migration
    guide, quoted in the Isaac coding-rules literature check). Columns, not
    rows: ``R[:, :, 0]`` is the body x-axis expressed in the parent frame, so
    ``R`` maps body -> parent and ``R.transpose(1, 2)`` maps parent -> body.
    Getting this backwards is not a theoretical risk -- the proxy shipped the
    transposed version once, on 2026-07-26, and only an identity test caught
    it. The mutation ``axes-stacked-as-rows`` reproduces that damage.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    col0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    col1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
    col2 = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return torch.stack([col0, col1, col2], dim=-1)


@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (N, 4) wxyz batches: ``a`` then ``b``.

    The rotation matrix of the result is ``R(a) @ R(b)``. Used to compose the
    fixture's yaw with its tilt into one orientation (D-037), which is what
    lets a yawed AND tilted pocket ride a single transform instead of two
    special cases.
    """
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) wxyz -> the inverse rotation. UNIT quaternions only.

    For a unit quaternion the conjugate IS the inverse, which is why no norm
    appears here. Everything this module receives comes from
    ``ArticulationData`` or from ``quat_mul`` of two unit quaternions, so the
    precondition holds; a check pins that conjugating twice is the identity.
    """
    return torch.stack((q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]), dim=-1)


@torch.jit.script
def rotate_into_frame(rot: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Express (N, 3) vectors in the frame whose axes are the COLUMNS of ``rot``.

    Component j is ``sum_i rot[i, j] * v[i]``, i.e. ``(R^T v)_j``. Written as a
    broadcast product and a sum rather than ``einsum`` or ``bmm``, because the
    offline stand-in implements neither.

    One home for this convention. The proxy env kept a rotation AND its
    transpose as two attributes and used each in a different place, which is a
    convention that can be silently half-applied.
    """
    return torch.sum(rot * v.reshape(-1, 3, 1), dim=1)


@torch.jit.script
def relative_pose(
    pos_a: torch.Tensor,
    quat_a: torch.Tensor,
    pos_b: torch.Tensor,
    quat_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose B expressed in frame A: ``T_a^-1 . T_b``. Returns ``(pos, quat)``.

    WHY THIS EXISTS -- it is what lets the SDF and SAPU queries run batched
    against ONE static mesh instead of one rebuilt mesh per env.

    Both AutoMate callers loop over envs in Python. Their reasons differ, and
    only one of them applies to us:

    * ``get_max_interpen_dists`` loops because it indexes ``asset_indices[i]``
      -- a different socket mesh per env. It ALREADY transforms the held
      points into the socket frame first
      (``wp.transform_multiply(plug_transform, socket_inv_transform)``), so
      with our single fixture the loop body stops depending on ``i`` at all.
    * ``get_sdf_reward`` loops because it moves a COPY of the part mesh to
      each env's goal pose and calls ``mesh_copy.refit()`` -- a BVH rebuild per
      env per step. Its mesh is already env-independent (``wp_plug_mesh``, no
      asset index); only the goal pose varies, and D-034 makes the goal pose
      vary for us too.

    The refit is avoidable because the measured quantity is a DISTANCE between
    the part's sampled points and the part's own surface at the goal pose, and
    a distance is invariant when the same rigid transform is applied to both
    sides. Applying ``T_goal^-1`` to both turns "points at ``T_curr`` against a
    mesh at ``T_goal``" into "points at ``T_goal^-1 . T_curr`` against a mesh
    at the identity" -- the same number, against a mesh built and refitted
    ONCE. That is what this function computes.

    NAMED AS A DEVIATION, not smuggled in: the algebra is standard rigid-body
    composition, but no source performs this substitution, so the SUBSTITUTION
    is ours. It is exact rather than an approximation, and the training-PC gate
    must still confirm it numerically against a per-env reference run before
    SDF gates anything.

    Convention: ``rel_pos`` is ``R_a^T (pos_b - pos_a)`` and ``rel_quat`` is
    ``conj(quat_a) * quat_b``, both wxyz.
    """
    rel_pos = rotate_into_frame(axes_from_quat(quat_a), pos_b - pos_a)
    rel_quat = quat_mul(quat_conjugate(quat_a), quat_b)
    return rel_pos, rel_quat


@torch.jit.script
def part_tip_pose(
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    tip_offset_local: torch.Tensor,
    env_origins: torch.Tensor,
    entrance_pos: torch.Tensor,
    pocket_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Where the part's task frame sits relative to the pocket opening.

    Returns ``(tip_rel, depth, x_axis)``:

    * ``tip_rel`` (N, 3) -- the task-frame point measured from the pocket
      entrance, IN THE POCKET FRAME. Observation channels 12:15 (D-107 (4)).
    * ``depth`` (N,) -- insertion depth, positive INTO the pocket.
    * ``x_axis`` (N, 3) -- the part's own x-axis in the pocket frame, which is
      all ``yaw_cos_sin`` needs.

    Arguments, and why each one is an argument:

    * ``body_pos`` / ``body_quat`` (N, 3) / (N, 4 wxyz) -- the welded tool
      link in the WORLD frame, straight out of ``ArticulationData``.
    * ``tip_offset_local`` (3,) OR (N, 3) -- flange to the leading tool point
      in the body frame. This is D-069's task frame; the number lives in
      ``insertion_tasks_cfg.py`` (``FLANGE_TO_PART_BOTTOM``), never here.
      The (N, 3) form arrived with D-183: the OBSERVATION adds a per-episode
      grasp offset per env, while every other caller passes the shared (3,).
      Both go through the same line below and the (3,) result is unchanged.
    * ``env_origins`` (N, 3) -- so the result is env-local and the same
      arithmetic serves 1 env and 4096.
    * ``entrance_pos`` (N, 3) -- per-env since D-034: the fixture pose is
      randomised, so the opening is not a constant.
    * ``pocket_rot`` (N, 3, 3) -- the pocket's own axes as COLUMNS in the env
      frame, per-env since D-037. An untilted, unyawed pocket passes identity
      and every line below still holds, so there is no second branch to keep
      in step. The env owns the rotation; this function owns what to do with
      it.

    ONE argument, not two. The proxy env stored ``_tilt_rot`` AND
    ``_tilt_rot_t`` and used each in a different place, which is a convention
    with two homes and a silent failure if they are ever swapped. The
    transpose is taken here, once.

    DEPTH SIGN: ``-tip_rel[:, 2]``, so deeper is larger. Measured from the
    ENTRANCE, never from a table surface -- the proxy measured from the plate
    top, which in the workcell frame is 0.1823 m below the opening, and every
    depth read that much too deep (RT-17). ``insertion_tasks_cfg.py`` owns the
    entrance; this function only subtracts.

    NOTE what is deliberately NOT here: the four-corner gate, the alignment
    dot product, ``cos 4 phi`` and the ``below_plate`` flag. All four are gone
    with the proxy task -- D-106 (3) evaluates no separate alignment check,
    and D-113 (6) leaves exactly two exits.
    """
    part_axes = axes_from_quat(body_quat)
    # THE OFFSET IS RESHAPED TO A COLUMN FIRST, and that is load-bearing since
    # D-183. A bare ``matmul(part_axes, tip_offset_local)`` works only for the
    # (3,) form: matmul then promotes the vector and drops the added dimension
    # again. Handed an (N, 3) it reads the second operand as a MATRIX instead
    # -- which RAISES for N != 3 and silently returns a (3, 3, 3) block for
    # N == 3, i.e. wrong numbers with no error on a three-env probe. As a
    # column, (3,) broadcasts to (1, 3, 1) and (N, 3) becomes (N, 3, 1); both
    # give (N, 3, 1), and the (3,) result is bit-identical to the old line.
    tip_local = body_pos + torch.matmul(
        part_axes, tip_offset_local.reshape(-1, 3, 1)
    ).reshape(-1, 3) - env_origins
    tip_rel = rotate_into_frame(pocket_rot, tip_local - entrance_pos)
    depth = -tip_rel[:, 2]
    axes_in_pocket = torch.matmul(pocket_rot.transpose(1, 2), part_axes)
    x_axis = axes_in_pocket[:, :, 0]
    return tip_rel, depth, x_axis


@torch.jit.script
def seated_goal_pose(
    entrance_pos: torch.Tensor,
    fixture_quat: torch.Tensor,
    seat_quat_local: torch.Tensor,
    tip_offset_local: torch.Tensor,
    seat_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The part's SEATED goal pose, per env: ``(goal_pos, goal_quat)``.

    M2.4b step 2. The SDF reward term measures "the part's sampled points
    against the part's own surface AT THE GOAL" (D-109 (3), via
    ``relative_pose``), so the env needs the pose the seated part HAS. It is
    DERIVED from constants that already exist:

    * goal tip  = ``entrance_pos + pocket_rot . (0, 0, -seat_depth)``. The
      pocket z-axis points OUT of the pocket (``part_tip_pose`` reads depth as
      ``-tip_rel[:, 2]``), so the seat sits ``seat_depth`` along ``-z``.
    * goal orientation = ``fixture_quat * seat_quat_local``. The part fits
      its pocket in exactly ONE rotational position (D-107 (2)), and that
      position is the pocket orientation COMPOSED WITH the tool frame's own
      seated rotation -- the tool_link hangs INVERTED (its +z runs down the
      tool chain), so ``seat_quat_local`` is a 180 deg turn about pocket y
      (``insertion_tasks_cfg.SEATED_TOOL_QUAT_LOCAL``, MEASURED by RT-102).
      **CORRECTED 2026-08-30:** the first build returned ``fixture_quat``
      alone, and RT-102 falsified it -- sdf mean-outside 25.76 mm at a
      physically seated pose, goal flange 125 mm under the entrance.
    * goal body pos = goal tip minus the tip offset rotated by the GOAL
      orientation, the inverse of the ``tip = body_pos + R . tip_offset``
      line in ``part_tip_pose``. With the flipped goal rotation the flange
      lands ABOVE the entrance (tip offset +0.152 along tool z, tool z
      pointing down), where the real flange is.

    OWN CONSTRUCTION, named per the handoff (2026-08-30): no published task
    computes a goal pose from a pocket entrance -- the Isaac Lab AutoMate code
    equates target pose with socket pose because its meshes are drawn
    coincident (see the SAPU header, adaptation (a)). The offline check
    proves the round trip: feeding the goal pose back through
    ``part_tip_pose`` must report exactly ``seat_depth`` and zero lateral
    offset, for a yawed AND tilted pocket, with the flip in place.

    Frames: ``entrance_pos`` and the returned ``goal_pos`` are ENV-LOCAL, the
    frame the env's buffers use. The part OBJ is exported in the ``tool_link``
    frame (RT-85), so this is the goal pose OF THAT BODY -- both sides of the
    SDF query must be ``tool_link`` poses in the same frame.

    Owns no number: ``seat_depth`` is ``insertion_tasks_cfg.POCKET_SEAT_DEPTH``,
    ``tip_offset_local`` is ``FLANGE_TO_PART_BOTTOM`` (as ``cfg.peg_tip_offset``)
    and ``seat_quat_local`` is ``SEATED_TOOL_QUAT_LOCAL``, all arriving from
    the caller. ``seat_quat_local`` may be ``(1, 4)`` or ``(N, 4)``.
    """
    goal_quat = quat_mul(fixture_quat, seat_quat_local)
    pocket_rot = axes_from_quat(fixture_quat)
    # pocket_rot . (0, 0, -seat_depth) is just -seat_depth times column 2.
    goal_tip = entrance_pos - seat_depth * pocket_rot[:, :, 2]
    goal_pos = goal_tip - torch.matmul(axes_from_quat(goal_quat), tip_offset_local)
    return goal_pos, goal_quat


# ===========================================================================
#  Reward (D-109, D-114)
# ===========================================================================


@torch.jit.script
def squash(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """The Factory/FORGE squashing function ``1 / (exp(a*x) + b + exp(-a*x))``.

    D-109 point (6): this REPLACES IndustReal's ``-log`` mapping rather than
    guarding it. The measured quantity stays the SDF distance; only the
    mapping is swapped. Two properties matter and are checked offline:

    * finite at ``x = 0``, where it takes ``1/(2+b)``. IndustReal's ``-log``
      is ``inf`` at a perfect seat, and rsl_rl 3.x has neither a NaN check nor
      a reward clamp -- an ``inf`` would surface one iteration later inside
      the advantage normalisation, far from its cause.
    * strictly positive everywhere, which is what kills the whole
      suicide/early-abort exploit family: every step pays something.

    Written in the algebraically identical but overflow-free form
    ``e / (1 + b*e + e^2)`` with ``e = exp(-|a*x|)``. The fine kernel has
    ``a = 10408``, so ``exp(a*x)`` overflows for any ``x`` past ~7 cm and the
    naive form would return 0 through an inf, i.e. the right answer by
    accident and a warning per step.

    NOTE for the report: the squashing function is an Isaac Lab CODE artefact,
    not a Factory-paper formula, and the FORGE curve exists in two versions
    (with / without prefactor). The Isaac Lab code has no prefactor. Cite the
    code.
    """
    e = torch.exp(-torch.abs(a * x))
    return e / (1.0 + b * e + e * e)


@torch.jit.script
def kernel_sum(
    sdf_dist: torch.Tensor,
    a_coarse: float,
    b_coarse: float,
    a_mid: float,
    b_mid: float,
    a_fine: float,
    b_fine: float,
) -> torch.Tensor:
    """Three squashed kernels over ONE SDF distance, peaking at 1.0 together.

    D-109 point (3): one SDF-distance term replaces the proxy's separate
    approach, depth, yaw and align terms -- it measures position AND rotation
    on a single manifold, so there is no weight between them to defend
    (Factory: "obviating tuning").

    D-109 point (9): the widths are SOLVED, not tuned, via dm_control's
    ``tolerance()`` (Tassa et al. 2018): ``a = arccosh(1/0.1)/margin =
    2.9932/margin``, at the DMC convention ``value_at_margin = 0.1``.

    WHAT THE THREE MARGINS ARE, corrected 2026-09-01 (inbox entry
    "Reward-Ueberarbeitung" 2026-09-01, p1-konzept-messung, point (1)): the mid
    and fine margins are measured TOLERANCES of this task (success band, cross
    play), the coarse one is a measured REACH -- the SDF distance at the
    workcell home pose. That is dm_control's own split: ``bounds`` is the
    tolerance zone and ``margin`` the decay reach outward from it. The coarse
    margin used to carry "the start scatter", which is neither, and the
    resulting kernel was numerically dead everywhere an episode actually
    starts. The numbers live in ``insertion_tasks_cfg`` and arrive as
    arguments; nothing here asserts one.

    The peaks sum to exactly 1.0 by construction: ``1/(2+b)`` per kernel, so
    ``b = 2, 2, 0`` gives ``0.25 + 0.25 + 0.5``. That identity is the check
    that catches a mistyped width or a swapped ``b``.
    """
    return (
        squash(sdf_dist, a_coarse, b_coarse)
        + squash(sdf_dist, a_mid, b_mid)
        + squash(sdf_dist, a_fine, b_fine)
    )


@torch.jit.script
def sapu_reward_scale(interpen_dist: torch.Tensor, interpen_thresh: float) -> torch.Tensor:
    """SAPU: scale the return by interpenetration (D-109 point (10)).

    Tunneling is the ONE exploit that survives the Block-5 term set -- every
    other one dissolves because the SDF distance grows there. SAPU is what
    answers it, so it is mandatory, not an extra.

    Form ported from IsaacGymEnvs ``get_sapu_reward_scale``: below the
    threshold the return is scaled by ``1 - tanh(d / thresh)``, at or above it
    the return is DISCARDED. Isaac Lab 2.3.2 has the underlying Warp kernel
    (``get_interpen_dist``) but no caller and no ``interpen_thresh`` at all.

    The threshold is HALF THE CROSS PLAY, 0.2938 mm -- the D-106 (3) rule,
    re-anchored on the doubled play and decided 2026-08-30. It is not
    IndustReal's 1 mm, which is larger than this task's entire cross play. The
    module owns no number: it arrives as ``interpen_thresh`` from
    ``insertion_tasks_cfg.INTERPEN_THRESH``, which derives it from ``PLAY_X``.

    CORRECTED 2026-08-30. This paragraph read "the threshold is 0.144 mm --
    the same number as the D-052 headline"; that was half the SUPERSEDED play
    (0.2876 mm), and D-121 doubled the input on 2026-08-28. The RULE never
    moved -- only its input did.

    The measurement is a pure Warp MESH query, never PhysX, so contact and
    rest offsets and solver settings do not enter. Two conditions on it are
    scene-side and are NOT verifiable here: the fixture SDF resolution (D-128
    keeps 2048; the old ">= 1536" floor died with the 0.144 mm threshold and
    is not replaced by a new floor -- against 0.2938 mm the arithmetic floor
    would be 644, below even D-062's 1024, so the resolution rests on D-128's
    margin argument alone), and the pocket mesh must be watertight or the sign
    flips silently.
    """
    scale = 1.0 - torch.tanh(interpen_dist / interpen_thresh)
    return torch.where(interpen_dist >= interpen_thresh, torch.zeros_like(scale), scale)


# THE ROW ORDER OF ``compute_reward_terms_insertion`` IS THIS MODULE'S OWN
# FACT, and this tuple is its only home (2026-09-02, after RT-134). The env
# accumulates one per-episode sum per row and logs it as
# ``Episode_Reward/<name>``; ``compute_rewards_insertion`` is the sum of the
# rows. RT-131 and RT-134 could show only rsl_rl's aggregate return, and the
# D-164 arithmetic (holding pays 58.89, an abort at step 20 paid 3.60) had
# to be redone offline -- with the rows on the curve each term is a number
# read off the run.
REWARD_TERMS: tuple = (
    "kernels",       # kernel_sum * SAPU scale
    "engaged",       # w_engaged bonus * SAPU scale
    "success",       # w_success bonus * SAPU scale
    "progress",      # w_progress * metres of NEW gated max depth * SAPU scale (D-165)
    "time",          # time_penalty_per_step, every step
    "action_rate",   # -action_rate_scale * ||a_t - a_(t-1)||
    "success_lump",  # step_task * steps_remaining on the success step
    "abort",         # abort_payment on the force-abort step
    "tilt_shaping",  # gamma * Phi(s') - Phi(s), Phi = w_tilt * sech(a * theta) vs the POCKET axis; Phi(s') = 0 on the success step (RT-171)
)


@torch.jit.script
def tilt_cos_theta(body_quat: torch.Tensor, pocket_rot: torch.Tensor) -> torch.Tensor:
    """``cos theta`` (N,): the part's +z axis against the pocket's -z axis.

    +1 when the part hangs straight down the pocket axis (the seated
    orientation), 0 at 90 deg, -1 for a part pointing OUT of the pocket. The
    minus sign is the one ``insertion_env._peg_geometry`` uses for its
    ``alignment`` (its ``[:, 2, 2]`` comment): the pocket z-axis points OUT of
    the pocket (``part_tip_pose`` reads depth as ``-tip_rel[:, 2]``), so a
    seated part's own +z is ANTIPARALLEL to it. The review of the RT-171 plan
    (``HANDOFF-RL.md`` section Open 2d (i)) caught the plan writing "+z
    against +z", which would have paid for an upside-down part.

    Against the POCKET axis, never the world vertical: ``tool_axis_tilt``
    below measures against world DOWN and is the wrong reference as soon as
    ``fixture_tilt_noise_rad > 0`` (hacking checklist: absolute world poses in
    the reward break under pose randomisation). ``pocket_rot`` is the same
    (N, 3, 3) column matrix ``part_tip_pose`` takes; the two lines here are
    that function's own ``axes_in_pocket`` arithmetic, repeated rather than
    returned, because ``part_tip_pose``'s three-tuple is pinned by the
    env-wiring check and by every offline caller.
    """
    axes_in_pocket = torch.matmul(pocket_rot.transpose(1, 2), axes_from_quat(body_quat))
    return -axes_in_pocket[:, 2, 2]


@torch.jit.script
def tilt_potential(cos_theta: torch.Tensor, w_tilt: float, a_tilt: float) -> torch.Tensor:
    """``Phi(s) = w_tilt * sech(a_tilt * theta)`` (N,), the alignment potential (RT-171).

    ``sech(a x) = 1 / cosh(a x) = 2 * squash(x, a, 0)``, so this is the
    D-109 (6) curve at ``b = 0`` and the D-109 (9) width rule
    (``a = arccosh(10) / margin``, ``insertion_tasks_cfg.KERNEL_A_TILT``):
    ``Phi = w`` upright, ``0.1 w`` at the margin, monotone in between, and
    its pull ``-dPhi/dtheta`` GROWS toward upright (maximum at
    ``theta = atanh(1/sqrt 2) / a``), so there is no trap halfway.

    ``w_tilt = 0`` makes this exactly zero, and the shaping row is then
    ``0 - 0``: the regression switch that reproduces the pre-RT-171 reward
    bit for bit.
    """
    theta = torch.acos(torch.clamp(cos_theta, min=-1.0, max=1.0))
    return w_tilt * 2.0 * squash(theta, a_tilt, 0.0)


@torch.jit.script
def compute_reward_terms_insertion(
    sdf_dist: torch.Tensor,
    interpen_dist: torch.Tensor,
    engaged: torch.Tensor,
    success: torch.Tensor,
    force_abort: torch.Tensor,
    action_rate: torch.Tensor,
    steps_remaining: torch.Tensor,
    depth_progress: torch.Tensor,
    tilt_phi: torch.Tensor,
    tilt_phi_prev: torch.Tensor,
    a_coarse: float,
    b_coarse: float,
    a_mid: float,
    b_mid: float,
    a_fine: float,
    b_fine: float,
    w_engaged: float,
    w_success: float,
    w_progress: float,
    interpen_thresh: float,
    abort_payment: float,
    time_penalty_per_step: float,
    action_rate_scale: float,
    shaping_gamma: float,
) -> torch.Tensor:
    """The per-step reward as a ``[len(REWARD_TERMS), N]`` stack of its terms.

    THE ALIGNMENT SHAPING ROW (RT-171, 2026-09-06). ``tilt_phi`` is
    ``tilt_potential`` of THIS step's pose (cached by ``_get_dones``),
    ``tilt_phi_prev`` the potential of the pose the policy was SHOWN before
    acting (cached by ``_get_observations``, so the first step of an episode
    reads the reset observation and never a stale link pose --
    ``HANDOFF-RL.md`` section Open 2d (iii)). The row is
    ``shaping_gamma * Phi(s') - Phi(s)`` (Ng, Harada & Russell 1999), with
    ``Phi(s') = 0`` on the SUCCESS step, the only true terminal (D-164): the
    theorem needs the absorbing state at one fixed potential. A truncation
    (timeout, force abort) keeps the real ``Phi(s')`` because rsl_rl
    bootstraps it (D-116 / D-164). Discounted over an episode the row
    telescopes to ``gamma^T * Phi(s_T) - Phi(s_0)``, path-independent, so it
    cannot carry the success-over-failure ordering -- the lump does -- and it
    cannot be farmed by tilting and righting in a circle. Price of the
    terminal convention: the success step pays ``-Phi(s)``, about ``-w_tilt``
    against a lump of ~615. ``shaping_gamma`` is PPO's gamma and has ONE
    home, ``agents/rsl_rl_ppo_cfg.py``; the env cfg reads it from there.

    NOT in ``step_task``: like ``progress``, the row is added after the lump
    is computed, or the success step would pay ``-w * steps_remaining``.

    Same arithmetic as before 2026-09-02; only the return changed from the
    sum to the rows, so the env can log each term per episode. The sum is
    ``compute_rewards_insertion`` below -- one home for the formula.

    THE CONTACT PENALTY AND THE LEASH ON THE SDF DISTANCE ARE GONE (D-184,
    2026-09-11). Both were added on 2026-09-03, before the switch to OSC;
    the training commands of RT-156 and RT-171 to RT-179 set both scales to
    0.0. Force is bounded by the OSC command cap (D-166) and the force abort;
    the tip target by the clamp box (D-180).

    THE DEPTH-PROGRESS TERM (D-165, 2026-09-02). ``depth_progress`` is the
    metres of NEW gated maximum depth the part reached this step -- the env
    computes it from its ``_max_depth`` buffer, so it is zero unless the part
    got deeper than ever before in this episode -- and it is paid
    ``w_progress`` per metre. The square-peg proxy's ``progress`` term
    (``proxytask_env.py:193-195``, ``w_depth = 100``), taken as it stands:
    telescoping over the episode to ``w_progress * max_depth``, so it cannot
    be farmed by going up and down. It exists because RT-135 measured the
    D-109 kernels FLAT beside the stage-2 opening (5.82e-5 per step across
    4.88 mm of lateral error, at an sdf of 14.3 mm) and RT-134 showed the
    policy parking there. It is SAPU-scaled like the task rows (a tunnelled
    part reads as depth) and sits OUTSIDE ``step_task``, so the success lump
    never multiplies a one-off delta by the steps remaining.

    THE WHOLE PER-STEP REWARD (D-109, D-114, and the 2026-09-01 revision).

    ``kernel_sum`` + two display bonuses, SAPU-scaled; MINUS a per-step time
    penalty and an action-rate penalty; PLUS either the remaining-return payout
    of a success termination or the fixed payment of a force abort.

    THE FIVE TERMS AND WHERE EACH COMES FROM. Home of the four new ones: inbox
    entry "Reward-Ueberarbeitung" 2026-09-01 (p1-konzept-messung).

    * ``task_return * scale`` -- unchanged (D-109, D-109 (10)).
    * ``time_penalty_per_step`` -- point (6), R_time = -1/T, Brahmbhatt et al.,
      ICRA 2023 (arXiv:2301.12587) Sec. III. Paid on EVERY step, terminal steps
      included; a step that ends the episode still took its time. It breaks
      D-109 (11)'s "always positive" on purpose, and the decision carries the
      two inequalities that keep the suicide exploit closed: living still earns
      kernel > 0 everywhere reachable, and the rule caps any per-step penalty
      at ``abort_payment / T``.
    * ``- action_rate_scale * ||a_t - a_(t-1)||`` -- point (8), the FORGE form
      (``forge_tasks_cfg.py:15``); Brahmbhatt has it as R_delta-a. The NORM
      arrives precomputed (``action_rate``): this module never sees an action
      buffer, and the env owns the per-env previous action.
    * the SUCCESS PAYOUT -- point (7), and OWN CONSTRUCTION, labelled as such.
      Success terminates (see ``compute_dones``), so the seated state becomes
      ABSORBING and its remaining income is paid as a lump:
      ``step_task * steps_remaining``. Together with the step's own
      ``step_task`` the episode is paid exactly what holding this state to the
      old timeout would have paid, so the UNDISCOUNTED return is identical to
      the previous run-to-timeout scheme; at gamma = 0.99 early success is
      slightly preferred. What is NOT in the lump is the time penalty of the
      forfeited steps -- the decision writes the lump as ``task_return(t) *
      (T - t)`` -- so ending early additionally SAVES those, which is the
      intended pressure and is stated rather than hidden.
    * ``abort_payment`` -- unchanged in form (D-114, Beltran-Hernandez 2020),
      now DECIDED in magnitude (point (4)): one kernel peak, a scale anchor.

    THE TIE-BREAK IS A CODE-LEVEL CHOICE, made here and stated: if the force
    abort and the success predicate fire on the SAME step, SUCCESS WINS -- the
    lump is paid and the abort payment is not. The part is seated; the D-052
    band plus the SAPU filter say so, and charging for the contact that seated
    it would price the solution. The decision text does not rule on the
    collision, so this is ours.

    HOW CAN A POLICY EXPLOIT THIS? -- one answer per term, per the project
    rule. Five of the proxy's six exploit questions dissolve here, and they
    dissolve for the SAME reason, which is worth stating once: the reward is a
    monotone function of ONE distance on ONE manifold, so any pose that is not
    closer to the seat scores lower, whatever direction it fails in.

    * **Tunneling** -- the one that survives. SAPU answers it: past half the
      cross play (0.2938 mm, D-106 (3)) of interpenetration the entire task
      return is discarded, bonuses included. Bonuses INSIDE the scaled term is the point: a tunnelled part
      reads "engaged" and "success" geometrically, so leaving the bonuses
      outside would pay 2.0 per step for cheating.
    * **Gate-farming** -- impossible: there is no gate and no progress term to
      re-collect. D-109 point (4) prescribes no sequence at all; if the other
      side of the part is easier, the policy may take it.
    * **Submarining / side-parking / yaw-parking** -- the SDF distance grows
      in all three, so the reward falls. These were three separate terms in
      the proxy and are one term here.
    * **Suicide / early abort** -- REARGUED 2026-09-01, because the time
      penalty broke the old one-line answer ("every step pays strictly
      positive, so ending early always forfeits"). The decision's inequality
      (i) replaces it: living earns ``kernel > 0`` everywhere reachable (0.23
      at the rung-0 start, 0.025 at the home pose) against a time cost of
      ``1/T = 0.0039`` per step, so a full episode's time bill of 1.0 never
      beats living. The proxy's remaining-reward trick (``_last_remaining``) is
      still NOT ported (D-114); the force abort still forfeits the remainder,
      and it now forfeits a payout as well.
    * **Existing instead of solving** -- NOT defeated, named. Since everything
      is positive, merely existing is rewarded. The gradient still points at
      the seat (Factory has the same property), and the offline check pins the
      seat-to-far ratio so a change in it becomes visible. Goes into the
      report as a property, not as a defect.

    COMPOSITION, decided here because no source composes these two: SAPU
    scales the TASK return only. Its source (IsaacGymEnvs
    ``get_sapu_reward_scale``) scales the task reward and knows no abort
    payment; the abort payment's source (Beltran-Hernandez 2020, via D-114)
    knows no SAPU. Each keeps its own semantics, so the payment is added AFTER
    the scaling and a tunnelling episode still pays for its abort in full.

    ``abort_payment`` is expected to be NEGATIVE and its magnitude is ``[open]``
    (D-114): it falls together with ``F_max`` at the scripted-insertion gate.
    Nothing here asserts a value for it -- ``insertion_env_cfg.py`` refuses to
    build an env while it is unset.
    """
    kernels = kernel_sum(sdf_dist, a_coarse, b_coarse, a_mid, b_mid, a_fine, b_fine)
    zero = torch.zeros_like(kernels)
    # D-109 point (5): both bonuses are paid PER TIMESTEP without a latch,
    # weight 1.0 each (Isaac Lab factory pattern). Named caveat, carried from
    # the decision: over an episode these two are about two thirds of the
    # total return, so the "sparse bonus" is in fact the dominant term.
    #
    # THE HIDDEN TIME REWARD IS GONE, 2026-09-01. This comment used to end:
    # "they are also a hidden TIME reward, because the episode never ends on
    # success (D-113): seating at step 50 pays 190x, at step 230 only 10x."
    # That reading was correct for the old scheme and is now false twice over.
    # Success ENDS the episode (point (7)), so the seated bonus is paid once
    # and then bought out at the same rate for every remaining step -- early
    # and late seating earn the same undiscounted total, and the incentive to
    # seat early is the explicit time penalty plus gamma, not an accident of
    # the horizon.
    engaged_pay = torch.where(engaged, torch.full_like(kernels, w_engaged), zero)
    success_pay = torch.where(success, torch.full_like(kernels, w_success), zero)
    scale = sapu_reward_scale(interpen_dist, interpen_thresh)
    # SAPU scales the TASK rows only -- the composition decided above, written
    # as ROWS since 2026-09-02 so the env can log each one. The step's task
    # value is their sum, ONE computation: the lump below reads it, and the
    # stack returns the same rows, so the logged terms and the paid step value
    # cannot drift apart. The time, action-rate, abort and tilt-shaping rows
    # are added AFTER the scale, each keeping its own semantics: a tunnelling
    # episode still pays its abort in full, still pays for its time, and a discarded task return also
    # makes its own payout zero, which is the point of computing the lump from
    # the SCALED step value.
    kernels_row = kernels * scale
    engaged_row = engaged_pay * scale
    success_row = success_pay * scale
    step_task = kernels_row + engaged_row + success_row
    # D-165: paid per metre of new gated max depth, SAPU-scaled, and NOT part
    # of step_task -- the lump below reads step_task, and a one-off progress
    # delta must not be bought out for the rest of the episode.
    progress_pay = w_progress * depth_progress * scale
    time_pay = torch.full_like(step_task, time_penalty_per_step)
    rate_pay = -action_rate_scale * action_rate
    lump = torch.where(success, step_task * steps_remaining, zero)
    payment = torch.full_like(step_task, abort_payment)
    # The tie-break: success beats the force abort on a shared step.
    abort_pay = torch.where(force_abort & (~success), payment, zero)
    # RT-171: potential-based alignment shaping, gamma * Phi(s') - Phi(s).
    # The success step is the absorbing state and reads Phi(s') = 0; every
    # other step, truncations included, reads the real potential.
    tilt_phi_next = torch.where(success, zero, tilt_phi)
    tilt_pay = shaping_gamma * tilt_phi_next - tilt_phi_prev
    # One row per REWARD_TERMS entry, in that order.
    return torch.stack([
        kernels_row,
        engaged_row,
        success_row,
        progress_pay,
        time_pay,
        rate_pay,
        lump,
        abort_pay,
        tilt_pay,
    ], dim=0)


@torch.jit.script
def compute_rewards_insertion(
    sdf_dist: torch.Tensor,
    interpen_dist: torch.Tensor,
    engaged: torch.Tensor,
    success: torch.Tensor,
    force_abort: torch.Tensor,
    action_rate: torch.Tensor,
    steps_remaining: torch.Tensor,
    depth_progress: torch.Tensor,
    tilt_phi: torch.Tensor,
    tilt_phi_prev: torch.Tensor,
    a_coarse: float,
    b_coarse: float,
    a_mid: float,
    b_mid: float,
    a_fine: float,
    b_fine: float,
    w_engaged: float,
    w_success: float,
    w_progress: float,
    interpen_thresh: float,
    abort_payment: float,
    time_penalty_per_step: float,
    action_rate_scale: float,
    shaping_gamma: float,
) -> torch.Tensor:
    """The whole per-step reward: the sum of ``compute_reward_terms_insertion``.

    Kept under its old name and signature so every offline check and the
    seated identity test read the same number the env pays. The env itself
    calls the terms function and sums the rows, so that the logged rows and
    the paid reward cannot drift apart.
    """
    return torch.sum(compute_reward_terms_insertion(
        sdf_dist, interpen_dist, engaged, success, force_abort, action_rate,
        steps_remaining, depth_progress, tilt_phi, tilt_phi_prev,
        a_coarse, b_coarse, a_mid, b_mid,
        a_fine, b_fine, w_engaged, w_success, w_progress, interpen_thresh,
        abort_payment, time_penalty_per_step, action_rate_scale,
        shaping_gamma,
    ), dim=0)


# ===========================================================================
#  Observation (D-107, D-114)
# ===========================================================================

# THE LAYOUT IS THIS MODULE'S OWN FACT, and this table is its only home.
# 28 channels: the Block-3 layout of 25 (D-107) plus the three force
# components D-114 added when it overturned D-089. The order is the proxy's,
# extended at the END so no existing index moves -- the same reason D-037
# appended the pocket quaternion last.
#
# Nothing may read these by counting; ``assemble_observation`` builds the
# vector in exactly this order and the offline check proves the two agree by
# filling every block with its own marker value.
OBS_SLICES: dict = {
    "joint_pos": (0, 6),      # D-107 (1)
    "joint_vel": (6, 12),     # finite difference, proxy pattern
    "tip_rel": (12, 15),      # tip vs pocket opening, PART-anchored, D-107 (4)
    "ee_quat": (15, 19),      # world-frame EE quaternion, wxyz
    "yaw_cos_sin": (19, 21),  # cos phi / sin phi, D-107 (2) -- NOT cos 4 phi
    "pocket_quat": (21, 25),  # sign-canonicalised, D-037
    "force": (25, 28),        # three force components, EMA-smoothed, D-114
}
OBS_DIM: int = 28

# THE TWO OBSERVATION MODES (D-188, 2026-09-13). ``force`` IS the table
# above -- the D-114 layout, 28 wide, the default every run before D-188
# trained under. ``wrench`` APPENDS the three torque components of the same
# joint wrench, so no existing index moves and every reader that addresses
# a block by name (``OBS_SLICES["force"]`` in zero_agent, scripted_insert,
# seat_probe, tilt_recovery_probe, check_seated_success) stays valid in
# both modes. Per block: unit, frame, scaling.
#
#   block        unit          frame                              scaling
#   joint_pos    rad           joint space                        none
#   joint_vel    rad/s         joint space, finite difference     none
#   tip_rel      m             pocket frame, PART-anchored        none
#   ee_quat      --            world, wxyz                        none
#   yaw_cos_sin  --            pocket frame                       none
#   pocket_quat  --            world, wxyz, w >= 0                none
#   force        N             force-link PARENT body frame,      none
#                              gravity-tared, EMA
#   torque       N m           same frame as force, about the     none
#                              PARENT link origin (assumed, see
#                              gravity_tare_torque), gravity-
#                              tared, EMA -- wrench mode only
#
# NO scaling anywhere: rsl_rl's ``actor_obs_normalization`` owns that
# (``agents/rsl_rl_ppo_cfg.py``), which is why N m may stand beside N.
OBS_MODES: tuple = ("force", "wrench")
TORQUE_WIDTH: int = 3


def obs_slices(mode: str) -> dict:
    """The slice table of one observation mode -- ``OBS_SLICES`` plus, in
    ``wrench`` mode, ``torque`` appended at the end.

    Refuses an unknown mode: a typo in ``env.obs_wrench_mode=`` must not
    silently run the other layout.
    """
    if mode not in OBS_MODES:
        raise ValueError(f"obs mode {mode!r} is not one of {OBS_MODES}")
    out = dict(OBS_SLICES)
    if mode == "wrench":
        out["torque"] = (OBS_DIM, OBS_DIM + TORQUE_WIDTH)
    return out


def obs_dim(mode: str) -> int:
    """Width of one observation mode: the end of its last slice."""
    return max(hi for _, hi in obs_slices(mode).values())


def obs_version(mode: str) -> str:
    """One string naming the layout a checkpoint was trained on: ``force-28``
    or ``wrench-31``. Printed by the startup report and checked against the
    checkpoint's actor width by ``scripts/tools/checkpoint_width.py``."""
    return f"{mode}-{obs_dim(mode)}"


def noise_masks(
    *,
    pocket_pos_std_m: float,
    force_std_n: float,
    torque_std_nm: float = 0.0,
    mode: str = "force",
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two per-channel sigma vectors of the observation noise (D-182).

    Returns ``(step_std, bias_std)``, both ``(obs_dim(mode),)``:

    * ``step_std`` -- the per-STEP Gaussian, non-zero only on ``force`` and,
      in ``wrench`` mode, on ``torque`` (its own sigma in N m, D-188 -- the
      3.5 N are NOT carried over to N m). A force reading jitters at every
      sample, so this is sensor noise.
    * ``bias_std`` -- the per-EPISODE UNIFORM bias, non-zero only on
      ``tip_rel``. Despite the name it holds the HALF WIDTH of the uniform
      draw, not a sigma (user, 2026-09-15; the name stays by user decision,
      see ``insertion_env_cfg.obs_noise_pocket_pos_std_m``).
      The pocket is mislocated once and then stays mislocated
      for the episode, so this is a localisation error, not jitter. Which is
      exactly why the two cannot share one tensor.

    A torque sigma in ``force`` mode is REFUSED, not ignored: there is no
    torque channel to put it on, and a sigma that silently lands nowhere is
    a run whose log says "torque noise 0.2" while the policy sees none.

    THIS IS OUR WHOLE SHARE of the noise model. Isaac Lab owns the hook
    (``DirectRLEnvCfg.observation_noise_model``), the reset schedule and both
    noise functions; ``obs_noise.InsertionObsNoise`` only hands ``step_std``
    to ``GaussianNoiseCfg`` and ``bias_std`` to ``UniformNoiseCfg``. The masks are built FROM ``OBS_SLICES``
    rather than from typed indices, because a sigma on the wrong block is
    invisible in a training curve -- the run is simply noisy somewhere else.

    NOT ``@torch.jit.script``: it reads the module-level ``OBS_SLICES`` dict,
    and it runs ONCE per env build rather than per step, so the hot-path rule
    does not apply. That also keeps it executable under the offline stand-in,
    where ``scripts/check_insertion_math.py`` checks it against the table.

    KEYWORD-ONLY, and the ``*`` is the whole point of this paragraph. Both
    sigmas are bare floats, so a POSITIONAL call carries nothing but their
    order: swapped, the model does not crash -- it trains with a 3.5 m
    pocket bias and 0.0025 N of force jitter, and every log line still looks
    plausible. That is the class of defect rule L-08 was written for
    (``docs/reference/pruefregeln.md``): a sweep for the parameter NAME does
    not see a positional argument. Keyword-only makes the swap unwritable
    rather than merely detectable -- a positional call is a ``TypeError`` at
    env build, before the first reset. The offline check
    ``scripts/check_insertion_math.py`` holds the ``*`` in place ("noise_masks
    refuses positional sigmas") and pins each keyword to its block BY VALUE.

    ``device`` is passed through from ``NoiseModel.__init__``. The library
    would move a CPU tensor to the data's device on the first call anyway
    (``noise_model.gaussian_noise`` lines 88-90), so this only avoids that
    first transfer and keeps the buffer where the observation already is.
    """
    slices = obs_slices(mode)
    if "torque" not in slices and torque_std_nm != 0.0:
        raise ValueError(
            f"torque_std_nm={torque_std_nm!r} in obs mode {mode!r}: this layout has no "
            "torque channel, so the sigma would land nowhere. Set it to 0.0 or run "
            "obs_wrench_mode='wrench'."
        )
    width = obs_dim(mode)
    step_std = torch.zeros(width, device=device)
    bias_std = torch.zeros(width, device=device)
    lo, hi = OBS_SLICES["force"]
    step_std[lo:hi] = force_std_n
    if "torque" in slices:
        lo, hi = slices["torque"]
        step_std[lo:hi] = torque_std_nm
    lo, hi = OBS_SLICES["tip_rel"]
    bias_std[lo:hi] = pocket_pos_std_m
    return step_std, bias_std


@torch.jit.script
def canonicalize_quat(q: torch.Tensor) -> torch.Tensor:
    """Force ``w >= 0`` on a wxyz quaternion batch (D-037).

    ``q`` and ``-q`` are the same rotation, so an uncanonicalised channel
    hands the policy two different inputs for one physical state. Written as a
    branch on the sign rather than a multiply by ``sign(w)``, because
    ``sign(0) = 0`` would zero the quaternion outright at exactly 180 degrees.
    """
    flip = q[:, 0:1] < 0.0
    return torch.where(flip, -q, q)


@torch.jit.script
def yaw_cos_sin(x_axis: torch.Tensor) -> torch.Tensor:
    """Encode the part's yaw as ``(cos phi, sin phi)`` -- D-107 point (2).

    NOT ``cos 4 phi / sin 4 phi``. The proxy peg was a square: four
    orientations fitted the pocket, so its encoding had to be C4-invariant and
    the reward carried a matching C4 yaw term. The real part fits its pocket
    in exactly ONE rotational position (the wave contour is unique), so a
    C4-invariant encoding would map four physically different states onto one
    input -- three of them impossible. The whole C4 apparatus goes, on both
    sides: this encoding here, and the separate yaw reward term, which D-109
    dissolved into the single SDF-distance term.

    ``phi`` is the yaw of the part's own x-axis about the pocket z-axis, taken
    from the axis projected into the pocket XY plane. A part tilted to exactly
    vertical projects to the zero vector, where ``atan2(0, 0)`` is 0 and the
    encoding reads (1, 0); that pose is far outside the alignment the task
    ever reaches, and 0 is a defined value rather than a NaN.
    """
    phi = torch.atan2(x_axis[:, 1], x_axis[:, 0])
    return torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1)


@torch.jit.script
def ema_update(prev: torch.Tensor, new: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exponential moving average, ``alpha`` weighting the NEW sample.

    D-114 puts the force channel through an EMA at 0.25 (the FORGE pattern).
    The number lives in the config, not here; what lives here is the
    convention, because "EMA 0.25" is ambiguous until someone says which side
    the 0.25 sits on. Fixed here as the weight of the new sample, which is the
    standard reading of a smoothing factor: ``alpha = 1`` passes the raw
    signal through, ``alpha = 0`` freezes.

    CONFIRMED against the source (2026-08-28), so this needs no training-PC
    run: ``forge_env.py:99-100`` computes
    ``alpha * force_sensor_world + (1 - alpha) * force_sensor_world_smooth``,
    i.e. ``alpha`` weights the NEW sample, exactly as written here. The
    earlier note claiming the Isaac Lab source is unavailable on the dev
    laptop was wrong: it is at ``C:\\IsaacLab``, VERSION 2.3.2.

    Two things come with that read and belong to the env, not here:
    ``forge_env.py:331`` clears the smoothing buffer to ZERO in
    ``_reset_idx`` rather than seeding it with the first sample, which also
    keeps a stale reset-step reading out of the seed; and the raw wrench is
    in the PARENT BODY frame and is handed over UNROTATED (inbox entry "The
    force wrench is in the parent body frame, not the world frame").
    """
    return alpha * new + (1.0 - alpha) * prev


@torch.jit.script
def assemble_observation(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    tip_rel: torch.Tensor,
    ee_quat: torch.Tensor,
    yaw_cs: torch.Tensor,
    pocket_quat: torch.Tensor,
    force: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the seven blocks into the 28-channel policy observation.

    The order IS ``OBS_SLICES``. Written as one ``cat`` rather than assembled
    by index so there is no second place a channel offset can be wrong; the
    offline check fills each block with its own marker and reads the named
    slices back, which fails on a swapped block and on a wrong slice table
    alike.

    Every block arrives finished. In particular ``tip_rel`` is PART-anchored
    (D-107 point (4)): it is measured from the part's own task frame, not from
    the flange, which is what makes the per-reset grasp offset visible to the
    policy without adding a single channel. And ``force`` is already
    EMA-smoothed and already reduced to its three FORCE components -- the
    wrench's three torque components are NOT handed to the policy in this,
    the ``force`` layout (D-114, FORGE pattern); ``assemble_observation_wrench``
    appends them (D-188). It is also UNROTATED; which frame that is, and why
    nothing rotates it, is owned by the inbox entry "The force wrench is in
    the parent body frame, not the world frame".
    """
    return torch.cat(
        (joint_pos, joint_vel, tip_rel, ee_quat, yaw_cs, pocket_quat, force), dim=-1
    )


@torch.jit.script
def assemble_observation_wrench(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    tip_rel: torch.Tensor,
    ee_quat: torch.Tensor,
    yaw_cs: torch.Tensor,
    pocket_quat: torch.Tensor,
    force: torch.Tensor,
    torque: torch.Tensor,
) -> torch.Tensor:
    """The ``wrench`` layout (D-188): the ``force`` layout with the three
    torque components APPENDED. Written as the 28-wide cat plus one more
    block rather than as a second eight-tuple, so the first 28 channels are
    the same function's output in both modes by construction.

    ``torque`` arrives finished exactly like ``force``: gravity-tared
    (``gravity_tare_torque``), EMA-smoothed with the same alpha, unrotated,
    in the force link's parent body frame. Not an ``Optional`` on
    ``assemble_observation`` -- this module imports nothing but torch and
    math, and a None branch inside a scripted function is a second place a
    layout can be wrong.
    """
    return torch.cat(
        (assemble_observation(joint_pos, joint_vel, tip_rel, ee_quat, yaw_cs, pocket_quat, force),
         torque),
        dim=-1,
    )


# ===========================================================================
#  Termination and the success predicate (D-113, D-114, D-106)
# ===========================================================================


@torch.jit.script
def gravity_tare(
    force_raw: torch.Tensor,
    parent_quat_w: torch.Tensor,
    hold_force_w: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """Remove the welded tool's OWN WEIGHT from the joint reaction force.

    WHY THIS EXISTS. ``body_incoming_joint_wrench_b`` is the wrench the parent
    link applies to the child link, so it carries everything the parent has to
    hold: the contact load AND the static weight of the child. FORGE, the
    pattern D-114 adopted, never sees that second part -- Factory and FORGE
    spawn the robot and the held asset with ``disable_gravity=True``
    (``factory_env_cfg.py:127``, ``factory_tasks_cfg.py:165``), so their joint
    wrench IS the contact force. This repo keeps gravity ON at the robot by
    supervisor decision (``ur5e_cfg.py``, 2026-08-25), and the same code then
    reads a different quantity: RT-59 measured tool_link at 8.0861 N against
    the tool's own 8.0834 N with no contact at all, ratio 1.000. This function
    closes that gap so the channel means what D-114 imported it to mean.

    ``hold_force_w`` is the force the parent must apply in the WORLD frame to
    hold the child up, i.e. ``-mass * gravity_vector``, shape (3,). It is
    handed in rather than derived here: the mass has its home in
    ``cfg.tool_mass_kg`` and the gravity vector in ``cfg.sim.gravity``.

    The rotation is unavoidable and is NOT a rotation of the signal. The
    wrench stays raw in the parent body frame (the 2026-08-28 frame decision
    is untouched); what turns is the KNOWN weight vector, which is fixed in
    the world and therefore moves in the sensor's frame as the wrist turns.
    Subtracting a constant 8.08 would only be correct while the wrist stands
    still.

    What this does NOT remove: the inertial term. Under acceleration the
    parent also holds ``mass * a``, and no static tare can know it. A real
    payload compensation has the same limit.
    """
    if not enabled:
        return force_raw
    weight = hold_force_w.reshape(1, 3) * torch.ones_like(force_raw)
    return force_raw - rotate_into_frame(axes_from_quat(parent_quat_w), weight)


@torch.jit.script
def gravity_tare_torque(
    torque_raw: torch.Tensor,
    parent_quat_w: torch.Tensor,
    hold_force_w: torch.Tensor,
    lever_w: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """Remove the welded tool's OWN WEIGHT MOMENT from the joint reaction
    torque (D-188) -- the torque twin of ``gravity_tare``.

    The parent holds the child up with ``hold_force_w`` applied at the
    child's centre of mass, so the moment it carries about the reference
    point is ``lever_w x hold_force_w`` with ``lever_w`` = child COM minus
    reference point, both in WORLD (the COM is the one the weld AUTHORED,
    D-076..D-079, read from ``body_com_pos_w`` -- no geometry is assumed
    here). That world moment is rotated into the parent body frame exactly
    like the weight in ``gravity_tare`` and subtracted from the raw torque,
    which stays unrotated. ``cross(a, b)`` then ``R^T``, which equals
    ``cross(R^T a, R^T b)`` for a proper rotation; one rotation instead of
    two.

    WHAT IS ASSUMED, AND WHERE IT IS PAID. Isaac Lab's own test
    (``test_articulation.py:1871-1882``) takes the moment about the PARENT
    link origin and rotates with the PARENT quaternion; the env passes that
    reference point in. Whether PhysX's reading agrees -- reference point
    AND sign -- is NOT known from this function: ``probe_wrench_bodies.py
    --tare-check`` measures both (RT-192). The Isaac test also writes the
    gravity term as ``m*g`` (pointing down) where this repo's force tare
    subtracts ``-m*g`` and measures ~0 N afterwards (RT-115, RT-180c); the
    measured sign wins for the force, and the torque follows it here until
    the probe says otherwise.

    Same limit as the force: the inertial moment under acceleration is not
    removed. The result is a gravity-compensated joint torque, not a
    reconstructed contact torque.
    """
    if not enabled:
        return torque_raw
    hold = hold_force_w.reshape(1, 3) * torch.ones_like(torque_raw)
    moment_w = torch.cross(lever_w, hold, dim=-1)
    return torque_raw - rotate_into_frame(axes_from_quat(parent_quat_w), moment_w)


@torch.jit.script
def force_magnitude(force: torch.Tensor) -> torch.Tensor:
    """``||F||`` from the three force components (D-114).

    The three components are what the policy sees and what the abort reads --
    the same quantity, which is the whole point of D-114: no episode may end
    on something the policy cannot observe. The wrench's three TORQUE
    components are read by neither.
    """
    return torch.linalg.norm(force, dim=-1)


@torch.jit.script
def compute_dones(
    episode_length_buf: torch.Tensor,
    max_episode_length: int,
    force_norm: torch.Tensor,
    f_max: float,
    success_now: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The exits: TWO terminal sources and the timeout (D-113 point (6)).

    Returns ``(terminated, truncated, force_abort)``.

    * ``truncated`` -- the timeout. D-113 point (4): the task is
      time-unlimited in Pardo's sense, so the step cap is a training device,
      not part of the task. It therefore belongs in ``truncated``, where the
      Isaac Lab wrapper sets ``extras["time_outs"]`` and rsl_rl's
      ``process_env_step`` adds ``gamma*V(s)`` back. Collapsing the two flags
      into one "done" would teach the policy that running out of time is real
      failure.
    * ``terminated`` -- a true terminal, and since D-164 (2026-09-02) it has
      exactly ONE source: the success. THE FORCE ABORT MOVED TO ``truncated``.
      It is a designer-imposed safety stop, not a terminal state of the task's
      own MDP -- the same category as the step cap -- so rsl_rl must add
      ``gamma*V(s)`` back for it (Pardo 2018 Eq. (6) and Sec. 5). Leaving it in
      ``terminated`` forfeited the whole remaining return and made one abort
      cost -55 where D-114 had written -1.0; RT-131 measured the consequence, a
      policy that stops attempting insertion by iteration 70. This SUPERSEDES
      D-113's assignment of the abort to ``terminated``. ``force_abort`` is
      still returned alongside, because the reward pays it its
      ``abort_payment`` and the abort rate is reported as its own number
      (D-113 point (9)); deriving it a second time at the call site would give
      one predicate two homes.

    SUCCESS NOW TERMINATES. Inbox entry "Reward-Ueberarbeitung" 2026-09-01
    (p1-konzept-messung), point (7), which revises D-113 point (1) and D-109
    point (7). ``success_now`` is the SAME per-step ``in_success_region``
    predicate the display bonus is paid on -- one predicate, new trigger, no
    second definition of "seated". The paragraph that stood here read:

        "No success termination (D-113 point (1)). The episode runs to the
        timeout even from a perfect seat -- the Isaac Lab
        factory/forge/AutoMate lockstep-reset pattern. ... A check requires a
        fully seated state to NOT terminate."

    What overturned it is not the precedent (Inoue 2017,
    Beltran-Hernandez 2020b, Petrovic 2022, Brahmbhatt 2023 all terminate on
    success) but an EXPLOIT the reward account of 2026-09-01 exposed: with the
    episode running on, parking at "engaged" for 256 steps paid 260-303 against
    145 for seating late, so the optimal policy was to hover and never seat.
    The payout in ``compute_rewards_insertion`` pays the seated rate for every
    remaining step, which is what makes parking unable to win. The episode
    ending is what makes the seated state absorbing rather than merely
    lucrative.

    THE OTHER EXITS STILL DO NOT EXIST, and each is a decision, not an
    omission:

    * **No workspace, joint-limit, stuck or divergence abort** (D-113 point
      (6)). The NVIDIA assembly tasks define none; Isaac Lab's library
      termination terms belong to locomotion. "Part dropped" is impossible
      under D-041's rigid chain.
    * **No ``below_plate`` exit.** That was the proxy's third exit, added
      because its reward paid for diving under the table. This reward has no
      such hole -- the SDF distance grows down there like anywhere else -- so
      the exit goes with it.
    * **No position abort.** Noted in D-113 point (8) as a named option if the
      force abort proves insufficient; NOT adopted.

    ``f_max`` is ``[measure]`` (D-114): it comes from the force distribution
    measured at the scripted-insertion gate, not from an estimate.
    ``insertion_env_cfg.py`` refuses to build an env while it is unset. The
    comparison is ``>=`` so that a force landing exactly on the limit aborts.
    """
    force_abort = force_norm >= f_max
    timeout = episode_length_buf >= max_episode_length - 1
    # SUCCESS is the ONLY true terminal: the task is done and the seated state
    # is absorbing, so its value is the lump and nothing may be bootstrapped on
    # top of it. Everything else that ends an episode early is a training
    # device and is therefore TRUNCATED (D-164). ``& ~success_now`` keeps the
    # two channels exclusive, which also closes the pre-existing overlap of a
    # success landing on the timeout step.
    terminated = success_now
    truncated = (timeout | force_abort) & (~success_now)
    return terminated, truncated, force_abort


@torch.jit.script
def in_pocket_cross_section(tip_rel: torch.Tensor, wall_x: float, wall_y: float) -> torch.Tensor:
    """Is the part tip LATERALLY over the pocket opening? (D-157.)

    ``depth`` is a projection onto the pocket axis and carries no lateral
    information whatsoever. A part lowered 109 mm BESIDE the fixture projects
    the same depth as one going in -- which is exactly what RT-107 measured:
    ``success_rate_recent`` 0.9765 at ``mean_max_depth_mm`` 0.0.

    The ``mean_max_depth_mm`` metric already gated on the pocket walls, which
    is why it read 0.0 and gave the hack away; the two PAYING predicates did
    not. This function is that gate, lifted out of the metric so both can use
    the one expression.

    Why the walls and not some new tolerance: the comparison is the metric's,
    unchanged -- strictly inside the half-widths of the pocket opening. No new
    number enters the task. ``wall_x`` / ``wall_y`` are
    ``insertion_tasks_cfg.POCKET_WALL_X`` / ``POCKET_WALL_Y`` and arrive as
    arguments, like every other geometry number in this module.

    ``tip_rel`` is the tip in the entrance-anchored pocket frame
    (``part_tip_pose``), so columns 0 and 1 are the lateral offsets.
    """
    return (tip_rel[:, 0].abs() < wall_x) & (tip_rel[:, 1].abs() < wall_y)


@torch.jit.script
def in_pocket_volume(
    cross_section: torch.Tensor, depth: torch.Tensor, underside_depth: float
) -> torch.Tensor:
    """Is the part tip in the pocket, and not BELOW the fixture? (RT-201s3.)

    ``in_pocket_cross_section`` reads x and y only, so a tip that has gone
    through the pocket floor and hangs under the free-standing fixture is
    still "in the pocket" there. RT-201s3 (2026-09-15) measured exactly that:
    (CORRECTED the same day after RT-201s3r: the part does not pass the
    floor; it goes AROUND the free-standing fixture and under it.)
    ``max_depth_max_mm`` at the clamp floor (~144 mm) in 719 of 785
    iterations, ``mean_max_depth_mm`` up to 72 mm, and the engaged row paying
    38-61 per episode in iterations 100-250 (seed 2 of the same command: 3-6).
    The gated depth fed that row, the progress row and the depth metric.

    The floor of the gate is the fixture's UNDERSIDE, not the pocket floor:
    between the two the part is inside the floor material, where SAPU already
    discards the task return, and a gate at the seat depth would switch the
    engaged row on and off with every tenth of a millimetre the floor contact
    penetrates. No new number: ``underside_depth`` is
    ``-insertion_tasks_cfg.POCKET_ASSET_BOTTOM_Z`` and arrives as an argument.
    Inclusive, so the underside itself is still inside.
    """
    return cross_section & (depth <= underside_depth)


@torch.jit.script
def in_success_region(
    depth: torch.Tensor,
    interpen_dist: torch.Tensor,
    depth_min: float,
    depth_max: float,
    interpen_thresh: float,
    in_pocket: torch.Tensor,
) -> torch.Tensor:
    """Is the part seated RIGHT NOW? (D-106 / D-052 band + SAPU filter.)

    Two conditions, both necessary:

    * the insertion depth lies inside the D-052 band -- a band, not a
      threshold, because past full seat the part is through the fixture, and
      "deeper is better" is exactly the reading that let the proxy score a
      success under the table;
    * interpenetration is below the SAPU threshold. Without this filter a part
      tunnelled through the pocket wall reads geometrically seated. Force does
      NOT enter here: D-051 keeps force out of the success definition, and
      D-113 point (9) reports the abort rate as a SEPARATE number instead.

    Used for THREE things now, all off this one predicate: the per-step
    display bonus (D-109 point (5)), the success TERMINATION and its payout
    (``compute_dones`` / ``compute_rewards_insertion``, 2026-09-01), and the
    reported episode success, which since that revision IS the termination
    event. ``success_and_hold`` -- "seated at the last step" -- is gone with
    the scheme it belonged to: an episode that reaches this region ends there,
    so "at the last step" and "at all" are the same statement. The band edges
    are ``[CAD pending]`` and arrive as arguments.
    """
    deep_enough = depth >= depth_min
    not_too_deep = depth <= depth_max
    clean = interpen_dist < interpen_thresh
    return deep_enough & not_too_deep & clean & in_pocket


@torch.jit.script
def interpen_unclean(interpen_dist: torch.Tensor, interpen_thresh: float) -> torch.Tensor:
    """Is this step's interpenetration AT OR ABOVE the SAPU threshold? One flag per env.

    The episode record (``episodes.csv``, user scope 2026-09-15) counts the
    policy steps for which this is True (column ``ip_steps_ge_t1``). The
    boundary is the exact complement of the strict ``<`` in
    ``in_success_region``'s clean test, and the same edge at which
    ``sapu_reward_scale`` writes zero (``interpen_dist >= interpen_thresh``):
    a counted step is a step whose task return is discarded and on which no
    success can fire. ``check_insertion_math.py`` pins the edge.

    WHAT THE THRESHOLD IS, AND WHAT IT IS NOT (user, 2026-09-15): an
    ALGORITHMIC acceptance threshold of this code (D-106 (3), half
    ``PLAY_X``). Half the play does not make it a physically admissible
    penetration -- the real pair admits none.
    """
    return interpen_dist >= interpen_thresh


@torch.jit.script
def is_engaged(depth: torch.Tensor, engaged_depth: float, in_pocket: torch.Tensor) -> torch.Tensor:
    """The "engaged" display bonus predicate (D-109 point (5)).

    ``engaged_depth`` is ``[CAD pending]`` -- the open number D-109 names
    explicitly. It arrives as an argument and the config refuses to build
    without it.

    ``in_pocket`` gates it laterally (D-157, ``in_pocket_cross_section``): the
    depth projection alone paid this bonus for a descent beside the fixture.
    """
    return (depth >= engaged_depth) & in_pocket


# ``success_and_hold`` STOOD HERE and is deleted, 2026-09-01. It was episode
# success = ``in_region & is_last_step`` -- "seated AT THE LAST STEP" (D-113
# point (2), IndustReal), written against a scheme in which the episode ran on
# after a seat and a policy could knock the part out again by step 200.
#
# The success termination (inbox entry "Reward-Ueberarbeitung" 2026-09-01,
# point (7)) removes the state the function guarded against: the episode ENDS
# on the seat, so the last step of a successful episode is the seating step,
# and ``in_region`` at reset time already is the "held" answer. Keeping the
# conjunction would have been strictly worse than deleting it -- it reads True
# only when the truncation flag also fires, which after the revision is the one
# case a successful episode never reaches. The episode success metric is now
# the termination EVENT, taken from the same ``in_success_region`` predicate.


# ===========================================================================
#  SAPU port: the pure half (D-109 point (10))
# ===========================================================================
#
# WHAT IS PORTABLE AND WHAT IS NOT. SAPU is two pieces:
#
#   1. ``get_interpen_dist`` -- a Warp kernel that queries the fixture mesh
#      once per SAMPLE POINT of the held asset. Isaac Lab 2.3.2 HAS it
#      (``industreal_algo_utils.py:343-379``) and it is a pure mesh query,
#      never PhysX. It cannot run here: it needs Warp, a built BVH and the
#      real meshes. Training PC only.
#   2. ``get_max_interpen_dists`` + ``get_sapu_reward_scale`` -- the caller
#      chain. Isaac Lab 2.3.2 has NEITHER, and no ``interpen_thresh`` config
#      at all; both exist only in IsaacGymEnvs and must be ported. They are
#      pure tensor arithmetic, so they live here and are checked offline.
#
# ``get_sapu_reward_scale`` is ``sapu_reward_scale`` above. The reduction is
# below.
#
# The three D-109 adaptations, and where each one lands:
#   (a) the target pose must come from OUR CAD -- the Isaac Lab code equates
#       target pose with pocket pose because its meshes are drawn coincident
#       and ours are not. That is an ARGUMENT at the call site, not math:
#       nothing here assumes where the target is.
#   (b) compute load: a Python loop over envs with a mesh rebuild and BVH
#       refit per env per step, at reference configs of only 128 envs. Must be
#       MEASURED against our env count (D-115 makes the env count a rule, not
#       a number). Training PC.
#   (c) no ``-log(0)`` guard is needed, because the ``-log`` mapping is gone
#       (see ``squash``).


@torch.jit.script
def interpen_from_signed(signed_dist: torch.Tensor) -> torch.Tensor:
    """Turn a SIGNED distance array into the source's interpenetration array.

    Input and output are both ``(num_envs, num_points)``.

    This exists because the port needs NO second Warp kernel, and that is a
    finding rather than a shortcut. ``get_interpen_dist``
    (``industreal_algo_utils.py:343-379``) runs the same
    ``wp.mesh_query_point`` + ``wp.mesh_eval_position`` pair our
    ``_sdf_batched`` already runs, computes the same
    ``signed_dist = sign * length(q - p)``, and differs in exactly one line:
    it writes the value only ``if signed_dist < 0.0``, into an array seeded
    with ``wp.zeros``. Clamping our signed array at ``max=0.0`` produces that
    array element for element.

    The one case where the two could differ is the miss -- no surface within
    ``max_dist``. The source leaves the seeded ``0.0``; ours writes
    ``+max_dist``, which is positive and clamps to the same ``0.0``. So they
    agree there too.

    Convention out, unchanged from the source and relied on by
    ``max_interpen_dist``: every element is ``<= 0``. Zero means the point
    clears the mesh, negative means it is inside by that many metres.

    WHAT THIS DOES NOT DO: it does not make the sign trustworthy.
    ``wp.mesh_query_point`` documents "NOTE: Mesh must be watertight!" and the
    fixture mesh is not (9 unpaired edges, RT-64). RT-90 measured the sign on
    the PART mesh far outside its surface; nothing has measured it on the
    FIXTURE mesh, and interpenetration is read at the surface, not far from
    it. See D-152 for the same gap on the part.
    """
    return torch.clamp(signed_dist, max=0.0)


@torch.jit.script
def max_interpen_dist(per_point_dist: torch.Tensor) -> torch.Tensor:
    """Reduce per-sample-point interpenetration to ONE depth per env.

    Input is ``(num_envs, num_points)``, the per-point output of the Warp
    query; output is ``(num_envs,)``.

    SIGN CONVENTION, read from the source rather than assumed: the kernel
    reports penetration as a NEGATIVE number. ``get_interpen_dist``
    (``industreal_algo_utils.py``) computes ``signed_dist = sign * length(q - p)``
    and writes it only under ``if signed_dist < 0.0``, into an array seeded
    with ``wp.zeros``. So every element is ``<= 0``: zero where the point
    clears the mesh, negative where it is inside. IsaacGymEnvs reduces the
    same array with ``-torch.min(...)`` for exactly this reason.

    Hence the negation BEFORE the clamp. An earlier version of this function
    assumed penetration was positive and returned
    ``amax(clamp(d, min=0.0))``, which on the real kernel output is
    identically zero -- SAPU and the success filter would both have been
    silently off while every check stayed green on hand-written positive
    inputs. D-109 point (10c) warns that a non-watertight mesh flips signs by
    itself; that failure is separate and still belongs to the scene stream.

    MAX, never mean. A part that touches the pocket wall at exactly one of its
    sample points is tunnelling there, and a mean over 1000 points would
    dilute a 1 mm spike into a micrometre. The offline check requires a single
    deep point to dominate.

    The clamp survives the negation and keeps its own job: a positive reading
    (clearance, should the kernel ever report one) must not become a negative
    penetration that cancels a real one elsewhere.

    Also carried from D-109 point (10c) and NOT fixable here: 1000 sample
    points are thin for a 143 mm part, and too few points systematically
    UNDER-report penetration.
    """
    return torch.amax(torch.clamp(-per_point_dist, min=0.0), dim=-1)


# ===========================================================================
#  The task-space action (Audit 2026-09-03 (a), Decision (3) and (4))
# ===========================================================================
#
# The policy's action is a 6-D POSE DELTA: three metres of translation and an
# axis-angle rotation, both in the env frame, both clamped to a per-step limit
# by the env. The target the controller tracks is
#
#     target = current pose + delta          (re-anchored EVERY physics step)
#
# which is Factory's form (``factory_env.py:268``, ``_apply_action``): the
# tracking error can never exceed one step limit, and that is the whole of
# the anti-windup. Two clamps sit on the target and both are Factory's FORM
# with OUR numbers (inbox entry "Audit 2026-09-03 (a)", Decision (3)/(4)):
#
#   * ``clamp_tip_in_box`` -- the leading tool point stays inside a box around
#     the pocket entrance (Factory: ``pos_action_bounds`` around the fixed
#     asset, ``factory_env.py:268-276``).
#   * ``clamp_tilt_to_cone`` -- the tool axis stays within a cone around the
#     POCKET AXIS. Factory pins roll and pitch to UPRIGHT (``:291-292``); the
#     real part goes in edge-first (D-071), so the cone replaces the pin and
#     its half-angle is the CAD tilt limit, 8.52 deg
#     (``Geometrie_Fuegeteil_Aufnahme.tex:264``). Yaw is untouched (D-106).
#
#     THE REFERENCE AXIS IS THE POCKET'S, NOT THE WORLD'S, since Phase 5 step
#     B7, and the change is a FRAME correction rather than a new number. The
#     8.52 deg is a CAD angle BETWEEN part and pocket -- the maximum lean with
#     only the front edge in the opening. Measured from the world vertical it
#     meant something else the moment the pocket stopped being upright: at the
#     AutoDR maximum tilt of 8.00 deg, merely ALIGNING with the pocket spends
#     8.00 of the 8.52 and leaves the policy 0.52 deg of its own. The source
#     and the code were in different frames; now they are in one. With an
#     upright pocket the two are the same construction, which is what the
#     offline check asserts bit-for-bit.
#
# Everything here is pure tensor algebra so the offline check can execute it
# under the numpy stand-in; the controller that consumes the target is Isaac
# Lab's ``OperationalSpaceController`` and lives in the env.


@torch.jit.script
def quat_from_axis_angle(aa: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """(N, 3) axis-angle vector -> (N, 4) wxyz unit quaternion.

    The vector's direction is the axis and its norm the angle. Below ``eps``
    the result is the identity rotation; the division is guarded so a zero
    delta (the zero action) never produces NaN. Same convention as Isaac
    Lab's ``quat_from_angle_axis`` and Factory's rotation delta.
    """
    angle = torch.linalg.norm(aa, dim=-1)
    safe = torch.where(angle > eps, angle, torch.ones_like(angle))
    axis = aa / safe.reshape(-1, 1)
    half = 0.5 * angle
    s = torch.sin(half)
    q = torch.stack((torch.cos(half), axis[:, 0] * s, axis[:, 1] * s, axis[:, 2] * s), dim=-1)
    identity = torch.zeros_like(q)
    identity[:, 0] = 1.0
    return torch.where((angle > eps).reshape(-1, 1), q, identity)


@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate (N, 3) vectors by (N, 4) wxyz quaternions: ``R(q) @ v``.

    Through ``axes_from_quat`` so this module keeps ONE rotation convention
    (columns are the body axes in the parent frame).
    """
    rot = axes_from_quat(q)
    return torch.sum(rot * v.reshape(-1, 1, 3), dim=2)


@torch.jit.script
def apply_pose_delta(
    pos: torch.Tensor,
    quat: torch.Tensor,
    delta_pos: torch.Tensor,
    delta_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``target = current + delta``: (N, 3) + (N, 3) metres, and the (N, 3)
    axis-angle rotation PRE-multiplied onto the (N, 4) wxyz orientation.

    Pre-multiplied, i.e. the rotation delta is expressed in the PARENT (env)
    frame, exactly as Factory composes it
    (``quat_mul(rot_actions_quat, fingertip_midpoint_quat)``,
    ``factory_env.py:288``). Post-multiplying would express the delta in the
    tool frame, and the two differ by the tool's own attitude.
    """
    return pos + delta_pos, quat_mul(quat_from_axis_angle(delta_rot), quat)


@torch.jit.script
def clamp_tip_in_box(
    pos: torch.Tensor,
    quat: torch.Tensor,
    tip_offset_local: torch.Tensor,
    centre: torch.Tensor,
    half_x: float,
    half_y: float,
    half_z: float,
) -> torch.Tensor:
    """Keep the LEADING TOOL POINT inside an axis-aligned box around ``centre``
    and return the body position that puts it there.

    ONE HALF-WIDTH PER AXIS (Phase 5 step B7). The three are not the same
    quantity. Sideways the box only has to hold the lateral reset band and the
    tip's own swing under tilt; along z it has to hold the OUTERMOST legal
    start height as well, and the action step that may still be commanded
    outward from there. A single half-width made the two share one number, and
    at ``start_height.hi_max`` the margin along z came out at exactly
    0.000000 m -- the axis with no room left was the one carrying the whole
    start band. Which reach each axis must hold is pinned in
    ``check_env_wiring`` against ``autodr.bounds_max()``, not against the
    static cfg constants: under AutoDR those are only the width-0 centres.

    The clamp acts on the tip, not on the controlled body: the box is "around
    the pocket entrance" (Decision (4)) and the tip is what enters the pocket.
    With the target orientation already fixed, the body position follows as
    ``tip - R(quat) @ tip_offset``; the orientation is NOT changed here. Inside
    the box the position comes back bit-identical.
    """
    # Broadcast by addition, not ``expand``: the offline stand-in has no expand.
    offset_w = quat_rotate(quat, tip_offset_local.reshape(1, 3) + torch.zeros_like(pos))
    tip = pos + offset_w
    d = tip - centre
    # Per-axis scalar clamps stacked back together, not a vector bound:
    # TorchScript's ``clamp`` takes numbers, and this is the form that survives
    # both ``jit.script`` and the offline stand-in.
    tip_c = centre + torch.stack(
        [
            torch.clamp(d[:, 0], min=-half_x, max=half_x),
            torch.clamp(d[:, 1], min=-half_y, max=half_y),
            torch.clamp(d[:, 2], min=-half_z, max=half_z),
        ],
        dim=-1,
    )
    return tip_c - offset_w


@torch.jit.script
def tool_axis_tilt(quat: torch.Tensor) -> torch.Tensor:
    """Angle (N,) in radians between the tool's +z axis and the world DOWN
    direction ``(0, 0, -1)``. Zero when the part hangs straight down.

    The tool's +z is the axis ``peg_tip_offset`` lies on
    (``insertion_env_cfg.peg_tip_offset`` is a pure +z offset), so this is the
    lean of the part away from vertical, whatever its yaw.
    """
    axis = axes_from_quat(quat)[:, :, 2]
    return torch.acos(torch.clamp(-axis[:, 2], min=-1.0, max=1.0))


@torch.jit.script
def clamp_tilt_to_cone(
    quat: torch.Tensor,
    max_tilt_rad: float,
    ref_quat: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Pull an (N, 4) wxyz orientation back until its tool axis leans at most
    ``max_tilt_rad`` from THIS EPISODE'S POCKET AXIS; orientations inside the
    cone come back bit-identical.

    ``ref_quat`` is the pocket orientation, (N, 4) wxyz, the same buffer the
    reset writes and the observation publishes. Its +z points OUT of the
    opening -- ``seated_goal_pose`` subtracts the seat depth along it -- so
    the direction the tool is measured against is ``-ref_z``.

    WHY THE POCKET AND NOT THE WORLD (Phase 5 step B7): ``max_tilt_rad`` is a
    CAD angle BETWEEN part and pocket, the maximum lean with only the front
    edge in the opening. Read from the world vertical it silently became a
    different quantity as soon as the pocket tilted: aligning with a pocket
    tilted by the AutoDR maximum already spent 8.00 of the 8.52 deg. The
    number did not change here; the frame it is measured in did, so that it
    is the frame its source is stated in.

    The correction is the SMALLEST rotation that does it: about the axis
    ``tool_axis x down`` by the excess angle, which moves the tool axis
    straight toward ``down`` and leaves the rotation ABOUT the tool axis
    (the yaw) alone -- yaw stays free (Decision (3)). A tool axis parallel or
    antiparallel to ``down`` has no unique such axis; the POCKET's own x axis
    is used there -- it is perpendicular to ``down`` by construction, where
    the world x axis only was while the pocket stood upright -- so the result
    is a valid rotation rather than NaN.

    UPRIGHT POCKET = THE OLD CONSTRUCTION, exactly. With ``ref_quat`` the
    identity, ``down`` is ``(0, 0, -1)``, ``tool x down`` is
    ``(-a_y, a_x, 0)`` and the fallback axis is world x -- the three
    expressions the previous version wrote out by hand. The offline check
    asserts that equality bit-for-bit rather than trusting this paragraph.
    """
    ref = axes_from_quat(ref_quat)
    tool = axes_from_quat(quat)[:, :, 2]
    down = -ref[:, :, 2]
    tilt = torch.acos(torch.clamp((tool * down).sum(dim=-1), min=-1.0, max=1.0))
    excess = torch.clamp(tilt - max_tilt_rad, min=0.0)
    # tool x down; its norm is sin(tilt).
    n = torch.cross(tool, down, dim=-1)
    n_norm = torch.linalg.norm(n, dim=-1)
    degenerate = n_norm < eps
    n_safe = torch.where(degenerate.reshape(-1, 1), ref[:, :, 0], n)
    n_unit = n_safe / torch.where(degenerate, torch.ones_like(n_norm), n_norm).reshape(-1, 1)
    fix = quat_from_axis_angle(n_unit * excess.reshape(-1, 1))
    return quat_mul(fix, quat)


# ===========================================================================
#  The lateral start offset on a disk (D-178 (1))
# ===========================================================================


@torch.jit.script
def disk_offset(u_r: torch.Tensor, u_phi: torch.Tensor, r_max: float) -> torch.Tensor:
    """(N,) unit draws for radius and angle -> (N, 2) pocket-frame (x, y).

    ``r = r_max * sqrt(u_r)`` and ``phi = 2 * pi * u_phi`` (D-178 (1)). The
    square root makes the draw uniform over the disk's AREA: ``r = r_max *
    u_r`` would put half of all draws inside ``r_max / 2``, a quarter of the
    area. A pure map from unit values the reset already drew -- the same kind
    of helper as ``autodr.map_unit_to_bounds``; the random numbers stay in the
    env.

    ``u_r = 1`` gives EXACTLY ``r_max`` (``sqrt(1.0)`` is 1.0), so a boundary
    env nailed to the upper edge of ``lat_r`` starts on that edge, not within
    a rounding error of it. ``r_max = 0`` gives zeros for every draw.

    ``math.pi`` inside a scripted function is Isaac Lab 2.3.2 practice
    (``cart_double_pendulum_env.py`` ``normalize_angle``); whether this
    function compiles on the training machine is UNVERIFIED until it runs
    there.
    """
    r = r_max * torch.sqrt(u_r)
    phi = 2.0 * math.pi * u_phi
    return torch.stack((r * torch.cos(phi), r * torch.sin(phi)), dim=-1)
