# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""The scripted insertion policy -- pure math (D-108 gate).

WHY THIS FILE EXISTS
--------------------
D-108 attaches a GATE to the action-space decision (six joint deltas,
``action_scale = 0.02`` rad): before the first training run, a SCRIPTED
insertion must place the part. It discriminates a control-chain failure
(H1 precision, H2 contact) from a policy failure -- if the chain cannot do it
scripted, no reward tuning can fix it and the D-108 fallback (task-space
deltas + impedance control) is the next thing to test.

The run also PAYS for three open numbers: the force distribution behind
``force_abort_f_max_n`` (today an invented number), ``engaged_depth_m``, and
the input for ``rung_step_sizes``.

WHAT IT MAY LOOK AT, AND WHAT IT MAY NOT
----------------------------------------
User decision, 2026-08-28: the target information comes ONLY from the
OBSERVATION -- the same 28 channels the policy will get. The script does not
read ``env._entrance_pos``, ``env._tilt_rot``, ``env._fixture_quat`` or
``env._peg_geometry()``. Reason given: both routes carry the same
information, and this one additionally exercises the observation.

The ROBOT model is not target information and stays allowed: the Jacobian,
the joint positions and the base pose are what any real controller has. The
split is target truth vs. proprioception, not env-internal vs. public.

Consumed observation blocks (``insertion_math.OBS_SLICES``):

* ``tip_rel``     12:15 -- tip minus pocket entrance, IN THE POCKET FRAME,
                           part-anchored (D-107 (4)). ``+z`` is above the
                           opening, so ``depth = -tip_rel_z``.
* ``pocket_quat``  21:25 -- pocket orientation in the env frame, wxyz.
* ``yaw_cos_sin``  19:21 -- ``(cos phi, sin phi)`` of the part about the
                           pocket z-axis.
* ``force``        25:28 -- the EMA-smoothed force, the abort input.

THE YAW TARGET IS LATCHED, NEVER ASSUMED
----------------------------------------
The real part fits its pocket in exactly ONE rotational position, and which
``phi`` that is, is written down nowhere in this repo. D-078 fixes the WELD
yaw at zero, which is a statement about the tool, not about the seated pose.

So this module assumes no angle. It LATCHES the yaw of the episode's first
step and holds it. With every reset-noise range at 0.0
(``insertion_env_cfg.py:199-201, 316, 326, 361``) that latched value IS the
home-pose orientation, which is the pose the cell was built around. The run
reports the residual instead of asserting it: if the home pose is truly
aligned, the reported errors are small; if it is not, that is a finding
rather than a silent wrong assumption.

When the yaw noise is later opened up via hydra, the target is the value
MEASURED at zero noise -- still not an assumed angle.

PROVENANCE: OWN CONSTRUCTION, and named as such
-----------------------------------------------
Neither the proxy repo nor Isaac Lab 2.3.2 ships a scripted insertion policy;
the proxy has no such script at all. The pieces that DO exist are reused
rather than rewritten: the rotation algebra comes from ``insertion_math``
(``axes_from_quat``), and the joint-space conversion is Isaac Lab's own
``DifferentialIKController`` in ``dls`` mode, driven by the caller. What is
new here is the phase machine and the observation-to-command mapping.

This module imports ``torch`` and ``insertion_math`` and nothing else, so
``scripts/check_scripted_insert.py`` executes the whole file under the numpy
stand-in.

IT OWNS NO TASK NUMBERS
-----------------------
Every gain, tolerance and rate arrives as an argument. The phase codes below
are this module's own fact, the way ``OBS_SLICES`` is ``insertion_math``'s.
"""

from __future__ import annotations

import torch

from .insertion_math import axes_from_quat, rotate_into_frame

# ===========================================================================
#  Phase codes -- this module's own fact, one home
# ===========================================================================
#
# Held as float rather than int because the stand-in's tensor surface is
# float-only, and because every consumer either compares or multiplies them.
PHASE_ALIGN = 0.0
PHASE_DESCEND = 1.0
PHASE_ABORT_FORCE = 2.0

PHASE_NAMES: dict = {
    PHASE_ALIGN: "ALIGN",
    PHASE_DESCEND: "DESCEND",
    PHASE_ABORT_FORCE: "ABORT_FORCE",
}


# ===========================================================================
#  The yaw target: latch it, do not assume it
# ===========================================================================


@torch.jit.script
def latch_yaw_target(
    yaw_cs: torch.Tensor,
    target_cs: torch.Tensor,
    latched: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture the episode's starting yaw once per env.

    Args:
        yaw_cs: ``(N, 2)`` current ``(cos phi, sin phi)`` from the observation.
        target_cs: ``(N, 2)`` the latched target so far.
        latched: ``(N, 1)`` 1.0 where a target is already held, else 0.0.

    Returns ``(target_cs, latched)``.

    Held as ``(cos, sin)`` rather than as an angle on purpose: the observation
    delivers the pair, and converting to an angle and back would introduce a
    branch cut this module has no reason to own.

    ``latched`` is ``(N, 1)`` and not ``(N,)`` so it broadcasts against the
    ``(N, 2)`` pair without a reshape -- a reshape is exactly the kind of step
    that silently transposes a batch.
    """
    new_target = torch.where(latched > 0.5, target_cs, yaw_cs)
    return new_target, torch.ones_like(latched)


@torch.jit.script
def signed_angle_difference(a_cs: torch.Tensor, b_cs: torch.Tensor) -> torch.Tensor:
    """Signed angle ``a - b``, wrapped to ``(-pi, pi]``, from ``(cos, sin)`` pairs.

    Computed from the difference identities rather than by subtracting two
    ``atan2`` results:

        sin(a - b) = sin a cos b - cos a sin b
        cos(a - b) = cos a cos b + sin a sin b

    Subtracting two angles would need an explicit wrap, and an unwrapped
    difference is a defect that only shows up near the branch cut -- i.e.
    exactly where a controller is asked to take the short way round.

    ONE HOME, TWO CALLERS, and the second one is why this was lifted out of
    ``yaw_error`` on 2026-09-01. RT-127 (training PC, git e38e687) measured the
    part's tilt reference at ``-180.0000 deg`` in all 16 envs: the tool axis
    starts ANTIPARALLEL to the pocket axis, so the tilt reference sits exactly
    ON the branch cut, which is the one place an unwrapped subtraction is
    guaranteed to be wrong. ``tilt_since_reference`` therefore needs the same
    arithmetic, and a second copy of it would be a second place for the wrap
    to go missing.

    Both arguments are ``(N, 2)``. Returns ``(N,)``.
    """
    cos_a = a_cs[:, 0]
    sin_a = a_cs[:, 1]
    cos_b = b_cs[:, 0]
    sin_b = b_cs[:, 1]
    return torch.atan2(sin_a * cos_b - cos_a * sin_b, cos_a * cos_b + sin_a * sin_b)


@torch.jit.script
def yaw_error(yaw_cs: torch.Tensor, target_cs: torch.Tensor) -> torch.Tensor:
    """Signed angle ``phi - phi_target``, wrapped to ``(-pi, pi]``.

    The arithmetic and the reason for it live in ``signed_angle_difference``.
    This name stays because the call sites read as yaw, not as trigonometry.

    Returns ``(N,)``.
    """
    return signed_angle_difference(yaw_cs, target_cs)


# ===========================================================================
#  Frame changes -- one function per hop, so each hop gets its own check
# ===========================================================================


@torch.jit.script
def rotate_out_of_frame(rot: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``R v`` -- the exact inverse direction of ``insertion_math.rotate_into_frame``.

    That function expresses a parent-frame vector in the frame whose axes are
    the COLUMNS of ``rot`` (i.e. ``R^T v``). This one is the way back: it takes
    a vector given IN that frame and expresses it in the parent.

    Component i is ``sum_j rot[i, j] v[j]``, written as a broadcast product and
    a sum, matching the idiom next door -- the offline stand-in implements
    neither ``einsum`` nor ``bmm``, and the two directions differ only in which
    axis the sum runs over. Keeping them side by side is what makes a swapped
    direction visible instead of plausible.
    """
    return torch.sum(rot * v.reshape(-1, 1, 3), dim=2)


@torch.jit.script
def rotate_by_quat(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """``R(quat) v`` -- take ``(N, 3)`` FROM the quat's frame INTO its parent.

    The quaternion expansion is REUSED from ``insertion_math.axes_from_quat``,
    not re-derived: a second copy of that expansion is precisely the defect
    class that module's header records (the proxy shipped a transposed copy on
    2026-07-26, and only an identity test caught it).
    """
    return rotate_out_of_frame(axes_from_quat(quat), vec)


@torch.jit.script
def rotate_by_quat_inv(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """``R(quat)^T v`` -- take ``(N, 3)`` FROM the parent INTO the quat's frame.

    The transpose, not a separately derived inverse: ``R`` is orthonormal, and
    the ``insertion_math`` check proves that against known rotations rather
    than against a second copy of the formula.
    """
    return rotate_into_frame(axes_from_quat(quat), vec)


@torch.jit.script
def axis_tilt_angle(part_quat: torch.Tensor, pocket_quat: torch.Tensor) -> torch.Tensor:
    """Angle in RADIANS between the part's insertion axis and the pocket axis.

    MEASUREMENT ONLY. Nothing in the controller reads this -- ``ALIGN`` nulls
    the lateral error, the height and the yaw, and the two tilt components of
    its axis-angle command are hard zeros (``command_in_pocket_frame``). So the
    part's tilt is neither corrected nor, until now, observed: a part that
    enters the pocket crooked jams, and the run reports only the force it
    produced. RT-69 measured 204 N at 19.5 mm against RT-68's 51 N at 12.2 mm,
    which is what a jam looks like and also what a normal insertion at these
    gains might look like. This is the number that separates the two.

    Both quaternions are ``wxyz`` and both express their body's axes in the
    SAME basis -- ``ee_quat`` in the world frame, ``pocket_quat`` in the env
    frame, and Isaac Lab's env origins are pure translations, so the two bases
    have the same orientation. Only the axes are compared, never a position,
    so the translation between them is irrelevant by construction.

    Column 2 of ``axes_from_quat`` is the body z-axis: for the part that is the
    tool axis the tip offset lies on (``peg_tip_offset`` is a pure ``+z`` offset
    in the tool frame), and for the pocket it is the insertion direction. Both
    are unit vectors out of an orthonormal matrix, so their dot product is the
    cosine directly -- clamped before ``acos`` because rounding can push it a
    few ulps past 1.0, where ``acos`` returns NaN.

    UNSIGNED and always in ``[0, pi]``. A direction, not a rotation: the
    question this answers is "how far off parallel", and the sign would need a
    reference axis nobody has decided on.
    """
    part_axis = axes_from_quat(part_quat)[:, :, 2]
    pocket_axis = axes_from_quat(pocket_quat)[:, :, 2]
    cos = torch.sum(part_axis * pocket_axis, dim=-1)
    return torch.acos(torch.clamp(cos, min=-1.0, max=1.0))


# ===========================================================================
#  The tilted (edge-first) entry -- D-071's path, now scripted
# ===========================================================================
#
# WHY THIS SECTION EXISTS AT ALL. D-071 records that the real part goes in
# TILTED, edge first, and says in the same breath that nothing in the code
# encodes the tilted entry. That was affordable while the scripted descent was
# only the D-108 gate. It stopped being affordable on 2026-09-01:
# ``insertion_env_cfg.py:953-963`` marks ``engaged_depth_m`` as a placeholder
# whose replacement RULE is "the MEASURED CAPTURE DEPTH of the tilted scripted
# insertion", and ``:940-952`` marks ``force_abort_f_max_n`` as a placeholder
# whose lower bound is "the largest measured peak force of a CLEAN scripted
# insertion, straight AND tilted (edge-first)". Neither number can be read off
# a run that cannot tilt.
#
# WHAT THIS SECTION DOES NOT DO. It sets no angle, no schedule, no threshold
# and no axis. Every number arrives as an argument from the measurement group
# of ``scripts/tilt_insert.py``, the same way ``aim_offsets`` takes its span
# from ``PLAY_X`` / ``PLAY_Y``. The controller FOLLOWS a commanded tilt; which
# tilt is worth commanding is the run's question, not this module's answer.
#
# Column 0 of the pocket frame is the SHORT axis (``PLAY_X``, 0.5876 mm),
# column 1 the LONG one (``PLAY_Y``, 1.60 mm) -- the order ``aim_offsets``
# already uses, kept identical so a tilt index and an aim index never mean
# two different axes.
TILT_AXES: dict = {"x": 0, "y": 1}


@torch.jit.script
def signed_tilt_angle(
    part_quat: torch.Tensor, pocket_quat: torch.Tensor, axis: int
) -> torch.Tensor:
    """SIGNED tilt of the part axis about ONE pocket in-plane axis, radians.

    ``axis_tilt_angle`` above answers "how far off parallel" and is
    deliberately unsigned, because a magnitude needs no reference axis. A
    CONTROLLER needs one: to remove a tilt you must know which way it leans.
    And for this task the sign is itself a result -- the part's underside is
    a sloped strip on one side and a parallel strip on the other
    (``insertion_tasks_cfg.py:1393-1409``), so the two tilt directions meet
    different geometry and are not equivalent.

    ``axis = 0`` is a rotation about the pocket's x-axis (the SHORT axis),
    ``axis = 1`` about its y-axis (the LONG one). The caller passes
    ``TILT_AXES[name]``, so no string reaches the jit.

    THE ALGEBRA. Let ``a`` be the part's z-axis expressed IN THE POCKET FRAME.
    A rotation by ``theta`` about pocket ``+x`` carries ``(0, 0, 1)`` to
    ``(0, -sin theta, cos theta)``, so ``theta = atan2(-a_y, a_z)``. About
    ``+y`` it carries it to ``(sin theta, 0, cos theta)``, so
    ``theta = atan2(a_x, a_z)``. ``atan2`` and not ``asin``: it stays correct
    past 90 deg and needs no clamp against rounding, which is the failure
    ``axis_tilt_angle`` has to guard ``acos`` from.

    Both quaternions are ``wxyz`` in the SAME basis, for exactly the reason
    ``axis_tilt_angle`` documents (env origins are pure translations). Only
    axes are compared, never a position.
    """
    part_axis_w = axes_from_quat(part_quat)[:, :, 2]
    a = rotate_by_quat_inv(pocket_quat, part_axis_w)
    if axis == 0:
        return torch.atan2(-a[:, 1], a[:, 2])
    return torch.atan2(a[:, 0], a[:, 2])


@torch.jit.script
def latch_tilt_reference(
    tilt_now: torch.Tensor,
    ref_cs: torch.Tensor,
    latched: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture the episode's starting tilt once per env, held as ``(cos, sin)``.

    Args:
        tilt_now: ``(N,)`` the current ``signed_tilt_angle``, radians.
        ref_cs: ``(N, 2)`` the reference latched so far.
        latched: ``(N, 1)`` 1.0 where a reference is already held, else 0.0.

    Returns ``(ref_cs, latched)``.

    WHY THIS EXISTS, MEASURED. ``signed_tilt_angle`` answers "how far is the
    part off the POCKET axis". RT-127 (training PC, git e38e687) read that
    number at step 1: ``signed p50 -180.0000 deg`` in all 16 envs, i.e. the
    tool axis starts antiparallel to the pocket axis. That is a property of
    how the part is held, not an error to be corrected, but the tilt
    controller was steering against the pocket axis and therefore saw a
    permanent 180 deg error: the 0.2 deg ALIGN gate never opened (16 of 16
    envs held), nothing descended (``depth p50 0.000 mm`` in RT-124/RT-125)
    and the wrist turned into the block. Silencing the controller restored
    the descent (RT-126, 19.07-21.12 mm), which located the cause here.

    THE PATTERN IS NOT NEW. ``latch_yaw_target`` above refuses exactly the
    same assumption for the yaw: capture the starting attitude once, then
    command the DIFFERENCE from it. This is that function for the tilt, and
    it is deliberately shaped the same way, including ``latched`` being
    ``(N, 1)`` so it broadcasts against the pair without a reshape.

    HELD AS ``(cos, sin)`` AND NOT AS THE ANGLE, for the reason
    ``signed_angle_difference`` documents -- and here it is not a precaution
    but the measured case: the reference is at -180 deg, exactly ON the
    branch cut, where subtracting angles is guaranteed to be wrong.
    """
    now_cs = torch.stack((torch.cos(tilt_now), torch.sin(tilt_now)), dim=1)
    new_ref = torch.where(latched > 0.5, ref_cs, now_cs)
    return new_ref, torch.ones_like(latched)


@torch.jit.script
def tilt_since_reference(tilt_now: torch.Tensor, ref_cs: torch.Tensor) -> torch.Tensor:
    """Tilt measured AWAY FROM the latched start attitude ``(N,)``, radians.

    This is the quantity the tilt controller regulates and the quantity the
    run reports as "the tilt": at step 1 it is zero by construction, and a
    commanded +8 deg means eight degrees away from how the part is held, which
    is what "edge-first" names. ``signed_tilt_angle`` against the pocket axis
    stays available next to it and is the ABSOLUTE attitude -- the two are not
    interchangeable and the run prints both.

    The wrap is not optional here; see ``latch_tilt_reference``.
    """
    now_cs = torch.stack((torch.cos(tilt_now), torch.sin(tilt_now)), dim=1)
    return signed_angle_difference(now_cs, ref_cs)


def tilt_offsets(num_envs: int, lo_rad: float, hi_rad: float, device=None) -> torch.Tensor:
    """Per-env commanded START tilt ``(N,)`` in radians, endpoints INCLUDED.

    The same instrument as ``aim_offsets`` and for the same reason: the tilt
    is not episode state, it is one fixed property per env for the whole run,
    so the result table can be read as a curve against it. A value re-drawn at
    reset would mix tilts inside one env's statistics.

    A SPAN THAT CROSSES ZERO gives the straight insertion and BOTH tilt signs
    in one run. That is what the F_max rule asks for -- "straight AND tilted
    (edge-first), several repeats, across the fixture noise" -- and it is the
    only way the straight case is measured under the same controller, the same
    gains and the same noise draw as the tilted ones. Two separate runs would
    differ in all three and the comparison would carry those differences.

    ``num_envs == 1`` returns ``lo_rad`` for that one env: one env cannot carry
    a sweep and ``num_envs - 1`` would divide by zero.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if num_envs == 1:
        return torch.tensor([lo_rad], device=device)
    step = (hi_rad - lo_rad) / (num_envs - 1)
    return torch.tensor([lo_rad + step * i for i in range(num_envs)], device=device)


@torch.jit.script
def tilt_schedule(
    depth: torch.Tensor, tilt_start: torch.Tensor, upright_depth_m: float
) -> torch.Tensor:
    """The COMMANDED tilt at the current depth ``(N,)``, radians.

    ABOVE the opening plane (``depth <= 0``) the full start tilt is held: the
    part is in free air and the edge-first attitude is what the entry needs.
    From the opening down to ``upright_depth_m`` the target falls LINEARLY to
    zero, and below that it stays zero.

    A RAMP AND NOT A STEP, because the pocket forbids the tilt long before the
    seat: D-106 (2) records that at full seat only 0.935 deg across and
    1.905 / 2.821 deg along fit between the walls. A step to zero at one depth
    would ask the arm for a rotation the walls are already enforcing, and the
    force that produced would be a property of the schedule rather than of the
    task.

    LINEAR AND NOT MATCHED TO THE GEOMETRY, deliberately. The admissible angle
    as a function of depth is a CAD quantity nobody has computed; deriving a
    profile here would put a task number in this module, which is the one thing
    it does not do. ``upright_depth_m`` arrives from the script, and the run
    reports the force and the interpenetration THAT ramp produced.

    ``upright_depth_m <= 0`` means "no tilt at all" and returns zeros rather
    than dividing. That is the straight controller, reachable without a second
    code path.
    """
    if upright_depth_m <= 0.0:
        return torch.zeros_like(tilt_start)
    frac = torch.clamp(1.0 - depth / upright_depth_m, min=0.0, max=1.0)
    return tilt_start * frac


# ===========================================================================
#  The phase machine
# ===========================================================================


# NOT ``@torch.jit.script``, and the reason is MEASURED, not stylistic.
# RT-66 (2026-08-28, training PC, git 53c2126) died on the import with
# "python value of type 'float' cannot be used as a value. Perhaps it is a
# closed over global variable?" -- TorchScript refuses to close over the
# module-level phase codes below. torch's own message offers two ways out:
# pass the value in as an argument, or drop the compilation. Passing three
# codes through two signatures would push a fact into every call site, so the
# decorator goes instead.
#
# It costs nothing that matters. The house rule scripts HOT reward and
# observation arithmetic, which runs inside every PPO step; this controller
# runs only in the D-108 gate run. And ``insertion_math`` is untouched by the
# whole problem, because it owns no numbers by design -- that rule is what
# kept the failure on this side of the boundary.
def advance_phase(
    phase: torch.Tensor,
    lateral_err: torch.Tensor,
    yaw_err: torch.Tensor,
    tip_z: torch.Tensor,
    force_norm: torch.Tensor,
    lateral_tol: float,
    yaw_tol: float,
    standoff_m: float,
    height_tol: float,
    force_abort_n: float,
    allow_descend: bool = True,
) -> torch.Tensor:
    """One transition step, batched. Returns the new ``(N,)`` phase.

    ``allow_descend=False`` is the H1 FREE-SPACE mode: the ``ALIGN ->
    DESCEND`` transition is removed, so the part regulates its lateral aim and
    its yaw at ``standoff_m`` above the opening and never touches anything.
    That is the instrument D-108's H1 asks for -- "command a lateral offset of
    the SAPU order in free space in front of the opening and measure the
    tracking error through the same joint-delta chain"
    (`docs/decisions_inbox.md`, "The D-108 gate drops the scripted descent").

    THE FORCE ABORT STAYS ARMED IN THAT MODE, and it is not decoration: a
    free-space run that reports a contact force has stopped being a free-space
    run, and the tracking number it produced would be measuring a wall. It has
    to become visible, not be suppressed because the mode did not expect it.

    Two transitions and no more:

    * ``ALIGN -> DESCEND`` once the lateral error, the yaw error AND the
      HEIGHT are all inside tolerance. All three, not any: descending on a yaw
      that is still turning drives the part into the wall, which would charge
      the control chain for a scheduling mistake.

      THE HEIGHT CONDITION IS THERE BECAUSE RT-67 FAILED WITHOUT IT. That run
      measured the home pose as already aligned to 1 um laterally and 0 rad in
      yaw, so the gate opened on step 1 and DESCEND took over 165 mm above the
      opening -- and DESCEND moves at the FINE insertion rate. 165 mm at
      0.5 mm per control step is 330 steps against that run's 240-step
      episode (D-113's 256 landed later, and 330 still overruns it), so
      every one of the 32 episodes ended in DESCEND at depth 0.0. The fast
      approach that ALIGN regulates (clipped at ``max_step_m``, not at the
      descend rate) never ran at all. The two phases move at deliberately
      different speeds; without this condition the slow one owns the whole
      travel.

      Written ``tip_z <= standoff + tol`` rather than ``abs(...) < tol``: a tip
      already BELOW the standoff should descend, not climb back up to a
      band.
    * ``anything -> ABORT_FORCE`` once the smoothed force reaches the abort
      limit. STICKY: an env that aborted never leaves, so a force that dips
      back under the limit after a jam cannot resurrect the descent and hide
      the event from the statistics.

    WHERE THE STICKINESS ACTUALLY COMES FROM. Not from a second ``or`` term on
    the abort condition -- that was written first and the D-080 counter-proof
    showed it was DEAD: no mutation could break it, because the ALIGN
    transition is already gated on ``phase == PHASE_ALIGN`` and an aborted env
    therefore falls through unchanged. Redundant code that reads like a
    guarantee is worse than none, so it is gone and the fallthrough carries
    the property, with a mutation that breaks it on purpose.

    There is deliberately NO ``SEATED`` transition. Whether the part is seated
    is what this run MEASURES (``engaged_depth_m`` is unset, D-109 (5)); a
    phase that declared it would have to carry the threshold the run is
    supposed to produce.
    """
    placed = torch.logical_and(lateral_err < lateral_tol, torch.abs(yaw_err) < yaw_tol)
    at_height = tip_z <= standoff_m + height_tol
    aligned = torch.logical_and(placed, at_height)
    if not allow_descend:
        aligned = torch.zeros_like(aligned)
    advanced = torch.where(
        torch.logical_and(phase == PHASE_ALIGN, aligned),
        torch.full_like(phase, PHASE_DESCEND),
        phase,
    )
    aborting = force_norm >= force_abort_n
    return torch.where(aborting, torch.full_like(phase, PHASE_ABORT_FORCE), advanced)


@torch.jit.script
def update_peak_depth(depth: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    """Running per-env maximum of the insertion depth.

    The deepest point reached is the quantity ``engaged_depth_m`` will be cut
    from, and it is NOT the final depth: a part that seats and then rebounds
    on the contact spring would report the rebound.
    """
    return torch.maximum(depth, peak)


# ===========================================================================
#  Observation -> task-space command
# ===========================================================================


# NOT ``@torch.jit.script``, same measured reason as ``advance_phase`` above.
def command_in_pocket_frame(
    tip_rel: torch.Tensor,
    phase: torch.Tensor,
    yaw_err: torch.Tensor,
    aim: torch.Tensor,
    standoff_m: float,
    lateral_gain: float,
    vertical_gain: float,
    yaw_gain: float,
    descend_rate_m: float,
    max_step_m: float,
    max_yaw_step_rad: float,
    stop_depth_m: float = -1.0,
) -> torch.Tensor:
    """The per-step pose delta, expressed IN THE POCKET FRAME.

    Returns ``(N, 6)``: three translation components then an axis-angle
    rotation, the layout ``DifferentialIKController`` expects for a relative
    ``pose`` command.

    Per phase:

    * ``ALIGN`` -- null the lateral error AGAINST ``aim``, hold the tip at
      ``standoff_m`` ABOVE the opening, turn the yaw back to the latched
      target. ``aim`` is the swept lateral target (``aim_offsets``); at the
      default sweep it is zero and this is the pocket axis. Height is
      regulated rather than frozen because the arm sags under gravity
      (D-026) and a frozen height would drift down during a long alignment.
    * ``DESCEND`` -- keep correcting laterally toward ``aim`` and in yaw, and
      add a constant downward rate. The lateral term is NOT dropped once the
      descent starts: it is what HOLDS the swept offset. Dropped, the part
      would be placed at the offset and then left to whatever the contact
      does with it, and the sweep would measure nothing.
      Constant rate, not proportional to the remaining depth: the
      run is a measurement of what the control chain does at a known
      commanded speed, and a speed that decays with depth would confound the
      force reading with the approach profile.
    * ``ABORT_FORCE`` -- command exactly zero. The arm holds its last target
      and the episode's numbers stop changing, so the recorded peak force is
      the one that triggered the abort and not whatever a continued push
      produced afterwards.

    ``stop_depth_m`` ENDS THE DESCENT, and it is OFF (negative) by default for
    the same reason ``advance_phase``'s ``allow_descend`` defaults to the old
    behaviour: the D-108 gate run and ``scripted_insert.py`` share this
    function and must not change because a second caller needed something.

    WHY IT EXISTS, MEASURED. RT-128 (training PC, git 473a411) put 9 of 17
    envs at 35.46-36.02 mm in a 36.0 mm pocket -- the part reached the floor.
    DESCEND has no end, so it kept commanding a constant rate into that floor;
    2072 samples piled into the last millimetre against ~50 per bin
    everywhere else, and the force rose from ~20 N to the 300 N abort. 43 of
    51 episodes ended that way and NOT ONE was clean, so F_max stayed
    unmeasurable. The force those samples carry is the controller pushing on
    a hard stop, not the insertion.

    At or below the stop depth the downward rate becomes zero and the arm
    HOLDS its last target, exactly as ``ABORT_FORCE`` does. The lateral and
    yaw terms keep working: a seated part still has to be held where it is.

    Every component is clipped to ``max_step_m`` / ``max_yaw_step_rad``. The
    clip is what keeps the commanded step inside what ``action_scale`` can
    actually deliver in one control step -- an unclipped proportional term
    would ask for a jump the integrator can only serve over many steps, which
    is the accumulated-target-lag failure ``_max_target_lag`` was added to
    measure (D-037 diagnosis).

    ``tip_rel`` sign convention, taken from ``_peg_geometry``:
    ``tip_rel = tip - entrance`` in the pocket frame, so ``+z`` is ABOVE the
    opening and ``depth = -tip_rel_z``. Descending therefore DECREASES
    ``tip_rel_z``, which is why the descend term is negative.
    """
    lateral = torch.clamp((aim - tip_rel[:, 0:2]) * lateral_gain, -max_step_m, max_step_m)

    # ALIGN: drive the height toward +standoff. DESCEND: a fixed rate down.
    height_err = standoff_m - tip_rel[:, 2]
    hold = torch.clamp(height_err * vertical_gain, -max_step_m, max_step_m)
    sink = torch.full_like(hold, -descend_rate_m)
    # ``depth = -tip_rel_z`` is the convention this docstring states; comparing
    # tip_rel_z against the depth directly would invert the test and stop the
    # descent in free air instead of at the floor.
    if stop_depth_m >= 0.0:
        seated = -tip_rel[:, 2] >= stop_depth_m
        sink = torch.where(seated, torch.zeros_like(sink), sink)
    dz = torch.where(phase == PHASE_DESCEND, sink, hold)

    # The yaw correction turns about the POCKET z-axis, which is this frame's
    # z. Negative sign: yaw_err is (current - target), so the command must
    # undo it.
    d_yaw = torch.clamp(-yaw_err * yaw_gain, -max_yaw_step_rad, max_yaw_step_rad)
    zero = torch.zeros_like(d_yaw)

    active = phase != PHASE_ABORT_FORCE
    return torch.stack(
        (
            torch.where(active, lateral[:, 0], zero),
            torch.where(active, lateral[:, 1], zero),
            torch.where(active, dz, zero),
            zero,
            zero,
            torch.where(active, d_yaw, zero),
        ),
        dim=-1,
    )


def command_in_pocket_frame_tilted(
    tip_rel: torch.Tensor,
    phase: torch.Tensor,
    yaw_err: torch.Tensor,
    aim: torch.Tensor,
    tilt_err: torch.Tensor,
    tilt_axis: int,
    standoff_m: float,
    lateral_gain: float,
    vertical_gain: float,
    yaw_gain: float,
    descend_rate_m: float,
    max_step_m: float,
    max_yaw_step_rad: float,
    tilt_gain: float,
    max_tilt_step_rad: float,
    stop_depth_m: float = -1.0,
) -> torch.Tensor:
    """``command_in_pocket_frame`` with the ONE tilt column filled.

    The straight controller writes hard zeros into the axis-angle x and y
    components. This one CALLS it unchanged and then overwrites the single
    column the commanded tilt lives in. Reuse and not a copy: the lateral, the
    height and the yaw terms keep one home, and a second copy of them here
    would be a second place for the descend rate to be wrong.

    ``tilt_err`` is ``current - target``: ``signed_tilt_angle`` measured
    against ``tilt_schedule``. The command NEGATES it, exactly as the yaw term
    negates ``yaw_err``, and is clipped to ``max_tilt_step_rad`` for the reason
    every other component is clipped -- an unclipped proportional rotation asks
    the integrator for a jump it can only serve over many steps, which is the
    accumulated-target-lag failure.

    ``ABORT_FORCE`` zeroing is not repeated as a second condition. The base
    call already zeroes every active column on that phase; this line applies
    the SAME mask to the tilt column, so there is one rule and not two.

    Not ``@torch.jit.script``: it calls ``command_in_pocket_frame``, which is
    not scripted either, and for the measured reason recorded above
    ``advance_phase`` (TorchScript refuses to close over the module-level phase
    codes). This controller runs in a measurement run, never inside a PPO step.
    """
    cmd = command_in_pocket_frame(
        tip_rel,
        phase,
        yaw_err,
        aim,
        standoff_m,
        lateral_gain,
        vertical_gain,
        yaw_gain,
        descend_rate_m,
        max_step_m,
        max_yaw_step_rad,
        stop_depth_m,
    )
    d_tilt = torch.clamp(-tilt_err * tilt_gain, -max_tilt_step_rad, max_tilt_step_rad)
    # SPELLED DIFFERENTLY FROM THE BASE FUNCTION'S OWN MASK, and that is not
    # style. The D-080 counter-proof patches source TEXT, and its mutation
    # "an-aborted-env-keeps-pushing" names the base function's line verbatim.
    # A second identical line in this file makes that mutation ambiguous, the
    # harness refuses it, and a check that can no longer be broken proves
    # nothing. Caught by `check_scripted_insert.py --self-test` on the first
    # run of this function, 2026-09-01.
    tilt_active = phase != PHASE_ABORT_FORCE
    cmd[:, 3 + tilt_axis] = torch.where(tilt_active, d_tilt, torch.zeros_like(d_tilt))
    return cmd


@torch.jit.script
def command_to_base_frame(
    command_pocket: torch.Tensor,
    pocket_quat: torch.Tensor,
    root_quat: torch.Tensor,
) -> torch.Tensor:
    """Take the ``(N, 6)`` delta from the pocket frame to the robot base frame.

    Two hops, because the two rotations come from two different places and
    collapsing them would leave no place to check either:

    1. pocket -> env, with ``pocket_quat`` (observation channels 21:25).
    2. env -> base, with the robot root quaternion. The env frame and the
       world frame differ by a TRANSLATION only (``scene.env_origins``), and a
       direction is unaffected by a translation -- so the root quaternion,
       which Isaac Lab reports in world, is the whole of this hop. The base
       orientation is READ from ``robot.data.root_pose_w`` by the caller, never
       assumed to be identity.

    Both the translation triple and the axis-angle triple rotate the same way:
    an axis-angle vector is a vector, so it transforms as one.

    The Jacobian from ``root_physx_view.get_jacobians()`` is expressed in the
    ROOT frame (Isaac Lab's own IK example converts the EE pose into that
    frame before calling ``compute``:
    ``scripts/tutorials/05_controllers/run_diff_ik.py:167``), which is why
    this is the frame the command has to arrive in.
    """
    lin_base = rotate_by_quat_inv(root_quat, rotate_by_quat(pocket_quat, command_pocket[:, 0:3]))
    rot_base = rotate_by_quat_inv(root_quat, rotate_by_quat(pocket_quat, command_pocket[:, 3:6]))
    return torch.cat((lin_base, rot_base), dim=-1)


# ===========================================================================
#  Joint targets -> the env's action
# ===========================================================================


@torch.jit.script
def joint_delta_to_action(
    joint_pos_des: torch.Tensor,
    joint_targets: torch.Tensor,
    action_scale: float,
) -> torch.Tensor:
    """Convert an absolute desired joint position into the env's action.

    ``_pre_physics_step`` integrates ``targets += action_scale * clamp(a)``,
    so the action that asks for ``joint_pos_des`` is
    ``(joint_pos_des - targets) / action_scale``, clamped to the same
    ``[-1, 1]`` the env clamps to.

    IT SUBTRACTS THE TARGET, NOT THE MEASURED POSITION. The env's integrator
    is what the action feeds; differencing against the measured joint position
    would command the accumulated tracking error a second time, and while the
    part is pressed against the fixture and the arm cannot follow, that error
    is exactly the accumulating lag ``_max_target_lag`` exists to record
    (D-037 diagnosis, ``insertion_env.py:681``). The result would be a
    scripted run that fights its own integrator and blames the control chain
    for it -- i.e. the gate returning the wrong verdict.

    Clamping here rather than relying on the env's clamp is deliberate: the
    metrics file records the action this script CHOSE, and a value the env
    silently clipped would make that record a fiction.
    """
    return torch.clamp((joint_pos_des - joint_targets) / action_scale, -1.0, 1.0)


@torch.jit.script
def osc_action_from_pocket_command(
    command_pocket: torch.Tensor,
    pocket_quat: torch.Tensor,
    pos_step_limit_m: float,
    rot_step_limit_rad: float,
) -> torch.Tensor:
    """The ``(N, 6)`` pocket-frame pose delta as the env's OSC action.

    Under ``control_mode = "osc"`` (inbox entry "Audit 2026-09-03 (a)") the
    env's action IS a pose delta in the ENV frame, scaled by the per-step
    limits: ``delta = clamp(a, -1, 1) * limit``. So the scripted command
    needs ONE hop (pocket -> env, ``pocket_quat``, observation channels
    21:25) and a division; no IK, no base frame, no joint integrator --
    the env's controller does the rest every physics step. The env frame
    and the world frame differ by a translation only, so a delta needs no
    second rotation.

    Clamped here, not left to the env, for the reason ``joint_delta_to_action``
    gives: the metrics file records the action this script CHOSE. The
    scripted clips (``--max-step`` 2 mm, ``--max-yaw-step`` 0.01 rad) sit
    well inside the placeholder limits (20 mm, 0.097 rad), so in practice
    the clamp never bites and the action magnitude is about 0.1.
    """
    lin_env = rotate_by_quat(pocket_quat, command_pocket[:, 0:3])
    rot_env = rotate_by_quat(pocket_quat, command_pocket[:, 3:6])
    return torch.cat(
        (
            torch.clamp(lin_env / pos_step_limit_m, -1.0, 1.0),
            torch.clamp(rot_env / rot_step_limit_rad, -1.0, 1.0),
        ),
        dim=-1,
    )


# ===========================================================================
#  The lateral AIM. Added after RT-74.
# ===========================================================================
#
# WHY IT IS NOT SIMPLY THE POCKET CENTRE. Every run from RT-68 to RT-74 aimed
# at ``tip_rel[:, 0:2] == 0`` and hit it to 1 um with the tool axis exactly
# parallel -- and the part still stopped at 19.5 mm of the 36 mm seat at
# ~204 N. RT-74's force-against-depth profile showed the force rising SMOOTHLY
# from ~8 mm with no jump at any bin, which points at friction over the travel
# rather than at an obstacle engaging at one depth.
#
# The centre is ONE point of a whole play window that nothing has tried:
# 0.5876 mm across (``PLAY_X``) and 1.60 mm along (``PLAY_Y``). So the aim
# becomes a swept quantity and the run measures peak depth AGAINST it.
#
# CONSTANT PER ENV FOR THE WHOLE RUN, and that is the point: the offset is not
# episode state. A value re-drawn at reset would mix offsets inside one env's
# statistics and make the table unreadable.
#
# OWNS NO NUMBERS: the span arrives as an argument, from the task config's
# ``PLAY_X`` / ``PLAY_Y`` via the script's measurement-settings group.
AIM_AXES: dict = {"none": -1, "x": 0, "y": 1}


def aim_offsets(num_envs: int, axis: str, span_m: float, device=None) -> torch.Tensor:
    """Per-env lateral aim ``(N, 2)`` in the pocket frame.

    Column 0 is the pocket's short axis, column 1 the long one -- the same
    order as ``tip_rel[:, 0:2]``, so the two subtract directly.

    ``axis="none"`` returns zeros, which reproduces every run before RT-75
    exactly. Otherwise the chosen column runs linearly from ``-span_m / 2`` to
    ``+span_m / 2``, endpoints INCLUDED: the endpoints are where the part's
    wall meets the pocket's, and that contact is the signal the sweep exists
    to find, not something to keep clear of.

    ``num_envs == 1`` returns zeros on every axis. One env cannot carry a
    sweep, and dividing by ``num_envs - 1`` would be a division by zero.
    """
    if axis not in AIM_AXES:
        raise ValueError(f"axis must be one of {sorted(AIM_AXES)}, got {axis!r}")
    if num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if span_m < 0.0:
        raise ValueError(f"span_m must not be negative, got {span_m}")
    col = AIM_AXES[axis]
    rows = [[0.0, 0.0] for _ in range(num_envs)]
    if col >= 0 and num_envs > 1:
        for i in range(num_envs):
            rows[i][col] = -span_m / 2.0 + span_m * i / (num_envs - 1)
    return torch.tensor(rows, device=device)


@torch.jit.script
def lateral_error(tip_rel: torch.Tensor, aim: torch.Tensor) -> torch.Tensor:
    """Planar distance from the AIM, in the pocket frame ``(N,)``.

    Only the two in-plane components: the vertical offset is the approach, not
    an error, and folding it in would keep ``ALIGN`` from ever completing.

    Measured against ``aim`` and not against the pocket axis, because the
    ALIGN -> DESCEND gate compares this against ``lateral_tol`` (0.2 mm by
    default). Against the axis, a 0.29 mm sweep offset would read as a
    permanent 0.29 mm error and the gate would never open.
    """
    return torch.linalg.norm(tip_rel[:, 0:2] - aim, dim=-1)


# ---------------------------------------------------------------------------
# H1: how well the joint-delta chain HOLDS a commanded lateral offset.
#
# D-108's H1 says terminal accuracy is set by PD tracking rather than by the
# action scale, and that "whether the control chain can command a 0.144 mm
# placement is unproven". THE 0.144 mm IS D-108's OWN WORDING AND IT IS STALE:
# it was half the superseded cross play. The live tolerance is 0.2938 mm (half
# the play D-121 doubled, decided 2026-08-30), which is what RT-84 measured
# against. Quoted rather than rewritten, because it is what D-108 says.
# The instrument the 2026-08-29 decision assigns to it
# is a free-space run: hold the part above the opening, command a lateral
# offset of the play's order, and read the RESIDUAL.
#
# Two numbers, and neither is a mean over the whole run. A mean over the run
# is dominated by the approach transient, which is not what H1 asks about.
# ---------------------------------------------------------------------------
def track_settle(errors: list, tol_m: float, tail: int) -> dict:
    """Settling of ONE env's lateral-error trace. Pure Python, no tensors.

    ``errors`` is that env's lateral error in metres, one entry per control
    step, in order. Returns::

        {"settle_step": int, "steady_max_m": float, "tail": int}

    ``settle_step`` is the first index from which the error stays STRICTLY
    below ``tol_m`` for every remaining step, and ``-1`` when no such index
    exists. Found by walking BACKWARDS, which is what makes a late excursion
    count: an error that dips under the tolerance at step 20, leaves it again
    at step 200 and returns at step 210 settled at 210, not at 20. The
    forward reading ("first step under tolerance") would call that chain
    settled while it was still swinging, and the swing is exactly the failure
    H1 is looking for.

    ``steady_max_m`` is the MAXIMUM over the last ``tail`` steps, not their
    mean. A mean hides an oscillation of twice its own size; a maximum
    reports the worst placement the chain actually delivered, which is the
    number the play has to accommodate.

    OWNS NO THRESHOLD. Both ``tol_m`` and ``tail`` arrive from the caller, and
    the verdict against the play is the caller's too.
    """
    n = len(errors)
    if n == 0:
        raise ValueError("errors is empty; a settling reading needs at least one step")
    if tail < 1:
        raise ValueError(f"tail must be positive, got {tail}")
    if tol_m <= 0.0:
        raise ValueError(f"tol_m must be positive, got {tol_m}")

    settle_step = 0
    for i in range(n - 1, -1, -1):
        if not errors[i] < tol_m:
            settle_step = i + 1
            break
    if settle_step >= n:
        settle_step = -1

    window = errors[max(0, n - tail):]
    return {"settle_step": settle_step, "steady_max_m": max(window), "tail": len(window)}


def first_depth_below(rows: list, value_key: str, tol: float):
    """The shallowest depth from which ``value_key`` STAYS below ``tol``, in mm.

    ``rows`` is a ``depth_profile`` output, so it is already ordered by depth
    and already carries each row's own depth range. Returns the ``depth_lo_mm``
    of that row, or ``None`` when no such row exists.

    THE RULE IS ``track_settle``'s, not a second one: walk BACKWARDS and take
    the first index from which every deeper row stays under the tolerance. A
    forward reading would report a bin where the quantity dips under the
    tolerance once and leaves it again two bins deeper -- and for the capture
    question that is precisely the wrong answer, because what is being asked is
    from where the pocket HOLDS the part, not where it first touched it.

    It is called with ``tail=1`` because the tail window is a settling
    statistic over time and there is no time here; only the index is used.

    OWNS NO THRESHOLD AND NAMES NO QUANTITY. ``value_key`` and ``tol`` arrive
    from the caller, and what a capture depth IS remains the decision this run
    reports the input for.
    """
    values = [r[value_key] for r in rows]
    if not values:
        return None
    settled = track_settle(values, tol, 1)
    if settled["settle_step"] < 0:
        return None
    return rows[settled["settle_step"]]["depth_lo_mm"]


# ---------------------------------------------------------------------------
# Force AGAINST depth. Added after RT-73, and it answers a question the peak
# numbers structurally cannot.
#
# WHAT THE PEAKS COULD NOT SAY. RT-68 through RT-72 all report one pair per
# episode: the deepest depth and the highest force. Two points. Every run
# stopped at ~19.5 mm of the 36 mm seat at ~204 N, with the axis tilt exactly
# 0 and the lateral error at 1 um -- so the part is straight, centred, and
# still stops. The user then WATCHED it (RT-73) and saw nothing wrong. A
# failure that is invisible and has no recorded geometry at its stopping depth
# can only be separated by WHEN the force arrives:
#
#   force rises smoothly from first contact  -> friction over the whole
#       travel, i.e. binding in the 0.5876 mm cross play. No obstacle.
#   force jumps at one depth                 -> something engages THERE, and
#       that depth names it.
#
# These are different causes with different fixes, and nothing measured so far
# distinguishes them.
#
# PURE PYTHON ON PURPOSE, and it is the one function in this module that is
# not a tensor op. The caller hands over one step's values as plain lists
# (4-16 numbers), so there is no per-env tensor loop and no new torch op for
# the offline stand-in to grow. It is checked by executing this module, like
# everything else here.
#
# OWNS NO NUMBERS: the bin width and the bin count arrive as arguments, from
# the script's measurement-settings group.
# ---------------------------------------------------------------------------
def bin_force_by_depth(
    depths: list,
    forces: list,
    bin_width_m: float,
    counts: list,
    force_max: list,
    force_sum: list,
) -> None:
    """The force binner: ``bin_values_by_depth`` under its original name.

    Kept as a name because ``scripted_insert.py`` and its checks call it, and
    because "bin the FORCE by depth" is what the D-108 gate run does. The
    mechanics moved down to ``bin_values_by_depth`` on 2026-09-01, when the
    tilt run needed the same accumulation for interpenetration, tilt angle and
    lateral offset. One implementation, four quantities -- a second copy would
    be a second place for the drop rules below to be wrong.
    """
    bin_values_by_depth(depths, forces, bin_width_m, counts, force_max, force_sum)


def bin_values_by_depth(
    depths: list,
    values: list,
    bin_width_m: float,
    counts: list,
    value_max: list,
    value_sum: list,
) -> None:
    """Accumulate one step's (depth, value) pairs into depth bins, IN PLACE.

    ``counts``, ``value_max`` and ``value_sum`` are the accumulators and must
    all have the same length, which is the bin count. Bin ``b`` covers
    ``[b * bin_width_m, (b + 1) * bin_width_m)``.

    A NEGATIVE depth is DROPPED, not clamped into bin 0. The tip starts
    165 mm above the opening, so most of every episode is spent at a negative
    depth carrying only the tool's own weight; folding that into the first bin
    would bury the first real contact under hundreds of free-air samples --
    which is the one thing this instrument exists to find. A depth beyond the
    last bin is dropped for the same reason: it is not measured wrongly, it is
    outside what was asked for, and silently piling it onto the last bin would
    invent a peak there.

    Returns nothing. The accumulators are mutated, because they live across
    the whole run and copying them per step would be the only cost of note.
    """
    n_bins = len(counts)
    if len(value_max) != n_bins or len(value_sum) != n_bins:
        raise ValueError(
            f"accumulators disagree: counts {n_bins}, value_max {len(value_max)}, "
            f"value_sum {len(value_sum)}"
        )
    if bin_width_m <= 0.0:
        raise ValueError(f"bin_width_m must be positive, got {bin_width_m}")
    if len(depths) != len(values):
        raise ValueError(f"{len(depths)} depths against {len(values)} values")
    for depth, value in zip(depths, values):
        if depth < 0.0:
            continue
        b = int(depth / bin_width_m)
        if b >= n_bins:
            continue
        counts[b] += 1
        value_sum[b] += value
        if value > value_max[b]:
            value_max[b] = value


def aim_sweep_rows(aim_mm: list, episode_env_ids: list) -> list:
    """One row per ENV, sorted by its aim -- the RT-75 table's skeleton.

    ``aim_mm`` is the full per-env aim table (``num_envs`` rows of ``[x, y]``,
    millimetres), ``episode_env_ids`` the env id of every finished episode in
    the order they were recorded. Each row carries the INDICES of its episodes:

        [{"env": 3, "aim_x_mm": -0.1175, "aim_y_mm": 0.0, "episodes": [3, 19]}, ...]

    It deliberately computes no statistic. The caller already owns
    ``_percentiles``, and a second median implementation here would be a
    second place for the same number to be wrong.

    EVERY ENV GETS A ROW, including one that finished no episode -- that row
    shows ``episodes: []``. An env dropped for being empty would look like a
    sweep with a missing offset instead of like an env that never terminated,
    and those are different findings. (This is the opposite choice from
    ``force_depth_profile``, where an unvisited bin carries no information: a
    depth nobody reached is not a result, an env that never finished is.)

    Sorted by aim and not by env index, because the table is read as a curve.
    """
    rows = []
    for i, a in enumerate(aim_mm):
        if len(a) != 2:
            raise ValueError(f"aim row {i} has {len(a)} components, expected 2")
        rows.append({"env": i, "aim_x_mm": a[0], "aim_y_mm": a[1], "episodes": []})
    for k, env_id in enumerate(episode_env_ids):
        if not 0 <= env_id < len(rows):
            raise ValueError(f"episode {k} names env {env_id}, outside 0..{len(rows) - 1}")
        rows[env_id]["episodes"].append(k)
    rows.sort(key=lambda r: (r["aim_x_mm"], r["aim_y_mm"], r["env"]))
    return rows


def force_depth_profile(
    bin_width_m: float,
    counts: list,
    force_max: list,
    force_sum: list,
) -> list:
    """The force profile: ``depth_profile`` under its original name and keys.

    Emits ``force_max_n`` / ``force_mean_n``, unchanged since RT-74, so every
    reader of ``scripted_insert_metrics.json`` keeps working. The mechanics
    moved to ``depth_profile`` on 2026-09-01 for the same reason the binner
    did: the tilt run reads three more quantities over the same depth axis.
    """
    return depth_profile(bin_width_m, counts, force_max, force_sum, "force", "n")


def depth_profile(
    bin_width_m: float,
    counts: list,
    value_max: list,
    value_sum: list,
    value_key: str,
    unit: str,
) -> list:
    """The accumulators as a list of per-bin dicts. EMPTY BINS ARE DROPPED.

    Dropping them is what makes the profile readable: at a 1 mm width most of
    the 40 bins below the seat are never visited, and a table of zeros hides
    the handful of rows that carry the answer. Each row keeps its own depth
    range, so a missing row is visible as a gap rather than as a shifted index.

    ``value_key`` and ``unit`` NAME the quantity in the output keys
    (``<value_key>_max_<unit>``, ``<value_key>_mean_<unit>``) because a metrics
    file that says ``max`` without saying max OF WHAT, IN WHAT, is a file the
    next reader has to guess at. The tilt run writes four profiles into one
    JSON and they must not be distinguishable only by their position.
    """
    out = []
    for b, n in enumerate(counts):
        if n == 0:
            continue
        out.append({
            "depth_lo_mm": b * bin_width_m * 1000.0,
            "depth_hi_mm": (b + 1) * bin_width_m * 1000.0,
            "samples": n,
            f"{value_key}_max_{unit}": value_max[b],
            f"{value_key}_mean_{unit}": value_sum[b] / n,
        })
    return out
