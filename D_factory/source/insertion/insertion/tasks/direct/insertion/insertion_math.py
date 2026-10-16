# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure reward / observation / termination math for the UR5e insertion."""

from __future__ import annotations

import math

import torch


@torch.jit.script
def axes_from_quat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) wxyz quaternion batch -> (N, 3, 3), the body axes as COLUMNS."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    col0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    col1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
    col2 = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return torch.stack([col0, col1, col2], dim=-1)


@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (N, 4) wxyz batches: ``a`` then ``b``."""
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
    """(N, 4) wxyz -> the inverse rotation."""
    return torch.stack((q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]), dim=-1)


@torch.jit.script
def rotate_into_frame(rot: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Express (N, 3) vectors in the frame whose axes are the COLUMNS of ``rot``."""
    return torch.sum(rot * v.reshape(-1, 3, 1), dim=1)


@torch.jit.script
def relative_pose(
    pos_a: torch.Tensor,
    quat_a: torch.Tensor,
    pos_b: torch.Tensor,
    quat_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pose B in frame A: inverse(T_a) @ T_b."""
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
    """Where the part's task frame sits relative to the pocket opening."""
    part_axes = axes_from_quat(body_quat)
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
    """The part's SEATED goal pose, per env: ``(goal_pos, goal_quat)``."""
    goal_quat = quat_mul(fixture_quat, seat_quat_local)
    pocket_rot = axes_from_quat(fixture_quat)
    goal_tip = entrance_pos - seat_depth * pocket_rot[:, :, 2]
    goal_pos = goal_tip - torch.matmul(axes_from_quat(goal_quat), tip_offset_local)
    return goal_pos, goal_quat


@torch.jit.script
def squash(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """The Factory/FORGE squashing function ``1 / (exp(a*x) + b + exp(-a*x))``."""
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
    """Three squashed kernels over ONE SDF distance, peaking at 1.0 together."""
    return (
        squash(sdf_dist, a_coarse, b_coarse)
        + squash(sdf_dist, a_mid, b_mid)
        + squash(sdf_dist, a_fine, b_fine)
    )


@torch.jit.script
def sapu_reward_scale(interpen_dist: torch.Tensor, interpen_thresh: float) -> torch.Tensor:
    """SAPU: scale the return by interpenetration (D-109 point (10))."""
    scale = 1.0 - torch.tanh(interpen_dist / interpen_thresh)
    return torch.where(interpen_dist >= interpen_thresh, torch.zeros_like(scale), scale)


REWARD_TERMS: tuple = (
    "kernels",
    "engaged",
    "success",
    "progress",
    "time",
    "action_rate",
    "success_lump",
    "abort",         # abort_payment on the force-abort step
    "tilt_shaping",  # gamma * Phi(s') - Phi(s), Phi = w_tilt * sech(a * theta) vs the POCKET axis; Phi(s') = 0 on the success step (RT-171)
)


@torch.jit.script
def tilt_cos_theta(body_quat: torch.Tensor, pocket_rot: torch.Tensor) -> torch.Tensor:
    """``cos theta`` (N,): the part's +z axis against the pocket's -z axis."""
    axes_in_pocket = torch.matmul(pocket_rot.transpose(1, 2), axes_from_quat(body_quat))
    return -axes_in_pocket[:, 2, 2]


@torch.jit.script
def tilt_potential(cos_theta: torch.Tensor, w_tilt: float, a_tilt: float) -> torch.Tensor:
    """``Phi(s) = w_tilt * sech(a_tilt * theta)`` (N,), the alignment potential (RT-171)."""
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
    """The per-step reward as a ``[len(REWARD_TERMS), N]`` stack of its terms."""
    kernels = kernel_sum(sdf_dist, a_coarse, b_coarse, a_mid, b_mid, a_fine, b_fine)
    zero = torch.zeros_like(kernels)
    engaged_pay = torch.where(engaged, torch.full_like(kernels, w_engaged), zero)
    success_pay = torch.where(success, torch.full_like(kernels, w_success), zero)
    scale = sapu_reward_scale(interpen_dist, interpen_thresh)
    kernels_row = kernels * scale
    engaged_row = engaged_pay * scale
    success_row = success_pay * scale
    step_task = kernels_row + engaged_row + success_row
    progress_pay = w_progress * depth_progress * scale
    time_pay = torch.full_like(step_task, time_penalty_per_step)
    rate_pay = -action_rate_scale * action_rate
    lump = torch.where(success, step_task * steps_remaining, zero)
    payment = torch.full_like(step_task, abort_payment)
    abort_pay = torch.where(force_abort & (~success), payment, zero)
    tilt_phi_next = torch.where(success, zero, tilt_phi)
    tilt_pay = shaping_gamma * tilt_phi_next - tilt_phi_prev
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
    """The whole per-step reward: the sum of ``compute_reward_terms_insertion``."""
    return torch.sum(compute_reward_terms_insertion(
        sdf_dist, interpen_dist, engaged, success, force_abort, action_rate,
        steps_remaining, depth_progress, tilt_phi, tilt_phi_prev,
        a_coarse, b_coarse, a_mid, b_mid,
        a_fine, b_fine, w_engaged, w_success, w_progress, interpen_thresh,
        abort_payment, time_penalty_per_step, action_rate_scale,
        shaping_gamma,
    ), dim=0)


OBS_SLICES: dict = {
    "joint_pos": (0, 6),
    "joint_vel": (6, 12),
    "tip_rel": (12, 15),
    "ee_quat": (15, 19),
    "yaw_cos_sin": (19, 21),
    "pocket_quat": (21, 25),
    "force": (25, 28),        # three force components, EMA-smoothed, D-114
}
OBS_DIM: int = 28

OBS_MODES: tuple = ("force", "wrench", "force_prevact")
TORQUE_WIDTH: int = 3
PREV_ACTION_WIDTH: int = 6


def obs_slices(mode: str) -> dict:
    """The slice table of one observation mode -- ``OBS_SLICES`` plus, in ``wrench`` mode, ``torque`` appended at the end, and in ``force_prevact`` mode ``prev_action`` appended at the end."""
    if mode not in OBS_MODES:
        raise ValueError(f"obs mode {mode!r} is not one of {OBS_MODES}")
    out = dict(OBS_SLICES)
    if mode == "wrench":
        out["torque"] = (OBS_DIM, OBS_DIM + TORQUE_WIDTH)
    if mode == "force_prevact":
        out["prev_action"] = (OBS_DIM, OBS_DIM + PREV_ACTION_WIDTH)
    return out


def obs_dim(mode: str) -> int:
    """Width of one observation mode: the end of its last slice."""
    return max(hi for _, hi in obs_slices(mode).values())


def obs_version(mode: str) -> str:
    """One string naming the layout a checkpoint was trained on: ``force-28`` or ``wrench-31``."""
    return f"{mode}-{obs_dim(mode)}"


def noise_masks(
    *,
    pocket_pos_std_m: float,
    force_std_n: float,
    torque_std_nm: float = 0.0,
    mode: str = "force",
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two per-channel sigma vectors of the observation noise (D-182)."""
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
    """Force ``w >= 0`` on a wxyz quaternion batch (D-037)."""
    flip = q[:, 0:1] < 0.0
    return torch.where(flip, -q, q)


@torch.jit.script
def yaw_cos_sin(x_axis: torch.Tensor) -> torch.Tensor:
    """Encode the part's yaw as ``(cos phi, sin phi)`` -- D-107 point (2)."""
    phi = torch.atan2(x_axis[:, 1], x_axis[:, 0])
    return torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1)


@torch.jit.script
def believed_pocket_frame(
    pocket_quat: torch.Tensor, yaw_err: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pocket frame the policy BELIEVES in (D-194): ``(axes, quat)``."""
    half = 0.5 * yaw_err
    zero = torch.zeros_like(half)
    q_err = torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)
    q = canonicalize_quat(quat_mul(pocket_quat, q_err))
    return axes_from_quat(q), q


@torch.jit.script
def ema_update(prev: torch.Tensor, new: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exponential moving average, ``alpha`` weighting the NEW sample."""
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
    """Concatenate the seven blocks into the 28-channel policy observation."""
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
    """The ``wrench`` layout (D-188): the ``force`` layout with the three torque components APPENDED."""
    return torch.cat(
        (assemble_observation(joint_pos, joint_vel, tip_rel, ee_quat, yaw_cs, pocket_quat, force),
         torque),
        dim=-1,
    )


@torch.jit.script
def assemble_observation_prevact(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    tip_rel: torch.Tensor,
    ee_quat: torch.Tensor,
    yaw_cs: torch.Tensor,
    pocket_quat: torch.Tensor,
    force: torch.Tensor,
    prev_action: torch.Tensor,
) -> torch.Tensor:
    """The ``force_prevact`` layout (RT-233): the ``force`` layout with the six previous-action channels APPENDED, built like ``assemble_observation_wrench`` so the first 28 channels are the same function's output in every mode."""
    return torch.cat(
        (assemble_observation(joint_pos, joint_vel, tip_rel, ee_quat, yaw_cs, pocket_quat, force),
         prev_action),
        dim=-1,
    )


@torch.jit.script
def gravity_tare(
    force_raw: torch.Tensor,
    parent_quat_w: torch.Tensor,
    hold_force_w: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """Remove the welded tool's OWN WEIGHT from the joint reaction force."""
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
    """Remove the welded tool's OWN WEIGHT MOMENT from the joint reaction torque (D-188) -- the torque twin of ``gravity_tare``."""
    if not enabled:
        return torque_raw
    hold = hold_force_w.reshape(1, 3) * torch.ones_like(torque_raw)
    moment_w = torch.cross(lever_w, hold, dim=-1)
    return torque_raw - rotate_into_frame(axes_from_quat(parent_quat_w), moment_w)


@torch.jit.script
def force_magnitude(force: torch.Tensor) -> torch.Tensor:
    """``||F||`` from the three force components (D-114)."""
    return torch.linalg.norm(force, dim=-1)


@torch.jit.script
def compute_dones(
    episode_length_buf: torch.Tensor,
    max_episode_length: int,
    force_norm: torch.Tensor,
    f_max: float,
    success_now: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The exits: TWO terminal sources and the timeout (D-113 point (6))."""
    force_abort = force_norm >= f_max
    timeout = episode_length_buf >= max_episode_length - 1
    terminated = success_now
    truncated = (timeout | force_abort) & (~success_now)
    return terminated, truncated, force_abort


@torch.jit.script
def in_pocket_cross_section(tip_rel: torch.Tensor, wall_x: float, wall_y: float) -> torch.Tensor:
    """Is the part tip LATERALLY over the pocket opening?"""
    return (tip_rel[:, 0].abs() < wall_x) & (tip_rel[:, 1].abs() < wall_y)


@torch.jit.script
def in_pocket_volume(
    cross_section: torch.Tensor, depth: torch.Tensor, underside_depth: float
) -> torch.Tensor:
    """Is the part tip in the pocket, and not BELOW the fixture?"""
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
    """Is the part seated RIGHT NOW?"""
    deep_enough = depth >= depth_min
    not_too_deep = depth <= depth_max
    clean = interpen_dist < interpen_thresh
    return deep_enough & not_too_deep & clean & in_pocket


@torch.jit.script
def interpen_unclean(interpen_dist: torch.Tensor, interpen_thresh: float) -> torch.Tensor:
    """Is this step's interpenetration AT OR ABOVE the SAPU threshold?"""
    return interpen_dist >= interpen_thresh


@torch.jit.script
def is_engaged(depth: torch.Tensor, engaged_depth: float, in_pocket: torch.Tensor) -> torch.Tensor:
    """The \"engaged\" display bonus predicate (D-109 point (5))."""
    return (depth >= engaged_depth) & in_pocket


@torch.jit.script
def interpen_from_signed(signed_dist: torch.Tensor) -> torch.Tensor:
    """Turn a SIGNED distance array into the source's interpenetration array."""
    return torch.clamp(signed_dist, max=0.0)


@torch.jit.script
def max_interpen_dist(per_point_dist: torch.Tensor) -> torch.Tensor:
    """Reduce per-sample-point interpenetration to ONE depth per env."""
    return torch.amax(torch.clamp(-per_point_dist, min=0.0), dim=-1)


@torch.jit.script
def quat_from_axis_angle(aa: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """(N, 3) axis-angle vector -> (N, 4) wxyz unit quaternion."""
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
    """Rotate (N, 3) vectors by (N, 4) wxyz quaternions: ``R(q) @ v``."""
    rot = axes_from_quat(q)
    return torch.sum(rot * v.reshape(-1, 1, 3), dim=2)


@torch.jit.script
def apply_pose_delta(
    pos: torch.Tensor,
    quat: torch.Tensor,
    delta_pos: torch.Tensor,
    delta_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``target = current + delta``: (N, 3) + (N, 3) metres, and the (N, 3) axis-angle rotation PRE-multiplied onto the (N, 4) wxyz orientation."""
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
    """Keep the LEADING TOOL POINT inside an axis-aligned box around ``centre`` and return the body position that puts it there."""
    offset_w = quat_rotate(quat, tip_offset_local.reshape(1, 3) + torch.zeros_like(pos))
    tip = pos + offset_w
    d = tip - centre
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
    """Angle (N,) in radians between the tool's +z axis and the world DOWN direction ``(0, 0, -1)``."""
    axis = axes_from_quat(quat)[:, :, 2]
    return torch.acos(torch.clamp(-axis[:, 2], min=-1.0, max=1.0))


@torch.jit.script
def clamp_tilt_to_cone(
    quat: torch.Tensor,
    max_tilt_rad: float,
    ref_quat: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Pull an (N, 4) wxyz orientation back until its tool axis leans at most ``max_tilt_rad`` from THIS EPISODE'S POCKET AXIS; orientations inside the cone come back bit-identical."""
    ref = axes_from_quat(ref_quat)
    tool = axes_from_quat(quat)[:, :, 2]
    down = -ref[:, :, 2]
    tilt = torch.acos(torch.clamp((tool * down).sum(dim=-1), min=-1.0, max=1.0))
    excess = torch.clamp(tilt - max_tilt_rad, min=0.0)
    n = torch.cross(tool, down, dim=-1)
    n_norm = torch.linalg.norm(n, dim=-1)
    degenerate = n_norm < eps
    n_safe = torch.where(degenerate.reshape(-1, 1), ref[:, :, 0], n)
    n_unit = n_safe / torch.where(degenerate, torch.ones_like(n_norm), n_norm).reshape(-1, 1)
    fix = quat_from_axis_angle(n_unit * excess.reshape(-1, 1))
    return quat_mul(fix, quat)


@torch.jit.script
def clamp_yaw_to_band(
    quat: torch.Tensor,
    max_yaw_rad: float,
    ref_quat: torch.Tensor,
) -> torch.Tensor:
    """Turn an (N, 4) wxyz orientation back about the reference axis until the yaw of its x axis lies within ``+-max_yaw_rad``; orientations inside the band come back bit-identical (D-196 (A))."""
    ref = axes_from_quat(ref_quat)
    x_ref = rotate_into_frame(ref, axes_from_quat(quat)[:, :, 0])
    phi = torch.atan2(x_ref[:, 1], x_ref[:, 0])
    excess = torch.clamp(torch.abs(phi) - max_yaw_rad, min=0.0)
    turn = -torch.sign(phi) * excess
    fix = quat_from_axis_angle(ref[:, :, 2] * turn.reshape(-1, 1))
    return quat_mul(fix, quat)


@torch.jit.script
def disk_offset(u_r: torch.Tensor, u_phi: torch.Tensor, r_max: float) -> torch.Tensor:
    """(N,) unit draws for radius and angle -> (N, 2) pocket-frame (x, y)."""
    r = r_max * torch.sqrt(u_r)
    phi = 2.0 * math.pi * u_phi
    return torch.stack((r * torch.cos(phi), r * torch.sin(phi)), dim=-1)
