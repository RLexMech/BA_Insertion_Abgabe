# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted insertion controller for approach, alignment and seating."""

from __future__ import annotations

import torch

from .insertion_math import axes_from_quat, rotate_into_frame

PHASE_ALIGN = 0.0
PHASE_DESCEND = 1.0
PHASE_ABORT_FORCE = 2.0

PHASE_NAMES: dict = {
    PHASE_ALIGN: "ALIGN",
    PHASE_DESCEND: "DESCEND",
    PHASE_ABORT_FORCE: "ABORT_FORCE",
}


@torch.jit.script
def latch_yaw_target(
    yaw_cs: torch.Tensor,
    target_cs: torch.Tensor,
    latched: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture the episode's starting yaw once per env."""
    new_target = torch.where(latched > 0.5, target_cs, yaw_cs)
    return new_target, torch.ones_like(latched)


@torch.jit.script
def signed_angle_difference(a_cs: torch.Tensor, b_cs: torch.Tensor) -> torch.Tensor:
    """Signed angle ``a - b``, wrapped to ``(-pi, pi]``, from ``(cos, sin)`` pairs."""
    cos_a = a_cs[:, 0]
    sin_a = a_cs[:, 1]
    cos_b = b_cs[:, 0]
    sin_b = b_cs[:, 1]
    return torch.atan2(sin_a * cos_b - cos_a * sin_b, cos_a * cos_b + sin_a * sin_b)


@torch.jit.script
def yaw_error(yaw_cs: torch.Tensor, target_cs: torch.Tensor) -> torch.Tensor:
    """Signed angle ``phi - phi_target``, wrapped to ``(-pi, pi]``."""
    return signed_angle_difference(yaw_cs, target_cs)


@torch.jit.script
def rotate_out_of_frame(rot: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``R v`` -- the exact inverse direction of ``insertion_math.rotate_into_frame``."""
    return torch.sum(rot * v.reshape(-1, 1, 3), dim=2)


@torch.jit.script
def rotate_by_quat(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """``R(quat) v`` -- take ``(N, 3)`` FROM the quat's frame INTO its parent."""
    return rotate_out_of_frame(axes_from_quat(quat), vec)


@torch.jit.script
def rotate_by_quat_inv(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """``R(quat)^T v`` -- take ``(N, 3)`` FROM the parent INTO the quat's frame."""
    return rotate_into_frame(axes_from_quat(quat), vec)


@torch.jit.script
def axis_tilt_angle(part_quat: torch.Tensor, pocket_quat: torch.Tensor) -> torch.Tensor:
    """Angle in RADIANS between the part's insertion axis and the pocket axis."""
    part_axis = axes_from_quat(part_quat)[:, :, 2]
    pocket_axis = axes_from_quat(pocket_quat)[:, :, 2]
    cos = torch.sum(part_axis * pocket_axis, dim=-1)
    return torch.acos(torch.clamp(cos, min=-1.0, max=1.0))


TILT_AXES: dict = {"x": 0, "y": 1}


@torch.jit.script
def signed_tilt_angle(
    part_quat: torch.Tensor, pocket_quat: torch.Tensor, axis: int
) -> torch.Tensor:
    """SIGNED tilt of the part axis about ONE pocket in-plane axis, radians."""
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
    """Capture the episode's starting tilt once per env, held as ``(cos, sin)``."""
    now_cs = torch.stack((torch.cos(tilt_now), torch.sin(tilt_now)), dim=1)
    new_ref = torch.where(latched > 0.5, ref_cs, now_cs)
    return new_ref, torch.ones_like(latched)


@torch.jit.script
def tilt_since_reference(tilt_now: torch.Tensor, ref_cs: torch.Tensor) -> torch.Tensor:
    """Tilt measured AWAY FROM the latched start attitude ``(N,)``, radians."""
    now_cs = torch.stack((torch.cos(tilt_now), torch.sin(tilt_now)), dim=1)
    return signed_angle_difference(now_cs, ref_cs)


def tilt_offsets(num_envs: int, lo_rad: float, hi_rad: float, device=None) -> torch.Tensor:
    """Per-env commanded START tilt ``(N,)`` in radians, endpoints INCLUDED."""
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
    """The COMMANDED tilt at the current depth ``(N,)``, radians."""
    if upright_depth_m <= 0.0:
        return torch.zeros_like(tilt_start)
    frac = torch.clamp(1.0 - depth / upright_depth_m, min=0.0, max=1.0)
    return tilt_start * frac


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
    """One transition step, batched."""
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
    """Running per-env maximum of the insertion depth."""
    return torch.maximum(depth, peak)


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
    """The per-step pose delta, expressed IN THE POCKET FRAME."""
    lateral = torch.clamp((aim - tip_rel[:, 0:2]) * lateral_gain, -max_step_m, max_step_m)

    height_err = standoff_m - tip_rel[:, 2]
    hold = torch.clamp(height_err * vertical_gain, -max_step_m, max_step_m)
    sink = torch.full_like(hold, -descend_rate_m)
    if stop_depth_m >= 0.0:
        seated = -tip_rel[:, 2] >= stop_depth_m
        sink = torch.where(seated, torch.zeros_like(sink), sink)
    dz = torch.where(phase == PHASE_DESCEND, sink, hold)

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
    """``command_in_pocket_frame`` with the ONE tilt column filled."""
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
    tilt_active = phase != PHASE_ABORT_FORCE
    cmd[:, 3 + tilt_axis] = torch.where(tilt_active, d_tilt, torch.zeros_like(d_tilt))
    return cmd


@torch.jit.script
def command_to_base_frame(
    command_pocket: torch.Tensor,
    pocket_quat: torch.Tensor,
    root_quat: torch.Tensor,
) -> torch.Tensor:
    """Take the ``(N, 6)`` delta from the pocket frame to the robot base frame."""
    lin_base = rotate_by_quat_inv(root_quat, rotate_by_quat(pocket_quat, command_pocket[:, 0:3]))
    rot_base = rotate_by_quat_inv(root_quat, rotate_by_quat(pocket_quat, command_pocket[:, 3:6]))
    return torch.cat((lin_base, rot_base), dim=-1)


@torch.jit.script
def joint_delta_to_action(
    joint_pos_des: torch.Tensor,
    joint_targets: torch.Tensor,
    action_scale: float,
) -> torch.Tensor:
    """Convert an absolute desired joint position into the env's action."""
    return torch.clamp((joint_pos_des - joint_targets) / action_scale, -1.0, 1.0)


@torch.jit.script
def osc_action_from_pocket_command(
    command_pocket: torch.Tensor,
    pocket_quat: torch.Tensor,
    pos_step_limit_m: float,
    rot_step_limit_rad: float,
) -> torch.Tensor:
    """The ``(N, 6)`` pocket-frame pose delta as the env's OSC action."""
    lin_env = rotate_by_quat(pocket_quat, command_pocket[:, 0:3])
    rot_env = rotate_by_quat(pocket_quat, command_pocket[:, 3:6])
    return torch.cat(
        (
            torch.clamp(lin_env / pos_step_limit_m, -1.0, 1.0),
            torch.clamp(rot_env / rot_step_limit_rad, -1.0, 1.0),
        ),
        dim=-1,
    )


AIM_AXES: dict = {"none": -1, "x": 0, "y": 1}


def aim_offsets(num_envs: int, axis: str, span_m: float, device=None) -> torch.Tensor:
    """Per-env lateral aim ``(N, 2)`` in the pocket frame."""
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
    """Planar distance from the AIM, in the pocket frame ``(N,)``."""
    return torch.linalg.norm(tip_rel[:, 0:2] - aim, dim=-1)


def track_settle(errors: list, tol_m: float, tail: int) -> dict:
    """Settling of ONE env's lateral-error trace."""
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
    """The shallowest depth from which ``value_key`` STAYS below ``tol``, in mm."""
    values = [r[value_key] for r in rows]
    if not values:
        return None
    settled = track_settle(values, tol, 1)
    if settled["settle_step"] < 0:
        return None
    return rows[settled["settle_step"]]["depth_lo_mm"]


def bin_force_by_depth(
    depths: list,
    forces: list,
    bin_width_m: float,
    counts: list,
    force_max: list,
    force_sum: list,
) -> None:
    """The force binner: ``bin_values_by_depth`` under its original name."""
    bin_values_by_depth(depths, forces, bin_width_m, counts, force_max, force_sum)


def bin_values_by_depth(
    depths: list,
    values: list,
    bin_width_m: float,
    counts: list,
    value_max: list,
    value_sum: list,
) -> None:
    """Accumulate one step's (depth, value) pairs into depth bins, IN PLACE."""
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
    """One row per ENV, sorted by its aim -- the RT-75 table's skeleton."""
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
    """The force profile: ``depth_profile`` under its original name and keys."""
    return depth_profile(bin_width_m, counts, force_max, force_sum, "force", "n")


def depth_profile(
    bin_width_m: float,
    counts: list,
    value_max: list,
    value_sum: list,
    value_key: str,
    unit: str,
) -> list:
    """The accumulators as a list of per-bin dicts."""
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
