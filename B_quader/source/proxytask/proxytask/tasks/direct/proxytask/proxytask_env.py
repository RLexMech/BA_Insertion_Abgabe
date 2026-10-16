# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import dataclasses
import json
import math
import pathlib
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.schemas import define_rigid_body_properties
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from . import proxytask_tasks_cfg
from .proxytask_env_cfg import ProxytaskEnvCfg, resolve_fixture_usd_path
from .proxytask_tasks_cfg import resolve_peg_robot_usd_path

CODE_MARKER = "square-peg-2026-08-16-d038"


def _axes_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Return the three body-frame axes of quaternion (w, x, y, z) as columns."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]),
            torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]),
            torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]),
        ],
        dim=1,
    )


def _axes_from_quat_batched(q: torch.Tensor) -> torch.Tensor:
    """Batched form of ``_axes_from_quat``: (N, 4) wxyz -> (N, 3, 3), axes as columns."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    col0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    col1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
    col2 = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return torch.stack([col0, col1, col2], dim=-1)


def _quat_mul_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz batches, with rotation matrix R(a) @ R(b)."""
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
def compute_rewards(
    tip_local: torch.Tensor,
    entrance_local: torch.Tensor,
    depth: torch.Tensor,
    gate: torch.Tensor,
    below_plate: torch.Tensor,
    alignment: torch.Tensor,
    cos4phi: torch.Tensor,
    remaining_steps: torch.Tensor,
    actions: torch.Tensor,
    max_depth: torch.Tensor,
    success_depth: float,
    w_approach: float,
    w_depth: float,
    w_success: float,
    w_action: float,
    w_misplaced: float,
    w_align: float,
    w_yaw: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reward, maximum insertion depth and success for each environment."""
    distance = torch.linalg.norm(tip_local - entrance_local, dim=-1)
    approach = -w_approach * distance
    depth_gated = torch.where(gate, depth, torch.zeros_like(depth))
    # Reward only new depth records to prevent rewards for repeated entry.
    new_max_depth = torch.maximum(max_depth, depth_gated)
    progress = w_depth * (new_max_depth - max_depth)
    success = (depth >= success_depth) & gate
    success_bonus = torch.where(success, torch.full_like(depth, w_success), torch.zeros_like(depth))
    action_penalty = -w_action * torch.sum(actions * actions, dim=-1)
    misplaced_depth = torch.where(gate, torch.zeros_like(depth), torch.clamp(depth, min=0.0))
    misplaced_penalty = -w_misplaced * misplaced_depth
    align_penalty = -w_align * (1.0 - alignment)
    yaw_penalty = -w_yaw * (1.0 - cos4phi) * 0.5
    per_step = (
        approach + progress + success_bonus + action_penalty
        + misplaced_penalty + align_penalty + yaw_penalty
    )
    # Charge the remaining cost so leaving the arena cannot avoid negative rewards.
    out_of_bounds_penalty = torch.where(
        below_plate,
        (approach + misplaced_penalty + align_penalty + yaw_penalty) * remaining_steps,
        torch.zeros_like(distance),
    )
    reward = per_step + out_of_bounds_penalty
    return reward, new_max_depth, success


class ProxytaskEnv(DirectRLEnv):
    """Square-peg insertion with joint-position deltas and 25 observation values."""

    cfg: ProxytaskEnvCfg

    def __init__(self, cfg: ProxytaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]

        try:
            self._peg_body_idx = self.robot.find_bodies(self.cfg.peg_body_name)[0][0]
        except (IndexError, KeyError, ValueError):
            self._peg_body_idx = None

        self._report_at_obs_calls = tuple(self.cfg.report_at_steps)
        self._obs_calls = 0

        self._entrance_base = torch.tensor(cfg.opening_entrance_pos, device=self.device)
        self._entrance_pos = self._entrance_base.unsqueeze(0).repeat(self.num_envs, 1)
        self._fixture_yaw = torch.zeros(self.num_envs, device=self.device)
        self._tip_offset_local = torch.tensor(cfg.peg_tip_offset, device=self.device)

        self._fixture_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._fixture_quat[:, 0] = 1.0
        self._tilt_active = (
            float(cfg.fixture_tilt_rad) != 0.0
            or float(cfg.fixture_tilt_noise_rad) != 0.0
            or float(cfg.fixture_yaw_rad) != 0.0
            or float(cfg.fixture_yaw_noise_rad) != 0.0
        )
        if self._tilt_active:
            deterministic = float(cfg.fixture_tilt_rad) != 0.0 or float(cfg.fixture_yaw_rad) != 0.0
            randomised = float(cfg.fixture_tilt_noise_rad) != 0.0 or float(cfg.fixture_yaw_noise_rad) != 0.0
            if deterministic and randomised:
                raise ValueError(
                    "a deterministic pocket angle (fixture_tilt_rad / fixture_yaw_rad) is "
                    "set together with a noise range (fixture_tilt_noise_rad / "
                    "fixture_yaw_noise_rad): one run cannot be a deterministic probe "
                    "(one fixed pose, all envs, the instrument the D-036 baseline was "
                    "measured with) and a randomised curriculum rung at the same time. "
                    "Set exactly one side (D-037/D-038)."
                )
            half_tilt = 0.5 * float(cfg.fixture_tilt_rad)
            half_yaw = 0.5 * float(cfg.fixture_yaw_rad)
            cyw, syw = math.cos(half_yaw), math.sin(half_yaw)
            ctw, sty = math.cos(half_tilt), math.sin(half_tilt)
            self._fixture_quat[:, 0] = cyw * ctw
            self._fixture_quat[:, 1] = -syw * sty
            self._fixture_quat[:, 2] = cyw * sty
            self._fixture_quat[:, 3] = syw * ctw
            self._fixture_yaw[:] = float(cfg.fixture_yaw_rad)
            self._tilt_rot = _axes_from_quat_batched(self._fixture_quat)
            self._tilt_rot_t = self._tilt_rot.transpose(1, 2).contiguous()
        else:
            self._tilt_rot = None
            self._tilt_rot_t = None

        self._reset_noise_scale = torch.full(
            (len(self.robot.joint_names),), float(cfg.reset_joint_noise), device=self.device
        )
        try:
            yaw_idx = self.robot.joint_names.index(cfg.yaw_joint_name)
        except ValueError as exc:
            raise ValueError(
                f"yaw joint '{cfg.yaw_joint_name}' not among {self.robot.joint_names}; "
                "the square task needs its yaw randomization on the tool-axis joint"
            ) from exc
        self._reset_noise_scale[yaw_idx] = float(cfg.reset_yaw_noise)

        self._step_dt = cfg.sim.dt * cfg.decimation

        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._prev_joint_pos = self.robot.data.default_joint_pos.clone()
        self._max_depth = torch.zeros(self.num_envs, device=self.device)
        self._max_target_lag = torch.zeros(self.num_envs, device=self.device)

        self._last_tip_local = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_depth = torch.zeros(self.num_envs, device=self.device)
        self._last_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_below_plate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_alignment = torch.ones(self.num_envs, device=self.device)
        self._last_cos4phi = torch.ones(self.num_envs, device=self.device)
        self._last_remaining = torch.zeros(self.num_envs, device=self.device)
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._ep_count = 0
        self._ep_success_count = 0
        self._ep_depth_sum = 0.0
        self._episodes_to_threshold: int | None = None
        window = max(2000, int(self.num_envs * self.cfg.metrics_window_iterations * 16 / 240))
        self._recent_successes: collections.deque = collections.deque(maxlen=window)
        self._recent_depths: collections.deque = collections.deque(maxlen=window)
        self._recent_success_steps: collections.deque = collections.deque(maxlen=window)
        self._recent_tilt_deg: collections.deque = collections.deque(maxlen=window)
        self._tilt_bin_edges_deg = (3.0, 6.0, 9.0, 12.0, 15.0)
        self._recent_target_lag: collections.deque = collections.deque(maxlen=window)
        self._metrics_path = pathlib.Path("logs") / "demo_metrics.json"
        self._metrics_dump_every = max(50, window // 8)

    def _setup_scene(self):
        fixture_spawn = self.cfg.fixture_cfg.spawn
        fixture_spawn.usd_path = resolve_fixture_usd_path()
        self._fixture_usd_path = fixture_spawn.usd_path
        half_tilt = 0.5 * float(self.cfg.fixture_tilt_rad)
        half_yaw = 0.5 * float(self.cfg.fixture_yaw_rad)
        cyw, syw = math.cos(half_yaw), math.sin(half_yaw)
        ctw, sty = math.cos(half_tilt), math.sin(half_tilt)
        fixture_spawn.func(
            "/World/envs/env_0/Fixture",
            fixture_spawn,
            translation=self.cfg.fixture_pos,
            orientation=(cyw * ctw, -syw * sty, cyw * sty, syw * ctw),
        )
        define_rigid_body_properties(
            "/World/envs/env_0/Fixture",
            sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        )
        self._fixture = RigidObject(self.cfg.fixture_cfg.replace(spawn=None))

        self.cfg.robot_cfg.spawn.usd_path = resolve_peg_robot_usd_path()
        self._robot_usd_path = self.cfg.robot_cfg.spawn.usd_path

        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["fixture"] = self._fixture
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        clamped = actions.clamp(-1.0, 1.0)
        self._joint_targets = self._joint_targets + self.cfg.action_scale * clamped
        limits = self.robot.data.soft_joint_pos_limits
        self._joint_targets = torch.clamp(self._joint_targets, limits[..., 0], limits[..., 1])
        lag = (self._joint_targets - self.robot.data.joint_pos).abs().amax(dim=-1)
        self._max_target_lag = torch.maximum(self._max_target_lag, lag)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets)

    def _peg_geometry(self):
        """Return tip position, depth, pocket gate, out-of-bounds flag, alignment, phi, cos(4 phi), sin(4 phi) and relative tip position."""
        origins = self.scene.env_origins
        peg_pos_w = self.robot.data.body_pos_w[:, self._peg_body_idx]
        peg_quat_w = self.robot.data.body_quat_w[:, self._peg_body_idx]
        peg_axes = _axes_from_quat_batched(peg_quat_w)
        tip_w = peg_pos_w + torch.matmul(peg_axes, self._tip_offset_local)
        tip_local = tip_w - origins

        # Measure depth, alignment and containment in the pocket coordinate frame.
        if self._tilt_rot is None:
            tip_rel = tip_local - self._entrance_pos
            depth = self.cfg.plate_top_z - tip_local[:, 2]
            axes = peg_axes
            below_plate = tip_local[:, 2] < self.cfg.plate_bottom_z
        else:
            tip_rel = torch.einsum("ni,nij->nj", tip_local - self._entrance_pos, self._tilt_rot)
            depth = -tip_rel[:, 2]
            axes = torch.bmm(self._tilt_rot_t, peg_axes)
            below_plate = tip_rel[:, 2] < (self.cfg.plate_bottom_z - self.cfg.plate_top_z)

        x_axis = axes[:, :, 0]
        y_axis = axes[:, :, 1]

        phi_raw = torch.atan2(x_axis[:, 1], x_axis[:, 0])
        phi = phi_raw - torch.round(phi_raw / (math.pi / 2.0)) * (math.pi / 2.0)
        # The square has four equivalent orientations, 90 degrees apart.
        cos4phi = torch.cos(4.0 * phi_raw)
        sin4phi = torch.sin(4.0 * phi_raw)

        half = self.cfg.held_asset.side / 2.0
        ex = x_axis[:, :2] * half
        ey = y_axis[:, :2] * half
        tip_xy = tip_rel[:, :2]
        half_pocket = self.cfg.fixed_asset.side / 2.0
        corners = torch.stack(
            (tip_xy + ex + ey, tip_xy + ex - ey, tip_xy - ex + ey, tip_xy - ex - ey),
            dim=1,
        )
        # All four tip corners must fit within both pocket axes.
        corners_inside = corners.abs().amax(dim=(1, 2)) < half_pocket

        alignment = -axes[:, 2, 2]
        # The 1 mm depth margin allows solver penetration and floating-point error.
        gate = (
            corners_inside
            & (depth <= self.cfg.fixed_asset.depth + 0.001)
            & (alignment >= self.cfg.gate_min_alignment)
        )
        return tip_local, depth, gate, below_plate, alignment, phi, cos4phi, sin4phi, tip_rel

    def _get_observations(self) -> dict:
        self._obs_calls += 1

        joint_pos = self.robot.data.joint_pos
        # Finite differences avoid the solver velocity offset observed at rest.
        joint_vel_fd = (joint_pos - self._prev_joint_pos) / self._step_dt
        self._prev_joint_pos = joint_pos.clone()

        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        if self._peg_body_idx is not None:
            geo = self._peg_geometry()
            tip_rel_entrance = geo[8]
            cos4phi = geo[6].unsqueeze(-1)
            sin4phi = geo[7].unsqueeze(-1)
        else:
            tip_rel_entrance = torch.zeros(self.num_envs, 3, device=self.device)
            cos4phi = torch.ones(self.num_envs, 1, device=self.device)
            sin4phi = torch.zeros(self.num_envs, 1, device=self.device)

        # Layout: joints 6+6, relative tip 3, tool quaternion 4, yaw 2, pocket quaternion 4.
        obs = torch.cat(
            (joint_pos, joint_vel_fd, tip_rel_entrance, ee_quat_w, cos4phi, sin4phi, self._fixture_quat), dim=-1
        )

        if self._obs_calls in self._report_at_obs_calls:
            self._print_startup_report(joint_vel_fd)

        return {"policy": obs}

    # DirectRLEnv calls this before _get_rewards(); cache geometry for that call.
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self._peg_body_idx is not None:
            tip_local, depth, gate, below_plate, alignment, _phi, cos4phi, _sin4phi, _tip_rel = self._peg_geometry()
            success = (depth >= self.cfg.fixed_asset.success_depth) & gate
            self._last_tip_local = tip_local
            self._last_depth = depth
            self._last_gate = gate
            self._last_below_plate = below_plate
            self._last_alignment = alignment
            self._last_cos4phi = cos4phi
            self._last_success = success
            self._last_remaining = (self.max_episode_length - 1 - self.episode_length_buf).clamp(min=0).float()
            terminated = success | below_plate
        else:
            terminated = torch.zeros_like(time_out)
            self._last_depth = torch.zeros(self.num_envs, device=self.device)
            self._last_success = terminated
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        if self._peg_body_idx is None:
            return torch.zeros(self.num_envs, device=self.device)
        reward, new_max_depth, _ = compute_rewards(
            self._last_tip_local,
            self._entrance_pos,
            self._last_depth,
            self._last_gate,
            self._last_below_plate,
            self._last_alignment,
            self._last_cos4phi,
            self._last_remaining,
            self.actions,
            self._max_depth,
            self.cfg.fixed_asset.success_depth,
            self.cfg.reward_w_approach,
            self.cfg.reward_w_depth,
            self.cfg.reward_w_success,
            self.cfg.reward_w_action,
            self.cfg.reward_w_misplaced,
            self.cfg.reward_w_align,
            self.cfg.reward_w_yaw,
        )
        self._max_depth = new_max_depth
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        ready = hasattr(self, "_joint_targets")

        if ready:
            self._log_finished_episodes(env_ids)

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        if ready:
            joint_pos = joint_pos + (torch.rand_like(joint_pos) * 2.0 - 1.0) * self._reset_noise_scale
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        try:
            self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        except (AttributeError, TypeError) as exc:
            if not getattr(self, "_target_warning_printed", False):
                self._target_warning_printed = True
                print(f"[proxytask] set_joint_position_target unavailable ({exc}); relying on USD drive targets")

        if ready:
            self._joint_targets[env_ids] = joint_pos
            self._prev_joint_pos[env_ids] = joint_pos
            self._max_depth[env_ids] = 0.0
            self._max_target_lag[env_ids] = 0.0

        if ready and (
            self.cfg.fixture_pos_noise_xy > 0.0
            or self.cfg.fixture_yaw_noise_rad > 0.0
            or self.cfg.fixture_tilt_noise_rad > 0.0
        ):
            idx = torch.as_tensor(list(env_ids), device=self.device) if not torch.is_tensor(env_ids) else env_ids
            n = int(idx.numel())
            entrance = self._entrance_base.unsqueeze(0).repeat(n, 1)
            entrance[:, :2] += (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * self.cfg.fixture_pos_noise_xy
            yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.fixture_yaw_noise_rad
            self._entrance_pos[idx] = entrance
            self._fixture_yaw[idx] = yaw
            quat_yaw = torch.zeros(n, 4, device=self.device)
            quat_yaw[:, 0] = torch.cos(yaw * 0.5)
            quat_yaw[:, 3] = torch.sin(yaw * 0.5)

            if self.cfg.fixture_tilt_noise_rad > 0.0:
                mag = torch.rand(n, device=self.device) * self.cfg.fixture_tilt_noise_rad
                azimuth = torch.rand(n, device=self.device) * (2.0 * math.pi)
                half = 0.5 * mag
                sin_half = torch.sin(half)
                quat_tilt = torch.stack(
                    (
                        torch.cos(half),
                        torch.cos(azimuth) * sin_half,
                        torch.sin(azimuth) * sin_half,
                        torch.zeros_like(half),
                    ),
                    dim=-1,
                )
                quat = _quat_mul_batched(quat_yaw, quat_tilt)
            else:
                quat = quat_yaw
            # q and -q describe the same rotation; use a consistent sign for observations.
            quat = torch.where(quat[:, :1] < 0.0, -quat, quat)
            self._fixture_quat[idx] = quat
            if self._tilt_rot is not None:
                rot = _axes_from_quat_batched(quat)
                self._tilt_rot[idx] = rot
                self._tilt_rot_t[idx] = rot.transpose(1, 2)

            pose_w = torch.cat((entrance + self.scene.env_origins[idx], quat), dim=-1)
            self._fixture.write_root_pose_to_sim(pose_w, env_ids=idx)
            self._fixture.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=idx)

    def _log_finished_episodes(self, env_ids: Sequence[int]) -> None:
        """Record finished episodes and publish aggregate metrics to RSL-RL and disk."""
        if len(env_ids) == 0 or self._peg_body_idx is None or self._obs_calls == 0:
            return
        idx = torch.as_tensor(list(env_ids), device=self.device) if not torch.is_tensor(env_ids) else env_ids
        successes = self._last_success[idx]
        depths = self._max_depth[idx]
        n = int(idx.numel())

        self._ep_count += n
        self._ep_success_count += int(successes.sum().item())
        self._ep_depth_sum += float(depths.sum().item())
        self._recent_successes.extend(successes.tolist())
        self._recent_depths.extend(depths.tolist())
        self._recent_success_steps.extend(self.episode_length_buf[idx][successes].tolist())
        fq_x, fq_y = self._fixture_quat[idx, 1], self._fixture_quat[idx, 2]
        r22 = 1.0 - 2.0 * (fq_x * fq_x + fq_y * fq_y)
        tilt_deg = torch.rad2deg(torch.acos(r22.clamp(-1.0, 1.0)))
        self._recent_tilt_deg.extend(tilt_deg.tolist())
        self._recent_target_lag.extend(self._max_target_lag[idx].tolist())

        cumulative_rate = self._ep_success_count / max(self._ep_count, 1)
        recent_rate = (
            sum(self._recent_successes) / len(self._recent_successes) if self._recent_successes else 0.0
        )
        recent_depth = (
            sum(self._recent_depths) / len(self._recent_depths) if self._recent_depths else 0.0
        )
        if (
            self._episodes_to_threshold is None
            and len(self._recent_successes) >= self._recent_successes.maxlen // 4
            and recent_rate >= self.cfg.metrics_success_threshold
        ):
            self._episodes_to_threshold = self._ep_count
        self.extras["log"] = {
            "success_rate": recent_rate,
            "success_rate_cumulative": cumulative_rate,
            "mean_max_depth_mm": recent_depth * 1000.0,
        }

        if self._ep_count % self._metrics_dump_every < n:
            self._write_metrics(cumulative_rate, recent_rate, recent_depth)

    def _success_by_tilt_bin(self) -> list[dict]:
        """Trailing-window success rate resolved by pocket tilt magnitude."""
        edges = self._tilt_bin_edges_deg
        counts = [0] * len(edges)
        hits = [0] * len(edges)
        for success, tilt in zip(self._recent_successes, self._recent_tilt_deg):
            b = len(edges) - 1
            for i, upper in enumerate(edges):
                if tilt < upper:
                    b = i
                    break
            counts[b] += 1
            hits[b] += int(bool(success))
        lower = 0.0
        out = []
        for i, upper in enumerate(edges):
            out.append(
                {
                    "tilt_deg_from": lower,
                    "tilt_deg_to": upper,
                    "episodes": counts[i],
                    "success_rate": (hits[i] / counts[i]) if counts[i] else None,
                }
            )
            lower = upper
        return out

    def _target_lag_stats(self) -> dict:
        """Joint-target lag over the trailing window, split by episode outcome."""
        if not self._recent_target_lag:
            return {}
        scale = float(self.cfg.action_scale) or 1.0
        ok = sorted(lag for lag, s in zip(self._recent_target_lag, self._recent_successes) if s)
        bad = sorted(lag for lag, s in zip(self._recent_target_lag, self._recent_successes) if not s)

        def _stats(values: list[float], prefix: str) -> dict:
            if not values:
                return {f"{prefix}_mean": None, f"{prefix}_p95": None, f"{prefix}_max": None,
                        f"{prefix}_mean_steps": None, f"{prefix}_p95_steps": None,
                        f"{prefix}_max_steps": None, f"{prefix}_episodes": 0}
            mean = sum(values) / len(values)
            p95 = values[min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)]
            top = values[-1]
            return {
                f"{prefix}_mean": mean,
                f"{prefix}_p95": p95,
                f"{prefix}_max": top,
                f"{prefix}_mean_steps": mean / scale,
                f"{prefix}_p95_steps": p95 / scale,
                f"{prefix}_max_steps": top / scale,
                f"{prefix}_episodes": len(values),
            }

        out = {"episodes": len(self._recent_target_lag), "max": max(self._recent_target_lag)}
        out["max_steps"] = out["max"] / scale
        out.update(_stats(ok, "success"))
        out.update(_stats(bad, "failure"))
        return out

    def _write_metrics(self, cumulative_rate: float, recent_rate: float, recent_depth: float) -> None:
        """Dump the run's metrics beside its own logs."""
        log_dir = getattr(self.cfg, "log_dir", None)
        path = pathlib.Path(log_dir) / "demo_metrics.json" if log_dir else self._metrics_path
        payload = {
            "episodes": self._ep_count,
            "success_rate_recent": recent_rate,
            "episodes_to_threshold": self._episodes_to_threshold,
            "threshold": self.cfg.metrics_success_threshold,
            "recent_window_episodes": len(self._recent_successes),
            "success_rate_cumulative": cumulative_rate,
            "mean_max_depth_mm": recent_depth * 1000.0,
            "peg_side_m": self.cfg.held_asset.side,
            "pocket_side_m": self.cfg.fixed_asset.side,
            "side_clearance_mm": proxytask_tasks_cfg.SIDE_CLEARANCE * 1000.0,
            "yaw_window_deg": math.degrees(proxytask_tasks_cfg.YAW_WINDOW_RAD),
            "reset_joint_noise_rad": self.cfg.reset_joint_noise,
            "reset_yaw_noise_rad": self.cfg.reset_yaw_noise,
            "fixture_pos_noise_xy_m": self.cfg.fixture_pos_noise_xy,
            "fixture_yaw_noise_rad": self.cfg.fixture_yaw_noise_rad,
            "fixture_tilt_rad": self.cfg.fixture_tilt_rad,
            "fixture_yaw_rad": self.cfg.fixture_yaw_rad,
            "fixture_tilt_noise_rad": self.cfg.fixture_tilt_noise_rad,
            "success_by_tilt_bin": self._success_by_tilt_bin(),
            "action_scale_rad": self.cfg.action_scale,
            "joint_target_lag_rad": self._target_lag_stats(),
            "mean_success_episode_steps": (
                sum(self._recent_success_steps) / len(self._recent_success_steps)
                if self._recent_success_steps
                else None
            ),
            "episode_length_s": self.cfg.episode_length_s,
            "num_envs": self.num_envs,
            "code_marker": CODE_MARKER,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:
            if not getattr(self, "_metrics_warning_printed", False):
                self._metrics_warning_printed = True
                print(f"[proxytask] could not write metrics to {path}: {exc}")

    def _asset_geometry_line(self) -> str:
        """Compare the authored asset's *measured* geometry against the cfg constants."""
        cfg = self.cfg
        try:
            sidecar = pathlib.Path(self._robot_usd_path).with_suffix(".author.json")
        except (AttributeError, TypeError) as exc:
            return f"UNCHECKED (no robot usd path: {exc})"
        if not sidecar.is_file():
            return (f"UNCHECKED -- no {sidecar.name} beside the peg USD; re-run "
                    "scripts/author_peg_ur10e.py to produce one")
        try:
            data = json.loads(sidecar.read_text())
            measured = data.get("after", {}).get("measured") or data.get("measured", {})
            verdict = data.get("verdict")
            size = measured.get("local_bbox_size_m")
            mass = measured.get("mass_kg")
        except (OSError, ValueError, AttributeError) as exc:
            return f"UNCHECKED ({sidecar.name} unreadable: {type(exc).__name__}: {exc})"
        if verdict is not None and verdict != "PASS":
            return (f"*** THE LAST AUTHORING RUN REPORTED {verdict} *** "
                    f"see {sidecar.name}; the asset was written despite failing checks")
        if not size or mass is None:
            return f"UNCHECKED ({sidecar.name} carries no measured geometry)"

        problems = []
        for axis in (0, 1):
            if abs(size[axis] - cfg.held_asset.side) > 1e-4:
                problems.append(
                    f"side (bbox axis {axis}) {size[axis]:.5f} m authored vs "
                    f"{cfg.held_asset.side:.5f} m configured"
                )
        if abs(size[2] - cfg.peg_length) > 1e-4:
            problems.append(f"length {size[2]:.5f} m authored vs {cfg.peg_length:.5f} m configured")
        if abs(mass - cfg.held_asset.mass) > 1e-5:
            problems.append(f"mass {mass:.5f} kg authored vs {cfg.held_asset.mass:.5f} kg configured")
        if problems:
            return ("*** ASSET DOES NOT MATCH THE CONFIGURATION *** " + "; ".join(problems)
                    + ". The USD is stale: re-run scripts/author_peg_ur10e.py, then "
                      "scripts/verify_peg_passability.py. Every peg number in this report is void.")
        return (f"PASS -- authored geometry matches cfg (side {size[0]:.5f} m, "
                f"length {size[2]:.5f} m, mass {mass:.5f} kg), from {sidecar.name}")

    def _fixture_geometry_line(self) -> str:
        """Compare the authored fixture geometry with the configured pocket dimensions."""
        cfg = self.cfg
        try:
            sidecar = pathlib.Path(self._fixture_usd_path).with_suffix(".author.json")
        except (AttributeError, TypeError) as exc:
            return f"UNCHECKED (no fixture usd path: {exc})"
        if not sidecar.is_file():
            return (f"UNCHECKED -- no {sidecar.name} beside the fixture USD. Generated tables "
                    "carry one; an imported or hand-renamed table does not, and its pocket size "
                    "is then unverifiable. Re-generate with scripts/author_tisch_square.py "
                    f"--pocket-side-mm {cfg.fixed_asset.side * 1000.0:g}")
        try:
            data = json.loads(sidecar.read_text())
            verdict = data.get("verdict")
            authored_side = data.get("pocket_side_m")
            authored_depth = data.get("pocket_depth_m")
            marker = data.get("marker")
        except (OSError, ValueError, AttributeError) as exc:
            return f"UNCHECKED ({sidecar.name} unreadable: {type(exc).__name__}: {exc})"
        if verdict is not None and verdict != "PASS":
            return (f"*** THE LAST TABLE GENERATION REPORTED {verdict} *** "
                    f"see {sidecar.name}; the asset was written despite failing its own checks")
        if authored_side is None:
            return f"UNCHECKED ({sidecar.name} carries no measured pocket geometry)"

        problems = []
        if abs(authored_side - cfg.fixed_asset.side) > 1e-5:
            problems.append(
                f"pocket side {authored_side*1000:.3f} mm authored vs "
                f"{cfg.fixed_asset.side*1000:.3f} mm configured"
            )
        if authored_depth is not None and abs(authored_depth - cfg.fixed_asset.depth) > 1e-5:
            problems.append(
                f"pocket depth {authored_depth*1000:.3f} mm authored vs "
                f"{cfg.fixed_asset.depth*1000:.3f} mm configured"
            )
        if problems:
            return ("*** FIXTURE DOES NOT MATCH THE CONFIGURATION *** " + "; ".join(problems)
                    + ". The table is for a different curriculum rung: generate the right one "
                      "with scripts/author_tisch_square.py --pocket-side-mm "
                      f"{cfg.fixed_asset.side * 1000.0:g}, or unset PROXYTASK_TISCH_USD. "
                      "Every clearance, yaw-window and success number in this report is void.")
        return (f"PASS -- generated pocket matches cfg (side {authored_side*1000:.3f} mm, "
                f"depth {(authored_depth or 0.0)*1000:.3f} mm), from {sidecar.name}"
                + (f", {marker}" if marker else ""))

    def _print_startup_report(self, joint_vel_fd: torch.Tensor) -> None:
        cfg = self.cfg
        origins = self.scene.env_origins
        commanded = self.robot.data.default_joint_pos
        down = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        sim_time = self._obs_calls * cfg.sim.dt * cfg.decimation
        print("=" * 72)
        print(f"PROXYTASK STARTUP REPORT (square-peg pivot, branch square-peg-insertion, UNVERIFIED)")
        print(f"step {self._obs_calls} of {self._report_at_obs_calls}, t = {sim_time:.3f} s after reset")
        print(f"code marker: {CODE_MARKER}")
        print(f"fixture usd: {self._fixture_usd_path}")
        print(f"robot usd:   {self._robot_usd_path}")
        print(f"fixture spawn translation: {cfg.fixture_pos}  (asset origin = opening plane at pocket centre)")
        print(f"fixture tilt (D-036 probe): {math.degrees(cfg.fixture_tilt_rad):+.2f} deg about env y through the pocket centre "
              f"({'ACTIVE -- one fixed angle in every env' if cfg.fixture_tilt_rad != 0.0 else 'inactive'})")
        print(f"fixture yaw (D-038 probe): {math.degrees(cfg.fixture_yaw_rad):+.2f} deg about env z through the pocket centre "
              f"({'ACTIVE -- one fixed angle in every env' if cfg.fixture_yaw_rad != 0.0 else 'inactive'})")
        print(f"fixture tilt noise (D-037): magnitude 0..{math.degrees(cfg.fixture_tilt_noise_rad):.2f} deg, "
              "direction uniform over 360 deg, resampled per episode "
              f"({'ACTIVE -- per-env pocket orientation' if cfg.fixture_tilt_noise_rad > 0.0 else 'inactive'})")
        print(f"pocket-frame measurement: {'ON -- gate/depth/alignment/phi and obs 12:15 in the pocket frame' if self._tilt_active else 'OFF -- pocket upright, env-local measurement'}")
        print(f"fixture pose noise (D-034): xy +-{cfg.fixture_pos_noise_xy:.3f} m, "
              f"yaw +-{cfg.fixture_yaw_noise_rad:.4f} rad "
              f"({'ACTIVE -- kinematic fixture teleported per reset' if (cfg.fixture_pos_noise_xy > 0.0 or cfg.fixture_yaw_noise_rad > 0.0) else 'inactive -- fixture stays at the spawn pose'})")
        print(f"robot base sits on the plate top: base z {cfg.robot_base_pos[2]:.4f} "
              f"= plate top {cfg.plate_top_z:.4f}; "
              f"{proxytask_tasks_cfg.BASE_TO_PLATE_EDGE*1000:.0f} mm from base centre to plate edge, "
              f"{proxytask_tasks_cfg.OPENING_TO_BASE_DISTANCE*1000:.0f} mm to the pocket centre")
        print(f"joint_names: {self.robot.joint_names}")
        print(f"body_names:  {self.robot.body_names}")
        print(f"ee_body:     {cfg.ee_body_name} (index {self._ee_body_idx})")
        spawn_cfg = cfg.robot_cfg.spawn
        declared = (
            "activate_contact_sensors" in {f.name for f in dataclasses.fields(spawn_cfg)}
            if dataclasses.is_dataclass(spawn_cfg)
            else None
        )
        print(f"activate_contact_sensors: {getattr(spawn_cfg, 'activate_contact_sensors', 'ABSENT')} "
              f"({'declared cfg field' if declared else 'NOT A DECLARED FIELD' if declared is False else 'undetermined'})")
        if self._peg_body_idx is None:
            print(f"peg_body:    *** MISSING *** '{cfg.peg_body_name}' is not among the bodies above.")
            print( "             The welded link did not survive authoring; every peg number below is void.")
        else:
            print(f"peg_body:    {cfg.peg_body_name} (index {self._peg_body_idx}), "
                  f"length {cfg.peg_length:.4f} m, side {cfg.held_asset.side:.4f} m, "
                  f"mass {cfg.held_asset.mass:.5f} kg")
            print(f"pocket:      side {cfg.fixed_asset.side:.4f} m, depth {cfg.fixed_asset.depth:.4f} m (blind), "
                  f"clearance {proxytask_tasks_cfg.SIDE_CLEARANCE*1000:.2f} mm per axis at phi=0, "
                  f"yaw window +-{math.degrees(proxytask_tasks_cfg.YAW_WINDOW_RAD):.2f} deg, "
                  f"success at {cfg.fixed_asset.success_depth*1000:.0f} mm")
        print(f"asset check (peg):     {self._asset_geometry_line()}")
        print(f"asset check (fixture): {self._fixture_geometry_line()}")
        resting_speed = torch.linalg.norm(joint_vel_fd[: min(2, self.num_envs)], dim=-1)
        print(f"resting FD joint speed (env 0..1): {resting_speed.tolist()} rad/s  (expect ~0.000 after reset)")
        print("observation layout (25-dim): 0:6 joint_pos | 6:12 joint_vel (finite-diff) | "
              f"12:15 peg tip - pocket entrance ({'POCKET frame, D-036' if self._tilt_active else 'env-local'}) | "
              "15:19 EE quat (wxyz) | 19:21 (cos 4 phi, sin 4 phi) | "
              "21:25 POCKET quat (wxyz, w >= 0, D-037)")
        for e in range(min(2, self.num_envs)):
            root_local = self.robot.data.root_pos_w[e] - origins[e]
            ee_pos_w = self.robot.data.body_pos_w[e, self._ee_body_idx]
            ee_quat_w = self.robot.data.body_quat_w[e, self._ee_body_idx]
            ee_local = ee_pos_w - origins[e]
            entrance_w = self._entrance_pos[e] + origins[e]
            fixture_local = self._fixture.data.root_pos_w[e] - origins[e]
            pose_err = torch.linalg.norm(fixture_local - self._entrance_pos[e])
            print(f"-- env {e} pose check --")
            print(f"  fixture prim (env-local): {fixture_local.tolist()}")
            print(f"  entrance buffer:          {self._entrance_pos[e].tolist()}, yaw {self._fixture_yaw[e].item():+.4f} rad")
            print(f"  |prim - buffer| = {pose_err.item():.2e} m  (expect ~0; MISMATCH = pose write broken)")
            fixture_quat = self._fixture.data.root_quat_w[e]
            qw, qx, qy, qz = (float(v) for v in fixture_quat.tolist())
            if qw < 0.0:
                qw, qx, qy, qz = -qw, -qx, -qy, -qz
            cw, cx, cy, cz = (float(v) for v in self._fixture_quat[e].tolist())
            quat_err = max(abs(qw - cw), abs(qx - cx), abs(qy - cy), abs(qz - cz))
            measured_mag = 2.0 * math.acos(max(-1.0, min(1.0, qw)))
            commanded_mag = 2.0 * math.acos(max(-1.0, min(1.0, cw)))
            axis_xy = math.hypot(qx, qy)
            azimuth = math.degrees(math.atan2(qy, qx)) if axis_xy > 1e-9 else 0.0
            print(f"  fixture prim quat (wxyz): {[round(v, 6) for v in (qw, qx, qy, qz)]}")
            print(f"  commanded quat (buffer):  {[round(v, 6) for v in (cw, cx, cy, cz)]}")
            print(f"  measured tilt magnitude:  {math.degrees(measured_mag):.3f} deg vs commanded "
                  f"{math.degrees(commanded_mag):.3f} deg, tilt axis azimuth {azimuth:+.1f} deg "
                  f"(0 = +x, 90 = +y), max|quat err| {quat_err:.2e}  "
                  f"({'OK' if quat_err <= 5e-5 else '*** TILT MISMATCH -- the commanded orientation did not land on the prim; every pocket-frame number below and observation channels 21:25 are VOID ***'})")
            if float(cfg.fixture_yaw_rad) != 0.0:
                r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
                r10 = 2.0 * (qx * qy + qw * qz)
                r20 = 2.0 * (qx * qz - qw * qy)
                signed_z = math.atan2(r10, r00)
                yaw_err = abs(signed_z - float(cfg.fixture_yaw_rad))
                print(f"  measured yaw about z:     {math.degrees(signed_z):+.3f} deg vs commanded "
                      f"{math.degrees(cfg.fixture_yaw_rad):+.3f} deg, |err| {math.degrees(yaw_err):.4f} deg  "
                      f"({'OK' if yaw_err <= 1e-4 else '*** YAW MISMATCH -- the spawn orientation did not land; every pocket-frame number below is VOID ***'})")
                tilt_zy = -math.asin(max(-1.0, min(1.0, r20)))
                print(f"  measured tilt about y (ZY): {math.degrees(tilt_zy):+.3f} deg vs commanded "
                      f"{math.degrees(cfg.fixture_tilt_rad):+.3f} deg")
            if float(cfg.fixture_tilt_rad) != 0.0 and float(cfg.fixture_yaw_rad) == 0.0:
                signed_y = 2.0 * math.atan2(qy, qw)
                tilt_err = abs(signed_y - float(cfg.fixture_tilt_rad))
                off_axis = math.hypot(qx, qz)
                print(f"  measured tilt about y:    {math.degrees(signed_y):+.3f} deg vs commanded "
                      f"{math.degrees(cfg.fixture_tilt_rad):+.3f} deg, |err| {math.degrees(tilt_err):.4f} deg  "
                      f"({'OK' if tilt_err <= 1e-4 else '*** TILT MISMATCH -- the spawn orientation did not land; every pocket-frame number below is VOID ***'})")
                print(f"  off-axis quat parts:      |(qx, qz)| = {off_axis:.2e}  "
                      f"(expect ~0; nonzero = the rotation is not about env y)")
            delta = ee_pos_w - entrance_w
            xy_offset = torch.linalg.norm(delta[:2])
            z_standoff = delta[2]
            axes = _axes_from_quat(ee_quat_w)
            print(f"-- env {e} --")
            print(f"  env_origin:            {origins[e].tolist()}")
            print(f"  joint_pos commanded:   {commanded[e].tolist()}")
            print(f"  joint_pos actual:      {self.robot.data.joint_pos[e].tolist()}")
            print(f"  joint_vel (FD):        {joint_vel_fd[e].tolist()}")
            print(f"  root env-local pos:    {root_local.tolist()}  (expect {cfg.robot_base_pos})")
            print(f"  EE world pos:          {ee_pos_w.tolist()}")
            print(f"  EE env-local pos:      {ee_local.tolist()}")
            print(f"  EE world quat (wxyz):  {ee_quat_w.tolist()}")
            print(f"  pocket entrance world: {entrance_w.tolist()}")
            print(f"  |EE - entrance|:       {torch.linalg.norm(delta).item():.6f} m")
            print(f"  XY offset to entrance: {xy_offset.item():.6f} m  (expect <= 0.005 before reset noise)")
            print(f"  Z standoff:            {z_standoff.item():.6f} m  (expect {cfg.home_standoff_z} +- 0.005)")
            for name, col in zip("xyz", range(3)):
                dot = torch.dot(axes[:, col], down).item()
                note = "  <-- expect >= 0.999 here" if name == "z" else ""
                print(f"  {name}-axis . (0,0,-1):    {dot:+.6f}{note}")
            if self._peg_body_idx is not None:
                peg_pos_w = self.robot.data.body_pos_w[e, self._peg_body_idx]
                peg_axes = _axes_from_quat(self.robot.data.body_quat_w[e, self._peg_body_idx])
                tip_w = peg_pos_w + peg_axes @ torch.tensor(cfg.peg_tip_offset, device=self.device)
                tip_local = tip_w - origins[e]
                if self._tilt_rot is None:
                    tip_rel_e = tip_local - self._entrance_pos[e]
                    axes_p = peg_axes
                    depth = cfg.plate_top_z - tip_local[2]
                else:
                    tip_rel_e = self._tilt_rot_t[e] @ (tip_local - self._entrance_pos[e])
                    axes_p = self._tilt_rot_t[e] @ peg_axes
                    depth = -tip_rel_e[2]
                along_tool = torch.dot(tip_w - ee_pos_w, axes[:, 2]).item()
                tilt_c = math.cos(commanded_mag)
                tilt_s = abs(math.sin(commanded_mag))
                home_tip_height = cfg.home_standoff_z - cfg.peg_length
                phi_raw = torch.atan2(axes_p[1, 0], axes_p[0, 0]).item()
                phi = phi_raw - round(phi_raw / (math.pi / 2.0)) * (math.pi / 2.0)
                half = cfg.held_asset.side / 2.0
                half_pocket = cfg.fixed_asset.side / 2.0
                ex = axes_p[:2, 0] * half
                ey = axes_p[:2, 1] * half
                tip_xy_vec = tip_rel_e[:2]
                corners = [tip_xy_vec + sx * ex + sy * ey for sx in (1, -1) for sy in (1, -1)]
                cheb = max(c.abs().max().item() for c in corners)
                tip_cheb = tip_xy_vec.abs().max().item()
                alignment_e = -axes_p[2, 2].item()
                gate_e = (
                    cheb < half_pocket
                    and depth.item() <= cfg.fixed_asset.depth + 0.001
                    and alignment_e >= cfg.gate_min_alignment
                )
                print(f"  peg origin env-local:  {(peg_pos_w - origins[e]).tolist()}")
                print(f"  peg TIP env-local:     {tip_local.tolist()}")
                print(f"  peg TIP rel. entrance: {[round(v, 6) for v in tip_rel_e.tolist()]}  "
                      f"({'POCKET frame (D-036)' if self._tilt_rot is not None else 'env-local; pocket upright'}"
                      f" -- this is observation channels 12:15)")
                print(f"  flange -> peg origin:  {torch.linalg.norm(peg_pos_w - ee_pos_w).item():.6f} m"
                      f"  (expect 0.000000)")
                print(f"  tip along tool axis:   {along_tool:+.6f} m  (expect {cfg.peg_length:+.4f})")
                print(f"  peg yaw phi (folded):  {math.degrees(phi):+.3f} deg  "
                      f"(expect +-0.000 at home BEFORE reset noise; ~uniform in +-45 after; "
                      f"free-yaw window +-{math.degrees(proxytask_tasks_cfg.YAW_WINDOW_RAD):.2f})")
                print(f"  (cos 4phi, sin 4phi):  ({math.cos(4.0 * phi_raw):+.4f}, {math.sin(4.0 * phi_raw):+.4f})"
                      f"  (expect (+1, 0) at home before reset noise)")
                print(f"  tip Chebyshev offset:  {tip_cheb:.6f} m  "
                      f"(expect <= {proxytask_tasks_cfg.SIDE_CLEARANCE:.4f} at phi=0 for the gate; "
                      f"at home under this tilt ~{home_tip_height * tilt_s:.4f})")
                print(f"  corner Chebyshev max:  {cheb:.6f} m  (gate needs < {half_pocket:.4f}); "
                      f"corners rel. entrance: {[ [round(v, 4) for v in c.tolist()] for c in corners ]}")
                print(f"  gate at this pose:     {'OPEN' if gate_e else 'closed'} "
                      f"(corners {'in' if cheb < half_pocket else 'OUT'}, "
                      f"depth {depth.item():+.4f} vs <= {cfg.fixed_asset.depth + 0.001:.4f}, "
                      f"alignment {alignment_e:+.4f} vs >= {cfg.gate_min_alignment})")
                print(f"  tip insertion depth:   {depth.item():+.6f} m  "
                      f"(expect {(cfg.peg_length - cfg.home_standoff_z) * tilt_c:+.4f} +- 0.005 at home"
                      f"{'' if self._tilt_rot is None else f', = {cfg.peg_length - cfg.home_standoff_z:+.4f} x cos(tilt)'})")

            dev = (self.robot.data.joint_pos[e] - commanded[e]).abs()
            worst = int(torch.argmax(dev).item())
            print(f"  max joint deviation:   {dev[worst].item():.6f} rad "
                  f"({torch.rad2deg(dev[worst]).item():.3f} deg) on {self.robot.joint_names[worst]}")
            hx, hy = cfg.plate_half_extents
            print("  body positions (env-local) and plate test"
                  f"{' -- WORLD-FRAME, indicative only under tilt' if self._tilt_rot is not None else ''}:")
            for bi, bname in enumerate(self.robot.body_names):
                bp = self.robot.data.body_pos_w[e, bi] - origins[e]
                bx, by, bz = bp[0].item(), bp[1].item(), bp[2].item()
                over_plate = abs(bx) <= hx and abs(by) <= hy
                in_plate = over_plate and cfg.plate_bottom_z <= bz < cfg.plate_top_z - 1e-3
                if in_plate:
                    note = "  <-- INSIDE THE PLATE"
                elif over_plate and bz < cfg.plate_bottom_z:
                    note = "  <-- under the plate, between the legs"
                elif not over_plate and bz < cfg.plate_top_z:
                    note = "  (beside the table, below plate level - no contact)"
                else:
                    note = ""
                print(f"    {bname:<16} ({bx:+.4f}, {by:+.4f}, {bz:+.4f})   "
                      f"clearance = {bz - cfg.plate_top_z:+.4f}{note}")
        print("=" * 72)
