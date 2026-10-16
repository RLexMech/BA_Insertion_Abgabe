# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from . import proxytask_tasks_cfg
from .proxytask_env_cfg import ProxytaskEnvCfg, resolve_fixture_usd_path
from .proxytask_tasks_cfg import resolve_peg_robot_usd_path

CODE_MARKER = "demo-sprint-2026-07-26a"


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


@torch.jit.script
def compute_rewards(
    tip_local: torch.Tensor,
    bore_local: torch.Tensor,
    depth: torch.Tensor,
    gate: torch.Tensor,
    below_plate: torch.Tensor,
    alignment: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reward, maximum insertion depth and success for each environment."""
    distance = torch.linalg.norm(tip_local - bore_local, dim=-1)
    approach = -w_approach * distance
    depth_gated = torch.where(gate, depth, torch.zeros_like(depth))
    # Reward only new depth records to prevent repeated rewards for moving in and out.
    new_max_depth = torch.maximum(max_depth, depth_gated)
    progress = w_depth * (new_max_depth - max_depth)
    success = (depth >= success_depth) & gate
    success_bonus = torch.where(success, torch.full_like(depth, w_success), torch.zeros_like(depth))
    action_penalty = -w_action * torch.sum(actions * actions, dim=-1)
    misplaced_depth = torch.where(gate, torch.zeros_like(depth), torch.clamp(depth, min=0.0))
    misplaced_penalty = -w_misplaced * misplaced_depth
    align_penalty = -w_align * (1.0 - alignment)
    per_step = approach + progress + success_bonus + action_penalty + misplaced_penalty + align_penalty
    # Charge the remaining cost so leaving the arena cannot avoid negative rewards.
    out_of_bounds_penalty = torch.where(
        below_plate,
        (approach + misplaced_penalty + align_penalty) * remaining_steps,
        torch.zeros_like(distance),
    )
    reward = per_step + out_of_bounds_penalty
    return reward, new_max_depth, success


class ProxytaskEnv(DirectRLEnv):
    """Cylindrical insertion with joint-position deltas and 19 observation values."""

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

        self._bore_local = torch.tensor(cfg.bore_entrance_pos, device=self.device)
        self._tip_offset_local = torch.tensor(cfg.peg_tip_offset, device=self.device)

        self._step_dt = cfg.sim.dt * cfg.decimation

        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._prev_joint_pos = self.robot.data.default_joint_pos.clone()
        self._max_depth = torch.zeros(self.num_envs, device=self.device)

        self._last_tip_local = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_depth = torch.zeros(self.num_envs, device=self.device)
        self._last_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_below_plate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_alignment = torch.ones(self.num_envs, device=self.device)
        self._last_remaining = torch.zeros(self.num_envs, device=self.device)
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._ep_count = 0
        self._ep_success_count = 0
        self._ep_depth_sum = 0.0
        self._episodes_to_threshold: int | None = None
        window = max(2000, int(self.num_envs * self.cfg.metrics_window_iterations * 16 / 240))
        self._recent_successes: collections.deque = collections.deque(maxlen=window)
        self._recent_depths: collections.deque = collections.deque(maxlen=window)
        self._metrics_path = pathlib.Path("logs") / "demo_metrics.json"
        self._metrics_dump_every = max(50, window // 8)

    def _setup_scene(self):
        fixture_cfg = self.cfg.fixture_cfg
        fixture_cfg.usd_path = resolve_fixture_usd_path()
        self._fixture_usd_path = fixture_cfg.usd_path
        fixture_cfg.func(
            "/World/envs/env_0/Fixture",
            fixture_cfg,
            translation=self.cfg.fixture_pos,
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        self.cfg.robot_cfg.spawn.usd_path = resolve_peg_robot_usd_path()
        self._robot_usd_path = self.cfg.robot_cfg.spawn.usd_path

        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        clamped = actions.clamp(-1.0, 1.0)
        self._joint_targets = self._joint_targets + self.cfg.action_scale * clamped
        limits = self.robot.data.soft_joint_pos_limits
        self._joint_targets = torch.clamp(self._joint_targets, limits[..., 0], limits[..., 1])

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets)

    def _peg_geometry(self):
        """Return environment-local tip position, depth, bore gate, out-of-bounds flag and alignment."""
        origins = self.scene.env_origins
        peg_pos_w = self.robot.data.body_pos_w[:, self._peg_body_idx]
        peg_quat_w = self.robot.data.body_quat_w[:, self._peg_body_idx]
        peg_axes = _axes_from_quat_batched(peg_quat_w)
        tip_w = peg_pos_w + torch.matmul(peg_axes, self._tip_offset_local)
        tip_local = tip_w - origins
        depth = self.cfg.plate_top_z - tip_local[:, 2]
        xy_inside = (
            torch.linalg.norm(tip_local[:, :2] - self._bore_local[:2], dim=-1)
            < proxytask_tasks_cfg.RADIAL_CLEARANCE
        )
        # Alignment is +1 when the peg axis points vertically down.
        alignment = -peg_axes[:, 2, 2]
        # The 1 mm depth margin allows solver penetration and floating-point error.
        gate = (
            xy_inside
            & (depth <= self.cfg.fixed_asset.depth + 0.001)
            & (alignment >= self.cfg.gate_min_alignment)
        )
        below_plate = tip_local[:, 2] < self.cfg.plate_bottom_z
        return tip_local, depth, gate, below_plate, alignment

    def _get_observations(self) -> dict:
        self._obs_calls += 1

        joint_pos = self.robot.data.joint_pos
        # Finite differences avoid the solver velocity offset observed at rest.
        joint_vel_fd = (joint_pos - self._prev_joint_pos) / self._step_dt
        self._prev_joint_pos = joint_pos.clone()

        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        if self._peg_body_idx is not None:
            tip_local = self._peg_geometry()[0]
            tip_rel_bore = tip_local - self._bore_local
        else:
            tip_rel_bore = torch.zeros(self.num_envs, 3, device=self.device)

        # Layout: joint position 6, velocity 6, relative tip position 3, quaternion (wxyz) 4.
        obs = torch.cat((joint_pos, joint_vel_fd, tip_rel_bore, ee_quat_w), dim=-1)

        if self._obs_calls in self._report_at_obs_calls:
            self._print_startup_report(joint_vel_fd)

        return {"policy": obs}

    # DirectRLEnv calls this before _get_rewards(); cache geometry for that call.
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self._peg_body_idx is not None:
            tip_local, depth, gate, below_plate, alignment = self._peg_geometry()
            success = (depth >= self.cfg.fixed_asset.success_depth) & gate
            self._last_tip_local = tip_local
            self._last_depth = depth
            self._last_gate = gate
            self._last_below_plate = below_plate
            self._last_alignment = alignment
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
            self._bore_local,
            self._last_depth,
            self._last_gate,
            self._last_below_plate,
            self._last_alignment,
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
        noise_scale = self.cfg.reset_joint_noise if ready else 0.0
        joint_pos = joint_pos + (torch.rand_like(joint_pos) * 2.0 - 1.0) * noise_scale
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
            "peg_diameter_m": self.cfg.held_asset.diameter,
            "radial_clearance_mm": proxytask_tasks_cfg.RADIAL_CLEARANCE * 1000.0,
            "reset_joint_noise_rad": self.cfg.reset_joint_noise,
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
        if abs(size[0] - cfg.held_asset.diameter) > 1e-4:
            problems.append(f"diameter {size[0]:.5f} m authored vs {cfg.held_asset.diameter:.5f} m configured")
        if abs(size[2] - cfg.peg_length) > 1e-4:
            problems.append(f"length {size[2]:.5f} m authored vs {cfg.peg_length:.5f} m configured")
        if abs(mass - cfg.held_asset.mass) > 1e-5:
            problems.append(f"mass {mass:.5f} kg authored vs {cfg.held_asset.mass:.5f} kg configured")
        if problems:
            return ("*** ASSET DOES NOT MATCH THE CONFIGURATION *** " + "; ".join(problems)
                    + ". The USD is stale: re-run scripts/author_peg_ur10e.py, then "
                      "scripts/verify_peg_passability.py. Every peg number in this report is void.")
        return (f"PASS -- authored geometry matches cfg (diameter {size[0]:.5f} m, "
                f"length {size[2]:.5f} m, mass {mass:.5f} kg), from {sidecar.name}")

    def _print_startup_report(self, joint_vel_fd: torch.Tensor) -> None:
        cfg = self.cfg
        origins = self.scene.env_origins
        bore = torch.tensor(cfg.bore_entrance_pos, device=self.device)
        commanded = self.robot.data.default_joint_pos
        down = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        sim_time = self._obs_calls * cfg.sim.dt * cfg.decimation
        print("=" * 72)
        print(f"PROXYTASK STARTUP REPORT (demo sprint, branch demo-insertion-sprint, UNVERIFIED)")
        print(f"step {self._obs_calls} of {self._report_at_obs_calls}, t = {sim_time:.3f} s after reset")
        print(f"code marker: {CODE_MARKER}")
        print(f"fixture usd: {self._fixture_usd_path}")
        print(f"robot usd:   {self._robot_usd_path}")
        print(f"fixture spawn translation: {cfg.fixture_pos}")
        print(f"robot base sits on the plate top: base z {cfg.robot_base_pos[2]:.4f} "
              f"= plate top {cfg.plate_top_z:.4f}; "
              f"{proxytask_tasks_cfg.BASE_TO_PLATE_EDGE*1000:.0f} mm from base centre to plate edge, "
              f"{proxytask_tasks_cfg.BORE_TO_BASE_DISTANCE*1000:.0f} mm to the bore")
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
                  f"length {cfg.peg_length:.4f} m, diameter {cfg.held_asset.diameter:.4f} m, "
                  f"mass {cfg.held_asset.mass:.5f} kg")
            print(f"bore:        diameter {cfg.fixed_asset.diameter:.4f} m, depth {cfg.fixed_asset.depth:.4f} m, "
                  f"radial clearance {proxytask_tasks_cfg.RADIAL_CLEARANCE*1000:.2f} mm (demo sprint), "
                  f"success at {cfg.fixed_asset.success_depth*1000:.0f} mm")
        print(f"asset check: {self._asset_geometry_line()}")
        resting_speed = torch.linalg.norm(joint_vel_fd[: min(2, self.num_envs)], dim=-1)
        print(f"resting FD joint speed (env 0..1): {resting_speed.tolist()} rad/s  (expect ~0.000 after reset)")
        print("observation layout (19-dim): 0:6 joint_pos | 6:12 joint_vel (finite-diff) | "
              "12:15 peg tip - bore entrance (env-local) | 15:19 EE quat (wxyz)")
        for e in range(min(2, self.num_envs)):
            root_local = self.robot.data.root_pos_w[e] - origins[e]
            ee_pos_w = self.robot.data.body_pos_w[e, self._ee_body_idx]
            ee_quat_w = self.robot.data.body_quat_w[e, self._ee_body_idx]
            ee_local = ee_pos_w - origins[e]
            bore_w = bore + origins[e]
            delta = ee_pos_w - bore_w
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
            print(f"  bore world pos:        {bore_w.tolist()}")
            print(f"  |EE - bore|:           {torch.linalg.norm(delta).item():.6f} m")
            print(f"  XY offset to bore:     {xy_offset.item():.6f} m  (expect <= 0.005)")
            print(f"  Z standoff to bore:    {z_standoff.item():.6f} m  (expect {cfg.home_standoff_z} +- 0.005)")
            for name, col in zip("xyz", range(3)):
                dot = torch.dot(axes[:, col], down).item()
                note = "  <-- expect >= 0.999 here" if name == "z" else ""
                print(f"  {name}-axis . (0,0,-1):    {dot:+.6f}{note}")
            if self._peg_body_idx is not None:
                peg_pos_w = self.robot.data.body_pos_w[e, self._peg_body_idx]
                peg_axes = _axes_from_quat(self.robot.data.body_quat_w[e, self._peg_body_idx])
                tip_w = peg_pos_w + peg_axes @ torch.tensor(cfg.peg_tip_offset, device=self.device)
                tip_local = tip_w - origins[e]
                tip_xy = torch.linalg.norm((tip_w - bore_w)[:2])
                depth = cfg.plate_top_z - tip_local[2]
                along_tool = torch.dot(tip_w - ee_pos_w, axes[:, 2]).item()
                print(f"  peg origin env-local:  {(peg_pos_w - origins[e]).tolist()}")
                print(f"  peg TIP env-local:     {tip_local.tolist()}")
                print(f"  flange -> peg origin:  {torch.linalg.norm(peg_pos_w - ee_pos_w).item():.6f} m"
                      f"  (expect 0.000000)")
                print(f"  tip along tool axis:   {along_tool:+.6f} m  (expect {cfg.peg_length:+.4f})")
                print(f"  tip XY offset to bore: {tip_xy.item():.6f} m  (expect <= {proxytask_tasks_cfg.RADIAL_CLEARANCE:.4f} for the gate)")
                print(f"  tip insertion depth:   {depth.item():+.6f} m  "
                      f"(expect {cfg.peg_length - cfg.home_standoff_z:+.4f} +- 0.005 at home)")

            dev = (self.robot.data.joint_pos[e] - commanded[e]).abs()
            worst = int(torch.argmax(dev).item())
            print(f"  max joint deviation:   {dev[worst].item():.6f} rad "
                  f"({torch.rad2deg(dev[worst]).item():.3f} deg) on {self.robot.joint_names[worst]}")
            hx, hy = cfg.plate_half_extents
            print("  body positions (env-local) and plate test:")
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
