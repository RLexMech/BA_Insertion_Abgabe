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

# Marker for the manual two-machine copy. If the startup report does not show
# this string, the copy on the training machine predates the current code.
#
# DEMO SPRINT (branch demo-insertion-sprint, 2026-07-26): a separate,
# radically simplified increment cut from increment-2-verified (90621a8), NOT
# a continuation of increment 3. Joint-delta actions, a dense reward, a
# 19-dim observation with finite-differenced velocities (D-030's layout minus
# the wrench/contact-sensor channels, since there is no force reward
# tonight), and a Ø25 mm peg (was Ø28 mm). UNVERIFIED until run on the
# training machine per the plan's step 5.
CODE_MARKER = "demo-sprint-2026-07-26a"


def _axes_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Return the three body-frame axes of quaternion (w, x, y, z) as columns.

    Written out instead of using isaaclab.utils.math.matrix_from_quat: that
    import was never covered by probe_assets.py, and an import error in this
    module would also break task registration. Single-quaternion form, used
    only by the startup report (env 0/1 printout); the per-step reward and
    observation paths use the batched form below, which increment 3 verified
    agrees with this one to 0.000e+00 m.
    """
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
    """Batched form of ``_axes_from_quat``: (N, 4) wxyz -> (N, 3, 3), axes as columns.

    Ported from increment 3 (proxytask_env.py, branch increment-3-contact-obs),
    where it was verified on the training machine against the single-quaternion
    form to 0.000e+00 m at both 2 and 128 environments. Copied as text, not
    imported: this branch does not depend on increment 3's module.
    """
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
    """Dense demo-sprint reward, as a free-standing scripted function outside
    the env class (project convention -- see the template's ``compute_rewards``
    for Cartpole, which the earlier increments left as a pattern to follow
    once a real reward exists).

    Five exploit questions this shape was checked against (CLAUDE.md: every
    reward edit must answer them):
      - Gate-farming (cycle in/out of the XY gate to re-collect a depth
        reward): worthless, because progress only pays on a NEW maximum depth
        (``max_depth`` is monotonic per episode), not on depth re-entered.
      - Tunneling through the plate beside the bore (PhysX penetration under
        stiff position targets): does not count as success, because success
        requires the XY gate in addition to depth.
      - **Diving under the plate.** ``depth`` is ``plate_top_z - tip_z`` and is
        unbounded below the plate, where it reads ~55 mm -- past the 25 mm
        success threshold -- while the 3D distance to the bore entrance is
        *smaller* than at the home pose. Until 2026-07-27 that paid the full
        depth reward plus the success bonus, about 16.5 points, for going
        under the table instead of into the hole, and a training run was
        observed doing exactly that. Two things close it. ``gate`` requires
        the tip to be inside the bore volume (``depth <= bore_depth``), so no
        state below the plate can pay depth or count as success. And closing
        the gate alone is not enough: the approach term still rates the space
        under the bore above the home pose, so ``w_misplaced`` charges for
        depth below the plate top that is *not* inside the bore, which is
        material the peg has no business being in. That makes going under the
        table strictly worse than staying above it, rather than merely
        unrewarded.
      - **Suicide, i.e. ending the episode early to stop paying the negative
        approach term.** Leaving the arena terminates, which would otherwise
        be worth far more than finishing the episode: at ~0.17 per step, a
        200-step remainder is ~34 points. The out-of-bounds penalty is
        therefore exactly the approach cost that would have been paid by
        standing still for the rest of the episode, which makes terminating
        neutral rather than profitable, while still freeing the simulation
        from a hopeless episode.
      - **Parking the tip on the bore with the peg lying across it.** A
        distance-only approach term is maximised by putting the tip on the
        hole from ANY direction, and a training run converged on exactly
        that: every arm reached in sideways, scored an approach term of
        about zero -- the best value reachable without inserting -- and
        stopped. Leaving that pose costs before it pays, because standing
        the peg up moves the tip away from the bore, so the policy never
        left it. ``w_align`` charges for the tool axis departing from
        world-down, and the gate additionally requires alignment, since a
        peg across the hole cannot enter it however close its tip is.
      - Escaping the approach term any other way: success and out-of-bounds
        are the only terminations, and both are priced above.
    """
    distance = torch.linalg.norm(tip_local - bore_local, dim=-1)
    approach = -w_approach * distance
    depth_gated = torch.where(gate, depth, torch.zeros_like(depth))
    new_max_depth = torch.maximum(max_depth, depth_gated)
    progress = w_depth * (new_max_depth - max_depth)
    success = (depth >= success_depth) & gate
    success_bonus = torch.where(success, torch.full_like(depth, w_success), torch.zeros_like(depth))
    action_penalty = -w_action * torch.sum(actions * actions, dim=-1)
    # Depth below the plate top that is not inside the bore: plate material,
    # or the space under the table. Charged per step, so the state is worse
    # than staying above the plate rather than merely unrewarded.
    misplaced_depth = torch.where(gate, torch.zeros_like(depth), torch.clamp(depth, min=0.0))
    misplaced_penalty = -w_misplaced * misplaced_depth
    # 0 when the tool axis points straight down, w_align when it lies flat.
    align_penalty = -w_align * (1.0 - alignment)
    per_step = approach + progress + success_bonus + action_penalty + misplaced_penalty + align_penalty
    # Leaving the arena ends the episode, so charge the per-step cost that the
    # remaining steps would have accrued. Priced, not punitive: terminating is
    # then neither a shortcut nor an extra punishment.
    out_of_bounds_penalty = torch.where(
        below_plate,
        (approach + misplaced_penalty + align_penalty) * remaining_steps,
        torch.zeros_like(distance),
    )
    reward = per_step + out_of_bounds_penalty
    return reward, new_max_depth, success


class ProxytaskEnv(DirectRLEnv):
    """Demo-sprint env: joint-delta actions, dense reward, 19-dim observation.

    Branched from the increment-2-verified commissioning env (fixture + arm
    with the welded peg, zero reward, timeout-only dones, no applied
    actions). This branch adds:
      - joint-position-delta actions (NVIDIA UR10-Reach pattern, no OSC),
      - a 19-dim observation with finite-differenced joint velocities
        (the raw solver ``joint_vel`` channel has a measured ~0.05 rad/s
        phantom at rest -- D-030 addendum 2 -- so it is not used here),
      - a dense, potential-based reward and an XY-gated success criterion,
      - small joint-space reset noise,
      - running success-rate/depth metrics, logged to ``self.extras["log"]``
        and periodically dumped to ``logs/demo_metrics.json``.

    Everything else (scene setup, peg-tip geometry, the startup report
    mechanics) is inherited unchanged from the verified increment-2 code.
    """

    cfg: ProxytaskEnvCfg

    def __init__(self, cfg: ProxytaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]

        # The peg is a welded link of the articulation (D-019), so it either
        # came through the USD authoring or it did not exist at all. A missing
        # body must degrade to a loud report line rather than an exception:
        # a crash here costs a full copy-run round trip on the training machine
        # and tells us less than the body list would.
        try:
            self._peg_body_idx = self.robot.find_bodies(self.cfg.peg_body_name)[0][0]
        except (IndexError, KeyError, ValueError):
            self._peg_body_idx = None

        self._report_at_obs_calls = tuple(self.cfg.report_at_steps)
        self._obs_calls = 0

        # Cached constants for the per-step geometry (bore entrance, tip
        # offset) -- avoid rebuilding these tensors every step.
        self._bore_local = torch.tensor(cfg.bore_entrance_pos, device=self.device)
        self._tip_offset_local = torch.tensor(cfg.peg_tip_offset, device=self.device)

        # Control-step duration, for the finite-difference joint velocity.
        # Same quantity the startup report already derives sim_time from.
        self._step_dt = cfg.sim.dt * cfg.decimation

        # Per-env buffers, all reset in _reset_idx.
        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._prev_joint_pos = self.robot.data.default_joint_pos.clone()
        self._max_depth = torch.zeros(self.num_envs, device=self.device)

        # Values from the current step's _get_dones() call, read back by
        # _get_rewards() and by _reset_idx() for the envs that just finished.
        # DirectRLEnv's step loop calls _get_dones() BEFORE _get_rewards(),
        # so the geometry/success computation lives in _get_dones() and is
        # cached forward -- not the other way round.
        self._last_tip_local = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_depth = torch.zeros(self.num_envs, device=self.device)
        self._last_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_below_plate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_alignment = torch.ones(self.num_envs, device=self.device)
        self._last_remaining = torch.zeros(self.num_envs, device=self.device)
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Running episode metrics (whole-run cumulative; simple and honest for
        # a single overnight run -- a rolling window is not worth the
        # complexity here).
        self._ep_count = 0
        self._ep_success_count = 0
        self._ep_depth_sum = 0.0
        self._episodes_to_threshold: int | None = None
        # Trailing window, so a run is ranked by how the policy performs NOW
        # rather than by an average dragged down by its untrained start.
        #
        # Sized against the env count rather than fixed: episodes per
        # iteration scale with it (about 273 at 4096 envs and 240-step
        # episodes, against 68 at 1024), so a fixed window covers four times
        # less training at 4096 and turns into a snapshot of a handful of
        # iterations. The floor keeps the statistics sound at small env
        # counts -- 2000 episodes put the standard error of a 90 % rate at
        # 0.7 points.
        window = max(2000, int(self.num_envs * self.cfg.metrics_window_iterations * 16 / 240))
        self._recent_successes: collections.deque = collections.deque(maxlen=window)
        self._recent_depths: collections.deque = collections.deque(maxlen=window)
        # Fallback only; the real path is the run's own log dir (see
        # _write_metrics), which train.py assigns to cfg.log_dir.
        self._metrics_path = pathlib.Path("logs") / "demo_metrics.json"
        # Also scaled: at 4096 envs a fixed 50 would rewrite the file several
        # times per iteration, which is pure I/O for no extra information.
        self._metrics_dump_every = max(50, window // 8)

    def _setup_scene(self):
        # Fixture into env_0 before cloning, so it replicates into every env.
        # Not registered in self.scene: a static collider has no runtime state.
        fixture_cfg = self.cfg.fixture_cfg
        fixture_cfg.usd_path = resolve_fixture_usd_path()
        self._fixture_usd_path = fixture_cfg.usd_path
        fixture_cfg.func(
            "/World/envs/env_0/Fixture",
            fixture_cfg,
            translation=self.cfg.fixture_pos,
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # Point the spawn at the peg-welded USD (D-019). Done here rather than
        # in ur10e_cfg.py for the same reason the fixture path is: resolving at
        # module import would raise on a machine without the asset and take
        # task registration down with it.
        self.cfg.robot_cfg.spawn.usd_path = resolve_peg_robot_usd_path()
        self._robot_usd_path = self.cfg.robot_cfg.spawn.usd_path

        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
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
        """Per-env peg-tip position (env-local), insertion depth, bore gate,
        out-of-arena flag and tool-axis alignment.

        Shared by the observation, reward and done computations so the tip
        math (peg pose -> tool axes -> tip offset -> env-local) is written
        once. Returns (tip_local (N,3), depth (N,), gate (N,) bool,
        below_plate (N,) bool, alignment (N,)).

        The gate is deliberately a **bore-volume** test, not just a lateral
        one. ``depth`` is ``plate_top_z - tip_z`` and keeps growing below the
        plate, where it passes the success threshold while the tip is nowhere
        near the hole; requiring ``depth <= bore_depth`` closes the space
        under the table that a training run was observed exploiting on
        2026-07-27.

        ``alignment`` is the peg's own tool axis dotted with world-down, i.e.
        +1 pointing straight into the plate. The home pose reads 0.9998. It
        gates depth as well, because a peg lying across the bore cannot enter
        it however close its tip is.
        """
        origins = self.scene.env_origins
        peg_pos_w = self.robot.data.body_pos_w[:, self._peg_body_idx]
        peg_quat_w = self.robot.data.body_quat_w[:, self._peg_body_idx]
        peg_axes = _axes_from_quat_batched(peg_quat_w)
        # peg_axes is (N, 3, 3), self._tip_offset_local is (3,): matmul promotes
        # the 1-D vector to a batched matrix-vector product and drops the
        # prepended dimension again, giving (N, 3) directly.
        tip_w = peg_pos_w + torch.matmul(peg_axes, self._tip_offset_local)
        tip_local = tip_w - origins
        depth = self.cfg.plate_top_z - tip_local[:, 2]
        xy_inside = (
            torch.linalg.norm(tip_local[:, :2] - self._bore_local[:2], dim=-1)
            < proxytask_tasks_cfg.RADIAL_CLEARANCE
        )
        # The blind bore bottoms out at bore_depth; anything deeper than that
        # is not in the hole, it is through or under the plate. The 1 mm
        # tolerance covers float32 (0.755 - 0.725 lands just above 0.030 and
        # would otherwise disqualify the bore bottom itself) and the small
        # penetration PhysX allows at the stop. It is 55x smaller than the
        # depth read under the plate, so it does not reopen that hole.
        # Peg tool axis (its own local +z) dotted with world-down. Columns of
        # peg_axes are the body axes, so [:, :, 2] is the tool axis and the
        # dot with (0, 0, -1) is minus its z component.
        alignment = -peg_axes[:, 2, 2]
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
        joint_vel_fd = (joint_pos - self._prev_joint_pos) / self._step_dt
        self._prev_joint_pos = joint_pos.clone()

        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        if self._peg_body_idx is not None:
            tip_local = self._peg_geometry()[0]
            tip_rel_bore = tip_local - self._bore_local
        else:
            tip_rel_bore = torch.zeros(self.num_envs, 3, device=self.device)

        obs = torch.cat((joint_pos, joint_vel_fd, tip_rel_bore, ee_quat_w), dim=-1)

        if self._obs_calls in self._report_at_obs_calls:
            self._print_startup_report(joint_vel_fd)

        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Runs BEFORE _get_rewards() in DirectRLEnv's step loop, so the
        # per-step peg geometry and success flag are computed here and cached
        # for _get_rewards() to reuse, not the other way round.
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
            # Steps this episode would still have run. Used to price the
            # out-of-bounds termination so that leaving is neutral rather than
            # a way to stop paying the approach term.
            self._last_remaining = (self.max_episode_length - 1 - self.episode_length_buf).clamp(min=0).float()
            # Below the plate the arm is under the table, where nothing it can
            # do earns anything: end the episode instead of spending the rest
            # of it there.
            terminated = success | below_plate
        else:
            terminated = torch.zeros_like(time_out)
            self._last_depth = torch.zeros(self.num_envs, device=self.device)
            self._last_success = terminated
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        if self._peg_body_idx is None:
            # No welded peg -- nothing meaningful to reward. Loud rather than
            # silent, mirroring the startup report's MISSING line.
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

        # Whether this env's own buffers exist yet. DirectRLEnv may reset from
        # inside super().__init__(), i.e. before the buffers below __init__'s
        # super() call are created; the base class's version of this method
        # touched nothing but robot data, so it was safe. Guarding rather than
        # assuming: an AttributeError here costs a full copy-run round trip on
        # the training machine, and the buffers are written again by the first
        # real reset anyway.
        ready = hasattr(self, "_joint_targets")

        if ready:
            self._log_finished_episodes(env_ids)

        super()._reset_idx(env_ids)

        # Home pose plus small joint-space noise (no IK needed, unlike a
        # cartesian randomization would require). Deliberately small: enough
        # that the policy cannot memorize one exact trajectory, not enough to
        # fight the fixed 13.4 mm gravity droop (D-026) on top of it.
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
        """Record success/depth for the envs about to reset, and surface them
        to rsl_rl (``self.extras["log"]``) and to disk, per the project rule
        that results are read from disk, not from prose.

        Two figures are kept, and the difference matters when comparing runs.
        The **cumulative** rate covers every episode of the run, including the
        untrained start, so it understates a policy that learned late. The
        **recent** rate covers the last ``_recent_window`` episodes and is the
        one to rank runs by: it answers "how good is this policy now", which
        is what picking a checkpoint asks.

        Skipped until the first episode has actually stepped: the startup
        reset would otherwise book num_envs zero-depth failures and drag the
        very success rate this run is meant to report.
        """
        if len(env_ids) == 0 or self._peg_body_idx is None or self._obs_calls == 0:
            return
        idx = torch.as_tensor(list(env_ids), device=self.device) if not torch.is_tensor(env_ids) else env_ids
        # Success terminates the episode on this branch, so an episode that
        # succeeded is resetting with the flag still set -- reading it here is
        # equivalent to "did this episode succeed". Depth is the gated maximum
        # reached inside the bore, which says more than the final-step value.
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
        # Sample efficiency: the episode count at which the policy first held
        # the threshold. Once two configurations both finish at 100 %, the
        # final rate no longer separates them and this is what does -- how
        # much experience each needed to get there. Latched, so it records
        # the first time rather than the last.
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
        """Dump the run's metrics beside its own logs.

        ``train.py`` assigns ``env_cfg.log_dir`` the run directory
        (``logs/rsl_rl/<experiment>/<timestamp>_<run_name>``), so each run
        writes its own file. Writing to a fixed path instead -- as this did
        until 2026-07-27 -- means concurrent or successive runs silently
        overwrite each other, which is exactly wrong for comparing policies.
        """
        log_dir = getattr(self.cfg, "log_dir", None)
        path = pathlib.Path(log_dir) / "demo_metrics.json" if log_dir else self._metrics_path
        payload = {
            "episodes": self._ep_count,
            "success_rate_recent": recent_rate,
            # None until the threshold is first held; the comparison figure
            # once several configurations all reach 100 %.
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
        """Compare the authored asset's *measured* geometry against the cfg constants.

        Ported from increment 3 (commit 4908adb), which added it after a stale
        copy on the training machine re-authored a peg the configuration had
        already moved away from. The failure mode is structural, not careless:
        everything in the environment reads the same constants, so a
        configuration that no longer matches the USD on disk is perfectly
        self-consistent and silent -- the report's own peg lines print
        ``cfg.held_asset.diameter``, which is the number that was wrong.

        It matters doubly on this branch, whose whole point is a changed peg
        diameter, and which shares a training-machine folder with increment 3's
        differently-sized peg.

        ``author_peg_ur10e.py`` writes a sidecar next to the USD with the
        geometry it measured off the authored stage. That sidecar is the only
        thing in the pipeline describing the asset rather than the intent, so it
        is what this compares against.
        """
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
            # The authoring report nests its measurements under "after"; the
            # top-level fallback covers a sidecar written by an older revision.
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

        # Diameter is the cross-section, i.e. the two equal axes of the local
        # box; length runs along local z. Compared at 0.1 mm, far below the
        # 3 mm step this branch makes from increment 3's peg.
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

        # Computed from cfg rather than read off the base class: step_dt was not
        # covered by probe_assets.py.
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
        # Resting finite-differenced joint speed. This is the demo sprint's
        # substitute for the solver-flag experiment: it must read ~0.000 at
        # the early/late reports below (velocity derived from positions, not
        # from the raw solver channel), unlike the raw joint_vel channel's
        # measured ~0.05 rad/s phantom (D-030 addendum 2).
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
