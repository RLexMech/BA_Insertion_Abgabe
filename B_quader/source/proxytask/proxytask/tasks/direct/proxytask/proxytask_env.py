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

# Marker for the manual two-machine copy. If the startup report does not show
# this string, the copy on the training machine predates the current code.
#
# SQUARE-PEG PIVOT (branch square-peg-insertion, 2026-07-28, D-029): cut from
# the demo sprint (a7c7ebb), keeping its joint-delta actions and dense-reward
# skeleton. The peg is a 30 x 30 x 50 mm square prism, the hole a
# 32 x 32 x 35 mm blind pocket in the re-imported table. New here: the
# four-corner containment gate, a 25-dim observation (D-029's (cos 4phi,
# sin 4phi) of the tool-axis yaw at 19:21, D-037's pocket quaternion at
# 21:25), a C4-invariant yaw reward
# term, and +-45 deg reset randomization on wrist_3. UNVERIFIED until run on
# the training machine.
CODE_MARKER = "square-peg-2026-08-16-d038"


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


def _quat_mul_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (N, 4) wxyz quaternion batches: a then b, i.e.
    the rotation matrix of the result is R(a) @ R(b).

    Used to compose the pocket's yaw with its tilt into one orientation
    (D-037). Written out rather than imported so this module keeps its
    single-file convention; the same expansion isaaclab.utils.math.quat_mul
    uses.
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
    """Dense reward for the square-peg task, as a free-standing scripted
    function outside the env class (project convention -- see the template's
    ``compute_rewards`` for Cartpole, which the earlier increments left as a
    pattern to follow once a real reward exists).

    Six exploit questions this shape was checked against (CLAUDE.md: every
    reward edit must answer them):
      - Gate-farming (cycle in/out of the containment gate to re-collect a
        depth reward): worthless, because progress only pays on a NEW maximum
        depth (``max_depth`` is monotonic per episode), not on depth
        re-entered.
      - Tunneling through the plate beside the pocket (PhysX penetration under
        stiff position targets): does not count as success, because success
        requires the containment gate in addition to depth.
      - **Diving under the plate.** ``depth`` is ``plate_top_z - tip_z`` and is
        unbounded below the plate, where it reads ~55 mm -- past the 25 mm
        success threshold -- while the 3D distance to the pocket entrance is
        *smaller* than at the home pose. Until 2026-07-27 that paid the full
        depth reward plus the success bonus, about 16.5 points, for going
        under the table instead of into the hole, and a training run was
        observed doing exactly that. Two things close it. ``gate`` requires
        the tip to be inside the pocket volume (``depth <= pocket_depth``,
        35 mm against the plate's 55 mm, so the margin holds), so no state
        below the plate can pay depth or count as success. And closing the
        gate alone is not enough: the approach term still rates the space
        under the pocket above the home pose, so ``w_misplaced`` charges for
        depth below the plate top that is *not* inside the pocket, which is
        material the peg has no business being in. That makes going under the
        table strictly worse than staying above it, rather than merely
        unrewarded.
      - **Suicide, i.e. ending the episode early to stop paying the negative
        approach term.** Leaving the arena terminates, which would otherwise
        be worth far more than finishing the episode: at ~0.17 per step, a
        200-step remainder is ~34 points. The out-of-bounds penalty is
        therefore the full recurring per-step state cost AT THE TERMINAL
        STATE -- approach plus misplaced plus align plus yaw -- times the
        remaining steps: exactly what standing still there would have accrued (standing
        still pays no action penalty and earns no progress, so those terms
        are correctly absent). Charging only the approach term, as an earlier
        revision of this comment claimed, would underprice termination from
        under the plate, where misplaced (~1.1/step) dominates approach
        (~0.13/step) -- dying there would then be ~1.1/step cheaper than
        standing still, and the exploit would reopen. Neutral, not punitive:
        terminating is neither a shortcut nor an extra punishment.
      - **Parking the tip on the pocket with the peg lying across it.** A
        distance-only approach term is maximised by putting the tip on the
        hole from ANY direction, and a training run converged on exactly
        that: every arm reached in sideways, scored an approach term of
        about zero -- the best value reachable without inserting -- and
        stopped. Leaving that pose costs before it pays, because standing
        the peg up moves the tip away from the pocket, so the policy never
        left it. ``w_align`` charges for the tool axis departing from
        world-down, and the gate additionally requires alignment, since a
        peg across the hole cannot enter it however close its tip is.
      - **Yaw-parking (new with the square peg).** Peg vertical, tip centred
        on the pocket, turned 45 degrees: approach ~0, align penalty ~0 --
        under the round task's terms this pose is indistinguishable from the
        correct pre-insertion pose, yet insertion is geometrically impossible
        (diagonal 42.4 mm > 32 mm opening). It is the exact yaw analogue of
        the sideways-parking attractor above, and with the +-45 deg reset
        noise on wrist_3 most episodes START near it. ``w_yaw`` charges
        (1 - cos 4 phi) / 2: zero at any of the four insertable orientations,
        maximal at 45 degrees, C4-invariant and continuous, so all four
        square symmetries of the goal stay equally rewarded and the policy is
        never asked to prefer one of them. Correcting yaw costs nothing
        before it pays: wrist_3 is a pure yaw actuator whose rotation leaves
        the tip -- and therefore the approach term -- exactly still, unlike
        the sideways case where the fix moves the tip away first.
      - Escaping the approach term any other way: success and out-of-bounds
        are the only terminations, and both are priced above.
    """
    distance = torch.linalg.norm(tip_local - entrance_local, dim=-1)
    approach = -w_approach * distance
    depth_gated = torch.where(gate, depth, torch.zeros_like(depth))
    new_max_depth = torch.maximum(max_depth, depth_gated)
    progress = w_depth * (new_max_depth - max_depth)
    success = (depth >= success_depth) & gate
    success_bonus = torch.where(success, torch.full_like(depth, w_success), torch.zeros_like(depth))
    action_penalty = -w_action * torch.sum(actions * actions, dim=-1)
    # Depth below the plate top that is not inside the pocket: plate material,
    # or the space under the table. Charged per step, so the state is worse
    # than staying above the plate rather than merely unrewarded.
    misplaced_depth = torch.where(gate, torch.zeros_like(depth), torch.clamp(depth, min=0.0))
    misplaced_penalty = -w_misplaced * misplaced_depth
    # 0 when the tool axis points straight down, w_align when it lies flat.
    align_penalty = -w_align * (1.0 - alignment)
    # 0 at any of the four insertable orientations, w_yaw at 45 degrees.
    # cos4phi rather than phi keeps the term C4-invariant and continuous.
    yaw_penalty = -w_yaw * (1.0 - cos4phi) * 0.5
    per_step = (
        approach + progress + success_bonus + action_penalty
        + misplaced_penalty + align_penalty + yaw_penalty
    )
    # Leaving the arena ends the episode, so charge the full recurring
    # per-step state cost AT THE TERMINAL STATE times the remaining steps --
    # exactly what standing still there would have accrued (standing still
    # pays no action penalty and earns no progress, so those terms are
    # correctly absent; the yaw term recurs and therefore belongs in the
    # price). Neutral, not punitive: terminating is then neither a shortcut
    # nor an extra punishment.
    out_of_bounds_penalty = torch.where(
        below_plate,
        (approach + misplaced_penalty + align_penalty + yaw_penalty) * remaining_steps,
        torch.zeros_like(distance),
    )
    reward = per_step + out_of_bounds_penalty
    return reward, new_max_depth, success


class ProxytaskEnv(DirectRLEnv):
    """Square-peg env: joint-delta actions, dense reward, 25-dim observation.

    Cut from the demo-sprint env (round Ø25 peg), keeping its verified
    skeleton: joint-position-delta actions (NVIDIA UR10-Reach pattern, no
    OSC), finite-differenced joint velocities (the raw solver ``joint_vel``
    channel has a measured ~0.05 rad/s phantom at rest -- D-030 addendum 2),
    the dense staged reward, and the running metrics dumped to
    ``demo_metrics.json``. The square pivot (D-029) changes:
      - the gate: a four-corner containment test of the tip cross-section
        against the 32 mm pocket, subsuming offset, yaw and tilt in one test,
      - the observation: 25-dim. Channels 19:21 are (cos 4 phi, sin 4 phi) of
        the peg yaw -- C4-invariant by construction, so the four insertable
        orientations are one point in observation space instead of four
        separate lessons -- and channels 21:25 are the pocket's own
        quaternion (D-037), without which a per-episode pocket orientation
        would be invisible to the policy,
      - the reward: a seventh term charging (1 - cos 4 phi) / 2,
      - the reset: +-45 deg uniform noise on wrist_3 (the yaw joint), small
        noise on the other five -- without it every start would lie inside
        the +-3.96 deg free-yaw window and the policy would never need to
        turn, making a 100 % success rate a non-result.
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

        # Pocket-entrance POSE, per env (D-034): position (N, 3) env-local plus
        # yaw (N,). The base pose is the cfg constant; when the noise ranges
        # are > 0 the reset re-samples the pose per episode and teleports the
        # kinematic fixture to match, so buffer and prim can only disagree if
        # the reset write is broken -- which is exactly what the startup
        # report's fixture line measures. At noise 0 nothing is ever written
        # and the buffers stay at the base pose: bit-identical pre-D-034
        # behaviour, the regression case.
        self._entrance_base = torch.tensor(cfg.opening_entrance_pos, device=self.device)
        self._entrance_pos = self._entrance_base.unsqueeze(0).repeat(self.num_envs, 1)
        self._fixture_yaw = torch.zeros(self.num_envs, device=self.device)
        self._tip_offset_local = torch.tensor(cfg.peg_tip_offset, device=self.device)

        # Pocket orientation, PER ENV (D-037; D-036 had one static matrix).
        #
        # _fixture_quat is the COMMANDED pocket orientation in the env frame
        # (wxyz, w >= 0) and is the single source of truth for it: _reset_idx
        # samples it and writes it to the kinematic fixture prim, _tilt_rot is
        # built from it, and observation channels 21:25 publish it to the
        # policy. Env origins are pure translations, so an orientation is the
        # same in the world and the env frame and the prim read-back in the
        # startup report can be compared against this buffer directly.
        #
        # Columns of _tilt_rot are the pocket-frame axes in env coordinates,
        # so _tilt_rot_t maps env-frame vectors INTO the pocket frame -- the
        # convention D-036's static R_y(tilt) already had, now one matrix per
        # env. None (not a stack of identities) whenever no tilt is configured
        # at all, so the untilted path stays byte-for-byte the pre-D-036 code:
        # the regression case costs no transform and cannot drift numerically.
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
            # Static probe: one fixed orientation, every env, whole run --
            # R_z(yaw) @ R_y(tilt), the same composition order as the
            # randomised reset (quat_yaw * quat_tilt). Under randomisation
            # this stays identity here and _reset_idx writes the per-episode
            # orientation instead. w stays positive for |yaw| <= 45 deg and
            # |tilt| <= 15 deg, so no sign canonicalisation is needed here.
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

        # Per-joint reset noise, built by joint NAME rather than by index:
        # the +-45 deg yaw randomization must land on wrist_3 and nowhere
        # else, and hard-coding column 5 would silently break if the
        # articulation ever enumerated its joints differently.
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

        # Control-step duration, for the finite-difference joint velocity.
        # Same quantity the startup report already derives sim_time from.
        self._step_dt = cfg.sim.dt * cfg.decimation

        # Per-env buffers, all reset in _reset_idx.
        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._prev_joint_pos = self.robot.data.default_joint_pos.clone()
        self._max_depth = torch.zeros(self.num_envs, device=self.device)
        # Worst joint-target lag reached in the current episode, per env; see
        # _pre_physics_step. Diagnosis instrument, not part of the task.
        self._max_target_lag = torch.zeros(self.num_envs, device=self.device)

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
        # cos 4 phi at phi = 0: the aligned pose, so the yaw penalty starts 0.
        self._last_cos4phi = torch.ones(self.num_envs, device=self.device)
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
        # Steps the SUCCESSFUL episodes took, over the same trailing window.
        # Answers "does the episode length bind?" as a read number: at ±5 cm
        # fixture noise the approach is longer, and if the mean creeps toward
        # the 240-step cap the length is the suspect, measured not guessed.
        self._recent_success_steps: collections.deque = collections.deque(maxlen=window)
        # Pocket tilt magnitude (degrees) of each episode in the same trailing
        # window, index-aligned with _recent_successes (D-037). Under a
        # randomised tilt the mean success rate is not a readable result: 90 %
        # can mean "90 % everywhere" or "100 % near upright and 0 % at the
        # edge", and the large angles are the entire point of the training.
        # Keeping the angle per episode lets _write_metrics report the rate per
        # angle bin instead of one number that hides its own failure mode.
        self._recent_tilt_deg: collections.deque = collections.deque(maxlen=window)
        # Upper bin edges in degrees; the last bin catches anything above.
        self._tilt_bin_edges_deg = (3.0, 6.0, 9.0, 12.0, 15.0)
        # Worst joint-target lag per episode, index-aligned with
        # _recent_successes so the statistic can be split by outcome. The
        # question it answers: do the FAILING episodes end with a wound-up
        # target the policy spent its steps unwinding? Diagnosis instrument.
        self._recent_target_lag: collections.deque = collections.deque(maxlen=window)
        # Fallback only; the real path is the run's own log dir (see
        # _write_metrics), which train.py assigns to cfg.log_dir.
        self._metrics_path = pathlib.Path("logs") / "demo_metrics.json"
        # Also scaled: at 4096 envs a fixed 50 would rewrite the file several
        # times per iteration, which is pure I/O for no extra information.
        self._metrics_dump_every = max(50, window // 8)

    def _setup_scene(self):
        # Fixture into env_0 before cloning, so it replicates into every env.
        # KINEMATIC rigid object since D-034 (was a static collider, D-018):
        # the pose randomisation needs a runtime pose to write, and kinematic
        # keeps the table immovable by contact forces while PhysX owns it.
        fixture_spawn = self.cfg.fixture_cfg.spawn
        fixture_spawn.usd_path = resolve_fixture_usd_path()
        self._fixture_usd_path = fixture_spawn.usd_path
        # Static tilt (D-036) and static yaw (D-038) enter HERE, as the spawn
        # orientation about the asset origin (= pocket centre): the kinematic
        # fixture keeps its spawn pose because the D-034 reset write is
        # inactive whenever a deterministic angle is set (enforced in
        # __init__). Composition R_z(yaw) @ R_y(tilt), matching __init__'s
        # _fixture_quat -- the startup report compares the prim quaternion
        # against that buffer, so a divergence here cannot stay silent.
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
        # The generated table is authored as bare colliders without a
        # RigidBodyAPI (static-collider convention, D-033), and the spawner's
        # rigid_props path is a silent no-op on such a prim:
        # modify_rigid_body_properties returns False when the API is absent
        # (isaaclab v2.3.2, sim/schemas/schemas.py) rather than applying it.
        # define_rigid_body_properties APPLIES the API first, so it is the
        # one call that actually turns the table into a (kinematic) rigid
        # body. Without it, RigidObject below finds no rigid body and env
        # creation fails on the training machine. No MassAPI is added: PhysX
        # ignores mass on kinematic bodies.
        define_rigid_body_properties(
            "/World/envs/env_0/Fixture",
            sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        )
        # Spawn already happened above (so the RigidBodyAPI edit lands before
        # cloning); hand the asset class a spawn-less copy or it would try to
        # spawn a second time.
        self._fixture = RigidObject(self.cfg.fixture_cfg.replace(spawn=None))

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
        # Registered so the scene manages its buffers/resets; the pose writes
        # in _reset_idx go through this handle.
        self.scene.rigid_objects["fixture"] = self._fixture
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        clamped = actions.clamp(-1.0, 1.0)
        self._joint_targets = self._joint_targets + self.cfg.action_scale * clamped
        limits = self.robot.data.soft_joint_pos_limits
        self._joint_targets = torch.clamp(self._joint_targets, limits[..., 0], limits[..., 1])
        # MEASUREMENT ONLY (D-037 diagnosis, 2026-08-16), nothing is changed by
        # it. The targets are an integrator clamped against the JOINT LIMITS
        # and not against the joint's actual position, so while the peg is
        # pressed against the plate and the arm cannot follow, the target keeps
        # accumulating. Every radian of accumulated lag has to be unwound at
        # action_scale per step before the arm moves again, which would look
        # exactly like the observed "presses on the table and barely searches".
        # This records the worst lag each episode reached; the trailing-window
        # statistics go to demo_metrics.json. Read against action_scale
        # (0.02 rad): a lag of 0.20 rad means ten dead steps.
        lag = (self._joint_targets - self.robot.data.joint_pos).abs().amax(dim=-1)
        self._max_target_lag = torch.maximum(self._max_target_lag, lag)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets)

    def _peg_geometry(self):
        """Per-env peg-tip position (env-local), insertion depth, pocket gate,
        out-of-arena flag, tool-axis alignment and yaw.

        Shared by the observation, reward and done computations so the tip
        math (peg pose -> tool axes -> tip offset -> env-local) is written
        once. Returns (tip_local (N,3), depth (N,), gate (N,) bool,
        below_plate (N,) bool, alignment (N,), phi (N,), cos4phi (N,),
        sin4phi (N,), tip_rel (N,3)). tip_rel is the tip offset from the
        pocket entrance IN THE POCKET FRAME (D-036); identical to
        tip_local - entrance while the pocket is untilted, and the tensor
        the observation's channels 12:15 consume.

        Deliberately called TWICE per control step -- once in _get_dones()
        (cached forward for _get_rewards()) and once in _get_observations().
        Merging the two into one cached call would be a bug, not a saving:
        DirectRLEnv runs _reset_idx() between them, so for envs that just
        finished, the observation must reflect the freshly reset state, not
        the terminal pose the dones/rewards were computed from. The reward
        path on the other hand must see the PRE-reset geometry, or terminal
        rewards would be priced at the home pose. Two calls at two times is
        the correct semantics; the cost is one batched matmul per step.

        **Lateral containment is a four-corner test**, not a centre-distance
        one: the four corners of the tip cross-section,
        ``c_i = tip +- (side/2) x_peg +- (side/2) y_peg``, must each lie
        within the pocket's half-side of the pocket centre on both XY axes
        (Chebyshev distance). Offset, yaw and tilt all move corners, so one
        test subsumes all three: at 1.0 mm of clearance per axis a centred
        peg passes only within +-3.96 deg of a C4 orientation, and a 45 deg
        peg (diagonal 42.4 mm > 32 mm opening) has all four corners outside
        at any offset. A centre-distance test would have called that
        geometrically impossible pose "inside".

        The gate is additionally a **pocket-volume** test. ``depth`` is
        ``plate_top_z - tip_z`` and keeps growing below the plate, where it
        passes the success threshold while the tip is nowhere near the
        pocket; requiring ``depth <= pocket_depth`` closes the space under
        the table that a training run was observed exploiting on 2026-07-27.
        The 1 mm tolerance on the pocket bottom covers float32 rounding and
        the small penetration PhysX allows at the stop; it is far below the
        ~55 mm depth read under the plate, so it does not reopen that hole.

        ``alignment`` is the peg's own tool axis dotted with world-down, i.e.
        +1 pointing straight into the plate. The home pose reads 0.9998. It
        stays in the gate as a cheap redundant guard even though the corner
        test already catches tilt (a tilted peg's deep corner leaves the
        square before alignment drops far).

        ``phi`` is the yaw of the peg's x-axis about world-z, folded into
        (-45 deg, 45 deg]: the pocket is C4-symmetric, so orientations 90 deg
        apart are the same task state. (cos 4 phi, sin 4 phi) are computed
        from the raw angle -- the encoding is fold-invariant by construction
        since cos(4(phi - k*pi/2)) = cos(4 phi).
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

        # Pocket-frame transform (D-036). Without tilt the expressions below
        # are the exact pre-D-036 code (entrance z IS plate_top_z, so
        # depth = plate_top_z - tip_z = -rel_z); with tilt, tip offset and
        # peg axes are rotated by R^T into the pocket frame first, so gate,
        # depth, alignment and phi all measure against the tilted pocket.
        # The reward's approach term takes tip_local and entrance unchanged:
        # it only uses their distance, which is rotation-invariant.
        if self._tilt_rot is None:
            tip_rel = tip_local - self._entrance_pos
            depth = self.cfg.plate_top_z - tip_local[:, 2]
            axes = peg_axes
            below_plate = tip_local[:, 2] < self.cfg.plate_bottom_z
        else:
            # Per-env matrices since D-037. einsum "ni,nij->nj" is the batched
            # form of the previous v @ R, i.e. component j = sum_i v_i R_ij =
            # (R^T v)_j: the same pocket-frame projection, one rotation per
            # env instead of one shared matrix.
            tip_rel = torch.einsum("ni,nij->nj", tip_local - self._entrance_pos, self._tilt_rot)
            depth = -tip_rel[:, 2]
            axes = torch.bmm(self._tilt_rot_t, peg_axes)
            below_plate = tip_rel[:, 2] < (self.cfg.plate_bottom_z - self.cfg.plate_top_z)

        # Columns of axes are the body axes in the pocket frame (world frame
        # while untilted).
        x_axis = axes[:, :, 0]
        y_axis = axes[:, :, 1]

        # Yaw of the peg x-axis about the pocket z axis (world z while
        # untilted). The mesh is authored axis-aligned to the flange x/y
        # (author_peg_ur10e.py), so this IS the orientation of the square's
        # sides, not of its diagonal.
        phi_raw = torch.atan2(x_axis[:, 1], x_axis[:, 0])
        phi = phi_raw - torch.round(phi_raw / (math.pi / 2.0)) * (math.pi / 2.0)
        cos4phi = torch.cos(4.0 * phi_raw)
        sin4phi = torch.sin(4.0 * phi_raw)

        # Four corners of the tip cross-section, projected to the pocket XY
        # plane. ex/ey are the pocket-XY footprints of the half-side vectors;
        # under peg-vs-pocket tilt they shrink (cos of the tilt), which errs
        # on the permissive side by at most 0.06 mm inside the 8.1 deg
        # alignment gate -- negligible against 1.0 mm of clearance.
        half = self.cfg.held_asset.side / 2.0
        ex = x_axis[:, :2] * half
        ey = y_axis[:, :2] * half
        # Per-env entrance since D-034; pocket-frame under tilt since D-036.
        # Since D-037 the rotation matrices are per-env and built from the
        # full fixture quaternion (yaw included), so a yawed pocket (D-038)
        # rides the same transform: phi and the corner test below are
        # evaluated in the yawed pocket frame, no separate yaw handling.
        tip_xy = tip_rel[:, :2]
        half_pocket = self.cfg.fixed_asset.side / 2.0
        corners = torch.stack(
            (tip_xy + ex + ey, tip_xy + ex - ey, tip_xy - ex + ey, tip_xy - ex - ey),
            dim=1,
        )  # (N, 4, 2)
        # Chebyshev: every corner inside the square on BOTH axes.
        corners_inside = corners.abs().amax(dim=(1, 2)) < half_pocket

        # Peg tool axis (its own local +z) dotted with pocket-down (world-down
        # while untilted), so [:, 2, 2] is the pocket-z component of the tool
        # axis and the dot with (0, 0, -1) is its negative. Under tilt this
        # demands alignment with the POCKET axis, which is the physically
        # insertable direction.
        alignment = -axes[:, 2, 2]
        gate = (
            corners_inside
            & (depth <= self.cfg.fixed_asset.depth + 0.001)
            & (alignment >= self.cfg.gate_min_alignment)
        )
        return tip_local, depth, gate, below_plate, alignment, phi, cos4phi, sin4phi, tip_rel

    def _get_observations(self) -> dict:
        self._obs_calls += 1

        joint_pos = self.robot.data.joint_pos
        joint_vel_fd = (joint_pos - self._prev_joint_pos) / self._step_dt
        self._prev_joint_pos = joint_pos.clone()

        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        if self._peg_body_idx is not None:
            geo = self._peg_geometry()
            # Pocket-frame since D-036 (geo[8]); while untilted this is the
            # same tensor expression as the previous tip_local - entrance.
            tip_rel_entrance = geo[8]
            cos4phi = geo[6].unsqueeze(-1)
            sin4phi = geo[7].unsqueeze(-1)
        else:
            tip_rel_entrance = torch.zeros(self.num_envs, 3, device=self.device)
            # (1, 0) is the aligned-yaw reading, matching the home pose.
            cos4phi = torch.ones(self.num_envs, 1, device=self.device)
            sin4phi = torch.zeros(self.num_envs, 1, device=self.device)

        # Channels 21:25 (D-037): the pocket's own orientation, appended last
        # so no existing index moves. Every channel before it is either robot
        # state or measured RELATIVE to the pocket, so without this block a
        # per-episode tilt is invisible to the policy -- two episodes tilted
        # differently would produce identical observations and demand
        # different joint motions. The buffer is already sign-canonicalised
        # (w >= 0) where it is written, so the encoding is unique.
        obs = torch.cat(
            (joint_pos, joint_vel_fd, tip_rel_entrance, ee_quat_w, cos4phi, sin4phi, self._fixture_quat), dim=-1
        )

        if self._obs_calls in self._report_at_obs_calls:
            self._print_startup_report(joint_vel_fd)

        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Runs BEFORE _get_rewards() in DirectRLEnv's step loop, so the
        # per-step peg geometry and success flag are computed here and cached
        # for _get_rewards() to reuse, not the other way round.
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

        # Home pose plus per-joint joint-space noise (no IK needed, unlike a
        # cartesian randomization would require). Five joints get +-0.01 rad
        # -- enough that the policy cannot memorize one exact trajectory, not
        # enough to fight the fixed 13.4 mm gravity droop (D-026) on top of
        # it. wrist_3, the tool-axis yaw joint, gets +-45 deg: a full C4
        # fundamental domain. Without that every start would already lie
        # inside the +-3.96 deg free-yaw window, the policy would never need
        # to turn the peg, and a 100 % success rate would be a non-result.
        # wrist_3 is a pure yaw actuator at the home pose, so the wide noise
        # moves phi and nothing else -- the tip stays put.
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

        # Fixture-pose randomisation (D-034): sample a new pocket pose for the
        # envs being reset and teleport the kinematic fixture to it. Guarded
        # by `ready` (buffers exist) AND by the ranges being > 0 -- at 0 no
        # write ever happens and the env is bit-identical to pre-D-034, which
        # is the commissioning regression case. The entrance buffer and the
        # prim pose are written from the SAME tensor, so they can only
        # disagree if the sim write itself fails; the startup report prints
        # both for the report envs, which is the identity test.
        if ready and (
            self.cfg.fixture_pos_noise_xy > 0.0
            or self.cfg.fixture_yaw_noise_rad > 0.0
            or self.cfg.fixture_tilt_noise_rad > 0.0
        ):
            idx = torch.as_tensor(list(env_ids), device=self.device) if not torch.is_tensor(env_ids) else env_ids
            n = int(idx.numel())
            entrance = self._entrance_base.unsqueeze(0).repeat(n, 1)
            entrance[:, :2] += (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * self.cfg.fixture_pos_noise_xy
            # Per-episode pocket yaw (D-038), uniform in +-range; composed
            # with the tilt below as quat_yaw * quat_tilt.
            yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.fixture_yaw_noise_rad
            self._entrance_pos[idx] = entrance
            self._fixture_yaw[idx] = yaw
            quat_yaw = torch.zeros(n, 4, device=self.device)
            quat_yaw[:, 0] = torch.cos(yaw * 0.5)
            quat_yaw[:, 3] = torch.sin(yaw * 0.5)

            # Per-episode tilt (D-037): magnitude uniform in [0, max], tilt
            # DIRECTION uniform over 360 deg, i.e. the tilt axis is a uniformly
            # random horizontal axis (cos a, sin a, 0) rather than env y alone.
            # Magnitude uniform in the ANGLE (not uniform over the spherical
            # cap, which would over-sample large angles): the per-bin success
            # counts want the angle bins populated evenly, since they are the
            # readout that says whether the large angles are learned.
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
            # q and -q are one rotation; the observation must encode each
            # orientation exactly once, so the sign is pinned to w >= 0. Within
            # the 15 deg cap w >= 0.991, so this never actually flips -- it is
            # a guarantee for the policy's input, not a live correction.
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
        # reached inside the pocket, which says more than the final-step value.
        successes = self._last_success[idx]
        depths = self._max_depth[idx]
        n = int(idx.numel())

        self._ep_count += n
        self._ep_success_count += int(successes.sum().item())
        self._ep_depth_sum += float(depths.sum().item())
        self._recent_successes.extend(successes.tolist())
        self._recent_depths.extend(depths.tolist())
        # episode_length_buf still holds the finished episodes' step counts:
        # this runs before super()._reset_idx() zeroes them.
        self._recent_success_steps.extend(self.episode_length_buf[idx][successes].tolist())
        # Same ordering dependency, and it is load-bearing (D-037): this method
        # is called at the TOP of _reset_idx, before the block that samples the
        # next episode's pocket orientation, so _fixture_quat still holds the
        # orientation the episode being logged actually ran under. Moving
        # either call would silently pair every success with the NEXT episode's
        # angle and make the per-bin rates meaningless.
        # Tilt = polar angle of the pocket z axis (acos of rotation-matrix
        # r22), NOT the quaternion's total rotation angle: with pocket yaw
        # (D-038) in the quaternion, 2*acos(w) would book a purely yawed
        # pocket as "tilted" and poison the per-bin rates.
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

    def _success_by_tilt_bin(self) -> list[dict]:
        """Trailing-window success rate resolved by pocket tilt magnitude.

        Returns one entry per bin with its bounds in degrees, the number of
        episodes that fell in it and their success rate. Empty bins are kept
        (rate None) rather than dropped: a bin that never filled is itself the
        finding -- it says the sampling never produced that angle -- and a list
        whose length changes between runs is harder to compare.

        Read against the overall rate: the two agree only when the policy is
        equally good at every angle, which is precisely the claim the training
        has to earn.
        """
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
        """Joint-target lag over the trailing window, split by episode outcome.

        The joint targets are an integrator clamped only against the joint
        limits (_pre_physics_step), so an arm held up by contact accumulates a
        target it cannot reach. This reports the worst lag per episode as a
        mean over the failed episodes, a mean over the successful ones, and the
        overall maximum, together with the same numbers expressed in control
        steps (lag / action_scale) -- the form the question is actually asked
        in: "how many steps does the policy spend commanding an arm that cannot
        move?"

        Both means are reported because only their RATIO discriminates. A large
        lag everywhere is the controller's normal following error; a large lag
        only on failures is the wind-up mechanism.

        The percentiles and maxima exist to make a CLAMP threshold derivable
        instead of guessed. Any anti-wind-up band has to sit above the lag the
        SUCCESSFUL episodes actually use -- part of that lag is the contact
        force the policy inserts with, so clamping below it would break the
        behaviour that already works -- and far enough below the failure lag to
        cut the runaway. Means alone cannot say where that gap is; p95 and max
        of the successful episodes can.
        """
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
            # Nearest-rank p95 on the sorted list; exact for the sample, and
            # with a 2000-episode window the rank is unambiguous.
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
            "peg_side_m": self.cfg.held_asset.side,
            # The curriculum variable since D-033. Written even though the
            # clearance below is derived from it: compare_runs.py reconstructs
            # the replay command from this file alone, and a rung it can only
            # infer is a rung it can get wrong.
            "pocket_side_m": self.cfg.fixed_asset.side,
            "side_clearance_mm": proxytask_tasks_cfg.SIDE_CLEARANCE * 1000.0,
            "yaw_window_deg": math.degrees(proxytask_tasks_cfg.YAW_WINDOW_RAD),
            "reset_joint_noise_rad": self.cfg.reset_joint_noise,
            "reset_yaw_noise_rad": self.cfg.reset_yaw_noise,
            # D-034: the fixture-pose randomisation this run trained/replayed
            # under. Recorded for the same reason as pocket_side_m -- a run
            # whose randomisation can only be inferred is one that gets
            # compared wrongly.
            "fixture_pos_noise_xy_m": self.cfg.fixture_pos_noise_xy,
            "fixture_yaw_noise_rad": self.cfg.fixture_yaw_noise_rad,
            # D-036: static pocket tilt this run measured under, same
            # rationale -- a tilt that can only be inferred from the folder
            # name is one that gets compared wrongly.
            "fixture_tilt_rad": self.cfg.fixture_tilt_rad,
            "fixture_yaw_rad": self.cfg.fixture_yaw_rad,
            # D-037: the per-episode tilt range this run trained under, and the
            # success rate resolved by tilt magnitude. The overall rate above
            # is an average over the whole range and cannot show whether the
            # large angles were learned; this list can, and it is the readout
            # the ladder's next rung is decided from.
            "fixture_tilt_noise_rad": self.cfg.fixture_tilt_noise_rad,
            "success_by_tilt_bin": self._success_by_tilt_bin(),
            # Diagnosis instrument (2026-08-16): the worst gap between the
            # integrated joint TARGET and the joint's actual position, per
            # episode, over the trailing window. Read every value against
            # action_scale below: the lag divided by action_scale is the number
            # of control steps the policy must spend unwinding before the arm
            # responds at all. If failed_mean is several times ok_mean, the
            # failures are dominated by pressing into the plate rather than by
            # aiming at the wrong place.
            "action_scale_rad": self.cfg.action_scale,
            "joint_target_lag_rad": self._target_lag_stats(),
            # Mean steps of the successful episodes in the trailing window;
            # None until one succeeded. Read against the 240-step cap.
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
        """Compare the authored asset's *measured* geometry against the cfg constants.

        Ported from increment 3 (commit 4908adb), which added it after a stale
        copy on the training machine re-authored a peg the configuration had
        already moved away from. The failure mode is structural, not careless:
        everything in the environment reads the same constants, so a
        configuration that no longer matches the USD on disk is perfectly
        self-consistent and silent -- the report's own peg lines print
        ``cfg.held_asset.side``, which is the number that was wrong.

        It matters doubly on this branch, which changes the peg's very
        cross-section, and which shares a training-machine folder with the
        earlier round pegs.

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

        # The square cross-section spans the two equal axes of the local box;
        # length runs along local z. BOTH cross axes are compared: a bbox
        # reading of ~0.042 m against a configured side of 0.030 m would mean
        # the mesh is rotated 45 deg relative to its own frame (the diagonal,
        # not the side, is axis-aligned) -- a real asset defect that must be
        # REPORTED here, never absorbed by the check. Compared at 0.1 mm.
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
        """The same check for the fixture, which is what varies now (D-033).

        Before D-033 the pocket was a compile-time constant and the resolver
        accepted exactly one filename, so the asset could not disagree with the
        configuration without someone overwriting a file. Now the pocket IS the
        curriculum variable: there is one table per rung, the name carries the
        size, and ``PROXYTASK_TISCH_USD`` can point anywhere at all. A table
        whose pocket is not the configured one would train a rung the metrics,
        the run tag and the yaw window all describe wrongly -- silently, since
        every number in this report is read from the cfg, not from the asset.

        ``author_tisch_square.py`` writes the geometry it measured off the
        authored stage into a sidecar, exactly as the peg script does; that
        sidecar is the only artefact describing the asset rather than the
        intent, so it is what this compares against.
        """
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

        # Computed from cfg rather than read off the base class: step_dt was not
        # covered by probe_assets.py.
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
        # Resting finite-differenced joint speed. This is the demo sprint's
        # substitute for the solver-flag experiment: it must read ~0.000 at
        # the early/late reports below (velocity derived from positions, not
        # from the raw solver channel), unlike the raw joint_vel channel's
        # measured ~0.05 rad/s phantom (D-030 addendum 2).
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
            # Identity test for the D-034 pose write: the entrance buffer and
            # the fixture prim are written from the same tensor, so a mismatch
            # here means the sim write failed and every entrance-relative
            # number below is void.
            fixture_local = self._fixture.data.root_pos_w[e] - origins[e]
            pose_err = torch.linalg.norm(fixture_local - self._entrance_pos[e])
            print(f"-- env {e} pose check --")
            print(f"  fixture prim (env-local): {fixture_local.tolist()}")
            print(f"  entrance buffer:          {self._entrance_pos[e].tolist()}, yaw {self._fixture_yaw[e].item():+.4f} rad")
            print(f"  |prim - buffer| = {pose_err.item():.2e} m  (expect ~0; MISMATCH = pose write broken)")
            # Orientation identity test for the D-036 tilt. The tilt enters ONLY
            # as the spawn orientation, and the position check above cannot see
            # it: a rotation about the asset origin leaves root_pos_w untouched,
            # so pose_err reads ~0 whether the tilt landed or silently failed.
            # Without this read-back a no-tilt run would print "ACTIVE" above
            # and then produce a plausible success rate measured against a
            # pocket frame that does not physically exist -- the same class of
            # false verdict as a control that cannot react.
            fixture_quat = self._fixture.data.root_quat_w[e]
            qw, qx, qy, qz = (float(v) for v in fixture_quat.tolist())
            if qw < 0.0:  # q and -q are one rotation; take the near-identity sign
                qw, qx, qy, qz = -qw, -qx, -qy, -qz
            # Since D-037 the commanded orientation is a per-env buffer and the
            # tilt axis is arbitrary, so the check is quaternion against
            # quaternion rather than one angle about y. This is strictly
            # stronger than the D-036 form: it catches a wrong axis, a dropped
            # yaw and a wrong magnitude alike, and _fixture_quat is the very
            # tensor the pocket-frame transform and observation channels 21:25
            # are built from -- if the prim disagrees with it, every
            # pocket-frame number below and every observation is void.
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
                # ZY read-back: the spawn orientation is R_z(yaw) @ R_y(tilt),
                # so yaw = atan2(r10, r00) and tilt = -asin(r20). Both stay
                # valid when the yaw and tilt probes are combined -- the pure-y
                # D-036 line below assumes yaw 0 and is skipped instead.
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
                # Deterministic probe: keep the exact D-036 line so that a
                # post-training curve is literally comparable with the logs the
                # 100 % / 50 % baseline was read from.
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
                # POCKET FRAME (D-036), the same transform _peg_geometry
                # applies. Without it this block reports a world-frame gate,
                # depth, alignment and corner test while the policy, the
                # reward and demo_metrics.json all consume the pocket-frame
                # ones: two numbers for one quantity, and the printed pair is
                # then the one the reader trusts. At tilt 0 the branch is the
                # pre-D-036 expression.
                if self._tilt_rot is None:
                    tip_rel_e = tip_local - self._entrance_pos[e]
                    axes_p = peg_axes
                    depth = cfg.plate_top_z - tip_local[2]
                else:
                    tip_rel_e = self._tilt_rot_t[e] @ (tip_local - self._entrance_pos[e])
                    axes_p = self._tilt_rot_t[e] @ peg_axes
                    depth = -tip_rel_e[2]
                along_tool = torch.dot(tip_w - ee_pos_w, axes[:, 2]).item()
                # Home-pose expectations scale with the tilt. The home tip sits
                # `home_tip_height` above the plate ON the tilt axis (x = 0),
                # so in the pocket frame it moves to x = -h*sin(tilt),
                # z = h*cos(tilt): the depth shrinks by cos and the tip gains a
                # pocket-xy offset of h*sin. Printing the untilted expectation
                # under tilt would read as a fault where there is none.
                # Per-env angle since D-037 (the tilt is no longer one number
                # for the whole run), taken from the commanded quaternion of
                # THIS env, which is what its pocket frame is built from.
                tilt_c = math.cos(commanded_mag)
                tilt_s = abs(math.sin(commanded_mag))
                home_tip_height = cfg.home_standoff_z - cfg.peg_length
                # Same math as _peg_geometry, single-env form: yaw of the peg
                # x-axis about the pocket z axis, folded to (-45, 45] deg, and
                # the four tip corners in the pocket xy plane.
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
            # NOT transformed: this test compares world z against a constant
            # plate_top_z and assumes the plate centred at env-local (0, 0). It
            # is a robot-vs-table clearance sanity check, not a task
            # measurement, and it is already approximate under --fixture-offset
            # (the plate moves, the test does not). Under tilt the plate top is
            # no longer a constant world z -- it falls by x*tan(tilt) -- so the
            # notes below are indicative only. Labelled rather than silently
            # wrong; the load-bearing numbers are the pocket-frame ones above.
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
