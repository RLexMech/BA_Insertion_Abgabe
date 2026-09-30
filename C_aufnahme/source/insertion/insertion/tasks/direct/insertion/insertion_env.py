# NOTE (Phase B copy, 2026-08-18): copied from the verified proxy task
# (square peg, old repo). Geometry constants and square-peg-specific
# derivations (C4 yaw window, per-axis clearance, analytic inertias) are
# PROXY-SPECIFIC and get replaced in later phases from measured real-task
# values. Do not treat any number in this file as a real-task value yet.
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import csv
import dataclasses
import json
import math
import pathlib
import time
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
    OperationalSpaceController,
    OperationalSpaceControllerCfg,
)
from isaaclab.sim.schemas import define_rigid_body_properties
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_inv, subtract_frame_transforms

from . import (
    autodr,
    insertion_math,
    insertion_paths,
    insertion_sdf,
    insertion_tasks_cfg,
    obs_noise,
    scripted_policy,
)
from .insertion_env_cfg import (
    PROXY_METRICS_KEYS,
    RL_PLACEHOLDERS,
    InsertionEnvCfg,
    resolve_control_mode,
    resolve_fixture_usd_path,
    resolve_obs_layout,
    resolve_proxy_task_values,
    resolve_start_lateral_offset,
    validate_rl_config,
)
from .insertion_tasks_cfg import (
    resolve_workcell_block_usd_path,
    resolve_workcell_tables_usd_path,
)

# Marker for the manual two-machine copy. If the startup report does not show
# this string, the copy on the training machine predates the current code.
#
# SQUARE-PEG PIVOT (branch square-peg-insertion, 2026-07-28, D-029): cut from
# the demo sprint (a7c7ebb), keeping its joint-delta actions and dense-reward
# skeleton. The peg is a 30 x 30 x 50 mm square prism, the hole a
# 32 x 32 x 35 mm blind pocket in the re-imported table. New here: the
# four-corner containment gate, a 25-dim observation (D-029's (cos 4phi,
# sin 4phi) of the tool-axis yaw at 19:21, D-037's pocket quaternion at
# 21:25), and a C4-invariant yaw reward term. The reset yaw randomisation
# this line used to claim (+-45 deg on wrist_3) is NOT active: the cfg
# defaults are reset_yaw_noise = 0.0 and reset_joint_noise = 0.0
# (insertion_env_cfg.py), and a run opts in per run. UNVERIFIED until run on
# the training machine.
CODE_MARKER = "insertion-osc-2026-09-03a"

# THE EPISODE RECORD (user scope, 2026-09-15): one row per finished episode in
# `<run folder>/episodes.csv`, for the thesis's interpenetration analysis.
# Definitions, units and the measurement set-up are written once beside it,
# `episodes_meta.json` (`_episode_log_meta`). Small on purpose: the user cut
# time series, extra thresholds and the raw draws. check_env_wiring.py pins
# this tuple; scripts/check_episode_log.py reads a run folder against it.
EPISODE_LOG_COLUMNS = (
    "run", "seed", "iteration", "env_id", "episode", "outcome", "steps",
    "ip_max_mm", "ip_steps_ge_t1", "force_max_filtered_n", "force_max_raw_n",
    "bounds_version", "dr_phase",
)


class InsertionEnv(DirectRLEnv):
    """UR5e insertion env: task-space pose-delta actions under Isaac Lab's
    OperationalSpaceController (inbox entry "Audit 2026-09-03 (a)";
    ``cfg.control_mode``, the joint-delta chain of D-108 kept as the PD
    half of that entry's measurement), the REAL task's observation, reward
    and termination (M2.4b step 3, 2026-08-30).

    Since M2.4b step 3 the reward is D-109's (three-kernel SDF distance +
    two display bonuses, SAPU-scaled, abort payment) and the termination is
    D-113's. Both were revised on 2026-09-01 -- inbox entry
    "Reward-Ueberarbeitung" 2026-09-01 (p1-konzept-messung): the reward gains a
    time penalty (-1/T) and an action-rate penalty, and SUCCESS now terminates
    the episode with its remaining return paid out, so ``terminated`` has two
    sources (force abort, success) beside the timeout in ``truncated``.

    Both call sites live in ``_get_dones`` / ``_get_rewards`` and the math
    lives in ``insertion_math`` / ``insertion_sdf``. The proxy's seven-term
    reward and its ``success | below_plate`` terminations are GONE (the
    success exit that came back on 2026-09-01 is not the proxy's: it fires on
    the D-052 band with the SAPU filter, and it is paid out rather than
    latched). What remains of the
    square-peg era is the ``_peg_geometry`` INSTRUMENT (seat_probe and the
    startup report read it; the task path does not) and the paragraph below,
    kept for the record.

    Cut from the demo-sprint env (round Ø25 peg), keeping its verified
    skeleton: joint-position-delta actions (NVIDIA UR10-Reach pattern, no
    OSC), finite-differenced joint velocities (the raw solver ``joint_vel``
    channel has a measured ~0.05 rad/s phantom at rest -- D-030 addendum 2),
    the dense staged reward, and the running metrics dumped to
    ``demo_metrics.json``. The square pivot (D-029) changes:
      - the gate: a four-corner containment test of the tip cross-section
        against the 32 mm pocket, subsuming offset, yaw and tilt in one test,
      - the observation: [REPLACED 2026-08-28, M2.4 -- see
        ``_get_observations``. The channels are now the REAL task's (D-107),
        built from ``insertion_math`` against the ``OBS_SLICES`` table:
        part-anchored 12:15, and (cos phi, sin phi) at 19:21 instead of the
        C4 encoding described below. The force block 25:28 is still missing,
        blocked on the scene stream's force-sensor link. The original
        sentence follows for the record.] 25-dim. Channels 19:21 are
        (cos 4 phi, sin 4 phi) of the peg yaw -- C4-invariant by
        construction, so the four insertable orientations are one point in
        observation space instead of four separate lessons -- and channels
        21:25 are the pocket's own quaternion (D-037), without which a
        per-episode pocket orientation would be invisible to the policy,
      - the reward: a seventh term charging (1 - cos 4 phi) / 2,
      - the reset: OPTIONAL uniform yaw noise on wrist_3 and joint noise on
        the other five. Both default to 0.0 (`reset_yaw_noise`,
        `reset_joint_noise`), so a plain run starts inside the +-3.96 deg
        free-yaw window; a run that wants the yaw lesson opts in per run.
        This paragraph claimed a fixed +-45 deg until 2026-08-31 -- a proxy
        leftover that had already produced one wrong handoff entry.
    """

    cfg: InsertionEnvCfg

    def __init__(self, cfg: InsertionEnvCfg, render_mode: str | None = None, **kwargs):
        # THE CONTROL MODE IS RESOLVED FIRST, before the base class reads
        # decimation / episode seconds / render interval from the cfg: under
        # OSC the policy rate is the 15 Hz placeholder (decimation 8) and
        # the 256-step episode keeps its step count. One home for that
        # derivation: insertion_env_cfg.resolve_control_mode.
        self._control_mode = resolve_control_mode(cfg)
        # THE OBSERVATION LAYOUT, SAME CONTRACT (D-188): `observation_space`
        # is 28 under `force` and 31 under `wrench`, and DirectRLEnv builds
        # its spaces from that field inside its own __init__. One home for
        # the width: insertion_math.obs_dim via insertion_env_cfg.resolve_obs_layout.
        self._obs_mode = resolve_obs_layout(cfg)
        # THE OBSERVATION NOISE, SAME CONTRACT, SAME REASON (D-182).
        # `DirectRLEnv` builds the model from this field inside its own
        # __init__ (direct_rl_env.py:210-213), so an assignment placed after
        # the super() call below is one the base class already read as None:
        # the run would train on the clean observation while every log line
        # still said the sigmas were set. One home for the both-zero rule and
        # for the cfg-to-model step: obs_noise.resolve_obs_noise_model.
        cfg.observation_noise_model = obs_noise.resolve_obs_noise_model(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        # THE ONE PLACE THE RL-CONFIG GUARD IS CALLED (M2.3, 2026-08-28).
        #
        # Four scripts build an env -- train, play, zero_agent, random_agent --
        # and every one of them reaches this constructor, so this is the only
        # spot where the guard covers all of them and where a script written
        # later cannot skip it by forgetting a line.
        #
        # It sits behind the switch because of the ring the cfg names: F_max
        # is measured with a built env, so an env that always demanded F_max
        # could never be the instrument that measures it. With the switch OFF
        # this env computes no real reward, no force abort and no curriculum,
        # so it reads none of the open values and has nothing to validate.
        #
        # AFTER super().__init__, not before: the base class normalises the
        # cfg (and may reset from inside it), and the guard reads the cfg the
        # env will actually run with, not the one that was handed in.
        self._rl_terms_enabled = bool(getattr(self.cfg, "rl_terms_enabled", True))
        if self._rl_terms_enabled:
            validate_rl_config(self.cfg)

        # Contact friction, D-111 (2): u = CONTACT_FRICTION on every material
        # of the robot (the welded part is one of its links) and the kinematic
        # fixture. Factory pattern, ported from Isaac Lab 2.3.2
        # isaaclab_tasks/direct/factory/factory_utils.py::set_friction --
        # static and dynamic friction get the same value there too. Applied at
        # runtime because no USD in this repo authors a physics material. The
        # startup report reads the values BACK from PhysX, so a silent no-op
        # cannot pass as applied.
        # THE BUFFERS ARE KEPT (Phase 5 step B4). Under a friction boundary
        # every reset rewrites these same two tensors for the envs that just
        # reset, and `get_material_properties()` is a PhysX read-back of the
        # WHOLE view -- calling it per reset would add a full-view read to
        # every reset on top of the write. What that would cost is NOT
        # measured; `friction_write_ms` measures the cached path only.
        # `.clone()` because the view may hand back its own internal
        # tensor: an aliased buffer would let a later PhysX-side change appear
        # in our rows without any write of ours, and then the value we log and
        # the value we wrote could differ with nothing to show it.
        _mat_env_ids = torch.arange(self.scene.num_envs, device="cpu")
        self._friction_bufs: list = []
        for _asset in (self.robot, self._fixture):
            _mats = _asset.root_physx_view.get_material_properties().clone()
            _mats[..., 0] = insertion_tasks_cfg.CONTACT_FRICTION  # static
            _mats[..., 1] = insertion_tasks_cfg.CONTACT_FRICTION  # dynamic
            _asset.root_physx_view.set_material_properties(_mats, _mat_env_ids)
            self._friction_bufs.append((_asset, _mats))
        # Column 2 is restitution. This env never RANDOMISES it, but the push
        # in `_apply_friction` sends whole rows, so it does get REWRITTEN --
        # from this snapshot, at every reset of that env. Harmless while
        # nothing else writes restitution; the moment something does, it must
        # write into THIS buffer or be silently reverted. The earlier wording
        # here said "not ours to write", which was true of the row edit and
        # false of the push.
        #
        # ONE dtype for both buffers, refused rather than repaired: the host
        # copy of the drawn column is made ONCE, outside the asset loop, and
        # a second dtype there would silently cast or raise inside the very
        # block whose cost the plan reads.
        _dtypes = {_b.dtype for _, _b in self._friction_bufs}
        if len(_dtypes) != 1:
            raise ValueError(
                f"the robot and fixture material buffers have different dtypes {_dtypes}; "
                "_apply_friction casts the drawn friction once for both"
            )
        self._friction_buf_dtype = next(iter(_dtypes))

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

        # THE FORCE-SENSOR LINK (D-114, observation channels 25:28).
        #
        # ``force_sensor_body_name`` is MEASURED, not assumed: RT-59 read the
        # incoming joint wrench at ``tool_link`` against its own authored
        # weight and got ratio 1.000. See the cfg field for the whole probe.
        #
        # Two failure paths, kept apart on purpose:
        #
        # * NO WELDED TOOL AT ALL -- ``_peg_body_idx`` is already None, the
        #   startup report already prints its MISSING line, and every
        #   pose-derived observation block is already degraded to a constant.
        #   The force block joins them rather than crashing a run that is
        #   deliberately allowed to start without the tool.
        # * THE TOOL IS THERE BUT THE NAMED LINK IS NOT. That is a config or
        #   asset mismatch and it does NOT degrade: a zero in place of a
        #   missing force signal is exactly the silent defect the pending
        #   guard exists to prevent, so it raises and names the body list.
        if self._peg_body_idx is None:
            self._force_body_idx = None
        else:
            try:
                self._force_body_idx = self.robot.find_bodies(self.cfg.force_sensor_body_name)[0][0]
            except (IndexError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"force-sensor body '{self.cfg.force_sensor_body_name}' is not among "
                    f"{self.robot.data.body_names}. The wrench for observation channels "
                    "25:28 (D-114) has nowhere to be read; a zero in its place would be a "
                    "silent defect, so this is a hard stop."
                ) from exc

        # THE GRAVITY TARE (2026-08-31). The wrench carries the child link's
        # own weight; FORGE never sees it because Factory/FORGE spawn robot
        # and held asset with disable_gravity=True, while this repo keeps
        # gravity ON at the robot (supervisor, 2026-08-25). RT-59 measured the
        # gap: 8.0861 N at tool_link with no contact, ratio 1.000 against m*g.
        #
        # The tare needs the PARENT link's orientation, and Isaac Lab 2.3.2
        # exposes no parent-index API on ArticulationData. The chain is
        # serial and the parent is therefore the preceding body -- but that is
        # CHECKED here, not assumed: two independent cfg fields name the two
        # links, and a rename of either would silently tare against the wrong
        # frame. A wrong frame is invisible in the magnitude at rest, which is
        # exactly the defect that must not pass quietly.
        self._force_parent_idx = None
        self._hold_force_w = None
        if self._force_body_idx is not None and bool(self.cfg.force_gravity_tare):
            if self.cfg.tool_mass_kg is None:
                raise ValueError(
                    "force_gravity_tare is on but cfg.tool_mass_kg is None. The tare "
                    "subtracts mass * gravity and has no mass to use; guessing one would "
                    "put an invented number into the observation and the abort."
                )
            parent_idx = self._force_body_idx - 1
            if parent_idx != self._ee_body_idx:
                raise ValueError(
                    f"the gravity tare expects the force-sensor body "
                    f"'{self.cfg.force_sensor_body_name}' (index {self._force_body_idx}) to "
                    f"hang directly off the ee body '{self.cfg.ee_body_name}' (index "
                    f"{self._ee_body_idx}), because the parent link's quaternion is the only "
                    f"frame the tare can be expressed in and Isaac Lab 2.3.2 exposes no "
                    f"parent-index API. Body list: {self.robot.data.body_names}."
                )
            self._force_parent_idx = parent_idx
            # The force the parent must apply to hold the child up: -m * g.
            # Both halves keep their own home (cfg.tool_mass_kg, cfg.sim.gravity);
            # nothing is typed here, so a changed gravity moves the tare with it.
            g = self.cfg.sim.gravity
            self._hold_force_w = torch.tensor(
                [-float(self.cfg.tool_mass_kg) * float(c) for c in g], device=self.device
            )

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
        # THE GRASP OBSERVATION OFFSET (D-183), (N, 3) in the PART/TOOL frame.
        # Only x is ever written: the gripper can slip along the part's SHORT
        # axis and centres itself along the long one, so y and z keep the zero
        # this line gives them and the reset only fills column 0.
        #
        # A BELIEF ERROR, NOT A POSE. It is added to `_tip_offset_local` in
        # `_get_observations` alone; `_get_dones`, `_apply_osc` and every
        # other reader keep the nominal offset, so the part stays where PhysX
        # put it and only what the policy BELIEVES about the tip moves. The
        # physics gap that leaves -- mass, centre of mass and lever arms stay
        # nominal -- is named in D-183 and is the price of route (c).
        self._grasp_obs_off = torch.zeros(self.num_envs, 3, device=self.device)
        # The seated tool orientation in the pocket frame (RT-102, measured):
        # (1, 4) so seated_goal_pose broadcasts it against the per-env
        # fixture quaternion. One home for the number: insertion_tasks_cfg.
        self._seat_quat_local = torch.tensor(
            insertion_tasks_cfg.SEATED_TOOL_QUAT_LOCAL, device=self.device
        ).unsqueeze(0)

        # THE LATERAL START RADIUS in metres (D-170; a disk since D-178). Read
        # ONCE here, so the field's shape is checked in one place and every
        # reader below sees one float. `resolve_start_lateral_offset` owns the
        # rule. The name `_start_lat_half` is older than the disk: it holds
        # the RADIUS now, not a half width.
        self._start_lat_half = resolve_start_lateral_offset(cfg.start_lateral_offset)

        # THE BOUNDS PROVIDER (Phase 5, plan section 10). None in "off" mode,
        # which is every run before Phase 5: the static cfg fields below are
        # then the only source of randomisation and nothing here changes.
        #
        # The centres are NOT in autodr.py -- `DR_DIMS` ships them as None and
        # `bind_centres` refuses a missing one, so the width-0 value of every
        # quantity is decided here and has exactly one home. TWO of the five
        # read a config number (`start_height` = cfg.start_tip_above_entrance,
        # H_min under AutoDR, D-179; `friction` =
        # insertion_tasks_cfg.CONTACT_FRICTION, D-181); the other three are
        # 0.0 STRUCTURALLY, not by typing a config value: `lat_r` is a disk
        # radius whose centre is the pocket axis (D-178), and `yaw`/`tilt` are
        # measured from the upright pocket. The matching cfg fields are
        # refused non-zero just below, so the two readings cannot disagree.
        # Three of the five are ONE-SIDED, and for those `DimSpec.bind` turns
        # the centre given here into the lower edge (D-179 (2)).
        self._dr = None
        # THE START FLOOR (SBC step 0; cfg.start_floor_m owns the reasons).
        # Its own refusal block, apart from the AutoDR static-field refusal
        # below: the floor is a curriculum on the start height, not a static
        # randomisation, and it must be refused for its own reasons in BOTH
        # dr modes.
        self._start_floor_on = False
        _floor_spec = None
        if cfg.start_tip_above_entrance is not None:
            _floor = float(cfg.start_floor_m)
            _high = float(cfg.start_tip_above_entrance)
            if _floor > _high:
                raise ValueError(
                    f"start_floor_m ({_floor:+.4f} m) lies above start_tip_above_entrance "
                    f"({_high:+.4f} m); the floor is the LOWER edge of the band. Equal = "
                    "no floor."
                )
            if _floor < _high:
                if _floor <= -float(cfg.depth_min):
                    raise ValueError(
                        f"start_floor_m ({_floor:+.4f} m) is at or below the success band "
                        f"(-depth_min = {-float(cfg.depth_min):+.4f} m): a start inside the "
                        "band is a one-step success with the full lump."
                    )
                _low_f = cfg.start_tip_above_entrance_low
                if _low_f is not None and float(_low_f) != _high:
                    raise ValueError(
                        f"start_floor_m ({_floor:+.4f} m) together with a static band "
                        f"(start_tip_above_entrance_low {float(_low_f):+.4f} m): two lower "
                        "edges for one band. Use the floor OR the static band."
                    )
                if float(cfg.reset_joint_noise) != 0.0 or float(cfg.reset_yaw_noise) != 0.0:
                    raise ValueError(
                        "start_floor_m below start_tip_above_entrance (a start INSIDE the "
                        "pocket) together with reset_joint_noise / reset_yaw_noise: an "
                        "un-commanded orientation inside the pocket is a wall contact at "
                        "reset (RT-120). Set both to 0.0."
                    )
                if str(cfg.dr_mode) == "off":
                    _live_off = {
                        name: float(getattr(cfg, name))
                        for name in ("fixture_tilt_rad", "fixture_yaw_rad",
                                     "fixture_tilt_noise_rad", "fixture_yaw_noise_rad")
                    }
                    _live_off["start_lateral_offset"] = float(self._start_lat_half)
                    _live_off = {k: v for k, v in _live_off.items() if v != 0.0}
                    if _live_off:
                        # Under 'off' these fields are LIVE every reset; a
                        # tilted or offset start below the opening plane is
                        # the RT-120 wall contact. Under 'autodr' the
                        # provider holds them on the centre itself.
                        raise ValueError(
                            f"start_floor_m with dr_mode='off' together with {_live_off}: "
                            "an angle or a lateral offset while the start is inside the "
                            "pocket is a wall contact at reset (RT-120). Set them to 0.0; "
                            "under dr_mode='autodr' the boundaries open only after the "
                            "floor reaches H_min."
                        )
                self._start_floor_on = True
                _floor_spec = autodr.FloorSpec(_floor, _high, int(cfg.start_floor_steps))
        if str(cfg.dr_mode) == "off" and self._start_floor_on:
            # THE NO-DR BRANCH OF STEP 0: the same machine with ONE dimension.
            # The floor first, then the start-height ceiling alone; every
            # other quantity keeps its static cfg value through _live_bounds.
            # The centre expression is the SAME text as in the autodr block
            # below -- one home for H_min (D-179 (3)), one pin in
            # check_env_wiring.
            self._dr = autodr.AutoDR(
                autodr.bind_centres(
                    {"start_height": float(cfg.start_tip_above_entrance)},
                    dims=tuple(d for d in autodr.DR_DIMS if d.name == "start_height"),
                ),
                floor=_floor_spec,
                stall_buffers=int(cfg.autodr_stall_buffers),
            )
        if str(cfg.dr_mode) == "autodr":
            if cfg.start_tip_above_entrance is None:
                raise ValueError(
                    "dr_mode='autodr' with start_tip_above_entrance = None (the home-pose "
                    "measurement setting): start height is one of the five randomised "
                    "quantities and its width-0 centre is that field. There is no start "
                    "pose to open a band around."
                )
            # EXACTLY the static fields a BOUNDARY owns, and no others. The
            # test is "does an autodr.DR_DIMS dimension draw this quantity",
            # not "is this randomisation":
            #   fixture_tilt_rad / fixture_yaw_rad / the two noises -> `yaw`,
            #     `tilt`, and the static probe writes the same _fixture_quat.
            #   start_lateral_offset                                -> `lat_r`.
            #   start_tip_above_entrance_low                        -> `start_height`.
            #     A `low` below the high makes _map_start_conditions draw the
            #     height from [low, high] off the SAME table column the
            #     boundary bounds. Its default is low == high, i.e. absent.
            #
            # NOT IN THE SET, and this is a plan decision, not an oversight
            # (plan section 1, "Nicht adaptiert, aber vorhanden"):
            # `fixture_pos_noise_xy` (0.005), the joint noise and the tilt
            # azimuth stay live under AutoDR. No boundary tracks them, so
            # there is no second source to collide with -- they are constant
            # disturbances, and the evaluation table carries a column for
            # each (autodr.TABLE_COLUMNS).
            _static_dr = {
                "fixture_yaw_noise_rad": float(cfg.fixture_yaw_noise_rad),
                "fixture_tilt_noise_rad": float(cfg.fixture_tilt_noise_rad),
                "fixture_tilt_rad": float(cfg.fixture_tilt_rad),
                "fixture_yaw_rad": float(cfg.fixture_yaw_rad),
                "start_lateral_offset": self._start_lat_half,
            }
            _live = {k: v for k, v in _static_dr.items() if v != 0.0}
            _low_static = cfg.start_tip_above_entrance_low
            if (
                _low_static is not None
                and float(_low_static) != float(cfg.start_tip_above_entrance)
            ):
                _live["start_tip_above_entrance_low"] = float(_low_static)
            if _live:
                # Refused, not merged. Both sources would write the SAME
                # buffers at the same reset from the same table column, so the
                # applied value would be whichever line ran last while the
                # AutoDR boundaries tracked only their own half -- the run
                # would randomise something no buffer measures and no log line
                # names.
                raise ValueError(
                    f"dr_mode='autodr' together with static randomisation {_live}: an "
                    "AutoDR boundary already owns each of these quantities. Set these "
                    "fields to 0.0 (start_tip_above_entrance_low: equal to "
                    "start_tip_above_entrance) and let the boundaries open, or use "
                    "dr_mode='off'. fixture_pos_noise_xy, the joint noise and the tilt "
                    "azimuth are NOT in this set -- no boundary tracks them and they stay "
                    "live (plan section 1)."
                )
            self._dr = autodr.AutoDR(
                autodr.bind_centres(
                    {
                        "lat_r": 0.0,
                        "yaw": 0.0,
                        "tilt": 0.0,
                        "start_height": float(cfg.start_tip_above_entrance),
                        "friction": insertion_tasks_cfg.CONTACT_FRICTION,
                    }
                ),
                floor=_floor_spec,
                stall_buffers=int(cfg.autodr_stall_buffers),
            )
        elif str(cfg.dr_mode) != "off":
            raise ValueError(
                f"dr_mode={cfg.dr_mode!r} is not a mode this env can run. Known: 'off' "
                "(static cfg fields) and 'autodr'. 'table' is the plan's evaluation mode "
                "and arrives with play.py in Phase C; it is refused rather than silently "
                "run as 'off', which would score a policy against the wrong distribution."
            )

        # THE REACH: how far each quantity can EVER open in this run.
        #
        # Every once-per-run branch below reads this and not the current
        # bounds. AutoDR starts at width 0, so at __init__ time its `bounds()`
        # report a zero span for yaw and tilt even though the pocket will
        # tilt later. A buffer sized from that answer is missing for the whole
        # run, and the failure is SILENT: the reset writes a tilted pocket,
        # `_tilt_rot` is still None, and the pocket-frame measurement quietly
        # stays env-local.
        #
        # With no provider the static fields ARE the reach, which is what
        # "off" mode has always meant.
        #
        # TWO reaches, and they are not the same question.
        #
        # `_angle_reach` -- can the pocket ever be at an angle, from ANY
        #   source, a fixed probe angle included. It sizes `_tilt_rot`.
        # `_fixture_pose_reach` -- does a RESET ever write a new FIXTURE pose.
        #   NOT "is anything drawn per episode": the start height, the lateral
        #   offset and the friction are all per-episode and none of them
        #   touches the fixture prim. A fixed probe angle is not in it either:
        #   that pose is built once in __init__ and a reset write would
        #   overwrite it with identity, which is the defect the `randomised`
        #   guard below already documents.
        if self._dr is not None:
            _reach = self._dr.bounds_max()
            # `.get` with the STATIC field as the fallback: the No-DR floor
            # provider carries only start_height, and for it the reach of an
            # absent quantity is what 'off' mode has always read. (Those
            # fields are refused non-zero with a floor, so the fallback is a
            # zero span there; the expression still names the true source.)
            _yaw_r = _reach.get("yaw", (-float(cfg.fixture_yaw_noise_rad), float(cfg.fixture_yaw_noise_rad)))
            _tilt_r = _reach.get("tilt", (0.0, float(cfg.fixture_tilt_noise_rad)))
            _yaw_span = _yaw_r[1] - _yaw_r[0] > 0.0
            _tilt_span = _tilt_r[1] - _tilt_r[0] > 0.0
            _angle_reach = _yaw_span or _tilt_span
            # fixture_pos_noise_xy is NOT an AutoDR quantity and is NOT
            # refused above (plan section 1: present, not adapted), so it is
            # live in this mode and still moves the fixture on its own.
            _fixture_pose_reach = _angle_reach or float(cfg.fixture_pos_noise_xy) > 0.0
        else:
            _angle_reach = (
                float(cfg.fixture_tilt_rad) != 0.0
                or float(cfg.fixture_tilt_noise_rad) != 0.0
                or float(cfg.fixture_yaw_rad) != 0.0
                or float(cfg.fixture_yaw_noise_rad) != 0.0
            )
            _fixture_pose_reach = (
                float(cfg.fixture_pos_noise_xy) > 0.0
                or float(cfg.fixture_yaw_noise_rad) > 0.0
                or float(cfg.fixture_tilt_noise_rad) > 0.0
            )
        # Does the reset write a fixture pose at all? One flag, read once in
        # _reset_idx. The guard used to be a direct read of the same three cfg
        # fields, which answered "no" for the whole of an AutoDR run.
        self._fixture_pose_reach = _fixture_pose_reach

        # THE FRICTION REACH (Phase 5 step B4). Third reach, same rule as the
        # two above and the same failure if it were read off `bounds()`:
        # AutoDR starts at width 0, so today's friction band is the single
        # point CONTACT_FRICTION, and a run that sized this branch from it
        # would never write friction again -- while the boundary opens and the
        # log reports a band the physics never saw.
        #
        # WITH NO PROVIDER THERE IS NO REACH. Friction is the one randomised
        # quantity with no static cfg field: nothing but a bounds provider can
        # widen it, so `dr_mode='off'` keeps exactly the one-shot D-111 write
        # above and pays nothing per reset. This is NOT the deleted fallback
        # "one value per env, drawn once at startup" -- that was a way to
        # randomise, and it is gone; this is the un-randomised nominal case.
        if self._dr is not None:
            _fric = self._dr.bounds_max().get(
                "friction",
                (insertion_tasks_cfg.CONTACT_FRICTION, insertion_tasks_cfg.CONTACT_FRICTION),
            )
            self._friction_reach = _fric[1] - _fric[0] > 0.0
        else:
            self._friction_reach = False

        # THE TILT REACH (Phase 5 step B5). Fourth reach, same rule as the
        # three above, and it fixes plan risk R1 one level BELOW the block
        # guard that already carries it. `_fixture_pose_reach` decides whether
        # the reset writes a fixture pose at all; the tilt block INSIDE that
        # branch had its own guard, and that one still read
        # `cfg.fixture_tilt_noise_rad` directly. Under AutoDR that field is
        # refused non-zero (see the provider above), so the inner guard
        # answered "no" from iteration 0 to the end: the pocket would have
        # yawed and never tilted, while `dr/tilt_hi` climbed on the curve and
        # the tilt bins reported a solved 0-degree task.
        if self._dr is not None:
            _tilt = self._dr.bounds_max().get("tilt", (0.0, float(cfg.fixture_tilt_noise_rad)))
            self._tilt_reach = _tilt[1] - _tilt[0] > 0.0
        else:
            self._tilt_reach = float(cfg.fixture_tilt_noise_rad) > 0.0

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
        # env. None (not a stack of identities) whenever no angle can EVER be
        # asked for, so the untilted path stays byte-for-byte the pre-D-036
        # code: the regression case costs no transform and cannot drift
        # numerically.
        #
        # "can ever be asked for" is `_angle_reach`, not the current width.
        # Under AutoDR both angle boundaries start ON the centre, so a test of
        # today's value answers "no angle" at iteration 0 and these two
        # buffers would stay None for a run whose whole point is to tilt.
        self._fixture_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._fixture_quat[:, 0] = 1.0
        self._tilt_active = _angle_reach
        if self._tilt_active:
            deterministic = float(cfg.fixture_tilt_rad) != 0.0 or float(cfg.fixture_yaw_rad) != 0.0
            # fixture_pos_noise_xy belongs HERE and not in _tilt_active above:
            # that one asks "is any orientation configured at all", which xy
            # offset does not answer. This one asks "does _reset_idx write a
            # per-episode pose", and xy offset alone makes it write one --
            # orientation included, as identity. Without this operand a run
            # with xy noise AND a deterministic fixture_tilt_rad passed the
            # guard and then had its static tilt silently overwritten with
            # identity at the first reset, so the run trained a task its own
            # config denied.
            randomised = (
                float(cfg.fixture_tilt_noise_rad) != 0.0
                or float(cfg.fixture_yaw_noise_rad) != 0.0
                or float(cfg.fixture_pos_noise_xy) != 0.0
            )
            if deterministic and randomised:
                raise ValueError(
                    "a deterministic pocket angle (fixture_tilt_rad / fixture_yaw_rad) is "
                    "set together with a per-episode randomisation (fixture_tilt_noise_rad / "
                    "fixture_yaw_noise_rad / fixture_pos_noise_xy): one run cannot be a "
                    "deterministic probe "
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
            self._tilt_rot = insertion_math.axes_from_quat(self._fixture_quat)
            self._tilt_rot_t = self._tilt_rot.transpose(1, 2).contiguous()
        else:
            self._tilt_rot = None
            self._tilt_rot_t = None

        # The pocket rotation as insertion_math.part_tip_pose wants it: one
        # (N, 3, 3) per env, pocket axes as COLUMNS in the env frame, with no
        # second branch for the untilted case. _tilt_rot is exactly that
        # matrix but is None when no tilt is configured at ALL -- a static
        # property of the config, decided here in __init__ and never changed
        # by a reset -- so the identity built once here serves that branch for
        # the whole run. Passing identity is not a special case in the math:
        # every line of part_tip_pose still holds.
        self._pocket_rot_identity = (
            torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        )

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
        # Kept because the startup report names this joint when it separates
        # the sampled yaw from a real tracking error.
        self._yaw_joint_idx = yaw_idx

        # ------------------------------------------------------- RUNG 0 START
        # D-161: the episode may start at a HEIGHT other than the home pose,
        # given as ``cfg.start_tip_above_entrance`` above the stage-2 opening
        # plane. Height -> joints is an IK problem, and the solver is the one
        # that is already verified for exactly this teleport: the dls
        # ``DifferentialIKController`` loop of ``check_seated_success.py``,
        # which RT-119 drove to eight heights with a residual under 0.001 mm.
        # It NEVER steps the sim (the RT-80 lesson), so it is safe inside a
        # partial reset -- Factory's own reset IK is not (``factory_env.py``
        # ``set_pos_inverse_kinematics`` calls ``step_sim_no_action``, and its
        # docstring says it may only run when ALL envs reset together).
        #
        # OFF (``None``) is the pre-D-161 bare home pose, which is what the
        # measurement scripts want and what a degraded env without the welded
        # tool has to fall back to.
        self._start_ik = None
        self._jacobi_idx = None
        # Diagnosis, written by every reset and read by _write_metrics: the
        # worst residual of the LAST solve, its iteration count, and the worst
        # residual seen in the whole run. A start pose that silently misses is
        # a start pose nobody can report the run against.
        self._start_solve_residual_mm = 0.0
        self._start_solve_iters = 0
        self._start_solve_worst_mm = 0.0
        self._start_solve_unconverged = 0
        # HOW MANY ENVS LOST THEIR AutoDR BOOKING to an unconverged start
        # solve (Phase 5 step B7, user decision 2026-09-10). Cumulative over
        # the run, like the counter above. A boundary that never fills while
        # this number climbs is the same defect seen from the other end.
        self._start_solve_flags_dropped = 0
        # START-HEIGHT SAMPLING (SBC, plan step C, 2026-09-02): the three
        # refusals the cfg field documents. Checked BEFORE the solver exists,
        # so a bad range never builds an env that silently starts elsewhere.
        # With no upper bound (home pose) there is no solver and the lower
        # bound is not read at all. ``_sampling`` is the one flag the prints
        # and the report key off: a range of zero width IS the fixed start.
        _low = cfg.start_tip_above_entrance_low
        self._start_sampling = False
        if _low is not None and cfg.start_tip_above_entrance is not None:
            _low = float(_low)
            self._start_sampling = _low < float(cfg.start_tip_above_entrance)
            if _low > float(cfg.start_tip_above_entrance):
                raise ValueError(
                    f"start_tip_above_entrance_low ({_low:+.4f} m) lies above "
                    f"start_tip_above_entrance ({float(cfg.start_tip_above_entrance):+.4f} m); "
                    "the range is [low, high] with low <= high."
                )
            if _low <= -float(cfg.depth_min):
                raise ValueError(
                    f"start_tip_above_entrance_low ({_low:+.4f} m) is at or below the "
                    f"success band (-depth_min = {-float(cfg.depth_min):+.4f} m): a start "
                    "inside the 33-36 mm band is a one-step success with the full lump. "
                    "The lower bound must stay above the band."
                )
            # `_angle_reach`, not the four cfg angle fields: under AutoDR all
            # four are 0.0 and the pocket angle comes from the boundaries, so
            # a direct read passes this guard at iteration 0 and the run walks
            # into RT-120 the moment the tilt boundary opens.
            if _low < 0.0 and (
                float(cfg.reset_joint_noise) != 0.0
                or float(cfg.reset_yaw_noise) != 0.0
                or _angle_reach
            ):
                raise ValueError(
                    "start_tip_above_entrance_low is negative (a start INSIDE the pocket) "
                    "together with an orientation noise or a pocket angle "
                    "(reset_joint_noise / reset_yaw_noise / fixture_tilt_rad / fixture_yaw_rad "
                    "/ fixture_tilt_noise_rad / fixture_yaw_noise_rad, or an angle boundary "
                    "the bounds provider can open): the start IK commands "
                    "position only, so an un-commanded orientation inside the pocket is a "
                    "wall contact at reset (RT-120). Set all six to 0.0 for a negative low, "
                    "and do not combine a negative low with dr_mode='autodr'."
                )
            # SAME SHAPE, DIFFERENT REASON (D-170). A negative low teleports
            # the part BELOW the opening plane; a lateral offset there aims it
            # into pocket WALL, which is the RT-120 wall contact again. It
            # also breaks the depth seed below (_solve_start_pose, "the part
            # is on the pocket axis here"): off the axis the D-157 lateral
            # gate is NOT satisfied and the seeded depth would not be a gated
            # depth. Both reasons point the same way, so the combination is
            # refused rather than special-cased.
            if _low < 0.0 and self._start_lat_half != 0.0:
                raise ValueError(
                    f"start_tip_above_entrance_low ({_low:+.4f} m) is negative (a start "
                    f"INSIDE the pocket) together with start_lateral_offset "
                    f"(radius {self._start_lat_half:.4f} m): "
                    "a lateral goal below the "
                    "opening plane aims the part into the pocket wall at reset (RT-120), "
                    "and the gated-depth seed assumes the part is on the pocket axis. "
                    "Set start_lateral_offset to 0.0 for a negative low, or "
                    "keep the start "
                    "above the plane."
                )
        if cfg.start_tip_above_entrance is not None and self._peg_body_idx is not None:
            # The Jacobian row block for a body. Isaac Lab's own example drops
            # one for a FIXED base (run_diff_ik.py:133-136). Read from the
            # articulation, never assumed -- same line as scripted_insert.py.
            self._jacobi_idx = (
                self._peg_body_idx - 1 if self.robot.is_fixed_base else self._peg_body_idx
            )
            self._start_ik = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=True,
                    ik_method="dls",
                    ik_params={"lambda_val": float(cfg.start_pose_ik_lambda)},
                ),
                num_envs=self.num_envs,
                device=self.device,
            )
            print(f"[insertion] rung-0 start pose: tool point "
                  f"{cfg.start_tip_above_entrance * 1000.0:+.3f} mm above the stage-2 "
                  f"opening plane (home pose is "
                  f"{insertion_tasks_cfg.WORKCELL_HOME_TIP_ABOVE_ENTRANCE * 1000.0:+.3f} mm); "
                  f"dls IK, <= {cfg.start_pose_solve_steps} iterations, tolerance "
                  f"{cfg.start_pose_solve_tol_m * 1000.0:.3f} mm, jacobian row block "
                  f"{self._jacobi_idx}, fixed base {self.robot.is_fixed_base}")
            if self._start_lat_half != 0.0:
                print(f"[insertion] rung-0 start LATERAL OFFSET (D-170, a disk since D-178): "
                      f"uniform over the AREA of a DISK of radius {self._start_lat_half * 1000.0:.3f} mm "
                      f"around the pocket axis, drawn per env at every reset. The part starts "
                      f"OFF the pocket axis; plays are PLAY_X / PLAY_Y, and D-153's shoulder "
                      f"hole is reached at HOLE_REACH_OFFSET_Y = "
                      f"{insertion_tasks_cfg.HOLE_REACH_OFFSET_Y * 1000.0:.4f} mm -- read "
                      f"stage1_lateral_y_mm.over_reach in demo_metrics.json.")
            else:
                print("[insertion] rung-0 start lateral offset OFF "
                      "(start_lateral_offset = 0.0): every episode starts ON the pocket "
                      "axis, which CANCELS the lateral part of fixture_pos_noise_xy and "
                      "reset_joint_noise (D-170).")
            if self._start_floor_on:
                print(f"[insertion] START FLOOR (SBC step 0): the start height is drawn per "
                      f"episode from [floor, {cfg.start_tip_above_entrance * 1000.0:+.3f}] mm, "
                      f"floor {float(cfg.start_floor_m) * 1000.0:+.3f} mm at the start of the "
                      f"run, raised in {int(cfg.start_floor_steps)} rungs by the AutoDR "
                      f"buffer rule (dr_mode={cfg.dr_mode}); see 'START FLOOR' in the report.")
            if self._start_sampling:
                print(f"[insertion] rung-0 start height SAMPLED per episode (SBC, plan step C): "
                      f"uniform in [{_low * 1000.0:+.3f}, "
                      f"{cfg.start_tip_above_entrance * 1000.0:+.3f}] mm above the stage-2 "
                      f"opening plane. Episodes below 0 start INSIDE the pocket; the gated "
                      f"max depth is seeded with the start depth, so D-165 pays nothing for "
                      f"the teleport itself.")
        elif cfg.start_tip_above_entrance is not None:
            print("[insertion] rung-0 start pose REQUESTED but there is no welded tool "
                  f"body '{cfg.peg_body_name}' -- the bare home pose runs instead.")
        else:
            print("[insertion] rung-0 start pose OFF (start_tip_above_entrance = None); "
                  "every episode starts at the bare home pose, "
                  f"{insertion_tasks_cfg.WORKCELL_HOME_TIP_ABOVE_ENTRANCE * 1000.0:+.3f} mm "
                  "above the stage-2 opening plane. D-161 measured the whole reward at "
                  "3.05e-23 there.")

        # The start height each env's CURRENT episode was commanded to, metres
        # above the opening plane. One scalar per env, written by
        # _solve_start_pose (sampled or the fixed bound) and read by
        # _log_finished_episodes at the top of the NEXT reset -- the same
        # ordering the tilt bins rely on (D-037). Filled with the fixed start
        # here so an env that never solves (home pose) still reports a height.
        self._start_height = torch.full(
            (self.num_envs,), self._start_tip_height, device=self.device
        )
        # The lateral start offset each env's CURRENT episode was commanded
        # to, metres in the POCKET frame (x, y), same lifetime and the same
        # writer as _start_height above (D-170). Zeros when the field is 0.0,
        # in which case _solve_start_pose never writes it and the goal is the
        # bare pocket axis -- the pre-D-170 behaviour. See the cfg field for
        # why that is "unchanged behaviour" and not "bit-identical".
        self._start_lat_off = torch.zeros((self.num_envs, 2), device=self.device)
        # The contact friction each env's CURRENT episode is running with
        # (Phase 5 step B4). Filled with the nominal value, which is what a
        # run without a friction reach keeps for good.
        #
        # TWO READERS since step B5. The startup report prints env 0..3 of
        # this buffer beside the PhysX read-back, and `_log_finished_episodes`
        # copies the finished episodes' values into `_recent_friction`. The
        # second one works only because of the lifetime the two buffers above
        # also have: written in `_reset_idx` AFTER `_log_finished_episodes`
        # has run for those envs, so at the top of the next reset the buffer
        # still holds what the ending episode ran under -- the same ordering
        # `_start_height` relies on. Said here because the ordering is the
        # part that is easy to get wrong later and impossible to see from the
        # read site.
        #
        # This buffer is what WE COMMANDED. It is not evidence that PhysX took
        # it -- the startup report's read-back is that, and only after a reset
        # has run.
        self._friction_applied = torch.full(
            (self.num_envs,), float(insertion_tasks_cfg.CONTACT_FRICTION), device=self.device
        )

        # THE RESET DRAW (Phase 5, plan section 10). ONE uniform row per
        # env, one column per random number that influences a reset
        # condition -- the sixteen of `autodr.TABLE_COLUMNS`, which is the
        # one home of that layout. Drawn ONCE per reset, BEFORE anything is
        # applied, so that:
        #   * every consumer reads the same row and no quantity is drawn
        #     twice or in two places (it was two before: _reset_idx drew
        #     the joint noise and the pocket pose, _solve_start_pose the
        #     height and the lateral offset);
        #   * a boundary env can be nailed by writing ONE column to 0 or 1
        #     (autodr.apply_boundary) before anything reads it;
        #   * the table mode replaces the draw with a table row and nothing
        #     downstream changes.
        # Values stay in [0, 1); each consumer maps its own column.
        self._reset_unit = torch.zeros(
            (self.num_envs, len(autodr.TABLE_COLUMNS)), device=self.device
        )
        self._col = dict(autodr.COLUMN_INDEX)

        # WHICH BOUNDARY each env's CURRENT episode is nailed to (Phase 5
        # step B5), -1 for a regular episode. Same lifetime and the same
        # ordering dependency as `_start_height`: written in
        # `_draw_reset_conditions`, read by `_log_finished_episodes` at the
        # top of the NEXT reset, where it still holds the FINISHED episode's
        # assignment. Booking a flag onto the wrong boundary is invisible --
        # the buffer fills at the right rate and moves the wrong edge.
        self._boundary_of = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # WHICH BOUNDS VERSION each env's current episode STARTED under (plan
        # section 4). The stop rule may only count episodes drawn from the
        # distribution that is current now; an episode that began before a
        # boundary moved carries the old stamp and is left out. Same lifetime
        # and the same reader as the buffer above.
        self._bounds_stamp = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        # The boundaries (seven for autodr.DR_DIMS) as tensors, built ONCE. `boundary_columns`
        # says which table column each boundary nails, `boundary_sides` which
        # edge (0.0 = lo, 1.0 = hi), and `keys` is the name list `record()`
        # takes. All three are in the SAME order, so the one index
        # `boundary_assignment` returns indexes all three -- that shared order
        # is the whole reason the env needs no mapping of its own.
        if self._dr is not None:
            self._boundary_cols = torch.tensor(
                self._dr.boundary_columns, dtype=torch.long, device=self.device
            )
            self._boundary_sides = torch.tensor(
                self._dr.boundary_sides,
                dtype=self._reset_unit.dtype,
                device=self.device,
            )
            self._boundary_keys = tuple(self._dr.keys)
        else:
            self._boundary_cols = None
            self._boundary_sides = None
            self._boundary_keys = ()
        # The table reserves SIX joint-noise columns (plan section 3). A
        # robot with a different joint count would silently misalign them,
        # so it is refused here rather than found in a metrics file.
        _n_joint_cols = sum(1 for c in autodr.TABLE_COLUMNS if c.startswith("joint_noise_"))
        if len(self.robot.joint_names) != _n_joint_cols:
            raise ValueError(
                f"the reset table has {_n_joint_cols} joint-noise columns but the "
                f"robot has {len(self.robot.joint_names)} joints "
                f"({self.robot.joint_names}); autodr.TABLE_COLUMNS and the "
                "articulation must agree"
            )
        self._joint_noise_cols = torch.tensor(
            [self._col[f"joint_noise_{i}"] for i in range(_n_joint_cols)],
            device=self.device, dtype=torch.long,
        )

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
        self._build_task_space_controller()
        # THE PREVIOUS STEP'S ACTION, per env, for the action-rate penalty
        # (inbox entry "Reward-Ueberarbeitung" 2026-09-01, point (8); FORGE
        # ``forge_tasks_cfg.py:15``). Written in _pre_physics_step and CLEARED
        # TO ZERO on reset, which is the FORGE convention (it resets its own
        # prev-action buffers) and has one visible consequence, stated rather
        # than discovered: the first step of an episode pays
        # ``scale * ||a_0 - 0||``, i.e. it prices the first command against
        # standing still. That is the behaviour we want at a reset pose the
        # policy has just been teleported to, and it is the same at every
        # reset, so it biases no episode against another.
        self._prev_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self._last_action_rate = torch.zeros(self.num_envs, device=self.device)
        # The EMA state of the force channel (D-114, alpha = ft_smoothing_factor).
        # Three components, not six: the policy is handed the FORCES only, the
        # wrench's three torque components are not part of the observation.
        # Cleared to ZERO in _reset_idx, not seeded with the first sample --
        # the FORGE convention (forge_env.py:331), which also keeps a stale
        # reset-step reading (the documented "stale values" issue) out of the
        # seed.
        self._force_smooth = torch.zeros(self.num_envs, 3, device=self.device)
        # MEASUREMENT ONLY (RT-189s1pf, 2026-09-13): the tared force BEFORE the
        # EMA and before the observation noise, for play.py's trace. Nothing in
        # the observation, reward or termination reads it.
        self._force_tared_raw = torch.zeros(self.num_envs, 3, device=self.device)
        # THE RESET-STEP TRAP, CLOSED (2026-09-13, inbox entry "The wrench EMA
        # skips the reset step"). DirectRLEnv.step runs _reset_idx and then
        # _get_observations with NO physics step between them
        # (direct_rl_env.py:396-410), and body_incoming_joint_wrench_b is
        # re-read from PhysX on every access (the `time_stamp` typo in
        # articulation_data.py:746 means its cache never hits). So the wrench
        # an episode's FIRST observation reads is the LAST wrench of the OLD
        # episode -- RT-189s1pf measured 8.08 N raw on row 0, the untared tool
        # weight against the new pose's quaternion. Until today the EMA took
        # 0.25 of that stale reading as the episode's first sample.
        # `_wrench_fresh` is raised in _reset_idx and consumed by the first
        # _get_observations after it: a fresh env gets NO EMA update that
        # step, its buffers stay at the reset zero, and `_wrench_valid` says
        # so for the log and the trace. This changes row 0 of EVERY policy's
        # force channel from 0.25 * stale to 0.0 -- named, not hidden.
        self._wrench_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._wrench_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # THE TORQUE TWIN (D-188): the same wrench's columns 3:6, tared with
        # gravity_tare_torque and smoothed with the SAME alpha. Filled in
        # EVERY observation mode -- under `force` it is a diagnosis channel
        # (log, trace truth block); under `wrench` it is obs 28:31. Cleared
        # in _reset_idx exactly like _force_smooth.
        self._torque_smooth = torch.zeros(self.num_envs, 3, device=self.device)
        self._torque_tared_raw = torch.zeros(self.num_envs, 3, device=self.device)
        self._max_torque_norm = torch.zeros(self.num_envs, device=self.device)
        # Share of steps whose wrench was valid, over the run (log instrument).
        self._wrench_valid_steps = 0
        self._wrench_steps = 0
        # THE ONCE-GUARD: the rsl_rl wrapper calls _get_observations a second
        # time without physics (vecenv_wrapper.py:148, at runner start), and
        # every such call used to apply the EMA again to the same reading.
        # The EMA now advances once per `common_step_counter` value.
        self._wrench_step = -1
        # The measured reference point of the moment, parent frame (cfg).
        self._torque_ref_offset_parent = torch.tensor(
            [float(v) for v in self.cfg.torque_ref_offset_parent_m], device=self.device
        ).reshape(1, 3).repeat(self.num_envs, 1)
        # Worst smoothed force magnitude reached in the current episode, per
        # env. This is the instrument the scripted-insertion gate reads to
        # replace the invented F_max (D-114); it is not part of the task.
        self._max_force_norm = torch.zeros(self.num_envs, device=self.device)
        # SOLVER-ACCURACY INSTRUMENT (inbox 2026-09-12, "16 solver iterations
        # vs Isaac Lab Factory's 192"). Worst SAPU interpenetration depth
        # reached in the current episode, per env, metres. The depth already
        # scales the reward (D-109); this buffer only makes it READABLE, so
        # that an env-count or solver-iteration change can be judged on the
        # penetration it causes instead of on the reward it earns.
        self._max_interpen_m = torch.zeros(self.num_envs, device=self.device)
        # RT-201s3 instrument (2026-09-15): did the tip go BELOW the fixture
        # underside inside the pocket cross-section in this episode? Harvested
        # and zeroed like _max_interpen_m. insertion_math.in_pocket_volume.
        self._below_fixture = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # EPISODE RECORD (2026-09-15, `EPISODE_LOG_COLUMNS`). The policy steps
        # of the current episode whose interpenetration was at or above the
        # SAPU threshold (`insertion_math.interpen_unclean`), and the worst
        # norm of the gravity-tared UNSMOOTHED force (the abort reads the
        # smoothed one, `_max_force_norm`). Both harvested and zeroed like
        # _max_interpen_m. `_episode_index` counts the episodes each env has
        # finished and is never reset.
        self._interpen_unclean_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._max_force_raw_norm = torch.zeros(self.num_envs, device=self.device)
        self._episode_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        # (file handle, csv writer), opened on the first harvested episode.
        self._episode_log = None
        # RAW-ACTION INSTRUMENT (D-168, 2026-09-05). The action-rate term is
        # the only price on the UNCLAMPED command, and D-168 cuts its scale by
        # a factor of 29. The named risk is that the action mean now drifts
        # outside [-1, 1], every draw saturates the clamp, and exploration is
        # dead again while sigma still reads healthy -- RT-149 already measured
        # |a| up to 3.5. These two buffers make that visible instead of
        # assumed: the worst |a| of the episode, and the running sum of the
        # per-step share of action components with |a| > 1 (divided by
        # _action_steps at reset to give the episode mean).
        self._max_action_abs = torch.zeros(self.num_envs, device=self.device)
        self._action_sat_sum = torch.zeros(self.num_envs, device=self.device)
        self._action_steps = torch.zeros(self.num_envs, device=self.device)
        # D-153 EXPOSURE INSTRUMENT (2026-08-30). The fixture mesh keeps an
        # unpaired ~1.5 mm triangle on the stage-1 shoulder. D-153 accepted it
        # because at fixture_pos_noise_xy = 0.0 the part never leaves the
        # pocket axis, and reopened it BY ITS OWN CLAUSE the moment that noise
        # goes above zero -- which it now is. The argument that follows from
        # the clause is the same every rung: stage 1 allows the part centre
        # +8.7000 mm of lateral y, the hole is reached at
        # HOLE_REACH_OFFSET_Y = +7.8847 mm, therefore the exposure is
        # POSSIBLE. This buffer turns that argument into a NUMBER: the worst
        # |part_y - pocket_y| the episode actually reached while the part was
        # inside stage 1. Instrument only -- nothing in the task reads it.
        self._max_stage1_lat_y = torch.zeros(self.num_envs, device=self.device)

        # Values from the current step's _get_dones() call, read back by
        # _get_rewards() and by _reset_idx() for the envs that just finished.
        # DirectRLEnv's step loop calls _get_dones() BEFORE _get_rewards(),
        # so the geometry/success computation lives in _get_dones() and is
        # cached forward -- not the other way round. All of them are the REAL
        # task's since M2.4b step 3; the proxy's gate/below_plate/alignment/
        # cos4phi/remaining caches died with the proxy reward.
        self._last_depth = torch.zeros(self.num_envs, device=self.device)
        self._last_sdf_dist = torch.zeros(self.num_envs, device=self.device)
        self._last_interpen_max = torch.zeros(self.num_envs, device=self.device)
        self._last_engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_in_region = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # The force abort ALONE, separate from _last_terminated since
        # 2026-09-01: terminated is now the union of the abort and the success,
        # while the reported abort rate (D-113 (9)) is about the abort only.
        self._last_force_abort = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Steps left in the episode at the current step, as a float -- the
        # multiplier of the success payout (inbox entry
        # "Reward-Ueberarbeitung" 2026-09-01, point (7)). Cached like the rest
        # of the _last_* family because episode_length_buf is zeroed by the
        # reset that the success termination triggers, so nothing downstream
        # could reconstruct it afterwards.
        self._last_steps_remaining = torch.zeros(self.num_envs, device=self.device)
        # Metres of NEW gated max depth this step (D-165, 2026-09-02): the
        # proxy's progress signal, computed in _get_dones from the _max_depth
        # buffer and paid w_depth_progress per metre. Zero unless the part
        # got deeper than ever before in this episode.
        self._last_depth_progress = torch.zeros(self.num_envs, device=self.device)
        # RT-171, the alignment potential Phi = w_tilt * sech(a * theta) of the
        # CURRENT step, cached by _get_dones like the rest of the family.
        self._last_tilt_phi = torch.zeros(self.num_envs, device=self.device)
        # ... and of the pose the policy was SHOWN before acting, written by
        # _get_observations from the very quaternion the observation carries.
        # That is what makes the first step of an episode honest: the reset
        # observation is computed AFTER _reset_idx and is the s_0 the policy
        # sees, so Phi(s_0) comes from there and never from a link pose read
        # inside the reset (the stale-values case, see _reset_idx). The row
        # gamma * Phi(s_t+1) - Phi(s_t) then telescopes over the episode to
        # gamma^T Phi(s_T) - Phi(s_0), independent of the path.
        self._tilt_phi_prev = torch.zeros(self.num_envs, device=self.device)
        # Episode success = the SUCCESS TERMINATION event (2026-09-01, point
        # (7)), which replaces success_and_hold's "seated at the last step":
        # the episode now ENDS on the seat, so the seating step IS the last
        # step. _log_finished_episodes reads it for the envs about to reset.
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # The two Warp queries of the real reward (D-109 (3) SDF distance,
        # D-109 (10) SAPU), built ONCE and never refitted (departure 1 of
        # insertion_sdf). Behind the switch AND the welded-tool guard: the
        # measurement env (rl_terms_enabled = False -- seat_probe,
        # scripted_insert, zero/random agents) must run without Warp, trimesh
        # or the OBJ files, and so must a degraded env without the tool.
        #
        # Both queries SAMPLE the part OBJ with the same seed; they differ in
        # what they QUERY: the reward asks the part's own surface at the goal
        # pose, SAPU asks the fixture mesh (one class, mesh_obj_path -- no
        # second kernel, see insertion_math's SAPU header).
        if self._rl_terms_enabled and self._peg_body_idx is not None:
            self._sdf_query = insertion_sdf.SdfDistanceQuery(
                insertion_paths.resolve_part_obj_path(),
                int(cfg.sdf_num_sample_points),
                float(cfg.sdf_max_dist),
                str(self.device),
                int(cfg.sdf_seed),
            )
            self._sapu_query = insertion_sdf.SdfDistanceQuery(
                insertion_paths.resolve_part_obj_path(),
                int(cfg.sdf_num_sample_points),
                float(cfg.sdf_max_dist),
                str(self.device),
                int(cfg.sdf_seed),
                mesh_obj_path=insertion_paths.resolve_pocket_obj_path(),
            )
            print(self._sdf_query.describe())
            print(self._sapu_query.describe())
        else:
            self._sdf_query = None
            self._sapu_query = None

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
        # iteration scale with it (about 256 at 4096 envs and the current
        # 256-step episode, against 64 at 1024), so a fixed window covers four
        # times less training at 4096 and turns into a snapshot of a handful of
        # iterations. The floor keeps the statistics sound at small env
        # counts -- 2000 episodes put the standard error of a 90 % rate at
        # 0.7 points.
        #
        # THE EPISODE LENGTH IS READ, NOT TYPED (S4, 2026-08-28). It was the
        # literal 240 until D-113's 256 landed, and a stale divisor here does
        # not fail anything -- it silently mis-sizes the window the success
        # rate is read from.
        # THE FLOOR IS ALSO THE STOP RULE'S MINIMUM (plan section 4). Own
        # decision, no methodical precedent: rather than invent a second
        # number, the plan pins the fresh window's minimum to this floor, so
        # the two have ONE home and cannot drift apart. Both readers are
        # below -- `window` here, `self._fresh_min` further down.
        window_floor = 2000
        window = max(window_floor, int(
            self.num_envs * self.cfg.metrics_window_iterations * 16 / self.max_episode_length
        ))
        self._recent_successes: collections.deque = collections.deque(maxlen=window)
        self._recent_depths: collections.deque = collections.deque(maxlen=window)
        # Steps the SUCCESSFUL episodes took, over the same trailing window.
        # Answers "does the episode length bind?" as a read number: at ±5 cm
        # fixture noise the approach is longer, and if the mean creeps toward
        # the episode cap the length is the suspect, measured not guessed.
        self._recent_success_steps: collections.deque = collections.deque(maxlen=window)
        # Pocket tilt magnitude (degrees) of each episode in the same trailing
        # window, index-aligned with _recent_successes (D-037). Under a
        # randomised tilt the mean success rate is not a readable result: 90 %
        # can mean "90 % everywhere" or "100 % near upright and 0 % at the
        # edge", and the large angles are the entire point of the training.
        # Keeping the angle per episode lets _write_metrics report the rate per
        # angle bin instead of one number that hides its own failure mode.
        self._recent_tilt_deg: collections.deque = collections.deque(maxlen=window)
        # Upper bin edges in DEGREES (D-178 (7)): five bins of 2 deg under the
        # 10 deg tilt ceiling (autodr.DR_DIMS `tilt`, hi_max
        # 0.17453292519943295 rad), the LAST edge ON that ceiling.
        # `_rate_by_bin` makes ONE bin per edge, and the last bin also takes
        # everything at or above the last edge (L-07). Rungs, not derived --
        # ceiling/5 x (1..5), the rule lateral (6..30 mm) and yaw (1..5 deg)
        # already follow.
        # CORRECTED 2026-09-12 (user). The edges were (3, 6, 9, 12, 15) deg,
        # the ladder of D-037 ("5 deg, then 15 deg") that D-178 superseded and
        # whose point (7) then re-cut lateral and yaw but not this table.
        # Against a 10 deg ceiling those edges made TWO of the five bins
        # unreadable: [12, inf) can never fill, and [9, 12) only ever sees
        # 9-10 deg. Under `dr_mode='autodr'` nothing can exceed the ceiling --
        # the `_static_dr` block above refuses a non-zero `fixture_tilt_rad`
        # or `fixture_tilt_noise_rad` -- so the old top bins were dead by
        # construction, not merely unfilled. `scripts/check_autodr.py` now
        # holds this last edge against `DR_DIMS`, so the two cannot drift
        # apart again.
        self._tilt_bin_edges_deg = (2.0, 4.0, 6.0, 8.0, 10.0)
        # Start height (metres above the opening plane) of each episode in
        # the same trailing window, index-aligned with _recent_successes.
        # Under start-height sampling (plan step C) the mean success rate is
        # not readable for the same reason as under a randomised tilt: it
        # cannot show whether the HARD end (the +30 mm start, outside) was
        # learned or only the easy end (inside). IndustReal names that
        # failure mode -- success in the partially-inserted starts and 0 in
        # the top bin -- and the per-bin table is the readout for it.
        self._recent_start_height_m: collections.deque = collections.deque(maxlen=window)
        # Upper bin edges in METRES, 10 mm wide; the last bin catches
        # anything at or above +30 mm (the fixed rung-0 start lands there).
        # D-179 (5), after the H_min measurement (2026-09-14): 10 mm bins over
        # the floor band (bin 0 is open below and catches the -0.030 floor),
        # then 20 mm bins up to the start_height ceiling, whose LAST edge is
        # autodr.DR_DIMS start_height hi_max (pinned in check_autodr).
        self._start_height_bin_edges_m = (
            -0.020, -0.010, 0.0, 0.010, 0.020, 0.040, 0.060, 0.080, 0.100, 0.120,
        )
        # LATERAL start offset magnitude (mm) of each episode in the same
        # trailing window, index-aligned with _recent_successes (D-170).
        # RT-174 trained a 6 mm lateral start and read 100 % overall -- and
        # that number cannot say whether a 6 mm start was learned or only a
        # 0.5 mm one, because the offset is drawn per episode and the mean
        # rate averages over the whole draw. Same argument as the tilt table
        # above, on the axis D-170 opened. The value is the NORM of the two
        # pocket-axis components, i.e. the RADIUS of the disk the offset is
        # drawn on (D-178): it is at most that radius, and there is no
        # r*sqrt(2) corner any more.
        self._recent_lateral_mm: collections.deque = collections.deque(maxlen=window)
        # Upper bin edges in MILLIMETRES (D-178 (7)): five bins of 6 mm, the
        # LAST edge on the `lat_r` ceiling (autodr.DR_DIMS). `_rate_by_bin`
        # makes ONE bin per edge, and the last bin also takes everything at or
        # above the last edge (L-07). Rungs, not derived. Empty bins are kept,
        # so a run at 0.0 offset reads all episodes in the first bin and the
        # table reduces to the overall rate.
        self._lateral_bin_edges_mm = (6.0, 12.0, 18.0, 24.0, 30.0)
        # Pocket YAW magnitude (degrees) of each episode in the same trailing
        # window, index-aligned with _recent_successes (D-038). The third
        # axis of the user's target set; the same readability argument as
        # tilt and lateral, and the yaw reward landscape has never been
        # measured (RT-152 skipped), so the per-bin rate is the only readout
        # a yaw run has. Magnitude, not signed: the pocket is drawn
        # symmetrically about 0 and +4 deg is the same task as -4 deg.
        self._recent_yaw_deg: collections.deque = collections.deque(maxlen=window)
        # Upper bin edges in DEGREES (D-178 (7)): five bins of 1 deg under the
        # 5 deg yaw ceiling (autodr.DR_DIMS), the LAST edge on that ceiling.
        # One bin per edge, the last also taking everything above (L-07).
        # Rungs, not derived.
        self._yaw_bin_edges_deg = (1.0, 2.0, 3.0, 4.0, 5.0)
        # Worst joint-target lag per episode, index-aligned with
        # _recent_successes so the statistic can be split by outcome. The
        # question it answers: do the FAILING episodes end with a wound-up
        # target the policy spent its steps unwinding? Diagnosis instrument.
        self._recent_target_lag: collections.deque = collections.deque(maxlen=window)
        # Worst smoothed force magnitude per episode, index-aligned with
        # _recent_successes exactly like the lag above. This is the DISTRIBUTION
        # D-114 asks for: F_max is to come from the force measured at the
        # scripted-insertion gate, not from the invented number that stands
        # there now, and a distribution cannot be recovered from a single mean.
        self._recent_force_norm: collections.deque = collections.deque(maxlen=window)
        # D-188: the torque twin, N m, same alignment.
        self._recent_torque_norm: collections.deque = collections.deque(maxlen=window)
        # Per-episode worst interpenetration, metres, index-aligned with the
        # deques above (inbox 2026-09-12).
        self._recent_interpen_max: collections.deque = collections.deque(maxlen=window)
        # RT-201s3: per-episode "went below the fixture underside" flag,
        # index-aligned with the deques above (2026-09-15).
        self._recent_below_fixture: collections.deque = collections.deque(maxlen=window)
        # D-168 raw-action readouts, index-aligned with the deques above.
        self._recent_action_abs_max: collections.deque = collections.deque(maxlen=window)
        self._recent_action_sat_frac: collections.deque = collections.deque(maxlen=window)
        # Whether each episode in the window ended in a FORCE ABORT, index-
        # aligned like the deques above. D-113 (9): force does not enter the
        # success definition (D-051), so the abort rate is reported as its own
        # number beside the success rate, never folded into it.
        self._recent_aborts: collections.deque = collections.deque(maxlen=window)
        # Worst stage-1 lateral y offset per episode, index-aligned with
        # _recent_successes like the deques above. The statistic it feeds
        # (_stage1_lateral_stats) answers D-153's open question as a measured
        # tail rather than as a bound: the p99 and the max against
        # HOLE_REACH_OFFSET_Y say whether any episode ever put the part where
        # the hole could be queried.
        self._recent_stage1_lat_y: collections.deque = collections.deque(maxlen=window)
        # THE THREE WINDOWS AUTODR READS (Phase 5 step B5).
        #
        # `_fresh_successes` is the STOP RULE's window (plan section 4). Two
        # filters, and both are load-bearing:
        #   * REGULAR episodes only. A boundary episode is nailed at an edge,
        #     so it is drawn from a different distribution; folding it in
        #     would make the stop rule depend on `p_boundary`.
        #   * only episodes that STARTED under the current `bounds_version`.
        #     The window is cleared the moment a boundary actually moves, so
        #     the run can never stop on evidence the old distribution made.
        # NOT a test rate: these episodes carry PPO's exploration noise. The
        # name says `train_...` for that reason and the plan forbids the other
        # reading.
        self._fresh_successes: collections.deque = collections.deque(maxlen=window)
        # The boundary episodes' own rate, its own curve. Reported beside the
        # regular one, never merged into it.
        self._recent_boundary_successes: collections.deque = collections.deque(maxlen=window)
        # The contact friction each FINISHED episode ran under, one entry per
        # episode, index-aligned with `_recent_successes` like every deque
        # above. This is the reader `_friction_applied` was built without in
        # step B4: until now the only consumer of that buffer was the startup
        # report, so nothing paired a friction with an outcome.
        self._recent_friction: collections.deque = collections.deque(maxlen=window)
        # THE MINIMUM the fresh window must hold before its rate means
        # anything. Read off the floor above -- one home, see there.
        self._fresh_min = window_floor
        # THE COST OF THE PER-RESET FRICTION WRITE, milliseconds, one entry
        # per RESET CALL (Phase 5 step B4, plan section 6). NOT index-aligned
        # with the deques above and it must not be read as if it were: those
        # hold one number per EPISODE, this one holds one number per call of
        # `_apply_friction`, and a reset call covers however many envs ended
        # together. `maxlen` is a plain 1000 for the same reason -- the
        # episode-count arithmetic that sizes `window` does not apply.
        #
        # This is the number that decides the plan's open question: whether
        # the per-reset main path is affordable, or whether the study has to
        # fall back to the redraw-per-boundary-move alternative. It is
        # therefore reported next to the step time of the SAME log, never on
        # its own.
        self._friction_write_ms: collections.deque = collections.deque(maxlen=1000)
        # Per-term reward sums of the RUNNING episode, one row per
        # insertion_math.REWARD_TERMS (2026-09-02, after RT-134). The Isaac
        # Lab direct-env pattern (anymal_c_env.py: _episode_sums ->
        # extras["log"]["Episode_Reward/<term>"] at reset), minus its
        # per-second normalisation: what is logged is the per-episode SUM,
        # the undiscounted return contribution the D-164 arithmetic is about.
        # The trailing window per term feeds demo_metrics.json, index-aligned
        # with _recent_successes like every deque above.
        self._episode_sums: dict = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in insertion_math.REWARD_TERMS
        }
        self._recent_reward_terms: dict = {
            name: collections.deque(maxlen=window) for name in insertion_math.REWARD_TERMS
        }
        # THE RT-107 TRIPWIRE (D-157). Success and the depth metric are now
        # gated by the same lateral test, so a successful episode CANNOT have
        # a zero max depth. This counts the episodes that break that anyway --
        # it is the run-time proof that the gate is actually wired to both,
        # not a statistic anyone tunes against. Anything but 0 means the two
        # paths have drifted apart again, which is what RT-107 was.
        self._ep_success_depth_violations = 0
        self._recent_success_depth_violations: collections.deque = collections.deque(maxlen=window)
        # Fallback only; the real path is the run's own log dir (see
        # _write_metrics), which train.py assigns to cfg.log_dir.
        self._metrics_path = pathlib.Path("logs") / "demo_metrics.json"
        # Also scaled: at 4096 envs a fixed 50 would rewrite the file several
        # times per iteration, which is pure I/O for no extra information.
        #
        # NOT overridable per run. A `metrics_dump_every` cfg field was added
        # on 2026-09-06 and REMOVED the same day: declared `int | None = None`,
        # Isaac Lab's configclass infers the type from the DEFAULT, so every
        # hydra override died with "Incorrect type under namespace:
        # /metrics_dump_every. Expected: NoneType, Received: int" (RT-157h30,
        # exit 1, PROBLEMS.md 2026-09-06). A short probe sizes itself with
        # num_envs instead: at 1024 envs one pass finishes 1024 episodes,
        # far past this interval.
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

        # Table plates (D-058), into env_0 before cloning like the fixture.
        # Deliberately NOT a RigidObject: nothing ever writes their pose, so
        # they stay bare colliders and need neither a RigidBodyAPI nor a scene
        # handle. The asset's own origin IS the environment origin (the robot
        # foot), so the translation is (0, 0, 0) rather than a placement.
        tables_spawn = sim_utils.UsdFileCfg(
            usd_path=resolve_workcell_tables_usd_path(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        self._tables_usd_path = tables_spawn.usd_path
        tables_spawn.func(
            "/World/envs/env_0/Workcell",
            tables_spawn,
            translation=self.cfg.workcell_tables_pos,
        )

        # The block AROUND the pocket (2026-08-24). Until then the block WAS the
        # fixture; now the fixture is the CAD cut-out above and the block is what
        # stands around it: four walls, a floor, and the rear wall that makes the
        # pocket reachable only from the front.
        #
        # OFF by default since 2026-08-30 (cfg.spawn_workcell_block), so the
        # fixture can move for the robustness tests. With the block OFF there is
        # NO cut-out-vs-wall coupling left: the pocket walls are cut into the
        # cut-out itself, so the free-standing fixture is the whole pocket. The
        # rear wall is what is lost -- the pocket becomes enterable from any
        # direction. Full reasoning and the cost live on the cfg field.
        #
        # With the block ON the old consequence returns unchanged and D-125
        # governs: the recess has ZERO clearance and only the cut-out has a
        # runtime pose, so fixture noise drives the two into each other. That is
        # why this branch does not exist to be flipped back on casually.
        #
        # Static scenery either way, exactly like the tables and for the same
        # reason: no runtime pose is ever written to it, so it needs neither a
        # RigidBodyAPI nor a scene handle (D-033).
        #
        # Same translation as the fixture: block_recess_boxes() is asset-local to
        # the POCKET ORIGIN, not to the block centre, so no arithmetic lines them
        # up -- they share one constant.
        if self.cfg.spawn_workcell_block:
            block_spawn = sim_utils.UsdFileCfg(
                usd_path=resolve_workcell_block_usd_path(),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            )
            self._block_usd_path = block_spawn.usd_path
            block_spawn.func(
                "/World/envs/env_0/Block",
                block_spawn,
                translation=self.cfg.workcell_block_pos,
            )
        else:
            # The report reads this unconditionally; None is what "no prim"
            # looks like there, and it must not read a stale path.
            self._block_usd_path = None

        # The UR5e spawns WITH THE TOOL WELDED IN (D-064): gripper and part are
        # one link of this articulation, authored by scripts/author_tool_ur5e.py.
        # The swap happens before the Articulation is built, because the spawner
        # reads usd_path at construction.
        #
        # A missing asset raises here rather than falling back to the bare arm,
        # and that is deliberate. The home pose now holds the flange 302 mm above
        # the block because 152 mm of tool hangs below it; spawning the bare arm
        # at that pose would give a scene that looks plausible, reports no error,
        # and is wrong by the length of the tool. The resolver's message names
        # the script to run.
        self.cfg.robot_cfg.spawn.usd_path = insertion_tasks_cfg.resolve_tool_robot_usd_path()
        self._robot_usd_path = self.cfg.robot_cfg.spawn.usd_path
        print(f"[insertion] robot asset: {self._robot_usd_path}")

        # INERT DRIVES UNDER OSC (Decision (2)): the controller returns joint
        # efforts, so the PhysX position drive must not fight it. Stiffness
        # AND damping of every actuator block go to 0 -- the OSC tutorial's
        # move (run_osc.py:101-104) and Factory's (factory_env_cfg.py:160-172).
        # maxForce, joint limits, velocity limits, armature and friction are
        # NOT touched and stay USD-authored (D-105 for those). Written into
        # the cfg BEFORE the Articulation is built, the same place the usd
        # path swap above lives, and the startup report reads the gains BACK
        # from PhysX so a silent no-op cannot pass as applied.
        if self._control_mode == "osc":
            for _act in self.cfg.robot_cfg.actuators.values():
                _act.stiffness = 0.0
                _act.damping = 0.0
        self.robot = Articulation(self.cfg.robot_cfg)
        # Ground plane. No longer at z = 0: with the robot foot AT the origin
        # (Factory convention) the floor is one table height below it.
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, self.cfg.floor_z),
        )
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
        # THE ACTION-RATE PENALTY'S measurement (inbox entry
        # "Reward-Ueberarbeitung" 2026-09-01, point (8)): the L2 norm of the
        # change in the commanded action, one number per env, read by
        # _get_rewards in the SAME step (Isaac Lab calls _pre_physics_step
        # before _get_dones / _get_rewards).
        #
        # ON THE RAW ACTION, not the clamped one: FORGE's term is the norm over
        # its action buffer, and this env's buffer is `self.actions`. Pricing
        # the clamped pair would make a policy that pushes further and further
        # outside [-1, 1] free of charge, which is exactly the drift the term
        # is meant to see.
        self._last_action_rate = torch.linalg.norm(self.actions - self._prev_actions, dim=-1)
        # CLONED, not aliased: _reset_idx writes zeros into this buffer, and a
        # reference to `self.actions` would zero the action buffer itself.
        self._prev_actions = self.actions.clone()
        # D-168 instrument, on the RAW draw and before the clamp below --
        # after it every reading would be 1.0 by construction.
        abs_actions = self.actions.abs()
        self._max_action_abs = torch.maximum(
            self._max_action_abs, abs_actions.amax(dim=-1)
        )
        self._action_sat_sum += (abs_actions > 1.0).float().mean(dim=-1)
        self._action_steps += 1.0
        clamped = actions.clamp(-1.0, 1.0)
        if self._control_mode == "osc":
            # THE POSE DELTA of this control step (Decision (3)/(4)): metres and
            # axis-angle radians in the env frame, scaled by the per-step
            # limits. It is HELD for the whole decimation window and applied
            # afresh against the CURRENT pose at every physics step in
            # _apply_osc -- Factory's re-anchoring, which is the anti-windup:
            # the target is never more than one step limit away.
            self._osc_delta[:, 0:3] = clamped[:, 0:3] * float(self.cfg.osc_pos_step_limit_m)
            self._osc_delta[:, 3:6] = clamped[:, 3:6] * float(self.cfg.osc_rot_step_limit_rad)
            # No integrator under OSC, so no lag to record: the joint_pd
            # diagnostic below reads 0 here and says so in demo_metrics.
            return
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
        # Called once per PHYSICS step (DirectRLEnv.step, decimation loop),
        # which is exactly where the OSC has to run (Decision (4): "re-anchored
        # every physics step", factory_env.py:253-331).
        if self._control_mode == "osc":
            self._apply_osc()
            return
        self.robot.set_joint_position_target(self._joint_targets)

    def _build_task_space_controller(self) -> None:
        """Isaac Lab's OperationalSpaceController, configured per Decision (2).

        The settings are the gravity-ON unit test's
        (``test/controllers/test_operational_space.py:350-358``:
        ``disable_gravity=False``, ``gravity_compensation=True``,
        ``inertial_dynamics_decoupling=True``, partial decoupling off,
        ``impedance_mode="fixed"``, critical damping) with OUR gains, plus
        ``nullspace_control="none"``: six joints, six task axes, the null
        space is empty and the OSC raises on anything else
        (``operational_space.py:492-494``, Decision (5)).

        The controlled frame is the WELDED TOOL LINK (``peg_body_name``),
        the same body the scripted chain takes its Jacobian at; the leading
        tool point is ``peg_tip_offset`` along its +z. Without a welded tool
        the flange stands in, loudly (the startup report's MISSING line).
        The Jacobian row block is ``body - 1`` for a fixed-base articulation,
        the convention the OSC tutorial and the reset IK both use.
        """
        self._osc = None
        self._osc_delta = torch.zeros(self.num_envs, 6, device=self.device)
        self._osc_target_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._osc_efforts = torch.zeros(self.num_envs, len(self.robot.joint_names), device=self.device)
        self._osc_body_idx = self._peg_body_idx if self._peg_body_idx is not None else self._ee_body_idx
        self._osc_jacobi_idx = (
            self._osc_body_idx - 1 if self.robot.is_fixed_base else self._osc_body_idx
        )
        if self._control_mode != "osc":
            return
        kp = float(self.cfg.osc_kp_pos)
        kr = float(self.cfg.osc_kp_rot)
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True,
            motion_stiffness_task=[kp, kp, kp, kr, kr, kr],
            motion_damping_ratio_task=[float(self.cfg.osc_damping_ratio)] * 6,
            nullspace_control="none",
        )
        self._osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.device)

    def _task_space_dynamics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Jacobian of the controlled body in the ROOT frame, the joint-space
        mass matrix and the gravity-compensation torques, straight from PhysX.

        Read the way Isaac Lab's own OSC action term and tutorial read them
        (``task_space_actions.py:648-651``, ``run_osc.py:315-323``): the
        Jacobian comes out in the world frame and is rotated into the root
        frame block-wise. Also used by the startup report for Lambda and the
        Jacobian condition number, so both readings are the controller's own.
        """
        n_joints = len(self.robot.joint_names)
        view = self.robot.root_physx_view
        jacobian_w = view.get_jacobians()[:, self._osc_jacobi_idx, :, :n_joints]
        mass_matrix = view.get_generalized_mass_matrices()[:, :n_joints, :n_joints]
        gravity = view.get_gravity_compensation_forces()[:, :n_joints]
        jacobian_b = jacobian_w.clone()
        root_rot = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jacobian_b[:, :3, :] = torch.bmm(root_rot, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot, jacobian_b[:, 3:, :])
        return jacobian_b, mass_matrix, gravity

    def _apply_osc(self) -> None:
        """One physics step of the task-space controller.

        target = current pose + this step's delta, then the two clamps of
        Decision (3)/(4) -- the leading tool point inside the box around the
        pocket entrance, the tool axis inside the 8.52 deg cone -- then the
        OSC in the root frame, then joint EFFORTS into the (inert) drives.
        The clamps live in ``insertion_math`` and are checked offline; this
        method is frame bookkeeping and Isaac Lab calls.
        """
        pos_w = self.robot.data.body_pos_w[:, self._osc_body_idx]
        quat_w = self.robot.data.body_quat_w[:, self._osc_body_idx]
        tgt_pos_w, tgt_quat_w = insertion_math.apply_pose_delta(
            pos_w, quat_w, self._osc_delta[:, 0:3], self._osc_delta[:, 3:6]
        )
        # THE CONE IS MEASURED FROM THIS EPISODE'S POCKET AXIS (step B7), not
        # from the world vertical. `_fixture_quat` is the pocket orientation
        # in the ENV frame, and env origins are pure translations, so it is
        # the same orientation in world -- the fact this file states where the
        # buffer is declared. Under AutoDR the pocket tilts, and a
        # world-vertical cone would spend the CAD limit on merely aligning
        # with it.
        tgt_quat_w = insertion_math.clamp_tilt_to_cone(
            tgt_quat_w, float(self.cfg.osc_tilt_clamp_rad), self._fixture_quat
        )
        # Box centre: THIS episode's pocket entrance, in world (the entrance
        # buffer is env-local; the same sum the fixture pose write uses).
        centre_w = self._entrance_pos + self.scene.env_origins
        # ONE HALF-WIDTH PER AXIS (step B7). z is wider than x/y because it is
        # the axis the start band lies on: it must still hold one action step
        # above the highest legal start height. Sideways there was never a
        # shortage.
        tgt_pos_w = insertion_math.clamp_tip_in_box(
            tgt_pos_w,
            tgt_quat_w,
            self._tip_offset_local,
            centre_w,
            float(self.cfg.osc_pos_clamp_m),
            float(self.cfg.osc_pos_clamp_m),
            float(self.cfg.osc_pos_clamp_z_m),
        )
        self._osc_target_pose_w[:, 0:3] = tgt_pos_w
        self._osc_target_pose_w[:, 3:7] = tgt_quat_w

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, tgt_pos_w, tgt_quat_w)
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, pos_w, quat_w)
        # EE velocity relative to the root, expressed in the root frame
        # (run_osc.py:336-341).
        rel_vel_w = self.robot.data.body_vel_w[:, self._osc_body_idx, :] - self.robot.data.root_vel_w
        ee_vel_b = torch.cat(
            (quat_apply_inverse(root_quat_w, rel_vel_w[:, 0:3]), quat_apply_inverse(root_quat_w, rel_vel_w[:, 3:6])),
            dim=-1,
        )
        jacobian_b, mass_matrix, gravity = self._task_space_dynamics()
        self._osc.set_command(torch.cat((tgt_pos_b, tgt_quat_b), dim=-1))
        efforts = self._osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=torch.cat((ee_pos_b, ee_quat_b), dim=-1),
            current_ee_vel_b=ee_vel_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
        )
        self._osc_efforts = efforts
        self.robot.set_joint_effort_target(efforts)

    def _peg_geometry(self):
        """Per-env peg-tip position (env-local), insertion depth, pocket gate,
        out-of-arena flag, tool-axis alignment and yaw.

        AN INSTRUMENT, NOT THE TASK PATH (since M2.4b step 3). The reward,
        the termination and the observation all read ``insertion_math``;
        nothing in the step loop calls this any more. It is kept because two
        VERIFIED readers depend on it: ``seat_probe.py`` (RT-81/RT-82) takes
        its last element ``tip_rel``, and the startup report's single-env
        mirror prints its gate/corner diagnostics. Its four-corner gate and
        C4 yaw math are the SQUARE PROXY's and are labelled as such where
        they are printed; do not wire them back into reward or termination.

        Returns (tip_local (N,3), depth (N,), gate (N,) bool,
        below_plate (N,) bool, alignment (N,), phi (N,), cos4phi (N,),
        sin4phi (N,), tip_rel (N,3)). tip_rel is the tip offset from the
        pocket entrance IN THE POCKET FRAME (D-036); identical to
        tip_local - entrance while the pocket is untilted. seat_probe
        unpacks ``*_, tip_rel``, so tip_rel stays LAST.

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
        ``entrance_z - tip_z`` and keeps growing below the entrance, where it
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
        peg_axes = insertion_math.axes_from_quat(peg_quat_w)
        # peg_axes is (N, 3, 3), self._tip_offset_local is (3,): matmul promotes
        # the 1-D vector to a batched matrix-vector product and drops the
        # prepended dimension again, giving (N, 3) directly.
        tip_w = peg_pos_w + torch.matmul(peg_axes, self._tip_offset_local)
        tip_local = tip_w - origins

        # Pocket-frame transform (D-036). Depth is measured from the ENTRANCE
        # in both branches. In the proxy the plate carried the hole, so
        # entrance z WAS plate_top_z and `plate_top_z - tip_z` was the same
        # number; in the workcell "plate" was remapped to TABLE 2, the surface
        # the fixture STANDS ON, which sits 0.1823 m below the pocket opening
        # (MEASURED offline 2026-08-23 from the constants, and read straight
        # out of run RT-17: reported depth -0.340277 against a tip 0.157977 m
        # above the entrance, difference 0.182300). Measuring insertion depth
        # from the table top made every depth read 182.3 mm too deep and put
        # the 25 mm success threshold inside the table. With tilt, tip offset and
        # peg axes are rotated by R^T into the pocket frame first, so gate,
        # depth, alignment and phi all measure against the tilted pocket.
        # The reward's approach term takes tip_local and entrance unchanged:
        # it only uses their distance, which is rotation-invariant.
        if self._tilt_rot is None:
            tip_rel = tip_local - self._entrance_pos
            depth = -tip_rel[:, 2]
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
        # (the proxy's author_peg_ur10e.py, deleted 2026-08-28), so this IS the
        # orientation of the square's sides, not of its diagonal.
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
        """The REAL task's observation (D-107), minus its force block.

        Every block below comes from ``insertion_math``; nothing here does
        pose arithmetic of its own. The layout has one home,
        ``insertion_math.OBS_SLICES``, and ``scripts/check_env_wiring.py``
        pins the order built here against that table.

        WHAT CHANGED FROM THE PROXY, and why each one:

        * 12:15 is PART-anchored. ``part_tip_pose`` measures from the part's
          own task frame (``peg_tip_offset`` = ``FLANGE_TO_PART_BOTTOM``,
          D-069), not from the flange. What rides on that anchor is NOT a
          physical grasp error -- there is none; the part stays welded where
          PhysX put it. A per-episode BELIEF error of
          +-``grasp_obs_offset_x_m`` on the part's short axis is ADDED to the
          tool point HERE and nowhere else (``self._grasp_obs_off`` below),
          so the policy believes the part sits off by that much while the
          physics is unchanged (D-183, which corrects D-107 (4): the grasp
          offset is HIDDEN in ``tip_rel``, not visible to the policy).
        * 19:21 is ``(cos phi, sin phi)``, NOT ``(cos 4 phi, sin 4 phi)``.
          The proxy peg was square and four orientations fitted its pocket, so
          its encoding had to be C4-invariant. The real part fits in exactly
          ONE rotational position, so a C4 encoding would map four physically
          different states -- three of them impossible -- onto one input
          (D-107 (2)). ``cos4phi`` appears nowhere in this method any more.
        * 21:25 goes through ``canonicalize_quat`` here rather than relying on
          the reset write. Same value, but the ``w >= 0`` guarantee now sits
          in the checked function instead of in a comment about a distant
          write site.

        * 25:28 is the force block, added 2026-08-28 once RT-59 measured WHERE
          to read the wrench (``tool_link``). Three FORCE components; the
          wrench's three torque components are not handed to the policy
          (D-114, the FORGE pattern). It is EMA-smoothed at
          ``ft_smoothing_factor`` and it is UNROTATED -- the wrench is in the
          parent body frame and nothing here turns it, which is what FORGE
          effectively does too (inbox entry "The force wrench is in the parent
          body frame, not the world frame").

        ``_peg_geometry`` is no longer in the task path at all (M2.4b step 3):
        the reward and the termination read ``insertion_math`` /
        ``insertion_sdf`` in ``_get_dones``. It survives as an INSTRUMENT --
        seat_probe and the startup report's single-env mirror read it.
        """
        self._obs_calls += 1

        joint_pos = self.robot.data.joint_pos
        joint_vel_fd = (joint_pos - self._prev_joint_pos) / self._step_dt
        self._prev_joint_pos = joint_pos.clone()

        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        if self._peg_body_idx is not None:
            pocket_rot = self._tilt_rot if self._tilt_rot is not None else self._pocket_rot_identity
            # THE ONE PLACE THE GRASP OFFSET IS APPLIED (D-183). The nominal
            # offset plus this episode's belief error, so `tip_rel` is where
            # the policy THINKS its tool point is. `part_tip_pose` takes the
            # (N, 3) form here and the shared (3,) everywhere else; both go
            # through its one reshaped matmul. `x_axis` below is built from
            # the part quaternion and the pocket frame alone, so the yaw the
            # policy reads is untouched by this -- checked offline, not
            # assumed (check_insertion_math, the yaw_cos_sin claim).
            tip_rel, _depth, x_axis = insertion_math.part_tip_pose(
                self.robot.data.body_pos_w[:, self._peg_body_idx],
                self.robot.data.body_quat_w[:, self._peg_body_idx],
                self._tip_offset_local + self._grasp_obs_off,
                self.scene.env_origins,
                self._entrance_pos,
                pocket_rot,
            )
            yaw_cs = insertion_math.yaw_cos_sin(x_axis)
            # RT-171: Phi of the pose this observation shows -- the SAME
            # quaternion and pocket frame as tip_rel above -- becomes Phi(s_t)
            # for the next reward. Written on every call, so a reset
            # observation overwrites the finished episode's last value.
            self._tilt_phi_prev = insertion_math.tilt_potential(
                insertion_math.tilt_cos_theta(
                    self.robot.data.body_quat_w[:, self._peg_body_idx], pocket_rot
                ),
                float(self.cfg.w_tilt),
                float(self.cfg.kernel_a_tilt),
            )
        else:
            # No welded tool -- the same loud-but-running degradation the
            # startup report's MISSING line describes. (1, 0) is the aligned
            # yaw reading, matching the home pose.
            tip_rel = torch.zeros(self.num_envs, 3, device=self.device)
            yaw_cs = torch.zeros(self.num_envs, 2, device=self.device)
            yaw_cs[:, 0] = 1.0

        pocket_quat = insertion_math.canonicalize_quat(self._fixture_quat)

        # The force block (D-114). Read RAW at the measured link and smoothed
        # in place; the EMA state is the buffer, so this line is the only
        # writer of it and _reset_idx is the only clearer.
        if self._force_body_idx is not None:
            # ALL SIX COLUMNS (D-188): 0:3 force, 3:6 torque, one read, the
            # same parent body frame for both.
            wrench_raw = self.robot.data.body_incoming_joint_wrench_b[:, self._force_body_idx, 0:6]
            force_raw = wrench_raw[:, 0:3]
            torque_raw = wrench_raw[:, 3:6]
            # Tare BEFORE the EMA, so the smoothed buffer is a smoothed CONTACT
            # force and not a smoothed sum. Taring afterwards would be wrong
            # whenever the wrist turns during the EMA window.
            if self._force_parent_idx is not None:
                parent_quat_w = self.robot.data.body_quat_w[:, self._force_parent_idx]
                force_raw = insertion_math.gravity_tare(
                    force_raw,
                    parent_quat_w,
                    self._hold_force_w,
                    True,
                )
                # THE LEVER: the weld-AUTHORED centre of mass of the tool
                # (D-076..D-079, body_com_pos_w) minus the MEASURED reference
                # point of the moment: the parent link origin shifted by
                # cfg.torque_ref_offset_parent_m in the parent frame (RT-192a2:
                # 0.1032 m up the tool axis, NOT the origin the Isaac test
                # assumes). The offset is rotated with the parent quaternion
                # through the same axes convention as the tare.
                ref_w = (self.robot.data.body_pos_w[:, self._force_parent_idx]
                         + torch.matmul(insertion_math.axes_from_quat(parent_quat_w),
                                        self._torque_ref_offset_parent.unsqueeze(-1)).squeeze(-1))
                lever_w = self.robot.data.body_com_pos_w[:, self._force_body_idx] - ref_w
                torque_raw = insertion_math.gravity_tare_torque(
                    torque_raw,
                    parent_quat_w,
                    self._hold_force_w,
                    lever_w,
                    True,
                )
            # ONCE PER STEP, AND NOT ON THE RESET STEP (2026-09-13, see the
            # buffer comment in __init__). A second call in the same step
            # (the rsl_rl wrapper's get_observations) leaves the EMA alone;
            # an env that _reset_idx just touched keeps its reset zero, because
            # the wrench PhysX hands over here is still the old episode's.
            if int(self.common_step_counter) != self._wrench_step:
                self._wrench_step = int(self.common_step_counter)
                keep = ~self._wrench_fresh
                self._force_smooth = torch.where(
                    keep.unsqueeze(-1),
                    insertion_math.ema_update(
                        self._force_smooth, force_raw, float(self.cfg.ft_smoothing_factor)
                    ),
                    self._force_smooth,
                )
                self._force_tared_raw.copy_(
                    torch.where(keep.unsqueeze(-1), force_raw, torch.zeros_like(force_raw))
                )
                self._torque_smooth = torch.where(
                    keep.unsqueeze(-1),
                    insertion_math.ema_update(
                        self._torque_smooth, torque_raw, float(self.cfg.ft_smoothing_factor)
                    ),
                    self._torque_smooth,
                )
                self._torque_tared_raw.copy_(
                    torch.where(keep.unsqueeze(-1), torque_raw, torch.zeros_like(torque_raw))
                )
                self._wrench_valid.copy_(keep)
                self._wrench_fresh.fill_(False)
                # Tensor sums, no per-step GPU sync; read out in _torque_stats.
                self._wrench_valid_steps = self._wrench_valid_steps + keep.sum()
                self._wrench_steps += int(keep.numel())
        # else: no welded tool at all. The buffer stays at its reset zero,
        # exactly like tip_rel above -- the degradation is already loud in the
        # startup report and is not made quieter here.
        force = self._force_smooth
        torque = self._torque_smooth
        # Diagnosis instrument for the scripted-insertion gate, not part of
        # the task: the worst magnitude this episode reached.
        self._max_force_norm = torch.maximum(
            self._max_force_norm, insertion_math.force_magnitude(force)
        )
        self._max_torque_norm = torch.maximum(
            self._max_torque_norm, insertion_math.force_magnitude(torque)
        )
        # Episode record: the same maximum on the tared RAW force -- before
        # the EMA and before observation noise. It reads zero on the reset
        # step, like the EMA beside it (`_force_tared_raw` is zeroed there).
        self._max_force_raw_norm = torch.maximum(
            self._max_force_raw_norm, insertion_math.force_magnitude(self._force_tared_raw)
        )

        # The order IS insertion_math.obs_slices(mode): the seven blocks of
        # OBS_SLICES, plus `torque` APPENDED under `wrench` (D-188). Assembled
        # by cat rather than by index so there is no second place a channel
        # offset can be wrong. It is NOT insertion_math's own
        # assemble_observation: that function is checked offline against the
        # same table, and calling it here would hide the order rather than
        # state it. The env-wiring check compares BOTH cats against the table.
        if self._obs_mode == "wrench":
            obs = torch.cat(
                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force, torque),
                dim=-1,
            )
        else:
            obs = torch.cat(
                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force), dim=-1
            )

        if self._obs_calls in self._report_at_obs_calls:
            self._print_startup_report(joint_vel_fd)

        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The REAL termination (D-113): exactly two exits, and the per-step
        predicates the reward reads.

        Runs BEFORE _get_rewards() in DirectRLEnv's step loop, so everything
        the reward needs is computed here and cached forward, not the other
        way round. All heavy lifting is ``insertion_math`` /
        ``insertion_sdf``; this method is call sites.

        The exits (``insertion_math.compute_dones``):

        * ``truncated`` -- the timeout (D-113 (4)).
        * ``terminated`` -- the union of TWO terminal sources since 2026-09-01
          (inbox entry "Reward-Ueberarbeitung" 2026-09-01, point (7)): the
          force abort, read from the SAME EMA-smoothed force the policy
          observes (D-114) -- by the step order that buffer was last written in
          the previous step's ``_get_observations``, which is exactly the value
          the policy acted on -- and the SUCCESS, taken from the same per-step
          ``in_success_region`` predicate the display bonus is paid on. The
          reward tells the two apart and pays them differently, so
          ``compute_dones`` hands back the bare ``force_abort`` flag as well.
        * ``in_success_region`` is therefore computed BEFORE ``compute_dones``
          here. It used to come after; the order is load-bearing now, not
          cosmetic.

        The proxy's ``below_plate`` exit is still GONE (M2.4b step 3): that
        hole existed only in the proxy reward, which no longer pays anything
        down there. What came back is only the success exit, and not in the
        proxy's shape -- the proxy latched, this one fires on the D-052 band
        with the SAPU filter.

        With ``rl_terms_enabled = False`` (the measurement env) the force
        abort is OFF and no Warp query runs -- which is what the cfg
        docstring has promised since 2026-08-30. Depth and its metric are
        still computed: they need no Warp and the measurement scripts read
        the metrics file.
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self._peg_body_idx is None:
            terminated = torch.zeros_like(time_out)
            self._last_depth = torch.zeros(self.num_envs, device=self.device)
            self._last_terminated = terminated
            # Both new caches follow the same null-branch rule as the rest:
            # nothing terminates, nothing succeeds, nothing is owed.
            self._last_force_abort = terminated
            self._last_steps_remaining = torch.zeros(self.num_envs, device=self.device)
            self._last_depth_progress = torch.zeros(self.num_envs, device=self.device)
            self._last_tilt_phi = torch.zeros(self.num_envs, device=self.device)
            self._last_success = terminated
            return terminated, time_out

        pocket_rot = self._tilt_rot if self._tilt_rot is not None else self._pocket_rot_identity
        body_pos_w = self.robot.data.body_pos_w[:, self._peg_body_idx]
        body_quat = self.robot.data.body_quat_w[:, self._peg_body_idx]
        tip_rel, depth, _x_axis = insertion_math.part_tip_pose(
            body_pos_w,
            body_quat,
            self._tip_offset_local,
            self.scene.env_origins,
            self._entrance_pos,
            pocket_rot,
        )
        self._last_depth = depth
        # RT-171: the alignment potential of THIS step's pose, same body and
        # same pocket frame as tip_rel. The reward reads it as Phi(s_t+1).
        self._last_tilt_phi = insertion_math.tilt_potential(
            insertion_math.tilt_cos_theta(body_quat, pocket_rot),
            float(self.cfg.w_tilt),
            float(self.cfg.kernel_a_tilt),
        )
        # The lateral gate, computed ONCE and used three times: by the depth
        # metric here, and by both paying predicates below (D-157). It used to
        # live only here, inline -- which is precisely how RT-107 happened:
        # the metric knew the part was 109 mm beside the fixture and read 0.0,
        # while `is_engaged` and `in_success_region` saw only the axis
        # projection and paid for the descent anyway.
        cross_section = insertion_math.in_pocket_cross_section(
            tip_rel,
            insertion_tasks_cfg.POCKET_WALL_X,
            insertion_tasks_cfg.POCKET_WALL_Y,
        )
        # THE GATE HAS A FLOOR (2026-09-15, RT-201s3): a tip under the fixture
        # underside is not in the pocket, so it books no depth, no progress and
        # no engaged bonus. Reason and numbers: insertion_math.in_pocket_volume.
        in_pocket = insertion_math.in_pocket_volume(
            cross_section, depth, -insertion_tasks_cfg.POCKET_ASSET_BOTTOM_Z
        )
        self._below_fixture = self._below_fixture | (cross_section & ~in_pocket)
        prev_max_depth = self._max_depth
        self._max_depth = torch.maximum(
            prev_max_depth, torch.where(in_pocket, depth, torch.zeros_like(depth))
        )
        # The proxy's progress signal (D-165): metres of NEW gated max depth
        # this step. Read from the SAME buffer the depth metric reports, so the
        # term pays exactly what `mean_max_depth_mm` shows and nothing beside
        # the pocket (the D-157 gate is in the buffer). torch.maximum returns
        # a new tensor, so prev_max_depth is the old value, not an alias.
        self._last_depth_progress = self._max_depth - prev_max_depth

        # D-153 exposure, recorded on EVERY step, above the rl_terms_enabled
        # early return on purpose: a measurement env (terms off) is exactly
        # where the approach trajectory is inspected, and the exposure is a
        # property of the approach, not of the reward.
        #
        # tip_rel is the part's task frame (the LEADING face centre) in the
        # entrance-anchored pocket frame, and peg_tip_offset has no lateral
        # component, so tip_rel[:, 1] IS `part_y - pocket_y`. The z band is
        # stage 1: its floor is the entrance plane (tip_rel z = 0, which is
        # the hole's own plane) and its rim sits STAGE1_DEPTH above it.
        # ABOVE the rim the part is in free air and the hole is out of reach;
        # BELOW the floor the stage-2 walls hold it to PLAY_Y / 2 = 0.8 mm.
        in_stage1 = (tip_rel[:, 2] >= 0.0) & (
            tip_rel[:, 2] <= insertion_tasks_cfg.STAGE1_DEPTH
        )
        self._max_stage1_lat_y = torch.maximum(
            self._max_stage1_lat_y,
            torch.where(
                in_stage1, tip_rel[:, 1].abs(), torch.zeros_like(tip_rel[:, 1])
            ),
        )

        if not self._rl_terms_enabled:
            terminated = torch.zeros_like(time_out)
            self._last_terminated = terminated
            self._last_force_abort = terminated
            self._last_steps_remaining = torch.zeros(self.num_envs, device=self.device)
            self._last_depth_progress = torch.zeros(self.num_envs, device=self.device)
            # A measurement env pays no reward, so no potential either; the
            # obs-side buffer is left alone -- it is a reading, not a payment.
            self._last_tilt_phi = torch.zeros(self.num_envs, device=self.device)
            self._last_success = terminated
            return terminated, time_out

        # Part pose ENV-LOCAL, the frame of _entrance_pos and of the goal
        # pose. relative_pose subtracts the positions, so any shared frame
        # works -- but it must be the SAME one on both sides.
        part_pos = body_pos_w - self.scene.env_origins

        # Reward distance: the part's sampled points against the part's own
        # surface AT THE GOAL (D-109 (3)), via the T_goal^-1 . T_curr
        # substitution -- one static mesh, no refit.
        goal_pos, goal_quat = insertion_math.seated_goal_pose(
            self._entrance_pos,
            self._fixture_quat,
            self._seat_quat_local,
            self._tip_offset_local,
            float(self.cfg.depth_max),
        )
        rel_pos, rel_quat = insertion_sdf.goal_relative_transform(
            part_pos, body_quat, goal_pos, goal_quat
        )
        sdf_dist = self._sdf_query.mean_outside_distance(rel_pos, rel_quat)

        # SAPU (D-109 (10)): the same sampled part points against the FIXTURE
        # mesh, reduced by MAXIMUM. NAMED ASSUMPTION: the fixture OBJ is in
        # the entrance-anchored pocket frame (RT-91's G2 frame guard measured
        # its planes against the pocket constants at 0.0000 mm), so the
        # fixture's pose IS (_entrance_pos, _fixture_quat). The teleport
        # success identity test verifies this chain end to end.
        sapu_pos, sapu_quat = insertion_sdf.goal_relative_transform(
            part_pos, body_quat, self._entrance_pos, self._fixture_quat
        )
        interpen = self._sapu_query.interpen_distances(sapu_pos, sapu_quat)
        interpen_max = insertion_math.max_interpen_dist(interpen)
        # Episode running maximum, harvested by _log_finished_episodes and
        # zeroed in _reset_idx -- the _max_force_norm pattern.
        self._max_interpen_m = torch.maximum(self._max_interpen_m, interpen_max)
        self._interpen_unclean_steps += insertion_math.interpen_unclean(
            interpen_max, float(self.cfg.interpen_thresh)
        ).to(torch.int32)

        in_region = insertion_math.in_success_region(
            depth,
            interpen_max,
            float(self.cfg.depth_min),
            float(self.cfg.depth_max),
            float(self.cfg.interpen_thresh),
            in_pocket,
        )
        force_norm = insertion_math.force_magnitude(self._force_smooth)
        terminated, truncated, force_abort = insertion_math.compute_dones(
            self.episode_length_buf,
            int(self.max_episode_length),
            force_norm,
            float(self.cfg.force_abort_f_max_n),
            in_region,
        )
        # THE PAYOUT MULTIPLIER (point (7)): the steps this episode would still
        # have run. The last step an episode reaches is the one where
        # ``episode_length_buf == max_episode_length - 1`` (the truncation rule
        # one line above, unchanged), so the steps left AFTER this one are
        # ``max_episode_length - 1 - episode_length_buf``. With the step's own
        # task return paid as usual, the total is exactly what holding this
        # state to the old timeout would have paid. Clamped at zero so the last
        # step itself pays no negative lump.
        self._last_steps_remaining = torch.clamp(
            (int(self.max_episode_length) - 1) - self.episode_length_buf, min=0
        ).float()
        self._last_sdf_dist = sdf_dist
        self._last_interpen_max = interpen_max
        self._last_engaged = insertion_math.is_engaged(
            depth, float(self.cfg.engaged_depth_m), in_pocket
        )
        self._last_in_region = in_region
        self._last_terminated = terminated
        self._last_force_abort = force_abort
        # EPISODE SUCCESS = THE SUCCESS-TERMINATION EVENT (2026-09-01, point
        # (7)). It is the same tensor as ``in_region`` and that is the whole
        # change: the episode ends where the predicate fires, so "seated now"
        # and "seated at the end" are one statement and ``success_and_hold``
        # has nothing left to add. An episode that ends any other way -- force
        # abort or timeout -- resets with this False, because a step that had
        # ``in_region`` True would have terminated as a success instead.
        self._last_success = in_region
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """The REAL reward (D-109 / D-114), one call site.

        Everything it reads was cached by this step's ``_get_dones``. With
        ``rl_terms_enabled = False`` the reward is EXACTLY zero -- the
        measurement env, and since M2.4b step 3 the cfg docstring's promise
        is the code's behaviour. A missing welded tool also pays zero, loud
        in the startup report rather than silent here.
        """
        if self._peg_body_idx is None or not self._rl_terms_enabled:
            return torch.zeros(self.num_envs, device=self.device)
        # One row per insertion_math.REWARD_TERMS (2026-09-02). The rows are
        # accumulated per running episode and logged when it ends
        # (_log_finished_episodes); the reward paid is their sum, computed
        # HERE from the same rows so the log and the payment cannot drift.
        terms = insertion_math.compute_reward_terms_insertion(
            self._last_sdf_dist,
            self._last_interpen_max,
            self._last_engaged,
            self._last_in_region,
            self._last_force_abort,
            self._last_action_rate,
            self._last_steps_remaining,
            self._last_depth_progress,
            self._last_tilt_phi,
            self._tilt_phi_prev,
            float(self.cfg.kernel_a_coarse),
            float(self.cfg.kernel_b_coarse),
            float(self.cfg.kernel_a_mid),
            float(self.cfg.kernel_b_mid),
            float(self.cfg.kernel_a_fine),
            float(self.cfg.kernel_b_fine),
            float(self.cfg.w_engaged),
            float(self.cfg.w_success),
            float(self.cfg.w_depth_progress),
            float(self.cfg.interpen_thresh),
            float(self.cfg.abort_payment),
            float(self.cfg.time_penalty_per_step),
            float(self.cfg.action_rate_scale),
            float(self.cfg.shaping_gamma),
        )
        # A loop over the TERMS, not over envs; each row is batched.
        for i, name in enumerate(insertion_math.REWARD_TERMS):
            self._episode_sums[name] += terms[i]
        return torch.sum(terms, dim=0)

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

        # ONE index tensor for the whole reset. It was rebuilt three times
        # in this method before, once per block that needed it.
        idx = (
            env_ids if torch.is_tensor(env_ids)
            else torch.as_tensor(list(env_ids), device=self.device)
        )

        # THE DRAW. Before super()._reset_idx and before the first consumer,
        # so the order inside this method is: record the finished episode ->
        # draw -> apply. Nothing below this line draws a reset condition of
        # its own; `check_env_wiring.py` enforces that.
        if ready:
            self._draw_reset_conditions(idx)

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
            # The six joint-noise columns of THIS reset's draw, in joint
            # order. Same distribution as the `torch.rand_like` this
            # replaces; the difference is only that the numbers were drawn
            # up front with everything else.
            u_joint = self._reset_unit[idx][:, self._joint_noise_cols]
            joint_pos = joint_pos + (u_joint * 2.0 - 1.0) * self._reset_noise_scale
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        try:
            self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        except (AttributeError, TypeError) as exc:
            if not getattr(self, "_target_warning_printed", False):
                self._target_warning_printed = True
                print(f"[insertion] set_joint_position_target unavailable ({exc}); relying on USD drive targets")

        if self._control_mode == "osc":
            # Zero efforts in the reset step (run_osc.py:219): the last
            # episode's torques must not act on the freshly written pose, and
            # the held delta is cleared so the first physics step re-anchors
            # on the pose the reset wrote.
            try:
                self.robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=idx)
            except (AttributeError, TypeError) as exc:
                # Same guard as the position target above: the base class
                # may reset before the PhysX view exists.
                if not getattr(self, "_effort_warning_printed", False):
                    self._effort_warning_printed = True
                    print(f"[insertion] set_joint_effort_target unavailable at reset ({exc})")
            if ready:
                self._osc_delta[idx] = 0.0
                self._osc_efforts[idx] = 0.0
        if ready:
            self._joint_targets[env_ids] = joint_pos
            self._prev_joint_pos[env_ids] = joint_pos
            self._max_depth[env_ids] = 0.0
            self._last_depth_progress[env_ids] = 0.0
            self._max_target_lag[env_ids] = 0.0
            # Cleared, not carried: the FORGE convention for prev-action
            # buffers. See the buffer's own comment in __init__ for what the
            # first step of an episode therefore pays.
            self._prev_actions[env_ids] = 0.0
            self._last_action_rate[env_ids] = 0.0
            # The force EMA is cleared to ZERO rather than re-seeded with the
            # next sample (FORGE, forge_env.py:331). Two reasons and both
            # matter here: an episode must not inherit the previous episode's
            # contact force, and the very first reading after a reset is the
            # documented "stale values" case -- seeding with it would carry a
            # value nobody may use as evidence into every later step.
            self._force_smooth[env_ids] = 0.0
            self._force_tared_raw[env_ids] = 0.0
            # Raised here, consumed by the next _get_observations: that call
            # reads the OLD episode's wrench for these envs and must not feed
            # it into the EMA (buffer comment in __init__).
            self._wrench_fresh[env_ids] = True
            self._torque_smooth[env_ids] = 0.0
            self._torque_tared_raw[env_ids] = 0.0
            self._max_torque_norm[env_ids] = 0.0
            self._max_force_norm[env_ids] = 0.0
            self._max_interpen_m[env_ids] = 0.0
            self._below_fixture[env_ids] = False
            self._interpen_unclean_steps[env_ids] = 0
            self._max_force_raw_norm[env_ids] = 0.0
            self._max_stage1_lat_y[env_ids] = 0.0
            self._max_action_abs[env_ids] = 0.0
            self._action_sat_sum[env_ids] = 0.0
            self._action_steps[env_ids] = 0.0
            # THE GRASP OBSERVATION OFFSET (D-183): one number per resetting
            # env, held for the whole episode, mapped in Factory's own form
            # `(2*u - 1)*a` (factory_env.py:763-769). It lands HERE, before
            # this method returns and the base class asks for the reset
            # observation, so the episode's first observation already carries
            # this episode's belief error rather than the last one's.
            #
            # THE NUMBER COMES FROM THE TABLE ROW, column `grasp_obs_x`, like
            # every other random number this reset uses -- it is NOT drawn
            # here. Two reasons, and the second one is why no amplitude guard
            # may ever be added:
            #   * THE REPLAY MUST BE EXACT. The evaluation replays a row so
            #     that two policies meet the identical episode. The row is not
            #     a set of poses -- every column is a raw number in [0, 1]
            #     that its own consumer maps (`_draw_reset_conditions`). A
            #     belief error drawn beside the row would be re-drawn on every
            #     replay, and the two policies would not be compared under the
            #     same belief error.
            #   * "OFF" MUST BE OFF. A separate draw consumes a random number
            #     even at `grasp_obs_offset_x_m = 0.0`, so the RNG stream of a
            #     run with the offset disabled differed from the stream of the
            #     same run before D-183. A FIXED-width row consumes the same
            #     count whatever the magnitude is, so the stream no longer
            #     depends on it -- by construction, not by a guard. A guard
            #     `if offset != 0` would restore exactly the defect it looks
            #     like it fixes: "on" and "off" would consume different counts
            #     again.
            self._grasp_obs_off[idx, 0] = (
                2.0 * self._reset_unit[idx][:, self._col["grasp_obs_x"]] - 1.0
            ) * float(self.cfg.grasp_obs_offset_x_m)

        # Fixture-pose randomisation (D-034): sample a new pocket pose for the
        # envs being reset and teleport the kinematic fixture to it. Guarded
        # by `ready` (buffers exist) AND by the REACH -- can this run ever ask
        # for a pose off the base one. At no reach no write ever happens and
        # the env is bit-identical to pre-D-034, which is the commissioning
        # regression case. The entrance buffer and the prim pose are written
        # from the SAME tensor, so they can only disagree if the sim write
        # itself fails; the startup report prints both for the report envs,
        # which is the identity test.
        #
        # The reach, not the three cfg fields it used to read: under AutoDR
        # all three are 0.0 for the whole run, so the direct read skipped
        # this block from iteration 0 to the end and the pocket never moved.
        if ready and self._fixture_pose_reach:
            n = int(idx.numel())
            u = self._reset_unit[idx]
            entrance = self._entrance_base.unsqueeze(0).repeat(n, 1)
            u_xy = torch.stack(
                (u[:, self._col["fixture_noise_x"]], u[:, self._col["fixture_noise_y"]]),
                dim=-1,
            )
            entrance[:, :2] += (u_xy * 2.0 - 1.0) * self.cfg.fixture_pos_noise_xy
            # THE LIVE BANDS of the two pocket angles (Phase 5 step B5). Both
            # read the static cfg field until B5, and both of those fields are
            # refused non-zero under AutoDR -- so the pocket stayed upright
            # for the whole run whatever `dr/yaw_hi` and `dr/tilt_hi` said.
            # `fixture_pos_noise_xy` above is NOT one of the five: no boundary
            # tracks it, it stays a constant disturbance (plan section 1).
            _b = self._live_bounds()
            # Per-episode pocket yaw (D-038), uniform in +-range; composed
            # with the tilt below as quat_yaw * quat_tilt.
            _y_lo, _y_hi = _b["yaw"]
            yaw = autodr.map_unit_to_bounds(
                u[:, self._col["yaw"]], float(_y_lo), float(_y_hi)
            )
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
            # THE REACH, not today's band: this branch decides whether a tilt
            # quaternion is built at all, and under AutoDR the band is a point
            # at 0 for as long as the tilt boundary has not moved. A guard on
            # the current width would answer "no" until the first move and
            # -- since the cfg field it used to read is refused non-zero in
            # that mode -- for the whole run. Same failure the block guard
            # above already documents, one level down (plan risk R1).
            if self._tilt_reach:
                _t_lo, _t_hi = _b["tilt"]
                mag = autodr.map_unit_to_bounds(
                    u[:, self._col["tilt"]], float(_t_lo), float(_t_hi)
                )
                azimuth = u[:, self._col["tilt_azimuth"]] * (2.0 * math.pi)
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
                quat = insertion_math.quat_mul(quat_yaw, quat_tilt)
            else:
                quat = quat_yaw
            # q and -q are one rotation; the observation must encode each
            # orientation exactly once, so the sign is pinned to w >= 0. Within
            # the 15 deg cap w >= 0.991, so this never actually flips -- it is
            # a guarantee for the policy's input, not a live correction.
            quat = torch.where(quat[:, :1] < 0.0, -quat, quat)
            self._fixture_quat[idx] = quat
            if self._tilt_rot is not None:
                rot = insertion_math.axes_from_quat(quat)
                self._tilt_rot[idx] = rot
                self._tilt_rot_t[idx] = rot.transpose(1, 2)

            pose_w = torch.cat((entrance + self.scene.env_origins[idx], quat), dim=-1)
            self._fixture.write_root_pose_to_sim(pose_w, env_ids=idx)
            self._fixture.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=idx)

        # LAST, and the order is load-bearing: the start pose is solved
        # against THIS episode's pocket, so it has to run after the fixture
        # pose above was sampled and written. Solving first would aim the tip
        # at the previous episode's entrance and, at a start below the
        # opening plane, drive the part into a wall.
        if ready:
            # Map the start conditions into their per-env buffers FIRST;
            # the solver below only realises what these two say.
            self._map_start_conditions(idx)
            self._solve_start_pose(env_ids)

        # Contact friction, ONE value per env per episode (Phase 5 step B4).
        # Position in the method does not matter physically -- the material is
        # not part of the pose and no block above or below reads it -- but it
        # must come after `_draw_reset_conditions`, whose row it maps.
        if ready and self._friction_reach:
            self._apply_friction(idx)

    def _live_bounds(self) -> dict:
        """The band THIS reset draws from, one entry per ``autodr.DR_DIMS``
        name, in the quantity's own unit: ``{name: (lo, hi)}``.

        ONE SEAM, and step B5 exists largely to build it. Until B5 the
        consumers each decided where their band came from, and all but one
        decided wrongly: only the friction read the provider. On the
        PRE-D-178 table that was FIVE of the six QUANTITIES and therefore
        NINE of the eleven BOUNDARIES (lat_x 2, lat_y 2, yaw 2, tilt 1,
        start_height 2) reading the static cfg fields -- the very fields
        AutoDR mode REFUSES non-zero. The two counts are different numbers
        and plan section 1 exists to keep them apart. All nine could open to
        their maximum while every reset kept drawing the centre. `dr/tilt_hi` would climb, the pocket would stay upright, and
        nothing in the log would disagree.

        WITH A PROVIDER the answer is the provider's LIVE bounds, so a
        boundary that moved is drawn from at the very next reset.

        WITH NO PROVIDER (``dr_mode='off'``, every run before Phase 5) the
        answer is the static cfg fields. The mappings are chosen so that an
        'off' run keeps the distribution it had -- which is what keeps RT-177
        comparable:

        * ``map_unit_to_bounds(u, -h, +h)`` is ``-h + u*2h``, the same
          distribution as the old ``(2u-1)*h`` and the same two endpoints.
          NOT bit-identical: the two round differently, so an 'off' run
          repeats its distribution, not its exact trajectory. That holds for
          yaw.
        * tilt and start_height ARE bit-identical: ``0.0 + u*(m-0.0)`` and
          ``lo + u*(hi-lo)`` are the expressions those two already used.
        * the lateral offset does NOT repeat its old distribution, by
          decision: the radius band is ``(0.0, R)`` and ``disk_offset`` draws
          a DISK of radius R (D-178 (1)). An 'off' run at a non-zero
          ``start_lateral_offset`` draws a disk where the D-176 build drew a
          square band of that half width.

        FRICTION with no provider is a zero-width point at
        ``CONTACT_FRICTION``. That is not a placeholder -- it is exactly what
        the one-shot D-111 write put on both assets, and `_friction_reach` is
        False in that mode, so nothing draws from it anyway. The entry is here
        because a dict missing one of the five names is a KeyError waiting for
        whichever consumer runs first.

        CALLED ONCE PER CONSUMER PER RESET, so AT MOST three times and often
        fewer: `_map_start_conditions` always calls it, the fixture block only
        behind `_fixture_pose_reach` and `_apply_friction` only behind
        `_friction_reach`. A `dr_mode='off'` run with the cfg defaults has no
        friction reach, so it calls this TWICE.

        DELIBERATELY NOT CACHED on the instance. A cache would be a hidden
        channel between the draw and the consumers, and the ordering bug it
        invites -- a consumer reading the PREVIOUS reset's band -- is exactly
        the class of defect this seam exists to remove. The cost is a
        five-entry dict: under a provider `AutoDR.bounds()` walks the five
        dims and makes SEVEN `value()` calls, not ten, because `lat_r`,
        `tilt` and `start_height` are one-sided and take their lower edge
        straight from the centre. NOT
        MEASURED against the reset it sits in; what IS measured per reset is
        the friction write (`dr/friction_write_ms_*`).
        """
        _yaw = float(self.cfg.fixture_yaw_noise_rad)
        _tilt = float(self.cfg.fixture_tilt_noise_rad)
        # The home-pose setting (`start_tip_above_entrance = None`) has no
        # start height to open a band around: no solver runs, and the height
        # buffer only carries the home value __init__ seeded it with. Answered
        # with that same value rather than left to `float(None)`.
        _hi = self.cfg.start_tip_above_entrance
        _hi = self._start_tip_height if _hi is None else float(_hi)
        _lo = self.cfg.start_tip_above_entrance_low
        # `low >= high` IS the fixed rung-0 height: a zero-width band, which
        # maps to `high` for every draw. Same answer the old branch gave.
        _lo = _hi if _lo is None or float(_lo) >= _hi else float(_lo)
        _static = {
            # ONE-SIDED, like the DimSpec (D-178 (2)): the disk radius is drawn
            # from [0, R]; the angle is the separate `lat_phi` column.
            "lat_r": (0.0, self._start_lat_half),
            "yaw": (-_yaw, _yaw),
            # ONE-SIDED, like the DimSpec: the magnitude is drawn from
            # [0, max] and the direction is the separate azimuth column.
            "tilt": (0.0, _tilt),
            "start_height": (_lo, _hi),
            "friction": (
                float(insertion_tasks_cfg.CONTACT_FRICTION),
                float(insertion_tasks_cfg.CONTACT_FRICTION),
            ),
        }
        if self._dr is not None:
            # WITH A PROVIDER its live bounds win for every quantity it
            # carries. Under 'autodr' that is all five; under 'off' with a
            # start floor it is start_height alone, and the other four keep
            # the static bands above (SBC step 0, No-DR branch).
            _static.update(self._dr.bounds())
        return _static

    def _draw_reset_conditions(self, idx: torch.Tensor) -> None:
        """ONE uniform row per resetting env, before any of it is applied.

        Sixteen columns, `autodr.TABLE_COLUMNS`, values in [0, 1). This is
        the ONLY place a reset condition is drawn. Splitting the draw from
        the application is what lets a boundary env be nailed (write one
        column to 0 or 1) and what lets the evaluation replace the draw with
        a pre-made table row, without either touching a consumer.

        THE BOUNDARY SAMPLING (Phase 5 step B5) is the second half, and it is
        Akkaya et al. 2019 algorithm 1: with probability ``p_boundary`` this
        env is a BOUNDARY env -- one of the seven boundaries is picked, that
        quantity is nailed AT the boundary, and the other four stay uniform in
        their current bounds. Without it no env is ever a boundary env, no
        buffer ever fills, no boundary can move, and an "AutoDR" run is a
        fixed-width run at width 0 that nothing in the log tells apart from a
        real one.
        """
        if idx.numel() == 0:
            return
        n = int(idx.numel())
        self._reset_unit[idx] = torch.rand(
            n, len(autodr.TABLE_COLUMNS), device=self.device
        )
        if self._dr is None:
            return
        # TWO INDEPENDENT DRAWS, and deliberately NOT two more table columns.
        # Whether an env is a boundary env is not a reset CONDITION: the
        # evaluation replays the sixteen columns and has no boundary envs at
        # all, so a column here would be a column the table had to ignore.
        # Drawn only under a provider, which is also what leaves
        # `dr_mode='off'` on exactly the RNG stream it had before Phase 5.
        _rb = torch.rand(n, 2, device=self.device)
        _is_b, _which = self._dr.boundary_assignment(_rb[:, 0], _rb[:, 1])
        # -1 = REGULAR episode. Written for every resetting env, not only for
        # the boundary ones: an env that was a boundary env last episode must
        # not stay one, or its outcome would be booked onto that boundary's
        # buffer for the rest of the run.
        self._boundary_of[idx] = torch.where(
            _is_b, _which, torch.full_like(_which, -1)
        )
        # NAIL the picked column, in the table, before any consumer maps it.
        # MEASURED on the PRE-D-178 table (eleven boundaries, 2026-09-10), not
        # asserted, and not re-measured for the seven of D-178/D-179:
        # `map_unit_to_bounds` hit `lo` exactly for every reachable band, and
        # hit `hi` exactly in 465 of the 616 reachable pairs -- the other 151
        # landed within 2.98e-8 of the edge, which on that table's 6 mm
        # lateral band was 3e-10 m. See `autodr.map_unit_to_bounds` for the
        # table and for why the symmetric alternative is worse at both ends.
        # The lateral RADIUS does not go through that map any more:
        # `insertion_math.disk_offset` lands `u_r = 1` on the radius exactly.
        _rows = idx[_is_b]
        if _rows.numel() > 0:
            _sel = _which[_is_b]
            autodr.apply_boundary(
                self._reset_unit,
                _rows,
                self._boundary_cols[_sel],
                self._boundary_sides[_sel],
            )
        # THE STAMP (plan section 4). This episode is drawn from the bounds
        # that are current NOW. When a boundary moves later, the stamp goes
        # stale and `_log_finished_episodes` leaves the episode out of the
        # fresh window -- which is what stops the run from stopping on
        # evidence the old distribution produced.
        self._bounds_stamp[idx] = self._dr.bounds_version

    def _map_start_conditions(self, idx: torch.Tensor) -> None:
        """Map this reset's draw onto the START HEIGHT and LATERAL OFFSET.

        Both were drawn inside ``_solve_start_pose`` until Phase 5. They are
        mapped here so that the solver only REALISES a start pose that was
        already decided -- and so that a boundary env, whose column is
        already nailed at this point, lands exactly on its boundary.

        The height is IndustReal's Uniform[z_low, z_high] (sec. IV.G); the
        offset is uniform over the AREA of a disk (D-178 (1)), no longer the
        symmetric +-range of D-170. Both bands come from `_live_bounds` (step
        B5), so that an opened boundary is drawn from at the next reset. The `low >= high` branch this method used to carry is
        gone: a zero-width band maps to `high` for every draw, which is the
        same answer, and one branch fewer cannot disagree with the seam.
        """
        if idx.numel() == 0:
            return
        u = self._reset_unit[idx]
        _b = self._live_bounds()
        _h_lo, _h_hi = _b["start_height"]
        self._start_height[idx] = autodr.map_unit_to_bounds(
            u[:, self._col["start_height"]], float(_h_lo), float(_h_hi)
        )
        # ALWAYS WRITTEN, at width 0 too (plan risk R1). The old guard read
        # the cfg field and skipped the write when it was 0.0; under a bounds
        # provider that field stays 0.0 for the whole run while the boundary
        # opens, so the guard would answer "no" to every episode of an AutoDR
        # run. At radius 0 the row is zero: `disk_offset` scales both
        # components by `0.0 * sqrt(u_r)`.
        #
        # A DISK (D-178 (1)): the radius from the `lat_r` column, uniform over
        # the AREA, the angle from the `lat_phi` column, which has no boundary.
        # Only the UPPER edge of the live band enters -- `lat_r` is one-sided
        # and its lower edge is the centre 0.0 on the pocket axis. Reading
        # `self._start_lat_half` here instead would be the defect the
        # quantities had before B5: the cfg field, not the live boundary.
        self._start_lat_off[idx] = insertion_math.disk_offset(
            u[:, self._col["lat_r"]], u[:, self._col["lat_phi"]], float(_b["lat_r"][1])
        )

    def _apply_friction(self, idx: torch.Tensor) -> None:
        """ONE contact friction per env, redrawn at every reset (plan sec. 6).

        STATIC = KINETIC, on the ROBOT and on the FIXTURE, from the SAME
        draw. Both bodies of the contact therefore carry the identical
        coefficient. Under three of PhysX's four combine modes that makes the
        PAIR value equal to the drawn value: `average`, `min` and `max` all
        agree when the two inputs agree. Under the fourth, `multiply`, it does
        NOT -- 0.4 against 0.4 pairs to 0.16. The mode is a field of
        `PhysicsMaterialCfg` (`sim/spawners/materials/physics_materials_cfg.py:47`,
        `Literal["average", "min", "multiply", "max"]`, default `average`) and
        NOTHING in this repo sets it, on either asset. So the pair value is
        the drawn value by the DEFAULT, not by construction, and plan section
        6's "PhysX average" is an assumption about that default. Nobody has
        read the mode back out of the stage; that read is owed.

        WHAT IS BORROWED AND WHAT IS NOT, read out of the installed 2.3.2
        tree rather than remembered:

        * THE FORM is Factory's ``factory_utils.set_friction`` (lines 31-37):
          fetch the material buffer, write columns 0 and 1 with the SAME
          value, push it back. That is where static = kinetic comes from.
        * THE PARTIAL-RESET CALL is ``envs/mdp/events.py:283``.
        * THE PER-RESET SCHEDULE IS OURS. FORGE randomises friction at
          ``mode="startup"`` (``forge_env_cfg.py:51-85``), i.e. once for the
          whole run, and only its FIXED asset gets a band (static 0.25-1.25,
          dynamic pinned at 0.25 -- static != kinetic THERE). Its held asset
          and its robot get 0.75 / 0.75, so FORGE does tie the two
          coefficients where it does not band them; what it has no precedent
          for is redrawing EVERY EPISODE. That is the own construction, and
          the plan says the same: no published work does exactly this.
          Two details of FORGE's startup draw, read out of
          ``events.py:260-263`` rather than assumed: the draw is per SHAPE
          per env (``randint`` over ``(len(env_ids), total_num_shapes)``),
          not one value per env, and it picks from ``num_buckets`` presets.
          Note that FORGE's startup mode is exactly the fallback this step
          DELETED -- under AutoDR it would leave friction nominal for the
          whole run while the boundary opens.

        THE PARTIAL-RESET CALL. ``set_material_properties`` takes the WHOLE
        buffer plus the subset of env ids to push, not a buffer of the subset
        -- Isaac Lab 2.3.2 ``envs/mdp/events.py:283`` does exactly this in its
        own partial-reset path. So the cached buffer is edited in its own rows
        and handed over whole. Getting this backwards would not raise; it
        would write the first ``n`` envs' materials with the resetting envs'
        values, and every env would run a friction no log names.

        NOT VERIFIED IN A SIMULATOR. Isaac Sim is not installed on the
        laptop, so the call pair above was READ out of the installed 2.3.2
        tree and never executed. The startup report's per-env read-back is
        the evidence that has to arrive from the smoke run; until it does,
        this method is UNVERIFIED.

        THE COST. The timer spans the whole per-reset friction block -- the
        device-to-host copy, the row edit and both PhysX calls -- not the two
        calls alone. That is deliberate and it is the wider reading: the
        question the plan asks is whether the per-reset main path is
        affordable, and the copy and the row edit are part of what a reset
        pays. A number that covered only the two calls would understate it.
        """
        if idx.numel() == 0:
            return
        # The LIVE band, not the reach: the reach decides whether this method
        # runs at all (once, in __init__), the current bounds decide what the
        # episode actually gets. At width 0 that is the single point
        # CONTACT_FRICTION and every env draws it exactly.
        #
        # Through `_live_bounds` since step B5, like the other quantities. It is the
        # same answer here -- this method only runs under a provider -- but
        # ONE seam is the point: "does every randomised quantity read the live
        # band" is then a question with a single place to look.
        lo, hi = self._live_bounds()["friction"]
        mu = autodr.map_unit_to_bounds(
            self._reset_unit[idx][:, self._col["friction"]], float(lo), float(hi)
        )
        # What we COMMANDED. Read by the startup report, which prints it
        # beside the PhysX read-back, and since step B5 by
        # `_log_finished_episodes`, which pairs it with the episode outcome
        # (`dr/friction_mean`). Written here and not after the pushes for one
        # reason only: this is where `mu` is, and a failed push raises rather
        # than returning, so no ordering of these two lines makes the pair
        # atomic.
        self._friction_applied[idx] = mu

        # THE TIMER IS FENCED FIRST, and that fence is load-bearing on a CUDA
        # device. `idx.to("cpu")` is a BLOCKING device-to-host copy: it cannot
        # return until the stream drains, and by this point `_reset_idx` has
        # already queued the joint-state write, the position target, the start
        # -pose solve and (under a fixture reach) the two fixture writes. With
        # no fence the first copy inside the timer absorbs the tail of ALL of
        # that, and `friction_write_ms` reports the reset instead of the
        # friction write -- on the training PC only, because the same code on
        # a CPU device is honest. That number is what plan section 6 uses to
        # decide whether this path survives, so it must not be inflated by
        # work that is not ours.
        if mu.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx_cpu = idx.to(device="cpu", dtype=torch.long)
        # ONE host copy of the drawn column, not one per asset. Both buffers
        # come from the same API and their dtypes were checked equal at cache
        # time, so a second copy would only pay the D2H cost twice -- and it
        # would pay it inside the very measurement the plan reads.
        _col = mu.to(device="cpu", dtype=self._friction_buf_dtype).unsqueeze(1)
        for _asset, _buf in self._friction_bufs:
            _buf[idx_cpu, :, 0] = _col  # static
            _buf[idx_cpu, :, 1] = _col  # dynamic
            # THE WHOLE ROW GOES, all three columns. Column 2 is restitution:
            # this env never randomises it, so what travels is the value the
            # asset was authored with, snapshotted at cache time. Consequence,
            # stated rather than discovered: a future writer of restitution
            # would be silently reverted at the next reset of that env unless
            # it writes into THIS buffer.
            _asset.root_physx_view.set_material_properties(_buf, idx_cpu)
        self._friction_write_ms.append((time.perf_counter() - t0) * 1000.0)

    def _solve_start_pose(self, env_ids: Sequence[int]) -> None:
        """Teleport the resetting envs so the leading tool point starts at
        ``cfg.start_tip_above_entrance`` on this episode's pocket axis.

        WHY (D-161, RT-119): at the home pose the tool point stands 165 mm
        above the opening plane and the whole D-109 reward is 3.05e-23. PPO
        cannot tell that from zero. The kernel widths are not touched -- they
        are solved from measured tolerances, D-109 (9) -- so the START moves
        into the band where the gradient is readable.

        THE SOLVER IS NOT NEW. It is the dls loop of
        ``scripts/check_seated_success.py`` ``_solve_to``, the one RT-119
        drove to eight heights with a residual under 0.001 mm.

        THE GOAL IS THE POCKET AXIS PLUS ``cfg.start_lateral_offset``
        (D-170). Until 2026-09-06 the lateral goal was hard zero, and the
        docstring called an off-axis goal "a counter-proof, not a start
        pose". That was wrong in one direction it never checked: this solve
        runs LAST in ``_reset_idx``, so a hard zero CANCELS the lateral part
        of D-034's fixture noise and of ``reset_joint_noise`` -- every
        episode started perfectly centred whatever those were set to
        (RT-160: worst residual 0.0354 mm over 17780 episodes). D-162
        decided the HEIGHT and never mentioned the side. At the 0.0 default
        nothing changes; a run opts in per hydra.

        NO SIM STEP. Each iteration writes the joint state and reads the tip
        and the Jacobian straight back, which is why this may run in a
        PARTIAL reset. Factory's reset IK steps the sim
        (``factory_env.py`` ``set_pos_inverse_kinematics`` ->
        ``step_sim_no_action``) and its own docstring restricts it to resets
        where every env resets together; ours has no such restriction and
        must not grow one.

        The command carries a ZERO rotation delta, so the tool orientation
        the reset just wrote is preserved -- including ``reset_yaw_noise``
        when it goes back on.
        """
        if self._start_ik is None:
            return
        idx = torch.as_tensor(list(env_ids), device=self.device) if not torch.is_tensor(env_ids) else env_ids
        if idx.numel() == 0:
            return
        # THE START HEIGHT AND THE LATERAL OFFSET ARE NOT DRAWN HERE any
        # more (Phase 5, plan section 10). `_map_start_conditions` wrote
        # `_start_height` and `_start_lat_off` from this reset's single
        # draw before this method was called; this method only realises
        # them. A draw here would be a second draw for the same episode and
        # would ignore a nailed boundary column.
        tol = float(self.cfg.start_pose_solve_tol_m)
        n_joints = len(self.robot.joint_names)
        # Canonicalised the same way the observation's pocket_quat is: the
        # command frame and the frame the policy is told about must be one.
        pocket_quat = insertion_math.canonicalize_quat(self._fixture_quat)
        err_m = 0.0
        used = 0
        with torch.no_grad():
            for step in range(int(self.cfg.start_pose_solve_steps)):
                *_, tip_rel = self._peg_geometry()
                d_pocket = torch.zeros(self.num_envs, 6, device=self.device)
                # x and y carry the D-170 lateral offset; at the 0.0 default
                # the buffer is zeros and this is the bare -tip_rel again.
                d_pocket[:, 0] = self._start_lat_off[:, 0] - tip_rel[:, 0]
                d_pocket[:, 1] = self._start_lat_off[:, 1] - tip_rel[:, 1]
                d_pocket[:, 2] = self._start_height - tip_rel[:, 2]
                # The residual is judged on the RESETTING envs only. The other
                # envs are mid-episode; their rows are computed because the
                # controller is batched over the whole scene, and they are
                # never written back.
                err_m = float(torch.linalg.norm(d_pocket[idx, 0:3], dim=-1).max().item())
                used = step
                if err_m < tol:
                    break
                root_pose_w = self.robot.data.root_pose_w
                cmd_base = scripted_policy.command_to_base_frame(
                    d_pocket, pocket_quat, root_pose_w[:, 3:7]
                )
                ee_pose_w = self.robot.data.body_pose_w[:, self._peg_body_idx]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_idx, :, :n_joints]
                self._start_ik.set_command(cmd_base, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
                joint_pos_des = self._start_ik.compute(
                    ee_pos_b, ee_quat_b, jacobian, self.robot.data.joint_pos
                ).clone()
                sub = joint_pos_des[idx]
                self.robot.write_joint_state_to_sim(sub, torch.zeros_like(sub), None, idx)
                try:
                    self.robot.set_joint_position_target(sub, env_ids=idx)
                except (AttributeError, TypeError):
                    pass
                # The integrator starts from where the arm actually IS, not
                # from the home pose: _pre_physics_step adds the action delta
                # to _joint_targets, and a stale target would snap the arm
                # back to 165 mm on the first step of the episode.
                self._joint_targets[idx] = sub
                self._prev_joint_pos[idx] = sub

            # SEED THE GATED MAX DEPTH WITH THE START DEPTH. _reset_idx zeroed
            # _max_depth before this solve; an episode that starts h below the
            # plane would otherwise book |h| of "new" depth on its first step
            # and D-165 would pay w * |h| for the teleport, every episode. Read
            # from the pose the solve actually reached, not from the command,
            # so the seed is exactly what the depth metric would measure. The
            # part is on the pocket axis here (the solve's own goal), so the
            # D-157 lateral gate is satisfied and the seed is a gated depth.
            # THE ON-AXIS CLAIM HOLDS BY GUARD, NOT BY LUCK (D-170): a lateral
            # offset is refused together with a negative
            # start_tip_above_entrance_low (__init__), so whenever the offset
            # is non-zero every start is ABOVE the plane, tip_rel z > 0, and
            # the clamp below seeds 0 -- no gated depth is claimed off-axis.
            # Price: mean_max_depth_mm now includes the start depth of the
            # inside-start episodes -- read it against the logged start height.
            *_, tip_rel = self._peg_geometry()
            self._max_depth[idx] = (-tip_rel[idx, 2]).clamp(min=0.0)

            # AN UNCONVERGED SOLVE MAY NOT BOOK AN AutoDR EPISODE (user
            # decision 2026-09-10). The episode did not start where the
            # boundary says it did, so a success or a failure recorded for it
            # is a measurement of a start that never happened -- and Akkaya's
            # buffer would move a real edge on it.
            #
            # PER ENV, and that is the point: `err_m` above is the MAX over
            # the whole resetting batch and one global counter, so until now
            # no env was even identifiable. Measured from the pose the solve
            # ACTUALLY reached (the `tip_rel` two lines up), not from the
            # loop's last `d_pocket`, which is one write stale whenever the
            # iteration budget runs out.
            _goal = torch.zeros(self.num_envs, 3, device=self.device)
            _goal[:, 0] = self._start_lat_off[:, 0]
            _goal[:, 1] = self._start_lat_off[:, 1]
            _goal[:, 2] = self._start_height
            _bad = torch.linalg.norm(_goal[idx] - tip_rel[idx, 0:3], dim=-1) >= tol
            _n_bad = int(_bad.sum().item())
            if _n_bad:
                # BOTH BOOKINGS GO, not just the boundary flag. Dropping the
                # flag alone would turn the episode into a REGULAR one and
                # feed it to the stop rule's fresh window instead -- the same
                # wrong start, counted somewhere else. `bounds_version` starts
                # at 0 and only rises, so -1 can never match it.
                _rows = idx[_bad]
                self._boundary_of[_rows] = -1
                self._bounds_stamp[_rows] = -1
                self._start_solve_flags_dropped += _n_bad
            # `_recent_successes` and the bins are deliberately NOT touched:
            # that window is D-037's and covers every episode the run ran,
            # which these did. The count above is what says how big the
            # effect is.

        self._start_solve_residual_mm = err_m * 1000.0
        self._start_solve_iters = used + 1
        self._start_solve_worst_mm = max(self._start_solve_worst_mm, self._start_solve_residual_mm)
        if err_m >= tol:
            self._start_solve_unconverged += 1
            if self._start_solve_unconverged == 1:
                print(f"[insertion] START POSE SOLVE DID NOT CONVERGE: residual "
                      f"{self._start_solve_residual_mm:.4f} mm after "
                      f"{self._start_solve_iters} iterations, tolerance "
                      f"{tol * 1000.0:.3f} mm. The episode still runs, but it does "
                      f"NOT start where the config says. Counted in "
                      f"start_pose_solve_unconverged_resets (demo_metrics.json); "
                      f"this warning prints once.")

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
        # Since 2026-09-01, _last_success is the SUCCESS-TERMINATION event
        # (inbox entry "Reward-Ueberarbeitung", point (7)) and no longer
        # success_and_hold's "seated at the last step". The reading of the
        # metric is unchanged for every other episode ending: a force-aborted
        # or timed-out episode resets with it False, which is the intent -- an
        # abort is not a success however deep it got. What changed is that a
        # successful episode is now SHORTER than the cap, so
        # ``_recent_success_steps`` below became the time-to-success rather
        # than a constant 255.
        #
        # ``aborts`` reads the FORCE ABORT alone, not _last_terminated: since
        # the success is a second terminal source, the union would book every
        # success as an abort as well and the D-113 (9) abort rate would read
        # ~1.0 on a solved task.
        successes = self._last_success[idx]
        depths = self._max_depth[idx]
        aborts = self._last_force_abort[idx]
        n = int(idx.numel())

        # -- AutoDR, algorithm 1's bookkeeping (Phase 5 step B5) -------------
        # HERE and nowhere else, because this is the one place that sees an
        # episode's OUTCOME beside the conditions it ran under. The same
        # ordering the tilt bins rely on carries it: `_boundary_of` and
        # `_bounds_stamp` still hold the FINISHED episode's values, because
        # `_draw_reset_conditions` overwrites both further down this reset.
        if self._dr is not None:
            _b_of = self._boundary_of[idx]
            _is_b = _b_of >= 0
            # ONE FLAG PER FINISHED BOUNDARY EPISODE, into that boundary's own
            # buffer. `record` is a scalar python call, so a loop is
            # unavoidable -- but the loop must not TOUCH THE DEVICE, and that
            # is the whole point of the two `tolist()` calls below.
            #
            # THE FORM THIS REPLACES, and why (rubric critic, round 1): the
            # loop indexed `_b_of[_j]` and `successes[_j]` INSIDE the body, so
            # every iteration did `int()` and `bool()` on a 0-dim CUDA tensor,
            # and each of those is a blocking device-to-host sync. At the
            # plan's 1024 envs an early synchronised reset ends every episode
            # at once: about 512 boundary envs, hence about 1024 syncs inside
            # ONE `_reset_idx`. Nothing raises, no number is wrong, and no log
            # line reports it -- `dr/friction_write_ms_*` fences a different
            # block. It also broke this repo's own rule against per-env python
            # loops in the hot path. TWO host copies now, both batched, and
            # the loop runs on plain python ints and bools.
            _b_keys = _b_of[_is_b].tolist()
            _b_flags = successes[_is_b].tolist()
            for _k, _f in zip(_b_keys, _b_flags):
                self._dr.record(self._boundary_keys[_k], bool(_f))
            self._recent_boundary_successes.extend(successes[_is_b].tolist())
            # THE FRESH WINDOW: regular AND stamped with the current version.
            _fresh = (~_is_b) & (self._bounds_stamp[idx] == self._dr.bounds_version)
            self._fresh_successes.extend(successes[_fresh].tolist())
            # UPDATE AFTER RECORD, never before. A buffer that filled in THIS
            # pass has to be processed in this pass; deferring it to the next
            # one leaves the flags sitting one pass longer and `record` starts
            # dropping the excess (`dr/flags_dropped`).
            #
            # AND ONCE PER RESET CALL, which is finer than the plan's "once
            # per logging pass" and deliberately so. It is the closest this
            # split can get to algorithm 1, which tests the buffer after every
            # single append, and it drives `dr/flags_dropped` toward 0. The
            # cost is one pass over the seven boundaries per reset call.
            _events = self._dr.update()
            for _ev in _events:
                # One marked line per PROCESSED buffer, moved or not: a full
                # buffer that held is as much a result as one that moved.
                # MEASURED 2026-09-10, because the prefix is only useful if it
                # survives the trip: `filter_rt_log.NOISE_RE` and `ANOMALY_RE`
                # both MISS this line, so `shorten_rt_log.py` keeps it under
                # its "every other line that is not known noise" rule. It is
                # NOT true that `/rt-check` greps for the prefix -- that skill
                # forbids grepping the log file at all; the prefix is for a
                # human reading the pasted log.
                print(_ev.line())
            if any(_ev.moved for _ev in _events):
                # The distribution just changed. Every episode still in the
                # window was drawn from the OLD one, so the window is void.
                # Only a REAL move gets here -- a clamped attempt writes an
                # event but does not move, or the window would be wiped for
                # nothing and the run could never reach the stop rule (R12).
                self._fresh_successes.clear()
        # The friction the finished episodes ran under. Outside the provider
        # branch on purpose: `_friction_applied` carries the nominal
        # CONTACT_FRICTION when nothing randomises it, which is a true
        # statement about the run and keeps the deque index-aligned with
        # `_recent_successes` in every mode.
        self._recent_friction.extend(self._friction_applied[idx].tolist())

        self._ep_count += n
        self._ep_success_count += int(successes.sum().item())
        self._ep_depth_sum += float(depths.sum().item())
        self._recent_successes.extend(successes.tolist())
        self._recent_depths.extend(depths.tolist())
        # The invariant, checked on exactly the two arrays it is about, right
        # where they are already sliced. See the counter's comment in __init__.
        violations = successes & (depths <= 0.0)
        self._ep_success_depth_violations += int(violations.sum().item())
        self._recent_success_depth_violations.extend(violations.tolist())
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
        # Same ordering dependency again: _start_height still holds the height
        # the ending episode was commanded to; _solve_start_pose overwrites it
        # for the next episode further down this reset.
        self._recent_start_height_m.extend(self._start_height[idx].tolist())
        # The SAME ordering dependency as the two lines above, and it is
        # load-bearing for both new tables: _start_lat_off is redrawn in
        # _solve_start_pose and _fixture_yaw in the fixture-noise block, and
        # BOTH run later in this reset than this method does. So both buffers
        # still hold the values the episode being logged actually ran under.
        # Moving this call below either block would pair every success with
        # the NEXT episode's offset and yaw.
        self._recent_lateral_mm.extend(
            (torch.linalg.norm(self._start_lat_off[idx], dim=-1) * 1000.0).tolist()
        )
        self._recent_yaw_deg.extend(
            torch.rad2deg(self._fixture_yaw[idx].abs()).tolist()
        )
        self._recent_target_lag.extend(self._max_target_lag[idx].tolist())
        self._recent_force_norm.extend(self._max_force_norm[idx].tolist())
        self._recent_torque_norm.extend(self._max_torque_norm[idx].tolist())
        self._recent_interpen_max.extend(self._max_interpen_m[idx].tolist())
        self._recent_below_fixture.extend(self._below_fixture[idx].tolist())
        # Episode record (2026-09-15): one CSV row per episode ending now. HERE
        # for the ordering reason of every line above: the buffers still hold
        # the finished episodes' values.
        self._append_episode_rows(idx, successes, aborts)
        # D-168: harvested HERE, before the reset below zeroes the buffers,
        # exactly like the force line above. clamp_min(1.0) on the step count
        # guards the division only -- an episode ending now has run at least
        # one _pre_physics_step, so it never binds in practice and never
        # hides a partial failure behind a silent 0.
        self._recent_action_abs_max.extend(self._max_action_abs[idx].tolist())
        self._recent_action_sat_frac.extend(
            (self._action_sat_sum[idx] / self._action_steps[idx].clamp_min(1.0)).tolist()
        )
        self._recent_aborts.extend(aborts.tolist())
        self._recent_stage1_lat_y.extend(self._max_stage1_lat_y[idx].tolist())
        # Per-term reward sums of the episodes ending now (2026-09-02): into
        # the trailing window, onto the curve, then CLEARED for the next
        # episode of these envs. _get_rewards ran before this reset
        # (direct_rl_env step order), so the ending episode's terminal lump
        # or abort payment is inside its sum.
        term_means: dict = {}
        for name in insertion_math.REWARD_TERMS:
            sums = self._episode_sums[name][idx]
            self._recent_reward_terms[name].extend(sums.tolist())
            term_means[name] = float(sums.mean())
            self._episode_sums[name][idx] = 0.0

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
            # D-113 (9): the force-abort rate is its own number beside the
            # success rate, never folded into it (D-051 keeps force out of
            # the success definition).
            "force_abort_rate": (
                sum(self._recent_aborts) / len(self._recent_aborts)
                if self._recent_aborts
                else 0.0
            ),
            # D-153 exposure, live on the TensorBoard curve. The full spread
            # is in demo_metrics.json; the MAXIMUM is what belongs on a curve,
            # because the question is whether it was ever reached at all.
            "stage1_lat_y_max_mm": (
                max(self._recent_stage1_lat_y) * 1000.0
                if self._recent_stage1_lat_y
                else 0.0
            ),
            # D-157 tripwire, on the curve so it is visible DURING the run and
            # not only in the dump. Must read 0.0 for the whole run.
            "success_depth_invariant_violations": float(
                sum(self._recent_success_depth_violations)
            ),
            # RT-201s3 tripwire (2026-09-15): share of the window's episodes
            # whose tip went BELOW the fixture underside inside the pocket
            # cross-section. The fixture stands free (workcell block OFF),
            # and RT-201s3r caught the true tip 123.8 mm beside the entrance
            # and 70.8 mm below it: the clamp box bounds the OSC TARGET, not
            # the tip, so the part goes AROUND the fixture and under it. A
            # non-zero value is that path, never a seat. Since the same day
            # that depth is no longer paid.
            "below_fixture_rate": (
                sum(self._recent_below_fixture) / len(self._recent_below_fixture)
                if self._recent_below_fixture
                else 0.0
            ),
            # Plan step C: the mean commanded start height of the episodes
            # ending now, mm above the opening plane. A flat +30.0 is the
            # fixed rung-0 start; under sampling it hovers at the range's
            # centre and says on the curve which distribution the run had.
            "start_height_mean_mm": float(self._start_height[idx].mean()) * 1000.0,
            # D-168 tripwire pair. action_rate_scale dropped 0.1 -> 0.0034, so
            # the raw command is now nearly unpriced. If action_sat_frac climbs
            # while sigma looks healthy, the exploration is dead inside the
            # clamp and the D-168 verdict is wrong.
            "action_abs_max_mean": (
                sum(self._recent_action_abs_max) / len(self._recent_action_abs_max)
                if self._recent_action_abs_max
                else 0.0
            ),
            "action_sat_frac": (
                sum(self._recent_action_sat_frac) / len(self._recent_action_sat_frac)
                if self._recent_action_sat_frac
                else 0.0
            ),
            # PLAN SECTION 6, the affordability question. Mean and MAX of the
            # per-reset friction write, milliseconds, over the last 1000 reset
            # calls. Both, not the mean alone: the mean says what the run
            # pays on average and the max says whether one reset can stall a
            # step. Read them against the iteration time rsl_rl prints in the
            # SAME log -- a millisecond figure on its own says nothing about
            # whether the main path is affordable.
            #
            # A run with no friction reach never calls `_apply_friction`, so
            # both read 0.0. That is "not written", not "free".
            "dr/friction_write_ms_mean": (
                sum(self._friction_write_ms) / len(self._friction_write_ms)
                if self._friction_write_ms
                else 0.0
            ),
            "dr/friction_write_ms_max": (
                max(self._friction_write_ms) if self._friction_write_ms else 0.0
            ),
        }
        # Episode_Reward/<term>: the mean per-episode SUM of each reward row
        # over the episodes ending now, one TensorBoard curve per term.
        for name, value in term_means.items():
            self.extras["log"]["Episode_Reward/" + name] = value
        self.extras["log"]["Episode_Reward/total"] = float(sum(term_means.values()))

        # Solver-accuracy instrument (inbox 2026-09-12). Per-episode WORST
        # interpenetration over the recent window: the mean says what a
        # typical episode pays, the max whether the collider ever gave way.
        # Both in mm. Judge env-count and solver-iteration changes on these
        # two curves, never on the reward.
        #
        # CONDITIONAL ON PURPOSE. With ``rl_terms_enabled = False`` the SAPU
        # query never runs (the early return in ``_get_dones``), so the buffer
        # stays at its reset zero. Writing that zero onto a curve would read
        # as "no interpenetration" when it means "not measured". The keys are
        # therefore ABSENT in a measurement run, which cannot be misread.
        if self._rl_terms_enabled and self._recent_interpen_max:
            self.extras["log"]["interpen_max_mean_mm"] = (
                sum(self._recent_interpen_max) / len(self._recent_interpen_max) * 1000.0
            )
            self.extras["log"]["interpen_max_max_mm"] = (
                max(self._recent_interpen_max) * 1000.0
            )

        # Force instrument (user, 2026-09-14, after RT-205). Per-episode
        # MAXIMUM of the smoothed force the abort reads, over the same recent
        # window as demo_metrics.json's force_norm_n -- as a curve, so the
        # force over training time reads off scalars.csv instead of a
        # checkpoint replay series. Mean, p95 and max in N; p95 with the index
        # rule of _force_stats. Not gated: like _force_stats, the buffer is
        # filled in every mode.
        if self._recent_force_norm:
            _fv = sorted(self._recent_force_norm)
            self.extras["log"]["force_max_mean_n"] = sum(_fv) / len(_fv)
            self.extras["log"]["force_max_p95_n"] = _fv[
                min(len(_fv) - 1, int(math.ceil(0.95 * len(_fv))) - 1)
            ]
            self.extras["log"]["force_max_max_n"] = _fv[-1]

        # -- the AutoDR curves (Phase 5 step B5, plan section 10) ------------
        # MERGED, not assigned: the dict literal above is the env's own log
        # and these are added to it. Under a provider only, so a
        # `dr_mode='off'` run keeps exactly the curves it had -- except
        # `dr/friction_write_ms_*`, which the literal above already carries
        # and which read 0.0 there.
        if self._dr is not None:
            # The seven boundary VALUES, the seven FILL levels,
            # `bounds_version`, `all_at_max` and `flags_dropped`, all from the
            # provider's own `scalars()`. The env does not restate any of them
            # -- one home per fact, and a copy here would go stale on resume.
            self.extras["log"].update(self._dr.scalars())
            _fresh_n = len(self._fresh_successes)
            # THE STOP RULE'S RATE (plan section 4). NOT a test rate: these
            # episodes carry PPO's exploration noise, and the plan forbids
            # calling this number a success rate of the policy.
            #
            # -1.0 IS THE "TOO FEW EPISODES" SENTINEL. A TensorBoard scalar
            # cannot carry the string the plan writes, and 0.0 would read as
            # "it fails everything" -- the one value that must never be
            # confused with an empty window, because the stop rule tests
            # `>= 0.80` against it. Every real rate is in [0, 1], so a
            # negative number cannot collide with one. `dr/fresh_n` beside it
            # says how full the window is, and the train-side rule must test
            # BOTH.
            self.extras["log"]["dr/train_success_regular_fresh"] = (
                sum(self._fresh_successes) / _fresh_n
                if _fresh_n >= self._fresh_min
                else -1.0
            )
            # HOW MANY EPISODES AutoDR NEVER SAW because their start solve
            # did not converge (step B7). Read it against `dr/<key>_fill`: a
            # boundary that stops filling while this climbs is not a hard
            # boundary, it is an unreachable start pose.
            self.extras["log"]["dr/start_solve_flags_dropped"] = float(
                self._start_solve_flags_dropped
            )
            self.extras["log"]["dr/fresh_n"] = float(_fresh_n)
            self.extras["log"]["dr/fresh_min"] = float(self._fresh_min)
            # The boundary episodes' rate, its own curve and never folded
            # into the one above.
            #
            # WHAT IT IS NOT, and the first version of this comment said it
            # was: it is NOT the number Akkaya's buffers average. Those are
            # seven separate 240-flag windows, each emptied on every update;
            # this is ONE trailing window POOLED over all seven boundaries
            # and spanning many `bounds_version` changes. It can sit at 0.45
            # while `tilt_hi` reads 0.95 and `yaw_lo` 0.05. Read it as one
            # coarse "are the nailed episodes succeeding at all" number; the
            # per-boundary answer is `dr/<key>_fill` plus the `[autodr]`
            # event lines, which carry each buffer's own rate.
            self.extras["log"]["dr/success_rate_boundary"] = (
                sum(self._recent_boundary_successes)
                / len(self._recent_boundary_successes)
                if self._recent_boundary_successes
                else 0.0
            )
            # WHAT THE PHYSICS ACTUALLY RAN, averaged over the window -- the
            # reader `_friction_applied` did not have in step B4. Read it
            # against `dr/friction_lo` and `dr/friction_hi`: the mean of a
            # uniform draw sits at the band's centre, so a mean that does not
            # follow the band means the write and the boundary have come
            # apart. It is what we COMMANDED, not a PhysX read-back.
            self.extras["log"]["dr/friction_mean"] = (
                sum(self._recent_friction) / len(self._recent_friction)
                if self._recent_friction
                else 0.0
            )

        # LAST KEY ON PURPOSE (user, 2026-09-13, after RT-XXX): the deepest
        # single episode of the trailing window, mm. ``mean_max_depth_mm``
        # above is a MEAN over episodes and cannot say whether any one
        # episode ever passed the 13 mm hover. rsl_rl prints the keys in
        # insertion order, so this one stands at the bottom of the block.
        self.extras["log"]["max_depth_max_mm"] = (
            max(self._recent_depths) * 1000.0 if self._recent_depths else 0.0
        )

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
        return self._rate_by_bin(
            self._recent_successes, self._recent_tilt_deg, self._tilt_bin_edges_deg,
            "tilt_deg_from", "tilt_deg_to", "success_rate", first_lower=0.0, scale=1.0,
        )

    def _success_by_start_height_bin(self) -> list[dict]:
        """Trailing-window success rate resolved by the episode's START
        HEIGHT (plan step C, SBC). Same shape as ``_success_by_tilt_bin``,
        bounds in mm above the opening plane, last bin open at the top.

        The SBC readout: success in the low (inside) bins with 0 in the top
        (+30 mm, outside) bin is the IndustReal "partially-inserted" overfit,
        named before the run; success in the top bin is the claim the
        training has to earn. Under the fixed start every episode falls in
        the last bin and the table reduces to the overall rate.
        """
        return self._rate_by_bin(
            self._recent_successes, self._recent_start_height_m, self._start_height_bin_edges_m,
            "start_mm_from", "start_mm_to", "success_rate", first_lower=None, scale=1000.0,
        )

    def _success_by_lateral_bin(self) -> list[dict]:
        """Trailing-window success rate resolved by the episode's LATERAL
        START OFFSET (D-170). Same shape as ``_success_by_tilt_bin``, bounds
        in mm from the pocket axis, last bin open at the top.

        The readout RT-174 did not have: it trained a 6 mm start, read 100 %
        overall, and that one number cannot separate "learned the 6 mm
        start" from "learned the draws near the axis". Read it against the
        overall rate exactly like the tilt table -- the two agree only when
        the policy is equally good at every offset.
        """
        return self._rate_by_bin(
            self._recent_successes, self._recent_lateral_mm, self._lateral_bin_edges_mm,
            "lateral_mm_from", "lateral_mm_to", "success_rate", first_lower=0.0, scale=1.0,
        )

    def _success_by_yaw_bin(self) -> list[dict]:
        """Trailing-window success rate resolved by pocket YAW magnitude
        (D-038). Same shape as the tilt table, bounds in degrees.

        Yaw is the one axis of the target set whose reward landscape has
        never been sampled (RT-152 skipped), and the success gate's yaw
        window is still the square proxy's (D-121). A single overall rate
        under yaw noise therefore hides exactly the question a yaw run asks.
        """
        return self._rate_by_bin(
            self._recent_successes, self._recent_yaw_deg, self._yaw_bin_edges_deg,
            "yaw_deg_from", "yaw_deg_to", "success_rate", first_lower=0.0, scale=1.0,
        )

    def _force_abort_by_start_height_bin(self) -> list[dict]:
        """Trailing-window FORCE-ABORT rate over the same start-height bins
        as ``_success_by_start_height_bin`` (2026-09-03).

        Read the two tables side by side: per bin, success + abort + the
        rest = 1, and the rest is "ran out of time without getting in". A
        bin at 40 % success reads differently when the other 60 % pressed
        too hard than when they never engaged -- the first is a force
        problem, the second an approach problem -- and the overall
        ``force_abort_rate`` cannot tell which bin it comes from.
        """
        return self._rate_by_bin(
            self._recent_aborts, self._recent_start_height_m, self._start_height_bin_edges_m,
            "start_mm_from", "start_mm_to", "force_abort_rate", first_lower=None, scale=1000.0,
        )

    def _rate_by_bin(
        self, outcomes, values, edges, from_key: str, to_key: str, rate_key: str,
        first_lower, scale: float,
    ) -> list[dict]:
        """The one binning routine behind every per-bin table.

        ``outcomes`` is the per-episode 0/1 sequence whose mean is reported
        under ``rate_key`` (successes or force aborts); ``values`` is
        index-aligned with it; ``edges`` are
        upper bin edges in the values' own unit and the last bin catches
        anything at or above the last edge. ``first_lower`` is the printed
        lower bound of the first bin (``None`` = open at the bottom); ``scale``
        converts the printed bounds (1.0 for degrees, 1000.0 for m -> mm).
        Empty bins are kept (rate ``None``) rather than dropped -- see the tilt
        table's docstring for why.
        """
        counts = [0] * len(edges)
        hits = [0] * len(edges)
        for success, value in zip(outcomes, values):
            b = len(edges) - 1
            for i, upper in enumerate(edges):
                if value < upper:
                    b = i
                    break
            counts[b] += 1
            hits[b] += int(bool(success))
        lower = first_lower
        out = []
        for i, upper in enumerate(edges):
            out.append(
                {
                    from_key: None if lower is None else lower * scale,
                    to_key: upper * scale,
                    "episodes": counts[i],
                    rate_key: (hits[i] / counts[i]) if counts[i] else None,
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

    def _force_stats(self) -> dict:
        """The per-episode peak of the smoothed force magnitude, as a spread.

        THIS IS THE INSTRUMENT THAT RETIRES THE INVENTED F_MAX. D-114 says the
        abort limit comes from the force distribution measured at the
        scripted-insertion gate; ``force_abort_f_max_n`` currently holds a number
        nobody measured (``RL_PLACEHOLDERS``). A mean alone cannot replace
        it -- a limit is a tail question -- so the percentiles are written out
        and split by outcome, the same shape ``_target_lag_stats`` already uses.

        ``over_f_max`` is the direct answer to "would the abort have fired?",
        counted against whatever ``force_abort_f_max_n`` currently is. While
        that number is a placeholder the count is a diagnosis, not a result.
        """
        values = sorted(self._recent_force_norm)
        if not values:
            return {"episodes": 0}

        def _pct(sorted_values: list[float], q: float) -> float:
            # Nearest-rank, identical to _target_lag_stats -- one convention
            # for percentiles in this file, not two.
            return sorted_values[min(len(sorted_values) - 1,
                                     int(math.ceil(q * len(sorted_values))) - 1)]

        f_max = getattr(self.cfg, "force_abort_f_max_n", None)
        ok = sorted(f for f, s in zip(self._recent_force_norm, self._recent_successes) if s)
        bad = sorted(f for f, s in zip(self._recent_force_norm, self._recent_successes) if not s)
        out = {
            "episodes": len(values),
            "mean_n": sum(values) / len(values),
            "p50_n": _pct(values, 0.50),
            "p95_n": _pct(values, 0.95),
            "p99_n": _pct(values, 0.99),
            "max_n": values[-1],
            "success_p95_n": _pct(ok, 0.95) if ok else None,
            "failure_p95_n": _pct(bad, 0.95) if bad else None,
            "f_max_n": f_max,
            "over_f_max": sum(1 for v in values if f_max is not None and v >= f_max),
        }
        return out

    def _torque_stats(self) -> dict:
        """The per-episode peak of the smoothed torque magnitude, N m, as a
        spread -- the D-188 twin of ``_force_stats``, same nearest-rank
        percentiles, plus the share of steps whose wrench was VALID (not the
        reset step). Written in every observation mode: under ``force`` it
        says what the policy did not see."""
        values = sorted(self._recent_torque_norm)
        out: dict = {
            "episodes": len(values),
            "obs_wrench_mode": self._obs_mode,
            "wrench_valid_frac": (float(self._wrench_valid_steps) / self._wrench_steps
                                  if self._wrench_steps else None),
        }
        if not values:
            return out

        def _pct(sorted_values: list[float], q: float) -> float:
            return sorted_values[min(len(sorted_values) - 1,
                                     int(math.ceil(q * len(sorted_values))) - 1)]

        out.update({
            "mean_nm": sum(values) / len(values),
            "p50_nm": _pct(values, 0.50),
            "p95_nm": _pct(values, 0.95),
            "p99_nm": _pct(values, 0.99),
            "max_nm": values[-1],
        })
        return out

    def _append_episode_rows(
        self, idx: torch.Tensor, successes: torch.Tensor, aborts: torch.Tensor
    ) -> None:
        """Append one row per finished episode to ``<run folder>/episodes.csv``.

        Columns: ``EPISODE_LOG_COLUMNS``; their definitions go into
        ``episodes_meta.json``, written once when the file is opened. One host
        copy per call (one stacked tensor) and one flush per call, so a run
        that dies mid-way (RT-201s2) keeps every row it harvested.

        SKIPPED, never zero-filled, in two cases. Without ``cfg.log_dir``
        there is no run folder. With ``rl_terms_enabled = False`` the SAPU
        query never runs, and a column of zeros would read as "no
        interpenetration" when it means "not measured" -- the trap
        ``_interpen_stats`` names.
        """
        log_dir = getattr(self.cfg, "log_dir", None)
        if not log_dir or not self._rl_terms_enabled:
            return
        if self._episode_log is None:
            run_dir = pathlib.Path(log_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / "episodes.csv"
            fh = open(path, "a", newline="", encoding="utf-8")
            writer = csv.writer(fh)
            if fh.tell() == 0:
                writer.writerow(EPISODE_LOG_COLUMNS)
            self._episode_log = (fh, writer)
            with open(run_dir / "episodes_meta.json", "w", encoding="utf-8") as mf:
                json.dump(self._episode_log_meta(), mf, indent=2)
            print(f"[insertion] episode record: {path} ({len(EPISODE_LOG_COLUMNS)} columns)")
        fh, writer = self._episode_log
        steps_per_it = getattr(self.cfg, "rollout_steps_per_iteration", None)
        # common_step_counter is raised BEFORE _get_dones of the same step
        # (direct_rl_env.py:389), so a run's first step is 1 and belongs to
        # iteration 0. -1 = train.py did not hand the rollout length over.
        iteration = (
            (int(self.common_step_counter) - 1) // int(steps_per_it) if steps_per_it else -1
        )
        n = int(idx.numel())
        if self._dr is not None:
            bounds = self._bounds_stamp[idx].to(torch.float64)
            phase = str(self._dr.phase)
        else:
            bounds = torch.full((n,), -1.0, dtype=torch.float64, device=self.device)
            phase = "none"
        table = torch.stack(
            [
                idx.to(torch.float64),
                self._episode_index[idx].to(torch.float64),
                self.episode_length_buf[idx].to(torch.float64),
                self._max_interpen_m[idx].to(torch.float64) * 1000.0,
                self._interpen_unclean_steps[idx].to(torch.float64),
                self._max_force_norm[idx].to(torch.float64),
                self._max_force_raw_norm[idx].to(torch.float64),
                bounds,
                successes.to(torch.float64),
                aborts.to(torch.float64),
                self.reset_time_outs[idx].to(torch.float64),
            ],
            dim=1,
        ).cpu().tolist()
        run = pathlib.Path(log_dir).name
        seed = self.cfg.seed
        # A host-side loop over rows ALREADY on the CPU -- file I/O, not the
        # batched GPU hot path the no-loop rule is about.
        writer.writerows(
            [
                run, seed, iteration, int(env_id), int(ep),
                "success" if ok else "force_abort" if ab else "timeout" if to else "other",
                int(steps), f"{ip_mm:.6f}", int(ip_steps), f"{f_filt:.4f}", f"{f_raw:.4f}",
                int(bv), phase,
            ]
            for env_id, ep, steps, ip_mm, ip_steps, f_filt, f_raw, bv, ok, ab, to in table
        )
        fh.flush()
        self._episode_index[idx] += 1

    def _episode_log_meta(self) -> dict:
        """What ``episodes.csv`` measures -- written once per run beside it."""
        cfg = self.cfg
        thresh = float(cfg.interpen_thresh)
        return {
            "schema": "episodes-v1",
            "code_marker": CODE_MARKER,
            "columns": {
                "run": "run folder name (timestamp + RT name)",
                "seed": "cfg.seed of the run",
                "iteration": "PPO iteration the episode ENDED in: (common_step_counter - 1) // "
                             "rollout_steps_per_iteration; -1 if train.py did not hand the length over",
                "env_id": "index of the parallel env",
                "episode": "0-based count of the episodes THIS env had finished before",
                "outcome": "success | force_abort | timeout | other, tested in that order",
                "steps": "episode_length_buf at the end = policy steps (see first_episode_random_length)",
                "ip_max_mm": "max over the episode's policy steps of (max over the sample points) "
                             "of the SAPU interpenetration, mm",
                "ip_steps_ge_t1": "policy steps with that per-step maximum >= t1 "
                                  "(insertion_math.interpen_unclean); divided by steps = time share",
                "force_max_filtered_n": "max over the episode of |EMA-smoothed tared force|, N; "
                                        "the signal the force abort reads",
                "force_max_raw_n": "max over the episode of |tared unsmoothed force|, N, "
                                   "before observation noise",
                "bounds_version": "AutoDR bounds version the episode STARTED under; -1 without a provider",
                "dr_phase": "AutoDR phase when the episode ENDED (floor | dr); none without a provider",
            },
            "interpenetration": {
                "method": "Warp mesh query, insertion_sdf.SdfDistanceQuery.interpen_distances -> "
                          "insertion_math.max_interpen_dist (SAPU port, D-109 (10)); not a PhysX reading",
                "body_pair": "part surface sample points (fuegeteil.obj, tool_link frame) against the "
                             "fixture mesh (aufnahme.obj, entrance-anchored pocket frame). The gripper "
                             "has no collider (D-077); the table is not queried.",
                "sample_points": int(cfg.sdf_num_sample_points),
                "sample_seed": int(cfg.sdf_seed),
                "query_max_dist_m": float(cfg.sdf_max_dist),
                "unit": "mm",
                "rate_hz": 1.0 / (cfg.sim.dt * cfg.decimation),
                "physics_rate_hz": 1.0 / cfg.sim.dt,
                "decimation": int(cfg.decimation),
                "sampling_note": "read once per policy step; the physics steps in between are not seen",
                "t1_mm": thresh * 1000.0,
                "t1_meaning": "ALGORITHMIC acceptance threshold of this code, INTERPEN_THRESH = "
                              "0.5 * PLAY_X (D-106 (3)): at or above it the SAPU scale is zero and "
                              "no success can fire. NOT a physically admissible penetration.",
                "mesh_note": "the fixture mesh is not watertight (RT-64); the reading at known "
                             "separated and overlapping poses is check_insertion_sdf.py --pose-ladder",
            },
            "force": {
                "abort_limit_n": float(cfg.force_abort_f_max_n),
                "ema_factor": float(cfg.ft_smoothing_factor),
            },
            "rollout_steps_per_iteration": getattr(cfg, "rollout_steps_per_iteration", None),
            "episode_cap_steps": int(self.max_episode_length),
            "first_episode_random_length": "train.py calls runner.learn(init_at_random_ep_len=True): "
                                           "episode 0 of every env starts at a random step count, "
                                           "so its steps is not its length",
            "num_envs": int(self.num_envs),
            "dr_mode": str(cfg.dr_mode),
            "obs_wrench_mode": self._obs_mode,
            "spawn_workcell_block": bool(cfg.spawn_workcell_block),
        }

    def _interpen_stats(self) -> dict:
        """Distribution of the per-episode WORST SAPU interpenetration over the
        recent window, in mm. Same nearest-rank percentiles as _force_stats.
        ``over_thresh`` counts episodes whose worst depth exceeded
        ``cfg.interpen_thresh`` -- the SAPU filter would have zeroed their
        reward. CORRECTED 2026-09-15 (user): it counts episodes that crossed
        this code's ALGORITHMIC acceptance threshold at some step. It is not
        a physically admissible penetration, and not a proof that the
        collider failed.
        """
        if not self._rl_terms_enabled:
            # Same trap as the curve keys above: with the terms off the SAPU
            # query never ran, so a zeroed tail would claim a clean collider.
            return {"episodes": 0, "measured": False,
                    "why": "rl_terms_enabled is False; the SAPU query never ran"}
        values = sorted(self._recent_interpen_max)
        if not values:
            return {"episodes": 0, "measured": True}

        def _pct(sorted_values: list[float], q: float) -> float:
            return sorted_values[min(len(sorted_values) - 1,
                                     int(math.ceil(q * len(sorted_values))) - 1)]

        thresh = float(self.cfg.interpen_thresh)
        return {
            "episodes": len(values),
            "mean_mm": sum(values) / len(values) * 1000.0,
            "p50_mm": _pct(values, 0.50) * 1000.0,
            "p95_mm": _pct(values, 0.95) * 1000.0,
            "p99_mm": _pct(values, 0.99) * 1000.0,
            "max_mm": values[-1] * 1000.0,
            "thresh_mm": thresh * 1000.0,
            "over_thresh": sum(1 for v in values if v > thresh),
        }

    def _stage1_lateral_stats(self) -> dict:
        """The D-153 exposure, measured instead of argued.

        Per episode this window holds the worst ``|part_y - pocket_y|`` the
        part reached while inside stage 1. The keys that matter:

        * ``reach_offset_mm`` -- the offset at which the part's +y face
          touches the hole (``HOLE_REACH_OFFSET_Y``, 7.8847 mm). Written out
          so the file is readable without the decision beside it.
        * ``stage1_allows_mm`` -- what stage 1 geometrically permits
          (8.7000 mm). The gap between these two IS the exposure.
        * ``over_reach`` -- how many episodes in the window actually got
          there. **This is the answer.** Zero over a full window is the
          measured statement that the exposure did not occur at this scatter;
          any non-zero count reopens D-153 with a number attached.

        Percentiles use the same nearest-rank convention as
        ``_target_lag_stats`` and ``_force_stats`` -- one convention in this
        file, not three.
        """
        values = sorted(self._recent_stage1_lat_y)
        if not values:
            return {"episodes": 0}

        def _pct(sorted_values: list[float], q: float) -> float:
            return sorted_values[min(len(sorted_values) - 1,
                                     int(math.ceil(q * len(sorted_values))) - 1)]

        reach = insertion_tasks_cfg.HOLE_REACH_OFFSET_Y
        stage1_allows = (
            insertion_tasks_cfg.POCKET_STAGE1_Y_RANGE[1]
            - insertion_tasks_cfg.PART_BBOX_M[1] / 2.0
        )
        return {
            "episodes": len(values),
            "mean_mm": sum(values) / len(values) * 1000.0,
            "p50_mm": _pct(values, 0.50) * 1000.0,
            "p95_mm": _pct(values, 0.95) * 1000.0,
            "p99_mm": _pct(values, 0.99) * 1000.0,
            "max_mm": values[-1] * 1000.0,
            "reach_offset_mm": reach * 1000.0,
            "stage1_allows_mm": stage1_allows * 1000.0,
            "over_reach": sum(1 for v in values if v >= reach),
        }

    def _reward_terms_stats(self) -> dict:
        """Mean per-episode sum of each reward term over the trailing window.

        2026-09-02, after RT-134. ``episodes`` is the window fill; a term's
        value is ``None`` until the first episode has been logged. ``total``
        is the sum of the term means, i.e. the mean undiscounted return over
        the same window -- the number to set beside rsl_rl's ``Mean reward``.
        """
        out: dict = {}
        n = 0
        for name, values in self._recent_reward_terms.items():
            n = len(values)
            out[name] = (sum(values) / n) if n else None
        stats: dict = {"episodes": n}
        stats.update(out)
        stats["total"] = sum(out.values()) if n else None
        return stats

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
            # PROXY, both of them, and both kept only because compare_runs.py
            # reads them as columns. The label travels in "proxy_task_values"
            # below; the real play is written next to them as its own pair.
            "side_clearance_mm": insertion_tasks_cfg.SIDE_CLEARANCE * 1000.0,
            "yaw_window_deg": math.degrees(insertion_tasks_cfg.YAW_WINDOW_RAD),
            # The MEASURED lateral play, added 2026-08-28. Two numbers, because
            # the real pair is not square: X across the lugs, Y along the part.
            # TOTAL play, not the proxy's per-axis half-clearance above.
            "play_x_mm": insertion_tasks_cfg.PLAY_X * 1000.0,
            "play_y_mm": insertion_tasks_cfg.PLAY_Y * 1000.0,
            "reset_joint_noise_rad": self.cfg.reset_joint_noise,
            "reset_yaw_noise_rad": self.cfg.reset_yaw_noise,
            # D-161 RUNG 0: WHERE THE EPISODE STARTED. A run whose start
            # height can only be inferred from the folder name is a run that
            # gets compared wrongly -- and this is the one number the last
            # failed run turned on. ``null`` means the bare home pose.
            "start_tip_above_entrance_mm": (
                None if self.cfg.start_tip_above_entrance is None
                else self.cfg.start_tip_above_entrance * 1000.0
            ),
            # Plan step C (SBC): the LOWER bound of the sampled start range,
            # ``null`` = fixed start at the line above. And the success rate
            # resolved by start-height bin -- the readout that separates
            # "learned the hard end" from "learned the inside only".
            # ONE number since D-178 (3): the disk RADIUS in metres. Files
            # written under D-176 (Phase 5 step B3) carry an (x, y) half-width
            # pair here, and RT-174/RT-175 a bare per-axis half width (0.006)
            # -- a reader of old and new files must accept every shape and
            # must not read an old number as a radius.
            "start_lateral_offset_m": self._start_lat_half,
            # D-170: the success rate resolved by the LATERAL start offset.
            # The overall rate averages over the whole draw and cannot say
            # which offsets were learned; this list can. At offset 0.0 every
            # episode falls in the first bin and the table reduces to the
            # overall rate, which is the commissioning case.
            "success_by_lateral_bin": self._success_by_lateral_bin(),
            "start_tip_above_entrance_low_mm": (
                None if self.cfg.start_tip_above_entrance_low is None
                else self.cfg.start_tip_above_entrance_low * 1000.0
            ),
            # THE START FLOOR (SBC step 0): the configured floor and its rung
            # count; the floor's CURRENT value and the phase live in
            # dr_state (the provider's own dict), one home per fact.
            "start_floor_m": (float(self.cfg.start_floor_m) if self._start_floor_on else None),
            "start_floor_steps": (int(self.cfg.start_floor_steps) if self._start_floor_on else None),
            "autodr_stall_buffers": int(self.cfg.autodr_stall_buffers),
            "success_by_start_height_bin": self._success_by_start_height_bin(),
            # Same bins, the ABORTS as the outcome (2026-09-03): per bin,
            # success + abort + timeout = 1, so the two tables together say
            # whether a bin fails by pressing too hard or by never engaging.
            "force_abort_by_start_height_bin": self._force_abort_by_start_height_bin(),
            "home_tip_above_entrance_mm":
                insertion_tasks_cfg.WORKCELL_HOME_TIP_ABOVE_ENTRANCE * 1000.0,
            # The start-pose IK's own report. A residual that grows, or a
            # nonzero unconverged count, means the episodes did NOT start
            # where the line above says they did.
            "start_pose_solve_last_residual_mm": self._start_solve_residual_mm,
            "start_pose_solve_worst_residual_mm": self._start_solve_worst_mm,
            "start_pose_solve_last_iterations": self._start_solve_iters,
            "start_pose_solve_tol_mm": self.cfg.start_pose_solve_tol_m * 1000.0,
            "start_pose_solve_unconverged_resets": self._start_solve_unconverged,
            "start_pose_solve_flags_dropped_envs": self._start_solve_flags_dropped,
            # WHICH SOURCE randomised this run (Phase 5). The four cfg angle
            # fields below all read 0.0 under AutoDR -- the angles come from
            # the boundaries -- so without this key an AutoDR run and a No-DR
            # run are indistinguishable in the file the results are read from,
            # and the two are NOT the same env: under a provider the pocket
            # frame is live and the reset writes a fixture pose every episode.
            # Same reason as pocket_side_m, one line below.
            "dr_mode": str(self.cfg.dr_mode),
            # The WHOLE provider state, the shape `AutoDR.as_dict` returns:
            # the bounds live at ["dr_state"]["bounds"], the version at
            # ["dr_state"]["bounds_version"]. Not lifted to a second flat key
            # -- one home per fact, and a flat copy goes stale on resume.
            "dr_state": (None if self._dr is None else self._dr.as_dict()),
            # THE STOP RULE'S OWN NUMBERS, on disk (plan section 4). The rule
            # is decided from a file, not from a scrolled curve, and the three
            # here are what it tests: is the fresh window full enough, what
            # does it say, and is every boundary at its maximum. `null` with
            # no provider -- there is no fresh window to report.
            #
            # `rate` is null rather than -1.0 while the window is short: JSON
            # has a null and a curve does not, so the sentinel the log needs
            # is not needed here and would only invite a reader to average it.
            "dr_fresh": (
                None if self._dr is None else {
                    "n": len(self._fresh_successes),
                    "min": self._fresh_min,
                    "rate": (
                        sum(self._fresh_successes) / len(self._fresh_successes)
                        if len(self._fresh_successes) >= self._fresh_min
                        else None
                    ),
                    "boundary_n": len(self._recent_boundary_successes),
                    "boundary_rate": (
                        sum(self._recent_boundary_successes)
                        / len(self._recent_boundary_successes)
                        if self._recent_boundary_successes
                        else None
                    ),
                    "all_at_max": self._dr.all_at_max(),
                }
            ),
            # WHAT THE PHYSICS RAN, over the same window as the success rate.
            # min/max, not just the mean: the mean alone cannot show that a
            # band opened, and the pair against `dr_state.bounds.friction`
            # says whether the draw and the boundary agree. Commanded values,
            # not a PhysX read-back -- that is the startup report's line.
            "friction_applied": (
                None if not self._recent_friction else {
                    "mean": sum(self._recent_friction) / len(self._recent_friction),
                    "min": min(self._recent_friction),
                    "max": max(self._recent_friction),
                    "episodes_in_window": len(self._recent_friction),
                }
            ),
            # THE AFFORDABILITY MEASUREMENT of plan section 6, on disk, because
            # the decision it feeds -- keep the per-reset main path or fall
            # back to the redraw-per-boundary-move form -- must be read from a
            # file and not off a scrolled console. `null` throughout when this
            # run has no friction reach and never writes a material.
            #
            # `samples_in_window` is the sample size behind the two figures --
            # a mean over three calls is not a cost. It is NOT the run's reset
            # count: it saturates at `window` once the deque is full, which is
            # why the two are written side by side and why the field is not
            # called `reset_calls`.
            "friction_write_ms": (
                None if not self._friction_write_ms else {
                    "mean": sum(self._friction_write_ms) / len(self._friction_write_ms),
                    "max": max(self._friction_write_ms),
                    "samples_in_window": len(self._friction_write_ms),
                    "window": self._friction_write_ms.maxlen,
                }
            ),
            # D-034: the fixture-pose randomisation this run trained/replayed
            # under. Recorded for the same reason as pocket_side_m -- a run
            # whose randomisation can only be inferred is one that gets
            # compared wrongly.
            "fixture_pos_noise_xy_m": self.cfg.fixture_pos_noise_xy,
            "fixture_yaw_noise_rad": self.cfg.fixture_yaw_noise_rad,
            # D-182 / D-183: WHAT THE POLICY SAW, as opposed to what the
            # physics did. Same argument as the two lines above -- a run whose
            # scatter can only be inferred from the commit gets compared
            # wrongly -- but these three are the only keys in this file that
            # qualify the OBSERVATION rather than the scene, and a success
            # rate under a mislocated pocket is a different claim from one
            # under a perfect one. All three are in RL_PLACEHOLDERS too; they
            # are written out here as well so a reader does not have to open
            # that sub-dict to compare two runs.
            "obs_noise_pocket_pos_std_m": self.cfg.obs_noise_pocket_pos_std_m,
            "force_obs_noise_std_n": self.cfg.force_obs_noise_std_n,
            # D-188: which layout this run's policy saw, and the torque sigma.
            "obs_wrench_mode": self._obs_mode,
            "obs_version": insertion_math.obs_version(self._obs_mode),
            "torque_obs_noise_std_nm": self.cfg.torque_obs_noise_std_nm,
            "grasp_obs_offset_x_m": self.cfg.grasp_obs_offset_x_m,
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
            # D-038, same argument on the yaw axis: the range this run drew
            # from is two lines above (fixture_yaw_noise_rad); this is the
            # rate resolved by the magnitude actually drawn.
            "success_by_yaw_bin": self._success_by_yaw_bin(),
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
            # The force distribution D-114 wants measured. Written next to the
            # success rate on purpose: a success rate produced under an
            # invented abort limit has to be readable together with the forces
            # that limit was guessed against.
            "force_norm_n": self._force_stats(),
            # D-188: the torque twin, N m, written in every mode.
            "torque_norm_nm": self._torque_stats(),
            # Solver-accuracy instrument (inbox 2026-09-12): the per-episode
            # worst interpenetration as a tail, mm, beside the SAPU threshold
            # it is measured against.
            "interpen_max_mm": self._interpen_stats(),
            # D-153's open exposure, as a measured tail. The decision was
            # accepted on the condition that fixture_pos_noise_xy = 0.0 and
            # reopens by its own clause above zero; this key is what closes it
            # with a number instead of with the bound. Read `over_reach`
            # first -- everything else is context for it.
            "stage1_lateral_y_mm": self._stage1_lateral_stats(),
            "reward_terms_recent_mean": self._reward_terms_stats(),
            # D-157 tripwire: successful episodes whose max depth inside the
            # pocket was zero. Both numbers must be 0. A non-zero value voids
            # the success rate in this same file -- it is the RT-107 failure
            # (success paid for a descent beside the fixture) coming back.
            "success_depth_invariant_violations": {
                "recent": int(sum(self._recent_success_depth_violations)),
                "cumulative": self._ep_success_depth_violations,
                "window_episodes": len(self._recent_success_depth_violations),
            },
            # RT-201s3 tripwire (2026-09-15), the same flag as the per-iteration
            # `below_fixture_rate`: episodes whose tip went below the fixture
            # underside inside the pocket cross-section. Read it with the log
            # key's comment: non-zero is the path under the free fixture.
            "below_fixture_rate": {
                "recent": (
                    sum(self._recent_below_fixture) / len(self._recent_below_fixture)
                    if self._recent_below_fixture
                    else 0.0
                ),
                "episodes": int(sum(self._recent_below_fixture)),
                "window_episodes": len(self._recent_below_fixture),
                "underside_depth_mm": -insertion_tasks_cfg.POCKET_ASSET_BOTTOM_Z * 1000.0,
            },
            # D-113 (9): the abort rate over the same trailing window as the
            # success rate, as its own key. In a measurement env
            # (rl_terms_enabled = False) the abort is off and this reads 0.0.
            "force_abort_rate": (
                sum(self._recent_aborts) / len(self._recent_aborts)
                if self._recent_aborts
                else 0.0
            ),
            # Mean steps of the successful episodes in the trailing window;
            # None until one succeeded. Read against the episode cap.
            "mean_success_episode_steps": (
                sum(self._recent_success_steps) / len(self._recent_success_steps)
                if self._recent_success_steps
                else None
            ),
            "episode_length_s": self.cfg.episode_length_s,
            "num_envs": self.num_envs,
            "code_marker": CODE_MARKER,
            # The success rate above was produced under these invented numbers.
            # It travels WITH them, because the metric rule reads results from
            # files rather than from prose -- and a placeholder that only lived
            # in a console line would be gone by the time anyone read the JSON.
            "rl_placeholders": {
                name: getattr(self.cfg, name, None) for name in RL_PLACEHOLDERS
            },
            # The controller the numbers were produced under (inbox entry
            # 'Audit 2026-09-03 (a)'): a success rate under joint_pd and one
            # under osc are different claims.
            "control_mode": self._control_mode,
            "policy_rate_hz": 1.0 / (self.cfg.sim.dt * self.cfg.decimation),
            # The ladder state, for the same reason: a success rate from a
            # fixed rung 0 and one from a climbed ladder are different claims,
            # and the console line that said which is gone by the time anyone
            # opens this file (D-110 (3)).
            "curriculum_enabled": bool(self.cfg.curriculum_enabled),
            "rung_step_sizes": self.cfg.rung_step_sizes,
            # The SCENE, added 2026-08-30 when the run tag stopped naming it.
            # With the block off the pocket has no rear wall and is enterable
            # from any direction, so the task is EASIER than the real cell
            # (D-156). That qualifies every number in this file, so it has to
            # be in the file -- the startup report says it too, but a console
            # line is gone by the time anyone opens the JSON, and CLAUDE.md
            # reads results from files, never from prose.
            "spawn_workcell_block": bool(self.cfg.spawn_workcell_block),
            # And the proxy numbers the score was produced under. Same reason
            # as the placeholders above and the same rule -- the console block
            # is gone by the time anyone opens this file. ``metrics_keys`` says
            # WHICH keys of this very file are proxy, so a reader does not have
            # to know the history to see it.
            "proxy_task_values": {
                path: {"value": value, "mark": mark}
                for path, value, mark in resolve_proxy_task_values(self.cfg)
            },
            "proxy_metrics_keys": PROXY_METRICS_KEYS,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:
            if not getattr(self, "_metrics_warning_printed", False):
                self._metrics_warning_printed = True
                print(f"[insertion] could not write metrics to {path}: {exc}")

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

        ``author_peg_ur10e.py`` (the proxy's, deleted 2026-08-28 -- git history)
        wrote a sidecar next to the USD with the
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
            return (f"UNCHECKED -- no {sidecar.name} beside the robot USD; re-run "
                    "scripts/author_tool_ur5e.py to produce one")
        try:
            data = json.loads(sidecar.read_text())
            # The authoring report nests its measurements under "after"; the
            # top-level fallback covers a sidecar written by an older revision.
            measured = data.get("after", {}).get("measured") or data.get("measured", {})
            verdict = data.get("verdict")
            # Two shapes of sidecar. author_tool_ur5e.py measures the part
            # RELATIVE TO the welded link and reports part_local_size_m;
            # the proxy's author_peg_ur10e.py authored its own points and reported
            # local_bbox_size_m. Which key is present says which asset this is,
            # so the check cannot compare a tool against peg constants.
            tool_size = measured.get("part_local_size_m")
            size = measured.get("local_bbox_size_m")
            span = measured.get("tool_local_z_range_m")
            mass = measured.get("mass_kg")
        except (OSError, ValueError, AttributeError) as exc:
            return f"UNCHECKED ({sidecar.name} unreadable: {type(exc).__name__}: {exc})"
        if verdict is not None and verdict != "PASS":
            return (f"*** THE LAST AUTHORING RUN REPORTED {verdict} *** "
                    f"see {sidecar.name}; the asset was written despite failing checks")
        if tool_size:
            problems = []
            for axis in range(3):
                if abs(tool_size[axis] - cfg.part_bbox_m[axis]) > 1e-4:
                    problems.append(
                        f"part axis {axis} {tool_size[axis]:.5f} m authored vs "
                        f"{cfg.part_bbox_m[axis]:.5f} m configured"
                    )
            # The wrong-side test, repeated here on the sidecar: the tool must
            # run from the flange face AWAY from the robot. Welded the other way
            # every edge length above still matches.
            want_span = insertion_tasks_cfg.TOOL_SPAN_LOCAL_Z_M
            if span and (abs(span[0] - want_span[0]) > 1e-4 or abs(span[1] - want_span[1]) > 1e-4):
                problems.append(
                    f"tool spans z {span[0]:+.5f}..{span[1]:+.5f} m in link-local coordinates, "
                    f"expected {want_span[0]:+.5f}..{want_span[1]:+.5f}"
                )
            if cfg.tool_mass_kg is not None and mass is not None and abs(mass - cfg.tool_mass_kg) > 1e-5:
                problems.append(f"mass {mass:.5f} kg authored vs {cfg.tool_mass_kg:.5f} kg configured")
            if problems:
                return ("*** ASSET DOES NOT MATCH THE CONFIGURATION *** " + "; ".join(problems)
                        + ". Re-run scripts/author_tool_ur5e.py. Every tool number in this report is void.")
            mass_txt = ("mass PENDING -- authored "
                        f"{mass:.5f} kg, cfg.tool_mass_kg not set" if cfg.tool_mass_kg is None
                        else f"mass {mass:.5f} kg")
            return (f"PASS -- authored tool matches cfg (part {tool_size[0]:.5f} x {tool_size[1]:.5f} x "
                    f"{tool_size[2]:.5f} m, span {span[0]:+.4f}..{span[1]:+.4f} m, {mass_txt}), "
                    f"from {sidecar.name}")

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
                    + ". The USD is stale: re-run scripts/author_tool_ur5e.py. Every "
                      "number about the welded body in this report is void.")
        return (f"PASS -- authored geometry matches cfg (side {size[0]:.5f} m, "
                f"length {size[2]:.5f} m, mass {mass:.5f} kg), from {sidecar.name}")

    def _fixture_geometry_line(self) -> str:
        """The same check for the fixture -- REWRITTEN 2026-08-24 for the CAD asset.

        What it used to do, and why that is gone: the fixture was a GENERATED
        square table (D-033), one file per curriculum rung, and the proxy's
        ``author_tisch_square.py`` (deleted 2026-08-28, S6) wrote the pocket
        side it had measured into an ``.author.json`` sidecar. This compared that against ``fixed_asset.side``.

        The fixture is now the user's CAD cut-out, which no script authors. It
        carries no ``.author.json`` at all, so the old code degraded to
        "UNCHECKED" on every run -- a check that cannot fail is worse than no
        check, because it reads like one that passed.

        What it compares instead: ``verify_fixture_usd.py`` writes the bounding
        box it MEASURED off the stage into ``<asset>.verify.json``. That is the
        only artefact describing the asset rather than the intent, so the
        measured box is compared against the constants derived from the STEP
        (``POCKET_ASSET_BBOX_M``, ``POCKET_LOCAL_Z_RANGE``). Both trace to the
        same CAD, by two different routes -- one through Isaac's importer, one
        through reading the STEP's cartesian points -- so a disagreement is real
        and not a restatement.
        """
        geo = insertion_tasks_cfg
        try:
            sidecar = pathlib.Path(self._fixture_usd_path).with_suffix(".verify.json")
        except (AttributeError, TypeError) as exc:
            return f"UNCHECKED (no fixture usd path: {exc})"
        if not sidecar.is_file():
            return (f"UNCHECKED -- no {sidecar.name} beside the fixture USD. It is written by "
                    "scripts/verify_fixture_usd.py; without it the imported geometry is "
                    "unverifiable from here. Run: python scripts/verify_fixture_usd.py "
                    f"--usd {self._fixture_usd_path} --per-prim")
        try:
            data = json.loads(sidecar.read_text())
            extents = data.get("bbox_extents_physical_m")
            z_range = data.get("bbox_z_range_m")
            mpu_ok = data.get("meters_per_unit_ok")
            up_ok = data.get("up_axis_ok")
            self_contained = data.get("self_contained_ok")
        except (OSError, ValueError, AttributeError) as exc:
            return f"UNCHECKED ({sidecar.name} unreadable: {type(exc).__name__}: {exc})"
        if extents is None or z_range is None:
            return f"UNCHECKED ({sidecar.name} carries no measured bounding box)"

        problems = []
        if mpu_ok is False:
            problems.append("metersPerUnit is not 1.0 (D-083 unit repair not applied)")
        if up_ok is False:
            problems.append("up-axis is not Z")
        if self_contained is False:
            problems.append("the asset carries EXTERNAL composition arcs")
        # Sorted, because CAD and stage may assign the axes differently while
        # the body is the same size. A swapped axis shows up in the z range
        # below, which is NOT sorted.
        want = sorted(geo.POCKET_ASSET_BBOX_M)
        # verify_fixture_usd.py writes this key as a DICT {"x":, "y":, "z":}
        # (verify_fixture_usd.py:147-149), not as a list. Iterating it yields
        # the KEYS, so the first version of this line ran float("x") and took
        # the whole startup report down with a ValueError (RT-40). Both shapes
        # are accepted rather than one being assumed, because the sidecar is
        # written by another script and its shape is not this file's to fix.
        try:
            values = extents.values() if isinstance(extents, dict) else extents
            got = sorted(float(v) for v in values)
        except (TypeError, ValueError) as exc:
            return (f"UNCHECKED ({sidecar.name} bbox_extents_physical_m is not three "
                    f"numbers: {extents!r} -- {type(exc).__name__})")
        if len(got) != 3:
            return (f"UNCHECKED ({sidecar.name} bbox_extents_physical_m has {len(got)} "
                    "values, expected 3)")
        for label, w, g in zip(("smallest", "middle", "largest"), want, got):
            if abs(g - w) > 1e-4:
                problems.append(
                    f"{label} edge {g*1000:.3f} mm measured vs {w*1000:.3f} mm from the STEP"
                )
        # This key IS a two-element list in the sidecar, unlike the one above --
        # checked rather than assumed, after RT-40.
        try:
            z_lo, z_hi = (float(v) for v in z_range)
        except (TypeError, ValueError) as exc:
            return (f"UNCHECKED ({sidecar.name} bbox_z_range_m is not two numbers: "
                    f"{z_range!r} -- {type(exc).__name__})")
        for label, w, g in zip(("z min", "z max"), geo.POCKET_LOCAL_Z_RANGE, (z_lo, z_hi)):
            if abs(g - w) > 1e-4:
                problems.append(f"{label} {g*1000:+.3f} mm measured vs {w*1000:+.3f} mm expected")

        if problems:
            return ("*** FIXTURE DOES NOT MATCH THE CAD CONSTANTS *** " + "; ".join(problems)
                    + f". Either the asset is stale or the constants are: see {sidecar.name} "
                      "against the POCKET_* block in insertion_tasks_cfg.py. Every depth, gate "
                      "and success number in this report is void until they agree.")
        return (f"PASS -- imported fixture matches the STEP constants (edges "
                f"{got[0]*1000:.3f} / {got[1]*1000:.3f} / {got[2]*1000:.3f} mm, z "
                f"{z_lo*1000:+.3f} .. {z_hi*1000:+.3f} mm), from {sidecar.name}")

    @property
    def _start_tip_height(self) -> float:
        """Where the leading tool point is SUPPOSED to stand at the start of an
        episode, above the stage-2 opening plane.

        The startup report judges the flange standoff, the insertion depth and
        the lateral offset against this. Since D-161 it is no longer the home
        pose by default, and a report that still printed the home number would
        flag a correct start pose as a 191 mm fault.
        """
        if self.cfg.start_tip_above_entrance is None or self._start_ik is None:
            return float(insertion_tasks_cfg.WORKCELL_HOME_TIP_ABOVE_ENTRANCE)
        # Under start-height sampling (plan step C) env 0's commanded height
        # is its own draw, held in the per-env buffer once the first reset
        # has solved it. The report judges env 0, so it reads env 0's height.
        # Before the buffer exists (this property also seeds it in __init__)
        # the fixed bound is the answer.
        buf = getattr(self, "_start_height", None)
        if buf is not None and (
            getattr(self, "_start_sampling", False) or getattr(self, "_start_floor_on", False)
        ):
            return float(buf[0])
        return float(self.cfg.start_tip_above_entrance)

    def _print_controller_report(self) -> None:
        """The controller block of the startup report (inbox entry "Audit
        2026-09-03 (a)", discriminating measurement): mode, gains, limits,
        the drive parameters READ BACK from PhysX per joint (type from the
        USD, stiffness, damping, maxForce), and at the reset pose the
        task-space inertia Lambda = (J M^-1 J^T)^-1 (largest eigenvalue of
        the translational and the rotational block, env 0) and the Jacobian
        condition number. Lambda_max is the input to Decision (4)'s kp
        inequality; the condition number is Decision (5)'s open reading
        (near a singularity the unit-mass decoupling no longer holds).
        Printed in BOTH modes so the PD half of the measurement carries the
        same numbers.
        """
        cfg = self.cfg
        print("--- controller (inbox entry 'Audit 2026-09-03 (a)') ---")
        _rate = 1.0 / (cfg.sim.dt * cfg.decimation)
        if self._control_mode == "osc":
            print(f"control_mode:   osc -- 6-D pose delta, Isaac Lab OperationalSpaceController, "
                  f"target = current pose + delta re-anchored every physics step (Factory form)")
            print(f"  osc cfg:      target pose_abs, impedance fixed, inertial decoupling ON (full), "
                  f"gravity compensation ON with gravity ON (test_operational_space.py:350-358), "
                  f"nullspace none")
            print(f"  kp:           pos {cfg.osc_kp_pos} N/m [placeholder], rot {cfg.osc_kp_rot} Nm/rad "
                  f"[placeholder], damping ratio {cfg.osc_damping_ratio} (critical, 2*sqrt(kp))")
            print(f"  step limits:  {cfg.osc_pos_step_limit_m} m, {cfg.osc_rot_step_limit_rad} rad per "
                  f"policy step [placeholder, Factory]")
            print(f"  tip clamp:    +-{cfg.osc_pos_clamp_m} m x/y [placeholder, D-180: disk reach "
                  f"+ one action step + set air] and "
                  f"+-{cfg.osc_pos_clamp_z_m} m z [placeholder, D-180: highest legal start "
                  f"+ one action step + set air] box around the pocket entrance; tilt cone "
                  f"{math.degrees(cfg.osc_tilt_clamp_rad):.2f} deg from THIS EPISODE'S POCKET AXIS "
                  f"(CAD, Geometrie_Fuegeteil_Aufnahme.tex:264); yaw free")
            print(f"  controlled body: {self.robot.body_names[self._osc_body_idx]} (index {self._osc_body_idx}, "
                  f"jacobian row block {self._osc_jacobi_idx}), tip offset {self._tip_offset_local.tolist()} m")
            print(f"  rates:        policy {_rate:.1f} Hz (decimation {cfg.decimation} [placeholder, Factory]), "
                  f"controller {1.0 / cfg.sim.dt:.1f} Hz; episode {self.max_episode_length} steps = "
                  f"{cfg.episode_length_s:.4f} s (step count kept, seconds follow -- re-derivation owed)")
        else:
            print(f"control_mode:   joint_pd -- D-108 joint-target integrator on the USD drives "
                  f"(action_scale {cfg.action_scale} rad/step), policy {_rate:.1f} Hz; the PD half of the "
                  f"Audit 2026-09-03 (a) measurement, not a training option")
        # Drive parameters READ BACK, never echoed from the cfg. Under OSC
        # stiffness and damping must read 0.0 on every joint; maxForce must
        # still be the USD's (150 / 28). The drive TYPE is a USD attribute
        # (force vs acceleration), read off env_0's joint prims.
        try:
            _view = self.robot.root_physx_view
            _stiff = _view.get_dof_stiffnesses()[0].tolist()
            _damp = _view.get_dof_dampings()[0].tolist()
            _fmax = _view.get_dof_max_forces()[0].tolist()
            _types: dict = {}
            try:
                from isaaclab.sim.utils.stage import get_current_stage as _gcs  # noqa: PLC0415
                from pxr import Usd as _Usd  # noqa: PLC0415
                _stage = _gcs()
                _root = _stage.GetPrimAtPath("/World/envs/env_0/Robot")
                for _p in _Usd.PrimRange(_root):
                    _a = _p.GetAttribute("drive:angular:physics:type")
                    if _a and _a.IsValid():
                        _types[_p.GetName()] = str(_a.Get())
            except Exception as _exc:  # noqa: BLE001 -- report line, never a crash
                _types = {"*": f"UNREAD ({type(_exc).__name__}: {_exc})"}
            print("  drives (PhysX read-back): joint | type (USD) | stiffness | damping | maxForce")
            for _j, _name in enumerate(self.robot.joint_names):
                _t = _types.get(_name, _types.get("*", "UNREAD"))
                print(f"    {_name:<20} {_t:<14} {_stiff[_j]:12.4f} {_damp[_j]:10.4f} {_fmax[_j]:10.2f}")
            if self._control_mode == "osc" and (max(abs(v) for v in _stiff) > 0.0 or max(abs(v) for v in _damp) > 0.0):
                print("  *** DRIVES NOT INERT under osc -- the PhysX drive still fights the controller ***")
        except Exception as _exc:  # noqa: BLE001 -- report line, never a crash
            print(f"  drives: UNREAD -- {type(_exc).__name__}: {_exc}")
        try:
            jacobian_b, mass_matrix, _gravity = self._task_space_dynamics()
            _j0 = jacobian_b[0:1]
            _m0 = mass_matrix[0:1]
            _lambda = torch.inverse(_j0 @ torch.inverse(_m0) @ _j0.mT)[0]
            _lam_lin = torch.linalg.eigvalsh(_lambda[0:3, 0:3])
            _lam_rot = torch.linalg.eigvalsh(_lambda[3:6, 3:6])
            _sv = torch.linalg.svdvals(_j0[0])
            _cond = float(_sv.max() / _sv.min()) if float(_sv.min()) > 0.0 else float("inf")
            _start = ("home pose" if cfg.start_tip_above_entrance is None
                      else f"start_tip_above_entrance {float(cfg.start_tip_above_entrance):+.4f} m")
            print(f"  Lambda (env 0, at the reset pose = {_start}): translational eigenvalues "
                  f"{[round(float(v), 4) for v in _lam_lin.tolist()]} kg, MAX {float(_lam_lin.max()):.4f} kg; "
                  f"rotational eigenvalues {[round(float(v), 5) for v in _lam_rot.tolist()]} kg m^2, "
                  f"MAX {float(_lam_rot.max()):.5f}")
            print(f"  Jacobian condition number (env 0, {self.robot.body_names[self._osc_body_idx]}): "
                  f"{_cond:.3f} (sigma max {float(_sv.max()):.4f} / min {float(_sv.min()):.6f})")
            print(f"  gravity compensation torques (env 0): {[round(float(v), 3) for v in _gravity[0].tolist()]} Nm "
                  f"vs maxForce {[round(float(v), 1) for v in self.robot.data.joint_effort_limits[0].tolist()]}")
            if self._control_mode == "osc":
                _kp_bound = 20.0 / (float(_lam_lin.max()) * float(cfg.osc_pos_step_limit_m))
                print(f"  Decision (4) reading: kp <= F_search / (Lambda_max * step_limit) = 20 N / "
                      f"({float(_lam_lin.max()):.4f} kg * {cfg.osc_pos_step_limit_m} m) = {_kp_bound:.1f} "
                      f"(unit check: kg*m in the denominator is a force per (N/m), READ, not decided)")
        except Exception as _exc:  # noqa: BLE001 -- report line, never a crash
            print(f"  Lambda / condition number: UNREAD -- {type(_exc).__name__}: {_exc}")
        print("--- end controller ---")

    def _print_startup_report(self, joint_vel_fd: torch.Tensor) -> None:
        cfg = self.cfg
        origins = self.scene.env_origins
        # The PD target that _reset_idx actually wrote, NOT the bare home pose.
        # They differ on purpose: reset_yaw_noise puts up to +-0.7854 rad of
        # per-episode yaw on the tool-axis joint. Reporting against the bare
        # home pose made that randomisation look like a tracking error of up to
        # 45 deg on wrist_3_joint (RT-2..RT-5). The home pose is printed
        # alongside so both stay visible.
        commanded = self._joint_targets
        home_pose = self.robot.data.default_joint_pos
        down = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        # Computed from cfg rather than read off the base class: step_dt was not
        # covered by probe_assets.py.
        sim_time = self._obs_calls * cfg.sim.dt * cfg.decimation
        print("=" * 72)
        # The title said "square-peg pivot, branch square-peg-insertion" until
        # 2026-08-28. Both halves were false: the env spawns the real CAD
        # fixture and the real tool-welded UR5e, and the branch is long gone.
        # What IS still the proxy's is the SCORING, and the title now says
        # which half is which rather than naming a branch that can go stale.
        print("INSERTION STARTUP REPORT -- REAL geometry and physics, PROXY scoring (UNVERIFIED)")
        print(f"step {self._obs_calls} of {self._report_at_obs_calls}, t = {sim_time:.3f} s after reset")
        print(f"code marker: {CODE_MARKER}")
        # ADDED after RT-71, which could not prove S4. The episode cap was
        # changed from the proxy's 240 to D-113 (3)'s 256, and NO line in any
        # report carried it -- so no log could distinguish the two. The
        # scripted run's own "32 episodes over 600 steps" comes out the same
        # for both. The value printed is max_episode_length, the number the
        # env actually resets on, not the seconds it was configured from:
        # seconds are the input and the ceil() is where the off-by-one lives.
        print(f"episode cap (D-113 (3)): {self.max_episode_length} control steps "
              f"= {cfg.episode_length_s:.6f} s at {1.0 / (cfg.sim.dt * cfg.decimation):.1f} Hz")
        self._print_controller_report()
        print(f"fixture usd: {self._fixture_usd_path}")
        print(f"robot usd:   {self._robot_usd_path}")
        print(f"fixture spawn translation: {cfg.fixture_pos}  (asset origin = opening plane at pocket centre)")
        print(f"fixture tilt (D-036 probe): {math.degrees(cfg.fixture_tilt_rad):+.2f} deg about env y (lateral axis; +tips the back-wall side down) through the pocket centre "
              f"({'ACTIVE -- one fixed angle in every env' if cfg.fixture_tilt_rad != 0.0 else 'inactive'})")
        print(f"fixture yaw (D-038 probe): {math.degrees(cfg.fixture_yaw_rad):+.2f} deg about env z through the pocket centre "
              f"({'ACTIVE -- one fixed angle in every env' if cfg.fixture_yaw_rad != 0.0 else 'inactive'})")
        print(f"fixture tilt noise (D-037): magnitude 0..{math.degrees(cfg.fixture_tilt_noise_rad):.2f} deg, "
              "direction uniform over 360 deg, resampled per episode "
              f"({'ACTIVE -- per-env pocket orientation' if cfg.fixture_tilt_noise_rad > 0.0 else 'inactive'})")
        print(f"pocket-frame measurement: {'ON -- gate/depth/alignment/phi and obs 12:15 in the pocket frame' if self._tilt_active else 'OFF -- pocket upright, env-local measurement'}")
        # The four cfg angles above are ALL 0.00 deg under AutoDR -- the angle
        # comes from the boundaries, which start on the centre. Without this
        # line the report reads "inactive" for a run that is about to tilt.
        print(f"dr_mode: {cfg.dr_mode}")
        if self._dr is not None:
            _b = self._dr.bounds()
            _m = self._dr.bounds_max()
            print(f"  provider: {type(self._dr).__name__}, {self._dr.n_boundaries} boundaries, "
                  f"bounds_version {self._dr.bounds_version}")
            print("  quantity            now [lo, hi]                 max [lo, hi]")
            # The provider's OWN dims: the No-DR floor provider carries only
            # start_height, and the other four read their static cfg bands
            # through _live_bounds.
            for _name in (d.name for d in self._dr.dims):
                print(f"    {_name:<16s}  [{_b[_name][0]:+.6f}, {_b[_name][1]:+.6f}]   "
                      f"[{_m[_name][0]:+.6f}, {_m[_name][1]:+.6f}]")
            if getattr(self._dr, "floor", None) is not None:
                print(f"  START FLOOR (SBC step 0): {self._dr.floor.f0 * 1000.0:+.1f} mm -> "
                      f"H_min {self._dr.floor.top * 1000.0:+.1f} mm in "
                      f"{self._dr.floor.n_steps} rungs; now "
                      f"{self._dr.value(autodr.FLOOR_KEY) * 1000.0:+.1f} mm, phase "
                      f"{self._dr.phase}. Below H_min every other boundary sits on its "
                      f"centre and every boundary episode is nailed to the floor.")
            # THE PREFIX SAYS PROVENANCE, NOT VERDICT (D-181, corrected
            # 2026-09-12). Each line already carries its OWN mark --
            # `[TESTWERT]` or `(sourced)`, decided per quantity in
            # `autodr.provenance_lines`. The old fixed prefix
            # `[TESTWERT-check]` announced EVERY line as a test value, the
            # sourced friction band included, which is exactly the D-181
            # correction undone one print statement along. One home for the
            # mark: the line itself.
            for _line in autodr.provenance_lines(self._dr.dims):
                print(f"  [provenance] {_line}")
            print(f"  ALL FIVE QUANTITIES ARE WIRED (Phase 5 step B5): every reset maps "
                  f"its draw onto the band above, through the one seam _live_bounds(). "
                  f"Until B5 only FRICTION did, and the other quantities read the static "
                  f"cfg fields this mode refuses non-zero -- their boundaries could reach "
                  f"their maximum while every episode ran at the centre. "
                  f"BOUNDARY SAMPLING is on: p_boundary {self._dr.p_boundary:.2f} over "
                  f"{self._dr.n_boundaries} boundaries, buffer m {self._dr.buffer_m}, "
                  f"advance at {self._dr.advance_at:.2f} / retreat at "
                  f"{self._dr.retreat_at:.2f}. The frame is there too -- _tilt_rot exists "
                  f"({self._tilt_rot is not None}), the reset writes a fixture pose "
                  f"({self._fixture_pose_reach}) and the tilt reach is "
                  f"{self._tilt_reach}. fixture_pos_noise_xy "
                  f"({float(cfg.fixture_pos_noise_xy):.4f} m), the joint noise, the "
                  f"tilt azimuth and the lateral angle lat_phi stay LIVE by design (plan "
                  f"section 1, D-178 (2)) and no boundary tracks them.")
            # NOT VERIFIED IN A SIMULATOR, and the report says so itself
            # rather than leaving the reader to assume the opposite. Every
            # claim in the paragraph above was read off the source on a
            # laptop with no Isaac Sim. The smoke run owes the evidence: the
            # `[autodr]` lines, a `dr/*_hi` curve that moves, and the
            # per-env friction read-back below.
            print("  NOT VERIFIED IN A SIMULATOR: this build has never run. Read the "
                  "[autodr] lines and dr/ curves of THIS run before believing it.")
        print(f"fixture pose noise (D-034): xy +-{cfg.fixture_pos_noise_xy:.3f} m, "
              f"yaw +-{cfg.fixture_yaw_noise_rad:.4f} rad "
              f"({'ACTIVE -- kinematic fixture teleported per reset' if (cfg.fixture_pos_noise_xy > 0.0 or cfg.fixture_yaw_noise_rad > 0.0) else 'inactive -- fixture stays at the spawn pose'})")
        # Workcell layout. The proxy printed "robot base sits on the plate top"
        # because base and pocket shared one plate; the real cell has the robot
        # AT the origin on table 1 and the fixture on table 2, 95.8 mm lower,
        # so the numbers that matter are the two table tops and the block.
        wc = insertion_tasks_cfg
        print("--- workcell (robot foot IS the env origin, Factory convention) ---")
        print(f"floor z:        {wc.FLOOR_Z:+.4f}   table 1 top: {wc.T1_TOP_Z:+.4f}   "
              f"table 2 top: {wc.T2_TOP_Z:+.4f}")
        print(f"plate drop:     {(wc.T1_TOP_Z - wc.T2_TOP_Z)*1000:.1f} mm "
              f"(measured {wc.PLATE_TOP_DROP*1000:.1f} mm, D-058 direct reading leads; "
              f"the 884/787 pair would give 97.0 mm)")
        print(f"block top:      {wc.BLOCK_TOP_Z:+.4f} = {wc.GUIDE_EDGE_ABOVE_T2*1000:.1f} mm above table 2; "
              f"{'ABOVE' if wc.BLOCK_TOP_Z > 0 else 'below'} the robot foot plane")
        print(f"insertion plane:{wc.INSERTION_PLANE_Z:+.4f} = block top - {wc.STAGE1_DEPTH*1000:.0f} mm stage 1; "
              "asset origin + entrance sit HERE (2026-08-19 convention)")
        # Reach is quoted to the POCKET, not to the block centre (2026-08-24).
        # The two differ by 49.8 mm in x, and the block centre is no longer a
        # point anything spawns at or aims for -- reporting it would name a
        # place where nothing stands.
        print(f"foot -> pocket: x {wc.POCKET_ORIGIN_X*1000:+.1f} mm (depth), y {wc.POCKET_ORIGIN_Y*1000:+.1f} mm (lateral, - = robot's right), "
              f"straight line {((wc.POCKET_ORIGIN_X**2 + wc.POCKET_ORIGIN_Y**2 + wc.BLOCK_TOP_Z**2)**0.5)*1000:.1f} mm "
              f"of 850 mm UR5e reach")
        print(f"block model:    {wc.BLOCK_MODEL_SIZE_X*1000:.1f} x {wc.BLOCK_SIZE_Y*1000:.1f} mm footprint "
              f"({wc.BLOCK_FRONT_TO_STAGE1*1000:.1f} front + {wc.POCKET_STAGE1_SIZE_X*1000:.4f} stage 1 "
              f"+ {wc.BLOCK_REAR_WALL_T*1000:.1f} rear wall), "
              f"{wc.BLOCK_BEHIND_POCKET_X*1000:.4f} mm of block behind the cut-out, "
              f"high wall {wc.BLOCK_REAR_WALL_H*1000:.0f} mm above the rim")
        print(f"tool home yaw:  {math.degrees(wc.TOOL_HOME_YAW_RAD):+.1f} deg about the tool axis "
              f"(weld yaw {math.degrees(wc.TOOL_WELD_YAW_RAD):+.1f} deg, D-078); "
              "the part's lugs face the pocket features at this angle")
        # The C4 yaw-reward warning that stood here is GONE with the proxy
        # reward (M2.4b step 3): the real reward has no yaw term at all --
        # D-109 (3) measures orientation and position on one SDF manifold.
        print(f"block boxes:    {len(wc.block_recess_boxes())} "
              f"(incl. the two +Y slot fillers beside the tab: "
              f"{(wc.POCKET_TAB_X_RANGE[0]-wc.POCKET_LOCAL_X_RANGE[0])*1000:.1f} and "
              f"{(wc.POCKET_LOCAL_X_RANGE[1]-wc.POCKET_TAB_X_RANGE[1])*1000:.1f} mm long, "
              f"{(wc.POCKET_LOCAL_Y_RANGE[1]-wc.POCKET_BODY_Y_MAX)*1000:.1f} mm deep)")
        print(f"tables usd:     {self._tables_usd_path}")
        # Loud, in the same spirit as the curriculum switch below: a run whose
        # scene has no rear wall must say so where nobody can miss it, or its
        # success rate will later be read as if the pocket had been reachable
        # only from the front.
        print("workcell block: "
              + (f"ON, usd = {self._block_usd_path}"
                 if bool(self.cfg.spawn_workcell_block)
                 else "*** OFF -- the Aufnahme stands free; no rear wall, the "
                      "pocket is enterable from any direction ***"))
        print(f"Y residual:     {wc.WORKCELL_Y_RESIDUAL*1000:+.1f} mm "
              "(left vs right table-2 overhang; right leads)")
        for _item in wc.WORKCELL_PENDING:
            print(f"*** PENDING *** {_item}")
        if wc.WORKCELL_PENDING:
            print("*** the cell is internally consistent but NOT absolutely placed ***")
        print("--- end workcell ---")
        print(f"joint_names: {self.robot.joint_names}")
        # DIAGNOSTIC (2026-08-21, RT-2): wrist_3_joint reads -1.0753 rad two
        # steps after a reset that wrote -1.5701, with FD speed ~0. The joint
        # never moved -- something rejected the written value. Print the limits
        # the asset carries next to what the reset writes: if the limit is
        # -1.0753 the value was clamped; if it is wider, the write itself is
        # the suspect. One line, measurement only, nothing is changed by it.
        _lim = self.robot.data.soft_joint_pos_limits[0]
        _def = self.robot.data.default_joint_pos[0]
        for _j, _name in enumerate(self.robot.joint_names):
            print(f"  joint limit  {_name:<20} [{_lim[_j, 0]:+.6f}, {_lim[_j, 1]:+.6f}]  "
                  f"reset writes {_def[_j]:+.6f}"
                  f"{'   <-- OUTSIDE' if not (_lim[_j, 0] <= _def[_j] <= _lim[_j, 1]) else ''}")
        print(f"body_names:  {self.robot.body_names}")
        print(f"ee_body:     {cfg.ee_body_name} (index {self._ee_body_idx})")
        # The wrench link, named in every run. It was a blocker until RT-59
        # measured it, and the one way it can go wrong now is silently: a
        # wrench read at the wrong index still produces a plausible channel.
        # Printing the resolved index next to the name is what makes that
        # visible in the log instead of in the reward.
        if self._force_body_idx is None:
            print(f"force_body:  *** MISSING *** no welded tool, so '{cfg.force_sensor_body_name}' "
                  "was not resolved; observation channels 25:28 stay at their reset zero.")
        else:
            print(f"force_body:  {cfg.force_sensor_body_name} (index {self._force_body_idx}), "
                  f"wrench read RAW in the parent body frame, EMA alpha "
                  f"{cfg.ft_smoothing_factor} on the NEW sample")
            if self._force_parent_idx is None:
                print("force tare:  OFF -- the channel carries the tool's own weight "
                      f"({'force_gravity_tare = False' if not cfg.force_gravity_tare else 'no welded tool'}). "
                      "FORGE's channel does not, because Factory/FORGE disable gravity.")
            else:
                _hold = [float(v) for v in self._hold_force_w.tolist()]
                print(f"force tare:  ON -- world hold force {[round(v, 4) for v in _hold]} N "
                      f"(= -{cfg.tool_mass_kg:.5f} kg * gravity {tuple(cfg.sim.gravity)}), "
                      f"rotated into the frame of '{cfg.ee_body_name}' "
                      f"(index {self._force_parent_idx}) and subtracted BEFORE the EMA. "
                      f"Free air must now read ~0 N, not {abs(_hold[2]):.2f} N.")
                print(f"torque tare: ON -- lever = authored tool COM minus the MEASURED reference "
                      f"point (origin of '{cfg.ee_body_name}' + {tuple(cfg.torque_ref_offset_parent_m)} m "
                      f"in its frame, RT-192a2), moment = lever x hold force, rotated into the same "
                      f"frame, subtracted BEFORE the EMA (D-188).")
        print(f"obs mode {insertion_math.obs_version(self._obs_mode)}: "
              f"{'torque 28:31 IN the policy observation' if self._obs_mode == 'wrench' else 'torque read and logged, NOT in the policy observation'}; "
              f"torque_obs_noise_std_nm {cfg.torque_obs_noise_std_nm} (0 = off, OPEN per D-188); "
              f"the wrench EMA skips the reset step and advances once per step (inbox 2026-09-13).")
        spawn_cfg = cfg.robot_cfg.spawn
        declared = (
            "activate_contact_sensors" in {f.name for f in dataclasses.fields(spawn_cfg)}
            if dataclasses.is_dataclass(spawn_cfg)
            else None
        )
        print(f"activate_contact_sensors: {getattr(spawn_cfg, 'activate_contact_sensors', 'ABSENT')} "
              f"({'declared cfg field' if declared else 'NOT A DECLARED FIELD' if declared is False else 'undetermined'})")
        # Read BACK from PhysX, not echoed from the constant: __init__ wrote
        # CONTACT_FRICTION into every material (D-111); if any material did
        # not take it, min and max diverge here and the line says so.
        _fric_r = self.robot.root_physx_view.get_material_properties()[..., 0:2]
        _fric_f = self._fixture.root_physx_view.get_material_properties()[..., 0:2]
        # THE HEADER DEPENDS ON THE REACH since B4. With a friction reach the
        # 0.4 is only the CENTRE the band opens around, and no material need
        # carry it any more; printing "u = 0.4 written" over a pooled range
        # that spans the whole band would be a false claim in the report the
        # smoke run is read from.
        _fric_head = (
            f"centre {insertion_tasks_cfg.CONTACT_FRICTION}, per-reset band, PROVISIONAL/F3"
            if self._friction_reach
            else f"u = {insertion_tasks_cfg.CONTACT_FRICTION} written once, PROVISIONAL/F3"
        )
        print(f"contact friction (D-111, {_fric_head}): "
              f"PhysX reads back robot [{_fric_r.min():.3f}, {_fric_r.max():.3f}], "
              f"fixture [{_fric_f.min():.3f}, {_fric_f.max():.3f}] (static+dynamic pooled)")
        # PER RESET SINCE PHASE 5 STEP B4, and this is the line that proves it.
        # The pooled min/max above cannot: it is one range over all envs and
        # all shapes, so four envs at four different coefficients and four envs
        # at one look the same in it as long as the range matches. These four
        # are read out of PhysX PER ENV, after the first reset has run, and are
        # printed beside what `_apply_friction` says it commanded. Equal
        # numbers across env 0..3 while the band is wider than a point mean the
        # write did NOT land -- the exact silent failure the partial-reset call
        # form can produce.
        if self._friction_reach:
            _n_rep = min(4, self.num_envs)
            _lo, _hi = self._dr.bounds()["friction"]
            _read = [float(_fric_r[_e, :, 0].mean()) for _e in range(_n_rep)]
            # THE FIXTURE TOO, per env, and this half is not decoration: the
            # pair value equals the drawn value only while both bodies carry
            # it. A robot list that varies beside a flat fixture list is the
            # "only one asset got written" failure, and no pooled range shows
            # it.
            _read_f = [float(_fric_f[_e, :, 0].mean()) for _e in range(_n_rep)]
            _cmd = [float(v) for v in self._friction_applied[:_n_rep].tolist()]
            print(f"friction after first reset env0..{_n_rep - 1}: PhysX robot "
                  f"{[round(v, 5) for v in _read]} vs commanded "
                  f"{[round(v, 5) for v in _cmd]}; band now "
                  f"[{_lo:.5f}, {_hi:.5f}] -- at width 0 all four are the "
                  f"centre and that is CORRECT, four DIFFERENT values are owed "
                  f"only once the band has opened")
            print(f"  fixture side, same envs: {[round(v, 5) for v in _read_f]} "
                  f"-- must match the robot list env for env")
            _w = list(self._friction_write_ms)
            print(f"friction write cost mean/max ms: "
                  f"{(sum(_w) / len(_w) if _w else 0.0):.3f} / "
                  f"{(max(_w) if _w else 0.0):.3f} over {len(_w)} sample(s) in a "
                  f"{self._friction_write_ms.maxlen}-call window (whole block: host "
                  f"copy + row edit + both PhysX calls, fenced against the reset). "
                  f"Read it against the iteration time in THIS log.")
        else:
            print(f"friction per reset: OFF -- no friction reach (dr_mode="
                  f"{cfg.dr_mode}). Every env keeps the one-shot D-111 value "
                  f"for the whole run and no reset writes a material.")
        # D-115 [verify on site]: the effective solver iteration counts.
        # NAMED CONFLICT, decided by reading the stage rather than by
        # plausibility: D-115 records "our scene sets neither", while
        # ur5e_cfg.py (articulation_props) authors 16/1 on the robot's
        # articulation root. Every prim under env_0's Robot and Fixture that
        # carries an authored value is printed; a prim without one falls to
        # the PhysX SDK default, which this read cannot see -- that case
        # prints as "no authored value", never as a number.
        try:
            from pxr import Usd as _Usd  # noqa: PLC0415
            from isaaclab.sim.utils.stage import get_current_stage as _gcs  # noqa: PLC0415
            _stage = _gcs()
            _attr_names = (
                "physxArticulation:solverPositionIterationCount",
                "physxArticulation:solverVelocityIterationCount",
                "physxRigidBody:solverPositionIterationCount",
                "physxRigidBody:solverVelocityIterationCount",
            )
            # ONE FLAG PER ROOT, and the reason is a defect this shape already
            # produced: with a single shared flag the robot's authored value
            # silenced the fallback line, so RT-50 printed ONE line for the
            # Robot and NOTHING for the Fixture. A reader could not tell
            # "walked it, found nothing" from "never looked". Each root now
            # reports its own result, always, so silence is impossible.
            for _root_path in ("/World/envs/env_0/Robot", "/World/envs/env_0/Fixture"):
                _root = _stage.GetPrimAtPath(_root_path)
                if not _root or not _root.IsValid():
                    print(f"solver iterations (D-115 verify): {_root_path}: prim not found")
                    continue
                _root_authored = False
                for _p in _Usd.PrimRange(_root):
                    _vals = []
                    for _an in _attr_names:
                        _a = _p.GetAttribute(_an)
                        if _a and _a.IsValid() and _a.HasAuthoredValue():
                            _vals.append(f"{_an.split(':', 1)[1]}={_a.Get()}")
                    if _vals:
                        _root_authored = True
                        print(f"solver iterations (D-115 verify): {_p.GetPath()}: {', '.join(_vals)}")
                if not _root_authored:
                    print(f"solver iterations (D-115 verify): {_root_path}: NO authored value on any "
                          "prim below -- the PhysX SDK default applies and is NOT readable here")
        except Exception as _exc:  # noqa: BLE001 -- report line, never a crash
            print(f"solver iterations (D-115 verify): UNCHECKED -- {type(_exc).__name__}: {_exc}")
        if self._peg_body_idx is None:
            print(f"peg_body:    *** MISSING *** '{cfg.peg_body_name}' is not among the bodies above.")
            print( "             The welded link did not survive authoring; every peg number below is void.")
        else:
            print(f"peg_body:    {cfg.peg_body_name} (index {self._peg_body_idx}), "
                  f"flange -> leading point {cfg.peg_length:.4f} m, "
                  f"part {cfg.part_bbox_m[0]:.4f} x {cfg.part_bbox_m[1]:.4f} x {cfg.part_bbox_m[2]:.4f} m, "
                  f"mass {'PENDING (not weighed)' if cfg.tool_mass_kg is None else f'{cfg.tool_mass_kg:.5f} kg'}")
            # Named because it is about to print numbers that assume otherwise:
            # the gate, the corner Chebyshev and the clearance below all model
            # the held body as a SQUARE of cfg.held_asset.side. The real part is
            # rectangular, 96.4 x 143.5 mm, and its long axis is not represented
            # anywhere in that math. Those lines describe the proxy peg, not the
            # part, until the rectangular rework lands.
            print(f"             GATE MATH IS STILL SQUARE: side {cfg.held_asset.side:.4f} m from the proxy "
                  f"peg; the part's {cfg.part_bbox_m[1]:.4f} m long axis is not modelled yet.")
            # THE LINE THAT MISLED RT-70, rebuilt 2026-08-28. It used to print
            # the proxy's single square clearance and its C4 yaw window as if
            # both described the user's part. Two of the five numbers now come
            # from the REAL measured geometry, and the two that have no real
            # counterpart say so instead of printing a proxy number:
            #   play      -> PLAY_X / PLAY_Y, measured, and there are TWO of
            #                them because the real pair is not square. They are
            #                TOTAL play, not the proxy's half-clearance, so the
            #                word changed with the number.
            #   yaw       -> no real value exists. D-106's bound was reopened
            #                by D-121 and is not recomputed, so the report
            #                states the gap rather than the proxy's window.
            #   side      -> still the square gate's, still read by it, and
            #                labelled through PROXY_TASK_VALUES below.
            # depth/success come from the fixture cfg and are printed here
            # rather than typed, so S2/S3 move them without touching this line.
            print(f"pocket:      side {cfg.fixed_asset.side:.4f} m (PROXY SQUARE, see the proxy block below), "
                  f"depth {cfg.fixed_asset.depth:.4f} m (blind), "
                  f"success at {cfg.fixed_asset.success_depth*1000:.0f} mm")
            print(f"             play (MEASURED, total, not per-axis half): "
                  f"X {wc.PLAY_X*1000:.4f} mm across the lugs, "
                  f"Y {wc.PLAY_Y*1000:.4f} mm along the part")
            print( "             yaw window: OPEN -- the proxy's C4 window does not apply to a part "
                   "that fits one way round, and D-106's bound is not recomputed against PLAY_X (D-121).")
        print(f"asset check (peg):     {self._asset_geometry_line()}")
        print(f"asset check (fixture): {self._fixture_geometry_line()}")
        # Resting finite-differenced joint speed. This is the demo sprint's
        # substitute for the solver-flag experiment: it must read ~0.000 at
        # the early/late reports below (velocity derived from positions, not
        # from the raw solver channel), unlike the raw joint_vel channel's
        # measured ~0.05 rad/s phantom (D-030 addendum 2).
        resting_speed = torch.linalg.norm(joint_vel_fd[: min(2, self.num_envs)], dim=-1)
        print(f"resting FD joint speed (env 0..1): {resting_speed.tolist()} rad/s  (expect ~0.000 after reset)")
        # Printed from OBS_SLICES rather than typed out, so this line cannot
        # describe a layout the env does not build. The NOT-BUILT marker is
        # DERIVED from ``observation_space`` -- a block is missing exactly when
        # it starts beyond the width the env actually reports.
        #
        # It was hardcoded on the name "force" until RT-65, and by then it was
        # a lie: that run built all 28 channels and its own report still said
        # "25:28 force <-- MISSING (scene stream S2)". The label outlived the
        # gap it described by one commit, because a name cannot know whether
        # the block behind it exists. A width can. This is the same reason
        # ``observation_space`` is pinned to ``OBS_DIM`` in the wiring check
        # rather than written as a literal.
        _built = int(self.cfg.observation_space)
        _table = insertion_math.obs_slices(self._obs_mode)
        obs_blocks = " | ".join(
            f"{lo}:{hi} {name}" + ("  <-- NOT BUILT" if lo >= _built else "")
            for name, (lo, hi) in _table.items()
        )
        print(f"observation layout (built {self.cfg.observation_space}-dim of "
              f"{insertion_math.obs_dim(self._obs_mode)}, mode {self._obs_mode}, "
              f"D-107/D-114/D-188): {obs_blocks}")
        print(f"  12:15 frame: {'POCKET, D-036' if self._tilt_active else 'env-local'}; "
              "19:21 is (cos phi, sin phi), NOT cos 4 phi (D-107 (2))")
        # THE OBSERVATION SCATTER, READ BACK (D-182, D-183). Until 2026-09-12
        # nothing in this env ever looked at either buffer: the noise model is
        # the BASE class's `self._observation_noise_model`, built inside
        # `DirectRLEnv.__init__` and only when the cfg field is set
        # (direct_rl_env.py:210-213), and the per-episode offset it holds is
        # the library's `_bias`, (num_envs, OBS_DIM) because
        # `obs_noise.InsertionObsNoise` pre-sizes it (noise_model.py:157).
        # Sliced through OBS_SLICES rather than typed, same reason as the
        # layout line above.
        #
        # WHAT THESE TWO LINES ARE FOR. `operation="abs"` on the bias cfg is
        # load-bearing: with the library default `"add"`, `reset()` would ADD
        # a fresh draw ONTO the old bias, so the pocket offset would random
        # walk away over a run instead of being re-drawn per episode
        # (obs_noise.py header). The report runs once at every
        # `cfg.report_at_steps` count, so the printouts can be held against
        # each other.
        # IT DISCRIMINATES ONLY ACROSS A RESET, AND THE TUPLE NOW STRADDLES
        # ONE (2026-09-12, warrant corrected the same day after critic round 2
        # (2)). `_bias` is written in `reset()` and nowhere else, so two
        # printouts with no reset between them are identical under BOTH
        # operations. THE PAIR TO COMPARE IS COUNT 2 AGAINST COUNT 258, and
        # only that pair.
        # WHAT IS NOT TRUE: that counts 2, 60 and 180 are identical BY
        # CONSTRUCTION because they all sit inside episode 0. Under
        # `train.py` that is false -- `train.py:725` passes
        # `init_at_random_ep_len=True`, so rsl_rl draws every env's
        # `episode_length_buf` from `randint(0, 256)` before the first step
        # (on_policy_runner.py:66-69) and the envs reset at DIFFERENT steps.
        # This report prints envs 0..3, and by count 60 the odds that at least
        # one of the four has already reset are better than even (the figure is
        # worked out where the counts live, `insertion_env_cfg.report_at_steps`).
        # A LINE THAT CHANGES BEFORE COUNT 258 IS AN EARLY RESET, NOT A
        # FINDING.
        # WHY 258 STILL HOLDS: the latest possible first reset belongs to the
        # env that drew 0, and `_obs_calls` runs 3 ahead of the step count
        # under `train.py` (the wrapper's own reset, the runner constructor,
        # then `learn`'s read before its first step). That lands on 258 -- past
        # every first reset with MARGIN ZERO, and the reset runs before
        # `_get_observations` in the same `step` (direct_rl_env.py:398-410), so
        # 258 already shows the new draw. The derivation has ONE home and it is
        # `insertion_env_cfg.report_at_steps`.
        # A bias that is RE-DRAWN moves inside a fixed spread (same sigma,
        # different numbers, no trend); a DRIFTING one grows, which is what
        # `operation="add"` would do. Reading 2 against 60 or 180 proves
        # nothing and never did.
        _n_obs = min(4, self.num_envs)
        _noise = getattr(self, "_observation_noise_model", None)
        if _noise is None:
            print("observation bias (D-182): OFF -- both sigmas are 0.0, so "
                  "resolve_obs_noise_model returned None and no model exists")
        else:
            _t_lo, _t_hi = insertion_math.OBS_SLICES["tip_rel"]
            _bias_rows = [[round(v, 6) for v in _row]
                          for _row in _noise._bias[:_n_obs, _t_lo:_t_hi].tolist()]
            print(f"observation bias env0..{_n_obs - 1} (D-182, tip_rel "
                  f"{_t_lo}:{_t_hi}, uniform +-{cfg.obs_noise_pocket_pos_std_m} m, "
                  f"drawn per EPISODE): {_bias_rows}")
        print(f"grasp belief error env0..{_n_obs - 1} (D-183, "
              f"+-{cfg.grasp_obs_offset_x_m} m on the part's short axis, drawn "
              f"per EPISODE): "
              f"{[round(v, 6) for v in self._grasp_obs_off[:_n_obs, 0].tolist()]}")
        # The switch and the guard, on one line, because a run whose reward is
        # zero must say so where nobody can miss it.
        print(f"rl_terms_enabled: {self._rl_terms_enabled} "
              + ("(validate_rl_config PASSED)" if self._rl_terms_enabled
                 else "*** MEASUREMENT ENV: no real reward, no force abort, no curriculum; "
                      "guard SKIPPED -- never a training run ***"))
        # The ladder switch, on its own line and in the same spirit as the one
        # above: a run without a curriculum must say so, or its success rate
        # will later be read as if a ladder had produced it. D-110 (3).
        print("curriculum: "
              + (f"ON, rung_step_sizes = {self.cfg.rung_step_sizes}"
                 if bool(self.cfg.curriculum_enabled)
                 else "*** OFF -- FIXED RUNG 0, no advance, no retreat; the start "
                      "scatter is whatever the cfg holds and never widens ***"))
        # The live reward/termination call sites (M2.4b step 3). Printed only
        # when the queries exist -- in a measurement env there is nothing to
        # describe and the MEASUREMENT line above already says why.
        if self._rl_terms_enabled and self._sdf_query is not None:
            print("reward (D-109 + the 2026-09-01 revision): kernel_sum(sdf_dist) "
                  "+ engaged + success bonuses, SAPU-scaled; "
                  f"time penalty {self.cfg.time_penalty_per_step:+.6f}/step "
                  f"(-1/T at T = {self.cfg.episode_steps}); action-rate penalty "
                  f"-{self.cfg.action_rate_scale} * ||a_t - a_(t-1)||")
            # RT-171: its own line -- a run
            # with the alignment term and one without must be told apart in
            # the log, and the gamma the row used must be visible next to it.
            print("alignment shaping (RT-171, potential-based): "
                  f"{self.cfg.shaping_gamma} * Phi(s') - Phi(s), Phi = "
                  f"{self.cfg.w_tilt} * sech({self.cfg.kernel_a_tilt:.4f}/rad * theta) "
                  f"vs the POCKET axis (margin {math.degrees(insertion_tasks_cfg.TILT_MARGIN_RAD):.2f} deg, "
                  "MEASURED, RT-158 mouth-mode median), Phi(s') = 0 on the success "
                  "step, not SAPU-scaled; gamma READ from agents/rsl_rl_ppo_cfg.py "
                  "-- a hydra override of agent.algorithm.gamma must set "
                  "env.shaping_gamma too; "
                  + ("*** OFF, w_tilt 0 -- the pre-RT-171 reward ***"
                     if float(self.cfg.w_tilt) == 0.0 else "ON"))
            print("termination (D-113 + the 2026-09-01 revision): timeout, force "
                  f"abort at {self.cfg.force_abort_f_max_n} N (payment "
                  f"{self.cfg.abort_payment}, both PLACEHOLDERS, see the block "
                  "above), AND success -- which ends the episode and pays its "
                  "remaining return as a lump. Episode success = that "
                  "termination event.")
            _g_pos, _g_quat = insertion_math.seated_goal_pose(
                self._entrance_pos,
                self._fixture_quat,
                self._seat_quat_local,
                self._tip_offset_local,
                float(self.cfg.depth_max),
            )
            print(f"seated goal pose (env 0, env-local): "
                  f"{[round(v, 6) for v in _g_pos[0].tolist()]}, "
                  f"quat (wxyz) {[round(v, 6) for v in _g_quat[0].tolist()]} "
                  f"= pocket quat * SEATED_TOOL_QUAT_LOCAL (RT-102)")
        # Numbers nobody measured on this setup (the legend lives with
        # ``RL_PLACEHOLDERS`` in insertion_env_cfg.py and only there -- some
        # are invented, some borrowed, some read off a published figure).
        # Printed FROM the dict rather than typed out, so this
        # block cannot describe a state the cfg does not hold. It exists because
        # an unset value stops a run and an unmeasured one does not: the only
        # remaining defence is that nobody can read a result off this run without
        # also seeing what the result was produced under. Same dict reaches
        # demo_metrics.json via _write_metrics.
        if RL_PLACEHOLDERS:
            print("*** PLACEHOLDER VALUES IN USE -- every number this run reports "
                  "is provisional ***")
            for name, mark in RL_PLACEHOLDERS.items():
                print(f"    {name} = {getattr(self.cfg, name)}  {mark}")
        # The proxy block, same shape and same reason as the placeholder block
        # above: printed FROM the table so it cannot describe a state the cfg
        # does not hold. It is a SEPARATE block because a proxy number is not a
        # placeholder -- it was measured, on a different task -- and saying
        # "invented" about a measured number would be its own falsehood.
        _proxy = resolve_proxy_task_values(self.cfg)
        if _proxy:
            print("*** PROXY TASK VALUES IN USE -- the physics is the real part in the real "
                  "pocket, but these SCORING numbers still describe the square peg ***")
            for path, value, mark in _proxy:
                print(f"    {path} = {value}  {mark}")
            print(f"    ... and demo_metrics.json keys {sorted(PROXY_METRICS_KEYS)} carry them.")
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
            axes = insertion_math.axes_from_quat(ee_quat_w.unsqueeze(0))[0]
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
            # This measures the FLANGE above the entrance, so its expectation is
            # the tool point's height plus the tool chain -- not home_standoff_z,
            # which is the tool POINT above the BLOCK TOP. Two corrections in one
            # line: +STAGE1_DEPTH for the plane, +FLANGE_TO_PART_BOTTOM for the
            # chain. Printing 0.150 here would flag a correct pose as 167 mm off.
            expect_flange_z = self._start_tip_height + cfg.peg_length
            # Closed-form UR5e FK on the home angles gives 0.317000 m exactly.
            # Two effects can move the printed number away from it:
            #
            #   * reset_joint_noise of +-0.01 rad on five joints scatters it by
            #     up to +-0.0098 m (4000 samples, measured offline 2026-08-23).
            #     Currently INERT -- reset_joint_noise is 0.0 in
            #     insertion_env_cfg.py. It returns the moment the noise does.
            #   * gravity droop. Since the drives come from the USD (supervisor
            #     decision 2026-08-25, ur5e_cfg.py) this is no longer a term
            #     worth budgeting for: RT-46 measured 0.317026 m at t = 3.000 s,
            #     i.e. 0.000026 m off nominal, and unchanged between step 60 and
            #     step 180.
            #
            # SUPERSEDED, for anyone reading an older log: under the UR10e
            # placeholder gains the same line read 0.309950 m at step 2 and
            # 0.290079 m from step 60 on (RT-17 env 0), a droop of -0.0199 m.
            print(f"  Z standoff (flange):   {z_standoff.item():.6f} m  (expect {expect_flange_z:.4f} at the "
                  f"NOISELESS start pose; tool point "
                  f"{self._start_tip_height:.4f} + chain {cfg.peg_length:.4f}"
                  f"{'' if self.cfg.start_tip_above_entrance is None else ', D-161 rung 0 -- NOT the home pose'})")
            print(f"                         with USD drives the droop is spent: RT-46 read 0.317026 m at "
                  f"t = 3.000 s, 0.000026 m off nominal. Reset noise would add +-0.0098 m, but it is 0.0 "
                  f"right now -- so judge this against the CENTRE")
            for name, col in zip("xyz", range(3)):
                dot = torch.dot(axes[:, col], down).item()
                note = "  <-- expect >= 0.999 here" if name == "z" else ""
                print(f"  {name}-axis . (0,0,-1):    {dot:+.6f}{note}")
            if self._peg_body_idx is not None:
                peg_pos_w = self.robot.data.body_pos_w[e, self._peg_body_idx]
                peg_quat = self.robot.data.body_quat_w[e, self._peg_body_idx]
                peg_axes = insertion_math.axes_from_quat(peg_quat.unsqueeze(0))[0]
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
                    depth = -tip_rel_e[2]
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
                # The home tip height above the ENTRANCE, which is not the
                # standoff: HOME_STANDOFF_Z is quoted above the block top and
                # the entrance sits STAGE1_DEPTH lower. It also no longer
                # subtracts the tool length -- since the tool chain was welded
                # on, the standoff IS the leading tool point's height, so the
                # old `standoff - length` would subtract it a second time.
                home_tip_height = self._start_tip_height
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
                # The free-yaw window used to be quoted here. It is the square
                # pair's C4 window and there is no real counterpart, so the
                # expectation now names the reset noise, which is what the
                # number is actually read against.
                print(f"  peg yaw phi (folded):  {math.degrees(phi):+.3f} deg  "
                      f"(expect +-0.000 at home BEFORE reset noise; "
                      f"~uniform in +-{math.degrees(cfg.reset_yaw_noise):.2f} after)")
                print(f"  (cos 4phi, sin 4phi):  ({math.cos(4.0 * phi_raw):+.4f}, {math.sin(4.0 * phi_raw):+.4f})"
                      f"  (expect (+1, 0) at home before reset noise)")
                # The bound quoted here was the proxy's half-clearance. The
                # real limit on this offset is HALF the measured across-play,
                # because the play is the total gap and the part can only use
                # half of it on either side.
                print(f"  tip Chebyshev offset:  {tip_cheb:.6f} m  "
                      f"(real limit is half the across-play, {insertion_tasks_cfg.PLAY_X/2.0:.6f} m; "
                      f"at the start pose under this tilt ~{abs(home_tip_height) * tilt_s:.4f})")
                print(f"  corner Chebyshev max:  {cheb:.6f} m  (gate needs < {half_pocket:.4f}); "
                      f"corners rel. entrance: {[ [round(v, 4) for v in c.tolist()] for c in corners ]}")
                print(f"  gate at this pose:     {'OPEN' if gate_e else 'closed'} "
                      f"(corners {'in' if cheb < half_pocket else 'OUT'}, "
                      f"depth {depth.item():+.4f} vs <= {cfg.fixed_asset.depth + 0.001:.4f}, "
                      f"alignment {alignment_e:+.4f} vs >= {cfg.gate_min_alignment})")
                print(f"  tip insertion depth:   {depth.item():+.6f} m  "
                      f"(expect {-home_tip_height * tilt_c:+.4f} at the NOISELESS home pose, judged the same "
                      f"way as the flange standoff above"
                      f"{'' if self._tilt_rot is None else f', = {-home_tip_height:+.4f} x cos(tilt)'})")

            dev = (self.robot.data.joint_pos[e] - commanded[e]).abs()
            worst = int(torch.argmax(dev).item())
            print(f"  max joint deviation:   {dev[worst].item():.6f} rad "
                  f"({torch.rad2deg(dev[worst]).item():.3f} deg) on {self.robot.joint_names[worst]}"
                  f"  (vs the WRITTEN target; expect ~0)")
            yaw_off = (self._joint_targets[e, self._yaw_joint_idx]
                       - home_pose[e, self._yaw_joint_idx])
            print(f"  reset yaw noise:       {yaw_off.item():+.6f} rad "
                  f"({torch.rad2deg(yaw_off).item():+.3f} deg) on {cfg.yaw_joint_name}"
                  f"  (D-038, sampled +-{cfg.reset_yaw_noise:.4f} rad -- NOT an error)")
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
