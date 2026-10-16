# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
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
    validate_table_mode,
)
from .insertion_tasks_cfg import (
    resolve_workcell_block_usd_path,
    resolve_workcell_tables_usd_path,
)

CODE_MARKER = "insertion-osc-2026-09-03a"

EPISODE_LOG_COLUMNS = (
    "run", "seed", "iteration", "env_id", "episode", "outcome", "steps",
    "ip_max_mm", "ip_steps_ge_t1", "force_max_filtered_n", "force_max_raw_n",
    "bounds_version", "dr_phase",
)

APPLIED_LOG_COLUMNS = (
    "row_id", "lat_x_m", "lat_y_m", "start_height_m", "yaw_rad",
    "pocket_qw", "pocket_qx", "pocket_qy", "pocket_qz",
    "pocket_dx_m", "pocket_dy_m", "pocket_dz_m", "grasp_obs_x_m", "pocket_obs_yaw_rad",
    "friction_cmd",
    "friction_robot_min", "friction_robot_max",
    "friction_fixture_min", "friction_fixture_max",
)


class InsertionEnv(DirectRLEnv):
    """UR5e insertion with task-space pose deltas and operational-space control."""

    cfg: InsertionEnvCfg

    def __init__(self, cfg: InsertionEnvCfg, render_mode: str | None = None, **kwargs):
        self._control_mode = resolve_control_mode(cfg)
        self._obs_mode = resolve_obs_layout(cfg)
        cfg.observation_noise_model = obs_noise.resolve_obs_noise_model(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        self._rl_terms_enabled = bool(getattr(self.cfg, "rl_terms_enabled", True))
        if self._rl_terms_enabled:
            validate_rl_config(self.cfg)

        _mat_env_ids = torch.arange(self.scene.num_envs, device="cpu")
        self._friction_bufs: list = []
        for _asset in (self.robot, self._fixture):
            _mats = _asset.root_physx_view.get_material_properties().clone()
            _mats[..., 0] = insertion_tasks_cfg.CONTACT_FRICTION
            _mats[..., 1] = insertion_tasks_cfg.CONTACT_FRICTION  # dynamic
            _asset.root_physx_view.set_material_properties(_mats, _mat_env_ids)
            self._friction_bufs.append((_asset, _mats))
        _dtypes = {_b.dtype for _, _b in self._friction_bufs}
        if len(_dtypes) != 1:
            raise ValueError(
                f"the robot and fixture material buffers have different dtypes {_dtypes}; "
                "_apply_friction casts the drawn friction once for both"
            )
        self._friction_buf_dtype = next(iter(_dtypes))

        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]

        try:
            self._peg_body_idx = self.robot.find_bodies(self.cfg.peg_body_name)[0][0]
        except (IndexError, KeyError, ValueError):
            self._peg_body_idx = None

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
            g = self.cfg.sim.gravity
            self._hold_force_w = torch.tensor(
                [-float(self.cfg.tool_mass_kg) * float(c) for c in g], device=self.device
            )

        self._report_at_obs_calls = tuple(self.cfg.report_at_steps)
        self._obs_calls = 0

        self._entrance_base = torch.tensor(cfg.opening_entrance_pos, device=self.device)
        self._entrance_pos = self._entrance_base.unsqueeze(0).repeat(self.num_envs, 1)
        self._fixture_yaw = torch.zeros(self.num_envs, device=self.device)
        self._tip_offset_local = torch.tensor(cfg.peg_tip_offset, device=self.device)
        self._grasp_obs_off = torch.zeros(self.num_envs, 3, device=self.device)
        self._pocket_obs_yaw = torch.zeros(self.num_envs, device=self.device)
        self._seat_quat_local = torch.tensor(
            insertion_tasks_cfg.SEATED_TOOL_QUAT_LOCAL, device=self.device
        ).unsqueeze(0)

        self._start_lat_half = resolve_start_lateral_offset(cfg.start_lateral_offset)

        self._dr = None
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
            self._dr = autodr.AutoDR(
                autodr.bind_centres(
                    {"start_height": float(cfg.start_tip_above_entrance)},
                    dims=tuple(d for d in autodr.DR_DIMS if d.name == "start_height"),
                ),
                floor=_floor_spec,
                stall_buffers=int(cfg.autodr_stall_buffers),
            )
        if str(cfg.dr_mode) in ("autodr", "fixed_dr", "table"):
            if cfg.start_tip_above_entrance is None:
                raise ValueError(
                    "dr_mode='autodr' with start_tip_above_entrance = None (the home-pose "
                    "measurement setting): start height is one of the five randomised "
                    "quantities and its width-0 centre is that field. There is no start "
                    "pose to open a band around."
                )
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
                raise ValueError(
                    f"dr_mode='autodr' together with static randomisation {_live}: an "
                    "AutoDR boundary already owns each of these quantities. Set these "
                    "fields to 0.0 (start_tip_above_entrance_low: equal to "
                    "start_tip_above_entrance) and let the boundaries open, or use "
                    "dr_mode='off'. fixture_pos_noise_xy, the joint noise and the tilt "
                    "azimuth are NOT in this set -- no boundary tracks them and they stay "
                    "live (plan section 1)."
                )
        self._eval_table = None
        if str(cfg.dr_mode) == "table":
            validate_table_mode(cfg)
            self._eval_table = autodr.RowTable.from_csv(
                cfg.dr_table_path, allow_outside=bool(cfg.dr_table_allow_outside)
            )
            if self._eval_table.n_rows != int(self.num_envs):
                raise ValueError(
                    f"dr_mode='table': {cfg.dr_table_path} has "
                    f"{self._eval_table.n_rows} rows and the env has {int(self.num_envs)} "
                    "envs. Env i plays row i; the two counts must be equal."
                )
            _table_dims = autodr.bind_centres(
                {
                    "lat_r": 0.0,
                    "yaw": 0.0,
                    "tilt": 0.0,
                    "start_height": float(cfg.start_tip_above_entrance),
                    "friction": insertion_tasks_cfg.CONTACT_FRICTION,
                }
            )
            if str(cfg.dr_table_width) == "nominal":
                _table_dims = tuple(d for d in _table_dims if d.name == "start_height")
            self._dr = autodr.FixedWidth.at_max(_table_dims)
        elif str(cfg.dr_mode) == "autodr":
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
        elif str(cfg.dr_mode) == "fixed_dr":
            self._dr = autodr.FixedDR(
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
                "(static cfg fields), 'autodr', 'fixed_dr' (D-192) and 'table' (the "
                "evaluation). Anything "
                "else is refused rather than silently run as 'off', which would train or "
                "score a policy against the wrong distribution."
            )

        if self._dr is not None:
            _reach = self._dr.bounds_max()
            _yaw_r = _reach.get("yaw", (-float(cfg.fixture_yaw_noise_rad), float(cfg.fixture_yaw_noise_rad)))
            _tilt_r = _reach.get("tilt", (0.0, float(cfg.fixture_tilt_noise_rad)))
            _yaw_span = _yaw_r[1] - _yaw_r[0] > 0.0
            _tilt_span = _tilt_r[1] - _tilt_r[0] > 0.0
            _angle_reach = _yaw_span or _tilt_span
            _fixture_pose_reach = (_angle_reach or float(cfg.fixture_pos_noise_xy) > 0.0
                                   or float(cfg.fixture_pos_noise_z) > 0.0)
        else:
            _angle_reach = (
                float(cfg.fixture_tilt_rad) != 0.0
                or float(cfg.fixture_tilt_noise_rad) != 0.0
                or float(cfg.fixture_yaw_rad) != 0.0
                or float(cfg.fixture_yaw_noise_rad) != 0.0
            )
            _fixture_pose_reach = (
                float(cfg.fixture_pos_noise_xy) > 0.0
                or float(cfg.fixture_pos_noise_z) > 0.0
                or float(cfg.fixture_yaw_noise_rad) > 0.0
                or float(cfg.fixture_tilt_noise_rad) > 0.0
            )
        self._fixture_pose_reach = _fixture_pose_reach

        if self._dr is not None:
            _fric = self._dr.bounds_max().get(
                "friction",
                (insertion_tasks_cfg.CONTACT_FRICTION, insertion_tasks_cfg.CONTACT_FRICTION),
            )
            self._friction_reach = _fric[1] - _fric[0] > 0.0
        else:
            self._friction_reach = False

        if self._dr is not None:
            _tilt = self._dr.bounds_max().get("tilt", (0.0, float(cfg.fixture_tilt_noise_rad)))
            self._tilt_reach = _tilt[1] - _tilt[0] > 0.0
        else:
            self._tilt_reach = float(cfg.fixture_tilt_noise_rad) > 0.0

        self._fixture_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._fixture_quat[:, 0] = 1.0
        self._tilt_active = _angle_reach
        if self._tilt_active:
            deterministic = float(cfg.fixture_tilt_rad) != 0.0 or float(cfg.fixture_yaw_rad) != 0.0
            randomised = (
                float(cfg.fixture_tilt_noise_rad) != 0.0
                or float(cfg.fixture_yaw_noise_rad) != 0.0
                or float(cfg.fixture_pos_noise_xy) != 0.0
                or float(cfg.fixture_pos_noise_z) != 0.0
            )
            if deterministic and randomised:
                raise ValueError(
                    "a deterministic pocket angle (fixture_tilt_rad / fixture_yaw_rad) is "
                    "set together with a per-episode randomisation (fixture_tilt_noise_rad / "
                    "fixture_yaw_noise_rad / fixture_pos_noise_xy / fixture_pos_noise_z): "
                    "one run cannot be a "
                    "deterministic probe "
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
            self._tilt_rot = insertion_math.axes_from_quat(self._fixture_quat)
            self._tilt_rot_t = self._tilt_rot.transpose(1, 2).contiguous()
        else:
            self._tilt_rot = None
            self._tilt_rot_t = None

        self._pocket_rot_identity = (
            torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        )

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
        self._yaw_joint_idx = yaw_idx

        self._start_ik = None
        self._jacobi_idx = None
        self._start_solve_residual_mm = 0.0
        self._start_solve_iters = 0
        self._start_solve_worst_mm = 0.0
        self._start_solve_unconverged = 0
        self._start_solve_flags_dropped = 0
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

        self._start_height = torch.full(
            (self.num_envs,), self._start_tip_height, device=self.device
        )
        self._start_lat_off = torch.zeros((self.num_envs, 2), device=self.device)
        self._friction_applied = torch.full(
            (self.num_envs,), float(insertion_tasks_cfg.CONTACT_FRICTION), device=self.device
        )

        self._reset_unit = torch.zeros(
            (self.num_envs, len(autodr.TABLE_COLUMNS)), device=self.device
        )
        self._col = dict(autodr.COLUMN_INDEX)
        self._eval_table_unit = (
            None if self._eval_table is None
            else torch.tensor(
                self._eval_table.rows, dtype=self._reset_unit.dtype, device=self.device
            )
        )

        self._boundary_of = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._bounds_stamp = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
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

        self._step_dt = cfg.sim.dt * cfg.decimation

        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._prev_joint_pos = self.robot.data.default_joint_pos.clone()
        self._max_depth = torch.zeros(self.num_envs, device=self.device)
        self._max_target_lag = torch.zeros(self.num_envs, device=self.device)
        self._build_task_space_controller()
        self._prev_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self._last_action_rate = torch.zeros(self.num_envs, device=self.device)
        self._ema_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._force_smooth = torch.zeros(self.num_envs, 3, device=self.device)
        self._force_tared_raw = torch.zeros(self.num_envs, 3, device=self.device)
        self._wrench_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._wrench_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._torque_smooth = torch.zeros(self.num_envs, 3, device=self.device)
        self._torque_tared_raw = torch.zeros(self.num_envs, 3, device=self.device)
        self._max_torque_norm = torch.zeros(self.num_envs, device=self.device)
        self._wrench_valid_steps = 0
        self._wrench_steps = 0
        self._wrench_step = -1
        self._torque_ref_offset_parent = torch.tensor(
            [float(v) for v in self.cfg.torque_ref_offset_parent_m], device=self.device
        ).reshape(1, 3).repeat(self.num_envs, 1)
        self._max_force_norm = torch.zeros(self.num_envs, device=self.device)
        self._max_interpen_m = torch.zeros(self.num_envs, device=self.device)
        self._below_fixture = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._interpen_unclean_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._max_force_raw_norm = torch.zeros(self.num_envs, device=self.device)
        self._episode_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._episode_log = None
        self._applied_written = False
        self._max_action_abs = torch.zeros(self.num_envs, device=self.device)
        self._action_sat_sum = torch.zeros(self.num_envs, device=self.device)
        self._action_steps = torch.zeros(self.num_envs, device=self.device)
        self._max_stage1_lat_y = torch.zeros(self.num_envs, device=self.device)

        self._last_depth = torch.zeros(self.num_envs, device=self.device)
        self._last_sdf_dist = torch.zeros(self.num_envs, device=self.device)
        self._last_interpen_max = torch.zeros(self.num_envs, device=self.device)
        self._last_engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_in_region = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_force_abort = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_steps_remaining = torch.zeros(self.num_envs, device=self.device)
        self._last_depth_progress = torch.zeros(self.num_envs, device=self.device)
        self._last_tilt_phi = torch.zeros(self.num_envs, device=self.device)
        self._tilt_phi_prev = torch.zeros(self.num_envs, device=self.device)
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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

        self._ep_count = 0
        self._ep_success_count = 0
        self._ep_depth_sum = 0.0
        self._episodes_to_threshold: int | None = None
        window_floor = 2000
        window = max(window_floor, int(
            self.num_envs * self.cfg.metrics_window_iterations * 16 / self.max_episode_length
        ))
        self._recent_successes: collections.deque = collections.deque(maxlen=window)
        self._recent_depths: collections.deque = collections.deque(maxlen=window)
        self._recent_success_steps: collections.deque = collections.deque(maxlen=window)
        self._recent_tilt_deg: collections.deque = collections.deque(maxlen=window)
        self._tilt_bin_edges_deg = (2.0, 4.0, 6.0, 8.0, 10.0)
        self._recent_start_height_m: collections.deque = collections.deque(maxlen=window)
        self._start_height_bin_edges_m = (
            -0.020, -0.010, 0.0, 0.010, 0.020, 0.070, 0.120, 0.170, 0.220, 0.270,
        )
        self._recent_lateral_mm: collections.deque = collections.deque(maxlen=window)
        self._lateral_bin_edges_mm = (6.0, 12.0, 18.0, 24.0, 30.0)
        self._recent_yaw_deg: collections.deque = collections.deque(maxlen=window)
        self._yaw_bin_edges_deg = (1.0, 2.0, 3.0, 4.0, 5.0)
        self._recent_target_lag: collections.deque = collections.deque(maxlen=window)
        self._recent_force_norm: collections.deque = collections.deque(maxlen=window)
        self._recent_torque_norm: collections.deque = collections.deque(maxlen=window)
        self._recent_interpen_max: collections.deque = collections.deque(maxlen=window)
        self._recent_below_fixture: collections.deque = collections.deque(maxlen=window)
        self._recent_action_abs_max: collections.deque = collections.deque(maxlen=window)
        self._recent_action_sat_frac: collections.deque = collections.deque(maxlen=window)
        self._recent_aborts: collections.deque = collections.deque(maxlen=window)
        self._recent_stage1_lat_y: collections.deque = collections.deque(maxlen=window)
        self._fresh_successes: collections.deque = collections.deque(maxlen=window)
        self._recent_boundary_successes: collections.deque = collections.deque(maxlen=window)
        self._recent_friction: collections.deque = collections.deque(maxlen=window)
        self._fresh_min = window_floor
        self._friction_write_ms: collections.deque = collections.deque(maxlen=1000)
        self._episode_sums: dict = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in insertion_math.REWARD_TERMS
        }
        self._recent_reward_terms: dict = {
            name: collections.deque(maxlen=window) for name in insertion_math.REWARD_TERMS
        }
        self._ep_success_depth_violations = 0
        self._recent_success_depth_violations: collections.deque = collections.deque(maxlen=window)
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
            self._block_usd_path = None

        self.cfg.robot_cfg.spawn.usd_path = insertion_tasks_cfg.resolve_tool_robot_usd_path()
        self._robot_usd_path = self.cfg.robot_cfg.spawn.usd_path
        print(f"[insertion] robot asset: {self._robot_usd_path}")

        if self._control_mode == "osc":
            for _act in self.cfg.robot_cfg.actuators.values():
                _act.stiffness = 0.0
                _act.damping = 0.0
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, self.cfg.floor_z),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["fixture"] = self._fixture
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._last_action_rate = torch.linalg.norm(self.actions - self._prev_actions, dim=-1)
        self._prev_actions = self.actions.clone()
        abs_actions = self.actions.abs()
        self._max_action_abs = torch.maximum(
            self._max_action_abs, abs_actions.amax(dim=-1)
        )
        self._action_sat_sum += (abs_actions > 1.0).float().mean(dim=-1)
        self._action_steps += 1.0
        clamped = actions.clamp(-1.0, 1.0)
        self._ema_actions = insertion_math.ema_update(
            self._ema_actions, clamped, float(self.cfg.action_ema_factor)
        )
        if self._control_mode == "osc":
            self._osc_delta[:, 0:3] = self._ema_actions[:, 0:3] * float(self.cfg.osc_pos_step_limit_m)
            self._osc_delta[:, 3:6] = self._ema_actions[:, 3:6] * float(self.cfg.osc_rot_step_limit_rad)
            return
        self._joint_targets = self._joint_targets + self.cfg.action_scale * self._ema_actions
        limits = self.robot.data.soft_joint_pos_limits
        self._joint_targets = torch.clamp(self._joint_targets, limits[..., 0], limits[..., 1])
        lag = (self._joint_targets - self.robot.data.joint_pos).abs().amax(dim=-1)
        self._max_target_lag = torch.maximum(self._max_target_lag, lag)

    def _apply_action(self) -> None:
        if self._control_mode == "osc":
            self._apply_osc()
            return
        self.robot.set_joint_position_target(self._joint_targets)

    def _build_task_space_controller(self) -> None:
        """Isaac Lab's OperationalSpaceController, configured per Decision (2)."""
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
        """Jacobian of the controlled body in the ROOT frame, the joint-space mass matrix and the gravity-compensation torques, straight from PhysX."""
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
        """One physics step of the task-space controller."""
        pos_w = self.robot.data.body_pos_w[:, self._osc_body_idx]
        quat_w = self.robot.data.body_quat_w[:, self._osc_body_idx]
        tgt_pos_w, tgt_quat_w = insertion_math.apply_pose_delta(
            pos_w, quat_w, self._osc_delta[:, 0:3], self._osc_delta[:, 3:6]
        )
        tgt_quat_w = insertion_math.clamp_tilt_to_cone(
            tgt_quat_w, float(self.cfg.osc_tilt_clamp_rad), self._fixture_quat
        )
        seat_quat_w = insertion_math.quat_mul(self._fixture_quat, self._seat_quat_local)
        tgt_quat_w = insertion_math.clamp_yaw_to_band(
            tgt_quat_w, float(self.cfg.osc_yaw_clamp_rad), seat_quat_w
        )
        centre_w = self._entrance_pos + self.scene.env_origins
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
        """Per-env peg-tip position (env-local), insertion depth, pocket gate, out-of-arena flag, tool-axis alignment and yaw."""
        origins = self.scene.env_origins
        peg_pos_w = self.robot.data.body_pos_w[:, self._peg_body_idx]
        peg_quat_w = self.robot.data.body_quat_w[:, self._peg_body_idx]
        peg_axes = insertion_math.axes_from_quat(peg_quat_w)
        tip_w = peg_pos_w + torch.matmul(peg_axes, self._tip_offset_local)
        tip_local = tip_w - origins

        if self._tilt_rot is None:
            tip_rel = tip_local - self._entrance_pos
            depth = -tip_rel[:, 2]
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
        corners_inside = corners.abs().amax(dim=(1, 2)) < half_pocket

        alignment = -axes[:, 2, 2]
        gate = (
            corners_inside
            & (depth <= self.cfg.fixed_asset.depth + 0.001)
            & (alignment >= self.cfg.gate_min_alignment)
        )
        return tip_local, depth, gate, below_plate, alignment, phi, cos4phi, sin4phi, tip_rel

    def _get_observations(self) -> dict:
        """The REAL task's observation (D-107), minus its force block."""
        self._obs_calls += 1

        joint_pos = self.robot.data.joint_pos
        joint_vel_fd = (joint_pos - self._prev_joint_pos) / self._step_dt
        self._prev_joint_pos = joint_pos.clone()

        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        belief_rot, pocket_quat = insertion_math.believed_pocket_frame(
            self._fixture_quat, self._pocket_obs_yaw
        )

        if self._peg_body_idx is not None:
            pocket_rot = self._tilt_rot if self._tilt_rot is not None else self._pocket_rot_identity
            tip_rel, _depth, x_axis = insertion_math.part_tip_pose(
                self.robot.data.body_pos_w[:, self._peg_body_idx],
                self.robot.data.body_quat_w[:, self._peg_body_idx],
                self._tip_offset_local + self._grasp_obs_off,
                self.scene.env_origins,
                self._entrance_pos,
                belief_rot,
            )
            yaw_cs = insertion_math.yaw_cos_sin(x_axis)
            self._tilt_phi_prev = insertion_math.tilt_potential(
                insertion_math.tilt_cos_theta(
                    self.robot.data.body_quat_w[:, self._peg_body_idx], pocket_rot
                ),
                float(self.cfg.w_tilt),
                float(self.cfg.kernel_a_tilt),
            )
        else:
            tip_rel = torch.zeros(self.num_envs, 3, device=self.device)
            yaw_cs = torch.zeros(self.num_envs, 2, device=self.device)
            yaw_cs[:, 0] = 1.0

        # The force block (D-114). Read RAW at the measured link and smoothed
        if self._force_body_idx is not None:
            wrench_raw = self.robot.data.body_incoming_joint_wrench_b[:, self._force_body_idx, 0:6]
            force_raw = wrench_raw[:, 0:3]
            torque_raw = wrench_raw[:, 3:6]
            if self._force_parent_idx is not None:
                parent_quat_w = self.robot.data.body_quat_w[:, self._force_parent_idx]
                force_raw = insertion_math.gravity_tare(
                    force_raw,
                    parent_quat_w,
                    self._hold_force_w,
                    True,
                )
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
                self._wrench_valid_steps = self._wrench_valid_steps + keep.sum()
                self._wrench_steps += int(keep.numel())
        force = self._force_smooth
        torque = self._torque_smooth
        self._max_force_norm = torch.maximum(
            self._max_force_norm, insertion_math.force_magnitude(force)
        )
        self._max_torque_norm = torch.maximum(
            self._max_torque_norm, insertion_math.force_magnitude(torque)
        )
        self._max_force_raw_norm = torch.maximum(
            self._max_force_raw_norm, insertion_math.force_magnitude(self._force_tared_raw)
        )

        obs = self._assemble_obs(joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force, torque)
        tip_rel_true, yaw_cs_true, pocket_quat_true = self._true_pocket_blocks()
        critic = self._assemble_obs(
            joint_pos, joint_vel_fd, tip_rel_true, ee_quat_w, yaw_cs_true, pocket_quat_true, force, torque
        )

        if self._obs_calls in self._report_at_obs_calls:
            self._print_startup_report(joint_vel_fd)

        if self._eval_table is not None and not self._applied_written:
            self._write_applied_csv()

        return {"policy": obs, "critic": critic}

    def _true_pocket_blocks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``tip_rel``, ``yaw_cos_sin`` and ``pocket_quat`` of the TRUE state, for the critic group (D-195)."""
        true_rot, pocket_quat = insertion_math.believed_pocket_frame(
            self._fixture_quat, torch.zeros(self.num_envs, device=self.device)
        )
        if self._peg_body_idx is None:
            tip_rel = torch.zeros(self.num_envs, 3, device=self.device)
            yaw_cs = torch.zeros(self.num_envs, 2, device=self.device)
            yaw_cs[:, 0] = 1.0
            return tip_rel, yaw_cs, pocket_quat
        tip_rel, _depth, x_axis = insertion_math.part_tip_pose(
            self.robot.data.body_pos_w[:, self._peg_body_idx],
            self.robot.data.body_quat_w[:, self._peg_body_idx],
            self._tip_offset_local,
            self.scene.env_origins,
            self._entrance_pos,
            true_rot,
        )
        return tip_rel, insertion_math.yaw_cos_sin(x_axis), pocket_quat

    def _assemble_obs(self, joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force, torque):
        """One observation group from its blocks, in the layout of the mode."""
        if self._obs_mode == "wrench":
            obs = torch.cat(
                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force, torque),
                dim=-1,
            )
        elif self._obs_mode == "force_prevact":
            prev_action = self._ema_actions
            obs = torch.cat(
                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force, prev_action),
                dim=-1,
            )
        else:
            obs = torch.cat(
                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force), dim=-1
            )
        return obs

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The REAL termination (D-113): exactly two exits, and the per-step predicates the reward reads."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self._peg_body_idx is None:
            terminated = torch.zeros_like(time_out)
            self._last_depth = torch.zeros(self.num_envs, device=self.device)
            self._last_terminated = terminated
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
        self._last_tilt_phi = insertion_math.tilt_potential(
            insertion_math.tilt_cos_theta(body_quat, pocket_rot),
            float(self.cfg.w_tilt),
            float(self.cfg.kernel_a_tilt),
        )
        cross_section = insertion_math.in_pocket_cross_section(
            tip_rel,
            insertion_tasks_cfg.POCKET_WALL_X,
            insertion_tasks_cfg.POCKET_WALL_Y,
        )
        in_pocket = insertion_math.in_pocket_volume(
            cross_section, depth, -insertion_tasks_cfg.POCKET_ASSET_BOTTOM_Z
        )
        self._below_fixture = self._below_fixture | (cross_section & ~in_pocket)
        prev_max_depth = self._max_depth
        self._max_depth = torch.maximum(
            prev_max_depth, torch.where(in_pocket, depth, torch.zeros_like(depth))
        )
        self._last_depth_progress = self._max_depth - prev_max_depth

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
            self._last_tilt_phi = torch.zeros(self.num_envs, device=self.device)
            self._last_success = terminated
            return terminated, time_out

        part_pos = body_pos_w - self.scene.env_origins

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

        sapu_pos, sapu_quat = insertion_sdf.goal_relative_transform(
            part_pos, body_quat, self._entrance_pos, self._fixture_quat
        )
        interpen = self._sapu_query.interpen_distances(sapu_pos, sapu_quat)
        interpen_max = insertion_math.max_interpen_dist(interpen)
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
        self._last_success = in_region
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """The REAL reward (D-109 / D-114), one call site."""
        if self._peg_body_idx is None or not self._rl_terms_enabled:
            return torch.zeros(self.num_envs, device=self.device)
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
        for i, name in enumerate(insertion_math.REWARD_TERMS):
            self._episode_sums[name] += terms[i]
        return torch.sum(terms, dim=0)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        ready = hasattr(self, "_joint_targets")

        if ready:
            self._log_finished_episodes(env_ids)

        idx = (
            env_ids if torch.is_tensor(env_ids)
            else torch.as_tensor(list(env_ids), device=self.device)
        )

        if ready:
            self._draw_reset_conditions(idx)

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        if ready:
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
            try:
                self.robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=idx)
            except (AttributeError, TypeError) as exc:
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
            self._prev_actions[env_ids] = 0.0
            self._last_action_rate[env_ids] = 0.0
            self._ema_actions[env_ids] = 0.0
            self._force_smooth[env_ids] = 0.0
            self._force_tared_raw[env_ids] = 0.0
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
            self._grasp_obs_off[idx, 0] = (
                2.0 * self._reset_unit[idx][:, self._col["grasp_obs_x"]] - 1.0
            ) * float(self.cfg.grasp_obs_offset_x_m)
            self._pocket_obs_yaw[idx] = (
                2.0 * self._reset_unit[idx][:, self._col["pocket_obs_yaw"]] - 1.0
            ) * float(self.cfg.pocket_obs_yaw_rad)

        if ready and self._fixture_pose_reach:
            n = int(idx.numel())
            u = self._reset_unit[idx]
            entrance = self._entrance_base.unsqueeze(0).repeat(n, 1)
            u_xy = torch.stack(
                (u[:, self._col["fixture_noise_x"]], u[:, self._col["fixture_noise_y"]]),
                dim=-1,
            )
            entrance[:, :2] += (u_xy * 2.0 - 1.0) * self.cfg.fixture_pos_noise_xy
            entrance[:, 2] += (
                u[:, self._col["fixture_noise_z"]] * 2.0 - 1.0
            ) * self.cfg.fixture_pos_noise_z
            _b = self._live_bounds()
            _y_lo, _y_hi = _b["yaw"]
            yaw = autodr.map_unit_to_bounds(
                u[:, self._col["yaw"]], float(_y_lo), float(_y_hi)
            )
            self._entrance_pos[idx] = entrance
            self._fixture_yaw[idx] = yaw
            quat_yaw = torch.zeros(n, 4, device=self.device)
            quat_yaw[:, 0] = torch.cos(yaw * 0.5)
            quat_yaw[:, 3] = torch.sin(yaw * 0.5)

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
            quat = torch.where(quat[:, :1] < 0.0, -quat, quat)
            self._fixture_quat[idx] = quat
            if self._tilt_rot is not None:
                rot = insertion_math.axes_from_quat(quat)
                self._tilt_rot[idx] = rot
                self._tilt_rot_t[idx] = rot.transpose(1, 2)

            pose_w = torch.cat((entrance + self.scene.env_origins[idx], quat), dim=-1)
            self._fixture.write_root_pose_to_sim(pose_w, env_ids=idx)
            self._fixture.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=idx)

        if ready:
            # Map the start conditions into their per-env buffers FIRST;
            self._map_start_conditions(idx)
            self._solve_start_pose(env_ids)

        if ready and self._friction_reach:
            self._apply_friction(idx)

    def _live_bounds(self) -> dict:
        """The band THIS reset draws from, one entry per ``autodr.DR_DIMS`` name, in the quantity's own unit: ``{name: (lo, hi)}``."""
        _yaw = float(self.cfg.fixture_yaw_noise_rad)
        _tilt = float(self.cfg.fixture_tilt_noise_rad)
        _hi = self.cfg.start_tip_above_entrance
        _hi = self._start_tip_height if _hi is None else float(_hi)
        _lo = self.cfg.start_tip_above_entrance_low
        _lo = _hi if _lo is None or float(_lo) >= _hi else float(_lo)
        _static = {
            "lat_r": (0.0, self._start_lat_half),
            "yaw": (-_yaw, _yaw),
            "tilt": (0.0, _tilt),
            "start_height": (_lo, _hi),
            "friction": (
                float(insertion_tasks_cfg.CONTACT_FRICTION),
                float(insertion_tasks_cfg.CONTACT_FRICTION),
            ),
        }
        if self._dr is not None:
            # WITH A PROVIDER its live bounds win for every quantity it
            _static.update(self._dr.bounds())
        return _static

    def _draw_reset_conditions(self, idx: torch.Tensor) -> None:
        """ONE uniform row per resetting env, before any of it is applied."""
        if idx.numel() == 0:
            return
        n = int(idx.numel())
        if self._eval_table_unit is not None:
            self._reset_unit[idx] = autodr.table_rows(self._eval_table_unit, idx)
        else:
            self._reset_unit[idx] = torch.rand(
                n, len(autodr.TABLE_COLUMNS), device=self.device
            )
        if self._dr is None:
            return
        _rb = torch.rand(n, 2, device=self.device)
        _is_b, _which = self._dr.boundary_assignment(_rb[:, 0], _rb[:, 1])
        self._boundary_of[idx] = torch.where(
            _is_b, _which, torch.full_like(_which, -1)
        )
        _rows = idx[_is_b]
        if _rows.numel() > 0:
            _sel = _which[_is_b]
            autodr.apply_boundary(
                self._reset_unit,
                _rows,
                self._boundary_cols[_sel],
                self._boundary_sides[_sel],
            )
        self._bounds_stamp[idx] = self._dr.bounds_version

    def _map_start_conditions(self, idx: torch.Tensor) -> None:
        """Map this reset's draw onto the START HEIGHT and LATERAL OFFSET."""
        if idx.numel() == 0:
            return
        u = self._reset_unit[idx]
        _b = self._live_bounds()
        _h_lo, _h_hi = _b["start_height"]
        self._start_height[idx] = autodr.map_unit_to_bounds(
            u[:, self._col["start_height"]], float(_h_lo), float(_h_hi)
        )
        self._start_lat_off[idx] = insertion_math.disk_offset(
            u[:, self._col["lat_r"]], u[:, self._col["lat_phi"]], float(_b["lat_r"][1])
        )

    def _apply_friction(self, idx: torch.Tensor) -> None:
        """ONE contact friction per env, redrawn at every reset (plan sec."""
        if idx.numel() == 0:
            return
        # The LIVE band, not the reach: the reach decides whether this method
        lo, hi = self._live_bounds()["friction"]
        mu = autodr.map_unit_to_bounds(
            self._reset_unit[idx][:, self._col["friction"]], float(lo), float(hi)
        )
        self._friction_applied[idx] = mu

        if mu.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx_cpu = idx.to(device="cpu", dtype=torch.long)
        _col = mu.to(device="cpu", dtype=self._friction_buf_dtype).unsqueeze(1)
        for _asset, _buf in self._friction_bufs:
            _buf[idx_cpu, :, 0] = _col  # static
            _buf[idx_cpu, :, 1] = _col  # dynamic
            _asset.root_physx_view.set_material_properties(_buf, idx_cpu)
        self._friction_write_ms.append((time.perf_counter() - t0) * 1000.0)

    def _solve_start_pose(self, env_ids: Sequence[int]) -> None:
        """Teleport the resetting envs so the leading tool point starts at ``cfg.start_tip_above_entrance`` on this episode's pocket axis."""
        if self._start_ik is None:
            return
        idx = torch.as_tensor(list(env_ids), device=self.device) if not torch.is_tensor(env_ids) else env_ids
        if idx.numel() == 0:
            return
        tol = float(self.cfg.start_pose_solve_tol_m)
        n_joints = len(self.robot.joint_names)
        pocket_quat = insertion_math.canonicalize_quat(self._fixture_quat)
        err_m = 0.0
        used = 0
        with torch.no_grad():
            for step in range(int(self.cfg.start_pose_solve_steps)):
                *_, tip_rel = self._peg_geometry()
                d_pocket = torch.zeros(self.num_envs, 6, device=self.device)
                d_pocket[:, 0] = self._start_lat_off[:, 0] - tip_rel[:, 0]
                d_pocket[:, 1] = self._start_lat_off[:, 1] - tip_rel[:, 1]
                d_pocket[:, 2] = self._start_height - tip_rel[:, 2]
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
                self._joint_targets[idx] = sub
                self._prev_joint_pos[idx] = sub

            # inside-start episodes -- read it against the logged start height.
            *_, tip_rel = self._peg_geometry()
            self._max_depth[idx] = (-tip_rel[idx, 2]).clamp(min=0.0)

            _goal = torch.zeros(self.num_envs, 3, device=self.device)
            _goal[:, 0] = self._start_lat_off[:, 0]
            _goal[:, 1] = self._start_lat_off[:, 1]
            _goal[:, 2] = self._start_height
            _bad = torch.linalg.norm(_goal[idx] - tip_rel[idx, 0:3], dim=-1) >= tol
            _n_bad = int(_bad.sum().item())
            if _n_bad:
                _rows = idx[_bad]
                self._boundary_of[_rows] = -1
                self._bounds_stamp[_rows] = -1
                self._start_solve_flags_dropped += _n_bad

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
        """Record success/depth for the envs about to reset, and surface them to rsl_rl (``self.extras[\"log\"]``) and to disk, per the project rule that results are read from disk, not from prose."""
        if len(env_ids) == 0 or self._peg_body_idx is None or self._obs_calls == 0:
            return
        idx = torch.as_tensor(list(env_ids), device=self.device) if not torch.is_tensor(env_ids) else env_ids
        successes = self._last_success[idx]
        depths = self._max_depth[idx]
        aborts = self._last_force_abort[idx]
        n = int(idx.numel())

        if self._dr is not None:
            _b_of = self._boundary_of[idx]
            _is_b = _b_of >= 0
            _b_keys = _b_of[_is_b].tolist()
            _b_flags = successes[_is_b].tolist()
            for _k, _f in zip(_b_keys, _b_flags):
                self._dr.record(self._boundary_keys[_k], bool(_f))
            self._recent_boundary_successes.extend(successes[_is_b].tolist())
            _fresh = (~_is_b) & (self._bounds_stamp[idx] == self._dr.bounds_version)
            self._fresh_successes.extend(successes[_fresh].tolist())
            _events = self._dr.update()
            for _ev in _events:
                print(_ev.line())
            if any(_ev.moved for _ev in _events):
                self._fresh_successes.clear()
        self._recent_friction.extend(self._friction_applied[idx].tolist())

        self._ep_count += n
        self._ep_success_count += int(successes.sum().item())
        self._ep_depth_sum += float(depths.sum().item())
        self._recent_successes.extend(successes.tolist())
        self._recent_depths.extend(depths.tolist())
        violations = successes & (depths <= 0.0)
        self._ep_success_depth_violations += int(violations.sum().item())
        self._recent_success_depth_violations.extend(violations.tolist())
        self._recent_success_steps.extend(self.episode_length_buf[idx][successes].tolist())
        fq_x, fq_y = self._fixture_quat[idx, 1], self._fixture_quat[idx, 2]
        r22 = 1.0 - 2.0 * (fq_x * fq_x + fq_y * fq_y)
        tilt_deg = torch.rad2deg(torch.acos(r22.clamp(-1.0, 1.0)))
        self._recent_tilt_deg.extend(tilt_deg.tolist())
        self._recent_start_height_m.extend(self._start_height[idx].tolist())
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
        self._append_episode_rows(idx, successes, aborts)
        self._recent_action_abs_max.extend(self._max_action_abs[idx].tolist())
        self._recent_action_sat_frac.extend(
            (self._action_sat_sum[idx] / self._action_steps[idx].clamp_min(1.0)).tolist()
        )
        self._recent_aborts.extend(aborts.tolist())
        self._recent_stage1_lat_y.extend(self._max_stage1_lat_y[idx].tolist())
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
            # the success definition).
            "force_abort_rate": (
                sum(self._recent_aborts) / len(self._recent_aborts)
                if self._recent_aborts
                else 0.0
            ),
            "stage1_lat_y_max_mm": (
                max(self._recent_stage1_lat_y) * 1000.0
                if self._recent_stage1_lat_y
                else 0.0
            ),
            "success_depth_invariant_violations": float(
                sum(self._recent_success_depth_violations)
            ),
            "below_fixture_rate": (
                sum(self._recent_below_fixture) / len(self._recent_below_fixture)
                if self._recent_below_fixture
                else 0.0
            ),
            "start_height_mean_mm": float(self._start_height[idx].mean()) * 1000.0,
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
            "dr/friction_write_ms_mean": (
                sum(self._friction_write_ms) / len(self._friction_write_ms)
                if self._friction_write_ms
                else 0.0
            ),
            "dr/friction_write_ms_max": (
                max(self._friction_write_ms) if self._friction_write_ms else 0.0
            ),
        }
        for name, value in term_means.items():
            self.extras["log"]["Episode_Reward/" + name] = value
        self.extras["log"]["Episode_Reward/total"] = float(sum(term_means.values()))

        if self._rl_terms_enabled and self._recent_interpen_max:
            self.extras["log"]["interpen_max_mean_mm"] = (
                sum(self._recent_interpen_max) / len(self._recent_interpen_max) * 1000.0
            )
            self.extras["log"]["interpen_max_max_mm"] = (
                max(self._recent_interpen_max) * 1000.0
            )

        if self._recent_force_norm:
            _fv = sorted(self._recent_force_norm)
            self.extras["log"]["force_max_mean_n"] = sum(_fv) / len(_fv)
            self.extras["log"]["force_max_p95_n"] = _fv[
                min(len(_fv) - 1, int(math.ceil(0.95 * len(_fv))) - 1)
            ]
            self.extras["log"]["force_max_max_n"] = _fv[-1]

        # and which read 0.0 there.
        if self._dr is not None:
            self.extras["log"].update(self._dr.scalars())
            _fresh_n = len(self._fresh_successes)
            self.extras["log"]["dr/train_success_regular_fresh"] = (
                sum(self._fresh_successes) / _fresh_n
                if _fresh_n >= self._fresh_min
                else -1.0
            )
            self.extras["log"]["dr/start_solve_flags_dropped"] = float(
                self._start_solve_flags_dropped
            )
            self.extras["log"]["dr/fresh_n"] = float(_fresh_n)
            self.extras["log"]["dr/fresh_min"] = float(self._fresh_min)
            self.extras["log"]["dr/success_rate_boundary"] = (
                sum(self._recent_boundary_successes)
                / len(self._recent_boundary_successes)
                if self._recent_boundary_successes
                else 0.0
            )
            self.extras["log"]["dr/friction_mean"] = (
                sum(self._recent_friction) / len(self._recent_friction)
                if self._recent_friction
                else 0.0
            )

        self.extras["log"]["max_depth_max_mm"] = (
            max(self._recent_depths) * 1000.0 if self._recent_depths else 0.0
        )

        if self._ep_count % self._metrics_dump_every < n:
            self._write_metrics(cumulative_rate, recent_rate, recent_depth)

    def _success_by_tilt_bin(self) -> list[dict]:
        """Trailing-window success rate resolved by pocket tilt magnitude."""
        return self._rate_by_bin(
            self._recent_successes, self._recent_tilt_deg, self._tilt_bin_edges_deg,
            "tilt_deg_from", "tilt_deg_to", "success_rate", first_lower=0.0, scale=1.0,
        )

    def _success_by_start_height_bin(self) -> list[dict]:
        """Trailing-window success rate resolved by the episode's START HEIGHT (plan step C, SBC)."""
        return self._rate_by_bin(
            self._recent_successes, self._recent_start_height_m, self._start_height_bin_edges_m,
            "start_mm_from", "start_mm_to", "success_rate", first_lower=None, scale=1000.0,
        )

    def _success_by_lateral_bin(self) -> list[dict]:
        """Trailing-window success rate resolved by the episode's LATERAL START OFFSET (D-170)."""
        return self._rate_by_bin(
            self._recent_successes, self._recent_lateral_mm, self._lateral_bin_edges_mm,
            "lateral_mm_from", "lateral_mm_to", "success_rate", first_lower=0.0, scale=1.0,
        )

    def _success_by_yaw_bin(self) -> list[dict]:
        """Trailing-window success rate resolved by pocket YAW magnitude (D-038)."""
        return self._rate_by_bin(
            self._recent_successes, self._recent_yaw_deg, self._yaw_bin_edges_deg,
            "yaw_deg_from", "yaw_deg_to", "success_rate", first_lower=0.0, scale=1.0,
        )

    def _force_abort_by_start_height_bin(self) -> list[dict]:
        """Trailing-window FORCE-ABORT rate over the same start-height bins as ``_success_by_start_height_bin`` (2026-09-03)."""
        return self._rate_by_bin(
            self._recent_aborts, self._recent_start_height_m, self._start_height_bin_edges_m,
            "start_mm_from", "start_mm_to", "force_abort_rate", first_lower=None, scale=1000.0,
        )

    def _rate_by_bin(
        self, outcomes, values, edges, from_key: str, to_key: str, rate_key: str,
        first_lower, scale: float,
    ) -> list[dict]:
        """The one binning routine behind every per-bin table."""
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

    def _force_stats(self) -> dict:
        """The per-episode peak of the smoothed force magnitude, as a spread."""
        values = sorted(self._recent_force_norm)
        if not values:
            return {"episodes": 0}

        def _pct(sorted_values: list[float], q: float) -> float:
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
        """The per-episode peak of the smoothed torque magnitude, N m, as a spread -- the D-188 twin of ``_force_stats``, same nearest-rank percentiles, plus the share of steps whose wrench was VALID (not the reset step)."""
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
        """Append one row per finished episode to ``<run folder>/episodes.csv``."""
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
        iteration = (
            (int(self.common_step_counter) - 1) // int(steps_per_it) if steps_per_it else -1
        )
        n = int(idx.numel())
        if self._dr is not None:
            bounds = self._bounds_stamp[idx].to(torch.float64)
            phase = str(getattr(self._dr, "phase", "fixed"))
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

    def _write_applied_csv(self) -> None:
        """Write ``<run folder>/applied.csv``: ALL envs, one row each, once."""
        if int(self.common_step_counter) < 1:
            return
        log_dir = getattr(self.cfg, "log_dir", None)
        if not log_dir:
            raise RuntimeError(
                "dr_mode='table' without cfg.log_dir: applied.csv (eval plan G3) "
                "has no folder to go to. play.py sets log_dir; set it here too."
            )
        n = int(self.num_envs)
        _fric = [
            _asset.root_physx_view.get_material_properties()[..., 0:2]
            .reshape(n, -1).to(device=self.device, dtype=torch.float64)
            for _asset in (self.robot, self._fixture)
        ]
        _pocket_d = self._entrance_pos - self._entrance_base.unsqueeze(0)
        table = torch.stack(
            [
                torch.arange(n, device=self.device, dtype=torch.float64),
                self._start_lat_off[:, 0].to(torch.float64),
                self._start_lat_off[:, 1].to(torch.float64),
                self._start_height.to(torch.float64),
                self._fixture_yaw.to(torch.float64),
                self._fixture_quat[:, 0].to(torch.float64),
                self._fixture_quat[:, 1].to(torch.float64),
                self._fixture_quat[:, 2].to(torch.float64),
                self._fixture_quat[:, 3].to(torch.float64),
                _pocket_d[:, 0].to(torch.float64),
                _pocket_d[:, 1].to(torch.float64),
                _pocket_d[:, 2].to(torch.float64),
                self._grasp_obs_off[:, 0].to(torch.float64),
                self._pocket_obs_yaw.to(torch.float64),
                self._friction_applied.to(torch.float64),
                _fric[0].min(dim=1).values,
                _fric[0].max(dim=1).values,
                _fric[1].min(dim=1).values,
                _fric[1].max(dim=1).values,
            ],
            dim=1,
        ).cpu().tolist()
        run_dir = pathlib.Path(log_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "applied.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(APPLIED_LOG_COLUMNS)
            writer.writerows(
                [int(row[0])] + [f"{v:.9g}" for v in row[1:]] for row in table
            )
        with open(run_dir / "applied_meta.json", "w", encoding="utf-8") as mf:
            json.dump(self._applied_log_meta(), mf, indent=2)
        self._applied_written = True
        print(f"[insertion] applied conditions: {path} ({n} rows, "
              f"{len(APPLIED_LOG_COLUMNS)} columns, written at policy step "
              f"{int(self.common_step_counter)})")

    def _applied_log_meta(self) -> dict:
        """What ``applied.csv`` carries -- written once per run beside it."""
        cfg = self.cfg
        return {
            "schema": "applied-v2",
            "code_marker": CODE_MARKER,
            "table": {
                "path": str(self._eval_table.path),
                "sha256": str(self._eval_table.sha256),
                "rows": int(self._eval_table.n_rows),
                "width": str(cfg.dr_table_width),
                "allow_outside": bool(self._eval_table.allow_outside),
            },
            "bounds": {k: [float(v[0]), float(v[1])] for k, v in self._live_bounds().items()},
            "fixture_pos_noise_xy_m": float(cfg.fixture_pos_noise_xy),
            "fixture_pos_noise_z_m": float(cfg.fixture_pos_noise_z),
            "grasp_obs_offset_x_m": float(cfg.grasp_obs_offset_x_m),
            "pocket_obs_yaw_rad": float(cfg.pocket_obs_yaw_rad),
            "reset_joint_noise_scale_rad": [float(v) for v in self._reset_noise_scale.tolist()],
            "written_at_policy_step": int(self.common_step_counter),
            "columns": {
                "row_id": "env index = table row index (autodr.table_rows)",
                "lat_x_m": "commanded lateral start offset of the tip, x, POCKET frame, m",
                "lat_y_m": "same, y",
                "start_height_m": "commanded tip height above the entrance, pocket z, m",
                "yaw_rad": "commanded pocket yaw about env z, rad",
                "pocket_qw": "commanded pocket orientation quat_yaw * quat_tilt, wxyz, w >= 0; "
                             "the tensor that went to write_root_pose_to_sim",
                "pocket_qx": "see pocket_qw",
                "pocket_qy": "see pocket_qw",
                "pocket_qz": "see pocket_qw",
                "pocket_dx_m": "commanded pocket position minus its base position, x, ENV frame, m",
                "pocket_dy_m": "same, y",
                "pocket_dz_m": "same, z (D-204 (4))",
                "grasp_obs_x_m": "grasp belief error on the OBSERVATION only (D-183), m",
                "pocket_obs_yaw_rad": "pocket-yaw belief error on the OBSERVATION only, about "
                                      "the pocket's own axis (D-194), rad",
                "friction_cmd": "friction _apply_friction commanded; the one-shot D-111 value "
                                "when the provider carries no friction",
                "friction_robot_min": "PhysX read-back, robot, min over shapes and static+dynamic",
                "friction_robot_max": "same, max",
                "friction_fixture_min": "PhysX read-back, fixture, min",
                "friction_fixture_max": "same, max",
            },
            "not_recorded": "the six joint-noise columns: their scale is "
                            "reset_joint_noise_scale_rad above; the realised start pose "
                            "(IK residual); the per-step observation noise",
        }

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
        """Distribution of the per-episode WORST SAPU interpenetration over the recent window, in mm."""
        if not self._rl_terms_enabled:
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
        """The D-153 exposure, measured instead of argued."""
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
        """Mean per-episode sum of each reward term over the trailing window."""
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
            "side_clearance_mm": insertion_tasks_cfg.SIDE_CLEARANCE * 1000.0,
            "yaw_window_deg": math.degrees(insertion_tasks_cfg.YAW_WINDOW_RAD),
            "play_x_mm": insertion_tasks_cfg.PLAY_X * 1000.0,
            "play_y_mm": insertion_tasks_cfg.PLAY_Y * 1000.0,
            "reset_joint_noise_rad": self.cfg.reset_joint_noise,
            "reset_yaw_noise_rad": self.cfg.reset_yaw_noise,
            "start_tip_above_entrance_mm": (
                None if self.cfg.start_tip_above_entrance is None
                else self.cfg.start_tip_above_entrance * 1000.0
            ),
            "start_lateral_offset_m": self._start_lat_half,
            "success_by_lateral_bin": self._success_by_lateral_bin(),
            "start_tip_above_entrance_low_mm": (
                None if self.cfg.start_tip_above_entrance_low is None
                else self.cfg.start_tip_above_entrance_low * 1000.0
            ),
            "start_floor_m": (float(self.cfg.start_floor_m) if self._start_floor_on else None),
            "start_floor_steps": (int(self.cfg.start_floor_steps) if self._start_floor_on else None),
            "autodr_stall_buffers": int(self.cfg.autodr_stall_buffers),
            "success_by_start_height_bin": self._success_by_start_height_bin(),
            "force_abort_by_start_height_bin": self._force_abort_by_start_height_bin(),
            "home_tip_above_entrance_mm":
                insertion_tasks_cfg.WORKCELL_HOME_TIP_ABOVE_ENTRANCE * 1000.0,
            "start_pose_solve_last_residual_mm": self._start_solve_residual_mm,
            "start_pose_solve_worst_residual_mm": self._start_solve_worst_mm,
            "start_pose_solve_last_iterations": self._start_solve_iters,
            "start_pose_solve_tol_mm": self.cfg.start_pose_solve_tol_m * 1000.0,
            "start_pose_solve_unconverged_resets": self._start_solve_unconverged,
            "start_pose_solve_flags_dropped_envs": self._start_solve_flags_dropped,
            "dr_mode": str(self.cfg.dr_mode),
            "dr_state": (None if self._dr is None else self._dr.as_dict()),
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
            "friction_applied": (
                None if not self._recent_friction else {
                    "mean": sum(self._recent_friction) / len(self._recent_friction),
                    "min": min(self._recent_friction),
                    "max": max(self._recent_friction),
                    "episodes_in_window": len(self._recent_friction),
                }
            ),
            "friction_write_ms": (
                None if not self._friction_write_ms else {
                    "mean": sum(self._friction_write_ms) / len(self._friction_write_ms),
                    "max": max(self._friction_write_ms),
                    "samples_in_window": len(self._friction_write_ms),
                    "window": self._friction_write_ms.maxlen,
                }
            ),
            "fixture_pos_noise_xy_m": self.cfg.fixture_pos_noise_xy,
            "fixture_pos_noise_z_m": self.cfg.fixture_pos_noise_z,
            "fixture_yaw_noise_rad": self.cfg.fixture_yaw_noise_rad,
            "obs_noise_pocket_pos_std_m": self.cfg.obs_noise_pocket_pos_std_m,
            "force_obs_noise_std_n": self.cfg.force_obs_noise_std_n,
            "obs_wrench_mode": self._obs_mode,
            "obs_version": insertion_math.obs_version(self._obs_mode),
            "torque_obs_noise_std_nm": self.cfg.torque_obs_noise_std_nm,
            "grasp_obs_offset_x_m": self.cfg.grasp_obs_offset_x_m,
            "pocket_obs_yaw_rad": self.cfg.pocket_obs_yaw_rad,
            "fixture_tilt_rad": self.cfg.fixture_tilt_rad,
            "fixture_yaw_rad": self.cfg.fixture_yaw_rad,
            "fixture_tilt_noise_rad": self.cfg.fixture_tilt_noise_rad,
            "success_by_tilt_bin": self._success_by_tilt_bin(),
            "success_by_yaw_bin": self._success_by_yaw_bin(),
            "action_scale_rad": self.cfg.action_scale,
            "joint_target_lag_rad": self._target_lag_stats(),
            "force_norm_n": self._force_stats(),
            "torque_norm_nm": self._torque_stats(),
            "interpen_max_mm": self._interpen_stats(),
            "stage1_lateral_y_mm": self._stage1_lateral_stats(),
            "reward_terms_recent_mean": self._reward_terms_stats(),
            "success_depth_invariant_violations": {
                "recent": int(sum(self._recent_success_depth_violations)),
                "cumulative": self._ep_success_depth_violations,
                "window_episodes": len(self._recent_success_depth_violations),
            },
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
            # (rl_terms_enabled = False) the abort is off and this reads 0.0.
            "force_abort_rate": (
                sum(self._recent_aborts) / len(self._recent_aborts)
                if self._recent_aborts
                else 0.0
            ),
            "mean_success_episode_steps": (
                sum(self._recent_success_steps) / len(self._recent_success_steps)
                if self._recent_success_steps
                else None
            ),
            "episode_length_s": self.cfg.episode_length_s,
            "num_envs": self.num_envs,
            "code_marker": CODE_MARKER,
            "rl_placeholders": {
                name: getattr(self.cfg, name, None) for name in RL_PLACEHOLDERS
            },
            "control_mode": self._control_mode,
            "policy_rate_hz": 1.0 / (self.cfg.sim.dt * self.cfg.decimation),
            "curriculum_enabled": bool(self.cfg.curriculum_enabled),
            "rung_step_sizes": self.cfg.rung_step_sizes,
            "spawn_workcell_block": bool(self.cfg.spawn_workcell_block),
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
        """Compare the authored asset's *measured* geometry against the cfg constants."""
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
            measured = data.get("after", {}).get("measured") or data.get("measured", {})
            verdict = data.get("verdict")
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
        """The same check for the fixture -- REWRITTEN 2026-08-24 for the CAD asset."""
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
        want = sorted(geo.POCKET_ASSET_BBOX_M)
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
        """Where the leading tool point is SUPPOSED to stand at the start of an episode, above the stage-2 opening plane."""
        if self.cfg.start_tip_above_entrance is None or self._start_ik is None:
            return float(insertion_tasks_cfg.WORKCELL_HOME_TIP_ABOVE_ENTRANCE)
        buf = getattr(self, "_start_height", None)
        if buf is not None and (
            getattr(self, "_start_sampling", False) or getattr(self, "_start_floor_on", False)
        ):
            return float(buf[0])
        return float(self.cfg.start_tip_above_entrance)

    def _print_controller_report(self) -> None:
        """Report controller settings, measured joint drives, task-space inertia and Jacobian conditioning."""
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
            print(f"  step limits:  {cfg.osc_pos_step_limit_m} m, {cfg.osc_rot_step_limit_rad} rad "
                  f"({math.degrees(cfg.osc_rot_step_limit_rad):.2f} deg) per policy step "
                  f"[placeholder, Factory thresholds, D-197]")
            print(f"  action EMA:   {cfg.action_ema_factor} on the new clamped draw, applied = EMA state "
                  f"[placeholder, Factory ema_factor, D-197]; obs prev_action = EMA state; "
                  f"action-rate term measured on the RAW draw, weight {cfg.action_rate_scale} "
                  f"(Factory 0.0)")
            print(f"  tip clamp:    +-{cfg.osc_pos_clamp_m} m x/y [placeholder, D-180 rule at D-197's step: "
                  f"disk reach + one action step + set air] and "
                  f"+-{cfg.osc_pos_clamp_z_m} m z [placeholder, D-180 rule at D-197's step: highest "
                  f"legal start + one action step + set air] box around the pocket entrance; tilt cone "
                  f"{math.degrees(cfg.osc_tilt_clamp_rad):.2f} deg from THIS EPISODE'S POCKET AXIS "
                  f"(CAD, Geometrie_Fuegeteil_Aufnahme.tex:264)")
            print(f"  yaw band:     +-{math.degrees(cfg.osc_yaw_clamp_rad):.2f} deg about THIS EPISODE'S SEATED "
                  f"orientation (pocket * SEATED_TOOL_QUAT_LOCAL) [placeholder, own, D-196 rule at D-197's "
                  f"step: worst start yaw + belief error + one rotation step]")
            try:
                _seat_w = insertion_math.quat_mul(self._fixture_quat, self._seat_quat_local)
                _x_seat = insertion_math.rotate_into_frame(
                    insertion_math.axes_from_quat(_seat_w),
                    insertion_math.axes_from_quat(self.robot.data.body_quat_w[:, self._osc_body_idx])[:, :, 0],
                )
                _yaw_deg = torch.rad2deg(torch.atan2(_x_seat[:, 1], _x_seat[:, 0]))
                print(f"  reset yaw vs seat: max |yaw| {float(_yaw_deg.abs().max()):.3f} deg over "
                      f"{self.num_envs} envs (env 0 {float(_yaw_deg[0]):+.3f}); expect about 0 at DR width 0 "
                      f"-- the home tool IS the seated orientation (RT-102, D-196 seam 0)")
            except Exception as _exc:  # noqa: BLE001 -- report line, never a crash
                print(f"  reset yaw vs seat: UNREAD -- {type(_exc).__name__}: {_exc}")
            print(f"  controlled body: {self.robot.body_names[self._osc_body_idx]} (index {self._osc_body_idx}, "
                  f"jacobian row block {self._osc_jacobi_idx}), tip offset {self._tip_offset_local.tolist()} m")
            print(f"  rates:        policy {_rate:.1f} Hz (decimation {cfg.decimation} [placeholder, Factory]), "
                  f"controller {1.0 / cfg.sim.dt:.1f} Hz; episode {self.max_episode_length} steps = "
                  f"{cfg.episode_length_s:.4f} s (step count kept, seconds follow -- re-derivation owed)")
        else:
            print(f"control_mode:   joint_pd -- D-108 joint-target integrator on the USD drives "
                  f"(action_scale {cfg.action_scale} rad/step), policy {_rate:.1f} Hz; the PD half of the "
                  f"Audit 2026-09-03 (a) measurement, not a training option")
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
        commanded = self._joint_targets
        home_pose = self.robot.data.default_joint_pos
        down = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        sim_time = self._obs_calls * cfg.sim.dt * cfg.decimation
        print("=" * 72)
        print("INSERTION STARTUP REPORT -- REAL geometry and physics, PROXY scoring (UNVERIFIED)")
        print(f"step {self._obs_calls} of {self._report_at_obs_calls}, t = {sim_time:.3f} s after reset")
        print(f"code marker: {CODE_MARKER}")
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
        print(f"dr_mode: {cfg.dr_mode}")
        if self._dr is not None:
            _b = self._dr.bounds()
            _m = self._dr.bounds_max()
            print(f"  provider: {type(self._dr).__name__}, {self._dr.n_boundaries} boundaries, "
                  f"bounds_version {self._dr.bounds_version}")
            print("  quantity            now [lo, hi]                 max [lo, hi]")
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
            for _line in autodr.provenance_lines(self._dr.dims):
                print(f"  [provenance] {_line}")
            if self._eval_table is not None:
                print(f"  [eval-table] {self._eval_table.path}  rows "
                      f"{self._eval_table.n_rows}  sha256 {self._eval_table.sha256}  "
                      f"width {cfg.dr_table_width}. Env i plays row i in every "
                      f"episode; no boundary sampling (p_boundary "
                      f"{self._dr.p_boundary:.2f}), nothing moves. Values outside "
                      f"[0, 1] allowed (OOD table): {self._eval_table.allow_outside}.")
            if isinstance(self._dr, autodr.AutoDR):
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
                      f"({float(cfg.fixture_pos_noise_xy):.4f} m) and _z "
                      f"({float(cfg.fixture_pos_noise_z):.4f} m), the joint noise, the "
                      f"tilt azimuth and the lateral angle lat_phi stay LIVE by design (plan "
                      f"section 1, D-178 (2)) and no boundary tracks them.")
            print("  NOT VERIFIED IN A SIMULATOR: this build has never run. Read the "
                  "[autodr] lines and dr/ curves of THIS run before believing it.")
        print(f"fixture pose noise (D-034, D-204): xy +-{cfg.fixture_pos_noise_xy:.3f} m, "
              f"z +-{cfg.fixture_pos_noise_z:.3f} m, "
              f"yaw +-{cfg.fixture_yaw_noise_rad:.4f} rad "
              f"({'ACTIVE -- kinematic fixture teleported per reset' if (cfg.fixture_pos_noise_xy > 0.0 or cfg.fixture_pos_noise_z > 0.0 or cfg.fixture_yaw_noise_rad > 0.0) else 'inactive -- fixture stays at the spawn pose'})")
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
        print(f"block boxes:    {len(wc.block_recess_boxes())} "
              f"(incl. the two +Y slot fillers beside the tab: "
              f"{(wc.POCKET_TAB_X_RANGE[0]-wc.POCKET_LOCAL_X_RANGE[0])*1000:.1f} and "
              f"{(wc.POCKET_LOCAL_X_RANGE[1]-wc.POCKET_TAB_X_RANGE[1])*1000:.1f} mm long, "
              f"{(wc.POCKET_LOCAL_Y_RANGE[1]-wc.POCKET_BODY_Y_MAX)*1000:.1f} mm deep)")
        print(f"tables usd:     {self._tables_usd_path}")
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
        _lim = self.robot.data.soft_joint_pos_limits[0]
        _def = self.robot.data.default_joint_pos[0]
        for _j, _name in enumerate(self.robot.joint_names):
            print(f"  joint limit  {_name:<20} [{_lim[_j, 0]:+.6f}, {_lim[_j, 1]:+.6f}]  "
                  f"reset writes {_def[_j]:+.6f}"
                  f"{'   <-- OUTSIDE' if not (_lim[_j, 0] <= _def[_j] <= _lim[_j, 1]) else ''}")
        print(f"body_names:  {self.robot.body_names}")
        print(f"ee_body:     {cfg.ee_body_name} (index {self._ee_body_idx})")
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
        _fric_r = self.robot.root_physx_view.get_material_properties()[..., 0:2]
        _fric_f = self._fixture.root_physx_view.get_material_properties()[..., 0:2]
        _fric_head = (
            f"centre {insertion_tasks_cfg.CONTACT_FRICTION}, per-reset band, PROVISIONAL/F3"
            if self._friction_reach
            else f"u = {insertion_tasks_cfg.CONTACT_FRICTION} written once, PROVISIONAL/F3"
        )
        print(f"contact friction (D-111, {_fric_head}): "
              f"PhysX reads back robot [{_fric_r.min():.3f}, {_fric_r.max():.3f}], "
              f"fixture [{_fric_f.min():.3f}, {_fric_f.max():.3f}] (static+dynamic pooled)")
        if self._friction_reach:
            _n_rep = min(4, self.num_envs)
            _lo, _hi = self._dr.bounds()["friction"]
            _read = [float(_fric_r[_e, :, 0].mean()) for _e in range(_n_rep)]
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
            print(f"             GATE MATH IS STILL SQUARE: side {cfg.held_asset.side:.4f} m from the proxy "
                  f"peg; the part's {cfg.part_bbox_m[1]:.4f} m long axis is not modelled yet.")
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
        resting_speed = torch.linalg.norm(joint_vel_fd[: min(2, self.num_envs)], dim=-1)
        print(f"resting FD joint speed (env 0..1): {resting_speed.tolist()} rad/s  (expect ~0.000 after reset)")
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
        print(f"pocket-yaw belief error env0..{_n_obs - 1} (D-194, "
              f"+-{cfg.pocket_obs_yaw_rad} rad about the pocket axis, drawn "
              f"per EPISODE): "
              f"{[round(v, 6) for v in self._pocket_obs_yaw[:_n_obs].tolist()]}")
        print(f"rl_terms_enabled: {self._rl_terms_enabled} "
              + ("(validate_rl_config PASSED)" if self._rl_terms_enabled
                 else "*** MEASUREMENT ENV: no real reward, no force abort, no curriculum; "
                      "guard SKIPPED -- never a training run ***"))
        print("curriculum: "
              + (f"ON, rung_step_sizes = {self.cfg.rung_step_sizes}"
                 if bool(self.cfg.curriculum_enabled)
                 else "*** OFF -- FIXED RUNG 0, no advance, no retreat; the start "
                      "scatter is whatever the cfg holds and never widens ***"))
        if self._rl_terms_enabled and self._sdf_query is not None:
            print("reward (D-109 + the 2026-09-01 revision): kernel_sum(sdf_dist) "
                  "+ engaged + success bonuses, SAPU-scaled; "
                  f"time penalty {self.cfg.time_penalty_per_step:+.6f}/step "
                  f"(-1/T at T = {self.cfg.episode_steps}); action-rate penalty "
                  f"-{self.cfg.action_rate_scale} * ||a_t - a_(t-1)||")
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
        if RL_PLACEHOLDERS:
            print("*** PLACEHOLDER VALUES IN USE -- every number this run reports "
                  "is provisional ***")
            for name, mark in RL_PLACEHOLDERS.items():
                print(f"    {name} = {getattr(self.cfg, name)}  {mark}")
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
            expect_flange_z = self._start_tip_height + cfg.peg_length
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
                if self._tilt_rot is None:
                    tip_rel_e = tip_local - self._entrance_pos[e]
                    axes_p = peg_axes
                    depth = -tip_rel_e[2]
                else:
                    tip_rel_e = self._tilt_rot_t[e] @ (tip_local - self._entrance_pos[e])
                    axes_p = self._tilt_rot_t[e] @ peg_axes
                    depth = -tip_rel_e[2]
                along_tool = torch.dot(tip_w - ee_pos_w, axes[:, 2]).item()
                tilt_c = math.cos(commanded_mag)
                tilt_s = abs(math.sin(commanded_mag))
                home_tip_height = self._start_tip_height
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
                      f"(expect +-0.000 at home BEFORE reset noise; "
                      f"~uniform in +-{math.degrees(cfg.reset_yaw_noise):.2f} after)")
                print(f"  (cos 4phi, sin 4phi):  ({math.cos(4.0 * phi_raw):+.4f}, {math.sin(4.0 * phi_raw):+.4f})"
                      f"  (expect (+1, 0) at home before reset noise)")
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
