# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Scene, controller, observation and reward configuration for insertion."""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from . import insertion_math, insertion_tasks_cfg
from .agents.rsl_rl_ppo_cfg import PPORunnerCfg
from .insertion_tasks_cfg import FixedAssetCfg, HeldAssetCfg
from .ur5e_cfg import UR5E_HOME_CFG


def resolve_fixture_usd_path() -> str:
    """Return the path to the fixture USD the ENV should spawn, or raise."""
    return insertion_tasks_cfg.resolve_pocket_usd_path()


def resolve_proxy_tisch_usd_path() -> str:
    """Return the path to the square-pocket Tisch USD, or raise."""
    env_override = os.environ.get("INSERTION_TISCH_USD")
    candidates = []
    if env_override:
        candidates.append(env_override)
    asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Tisch")
    candidates.append(os.path.join(asset_dir, insertion_tasks_cfg.default_fixture_usd_name()))
    for path in candidates:
        if os.path.isfile(path):
            return path
    pocket_mm = insertion_tasks_cfg.POCKET_SIDE * 1000.0
    raise FileNotFoundError(
        "Square-pocket Tisch USD not found. Tried: "
        + "; ".join(candidates)
        + ". Its generator (scripts/author_tisch_square.py) was DELETED on "
        "2026-08-28 with the dead proxy code (S6), so the table cannot be "
        "rebuilt on this branch -- recover the script from git history, or set "
        f"INSERTION_TISCH_USD to a table whose pocket really is {pocket_mm:g} mm. "
        "A size-less tisch_square.usd and the round tisch.usd are NOT "
        "accepted; the pocket size is part of the filename because it is the curriculum "
        "variable."
    )


@configclass
class InsertionEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 256 / 60
    action_space = 6
    observation_space = insertion_math.OBS_DIM
    state_space = 0

    action_scale = 0.02
    control_mode: str = "osc"
    obs_wrench_mode: str = "force"
    osc_kp_pos: float = 100.0
    osc_kp_rot: float = 30.0
    osc_damping_ratio: float = 1.0
    osc_pos_step_limit_m: float = 0.02
    osc_rot_step_limit_rad: float = 0.097
    osc_pos_clamp_m: float = 0.08
    osc_pos_clamp_z_m: float = 0.1435
    osc_tilt_clamp_rad: float = math.radians(8.52)
    osc_decimation: int = 8
    reset_joint_noise = 0.0
    reset_yaw_noise = 0.0

    start_tip_above_entrance: float | None = (
        insertion_tasks_cfg.RUNG0_START_TIP_ABOVE_ENTRANCE
    )
    start_tip_above_entrance_low: float | None = (
        insertion_tasks_cfg.RUNG0_START_TIP_ABOVE_ENTRANCE
    )
    start_floor_m: float = -0.030
    start_floor_steps: int = 5
    autodr_stall_buffers: int = 3
    start_lateral_offset: float = 0.0
    start_pose_solve_steps: int = 120
    start_pose_solve_tol_m: float = 0.00005
    start_pose_ik_lambda: float = 0.05
    yaw_joint_name = "wrist_3_joint"
    gate_min_alignment = 0.99

    metrics_window_iterations = 20


    metrics_success_threshold = 0.9

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    robot_cfg: ArticulationCfg = UR5E_HOME_CFG
    ee_body_name = "wrist_3_link"

    workcell_tables_pos = insertion_tasks_cfg.WORKCELL_TABLES_POS

    spawn_workcell_block: bool = False
    workcell_block_pos = insertion_tasks_cfg.WORKCELL_BLOCK_POS
    floor_z = insertion_tasks_cfg.FLOOR_Z

    fixture_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Fixture",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=insertion_tasks_cfg.WORKCELL_BLOCK_POS),
    )

    dr_mode: str = "off"

    dr_table_path: str | None = None
    dr_table_width: str = "target"

    fixture_pos_noise_xy = 0.005
    fixture_yaw_noise_rad = 0.0
    fixture_tilt_rad = 0.0
    fixture_yaw_rad = 0.0
    fixture_tilt_noise_rad = 0.0

    fixture_pos = insertion_tasks_cfg.WORKCELL_BLOCK_POS
    plate_top_z = insertion_tasks_cfg.WORKCELL_PLATE_TOP_Z
    plate_bottom_z = insertion_tasks_cfg.WORKCELL_PLATE_BOTTOM_Z
    plate_half_extents = insertion_tasks_cfg.WORKCELL_PLATE_HALF_EXTENTS
    opening_entrance_pos = insertion_tasks_cfg.WORKCELL_ENTRANCE_POS
    robot_base_pos = insertion_tasks_cfg.WORKCELL_ROBOT_BASE_POS
    home_standoff_z = insertion_tasks_cfg.HOME_STANDOFF_Z

    fixed_asset: FixedAssetCfg = FixedAssetCfg(
        depth=insertion_tasks_cfg.POCKET_SEAT_DEPTH,
        success_depth=insertion_tasks_cfg.SEATED_SUCCESS_DEPTH,
    )
    held_asset: HeldAssetCfg = HeldAssetCfg()
    peg_body_name = insertion_tasks_cfg.TOOL_LINK_NAME
    peg_tip_offset = (0.0, 0.0, insertion_tasks_cfg.FLANGE_TO_PART_BOTTOM)
    peg_length = insertion_tasks_cfg.FLANGE_TO_PART_BOTTOM

    part_bbox_m = insertion_tasks_cfg.PART_BBOX_M

    tool_mass_kg: float | None = 0.824

    report_at_steps = (2, 60, 180, 258)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=3.0, replicate_physics=True)


    force_abort_f_max_n: float = 30.0

    abort_payment: float = -1.0

    engaged_depth_m: float = insertion_tasks_cfg.ENGAGED_DEPTH

    curriculum_enabled: bool = False

    rung_step_sizes: dict | None = None

    force_sensor_body_name: str = "tool_link"

    force_gravity_tare: bool = True

    obs_noise_pocket_pos_std_m: float = 0.005
    force_obs_noise_std_n: float = 3.5
    torque_obs_noise_std_nm: float = 0.0
    torque_ref_offset_parent_m: tuple[float, float, float] = (0.0, 0.0, -0.1032)

    grasp_obs_offset_x_m: float = 0.003

    ft_smoothing_factor: float = 0.25


    kernel_a_coarse: float = insertion_tasks_cfg.KERNEL_A_COARSE
    kernel_b_coarse: float = insertion_tasks_cfg.KERNEL_B_COARSE
    kernel_a_mid: float = insertion_tasks_cfg.KERNEL_A_MID
    kernel_b_mid: float = insertion_tasks_cfg.KERNEL_B_MID
    kernel_a_fine: float = insertion_tasks_cfg.KERNEL_A_FINE
    kernel_b_fine: float = insertion_tasks_cfg.KERNEL_B_FINE
    kernel_a_tilt: float = insertion_tasks_cfg.KERNEL_A_TILT

    w_engaged: float = 1.0
    w_success: float = 1.0

    w_depth_progress: float = 100.0

    w_tilt: float = 0.0
    shaping_gamma: float = PPORunnerCfg().algorithm.gamma


    episode_steps: int = math.ceil(episode_length_s / (sim.dt * decimation))

    time_penalty_per_step: float = -1.0 / episode_steps

    action_rate_scale: float = 0.0034

    interpen_thresh: float = insertion_tasks_cfg.INTERPEN_THRESH

    depth_min: float = insertion_tasks_cfg.SEATED_SUCCESS_DEPTH
    depth_max: float = insertion_tasks_cfg.POCKET_SEAT_DEPTH


    sdf_num_sample_points: int = 64000

    sdf_max_dist: float = 1.5

    sdf_seed: int = 0

    rl_terms_enabled: bool = True


RL_PENDING: tuple[str, ...] = (
    "rung_step_sizes",
)

RL_PENDING_MARKS: dict = {
    "rung_step_sizes": "[open] Set from learning curves",
}

CURRICULUM_PENDING: tuple[str, ...] = (
    "rung_step_sizes",
)

RL_PLACEHOLDERS: dict = {
    "force_abort_f_max_n": (
        "[placeholder] Chosen simulation force-abort threshold, not a measured damage limit. "
        "Lowering it can abort otherwise successful insertions. Inspect force p95 and abort "
        "rate; the rigid tool chain cannot establish a real suction-cup force limit."
    ),
    "engaged_depth_m": (
        "[placeholder] Ad hoc fraction of seat depth used only by the engaged display bonus. "
        "Scripted-controller capture depth is not a validated threshold for a learned "
        "policy."
    ),
    "abort_payment": (
        "[placeholder] Fixed negative reward on abort, scaled to one shaping-kernel peak. "
        "This magnitude is a design choice, not a measured quantity; abort also forfeits the "
        "remaining episode return."
    ),
    "osc_kp_pos": (
        "[placeholder] Task-space position stiffness in N/m. Candidate bound: kp <= F_search "
        "/ (Lambda_max * step_limit), using task-space inertia at reset. Contact-free "
        "default gains do not validate a contact-force limit."
    ),
    "osc_kp_rot": (
        "[placeholder] Task-space rotation stiffness in Nm/rad, inherited from Factory "
        "CtrlCfg. Not independently validated for this task."
    ),
    "osc_pos_step_limit_m": (
        "[placeholder] Maximum tool translation per action in metres, inherited from Factory "
        "pos_action_threshold. A task-specific allowed tool speed has not been established."
    ),
    "osc_rot_step_limit_rad": (
        "[placeholder] Maximum tool rotation per action in radians, inherited from Factory "
        "rot_action_threshold. A task-specific allowed angular speed has not been "
        "established."
    ),
    "osc_pos_clamp_m": (
        "[placeholder] Horizontal workspace half-width: lateral radius plus tool-tip swing "
        "at maximum tilt, one action step and clearance. check_env_wiring checks the lower "
        "bound and excess clearance."
    ),
    "osc_pos_clamp_z_m": (
        "[placeholder] Vertical workspace half-width: maximum start-height reach including "
        "tilt and lateral offset, one action step and clearance. check_env_wiring checks "
        "world-frame coverage and the home-tip standoff."
    ),
    "osc_decimation": (
        "[placeholder] Physics steps per policy step, inherited from Factory. Policy rate is "
        "physics rate divided by decimation; the fixed episode step count determines the "
        "duration."
    ),
    "obs_noise_pocket_pos_std_m": (
        "[placeholder] Half-width in metres of a per-episode uniform pocket-position "
        "observation bias, despite the std suffix. TacSL reports uniform observation noise; "
        "drawing it once per episode is a modelling assumption."
    ),
    "force_obs_noise_std_n": (
        "[placeholder] Per-step Gaussian force-observation noise in newtons. Interpreting "
        "the UR5e datasheet precision as a standard deviation is an assumption; the "
        "datasheet does not define it that way."
    ),
    "torque_obs_noise_std_nm": (
        "[placeholder] Per-step Gaussian torque-observation noise in Nm, used only in wrench "
        "mode. Disabled by default. Interpreting torque-sensor precision as a standard "
        "deviation would require the same assumption as for force."
    ),
    "grasp_obs_offset_x_m": (
        "[placeholder] Half-width in metres of uniform observation-only grasp offset along "
        "the short part axis. Factory and FORGE use a comparable magnitude; its "
        "applicability to this gripper is an estimate, not a measurement."
    ),
}

PROXY_TASK_VALUES: dict = {
    "cfg.w_depth_progress": (
        "[proxy] the square-peg proxy's reward_w_depth, paid per metre of "
        "NEW gated max depth. The proxy solved 1 mm clearance with it "
        "at 100; nothing on this task has sized it. Zero switches the term "
        "off and reproduces the previous reward."
    ),
    "cfg.held_asset.side": (
        "[proxy] the held body modelled as a SQUARE of this side. The real "
        "part is rectangular with lugs (PART_BODY_X / PART_BBOX_M). Read by "
        "the four-corner gate; moves with the SDF rework, not alone."
    ),
    "cfg.fixed_asset.side": (
        "[proxy] the square pocket edge the gate tests against. The real "
        "opening is POCKET_OPENING_X x POCKET_OPENING_Y and is not square. "
        "Read by the four-corner gate; moves with the SDF rework."
    ),
    "cfg.gate_min_alignment": (
        "[proxy] tool-axis alignment floor, calibrated on the square peg. "
        "The across-tilt limit has not been recomputed."
    ),
    "tasks.SIDE_CLEARANCE": (
        "[proxy] half-clearance per axis of the square pair. The real task "
        "has TWO different total plays, PLAY_X and PLAY_Y, which the report "
        "now prints instead. Kept only because compare_runs.py reads the "
        "metrics key side_clearance_mm."
    ),
    "tasks.YAW_WINDOW_RAD": (
        "[proxy] the C4 free-yaw window of a square in a square. The real "
        "part fits ONE way round, so there is no equivalent; the yaw "
        "bound has not been recomputed. The report no "
        "longer prints it. Kept only because compare_runs.py reads the "
        "metrics key yaw_window_deg."
    ),
}

PROXY_METRICS_KEYS: dict = {
    "peg_side_m": "cfg.held_asset.side",
    "pocket_side_m": "cfg.fixed_asset.side",
    "side_clearance_mm": "tasks.SIDE_CLEARANCE",
    "yaw_window_deg": "tasks.YAW_WINDOW_RAD",
}


class ProxyTableBroken(RuntimeError):
    """A path in ``PROXY_TASK_VALUES`` does not resolve. Code fault, not data."""


def resolve_proxy_task_values(cfg) -> list[tuple[str, object, str]]:
    """Return ``(path, live value, mark)`` for every proxy value still in use."""
    roots = {"cfg": cfg, "tasks": insertion_tasks_cfg}
    out: list[tuple[str, object, str]] = []
    for path, mark in PROXY_TASK_VALUES.items():
        head, _, rest = path.partition(".")
        if head not in roots or not rest:
            raise ProxyTableBroken(
                f"PROXY_TASK_VALUES path {path!r} does not start with a known "
                f"root; expected one of {sorted(roots)}."
            )
        obj = roots[head]
        for part in rest.split("."):
            if not hasattr(obj, part):
                raise ProxyTableBroken(
                    f"PROXY_TASK_VALUES path {path!r} does not resolve: "
                    f"{type(obj).__name__} has no attribute {part!r}. That is "
                    "a rename or a typo in this file, not a missing value."
                )
            obj = getattr(obj, part)
        out.append((path, obj, mark))
    unknown = [k for k, p in PROXY_METRICS_KEYS.items() if p not in PROXY_TASK_VALUES]
    if unknown:
        raise ProxyTableBroken(
            "PROXY_METRICS_KEYS points at entries that do not exist in "
            f"PROXY_TASK_VALUES: {unknown}."
        )
    return out


CONTROL_MODES: tuple[str, ...] = ("osc", "joint_pd")


def resolve_start_lateral_offset(value) -> float:
    """``cfg.start_lateral_offset`` as the RADIUS of the start disk, in metres."""
    if isinstance(value, (list, tuple)):
        raise ValueError(
            f"start_lateral_offset={value!r}: the (x, y) half-width pair is "
            "unsupported. The lateral start offset is a disk and this "
            "field is ONE radius in metres: env.start_lateral_offset=<radius>."
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"start_lateral_offset={value!r} ({type(value).__name__}) is not a number: "
            "expected ONE radius in metres."
        )
    radius = float(value)
    if radius < 0.0:
        raise ValueError(
            f"start_lateral_offset={value!r}: a radius cannot be negative. Use 0.0 "
            "to start on the pocket axis."
        )
    return radius


def resolve_control_mode(cfg) -> str:
    """Derive the rate fields from ``cfg.control_mode`` and return the mode."""
    mode = str(cfg.control_mode)
    if mode not in CONTROL_MODES:
        raise RlConfigIncomplete(
            f"control_mode is {mode!r}; the only modes are {CONTROL_MODES}."
        )
    if mode == "osc":
        dec = int(cfg.osc_decimation)
        if dec < 1:
            raise RlConfigIncomplete(f"osc_decimation must be >= 1, got {dec}")
        cfg.decimation = dec
        cfg.sim.render_interval = dec
        cfg.episode_length_s = int(cfg.episode_steps) * float(cfg.sim.dt) * dec
    return mode


def resolve_obs_layout(cfg) -> str:
    """Derive ``observation_space`` from ``cfg.obs_wrench_mode`` and return the mode."""
    mode = str(cfg.obs_wrench_mode)
    try:
        width = insertion_math.obs_dim(mode)
    except ValueError as exc:
        raise RlConfigIncomplete(
            f"obs_wrench_mode is {mode!r}; the only modes are "
            f"{insertion_math.OBS_MODES}."
        ) from exc
    cfg.observation_space = width
    return mode


class RlConfigIncomplete(RuntimeError):
    """An env that cannot be configured is a hard stop, never a default."""


DR_TABLE_WIDTHS: tuple[str, ...] = ("target", "nominal")


def validate_table_mode(cfg) -> None:
    """Refuse every ``dr_mode='table'`` combination that must never start."""
    path = cfg.dr_table_path
    if str(cfg.dr_mode) != "table":
        if path is not None:
            raise RlConfigIncomplete(
                f"dr_table_path is {path!r} while dr_mode={cfg.dr_mode!r}. Nothing "
                "reads the table in that mode, so the run would LOOK like an "
                "evaluation and draw its own resets. Set dr_mode='table' or leave "
                "the path None."
            )
        return
    if path is None or not str(path).strip():
        raise RlConfigIncomplete(
            "dr_mode='table' without dr_table_path. The table IS the test cases; "
            "there is no default table."
        )
    if str(cfg.dr_table_width) not in DR_TABLE_WIDTHS:
        raise RlConfigIncomplete(
            f"dr_table_width is {cfg.dr_table_width!r}; known widths: {DR_TABLE_WIDTHS}."
        )
    high = cfg.start_tip_above_entrance
    if high is None or float(cfg.start_floor_m) != float(high):
        raise RlConfigIncomplete(
            f"dr_mode='table' with start_floor_m={cfg.start_floor_m!r} and "
            f"start_tip_above_entrance={high!r}. An evaluation has NO start floor "
            "and needs a start height: leave start_floor_m equal to "
            "start_tip_above_entrance (eval_plan.md G8)."
        )


def validate_rl_config(cfg) -> None:
    """Refuse to build an env while a decided-but-unmeasured number is unset."""
    named = RL_PENDING + tuple(RL_PLACEHOLDERS) + (
        "curriculum_enabled", "episode_steps", "time_penalty_per_step",
        "action_rate_scale", "w_depth_progress",
        "w_tilt", "shaping_gamma", "kernel_a_tilt",
        "dr_mode",
        "dr_table_path", "dr_table_width", "start_floor_m", "start_tip_above_entrance",
    )
    absent = [name for name in named if not hasattr(cfg, name)]
    if absent:
        raise RlConfigIncomplete(
            "the RL config guard reads config fields that do not exist on "
            f"{type(cfg).__name__}: {', '.join(absent)}. That is a code fault "
            "in this file (a rename or a typo), not a missing measurement -- "
            "no number can clear it."
        )
    emptied = [name for name in RL_PLACEHOLDERS if getattr(cfg, name) is None]
    if emptied:
        lines = [f"  {name}: {RL_PLACEHOLDERS[name]}" for name in emptied]
        raise RlConfigIncomplete(
            "a PLACEHOLDER value was emptied. It never carried a measurement, but "
            "the env was built to read it:\n" + "\n".join(lines)
        )
    ladder_on = bool(cfg.curriculum_enabled)
    if ladder_on and str(cfg.dr_mode) != "off":
        raise RlConfigIncomplete(
            f"curriculum_enabled is True together with dr_mode={cfg.dr_mode!r}. "
            "The ladder and AutoDR both widen START HEIGHT and TILT, so on "
            "those two the range an episode actually gets would be whichever "
            "mechanism wrote last, while each one tracked only its own half. "
            "Run the ladder (dr_mode='off') or run AutoDR "
            "(curriculum_enabled=False) -- never both."
        )
    validate_table_mode(cfg)
    if not ladder_on:
        set_anyway = [name for name in CURRICULUM_PENDING if getattr(cfg, name) is not None]
        if set_anyway:
            lines = [f"  {name} = {getattr(cfg, name)!r}" for name in set_anyway]
            raise RlConfigIncomplete(
                "curriculum_enabled is False, so the ladder never advances and "
                "nothing reads a step size -- but one is set:\n"
                + "\n".join(lines)
                + "\nEither switch the ladder on, or leave the value None. A step "
                "size that nothing reads is a number that looks decided and is not."
            )
    demanded = RL_PENDING if ladder_on else tuple(n for n in RL_PENDING if n not in CURRICULUM_PENDING)
    missing = [name for name in demanded if getattr(cfg, name) is None]
    if missing:
        lines = [f"  {name}: {RL_PENDING_MARKS[name]}" for name in missing]
        raise RlConfigIncomplete(
            "the following real-task values are still UNSET and must be measured "
            "or decided before an env may be built:\n" + "\n".join(lines)
        )

    steps = int(cfg.episode_steps)
    if steps <= 0:
        raise RlConfigIncomplete(
            f"episode_steps is {steps}. It is "
            "ceil(episode_length_s / (sim.dt * decimation)) and must be "
            "positive -- the time penalty divides by it."
        )
    want_time = -1.0 / steps
    if abs(float(cfg.time_penalty_per_step) - want_time) > 1e-12:
        raise RlConfigIncomplete(
            f"time_penalty_per_step is {float(cfg.time_penalty_per_step)!r}, but "
            f"the decided rule is R_time = -1/T = {want_time!r} at T = {steps} "
            "steps (Brahmbhatt et al., ICRA 2023, arXiv:2301.12587 Sec. III). A "
            "time penalty that is not -1/T no longer sums to one abort payment "
            "over an episode, and the two safety inequalities are "
            "stated for -1/T alone."
        )
    if float(cfg.action_rate_scale) < 0.0:
        raise RlConfigIncomplete(
            f"action_rate_scale is {float(cfg.action_rate_scale)!r}. The reward "
            "subtracts scale * ||a_t - a_(t-1)||, so a negative scale rewards "
            "the jerk the term exists to price."
        )
    if float(cfg.w_depth_progress) < 0.0:
        raise RlConfigIncomplete(
            f"w_depth_progress is {float(cfg.w_depth_progress)!r}. The reward "
            "pays w * metres of new gated max depth, so a negative weight "
            "charges the part for entering the pocket."
        )
    if float(cfg.w_tilt) < 0.0:
        raise RlConfigIncomplete(
            f"w_tilt is {float(cfg.w_tilt)!r}. The alignment potential is "
            "w * sech(a * theta), so a negative weight pays for tilting."
        )
    if not float(cfg.kernel_a_tilt) > 0.0:
        raise RlConfigIncomplete(
            f"kernel_a_tilt is {float(cfg.kernel_a_tilt)!r}; the tilt kernel "
            "width must be positive (arccosh(10) / margin)."
        )
    if not 0.0 < float(cfg.shaping_gamma) <= 1.0:
        raise RlConfigIncomplete(
            f"shaping_gamma is {float(cfg.shaping_gamma)!r}; the shaping row "
            "gamma * Phi(s') - Phi(s) needs PPO's discount in (0, 1]."
        )
    for _name in ("obs_noise_pocket_pos_std_m", "force_obs_noise_std_n",
                  "torque_obs_noise_std_nm", "grasp_obs_offset_x_m"):
        if float(getattr(cfg, _name)) < 0.0:
            raise RlConfigIncomplete(
                f"{_name} is {float(getattr(cfg, _name))!r}. It is a scatter "
                "width -- a Gaussian sigma or a uniform half width -- so it "
                "may be zero to switch the scatter off, but never negative."
            )
