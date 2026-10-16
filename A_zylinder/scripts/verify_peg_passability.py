# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Check whether the attached peg can enter the bore without collision."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import traceback

import numpy as np

from isaaclab.app import AppLauncher

SCRIPT_MARKER = "verify_peg_passability-2026-07-26a"
SECTIONS = "setup, ik-path, centred-pass, offset-pass, verdict"

parser = argparse.ArgumentParser(description="Measure peg/bore passability (peg increment).")
parser.add_argument("--step", type=float, default=0.001, help="Descent increment [m] (default 1 mm).")
parser.add_argument("--depth-end", type=float, default=0.027, help="Deepest tip depth [m]; bore bottom is 0.030.")
parser.add_argument("--offset", type=float, default=0.004, help="Negative-control lateral offset [m].")
parser.add_argument("--hold-steps", type=int, default=4, help="Written steps per waypoint before the free step.")
parser.add_argument("--json", type=str, default=None, help="Metrics output path; default sits next to the peg USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

print(f"[passability] marker: {SCRIPT_MARKER}; sections: {SECTIONS}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from proxytask.tasks.direct.proxytask.proxytask_env import ProxytaskEnv  # noqa: E402
from proxytask.tasks.direct.proxytask.proxytask_env_cfg import ProxytaskEnvCfg  # noqa: E402
from proxytask.tasks.direct.proxytask import proxytask_tasks_cfg as geom  # noqa: E402

_IK_PATH = pathlib.Path(__file__).resolve().parent / "tools" / "compute_home_pose.py"
_spec = importlib.util.spec_from_file_location("chp", _IK_PATH)
chp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chp)

DH = chp.DH_TABLES["ur10e"]
RZ_PI = np.diag([-1.0, -1.0, 1.0])
R_ENV_TOOL_DOWN = np.diag([1.0, -1.0, -1.0])


def ik_for_depth(depth: float, offset_xy: tuple[float, float], q_prev: np.ndarray) -> np.ndarray | None:
    """Joint angles putting the peg tip at ``depth`` below the plate top."""
    flange_z = geom.PLATE_TOP_Z - depth + geom.PEG_LENGTH
    p_env_rel = np.array(
        [
            geom.BORE_ENTRANCE_POS[0] + offset_xy[0] - geom.ROBOT_BASE_POS[0],
            geom.BORE_ENTRANCE_POS[1] + offset_xy[1] - geom.ROBOT_BASE_POS[1],
            flange_z - geom.ROBOT_BASE_POS[2],
        ]
    )
    target = np.eye(4)
    target[:3, 3] = RZ_PI.T @ p_env_rel
    target[:3, :3] = RZ_PI.T @ R_ENV_TOOL_DOWN

    best, best_dist = None, float("inf")
    for raw in chp.inverse_kinematics(target, DH):
        q = chp.wrap_to_pi(np.asarray(raw))
        if float(np.linalg.norm(chp.forward_kinematics(q, DH)[:3, 3] - target[:3, 3])) > 1e-9:
            continue
        dist = float(np.max(np.abs(chp.wrap_to_pi(q - q_prev))))
        if dist < best_dist:
            best, best_dist = q, dist
    if best is None or best_dist > 0.05:
        return None
    return best


def main() -> None:
    cfg = ProxytaskEnvCfg()
    cfg.scene.num_envs = 1
    cfg.report_at_steps = (2,)
    env = ProxytaskEnv(cfg)
    robot = env.robot
    device = env.device

    result: dict = {
        "marker": SCRIPT_MARKER,
        "peg_length_m": geom.PEG_LENGTH,
        "peg_diameter_m": geom.PEG_DIAMETER,
        "bore_diameter_m": geom.BORE_DIAMETER,
        "radial_clearance_m": geom.RADIAL_CLEARANCE,
        "success_depth_m": geom.SUCCESS_DEPTH,
        "negative_control_offset_m": args_cli.offset,
    }

    peg_idx = env._peg_body_idx
    if peg_idx is None:
        result["verdict"] = "NO_PEG"
        result["error"] = (
            f"body '{cfg.peg_body_name}' is absent; the welded link did not survive authoring. "
            f"bodies: {robot.body_names}"
        )
        print(f"[passability] {result['error']}")
        return finish(result, env)

    try:
        dh_to_art = [robot.joint_names.index(n) for n in chp.JOINT_NAMES_DH_ORDER]
    except ValueError as exc:
        result["verdict"] = "INCONCLUSIVE"
        result["error"] = f"joint name mismatch: {exc}; articulation has {robot.joint_names}"
        print(f"[passability] {result['error']}")
        return finish(result, env)
    result["joint_names"] = list(robot.joint_names)
    result["dh_to_articulation_index"] = dh_to_art
    print(f"[passability] joint map (DH -> articulation): {dh_to_art}")
    print(f"[passability] peg body '{cfg.peg_body_name}' at index {peg_idx}")

    env.reset()

    depth_start = geom.PEG_LENGTH - geom.HOME_STANDOFF_Z
    depths = [
        depth_start + i * args_cli.step
        for i in range(int(round((args_cli.depth_end - depth_start) / args_cli.step)) + 1)
    ]
    print(f"[passability] {len(depths)} waypoints from {depths[0]:+.4f} m to {depths[-1]:+.4f} m "
          f"in {args_cli.step*1000:.1f} mm steps")

    offset_depths = [d for d in depths if d <= 0.006]
    for label, offset, sweep in (
        ("centred", (0.0, 0.0), depths),
        ("offset", (args_cli.offset, 0.0), offset_depths),
    ):
        print(f"[passability] --- {label} pass, lateral offset {offset[0]*1000:.1f} mm, "
              f"{len(sweep)} waypoints to {sweep[-1]:+.4f} m ---")
        result[label] = run_pass(env, robot, device, peg_idx, dh_to_art, sweep, offset)

    result.update(judge(result))
    return finish(result, env)


def run_pass(env, robot, device, peg_idx, dh_to_art, depths, offset) -> dict:
    """Walk the descent once and record the free-step reaction at every depth."""
    env.reset()
    q_prev = np.array([float(robot.data.default_joint_pos[0, dh_to_art[i]]) for i in range(6)])
    waypoints: list[dict] = []
    unreachable: list[float] = []

    for depth in depths:
        q_dh = ik_for_depth(depth, offset, q_prev)
        if q_dh is None:
            unreachable.append(round(depth, 5))
            continue
        q_prev = q_dh

        q_art = torch.zeros_like(robot.data.default_joint_pos[:1])
        for i, art_i in enumerate(dh_to_art):
            q_art[0, art_i] = float(q_dh[i])
        zero_vel = torch.zeros_like(q_art)

        for _ in range(args_cli.hold_steps):
            step_sim(env, robot, q_art, zero_vel)
        step_sim(env, robot, None, None)

        delta = robot.data.joint_pos[0] - q_art[0]
        deviation = float(torch.max(torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta)))).item())
        if math.isnan(deviation) or math.isinf(deviation):
            print(f"[passability] non-finite deviation at depth {depth:+.4f} m; recorded as a hard reaction")
            deviation = 1e3
        peg_vel = peg_speed(robot, peg_idx)
        tip = peg_tip_env_local(env, robot, peg_idx)
        waypoints.append(
            {
                "commanded_depth_m": round(depth, 6),
                "measured_depth_m": round(env.cfg.plate_top_z - tip[2], 6),
                "tip_xy_offset_m": round(math.hypot(tip[0] - env.cfg.bore_entrance_pos[0],
                                                    tip[1] - env.cfg.bore_entrance_pos[1]), 6),
                "free_step_joint_deviation_rad": round(deviation, 8),
                "free_step_peg_speed_mps": None if peg_vel is None else round(peg_vel, 8),
            }
        )

    return {"waypoints": waypoints, "unreachable_depths_m": unreachable}


def step_sim(env, robot, q, qd) -> None:
    """One physics step, optionally writing the joint state first."""
    if q is not None:
        _guard("write_joint_state_to_sim", lambda: robot.write_joint_state_to_sim(q, qd, None, None))
        _guard("set_joint_position_target", lambda: robot.set_joint_position_target(q))
    _guard("scene.write_data_to_sim", lambda: env.scene.write_data_to_sim())
    _guard("sim.step", lambda: env.sim.step(render=False))
    _guard("scene.update", lambda: env.scene.update(env.cfg.sim.dt))


_WARNED: set[str] = set()


def _guard(name: str, call) -> None:
    try:
        call()
    except (AttributeError, TypeError) as exc:
        if name not in _WARNED:
            _WARNED.add(name)
            print(f"[passability] WARNING: {name} unavailable ({exc}); continuing without it")


def peg_speed(robot, peg_idx) -> float | None:
    """Peg linear speed, or None if this Isaac Lab version does not expose it."""
    try:
        return float(torch.linalg.norm(robot.data.body_lin_vel_w[0, peg_idx]).item())
    except (AttributeError, IndexError, TypeError):
        if "body_lin_vel_w" not in _WARNED:
            _WARNED.add("body_lin_vel_w")
            print("[passability] WARNING: body_lin_vel_w unavailable; peg speed omitted, verdict unaffected")
        return None


def peg_tip_env_local(env, robot, peg_idx) -> list[float]:
    pos = robot.data.body_pos_w[0, peg_idx] - env.scene.env_origins[0]
    q = robot.data.body_quat_w[0, peg_idx]
    w, x, y, z = (float(v) for v in q)
    axis_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return [float(pos[i]) + geom.PEG_LENGTH * axis_z[i] for i in range(3)]


REACTION_FACTOR = 2.0


def judge(result: dict) -> dict:
    """Turn the two passes into a verdict by paired comparison."""
    centred = result["centred"]["waypoints"]
    offset = result["offset"]["waypoints"]
    if not centred or not offset:
        return {"verdict": "INCONCLUSIVE", "reason": "a pass produced no reachable waypoints"}

    def react(w):
        return w["free_step_joint_deviation_rad"]

    def contact_free_max(waypoints) -> float | None:
        vals = [react(w) for w in waypoints if w["commanded_depth_m"] < -0.005]
        return max(vals) if vals else None

    centred_base = contact_free_max(centred)
    offset_base = contact_free_max(offset)
    if centred_base is None or offset_base is None:
        return {"verdict": "INCONCLUSIVE", "reason": "a pass has no contact-free waypoints to calibrate on"}

    threshold = max(REACTION_FACTOR * centred_base, 1e-6)
    centred_max = max(
        (react(w) for w in centred if 0.0 <= w["commanded_depth_m"] <= result["success_depth_m"]), default=0.0
    )
    offset_max = max((react(w) for w in offset if w["commanded_depth_m"] >= 0.0), default=0.0)
    centred_ratio = centred_max / centred_base if centred_base > 0 else float("inf")
    offset_ratio = offset_max / offset_base if offset_base > 0 else float("inf")

    first_reaction = next(
        (
            w["commanded_depth_m"]
            for w in sorted(offset, key=lambda x: x["commanded_depth_m"])
            if w["commanded_depth_m"] >= -0.005 and react(w) > REACTION_FACTOR * offset_base
        ),
        None,
    )

    deepest_clean = float("-inf")
    for w in sorted(centred, key=lambda x: x["commanded_depth_m"]):
        if react(w) > threshold:
            break
        deepest_clean = w["commanded_depth_m"]

    metrics = {
        "centred_contact_free_max_rad": centred_base,
        "offset_contact_free_max_rad": offset_base,
        "centred_max_in_bore_rad": centred_max,
        "offset_max_at_entrance_rad": offset_max,
        "centred_ratio_to_own_baseline": centred_ratio,
        "offset_ratio_to_own_baseline": offset_ratio,
        "reaction_factor_used": REACTION_FACTOR,
        "verdict_stable_for_factors_between": [centred_ratio, offset_ratio],
        "offset_first_reaction_depth_m": first_reaction,
        "threshold_rad": threshold,
        "deepest_clean_depth_m": None if deepest_clean == float("-inf") else deepest_clean,
        "reached_success_depth": deepest_clean >= result["success_depth_m"] - 1e-9,
    }
    deepest_txt = "never clear of the plate" if deepest_clean == float("-inf") else f"{deepest_clean*1000:.1f} mm"

    collider_reacts = offset_ratio > REACTION_FACTOR and first_reaction is not None
    peg_passes = centred_ratio <= REACTION_FACTOR and metrics["reached_success_depth"]
    if not collider_reacts:
        verdict, reason = "COLLISION_ABSENT", (
            f"the negative control stayed within {offset_ratio:.2f}x of its own contact-free noise "
            f"even with the peg axis {args_cli.offset*1000:.0f} mm off the bore, so the fixture collider is not acting on "
            "the peg; a clean centred pass proves nothing until this is fixed"
        )
    elif peg_passes:
        verdict, reason = "PASSABLE", (
            f"the centred peg reached {result['success_depth_m']*1000:.0f} mm at {centred_ratio:.2f}x its own "
            f"contact-free noise, i.e. without touching anything, while the "
            f"{args_cli.offset*1000:.0f} mm offset control rose to "
            f"{offset_ratio:.2f}x its own noise from {first_reaction*1000:.0f} mm depth onward. The two "
            f"differ only in lateral position, so the verdict rests on that separation and holds for any "
            f"reaction factor between {centred_ratio:.2f} and {offset_ratio:.2f}"
        )
    elif centred_max > threshold:
        verdict, reason = "BLOCKED", (
            f"the centred peg stays within baseline only to {deepest_txt} and reacts beyond it, "
            f"short of the {result['success_depth_m']*1000:.0f} mm success depth; the bore is obstructed"
        )
    else:
        verdict, reason = "INCONCLUSIVE", "the peg stayed quiet but never reached the success depth"
    return {"metrics": metrics, "verdict": verdict, "reason": reason}


def finish(result: dict, env=None) -> None:
    out = pathlib.Path(args_cli.json) if args_cli.json else None
    if out is None:
        try:
            out = pathlib.Path(geom.resolve_peg_robot_usd_path()).with_suffix(".passability.json")
        except FileNotFoundError:
            out = pathlib.Path(__file__).resolve().parent / "peg_passability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("=" * 72)
    for key, value in result.get("metrics", {}).items():
        print(f"[passability] {key}: {value}")
    print(f"[passability] VERDICT: {result.get('verdict', 'INCONCLUSIVE')}")
    print(f"[passability] {result.get('reason', result.get('error', ''))}")
    print(f"[passability] metrics written to {out}")
    print("=" * 72)
    if env is not None:
        try:
            env.close()
        except Exception:  # noqa: BLE001 - closing must never mask the result above
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - a traceback here is the whole return on a round trip
        traceback.print_exc()
    finally:
        simulation_app.close()
