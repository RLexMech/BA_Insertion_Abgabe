# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure whether the welded peg can occupy the bore -- the peg increment's headline question.

The fixture carries an exact triangle-mesh collider (``add_fixture_collision.py``,
approximation ``none``), so the convex-hull worry deferred in D-020 addendum 2
should be settled. "Should be" is not a measurement, and nothing has ever put a
solid object into that hole. This script does.

Method: a penetration probe, not a servoed insertion. At each waypoint the arm
is written kinematically to an IK solution and held for a few steps, then one
**free** step runs in which nothing is written. If the peg overlaps plate
material, PhysX depenetration shows up in that free step as joint motion and a
peg velocity spike; if the peg sits in clear space, the free step barely moves.
Kinematic writes are used deliberately: the shipped PD gains droop by about
13 mm at steady state (D-026), which is larger than the 1.0 mm radial clearance,
so a servoed descent would measure the gains rather than the geometry.

Three properties make the verdict defensible:

* **Self-calibrating.** The descent starts at the home pose, 100 mm above the
  plate, where contact is geometrically impossible. Those waypoints supply the
  free-step baseline, so the in-bore readings are judged against this robot's
  own gravity and solver noise rather than an invented threshold.
* **Negative control.** A second pass offsets the peg axis so its edge misses
  the bore. That pass *must* react. If it does not, the collider is inert and a
  clean centred pass would have meant nothing -- reported as COLLISION_ABSENT.
* **No new scene code.** The real ``ProxytaskEnv`` is instantiated and driven
  from outside. Nothing in the env is modified, and the geometry under test is
  the one training will use.

UNVERIFIED -- authored on the dev PC (no Isaac installation). On the training
machine, after author_peg_ur10e.py has produced the asset:

    conda activate env_isaaclab
    python scripts\\verify_peg_passability.py
"""

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

# The IK is a standalone numpy module with no Isaac dependency, loaded by path
# because scripts/tools is not a package.
_IK_PATH = pathlib.Path(__file__).resolve().parent / "tools" / "compute_home_pose.py"
_spec = importlib.util.spec_from_file_location("chp", _IK_PATH)
chp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chp)

DH = chp.DH_TABLES["ur10e"]
# Established by matching measured body positions, not assumed (see
# scripts/tools/screen_home_pose_branches.py): p_env = BASE + Rz(pi) . p_DH.
RZ_PI = np.diag([-1.0, -1.0, 1.0])
# Orientation held constant for the whole descent, at exactly the value that
# produced the verified tool-down home pose. Because only z changes, the open
# question of how DH frame 6 maps onto the USD wrist_3_link axes never has to be
# answered here: the descent is a pure translation of a pose already measured.
R_ENV_TOOL_DOWN = np.diag([1.0, -1.0, -1.0])


def ik_for_depth(depth: float, offset_xy: tuple[float, float], q_prev: np.ndarray) -> np.ndarray | None:
    """Joint angles putting the peg tip at ``depth`` below the plate top.

    ``depth`` is positive into the bore. The branch is chosen by continuity with
    ``q_prev`` rather than by index: branch ordering depends on the target, so
    an index that is right at the home pose need not stay right 100 mm lower.
    """
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
        # Verify the branch really reaches the commanded pose before trusting it.
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
    # The startup report would fire mid-descent and interleave with the metrics.
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

    # Joint ordering: the IK works in DH order, the articulation in its own.
    # Getting this wrong produces a plausible-looking but entirely wrong motion,
    # so the mapping is built by name and reported.
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

    # Start at the home pose, whose depth is negative by peg length minus
    # standoff: the tip hangs 100 mm above the plate.
    depth_start = geom.PEG_LENGTH - geom.HOME_STANDOFF_Z
    depths = [
        depth_start + i * args_cli.step
        for i in range(int(round((args_cli.depth_end - depth_start) / args_cli.step)) + 1)
    ]
    print(f"[passability] {len(depths)} waypoints from {depths[0]:+.4f} m to {depths[-1]:+.4f} m "
          f"in {args_cli.step*1000:.1f} mm steps")

    # The negative control stops just past the entrance. It only has to show
    # that the collider reacts; driving a deliberately misaligned peg 27 mm into
    # solid plate would ask PhysX to resolve a penetration it was never meant to
    # see, and the resulting depenetration impulses say nothing about the bore.
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

        # Joint count from a buffer that is certain to exist, rather than from
        # robot.num_joints, which probe_assets.py never covered.
        q_art = torch.zeros_like(robot.data.default_joint_pos[:1])
        for i, art_i in enumerate(dh_to_art):
            q_art[0, art_i] = float(q_dh[i])
        zero_vel = torch.zeros_like(q_art)

        # Hold: rewrite the state every step so the actuators settle onto the
        # commanded configuration without being allowed to drift.
        for _ in range(args_cli.hold_steps):
            step_sim(env, robot, q_art, zero_vel)
        # Free: nothing is written. Any motion now comes from physics, and at
        # these depths the only physical cause available is depenetration.
        step_sim(env, robot, None, None)

        # Wrapped to +-pi: two joints run near 165 deg at the deepest waypoints,
        # and an unwrapped difference across the +-pi seam would read as a 2 pi
        # spike, i.e. a spurious BLOCKED verdict.
        delta = robot.data.joint_pos[0] - q_art[0]
        deviation = float(torch.max(torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta)))).item())
        if math.isnan(deviation) or math.isinf(deviation):
            # A solver blow-up is itself a reaction, but NaN propagates silently
            # through max() comparisons and would corrupt the verdict rather
            # than trip it. Mapped to a value that reads as "reacted violently".
            print(f"[passability] non-finite deviation at depth {depth:+.4f} m; recorded as a hard reaction")
            deviation = 1e3  # finite on purpose: json.dumps writes inf as the non-standard Infinity
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
    """One physics step, optionally writing the joint state first.

    Every Isaac Lab call here is guarded. A missing method on this version must
    degrade to a warning: on the training machine an exception costs a whole
    copy-and-run round trip and reveals less than a printed name would.
    """
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
    """Peg linear speed, or None if this Isaac Lab version does not expose it.

    ``body_lin_vel_w`` was never covered by probe_assets.py -- D-020 addendum 2
    confirmed only ``body_pos_w`` and ``body_quat_w``. It is a diagnostic here,
    not part of the verdict, so an absent attribute must not end the run: losing
    a 256-waypoint measurement over a secondary number would cost a full
    copy-and-run round trip.
    """
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
    # Third column of the rotation matrix: the body z-axis in world coordinates.
    axis_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return [float(pos[i]) + geom.PEG_LENGTH * axis_z[i] for i in range(3)]


# How far above its own contact-free maximum a pass must rise before the excess
# counts as contact. The first run on the training machine showed why this must
# not be a number picked in advance: with a flat 5x, the negative control
# reached 4.78x and was declared inert, although the centred pass sat at 1.05x.
# The separation in the data is a factor of 4.5; any cut between roughly 1.1 and
# 4.7 yields the same verdict, so the script now reports that stable interval
# and the reader can see the answer does not hinge on this constant.
REACTION_FACTOR = 2.0


def judge(result: dict) -> dict:
    """Turn the two passes into a verdict by paired comparison.

    Each pass is calibrated against **its own** contact-free waypoints, and the
    two are then compared at matched depths. That pairing is what makes the
    result an argument rather than a threshold: the centred and offset runs
    differ by 4 mm of lateral position and nothing else, so identical gravity
    and nearly identical arm configurations cancel, and what remains is contact.
    """
    centred = result["centred"]["waypoints"]
    offset = result["offset"]["waypoints"]
    if not centred or not offset:
        return {"verdict": "INCONCLUSIVE", "reason": "a pass produced no reachable waypoints"}

    def react(w):
        return w["free_step_joint_deviation_rad"]

    def contact_free_max(waypoints) -> float | None:
        # At least 5 mm clear of the plate, where contact is geometrically
        # impossible: pure gravity and solver noise. Taken per pass, because a
        # laterally shifted arm has a slightly different gravity torque and must
        # not be judged against the other pass's noise.
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

    # Where the control first leaves its own quiet range. Physically this should
    # be the plate top surface, i.e. within about a millimetre of depth 0; a
    # reaction that only starts much deeper would mean something other than the
    # plate is being hit.
    first_reaction = next(
        (
            w["commanded_depth_m"]
            for w in sorted(offset, key=lambda x: x["commanded_depth_m"])
            if w["commanded_depth_m"] >= -0.005 and react(w) > REACTION_FACTOR * offset_base
        ),
        None,
    )

    # Contiguous, not the maximum over all clean waypoints. Because each
    # waypoint is teleported to independently, a single quiet reading deeper
    # than an obstruction would otherwise report the bore as reached. The peg
    # can only be said to have gone in as far as the last depth it reached
    # without anything before it reacting.
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
        # None rather than -inf: json.dumps would emit the non-standard literal
        # -Infinity, which a strict reader on the other machine rejects.
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
