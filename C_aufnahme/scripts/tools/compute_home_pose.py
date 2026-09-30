# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Compute the UR home-pose joint angles for the peg-insertion scene, offline.

Closed-form UR inverse kinematics, numpy only — no Isaac Lab import, no
simulator. Runs on the dev PC and on the training machine (Python 3.11
compatible). Prints a table; writes nothing. The selected joint values are
transcribed by hand into the robot cfg (``ur10e_cfg.py`` when this was
written; that file was deleted on 2026-08-28, S6), so the home pose stays a
reviewable
constant rather than a runtime result, and the environment carries no IK
dependency.

Target (D-022/D-023, env-local): the robot base sits at (0, +0.190, 0.755), the
bore entrance at (0, -0.225, 0.755), i.e. 0.415 m along -Y. The flange is
commanded 0.150 m above the plate top, tool axis pointing straight down. In the
base frame that is position (0, -0.415, 0.150) and orientation diag(1, -1, -1),
the same 180-deg-about-X convention D-022 records for the task frame — so the
later insertion controller sees zero orientation error at the home pose.

What this settles for D-023: ``|sin q5|`` is the wrist-singularity margin and
the shoulder radius / d4 ratio is the shoulder margin. Both are printed per
branch. D-023's open caveat is about exactly this configuration.

CONVENTION HAZARD, not resolvable from here: the UR DH base frame and the
URDF/USD ``base_link`` frame differ by a rotation of pi about z. Rotating the
base frame about z by pi shifts q1 by pi and leaves q2..q6 unchanged, so both
q1 families are printed. Pick one, read the environment startup report, and if
the tool points down correctly but the XY offset is mirrored, switch families.

Usage:

    python scripts/tools/compute_home_pose.py --robot ur10e

Standard Denavit-Hartenberg convention throughout:

    T_i = [[ cq, -sq*ca,  sq*sa, a*cq],
           [ sq,  cq*ca, -cq*sa, a*sq],
           [  0,     sa,     ca,    d],
           [  0,      0,      0,    1]]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

# Published Universal Robots DH parameters.
#
# The link constants below are now CHECKED against the shipped assets, not just
# internally consistent: scripts/probe_assets.py printed each revolute joint's
# localPos0 in its parent link frame (training machine, Isaac Sim 5.1,
# 2026-08-18), and every non-zero |a| and d matches its DH entry exactly. That
# retires the old "UNVERIFIED against the actual USD" caveat for the link
# LENGTHS.
#
#   ur5e   pan (0,0,0.1625) -> d1 | elbow (-0.425,0,0) -> a2
#          wrist_1 (-0.3922,0,0.1333) -> a3, d4
#          wrist_2 (0,-0.0997,0) -> d5 | wrist_3 (0,0.0996,0) -> d6
#   ur10e  0.1807 | -0.6127 | -0.57155, 0.17415 | -0.11985 | 0.11655
#
# What is still NOT settled by that check: the ZERO of each joint and the sense
# of rotation. The USD joint frames carry quaternions (e.g. localRot0 =
# (-4.371e-8,0,0,1) on shoulder_pan, a pi rotation about z), so a solution from
# this script can be kinematically right and still land mirrored or offset in
# the simulator. That is exactly the trap D-025 fell into on the UR10e, where
# the flange ended up at +y instead of -y. Confirm against the environment
# startup report before trusting any pose from here.
DH_TABLES = {
    # a, d, alpha  per joint 1..6
    "ur5e": {
        "a": [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0],
        "d": [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996],
        "alpha": [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0],
    },
    "ur10e": {
        "a": [0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0],
        "d": [0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655],
        "alpha": [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0],
    },
    "ur10": {
        "a": [0.0, -0.612, -0.5723, 0.0, 0.0, 0.0],
        "d": [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
        "alpha": [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0],
    },
}

# D-022 / D-023, env-local.
#
# RESTATED, not imported. One home for these three numbers is
# insertion_tasks_cfg.py: ROBOT_BASE_OFFSET_Y (0.190), OPENING_OFFSET_Y
# (-0.225) and PLATE_TOP_Z (0.755, = FIXTURE_POS[2]). That module imports
# isaaclab.utils.configclass, so asking it costs a SimulationApp and this
# tool would stop running on the laptop -- the same reason insertion_paths.py
# was split out after RT-88. Change the value there and here together.
ROBOT_BASE_POS = (0.0, 0.190, 0.755)
BORE_ENTRANCE_POS = (0.0, -0.225, 0.755)
DEFAULT_STANDOFF = 0.150

# Same restatement, same reason: the owning list is UR5E_HOME_JOINT_POS in
# source/insertion/insertion/tasks/direct/insertion/ur5e_cfg.py, whose keys
# are in this order.
JOINT_NAMES_DH_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def dh_transform(q: float, a: float, d: float, alpha: float) -> np.ndarray:
    cq, sq = math.cos(q), math.sin(q)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [cq, -sq * ca, sq * sa, a * cq],
            [sq, cq * ca, -cq * sa, a * sq],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def forward_kinematics(q: np.ndarray, dh: dict) -> np.ndarray:
    """T06 from the six joint angles. Implemented independently of the IK."""
    t = np.eye(4)
    for i in range(6):
        t = t @ dh_transform(float(q[i]), dh["a"][i], dh["d"][i], dh["alpha"][i])
    return t


def inverse_kinematics(target: np.ndarray, dh: dict) -> list[np.ndarray]:
    """All real branches of the closed-form UR IK for a given T06.

    Returns up to 8 solutions: 2 (shoulder) x 2 (wrist flip) x 2 (elbow).
    """
    d1, d4, d5, d6 = dh["d"][0], dh["d"][3], dh["d"][4], dh["d"][5]
    a2, a3 = dh["a"][1], dh["a"][2]

    solutions: list[np.ndarray] = []

    # --- q1 --------------------------------------------------------------
    # O5 lies at a fixed offset d4 along z1 from the shoulder, which gives
    # px5*sin(q1) - py5*cos(q1) = d4.
    p05 = target @ np.array([0.0, 0.0, -d6, 1.0])
    radius = math.hypot(p05[0], p05[1])
    if radius < abs(d4):
        return []  # inside the shoulder-singularity cylinder: no real solution
    phi = math.atan2(p05[1], p05[0])
    psi = math.asin(d4 / radius)
    for q1 in (phi + psi, phi + math.pi - psi):
        # --- q5 ----------------------------------------------------------
        # (P06 - O1).z1 = d4 + d6*cos(q5)
        cos_q5 = (target[0, 3] * math.sin(q1) - target[1, 3] * math.cos(q1) - d4) / d6
        if abs(cos_q5) > 1.0 + 1e-12:
            continue
        cos_q5 = max(-1.0, min(1.0, cos_q5))
        for q5 in (math.acos(cos_q5), -math.acos(cos_q5)):
            sin_q5 = math.sin(q5)
            # --- q6 ------------------------------------------------------
            # z1 expressed in frame 6 is (s5*c6, -s5*s6, c5).
            z1 = np.array([math.sin(q1), -math.cos(q1), 0.0])
            if abs(sin_q5) < 1e-9:
                q6 = 0.0  # wrist singularity: q4 and q6 are coupled, pick one
            else:
                z1_x6 = float(z1 @ target[:3, 0])
                z1_y6 = float(z1 @ target[:3, 1])
                q6 = math.atan2(-z1_y6 / sin_q5, z1_x6 / sin_q5)

            # --- q3, q2, q4 ----------------------------------------------
            t01 = dh_transform(q1, dh["a"][0], d1, dh["alpha"][0])
            t45 = dh_transform(q5, dh["a"][4], d5, dh["alpha"][4])
            t56 = dh_transform(q6, dh["a"][5], d6, dh["alpha"][5])
            t14 = np.linalg.inv(t01) @ target @ np.linalg.inv(t45 @ t56)

            px, py = t14[0, 3], t14[1, 3]
            planar_sq = px * px + py * py
            cos_q3 = (planar_sq - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
            if abs(cos_q3) > 1.0 + 1e-12:
                continue
            cos_q3 = max(-1.0, min(1.0, cos_q3))
            for q3 in (math.acos(cos_q3), -math.acos(cos_q3)):
                q2 = math.atan2(py, px) - math.atan2(
                    a3 * math.sin(q3), a2 + a3 * math.cos(q3)
                )
                q234 = math.atan2(t14[1, 0], t14[0, 0])
                q4 = q234 - q2 - q3
                solutions.append(np.array([q1, q2, q3, q4, q5, q6]))
    return solutions


def wrap_to_pi(q: np.ndarray) -> np.ndarray:
    return (q + math.pi) % (2.0 * math.pi) - math.pi


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    pos = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rot = float(np.linalg.norm(a[:3, :3] - b[:3, :3]))
    return pos, rot


def self_check(dh: dict, name: str) -> None:
    """FK/IK round-trips. Assertions, not printed numbers to be skimmed past."""
    print(f"--- self-check ({name}) ---")

    rng = np.random.default_rng(20260726)
    checked = 0
    for _ in range(200):
        q_ref = rng.uniform(-math.pi, math.pi, size=6)
        t_ref = forward_kinematics(q_ref, dh)
        sols = inverse_kinematics(t_ref, dh)
        if not sols:
            continue
        # Every branch must reproduce the pose it was solved for.
        for q in sols:
            pos_err, rot_err = pose_error(forward_kinematics(q, dh), t_ref)
            assert pos_err < 1e-9, f"FK/IK position error {pos_err:.3e}"
            assert rot_err < 1e-9, f"FK/IK rotation error {rot_err:.3e}"
        # The reference configuration itself must appear among the branches.
        best = min(float(np.linalg.norm(wrap_to_pi(q - q_ref))) for q in sols)
        assert best < 1e-6, f"reference configuration not recovered (closest {best:.3e})"
        checked += 1

    assert checked > 150, f"only {checked} round-trips exercised"
    print(f"  {checked} random FK->IK->FK round-trips passed (tol 1e-9)")


def main() -> None:
    parser = argparse.ArgumentParser(description="UR home-pose IK for the insertion scene.")
    parser.add_argument(
        "--robot",
        choices=sorted(DH_TABLES),
        default="ur10e",
        help="DH table to use. Select per the asset the probe actually found.",
    )
    parser.add_argument(
        "--standoff",
        type=float,
        default=DEFAULT_STANDOFF,
        help="Flange height above the plate top surface [m].",
    )
    args = parser.parse_args()

    dh = DH_TABLES[args.robot]
    self_check(dh, args.robot)

    # Target in the robot base frame.
    dx = BORE_ENTRANCE_POS[0] - ROBOT_BASE_POS[0]
    dy = BORE_ENTRANCE_POS[1] - ROBOT_BASE_POS[1]
    target = np.eye(4)
    target[:3, :3] = np.diag([1.0, -1.0, -1.0])  # 180 deg about X: tool axis down
    target[:3, 3] = [dx, dy, args.standoff]

    print()
    print(f"--- target ({args.robot}) ---")
    print(f"  base (env-local):  {ROBOT_BASE_POS}")
    print(f"  bore (env-local):  {BORE_ENTRANCE_POS}")
    print(f"  flange in base frame: ({dx:.4f}, {dy:.4f}, {args.standoff:.4f}) m")
    print("  orientation: diag(1, -1, -1), tool axis along -Z (D-022 convention)")

    d4 = dh["d"][3]
    d6 = dh["d"][5]
    p05 = target @ np.array([0.0, 0.0, -d6, 1.0])
    radius = math.hypot(p05[0], p05[1])
    print(f"  shoulder margin: radius {radius:.4f} m / d4 {d4:.5f} m = {radius / d4:.2f}")

    solutions = inverse_kinematics(target, dh)
    if not solutions:
        print("\nNo real IK solution for this target. Reconsider standoff or base placement.")
        raise SystemExit(1)

    print()
    print(f"--- branches ({len(solutions)}) ---")
    header = (
        f"{'#':>2}  {'q1':>8} {'q2':>8} {'q3':>8} {'q4':>8} {'q5':>8} {'q6':>8}"
        f"  {'|sin q5|':>9} {'|sin q3|':>9}  {'in +-180':>8}  elbow branch"
    )
    print(header)
    print("-" * len(header))

    ranked = []
    for idx, q_raw in enumerate(solutions):
        q = wrap_to_pi(q_raw)
        pos_err, rot_err = pose_error(forward_kinematics(q, dh), target)
        assert pos_err < 1e-9 and rot_err < 1e-9, f"branch {idx} fails FK check"

        wrist_margin = abs(math.sin(q[4]))
        elbow_margin = abs(math.sin(q[2]))
        within = bool(np.all(np.abs(q) <= math.pi + 1e-9))
        # Label only, not a verified claim: which q3 sign corresponds to
        # "elbow up" depends on the asset's joint sign convention, which the
        # startup report settles. The selection rule prefers q3 < 0 so the two
        # mirror branches are broken deterministically, not because one is
        # known to be the raised elbow.
        elbow = "q3<0" if q[2] < 0.0 else "q3>0"
        deg = np.degrees(q)
        print(
            f"{idx:>2}  "
            + " ".join(f"{v:8.2f}" for v in deg)
            + f"  {wrist_margin:9.4f} {elbow_margin:9.4f}  {str(within):>8}  {elbow}"
        )
        ranked.append((within, q[2] < 0.0, wrist_margin, -float(np.linalg.norm(q)), idx, q))

    # Selection rule: within +-pi, q3 < 0, maximise |sin q5|, then smallest ||q||.
    ranked.sort(reverse=True)
    within, elbow_neg, wrist_margin, _, idx, q = ranked[0]

    print()
    print("--- selected ---")
    print(f"  branch {idx}  (within +-180: {within}, q3 < 0: {elbow_neg})")
    print(f"  wrist-singularity margin |sin q5| = {wrist_margin:.4f}")
    if wrist_margin < 0.3:
        print("  WARNING: below 0.3 — this is a real finding for D-023.")
        print("  Reconsider the standoff height or the base placement before proceeding.")
    else:
        print("  -> comfortably clear of a wrist singularity (D-023 caveat answered).")

    for label, offset in (("base frame as given", 0.0), ("base frame rotated pi about z", math.pi)):
        shifted = wrap_to_pi(q + np.array([offset, 0.0, 0.0, 0.0, 0.0, 0.0]))
        print()
        print(f"  joint_pos, {label}:")
        print("  {")
        for name, value in zip(JOINT_NAMES_DH_ORDER, shifted):
            print(f'      "{name}": {value:+.6f},   # {math.degrees(value):+8.2f} deg')
        print("  }")

    print()
    print("  Only q1 differs between the two: redefining the base yaw leaves q2..q6")
    print("  untouched. Load one, read the startup report, switch if XY is mirrored.")


if __name__ == "__main__":
    main()
