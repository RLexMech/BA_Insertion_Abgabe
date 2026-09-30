"""Translate the env's per-joint reset noise (radians) into millimetres and degrees at the peg tip.

The cfg knobs are joint-space, because randomising there needs no IK at
reset. That makes them cheap but opaque: 0.01 rad on five joints plus
+-45 deg on wrist_3 says nothing about how far the peg actually starts from
the pocket, laterally and in yaw -- which are the numbers that matter
against 1.0 mm of per-axis clearance and a +-3.96 deg free-yaw window, and
the ones a thesis has to report.

This runs the same closed-form UR forward kinematics that
``scripts/tools/compute_home_pose.py`` verified on the dev PC (200 random
FK -> IK -> FK round trips at 1e-9, and the DH table reproduced from the USD
joint offsets), applies uniform noise to the six home-pose joint angles, and
reports the resulting spread of the peg tip. numpy only -- no Isaac, no
PyTorch, no simulation.

Two caveats, both real:

* This is the spread the *commanded* pose has at reset. The arm then droops
  under gravity by about 13.4 mm (D-026), which is a systematic offset on
  top of this scatter, not part of it.
* Tip position is the flange pose advanced 50 mm along the tool axis, which
  the training-machine report measured as exact (``tip along tool axis
  +0.050000 m``). Joint noise therefore tilts the peg as well as moving it,
  and the angular column reports that tilt.

    python scripts/reset_noise_spread.py
    python scripts/reset_noise_spread.py --noise 0.02 0.05 --yaw-noise 0.7854 --samples 20000
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
HOME_POSE_TOOL = HERE / "tools" / "compute_home_pose.py"

# Home pose, branch 3, from source/.../ur10e_cfg.py. Repeated here rather than
# imported because that module pulls in isaaclab, which the dev PC lacks.
HOME_Q = np.array([2.003843, -1.911842, -2.265118, 2.606164, -1.570796, 0.433047])
PEG_LENGTH = 0.050
# Env-local pocket entrance and robot base (D-022/D-023), to express the tip
# in the same frame the startup report and the reward use. The square
# re-import moved the asset origin to this very point, so entrance ==
# fixture position now.
ENTRANCE = np.array([0.0, -0.225, 0.755])
ROBOT_BASE = np.array([0.0, 0.190, 0.755])
# Free-yaw window of the 30 mm peg in the 32 mm pocket (D-029): the p95 yaw
# column below is read against this.
YAW_WINDOW_DEG = 3.96
# wrist_3, the tool-axis yaw joint, is the last entry of HOME_Q.
YAW_JOINT = 5


def load_fk():
    spec = importlib.util.spec_from_file_location("compute_home_pose", HOME_POSE_TOOL)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {HOME_POSE_TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["compute_home_pose"] = module
    spec.loader.exec_module(module)
    return module.forward_kinematics, module.DH_TABLES


def tip_positions(q: np.ndarray, fk, dh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Peg tip position, tool axis and tool x-axis for a batch of joint configurations."""
    tips = np.empty((q.shape[0], 3))
    axes = np.empty((q.shape[0], 3))
    xaxes = np.empty((q.shape[0], 3))
    for i in range(q.shape[0]):
        T = fk(q[i], dh)
        tool_axis = T[:3, 2]
        tips[i] = T[:3, 3] + PEG_LENGTH * tool_axis
        axes[i] = tool_axis
        xaxes[i] = T[:3, 0]
    return tips, axes, xaxes


def folded_yaw_deg(xaxes: np.ndarray, nominal_x: np.ndarray) -> np.ndarray:
    """Yaw of each tool x-axis about world z, relative to the nominal one,
    folded into (-45, 45] deg -- the same C4 fold _peg_geometry applies."""
    phi = np.arctan2(xaxes[:, 1], xaxes[:, 0]) - np.arctan2(nominal_x[1], nominal_x[0])
    phi = phi - np.round(phi / (np.pi / 2.0)) * (np.pi / 2.0)
    return np.degrees(np.abs(phi))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.10],
                        help="reset_joint_noise values for the five non-yaw joints [rad].")
    parser.add_argument("--yaw-noise", type=float, default=0.7854,
                        help="reset_yaw_noise on wrist_3 [rad]; the cfg default is +-45 deg.")
    parser.add_argument("--samples", type=int, default=20000, help="Samples per noise level.")
    parser.add_argument("--table", default="ur10e", help="DH table name in compute_home_pose.py.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fk, tables = load_fk()
    if args.table not in tables:
        raise SystemExit(f"unknown DH table {args.table!r}; have {sorted(tables)}")
    dh = tables[args.table]

    nominal_tip, nominal_axis, nominal_x = tip_positions(HOME_Q.reshape(1, 6), fk, dh)
    nominal_tip, nominal_axis, nominal_x = nominal_tip[0], nominal_axis[0], nominal_x[0]
    # The DH base frame is rotated by pi about z against the USD base_link
    # frame -- the caveat D-025 recorded and HANDOFF still lists as open. It
    # is confirmed here: without the rotation the tip lands at y = +0.415
    # relative to the base, mirrored through the base, and the offset to the
    # bore comes out 830 mm instead of 0. Applying it reproduces the
    # training-machine report exactly (0.000 mm lateral, 100 mm above the
    # plate). Lengths are unaffected, so every spread below is valid either
    # way; only this reference line depends on it.
    tip_env = ROBOT_BASE + np.array([-nominal_tip[0], -nominal_tip[1], nominal_tip[2]])
    print(f"home pose tip, base frame : {np.round(nominal_tip, 6).tolist()} m")
    print(f"home pose tip, env-local  : {np.round(tip_env, 6).tolist()} m")
    print(f"  lateral offset to pocket: {np.linalg.norm((tip_env - ENTRANCE)[:2]) * 1000:.3f} mm")
    print(f"  height above plate      : {(tip_env - ENTRANCE)[2] * 1000:.1f} mm")
    print(f"  tool axis . (0,0,-1)    : {-nominal_axis[2]:+.6f}")
    print()

    rng = np.random.default_rng(args.seed)
    yaw_noise = args.yaw_noise
    header = (f"{'noise [rad]':>11} {'+-deg':>7} {'lat mean':>9} {'lat p95':>8} {'lat max':>8} "
              f"{'vert p95':>9} {'tilt p95':>9} {'yaw p95':>8}")
    print(f"wrist_3 yaw noise: +-{yaw_noise:.4f} rad (+-{np.degrees(yaw_noise):.1f} deg) on every row\n")
    print(header)
    print("-" * len(header))
    for noise in args.noise:
        if noise == 0.0 and yaw_noise == 0.0:
            q = HOME_Q.reshape(1, 6)
        else:
            # Per-joint noise, mirroring _reset_idx: five joints at `noise`,
            # wrist_3 at `yaw_noise`.
            scale = np.full(6, noise)
            scale[YAW_JOINT] = yaw_noise
            q = HOME_Q + rng.uniform(-1.0, 1.0, size=(args.samples, 6)) * scale
        tips, axes, xaxes = tip_positions(q, fk, dh)
        lateral = np.linalg.norm(tips[:, :2] - nominal_tip[:2], axis=1) * 1000.0
        vertical = np.abs(tips[:, 2] - nominal_tip[2]) * 1000.0
        # Angle between the perturbed tool axis and the nominal one.
        cos = np.clip(axes @ nominal_axis, -1.0, 1.0)
        tilt = np.degrees(np.arccos(cos))
        yaw = folded_yaw_deg(xaxes, nominal_x)
        print(f"{noise:>11.3f} {np.degrees(noise):>7.2f} {lateral.mean():>9.2f} "
              f"{np.percentile(lateral, 95):>8.2f} {lateral.max():>8.2f} "
              f"{np.percentile(vertical, 95):>9.2f} {np.percentile(tilt, 95):>9.2f} "
              f"{np.percentile(yaw, 95):>8.2f}")

    print("\nColumns: lateral/vertical displacement of the peg tip from its nominal")
    print("position [mm], tool-axis tilt [deg], and |yaw| folded into the C4")
    print("fundamental domain [deg]. p95 = 95th percentile.")
    print(f"\nFor scale: 1.0 mm of clearance per axis; free-yaw window +-{YAW_WINDOW_DEG} deg.")
    print("A yaw p95 far above the window means most episodes must actually turn the")
    print("peg before inserting -- which is the point of the wide wrist_3 noise:")
    print("without it every start lies inside the window and success proves nothing.")
    print("Lateral spread far above the clearance likewise forces a search, not a")
    print("replayed trajectory. wrist_3 is a pure yaw actuator at home, so the yaw")
    print("noise should barely move the lateral columns between rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
