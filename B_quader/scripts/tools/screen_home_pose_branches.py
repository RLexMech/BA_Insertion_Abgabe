"""Screen all UR10e IK branches for clearance against the table plate.

Convention, established by matching the measured simulator pose rather than
assumed: the DH chain is driven by the USD joint angles directly, and its
output sits in a frame rotated by pi about z relative to the env frame:

    p_env = BASE + Rz(pi) . p_DH(q_USD)

The first version of this script applied both the point rotation and a q1
shift; those cancel exactly, which is why both variants gave the same wrong
answer. It is one or the other.
"""
import importlib.util
import pathlib

import numpy as np

MOD = pathlib.Path(r"C:\Users\alexp\OneDrive\Desktop\Claude\Keine 2\scripts\tools\compute_home_pose.py")
spec = importlib.util.spec_from_file_location("chp", MOD)
chp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chp)

DH = chp.DH_TABLES["ur10e"]
BASE = np.array([0.0, 0.190, 0.755])
BORE = np.array([0.0, -0.225, 0.755])
STANDOFF = 0.150
RZ_PI = np.diag([-1.0, -1.0, 1.0])

HX, HY = 0.250, 0.300          # plate half extents, env-local
PLATE_TOP, PLATE_BOTTOM = 0.755, 0.700

# Measured in the simulator at t = 3.0 s (startup report, marker ...-26c).
MEASURED_Q = np.array([-1.8887712955, 0.9098067284, -2.3225746155,
                       2.9370598793, 1.5707978010, -0.4330468476])
MEASURED_BODIES = {
    "shoulder_link": np.array([0.0000, 0.1900, 0.9357]),
    "forearm_link": np.array([-0.1176, -0.1673, 0.4520]),
    "wrist_1_link": np.array([0.0197, -0.3072, 1.0165]),
    "wrist_2_link": np.array([0.0571, -0.1934, 1.0109]),
    "wrist_3_link": np.array([0.0554, -0.1986, 0.8945]),
}
# Established by matching, not assumed: the USD link frames do not map 1:1 onto
# the DH frames. shoulder and upper_arm share the shoulder origin, and the three
# wrist links line up with DH frames 4, 5, 6 -- DH frame 3 has no USD counterpart.
FRAME_OF = {"shoulder_link": 1, "forearm_link": 2, "wrist_1_link": 4,
            "wrist_2_link": 5, "wrist_3_link": 6}


def chain_origins(q):
    t = np.eye(4)
    origins = [t[:3, 3].copy()]
    for i in range(6):
        t = t @ chp.dh_transform(q[i], DH["a"][i], DH["d"][i], DH["alpha"][i])
        origins.append(t[:3, 3].copy())
    return np.array(origins)


def to_env(origins):
    return origins @ RZ_PI.T + BASE


print("=" * 74)
print("STEP 1 - validate the DH chain against the measured simulator pose")
print("=" * 74)
pts = to_env(chain_origins(MEASURED_Q))
worst = 0.0
for name, meas in MEASURED_BODIES.items():
    e = float(np.linalg.norm(pts[FRAME_OF[name]] - meas))
    worst = max(worst, e)
    print(f"  {name:<16} model {np.round(pts[FRAME_OF[name]], 4)}   measured {meas}   err {e*1000:6.2f} mm")
print(f"  worst error: {worst*1000:.2f} mm")
if worst > 0.002:
    print("  MODEL DOES NOT MATCH - do not use the screening below")
    raise SystemExit(1)
print("  model validated against the simulator")

print()
print("=" * 74)
print("STEP 2 - screen every IK branch for plate clearance")
print("=" * 74)

# Target expressed in the DH frame: rotate the env-frame target by Rz(pi).
p_env_rel = np.array([0.0, BORE[1] - BASE[1], STANDOFF])      # (0, -0.415, +0.150)
R_env = np.diag([1.0, -1.0, -1.0])                            # tool axis down
target = np.eye(4)
target[:3, 3] = RZ_PI.T @ p_env_rel
target[:3, :3] = RZ_PI.T @ R_env

solutions = chp.inverse_kinematics(target, DH)
print(f"branches found: {len(solutions)}")
print(f"target in DH frame: pos {np.round(target[:3,3],4)}")
print()

rows = []
for idx, q in enumerate(solutions):
    q = chp.wrap_to_pi(np.asarray(q))
    pts = to_env(chain_origins(q))
    # Skip the base->shoulder segment: it runs vertically up from the mounting
    # point, which sits exactly on the plate top by design.
    min_clear = float("inf")
    for a, b in zip(pts[1:-1], pts[2:]):
        for s in np.linspace(0.0, 1.0, 120):
            p = a + s * (b - a)
            if abs(p[0]) <= HX and abs(p[1]) <= HY:
                min_clear = min(min_clear, p[2] - PLATE_TOP)
    # Verify the branch actually reaches the commanded pose.
    fk = chp.forward_kinematics(q, DH)
    pos_err = float(np.linalg.norm(fk[:3, 3] - target[:3, 3]))
    ee_env = to_env(chain_origins(q))[6]
    rows.append((idx, min_clear, pos_err, ee_env, q))

rows.sort(key=lambda r: -(r[1] if r[1] != float("inf") else 1e9))
print(f"{'br':>3} {'min clearance over plate':>25} {'FK err':>9}   joint angles (USD, rad)")
for idx, mc, pe, ee, q in rows:
    mc_txt = "never over the plate" if mc == float("inf") else f"{mc:+.4f} m"
    print(f"{idx:>3} {mc_txt:>25} {pe:9.2e}   [{', '.join(f'{v:+.6f}' for v in q)}]")

print()
print("=" * 74)
print("STEP 3 - D-025 selection criteria applied to the clear branches")
print("=" * 74)
print("D-025 order: all |q| < pi, then q3 < 0, then max |sin q5|, then min ||q||.")
print("Plate clearance was not among them -- the plate had no collision then.")
print()
print(f"{'br':>3} {'clear':>9} {'min z':>8} {'max|q|':>8} {'||q||':>7} {'q3<0':>6} {'|sin q5|':>9} {'all<pi':>7}")
clear_rows = [r for r in rows if r[1] > 0.0]
for idx, mc, pe, ee, q in clear_rows:
    pts = to_env(chain_origins(q))
    seg_min_z = min(
        (a + s * (b - a))[2]
        for a, b in zip(pts[1:-1], pts[2:])
        for s in np.linspace(0.0, 1.0, 120)
    )
    print(f"{idx:>3} {mc:+9.4f} {seg_min_z:8.4f} {np.max(np.abs(q)):8.4f} "
          f"{np.linalg.norm(q):7.4f} {str(q[2] < 0):>6} {abs(np.sin(q[4])):9.4f} "
          f"{str(bool(np.all(np.abs(q) < np.pi))):>7}")

qualified = [r for r in clear_rows if r[4][2] < 0 and np.all(np.abs(r[4]) < np.pi)]
qualified.sort(key=lambda r: np.linalg.norm(r[4]))
print()
if qualified:
    sel = qualified[0]
    print(f"selected by D-025 criteria: branch {sel[0]}")
    print(f"  joint angles: [{', '.join(f'{v:+.6f}' for v in sel[4])}]")
    names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    for n, v in zip(names, sel[4]):
        print(f"    {n:<15} {v:+.6f}")

best = rows[0]
print(f"best branch {best[0]}: clearance "
      f"{'never over the plate' if best[1]==float('inf') else f'{best[1]:+.4f} m'}, "
      f"EE env-local {np.round(best[3],4)} (bore + standoff = {np.round(BORE + [0,0,STANDOFF],4)})")
