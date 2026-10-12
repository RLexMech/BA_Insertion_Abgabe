"""Screen every UR IK branch for clearance against the real workcell.

Convention, established by MATCHING measured simulator poses rather than
assumed: the DH chain is driven by the USD joint angles directly, and its
output sits in a frame rotated by pi about z relative to the env frame:

    p_env = BASE + Rz(pi) . p_DH(q_USD)

The first version applied both the point rotation and a q1 shift; those cancel
exactly, which is why both variants gave the same wrong answer. It is one or
the other.

Step 1 re-derives that convention per robot from a measured pose instead of
trusting it. For the UR5e the measurement comes from the workcell startup
report of 2026-08-18 (t = 3.0 s, all joint velocities zero, so the pose is
settled). The identity convention is carried as a NEGATIVE CONTROL: it is
wrong by up to a metre, which is what makes a 0.07 mm match meaningful rather
than a coincidence of a lenient tolerance.

Obstacles are the real cell (D-058): both table plates AND the fixture block,
each an axis-aligned box. The proxy version screened a single plate, which is
why it could not have caught an arm swinging through the lower table.

Since D-204 (2026-09-28) the "tool point is lowest" gate is read over the
BLOCK, from the block top, and must hold along the whole reset descent: every
tool height from the lowest legal start to the home, over every corner of the
fixture spread (step 2b). The reset IK stays inside the home's branch family.

Runs offline -- pure kinematics, no Isaac.

    python scripts\\tools\\screen_home_pose_branches.py
    python scripts\\tools\\screen_home_pose_branches.py --standoff 0.400 --fixture-spread 0.05
    python scripts\\tools\\screen_home_pose_branches.py --robot ur10e
"""
import argparse
import ast
import importlib.util
import pathlib
import sys
import types

import numpy as np

ap = argparse.ArgumentParser(description="Screen UR IK branches against the workcell.")
ap.add_argument("--robot", choices=["ur5e", "ur10e"], default="ur5e")
ap.add_argument("--standoff", type=float, default=None,
                help="Height of the LEADING TOOL POINT above the block top face [m]. Default: cfg HOME_STANDOFF_Z. The flange sits this much plus the tool chain length higher.")
ap.add_argument("--fixture-spread", type=float, default=None,
                help="Fixture displacement per axis [m] the descent sweep covers (step 2b), x, y AND z. Default: InsertionEnvCfg.fixture_pos_noise_xy for x/y and fixture_pos_noise_z for z, read from insertion_env_cfg.py by AST. Give it only for a what-if.")
args = ap.parse_args()

HERE = pathlib.Path(__file__).resolve().parent
MOD = HERE / "compute_home_pose.py"
if not MOD.is_file():
    raise SystemExit(f"compute_home_pose.py not found next to this script: {MOD}")
spec = importlib.util.spec_from_file_location("chp", MOD)
chp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chp)

# --- cell geometry, read from the ONE owner -------------------------------
CFG_PATH = (HERE.parent.parent / "source" / "insertion" / "insertion" / "tasks"
            / "direct" / "insertion" / "insertion_tasks_cfg.py")
if "isaaclab" not in sys.modules:                    # the dev laptop has no Isaac
    _pkg = types.ModuleType("isaaclab")
    _utils = types.ModuleType("isaaclab.utils")
    _utils.configclass = lambda c: c
    _pkg.utils = _utils
    sys.modules["isaaclab"] = _pkg
    sys.modules["isaaclab.utils"] = _utils
# A throwaway package whose __path__ is the real task directory: the cfg does
# `from .insertion_paths import ...` (a RELATIVE import, since the path block
# moved out on 2026-08-29), and exec'd into a bare module that raised
# "attempted relative import with no known parent package" -- this tool did
# not run from then until 2026-09-28. Same device as
# scripts/check_fixture_mesh.py load_cfg(); nothing on disk is touched.
_PKG_NAME = "_screen_cfg_pkg"
_shell = types.ModuleType(_PKG_NAME)
_shell.__path__ = [str(CFG_PATH.parent)]
sys.modules[_PKG_NAME] = _shell
geom = types.ModuleType(f"{_PKG_NAME}.insertion_tasks_cfg")
geom.__file__ = str(CFG_PATH)
geom.__package__ = _PKG_NAME
sys.modules[geom.__name__] = geom
# Compiled from source rather than imported: __pycache__ once let a checker in
# this repo certify numbers that were no longer in the file. See the docstring
# of load_cfg() in scripts/check_workcell_geometry.py.
exec(compile(CFG_PATH.read_text(encoding="utf-8"), str(CFG_PATH), "exec"), geom.__dict__)

# The fixture spread lives in the ENV cfg, which imports isaaclab for real, so
# it is read by AST from its one home instead of restated here.
ENV_CFG_PATH = CFG_PATH.parent / "insertion_env_cfg.py"
SPREAD_FIELDS = ("fixture_pos_noise_xy", "fixture_pos_noise_z")


def _env_cfg_literal(field):
    tree = ast.parse(ENV_CFG_PATH.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InsertionEnvCfg")
    for n in cls.body:
        tgt = n.targets[0] if isinstance(n, ast.Assign) else getattr(n, "target", None)
        if isinstance(tgt, ast.Name) and tgt.id == field and n.value is not None:
            return float(ast.literal_eval(n.value))
    raise SystemExit(f"InsertionEnvCfg.{field} not found as a literal in {ENV_CFG_PATH}")


if args.fixture_spread is None:
    SPREAD_XY, SPREAD_Z = (_env_cfg_literal(f) for f in SPREAD_FIELDS)
    FIXTURE_SPREAD_SRC = "InsertionEnvCfg." + " / ".join(SPREAD_FIELDS)
else:
    SPREAD_XY = SPREAD_Z = float(args.fixture_spread)
    FIXTURE_SPREAD_SRC = "--fixture-spread, a what-if"

DH = chp.DH_TABLES[args.robot]
RZ_PI = np.diag([-1.0, -1.0, 1.0])
FRAME_OF = {"shoulder_link": 1, "forearm_link": 2, "wrist_1_link": 4,
            "wrist_2_link": 5, "wrist_3_link": 6}

# Matched pairs: the ACTUAL joint angles together with the ACTUAL body
# positions the simulator reported for them. Not a commanded pose -- gravity
# droop makes commanded and actual differ, and only the actual pair is a valid
# forward-kinematics test.
#
# These angles were captured under the UR10e PLACEHOLDER gains, so they are not
# the angles a run reports today (RT-46, USD drives: max joint deviation
# 0.000029 rad, so commanded and actual now nearly coincide). That does not
# invalidate the test: it only needs the angles and the positions to come from
# the SAME run, which they do. Do not read this block as a current pose.
MEASURED = {
    # Workcell startup report, marker square-peg-2026-08-16-d038, step 180.
    "ur5e": {
        "base": np.zeros(3),
        "q": np.array([3.134709358215332, -1.5560652017593384, 1.5989718437194824,
                       -1.5566023588180542, -1.5667004585266113, 0.6878584623336792]),
        "bodies": {"shoulder_link": np.array([0.0000, -0.0000, 0.1625]),
                   "forearm_link": np.array([-0.0063, 0.0000, 0.5875]),
                   "wrist_1_link": np.array([-0.3990, -0.1306, 0.5706]),
                   "wrist_2_link": np.array([-0.4985, -0.1299, 0.5649]),
                   "wrist_3_link": np.array([-0.4929, -0.1303, 0.4655])},
    },
    # Proxy task, old repo, startup report marker ...-26c. Kept so the UR10e
    # result stays reproducible after the switch.
    "ur10e": {
        "base": np.array([0.0, 0.190, 0.755]),
        "q": np.array([-1.8887712955, 0.9098067284, -2.3225746155,
                       2.9370598793, 1.5707978010, -0.4330468476]),
        "bodies": {"shoulder_link": np.array([0.0000, 0.1900, 0.9357]),
                   "forearm_link": np.array([-0.1176, -0.1673, 0.4520]),
                   "wrist_1_link": np.array([0.0197, -0.3072, 1.0165]),
                   "wrist_2_link": np.array([0.0571, -0.1934, 1.0109]),
                   "wrist_3_link": np.array([0.0554, -0.1986, 0.8945])},
    },
}
BASE = MEASURED[args.robot]["base"]
# The standoff is quoted for the LEADING TOOL POINT, not the flange. For the
# ur5e that point is the part bottom, which hangs FLANGE_TO_PART_BOTTOM below
# the flange (D-069). The ur10e path is the legacy proxy robot and has no tool
# chain in this repo, so its length is zero and its behaviour is unchanged.
TOOL_CHAIN_LEN = geom.FLANGE_TO_PART_BOTTOM if args.robot == "ur5e" else 0.0
PART_STANDOFF = geom.HOME_STANDOFF_Z if args.standoff is None else args.standoff
FLANGE_STANDOFF = PART_STANDOFF + TOOL_CHAIN_LEN


def chain_origins(q):
    t = np.eye(4)
    out = [t[:3, 3].copy()]
    for i in range(6):
        t = t @ chp.dh_transform(q[i], DH["a"][i], DH["d"][i], DH["alpha"][i])
        out.append(t[:3, 3].copy())
    return np.array(out)


def to_env(origins, rot=RZ_PI):
    return origins @ rot.T + BASE


def obstacles(off=(0.0, 0.0, 0.0)):
    """Axis-aligned boxes as (x0, x1, y0, y1, top, name).

    ``off`` is a fixture displacement (D-203/D-204): the block and its rear
    wall ARE the kinematic fixture the env teleports per reset
    (``insertion_env.py`` ``_fixture.write_root_pose_to_sim``), so they move
    with it; the tables do not.
    """
    if args.robot != "ur5e":
        hx, hy = geom.PLATE_HALF_EXTENTS
        return [(-hx, hx, -hy, hy, geom.PLATE_TOP_Z, "proxy plate")]
    dx, dy, dz = off
    t1c, t2c = geom.TABLE1_POS, geom.TABLE2_POS
    # The block is read from its NEAR FACE outwards, not from its centre
    # (2026-08-24). Two reasons: the model is 225.0876 mm deep, not 210 (the
    # 15 mm rear wall stands past the raw block and has collision), and the
    # near face is the anchored end of the measured chain.
    bx0 = geom.BLOCK_NEAR_EDGE_X
    bx1 = bx0 + geom.BLOCK_MODEL_SIZE_X
    by0 = geom.BLOCK_CENTRE_Y - geom.BLOCK_SIZE_Y / 2
    by1 = geom.BLOCK_CENTRE_Y + geom.BLOCK_SIZE_Y / 2
    # The rear wall is a SEPARATE obstacle, and it has to be: this model is
    # 2.5D -- one flat top per box -- so folding a 100 mm wall into the block
    # would raise the whole block top and refuse every branch. Its footprint is
    # the cut-out's rear strip, converted from asset-local to env coordinates.
    # Its footprint is the whole 15 mm rear wall, from the stage-1 outline to
    # the block's far face -- NOT to the cut-out's rear face. Since 2026-08-24
    # (evening) the block stands BLOCK_BEHIND_POCKET_X proud of the cut-out.
    wx0 = geom.POCKET_ORIGIN_X + geom.POCKET_STAGE1_X_RANGE[1]
    wx1 = bx1
    return [
        (t1c[0] - geom.T1_SIZE_X / 2, t1c[0] + geom.T1_SIZE_X / 2,
         t1c[1] - geom.T1_SIZE_Y / 2, t1c[1] + geom.T1_SIZE_Y / 2,
         geom.T1_TOP_Z, "table 1"),
        (t2c[0] - geom.T2_SIZE_X / 2, t2c[0] + geom.T2_SIZE_X / 2,
         t2c[1] - geom.T2_SIZE_Y / 2, t2c[1] + geom.T2_SIZE_Y / 2,
         geom.T2_TOP_Z, "table 2"),
        (bx0 + dx, bx1 + dx, by0 + dy, by1 + dy, geom.BLOCK_TOP_Z + dz, "block"),
        (wx0 + dx, wx1 + dx, by0 + dy, by1 + dy,
         geom.BLOCK_TOP_Z + dz + geom.BLOCK_REAR_WALL_H, "rear wall"),
    ]


OBST = obstacles()
_S = np.linspace(0.0, 1.0, 120)


def arm_points(pts):
    """Sample points along every arm segment, (N, 3).

    The base->shoulder segment is skipped: it runs vertically out of the
    mounting point and is inside the table footprint by construction.
    """
    a, b = pts[1:-1], pts[2:]
    return (a[:, None, :] + _S[None, :, None] * (b - a)[:, None, :]).reshape(-1, 3)


def min_clearance(pts, obst=None):
    """Smallest (point z - obstacle top) over every arm segment: a COLLISION
    reading, each box against its own top."""
    p = arm_points(pts)
    worst, who = float("inf"), None
    for x0, x1, y0, y1, top, name in (OBST if obst is None else obst):
        inside = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
        if inside.any():
            c = float((p[inside, 2] - top).min())
            if c < worst:
                worst, who = c, name
    return worst, who


def block_clearance(pts, obst):
    """Lowest arm point over the BLOCK FOOTPRINT, measured from the BLOCK TOP.

    The plane matters (D-204 (2)). The gate compares this against the tool
    point's height, which is quoted above the block top. Until 2026-09-28 the
    gate took `min_clearance` over ALL boxes, each from its own top, and the
    minimum was the fixed shoulder origin, +0.1865 m over table 1 at every
    standoff (DH d1 0.1625 m + the 24 mm plate): above ~0.19 m every branch
    was rejected for a point that cannot move. The rear wall stands inside
    the block footprint and is read from the block top here too; its own
    collision is `min_clearance`'s job.
    """
    x0, x1, y0, y1, top, _ = next(o for o in obst if o[5] == "block")
    p = arm_points(pts)
    inside = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
    return float((p[inside, 2] - top).min()) if inside.any() else float("inf")


def family(q, target):
    """The closed-form branch a solution belongs to: shoulder, wrist flip
    (sign q5), elbow (sign q3) -- the three binary choices of
    `compute_home_pose.inverse_kinematics`. A descent over the pocket keeps
    the tool down and never straightens the elbow or crosses the shoulder
    cylinder, so it stays inside one family; the reset IK (dls, seeded from
    home) follows it continuously.

    The SHOULDER choice is not q1 itself: q1 moves when the fixture moves
    sideways. The IK takes q1 = phi + psi or phi + pi - psi, with phi the
    azimuth of the wrist centre O5 and psi = asin(d4/r) < pi/2, so the two
    lie on either side of phi + pi/2 and the sign of sin(q1 - phi - pi/2)
    names the choice for any target. Same two lines as the IK, restated."""
    p05 = target @ np.array([0.0, 0.0, -DH["d"][5], 1.0])
    phi = np.arctan2(p05[1], p05[0])
    return (int(np.sign(np.sin(q[0] - phi - np.pi / 2))),
            int(np.sign(q[4])), int(np.sign(q[2])))


print("=" * 78)
print(f"STEP 1 - validate the {args.robot} DH chain against the measured simulator pose")
print("=" * 78)
meas = MEASURED[args.robot]
scores = {}
for label, rot in (("Rz(pi)", RZ_PI), ("identity [NEGATIVE CONTROL]", np.eye(3))):
    p = to_env(chain_origins(meas["q"]), rot)
    scores[label] = max(float(np.linalg.norm(p[FRAME_OF[n]] - v))
                        for n, v in meas["bodies"].items())
    print(f"  {label:<28} worst error {scores[label]*1000:9.2f} mm")
pts = to_env(chain_origins(meas["q"]))
for name, val in meas["bodies"].items():
    err = float(np.linalg.norm(pts[FRAME_OF[name]] - val))
    print(f"    {name:<16} model {np.round(pts[FRAME_OF[name]], 4)}   "
          f"measured {np.round(val, 4)}   err {err*1000:6.2f} mm")
if scores["Rz(pi)"] > 0.002:
    raise SystemExit("  MODEL DOES NOT MATCH - do not use the screening below")
if scores["identity [NEGATIVE CONTROL]"] < 0.010:
    raise SystemExit("  NEGATIVE CONTROL FAILED - the test cannot tell the conventions apart")
print(f"  validated: Rz(pi) matches to {scores['Rz(pi)']*1000:.2f} mm while the control "
      f"is off by {scores['identity [NEGATIVE CONTROL]']*1000:.0f} mm")

print()
print("=" * 78)
print("STEP 2 - screen every IK branch against the workcell")
print("=" * 78)
if args.robot == "ur5e":
    # THE POCKET, not the block centre (2026-08-24). Until then this aimed at
    # the centre of a solid box, which was never a place the part had to go;
    # the two are 49.8 mm apart in x. The z is unchanged: the block top IS the
    # stage-1 rim (block_recess_boxes(): top == STAGE1_DEPTH above the origin).
    tgt_env = np.array([geom.POCKET_ORIGIN_X, geom.POCKET_ORIGIN_Y,
                        geom.BLOCK_TOP_Z + FLANGE_STANDOFF])
else:
    tgt_env = np.array([0.0, geom.OPENING_OFFSET_Y, geom.PLATE_TOP_Z + FLANGE_STANDOFF])
# Tool axis down, then spun about that axis by TOOL_HOME_YAW_RAD. The spin is
# POST-multiplied on purpose: post-multiplying rotates in the TOOL frame, which
# is the frame the angle is quoted in. Pre-multiplying would spin about world z
# and tilt nothing, which happens to look identical for a tool pointing exactly
# down and would stop being identical the moment the approach is tilted.
#
# The ur10e branch is the legacy proxy robot and carries no tool, so it keeps
# its zero rather than inheriting a number that means nothing for it.
TOOL_YAW = geom.TOOL_HOME_YAW_RAD if args.robot == "ur5e" else 0.0
_c, _s = np.cos(TOOL_YAW), np.sin(TOOL_YAW)
RZ_TOOL = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])
ROT_TARGET = RZ_PI.T @ (np.diag([1.0, -1.0, -1.0]) @ RZ_TOOL)
print(f"  tool yaw at home: {np.degrees(TOOL_YAW):+.1f} deg about the tool axis "
      f"({'the part''s lugs face the pocket features' if TOOL_YAW else 'convention zero'})")

for x0, x1, y0, y1, top, name in OBST:
    print(f"  obstacle {name:<9} x {x0:+.3f}..{x1:+.3f}  y {y0:+.3f}..{y1:+.3f}  top {top:+.4f}")
print(f"  flange target (env): {np.round(tgt_env, 4)}   "
      f"flange {FLANGE_STANDOFF:.3f} m above the block top")
print(f"  tool chain {TOOL_CHAIN_LEN:.4f} m  ->  leading tool point "
      f"{PART_STANDOFF:.3f} m above the block top")

# The gate is NOT merely "clearance > 0". A branch that clears the lower table
# by 55 mm while the tool hangs 150 mm up has some other part of the arm as its
# lowest point, and that part will be the first thing to touch when the policy
# moves. What the gate protects is therefore not "nothing collides" but "the
# thing that leads downwards is the thing you mean to insert".
#
# Written first as "the flange is the lowest point" (the proxy said it in prose
# but never encoded it, because with one plate flange and lowest point
# coincided). That wording died the moment a gripper and a part were welded
# under the flange: with a 152 mm chain the flange can never be lowest again,
# and all eight branches were rejected at the standoff the cell actually needs.
# The INTENT is unchanged -- only the reference point moves from the flange
# down to the leading tool point.
#
# D-204 (2), 2026-09-28: the comparison is made OVER THE BLOCK, from the block
# top (`block_clearance`) -- the plane the tool standoff is quoted in -- and
# every box, tables included, must still clear by > 0 (`min_clearance`, the
# proxy screening's rule). The ur10e legacy path keeps the old reading.
TOOL_IS_LOWEST = 1e-4


def solve_at(tool_above_top, off=(0.0, 0.0, 0.0)):
    """Every IK branch with the leading tool point `tool_above_top` above the
    (displaced) block top, over the (displaced) pocket. Rows:
    (idx, gate_clear, collide_clear, collide_who, fk_err, q, family, passes)."""
    obst = obstacles(off)
    if args.robot == "ur5e":
        tgt = np.array([geom.POCKET_ORIGIN_X + off[0], geom.POCKET_ORIGIN_Y + off[1],
                        geom.BLOCK_TOP_Z + off[2] + tool_above_top + TOOL_CHAIN_LEN])
    else:
        tgt = np.array([0.0, geom.OPENING_OFFSET_Y, geom.PLATE_TOP_Z + tool_above_top])
    target = np.eye(4)
    target[:3, 3] = RZ_PI.T @ (tgt - BASE)
    target[:3, :3] = ROT_TARGET
    out = []
    for idx, sol in enumerate(chp.inverse_kinematics(target, DH)):
        q = chp.wrap_to_pi(np.asarray(sol))
        pts = to_env(chain_origins(q))
        c_all, who = min_clearance(pts, obst)
        gate = block_clearance(pts, obst) if args.robot == "ur5e" else c_all
        fk = chp.forward_kinematics(q, DH)
        ok = c_all > 0.0 and gate >= tool_above_top - TOOL_IS_LOWEST
        out.append((idx, gate, c_all, who, float(np.linalg.norm(fk[:3, 3] - target[:3, 3])),
                    q, family(q, target), ok))
    return out


rows = solve_at(PART_STANDOFF)
print(f"  branches found: {len(rows)}")
print()
rows.sort(key=lambda r: -(r[1] if r[1] != float("inf") else 1e9))

print(f"{'br':>3} {'over block':>12} {'min clearance':>14} {'over':>9} {'FK err':>9}   "
      f"joint angles (USD, rad)")
for idx, gate, clear, who, ferr, q, fam, ok in rows:
    g = "never" if gate == float("inf") else f"{gate:+.4f} m"
    txt = "never over a box" if clear == float("inf") else f"{clear:+.4f} m"
    print(f"{idx:>3} {g:>12} {txt:>14} {str(who or '-'):>9} {ferr:9.2e}   "
          f"[{', '.join(f'{v:+.6f}' for v in q)}]")

print()
print("=" * 78)
print("STEP 2b - the gate along the whole reset descent (D-204 (2))")
print("=" * 78)
# THE RESET IK DESCENDS INSIDE THE HOME'S FAMILY (`family` above): a home whose
# family breaks the gate lower down leads the arm into exactly the pose the
# gate refuses as a home. Measured 2026-09-28: at 0.400 m D-025 alone picked a
# shoulder-flipped family that hangs 140 mm below the tool point, 9.8 mm above
# the block, at 0.150 m. So a family survives only if it passes the gate at
# EVERY start the env can ask for: tool heights from the lowest legal start
# (the env refuses a start at or below -depth_min = -SEATED_SUCCESS_DEPTH
# above the entrance; the entrance is STAGE1_DEPTH below the block top) up to
# the home, over every corner of the fixture spread (block and wall move with
# it). NOT swept, named: the lateral start offset and the pocket tilt/yaw
# (the target here is the pocket axis, tool down).
survivors = None
if args.robot == "ur5e":
    low = -geom.SEATED_SUCCESS_DEPTH - geom.STAGE1_DEPTH
    heights = list(np.arange(low, PART_STANDOFF, 0.010)) + [PART_STANDOFF]
    _xy = sorted({-SPREAD_XY, 0.0, SPREAD_XY})
    _z = sorted({-SPREAD_Z, 0.0, SPREAD_Z})
    corners = [(dx, dy, dz) for dx in _xy for dy in _xy for dz in _z]
    if SPREAD_XY == 0.0 and SPREAD_Z == 0.0:
        corners = [(0.0, 0.0, 0.0)]
    print(f"  heights {low:+.4f} .. {PART_STANDOFF:+.4f} m above the block top "
          f"({len(heights)} steps of 10 mm), fixture spread xy +-{SPREAD_XY:.3f} m, "
          f"z +-{SPREAD_Z:.3f} m "
          f"({FIXTURE_SPREAD_SRC}), {len(corners)} fixture poses")
    home_fams = {r[6] for r in rows}
    worst = {f: (float("inf"), None) for f in home_fams}
    failed = {f: None for f in home_fams}
    for h in heights:
        for off in corners:
            got = {r[6]: r for r in solve_at(h, off)}
            for f in home_fams:
                if failed[f] is not None:
                    continue
                r = got.get(f)
                if r is None:
                    failed[f] = (h, off, "no IK solution (reach)")
                    continue
                if not r[7]:
                    failed[f] = (h, off, f"over block {r[1]:+.4f}, min clearance "
                                         f"{r[2]:+.4f} ({r[3]})")
                    continue
                margin = r[1] - h
                if margin < worst[f][0]:
                    worst[f] = (margin, (h, off))
    home_idx = {r[6]: r[0] for r in rows}
    for f in sorted(home_fams, key=lambda f: home_idx[f]):
        if failed[f] is None:
            m, (h, off) = worst[f]
            print(f"  br {home_idx[f]}: PASSES the descent; tightest: lowest arm point "
                  f"{m*1000:+.1f} mm vs the tool point, tool {h:+.3f} m, fixture "
                  f"{tuple(round(v, 3) for v in off)}")
        else:
            h, off, why = failed[f]
            print(f"  br {home_idx[f]}: FAILS at tool {h:+.3f} m, fixture "
                  f"{tuple(round(v, 3) for v in off)}: {why}")
    survivors = {f for f in home_fams if failed[f] is None}
else:
    print("  ur10e legacy path: no descent sweep.")

print()
print("=" * 78)
print("STEP 3 - D-025 selection criteria applied to the clear branches")
print("=" * 78)
print("D-025 order: all |q| < pi, then q3 < 0, then max |sin q5|, then min ||q||.")
print("Clearance is applied FIRST -- D-025 predates the tables having collision.")
print("Since D-204 the clearance gate holds at the home AND along the descent (step 2b).")
print()
clear_rows = [r for r in rows if r[7] and (survivors is None or r[6] in survivors)]
shallow = [r for r in rows if not r[7]]
dropped = [r for r in rows if r[7] and survivors is not None and r[6] not in survivors]
if shallow:
    print(f"excluded at the home, some arm part hangs below the leading tool point "
          f"or touches a box ({len(shallow)} branches):")
    for idx, gate, clear, who, ferr, q, fam, ok in shallow:
        print(f"  br {idx}: lowest point over the block {gate:+.4f} m, i.e. "
              f"{(PART_STANDOFF-gate)*1000:.0f} mm below the tool point; "
              f"min clearance {clear:+.4f} m over {who}")
    print()
if dropped:
    print(f"excluded on the descent (step 2b): "
          f"{', '.join(f'br {r[0]}' for r in dropped)}")
    print()
print(f"{'br':>3} {'clear':>10} {'max|q|':>8} {'||q||':>7} {'q3<0':>6} "
      f"{'|sin q5|':>9} {'all<pi':>7}")
for idx, gate, clear, who, ferr, q, fam, ok in clear_rows:
    print(f"{idx:>3} {gate:+10.4f} {np.max(np.abs(q)):8.4f} {np.linalg.norm(q):7.4f} "
          f"{str(bool(q[2] < 0)):>6} {abs(np.sin(q[4])):9.4f} "
          f"{str(bool(np.all(np.abs(q) < np.pi))):>7}")

qualified = sorted((r for r in clear_rows if r[5][2] < 0 and np.all(np.abs(r[5]) < np.pi)),
                   key=lambda r: np.linalg.norm(r[5]))
print()
if not clear_rows:
    print("NO BRANCH KEEPS THE TOOL POINT LOWEST, AT THE HOME AND ON THE DESCENT. "
          "The standoff or the target has to change.")
elif not qualified:
    print("Branches clear the cell, but none satisfies the D-025 shape criteria.")
    print(f"Widest clearance is branch {clear_rows[0][0]}; decide explicitly before use.")
else:
    sel = qualified[0]
    print(f"selected: branch {sel[0]}   over the block {sel[1]:+.4f} m, "
          f"min clearance {sel[2]:+.4f} m over {sel[3]}")
    # Restated, not imported: the owning list is UR5E_HOME_JOINT_POS in
    # source/insertion/insertion/tasks/direct/insertion/ur5e_cfg.py, whose
    # keys carry this order. That module needs isaaclab, which this tool
    # deliberately does not, so the two move together by hand.
    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    for name, val in zip(names, sel[5]):
        print(f'    "{name}": {val:.6f},')
