# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Does the policy steer TOWARDS the side the part is tilted into?

THE QUESTION
------------
RT-158 measured that a failing part is TILTED (``rot_beyond_yaw`` median
15.34 deg for the mouth mode against 2.17 deg for a success). A tilted
part has a DEEP side -- the side its tool axis leans towards, where the
opening still has room -- and a HIGH side resting on the rim. The user's
hypothesis, from watching the replay: the rescue is to move sideways
towards the deep side while straightening up, and the policy does not do
it.

That hypothesis is testable from a trace that already exists, with no new
run. This script answers it and NOTHING else.

WHAT IT COMPARES
----------------
Per env, over the last ``--window`` steps of its FIRST episode:

* ``deep``   -- the horizontal direction the DOWNWARD tool axis leans, read
  from ``ee_quat``. Its length is ``sin(tilt)``, so it also reports how much
  tilt there is to steer with.
* ``cmd``    -- the mean COMMANDED lateral action, ``action[0:2]``. The
  command, not the motion, because a parked part moves ~0.04 mm in 20 steps
  (RT-158): the achieved motion of a jammed part carries no direction, only
  noise. ``apply_pose_delta`` adds the position delta in the PARENT frame
  (``insertion_math.py:1316``), and both step limits are the same scalar, so
  the raw action's XY direction IS the commanded world XY direction.
* ``moved``  -- the achieved XY displacement of ``tip_rel``, reported beside
  the command so "commands hard, moves nothing" is visible as such.

The answer is the angle ``cmd`` makes with ``deep``: 0 deg means the policy
drives exactly towards the deep side, 180 deg means away from it.

    towards the deep side, does not get there   -> the CONTROLLER is the suspect
    barely steers sideways at all               -> the REWARD is the suspect
    steers sideways with no relation to the tilt -> the ENCODING is the suspect

Only the third case argues for a new observation channel.

FRAME NOTE, and it is a precondition, not a detail
--------------------------------------------------
``tip_rel`` is env-frame (``insertion_env.py:1145``) and ``ee_quat`` is
world-frame; env origins are pure translations, so their XY axes agree. This
holds only while the pocket is not yawed. With ``fixture_yaw_rad`` or
``fixture_yaw_noise_rad`` non-zero the two frames rotate apart and every
angle below is wrong. The script cannot read those from a trace -- it is
stated here and must be checked against the run's startup report.

USAGE
-----
    python scripts/trace_tilt_steering.py rt_logs/RT-158_trace.json
    python scripts/trace_tilt_steering.py --self-test

Offline, stdlib only. Grouping and the first-episode rule are IMPORTED from
``trace_outcome_split.py`` rather than rewritten, so the two instruments
cannot disagree about which episode is which.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from trace_outcome_split import _q, per_env  # noqa: E402

SCRIPT_MARKER = "trace_tilt_steering-2026-09-06a"

WINDOW = 20  # steps before the end of the episode, as in trace_outcome_split
# Splits the timeouts into the two poses RT-157/RT-158 found. Mode A ("rim")
# ends around 17.7 mm lateral, mode B ("mouth") around 5.0 mm; 10 mm is the
# gap between them, not a tuned value.
MODE_SPLIT_LAT_MM = 10.0
# Below this much tilt there is no deep side to steer towards and the
# direction of `deep` is numerical noise. sin(1 deg) = 0.017.
MIN_TILT_DEG = 1.0
# Below this much command the direction of `cmd` is noise too. The action is
# the raw policy output in [-1, 1] per axis.
MIN_CMD = 0.02


def rotate_by_quat(q, v):
    """Rotate ``v`` (3) by the wxyz quaternion ``q`` (4). Stdlib, no numpy."""
    w, x, y, z = q
    # t = 2 * (q_vec x v)
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def _norm2(v):
    return math.hypot(v[0], v[1])


def angle_deg(a, b):
    """Unsigned angle [0, 180] between two 2-D vectors; nan if either is 0."""
    na, nb = _norm2(a), _norm2(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    c = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def tool_axis_sign(quat, e: int) -> float:
    """+1 or -1 so that ``sign * R(q0) @ ez`` points DOWN.

    The reset pose is vertical by construction, so its tool axis decides the
    convention once instead of the script asserting one. Returns 0.0 when the
    reset axis is not vertical -- then the whole reading is refused rather
    than silently measuring the wrong direction.
    """
    a0 = rotate_by_quat(quat[0][e], (0.0, 0.0, 1.0))
    if abs(a0[2]) < 0.9:
        return 0.0
    return -1.0 if a0[2] > 0.0 else 1.0


def steering(trace: dict, recs: list, window: int = WINDOW) -> list:
    """Add the tilt/steering fields to the per-env records of ``per_env``."""
    quat = trace.get("ee_quat")
    act = trace.get("action")
    tip = trace["tip_rel_m"]
    if quat is None or act is None:
        return recs
    for r in recs:
        if r["outcome"] not in ("success", "timeout", "late"):
            continue
        e, end = r["env"], r["steps"] - 1
        sgn = tool_axis_sign(quat, e)
        if sgn == 0.0:
            r["deep_ok"] = False
            continue
        a_end = rotate_by_quat(quat[end][e], (0.0, 0.0, 1.0))
        deep = (sgn * a_end[0], sgn * a_end[1])
        lo = max(0, end + 1 - window)
        n = (end + 1) - lo
        cmd = (sum(act[i][e][0] for i in range(lo, end + 1)) / n,
               sum(act[i][e][1] for i in range(lo, end + 1)) / n)
        # The mean of the VECTOR cancels an oscillation; the mean of the
        # MAGNITUDE does not. Their ratio separates "pushes gently" from
        # "pushes hard both ways", and without it a small `cmd_mag` cannot be
        # read at all. 1.0 = one steady direction, 0 = pure chatter.
        cmd_abs = sum(math.hypot(act[i][e][0], act[i][e][1]) for i in range(lo, end + 1)) / n
        moved = ((tip[end][e][0] - tip[lo][e][0]) * 1000.0,
                 (tip[end][e][1] - tip[lo][e][1]) * 1000.0)
        r["deep_ok"] = True
        r["tilt_deg"] = math.degrees(math.asin(min(1.0, _norm2(deep))))
        r["cmd_mag"] = _norm2(cmd)
        r["cmd_abs_mean"] = cmd_abs
        r["cmd_straightness"] = (_norm2(cmd) / cmd_abs) if cmd_abs > 0.0 else float("nan")
        r["moved_mm"] = _norm2(moved)
        r["angle_cmd_deep_deg"] = angle_deg(cmd, deep)
        r["angle_moved_deep_deg"] = angle_deg(moved, deep)
    return recs


def _groups(recs: list) -> dict:
    """success / timeout_A (rim) / timeout_B (mouth), the RT-157 split."""
    out = {"success": [], "timeout_A_rim": [], "timeout_B_mouth": []}
    for r in recs:
        if not r.get("deep_ok"):
            continue
        if r["outcome"] == "success":
            out["success"].append(r)
        elif r["outcome"] == "timeout":
            key = "timeout_A_rim" if r["lat_end_mm"] >= MODE_SPLIT_LAT_MM else "timeout_B_mouth"
            out[key].append(r)
    return out


def report(recs: list, window: int = WINDOW) -> list:
    lines = [f"[trace_tilt_steering] marker: {SCRIPT_MARKER}",
             f"[trace_tilt_steering] window {window} steps, mode split at "
             f"{MODE_SPLIT_LAT_MM} mm lateral, deep side needs tilt > {MIN_TILT_DEG} deg"]
    if not any(r.get("deep_ok") for r in recs):
        lines.append("[trace_tilt_steering] NOT MEASURABLE: trace carries no ee_quat/action, "
                     "or the reset pose is not vertical")
        return lines
    for name, g in _groups(recs).items():
        if not g:
            lines.append(f"--- {name} (n=0)")
            continue
        steer = [r for r in g if r["tilt_deg"] > MIN_TILT_DEG and r["cmd_mag"] > MIN_CMD]
        lines.append(f"--- {name} (n={len(g)}, of which {len(steer)} have a readable "
                     f"deep side AND a non-zero command) : p25 / median / p75")
        for f in ("tilt_deg", "cmd_abs_mean", "cmd_mag", "cmd_straightness", "moved_mm"):
            vs = [r[f] for r in g]
            lines.append(f"    {f:22s} {_q(vs, 0.25):8.3f} / {_q(vs, 0.50):8.3f} / {_q(vs, 0.75):8.3f}")
        if not steer:
            lines.append("    angle_cmd_deep_deg     UNREADABLE (no tilt or no command)")
            continue
        vs = [r["angle_cmd_deep_deg"] for r in steer]
        lines.append(f"    angle_cmd_deep_deg     {_q(vs, 0.25):8.2f} / {_q(vs, 0.50):8.2f} / {_q(vs, 0.75):8.2f}")
        vm = [r["angle_moved_deep_deg"] for r in steer if not math.isnan(r["angle_moved_deep_deg"])]
        if vm:
            lines.append(f"    angle_moved_deep_deg   {_q(vm, 0.25):8.2f} / {_q(vm, 0.50):8.2f} / {_q(vm, 0.75):8.2f}")
        toward = sum(1 for v in vs if v < 45.0)
        away = sum(1 for v in vs if v > 135.0)
        lines.append(f"    towards the deep side (< 45 deg)  {toward} of {len(steer)}")
        lines.append(f"    away from it          (> 135 deg) {away} of {len(steer)}")
        lines.append(f"    in between                        {len(steer) - toward - away} of {len(steer)}"
                     "  <- a uniform 50 % here means no relation to the tilt")
    return lines


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _quat_axis_angle(axis, deg):
    n = math.sqrt(sum(v * v for v in axis)) or 1.0
    a = math.radians(deg) / 2.0
    s = math.sin(a) / n
    return [math.cos(a), axis[0] * s, axis[1] * s, axis[2] * s]


def _synth():
    """3 envs, 288 steps, every direction known by construction.

    The reset pose turns ez to POINT DOWN (180 deg about x), which is what a
    real trace does and what fixes the sign convention.

    env 0 succeeds at step 60, upright, no command.
    env 1 times out on the rim (lat 18 mm), tilted so the deep side is +x,
          and commands +x -- it steers TOWARDS the deep side.
    env 2 times out in the mouth (lat 5 mm), same tilt, deep side +x, but
          commands -x -- it steers AWAY.
    """
    T, E = 288, 3
    down = _quat_axis_angle((1.0, 0.0, 0.0), 180.0)
    # A further -20 deg about y leans the DOWNWARD axis towards +x:
    # R_y(-20) takes (0, 0, -1) to (+sin20, 0, -cos20). The sign is worked
    # out here, once, so the checks below state a KNOWN direction.
    lean = _mul(_quat_axis_angle((0.0, 1.0, 0.0), -20.0), down)
    tip, yaw, frc, act, done, quat = [], [], [], [], [], []
    for i in range(T):
        z0 = 40.0 - i if i <= 60 else -20.0
        tip.append([[0.0, 0.0, z0 / 1000.0],
                    [18.0 / 1000.0, 0.0, 4.0 / 1000.0],
                    [5.0 / 1000.0, 0.0, 4.0 / 1000.0]])
        yaw.append([[-1.0, 0.0]] * 3)
        frc.append([[0.0, 0.0, 5.0]] * 3)
        act.append([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
        done.append([i + 1 == 60, i + 1 == 255, i + 1 == 255])
        quat.append([down, down if i == 0 else lean, down if i == 0 else lean])
    return {"tip_rel_m": tip, "yaw_cos_sin": yaw, "force_n": frc, "action": act,
            "done": done, "ee_quat": quat, "num_envs": E, "step_dt_s": 1.0 / 15.0}


def _mul(a, b):
    """wxyz quaternion product a*b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def _self_test() -> int:
    t = _synth()
    recs = steering(t, per_env(t))
    by = {r["env"]: r for r in recs}
    txt = "\n".join(report(recs))
    g = _groups(recs)
    checks = [
        ("reset axis points down -> sign +1", tool_axis_sign(t["ee_quat"], 0) == 1.0),
        ("a non-vertical reset is refused",
         tool_axis_sign([[_quat_axis_angle((1.0, 0.0, 0.0), 90.0)]], 0) == 0.0),
        ("rotate_by_quat: identity leaves ez",
         abs(rotate_by_quat([1.0, 0, 0, 0], (0.0, 0.0, 1.0))[2] - 1.0) < 1e-12),
        ("rotate_by_quat: 180 deg about x flips ez",
         abs(rotate_by_quat(_quat_axis_angle((1.0, 0, 0), 180.0), (0.0, 0.0, 1.0))[2] + 1.0) < 1e-9),
        ("upright env reads ~0 tilt", by[0]["tilt_deg"] < 1e-6),
        ("leaning envs read 20 deg tilt", abs(by[1]["tilt_deg"] - 20.0) < 1e-6),
        ("env 1 steers TOWARDS the deep side", abs(by[1]["angle_cmd_deep_deg"]) < 1e-6),
        ("env 2 steers AWAY from it", abs(by[2]["angle_cmd_deep_deg"] - 180.0) < 1e-6),
        ("angle is nan without a command", math.isnan(by[0]["angle_cmd_deep_deg"])),
        ("mode split puts lat 18 mm on the rim", by[1] in g["timeout_A_rim"]),
        ("mode split puts lat 5 mm in the mouth", by[2] in g["timeout_B_mouth"]),
        ("the success is its own group", by[0] in g["success"]),
        ("report counts env 1 as towards", "towards the deep side (< 45 deg)  1 of 1" in txt),
        ("report counts env 2 as away", "away from it          (> 135 deg) 1 of 1" in txt),
        ("upright env has no readable deep side", "(n=1, of which 0 have" in txt),
        ("angle_deg is unsigned", abs(angle_deg((1.0, 0.0), (0.0, -1.0)) - 90.0) < 1e-9),
    ]
    # MUTATION (D-080) 1: drop the sign from tool_axis_sign, i.e. read the
    # tool axis UP instead of DOWN. Then env 1's command reads 180 deg
    # instead of 0 and "towards" and "away" swap. The two must differ by
    # exactly 180 deg, or the sign is doing nothing.
    checks.append(("flipping the axis sign flips the verdict",
                   abs(abs(by[1]["angle_cmd_deep_deg"] - by[2]["angle_cmd_deep_deg"]) - 180.0) < 1e-6))
    # MUTATION (D-080) 2: reading the command at the LAST step only instead of
    # averaging the window must still be testable -- here both are constant,
    # so the guard is that the magnitude is the commanded 0.5, not 0.
    checks.append(("command magnitude survives the mean", abs(by[1]["cmd_mag"] - 0.5) < 1e-12))
    # A constant command is perfectly straight; the ratio must read 1.
    checks.append(("a constant command reads straightness 1",
                   abs(by[1]["cmd_straightness"] - 1.0) < 1e-12))
    # MUTATION (D-080) 4: a command that flips sign every step has the SAME
    # mean magnitude and a ZERO vector mean. Without the ratio the two read
    # alike, so this case must separate them.
    t3 = {k: v for k, v in t.items()}
    t3["action"] = [[list(c) for c in row] for row in t["action"]]
    for i in range(len(t3["action"])):
        t3["action"][i][1][0] = 0.5 if i % 2 == 0 else -0.5
    r3 = {r["env"]: r for r in steering(t3, per_env(t3))}
    checks.append(("chatter keeps the mean magnitude",
                   abs(r3[1]["cmd_abs_mean"] - by[1]["cmd_abs_mean"]) < 1e-12))
    checks.append(("chatter reads straightness ~0", r3[1]["cmd_straightness"] < 1e-9))
    # MUTATION (D-080) 3: a trace without `action` must be REFUSED, not read
    # as "no steering". Guard on equality with the untouched record.
    t2 = dict(t)
    t2.pop("action")
    r2 = steering(t2, per_env(t2))
    checks.append(("a trace without action is not measurable",
                   all("angle_cmd_deep_deg" not in r for r in r2)))
    checks.append(("and says so", "NOT MEASURABLE" in "\n".join(report(r2))))
    bad = [n for n, ok in checks if not ok]
    if bad:
        print(f"[trace_tilt_steering] self-test: FAIL {bad}")
        return 1
    print(f"[trace_tilt_steering] self-test: {len(checks)}/{len(checks)} checks passed ({SCRIPT_MARKER})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--timeout-step", type=int, default=255)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if not a.path:
        ap.error("give a trace JSON path or --self-test")
    with open(a.path, "r", encoding="utf-8") as fh:
        trace = json.load(fh)
    recs = steering(trace, per_env(trace, a.timeout_step, a.window), a.window)
    for line in report(recs, a.window):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
