"""Split a ``play.py --trace-obs`` run into SUCCESS and TIMEOUT and describe both.

Companion to ``trace_first_episode.py`` (RT-157, 2026-09-06). That script counts
outcomes; this one asks what the two groups DO. One first episode per env, so
every env contributes exactly one sample to exactly one group.

What the trace carries, and nothing more: ``tip_rel`` (3), ``ee_quat`` (4),
``yaw_cos_sin`` (2), the contact force (3), the action, ``done``. The slice
table it was written from travels inside the file, under ``slices``.

  * ``tip_rel`` is the leading tool point relative to the pocket entrance.
    RT-157h40t step 1 reads z = +39.99 mm at a commanded start of +40 mm, so
    +z is ABOVE the entrance plane and z < 0 means the point is inside.
  * yaw is the cos/sin pair; the reset pose reads cos = -1, i.e. phi = 180 deg
    is ALIGNED. The reported error is ``180 - |phi|``.
  * ``ee_quat`` (15:19, wxyz) is in traces written from 2026-09-06 on. Older
    traces do not have it and still load; the tilt columns are then absent.

TILT, and why it is measured this way. Which LOCAL axis of the EE frame runs
along the insertion direction is not established anywhere in this repo, so no
absolute tilt against the pocket axis is computed here -- that would be an
invented convention. What needs no convention is the rotation each env has
undergone SINCE ITS OWN RESET: the reset pose is aligned, so the angle between
``ee_quat`` at the end and at step 1 is the total rotation away from aligned.
Compare it with the yaw error from the same step:

  * total rotation ~= |yaw error|  ->  the part turned about the pocket axis
    only. No tilt.
  * total rotation >> |yaw error|  ->  rotation the yaw channel cannot see,
    i.e. the part is tilted.

The gap is reported per group; it does NOT say about which axis the tilt is.

The last-20-step window before the end of the episode separates a part that is
PARKED (small z travel, standing force) from one that is still moving.

TRUTH BLOCK (2026-09-13, RT-189s1pf). Traces written by play.py from that day
on carry ``truth``: the force before the observation noise (tared raw and EMA),
the true tip pose, the tool body pose, the OSC delta and clamped target, and
the SAPU interpenetration. For each FIRST episode the block reports, over the
steps ``--truth-from`` .. end: the mean force vector and its magnitude (raw and
EMA), the true tip's travel per axis, the mean position error target - body,
and the interpenetration. It prints numbers only; it draws no contact verdict.
The ``parked_in_contact`` line above reads the NOISED force channel and is no
contact evidence in noised traces (RT-189s1p correction, VERDICTS.md).

Usage:
    python scripts/trace_outcome_split.py rt_logs/RT-157h40t_trace.json
    python scripts/trace_outcome_split.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys

SCRIPT_MARKER = "trace_outcome_split-2026-09-13b"

DEFAULT_TIMEOUT_STEP = 255  # see trace_first_episode.py
WINDOW = 20  # steps before the end of the episode


def _q(xs, p):
    """Nearest-rank percentile of a non-empty list; p in [0, 1]."""
    s = sorted(xs)
    if not s:
        return float("nan")
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def quat_angle_deg(a, b) -> float:
    """Angle [deg] of the rotation taking wxyz quaternion ``a`` to ``b``.

    ``q`` and ``-q`` are the same rotation, so the dot product is taken in
    absolute value; without that a sign flip of the stored quaternion would
    read as a 180 degree turn. Clamped before ``acos`` because a dot product
    of 1.0000000002 is a float artefact, not a domain error.
    """
    d = abs(sum(x * y for x, y in zip(a, b)))
    d = min(1.0, max(-1.0, d))
    return math.degrees(2.0 * math.acos(d))


def per_env(trace: dict, timeout_step: int = DEFAULT_TIMEOUT_STEP,
            window: int = WINDOW) -> list:
    """One record per env for its FIRST episode."""
    tip, yaw, frc, done = trace["tip_rel_m"], trace["yaw_cos_sin"], trace["force_n"], trace["done"]
    quat = trace.get("ee_quat")
    n_steps = len(done)
    n_env = len(done[0]) if n_steps else 0
    out = []
    for e in range(n_env):
        end = None
        for i in range(n_steps):
            if bool(done[i][e]):
                end = i
                break
        if end is None:
            out.append({"env": e, "outcome": "censored"})
            continue
        step = end + 1
        outcome = "success" if step < timeout_step else ("timeout" if step == timeout_step else "late")
        xs = [tip[i][e][0] * 1000.0 for i in range(end + 1)]
        ys = [tip[i][e][1] * 1000.0 for i in range(end + 1)]
        zs = [tip[i][e][2] * 1000.0 for i in range(end + 1)]
        lat = [math.hypot(a, b) for a, b in zip(xs, ys)]
        fn = [math.sqrt(sum(v * v for v in frc[i][e])) for i in range(end + 1)]
        phi = [abs(math.degrees(math.atan2(yaw[i][e][1], yaw[i][e][0]))) for i in range(end + 1)]
        w = slice(max(0, end + 1 - window), end + 1)
        rec_tilt = {}
        if quat is not None:
            q0 = quat[0][e]
            rot = [quat_angle_deg(q0, quat[i][e]) for i in range(end + 1)]
            yaw_err_end = 180.0 - phi[-1]
            rec_tilt = {
                "rot_total_end_deg": rot[-1],
                "rot_total_max_deg": max(rot),
                # what the yaw channel cannot account for; never negative
                "rot_beyond_yaw_end_deg": max(0.0, rot[-1] - abs(yaw_err_end)),
            }
        out.append({
            "env": e,
            "outcome": outcome,
            "steps": step,
            "z_end_mm": zs[-1],
            "z_min_mm": min(zs),
            "lat_end_mm": lat[-1],
            "lat_max_mm": max(lat),
            "yaw_err_end_deg": 180.0 - phi[-1],
            "yaw_err_max_deg": max(180.0 - p for p in phi),
            "force_end_n": fn[-1],
            "force_max_n": max(fn),
            "w_force_mean_n": sum(fn[w]) / len(fn[w]),
            "w_z_travel_mm": max(zs[w]) - min(zs[w]),
            "w_lat_travel_mm": max(lat[w]) - min(lat[w]),
            **rec_tilt,
        })
    return out


def _vmean(rows):
    n = len(rows)
    return [sum(r[k] for r in rows) / n for k in range(len(rows[0]))]


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def truth_per_env(trace: dict, truth_from: int = 20) -> list:
    """One record per env over steps ``truth_from`` .. end of its FIRST episode.

    Returns ``[]`` for a trace without the truth block. An env whose first
    episode ends before ``truth_from`` gets ``{"env": e, "window": 0}``; an env
    with no done in the trace uses the whole trace as its episode.
    """
    tr = trace.get("truth")
    if not tr:
        return []
    done = trace["done"]
    n_steps = len(done)
    n_env = len(done[0]) if n_steps else 0
    out = []
    for e in range(n_env):
        end = next((i for i in range(n_steps) if bool(done[i][e])), n_steps - 1)
        idx = list(range(truth_from, end + 1))
        if not idx:
            out.append({"env": e, "window": 0})
            continue
        raw = [tr["force_tared_raw_n"][i][e] for i in idx]
        ema = [tr["force_ema_n"][i][e] for i in idx]
        tip = [[v * 1000.0 for v in tr["tip_true_m"][i][e]] for i in idx]
        err = [[(tr["osc_target_pose_w"][i][e][k] - tr["body_pose_w"][i][e][k]) * 1000.0
                for k in range(3)] for i in idx]
        pen = [tr["interpen_max_m"][i][e] * 1000.0 for i in idx]
        raw_m, ema_m, err_m = _vmean(raw), _vmean(ema), _vmean(err)
        rec = {
            "env": e,
            "window": len(idx),
            "raw_mean_n": raw_m,
            "raw_mean_norm_n": _norm(raw_m),
            "ema_mean_n": ema_m,
            "ema_mean_norm_n": _norm(ema_m),
            "tip_end_mm": tip[-1],
            "tip_travel_mm": [max(t[k] for t in tip) - min(t[k] for t in tip) for k in range(3)],
            "err_mean_mm": err_m,
            "err_mean_norm_mm": _norm(err_m),
            "pen_max_mm": max(pen),
            "pen_pos_frac": sum(1 for v in pen if v > 0.0) / len(pen),
        }
        # THE TORQUE (D-188, traces written from 2026-09-13 on): tared raw
        # and EMA, same window, N m. An older trace has no such keys and gets
        # no such fields -- the report prints without them rather than failing.
        if "torque_tared_raw_nm" in tr:
            tq = [tr["torque_tared_raw_nm"][i][e] for i in idx]
            tq_m = _vmean(tq)
            rec["torque_raw_mean_nm"] = tq_m
            rec["torque_raw_mean_norm_nm"] = _norm(tq_m)
            rec["torque_raw_max_norm_nm"] = max(_norm(v) for v in tq)
        if "wrench_valid" in tr:
            rec["wrench_valid_frac"] = sum(1 for i in idx if bool(tr["wrench_valid"][i][e])) / len(idx)
        out.append(rec)
    return out


def _f3(v):
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


def truth_report(recs: list, truth_from: int) -> list:
    if not recs:
        return ["--- truth: NOT IN THIS TRACE (written before 2026-09-13 or not the insertion env)"]
    lines = [f"--- truth per env, steps {truth_from}..end of the first episode "
             "(force before obs noise; tip true, pocket frame; err = OSC target - tool body, world)"]
    for r in recs:
        if not r["window"]:
            lines.append(f"    env {r['env']}: episode ends before step {truth_from}, no window")
            continue
        lines.append(
            f"    env {r['env']} n={r['window']} | raw F mean {_f3(r['raw_mean_n'])} N "
            f"norm {r['raw_mean_norm_n']:.3f} | ema F mean {_f3(r['ema_mean_n'])} N "
            f"norm {r['ema_mean_norm_n']:.3f} | tip end {_f3(r['tip_end_mm'])} mm "
            f"travel {_f3(r['tip_travel_mm'])} mm | err mean {_f3(r['err_mean_mm'])} mm "
            f"norm {r['err_mean_norm_mm']:.3f} | interpen max {r['pen_max_mm']:.4f} mm, "
            f">0 in {r['pen_pos_frac']:.2f} of steps"
            + (f" | raw tau mean {_f3(r['torque_raw_mean_nm'])} Nm norm "
               f"{r['torque_raw_mean_norm_nm']:.4f} max {r['torque_raw_max_norm_nm']:.4f}"
               if "torque_raw_mean_nm" in r else "")
            + (f" | wrench valid {r['wrench_valid_frac']:.3f}"
               if "wrench_valid_frac" in r else ""))
    return lines


FIELDS = ["steps", "z_end_mm", "z_min_mm", "lat_end_mm", "lat_max_mm",
          "yaw_err_end_deg", "yaw_err_max_deg", "force_end_n", "force_max_n",
          "w_force_mean_n", "w_z_travel_mm", "w_lat_travel_mm"]
# Only present in traces written from 2026-09-06 on; an older trace prints
# the same report without these rows rather than failing on a missing key.
TILT_FIELDS = ["rot_total_end_deg", "rot_total_max_deg", "rot_beyond_yaw_end_deg"]


def report(recs: list, parked_force_n: float = 3.0, parked_travel_mm: float = 1.0,
           tilt_deg: float = 2.0) -> list:
    groups = {}
    for r in recs:
        groups.setdefault(r["outcome"], []).append(r)
    lines = [f"[trace_outcome_split] marker: {SCRIPT_MARKER}",
             f"[trace_outcome_split] envs {len(recs)} groups " +
             " ".join(f"{k}:{len(v)}" for k, v in sorted(groups.items()))]
    for name in ("success", "timeout", "late", "censored"):
        g = groups.get(name)
        if not g or name in ("censored", "late"):
            continue
        has_tilt = TILT_FIELDS[0] in g[0]
        lines.append(f"--- {name} (n={len(g)}) : field p25 / median / p75")
        for f in FIELDS + (TILT_FIELDS if has_tilt else []):
            vs = [r[f] for r in g]
            lines.append(f"    {f:23s} {_q(vs, 0.25):9.2f} / {_q(vs, 0.50):9.2f} / {_q(vs, 0.75):9.2f}")
        if has_tilt:
            tilted = sum(1 for r in g if r["rot_beyond_yaw_end_deg"] > tilt_deg)
            lines.append(f"    tilted  {tilted} of {len(g)} "
                         f"(rotation beyond yaw > {tilt_deg} deg at the end of the episode)")
        else:
            lines.append("    tilt  NOT MEASURABLE: this trace carries no ee_quat")
        parked = sum(1 for r in g
                     if r["w_force_mean_n"] >= parked_force_n and r["w_z_travel_mm"] <= parked_travel_mm)
        inside = sum(1 for r in g if r["z_min_mm"] < 0.0)
        lines.append(f"    parked_in_contact  {parked} of {len(g)} "
                     f"(force >= {parked_force_n} N and z travel <= {parked_travel_mm} mm over the last {WINDOW} steps)")
        lines.append(f"    reached_below_entrance  {inside} of {len(g)} (z_min < 0 mm)")
    return lines


def _quat_axis_angle(axis, deg):
    """wxyz quaternion for a rotation of ``deg`` about a unit-ish ``axis``."""
    n = math.sqrt(sum(v * v for v in axis)) or 1.0
    a = math.radians(deg) / 2.0
    s = math.sin(a) / n
    return [math.cos(a), axis[0] * s, axis[1] * s, axis[2] * s]


def _synth(with_quat: bool = True):
    """2 envs, 288 steps: env 0 succeeds at 60 pushing in; env 1 parks on the rim.

    Orientation, when ``with_quat``: env 0 stays exactly at its reset pose;
    env 1 turns 25 deg about x while its yaw channel reads a 10 deg error.
    Those two are deliberately INCONSISTENT as physics: the point is to test
    the arithmetic in isolation, with both the total rotation (25) and the
    part the yaw channel cannot account for (25 - 10 = 15) known exactly and
    not recomputed by the code under test.
    """
    T, E = 288, 2
    tip, yaw, frc, done = [], [], [], []
    quat = [] if with_quat else None
    for i in range(T):
        rt, ry, rf, rd = [], [], [], []
        rq = []
        # env 0: z 40 -> -20 by step 60, then done
        z0 = 40.0 - i * 1.0 if i <= 60 else -20.0
        rt.append([0.0, 0.0, z0 / 1000.0])
        ry.append([-1.0, 0.0])
        rf.append([0.0, 0.0, 5.0])
        rd.append(i + 1 == 60)
        rq.append([1.0, 0.0, 0.0, 0.0])  # env 0 never leaves its reset pose
        # env 1: z stalls at 2 mm from step 30 on, 10 N standing, 8 mm off axis
        z1 = 40.0 - i * 1.2 if i < 30 else 4.0
        rt.append([8.0 / 1000.0, 0.0, z1 / 1000.0])
        ry.append([math.cos(math.radians(170.0)), math.sin(math.radians(170.0))])
        rf.append([0.0, 0.0, 10.0])
        rd.append(i + 1 == 255)
        rq.append([1.0, 0.0, 0.0, 0.0] if i == 0 else _quat_axis_angle((1.0, 0.0, 0.0), 25.0))
        tip.append(rt); yaw.append(ry); frc.append(rf); done.append(rd)
        if quat is not None:
            quat.append(rq)
    out = {"tip_rel_m": tip, "yaw_cos_sin": yaw, "force_n": frc, "done": done,
           "num_envs": E, "step_dt_s": 1.0 / 15.0}
    # Truth: env 1 has raw force 9 N in z BEFORE step 20 and 2 N from step 20
    # on, EMA 1.5 N, the OSC target 1 mm below the body, a still true tip and
    # 0.01 mm interpenetration on every even step. Env 0 is all zeros.
    tr = {k: [] for k in ("force_tared_raw_n", "force_ema_n", "tip_true_m", "body_pose_w",
                          "osc_delta", "osc_target_pose_w", "interpen_max_m")}
    for i in range(T):
        tr["force_tared_raw_n"].append([[0.0, 0.0, 0.0], [0.0, 0.0, 9.0 if i < 20 else 2.0]])
        tr["force_ema_n"].append([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
        tr["tip_true_m"].append([[0.0, 0.0, 0.0], [0.008, 0.0, 0.004]])
        tr["body_pose_w"].append([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]])
        tr["osc_delta"].append([[0.0] * 6, [0.0] * 6])
        tr["osc_target_pose_w"].append([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.299, 1.0, 0.0, 0.0, 0.0]])
        tr["interpen_max_m"].append([0.0, 1e-5 if i % 2 == 0 else 0.0])
    # D-188: env 1 carries a 0.9 N m torque about y before step 20 and 0.3 N m
    # from step 20 on (so a window that starts at 0 would read a different
    # mean); env 0 is zero. wrench_valid is False on row 0 only.
    tr["torque_tared_raw_nm"] = [[[0.0, 0.0, 0.0], [0.0, 0.9 if i < 20 else 0.3, 0.0]]
                                 for i in range(T)]
    tr["torque_ema_nm"] = [[[0.0, 0.0, 0.0], [0.0, 0.25, 0.0]] for _ in range(T)]
    tr["wrench_valid"] = [[i > 0, i > 0] for i in range(T)]
    out["truth"] = tr
    if quat is not None:
        out["ee_quat"] = quat
    return out


def _self_test() -> int:
    recs = per_env(_synth())
    s = [r for r in recs if r["outcome"] == "success"]
    t = [r for r in recs if r["outcome"] == "timeout"]
    checks = [
        ("one success", len(s) == 1),
        ("one timeout", len(t) == 1),
        ("success ends at 60", s[0]["steps"] == 60),
        ("timeout ends at 255", t[0]["steps"] == 255),
        ("success went below entrance", s[0]["z_min_mm"] < 0),
        ("timeout stayed above", t[0]["z_min_mm"] > 0),
        ("timeout lateral 8 mm", abs(t[0]["lat_end_mm"] - 8.0) < 1e-6),
        ("success lateral 0", abs(s[0]["lat_end_mm"]) < 1e-6),
        ("yaw error 0 for aligned", abs(s[0]["yaw_err_end_deg"]) < 1e-6),
        ("yaw error 10 deg", abs(t[0]["yaw_err_end_deg"] - 10.0) < 1e-6),
        ("timeout parked: no z travel", abs(t[0]["w_z_travel_mm"]) < 1e-9),
        ("timeout standing force 10 N", abs(t[0]["w_force_mean_n"] - 10.0) < 1e-9),
        ("success still moving", s[0]["w_z_travel_mm"] > 1.0),
    ]
    txt = "\n".join(report(recs))
    checks.append(("parked counted for timeout", "parked_in_contact  1 of 1" in txt))
    checks.append(("below entrance counted", "reached_below_entrance  1 of 1" in txt))
    # MUTATION (D-080): reading the LAST step instead of the deepest would make
    # the success look shallow. The two must differ.
    checks.append(("z_min differs from z_end", s[0]["z_min_mm"] <= s[0]["z_end_mm"]))
    # percentile helper on a known list
    checks.append(("median of 1..5 is 3", _q([1, 2, 3, 4, 5], 0.5) == 3))
    # -- tilt, from ee_quat -------------------------------------------------
    checks.append(("success stayed at its reset pose",
                   abs(s[0]["rot_total_end_deg"]) < 1e-9))
    checks.append(("timeout turned 25 deg in total",
                   abs(t[0]["rot_total_end_deg"] - 25.0) < 1e-6))
    checks.append(("15 deg of it the yaw channel cannot see",
                   abs(t[0]["rot_beyond_yaw_end_deg"] - 15.0) < 1e-6))
    checks.append(("beyond-yaw never goes negative",
                   s[0]["rot_beyond_yaw_end_deg"] >= 0.0))
    checks.append(("tilted counted for the timeout", "tilted  1 of 1" in txt))
    checks.append(("tilted not counted for the success", "tilted  0 of 1" in txt))
    # q and -q are the same rotation: a sign flip must read 0, not 180 deg
    checks.append(("quaternion sign flip is not a rotation",
                   abs(quat_angle_deg([1.0, 0, 0, 0], [-1.0, 0, 0, 0])) < 1e-6))
    # MUTATION (D-080): dropping the abs() in quat_angle_deg would turn that
    # sign flip into 360 deg and break the check above.
    # -- an OLD trace without ee_quat must still work, and say so ------------
    old = per_env(_synth(with_quat=False))
    checks.append(("old trace loses no other field", "lat_end_mm" in old[0]))
    checks.append(("old trace carries no tilt field",
                   "rot_total_end_deg" not in old[0]))
    checks.append(("old trace says tilt is not measurable",
                   "tilt  NOT MEASURABLE" in "\n".join(report(old))))
    # -- truth block ---------------------------------------------------------
    tru = truth_per_env(_synth(), 20)
    t1 = tru[1]
    checks.append(("truth: env 0 ends at step index 59, window 40 steps", tru[0]["window"] == 40))
    checks.append(("truth: env 1 window 235 steps (20..254)", t1["window"] == 235))
    # MUTATION (D-080): starting the window at 0 instead of truth_from pulls the
    # 9 N approach steps into the mean and breaks this check.
    checks.append(("truth: raw force mean 2 N, approach excluded",
                   abs(t1["raw_mean_norm_n"] - 2.0) < 1e-9))
    checks.append(("truth: EMA mean 1.5 N kept apart from raw",
                   abs(t1["ema_mean_norm_n"] - 1.5) < 1e-9))
    # MUTATION (D-080): body - target instead of target - body flips the sign.
    checks.append(("truth: target 1 mm BELOW the body reads err z -1 mm",
                   abs(t1["err_mean_mm"][2] + 1.0) < 1e-6))
    checks.append(("truth: still tip has zero travel", max(t1["tip_travel_mm"]) < 1e-9))
    checks.append(("truth: interpen max 0.01 mm", abs(t1["pen_max_mm"] - 0.01) < 1e-12))
    checks.append(("truth: interpen > 0 on 118 of 235 steps",
                   abs(t1["pen_pos_frac"] - 118 / 235) < 1e-12))
    checks.append(("truth: old trace says NOT IN THIS TRACE",
                   "NOT IN THIS TRACE" in "\n".join(truth_report(truth_per_env(_synth(False) | {"truth": None}, 20), 20))))
    checks.append(("truth: report prints env 1", "env 1 n=235" in "\n".join(truth_report(tru, 20))))
    # -- torque (D-188) ---------------------------------------------------------
    checks.append(("truth: raw torque mean 0.3 Nm about y, approach excluded",
                   abs(t1["torque_raw_mean_nm"][1] - 0.3) < 1e-12
                   and abs(t1["torque_raw_mean_norm_nm"] - 0.3) < 1e-12))
    checks.append(("truth: raw torque max norm is 0.3 in the window (0.9 lies before it)",
                   abs(t1["torque_raw_max_norm_nm"] - 0.3) < 1e-12))
    checks.append(("truth: wrench valid on every window step (row 0 is outside it)",
                   abs(t1["wrench_valid_frac"] - 1.0) < 1e-12))
    checks.append(("truth: report prints raw tau",
                   "raw tau mean (+0.000, +0.300, +0.000) Nm" in "\n".join(truth_report(tru, 20))))
    # MUTATION (D-080): a window from step 0 reads the 0.9 N m approach too.
    checks.append(("truth: a window from step 0 changes the torque mean (the window binds)",
                   truth_per_env(_synth(), 0)[1]["torque_raw_mean_nm"][1] > 0.3))
    _old = _synth()
    for _k in ("torque_tared_raw_nm", "torque_ema_nm", "wrench_valid"):
        del _old["truth"][_k]
    _old_rep = "\n".join(truth_report(truth_per_env(_old, 20), 20))
    checks.append(("truth: a pre-D-188 trace prints WITHOUT the tau fields, no failure",
                   "env 1 n=235" in _old_rep and "raw tau" not in _old_rep
                   and "wrench valid" not in _old_rep))
    bad = [n for n, ok in checks if not ok]
    if bad:
        print(f"[trace_outcome_split] self-test: FAIL {bad}")
        return 1
    print(f"[trace_outcome_split] self-test: {len(checks)}/{len(checks)} checks passed ({SCRIPT_MARKER})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--timeout-step", type=int, default=DEFAULT_TIMEOUT_STEP)
    ap.add_argument("--truth-from", type=int, default=20,
                    help="first step of the truth window (default 20)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if not a.path:
        ap.error("give a trace JSON path or --self-test")
    with open(a.path, "r", encoding="utf-8") as fh:
        trace = json.load(fh)
    for line in report(per_env(trace, a.timeout_step)):
        print(line)
    for line in truth_report(truth_per_env(trace, a.truth_from), a.truth_from):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
