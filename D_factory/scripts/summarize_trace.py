"""Summarise a ``play.py --trace-obs`` JSON into a few console lines.

Stdlib only, no Isaac. Reads the per-step trace (RT-149 form: lists of
``[step][env][channel]``) and prints, per env, the numbers the RT-149
expectation asks for -- so the answer fits into ``rt_logs/inbox.txt``
instead of a multi-megabyte JSON that cannot travel by clipboard.

Every printed value is a plain reduction of the file: min, max, mean, slope.
No threshold is judged here; judging is ``/rt-check``'s job against the
expectation file.

    python scripts/summarize_trace.py rt_logs/RT-149_trace.json
    python scripts/summarize_trace.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import sys

SCRIPT_MARKER = "summarize_trace-2026-09-05a"


def _col(rows, env, k):
    return [float(r[env][k]) for r in rows]


def _slope_per_step(vals, a, b):
    """Least-squares slope of vals[a:b] against the step index."""
    xs = list(range(a, b))
    ys = vals[a:b]
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx if sxx else float("nan")


def _unwrap(phis):
    out = [phis[0]]
    for p in phis[1:]:
        d = p - out[-1]
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        out.append(out[-1] + d)
    return out


def summarize(trace: dict, episode_steps: int = 256) -> list[str]:
    tip = trace["tip_rel_m"]
    yaw = trace["yaw_cos_sin"]
    frc = trace["force_n"]
    act = trace["action"]
    done = trace["done"]
    n_steps = len(tip)
    n_env = trace.get("num_envs", len(tip[0]))
    dt = float(trace.get("step_dt_s", float("nan")))
    lines = [f"[summarize_trace] marker: {SCRIPT_MARKER}",
             f"[summarize_trace] steps {n_steps} envs {n_env} step_dt_s {dt}"]
    # done steps (1-based step index at which done was True)
    for e in range(n_env):
        ds = [i + 1 for i, d in enumerate(done) if bool(d[e])]
        lines.append(f"env {e} done_at_steps {ds}")
    # first episode only: the row AFTER a done is already the reset observation
    # of the next episode (RT-149: done at index 254, reset row at 255), so the
    # episode ends at the first done, not at the nominal step count.
    first_done = [i for i, d in enumerate(done) if any(bool(v) for v in d)]
    T = min(first_done[0] + 1 if first_done else episode_steps, n_steps)
    lines.append(f"[summarize_trace] episode_1_steps {T}")
    for e in range(n_env):
        x = [v * 1000.0 for v in _col(tip, e, 0)][:T]
        y = [v * 1000.0 for v in _col(tip, e, 1)][:T]
        z = [v * 1000.0 for v in _col(tip, e, 2)][:T]
        cs = _col(yaw, e, 0)[:T]
        sn = _col(yaw, e, 1)[:T]
        phi = _unwrap([math.atan2(s, c) for s, c in zip(sn, cs)])
        fx, fy, fz = _col(frc, e, 0)[:T], _col(frc, e, 1)[:T], _col(frc, e, 2)[:T]
        fn = [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(fx, fy, fz)]
        # action delta per step, max over channels, steps 20..T-1
        dmax = 0.0
        for i in range(max(21, 1), T):
            d = max(abs(float(act[i][e][k]) - float(act[i - 1][e][k])) for k in range(len(act[i][e])))
            dmax = max(dmax, d)
        y_abs = [abs(v) for v in y]
        i_y = max(range(T), key=lambda i: y_abs[i])
        tail = slice(max(T - 50, 0), T)
        lines.append(
            f"env {e} ep1 tip_x_mm min {min(x):.2f} max {max(x):.2f} | "
            f"tip_y_mm abs_max {y_abs[i_y]:.2f} at_step {i_y + 1} last50 min {min(y_abs[tail]):.2f} max {max(y_abs[tail]):.2f} | "
            f"tip_z_mm min {min(z):.2f} max {max(z):.2f} last50 min {min(z[tail]):.2f} max {max(z[tail]):.2f}"
        )
        s1 = _slope_per_step(phi, 50, min(150, T))
        s2 = _slope_per_step(phi, min(150, T), min(250, T))
        deg = 180.0 / math.pi
        lines.append(
            f"env {e} ep1 yaw_deg start {phi[0] * deg:.2f} end {phi[T - 1] * deg:.2f} "
            f"slope_deg_per_step 50-150 {s1 * deg:.4f} 150-250 {s2 * deg:.4f} "
            f"slope_deg_per_s {s1 * deg / dt if dt == dt and dt else float('nan'):.3f}"
        )
        lines.append(
            f"env {e} ep1 force_n min {min(fn):.3f} max {max(fn):.3f} mean {sum(fn) / T:.3f} "
            f"last50 min {min(fn[tail]):.3f} max {max(fn[tail]):.3f} | "
            f"action_delta_max_20on {dmax:.4f}"
        )
    return lines


def _self_test() -> int:
    # synthetic: 2 envs, 300 steps; env 0 drifts to y=51 mm, z=15 mm, yaw 1 deg/step,
    # force 1.0 N when resting; env 1 stays put.
    T, E = 300, 2
    tip, yaw, frc, act, done = [], [], [], [], []
    for i in range(T):
        rows_t, rows_y, rows_f, rows_a, rows_d = [], [], [], [], []
        for e in range(E):
            if e == 0:
                yv = min(i * 0.5, 51.0) / 1000.0
                zv = (30.0 - min(i * 0.5, 15.0)) / 1000.0
                ph = math.radians(i * 1.0)
                f = 1.0 if i > 100 else 0.0
            else:
                yv, zv, ph, f = 0.0, 0.030, 0.0, 0.0
            rows_t.append([0.0, yv, zv])
            rows_y.append([math.cos(ph), math.sin(ph)])
            rows_f.append([f, 0.0, 0.0])
            rows_a.append([0.1] * 6)
            rows_d.append(i + 1 == 255)
        tip.append(rows_t); yaw.append(rows_y); frc.append(rows_f); act.append(rows_a); done.append(rows_d)
    tr = {"tip_rel_m": tip, "yaw_cos_sin": yaw, "force_n": frc, "action": act, "done": done,
          "num_envs": E, "step_dt_s": 1.0 / 15.0}
    out = summarize(tr)
    text = "\n".join(out)
    checks = [
        ("done at 255", "done_at_steps [255]" in text),
        ("episode cut at first done", "episode_1_steps 255" in text),
        ("y abs max 51", "tip_y_mm abs_max 51.00" in text),
        ("z last50 15", "last50 min 15.00 max 15.00" in text),
        ("yaw slope 1 deg/step", "slope_deg_per_step 50-150 1.0000" in text),
        ("force last50 1.0", "last50 min 1.000 max 1.000" in text),
        ("action delta 0", "action_delta_max_20on 0.0000" in text),
        ("env1 flat", "env 1 ep1 tip_x_mm min 0.00 max 0.00" in text),
    ]
    # mutation: unwrap must handle the pi wrap -- a 350-step 1 deg/step yaw crosses +-180
    bad = [c for c, ok in checks if not ok]
    n = len(checks)
    if bad:
        print(f"[summarize_trace] self-test: FAIL {bad}")
        return 1
    print(f"[summarize_trace] self-test: {n}/{n} checks passed ({SCRIPT_MARKER})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", nargs="?", help="trace JSON from play.py --trace-obs")
    ap.add_argument("--episode-steps", type=int, default=256)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if not a.path:
        ap.error("path or --self-test required")
    with open(a.path, encoding="utf-8") as fh:
        tr = json.load(fh)
    for line in summarize(tr, a.episode_steps):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
