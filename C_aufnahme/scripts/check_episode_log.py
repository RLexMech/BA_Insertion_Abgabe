# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Check a run folder's episode record (``episodes.csv``) -- stdlib only.

WHAT IT READS
-------------
``insertion_env._append_episode_rows`` writes one row per finished episode to
``<run folder>/episodes.csv`` and, once, ``episodes_meta.json`` beside it
(user scope 2026-09-15: the interpenetration analysis of the thesis). This
script reads a run folder and answers "is the record complete and
self-consistent", so no run of the seed study has to be repeated for a column
that was empty or wrong.

THE CHECKS (names printed verbatim)
-----------------------------------
C1-C11 are claims about the file alone. C12-C16 tie the file to
``demo_metrics.json`` of the SAME run: that file's trailing window holds the
last ``interpen_max_mm.episodes`` harvested episodes ending at episode count
``episodes``, and the env extends its deques and writes the CSV rows in the
same call and the same order, so the window is exactly
``rows[episodes - W : episodes]``. The two numbers must agree to the
precision the CSV writes (6 decimals in mm, 4 in N).

WHAT IT DOES NOT SHOW
---------------------
That the interpenetration VALUES are physically right. That is the pose
ladder, ``check_insertion_sdf.py --pose-ladder``.

    python scripts/check_episode_log.py --self-test
    python scripts/check_episode_log.py --find-run RT-206a
    python scripts/check_episode_log.py --run-dir logs\rsl_rl\ur5e_insertion\<folder>
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import sys

SCRIPT_MARKER = "check_episode_log-2026-09-15a"

# The agreed column set (user, 2026-09-15). The check's OWN literal: importing
# the env's tuple would assert it against itself.
EXPECTED_COLUMNS = (
    "run", "seed", "iteration", "env_id", "episode", "outcome", "steps",
    "ip_max_mm", "ip_steps_ge_t1", "force_max_filtered_n", "force_max_raw_n",
    "bounds_version", "dr_phase",
)
OUTCOMES = ("success", "force_abort", "timeout", "other")
META_KEYS = ("method", "body_pair", "rate_hz", "t1_mm", "t1_meaning", "unit")

# The written precision, and nothing else: ip_max_mm is formatted to 6
# decimals, forces to 4. A value can sit half a last digit off the in-memory
# number the env compared, so an equality across the two is judged to that.
TOL_MM = 1e-5
TOL_N = 2e-4

CHECK_NAMES = (
    "C1 header is exactly the agreed column set",
    "C2 meta names method, body pair, unit, rate and t1",
    "C3 at least one row",
    "C4 outcome is success, force_abort, timeout or other",
    "C5 steps within 1..episode cap",
    "C6 0 <= ip_steps_ge_t1 <= steps",
    "C7 ip_steps_ge_t1 > 0 exactly when ip_max_mm >= t1",
    "C8 forces are non-negative",
    "C9 run and seed are constant",
    "C10 (env_id, episode) is unique and counts up from 0 per env",
    "C11 iteration is >= 0 and never falls",
    "C12 the CSV holds demo_metrics.json's episode count",
    "C13 demo window: interpenetration maximum matches",
    "C14 demo window: over_thresh count matches",
    "C15 demo window: force-abort rate matches",
    "C16 demo window: filtered force maximum matches",
)


def judge(header, rows, meta, demo) -> dict:
    """All checks on parsed data. Returns {name: (ok, detail)}. Pure."""
    out: dict = {}

    def put(i, ok, detail=""):
        out[CHECK_NAMES[i]] = (bool(ok), detail)

    put(0, tuple(header) == EXPECTED_COLUMNS, f"{list(header)}")
    ip = (meta or {}).get("interpenetration", {})
    missing = [k for k in META_KEYS if k not in ip]
    put(1, not missing and "episode_cap_steps" in (meta or {}), f"missing {missing}")
    put(2, len(rows) > 0, f"{len(rows)} rows")

    cap = int((meta or {}).get("episode_cap_steps", 0))
    t1 = float(ip.get("t1_mm", float("nan")))

    bad = [r for r in rows if r["outcome"] not in OUTCOMES]
    put(3, not bad, f"{len(bad)} bad, first {bad[:1]}")
    bad = [r for r in rows if not (1 <= int(r["steps"]) <= cap)]
    put(4, not bad, f"cap {cap}; {len(bad)} bad, first {bad[:1]}")
    bad = [r for r in rows if not (0 <= int(r["ip_steps_ge_t1"]) <= int(r["steps"]))]
    put(5, not bad, f"{len(bad)} bad, first {bad[:1]}")
    bad = []
    for r in rows:
        v, k = float(r["ip_max_mm"]), int(r["ip_steps_ge_t1"])
        if k > 0 and v < t1 - TOL_MM:
            bad.append(r)
        elif k == 0 and v >= t1 + TOL_MM:
            bad.append(r)
    put(6, not bad and t1 == t1, f"t1 {t1} mm; {len(bad)} bad, first {bad[:1]}")
    bad = [r for r in rows
           if float(r["force_max_filtered_n"]) < 0.0 or float(r["force_max_raw_n"]) < 0.0
           or float(r["ip_max_mm"]) < 0.0]
    put(7, not bad, f"{len(bad)} bad, first {bad[:1]}")
    put(8, len({(r["run"], r["seed"]) for r in rows}) <= 1,
        f"{sorted({(r['run'], r['seed']) for r in rows})[:3]}")

    nxt: dict = {}
    bad = []
    for r in rows:
        e, k = int(r["env_id"]), int(r["episode"])
        if k != nxt.get(e, 0):
            bad.append(r)
        nxt[e] = k + 1
    put(9, not bad, f"{len(bad)} bad, first {bad[:1]}")
    last, bad = 0, []
    for r in rows:
        it = int(r["iteration"])
        if it < 0 or it < last:
            bad.append(r)
        last = max(last, it)
    put(10, not bad, f"{len(bad)} bad, first {bad[:1]}")

    # -- the tie to demo_metrics.json --------------------------------------
    n_ep = int((demo or {}).get("episodes", -1))
    istat = (demo or {}).get("interpen_max_mm", {})
    w = int(istat.get("episodes", 0))
    aligned = demo is not None and 0 < w <= n_ep <= len(rows)
    put(11, aligned, f"demo episodes {n_ep}, window {w}, csv rows {len(rows)}")
    if not aligned:
        for i in range(12, 16):
            put(i, False, "not aligned (C12)")
        return out
    win = rows[n_ep - w:n_ep]
    ips = [float(r["ip_max_mm"]) for r in win]
    put(12, abs(max(ips) - float(istat.get("max_mm", float("nan")))) <= TOL_MM,
        f"csv {max(ips):.6f} demo {istat.get('max_mm')}")
    thresh = float(istat.get("thresh_mm", float("nan")))
    count = sum(1 for v in ips if v > thresh)
    edge = sum(1 for v in ips if abs(v - thresh) <= TOL_MM)
    put(13, abs(count - int(istat.get("over_thresh", -1))) <= edge,
        f"csv {count} demo {istat.get('over_thresh')} ({edge} within the written precision of t1)")
    rate = sum(1 for r in win if r["outcome"] == "force_abort") / w
    put(14, abs(rate - float(demo.get("force_abort_rate", float("nan")))) <= 1e-9,
        f"csv {rate} demo {demo.get('force_abort_rate')}")
    fmax = max(float(r["force_max_filtered_n"]) for r in win)
    dmax = float((demo.get("force_norm_n") or {}).get("max_n", float("nan")))
    put(15, abs(fmax - dmax) <= TOL_N, f"csv {fmax:.4f} demo {dmax}")
    return out


def summary(rows, meta) -> list[str]:
    """What a reader wants from a smoke: counts and the interpenetration tail."""
    if not rows:
        return []
    t1 = float(meta["interpenetration"]["t1_mm"])
    ips = sorted(float(r["ip_max_mm"]) for r in rows)

    def pct(q):
        return ips[min(len(ips) - 1, max(0, int(-(-q * len(ips) // 1)) - 1))]

    steps = sum(int(r["steps"]) for r in rows)
    over = sum(int(r["ip_steps_ge_t1"]) for r in rows)
    outc = {o: sum(1 for r in rows if r["outcome"] == o) for o in OUTCOMES}
    return [
        f"rows {len(rows)}, envs {len({r['env_id'] for r in rows})}, "
        f"iterations {rows[0]['iteration']}..{rows[-1]['iteration']}, outcomes {outc}",
        f"ip_max_mm p50 {pct(0.50):.4f} p95 {pct(0.95):.4f} p99 {pct(0.99):.4f} max {ips[-1]:.4f}; "
        f"episodes >= t1 ({t1:.4f} mm): {sum(1 for v in ips if v >= t1)}; "
        f"steps >= t1: {over} of {steps}",
        f"force_max_filtered_n max {max(float(r['force_max_filtered_n']) for r in rows):.2f}, "
        f"force_max_raw_n max {max(float(r['force_max_raw_n']) for r in rows):.2f}",
    ]


def load_run(run_dir: pathlib.Path):
    with open(run_dir / "episodes.csv", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        rows = [dict(zip(header, line)) for line in reader]
    meta = json.loads((run_dir / "episodes_meta.json").read_text(encoding="utf-8"))
    demo_path = run_dir / "demo_metrics.json"
    demo = json.loads(demo_path.read_text(encoding="utf-8")) if demo_path.is_file() else None
    return header, rows, meta, demo


def find_run(name: str, root: pathlib.Path) -> pathlib.Path:
    """Newest run folder under ``root`` whose name carries ``__<name>``."""
    hits = [p for p in root.glob("*/*") if p.is_dir()
            and (p.name.endswith(f"__{name}") or f"__{name}_" in p.name)]
    if not hits:
        raise SystemExit(f"[check_episode_log] no run folder with '__{name}' under {root}")
    return max(hits, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Self-test: the checks against built tables, each broken one way (D-080)
# ---------------------------------------------------------------------------

def _good():
    """Three envs, four episodes each, one row over t1; demo window = last 8."""
    t1 = 0.293800
    rows = []
    it = 0
    for k in range(4):
        for e in range(3):
            ip = 0.1 * (e + 1) if not (k == 3 and e == 2) else 0.5
            rows.append({
                "run": "09-15_10-00-00__RT-X_seed1", "seed": "1", "iteration": str(it),
                "env_id": str(e), "episode": str(k),
                "outcome": ("success", "timeout", "force_abort")[e], "steps": "20",
                "ip_max_mm": f"{ip:.6f}", "ip_steps_ge_t1": "2" if ip >= t1 else "0",
                "force_max_filtered_n": f"{10.0 + e:.4f}", "force_max_raw_n": f"{12.0 + e:.4f}",
                "bounds_version": "0", "dr_phase": "floor",
            })
        it += 1
    meta = {"episode_cap_steps": 256, "interpenetration": {
        "method": "m", "body_pair": "b", "unit": "mm", "rate_hz": 15.0,
        "t1_mm": t1, "t1_meaning": "algorithmic"}}
    win = rows[4:12]
    demo = {
        "episodes": 12,
        "force_abort_rate": sum(1 for r in win if r["outcome"] == "force_abort") / 8,
        "interpen_max_mm": {"episodes": 8, "max_mm": 0.5, "thresh_mm": t1,
                            "over_thresh": sum(1 for r in win if float(r["ip_max_mm"]) > t1)},
        "force_norm_n": {"max_n": 12.0},
    }
    return list(EXPECTED_COLUMNS), rows, meta, demo


def _mut(fn):
    h, r, m, d = _good()
    h, r, m, d = copy.deepcopy(h), copy.deepcopy(r), copy.deepcopy(m), copy.deepcopy(d)
    fn(h, r, m, d)
    return h, r, m, d


def _set(i, key, val):
    def fn(h, r, m, d):
        r[i][key] = val
    return fn


CASES = (
    ("extra-column", lambda h, r, m, d: h.append("ip_mean_mm"), ("C1",)),
    ("meta-without-body-pair", lambda h, r, m, d: m["interpenetration"].pop("body_pair"), ("C2",)),
    ("empty-file", lambda h, r, m, d: (r.clear(), d.update(episodes=0)),
     ("C3", "C12", "C13", "C14", "C15", "C16")),
    ("unknown-outcome", _set(0, "outcome", "crash"), ("C4",)),
    ("steps-over-cap", _set(0, "steps", "257"), ("C5",)),
    ("more-unclean-steps-than-steps", _set(11, "ip_steps_ge_t1", "21"), ("C6",)),
    ("deep-episode-with-no-unclean-step", _set(0, "ip_max_mm", "0.400000"), ("C7",)),
    ("negative-raw-force", _set(1, "force_max_raw_n", "-1.0000"), ("C8",)),
    ("seed-changes", _set(2, "seed", "2"), ("C9",)),
    ("episode-repeats", _set(3, "episode", "0"), ("C10",)),
    ("iteration-falls", _set(4, "iteration", "0"), ("C11",)),
    ("demo-counts-more-episodes", lambda h, r, m, d: d.update(episodes=13),
     ("C12", "C13", "C14", "C15", "C16")),
    ("demo-interpen-max-differs", lambda h, r, m, d: d["interpen_max_mm"].update(max_mm=0.6), ("C13",)),
    ("demo-over-thresh-differs",
     lambda h, r, m, d: d["interpen_max_mm"].update(over_thresh=d["interpen_max_mm"]["over_thresh"] + 2),
     ("C14",)),
    ("demo-abort-rate-differs", lambda h, r, m, d: d.update(force_abort_rate=0.5), ("C15",)),
    ("demo-force-max-differs", lambda h, r, m, d: d["force_norm_n"].update(max_n=13.0), ("C16",)),
)


def self_test() -> int:
    print(f"[check_episode_log] marker: {SCRIPT_MARKER}")
    base = judge(*_good())
    red = [k for k, (ok, _) in base.items() if not ok]
    if red or len(base) != len(CHECK_NAMES):
        print(f"[check_episode_log]   baseline: FAIL {red}")
        return 1
    print(f"[check_episode_log]   baseline: PASS ({len(base)} checks green)")
    bad = 0
    for name, fn, want in CASES:
        got = judge(*_mut(fn))
        flipped = {k.split(" ", 1)[0] for k, (ok, _) in got.items() if not ok}
        if flipped == set(want):
            print(f"[check_episode_log]   case {name}: PASS (flips exactly {sorted(flipped)})")
        else:
            print(f"[check_episode_log]   case {name}: FAIL -- expected {sorted(want)}, got {sorted(flipped)}")
            bad += 1
    total = 1 + len(CASES)
    print(f"[check_episode_log] self-test: {total - bad}/{total} (1 baseline + {len(CASES)} cases)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", help="Offline cases, no run folder.")
    ap.add_argument("--run-dir", type=str, default=None, help="One run folder.")
    ap.add_argument("--find-run", type=str, default=None,
                    help="RT name; picks the newest folder '__<name>' under --log-root.")
    ap.add_argument("--log-root", type=str, default=str(pathlib.Path("logs") / "rsl_rl"),
                    help="Root holding <experiment>/<run folder>. Default logs/rsl_rl.")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.run_dir:
        run_dir = pathlib.Path(args.run_dir)
    elif args.find_run:
        run_dir = find_run(args.find_run, pathlib.Path(args.log_root))
    else:
        ap.error("give --self-test, --run-dir or --find-run")
    print(f"[check_episode_log] marker: {SCRIPT_MARKER}")
    print(f"[check_episode_log] run folder: {run_dir}")
    header, rows, meta, demo = load_run(run_dir)
    result = judge(header, rows, meta, demo)
    for name, (ok, detail) in result.items():
        print(f"[check_episode_log]   {'PASS' if ok else 'FAIL'}  {name}  -- {detail[:200]}")
    for line in summary(rows, meta):
        print(f"[check_episode_log] {line}")
    fails = [k for k, (ok, _) in result.items() if not ok]
    print(f"[check_episode_log] VERDICT: {'COMPLETE' if not fails else 'INCOMPLETE ' + str(len(fails))}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
