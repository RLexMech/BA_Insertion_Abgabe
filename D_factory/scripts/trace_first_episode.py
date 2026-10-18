'Summarize first-episode outcomes from replay traces.'

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

SCRIPT_MARKER = "trace_first_episode-2026-09-06b"


DEFAULT_TIMEOUT_STEP = 255


def first_episode(done, timeout_step: int = DEFAULT_TIMEOUT_STEP) -> dict:
    'Per env: 1-based step of the FIRST done, classified.'
    n_steps = len(done)
    n_env = len(done[0]) if n_steps else 0
    first = [None] * n_env
    for i, row in enumerate(done):
        for e in range(n_env):
            if first[e] is None and bool(row[e]):
                first[e] = i + 1
    success = sum(1 for s in first if s is not None and s < timeout_step)
    timeout = sum(1 for s in first if s is not None and s == timeout_step)
    late = sum(1 for s in first if s is not None and s > timeout_step)
    censored = sum(1 for s in first if s is None)


    completed = 0
    pooled_success = 0
    for e in range(n_env):
        last = 0
        for i in range(n_steps):
            if bool(done[i][e]):
                length = i + 1 - last
                last = i + 1
                completed += 1
                if length < timeout_step:
                    pooled_success += 1
    return {
        "steps": n_steps,
        "envs": n_env,
        "first_done_step": first,
        "success": success,
        "timeout": timeout,
        "late": late,
        "censored": censored,
        "completed_episodes": completed,
        "pooled_success": pooled_success,
        "running_at_cut": n_env,
        "timeout_step": timeout_step,
    }


def report(r: dict, force_abort_rate) -> list:
    n = r["envs"]
    lines = [f"[trace_first_episode] marker: {SCRIPT_MARKER}",
             f"[trace_first_episode] steps {r['steps']} envs {n} timeout_step {r['timeout_step']}"]
    hist = Counter(s for s in r["first_done_step"] if s is not None)
    top = ", ".join(f"{s}:{c}" for s, c in hist.most_common(5))
    lines.append(f"[trace_first_episode] first_done_step top5 {top}")
    lines.append(
        f"[trace_first_episode] first_episode success {r['success']} timeout {r['timeout']} "
        f"late {r['late']} censored {r['censored']} of {n}")
    if n:
        lines.append(f"[trace_first_episode] first_episode_success_rate {r['success'] / n:.4f}")
    started = r["completed_episodes"] + r["running_at_cut"]
    if started:
        lines.append(
            f"[trace_first_episode] pooled_view completed {r['completed_episodes']} "
            f"running_at_cut {r['running_at_cut']} censored_share {r['running_at_cut'] / started:.4f}")
    if r["completed_episodes"]:
        lines.append(
            f"[trace_first_episode] pooled_success_rate {r['pooled_success']}/{r['completed_episodes']} "
            f"= {r['pooled_success'] / r['completed_episodes']:.4f} (what the env counter reports)")
    if r["censored"] or r["late"]:
        lines.append("[trace_first_episode] VERDICT INVALID: trace too short or timeout_step wrong")
    elif force_abort_rate is None:
        lines.append("[trace_first_episode] NO VERDICT: force_abort_rate not given")
    elif force_abort_rate != 0.0:
        lines.append(
            f"[trace_first_episode] NO VERDICT: force_abort_rate {force_abort_rate} != 0 -- "
            "an early done is not provably a success")
    else:
        lines.append("[trace_first_episode] VERDICT VALID: force_abort_rate 0, early done = success")
    return lines


def _synth(pattern, n_steps: int):
    'pattern[e] = 1-based step at which env e is done first (None = never).'
    return [[(p is not None and p == i + 1) for p in pattern] for i in range(n_steps)]


def _self_test() -> int:
    checks = []

    d = _synth([70, 255, None], 288)
    r = first_episode(d)
    checks.append(("success counted", r["success"] == 1))
    checks.append(("timeout counted", r["timeout"] == 1))
    checks.append(("censored counted", r["censored"] == 1))
    checks.append(("completed episodes", r["completed_episodes"] == 2))
    txt = "\n".join(report(r, 0.0))
    checks.append(("censored blocks verdict", "VERDICT INVALID" in txt))

    d = _synth([10, 20, 255, 255], 288)
    r = first_episode(d)
    checks.append(("rate 0.5", abs(r["success"] / r["envs"] - 0.5) < 1e-9))
    txt = "\n".join(report(r, 0.0))
    checks.append(("valid verdict", "VERDICT VALID" in txt))
    checks.append(("rate printed", "first_episode_success_rate 0.5000" in txt))

    txt = "\n".join(report(r, 0.01))
    checks.append(("force abort blocks", "NO VERDICT: force_abort_rate" in txt))
    txt = "\n".join(report(r, None))
    checks.append(("missing rate blocks", "NO VERDICT: force_abort_rate not given" in txt))

    d = _synth([30, None], 288)
    d[100][0] = True
    d[254][1] = True
    r = first_episode(d)
    checks.append(("first done only", r["first_done_step"] == [30, 255]))
    checks.append(("second episode still completed", r["completed_episodes"] == 3))


    checks.append(("pooled counts later episodes", r["pooled_success"] == 2))
    checks.append(("pooled above first-episode", r["pooled_success"] / r["completed_episodes"] > r["success"] / r["envs"]))
    txt = "\n".join(report(r, 0.0))
    checks.append(("pooled printed", "pooled_success_rate 2/3 = 0.6667" in txt))


    d = _synth([10], 288)
    d[264][0] = True
    r = first_episode(d)
    checks.append(("second timeout not a success", r["completed_episodes"] == 2 and r["pooled_success"] == 1))

    d = _synth([260, 10], 288)
    r = first_episode(d)
    checks.append(("late flagged", r["late"] == 1 and r["success"] == 1))
    checks.append(("late blocks verdict", "VERDICT INVALID" in "\n".join(report(r, 0.0))))


    d = _synth([255, 255, 255, 255], 288)
    r = first_episode(d)
    checks.append(("all timeouts -> rate 0", r["success"] == 0 and r["timeout"] == 4))
    bad = [n for n, ok in checks if not ok]
    if bad:
        print(f"[trace_first_episode] self-test: FAIL {bad}")
        return 1
    print(f"[trace_first_episode] self-test: {len(checks)}/{len(checks)} checks passed ({SCRIPT_MARKER})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", help="trace JSON from play.py --trace-obs")
    ap.add_argument("--timeout-step", type=int, default=DEFAULT_TIMEOUT_STEP)
    ap.add_argument("--force-abort-rate", type=float, default=None,
                    help="force_abort_rate from the run's demo_metrics.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if not a.path:
        ap.error("give a trace JSON path or --self-test")
    with open(a.path, "r", encoding="utf-8") as fh:
        trace = json.load(fh)
    r = first_episode(trace["done"], a.timeout_step)
    for line in report(r, a.force_abort_rate):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
