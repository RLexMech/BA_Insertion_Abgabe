#!/usr/bin/env python3
"""shorten_rt_log.py -- cut a training-PC log down to what /rt-check reads.

WHY (2026-09-05, user request after RT-150). One rt_log run is ~680 lines,
and ~600 of them are Isaac/Kit boot noise plus the INSERTION STARTUP REPORT
printed three times (steps 2, 60, 180). The user pastes the whole thing into
`rt_logs/inbox.txt` by hand, and the JSON the script wrote (`--out`) has to
be pasted separately -- RT-135's JSONs went missing that way and three
expectation points could not be judged. This script produces ONE short file
that carries everything `/rt-check` needs, so one paste is the whole handover.

WHAT IT KEEPS, in order of precedence:
  1. every `[rt_log]` line (header, cmd, git, exit code) -- the run identity;
  2. everything from the first `Traceback` to the run's exit line;
  3. any line `filter_rt_log.ANOMALY_RE` matches (FAIL, Error, error:, ...);
  4. the FIRST startup report of each run in full; a later repeat becomes one
     line naming the step, PLUS the lines that genuinely differ between
     printouts -- the observation bias and the grasp belief error. Those are
     the whole reason `report_at_steps` has a count past the episode cap;
  5. every other line that is not known noise.
WHAT IT DROPS: `filter_rt_log.NOISE_RE` (capitalised Kit tags, ISO
timestamps, GPU table rows), the Kit user-config line, ANSI-coloured banner
lines, tab-indented Isaac sub-lines, blank lines.

`--json PATH` appends the script's own metrics file in a compact rendering
(scalar lists on one line) when it is at most MAX_JSON_LINES long; a larger
file is named with its size and NOT appended (the printing script is
expected to carry a SUMMARY block of its own -- `check_seated_success.py`
reward curve does since 2026-09-05).

Stdlib only; runs on the laptop. `--self-test` builds a synthetic log and
applies the D-080 mutations (a repeat report that must vanish, a Traceback
inside noise that must survive, a `[rt_log]` line that must never drop).
The noise/anomaly/divider rules are IMPORTED from `filter_rt_log.py`, the
one home for them -- this file adds only the keep/drop precedence.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import filter_rt_log as frl  # noqa: E402  (same directory, one home for the regexes)

SCRIPT_MARKER = "shorten_rt_log-2026-09-05"
MAX_JSON_LINES = 200

RT_LINE_RE = re.compile(r"^\[rt_log\] ")
TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\)")
USER_CONFIG_RE = re.compile(r"^Loading user config located at")
STEP_RE = re.compile(r"^step (\d+) of")
# THE LINES A REPEATED STARTUP REPORT MUST STILL HAND OVER (2026-09-12).
# Until af45f94 a repeat really was an "identical instrument" and dropping it
# whole was right. It is not any more: since then the report prints the noise
# model's per-episode bias and the grasp belief error, and those carry
# DIFFERENT numbers at every count. `report_at_steps` gained a fourth count
# past the episode cap for exactly one purpose -- so a reader can hold the
# late printout against the early one and see whether the bias is re-drawn or
# accumulating. Dropping the repeat deletes the only evidence for that.
# The step line rides along so the kept numbers can be told apart.
REPORT_KEEP_RE = re.compile(
    r"^(step \d+ of|observation bias |grasp belief error )"
)
ANSI = "\x1b["


def report_blocks(lines: list[str]) -> dict[int, int]:
    """{start: end} (0-based, end-exclusive) of every startup report.

    A block starts at the divider before `REPORT_MARKER` (or at the marker)
    and ends after the NEXT divider, or before the next `[rt_log]` line,
    whichever comes first. No line cap: the real report is ~185 lines and
    `filter_rt_log.MAX_BLOCK_LINES` (150) would cut it mid-way.
    """
    blocks: dict[int, int] = {}
    i = 0
    while i < len(lines):
        if frl.REPORT_MARKER in lines[i]:
            start = i - 1 if i > 0 and frl.DIVIDER_RE.match(lines[i - 1]) else i
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if RT_LINE_RE.match(lines[j]):
                    end = j
                    break
                if frl.DIVIDER_RE.match(lines[j]):
                    end = j + 1
                    break
            blocks[start] = end
            i = end
            continue
        i += 1
    return blocks


def is_noise(line: str) -> bool:
    if not line.strip():
        return True
    if frl.ANOMALY_RE.search(line):
        return False
    return bool(
        frl.NOISE_RE.match(line)
        or USER_CONFIG_RE.match(line)
        or ANSI in line
        or line.startswith("\t")
    )


def shorten(lines: list[str]) -> tuple[list[str], dict]:
    """The short log and a small count dict (kept, dropped, reports_dropped)."""
    blocks = report_blocks(lines)
    out: list[str] = []
    stats = {"kept": 0, "dropped": 0, "reports_dropped": 0}
    seen_report = False
    in_traceback = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if RT_LINE_RE.match(line):
            if frl.RT_CMD_RE.match(line):
                seen_report = False          # a new run: its first report is kept
            if frl.RT_EXIT_RE.match(line):
                in_traceback = False
            out.append(line)
            stats["kept"] += 1
            i += 1
            continue
        if in_traceback:
            out.append(line)
            stats["kept"] += 1
            i += 1
            continue
        if TRACEBACK_RE.match(line):
            in_traceback = True
            out.append(line)
            stats["kept"] += 1
            i += 1
            continue
        if i in blocks:
            end = blocks[i]
            if not seen_report:
                seen_report = True
                for k in range(i, end):
                    if not is_noise(lines[k]):
                        out.append(lines[k])
                        stats["kept"] += 1
                    else:
                        stats["dropped"] += 1
            else:
                step = "?"
                keep = []
                for k in range(i, end):
                    m = STEP_RE.match(lines[k])
                    if m:
                        step = m.group(1)
                    if REPORT_KEEP_RE.match(lines[k]):
                        keep.append(lines[k])
                out.append(f"[shorten] startup report repeat (step {step}) cut to "
                           f"{len(keep)} of {end - i} lines -- the rest repeats the "
                           f"first report; the scatter lines do NOT and are kept")
                out.extend(keep)
                stats["kept"] += len(keep)
                stats["reports_dropped"] += 1
                stats["dropped"] += (end - i) - len(keep)
            i = end
            continue
        if is_noise(line):
            stats["dropped"] += 1
        else:
            out.append(line)
            stats["kept"] += 1
        i += 1
    return out, stats


def compact_json(obj, indent: int = 0) -> list[str]:
    """Dicts one key per line; lists of scalars on ONE line; nested otherwise."""
    pad = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and not _scalar_list(v):
                lines.append(f"{pad}{k}:")
                lines.extend(compact_json(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {json.dumps(v)}")
        return lines
    if isinstance(obj, list):
        if _scalar_list(obj):
            return [f"{pad}{json.dumps(obj)}"]
        lines = []
        for idx, v in enumerate(obj):
            lines.append(f"{pad}- [{idx}]")
            lines.extend(compact_json(v, indent + 1))
        return lines
    return [f"{pad}{json.dumps(obj)}"]


def _scalar_list(v) -> bool:
    return isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v)


def json_section(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return [f"[shorten] json {path} NOT FOUND -- the run wrote no metrics file"]
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"[shorten] json {path} UNREADABLE: {exc}"]
    body = compact_json(obj)
    if len(body) > MAX_JSON_LINES:
        return [f"[shorten] json {path}: {len(body)} compact lines > {MAX_JSON_LINES}, "
                f"NOT appended -- read the script's SUMMARY lines above, or the file"]
    return [f"[shorten] json {path} ({len(body)} lines, lists inline):"] + body


def build(lines: list[str], json_path: pathlib.Path | None, full_path: str) -> list[str]:
    out, stats = shorten(lines)
    if json_path is not None:
        out.extend(json_section(json_path))
    out.append(f"[shorten] {stats['kept']} of {len(lines)} lines kept, "
               f"{stats['reports_dropped']} report repeats dropped ({SCRIPT_MARKER}); "
               f"full log: {full_path}")
    return out


# --------------------------------------------------------------------------
# self-test (D-080: each rule has a mutation that must break it)
# --------------------------------------------------------------------------


def _synthetic_log() -> list[str]:
    report = lambda step: [  # noqa: E731
        "=" * 72,
        "INSERTION STARTUP REPORT -- REAL geometry",
        f"step {step} of (2, 60, 180, 258), t = 0.1 s after reset",
        "kp:           pos 100.0 N/m",
        # The two lines that are NOT the same at every count. Their values
        # differ per step on purpose, so a test that lets the repeat be
        # dropped cannot pass by accident.
        f"observation bias env0..3 (D-182, tip_rel 12:15): [[0.00{step:03d}]]",
        f"grasp belief error env0..3 (D-183): [0.00{step:03d}]",
        "[INFO]: noise inside the report",
        "=" * 72,
    ]
    return (
        ["=" * 62,
         "[rt_log] 2026-09-05 14:22:54  cwd: C:\\x",
         "[rt_log] cmd: python -u scripts/check_seated_success.py --out m.json",
         "[rt_log] git: abc1234",
         "=" * 62,
         "[INFO][AppLauncher]: Using device: cuda:0",
         "Loading user config located at: 'c:/x/user.config'",
         "2026-09-05T12:22:55Z [356ms] [Warning] [omni.usd_config.extension] Enable X",
         "|-----|", "| GPU | Name |", "",
         "\x1b[36m[INFO][IsaacLab]: Logging to file: x\x1b[0m",
         "[INFO]: Base environment:", "\tEnvironment device    : cuda:0",
         "[check_seated_success] marker: check_seated_success-2026-09-03d"]
        + report(2)
        + ["[check_seated_success] curve  h +30.0 mm | y +0.00 mm | kernel_sum 2.235000e-01"]
        + report(60)
        + ["Mean reward: 12.3",              # an unprefixed rsl_rl-style line: must stay
           "2026-09-05T12:23:02Z [7,810ms] [Error] [omni.physx.plugin] bad joint",
           "[check_seated_success] VERDICT: REWARD_CURVE_MEASURED",
           "[rt_log] exit code: 0"]
    )


def _self_test() -> int:
    log = _synthetic_log()
    out, stats = shorten(log)
    text = "\n".join(out)
    checks = [
        ("rt_log header kept", "[rt_log] git: abc1234" in text),
        ("exit line kept", "[rt_log] exit code: 0" in text),
        ("own-script line kept", "marker: check_seated_success-2026-09-03d" in text),
        ("curve row kept", "kernel_sum 2.235000e-01" in text),
        ("unprefixed rsl_rl line kept", "Mean reward: 12.3" in text),
        ("VERDICT kept", "VERDICT: REWARD_CURVE_MEASURED" in text),
        ("[Error] line kept (anomaly beats noise)", "[Error] [omni.physx.plugin] bad joint" in text),
        ("first report kept", "step 2 of (2, 60, 180, 258)" in text and "kp:           pos 100.0 N/m" in text),
        ("report noise dropped", "[INFO]: noise inside the report" not in text),
        ("second report cut, not dropped whole",
         "startup report repeat (step 60) cut to 3 of 8 lines" in text),
        # THE POINT OF THE CUT (2026-09-12): the lines that carry different
        # numbers at every count must survive, or the late printout -- the
        # only one past the episode cap -- reaches nobody.
        ("the repeat keeps its observation bias line",
         "observation bias env0..3 (D-182, tip_rel 12:15): [[0.00060]]" in text),
        ("the repeat keeps its grasp belief line",
         "grasp belief error env0..3 (D-183): [0.00060]" in text),
        ("the repeat keeps its step line, so the numbers can be told apart",
         "step 60 of (2, 60, 180, 258)" in text),
        ("the repeat still drops the lines that DO repeat",
         text.count("kp:           pos 100.0 N/m") == 1),
        ("AppLauncher INFO dropped", "Using device" not in text),
        ("user config dropped", "Loading user config" not in text),
        ("carb warning dropped", "Enable X" not in text),
        ("GPU table dropped", "| GPU | Name |" not in text),
        ("ANSI banner dropped", "Logging to file" not in text),
        ("tab sub-line dropped", "Environment device" not in text),
        ("blank lines dropped", "" not in out),
        ("count: one report repeat", stats["reports_dropped"] == 1),
    ]
    # Mutation 1: a Traceback buried in noise -- everything after it survives.
    tb = log[:-1] + ["Traceback (most recent call last):",
                     "\tFile x, line 1",
                     "2026-09-05T12:23:02Z [1ms] [Warning] [x] after the traceback",
                     "RuntimeError: boom",
                     "[rt_log] exit code: 1"]
    out_tb, _ = shorten(tb)
    t2 = "\n".join(out_tb)
    checks += [
        ("traceback: tab line kept", "\tFile x, line 1" in t2),
        ("traceback: warning after it kept", "after the traceback" in t2),
        ("traceback: error line kept", "RuntimeError: boom" in t2),
        ("traceback: exit 1 kept", "[rt_log] exit code: 1" in t2),
    ]
    # Mutation 2: a second run in the same file -- ITS first report is kept.
    two = log + log
    out_two, st2 = shorten(two)
    checks += [
        ("two runs: two first reports kept", "\n".join(out_two).count("step 2 of") == 2),
        ("two runs: two repeats dropped", st2["reports_dropped"] == 2),
    ]
    # Mutation 3: the rule order -- a [rt_log] line that LOOKS like noise stays.
    out3, _ = shorten(["[rt_log] cmd: python x", "[rt_log] exit code: 0"])
    checks.append(("rt_log lines never drop", len(out3) == 2))
    # Mutation 4: compact JSON -- scalar lists inline, rows expanded, size cap.
    obj = {"marker": "m", "curve_heights_mm": [30.0, 5.0],
           "curve": [{"h": 30.0, "kernel_sum": [0.1, 0.1]}], "verdict_ok": True}
    cj = compact_json(obj)
    checks += [
        ("json: scalar list inline", "curve_heights_mm: [30.0, 5.0]" in cj),
        ("json: row expanded", "- [0]" in [s.strip() for s in cj]),
        ("json: bool", "verdict_ok: true" in cj),
        ("json: size cap fires", json_section_size_cap()),
    ]
    bad = [name for name, ok in checks if not ok]
    n = len(checks)
    if bad:
        print(f"[shorten_rt_log] self-test: FAIL {bad}")
        return 1
    print(f"[shorten_rt_log] self-test: {n}/{n} checks passed ({SCRIPT_MARKER})")
    return 0


def json_section_size_cap() -> bool:
    import tempfile
    big = {"rows": [{"i": i, "v": [1, 2]} for i in range(MAX_JSON_LINES)]}
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "big.json"
        p.write_text(json.dumps(big), encoding="utf-8")
        sec = json_section(p)
        small = pathlib.Path(d) / "small.json"
        small.write_text(json.dumps({"a": [1, 2]}), encoding="utf-8")
        sec_small = json_section(small)
        missing = json_section(pathlib.Path(d) / "none.json")
    return (len(sec) == 1 and "NOT appended" in sec[0]
            and len(sec_small) == 2 and sec_small[1] == "a: [1, 2]"
            and "NOT FOUND" in missing[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", nargs="?", help="full rt_log file")
    ap.add_argument("--out", help="short file to write (default: <log>.short.txt)")
    ap.add_argument("--json", help="metrics JSON the run wrote (--out of the wrapped script)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.log:
        ap.error("log path required (or --self-test)")
    src = pathlib.Path(args.log)
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    dst = pathlib.Path(args.out) if args.out else src.with_suffix(".short.txt")
    out = build(lines, pathlib.Path(args.json) if args.json else None, str(src.resolve()))
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[shorten_rt_log] {len(lines)} -> {len(out)} lines: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
