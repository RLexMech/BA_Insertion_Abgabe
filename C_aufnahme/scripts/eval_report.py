# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""ONE command for the whole evaluation report (eval_plan.md section 8b):
collect -> stats -> figures, plus ``report_manifest.json``. Laptop only, in
``.venv-analysis`` (D-189).

The collector compares ``code_commit`` over EVERY run it finds, and
``results/eval/technik/`` can contain runs from different code versions.
The chain therefore runs on a
temporary COPY of exactly the four evaluation blocks, plus the two Fixed-DR
blocks ``T-nominal-F`` / ``T-final-F`` (D-192) when BOTH exist -- one without
the other is refused; nothing in ``technik/`` is read, moved or changed. Its outputs are copied back into ``--results``:
``episodes_all.csv``, ``collect_report.json``, ``summary.json``,
``summary_tables.csv``. The figures are drawn from there.

``report_manifest.json`` names every input by SHA-256 (all files of the staged
blocks, the training folder), every output, the library versions and the git
commit, so a figure in the thesis can be traced to the bytes it came from.

    .venv-analysis/Scripts/python.exe scripts/eval_report.py
    .venv-analysis/Scripts/python.exe scripts/eval_report.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile

SCRIPT_MARKER = "eval_report-2026-09-23b"
REPO = pathlib.Path(__file__).resolve().parents[1]
BLOCKS = ("T-nominal", "T-auswahl", "T-final", "T-grenze")
FIXED_BLOCKS = ("T-nominal-F", "T-final-F")
COPY_BACK = ("episodes_all.csv", "collect_report.json", "summary.json", "summary_tables.csv")


class ReportError(RuntimeError):
    """The report cannot be built from what is there."""


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage(results: pathlib.Path, tmp: pathlib.Path) -> list:
    """Copy exactly BLOCKS (+ both FIXED_BLOCKS when there) into ``tmp``."""
    missing = [b for b in BLOCKS if not (results / b).is_dir()]
    if missing:
        raise ReportError(f"blocks missing under {results}: {missing}")
    fixed = [b for b in FIXED_BLOCKS if (results / b).is_dir()]
    if 0 < len(fixed) < len(FIXED_BLOCKS):
        raise ReportError(f"Fixed-DR blocks: only {fixed} under {results}, want both {list(FIXED_BLOCKS)}")
    for b in (*BLOCKS, *fixed):
        shutil.copytree(results / b, tmp / b)
    return sorted(p.name for p in tmp.iterdir() if p.is_dir())


def hash_tree(root: pathlib.Path, base: pathlib.Path) -> dict:
    return {p.relative_to(base).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _run(args: list) -> None:
    print(f"[eval_report] $ {' '.join(str(a) for a in args)}")
    r = subprocess.run([sys.executable, *map(str, args)], cwd=REPO)
    if r.returncode != 0:
        raise ReportError(f"{args[0]} exited {r.returncode}")


def _git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip()


def build(results: pathlib.Path, training: pathlib.Path, figures: pathlib.Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        staged = stage(results, t)
        _run(["scripts/eval_collect.py", "--results", t])
        _run(["scripts/eval_stats.py", "--results", t])
        for name in COPY_BACK:
            if not (t / name).is_file():
                raise ReportError(f"the chain wrote no {name}")
            shutil.copyfile(t / name, results / name)
    _run(["scripts/eval_figures.py", "--results", results, "--training", training, "--out", figures])
    inputs = {}
    for b in staged:
        inputs.update(hash_tree(results / b, REPO))
    inputs.update(hash_tree(training, REPO))
    outputs = {(results / n).relative_to(REPO).as_posix(): sha256(results / n) for n in COPY_BACK}
    outputs.update(hash_tree(figures, REPO))
    import matplotlib, numpy, pandas, scipy, statsmodels   # noqa: E401 - versions only
    manifest = {
        "marker": SCRIPT_MARKER, "blocks": staged,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_paths": [ln for ln in _git("status", "--porcelain", "--untracked-files=no").splitlines()],
        "python": platform.python_version(),
        "versions": {m.__name__: m.__version__ for m in (matplotlib, numpy, pandas, scipy, statsmodels)},
        "inputs": inputs, "outputs": outputs,
    }
    with open(results / "report_manifest.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


# Self-test (D-080): checks on built data; each mutation of THIS file's source

CHECKS = (
    "stage copies exactly the four blocks, never technik",
    "stage refuses when a block is missing",
    "hash_tree = hashlib over each file's bytes, keyed by path",
    "stage takes both Fixed-DR blocks when both are there",
    "stage refuses when only one Fixed-DR block is there",
)


def _run_checks(ns) -> dict:
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        src = t / "results"
        for b in (*BLOCKS, "technik"):
            (src / b / "autodr_s1").mkdir(parents=True)
            (src / b / "autodr_s1" / "x.txt").write_text(b, encoding="utf-8")
        dst = t / "stage"
        dst.mkdir()
        try:
            res[CHECKS[0]] = ns["stage"](src, dst) == sorted(BLOCKS)
        except Exception:   # noqa: BLE001
            res[CHECKS[0]] = False
        shutil.rmtree(src / "T-grenze")
        dst2 = t / "stage2"
        dst2.mkdir()
        try:
            ns["stage"](src, dst2)
            res[CHECKS[1]] = False
        except Exception as exc:   # noqa: BLE001 - ReportError of the executed copy
            res[CHECKS[1]] = "blocks missing" in str(exc) and not (dst2 / "T-nominal").exists()
        got = ns["hash_tree"](src / "T-final", t)
        want = {"results/T-final/autodr_s1/x.txt": hashlib.sha256(b"T-final").hexdigest()}
        res[CHECKS[2]] = got == want
        full = t / "full"
        for b in (*BLOCKS, "T-nominal-F", "T-final-F", "technik"):
            (full / b / "x").mkdir(parents=True)
        dst3 = t / "stage3"
        dst3.mkdir()
        try:
            res[CHECKS[3]] = ns["stage"](full, dst3) == sorted((*BLOCKS, "T-nominal-F", "T-final-F"))
        except Exception:   # noqa: BLE001
            res[CHECKS[3]] = False
        shutil.rmtree(full / "T-nominal-F")
        dst4 = t / "stage4"
        dst4.mkdir()
        try:
            ns["stage"](full, dst4)
            res[CHECKS[4]] = False
        except Exception as exc:   # noqa: BLE001
            res[CHECKS[4]] = "Fixed-DR blocks" in str(exc) and not (dst4 / "T-nominal").exists()
    return res


MUTATIONS = (
    ("technik-staged-too", 'BLOCKS = ("T-nominal", "T-auswahl", "T-final", "T-grenze")',
     'BLOCKS = ("T-nominal", "T-auswahl", "T-final", "T-grenze", "technik")', (CHECKS[0], CHECKS[3])),
    ("missing-block-guard-dropped", "    if missing:\n        raise ReportError",
     "    if False:\n        raise ReportError", (CHECKS[1],)),
    ("hash-of-the-path-not-the-bytes", "return {p.relative_to(base).as_posix(): sha256(p) for p",
     "return {p.relative_to(base).as_posix(): hashlib.sha256(str(p).encode()).hexdigest() for p",
     (CHECKS[2],)),
    ("fixed-dr-blocks-never-staged", 'FIXED_BLOCKS = ("T-nominal-F", "T-final-F")', "FIXED_BLOCKS = ()",
     (CHECKS[3], CHECKS[4])),
    ("half-fixed-dr-guard-dropped", "    if 0 < len(fixed) < len(FIXED_BLOCKS):", "    if False:",
     (CHECKS[4],)),
)
_SELF_TEST_MARKER = "# Self-test (D-080): checks on built data"


def self_test() -> int:
    print(f"[eval_report] marker: {SCRIPT_MARKER}")
    src = pathlib.Path(__file__).read_text(encoding="utf-8")

    def _exec(text):
        ns = {"__name__": "eval_report_under_test", "__file__": __file__}
        exec(compile(text, __file__, "exec"), ns)  # noqa: S102 - the point of the self-test
        return ns

    base = _run_checks(_exec(src))
    bad = 0
    for name in CHECKS:
        ok = base.get(name) is True
        print(f"[eval_report]   {'PASS' if ok else 'FAIL'}  {name}")
        bad += not ok
    cut = src.index(_SELF_TEST_MARKER)
    head, tail = src[:cut], src[cut:]
    for mname, old, new, want in MUTATIONS:
        if head.count(old) != 1:
            print(f"[eval_report]   mutation {mname}: anchor matches {head.count(old)} times -- FAIL")
            bad += 1
            continue
        got = _run_checks(_exec(head.replace(old, new) + tail))
        flipped = {k for k, v in got.items() if v is not True}
        ok = flipped == set(want)
        print(f"[eval_report]   mutation {mname}: {'PASS' if ok else 'FAIL'}"
              + ("" if ok else f" (flipped {sorted(flipped)}, declared {sorted(want)})"))
        bad += not ok
    undefended = [c for c in CHECKS if not any(c in w for *_, w in MUTATIONS)]
    print(f"[eval_report]   every check is flipped by some mutation: "
          f"{'PASS' if not undefended else 'FAIL ' + str(undefended)}")
    bad += bool(undefended)
    total = len(CHECKS) + len(MUTATIONS) + 1
    print(f"[eval_report] self-test: {total - bad}/{total}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--results", type=str, default="results/eval")
    ap.add_argument("--training", type=str, default="results/training")
    ap.add_argument("--figures", type=str, default="docs/figures/eval")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    print(f"[eval_report] marker: {SCRIPT_MARKER}")
    m = build(pathlib.Path(args.results).resolve(), pathlib.Path(args.training).resolve(),
              pathlib.Path(args.figures).resolve())
    print(f"[eval_report] blocks {m['blocks']}; {len(m['inputs'])} inputs, {len(m['outputs'])} outputs hashed; "
          f"git {m['git_commit'][:8]}{' DIRTY' if m['git_dirty_paths'] else ''}")
    print(f"[eval_report] wrote {pathlib.Path(args.results) / 'report_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
