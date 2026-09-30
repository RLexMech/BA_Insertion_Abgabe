# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Run the counter-proofs. The runner knows no asset (D-080).

WHAT IT DOES
------------
Four modes, three of which need neither Isaac nor USD:

  --offline                 the whole laptop half: the two harness self-tests,
                            plus the static claim audit over every script that
                            declares a Suite. Run this before anything ships.
  --audit-source FILE       the six rules read off one source file, via ast.
  --diff-predicates FILE --against REV
                            proves a migration MOVED the check expressions
                            instead of rewriting them. Compares the normalised
                            source of every check against a git revision.
  --suite NAME|all          the counter-proof itself: take the finished asset,
                            break it in exactly one way per declared mutation,
                            and require the named checks to flip. NEEDS Isaac,
                            so it runs on the training PC only.

WHY --suite IS A SEPARATE COMMAND, not part of every authoring run
------------------------------------------------------------------
An authoring run only DECLARES its mutations; it never applies one. Declaring
costs nothing and adds no stage open, so the normal run is unchanged. The
runner pays the whole cost, on demand, after the asset exists.

SAFETY: the runner edits the PRODUCTION asset
---------------------------------------------
Each mutation opens the subject through ``Sdf.Layer.OpenAsAnonymous``, so the
root layer has no file behind it and an accidental ``Save()`` is impossible.
The runner records the subject's size and modification time before and after
and refuses to report success if either changed.

THE SELF-TEST CONTRACT, written before the implementation
---------------------------------------------------------
``python scripts/selftest_checks.py --self-test`` proves the predicate-diff
extraction, which is what makes a migration reviewable:

   1. the OLD form (a ``checks = {...}`` dict literal) is extracted by name
   2. the NEW form (``suite.add("name", expr, ...)``) is extracted the same way
   3. a check MOVED from the old form to the new one with an unchanged
      expression produces NO difference  -- the case the migration depends on
   4. a check whose expression changed IS reported
   5. a check that disappeared IS reported
   6. a check that appeared IS reported

Usage:

    python scripts/selftest_checks.py --offline
    python scripts/selftest_checks.py --diff-predicates scripts/author_tool_ur5e.py --against HEAD~1
    .\scripts\rt_log.ps1 RT-17 python scripts/selftest_checks.py --suite tool_ur5e
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_MARKER = "selftest_checks-2026-08-24a"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


checks = _load(REPO_ROOT / "scripts" / "tools" / "checks.py", "checks")

# Scripts whose suites the counter-proof can run. A script joins this list only
# when it has been migrated AND its mutations are declared -- an empty entry
# would report a green counter-proof over nothing, which is the silent failure
# mode this whole exercise exists to prevent.
SUITES: dict[str, dict] = {}

# Scripts already migrated to the harness, relative to the repo root. The
# static audit reads exactly these.
MIGRATED: tuple[str, ...] = (
    "scripts/author_tool_ur5e.py",
)


# ===========================================================================
#  Predicate diff -- did the migration MOVE the expressions or rewrite them?
# ===========================================================================
def extract_predicates(source: str, filename: str = "<src>") -> dict[str, str]:
    """Map check name -> normalised source of its boolean expression.

    Understands both shapes so a migration can be compared across the change:
    the old ``checks = {"name": expr, ...}`` literal and the new
    ``suite.add("name", expr, kind=...)`` call.
    """
    out: dict[str, str] = {}
    tree = ast.parse(source, filename)
    for node in ast.walk(tree):
        # old form
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not any(t.endswith("checks") for t in targets):
                continue
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    out[k.value] = ast.unparse(v)
        # new form
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add":
            if len(node.args) >= 2 and isinstance(node.args[0], ast.Constant):
                name = node.args[0].value
                if isinstance(name, str):
                    out[name] = ast.unparse(node.args[1])
    return out


def diff_predicates(path: pathlib.Path, rev: str) -> list[str]:
    rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    old = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    if old.returncode != 0:
        return [f"{checks.VIOLATION}: cannot read {rel} at {rev}: {old.stderr.strip()}"]
    before = extract_predicates(old.stdout, f"{rev}:{rel}")
    after = extract_predicates(path.read_text(encoding="utf-8"), rel)
    return compare_predicates(before, after)


def compare_predicates(before: dict[str, str], after: dict[str, str]) -> list[str]:
    findings: list[str] = []
    for name in sorted(set(before) - set(after)):
        findings.append(f"{checks.VIOLATION}: check {name!r} disappeared.")
    for name in sorted(set(after) - set(before)):
        findings.append(f"{checks.NOTE}: check {name!r} is new.")
    for name in sorted(set(before) & set(after)):
        if before[name] != after[name]:
            findings.append(
                f"{checks.VIOLATION}: check {name!r} changed its expression:\n"
                f"    before: {before[name]}\n"
                f"    after : {after[name]}"
            )
    return findings


# ===========================================================================
#  Offline mode
# ===========================================================================
def _captured_lines(r) -> list:
    """Both streams, never one OR the other.

    A `--self-test` that dies on an uncaught exception prints its PASS lines
    to stdout and the traceback to stderr. `r.stdout or r.stderr` keeps the
    PASS lines and throws the cause away -- that is how RT-108-gate reported
    "all PASS" next to exit 1 (2026-08-31).
    """
    out = (r.stdout or "").strip().splitlines()
    err = (r.stderr or "").strip().splitlines()
    return out + err


def _verdict_line(r) -> str:
    """The one line that carries the verdict, which is the LAST line of stdout.

    Not the last line overall: a check can leave a harmless warning on stderr
    (numpy's "invalid value encountered in arccos" comes out of a mutation
    doing its job) and that warning would then stand where the verdict
    belongs. On a failure the caller dumps `_captured_lines` in full, so
    nothing is lost by picking stdout here.
    """
    out = (r.stdout or "").strip().splitlines()
    return out[-1] if out else (_captured_lines(r) or [""])[-1]


def run_offline() -> int:
    rc = 0
    print(f"[selftest] {RUNNER_MARKER}; repo {REPO_ROOT}")

    for tool in ("checks.py", "usd_predicates.py", "torch_shim.py", "mesh_export.py",
                 "mesh_spacing.py", "mesh_topology.py"):
        p = REPO_ROOT / "scripts" / "tools" / tool
        r = subprocess.run([sys.executable, str(p), "--self-test"], capture_output=True, text=True)
        out = _captured_lines(r)
        print(_verdict_line(r) or f"[selftest] {tool}: no output")
        if r.returncode != 0:
            for line in out:
                print(f"[selftest]   {line}")
            print(f"[selftest] {tool}: FAIL")
            rc = 1

    # Offline check scripts whose --self-test IS their D-080 counter-proof:
    # they mutate their own subject one way at a time and require exactly the
    # named checks to flip. Named explicitly for the same reason MIGRATED is.
    for script in ("check_insertion_math.py", "check_curriculum_ladder.py",
                   "check_env_wiring.py", "check_scripted_insert.py",
                   "check_insertion_sdf.py", "check_fixture_mesh.py",
                   "check_seated_success.py",
                   # Added 2026-09-14 (SBC step 0): until then the AutoDR
                   # counter-proof ran only by hand (scripts/CLAUDE.md noted
                   # the gap).
                   "check_autodr.py",
                   # Added 2026-09-15 with the episode record.
                   "check_episode_log.py"):
        p = REPO_ROOT / "scripts" / script
        r = subprocess.run([sys.executable, str(p), "--self-test"], capture_output=True, text=True)
        out = _captured_lines(r)
        print(f"[selftest] {script}: {_verdict_line(r) or 'no output'}")
        if r.returncode != 0:
            for line in out:
                print(f"[selftest]   {line}")
            rc = 1

    # The static audit, over the scripts that have been migrated. Named
    # explicitly rather than sniffed out of the source: grepping for "Suite("
    # also hits this file's own test fixtures, and an audit that reports PASS
    # over a file it did not really read is the silent failure mode again.
    if not MIGRATED:
        print("[selftest] claim audit: no migrated script yet (nothing to audit)")
    for p in (REPO_ROOT / m for m in MIGRATED):
        findings = checks.audit_source(p, REPO_ROOT)
        for f in findings:
            print(f"[selftest] {f}")
        bad = [f for f in findings if f.startswith(checks.VIOLATION)]
        print(f"[selftest] claim audit {p.name}: {'PASS' if not bad else f'FAIL ({len(bad)})'}")
        if bad:
            rc = 1

    print(f"[selftest] offline: {'PASS' if rc == 0 else 'FAIL'}")
    return rc


# ===========================================================================
#  Counter-proof mode (needs Isaac; training PC only)
# ===========================================================================
def run_counter_proof(which: str) -> int:
    if not SUITES:
        print("[selftest] no suite is registered yet -- nothing to counter-prove.")
        print("[selftest] a script joins SUITES only once it is migrated AND declares its mutations;")
        print("[selftest] an empty entry would report a green counter-proof over nothing.")
        return 1
    raise NotImplementedError("registered in stage 2")


# ===========================================================================
#  Self-test of the diff extraction
# ===========================================================================
OLD_FORM = '''
def verify(stage, before):
    checks = {
        "units_are_metres": abs(after["meters_per_unit"] - 1.0) < 1e-9,
        "up_axis_is_z": after["up_axis"] == "Z",
        "one_new_joint": after["joint_count"] == before["joint_count"] + 1,
    }
    return checks
'''

NEW_FORM_FAITHFUL = '''
def build_suite(stage, before):
    s = checks.Suite("author_tool", "x")
    s.add("units_are_metres", abs(after["meters_per_unit"] - 1.0) < 1e-9, kind=ARITHMETIC, why="w")
    s.add("up_axis_is_z", after["up_axis"] == "Z", kind=API_SHAPE, why="w")
    s.add("one_new_joint", after["joint_count"] == before["joint_count"] + 1, kind=INVARIANCE, why="w")
    return s
'''

NEW_FORM_DRIFTED = '''
def build_suite(stage, before):
    s = checks.Suite("author_tool", "x")
    s.add("units_are_metres", abs(after["meters_per_unit"] - 1.0) < 1e-6, kind=ARITHMETIC, why="w")
    s.add("one_new_joint", after["joint_count"] == before["joint_count"] + 1, kind=INVARIANCE, why="w")
    s.add("brand_new", True, kind=ARITHMETIC, why="w")
    return s
'''


def _self_test() -> int:
    checked = 0
    old = extract_predicates(OLD_FORM)
    new = extract_predicates(NEW_FORM_FAITHFUL)

    # 1 / 2
    assert set(old) == {"units_are_metres", "up_axis_is_z", "one_new_joint"}, old
    checked += 1
    assert set(new) == set(old), new
    checked += 1

    # 3 -- the case the whole migration rests on
    assert compare_predicates(old, new) == [], compare_predicates(old, new)
    checked += 1

    drifted = extract_predicates(NEW_FORM_DRIFTED)
    findings = compare_predicates(old, drifted)
    # 4 / 5 / 6
    assert any("changed its expression" in f and "units_are_metres" in f for f in findings), findings
    checked += 1
    assert any("disappeared" in f and "up_axis_is_z" in f for f in findings), findings
    checked += 1
    assert any("is new" in f and "brand_new" in f for f in findings), findings
    checked += 1

    # 7 -- the swallowed-traceback case (RT-108-gate, 2026-08-31): a run that
    # prints PASS lines AND dies must show the cause, not only the PASS lines.
    class _R:
        stdout = "[check] A PASS\n[check] B PASS\n"
        stderr = "Traceback (most recent call last):\nRuntimeError: boom\n"
    lines = _captured_lines(_R())
    assert lines[-1] == "RuntimeError: boom", lines
    assert "[check] A PASS" in lines, lines
    checked += 1

    # 8 -- the verdict must survive a harmless warning on stderr.
    assert _verdict_line(_R()) == "[check] B PASS", _verdict_line(_R())
    checked += 1

    assert checked == 8, f"only {checked} of 8 cases ran"
    print(f"[selftest] self-test: {checked}/8 cases passed ({RUNNER_MARKER})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the check counter-proofs (D-080).")
    parser.add_argument("--offline", action="store_true", help="Everything that needs no Isaac and no USD.")
    parser.add_argument("--self-test", action="store_true", help="Prove the predicate-diff extraction.")
    parser.add_argument("--audit-source", type=str, default=None, help="Run the six rules over one .py file.")
    parser.add_argument("--diff-predicates", type=str, default=None,
                        help="Compare every check expression in this file against a git revision.")
    parser.add_argument("--against", type=str, default="HEAD~1", help="Revision for --diff-predicates.")
    parser.add_argument("--suite", type=str, default=None, help="Counter-proof one suite, or 'all'. Needs Isaac.")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if args.audit_source:
        findings = checks.audit_source(args.audit_source, REPO_ROOT)
        for f in findings:
            print(f"[selftest] {f}")
        bad = [f for f in findings if f.startswith(checks.VIOLATION)]
        print(f"[selftest] audit-source {args.audit_source}: {'PASS' if not bad else f'FAIL ({len(bad)})'}")
        return 1 if bad else 0
    if args.diff_predicates:
        findings = diff_predicates(pathlib.Path(args.diff_predicates), args.against)
        for f in findings:
            print(f"[selftest] {f}")
        bad = [f for f in findings if f.startswith(checks.VIOLATION)]
        print(f"[selftest] diff-predicates {args.diff_predicates} vs {args.against}: "
              f"{'PASS' if not bad else f'FAIL ({len(bad)})'}")
        return 1 if bad else 0
    if args.suite:
        return run_counter_proof(args.suite)
    if args.offline:
        return run_offline()

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
