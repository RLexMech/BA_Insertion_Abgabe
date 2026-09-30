# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Offline check of ``curriculum.py`` -- the manual multi-axis ladder (D-110).

The ladder is plain Python: no torch, no Isaac, no tensors. It is therefore
the one piece of this stream that the laptop can import for real rather than
execute under a stand-in. The mutation harness still loads it from SOURCE,
because the counter-proof needs a broken copy without touching the file.

    python scripts/check_curriculum_ladder.py
    python scripts/check_curriculum_ladder.py --self-test
"""

from __future__ import annotations

import pathlib
import sys

LADDER_SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "source"
    / "insertion"
    / "insertion"
    / "tasks"
    / "direct"
    / "insertion"
    / "curriculum.py"
)

# Probe values. The real step sizes are [open] -- D-110 point (3) says they
# come from this project's own learning curves, not from IndustReal's 5 mm.
# Nothing here asserts a step size; the checks only exercise the rule.
AXES = ("start_height", "fixture_offset", "tilt")
BASE = {"start_height": 0.005, "fixture_offset": 0.001, "tilt": 0.01}
STEP = {"start_height": 0.010, "fixture_offset": 0.002, "tilt": 0.02}


def load_ladder(mutations: tuple[tuple[str, str], ...] = ()) -> dict:
    src = LADDER_SRC.read_text(encoding="utf-8")
    for old, new in mutations:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"mutation {old!r} matches {n} times in {LADDER_SRC.name}, expected 1")
        src = src.replace(old, new)
    ns: dict = {"__name__": "curriculum"}
    exec(compile(src, str(LADDER_SRC), "exec"), ns)  # noqa: S102 - the point of the file
    return ns


def build_checks(ns) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    def check(name: str, cond, detail: str = "") -> None:
        out.append((name, bool(cond), detail))

    Ladder = ns["Ladder"]
    UNSET = ns["UNSET"]
    LadderError = ns["LadderError"]

    def fresh(max_rung=4):
        return Ladder(axes=AXES, base=BASE, step=STEP, max_rung=max_rung)

    # -- the UNSET guard: step sizes are [open] (D-110 point 3) -------------
    # LadderError SPECIFICALLY, not "some exception". A ladder that crashes
    # with a TypeError three lines deeper has also refused to build, but it
    # refused by accident and says nothing about which number is missing --
    # which is the whole job of the guard.
    def refuses(name: str, **kwargs) -> None:
        try:
            Ladder(axes=AXES, max_rung=4, **kwargs)
        except LadderError as exc:
            check(name, True, str(exc)[:60])
        except Exception as exc:  # noqa: BLE001 - a crash is not a refusal
            check(name, False, f"raised {type(exc).__name__} instead of LadderError")
        else:
            check(name, False, "no error raised")

    refuses("an unset step size refuses to build a ladder", base=BASE, step=UNSET)
    refuses("ONE unset axis is enough to refuse", base=BASE, step={**STEP, "tilt": UNSET})
    # A step that shrinks the range would SHIFT the ladder instead of widening
    # it -- the one hard constraint of D-110 point (2).
    refuses("a negative step is refused (widen, never shift)",
            base=BASE, step={**STEP, "tilt": -0.01})

    # -- the 80/10 rule (D-110 point 3, IndustReal PAPER thresholds) --------
    lad = fresh()
    check("a fresh ladder starts on rung 0", lad.rung == 0, str(lad.rung))
    check("rung 0 carries the base ranges", lad.ranges == BASE, str(lad.ranges))

    check("81 % advances", fresh().update(0.81) == "advance")
    check("exactly 80 % does NOT advance -- the threshold is strict",
          fresh().update(0.80) == "hold")
    check("50 % holds", fresh().update(0.50) == "hold")
    check("exactly 10 % holds -- the lower threshold is strict too",
          fresh().update(0.10) == "hold")
    lad = fresh()
    lad.update(0.9)
    check("9 % retreats once there is a rung to retreat to",
          lad.update(0.09) == "retreat")

    lad = fresh()
    lad.update(0.9)
    check("advancing moves one rung, not more", lad.rung == 1, str(lad.rung))

    # -- widen, never shift --------------------------------------------------
    lad = fresh()
    seen = [dict(lad.ranges)]
    for _ in range(4):
        lad.update(0.9)
        seen.append(dict(lad.ranges))
    check("the ladder stops at max_rung", lad.rung == 4, str(lad.rung))
    check("advancing past the top holds instead of overflowing",
          lad.update(0.99) == "hold" and lad.rung == 4, str(lad.rung))
    widened = all(
        seen[i + 1][a] > seen[i][a] for i in range(len(seen) - 1) for a in AXES
    )
    check("every axis widens on every advance -- multi-axis, together", widened)
    contains = all(
        seen[i + 1][a] >= seen[i][a] for i in range(len(seen) - 1) for a in AXES
    )
    check("each rung CONTAINS the previous one (widen, never shift)", contains)
    check("the top rung is base + 4 steps",
          abs(seen[-1]["tilt"] - (BASE["tilt"] + 4 * STEP["tilt"])) < 1e-12,
          f"{seen[-1]['tilt']:.4f}")

    # -- retreat: one rung back AND half the step ---------------------------
    lad = fresh()
    lad.update(0.9)
    lad.update(0.9)
    at_two = dict(lad.ranges)
    at_one = dict(seen[1])
    check("two advances reach rung 2", lad.rung == 2, str(lad.rung))
    ev = lad.update(0.05)
    check("under 10 % retreats", ev == "retreat", ev)
    check("retreat goes exactly ONE rung back", lad.rung == 1, str(lad.rung))
    check("retreat restores the previous rung's ranges", lad.ranges == at_one,
          f"{lad.ranges} vs {at_one}")
    lad.update(0.9)
    half = {a: at_one[a] + STEP[a] / 2.0 for a in AXES}
    check("the re-widening after a retreat uses HALF the step",
          all(abs(lad.ranges[a] - half[a]) < 1e-12 for a in AXES),
          f"{lad.ranges} vs {half}")
    check("the halved re-widening is narrower than the original rung 2",
          all(lad.ranges[a] < at_two[a] for a in AXES))

    lad.update(0.05)
    lad.update(0.9)
    quarter = {a: at_one[a] + STEP[a] / 4.0 for a in AXES}
    check("a second retreat halves the step again",
          all(abs(lad.ranges[a] - quarter[a]) < 1e-12 for a in AXES),
          f"{lad.ranges}")

    # Retreating off the bottom is impossible. Named limitation rather than an
    # invented rule: the ladder holds at rung 0 and says so, so a human sees
    # that the FIRST rung is too hard instead of the ladder quietly shrinking.
    lad = fresh()
    ev = lad.update(0.01)
    check("retreat at rung 0 is reported, not invented", ev == "retreat_blocked", ev)
    check("retreat at rung 0 leaves the rung alone", lad.rung == 0, str(lad.rung))
    check("retreat at rung 0 leaves the step alone", lad.step_scale == 1.0,
          str(lad.step_scale))

    # -- telemetry (plan step 4): every run says which rung it is on --------
    lad = fresh()
    lad.update(0.9)
    d = lad.as_dict()
    check("as_dict reports the rung", d.get("rung") == 1, str(d))
    check("as_dict reports the step scale", d.get("step_scale") == 1.0, str(d))
    check("as_dict reports every axis range", all(a in d["ranges"] for a in AXES), str(d))

    return out


MUTATIONS: tuple[tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]], ...] = (
    (
        "advance-threshold-inclusive",
        (("if success_rate > self.advance_at:", "if success_rate >= self.advance_at:"),),
        ("exactly 80 % does NOT advance -- the threshold is strict",),
    ),
    (
        "retreat-threshold-inclusive",
        (("if success_rate < self.retreat_at:", "if success_rate <= self.retreat_at:"),),
        ("exactly 10 % holds -- the lower threshold is strict too",),
    ),
    (
        "retreat-forgets-to-halve",
        (("self.step_scale = self.step_scale * RETREAT_STEP_FACTOR", "self.step_scale = self.step_scale"),),
        (
            "the re-widening after a retreat uses HALF the step",
            "a second retreat halves the step again",
            "the halved re-widening is narrower than the original rung 2",
        ),
    ),
    (
        "advance-jumps-two-rungs",
        (("self._rung += 1", "self._rung += 2"),),
        (
            "advancing moves one rung, not more",
            "the top rung is base + 4 steps",
            "as_dict reports the rung",
            "two advances reach rung 2",
            "retreat goes exactly ONE rung back",
            "every axis widens on every advance -- multi-axis, together",
            "a second retreat halves the step again",
        ),
    ),
    (
        "ladder-shifts-instead-of-widening",
        (("width = base_width + self._steps_taken[axis]",
          "width = self._steps_taken[axis]"),),
        ("rung 0 carries the base ranges", "the top rung is base + 4 steps"),
    ),
    (
        "unset-step-slips-through",
        (("if value is UNSET:", "if False:"),),
        ("ONE unset axis is enough to refuse",),
    ),
    (
        "negative-step-slips-through",
        (("if value <= 0.0:", "if False:"),),
        ("a negative step is refused (widen, never shift)",),
    ),
)


def _run(mutations=()) -> dict[str, bool]:
    return {name: ok for name, ok, _ in build_checks(load_ladder(mutations))}


def _self_test() -> int:
    print("[check_curriculum_ladder] counter-proof (D-080)")
    baseline = _run()
    rc = 0
    if not all(baseline.values()):
        bad = [k for k, v in baseline.items() if not v]
        print(f"  FAIL  baseline is not green, counter-proof is meaningless: {bad}")
        return 1
    print(f"  PASS  baseline: {len(baseline)} checks green")

    for name, muts, expected in MUTATIONS:
        try:
            got = _run(muts)
        except Exception as exc:  # noqa: BLE001 - a crash is a flip of everything
            print(f"  FAIL  {name}: the mutated ladder raised {type(exc).__name__}: {exc}")
            rc = 1
            continue
        # A check that DISAPPEARED counts as flipped: a mutation can remove the
        # branch a check lived in, and "it never ran" is not "it passed".
        flipped = {k for k in baseline if k not in got or not got[k]}
        unknown = set(expected) - set(baseline)
        missing = set(expected) - flipped
        extra = flipped - set(expected)
        if unknown:
            print(f"  FAIL  {name}: names no existing check {sorted(unknown)}")
            rc = 1
        elif not flipped:
            print(f"  FAIL  {name}: flipped NOTHING -- these checks can never fail")
            rc = 1
        elif missing or extra:
            print(f"  FAIL  {name}: missing {sorted(missing)} extra {sorted(extra)}")
            rc = 1
        else:
            print(f"  PASS  {name}: flips exactly {len(flipped)} declared check(s)")

    print()
    print("COUNTER-PROOF PASSED" if rc == 0 else "COUNTER-PROOF FAILED")
    return rc


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    results = build_checks(load_ladder())
    failures = []
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)
    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print(f"ALL {len(results)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
