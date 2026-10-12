# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""A named check that must prove it can distinguish -- the harness (D-080).

WHY THIS EXISTS
---------------
Run RT-13 (2026-08-23) welded the tool into the UR5e correctly and was refused
by two of its own twenty checks. Both checks were wrong, not the asset:

  * ``diagonal_inertia_left_to_physx`` asserted ``Get() is None``. USD returns
    the SCHEMA FALLBACK for an unauthored attribute, and ``UsdPhysics`` gives
    ``physics:diagonalInertia`` a fallback of (0,0,0) -- itself the sentinel
    that asks PhysX to compute the tensor. The assertion could never pass.
  * ``self_contained`` asserted the composition-dependency list is empty. A
    flattened layer always reports exactly one dependency whose path is the
    empty string; measured in RT-15, the SHIPPED UR5e does the same with no
    tool involved. Also could never pass.

Neither mistake was findable by reading. Both cost a round trip to the training
machine. And the CORRECT form of the second one already existed in this repo,
in ``verify_fixture_usd.py`` -- so it was written wrong twice while a working
version sat three files away.

A check is only a check if it can DISTINGUISH: it needs a world where it says
PASS and a world where it says FAIL. Two ways to lose that, and the second is
the dangerous one:

  can never pass  -> blocks a run. Visible, costs one round trip.
  can never fail  -> certifies nothing, and NOBODY EVER FINDS OUT.

WHAT THIS MODULE DOES
---------------------
Three legs, and what each one actually buys (measured, not asserted):

  A  CLAIM KIND      every check declares what KIND of claim it makes, the way
                     every constant in insertion_tasks_cfg.py already declares
                     its provenance (M/R/A/P). An ``api-shape`` claim -- "Get()
                     returns None", "the list is empty" -- is not knowable by
                     reading and must be paid for once by a PROBE run.
                     Catches "can never pass" BEFORE the expensive script runs.
  B  MUTATION        the counter-proof. Break the KNOWN-GOOD asset in exactly
                     one way and require the named checks to flip. The ONLY leg
                     that catches "can never fail".
  C  EXPECT-BEFORE   what the check must report on the UNMODIFIED input. A check
                     that already passes there is testing the input, not the
                     work. Would have caught the inertia bug: the old assertion
                     reads True on a robot that has no tool_link at all.

This module is STDLIB ONLY -- no pxr, no isaaclab, no numpy. It stores names,
booleans and bookkeeping; every USD call stays in the caller and arrives here
already evaluated. That is deliberate: it makes the harness itself testable on
the dev laptop, where ``import pxr`` fails. ``scripts/tools/`` is the repo's
"offline tools (no Isaac needed)" directory and is not a package -- load this
by path with ``importlib.util.spec_from_file_location``, as
``check_demo_reward_math.py`` already does for ``torch_shim.py``.

THE SELF-TEST CONTRACT, written before the implementation
---------------------------------------------------------
``python scripts/tools/checks.py --self-test`` must prove all of the following
with invented checks and invented mutations, and NO USD anywhere:

   1. an api-shape check with no probe is reported            (rule 1)
   2. an api-shape check WITH a probe is not reported         (rule 1, control)
   3. a probe naming a file that is not on disk is reported   (rule 4)
   4. an invariance check with expect_before=PASS is reported (rule 3)
   5. an api-shape check with no mutation is reported         (rule 2)
   6. Unprovable(reason) satisfies rule 2 but is announced    (rule 2, escape)
   7. a mutation whose flips name an unknown check is reported(rule 5)
   8. a mutation with an empty flips tuple is reported        (rule 5)
   9. two mutations sharing a name are reported               (rule 5)
  10. a suite in which NO check expects FAIL before is reported(rule 6)
  11. ``no_before_state=True`` suppresses rule 6 only         (rule 6, control)
  12. a mutation that flips NOTHING fails the counter-proof   (can never fail)
  13. a mutation that flips an EXTRA check fails it too       (not surgical)
  14. a mutation that flips exactly its declared set passes
  15. compare_before reports a check that passes before authoring
  16. compare_before is silent when every check matches its declaration
  17. as_dict() is byte-identical to the plain dict[str, bool] it replaces
  18. a check whose thunk raises reports ERROR, and ERROR is not PASS
  19. the RT-13 inertia bug, replayed: the OLD assertion evaluated on a
      before-state dict reads True, so compare_before catches it
  20. a clean suite produces no findings at all                (whole-file control)
  21. a preservation check that expects FAIL before is reported (rule 3, mirror)
  22. a preservation check with no mutation is reported          (rule 2)
  23. a preservation check that expects PASS before and carries a mutation is
      clean                                                      (rule 3, control)
  24. audit_source resolves the DOTTED constant form (harness.API_SHAPE) that
      the migrated scripts actually write -- reading only bare names made the
      static audit fire no rule at all and pass every file

The last one matters as much as the rest: a harness that flags everything is
as useless as one that flags nothing.

Cases 21-23 exist because the first real suite produced a check the rules got
wrong: ``articulation_roots_unchanged`` is SUPPOSED to pass before and after --
that is the claim (D-018/D-019, the Robot Assembler stripping the
ArticulationRootAPI). The kinds were split into INVARIANCE (something must
change) and PRESERVATION (something must not) rather than weakening the rule
that caught it.

Usage:

    python scripts/tools/checks.py --self-test
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable

CHECKS_MARKER = "checks-2026-08-24c"

# --- claim kinds -----------------------------------------------------------
ARITHMETIC = "arithmetic"
API_SHAPE = "api-shape"
INVARIANCE = "invariance"
PRESERVATION = "preservation"
KINDS = (ARITHMETIC, API_SHAPE, INVARIANCE, PRESERVATION)

# --- outcomes --------------------------------------------------------------
PASS = "PASS"
FAIL = "FAIL"
NA = "n/a"
ERROR = "ERROR"

VIOLATION = "VIOLATION"
NOTE = "NOTE"


@dataclass(frozen=True)
class Probe:
    """What paid for an api-shape claim: one run that MEASURED the API's shape.

    ``script`` is checked against the repo, so a probe cannot outlive the file
    that produced it. ``found`` is free text and is not machine-checkable --
    same standing as the M/R/A/P provenance tags on the constants. The LINK is
    what this makes checkable; the CLAIM is made checkable by the mutation.
    """

    run: str
    script: str
    question: str
    found: str

    def as_dict(self) -> dict:
        return {"run": self.run, "script": self.script, "question": self.question, "found": self.found}


@dataclass(frozen=True)
class Mutation:
    """Break the known-good subject in exactly ONE way.

    ``apply`` receives an open stage (or whatever the caller's suite builder
    consumes), edits it, and returns a non-empty description of what it did.
    It is DECLARED by the authoring script and only ever CALLED by the runner,
    so declaring a mutation costs the authoring run nothing.

    ``flips`` is the exact set of checks that must turn FAIL. Fewer means the
    check cannot fail. More means the mutation is not surgical and proves
    nothing about the check it names.
    """

    name: str
    apply: Callable[[Any], str]
    flips: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"name": self.name, "flips": list(self.flips)}


@dataclass(frozen=True)
class Unprovable:
    """Escape hatch for a check that genuinely cannot be counter-proved.

    Satisfies the mutation requirement, and is announced every single time so
    it stays a decision rather than an omission.
    """

    reason: str

    def as_dict(self) -> dict:
        return {"unprovable": self.reason}


@dataclass
class _Check:
    name: str
    ok: Any  # bool, or a zero-argument callable returning bool
    kind: str
    why: str
    expect_before: str
    probe: Probe | None
    mutation: Mutation | Unprovable | None
    _cached: str | None = field(default=None, repr=False)


class Suite:
    """A named set of checks that carries WHY it can be trusted.

    ``as_dict()`` returns the plain ``dict[str, bool]`` the scripts used before
    this module existed, so a migration is a move, not a rewrite, and the
    sidecar's ``checks`` key does not change shape.
    """

    def __init__(self, tag: str, subject: str, *, no_before_state: bool = False) -> None:
        self.tag = tag
        self.subject = subject
        self.no_before_state = bool(no_before_state)
        self._checks: dict[str, _Check] = {}
        self._values: dict[str, Any] = {}

    # -- building ----------------------------------------------------------
    def add(
        self,
        name: str,
        ok: Any,
        *,
        kind: str,
        why: str,
        expect_before: str = FAIL,
        probe: Probe | None = None,
        mutation: Mutation | Unprovable | None = None,
    ) -> None:
        if name in self._checks:
            raise ValueError(f"duplicate check name {name!r} in suite {self.tag!r}")
        if kind not in KINDS:
            raise ValueError(f"check {name!r}: kind must be one of {KINDS}, got {kind!r}")
        if expect_before not in (PASS, FAIL, NA):
            raise ValueError(f"check {name!r}: expect_before must be PASS/FAIL/NA, got {expect_before!r}")
        if not why:
            raise ValueError(f"check {name!r}: 'why' must say what the check is for")
        self._checks[name] = _Check(name, ok, kind, why, expect_before, probe, mutation)

    def value(self, name: str, v: Any) -> None:
        """Record a measurement that is not a check. Lands in the sidecar."""
        self._values[name] = v

    # -- reading -----------------------------------------------------------
    def names(self) -> tuple[str, ...]:
        return tuple(self._checks)

    def result(self, name: str) -> str:
        """PASS / FAIL / ERROR. A thunk that raises is ERROR, never PASS."""
        c = self._checks[name]
        if c._cached is None:
            try:
                raw = c.ok() if callable(c.ok) else c.ok
                c._cached = PASS if bool(raw) else FAIL
            except Exception as exc:  # noqa: BLE001 -- an unusable check is a result
                c._cached = ERROR
                self._values.setdefault("_errors", {})[name] = f"{type(exc).__name__}: {exc}"
        return c._cached

    def as_dict(self) -> dict[str, bool]:
        """The legacy shape. ERROR counts as False -- never as a pass."""
        return {n: self.result(n) == PASS for n in self._checks}

    def values(self) -> dict[str, Any]:
        return dict(self._values)

    def claims(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for n, c in self._checks.items():
            entry: dict[str, Any] = {
                "kind": c.kind,
                "why": c.why,
                "expect_before": c.expect_before,
                "result": self.result(n),
            }
            if c.probe is not None:
                entry["probe"] = c.probe.as_dict()
            if isinstance(c.mutation, Mutation):
                entry["mutation"] = c.mutation.as_dict()
            elif isinstance(c.mutation, Unprovable):
                entry["mutation"] = c.mutation.as_dict()
            out[n] = entry
        return out

    def mutations(self) -> tuple[Mutation, ...]:
        return tuple(c.mutation for c in self._checks.values() if isinstance(c.mutation, Mutation))

    def expectations(self) -> dict[str, str]:
        return {n: c.expect_before for n, c in self._checks.items()}

    def unprovable(self) -> dict[str, str]:
        return {
            n: c.mutation.reason
            for n, c in self._checks.items()
            if isinstance(c.mutation, Unprovable)
        }

    # -- verdict -----------------------------------------------------------
    def failed(self) -> tuple[str, ...]:
        return tuple(n for n in self._checks if self.result(n) != PASS)

    def verdict(self) -> str:
        return PASS if not self.failed() else FAIL

    def exit_code(self) -> int:
        """D-081: a check-bearing script exits non-zero when a check fails."""
        return 0 if self.verdict() == PASS else 1

    def print_lines(self) -> None:
        """``[tag] name: PASS|FAIL``.

        The format is constrained by scripts/filter_rt_log.py: the line must
        start with a lowercase ``[tag]`` to be picked up as own-script output,
        and must contain the literal token FAIL on failure to reach the anomaly
        section. Change either and /rt-check goes blind.
        """
        for n in self._checks:
            print(f"[{self.tag}] {n}: {self.result(n)}")
        # The unprovable checks are NOT announced here: audit() already says so,
        # and RT-17 printed each of them twice in the same run.


# --- the six rules ---------------------------------------------------------
def audit(suite: Suite, repo_root: pathlib.Path | str) -> list[str]:
    """Check the CHECKS. Returns findings, prefixed VIOLATION or NOTE.

    An empty list means every claim in the suite is paid for and counter-proved.
    """
    root = pathlib.Path(repo_root)
    findings: list[str] = []
    seen_mutations: dict[str, str] = {}
    names = set(suite.names())

    for n, c in suite._checks.items():  # noqa: SLF001 -- same module
        # rule 1: an api-shape claim is not knowable by reading.
        if c.kind == API_SHAPE and c.probe is None:
            findings.append(
                f"{VIOLATION}: {n} is an {API_SHAPE} claim with no probe. "
                "What the API returns must be MEASURED once before it is asserted."
            )
        # rule 4: a probe cannot outlive the script that produced it.
        if c.probe is not None and not (root / c.probe.script).is_file():
            findings.append(
                f"{VIOLATION}: {n} cites probe {c.probe.run} in {c.probe.script}, "
                "which is not in the repo."
            )
        # rule 3: a claim about CHANGE that already holds before the work is
        # not about the work. Its mirror image: a claim about PRESERVATION that
        # does NOT hold before the work has nothing to preserve.
        #
        # The two kinds were one kind until this rule met a real suite:
        # ``articulation_roots_unchanged`` is supposed to pass before AND after
        # -- that IS the claim (D-018/D-019: the Robot Assembler stripped the
        # ArticulationRootAPI). Folding it in with ``one_new_joint`` made the
        # rule flag a correct check, so the kinds were split rather than the
        # rule weakened.
        if c.kind == INVARIANCE and c.expect_before == PASS:
            findings.append(
                f"{VIOLATION}: {n} is an {INVARIANCE} claim that expects PASS before the work. "
                "Then it is about the input, not about what the script did. "
                f"If it asserts that something must NOT change, its kind is {PRESERVATION}."
            )
        if c.kind == PRESERVATION and c.expect_before != PASS:
            findings.append(
                f"{VIOLATION}: {n} is a {PRESERVATION} claim that expects {c.expect_before} before the work. "
                "A property that does not hold beforehand cannot be preserved."
            )
        # rule 2: it must be shown that the check can fail.
        if c.kind in (API_SHAPE, INVARIANCE, PRESERVATION):
            if c.mutation is None:
                findings.append(
                    f"{VIOLATION}: {n} has no counter-proof. Name a Mutation that must break it, "
                    "or state Unprovable(reason)."
                )
            elif isinstance(c.mutation, Unprovable):
                findings.append(f"{NOTE}: {n} is declared unprovable -- {c.mutation.reason}")
        # rule 5: the counter-proof must itself be well formed.
        if isinstance(c.mutation, Mutation):
            m = c.mutation
            if not m.flips:
                findings.append(f"{VIOLATION}: mutation {m.name!r} flips nothing, so it proves nothing.")
            for target in m.flips:
                if target not in names:
                    findings.append(
                        f"{VIOLATION}: mutation {m.name!r} claims to flip {target!r}, "
                        "which is not a check in this suite."
                    )
            if m.name in seen_mutations and seen_mutations[m.name] != n:
                findings.append(
                    f"{VIOLATION}: mutation name {m.name!r} is used by both "
                    f"{seen_mutations[m.name]!r} and {n!r}."
                )
            seen_mutations.setdefault(m.name, n)

    # rule 6: a suite where nothing is expected to change cannot see the work.
    if not suite.no_before_state and not any(
        c.expect_before == FAIL for c in suite._checks.values()  # noqa: SLF001
    ):
        findings.append(
            f"{VIOLATION}: no check in suite {suite.tag!r} expects FAIL before the work. "
            "The suite cannot tell the input from the result. "
            "Pass no_before_state=True if the script is a generator with no input state."
        )
    return findings


def audit_source(py_file: pathlib.Path | str, repo_root: pathlib.Path | str) -> list[str]:
    """The same rules read off the SOURCE, offline, with no stage and no Isaac.

    Blind spot, stated rather than discovered: checks added in a loop or a
    comprehension are invisible here (fix_fixture_asset.py builds
    ``direct_extent_{axis}`` that way). The runtime ``audit()`` covers exactly
    those, which is why both are kept.
    """
    path = pathlib.Path(py_file)
    root = pathlib.Path(repo_root)
    findings: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add"):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if "kind" not in kw:
            continue
        kind = _literal(kw.get("kind"))
        name = _literal(node.args[0]) if node.args else "<computed>"
        if kind == API_SHAPE and "probe" not in kw:
            findings.append(f"{VIOLATION}: {path.name}: {name} is {API_SHAPE} with no probe=")
        if kind in (API_SHAPE, INVARIANCE) and "mutation" not in kw:
            findings.append(f"{VIOLATION}: {path.name}: {name} has no mutation=")
        if kind == INVARIANCE and _literal(kw.get("expect_before")) == PASS:
            findings.append(f"{VIOLATION}: {path.name}: {name} is {INVARIANCE} and expects PASS before")
        probe = kw.get("probe")
        if isinstance(probe, ast.Call):
            script = None
            for k in probe.keywords:
                if k.arg == "script":
                    script = _literal(k.value)
            if script and not (root / script).is_file():
                findings.append(f"{VIOLATION}: {path.name}: {name} cites a probe script not in the repo: {script}")
    return findings


_CONSTANTS = {
    "API_SHAPE": API_SHAPE, "INVARIANCE": INVARIANCE, "PRESERVATION": PRESERVATION,
    "ARITHMETIC": ARITHMETIC, "PASS": PASS, "FAIL": FAIL, "NA": NA,
}


def _literal(node: Any) -> Any:
    """Resolve a literal, a bare name, or a dotted name.

    The dotted form matters and was missed once already: the migrated scripts
    load this module by path under a local alias and write
    ``kind=harness.API_SHAPE``. Reading only ``ast.Name`` made every kind come
    back None, so NO rule fired and the static audit passed every file
    vacuously -- a check that cannot fail, inside the tool built to forbid
    exactly that. Case 24 pins it.
    """
    if node is None:
        return None
    if isinstance(node, ast.Attribute) and node.attr in _CONSTANTS:
        return _CONSTANTS[node.attr]
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


# --- leg C: the before-state -----------------------------------------------
def compare_before(before: Suite, after: Suite) -> list[str]:
    """Every check must report on the UNMODIFIED input what it declared.

    A check that already passes there is certifying the input rather than the
    work -- which is exactly how the RT-13 inertia assertion looked: on a robot
    with no tool_link at all, ``measured.get(...) is None`` reads True.
    """
    findings: list[str] = []
    for n in after.names():
        want = after.expectations()[n]
        if want == NA:
            continue
        if n not in before.names():
            findings.append(f"{NOTE}: {n} does not exist in the before-suite; expectation not tested.")
            continue
        got = before.result(n)
        if got != want:
            findings.append(
                f"{VIOLATION}: {n} reports {got} on the UNMODIFIED input, declared {want}. "
                + ("A check that already passes before the work is about the input."
                   if got == PASS else "The declaration and the check disagree.")
            )
    return findings


# --- leg B: the counter-proof ----------------------------------------------
def counter_proof(baseline: Suite, mutated: dict[str, Suite]) -> list[str]:
    """Each mutation must flip EXACTLY the checks it names.

    Fewer -> the check cannot fail (the silent failure mode).
    More   -> the mutation is not surgical and proves nothing about that check.
    """
    findings: list[str] = []
    base = {n: baseline.result(n) for n in baseline.names()}
    for m in baseline.mutations():
        if m.name not in mutated:
            findings.append(f"{VIOLATION}: mutation {m.name!r} was never run.")
            continue
        after = mutated[m.name]
        flipped = {
            n for n in baseline.names()
            if base.get(n) == PASS and n in after.names() and after.result(n) != PASS
        }
        want = set(m.flips)
        missing = sorted(want - flipped)
        extra = sorted(flipped - want)
        if missing:
            findings.append(
                f"{VIOLATION}: mutation {m.name!r} did not break {missing}. "
                "Either the mutation does not bite, or the check cannot fail at all."
            )
        if extra:
            findings.append(
                f"{VIOLATION}: mutation {m.name!r} also broke {extra}. "
                "It is not surgical, so it proves nothing about the check it names."
            )
    return findings


# --- sidecar ---------------------------------------------------------------
def sidecar_block(suite: Suite, before_table: Any = None, findings: list[str] | None = None) -> dict:
    """The keys this harness ADDS. Never touches the legacy ones.

    Additive JSON cannot break a ``data.get(...)`` reader, and
    insertion_env.py is exactly that reader for ``.author.json``.
    """
    return {
        "harness": CHECKS_MARKER,
        "claims": suite.claims(),
        "before_table": before_table if before_table is not None else "UNAVAILABLE",
        "claims_audit": list(findings or []),
        "measured_values": suite.values(),
    }


# ===========================================================================
#  Self-test -- the 20 cases listed in the module docstring. No USD anywhere.
# ===========================================================================
def _self_test(repo_root: pathlib.Path) -> int:
    checked = 0
    real = "scripts/tools/checks.py"          # exists, by construction
    ghost = "scripts/does_not_exist_ever.py"  # does not
    good_probe = Probe("RT-15", real, "Q1", "HasAuthoredValue=False, source=Fallback")

    def mut(name: str, flips: tuple[str, ...]) -> Mutation:
        return Mutation(name, lambda _s: f"applied {name}", flips)

    def has(findings: list[str], needle: str) -> bool:
        return any(needle in f for f in findings)

    # 1 / 2 -- api-shape without and with a probe
    s = Suite("t", "x")
    s.add("unpaid", True, kind=API_SHAPE, why="w", mutation=mut("m1", ("unpaid",)))
    f = audit(s, repo_root)
    assert has(f, "no probe"), f
    checked += 1
    s = Suite("t", "x")
    s.add("paid", True, kind=API_SHAPE, why="w", probe=good_probe, mutation=mut("m1", ("paid",)))
    assert audit(s, repo_root) == [], audit(s, repo_root)
    checked += 1

    # 3 -- a probe pointing at a file that is not there
    s = Suite("t", "x")
    s.add("stale", True, kind=API_SHAPE, why="w",
          probe=Probe("RT-15", ghost, "Q1", "x"), mutation=mut("m1", ("stale",)))
    assert has(audit(s, repo_root), "not in the repo"), audit(s, repo_root)
    checked += 1

    # 4 -- invariance that already holds before the work
    s = Suite("t", "x")
    s.add("inv", True, kind=INVARIANCE, why="w", expect_before=PASS, mutation=mut("m1", ("inv",)))
    assert has(audit(s, repo_root), "expects PASS before"), audit(s, repo_root)
    checked += 1

    # 5 -- api-shape with no counter-proof
    s = Suite("t", "x")
    s.add("nomut", True, kind=API_SHAPE, why="w", probe=good_probe)
    assert has(audit(s, repo_root), "no counter-proof"), audit(s, repo_root)
    checked += 1

    # 6 -- Unprovable satisfies rule 2 but is announced
    s = Suite("t", "x")
    s.add("hard", True, kind=API_SHAPE, why="w", probe=good_probe,
          mutation=Unprovable("the API offers no way to author this attribute"))
    f = audit(s, repo_root)
    assert not has(f, VIOLATION), f
    assert has(f, "declared unprovable"), f
    checked += 1

    # 7 -- flips names a check that does not exist
    s = Suite("t", "x")
    s.add("a", True, kind=API_SHAPE, why="w", probe=good_probe, mutation=mut("m1", ("ghost_check",)))
    assert has(audit(s, repo_root), "not a check in this suite"), audit(s, repo_root)
    checked += 1

    # 8 -- empty flips
    s = Suite("t", "x")
    s.add("a", True, kind=API_SHAPE, why="w", probe=good_probe, mutation=mut("m1", ()))
    assert has(audit(s, repo_root), "flips nothing"), audit(s, repo_root)
    checked += 1

    # 9 -- two checks sharing one mutation name
    s = Suite("t", "x")
    s.add("a", True, kind=API_SHAPE, why="w", probe=good_probe, mutation=mut("same", ("a",)))
    s.add("b", True, kind=API_SHAPE, why="w", probe=good_probe, mutation=mut("same", ("b",)))
    assert has(audit(s, repo_root), "is used by both"), audit(s, repo_root)
    checked += 1

    # 10 / 11 -- a suite that cannot see the work, and the generator exemption
    s = Suite("t", "x")
    s.add("a", True, kind=ARITHMETIC, why="w", expect_before=PASS)
    assert has(audit(s, repo_root), "cannot tell the input from the result"), audit(s, repo_root)
    checked += 1
    s = Suite("t", "x", no_before_state=True)
    s.add("a", True, kind=ARITHMETIC, why="w", expect_before=NA)
    assert audit(s, repo_root) == [], audit(s, repo_root)
    checked += 1

    # 12 / 13 / 14 -- the counter-proof itself
    base = Suite("t", "x")
    base.add("target", True, kind=API_SHAPE, why="w", probe=good_probe, mutation=mut("m", ("target",)))
    base.add("other", True, kind=ARITHMETIC, why="w")

    inert = Suite("t", "x")
    inert.add("target", True, kind=API_SHAPE, why="w", probe=good_probe, mutation=mut("m", ("target",)))
    inert.add("other", True, kind=ARITHMETIC, why="w")
    assert has(counter_proof(base, {"m": inert}), "did not break"), counter_proof(base, {"m": inert})
    checked += 1

    blunt = Suite("t", "x")
    blunt.add("target", False, kind=API_SHAPE, why="w", probe=good_probe, mutation=mut("m", ("target",)))
    blunt.add("other", False, kind=ARITHMETIC, why="w")
    assert has(counter_proof(base, {"m": blunt}), "also broke"), counter_proof(base, {"m": blunt})
    checked += 1

    surgical = Suite("t", "x")
    surgical.add("target", False, kind=API_SHAPE, why="w", probe=good_probe, mutation=mut("m", ("target",)))
    surgical.add("other", True, kind=ARITHMETIC, why="w")
    assert counter_proof(base, {"m": surgical}) == [], counter_proof(base, {"m": surgical})
    checked += 1

    # 15 / 16 -- the before-state
    after = Suite("t", "x")
    after.add("a", True, kind=ARITHMETIC, why="w", expect_before=FAIL)
    before_bad = Suite("t", "x")
    before_bad.add("a", True, kind=ARITHMETIC, why="w", expect_before=FAIL)
    assert has(compare_before(before_bad, after), "about the input"), compare_before(before_bad, after)
    checked += 1
    before_good = Suite("t", "x")
    before_good.add("a", False, kind=ARITHMETIC, why="w", expect_before=FAIL)
    assert compare_before(before_good, after) == [], compare_before(before_good, after)
    checked += 1

    # 17 -- the legacy shape is unchanged
    s = Suite("t", "x")
    s.add("a", True, kind=ARITHMETIC, why="w")
    s.add("b", False, kind=ARITHMETIC, why="w")
    assert s.as_dict() == {"a": True, "b": False}, s.as_dict()
    assert json.dumps(s.as_dict()) == json.dumps({"a": True, "b": False})
    checked += 1

    # 18 -- a check that cannot be evaluated is ERROR, and ERROR is not a pass
    def boom() -> bool:
        raise RuntimeError("no such prim")

    s = Suite("t", "x")
    s.add("broken", boom, kind=ARITHMETIC, why="w")
    assert s.result("broken") == ERROR, s.result("broken")
    assert s.as_dict() == {"broken": False}
    assert s.verdict() == FAIL and s.exit_code() == 1
    checked += 1

    # 19 -- the RT-13 bug, replayed on the real assertion
    measured_before: dict = {}          # no tool_link on the shipped robot
    old_assertion = measured_before.get("diagonal_inertia") is None
    assert old_assertion is True, "the replay must reproduce the bug, not hide it"
    before = Suite("t", "x")
    before.add("diagonal_inertia_left_to_physx", old_assertion, kind=API_SHAPE, why="w",
               probe=good_probe, mutation=mut("m", ("diagonal_inertia_left_to_physx",)))
    live = Suite("t", "x")
    live.add("diagonal_inertia_left_to_physx", True, kind=API_SHAPE, why="w", expect_before=FAIL,
             probe=good_probe, mutation=mut("m", ("diagonal_inertia_left_to_physx",)))
    assert has(compare_before(before, live), "about the input"), compare_before(before, live)
    checked += 1

    # 20 -- and a clean suite must produce nothing at all
    s = Suite("author_tool", "ur5e_tool.usd")
    s.add("size", True, kind=ARITHMETIC, why="the part measures what the CAD says")
    s.add("inertia", True, kind=API_SHAPE, why="the tensor is left to PhysX",
          probe=good_probe, mutation=mut("author_inertia", ("inertia",)))
    s.add("one_new_body", True, kind=INVARIANCE, why="exactly one body is added",
          mutation=mut("deactivate_link", ("one_new_body",)))
    assert audit(s, repo_root) == [], audit(s, repo_root)
    assert s.verdict() == PASS and s.exit_code() == 0
    checked += 1

    # 21 / 22 / 23 -- the preservation kind, split out when it met a real
    # suite: "articulation_roots_unchanged" must pass BEFORE and after.
    s = Suite("t", "x")
    s.add("keeps", True, kind=PRESERVATION, why="w", expect_before=FAIL, mutation=mut("m", ("keeps",)))
    s.add("changes", True, kind=INVARIANCE, why="w", mutation=mut("m2", ("changes",)))
    assert has(audit(s, repo_root), "cannot be preserved"), audit(s, repo_root)
    checked += 1

    s = Suite("t", "x")
    s.add("keeps", True, kind=PRESERVATION, why="w", expect_before=PASS)
    s.add("changes", True, kind=INVARIANCE, why="w", mutation=mut("m2", ("changes",)))
    assert has(audit(s, repo_root), "no counter-proof"), audit(s, repo_root)
    checked += 1

    s = Suite("t", "x")
    s.add("keeps", True, kind=PRESERVATION, why="w", expect_before=PASS,
          mutation=mut("m", ("keeps",)))
    s.add("changes", True, kind=INVARIANCE, why="w", mutation=mut("m2", ("changes",)))
    assert audit(s, repo_root) == [], audit(s, repo_root)
    checked += 1

    # 24 -- audit_source must read the DOTTED constant form the migrated
    # scripts actually use. Missing it made the static audit pass every file
    # without firing a single rule.
    import tempfile
    src = (
        "def build(s):\n"
        "    s.add('unpaid', True, kind=harness.API_SHAPE, why='w')\n"
        "    s.add('fine', True, kind=harness.ARITHMETIC, why='w')\n"
    )
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "migrated_example.py"
        f.write_text(src, encoding="utf-8")
        found = audit_source(f, repo_root)
    assert has(found, "no probe="), found
    assert has(found, "no mutation="), found
    assert not any("fine" in x for x in found), found
    checked += 1

    assert checked == 24, f"only {checked} of 24 cases ran"
    print(f"[checks] self-test: {checked}/24 cases passed ({CHECKS_MARKER})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="The check harness (D-080).")
    parser.add_argument("--self-test", action="store_true", help="Prove the harness itself. No USD needed.")
    parser.add_argument("--audit-source", type=str, default=None, help="Run the static rules over one .py file.")
    args = parser.parse_args()
    repo_root = pathlib.Path(__file__).resolve().parents[2]

    if args.audit_source:
        findings = audit_source(args.audit_source, repo_root)
        for f in findings:
            print(f"[checks] {f}")
        print(f"[checks] audit-source {args.audit_source}: "
              f"{'PASS' if not findings else f'FAIL ({len(findings)})'}")
        return 1 if any(f.startswith(VIOLATION) for f in findings) else 0

    if args.self_test:
        return _self_test(repo_root)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
