# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""The manual multi-axis curriculum ladder (D-110).

WHY THIS FILE IS NEW CODE
-------------------------
D-110 point (2) adopts "the proxy ladder machinery". The proxy has none. Its
rung is an ASSET plus a config value -- the pocket side length is part of the
table USD's filename, so a rung is chosen by generating a different table and
pointing the config at it (`proxytask_env_cfg.py`, the comment block about
loading the wrong rung's file). There is no state machine to port, so this one
is written here, and that is said rather than implied. The RULE is not
invented: the 80/10 form is IndustReal's, and the multi-axis widening is the
proxy's own practice.

Plain Python, stdlib only -- no torch, no Isaac. That makes it the one piece
of this stream the dev laptop can import for real.

WHAT A RUNG IS
--------------
Three axes widened TOGETHER (D-110 point (2)): start height above the opening,
fixture offset, tilt. Every range is symmetric around the target pose, so a
rung is one half-width per axis and nothing else. That is what makes the hard
constraint checkable: WIDEN, NEVER SHIFT (IndustReal's shift ablation). With
symmetric ranges, "rung k+1 contains rung k" reduces to "no half-width
shrinks", which the offline check tests directly.

THE RULE
--------
Advance above 80 % success, retreat below 10 % (D-110 point (3), the
IndustReal PAPER thresholds). Retreat means: one rung back, and re-widen with
HALF the step. Footnote for the report: the released IndustReal CODE uses
0.75 / 0.50 instead; the paper values are the chosen ones.

THE STEP SIZES ARE `[open]`
---------------------------
D-110 point (3): they come from this project's own learning curves, not from
IndustReal's 5 mm, which is a different ladder variable. Passing an unset step
raises rather than defaulting -- a ladder with a silently invented step size
would train a curriculum nobody decided.
"""

from __future__ import annotations

# The sentinel for a decided-but-not-yet-measured number. Same contract as
# WORKCELL_PENDING in insertion_tasks_cfg.py, with teeth: that one reports,
# this one refuses.
UNSET = None

# D-110 point (3). Named rather than written into the comparison so the report
# can point at one place, and so the paper-vs-code discrepancy has a home.
ADVANCE_AT = 0.80
RETREAT_AT = 0.10
RETREAT_STEP_FACTOR = 0.5


class LadderError(RuntimeError):
    """A ladder that cannot be built is a hard stop, never a default."""


class Ladder:
    """Curriculum state: which rung, how wide each axis, how big the step.

    ``update(success_rate)`` returns one of ``advance`` / ``retreat`` /
    ``retreat_blocked`` / ``hold``. The caller logs it -- a rung change writes
    its own marked line, and ``as_dict()`` feeds the run tag and the metrics
    file so no log line is ever ambiguous about which curriculum ran.
    """

    def __init__(self, axes, base, step, max_rung: int,
                 advance_at: float = ADVANCE_AT, retreat_at: float = RETREAT_AT):
        self.axes = tuple(axes)
        self.max_rung = int(max_rung)
        self.advance_at = float(advance_at)
        self.retreat_at = float(retreat_at)
        self.step_scale = 1.0
        self._rung = 0
        self._history: list = []

        if base is UNSET or step is UNSET:
            raise LadderError(
                "curriculum base/step ranges are UNSET -- D-110 leaves the rung step "
                "sizes [open]; they come from this project's own learning curves."
            )
        for axis in self.axes:
            if axis not in base or axis not in step:
                raise LadderError(f"axis {axis!r} is missing from base or step")
        self.base = {a: base[a] for a in self.axes}
        self.step = {}
        for axis in self.axes:
            value = step[axis]
            if value is UNSET:
                raise LadderError(
                    f"curriculum step for axis {axis!r} is UNSET -- [open] per D-110"
                )
            if value <= 0.0:
                raise LadderError(
                    f"curriculum step for axis {axis!r} is {value} -- a non-positive step "
                    "would SHIFT the ladder instead of widening it (D-110 forbids it)"
                )
            self.step[axis] = float(value)
        for axis in self.axes:
            if self.base[axis] is UNSET:
                raise LadderError(f"curriculum base for axis {axis!r} is UNSET")
        self._steps_taken = {a: 0.0 for a in self.axes}

    # -- state ---------------------------------------------------------------
    @property
    def rung(self) -> int:
        return self._rung

    @property
    def ranges(self) -> dict:
        """Half-width per axis, symmetric around the target pose."""
        out = {}
        for axis in self.axes:
            base_width = self.base[axis]
            width = base_width + self._steps_taken[axis]
            out[axis] = width
        return out

    def as_dict(self) -> dict:
        """Telemetry: every run tag and metrics file names the rung."""
        return {"rung": self._rung, "step_scale": self.step_scale, "ranges": self.ranges}

    # -- the rule ------------------------------------------------------------
    def update(self, success_rate: float) -> str:
        """Apply the 80/10 rule to one measured success rate.

        Both thresholds are STRICT: exactly 80 % holds, exactly 10 % holds.
        A boundary that advances on the nose would make the rule depend on how
        many episodes the rate was averaged over.
        """
        if success_rate > self.advance_at:
            if self._rung >= self.max_rung:
                return "hold"
            increment = {a: self.step[a] * self.step_scale for a in self.axes}
            for axis in self.axes:
                self._steps_taken[axis] += increment[axis]
            self._history.append(increment)
            self._rung += 1
            return "advance"
        if success_rate < self.retreat_at:
            if not self._history:
                # Rung 0 is the bottom. Reported, not invented: a first rung
                # that is already too hard is a fact a human must see, not
                # something the ladder should quietly shrink its way out of.
                return "retreat_blocked"
            increment = self._history.pop()
            for axis in self.axes:
                self._steps_taken[axis] -= increment[axis]
            self._rung -= 1
            self.step_scale = self.step_scale * RETREAT_STEP_FACTOR
            return "retreat"
        return "hold"
