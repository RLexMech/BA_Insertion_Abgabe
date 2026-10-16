# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Success-based curriculum for insertion start heights."""

from __future__ import annotations

UNSET = None

ADVANCE_AT = 0.80
RETREAT_AT = 0.10
RETREAT_STEP_FACTOR = 0.5


class LadderError(RuntimeError):
    """A ladder that cannot be built is a hard stop, never a default."""


class Ladder:
    """Curriculum state: which rung, how wide each axis, how big the step."""

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

    def update(self, success_rate: float) -> str:
        """Apply the 80/10 rule to one measured success rate."""
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
                return "retreat_blocked"
            increment = self._history.pop()
            for axis in self.axes:
                self._steps_taken[axis] -= increment[axis]
            self._rung -= 1
            self.step_scale = self.step_scale * RETREAT_STEP_FACTOR
            return "retreat"
        return "hold"
