# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive domain randomization with boundary trials and per-episode sampling."""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from dataclasses import dataclass, replace

from .curriculum import ADVANCE_AT, RETREAT_AT

TESTWERT = "[TESTWERT]"

AKKAYA_CONSTANT_SOURCE = (
    "Akkaya et al. 2019, Appendix C.3 Table 15 (arXiv:1910.07113 p. 44), "
    "read from the PDF 2026-09-09: queue length m 240, boundary sampling "
    "probability 0.5"
)
BUFFER_M = 240
P_BOUNDARY = 0.5
DELTA_STEPS_SOURCE = (
    "[TESTWERT] Delta = maximum/10 from a fill-rate "
    "estimate, NOT Akkaya's absolute 0.02; requires empirical validation"
)
DELTA_STEPS = 10

TABLE_COLUMNS: tuple[str, ...] = (
    "lat_r",
    "lat_phi",
    "yaw",
    "tilt",
    "tilt_azimuth",     # 4  fixture tilt DIRECTION -- no boundary, no buffer
    "start_height",
    "friction",
    "fixture_noise_x",
    "fixture_noise_y",
    "joint_noise_0",
    "joint_noise_1",
    "joint_noise_2",
    "joint_noise_3",
    "joint_noise_4",
    "joint_noise_5",    # 14
    "grasp_obs_x",      # 15 grasp belief error, OBSERVATION only (D-183) --
)
COLUMN_INDEX: dict = {name: i for i, name in enumerate(TABLE_COLUMNS)}


class AutoDRError(RuntimeError):
    """A randomisation table that cannot be built is a hard stop, never a default."""


def _field(state, key, kind):
    """Read one field of a resume state, or refuse BY NAME."""
    if not isinstance(state, dict):
        raise AutoDRError(
            f"resume state must be a dict, got {type(state).__name__}"
        )
    if key not in state:
        raise AutoDRError(f"resume state has no {key!r}")
    v = state[key]
    if kind is int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise AutoDRError(f"resume {key!r} must be a whole number, got {v!r}")
    elif kind is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AutoDRError(f"resume {key!r} must be a number, got {v!r}")
        if v != v:
            raise AutoDRError(f"resume {key!r} is NaN")
    return v


def _named_dims(entries) -> dict:
    """Index a resume state's ``dims`` list by name, refusing by name."""
    return {_field(e, "name", str): e for e in entries}


@dataclass(frozen=True)
class DimSpec:
    """One randomised physical quantity and how far it is allowed to open."""

    name: str
    unit: str
    lo_max: float
    hi_max: float
    one_sided: bool
    source: str
    centre: float | None = None
    n_steps: int = DELTA_STEPS

    @property
    def column(self) -> int:
        """Index into the 16-column table layout."""
        return COLUMN_INDEX[self.name]

    @property
    def sides(self) -> tuple[str, ...]:
        return ("hi",) if self.one_sided else ("lo", "hi")

    def bound_max(self, side: str) -> float:
        return self.hi_max if side == "hi" else self.lo_max

    def bound_at(self, side: str, steps: int) -> float:
        """Boundary value after ``steps`` deltas. ``steps == n_steps`` is EXACT."""
        if self.centre is None:
            raise AutoDRError(f"dimension {self.name!r} has no centre bound")
        frac = float(steps) / float(self.n_steps)
        return self.centre + frac * (self.bound_max(side) - self.centre)

    def delta(self, side: str) -> float:
        """One step, signed away from the centre."""
        if self.centre is None:
            raise AutoDRError(f"dimension {self.name!r} has no centre bound")
        return (self.bound_max(side) - self.centre) / float(self.n_steps)

    def bind(self, centre: float) -> "DimSpec":
        """Return a copy carrying the centre from the task config."""
        if self.n_steps < 1:
            raise AutoDRError(
                f"dimension {self.name!r}: n_steps must be >= 1, got {self.n_steps}"
            )
        if self.one_sided:
            self = replace(self, lo_max=float(centre))
        if self.lo_max > self.hi_max:
            raise AutoDRError(
                f"dimension {self.name!r}: lo_max {self.lo_max} is above hi_max {self.hi_max}"
            )
        c = float(centre)
        for side in self.sides:
            if self.bound_max(side) == c:
                raise AutoDRError(
                    f"dimension {self.name!r}: the {side} maximum equals the "
                    f"centre ({c}), so that boundary has nowhere to go -- use "
                    "one_sided=True if the quantity really has only one edge"
                )
        if not (self.lo_max <= c <= self.hi_max):
            raise AutoDRError(
                f"dimension {self.name!r}: centre {c} is outside [{self.lo_max}, {self.hi_max}]"
            )
        return replace(self, centre=c)


_D178 = "[TESTWERT] Phase-5 ceiling, decided not derived -- D-178 (5)"

DR_DIMS: tuple[DimSpec, ...] = (
    DimSpec("lat_r", "m", 0.0, 0.030, True, _D178),
    DimSpec("yaw", "rad", -0.08726646259971648, 0.08726646259971648, False, _D178),
    DimSpec("tilt", "rad", 0.0, 0.17453292519943295, True, _D178),
    DimSpec("start_height", "m", 0.0, 0.120, True,
            "[TESTWERT] hi = Phase-5 ceiling, decided not derived -- D-179 (1); "
            "lo = the centre H_min, not yet measured -- D-179 (4)"),
    DimSpec("friction", "1", 0.08, 0.20, False,
            "DuPont PA66/POM table span, assumed PA6/POM pairing, "
            "no PA6 value -- D-181 (2)"),
)
DR_DIM_NAMES: tuple[str, ...] = tuple(d.name for d in DR_DIMS)


def bind_centres(centres: dict, dims: tuple = DR_DIMS) -> tuple:
    """Bind every centre from the task config. Missing centre -> refuse."""
    missing = [d.name for d in dims if d.name not in centres]
    if missing:
        raise AutoDRError(
            f"no centre given for {missing} -- centres come from the task config "
            "(insertion_tasks_cfg / env cfg), never from autodr.py"
        )
    return tuple(d.bind(centres[d.name]) for d in dims)


def boundary_keys(dims: tuple) -> tuple:
    """The boundary names, one per side, in a fixed order. A one-sided quantity has no `_lo`."""
    return tuple(f"{d.name}_{side}" for d in dims for side in d.sides)


FLOOR_KEY = "start_height_floor"
FLOOR_DIM = "start_height"


@dataclass(frozen=True)
class FloorSpec:
    """The start floor: ``f0`` at step 0, ``top`` (= H_min) at step ``n_steps``."""

    f0: float
    top: float
    n_steps: int

    def check(self) -> "FloorSpec":
        if int(self.n_steps) < 1:
            raise AutoDRError(f"floor: n_steps must be >= 1, got {self.n_steps}")
        if not float(self.f0) < float(self.top):
            raise AutoDRError(
                f"floor: f0 {self.f0} must lie below top (H_min) {self.top}"
            )
        return self

    def value_at(self, steps: int) -> float:
        """Floor after ``steps`` moves. ``steps == n_steps`` is EXACTLY ``top``."""
        frac = float(steps) / float(int(self.n_steps))
        return float(self.f0) + frac * (float(self.top) - float(self.f0))


def testwert_dims(dims: tuple = DR_DIMS) -> tuple:
    """Every quantity whose limits are decided-but-not-derived."""
    return tuple(d.name for d in dims if TESTWERT in d.source)


def provenance_lines(dims: tuple = DR_DIMS) -> tuple:
    """The provenance report, one line per un-derived number."""
    lines = [f"buffer m / p_boundary: {AKKAYA_CONSTANT_SOURCE}",
             f"delta steps ({DELTA_STEPS}): {DELTA_STEPS_SOURCE}"]
    for d in dims:
        mark = TESTWERT if TESTWERT in d.source else "sourced"
        lines.append(
            f"{d.name} [{d.unit}] {d.lo_max} .. {d.hi_max} ({mark}): {d.source}"
        )
    return tuple(lines)


@dataclass(frozen=True)
class BoundEvent:
    """What one full buffer did. ``moved`` decides the `bounds_version` bump."""

    key: str
    action: str
    rate: float
    before: float
    after: float
    buffer_m: int = BUFFER_M
    note: str = ""

    @property
    def moved(self) -> bool:
        """Did a NUMBER change?"""
        return self.action in ("expand", "shrink") and self.after != self.before

    def line(self) -> str:
        """Emit a diagnostic log line with the searchable `[autodr]` prefix."""
        return (
            f"[autodr] {self.key}: rate {self.rate:.4f} over {self.buffer_m} "
            f"-> {self.action} {self.before:.6g} -> {self.after:.6g}" + self.note
        )


class _Provider:
    """Shared surface of every bounds provider (`BoundsProvider`)."""

    p_boundary: float = 0.0

    def __init__(self, dims: tuple) -> None:
        dims = tuple(dims)
        if not dims:
            raise AutoDRError("no dimensions given")
        names = [d.name for d in dims]
        if len(set(names)) != len(names):
            raise AutoDRError(f"duplicate dimension name in {names}")
        for d in dims:
            if d.name not in COLUMN_INDEX:
                raise AutoDRError(
                    f"dimension {d.name!r} is not a column of TABLE_COLUMNS"
                )
            if d.centre is None:
                raise AutoDRError(
                    f"dimension {d.name!r} has no centre -- call bind_centres() first"
                )
        self.dims = dims
        self.keys = boundary_keys(self.dims)
        self._dim_of = {d.name: d for d in self.dims}
        self._bounds_version = 0

    @property
    def n_boundaries(self) -> int:
        return len(self.keys)

    @property
    def boundary_columns(self) -> tuple:
        """Table column each boundary nails, aligned with ``keys``."""
        return tuple(self._dim_of[k.rsplit("_", 1)[0]].column for k in self.keys)

    @property
    def boundary_sides(self) -> tuple:
        """0.0 for a lower edge, 1.0 for an upper edge -- the unit value to nail to."""
        return tuple(1.0 if k.rsplit("_", 1)[1] == "hi" else 0.0 for k in self.keys)

    @property
    def bounds_version(self) -> int:
        return self._bounds_version

    def bounds(self) -> dict:
        """``{name: (lo, hi)}`` in the quantity's own unit."""
        raise NotImplementedError

    def bounds_max(self) -> dict:
        """The WIDEST ``{name: (lo, hi)}`` this provider can ever reach."""
        raise NotImplementedError

    def bounds_arrays(self) -> tuple:
        """``(lo, hi)`` as two 16-long lists, one entry per table column."""
        lo = [0.0] * len(TABLE_COLUMNS)
        hi = [1.0] * len(TABLE_COLUMNS)
        for name, pair in self.bounds().items():
            lo[COLUMN_INDEX[name]] = pair[0]
            hi[COLUMN_INDEX[name]] = pair[1]
        return lo, hi

    def boundary_assignment(self, rand_a, rand_b):
        """Which resetting envs are boundary envs, and on which boundary."""
        return boundary_assignment(rand_a, rand_b, self.n_boundaries, self.p_boundary)

    def record(self, key: str, success: bool) -> None:
        raise NotImplementedError

    def update(self) -> list:
        raise NotImplementedError

    def all_at_max(self) -> bool:
        raise NotImplementedError

    def as_dict(self) -> dict:
        raise NotImplementedError

    def scalars(self) -> dict:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError


class AutoDR(_Provider):
    """Akkaya Algorithm 1 over the table's boundaries (seven for ``DR_DIMS``)."""

    def __init__(
        self,
        dims: tuple,
        *,
        buffer_m: int = BUFFER_M,
        p_boundary: float = P_BOUNDARY,
        advance_at: float = ADVANCE_AT,
        retreat_at: float = RETREAT_AT,
        floor: "FloorSpec | None" = None,
        stall_buffers: int = 0,
    ) -> None:
        super().__init__(dims)
        self._check_params(buffer_m, p_boundary, advance_at, retreat_at)
        if int(stall_buffers) < 0:
            raise AutoDRError(f"stall_buffers {stall_buffers} is negative")
        self.stall_buffers = int(stall_buffers)
        self.buffer_m = int(buffer_m)
        self.p_boundary = float(p_boundary)
        self.advance_at = float(advance_at)
        self.retreat_at = float(retreat_at)
        self.floor = None
        if floor is not None:
            floor.check()
            sh = self._dim_of.get(FLOOR_DIM)
            if sh is None:
                raise AutoDRError(
                    f"a floor needs the {FLOOR_DIM!r} dimension in the provider"
                )
            if float(floor.top) != float(sh.centre):
                raise AutoDRError(
                    f"floor top {floor.top} != start_height centre {sh.centre}"
                )
            self.floor = floor
            self.keys = tuple(self.keys) + (FLOOR_KEY,)
        self._steps = {k: 0 for k in self.keys}
        self._n = {k: 0 for k in self.keys}
        self._s = {k: 0 for k in self.keys}
        self._dropped = {k: 0 for k in self.keys}
        self._last_rate = {k: -1.0 for k in self.keys}
        self._holds = {k: 0 for k in self.keys}

    @staticmethod
    def _check_params(buffer_m, p_boundary, advance_at, retreat_at) -> None:
        """The parameter contract."""
        if int(buffer_m) < 1:
            raise AutoDRError(f"buffer_m must be >= 1, got {buffer_m}")
        if not 0.0 <= float(p_boundary) <= 1.0:
            raise AutoDRError(f"p_boundary must be in [0, 1], got {p_boundary}")
        if not 0.0 <= float(retreat_at) < float(advance_at) <= 1.0:
            raise AutoDRError(
                f"need 0 <= retreat_at ({retreat_at}) < advance_at ({advance_at}) <= 1"
            )

    def _split(self, key: str) -> tuple:
        name, side = key.rsplit("_", 1)
        return self._dim_of[name], side

    def _n_steps_of(self, key: str) -> int:
        if key == FLOOR_KEY:
            return int(self.floor.n_steps)
        return self._dim_of[key.rsplit("_", 1)[0]].n_steps

    @property
    def phase(self) -> str:
        """``\"floor\"`` while the floor is below H_min, else ``\"dr\"``."""
        if self.floor is not None and self._steps[FLOOR_KEY] < self.floor.n_steps:
            return "floor"
        return "dr"

    def value(self, key: str) -> float:
        if key not in self._steps:
            raise AutoDRError(f"unknown boundary {key!r}; known: {self.keys}")
        if key == FLOOR_KEY:
            return self.floor.value_at(self._steps[key])
        dim, side = self._split(key)
        return dim.bound_at(side, self._steps[key])

    def bounds(self) -> dict:
        if self.phase == "floor":
            out = {d.name: (d.centre, d.centre) for d in self.dims}
            out[FLOOR_DIM] = (self.value(FLOOR_KEY), self._dim_of[FLOOR_DIM].centre)
            return out
        out = {}
        for d in self.dims:
            lo = d.centre if d.one_sided else self.value(f"{d.name}_lo")
            out[d.name] = (lo, self.value(f"{d.name}_hi"))
        return out

    def boundary_assignment(self, rand_a, rand_b):
        """Phase floor: every boundary env is nailed to the floor (the last key)."""
        if self.floor is None:
            return boundary_assignment(rand_a, rand_b, self.n_boundaries, self.p_boundary)
        n_dr = self.n_boundaries - 1
        is_b, index = boundary_assignment(rand_a, rand_b, n_dr, self.p_boundary)
        if self.phase == "floor":
            index = index * 0 + n_dr
        return is_b, index

    def bounds_max(self) -> dict:
        """Every boundary clamps at its dimension maximum, so that IS the reach."""
        return {d.name: (d.lo_max, d.hi_max) for d in self.dims}

    def fill(self) -> dict:
        return dict(self._n)

    def dropped(self) -> dict:
        """Flags that arrived at a full buffer, per boundary. Must stay 0."""
        return dict(self._dropped)

    def all_at_max(self) -> bool:
        """Integer comparison -- no epsilon, because the steps are integers."""
        return all(self._steps[k] >= self._n_steps_of(k) for k in self.keys)

    def record(self, key: str, success: bool) -> None:
        """One finished BOUNDARY episode."""
        if key not in self._n:
            raise AutoDRError(f"unknown boundary {key!r}; known: {self.keys}")
        if self._n[key] >= self.buffer_m:
            self._dropped[key] += 1
            return
        self._n[key] += 1
        self._s[key] += 1 if success else 0

    def update(self) -> list:
        """Process every FULL buffer."""
        events = []
        for key in self.keys:
            if self._n[key] < self.buffer_m:
                continue
            rate = self._s[key] / float(self._n[key])
            self._n[key] = 0
            self._s[key] = 0
            self._last_rate[key] = rate
            n_steps = self._n_steps_of(key)
            before = self.value(key)
            if rate >= self.advance_at:
                if self._steps[key] >= n_steps:
                    action = "clamped_max"
                else:
                    self._steps[key] += 1
                    action = "expand"
            elif rate <= self.retreat_at:
                if self._steps[key] <= 0:
                    action = "clamped_zero"
                else:
                    self._steps[key] -= 1
                    action = "shrink"
            else:
                action = "hold"
            after = self.value(key)
            note = ""
            if key == FLOOR_KEY and action == "expand" and self._steps[key] >= n_steps:
                note = " -- reached H_min, phase dr"
            ev = BoundEvent(key, action, rate, before, after, self.buffer_m, note)
            if ev.moved:
                self._bounds_version += 1
                self._holds[key] = 0
            elif action != "clamped_max":
                self._holds[key] += 1
            if 0 < self.stall_buffers <= self._holds[key]:
                ev = BoundEvent(
                    key, action, rate, before, after, self.buffer_m,
                    note + f" -- STALL {self._holds[key]} full buffers without a "
                    f"move (review bar {self.stall_buffers})",
                )
            events.append(ev)
        return events

    def as_dict(self) -> dict:
        return {
            "mode": "autodr",
            "bounds_version": self._bounds_version,
            "bounds": {k: list(v) for k, v in self.bounds().items()},
            "steps": dict(self._steps),
            "fill": dict(self._n),
            "buffer_m": self.buffer_m,
            "p_boundary": self.p_boundary,
            "advance_at": self.advance_at,
            "retreat_at": self.retreat_at,
            "all_at_max": self.all_at_max(),
            "testwert": list(testwert_dims(self.dims)),
            "phase": self.phase,
            "floor": self._floor_block(),
            "last_rate": dict(self._last_rate),
            "holds": dict(self._holds),
            "stall_buffers": self.stall_buffers,
        }

    def _floor_block(self):
        if self.floor is None:
            return None
        return {
            "f0": float(self.floor.f0), "top": float(self.floor.top),
            "n_steps": int(self.floor.n_steps),
        }

    def scalars(self) -> dict:
        """Flat `dr/...` keys for `extras["log"]`."""
        out = {}
        for key in self.keys:
            out[f"dr/{key}"] = self.value(key)
            out[f"dr/{key}_fill"] = float(self._n[key])
            out[f"dr/{key}_last_rate"] = float(self._last_rate[key])
            out[f"dr/{key}_holds"] = float(self._holds[key])
        out["dr/bounds_version"] = float(self._bounds_version)
        out["dr/all_at_max"] = 1.0 if self.all_at_max() else 0.0
        out["dr/flags_dropped"] = float(sum(self._dropped.values()))
        if self.floor is not None:
            out["dr/phase"] = 0.0 if self.phase == "floor" else 1.0
        return out

    def state_dict(self) -> dict:
        """Everything a resume needs."""
        return {
            "format": "autodr-3",
            "floor": self._floor_block(),
            "keys": list(self.keys),
            "steps": dict(self._steps),
            "buf_n": dict(self._n),
            "buf_s": dict(self._s),
            "dropped": dict(self._dropped),
            "last_rate": dict(self._last_rate),
            "holds": dict(self._holds),
            "stall_buffers": self.stall_buffers,
            "bounds_version": self._bounds_version,
            "buffer_m": self.buffer_m,
            "p_boundary": self.p_boundary,
            "advance_at": self.advance_at,
            "retreat_at": self.retreat_at,
            "dims": [
                {
                    "name": d.name, "unit": d.unit, "centre": d.centre,
                    "lo_max": d.lo_max, "hi_max": d.hi_max,
                    "one_sided": d.one_sided, "n_steps": d.n_steps, "source": d.source,
                }
                for d in self.dims
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        """Refuse anything that is not the SAME boundary set."""
        if not isinstance(state, dict):
            raise AutoDRError(
                f"resume state must be a dict, got {type(state).__name__}"
            )
        if state.get("format") != "autodr-3":
            raise AutoDRError(f"unknown autodr state format {state.get('format')!r}")
        if tuple(_field(state, "keys", list)) != self.keys:
            raise AutoDRError(
                f"resume boundary set differs: {state['keys']} vs {list(self.keys)}"
            )
        if "floor" not in state:
            raise AutoDRError("resume state has no 'floor' block")
        fl_in = state["floor"]
        if (fl_in is None) != (self.floor is None):
            raise AutoDRError(
                f"resume floor {fl_in!r} against configured floor {self._floor_block()!r}"
            )
        if fl_in is not None:
            for field, kind in (("f0", float), ("top", float), ("n_steps", int)):
                if _field(fl_in, field, kind) != getattr(self.floor, field):
                    raise AutoDRError(
                        f"resume floor.{field}: {fl_in[field]} != {getattr(self.floor, field)}"
                    )
        saved = _named_dims(_field(state, "dims", list))
        for d in self.dims:
            s = saved.get(d.name)
            if s is None:
                raise AutoDRError(f"resume state has no dimension {d.name!r}")
            for field in ("centre", "lo_max", "hi_max"):
                if _field(s, field, float) != getattr(d, field):
                    raise AutoDRError(
                        f"resume {d.name}.{field}: {s[field]} != {getattr(d, field)}"
                    )
            for field in ("n_steps",):
                if _field(s, field, int) != getattr(d, field):
                    raise AutoDRError(
                        f"resume {d.name}.{field}: {s[field]} != {getattr(d, field)}"
                    )
        self._check_params(
            _field(state, "buffer_m", int), _field(state, "p_boundary", float),
            _field(state, "advance_at", float), _field(state, "retreat_at", float),
        )
        buffer_m = _field(state, "buffer_m", int)
        steps_in = _field(state, "steps", dict)
        n_in = _field(state, "buf_n", dict)
        s_in = _field(state, "buf_s", dict)
        lr_in = _field(state, "last_rate", dict)
        ho_in = _field(state, "holds", dict)
        stall = _field(state, "stall_buffers", int)
        if stall < 0:
            raise AutoDRError(f"resume stall_buffers {stall} is negative")
        if stall != self.stall_buffers:
            raise AutoDRError(
                f"resume stall_buffers {stall} != configured {self.stall_buffers}"
            )
        last_rate, holds = {}, {}
        drop_in = _field(state, "dropped", dict)
        for block_name, block in (("steps", steps_in), ("buf_n", n_in),
                                  ("buf_s", s_in), ("dropped", drop_in)):
            extra = sorted(set(block) - set(self.keys))
            if extra:
                raise AutoDRError(
                    f"resume {block_name} names unknown boundaries {extra}"
                )
        steps, buf_n, buf_s, dropped = {}, {}, {}, {}
        for k in self.keys:
            n_steps = self._n_steps_of(k)
            # A step count out of range is the dangerous one: bound_at
            st = _field(steps_in, k, int)
            if not 0 <= st <= n_steps:
                raise AutoDRError(
                    f"resume steps[{k!r}] = {st} outside [0, {n_steps}]"
                )
            n = _field(n_in, k, int)
            sc = _field(s_in, k, int)
            if not 0 <= sc <= n <= buffer_m:
                raise AutoDRError(
                    f"resume buffer {k!r}: need 0 <= successes {sc} <= "
                    f"flags {n} <= buffer_m {buffer_m}"
                )
            steps[k], buf_n[k], buf_s[k] = st, n, sc
            dk = _field(drop_in, k, int)
            if dk < 0:
                raise AutoDRError(f"resume dropped[{k!r}] = {dk} is negative")
            dropped[k] = dk
            lr = _field(lr_in, k, float)
            if not (lr == -1.0 or 0.0 <= lr <= 1.0):
                raise AutoDRError(f"resume last_rate[{k!r}] = {lr} is not -1 or a rate")
            hk = _field(ho_in, k, int)
            if hk < 0:
                raise AutoDRError(f"resume holds[{k!r}] = {hk} is negative")
            last_rate[k], holds[k] = lr, hk
        version = _field(state, "bounds_version", int)
        if version < 0:
            raise AutoDRError(f"resume bounds_version {version} is negative")
        self._steps, self._n, self._s, self._dropped = steps, buf_n, buf_s, dropped
        self._last_rate, self._holds = last_rate, holds
        self._bounds_version = version
        self.buffer_m = buffer_m
        self.p_boundary = float(state["p_boundary"])
        self.advance_at = float(state["advance_at"])
        self.retreat_at = float(state["retreat_at"])

    def save(self, path) -> None:
        p = pathlib.Path(path)
        try:
            p.write_text(
                json.dumps(self.state_dict(), indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError as exc:
            raise AutoDRError(f"cannot write the autodr state to {p}: {exc}") from exc

    def load(self, path) -> None:
        p = pathlib.Path(path)
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise AutoDRError(f"cannot read the autodr state at {p}: {exc}") from exc
        try:
            state = json.loads(raw)
        except ValueError as exc:
            raise AutoDRError(f"{p} is not valid JSON: {exc}") from exc
        self.load_state_dict(state)


class FixedWidth(_Provider):
    """Boundaries that never move: the No-DR condition and every evaluation."""

    def __init__(self, dims: tuple, *, steps: int) -> None:
        super().__init__(dims)
        self._fixed_steps = int(steps)
        if self._fixed_steps < 0:
            raise AutoDRError(f"steps must be >= 0, got {steps}")
        for d in self.dims:
            if self._fixed_steps > d.n_steps:
                raise AutoDRError(
                    f"steps {self._fixed_steps} above n_steps {d.n_steps} for {d.name!r}"
                )

    @classmethod
    def at_centre(cls, dims) -> "FixedWidth":
        """Width 0 -- the No-DR condition."""
        return cls(dims, steps=0)

    @classmethod
    def at_max(cls, dims) -> "FixedWidth":
        """Full width -- what the evaluation table is mapped onto."""
        steps = {d.n_steps for d in dims}
        if len(steps) != 1:
            raise AutoDRError(
                f"at_max needs one common n_steps, got {sorted(steps)} -- "
                "pass FixedWidth(dims, steps=...) explicitly"
            )
        return cls(dims, steps=steps.pop())

    def bounds(self) -> dict:
        out = {}
        for d in self.dims:
            steps = min(self._fixed_steps, d.n_steps)
            lo = d.centre if d.one_sided else d.bound_at("lo", steps)
            out[d.name] = (lo, d.bound_at("hi", steps))
        return out

    def bounds_max(self) -> dict:
        """Identical to ``bounds()``: a fixed width never opens any further."""
        return self.bounds()

    def record(self, key: str, success: bool) -> None:
        return None

    def update(self) -> list:
        return []

    def all_at_max(self) -> bool:
        """Whether the FIXED width already sits at every maximum."""
        return all(self._fixed_steps >= d.n_steps for d in self.dims)

    def as_dict(self) -> dict:
        return {
            "mode": "fixed",
            "bounds_version": 0,
            "bounds": {k: list(v) for k, v in self.bounds().items()},
            "steps": {k: self._fixed_steps for k in self.keys},
            "fill": {k: 0 for k in self.keys},
            "all_at_max": self.all_at_max(),
            "testwert": list(testwert_dims(self.dims)),
        }

    def scalars(self) -> dict:
        out = {}
        for key in self.keys:
            name, side = key.rsplit("_", 1)
            d = self._dim_of[name]
            out[f"dr/{key}"] = d.bound_at(side, min(self._fixed_steps, d.n_steps))
            out[f"dr/{key}_fill"] = 0.0
            out[f"dr/{key}_last_rate"] = -1.0
            out[f"dr/{key}_holds"] = 0.0
        out["dr/bounds_version"] = 0.0
        out["dr/all_at_max"] = 1.0 if self.all_at_max() else 0.0
        out["dr/flags_dropped"] = 0.0
        return out

    def state_dict(self) -> dict:
        """Carries the DIMENSION TABLE, not just the width."""
        return {
            "format": "fixed-1",
            "steps": self._fixed_steps,
            "keys": list(self.keys),
            "dims": [
                {
                    "name": d.name, "centre": d.centre, "lo_max": d.lo_max,
                    "hi_max": d.hi_max, "n_steps": d.n_steps,
                }
                for d in self.dims
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise AutoDRError(
                f"resume state must be a dict, got {type(state).__name__}"
            )
        if state.get("format") != "fixed-1":
            raise AutoDRError(f"unknown fixed state format {state.get('format')!r}")
        if tuple(_field(state, "keys", list)) != self.keys:
            raise AutoDRError(
                f"resume boundary set differs: {state['keys']} vs {list(self.keys)}"
            )
        saved = _named_dims(_field(state, "dims", list))
        for d in self.dims:
            sd = saved.get(d.name)
            if sd is None:
                raise AutoDRError(f"fixed state has no dimension {d.name!r}")
            for field in ("centre", "lo_max", "hi_max", "n_steps"):
                kind = int if field == "n_steps" else float
                if _field(sd, field, kind) != getattr(d, field):
                    raise AutoDRError(
                        f"fixed {d.name}.{field}: {sd[field]} != {getattr(d, field)}"
                    )
        steps = _field(state, "steps", int)
        if steps < 0 or any(steps > d.n_steps for d in self.dims):
            raise AutoDRError(f"fixed steps {steps} outside every dimension's range")
        self._fixed_steps = steps


class FixedDR(AutoDR):
    """The Fixed-DR training arm (D-192): AutoDR's start floor, then full width."""

    FORMAT = "fixed-dr-1"

    def __init__(self, dims: tuple, *, floor: "FloorSpec", **kwargs) -> None:
        if floor is None:
            raise AutoDRError(
                "FixedDR needs a start floor; without one it is FixedWidth.at_max"
            )
        super().__init__(dims, floor=floor, **kwargs)
        for key in self.keys:
            if key != FLOOR_KEY:
                self._steps[key] = self._n_steps_of(key)

    def value(self, key: str) -> float:
        if key != FLOOR_KEY and key in self._steps and self.phase == "floor":
            dim, side = self._split(key)
            return dim.bound_at(side, 0)
        return super().value(key)

    def boundary_assignment(self, rand_a, rand_b):
        """Phase floor: AutoDR's draw. Phase dr: no boundary env at all."""
        if self.phase == "floor":
            return super().boundary_assignment(rand_a, rand_b)
        return boundary_assignment(rand_a, rand_b, self.n_boundaries - 1, 0.0)

    def as_dict(self) -> dict:
        out = super().as_dict()
        out["mode"] = "fixed_dr"
        return out

    def state_dict(self) -> dict:
        out = super().state_dict()
        out["format"] = self.FORMAT
        return out

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("format") != self.FORMAT:
            raise AutoDRError(
                f"unknown fixed-dr state format "
                f"{state.get('format') if isinstance(state, dict) else state!r}"
            )
        steps_in = _field(state, "steps", dict)
        moved = [k for k in self.keys
                 if k != FLOOR_KEY and steps_in.get(k) != self._n_steps_of(k)]
        if moved:
            raise AutoDRError(f"fixed-dr state has boundaries off their maximum: {moved}")
        super().load_state_dict(dict(state, format="autodr-3"))


class RowTable:
    """A pre-drawn evaluation table: N rows, 16 columns, in [0, 1]."""

    def __init__(self, rows: list, sha256: str, path: str = "") -> None:
        if not rows:
            raise AutoDRError("evaluation table is empty")
        width = len(TABLE_COLUMNS)
        for i, r in enumerate(rows):
            if len(r) != width:
                raise AutoDRError(f"row {i} has {len(r)} values, expected {width}")
            for v in r:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise AutoDRError(f"row {i} holds {v!r}, which is not a number")
                if not 0.0 <= v <= 1.0:
                    raise AutoDRError(f"row {i} holds {v}, outside [0, 1]")
        self.rows = [list(r) for r in rows]
        self.sha256 = sha256
        self.path = str(path)
        self._cursor = 0
        self._served = 0

    @classmethod
    def from_csv(cls, path) -> "RowTable":
        p = pathlib.Path(path)
        try:
            raw = p.read_bytes()
        except OSError as exc:
            raise AutoDRError(f"cannot read the table at {p}: {exc}") from exc
        sha = hashlib.sha256(raw).hexdigest()
        reader = csv.reader(raw.decode("utf-8").splitlines())
        header = next(reader, None)
        if header is None:
            raise AutoDRError(f"{p} is empty")
        if tuple(h.strip() for h in header) != TABLE_COLUMNS:
            raise AutoDRError(
                f"{p} header does not match TABLE_COLUMNS\n  got      {header}\n"
                f"  expected {list(TABLE_COLUMNS)}"
            )
        try:
            rows = [[float(v) for v in row] for row in reader if row]
        except ValueError as exc:
            raise AutoDRError(f"{p}: a cell is not a number -- {exc}") from exc
        return cls(rows, sha, str(p))

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def take(self, k: int) -> list:
        """The next ``k`` row IDs, wrapping. Order is the table's order."""
        if k < 0:
            raise AutoDRError(f"take({k}): negative count")
        out = [(self._cursor + i) % self.n_rows for i in range(k)]
        self._cursor = (self._cursor + k) % self.n_rows
        self._served += k
        return out

    @property
    def cursor(self) -> int:
        """Where the next `take` starts. Wraps; not a count."""
        return self._cursor

    @property
    def served(self) -> int:
        """How many rows were handed out in total."""
        return self._served

    def reset(self) -> None:
        self._cursor = 0
        self._served = 0


class GridTable(RowTable):
    """A raster table: ``cells * reps`` rows for ONE pair."""

    def __init__(self, rows, sha256, pair, cells: int, reps: int, path: str = "") -> None:
        super().__init__(rows, sha256, path)
        if int(cells) * int(reps) != self.n_rows:
            raise AutoDRError(
                f"grid table has {self.n_rows} rows, expected cells*reps = {cells}*{reps}"
            )
        if int(cells) < 1 or int(reps) < 1:
            raise AutoDRError(f"cells and reps must be >= 1, got {cells} and {reps}")
        pair = tuple(pair)
        if len(pair) != 2 or pair[0] == pair[1]:
            raise AutoDRError(f"a grid pair is two DISTINCT quantities, got {pair}")
        for name in pair:
            if name not in DR_DIM_NAMES:
                raise AutoDRError(
                    f"grid pair names {name!r}, which is not one of the "
                    f"randomised quantities {DR_DIM_NAMES}"
                )
        self.pair = pair
        self.cells = int(cells)
        self.reps = int(reps)

    def cell_of(self, row_id: int) -> int:
        return row_id // self.reps

    def rep_of(self, row_id: int) -> int:
        return row_id % self.reps


def map_unit_to_bounds(unit, lo, hi):
    """Map a uniform draw in [0, 1] onto ``[lo, hi]``."""
    return lo + unit * (hi - lo)


def apply_boundary(unit, rows, cols, side):
    """Nail one column of the unit sample to its boundary."""
    unit[rows, cols] = side
    return unit


def table_rows(table_unit, env_ids):
    """The evaluation rows of the resetting envs: env ``i`` plays row ``i``."""
    return table_unit[env_ids]


def boundary_assignment(rand_a, rand_b, n_boundaries: int, p_boundary: float):
    """``(is_boundary, index)`` for a batch of resetting envs."""
    if n_boundaries < 1:
        raise AutoDRError(f"n_boundaries must be >= 1, got {n_boundaries}")
    if not 0.0 <= p_boundary <= 1.0:
        # Out of range this silently makes every env, or no env, a boundary
        # env; NaN makes none of them and raises nothing.
        raise AutoDRError(f"p_boundary must be in [0, 1], got {p_boundary}")
    is_boundary = rand_a < p_boundary
    index = (rand_b * n_boundaries).long().clamp(0, n_boundaries - 1)
    return is_boundary, index
