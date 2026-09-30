# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Automatic Domain Randomization -- the boundary state machine (Phase 5).

WHY THIS FILE IS NEW CODE
-------------------------
The RULE is not invented. It is Algorithm 1 of Akkaya et al. 2019 ("Solving
Rubik's Cube with a Robot Hand", arXiv:1910.07113): every randomised quantity
gets a lower and an upper distribution boundary, each boundary owns a buffer
of success flags collected on episodes NAILED to that boundary, and a full
buffer moves the boundary out or in. Nothing about that is ours.

What IS ours is the wiring, and it is said rather than implied:

  * The proxy repo has no ADR. `curriculum.py` in this package is a manual
    multi-axis LADDER, a different mechanism (rungs, all axes together).
    It is not extended here -- it is left alone and only its two THRESHOLDS
    are imported, so the 80/10 numbers keep exactly one home.
  * Isaac Lab 2.3.2 ships no ADR either. So the state machine is written here.

ONE DIFFERENCE FROM `curriculum.py`, WHICH IS NOT A DEVIATION FROM AKKAYA
--------------------------------------------------------------------------
INCLUSIVE THRESHOLDS. Algorithm 1 raises the boundary when the buffer
average is `>= t_H` and lowers it when it is `<= t_L` -- VERIFIED verbatim
from the ar5iv HTML on 2026-09-09:
  "if p>=tH then phi_i <- phi_i + Delta else if p<=tL then phi_i <- phi_i - Delta".
`Ladder.update` is STRICT on both ends (exactly 0.80 holds). This file
follows AKKAYA, because this file implements Akkaya, so the two files
answer 0.80 differently on purpose. Against Algorithm 1 this is a MATCH,
not a deviation, and it is listed here rather than below for that reason.
NAMED CONFLICT INSIDE THE PAPER: section 5.2's prose says the boundary
moves when performance is "better than the high threshold t_H" / "worse
than the low threshold t_L" -- STRICT, where the pseudocode is inclusive.
The pseudocode is followed because it is the normative statement of the
algorithm, and the disagreement is recorded rather than resolved by
plausibility. At m = 240 it can only matter on exact ties (192/240 and
24/240).

EIGHT DEVIATIONS FROM ALGORITHM 1 -- ALL OF THEM NAMED
-------------------------------------------------------
Read literally, Algorithm 1 is not what this file runs. Every difference is
listed here, because a deviation that lives only in the code is a deviation
nobody decided.

(1) THE PERFORMANCE MEASURE IS A DIFFERENT QUANTITY. Algorithm 1 buffers
    whatever `EvaluatePerformance` returns; in the Rubik's-cube task that is
    a COUNT of successes per episode, and `t_H` / `t_L` are counts. This file
    buffers a BINARY success flag and compares the buffer's MEAN against
    0.80 / 0.10 -- IndustReal's SBC rates (D-110 point (3)), imported from
    `curriculum.py` so those two numbers keep one home. Adapting the measure
    is what makes the two rates meaningful at all, and it is the largest of
    the eight.
    VERIFIED: section 5.1 of the PDF says the thresholds are "configured as
    the lower and upper bounds on the number of successes in an episode",
    and Table 15 gives t_H = 20 / t_L = 10 -- counts, not rates.

(2) BOUNDARY-ENV SELECTION IS UNIFORM OVER BOUNDARIES. Akkaya picks the
    DIMENSION first and then the side. With three one-sided quantities of
    five that would give each one-sided boundary (`lat_r_hi`, `tilt_hi`,
    `start_height_hi`) a weight of 1/5 while every two-sided boundary gets
    1/10. This file draws uniformly over the SEVEN boundaries (each 1/7).
    That is NOT what the plan's fill-rate arithmetic assumed -- corrected
    2026-09-12: the plan divided by ELEVEN, because D-178 had not yet cut
    the lateral box to a disk. The consequence is worked through at
    `DELTA_STEPS` below.

(3) THE SIGN OF DELTA IS RESOLVED AS "AWAY FROM THE CENTRE". Algorithm 1
    prints `phi_i <- phi_i + Delta` unsigned, which read literally would
    NARROW a lower boundary on success. This file moves every boundary
    outwards on success and inwards on failure. That matches the paper's
    intent, but the pseudocode does not say it, so the reading is ours.

(4) A CEILING PER BOUNDARY, NOT ONE GLOBAL CEILING. Algorithm 1's
    pseudocode carries no clamp on phi, but the RUNS did clamp: Appendix
    C.3 Table 15 lists "Maximum value of a phi  4.0" (READ from the PDF on
    2026-09-09, arXiv:1910.07113 p. 44). Having a ceiling is therefore ADR
    practice, not our addition. What IS ours is that every boundary gets
    its OWN ceiling in its own physical unit, from plan section 2, so that
    "all boundaries at maximum" can be the stop rule. Akkaya's single 4.0
    is a bound on a normalised phi, one number for every parameter.
    The warning still stands: where OUR ceilings sit is a number we chose
    (the four pose and height ceilings, all `[TESTWERT]`), so the report must
    never present a ceiling as a result. The friction band is not one of
    them -- it is the span of a source (D-181 (2)) and carries no marker.
    An earlier version of this docstring claimed ADR sets no upper limit at
    all. That was a universal negative about an appendix it had not read,
    and it was wrong.

(5) A FLOOR AT WIDTH 0. Symmetrically, a retreat at width 0 clamps instead
    of inverting the range. Plan section 2 asks for it; Algorithm 1 has no
    such rule.

(6) THE STEP IS RELATIVE, NOT ABSOLUTE. Table 15 gives ONE step for every
    parameter: "ADR step size  0.02" on a normalised phi capped at 4.0,
    i.e. 200 steps from the calibrated point to the cap (OUR arithmetic;
    the paper states neither figure). Here each boundary walks its range in
    `n_steps` equal parts, so the step carries the quantity's own unit and
    every boundary needs the same number of buffer fills to open fully.
    That is what makes plan risk R2 checkable ("do the boundaries move at
    all in 1500 iterations"), and it is why `DELTA_STEPS` is `[TESTWERT]`:
    10 comes from a fill-rate estimate, not from Akkaya and not from a
    measurement.

(7) A FULL BUFFER DROPS, IT DOES NOT QUEUE. Algorithm 1 tests the buffer
    length after EVERY append, so overflow cannot arise there. Here
    `record` and `update` are separate calls, so a logging pass with more
    than `buffer_m` boundary episodes on one boundary loses the excess.
    See `AutoDR.record` for why queueing them would be worse, and
    `dr/flags_dropped` for the curve that must read 0.

(8) ADR DOES NOT OWN EVERY RANDOMISATION. In Akkaya the randomised set IS
    the ADR parameter set (section 5.4, Appendix B Tables 9-12). Here eleven
    of the sixteen table columns stay at a FIXED width beside it: the tilt
    azimuth, the lateral angle `lat_phi` (D-178 (2)), `fixture_pos_noise_xy`
    x and y, the six joint-noise channels, and the grasp belief error
    `grasp_obs_x` (D-183) (see `TABLE_COLUMNS`). So a
    boundary episode is not "this boundary nailed, everything else drawn
    from P_phi" -- it is that plus a fixed manual randomisation, and that
    fixed part feeds the buffered success rate like anything else. Plan
    section 1 lists the azimuth, the fixture noise and the joint noise as
    "nicht adaptiert, aber vorhanden" and puts them in the test table;
    `lat_phi` joined them with D-178. This is the same fact stated as what
    it costs.

WHAT IS *NOT* A DEVIATION
-------------------------
`BUFFER_M = 240` and `P_BOUNDARY = 0.5` are Akkaya's own numbers, verbatim
from Appendix C.3 Table 15: "Performance queue length m  240" and
"Boundary sampling probability  0.5". No citation debt is outstanding on
either.

WHAT A BOUNDARY IS -- THREE LEVELS, NEVER MIXED (plan section 1)
----------------------------------------------------------------
  physical quantity   one number drawn per episode. There are FIVE:
                      lat_r, yaw, tilt, start_height, friction.
  boundary            one edge of that quantity's distribution. There are
                      SEVEN, not ten: `lat_r`, `tilt` and `start_height` are
                      ONE-SIDED -- each opens upward from its centre only
                      (the disk radius from 0, D-178; the tilt magnitude from
                      0; the start height from H_min, D-179), so its lower
                      edge IS the centre and can never move.
  buffer              one success buffer per MOVABLE boundary: seven.

The tilt AZIMUTH and the lateral ANGLE `lat_phi` are NOT in this table. Each
is an independent draw, uniform over 0..2*pi, with no boundary and no buffer
-- the fixture tilts in every direction and the lateral offset points in
every direction from the first episode onwards. Each still owns a column in
the evaluation table (`TABLE_COLUMNS`), because it is a random number that
affects a reset condition.

The GRASP BELIEF ERROR `grasp_obs_x` (D-183) is a third column of that kind,
and the one whose membership is not obvious: it moves no part. It is the
per-episode error in where the policy BELIEVES the suction cups hold the
part, applied to the observation alone. It owns a column for the same reason
the other two do -- the evaluation replays a row so that two policies meet
the identical episode, and an episode is not identical if the belief it
starts from is re-drawn. Keeping it out would also make `grasp_obs_x = 0`
consume a random number that the table-less run does not: a FIXED-width row
consumes the same count whatever the magnitude is, so the RNG stream stops
depending on it at all.

WIDTHS ARE COUNTED IN STEPS, NOT ACCUMULATED IN FLOATS
-------------------------------------------------------
A boundary is stored as an INTEGER step count in `[0, n_steps]`, and its
value is `centre + (steps / n_steps) * (bound_max - centre)`. Ten additions
of `max/10` in floating point do not land on `max`; an integer count does,
exactly. "All boundaries at maximum" (the stop rule, plan section 4) is then
an integer comparison with no epsilon, and clamping is `min`/`max` on ints.

`bounds_version` RISES ONLY ON A REAL CHANGE
---------------------------------------------
Plan section 4 and risk R12. An expansion attempt on a boundary that already
sits at its maximum -- or a retreat at width 0 -- changes no number and must
NOT bump the version, otherwise the "fresh window" is invalidated forever and
the run can never stop.

FOUR MAXIMA ARE `[TESTWERT]`; THE FRICTION BAND IS SOURCED
-----------------------------------------------------------
The ceilings of `lat_r`, `yaw`, `tilt` and `start_height` are DECIDED as the
Phase-5 targets (D-178 (5), D-179 (1)) -- decided is not derived, so all four
stay marked; the lower start-height limit is a placeholder that `bind`
overwrites with the centre (D-179). The friction band is the span of the
DuPont PA66/POM tables for an ASSUMED PA6/POM pairing (D-181 (2)): it HAS a
source, so it carries NO marker (user, 2026-09-12). Each entry's own `source`
string says which it is. A later decision replaces either HERE and nowhere
else.

The CENTRES are not in this file. They arrive as arguments from the task
config (`insertion_tasks_cfg.py` / the env config), because that is where the
task's geometry and material facts already live (`CONTACT_FRICTION`, the
start height). One home per fact.

Plain Python, stdlib only -- no torch, no Isaac, no numpy. The three array
helpers at the bottom take tensors but import nothing: they use only
operators and indexing that the caller's tensor type provides. That keeps the
whole file importable on the dev laptop while staying idiomatic torch on the
training machine.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from dataclasses import dataclass, replace

from .curriculum import ADVANCE_AT, RETREAT_AT

# The marker for a number that is decided-but-not-derived. Same contract as
# the `A` provenance tag in insertion_tasks_cfg.py: it does not stop the run,
# it makes the run REPORT that it is provisional.
TESTWERT = "[TESTWERT]"

# --- Akkaya Algorithm 1 constants ------------------------------------------
AKKAYA_CONSTANT_SOURCE = (
    "Akkaya et al. 2019, Appendix C.3 Table 15 (arXiv:1910.07113 p. 44), "
    "read from the PDF 2026-09-09: queue length m 240, boundary sampling "
    "probability 0.5"
)
BUFFER_M = 240
P_BOUNDARY = 0.5
# Delta per boundary = (bound_max - centre) / DELTA_STEPS. This is deviation
# (6) -- corrected 2026-09-12, it read "(7)", which is the drop-do-not-queue
# rule. Akkaya's Table 15 gives ONE absolute step, 0.02 on a normalised phi
# capped at 4.0 (200 steps from the centre to the cap); ours is relative,
# so every boundary
# opens in the same NUMBER of buffer fills whatever its unit. 10 is
# [TESTWERT] -- the plan marks it too (section 1) -- and comes from a
# fill-rate ESTIMATE, not from any measurement.
# THE ESTIMATE WAS RE-DONE 2026-09-12 AND ITS RESULT MOVED. The plan's
# arithmetic (`fragen-in-chat-plan-parallel-creek.md` section 1) divided by
# ELEVEN boundaries and got "roughly 18 fills per boundary in 1500
# iterations". D-178 cut the boundary count to SEVEN, and the divisor is the
# only input that changed: with the plan's own other inputs -- 1024 envs,
# 16 steps per env per iteration, a 256-step episode, so 64 finished episodes
# per iteration, times P_BOUNDARY 0.5, against BUFFER_M 240 -- it is
# 64*0.5/7 = 4.571 flags per boundary per iteration, 240/4.571 = 52.5
# iterations per fill, and 1500/52.5 = ROUGHLY 29 FILLS, not 18. The three
# run inputs are NOT owned by this file (they are run parameters), so the
# number moves with them. Confirm against `dr/<boundary>_fill` after the
# smoke run; that measurement, not this estimate, is what DELTA_STEPS = 10
# has to survive.
DELTA_STEPS_SOURCE = (
    "[TESTWERT] Phase-5 plan section 1: Delta = maximum/10 from a fill-rate "
    "estimate, NOT Akkaya's absolute 0.02; confirm after the smoke run"
)
DELTA_STEPS = 10

# --- the evaluation-table layout (plan section 3) --------------------------
# SIXTEEN columns: one per random number drawn at reset that an evaluation
# must be able to replay -- the reset CONDITIONS, and since D-183 the grasp
# belief error the episode's first observation carries.
# This tuple is the ONE home of that layout. The env reset reads it from
# here, and so must every table tool written later, so a column can never be
# added in one place only.
TABLE_COLUMNS: tuple[str, ...] = (
    "lat_r",            # 0  lateral start offset RADIUS (disk, D-178)
    "lat_phi",          # 1  lateral start offset ANGLE -- no boundary, no buffer
    "yaw",              # 2  fixture yaw
    "tilt",             # 3  fixture tilt MAGNITUDE
    "tilt_azimuth",     # 4  fixture tilt DIRECTION -- no boundary, no buffer
    "start_height",     # 5  tip height above the entrance
    "friction",         # 6  contact friction, static = kinetic
    "fixture_noise_x",  # 7  fixture_pos_noise_xy, x   -- not adapted
    "fixture_noise_y",  # 8  fixture_pos_noise_xy, y   -- not adapted
    "joint_noise_0",    # 9  reset_joint_noise, 6 joints -- not adapted,
    "joint_noise_1",    # 10 scale is 0.0 today. The columns exist anyway so
    "joint_noise_2",    # 11 a table stays valid when the scale is turned on.
    "joint_noise_3",    # 12
    "joint_noise_4",    # 13
    "joint_noise_5",    # 14
    "grasp_obs_x",      # 15 grasp belief error, OBSERVATION only (D-183) --
                        #    no boundary, no buffer; see the module docstring
)
COLUMN_INDEX: dict = {name: i for i, name in enumerate(TABLE_COLUMNS)}


class AutoDRError(RuntimeError):
    """A randomisation table that cannot be built is a hard stop, never a default."""


def _field(state, key, kind):
    """Read one field of a resume state, or refuse BY NAME.

    A truncated or hand-edited ``autodr_<it>.json`` is exactly what a resume
    meets, and a bare ``KeyError`` refuses by accident: it names no cause and
    no file. Types are checked rather than coerced, because ``int(3.9)`` is
    3 -- a resume that silently continues at a different width than the file
    states is the one thing ``state_dict`` promises cannot happen.
    """
    if not isinstance(state, dict):
        raise AutoDRError(
            f"resume state must be a dict, got {type(state).__name__}"
        )
    if key not in state:
        raise AutoDRError(f"resume state has no {key!r}")
    v = state[key]
    if kind is int:
        # bool is an int subclass; True would silently become 1.
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
    # No isinstance guard here: `_field` already refuses a non-dict by name,
    # and a second copy of that check is a branch no mutation can reach.
    return {_field(e, "name", str): e for e in entries}


@dataclass(frozen=True)
class DimSpec:
    """One randomised physical quantity and how far it is allowed to open.

    ``centre`` is the width-0 value: the number every run starts at and the
    number a No-DR run keeps. It is ``None`` in ``DR_DIMS`` and MUST be bound
    from the task config before the table can be used -- an invented centre
    would randomise around a pose nobody decided.

    ``lo_max`` / ``hi_max`` are the outermost values the boundaries may ever
    reach. For a one-sided quantity ``bind`` writes the centre into
    ``lo_max`` (D-179 (2)); the table's own ``lo_max`` is only a placeholder.

    ``source`` is free text, exactly like the M/R/A/P provenance tags on the
    constants in ``insertion_tasks_cfg.py``. It is not machine-checkable; what
    IS checkable is that a spec whose source contains ``[TESTWERT]`` shows up
    in ``testwert_dims()`` and therefore in the startup report.
    """

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
        """Return a copy carrying the centre from the task config.

        This is the one gate every table passes through, so it is also
        where the spec's own shape is checked: ``n_steps == 0`` divides by
        zero in ``bound_at``, a negative one makes ``all_at_max`` true at
        width 0, and inverted maxima give a lower edge above the upper one.
        """
        if self.n_steps < 1:
            raise AutoDRError(
                f"dimension {self.name!r}: n_steps must be >= 1, got {self.n_steps}"
            )
        # D-179 (2): a one-sided quantity's lower edge IS its centre, so the
        # centre becomes `lo_max` BEFORE every range check below. NAMED LOSS:
        # any centre the env passes for a one-sided quantity silently becomes
        # its lower edge; the text pin of the env's centre dict in
        # check_env_wiring is what still pins those centres.
        if self.one_sided:
            self = replace(self, lo_max=float(centre))
        if self.lo_max > self.hi_max:
            raise AutoDRError(
                f"dimension {self.name!r}: lo_max {self.lo_max} is above hi_max {self.hi_max}"
            )
        c = float(centre)
        for side in self.sides:
            if self.bound_max(side) == c:
                # A MOVABLE boundary that already sits on the centre has a
                # delta of 0: every buffer would report a move that changes
                # no number, and `bounds_version` would climb for nothing.
                # If a quantity really is one-sided, say so with
                # `one_sided=True` instead of collapsing one of its maxima.
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


# ---------------------------------------------------------------------------
# THE TABLE. Five quantities, seven movable boundaries.
#
# NOT ONE OF THE FOUR CEILINGS BELOW IS DERIVED, and the five entries do not
# all have the same standing -- each entry's own `source` string says which:
#   lat_r, yaw, tilt, start_height.hi_max -- ceilings DECIDED as the Phase-5
#                                            targets (D-178 (5), D-179 (1))
#   start_height.lo_max                   -- a placeholder that `bind`
#                                            overwrites with the centre (D-179)
#   friction lo/hi                        -- the span of the DuPont PA66/POM
#                                            tables, for an ASSUMED PA6/POM
#                                            pairing (D-181 (2))
# The four ceilings keep their `[TESTWERT]` marker; the friction band has a
# source and is NOT marked (user, 2026-09-12). A decision replaces either
# HERE, and nowhere else.
# ---------------------------------------------------------------------------
_D178 = "[TESTWERT] Phase-5 ceiling, decided not derived -- D-178 (5)"

DR_DIMS: tuple[DimSpec, ...] = (
    # ONE-SIDED. The RADIUS of a disk (D-178 (1)-(2)): the env maps it with
    # insertion_math.disk_offset, uniform over the area; the angle is the
    # separate `lat_phi` column and is never an AutoDR boundary.
    DimSpec("lat_r", "m", 0.0, 0.030, True, _D178),
    DimSpec("yaw", "rad", -0.08726646259971648, 0.08726646259971648, False, _D178),
    # ONE-SIDED. The magnitude is drawn from [0, hi]; the direction is the
    # separate `tilt_azimuth` column and is never an AutoDR boundary.
    DimSpec("tilt", "rad", 0.0, 0.17453292519943295, True, _D178),
    # ONE-SIDED (D-179). The centre is H_min, whose one home is
    # insertion_env_cfg.start_tip_above_entrance; lo_max 0.0 is a documented
    # placeholder that `bind` overwrites with that centre.
    DimSpec("start_height", "m", 0.0, 0.120, True,
            "[TESTWERT] hi = Phase-5 ceiling, decided not derived -- D-179 (1); "
            "lo = the centre H_min, not yet measured -- D-179 (4)"),
    # Centre is CONTACT_FRICTION (D-181 (3)). The BAND is the span of the
    # DuPont PA66-on-POM tables for an ASSUMED PA6-on-POM pairing; no primary
    # value is PA6.
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


# ---------------------------------------------------------------------------
# THE START FLOOR (SBC step 0, IndustReal sec. IV.G, `z_low` raised with
# success). An EIGHTH boundary with the opposite sense: "expand" moves the
# floor UP, towards H_min. Its key is `start_height_floor` so that the
# provider's `rsplit("_", 1)` geometry resolves it to the start_height column
# with side 0.0 (a lower edge) WITHOUT touching boundary_columns,
# boundary_sides, apply_boundary or the env's reset draw. While the floor is
# below H_min (phase "floor") every other boundary sits on its centre and every
# boundary episode is nailed to the floor; once it reaches H_min (phase "dr")
# the provider behaves exactly as it did without a floor. Plan:
# Pläne/SBC_Schritt0_Entwurf.md (2026-09-13/14).
# ---------------------------------------------------------------------------
FLOOR_KEY = "start_height_floor"
FLOOR_DIM = "start_height"


@dataclass(frozen=True)
class FloorSpec:
    """The start floor: ``f0`` at step 0, ``top`` (= H_min) at step ``n_steps``.

    ``top`` must be the bound centre of the ``start_height`` dimension -- one
    home for H_min (D-179 (3)); the provider refuses anything else.
    """

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
    """Every quantity whose limits are decided-but-not-derived.

    The startup report prints this, so a run can never quietly look finished.
    """
    return tuple(d.name for d in dims if TESTWERT in d.source)


def provenance_lines(dims: tuple = DR_DIMS) -> tuple:
    """The provenance report, one line per un-derived number.

    Every `[TESTWERT]` in this module reaches a human through here, so a run
    can never quietly look finished -- the same contract `WORKCELL_PENDING`
    has in `insertion_tasks_cfg.py`.

    WIRED since Phase 5 step B2: `InsertionEnv._print_startup_report` prints
    every line whenever a provider exists. That report runs at the observation
    calls `InsertionEnvCfg.report_at_steps` names, so the lines appear shortly
    AFTER the run starts, not before it -- a run that dies in `__init__` prints
    none of them. The other caller is `scripts/check_autodr.py`.
    """
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
    action: str      # expand | shrink | hold | clamped_max | clamped_zero
    rate: float
    before: float
    after: float
    buffer_m: int = BUFFER_M
    # Free text appended to the log line; the floor's transition uses it.
    note: str = ""

    @property
    def moved(self) -> bool:
        """Did a NUMBER change? Not "was a move attempted".

        MEASURED 2026-09-09: with a two-sided quantity whose lower maximum
        equals its centre -- exactly what the docstring records for RT-177's
        start height -- delta is 0.0, so ten "expansions" left the bounds at
        (0.030, 0.030) and still drove `bounds_version` to 10. That is the
        R12 failure this module claims to prevent: a fresh window
        invalidated by a boundary with nowhere to go, and a run that can
        never stop. `bind` refuses that table now, and this is the runtime
        backstop for any other way of reaching a zero-width step.
        """
        return self.action in ("expand", "shrink") and self.after != self.before

    def line(self) -> str:
        """The marked log line. `/rt-check` greps for the `[autodr]` prefix."""
        return (
            f"[autodr] {self.key}: rate {self.rate:.4f} over {self.buffer_m} "
            f"-> {self.action} {self.before:.6g} -> {self.after:.6g}" + self.note
        )


class _Provider:
    """Shared surface of every bounds provider (the plan's `BoundsProvider`).

    Every provider answers the same questions, so the env never asks "which
    mode am I in". Sub-classes override only what actually differs.
    """

    p_boundary: float = 0.0

    def __init__(self, dims: tuple) -> None:
        # Materialise FIRST. ``if not dims`` is False for a generator, the
        # loop below would then consume it, ``tuple(dims)`` would come out
        # empty, and ``all_at_max()`` -- ``all([])`` -- would answer True.
        # That value feeds the plan's stop rule, so an empty provider would
        # stop a run that randomised nothing, and raise nothing on the way.
        dims = tuple(dims)
        if not dims:
            raise AutoDRError("no dimensions given")
        names = [d.name for d in dims]
        if len(set(names)) != len(names):
            # A repeated name gives `keys` two entries that share ONE buffer:
            # the boundary draw's 1/n weights go wrong, two indices alias, and
            # record/update under-count. Nothing downstream notices.
            raise AutoDRError(f"duplicate dimension name in {names}")
        for d in dims:
            if d.name not in COLUMN_INDEX:
                # Caught here rather than as a bare KeyError out of
                # bounds_arrays() half a run later.
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

    # -- geometry of the boundary set ---------------------------------------
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

    # -- what the env needs at reset ----------------------------------------
    def bounds(self) -> dict:
        """``{name: (lo, hi)}`` in the quantity's own unit."""
        raise NotImplementedError

    def bounds_max(self) -> dict:
        """The WIDEST ``{name: (lo, hi)}`` this provider can ever reach.

        ``bounds()`` answers "what does the reset draw from RIGHT NOW";
        this answers "what could it ever draw from before the run ends".
        The two differ for AutoDR and only for AutoDR: it starts at width 0
        and opens later, so at iteration 0 ``bounds()`` reports a zero span
        for every quantity even though the pocket will tilt.

        The env allocates its per-env buffers ONCE, in ``__init__``, so a
        buffer that is sized from ``bounds()`` is sized from the width-0
        answer and is then missing for the whole run. Buffer allocation and
        every other once-per-run branch must ask THIS method instead.

        ABSTRACT ON PURPOSE, unlike a one-line default would be. The answer
        is a SAFETY property -- it turns the env's pocket-frame transform and
        its per-reset fixture write on and off -- and the two right answers
        pull in opposite directions: a provider whose boundaries move reaches
        the dimension maxima, a provider whose width is fixed reaches exactly
        its own bounds. A default of either kind is silently wrong for the
        other, and a new provider that forgot to answer would inherit that
        wrongness with no error. Each provider says it in one line.
        """
        raise NotImplementedError

    def bounds_arrays(self) -> tuple:
        """``(lo, hi)`` as two 16-long lists, one entry per table column.

        Columns that are not AutoDR quantities get ``(0.0, 1.0)``, i.e. the
        unit sample passes through untouched -- the env scales those itself.
        """
        lo = [0.0] * len(TABLE_COLUMNS)
        hi = [1.0] * len(TABLE_COLUMNS)
        for name, pair in self.bounds().items():
            lo[COLUMN_INDEX[name]] = pair[0]
            hi[COLUMN_INDEX[name]] = pair[1]
        return lo, hi

    def boundary_assignment(self, rand_a, rand_b):
        """Which resetting envs are boundary envs, and on which boundary.

        Tensors in, tensors out; no Python loop over envs. ``rand_a`` and
        ``rand_b`` are two independent uniform draws of shape ``(n,)``.
        """
        return boundary_assignment(rand_a, rand_b, self.n_boundaries, self.p_boundary)

    # -- the rule ------------------------------------------------------------
    def record(self, key: str, success: bool) -> None:
        raise NotImplementedError

    def update(self) -> list:
        raise NotImplementedError

    def all_at_max(self) -> bool:
        raise NotImplementedError

    # -- telemetry -----------------------------------------------------------
    def as_dict(self) -> dict:
        raise NotImplementedError

    def scalars(self) -> dict:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError


class AutoDR(_Provider):
    """Akkaya Algorithm 1 over the table's boundaries (seven for ``DR_DIMS``).

    A buffer never stores its flags. It only ever needs how many flags it
    holds and how many of them were successes, because the only question
    asked of it is the mean of a FULL buffer. Two integers per boundary make
    ``state_dict`` exact and make a resume bit-identical.
    """

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
        # STALL REVIEW (user, 2026-09-14). A boundary whose full buffers keep
        # landing between the thresholds never writes a move and never
        # bumps the version -- it just sits there. `stall_buffers` full
        # buffers IN A ROW without a real move mark the event line with a
        # STALL note. The rule changes NOTHING; it is a review trigger for
        # the reader. 0 = no note ever.
        self.stall_buffers = int(stall_buffers)
        self.buffer_m = int(buffer_m)
        self.p_boundary = float(p_boundary)
        self.advance_at = float(advance_at)
        self.retreat_at = float(retreat_at)
        # THE FLOOR (SBC step 0). Appended LAST so that the seven boundaries
        # keep indices 0..6 and the dr-phase draw excludes it by count.
        self.floor = None
        if floor is not None:
            floor.check()
            sh = self._dim_of.get(FLOOR_DIM)
            if sh is None:
                raise AutoDRError(
                    f"a floor needs the {FLOOR_DIM!r} dimension in the provider"
                )
            if float(floor.top) != float(sh.centre):
                # One home for H_min (D-179 (3)): the floor's top IS the
                # start_height centre, never a second number.
                raise AutoDRError(
                    f"floor top {floor.top} != start_height centre {sh.centre}"
                )
            self.floor = floor
            self.keys = tuple(self.keys) + (FLOOR_KEY,)
        # Width 0: every boundary sits on the centre.
        self._steps = {k: 0 for k in self.keys}
        self._n = {k: 0 for k in self.keys}
        self._s = {k: 0 for k in self.keys}
        self._dropped = {k: 0 for k in self.keys}
        # Rate of the LAST FULL buffer per key, -1.0 before the first one
        # (a rate is never negative, so -1 cannot be mistaken for one); and
        # the count of full buffers IN A ROW that did not move the key.
        self._last_rate = {k: -1.0 for k in self.keys}
        self._holds = {k: 0 for k in self.keys}

    @staticmethod
    def _check_params(buffer_m, p_boundary, advance_at, retreat_at) -> None:
        """The parameter contract. Called by ``__init__`` AND by ``load_state_dict``.

        A resume is exactly where a truncated or hand-edited
        ``autodr_<it>.json`` turns up, and the loader is the only gate it
        passes. Validating in the constructor alone left ``buffer_m = 0``
        (ZeroDivisionError in ``update``) and an inverted threshold pair
        reachable through the resume path.
        """
        if int(buffer_m) < 1:
            raise AutoDRError(f"buffer_m must be >= 1, got {buffer_m}")
        if not 0.0 <= float(p_boundary) <= 1.0:
            raise AutoDRError(f"p_boundary must be in [0, 1], got {p_boundary}")
        if not 0.0 <= float(retreat_at) < float(advance_at) <= 1.0:
            # Deviation (1) makes the measure a RATE in [0, 1] -- corrected
            # 2026-09-12, it read "(2)", which is the uniform boundary
            # draw. A threshold
            # outside that range passes `retreat_at < advance_at` and then
            # freezes every boundary for the whole run: all_at_max never
            # turns true, the stop rule never fires, and no log line says
            # the run is inert. Akkaya's own t_H = 20 / t_L = 10 are COUNTS
            # and would do exactly that if pasted in here.
            raise AutoDRError(
                f"need 0 <= retreat_at ({retreat_at}) < advance_at ({advance_at}) <= 1"
            )

    # -- reading -------------------------------------------------------------
    def _split(self, key: str) -> tuple:
        name, side = key.rsplit("_", 1)
        return self._dim_of[name], side

    def _n_steps_of(self, key: str) -> int:
        if key == FLOOR_KEY:
            return int(self.floor.n_steps)
        return self._dim_of[key.rsplit("_", 1)[0]].n_steps

    @property
    def phase(self) -> str:
        """``"floor"`` while the floor is below H_min, else ``"dr"``.

        DERIVED from the floor's step count, never stored: a resume that
        loads the steps loads the phase with them.
        """
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
            # THE INVARIANT that replaces the env's static negative-start
            # guards: below H_min every other quantity is exactly its
            # centre (a start inside the pocket with an angle or an offset is
            # a wall contact at reset, RT-120), and the start height is drawn
            # from [floor, H_min].
            out = {d.name: (d.centre, d.centre) for d in self.dims}
            out[FLOOR_DIM] = (self.value(FLOOR_KEY), self._dim_of[FLOOR_DIM].centre)
            return out
        out = {}
        for d in self.dims:
            lo = d.centre if d.one_sided else self.value(f"{d.name}_lo")
            out[d.name] = (lo, self.value(f"{d.name}_hi"))
        return out

    def boundary_assignment(self, rand_a, rand_b):
        """Phase floor: every boundary env is nailed to the floor (the last
        key). Phase dr: drawn over the seven as without a floor -- the floor
        is EXCLUDED by count, or it would take 1/8 of every boundary episode
        for the rest of the run and clamp at max each fill."""
        if self.floor is None:
            return boundary_assignment(rand_a, rand_b, self.n_boundaries, self.p_boundary)
        n_dr = self.n_boundaries - 1
        is_b, index = boundary_assignment(rand_a, rand_b, n_dr, self.p_boundary)
        if self.phase == "floor":
            index = index * 0 + n_dr
        return is_b, index

    def bounds_max(self) -> dict:
        """Every boundary clamps at its dimension maximum, so that IS the reach.

        Independent of the current width, which is the point: at width 0 this
        already reports the full angle span the pocket will eventually have.
        """
        return {d.name: (d.lo_max, d.hi_max) for d in self.dims}

    def fill(self) -> dict:
        return dict(self._n)

    def dropped(self) -> dict:
        """Flags that arrived at a full buffer, per boundary. Must stay 0."""
        return dict(self._dropped)

    def all_at_max(self) -> bool:
        """Integer comparison -- no epsilon, because the steps are integers."""
        return all(self._steps[k] >= self._n_steps_of(k) for k in self.keys)

    # -- the rule ------------------------------------------------------------
    def record(self, key: str, success: bool) -> None:
        """One finished BOUNDARY episode. Called before the env resamples.

        A flag that arrives at a FULL buffer is DROPPED. Algorithm 1 never
        drops, because it tests the buffer length after every single
        append, so overflow cannot arise there. Here ``record`` and
        ``update`` are separate calls -- the env records per finished
        episode and updates once per logging pass -- so a pass carrying
        more than ``buffer_m`` boundary episodes on ONE boundary loses the
        excess.

        Queueing them instead would be worse: a queued flag was measured
        under the OLD boundary and would land in the window of the new one.
        So the excess is dropped AND COUNTED, and the count is published as
        ``dr/flags_dropped``. It must read 0; a rising curve means the
        logging pass is too coarse for the fill rate.
        """
        if key not in self._n:
            raise AutoDRError(f"unknown boundary {key!r}; known: {self.keys}")
        if self._n[key] >= self.buffer_m:
            self._dropped[key] += 1
            return
        self._n[key] += 1
        self._s[key] += 1 if success else 0

    def update(self) -> list:
        """Process every FULL buffer. Returns the events, in key order.

        Akkaya's thresholds are INCLUSIVE (`>=` / `<=`) -- see the module
        docstring for why this differs from ``curriculum.Ladder``.
        """
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
                # HOLD: the rate sits between the thresholds. Before
                # 2026-09-14 this wrote NO event, so a boundary parked at
                # 0.40 was invisible in the log except as a sawtooth in its
                # `_fill` curve. Now it writes a line like every other full
                # buffer; it still moves nothing and bumps no version.
                action = "hold"
            after = self.value(key)
            note = ""
            if key == FLOOR_KEY and action == "expand" and self._steps[key] >= n_steps:
                # The floor stands on H_min: from the next reset on the
                # provider draws as it did without a floor. `/rt-check`
                # greps this.
                note = " -- reached H_min, phase dr"
            ev = BoundEvent(key, action, rate, before, after, self.buffer_m, note)
            # R12: only a REAL change bumps the version. A clamped attempt
            # leaves the fresh window alone, otherwise the run never stops.
            if ev.moved:
                self._bounds_version += 1
                self._holds[key] = 0
            elif action != "clamped_max":
                # A boundary parked at its ceiling is finished, not stalled.
                self._holds[key] += 1
            if 0 < self.stall_buffers <= self._holds[key]:
                ev = BoundEvent(
                    key, action, rate, before, after, self.buffer_m,
                    note + f" -- STALL {self._holds[key]} full buffers without a "
                    f"move (review bar {self.stall_buffers})",
                )
            events.append(ev)
        return events

    # -- telemetry -----------------------------------------------------------
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
            # Stagnation readings (2026-09-14): the rate of the last FULL
            # buffer (-1 until there is one) and the full buffers in a row
            # that did not move the key. A hold writes no `_hi` change, so
            # without these two a stalled boundary has no curve of its own.
            out[f"dr/{key}_last_rate"] = float(self._last_rate[key])
            out[f"dr/{key}_holds"] = float(self._holds[key])
        out["dr/bounds_version"] = float(self._bounds_version)
        out["dr/all_at_max"] = 1.0 if self.all_at_max() else 0.0
        # ONE aggregate curve, not seven: this must read 0 for the whole
        # run, and seven flat zeroes would be seven curves nobody reads.
        out["dr/flags_dropped"] = float(sum(self._dropped.values()))
        if self.floor is not None:
            # 0 = floor phase, 1 = dr phase. Only with a floor, so that a
            # run without one logs exactly the curves it logged before.
            out["dr/phase"] = 0.0 if self.phase == "floor" else 1.0
        return out

    def state_dict(self) -> dict:
        """Everything a resume needs (plan section 5, risk R13)."""
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
        """Refuse anything that is not the SAME boundary set.

        A resume that silently accepts a different table would continue a run
        against boundaries it never trained on.
        """
        if not isinstance(state, dict):
            raise AutoDRError(
                f"resume state must be a dict, got {type(state).__name__}"
            )
        if state.get("format") != "autodr-3":
            # "autodr-1" (no floor block) and "autodr-2" (no stall block) are
            # refused too: no run alive on 2026-09-14 resumes across the
            # bumps, and a silent upgrade would hide a state written by a
            # provider of a different shape.
            raise AutoDRError(f"unknown autodr state format {state.get('format')!r}")
        if tuple(_field(state, "keys", list)) != self.keys:
            raise AutoDRError(
                f"resume boundary set differs: {state['keys']} vs {list(self.keys)}"
            )
        # The floor block must match the configured floor field for field --
        # a floor at another f0 or with another rung count is a different
        # curriculum, and `None` against a floor is a different run shape.
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
        # The SAME contract the constructor enforces. Without this a
        # hand-edited or truncated file put buffer_m = 0 (ZeroDivisionError
        # on the next update), an inverted threshold pair, or p_boundary
        # above 1 into a resumed run, all silently.
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
        # A block carrying a boundary this table does not have is a renamed
        # or hand-duplicated key, i.e. a state from a different run.
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
            # extrapolates happily, so steps = 999 gives a boundary a
            # hundred times past its maximum and all_at_max still reads
            # False, while a negative one puts an UPPER edge below the
            # centre. Neither raises anywhere else.
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
    """Boundaries that never move: the No-DR condition and every evaluation.

    ``p_boundary`` is 0, so no env is ever nailed to an edge, ``record`` and
    ``update`` do nothing, and ``bounds_version`` stays 0 for the whole run.
    """

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
        """Width 0 -- the No-DR condition (plan section 2)."""
        return cls(dims, steps=0)

    @classmethod
    def at_max(cls, dims) -> "FixedWidth":
        """Full width -- what the evaluation table is mapped onto (plan section 3)."""
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
        """Identical to ``bounds()``: a fixed width never opens any further.

        NOT the base class's dimension maxima. A No-DR run (``at_centre``)
        randomises nothing, and inheriting the maxima here would make the
        env allocate the tilt buffers and write a per-episode pocket pose
        for a run whose pocket never moves -- the pre-D-034 regression case
        the env's own comments protect.
        """
        return self.bounds()

    def record(self, key: str, success: bool) -> None:
        return None

    def update(self) -> list:
        return []

    def all_at_max(self) -> bool:
        """Whether the FIXED width already sits at every maximum.

        So ``at_max`` answers True and ``at_centre`` -- the No-DR condition,
        plan section 2 -- answers False. Nothing here ever moves, so neither
        answer changes during a run and neither is a stop condition: the
        plan's stop rule belongs to AutoDR alone, and combining it with a
        fixed width is a config error the env config refuses.
        """
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
            # Same key set as AutoDR: no buffer ever fills here, so "no full
            # buffer yet" (-1) and "no holds" (0) are the true readings.
            out[f"dr/{key}_last_rate"] = -1.0
            out[f"dr/{key}_holds"] = 0.0
        out["dr/bounds_version"] = 0.0
        out["dr/all_at_max"] = 1.0 if self.all_at_max() else 0.0
        # Published as a flat zero rather than omitted: a curve that appears
        # and disappears with `dr_mode` cannot be compared across conditions.
        out["dr/flags_dropped"] = 0.0
        return out

    def state_dict(self) -> dict:
        """Carries the DIMENSION TABLE, not just the width.

        ``at_max`` is the evaluation provider (plan section 3), and the
        whole premise of the shared final table is that every policy is
        scored against identical bounds. A state that carried only the step
        count loaded into a table with different maxima without a word.
        """
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


class RowTable:
    """A pre-drawn evaluation table: N rows, 16 columns, in [0, 1].

    The table is the identity of a test case (plan section 3). Rows are handed
    out in FIXED order, so refilling an env that finished early cannot
    REORDER the cases: the caller gets row IDs back and reports its results
    against them.

    The pointer WRAPS, so it can hand the same id out twice -- a one-row
    table answers `take(3)` with `[0, 0, 0]`. That is deliberate (an env
    that finishes early must get work), and it is the caller's job to stop
    at ``served >= n_rows``: past that point the extra episodes are repeats
    and must not be counted again.

    ``take`` returns IDs, never the rows themselves, so the caller can hold
    the whole table on the GPU and index it -- no Python loop over envs.
    """

    def __init__(self, rows: list, sha256: str, path: str = "") -> None:
        if not rows:
            raise AutoDRError("evaluation table is empty")
        width = len(TABLE_COLUMNS)
        for i, r in enumerate(rows):
            if len(r) != width:
                raise AutoDRError(f"row {i} has {len(r)} values, expected {width}")
            for v in r:
                # bool is an int subclass, so True would pass the range test
                # and then scale to the boundary's upper edge.
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise AutoDRError(f"row {i} holds {v!r}, which is not a number")
                if not 0.0 <= v <= 1.0:
                    raise AutoDRError(f"row {i} holds {v}, outside [0, 1]")
        # A COPY: the caller keeps its list, and a later write through it
        # cannot walk back the range and width validation above.
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
            # A bare ValueError refuses by accident and names no cause --
            # the same doctrine the check script applies to its own
            # `refuses()` helper.
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
        """How many rows were handed out in total. Monotone, never wraps.

        The refill bookkeeping of plan section 8 needs the COUNT: it is how
        the eval knows every row has been played. The cursor wraps, so it
        answered 2 after seven rows on a five-row table.
        """
        return self._served

    def reset(self) -> None:
        self._cursor = 0
        self._served = 0


class GridTable(RowTable):
    """A raster table (plan section 8): ``cells * reps`` rows for ONE pair.

    Row ``i`` is repetition ``i % reps`` of cell ``i // reps``. The two raster
    quantities are fixed per cell; the disturbance columns vary between the
    repetitions, so every policy sees the same repetitions.
    """

    def __init__(self, rows, sha256, pair, cells: int, reps: int, path: str = "") -> None:
        super().__init__(rows, sha256, path)
        if int(cells) * int(reps) != self.n_rows:
            raise AutoDRError(
                f"grid table has {self.n_rows} rows, expected cells*reps = {cells}*{reps}"
            )
        if int(cells) < 1 or int(reps) < 1:
            # Negative sizes build happily and then make cell_of() return
            # negative ids, which index a raster map from the end.
            raise AutoDRError(f"cells and reps must be >= 1, got {cells} and {reps}")
        # C(5, 2) = 10 PAIRS over the FIVE randomised quantities of D-178 /
        # D-179 (plan section 8 counts 15 over the pre-D-178 six). Neither the
        # tilt azimuth nor the lateral angle `lat_phi` is a pair partner. A
        # triple, a repeat, or a disturbance column would silently mislabel a
        # map.
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


# ---------------------------------------------------------------------------
# The array helpers. Tensors in, tensors out; this module imports NOTHING, so
# all three work with real torch on the training machine and with the numpy
# stand-in (scripts/tools/torch_shim.py) on the laptop.
# ---------------------------------------------------------------------------
def map_unit_to_bounds(unit, lo, hi):
    """Map a uniform draw in [0, 1] onto ``[lo, hi]``.

    ``map(0) == lo`` EXACTLY, always. ``map(1) == hi`` only USUALLY -- and the
    docstring claimed both until it was measured on 2026-09-10.

    MEASURED on 2026-09-10 over all 616 ``(lo, hi)`` pairs the eleven
    boundaries of the PRE-D-178 table could reach -- not re-measured for the
    seven boundaries of D-178/D-179 -- in float32, which is the dtype the env
    computes in. The lateral RADIUS no longer goes through this map:
    ``insertion_math.disk_offset`` lands ``u_r = 1`` on the radius exactly.
    On that table:

    * ``map(0) != lo``:   0 of 616, in float32 and in float64. ``lo + u*0`` is
      exact for every finite ``u``.
    * ``map(1) != hi``: 151 of 616 in float32 (132 of 616 in float64), worst
      error 2.98e-8 -- e.g. friction lo-step 10 / hi-step 3, where ``hi`` is
      0.46000000834 and ``map(1)`` is 0.45999997854. ``lo + (hi - lo)`` is the
      ordinary lerp endpoint problem and it does not go away.

    So nailing an upper boundary lands the episode within 3e-8 of the edge,
    not ON it. That is physically nothing -- 3e-10 m on a 6 mm lateral band --
    and it is NOT nothing for a claim, so the claim is the measured one.

    THE FORM IS STILL THE RIGHT ONE, and that was measured against its
    alternative rather than argued: the symmetric ``mid + (2u-1)*half`` misses
    ``lo`` in 215 of 616 AND ``hi`` in 132 of 616, with the same worst error.
    This form is exact at one end always and misses the other less often.
    """
    return lo + unit * (hi - lo)


def apply_boundary(unit, rows, cols, side):
    """Nail one column of the unit sample to its boundary. In place.

    ``rows`` / ``cols`` select the boundary envs and the column each one is
    nailed in; ``side`` is 0.0 for a lower edge and 1.0 for an upper edge.
    This is the ONE place the 0 = lo / 1 = hi convention is written down.
    """
    unit[rows, cols] = side
    return unit


def boundary_assignment(rand_a, rand_b, n_boundaries: int, p_boundary: float):
    """``(is_boundary, index)`` for a batch of resetting envs.

    ``rand_a`` decides WHETHER the env is a boundary env, ``rand_b`` decides
    WHICH of the ``n_boundaries`` boundaries. The index is drawn uniformly
    over all boundaries -- see the module docstring for why this is not
    Akkaya's dimension-then-side draw.
    """
    if n_boundaries < 1:
        # clamp(0, -1) yields -1, which indexes the LAST row in torch --
        # a silently wrong boundary rather than an error.
        raise AutoDRError(f"n_boundaries must be >= 1, got {n_boundaries}")
    if not 0.0 <= p_boundary <= 1.0:
        # Out of range this silently makes every env, or no env, a boundary
        # env; NaN makes none of them and raises nothing.
        raise AutoDRError(f"p_boundary must be in [0, 1], got {p_boundary}")
    is_boundary = rand_a < p_boundary
    index = (rand_b * n_boundaries).long().clamp(0, n_boundaries - 1)
    return is_boundary, index
