# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""The TELEPORT SUCCESS IDENTITY TEST (strict order, step 2; M2.4b NEXT C)."""

from __future__ import annotations

import math
import sys

SCRIPT_MARKER = "check_seated_success-2026-09-12a"


def _squash(x_m: float, a: float, b: float) -> float:
    """``insertion_math.squash`` in stdlib floats, same overflow-free form."""
    e = math.exp(-abs(a * x_m))
    return e / (1.0 + b * e + e * e)


def kernel_sum_mm(sdf_mm: float, p: dict) -> float:
    """``insertion_math.kernel_sum`` in stdlib floats, distance in MILLIMETRES."""
    x = sdf_mm / 1000.0
    return (
        _squash(x, p["a_coarse"], p["b_coarse"])
        + _squash(x, p["a_mid"], p["b_mid"])
        + _squash(x, p["a_fine"], p["b_fine"])
    )


def expected_reward(
    sdf_mm: float, interpen_mm: float, engaged: bool, success: bool, p: dict,
    action_rate: float = 0.0, steps_remaining: float = 0.0,
) -> float:
    """The env's reward re-derived from the PRINTED measurements."""
    kernels = kernel_sum_mm(sdf_mm, p)
    task = kernels + (p["w_engaged"] if engaged else 0.0) + (p["w_success"] if success else 0.0)
    d = interpen_mm / 1000.0
    thresh = p["interpen_thresh_m"]
    scale = 0.0 if d >= thresh else 1.0 - math.tanh(d / thresh)
    step_value = task * scale
    r = step_value + p["time_penalty_per_step"] - p["action_rate_scale"] * action_rate
    if success:
        r += step_value * steps_remaining
    return r


def judge_run(
    depths_mm: list,
    in_region: list,
    engaged: list,
    rewards: list,
    sdf_mm: list,
    interpen_mm: list,
    terminated_any: bool,
    success_term_any: bool,
    force_abort_any: bool,
    solve_converged: bool,
    band_mm: tuple,
    reward_peak: float,
    reward_params: dict,
    shallow: bool,
    sdf_slack_mm: float = 0.5,
    reward_tol: float = 1e-3,
    lateral: bool = False,
    lat_y_mm: list | None = None,
    wall_y_mm: float = 0.0,
    max_depth_mm: list | None = None,
    force_n: list | None = None,
    free_air_force_tol_n: float = 1.0,
    steps_remaining: list | None = None,
    action_rate: list | None = None,
) -> tuple[bool, list]:
    """Judge one run."""
    points: list = []

    def point(name: str, ok: bool, detail: str = "") -> None:
        points.append((name, bool(ok), detail))

    rem = steps_remaining or [0.0] * len(rewards)
    rate = action_rate or [0.0] * len(rewards)

    def identity(tag: str) -> tuple:
        """``(ok, detail)`` for the reward identity over every env."""
        bad = []
        for i, r in enumerate(rewards):
            want = expected_reward(
                sdf_mm[i], interpen_mm[i], bool(engaged[i]), bool(in_region[i]),
                reward_params, float(rate[i]), float(rem[i]),
            )
            if abs(r - want) > reward_tol:
                bad.append(f"env {i}: paid {r:.6f} vs formula {want:.6f}")
        return (not bad), ("; ".join(bad) if bad else f"{tag} within {reward_tol:g}")

    lo, hi = band_mm
    point("P1 the kinematic solve converged", solve_converged,
          "an unconverged teleport is not a seated state; nothing below is judged from it"
          if not solve_converged else "")

    if lateral:
        lat = lat_y_mm or []
        inside = [y for y in lat if abs(y) <= wall_y_mm]
        point(f"L2 every env really stands OUTSIDE the pocket wall "
              f"(|y| > {wall_y_mm:.2f} mm)",
              bool(lat) and not inside,
              f"still over the pocket: {[round(y, 3) for y in inside]}"
              if inside else f"|y| = {[round(y, 2) for y in lat]}")
        shy = [d for d in depths_mm if not (lo <= d <= hi)]
        point(f"L3 the projected depth is INSIDE the {lo:.1f}-{hi:.1f} mm band "
              f"(the hack pose, or nothing is exercised)",
              not shy, f"outside: {[round(d, 3) for d in shy]}")
        point("L4 the success predicate does NOT fire beside the pocket (D-157)",
              not any(in_region), f"{sum(map(bool, in_region))}/{len(in_region)} in region")
        point("L5 the engaged bonus does NOT fire beside the pocket (D-157)",
              not any(engaged), f"{sum(map(bool, engaged))}/{len(engaged)} engaged")
        booked = [d for d in (max_depth_mm or []) if d > 0.0]
        point("L6 the depth metric books nothing beside the pocket",
              not booked, f"booked: {[round(d, 4) for d in booked]}")
        ok_id, detail = identity("lateral reward")
        point("L7 the paid reward equals the D-109 formula on the printed numbers",
              ok_id, detail)
        heavy = [f for f in (force_n or []) if abs(f) > free_air_force_tol_n]
        point(f"L8 the force in FREE AIR is under {free_air_force_tol_n:.1f} N "
              f"(the gravity tare removed the tool's own weight)",
              bool(force_n) and not heavy,
              f"still loaded: {[round(f, 3) for f in heavy]}" if heavy
              else ("no force handed in" if not force_n
                    else f"|F| = {[round(f, 3) for f in (force_n or [])]}"))
    elif not shallow:
        bad_depth = [d for d in depths_mm if not (lo <= d <= hi)]
        point(f"P2 every achieved depth is inside the {lo:.1f}-{hi:.1f} mm band",
              not bad_depth, f"outside: {[round(d, 3) for d in bad_depth]}")
        point("P3 the success predicate fires in every env",
              all(in_region), f"{sum(map(bool, in_region))}/{len(in_region)} in region")
        point("P4 the episode TERMINATES at the seat (the success exit)",
              terminated_any and success_term_any,
              f"terminated {terminated_any}, success exit {success_term_any}")
        point("P5 the exit is the SUCCESS, not the force abort",
              not force_abort_any)
        point("P6 the engaged bonus fires (depth past 30 % of the seat)",
              all(engaged), f"{sum(map(bool, engaged))}/{len(engaged)} engaged")
        over = []
        for i, s in enumerate(sdf_mm):
            bound = (hi - depths_mm[i]) + sdf_slack_mm
            if s > bound:
                over.append(f"env {i}: sdf {s:.4f} mm > bound {bound:.4f} mm")
        point(f"P7 the sdf mean-outside stays under the rigid-shift bound "
              f"(band_max - depth + {sdf_slack_mm:.1f} mm)",
              not over, "; ".join(over))
        ok_id, detail = identity("seated reward")
        point("P8 the paid reward equals the D-109 formula on the printed numbers",
              ok_id, detail)
    else:
        deep = [d for d in depths_mm if d >= lo]
        point(f"S2 every achieved depth stays below the band ({lo:.1f} mm)",
              not deep, f"at or past the band: {[round(d, 3) for d in deep]}")
        point("S3 the success predicate does NOT fire",
              not any(in_region), f"{sum(map(bool, in_region))}/{len(in_region)} in region")
        point("S4 the episode does not terminate", not terminated_any)
        ceiling = reward_peak - 1.0
        rich = [r for r in rewards if r >= ceiling]
        point(f"S5 the reward stays below {ceiling:.2f} (no success bonus outside the band)",
              not rich, f"at or above: {[round(r, 4) for r in rich]}")
        ok_id, detail = identity("shallow reward")
        point("S6 the paid reward equals the D-109 formula on the printed numbers",
              ok_id, detail)

    return all(ok for _, ok, _ in points), points


LATERAL_PROBE_MARGIN_M = 0.005


def lateral_probe_min_y_m(block_outer_y_m: float, part_bbox_y_m: float) -> float:
    """The smallest lateral offset at which an UPRIGHT part clears the block."""
    return block_outer_y_m + 0.5 * part_bbox_y_m


def lateral_probe_default_y_m(
    block_outer_y_m: float, part_bbox_x_m: float, part_bbox_y_m: float,
    margin_m: float = LATERAL_PROBE_MARGIN_M,
) -> float:
    """The probe's default offset: clear for ANY yaw, plus the probe margin."""
    return block_outer_y_m + 0.5 * math.hypot(part_bbox_x_m, part_bbox_y_m) + margin_m


def lateral_probe_clamp_m(
    lateral_y_m: float, clamp_m: float, margin_m: float = LATERAL_PROBE_MARGIN_M,
) -> float:
    """The OSC tip clamp half-width the ``--lateral`` probe needs, in metres."""
    return max(float(clamp_m), abs(float(lateral_y_m)) + float(margin_m))


def lateral_probe_y_error(
    y_mm: float, block_outer_y_m: float, part_bbox_y_m: float
) -> str | None:
    """``None`` if the requested offset is free air, else the refusal text."""
    min_mm = lateral_probe_min_y_m(block_outer_y_m, part_bbox_y_m) * 1000.0
    if abs(y_mm) >= min_mm:
        return None
    return (
        f"--lateral-y-mm {y_mm:.1f} buries the part: an upright part of "
        f"{part_bbox_y_m * 1000.0:.1f} mm long-axis extent reaches "
        f"{abs(y_mm) - part_bbox_y_m * 1000.0 / 2.0:.1f} mm, still inside the "
        f"block's outer edge at {block_outer_y_m * 1000.0:.2f} mm. The probe "
        f"needs |y| >= {min_mm:.1f} mm (free air); drop --lateral-y-mm to get "
        f"the yaw-safe default. RT-112/RT-113 measured what happens below "
        f"this bound."
    )


_GUARD_BLOCK_Y_M = 0.10045
_GUARD_PART_Y_M = 0.143500

GUARD_CASES: tuple = (
    ("the RT-112/RT-113 offset 109.0 mm is REFUSED", 109.0, False),
    ("0 mm (straight over the pocket) is REFUSED", 0.0, False),
    ("the bare clearance 172.2 mm is accepted", 172.2, True),
    ("the computed default ~191.9 mm is accepted", 191.9, True),
    ("a negative offset is judged on its magnitude", -191.9, True),
    ("a negative buried offset is REFUSED", -109.0, False),
)


LATERAL_CLAMP_CASES: tuple = (
    ("the 191.889 mm default widens the 0.08 m box to 196.889 mm",
     0.1918892059209824, 0.08, 0.1968892059209824),
    ("a negative offset asks for the same half-width as its mirror",
     -0.1918892059209824, 0.08, 0.1968892059209824),
    ("the bare clearance 172.2 mm widens the box to 177.2 mm",
     0.1722, 0.08, 0.1772),
    ("a box that already covers the offset is kept, not narrowed",
     0.010, 0.08, 0.08),
)


def _guard_self_test() -> bool:
    """Run GUARD_CASES, then the mutation that must break the rule (D-080)."""
    ok = True
    for name, y_mm, expect_ok in GUARD_CASES:
        got_ok = lateral_probe_y_error(y_mm, _GUARD_BLOCK_Y_M, _GUARD_PART_Y_M) is None
        good = got_ok is expect_ok
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")

    still_refused = [
        y for _, y, expect_ok in GUARD_CASES if not expect_ok
        and lateral_probe_y_error(y, 0.0, 0.0) is not None
    ]
    good = not still_refused
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'}  the mutation (block shrunk to a "
          f"point) lifts every refusal"
          + (f"  -- still refused: {still_refused}" if still_refused else ""))

    for name, y_m, clamp_m, expect_m in LATERAL_CLAMP_CASES:
        got_m = lateral_probe_clamp_m(y_m, clamp_m)
        good = abs(got_m - expect_m) < 1e-12
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}  "
              f"(got {got_m * 1000.0:.3f} mm, want {expect_m * 1000.0:.3f} mm)")

    widened = [
        name for name, y_m, _clamp_m, _expect_m in LATERAL_CLAMP_CASES
        if lateral_probe_clamp_m(y_m, 1.0) != 1.0
    ]
    good = not widened
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'}  MUTATION: a 1.0 m box is returned "
          f"unchanged for every case ({widened or 'none widened, as required'})")
    return ok


def curve_grid(heights_mm, laterals_mm, tilts_deg=(0.0,), laterals_x_mm=(0.0,)) -> list:
    """The measurement grid as ``(h_mm, y_mm, x_mm, tilt_deg)``."""
    return [
        (float(h), float(y), float(x), float(t))
        for t in tilts_deg for h in heights_mm
        for y in laterals_mm for x in laterals_x_mm
    ]


def curve_lateral_error(heights_mm, laterals_mm, clamp_mm=None,
                        laterals_x_mm=(0.0,)) -> str | None:
    """``None`` if the grid is measurable, else the refusal text."""
    laterals = [float(y) for y in laterals_mm]
    laterals_x = [float(x) for x in laterals_x_mm]
    heights = [float(h) for h in heights_mm]
    nonzero = [y for y in laterals if y != 0.0]
    nonzero_x = [x for x in laterals_x if x != 0.0]
    below = [h for h in heights if h < 0.0]
    if nonzero and below:
        return (
            f"--curve-lateral-y-mm {nonzero} together with a height below the "
            f"entrance plane {below}: inside the pocket the stage-2 walls hold "
            f"the part, the solver would push it into the wall and the force "
            f"abort would end the sweep there. Sweep laterals at heights >= 0 "
            f"only (RT-134's landing was at depth 0), or drop the offsets."
        )
    if nonzero_x and below:
        return (
            f"--curve-lateral-x-mm {nonzero_x} together with a height below "
            f"the entrance plane {below}: inside the pocket the stage-2 walls "
            f"hold the part across the short axis to PLAY_X / 2, the solver "
            f"would push it into the wall and the force abort would end the "
            f"sweep there. Sweep x offsets at heights >= 0 only, or drop them."
        )
    if clamp_mm is not None:
        high = [h for h in heights if abs(h) > float(clamp_mm)]
        if high:
            return (
                f"--curve-heights-mm {high} lies outside the OSC tip clamp box, "
                f"half-width osc_pos_clamp_m = {float(clamp_mm):.1f} mm around "
                f"the entrance. The box is componentwise, so a height beyond it "
                f"is pulled toward the entrance during the settle just as a "
                f"lateral is (RT-151b). Keep |h| <= {float(clamp_mm):.1f} mm."
            )
        beyond = [y for y in laterals if abs(y) > float(clamp_mm)]
        if beyond:
            return (
                f"--curve-lateral-y-mm {beyond} lies outside the OSC tip clamp "
                f"box, half-width osc_pos_clamp_m = {float(clamp_mm):.1f} mm "
                f"around the entrance. The teleport would reach it and the "
                f"settle steps would pull it back (RT-151b: 60.0 -> 52.454 mm "
                f"at residual 0.000 mm), so the reward would be read at an "
                f"uncommanded pose. Keep |y| <= {float(clamp_mm):.1f} mm, or "
                f"raise osc_pos_clamp_m and say so in the expectation."
            )
        beyond_x = [x for x in laterals_x if abs(x) > float(clamp_mm)]
        if beyond_x:
            return (
                f"--curve-lateral-x-mm {beyond_x} lies outside the OSC tip "
                f"clamp box, half-width osc_pos_clamp_m = "
                f"{float(clamp_mm):.1f} mm around the entrance. The box is "
                f"componentwise, so x is pulled back during the settle just "
                f"as y is (RT-151b: 60.0 -> 52.454 mm at residual 0.000 mm), "
                f"and the reward would be read at an uncommanded pose. Keep "
                f"|x| <= {float(clamp_mm):.1f} mm, or raise osc_pos_clamp_m "
                f"and say so in the expectation."
            )
    return None


CURVE_GRID_CASES: tuple = (
    ("the default lateral list [0.0] is accepted with the old height grid",
     [165.0, 0.0, -34.0], [0.0], True),
    ("laterals at ONE height above the plane are accepted",
     [1.0], [0.0, 1.0, 2.0, 4.88, 8.0], True),
    ("laterals at depth 0 exactly are accepted",
     [0.0], [0.0, 4.88], True),
    ("a non-zero lateral with a NEGATIVE height is REFUSED",
     [1.0, -10.0], [0.0, 4.88], False),
    ("a negative height with lateral 0 only is accepted",
     [-10.0, -34.0], [0.0], True),
)


CURVE_CLAMP_CASES: tuple = (
    ("laterals well inside the clamp box are accepted",
     [0.0, 5.0, 8.7, 45.0], 50.0, True),
    ("a lateral exactly AT the clamp half-width is accepted",
     [0.0, 50.0], 50.0, True),
    ("RT-151b's y 51.3 mm is REFUSED (it settled to 50.320 mm)",
     [0.0, 51.3], 50.0, False),
    ("RT-151b's y 60.0 mm is REFUSED (it settled to 52.454 mm)",
     [0.0, 45.0, 60.0], 50.0, False),
    ("a NEGATIVE lateral beyond the box is refused on its magnitude",
     [0.0, -60.0], 50.0, False),
    ("clamp_mm None leaves the rule off -- y 60 passes the height rule alone",
     [0.0, 60.0], None, True),
)

CURVE_CLAMP_HEIGHT_CASES: tuple = (
    ("heights inside the clamp box are accepted", [30.0, 16.0, 45.0], 50.0, True),
    ("a height exactly AT the clamp half-width is accepted", [50.0, 0.0], 50.0, True),
    ("RT-123's +165 mm stand-off is REFUSED (it is pulled down while it settles)",
     [165.0, 30.0], 50.0, False),
    ("a height BELOW the box is refused on its magnitude too", [-60.0], 50.0, False),
    ("clamp_mm None leaves the height rule off -- +165 still accepted",
     [165.0, 0.0], None, True),
)


CURVE_X_CASES: tuple = (
    ("the default x list [0.0] is accepted with the old height grid",
     [165.0, 0.0, -34.0], [0.0], True),
    ("x offsets at ONE height above the plane are accepted",
     [1.0], [0.0, 10.0, 25.0, 48.0], True),
    ("x offsets at depth 0 exactly are accepted", [0.0], [0.0, 25.0], True),
    ("a non-zero x with a NEGATIVE height is REFUSED",
     [1.0, -10.0], [0.0, 25.0], False),
    ("a NEGATIVE x with a negative height is refused too",
     [-10.0], [0.0, -25.0], False),
    ("a negative height with x 0 only is accepted", [-10.0, -34.0], [0.0], True),
)

CURVE_X_CLAMP_CASES: tuple = (
    ("RT-149's own band, |x| 48-49.9 mm, is INSIDE the box and accepted",
     [0.0, 48.0, 49.9], 50.0, True),
    ("an x exactly AT the clamp half-width is accepted", [0.0, 50.0], 50.0, True),
    ("an x past the half-width is REFUSED", [0.0, 55.0], 50.0, False),
    ("a NEGATIVE x beyond the box is refused on its magnitude",
     [0.0, -60.0], 50.0, False),
    ("clamp_mm None leaves the rule off -- x 60 passes the height rule alone",
     [0.0, 60.0], None, True),
)


def _curve_x_self_test() -> bool:
    """CURVE_X_CASES and CURVE_X_CLAMP_CASES, then both mutations (D-080)."""
    ok = True
    for name, heights, laterals_x, expect_ok in CURVE_X_CASES:
        got_ok = curve_lateral_error(heights, [0.0], None, laterals_x) is None
        good = got_ok is expect_ok
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")

    for name, laterals_x, clamp, expect_ok in CURVE_X_CLAMP_CASES:
        got_ok = curve_lateral_error([16.0, 30.0], [0.0], clamp, laterals_x) is None
        good = got_ok is expect_ok
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")

    still_h = [
        name for name, heights, laterals_x, expect_ok in CURVE_X_CASES
        if not expect_ok
        and curve_lateral_error([abs(h) for h in heights], [0.0], None,
                                laterals_x) is not None
    ]
    good = not still_h
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'}  MUTATION: heights lifted above the "
          f"plane lift every X wall refusal ({still_h or 'none left, as required'})")

    still_c = [
        name for name, laterals_x, clamp, expect_ok in CURVE_X_CLAMP_CASES
        if not expect_ok
        and curve_lateral_error([16.0, 30.0], [0.0], 1000.0, laterals_x) is not None
    ]
    good = not still_c
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'}  MUTATION: a 1000 mm clamp lifts "
          f"every X clamp refusal ({still_c or 'none left, as required'})")

    grid = curve_grid([1.0], [0.0, 4.88], [0.0], [0.0, 25.0])
    good = grid == [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 25.0, 0.0),
                    (1.0, 4.88, 0.0, 0.0), (1.0, 4.88, 25.0, 0.0)]
    outer = [(float(h), float(y), float(x), 0.0)
             for h in [1.0] for x in [0.0, 25.0] for y in [0.0, 4.88]]
    good = good and outer != grid
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'}  curve_grid puts x INNERMOST; the "
          f"x-outside-y mutation changes the order")
    return ok


def _curve_clamp_self_test() -> bool:
    """CURVE_CLAMP_CASES, then the mutation (D-080)."""
    ok = True
    for name, laterals, clamp, expect_ok in CURVE_CLAMP_CASES:
        got_ok = curve_lateral_error([16.0, 30.0], laterals, clamp) is None
        good = got_ok is expect_ok
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")

    for name, heights, clamp, expect_ok in CURVE_CLAMP_HEIGHT_CASES:
        got_ok = curve_lateral_error(heights, [0.0], clamp) is None
        good = got_ok is expect_ok
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")

    still_h = [
        name for name, heights, clamp, expect_ok in CURVE_CLAMP_HEIGHT_CASES
        if not expect_ok
        and curve_lateral_error(heights, [0.0], 1000.0) is not None
    ]
    good = not still_h
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'}  MUTATION: a 1000 mm clamp lifts "
          f"every HEIGHT refusal ({still_h or 'none left, as required'})")

    still_refused = [
        name for name, laterals, clamp, expect_ok in CURVE_CLAMP_CASES
        if not expect_ok
        and curve_lateral_error([16.0, 30.0], laterals, 1000.0) is not None
    ]
    good = not still_refused
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'}  MUTATION: a 1000 mm clamp lifts "
          f"every clamp refusal ({still_refused or 'none left, as required'})")
    return ok


def _curve_grid_self_test() -> bool:
    """CURVE_GRID_CASES, the grid order, then the mutation (D-080)."""
    ok = True
    for name, heights, laterals, expect_ok in CURVE_GRID_CASES:
        got_ok = curve_lateral_error(heights, laterals) is None
        good = got_ok is expect_ok
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")

    grid = curve_grid([1.0, 0.0], [0.0, 4.88])
    good = grid == [(1.0, 0.0, 0.0, 0.0), (1.0, 4.88, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0), (0.0, 4.88, 0.0, 0.0)]
    old = curve_grid([165.0, 0.0, -34.0], [0.0])
    good = good and old == [(165.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0),
                            (-34.0, 0.0, 0.0, 0.0)]
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'}  curve_grid is heights x laterals, "
          f"heights outer; [0.0] reproduces the height-only rows")

    ladder = curve_grid([0.0, -10.0], [0.0], [0.0, 10.0])
    good = ladder == [(0.0, 0.0, 0.0, 0.0), (-10.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 0.0, 10.0), (-10.0, 0.0, 0.0, 10.0)]
    inner = [(float(h), float(y), 0.0, float(t))
             for h in [0.0, -10.0] for t in [0.0, 10.0] for y in [0.0]]
    good = good and inner != ladder
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'}  curve_grid puts the tilt OUTERMOST "
          f"(one descent per tilt); the tilt-inside mutation changes the order")

    still_refused = [
        name for name, heights, laterals, expect_ok in CURVE_GRID_CASES
        if not expect_ok
        and curve_lateral_error([abs(h) for h in heights], laterals) is not None
    ]
    good = not still_refused
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'}  the mutation (heights lifted above "
          f"the plane) lifts every refusal"
          + (f"  -- still refused: {still_refused}" if still_refused else ""))
    return ok


_BAND = (33.0, 36.0)
_PEAK = 3.0
_PARAMS = dict(
    a_coarse=17.2809, b_coarse=2.0,
    a_mid=997.7411, b_mid=2.0,
    a_fine=5094.3202, b_fine=0.0,
    w_engaged=1.0, w_success=1.0,
    interpen_thresh_m=0.0002938,
    time_penalty_per_step=-1.0 / 256.0,
    action_rate_scale=0.1,
)

_FIXTURE_REMAINING = 224.0


def _paid(sdf: list, interpen: list, engaged: bool, success: bool,
          steps_remaining: float = 0.0) -> list:
    """The reward a CORRECT env would pay for these measurements."""
    return [expected_reward(sdf[i], interpen[i], engaged, success, _PARAMS,
                            0.0, steps_remaining)
            for i in range(len(sdf))]


_SEATED_SDF = [2.0, 2.1]
_SEATED_INT = [0.01, 0.02]
_SHALLOW_SDF = [21.0, 20.9]
_SHALLOW_INT = [0.005, 0.006]
_SEATED = dict(
    depths_mm=[34.0, 33.9], in_region=[True, True], engaged=[True, True],
    sdf_mm=list(_SEATED_SDF), interpen_mm=list(_SEATED_INT),
    rewards=_paid(_SEATED_SDF, _SEATED_INT, True, True, _FIXTURE_REMAINING),
    terminated_any=True, success_term_any=True, force_abort_any=False,
    steps_remaining=[_FIXTURE_REMAINING, _FIXTURE_REMAINING],
    solve_converged=True, band_mm=_BAND, reward_peak=_PEAK,
    reward_params=_PARAMS, shallow=False,
)
_SHALLOW = dict(
    depths_mm=[15.0, 15.1], in_region=[False, False], engaged=[True, True],
    sdf_mm=list(_SHALLOW_SDF), interpen_mm=list(_SHALLOW_INT),
    rewards=_paid(_SHALLOW_SDF, _SHALLOW_INT, True, False),
    terminated_any=False, success_term_any=False, force_abort_any=False,
    solve_converged=True, band_mm=_BAND, reward_peak=_PEAK,
    reward_params=_PARAMS, shallow=True,
)
_WALL_Y_MM = 72.55
_LAT_SDF = [110.0, 110.1]
_LAT_INT = [0.0, 0.0]
_LATERAL = dict(
    depths_mm=[34.0, 33.9], in_region=[False, False], engaged=[False, False],
    sdf_mm=list(_LAT_SDF), interpen_mm=list(_LAT_INT),
    rewards=_paid(_LAT_SDF, _LAT_INT, False, False),
    terminated_any=False, success_term_any=False, force_abort_any=False,
    solve_converged=True, band_mm=_BAND, reward_peak=_PEAK,
    reward_params=_PARAMS, shallow=False, lateral=True,
    lat_y_mm=[109.0, 109.0], wall_y_mm=_WALL_Y_MM, max_depth_mm=[0.0, 0.0],
    force_n=[0.02, 0.03],
)

JUDGE_CASES: tuple = (
    ("a clean seated run passes", _SEATED, True),
    ("a clean shallow control passes", _SHALLOW, True),
    ("a seat that does NOT terminate FAILS (the success exit is missing)",
     {**_SEATED, "terminated_any": False, "success_term_any": False}, False),
    ("a seat that ends on the FORCE ABORT instead FAILS",
     {**_SEATED, "force_abort_any": True}, False),
    ("a termination that is neither exit FAILS instead of passing for free",
     {**_SEATED, "success_term_any": False}, False),
    ("a predicate that does not fire at the seat FAILS",
     {**_SEATED, "in_region": [True, False]}, False),
    ("an achieved depth outside the band FAILS",
     {**_SEATED, "depths_mm": [34.0, 36.4]}, False),
    ("an unconverged solve FAILS instead of being read as contact",
     {**_SEATED, "solve_converged": False}, False),
    ("a shallow run whose predicate fires FAILS (the paid counter-proof works)",
     {**_SHALLOW, "in_region": [False, True]}, False),
    ("a shallow run paying the success bonus FAILS",
     {**_SHALLOW, "rewards": [1.05, 2.4]}, False),
    ("RT-102's own defect FAILS: a seated pose 25.76 mm off the goal",
     {**_SEATED, "sdf_mm": [25.76, 25.76],
      "rewards": _paid([25.76, 25.76], _SEATED_INT, True, True, _FIXTURE_REMAINING)}, False),
    ("an sdf just past the bound FAILS (33.9 mm deep -> bound 2.6 mm)",
     {**_SEATED, "sdf_mm": [2.0, 2.61],
      "rewards": _paid([2.0, 2.61], _SEATED_INT, True, True, _FIXTURE_REMAINING)}, False),
    ("an sdf just inside the bound PASSES (the bound is not a window)",
     {**_SEATED, "sdf_mm": [2.0, 2.55],
      "rewards": _paid([2.0, 2.55], _SEATED_INT, True, True, _FIXTURE_REMAINING)}, True),
    ("a MICROMETRE sdf still passes (a fuller seat is not punished)",
     {**_SEATED, "sdf_mm": [0.0013, 0.0011],
      "rewards": _paid([0.0013, 0.0011], _SEATED_INT, True, True, _FIXTURE_REMAINING)}, True),
    ("a payout counted one step too long FAILS (P8)",
     {**_SEATED, "steps_remaining": [_FIXTURE_REMAINING, _FIXTURE_REMAINING - 1.0]}, False),
    ("a seated run paid WITHOUT the payout FAILS (P8)",
     {**_SEATED, "rewards": _paid(_SEATED_SDF, _SEATED_INT, True, True, 0.0)}, False),
    ("a reward off the D-109 formula by 0.01 FAILS (P8)",
     {**_SEATED, "rewards": [_SEATED["rewards"][0], _SEATED["rewards"][1] + 0.01]}, False),
    ("a reward missing the engaged bonus FAILS (P8 catches it without a window)",
     {**_SEATED, "rewards": [_SEATED["rewards"][0], _SEATED["rewards"][1] - 1.0]}, False),
    ("a reward past the peak FAILS (a bonus paid twice)",
     {**_SEATED, "rewards": [_SEATED["rewards"][0], 3.9]}, False),
    ("an UNSCALED reward at heavy interpenetration FAILS (SAPU not applied)",
     {**_SEATED, "interpen_mm": [0.01, 0.20]}, False),
    ("interpenetration PAST the SAPU threshold pays only the time penalty",
     {**_SEATED, "interpen_mm": [0.01, 0.30],
      "rewards": [_SEATED["rewards"][0], _PARAMS["time_penalty_per_step"]]}, True),
    ("a shallow reward off the formula FAILS (S6)",
     {**_SHALLOW, "rewards": [_SHALLOW["rewards"][0], _SHALLOW["rewards"][1] - 0.02]}, False),
    ("a clean lateral counter-proof passes", _LATERAL, True),
    ("THE RT-107 HACK FAILS: the success predicate fires beside the pocket",
     {**_LATERAL, "in_region": [False, True],
      "rewards": _paid(_LAT_SDF, _LAT_INT, False, False)}, False),
    ("the RT-107 hack FAILS on the engaged bonus too",
     {**_LATERAL, "engaged": [True, True],
      "rewards": _paid(_LAT_SDF, _LAT_INT, True, False)}, False),
    ("a lateral run booking depth beside the pocket FAILS (the metric's gate)",
     {**_LATERAL, "max_depth_mm": [0.0, 33.9]}, False),
    ("a lateral run still OVER the pocket FAILS instead of passing for free",
     {**_LATERAL, "lat_y_mm": [109.0, 40.0]}, False),
    ("a lateral run whose projected depth missed the band FAILS",
     {**_LATERAL, "depths_mm": [34.0, 12.0]}, False),
    ("the PRE-TARE force 8.08 N in free air FAILS (L8: the tool's own weight)",
     {**_LATERAL, "force_n": [8.08, 8.08]}, False),
    ("a DOUBLED tare (wrong sign) FAILS too (L8 reads 16.17 N)",
     {**_LATERAL, "force_n": [16.17, 16.17]}, False),
    ("a lateral run with no force handed in FAILS instead of passing for free",
     {**_LATERAL, "force_n": []}, False),
    ("a lateral reward off the formula FAILS (L7)",
     {**_LATERAL, "rewards": [_LATERAL["rewards"][0], _LATERAL["rewards"][1] + 0.02]}, False),
)


def _cross_check_against_insertion_math() -> bool:
    """Pin ``expected_reward`` to the REAL `compute_rewards_insertion`."""
    import importlib.util
    import pathlib

    here = pathlib.Path(__file__).resolve()
    shim_path = here.parent / "tools" / "torch_shim.py"
    spec = importlib.util.spec_from_file_location("torch_shim", shim_path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    torch, _ = shim.load()
    sys.modules.setdefault("torch", torch)

    math_src = (
        here.parents[1] / "source" / "insertion" / "insertion" / "tasks"
        / "direct" / "insertion" / "insertion_math.py"
    )
    spec = importlib.util.spec_from_file_location("insertion_math_offline", math_src)
    im = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(im)

    grid = [
        (0.0, 0.0), (0.0013, 0.0), (2.0, 0.01), (2.5, 0.05),
        (21.0, 0.005), (25.76, 0.02), (100.0, 0.0),
        (2.0, 0.2937), (2.0, 0.2938), (2.0, 0.30), (2.0, 1.0),
    ]
    flags = ((True, True), (True, False), (False, False))
    extras = ((0.0, 0.0), (0.0, 224.0), (0.4, 100.0), (1.7, 0.0))
    p = _PARAMS
    worst = 0.0
    n = 0
    for sdf, ipen in grid:
        for engaged, success in flags:
            for rate, remaining in extras:
                n += 1
                mine = expected_reward(sdf, ipen, engaged, success, p, rate, remaining)
                theirs = float(im.compute_rewards_insertion(
                    torch.tensor([sdf / 1000.0]), torch.tensor([ipen / 1000.0]),
                    torch.tensor([engaged]) > 0.5, torch.tensor([success]) > 0.5,
                    torch.tensor([False]) > 0.5,
                    torch.tensor([rate]), torch.tensor([remaining]),
                    torch.tensor([0.0]),
                    torch.tensor([0.0]), torch.tensor([0.0]),
                    p["a_coarse"], p["b_coarse"], p["a_mid"], p["b_mid"],
                    p["a_fine"], p["b_fine"], p["w_engaged"], p["w_success"],
                    0.0,
                    p["interpen_thresh_m"], -1.0,
                    p["time_penalty_per_step"], p["action_rate_scale"],
                    0.0,
                ).tolist()[0])
                worst = max(worst, abs(mine - theirs))
    good = worst < 1e-12
    print(f"  {'PASS' if good else 'FAIL'}  expected_reward == insertion_math."
          f"compute_rewards_insertion over {n} points "
          f"(worst |diff| {worst:.2e})")
    return good


def _curve_decay_self_test() -> bool:
    """The reward-curve mode's OWN claim, pinned offline (D-080)."""
    p = dict(_PARAMS)
    at_seat = kernel_sum_mm(0.0, p)
    at_home = kernel_sum_mm(165.0, p)
    at_margin = kernel_sum_mm(173.21, p)
    shipped_ok = (
        (abs(at_seat - 1.0) < 1e-9)
        and (at_home > 0.01)
        and (abs(at_margin - 0.045456) < 1e-4)
    )
    narrow = dict(p, a_coarse=math.acosh(10.0) / 0.010)
    at_home_narrow = kernel_sum_mm(165.0, narrow)
    good = shipped_ok and at_home_narrow < 1e-15
    print(f"  {'PASS' if good else 'FAIL'}  kernel decay: seat {at_seat:.4f}, "
          f"165 mm {at_home:.4f}, at the 173.21 mm margin {at_margin:.4f}; "
          f"with the coarse margin narrowed back to the superseded 10 mm the "
          f"165 mm value is {at_home_narrow:.3e}")
    return good


def _self_test() -> int:
    ok = _cross_check_against_insertion_math()
    ok = _curve_decay_self_test() and ok
    ok = _guard_self_test() and ok
    ok = _curve_grid_self_test() and ok
    ok = _curve_clamp_self_test() and ok
    ok = _curve_x_self_test() and ok
    for name, kwargs, expected in JUDGE_CASES:
        got, points = judge_run(**kwargs)
        good = got is expected
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")
        if not good:
            for pname, pok, detail in points:
                print(f"        {'ok ' if pok else 'BAD'} {pname}  {detail}")
    n = (1
         + 1
         + len(GUARD_CASES) + 1
         + len(LATERAL_CLAMP_CASES) + 1
         + len(CURVE_GRID_CASES) + 3
         + len(CURVE_CLAMP_CASES)
         + len(CURVE_CLAMP_HEIGHT_CASES) + 2
         + len(CURVE_X_CASES)
         + len(CURVE_X_CLAMP_CASES) + 3
         + len(JUDGE_CASES))
    print(f"[check_seated_success] self-test: {n}/{n} checks passed ({SCRIPT_MARKER})"
          if ok else "[check_seated_success] self-test: FAIL")
    return 0 if ok else 1


if "--self-test" in sys.argv:
    raise SystemExit(_self_test())


"""Launch Isaac Sim Simulator first."""

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import pathlib  # noqa: E402
import traceback  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

_TOOLS = pathlib.Path(__file__).resolve().parent / "tools"
_spec = importlib.util.spec_from_file_location("isaac_exit", _TOOLS / "isaac_exit.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["isaac_exit"] = _mod
_spec.loader.exec_module(_mod)
exit_with = _mod.exit_with

parser = argparse.ArgumentParser(
    description="Teleport success identity test: seat the part, ask the REAL reward and termination."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--out", type=str, default="check_seated_success_metrics.json",
    help="Metrics file. Numbers are read from files, never from prose.",
)
parser.add_argument(
    "--shallow", action="store_true", default=False,
    help=(
        "The PAID COUNTER-PROOF (D-080): teleport to 15 mm instead of the "
        "seat. The predicate must NOT fire and the reward must stay low; a "
        "green seated run without this run proves nothing."
    ),
)
parser.add_argument(
    "--lateral", action="store_true", default=False,
    help=(
        "THE SECOND PAID COUNTER-PROOF (D-080, D-157): teleport to the SEATED "
        "depth but beside the pocket, so the axis projection reads a perfect "
        "seat. Both paying predicates must stay silent and the depth metric "
        "must book nothing. It reproduces the MECHANISM of RT-107's hack, not "
        "its pose -- RT-107's own offset is unreachable for an upright "
        "teleport (see --lateral-y-mm)."
    ),
)
parser.add_argument(
    "--reward-curve", action="store_true", default=False,
    help=(
        "MEASUREMENT MODE, not an identity test: teleport the part to a "
        "series of heights above the pocket entrance and print, per height, "
        "the SDF distance, the kernel term alone, both bonus predicates and "
        "the reward the env actually pays. It answers ONE question -- how far "
        "from the seat does the D-109 shaping signal still have a magnitude a "
        "policy could follow? Nothing here judges the task; the verdict is "
        "made by /rt-check against an expectation written BEFORE the run."
    ),
)
probe = parser.add_argument_group("probe (measurement settings, not task decisions)")
probe.add_argument(
    "--lateral-y-mm", type=float, default=None,
    help=(
        "Lateral y offset for --lateral, in mm. DEFAULT: computed in main() "
        "from the geometry constants (`lateral_probe_default_y_m`, ~191.9 mm "
        "today) -- free air for any yaw. It is computed and not typed here "
        "because argparse runs BEFORE AppLauncher and must not import "
        "`insertion_tasks_cfg`. An explicit value below the bare clearance "
        "(`lateral_probe_min_y_m`, ~172.2 mm) is REFUSED: there the upright "
        "part overlaps block material and the solver, not the predicate, "
        "decides the run. That is what RT-112/RT-113 measured at 109.0 mm."
    ),
)
probe.add_argument(
    "--depth-mm", type=float, default=34.0,
    help=(
        "Commanded seated depth. Default 34.0: inside the 33.0-36.0 mm band "
        "with margin on both edges, and above RT-81's deepest verified rung "
        "(33 mm at 8.08 N, which was the UNTARED reading -- the tool's own "
        "weight, with no measurable contact on top of it). --shallow "
        "overrides this to 15.0."
    ),
)
probe.add_argument(
    "--curve-heights-mm", type=float, nargs="+",
    default=[165.0, 100.0, 50.0, 20.0, 10.0, 3.0, 0.0, -10.0, -20.0, -34.0],
    help=(
        "Heights for --reward-curve, in mm ABOVE the pocket entrance; a "
        "NEGATIVE value is a depth INSIDE the pocket. The default sweeps the "
        "reset stand-off (165 mm, WORKCELL_HOME_TIP_ABOVE_ENTRANCE) down to "
        "the seat (-34 mm, the --depth-mm default). The list is a measurement "
        "grid, not a task constant."
    ),
)
probe.add_argument(
    "--curve-lateral-y-mm", type=float, nargs="+", default=[0.0],
    help=(
        "Pocket-frame y offsets for --reward-curve, in mm, applied at EVERY "
        "height (grid = heights x laterals, heights outer). Default [0.0] is "
        "the pre-2026-09-02 height-only curve, row for row. Added after "
        "RT-134 landed the part 4.88 mm beside the stage-2 axis at depth 0: "
        "the SDF response to a LATERAL offset had never been measured. A "
        "non-zero offset together with a NEGATIVE height is REFUSED "
        "(`curve_lateral_error`): inside the pocket the walls hold the part "
        "and the force abort would end the sweep. A measurement grid, not a "
        "task constant."
    ),
)
probe.add_argument(
    "--curve-lateral-x-mm", type=float, nargs="+", default=[0.0],
    help=(
        "Pocket-frame x offsets for --reward-curve, in mm, applied at EVERY "
        "(tilt, height, y) point as the INNERMOST loop. Default [0.0] is the "
        "pre-2026-09-05 curve, row for row -- x used to be pinned to zero "
        "inside the solve and no flag could move it. WHY: RT-149 measured 5 "
        "of 8 envs at |x| 48 to 49.9 mm, hard against the +-50 mm OSC tip "
        "clamp, and the SDF response along x had never been swept. Same two "
        "refusals as the y list (`curve_lateral_error`): no non-zero x below "
        "the entrance plane (the stage-2 walls hold the part across the short "
        "axis to PLAY_X / 2), and no |x| past the OSC clamp half-width."
    ),
)
probe.add_argument(
    "--curve-tilt-deg", type=float, nargs="+", default=[0.0],
    help=(
        "THE TILT LADDER (2026-09-03, after RT-141p): tilt angles in DEGREES "
        "for --reward-curve, applied as the OUTERMOST loop -- one full "
        "height descent per tilt. Signed about the pocket axis named by "
        "--curve-tilt-axis (`scripted_policy.signed_tilt_angle`). Default "
        "[0.0] is the upright curve, row for row. WHY: RT-141's policy rests "
        "the part CROOKED on the fixture rim in every env and never aligns "
        "or pushes; whether the reward even rises from that pose to the seat "
        "has never been measured. A tilt together with a NEGATIVE height is "
        "ALLOWED -- the crooked part in the mouth IS the pose in question -- "
        "and where it jams, the force abort ends THAT tilt's descent only. A "
        "measurement grid, not a task constant."
    ),
)
probe.add_argument(
    "--curve-tilt-axis", choices=("x", "y"), default="y",
    help=(
        "Pocket-frame axis the --curve-tilt-deg rotation is about "
        "(`scripted_policy.TILT_AXES`): x is the SHORT pocket axis, so the "
        "part leans along its LONG side; y is the LONG axis, so the part "
        "leans ACROSS its width -- the lean the RT-141p screenshot shows and "
        "the one with the 0.5876 mm play (D-106: across-tilt bound 0.46 deg, "
        "long-tilt bound ~2.5 deg at full depth). Default y."
    ),
)
probe.add_argument(
    "--solve-tol-tilt-deg", type=float, default=0.05,
    help="Kinematic convergence tolerance on the commanded tilt; judges the IK only.",
)
probe.add_argument(
    "--curve-settle-steps", type=int, default=4,
    help=(
        "Zero-action steps per curve point. Small on purpose: every point "
        "spends steps out of the SAME 256-step episode, and a truncation "
        "mid-curve would auto-reset the arm back to home and silently measure "
        "the wrong pose from then on. 10 points x 4 steps = 40 << 256."
    ),
)
probe.add_argument(
    "--curve-coarse-margin-mm", type=float, default=None,
    help=(
        "THE PAID COUNTER-PROOF for --reward-curve (D-080). Overrides the "
        "COARSE kernel width on the env config only, before the env is built: "
        "a = acosh(10)/margin. TURNED AROUND 2026-09-01: the shipped margin is "
        "the measured travel reach (173.21 mm), so the counter-proof is now "
        "NARROWING it back to the superseded 10, which MUST collapse the value "
        "at the 165 mm reset stand-off into the ~1e-22 range. If that does not "
        "happen, the curve is not measuring what it claims to measure. A "
        "measurement setting, never a task decision -- "
        "KERNEL_MARGIN_COARSE keeps its home in insertion_tasks_cfg.py."
    ),
)
probe.add_argument("--solve-steps", type=int, default=120,
                   help="Kinematic IK teleport iterations (seat_probe's mechanism, RT-81/82).")
probe.add_argument("--settle-steps", type=int, default=30,
                   help="Zero-action control steps before anything is read (reset-step trap).")
probe.add_argument("--ik-lambda", type=float, default=0.05, help="dls damping.")
probe.add_argument("--solve-tol-mm", type=float, default=0.05,
                   help="Kinematic convergence tolerance; judges the IK only.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

try:
    import json  # noqa: E402

    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402

    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
    from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

    import isaaclab_tasks  # noqa: F401, E402
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

    import insertion.tasks  # noqa: F401, E402
    from insertion.tasks.direct.insertion import (  # noqa: E402
        insertion_math,
        insertion_tasks_cfg,
        scripted_policy as sp,
    )
except BaseException:
    traceback.print_exc()
    exit_with(simulation_app, 1, tag="check_seated_success-import")


def _slice(obs: torch.Tensor, name: str) -> torch.Tensor:
    lo, hi = insertion_math.OBS_SLICES[name]
    return obs[:, lo:hi]


def _run_reward_curve(*, env, unwrapped, solve_to, zero_action, reward_params,
                      curve_margin_mm, env_cfg, num_envs, tilt_baseline_rad) -> int:
    """Measure the D-109 reward against distance from the seat."""
    heights_mm = [float(h) for h in args_cli.curve_heights_mm]
    laterals_mm = [float(y) for y in args_cli.curve_lateral_y_mm]
    laterals_x_mm = [float(x) for x in args_cli.curve_lateral_x_mm]
    tilts_deg = [float(t) for t in args_cli.curve_tilt_deg]
    tilt_axis = sp.TILT_AXES[args_cli.curve_tilt_axis]
    grid = curve_grid(heights_mm, laterals_mm, tilts_deg, laterals_x_mm)
    settle = int(args_cli.curve_settle_steps)
    term_names = list(insertion_math.REWARD_TERMS)
    wall_x_m = float(insertion_tasks_cfg.POCKET_WALL_X)
    wall_y_m = float(insertion_tasks_cfg.POCKET_WALL_Y)
    print("[check_seated_success] ---- reward curve (MEASUREMENT, not a verdict) ----")
    print("[check_seated_success] curve: the `reward` column carries the whole "
          "per-step reward -- kernels + bonuses (SAPU-scaled), the -1/T time "
          "penalty, the action-rate term (zero here) and, inside the success "
          "band, the payout. Read the decay off `kernel_sum`.")
    print(f"[check_seated_success] curve: {len(heights_mm)} heights x "
          f"{len(laterals_mm)} laterals = {len(grid)} points x {settle} settle "
          f"steps = {len(grid) * settle} of the {int(unwrapped.max_episode_length)} "
          f"step episode")
    if laterals_mm != [0.0]:
        print(f"[check_seated_success] curve: LATERAL sweep, y = {laterals_mm} mm "
              f"in the pocket frame (RT-134 landed at p50 4.88 mm)")
    if tilts_deg != [0.0]:
        _base_deg = math.degrees(tilt_baseline_rad[args_cli.curve_tilt_axis])
        print(f"[check_seated_success] curve: TILT LADDER, tilt = {tilts_deg} deg "
              f"ON TOP OF the {_base_deg:+.3f} deg reset baseline, about pocket "
              f"{args_cli.curve_tilt_axis} (signed_tilt_angle axis {tilt_axis}); "
              f"one descent per tilt, a jam ends that descent only")
    print(f"[check_seated_success] curve: reward terms per step, in order: "
          f"{term_names}")
    if curve_margin_mm is not None:
        print(f"[check_seated_success] curve: COUNTER-PROOF -- coarse margin forced to "
              f"{curve_margin_mm:.1f} mm, a_coarse = {float(env_cfg.kernel_a_coarse):.2f} /m "
              f"(shipped value NOT in use)")

    rows = []
    all_converged = True
    all_poses_held = True
    reset_happened = False
    end_reason = None
    end_by_tilt: dict = {}
    skip_tilt = None
    for h_mm, y_mm, x_mm, t_deg in grid:
        if skip_tilt is not None and t_deg == skip_tilt:
            continue
        skip_tilt = None
        residual_mm, converged, tilt_residual_deg = solve_to(
            h_mm / 1000.0, y_mm / 1000.0,
            tilt_baseline_rad[args_cli.curve_tilt_axis] + math.radians(t_deg),
            x_mm / 1000.0,
        )
        all_converged = all_converged and converged
        rew = None
        terms_step = {name: [0.0] * num_envs for name in term_names}
        point_reset = False
        for _ in range(settle):
            with torch.no_grad():
                sums_before = {name: unwrapped._episode_sums[name].clone()
                               for name in term_names}
                obs_dict, rew, terminated, truncated, _ = env.step(zero_action)
                ended = bool(terminated.any()) or bool(truncated.any())
                if not ended:
                    terms_step = {
                        name: [float(v) for v in
                               (unwrapped._episode_sums[name] - sums_before[name]).tolist()]
                        for name in term_names
                    }
            if ended:
                reset_happened = True
                point_reset = True
                if bool(unwrapped._last_success.any()) and not bool(
                    unwrapped._last_force_abort.any()
                ):
                    reason = "success_termination"
                elif bool(unwrapped._last_force_abort.any()):
                    reason = "force_abort"
                else:
                    reason = "timeout"
                end_by_tilt[t_deg] = reason
                if t_deg == 0.0:
                    end_reason = reason
                print(f"[check_seated_success] curve: RESET during the "
                      f"h {h_mm:.1f} mm / y {y_mm:.2f} mm / x {x_mm:.2f} mm / "
                      f"tilt {t_deg:.1f} deg point "
                      f"({reason}; terminated {bool(terminated.any())}, truncated "
                      f"{bool(truncated.any())}) -- this tilt's descent ends here, "
                      f"its later heights would read the home pose.")
                skip_tilt = t_deg
                break
        with torch.no_grad():
            obs = obs_dict["policy"]
            tip_rel = _slice(obs, "tip_rel")
            ee_quat_w = _slice(obs, "ee_quat")
            pocket_quat_now = _slice(obs, "pocket_quat")
            tilt_got_deg = [math.degrees(float(v)) for v in
                            sp.signed_tilt_angle(ee_quat_w, pocket_quat_now, tilt_axis).tolist()]
            tilt_abs_deg = [math.degrees(float(v)) for v in
                            sp.axis_tilt_angle(ee_quat_w, pocket_quat_now).tolist()]
            sdf_mm = [float(v) * 1000.0 for v in unwrapped._last_sdf_dist.tolist()]
            interpen_mm = [float(v) * 1000.0 for v in unwrapped._last_interpen_max.tolist()]
            engaged = [bool(v) for v in unwrapped._last_engaged.tolist()]
            in_region = [bool(v) for v in unwrapped._last_in_region.tolist()]
            reward_list = [float(v) for v in rew.tolist()]
            depth_mm = [float(v) * -1000.0 for v in tip_rel[:, 2].tolist()]
            lat_y_mm = [float(v) * 1000.0 for v in tip_rel[:, 1].tolist()]
            lat_x_mm = [float(v) * 1000.0 for v in tip_rel[:, 0].tolist()]
            max_depth_mm = [float(v) * 1000.0 for v in unwrapped._max_depth.tolist()]
            in_pocket = [bool(v) for v in insertion_math.in_pocket_cross_section(
                tip_rel, wall_x_m, wall_y_m).tolist()]
            force_n = [float(v) for v in
                       insertion_math.force_magnitude(_slice(obs, "force")).tolist()]
        kern = [kernel_sum_mm(v, reward_params) for v in sdf_mm]
        pose_drift_mm = max(
            max(abs(v - y_mm) for v in lat_y_mm),
            max(abs(v - x_mm) for v in lat_x_mm),
            max(abs(v + h_mm) for v in depth_mm),
        )
        pose_held = pose_drift_mm <= float(args_cli.solve_tol_mm)
        all_poses_held = all_poses_held and pose_held
        print(f"[check_seated_success] curve  h {h_mm:+8.1f} mm | y {y_mm:+6.2f} mm "
              f"(got {lat_y_mm[0]:+7.3f}) | x {x_mm:+6.2f} mm "
              f"(got {lat_x_mm[0]:+7.3f}) | tilt {t_deg:+5.1f} deg (got "
              f"{tilt_got_deg[0]:+6.2f}, |axis| {tilt_abs_deg[0]:5.2f}) | depth "
              f"{depth_mm[0]:+9.3f} mm | sdf {sdf_mm[0]:10.4f} mm | interpen "
              f"{interpen_mm[0]:7.3f} mm | kernel_sum "
              f"{kern[0]:.6e} | reward {reward_list[0]:.6e} | engaged {engaged[0]!s:5} | "
              f"success {in_region[0]!s:5} | in_pocket {in_pocket[0]!s:5} | force "
              f"{force_n[0]:7.2f} N | residual "
              f"{residual_mm:.3f} mm / {tilt_residual_deg:.3f} deg | drift "
              f"{pose_drift_mm:.3f} mm"
              f"{'' if converged else '  NOT CONVERGED'}"
              f"{'' if pose_held else '  POSE NOT HELD -- the settle moved it, this row is not the commanded pose'}"
              f"{'  RESET ON THIS POINT (obs columns show the post-reset home pose)' if point_reset else ''}")
        print("[check_seated_success] terms  "
              + " | ".join(f"{name} {terms_step[name][0]:+.5f}" for name in term_names))
        rows.append({
            "height_above_entrance_mm": h_mm,
            "lateral_y_mm": y_mm,
            "lateral_x_mm": x_mm,
            "tilt_deg": t_deg,
            "tilt_axis": args_cli.curve_tilt_axis,
            "achieved_tilt_deg": tilt_got_deg,
            "achieved_axis_tilt_deg": tilt_abs_deg,
            "achieved_lateral_y_mm": lat_y_mm,
            "achieved_lateral_x_mm": lat_x_mm,
            "pose_drift_mm": pose_drift_mm,
            "pose_held": pose_held,
            "solve_residual_mm": residual_mm,
            "solve_residual_tilt_deg": tilt_residual_deg,
            "solve_converged": converged,
            "achieved_depth_mm": depth_mm,
            "max_depth_mm": max_depth_mm,
            "in_pocket": in_pocket,
            "sdf_mean_outside_mm": sdf_mm,
            "interpen_max_mm": interpen_mm,
            "kernel_sum": kern,
            "reward": reward_list,
            "reward_terms_per_step": terms_step,
            "engaged": engaged,
            "in_success_region": in_region,
            "force_n": force_n,
            "reset_on_this_point": point_reset,
        })

    ok = (all_converged and all_poses_held
          and (not reset_happened or end_reason in (None, "success_termination")))
    metrics = {
        "run": "check_seated_success (reward curve MEASUREMENT)",
        "marker": SCRIPT_MARKER,
        "task": args_cli.task,
        "num_envs": num_envs,
        "mode": "reward_curve",
        "fixture_pos_noise_xy_m": float(env_cfg.fixture_pos_noise_xy),
        "obs_noise_pocket_pos_std_m": float(env_cfg.obs_noise_pocket_pos_std_m),
        "force_obs_noise_std_n": float(env_cfg.force_obs_noise_std_n),
        "grasp_obs_offset_x_m": float(env_cfg.grasp_obs_offset_x_m),
        "curve_settle_steps": settle,
        "curve_heights_mm": heights_mm,
        "curve_lateral_y_mm": laterals_mm,
        "curve_lateral_x_mm": laterals_x_mm,
        "curve_tilt_deg": tilts_deg,
        "curve_tilt_axis": args_cli.curve_tilt_axis,
        "reward_terms": term_names,
        "sweep_end_by_tilt": {str(k): v for k, v in end_by_tilt.items()},
        "curve_coarse_margin_mm": curve_margin_mm,
        "reward_params": reward_params,
        "home_tip_above_entrance_mm":
            float(insertion_tasks_cfg.WORKCELL_HOME_TIP_ABOVE_ENTRANCE) * 1000.0,
        "kernel_margin_coarse_mm":
            float(insertion_tasks_cfg.KERNEL_MARGIN_COARSE) * 1000.0,
        "curve": rows,
        "reward_column_includes": (
            "kernel_sum + engaged + success bonuses, SAPU-scaled; the -1/T time "
            "penalty; the action-rate penalty (zero under zero actions); and at "
            "a point inside the success band the equivalent-return payout"
        ),
        "time_penalty_per_step": float(env_cfg.time_penalty_per_step),
        "action_rate_scale": float(env_cfg.action_rate_scale),
        "all_solves_converged": all_converged,
        "all_poses_held": all_poses_held,
        "pose_drift_tol_mm": float(args_cli.solve_tol_mm),
        "reset_during_sweep": reset_happened,
        "sweep_end_reason": end_reason,
        "verdict_ok": ok,
        "curriculum_enabled": False,
    }
    print("[check_seated_success] SUMMARY reward curve -- the JSON's key fields, all envs")
    print(f"[check_seated_success] SUMMARY grid: heights_mm {heights_mm} | laterals_mm "
          f"{laterals_mm} | laterals_x_mm {laterals_x_mm} | tilts_deg {tilts_deg} "
          f"about {args_cli.curve_tilt_axis} | settle "
          f"{settle} | coarse_margin_mm {curve_margin_mm} | kernel_margin_coarse_mm "
          f"{metrics['kernel_margin_coarse_mm']:.2f} | time_penalty_per_step "
          f"{metrics['time_penalty_per_step']:+.6f} | num_envs {num_envs}")
    for r in rows:
        ks = " ".join(f"{v:.6e}" for v in r["kernel_sum"])
        sdf = " ".join(f"{v:.3f}" for v in r["sdf_mean_outside_mm"])
        got_y = " ".join(f"{v:.3f}" for v in r["achieved_lateral_y_mm"])
        got_x = " ".join(f"{v:.3f}" for v in r["achieved_lateral_x_mm"])
        got_depth = " ".join(f"{v:.3f}" for v in r["achieved_depth_mm"])
        print(f"[check_seated_success] SUMMARY row h {r['height_above_entrance_mm']:+7.1f} "
              f"y {r['lateral_y_mm']:+6.2f} x {r['lateral_x_mm']:+6.2f} "
              f"t {r['tilt_deg']:+5.1f} | kernel_sum {ks} | "
              f"sdf_mm {sdf} | interpen_max_mm {max(r['interpen_max_mm']):.3f} | "
              f"force_max_n {max(r['force_n']):.2f} | engaged {any(r['engaged'])!s} "
              f"success {any(r['in_success_region'])!s} in_pocket {all(r['in_pocket'])!s} | "
              f"max_depth_mm {max(r['max_depth_mm']):.3f} | got_y_mm {got_y} | "
              f"got_x_mm {got_x} | "
              f"got_depth_mm {got_depth} | drift {r['pose_drift_mm']:.3f} mm held "
              f"{r['pose_held']!s} | residual "
              f"{r['solve_residual_mm']:.3f} mm conv {r['solve_converged']!s} "
              f"reset {r['reset_on_this_point']!s}")
    print(f"[check_seated_success] SUMMARY end: all_solves_converged {all_converged!s} | "
          f"all_poses_held {all_poses_held!s} (tol {float(args_cli.solve_tol_mm):.3f} mm) | "
          f"reset_during_sweep {reset_happened!s} | sweep_end_reason {end_reason} | "
          f"sweep_end_by_tilt {metrics['sweep_end_by_tilt']} | verdict_ok {ok!s}")
    out = pathlib.Path(args_cli.out)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[check_seated_success] metrics -> {out.resolve()}")
    print(f"[check_seated_success] VERDICT: "
          f"{'REWARD_CURVE_MEASURED' if ok else 'REWARD_CURVE_UNUSABLE'}")
    return 0 if ok else 1


def main() -> int:
    if args_cli.shallow and args_cli.lateral:
        raise SystemExit("[check_seated_success] --shallow and --lateral are "
                         "two different counter-proofs; run them separately.")
    if args_cli.reward_curve and (args_cli.shallow or args_cli.lateral):
        raise SystemExit("[check_seated_success] --reward-curve is a measurement "
                         "sweep, not a counter-proof; run it on its own.")
    if args_cli.curve_coarse_margin_mm is not None and not args_cli.reward_curve:
        raise SystemExit("[check_seated_success] --curve-coarse-margin-mm only "
                         "means anything with --reward-curve.")
    if any(float(y) != 0.0 for y in args_cli.curve_lateral_y_mm) and not args_cli.reward_curve:
        raise SystemExit("[check_seated_success] --curve-lateral-y-mm only "
                         "means anything with --reward-curve.")
    if any(float(x) != 0.0 for x in args_cli.curve_lateral_x_mm) and not args_cli.reward_curve:
        raise SystemExit("[check_seated_success] --curve-lateral-x-mm only "
                         "means anything with --reward-curve.")
    _grid_err = curve_lateral_error(args_cli.curve_heights_mm,
                                    args_cli.curve_lateral_y_mm,
                                    None, args_cli.curve_lateral_x_mm)
    if _grid_err is not None:
        raise SystemExit(f"[check_seated_success] {_grid_err}")
    depth_mm = 15.0 if args_cli.shallow else float(args_cli.depth_mm)
    _block_outer_y_m = float(insertion_tasks_cfg.POCKET_LOCAL_Y_RANGE[1])
    _part_bbox = insertion_tasks_cfg.PART_BBOX_M
    if args_cli.lateral_y_mm is None:
        lateral_y_mm = lateral_probe_default_y_m(
            _block_outer_y_m, float(_part_bbox[0]), float(_part_bbox[1])
        ) * 1000.0
    else:
        lateral_y_mm = float(args_cli.lateral_y_mm)
        err = lateral_probe_y_error(lateral_y_mm, _block_outer_y_m, float(_part_bbox[1]))
        if err is not None:
            raise SystemExit(f"[check_seated_success] {err}")
    lateral_y_m = (lateral_y_mm / 1000.0) if args_cli.lateral else 0.0
    wall_y_mm = float(insertion_tasks_cfg.POCKET_WALL_Y) * 1000.0
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.reward_curve:
        _clamp_err = curve_lateral_error(
            args_cli.curve_heights_mm, args_cli.curve_lateral_y_mm,
            float(env_cfg.osc_pos_clamp_m) * 1000.0,
            args_cli.curve_lateral_x_mm,
        )
        if _clamp_err is not None:
            raise SystemExit(f"[check_seated_success] {_clamp_err}")
    if args_cli.lateral:
        env_cfg.osc_pos_clamp_m = lateral_probe_clamp_m(
            lateral_y_m, float(env_cfg.osc_pos_clamp_m)
        )
    env_cfg.rl_terms_enabled = True
    env_cfg.curriculum_enabled = False
    env_cfg.fixture_pos_noise_xy = 0.0
    env_cfg.obs_noise_pocket_pos_std_m = 0.0
    env_cfg.force_obs_noise_std_n = 0.0
    env_cfg.grasp_obs_offset_x_m = 0.0
    env_cfg.w_depth_progress = 0.0
    env_cfg.w_tilt = 0.0
    env_cfg.start_tip_above_entrance = None
    curve_margin_mm = args_cli.curve_coarse_margin_mm
    if curve_margin_mm is not None:
        if curve_margin_mm <= 0.0:
            raise SystemExit("[check_seated_success] --curve-coarse-margin-mm "
                             "must be positive; a = acosh(10)/margin.")
        env_cfg.kernel_a_coarse = math.acosh(10.0) / (curve_margin_mm / 1000.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.robot
    num_envs = unwrapped.num_envs

    peg_body_idx = unwrapped._peg_body_idx
    if peg_body_idx is None:
        raise SystemExit("[check_seated_success] no welded tool -- nothing to seat.")
    n_joints = len(robot.joint_names)
    jacobi_idx = peg_body_idx - 1 if robot.is_fixed_base else peg_body_idx

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls",
            ik_params={"lambda_val": args_cli.ik_lambda},
        ),
        num_envs=num_envs, device=device,
    )

    band_mm = (float(env_cfg.depth_min) * 1000.0, float(env_cfg.depth_max) * 1000.0)
    reward_peak = 1.0 + float(env_cfg.w_engaged) + float(env_cfg.w_success)
    reward_params = {
        "a_coarse": float(env_cfg.kernel_a_coarse), "b_coarse": float(env_cfg.kernel_b_coarse),
        "a_mid": float(env_cfg.kernel_a_mid), "b_mid": float(env_cfg.kernel_b_mid),
        "a_fine": float(env_cfg.kernel_a_fine), "b_fine": float(env_cfg.kernel_b_fine),
        "w_engaged": float(env_cfg.w_engaged), "w_success": float(env_cfg.w_success),
        "interpen_thresh_m": float(env_cfg.interpen_thresh),
        "time_penalty_per_step": float(env_cfg.time_penalty_per_step),
        "action_rate_scale": float(env_cfg.action_rate_scale),
    }

    print("[check_seated_success] ---- teleport success identity test ----")
    print(f"[check_seated_success] marker: {SCRIPT_MARKER}")
    if args_cli.lateral:
        _mode = (f"LATERAL COUNTER-PROOF (y = {lateral_y_mm:.2f} mm, "
                 f"block outer edge {_block_outer_y_m * 1000.0:.2f} mm, "
                 f"wall {wall_y_mm:.2f} mm, OSC tip clamp widened to "
                 f"{float(env_cfg.osc_pos_clamp_m) * 1000.0:.2f} mm)")
    elif args_cli.shallow:
        _mode = "SHALLOW COUNTER-PROOF"
    else:
        _mode = "SEATED"
    print(f"[check_seated_success] mode: {_mode}, "
          f"commanded depth {depth_mm:.1f} mm, band {band_mm[0]:.1f}-{band_mm[1]:.1f} mm, "
          f"reward peak {reward_peak:.2f}")

    zero_action = torch.zeros(num_envs, unwrapped.cfg.action_space, device=device)
    target_rel_z = -depth_mm / 1000.0

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    pocket_quat = _slice(obs, "pocket_quat").clone()

    def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _circular_mean(a: torch.Tensor) -> float:
        return float(torch.atan2(torch.sin(a).mean(), torch.cos(a).mean()))

    with torch.no_grad():
        _baseline_y = sp.signed_tilt_angle(
            _slice(obs, "ee_quat"), pocket_quat, sp.TILT_AXES["y"])
        _baseline_x = sp.signed_tilt_angle(
            _slice(obs, "ee_quat"), pocket_quat, sp.TILT_AXES["x"])
    baseline_rad = {"y": _circular_mean(_baseline_y), "x": _circular_mean(_baseline_x)}
    baseline_spread_deg = {
        "y": math.degrees(float(_wrap_pi(_baseline_y - baseline_rad["y"]).abs().max())),
        "x": math.degrees(float(_wrap_pi(_baseline_x - baseline_rad["x"]).abs().max())),
    }
    print(f"[check_seated_success] tilt baseline at reset (obs ee_quat vs "
          f"pocket_quat, signed): y {math.degrees(baseline_rad['y']):+.3f} deg "
          f"(spread {baseline_spread_deg['y']:.3f}), x "
          f"{math.degrees(baseline_rad['x']):+.3f} deg "
          f"(spread {baseline_spread_deg['x']:.3f}) -- every commanded "
          f"--curve-tilt-deg is added ON TOP of this baseline, not measured "
          f"against a literal zero.")
    if max(baseline_spread_deg.values()) > 1.0:
        raise SystemExit(
            f"[check_seated_success] tilt baseline spread "
            f"{max(baseline_spread_deg.values()):.3f} deg across envs at reset "
            f"-- the envs are not at the same pose (fixture yaw noise active?), "
            f"the ladder needs one shared baseline. REFUSED before any tilt is "
            f"commanded.")

    tilt_axis = sp.TILT_AXES[args_cli.curve_tilt_axis]
    tol_tilt_rad = math.radians(float(args_cli.solve_tol_tilt_deg))
    ee_body_idx = unwrapped._ee_body_idx

    def _solve_to(goal_rel_z: float, goal_lateral_y: float, goal_tilt_rad: float | None = None,
                  goal_lateral_x: float = 0.0):
        err_mm = None
        err_tilt = None
        tilt_active = goal_tilt_rad is not None
        for _ in range(int(args_cli.solve_steps)):
            with torch.no_grad():
                *_, tip_rel = unwrapped._peg_geometry()
                d_pocket = torch.zeros(num_envs, 6, device=device)
                d_pocket[:, 0] = goal_lateral_x - tip_rel[:, 0]
                d_pocket[:, 1] = goal_lateral_y - tip_rel[:, 1]
                d_pocket[:, 2] = goal_rel_z - tip_rel[:, 2]
                if tilt_active:
                    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
                    tilt_now = sp.signed_tilt_angle(ee_quat_w, pocket_quat, tilt_axis)
                    d_tilt = _wrap_pi(goal_tilt_rad - tilt_now)
                    d_pocket[:, 3 + tilt_axis] = d_tilt
                    err_tilt = float(d_tilt.abs().max().item())
                err_mm = float(torch.linalg.norm(d_pocket[:, 0:3], dim=-1).max().item() * 1000.0)

                root_pose_w = robot.data.root_pose_w
                cmd_base = sp.command_to_base_frame(d_pocket, pocket_quat, root_pose_w[:, 3:7])
                ee_pose_w = robot.data.body_pose_w[:, peg_body_idx]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, :n_joints]
                ik.set_command(cmd_base, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
                joint_pos_des = ik.compute(ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos).clone()
                robot.write_joint_state_to_sim(joint_pos_des, torch.zeros_like(joint_pos_des))
                robot.set_joint_position_target(joint_pos_des)
                unwrapped._joint_targets[:] = joint_pos_des
            if (err_mm is not None and err_mm < float(args_cli.solve_tol_mm)
                    and (not tilt_active
                         or (err_tilt is not None and err_tilt < tol_tilt_rad))):
                break

        with torch.no_grad():
            *_, tip_rel = unwrapped._peg_geometry()
            err_mm = float(torch.linalg.norm(
                torch.stack((tip_rel[:, 0] - goal_lateral_x,
                             tip_rel[:, 1] - goal_lateral_y,
                             tip_rel[:, 2] - goal_rel_z), dim=-1),
                dim=-1,
            ).max().item() * 1000.0)
            if tilt_active:
                ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
                tilt_now = sp.signed_tilt_angle(ee_quat_w, pocket_quat, tilt_axis)
                err_tilt = float(_wrap_pi(goal_tilt_rad - tilt_now).abs().max().item())
        converged = err_mm < float(args_cli.solve_tol_mm) and (
            not tilt_active or err_tilt < tol_tilt_rad)
        err_tilt_deg = math.degrees(err_tilt) if tilt_active else 0.0
        return err_mm, converged, err_tilt_deg

    if args_cli.reward_curve:
        rc = _run_reward_curve(
            env=env, unwrapped=unwrapped, solve_to=_solve_to,
            zero_action=zero_action, reward_params=reward_params,
            curve_margin_mm=curve_margin_mm, env_cfg=env_cfg, num_envs=num_envs,
            tilt_baseline_rad=baseline_rad,
        )
        env.close()
        return rc

    solved_err_mm, solve_converged, _ = _solve_to(target_rel_z, lateral_y_m)
    print(f"[check_seated_success] solve residual {solved_err_mm:.3f} mm "
          f"({'converged' if solve_converged else 'NOT CONVERGED -- not a contact result'})")

    terminated_any = False
    success_term_any = False
    force_abort_any = False
    steps_remaining = [0.0] * num_envs
    rewards = torch.zeros(num_envs, device=device)
    for step_i in range(int(args_cli.settle_steps)):
        with torch.no_grad():
            pre_force_n = [float(v) for v in
                           insertion_math.force_magnitude(unwrapped._force_smooth).tolist()]
            *_, pre_tip_rel = unwrapped._peg_geometry()
            pre_tip_mm = [[round(float(c) * 1000.0, 2) for c in row]
                          for row in pre_tip_rel.tolist()]
            pre_max_depth_mm = [float(v) * 1000.0 for v in unwrapped._max_depth.tolist()]

            obs_dict, rew, terminated, truncated, _ = env.step(zero_action)
            obs = obs_dict["policy"]
            rewards = rew
            if bool(terminated.any()):
                terminated_any = True
                success_term_any = bool(unwrapped._last_success.any())
                force_abort_any = bool(unwrapped._last_force_abort.any())
                steps_remaining = [float(v) for v in
                                   unwrapped._last_steps_remaining.tolist()]
                if success_term_any and not force_abort_any:
                    print(f"[check_seated_success] SUCCESS TERMINATION at settle "
                          f"step {step_i}: success "
                          f"{[bool(v) for v in unwrapped._last_success.tolist()]}, "
                          f"steps remaining {[round(v, 1) for v in steps_remaining]}, "
                          f"paid {[round(float(v), 4) for v in rew.tolist()]}")
                    break
                print(f"[check_seated_success] ABORT at settle step {step_i}: "
                      f"terminated {[bool(v) for v in terminated.tolist()]}")
                print(f"[check_seated_success]   pre-step force N (the abort "
                      f"input, limit {float(unwrapped.cfg.force_abort_f_max_n):.1f} N): "
                      f"{[round(v, 2) for v in pre_force_n]}")
                print(f"[check_seated_success]   pre-step tip_rel xyz mm: {pre_tip_mm}")
                print(f"[check_seated_success]   abort-step depth mm: "
                      f"{[round(float(v) * 1000.0, 3) for v in unwrapped._last_depth.tolist()]}   "
                      f"interpen max mm: "
                      f"{[round(float(v) * 1000.0, 4) for v in unwrapped._last_interpen_max.tolist()]}")
                print(f"[check_seated_success]   abort-step in_region: "
                      f"{[bool(v) for v in unwrapped._last_in_region.tolist()]}   engaged: "
                      f"{[bool(v) for v in unwrapped._last_engaged.tolist()]}")
                break

    with torch.no_grad():
        if terminated_any:
            tip_rel = pre_tip_rel
            force_n = pre_force_n
            max_depth_mm = pre_max_depth_mm
        else:
            tip_rel = _slice(obs, "tip_rel")
            force = _slice(obs, "force")
            force_n = [float(v) for v in insertion_math.force_magnitude(force).tolist()]
            max_depth_mm = [float(v) * 1000.0 for v in unwrapped._max_depth.tolist()]
        depths = [float(v) * -1000.0 for v in tip_rel[:, 2].tolist()]
        lat_y_mm = [float(v) * 1000.0 for v in tip_rel[:, 1].tolist()]
        in_region = [bool(v) for v in unwrapped._last_in_region.tolist()]
        engaged = [bool(v) for v in unwrapped._last_engaged.tolist()]
        reward_list = [float(v) for v in rewards.tolist()]
        sdf_mm = [float(v) * 1000.0 for v in unwrapped._last_sdf_dist.tolist()]
        interpen_mm = [float(v) * 1000.0 for v in unwrapped._last_interpen_max.tolist()]
        if unwrapped._force_body_idx is not None:
            _raw_w = unwrapped.robot.data.body_incoming_joint_wrench_b[
                :, unwrapped._force_body_idx, 0:3]
            force_raw_n = [float(v) for v in
                           insertion_math.force_magnitude(_raw_w).tolist()]
        else:
            force_raw_n = []

    print(f"[check_seated_success] achieved depth mm: {[round(d, 3) for d in depths]}")
    print(f"[check_seated_success] sdf mean-outside mm: {[round(v, 4) for v in sdf_mm]}   "
          f"interpen max mm: {[round(v, 4) for v in interpen_mm]}")
    action_rate = [float(v) for v in unwrapped._last_action_rate.tolist()]
    expected_list = [
        expected_reward(sdf_mm[i], interpen_mm[i], engaged[i], in_region[i],
                        reward_params, action_rate[i], steps_remaining[i])
        for i in range(num_envs)
    ]
    print(f"[check_seated_success] force N: {[round(v, 2) for v in force_n]}   "
          f"reward: {[round(v, 4) for v in reward_list]}")
    print(f"[check_seated_success] force N RAW (untared wrench, the control): "
          f"{[round(v, 2) for v in force_raw_n]}")
    print(f"[check_seated_success] reward recomputed offline: "
          f"{[round(v, 4) for v in expected_list]}   (P8 identity; the column "
          f"includes the -1/T time penalty, the action-rate term and the "
          f"success payout)")
    print(f"[check_seated_success] steps remaining at the exit: "
          f"{[round(v, 1) for v in steps_remaining]}   action-rate norm: "
          f"{[round(v, 6) for v in action_rate]}")
    print(f"[check_seated_success] in_region: {in_region}   engaged: {engaged}   "
          f"terminated_any: {terminated_any}   success exit: {success_term_any}   "
          f"force abort: {force_abort_any}")
    print(f"[check_seated_success] lateral y mm: {[round(v, 2) for v in lat_y_mm]}   "
          f"(pocket wall {wall_y_mm:.2f} mm)   "
          f"max_depth metric mm: {[round(v, 4) for v in max_depth_mm]}")

    ok, points = judge_run(
        depths_mm=depths, in_region=in_region, engaged=engaged, rewards=reward_list,
        sdf_mm=sdf_mm, interpen_mm=interpen_mm,
        terminated_any=terminated_any, success_term_any=success_term_any,
        force_abort_any=force_abort_any,
        solve_converged=solve_converged, band_mm=band_mm, reward_peak=reward_peak,
        reward_params=reward_params, shallow=bool(args_cli.shallow),
        lateral=bool(args_cli.lateral), lat_y_mm=lat_y_mm, wall_y_mm=wall_y_mm,
        max_depth_mm=max_depth_mm, force_n=force_n,
        steps_remaining=steps_remaining, action_rate=action_rate,
    )
    for name, pok, detail in points:
        print(f"[check_seated_success]   {'PASS' if pok else 'FAIL'}  {name}"
              + (f"  -- {detail}" if detail else ""))

    metrics = {
        "run": "check_seated_success (teleport success identity test)",
        "marker": SCRIPT_MARKER,
        "task": args_cli.task,
        "num_envs": num_envs,
        "mode": ("lateral" if args_cli.lateral
                 else "shallow" if args_cli.shallow else "seated"),
        "lateral_y_mm": lat_y_mm,
        "commanded_lateral_y_mm": lateral_y_mm if args_cli.lateral else None,
        "osc_pos_clamp_m": float(env_cfg.osc_pos_clamp_m),
        "settle_steps": int(args_cli.settle_steps),
        "block_outer_y_mm": _block_outer_y_m * 1000.0,
        "fixture_pos_noise_xy_m": float(env_cfg.fixture_pos_noise_xy),
        "obs_noise_pocket_pos_std_m": float(env_cfg.obs_noise_pocket_pos_std_m),
        "force_obs_noise_std_n": float(env_cfg.force_obs_noise_std_n),
        "grasp_obs_offset_x_m": float(env_cfg.grasp_obs_offset_x_m),
        "pocket_wall_y_mm": wall_y_mm,
        "max_depth_metric_mm": max_depth_mm,
        "commanded_depth_mm": depth_mm,
        "band_mm": list(band_mm),
        "reward_peak": reward_peak,
        "solve_residual_mm": solved_err_mm,
        "solve_converged": solve_converged,
        "achieved_depth_mm": depths,
        "sdf_mean_outside_mm": sdf_mm,
        "interpen_max_mm": interpen_mm,
        "force_n": force_n,
        "force_n_raw_untared": force_raw_n,
        "reward": reward_list,
        "reward_expected": expected_list,
        "reward_params": reward_params,
        "reward_column_includes": (
            "kernel_sum + engaged + success bonuses, SAPU-scaled; the -1/T time "
            "penalty; the action-rate penalty; and on a success-termination "
            "step the equivalent-return payout (step value x steps_remaining)"
        ),
        "steps_remaining_at_exit": steps_remaining,
        "action_rate_norm": action_rate,
        "in_region": in_region,
        "engaged": engaged,
        "terminated_any": terminated_any,
        "success_termination": success_term_any,
        "force_abort": force_abort_any,
        "points": [{"point": n, "ok": o, "detail": d} for n, o, d in points],
        "verdict_ok": ok,
        "curriculum_enabled": False,
    }
    out = pathlib.Path(args_cli.out)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[check_seated_success] metrics -> {out.resolve()}")

    if args_cli.lateral:
        print(f"[check_seated_success] VERDICT: "
              f"{'LATERAL_CONTROL_HOLDS' if ok else 'LATERAL_CONTROL_FAILED'}")
    elif args_cli.shallow:
        print(f"[check_seated_success] VERDICT: "
              f"{'SHALLOW_CONTROL_HOLDS' if ok else 'SHALLOW_CONTROL_FAILED'}")
    else:
        print(f"[check_seated_success] VERDICT: "
              f"{'SEATED_IDENTITY_HOLDS' if ok else 'SEATED_IDENTITY_FAILED'}")

    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException:
        traceback.print_exc()
        exit_with(simulation_app, 1, tag="check_seated_success-main")
    exit_with(simulation_app, rc, tag="check_seated_success")
