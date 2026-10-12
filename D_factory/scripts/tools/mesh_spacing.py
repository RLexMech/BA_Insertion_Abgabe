# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""The weld sweep's tolerance range, measured from the mesh instead of typed.

WHY THIS EXISTS
---------------
``add_fixture_collision.py --weld-sweep`` re-welds a mesh's points by position
over a tolerance range and recounts the unpaired edges. The range was a hand
written tuple, and it was wrong twice in opposite directions:

  * RT-62 -- the range stopped at 1e-5 of the diagonal, 2.49 um on this
    fixture. Too FINE: the top row never reached a gap that matters, so the
    question the sweep exists to answer stayed open.
  * RT-63 -- the top was raised by hand to 1e-3, 0.249 mm. Too COARSE: the
    floor ROSE from 9 unpaired edges to 12 and five edges came out used four
    times. At that tolerance the weld fused two points that are genuinely
    different, so the sweep manufactured the defect it was measuring.

A hand-set range fails in both directions and neither failure is visible in
the output -- the log looks like a sweep either way. The mesh, however, knows
the answer: no weld tolerance may reach the smallest distance between two
DISTINCT points, because reaching it destroys real geometry.

THE BOUND, derived rather than chosen
-------------------------------------
The weld rounds each coordinate onto a grid of width ``t`` (``round(x / t)``).
Two points share a cell only if they differ by at most ``t`` on EVERY axis, so
the largest separation that can still be welded is ``t * sqrt(3)`` (the cell
diagonal). Therefore

    no distinct pair can ever be welded  <=>  t * sqrt(3) < d_min
                                         <=>  t < d_min / sqrt(3)

``d_min / sqrt(3)`` is the HARD BOUND. It is arithmetic, not a judgement.

The only chosen number in this file is ``DEFAULT_MARGIN = 10.0``: the range
runs in decades, so the top rung is placed one full decade below the hard
bound. One decade, because that is the granularity of the range itself.

EXACT DUPLICATES ARE NOT DISTINCT
---------------------------------
Points that carry the exact same coordinates are precisely what the sweep is
supposed to merge (RT-60: 894 of 903 unpaired edges vanish at the very first
tolerance). They are collapsed before the spacing is measured, and their count
is reported, so ``d_min`` describes the mesh's real vertex spacing.

If ``d_min`` comes out so small that not one decade survives, this module
returns an EMPTY range. It invents no fallback. An empty range is a finding to
report, not a number to guess.

THE SELF-TEST CONTRACT, written before the implementation
---------------------------------------------------------
``python scripts/tools/mesh_spacing.py --self-test`` must prove:

   1. two points -> their distance, and the pair that produced it
   2. exact duplicates are collapsed, counted, and never returned as d_min 0
   3. a known grid returns the grid pitch, not a diagonal of it
   4. the sweep-line agrees with brute force on a case built to defeat an
      x-only break (the closest pair sits late in x order)
   5. a point set with fewer than two distinct positions returns None
   6. the top rung of the range is strictly below the hard bound over margin
   7. RT-63 replayed: the over-welding rung is reproduced as too coarse, and
      the derived range refuses it -- the bug is shown, not described
   8. a d_min below the range floor yields an empty range, not a guess
   9. the range is contiguous decades starting at the floor fraction

Usage:

    python scripts/tools/mesh_spacing.py --self-test
"""

from __future__ import annotations

import argparse
import math

SPACING_MARKER = "mesh_spacing-2026-08-28a"

# The range runs in decades; the top rung sits one full decade below the hard
# bound. This is the ONLY chosen number here -- everything else is derived.
DEFAULT_MARGIN = 10.0

# Far below float32 spacing, so the first row is always "no weld at all".
# Carried over unchanged from the hand-written tuple this module replaces.
DEFAULT_FLOOR_FRACTION = 1e-9


# ===========================================================================
#  Spacing -- pure, no pxr, no numpy
# ===========================================================================
def unique_positions(points) -> tuple[list, list, int]:
    """Collapse exact-duplicate coordinates.

    Returns ``(positions, first_index, n_duplicate_points)`` where
    ``first_index[k]`` is the index in the ORIGINAL array that first carried
    position ``k``. The duplicate count is points dropped, not groups.
    """
    seen: dict = {}
    positions: list = []
    first_index: list = []
    n_dup = 0
    for i, p in enumerate(points):
        key = (float(p[0]), float(p[1]), float(p[2]))
        if key in seen:
            n_dup += 1
            continue
        seen[key] = len(positions)
        positions.append(key)
        first_index.append(i)
    return positions, first_index, n_dup


def min_point_spacing(points) -> dict | None:
    """Smallest distance between two DISTINCT point positions.

    Exact. Sorts by x, then compares each point only against those that follow
    while their x-gap is still shorter than the best distance found so far --
    the standard sweep line. Collapsing duplicates before sorting keeps the
    comparison count low on CAD meshes, where many points share an x.

    Returns ``None`` when fewer than two distinct positions exist. Otherwise a
    dict with ``d_min``, the two ORIGINAL indices, the two coordinates, the
    number of distinct positions and the number of exact duplicates dropped.
    """
    positions, first_index, n_dup = unique_positions(points)
    n = len(positions)
    if n < 2:
        return None

    order = sorted(range(n), key=lambda k: positions[k])
    best_sq = float("inf")
    best_pair = (-1, -1)
    for a in range(n):
        i = order[a]
        xi, yi, zi = positions[i]
        for b in range(a + 1, n):
            j = order[b]
            xj, yj, zj = positions[j]
            dx = xj - xi
            if dx * dx >= best_sq:
                break  # x is sorted: every later j is at least this far away
            dy = yj - yi
            dz = zj - zi
            d_sq = dx * dx + dy * dy + dz * dz
            if d_sq < best_sq:
                best_sq = d_sq
                best_pair = (i, j)

    i, j = best_pair
    return {
        "d_min": math.sqrt(best_sq),
        "index_a": first_index[i],
        "index_b": first_index[j],
        "point_a": list(positions[i]),
        "point_b": list(positions[j]),
        "n_distinct_positions": n,
        "n_exact_duplicate_points": n_dup,
    }


def _brute_force_spacing(positions) -> float:
    """Reference implementation for the self-test. O(n^2), obviously correct."""
    best = float("inf")
    for a in range(len(positions)):
        for b in range(a + 1, len(positions)):
            d = math.dist(positions[a], positions[b])
            if 0.0 < d < best:
                best = d
    return best


# ===========================================================================
#  The range -- pure
# ===========================================================================
def weld_tolerance_series(diag: float,
                          d_min: float,
                          floor_fraction: float = DEFAULT_FLOOR_FRACTION,
                          margin: float = DEFAULT_MARGIN) -> dict:
    """Decade range of weld tolerances that provably cannot fuse two points.

    ``diag`` and ``d_min`` are both in the mesh's own point units; the range is
    returned as FRACTIONS OF THE DIAGONAL, which is what the sweep consumes and
    what makes it independent of whether the points are metres or millimetres
    (RT-61).

    Returns a dict with ``fractions`` (possibly empty), ``hard_bound``
    (``d_min / sqrt(3)``, the largest tolerance that still cannot fuse a
    distinct pair), ``allowed_top`` (hard bound divided by the margin) and
    ``top_fraction`` (``None`` when nothing fits).
    """
    hard_bound = d_min / math.sqrt(3.0)
    allowed_top = hard_bound / margin
    out = {
        "hard_bound_point_units": hard_bound,
        "allowed_top_point_units": allowed_top,
        "margin": margin,
        "floor_fraction": floor_fraction,
        "fractions": (),
        "top_fraction": None,
    }
    if diag <= 0.0 or allowed_top <= 0.0:
        return out

    k = math.floor(math.log10(allowed_top / diag))
    # log10 rounding can land one ulp high; step down until the rung is
    # genuinely inside the allowance. Never step up.
    while 10.0 ** k * diag > allowed_top:
        k -= 1

    floor_exp = round(math.log10(floor_fraction))
    if k < floor_exp:
        return out  # not one decade fits -- report it, do not invent one

    fractions = tuple(10.0 ** e for e in range(floor_exp, k + 1))
    out["fractions"] = fractions
    out["top_fraction"] = fractions[-1]
    return out


# ===========================================================================
#  Self-test -- the nine cases above. No USD, no Isaac, no numpy.
# ===========================================================================
def _self_test() -> int:
    checked = 0

    # 1 -- two points
    r = min_point_spacing([(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)])
    assert r is not None and abs(r["d_min"] - 5.0) < 1e-12, r
    assert {r["index_a"], r["index_b"]} == {0, 1}, r
    checked += 1

    # 2 -- exact duplicates are collapsed and counted, never reported as 0
    r = min_point_spacing([(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
                           (1.0, 1.0, 3.0)])
    assert r is not None and abs(r["d_min"] - 2.0) < 1e-12, r
    assert r["n_exact_duplicate_points"] == 2, r
    assert r["n_distinct_positions"] == 2, r
    checked += 1

    # 3 -- a grid returns the pitch, not a face or body diagonal
    grid = [(x * 0.5, y * 0.5, z * 0.5)
            for x in range(4) for y in range(4) for z in range(4)]
    r = min_point_spacing(grid)
    assert r is not None and abs(r["d_min"] - 0.5) < 1e-12, r
    checked += 1

    # 4 -- the sweep line must agree with brute force where an x-only break
    # could cut too early: the closest pair sits at the end of the x order.
    tricky = [(0.0, 0.0, 0.0), (1.0, 50.0, 0.0), (2.0, 0.0, 0.0),
              (3.0, 50.0, 0.0), (7.0, -3.0, 2.0), (7.0, -3.0, 2.05)]
    r = min_point_spacing(tricky)
    assert r is not None
    assert abs(r["d_min"] - _brute_force_spacing(tricky)) < 1e-12, r
    assert abs(r["d_min"] - 0.05) < 1e-12, r
    checked += 1

    # 5 -- fewer than two distinct positions
    assert min_point_spacing([(2.0, 2.0, 2.0), (2.0, 2.0, 2.0)]) is None
    assert min_point_spacing([]) is None
    checked += 1

    # 6 -- the top rung is strictly inside the allowance and inside the bound
    diag = 248.96
    s = weld_tolerance_series(diag, d_min=1.0)
    assert s["top_fraction"] is not None
    assert s["top_fraction"] * diag <= s["allowed_top_point_units"], s
    assert s["top_fraction"] * diag * math.sqrt(3.0) < 1.0, s
    checked += 1

    # 7 -- RT-63 replayed. That run welded at 1e-3 of a 248.96 diagonal =
    # 0.249 point units and fused distinct points, so the mesh's real spacing
    # is smaller than that rung's reach of 0.249 * sqrt(3) = 0.431. Shown here
    # on a spacing of 0.2 point units: the hand-set rung provably reaches
    # across it, and the derived range refuses that rung by three decades.
    rt63_spacing = 0.2
    old_top_tol = 1e-3 * 248.96
    assert old_top_tol * math.sqrt(3.0) > rt63_spacing,         "the replay must reproduce the over-weld, not hide it"
    rt63 = weld_tolerance_series(248.96, d_min=rt63_spacing)
    assert rt63["top_fraction"] is not None
    assert rt63["top_fraction"] * 248.96 < old_top_tol, rt63
    assert rt63["top_fraction"] * 248.96 * math.sqrt(3.0) < rt63_spacing, rt63
    assert rt63["top_fraction"] == 1e-5, rt63
    checked += 1

    # 8 -- nothing fits: empty range, no invented rung
    tiny = weld_tolerance_series(248.96, d_min=1e-9)
    assert tiny["fractions"] == (), tiny
    assert tiny["top_fraction"] is None, tiny
    checked += 1

    # 9 -- contiguous decades from the floor
    s = weld_tolerance_series(248.96, d_min=1.0)
    exps = [round(math.log10(f)) for f in s["fractions"]]
    assert exps == list(range(-9, exps[-1] + 1)), exps
    assert abs(s["fractions"][0] - DEFAULT_FLOOR_FRACTION) < 1e-20, s
    checked += 1

    assert checked == 9, f"only {checked} of 9 cases ran"
    print(f"[mesh_spacing] self-test: {checked}/9 cases passed ({SPACING_MARKER})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measured weld-tolerance bound for the fixture weld sweep.")
    parser.add_argument("--self-test", action="store_true",
                        help="Prove the spacing and range logic. Needs no USD and no Isaac.")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
