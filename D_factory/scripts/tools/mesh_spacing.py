# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""The weld sweep's tolerance range, measured from the mesh instead of typed."""

from __future__ import annotations

import argparse
import math

SPACING_MARKER = "mesh_spacing-2026-08-28a"

DEFAULT_MARGIN = 10.0

DEFAULT_FLOOR_FRACTION = 1e-9


def unique_positions(points) -> tuple[list, list, int]:
    """Collapse exact-duplicate coordinates."""
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
    """Smallest distance between two DISTINCT point positions."""
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
                break
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


def weld_tolerance_series(diag: float,
                          d_min: float,
                          floor_fraction: float = DEFAULT_FLOOR_FRACTION,
                          margin: float = DEFAULT_MARGIN) -> dict:
    """Decade range of weld tolerances that provably cannot fuse two points."""
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
    while 10.0 ** k * diag > allowed_top:
        k -= 1

    floor_exp = round(math.log10(floor_fraction))
    if k < floor_exp:
        return out

    fractions = tuple(10.0 ** e for e in range(floor_exp, k + 1))
    out["fractions"] = fractions
    out["top_fraction"] = fractions[-1]
    return out


def _self_test() -> int:
    checked = 0

    r = min_point_spacing([(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)])
    assert r is not None and abs(r["d_min"] - 5.0) < 1e-12, r
    assert {r["index_a"], r["index_b"]} == {0, 1}, r
    checked += 1

    r = min_point_spacing([(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
                           (1.0, 1.0, 3.0)])
    assert r is not None and abs(r["d_min"] - 2.0) < 1e-12, r
    assert r["n_exact_duplicate_points"] == 2, r
    assert r["n_distinct_positions"] == 2, r
    checked += 1

    grid = [(x * 0.5, y * 0.5, z * 0.5)
            for x in range(4) for y in range(4) for z in range(4)]
    r = min_point_spacing(grid)
    assert r is not None and abs(r["d_min"] - 0.5) < 1e-12, r
    checked += 1

    tricky = [(0.0, 0.0, 0.0), (1.0, 50.0, 0.0), (2.0, 0.0, 0.0),
              (3.0, 50.0, 0.0), (7.0, -3.0, 2.0), (7.0, -3.0, 2.05)]
    r = min_point_spacing(tricky)
    assert r is not None
    assert abs(r["d_min"] - _brute_force_spacing(tricky)) < 1e-12, r
    assert abs(r["d_min"] - 0.05) < 1e-12, r
    checked += 1

    assert min_point_spacing([(2.0, 2.0, 2.0), (2.0, 2.0, 2.0)]) is None
    assert min_point_spacing([]) is None
    checked += 1

    diag = 248.96
    s = weld_tolerance_series(diag, d_min=1.0)
    assert s["top_fraction"] is not None
    assert s["top_fraction"] * diag <= s["allowed_top_point_units"], s
    assert s["top_fraction"] * diag * math.sqrt(3.0) < 1.0, s
    checked += 1

    rt63_spacing = 0.2
    old_top_tol = 1e-3 * 248.96
    assert old_top_tol * math.sqrt(3.0) > rt63_spacing,         "the replay must reproduce the over-weld, not hide it"
    rt63 = weld_tolerance_series(248.96, d_min=rt63_spacing)
    assert rt63["top_fraction"] is not None
    assert rt63["top_fraction"] * 248.96 < old_top_tol, rt63
    assert rt63["top_fraction"] * 248.96 * math.sqrt(3.0) < rt63_spacing, rt63
    assert rt63["top_fraction"] == 1e-5, rt63
    checked += 1

    tiny = weld_tolerance_series(248.96, d_min=1e-9)
    assert tiny["fractions"] == (), tiny
    assert tiny["top_fraction"] is None, tiny
    checked += 1

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
