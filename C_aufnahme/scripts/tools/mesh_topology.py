# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Triangle-mesh topology and where a defect SITS."""

from __future__ import annotations

import argparse

MESH_TOPOLOGY_MARKER = "mesh_topology-2026-08-30a"


def edge_use_counts(counts, indices) -> dict:
    """``{(a, b): times used}`` over a flat face array, paired by point INDEX."""
    total = sum(int(n) for n in counts)
    if total != len(indices):
        raise ValueError(
            f"face corner counts sum to {total} but the index array has "
            f"{len(indices)} entries -- the two arrays do not describe the same mesh"
        )
    edges: dict = {}
    k = 0
    for n in counts:
        n = int(n)
        face = indices[k:k + n]
        k += n
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1
    return edges


def unpaired_edges(counts, indices) -> int:
    """Edges NOT used by exactly two faces."""
    return sum(1 for c in edge_use_counts(counts, indices).values() if c != 2)


def split_unpaired(counts, indices) -> dict:
    """``{\"boundary\": n, \"non_manifold\": n, \"total\": n}`` -- the split, measured."""
    counts_by_use = edge_use_counts(counts, indices)
    bnd = sum(1 for c in counts_by_use.values() if c == 1)
    nm = sum(1 for c in counts_by_use.values() if c >= 3)
    return {"boundary": bnd, "non_manifold": nm, "total": bnd + nm}


def weld_remap(points, tol: float) -> tuple:
    """Group points by ROUNDED position."""
    if tol <= 0.0:
        raise ValueError(f"weld tolerance must be positive, got {tol!r}")
    key_of: dict = {}
    rep: dict = {}
    remap = [0] * len(points)
    for i, p in enumerate(points):
        key = (round(float(p[0]) / tol), round(float(p[1]) / tol), round(float(p[2]) / tol))
        if key not in key_of:
            key_of[key] = len(key_of)
            rep[key_of[key]] = i
        remap[i] = key_of[key]
    return remap, rep


def welded_unpaired_edges(points, counts, indices, tol: float) -> tuple:
    """``(unpaired count, unique point count)`` after welding at ``tol``."""
    remap, _ = weld_remap(points, tol)
    merged = [remap[int(i)] for i in indices]
    return unpaired_edges(counts, merged), len(set(remap))


def survivor_edges(points, counts, indices, tol: float | None = None) -> list:
    """Every edge not used by exactly two faces, WITH its two endpoints."""
    if tol is None:
        merged = [int(i) for i in indices]
        rep = {i: i for i in range(len(points))}
    else:
        remap, rep = weld_remap(points, tol)
        merged = [remap[int(i)] for i in indices]

    out = []
    for (a, b), c in edge_use_counts(counts, merged).items():
        if c == 2:
            continue
        pa, pb = points[rep[a]], points[rep[b]]
        out.append({
            "uses": c,
            "kind": "boundary" if c == 1 else "non-manifold",
            "a": [float(v) for v in pa],
            "b": [float(v) for v in pb],
        })
    return out


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to an axis-aligned box."""
    s = 0.0
    for a in range(3):
        lo_a, hi_a = float(lo[a]), float(hi[a])
        if lo_a > hi_a:
            raise ValueError(f"box axis {a} is inverted: lo {lo_a} > hi {hi_a}")
        d = 0.0
        if p[a] < lo_a:
            d = lo_a - float(p[a])
        elif p[a] > hi_a:
            d = float(p[a]) - hi_a
        s += d * d
    return s ** 0.5


def segment_box_distance(a, b, lo, hi, iterations: int = 200) -> float:
    """Distance from the SEGMENT a--b to an axis-aligned box."""
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    dx, dy, dz = float(b[0]) - ax, float(b[1]) - ay, float(b[2]) - az

    def at(t: float) -> float:
        return point_box_distance((ax + t * dx, ay + t * dy, az + t * dz), lo, hi)

    t_lo, t_hi = 0.0, 1.0
    for _ in range(int(iterations)):
        m1 = t_lo + (t_hi - t_lo) / 3.0
        m2 = t_hi - (t_hi - t_lo) / 3.0
        if at(m1) <= at(m2):
            t_hi = m2
        else:
            t_lo = m1
    return min(at(t_lo), at(t_hi), at(0.0), at(1.0))


def plane_hits(points, axis: int, value: float, tol: float) -> int:
    """How many points lie within ``tol`` of the plane ``axis = value``."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")
    if tol < 0.0:
        raise ValueError(f"tolerance must not be negative, got {tol!r}")
    return sum(1 for p in points if abs(float(p[axis]) - float(value)) <= tol)


def _cube(scale: float = 1.0, unwelded: bool = False, drop_faces: int = 0) -> tuple:
    """A closed axis-aligned cube of side ``scale``, as ``(points, counts, indices)``."""
    s = scale
    pts = [(0.0, 0.0, 0.0), (s, 0.0, 0.0), (s, s, 0.0), (0.0, s, 0.0),
           (0.0, 0.0, s), (s, 0.0, s), (s, s, s), (0.0, s, s)]
    tris = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (2, 3, 7), (2, 7, 6),
        (0, 4, 7), (0, 7, 3),
        (1, 2, 6), (1, 6, 5),
    ]
    if drop_faces:
        tris = tris[:len(tris) - drop_faces]
    if unwelded:
        pts2, tris2 = [], []
        for tri in tris:
            base = len(pts2)
            pts2.extend(pts[i] for i in tri)
            tris2.append((base, base + 1, base + 2))
        pts, tris = pts2, tris2
    counts = [3] * len(tris)
    indices = [i for tri in tris for i in tri]
    return pts, counts, indices


def _self_test() -> int:
    checked = 0

    pts, counts, idx = _cube()
    uses = edge_use_counts(counts, idx)
    assert len(uses) == 18, f"a closed cube has 18 edges, got {len(uses)}"
    assert set(uses.values()) == {2}, "every edge of a closed cube is used twice"
    assert unpaired_edges(counts, idx) == 0
    assert split_unpaired(counts, idx) == {"boundary": 0, "non_manifold": 0, "total": 0}
    checked += 1

    pts, counts, idx = _cube(drop_faces=1)
    assert unpaired_edges(counts, idx) == 3
    assert split_unpaired(counts, idx) == {"boundary": 3, "non_manifold": 0, "total": 3}
    checked += 1

    pts, counts, idx = _cube()
    idx2 = list(idx) + [0, 2, 1]
    counts2 = list(counts) + [3]
    assert split_unpaired(counts2, idx2) == {"boundary": 0, "non_manifold": 3, "total": 3}
    assert unpaired_edges(counts2, idx2) == 3, "unpaired_edges counts c != 2, both kinds"
    checked += 1

    pts, counts, idx = _cube(unwelded=True)
    assert len(pts) == 36
    assert unpaired_edges(counts, idx) == 36
    n_open, n_unique = welded_unpaired_edges(pts, counts, idx, tol=1e-6)
    assert n_open == 0, f"a position weld must close the indexing artefact, got {n_open}"
    assert n_unique == 8, f"the cube has 8 distinct corners, weld found {n_unique}"
    checked += 1

    for scale in (1.0, 1000.0):
        pts, counts, idx = _cube(scale=scale, unwelded=True)
        diag = (3.0 * scale * scale) ** 0.5
        n_open, n_unique = welded_unpaired_edges(pts, counts, idx, tol=1e-6 * diag)
        assert n_open == 0, f"scale {scale}: fractional tolerance must still close it"
        assert n_unique == 8, f"scale {scale}: got {n_unique} unique points"
    checked += 1

    pts, counts, idx = _cube(unwelded=True, drop_faces=1)
    assert welded_unpaired_edges(pts, counts, idx, tol=1e-6)[0] == 3
    checked += 1

    pts, counts, idx = _cube(drop_faces=1)
    surv = survivor_edges(pts, counts, idx)
    assert len(surv) == 3 and all(s["kind"] == "boundary" for s in surv)
    assert all(abs(s["a"][0] - 1.0) < 1e-12 and abs(s["b"][0] - 1.0) < 1e-12 for s in surv), \
        f"survivor endpoints are not on the x = 1 face: {surv}"
    checked += 1

    pts, counts, idx = _cube(unwelded=True, drop_faces=1)
    surv_w = survivor_edges(pts, counts, idx, tol=1e-6)
    assert len(surv_w) == 3
    assert all(abs(s["a"][0] - 1.0) < 1e-12 and abs(s["b"][0] - 1.0) < 1e-12 for s in surv_w)
    checked += 1

    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    assert point_box_distance((0.5, 0.5, 0.5), lo, hi) == 0.0
    assert point_box_distance((1.0, 0.5, 0.5), lo, hi) == 0.0, "on the face is touching"
    assert abs(point_box_distance((3.0, 0.5, 0.5), lo, hi) - 2.0) < 1e-12
    assert abs(point_box_distance((4.0, 5.0, 1.0), lo, hi) - 5.0) < 1e-12, "3-4-5 corner"
    checked += 1

    assert segment_box_distance((-2.0, 0.5, 0.5), (2.0, 0.5, 0.5), lo, hi) == 0.0
    assert abs(segment_box_distance((-2.0, 0.5, 3.0), (2.0, 0.5, 3.0), lo, hi) - 2.0) < 1e-9
    d_mid = segment_box_distance((-3.0, 0.5, 3.0), (5.0, 0.5, 3.0), lo, hi)
    assert abs(d_mid - 2.0) < 1e-9, f"midpoint case gave {d_mid}"
    d_ends = min(point_box_distance((-3.0, 0.5, 3.0), lo, hi),
                 point_box_distance((5.0, 0.5, 3.0), lo, hi))
    assert d_ends > d_mid + 1.0, "the endpoint-only shortcut must be visibly wrong here"
    assert abs(segment_box_distance((4.0, 5.0, 1.0), (4.0, 5.0, 1.0), lo, hi) - 5.0) < 1e-9
    checked += 1

    pts, _, _ = _cube()
    assert plane_hits(pts, 0, 0.0, 1e-12) == 4
    assert plane_hits(pts, 0, 1.0, 1e-12) == 4
    assert plane_hits(pts, 0, 0.5, 1e-12) == 0, "the frame guard must be able to say ZERO"
    assert plane_hits(pts, 2, 1.0, 1e-12) == 4
    checked += 1

    try:
        edge_use_counts([3, 3], [0, 1, 2])
    except ValueError as exc:
        assert "do not describe the same mesh" in str(exc)
    else:  # pragma: no cover - the assert above is the test
        raise AssertionError("a truncated index array was accepted")
    try:
        weld_remap([(0.0, 0.0, 0.0)], 0.0)
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a zero weld tolerance was accepted")
    checked += 1

    assert checked == 12, f"only {checked} of 12 cases ran"
    print(f"[mesh_topology] self-test: {checked}/12 cases passed ({MESH_TOPOLOGY_MARKER})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Triangle-mesh topology and defect location."
    )
    parser.add_argument("--self-test", action="store_true",
                        help="Prove the topology and the distances. No USD, no Isaac.")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
