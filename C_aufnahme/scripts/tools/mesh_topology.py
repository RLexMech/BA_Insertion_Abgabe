# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Triangle-mesh topology and where a defect SITS. Stdlib only (D-080).

WHY THIS EXISTS
---------------
Two questions about the fixture mesh are open, and only the second one is new:

  1. How many edges are NOT used by exactly two faces? RT-56 / RT-60..RT-64
     answered that for the USD.
  2. WHERE do the survivors sit? A count is not a criterion. If they lie on the
     outer rim, no sample point of a seated part ever comes near them; if they
     lie in the pocket wall, SAPU reads its sign exactly there.

The counting half already existed, welded to USD: ``add_fixture_collision.py``
imports ``isaaclab.app.AppLauncher`` and calls ``parser.parse_args()`` at module
scope, so reaching one pure function there costs a whole ``SimulationApp``. That
is the RT-88 lesson a second time. The rule for this repo is ONE HOME PER FACT,
so the edge-pairing rule, the position weld and the survivor localisation move
here and ``add_fixture_collision.py`` delegates.

WHAT IS NEW HERE AND WHAT IS NOT
--------------------------------
NOT new -- lifted unchanged, so RT-60..RT-64 stay comparable:

* the edge key ``(a, b) if a < b else (b, a)`` and the ``c != 2`` rule
  (``add_fixture_collision.unpaired_edges``);
* the position weld by ROUNDING, not by clustering. A cluster search merges
  chains of near points transitively and can weld across a real gap, which is
  the very thing being tested.

NEW, and named as ours because no source in this repo has it:

* ``segment_box_distance`` -- how far an edge is from the volume the part
  sweeps. It is the criterion question 2 needs and the repo had no
  point-to-box or segment-to-box helper at all.
* ``plane_hits`` -- the FRAME GUARD. ``verify_fixture_usd.py`` reads dense
  coordinate planes off the USD to prove the cavity survived the import; the
  same method has to run on the OBJ, because an address in the wrong frame is
  worse than no address. That file's version is inline and pxr-bound
  (``verify_fixture_usd.py:354-395``), so the method is copied, not the code.

NO UNITS ANYWHERE
-----------------
Every length in and out is in the caller's own units. ``weld_remap`` takes an
absolute tolerance and the caller derives it from the mesh's own bounding-box
diagonal -- the correction RT-61's unit defect earned, where absolute metre
tolerances silently reached 0.1 um on a millimetre mesh. The self-test builds
the SAME mesh at scale 1 and at scale 1000 and requires the identical verdict.

    python scripts/tools/mesh_topology.py --self-test
"""

from __future__ import annotations

import argparse

MESH_TOPOLOGY_MARKER = "mesh_topology-2026-08-30a"


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------
def edge_use_counts(counts, indices) -> dict:
    """``{(a, b): times used}`` over a flat face array, paired by point INDEX.

    ``counts[i]`` is face i's corner count and ``indices`` is the flat corner
    list, the two arrays USD stores and the two ``faces_from_flat`` splits. The
    key is the sorted index pair, so the two half-edges of an interior edge
    land on the same key regardless of winding.

    Pairing by INDEX, not by position, is the whole point: two coincident
    points with different indices leave both their edges unpaired, and telling
    that apart from a real crack is what ``weld_remap`` is for.
    """
    # Checked BEFORE the walk, not after. A short index array otherwise makes
    # the last face silently smaller and the count comes back plausible.
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
    """Edges NOT used by exactly two faces. 0 = closed (watertight) topology.

    NOTE the name, and it is the exact wording ``add_fixture_collision.py``
    settled on: this counts ``c != 2``, so it includes both BOUNDARY edges (one
    face) and NON-MANIFOLD edges (three or more). A reported count of n is
    "n edges not used by exactly two faces". ``split_unpaired`` measures the
    split; this function deliberately does not, because every RT number from
    RT-56 on quotes this figure and it must keep meaning the same thing.
    """
    return sum(1 for c in edge_use_counts(counts, indices).values() if c != 2)


def split_unpaired(counts, indices) -> dict:
    """``{"boundary": n, "non_manifold": n, "total": n}`` -- the split, measured.

    The two kinds need opposite readings. A boundary edge is a hole: the
    surface stops there and the inside/outside question has no answer at that
    edge. A non-manifold edge is three or more faces meeting, which is a
    modelling defect but leaves the surface closed.
    """
    counts_by_use = edge_use_counts(counts, indices)
    bnd = sum(1 for c in counts_by_use.values() if c == 1)
    nm = sum(1 for c in counts_by_use.values() if c >= 3)
    return {"boundary": bnd, "non_manifold": nm, "total": bnd + nm}


# ---------------------------------------------------------------------------
# The position weld
# ---------------------------------------------------------------------------
def weld_remap(points, tol: float) -> tuple:
    """Group points by ROUNDED position. Returns ``(remap, representative)``.

    ``remap[i]`` is the merged index of input point i; ``representative[m]`` is
    the index of the FIRST input point that landed in merged group m, so a
    survivor edge can be printed with a real coordinate rather than an average
    nobody measured.

    Rounding, not clustering, and the reason is in the module docstring. The
    key is ``round(p / tol)`` per axis, which is what RT-60..RT-64 swept with.

    ``tol`` must be positive. A tolerance of zero would make every point its own
    group in the best case and divide by zero in fact, and "no weld at all" is
    already expressed by not calling this function.
    """
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
    """Every edge not used by exactly two faces, WITH its two endpoints.

    ``tol=None`` reads the mesh AS AUTHORED -- which is what Warp receives:
    ``wp.Mesh`` is built from ``trimesh.load(..., process=False)`` and nothing
    welds in between. With a tolerance, the points are welded first and the
    endpoints come from each group's representative.

    Each entry is ``{"uses", "kind", "a", "b"}`` with ``kind`` in
    ``{"boundary", "non-manifold"}``. Order is the edge dictionary's insertion
    order, i.e. face order -- deterministic, so two runs of the same mesh print
    the same list.
    """
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


# ---------------------------------------------------------------------------
# Where a defect sits
# ---------------------------------------------------------------------------
def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to an axis-aligned box. 0.0 inside or on it.

    The standard clamp form: per axis take how far the point is outside on
    either side, zero if it is between, then the norm. Points INSIDE report
    0.0, not a negative depth -- this answers "does it touch the volume", and a
    signed depth would need a second convention nobody here needs.
    """
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
    """Distance from the SEGMENT a--b to an axis-aligned box. 0.0 if it touches.

    ``d(t) = point_box_distance(a + t*(b - a), lo, hi)`` is CONVEX on
    ``t in [0, 1]``: each axis term is the max of two affine functions and
    zero, hence convex and non-negative, and the Euclidean norm of a vector of
    convex non-negative functions is convex. A convex function on an interval
    has one minimum, so a ternary search finds it -- no sampling grid anybody
    had to choose, and no closed form to get subtly wrong.

    ``iterations`` halves the bracket each round; the default leaves a bracket
    far below double precision, so the result is the minimum to machine
    accuracy. It is an argument because a caller with a million edges may want
    to say so out loud rather than discover the cost.

    An EDGE, not a face: a boundary edge is a curve, and asking whether that
    curve enters the volume the part sweeps is exactly the RT-91 question.
    """
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
    """How many points lie within ``tol`` of the plane ``axis = value``.

    THE FRAME GUARD. A coordinate is only an address if the mesh is in the
    frame the address is written in. The fixture is prismatic, so every face
    puts many vertices on one coordinate and the pocket walls are heavily
    populated planes -- the method ``verify_fixture_usd.py`` uses on the USD
    (``--planes``), applied to the OBJ. Zero hits on an expected plane means
    the OBJ is not in the frame the pocket constants describe, and every
    distance computed against them would be a fluent, wrong number.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")
    if tol < 0.0:
        raise ValueError(f"tolerance must not be negative, got {tol!r}")
    return sum(1 for p in points if abs(float(p[axis]) - float(value)) <= tol)


# ---------------------------------------------------------------------------
# The counter-proof
# ---------------------------------------------------------------------------
def _cube(scale: float = 1.0, unwelded: bool = False, drop_faces: int = 0) -> tuple:
    """A closed axis-aligned cube of side ``scale``, as ``(points, counts, indices)``.

    ``unwelded`` gives every triangle its own three points, which is case (a) of
    the open-mesh question: geometrically closed, topologically shredded.
    ``drop_faces`` removes that many triangles from the end, which is case (b):
    a real hole.
    """
    s = scale
    pts = [(0.0, 0.0, 0.0), (s, 0.0, 0.0), (s, s, 0.0), (0.0, s, 0.0),
           (0.0, 0.0, s), (s, 0.0, s), (s, s, s), (0.0, s, s)]
    tris = [
        (0, 2, 1), (0, 3, 2),      # z = 0
        (4, 5, 6), (4, 6, 7),      # z = s
        (0, 1, 5), (0, 5, 4),      # y = 0
        (2, 3, 7), (2, 7, 6),      # y = s
        (0, 4, 7), (0, 7, 3),      # x = 0
        (1, 2, 6), (1, 6, 5),      # x = s
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

    # 1 -- a closed cube has no unpaired edge, and 12 triangles carry 18 edges.
    pts, counts, idx = _cube()
    uses = edge_use_counts(counts, idx)
    assert len(uses) == 18, f"a closed cube has 18 edges, got {len(uses)}"
    assert set(uses.values()) == {2}, "every edge of a closed cube is used twice"
    assert unpaired_edges(counts, idx) == 0
    assert split_unpaired(counts, idx) == {"boundary": 0, "non_manifold": 0, "total": 0}
    checked += 1

    # 2 -- removing one triangle opens exactly its three edges, and they are
    # BOUNDARY, not non-manifold. The split has to tell them apart.
    pts, counts, idx = _cube(drop_faces=1)
    assert unpaired_edges(counts, idx) == 3
    assert split_unpaired(counts, idx) == {"boundary": 3, "non_manifold": 0, "total": 3}
    checked += 1

    # 3 -- a NON-MANIFOLD edge is counted as one, not as a hole. Duplicating a
    # face puts three uses on each of its edges.
    pts, counts, idx = _cube()
    idx2 = list(idx) + [0, 2, 1]
    counts2 = list(counts) + [3]
    assert split_unpaired(counts2, idx2) == {"boundary": 0, "non_manifold": 3, "total": 3}
    assert unpaired_edges(counts2, idx2) == 3, "unpaired_edges counts c != 2, both kinds"
    checked += 1

    # 4 -- the INDEXING artefact: the same closed cube, every triangle with its
    # own points. 36 points, 36 unpaired edges as authored, 0 after a weld.
    # This is the case Warp receives if the OBJ is unwelded, so it is not a
    # hypothetical.
    pts, counts, idx = _cube(unwelded=True)
    assert len(pts) == 36
    assert unpaired_edges(counts, idx) == 36
    n_open, n_unique = welded_unpaired_edges(pts, counts, idx, tol=1e-6)
    assert n_open == 0, f"a position weld must close the indexing artefact, got {n_open}"
    assert n_unique == 8, f"the cube has 8 distinct corners, weld found {n_unique}"
    checked += 1

    # 5 -- THE UNIT CONTROL, and it is the one RT-61's defect earned. The same
    # mesh at scale 1 and at scale 1000 must give the SAME verdict when the
    # tolerance is a fraction of the mesh's own size. An absolute tolerance
    # does not, which is exactly how the old sweep claimed 0.1 mm and delivered
    # 0.1 um.
    for scale in (1.0, 1000.0):
        pts, counts, idx = _cube(scale=scale, unwelded=True)
        diag = (3.0 * scale * scale) ** 0.5
        n_open, n_unique = welded_unpaired_edges(pts, counts, idx, tol=1e-6 * diag)
        assert n_open == 0, f"scale {scale}: fractional tolerance must still close it"
        assert n_unique == 8, f"scale {scale}: got {n_unique} unique points"
    # and the counter-case: an ABSOLUTE tolerance that works at scale 1 is
    # 1000x too small at scale 1000 only if the defect is at that size. The
    # cube's duplicates are EXACT, so no tolerance separates them -- the honest
    # control is that the two scales agree, not that the absolute one fails.
    checked += 1

    # 6 -- a real crack survives every weld. Welding cannot invent a triangle.
    pts, counts, idx = _cube(unwelded=True, drop_faces=1)
    assert welded_unpaired_edges(pts, counts, idx, tol=1e-6)[0] == 3
    checked += 1

    # 7 -- survivor_edges AS AUTHORED reports real coordinates, and the three
    # edges of the dropped triangle are the ones that come back.
    pts, counts, idx = _cube(drop_faces=1)
    surv = survivor_edges(pts, counts, idx)
    assert len(surv) == 3 and all(s["kind"] == "boundary" for s in surv)
    # the dropped triangle was (1, 6, 5) -> corners (s,0,0), (s,s,s), (s,0,s):
    # every survivor endpoint sits on the x = s face.
    assert all(abs(s["a"][0] - 1.0) < 1e-12 and abs(s["b"][0] - 1.0) < 1e-12 for s in surv), \
        f"survivor endpoints are not on the x = 1 face: {surv}"
    checked += 1

    # 8 -- survivor_edges WITH a weld reports the representative's coordinate,
    # not a merged average. Same cube unwelded and cracked: the addresses must
    # match case 7's.
    pts, counts, idx = _cube(unwelded=True, drop_faces=1)
    surv_w = survivor_edges(pts, counts, idx, tol=1e-6)
    assert len(surv_w) == 3
    assert all(abs(s["a"][0] - 1.0) < 1e-12 and abs(s["b"][0] - 1.0) < 1e-12 for s in surv_w)
    checked += 1

    # 9 -- point_box_distance: inside is 0, a face offset is the offset, and a
    # corner offset is the DIAGONAL. The corner case is the one a per-axis max
    # would get wrong.
    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    assert point_box_distance((0.5, 0.5, 0.5), lo, hi) == 0.0
    assert point_box_distance((1.0, 0.5, 0.5), lo, hi) == 0.0, "on the face is touching"
    assert abs(point_box_distance((3.0, 0.5, 0.5), lo, hi) - 2.0) < 1e-12
    assert abs(point_box_distance((4.0, 5.0, 1.0), lo, hi) - 5.0) < 1e-12, "3-4-5 corner"
    checked += 1

    # 10 -- segment_box_distance. Four cases with an answer known by hand:
    #   crossing the box -> 0; parallel and offset -> the offset; a segment
    #   whose ENDS are both far but whose middle is near -> the middle's
    #   distance, which is the case a two-endpoint test gets wrong; and a
    #   degenerate segment (a == b) -> the point distance.
    assert segment_box_distance((-2.0, 0.5, 0.5), (2.0, 0.5, 0.5), lo, hi) == 0.0
    assert abs(segment_box_distance((-2.0, 0.5, 3.0), (2.0, 0.5, 3.0), lo, hi) - 2.0) < 1e-9
    d_mid = segment_box_distance((-3.0, 0.5, 3.0), (5.0, 0.5, 3.0), lo, hi)
    assert abs(d_mid - 2.0) < 1e-9, f"midpoint case gave {d_mid}"
    d_ends = min(point_box_distance((-3.0, 0.5, 3.0), lo, hi),
                 point_box_distance((5.0, 0.5, 3.0), lo, hi))
    assert d_ends > d_mid + 1.0, "the endpoint-only shortcut must be visibly wrong here"
    assert abs(segment_box_distance((4.0, 5.0, 1.0), (4.0, 5.0, 1.0), lo, hi) - 5.0) < 1e-9
    checked += 1

    # 11 -- plane_hits counts the plane, not the axis. The cube puts 4 corners
    # on x = 0 and 4 on x = 1, and nothing on x = 0.5.
    pts, _, _ = _cube()
    assert plane_hits(pts, 0, 0.0, 1e-12) == 4
    assert plane_hits(pts, 0, 1.0, 1e-12) == 4
    assert plane_hits(pts, 0, 0.5, 1e-12) == 0, "the frame guard must be able to say ZERO"
    assert plane_hits(pts, 2, 1.0, 1e-12) == 4
    checked += 1

    # 12 -- the arrays must describe the same mesh. A truncated index array is
    # a defect that would otherwise produce a plausible, wrong count.
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
        description="Triangle-mesh topology and defect location (D-080)."
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
