# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""The pure half of the asset-mesh export (D-080).

WHY THIS EXISTS
---------------
``insertion_sdf.py`` needs the part and the fixture as OBJ files, because that
is what the published pattern consumes: Isaac Lab 2.3.2
``isaaclab_tasks/direct/automate/industreal_algo_utils.py::load_asset_mesh_in_warp``
loads an OBJ with ``trimesh``, hands its vertices and faces straight to
``wp.Mesh`` and samples the surface. It never reads the USD stage.

The USDs live only on the training PC, so the extraction has to run there. What
does NOT have to run there is the arithmetic: transforming points into another
frame, merging several mesh prims into one, and writing the OBJ text. Every one
of those is where a defect would be silent -- a frame composed the wrong way
round produces a perfectly valid OBJ of a part in the wrong place -- so they
live here, in stdlib Python, and are proved on the laptop.

The split is the one ``usd_predicates.py`` already uses in this repo: the
collecting half touches ``pxr``, the deciding half does not. ``pxr`` is not
imported anywhere in this file.

THE SELF-TEST CONTRACT, written before the implementation
---------------------------------------------------------
``python scripts/tools/mesh_export.py --self-test`` must prove:

  1. a point is transformed by a known rotation, not by its transpose;
  2. a translation reaches the points, i.e. the 4th column is not dropped;
  3. transforming by a matrix and then by its inverse returns the input;
  4. merging two meshes offsets the second one's indices by the first one's
     vertex count, and by nothing else;
  5. merging preserves the face order and the total face count;
  6. a face with four corners is REFUSED, with its own count in the message;
  7. an empty mesh is refused rather than silently written;
  8. the bounding box reads the extent per axis, not the largest coordinate;
  9. the OBJ text is 1-BASED, the one off-by-one that makes a valid file of a
     broken mesh;
 10. the OBJ round-trips: parsing the text back yields the input points and
     faces.

NOT ITS JOB
-----------
Choosing the number of SDF sample points (that belongs to ``insertion_sdf.py``),
deciding what to do about a non-watertight surface (D-109 (10c), scene stream),
and anything that writes to a USD.
"""

from __future__ import annotations

import argparse

MESH_EXPORT_MARKER = "mesh_export-2026-08-29a"


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def transform_points(points: list, matrix: list) -> list:
    """Apply a 4x4 ROW-major matrix to ``[(x, y, z), ...]``. Returns a new list.

    ``matrix[r][c]`` multiplies as ``out_r = sum_c matrix[r][c] * p_c``, with
    ``p_3 = 1`` -- the ordinary column-vector convention. USD's ``Gf.Matrix4d``
    is ROW-vector (``p * M``), so the caller transposes on the way in; that
    transpose is stated at the call site rather than hidden here, because a
    convention applied twice is the failure this function cannot see.

    The homogeneous divide is NOT performed. Every matrix this repo hands over
    is a rigid transform with a last row of ``(0, 0, 0, 1)``, and a projective
    matrix would be a defect upstream, not something to quietly normalise away.
    """
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError(f"matrix must be 4x4, got {len(matrix)} rows")
    out = []
    for p in points:
        if len(p) != 3:
            raise ValueError(f"point {p!r} is not a 3-vector")
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        out.append((
            matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
            matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
        ))
    return out


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------
def merge_meshes(parts: list) -> tuple:
    """Concatenate ``[(points, counts, indices), ...]`` into one mesh.

    Returns ``(points, counts, indices)``. The n-th part's indices are shifted
    by the number of vertices in the parts before it -- the whole reason this
    is a function and not a pair of ``extend`` calls at the call site. An asset
    that arrives as several ``UsdGeom.Mesh`` prims (a fixture block plus its
    stages, say) is one surface for the SDF, and concatenating the vertex lists
    without shifting the indices would silently fold every later part onto the
    first one's triangles.

    Order is preserved, so a face's position in the merged list still names
    which part it came from.
    """
    points: list = []
    counts: list = []
    indices: list = []
    for k, part in enumerate(parts):
        if len(part) != 3:
            raise ValueError(f"part {k} is not a (points, counts, indices) triple")
        p, c, i = part
        offset = len(points)
        if i and max(i) >= len(p):
            raise ValueError(
                f"part {k} indexes vertex {max(i)} but carries only {len(p)} vertices"
            )
        points.extend(p)
        counts.extend(c)
        indices.extend(idx + offset for idx in i)
    return points, counts, indices


def require_triangles(counts: list) -> int:
    """Return the triangle count, or raise with the histogram of corner counts.

    NOTHING IS TRIANGULATED HERE, and that is deliberate. Fanning a quad is
    correct for a convex one and wrong for a concave one, so it is a decision
    about the asset, not a reformatting step -- and this file has no way to
    tell which kind arrived. A run that meets a quad therefore stops and prints
    what it met, which turns one training-PC run into the answer instead of
    into a silently fanned surface nobody checks again.
    """
    if not counts:
        raise ValueError("the mesh has no faces; there is nothing to export")
    hist: dict = {}
    for c in counts:
        hist[int(c)] = hist.get(int(c), 0) + 1
    if set(hist) != {3}:
        rows = ", ".join(f"{k} corners: {v} faces" for k, v in sorted(hist.items()))
        raise ValueError(
            "every face must be a triangle for a Warp mesh, but the asset carries "
            f"{rows}. Nothing was written. Re-export the asset triangulated, or "
            "decide here how a face of that size is split -- that is a decision "
            "about the geometry, not a reformatting step."
        )
    return hist[3]


def faces_from_flat(counts: list, indices: list) -> list:
    """Cut the flat index list into per-face tuples using ``counts``."""
    faces = []
    at = 0
    for c in counts:
        c = int(c)
        if at + c > len(indices):
            raise ValueError(
                f"faceVertexCounts asks for {at + c} indices but only "
                f"{len(indices)} are present"
            )
        faces.append(tuple(indices[at:at + c]))
        at += c
    if at != len(indices):
        raise ValueError(
            f"{len(indices) - at} index entries are left over after the last face; "
            "counts and indices disagree"
        )
    return faces


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def bbox(points: list) -> dict:
    """Axis-aligned bounds. ``{"min": (..), "max": (..), "extent": (..)}``.

    The EXTENT is the span per axis, ``max - min`` -- not the largest
    coordinate. The cross-check this feeds compares against ``PART_BBOX_M``,
    which is a span, and a part whose origin is not at its centre would pass a
    largest-coordinate test by accident.
    """
    if not points:
        raise ValueError("no points; there is no bounding box")
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return {
        "min": tuple(lo),
        "max": tuple(hi),
        "extent": tuple(hi[i] - lo[i] for i in range(3)),
    }


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------
def obj_text(points: list, faces: list, header: str = "") -> str:
    """Wavefront OBJ text: ``v`` lines then ``f`` lines, indices 1-BASED.

    OBJ counts vertices from 1. USD counts from 0. Writing the USD index
    straight into an ``f`` line produces a FILE THAT LOADS -- trimesh accepts
    it, Warp builds a BVH from it -- and every triangle references the wrong
    vertex. There is no error to see, only a mesh that is subtly not the part.
    That is the reason this function exists instead of an f-string at the call
    site, and case 9 of the self-test is that off-by-one.

    Only ``v`` and ``f`` are written. No normals, no texture coordinates, no
    groups: ``load_asset_mesh_in_warp`` reads ``.vertices`` and ``.faces`` and
    nothing else, and anything further would be unverified decoration.
    """
    lines = []
    for line in header.splitlines():
        lines.append(f"# {line}" if line else "#")
    for p in points:
        lines.append(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}")
    for f in faces:
        if not f:
            raise ValueError("a face with no corners cannot be written")
        lines.append("f " + " ".join(str(int(i) + 1) for i in f))
    return "\n".join(lines) + "\n"


def parse_obj(text: str) -> tuple:
    """Read back ``v`` and ``f`` lines. Used by the self-test's round trip.

    Deliberately minimal and deliberately NOT the writer's inverse by
    construction: it re-derives the 1-based convention rather than sharing a
    constant with ``obj_text``, so a round trip proves the convention instead
    of proving that one symbol equals itself.
    """
    points: list = []
    faces: list = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") or not line:
            continue
        head, _, rest = line.partition(" ")
        if head == "v":
            xs = rest.split()
            if len(xs) != 3:
                raise ValueError(f"vertex line has {len(xs)} components: {raw!r}")
            points.append(tuple(float(x) for x in xs))
        elif head == "f":
            faces.append(tuple(int(tok.split("/")[0]) - 1 for tok in rest.split()))
    return points, faces


# ---------------------------------------------------------------------------
# The counter-proof
# ---------------------------------------------------------------------------
def _close(a, b, tol=1e-12) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _self_test() -> int:
    checked = 0

    # 90 deg about z, ROW-major column-vector convention: x -> y, y -> -x.
    rot_z = [[0.0, -1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]

    # 1 -- the rotation, not its transpose. The transpose sends x -> -y, so
    # this single point separates them.
    out = transform_points([(1.0, 0.0, 0.0)], rot_z)
    assert _close(out[0], (0.0, 1.0, 0.0)), out
    checked += 1

    # 2 -- the translation column reaches the point.
    trans = [[1.0, 0.0, 0.0, 2.0],
             [0.0, 1.0, 0.0, -3.0],
             [0.0, 0.0, 1.0, 0.5],
             [0.0, 0.0, 0.0, 1.0]]
    assert _close(transform_points([(1.0, 1.0, 1.0)], trans)[0], (3.0, -2.0, 1.5))
    checked += 1

    # 3 -- matrix then inverse is the identity. The inverse of rot_z is
    # rot_z transposed (pure rotation), written out rather than computed.
    inv_z = [[0.0, 1.0, 0.0, 0.0],
             [-1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
    p = [(0.3, -1.7, 4.2)]
    assert _close(transform_points(transform_points(p, rot_z), inv_z)[0], p[0])
    checked += 1

    # 4 -- merging offsets the SECOND part's indices by the FIRST part's
    # vertex count. Three vertices in part A, so part B's 0,1,2 become 3,4,5.
    a = ([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [3], [0, 1, 2])
    b = ([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)], [3], [0, 1, 2])
    mp, mc, mi = merge_meshes([a, b])
    assert len(mp) == 6, mp
    assert mi == [0, 1, 2, 3, 4, 5], mi
    checked += 1

    # 5 -- face order and face count survive the merge.
    assert mc == [3, 3], mc
    checked += 1

    # 6 -- a quad is refused, and the message names its corner count.
    try:
        require_triangles([3, 4, 3])
    except ValueError as exc:
        assert "4 corners" in str(exc), str(exc)
    else:
        raise AssertionError("a quad must be refused, not fanned")
    checked += 1

    # 7 -- an empty mesh is refused rather than written as an empty file.
    try:
        require_triangles([])
    except ValueError as exc:
        assert "no faces" in str(exc), str(exc)
    else:
        raise AssertionError("an empty mesh must be refused")
    checked += 1

    # 8 -- the bbox reports the SPAN, not the largest coordinate. These points
    # sit between 10 and 11 on x, so the extent is 1.0 and the max is 11.0.
    bb = bbox([(10.0, 0.0, 0.0), (11.0, 2.0, -1.0)])
    assert _close(bb["extent"], (1.0, 2.0, 1.0)), bb
    assert _close(bb["max"], (11.0, 2.0, 0.0)), bb
    checked += 1

    # 9 -- OBJ faces are 1-BASED. Vertex 0 must be written as 1.
    text = obj_text([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1, 2)])
    assert "f 1 2 3" in text, text
    assert "f 0 1 2" not in text, text
    checked += 1

    # 10 -- the round trip. Written, parsed back, identical.
    pts = [(0.001, -0.002, 0.003), (1.5, 0.0, 0.0), (0.0, 2.25, 0.0)]
    faces = [(0, 1, 2), (2, 1, 0)]
    rp, rf = parse_obj(obj_text(pts, faces, header="round trip"))
    assert len(rp) == 3 and all(_close(x, y, 1e-9) for x, y in zip(rp, pts)), rp
    assert rf == faces, rf
    checked += 1

    # A bonus that is not a numbered case but would be a silent defect: the
    # flat index list must cut back into exactly the faces it came from.
    assert faces_from_flat(mc, mi) == [(0, 1, 2), (3, 4, 5)]

    assert checked == 10, f"only {checked} of 10 cases ran"
    print(f"[mesh_export] self-test: {checked}/10 cases passed ({MESH_EXPORT_MARKER})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure half of the asset-mesh export (D-080).")
    parser.add_argument("--self-test", action="store_true",
                        help="Prove the arithmetic. Needs no USD and no Isaac.")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
