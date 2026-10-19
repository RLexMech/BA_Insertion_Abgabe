# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""The pure half of the asset-mesh export."""

from __future__ import annotations

import argparse

MESH_EXPORT_MARKER = "mesh_export-2026-08-29a"


def transform_points(points: list, matrix: list) -> list:
    """Apply a 4x4 ROW-major matrix to ``[(x, y, z), ...]``."""
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


def merge_meshes(parts: list) -> tuple:
    """Concatenate ``[(points, counts, indices), ...]`` into one mesh."""
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
    """Return the triangle count, or raise with the histogram of corner counts."""
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
            "decide here how a face of that size is split -- that is a design choice "
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


def bbox(points: list) -> dict:
    """Axis-aligned bounds."""
    if not points:
        raise ValueError("no points; there is no bounding box")
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return {
        "min": tuple(lo),
        "max": tuple(hi),
        "extent": tuple(hi[i] - lo[i] for i in range(3)),
    }


def obj_text(points: list, faces: list, header: str = "") -> str:
    """Wavefront OBJ text: ``v`` lines then ``f`` lines, indices 1-BASED."""
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
    """Read back ``v`` and ``f`` lines."""
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


def _close(a, b, tol=1e-12) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _self_test() -> int:
    checked = 0

    rot_z = [[0.0, -1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]

    out = transform_points([(1.0, 0.0, 0.0)], rot_z)
    assert _close(out[0], (0.0, 1.0, 0.0)), out
    checked += 1

    trans = [[1.0, 0.0, 0.0, 2.0],
             [0.0, 1.0, 0.0, -3.0],
             [0.0, 0.0, 1.0, 0.5],
             [0.0, 0.0, 0.0, 1.0]]
    assert _close(transform_points([(1.0, 1.0, 1.0)], trans)[0], (3.0, -2.0, 1.5))
    checked += 1

    inv_z = [[0.0, 1.0, 0.0, 0.0],
             [-1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
    p = [(0.3, -1.7, 4.2)]
    assert _close(transform_points(transform_points(p, rot_z), inv_z)[0], p[0])
    checked += 1

    a = ([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [3], [0, 1, 2])
    b = ([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)], [3], [0, 1, 2])
    mp, mc, mi = merge_meshes([a, b])
    assert len(mp) == 6, mp
    assert mi == [0, 1, 2, 3, 4, 5], mi
    checked += 1

    assert mc == [3, 3], mc
    checked += 1

    try:
        require_triangles([3, 4, 3])
    except ValueError as exc:
        assert "4 corners" in str(exc), str(exc)
    else:
        raise AssertionError("a quad must be refused, not fanned")
    checked += 1

    try:
        require_triangles([])
    except ValueError as exc:
        assert "no faces" in str(exc), str(exc)
    else:
        raise AssertionError("an empty mesh must be refused")
    checked += 1

    bb = bbox([(10.0, 0.0, 0.0), (11.0, 2.0, -1.0)])
    assert _close(bb["extent"], (1.0, 2.0, 1.0)), bb
    assert _close(bb["max"], (11.0, 2.0, 0.0)), bb
    checked += 1

    text = obj_text([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1, 2)])
    assert "f 1 2 3" in text, text
    assert "f 0 1 2" not in text, text
    checked += 1

    pts = [(0.001, -0.002, 0.003), (1.5, 0.0, 0.0), (0.0, 2.25, 0.0)]
    faces = [(0, 1, 2), (2, 1, 0)]
    rp, rf = parse_obj(obj_text(pts, faces, header="round trip"))
    assert len(rp) == 3 and all(_close(x, y, 1e-9) for x, y in zip(rp, pts)), rp
    assert rf == faces, rf
    checked += 1

    assert faces_from_flat(mc, mi) == [(0, 1, 2), (3, 4, 5)]

    assert checked == 10, f"only {checked} of 10 cases ran"
    print(f"[mesh_export] self-test: {checked}/10 cases passed ({MESH_EXPORT_MARKER})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure half of the asset-mesh export.")
    parser.add_argument("--self-test", action="store_true",
                        help="Prove the arithmetic. Needs no USD and no Isaac.")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
