# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Check fixture mesh identity, coordinate frame and open-edge locations.

Reports whether open edges intersect the part sweep. Does not repair the OBJ.
Use --self-test for the offline geometry checks."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import types

SCRIPT_MARKER = "check_fixture_mesh-2026-08-30b-weldedgate"

_HERE = pathlib.Path(__file__).resolve().parent
_TOOLS = _HERE / "tools"
_PKG_DIR = (
    _HERE.parent / "source" / "insertion" / "insertion" / "tasks" / "direct" / "insertion"
)
_CFG_PATH = _PKG_DIR / "insertion_tasks_cfg.py"

EXPECT_VERTICES = 903
EXPECT_FACES = 757

PLANE_TOL_M = 1.0e-4

MAX_PRINTED_SURVIVORS = 24


def _load_tool(stem: str) -> types.ModuleType:
    """Load a module out of ``scripts/tools`` by path."""
    spec = importlib.util.spec_from_file_location(stem, _TOOLS / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod


def load_paths_module() -> types.ModuleType:
    """``insertion_paths`` -- stdlib only, no Isaac, no SimulationApp."""
    spec = importlib.util.spec_from_file_location(
        "_insertion_paths", _PKG_DIR / "insertion_paths.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_cfg() -> types.ModuleType:
    """Read ``insertion_tasks_cfg`` from SOURCE with ``isaaclab`` stubbed."""
    if "isaaclab" not in sys.modules:
        pkg = types.ModuleType("isaaclab")
        utils = types.ModuleType("isaaclab.utils")
        utils.configclass = lambda c: c
        pkg.utils = utils
        sys.modules["isaaclab"] = pkg
        sys.modules["isaaclab.utils"] = utils

    pkg_name = "_fixture_cfg_pkg"
    for name in list(sys.modules):
        if name == pkg_name or name.startswith(pkg_name + "."):
            del sys.modules[name]
    shell = types.ModuleType(pkg_name)
    shell.__path__ = [str(_PKG_DIR)]
    sys.modules[pkg_name] = shell

    src = _CFG_PATH.read_text(encoding="utf-8")
    mod = types.ModuleType(f"{pkg_name}.insertion_tasks_cfg")
    mod.__file__ = str(_CFG_PATH)
    mod.__package__ = pkg_name
    sys.modules[mod.__name__] = mod
    exec(compile(src, str(_CFG_PATH), "exec"), mod.__dict__)  # noqa: S102
    return mod


def expected_planes(cfg) -> list:
    """The seven dense coordinate planes, in METRES, from the cfg."""
    return [
        ("pocket wall -x", 0, -cfg.POCKET_WALL_X),
        ("pocket wall +x", 0, +cfg.POCKET_WALL_X),
        ("pocket wall -y", 1, -cfg.POCKET_WALL_Y),
        ("pocket wall +y", 1, +cfg.POCKET_WALL_Y),
        ("pocket floor", 2, cfg.POCKET_FLOOR_Z),
        ("stage 1 -x", 0, cfg.POCKET_STAGE1_X_RANGE[0]),
        ("stage 1 +x", 0, cfg.POCKET_STAGE1_X_RANGE[1]),
    ]


def frame_guard(points, planes, tol_m: float, population_tol: float = 1e-6) -> tuple:
    """Does the OBJ sit in the frame the pocket constants are written in?"""
    rows = []
    ok = True
    for label, axis, want in planes:
        best_v, best_d = None, None
        for p in points:
            v = float(p[axis])
            d = abs(v - float(want))
            if best_d is None or d < best_d:
                best_v, best_d = v, d
        n = 0 if best_v is None else sum(
            1 for p in points if abs(float(p[axis]) - best_v) <= population_tol
        )
        hit = best_d is not None and best_d <= tol_m
        ok = ok and hit
        rows.append({"feature": label, "axis": "xyz"[axis], "want_m": want,
                     "nearest_m": best_v, "nearest_points": n,
                     "error_m": best_d, "ok": hit})
    return ok, rows


def swept_volume_boxes(cfg) -> list:
    """The two volumes the part passes through, as ``(name, lo, hi)`` in metres."""
    return [
        ("stage 2 (pocket cavity)",
         (-cfg.POCKET_WALL_X, -cfg.POCKET_WALL_Y, cfg.POCKET_FLOOR_Z),
         (+cfg.POCKET_WALL_X, +cfg.POCKET_WALL_Y, 0.0)),
        ("stage 1 (mouth)",
         (cfg.POCKET_STAGE1_X_RANGE[0], cfg.POCKET_STAGE1_Y_RANGE[0], 0.0),
         (cfg.POCKET_STAGE1_X_RANGE[1], cfg.POCKET_STAGE1_Y_RANGE[1], cfg.POCKET_RIM_Z)),
    ]


def edge_clearances(topo, survivors, boxes) -> list:
    """Per survivor edge: its distance to every box."""
    out = []
    for s in survivors:
        dists = [topo.segment_box_distance(s["a"], s["b"], lo, hi) for _, lo, hi in boxes]
        row = dict(s)
        row["distances"] = dists
        row["min_distance"] = min(dists) if dists else float("inf")
        out.append(row)
    out.sort(key=lambda r: r["min_distance"])
    return out


def verdict_of(rows) -> str:
    """``EDGES_OUTSIDE_SWEPT_VOLUME`` only when EVERY survivor clears EVERY box."""
    if not rows:
        return "NO_OPEN_EDGES"
    return ("EDGES_OUTSIDE_SWEPT_VOLUME"
            if all(r["min_distance"] > 0.0 for r in rows)
            else "EDGES_IN_SWEPT_VOLUME")


def run(args) -> int:
    mesh_export = _load_tool("mesh_export")
    mesh_spacing = _load_tool("mesh_spacing")
    topo = _load_tool("mesh_topology")

    obj_path = args.obj if args.obj else load_paths_module().resolve_pocket_obj_path()

    print(f"[check_fixture_mesh] marker: {SCRIPT_MARKER}")
    print(f"[check_fixture_mesh] tool markers: {mesh_export.MESH_EXPORT_MARKER}, "
          f"{mesh_spacing.SPACING_MARKER}, {topo.MESH_TOPOLOGY_MARKER}")
    print(f"[check_fixture_mesh] obj: {obj_path}")

    points, faces = mesh_export.parse_obj(
        pathlib.Path(obj_path).read_text(encoding="utf-8")
    )
    counts = [len(f) for f in faces]
    indices = [i for f in faces for i in f]
    print(f"[check_fixture_mesh] parsed {len(points)} vertices, {len(faces)} faces, "
          f"{len(indices)} face corners")

    g1 = len(points) == args.expect_vertices and len(faces) == args.expect_faces
    print(f"[check_fixture_mesh] G1 mesh identity: {len(points)}/{len(faces)} against "
          f"RT-86's {args.expect_vertices}/{args.expect_faces} -- "
          f"{'PASS' if g1 else 'FAIL'}")

    n_tri = mesh_export.require_triangles(counts)
    print(f"[check_fixture_mesh] all {n_tri} faces are triangles")

    box = mesh_export.bbox(points)
    diag = sum(e * e for e in box["extent"]) ** 0.5
    print(f"[check_fixture_mesh] bbox min ({box['min'][0]:.6f}, {box['min'][1]:.6f}, "
          f"{box['min'][2]:.6f}) max ({box['max'][0]:.6f}, {box['max'][1]:.6f}, "
          f"{box['max'][2]:.6f}) m")
    print(f"[check_fixture_mesh] extent {box['extent'][0]:.6f} x {box['extent'][1]:.6f} "
          f"x {box['extent'][2]:.6f} m, diagonal {diag:.6f} m")

    cfg = load_cfg()
    planes = expected_planes(cfg)
    g2, plane_rows = frame_guard(points, planes, args.plane_tol_m)
    print(f"[check_fixture_mesh] G2 frame guard, tol {args.plane_tol_m * 1000.0:.4f} mm:")
    for r in plane_rows:
        print(f"[check_fixture_mesh]   {r['feature']:<15} {r['axis']} want "
              f"{r['want_m'] * 1000.0:+9.4f} mm  nearest "
              f"{r['nearest_m'] * 1000.0:+9.4f} mm ({r['nearest_points']} pts)  "
              f"error {r['error_m'] * 1000.0:8.4f} mm  "
              f"{'ok' if r['ok'] else 'MISS'}")
    print(f"[check_fixture_mesh] G2 frame guard: {'PASS' if g2 else 'FAIL'}")
    if not g2:
        print("[check_fixture_mesh] *** the OBJ is NOT in the frame the pocket "
              "constants describe. Every distance below would be a wrong number, "
              "so none is printed. ***")
        return 1

    authored = topo.split_unpaired(counts, indices)
    print(f"[check_fixture_mesh] as authored (what Warp receives): "
          f"{authored['total']} unpaired edges "
          f"({authored['boundary']} boundary, {authored['non_manifold']} non-manifold)")
    if len(points) == len(indices):
        print("[check_fixture_mesh] *** points == face corners: NO index is shared "
              "anywhere. The OBJ carries no vertex weld at all. ***")

    spacing = mesh_spacing.min_point_spacing(points)
    welded = None
    top_tol = None
    if spacing is None:
        print("[check_fixture_mesh] fewer than two distinct positions: nothing swept.")
    else:
        series = mesh_spacing.weld_tolerance_series(diag, spacing["d_min"])
        print(f"[check_fixture_mesh] distinct positions {spacing['n_distinct_positions']}, "
              f"{spacing['n_exact_duplicate_points']} exact duplicates dropped")
        print(f"[check_fixture_mesh] closest DISTINCT pair {spacing['d_min']:.6e} m; "
              f"hard weld bound {series['hard_bound_point_units']:.6e} m")
        if not series["fractions"]:
            print("[check_fixture_mesh] *** no tolerance decade fits. Nothing swept. ***")
        else:
            for frac in series["fractions"]:
                tol = frac * diag
                n_open, n_unique = topo.welded_unpaired_edges(points, counts, indices, tol)
                print(f"[check_fixture_mesh]   weld {frac:.0e} of diag = {tol:.3e} m: "
                      f"{n_unique:>7} unique points, {n_open:>6} unpaired edges")
                welded, top_tol = n_open, tol

    boxes = swept_volume_boxes(cfg)
    print("[check_fixture_mesh] swept-volume boxes (rectangular hulls, metres):")
    for name, lo, hi in boxes:
        print(f"[check_fixture_mesh]   {name}: x {lo[0]:+.6f}..{hi[0]:+.6f}  "
              f"y {lo[1]:+.6f}..{hi[1]:+.6f}  z {lo[2]:+.6f}..{hi[2]:+.6f}")

    readings = [("AUTHORED", None)]
    if top_tol is not None:
        readings.append(("WELDED", top_tol))

    gated = {"WELDED"}
    all_ok = True
    zero_edges: list = []
    for label, tol in readings:
        survivors = topo.survivor_edges(points, counts, indices, tol)
        rows = edge_clearances(topo, survivors, boxes)
        v = verdict_of(rows)
        gate = label in gated
        print(f"[check_fixture_mesh] --- {label}: {len(rows)} survivor edges, "
              f"verdict {v} ({'GATED' if gate else 'reported only, see the gate note'})")
        for r in rows[:MAX_PRINTED_SURVIVORS]:
            a, b = r["a"], r["b"]
            ds = "  ".join(f"{d * 1000.0:9.4f}" for d in r["distances"])
            print(f"[check_fixture_mesh]   {label} {r['kind']:<13} used {r['uses']}x  "
                  f"({a[0] * 1000:9.4f},{a[1] * 1000:9.4f},{a[2] * 1000:9.4f}) -> "
                  f"({b[0] * 1000:9.4f},{b[1] * 1000:9.4f},{b[2] * 1000:9.4f}) mm  "
                  f"| dist to boxes [mm]: {ds}")
        if len(rows) > MAX_PRINTED_SURVIVORS:
            print(f"[check_fixture_mesh]   ... {len(rows) - MAX_PRINTED_SURVIVORS} more, "
                  f"all farther than the ones above (sorted closest first)")
        if rows:
            print(f"[check_fixture_mesh]   {label} CLOSEST survivor to any swept volume: "
                  f"{rows[0]['min_distance'] * 1000.0:.4f} mm")
        if gate:
            zero_edges = [r for r in rows if r["min_distance"] <= 0.0]
            all_ok = all_ok and v in ("EDGES_OUTSIDE_SWEPT_VOLUME", "NO_OPEN_EDGES")

    print(f"[check_fixture_mesh] ZERO-EDGE COUNT: {len(zero_edges)} welded survivor(s) "
          f"touch a swept volume")
    for r in zero_edges:
        a, b = r["a"], r["b"]
        print(f"[check_fixture_mesh] ZERO-EDGE  used {r['uses']}x  "
              f"({a[0] * 1000:9.4f},{a[1] * 1000:9.4f},{a[2] * 1000:9.4f}) -> "
              f"({b[0] * 1000:9.4f},{b[1] * 1000:9.4f},{b[2] * 1000:9.4f}) mm  "
              f"| dist to boxes [mm]: "
              + "  ".join(f"{d * 1000.0:9.4f}" for d in r["distances"]))

    if welded is not None:
        print(f"[check_fixture_mesh] cross-check vs RT-61 COUNT: RT-61 found 9 boundary / "
              f"0 non-manifold; this OBJ welds to {welded} unpaired edges. "
              f"{'AGREES' if welded == 9 else 'DISAGREES -- report it, do not explain it away'}")
        print(f"[check_fixture_mesh] cross-check vs RT-61 ADDRESS: RT-61 claimed ALL nine "
              f"lie on the outer rim, clear of the pocket; this run finds "
              f"{len(zero_edges)} welded survivor(s) touching a swept volume. "
              f"{'AGREES' if not zero_edges else 'DISAGREES -- the ZERO-EDGE lines above are the finding'}")

    ok = g1 and g2 and all_ok
    print(f"[check_fixture_mesh] VERDICT: {'OPEN_EDGES_CLEAR_OF_THE_PART' if ok else 'OPEN_EDGES_NOT_CLEAR'}")
    return 0 if ok else 1


class _FakeCfg:
    """The pocket constants only, so the self-test needs no cfg file."""
    POCKET_WALL_X = 1.0
    POCKET_WALL_Y = 2.0
    POCKET_FLOOR_Z = -3.0
    POCKET_STAGE1_X_RANGE = (-4.0, 5.0)
    POCKET_STAGE1_Y_RANGE = (-2.0, 3.0)
    POCKET_RIM_Z = 0.5


def _self_test() -> int:
    topo = _load_tool("mesh_topology")
    checked = 0
    cfg = _FakeCfg()
    boxes = swept_volume_boxes(cfg)
    assert len(boxes) == 2
    assert boxes[0][1] == (-1.0, -2.0, -3.0) and boxes[0][2] == (1.0, 2.0, 0.0)
    assert boxes[1][1] == (-4.0, -2.0, 0.0) and boxes[1][2] == (5.0, 3.0, 0.5)
    checked += 1

    planes = expected_planes(cfg)
    assert len(planes) == 7
    on_frame = [(-1.0, -2.0, -3.0), (1.0, 2.0, 0.0), (-4.0, 0.0, 0.0), (5.0, 0.0, 0.0)]
    ok, rows = frame_guard(on_frame, planes, 1e-4)
    assert ok, [r for r in rows if not r["ok"]]
    assert len(rows) == 7
    checked += 1

    off_frame = [(x + 1e-3, y + 1e-3, z + 1e-3) for x, y, z in on_frame]
    ok_bad, rows_bad = frame_guard(off_frame, planes, 1e-4)
    assert not ok_bad, "the frame guard accepted a mesh shifted by 1 mm"
    assert all(abs(r["error_m"] - 1e-3) < 1e-9 for r in rows_bad), rows_bad
    checked += 1

    ok_small, _ = frame_guard(
        [(x + 1e-5, y + 1e-5, z + 1e-5) for x, y, z in on_frame], planes, 1e-4
    )
    assert ok_small, "the frame guard rejected a 0.01 mm deviation inside its own tolerance"
    checked += 1

    pts, counts, idx = topo._cube(scale=1.0, drop_faces=1)
    far = [(x + 50.0, y + 50.0, z + 50.0) for x, y, z in pts]
    rows = edge_clearances(topo, topo.survivor_edges(far, counts, idx), boxes)
    assert len(rows) == 3, rows
    assert verdict_of(rows) == "EDGES_OUTSIDE_SWEPT_VOLUME"
    assert rows[0]["min_distance"] > 40.0
    checked += 1

    inside = [(x * 0.1, y * 0.1, z * 0.1 - 1.0) for x, y, z in pts]
    rows_in = edge_clearances(topo, topo.survivor_edges(inside, counts, idx), boxes)
    assert verdict_of(rows_in) == "EDGES_IN_SWEPT_VOLUME", rows_in
    assert rows_in[0]["min_distance"] == 0.0
    checked += 1

    pts, counts, idx = topo._cube()
    assert verdict_of(edge_clearances(topo, topo.survivor_edges(pts, counts, idx), boxes)) \
        == "NO_OPEN_EDGES"
    checked += 1

    pts_a, counts_a, idx_a = topo._cube(drop_faces=1)
    near = [(x * 0.1 + 3.0, y * 0.1, z * 0.1) for x, y, z in pts_a]
    far2 = [(x * 0.1 + 30.0, y * 0.1, z * 0.1) for x, y, z in pts_a]
    surv = (topo.survivor_edges(far2, counts_a, idx_a)
            + topo.survivor_edges(near, counts_a, idx_a))
    rows_mix = edge_clearances(topo, surv, boxes)
    assert rows_mix[0]["min_distance"] < rows_mix[-1]["min_distance"]
    assert rows_mix == sorted(rows_mix, key=lambda r: r["min_distance"])
    checked += 1

    assert checked == 8, f"only {checked} of 8 cases ran"
    print(f"[check_fixture_mesh] self-test: {checked}/8 cases passed, including the two "
          f"mutations (shifted frame, edges moved into the pocket) ({SCRIPT_MARKER})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true",
                        help="Offline mutation tests. Needs no OBJ.")
    parser.add_argument("--obj", type=str, default=None,
                        help="Fixture OBJ. Default: insertion_paths.resolve_pocket_obj_path().")
    parser.add_argument("--expect-vertices", type=int, default=EXPECT_VERTICES,
                        help="RT-86's measured vertex count. A different one is the wrong OBJ.")
    parser.add_argument("--expect-faces", type=int, default=EXPECT_FACES,
                        help="RT-86's measured triangle count.")
    parser.add_argument("--plane-tol-m", type=float, default=PLANE_TOL_M,
                        help="Frame-guard tolerance in metres. Default is export_asset_mesh's "
                             "own --tol (0.1 mm). Do not raise it to make a run pass.")
    args = parser.parse_args()
    return _self_test() if args.self_test else run(args)


if __name__ == "__main__":
    sys.exit(main())
