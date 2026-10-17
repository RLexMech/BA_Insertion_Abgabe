# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Apply the USD Physics collision schema to the fixture meshes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
import traceback

from isaaclab.app import AppLauncher

_TOOLS = pathlib.Path(__file__).resolve().parent / "tools"
_spec = importlib.util.spec_from_file_location("isaac_exit", _TOOLS / "isaac_exit.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["isaac_exit"] = _mod
_spec.loader.exec_module(_mod)
exit_with = _mod.exit_with

_spec2 = importlib.util.spec_from_file_location("mesh_spacing", _TOOLS / "mesh_spacing.py")
_mod2 = importlib.util.module_from_spec(_spec2)
sys.modules["mesh_spacing"] = _mod2
_spec2.loader.exec_module(_mod2)
min_point_spacing = _mod2.min_point_spacing
weld_tolerance_series = _mod2.weld_tolerance_series
SPACING_MARKER = _mod2.SPACING_MARKER

SCRIPT_MARKER = "add_fixture_collision-2026-08-28e-weldsweep-selfbound"

parser = argparse.ArgumentParser(description="Apply collision schema to the fixture meshes.")
parser.add_argument("--usd", type=str, required=True, help="Path to the fixture USD file.")
parser.add_argument(
    "--approximation",
    type=str,
    default="none",
    choices=["none", "sdf", "convexHull", "convexDecomposition", "boundingCube", "boundingSphere", "meshSimplification"],
    help="Collision approximation; 'none' is the exact triangle mesh (proxy table), 'sdf' the D-062 route for the real fixture.",
)
parser.add_argument(
    "--sdf-resolution",
    type=int,
    default=None,
    help="SDF resolution. REQUIRED with --approximation sdf, refused otherwise. "
         "Fixture setting: 2048. The minimum required resolution needs reassessment "
         "after the 2026-08-28 cross-play correction. No default on purpose.",
)
parser.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing.")
parser.add_argument(
    "--weld-sweep",
    action="store_true",
    help="MEASURE ONLY, writes nothing and authors nothing: re-weld each mesh's points by "
         "position over a tolerance sweep and recount the unpaired edges. Separates an "
         "indexing artefact (count falls to 0) from a real crack (count hits a floor). "
         "The discriminating measurement for the RT-56 open-mesh finding.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

if args_cli.weld_sweep and (args_cli.dry_run or args_cli.sdf_resolution is not None):
    parser.error("--weld-sweep is a measurement mode on its own; it authors nothing, so "
                 "--dry-run and --sdf-resolution do not apply.")
if args_cli.approximation == "sdf" and args_cli.sdf_resolution is None and not args_cli.weld_sweep:
    parser.error("--approximation sdf requires an explicit --sdf-resolution (D-109 (10b): no silent default).")
if args_cli.approximation != "sdf" and args_cli.sdf_resolution is not None:
    parser.error("--sdf-resolution only applies to --approximation sdf.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402


def mesh_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    return [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]


def unpaired_edges(counts, indices) -> int:
    """Edges NOT used by exactly two faces, paired by point INDEX."""
    edges: dict[tuple[int, int], int] = {}
    k = 0
    for n in counts:
        face = indices[k:k + n]
        k += n
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1
    return sum(1 for c in edges.values() if c != 2)


def boundary_edge_count(prim: Usd.Prim) -> int | None:
    """``unpaired_edges`` on a mesh prim."""
    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if counts is None or indices is None:
        return None
    return unpaired_edges(counts, indices)


def weld_sweep(prim: Usd.Prim) -> dict | None:
    """Re-weld the points by POSITION and recount."""
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or counts is None or indices is None:
        return None

    n_points, n_corners = len(points), sum(int(c) for c in counts)

    lo = [min(float(p[a]) for p in points) for a in range(3)]
    hi = [max(float(p[a]) for p in points) for a in range(3)]
    size = [hi[a] - lo[a] for a in range(3)]
    diag = sum(s * s for s in size) ** 0.5
    mpu = UsdGeom.GetStageMetersPerUnit(prim.GetStage())

    out: dict = {
        "path": str(prim.GetPath()),
        "n_points": n_points,
        "n_faces": len(counts),
        "n_face_corners": n_corners,
        "points_equal_corners": n_points == n_corners,
        "as_authored": unpaired_edges(counts, indices),
        "extent_point_units": size,
        "diagonal_point_units": diag,
        "stage_meters_per_unit": mpu,
        "sweep": {},
    }
    print(f"[weld-sweep] {prim.GetPath()}")
    print(f"[weld-sweep]   points {n_points}, faces {len(counts)}, face corners {n_corners}")
    print(f"[weld-sweep]   extent in POINT UNITS: {size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}, "
          f"diagonal {diag:.4f}; stage metersPerUnit {mpu}")
    print("[weld-sweep]   (point units are NOT assumed to be metres -- every tolerance below "
          "is a fraction of that diagonal)")
    if n_points == n_corners:
        print("[weld-sweep]   *** points == face corners: NO index is shared anywhere. "
              "The vertex weld never ran. ***")
    print(f"[weld-sweep]   as authored: {out['as_authored']} unpaired edges")

    spacing = min_point_spacing(points)
    if spacing is None:
        print("[weld-sweep]   *** fewer than two DISTINCT point positions: no spacing "
              "exists, so no tolerance range can be bounded. Nothing swept. ***")
        out["min_point_spacing_point_units"] = None
        out["skipped"] = "fewer than two distinct point positions"
        return out

    series = weld_tolerance_series(diag, spacing["d_min"])
    fractions = series["fractions"]

    out["min_point_spacing_point_units"] = spacing["d_min"]
    out["min_spacing_pair"] = {"index_a": spacing["index_a"], "index_b": spacing["index_b"],
                               "point_a": spacing["point_a"], "point_b": spacing["point_b"]}
    out["n_distinct_positions"] = spacing["n_distinct_positions"]
    out["n_exact_duplicate_points"] = spacing["n_exact_duplicate_points"]
    out["hard_weld_bound_point_units"] = series["hard_bound_point_units"]
    out["weld_margin_factor"] = series["margin"]
    out["tolerance_fractions_of_diagonal"] = list(fractions)
    out["top_fraction_used"] = series["top_fraction"]

    pa, pb = spacing["point_a"], spacing["point_b"]
    print(f"[weld-sweep]   distinct positions {spacing['n_distinct_positions']}, "
          f"{spacing['n_exact_duplicate_points']} exact duplicate points dropped")
    print(f"[weld-sweep]   MEASURED smallest distance between two DISTINCT points: "
          f"{spacing['d_min']:.6e} point units")
    print(f"[weld-sweep]     between index {spacing['index_a']} "
          f"({pa[0]:.6f},{pa[1]:.6f},{pa[2]:.6f}) and index {spacing['index_b']} "
          f"({pb[0]:.6f},{pb[1]:.6f},{pb[2]:.6f})")
    print(f"[weld-sweep]   hard bound (d_min / sqrt(3), largest tolerance that still "
          f"cannot fuse them): {series['hard_bound_point_units']:.6e} point units")
    print(f"[weld-sweep]   safety margin {series['margin']:g}x -> top tolerance allowed "
          f"{series['allowed_top_point_units']:.6e} point units")

    if not fractions:
        print("[weld-sweep]   *** NO TOLERANCE FITS: the measured spacing leaves not one "
              "decade above the range floor. No range is invented. Nothing swept. ***")
        out["skipped"] = "measured spacing leaves no tolerance decade"
        return out

    print(f"[weld-sweep]   range: {fractions[0]:.0e} .. {fractions[-1]:.0e} of the diagonal "
          f"({len(fractions)} rungs), top rung = {fractions[-1] * diag:.6e} point units")

    for frac in fractions:
        tol = frac * diag
        key_of: dict[tuple[int, int, int], int] = {}
        remap = [0] * n_points
        for i, p in enumerate(points):
            key = (round(float(p[0]) / tol), round(float(p[1]) / tol), round(float(p[2]) / tol))
            if key not in key_of:
                key_of[key] = len(key_of)
            remap[i] = key_of[key]
        merged = [remap[int(i)] for i in indices]
        n_open = unpaired_edges(counts, merged)
        out["sweep"][f"{frac:.0e}"] = {"fraction_of_diagonal": frac,
                                       "tolerance_point_units": tol,
                                       "unique_points": len(key_of),
                                       "unpaired_edges": n_open}
        print(f"[weld-sweep]   tol {frac:.0e} of diag = {tol:.3e} point units: "
              f"{len(key_of):>7} unique points, {n_open:>6} unpaired edges")

    finals = [v["unpaired_edges"] for v in out["sweep"].values()]
    floor = min(finals)
    authored = out["as_authored"]
    removed = authored - floor
    out["floor"] = floor
    out["removed_by_weld"] = removed
    print(f"[weld-sweep]   welding removes {removed} of {authored} unpaired edges; "
          f"floor {floor}")

    if floor == 0:
        out["verdict"] = "INDEXING_ARTEFACT"
        print("[weld-sweep]   VERDICT: INDEXING ARTEFACT -- the count reaches 0 under a "
              "position weld, so the surface is geometrically closed and the authored "
              "count was entirely unshared indices.")
    elif removed == 0:
        out["verdict"] = "GEOMETRIC_CRACK"
        print(f"[weld-sweep]   VERDICT: GEOMETRIC CRACK -- welding removes nothing, so all "
              f"{authored} unpaired edges are geometry.")
    else:
        out["verdict"] = "MIXED"
        print(f"[weld-sweep]   VERDICT: MIXED -- {removed} of {authored} were unshared "
              f"indices, {floor} survive every weld and are geometry. The floor is what "
              f"the SDF sign question is actually about; the rest was never a hole.")

    if 0 < floor <= 64:
        tol = fractions[-1] * diag
        key_of: dict[tuple[int, int, int], int] = {}
        rep: dict[int, int] = {}
        remap = [0] * n_points
        for i, p in enumerate(points):
            key = (round(float(p[0]) / tol), round(float(p[1]) / tol), round(float(p[2]) / tol))
            if key not in key_of:
                key_of[key] = len(key_of)
                rep[key_of[key]] = i
            remap[i] = key_of[key]
        merged = [remap[int(i)] for i in indices]
        edges: dict[tuple[int, int], int] = {}
        k = 0
        for n in counts:
            face = merged[k:k + n]
            k += n
            for j in range(n):
                a, b = int(face[j]), int(face[(j + 1) % n])
                edges[(a, b) if a < b else (b, a)] = edges.get((a, b) if a < b else (b, a), 0) + 1
        survivors = []
        for (a, b), c in edges.items():
            if c == 2:
                continue
            pa, pb = points[rep[a]], points[rep[b]]
            survivors.append({
                "uses": c,
                "kind": "boundary" if c == 1 else "non-manifold",
                "a": [float(v) for v in pa],
                "b": [float(v) for v in pb],
            })
        out["survivors"] = survivors
        n_bnd = sum(1 for s in survivors if s["uses"] == 1)
        print(f"[weld-sweep]   survivors: {n_bnd} boundary (one face), "
              f"{len(survivors) - n_bnd} non-manifold (three or more)")
        for s in survivors:
            a, b = s["a"], s["b"]
            print(f"[weld-sweep]     {s['kind']:<13} used {s['uses']}x  "
                  f"({a[0]:9.4f},{a[1]:9.4f},{a[2]:9.4f}) -> "
                  f"({b[0]:9.4f},{b[1]:9.4f},{b[2]:9.4f})  [point units]")
    return out


def describe(stage: Usd.Stage) -> dict:
    meshes = mesh_prims(stage)
    return {
        "mesh_count": len(meshes),
        "meshes": [
            {
                "path": str(p.GetPath()),
                "has_collision_api": bool(p.HasAPI(UsdPhysics.CollisionAPI)),
                "has_mesh_collision_api": bool(p.HasAPI(UsdPhysics.MeshCollisionAPI)),
                "approximation": (
                    str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get())
                    if p.HasAPI(UsdPhysics.MeshCollisionAPI)
                    else None
                ),
                "has_sdf_api": bool(p.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)),
                "sdf_resolution": (
                    PhysxSchema.PhysxSDFMeshCollisionAPI(p).GetSdfResolutionAttr().Get()
                    if p.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
                    else None
                ),
            }
            for p in meshes
        ],
    }


def main() -> int:
    print(f"[add_fixture_collision] marker: {SCRIPT_MARKER}")
    usd_path = pathlib.Path(args_cli.usd).resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"No such USD file: {usd_path}")
    print(f"[add_fixture_collision] resolved input: {usd_path}")
    if args_cli.approximation == "sdf" and not args_cli.weld_sweep:
        print(f"[add_fixture_collision] sdf resolution: {args_cli.sdf_resolution} "
              f"(voxel {189.1/args_cli.sdf_resolution:.4f} mm over the 189.1 mm edge; "
              f"D-109 (10b) floor under re-derivation)")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {usd_path}")

    if args_cli.weld_sweep:
        meshes = mesh_prims(stage)
        if not meshes:
            raise RuntimeError("No UsdGeom.Mesh prims found; nothing to sweep.")
        report = [weld_sweep(p) for p in meshes]
        sidecar = usd_path.with_suffix(usd_path.suffix + ".weldsweep.json")
        sidecar.write_text(json.dumps(
            {"marker": SCRIPT_MARKER, "spacing_marker": SPACING_MARKER,
             "usd": str(usd_path),
             "meshes": [r for r in report if r is not None]}, indent=2))
        print(f"[weld-sweep] sidecar: {sidecar}")
        print("[weld-sweep] nothing was written to the USD.")
        skipped = [r for r in report if r is not None and "skipped" in r]
        if skipped:
            for r in skipped:
                print(f"[weld-sweep] NOT SWEPT: {r['path']} -- {r['skipped']}")
            print("[weld-sweep] exit 3: at least one mesh could not be bounded from its "
                  "own geometry. No range was guessed; decide by hand.")
            return 3
        return 0

    before = describe(stage)
    print("[add_fixture_collision] BEFORE:")
    print(json.dumps(before, indent=2))

    watertight = {}
    for prim in mesh_prims(stage):
        n_open = boundary_edge_count(prim)
        watertight[str(prim.GetPath())] = n_open
        state = "UNMEASURABLE (no topology attrs)" if n_open is None else (
            "WATERTIGHT (0 boundary edges)" if n_open == 0 else f"*** OPEN MESH: {n_open} boundary edges ***"
        )
        print(f"[add_fixture_collision] watertight (D-109 10c): {prim.GetPath()}: {state}")

    if before["mesh_count"] == 0:
        raise RuntimeError(
            "No UsdGeom.Mesh prims found. Either the asset is still instanced "
            "(run fix_fixture_asset.py first) or the geometry is not mesh-based."
        )

    if args_cli.dry_run:
        print(f"[add_fixture_collision] --dry-run: would apply CollisionAPI + MeshCollisionAPI "
              f"(approximation '{args_cli.approximation}') to {before['mesh_count']} mesh prims.")
        return 0

    backup = usd_path.with_suffix(usd_path.suffix + ".precollision.bak")
    shutil.copy2(usd_path, backup)
    print(f"[add_fixture_collision] backup written: {backup}")

    for prim in mesh_prims(stage):
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set(args_cli.approximation)
        if args_cli.approximation == "sdf":
            sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            sdf_api.CreateSdfResolutionAttr().Set(int(args_cli.sdf_resolution))
        print(f"[add_fixture_collision] applied to {prim.GetPath()}")

    candidate = usd_path.with_name(usd_path.stem + ".collision" + usd_path.suffix)
    stage.Flatten().Export(str(candidate))
    print(f"[add_fixture_collision] candidate exported: {candidate}")

    check_stage = Usd.Stage.Open(str(candidate))
    if check_stage is None:
        raise RuntimeError(f"Could not re-open the candidate: {candidate}")
    after = describe(check_stage)

    checks = {
        "mesh_count_unchanged": after["mesh_count"] == before["mesh_count"],
        "all_have_collision_api": all(m["has_collision_api"] for m in after["meshes"]),
        "all_have_mesh_collision_api": all(m["has_mesh_collision_api"] for m in after["meshes"]),
        "approximation_set": all(m["approximation"] == args_cli.approximation for m in after["meshes"]),
    }
    if args_cli.approximation == "sdf":
        checks["sdf_api_applied"] = all(m["has_sdf_api"] for m in after["meshes"])
        checks["sdf_resolution_set"] = all(
            m["sdf_resolution"] == int(args_cli.sdf_resolution) for m in after["meshes"]
        )
    print("[add_fixture_collision] AFTER (fresh stage, candidate file):")
    print(json.dumps({"describe": after, "checks": checks}, indent=2))

    report = {
        "input": str(usd_path),
        "backup": str(backup),
        "candidate": str(candidate),
        "approximation": args_cli.approximation,
        "sdf_resolution": args_cli.sdf_resolution,
        "boundary_edges_per_mesh": watertight,
        "before": before,
        "after": after,
        "checks": checks,
    }
    report_path = usd_path.with_suffix(".collision.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[add_fixture_collision] wrote {report_path}")

    for name, passed in checks.items():
        print(f"[add_fixture_collision] {name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        print("[add_fixture_collision] FAIL; original left untouched.")
        return 1

    del stage, check_stage
    try:
        shutil.copy2(candidate, usd_path)
    except OSError as exc:
        print(f"[add_fixture_collision] could not replace the original: {exc}")
        print(f"[add_fixture_collision] copy it manually: {candidate} -> {usd_path}")
        return 1
    print(f"[add_fixture_collision] PASS; {usd_path.name} replaced (backup at {backup.name})")
    print("[add_fixture_collision] next: re-run zero_agent; the modify_collision_properties warning must be gone.")
    return 0


if __name__ == "__main__":
    _code = 1
    try:
        _code = main()
        if _code is None:
            _code = 1
    except SystemExit as _exc:
        _code = int(_exc.code or 0)
    except BaseException:
        traceback.print_exc()
        _code = 1
    exit_with(simulation_app, _code, tag="add_fixture_collision")
