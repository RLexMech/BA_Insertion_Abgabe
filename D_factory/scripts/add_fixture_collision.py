# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Apply the USD Physics collision schema to the fixture meshes.

Measured on the training machine 2026-07-26. After de-instancing, Isaac Lab
still warns:

    Could not perform 'modify_collision_properties' on any prims under:
    '/World/envs/env_0/Fixture' ... (1) The desired attribute does not exist on
    any of the prims. ... Discovered list of instanced prim paths: []

The instanced-prim list is now empty, so the instancing defect is gone and only
reason (1) remains: `modify_collision_properties` **modifies** existing collision
attributes, it does not **apply** the schema. A CAD import carries no
`UsdPhysics.CollisionAPI`, so `collision_props` in the spawn cfg silently reaches
nothing and the fixture has no collision at all.

This applies the schema in the asset, where it belongs -- every consumer of the
file then gets a collider, not just our env.

Approximation ``none`` is the exact triangle mesh -- the proxy-table value
(D-018 static collider). For the REAL fixture the decided route is ``sdf``
(D-062), and since D-109 (10b) the resolution has a floor: the voxel
(189.1/res, 189.1 mm being the fixture's largest edge) must not be coarser
than the SAPU interpenetration filter, or the filter punishes the collider
instead of the policy. Therefore ``--approximation sdf`` REQUIRES an explicit
``--sdf-resolution``; there is no silent default. The SDF authoring calls are
the ones `author_tool_ur5e.py` already uses on the part (MeshCollisionAPI
'sdf' + PhysxSDFMeshCollisionAPI resolution).

The authored value is 2048 (user, 2026-08-27) -- voxel 0.0923 mm.

[CORRECTION 2026-08-28] This docstring said the filter bound is 0.144 mm and
the floor therefore >= 1314. Both figures came from a cross play of 0.2876 mm,
which rested on a superseded 90.3 mm caliper reading of the part body. The
body is 90.00 mm (`insertion_tasks_cfg.py` PART_BODY_X) and the play is
0.5876 mm, so the filter bound and the floor both move -- the floor falls far
below 1024. 2048 REMAINS VALID because finer never hurts, but its original
justification is gone. Re-deriving the filter bound belongs to D-109 (10),
not here. Entry: `docs/decisions_inbox.md`, "The part body is 90.00 mm
across, not 90.3".

Idempotent: applying an API that is already present is a no-op, so re-running is
safe. The input is never edited in place -- backup, candidate, verify, replace
only on pass, matching `fix_stage_units.py`.

Exit code per D-081 via tools/isaac_exit.py: 0 pass/dry-run, 1 any failure.

The env-side gap this closes, measured RT-40: the spawn's
`modify_collision_properties` warning on '/World/envs/env_0/Fixture' -- the CAD
import carries no collision API, so `collision_enabled=True` reaches nothing.

UNVERIFIED -- authored on the dev PC (no Isaac installation), and the SDF path
below has NEVER RUN on any machine. The code for it is complete (MeshCollisionAPI
'sdf' + PhysxSDFMeshCollisionAPI resolution, both read back from a fresh stage);
what is missing is the run. On the training machine, real fixture:

    conda activate env_isaaclab
    python scripts\\add_fixture_collision.py --usd <path>\\Aufnahme_real_v1_mm.usd --approximation sdf --sdf-resolution 2048

``--weld-sweep`` is a MEASUREMENT MODE that authors nothing. It answers the
open question RT-56 left behind (903 unpaired edges on the fixture): is that
an indexing artefact or a real crack? It re-welds each mesh's points by
POSITION over a tolerance sweep and recounts. Falls to 0 -> artefact; hits a
floor -> real defect, and the floor is its size. Read-only, no simulation.

The range is NOT typed in. The sweep measures the smallest distance between two
distinct points of the mesh and places its top tolerance a decade below the
largest value that provably cannot fuse such a pair; both numbers are printed
(scripts/tools/mesh_spacing.py). RT-62 set the range too fine by hand and RT-63
too coarse, and neither error was visible in the log:

    python scripts\\add_fixture_collision.py --usd <path>\\Aufnahme_real_v1_mm.usd --weld-sweep
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
import traceback

from isaaclab.app import AppLauncher

# Loaded by path because scripts/tools is not a package -- the idiom
# author_tool_ur5e.py already uses.
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

# Printed on every run. RT-2 and RT-37 both delivered an OLD script silently --
# the log looked plausible and measured nothing. A log without this line is a run
# that proves nothing, whatever its exit code says.
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
         "Chosen for the fixture: 2048 (user, 2026-08-27). The D-109 (10b) floor is being re-derived "
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

# pxr is importable only after the app is up.
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402


def mesh_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    return [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]


def unpaired_edges(counts, indices) -> int:
    """Edges NOT used by exactly two faces, paired by point INDEX.

    Pure: takes the two topology arrays, touches no USD. That matters because
    --weld-sweep re-runs it on REMAPPED indices, and the count has to mean the
    same thing both times.

    NOTE the name. This counts ``c != 2``, so it includes both boundary edges
    (one face) and non-manifold edges (three or more). The docstring here used
    to say "used by exactly one face", which the code never did -- a reported
    count of n is "n edges not used by exactly two faces", and the split
    between the two kinds is NOT measured.

    DUPLICATE, AND NAMED AS ONE (2026-08-30). The rule now has a home outside
    Isaac: ``scripts/tools/mesh_topology.py`` carries ``edge_use_counts``,
    ``unpaired_edges``, ``split_unpaired`` and the weld, stdlib only and
    self-tested on the laptop. This file cannot simply import it and stay
    verified: ``add_fixture_collision.py`` launches a ``SimulationApp`` at
    module scope, so nothing in it runs offline, and replacing this body would
    ship an untested edit to a script RT-60..RT-64 verified. The delegation is
    a FOLLOW-UP whose verification is a re-run of RT-61's command. Until then
    two copies exist; if one is edited, edit both.
    """
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
    """``unpaired_edges`` on a mesh prim. 0 = closed (watertight) topology.

    MEASUREMENT for D-109 (10c): SAPU's sign flips silently on an open mesh.
    Reported and written to the sidecar, never gated on -- a hole is a finding
    to report, not something this script may repair (repo rule: conflicts are
    named, not resolved).
    """
    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if counts is None or indices is None:
        return None
    return unpaired_edges(counts, indices)


# Weld tolerances as FRACTIONS OF THE MESH'S OWN BOUNDING-BOX DIAGONAL, not
# as absolute lengths.
#
# RT-61 is why. The first version listed absolute metres and divided the raw
# point values by them -- but the fixture mesh stores its points in
# MILLIMETRES (the stage is metres and a 0.001 scale op sits above), so every
# tolerance came out 1000x finer than its label and the printed coordinates
# 1000x too large. The sweep claimed to reach 0.1 mm and actually stopped at
# 0.1 um. Nothing in the numbers was wrong; the units were, and a unit error
# in a tolerance is invisible because the output still looks like a sweep.
#
# A fraction of the mesh's own diagonal cannot have that failure: it means
# the same physical size whether the points are metres, millimetres or inches.
#
# RANGE, and RT-62 corrected it too. The comment claimed the top fraction
# "lands near 0.2 mm", which is true of 1e-3 -- but the tuple stopped at
# 1e-5, i.e. 2.49 um on this 248.96 mm diagonal, 236x finer than the
# 0.5876 mm cross play. The unit fix was right and the reach was still
# short, which is its own lesson: a corrected mechanism does not audit its
# own constants.
#
# RT-63 then ran with the top fraction at 1e-3, ~0.249 mm here, and OVER-WELDED:
# the floor ROSE from 9 unpaired edges to 12 and five edges came out used four
# times, i.e. the weld fused points that are genuinely different and the sweep
# manufactured the defect it was there to measure. Too fine in RT-62, too coarse
# in RT-63 -- a hand-set range fails in both directions, and neither failure
# shows in the log, because the output looks like a sweep either way.
#
# So the range is no longer a constant. It is DERIVED per mesh from the mesh's
# own smallest distance between two distinct points, in
# scripts/tools/mesh_spacing.py, which carries the derivation and a laptop
# self-test. The bottom stays at 1e-9 of the diagonal, far below float32
# spacing, so the first row is always "no weld at all".


def weld_sweep(prim: Usd.Prim) -> dict | None:
    """Re-weld the points by POSITION and recount. Writes nothing.

    THE DISCRIMINATING MEASUREMENT for the open-mesh finding of RT-56 (903
    unpaired edges). Two causes produce the same count and need opposite
    fixes:

      (a) an INDEXING artefact -- the surface is geometrically closed, but
          coincident points carry different indices, so the edge pairing
          fails. Welding by position removes it.
      (b) a GEOMETRIC crack -- there is a real gap in the surface. Welding by
          position cannot remove it.

    The sweep separates them by SHAPE, not by a threshold: if the count falls
    to 0 as the tolerance grows, it is (a), and the tolerance at which it
    falls is the numerical spread between the duplicate points. If it settles
    on a non-zero floor, it is (b), and the floor is the size of the real
    defect.

    Reading the free first line matters too: ``len(points) == sum(counts)``
    means no index is shared anywhere, i.e. the weld never ran at all.

    Method and sources: docs/reference/literature_check_open_mesh_fixture_2026-08-28.md
    """
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or counts is None or indices is None:
        return None

    n_points, n_corners = len(points), sum(int(c) for c in counts)

    # The point data's own extent. Everything below is expressed against it,
    # so the sweep never has to claim to know the unit (see the tolerance
    # comment block above). The unit IS reported, from the stage and
    # from the extent, but nothing is computed from that report.
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

    # THE RANGE IS MEASURED, NOT TYPED. See the comment block above and
    # scripts/tools/mesh_spacing.py for the derivation of the hard bound.
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
        # Round to the tolerance and group. Rounding, not clustering: a
        # cluster search would merge chains of near points transitively and
        # could weld across a real gap, which is the very thing being tested.
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

    # The comparison that decides is AS-AUTHORED against the FLOOR, not the
    # sweep against itself. RT-60 caught this: the count fell 903 -> 9 at the
    # very first tolerance and then held, so a sweep-only test read "never
    # moves" and printed GEOMETRIC CRACK for a mesh that was 99 % an indexing
    # artefact. Whatever the weld removes is indexing; whatever survives it is
    # geometry; a run can show both at once and usually does.
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

    # Where the survivors are. A count is not actionable; a coordinate is.
    # Only for a floor small enough to read -- a large one is a different
    # problem and a wall of coordinates would bury the number that matters.
    if 0 < floor <= 64:
        # The MEASURED top rung, not a typed constant. RT-63 printed survivor
        # coordinates from an over-welding tolerance and they were artefacts.
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
        # The floor is NOT printed as a number any more: the 1314 it used to
        # name came from the superseded 0.2876 mm cross play (see the
        # [CORRECTION] in the module docstring) and a stale bound in a log is
        # worse than no bound.
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

    # D-109 (10c) measurement, once per mesh. INFO, not a gate.
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
            # Same two calls author_tool_ur5e.py makes on the part mesh.
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
        # Read back from the fresh candidate stage, not echoed from the flag.
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

    # The stage above still has the input open, and on Windows that can block the
    # overwrite. Drop it before copying, and report the failure instead of dying
    # silently -- the previous repair run lost its replace step exactly here.
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
    # D-081: the exit happens BEFORE the shutdown, because close() does not
    # return (measured, RT-22) and this script's old `finally: close()` was
    # exactly the shape that reported exit 0 on failing runs elsewhere
    # (RT-27, RT-34).
    _code = 1
    try:
        _code = main()
        if _code is None:
            _code = 1  # a path forgot its return; loud is better than 0
    except SystemExit as _exc:
        _code = int(_exc.code or 0)
    except BaseException:
        traceback.print_exc()
        _code = 1
    exit_with(simulation_app, _code, tag="add_fixture_collision")
