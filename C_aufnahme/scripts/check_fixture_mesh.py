# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""RT-91 (b): WHERE do the fixture mesh's open edges sit? No GPU, no Isaac.

THE QUESTION, AND WHY THE COUNT IS NOT IT
------------------------------------------
``wp.mesh_query_point`` carries the note ``Mesh must be watertight!`` and says
nothing about what happens when it is not. SAPU (D-109 point (10)) queries the
part's sample points against the FIXTURE mesh, and that surface is provably
open. So the sign of the interpenetration distance is in question -- but only
WHERE the surface is open. If the open edges sit on the outer rim, no sample
point of a seated part ever comes near them and the worry is empirically dead;
if they sit in the pocket wall, SAPU reads its sign exactly there.

**The count is not the criterion. The LOCATION is.** RT-90 makes that concrete
in both directions: the PART mesh carries 380 unpaired edges and its sign still
measured correct -- but it was measured AT LEAST TWO FULL EXTENTS OUTSIDE the
mesh, and SAPU reads the sign AT THE SURFACE. Neither run transfers to the
other.

WHAT IS ALREADY MEASURED, AND WHAT IS NOT
------------------------------------------
RT-61 answered this for the **USD** (``HANDOFF-SZENE.md``, "THE 9 EDGES HAVE AN
ADDRESS"): after a position weld, nine survivors, all at ``y = 90.450 mm`` and
``z = +15.000 mm``, spanning x from ``-84.794`` to ``+65.206`` -- the outer top
rim where the tab meets the body at +Y, 17.90 mm outside ``POCKET_WALL_Y`` and
above the opening plane.

That is not this run's answer, for one reason: **Warp never sees the USD, and
Warp does not weld.** ``SdfDistanceQuery`` builds ``wp.Mesh`` from
``trimesh.load(obj, process=False)``, so the topology Warp receives is the
OBJ's AS AUTHORED. Nobody has counted that. RT-86's parity argument does not
settle it either: 757 triangles carry 2271 half-edges, and ``2271 = 2*1131 + 9``
and ``2271 = 2*684 + 903`` are both true, so an odd count rules nothing in.

This script therefore measures the OBJ twice -- as authored, which is what Warp
gets, and welded, which is what RT-61 measured -- and gives every survivor an
ADDRESS and a DISTANCE to the two volumes the part actually sweeps.

WHAT IT DECIDES: NOTHING
------------------------
No task number is written by this file. It reports a location and a clearance;
whether that clearance is enough is a decision, and decisions live in
``DECISIONS.md``.

THE THREE NUMBERS IT USES, AND WHERE EACH LIVES
------------------------------------------------
* **903 / 757** -- the fixture OBJ's vertex and face counts, MEASURED by RT-86
  (``rt_logs/VERDICTS.md``). A different pair is the wrong OBJ, not a finding.
* **1.0e-4 m** -- the frame guard's tolerance. Home:
  ``export_asset_mesh.py --tol`` (default 0.1 mm), the tolerance the SAME export
  path already uses for "did the export preserve the geometry", which is exactly
  the claim the frame guard makes. Not picked here.
* **the weld tolerance series** -- not a number at all: fractions of the mesh's
  own bounding-box diagonal, derived by ``tools/mesh_spacing.py`` from the
  measured closest distinct pair. That indirection is what RT-61's unit defect
  earned, where absolute metre tolerances silently reached 0.1 um on a
  millimetre mesh.

NO AppLauncher, ON PURPOSE
--------------------------
Nothing here opens a USD stage. The OBJ is read with the repo's own stdlib
parser and the pocket constants come out of ``insertion_tasks_cfg.py`` with
``isaaclab`` stubbed -- the route ``check_workcell_geometry.py`` established.
The project's AppLauncher-first rule applies to scripts that import Isaac, and
this one does not. It costs seconds and needs no GPU.

    python scripts/check_fixture_mesh.py --self-test
    .\scripts\rt_log.ps1 RT-91 python scripts\check_fixture_mesh.py
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import types

# Printed on every run. A log without this line is a run that proves nothing,
# whatever its exit code says (RT-2 / RT-37 both delivered old code silently).
SCRIPT_MARKER = "check_fixture_mesh-2026-08-30b-weldedgate"

_HERE = pathlib.Path(__file__).resolve().parent
_TOOLS = _HERE / "tools"
_PKG_DIR = (
    _HERE.parent / "source" / "insertion" / "insertion" / "tasks" / "direct" / "insertion"
)
_CFG_PATH = _PKG_DIR / "insertion_tasks_cfg.py"

# MEASURED by RT-86, not chosen. rt_logs/VERDICTS.md: "1 Mesh-Prim, 903
# Vertices, 757 Dreiecke, keine Nicht-Dreiecke".
EXPECT_VERTICES = 903
EXPECT_FACES = 757

# The frame guard's tolerance. Home: export_asset_mesh.py --tol, default 1.0e-4
# m, the same export path's own "did this preserve the geometry" tolerance. Do
# NOT adjust it to make a run pass -- that is the note that file already
# carries, and it applies here for the same reason.
PLANE_TOL_M = 1.0e-4

# How many survivor edges are printed in full. The MINIMUM distance is computed
# over every survivor and the verdict rests on that; this only bounds the wall
# of coordinates a 903-edge as-authored reading would otherwise produce.
MAX_PRINTED_SURVIVORS = 24


# ---------------------------------------------------------------------------
# Loading, all of it stdlib or repo-local
# ---------------------------------------------------------------------------
def _load_tool(stem: str) -> types.ModuleType:
    """Load a module out of ``scripts/tools`` by path.

    ``scripts/tools`` is not a package -- the idiom ``add_fixture_collision.py``
    and ``export_asset_mesh.py`` already use.
    """
    spec = importlib.util.spec_from_file_location(stem, _TOOLS / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod


def load_paths_module() -> types.ModuleType:
    """``insertion_paths`` -- stdlib only, no Isaac, no SimulationApp.

    RT-88's lesson: asking ``insertion_tasks_cfg`` for a FILE PATH used to cost
    a simulation app. The path block has its own module now for exactly that.
    """
    spec = importlib.util.spec_from_file_location(
        "_insertion_paths", _PKG_DIR / "insertion_paths.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_cfg() -> types.ModuleType:
    """Read ``insertion_tasks_cfg`` from SOURCE with ``isaaclab`` stubbed.

    Copied in method from ``check_workcell_geometry.load_cfg``, including its
    reason for compiling the source text instead of using
    ``spec.loader.exec_module``: on 2026-08-18 a ``__pycache__`` entry whose
    mtime and size still matched made a checker certify numbers that were no
    longer in the file. The cache is taken out of the path entirely.

    Stubbing is honest here in a way that copying the constants into this file
    would not be -- the check then reads the SAME constants the environment
    reads, which is the one-home rule.

    ONE ADDITION over ``check_workcell_geometry``'s version, and it is not
    style: ``insertion_tasks_cfg.py:488`` does ``from .insertion_paths import
    ...``, a RELATIVE import, which raises ``ImportError: attempted relative
    import with no known parent package`` when the source is exec'd into a bare
    module. A throwaway package with ``__path__`` pointing at the real directory
    satisfies it -- the same device ``check_insertion_sdf.load_sdf_module`` uses
    for ``insertion_sdf.py``. Nothing is installed and nothing on disk is
    touched.
    """
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


# ---------------------------------------------------------------------------
# The decisions. Pure: the self-test drives these with synthetic input.
# ---------------------------------------------------------------------------
def expected_planes(cfg) -> list:
    """The seven dense coordinate planes, in METRES, from the cfg.

    The SAME seven ``verify_fixture_usd.py:392-399`` checks against the USD,
    with the same labels, so a disagreement between the two runs is about the
    EXPORT and not about which planes were looked at. Every value is a measured
    CAD number with one home in ``insertion_tasks_cfg.py``; none is typed here.
    """
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
    """Does the OBJ sit in the frame the pocket constants are written in?

    For each expected plane, find the POPULATED coordinate nearest to it and
    report that coordinate, how many vertices share it, and the error. Passes
    when every one of them is within ``tol_m``.

    WHY THIS GATES AND THE ADDRESS DOES NOT. A coordinate is only an address if
    the mesh is in the frame the address is written in. The fixture OBJ was
    exported with NO ``--frame-prim`` (``export_asset_mesh.py:42``), i.e. in the
    fixture stage's world frame, and nothing has ever checked that this equals
    the asset frame the constants describe. Without this guard every distance
    below would be a fluent, wrong number, and a wrong address is worse than no
    address.

    The ERROR is exact: the smallest ``|p[axis] - want|`` over all vertices,
    not the distance to a rounded bucket. An earlier version bucketed first and
    reported the bucket's error, which printed 0.0002 mm of pure quantisation
    for a coordinate that matched the constant exactly -- a number that invites
    exactly the wrong conclusion. ``population_tol`` is used ONLY to count how
    many vertices share that nearest coordinate, which is the "is this a dense
    plane or a lone stray vertex" reading ``verify_fixture_usd.py --planes``
    reports on the USD. No minimum population is imposed: the count is printed
    and judged by a reader, not gated on a threshold nobody derived.
    """
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
    """The two volumes the part passes through, as ``(name, lo, hi)`` in metres.

    Stage 2 is the pocket cavity proper: the walls at ``+-POCKET_WALL_X/Y``, the
    floor at ``POCKET_FLOOR_Z``, the opening plane at z = 0. Stage 1 is the
    wider mouth above it, ``POCKET_STAGE1_X_RANGE`` x ``POCKET_STAGE1_Y_RANGE``
    up to ``POCKET_RIM_Z``.

    Both are the RECTANGULAR HULL of their stage, which is deliberately
    GENEROUS: the real cross-section has notches and a tab, so the hull contains
    the swept volume and can only over-report an edge as "inside". A guard that
    errs toward the alarm is the right direction for this question.
    """
    return [
        ("stage 2 (pocket cavity)",
         (-cfg.POCKET_WALL_X, -cfg.POCKET_WALL_Y, cfg.POCKET_FLOOR_Z),
         (+cfg.POCKET_WALL_X, +cfg.POCKET_WALL_Y, 0.0)),
        ("stage 1 (mouth)",
         (cfg.POCKET_STAGE1_X_RANGE[0], cfg.POCKET_STAGE1_Y_RANGE[0], 0.0),
         (cfg.POCKET_STAGE1_X_RANGE[1], cfg.POCKET_STAGE1_Y_RANGE[1], cfg.POCKET_RIM_Z)),
    ]


def edge_clearances(topo, survivors, boxes) -> list:
    """Per survivor edge: its distance to every box. Sorted CLOSEST FIRST.

    Each entry keeps the survivor dict and adds ``distances`` (one per box, same
    order) and ``min_distance``. Sorting by ``min_distance`` puts the edge that
    decides the verdict at the top of the printout, whatever the face order was.
    """
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
    """``EDGES_OUTSIDE_SWEPT_VOLUME`` only when EVERY survivor clears EVERY box.

    ``> 0.0`` and not ``>= 0.0``: ``segment_box_distance`` returns exactly 0.0
    when the edge touches the box, and an edge lying ON the pocket wall is the
    case this whole run exists to catch.
    """
    if not rows:
        return "NO_OPEN_EDGES"
    return ("EDGES_OUTSIDE_SWEPT_VOLUME"
            if all(r["min_distance"] > 0.0 for r in rows)
            else "EDGES_IN_SWEPT_VOLUME")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
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

    # --- G1: is this the OBJ RT-86 wrote? A different pair is the wrong file,
    # and every number below would then describe something else.
    g1 = len(points) == args.expect_vertices and len(faces) == args.expect_faces
    print(f"[check_fixture_mesh] G1 mesh identity: {len(points)}/{len(faces)} against "
          f"RT-86's {args.expect_vertices}/{args.expect_faces} -- "
          f"{'PASS' if g1 else 'FAIL'}")

    # Triangles only: wp.Mesh takes a flat triangle index array and nothing else.
    n_tri = mesh_export.require_triangles(counts)
    print(f"[check_fixture_mesh] all {n_tri} faces are triangles")

    box = mesh_export.bbox(points)
    diag = sum(e * e for e in box["extent"]) ** 0.5
    print(f"[check_fixture_mesh] bbox min ({box['min'][0]:.6f}, {box['min'][1]:.6f}, "
          f"{box['min'][2]:.6f}) max ({box['max'][0]:.6f}, {box['max'][1]:.6f}, "
          f"{box['max'][2]:.6f}) m")
    print(f"[check_fixture_mesh] extent {box['extent'][0]:.6f} x {box['extent'][1]:.6f} "
          f"x {box['extent'][2]:.6f} m, diagonal {diag:.6f} m")

    # --- G2: the frame guard.
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

    # --- topology AS AUTHORED. This is what Warp receives: SdfDistanceQuery
    # builds wp.Mesh from trimesh.load(process=False) and nothing welds.
    authored = topo.split_unpaired(counts, indices)
    print(f"[check_fixture_mesh] as authored (what Warp receives): "
          f"{authored['total']} unpaired edges "
          f"({authored['boundary']} boundary, {authored['non_manifold']} non-manifold)")
    if len(points) == len(indices):
        print("[check_fixture_mesh] *** points == face corners: NO index is shared "
              "anywhere. The OBJ carries no vertex weld at all. ***")

    # --- the weld sweep, the way RT-60..RT-64 ran it: tolerances as FRACTIONS
    # of this mesh's own diagonal, bounded by its own measured closest pair.
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

    # --- the addresses. Both readings, because they answer different questions:
    # as authored is what Warp queries, welded is what RT-61 measured.
    boxes = swept_volume_boxes(cfg)
    print("[check_fixture_mesh] swept-volume boxes (rectangular hulls, metres):")
    for name, lo, hi in boxes:
        print(f"[check_fixture_mesh]   {name}: x {lo[0]:+.6f}..{hi[0]:+.6f}  "
              f"y {lo[1]:+.6f}..{hi[1]:+.6f}  z {lo[2]:+.6f}..{hi[2]:+.6f}")

    readings = [("AUTHORED", None)]
    if top_tol is not None:
        readings.append(("WELDED", top_tol))

    # THE GATE RESTS ON THE WELDED READING ONLY, and RT-91 is what earned that
    # (2026-08-30). The first version gated on both, and the as-authored
    # reading then failed the run for a reason that is not a hole:
    #
    #   * RT-91 measured 903 unpaired edges as authored and 9 after a 2.49 um
    #     weld, so 894 of them are INDEX artefacts -- the surface is closed
    #     there and only the indices are unshared.
    #   * The swept-volume boxes are the rectangular hulls of the pocket
    #     stages, so a box FACE *is* the pocket wall. Any edge lying on that
    #     wall reads exactly 0.0000 mm whether it is a hole or an ordinary
    #     interior edge. With 894 artefacts spread over the whole surface, a
    #     zero was guaranteed and carried no information.
    #
    # For a WELDED survivor the same 0.0000 mm does mean something: it is not
    # an index artefact, it is a real open edge, and it lies on the surface
    # SAPU reads its sign at. That is the case worth failing on.
    #
    # The as-authored reading is still PRINTED, because it is what Warp
    # receives: `wp.Mesh` is built from `trimesh.load(process=False)`.
    # Whether `wp.mesh_query_point` pairs edges by index or rebuilds adjacency
    # by position is UNKNOWN for Warp -- documented for neither. That is a
    # named gap, not something this gate may decide.
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

    # ONE LINE PER DANGEROUS EDGE, ON ITS OWN, WITH ITS OWN WORD. RT-91's log
    # printed all nine welded survivors and `filter_rt_log.py` still could not
    # surface three of them: every survivor line looks like every other, so the
    # filter kept returning the as-authored block. The edges that decide the
    # verdict now carry a token no other line uses.
    print(f"[check_fixture_mesh] ZERO-EDGE COUNT: {len(zero_edges)} welded survivor(s) "
          f"touch a swept volume")
    for r in zero_edges:
        a, b = r["a"], r["b"]
        print(f"[check_fixture_mesh] ZERO-EDGE  used {r['uses']}x  "
              f"({a[0] * 1000:9.4f},{a[1] * 1000:9.4f},{a[2] * 1000:9.4f}) -> "
              f"({b[0] * 1000:9.4f},{b[1] * 1000:9.4f},{b[2] * 1000:9.4f}) mm  "
              f"| dist to boxes [mm]: "
              + "  ".join(f"{d * 1000.0:9.4f}" for d in r["distances"]))

    # --- the cross-check against RT-61, stated rather than assumed. RT-61
    # measured the USD after a weld; this run measured the OBJ. Agreement is a
    # result and disagreement is a result; neither is inferred from the other.
    if welded is not None:
        print(f"[check_fixture_mesh] cross-check vs RT-61 COUNT: RT-61 found 9 boundary / "
              f"0 non-manifold; this OBJ welds to {welded} unpaired edges. "
              f"{'AGREES' if welded == 9 else 'DISAGREES -- report it, do not explain it away'}")
        # RT-61 made a SECOND claim, and it is the one this run exists to test:
        # "all nine sit on the outer top rim", i.e. every one of them clear of
        # the swept volume. That claim is checked separately from the count,
        # because RT-91 showed the two can disagree -- and RT-61's own listed
        # break points (-84.794 / -58.794 / -32.794 / +34.206 / +49.706 /
        # +65.206) span six edges, not nine, so "all nine" was already loose.
        print(f"[check_fixture_mesh] cross-check vs RT-61 ADDRESS: RT-61 claimed ALL nine "
              f"lie on the outer rim, clear of the pocket; this run finds "
              f"{len(zero_edges)} welded survivor(s) touching a swept volume. "
              f"{'AGREES' if not zero_edges else 'DISAGREES -- the ZERO-EDGE lines above are the finding'}")

    ok = g1 and g2 and all_ok
    print(f"[check_fixture_mesh] VERDICT: {'OPEN_EDGES_CLEAR_OF_THE_PART' if ok else 'OPEN_EDGES_NOT_CLEAR'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# The counter-proof (D-080): every check gets input that MUST break it
# ---------------------------------------------------------------------------
class _FakeCfg:
    """The pocket constants only, so the self-test needs no cfg file.

    Values are NOT the real ones and are not meant to be -- the self-test proves
    the DECISIONS, and using the real numbers here would quietly make this file
    a second home for them.
    """
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

    # 2 -- the frame guard passes when the vertices sit on the expected planes.
    planes = expected_planes(cfg)
    assert len(planes) == 7
    on_frame = [(-1.0, -2.0, -3.0), (1.0, 2.0, 0.0), (-4.0, 0.0, 0.0), (5.0, 0.0, 0.0)]
    ok, rows = frame_guard(on_frame, planes, 1e-4)
    assert ok, [r for r in rows if not r["ok"]]
    assert len(rows) == 7
    checked += 1

    # 3 -- THE MUTATION. Shift every vertex by 1 mm and the guard MUST fail. A
    # frame guard that survives a translated mesh guards nothing.
    off_frame = [(x + 1e-3, y + 1e-3, z + 1e-3) for x, y, z in on_frame]
    ok_bad, rows_bad = frame_guard(off_frame, planes, 1e-4)
    assert not ok_bad, "the frame guard accepted a mesh shifted by 1 mm"
    assert all(abs(r["error_m"] - 1e-3) < 1e-9 for r in rows_bad), rows_bad
    checked += 1

    # 4 -- and it must NOT fail on a shift well inside the tolerance, or it
    # would fail every honest export. 0.01 mm is a tenth of the tolerance.
    ok_small, _ = frame_guard(
        [(x + 1e-5, y + 1e-5, z + 1e-5) for x, y, z in on_frame], planes, 1e-4
    )
    assert ok_small, "the frame guard rejected a 0.01 mm deviation inside its own tolerance"
    checked += 1

    # 5 -- a cube far from both boxes: every survivor clears, verdict OUTSIDE.
    pts, counts, idx = topo._cube(scale=1.0, drop_faces=1)
    far = [(x + 50.0, y + 50.0, z + 50.0) for x, y, z in pts]
    rows = edge_clearances(topo, topo.survivor_edges(far, counts, idx), boxes)
    assert len(rows) == 3, rows
    assert verdict_of(rows) == "EDGES_OUTSIDE_SWEPT_VOLUME"
    assert rows[0]["min_distance"] > 40.0
    checked += 1

    # 6 -- THE MUTATION that matters most: move the SAME cracked cube into the
    # pocket cavity. The three open edges now sit inside the swept volume and
    # the verdict MUST flip. Without this, a green run proves only that the
    # arithmetic ran.
    inside = [(x * 0.1, y * 0.1, z * 0.1 - 1.0) for x, y, z in pts]
    rows_in = edge_clearances(topo, topo.survivor_edges(inside, counts, idx), boxes)
    assert verdict_of(rows_in) == "EDGES_IN_SWEPT_VOLUME", rows_in
    assert rows_in[0]["min_distance"] == 0.0
    checked += 1

    # 7 -- a CLOSED mesh has no survivors at all, and that is its own verdict
    # rather than a silent pass through an empty loop.
    pts, counts, idx = topo._cube()
    assert verdict_of(edge_clearances(topo, topo.survivor_edges(pts, counts, idx), boxes)) \
        == "NO_OPEN_EDGES"
    checked += 1

    # 8 -- the sort really is closest-first, so the printed head is the head
    # that decides. Two cracked cubes at different distances.
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
                        help="Offline counter-proof of the decisions. Needs no OBJ.")
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
