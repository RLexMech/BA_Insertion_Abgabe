# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Export a USD asset's mesh as a Wavefront OBJ. Runs on the training PC only.

WHY
---
``insertion_sdf.py`` -- the Warp half of M2.2 -- needs the part and the fixture
as OBJ files, because that is what the published pattern consumes. Isaac Lab
2.3.2 ``isaaclab_tasks/direct/automate/industreal_algo_utils.py::
load_asset_mesh_in_warp`` loads an OBJ with ``trimesh``, hands its vertices and
faces to ``wp.Mesh`` and samples the surface with
``trimesh.sample.sample_surface_even``. It never reads the USD stage.

WHAT IT WRITES, AND IN WHOSE FRAME
----------------------------------
The OBJ carries the mesh expressed in ``--frame-prim``'s frame, and choosing
that frame is the whole point of the script. The env reads the part's pose at
``peg_body_name = TOOL_LINK_NAME``, not at the part prim, so the part mesh has
to be expressed at ``tool_link`` or every SDF distance is off by the constant
transform between the two. Exporting from the USD gives that for free;
exporting from the STEP would mean applying the FUEGETEIL -> tool_link
transform by hand.

The relative transform is RIGID and pose-independent -- the part is welded to
``tool_link`` -- so the joints' rest pose does not enter the result. It is
exact, not an approximation.

Units are METRES, because the stage is ``metersPerUnit 1.0`` (RT-36) and
``trimesh`` performs no conversion. The script REFUSES a stage that says
anything else rather than scaling it silently.

THIS SCRIPT WRITES NOTHING TO A USD. It opens the stage read-only.

The arithmetic is not here. Transforming points into another frame, merging
several mesh prims and writing the OBJ text all live in
``scripts/tools/mesh_export.py``, which has no ``pxr`` and is proved on the
laptop (``--self-test``, 10 cases). This file collects numbers and reports.

    .\scripts\rt_log.ps1 RT-85 python scripts\export_asset_mesh.py --usd source\insertion\insertion\tasks\direct\insertion\assets\Robot\ur5e_tool.usd --prim FUEGETEIL --frame-prim tool_link --out source\insertion\insertion\tasks\direct\insertion\assets\Robot\fuegeteil.obj --expect-part-bbox

    .\scripts\rt_log.ps1 RT-86 python scripts\export_asset_mesh.py --usd source\insertion\insertion\tasks\direct\insertion\assets\Werkzeug\Aufnahme_real_v1_mm.usd --out source\insertion\insertion\tasks\direct\insertion\assets\Werkzeug\aufnahme.obj
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import traceback

from isaaclab.app import AppLauncher

# Loaded by path because scripts/tools is not a package -- the idiom
# verify_fixture_usd.py and author_tool_ur5e.py already use.
_TOOLS = pathlib.Path(__file__).resolve().parent / "tools"


def _load_tool(stem: str):
    spec = importlib.util.spec_from_file_location(stem, _TOOLS / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod


exit_with = _load_tool("isaac_exit").exit_with
me = _load_tool("mesh_export")

SCRIPT_MARKER = "export_asset_mesh-2026-08-29a"

parser = argparse.ArgumentParser(description="Export a USD asset's mesh as an OBJ for Warp.")
parser.add_argument("--usd", type=str, required=True, help="Path to the source USD. Opened read-only.")
parser.add_argument(
    "--prim",
    type=str,
    default=None,
    help=(
        "Root to collect meshes under: a full prim PATH if it starts with '/', "
        "otherwise a prim NAME that must occur exactly once. Default: the whole "
        "stage."
    ),
)
parser.add_argument(
    "--frame-prim",
    type=str,
    default=None,
    help=(
        "Express the points in THIS prim's frame (same path-or-name rule). "
        "Default: the stage's own world frame. For the part this must be "
        "tool_link -- the body whose pose the env reads."
    ),
)
parser.add_argument("--out", type=str, required=True, help="Where to write the .obj.")
parser.add_argument(
    "--expect-part-bbox",
    action="store_true",
    help=(
        "Cross-check the exported extents against insertion_tasks_cfg.PART_BBOX_M "
        "(RT-12, measured on the imported USD with a different tool in a different "
        "frame). Compared SORTED: this frame differs from the measured one by a "
        "rotation about z and possibly a 180 deg flip, both of which permute the "
        "axes without changing the lengths. Mismatch beyond --tol exits 1."
    ),
)
parser.add_argument(
    "--tol",
    type=float,
    default=1.0e-4,
    help="Tolerance for --expect-part-bbox, in metres (default 0.1 mm).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# GUARDED for the same reason scripted_insert.py guards its imports: a failure
# here runs BEFORE __main__ and would print a traceback while the wrapper still
# reported exit 0 (RT-40). The import path leaves through the same door.
try:
    from pxr import Gf, Usd, UsdGeom  # noqa: E402
except BaseException:
    traceback.print_exc()
    exit_with(simulation_app, 1, tag="export_asset_mesh-import")


def resolve_prim(stage, spec: str):
    """A prim from a PATH ('/a/b') or from a NAME that must be unique.

    Ambiguity is an error, not a first match. Two prims called ``FUEGETEIL``
    would make the export depend on traversal order, and a mesh exported from
    the wrong one of two identical names is exactly the defect no downstream
    number could reveal.
    """
    if spec.startswith("/"):
        prim = stage.GetPrimAtPath(spec)
        if not prim or not prim.IsValid():
            raise SystemExit(f"[export_asset_mesh] no prim at path {spec!r}")
        return prim
    hits = [p for p in stage.Traverse() if p.GetName() == spec]
    if not hits:
        raise SystemExit(
            f"[export_asset_mesh] no prim named {spec!r} on this stage. "
            "Run scripts/list_usd_prims.py to see what is there."
        )
    if len(hits) > 1:
        paths = ", ".join(str(p.GetPath()) for p in hits)
        raise SystemExit(
            f"[export_asset_mesh] {len(hits)} prims are named {spec!r}: {paths}. "
            "Pass a full path instead -- picking one by traversal order would "
            "make the export depend on the file's layout."
        )
    return hits[0]


def to_column_major(m) -> list:
    """``Gf.Matrix4d`` (ROW-vector, ``p * M``) as a column-vector 4x4 list.

    USD multiplies a point on the LEFT; ``mesh_export.transform_points``
    multiplies on the right. The two conventions are transposes of each other,
    so the transpose happens exactly once, here, in the open. A convention
    applied twice -- or not at all -- yields a valid OBJ of a part in the wrong
    orientation, and nothing downstream can see it.
    """
    return [[float(m[c][r]) for c in range(4)] for r in range(4)]


def main() -> int:
    usd_path = pathlib.Path(args_cli.usd).resolve()
    out_path = pathlib.Path(args_cli.out).resolve()
    print(f"[export_asset_mesh] marker: {SCRIPT_MARKER}")
    print(f"[export_asset_mesh] usd: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise SystemExit(f"[export_asset_mesh] could not open stage: {usd_path}")

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)
    print(f"[export_asset_mesh] metersPerUnit {mpu}, up axis {up}")
    if abs(mpu - 1.0) > 1e-9:
        raise SystemExit(
            f"[export_asset_mesh] the stage says metersPerUnit {mpu}, not 1.0. "
            "trimesh performs no unit conversion, so the OBJ would silently be "
            "in the wrong unit. Repair the stage with scripts/fix_stage_units.py "
            "(D-067); nothing is scaled here."
        )
    if str(up) != "Z":
        raise SystemExit(
            f"[export_asset_mesh] the stage up axis is {up}, not Z. A Y-up asset "
            "on a Z-up stage is tilted 90 degrees (D-016); repair the stage, do "
            "not rotate here."
        )

    root = resolve_prim(stage, args_cli.prim) if args_cli.prim else stage.GetPseudoRoot()
    print(f"[export_asset_mesh] collecting meshes under {root.GetPath()}")

    time = Usd.TimeCode.Default()
    if args_cli.frame_prim:
        frame_prim = resolve_prim(stage, args_cli.frame_prim)
        frame_x = UsdGeom.Xformable(frame_prim)
        if not frame_x:
            raise SystemExit(
                f"[export_asset_mesh] {frame_prim.GetPath()} is not Xformable, so it "
                "carries no frame to express the points in."
            )
        frame_inv = frame_x.ComputeLocalToWorldTransform(time).GetInverse()
        print(f"[export_asset_mesh] target frame: {frame_prim.GetPath()}")
    else:
        frame_inv = Gf.Matrix4d(1.0)
        print("[export_asset_mesh] target frame: the stage's world frame")

    parts = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get(time)
        counts = mesh.GetFaceVertexCountsAttr().Get(time)
        idx = mesh.GetFaceVertexIndicesAttr().Get(time)
        if pts is None or counts is None or idx is None:
            raise SystemExit(
                f"[export_asset_mesh] {prim.GetPath()} is a Mesh but is missing "
                "points, faceVertexCounts or faceVertexIndices. Nothing written."
            )
        # local -> world -> target frame, in USD's own row-vector order, then
        # transposed once for the pure helper.
        rel = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time) * frame_inv
        moved = me.transform_points([(p[0], p[1], p[2]) for p in pts], to_column_major(rel))
        parts.append((moved, [int(c) for c in counts], [int(i) for i in idx]))
        print(f"[export_asset_mesh]   {prim.GetPath()}: "
              f"{len(moved)} vertices, {len(counts)} faces")

    if not parts:
        raise SystemExit(
            f"[export_asset_mesh] no UsdGeom.Mesh under {root.GetPath()}. "
            "Nothing written. Check --prim against scripts/list_usd_prims.py."
        )

    points, counts, indices = me.merge_meshes(parts)
    n_tris = me.require_triangles(counts)
    faces = me.faces_from_flat(counts, indices)
    bb = me.bbox(points)

    print(f"[export_asset_mesh] merged {len(parts)} mesh prim(s): "
          f"{len(points)} vertices, {n_tris} triangles")
    print("[export_asset_mesh] extents mm: "
          f"x {bb['extent'][0] * 1000.0:.4f}  "
          f"y {bb['extent'][1] * 1000.0:.4f}  "
          f"z {bb['extent'][2] * 1000.0:.4f}")
    print("[export_asset_mesh] min mm: "
          + "  ".join(f"{v * 1000.0:+.4f}" for v in bb["min"])
          + "   max mm: "
          + "  ".join(f"{v * 1000.0:+.4f}" for v in bb["max"]))

    verdict = 0
    if args_cli.expect_part_bbox:
        # Imported here and not at module scope: the cross-check is optional
        # and this file is otherwise task-agnostic, so a machine without the
        # package can still run the fixture export.
        from insertion.tasks.direct.insertion import insertion_tasks_cfg  # noqa: E402

        want = sorted(float(v) for v in insertion_tasks_cfg.PART_BBOX_M)
        got = sorted(bb["extent"])
        worst = max(abs(a - b) for a, b in zip(want, got))
        ok = worst <= args_cli.tol
        print("[export_asset_mesh] PART_BBOX_M cross-check (sorted extents, mm): "
              f"want {[round(v * 1000.0, 4) for v in want]} "
              f"got {[round(v * 1000.0, 4) for v in got]} "
              f"worst {worst * 1000.0:.4f} mm vs tol {args_cli.tol * 1000.0:.4f} mm "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("[export_asset_mesh]   The exported mesh is not the part this repo "
                  "measured in RT-12. Do NOT adjust the tolerance: either the wrong "
                  "prim was collected, or the frame composition is wrong, or the "
                  "asset moved. Nothing downstream may use this file.")
            verdict = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"exported by {SCRIPT_MARKER}\n"
        f"source usd: {usd_path}\n"
        f"prim root: {root.GetPath()}\n"
        f"frame: {args_cli.frame_prim or 'stage world'}\n"
        f"units: metres (stage metersPerUnit {mpu})\n"
        f"{len(points)} vertices, {n_tris} triangles, merged from {len(parts)} mesh prim(s)"
    )
    out_path.write_text(me.obj_text(points, faces, header=header), encoding="utf-8")
    print(f"[export_asset_mesh] obj -> {out_path}")
    if verdict:
        print("[export_asset_mesh] WRITTEN BUT REJECTED: the cross-check failed. "
              "The file is on disk so it can be inspected, not so it can be used.")
    return verdict


if __name__ == "__main__":
    # D-081: the exit status is set BEFORE the shutdown. close() does not
    # return (RT-22), so anything after it decides nothing.
    _code = 1
    try:
        _code = main()
    except SystemExit as _exc:
        # SystemExit's code is a MESSAGE here, not a number: every fail-fast
        # above raises SystemExit(str). int() on that raises ValueError and
        # would turn a clean refusal into a traceback, so the two cases are
        # separated instead of assumed.
        if _exc.code is None:
            _code = 0
        elif isinstance(_exc.code, int):
            _code = _exc.code
        else:
            print(_exc.code)
            _code = 1
    except BaseException:
        traceback.print_exc()
        _code = 1
    exit_with(simulation_app, _code, tag="export_asset_mesh")
