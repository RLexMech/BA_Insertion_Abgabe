# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the square-pocket table USD from code (D-033) -- no CAD import.

The table is COMPOSED, not carved: nine axis-aligned cuboids (four plate
pieces around the pocket, the pocket floor, four legs), each a
``UsdGeom.Cube`` carrying its own ``UsdPhysics.CollisionAPI``. Boxes are
native PhysX primitive colliders, so there is no mesh approximation step at
all -- the sealed-pocket failure mode of a convex hull over a non-convex
blind pocket (D-029 risk 1) is impossible by construction, not merely
checked. Each pocket wall face is a single flat box face, so the peg slides
on plane geometry with seams only in the four corners, where a real pocket
has corners anyway.

The asset origin sits ON the opening plane AT the pocket centre, exactly the
convention `proxytask_tasks_cfg.py` derives from (D-029): spawned at
(0, -0.225, 0.755) the plate top lands at z = 0.755 and the legs on the
ground. The world-space expectations of `verify_fixture_usd.py` (bbox
0.500 x 0.600 x 0.755, z range -0.755 .. 0.000) and `verify_fixture_spawn.py`
are unchanged.

The pocket side length is the curriculum variable now (D-033): one welded
30 mm peg, one generated table per rung. The output filename carries the
size (``tisch_square_b45.usd``), so rungs cannot overwrite each other and
the resolver's name-is-contract rule keeps a stale table impossible to load
silently.

UNVERIFIED -- authored on the dev PC (no Isaac installation). On the training
machine:

    conda activate env_isaaclab
    python scripts\\author_tisch_square.py                     # target rung, 32 mm
    python scripts\\author_tisch_square.py --pocket-side-mm 45  # stage 0
"""

from __future__ import annotations

import argparse
import json
import pathlib

from isaaclab.app import AppLauncher

# Bumped whenever this file changes. The two-machine workflow copies scripts by
# hand, and a stale copy has already cost two "it does not work" detours; the
# first line of output makes the copy's identity visible before anything else.
SCRIPT_MARKER = "author_tisch_square-2026-08-05b"

parser = argparse.ArgumentParser(description="Generate the square-pocket table USD (D-033).")
parser.add_argument("--pocket-side-mm", type=float, default=None,
                    help="Pocket opening edge length [mm]; default is the cfg value "
                         "(itself overridable via PROXYTASK_POCKET_SIDE_MM).")
parser.add_argument("--out", type=str, default=None,
                    help="Output USD; default is the package assets/Tisch path with the size in the name.")
parser.add_argument("--dry-run", action="store_true", help="Report what would be authored, write nothing.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

print(f"[author_tisch] marker: {SCRIPT_MARKER}; sections: plan, author, verify, write")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pxr and the proxytask modules are importable only after the app is up.
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from proxytask.tasks.direct.proxytask import proxytask_tasks_cfg as geom  # noqa: E402

DEFAULT_PRIM = "tisch_square"

# Plate footprint in the asset frame (origin = opening plane at the pocket
# centre; the plate centre sits at +OPENING_OFFSET_Y from it, i.e. +0.225).
PLATE_X = (-geom.PLATE_HALF_EXTENTS[0], geom.PLATE_HALF_EXTENTS[0])          # -0.250 .. 0.250
# The pocket sits at plate-local y = OPENING_OFFSET_Y = -0.225; with the
# asset origin ON the pocket centre, the plate centre lands at the negated
# offset and the plate spans -0.075 .. +0.525.
_PLATE_CENTRE_Y = -geom.OPENING_OFFSET_Y
PLATE_Y = (_PLATE_CENTRE_Y - geom.PLATE_HALF_EXTENTS[1], _PLATE_CENTRE_Y + geom.PLATE_HALF_EXTENTS[1])
PLATE_Z = (-geom.PLATE_THICKNESS, 0.0)                                       # -0.055 .. 0.000
TABLE_HEIGHT = 0.755
LEG_SECTION = 0.050
LEG_INSET = 0.025  # legs sit fully inside the plate footprint


def box_specs(pocket_side: float, pocket_depth: float) -> list[dict]:
    """The nine cuboids as {name, min, max} axis-aligned boxes, asset-local.

    The four plate pieces tile the plate exactly (no overlaps, no gaps): the
    south/north pieces span the full x width, the west/east pieces fill the
    remaining band beside the pocket. Each pocket wall face therefore belongs
    to exactly ONE box and is a single plane.
    """
    h = pocket_side / 2.0
    x0, x1 = PLATE_X
    y0, y1 = PLATE_Y
    z0, z1 = PLATE_Z
    floor_top = -pocket_depth
    leg_z = (-TABLE_HEIGHT, z0)
    leg_xs = (x0 + LEG_INSET, x1 - LEG_SECTION - LEG_INSET)
    leg_ys = (y0 + LEG_INSET, y1 - LEG_SECTION - LEG_INSET)
    boxes = [
        {"name": "plate_south", "min": (x0, y0, z0), "max": (x1, -h, z1)},
        {"name": "plate_north", "min": (x0, h, z0), "max": (x1, y1, z1)},
        {"name": "plate_west", "min": (x0, -h, z0), "max": (-h, h, z1)},
        {"name": "plate_east", "min": (h, -h, z0), "max": (x1, h, z1)},
        {"name": "pocket_floor", "min": (-h, -h, z0), "max": (h, h, floor_top)},
    ]
    for i, lx in enumerate(leg_xs):
        for j, ly in enumerate(leg_ys):
            boxes.append({
                "name": f"leg_{'xp' if i else 'xn'}{'yp' if j else 'yn'}",
                "min": (lx, ly, leg_z[0]),
                "max": (lx + LEG_SECTION, ly + LEG_SECTION, leg_z[1]),
            })
    return boxes


def author(stage: Usd.Stage, boxes: list[dict]) -> None:
    """One scaled UsdGeom.Cube per box, each with a primitive box collider.

    The default prim stays op-free (asset contract): Isaac Lab's spawner
    authors [translate, orient, scale] on the referencing prim, which REPLACES
    the referenced xformOpOrder -- any op here would be silently dropped, the
    2026-07-26 failure. All placement lives one level down, on the cubes.
    """
    root = UsdGeom.Xform.Define(stage, f"/{DEFAULT_PRIM}")
    stage.SetDefaultPrim(root.GetPrim())
    for box in boxes:
        lo, hi = box["min"], box["max"]
        centre = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
        size = [hi[i] - lo[i] for i in range(3)]
        cube = UsdGeom.Cube.Define(stage, f"/{DEFAULT_PRIM}/{box['name']}")
        # A unit cube (size 1 spans -0.5..0.5) scaled per axis: the scale IS
        # the edge length, which keeps the sidecar trivially readable.
        cube.CreateSizeAttr().Set(1.0)
        # Authored via the schema accessor rather than GetAttribute("extent"):
        # extent is the unit cube's, pre-transform, and CreateExtentAttr is the
        # form that is guaranteed to author rather than silently no-op.
        cube.CreateExtentAttr().Set(
            [Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)]
        )
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
        xf.AddScaleOp().Set(Gf.Vec3f(*[float(s) for s in size]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def verify(stage: Usd.Stage, pocket_side: float, pocket_depth: float) -> tuple[dict, dict]:
    """Measure the authored stage back and judge it. Checks mirror the asset
    contract plus the pocket dimensions, read from the prims rather than from
    the inputs, so a coding slip in author() cannot self-certify."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def bbox(prim_path: str):
        rng = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        return [list(rng.GetMin()), list(rng.GetMax())]

    whole = bbox(f"/{DEFAULT_PRIM}")
    size = [whole[1][i] - whole[0][i] for i in range(3)]
    # Opening measured between the INNER faces of the four plate pieces.
    east = bbox(f"/{DEFAULT_PRIM}/plate_east")
    west = bbox(f"/{DEFAULT_PRIM}/plate_west")
    north = bbox(f"/{DEFAULT_PRIM}/plate_north")
    south = bbox(f"/{DEFAULT_PRIM}/plate_south")
    floor = bbox(f"/{DEFAULT_PRIM}/pocket_floor")
    opening_x = east[0][0] - west[1][0]
    opening_y = north[0][1] - south[1][1]
    measured_depth = 0.0 - floor[1][2]

    default_prim = stage.GetDefaultPrim()
    op_attr = default_prim.GetAttribute("xformOpOrder")
    root_ops = list(op_attr.Get() or []) if op_attr else []
    collision_count = sum(
        1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
    )
    mesh_collision_count = sum(
        1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.MeshCollisionAPI)
    )
    instanceable = [str(p.GetPath()) for p in stage.Traverse() if p.IsInstanceable()]

    measured = {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "root_xform_ops": [str(o) for o in root_ops],
        "bbox_min": whole[0],
        "bbox_max": whole[1],
        "bbox_size": size,
        "opening_x_m": opening_x,
        "opening_y_m": opening_y,
        "pocket_depth_m": measured_depth,
        "collision_prims": collision_count,
        "mesh_collision_prims": mesh_collision_count,
        "instanceable_prims": instanceable,
    }
    tol = 1e-6
    checks = {
        "meters_per_unit_is_1": abs(measured["meters_per_unit"] - 1.0) < 1e-9,
        "up_axis_is_z": measured["up_axis"] == "Z",
        "default_prim_op_free": not root_ops,
        "bbox_extents_ok": all(abs(size[i] - e) < 1e-4 for i, e in enumerate((0.500, 0.600, 0.755))),
        "bbox_z_range_ok": abs(whole[0][2] - (-0.755)) < 1e-4 and abs(whole[1][2]) < 1e-4,
        "opening_is_square_and_right": abs(opening_x - pocket_side) < tol and abs(opening_y - pocket_side) < tol,
        "pocket_depth_right": abs(measured_depth - pocket_depth) < tol,
        "all_nine_boxes_collide": collision_count == 9,
        # Box primitives need no approximation; a MeshCollisionAPI here would
        # mean something was authored as a mesh after all.
        "no_mesh_collision_api": mesh_collision_count == 0,
        "no_instanceable_prims": not instanceable,
    }
    return measured, checks


def main() -> None:
    pocket_side = (
        args_cli.pocket_side_mm / 1000.0 if args_cli.pocket_side_mm is not None else geom.POCKET_SIDE
    )
    pocket_depth = geom.POCKET_DEPTH
    if not geom.PEG_SIDE < pocket_side <= 0.060:
        raise SystemExit(
            f"pocket side {pocket_side*1000:g} mm is outside ({geom.PEG_SIDE*1000:g}, 60] mm: "
            "at or below the peg side nothing can insert, far above it the plate pieces vanish"
        )

    name = geom.default_fixture_usd_name(pocket_side)
    out_path = pathlib.Path(args_cli.out) if args_cli.out else (
        pathlib.Path(geom.__file__).resolve().parent / "assets" / "Tisch" / name
    )
    boxes = box_specs(pocket_side, pocket_depth)

    print(f"[author_tisch] pocket {pocket_side*1000:g} x {pocket_side*1000:g} x "
          f"{pocket_depth*1000:g} mm blind; peg {geom.PEG_SIDE*1000:g} mm -> "
          f"clearance {(pocket_side - geom.PEG_SIDE)/2*1000:.2f} mm per axis")
    for box in boxes:
        print(f"[author_tisch]   {box['name']:<14} min {box['min']}  max {box['max']}")
    if args_cli.dry_run:
        print("[author_tisch] --dry-run: nothing written.")
        return

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    author(stage, boxes)
    measured, checks = verify(stage, pocket_side, pocket_depth)

    verdict = "PASS" if all(checks.values()) else "FAIL"
    for cname, ok in checks.items():
        print(f"[author_tisch]   {'PASS' if ok else 'FAIL'}  {cname}")
    print(f"[author_tisch] verdict: {verdict}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out_path))
    sidecar = out_path.with_suffix(".author.json")
    sidecar.write_text(json.dumps({
        "marker": SCRIPT_MARKER,
        "verdict": verdict,
        "pocket_side_m": pocket_side,
        "pocket_depth_m": pocket_depth,
        "peg_side_m": geom.PEG_SIDE,
        "boxes": boxes,
        "measured": measured,
        "checks": checks,
    }, indent=2))
    print(f"[author_tisch] wrote {out_path}")
    print(f"[author_tisch] wrote {sidecar}")
    if verdict != "PASS":
        raise SystemExit("[author_tisch] asset written but FAILED its own checks -- do not use it")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
