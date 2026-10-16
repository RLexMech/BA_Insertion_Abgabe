# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the real workcell USDs: the two table plates, and the fixture block."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from isaaclab.app import AppLauncher

SCRIPT_MARKER = "author_workcell-2026-08-24b"

parser = argparse.ArgumentParser(description="Generate the real workcell USDs (D-058, D-060).")
parser.add_argument("--out-dir", type=str, default=None,
                    help="Directory for both USDs. Default is the package assets/Arbeitszelle "
                         "path. Existing files are NEVER overwritten without --force.")
parser.add_argument("--force", action="store_true",
                    help="Allow replacing existing output files.")
parser.add_argument("--dry-run", action="store_true",
                    help="Report what would be authored, write nothing.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

print(f"[author_workcell] marker: {SCRIPT_MARKER}; sections: plan, author, verify, write")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "source" / "insertion"))
import insertion  # noqa: E402
print(f"[author_workcell] insertion package imported from: {pathlib.Path(insertion.__file__).resolve().parent}")
from insertion.tasks.direct.insertion import insertion_tasks_cfg as geom  # noqa: E402

TABLES_PRIM = "arbeitszelle_tische"
BLOCK_PRIM = "aufnahme_block"

TOL = 1e-6


def table_boxes() -> list[dict]:
    """The two plates, asset-local == env-local."""
    return [
        {"name": "tisch1_platte",
         "centre": geom.TABLE1_POS,
         "size": (geom.T1_SIZE_X, geom.T1_SIZE_Y, geom.TABLE_PLATE_T)},
        {"name": "tisch2_platte",
         "centre": geom.TABLE2_POS,
         "size": (geom.T2_SIZE_X, geom.T2_SIZE_Y, geom.TABLE_PLATE_T)},
    ]


def block_boxes() -> list[dict]:
    """The five boxes around the pocket, asset-local to the STAGE-2 OPENING PLANE."""
    return geom.block_recess_boxes()


def author(stage: Usd.Stage, default_prim: str, boxes: list[dict]) -> None:
    """One scaled UsdGeom.Cube per box, each with a primitive box collider."""
    root = UsdGeom.Xform.Define(stage, f"/{default_prim}")
    stage.SetDefaultPrim(root.GetPrim())
    for box in boxes:
        cube = UsdGeom.Cube.Define(stage, f"/{default_prim}/{box['name']}")
        cube.CreateSizeAttr().Set(1.0)
        cube.CreateExtentAttr().Set(
            [Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)]
        )
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in box["centre"]]))
        xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in box["size"]]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def stage_facts(stage: Usd.Stage, default_prim: str) -> dict:
    """Contract facts every asset here must satisfy, read from the PRIMS."""
    prim = stage.GetDefaultPrim()
    op_attr = prim.GetAttribute("xformOpOrder")
    root_ops = list(op_attr.Get() or []) if op_attr else []
    return {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "default_prim": str(prim.GetPath()) if prim else None,
        "root_xform_ops": [str(o) for o in root_ops],
        "collision_prims": sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)),
        "mesh_collision_prims": sum(
            1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.MeshCollisionAPI)),
        "instanceable_prims": [str(p.GetPath()) for p in stage.Traverse() if p.IsInstanceable()],
    }


def contract_checks(facts: dict, default_prim: str, expected_colliders: int) -> dict:
    return {
        "meters_per_unit_is_1": abs(facts["meters_per_unit"] - 1.0) < 1e-9,
        "up_axis_is_z": facts["up_axis"] == "Z",
        "default_prim_is_expected": facts["default_prim"] == f"/{default_prim}",
        "default_prim_op_free": not facts["root_xform_ops"],
        "collider_count_right": facts["collision_prims"] == expected_colliders,
        "no_mesh_collision_api": facts["mesh_collision_prims"] == 0,
        "no_instanceable_prims": not facts["instanceable_prims"],
    }


def bbox_fn(stage: Usd.Stage):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )

    def bbox(prim_path: str):
        rng = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
        return [list(rng.GetMin()), list(rng.GetMax())]

    return bbox


def verify_tables(stage: Usd.Stage) -> tuple[dict, dict]:
    bbox = bbox_fn(stage)
    t1 = bbox(f"/{TABLES_PRIM}/tisch1_platte")
    t2 = bbox(f"/{TABLES_PRIM}/tisch2_platte")
    t1_top, t2_top = t1[1][2], t2[1][2]

    facts = stage_facts(stage, TABLES_PRIM)
    facts.update({"tisch1_top_z": t1_top, "tisch2_top_z": t2_top,
                  "plate_drop_m": t1_top - t2_top})

    checks = contract_checks(facts, TABLES_PRIM, expected_colliders=2)
    checks.update({
        "plate_drop_matches_measurement": abs((t1_top - t2_top) - geom.PLATE_TOP_DROP) < 1e-5,
        "tisch1_top_matches_cfg": abs(t1_top - geom.T1_TOP_Z) < TOL,
        "tisch2_top_matches_cfg": abs(t2_top - geom.T2_TOP_Z) < TOL,
        "robot_foot_above_tisch1": t1_top < 0.0,
    })
    return facts, checks


def verify_block(stage: Usd.Stage) -> tuple[dict, dict]:
    """Read the five recess boxes back off the PRIMS and pin their asymmetry."""
    bbox = bbox_fn(stage)
    lo, hi = bbox(f"/{BLOCK_PRIM}")
    walls = {b["name"]: bbox(f"/{BLOCK_PRIM}/{b['name']}") for b in block_boxes()}
    rim_boxes = [n for n in walls if n not in ("block_rueckwand", "block_boden")]

    facts = stage_facts(stage, BLOCK_PRIM)
    facts.update({
        "bbox_min": lo,
        "bbox_max": hi,
        "box_bboxes": walls,
        "box_sizes_mm": {b["name"]: [v * 1000.0 for v in b["size"]] for b in block_boxes()},
    })

    checks = contract_checks(facts, BLOCK_PRIM, expected_colliders=8)
    checks.update({
        "near_face_plus_80mm_is_stage1":
            abs((lo[0] + geom.BLOCK_FRONT_TO_STAGE1) - geom.POCKET_STAGE1_X_RANGE[0]) < TOL,
        "model_depth_is_block_plus_rear_wall":
            abs((hi[0] - lo[0]) - geom.BLOCK_MODEL_SIZE_X) < TOL,
        "width_is_block_width": abs((hi[1] - lo[1]) - geom.BLOCK_SIZE_Y) < TOL,
        "walls_top_at_stage1_depth":
            all(abs(walls[n][1][2] - geom.STAGE1_DEPTH) < TOL for n in rim_boxes),
        "rear_wall_stands_above_rim":
            abs(hi[2] - (geom.STAGE1_DEPTH + geom.BLOCK_REAR_WALL_H)) < TOL,
        "bottom_at_height_below_rim": abs(lo[2] + (geom.BLOCK_HEIGHT - geom.STAGE1_DEPTH)) < TOL,
        "recess_x_matches_cutout":
            abs(walls["block_wand_x_minus"][1][0] - geom.POCKET_LOCAL_X_RANGE[0]) < TOL,
        "recess_x_plus_matches_cutout":
            abs(walls["block_wand_x_plus"][0][0] - geom.POCKET_LOCAL_X_RANGE[1]) < TOL,
        "recess_y_minus_matches_cutout":
            abs(walls["block_wand_y_minus"][1][1] - geom.POCKET_LOCAL_Y_RANGE[0]) < TOL,
        "recess_y_plus_matches_cutout":
            abs(walls["block_wand_y_plus"][0][1] - geom.POCKET_LOCAL_Y_RANGE[1]) < TOL,
        "recess_floor_matches_cutout":
            abs(walls["block_boden"][1][2] - geom.POCKET_ASSET_BOTTOM_Z) < TOL,
        "no_box_has_negative_size":
            all(min(b["size"]) > 0.0 for b in block_boxes()),
        "not_centred_in_x": abs(lo[0] + hi[0]) > 1e-4,
        "not_centred_in_y": abs(lo[1] + hi[1]) > 1e-4,
        "rear_wall_is_the_measured_thickness":
            abs((walls["block_rueckwand"][1][0] - walls["block_rueckwand"][0][0])
                - geom.BLOCK_REAR_WALL_T) < TOL,
        "block_stands_proud_of_the_cutout":
            abs((walls["block_wand_x_plus"][1][0] - walls["block_wand_x_plus"][0][0])
                - geom.BLOCK_BEHIND_POCKET_X) < TOL
            and geom.BLOCK_BEHIND_POCKET_X > 0.0,
        "plus_x_wall_spans_the_full_height":
            abs(walls["block_wand_x_plus"][1][2] - geom.STAGE1_DEPTH) < TOL
            and abs(walls["block_wand_x_plus"][0][2]
                    - (geom.STAGE1_DEPTH - geom.BLOCK_HEIGHT)) < TOL,
        "slot_minus_x_spans_recess_edge_to_tab":
            abs(walls["block_spalt_y_plus_minus_x"][0][0] - geom.POCKET_LOCAL_X_RANGE[0]) < TOL
            and abs(walls["block_spalt_y_plus_minus_x"][1][0] - geom.POCKET_TAB_X_RANGE[0]) < TOL,
        "slot_plus_x_spans_tab_to_recess_edge":
            abs(walls["block_spalt_y_plus_plus_x"][0][0] - geom.POCKET_TAB_X_RANGE[1]) < TOL
            and abs(walls["block_spalt_y_plus_plus_x"][1][0] - geom.POCKET_LOCAL_X_RANGE[1]) < TOL,
        "slots_span_body_edge_to_bbox_edge":
            all(abs(walls[n][0][1] - geom.POCKET_BODY_Y_MAX) < TOL
                and abs(walls[n][1][1] - geom.POCKET_LOCAL_Y_RANGE[1]) < TOL
                for n in ("block_spalt_y_plus_minus_x", "block_spalt_y_plus_plus_x")),
    })
    return facts, checks


def cross_checks() -> dict:
    """Relations BETWEEN the two assets, which neither stage can see alone."""
    spawn_z = geom.WORKCELL_BLOCK_POS[2]
    return {
        "block_bottom_meets_tisch2_top":
            abs((spawn_z - (geom.BLOCK_HEIGHT - geom.STAGE1_DEPTH)) - geom.T2_TOP_Z) < 1e-5,
        "block_top_is_guide_edge": abs((spawn_z + geom.STAGE1_DEPTH) - geom.BLOCK_TOP_Z) < TOL,
        "spawn_is_insertion_plane": abs(spawn_z - geom.INSERTION_PLANE_Z) < TOL,
        "block_top_above_robot_foot": spawn_z + geom.STAGE1_DEPTH > 0.0,
        "tables_spawn_at_env_origin": geom.WORKCELL_TABLES_POS == (0.0, 0.0, 0.0),
        "spawn_is_pocket_origin_x": abs(geom.WORKCELL_BLOCK_POS[0] - geom.POCKET_ORIGIN_X) < TOL,
        "spawn_is_pocket_origin_y": abs(geom.WORKCELL_BLOCK_POS[1] - geom.POCKET_ORIGIN_Y) < TOL,
        "block_and_fixture_share_the_spawn":
            geom.WORKCELL_BLOCK_POS == geom.WORKCELL_ENTRANCE_POS,
    }


def build(default_prim: str, boxes: list[dict], verifier) -> tuple[Usd.Stage, dict, dict]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    author(stage, default_prim, boxes)
    facts, checks = verifier(stage)
    return stage, facts, checks


def main() -> None:
    out_dir = (
        pathlib.Path(args_cli.out_dir) if args_cli.out_dir
        else pathlib.Path(geom.__file__).resolve().parent / "assets" / "Arbeitszelle"
    )
    targets = [
        ("tables", TABLES_PRIM, table_boxes(), verify_tables,
         out_dir / geom.WORKCELL_TABLES_USD_NAME, geom.WORKCELL_TABLES_POS),
        ("block", BLOCK_PRIM, block_boxes(), verify_block,
         out_dir / geom.WORKCELL_BLOCK_USD_NAME, geom.WORKCELL_BLOCK_POS),
    ]

    print("[author_workcell] frame: UR5e base_link origin = (0,0,0); table tops are NEGATIVE")
    for label, _prim, boxes, _v, path, spawn in targets:
        print(f"[author_workcell] {label}: spawns at "
              f"({spawn[0]:+.4f}, {spawn[1]:+.4f}, {spawn[2]:+.4f}) -> {path}")
        for box in boxes:
            c, s = box["centre"], box["size"]
            print(f"[author_workcell]     {box['name']:<16} centre "
                  f"({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})  size "
                  f"({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})")
    for item in geom.WORKCELL_PENDING:
        print(f"[author_workcell] PENDING {item}")

    cross = cross_checks()
    for cname, ok in cross.items():
        print(f"[author_workcell]   {'PASS' if ok else 'FAIL'}  cross:{cname}")

    if args_cli.dry_run:
        print("[author_workcell] --dry-run: nothing written.")
        return
    for _l, _p, _b, _v, path, _s in targets:
        if path.exists() and not args_cli.force:
            raise SystemExit(
                f"[author_workcell] {path} exists. Refusing to overwrite an asset a run may "
                "already have been measured against. Pass --force, or --out-dir with a new path."
            )

    overall_ok = all(cross.values())
    for label, prim, boxes, verifier, path, spawn in targets:
        stage, facts, checks = build(prim, boxes, verifier)
        for cname, ok in checks.items():
            print(f"[author_workcell]   {'PASS' if ok else 'FAIL'}  {label}:{cname}")
        overall_ok = overall_ok and all(checks.values())

        path.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(path))
        sidecar = path.with_suffix(".author.json")
        sidecar.write_text(json.dumps({
            "marker": SCRIPT_MARKER,
            "asset": label,
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "frame": "UR5e base_link origin = (0,0,0); +X robot -> fixture; +Y robot's left; +Z up (rotated 2026-08-19)",
            "spawn_translation": list(spawn),
            "pending": list(geom.WORKCELL_PENDING),
            "pocket_opening_m": [geom.POCKET_OPENING_X, geom.POCKET_OPENING_Y],
            "pocket_floor_z_m": geom.POCKET_FLOOR_Z,
            "recess_bbox_m": list(geom.POCKET_ASSET_BBOX_M),
            "y_residual_mm": geom.WORKCELL_POCKET_Y_RESIDUAL * 1000.0,
            "minus_y_vs_b12_mm": geom.WORKCELL_POCKET_MINUSY_VS_B12 * 1000.0,
            "boxes": [{"name": b["name"], "centre": list(b["centre"]), "size": list(b["size"])}
                      for b in boxes],
            "measured": facts,
            "checks": checks,
            "cross_checks": cross,
        }, indent=2), encoding="utf-8")
        print(f"[author_workcell] wrote {path}")
        print(f"[author_workcell] wrote {sidecar}")

    print(f"[author_workcell] verdict: {'PASS' if overall_ok else 'FAIL'}")
    if not overall_ok:
        raise SystemExit("[author_workcell] verification FAILED -- do not use these assets")


if __name__ == "__main__":
    main()
    simulation_app.close()
