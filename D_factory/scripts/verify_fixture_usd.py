# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the imported table/hole fixture USD against the D-016 invariants."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import traceback

from isaaclab.app import AppLauncher

_TOOLS = pathlib.Path(__file__).resolve().parent / "tools"
_spec = importlib.util.spec_from_file_location("isaac_exit", _TOOLS / "isaac_exit.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["isaac_exit"] = _mod
_spec.loader.exec_module(_mod)
exit_with = _mod.exit_with

parser = argparse.ArgumentParser(description="Verify fixture USD (D-016).")
parser.add_argument("--usd", type=str, required=True, help="Path to the fixture USD file.")
parser.add_argument(
    "--nominal",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional nominal overall bbox extents in meters for an explicit pass/fail.",
)
parser.add_argument(
    "--per-prim",
    action="store_true",
    help="Report a world bbox for EVERY mesh prim, in millimetres, instead of only "
    "the whole-stage box. Needed for multi-body imports: the assembly USD holds "
    "gripper and part, so the stage box is their union and matches neither.",
)
parser.add_argument(
    "--nominal-mm",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Nominal edge lengths in MILLIMETRES. With --per-prim, every mesh prim is "
    "compared against them and the closest one is named. Millimetres because that "
    "is the unit the CAD nominals are written in; the deviation is the "
    "tessellation error and is the point of the run.",
)
parser.add_argument(
    "--planes",
    action="store_true",
    help=(
        "Report the DENSE COORDINATE PLANES of every mesh prim, in millimetres, "
        "per axis. A bounding box cannot see into a pocket -- it reports the "
        "OUTER block and passes even if the import lost, shrank or shifted the "
        "cavity. The dense planes can: a prismatic cut-out puts many vertices on "
        "each of its faces, so the pocket walls appear as the two heavily "
        "populated planes at +-POCKET_WALL_X. Same method the STEP was read with "
        "(insertion_tasks_cfg.py, 'Inputs: the fixture CAD itself')."
    ),
)
parser.add_argument(
    "--plane-min-points",
    type=int,
    default=4,
    help=(
        "Minimum vertex count for a coordinate to be reported as a plane. Below "
        "this the list fills with tessellation noise and buries the faces. 4 is "
        "the value the STEP read used."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom  # noqa: E402

NOMINALS_M = {
    "plate_thickness": 0.055,
    "table_height": 0.755,
    "leg_length": 0.700,
    "pocket_side": 0.032,
    "pocket_depth": 0.035,
}


def main() -> int:
    usd_path = pathlib.Path(args_cli.usd).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {usd_path}")

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    bound = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
    box_range = bound.ComputeAlignedRange()
    size = box_range.GetSize()
    extents_stage = {"x": float(size[0]), "y": float(size[1]), "z": float(size[2])}
    mpu = float(meters_per_unit)
    extents_physical = {k: v * mpu for k, v in extents_stage.items()}
    bbox_min_stage = [float(box_range.GetMin()[i]) for i in range(3)]
    bbox_max_stage = [float(box_range.GetMax()[i]) for i in range(3)]

    looks_mm_scaled = any(v > 10.0 for v in extents_physical.values())

    z_min_m, z_max_m = bbox_min_stage[2] * mpu, bbox_max_stage[2] * mpu
    upright = abs(z_min_m - (-0.755)) < 1e-3 and abs(z_max_m) < 1e-3
    inverted = abs(z_min_m) < 1e-3 and abs(z_max_m - 0.755) < 1e-3
    orientation = "upright" if upright else ("inverted" if inverted else "unrecognized")
    orientation_applicable = upright or inverted

    arcs = []
    for prim in stage.Traverse():
        for arc_kind, meta_key in (("reference", "references"), ("payload", "payload")):
            list_op = prim.GetMetadata(meta_key)
            if not list_op:
                continue
            items = getattr(list_op, "GetAddedOrExplicitItems", list)()
            for item in items:
                asset_path = str(getattr(item, "assetPath", "") or "")
                arcs.append(
                    {
                        "prim": str(prim.GetPath()),
                        "arc": arc_kind,
                        "asset_path": asset_path,
                        "target_prim_path": str(getattr(item, "primPath", "") or ""),
                        "internal": not asset_path,
                    }
                )
    external_arcs = [a for a in arcs if not a["internal"]]

    result = {
        "usd": str(usd_path),
        "meters_per_unit": mpu,
        "meters_per_unit_ok": abs(mpu - 1.0) < 1e-9,
        "up_axis": str(up_axis),
        "up_axis_ok": str(up_axis) == "Z",
        "bbox_extents_stage_units": extents_stage,
        "bbox_extents_physical_m": extents_physical,
        "bbox_min_stage_units": bbox_min_stage,
        "bbox_max_stage_units": bbox_max_stage,
        "bbox_z_range_m": [z_min_m, z_max_m],
        "orientation": orientation,
        "orientation_ok": upright,
        "orientation_applicable": orientation_applicable,
        "composition_arcs": arcs,
        "external_arcs": external_arcs,
        "self_contained_ok": not external_arcs,
        "looks_mm_scaled": looks_mm_scaled,
        "nominals_m": NOMINALS_M,
    }
    if args_cli.nominal is not None:
        got = sorted(extents_physical.values())
        want = sorted(args_cli.nominal)
        result["nominal_check"] = {
            "nominal_sorted_m": want,
            "measured_sorted_m": got,
            "max_abs_error_m": max(abs(g - w) for g, w in zip(got, want)),
        }

    if args_cli.per_prim:
        MM_PER_M = 1000.0
        prim_rows = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            p_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if p_range.IsEmpty():
                continue
            p_size = p_range.GetSize()
            edges_mm = [float(p_size[i]) * mpu * MM_PER_M for i in range(3)]
            prim_rows.append(
                {
                    "prim": str(prim.GetPath()),
                    "edges_mm": edges_mm,
                    "min_mm": [float(p_range.GetMin()[i]) * mpu * MM_PER_M for i in range(3)],
                    "max_mm": [float(p_range.GetMax()[i]) * mpu * MM_PER_M for i in range(3)],
                }
            )

        if args_cli.nominal_mm is not None:
            want_mm = sorted(args_cli.nominal_mm)
            for row in prim_rows:
                got_mm = sorted(row["edges_mm"])
                row["dev_mm"] = [g - w for g, w in zip(got_mm, want_mm)]
                row["max_abs_dev_mm"] = max(abs(d) for d in row["dev_mm"])
            best = min(prim_rows, key=lambda r: r["max_abs_dev_mm"], default=None)
            result["per_prim_nominal_mm"] = sorted(args_cli.nominal_mm)
            result["per_prim_best_match"] = best["prim"] if best else None
            result["per_prim_best_max_abs_dev_mm"] = best["max_abs_dev_mm"] if best else None

        result["per_prim"] = prim_rows
        result["per_prim_mesh_count"] = len(prim_rows)

    if args_cli.planes:
        MM_PER_M = 1000.0
        try:
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "source" / "insertion"))
            import insertion
            print(f"[verify_fixture_usd] insertion package imported from: {pathlib.Path(insertion.__file__).resolve().parent}")
            from insertion.tasks.direct.insertion import insertion_tasks_cfg as _cfg
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            _cfg = None
            result["planes_reference_error"] = str(exc)

        plane_rows = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if not pts:
                continue
            xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            world = [xform.Transform(p) for p in pts]
            axes = {}
            for i, name in enumerate("xyz"):
                counts: dict = {}
                for p in world:
                    key = round(float(p[i]) * mpu * MM_PER_M, 4)
                    counts[key] = counts.get(key, 0) + 1
                axes[name] = sorted(
                    ((v, n) for v, n in counts.items() if n >= args_cli.plane_min_points)
                )
            plane_rows.append({"prim": str(prim.GetPath()), "vertices": len(world), "planes": axes})

        result["planes"] = plane_rows
        result["plane_min_points"] = int(args_cli.plane_min_points)

        if _cfg is not None:
            expected = [
                ("pocket wall -x", "x", -_cfg.POCKET_WALL_X * MM_PER_M),
                ("pocket wall +x", "x", +_cfg.POCKET_WALL_X * MM_PER_M),
                ("pocket wall -y", "y", -_cfg.POCKET_WALL_Y * MM_PER_M),
                ("pocket wall +y", "y", +_cfg.POCKET_WALL_Y * MM_PER_M),
                ("pocket floor", "z", _cfg.POCKET_FLOOR_Z * MM_PER_M),
                ("stage 1 -x", "x", _cfg.POCKET_STAGE1_X_RANGE[0] * MM_PER_M),
                ("stage 1 +x", "x", _cfg.POCKET_STAGE1_X_RANGE[1] * MM_PER_M),
            ]
            checks = []
            for label, axis, want in expected:
                best_v, best_n, best_d = None, 0, None
                for row in plane_rows:
                    for v, n in row["planes"][axis]:
                        d = abs(v - want)
                        if best_d is None or d < best_d:
                            best_v, best_n, best_d = v, n, d
                checks.append(
                    {
                        "feature": label,
                        "axis": axis,
                        "want_mm": want,
                        "nearest_mm": best_v,
                        "nearest_points": best_n,
                        "error_mm": best_d,
                    }
                )
            result["plane_checks"] = checks

    out_path = usd_path.with_suffix(".verify.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[verify_fixture_usd] wrote {out_path}")

    ok = result["meters_per_unit_ok"] and result["up_axis_ok"] and not looks_mm_scaled
    print(f"[verify_fixture_usd] D-016 invariants: {'PASS' if ok else 'FAIL'}")

    if args_cli.per_prim:
        print(f"[verify_fixture_usd] mesh prims: {result['per_prim_mesh_count']}")
        for row in result["per_prim"]:
            edges = " x ".join(f"{e:.3f}" for e in row["edges_mm"])
            line = f"[verify_fixture_usd]   {row['prim']}: {edges} mm"
            if "max_abs_dev_mm" in row:
                devs = ", ".join(f"{d:+.3f}" for d in row["dev_mm"])
                line += f"   dev(sorted) [{devs}] mm"
            print(line)
        if result.get("per_prim_best_match"):
            dev = result["per_prim_best_max_abs_dev_mm"]
            print(
                f"[verify_fixture_usd] closest to nominal: {result['per_prim_best_match']} "
                f"at {dev:.3f} mm"
            )
            print(
                "[verify_fixture_usd] tessellation deviation is REPORTED, not judged -- "
                "compare it against the 0.2 mm short-axis clearance (D-057)"
            )

    if args_cli.planes:
        print(f"[verify_fixture_usd] dense planes, mm, >= {args_cli.plane_min_points} vertices:")
        for row in result["planes"]:
            print(f"[verify_fixture_usd]   {row['prim']}  ({row['vertices']} vertices)")
            for axis in "xyz":
                cells = "  ".join(f"{v:+.4f}({n})" for v, n in row["planes"][axis])
                print(f"[verify_fixture_usd]     {axis}: {cells if cells else '(none)'}")
        if "planes_reference_error" in result:
            print("[verify_fixture_usd] NO REFERENCE: could not import "
                  f"insertion_tasks_cfg ({result['planes_reference_error']}). "
                  "The planes above are UNCHECKED.")
        for chk in result.get("plane_checks", []):
            if chk["nearest_mm"] is None:
                print(f"[verify_fixture_usd]   {chk['feature']:<15} want {chk['want_mm']:+9.4f} mm"
                      "   NO PLANE ON THIS AXIS AT ALL")
                continue
            print(f"[verify_fixture_usd]   {chk['feature']:<15} want {chk['want_mm']:+9.4f} mm  "
                  f"nearest {chk['nearest_mm']:+9.4f} mm ({chk['nearest_points']} pts)  "
                  f"error {chk['error_mm']:.4f} mm")
        print("[verify_fixture_usd] plane errors are REPORTED, not judged -- "
              "compare against the 0.2938 mm per-side cross play (PLAY_X / 2). "
              "A feature with no plane near it at all is the finding.")

    orientation_applies = result["orientation_applicable"] and not args_cli.per_prim
    contract_ok = ok and result["self_contained_ok"] and (
        result["orientation_ok"] if orientation_applies else True
    )
    if args_cli.per_prim:
        print(f"[verify_fixture_usd] orientation: {orientation} (CAD assembly, contract not applied)")
    elif not orientation_applies:
        print(f"[verify_fixture_usd] orientation: {orientation} -- OUTSIDE the square-proxy "
              "contract, not applied. This asset has NO bbox orientation check; for the "
              "real fixture the flip evidence is --planes (stage 1 is asymmetric).")
    else:
        print(f"[verify_fixture_usd] orientation: {orientation}")
    internal_count = sum(1 for a in arcs if a["internal"])
    print(f"[verify_fixture_usd] composition arcs: {len(arcs)} ({internal_count} internal, {len(external_arcs)} external)")
    for a in external_arcs:
        print(f"[verify_fixture_usd] EXTERNAL {a['arc']} on {a['prim']} -> {a['asset_path']}")
    if external_arcs:
        print("[verify_fixture_usd] asset depends on other files; re-export flattened")
    print(f"[verify_fixture_usd] asset contract: {'PASS' if contract_ok else 'FAIL'}")
    return 0 if contract_ok else 1


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
    exit_with(simulation_app, _code, tag="verify_fixture_usd")
