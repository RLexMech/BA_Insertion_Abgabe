# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the imported table/hole fixture USD against the D-016 invariants.

UNVERIFIED — authored on the dev PC (no Isaac installation); run on the
training machine:

    cd C:\\Isaaclab
    conda activate env_isaaclab
    .\\isaaclab.bat -p <repo>\\scripts\\verify_fixture_usd.py --usd <path\\to\\tisch2.usd>

Checks (D-016, DECISIONS.md):
  1. metersPerUnit == 1.0 on the stage.
  2. Up-axis == Z (the Isaac Lab mesh converter emits Y-up; a Y-up asset on a
     Z-up stage is tilted 90 degrees).
  3. World-space bounding box against the nominal table dimensions
     (task_specification.md): plate thickness 0.055 m, table height 0.755 m,
     leg length 0.70 m. A ~1000x bounding box means the mm/meter scale was not
     resolved — fix at the USD level and re-save, never by eyeballing.

Bore diameter (nominal 0.030 m) is NOT measured here: a robust measurement
needs the bore-entrance Xform (D-017) as axis reference and is added once that
frame exists in the USD. Until then, measure the bore interactively in the
stage and record the value.

Results are written as JSON next to the USD (metrics rule, CLAUDE.md) and
printed; the JSON is the artifact to quote, not the console text.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from isaaclab.app import AppLauncher

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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pxr is importable only after the app is up.
from pxr import Usd, UsdGeom  # noqa: E402

NOMINALS_M = {
    "plate_thickness": 0.055,
    "table_height": 0.755,  # measured from Creo source, 2026-07-25 (was 0.750 placeholder)
    "leg_length": 0.700,
    "bore_diameter": 0.030,  # not measured here, recorded for reference
}


def main() -> None:
    usd_path = pathlib.Path(args_cli.usd).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {usd_path}")

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)

    # World-space bbox over default + render purposes of the whole stage.
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    bound = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
    box_range = bound.ComputeAlignedRange()
    size = box_range.GetSize()
    # Stage units, NOT metres: ComputeWorldBound ignores metersPerUnit. Keeping the
    # two apart matters -- a run on 2026-07-26 reported 500/600/755 here with
    # metersPerUnit = 0.001, which is a correctly sized asset on a millimetre stage,
    # and the old "_m" suffix invited reading it as a 1000x oversize error.
    extents_stage = {"x": float(size[0]), "y": float(size[1]), "z": float(size[2])}
    mpu = float(meters_per_unit)
    extents_physical = {k: v * mpu for k, v in extents_stage.items()}
    bbox_min_stage = [float(box_range.GetMin()[i]) for i in range(3)]
    bbox_max_stage = [float(box_range.GetMax()[i]) for i in range(3)]

    # Heuristic scale check: if any physical extent exceeds 10 m, the fixture is
    # almost certainly still in millimetres (a table is < 2 m in every direction).
    looks_mm_scaled = any(v > 10.0 for v in extents_physical.values())

    # Orientation. Extents alone cannot distinguish upright from upside down --
    # both give 0.500 x 0.600 x 0.755. The z range can: the asset origin sits at
    # the plate underside, so upright is -0.700 .. +0.055 (D-022,
    # docs/asset_contract_tisch.md), and inverted is -0.055 .. +0.700.
    z_min_m, z_max_m = bbox_min_stage[2] * mpu, bbox_max_stage[2] * mpu
    upright = abs(z_min_m - (-0.700)) < 1e-3 and abs(z_max_m - 0.055) < 1e-3
    inverted = abs(z_min_m - (-0.055)) < 1e-3 and abs(z_max_m - 0.700) < 1e-3
    orientation = "upright" if upright else ("inverted" if inverted else "unrecognized")

    # Composition arcs, with their asset paths. Presence alone is not the question:
    # an INTERNAL arc (empty assetPath) targets a prim in this same file and is
    # self-contained, while an EXTERNAL one makes the asset depend on a second file.
    # An external arc to a missing file still resolves the default prim, so the spawn
    # succeeds and the fixture is silently invisible -- the 2026-07-26 failure.
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
        "composition_arcs": arcs,
        "external_arcs": external_arcs,
        "self_contained_ok": not external_arcs,
        "looks_mm_scaled": looks_mm_scaled,
        "nominals_m": NOMINALS_M,
    }
    if args_cli.nominal is not None:
        # Compare sorted extents against sorted nominals: axis assignment may
        # differ between CAD and stage even when the size is right.
        got = sorted(extents_physical.values())
        want = sorted(args_cli.nominal)
        result["nominal_check"] = {
            "nominal_sorted_m": want,
            "measured_sorted_m": got,
            "max_abs_error_m": max(abs(g - w) for g, w in zip(got, want)),
        }

    out_path = usd_path.with_suffix(".verify.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[verify_fixture_usd] wrote {out_path}")

    ok = result["meters_per_unit_ok"] and result["up_axis_ok"] and not looks_mm_scaled
    print(f"[verify_fixture_usd] D-016 invariants: {'PASS' if ok else 'FAIL'}")
    # Reported separately from the D-016 verdict: orientation and payload freedom are
    # asset-contract requirements (docs/asset_contract_tisch.md), not D-016 invariants.
    contract_ok = ok and result["orientation_ok"] and result["self_contained_ok"]
    print(f"[verify_fixture_usd] orientation: {orientation}")
    internal_count = sum(1 for a in arcs if a["internal"])
    print(f"[verify_fixture_usd] composition arcs: {len(arcs)} ({internal_count} internal, {len(external_arcs)} external)")
    for a in external_arcs:
        print(f"[verify_fixture_usd] EXTERNAL {a['arc']} on {a['prim']} -> {a['asset_path']}")
    if external_arcs:
        print("[verify_fixture_usd] asset depends on other files; re-export flattened")
    print(f"[verify_fixture_usd] asset contract: {'PASS' if contract_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
