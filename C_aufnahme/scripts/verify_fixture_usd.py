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

SQUARE ASSET (D-029, D-033, tisch_square_b{mm}.usd): the asset origin moved.
It now sits ON the opening plane at the pocket centre, so an upright asset spans
z = -0.755 .. 0.000 (it used to be -0.700 .. +0.055 with the origin at the
plate underside). The orientation check below encodes the NEW convention;
running this script on the old round tisch.usd will therefore report
"unrecognized", which is correct — that asset does not satisfy the square
branch's contract.

AND "unrecognized" NO LONGER FAILS THE RUN (2026-08-29). It used to. RT-77
verified the REAL fixture, every substantive check passed, and the script still
exited 1 on this one line, because the square PROXY table's z-span is the only
orientation contract in the file and the real fixture neither satisfies it nor
should. A check that returns non-zero without a defect teaches people to ignore
it, which is the inverse of the fail-fast rule. The contract now applies only to
assets that are IN its branch:

  upright       -> PASS, the square proxy asset is the right way up
  inverted      -> FAIL, it is in the branch and flipped: a real defect
  unrecognized  -> NOT APPLICABLE, printed and kept out of the verdict

NAMED GAP, not repaired here: the real fixture therefore has NO bounding-box
orientation contract at all. Giving it one needs the asset's own z-span, which
nothing has measured -- `POCKET_LOCAL_Z_RANGE` in insertion_tasks_cfg.py is the
POCKET cut-out's range, not the imported block's. Until then the flip evidence
for the real fixture is --planes: stage 1 spans -74.7938 .. +55.2938 mm, an
ASYMMETRIC pair, so a flipped import moves those planes to the other side.

Since D-033 the square table was GENERATED (scripts/author_tisch_square.py,
deleted 2026-08-28 with the dead proxy code, S6) and its filename carried the
pocket size, one file per curriculum rung. The values
this script expects are unchanged by that: the generator reproduces the same
bbox and origin convention on purpose, so this check stays a genuine
cross-check on the generator rather than a restatement of it.

Pocket dimensions are NOT measured here: a bbox cannot see into a pocket. The
geometric check was the generator's own verify pass (plus
scripts/check_tisch_geometry.py, offline); the physical one was
scripts/verify_peg_passability.py's job (centred pass + 45 deg yaw control).
Both scripts went with the generator on 2026-08-28 (S6); nothing on this
branch replaces them, because nothing on this branch builds a square table.

SECOND ROLE, added 2026-08-22: --per-prim for CAD assembly imports. The checks
above all reduce the stage to ONE bounding box, which is the right question for
a single-body fixture and the wrong one for
CAD/Step/Greifer_Bauteil/greifer_bauteil_asm.stp -- that file carries the
gripper AND the part, so the stage box is their union and matches neither
nominal. --per-prim reports a world box per mesh prim in MILLIMETRES, and
--nominal-mm names the closest prim and its per-edge deviation.

That deviation is the TESSELLATION ERROR, and it is the reason this option
exists. ChordHeightRatio decides how far the triangulated surface may sit from
the true one; at the coarse end of the HOOPS range it is millimetres, which is
larger than the 0.2 mm short-axis clearance the success criterion is derived
from (D-057). A too-small part in a too-large pocket makes the task easier than
specified without anything looking wrong. The number is reported, never judged:
what is acceptable depends on the pair and is a decision, not a threshold.

    .\\isaaclab.bat -p <repo>\\scripts\\verify_fixture_usd.py \\
        --usd <path\\to\\greifer_bauteil.usd> --per-prim \\
        --nominal-mm 143.50 96.41 50.506

Those three come from the run of 2026-08-22 (D-075) and replace the figures
this example first carried. The short axis is 96.41, NOT the body width:
a bounding box includes the two LUGS that engage the pocket notches (D-059),
and the body width is the distance between them -- 90.00 mm, `PART_BODY_X`
(corrected 2026-08-28; this line said 90.3 until the part CAD was read).
Feeding a nominal that ignores a real
feature makes the script report a 6 mm "error" that is not one -- which is
exactly what happened on the first run.

Results are written as JSON next to the USD (metrics rule, CLAUDE.md) and
printed; the JSON is the artifact to quote, not the console text.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
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

# pxr is importable only after the app is up.
from pxr import Usd, UsdGeom  # noqa: E402

NOMINALS_M = {
    "plate_thickness": 0.055,
    "table_height": 0.755,  # measured from Creo source, 2026-07-25 (was 0.750 placeholder)
    "leg_length": 0.700,
    "pocket_side": 0.032,   # not measured here, recorded for reference (D-029)
    "pocket_depth": 0.035,  # blind; not measured here either
}


def main() -> int:
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
    # both give 0.500 x 0.600 x 0.755. The z range can: the square re-import's
    # origin sits ON the opening plane at the pocket centre (D-029,
    # docs/asset_contract_tisch.md square addendum), so upright is
    # -0.755 .. 0.000 and inverted is 0.000 .. +0.755. The OLD round asset
    # (origin at the plate underside, -0.700 .. +0.055) reads "unrecognized"
    # here by design: it does not satisfy this branch's contract.
    #
    # An asset that matches NEITHER pair is outside this branch, so the contract
    # says nothing about it and must not decide its exit code (see the docstring,
    # "AND unrecognized NO LONGER FAILS THE RUN"). orientation_applicable carries
    # that distinction; orientation_ok stays the branch's own verdict.
    z_min_m, z_max_m = bbox_min_stage[2] * mpu, bbox_max_stage[2] * mpu
    upright = abs(z_min_m - (-0.755)) < 1e-3 and abs(z_max_m) < 1e-3
    inverted = abs(z_min_m) < 1e-3 and abs(z_max_m - 0.755) < 1e-3
    orientation = "upright" if upright else ("inverted" if inverted else "unrecognized")
    orientation_applicable = upright or inverted

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
        "orientation_applicable": orientation_applicable,
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

    # --- per-prim boxes (CAD imports with more than one body) ----------------
    # The whole-stage box above answers "is this asset the right size", which is
    # the D-016 question. It cannot answer "is THIS body the right size", and a
    # CAD assembly import has several: greifer_bauteil_asm.stp carries the
    # gripper and the part in one file, so the stage box is their union and
    # equals neither nominal.
    #
    # Millimetres throughout this section. The CAD nominals are in mm, the
    # clearances that decide whether the deviation matters are in mm (0.2 mm on
    # the short axis, D-057), and converting them to metres here would only
    # invite the reading error the extents_stage / extents_physical split above
    # already had to be defended against once.
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
            # stage units -> metres -> millimetres, same two-step as above.
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
                # Per-edge deviation, not just the maximum: a tessellation that
                # loses 0.02 mm on one edge and 0.15 mm on another is a
                # different finding from one that loses 0.08 mm on all three,
                # and only the per-edge list tells them apart.
                row["dev_mm"] = [g - w for g, w in zip(got_mm, want_mm)]
                row["max_abs_dev_mm"] = max(abs(d) for d in row["dev_mm"])
            best = min(prim_rows, key=lambda r: r["max_abs_dev_mm"], default=None)
            result["per_prim_nominal_mm"] = sorted(args_cli.nominal_mm)
            result["per_prim_best_match"] = best["prim"] if best else None
            result["per_prim_best_max_abs_dev_mm"] = best["max_abs_dev_mm"] if best else None

        result["per_prim"] = prim_rows
        result["per_prim_mesh_count"] = len(prim_rows)

    # --- dense coordinate planes (does the CAVITY still exist?) --------------
    # WHY A BOUNDING BOX IS NOT ENOUGH, and it is the defect this exists to
    # catch: the `asset check (fixture)` line every run prints reports the OUTER
    # edges (61 / 150 / 189.1 mm). Those come from the block. An import that
    # lost the pocket, shrank it, or shifted it off the axis leaves all three
    # unchanged and still prints PASS -- so no log on this branch could tell a
    # correct fixture from a solid one.
    #
    # The dense planes can. The fixture is prismatic, so every face puts many
    # vertices on one coordinate, and the pocket walls are the two heavily
    # populated planes at +-POCKET_WALL_X. Read the same way the STEP was read
    # (insertion_tasks_cfg.py, "Inputs: the fixture CAD itself"), so the two are
    # directly comparable and a disagreement means the IMPORT, not the method.
    if args_cli.planes:
        MM_PER_M = 1000.0
        try:
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
            # Points are LOCAL. The pocket constants are in the fixture's own
            # frame, so the local-to-world transform has to be applied or a
            # translated prim would report every plane shifted.
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

        # The verdict. Each expected plane is a MEASURED CAD number with one
        # home in insertion_tasks_cfg.py; none is typed here.
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
            # No PASS/FAIL on this number. What counts as acceptable depends on
            # the clearance of the mating pair (0.5 mm long axis, 0.2 mm short,
            # D-057) and on how much of it the OTHER body's tessellation already
            # spends. The measurement belongs in the record; the judgement is a
            # decision, not a threshold this script gets to invent.
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
            # NOT silently skipped: without the config the planes are numbers
            # with nothing to check them against, and a run that printed them
            # anyway would read like a verified import.
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
        # REPORTED, not judged -- same rule as the tessellation deviation above.
        # What error is acceptable depends on the 0.2938 mm per-side clearance,
        # and picking a threshold here would be this script inventing a
        # decision. A MISSING plane is the finding that needs no threshold.
        print("[verify_fixture_usd] plane errors are REPORTED, not judged -- "
              "compare against the 0.2938 mm per-side cross play (PLAY_X / 2). "
              "A feature with no plane near it at all is the finding.")

    # Reported separately from the D-016 verdict: orientation and payload freedom are
    # asset-contract requirements (docs/asset_contract_tisch.md), not D-016 invariants.
    # The orientation contract is fixture-specific -- it encodes the SQUARE PROXY
    # table's origin on the opening plane -- so it is applied to the assets it
    # describes and to no others. Two ways out of the verdict, for two reasons:
    #   --per-prim   a CAD assembly's stage box is the union of several bodies,
    #                so no single z-span contract can exist for it;
    #   unrecognized the asset is outside the square branch (the real fixture is),
    #                and the contract has nothing to say about it. RT-77 exited 1
    #                on exactly this with no defect present -- 2026-08-29 fix.
    # "inverted" still FAILS: that asset IS in the branch and is upside down.
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
    # D-081: exit BEFORE the shutdown. RT-27 measured this script ending a run
    # with a traceback and `exit code: 0` -- close() does not return (RT-22),
    # so anything after it, SystemExit included, was dead code.
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
    exit_with(simulation_app, _code, tag="verify_fixture_usd")
