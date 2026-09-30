# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a millimetre-authored CAD import into a metre stage (D-016 invariant 1).

Symptom this fixes: the Isaac Sim CAD import leaves the stage at
``metersPerUnit = 0.001`` with geometry authored in millimetre numbers. The
physical size is then correct only if a reader honours ``metersPerUnit`` --
Isaac Lab does not, so the asset spawns 1000x too large.

The fix is two coupled edits, never one alone:
  1. ``metersPerUnit`` -> 1.0 on the stage, and
  2. a uniform scale of the OLD ``metersPerUnit`` (0.001) on the default prim,
     placed FIRST in ``xformOpOrder`` so it scales any existing translation too.
Setting only (1) would resize the table to 500 m.

UNVERIFIED -- authored on the dev PC (no Isaac installation). Copy to the
training machine before running (CLAUDE.md two-repo sync workflow):

    cd C:\\Isaaclab
    conda activate env_isaaclab
    .\\isaaclab.bat -p <repo>\\scripts\\fix_stage_units.py --usd <path>\\Tisch.usd

Safety model, because each run on the training machine costs a manual round
trip: the input file is never edited in place. The script writes a backup, then
exports a flattened candidate, then re-opens that candidate in a fresh stage and
checks it against the asset contract (docs/asset_contract_tisch.md). The
original is replaced only if that check passes. Every intermediate number is
printed, so a failed run still explains itself instead of needing a second trip.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fix stage units of a CAD import (D-016).")
parser.add_argument("--usd", type=str, required=True, help="Path to the USD file to fix.")
parser.add_argument(
    "--expect-extents",
    type=float,
    nargs=3,
    default=[0.500, 0.600, 0.755],
    metavar=("X", "Y", "Z"),
    help="Expected physical bbox extents in metres after the fix.",
)
parser.add_argument(
    "--expect-z-range",
    type=float,
    nargs=2,
    default=[-0.700, 0.055],
    metavar=("MIN", "MAX"),
    help="Expected physical bbox z range in metres; encodes 'upright, origin at plate underside'.",
)
parser.add_argument("--tol", type=float, default=1.0e-4, help="Tolerance in metres (default 0.1 mm).")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Report the current state and the planned edit, write nothing.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pxr is importable only after the app is up.
from pxr import Gf, Usd, UsdGeom  # noqa: E402


def measure(stage: Usd.Stage) -> dict:
    """Return the stage's unit metadata and its world bbox, in stage units and in metres."""
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    if rng.IsEmpty():
        return {"meters_per_unit": mpu, "up_axis": str(UsdGeom.GetStageUpAxis(stage)), "bbox_empty": True}
    lo = [float(rng.GetMin()[i]) for i in range(3)]
    hi = [float(rng.GetMax()[i]) for i in range(3)]
    return {
        "meters_per_unit": mpu,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "bbox_empty": False,
        "bbox_min_stage_units": lo,
        "bbox_max_stage_units": hi,
        # The physical size is the stage-unit number times metersPerUnit. Reporting
        # only stage units is what made an earlier run look correct when it was not.
        "bbox_min_m": [v * mpu for v in lo],
        "bbox_max_m": [v * mpu for v in hi],
        "extents_m": [(hi[i] - lo[i]) * mpu for i in range(3)],
    }


def describe_composition(stage: Usd.Stage) -> dict:
    """Report payload/reference arcs -- a payload to a missing file spawns invisibly."""
    payloads, references = [], []
    for prim in stage.Traverse():
        if prim.HasAuthoredPayloads():
            payloads.append(str(prim.GetPath()))
        if prim.HasAuthoredReferences():
            references.append(str(prim.GetPath()))
    default_prim = stage.GetDefaultPrim()
    return {
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "root_children": [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()],
        "prims_with_payloads": payloads,
        "prims_with_references": references,
    }


def describe_xform_ops(prim: Usd.Prim) -> list[str]:
    xf = UsdGeom.Xformable(prim)
    if not xf:
        return []
    return [f"{op.GetOpName()} = {op.Get()}" for op in xf.GetOrderedXformOps()]


def apply_unit_scale(prim: Usd.Prim, factor: float) -> None:
    """Scale `prim` by `factor`, outermost, so existing translations scale with it."""
    xf = UsdGeom.Xformable(prim)
    if not xf:
        raise RuntimeError(f"{prim.GetPath()} is not Xformable; cannot carry the unit scale.")

    existing = xf.GetOrderedXformOps()
    for op in existing:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            old = op.Get()
            op.Set(Gf.Vec3f(float(old[0]) * factor, float(old[1]) * factor, float(old[2]) * factor))
            print(f"[fix_stage_units] multiplied existing {op.GetOpName()} by {factor}")
            return

    scale_op = xf.AddScaleOp(opSuffix="unitsResolve")
    scale_op.Set(Gf.Vec3f(factor, factor, factor))
    # Outermost: first entry in xformOpOrder is applied last to the point, i.e. it
    # wraps everything below it, so any translate authored in millimetres is scaled too.
    xf.SetXformOpOrder([scale_op] + list(existing), xf.GetResetXformStack())
    print(f"[fix_stage_units] added {scale_op.GetOpName()} = {factor}, placed first in xformOpOrder")


def contract_check(m: dict) -> dict:
    """Check a measurement against the asset contract. Returns a verdict dict."""
    tol = args_cli.tol
    checks = {
        "meters_per_unit_is_1": abs(m["meters_per_unit"] - 1.0) < 1e-9,
        "up_axis_is_z": m["up_axis"] == "Z",
    }
    if m.get("bbox_empty", True):
        checks["geometry_present"] = False
        return {"checks": checks, "pass": False}
    checks["geometry_present"] = True
    for i, axis in enumerate("xyz"):
        checks[f"extent_{axis}"] = abs(m["extents_m"][i] - args_cli.expect_extents[i]) < tol
    checks["z_min_upright"] = abs(m["bbox_min_m"][2] - args_cli.expect_z_range[0]) < tol
    checks["z_max_upright"] = abs(m["bbox_max_m"][2] - args_cli.expect_z_range[1]) < tol
    return {"checks": checks, "pass": all(checks.values())}


def main() -> None:
    usd_path = pathlib.Path(args_cli.usd).resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"No such USD file: {usd_path}")
    # resolve() reports the on-disk casing on Windows; a mismatch matters on Linux.
    print(f"[fix_stage_units] resolved input: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {usd_path}")

    before = measure(stage)
    composition = describe_composition(stage)
    print("[fix_stage_units] BEFORE:")
    print(json.dumps({"measurement": before, "composition": composition}, indent=2))

    target = stage.GetDefaultPrim()
    if not target:
        children = stage.GetPseudoRoot().GetChildren()
        if len(children) != 1:
            raise RuntimeError(
                "No defaultPrim and the root has "
                f"{len(children)} children {[str(c.GetPath()) for c in children]}; "
                "cannot decide where the unit scale belongs. Set a defaultPrim in Isaac Sim first."
            )
        target = children[0]
        print(f"[fix_stage_units] no defaultPrim; using the single root child {target.GetPath()}")
    print(f"[fix_stage_units] unit scale target: {target.GetPath()}")
    print(f"[fix_stage_units] its xformOps before: {describe_xform_ops(target)}")

    verdict_before = contract_check(before)
    if verdict_before["pass"]:
        print("[fix_stage_units] the file already satisfies the contract; nothing to do.")
        print(json.dumps(verdict_before, indent=2))
        return

    mpu = before["meters_per_unit"]
    if abs(mpu - 1.0) < 1e-9:
        raise RuntimeError(
            "metersPerUnit is already 1.0, so the size error is not a unit-declaration "
            "problem and this script must not touch the file. Measurement above."
        )

    print(f"[fix_stage_units] plan: metersPerUnit {mpu} -> 1.0, and scale {target.GetPath()} by {mpu}")
    if args_cli.dry_run:
        print("[fix_stage_units] --dry-run: nothing written.")
        return

    backup = usd_path.with_suffix(usd_path.suffix + ".bak")
    shutil.copy2(usd_path, backup)
    print(f"[fix_stage_units] backup written: {backup}")

    apply_unit_scale(target, mpu)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    print(f"[fix_stage_units] xformOps after: {describe_xform_ops(target)}")

    # Flatten so the result cannot depend on a sibling payload file, the failure
    # mode that produced an invisible fixture on 2026-07-26.
    candidate = usd_path.with_name(usd_path.stem + ".fixed" + usd_path.suffix)
    stage.Flatten().Export(str(candidate))
    print(f"[fix_stage_units] candidate exported: {candidate}")

    check_stage = Usd.Stage.Open(str(candidate))
    if check_stage is None:
        raise RuntimeError(f"Could not re-open the candidate: {candidate}")
    after = measure(check_stage)
    after_composition = describe_composition(check_stage)
    verdict = contract_check(after)
    print("[fix_stage_units] AFTER (fresh stage, candidate file):")
    print(json.dumps({"measurement": after, "composition": after_composition, "verdict": verdict}, indent=2))

    report = {
        "input": str(usd_path),
        "backup": str(backup),
        "candidate": str(candidate),
        "before": before,
        "before_composition": composition,
        "after": after,
        "after_composition": after_composition,
        "verdict": verdict,
        "expected_extents_m": list(args_cli.expect_extents),
        "expected_z_range_m": list(args_cli.expect_z_range),
        "tolerance_m": args_cli.tol,
    }
    report_path = usd_path.with_suffix(".fix_units.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[fix_stage_units] wrote {report_path}")

    if not verdict["pass"]:
        failed = [k for k, v in verdict["checks"].items() if not v]
        print(f"[fix_stage_units] CONTRACT FAIL on {failed}")
        print(f"[fix_stage_units] original left untouched; inspect {candidate} and the report.")
        return

    shutil.copy2(candidate, usd_path)
    print(f"[fix_stage_units] contract PASS; {usd_path.name} replaced (backup at {backup.name})")
    print("[fix_stage_units] next: re-run verify_fixture_usd.py on the same path to confirm independently.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
