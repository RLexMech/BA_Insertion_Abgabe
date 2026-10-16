# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a millimetre-authored CAD import into a metre stage (D-016 invariant 1)."""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import shutil
import sys
import traceback

from isaaclab.app import AppLauncher

_TOOLS = pathlib.Path(__file__).resolve().parent / "tools"


def _load_tool(stem: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(stem, _TOOLS / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod


exit_with = _load_tool("isaac_exit").exit_with

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


def main() -> int:
    usd_path = pathlib.Path(args_cli.usd).resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"No such USD file: {usd_path}")
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
        return 0

    mpu = before["meters_per_unit"]
    if abs(mpu - 1.0) < 1e-9:
        raise RuntimeError(
            "metersPerUnit is already 1.0, so the size error is not a unit-declaration "
            "problem and this script must not touch the file. Measurement above."
        )

    print(f"[fix_stage_units] plan: metersPerUnit {mpu} -> 1.0, and scale {target.GetPath()} by {mpu}")
    if args_cli.dry_run:
        print("[fix_stage_units] --dry-run: nothing written.")
        return 0

    backup = usd_path.with_suffix(usd_path.suffix + ".bak")
    shutil.copy2(usd_path, backup)
    print(f"[fix_stage_units] backup written: {backup}")

    apply_unit_scale(target, mpu)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    print(f"[fix_stage_units] xformOps after: {describe_xform_ops(target)}")

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
        return 1

    print("[fix_stage_units] releasing both stage references before the replace "
          "(discriminating step, RT-25 follow-up)")
    target = None
    stage = None
    check_stage = None
    gc.collect()

    try:
        shutil.copy2(candidate, usd_path)
    except Exception as exc:              # noqa: BLE001 -- naming it IS the job
        print(f"[fix_stage_units] REPLACE FAILED: {type(exc).__name__}: {exc}")
        print(f"[fix_stage_units] the metre file is correct and stands at {candidate}; "
              f"{usd_path.name} is still the original (backup at {backup.name}).")
        print("[fix_stage_units] the stages were released before this attempt, so the "
              "memory-map hypothesis is REFUTED as well and the handle is held elsewhere.")
        traceback.print_exc()
        return 2

    print(f"[fix_stage_units] contract PASS; {usd_path.name} replaced (backup at {backup.name})")
    print("[fix_stage_units] next: re-run verify_fixture_usd.py on the same path to confirm independently.")
    return 0


if __name__ == "__main__":
    _code = 1
    try:
        _code = main() or 0
    except SystemExit as _exc:
        _code = int(_exc.code or 0)
    except BaseException:                 # noqa: BLE001 -- print it, then exit non-zero
        traceback.print_exc()
        _code = 1
    exit_with(simulation_app, _code, tag="fix_stage_units")
