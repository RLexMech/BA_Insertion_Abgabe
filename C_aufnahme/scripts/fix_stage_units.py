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

THE DEFAULTS DESCRIBE THE TABLE, NOT EVERY ASSET. ``--expect-extents`` and
``--expect-z-range`` default to 0.500/0.600/0.755 m and -0.700..0.055 m, which
is the square table's contract. Run this on any other asset without setting
them and the unit fix will succeed while the verdict reports CONTRACT FAIL on
``extent_*`` and ``z_*_upright`` -- and because the original is only replaced
on a pass, the correct metre file is left sitting next to it as
``<name>.fixed.usd`` while the millimetre original stays in place. That
happened on 2026-08-22 with the gripper/part assembly (D-075). For a tool asset
authored to the D-068 origin, the flange face is z = 0 and the part bottom is
FLANGE_TO_PART_BOTTOM below it, so:

    .\\scripts\\rt_log.ps1 RT-10 python scripts/fix_stage_units.py \\
        --usd <...>\\assets\\Werkzeug\\greifer_bauteil_asm.usd \\
        --expect-extents 0.09641 0.14350 0.15200 --expect-z-range 0.0 0.15200

The z range there is an independent check, not a restatement of the file: it
comes from the origin convention plus the cell measurement.

Safety model, because each run on the training machine costs a manual round
trip: the input file is never edited in place. The script writes a backup, then
exports a flattened candidate, then re-opens that candidate in a fresh stage and
checks it against the asset contract (docs/asset_contract_tisch.md). The
original is replaced only if that check passes. Every intermediate number is
printed, so a failed run still explains itself instead of needing a second trip.

Exit codes (D-081): 0 replaced, nothing to do, or --dry-run; 1 CONTRACT FAIL;
2 the replace itself failed, in which case the line above it names the
exception. The old `finally: simulation_app.close()` is gone -- close() does
not return (RT-22), so it ate both the traceback and the code.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import shutil
import sys
import traceback

from isaaclab.app import AppLauncher

# Loaded by path because scripts/tools is not a package -- the idiom
# author_tool_ur5e.py already uses. Stdlib only, so it loads before the
# simulator does.
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
# PROVENANCE. The docstring above pays for these three numbers as a
# measurement of this asset. The SAME three also stand in
# insertion_tasks_cfg.py as the PROXY plate: PLATE_HALF_EXTENTS (0.250,
# 0.300) doubled is 0.500 x 0.600, and PLATE_TOP_Z is 0.755. Whether that is
# one fact or two facts that happen to agree is NOT established here.
# Unlike the tools/ scripts, this file already runs inside an isaaclab
# environment, so importing the constant is possible -- nobody has wired it,
# and it cannot be tested on the dev laptop. Until then: change one, check
# the other.
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


def main() -> int:
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
        return 1

    # WHY THE REPLACE IS NOT A BARE shutil.copy2 ANY MORE.
    #
    # THE SYMPTOM (2026-08-22): the report said `"pass": true`, the process
    # exited 0, the two lines below never printed, and the original was NOT
    # replaced. Three runs later it is reproduced three times and named once.
    #
    # RT-23 and RT-24 (2026-08-24): the defect reproduces on a fresh copy,
    # deterministically, with bit-identical numbers. Not a one-off.
    #
    # RT-25, the run that named it -- MEASURED, not deduced:
    #
    #     REPLACE FAILED: OSError: [Errno 22] Invalid argument: <usd_path>
    #     shutil.py:258, in copyfile -> with open(dst, 'wb') as fdst:
    #
    # So the failure is in OPENING THE DESTINATION for writing. It is NOT
    # PermissionError and NOT WinError 32 -- the hypothesis that stood here
    # predicted exactly those and is REFUTED. It is EINVAL.
    #
    # Why the silence, and only this: D-081. The old
    # `finally: simulation_app.close()` swallowed the traceback and the code,
    # because close() does not return (RT-22). That half is settled.
    #
    # NEW HYPOTHESIS, and it is UNTESTED -- do not quote it as the cause:
    # `stage` was opened FROM usd_path and `check_stage` from candidate, USD
    # memory-maps crate files, and a mapped file cannot be opened for writing.
    # The step below is what discriminates: the two stage references are
    # dropped first, and the copy is retried. If it then succeeds, the mapping
    # is the mechanism and the log says so in one run. If EINVAL comes back
    # with the stages gone, the mapping is ruled out and something else holds
    # that handle -- also an answer, and a different one.
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
    # D-081. What stood here was `try: main() finally: simulation_app.close()`,
    # and close() does not return (measured, RT-22) -- so a raised exception
    # died in the finally without a traceback and the run reported exit 0.
    # That is the mechanism behind the 2026-08-22 entry in HANDOFF-SZENE.md.
    _code = 1
    try:
        _code = main() or 0
    except SystemExit as _exc:            # argparse
        _code = int(_exc.code or 0)
    except BaseException:                 # noqa: BLE001 -- print it, then exit non-zero
        traceback.print_exc()
        _code = 1
    exit_with(simulation_app, _code, tag="fix_stage_units")
