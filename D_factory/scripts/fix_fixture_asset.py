# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Make the fixture asset survive being referenced: unit scale and instancing.

Two defects, one root cause -- the asset's structure -- measured on the training
machine 2026-07-26 by `verify_fixture_spawn.py`:

1. **Unit scale dropped.** The 0.001 unit compensation lives in the root prim's
   ``xformOpOrder``. Isaac Lab's spawner authors ``[translate, orient, scale]``
   on the *referencing* prim, and in USD composition that replaces the
   referenced ``xformOpOrder`` rather than merging with it. The scale op is
   silently lost and the table spawns 1000x oversized. Measured: variant B and
   the real spawner D both gave 500 x 600 x 755; the asset opened directly (A)
   gave the correct 0.500 x 0.600 x 0.755.

2. **Collision properties not applied.** The CAD converter left the geometry
   behind an *instanceable* prim. Isaac Lab reported
   ``Could not perform 'modify_collision_properties' ... exists on an instanced
   prim ... ['/World/Fixture/tisch']``. The fixture would have had no collision
   at all -- invisible in increment 1, and a mystery one increment later when
   the peg refuses to touch anything.

The repair, and why this shape:

* Wrap the geometry in a **clean parent Xform** that carries no xformOps and is
  the default prim. A referencing context then overwrites an *empty* op order,
  and the unit scale one level down survives. This is variant C of the
  measurement, which was verified to give 0.500 x 0.600 x 0.755 with the legs
  at z = -3.3e-08 and the plate top at 0.755. It is not a guess.
* **De-instance** every instanceable prim so schema edits reach the geometry.
* **Flatten** on export, so the result carries no external arc (the 2026-07-26
  invisible-fixture failure).

The unit scale is deliberately NOT baked into point data. Moving it one level
down is the smaller edit, and it is the one that was measured.

Verification, and this is the part that was missing before: the candidate is
checked **in the referenced context**, mimicking what the spawner authors, not
only by opening it directly. `verify_fixture_usd.py` passed this asset while it
was broken precisely because it only ever tested the direct-open case, which is
the one usage that never occurs in practice.

UNVERIFIED -- authored on the dev PC (no Isaac installation). On the training
machine:

    conda activate env_isaaclab
    python scripts\\fix_fixture_asset.py --usd <path>\\tisch.usd

Add --dry-run to measure and print the plan without writing anything.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import pathlib
import shutil
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

parser = argparse.ArgumentParser(description="Repair the fixture asset for referenced use.")
parser.add_argument("--usd", type=str, required=True, help="Path to the fixture USD file.")
parser.add_argument(
    "--spawn-pos",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.700],
    metavar=("X", "Y", "Z"),
    help="Spawn translation used for the referenced-context check (D-022).",
)
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
    help="Expected physical bbox extents in metres.",
)
parser.add_argument(
    "--expect-z-range",
    type=float,
    nargs=2,
    default=[-0.700, 0.055],
    metavar=("MIN", "MAX"),
    help="Expected bbox z range in metres when opened directly.",
)
parser.add_argument("--tol", type=float, default=1.0e-3, help="Tolerance in metres (default 1 mm).")
parser.add_argument("--dry-run", action="store_true", help="Measure and plan, write nothing.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pxr is importable only after the app is up.
from pxr import Gf, Usd, UsdGeom  # noqa: E402

WRAPPER_PATH = "/tisch"
GEOM_PATH = "/tisch/Geom"


def world_bbox(stage: Usd.Stage, prim_path: str | None = None) -> dict:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    prim = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()
    if not prim or not prim.IsValid():
        return {"error": f"prim not found: {prim_path}"}
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return {"empty": True}
    lo = [float(rng.GetMin()[i]) for i in range(3)]
    hi = [float(rng.GetMax()[i]) for i in range(3)]
    return {"min": lo, "max": hi, "size": [hi[i] - lo[i] for i in range(3)]}


def instanceable_prims(stage: Usd.Stage) -> list[str]:
    return [str(p.GetPath()) for p in stage.Traverse() if p.IsInstanceable()]


def deinstance_all(stage: Usd.Stage) -> list[str]:
    """Clear the instanceable flag everywhere, repeating for nested cases.

    Traverse() does not descend into instance prototypes, so a prim hidden below
    an instanced one only becomes visible after its parent is de-instanced.
    """
    cleared: list[str] = []
    for _ in range(10):
        found = [p for p in stage.Traverse() if p.IsInstanceable()]
        if not found:
            break
        for prim in found:
            prim.SetInstanceable(False)
            cleared.append(str(prim.GetPath()))
    return cleared


def referenced_context_measure(usd_path: pathlib.Path, spawn_pos) -> dict:
    """Measure the asset the way Isaac Lab actually uses it.

    Authors translate/orient/scale on the referencing prim -- exactly the op set
    the spawner was measured to write -- so a unit scale that cannot survive
    that will show up here rather than three steps later in a training run.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    xf = UsdGeom.Xform.Define(stage, "/World/Fixture")
    xf.AddTranslateOp().Set(Gf.Vec3d(*spawn_pos))
    xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    xf.GetPrim().GetReferences().AddReference(str(usd_path))
    return {
        "bbox": world_bbox(stage, "/World/Fixture"),
        "instanceable_prims": instanceable_prims(stage),
    }


def check_candidate(candidate: pathlib.Path) -> dict:
    """Direct-open contract check plus the referenced-context check."""
    tol = args_cli.tol
    direct_stage = Usd.Stage.Open(str(candidate))
    if direct_stage is None:
        return {"checks": {"candidate_opens": False}, "pass": False}

    direct = {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(direct_stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(direct_stage)),
        "default_prim": str(direct_stage.GetDefaultPrim().GetPath()) if direct_stage.GetDefaultPrim() else None,
        "bbox": world_bbox(direct_stage),
        "instanceable_prims": instanceable_prims(direct_stage),
        "external_arcs": [
            str(p.GetPath())
            for p in direct_stage.Traverse()
            if p.HasAuthoredPayloads()
        ],
    }
    ref = referenced_context_measure(candidate, args_cli.spawn_pos)

    checks: dict[str, bool] = {
        "candidate_opens": True,
        "meters_per_unit_is_1": abs(direct["meters_per_unit"] - 1.0) < 1e-9,
        "up_axis_is_z": direct["up_axis"] == "Z",
        "default_prim_set": direct["default_prim"] is not None,
        "no_payloads": not direct["external_arcs"],
        "no_instanceable_prims_direct": not direct["instanceable_prims"],
        # The two that the old pipeline never tested:
        "no_instanceable_prims_referenced": not ref["instanceable_prims"],
    }
    dbox, rbox = direct["bbox"], ref["bbox"]
    if "size" in dbox:
        for i, axis in enumerate("xyz"):
            checks[f"direct_extent_{axis}"] = abs(dbox["size"][i] - args_cli.expect_extents[i]) < tol
        checks["direct_z_min"] = abs(dbox["min"][2] - args_cli.expect_z_range[0]) < tol
        checks["direct_z_max"] = abs(dbox["max"][2] - args_cli.expect_z_range[1]) < tol
    else:
        checks["direct_geometry_present"] = False
    if "size" in rbox:
        for i, axis in enumerate("xyz"):
            checks[f"referenced_extent_{axis}"] = abs(rbox["size"][i] - args_cli.expect_extents[i]) < tol
        # With the D-022 spawn offset the legs must land on the ground plane and
        # the plate top at 0.755 m.
        checks["referenced_legs_on_ground"] = abs(rbox["min"][2] - 0.0) < tol
        checks["referenced_plate_top"] = abs(rbox["max"][2] - (args_cli.spawn_pos[2] + args_cli.expect_z_range[1])) < tol
    else:
        checks["referenced_geometry_present"] = False

    return {"direct": direct, "referenced": ref, "checks": checks, "pass": all(checks.values())}


def main() -> int:
    usd_path = pathlib.Path(args_cli.usd).resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"No such USD file: {usd_path}")
    print(f"[fix_fixture_asset] resolved input: {usd_path}")

    src = Usd.Stage.Open(str(usd_path))
    if src is None:
        raise RuntimeError(f"Could not open stage: {usd_path}")
    src_default = src.GetDefaultPrim()
    before = {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(src)),
        "up_axis": str(UsdGeom.GetStageUpAxis(src)),
        "default_prim": str(src_default.GetPath()) if src_default else None,
        "instanceable_prims": instanceable_prims(src),
        "bbox_direct": world_bbox(src),
        "referenced_context": referenced_context_measure(usd_path, args_cli.spawn_pos),
    }
    print("[fix_fixture_asset] BEFORE:")
    print(json.dumps(before, indent=2))

    if not src_default:
        raise RuntimeError("The asset has no defaultPrim; set one in Isaac Sim before repairing.")

    print(f"[fix_fixture_asset] plan: wrap {before['default_prim']} under {GEOM_PATH}, "
          f"clean {WRAPPER_PATH} as defaultPrim, de-instance, flatten")
    if args_cli.dry_run:
        print("[fix_fixture_asset] --dry-run: nothing written.")
        return 0

    # Build the wrapper: an op-free default prim whose child carries the asset.
    out = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(out, 1.0)
    UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
    wrapper = UsdGeom.Xform.Define(out, WRAPPER_PATH)
    geom = UsdGeom.Xform.Define(out, GEOM_PATH)
    if not geom.GetPrim().GetReferences().AddReference(str(usd_path)):
        raise RuntimeError(f"Could not reference {usd_path}")
    out.SetDefaultPrim(wrapper.GetPrim())
    cleared = deinstance_all(out)
    print(f"[fix_fixture_asset] de-instanced: {cleared or 'nothing was instanceable'}")

    backup = usd_path.with_suffix(usd_path.suffix + ".prefix.bak")
    shutil.copy2(usd_path, backup)
    print(f"[fix_fixture_asset] backup written: {backup}")

    candidate = usd_path.with_name(usd_path.stem + ".wrapped" + usd_path.suffix)
    out.Flatten().Export(str(candidate))
    print(f"[fix_fixture_asset] candidate exported: {candidate}")

    # De-instancing may need repeating once the reference is flattened in.
    cand_stage = Usd.Stage.Open(str(candidate))
    again = deinstance_all(cand_stage)
    if again:
        cand_stage.GetRootLayer().Save()
        print(f"[fix_fixture_asset] de-instanced after flatten: {again}")

    verdict = check_candidate(candidate)
    print("[fix_fixture_asset] AFTER (fresh stage, candidate file):")
    print(json.dumps(verdict, indent=2))

    report = {
        "input": str(usd_path),
        "backup": str(backup),
        "candidate": str(candidate),
        "before": before,
        "after": verdict,
        "spawn_pos_m": list(args_cli.spawn_pos),
        "tolerance_m": args_cli.tol,
    }
    report_path = usd_path.with_suffix(".fix_asset.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[fix_fixture_asset] wrote {report_path}")

    for name, passed in verdict["checks"].items():
        print(f"[fix_fixture_asset] {name}: {'PASS' if passed else 'FAIL'}")

    if not verdict["pass"]:
        failed = [k for k, v in verdict["checks"].items() if not v]
        print(f"[fix_fixture_asset] CONTRACT FAIL on {failed}")
        print(f"[fix_fixture_asset] original left untouched; inspect {candidate} and the report.")
        return 1

    # THE CLEAN TEST from HANDOFF-SZENE (RT-32 vs RT-35): fix_stage_units.py
    # releases its stage references and calls gc.collect() before the copy and
    # its replace WORKED (RT-32); this script held `cand_stage` open and its
    # replace went silent (RT-35). Port the release step, wrap the copy, name
    # the exception. If the copy now succeeds, the memory-map hypothesis is
    # supported by a like-for-like run; if it fails WITH the stages released,
    # the mapping is refuted here too and the handle is held elsewhere.
    print("[fix_fixture_asset] releasing stage references before the replace "
          "(discriminating step, RT-35 follow-up)")
    src = None
    out = None
    cand_stage = None
    gc.collect()
    try:
        shutil.copy2(candidate, usd_path)
    except Exception as exc:              # noqa: BLE001 -- naming it IS the job
        print(f"[fix_fixture_asset] REPLACE FAILED: {type(exc).__name__}: {exc}")
        print(f"[fix_fixture_asset] the candidate is correct and stands at {candidate}; "
              f"{usd_path.name} is still the original (backup at {backup.name}).")
        return 2
    print(f"[fix_fixture_asset] contract PASS; {usd_path.name} replaced (backup at {backup.name})")
    print("[fix_fixture_asset] next: re-run verify_fixture_spawn.py; D must read 'metres (correct)'")
    print("[fix_fixture_asset] and the modify_collision_properties warning must be gone.")
    return 0


if __name__ == "__main__":
    # D-081: exit BEFORE the shutdown. RT-34 measured this script printing
    # CONTRACT FAIL and exiting 0 -- close() does not return (RT-22), so the
    # old `finally: close()` swallowed every code.
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
    exit_with(simulation_app, _code, tag="fix_fixture_asset")
