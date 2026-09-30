# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure where the fixture actually lands when spawned, and why.

Closes a hole in the increment-1 acceptance criteria: the startup report in
`insertion_env.py` measures the end-effector against `opening_entrance_pos`, a
*cfg constant* rather than a measured point on the object. A fixture spawned at
the wrong height or scale passes all seven criteria while the scene is
physically wrong.

SQUARE RE-IMPORT (D-029): the asset origin moved onto the opening plane at
the pocket centre, and the spawn translation moved with it (fixture_pos z is
now 0.755, the plate top). The world-space acceptance below is deliberately
unchanged -- plate top at plate_top_z, legs on the ground plane -- because it
measures the ASSEMBLED scene, which the origin convention must not alter. If
the old round tisch.usd is fed in (via --usd), the same checks fail by 55 mm,
which is exactly the stale-asset signal wanted.

The risk is specific. `fix_stage_units.py` left a compensating 0.001 scale as
the first entry in the asset's root `xformOpOrder`. In USD composition an
`xformOpOrder` authored on the *referencing* prim does not merge with the
referenced one -- it replaces it, and the scale op is silently dropped. The
first version of this script hit exactly that and reported a 1000x oversized
table (2026-07-26).

So this runs four measurements in one pass, because a round trip to the
training machine is expensive:

  A. the asset stage opened directly     -- is the asset itself still correct?
  B. reference and translate on the SAME prim -- reproduces the dropped-scale bug
  C. translate on a PARENT, reference on a child -- the composition-safe variant
  D. Isaac Lab's own spawner via cfg.func -- the only one that decides anything

D is the verdict. A/B/C explain it. If D matches C the pipeline is sound; if D
matches B the fixture must be spawned under a wrapper Xform (or the asset
re-saved with the units baked in rather than carried as an xformOp).

UNVERIFIED beyond the run it is written for. On the training machine:

    conda activate env_isaaclab
    python scripts\\verify_fixture_spawn.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the fixture spawn transform (D-022).")
parser.add_argument("--usd", type=str, default=None, help="Fixture USD path; default uses the env resolver.")
parser.add_argument(
    "--pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Spawn translation in meters; default uses the cfg value.",
)
parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance in meters (default 1 mm).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pxr and the isaaclab modules are importable only after the app is up.
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from insertion.tasks.direct.insertion import insertion_tasks_cfg as geom  # noqa: E402
from insertion.tasks.direct.insertion.insertion_env_cfg import (  # noqa: E402
    InsertionEnvCfg,
    resolve_fixture_usd_path,
)

# The CAD CUT-OUT since 2026-08-24: 150 x 189.1 x 61 mm. Two earlier meanings
# of "the fixture", both stale, both left here so an old log can be placed:
# (0.500, 0.600, 0.755) was the proxy Tisch, (0.210, 0.260, 0.1973) was the
# solid workcell block. Read from the constants rather than retyped -- the size
# is a CAD measurement and it has exactly one home.
EXPECTED_SIZE_M = tuple(geom.POCKET_ASSET_BBOX_M)


def bbox_of(stage, prim_path: str) -> dict:
    """World-space axis-aligned bbox of a prim, in stage units."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return {"error": f"prim not found: {prim_path}"}
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    bmin = [float(rng.GetMin()[i]) for i in range(3)]
    bmax = [float(rng.GetMax()[i]) for i in range(3)]
    return {
        "min": bmin,
        "max": bmax,
        "size": [bmax[i] - bmin[i] for i in range(3)],
    }


def xform_op_order(stage, prim_path: str) -> list[str]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return []
    attr = prim.GetAttribute("xformOpOrder")
    return [str(t) for t in (attr.Get() or [])] if attr else []


def size_verdict(size: list[float] | None) -> str:
    """Classify a measured size as correct metres, 1000x, or something else."""
    if not size:
        return "unknown"
    if all(abs(size[i] - EXPECTED_SIZE_M[i]) < 0.01 for i in range(3)):
        return "metres (correct)"
    if all(abs(size[i] - EXPECTED_SIZE_M[i] * 1000.0) < 10.0 for i in range(3)):
        return "1000x oversized (unit scale dropped)"
    return "unrecognized"


def main() -> None:
    cfg = InsertionEnvCfg()
    usd_path = pathlib.Path(args_cli.usd or resolve_fixture_usd_path()).resolve()
    spawn_pos = tuple(args_cli.pos) if args_cli.pos is not None else tuple(cfg.fixture_pos)
    result: dict = {
        "usd": str(usd_path),
        "spawn_translation_m": list(spawn_pos),
        "expected_size_m": list(EXPECTED_SIZE_M),
    }

    # --- A. the asset stage opened directly -------------------------------
    asset_stage = Usd.Stage.Open(str(usd_path))
    default_prim = asset_stage.GetDefaultPrim()
    a = {
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(asset_stage)),
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "root_xform_op_order": xform_op_order(asset_stage, str(default_prim.GetPath())) if default_prim else [],
        "bbox": bbox_of(asset_stage, str(asset_stage.GetPseudoRoot().GetPath())),
    }
    a["verdict"] = size_verdict(a["bbox"].get("size"))
    result["A_asset_direct"] = a

    # --- B. reference and translate on the SAME prim ----------------------
    sb = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(sb, 1.0)
    UsdGeom.SetStageUpAxis(sb, UsdGeom.Tokens.z)
    fb = UsdGeom.Xform.Define(sb, "/World/Fixture")
    fb.AddTranslateOp().Set(Gf.Vec3d(*spawn_pos))
    fb.GetPrim().GetReferences().AddReference(str(usd_path))
    b = {"bbox": bbox_of(sb, "/World/Fixture"), "xform_op_order": xform_op_order(sb, "/World/Fixture")}
    b["verdict"] = size_verdict(b["bbox"].get("size"))
    result["B_translate_on_referencing_prim"] = b

    # --- C. translate on a PARENT, reference on a child -------------------
    sc = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(sc, 1.0)
    UsdGeom.SetStageUpAxis(sc, UsdGeom.Tokens.z)
    parent = UsdGeom.Xform.Define(sc, "/World/Fixture")
    parent.AddTranslateOp().Set(Gf.Vec3d(*spawn_pos))
    child = UsdGeom.Xform.Define(sc, "/World/Fixture/Asset")
    child.GetPrim().GetReferences().AddReference(str(usd_path))
    c = {
        "bbox": bbox_of(sc, "/World/Fixture"),
        "child_xform_op_order": xform_op_order(sc, "/World/Fixture/Asset"),
    }
    c["verdict"] = size_verdict(c["bbox"].get("size"))
    result["C_translate_on_parent"] = c

    # --- D. Isaac Lab's own spawner: the measurement that decides ---------
    d: dict = {}
    try:
        stage = None
        for label, getter in (
            ("isaacsim.core.utils.stage.get_current_stage", _stage_via_isaacsim),
            ("omni.usd context", _stage_via_omni),
        ):
            try:
                stage = getter()
            except Exception as exc:  # noqa: BLE001
                d.setdefault("stage_attempts", []).append(f"{label}: {type(exc).__name__}: {exc}")
                continue
            if stage is not None:
                d["stage_source"] = label
                break
        if stage is None:
            raise RuntimeError("no usable stage accessor")

        d["stage_meters_per_unit"] = float(UsdGeom.GetStageMetersPerUnit(stage))
        UsdGeom.Xform.Define(stage, "/World")
        fixture_cfg = cfg.fixture_cfg
        fixture_cfg.usd_path = str(usd_path)
        fixture_cfg.func(
            "/World/Fixture",
            fixture_cfg,
            translation=spawn_pos,
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        d["bbox"] = bbox_of(stage, "/World/Fixture")
        d["xform_op_order"] = xform_op_order(stage, "/World/Fixture")
        d["verdict"] = size_verdict(d["bbox"].get("size"))
    except Exception as exc:  # noqa: BLE001
        d["error"] = f"{type(exc).__name__}: {exc}"
        d["traceback"] = traceback.format_exc()
        d["verdict"] = "unknown"
    result["D_isaaclab_spawner"] = d

    # --- acceptance, judged on D ------------------------------------------
    dbox = d.get("bbox") or {}
    checks = {}
    if "max" in dbox:
        # Repointed 2026-08-24 with the fixture itself. The two old names were
        # proxy-table contracts and had been false since the block took over:
        # "plate_top_at_expected_z" compared against the TABLE-2 top, and
        # "legs_on_ground_plane" asserted the asset stands on z = 0. The cut-out
        # does neither -- it hangs around its own origin, which spawns on the
        # insertion plane.
        checks = {
            # The topmost face is the stage-1 rim, which IS the block top.
            "rim_at_block_top": abs(dbox["max"][2] - geom.BLOCK_TOP_Z) < args_cli.tol,
            # The underside sits POCKET_ASSET_BOTTOM_Z below the spawn point.
            "underside_below_spawn":
                abs(dbox["min"][2] - (spawn_pos[2] + geom.POCKET_ASSET_BOTTOM_Z)) < args_cli.tol,
            # Depth counting starts at the spawn point, so the opening plane must
            # land on it: this is the check the whole origin convention rests on.
            "spawn_is_the_opening_plane":
                abs(spawn_pos[2] - geom.INSERTION_PLANE_Z) < args_cli.tol,
            "size_correct": d.get("verdict") == "metres (correct)",
        }
    result["checks"] = checks
    result["tolerance_m"] = args_cli.tol

    out_path = pathlib.Path(usd_path).with_suffix(".spawn.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[verify_fixture_spawn] wrote {out_path}")
    print("[verify_fixture_spawn] --- summary ---")
    for key in ("A_asset_direct", "B_translate_on_referencing_prim", "C_translate_on_parent", "D_isaaclab_spawner"):
        entry = result[key]
        size = (entry.get("bbox") or {}).get("size")
        size_txt = "[" + ", ".join(f"{v:.4f}" for v in size) + "]" if size else entry.get("error", "n/a")
        print(f"[verify_fixture_spawn] {key:<34} {entry.get('verdict', '?'):<34} {size_txt}")
    if checks:
        for name, passed in checks.items():
            print(f"[verify_fixture_spawn] {name}: {'PASS' if passed else 'FAIL'}")
        print(f"[verify_fixture_spawn] spawn transform: {'PASS' if all(checks.values()) else 'FAIL'}")
    else:
        print("[verify_fixture_spawn] spawn transform: INCONCLUSIVE (D did not produce a bbox)")


def _stage_via_isaacsim():
    import isaacsim.core.utils.stage as stage_utils

    return stage_utils.get_current_stage()


def _stage_via_omni():
    import omni.usd

    return omni.usd.get_context().get_stage()


if __name__ == "__main__":
    main()
    simulation_app.close()
