# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Apply the USD Physics collision schema to the fixture meshes.

Measured on the training machine 2026-07-26. After de-instancing, Isaac Lab
still warns:

    Could not perform 'modify_collision_properties' on any prims under:
    '/World/envs/env_0/Fixture' ... (1) The desired attribute does not exist on
    any of the prims. ... Discovered list of instanced prim paths: []

The instanced-prim list is now empty, so the instancing defect is gone and only
reason (1) remains: `modify_collision_properties` **modifies** existing collision
attributes, it does not **apply** the schema. A CAD import carries no
`UsdPhysics.CollisionAPI`, so `collision_props` in the spawn cfg silently reaches
nothing and the fixture has no collision at all.

This applies the schema in the asset, where it belongs -- every consumer of the
file then gets a collider, not just our env.

Approximation is set to ``none``, i.e. the exact triangle mesh. That is the value
the blind bore needs, and PhysX permits it precisely because the fixture is a
static collider with no rigid-body dynamics (D-018). Doing it here also settles
the question that was deferred to the peg increment because `UsdFileCfg` has no
`mesh_collision_props` field (D-020 addendum 2): a convex hull would have sealed
the bore, and now none is used.

Idempotent: applying an API that is already present is a no-op, so re-running is
safe. The input is never edited in place -- backup, candidate, verify, replace
only on pass, matching `fix_stage_units.py`.

UNVERIFIED -- authored on the dev PC (no Isaac installation). On the training
machine:

    conda activate env_isaaclab
    python scripts\\add_fixture_collision.py --usd <path>\\tisch.usd
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Apply collision schema to the fixture meshes.")
parser.add_argument("--usd", type=str, required=True, help="Path to the fixture USD file.")
parser.add_argument(
    "--approximation",
    type=str,
    default="none",
    choices=["none", "convexHull", "convexDecomposition", "boundingCube", "boundingSphere", "meshSimplification"],
    help="Collision approximation; 'none' is the exact triangle mesh the blind bore needs.",
)
parser.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pxr is importable only after the app is up.
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def mesh_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    return [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]


def describe(stage: Usd.Stage) -> dict:
    meshes = mesh_prims(stage)
    return {
        "mesh_count": len(meshes),
        "meshes": [
            {
                "path": str(p.GetPath()),
                "has_collision_api": bool(p.HasAPI(UsdPhysics.CollisionAPI)),
                "has_mesh_collision_api": bool(p.HasAPI(UsdPhysics.MeshCollisionAPI)),
                "approximation": (
                    str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get())
                    if p.HasAPI(UsdPhysics.MeshCollisionAPI)
                    else None
                ),
            }
            for p in meshes
        ],
    }


def main() -> None:
    usd_path = pathlib.Path(args_cli.usd).resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"No such USD file: {usd_path}")
    print(f"[add_fixture_collision] resolved input: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {usd_path}")

    before = describe(stage)
    print("[add_fixture_collision] BEFORE:")
    print(json.dumps(before, indent=2))

    if before["mesh_count"] == 0:
        raise RuntimeError(
            "No UsdGeom.Mesh prims found. Either the asset is still instanced "
            "(run fix_fixture_asset.py first) or the geometry is not mesh-based."
        )

    if args_cli.dry_run:
        print(f"[add_fixture_collision] --dry-run: would apply CollisionAPI + MeshCollisionAPI "
              f"(approximation '{args_cli.approximation}') to {before['mesh_count']} mesh prims.")
        return

    backup = usd_path.with_suffix(usd_path.suffix + ".precollision.bak")
    shutil.copy2(usd_path, backup)
    print(f"[add_fixture_collision] backup written: {backup}")

    for prim in mesh_prims(stage):
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set(args_cli.approximation)
        print(f"[add_fixture_collision] applied to {prim.GetPath()}")

    candidate = usd_path.with_name(usd_path.stem + ".collision" + usd_path.suffix)
    stage.Flatten().Export(str(candidate))
    print(f"[add_fixture_collision] candidate exported: {candidate}")

    check_stage = Usd.Stage.Open(str(candidate))
    if check_stage is None:
        raise RuntimeError(f"Could not re-open the candidate: {candidate}")
    after = describe(check_stage)

    checks = {
        "mesh_count_unchanged": after["mesh_count"] == before["mesh_count"],
        "all_have_collision_api": all(m["has_collision_api"] for m in after["meshes"]),
        "all_have_mesh_collision_api": all(m["has_mesh_collision_api"] for m in after["meshes"]),
        "approximation_set": all(m["approximation"] == args_cli.approximation for m in after["meshes"]),
    }
    print("[add_fixture_collision] AFTER (fresh stage, candidate file):")
    print(json.dumps({"describe": after, "checks": checks}, indent=2))

    report = {
        "input": str(usd_path),
        "backup": str(backup),
        "candidate": str(candidate),
        "approximation": args_cli.approximation,
        "before": before,
        "after": after,
        "checks": checks,
    }
    report_path = usd_path.with_suffix(".collision.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[add_fixture_collision] wrote {report_path}")

    for name, passed in checks.items():
        print(f"[add_fixture_collision] {name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        print("[add_fixture_collision] FAIL; original left untouched.")
        return

    # The stage above still has the input open, and on Windows that can block the
    # overwrite. Drop it before copying, and report the failure instead of dying
    # silently -- the previous repair run lost its replace step exactly here.
    del stage, check_stage
    try:
        shutil.copy2(candidate, usd_path)
    except OSError as exc:
        print(f"[add_fixture_collision] could not replace the original: {exc}")
        print(f"[add_fixture_collision] copy it manually: {candidate} -> {usd_path}")
        return
    print(f"[add_fixture_collision] PASS; {usd_path.name} replaced (backup at {backup.name})")
    print("[add_fixture_collision] next: re-run zero_agent; the modify_collision_properties warning must be gone.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
