# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Create a UR10e USD copy with a rigidly attached square peg."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil

from isaaclab.app import AppLauncher

SCRIPT_MARKER = "author_peg_ur10e-2026-07-28a-square"

parser = argparse.ArgumentParser(description="Weld the square peg into a copy of the UR10e USD (D-019, D-029).")
parser.add_argument("--src", type=str, default=None, help="Source UR10e USD; default is the shipped UR10e_CFG path.")
parser.add_argument("--out", type=str, default=None, help="Output USD; default is the package assets/Robot path.")
parser.add_argument("--peg-approx", type=str, default="convexHull", help="Collision approximation for the peg.")
parser.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

print(f"[author_peg] marker: {SCRIPT_MARKER}; sections: describe, author, verify, replace")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

from isaaclab_assets.robots.universal_robots import UR10e_CFG  # noqa: E402

from proxytask.tasks.direct.proxytask.proxytask_tasks_cfg import (  # noqa: E402
    PEG_I_AXIAL,
    PEG_I_TRANS_COM,
    PEG_I_TRANS_FLANGE,
    PEG_LENGTH,
    PEG_MASS,
    PEG_SIDE,
    SIDE_CLEARANCE,
    YAW_WINDOW_RAD,
    default_peg_robot_usd_path,
)

FLANGE_BODY = "wrist_3_link"
PEG_LINK_NAME = "peg_link"
PEG_GEOM_NAME = "geometry"
PEG_JOINT_NAME = "peg_weld"


def rigid_body_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    return [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def joint_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    """Every physics joint prim."""
    return [
        p
        for p in stage.Traverse()
        if bool(UsdPhysics.Joint(p)) or p.IsA(UsdPhysics.Joint) or str(p.GetTypeName()).endswith("Joint")
    ]


def articulation_roots(stage: Usd.Stage) -> list[str]:
    return [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]


def find_prim_by_name(stage: Usd.Stage, name: str) -> Usd.Prim | None:
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def describe(stage: Usd.Stage) -> dict:
    default_prim = stage.GetDefaultPrim()
    bodies = rigid_body_prims(stage)
    joints = joint_prims(stage)
    return {
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "rigid_body_count": len(bodies),
        "rigid_bodies": [str(p.GetPath()) for p in bodies],
        "joint_count": len(joints),
        "joints": [str(p.GetPath()) for p in joints],
        "articulation_roots": articulation_roots(stage),
    }


def box_mesh_points(side: float, length: float) -> tuple[list, list, list]:
    """Return (points, faceVertexCounts, faceVertexIndices) for a closed box."""
    h = side / 2.0
    points = [
        Gf.Vec3f(-h, -h, 0.0),
        Gf.Vec3f(h, -h, 0.0),
        Gf.Vec3f(h, h, 0.0),
        Gf.Vec3f(-h, h, 0.0),
        Gf.Vec3f(-h, -h, length),
        Gf.Vec3f(h, -h, length),
        Gf.Vec3f(h, h, length),
        Gf.Vec3f(-h, h, length),
    ]
    counts = [4, 4, 4, 4, 4, 4]
    indices = [
        0, 3, 2, 1,
        4, 5, 6, 7,
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
        3, 0, 4, 7,
    ]
    return points, counts, indices


def author_peg(stage: Usd.Stage, flange: Usd.Prim, parent: Usd.Prim, approximation: str) -> dict:
    """Author peg_link + geometry + mass + collision + the weld joint."""
    peg_path = parent.GetPath().AppendChild(PEG_LINK_NAME)

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    flange_world = cache.GetLocalToWorldTransform(flange)
    parent_world = cache.GetLocalToWorldTransform(parent)
    peg_local = flange_world * parent_world.GetInverse()

    peg_xform = UsdGeom.Xform.Define(stage, peg_path)
    peg_xform.AddTransformOp().Set(peg_local)
    peg_prim = peg_xform.GetPrim()

    half = PEG_SIDE / 2.0
    points, counts, indices = box_mesh_points(PEG_SIDE, PEG_LENGTH)
    mesh = UsdGeom.Mesh.Define(stage, peg_path.AppendChild(PEG_GEOM_NAME))
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    mesh.CreateExtentAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(-half, -half, 0.0), Gf.Vec3f(half, half, PEG_LENGTH)])
    )
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.45, 0.0)]))

    UsdPhysics.RigidBodyAPI.Apply(peg_prim)
    mass_api = UsdPhysics.MassAPI.Apply(peg_prim)
    mass_api.CreateMassAttr().Set(float(PEG_MASS))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, float(PEG_LENGTH / 2.0)))
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(float(PEG_I_TRANS_COM), float(PEG_I_TRANS_COM), float(PEG_I_AXIAL))
    )
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set(approximation)

    joint_path = parent.GetPath().AppendChild(PEG_JOINT_NAME)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([flange.GetPath()])
    joint.CreateBody1Rel().SetTargets([peg_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    return {
        "peg_path": str(peg_path),
        "geometry_path": str(mesh.GetPath()),
        "joint_path": str(joint_path),
        "cross_section": "square",
        "side_m": float(PEG_SIDE),
    }


def verify(stage: Usd.Stage, before: dict, approximation: str) -> tuple[dict, dict]:
    after = describe(stage)
    peg = find_prim_by_name(stage, PEG_LINK_NAME)
    geom = stage.GetPrimAtPath(peg.GetPath().AppendChild(PEG_GEOM_NAME)) if peg else None
    joint_prim = find_prim_by_name(stage, PEG_JOINT_NAME)

    measured = {}
    if geom and geom.IsValid():
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeWorldBound(geom).ComputeAlignedRange()
        measured["world_bbox_size_m"] = [float(rng.GetMax()[i] - rng.GetMin()[i]) for i in range(3)]

        points = UsdGeom.Mesh(geom).GetPointsAttr().Get()
        if points:
            lo = [min(float(p[i]) for p in points) for i in range(3)]
            hi = [max(float(p[i]) for p in points) for i in range(3)]
            measured["local_bbox_size_m"] = [hi[i] - lo[i] for i in range(3)]
            measured["local_z_range_m"] = [lo[2], hi[2]]

        measured["approximation"] = (
            str(UsdPhysics.MeshCollisionAPI(geom).GetApproximationAttr().Get())
            if geom.HasAPI(UsdPhysics.MeshCollisionAPI)
            else None
        )
    if peg and peg.IsValid() and peg.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(peg)
        measured["mass_kg"] = float(mass_api.GetMassAttr().Get() or 0.0)
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        measured["diagonal_inertia"] = [float(v) for v in inertia] if inertia else None
        com = mass_api.GetCenterOfMassAttr().Get()
        measured["centre_of_mass_m"] = [float(v) for v in com] if com else None

    targets = []
    if joint_prim and joint_prim.IsValid():
        joint = UsdPhysics.FixedJoint(joint_prim)
        targets = [str(t) for t in (joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets())]
    measured["joint_targets"] = targets

    size = measured.get("local_bbox_size_m") or [0.0, 0.0, 0.0]
    z_range = measured.get("local_z_range_m") or [0.0, 0.0]
    nominal = (PEG_SIDE, PEG_SIDE, PEG_LENGTH)

    def close(got: float | None, want: float) -> bool:
        """Compare relatively to account for float32 rounding of USD mass and inertia."""
        return got is not None and math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-15)

    checks = {
        "peg_link_exists": bool(peg and peg.IsValid()),
        "peg_link_is_rigid_body": bool(peg and peg.HasAPI(UsdPhysics.RigidBodyAPI)),
        "peg_mass_correct": close(measured.get("mass_kg"), PEG_MASS),
        "peg_inertia_is_about_com": (
            measured.get("diagonal_inertia") is not None
            and close(measured["diagonal_inertia"][0], PEG_I_TRANS_COM)
        ),
        "geometry_has_collision": bool(geom and geom.HasAPI(UsdPhysics.CollisionAPI)),
        "approximation_set": measured.get("approximation") == approximation,
        "geometry_size_correct": all(abs(size[i] - nominal[i]) < 1e-4 for i in range(3)),
        "geometry_spans_flange_to_tip": abs(z_range[0]) < 1e-5 and abs(z_range[1] - PEG_LENGTH) < 1e-5,
        "weld_joint_exists": bool(joint_prim and joint_prim.IsValid()),
        "weld_targets_flange_and_peg": (
            len(targets) == 2 and any(FLANGE_BODY in t for t in targets) and any(PEG_LINK_NAME in t for t in targets)
        ),
        "one_new_rigid_body": after["rigid_body_count"] == before["rigid_body_count"] + 1,
        "one_new_joint": after["joint_count"] == before["joint_count"] + 1,
        "articulation_roots_unchanged": after["articulation_roots"] == before["articulation_roots"],
    }
    return {"describe": after, "measured": measured}, checks


def main() -> None:
    src = args_cli.src or UR10e_CFG.spawn.usd_path
    print(f"[author_peg] source: {src}")

    out_path = pathlib.Path(args_cli.out).resolve() if args_cli.out else pathlib.Path(default_peg_robot_usd_path())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[author_peg] output: {out_path}")

    steiner = PEG_I_TRANS_COM + PEG_MASS * (PEG_LENGTH / 2.0) ** 2
    print(f"[author_peg] inertia check: I_com {PEG_I_TRANS_COM:.4e} + m*(L/2)^2 = {steiner:.4e} "
          f"vs documented flange value {PEG_I_TRANS_FLANGE:.4e} "
          f"({'consistent' if abs(steiner - PEG_I_TRANS_FLANGE) < 5e-7 else 'INCONSISTENT'})")

    src_stage = Usd.Stage.Open(str(src))
    if src_stage is None:
        raise RuntimeError(f"Could not open source stage: {src}")
    before = describe(src_stage)
    print("[author_peg] BEFORE:")
    print(json.dumps(before, indent=2))

    flange = find_prim_by_name(src_stage, FLANGE_BODY)
    if flange is None:
        raise RuntimeError(f"No prim named '{FLANGE_BODY}' in {src}; body list: {before['rigid_bodies']}")

    parent = src_stage.GetDefaultPrim()
    if parent is None:
        raise RuntimeError("Source stage has no default prim; cannot place peg_link.")
    if parent.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            f"Default prim {parent.GetPath()} is itself a rigid body. Nesting peg_link under it "
            "would be invalid; place it elsewhere and re-run."
        )
    print(f"[author_peg] flange: {flange.GetPath()}; peg_link parent: {parent.GetPath()}")

    if args_cli.dry_run:
        print(f"[author_peg] --dry-run: would write {PEG_LINK_NAME} + {PEG_JOINT_NAME} under "
              f"{parent.GetPath()} and export to {out_path}. Nothing written.")
        return

    if out_path.is_file():
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.copy2(out_path, backup)
        print(f"[author_peg] previous output backed up: {backup}")
    else:
        backup = None

    candidate = out_path.with_name(out_path.stem + ".candidate" + out_path.suffix)
    src_stage.Flatten().Export(str(candidate))
    print(f"[author_peg] flattened candidate exported: {candidate}")
    del src_stage

    work = Usd.Stage.Open(str(candidate))
    if work is None:
        raise RuntimeError(f"Could not open the flattened candidate: {candidate}")
    work_flange = find_prim_by_name(work, FLANGE_BODY)
    work_parent = work.GetDefaultPrim()
    if work_flange is None or work_parent is None:
        raise RuntimeError("Flattening lost the flange or the default prim; the source asset is not as expected.")

    authored = author_peg(work, work_flange, work_parent, args_cli.peg_approx)
    work.GetRootLayer().Save()
    print("[author_peg] authored:")
    print(json.dumps(authored, indent=2))
    print(f"[author_peg] clearance: {SIDE_CLEARANCE*1000:.3f} mm per axis at zero yaw, "
          f"yaw window +-{math.degrees(YAW_WINDOW_RAD):.2f} deg "
          f"(side {PEG_SIDE*1000:.1f} mm in the 32 mm pocket)")
    del work

    check_stage = Usd.Stage.Open(str(candidate))
    if check_stage is None:
        raise RuntimeError(f"Could not re-open the candidate for verification: {candidate}")
    after, checks = verify(check_stage, before, args_cli.peg_approx)

    print("[author_peg] AFTER (fresh stage, candidate file):")
    print(json.dumps(after, indent=2))
    for name, passed in checks.items():
        print(f"[author_peg] {name}: {'PASS' if passed else 'FAIL'}")

    report = {
        "marker": SCRIPT_MARKER,
        "source": str(src),
        "output": str(out_path),
        "candidate": str(candidate),
        "backup": str(backup) if backup else None,
        "authored": authored,
        "before": before,
        "after": after,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    report_path = out_path.with_suffix(".author.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[author_peg] wrote {report_path}")

    if not all(checks.values()):
        print("[author_peg] FAIL; candidate kept for inspection, output not replaced.")
        return

    del check_stage
    try:
        shutil.copy2(candidate, out_path)
    except OSError as exc:
        print(f"[author_peg] could not write the output: {exc}")
        print(f"[author_peg] copy it manually: {candidate} -> {out_path}")
        return
    print(f"[author_peg] PASS; {out_path.name} written")
    print("[author_peg] next: scripts/list_usd_prims.py on the output, then zero_agent --num_envs 16; "
          "the startup report must show peg_body with a tip 0.0500 m along the tool axis.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
