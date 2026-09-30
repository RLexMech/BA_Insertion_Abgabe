# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Author the SQUARE peg as a welded link into a copy of the shipped UR10e USD.

D-005 models the peg as a rigid body fixed to the flange, with no gripper.
D-019 settles *how*: a welded link inside the robot USD, not a separately
spawned body joined at scene-build time. That path was tried and fails with
"no bodies defined at body0/body1 ... Failed to create articulation", and the
Robot Assembler strips the ArticulationRoot API (D-018 addendum). A link inside
the asset makes the peg's mass and inertia part of the articulation and gives a
contact sensor a stable target -- the condition under which the welded variant
was chosen.

SQUARE-PEG PIVOT (D-029): the geometry is a 30 x 30 x 50 mm square prism, an
explicit 8-vertex box mesh. Two points that matter and are easy to get wrong:

1. **The box is authored axis-aligned to flange-local x/y, and explicitly
   so.** With the identity weld below, the prism's faces are normal to the
   flange's x and y axes, so the peg's yaw about the tool axis IS the flange
   yaw the startup report measures. The tempting shortcut -- reusing the old
   cylinder tessellation with segments=4 -- produces a square rotated 45
   degrees (corners on the axes), i.e. a second yaw zero point that could
   silently cancel or double an equal error in the pocket asset. The env's
   phi report line exists to catch exactly this class of mistake; do not
   hand it one.
2. Collision approximation stays **convexHull**, not the ``none`` used for
   the fixture. That is not a preference: the fixture is a static collider,
   so PhysX accepts its exact triangle mesh, while the peg is part of a
   dynamic articulation, and PhysX permits only convex shapes for dynamic
   bodies. A box is its own convex hull, so unlike the cylinder there is no
   faceting error at all -- the old faceting metric is gone because it is
   identically zero, not because it stopped mattering.

The input is the shipped asset and is never written to. Output is a new file,
backup/candidate/verify/replace as in ``add_fixture_collision.py``.

UNVERIFIED -- authored on the dev PC (no Isaac installation). On the training
machine, where the output should also be produced so the USD never crosses
machines:

    conda activate env_isaaclab
    python scripts\\author_peg_ur10e.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil

from isaaclab.app import AppLauncher

# Bumped whenever this file changes. The two-machine workflow copies scripts by
# hand, and a stale copy has already cost two "it does not work" detours; the
# first line of output makes the copy's identity visible before anything else.
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

# pxr and the isaaclab/proxytask modules are importable only after the app is up.
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

from isaaclab_assets.robots.universal_robots import UR10e_CFG  # noqa: E402

from proxytask.tasks.direct.proxytask.proxytask_tasks_cfg import (  # noqa: E402
    PEG_I_AXIAL,
    PEG_I_TRANS_COM,
    PEG_I_TRANS_FLANGE,
    PEG_LENGTH,
    PEG_MASS,
    PEG_SIDE,
    # The clearance is read from the constants, not written out. It was a
    # literal "1.000 mm" until 2026-07-27, which kept printing the stage-1
    # value after the peg had moved -- the same stale-constant failure that
    # let a superseded peg be re-authored on 2026-07-26.
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
    """Every physics joint prim.

    ``IsA(UsdPhysics.Joint)`` alone would rest on an abstract base schema
    matching its concrete subclasses. If it silently matched nothing, the
    before and after counts would both be zero and the ``one_new_joint`` check
    would reject a correctly authored asset. The schema-instance test is the
    idiomatic one; the type-name test is the fallback that makes the count
    independent of that behaviour.
    """
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
    """Return (points, faceVertexCounts, faceVertexIndices) for a closed box.

    The long axis is +z and the mesh spans z in [0, length], so the peg starts
    at the flange and its tip is at z = length. The cross-section is a
    ``side`` x ``side`` square whose faces are normal to LOCAL x and y --
    axis-aligned by construction, corners at (+-side/2, +-side/2). This is
    deliberate and load-bearing: the identity weld makes local x/y the flange
    axes, so the authored yaw zero point is the flange's, which is what the
    env's phi report line measures against the pocket. A 45-degree-rotated
    square (corners on the axes, what cylinder tessellation with segments=4
    would have produced) would be a second, hidden yaw zero point.

    Eight vertices, six quads, wound outward (CCW seen from outside). Exact:
    a box is its own convex hull, so there is no faceting error to report.
    """
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
        0, 3, 2, 1,  # bottom, normal -z (the flange plane)
        4, 5, 6, 7,  # top, normal +z (the tip face)
        0, 1, 5, 4,  # normal -y
        1, 2, 6, 5,  # normal +x
        2, 3, 7, 6,  # normal +y
        3, 0, 4, 7,  # normal -x
    ]
    return points, counts, indices


def author_peg(stage: Usd.Stage, flange: Usd.Prim, parent: Usd.Prim, approximation: str) -> dict:
    """Author peg_link + geometry + mass + collision + the weld joint."""
    peg_path = parent.GetPath().AppendChild(PEG_LINK_NAME)

    # peg_link must reproduce the flange's world transform. Copying the flange's
    # *local* transform would only be correct if the two shared a parent, which
    # is not guaranteed, so the local transform is derived from the two world
    # transforms instead.
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    flange_world = cache.GetLocalToWorldTransform(flange)
    parent_world = cache.GetLocalToWorldTransform(parent)
    peg_local = flange_world * parent_world.GetInverse()

    peg_xform = UsdGeom.Xform.Define(stage, peg_path)
    peg_xform.AddTransformOp().Set(peg_local)
    peg_prim = peg_xform.GetPrim()

    # Geometry. An explicit box mesh, axis-aligned to the flange axes; see the
    # module docstring and box_mesh_points for why NOT a rotated square.
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

    # Physics on the link. The inertia is about the centre of mass, which is
    # what MassAPI:diagonalInertia means; PEG_I_TRANS_FLANGE is 3.4x larger and
    # would silently make the peg harder to tilt.
    UsdPhysics.RigidBodyAPI.Apply(peg_prim)
    mass_api = UsdPhysics.MassAPI.Apply(peg_prim)
    mass_api.CreateMassAttr().Set(float(PEG_MASS))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, float(PEG_LENGTH / 2.0)))
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(float(PEG_I_TRANS_COM), float(PEG_I_TRANS_COM), float(PEG_I_AXIAL))
    )
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Collision on the geometry. convexHull is required, not chosen: PhysX
    # accepts only convex shapes for dynamic bodies.
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set(approximation)

    # The weld. Both local frames are identity because peg_link was placed at
    # the flange's world transform above.
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
        # The world bound is a diagnostic, NOT the size check. peg_link carries
        # the flange's rest transform, which rotates the peg's local +z onto a
        # different world axis, so a world-space box compared against local
        # nominal dimensions rejects a correct asset -- it reads
        # (0.030, 0.050, 0.030) instead of (0.030, 0.030, 0.050). That the
        # measurement is an exact permutation is itself informative: it proves
        # the rest orientation is a clean 90 degree step rather than a skew,
        # which would inflate the box in more than one axis. For the square
        # peg a skew ABOUT the tool axis would inflate x and y specifically
        # (up to sqrt 2 * side = 42.43 mm on the diagonal) -- if this line
        # ever reports ~0.042, that is a real 45-degree yaw offset in the
        # asset, not a measurement artefact, and "fixing" the check to accept
        # it would make the gate, the clearance and the metrics file
        # consistently wrong together.
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = cache.ComputeWorldBound(geom).ComputeAlignedRange()
        measured["world_bbox_size_m"] = [float(rng.GetMax()[i] - rng.GetMin()[i]) for i in range(3)]

        # The real check: the authored points, in their own space, with no
        # transform semantics involved.
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
    # Both cross-section axes carry information now: a wrong-aspect box
    # (side x diagonal, say) would pass a single-axis check.
    nominal = (PEG_SIDE, PEG_SIDE, PEG_LENGTH)

    def close(got: float | None, want: float) -> bool:
        """Relative comparison. USD ``physics:mass`` and ``physics:diagonalInertia``
        are float attributes, so every value written comes back rounded to
        float32. An absolute tolerance here is a trap: 0.03202 round-trips with
        an error of 1.12e-09, which an earlier 1e-09 tolerance rejected -- the
        check would have failed on a correctly authored asset."""
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
        # Asset-level wrong-side test: the peg must start at the flange plane
        # and run to +PEG_LENGTH. Authored backwards it would span [-0.050, 0],
        # which every size check above would still call correct.
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

    # The env resolver's own default, so a successful run is immediately visible
    # to the env without setting any environment variable. Asked for by name
    # rather than derived from the resolver's failure message, which only works
    # until someone rewords it.
    out_path = pathlib.Path(args_cli.out).resolve() if args_cli.out else pathlib.Path(default_peg_robot_usd_path())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[author_peg] output: {out_path}")

    # Steiner self-test on the numbers actually being written. If the task
    # specification's two inertia figures ever drift apart, this catches it
    # before a run, not after.
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

    # A rigid body may not be nested inside another rigid body, so peg_link goes
    # under the default prim. Checked rather than assumed: if the default prim
    # ever carries RigidBodyAPI, the authored asset would be silently invalid.
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

    # Flatten first: the shipped asset references other layers (possibly on
    # Nucleus), and the result has to stand alone on the training machine.
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
    # No faceting line any more: the box mesh is exact (its own convex hull).
    # The numbers that now decide passability are the pair below.
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
