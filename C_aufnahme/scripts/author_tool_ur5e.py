# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Weld the imported gripper+part assembly into a copy of the UR5e USD.

The real counterpart of the proxy's ``author_peg_ur10e.py``, which welded a
generated BOX peg into the UR10e. That script was DELETED on 2026-08-28 with
the rest of the dead proxy code (S6); read it in git history if the origin of
a line here is in question. Everything structural is ported from it rather
than reinvented: source asset never written to, flatten -> author -> re-open a fresh
stage -> verify -> replace only on a pass, sidecar report, marker line first.

WHY A WELDED LINK AT ALL (D-018/D-019, carried over): a runtime fixed joint
fails with "no bodies defined at body0/body1", and the Robot Assembler strips
the ArticulationRoot API. A link inside the asset makes the tool's mass part of
the articulation and gives a contact sensor a stable target.

THREE DECISIONS THIS SCRIPT ENCODES, all settled with the user 2026-08-23:

  1. ONE link, two meshes. ``tool_link`` carries the whole tool. The gripper is
     rigid by decision (CLAUDE.md), so a second body joined by a second fixed
     joint would model a degree of freedom that does not exist. It also makes
     the CAD's 24.5 mm overlap between gripper adapter and part a non-issue
     structurally, instead of needing a FilteredPairs the repo uses nowhere.
  2. Collision on the PART only, as SDF. Not convexHull: a convex hull bridges
     the gap between the part's two lugs and turns it into a solid 96.41 mm
     block, which cannot enter a 90.588 mm pocket at all (D-075, D-062). The
     gripper gets NO collider -- it never touches the fixture, the cups sit
     inside the part, and the missing collider is what makes decision 1 safe.
  3. Yaw = 0 by definition. The gripper's real mounting angle on the flange is
     a statement about the cell and is in no CAD file; fixing the zero point
     here moves it into the wrist_3 home angle, where it is one measurable
     number. ``--tool-yaw-deg`` exists to correct it later without re-deriving
     anything.

MASSES ARE REQUIRED ARGUMENTS WITH NO DEFAULT. The suction cups are missing
from the CAD, so no density times volume can stand in for a weighing. Mass and
centre of mass are authored (the COM from the two body centroids the assembly
STEP states, weighted by these masses); ``diagonalInertia`` deliberately is
NOT, so PhysX derives it from the collision geometry -- which is the part
alone. That approximation is named in TOOL_CHAIN_PENDING, not hidden.

UNVERIFIED -- authored on the dev PC (no Isaac installation). On the training
machine:

    conda activate env_isaaclab
    python scripts\\author_tool_ur5e.py --part-mass <kg> --gripper-mass <kg> --dry-run
    python scripts\\author_tool_ur5e.py --part-mass <kg> --gripper-mass <kg>
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import sys
import traceback

from isaaclab.app import AppLauncher

# The harness is stdlib only, so it loads before the simulator does. Loaded by
# path because scripts/tools is not a package -- the same idiom
# check_demo_reward_math.py uses for torch_shim.py.
_TOOLS = pathlib.Path(__file__).resolve().parent / "tools"


def _load_tool(stem: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(stem, _TOOLS / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load_tool("checks")
predicates = _load_tool("usd_predicates")
exit_with = _load_tool("isaac_exit").exit_with
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SCRIPT_MARKER = "author_tool_ur5e-2026-08-23b"
"""Stamped into every run and into the sidecar. Two machines share this file by
hand; a stale copy has to be visible in its first output line, not inferred
from a number that looks odd three checks later."""

parser = argparse.ArgumentParser(description="Weld gripper + part into a copy of the UR5e USD.")
parser.add_argument("--src", type=str, default=None, help="Source UR5e USD; default is the shipped UR5E cfg path.")
parser.add_argument("--tool-usd", type=str, default=None, help="Imported gripper+part assembly; default from the resolver.")
parser.add_argument("--out", type=str, default=None, help="Output USD; default is the package assets/Robot path.")
# D-079 stands: authoring without both masses is refused. The requirement moved
# from argparse into main() so that --counter-proof, which reads them back out
# of the sidecar of a finished asset, does not have to restate a weighing.
parser.add_argument("--part-mass", type=float, default=None, help="Mass of the part in kg. No default: it is a weighing, not a derivation. Required unless --counter-proof.")
parser.add_argument("--gripper-mass", type=float, default=None, help="Mass of the gripper incl. the cups missing from the CAD, in kg. Required unless --counter-proof.")
parser.add_argument("--tool-yaw-deg", type=float, default=None, help="Rotation of the tool about the tool axis at the weld. Default: TOOL_WELD_YAW_RAD (zero by definition).")
parser.add_argument("--sdf-resolution", type=int, default=1024, help="SDF resolution for the part collider, as in convert_step_asset.py.")
parser.add_argument("--tol", type=float, default=1.0e-4, help="Geometry tolerance in metres (default 0.1 mm).")
parser.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing.")
parser.add_argument("--counter-proof", action="store_true", help="Author nothing. Break the FINISHED asset one way per declared mutation and require exactly the named checks to flip (D-080).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

print(f"[author_tool] marker: {SCRIPT_MARKER}; sections: describe, author, verify, replace")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pxr and the insertion modules are importable only after the app is up.
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from insertion.tasks.direct.insertion.insertion_tasks_cfg import (  # noqa: E402
    PART_BBOX_M,
    TOOL_GRIPPER_PRIM_NAME,
    TOOL_JOINT_NAME,
    TOOL_LINK_NAME,
    TOOL_PART_PRIM_NAME,
    TOOL_REF_NAME,
    TOOL_SPAN_LOCAL_Z_M,
    TOOL_WELD_YAW_RAD,
    default_tool_robot_usd_path,
    resolve_tool_usd_path,
    tool_com_flange,
)
from insertion.tasks.direct.insertion.ur5e_cfg import UR5E_HOME_CFG  # noqa: E402

FLANGE_BODY = "wrist_3_link"
"""The UR5e ships without an ``ee_link``; the flange body is wrist_3_link
(probed, recorded in ur5e_cfg.py)."""


# --- stage description, ported unchanged from the proxy's author_peg_ur10e.py
def rigid_body_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    return [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def joint_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    """Every physics joint prim; three tests because one of them may match
    nothing silently and a zero-vs-zero count would pass a broken asset."""
    return [
        p
        for p in stage.Traverse()
        if bool(UsdPhysics.Joint(p)) or p.IsA(UsdPhysics.Joint) or str(p.GetTypeName()).endswith("Joint")
    ]


def articulation_roots(stage: Usd.Stage) -> list[str]:
    return [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]


def find_prim_by_name(root: Usd.Prim | Usd.Stage, name: str) -> Usd.Prim | None:
    prims = root.Traverse() if isinstance(root, Usd.Stage) else Usd.PrimRange(root)
    for prim in prims:
        if prim.GetName() == name:
            return prim
    return None


def external_arcs(stage: Usd.Stage) -> list[str]:
    """Composition arcs pointing at another FILE.

    An external arc to a file the training machine does not have resolves to
    nothing and spawns an invisible asset -- the 2026-07-26 failure recorded in
    the fixture scripts. After the final flatten there must be none.

    EMPTY paths are dropped, and that is not a relaxation. A flattened layer
    reports exactly one dependency whose path is the empty string, with no
    sublayer, reference or payload anywhere in the stage to account for it.
    MEASURED, run RT-15 on 2026-08-23: the SHIPPED UR5e produces the very same
    single empty entry after a plain flatten, with no tool involved
    (scripts/diagnose_tool_candidate.py, question Q3). An empty asset path is
    not a file name, so it cannot fail to resolve on the training machine --
    which is the entire risk this function exists to catch. Counting it made
    the check unpassable by any flattened asset, including a perfect one.
    """
    return predicates.external_arcs(stage)


def describe(stage: Usd.Stage) -> dict:
    default_prim = stage.GetDefaultPrim()
    bodies = rigid_body_prims(stage)
    joints = joint_prims(stage)
    return {
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "rigid_body_count": len(bodies),
        "rigid_bodies": [str(p.GetPath()) for p in bodies],
        "joint_count": len(joints),
        "joints": [str(p.GetPath()) for p in joints],
        "articulation_roots": articulation_roots(stage),
        "external_arcs": external_arcs(stage),
    }


def author_tool(
    stage: Usd.Stage,
    flange: Usd.Prim,
    parent: Usd.Prim,
    tool_usd: pathlib.Path,
    part_mass: float,
    gripper_mass: float,
    yaw_rad: float,
    sdf_resolution: int,
) -> dict:
    """Author tool_link + its reference + physics + the weld. Returns what was written."""
    tool_path = parent.GetPath().AppendChild(TOOL_LINK_NAME)

    # tool_link must reproduce the flange's world transform. Copying the
    # flange's *local* transform would only be correct if the two shared a
    # parent, which is not guaranteed, so it is derived from the two world
    # transforms instead.
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    flange_world = cache.GetLocalToWorldTransform(flange)
    parent_world = cache.GetLocalToWorldTransform(parent)
    tool_local = flange_world * parent_world.GetInverse()
    if yaw_rad != 0.0:
        # Rotation about the tool axis, applied in the flange frame, i.e.
        # BEFORE the flange placement. Zero by definition today; the path
        # exists so a measured mounting angle needs no re-derivation.
        spin = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(yaw_rad)))
        tool_local = spin * tool_local

    tool_xform = UsdGeom.Xform.Define(stage, tool_path)
    tool_xform.AddTransformOp().Set(tool_local)
    tool_prim = tool_xform.GetPrim()

    # The geometry arrives as ONE reference to the WHOLE assembly, on an
    # op-free child. Two reasons, both measured rather than assumed:
    #   * referencing the two mesh prims individually would leave behind the
    #     unitsResolve scale of 0.001 that fix_stage_units.py wrote on the
    #     assembly's DEFAULT PRIM (RT-12) -- an ancestor of both meshes -- and
    #     the tool would come in 1000x too large.
    #   * the child stays op-free because xform ops authored on a referencing
    #     prim REPLACE the referenced prim's xformOpOrder (the bug
    #     fix_fixture_asset.py exists for). The placement therefore lives one
    #     level up, on tool_link.
    ref_prim = UsdGeom.Xform.Define(stage, tool_path.AppendChild(TOOL_REF_NAME)).GetPrim()
    ref_prim.GetReferences().AddReference(str(tool_usd))

    part_mesh = find_prim_by_name(ref_prim, TOOL_PART_PRIM_NAME)
    gripper_mesh = find_prim_by_name(ref_prim, TOOL_GRIPPER_PRIM_NAME)
    if part_mesh is None or gripper_mesh is None:
        raise RuntimeError(
            f"The reference to {tool_usd} does not expose both bodies: "
            f"{TOOL_PART_PRIM_NAME}={part_mesh}, {TOOL_GRIPPER_PRIM_NAME}={gripper_mesh}."
        )
    # The named prims are Xforms; the collider belongs on the Mesh below them.
    part_geom = part_mesh if part_mesh.IsA(UsdGeom.Mesh) else find_prim_by_name(part_mesh, "Mesh")
    if part_geom is None or not part_geom.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"No mesh below {part_mesh.GetPath()}; cannot author collision.")

    # Physics on the link. Mass and COM are written; the inertia tensor is
    # left to PhysX on purpose -- see TOOL_CHAIN_PENDING.
    UsdPhysics.RigidBodyAPI.Apply(tool_prim)
    mass_api = UsdPhysics.MassAPI.Apply(tool_prim)
    total_mass = float(part_mass) + float(gripper_mass)
    com = tool_com_flange(part_mass, gripper_mass)
    mass_api.CreateMassAttr().Set(total_mass)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in com]))

    # Collision on the PART mesh only, as SDF. PhysX accepts exact triangle
    # meshes for static colliders only; for a dynamic body the choice is
    # convex or SDF, and convex is ruled out by the lugs (D-075).
    UsdPhysics.CollisionAPI.Apply(part_geom)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(part_geom)
    mesh_collision.CreateApproximationAttr().Set("sdf")
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(part_geom)
    sdf_api.CreateSdfResolutionAttr().Set(int(sdf_resolution))

    # The weld, last. Both local frames are identity because tool_link was
    # placed at the flange's world transform above.
    joint_path = parent.GetPath().AppendChild(TOOL_JOINT_NAME)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([flange.GetPath()])
    joint.CreateBody1Rel().SetTargets([tool_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    return {
        "tool_link": str(tool_path),
        "reference_prim": str(ref_prim.GetPath()),
        "referenced_asset": str(tool_usd),
        "part_geom": str(part_geom.GetPath()),
        "gripper_prim": str(gripper_mesh.GetPath()),
        "joint": str(joint_path),
        "mass_kg": total_mass,
        "centre_of_mass_m": [float(v) for v in com],
        "yaw_rad": float(yaw_rad),
        "sdf_resolution": int(sdf_resolution),
    }



# ===========================================================================
#  Mutations -- the counter-proof (D-080)
# ===========================================================================
# Each one breaks the FINISHED asset in exactly one way. They are DECLARED
# beside the check they must break and are never applied by an authoring run;
# only `--counter-proof` calls them, on an anonymous copy that cannot be saved.
#
# Every ``flips`` set below was DERIVED BY READING describe()/verify(), before
# any of them ran. That is the point: if a derivation is wrong, the run says so,
# and what comes back is a fact about the checks rather than a confirmation.


def _tool_of(stage: Usd.Stage) -> Usd.Prim:
    return find_prim_by_name(stage, TOOL_LINK_NAME)


def _part_geom_of(stage: Usd.Stage) -> Usd.Prim:
    part_mesh = find_prim_by_name(_tool_of(stage), TOOL_PART_PRIM_NAME)
    return part_mesh if part_mesh.IsA(UsdGeom.Mesh) else find_prim_by_name(part_mesh, "Mesh")


def _mut_stage_units_mm(stage: Usd.Stage) -> str:
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    return "metersPerUnit set to 0.001"


def _mut_up_axis_y(stage: Usd.Stage) -> str:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    return "stage up axis set to Y"


def _mut_strip_rigid_body_api(stage: Usd.Stage) -> str:
    _tool_of(stage).RemoveAPI(UsdPhysics.RigidBodyAPI)
    return "RigidBodyAPI removed from tool_link"


def _mut_author_diagonal_inertia(stage: Usd.Stage) -> str:
    UsdPhysics.MassAPI(_tool_of(stage)).CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    return "physics:diagonalInertia authored as (1,1,1)"


def _mut_clear_mass(stage: Usd.Stage) -> str:
    UsdPhysics.MassAPI(_tool_of(stage)).GetMassAttr().Clear()
    return "physics:mass cleared back to its fallback"


def _mut_shift_centre_of_mass(stage: Usd.Stage) -> str:
    attr = UsdPhysics.MassAPI(_tool_of(stage)).GetCenterOfMassAttr()
    v = attr.Get()
    attr.Set(Gf.Vec3f(float(v[0]) + 0.01, float(v[1]), float(v[2])))
    return "centre of mass shifted +10 mm in x"


def _mut_add_external_reference(stage: Usd.Stage) -> str:
    path = _tool_of(stage).GetPath().AppendChild("mutation_ref")
    UsdGeom.Xform.Define(stage, path).GetPrim().GetReferences().AddReference("./no_such_file.usd")
    return "reference to ./no_such_file.usd added"


def _mut_collide_the_gripper(stage: Usd.Stage) -> str:
    g = find_prim_by_name(_tool_of(stage), TOOL_GRIPPER_PRIM_NAME)
    mesh = g if g.IsA(UsdGeom.Mesh) else find_prim_by_name(g, "Mesh")
    UsdPhysics.CollisionAPI.Apply(mesh)
    return "CollisionAPI applied to the gripper mesh"


def _mut_convex_hull(stage: Usd.Stage) -> str:
    UsdPhysics.MeshCollisionAPI(_part_geom_of(stage)).GetApproximationAttr().Set("convexHull")
    return "part collision approximation set to convexHull"


def _mut_strip_sdf_api(stage: Usd.Stage) -> str:
    _part_geom_of(stage).RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
    return "PhysxSDFMeshCollisionAPI removed from the part mesh"


def _mut_strip_collision_api(stage: Usd.Stage) -> str:
    _part_geom_of(stage).RemoveAPI(UsdPhysics.CollisionAPI)
    return "CollisionAPI removed from the part mesh"


# MEASURED, run RT-17 (2026-08-23): ``ComputeRelativeBound(tool, tool)`` does
# NOT include tool_link's own xform ops. Both mutations below used to be applied
# to tool_link and flipped NOTHING -- declared as an open question beforehand,
# answered by the run. Two consequences, and the second one is the real find:
#
#   * a size mutation has to act strictly BELOW tool_link to be seen at all.
#     FUEGETEIL sits between the measured mesh and tool_link, so a transform
#     there is unambiguously inside both bounds.
#   * the two size checks could NOT see a defect introduced at the link level.
#     A tool_link rotated 180 degrees would hang the tool the wrong way in the
#     world and ``tool_spans_flange_to_part_bottom`` -- the wrong-side test --
#     would still have said PASS. That hole is closed by the new check
#     ``tool_link_pose_matches_flange`` below, which the flip now targets.
def _part_xform_of(stage: Usd.Stage) -> Usd.Prim:
    return find_prim_by_name(_tool_of(stage), TOOL_PART_PRIM_NAME)


def _mut_scale_part_xform(stage: Usd.Stage) -> str:
    UsdGeom.Xformable(_part_xform_of(stage)).AddScaleOp().Set(Gf.Vec3f(1.01, 1.01, 1.01))
    return f"{TOOL_PART_PRIM_NAME} scaled by 1.01 (below tool_link, so inside the measured bound)"


def _mut_flip_tool_ref(stage: Usd.Stage) -> str:
    # MEASURED, run RT-21: rotating FUEGETEIL 180 deg about its OWN X flipped
    # nothing. An axis-aligned bounding box is blind to that rotation, because
    # the part's box is symmetric about the part's own origin -- the box before
    # and after is the same box. The mutation was blunt, not the check.
    #
    # A flip that the wrong-side test MUST see has to act at the FLANGE origin,
    # which is where the reference child sits. There the z range moves from
    # [0, +0.152] to [-0.152, 0], which is precisely the defect the check names.
    #
    # Adding an op on a referencing prim REPLACES the referenced prim's
    # xformOpOrder -- the hazard fix_fixture_asset.py exists for, and the reason
    # this child is deliberately op-free in the authoring path. For a mutation
    # that is acceptable and is itself informative: if this flips MORE than the
    # one declared check, the referenced assembly carries ops of its own that
    # the weld silently depends on, and that is worth knowing.
    ref = find_prim_by_name(_tool_of(stage), TOOL_REF_NAME)
    UsdGeom.Xformable(ref).AddRotateXOp().Set(180.0)
    return f"the referenced assembly ({TOOL_REF_NAME}) rotated 180 deg about the flange X axis"


def _mut_flip_tool_link(stage: Usd.Stage) -> str:
    UsdGeom.Xformable(_tool_of(stage)).AddRotateXOp().Set(180.0)
    return "tool_link rotated 180 deg about its own X"


def _mut_remove_weld_joint(stage: Usd.Stage) -> str:
    stage.RemovePrim(find_prim_by_name(stage, TOOL_JOINT_NAME).GetPath())
    return "tool_weld removed"


def _mut_retarget_weld_body0(stage: Usd.Stage) -> str:
    joint = UsdPhysics.FixedJoint(find_prim_by_name(stage, TOOL_JOINT_NAME))
    joint.GetBody0Rel().SetTargets([find_prim_by_name(stage, "base_link").GetPath()])
    return "tool_weld body0 retargeted from the flange to base_link"


def _mut_add_spare_joint(stage: Usd.Stage) -> str:
    UsdPhysics.FixedJoint.Define(stage, stage.GetDefaultPrim().GetPath().AppendChild("mutation_joint"))
    return "a second fixed joint added"


def _mut_add_spare_rigid_body(stage: Usd.Stage) -> str:
    path = stage.GetDefaultPrim().GetPath().AppendChild("mutation_body")
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
    return "a spare rigid body added"


def _mut_add_articulation_root(stage: Usd.Stage) -> str:
    path = stage.GetDefaultPrim().GetPath().AppendChild("mutation_root")
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
    return "a second articulation root added"


# The probes that paid for the api-shape claims. Both come from run RT-15,
# whose whole purpose was to MEASURE what these APIs return rather than assume.
_PROBE_FALLBACK = harness.Probe(
    run="RT-15",
    script="scripts/diagnose_tool_candidate.py",
    question="Q1",
    found="HasAuthoredValue=False with resolve source Usd.ResolveInfoSourceFallback; "
          "physics:mass and physics:centerOfMass report True as controls",
)
_PROBE_FLOAT32 = harness.Probe(
    run="RT-15",
    script="scripts/diagnose_tool_candidate.py",
    question="Q1 control",
    found="physics:mass reads back 0.8240000009536743 for an authored 0.824 -- "
          "USD stores it as float32, so the comparison must be relative",
)
_PROBE_EMPTY_ARC = harness.Probe(
    run="RT-15",
    script="scripts/diagnose_tool_candidate.py",
    question="Q2/Q3",
    found="a flattened layer reports exactly one composition dependency and its path is "
          "the empty string; the SHIPPED UR5e alone does the same, with no tool involved",
)

def verify(stage: Usd.Stage, before: dict, part_mass: float, gripper_mass: float, tol: float):
    after = describe(stage)
    tool = find_prim_by_name(stage, TOOL_LINK_NAME)
    joint_prim = find_prim_by_name(stage, TOOL_JOINT_NAME)
    part_mesh = find_prim_by_name(tool, TOOL_PART_PRIM_NAME) if tool else None
    gripper_mesh = find_prim_by_name(tool, TOOL_GRIPPER_PRIM_NAME) if tool else None
    part_geom = None
    if part_mesh is not None:
        part_geom = part_mesh if part_mesh.IsA(UsdGeom.Mesh) else find_prim_by_name(part_mesh, "Mesh")

    measured: dict = {}
    if tool is not None:
        # Sizes are measured RELATIVE TO tool_link, never in world space. The
        # link carries the flange's rest transform, which permutes the axes; a
        # world box compared against local nominals rejects a correct asset,
        # and that mistake already cost the proxy a round trip (its
        # PROBLEMS.md carries it as its own row).
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        whole = cache.ComputeRelativeBound(tool, tool).ComputeAlignedRange()
        if not whole.IsEmpty():
            measured["tool_local_z_range_m"] = [float(whole.GetMin()[2]), float(whole.GetMax()[2])]
            measured["tool_local_size_m"] = [float(whole.GetSize()[i]) for i in range(3)]
        if part_geom is not None:
            prng = cache.ComputeRelativeBound(part_geom, tool).ComputeAlignedRange()
            if not prng.IsEmpty():
                measured["part_local_size_m"] = [float(prng.GetSize()[i]) for i in range(3)]
                measured["part_local_z_range_m"] = [float(prng.GetMin()[2]), float(prng.GetMax()[2])]

        if tool.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(tool)
            measured["mass_kg"] = float(mass_api.GetMassAttr().Get() or 0.0)
            com = mass_api.GetCenterOfMassAttr().Get()
            measured["centre_of_mass_m"] = [float(v) for v in com] if com is not None else None
            # Read whether the tensor is AUTHORED, not what it reads back.
            # ``Get()`` on an unauthored attribute returns the schema fallback,
            # and UsdPhysics gives diagonalInertia a fallback of (0,0,0) -- the
            # sentinel that asks PhysX to compute the tensor. MEASURED, run
            # RT-15 on 2026-08-23: HasAuthoredValue=False while the resolve
            # info reads Usd.ResolveInfoSourceFallback, and the two attributes
            # this script does write report HasAuthoredValue=True as a control.
            inertia_attr = mass_api.GetDiagonalInertiaAttr()
            measured["diagonal_inertia_authored"] = bool(inertia_attr.HasAuthoredValue())
            inertia = inertia_attr.Get()
            measured["diagonal_inertia"] = [float(v) for v in inertia] if inertia is not None else None

    if part_geom is not None:
        measured["part_approximation"] = (
            str(UsdPhysics.MeshCollisionAPI(part_geom).GetApproximationAttr().Get())
            if part_geom.HasAPI(UsdPhysics.MeshCollisionAPI)
            else None
        )

    # The link's OWN placement, which no bound measurement sees (RT-17). The
    # authoring sets tool_link's local transform so that its WORLD transform
    # reproduces the flange's, spun by the weld yaw (zero by definition, D-078).
    # Recomputing that here is what makes a link-level defect visible at all.
    flange_prim = find_prim_by_name(stage, FLANGE_BODY)
    if tool is not None and flange_prim is not None:
        xf = UsdGeom.XformCache(Usd.TimeCode.Default())
        want = xf.GetLocalToWorldTransform(flange_prim)
        if TOOL_WELD_YAW_RAD != 0.0:
            spin = Gf.Matrix4d().SetRotate(
                Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(TOOL_WELD_YAW_RAD)))
            want = spin * want
        got = xf.GetLocalToWorldTransform(tool)
        measured["tool_pose_delta_max"] = max(
            abs(float(got[r][c]) - float(want[r][c])) for r in range(4) for c in range(4)
        )

    targets = []
    if joint_prim is not None and joint_prim.IsValid():
        joint = UsdPhysics.FixedJoint(joint_prim)
        targets = [str(t) for t in (joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets())]
    measured["joint_targets"] = targets

    def close(got: float | None, want: float) -> bool:
        """Relative comparison. USD ``physics:mass`` and ``physics:centerOfMass``
        are float attributes, so every value written comes back rounded to
        float32; an absolute tolerance here rejected a correct asset once
        already in the proxy repo."""
        return got is not None and math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-12)

    total_mass = float(part_mass) + float(gripper_mass)
    want_com = tool_com_flange(part_mass, gripper_mass)
    got_com = measured.get("centre_of_mass_m") or [None, None, None]
    z_range = measured.get("tool_local_z_range_m") or [0.0, 0.0]
    part_size = measured.get("part_local_size_m") or [0.0, 0.0, 0.0]

    # Collision on the gripper would be a defect, not an omission: the adapter
    # overlaps the part by 24.5 mm, so a collider there fights the part inside
    # the same body.
    gripper_colliders = []
    if gripper_mesh is not None:
        gripper_colliders = [
            str(p.GetPath()) for p in Usd.PrimRange(gripper_mesh) if p.HasAPI(UsdPhysics.CollisionAPI)
        ]
    measured["gripper_collider_prims"] = gripper_colliders

    # Every EXPRESSION below is unchanged from the plain dict this replaced --
    # scripts/selftest_checks.py --diff-predicates proves that against git,
    # offline. What is new is the bookkeeping each check now has to carry: what
    # KIND of claim it makes, what it must report BEFORE the weld, which probe
    # run paid for it, and the one mutation that must break it (D-080).
    suite = harness.Suite("author_tool", str(stage.GetRootLayer().identifier))

    suite.add("units_are_metres", abs(after["meters_per_unit"] - 1.0) < 1e-9,
              kind=harness.PRESERVATION, expect_before=harness.PASS,
              why="The weld must not touch the stage unit declaration; a mm robot would be 1000x.",
              mutation=harness.Mutation("stage_units_mm", _mut_stage_units_mm, ("units_are_metres",)))
    suite.add("up_axis_is_z", after["up_axis"] == "Z",
              kind=harness.PRESERVATION, expect_before=harness.PASS,
              why="Same: the up axis is the shipped robot's and the weld must leave it alone.",
              mutation=harness.Mutation("up_axis_y", _mut_up_axis_y, ("up_axis_is_z",)))
    suite.add("tool_link_exists", bool(tool is not None and tool.IsValid()),
              kind=harness.INVARIANCE,
              why="The link the whole weld is about.",
              mutation=harness.Unprovable(
                  "every other check in this suite reads THROUGH this prim, so any mutation that "
                  "removes it flips the entire table and proves nothing about this check in "
                  "particular. Its failure is covered by the 18 checks that cannot be evaluated "
                  "without it."))
    suite.add("tool_link_is_rigid_body", bool(tool is not None and tool.HasAPI(UsdPhysics.RigidBodyAPI)),
              kind=harness.INVARIANCE,
              why="A welded link that is not a rigid body is geometry, not a body PhysX simulates.",
              mutation=harness.Mutation("strip_rigid_body_api", _mut_strip_rigid_body_api,
                                        ("tool_link_is_rigid_body", "one_new_rigid_body")))
    suite.add("tool_mass_correct", close(measured.get("mass_kg"), total_mass),
              kind=harness.API_SHAPE, probe=_PROBE_FLOAT32,
              why="The sum of the two weighings, read back through a float32 attribute (D-079).",
              mutation=harness.Mutation("clear_mass", _mut_clear_mass, ("tool_mass_correct",)))
    suite.add("tool_com_correct", all(close(got_com[i], want_com[i]) for i in range(3)),
              kind=harness.API_SHAPE, probe=_PROBE_FLOAT32,
              why="Mass-weighted from the two CAD centroids (D-079); same float32 read-back.",
              mutation=harness.Mutation("shift_centre_of_mass", _mut_shift_centre_of_mass,
                                        ("tool_com_correct",)))
    # Named, not forgotten: PhysX derives the tensor from the collision
    # geometry. The check asserts the omission is real, so the PENDING
    # entry cannot quietly stop being true.
    suite.add("diagonal_inertia_left_to_physx", measured.get("diagonal_inertia_authored") is False,
              kind=harness.API_SHAPE, probe=_PROBE_FALLBACK,
              why="D-079 leaves the tensor to PhysX. This is the check that was WRONG in RT-13: "
                  "it asserted Get() is None, which USD never returns for an attribute that has "
                  "a schema fallback.",
              mutation=harness.Mutation("author_diagonal_inertia", _mut_author_diagonal_inertia,
                                        ("diagonal_inertia_left_to_physx",)))
    suite.add("part_mesh_found", part_geom is not None,
              kind=harness.INVARIANCE,
              why="The mesh the collider goes on.",
              mutation=harness.Unprovable(
                  "same cascade as tool_link_exists: the five part checks all resolve through this "
                  "prim, so removing it says nothing about this check alone."))
    suite.add("part_has_collision", bool(part_geom is not None and part_geom.HasAPI(UsdPhysics.CollisionAPI)),
              kind=harness.INVARIANCE,
              why="Without a collider the part passes straight through the fixture.",
              mutation=harness.Mutation("strip_collision_api", _mut_strip_collision_api,
                                        ("part_has_collision",)))
    suite.add("part_approximation_is_sdf", measured.get("part_approximation") == "sdf",
              kind=harness.ARITHMETIC,
              why="D-077: convex hull is ruled out by the lugs, so the approximation must read sdf.",
              mutation=harness.Mutation("convex_hull", _mut_convex_hull, ("part_approximation_is_sdf",)))
    suite.add("part_has_sdf_api", bool(part_geom is not None and part_geom.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)),
              kind=harness.INVARIANCE,
              why="The token alone is not the collider; the PhysX schema carries the resolution.",
              mutation=harness.Mutation("strip_sdf_api", _mut_strip_sdf_api, ("part_has_sdf_api",)))
    suite.add("gripper_has_no_collision", gripper_mesh is not None and not gripper_colliders,
              kind=harness.INVARIANCE,
              why="D-077: the adapter overlaps the part by 24.5 mm, so a collider there would fight "
                  "the part inside the same body.",
              mutation=harness.Mutation("collide_the_gripper", _mut_collide_the_gripper,
                                        ("gripper_has_no_collision",)))
    suite.add("part_size_matches_cad", all(abs(part_size[i] - PART_BBOX_M[i]) < tol for i in range(3)),
              kind=harness.ARITHMETIC,
              why="The imported part must still measure what the CAD says (D-073/D-075).",
              mutation=harness.Mutation("scale_part_xform", _mut_scale_part_xform,
                                        ("part_size_matches_cad", "tool_spans_flange_to_part_bottom")))
    # The wrong-side test. Welded upside down the tool would span
    # -0.152 .. 0 and every size check above would still call it correct.
    suite.add("tool_spans_flange_to_part_bottom", (
                  abs(z_range[0] - TOOL_SPAN_LOCAL_Z_M[0]) < tol and abs(z_range[1] - TOOL_SPAN_LOCAL_Z_M[1]) < tol
              ),
              kind=harness.ARITHMETIC,
              why="The wrong-side test: welded upside down, every edge length still measures right. "
                  "REACH, measured in RT-21 rather than assumed: this sees a flip about the FLANGE "
                  "origin, which is the weld defect it exists for. It is BLIND to the part being "
                  "flipped about its OWN centre -- an axis-aligned box is the same box either way. "
                  "That case would put the lugs and the chamfer on the wrong side and no check here "
                  "would notice; it needs a geometric feature, not a bound. Named as "
                  "PART_FLIPPED_IN_PLACE in TOOL_CHAIN_PENDING.",
              mutation=harness.Mutation("flip_tool_ref", _mut_flip_tool_ref,
                                        ("tool_spans_flange_to_part_bottom",)))
    # Added 2026-08-23 because the counter-proof PROVED a hole, not because a
    # run failed: RT-17 showed that neither size check reacts to a transform on
    # tool_link itself, so a tool welded on the wrong way round at the LINK
    # level passed all twenty checks. This is the check that sees it, and
    # flip_tool_link -- the mutation that flipped nothing before -- is now its
    # counter-proof.
    suite.add("tool_link_pose_matches_flange",
              measured.get("tool_pose_delta_max") is not None and measured["tool_pose_delta_max"] < tol,
              kind=harness.ARITHMETIC,
              why="tool_link must reproduce the flange's world transform, spun by the weld yaw "
                  "(D-078, zero today). No bound measurement can see this: RT-17 measured that "
                  "ComputeRelativeBound(tool, tool) ignores the link's own xform ops.",
              mutation=harness.Mutation("flip_tool_link", _mut_flip_tool_link,
                                        ("tool_link_pose_matches_flange",)))
    suite.add("weld_joint_exists", bool(joint_prim is not None and joint_prim.IsValid()),
              kind=harness.INVARIANCE,
              why="D-018/D-019: the joint must exist at AUTHORING time, not at runtime.",
              mutation=harness.Mutation("remove_weld_joint", _mut_remove_weld_joint,
                                        ("weld_joint_exists", "weld_targets_flange_and_tool", "one_new_joint")))
    suite.add("weld_targets_flange_and_tool", (
                  len(targets) == 2
                  and any(FLANGE_BODY in t for t in targets)
                  and any(TOOL_LINK_NAME in t for t in targets)
              ),
              kind=harness.INVARIANCE,
              why="A joint that exists but joins the wrong two bodies is the D-018 failure again.",
              mutation=harness.Mutation("retarget_weld_body0", _mut_retarget_weld_body0,
                                        ("weld_targets_flange_and_tool",)))
    suite.add("one_new_rigid_body", after["rigid_body_count"] == before["rigid_body_count"] + 1,
              kind=harness.INVARIANCE,
              why="D-076: the tool is ONE link. Two would be a second body nobody asked for.",
              mutation=harness.Mutation("add_spare_rigid_body", _mut_add_spare_rigid_body,
                                        ("one_new_rigid_body",)))
    suite.add("one_new_joint", after["joint_count"] == before["joint_count"] + 1,
              kind=harness.INVARIANCE,
              why="Exactly the weld, and nothing else.",
              mutation=harness.Mutation("add_spare_joint", _mut_add_spare_joint, ("one_new_joint",)))
    suite.add("articulation_roots_unchanged", after["articulation_roots"] == before["articulation_roots"],
              kind=harness.PRESERVATION, expect_before=harness.PASS,
              why="D-018/D-019: the Robot Assembler STRIPPED the ArticulationRootAPI. This asserts "
                  "the weld does not repeat that, so it must hold before AND after -- that is the "
                  "claim, not a weakness in it.",
              mutation=harness.Mutation("add_articulation_root", _mut_add_articulation_root,
                                        ("articulation_roots_unchanged",)))
    suite.add("self_contained", not after["external_arcs"],
              kind=harness.API_SHAPE, probe=_PROBE_EMPTY_ARC,
              why="An arc to a file the training machine does not have spawns an INVISIBLE asset "
                  "(2026-07-26). The other check RT-13 got wrong: it counted the empty path too.",
              mutation=harness.Mutation("add_external_reference", _mut_add_external_reference,
                                        ("self_contained",)))
    return {**after, "measured": measured}, suite



def counter_proof_mode(out_path: pathlib.Path) -> int:
    """Break the FINISHED asset one way per declared mutation (D-080).

    Nothing here writes. Every mutation is applied to an ANONYMOUS layer, which
    has no file behind it, so an accidental Save() is impossible -- and the
    subject's size and modification time are compared before and after anyway,
    because this is the one mode that opens the production asset in order to
    damage it.

    ``before`` and the two masses come out of the sidecar rather than being
    restated on the command line: the sidecar is the record of what the
    authoring run actually did, and reading it back cross-checks it for free.
    """
    if not out_path.is_file():
        print(f"[author_tool] no asset at {out_path}; author it first.")
        return 1
    sidecar = out_path.with_suffix(".author.json")
    if not sidecar.is_file():
        print(f"[author_tool] no sidecar at {sidecar}; the counter-proof needs the before-state it records.")
        return 1
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    before = record.get("before")
    part_mass = record.get("part_mass_kg")
    gripper_mass = record.get("gripper_mass_kg")
    if before is None or part_mass is None or gripper_mass is None:
        print(f"[author_tool] {sidecar.name} lacks before/part_mass_kg/gripper_mass_kg; cannot counter-prove.")
        return 1

    stat_before = out_path.stat()
    print(f"[author_tool] counter-proof subject: {out_path}")
    print(f"[author_tool] before-state and masses read from {sidecar.name}: "
          f"{part_mass} + {gripper_mass} kg")

    baseline_stage = Usd.Stage.Open(str(out_path))
    _, baseline = verify(baseline_stage, before, part_mass, gripper_mass, args_cli.tol)
    baseline.print_lines()
    print(f"[author_tool] baseline verdict: {baseline.verdict()}")
    if baseline.verdict() != harness.PASS:
        print("[author_tool] the subject does not pass its own checks; a counter-proof over a "
              "failing baseline would compare nothing. Re-author first.")
        return 1
    del baseline_stage

    mutated: dict = {}
    for m in baseline.mutations():
        layer = Sdf.Layer.OpenAsAnonymous(str(out_path))
        stage = Usd.Stage.Open(layer)
        note = m.apply(stage)
        _, suite = verify(stage, before, part_mass, gripper_mass, args_cli.tol)
        flipped = sorted(n for n in baseline.names()
                         if baseline.result(n) == harness.PASS and suite.result(n) != harness.PASS)
        print(f"[author_tool] mutation {m.name}: {note}")
        print(f"[author_tool] mutation {m.name}: declared {sorted(m.flips)} -- flipped {flipped}")
        mutated[m.name] = suite
        del stage, layer

    findings = harness.counter_proof(baseline, mutated)
    for f in findings:
        print(f"[author_tool] {f}")

    stat_after = out_path.stat()
    untouched = (stat_before.st_size == stat_after.st_size
                 and stat_before.st_mtime == stat_after.st_mtime)
    print(f"[author_tool] subject untouched: {'PASS' if untouched else 'FAIL'} "
          f"({stat_before.st_size} bytes)")
    for n, reason in baseline.unprovable().items():
        print(f"[author_tool] NOTE {n}: no counter-proof -- {reason}")

    ok = not findings and untouched
    print(f"[author_tool] counter-proof over {len(mutated)} mutations: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    out_for_mode = (pathlib.Path(args_cli.out).resolve() if args_cli.out
                    else pathlib.Path(default_tool_robot_usd_path()))
    if args_cli.counter_proof:
        return counter_proof_mode(out_for_mode)

    # D-079 unchanged, only moved: authoring without both weighings is refused.
    if args_cli.part_mass is None or args_cli.gripper_mass is None:
        raise SystemExit("[author_tool] --part-mass and --gripper-mass are required for authoring. "
                         "They are weighings, not derivations (D-079).")

    src = args_cli.src or UR5E_HOME_CFG.spawn.usd_path
    tool_usd = pathlib.Path(args_cli.tool_usd).resolve() if args_cli.tool_usd else pathlib.Path(resolve_tool_usd_path())
    out_path = pathlib.Path(args_cli.out).resolve() if args_cli.out else pathlib.Path(default_tool_robot_usd_path())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaw_rad = math.radians(args_cli.tool_yaw_deg) if args_cli.tool_yaw_deg is not None else TOOL_WELD_YAW_RAD

    print(f"[author_tool] source robot: {src}")
    print(f"[author_tool] tool assembly: {tool_usd}")
    print(f"[author_tool] output: {out_path}")
    print(f"[author_tool] masses: part {args_cli.part_mass:.5f} kg + gripper {args_cli.gripper_mass:.5f} kg "
          f"= {args_cli.part_mass + args_cli.gripper_mass:.5f} kg")
    com = tool_com_flange(args_cli.part_mass, args_cli.gripper_mass)
    print(f"[author_tool] centre of mass (flange frame): "
          f"({com[0]:+.6f}, {com[1]:+.6f}, {com[2]:+.6f}) m")
    print(f"[author_tool] weld yaw: {math.degrees(yaw_rad):+.4f} deg "
          f"({'zero by definition' if yaw_rad == 0.0 else 'measured, supplied'})")

    src_stage = Usd.Stage.Open(str(src))
    if src_stage is None:
        raise RuntimeError(f"Could not open source stage: {src}")
    before = describe(src_stage)
    print("[author_tool] BEFORE:")
    print(json.dumps(before, indent=2))

    tool_stage = Usd.Stage.Open(str(tool_usd))
    if tool_stage is None:
        raise RuntimeError(f"Could not open the tool assembly: {tool_usd}")
    tool_mpu = float(UsdGeom.GetStageMetersPerUnit(tool_stage))
    robot_mpu = float(before["meters_per_unit"])
    print(f"[author_tool] metersPerUnit -- robot {robot_mpu}, tool {tool_mpu}")
    # USD does NOT rescale a reference across a units mismatch: the referencing
    # stage's metersPerUnit wins and the geometry keeps its numbers. A mm tool
    # referenced into a metre robot would be 1000x too large and every pose
    # check downstream would be measuring a different asset than intended.
    if abs(tool_mpu - robot_mpu) > 1e-12:
        raise RuntimeError(
            f"Unit mismatch: robot stage {robot_mpu}, tool stage {tool_mpu}. A reference does not "
            "rescale. Run scripts/fix_stage_units.py on the tool asset first (D-016)."
        )
    del tool_stage

    # Leg C of D-080: run the SAME checks on the UNMODIFIED robot and compare
    # against what each one declared it must report there. A check that already
    # passes here is certifying the input, not the weld -- which is exactly how
    # the RT-13 inertia assertion looked. REPORT, never a gate: a new code path
    # that can abort a working authoring run would be a new way to burn the
    # round trip this whole exercise exists to save.
    before_table = "UNAVAILABLE"
    try:
        _, before_suite = verify(src_stage, before, args_cli.part_mass, args_cli.gripper_mass, args_cli.tol)
        # Both arguments are the same suite on purpose: the DECLARATIONS live on
        # the check, so the builder run against the source stage carries both the
        # expectation and the result it actually produced there.
        mismatches = harness.compare_before(before_suite, before_suite)
        before_table = {n: before_suite.result(n) for n in before_suite.names()}
    except Exception as exc:  # noqa: BLE001 -- a report must not stop the work
        mismatches = []
        before_table = f"UNAVAILABLE ({type(exc).__name__}: {exc})"
    for f in mismatches:
        print(f"[author_tool] before-state {f}")

    flange = find_prim_by_name(src_stage, FLANGE_BODY)
    if flange is None:
        raise RuntimeError(f"No prim named '{FLANGE_BODY}' in {src}; body list: {before['rigid_bodies']}")
    parent = src_stage.GetDefaultPrim()
    if parent is None:
        raise RuntimeError(f"Source stage has no default prim; cannot place {TOOL_LINK_NAME}.")
    if parent.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            f"Default prim {parent.GetPath()} is itself a rigid body. Nesting {TOOL_LINK_NAME} under it "
            "would be invalid; place it elsewhere and re-run."
        )
    print(f"[author_tool] flange: {flange.GetPath()}   parent: {parent.GetPath()}")

    if args_cli.dry_run:
        print(f"[author_tool] --dry-run: would write {TOOL_LINK_NAME} + {TOOL_JOINT_NAME} under "
              f"{parent.GetPath()}, reference {tool_usd.name}, and export to {out_path}. Nothing written.")
        return 0

    backup = None
    if out_path.is_file():
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.copy2(out_path, backup)
        print(f"[author_tool] previous output backed up: {backup}")

    # Flatten the shipped asset into an in-memory layer and author on that: the
    # shipped UR5e references other layers (possibly on Nucleus) and the result
    # has to stand alone on the training machine.
    work = Usd.Stage.Open(src_stage.Flatten())
    del src_stage
    work_flange = find_prim_by_name(work, FLANGE_BODY)
    work_parent = work.GetDefaultPrim()
    if work_flange is None or work_parent is None:
        raise RuntimeError("Flattening lost the flange or the default prim; the source asset is not as expected.")

    authored = author_tool(
        work, work_flange, work_parent, tool_usd,
        args_cli.part_mass, args_cli.gripper_mass, yaw_rad, args_cli.sdf_resolution,
    )
    print("[author_tool] authored:")
    print(json.dumps(authored, indent=2))

    # Second flatten, at export: it absorbs the reference to the tool assembly,
    # so the written robot USD does not depend on a second file at run time.
    candidate = out_path.with_name(out_path.stem + ".candidate" + out_path.suffix)
    work.Flatten().Export(str(candidate))
    print(f"[author_tool] candidate exported: {candidate}")
    del work

    check_stage = Usd.Stage.Open(str(candidate))
    if check_stage is None:
        raise RuntimeError(f"Could not re-open the candidate for verification: {candidate}")
    after, suite = verify(check_stage, before, args_cli.part_mass, args_cli.gripper_mass, args_cli.tol)
    audit_findings = harness.audit(suite, REPO_ROOT)
    print("[author_tool] AFTER (fresh stage, candidate file):")
    print(json.dumps(after, indent=2))
    suite.print_lines()
    for f in audit_findings:
        print(f"[author_tool] claim audit {f}")

    report = {
        "marker": SCRIPT_MARKER,
        "source": str(src),
        "tool_asset": str(tool_usd),
        "output": str(out_path),
        "candidate": str(candidate),
        "backup": str(backup) if backup else None,
        "part_mass_kg": args_cli.part_mass,
        "gripper_mass_kg": args_cli.gripper_mass,
        "authored": authored,
        "before": before,
        "after": after,
        "checks": suite.as_dict(),
        "verdict": suite.verdict(),
        **harness.sidecar_block(suite, before_table, audit_findings),
    }
    report_path = out_path.with_suffix(".author.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[author_tool] wrote {report_path}")

    if suite.verdict() != harness.PASS:
        failed = list(suite.failed())
        print(f"[author_tool] FAIL on {failed}; candidate kept for inspection, output not replaced.")
        return suite.exit_code()

    del check_stage
    try:
        shutil.copy2(candidate, out_path)
    except OSError as exc:
        print(f"[author_tool] could not write the output: {exc}")
        print(f"[author_tool] copy it manually: {candidate} -> {out_path}")
        return 1
    print(f"[author_tool] PASS; {out_path.name} written")
    print("[author_tool] counter-proof (D-080) has NOT run in this mode; it is a separate command: "
          "scripts/author_tool_ur5e.py --counter-proof")
    print("[author_tool] next: scripts/list_usd_prims.py on the output, then zero_agent --num_envs 4; "
          f"the startup report must show {TOOL_LINK_NAME} among the bodies and the leading tool point "
          "0.150 m above the block top face.")


if __name__ == "__main__":
    # D-081, and the shutdown that used to swallow it.
    #
    # MEASURED, run RT-22: neither of the two lines that used to stand here --
    # one inside `except SystemExit` around close(), one after the finally --
    # ever printed, on a run that PASSED. simulation_app.close() does not
    # return, so `raise SystemExit(n)` after it was dead code and every failing
    # run reported `[rt_log] exit code: 0` to the line /rt-check reads.
    #
    # Ruled out first, in order: RT-20 showed the wrapper reports a non-zero
    # code correctly, so it was not the messenger.
    _code = 1
    try:
        _code = main() or 0
    except SystemExit as _exc:          # argparse and the D-079 refusal
        _code = int(_exc.code or 0)
    except BaseException:
        traceback.print_exc()
        _code = 1
    exit_with(simulation_app, _code, tag="author_tool")
