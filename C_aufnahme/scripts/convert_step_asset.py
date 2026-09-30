# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Pilot STEP -> USD conversion for ONE asset (route (b), pipeline research
2026-08-21: docs/reference/literature_check_step_usd_pipeline_2026-08-21.md).

Two stages:

1. STEP -> STL via headless FreeCAD (``freecadcmd``). FreeCAD works in mm, so
   the STL coordinates are numerically mm. STL itself carries no unit. Skipped
   when ``--stl`` provides a ready mesh.
2. STL -> USD via Isaac Lab's ``MeshConverter`` with ``scale=(0.001,)*3``
   (mm -> m), ``make_instanceable=False`` (the default True broke collision
   authoring in the proxy-era import chain), Z-up and ``metersPerUnit=1.0``
   authored by the converter itself. Optional SDF collision
   (``SDFMeshPropertiesCfg``) authored by the converter, no separate step.

After conversion the output stage is re-opened and checked; results go to
stdout as PASS/FAIL lines and to a ``<out>.convert.json`` sidecar. This is a
PILOT tool: it measures and reports; the per-asset contracts live in the
author_* scripts that build on it.

UNVERIFIED until run: the dev laptop has no Isaac and no FreeCAD. Whether
FreeCAD is installed on the training machine is UNKNOWN -- stage 1 checks and
says what is missing. On the training machine:

    conda activate env_isaaclab
    python scripts/convert_step_asset.py --step CAD/Aufnahme_real_v1.stp \\
        --nominal-mm 150.0 189.1 61.0 --collision sdf
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Bumped whenever this file changes; first line of output identifies the copy
# in the two-machine workflow.
SCRIPT_MARKER = "convert_step_asset-2026-08-21a"

# The FreeCAD tessellation macro. Written to a temp file and executed by
# freecadcmd. Kept as a template string so stage 1 needs no FreeCAD on the
# machine that WRITES this script. Pattern (Part.Shape().read + MeshPart)
# is hint-level per the research doc -- first real run verifies it.
FREECAD_MACRO = """\
import Mesh
import MeshPart
import Part

shape = Part.Shape()
shape.read(r"{step_path}")
bb = shape.BoundBox
print("[freecad] shape bbox (mm): "
      f"x {{bb.XLength:.3f}} y {{bb.YLength:.3f}} z {{bb.ZLength:.3f}}")
print("[freecad] shape bbox z-range (mm): "
      f"[{{bb.ZMin:.3f}}, {{bb.ZMax:.3f}}]")
mesh = MeshPart.meshFromShape(
    Shape=shape,
    LinearDeflection={linear_deflection_mm},
    AngularDeflection=0.523599,  # 30 deg
    Relative=False,
)
print(f"[freecad] mesh: {{mesh.CountFacets}} facets, {{mesh.CountPoints}} points")
Mesh.export([mesh], r"{stl_path}")
print("[freecad] wrote " + r"{stl_path}")
"""

FREECAD_CANDIDATES = ("freecadcmd", "FreeCADCmd", "freecad.cmd")


def step_to_stl(step_path: pathlib.Path, stl_path: pathlib.Path,
                linear_deflection_mm: float) -> None:
    """Stage 1: tessellate the STEP with headless FreeCAD."""
    exe = None
    for name in FREECAD_CANDIDATES:
        exe = shutil.which(name)
        if exe:
            break
    if exe is None:
        sys.exit(
            "[convert_step_asset] FAIL: no FreeCAD CLI found "
            f"(tried {', '.join(FREECAD_CANDIDATES)}).\n"
            "  Install FreeCAD (e.g. 'sudo apt install freecad' or the "
            "AppImage), or convert the STEP to STL elsewhere and re-run "
            "with --stl <file>. The STL must be in millimetres."
        )
    macro = FREECAD_MACRO.format(
        step_path=str(step_path),
        stl_path=str(stl_path),
        linear_deflection_mm=linear_deflection_mm,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(macro)
        macro_path = fh.name
    print(f"[convert_step_asset] stage 1: {exe} (LinearDeflection "
          f"{linear_deflection_mm} mm)")
    result = subprocess.run([exe, macro_path], capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0 or not stl_path.exists():
        sys.exit("[convert_step_asset] FAIL: FreeCAD tessellation did not "
                 "produce the STL (see output above).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot STEP->USD conversion (research route (b)).")
    parser.add_argument("--step", type=str, default=None,
                        help="Input STEP file (mm). Stage 1 tessellates it "
                             "with headless FreeCAD.")
    parser.add_argument("--stl", type=str, default=None,
                        help="Ready STL in millimetres; skips stage 1.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output USD path. Default: <input stem>_pilot.usd "
                             "next to the input. NEVER a resolver name -- the "
                             "pilot must not shadow a real asset.")
    parser.add_argument("--nominal-mm", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Expected outer bbox in mm; compared against the "
                             "converted USD (sorted extents, +-0.5 mm).")
    parser.add_argument("--collision", choices=("none", "sdf"), default="none",
                        help="Author collision in the converter. 'sdf' uses "
                             "SDFMeshPropertiesCfg (D-062).")
    parser.add_argument("--sdf-resolution", type=int, default=1024)
    parser.add_argument("--linear-deflection-mm", type=float, default=0.05,
                        help="FreeCAD tessellation chord tolerance. 0.05 mm "
                             "keeps faceting well under the 0.2 mm clearance.")
    parser.add_argument("--force", action="store_true",
                        help="Allow replacing an existing output file.")
    from isaaclab.app import AppLauncher  # noqa: PLC0415
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True

    print(f"[convert_step_asset] marker: {SCRIPT_MARKER}")

    if (args_cli.step is None) == (args_cli.stl is None):
        sys.exit("[convert_step_asset] FAIL: give exactly one of --step / --stl.")

    if args_cli.stl is not None:
        stl_path = pathlib.Path(args_cli.stl).resolve()
        if not stl_path.exists():
            sys.exit(f"[convert_step_asset] FAIL: {stl_path} does not exist.")
        in_path = stl_path
    else:
        step_path = pathlib.Path(args_cli.step).resolve()
        if not step_path.exists():
            sys.exit(f"[convert_step_asset] FAIL: {step_path} does not exist.")
        stl_path = step_path.with_suffix(".stl")
        step_to_stl(step_path, stl_path, args_cli.linear_deflection_mm)
        in_path = step_path

    out_path = (pathlib.Path(args_cli.out).resolve() if args_cli.out
                else in_path.with_name(in_path.stem + "_pilot.usd"))
    if out_path.exists() and not args_cli.force:
        sys.exit(f"[convert_step_asset] FAIL: {out_path} exists; use --force.")

    # Isaac only from here on: stage 1 fails fast without booting the app.
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app  # noqa: F841

    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
    from isaaclab.sim import schemas_cfg  # noqa: E402

    # --- stage 2: MeshConverter -------------------------------------------
    collision_props = None
    mesh_collision_props = None
    if args_cli.collision == "sdf":
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
        mesh_collision_props = schemas_cfg.SDFMeshPropertiesCfg(
            sdf_resolution=args_cli.sdf_resolution)

    converter_cfg = MeshConverterCfg(
        asset_path=str(stl_path),
        usd_dir=str(out_path.parent),
        usd_file_name=out_path.name,
        force_usd_conversion=True,
        make_instanceable=False,     # default True; broke collision authoring
                                     # in the proxy import chain
        scale=(0.001, 0.001, 0.001),  # STL values are mm, stage is metres
        collision_props=collision_props,
        mesh_collision_props=mesh_collision_props,
    )
    print(f"[convert_step_asset] stage 2: MeshConverter {stl_path.name} "
          f"-> {out_path.name} (collision={args_cli.collision})")
    converter = MeshConverter(converter_cfg)
    usd_path = pathlib.Path(converter.usd_path)

    # --- checks on the re-opened output -----------------------------------
    stage = Usd.Stage.Open(str(usd_path))
    checks: dict[str, bool] = {}
    report: dict = {"marker": SCRIPT_MARKER, "input": str(in_path),
                    "stl": str(stl_path), "output": str(usd_path),
                    "collision": args_cli.collision}

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    checks["meters_per_unit_is_1"] = abs(mpu - 1.0) < 1e-9
    report["meters_per_unit"] = mpu

    up = UsdGeom.GetStageUpAxis(stage)
    checks["up_axis_is_z"] = up == UsdGeom.Tokens.z
    report["up_axis"] = str(up)

    instanceable = [str(p.GetPath()) for p in stage.Traverse()
                    if p.IsInstanceable()]
    checks["no_instanceable_prims"] = not instanceable
    report["instanceable_prims"] = instanceable

    external = [str(ref.assetPath) for ref in
                stage.GetRootLayer().GetCompositionAssetDependencies()]
    checks["self_contained"] = not external
    report["external_dependencies"] = external

    default_prim = stage.GetDefaultPrim()
    report["default_prim"] = str(default_prim.GetPath()) if default_prim else None
    if default_prim:
        ops = UsdGeom.Xformable(default_prim).GetOrderedXformOps()
        report["default_prim_xform_ops"] = [op.GetOpName() for op in ops]

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_])
    bound = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    size = [bound.GetMax()[i] - bound.GetMin()[i] for i in range(3)]
    report["bbox_min_m"] = [bound.GetMin()[i] for i in range(3)]
    report["bbox_max_m"] = [bound.GetMax()[i] for i in range(3)]
    report["bbox_size_m"] = size
    report["bbox_z_range_m"] = [bound.GetMin()[2], bound.GetMax()[2]]
    checks["not_mm_scaled"] = max(size) < 10.0  # a 1000x asset is >= 61 m here

    if args_cli.nominal_mm:
        nominal_m = sorted(v / 1000.0 for v in args_cli.nominal_mm)
        got = sorted(size)
        checks["bbox_matches_nominal"] = all(
            abs(g - n) <= 0.0005 for g, n in zip(got, nominal_m))
        report["nominal_sorted_m"] = nominal_m
        report["bbox_sorted_m"] = got

    if args_cli.collision == "sdf":
        mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        sdf_prims = [str(p.GetPath()) for p in mesh_prims
                     if p.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)]
        approx = []
        for p in mesh_prims:
            api = UsdPhysics.MeshCollisionAPI(p)
            if api:
                attr = api.GetApproximationAttr()
                if attr:
                    approx.append((str(p.GetPath()), str(attr.Get())))
        checks["sdf_api_present"] = bool(sdf_prims)
        report["sdf_prims"] = sdf_prims
        report["mesh_collision_approximations"] = approx
        report["sdf_resolution_requested"] = args_cli.sdf_resolution

    report["mesh_prim_count"] = sum(1 for p in stage.Traverse()
                                    if p.IsA(UsdGeom.Mesh))

    # --- verdict -----------------------------------------------------------
    for name, ok in checks.items():
        print(f"[convert_step_asset] {'PASS' if ok else 'FAIL'}: {name}")
    print(f"[convert_step_asset] bbox (m): "
          f"{size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}, "
          f"z-range [{report['bbox_z_range_m'][0]:.4f}, "
          f"{report['bbox_z_range_m'][1]:.4f}]")
    verdict = all(checks.values())
    report["checks"] = checks
    report["verdict"] = "PASS" if verdict else "FAIL"

    sidecar = usd_path.with_suffix(usd_path.suffix + ".convert.json")
    sidecar.write_text(json.dumps(report, indent=2))
    print(f"[convert_step_asset] sidecar: {sidecar}")
    print(f"[convert_step_asset] verdict: {report['verdict']}")
    if not verdict:
        sys.exit(1)


if __name__ == "__main__":
    main()
