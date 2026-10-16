# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure WHY two checks of author_tool_ur5e.py fail on its candidate."""

from __future__ import annotations

import argparse
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose the two failing checks of author_tool_ur5e.py.")
parser.add_argument("--usd", type=str, default=None, help="Candidate robot USD. Default: the standard candidate path.")
parser.add_argument("--link", type=str, default="tool_link", help="Name of the welded link.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402

from insertion.tasks.direct.insertion.insertion_tasks_cfg import (  # noqa: E402
    default_tool_robot_usd_path,
)
from insertion.tasks.direct.insertion.ur5e_cfg import UR5E_HOME_CFG  # noqa: E402


def find_prim_by_name(stage: Usd.Stage, name: str) -> Usd.Prim | None:
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def report_arcs(stage: Usd.Stage, tag: str) -> None:
    """Print every composition dependency with its REPR, so an empty string is visible as '' instead of vanishing, and then say which prim produced it."""
    root = stage.GetRootLayer()
    deps = list(root.GetCompositionAssetDependencies())
    print(f"[diag] {tag}: {len(deps)} composition asset dependencies")
    for i, dep in enumerate(deps):
        print(f"[diag] {tag}:   [{i}] repr={dep!r}  empty={dep == ''}")
    print(f"[diag] {tag}: subLayerPaths = {[str(p) for p in root.subLayerPaths]!r}")

    for prim in stage.Traverse():
        refs = prim.GetMetadata("references")
        if refs is not None:
            for item in list(refs.prependedItems) + list(refs.appendedItems) + list(refs.explicitItems):
                print(f"[diag] {tag}:   REFERENCE on {prim.GetPath()}: "
                      f"assetPath={item.assetPath!r} primPath={str(item.primPath)!r}")
        pay = prim.GetMetadata("payload")
        if pay is not None:
            for item in list(pay.prependedItems) + list(pay.appendedItems) + list(pay.explicitItems):
                print(f"[diag] {tag}:   PAYLOAD on {prim.GetPath()}: "
                      f"assetPath={item.assetPath!r} primPath={str(item.primPath)!r}")


def main() -> None:
    if args_cli.usd:
        candidate = pathlib.Path(args_cli.usd).resolve()
    else:
        out = pathlib.Path(default_tool_robot_usd_path())
        candidate = out.with_name(out.stem + ".candidate" + out.suffix)
    print(f"[diag] candidate: {candidate}")
    if not candidate.is_file():
        raise RuntimeError(f"No candidate at {candidate}. Run author_tool_ur5e.py first (without --dry-run).")

    stage = Usd.Stage.Open(str(candidate))
    if stage is None:
        raise RuntimeError(f"Could not open {candidate}")

    link = find_prim_by_name(stage, args_cli.link)
    if link is None:
        raise RuntimeError(f"No prim named {args_cli.link} in the candidate.")
    print(f"[diag] Q1 link: {link.GetPath()}  hasMassAPI={link.HasAPI(UsdPhysics.MassAPI)}")
    attr = UsdPhysics.MassAPI(link).GetDiagonalInertiaAttr()
    print(f"[diag] Q1 diagonalInertia: IsValid={attr.IsValid()}")
    print(f"[diag] Q1 diagonalInertia: HasAuthoredValue={attr.HasAuthoredValue()}")
    print(f"[diag] Q1 diagonalInertia: HasValue={attr.HasValue()}  Get()={attr.Get()!r}")
    print(f"[diag] Q1 diagonalInertia: resolve source={attr.GetResolveInfo().GetSource()!r}")
    for name, a in (("mass", UsdPhysics.MassAPI(link).GetMassAttr()),
                    ("centerOfMass", UsdPhysics.MassAPI(link).GetCenterOfMassAttr())):
        print(f"[diag] Q1 control {name}: HasAuthoredValue={a.HasAuthoredValue()}  Get()={a.Get()!r}")

    report_arcs(stage, "Q2 candidate")

    src = UR5E_HOME_CFG.spawn.usd_path
    print(f"[diag] Q3 control source robot: {src}")
    src_stage = Usd.Stage.Open(src)
    if src_stage is None:
        print("[diag] Q3 could not open the shipped robot; control skipped.")
        return
    flat = Usd.Stage.Open(src_stage.Flatten())
    report_arcs(flat, "Q3 flattened-ur5e-only")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
