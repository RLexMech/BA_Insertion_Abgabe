# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Measure WHY two checks of author_tool_ur5e.py fail on its candidate.

RT-13 (2026-08-23) authored the tool into the UR5e and 18 of 20 checks passed,
including every geometric one. Two failed:

    diagonal_inertia_left_to_physx: FAIL   (measured [0.0, 0.0, 0.0])
    self_contained:                 FAIL   (external_arcs == [""])

Both look like DEFECTS OF THE CHECK rather than of the asset, but "looks like"
is not a measurement, and weakening a check to make it pass is how a check
stops testing and starts certifying. So this script measures the cause first
and changes nothing.

WRITTEN BEFORE THE RUN -- the two questions, and what each answer means:

Q1  Is ``physics:diagonalInertia`` AUTHORED on tool_link, or is [0,0,0] the
    schema fallback that ``Get()`` returns for an unauthored attribute?
      -> authored = False, resolve source = fallback
         The asset is RIGHT (PhysX will derive the tensor, and (0,0,0) is the
         sentinel that asks it to). The CHECK is wrong: it tests ``Get() is
         None`` where it must test ``HasAuthoredValue()``.
      -> authored = True
         The asset is WRONG: something writes the tensor. Fix the authoring,
         not the check.

Q2  Where does the ONE EMPTY composition dependency come from?
      -> an internal reference or payload (assetPath == "") on some prim
         The asset is RIGHT: an empty asset path points INTO the same layer and
         can never resolve to a missing file, which is the whole risk the check
         exists for. The CHECK is wrong: it must ignore empty entries.
      -> a real file path that merely prints as empty
         The asset is WRONG and the check caught a genuine dangling arc.

Q3  (control) Does the SHIPPED UR5e already produce the same empty entry after
    a plain flatten, with no tool involved?
      -> yes: the entry has nothing to do with the tool reference.
      -> no:  the tool reference is what introduces it, and Q2 must say how.

Usage on the training PC:

    .\scripts\rt_log.ps1 RT-15 python scripts/diagnose_tool_candidate.py
"""

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

# No sys.path juggling: author_tool_ur5e.py imports the same package plainly,
# so `insertion` is installed in the training env and adding a second path
# would only risk importing it twice under two names.
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
    """Print every composition dependency with its REPR, so an empty string is
    visible as '' instead of vanishing, and then say which prim produced it."""
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

    # ---------------------------------------------------------------- Q1
    link = find_prim_by_name(stage, args_cli.link)
    if link is None:
        raise RuntimeError(f"No prim named {args_cli.link} in the candidate.")
    print(f"[diag] Q1 link: {link.GetPath()}  hasMassAPI={link.HasAPI(UsdPhysics.MassAPI)}")
    attr = UsdPhysics.MassAPI(link).GetDiagonalInertiaAttr()
    print(f"[diag] Q1 diagonalInertia: IsValid={attr.IsValid()}")
    print(f"[diag] Q1 diagonalInertia: HasAuthoredValue={attr.HasAuthoredValue()}")
    print(f"[diag] Q1 diagonalInertia: HasValue={attr.HasValue()}  Get()={attr.Get()!r}")
    print(f"[diag] Q1 diagonalInertia: resolve source={attr.GetResolveInfo().GetSource()!r}")
    # The same question for the two attributes that ARE authored, as a control:
    # if these also report HasAuthoredValue=True, the API is reading correctly.
    for name, a in (("mass", UsdPhysics.MassAPI(link).GetMassAttr()),
                    ("centerOfMass", UsdPhysics.MassAPI(link).GetCenterOfMassAttr())):
        print(f"[diag] Q1 control {name}: HasAuthoredValue={a.HasAuthoredValue()}  Get()={a.Get()!r}")

    # ---------------------------------------------------------------- Q2
    report_arcs(stage, "Q2 candidate")

    # ---------------------------------------------------------------- Q3
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
