# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Add the bore entrance frame to Tisch.usd; local +z points into the bore."""

from __future__ import annotations

import argparse
import pathlib
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Add the D-017 bore task frame to a fixture USD.")
parser.add_argument("--usd", type=str, required=True, help="Path to the fixture USD file (edited in place).")
parser.add_argument(
    "--parent-path",
    type=str,
    default="/World/tisch5",
    help="Prim path of the table's root Xform (the one sitting at the CAD raw origin).",
)
parser.add_argument("--frame-name", type=str, default="TaskFrame_BoreEntrance")
parser.add_argument("--offset", type=float, nargs=3, default=(0.0, -0.225, 0.055), metavar=("X", "Y", "Z"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402


def main() -> None:
    usd_path = pathlib.Path(args_cli.usd).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {usd_path}")

    parent = stage.GetPrimAtPath(args_cli.parent_path)
    if not parent.IsValid():
        raise RuntimeError(f"Parent prim not found: {args_cli.parent_path}")

    frame_path = Sdf.Path(args_cli.parent_path).AppendChild(args_cli.frame_name)
    xform = UsdGeom.Xform.Define(stage, frame_path)

    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*args_cli.offset))
    rotate_op = xform.AddRotateXOp()
    rotate_op.Set(180.0)

    stage.GetRootLayer().Save()

    print(f"[add_bore_task_frame] added {frame_path} under {args_cli.parent_path}")
    print(f"[add_bore_task_frame] translate={args_cli.offset}, rotateX=180 (local +z now points into the bore)")
    print(f"[add_bore_task_frame] saved {usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
