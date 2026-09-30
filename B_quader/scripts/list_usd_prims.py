# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""List every prim in a USD stage with its type and world-space translation.

Use this to check whether a coordinate system / Xform authored in the source
CAD tool (e.g. a bore-entrance frame for D-017) actually survived the STEP to
USD import, and where it ended up.

    cd C:\\Isaaclab
    conda activate env_isaaclab
    .\\isaaclab.bat -p <repo>\\scripts\\list_usd_prims.py --usd <path\\to\\Tisch.usd>
"""

from __future__ import annotations

import argparse
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List all prims in a USD stage.")
parser.add_argument("--usd", type=str, required=True, help="Path to the USD file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom  # noqa: E402


def main() -> None:
    usd_path = pathlib.Path(args_cli.usd).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {usd_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    print(f"[list_usd_prims] {usd_path}")
    for prim in stage.Traverse():
        indent = "  " * prim.GetPath().pathElementCount
        world_xf = xform_cache.GetLocalToWorldTransform(prim)
        translation = world_xf.ExtractTranslation()
        pos = f"({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})"
        print(f"{indent}{prim.GetName()}  [{prim.GetTypeName()}]  world_pos={pos}  path={prim.GetPath()}")


if __name__ == "__main__":
    main()
    simulation_app.close()
