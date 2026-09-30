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
parser.add_argument(
    "--usd",
    type=str,
    required=True,
    help=(
        "Path to the USD file. A leading 'ISAACLAB_NUCLEUS_DIR/' or "
        "'ISAAC_NUCLEUS_DIR/' is expanded after the app starts."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom  # noqa: E402

# These constants read a carb setting, so importing them before AppLauncher
# raises ModuleNotFoundError. That is why the expansion lives here and not in
# the argument parser above.
from isaaclab.utils import assets as _assets  # noqa: E402

# Which of these names the installed Isaac Lab actually carries is not known
# from the laptop, so a missing one is skipped instead of killing the run.
NUCLEUS_PREFIXES = {
    name: getattr(_assets, name)
    for name in ("ISAACLAB_NUCLEUS_DIR", "ISAAC_NUCLEUS_DIR", "NUCLEUS_ASSET_ROOT_DIR")
    if hasattr(_assets, name)
}


def expand_nucleus(raw: str) -> str:
    """Turn 'ISAACLAB_NUCLEUS_DIR/Factory/x.usd' into the resolved URL."""
    for name in sorted(NUCLEUS_PREFIXES, key=len, reverse=True):
        root = NUCLEUS_PREFIXES[name]
        if raw == name or raw.startswith(name + "/"):
            tail = raw[len(name) :].lstrip("/")
            resolved = f"{root}/{tail}" if tail else root
            print(f"[list_usd_prims] {name} = {root}")
            return resolved
    if raw.split("/", 1)[0].endswith("_DIR"):
        raise SystemExit(
            f"[list_usd_prims] unknown nucleus prefix in --usd: {raw!r}. "
            f"Known here: {sorted(NUCLEUS_PREFIXES) or 'none'}"
        )
    return raw


def main() -> None:
    # A Nucleus/omniverse URL must not go through pathlib -- resolve() mangles
    # the scheme. Needed for reading shipped reference assets, e.g. Factory's
    # franka_mimic.usd (the FORGE force_sensor link, D-114 groundwork).
    raw_usd = expand_nucleus(args_cli.usd)
    if "://" in raw_usd:
        usd_path = raw_usd
    else:
        usd_path = str(pathlib.Path(raw_usd).resolve())
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open stage: {usd_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    print(f"[list_usd_prims] {usd_path}")
    for prim in stage.Traverse():
        indent = "  " * prim.GetPath().pathElementCount
        world_xf = xform_cache.GetLocalToWorldTransform(prim)
        translation = world_xf.ExtractTranslation()
        pos = f"({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})"
        line = f"{indent}{prim.GetName()}  [{prim.GetTypeName()}]  world_pos={pos}  path={prim.GetPath()}"
        # Authored mass properties, printed only where they exist: the point of
        # reading a reference robot is what its links CARRY, not just where
        # they sit (e.g. the force_sensor link's mass/inertia, D-114).
        mass_bits = []
        for attr_name, label in (
            ("physics:mass", "mass"),
            ("physics:diagonalInertia", "diagInertia"),
            ("physics:centerOfMass", "com"),
        ):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid() and attr.HasAuthoredValue():
                mass_bits.append(f"{label}={attr.Get()}")
        if mass_bits:
            line += "  " + "  ".join(mass_bits)
        print(line)


if __name__ == "__main__":
    main()
    simulation_app.close()
