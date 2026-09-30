# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline geometry check of the generated table (D-033). No Isaac required.

``author_tisch_square.py`` verifies the USD it wrote by measuring the authored
prims back. That check is the stronger one, but it runs only on the training
machine, which means a geometry slip made on the dev PC is invisible until the
next transfer. This script attacks the same arithmetic one stage earlier --
``box_specs`` alone, before any USD exists -- so the box layout is falsifiable
here, in the same spirit as ``check_demo_reward_math.py`` for the reward.

What it asserts, per pocket size across the whole curriculum:

- the nine boxes union to the contract bbox, 0.500 x 0.600 x 0.755, z -0.755..0
- no two boxes overlap (overlapping static colliders are a PhysX contact-noise
  source, and they would also make the tiling check pass by accident)
- the four plate pieces plus the floor fill the plate slab EXACTLY, minus the
  pocket void: a gap here is a hole in the table that no visual would reveal
- the pocket mouth is empty down to the floor -- the sealed-pocket failure mode
  (D-029 risk 1) restated as a measurement rather than an argument
- the walls are full-thickness, the floor leaves 20 mm of material, the legs
  stay inside the footprint and reach the ground

Run:

    python scripts\\check_tisch_geometry.py
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys
import types

REPO = pathlib.Path(__file__).resolve().parent.parent
AUTHOR_SCRIPT = REPO / "scripts" / "author_tisch_square.py"
TASKS_CFG = (
    REPO / "source" / "proxytask" / "proxytask" / "tasks" / "direct" / "proxytask"
    / "proxytask_tasks_cfg.py"
)

# The curriculum, plus the bounds of the accepted range. 60 mm is the widest
# pocket _pocket_from_env admits and the case where the plate pieces around the
# pocket are thinnest, so it is the one most likely to degenerate.
POCKET_SIDES_MM = (32.0, 34.0, 36.0, 40.0, 45.0, 60.0)
TOL = 1e-12

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def stub_isaac() -> None:
    """Make ``author_tisch_square`` importable without Isaac.

    Only ``box_specs`` is exercised, and it is pure arithmetic: the pxr and
    AppLauncher names it imports at module scope are never called here. The
    real ``proxytask_tasks_cfg`` IS loaded, by path, because the plate figures
    under test come from it -- stubbing that too would leave this script
    checking its own assumptions.
    """
    app = types.ModuleType("isaaclab.app")

    class AppLauncher:  # noqa: D401 - stub
        def __init__(self, args):
            self.app = types.SimpleNamespace(close=lambda: None)

        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true")

    app.AppLauncher = AppLauncher
    isaaclab = types.ModuleType("isaaclab")
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    isaaclab.app, isaaclab.utils = app, utils
    sys.modules.update({"isaaclab": isaaclab, "isaaclab.app": app, "isaaclab.utils": utils})

    pxr = types.ModuleType("pxr")
    for name in ("Gf", "Usd", "UsdGeom", "UsdPhysics"):
        sub = types.ModuleType(f"pxr.{name}")
        setattr(pxr, name, sub)
        sys.modules[f"pxr.{name}"] = sub
    sys.modules["pxr"] = pxr

    # Graft the real tasks_cfg into the package hierarchy. Importing the
    # proxytask package normally would run its __init__, which needs
    # isaaclab_tasks; this leaf is all the generator reads.
    cfg_spec = importlib.util.spec_from_file_location(
        "proxytask.tasks.direct.proxytask.proxytask_tasks_cfg", TASKS_CFG
    )
    cfg = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg)
    chain = (
        "proxytask",
        "proxytask.tasks",
        "proxytask.tasks.direct",
        "proxytask.tasks.direct.proxytask",
    )
    for name in chain:
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg
    for parent, child in zip(chain, chain[1:]):
        setattr(sys.modules[parent], child.rsplit(".", 1)[1], sys.modules[child])
    sys.modules[chain[-1]].proxytask_tasks_cfg = cfg
    sys.modules[chain[-1] + ".proxytask_tasks_cfg"] = cfg

    sys.argv = [AUTHOR_SCRIPT.name]


def volume(box: dict) -> float:
    lo, hi = box["min"], box["max"]
    return (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])


def overlap_volume(a: dict, b: dict) -> float:
    v = 1.0
    for i in range(3):
        lo, hi = max(a["min"][i], b["min"][i]), min(a["max"][i], b["max"][i])
        if hi - lo <= TOL:
            return 0.0
        v *= hi - lo
    return v


def main() -> None:
    stub_isaac()
    spec = importlib.util.spec_from_file_location("author_tisch_square", AUTHOR_SCRIPT)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    plate_x, plate_y, plate_z = gen.PLATE_X, gen.PLATE_Y, gen.PLATE_Z
    depth = gen.geom.POCKET_DEPTH
    print(f"[check_tisch] generator marker: {gen.SCRIPT_MARKER}")
    print(f"[check_tisch] plate x={plate_x} y={plate_y} z={plate_z}, pocket depth {depth * 1000:g} mm")

    for b_mm in POCKET_SIDES_MM:
        b = b_mm / 1000.0
        h = b / 2.0
        print(f"\n[check_tisch] pocket {b_mm:g} mm")
        boxes = gen.box_specs(b, depth)
        named = {x["name"]: x for x in boxes}
        plate = [named[n] for n in ("plate_south", "plate_north", "plate_west", "plate_east")]
        legs = [x for x in boxes if x["name"].startswith("leg_")]
        floor = named["pocket_floor"]

        check("nine boxes", len(boxes) == 9, str(len(boxes)))
        check(
            "no box is degenerate",
            all(all(x["max"][i] - x["min"][i] > TOL for i in range(3)) for x in boxes),
        )

        lo = [min(x["min"][i] for x in boxes) for i in range(3)]
        hi = [max(x["max"][i] for x in boxes) for i in range(3)]
        size = [hi[i] - lo[i] for i in range(3)]
        check(
            "bbox 0.500 x 0.600 x 0.755",
            all(abs(size[i] - e) < 1e-9 for i, e in enumerate((0.500, 0.600, 0.755))),
            str([round(s, 6) for s in size]),
        )
        check(
            "bbox z range -0.755 .. 0.000",
            abs(lo[2] + 0.755) < 1e-9 and abs(hi[2]) < 1e-9,
            f"{lo[2]:.6f} .. {hi[2]:.6f}",
        )

        worst = max(
            (overlap_volume(a, c), a["name"], c["name"])
            for a, c in itertools.combinations(boxes, 2)
        )
        check("no pairwise overlap", worst[0] == 0.0, f"worst {worst[1]}/{worst[2]} = {worst[0]}")

        slab = (plate_x[1] - plate_x[0]) * (plate_y[1] - plate_y[0]) * (plate_z[1] - plate_z[0])
        void = b * b * depth
        filled = sum(volume(x) for x in plate) + volume(floor)
        check(
            "plate tiles exactly (slab minus pocket void)",
            abs(filled - (slab - void)) < 1e-15,
            f"{filled:.9f} m^3 vs {slab - void:.9f} m^3",
        )

        check(
            "pocket inner faces at +-b/2",
            all(
                abs(v) < TOL
                for v in (
                    named["plate_west"]["max"][0] + h,
                    named["plate_east"]["min"][0] - h,
                    named["plate_south"]["max"][1] + h,
                    named["plate_north"]["min"][1] - h,
                )
            ),
        )
        check(
            "walls span the full plate thickness",
            all(
                abs(x["min"][2] - plate_z[0]) < TOL and abs(x["max"][2] - plate_z[1]) < TOL
                for x in plate
            ),
        )
        check(
            "floor top at -depth, bottom at plate bottom",
            abs(floor["max"][2] + depth) < TOL and abs(floor["min"][2] - plate_z[0]) < TOL,
        )
        check(
            "floor covers the whole pocket footprint",
            abs(floor["min"][0] + h) < TOL
            and abs(floor["max"][0] - h) < TOL
            and abs(floor["min"][1] + h) < TOL
            and abs(floor["max"][1] - h) < TOL,
        )
        material = floor["max"][2] - floor["min"][2]
        check(
            "material left under the pocket",
            abs(material - (plate_z[1] - plate_z[0] - depth)) < TOL,
            f"{material * 1000:.1f} mm",
        )

        # The sealed-pocket failure mode, as a measurement: a thin probe filling
        # the mouth from the floor to the opening plane must hit nothing.
        mouth = {"min": (-h, -h, -depth + 1e-9), "max": (h, h, -1e-9)}
        blockers = [x["name"] for x in boxes if overlap_volume(mouth, x) > 0.0]
        check("pocket mouth is open, not sealed", not blockers, ", ".join(blockers))

        check(
            "legs inside the plate footprint",
            all(
                x["min"][0] >= plate_x[0] - TOL
                and x["max"][0] <= plate_x[1] + TOL
                and x["min"][1] >= plate_y[0] - TOL
                and x["max"][1] <= plate_y[1] + TOL
                for x in legs
            ),
        )
        check(
            "legs span ground to plate underside",
            all(
                abs(x["min"][2] + 0.755) < TOL and abs(x["max"][2] - plate_z[0]) < TOL
                for x in legs
            ),
        )
        check(
            "four legs, 50 x 50 mm section",
            len(legs) == 4
            and all(
                abs((x["max"][0] - x["min"][0]) - gen.LEG_SECTION) < TOL
                and abs((x["max"][1] - x["min"][1]) - gen.LEG_SECTION) < TOL
                for x in legs
            ),
        )
        centres = sorted(round((x["min"][0] + x["max"][0]) / 2.0, 9) for x in legs)
        check(
            "legs symmetric about x = 0",
            abs(centres[0] + centres[-1]) < 1e-9,
            f"centres {sorted(set(centres))}",
        )

    print()
    if failures:
        print(f"[check_tisch] FAILED ({len(failures)}): {sorted(set(failures))}")
        raise SystemExit(1)
    print("[check_tisch] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
