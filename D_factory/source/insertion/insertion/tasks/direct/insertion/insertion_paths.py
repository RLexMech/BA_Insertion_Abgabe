# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Where the Warp OBJ meshes live. Stdlib only -- no Isaac, no torch.

WHY THIS IS ITS OWN FILE
------------------------
The block below used to sit in ``insertion_tasks_cfg.py``. That file imports
``isaaclab.utils.configclass`` at module level, so importing it without a
running ``SimulationApp`` raises ``ModuleNotFoundError: No module named
'isaaclab.utils'`` -- which is exactly what RT-88 hit: the sign probe wanted a
FILE PATH and paid for a simulation app to get it.

A path is not a config. It needs ``os`` and nothing else, so anything may ask
for it: a check script with no Isaac, the laptop, a tool. The fact keeps ONE
home -- this file -- and ``insertion_tasks_cfg.py`` re-exports the names so
every existing caller is unaffected.
"""

from __future__ import annotations

import os


# --- the Warp meshes: one OBJ per asset, beside its own USD -----------------
# The SDF reward and the SAPU interpenetration filter do NOT read the USD
# stage. Isaac Lab 2.3.2 `industreal_algo_utils.load_asset_mesh_in_warp` loads
# an OBJ with `trimesh` and hands its vertices and faces to `wp.Mesh`; AutoMate
# keeps that OBJ in the same `assembly_dir` as the USD it belongs to, and this
# module keeps the same layout.
#
# NEITHER FILE IS AUTHORED HERE. Both come out of `scripts/export_asset_mesh.py`
# on the training machine, like every other asset in this module: the resolver
# can look one up and say what to run, never produce it.
#
# THE PART'S OBJ IS IN THE `tool_link` FRAME, not the part prim's. The env
# reads the part's pose at `peg_body_name` (= TOOL_LINK_NAME), so a mesh
# expressed anywhere else would be offset by a constant transform that no
# downstream number could reveal. The export flag that pins this is
# `--frame-prim tool_link`.
PART_OBJ_NAME = "fuegeteil.obj"
POCKET_OBJ_NAME = "aufnahme.obj"


def default_part_obj_path() -> str:
    """Where the part's Warp mesh lives: beside the tool-welded UR5e USD."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "Robot", PART_OBJ_NAME
    )


def default_pocket_obj_path() -> str:
    """Where the fixture's Warp mesh lives: beside the fixture cut-out USD."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "Werkzeug", POCKET_OBJ_NAME
    )


def resolve_part_obj_path() -> str:
    """Path to the part's Warp mesh, or raise with the command that makes it."""
    env_override = os.environ.get("INSERTION_PART_OBJ")
    candidates = [p for p in (env_override, default_part_obj_path()) if p]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Part OBJ not found. Tried: "
        + "; ".join(candidates)
        + ". Export it on the training machine with scripts/export_asset_mesh.py "
        "--prim FUEGETEIL --frame-prim tool_link --expect-part-bbox, "
        "or set INSERTION_PART_OBJ."
    )


def resolve_pocket_obj_path() -> str:
    """Path to the fixture's Warp mesh, or raise with the command that makes it."""
    env_override = os.environ.get("INSERTION_POCKET_OBJ")
    candidates = [p for p in (env_override, default_pocket_obj_path()) if p]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Fixture OBJ not found. Tried: "
        + "; ".join(candidates)
        + ". Export it on the training machine with scripts/export_asset_mesh.py "
        "against the fixture USD, or set INSERTION_POCKET_OBJ."
    )

