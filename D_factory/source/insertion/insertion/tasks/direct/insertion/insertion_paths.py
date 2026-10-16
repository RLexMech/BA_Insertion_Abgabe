# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Where the Warp OBJ meshes live."""

from __future__ import annotations

import os


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
