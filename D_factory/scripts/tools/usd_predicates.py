# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""One implementation of each recurring USD predicate (D-080)."""

from __future__ import annotations

import argparse
import copy
from typing import Any

PREDICATES_MARKER = "usd_predicates-2026-08-24a"


def collect_arcs(stage: Any) -> list[dict]:
    """Every reference and payload in the stage, as plain data."""
    arcs: list[dict] = []
    for prim in stage.Traverse():
        for arc_kind, meta_key in (("reference", "references"), ("payload", "payload")):
            list_op = prim.GetMetadata(meta_key)
            if not list_op:
                continue
            items = getattr(list_op, "GetAddedOrExplicitItems", list)()
            for item in items:
                arcs.append(
                    {
                        "prim": str(prim.GetPath()),
                        "arc": arc_kind,
                        "asset_path": str(getattr(item, "assetPath", "") or ""),
                        "target_prim_path": str(getattr(item, "primPath", "") or ""),
                    }
                )
    return arcs


def collect_root_dependencies(stage: Any) -> list[str]:
    """The root layer's composition asset dependencies, as strings."""
    return [str(p) for p in stage.GetRootLayer().GetCompositionAssetDependencies()]


def classify_arcs(arcs: list[dict]) -> dict[str, list[dict]]:
    """Pure."""
    internal = [a for a in arcs if not a.get("asset_path")]
    external = [a for a in arcs if a.get("asset_path")]
    return {"internal": internal, "external": external}


def classify_dependencies(deps: list[str]) -> dict[str, list[str]]:
    """Same rule for the root-layer dependency list."""
    return {
        "internal": [d for d in deps if not d],
        "external": [d for d in deps if d],
    }


def external_arcs(stage: Any) -> list[str]:
    """Composition arcs pointing at another FILE. Empty paths are not files."""
    return classify_dependencies(collect_root_dependencies(stage))["external"]


def is_self_contained(stage: Any) -> bool:
    """True when nothing in this stage depends on a second file."""
    return not external_arcs(stage)


def authored(attr: Any) -> bool:
    """Is the attribute WRITTEN, as opposed to reading back a schema fallback?"""
    return bool(attr is not None and attr.IsValid() and attr.HasAuthoredValue())


def find_prim_by_name(root: Any, name: str) -> Any:
    """First prim with this name, under a stage or under a prim."""
    from pxr import Usd  # noqa: PLC0415 -- kept out of module scope on purpose

    prims = root.Traverse() if isinstance(root, Usd.Stage) else Usd.PrimRange(root)
    for prim in prims:
        if prim.GetName() == name:
            return prim
    return None


def _self_test() -> int:
    checked = 0
    empty = {"prim": "/a", "arc": "reference", "asset_path": "", "target_prim_path": "/b"}
    real = {"prim": "/c", "arc": "payload", "asset_path": "../Robotiq/x.usd", "target_prim_path": ""}

    assert classify_arcs([empty]) == {"internal": [empty], "external": []}
    checked += 1
    assert classify_arcs([real]) == {"internal": [], "external": [real]}
    checked += 1
    assert classify_arcs([empty, real])["external"] == [real]
    checked += 1
    assert classify_arcs([]) == {"internal": [], "external": []}
    checked += 1

    assert classify_dependencies([""])["external"] == []
    checked += 1
    assert classify_dependencies(["configuration/ur5e_base.usd"])["external"] == [
        "configuration/ur5e_base.usd"
    ]
    checked += 1

    rt13_deps = [""]
    old_assertion = not rt13_deps
    new_assertion = is_self_contained_from(rt13_deps)
    assert old_assertion is False, "the replay must reproduce the bug, not hide it"
    assert new_assertion is True, "the corrected rule must accept a flattened asset"
    checked += 1

    arcs = [copy.deepcopy(empty), copy.deepcopy(real)]
    before = copy.deepcopy(arcs)
    classify_arcs(arcs)
    assert arcs == before
    checked += 1

    assert checked == 8, f"only {checked} of 8 cases ran"
    print(f"[usd_predicates] self-test: {checked}/8 cases passed ({PREDICATES_MARKER})")
    return 0


def is_self_contained_from(deps: list[str]) -> bool:
    """``is_self_contained`` decided on an already-collected list. Pure."""
    return not classify_dependencies(deps)["external"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Shared USD predicates (D-080).")
    parser.add_argument("--self-test", action="store_true",
                        help="Prove the classifying half. Needs no USD and no Isaac.")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
