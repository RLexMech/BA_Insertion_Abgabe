# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""One implementation of each recurring USD predicate (D-080).

WHY THIS EXISTS
---------------
``self_contained`` was written three times in this repo and got it right once:

  * ``verify_fixture_usd.py:170-192``  -- CORRECT. Collects every reference and
    payload with its asset path and classifies an EMPTY path as ``internal``.
  * ``author_tool_ur5e.py``            -- wrong. Counted every composition
    dependency, empty ones included, so no flattened asset could ever pass.
    Cost run RT-13. Fixed in f811076.
  * ``convert_step_asset.py:215``      -- wrong, and still live. Writes
    ``str(ref.assetPath)`` over a list of STRINGS, so it raises AttributeError
    the moment the list is non-empty. That script has never run.

Three copies, three chances to get it wrong, and the correct one sat three
files away the whole time. The repo rule "check whether logic already exists"
only works if there is ONE place to find it. This is that place.

THE SPLIT, and why it is not cosmetic
-------------------------------------
Every predicate is two functions:

    collect_*   touches pxr. Reads the stage, returns plain data.
    classify_*  pure Python. Decides what the data MEANS.

The RT-13 bug was entirely in the second half -- what an empty asset path
means -- and the second half is therefore TESTABLE ON THE LAPTOP, where
``import pxr`` fails. Splitting puts the part that was wrong where it can be
checked without a round trip to the training machine.

``pxr`` is imported INSIDE the collecting functions, not at module scope, so
this file can be imported and self-tested on a machine that has no USD at all.

THE SELF-TEST CONTRACT, written before the implementation
---------------------------------------------------------
``python scripts/tools/usd_predicates.py --self-test`` must prove:

   1. an arc with an EMPTY asset path is internal, never external
      -- the RT-15 measurement: a flattened layer always reports one, and the
         shipped UR5e alone does it too
   2. an arc with a real asset path is external
   3. a mixed list yields only the real one as external
   4. an empty list yields two empty lists
   5. ``is_self_contained`` is True for a stage whose only arc is empty
   6. ``is_self_contained`` is False for one real external arc
   7. the OLD assertion (``not dependencies``) is replayed and shown to be
      False on exactly the input RT-13 hit -- the bug must be reproduced here,
      not described
   8. ``classify_arcs`` does not mutate its input

Usage:

    python scripts/tools/usd_predicates.py --self-test
"""

from __future__ import annotations

import argparse
import copy
from typing import Any

PREDICATES_MARKER = "usd_predicates-2026-08-24a"


# ===========================================================================
#  Composition arcs
# ===========================================================================
def collect_arcs(stage: Any) -> list[dict]:
    """Every reference and payload in the stage, as plain data.

    Ported from the CORRECT implementation, ``verify_fixture_usd.py:173-190``.
    Returns dicts so the classifying half never needs pxr.
    """
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
    """The root layer's composition asset dependencies, as strings.

    MEASURED, run RT-15 (2026-08-23): this returns STRINGS, not objects with an
    ``assetPath`` attribute -- ``convert_step_asset.py:215`` assumes otherwise
    and raises AttributeError as soon as the list is non-empty. And a FLATTENED
    layer always contains exactly one entry whose string is EMPTY, the shipped
    UR5e included, with no sublayer, reference or payload to account for it.
    """
    return [str(p) for p in stage.GetRootLayer().GetCompositionAssetDependencies()]


def classify_arcs(arcs: list[dict]) -> dict[str, list[dict]]:
    """Pure. An EMPTY asset path targets this same file and is self-contained.

    This is the whole of the RT-13 defect, isolated so it can be tested without
    a simulator: an external arc to a file the training machine does not have
    resolves to nothing and spawns an invisible asset (the 2026-07-26 failure).
    An empty path is not a file name and cannot fail to resolve, so counting it
    made ``self_contained`` unpassable by any flattened asset, however correct.
    """
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


# ===========================================================================
#  Attributes
# ===========================================================================
def authored(attr: Any) -> bool:
    """Is the attribute WRITTEN, as opposed to reading back a schema fallback?

    MEASURED, run RT-15 (2026-08-23): ``Get()`` on an unauthored attribute
    returns the schema's fallback value, and ``UsdPhysics`` gives
    ``physics:diagonalInertia`` a fallback of (0,0,0) -- itself the sentinel
    that asks PhysX to compute the tensor. So ``Get() is None`` is never true
    and cannot be used to mean "not written". Controls in the same run:
    ``physics:mass`` and ``physics:centerOfMass``, which ARE written, report
    HasAuthoredValue=True.
    """
    return bool(attr is not None and attr.IsValid() and attr.HasAuthoredValue())


# ===========================================================================
#  Prims
# ===========================================================================
def find_prim_by_name(root: Any, name: str) -> Any:
    """First prim with this name, under a stage or under a prim.

    Three copies of this existed (author_tool_ur5e.py, the proxy's
    author_peg_ur10e.py -- deleted 2026-08-28, S6 -- and
    diagnose_tool_candidate.py). One now.
    """
    from pxr import Usd  # noqa: PLC0415 -- kept out of module scope on purpose

    prims = root.Traverse() if isinstance(root, Usd.Stage) else Usd.PrimRange(root)
    for prim in prims:
        if prim.GetName() == name:
            return prim
    return None


# ===========================================================================
#  Self-test -- the eight cases above. No USD anywhere.
# ===========================================================================
def _self_test() -> int:
    checked = 0
    empty = {"prim": "/a", "arc": "reference", "asset_path": "", "target_prim_path": "/b"}
    real = {"prim": "/c", "arc": "payload", "asset_path": "../Robotiq/x.usd", "target_prim_path": ""}

    # 1 / 2 / 3 / 4
    assert classify_arcs([empty]) == {"internal": [empty], "external": []}
    checked += 1
    assert classify_arcs([real]) == {"internal": [], "external": [real]}
    checked += 1
    assert classify_arcs([empty, real])["external"] == [real]
    checked += 1
    assert classify_arcs([]) == {"internal": [], "external": []}
    checked += 1

    # 5 / 6 -- on the dependency-string form, which is what the authoring
    # scripts actually read
    assert classify_dependencies([""])["external"] == []
    checked += 1
    assert classify_dependencies(["configuration/ur5e_base.usd"])["external"] == [
        "configuration/ur5e_base.usd"
    ]
    checked += 1

    # 7 -- the RT-13 bug, reproduced rather than described. This is the exact
    # list the candidate reported: one entry, empty.
    rt13_deps = [""]
    old_assertion = not rt13_deps           # the check as it stood
    new_assertion = is_self_contained_from(rt13_deps)
    assert old_assertion is False, "the replay must reproduce the bug, not hide it"
    assert new_assertion is True, "the corrected rule must accept a flattened asset"
    checked += 1

    # 8 -- classification must not chew on its input
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
