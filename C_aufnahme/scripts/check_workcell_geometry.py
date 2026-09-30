# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Close the workcell dimension chain offline. No Isaac, no USD, no GPU.

Runs on the dev laptop. It answers one question: are the numbers in the
WORKCELL section of ``insertion_tasks_cfg.py`` mutually consistent, and where
do the on-site measurements over-determine each other?

Why it exists: the cell was measured edge-to-edge with a tape, and several
readings constrain the same quantity. A chain that silently absorbs a
contradiction looks exactly like a chain that closes. Every residual below is
printed as a number, and the ones that are real disagreements are named.

    python scripts/check_workcell_geometry.py
    python scripts/check_workcell_geometry.py --json out/workcell_check_1.json

Exit code 0 on PASS, 1 on FAIL. PENDING inputs do not fail the run -- they are
reported separately, because the chain is internally consistent without them
and only the ABSOLUTE placement waits on the photo.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import types

MM = 1000.0

CFG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "source"
    / "insertion"
    / "insertion"
    / "tasks"
    / "direct"
    / "insertion"
    / "insertion_tasks_cfg.py"
)


def load_cfg() -> types.ModuleType:
    """Read the cfg module FROM SOURCE with ``isaaclab`` stubbed out.

    The dev laptop has no Isaac. The WORKCELL constants are plain floats and
    need nothing from isaaclab, but the module imports ``configclass`` at the
    top, so a bare import would die before reaching them. Stubbing is honest
    here in a way that copying the numbers into this file would not be: the
    check then reads the SAME constants the environment reads.

    The source text is compiled here instead of going through
    ``spec.loader.exec_module``, and that is not style. Bytecode caching bit
    this script on 2026-08-18: a negative-control run edited one constant from
    0.105 to 0.505, ``__pycache__`` was written from the edited file, and the
    restore produced a source of IDENTICAL LENGTH within the SAME SECOND. The
    cache validity header (mtime + size) therefore still matched, and the next
    run silently reported the numbers of a file that no longer existed. A
    checker that can certify values which are not in the file is worse than no
    checker, so the cache is taken out of the path entirely.
    """
    if "isaaclab" not in sys.modules:
        pkg = types.ModuleType("isaaclab")
        utils = types.ModuleType("isaaclab.utils")
        utils.configclass = lambda c: c
        pkg.utils = utils
        sys.modules["isaaclab"] = pkg
        sys.modules["isaaclab.utils"] = utils

    src = CFG_PATH.read_text(encoding="utf-8")
    mod = types.ModuleType("_workcell_cfg")
    mod.__file__ = str(CFG_PATH)
    exec(compile(src, str(CFG_PATH), "exec"), mod.__dict__)  # noqa: S102
    return mod


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.failed = 0

    def check(self, label: str, lhs_mm: float, rhs_mm: float, tol_mm: float, note: str = "") -> None:
        diff = lhs_mm - rhs_mm
        ok = abs(diff) <= tol_mm
        if not ok:
            self.failed += 1
        self.rows.append(
            {"label": label, "lhs_mm": lhs_mm, "rhs_mm": rhs_mm,
             "diff_mm": diff, "tol_mm": tol_mm, "ok": ok, "note": note}
        )
        flag = "OK  " if ok else "FAIL"
        tail = f"   {note}" if note else ""
        print(f"  {flag} {label:46s} {lhs_mm:9.1f} vs {rhs_mm:9.1f}   "
              f"diff {diff:+7.1f}  (tol {tol_mm:g}){tail}")

    def require(self, label: str, ok: bool, detail: str) -> None:
        """One-sided condition. Separate from ``check`` on purpose: expressing
        'must be inside' as a two-sided tolerance invites picking a tolerance
        so wide that the test cannot fail, which is what the first draft of
        this file did."""
        if not ok:
            self.failed += 1
        self.rows.append({"label": label, "ok": ok, "note": detail})
        print(f"  {'OK  ' if ok else 'FAIL'} {label:46s} {detail}")

    def value(self, label: str, value_mm: float, note: str = "") -> None:
        self.rows.append({"label": label, "value_mm": value_mm, "note": note})
        tail = f"   {note}" if note else ""
        print(f"       {label:46s} {value_mm:9.1f} mm{tail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, default=None,
                    help="Write the full report to this path. No default: a "
                         "default would overwrite an earlier run.")
    args = ap.parse_args()

    c = load_cfg()
    r = Report()

    print("=" * 78)
    print("1. Closure of the on-site chains (these must close by construction)")
    print("=" * 78)
    r.check("rect left + width + right   vs  table 2 Y",
            (c.RECT_TO_T2_LEFT + c.RECT_SIZE_Y + c.RECT_TO_T2_RIGHT) * MM,
            c.T2_SIZE_Y * MM, 5.0)
    r.check("rect rear + depth + front   vs  table 2 X",
            (c.RECT_TO_T2_REAR + c.RECT_SIZE_X + c.RECT_TO_T2_FRONT) * MM,
            c.T2_SIZE_X * MM, 5.0)

    print()
    print("=" * 78)
    print("2. Over-determined readings (a residual here is a real disagreement)")
    print("=" * 78)
    r.check("left overhang: derived  vs  measured",
            (c.T2_SIZE_Y - c.T2_OVERHANG_RIGHT - c.T1_SIZE_Y) * MM,
            c.T2_OVERHANG_LEFT * MM, 20.0,
            "right overhang LEADS (user 2026-08-18)")
    r.value("=> WORKCELL_Y_RESIDUAL", c.WORKCELL_Y_RESIDUAL * MM,
            "table 1 shifts by half of this if both were averaged")
    r.check("plate drop: absolute pair  vs  direct reading",
            (0.884 - 0.787) * MM, c.PLATE_TOP_DROP * MM, 2.0,
            "D-058: the direct reading LEADS")

    print()
    print("=" * 78)
    print("3. Derived layout, read outwards from the robot foot at (0,0,0)")
    print("=" * 78)
    r.value("table 1 top   z", c.T1_TOP_Z * MM, "= -(mounting plate thickness)")
    r.value("table 2 top   z", c.T2_TOP_Z * MM)
    r.value("block top     z", c.BLOCK_TOP_Z * MM, "the pocket rim / guide edge")
    r.check("table1 top - table2 top  vs  measured drop",
            (c.T1_TOP_Z - c.T2_TOP_Z) * MM, c.PLATE_TOP_DROP * MM, 0.001)
    r.value("block centre  x (depth)", c.BLOCK_CENTRE_X * MM, "robot foot -> block")
    r.value("block centre  y (lateral)", c.BLOCK_CENTRE_Y * MM)

    print()
    print("=" * 78)
    print("3b. Frame signs (2026-08-19 rotation: +X robot->fixture, +Y robot's left)")
    print("=" * 78)
    # Sign tests, not magnitudes: every check above is a magnitude or a
    # containment, so a mirrored cell would pass all of them. These pin the
    # two facts the rotation hangs on.
    r.require("block is IN FRONT of the robot (+X)",
              c.BLOCK_CENTRE_X > 0.0,
              f"x = {c.BLOCK_CENTRE_X * MM:+.1f} mm (must be > 0)")
    r.require("block is to the robot's RIGHT (-Y)",
              c.BLOCK_CENTRE_Y < 0.0,
              f"y = {c.BLOCK_CENTRE_Y * MM:+.1f} mm (must be < 0; "
              "user call 2026-08-18: fixture to the robot's right)")
    # Regression against the pre-rotation cell: the rotation is x_new = y_old,
    # y_new = -x_old applied to block (0.133, 0.4512, 0.0775).
    r.check("block centre x  vs  pre-rotation y", c.BLOCK_CENTRE_X * MM, 451.2, 0.1)
    r.check("block centre y  vs  -(pre-rotation x)", c.BLOCK_CENTRE_Y * MM, -133.0, 0.1)
    r.check("block top z unchanged by the rotation", c.BLOCK_TOP_Z * MM, 77.5, 0.1)
    # Origin convention 2026-08-19: the asset origin/entrance is the stage-2
    # opening plane, STAGE1_DEPTH (measured in the user's CAD) below the rim.
    r.check("insertion plane = block top - stage 1",
            c.INSERTION_PLANE_Z * MM, 77.5 - 15.0, 0.1)
    r.check("fixture spawns on the insertion plane",
            c.WORKCELL_BLOCK_POS[2] * MM, c.INSERTION_PLANE_Z * MM, 0.001)

    print()
    print("=" * 78)
    print("3c. The pocket INSIDE the block (2026-08-24 CAD placement)")
    print("=" * 78)
    # Until 2026-08-24 the fixture was spawned at the block CENTRE, which was
    # never measured. B12 of the Massblatt gives four edge distances instead,
    # and every one of them was re-derived from the STEP. These are the closing
    # probes for that chain: each runs from one block face, through the pocket,
    # out the other side, and must land on the block dimension.
    # The depth chain is now OVER-DETERMINED, and this is the honest form of
    # it. Two independent readings constrain the same distance: the tape says
    # the raw block is 210 mm, while the B12 gap (80 mm, re-measured on the
    # real block) plus the stage-1 outline (130.0876 mm, STEP point cloud)
    # says 210.0876. The 0.0876 mm is reported, not absorbed.
    #
    # Deliberately NOT checked here any more: "80 + stage 1 + wall == model
    # depth". BLOCK_MODEL_SIZE_X is DEFINED as that sum since the rear wall was
    # re-measured at 15 mm, so the check would compare a number with itself.
    r.check("depth chain: 80 front + 130.088 stage 1 vs the raw block width",
            (c.BLOCK_FRONT_TO_STAGE1 + c.POCKET_STAGE1_SIZE_X) * MM,
            c.BLOCK_SIZE_X * MM, 0.1,
            "OVER-DETERMINED: tape reading against gap + STEP outline")
    r.value("model depth past the raw block", (c.BLOCK_MODEL_SIZE_X - c.BLOCK_SIZE_X) * MM,
            "the 15 mm rear wall reaches past the 210 mm raw block by this much")
    # The rear wall was 10 mm until 2026-08-24 evening, when the user
    # re-measured it at 15 on the real block. The cut-out's OWN rear wall is
    # 9.9124 mm (STEP), so the block must now stand PROUD of the fixture --
    # flush is what produced the two-tone back face the user reported.
    _cutout_rear_wall = c.POCKET_LOCAL_X_RANGE[1] - c.POCKET_STAGE1_X_RANGE[1]
    r.value("the cut-out's own rear wall (STEP)", _cutout_rear_wall * MM,
            "inside the block's 15 mm wall, not equal to it any more")
    r.require("the block stands PROUD of the cut-out at +X",
              c.BLOCK_BEHIND_POCKET_X > 0.0,
              f"{c.BLOCK_BEHIND_POCKET_X * MM:+.4f} mm of block behind the cut-out "
              "(must be > 0; zero would be flush, negative would poke out)")
    r.check("rear wall = the cut-out's own wall + what the block adds",
            (_cutout_rear_wall + c.BLOCK_BEHIND_POCKET_X) * MM,
            c.BLOCK_REAR_WALL_T * MM, 0.001)
    r.check("depth chain, second form: 70 front wall + 150 cut-out + block behind",
            ((c.POCKET_LOCAL_X_RANGE[0] - (c.BLOCK_NEAR_EDGE_X - c.POCKET_ORIGIN_X))
             + c.POCKET_ASSET_BBOX_M[0] + c.BLOCK_BEHIND_POCKET_X) * MM,
            c.BLOCK_MODEL_SIZE_X * MM, 0.001)

    # THE TWO HOLES the first render showed (user, 2026-08-24 evening). The
    # +Y body edge is inset from the bbox except under a middle tab, so a
    # recess cut to the bbox opens two slots. Closure: slot + tab + slot must
    # be the whole recess width, or a strip is still missing.
    _slot_a = c.POCKET_TAB_X_RANGE[0] - c.POCKET_LOCAL_X_RANGE[0]
    _slot_b = c.POCKET_LOCAL_X_RANGE[1] - c.POCKET_TAB_X_RANGE[1]
    _tab = c.POCKET_TAB_X_RANGE[1] - c.POCKET_TAB_X_RANGE[0]
    r.check("+Y slots: 52.0 + tab 67.0 + 31.0 spans the whole recess width",
            (_slot_a + _tab + _slot_b) * MM, c.POCKET_ASSET_BBOX_M[0] * MM, 0.001)
    r.check("+Y slot depth = bbox edge - body edge",
            (c.POCKET_LOCAL_Y_RANGE[1] - c.POCKET_BODY_Y_MAX) * MM, 10.0, 0.001,
            "10 mm, measured off the STEP at both z = -46 and z = +15")
    r.require("the +Y tab really is inset from both recess edges",
              _slot_a > 0.0 and _slot_b > 0.0,
              f"slot -x {_slot_a * MM:.1f} mm, slot +x {_slot_b * MM:.1f} mm "
              "(both must be > 0, and they are the two holes that were open)")
    # CONSTRUCTION CHECK, not evidence, and the tolerance says so: since
    # 2026-08-24 (evening) BLOCK_MINUSY_TO_STAGE1 is SET to the value that
    # closes this sum, so it closes exactly or something else moved without it.
    # The 1.1 mm tolerance it used to carry would now pass for free.
    r.check("lateral chain: 50 (+Y) + 153.0 stage 1 + 57 (-Y) = block width",
            (c.BLOCK_PLUSY_TO_STAGE1 + c.POCKET_STAGE1_SIZE_Y
             + c.BLOCK_MINUSY_TO_STAGE1) * MM,
            c.BLOCK_SIZE_Y * MM, 0.001,
            "closes BY CONSTRUCTION: -Y is fitted, not measured")
    r.require("the Y residual is zero because -Y was fitted, not because it agrees",
              abs(c.WORKCELL_POCKET_Y_RESIDUAL) * MM < 0.001,
              f"{c.WORKCELL_POCKET_Y_RESIDUAL * MM:+.4f} mm -- a non-zero value here means "
              "one of 50 / 153.0 / 260 moved and the modelled -Y wall was not updated")
    # THE DISAGREEMENT ITSELF, which the zero above hides. It did not go away
    # when the 153.0 was re-measured; it moved out of the pocket length and
    # into this wall, and it is still one unmeasured millimetre.
    r.value("modelled -Y wall vs the B12 drawing reading",
            c.WORKCELL_POCKET_MINUSY_VS_B12 * MM,
            "B12 says 58.0, the model uses 57.0; nothing in the task touches "
            "that wall, and a calliper on it is what closes this")

    r.require("pocket origin is IN FRONT of the robot (+X)",
              c.POCKET_ORIGIN_X > 0.0,
              f"x = {c.POCKET_ORIGIN_X * MM:+.1f} mm (must be > 0)")
    r.require("pocket origin is to the robot's RIGHT (-Y)",
              c.POCKET_ORIGIN_Y < 0.0,
              f"y = {c.POCKET_ORIGIN_Y * MM:+.1f} mm (must be < 0)")
    r.check("fixture spawns at the pocket origin, x",
            c.WORKCELL_BLOCK_POS[0] * MM, c.POCKET_ORIGIN_X * MM, 0.001)
    r.check("fixture spawns at the pocket origin, y",
            c.WORKCELL_BLOCK_POS[1] * MM, c.POCKET_ORIGIN_Y * MM, 0.001)
    # The move this whole section exists for. Quoted so a reader of the log can
    # see WHY the home pose had to be recomputed.
    r.value("pocket origin vs the old block centre, x",
            (c.POCKET_ORIGIN_X - c.BLOCK_CENTRE_X) * MM,
            "the fixture moved this far when it stopped being spawned centred")
    r.value("pocket origin vs the old block centre, y",
            (c.POCKET_ORIGIN_Y - c.BLOCK_CENTRE_Y) * MM)

    # A sign slip anywhere above produces a box with a negative edge, which USD
    # authors as an inverted cube without complaining.
    boxes = c.block_recess_boxes()
    worst = min((min(b["size"]), b["name"]) for b in boxes)
    r.require("no recess box has a negative or zero edge",
              worst[0] > 0.0,
              f"smallest edge {worst[0] * MM:.1f} mm, in {worst[1]} "
              f"({len(boxes)} boxes)")
    # The recess must be the cut-out's bounding box EXACTLY -- same material,
    # faces touching, no clearance (user, 2026-08-24).
    r.check("recess opening in X equals the cut-out bbox",
            (c.POCKET_LOCAL_X_RANGE[1] - c.POCKET_LOCAL_X_RANGE[0]) * MM,
            c.POCKET_ASSET_BBOX_M[0] * MM, 0.001)
    r.check("recess opening in Y equals the cut-out bbox",
            (c.POCKET_LOCAL_Y_RANGE[1] - c.POCKET_LOCAL_Y_RANGE[0]) * MM,
            c.POCKET_ASSET_BBOX_M[1] * MM, 0.001)

    print()
    print("=" * 78)
    print("4. Containment (a body outside its parent means a sign error)")
    print("=" * 78)
    block_near = c.BLOCK_NEAR_EDGE_X
    # MODEL width, not raw block width (2026-08-24): the fixture's own 10 mm
    # rear wall stands past the block and has collision, so it is the far face
    # that has to fit inside the rectangle.
    block_far = block_near + c.BLOCK_MODEL_SIZE_X
    rect_near = c.RECT_NEAR_EDGE_X
    rect_far = rect_near + c.RECT_SIZE_X
    slack_far = (rect_far - block_far) * MM
    slack_near = (block_near - rect_near) * MM
    r.require("block sits inside the rectangle in X (depth)",
              slack_far >= 0.0 and slack_near >= 0.0,
              f"slack near {slack_near:+.1f} mm, slack far {slack_far:+.1f} mm "
              f"(both must be >= 0)")
    slack_y = (c.RECT_SIZE_Y - c.BLOCK_SIZE_Y) / 2.0 * MM
    r.require("block sits inside the rectangle in Y (lateral)",
              slack_y >= 0.0,
              f"slack {slack_y:+.1f} mm per side (block assumed centred, A)")

    print()
    print("=" * 78)
    print("5. Reachability sanity (UR5e published reach 850 mm)")
    print("=" * 78)
    # Quoted to the POCKET, not to the block centre (2026-08-24). The block
    # centre is not a point anything spawns at or aims for any more, and it is
    # 49.8 mm nearer the robot -- reporting it would flatter the margin.
    dist = (c.POCKET_ORIGIN_X ** 2 + c.POCKET_ORIGIN_Y ** 2 + c.BLOCK_TOP_Z ** 2) ** 0.5
    r.value("robot foot -> pocket rim, straight line", dist * MM)
    r.require("the pocket is inside the UR5e reach",
              dist * MM <= 850.0,
              f"{dist * MM:.1f} mm of 850 mm published reach "
              f"({850.0 - dist * MM:+.1f} mm margin)")
    # The far end of the fixture, which is what the arm has to clear on the way
    # in. Farther than the pocket by half the stage-1 outline.
    _far = ((c.POCKET_ORIGIN_X + c.POCKET_LOCAL_X_RANGE[1]) ** 2
            + c.POCKET_ORIGIN_Y ** 2
            + (c.BLOCK_TOP_Z + c.BLOCK_REAR_WALL_H) ** 2) ** 0.5
    r.require("the rear wall top is inside the UR5e reach",
              _far * MM <= 850.0,
              f"{_far * MM:.1f} mm of 850 mm ({850.0 - _far * MM:+.1f} mm margin); "
              "the 100 mm rear wall is the farthest thing the arm can hit")
    r.value("block top above the robot foot plane", c.BLOCK_TOP_Z * MM,
            "positive means the pocket rim sits ABOVE the robot base")

    print()
    print("=" * 78)
    print("6. PENDING inputs -- named unknowns that block ABSOLUTE placement")
    print("=" * 78)
    for item in c.WORKCELL_PENDING:
        print(f"  PENDING  {item}")
    if not c.WORKCELL_PENDING:
        print("  none -- every input in the chain is a measurement")

    verdict = "PASS" if r.failed == 0 else f"FAIL ({r.failed} check(s))"
    print()
    # The suffix has to follow the list, not restate a fixed sentence: once the
    # last PENDING entry was resolved the old wording still printed "UNVERIFIED
    # while PENDING is non-empty" under an empty list, which is the kind of
    # stale verdict this whole script exists to prevent.
    if c.WORKCELL_PENDING:
        suffix = (f"-- absolute placement UNVERIFIED, {len(c.WORKCELL_PENDING)} "
                  "input(s) still PENDING")
    else:
        suffix = ("-- no PENDING inputs; every dimension traces to a measurement. "
                  "Still UNVERIFIED in simulation until a run confirms it.")
    print(f"VERDICT: {verdict}   {suffix}")

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            print(f"refusing to overwrite {out}")
            return 1
        out.write_text(json.dumps(
            {"verdict": verdict, "failed": r.failed,
             "pending": list(c.WORKCELL_PENDING), "rows": r.rows},
            indent=2), encoding="utf-8")
        print(f"wrote {out}")

    return 0 if r.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
