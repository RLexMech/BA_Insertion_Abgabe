# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline check of ``autodr.py`` -- the AutoDR boundary state machine."""

from __future__ import annotations

import ast
import concurrent.futures
import csv
import hashlib
import importlib.util
import math
import pathlib
import random
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "source" / "insertion" / "insertion" / "tasks" / "direct" / "insertion"
AUTODR_SRC = PKG_DIR / "autodr.py"
CURRICULUM_SRC = PKG_DIR / "curriculum.py"
ENV_SRC = PKG_DIR / "insertion_env.py"

PKG = "_autodr_check_pkg"

CENTRES = {
    "lat_r": 0.0,
    "yaw": 0.0,
    "tilt": 0.0,
    "start_height": 0.030,
    "friction": 0.10,
}

EXPECTED_KEYS = (
    "lat_r_hi",
    "yaw_lo", "yaw_hi",
    "tilt_hi",
    "start_height_hi",
    "friction_lo", "friction_hi",
)

EXPECTED_TESTWERT = ("lat_r", "yaw", "tilt", "start_height")
MARKED_PROVENANCE_LINES = 5

UNPROVABLE: dict = {}


def _load_torch():
    p = ROOT / "scripts" / "tools" / "torch_shim.py"
    spec = importlib.util.spec_from_file_location("torch_shim", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load(announce=False)[0]


def load_autodr(mutations: tuple = ()) -> dict:
    """Exec ``autodr.py`` (optionally broken) with its relative import working."""
    src = AUTODR_SRC.read_text(encoding="utf-8")
    for old, new in mutations:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"mutation {old!r} matches {n} times in {AUTODR_SRC.name}, expected 1")
        src = src.replace(old, new)
    return load_autodr_src(src)


def load_autodr_src(src: str) -> dict:
    """The exec half of ``load_autodr``, for a caller that read the source itself."""
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules[PKG] = pkg

    cur = types.ModuleType(PKG + ".curriculum")
    exec(compile(CURRICULUM_SRC.read_text(encoding="utf-8"), str(CURRICULUM_SRC), "exec"),
         cur.__dict__)  # noqa: S102 - the point of the file
    sys.modules[PKG + ".curriculum"] = cur

    mod = types.ModuleType(PKG + ".autodr")
    mod.__package__ = PKG
    sys.modules[PKG + ".autodr"] = mod
    exec(compile(src, str(AUTODR_SRC), "exec"), mod.__dict__)  # noqa: S102 - the point of the file
    return mod.__dict__


def env_bin_edges(attr: str) -> tuple:
    """The bin edges of ONE readout table, read out of ``insertion_env.py``."""
    tree = ast.parse(ENV_SRC.read_text(encoding="utf-8"), str(ENV_SRC))
    found = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute) and t.attr == attr
    ]
    if len(found) != 1:
        raise SystemExit(
            f"{ENV_SRC.name}: expected exactly 1 assignment to self.{attr}, "
            f"found {len(found)} -- the bin-edge check has no left-hand side"
        )
    edges = ast.literal_eval(found[0])
    if (not isinstance(edges, tuple) or not edges
            or not all(isinstance(v, float) for v in edges)):
        raise SystemExit(f"{ENV_SRC.name}: self.{attr} is not a tuple of floats: {edges!r}")
    return edges


def magnitude_ceiling(dim) -> float:
    """How far the quantity can get from zero once its boundary is at the top."""
    return max(abs(dim.lo_max), abs(dim.hi_max))


def same_number(a: float, b: float) -> bool:
    """Equal up to unit-conversion rounding."""
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)


def build_checks(ns, torch) -> list:
    out: list = []

    def check(name: str, cond, detail: str = "") -> None:
        out.append((name, bool(cond), detail))

    AutoDR = ns["AutoDR"]
    FixedWidth = ns["FixedWidth"]
    RowTable = ns["RowTable"]
    GridTable = ns["GridTable"]
    AutoDRError = ns["AutoDRError"]
    DR_DIMS = ns["DR_DIMS"]
    DimSpec = ns["DimSpec"]
    TABLE_COLUMNS = ns["TABLE_COLUMNS"]
    COLUMN_INDEX = ns["COLUMN_INDEX"]
    bind_centres = ns["bind_centres"]
    testwert_dims = ns["testwert_dims"]
    map_unit_to_bounds = ns["map_unit_to_bounds"]
    apply_boundary = ns["apply_boundary"]
    boundary_assignment = ns["boundary_assignment"]
    provenance_lines = ns["provenance_lines"]
    DR_DIM_NAMES = ns["DR_DIM_NAMES"]

    DIMS = bind_centres(CENTRES)
    dim_of = {d.name: d for d in DIMS}
    width = len(TABLE_COLUMNS)

    def fresh():
        return AutoDR(DIMS)

    def refuses(name: str, fn) -> None:
        """AutoDRError SPECIFICALLY."""
        try:
            fn()
        except AutoDRError as exc:
            check(name, True, str(exc)[:70])
        except Exception as exc:  # noqa: BLE001 - a crash is not a refusal
            check(name, False, f"raised {type(exc).__name__} instead of AutoDRError")
        else:
            check(name, False, "no error raised")

    def fill(prov, key, n_success, n_total):
        for i in range(n_total):
            prov.record(key, i < n_success)

    p = fresh()
    check("the boundary set has exactly seven entries", p.n_boundaries == 7,
          f"{p.n_boundaries}: {p.keys}")
    check("tilt_lo is NOT a boundary -- the tilt magnitude is one-sided",
          "tilt_lo" not in p.keys, str(p.keys))
    check("there are seven success buffers, one per boundary", len(p.fill()) == 7,
          str(len(p.fill())))
    check("the seven boundaries come in the fixed documented order",
          p.keys == EXPECTED_KEYS, str(p.keys))

    refuses("a missing centre refuses to build the table",
            lambda: bind_centres({k: v for k, v in CENTRES.items() if k != "friction"}))
    refuses("a centre outside the maxima refuses",
            lambda: bind_centres({**CENTRES, "friction": 9.9}))
    _sh_table = next(d for d in DR_DIMS if d.name == "start_height")
    check("bind sets lo_max = centre for one-sided dims",
          all(d.lo_max == d.centre for d in DIMS if d.one_sided)
          and dim_of["start_height"].lo_max == CENTRES["start_height"]
          and _sh_table.lo_max != CENTRES["start_height"],
          f"{[(d.name, d.lo_max, d.centre) for d in DIMS if d.one_sided]}, "
          f"table placeholder {_sh_table.lo_max}")
    refuses("a dimension with n_steps below 1 refuses",
            lambda: DimSpec("lat_r", "m", -1.0, 1.0, False, "probe", n_steps=0).bind(0.0))
    try:
        DimSpec("lat_r", "m", 1.0, -1.0, False, "probe").bind(0.0)
        _inv = "no error raised"
    except AutoDRError as exc:
        _inv = str(exc)
    check("inverted maxima are refused BY NAME, not by the centre-range guard",
          "is above hi_max" in _inv, _inv[:70])
    refuses("a movable boundary with nowhere to go refuses at bind time",
            lambda: DimSpec("start_height", "m", 0.030, 0.050, False, "probe")
            .bind(0.030))
    _zero = ns["BoundEvent"]("lat_r_hi", "expand", 1.0, 0.03, 0.03)
    check("an event whose value did NOT change does not count as moved",
          _zero.moved is False
          and ns["BoundEvent"]("lat_r_hi", "expand", 1.0, 0.0, 0.1).moved is True,
          f"{_zero.before} -> {_zero.after}")

    gen = AutoDR(d for d in DIMS)
    check("a generator of dimensions is materialised, not consumed away",
          gen.n_boundaries == 7 and gen.all_at_max() is False
          and len(gen.bounds()) == 5,
          f"{gen.n_boundaries} boundaries, {len(gen.bounds())} quantities")
    refuses("an empty dimension table refuses", lambda: AutoDR(()))
    refuses("a repeated dimension name refuses -- two keys, ONE buffer",
            lambda: AutoDR((DIMS[0], DIMS[0], DIMS[3])))
    refuses("a dimension that is not a table column refuses",
            lambda: AutoDR((DimSpec("wobble", "m", -1.0, 1.0, False, "probe")
                            .bind(0.0),)))

    refuses("buffer_m below 1 refuses", lambda: AutoDR(DIMS, buffer_m=0))
    refuses("p_boundary outside [0, 1] refuses", lambda: AutoDR(DIMS, p_boundary=7.5))
    refuses("a retreat threshold above the advance threshold refuses",
            lambda: AutoDR(DIMS, advance_at=0.1, retreat_at=0.9))
    refuses("an advance threshold above 1 refuses -- the measure is a rate",
            lambda: AutoDR(DIMS, advance_at=20.0, retreat_at=10.0))
    refuses("a negative retreat threshold refuses",
            lambda: AutoDR(DIMS, advance_at=0.8, retreat_at=-5.0))

    p = fresh()
    b = p.bounds()
    check("a fresh provider starts at width 0 -- lo == hi == centre",
          all(b[d.name][0] == d.centre and b[d.name][1] == d.centre for d in DIMS), str(b))
    check("bounds_version starts at 0", p.bounds_version == 0, str(p.bounds_version))
    check("all_at_max is False at width 0", p.all_at_max() is False)

    m = p.bounds_max()
    check("bounds_max gives the dimension maxima, not the width-0 bounds",
          all(m[d.name] == (d.lo_max, d.hi_max) for d in DIMS), str(m))
    check("bounds_max reports a POSITIVE angle span at width 0, bounds does not",
          m["yaw"][1] - m["yaw"][0] > 0.0 and m["tilt"][1] - m["tilt"][0] > 0.0
          and b["yaw"][1] - b["yaw"][0] == 0.0 and b["tilt"][1] - b["tilt"][0] == 0.0,
          f"max yaw {m['yaw']} tilt {m['tilt']}; now yaw {b['yaw']} tilt {b['tilt']}")
    q = fresh()
    fill(q, "tilt_hi", 240, 240)
    q.update()
    check("a moved boundary stays inside bounds_max, which does not move",
          q.bounds()["tilt"][1] <= q.bounds_max()["tilt"][1]
          and q.bounds_max() == m,
          f"{q.bounds()['tilt']} in {q.bounds_max()['tilt']}")
    check("FixedWidth.at_centre: bounds_max == bounds, zero span everywhere",
          FixedWidth.at_centre(DIMS).bounds_max() == FixedWidth.at_centre(DIMS).bounds()
          and all(v[1] - v[0] == 0.0
                  for v in FixedWidth.at_centre(DIMS).bounds_max().values()),
          str(FixedWidth.at_centre(DIMS).bounds_max()))
    _base = ns["_Provider"]
    _inherits = False
    try:
        _base.bounds_max(FixedWidth.at_centre(DIMS))
        _inherits = True
    except NotImplementedError:
        pass
    check("_Provider.bounds_max is abstract -- no provider inherits an answer",
          not _inherits, f"base returned an answer: {_inherits}")
    check("FixedWidth.at_max: bounds_max == bounds == the maxima",
          FixedWidth.at_max(DIMS).bounds_max() == FixedWidth.at_max(DIMS).bounds()
          and all(FixedWidth.at_max(DIMS).bounds_max()[d.name][1] == d.hi_max
                  for d in DIMS),
          str(FixedWidth.at_max(DIMS).bounds_max()))

    p = fresh()
    fill(p, "lat_r_hi", 239, 239)
    ev = p.update()
    check("239 flags change nothing -- the buffer must be full",
          ev == [] and p.value("lat_r_hi") == 0.0 and p.fill()["lat_r_hi"] == 239,
          f"events {len(ev)}, fill {p.fill()['lat_r_hi']}")
    refuses("record refuses an unknown boundary", lambda: p.record("tilt_lo", True))
    refuses("value refuses the same unknown boundary, the same way",
            lambda: p.value("tilt_lo"))

    p = fresh()
    fill(p, "lat_r_hi", 240, 240)
    ev = p.update()
    delta = dim_of["lat_r"].delta("hi")
    check("240 flags at 1.00 expand the boundary by exactly one delta",
          len(ev) == 1 and ev[0].action == "expand"
          and abs(p.value("lat_r_hi") - delta) < 1e-15,
          f"{[e.action for e in ev]} -> {p.value('lat_r_hi'):.6g} vs {delta:.6g}")
    check("a processed buffer is emptied", p.fill()["lat_r_hi"] == 0,
          str(p.fill()["lat_r_hi"]))
    _ns = dim_of["lat_r"].n_steps
    check("the delta is (maximum - centre) / n_steps",
          abs(delta - (dim_of["lat_r"].hi_max - 0.0) / float(_ns)) < 1e-18,
          f"{delta:.6g} at n_steps {_ns}")

    p = fresh()
    fill(p, "lat_r_hi", 500, 1000)
    check("flags arriving at a full buffer are dropped AND counted",
          p.fill()["lat_r_hi"] == 240 and p.dropped()["lat_r_hi"] == 760
          and p.scalars().get("dr/flags_dropped") == 760.0,
          f"fill {p.fill()['lat_r_hi']}, dropped {p.dropped()['lat_r_hi']}")

    p = fresh()
    fill(p, "lat_r_hi", 192, 240)
    ev = p.update()
    check("exactly 0.80 EXPANDS -- Akkaya uses >=, unlike curriculum.Ladder",
          len(ev) == 1 and ev[0].action == "expand", f"{[e.action for e in ev]}")

    p = fresh()
    fill(p, "lat_r_hi", 240, 240)
    p.update()
    fill(p, "lat_r_hi", 24, 240)
    ev = p.update()
    check("exactly 0.10 SHRINKS -- Akkaya uses <=",
          len(ev) == 1 and ev[0].action == "shrink" and p.value("lat_r_hi") == 0.0,
          f"{[e.action for e in ev]} -> {p.value('lat_r_hi')}")

    p = fresh()
    fill(p, "lat_r_hi", 120, 240)
    ev = p.update()
    check("0.50 holds, writes a hold event that moves nothing, and STILL empties the buffer",
          len(ev) == 1 and ev[0].action == "hold" and ev[0].moved is False
          and ev[0].line().startswith("[autodr] lat_r_hi: rate 0.5000 over 240 -> hold")
          and p.value("lat_r_hi") == 0.0 and p.fill()["lat_r_hi"] == 0
          and p.bounds_version == 0,
          f"events {[e.action for e in ev]}, fill {p.fill()['lat_r_hi']}, "
          f"version {p.bounds_version}")

    p = fresh()
    fill(p, "friction_hi", 240, 240)
    fill(p, "yaw_lo", 240, 240)
    ev = p.update()
    check("two full buffers fire in ONE update, in key order, version +2",
          [e.key for e in ev] == ["yaw_lo", "friction_hi"] and p.bounds_version == 2,
          f"{[e.key for e in ev]}, version {p.bounds_version}")

    p = fresh()
    for _ in range(_ns):
        fill(p, "lat_r_hi", 240, 240)
        p.update()
    check("n_steps expansions land on the maximum EXACTLY (no float drift)",
          p.value("lat_r_hi") == dim_of["lat_r"].hi_max,
          f"{p.value('lat_r_hi'):.17g} vs {dim_of['lat_r'].hi_max:.17g}")
    fill(p, "lat_r_hi", 240, 240)
    ev = p.update()
    check("one expansion past n_steps clamps and reports clamped_max",
          len(ev) == 1 and ev[0].action == "clamped_max"
          and p.value("lat_r_hi") == dim_of["lat_r"].hi_max,
          f"{[e.action for e in ev]}")

    p = fresh()
    fill(p, "lat_r_hi", 0, 240)
    ev = p.update()
    check("a retreat at width 0 clamps and reports clamped_zero",
          len(ev) == 1 and ev[0].action == "clamped_zero" and p.value("lat_r_hi") == 0.0,
          f"{[e.action for e in ev]}")

    p = fresh()
    for _ in range(_ns):
        fill(p, "friction_lo", 240, 240)
        p.update()
    check("a lower boundary walks DOWN to lo_max and lands on it exactly",
          p.value("friction_lo") == dim_of["friction"].lo_max,
          f"{p.value('friction_lo'):.17g} vs {dim_of['friction'].lo_max:.17g}")

    p = fresh()
    fill(p, "yaw_hi", 240, 240)
    p.update()
    check("a real move bumps bounds_version by exactly 1", p.bounds_version == 1,
          str(p.bounds_version))
    fill(p, "yaw_hi", 120, 240)
    p.update()
    check("a hold does not bump bounds_version", p.bounds_version == 1,
          str(p.bounds_version))
    p = fresh()
    for _ in range(_ns):
        fill(p, "yaw_hi", 240, 240)
        p.update()
    v10 = p.bounds_version
    fill(p, "yaw_hi", 240, 240)
    p.update()
    check("clamped_max does NOT bump bounds_version -- the fresh window survives",
          p.bounds_version == v10, f"{p.bounds_version} vs {v10}")
    p = fresh()
    fill(p, "yaw_hi", 0, 240)
    p.update()
    check("clamped_zero does NOT bump bounds_version", p.bounds_version == 0,
          str(p.bounds_version))

    p = fresh()
    for key in p.keys[:-1]:
        for _ in range(_ns):
            fill(p, key, 240, 240)
            p.update()
    check("all_at_max stays False while one boundary is short", p.all_at_max() is False)
    for _ in range(_ns):
        fill(p, p.keys[-1], 240, 240)
        p.update()
    check("all_at_max turns True only when EVERY boundary is at its maximum",
          p.all_at_max() is True, str(p.as_dict()["steps"]))

    p = fresh()
    fill(p, "tilt_hi", 240, 240)
    p.update()
    fill(p, "friction_lo", 100, 100)
    st = p.state_dict()
    q = fresh()
    q.load_state_dict(st)
    check("state_dict round-trips steps, buffers and bounds_version",
          q.as_dict() == p.as_dict() and q.state_dict() == st,
          f"version {q.bounds_version} vs {p.bounds_version}")

    with tempfile.TemporaryDirectory() as td:
        jf = pathlib.Path(td) / "autodr_42.json"
        p.save(jf)
        r = fresh()
        r.load(jf)
        check("save/load round-trips through the autodr_<it>.json file itself",
              r.state_dict() == st, f"{jf.name}, {jf.stat().st_size} bytes")
        junk_json = pathlib.Path(td) / "broken.json"
        junk_json.write_text("{not json", encoding="utf-8")
        refuses("a malformed autodr_<it>.json refuses BY NAME, not as a JSON error",
                lambda: fresh().load(junk_json))
        refuses("an unreadable autodr state path refuses BY NAME",
                lambda: fresh().load(pathlib.Path(td) / "does_not_exist.json"))

    bad = dict(st)
    bad["keys"] = list(st["keys"])[:-1]
    refuses("a resume with a different boundary set refuses",
            lambda: fresh().load_state_dict(bad))
    moved = dict(st)
    moved["dims"] = [dict(d) for d in st["dims"]]
    moved["dims"][0]["hi_max"] = st["dims"][0]["hi_max"] * 2.0
    refuses("a resume with a changed maximum refuses",
            lambda: fresh().load_state_dict(moved))
    over = dict(st)
    over["steps"] = dict(st["steps"])
    over["steps"]["lat_r_hi"] = 999
    refuses("a resume with a step count past the maximum refuses",
            lambda: fresh().load_state_dict(over))
    neg = dict(st)
    neg["steps"] = dict(st["steps"])
    neg["steps"]["lat_r_hi"] = -7
    refuses("a resume with a NEGATIVE step count refuses",
            lambda: fresh().load_state_dict(neg))
    impossible = dict(st)
    impossible["buf_n"] = dict(st["buf_n"])
    impossible["buf_n"]["lat_r_hi"] = 1_000_000
    refuses("a resume with more flags than the buffer holds refuses",
            lambda: fresh().load_state_dict(impossible))
    zero_m = dict(fresh().state_dict())
    zero_m["buffer_m"] = 0
    refuses("a resume re-runs the parameter contract (buffer_m 0)",
            lambda: fresh().load_state_dict(zero_m))
    wide_thresh = dict(fresh().state_dict())
    wide_thresh["advance_at"] = 100.0
    wide_thresh["retreat_at"] = -5.0
    refuses("a resume with thresholds outside [0, 1] refuses",
            lambda: fresh().load_state_dict(wide_thresh))
    _missing_ok = True
    _missing_detail = ""
    for _k in ("keys", "steps", "buf_n", "buf_s", "bounds_version",
               "buffer_m", "p_boundary", "advance_at", "retreat_at", "dims"):
        _trunc = {k: v for k, v in st.items() if k != _k}
        try:
            fresh().load_state_dict(_trunc)
            _missing_ok = False
            _missing_detail = f"{_k}: no error"
        except AutoDRError:
            pass
        except Exception as exc:  # noqa: BLE001
            _missing_ok = False
            _missing_detail = f"{_k}: {type(exc).__name__}"
    check("EVERY missing key in a resume state refuses with AutoDRError",
          _missing_ok, _missing_detail or "all ten refuse by name")
    refuses("a resume state that is not a dict refuses",
            lambda: fresh().load_state_dict([1, 2, 3]))
    listed = dict(st)
    listed["steps"] = list(st["steps"].keys())
    refuses("a resume whose steps block is a LIST refuses",
            lambda: fresh().load_state_dict(listed))
    frac_steps = dict(st)
    frac_steps["steps"] = dict(st["steps"])
    frac_steps["steps"]["lat_r_hi"] = 3.9
    refuses("a resume with a FRACTIONAL step count refuses, never truncates",
            lambda: fresh().load_state_dict(frac_steps))
    str_m = dict(st)
    str_m["buffer_m"] = "240"
    refuses("a resume with a STRING where a number belongs refuses",
            lambda: fresh().load_state_dict(str_m))
    neg_drop = dict(st)
    neg_drop["dropped"] = {k: -99 for k in st["keys"]}
    refuses("a resume with a negative dropped count refuses",
            lambda: fresh().load_state_dict(neg_drop))
    _dimless = dict(st)
    _dimless["dims"] = [dict(d) for d in st["dims"]]
    _dimless["dims"][0].pop("n_steps")
    refuses("a resume whose dims entry is missing a field refuses BY NAME",
            lambda: fresh().load_state_dict(_dimless))
    _dimjunk = dict(st)
    _dimjunk["dims"] = [1, 2, 3]
    refuses("a resume whose dims block holds non-dicts refuses",
            lambda: fresh().load_state_dict(_dimjunk))
    _nodrop = {k: v for k, v in st.items() if k != "dropped"}
    refuses("a resume with no dropped block refuses -- the curve is evidence",
            lambda: fresh().load_state_dict(_nodrop))
    _extra = dict(st)
    _extra["steps"] = {**st["steps"], "wobble_hi": 1}
    refuses("a resume naming a boundary this table does not have refuses",
            lambda: fresh().load_state_dict(_extra))

    check("every quantity but friction carries [TESTWERT]",
          set(testwert_dims(DR_DIMS)) == {d.name for d in DR_DIMS} - {"friction"},
          str(testwert_dims(DR_DIMS)))
    prov = provenance_lines(DR_DIMS)
    _marked = sum(1 for line in prov if "[TESTWERT]" in line)
    check("the provenance report names every quantity and both constants",
          len(prov) == len(DR_DIMS) + 2
          and all(any(d.name in line for line in prov) for d in DR_DIMS)
          and _marked == MARKED_PROVENANCE_LINES
          and len(testwert_dims(DR_DIMS)) + 1 == MARKED_PROVENANCE_LINES,
          f"{len(prov)} lines, {_marked} marked, "
          f"{len(testwert_dims(DR_DIMS)) + 1} derived, anchor {MARKED_PROVENANCE_LINES}")
    _fric_line = [line for line in prov if line.startswith("friction ")]
    check("the friction line is reported as sourced, not as a [TESTWERT]",
          len(_fric_line) == 1 and "(sourced)" in _fric_line[0]
          and "[TESTWERT]" not in _fric_line[0],
          _fric_line[0][:70] if _fric_line else "no friction line")
    check("AutoDR.as_dict publishes the decided-not-derived names",
          tuple(fresh().as_dict()["testwert"]) == EXPECTED_TESTWERT,
          str(fresh().as_dict()["testwert"]))
    check("FixedWidth.as_dict publishes the decided-not-derived names",
          tuple(FixedWidth.at_centre(DIMS).as_dict()["testwert"]) == EXPECTED_TESTWERT,
          str(FixedWidth.at_centre(DIMS).as_dict()["testwert"]))

    _lat_edges = env_bin_edges("_lateral_bin_edges_mm")
    _lat_ceil_mm = magnitude_ceiling(next(d for d in DR_DIMS if d.name == "lat_r")) * 1000.0
    check("the last LATERAL bin edge is the lat_r ceiling",
          same_number(_lat_edges[-1], _lat_ceil_mm),
          f"edge {_lat_edges[-1]:.10g} mm vs ceiling {_lat_ceil_mm:.10g} mm")
    _yaw_edges = env_bin_edges("_yaw_bin_edges_deg")
    _yaw_ceil_deg = math.degrees(magnitude_ceiling(next(d for d in DR_DIMS if d.name == "yaw")))
    check("the last YAW bin edge is the yaw magnitude ceiling",
          same_number(_yaw_edges[-1], _yaw_ceil_deg),
          f"edge {_yaw_edges[-1]:.10g} deg vs ceiling {_yaw_ceil_deg:.10g} deg")
    _tilt_edges = env_bin_edges("_tilt_bin_edges_deg")
    _tilt_ceil_deg = math.degrees(magnitude_ceiling(next(d for d in DR_DIMS if d.name == "tilt")))
    check("the last TILT bin edge is the tilt magnitude ceiling",
          same_number(_tilt_edges[-1], _tilt_ceil_deg),
          f"edge {_tilt_edges[-1]:.10g} deg vs ceiling {_tilt_ceil_deg:.10g} deg")
    _h_edges = env_bin_edges("_start_height_bin_edges_m")
    _h_ceil_m = next(d for d in DR_DIMS if d.name == "start_height").hi_max
    check("the last HEIGHT bin edge is the start_height ceiling",
          same_number(_h_edges[-1], _h_ceil_m),
          f"edge {_h_edges[-1]:.10g} m vs ceiling {_h_ceil_m:.10g} m")

    nod = FixedWidth.at_centre(DIMS)
    check("No-DR keeps every quantity at its centre",
          all(v == (d.centre, d.centre) for d, v in zip(DIMS, nod.bounds().values())),
          str(nod.bounds()))
    zeros = torch.zeros(64) + 0.5
    is_b, _ = nod.boundary_assignment(zeros, zeros)
    check("No-DR never nails a boundary env, for any draw",
          not any(bool(v) for v in is_b), str(sum(1 for v in is_b if v)))
    nod.record("lat_r_hi", True)
    check("FixedWidth swallows a record and never emits an event",
          nod.update() == [] and nod.bounds_version == 0
          and nod.as_dict()["fill"]["lat_r_hi"] == 0)
    ev_prov = FixedWidth.at_max(DIMS)
    check("the eval provider opens every quantity to its maximum",
          all(ev_prov.bounds()[d.name] == (d.lo_max, d.hi_max) for d in DIMS),
          str(ev_prov.bounds()))
    fw_state = ev_prov.state_dict()
    fw2 = FixedWidth.at_centre(DIMS)
    fw2.load_state_dict(fw_state)
    check("a FixedWidth state round-trips its width",
          fw2.bounds() == ev_prov.bounds(), str(fw2.bounds()))
    wide = [dict(d) for d in fw_state["dims"]]
    wide[0]["hi_max"] = wide[0]["hi_max"] * 3.0
    foreign = {**fw_state, "dims": wide}
    refuses("a FixedWidth state from a DIFFERENT table refuses",
            lambda: FixedWidth.at_centre(DIMS).load_state_dict(foreign))
    check("all_at_max is False for No-DR and True for the eval provider",
          nod.all_at_max() is False and ev_prov.all_at_max() is True,
          f"{nod.all_at_max()} / {ev_prov.all_at_max()}")
    check("both providers publish one and the same scalar key set",
          set(nod.scalars()) == set(fresh().scalars())
          and nod.scalars()["dr/flags_dropped"] == 0.0,
          str(sorted(set(fresh().scalars()) ^ set(nod.scalars()))))

    check("the evaluation table has eighteen columns", width == 18, str(width))
    check("the tilt AZIMUTH owns a column but is not an AutoDR quantity",
          "tilt_azimuth" in TABLE_COLUMNS
          and "tilt_azimuth" not in {d.name for d in DIMS})
    _grasp_col = COLUMN_INDEX.get("grasp_obs_x")
    check("the grasp belief error owns a table column and nails NO boundary",
          _grasp_col is not None
          and "grasp_obs_x" not in DR_DIM_NAMES
          and _grasp_col not in p.boundary_columns,
          f"column {_grasp_col}, boundary columns {p.boundary_columns}")
    _pyaw_col = COLUMN_INDEX.get("pocket_obs_yaw")
    check("the pocket-yaw belief error owns a table column and nails NO boundary",
          _pyaw_col is not None
          and "pocket_obs_yaw" not in DR_DIM_NAMES
          and _pyaw_col not in p.boundary_columns,
          f"column {_pyaw_col}, boundary columns {p.boundary_columns}")
    _fz_col = COLUMN_INDEX.get("fixture_noise_z")
    check("the fixture z spread owns the LAST table column and nails NO boundary",
          _fz_col == len(TABLE_COLUMNS) - 1
          and "fixture_noise_z" not in DR_DIM_NAMES
          and _fz_col not in p.boundary_columns,
          f"column {_fz_col}, boundary columns {p.boundary_columns}")
    cols = {d.name: d.column for d in DIMS}
    check("each quantity maps to its OWN table column",
          len(set(cols.values())) == 5 and cols["tilt"] == TABLE_COLUMNS.index("tilt"),
          str(cols))
    i = p.keys.index("tilt_hi")
    check("boundary_columns and boundary_sides line up with the keys",
          p.boundary_columns[i] == TABLE_COLUMNS.index("tilt")
          and p.boundary_sides[i] == 1.0
          and p.boundary_sides[p.keys.index("friction_lo")] == 0.0,
          f"col {p.boundary_columns[i]}, side {p.boundary_sides[i]}")
    lo_arr, hi_arr = FixedWidth.at_max(DIMS).bounds_arrays()
    az = TABLE_COLUMNS.index("fixture_noise_x")
    check("bounds_arrays fills the DR columns and passes the rest through as (0, 1)",
          len(lo_arr) == width and len(hi_arr) == width
          and lo_arr[TABLE_COLUMNS.index("friction")] == dim_of["friction"].lo_max
          and (lo_arr[az], hi_arr[az]) == (0.0, 1.0),
          f"fixture_noise_x ({lo_arr[az]}, {hi_arr[az]})")
    a_lo, a_hi = fresh().bounds_arrays()
    check("AutoDR bounds_arrays reports width 0 on the DR columns at the start",
          a_lo[TABLE_COLUMNS.index("friction")] == CENTRES["friction"]
          and a_hi[TABLE_COLUMNS.index("friction")] == CENTRES["friction"]
          and (a_lo[az], a_hi[az]) == (0.0, 1.0),
          f"friction ({a_lo[TABLE_COLUMNS.index('friction')]}, "
          f"{a_hi[TABLE_COLUMNS.index('friction')]})")

    rows = [[(r * width + c) % 7 / 7.0 for c in range(width)] for r in range(5)]
    t = RowTable(rows, "deadbeef")
    check("take() hands out row ids in table order and wraps",
          t.take(3) == [0, 1, 2] and t.take(4) == [3, 4, 0, 1], "")
    check("served counts every row handed out and never wraps",
          t.served == 7 and t.cursor == 2, f"served {t.served}, cursor {t.cursor}")
    t.reset()
    check("reset rewinds BOTH the cursor and the served count",
          t.served == 0 and t.cursor == 0 and t.take(2) == [0, 1],
          f"served {t.served}, cursor {t.cursor}")
    refuses("a boolean cell refuses -- True would scale to the upper edge",
            lambda: RowTable([[True] * width], "x"))
    _mine = [[0.5] * width]
    _t2 = RowTable(_mine, "x")
    _mine[0][0] = 99.0
    check("the table is COPIED, so a later write cannot walk back validation",
          _t2.rows[0][0] == 0.5, str(_t2.rows[0][0]))
    _one = RowTable([[0.25] * width], "x")
    check("a wrapped take REPEATS a row id -- served is what says so",
          _one.take(3) == [0, 0, 0] and _one.served == 3 and _one.n_rows == 1,
          f"served {_one.served} of {_one.n_rows}")
    refuses("a row value outside [0, 1] refuses",
            lambda: RowTable([[1.5] * width], "x"))
    refuses("a row of the wrong width refuses",
            lambda: RowTable([[0.5] * (width - 1)], "x"))
    try:
        _ood = RowTable([[2.25, -0.25] + [1.5] * (width - 2)], "x", allow_outside=True)
        _ood_ok = _ood.rows[0][:3] == [2.25, -0.25, 1.5] and _ood.allow_outside is True
    except AutoDRError:
        _ood_ok = False
    check("an OOD table (allow_outside) keeps values beyond [0, 1]", _ood_ok)
    refuses("allow_outside still refuses a value that is not finite",
            lambda: RowTable([[float("nan")] * width], "x", allow_outside=True))

    with tempfile.TemporaryDirectory() as td:
        good = pathlib.Path(td) / "t.csv"
        good.write_text(
            ",".join(TABLE_COLUMNS) + "\n"
            + "\n".join(",".join(f"{v:.6f}" for v in r) for r in rows),
            encoding="utf-8",
        )
        ood_csv = pathlib.Path(td) / "o.csv"
        ood_csv.write_text(",".join(TABLE_COLUMNS) + "\n" + ",".join(["1.5"] * width) + "\n",
                           encoding="utf-8")
        try:
            _oo = RowTable.from_csv(ood_csv, allow_outside=True).rows[0][0] == 1.5
        except AutoDRError:
            _oo = False
        check("from_csv hands allow_outside through to the table", _oo)
        loaded = RowTable.from_csv(good)
        check("from_csv reads the rows and hashes the FILE",
              loaded.n_rows == 5
              and loaded.sha256 == hashlib.sha256(good.read_bytes()).hexdigest(),
              loaded.sha256[:12])
        bad_csv = pathlib.Path(td) / "b.csv"
        bad_csv.write_text("a,b\n0.1,0.2\n", encoding="utf-8")
        try:
            RowTable.from_csv(bad_csv)
            _hdr = "no error raised"
        except AutoDRError as exc:
            _hdr = str(exc)
        check("a header that is not TABLE_COLUMNS is refused BY NAME",
              "header does not match TABLE_COLUMNS" in _hdr, _hdr[:70])
        junk = pathlib.Path(td) / "j.csv"
        junk.write_text(",".join(TABLE_COLUMNS) + "\n" + ",".join(["abc"] * width) + "\n",
                        encoding="utf-8")
        refuses("a cell that is not a number refuses with AutoDRError, not ValueError",
                lambda: RowTable.from_csv(junk))

    grid_rows = [[0.5] * width for _ in range(6)]
    g = GridTable(grid_rows, "x", ("lat_r", "tilt"), cells=3, reps=2)
    check("a grid row belongs to cell id // reps, repetition id % reps",
          [g.cell_of(i) for i in range(6)] == [0, 0, 1, 1, 2, 2]
          and [g.rep_of(i) for i in range(6)] == [0, 1, 0, 1, 0, 1])
    refuses("a grid table whose cells * reps misses the row count refuses",
            lambda: GridTable(grid_rows, "x", ("lat_r", "tilt"), cells=3, reps=3))
    refuses("a grid pair that is not exactly two quantities refuses",
            lambda: GridTable(grid_rows, "x", ("lat_r", "tilt", "yaw"), cells=3, reps=2))
    refuses("a grid pair that repeats one quantity refuses",
            lambda: GridTable(grid_rows, "x", ("lat_r", "lat_r"), cells=3, reps=2))
    refuses("a grid pair naming a non-randomised column refuses",
            lambda: GridTable(grid_rows, "x", ("lat_r", "tilt_azimuth"),
                              cells=3, reps=2))
    refuses("a grid pair naming the lateral ANGLE refuses -- lat_phi is a column, not a quantity",
            lambda: GridTable(grid_rows, "x", ("lat_r", "lat_phi"),
                              cells=3, reps=2))
    refuses("a grid table with negative cells or reps refuses",
            lambda: GridTable(grid_rows, "x", ("lat_r", "tilt"), cells=-2, reps=-3))

    lo, hi = 0.2, 0.6
    check("map(0) == lo EXACTLY", map_unit_to_bounds(0.0, lo, hi) == lo,
          repr(map_unit_to_bounds(0.0, lo, hi)))
    check("map(1) == hi EXACTLY", map_unit_to_bounds(1.0, lo, hi) == hi,
          repr(map_unit_to_bounds(1.0, lo, hi)))

    unit = torch.ones(4, width) * 0.5
    r_idx = torch.tensor([0, 2]).long()
    c_idx = torch.tensor([3, 6]).long()
    apply_boundary(unit, r_idx, c_idx, torch.tensor([1.0, 0.0]))
    check("apply_boundary nails 1.0 for an upper edge and 0.0 for a lower edge",
          float(unit[0, 3]) == 1.0 and float(unit[2, 6]) == 0.0
          and float(unit[1, 3]) == 0.5,
          f"{float(unit[0, 3])}, {float(unit[2, 6])}, {float(unit[1, 3])}")

    with open(ROOT / "scripts" / "eval" / "eval_table_seed20_c18.csv", newline="",
              encoding="utf-8") as _fh:
        _eval_rows = [[float(v) for v in r] for r in list(csv.reader(_fh))[1:]]
    _w = len(_eval_rows[0])
    _tbl = torch.tensor(_eval_rows)
    _part = torch.tensor([0, 7, 999]).long()
    _got = ns["table_rows"](_tbl, _part)
    _reset_unit = torch.zeros(len(_eval_rows), _w)
    _all = torch.arange(len(_eval_rows)).long()
    _reset_unit[_all] = ns["table_rows"](_tbl, _all)
    check("table_rows: env i plays row i, for a partial reset and for the full first reset",
          tuple(_tbl.shape) == (1000, _w) and tuple(_got.shape) == (3, _w)
          and all(abs(float(_got[j, c]) - _eval_rows[i][c]) < 1e-6
                  for j, i in enumerate((0, 7, 999)) for c in range(_w))
          and all(abs(float(_reset_unit[i, c]) - _eval_rows[i][c]) < 1e-6
                  for i in (0, 1, 500, 999) for c in (0, 5, 15)),
          f"table {tuple(_tbl.shape)}, partial {tuple(_got.shape)}")

    n = 10_000
    shares = []
    for seed in (7, 11, 13):
        rng = random.Random(seed)
        ra = torch.tensor([rng.random() for _ in range(n)])
        rb = torch.tensor([rng.random() for _ in range(n)])
        is_b, idx = boundary_assignment(ra, rb, p.n_boundaries, p.p_boundary)
        shares.append(float(sum(1 for v in is_b if v)) / n)
    check("p_b over 10^4 draws lands in [0.47, 0.53], for three seeds",
          all(0.47 <= sh <= 0.53 for sh in shares),
          ", ".join(f"{sh:.4f}" for sh in shares))
    counts = [0] * p.n_boundaries
    for v in idx:
        counts[int(v)] += 1
    expect = n / p.n_boundaries
    check("the boundary index is uniform over all seven boundaries",
          all(0.85 * expect <= c <= 1.15 * expect for c in counts), str(counts))
    edge = torch.tensor([0.0, 0.999999, 1.0])
    _, idx_edge = boundary_assignment(edge, edge, p.n_boundaries, p.p_boundary)
    check("the boundary index never leaves [0, n-1], not even at rand == 1.0",
          [int(v) for v in idx_edge] == [0, p.n_boundaries - 1, p.n_boundaries - 1],
          str([int(v) for v in idx_edge]))
    refuses("boundary_assignment refuses a non-positive boundary count",
            lambda: boundary_assignment(edge, edge, 0, 0.5))
    refuses("boundary_assignment refuses a probability outside [0, 1]",
            lambda: boundary_assignment(edge, edge, 7, 5.0))

    chain = fresh()
    for key in chain.keys:
        for _ in range(4):
            fill(chain, key, 240, 240)
            chain.update()
    lo_c, hi_c = chain.bounds_arrays()
    n_b = chain.n_boundaries
    exact = True
    detail = ""
    for bi in range(n_b):
        u = torch.ones(1, width) * 0.5
        col = chain.boundary_columns[bi]
        side = chain.boundary_sides[bi]
        apply_boundary(u, torch.tensor([0]).long(), torch.tensor([col]).long(),
                       torch.tensor([side]))
        got = map_unit_to_bounds(float(u[0, col]), lo_c[col], hi_c[col])
        want = chain.value(chain.keys[bi])
        if got != want:
            exact = False
            detail = f"{chain.keys[bi]}: {got!r} != {want!r}"
    check("the nailed value equals the boundary EXACTLY, for all seven",
          exact, detail or f"{n_b}/{n_b} exact at 4 steps of width")

    p = fresh()
    fill(p, "tilt_hi", 240, 240)
    p.update()
    sc = p.scalars()
    bound_keys = [k for k in sc if k.startswith("dr/") and k[3:] in EXPECTED_KEYS]
    fill_keys = [k for k in sc if k.endswith("_fill")]
    rate_keys = [k for k in sc if k.endswith("_last_rate")]
    hold_keys = [k for k in sc if k.endswith("_holds")]
    check("scalars() publishes 7 bounds, 7 fill levels, 7 last rates, 7 hold counts and three aggregates",
          len(bound_keys) == 7 and len(fill_keys) == 7
          and len(rate_keys) == 7 and len(hold_keys) == 7
          and {"dr/bounds_version", "dr/all_at_max", "dr/flags_dropped"} <= set(sc)
          and len(sc) == 31
          and sc["dr/tilt_hi_last_rate"] == 1.0 and sc["dr/tilt_hi_holds"] == 0.0
          and sc["dr/yaw_lo_last_rate"] == -1.0,
          f"{len(bound_keys)} bounds, {len(fill_keys)} fills, {len(rate_keys)} rates, "
          f"{len(hold_keys)} holds, {len(sc)} total")
    ev = ns["BoundEvent"]("tilt_hi", "expand", 1.0, 0.0, 0.014)
    check("a bound event writes a marked [autodr] line",
          ev.line().startswith("[autodr] tilt_hi:"), ev.line())

    FloorSpec = ns["FloorSpec"]
    FLOOR_KEY = ns["FLOOR_KEY"]
    H_MIN = CENTRES["start_height"]
    F0 = -0.030
    FLOOR = FloorSpec(F0, H_MIN, 5)

    def floored():
        return AutoDR(DIMS, floor=FLOOR)

    def attempt(fn):
        """(value, None) or (None, exc) -- a refusal inside a check must be a FAIL of that check, not a crash of the whole run."""
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001
            return None, exc

    f = floored()
    check("with a floor the boundary set has eight entries and the floor comes LAST",
          f.n_boundaries == 8 and f.keys[-1] == FLOOR_KEY and f.keys[:7] == EXPECTED_KEYS,
          str(f.keys))
    check("the floor key resolves to the start_height column with side 0.0 -- no geometry change",
          f.boundary_columns[-1] == COLUMN_INDEX["start_height"]
          and f.boundary_sides[-1] == 0.0,
          f"col {f.boundary_columns[-1]}, side {f.boundary_sides[-1]}")
    check("the floor starts at f0, ends EXACTLY on H_min, and phase is floor at step 0",
          f.value(FLOOR_KEY) == F0 and FLOOR.value_at(5) == H_MIN and f.phase == "floor",
          f"{f.value(FLOOR_KEY)}, {FLOOR.value_at(5)}, {f.phase}")
    check("map(0) on the floor band IS the floor, exactly",
          map_unit_to_bounds(0.0, F0, H_MIN) == F0, repr(map_unit_to_bounds(0.0, F0, H_MIN)))
    for _ in range(10):
        for key in f.keys[:7]:
            fill(f, key, 240, 240)
        f.update()
    b = f.bounds()
    check("in phase floor every other quantity is its centre and start_height is [floor, H_min], "
          "even with the seven at their maxima",
          all(b[nm] == (dim_of[nm].centre, dim_of[nm].centre)
              for nm in DR_DIM_NAMES if nm != "start_height")
          and b["start_height"] == (F0, H_MIN),
          str(b))
    check("all_at_max is False in phase floor even with the seven at their maxima",
          f.all_at_max() is False, str(f.as_dict()["steps"]))
    rng = random.Random(3)
    ra = torch.tensor([rng.random() for _ in range(4000)])
    rb = torch.tensor([rng.random() for _ in range(4000)])
    is_b, idx = f.boundary_assignment(ra, rb)
    nailed = [int(v) for v, ok in zip(idx, is_b) if ok]
    check("in phase floor every boundary env is nailed to the floor (index 7)",
          nailed and all(v == 7 for v in nailed) and 0.45 <= len(nailed) / 4000 <= 0.55,
          f"{len(nailed)} boundary envs, indices {sorted(set(nailed))}")

    g = floored()
    fill(g, FLOOR_KEY, 240, 240)
    ev1 = g.update()
    up = g.value(FLOOR_KEY)
    fill(g, FLOOR_KEY, 0, 240)
    ev2 = g.update()
    down = g.value(FLOOR_KEY)
    fill(g, FLOOR_KEY, 0, 240)
    ev3 = g.update()
    check("a full floor buffer at rate 1.0 moves the floor UP one rung (version +1); "
          "at rate 0.0 it moves back DOWN; at f0 it clamps without a version bump",
          up == FLOOR.value_at(1) and up > F0 and ev1[0].action == "expand"
          and down == F0 and ev2[0].action == "shrink"
          and ev3[0].action == "clamped_zero" and g.bounds_version == 2,
          f"{up}, {down}, {[e.action for e in ev1 + ev2 + ev3]}, version {g.bounds_version}")

    h = floored()
    last = []
    for _ in range(5):
        fill(h, FLOOR_KEY, 240, 240)
        last = h.update()
    check("the fifth move puts the floor on H_min, notes the transition, and switches to phase dr "
          "with the bounds of a floorless provider at version 0",
          h.phase == "dr" and h.value(FLOOR_KEY) == H_MIN
          and "reached H_min, phase dr" in last[-1].line()
          and h.bounds() == fresh().bounds(),
          f"{h.phase}, {last[-1].line() if last else '-'}")
    is_b2, idx2 = h.boundary_assignment(ra, rb)
    seen = sorted({int(v) for v, ok in zip(idx2, is_b2) if ok})
    check("in phase dr the boundary draw covers the seven and never reaches the floor",
          seen == list(range(7)), str(seen))
    fill(h, FLOOR_KEY, 240, 240)
    ev4 = h.update()
    check("a further full floor buffer clamps at max without a version bump",
          ev4 and ev4[0].action == "clamped_max" and h.bounds_version == 5,
          f"{[e.action for e in ev4]}, version {h.bounds_version}")
    sc = floored().scalars()
    check("with a floor scalars() adds the floor, its fill and dr/phase (36 curves)",
          len(sc) == 36 and sc.get("dr/phase") == 0.0
          and sc.get(f"dr/{FLOOR_KEY}") == F0 and f"dr/{FLOOR_KEY}_fill" in sc,
          f"{len(sc)} curves, phase {sc.get('dr/phase')}")

    st_h = h.state_dict()
    q, err = attempt(lambda: (lambda p_: (p_.load_state_dict(st_h), p_)[1])(floored()))
    check("an autodr-3 state round-trips the floor, its steps and the phase",
          err is None and st_h.get("format") == "autodr-3"
          and st_h.get("floor") == {"f0": F0, "top": H_MIN, "n_steps": 5}
          and q.as_dict() == h.as_dict() and q.phase == "dr",
          f"{err!r}" if err else f"phase {q.phase}")
    refuses("an autodr-1 state refuses -- no run resumes across the format bump",
            lambda: fresh().load_state_dict({**fresh().state_dict(), "format": "autodr-1"}))
    refuses("a state without a floor block refuses",
            lambda: floored().load_state_dict({k: v for k, v in st_h.items() if k != "floor"}))
    refuses("a floor with another rung count refuses on resume -- same keys, other curriculum",
            lambda: AutoDR(DIMS, floor=FloorSpec(F0, H_MIN, 6)).load_state_dict(st_h))
    over = dict(st_h)
    over["steps"] = {**st_h["steps"], FLOOR_KEY: 6}
    refuses("a resume with floor steps above its rung count refuses",
            lambda: floored().load_state_dict(over))
    refuses("a floor whose top is not the start_height centre refuses -- one home for H_min",
            lambda: AutoDR(DIMS, floor=FloorSpec(F0, H_MIN + 0.001, 5)))
    refuses("a floor at or above H_min refuses",
            lambda: FloorSpec(H_MIN, H_MIN, 5).check())
    refuses("a floor without the start_height dimension refuses",
            lambda: AutoDR(bind_centres(CENTRES, dims=tuple(d for d in DR_DIMS
                                                           if d.name != "start_height")),
                           floor=FLOOR))

    nd = AutoDR(bind_centres({"start_height": H_MIN}, dims=(_sh_table,)), floor=FLOOR)
    for _ in range(5):
        fill(nd, FLOOR_KEY, 240, 240)
        nd.update()
    for _ in range(10):
        fill(nd, "start_height_hi", 240, 240)
        nd.update()
    check("the No-DR provider has two keys, reaches all_at_max after floor and ceiling, "
          "and its final band is [H_min, the start_height ceiling]",
          nd.keys == ("start_height_hi", FLOOR_KEY) and nd.all_at_max() is True
          and nd.phase == "dr"
          and nd.bounds()["start_height"] == (H_MIN, _sh_table.hi_max),
          f"{nd.keys}, {nd.bounds()}")

    p = fresh()
    check("before the first full buffer last_rate reads -1 and holds 0",
          p.scalars()["dr/lat_r_hi_last_rate"] == -1.0
          and p.scalars()["dr/lat_r_hi_holds"] == 0.0,
          str({k: v for k, v in p.scalars().items() if k.startswith("dr/lat_r_hi")}))
    fill(p, "lat_r_hi", 120, 240)
    p.update()
    fill(p, "lat_r_hi", 96, 240)
    ev = p.update()
    check("two holds in a row: last_rate is the LAST rate (0.40) and holds counts 2",
          p.scalars()["dr/lat_r_hi_last_rate"] == 0.4
          and p.scalars()["dr/lat_r_hi_holds"] == 2.0 and p.bounds_version == 0,
          f"rate {p.scalars()['dr/lat_r_hi_last_rate']}, holds {p.scalars()['dr/lat_r_hi_holds']}")
    check("without a stall bar (0) a hold line carries no STALL note",
          len(ev) == 1 and "STALL" not in ev[0].line(), ev[0].line() if ev else "no event")
    fill(p, "lat_r_hi", 240, 240)
    p.update()
    check("a real move resets holds to 0 and records rate 1.0",
          p.scalars()["dr/lat_r_hi_holds"] == 0.0
          and p.scalars()["dr/lat_r_hi_last_rate"] == 1.0,
          f"holds {p.scalars()['dr/lat_r_hi_holds']}")
    check("holds is per key -- yaw_hi stays at 0 while lat_r_hi counted",
          p.scalars()["dr/yaw_hi_holds"] == 0.0 and p.scalars()["dr/yaw_hi_last_rate"] == -1.0,
          str(p.scalars()["dr/yaw_hi_holds"]))
    fill(p, "lat_r_hi", 0, 240)
    p.update()
    fill(p, "lat_r_hi", 0, 240)
    p.update()
    check("clamped_zero counts as a hold (the boundary did not move)",
          p.scalars()["dr/lat_r_hi_holds"] == 1.0, str(p.scalars()["dr/lat_r_hi_holds"]))
    q = fresh()
    for _ in range(_ns + 2):
        fill(q, "lat_r_hi", 240, 240)
        q.update()
    check("clamped_max does NOT count as a hold -- a boundary at its ceiling is done",
          q.scalars()["dr/lat_r_hi_holds"] == 0.0 and q.all_at_max() is False,
          str(q.scalars()["dr/lat_r_hi_holds"]))

    st = AutoDR(DIMS, stall_buffers=2)
    fill(st, "tilt_hi", 120, 240)
    ev1 = st.update()
    fill(st, "tilt_hi", 120, 240)
    ev2 = st.update()
    check("with stall bar 2 the FIRST hold carries no STALL note, the SECOND does",
          len(ev1) == 1 and "STALL" not in ev1[0].line()
          and len(ev2) == 1 and ev2[0].action == "hold"
          and ev2[0].line().endswith(" -- STALL 2 full buffers without a move (review bar 2)"),
          f"{ev1[0].line() if ev1 else ''} | {ev2[0].line() if ev2 else ''}")
    check("the STALL note changes nothing -- no move, no version bump",
          st.value("tilt_hi") == 0.0 and st.bounds_version == 0
          and st.as_dict()["stall_buffers"] == 2 and st.as_dict()["holds"]["tilt_hi"] == 2,
          f"{st.value('tilt_hi')}, version {st.bounds_version}")
    refuses("a negative stall bar refuses", lambda: AutoDR(DIMS, stall_buffers=-1))

    st_s = st.state_dict()
    r, err = attempt(lambda: (lambda p_: (p_.load_state_dict(st_s), p_)[1])(AutoDR(DIMS, stall_buffers=2)))
    check("an autodr-3 state round-trips last_rate, holds and the stall bar",
          err is None and st_s.get("format") == "autodr-3"
          and st_s["last_rate"]["tilt_hi"] == 0.5 and st_s["holds"]["tilt_hi"] == 2
          and st_s["stall_buffers"] == 2
          and r.scalars() == st.scalars() and r.as_dict() == st.as_dict(),
          f"{err!r}" if err else "")
    refuses("an autodr-2 state refuses -- no stall block, no resume across the bump",
            lambda: fresh().load_state_dict({**fresh().state_dict(), "format": "autodr-2"}))
    refuses("a resume whose stall bar differs from the configured one refuses",
            lambda: fresh().load_state_dict(st_s))
    refuses("a resume without a holds block refuses",
            lambda: AutoDR(DIMS, stall_buffers=2).load_state_dict(
                {k: v for k, v in st_s.items() if k != "holds"}))
    refuses("a resume with a negative hold count refuses",
            lambda: AutoDR(DIMS, stall_buffers=2).load_state_dict(
                {**st_s, "holds": {**st_s["holds"], "tilt_hi": -1}}))
    refuses("a resume with a last_rate above 1 refuses",
            lambda: AutoDR(DIMS, stall_buffers=2).load_state_dict(
                {**st_s, "last_rate": {**st_s["last_rate"], "tilt_hi": 2.0}}))

    FixedDR = ns["FixedDR"]

    def fixed_dr():
        return FixedDR(DIMS, floor=FLOOR)

    a_ref, x = floored(), fixed_dr()
    same = True
    for rate in (240, 0, 240, 0, 240, 240, 240, 240, 240):
        if x.phase != "floor":
            break
        same = (same and x.bounds() == a_ref.bounds()
                and [bool(v) for v in x.boundary_assignment(ra, rb)[0]]
                == [bool(v) for v in a_ref.boundary_assignment(ra, rb)[0]]
                and [int(v) for v in x.boundary_assignment(ra, rb)[1]]
                == [int(v) for v in a_ref.boundary_assignment(ra, rb)[1]])
        fill(x, FLOOR_KEY, rate, 240)
        fill(a_ref, FLOOR_KEY, rate, 240)
        ex, ea = x.update(), a_ref.update()
        same = same and [e.line() for e in ex] == [e.line() for e in ea]
    check("Fixed-DR phase floor: bounds, boundary draw and floor events equal AutoDR's, "
          "step for step",
          same and x.phase == "dr" and a_ref.phase == "dr"
          and x.bounds_version == a_ref.bounds_version,
          f"{same}, {x.phase}/{a_ref.phase}, version {x.bounds_version}/{a_ref.bounds_version}")
    check("Fixed-DR phase dr: bounds equal FixedWidth.at_max exactly",
          x.bounds() == FixedWidth.at_max(DIMS).bounds(), str(x.bounds()))
    is_bx, _ = x.boundary_assignment(ra, rb)
    check("Fixed-DR phase dr: no env is a boundary env, for any draw",
          not any(bool(v) for v in is_bx), str(sum(1 for v in is_bx if v)))
    sc_f, sc_d = fixed_dr().scalars(), x.scalars()
    check("Fixed-DR: dr/<key> reads the centre in phase floor and the maximum in phase dr; "
          "all_at_max only in phase dr",
          all(sc_f[f"dr/{k}"] == dim_of[k.rsplit('_', 1)[0]].centre for k in EXPECTED_KEYS)
          and all(sc_d[f"dr/{k}"] == dim_of[k.rsplit('_', 1)[0]].bound_max(k.rsplit('_', 1)[1])
                  for k in EXPECTED_KEYS)
          and fixed_dr().all_at_max() is False and x.all_at_max() is True,
          f"{[sc_f[f'dr/{k}'] for k in EXPECTED_KEYS]} / {[sc_d[f'dr/{k}'] for k in EXPECTED_KEYS]}")
    refuses("Fixed-DR without a floor refuses", lambda: FixedDR(DIMS, floor=None))
    st_x = x.state_dict()
    y, err = attempt(lambda: (lambda p_: (p_.load_state_dict(st_x), p_)[1])(fixed_dr()))
    _auto_takes, _ = attempt(lambda: floored().load_state_dict(st_x) or True)
    _fixed_takes, _ = attempt(lambda: fixed_dr().load_state_dict(st_h) or True)
    check("a Fixed-DR state round-trips under its own format; AutoDR refuses it and it "
          "refuses AutoDR's",
          err is None and st_x.get("format") == "fixed-dr-1" and y.as_dict() == x.as_dict()
          and _auto_takes is None and _fixed_takes is None,
          f"{err!r}, auto takes {_auto_takes}, fixed takes {_fixed_takes}")
    refuses("a Fixed-DR state with a boundary off its maximum refuses",
            lambda: fixed_dr().load_state_dict(
                {**st_x, "steps": {**st_x["steps"], "lat_r_hi": 0}}))
    def simulated_run():
        sim = fixed_dr()
        rng_s = random.Random(7)
        for _ in range(20):
            sa = torch.tensor([rng_s.random() for _ in range(1000)])
            sb = torch.tensor([rng_s.random() for _ in range(1000)])
            ok = sim.phase == "floor"
            s_is, s_idx = sim.boundary_assignment(sa, sb)
            for v, i in zip(s_is, s_idx):
                if bool(v):
                    sim.record(sim.keys[int(i)], ok)
            sim.update()
        return sim

    sim, err = attempt(simulated_run)
    check("Fixed-DR in a simulated run: the floor reaches H_min and the seven never move",
          err is None and sim.phase == "dr"
          and sim.bounds() == FixedWidth.at_max(DIMS).bounds()
          and all(sim.as_dict()["fill"][k] == 0 for k in EXPECTED_KEYS),
          f"{err!r}" if err else f"{sim.phase}, {sim.bounds()}")

    return out


MUTATIONS: tuple = (
    (
        "floor-phase-bounds-read-the-seven-steps",
        (("            out = {d.name: (d.centre, d.centre) for d in self.dims}\n"
          "            out[FLOOR_DIM] = (self.value(FLOOR_KEY), self._dim_of[FLOOR_DIM].centre)\n"
          "            return out",
          "            out = {}\n"
          "            for d in self.dims:\n"
          "                lo = d.centre if d.one_sided else self.value(f\"{d.name}_lo\")\n"
          "                out[d.name] = (lo, self.value(f\"{d.name}_hi\"))\n"
          "            out[FLOOR_DIM] = (self.value(FLOOR_KEY), self._dim_of[FLOOR_DIM].centre)\n"
          "            return out"),),
        ("in phase floor every other quantity is its centre and start_height is [floor, H_min], "
         "even with the seven at their maxima",),
    ),
    (
        "floor-not-excluded-from-the-dr-draw",
        (("        n_dr = self.n_boundaries - 1", "        n_dr = self.n_boundaries"),),
        (
            "in phase floor every boundary env is nailed to the floor (index 7)",
            "in phase dr the boundary draw covers the seven and never reaches the floor",
        ),
    ),
    (
        "floor-envs-not-nailed",
        (("            index = index * 0 + n_dr", "            index = index"),),
        ("in phase floor every boundary env is nailed to the floor (index 7)",),
    ),
    (
        "phase-never-leaves-floor",
        (("        if self.floor is not None and self._steps[FLOOR_KEY] < self.floor.n_steps:",
          "        if self.floor is not None:"),),
        (
            "the fifth move puts the floor on H_min, notes the transition, and switches to phase dr "
            "with the bounds of a floorless provider at version 0",
            "in phase dr the boundary draw covers the seven and never reaches the floor",
            "an autodr-3 state round-trips the floor, its steps and the phase",
            "the No-DR provider has two keys, reaches all_at_max after floor and ceiling, "
            "and its final band is [H_min, the start_height ceiling]",
        ),
    ),
    (
        "floor-transition-note-dropped",
        (('                note = " -- reached H_min, phase dr"', '                note = ""'),),
        ("the fifth move puts the floor on H_min, notes the transition, and switches to phase dr "
         "with the bounds of a floorless provider at version 0",),
    ),
    (
        "floor-moves-down-not-up",
        (("        return float(self.f0) + frac * (float(self.top) - float(self.f0))",
          "        return float(self.top) - frac * (float(self.top) - float(self.f0))"),),
        (
            "the floor starts at f0, ends EXACTLY on H_min, and phase is floor at step 0",
            "in phase floor every other quantity is its centre and start_height is [floor, H_min], "
            "even with the seven at their maxima",
            "a full floor buffer at rate 1.0 moves the floor UP one rung (version +1); "
            "at rate 0.0 it moves back DOWN; at f0 it clamps without a version bump",
            "the fifth move puts the floor on H_min, notes the transition, and switches to phase dr "
            "with the bounds of a floorless provider at version 0",
            "with a floor scalars() adds the floor, its fill and dr/phase (36 curves)",
        ),
    ),
    (
        "all-at-max-ignores-the-floor",
        (("        return all(self._steps[k] >= self._n_steps_of(k) for k in self.keys)",
          "        return all(self._steps[k] >= self._n_steps_of(k) for k in self.keys if k != FLOOR_KEY)"),),
        ("all_at_max is False in phase floor even with the seven at their maxima",),
    ),
    (
        "dr-phase-scalar-constant",
        (('            out["dr/phase"] = 0.0 if self.phase == "floor" else 1.0',
          '            out["dr/phase"] = 1.0'),),
        ("with a floor scalars() adds the floor, its fill and dr/phase (36 curves)",),
    ),
    (
        "old-state-format-accepted",
        (('        if state.get("format") != "autodr-3":',
          '        if state.get("format") not in ("autodr-1", "autodr-3"):'),),
        ("an autodr-1 state refuses -- no run resumes across the format bump",),
    ),
    (
        "floor-block-not-compared-on-resume",
        (('        if "floor" not in state:\n            raise AutoDRError("resume state has no \'floor\' block")\n        fl_in = state["floor"]',
          '        fl_in = state.get("floor")'),
         ("        if (fl_in is None) != (self.floor is None):", "        if False:"),
         ("        if fl_in is not None:\n            for field, kind in", "        if False:\n            for field, kind in"),),
        (
            "a state without a floor block refuses",
            "a floor with another rung count refuses on resume -- same keys, other curriculum",
        ),
    ),
    (
        "floor-steps-range-read-off-the-dim",
        (("            n_steps = self._n_steps_of(k)\n            # A step count out of range",
          "            n_steps = 10\n            # A step count out of range"),),
        ("a resume with floor steps above its rung count refuses",),
    ),
    (
        "floor-top-not-pinned-to-the-centre",
        (("            if float(floor.top) != float(sh.centre):", "            if False:"),),
        ("a floor whose top is not the start_height centre refuses -- one home for H_min",),
    ),
    (
        "floor-above-top-accepted",
        (("        if not float(self.f0) < float(self.top):", "        if False:"),),
        ("a floor at or above H_min refuses",),
    ),
    (
        "floor-without-start-height-accepted",
        (("            if sh is None:\n                raise AutoDRError(\n                    f\"a floor needs the {FLOOR_DIM!r} dimension in the provider\"\n                )",
          "            if False:\n                pass"),),
        ("a floor without the start_height dimension refuses",),
    ),
    (
        "state-dict-drops-the-floor-block",
        (('            "format": "autodr-3",\n            "floor": self._floor_block(),',
          '            "format": "autodr-3",\n            "floor": None,'),),
        ("an autodr-3 state round-trips the floor, its steps and the phase",),
    ),
    (
        "floor-key-appended-first-not-last",
        (("            self.keys = tuple(self.keys) + (FLOOR_KEY,)",
          "            self.keys = (FLOOR_KEY,) + tuple(self.keys)"),),
        (
            "with a floor the boundary set has eight entries and the floor comes LAST",
            "the floor key resolves to the start_height column with side 0.0 -- no geometry change",
            "in phase floor every boundary env is nailed to the floor (index 7)",
            "in phase floor every other quantity is its centre and start_height is [floor, H_min], "
            "even with the seven at their maxima",
            "the No-DR provider has two keys, reaches all_at_max after floor and ceiling, "
            "and its final band is [H_min, the start_height ceiling]",
        ),
    ),
    (
        "duplicate-dimension-names-accepted",
        (("        if len(set(names)) != len(names):", "        if False:"),),
        ("a repeated dimension name refuses -- two keys, ONE buffer",),
    ),
    (
        "unknown-dimension-name-accepted",
        (("            if d.name not in COLUMN_INDEX:", "            if False:"),),
        ("a dimension that is not a table column refuses",),
    ),
    (
        "threshold-domain-check-dropped",
        (("if not 0.0 <= float(retreat_at) < float(advance_at) <= 1.0:",
          "if not float(retreat_at) < float(advance_at):"),),
        (
            "an advance threshold above 1 refuses -- the measure is a rate",
            "a negative retreat threshold refuses",
            "a resume with thresholds outside [0, 1] refuses",
        ),
    ),
    (
        "resume-missing-key-is-a-bare-KeyError",
        (('    if key not in state:\n        raise AutoDRError(f"resume state has no {key!r}")',
          "    if key not in state:\n        pass"),),
        (
            "EVERY missing key in a resume state refuses with AutoDRError",
            "a resume whose dims entry is missing a field refuses BY NAME",
            "a resume with no dropped block refuses -- the curve is evidence",
        ),
    ),
    (
        "resume-state-need-not-be-a-dict",
        (("        if not isinstance(state, dict):\n"
          "            raise AutoDRError(\n"
          '                f"resume state must be a dict, got {type(state).__name__}"\n'
          "            )\n"
          '        if state.get("format") != "autodr-3":',
          '        if state.get("format") != "autodr-3":'),),
        ("a resume state that is not a dict refuses",),
    ),
    (
        "nested-resume-block-need-not-be-a-dict",
        (("    if not isinstance(state, dict):\n"
          "        raise AutoDRError(\n"
          '            f"resume state must be a dict, got {type(state).__name__}"\n'
          "        )\n"
          "    if key not in state:",
          "    if key not in state:"),),
        (
            "a resume whose steps block is a LIST refuses",
            "a resume whose dims block holds non-dicts refuses",
        ),
    ),
    (
        "resume-truncates-a-fractional-step",
        (("        if isinstance(v, bool) or not isinstance(v, int):\n"
          '            raise AutoDRError(f"resume {key!r} must be a whole number, got {v!r}")',
          "        if False:\n"
          '            raise AutoDRError(f"resume {key!r} must be a whole number, got {v!r}")'),),
        (
            "a resume with a FRACTIONAL step count refuses, never truncates",
            "a resume with a STRING where a number belongs refuses",
        ),
    ),
    (
        "resume-accepts-a-negative-dropped-count",
        (("            if dk < 0:", "            if False:"),),
        ("a resume with a negative dropped count refuses",),
    ),
    (
        "provenance-report-loses-the-constants",
        (('    lines = [f"buffer m / p_boundary: {AKKAYA_CONSTANT_SOURCE}",\n'
          '             f"delta steps ({DELTA_STEPS}): {DELTA_STEPS_SOURCE}"]',
          "    lines = []"),),
        ("the provenance report names every quantity and both constants",),
    ),
    (
        "fixedwidth-all_at_max-is-always-true",
        (("        return all(self._fixed_steps >= d.n_steps for d in self.dims)",
          "        return True"),),
        ("all_at_max is False for No-DR and True for the eval provider",),
    ),
    (
        "fixedwidth-omits-the-dropped-curve",
        (('        out["dr/flags_dropped"] = 0.0', '        out["dr/nope"] = 0.0'),),
        ("both providers publish one and the same scalar key set",),
    ),
    (
        "autodr-bounds_arrays-does-not-start-at-width-zero",
        (("        for name, pair in self.bounds().items():",
          "        for name, pair in {}.items():"),),
        (
            "AutoDR bounds_arrays reports width 0 on the DR columns at the start",
            "bounds_arrays fills the DR columns and passes the rest through as (0, 1)",
            "the nailed value equals the boundary EXACTLY, for all seven",
        ),
    ),
    (
        "reset-forgets-the-served-count",
        (("    def reset(self) -> None:\n        self._cursor = 0\n        self._served = 0",
          "    def reset(self) -> None:\n        self._cursor = 0"),),
        ("reset rewinds BOTH the cursor and the served count",),
    ),
    (
        "boolean-cells-accepted",
        (("                if isinstance(v, bool) or not isinstance(v, (int, float)):",
          "                if False:"),),
        ("a boolean cell refuses -- True would scale to the upper edge",),
    ),
    (
        "grid-pair-may-repeat-a-quantity",
        (("or pair[0] == pair[1]:", ":"),),
        ("a grid pair that repeats one quantity refuses",),
    ),
    (
        "grid-pair-may-name-a-disturbance-column",
        (("            if name not in DR_DIM_NAMES:", "            if name not in COLUMN_INDEX:"),),
        (
            "a grid pair naming a non-randomised column refuses",
            "a grid pair naming the lateral ANGLE refuses -- lat_phi is a column, not a quantity",
        ),
    ),
    (
        "grid-sizes-may-be-negative",
        (("        if int(cells) < 1 or int(reps) < 1:", "        if False:"),),
        ("a grid table with negative cells or reps refuses",),
    ),
    (
        "boundary_assignment-probability-unguarded",
        (("    if not 0.0 <= p_boundary <= 1.0:\n"
          "        # Out of range this silently makes every env, or no env, a boundary\n"
          "        # env; NaN makes none of them and raises nothing.\n"
          '        raise AutoDRError(f"p_boundary must be in [0, 1], got {p_boundary}")',
          "    if False:\n"
          '        raise AutoDRError(f"p_boundary must be in [0, 1], got {p_boundary}")'),),
        ("boundary_assignment refuses a probability outside [0, 1]",),
    ),
    (
        "moved-ignores-whether-the-value-changed",
        ((" and self.after != self.before", ""),),
        ("an event whose value did NOT change does not count as moved",),
    ),
    (
        "malformed-json-escapes-raw",
        (('raise AutoDRError(f"{p} is not valid JSON: {exc}") from exc', "raise"),),
        ("a malformed autodr_<it>.json refuses BY NAME, not as a JSON error",),
    ),
    (
        "unreadable-state-path-escapes-raw",
        (('raise AutoDRError(f"cannot read the autodr state at {p}: {exc}") from exc',
          "raise"),),
        ("an unreadable autodr state path refuses BY NAME",),
    ),
    (
        "unknown-boundaries-in-a-resume-block-accepted",
        (("            if extra:", "            if False:"),),
        ("a resume naming a boundary this table does not have refuses",),
    ),
    (
        "the-table-is-stored-by-reference",
        (("        self.rows = [list(r) for r in rows]", "        self.rows = rows"),),
        ("the table is COPIED, so a later write cannot walk back validation",),
    ),
    (
        "advance-threshold-exclusive",
        (("if rate >= self.advance_at:", "if rate > self.advance_at:"),),
        ("exactly 0.80 EXPANDS -- Akkaya uses >=, unlike curriculum.Ladder",),
    ),
    (
        "retreat-threshold-exclusive",
        (("elif rate <= self.retreat_at:", "elif rate < self.retreat_at:"),),
        ("exactly 0.10 SHRINKS -- Akkaya uses <=",),
    ),
    (
        "buffer-fires-one-flag-early",
        (("if self._n[key] < self.buffer_m:", "if self._n[key] < self.buffer_m - 1:"),),
        ("239 flags change nothing -- the buffer must be full",),
    ),
    (
        "clamped-attempt-bumps-the-version",
        (("if ev.moved:", "if True:"),),
        (
            "clamped_max does NOT bump bounds_version -- the fresh window survives",
            "clamped_zero does NOT bump bounds_version",
            "a full floor buffer at rate 1.0 moves the floor UP one rung (version +1); "
            "at rate 0.0 it moves back DOWN; at f0 it clamps without a version bump",
            "a further full floor buffer clamps at max without a version bump",
        ),
    ),
    (
        "expansion-never-clamps",
        (("if self._steps[key] >= n_steps:", "if False:"),),
        (
            "one expansion past n_steps clamps and reports clamped_max",
            "clamped_max does NOT bump bounds_version -- the fresh window survives",
            "a further full floor buffer clamps at max without a version bump",
        ),
    ),
    (
        "retreat-never-clamps",
        (("if self._steps[key] <= 0:", "if False:"),),
        (
            "a retreat at width 0 clamps and reports clamped_zero",
            "clamped_zero does NOT bump bounds_version",
        ),
    ),
    (
        "one-sided-quantity-grows-a-lower-boundary",
        (('DimSpec("tilt", "rad", 0.0, 0.17453292519943295, True, _CEILING_NOTE),',
          'DimSpec("tilt", "rad", -0.17453292519943295, 0.17453292519943295, False, _CEILING_NOTE),'),),
        (
            "the boundary set has exactly seven entries",
            "tilt_lo is NOT a boundary -- the tilt magnitude is one-sided",
            "there are seven success buffers, one per boundary",
            "the seven boundaries come in the fixed documented order",
            "a generator of dimensions is materialised, not consumed away",
            "record refuses an unknown boundary",
            "value refuses the same unknown boundary, the same way",
            "scalars() publishes 7 bounds, 7 fill levels, 7 last rates, 7 hold counts and three aggregates",
        ),
    ),
    (
        "the-lat_r-ceiling-outgrows-the-last-lateral-bin-edge",
        (('DimSpec("lat_r", "m", 0.0, 0.030, True, _CEILING_NOTE),',
          'DimSpec("lat_r", "m", 0.0, 0.040, True, _CEILING_NOTE),'),),
        ("the last LATERAL bin edge is the lat_r ceiling",),
    ),
    (
        "the-yaw-ceiling-outgrows-the-last-yaw-bin-edge",
        (('DimSpec("yaw", "rad", -0.08726646259971648, 0.08726646259971648, False, _CEILING_NOTE),',
          'DimSpec("yaw", "rad", -0.10471975511965977, 0.10471975511965977, False, _CEILING_NOTE),'),),
        ("the last YAW bin edge is the yaw magnitude ceiling",),
    ),
    (
        "the-tilt-ceiling-outgrows-the-last-tilt-bin-edge",
        (('DimSpec("tilt", "rad", 0.0, 0.17453292519943295, True, _CEILING_NOTE),',
          'DimSpec("tilt", "rad", 0.0, 0.20943951023931953, True, _CEILING_NOTE),'),),
        ("the last TILT bin edge is the tilt magnitude ceiling",),
    ),
    (
        "the-height-ceiling-outgrows-the-last-height-bin-edge",
        (('DimSpec("start_height", "m", 0.0, 0.270, True,',
          'DimSpec("start_height", "m", 0.0, 0.280, True,'),),
        ("the last HEIGHT bin edge is the start_height ceiling",),
    ),
    (
        "boundary-sides-swapped",
        (('return ("hi",) if self.one_sided else ("lo", "hi")',
          'return ("hi",) if self.one_sided else ("hi", "lo")'),),
        ("the seven boundaries come in the fixed documented order",),
    ),
    (
        "lower-edge-reads-the-upper-maximum",
        (('return self.hi_max if side == "hi" else self.lo_max', "return self.hi_max"),),
        (
            "a lower boundary walks DOWN to lo_max and lands on it exactly",
            "the eval provider opens every quantity to its maximum",
            "bounds_arrays fills the DR columns and passes the rest through as (0, 1)",
            "a movable boundary with nowhere to go refuses at bind time",
        ),
    ),
    (
        "processed-buffer-keeps-one-flag",
        (("            self._n[key] = 0\n            self._s[key] = 0",
          "            self._n[key] = 1\n            self._s[key] = 0"),),
        (
            "a processed buffer is emptied",
            "0.50 holds, writes a hold event that moves nothing, and STILL empties the buffer",
        ),
    ),
    (
        "update-walks-the-boundaries-backwards",
        (("        events = []\n        for key in self.keys:",
          "        events = []\n        for key in reversed(self.keys):"),),
        ("two full buffers fire in ONE update, in key order, version +2",),
    ),
    (
        "overflow-flags-are-dropped-silently",
        (("            self._dropped[key] += 1\n            return", "            return"),),
        ("flags arriving at a full buffer are dropped AND counted",),
    ),
    (
        "record-accepts-any-key",
        (("if key not in self._n:", "if False:"),),
        ("record refuses an unknown boundary",),
    ),
    (
        "generator-dimensions-slip-through",
        (("        dims = tuple(dims)\n        if not dims:", "        if not dims:"),),
        ("a generator of dimensions is materialised, not consumed away",),
    ),
    (
        "empty-dimension-table-accepted",
        (('        if not dims:\n            raise AutoDRError("no dimensions given")',
          "        if not dims:\n            pass"),),
        ("an empty dimension table refuses",),
    ),
    (
        "buffer_m-contract-dropped",
        (("if int(buffer_m) < 1:", "if False:"),),
        (
            "buffer_m below 1 refuses",
            "a resume re-runs the parameter contract (buffer_m 0)",
        ),
    ),
    (
        "resume-skips-the-parameter-contract",
        ((
            '        self._check_params(\n'
            '            _field(state, "buffer_m", int), _field(state, "p_boundary", float),\n'
            '            _field(state, "advance_at", float), _field(state, "retreat_at", float),\n'
            "        )",
            "        pass",
        ),),
        (
            "a resume re-runs the parameter contract (buffer_m 0)",
            "a resume with thresholds outside [0, 1] refuses",
            "EVERY missing key in a resume state refuses with AutoDRError",
        ),
    ),
    (
        "p_boundary-contract-dropped",
        (("if not 0.0 <= float(p_boundary) <= 1.0:", "if False:"),),
        ("p_boundary outside [0, 1] refuses",),
    ),
    (
        "threshold-order-contract-dropped",
        (("if not 0.0 <= float(retreat_at) < float(advance_at) <= 1.0:", "if False:"),),
        (
            "a retreat threshold above the advance threshold refuses",
            "an advance threshold above 1 refuses -- the measure is a rate",
            "a negative retreat threshold refuses",
            "a resume with thresholds outside [0, 1] refuses",
        ),
    ),
    (
        "n_steps-contract-dropped",
        (("if self.n_steps < 1:", "if False:"),),
        ("a dimension with n_steps below 1 refuses",),
    ),
    (
        "inverted-maxima-accepted",
        (("if self.lo_max > self.hi_max:", "if False:"),),
        ("inverted maxima are refused BY NAME, not by the centre-range guard",),
    ),
    (
        "resume-drops-the-version",
        (('self._bounds_version = version', "self._bounds_version = 0"),),
        (
            "state_dict round-trips steps, buffers and bounds_version",
            "save/load round-trips through the autodr_<it>.json file itself",
        ),
    ),
    (
        "resume-accepts-a-foreign-boundary-set",
        ((
            '            raise AutoDRError(f"unknown autodr state format {state.get(\'format\')!r}")\n'
            '        if tuple(_field(state, "keys", list)) != self.keys:',
            '            raise AutoDRError(f"unknown autodr state format {state.get(\'format\')!r}")\n'
            "        if False:",
        ),),
        (
            "a resume with a different boundary set refuses",
            "EVERY missing key in a resume state refuses with AutoDRError",
        ),
    ),
    (
        "resume-ignores-a-changed-maximum",
        (("if _field(s, field, float) != getattr(d, field):", "if False:"),),
        ("a resume with a changed maximum refuses",),
    ),
    (
        "resume-ignores-an-out-of-range-step-count",
        (("if not 0 <= st <= n_steps:", "if False:"),),
        (
            "a resume with floor steps above its rung count refuses",
            "a resume with a step count past the maximum refuses",
            "a resume with a NEGATIVE step count refuses",
        ),
    ),
    (
        "resume-ignores-an-impossible-buffer",
        (("if not 0 <= sc <= n <= buffer_m:", "if False:"),),
        ("a resume with more flags than the buffer holds refuses",),
    ),
    (
        "fixedwidth-state-ignores-the-table",
        (("                if _field(sd, field, kind) != getattr(d, field):",
          "                if False:"),),
        ("a FixedWidth state from a DIFFERENT table refuses",),
    ),
    (
        "fixedwidth-state-loses-its-width",
        (("        self._fixed_steps = steps", "        self._fixed_steps = 0"),),
        ("a FixedWidth state round-trips its width",),
    ),
    (
        "fixedwidth-emits-an-event",
        (("    def update(self) -> list:\n        return []",
          "    def update(self) -> list:\n        return [1]"),),
        ("FixedWidth swallows a record and never emits an event",),
    ),
    (
        "no-dr-nails-boundary-envs",
        (("    p_boundary: float = 0.0", "    p_boundary: float = 1.0"),),
        ("No-DR never nails a boundary env, for any draw",),
    ),
    (
        "unit-mapping-loses-the-lower-edge",
        (("return lo + unit * (hi - lo)", "return unit * (hi - lo)"),),
        (
            "map(0) == lo EXACTLY", "map(1) == hi EXACTLY",
            "the nailed value equals the boundary EXACTLY, for all seven",
        ),
    ),
    (
        "boundary-index-clamped-one-too-high",
        (("index = (rand_b * n_boundaries).long().clamp(0, n_boundaries - 1)",
          "index = (rand_b * n_boundaries).long().clamp(0, n_boundaries)"),),
        ("the boundary index never leaves [0, n-1], not even at rand == 1.0",),
    ),
    (
        "boundary-index-spans-too-few-boundaries",
        (("index = (rand_b * n_boundaries).long()",
          "index = (rand_b * 3).long()"),),
        (
            "the boundary index is uniform over all seven boundaries",
            "the boundary index never leaves [0, n-1], not even at rand == 1.0",
        ),
    ),
    (
        "boundary-count-guard-dropped",
        (("if n_boundaries < 1:", "if False:"),),
        ("boundary_assignment refuses a non-positive boundary count",),
    ),
    (
        "boundary-probability-halved",
        (("is_boundary = rand_a < p_boundary", "is_boundary = rand_a < p_boundary * 0.5"),),
        ("p_b over 10^4 draws lands in [0.47, 0.53], for three seeds",),
    ),
    (
        "apply_boundary-nails-the-opposite-edge",
        (("unit[rows, cols] = side", "unit[rows, cols] = 1.0 - side"),),
        (
            "apply_boundary nails 1.0 for an upper edge and 0.0 for a lower edge",
            "the nailed value equals the boundary EXACTLY, for all seven",
        ),
    ),
    (
        "table-rows-shifted-by-one",
        (("return table_unit[env_ids]",
          "return table_unit[(env_ids + 1) % table_unit.shape[0]]"),),
        ("table_rows: env i plays row i, for a partial reset and for the full first reset",),
    ),
    (
        "testwert-marker-dropped",
        (("return tuple(d.name for d in dims if TESTWERT in d.source)", "return ()"),),
        (
            "every quantity but friction carries [TESTWERT]",
            "the provenance report names every quantity and both constants",
            "AutoDR.as_dict publishes the decided-not-derived names",
            "FixedWidth.as_dict publishes the decided-not-derived names",
        ),
    ),
    (
        "a-decided-ceiling-quietly-loses-its-marker",
        (('            "[TESTWERT] hi = ceiling, decided not derived; "',
          '            "hi = ceiling, decided not derived; "'),),
        (
            "every quantity but friction carries [TESTWERT]",
            "the provenance report names every quantity and both constants",
            "AutoDR.as_dict publishes the decided-not-derived names",
            "FixedWidth.as_dict publishes the decided-not-derived names",
        ),
    ),
    (
        "the-published-testwert-list-loses-a-name",
        (
            ('            "retreat_at": self.retreat_at,\n'
             '            "all_at_max": self.all_at_max(),\n'
             '            "testwert": list(testwert_dims(self.dims)),',
             '            "retreat_at": self.retreat_at,\n'
             '            "all_at_max": self.all_at_max(),\n'
             '            "testwert": list(testwert_dims(self.dims))[:-1],'),
            ('            "fill": {k: 0 for k in self.keys},\n'
             '            "all_at_max": self.all_at_max(),\n'
             '            "testwert": list(testwert_dims(self.dims)),',
             '            "fill": {k: 0 for k in self.keys},\n'
             '            "all_at_max": self.all_at_max(),\n'
             '            "testwert": list(testwert_dims(self.dims))[:-1],'),
        ),
        (
            "AutoDR.as_dict publishes the decided-not-derived names",
            "FixedWidth.as_dict publishes the decided-not-derived names",
        ),
    ),
    (
        "the-sourced-friction-band-is-marked-again",
        (('            "DuPont PA66/POM table span, assumed PA6/POM pairing, "',
          '            "[TESTWERT] DuPont PA66/POM table span, assumed PA6/POM pairing, "'),),
        (
            "every quantity but friction carries [TESTWERT]",
            "the friction line is reported as sourced, not as a [TESTWERT]",
            "the provenance report names every quantity and both constants",
            "AutoDR.as_dict publishes the decided-not-derived names",
            "FixedWidth.as_dict publishes the decided-not-derived names",
        ),
    ),
    (
        "column-lookup-collapses",
        (("return COLUMN_INDEX[self.name]", "return 0"),),
        (
            "each quantity maps to its OWN table column",
            "boundary_columns and boundary_sides line up with the keys",
            "the nailed value equals the boundary EXACTLY, for all seven",
        ),
    ),
    (
        "bounds_arrays-passes-nothing-through",
        (("hi = [1.0] * len(TABLE_COLUMNS)", "hi = [0.0] * len(TABLE_COLUMNS)"),),
        (
            "bounds_arrays fills the DR columns and passes the rest through as (0, 1)",
            "AutoDR bounds_arrays reports width 0 on the DR columns at the start",
        ),
    ),
    (
        "a-table-column-is-lost",
        (('    "joint_noise_5",    # 14\n', ""),),
        ("the evaluation table has eighteen columns",),
    ),
    (
        "the-azimuth-column-is-dropped",
        (('    "tilt_azimuth",     # 4  fixture tilt DIRECTION -- no boundary, no buffer\n',
          ""),),
        (
            "the tilt AZIMUTH owns a column but is not an AutoDR quantity",
            "the evaluation table has eighteen columns",
        ),
    ),
    (
        "the-grasp-column-is-renamed",
        (('    "grasp_obs_x",      # 15 grasp belief error',
          '    "grasp_obs_off_x",  # 15 grasp belief error'),),
        ("the grasp belief error owns a table column and nails NO boundary",),
    ),
    (
        "the-pocket-yaw-column-is-renamed",
        (('    "pocket_obs_yaw",   # 16 pocket-yaw belief error',
          '    "pocket_yaw_obs",   # 16 pocket-yaw belief error'),),
        ("the pocket-yaw belief error owns a table column and nails NO boundary",),
    ),
    (
        "the-fixture-z-column-is-renamed",
        (('    "fixture_noise_z",  # 17 fixture_pos_noise_z',
          '    "fixture_z_noise",  # 17 fixture_pos_noise_z'),),
        ("the fixture z spread owns the LAST table column and nails NO boundary",),
    ),
    (
        "every-boundary-nails-the-grasp-column",
        (('return tuple(self._dim_of[k.rsplit("_", 1)[0]].column for k in self.keys)',
          'return tuple(COLUMN_INDEX["grasp_obs_x"] for k in self.keys)'),),
        (
            "the grasp belief error owns a table column and nails NO boundary",
            "boundary_columns and boundary_sides line up with the keys",
            "the nailed value equals the boundary EXACTLY, for all seven",
        ),
    ),
    (
        "all_at_max-ignores-the-step-count",
        ((
            "self._steps[k] >= self._n_steps_of(k) for k in self.keys",
            "self._steps[k] >= 0 for k in self.keys",
        ),),
        (
            "all_at_max is False in phase floor even with the seven at their maxima",
            "all_at_max is False at width 0",
            "all_at_max stays False while one boundary is short",
            "a generator of dimensions is materialised, not consumed away",
        ),
    ),
    (
        "delta-is-not-one-tenth",
        (("return (self.bound_max(side) - self.centre) / float(self.n_steps)",
          "return (self.bound_max(side) - self.centre) / 4.0"),),
        (
            "the delta is (maximum - centre) / n_steps",
            "240 flags at 1.00 expand the boundary by exactly one delta",
        ),
    ),
    (
        "step-fraction-drifts",
        (("frac = float(steps) / float(self.n_steps)",
          "frac = float(steps) / float(self.n_steps + 1)"),),
        (
            "n_steps expansions land on the maximum EXACTLY (no float drift)",
            "a lower boundary walks DOWN to lo_max and lands on it exactly",
            "the eval provider opens every quantity to its maximum",
            "FixedWidth.at_max: bounds_max == bounds == the maxima",
            "bounds_arrays fills the DR columns and passes the rest through as (0, 1)",
            "240 flags at 1.00 expand the boundary by exactly one delta",
            "one expansion past n_steps clamps and reports clamped_max",
        ),
    ),
    (
        "take-loses-the-table-order",
        (("out = [(self._cursor + i) % self.n_rows for i in range(k)]",
          "out = sorted((self._cursor + i) % self.n_rows for i in range(k))"),),
        ("take() hands out row ids in table order and wraps",),
    ),
    (
        "served-stops-counting",
        (("        self._served += k\n        return out", "        return out"),),
        (
            "served counts every row handed out and never wraps",
            "a wrapped take REPEATS a row id -- served is what says so",
        ),
    ),
    (
        "row-range-check-dropped",
        (("if not 0.0 <= v <= 1.0:", "if False:"),),
        ("a row value outside [0, 1] refuses",),
    ),
    (
        "row-width-check-dropped",
        (("if len(r) != width:", "if False:"),),
        ("a row of the wrong width refuses",),
    ),
    (
        "allow-outside-ignored",
        (("                if allow_outside:\n                    continue",
          "                if False:\n                    continue"),),
        ("an OOD table (allow_outside) keeps values beyond [0, 1]",
         "from_csv hands allow_outside through to the table"),
    ),
    (
        "finite-check-dropped",
        (("if not math.isfinite(v):", "if False:"),),
        ("allow_outside still refuses a value that is not finite",),
    ),
    (
        "from-csv-drops-allow-outside",
        (("return cls(rows, sha, str(p), allow_outside=allow_outside)", "return cls(rows, sha, str(p))"),),
        ("from_csv hands allow_outside through to the table",),
    ),
    (
        "csv-hash-is-not-the-file",
        (("sha = hashlib.sha256(raw).hexdigest()", 'sha = hashlib.sha256(b"").hexdigest()'),),
        ("from_csv reads the rows and hashes the FILE",),
    ),
    (
        "csv-header-check-dropped",
        (("if tuple(h.strip() for h in header) != TABLE_COLUMNS:", "if False:"),),
        ("a header that is not TABLE_COLUMNS is refused BY NAME",),
    ),
    (
        "csv-cell-error-escapes-raw",
        (('raise AutoDRError(f"{p}: a cell is not a number -- {exc}") from exc', "raise"),),
        ("a cell that is not a number refuses with AutoDRError, not ValueError",),
    ),
    (
        "grid-cell-arithmetic-collapses",
        (("return row_id // self.reps", "return row_id"),),
        ("a grid row belongs to cell id // reps, repetition id % reps",),
    ),
    (
        "grid-size-check-dropped",
        (("if int(cells) * int(reps) != self.n_rows:", "if False:"),),
        ("a grid table whose cells * reps misses the row count refuses",),
    ),
    (
        "grid-pair-arity-check-dropped",
        (("        if len(pair) != 2 or pair[0] == pair[1]:", "        if False:"),),
        (
            "a grid pair that is not exactly two quantities refuses",
            "a grid pair that repeats one quantity refuses",
        ),
    ),
    (
        "event-line-loses-its-marker",
        (('f"[autodr] {self.key}: rate {self.rate:.4f} over {self.buffer_m} "',
          'f"[adr] {self.key}: rate {self.rate:.4f} over {self.buffer_m} "'),),
        ("a bound event writes a marked [autodr] line",),
    ),
    (
        "flags_dropped-curve-lost",
        (('out["dr/flags_dropped"] = float(sum(self._dropped.values()))',
          'out["dr/nope"] = float(sum(self._dropped.values()))'),),
        (
            "scalars() publishes 7 bounds, 7 fill levels, 7 last rates, 7 hold counts and three aggregates",
            "flags arriving at a full buffer are dropped AND counted",
            "both providers publish one and the same scalar key set",
        ),
    ),
    (
        "bounds_version-never-rises",
        (("self._bounds_version += 1", "self._bounds_version += 0"),),
        (
            "a real move bumps bounds_version by exactly 1",
            "two full buffers fire in ONE update, in key order, version +2",
            "a hold does not bump bounds_version",
        ),
    ),
    (
        "bounds_version-does-not-start-at-zero",
        (("        self._bounds_version = 0", "        self._bounds_version = 5"),),
        (
            "bounds_version starts at 0",
            "a real move bumps bounds_version by exactly 1",
            "two full buffers fire in ONE update, in key order, version +2",
            "a hold does not bump bounds_version",
            "clamped_zero does NOT bump bounds_version",
            "FixedWidth swallows a record and never emits an event",
        ),
    ),
    (
        "all_at_max-never-turns-true",
        ((
            "self._steps[k] >= self._n_steps_of(k) for k in self.keys",
            "self._steps[k] > self._n_steps_of(k) for k in self.keys",
        ),),
        (
            "all_at_max turns True only when EVERY boundary is at its maximum",
            "the No-DR provider has two keys, reaches all_at_max after floor and ceiling, "
            "and its final band is [H_min, the start_height ceiling]",
        ),
    ),
    (
        "provider-bounds_max-gets-a-default-again",
        (('        raise NotImplementedError\n\n    def bounds_arrays(self) -> tuple:',
          '        return {d.name: (d.lo_max, d.hi_max) for d in self.dims}\n\n    def bounds_arrays(self) -> tuple:'),),
        (
            "_Provider.bounds_max is abstract -- no provider inherits an answer",
        ),
    ),
    (
        "bounds_max-collapses-onto-the-current-bounds",
        (("        return {d.name: (d.lo_max, d.hi_max) for d in self.dims}",
          "        return self.bounds()"),),
        (
            "bounds_max gives the dimension maxima, not the width-0 bounds",
            "bounds_max reports a POSITIVE angle span at width 0, bounds does not",
            "a moved boundary stays inside bounds_max, which does not move",
        ),
    ),
    (
        "fixedwidth-inherits-the-dimension-maxima",
        (("        return self.bounds()\n\n    def record(self, key: str, success: bool) -> None:\n        return None",
          "        return {d.name: (d.lo_max, d.hi_max) for d in self.dims}\n\n"
          "    def record(self, key: str, success: bool) -> None:\n        return None"),),
        (
            "FixedWidth.at_centre: bounds_max == bounds, zero span everywhere",
        ),
    ),
    (
        "eval-provider-does-not-start-at-full-width",
        (("        return cls(dims, steps=steps.pop())", "        return cls(dims, steps=0)"),),
        (
            "FixedWidth.at_max: bounds_max == bounds == the maxima",
            "the eval provider opens every quantity to its maximum",
            "all_at_max is False for No-DR and True for the eval provider",
            "bounds_arrays fills the DR columns and passes the rest through as (0, 1)",
        ),
    ),
    (
        "no-dr-does-not-start-at-the-centre",
        (("        return cls(dims, steps=0)", "        return cls(dims, steps=1)"),),
        (
            "No-DR keeps every quantity at its centre",
            "FixedWidth.at_centre: bounds_max == bounds, zero span everywhere",
        ),
    ),
    (
        "a-fresh-provider-does-not-start-at-width-zero",
        (("        self._steps = {k: 0 for k in self.keys}",
          "        self._steps = {k: 2 for k in self.keys}"),),
        (
            "a fresh provider starts at width 0 -- lo == hi == centre",
            "bounds_max reports a POSITIVE angle span at width 0, bounds does not",
            "239 flags change nothing -- the buffer must be full",
            "240 flags at 1.00 expand the boundary by exactly one delta",
            "exactly 0.10 SHRINKS -- Akkaya uses <=",
            "0.50 holds, writes a hold event that moves nothing, and STILL empties the buffer",
            "a retreat at width 0 clamps and reports clamped_zero",
            "clamped_zero does NOT bump bounds_version",
            "AutoDR bounds_arrays reports width 0 on the DR columns at the start",
        ),
    ),
    (
        "centre-range-check-dropped",
        (("if not (self.lo_max <= c <= self.hi_max):", "if False:"),),
        ("a centre outside the maxima refuses",),
    ),
    (
        "one-sided-lower-edge-keeps-the-placeholder",
        (("lo_max=float(centre)", "lo_max=self.lo_max"),),
        (
            "bind sets lo_max = centre for one-sided dims",
            "the eval provider opens every quantity to its maximum",
        ),
    ),
    (
        "missing-centre-accepted",
        (("    missing = [d.name for d in dims if d.name not in centres]", "    missing = []"),),
        ("a missing centre refuses to build the table",),
    ),
)


STALL_MUTATIONS: tuple = (
    (
        "hold-writes-no-event",
        (('                action = "hold"\n', '                continue\n'),),
        (
            "0.50 holds, writes a hold event that moves nothing, and STILL empties the buffer",
            "without a stall bar (0) a hold line carries no STALL note",
            "with stall bar 2 the FIRST hold carries no STALL note, the SECOND does",
            "two holds in a row: last_rate is the LAST rate (0.40) and holds counts 2",
            "the STALL note changes nothing -- no move, no version bump",
            "an autodr-3 state round-trips last_rate, holds and the stall bar",
        ),
    ),
    (
        "holds-never-reset",
        (("                self._holds[key] = 0\n", "                pass\n"),),
        (
            "a real move resets holds to 0 and records rate 1.0",
            "clamped_zero counts as a hold (the boundary did not move)",
        ),
    ),
    (
        "last-rate-starts-at-zero",
        (("        self._last_rate = {k: -1.0 for k in self.keys}",
          "        self._last_rate = {k: 0.0 for k in self.keys}"),),
        ("before the first full buffer last_rate reads -1 and holds 0",),
    ),
    (
        "holds-counted-on-every-key",
        (("                self._holds[key] += 1",
          "                self._holds.update({k_: v_ + 1 for k_, v_ in self._holds.items()})"),),
        ("holds is per key -- yaw_hi stays at 0 while lat_r_hi counted",),
    ),
    (
        "clamped-max-counts-as-a-hold",
        (('            elif action != "clamped_max":', "            else:"),),
        ("clamped_max does NOT count as a hold -- a boundary at its ceiling is done",),
    ),
    (
        "last-rate-never-recorded",
        (("            self._last_rate[key] = rate\n", ""),),
        (
            "scalars() publishes 7 bounds, 7 fill levels, 7 last rates, 7 hold counts and three aggregates",
            "two holds in a row: last_rate is the LAST rate (0.40) and holds counts 2",
            "a real move resets holds to 0 and records rate 1.0",
            "an autodr-3 state round-trips last_rate, holds and the stall bar",
        ),
    ),
    (
        "stall-note-never-written",
        (("            if 0 < self.stall_buffers <= self._holds[key]:",
          "            if False:"),),
        ("with stall bar 2 the FIRST hold carries no STALL note, the SECOND does",),
    ),
    (
        "stall-note-on-the-first-hold",
        (("            if 0 < self.stall_buffers <= self._holds[key]:",
          "            if 0 < self.stall_buffers:"),),
        ("with stall bar 2 the FIRST hold carries no STALL note, the SECOND does",),
    ),
    (
        "scalars-publish-a-flat-zero-hold-count",
        (('            out[f"dr/{key}_holds"] = float(self._holds[key])',
          '            out[f"dr/{key}_holds"] = 0.0'),),
        (
            "two holds in a row: last_rate is the LAST rate (0.40) and holds counts 2",
            "clamped_zero counts as a hold (the boundary did not move)",
        ),
    ),
    (
        "resume-invents-a-missing-holds-block",
        (('        ho_in = _field(state, "holds", dict)',
          '        ho_in = state.get("holds", {k: 0 for k in self.keys})'),),
        ("a resume without a holds block refuses",),
    ),
    (
        "resume-accepts-the-stall-less-format",
        (('        if state.get("format") != "autodr-3":',
          '        if state.get("format") not in ("autodr-2", "autodr-3"):'),),
        ("an autodr-2 state refuses -- no stall block, no resume across the bump",),
    ),
    (
        "stall-note-bumps-the-version",
        (("            if 0 < self.stall_buffers <= self._holds[key]:\n                ev = BoundEvent(",
          "            if 0 < self.stall_buffers <= self._holds[key]:\n                self._bounds_version += 1\n                ev = BoundEvent("),),
        ("the STALL note changes nothing -- no move, no version bump",),
    ),
    (
        "negative-stall-bar-accepted",
        (("        if int(stall_buffers) < 0:", "        if False:"),),
        ("a negative stall bar refuses",),
    ),
    (
        "resume-ignores-the-stall-bar",
        (("        if stall != self.stall_buffers:", "        if False:"),),
        ("a resume whose stall bar differs from the configured one refuses",),
    ),
    (
        "resume-accepts-a-negative-hold-count",
        (("            if hk < 0:", "            if False:"),),
        ("a resume with a negative hold count refuses",),
    ),
    (
        "resume-accepts-a-last-rate-above-1",
        (("            if not (lr == -1.0 or 0.0 <= lr <= 1.0):", "            if False:"),),
        ("a resume with a last_rate above 1 refuses",),
    ),
)

FIXED_DR_MUTATIONS: tuple = (
    (
        "fixed-dr-seven-start-at-width-0",
        (("                self._steps[key] = self._n_steps_of(key)", "                pass"),),
        (
            "Fixed-DR phase dr: bounds equal FixedWidth.at_max exactly",
            "Fixed-DR: dr/<key> reads the centre in phase floor and the maximum in phase dr; "
            "all_at_max only in phase dr",
            "a Fixed-DR state round-trips under its own format; AutoDR refuses it and it "
            "refuses AutoDR's",
            "Fixed-DR in a simulated run: the floor reaches H_min and the seven never move",
        ),
    ),
    (
        "fixed-dr-value-reports-the-maximum-in-phase-floor",
        (('        if key != FLOOR_KEY and key in self._steps and self.phase == "floor":',
          "        if False:"),),
        ("Fixed-DR: dr/<key> reads the centre in phase floor and the maximum in phase dr; "
         "all_at_max only in phase dr",),
    ),
    (
        "fixed-dr-keeps-boundary-sampling-in-phase-dr",
        (("        return boundary_assignment(rand_a, rand_b, self.n_boundaries - 1, 0.0)",
          "        return boundary_assignment(rand_a, rand_b, self.n_boundaries - 1, self.p_boundary)"),),
        (
            "Fixed-DR phase dr: no env is a boundary env, for any draw",
            "Fixed-DR in a simulated run: the floor reaches H_min and the seven never move",
        ),
    ),
    (
        "fixed-dr-floor-phase-without-boundary-episodes",
        (("            return super().boundary_assignment(rand_a, rand_b)",
          "            return boundary_assignment(rand_a, rand_b, self.n_boundaries - 1, 0.0)"),),
        (
            "Fixed-DR phase floor: bounds, boundary draw and floor events equal AutoDR's, "
            "step for step",
            "Fixed-DR in a simulated run: the floor reaches H_min and the seven never move",
        ),
    ),
    (
        "fixed-dr-accepts-no-floor",
        (('        if floor is None:\n            raise AutoDRError(\n                "FixedDR needs',
          '        if False:\n            raise AutoDRError(\n                "FixedDR needs'),),
        ("Fixed-DR without a floor refuses",),
    ),
    (
        "fixed-dr-writes-the-autodr-format",
        (('        out["format"] = self.FORMAT', "        pass"),),
        ("a Fixed-DR state round-trips under its own format; AutoDR refuses it and it "
         "refuses AutoDR's",),
    ),
    (
        "fixed-dr-accepts-a-moved-boundary",
        (('        if moved:\n            raise AutoDRError(f"fixed-dr state',
          '        if False:\n            raise AutoDRError(f"fixed-dr state'),),
        ("a Fixed-DR state with a boundary off its maximum refuses",),
    ),
)
MUTATIONS = MUTATIONS + STALL_MUTATIONS + FIXED_DR_MUTATIONS

FLOOR_EXTRA_FLIPS: dict = {
    "all_at_max-ignores-the-step-count": (
        "clamped_max does NOT count as a hold -- a boundary at its ceiling is done",
    ),
    "event-line-loses-its-marker": (
        "0.50 holds, writes a hold event that moves nothing, and STILL empties the buffer",
    ),
    "processed-buffer-keeps-one-flag": (
        "a real move resets holds to 0 and records rate 1.0",
    ),
    "resume-missing-key-is-a-bare-KeyError": (
        "a resume without a holds block refuses",
    ),
    "clamped-attempt-bumps-the-version": (
        "0.50 holds, writes a hold event that moves nothing, and STILL empties the buffer",
        "a hold does not bump bounds_version",
        "an autodr-3 state round-trips last_rate, holds and the stall bar",
        "clamped_zero counts as a hold (the boundary did not move)",
        "the STALL note changes nothing -- no move, no version bump",
        "two holds in a row: last_rate is the LAST rate (0.40) and holds counts 2",
        "with stall bar 2 the FIRST hold carries no STALL note, the SECOND does",
    ),
    "last-rate-starts-at-zero": (
        "holds is per key -- yaw_hi stays at 0 while lat_r_hi counted",
        "scalars() publishes 7 bounds, 7 fill levels, 7 last rates, 7 hold counts and three aggregates",
    ),
    "expansion-never-clamps": (
        "an autodr-3 state round-trips the floor, its steps and the phase",
    ),
    "retreat-never-clamps": (
        "clamped_zero counts as a hold (the boundary did not move)",
        "a full floor buffer at rate 1.0 moves the floor UP one rung (version +1); "
        "at rate 0.0 it moves back DOWN; at f0 it clamps without a version bump",
    ),
    "one-sided-quantity-grows-a-lower-boundary": (
        "in phase dr the boundary draw covers the seven and never reaches the floor",
        "in phase floor every boundary env is nailed to the floor (index 7)",
        "with a floor scalars() adds the floor, its fill and dr/phase (36 curves)",
        "with a floor the boundary set has eight entries and the floor comes LAST",
    ),
    "boundary-sides-swapped": (
        "with a floor the boundary set has eight entries and the floor comes LAST",
    ),
    "resume-drops-the-version": (
        "an autodr-3 state round-trips the floor, its steps and the phase",
    ),
    "unit-mapping-loses-the-lower-edge": (
        "map(0) on the floor band IS the floor, exactly",
    ),
    "boundary-index-spans-too-few-boundaries": (
        "in phase dr the boundary draw covers the seven and never reaches the floor",
    ),
    "boundary-probability-halved": (
        "in phase floor every boundary env is nailed to the floor (index 7)",
    ),
    "column-lookup-collapses": (
        "the floor key resolves to the start_height column with side 0.0 -- no geometry change",
    ),
    "every-boundary-nails-the-grasp-column": (
        "the floor key resolves to the start_height column with side 0.0 -- no geometry change",
    ),
    "step-fraction-drifts": (
        "the No-DR provider has two keys, reaches all_at_max after floor and ceiling, "
        "and its final band is [H_min, the start_height ceiling]",
    ),
    "bounds_version-never-rises": (
        "a full floor buffer at rate 1.0 moves the floor UP one rung (version +1); "
        "at rate 0.0 it moves back DOWN; at f0 it clamps without a version bump",
        "a further full floor buffer clamps at max without a version bump",
    ),
    "bounds_version-does-not-start-at-zero": (
        "a full floor buffer at rate 1.0 moves the floor UP one rung (version +1); "
        "at rate 0.0 it moves back DOWN; at f0 it clamps without a version bump",
        "a further full floor buffer clamps at max without a version bump",
        "0.50 holds, writes a hold event that moves nothing, and STILL empties the buffer",
        "the STALL note changes nothing -- no move, no version bump",
        "two holds in a row: last_rate is the LAST rate (0.40) and holds counts 2",
    ),
    "a-fresh-provider-does-not-start-at-width-zero": (
        "a full floor buffer at rate 1.0 moves the floor UP one rung (version +1); "
        "at rate 0.0 it moves back DOWN; at f0 it clamps without a version bump",
        "a further full floor buffer clamps at max without a version bump",
        "clamped_zero counts as a hold (the boundary did not move)",
        "the STALL note changes nothing -- no move, no version bump",
        "in phase floor every other quantity is its centre and start_height is [floor, H_min], "
        "even with the seven at their maxima",
        "the fifth move puts the floor on H_min, notes the transition, and switches to phase dr "
        "with the bounds of a floorless provider at version 0",
        "the floor starts at f0, ends EXACTLY on H_min, and phase is floor at step 0",
        "with a floor scalars() adds the floor, its fill and dr/phase (36 curves)",
    ),
}


_FD_FLOOR = ("Fixed-DR phase floor: bounds, boundary draw and floor events equal AutoDR's, "
             "step for step")
_FD_AT_MAX = "Fixed-DR phase dr: bounds equal FixedWidth.at_max exactly"
_FD_NO_B = "Fixed-DR phase dr: no env is a boundary env, for any draw"
_FD_CURVES = ("Fixed-DR: dr/<key> reads the centre in phase floor and the maximum in phase dr; "
              "all_at_max only in phase dr")
_FD_STATE = ("a Fixed-DR state round-trips under its own format; AutoDR refuses it and it "
             "refuses AutoDR's")
_FD_SIM = "Fixed-DR in a simulated run: the floor reaches H_min and the seven never move"

FIXED_DR_EXTRA_FLIPS: dict = {
    "floor-not-excluded-from-the-dr-draw": (_FD_SIM,),
    "floor-envs-not-nailed": (_FD_SIM,),
    "phase-never-leaves-floor": (_FD_SIM, _FD_AT_MAX, _FD_NO_B, _FD_FLOOR, _FD_CURVES),
    "all-at-max-ignores-the-floor": (_FD_CURVES,),
    "state-dict-drops-the-floor-block": (_FD_STATE,),
    "floor-key-appended-first-not-last": (_FD_SIM,),
    "resume-drops-the-version": (_FD_STATE,),
    "all_at_max-ignores-the-step-count": (_FD_CURVES,),
    "step-fraction-drifts": (_FD_CURVES,),
    "all_at_max-never-turns-true": (_FD_CURVES,),
    "eval-provider-does-not-start-at-full-width": (_FD_SIM, _FD_AT_MAX),
}


def _expected_flips(name: str, expected) -> tuple:
    return (tuple(expected) + tuple(FLOOR_EXTRA_FLIPS.get(name, ()))
            + tuple(FIXED_DR_EXTRA_FLIPS.get(name, ())))


def _run(torch, mutations=()) -> dict:
    return {name: ok for name, ok, _ in build_checks(load_autodr(mutations), torch)}


_WORKER_TORCH = None


def _run_mutation(index: int):
    """One mutation by its table index -- the unit of the process pool."""
    global _WORKER_TORCH
    if _WORKER_TORCH is None:
        _WORKER_TORCH = _load_torch()
    name, muts, _ = MUTATIONS[index]
    try:
        return name, _run(_WORKER_TORCH, muts)
    except Exception as exc:  # noqa: BLE001 - reported by the parent
        return name, exc


def _coverage(baseline: dict) -> list:
    """Every check needs a mutation that detects the corresponding fault."""
    defended = {
        name for mname, _, expected in MUTATIONS for name in _expected_flips(mname, expected)
    }
    return sorted(set(baseline) - defended - set(UNPROVABLE))


def _self_test(torch) -> int:
    print("[check_autodr] counter-proof")
    baseline = _run(torch)
    rc = 0
    if not all(baseline.values()):
        bad = [k for k, v in baseline.items() if not v]
        print(f"  FAIL  baseline is not green, counter-proof is meaningless: {bad}")
        return 1
    print(f"  PASS  baseline: {len(baseline)} checks green")

    results: dict = {}
    with concurrent.futures.ProcessPoolExecutor() as pool:
        for name, got in pool.map(_run_mutation, range(len(MUTATIONS)), chunksize=4):
            results[name] = got
    for name, muts, expected in MUTATIONS:
        expected = _expected_flips(name, expected)
        got = results[name]
        if isinstance(got, Exception):
            exc = got
            print(f"  FAIL  {name}: the mutated module raised {type(exc).__name__}: {exc}")
            rc = 1
            continue
        flipped = {k for k in baseline if k not in got or not got[k]}
        unknown = set(expected) - set(baseline)
        missing = set(expected) - flipped
        extra = flipped - set(expected)
        if unknown:
            print(f"  FAIL  {name}: names no existing check {sorted(unknown)}")
            rc = 1
        elif not flipped:
            print(f"  FAIL  {name}: flipped NOTHING -- these checks can never fail")
            rc = 1
        elif missing or extra:
            print(f"  FAIL  {name}: missing {sorted(missing)} extra {sorted(extra)}")
            rc = 1
        else:
            print(f"  PASS  {name}: flips exactly {len(flipped)} declared check(s)")

    print()
    undefended = _coverage(baseline)
    if undefended:
        print(f"  FAIL  {len(undefended)} check(s) no mutation defends -- unproven:")
        for name in undefended:
            print(f"          {name}")
        rc = 1
    else:
        print(f"  PASS  coverage: all {len(baseline)} checks are defended by a mutation")
    for name, reason in UNPROVABLE.items():
        print(f"  NOTE  unprovable: {name} -- {reason}")

    print()
    print("COUNTER-PROOF PASSED" if rc == 0 else "COUNTER-PROOF FAILED")
    return rc


def main() -> int:
    torch = _load_torch()
    if "--self-test" in sys.argv[1:]:
        return _self_test(torch)
    results = build_checks(load_autodr(), torch)
    failures = []
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)
    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print(f"ALL {len(results)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
