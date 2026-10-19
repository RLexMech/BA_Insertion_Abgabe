# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Validate evaluation-table identity, coverage and applied conditions.

Read applied.csv, applied_meta.json, episodes.csv, eval_manifest.json,
the committed table and its JSON sidecar.
G1: hashes must agree across these files, all log reports and --filehash.
Without external log/filehash evidence G1 is incomplete, never PASS.
G2: exactly one episode==0 per env, N table rows/envs, all first episodes done.
G3: independently map table rows with float64 math, tolerance 1e-6; compare
configured and recorded ranges. Noise tags pn2/gx2/py2 double their named
widths (D-191, D-206). Values outside [0,1] require matching OOD/wide
sidecar authorisation and run switch (D-227).

The mapping does not reuse environment math. Stdlib-only; --self-test
uses synthetic runs.

Usage:
    python scripts/check_eval_conditions.py --self-test
    python scripts/check_eval_conditions.py --run-dir <eval folder> ^
        --table scripts/eval/eval_table_seed20_c18.csv --log rt_logs/RT-N/RT-N.txt --filehash <HEX>
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import pathlib
import re
import sys

SCRIPT_MARKER = "check_eval_conditions-2026-10-09a"

# The table layout, the check's own literal (autodr.TABLE_COLUMNS).
TABLE_COLUMNS = (
    "lat_r", "lat_phi", "yaw", "tilt", "tilt_azimuth", "start_height", "friction",
    "fixture_noise_x", "fixture_noise_y",
    "joint_noise_0", "joint_noise_1", "joint_noise_2", "joint_noise_3",
    "joint_noise_4", "joint_noise_5", "grasp_obs_x", "pocket_obs_yaw",
    "fixture_noise_z",
)
# applied.csv, the check's own literal (insertion_env.APPLIED_LOG_COLUMNS; L-06:
# the list comes from the function that WRITES the file).
APPLIED_COLUMNS = (
    "row_id", "lat_x_m", "lat_y_m", "start_height_m", "yaw_rad",
    "pocket_qw", "pocket_qx", "pocket_qy", "pocket_qz",
    "pocket_dx_m", "pocket_dy_m", "pocket_dz_m", "grasp_obs_x_m", "pocket_obs_yaw_rad",
    "friction_cmd", "friction_robot_min", "friction_robot_max",
    "friction_fixture_min", "friction_fixture_max",
)
FRICTION_READBACKS = APPLIED_COLUMNS[15:]

# Parameter sources ( autodr.DR_DIMS, D-178/179/181). `nominal`:
# everything but the start height sits on its centre; friction centre 0.140 is
# D-111's CONTACT_FRICTION, read back in RT-207s1.
BANDS = {
    "target": {
        "lat_r": (0.0, 0.030),
        "yaw": (-math.radians(5.0), math.radians(5.0)),
        "tilt": (0.0, math.radians(10.0)),
        "start_height": (0.020, 0.270),
        "friction": (0.08, 0.20),
    },
    "nominal": {
        "lat_r": (0.0, 0.0),
        "yaw": (0.0, 0.0),
        "tilt": (0.0, 0.0),
        "start_height": (0.020, 0.270),
        "friction": (0.140, 0.140),
    },
}
FIXTURE_NOISE_XY_M = 0.05    # cfg fixture_pos_noise_xy (D-203/D-204)
FIXTURE_NOISE_Z_M = 0.05     # cfg fixture_pos_noise_z (D-204 (4))
GRASP_OBS_X_M = 0.003        # D-183
POCKET_OBS_YAW_RAD = 0.0223402   # D-194, Belege_Streuwerte B6 (the run's value)
POCKET_OBS_NOISE_M = 0.005   # cfg obs_noise_pocket_pos_std_m (uniform half width, Belege B4a)
# D-191 (3) / D-206 (3): the observation-noise blocks, x2 of ONE training value
# each (IndustReal Table X; Belege_Streuwerte F3). The run's --eval-tag names the
# block. `scripts/rl_games/run_eval.ps1` `$NoiseOverride` hands the same values to
# the env; the self-test holds the two against each other.
NOISE_TAGS = {
    "pn2": {"obs_noise_pocket_pos_std_m": 2.0 * POCKET_OBS_NOISE_M},
    "gx2": {"grasp_obs_offset_x_m": 2.0 * GRASP_OBS_X_M},
    "py2": {"pocket_obs_yaw_rad": 2.0 * POCKET_OBS_YAW_RAD},
}
TOL = 1e-6                   # eval_plan.md G3
TOL_BAND = 1e-9              # the env's bands are float64 arithmetic on the same numbers


def expected_noise(tag) -> dict:
    """The three observation-noise widths a run with this tag must have used."""
    out = {"obs_noise_pocket_pos_std_m": POCKET_OBS_NOISE_M, "grasp_obs_offset_x_m": GRASP_OBS_X_M,
           "pocket_obs_yaw_rad": POCKET_OBS_YAW_RAD}
    out.update(NOISE_TAGS.get(tag or "", {}))
    return out

CHECK_NAMES = (
    "G1a table bytes = sidecar = manifest = applied_meta",
    "G1b the run log's [eval-table] line carries the same hash",
    "G1c the training PC's Get-FileHash is the same hash",
    "G2a one episode-0 row per env id 0..N-1",
    "G2b N = table rows = manifest num_envs = envs that finished",
    "G3a applied.csv has the agreed columns and one row per table row, in order",
    "G3b the env's bands are the plan's bands for this width",
    "G3c every applied value = the table row mapped, within 1e-6",
    "G3d friction read back from PhysX = commanded, on both assets",
    # D-206 (3), from p6-TestAusserhalb 019ae3d8: a plain table passes this
    # one by construction.
    "G3e values outside [0, 1] only where an OOD sidecar nails or a wide sidecar stretches, and the run's switch matches",
)


# The table kinds whose nailed cells may leave [0, 1] (make_eval_table.py):
# ``ood`` (D-206 (3)) and ``sweep`` (RT-264), one sidecar schema.
OUTSIDE_KINDS = ("ood", "sweep")


def ood_cells(sidecar: dict) -> dict:
    """``{(row, column): unit}`` an OOD or sweep sidecar nails; {} for any other kind."""
    if sidecar.get("kind") not in OUTSIDE_KINDS:
        return {}
    per = int(sidecar.get("rows_per_block") or 0)
    out = {}
    for b in sidecar.get("blocks", []):
        for r in range(int(b["block"]) * per, (int(b["block"]) + 1) * per):
            out[(r, int(b["column"]))] = float(b["nailed_to"])
    return out


def wide_ranges(sidecar: dict) -> dict:
    """``{column: (unit_lo, unit_hi)}`` a wide sidecar stretches (D-227,
    ``make_eval_table.py --kind wide``); {} for any other kind."""
    if sidecar.get("kind") != "wide":
        return {}
    return {int(c["column"]): (float(c["unit_lo"]), float(c["unit_hi"])) for c in sidecar.get("columns", [])}


def _lerp(u: float, lo: float, hi: float) -> float:
    return lo + u * (hi - lo)


def map_row(u: dict, width: str, grasp_x_m: float = GRASP_OBS_X_M,
            pocket_yaw_rad: float = POCKET_OBS_YAW_RAD) -> dict:
    """One table row (unit values by column name) -> the applied.csv values.
    ``grasp_x_m`` / ``pocket_yaw_rad`` are the belief-error half widths the run
    is EXPECTED to use (``expected_noise``; x2 under gx2 / py2)."""
    b = BANDS[width]
    r = b["lat_r"][1] * math.sqrt(u["lat_r"])
    phi = 2.0 * math.pi * u["lat_phi"]
    yaw = _lerp(u["yaw"], *b["yaw"])
    half = 0.5 * _lerp(u["tilt"], *b["tilt"])
    az = 2.0 * math.pi * u["tilt_azimuth"]
    c, s = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    a, bx, by = math.cos(half), math.cos(az) * math.sin(half), math.sin(az) * math.sin(half)
    # Hamilton product (c, 0, 0, s) * (a, bx, by, 0), wxyz, then w >= 0.
    q = [c * a, c * bx - s * by, c * by + s * bx, s * a]
    if q[0] < 0.0:
        q = [-v for v in q]
    mu = _lerp(u["friction"], *b["friction"])
    out = {
        "lat_x_m": r * math.cos(phi), "lat_y_m": r * math.sin(phi),
        "start_height_m": _lerp(u["start_height"], *b["start_height"]),
        "yaw_rad": yaw,
        "pocket_qw": q[0], "pocket_qx": q[1], "pocket_qy": q[2], "pocket_qz": q[3],
        "pocket_dx_m": (2.0 * u["fixture_noise_x"] - 1.0) * FIXTURE_NOISE_XY_M,
        "pocket_dy_m": (2.0 * u["fixture_noise_y"] - 1.0) * FIXTURE_NOISE_XY_M,
        "pocket_dz_m": (2.0 * u["fixture_noise_z"] - 1.0) * FIXTURE_NOISE_Z_M,
        "grasp_obs_x_m": (2.0 * u["grasp_obs_x"] - 1.0) * grasp_x_m,
        "pocket_obs_yaw_rad": (2.0 * u["pocket_obs_yaw"] - 1.0) * pocket_yaw_rad,
        "friction_cmd": mu,
    }
    return out


def judge(table, sidecar, table_sha, log_shas, filehash, ep_header, ep_rows,
          ap_header, ap_rows, meta, manifest) -> dict:
    """``{check name: (ok, detail)}`` over CHECK_NAMES. ``ok`` is None for a
    check whose outside evidence was not handed over (INCOMPLETE).

    The noise widths are EXPECTED from the manifest's tag (``expected_noise``).
    G3b holds the run's records against them: grasp and pocket yaw in
    ``applied_meta.json`` AND in the manifest's ``noise``, pocket noise in the
    manifest's ``noise`` (torch draws it, no applied column). G3c maps the table
    with the expected grasp and yaw widths. So a x2 run passes only under its
    own tag (D-191 (3), D-206 (3))."""
    want_noise = expected_noise(manifest.get("tag"))
    grasp_x_m = want_noise["grasp_obs_offset_x_m"]
    yaw_rad = want_noise["pocket_obs_yaw_rad"]
    out: dict = {}
    n = len(table)
    width = str(manifest.get("width"))

    hashes = {
        "sidecar": sidecar.get("sha256"),
        "manifest": manifest.get("table_sha256"),
        "applied_meta": (meta.get("table") or {}).get("sha256"),
    }
    out[CHECK_NAMES[0]] = (all(h == table_sha for h in hashes.values()),
                           f"table {table_sha[:12]}.. vs {hashes}")
    if log_shas is None:
        out[CHECK_NAMES[1]] = (None, "no --log given")
    else:
        # One line per startup report (`report_at_steps`, RT-209: four). At
        # least one, and EVERY one the table's hash -- never "any".
        out[CHECK_NAMES[1]] = (bool(log_shas) and all(h == table_sha for h in log_shas),
                               f"{len(log_shas)} [eval-table] line(s): {[h[:12] for h in log_shas]}")
    if filehash is None:
        out[CHECK_NAMES[2]] = (None, "no --filehash given")
    else:
        out[CHECK_NAMES[2]] = (filehash.strip().lower() == table_sha, filehash.strip().lower()[:12])

    first = [r for r in ep_rows if r.get("episode") == "0"]
    ids = sorted(int(r["env_id"]) for r in first) if "env_id" in ep_header else []
    out[CHECK_NAMES[3]] = (ids == list(range(n)),
                           f"{len(first)} episode-0 rows, {len(set(ids))} distinct ids, table {n}")
    counts = {"table": n, "sidecar": sidecar.get("n_rows"), "num_envs": manifest.get("num_envs"),
              "table_rows": manifest.get("table_rows"),
              "finished": manifest.get("envs_finished_first_episode")}
    out[CHECK_NAMES[4]] = (all(v == n for v in counts.values()), str(counts))

    shape_ok = (tuple(ap_header) == APPLIED_COLUMNS and len(ap_rows) == n
                and [r.get("row_id") for r in ap_rows] == [str(i) for i in range(n)])
    out[CHECK_NAMES[5]] = (shape_ok, f"{len(ap_rows)} rows, header ok {tuple(ap_header) == APPLIED_COLUMNS}")

    want_b = BANDS.get(width)
    got_b = meta.get("bounds") or {}
    band_ok = want_b is not None and set(got_b) == set(want_b) and all(
        abs(got_b[k][i] - want_b[k][i]) <= TOL_BAND for k in want_b for i in (0, 1))
    noise = manifest.get("noise") or {}
    run_noise = [("applied_meta grasp", meta.get("grasp_obs_offset_x_m"), grasp_x_m),
                 ("applied_meta yaw", meta.get("pocket_obs_yaw_rad"), yaw_rad),
                 ("manifest grasp", noise.get("grasp_obs_offset_x_m"), grasp_x_m),
                 ("manifest yaw", noise.get("pocket_obs_yaw_rad"), yaw_rad),
                 ("manifest pocket", noise.get("obs_noise_pocket_pos_std_m"),
                  want_noise["obs_noise_pocket_pos_std_m"])]
    band_ok = band_ok and abs(float(meta.get("fixture_pos_noise_xy_m", -1)) - FIXTURE_NOISE_XY_M) <= TOL_BAND \
        and abs(float(meta.get("fixture_pos_noise_z_m", -1)) - FIXTURE_NOISE_Z_M) <= TOL_BAND \
        and all(v is not None and abs(float(v) - w) <= TOL_BAND for _, v, w in run_noise)
    out[CHECK_NAMES[6]] = (band_ok, f"width {width!r}, tag {manifest.get('tag')!r}, noise (name, run, expected) "
                                    f"{run_noise}: {got_b}")

    worst, worst_at, bad = 0.0, "", 0
    fr_worst, fr_at, fr_bad = 0.0, "", 0
    if shape_ok and want_b is not None:
        for i, (u_row, a) in enumerate(zip(table, ap_rows)):
            want = map_row(dict(zip(TABLE_COLUMNS, u_row)), width, grasp_x_m, yaw_rad)
            for k, w in want.items():
                d = abs(float(a[k]) - w)
                if d > worst:
                    worst, worst_at = d, f"row {i} {k}"
                bad += d > TOL
            for k in FRICTION_READBACKS:
                d = abs(float(a[k]) - want["friction_cmd"])
                if d > fr_worst:
                    fr_worst, fr_at = d, f"row {i} {k}"
                fr_bad += d > TOL
    mapped = shape_ok and want_b is not None
    out[CHECK_NAMES[7]] = (mapped and bad == 0,
                           f"{bad} value(s) over {TOL:g}; worst {worst:.3g} at {worst_at}")
    out[CHECK_NAMES[8]] = (mapped and fr_bad == 0,
                           f"{fr_bad} read-back(s) over {TOL:g}; worst {fr_worst:.3g} at {fr_at}")

    # G3e: every cell outside [0, 1] is one the OOD sidecar nails, with its
    # value, or lies in a column a wide sidecar stretches, inside its range
    # (D-227); every nailed cell holds it; and the env ran with the switch ON
    # exactly when the table is an OOD, sweep or wide table (applied_meta, env
    # cfg dr_table_allow_outside). A record without the key reads as off.
    nailed = ood_cells(sidecar)
    wide = wide_ranges(sidecar)

    def _held(i, j, v):
        if nailed.get((i, j)) == v:
            return True
        return j in wide and wide[j][0] <= v <= wide[j][1]

    stray = [(i, j, v) for i, row in enumerate(table) for j, v in enumerate(row)
             if not 0.0 <= v <= 1.0 and not _held(i, j, v)]
    unheld = [k for k, v in nailed.items() if k[0] >= n or table[k[0]][k[1]] != v]
    switch = bool((meta.get("table") or {}).get("allow_outside", False))
    out[CHECK_NAMES[9]] = (not stray and not unheld and switch == bool(nailed or wide),
                           f"kind {sidecar.get('kind')!r}, {len(nailed)} nailed cells, "
                           f"{len(wide)} stretched columns, "
                           f"{len(stray)} stray, {len(unheld)} not held, switch {switch}")
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_csv(path: pathlib.Path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        return header, [dict(zip(header, line)) for line in reader if line]


def load_table(path: pathlib.Path):
    raw = path.read_bytes()
    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    if tuple(h.strip() for h in lines[0].split(",")) != TABLE_COLUMNS:
        raise SystemExit(f"[check_eval_conditions] {path} header is not TABLE_COLUMNS")
    table = [[float(v) for v in ln.split(",")] for ln in lines[1:]]
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return table, sidecar, hashlib.sha256(raw).hexdigest()


def log_hashes(path: pathlib.Path) -> list:
    text = path.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"\[eval-table\].*?sha256 ([0-9a-f]{64})", text)


# ---------------------------------------------------------------------------
# Self-test: a built run, each broken one way (D-080)
# ---------------------------------------------------------------------------

def _good(width: str = "target"):
    n = 6
    # Deterministic unit values in every TABLE_COLUMNS column, 0 and 1 included.
    table = [[((7 * i + 3 * j) % 11) / 10.0 for j in range(len(TABLE_COLUMNS))] for i in range(n)]
    sha = "c3" * 32
    sidecar = {"sha256": sha, "n_rows": n}
    ap_rows = []
    for i, u_row in enumerate(table):
        want = map_row(dict(zip(TABLE_COLUMNS, u_row)), width)
        row = {"row_id": str(i)}
        row.update({k: f"{v:.9g}" for k, v in want.items()})
        row.update({k: f"{want['friction_cmd']:.9g}" for k in FRICTION_READBACKS})
        ap_rows.append(row)
    ep_rows = [{"env_id": str(i), "episode": "0"} for i in range(n)]
    ep_rows += [{"env_id": "2", "episode": "1"}]          # a G4 repeat, not counted
    meta = {"table": {"sha256": sha}, "bounds": {k: list(v) for k, v in BANDS[width].items()},
            "fixture_pos_noise_xy_m": FIXTURE_NOISE_XY_M, "fixture_pos_noise_z_m": FIXTURE_NOISE_Z_M,
            "grasp_obs_offset_x_m": GRASP_OBS_X_M,
            "pocket_obs_yaw_rad": POCKET_OBS_YAW_RAD}
    manifest = {"width": width, "table_sha256": sha, "num_envs": n, "table_rows": n,
                "envs_finished_first_episode": n, "tag": "",
                "noise": {"grasp_obs_offset_x_m": GRASP_OBS_X_M, "pocket_obs_yaw_rad": POCKET_OBS_YAW_RAD,
                          "obs_noise_pocket_pos_std_m": POCKET_OBS_NOISE_M}}
    meta["table"]["allow_outside"] = False
    return dict(table=table, sidecar=sidecar, table_sha=sha, log_shas=[sha], filehash=sha.upper(),
                ep_header=["env_id", "episode"], ep_rows=ep_rows,
                ap_header=list(APPLIED_COLUMNS), ap_rows=ap_rows, meta=meta, manifest=manifest)


def _bump(row: int, key: str, delta: float):
    def fn(g):
        g["ap_rows"][row][key] = f"{float(g['ap_rows'][row][key]) + delta:.9g}"
    return fn


def _shift_rows(g):
    """The plan's row mutation on the file side: env i carries row i+1's values."""
    vals = [dict(r) for r in g["ap_rows"]]
    for i, r in enumerate(g["ap_rows"]):
        nxt = vals[(i + 1) % len(vals)]
        r.update({k: v for k, v in nxt.items() if k != "row_id"})


def _rebuild_applied(g, width: str = "target", grasp_x_m: float = GRASP_OBS_X_M,
                     yaw_rad: float = POCKET_OBS_YAW_RAD):
    """applied.csv as the env would write it for the CURRENT table -- so a
    case that edits the table isolates G3e instead of also tripping G3c."""
    for i, u_row in enumerate(g["table"]):
        want = map_row(dict(zip(TABLE_COLUMNS, u_row)), width, grasp_x_m, yaw_rad)
        g["ap_rows"][i].update({k: f"{v:.9g}" for k, v in want.items()})
        g["ap_rows"][i].update({k: f"{want['friction_cmd']:.9g}" for k in FRICTION_READBACKS})


def _make_ood(g):
    """A legal OOD run: two blocks of three rows, lat_r nailed to 2.25 (x1.5
    radius), yaw nailed to -0.25 (x1.5 lower edge), the switch on."""
    g["sidecar"].update(kind="ood", rows_per_block=3, blocks=[
        {"block": 0, "key": "lat_r_hi", "column": 0, "nailed_to": 2.25},
        {"block": 1, "key": "yaw_lo", "column": 2, "nailed_to": -0.25}])
    for i in range(3):
        g["table"][i][0] = 2.25
        g["table"][3 + i][2] = -0.25
    _rebuild_applied(g)
    g["meta"]["table"]["allow_outside"] = True


def _ood_then(fn):
    def both(g):
        _make_ood(g)
        fn(g)
    return both


def _make_wide(g):
    """A legal wide run (D-227): lat_r and yaw stretched to their x1.5 edges,
    one value outside [0, 1] in each, the switch on."""
    g["sidecar"].update(kind="wide", columns=[
        {"key": "lat_r", "column": 0, "unit_lo": 0.0, "unit_hi": 2.25},
        {"key": "yaw", "column": 2, "unit_lo": -0.25, "unit_hi": 1.25}])
    g["table"][0][0] = 2.0
    g["table"][1][2] = -0.2
    _rebuild_applied(g)
    g["meta"]["table"]["allow_outside"] = True


def _wide_then(fn):
    def both(g):
        _make_wide(g)
        fn(g)
    return both


def _stray_cell(g):
    g["table"][0][3] = 1.3      # tilt beyond the band in a cell the sidecar does not nail
    _rebuild_applied(g)


def _grasp_run(width_m: float, applied_m: float | None = None):
    """A run whose env used grasp half width ``width_m`` (both records say so);
    its applied.csv mapped with ``applied_m`` (default: the same width)."""
    def fn(g):
        g["meta"]["grasp_obs_offset_x_m"] = width_m
        g["manifest"]["noise"]["grasp_obs_offset_x_m"] = width_m
        _rebuild_applied(g, grasp_x_m=width_m if applied_m is None else applied_m)
    return fn


def _yaw_run(width_rad: float, applied_rad: float | None = None):
    """The py2 twin of ``_grasp_run``: pocket-yaw belief half width ``width_rad``."""
    def fn(g):
        g["meta"]["pocket_obs_yaw_rad"] = width_rad
        g["manifest"]["noise"]["pocket_obs_yaw_rad"] = width_rad
        _rebuild_applied(g, yaw_rad=width_rad if applied_rad is None else applied_rad)
    return fn


def _tag(fn, tag: str):
    """The same run with the manifest tag ``tag``."""
    def both(g):
        fn(g)
        g["manifest"]["tag"] = tag
    return both


def _pocket(width_m: float):
    return lambda g: g["manifest"]["noise"].update(obs_noise_pocket_pos_std_m=width_m)


CASES = (
    # D-206 (3), from p6-TestAusserhalb 019ae3d8: G3e.
    ("ood-table-with-its-sidecar-and-switch", _make_ood, ()),
    ("ood-value-in-a-cell-the-sidecar-does-not-nail", _ood_then(_stray_cell), ("G3e",)),
    ("ood-run-without-the-switch",
     _ood_then(lambda g: g["meta"]["table"].update(allow_outside=False)), ("G3e",)),
    ("ood-nailed-cell-not-held",
     _ood_then(lambda g: (g["table"][1].__setitem__(0, 1.0), _rebuild_applied(g))), ("G3e",)),
    ("plain-table-with-the-switch-on", lambda g: g["meta"]["table"].update(allow_outside=True), ("G3e",)),
    # RT-264: a sweep sidecar nails like an OOD one; an unknown kind nails nothing.
    ("sweep-table-with-its-sidecar-and-switch", lambda g: (_make_ood(g), g["sidecar"].update(kind="sweep")), ()),
    ("outside-values-under-an-unknown-kind", lambda g: (_make_ood(g), g["sidecar"].update(kind="other")), ("G3e",)),
    ("plain-table-with-a-value-outside", _stray_cell, ("G3e",)),
    # D-227: a wide sidecar allows its stretched columns, inside their range.
    ("wide-table-with-its-sidecar-and-switch", _make_wide, ()),
    ("wide-value-beyond-its-column-range",
     _wide_then(lambda g: (g["table"][2].__setitem__(0, 2.3), _rebuild_applied(g))), ("G3e",)),
    ("wide-value-in-a-column-not-stretched", _wide_then(_stray_cell), ("G3e",)),
    ("wide-run-without-the-switch",
     _wide_then(lambda g: g["meta"]["table"].update(allow_outside=False)), ("G3e",)),
    # D-191 (3) (a766ae9e) and D-206 (3): the observation-noise x2 blocks. The
    # expected widths come from the manifest tag and are read back from the
    # run's records.
    ("gx2-run-tagged-gx2", _tag(_grasp_run(2.0 * GRASP_OBS_X_M), "gx2"), ()),
    ("gx2-width-without-the-gx2-tag", _grasp_run(2.0 * GRASP_OBS_X_M), ("G3b", "G3c")),
    ("gx2-tag-but-the-run-used-d183", _tag(lambda g: None, "gx2"), ("G3b", "G3c")),
    ("manifest-records-another-grasp-width",
     lambda g: g["manifest"]["noise"].update(grasp_obs_offset_x_m=2.0 * GRASP_OBS_X_M), ("G3b",)),
    ("manifest-has-no-noise-record", lambda g: g["manifest"].pop("noise"), ("G3b",)),
    ("gx2-records-but-applied-scaled-by-d183",
     _tag(_grasp_run(2.0 * GRASP_OBS_X_M, GRASP_OBS_X_M), "gx2"), ("G3c",)),
    ("pn2-run-tagged-pn2", _tag(_pocket(2.0 * POCKET_OBS_NOISE_M), "pn2"), ()),
    ("pn2-tag-but-pocket-noise-not-doubled", _tag(lambda g: None, "pn2"), ("G3b",)),
    ("doubled-pocket-noise-without-the-pn2-tag", _pocket(2.0 * POCKET_OBS_NOISE_M), ("G3b",)),
    ("py2-run-tagged-py2", _tag(_yaw_run(2.0 * POCKET_OBS_YAW_RAD), "py2"), ()),
    ("py2-width-without-the-py2-tag", _yaw_run(2.0 * POCKET_OBS_YAW_RAD), ("G3b", "G3c")),
    ("py2-tag-but-the-run-used-d194", _tag(lambda g: None, "py2"), ("G3b", "G3c")),
    ("manifest-records-another-yaw-width",
     lambda g: g["manifest"]["noise"].update(pocket_obs_yaw_rad=2.0 * POCKET_OBS_YAW_RAD), ("G3b",)),
    ("py2-records-but-applied-scaled-by-d194",
     _tag(_yaw_run(2.0 * POCKET_OBS_YAW_RAD, POCKET_OBS_YAW_RAD), "py2"), ("G3c",)),
    # The plan's own mutation for D1: one value + 1e-3.
    ("one-value-plus-1e-3", _bump(3, "start_height_m", 1e-3), ("G3c",)),
    ("a-quaternion-component-moves", _bump(1, "pocket_qx", 1e-4), ("G3c",)),
    ("every-env-plays-the-next-row", _shift_rows, ("G3c", "G3d")),
    ("fixture-friction-did-not-land", _bump(4, "friction_fixture_max", 0.01), ("G3d",)),
    ("applied-loses-a-row", lambda g: g["ap_rows"].pop(), ("G3a", "G3c", "G3d")),
    ("applied-columns-swapped",
     lambda g: g["ap_header"].__setitem__(slice(1, 3), ["lat_y_m", "lat_x_m"]), ("G3a", "G3c", "G3d")),
    ("env-ran-another-band", lambda g: g["meta"]["bounds"]["lat_r"].__setitem__(1, 0.006), ("G3b",)),
    ("manifest-names-another-table", lambda g: g["manifest"].update(table_sha256="00" * 32), ("G1a",)),
    ("log-line-carries-another-hash", lambda g: g.update(log_shas=["00" * 32]), ("G1b",)),
    ("log-has-no-eval-table-line", lambda g: g.update(log_shas=[]), ("G1b",)),
    # RT-209: the env prints its startup report at every `report_at_steps`
    # entry, four by default, each with its own [eval-table] line. Legal.
    ("report-prints-four-times", lambda g: g.update(log_shas=[g["table_sha"]] * 4), ()),
    ("one-of-four-lines-carries-another-hash",
     lambda g: g.update(log_shas=[g["table_sha"]] * 2 + ["00" * 32, g["table_sha"]]), ("G1b",)),
    ("filehash-differs", lambda g: g.update(filehash="00" * 32), ("G1c",)),
    ("an-env-has-no-first-episode", lambda g: g["ep_rows"].pop(0), ("G2a",)),
    ("an-env-id-twice", lambda g: g["ep_rows"][1].update(env_id="0"), ("G2a",)),
    ("one-env-never-finished", lambda g: g["manifest"].update(envs_finished_first_episode=5), ("G2b",)),
    # D-194: the evaluation ran without `env.pocket_obs_yaw_rad` (default 0.0 = off).
    ("yaw-belief-error-was-off",
     lambda g: [r.update(pocket_obs_yaw_rad="0") for r in g["ap_rows"]], ("G3c",)),
    # ... and the env says so itself, in applied_meta.json.
    ("env-says-yaw-belief-error-off", lambda g: g["meta"].update(pocket_obs_yaw_rad=0.0), ("G3b",)),
)


def _applied_columns_vs_env() -> tuple[bool, str]:
    """``APPLIED_COLUMNS`` against the tuple the env WRITES,
    ``insertion_env.APPLIED_LOG_COLUMNS``, read as AST (the env needs Isaac).
    Only the column NAMES are shared; the mapping stays the check's own."""
    env_py = (pathlib.Path(__file__).resolve().parents[1] / "source" / "insertion" / "insertion"
              / "tasks" / "direct" / "insertion" / "insertion_env.py")
    for node in ast.parse(env_py.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "APPLIED_LOG_COLUMNS" for t in node.targets):
            env_cols = ast.literal_eval(node.value)
            return env_cols == APPLIED_COLUMNS, f"env {len(env_cols)}, check {len(APPLIED_COLUMNS)} columns"
    return False, "APPLIED_LOG_COLUMNS not found in insertion_env.py"


def _yaw_belief_example() -> tuple[bool, str]:
    """D-194 worked example, not recomputed: Belege_Streuwerte B6 says uniform in
    +-0.0223402 rad, and the env maps the column like ``grasp_obs_x``
    (``insertion_env._reset_idx``). Unit 0 / 0.5 / 1 -> lower edge / 0 / upper
    edge, at every width (a belief error has no band)."""
    base = {k: 0.5 for k in TABLE_COLUMNS}
    want = [-0.0223402, 0.0, 0.0223402] * len(BANDS)
    got = [map_row(dict(base, pocket_obs_yaw=u), w).get("pocket_obs_yaw_rad")
           for w in BANDS for u in (0.0, 0.5, 1.0)]
    ok = all(g is not None and abs(g - x) <= TOL_BAND for g, x in zip(got, want))
    return ok, f"got {got}"


def _noise_tags_vs_run_eval() -> tuple[bool, str]:
    """``NOISE_TAGS`` against the Hydra overrides ``scripts/rl_games/run_eval.ps1``
    hands the env (``$NoiseOverride``, read as text): the widths the check
    EXPECTS must be the ones the runs GET."""
    ps1 = pathlib.Path(__file__).resolve().parent / "rl_games" / "run_eval.ps1"
    got = {t: {k: float(v)} for t, k, v in re.findall(
        r"^\s*(\w+)\s*=\s*'env\.(\w+)=([-+0-9.eE]+)'\s*$", ps1.read_text(encoding="utf-8"), re.M)}
    ok = set(got) == set(NOISE_TAGS) and all(
        set(got[t]) == set(NOISE_TAGS[t]) and all(abs(got[t][k] - NOISE_TAGS[t][k]) <= TOL_BAND
                                                  for k in got[t]) for t in got)
    return ok, f"run_eval.ps1 {got}"


_ORACLE_CACHE: dict = {}


def _oracle_vs_production(table_name: str = "eval_table_seed20_c18.csv",
                          n_rows: int = 1000) -> tuple[bool, str]:
    """``map_row`` against the functions the env really calls, on the committed
    seed-20 table with the shape the env hands over, (1000, 18) (L-14).

    The baseline above builds its applied.csv WITH ``map_row``, so it cannot see
    a wrong oracle. This can: ``autodr.map_unit_to_bounds``,
    ``insertion_math.disk_offset`` and ``insertion_math.quat_mul`` run here under
    ``torch_shim`` (or real torch) and must agree within ``TOL``. What stays
    unproven offline is only that the ENV calls them as ``_reset_idx`` reads.
    """
    import importlib.util

    repo = pathlib.Path(__file__).resolve().parents[1]
    pkg = repo / "source" / "insertion" / "insertion" / "tasks" / "direct" / "insertion"

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Loaded ONCE per process (p6 019ae3d8): a second torch_shim.load() finds
    # the shim already registered as `torch` and fails.
    if "production" not in _ORACLE_CACHE:
        torch, _ = _load("torch_shim", repo / "scripts" / "tools" / "torch_shim.py").load()
        sys.modules.setdefault("torch", torch)
        # autodr.py has a relative import; check_autodr.py owns the scaffolding for it.
        autodr = _load("check_autodr", repo / "scripts" / "check_autodr.py").load_autodr()
        ns: dict = {"torch": torch, "math": math, "__name__": "insertion_math"}
        exec(compile((pkg / "insertion_math.py").read_text(encoding="utf-8"), "insertion_math.py", "exec"), ns)  # noqa: S102
        _ORACLE_CACHE["production"] = (torch, autodr, ns)
    torch, autodr, ns = _ORACLE_CACHE["production"]

    table, _, _ = load_table(repo / "scripts" / "eval" / table_name)
    if tuple(autodr["TABLE_COLUMNS"]) != TABLE_COLUMNS:
        return False, "TABLE_COLUMNS differs from autodr.TABLE_COLUMNS"
    col = {k: [row[i] for row in table] for i, k in enumerate(TABLE_COLUMNS)}
    b = BANDS["target"]
    t = torch.tensor
    lat = ns["disk_offset"](t(col["lat_r"]), t(col["lat_phi"]), float(b["lat_r"][1])).tolist()
    prod = {k: autodr["map_unit_to_bounds"](t(col[k]), float(b[k][0]), float(b[k][1])).tolist()
            for k in ("yaw", "tilt", "start_height", "friction")}
    q_yaw = [[math.cos(0.5 * y), 0.0, 0.0, math.sin(0.5 * y)] for y in prod["yaw"]]
    q_tilt = [[math.cos(0.5 * m), math.cos(2.0 * math.pi * a) * math.sin(0.5 * m),
               math.sin(2.0 * math.pi * a) * math.sin(0.5 * m), 0.0]
              for m, a in zip(prod["tilt"], col["tilt_azimuth"])]
    quat = ns["quat_mul"](t(q_yaw), t(q_tilt)).tolist()
    worst, at = 0.0, ""
    for i, row in enumerate(table):
        want = map_row(dict(zip(TABLE_COLUMNS, row)), "target")
        got = {"lat_x_m": lat[i][0], "lat_y_m": lat[i][1], "yaw_rad": prod["yaw"][i],
               "start_height_m": prod["start_height"][i], "friction_cmd": prod["friction"][i],
               "pocket_qw": quat[i][0], "pocket_qx": quat[i][1],
               "pocket_qy": quat[i][2], "pocket_qz": quat[i][3]}
        for k, v in got.items():
            if abs(v - want[k]) > worst:
                worst, at = abs(v - want[k]), f"row {i} {k}"
    return len(table) == n_rows and worst <= TOL, f"{len(table)} rows, worst {worst:.3g} at {at}"


def self_test() -> int:
    print(f"[check_eval_conditions] marker: {SCRIPT_MARKER}")
    bad = 0
    ok, detail = _oracle_vs_production()
    print(f"[check_eval_conditions]   oracle = production math on the seed-20 table: "
          f"{'PASS' if ok else 'FAIL'} ({detail})")
    bad += not ok
    # D-206 (3): the same comparison where the unit values leave [0, 1].
    ok, detail = _oracle_vs_production("eval_table_ood_seed10_c18.csv", 1050)
    print(f"[check_eval_conditions]   oracle = production math on the OOD table: "
          f"{'PASS' if ok else 'FAIL'} ({detail})")
    bad += not ok
    # D-227: and on the wide table (four columns stretched at once).
    ok, detail = _oracle_vs_production("eval_table_wide_seed10_c18.csv", 1000)
    print(f"[check_eval_conditions]   oracle = production math on the wide table: "
          f"{'PASS' if ok else 'FAIL'} ({detail})")
    bad += not ok
    ok, detail = _noise_tags_vs_run_eval()
    print(f"[check_eval_conditions]   NOISE_TAGS = rl_games/run_eval.ps1 $NoiseOverride: "
          f"{'PASS' if ok else 'FAIL'} ({detail})")
    bad += not ok
    ok, detail = _yaw_belief_example()
    print(f"[check_eval_conditions]   pocket-yaw belief error, worked example: "
          f"{'PASS' if ok else 'FAIL'} ({detail})")
    bad += not ok
    ok, detail = _applied_columns_vs_env()
    print(f"[check_eval_conditions]   APPLIED_COLUMNS = insertion_env.APPLIED_LOG_COLUMNS: "
          f"{'PASS' if ok else 'FAIL'} ({detail})")
    bad += not ok
    for width in BANDS:
        base = judge(**_good(width))
        red = [k for k, (ok, _) in base.items() if ok is not True]
        ok = not red and tuple(base) == CHECK_NAMES
        print(f"[check_eval_conditions]   baseline {width}: {'PASS' if ok else 'FAIL ' + str(red)}")
        bad += not ok
    g = _good()
    g.update(log_shas=None, filehash=None)
    inc = [k.split(" ", 1)[0] for k, (ok, _) in judge(**g).items() if ok is None]
    ok = inc == ["G1b", "G1c"]
    print(f"[check_eval_conditions]   no outside evidence -> INCOMPLETE {inc}: {'PASS' if ok else 'FAIL'}")
    bad += not ok
    for name, fn, want in CASES:
        g = copy.deepcopy(_good())
        fn(g)
        flipped = {k.split(" ", 1)[0] for k, (ok, _) in judge(**g).items() if ok is False}
        ok = flipped == set(want)
        print(f"[check_eval_conditions]   case {name}: "
              f"{'PASS' if ok else 'FAIL'} (flips {sorted(flipped)}, declared {sorted(want)})")
        bad += not ok
    undefended = [n.split(" ", 1)[0] for n in CHECK_NAMES
                  if not any(n.split(" ", 1)[0] in w for _, _, w in CASES)]
    print(f"[check_eval_conditions]   every check is flipped by some case: "
          f"{'PASS' if not undefended else 'FAIL ' + str(undefended)}")
    bad += bool(undefended)
    total = len(BANDS) + 8 + len(CASES)
    print(f"[check_eval_conditions] self-test: {total - bad}/{total}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", help="Offline cases, no run folder.")
    ap.add_argument("--run-dir", type=str, default=None, help="The evaluation folder.")
    ap.add_argument("--table", type=str, default=None, help="The committed table CSV.")
    ap.add_argument("--log", type=str, default=None, help="The run's rt_log text (G1b).")
    ap.add_argument("--filehash", type=str, default=None,
                    help="SHA256 that Get-FileHash printed on the training PC (G1c).")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.run_dir or not args.table:
        ap.error("give --self-test, or --run-dir and --table")
    run_dir = pathlib.Path(args.run_dir)
    print(f"[check_eval_conditions] marker: {SCRIPT_MARKER}")
    print(f"[check_eval_conditions] run folder: {run_dir}")
    table, sidecar, table_sha = load_table(pathlib.Path(args.table))
    ep_header, ep_rows = _read_csv(run_dir / "episodes.csv")
    ap_header, ap_rows = _read_csv(run_dir / "applied.csv")
    meta = json.loads((run_dir / "applied_meta.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "eval_manifest.json").read_text(encoding="utf-8"))
    result = judge(table, sidecar, table_sha,
                   log_hashes(pathlib.Path(args.log)) if args.log else None, args.filehash,
                   ep_header, ep_rows, ap_header, ap_rows, meta, manifest)
    print(f"[check_eval_conditions] tag {manifest.get('tag')!r}: expected noise {expected_noise(manifest.get('tag'))}")
    word = {True: "PASS", False: "FAIL", None: "INCOMPLETE"}
    for name, (ok, detail) in result.items():
        print(f"[check_eval_conditions]   {word[ok]:<10s}  {name}  -- {detail[:220]}")
    fails = [k for k, (ok, _) in result.items() if ok is False]
    open_ = [k for k, (ok, _) in result.items() if ok is None]
    verdict = "FAIL" if fails else "INCOMPLETE" if open_ else "PASS"
    print(f"[check_eval_conditions] VERDICT: {verdict} "
          f"({len(fails)} failed, {len(open_)} without outside evidence)")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
