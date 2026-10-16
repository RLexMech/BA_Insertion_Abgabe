# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""G1-G3 of eval_plan.md: did the evaluation run play THE table, every row once,
under the conditions the table says?

Reads one evaluation folder (``play.py --eval-table`` writes it):
``applied.csv``, ``applied_meta.json``, ``episodes.csv``, ``eval_manifest.json``
-- and the committed table CSV with its ``.json`` sidecar.

  G1  ONE hash everywhere: the table file's bytes, its sidecar, the manifest,
      ``applied_meta.json``, EVERY ``[eval-table]`` line of the run log (--log;
      one per startup report, at least one)
      and the training PC's ``Get-FileHash`` (--filehash). The last two are
      EVIDENCE FROM OUTSIDE; without both G1 is INCOMPLETE, never PASS.
  G2  ``episodes.csv`` holds exactly one ``episode == 0`` row per env id
      0..N-1, N = table rows = manifest ``num_envs``, and the manifest says
      every env finished its first episode.
  G3  ``applied.csv`` = the table row mapped by THIS script, tolerance 1e-6.

THE MAPPING IS THE CHECK'S OWN (float64, stdlib ``math``), not an import of
``autodr`` / ``insertion_math``: a check that maps with the env's functions
holds the env against itself. The bands are own literals from eval_plan.md
section 1.1 and are FIRST held against the bands the env wrote into
``applied_meta.json``.

Stdlib only; runs on either machine.

    python scripts/check_eval_conditions.py --self-test
    python scripts/check_eval_conditions.py --run-dir <eval folder> ^
        --table scripts/eval/eval_table_seed20.csv --log rt_logs/RT-N/RT-N.txt --filehash <HEX>
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import pathlib
import re
import sys

SCRIPT_MARKER = "check_eval_conditions-2026-09-22a"

TABLE_COLUMNS = (
    "lat_r", "lat_phi", "yaw", "tilt", "tilt_azimuth", "start_height", "friction",
    "fixture_noise_x", "fixture_noise_y",
    "joint_noise_0", "joint_noise_1", "joint_noise_2", "joint_noise_3",
    "joint_noise_4", "joint_noise_5", "grasp_obs_x",
)
APPLIED_COLUMNS = (
    "row_id", "lat_x_m", "lat_y_m", "start_height_m", "yaw_rad",
    "pocket_qw", "pocket_qx", "pocket_qy", "pocket_qz",
    "pocket_dx_m", "pocket_dy_m", "grasp_obs_x_m", "friction_cmd",
    "friction_robot_min", "friction_robot_max",
    "friction_fixture_min", "friction_fixture_max",
)
FRICTION_READBACKS = APPLIED_COLUMNS[13:]

BANDS = {
    "target": {
        "lat_r": (0.0, 0.030),
        "yaw": (-math.radians(5.0), math.radians(5.0)),
        "tilt": (0.0, math.radians(10.0)),
        "start_height": (0.020, 0.120),
        "friction": (0.08, 0.20),
    },
    "nominal": {
        "lat_r": (0.0, 0.0),
        "yaw": (0.0, 0.0),
        "tilt": (0.0, 0.0),
        "start_height": (0.020, 0.120),
        "friction": (0.140, 0.140),
    },
}
FIXTURE_NOISE_XY_M = 0.005
GRASP_OBS_X_M = 0.003
TOL = 1e-6
TOL_BAND = 1e-9

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
)


def _lerp(u: float, lo: float, hi: float) -> float:
    return lo + u * (hi - lo)


def map_row(u: dict, width: str) -> dict:
    """One table row (unit values by column name) -> the applied.csv values."""
    b = BANDS[width]
    r = b["lat_r"][1] * math.sqrt(u["lat_r"])
    phi = 2.0 * math.pi * u["lat_phi"]
    yaw = _lerp(u["yaw"], *b["yaw"])
    half = 0.5 * _lerp(u["tilt"], *b["tilt"])
    az = 2.0 * math.pi * u["tilt_azimuth"]
    c, s = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    a, bx, by = math.cos(half), math.cos(az) * math.sin(half), math.sin(az) * math.sin(half)
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
        "grasp_obs_x_m": (2.0 * u["grasp_obs_x"] - 1.0) * GRASP_OBS_X_M,
        "friction_cmd": mu,
    }
    return out


def judge(table, sidecar, table_sha, log_shas, filehash, ep_header, ep_rows,
          ap_header, ap_rows, meta, manifest) -> dict:
    """``{check name: (ok, detail)}`` over CHECK_NAMES."""
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
    band_ok = band_ok and abs(float(meta.get("fixture_pos_noise_xy_m", -1)) - FIXTURE_NOISE_XY_M) <= TOL_BAND \
        and abs(float(meta.get("grasp_obs_offset_x_m", -1)) - GRASP_OBS_X_M) <= TOL_BAND
    out[CHECK_NAMES[6]] = (band_ok, f"width {width!r}: {got_b}")

    worst, worst_at, bad = 0.0, "", 0
    fr_worst, fr_at, fr_bad = 0.0, "", 0
    if shape_ok and want_b is not None:
        for i, (u_row, a) in enumerate(zip(table, ap_rows)):
            want = map_row(dict(zip(TABLE_COLUMNS, u_row)), width)
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
    return out


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


def _good(width: str = "target"):
    n = 6
    table = [[((7 * i + 3 * j) % 11) / 10.0 for j in range(16)] for i in range(n)]
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
    ep_rows += [{"env_id": "2", "episode": "1"}]
    meta = {"table": {"sha256": sha}, "bounds": {k: list(v) for k, v in BANDS[width].items()},
            "fixture_pos_noise_xy_m": FIXTURE_NOISE_XY_M, "grasp_obs_offset_x_m": GRASP_OBS_X_M}
    manifest = {"width": width, "table_sha256": sha, "num_envs": n, "table_rows": n,
                "envs_finished_first_episode": n}
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


CASES = (
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
    ("report-prints-four-times", lambda g: g.update(log_shas=[g["table_sha"]] * 4), ()),
    ("one-of-four-lines-carries-another-hash",
     lambda g: g.update(log_shas=[g["table_sha"]] * 2 + ["00" * 32, g["table_sha"]]), ("G1b",)),
    ("filehash-differs", lambda g: g.update(filehash="00" * 32), ("G1c",)),
    ("an-env-has-no-first-episode", lambda g: g["ep_rows"].pop(0), ("G2a",)),
    ("an-env-id-twice", lambda g: g["ep_rows"][1].update(env_id="0"), ("G2a",)),
    ("one-env-never-finished", lambda g: g["manifest"].update(envs_finished_first_episode=5), ("G2b",)),
)


def _oracle_vs_production() -> tuple[bool, str]:
    """``map_row`` against the functions the env really calls, on the committed seed-20 table with the shape the env hands over, (1000, 16) (L-14)."""
    import importlib.util

    repo = pathlib.Path(__file__).resolve().parents[1]
    pkg = repo / "source" / "insertion" / "insertion" / "tasks" / "direct" / "insertion"

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    torch, _ = _load("torch_shim", repo / "scripts" / "tools" / "torch_shim.py").load()
    sys.modules.setdefault("torch", torch)
    autodr = _load("check_autodr", repo / "scripts" / "check_autodr.py").load_autodr()
    ns: dict = {"torch": torch, "math": math, "__name__": "insertion_math"}
    exec(compile((pkg / "insertion_math.py").read_text(encoding="utf-8"), "insertion_math.py", "exec"), ns)  # noqa: S102

    table, _, _ = load_table(repo / "scripts" / "eval" / "eval_table_seed20.csv")
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
    return len(table) == 1000 and worst <= TOL, f"{len(table)} rows, worst {worst:.3g} at {at}"


def self_test() -> int:
    print(f"[check_eval_conditions] marker: {SCRIPT_MARKER}")
    bad = 0
    ok, detail = _oracle_vs_production()
    print(f"[check_eval_conditions]   oracle = production math on the seed-20 table: "
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
    total = len(BANDS) + 3 + len(CASES)
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
