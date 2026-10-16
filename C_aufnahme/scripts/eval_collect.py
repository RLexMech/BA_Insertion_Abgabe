# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Collect every evaluation run into ONE long table (eval_plan.md section 8b, step 1).

Input layout (copied from the training PC, one folder per play.py run):

    results/eval/<block>/<method>_s<seed>[_<tag>]/
        episodes.csv  episodes_meta.json  applied.csv  applied_meta.json  eval_manifest.json

``<block>`` is one of ``BLOCKS``; ``<method>`` is ``autodr``, ``nodr`` or
``fixeddr``. The folder name is a CLAIM and is held against the manifest's
checkpoint path, whose run folder train.py names ``..._autodr_..._seed<N>`` or
``..._fixed_dr_..._seed<N>`` (``dr_mode`` tag; 'off' has none,
train.py:334-335; seed tag train.py:444). Fixed-DR (D-192) plays only in
``T-nominal-F`` / ``T-final-F``, and those blocks hold nothing else.

Refuses with exit 1 -- equality, never "not empty" -- when:
  * a run fails G1a / G2 / G3 of ``check_eval_conditions.judge`` (reused, not copied);
  * a manifest names another table or width than its block, or a dirty clone;
  * the runs disagree on ``code_commit`` or ``plan_commit``, or a plan commit is missing;
  * the folder name and the checkpoint path disagree on method or seed;
  * a Fixed-DR folder sits outside the ``-F`` blocks, or another one inside;
  * two folders of one block are the same policy without a tag.

Output: ``<results>/episodes_all.csv`` (only ``episode == 0`` rows -- the
window is episodes.csv's own column, no second window rule, L-15) and
``<results>/collect_report.json`` (per run, plus G4 and G6 as reported shares).

    .venv-analysis/Scripts/python.exe scripts/eval_collect.py --results results/eval
    .venv-analysis/Scripts/python.exe scripts/eval_collect.py --self-test
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import pathlib
import re
import sys
import tempfile

SCRIPT_MARKER = "eval_collect-2026-09-24a"
REPO = pathlib.Path(__file__).resolve().parents[1]
TABLE_DIR = REPO / "scripts" / "eval"

BLOCKS = {
    "T-nominal": ("eval_table_seed10.csv", "nominal"),
    "T-auswahl": ("eval_table_seed30.csv", "target"),
    "T-final": ("eval_table_seed10.csv", "target"),
    "T-grenze": ("eval_table_boundary_seed10.csv", "target"),
    "technik": ("eval_table_seed20.csv", None),
    "T-nominal-F": ("eval_table_seed10.csv", "nominal"),
    "T-final-F": ("eval_table_seed10.csv", "target"),
}
METHODS = ("autodr", "nodr", "fixeddr")
FIXED_BLOCKS = ("T-nominal-F", "T-final-F")
_MODE_TAG = {"_autodr_": "autodr", "_fixed_dr_": "fixeddr"}
_FOLDER = re.compile(r"^(autodr|nodr|fixeddr)_s(\d+)(?:_([A-Za-z0-9-]+))?$")
_CKPT_SEED = re.compile(r"_seed(\d+)$")

OUT_COLUMNS = (
    "block", "method", "seed", "tag", "width", "row_id", "boundary_key",
    "outcome", "success_train", "success_strict", "force_abort",
    "steps", "force_max_filtered_n", "force_max_raw_n", "ip_max_mm", "force_limit_n",
    "lat_r_mm", "yaw_abs_deg", "tilt_deg", "start_height_m", "friction",
)


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CEC = _load_module("check_eval_conditions", REPO / "scripts" / "check_eval_conditions.py")


def identity_from_checkpoint(path: str) -> tuple[str, int] | None:
    """``(method, seed)`` out of the training run folder that holds the checkpoint."""
    run = pathlib.PureWindowsPath(str(path)).parent.name
    m = _CKPT_SEED.search(run)
    if m is None:
        return None
    hits = [method for tag, method in _MODE_TAG.items() if tag in run]
    if len(hits) > 1:
        return None
    return (hits[0] if hits else "nodr"), int(m.group(1))


def tilt_deg(qx: float, qy: float) -> float:
    """Polar angle of the pocket z axis -- the env's own tilt reading (`_log_finished_episodes`, acos of r22), not the quaternion's total angle."""
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(math.acos(max(-1.0, min(1.0, r22))))


def collect_run(block: str, folder: str, run: dict, table: dict) -> tuple[list, list, dict]:
    """``(rows, problems, info)`` for one run. ``run`` holds the parsed files."""
    problems: list[str] = []
    m = _FOLDER.match(folder)
    if m is None:
        return [], [f"{block}/{folder}: folder name is not <autodr|nodr|fixeddr>_s<seed>[_<tag>]"], {}
    method, seed, tag = m.group(1), int(m.group(2)), m.group(3) or ""
    if (method == "fixeddr") != (block in FIXED_BLOCKS):
        want = "fixeddr" if block in FIXED_BLOCKS else "autodr/nodr"
        problems.append(f"block {block} takes only {want}, folder is {method}")
    man = run["manifest"]
    want_width = BLOCKS[block][1]
    width = str(man.get("width"))
    if want_width is not None and width != want_width:
        problems.append(f"width {width!r}, block wants {want_width!r}")
    if man.get("table_sha256") != table["sha"]:
        problems.append(f"table {str(man.get('table_sha256'))[:12]}.. is not {BLOCKS[block][0]}")
    if man.get("code_dirty") is not False:
        problems.append(f"code_dirty is {man.get('code_dirty')!r}")
    ident = identity_from_checkpoint(man.get("checkpoint_path", ""))
    if ident != (method, seed):
        problems.append(f"folder says {(method, seed)}, checkpoint path says {ident}")

    verdict = CEC.judge(table["rows"], table["sidecar"], table["sha"], None, None,
                        run["ep_header"], run["ep_rows"], run["ap_header"], run["ap_rows"],
                        run["applied_meta"], man)
    for name, (ok, detail) in verdict.items():
        if ok is False:
            problems.append(f"{name.split(' ', 1)[0]} FAILED: {detail[:160]}")

    limit = float(run["episodes_meta"]["force"]["abort_limit_n"])
    applied = {r["row_id"]: r for r in run["ap_rows"]}
    per_block = table["sidecar"].get("rows_per_block")
    keys = [b["key"] for b in table["sidecar"].get("blocks", [])]
    rows = []
    for r in run["ep_rows"]:
        if r.get("episode") != "0":
            continue
        rid = int(r["env_id"])
        a = applied.get(str(rid))
        if a is None:
            continue
        f_filt = float(r["force_max_filtered_n"])
        ok = r["outcome"] == "success"
        rows.append({
            "block": block, "method": method, "seed": seed, "tag": tag, "width": width,
            "row_id": rid,
            "boundary_key": keys[rid // per_block] if per_block else "",
            "outcome": r["outcome"],
            "success_train": int(ok),
            "success_strict": int(ok and f_filt < limit),
            "force_abort": int(f_filt >= limit),
            "steps": int(r["steps"]),
            "force_max_filtered_n": f_filt,
            "force_max_raw_n": float(r["force_max_raw_n"]),
            "ip_max_mm": float(r["ip_max_mm"]),
            "force_limit_n": limit,
            "lat_r_mm": 1000.0 * math.hypot(float(a["lat_x_m"]), float(a["lat_y_m"])),
            "yaw_abs_deg": math.degrees(abs(float(a["yaw_rad"]))),
            "tilt_deg": tilt_deg(float(a["pocket_qx"]), float(a["pocket_qy"])),
            "start_height_m": float(a["start_height_m"]),
            "friction": float(a["friction_cmd"]),
        })
    first = {r["env_id"]: r["outcome"] for r in run["ep_rows"] if r.get("episode") == "0"}
    second = {r["env_id"]: r["outcome"] for r in run["ep_rows"] if r.get("episode") == "1"}
    same = sum(1 for k, v in second.items() if first.get(k) == v)
    info = {"block": block, "folder": folder, "method": method, "seed": seed, "tag": tag,
            "width": width, "code_commit": man.get("code_commit"),
            "plan_commit": man.get("plan_commit"),
            "g4_repeats": len(second), "g4_same_outcome": same,
            "problems": problems}
    return rows, problems, info


def g6_report(rows: list) -> list:
    """G6: the same policy on the same table twice (technik, two tags) -- share of rows whose outcome differs."""
    by = {}
    for r in rows:
        if r["block"] != "technik":
            continue
        by.setdefault((r["method"], r["seed"], r["width"]), {}).setdefault(r["tag"], {})[r["row_id"]] = r["outcome"]
    out = []
    for (method, seed, width), tags in sorted(by.items()):
        names = sorted(tags)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = tags[names[i]], tags[names[j]]
                common = sorted(set(a) & set(b))
                diff = sum(1 for k in common if a[k] != b[k])
                out.append({"method": method, "seed": seed, "width": width,
                            "tags": [names[i], names[j]], "rows": len(common),
                            "rows_with_other_outcome": diff})
    return out


def collect(runs: dict, tables: dict) -> tuple[list, list, dict]:
    """``runs``: {(block, folder): parsed files}. Returns (rows, problems, report)."""
    rows, problems, infos = [], [], []
    for (block, folder) in sorted(runs):
        r, p, info = collect_run(block, folder, runs[(block, folder)], tables[block])
        rows += r
        problems += [f"{block}/{folder}: {x}" for x in p]
        if info:
            infos.append(info)
    codes = sorted({str(i["code_commit"]) for i in infos})
    plans = sorted({str(i["plan_commit"]) for i in infos})
    if len(codes) > 1:
        problems.append(f"runs disagree on code_commit: {codes}")
    if len(plans) > 1 or "None" in plans:
        problems.append(f"runs disagree on plan_commit or miss it: {plans}")
    seen = {}
    for i in infos:
        key = (i["block"], i["method"], i["seed"], i["tag"])
        if key in seen:
            problems.append(f"{i['block']}: {seen[key]} and {i['folder']} are the same policy and tag")
        seen[key] = i["folder"]
    report = {"marker": SCRIPT_MARKER, "runs": infos, "g6": g6_report(rows),
              "code_commit": codes, "plan_commit": plans, "problems": problems}
    return rows, problems, report


def load_tables() -> dict:
    out = {}
    for block, (fname, _) in BLOCKS.items():
        rows, sidecar, sha = CEC.load_table(TABLE_DIR / fname)
        out[block] = {"rows": rows, "sidecar": sidecar, "sha": sha}
    return out


def load_runs(results: pathlib.Path) -> dict:
    runs = {}
    for block in BLOCKS:
        bdir = results / block
        if not bdir.is_dir():
            continue
        for folder in sorted(p for p in bdir.iterdir() if p.is_dir()):
            ep_header, ep_rows = CEC._read_csv(folder / "episodes.csv")
            ap_header, ap_rows = CEC._read_csv(folder / "applied.csv")
            runs[(block, folder.name)] = {
                "ep_header": ep_header, "ep_rows": ep_rows,
                "ap_header": ap_header, "ap_rows": ap_rows,
                "applied_meta": json.loads((folder / "applied_meta.json").read_text(encoding="utf-8")),
                "episodes_meta": json.loads((folder / "episodes_meta.json").read_text(encoding="utf-8")),
                "manifest": json.loads((folder / "eval_manifest.json").read_text(encoding="utf-8")),
            }
    return runs


def write_outputs(results: pathlib.Path, rows: list, report: dict) -> None:
    with open(results / "episodes_all.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(results / "collect_report.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)


SHA_A = "a1" * 32
SHA_B = "b2" * 32
CODE = "c" * 40
PLAN = "d" * 40


def _fake_table(sha: str, n: int = 4, boundary: bool = False) -> dict:
    rows = [[((5 * i + 3 * j) % 7) / 6.0 for j in range(16)] for i in range(n)]
    sidecar = {"sha256": sha, "n_rows": n}
    if boundary:
        sidecar.update(rows_per_block=2, blocks=[{"key": "lat_r_hi"}, {"key": "yaw_lo"}])
    return {"rows": rows, "sidecar": sidecar, "sha": sha}


def _fake_run(table: dict, method: str, seed: int, width: str, outcomes: list) -> dict:
    n = len(table["rows"])
    ap_rows = []
    for i, u in enumerate(table["rows"]):
        want = CEC.map_row(dict(zip(CEC.TABLE_COLUMNS, u)), width)
        row = {"row_id": str(i)}
        row.update({k: f"{v:.9g}" for k, v in want.items()})
        row.update({k: f"{want['friction_cmd']:.9g}" for k in CEC.FRICTION_READBACKS})
        ap_rows.append(row)
    ep_rows = []
    for i, (out, force) in enumerate(outcomes):
        ep_rows.append({"env_id": str(i), "episode": "0", "outcome": out, "steps": "40",
                        "force_max_filtered_n": f"{force:.4f}", "force_max_raw_n": f"{force + 1:.4f}",
                        "ip_max_mm": "0.1"})
    ep_rows.append(dict(ep_rows[0], episode="1"))
    tag = {"autodr": "_autodr_", "fixeddr": "_fixed_dr_"}.get(method, "_")
    return {
        "ep_header": ["env_id", "episode", "outcome", "steps", "force_max_filtered_n",
                      "force_max_raw_n", "ip_max_mm"],
        "ep_rows": ep_rows, "ap_header": list(CEC.APPLIED_COLUMNS), "ap_rows": ap_rows,
        "applied_meta": {"table": {"sha256": table["sha"]},
                         "bounds": {k: list(v) for k, v in CEC.BANDS[width].items()},
                         "fixture_pos_noise_xy_m": CEC.FIXTURE_NOISE_XY_M,
                         "grasp_obs_offset_x_m": CEC.GRASP_OBS_X_M},
        "episodes_meta": {"force": {"abort_limit_n": 30.0}},
        "manifest": {"width": width, "table_sha256": table["sha"], "num_envs": n, "table_rows": n,
                     "envs_finished_first_episode": n, "code_commit": CODE, "plan_commit": PLAN,
                     "code_dirty": False,
                     "checkpoint_path": f"C:\\logs\\rsl_rl\\x\\09-15_13-50-36__RT-9s{seed}{tag}start_seed{seed}\\model_1499.pt"},
    }


def _good():
    tables = {"T-final": _fake_table(SHA_A), "T-nominal": _fake_table(SHA_A),
              "T-grenze": _fake_table(SHA_B, boundary=True), "T-final-F": _fake_table(SHA_A)}
    outs = [("success", 12.0), ("success", 31.0), ("timeout", 5.0), ("force_abort", 30.0)]
    runs = {
        ("T-final", "autodr_s1"): _fake_run(tables["T-final"], "autodr", 1, "target", outs),
        ("T-final", "nodr_s1"): _fake_run(tables["T-final"], "nodr", 1, "target", outs),
        ("T-nominal", "nodr_s1"): _fake_run(tables["T-nominal"], "nodr", 1, "nominal", outs),
        ("T-grenze", "autodr_s1"): _fake_run(tables["T-grenze"], "autodr", 1, "target", outs),
        ("T-final-F", "fixeddr_s1"): _fake_run(tables["T-final-F"], "fixeddr", 1, "target", outs),
    }
    return runs, tables


def _set_man(key, run, val):
    return lambda runs, tables: runs[run]["manifest"].update({key: val})


CASES = (
    ("manifest-names-another-table", _set_man("table_sha256", ("T-final", "nodr_s1"), SHA_B), "is not"),
    ("final-run-at-nominal-width", _set_man("width", ("T-final", "nodr_s1"), "nominal"), "width"),
    ("dirty-clone", _set_man("code_dirty", ("T-final", "nodr_s1"), True), "code_dirty"),
    ("another-code-commit", _set_man("code_commit", ("T-final", "nodr_s1"), "e" * 40), "code_commit"),
    ("plan-commit-missing", _set_man("plan_commit", ("T-final", "nodr_s1"), None), "plan_commit"),
    ("folder-says-autodr-checkpoint-says-nodr",
     lambda runs, tables: runs.__setitem__(("T-final", "autodr_s2"), runs.pop(("T-final", "nodr_s1"))),
     "checkpoint path says"),
    ("an-applied-value-is-off",
     lambda runs, tables: runs[("T-final", "autodr_s1")]["ap_rows"][2].update(yaw_rad="0.5"), "G3c"),
    ("an-env-lost-its-first-episode",
     lambda runs, tables: runs[("T-final", "autodr_s1")]["ep_rows"].pop(3), "G2a"),
    ("unreadable-folder-name",
     lambda runs, tables: runs.__setitem__(("T-final", "ppo_1"), runs.pop(("T-final", "nodr_s1"))),
     "folder name"),
    ("folder-says-nodr-checkpoint-says-fixed-dr",
     lambda runs, tables: runs.__setitem__(("T-final-F", "nodr_s1"), runs.pop(("T-final-F", "fixeddr_s1"))),
     "checkpoint path says"),
    ("fixed-dr-run-in-T-final",
     lambda runs, tables: runs.__setitem__(("T-final", "fixeddr_s1"), runs.pop(("T-final-F", "fixeddr_s1"))),
     "takes only"),
    ("autodr-run-in-T-final-F",
     lambda runs, tables: runs.__setitem__(("T-final-F", "autodr_s2"), _fake_run(
         tables["T-final-F"], "autodr", 2, "target", [("success", 1.0)] * 4)),
     "takes only"),
)


def self_test() -> int:
    print(f"[eval_collect] marker: {SCRIPT_MARKER}")
    bad = 0
    runs, tables = _good()
    rows, problems, report = collect(runs, tables)
    fin = [r for r in rows if r["block"] == "T-final" and r["method"] == "autodr"]
    checks = [
        ("baseline has no problem", problems == []),
        ("episode-0 rows only: 4 per run, 5 runs", len(rows) == 20),
        ("success_train counts the success in the abort step",
         [r["success_train"] for r in fin] == [1, 1, 0, 0]),
        ("success_strict drops it (force 31 >= 30)", [r["success_strict"] for r in fin] == [1, 0, 0, 0]),
        ("force_abort reads the force, >= limit (30.0 counts)", [r["force_abort"] for r in fin] == [0, 1, 0, 1]),
        ("identity from the checkpoint path", identity_from_checkpoint(
            runs[("T-final", "nodr_s1")]["manifest"]["checkpoint_path"]) == ("nodr", 1)),
        ("identity of the real RT-206s1 folder", identity_from_checkpoint(
            r"logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF"
            r"_start+20mm_floor-30mm_seed1\model_1499.pt") == ("autodr", 1)),
        ("identity of the real RT-207s1 folder", identity_from_checkpoint(
            r"logs\rsl_rl\ur5e_insertion\09-17_09-58-45__RT-207s1_offset5mm_currOFF"
            r"_start+20mm_floor-30mm_seed1\model_1499.pt") == ("nodr", 1)),
        ("identity of a Fixed-DR folder (RT-231 name, constructed)", identity_from_checkpoint(
            r"logs\rsl_rl\ur5e_insertion\09-24_10-00-00__RT-231s1_fixed_dr_offset5mm_currOFF"
            r"_start+20mm_floor-30mm_seed1\model_1499.pt") == ("fixeddr", 1)),
        ("boundary key = row // rows_per_block",
         [r["boundary_key"] for r in rows if r["block"] == "T-grenze"] == ["lat_r_hi", "lat_r_hi", "yaw_lo", "yaw_lo"]),
        ("tilt reading: a pure yaw quaternion reads 0 deg", tilt_deg(0.0, 0.0) == 0.0),
        ("tilt reading: 10 deg about x reads 10 deg",
         abs(tilt_deg(math.sin(math.radians(5.0)), 0.0) - 10.0) < 1e-9),
        ("G4: the repeat episode is counted", all(i["g4_repeats"] == 1 and i["g4_same_outcome"] == 1
                                                 for i in report["runs"])),
    ]
    for name, ok in checks:
        print(f"[eval_collect]   {'PASS' if ok else 'FAIL'}  {name}")
        bad += not ok
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(pathlib.Path(tmp), rows, report)
        with open(pathlib.Path(tmp) / "episodes_all.csv", newline="", encoding="utf-8") as fh:
            back = list(csv.DictReader(fh))
        ok = len(back) == 20 and tuple(back[0]) == OUT_COLUMNS
        print(f"[eval_collect]   {'PASS' if ok else 'FAIL'}  episodes_all.csv round-trips with OUT_COLUMNS")
        bad += not ok
    for name, fn, needle in CASES:
        runs, tables = _good()
        runs, tables = copy.deepcopy(runs), copy.deepcopy(tables)
        fn(runs, tables)
        _, problems, _ = collect(runs, tables)
        ok = any(needle in p for p in problems)
        print(f"[eval_collect]   case {name}: {'PASS' if ok else 'FAIL'} ({len(problems)} problem(s))")
        bad += not ok
    total = len(checks) + 1 + len(CASES)
    print(f"[eval_collect] self-test: {total - bad}/{total}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--results", type=str, default="results/eval")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    results = pathlib.Path(args.results)
    print(f"[eval_collect] marker: {SCRIPT_MARKER}")
    rows, problems, report = collect(load_runs(results), load_tables())
    if not report["runs"]:
        print(f"[eval_collect] REFUSED: no run folder under {results}/<block>/")
        return 1
    write_outputs(results, rows, report)
    for i in report["runs"]:
        print(f"[eval_collect]   {i['block']}/{i['folder']}: {len(i['problems'])} problem(s)")
    for p in problems:
        print(f"[eval_collect]   PROBLEM {p}")
    print(f"[eval_collect] {len(rows)} rows -> {results / 'episodes_all.csv'}; "
          f"VERDICT {'OK' if not problems else 'REFUSED'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
