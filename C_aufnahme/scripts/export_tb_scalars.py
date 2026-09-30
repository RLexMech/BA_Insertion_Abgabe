# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Dump every TensorBoard scalar of ONE run folder to a small wide CSV.

WHY
---
The reward / success / depth curves over iterations exist only in the
``events.out.tfevents*`` files of ``logs/rsl_rl/ur5e_insertion/<run>/`` on
the training PC. ``logs/`` never leaves that machine (gitignored, large).
This script turns the scalars of one run into a CSV of a few hundred KB that
travels by hand like ``demo_metrics.json`` and is committed under
``docs/figures/RT-<N>_scalars.csv`` (the ``*.csv`` ignore rule has an
exception for exactly that name). ``scripts/plot_run_curves.py`` draws the
run figures from it on the laptop.

NO TAG NAME IS ASSUMED. rsl_rl versions differ in how they name their tags
(``Loss/value`` vs ``Loss/value_function``, ``Policy/mean_std`` vs
``Policy/mean_noise_std``) and the runner source is not on the laptop. So
every scalar tag of the run is exported and printed; the plotter resolves
its curves from candidate lists and shows this list on a miss.

FORMAT (wide, one row per logged step)
--------------------------------------
    step,wall_time,<tag_1>,<tag_2>,...

rsl_rl writes every scalar once per iteration, so the rows are dense; a
tag missing at a step leaves its cell empty. ``wall_time`` is the seconds
since the epoch of the first tag seen at that step. Wide was chosen over
long (``tag,step,value``) because it is about seven times smaller for the
same run and reads as a table.

RUN ON THE TRAINING PC (conda env active, repo root):

    python scripts\export_tb_scalars.py --run-dir <absolute run folder> --out docs\figures\RT-156_scalars.csv
    python scripts\export_tb_scalars.py --load-run 09-05_19-45-03 --out docs\figures\RT-156_scalars.csv

``--load-run`` matches the START of a folder name under ``--log-root``
(default ``logs/rsl_rl/ur5e_insertion``) and refuses anything but exactly
one match -- the same trap as ``play.py --load_run`` (PROBLEMS.md, regex).
``--tags SUBSTR ...`` keeps only tags containing one of the substrings.

``--self-test`` runs WITHOUT tensorboard (the import is lazy): it writes a
synthetic wide table, reads it back, checks the tag filter and the folder
resolver in a temp dir, and never touches ``docs/``.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import tempfile

HEADER_FIXED = ("step", "wall_time")
EVENT_GLOB = "events.out.tfevents*"


# --------------------------------------------------------------------------
# folder resolution
# --------------------------------------------------------------------------

def resolve_run_dir(log_root: pathlib.Path, load_run: str) -> pathlib.Path:
    """The ONE folder under ``log_root`` whose name starts with ``load_run``."""
    if not log_root.is_dir():
        raise FileNotFoundError(f"log root not found: {log_root}")
    hits = sorted(p for p in log_root.iterdir() if p.is_dir() and p.name.startswith(load_run))
    if len(hits) != 1:
        names = ", ".join(p.name for p in hits) or "none"
        raise ValueError(f"--load-run {load_run!r} must match exactly one folder under {log_root}; matches: {names}")
    return hits[0]


# --------------------------------------------------------------------------
# reading (training PC only; tensorboard imported here, not at module top)
# --------------------------------------------------------------------------

def read_scalars(run_dir: pathlib.Path) -> tuple[list[str], dict[int, dict]]:
    """All scalar tags of ``run_dir`` -> (sorted tags, {step: {tag: value, "wall_time": t}})."""
    if not any(run_dir.glob(EVENT_GLOB)):
        raise FileNotFoundError(f"no {EVENT_GLOB} in {run_dir}")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = sorted(acc.Tags().get("scalars", []))
    if not tags:
        raise ValueError(f"no scalar tags in {run_dir}")
    rows: dict[int, dict] = {}
    for tag in tags:
        for ev in acc.Scalars(tag):
            row = rows.setdefault(int(ev.step), {"wall_time": float(ev.wall_time)})
            row[tag] = float(ev.value)
    return tags, rows


def filter_tags(tags: list[str], keep: list[str] | None) -> list[str]:
    """Tags containing at least one of the substrings in ``keep`` (None = all)."""
    if not keep:
        return list(tags)
    return [t for t in tags if any(k in t for k in keep)]


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def write_csv(path: pathlib.Path, tags: list[str], rows: dict[int, dict]) -> int:
    """Write the wide table; returns the number of data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(HEADER_FIXED) + tags)
        for step in sorted(rows):
            row = rows[step]
            cells = [str(step), f"{row['wall_time']:.3f}"]
            for tag in tags:
                v = row.get(tag)
                cells.append("" if v is None else f"{v:.7g}")
            w.writerow(cells)
            n += 1
    return n


def read_csv(path: pathlib.Path) -> tuple[list[str], dict[int, dict]]:
    """Inverse of ``write_csv``; refuses a file whose header does not start with step,wall_time."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r, None)
        if header is None or tuple(header[:2]) != HEADER_FIXED:
            raise ValueError(f"{path}: header must start with {','.join(HEADER_FIXED)}; got {header}")
        tags = header[2:]
        rows: dict[int, dict] = {}
        for cells in r:
            if not cells:
                continue
            step = int(cells[0])
            row = {"wall_time": float(cells[1])}
            for tag, cell in zip(tags, cells[2:]):
                if cell != "":
                    row[tag] = float(cell)
            rows[step] = row
    return tags, rows


# --------------------------------------------------------------------------
# self-test (offline, no tensorboard)
# --------------------------------------------------------------------------

def _self_test() -> int:
    rc = 0

    def check(name: str, ok: bool) -> None:
        nonlocal rc
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        rc |= 0 if ok else 1

    tags = ["Episode/success_rate", "Loss/value_function", "Train/mean_reward"]
    rows = {}
    for step in range(0, 30, 5):
        rows[step] = {"wall_time": 1.7e9 + step, tags[0]: step / 30.0, tags[2]: 10.0 * step}
        if step != 10:  # one hole: a tag missing at one step stays empty
            rows[step][tags[1]] = 0.5 / (step + 1)

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "synthetic_scalars.csv"
        n = write_csv(out, tags, rows)
        check("write_csv writes one row per step", n == 6)
        got_tags, got_rows = read_csv(out)
        check("round trip keeps the tag list", got_tags == tags)
        same = got_rows.keys() == rows.keys() and all(
            abs(got_rows[s].get(t, float("nan")) - rows[s].get(t, float("nan"))) < 1e-6
            for s in rows for t in tags if t in rows[s]
        )
        check("round trip keeps every value", same)
        check("a missing cell stays missing", tags[1] not in got_rows[10] and tags[1] in got_rows[5])
        check("header row starts with step,wall_time", out.read_text(encoding="utf-8").startswith("step,wall_time,"))

        bad = pathlib.Path(d) / "bad.csv"
        bad.write_text("iteration,value\n1,2\n", encoding="utf-8")
        try:
            read_csv(bad)
            check("a foreign header is refused", False)
        except ValueError:
            check("a foreign header is refused", True)

        check("tag filter keeps only matching tags", filter_tags(tags, ["Episode/", "Train/"]) == [tags[0], tags[2]])
        check("no filter keeps all tags", filter_tags(tags, None) == tags)

        root = pathlib.Path(d) / "logs" / "rsl_rl" / "ur5e_insertion"
        for name in ("09-05_19-45-03_offset5mm_seed42", "09-05_18-11-26_offset5mm_seed42", "09-02_14-20-45_x"):
            (root / name).mkdir(parents=True)
        check("resolver: a unique prefix resolves", resolve_run_dir(root, "09-05_19-45").name.startswith("09-05_19-45-03"))
        try:
            resolve_run_dir(root, "09-05_")
            check("resolver: an ambiguous prefix is refused", False)
        except ValueError:
            check("resolver: an ambiguous prefix is refused", True)
        try:
            resolve_run_dir(root, "09-09_")
            check("resolver: a missing prefix is refused", False)
        except ValueError:
            check("resolver: a missing prefix is refused", True)
        try:
            read_scalars(root / "09-02_14-20-45_x")
            check("a folder without event files is refused", False)
        except FileNotFoundError:
            check("a folder without event files is refused", True)

    print("SELF-TEST PASSED" if rc == 0 else "SELF-TEST FAILED")
    return rc


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=pathlib.Path, help="absolute run folder (holds events.out.tfevents*)")
    ap.add_argument("--load-run", default=None, help="folder-name PREFIX under --log-root, must match exactly one")
    ap.add_argument("--log-root", type=pathlib.Path, default=pathlib.Path("logs/rsl_rl/ur5e_insertion"))
    ap.add_argument("--out", type=pathlib.Path, help="CSV path, e.g. docs/figures/RT-156_scalars.csv")
    ap.add_argument("--tags", nargs="*", default=None, help="keep only tags containing one of these substrings")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.out is None or (a.run_dir is None) == (a.load_run is None):
        ap.error("--out and exactly one of --run-dir / --load-run are required (or --self-test)")
    run_dir = a.run_dir if a.run_dir is not None else resolve_run_dir(a.log_root, a.load_run)
    print(f"run folder: {run_dir}")
    all_tags, rows = read_scalars(run_dir)
    tags = filter_tags(all_tags, a.tags)
    if not tags:
        print(f"no tag matches --tags {a.tags}; tags present: {all_tags}")
        return 1
    n = write_csv(a.out, tags, rows)
    steps = sorted(rows)
    print(f"tags exported: {len(tags)} of {len(all_tags)}")
    for t in all_tags:
        print(f"  {'+' if t in tags else '-'} {t}")
    print(f"rows: {n}  steps {steps[0]} .. {steps[-1]}")
    print(f"wrote {a.out} ({a.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
