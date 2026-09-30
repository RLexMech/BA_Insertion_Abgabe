# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Run curves over iterations from a ``RT-<N>_scalars.csv``, PNG + PDF.

WHAT IT DRAWS
-------------
One figure per curve, each as ``<out-dir>/RT-<N>_<curve>.png`` AND ``.pdf``
(same figure, two ``savefig`` calls; the PDF is the thesis copy):

    reward       mean episode reward
    success      trailing-window success rate, in %
    depth        mean maximum insertion depth, mm
    force_abort  force-abort rate, in %

An optional reference run (``--ref``, ``--ref-run``) is drawn on the same
axes in a second colour; the legend names the two run IDs.

WHERE THE TAG NAMES COME FROM
-----------------------------
The CSV is the wide table of ``scripts/export_tb_scalars.py`` (header
``step,wall_time,<tag>,...``). rsl_rl versions name their tags differently
and the runner source is not on the laptop, so each curve has a CANDIDATE
list and the first tag present wins. A curve none of whose candidates is in
the file is SKIPPED with the file's tag list printed -- extend the list, or
pass ``--reward-tag`` to pick the reward series by hand.

Offline, no Isaac: ``csv`` + ``matplotlib``. Style follows
``plot_start_height_bins.py`` (one hue per series, German axis labels with
units, ``--no-title`` for thesis figures whose caption names them).

    python scripts/plot_run_curves.py --scalars docs/figures/RT-156_scalars.csv --run RT-156 \
        --ref docs/figures/RT-138_scalars.csv --ref-run RT-138 --out-dir docs/figures
    python scripts/plot_run_curves.py --self-test

``--self-test`` draws SYNTHETIC curves into a temp dir, checks the tag
resolver (first candidate present, skip on miss), the header guard and the
two output files, and never writes under ``docs/``.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import tempfile

HEADER_FIXED = ("step", "wall_time")

# One hue per series (dataviz: categorical pair; text wears text tokens).
RUN = "#3B6FD4"
REF = "#D97706"
INK = "#1F2933"
INK_MUTED = "#6B7280"
GRID = "#E5E7EB"

# curve -> (candidate tags in order, y label, scale, y limits or None)
CURVES: dict[str, tuple[tuple[str, ...], str, float, tuple[float, float] | None]] = {
    "reward": (("Train/mean_reward", "Episode_Reward/total"), "Mittlerer Episoden-Reward [–]", 1.0, None),
    "success": (("Episode/success_rate", "success_rate"), "Erfolgsrate [%]", 100.0, (0.0, 100.0)),
    "depth": (("Episode/mean_max_depth_mm", "mean_max_depth_mm"), "Mittlere maximale Fügetiefe [mm]", 1.0, None),
    "force_abort": (("Episode/force_abort_rate", "force_abort_rate"), "Kraftabbruch-Rate [%]", 100.0, (0.0, 100.0)),
}
TITLES = {"reward": "Reward", "success": "Erfolgsrate", "depth": "Fügetiefe", "force_abort": "Kraftabbruch-Rate"}


# --------------------------------------------------------------------------
# CSV + resolver
# --------------------------------------------------------------------------

def read_csv(path: pathlib.Path) -> tuple[list[str], list[int], dict[str, list[float | None]]]:
    """(tags, steps, {tag: values aligned to steps, None where empty})."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r, None)
        if header is None or tuple(header[:2]) != HEADER_FIXED:
            raise ValueError(f"{path}: header must start with {','.join(HEADER_FIXED)}; got {header}")
        tags = header[2:]
        steps: list[int] = []
        series: dict[str, list[float | None]] = {t: [] for t in tags}
        for cells in r:
            if not cells:
                continue
            steps.append(int(cells[0]))
            for tag, cell in zip(tags, cells[2:]):
                series[tag].append(float(cell) if cell != "" else None)
    return tags, steps, series


def resolve(tags: list[str], candidates: tuple[str, ...]) -> str | None:
    """The first candidate that is a tag of the file; None if none is."""
    for c in candidates:
        if c in tags:
            return c
    return None


def smooth(values: list[float | None], window: int) -> list[float | None]:
    """Trailing moving average over ``window`` samples; None passes through as a gap."""
    if window <= 1:
        return list(values)
    out: list[float | None] = []
    buf: list[float] = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def draw_curve(curve: str, runs: list[tuple[str, list[int], list[float | None]]], out_base: pathlib.Path,
               title: bool = True, tag_note: str = "") -> list[pathlib.Path]:
    """Draw one curve for one or two runs; returns the two files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, ylabel, scale, ylim = CURVES[curve]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    for (label, steps, values), colour in zip(runs, (RUN, REF)):
        xs = [s for s, v in zip(steps, values) if v is not None]
        ys = [v * scale for v in values if v is not None]
        ax.plot(xs, ys, color=colour, linewidth=1.4, label=label, zorder=3)
    ax.set_xlabel("Iteration", color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    if len(runs) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    if title:
        text = f"{runs[0][0]}: {TITLES[curve]} über Iterationen"
        if tag_note:
            text += f"  ({tag_note})"
        ax.set_title(text, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in (".png", ".pdf"):
        p = out_base.with_suffix(ext)
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    return written


def plot_all(scalars: pathlib.Path, run: str, out_dir: pathlib.Path, which: list[str], ref: pathlib.Path | None,
             ref_run: str, reward_tag: str | None, window: int, title: bool) -> int:
    """Draw the requested curves; returns how many were drawn."""
    tags, steps, series = read_csv(scalars)
    ref_data = read_csv(ref) if ref is not None else None
    drawn = 0
    for curve in which:
        candidates = CURVES[curve][0]
        if curve == "reward" and reward_tag:
            candidates = (reward_tag,)
        tag = resolve(tags, candidates)
        if tag is None:
            print(f"SKIP {curve}: none of {list(candidates)} in {scalars.name}; tags present: {tags}")
            continue
        runs = [(run, steps, smooth(series[tag], window))]
        if ref_data is not None:
            r_tags, r_steps, r_series = ref_data
            r_tag = resolve(r_tags, candidates)
            if r_tag is None:
                print(f"  reference {ref_run}: none of {list(candidates)} in {ref.name}; drawn without reference")
            else:
                runs.append((ref_run, r_steps, smooth(r_series[r_tag], window)))
        files = draw_curve(curve, runs, out_dir / f"{run}_{curve}", title=title, tag_note=tag)
        print(f"  {curve:<12} <- {tag:<32} " + ", ".join(str(p) for p in files))
        drawn += 1
    return drawn


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _self_test() -> int:
    rc = 0

    def check(name: str, ok: bool) -> None:
        nonlocal rc
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        rc |= 0 if ok else 1

    check("resolver takes the FIRST present candidate", resolve(["b", "a"], ("a", "b")) == "a")
    check("resolver falls through to a later candidate", resolve(["b"], ("a", "b")) == "b")
    check("resolver returns None on a miss", resolve(["c"], ("a", "b")) is None)
    check("smooth window 1 is the identity", smooth([1.0, None, 3.0], 1) == [1.0, None, 3.0])
    check("smooth window 2 averages and keeps gaps", smooth([1.0, 3.0, None, 5.0], 2) == [1.0, 2.0, None, 4.0])

    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        csv_path = root / "synthetic_scalars.csv"
        lines = ["step,wall_time,Train/mean_reward,Episode/success_rate"]
        for i in range(30):
            lines.append(f"{i},{1.7e9 + i:.3f},{10.0 * i:.7g},{i / 30.0:.7g}")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tags, steps, series = read_csv(csv_path)
        check("read_csv returns tags, 30 steps, aligned series", tags == ["Train/mean_reward", "Episode/success_rate"]
              and len(steps) == 30 and len(series[tags[0]]) == 30)

        bad = root / "bad.csv"
        bad.write_text("iteration,reward\n1,2\n", encoding="utf-8")
        try:
            read_csv(bad)
            check("a foreign header is refused", False)
        except ValueError:
            check("a foreign header is refused", True)

        out_dir = root / "figs"
        n = plot_all(csv_path, "SYNTHETIC", out_dir, ["reward", "success", "depth"], None, "", None, 1, True)
        check("two curves drawn, the third (depth) skipped without exception", n == 2)
        for curve in ("reward", "success"):
            for ext in (".png", ".pdf"):
                p = out_dir / f"SYNTHETIC_{curve}{ext}"
                check(f"{p.name} written (> 1000 bytes)", p.is_file() and p.stat().st_size > 1000)
        check("skipped curve leaves no file", not (out_dir / "SYNTHETIC_depth.png").exists())

        n = plot_all(csv_path, "SYNTHETIC", out_dir, ["reward"], csv_path, "REF", None, 3, False)
        check("reference overlay + smoothing + --no-title draws", n == 1)

    print("SELF-TEST PASSED" if rc == 0 else "SELF-TEST FAILED")
    return rc


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scalars", type=pathlib.Path, help="docs/figures/RT-<N>_scalars.csv")
    ap.add_argument("--run", default="", help="run label, e.g. RT-156 (also the file-name prefix)")
    ap.add_argument("--ref", type=pathlib.Path, default=None, help="reference run CSV to overlay")
    ap.add_argument("--ref-run", default="Referenz", help="label of the reference run, e.g. RT-138")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("docs/figures"))
    ap.add_argument("--which", default="reward,success,depth,force_abort",
                    help="comma list of curves: " + ",".join(CURVES))
    ap.add_argument("--reward-tag", default=None, help="tag to use for the reward curve instead of the candidates")
    ap.add_argument("--smooth", type=int, default=1, help="trailing moving-average window in iterations (1 = off)")
    ap.add_argument("--no-title", action="store_true", help="omit the in-figure title (thesis: the caption names it)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.scalars is None or not a.run:
        ap.error("--scalars and --run are required (or --self-test)")
    which = [w.strip() for w in a.which.split(",") if w.strip()]
    unknown = [w for w in which if w not in CURVES]
    if unknown:
        ap.error(f"unknown curve(s) {unknown}; known: {list(CURVES)}")
    n = plot_all(a.scalars, a.run, a.out_dir, which, a.ref, a.ref_run, a.reward_tag, a.smooth, not a.no_title)
    print(f"drawn {n} of {len(which)} curves")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
