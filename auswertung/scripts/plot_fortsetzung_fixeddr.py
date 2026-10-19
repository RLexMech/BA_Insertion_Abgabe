"""H25: Experiment D, the three Fixed-DR seeds that had not learned in 100 epochs
(s1, s2, s5), resumed from `ep_100` to epoch 250 (RT-253, D-209, exploratory,
outside the method comparison). Training success rate over the epoch, as H6.

Sources: epochs 1..100 = RT-243s1/s2/s5, read from the tracked
`results/training_rlgames/curves.csv` (method fixeddr; the one home of the
final-study curves, written by `eval_figures.py --extract-training`).
Epochs 101..250 = RT-253s1/s2/s5 (`rt_logs/RT-253s/`, git-ignored), reduced by
`--extract` with `eval_figures.read_rl_games_run` (the one rl_games reader) to
the tracked `results/training_rlgames/fortsetzung_RT-253.csv`. Two steps:

    .venv-analysis/Scripts/python.exe scripts/plot_fortsetzung_fixeddr.py --extract
        rt_logs/RT-253s/RT-253s{1,2,5}_* -> results/training_rlgames/fortsetzung_RT-253.csv
    .venv-analysis/Scripts/python.exe scripts/plot_fortsetzung_fixeddr.py
        -> docs/figures/eval_rlgames/H25_fortsetzung_fixed_dr.{pdf,svg,png}

Each seed's line breaks at the resume (two runs). Three runs: each seed is its own line, no mean and no band (2-4 runs). No smoothing. Raw values, as H6: s1 and s2 log no success rate in
epoch 101 (no episode ended), s5's epochs 101/102 read 1.000/0.987 -- censoring
right after the reset of all environments (RT-253 OUTCOME), not a reading; the
caption names it. The script prints the caption facts.
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

NAME = "H25_fortsetzung_fixed_dr"
SEEDS = (1, 2, 5)
RUN = "RT-253s{}"
SRC = REPO / "rt_logs" / "RT-253s"
CURVES = REPO / "results" / "training_rlgames" / "curves.csv"
DATA = REPO / "results" / "training_rlgames" / "fortsetzung_RT-253.csv"
OUT = REPO / "docs" / "figures" / "eval_rlgames"
COLUMNS = ("method", "seed", "iteration", "env_steps", "Episode/success_rate")
RESUME_AT = 100     # last epoch of the final study; RT-253 resumes from ep_100 (D-209)
LAST_EPOCH = 250    # --max_iterations 250 (RT-253 expectation)


def extract() -> None:
    import eval_figures as EF
    rows = []
    for s in SEEDS:
        run = RUN.format(s)
        r = EF.read_rl_games_run(SRC, run)
        its = [x["iteration"] for x in r]
        if its[0] <= RESUME_AT or its[-1] != LAST_EPOCH:
            raise SystemExit(f"[{NAME}] {run}: epochs {its[0]}..{its[-1]}, expected {RESUME_AT + 1}..{LAST_EPOCH}")
        rows += [["fixeddr", s, x["iteration"], x["env_steps"], x["Episode/success_rate"]] for x in r]
        print(f"[{NAME}] {run}: {len(r)} epochs {its[0]}..{its[-1]}")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"[{NAME}] wrote {DATA} ({len(rows)} rows)")


def load() -> dict:
    """Per seed: epochs and success rate 1..250, joined at RESUME_AT. Refuses a
    gap in the step axis (other factor) or a missing seed."""
    import pandas as pd
    import eval_figures as EF
    head = pd.read_csv(CURVES)
    head = head[head["method"] == "fixeddr"]
    tail = pd.read_csv(DATA)
    if tuple(tail.columns) != COLUMNS:
        raise EF.FigError(f"{DATA} has columns {tuple(tail.columns)}, expected {COLUMNS}")
    factor = EF.iteration_factor(pd.concat([head, tail]))   # one factor over both parts, or FigError
    out = {}
    for s in SEEDS:
        a = head[head["seed"] == s].sort_values("iteration")
        b = tail[tail["seed"] == s].sort_values("iteration")
        if len(a) != RESUME_AT or int(a["iteration"].iloc[-1]) != RESUME_AT or b.empty:
            raise EF.FigError(f"{NAME}: Fixed-DR s{s}: {len(a)} epochs in curves.csv, {len(b)} in {DATA.name}")
        g = pd.concat([a, b])
        out[s] = {"it": g["iteration"].to_numpy(dtype=np.int64),
                  "rate": g["Episode/success_rate"].to_numpy(dtype=float)}
    return {"factor": factor, "seeds": out}


def draw(d: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import eval_figures as EF
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": NAME,
                         "font.family": "sans-serif", "font.sans-serif": [EF.FONT], "font.size": EF.FONT_PT})
    colour = EF.METHOD_COLOR["fixeddr"]
    fig, ax = plt.subplots(figsize=(EF.FIG_W, 3.0))
    ax.axvline(RESUME_AT, color=EF.INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)   # D5: grey, not a method
    # Two runs per seed (RT-243 to 100, RT-253 from 101): two segments, not joined across the resume.
    for s, v in d["seeds"].items():
        for part in (v["it"] <= RESUME_AT, v["it"] > RESUME_AT):
            ax.plot(v["it"][part], 100 * v["rate"][part], color=colour, linewidth=1.2, zorder=2)
    # Direct labels at the right end (C7): s5 alone at the top, s1 and s2 on top of each other at 0.
    end = {s: 100 * v["rate"][-1] for s, v in d["seeds"].items()}
    ax.text(LAST_EPOCH - 2, end[5] - 4, "Seed 5", color=EF.INK, fontsize=EF.FONT_PT, ha="right", va="top")
    # s1 and s2 run on the x axis line at 0 %: an arrow from their label down to them.
    ax.annotate("Seed 1 und Seed 2", xy=(LAST_EPOCH - 30, max(end[1], end[2])),
                xytext=(LAST_EPOCH - 30, max(end[1], end[2]) + 18), color=EF.INK, fontsize=EF.FONT_PT,
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color=EF.INK, linewidth=1.0, shrinkA=2, shrinkB=1))
    ax.set_ylim(0, 102)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(0, LAST_EPOCH + 1)
    ax.set_xticks(range(0, LAST_EPOCH + 1, 50))
    EF._style(ax, EF.YLABEL_TRAINING_RATE, EF.xlabel_epoch(d["factor"]))
    leg = EF._h_legend(fig, [Line2D([], [], color=colour, linewidth=1.2, label="Fixed-DR einzelne Seeds"),
                             Line2D([], [], color=EF.INK_MUTED, linewidth=1.0, linestyle=":",
                                    label=f"senkrechte Linie: Epoche {RESUME_AT},\nEnde der Endstudie")],
                       EF.CURVE_LEGEND_NCOL)
    EF._legend_room(fig, leg, ax, EF.CURVE_AXES_H)
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "svg", "png"):
        p = OUT / f"{NAME}.{ext}"
        fig.savefig(p, dpi=EF.PNG_DPI, metadata=EF.SAVE_META[ext])
        written.append(p)
    plt.close(fig)
    return written


def caption_facts(d: dict) -> None:
    """The numbers the caption names, computed here (no value by hand)."""
    for s, v in d["seeds"].items():
        it, r = v["it"], v["rate"]
        last10 = r[(it > LAST_EPOCH - 10)].mean()
        at100 = r[it == RESUME_AT][0]
        print(f"[{NAME}] caption: Seed {s}: epoch {RESUME_AT} {100 * at100:.1f} %, "
              f"mean epochs {LAST_EPOCH - 9}..{LAST_EPOCH} {100 * last10:.1f} %, epochs {it[0]}..{it[-1]} ({len(it)} values)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract", action="store_true", help="step 1: RT-253 scalars -> results/training_rlgames CSV")
    a = ap.parse_args()
    if a.extract:
        extract()
        return 0
    d = load()
    for p in draw(d):
        print(f"[{NAME}] wrote {p}")
    caption_facts(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
