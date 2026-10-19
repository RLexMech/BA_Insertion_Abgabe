# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""H9: the MLP ablation (D-214, RT-260s) beside the LSTM (RT-242s) in training.

Four panels over the training iteration, mean line and the 95 % bootstrap
interval of that mean as band over the five seeds per network (D-218), each
seed as a thin line under them (D-219): training success rate, return per episode,
episode length, and how far AutoDR has opened its boundaries. Both chains are
the AutoDR arm with the same command, seeds and spread; only the memory
differs, so the TRAINING curves may stand side by side here (unlike across
arms, H5/H6). Laptop only, in ``.venv-analysis`` (D-189).

A thin caller of ``eval_figures.py`` like ``plot_seed_curves.py``: the
per-run reading (``read_rl_games_run``, axis and epoch guard), the mean
(``_mean_std``), the interval (``_boot_ci``), the band, the axis style, the font and the save
metadata are its. OWN here, named: two columns ``read_rl_games_run`` does not
carry -- ``Episode/dr/bounds_version`` (real AutoDR moves; 75 = all seven
boundaries at their maximum after the five floor rungs, RT-242s/RT-260s
OUTCOME) and rl_games' ``episode_lengths/iter`` -- read from the same
``scalars.csv`` on the same epoch rows, and refused when a row is missing.

    .venv-analysis/Scripts/python.exe scripts/plot_mlp_vs_lstm.py
    .venv-analysis/Scripts/python.exe scripts/plot_mlp_vs_lstm.py --ablation force   # H11, D-215
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import eval_figures as EF  # noqa: E402

SCRIPT_MARKER = "plot_mlp_vs_lstm-2026-10-06b"
# LSTM keeps the Auto-DR blue of H5/H6 (it is that arm); the MLP takes the
# ablation colour, dashed (eval_figures.ABLATION_COLOR, its one home with the
# validation note). --ablation force (H11, D-215, 2026-10-04): the same form
# and pair for the force-sensor ablation -- RT-242s (actor sees force) against
# RT-261s (actor blind to force, critic keeps it); colour, line and label are
# eval_figures.FORCE_ARM_STYLE, which H12 (D-216) draws too.
ABLATIONS = {
    "lstm": ("H9_mlp_gegen_lstm", (
        ("lstm", "RT-242s", "mit LSTM", EF.METHOD_COLOR["autodr"], "-"),
        ("mlp", "RT-260s", "ohne LSTM (nur MLP)", EF.ABLATION_COLOR, "--"),
    )),
    "force": ("H11_mit_gegen_ohne_kraft", (
        ("force", "RT-242s", EF.FORCE_ARM_STYLE["autodr"][2], *EF.FORCE_ARM_STYLE["autodr"][:2]),
        ("noforce", "RT-261s", EF.FORCE_ARM_STYLE["autodr_nf"][2], *EF.FORCE_ARM_STYLE["autodr_nf"][:2]),
    )),
}
NAME, NETS = ABLATIONS["lstm"]
BOUNDS_TOTAL = 75.0   # 5 floor rungs + 7 boundaries x 10 steps (autodr.DR_DIMS, RT-242s OUTCOME P3)
# Plot-area height per panel in inches.
AXES_H = 1.79
EXTRA = {"bounds": "Episode/dr/bounds_version", "length": "episode_lengths/iter"}


def _extra_columns(path: pathlib.Path, epochs: np.ndarray) -> dict:
    """bounds_version and episode length on the given epochs (step = epoch on
    both tags' rows, as on read_rl_games_run's Episode/* rows)."""
    got = {k: {} for k in EXTRA}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            step = int(float(r["step"]))
            for k, col in EXTRA.items():
                if r.get(col, "") != "":
                    got[k][step] = float(r[col])
    out = {}
    for k in EXTRA:
        missing = [int(e) for e in epochs if int(e) not in got[k]]
        if missing:
            raise EF.FigError(f"{path.name}: {EXTRA[k]} missing on epochs {missing[:5]}")
        out[k] = np.array([got[k][int(e)] for e in epochs], dtype=float)
    return out


def load_net(rt_logs: pathlib.Path, prefix: str) -> dict:
    folder = rt_logs / prefix
    runs = EF._rl_games_runs(folder, prefix)
    its = steps = None
    per = {"success": [], "return": [], "length": [], "bounds": []}
    for run in runs:
        rows = EF.read_rl_games_run(folder, run)
        it = np.array([r["iteration"] for r in rows], dtype=np.int64)
        x = np.array([r["env_steps"] for r in rows], dtype=np.int64)
        if its is None:
            its, steps = it, x
        elif not (np.array_equal(it, its) and np.array_equal(x, steps)):
            raise EF.FigError(f"{run}: another epoch grid than {runs[0]}")
        per["success"].append([float(r["Episode/success_rate"]) for r in rows])
        per["return"].append([float(r[EF.RETURN_TAG]) for r in rows])
        extra = _extra_columns(folder / f"{run}_scalars.csv", it)
        per["length"].append(extra["length"])
        per["bounds"].append(extra["bounds"] / BOUNDS_TOTAL)
    out = {"it": its, "x": steps, "runs": runs}
    for k, v in per.items():
        a = np.asarray(v, dtype=float)
        if not np.isfinite(a).all():
            raise EF.FigError(f"{prefix}: {int((~np.isfinite(a)).sum())} empty {k} cells")
        # mean, its interval, and the seeds themselves (thin lines, D-219)
        out[k] = (EF._mean_std(a, NAME)[0], EF._boot_ci(a, NAME), a)
    return out


def draw(plt, d: dict, factor: int):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(EF.FIG_W, 4.8), sharex=True)
    panels = (
        ("success", "Erfolgsrate im Training", "in %", True),
        ("return", "Return je Episode", "Reward-Summe", False),
        ("length", "Episodenlänge", "Schritte", False),
        # T2's word (Grafik_Regelwerk G, 2026-10-11); "of the 75 steps" goes to the caption.
        ("bounds", EF.YLABEL_OPENING.removesuffix(" in %"), "in %", True),
    )
    for ax, (key, title, unit, rate) in zip(axes.flat, panels):
        k = 100.0 if rate else 1.0
        # The seeds first (D-219), then both bands, then both means (as eval_figures F7).
        for net, _, _, colour, style in NETS:
            EF._seed_lines(ax, d[net]["it"], list(d[net][key][2]), colour, rate=rate, linestyle=style)
        for net, _, _, colour, _ in NETS:
            _mean, ci, _seeds = d[net][key]
            EF._band(ax, d[net]["it"], ci, colour, rate=rate)
        for net, _, _, colour, style in NETS:
            mean, _ci, _seeds = d[net][key]
            ax.plot(d[net]["it"], k * mean, color=colour, linewidth=1.44, linestyle=style)
        if rate:
            ax.set_ylim(0, 102)
            ax.set_yticks(range(0, 101, 20))
        ax.set_xlim(0, max(int(v["it"][-1]) for v in d.values()) + 1)
        ax.set_title(title, fontsize=EF.FONT_PT, color=EF.INK, loc="left")
        EF._style(ax, unit)
    for ax in axes[1]:
        # Two lines: at 12 pt one line is wider than the panel (2026-10-11).
        ax.set_xlabel(EF.xlabel_epoch(factor).replace(" (", "\n("), color=EF.INK, fontsize=EF.FONT_PT)
    n = {len(v["runs"]) for v in d.values()}
    if len(n) != 1:
        raise EF.FigError(f"{NAME}: seed counts differ {sorted(n)}")
    handles = [Line2D([], [], color=c, linewidth=1.6, linestyle=s, label=lab) for _, _, lab, c, s in NETS]
    leg = EF._h_legend(fig, handles + list(EF._curve_extra(n.pop())), EF.CURVE_LEGEND_NCOL)
    EF._legend_room(fig, leg, axes[0, 0], AXES_H)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rt-logs", default=str(REPO / "rt_logs"))
    ap.add_argument("--out", default=str(REPO / "docs" / "figures" / "eval_rlgames"))
    ap.add_argument("--ablation", choices=sorted(ABLATIONS), default="lstm",
                    help="lstm = H9 (RT-242s vs RT-260s, default); force = H11 (RT-242s vs RT-261s)")
    args = ap.parse_args()
    global NAME, NETS
    NAME, NETS = ABLATIONS[args.ablation]
    a, b = NETS[0][0], NETS[1][0]
    d = {}
    for net, prefix, *_ in NETS:
        d[net] = load_net(pathlib.Path(args.rt_logs), prefix)
        v = d[net]
        print(f"[{SCRIPT_MARKER}] {net}: {', '.join(v['runs'])}; epochs {v['it'][0]}..{v['it'][-1]}; "
              f"at 100: success {v['success'][0][-1]:.4f}, return {v['return'][0][-1]:.1f}, "
              f"length {v['length'][0][-1]:.1f}, bounds {v['bounds'][0][-1]:.3f}")
    if not np.array_equal(d[a]["it"], d[b]["it"]) or not np.array_equal(d[a]["x"], d[b]["x"]):
        raise EF.FigError(f"{NAME}: the two chains log different epochs or env steps")
    steps = np.diff(d[a]["x"])
    if steps.size == 0 or not (steps == steps[0]).all():
        raise EF.FigError(f"{NAME}: env steps per iteration are not constant")
    factor = int(steps[0])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.findfont(font_manager.FontProperties(family=EF.FONT), fallback_to_default=False)
    plt.rcParams.update({"font.size": EF.FONT_PT, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "eval_figures",
                         "font.family": "sans-serif", "font.sans-serif": [EF.FONT]})
    fig = draw(plt, d, factor)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext, meta in EF.SAVE_META.items():
        p = out / f"{NAME}.{ext}"
        fig.savefig(p, dpi=EF.PNG_DPI, bbox_inches="tight", metadata=meta)
        print(f"[{SCRIPT_MARKER}] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
