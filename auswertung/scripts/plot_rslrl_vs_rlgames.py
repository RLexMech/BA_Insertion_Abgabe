# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""H10: the switch to rl_games with Factory's PPO setup (D-200) in training --
RT-238s1 (rsl_rl, Isaac Lab's Cartpole PPO values, LSTM 2 x 256) against
RT-240s1 (rl_games, Factory/FORGE yaml, LSTM 2 x 1024), seed 1 each.

Same task: both runs are the AutoDR arm with the same env command (D-204 not
yet built: start ceiling 0.120 m, fixture spread 0.005 m); `git diff 5512e64
de2c319 -- source` touches only the agent configs and their registration.
What differs is the D-200 bundle -- library, PPO values, network size -- so the
figure shows the switch as a whole, not one cause (D-200, consequence 4).
Both collect 256 envs x 128 steps = 32 768 env steps per iteration/epoch, so
the x axis is total env steps for both (D-195), one seed each, lines only.

Two panels side by side on one y axis: training
success rate and AutoDR opening (bounds_version / 75). A dotted line per run
marks the first step with all 75 opening steps done; the footer names both
values (C9). Both runs are the AutoDR arm, so rl_games keeps the Auto-DR blue
and rsl_rl takes grey, solid (D1, D4, D7). Return and learning rate are read
and printed, not drawn (rsl_rl's learning rate sat at its 1e-5 floor, VERDICTS
RT-238s1). Axis rules: rsl_rl `tools/run_axis.env_steps`; rl_games
`run_axis.rl_games_x` with F = 256 x 128 (`--num_envs 256` in the RT-240s1
command, `horizon_length: 128` in the yaml at `de2c319`), checked against the
CSV's `info/epochs` rows by `run_axis.check_rl_games_epochs`.

    .venv-analysis/Scripts/python.exe scripts/plot_rslrl_vs_rlgames.py [--pair H10|H16]

``--pair H16`` draws the same two panels for the end state of stage C (RT-206s1)
against the end state of stage D (RT-242s1), see PAIRS.
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

run_axis = EF.run_axis   # the one loaded copy of tools/run_axis.py

SCRIPT_MARKER = "plot_rslrl_vs_rlgames-2026-10-08a"
ENVS = 256
BOUNDS_TOTAL = 75.0   # 5 floor rungs + 7 boundaries x 10 steps (RT-240s1 / RT-242s OUTCOME; RT-206s1 max 75)
# Two pairs, one drawing. Each run brings its own steps per iteration (run_axis
# factor = steps x ENVS; all four runs used --num_envs 256).
#  H10: the D-200 switch alone, same task (D.3).
#  H16: the end state of stage C against the end state of stage D, AutoDR seed 1
# NOT the same task: D spreads the fixture +-50 mm in
#       x, y, z instead of +-5 mm in x, y, starts higher and adds the yaw belief
#       error (eval_manifest.json of both studies); the caption must say so (D-200 (4)).
#       RT-206s1: rsl_rl num_steps_per_env 16 at git 5a6cc8e (agents/rsl_rl_ppo_cfg.py:30).
PAIRS = {
    "H10": {"name": "H10_rsl_rl_gegen_rl_games",
            "rsl_rl": (REPO.parent / "p5_mitAktion" / "rt_logs" / "RT-238s1" / "RT-238s1_scalars.csv", 128,
                       "vorher: rsl_rl, PPO-Werte der Isaac-Lab-Vorlage"),
            "rl_games": (REPO / "rt_logs" / "RT-240s1" / "RT-240s1_scalars.csv", 128,
                         "nachher: rl_games, PPO-Konfiguration von Factory")},
    # Run identifiers and configuration differences are given in the caption.
    "H16": {"name": "H16_endstand_c_gegen_d",
            "rsl_rl": (REPO / "docs" / "figures" / "RT-206s1_scalars.csv", 16, "Endstand C"),
            "rl_games": (REPO / "rt_logs" / "RT-242s" / "RT-242s1_scalars.csv", 128, "Endstand D")},
}
# Plot-area height per panel in inches.
AXES_H = 1.86
# rl_games keeps the Auto-DR blue of H5/H6 (the setup every later run uses);
# rsl_rl is the same arm in the older setup: grey, solid (Grafik_Regelwerk D7 --
# the 2026-10-03 ochre read as No-DR's orange, and dashed means ablation, D4).
RUNS = {
    "rsl_rl": {"colour": EF.INK_MUTED, "style": "-",
               "tags": {"success": "Episode/success_rate", "bounds": "dr/bounds_version",
                        "return": "Episode_Reward/total", "lr": "Loss/learning_rate"}},
    "rl_games": {"colour": EF.METHOD_COLOR["autodr"], "style": "-",
                 "tags": {"success": "Episode/success_rate", "bounds": "Episode/dr/bounds_version",
                          "return": "Episode/Episode_Reward/total", "lr": "info/last_lr"}},
}


def read(path: pathlib.Path, lib: str, steps: int) -> dict:
    """{key: (x env steps, values)} for the four tags of one run; ``steps`` = its
    steps per env and iteration/epoch."""
    tags = RUNS[lib]["tags"]
    series = {k: ([], []) for k in tags}
    pairs = []
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        missing = [t for t in tags.values() if t not in rd.fieldnames]
        if missing:
            raise EF.FigError(f"{path.name}: no column {missing}")
        for r in rd:
            step = int(float(r["step"]))
            if lib == "rl_games" and r.get("info/epochs", ""):
                pairs.append((step, float(r["info/epochs"])))
            for k, t in tags.items():
                if r.get(t, "") == "":
                    continue
                if lib == "rsl_rl":
                    x = run_axis.env_steps(step, steps, ENVS)
                else:
                    x = run_axis.rl_games_x(t, step, steps * ENVS)
                if x is not None:
                    series[k][0].append(x)
                    series[k][1].append(float(r[t]))
    if lib == "rl_games":
        try:
            run_axis.check_rl_games_epochs(pairs, steps * ENVS)
        except SystemExit as exc:
            raise EF.FigError(f"{path.name}: {exc}") from None
    out = {}
    for k, (xs, vs) in series.items():
        order = np.argsort(xs, kind="stable")
        x, v = np.asarray(xs)[order], np.asarray(vs, dtype=float)[order]
        if len(np.unique(x)) != len(x) or not np.isfinite(v).all():
            raise EF.FigError(f"{path.name}: {k} has doubled x or empty cells")
        out[k] = (x, v)
    return out


def all_open(s: dict):
    """First env step with all BOUNDS_TOTAL opening steps done, or None."""
    xb, b = s["bounds"]
    return float(xb[np.argmax(b >= BOUNDS_TOTAL)]) if (b >= BOUNDS_TOTAL).any() else None


def draw(plt, d: dict, pair: dict):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(EF.FIG_W, 2.9), sharex=True, sharey=True)
    # Each panel its own y axis word (Grafik_Regelwerk G, T1/T2): one shared
    # "Anteil in %" named neither quantity. Two lines: at 12 pt one line is taller than the panel.
    panels = (("success", EF.YLABEL_TRAINING_RATE.replace(" im Training", "\nim Training"), 100.0),
              ("bounds", EF.YLABEL_OPENING.replace(" Grenzen", "\nGrenzen"), 100.0 / BOUNDS_TOTAL))
    xmax = max(float(d[lib][k][0][-1]) for lib in d for k in ("success", "bounds")) / 1e6
    marks = {lib: all_open(d[lib]) for lib in RUNS}
    for ax, (key, ylabel, k) in zip(axes, panels):
        for lib, spec in RUNS.items():
            x, v = d[lib][key]
            ax.plot(x / 1e6, k * v, color=spec["colour"], linestyle=spec["style"], linewidth=1.44)
            if marks[lib] is not None:
                ax.axvline(marks[lib] / 1e6, color=spec["colour"], linewidth=0.8, linestyle=":")
        ax.set_ylim(0, 102)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlim(0, xmax * 1.02)
        ax.tick_params(labelleft=True)   # sharey hides the right panel's numbers; its axis is another quantity
        EF._style(ax, ylabel, EF.XLABEL_TRAINING)
    handles = [Line2D([], [], color=s["colour"], linestyle=s["style"], linewidth=1.6, label=pair[lib][2])
               for lib, s in RUNS.items()]
    # C9: every dotted line named with its value and unit -- in the legend, the footer
    # line ran past the figure width at 12 pt (2026-10-11).
    text = "\n".join(f"{pair[lib][2].split(':')[0]} nach {EF._num(marks[lib] / 1e6, '.1f')} Mio. Policy-Schritten"
                     for lib in ("rl_games", "rsl_rl") if marks[lib] is not None)
    handles.append(Line2D([], [], color=EF.INK_MUTED, linewidth=0.8, linestyle=":",
                          label=f"gepunktet: alle Grenzen offen,\n{text}"))
    leg = EF._h_legend(fig, handles, 2)
    EF._legend_room(fig, leg, axes[0], AXES_H)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", choices=sorted(PAIRS), default="H10")
    ap.add_argument("--out", default=str(REPO / "docs" / "figures" / "eval_rlgames"))
    args = ap.parse_args()
    pair = PAIRS[args.pair]
    d = {lib: read(pathlib.Path(pair[lib][0]), lib, pair[lib][1]) for lib in RUNS}
    for lib, s in d.items():
        x, sr = s["success"]
        xb, b = s["bounds"]
        full = xb[np.argmax(b >= BOUNDS_TOTAL)] if (b >= BOUNDS_TOTAL).any() else None
        print(f"[{SCRIPT_MARKER}] {lib}: {len(x)} rows, x {x[0]}..{x[-1]}; success at end {sr[-1]:.4f}; "
              f"bounds 75 first at {full} env steps; lr {s['lr'][1].min():.2e}..{s['lr'][1].max():.2e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.findfont(font_manager.FontProperties(family=EF.FONT), fallback_to_default=False)
    plt.rcParams.update({"font.size": EF.FONT_PT, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "eval_figures",
                         "font.family": "sans-serif", "font.sans-serif": [EF.FONT]})
    fig = draw(plt, d, pair)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext, meta in EF.SAVE_META.items():
        p = out / f"{pair['name']}.{ext}"
        fig.savefig(p, dpi=EF.PNG_DPI, bbox_inches="tight", metadata=meta)
        print(f"[{SCRIPT_MARKER}] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
