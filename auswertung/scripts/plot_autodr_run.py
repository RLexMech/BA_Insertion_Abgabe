# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Eight report figures for ONE AutoDR run (RT-206 series; rl_games runs since
D-200), PNG + PDF.

WHAT IT DRAWS (``<out-dir>/<run>_<name>.png`` and ``.pdf``)
------------------------------------------------------------
    erfolg     success rate over env steps, phase-dr and all-at-max marked
    stufen     AutoDR boundary progress per dimension, 0 % = centre, 100 % = maximum
               (small multiples; start floor as its own panel)
    kraft      per-episode peak force: mean and p95 with the abort limit;
               below it the force-abort rate (two panels, never two y-axes)
    ppo        episode reward, surrogate loss, value loss, entropy
               (EV and KL are not logged, D-116)
    physik     max depth (with the pocket-gate floor), below-fixture rate,
               engaged reward, mean interpenetration
    endfenster final-window bin tables of demo_metrics.json: success per
               start height / tilt / yaw / lateral bin, force abort per height
    randerfolg success rate of the last FULL boundary buffer per AutoDR key
               (``dr/<key>_last_rate``, -1 = none yet, not drawn) with the
               advance / retreat thresholds: which boundary stalls
    dauer      steps to success per iteration from ``episodes.csv`` (median
               and p90, in s = steps x ``run_axis.read_policy_dt``), episode cap
               dashed; only with ``--episodes`` and ``--episodes-meta``

RL_GAMES RUNS: the library is told from ``--agent-yaml``
(``run_axis.read_run_factor``); the x of every tag follows its rl_games step
axis (``run_axis.rl_games_x``); the AutoDR state is the checkpoint's
``env_state`` saved as json (D-200), whose
``autodr`` key has the same form as the rsl_rl sidecar's. Tag names: see
``RL_GAMES_TAG``.

WHERE THE NUMBERS COME FROM
---------------------------
* curves: the run's ``scalars.csv`` (``scripts/export_tb_scalars.py``). Only
  rows that carry ``Episode/success_rate`` are iteration rows; the others
  hold the two ``*/time`` tags only.
* x axis: TOTAL env steps, all envs summed (D-195 CURVE AXIS,
  ``tools/run_axis.py``), factor from the run's own ``params/agent.yaml``
  and ``params/env.yaml``. The event markers take the same x. Wall time is
  one console line, never a figure.
* normalisation of the boundaries: ``dims`` and ``floor`` of the run's
  ``autodr_<it>.json`` sidecar -- never typed here.
* abort limit and gate floor: ``demo_metrics.json`` (``force_norm_n.f_max_n``,
  ``below_fixture_rate.underside_depth_mm``).
A missing tag or key stops the script with its name (fail fast).

    python scripts/plot_autodr_run.py --scalars docs/figures/RT-206s1_scalars.csv \
        --metrics docs/figures/RT-206s1_demo_metrics.json \
        --autodr docs/figures/RT-206s1_autodr_1499.json --run RT-206s1 \
        --agent-yaml rt_logs/RT-206s1/RT-206s1_agent.yaml --env-yaml rt_logs/RT-206s1/RT-206s1_env.yaml \
        --out-dir docs/figures
    python scripts/plot_autodr_run.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import tempfile

# Loaded by path because scripts/tools is not a package (author_tool_ur5e.py idiom).
_spec = importlib.util.spec_from_file_location("run_axis", pathlib.Path(__file__).resolve().parent / "tools" / "run_axis.py")
run_axis = importlib.util.module_from_spec(_spec)
sys.modules["run_axis"] = run_axis
_spec.loader.exec_module(run_axis)

XLABEL = "Policy-Schritte in Mio."   # D-195 CURVE AXIS
# fig_stufen: plot-area height per panel in inches (RT-201_stufen).
STUFEN_AXES_H = 1.51
RUN = "#3B6FD4"   # series 1 (plot_run_curves.py precedent, validated with REF)
REF = "#D97706"   # series 2
INK = "#1F2933"
INK_MUTED = "#6B7280"
GRID = "#E5E7EB"

NAMES = ("erfolg", "stufen", "kraft", "ppo", "physik", "endfenster", "randerfolg", "dauer")

# rsl_rl tag -> rl_games tag. rl_games writes every env key as "Episode/<key>"
# (IsaacAlgoObserver, algo_observer.py:127 @ 6b3534f2); rsl_rl 3.0.1 wrote a key
# with "/" as is. The trainer's tags have their own names. Value loss = the
# separate critic's, whose values give the advantages (a2c_common.py:428-429,
# central_value.py:236); losses/c_loss is the actor's auxiliary value head.
RL_GAMES_TAG = {
    "Train/mean_reward": "rewards/iter",
    "Loss/surrogate": "losses/a_loss",
    "Loss/value_function": "losses/cval_loss",
    "Loss/entropy": "losses/entropy",
}
# Words, font, size and resolution from eval_figures (Grafik_Regelwerk F2 and G, 2026-10-11).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import eval_figures as EF  # noqa: E402

PANEL_LABEL = dict(EF.F8_PANELS)   # thesis words (Tab. 4-1), incl. "start_height_floor"
DIM_LABEL = {k: v for k, v in PANEL_LABEL.items() if k != "start_height_floor"}


def load_curves(path: pathlib.Path, factor: tuple[int, int]) -> dict:
    """``{tag: (x, y)}`` over the iteration rows, x in total env steps with the
    run's own ``factor = (num_steps_per_env, num_envs)``; prints the run's
    wall-time line. Every figure and event marker takes its x from here."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "step" not in rows[0]:
        raise SystemExit(f"{path}: no 'step' column")
    out: dict[str, tuple[list, list]] = {}
    its, walls = [], []
    for r in rows:
        if r.get("Episode/success_rate", "") == "":
            continue
        s = int(float(r["step"]))
        its.append(s)
        walls.append(float(r["wall_time"]))
        x = run_axis.env_steps(s, *factor)
        for k, v in r.items():
            if k in ("step", "wall_time") or v in ("", None):
                continue
            xs, ys = out.setdefault(k, ([], []))
            xs.append(x)
            ys.append(float(v))
    print(f"[plot_autodr_run] {path.name}: " + run_axis.wall_time_line(its, walls))
    return out


def load_curves_rl_games(path: pathlib.Path, factor: tuple[int, int]) -> dict:
    """``{tag: (x, y)}`` of an rl_games run: every tag on its own step axis
    (``run_axis.rl_games_x``), read per TAG because the axes share rows."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "step" not in rows[0]:
        raise SystemExit(f"{path}: no 'step' column")
    F = factor[0] * factor[1]
    epochs = [(int(float(r["step"])), float(r["info/epochs"])) for r in rows if r.get("info/epochs", "") != ""]
    run_axis.check_rl_games_epochs(epochs, F)
    pts: dict[str, list] = {}
    wall = 0
    for r in rows:
        s = int(float(r["step"]))
        for k, v in r.items():
            if k in ("step", "wall_time") or v in ("", None):
                continue
            x = run_axis.rl_games_x(k, s, F)
            if x is None:
                wall = max(wall, s)
            else:
                pts.setdefault(k, []).append((x, float(v)))
    out = {k: ([x for x, _ in sorted(p)], [y for _, y in sorted(p)]) for k, p in pts.items()}
    print(f"[plot_autodr_run] {path.name}: rl_games, {len(epochs)} epochs x F = {F}; "
          f"wall time {wall / 3600:.2f} h (largest */time step)")
    return out


def need(curves: dict, tag: str):
    for t in (tag, RL_GAMES_TAG.get(tag, "Episode/" + tag)):
        if t in curves:
            return curves[t]
    raise SystemExit(f"tag {tag!r} missing from scalars.csv")


def first_step(curves: dict, tag: str, at_least: float):
    xs, ys = need(curves, tag)
    for x, y in zip(xs, ys):
        if y >= at_least:
            return x
    return None


def style(ax, ylabel: str, xlabel: bool = True):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, labelsize=EF.FONT_PT)   # numbers black (Grafik_Regelwerk C5)
    ax.set_ylabel(ylabel, color=INK, fontsize=EF.FONT_PT)
    if xlabel:
        env_axis(ax)


def mio(x: float, digits: int | None = None) -> str:
    """Env steps as millions with a decimal comma (German thesis text)."""
    s = f"{x / 1e6:g}" if digits is None else f"{x / 1e6:.{digits}f}"
    return s.replace(".", ",")


def env_axis(ax):
    """The D-195 x axis label, ticks in millions instead of matplotlib's 1e6 offset."""
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: mio(v)))
    ax.set_xlabel(XLABEL, color=INK, fontsize=EF.FONT_PT)


def mark_events(ax, events):
    for x, text in events:
        if x is None:
            continue
        ax.axvline(x, color=INK_MUTED, linewidth=0.8, linestyle=":")
        ax.text(x, 1.0, " " + text, transform=ax.get_xaxis_transform(), color=INK_MUTED,
                fontsize=7, va="top", ha="left")


def fig_erfolg(plt, c, ev):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    xs, ys = need(c, "Episode/success_rate")
    ax.plot(xs, [100 * y for y in ys], color=RUN, linewidth=1.2)
    ax.set_ylim(0, 100)
    style(ax, "Erfolgsrate [%]")
    mark_events(ax, ev)
    return fig


def fig_stufen(plt, c, sidecar, ev):
    from matplotlib.lines import Line2D
    ad = sidecar["autodr"]
    dims = {d["name"]: d for d in ad["dims"]}
    fl = ad["floor"]
    panels = ([("start_height_floor", PANEL_LABEL["start_height_floor"], None)]
              + [(n, DIM_LABEL.get(n, n), dims[n]) for n in dims])
    fig, axes = plt.subplots(2, 3, figsize=(EF.FIG_W, 4.4), sharex=True, sharey=True)
    for ax, (name, label, d) in zip(axes.flat, panels):
        if d is None:
            xs, ys = need(c, "dr/start_height_floor")
            ax.plot(xs, [100 * (y - fl["f0"]) / (fl["top"] - fl["f0"]) for y in ys], color=RUN, linewidth=1.2)
        else:
            # Upper and lower limit of a two-sided key move almost together, so each
            # line carries its own symbol (filled circle / open triangle), evenly spaced
            # along the line and shifted by half a spacing so the two never coincide.
            # Lower limit in grey, not REF: orange means No-DR in every other figure
            # (Grafik_Regelwerk D1, D7).
            hx, hy = need(c, f"dr/{name}_hi")
            ax.plot(hx, [100 * (y - d["centre"]) / (d["hi_max"] - d["centre"]) for y in hy], color=RUN,
                    linewidth=1.2, marker="o", markersize=2.5, markevery=(0.0, 0.1), label="obere Grenze")
            if not d["one_sided"]:
                lx, ly = need(c, f"dr/{name}_lo")
                ax.plot(lx, [100 * (y - d["centre"]) / (d["lo_max"] - d["centre"]) for y in ly], color=INK_MUTED,
                        linewidth=1.2, marker="^", markersize=3.5, markerfacecolor="none", markeredgewidth=0.9,
                        markevery=(0.05, 0.1), label="untere Grenze")
        # 12 pt (2026-10-11): a title with a comma breaks there, else it is wider than its panel.
        ax.set_title(label.replace(", ", ",\n"), fontsize=EF.FONT_PT, color=INK)
        ax.set_ylim(-5, 108)
        style(ax, "", xlabel=False)
        for x, _ in ev:
            if x is not None:
                ax.axvline(x, color=INK_MUTED, linewidth=0.8, linestyle=":")
    from matplotlib.ticker import MultipleLocator
    for ax in axes[1]:
        env_axis(ax)
        ax.set_xlabel("")
        ax.xaxis.set_major_locator(MultipleLocator(1e6))   # at 12 pt the default showed only 0 and 2 Mio.
    # One x title under the middle panel and the y title in two lines: at 12 pt they are
    # wider / taller than one panel (2026-10-11).
    axes[1][1].set_xlabel(XLABEL, color=INK, fontsize=EF.FONT_PT)
    for ax in axes[:, 0]:
        ax.set_ylabel(EF.YLABEL_OPENING.replace(" Grenzen", "\nGrenzen"), color=INK, fontsize=EF.FONT_PT)
    handles, labels = axes.flat[2].get_legend_handles_labels()
    # The dotted lines in the legend, each with its value (C9); the footer line ran past the width at 12 pt.
    marks = ",\n".join(f"{text}\nbei {mio(x, 2)} Mio. Policy-Schritten" for x, text in ev if x is not None)
    handles.append(Line2D([], [], color=INK_MUTED, linewidth=0.8, linestyle=":"))
    labels.append(f"gepunktet: {marks}")
    leg = fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=EF.FONT_PT)
    EF._legend_room(fig, leg, axes[0][0], STUFEN_AXES_H)
    return fig


def fig_kraft(plt, c, metrics, ev):
    f_max = float(metrics["force_norm_n"]["f_max_n"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True)
    for tag, col, lab in (("Episode/force_max_mean_n", RUN, "Mittel der Episoden-Spitzen"),
                          ("Episode/force_max_p95_n", REF, "p95 der Episoden-Spitzen")):
        xs, ys = need(c, tag)
        a1.plot(xs, ys, color=col, linewidth=1.2, label=lab)
    a1.axhline(f_max, color=INK, linewidth=0.9, linestyle="--")
    a1.text(1.0, f_max, f"Abbruchgrenze {f_max:g} N ", color=INK, fontsize=7, va="top", ha="right",
            transform=a1.get_yaxis_transform())
    a1.set_ylim(bottom=0)
    style(a1, "Kraft [N]", xlabel=False)
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    mark_events(a1, ev)
    xs, ys = need(c, "Episode/force_abort_rate")
    a2.plot(xs, [100 * y for y in ys], color=RUN, linewidth=1.2)
    a2.set_ylim(bottom=0)
    style(a2, "Kraftabbrüche [%]")
    for x, _ in ev:
        if x is not None:
            a2.axvline(x, color=INK_MUTED, linewidth=0.8, linestyle=":")
    fig.tight_layout()
    return fig


def fig_grid(plt, c, spec, ev):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for ax, (tag, label, scale, hline) in zip(axes.flat, spec):
        xs, ys = need(c, tag)
        ax.plot(xs, [scale * y for y in ys], color=RUN, linewidth=1.0)
        if hline is not None:
            ax.axhline(hline[0], color=INK, linewidth=0.9, linestyle="--")
            ax.text(0.0, hline[0], " " + hline[1], color=INK, fontsize=7, va="bottom",
                    transform=ax.get_yaxis_transform())
        style(ax, label, xlabel=False)
        for x, _ in ev:
            if x is not None:
                ax.axvline(x, color=INK_MUTED, linewidth=0.8, linestyle=":")
    for ax in axes[1]:
        env_axis(ax)
    fig.tight_layout()
    return fig


def _bars(ax, labels, rates, counts, ylabel, pct_max=100.0):
    pos = list(range(len(labels)))
    ax.bar(pos, [100 * r for r in rates], color=RUN, width=0.7)
    for p, r, n in zip(pos, rates, counts):
        ax.text(p, 100 * r, f"{100 * r:.1f}\nn={n}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7, color=INK_MUTED, rotation=35, ha="right")
    ax.set_ylim(0, pct_max)
    style(ax, ylabel, xlabel=False)


def _bin_rows(metrics, key, lo, hi, rate_key):
    if key not in metrics:
        raise SystemExit(f"key {key!r} missing from demo_metrics.json")
    rows = [b for b in metrics[key] if b["episodes"]]
    empty = len(metrics[key]) - len(rows)
    labels = [f"{b[lo]:g}–{b[hi]:g}" for b in rows]
    return labels, [b[rate_key] for b in rows], [b["episodes"] for b in rows], empty


def fig_endfenster(plt, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0))
    spec = [
        ("success_by_start_height_bin", "start_mm_from", "start_mm_to", "success_rate", "Starthöhe [mm]", "Erfolg [%]", 115),
        ("force_abort_by_start_height_bin", "start_mm_from", "start_mm_to", "force_abort_rate", "Starthöhe [mm]", "Kraftabbruch [%]", None),
        ("success_by_tilt_bin", "tilt_deg_from", "tilt_deg_to", "success_rate", "Kippung [°]", "Erfolg [%]", 115),
        ("success_by_yaw_bin", "yaw_deg_from", "yaw_deg_to", "success_rate", "|Gierwinkel| [°]", "Erfolg [%]", 115),
        ("success_by_lateral_bin", "lateral_mm_from", "lateral_mm_to", "success_rate", "Seitl. Versatz [mm]", "Erfolg [%]", 115),
    ]
    for ax, (key, lo, hi, rk, xl, yl, top) in zip(axes.flat, spec):
        labels, rates, counts, empty = _bin_rows(metrics, key, lo, hi, rk)
        if top is None:
            top = max(5.0, 100 * max(rates) * 1.6) if rates else 5.0
        _bars(ax, labels, rates, counts, yl, top)
        ax.set_xlabel(xl, color=INK, fontsize=8)
        if empty:
            print(f"[plot_autodr_run] {key}: {empty} empty bins not drawn")
    axes.flat[5].axis("off")
    fig.tight_layout()
    return fig


def key_label(key: str) -> str:
    if key == "start_height_floor":
        return "Startboden"
    dim, side = key.rsplit("_", 1)
    return f"{DIM_LABEL.get(dim, dim)}, {'oben' if side == 'hi' else 'unten'}"


def fig_randerfolg(plt, c, sidecar, ev):
    ad = sidecar["autodr"]
    keys = list(ad["keys"])
    adv, ret = float(ad["advance_at"]), float(ad["retreat_at"])
    nrows = math.ceil(len(keys) / 3)
    fig, axes = plt.subplots(nrows, 3, figsize=(7.2, 2.0 * nrows + 0.5), sharex=True, sharey=True, squeeze=False)
    for ax, key in zip(axes.flat, keys):
        xs, ys = need(c, f"dr/{key}_last_rate")
        pts = [(x, 100 * y) for x, y in zip(xs, ys) if y >= 0]   # -1 = no full buffer yet
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=RUN, linewidth=1.0)
        else:
            ax.text(0.5, 0.5, "kein voller Puffer", transform=ax.transAxes, ha="center", fontsize=7,
                    color=INK_MUTED)
        for lvl in (adv, ret):
            ax.axhline(100 * lvl, color=INK, linewidth=0.8, linestyle="--")
        _, hy = need(c, f"dr/{key}_holds")
        print(f"[plot_autodr_run] randerfolg {key}: last rate "
              f"{f'{pts[-1][1]:.1f} %' if pts else 'none'}, holds {hy[-1]:g}")
        ax.set_title(key_label(key), fontsize=8, color=INK)
        ax.set_ylim(-3, 103)
        style(ax, "", xlabel=False)
        for x, _ in ev:
            if x is not None:
                ax.axvline(x, color=INK_MUTED, linewidth=0.8, linestyle=":")
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    for ax in axes[-1]:
        env_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Erfolg am Rand [%]", color=INK, fontsize=9)
    fig.text(0.5, 0.005, f"gestrichelt: vor ab {100 * adv:g} %, zurück bis {100 * ret:g} %",
             ha="center", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def dauer_points(rows, factor, lib: str) -> list[tuple[int, int, int]]:
    """``(x, median steps, p90 steps)`` per iteration over the successful
    episodes of ``episodes.csv``. rsl_rl: episode 0 of each env is dropped
    (init_at_random_ep_len, its steps is not its length); rl_games starts no
    env at a random step (D-200)."""
    by_it: dict[int, list[int]] = {}
    for r in rows:
        if r["outcome"] != "success" or (lib == "rsl_rl" and int(r["episode"]) == 0):
            continue
        it = int(r["iteration"])
        if it < 0:
            raise SystemExit("episodes.csv: iteration -1 (train.py did not hand the rollout length over)")
        by_it.setdefault(it, []).append(int(r["steps"]))
    out = []
    for it in sorted(by_it):
        v = sorted(by_it[it])
        # p90 with the index rule of _force_stats (insertion_env.py)
        out.append((run_axis.env_steps(it, *factor), int(statistics.median(v)),
                    v[min(len(v) - 1, int(math.ceil(0.9 * len(v))) - 1)]))
    return out


def fig_dauer(plt, pts, dt, cap_steps, ev):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    xs = [p[0] for p in pts]
    ax.plot(xs, [p[1] * dt for p in pts], color=RUN, linewidth=1.2, label="Median")
    ax.plot(xs, [p[2] * dt for p in pts], color=REF, linewidth=1.2, label="p90")
    ax.axhline(cap_steps * dt, color=INK, linewidth=0.9, linestyle="--")
    ax.text(1.0, cap_steps * dt, f"Episodenlimit {cap_steps * dt:g} s ", color=INK, fontsize=7, va="top",
            ha="right", transform=ax.get_yaxis_transform())
    ax.set_ylim(bottom=0)
    style(ax, "Dauer bis Erfolg [s]")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    mark_events(ax, ev)
    return fig


def events(c: dict) -> list:
    """The two dotted markers every curve figure carries (thesis word: Startcurriculum)."""
    return [(first_step(c, "dr/phase", 1.0), "Startcurriculum fertig"),
            (first_step(c, "dr/all_at_max", 1.0), "alle Grenzen am Maximum")]


def draw_all(scalars, factor, metrics_path, autodr_path, run, out_dir, lib="rsl_rl",
             episodes=None, episodes_meta=None, dt=None, only=None) -> list:
    """``only``: figure names (NAMES) to draw; None draws every figure the inputs allow
    (2026-10-11: RT-201's demo_metrics.json has no below_fixture_rate, so only "stufen")."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # The thesis font (Grafik_Regelwerk C8), as eval_figures.draw_all sets it.
    plt.rcParams.update({"font.size": EF.FONT_PT, "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "eval_figures",
                         "font.family": "sans-serif", "font.sans-serif": [EF.FONT]})

    c = load_curves_rl_games(scalars, factor) if lib == "rl_games" else load_curves(scalars, factor)
    metrics = json.loads(pathlib.Path(metrics_path).read_text(encoding="utf-8"))
    sidecar = json.loads(pathlib.Path(autodr_path).read_text(encoding="utf-8"))
    ev = events(c)
    want = (lambda n: True) if not only else (lambda n: n in only)

    def physik():
        gate = float(metrics["below_fixture_rate"]["underside_depth_mm"])
        return fig_grid(plt, c, [
            ("Episode/max_depth_max_mm", "Größte Tiefe [mm]", 1.0, (gate, f"Unterseite Aufnahme {gate:g} mm")),
            ("Episode/below_fixture_rate", "Unter der Aufnahme [%]", 100.0, None),
            ("Episode_Reward/engaged", "Reward-Term engaged [–]", 1.0, None),
            ("Episode/interpen_max_mean_mm", "Durchdringung, Mittel [mm]", 1.0, None),
        ], ev)

    makers = {
        "erfolg": lambda: fig_erfolg(plt, c, ev),
        "stufen": lambda: fig_stufen(plt, c, sidecar, ev),
        "kraft": lambda: fig_kraft(plt, c, metrics, ev),
        "ppo": lambda: fig_grid(plt, c, [
            ("Train/mean_reward", "Episoden-Reward [–]", 1.0, None),
            ("Loss/surrogate", "Policy-Loss [–]", 1.0, None),
            ("Loss/value_function", "Value-Loss [–]", 1.0, None),
            ("Loss/entropy", "Entropie [–]", 1.0, None),
        ], ev),
        "physik": physik,
        "endfenster": lambda: fig_endfenster(plt, metrics),
    }
    figs = {n: make() for n, make in makers.items() if want(n)}
    keys = sidecar["autodr"].get("keys") or []
    first = f"dr/{keys[0]}_last_rate" if keys else None
    try:
        has_rate = first is not None and need(c, first) is not None
    except SystemExit:
        has_rate = False
    if not want("randerfolg"):
        pass
    elif has_rate:
        figs["randerfolg"] = fig_randerfolg(plt, c, sidecar, ev)
    else:
        print(f"[plot_autodr_run] randerfolg: skipped, no AutoDR keys in the state or no {first!r} "
              f"in scalars.csv (logged since 2026-09-14)")
    if not want("dauer"):
        pass
    elif episodes is None or episodes_meta is None:
        print("[plot_autodr_run] dauer: skipped, no --episodes / --episodes-meta")
    else:
        with open(episodes, newline="", encoding="utf-8") as f:
            pts = dauer_points(csv.DictReader(f), factor, lib)
        meta = json.loads(pathlib.Path(episodes_meta).read_text(encoding="utf-8"))
        figs["dauer"] = fig_dauer(plt, pts, dt, int(meta["episode_cap_steps"]), ev)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in [n for n in NAMES if n in figs]:
        fig = figs[name]
        for ext in ("png", "pdf", "svg"):   # 300 dpi + vector (Grafik_Regelwerk E3)
            p = out_dir / f"{run}_{name}.{ext}"
            fig.savefig(p, dpi=EF.PNG_DPI, bbox_inches="tight", metadata=EF.SAVE_META[ext])
            written.append(p)
        plt.close(fig)
    for p in written:
        print(f"[plot_autodr_run] wrote {p}")
    return written


def _synthetic(tmp: pathlib.Path, drop_tag: str | None = None, lib: str = "rsl_rl"):
    dims = [
        {"name": "lat_r", "centre": 0.0, "lo_max": 0.0, "hi_max": 0.03, "one_sided": True},
        {"name": "yaw", "centre": 0.0, "lo_max": -0.1, "hi_max": 0.1, "one_sided": False},
        {"name": "tilt", "centre": 0.0, "lo_max": 0.0, "hi_max": 0.2, "one_sided": True},
        {"name": "start_height", "centre": 0.02, "lo_max": 0.02, "hi_max": 0.12, "one_sided": True},
        {"name": "friction", "centre": 0.14, "lo_max": 0.08, "hi_max": 0.2, "one_sided": False},
    ]
    keys = [f"{d['name']}_{s}" for d in dims for s in (("hi",) if d["one_sided"] else ("lo", "hi"))]
    keys.append("start_height_floor")
    (tmp / "a.json").write_text(json.dumps({"autodr": {"dims": dims, "floor": {"f0": -0.03, "top": 0.02},
                                                       "keys": keys, "advance_at": 0.8, "retreat_at": 0.5}}))
    tags = ["Episode/success_rate", "dr/phase", "dr/all_at_max", "dr/start_height_floor",
            "Episode/force_max_mean_n", "Episode/force_max_p95_n", "Episode/force_abort_rate",
            "Train/mean_reward", "Loss/surrogate", "Loss/value_function", "Loss/entropy",
            "Episode/max_depth_max_mm", "Episode/below_fixture_rate", "Episode_Reward/engaged",
            "Episode/interpen_max_mean_mm"]
    for d in dims:
        tags.append(f"dr/{d['name']}_hi")
        if not d["one_sided"]:
            tags.append(f"dr/{d['name']}_lo")
    tags += [f"dr/{k}_{s}" for k in keys for s in ("last_rate", "holds")]
    tags = [t for t in tags if t != drop_tag]
    val = lambda t, i: min(1.0, i / 20) if "/" in t else 0
    if lib == "rl_games":
        # D-200: one wide table, three step axes (run_axis docstring), F = 16 * 8 = 128.
        # rewards/time at 10 * epoch s shares the rows of epochs 10, 20, 30, 40.
        env_tag = lambda t: t if t.startswith("Episode/") else "Episode/" + t
        cells: dict[int, dict[str, float]] = {}
        for e in range(1, 41):
            for t in tags:
                trainer = RL_GAMES_TAG.get(t)
                if trainer is None:
                    cells.setdefault(e, {})[env_tag(t)] = val(t, e - 1)
                elif trainer == "losses/cval_loss":
                    cells.setdefault(e * 128, {})[trainer] = val(t, e - 1)
                elif trainer == "rewards/iter":
                    cells.setdefault(e, {})[trainer] = val(t, e - 1)
                else:
                    cells.setdefault((e - 1) * 128, {})[trainer] = val(t, e - 1)
            cells.setdefault((e - 1) * 128, {})["info/epochs"] = e
            cells.setdefault(10 * e, {})["rewards/time"] = 1.0
        cols = sorted({k for row in cells.values() for k in row})
        with open(tmp / "s.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "wall_time"] + cols)
            for s in sorted(cells):
                w.writerow([s, float(s)] + [cells[s].get(k, "") for k in cols])
        (tmp / "agent.yaml").write_text("params:\n  seed: 1\n  config:\n    horizon_length: 16\n", encoding="utf-8")
    else:
        with open(tmp / "s.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "wall_time"] + tags)
            for i in range(40):
                w.writerow([i, 120.0 * i] + [val(t, i) for t in tags])
                w.writerow([i, 99999.0] + ["" for _ in tags])  # a */time-only row
        # D-195: the run's own factor, one iteration = 16 * 8 = 128 env steps.
        (tmp / "agent.yaml").write_text("seed: 1\nnum_steps_per_env: 16\n", encoding="utf-8")
    (tmp / "env.yaml").write_text("seed: 1\nscene:\n  num_envs: 8\n  env_spacing: 3.0\n", encoding="utf-8")
    # episodes.csv: iteration 0 holds episode 0 (10 steps) and episode 1 (50 steps).
    with open(tmp / "e.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "env_id", "episode", "outcome", "steps"])
        w.writerow([0, 0, 0, "success", 10])
        for it in range(40):
            w.writerow([it, 1, it + 1, "success", 50 + it])
            w.writerow([it, 2, it + 1, "timeout", 256])
    (tmp / "em.json").write_text(json.dumps({"episode_cap_steps": 256}))
    b = lambda k0, k1, rk: [{k0: 0.0, k1: 1.0, "episodes": 0, rk: None}, {k0: 1.0, k1: 2.0, "episodes": 5, rk: 0.8}]
    m = {"force_norm_n": {"f_max_n": 30.0}, "below_fixture_rate": {"underside_depth_mm": 46.0},
         "success_by_start_height_bin": b("start_mm_from", "start_mm_to", "success_rate"),
         "force_abort_by_start_height_bin": b("start_mm_from", "start_mm_to", "force_abort_rate"),
         "success_by_tilt_bin": b("tilt_deg_from", "tilt_deg_to", "success_rate"),
         "success_by_yaw_bin": b("yaw_deg_from", "yaw_deg_to", "success_rate"),
         "success_by_lateral_bin": b("lateral_mm_from", "lateral_mm_to", "success_rate")}
    (tmp / "m.json").write_text(json.dumps(m))


# Self-test (D-080): the x axis in env steps, on the source behind a namespace
AXIS_CHECKS = (
    "T2 the phase-dr marker sits at env_steps(20, 16, 8) = 2688",
    "T5 every x = (it + 1) * 16 * 8",
    "T6 wall-time line counts the 40 iteration rows only",
    "R1 rl_games Episode/success_rate at epoch x 128",
    "R2 rl_games losses/a_loss at epoch x 128 (its step is (epoch - 1) x 128)",
    "R3 rl_games losses/cval_loss at epoch x 128 (its step is epoch x 128)",
    "R4 rl_games Loss/value_function resolves to losses/cval_loss, dr/phase to Episode/dr/phase; no */time curve",
    "D1 dauer: x = (it + 1) x 128; rsl_rl drops episode 0 (median 50 steps), rl_games keeps it (median 30)",
)
MUTATIONS = (
    ("marker-left-in-iterations", "x = run_axis.env_steps(s, *factor)", "x = s", (AXIS_CHECKS[0], AXIS_CHECKS[1])),
    ("rl-games-one-axis-per-row", "x = run_axis.rl_games_x(k, s, F)", "x = s * F", (AXIS_CHECKS[4], AXIS_CHECKS[5], AXIS_CHECKS[6])),
    ("dauer-keeps-random-episode-0", 'lib == "rsl_rl" and int(r["episode"]) == 0', "False", (AXIS_CHECKS[7],)),
)


def _axis_checks(ns: dict, tmp: pathlib.Path) -> dict[str, bool]:
    import contextlib
    import io

    res = dict.fromkeys(AXIS_CHECKS, False)
    out = io.StringIO()
    try:
        factor = ns["run_axis"].read_factor(tmp / "agent.yaml", tmp / "env.yaml")
        with contextlib.redirect_stdout(out):
            c = ns["load_curves"](tmp / "s.csv", factor)
        # draw_all takes its event markers from first_step on these same x.
        res[AXIS_CHECKS[0]] = ns["first_step"](c, "dr/phase", 1.0) == 2688
        res[AXIS_CHECKS[1]] = c["Episode/success_rate"][0] == [(i + 1) * 128 for i in range(40)]
        # 39 intervals of 120 s = 1.30 h; the */time-only rows (wall 99999) would move both numbers.
        res[AXIS_CHECKS[2]] = "wall time 1.30 h over 40 iterations (steps 0..39," in out.getvalue()
    except BaseException:   # noqa: BLE001 - every check stays False
        pass
    want = [128 * e for e in range(1, 41)]
    try:
        rlg = tmp / "rlg"
        lib, factor = ns["run_axis"].read_run_factor(rlg / "agent.yaml", rlg / "env.yaml")
        with contextlib.redirect_stdout(out):
            c = ns["load_curves_rl_games"](rlg / "s.csv", factor)
        res[AXIS_CHECKS[3]] = lib == "rl_games" and c["Episode/success_rate"][0] == want
        res[AXIS_CHECKS[4]] = c["losses/a_loss"][0] == want
        res[AXIS_CHECKS[5]] = c["losses/cval_loss"][0] == want
        res[AXIS_CHECKS[6]] = (ns["need"](c, "Loss/value_function") is c["losses/cval_loss"]
                               and ns["need"](c, "dr/phase") is c["Episode/dr/phase"]
                               and not any(k.endswith("/time") for k in c))
    except BaseException:   # noqa: BLE001
        pass
    try:
        with open(tmp / "e.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rsl, rlg_pts = ns["dauer_points"](rows, (16, 8), "rsl_rl"), ns["dauer_points"](rows, (16, 8), "rl_games")
        res[AXIS_CHECKS[7]] = ([p[0] for p in rsl] == want and rsl[0][1] == 50 and rlg_pts[0][1] == 30)
    except BaseException:   # noqa: BLE001
        pass
    return res


def self_test() -> int:
    fails = 0
    src = pathlib.Path(__file__).read_text(encoding="utf-8")

    def _exec(text):
        ns = {"__name__": "plot_autodr_run_under_test", "__file__": __file__}
        exec(compile(text, __file__, "exec"), ns)  # noqa: S102 - the point of the self-test
        return ns

    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        _synthetic(tmp)
        (tmp / "rlg").mkdir()
        _synthetic(tmp / "rlg", lib="rl_games")
        factor = run_axis.read_factor(tmp / "agent.yaml", tmp / "env.yaml")
        c = load_curves(tmp / "s.csv", factor)
        ok = len(c["Episode/success_rate"][0]) == 40
        print(f"[{'PASS' if ok else 'FAIL'}] T1 time-only rows dropped (40 iteration rows)")
        fails += not ok
        for name, ok in _axis_checks(_exec(src), tmp).items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            fails += not ok
        out = tmp / "out"
        written = draw_all(tmp / "s.csv", factor, tmp / "m.json", tmp / "a.json", "RT-X", out,
                           "rsl_rl", tmp / "e.csv", tmp / "em.json", 1 / 15)
        ok = len(written) == 24 and all(p.stat().st_size > 0 for p in written)
        print(f"[{'PASS' if ok else 'FAIL'}] T3 eight figures x three formats (png, pdf, svg) written")
        fails += not ok
        rlg = tmp / "rlg"
        written = draw_all(rlg / "s.csv", (16, 8), rlg / "m.json", rlg / "a.json", "RT-R", rlg / "out",
                           "rl_games", rlg / "e.csv", rlg / "em.json", 1 / 15)
        ok = len(written) == 24 and all(p.stat().st_size > 0 for p in written)
        print(f"[{'PASS' if ok else 'FAIL'}] T7 rl_games run: eight figures x three formats (png, pdf, svg) written")
        fails += not ok
        # MUTATION: one required tag missing must stop the script, not draw a blank panel.
        mut = tmp / "mut"
        mut.mkdir()
        _synthetic(mut, drop_tag="dr/yaw_lo")
        try:
            draw_all(mut / "s.csv", factor, mut / "m.json", mut / "a.json", "RT-M", mut / "out")
            ok = False
        except SystemExit as e:
            ok = "dr/yaw_lo" in str(e)
        print(f"[{'PASS' if ok else 'FAIL'}] T4 MUTATION missing tag dr/yaw_lo stops with its name")
        fails += not ok
        cut = src.index("# Self-test (D-080): the x axis in env steps")
        head, tail = src[:cut], src[cut:]
        for mname, old, new, want in MUTATIONS:
            if head.count(old) != 1:
                ok, note = False, f" (anchor matches {head.count(old)} times)"
            else:
                flipped = {k for k, v in _axis_checks(_exec(head.replace(old, new) + tail), tmp).items() if not v}
                ok, note = flipped == set(want), f" (flipped {sorted(flipped)})"
            print(f"[{'PASS' if ok else 'FAIL'}] MUTATION {mname} flips exactly {sorted(want)}" + ("" if ok else note))
            fails += not ok
    total = 4 + len(AXIS_CHECKS) + len(MUTATIONS)
    print(f"self-test: {total - fails}/{total} PASS")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scalars", type=pathlib.Path)
    ap.add_argument("--metrics", type=pathlib.Path)
    ap.add_argument("--autodr", type=pathlib.Path)
    ap.add_argument("--run", default="")
    ap.add_argument("--agent-yaml", type=pathlib.Path,
                    help="the run's params/agent.yaml (num_steps_per_env, or rl_games' horizon_length)")
    ap.add_argument("--env-yaml", type=pathlib.Path, help="the run's params/env.yaml (scene.num_envs; policy dt via run_axis.read_policy_dt)")
    ap.add_argument("--episodes", type=pathlib.Path, help="the run's episodes.csv (figure dauer)")
    ap.add_argument("--episodes-meta", type=pathlib.Path, help="the run's episodes_meta.json (figure dauer)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("docs/figures"))
    ap.add_argument("--only", default="", help=f"comma list of {','.join(NAMES)}; default: all the inputs allow")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.scalars and a.metrics and a.autodr and a.run and a.agent_yaml and a.env_yaml):
        ap.error("--scalars, --metrics, --autodr, --run, --agent-yaml and --env-yaml are required")
    lib, factor = run_axis.read_run_factor(a.agent_yaml, a.env_yaml)
    dt = run_axis.read_policy_dt(a.env_yaml) if a.episodes else None
    only = [n for n in a.only.split(",") if n]
    if set(only) - set(NAMES):
        raise SystemExit(f"--only: unknown {sorted(set(only) - set(NAMES))}; known {list(NAMES)}")
    draw_all(a.scalars, factor, a.metrics, a.autodr, a.run, a.out_dir, lib, a.episodes, a.episodes_meta, dt,
             only or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
