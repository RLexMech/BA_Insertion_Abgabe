# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Six report figures for ONE AutoDR run (RT-206 series), PNG + PDF.

WHAT IT DRAWS (``<out-dir>/<run>_<name>.png`` and ``.pdf``)
------------------------------------------------------------
    erfolg     success rate over iterations, phase-dr and all-at-max marked
    stufen     AutoDR boundary progress per dimension, 0 = centre, 1 = maximum
               (small multiples; start floor as its own panel)
    kraft      per-episode peak force: mean and p95 with the abort limit;
               below it the force-abort rate (two panels, never two y-axes)
    ppo        episode reward, surrogate loss, value loss, entropy
               (the CLAUDE.md order; EV and KL are not logged, D-116)
    physik     max depth (with the pocket-gate floor), below-fixture rate,
               engaged reward, mean interpenetration
    endfenster final-window bin tables of demo_metrics.json: success per
               start height / tilt / yaw / lateral bin, force abort per height

WHERE THE NUMBERS COME FROM
---------------------------
* curves: the run's ``scalars.csv`` (``scripts/export_tb_scalars.py``). Only
  rows that carry ``Episode/success_rate`` are iteration rows; the others
  hold the two ``*/time`` tags only.
* normalisation of the boundaries: ``dims`` and ``floor`` of the run's
  ``autodr_<it>.json`` sidecar -- never typed here.
* abort limit and gate floor: ``demo_metrics.json`` (``force_norm_n.f_max_n``,
  ``below_fixture_rate.underside_depth_mm``).
A missing tag or key stops the script with its name (fail fast).

    python scripts/plot_autodr_run.py --scalars docs/figures/RT-206s1_scalars.csv \
        --metrics docs/figures/RT-206s1_demo_metrics.json \
        --autodr docs/figures/RT-206s1_autodr_1499.json --run RT-206s1 --out-dir docs/figures
    python scripts/plot_autodr_run.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import tempfile

RUN = "#3B6FD4"   # series 1 (plot_run_curves.py precedent, validated with REF)
REF = "#D97706"   # series 2
INK = "#1F2933"
INK_MUTED = "#6B7280"
GRID = "#E5E7EB"

NAMES = ("erfolg", "stufen", "kraft", "ppo", "physik", "endfenster")
DIM_LABEL = {
    "lat_r": "Seitlicher Versatz (Radius)",
    "yaw": "Gierwinkel",
    "tilt": "Kippung",
    "start_height": "Starthöhe (Decke)",
    "friction": "Reibung",
}


def load_curves(path: pathlib.Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "step" not in rows[0]:
        raise SystemExit(f"{path}: no 'step' column")
    out: dict[str, tuple[list, list]] = {}
    for r in rows:
        if r.get("Episode/success_rate", "") == "":
            continue
        s = int(float(r["step"]))
        for k, v in r.items():
            if k in ("step", "wall_time") or v in ("", None):
                continue
            xs, ys = out.setdefault(k, ([], []))
            xs.append(s)
            ys.append(float(v))
    return out


def need(curves: dict, tag: str):
    if tag not in curves:
        raise SystemExit(f"tag {tag!r} missing from scalars.csv")
    return curves[tag]


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
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if xlabel:
        ax.set_xlabel("PPO-Iteration", color=INK, fontsize=9)


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
    ad = sidecar["autodr"]
    dims = {d["name"]: d for d in ad["dims"]}
    fl = ad["floor"]
    panels = [("start_height_floor", "Startboden", None)] + [(n, DIM_LABEL.get(n, n), dims[n]) for n in dims]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for ax, (name, label, d) in zip(axes.flat, panels):
        if d is None:
            xs, ys = need(c, "dr/start_height_floor")
            ax.plot(xs, [(y - fl["f0"]) / (fl["top"] - fl["f0"]) for y in ys], color=RUN, linewidth=1.2)
        else:
            hx, hy = need(c, f"dr/{name}_hi")
            ax.plot(hx, [(y - d["centre"]) / (d["hi_max"] - d["centre"]) for y in hy], color=RUN,
                    linewidth=1.2, label="obere Grenze")
            if not d["one_sided"]:
                lx, ly = need(c, f"dr/{name}_lo")
                ax.plot(lx, [(y - d["centre"]) / (d["lo_max"] - d["centre"]) for y in ly], color=REF,
                        linewidth=1.2, label="untere Grenze")
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_ylim(-0.05, 1.08)
        style(ax, "", xlabel=False)
        for x, _ in ev:
            if x is not None:
                ax.axvline(x, color=INK_MUTED, linewidth=0.8, linestyle=":")
    for ax in axes[1]:
        ax.set_xlabel("PPO-Iteration", color=INK, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Anteil am Maximum", color=INK, fontsize=9)
    handles, labels = axes.flat[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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
        ax.set_xlabel("PPO-Iteration", color=INK, fontsize=9)
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


def draw_all(scalars, metrics_path, autodr_path, run, out_dir) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = load_curves(scalars)
    metrics = json.loads(pathlib.Path(metrics_path).read_text(encoding="utf-8"))
    sidecar = json.loads(pathlib.Path(autodr_path).read_text(encoding="utf-8"))
    ev = [(first_step(c, "dr/phase", 1.0), "Boden fertig"), (first_step(c, "dr/all_at_max", 1.0), "alle Grenzen am Maximum")]
    gate = float(metrics["below_fixture_rate"]["underside_depth_mm"])
    figs = {
        "erfolg": fig_erfolg(plt, c, ev),
        "stufen": fig_stufen(plt, c, sidecar, ev),
        "kraft": fig_kraft(plt, c, metrics, ev),
        "ppo": fig_grid(plt, c, [
            ("Train/mean_reward", "Episoden-Reward [–]", 1.0, None),
            ("Loss/surrogate", "Policy-Loss [–]", 1.0, None),
            ("Loss/value_function", "Value-Loss [–]", 1.0, None),
            ("Loss/entropy", "Entropie [–]", 1.0, None),
        ], ev),
        "physik": fig_grid(plt, c, [
            ("Episode/max_depth_max_mm", "Größte Tiefe [mm]", 1.0, (gate, f"Unterseite Aufnahme {gate:g} mm")),
            ("Episode/below_fixture_rate", "Unter der Aufnahme [%]", 100.0, None),
            ("Episode_Reward/engaged", "Reward-Term engaged [–]", 1.0, None),
            ("Episode/interpen_max_mean_mm", "Durchdringung, Mittel [mm]", 1.0, None),
        ], ev),
        "endfenster": fig_endfenster(plt, metrics),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in NAMES:
        fig = figs[name]
        for ext in ("png", "pdf"):
            p = out_dir / f"{run}_{name}.{ext}"
            fig.savefig(p, dpi=160, bbox_inches="tight")
            written.append(p)
        plt.close(fig)
    for p in written:
        print(f"[plot_autodr_run] wrote {p}")
    return written


def _synthetic(tmp: pathlib.Path, drop_tag: str | None = None):
    dims = [
        {"name": "lat_r", "centre": 0.0, "lo_max": 0.0, "hi_max": 0.03, "one_sided": True},
        {"name": "yaw", "centre": 0.0, "lo_max": -0.1, "hi_max": 0.1, "one_sided": False},
        {"name": "tilt", "centre": 0.0, "lo_max": 0.0, "hi_max": 0.2, "one_sided": True},
        {"name": "start_height", "centre": 0.02, "lo_max": 0.02, "hi_max": 0.12, "one_sided": True},
        {"name": "friction", "centre": 0.14, "lo_max": 0.08, "hi_max": 0.2, "one_sided": False},
    ]
    (tmp / "a.json").write_text(json.dumps({"autodr": {"dims": dims, "floor": {"f0": -0.03, "top": 0.02}}}))
    tags = ["Episode/success_rate", "dr/phase", "dr/all_at_max", "dr/start_height_floor",
            "Episode/force_max_mean_n", "Episode/force_max_p95_n", "Episode/force_abort_rate",
            "Train/mean_reward", "Loss/surrogate", "Loss/value_function", "Loss/entropy",
            "Episode/max_depth_max_mm", "Episode/below_fixture_rate", "Episode_Reward/engaged",
            "Episode/interpen_max_mean_mm"]
    for d in dims:
        tags.append(f"dr/{d['name']}_hi")
        if not d["one_sided"]:
            tags.append(f"dr/{d['name']}_lo")
    tags = [t for t in tags if t != drop_tag]
    with open(tmp / "s.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "wall_time"] + tags)
        for i in range(40):
            w.writerow([i, 0.0] + [min(1.0, i / 20) if "/" in t else 0 for t in tags])
            w.writerow([i, 0.0] + ["" for _ in tags])  # a */time-only row
    b = lambda k0, k1, rk: [{k0: 0.0, k1: 1.0, "episodes": 0, rk: None}, {k0: 1.0, k1: 2.0, "episodes": 5, rk: 0.8}]
    m = {"force_norm_n": {"f_max_n": 30.0}, "below_fixture_rate": {"underside_depth_mm": 46.0},
         "success_by_start_height_bin": b("start_mm_from", "start_mm_to", "success_rate"),
         "force_abort_by_start_height_bin": b("start_mm_from", "start_mm_to", "force_abort_rate"),
         "success_by_tilt_bin": b("tilt_deg_from", "tilt_deg_to", "success_rate"),
         "success_by_yaw_bin": b("yaw_deg_from", "yaw_deg_to", "success_rate"),
         "success_by_lateral_bin": b("lateral_mm_from", "lateral_mm_to", "success_rate")}
    (tmp / "m.json").write_text(json.dumps(m))


def self_test() -> int:
    fails = 0
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        _synthetic(tmp)
        c = load_curves(tmp / "s.csv")
        ok = len(c["Episode/success_rate"][0]) == 40
        print(f"[{'PASS' if ok else 'FAIL'}] T1 time-only rows dropped (40 iteration rows)")
        fails += not ok
        ok = first_step(c, "dr/phase", 1.0) == 20
        print(f"[{'PASS' if ok else 'FAIL'}] T2 first step at >= 1.0 is 20")
        fails += not ok
        out = tmp / "out"
        written = draw_all(tmp / "s.csv", tmp / "m.json", tmp / "a.json", "RT-X", out)
        ok = len(written) == 12 and all(p.stat().st_size > 0 for p in written)
        print(f"[{'PASS' if ok else 'FAIL'}] T3 six figures x two formats written")
        fails += not ok
        # MUTATION: one required tag missing must stop the script, not draw a blank panel.
        mut = tmp / "mut"
        mut.mkdir()
        _synthetic(mut, drop_tag="dr/yaw_lo")
        try:
            draw_all(mut / "s.csv", mut / "m.json", mut / "a.json", "RT-M", mut / "out")
            ok = False
        except SystemExit as e:
            ok = "dr/yaw_lo" in str(e)
        print(f"[{'PASS' if ok else 'FAIL'}] T4 MUTATION missing tag dr/yaw_lo stops with its name")
        fails += not ok
    print(f"self-test: {4 - fails}/4 PASS")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scalars", type=pathlib.Path)
    ap.add_argument("--metrics", type=pathlib.Path)
    ap.add_argument("--autodr", type=pathlib.Path)
    ap.add_argument("--run", default="")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("docs/figures"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.scalars and a.metrics and a.autodr and a.run):
        ap.error("--scalars, --metrics, --autodr and --run are required")
    draw_all(a.scalars, a.metrics, a.autodr, a.run, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
