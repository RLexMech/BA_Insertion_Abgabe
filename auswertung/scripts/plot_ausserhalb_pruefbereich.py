"""H17: success of Auto-DR and Fixed-DR inside and outside the Prüfbereich, with
the number of test cases at every point (thesis 5.2.2).

One row per disturbance (Startversatz, |Gierung|, Kippung, Reibwert), one column
per arm. Inside the Prüfbereich: the main test's bins (F3's data, ``data_f3``).
Outside: the T-ood block at 1.5x the range limit (|yaw| pools yaw_lo and yaw_hi)
and, for tilt and friction, the steps of the sweep (RT-264/265, ``data_h14``)
that lie outside the range. Start height has no point outside (the controller's
z range, Kapitel 4.3.2) and no row. Every mean over the five seeds carries its
95 % percentile bootstrap interval (D-218) and shows the seeds (D-219).

Inputs (final study, Experiment D, RT-242/243/244): ``results/eval_rlgames/``
(summary.json, T-ood manifests) and ``results/eval_rlgames_sweep/``. Reuses
``eval_figures`` (data_f3, data_h14, _boot_ci, the OOD/sweep input checks and
the style helpers) and changes nothing there. Thin caller, ``.venv-analysis``:

    .venv-analysis/Scripts/python.exe scripts/plot_ausserhalb_pruefbereich.py

Writes ``docs/figures/eval_rlgames/H17_ausserhalb_pruefbereich.{pdf,svg,png}``.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import eval_figures as EF  # noqa: E402
from eval_figures import FigError  # noqa: E402

NAME = "H17_ausserhalb_pruefbereich"
ARMS = ("autodr", "fixeddr")   # the arms that played T-ood (D-208)
# Row: F3 bin column, T-ood key groups (one point per group; two keys are pooled),
# sweep block or None, factor sidecar unit -> axis unit, axis label (F3's wording).
PANELS = (
    ("lat_r_mm", (("lat_r_hi",),), None, 1000.0, "Startversatz in mm"),
    ("yaw_abs_deg", (("yaw_lo", "yaw_hi"),), None, 180.0 / math.pi, "Betrag der Gierung in °"),
    ("tilt_deg", (("tilt_hi",),), "T-sweep-tilt", 180.0 / math.pi, "Kippung in °"),
    ("friction", (("friction_lo",), ("friction_hi",)), "T-sweep-fric", 1.0, "Reibwert"),
)
MARK = {"bins": ("o", "-"), "sweep": ("^", "--"), "ood": ("D", "")}
LABEL = {"bins": "Haupttest, je Bereich der Störgröße",
         "sweep": "Stufentest (eine Störgröße fest, RT-264/265)",
         "ood": "Test außerhalb des Prüfbereichs (1,5-facher Grenzwert)"}


def _stat(stack, fig):
    stack = [np.asarray(v, dtype=float) for v in stack]
    lo, hi = EF._boot_ci(stack, fig)
    return {"mean": np.mean(stack, axis=0), "ci": (lo, hi)}


def data_h17(summary: dict, ood_side: dict, h14: dict) -> dict:
    """Per row: x and n of the bins, the T-ood points and the outside sweep
    steps, and per arm each seed's rate, the mean and its interval. Refused:
    T-ood keys of one group at different |values|, test-case counts that differ
    between policies, a sweep arm with other seeds than the main test."""
    f3 = EF.data_f3(summary)
    ood = summary["OOD"]["policies"]
    side = {b["key"]: b for b in ood_side["blocks"]}
    out = {}
    for col, groups, sweep_block, fac, xlabel in PANELS:
        p3 = f3[col]
        rng = (p3["lower"][0], p3["upper"][-1])
        panel = {"xlabel": xlabel, "range": rng, "arms": {}, "bins": {"x": p3["x"], "n": p3["n"]}}
        pts = []
        for keys in groups:
            xs = {round(abs(side[k]["physical"]) * fac, 6) for k in keys}
            if len(xs) != 1:
                raise FigError(f"H17: T-ood {keys} at different |values| {sorted(xs)}")
            pts.append((xs.pop(), keys))
        panel["ood"] = {"x": [x for x, _k in pts]}
        keep = []
        if sweep_block:
            s = h14[sweep_block]
            keep = [i for i, x in enumerate(s["x"]) if not rng[0] - 1e-9 <= x <= rng[1] + 1e-9]
            panel["sweep"] = {"x": [s["x"][i] for i in keep], "n": [s["n"]] * len(keep)}
        n_ood = None
        for m in ARMS:
            seeds_b = p3["methods"][m]["seeds"]
            ids = sorted(int(k.rsplit("_s", 1)[1]) for k in seeds_b)
            a = {"bins": {"seeds": {s: seeds_b[f"{m}_s{s}"] for s in ids}}}
            a["bins"].update(_stat(list(a["bins"]["seeds"].values()), "H17"))
            rows, ns = {}, []
            for s in ids:
                blocks = ood[f"{m}_s{s}"]["blocks"]
                ks = [sum(blocks[k]["k"] for k in keys) for _x, keys in pts]
                nn = [sum(blocks[k]["n"] for k in keys) for _x, keys in pts]
                rows[s] = [k / n for k, n in zip(ks, nn)]
                ns.append(nn)
            if any(nn != ns[0] for nn in ns) or (n_ood is not None and ns[0] != n_ood):
                raise FigError(f"H17: {col} T-ood test-case counts differ between policies: {ns}")
            n_ood = ns[0]
            a["ood"] = {"seeds": rows, **_stat(list(rows.values()), "H17")}
            if sweep_block:
                sa = h14[sweep_block]["arms"][m]
                if sorted(sa["seeds"]) != ids:
                    raise FigError(f"H17: {sweep_block} {m} seeds {sorted(sa['seeds'])} != {ids}")
                sd = {s: np.asarray(sa["seeds"][s])[keep] for s in ids}
                a["sweep"] = {"seeds": sd, **_stat(list(sd.values()), "H17")}
            panel["arms"][m] = a
        panel["ood"]["n"] = n_ood
        out[col] = panel
    return out


def _split(x, y, cut):
    """NaN before each cut point: no line across the Prüfbereich between the
    low and the high side (friction)."""
    i = np.flatnonzero(cut)
    return np.insert(np.asarray(x, dtype=float), i, np.nan), np.insert(np.asarray(y, dtype=float), i, np.nan)


def draw_h17(plt, d: dict):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory as blend
    panels = list(d.values())
    fig, axes = plt.subplots(len(panels), len(ARMS), figsize=(EF.FIG_W, 8.4), sharey=True)
    for r, p in enumerate(panels):
        xs_all = p["bins"]["x"] + p["ood"]["x"] + p.get("sweep", {}).get("x", [])
        xmin = min(xs_all) if p["range"][0] > 0 else 0.0
        xmax = max(xs_all + [p["range"][1]])
        pad = 0.06 * (xmax - xmin)
        for c, m in enumerate(ARMS):
            ax, colour, a = axes[r, c], EF.METHOD_COLOR[m], p["arms"][m]
            ax.axvspan(*p["range"], color=EF.GRID, alpha=0.7, linewidth=0, zorder=0)
            for part in ("bins", "sweep", "ood"):
                if part not in a:
                    continue
                x = np.asarray(p[part]["x"], dtype=float)
                mk, ls = MARK[part]
                cut = np.r_[False, (x[:-1] < p["range"][0]) & (x[1:] > p["range"][1])]
                for v in a[part]["seeds"].values():
                    if ls:
                        ax.plot(*_split(x, 100 * np.asarray(v), cut), color=colour, linestyle=ls, zorder=1,
                                **EF.SEED_LINE)
                    else:
                        ax.scatter(x, 100 * np.asarray(v), s=6, color=colour, alpha=0.45, linewidths=0, zorder=1)
                lo, hi = (100 * np.asarray(v) for v in a[part]["ci"])
                mean = 100 * np.asarray(a[part]["mean"])
                if ls:
                    for seg in np.split(np.arange(len(x)), np.flatnonzero(cut)):
                        ax.fill_between(x[seg], lo[seg], hi[seg], color=colour, alpha=0.2, linewidth=0)
                    ax.plot(*_split(x, mean, cut), color=colour, linestyle=ls, linewidth=1.44, marker=mk,
                            markersize=3.6, markeredgecolor="white", markeredgewidth=0.5, zorder=3, clip_on=False)
                else:
                    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt=mk, color=colour, markersize=4,
                                elinewidth=1.0, capsize=0, markeredgecolor="white", markeredgewidth=0.5,
                                zorder=4, clip_on=False)
            # Test cases per policy above every point: bins and steps grey, T-ood black one row higher.
            tr = blend(ax.transData, ax.transAxes)
            for x, n in zip(p["bins"]["x"] + p.get("sweep", {}).get("x", []),
                            p["bins"]["n"] + p.get("sweep", {}).get("n", [])):
                ax.text(x, 1.04, str(n), transform=tr, ha="center", va="bottom", fontsize=7, color=EF.INK_MUTED)
            for x, n in zip(p["ood"]["x"], p["ood"]["n"]):
                ax.text(x, 1.17, str(n), transform=tr, ha="center", va="bottom", fontsize=7, color=EF.INK,
                        fontweight="bold")
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 25))
            EF._style(ax, "Erfolgsrate in %" if c == 0 else "", p["xlabel"])
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(EF.METHOD_LABEL[m], fontsize=9, color=EF.INK, pad=26, fontweight="bold")
    n_seeds = {len(p["arms"][m]["bins"]["seeds"]) for p in panels for m in ARMS}
    if len(n_seeds) != 1:
        raise FigError(f"H17: seed counts {sorted(n_seeds)}; one legend cannot name them")
    h = [Line2D([], [], color=EF.INK_MUTED, marker=MARK[k][0], linestyle=MARK[k][1] or "none", linewidth=1.4,
                markersize=4, label=LABEL[k]) for k in ("bins", "sweep", "ood")]
    h += [EF._seed_line_handle(), EF._band_patch(n_seeds.pop()),
          Patch(facecolor=EF.GRID, alpha=0.7, label="grauer Bereich = Prüfbereich"),
          Line2D([], [], color="none", label="Zahlen über den Feldern = Testfälle je Policy\n"
                                             "(grau: Bereich bzw. Stufe, schwarz: 1,5-facher Grenzwert)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=2.4)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=pathlib.Path, default=REPO / "results/eval_rlgames")
    ap.add_argument("--sweep", type=pathlib.Path, default=REPO / "results/eval_rlgames_sweep")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "docs/figures/eval_rlgames")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.findfont(font_manager.FontProperties(family=EF.FONT), fallback_to_default=False)
    except ValueError as exc:
        raise FigError(f"font {EF.FONT!r} not installed; matplotlib would fall back silently") from exc
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "eval_figures",
                         "font.family": "sans-serif", "font.sans-serif": [EF.FONT]})
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    side = EF._ood_inputs(args.results, EF._ood_sidecar_path())
    sdf = pd.read_csv(args.sweep / "episodes_all.csv", keep_default_na=False, dtype={"tag": str, "boundary_key": str})
    fig = draw_h17(plt, data_h17(summary, side, EF.data_h14(sdf, EF._sweep_inputs(args.sweep))))
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        p = args.out / f"{NAME}.{ext}"
        fig.savefig(p, dpi=EF.PNG_DPI, bbox_inches="tight", metadata=EF.SAVE_META[ext])
        print(f"[H17] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
