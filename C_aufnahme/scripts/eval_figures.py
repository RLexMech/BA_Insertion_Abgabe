# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""The evaluation figures F1-F10 (docs/eval_figures_plan.md; eval_plan.md section 8b
step 3). Laptop only, in ``.venv-analysis`` (D-189).

Every figure has a DATA function (pure; returns the arrays it draws; the
self-test checks those) and a DRAW function (matplotlib). Numbers that
``summary.json`` holds are read from there, never recomputed; only what it
does not hold (F6 curves, F9 cells, F10 curve) comes from ``episodes_all.csv``.
Wilson intervals and the bin rule are imported from ``eval_stats.py`` (one home).

  F1  success rate per seed (dot plot, Weissgerber 2015), T-nominal | T-final
  F2  AutoDR - No-DR: Welch thick, bootstrap thin, per block, zero line
  F3  success over each disturbance (4 panels; friction has no bin edges),
      seeds thin, mean thick, No-DR's training value marked
  F4  final policy: 7 boundary blocks, overall T-final rate as line + Wilson band
  F5  outcome per policy on T-final, stacked, status colours
  F6  (a) peak force ECDF over successes with the abort limit
      (b) share of ALL episodes successful by step k
  F7  training success over PPO iteration (seeds thin, mean thick)
  F8  AutoDR boundaries over training, 0 = centre, 1 = maximum
  F9  heat map lateral offset x tilt, T-final, seeds pooled, rate + row count
  F10 success counted only below an interpenetration threshold (IndustReal Fig. 3)
F7-F10 were added AFTER the results were known: descriptive, no criterion.
Fixed-DR (D-192, added afterwards; user 2026-09-23 "B"): F1, F2 (two K5
rows), F3, F5, F6, F7, F9, F10 draw it as a third arm when T-nominal-F AND
T-final-F exist (F7: when RT-231s1..s5 were extracted); otherwise every
figure is the two-method one. F4 (final policy) and F8 (AutoDR growth) have
no Fixed-DR content.

Training curves: ``--extract-training rt_logs`` reduces the ten git-ignored
``*_scalars.csv`` to ``results/training/curves.csv`` (plus the AutoDR sidecars
and No-DR's training values), which the figures then read.

    .venv-analysis/Scripts/python.exe scripts/eval_figures.py --extract-training rt_logs
    .venv-analysis/Scripts/python.exe scripts/eval_figures.py --results results/eval
    .venv-analysis/Scripts/python.exe scripts/eval_figures.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pathlib
import re
import shutil
import sys

import numpy as np
import pandas as pd

SCRIPT_MARKER = "eval_figures-2026-09-24a"
REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ES = _load_module("eval_stats", REPO / "scripts" / "eval_stats.py")
KEY = ES.KEY

# House pair (plot_run_curves.py:52-53, plot_autodr_run.py:47-48). Fixed-DR
# (D-192) takes the dataviz reference palette's third slot, aqua, at its darker
# step #199e70 (references/palette.md, dark column): the light step #1baf7a
# gave 2.82:1 on white. Validator 2026-09-23, light, surface #FFFFFF, all
# pairs: ALL CHECKS PASS; worst CVD #199e70<->#D97706 protan 9.0, tritan 7.4
# (6-8 band: legal with the legend / axis labels every figure carries).
# Colour follows the METHOD.
METHOD_COLOR = {"autodr": "#3B6FD4", "nodr": "#D97706", "fixeddr": "#199e70"}
METHOD_LABEL = {"autodr": "AutoDR", "nodr": "No-DR", "fixeddr": "Fixed-DR"}
# The pre-registered pair; every figure needs both. Fixed-DR is optional: it
# is drawn where its blocks exist (ARM_BLOCK), else the figure keeps two methods.
METHODS = ("autodr", "nodr")
ARMS = (*METHODS, "fixeddr")
# Fixed-DR plays the same tables in its own blocks (eval_plan.md section 3.2).
ARM_BLOCK = {"T-nominal": "T-nominal-F", "T-final": "T-final-F"}
INK = "#1F2933"
INK_MUTED = "#6B7280"
GRID = "#E5E7EB"
# dataviz status palette (references/palette.md), in the stack order that
# passes the validator's adjacent CVD and normal-vision checks (2026-09-23).
OUTCOMES = (
    ("success", "Erfolg", "#0ca30c"),
    ("success_with_force_violation", "Erfolg mit Kraftverletzung", "#fab219"),
    ("force_abort", "Kraftabbruch", "#d03b3b"),
    ("timeout", "Zeitlimit", "#ec835a"),
)
HEAT_CMAP = "Purples"   # one hue, light -> dark; neither method hue

# The ten frozen policies (eval_plan.md "Kontext"; scripts/run_eval.ps1).
TRAINING_RUNS = {(m, s): f"RT-{206 if m == 'autodr' else 207}s{s}" for m in METHODS for s in range(1, 6)}
# The five Fixed-DR training runs (D-192, rt_logs/RT-231_expectation.md): F7
# only, extracted when all five folders exist, refused when some do.
FIXED_RUNS = {("fixeddr", s): f"RT-231s{s}" for s in range(1, 6)}
TRAINING_TAGS = ("Episode/success_rate", "dr/lat_r_hi", "dr/yaw_lo", "dr/yaw_hi", "dr/tilt_hi",
                 "dr/start_height_hi", "dr/start_height_floor", "dr/friction_lo", "dr/friction_hi")
# No-DR's training values of the F3 axes: env.yaml keys (RT-207s<N>_env.yaml).
NODR_ENV_KEYS = ("dr_mode", "start_lateral_offset", "fixture_yaw_rad", "fixture_yaw_noise_rad",
                 "fixture_tilt_rad", "fixture_tilt_noise_rad")

# F3 panels: episodes_all column, axis label, factor to the axis unit.
F3_PANELS = (
    ("lat_r_mm", "Seitlicher Versatz in mm", 1.0),
    ("yaw_abs_deg", "Gierwinkel (Betrag) in °", 1.0),
    ("tilt_deg", "Kippung in °", 1.0),
    ("start_height_m", "Starthöhe in mm", 1000.0),
)
# F4: the seven T-grenze keys (eval_table_boundary_seed10.json order).
K3_ORDER = ("lat_r_hi", "yaw_lo", "yaw_hi", "tilt_hi", "start_height_hi", "friction_lo", "friction_hi")
K3_LABEL = {"lat_r_hi": "Versatz\nmax", "yaw_lo": "Gier\nmin", "yaw_hi": "Gier\nmax",
            "tilt_hi": "Kippung\nmax", "start_height_hi": "Starthöhe\nmax",
            "friction_lo": "Reibung\nmin", "friction_hi": "Reibung\nmax"}
F8_PANELS = (("start_height_floor", "Startboden"), ("lat_r", "Seitlicher Versatz"), ("yaw", "Gierwinkel"),
             ("tilt", "Kippung"), ("start_height", "Starthöhe (Decke)"), ("friction", "Reibung"))
FIG_W = 6.3   # inches, ~16 cm text width
# The thesis body is Helvetica (thesis/includes/preamble.tex: helvet + \sfdefault);
# Arial has Helvetica's metrics and is on the laptop (user 2026-09-23).
FONT = "Arial"


class FigError(RuntimeError):
    """The input cannot draw what the plan asks."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _policies(summary: dict, block: str) -> list:
    return sorted((p for p in summary["policies"] if p["block"] == block and p["tag"] == ""),
                  key=lambda p: (p["method"], p["seed"]))


def bin_index(values, edges) -> np.ndarray:
    """The `_rate_by_bin` rule of eval_stats.bin_rates, as an index array."""
    idx = np.searchsorted(np.asarray(edges, dtype=float), np.asarray(values, dtype=float), side="right")
    return np.minimum(idx, len(edges) - 1)


def bin_lower(edges, i: int) -> float:
    """Lower edge of bin i; the quantities of F3/F9 start at 0 (one-sided, |yaw|)."""
    return float(edges[i - 1]) if i > 0 else 0.0


def _de_digits(s: str) -> str:
    """Decimal comma, as the thesis text sets it (thesis/includes/preamble.tex, siunitx)."""
    return s.replace(".", ",")


def _num(v: float, fmt: str) -> str:
    """A number for figure text: decimal comma and a real minus sign (U+2212)."""
    return _de_digits(format(v, fmt)).replace("-", "−")


def _de_formatter():
    """matplotlib's default tick formatter with the decimal comma."""
    from matplotlib.ticker import ScalarFormatter

    class _DeScalarFormatter(ScalarFormatter):
        def __call__(self, x, pos=None):
            return _de_digits(super().__call__(x, pos))

    return _DeScalarFormatter()


# ---------------------------------------------------------------------------
# DATA functions -- pure, one per figure
# ---------------------------------------------------------------------------

def arms_present(summary: dict) -> tuple:
    return _arms_of({p["block"] for p in summary["policies"] if p["method"] == "fixeddr"})


def arms_in(df: pd.DataFrame) -> tuple:
    """arms_present for the episode table (F6, F9, F10)."""
    return _arms_of(set(df.loc[df["method"] == "fixeddr", "block"]))


def _arms_of(blocks: set) -> tuple:
    """METHODS, plus Fixed-DR when BOTH of its blocks hold runs (D-192).
    ``blocks``: the blocks Fixed-DR runs sit in. One block without the other
    is refused, never drawn half."""
    if not blocks:
        return METHODS
    if blocks == set(ARM_BLOCK.values()):
        return ARMS
    raise FigError(f"Fixed-DR runs in {sorted(blocks)}, want both {sorted(ARM_BLOCK.values())} or none")


def _arm_policies(summary: dict, block: str) -> list:
    """The pair's policies of ``block`` and Fixed-DR's of its own block."""
    return (_policies(summary, block)
            + [p for p in _policies(summary, ARM_BLOCK[block]) if p["method"] == "fixeddr"])


def data_f1(summary: dict) -> dict:
    out = {}
    arms = arms_present(summary)
    for block in ("T-nominal", "T-final"):
        pol = _arm_policies(summary, block)
        out[block] = {}
        for m in arms:
            ps = [p for p in pol if p["method"] == m]
            if not ps:
                raise FigError(f"F1: no {m} runs in {block}")
            rates = [p[KEY] for p in ps]
            out[block][m] = {"seeds": [p["seed"] for p in ps], "rates": rates,
                             "wilson": [p[f"{KEY}_wilson"] for p in ps],
                             "mean": float(np.mean(rates))}
    return out


def data_f2(summary: dict) -> list:
    rows = []
    for kname, block in (("K2", "T-nominal"), ("K1", "T-final")):
        k = summary[kname]
        if k.get("block") != block:
            raise FigError(f"F2: {kname} is not about {block}")
        rows.append({"block": block, "key": kname, "diff": k["diff"], "label": "AutoDR − No-DR",
                     "welch": list(k["welch_ci"]), "boot": list(k["bootstrap_percentile_ci"])})
    # K5 (D-192), only when the Fixed-DR blocks exist; eval_stats writes
    # {"missing": ...} otherwise, and an "error" is not drawn over.
    if arms_present(summary) == ARMS:
        for kname, block in (("K5_nominal", "T-nominal"), ("K5", "T-final")):
            k = summary[kname]
            if "diff" not in k or k.get("blocks", [None])[0] != block:
                raise FigError(f"F2: {kname} not computed for {block}: {k}")
            rows.append({"block": block, "key": kname, "diff": k["diff"], "label": "AutoDR − Fixed-DR",
                         "welch": list(k["welch_ci"]), "boot": list(k["bootstrap_percentile_ci"])})
    return rows


def data_f3(summary: dict) -> dict:
    bins = summary["bins_T_final"]
    out = {}
    for col, _label, fac in F3_PANELS:
        edges = ES.BIN_EDGES[col]
        keys_of = {m: sorted(k for k in bins if k.startswith(m + "_s")) for m in arms_present(summary)}
        ns = None
        for m, keys in keys_of.items():
            if not keys:
                raise FigError(f"F3: no {m} bins")
            for k in keys:
                n_k = [b["n"] for b in bins[k][col]]
                if ns is None:
                    ns = n_k
                elif n_k != ns:
                    raise FigError(f"F3: {k} {col} holds other bin counts than the first policy")
        # Same table rows for every policy -> the same bins are empty for all.
        keep = [i for i, n in enumerate(ns) if n > 0]
        per = {}
        for m, keys in keys_of.items():
            seeds = {k: [bins[k][col][i]["rate"] for i in keep] for k in keys}
            per[m] = {"seeds": seeds, "mean": [float(np.mean([seeds[k][j] for k in keys]))
                                               for j in range(len(keep))]}
        out[col] = {"x": [fac * 0.5 * (bin_lower(edges, i) + edges[i]) for i in keep],
                    "lower": [fac * bin_lower(edges, i) for i in keep],
                    "upper": [fac * float(edges[i]) for i in keep],
                    "n": [ns[i] for i in keep], "methods": per}
    return out


def data_f4(summary: dict) -> dict:
    k3 = summary["K3"]
    if "blocks" not in k3:
        raise FigError(f"F4: K3 not computed: {k3}")
    if set(k3["blocks"]) != set(K3_ORDER):
        raise FigError(f"F4: K3 keys {sorted(k3['blocks'])} != {sorted(K3_ORDER)}")
    ov = k3["overall"]
    return {"keys": list(K3_ORDER), "rates": [k3["blocks"][k]["rate"] for k in K3_ORDER],
            "wilson": [k3["blocks"][k]["wilson"] for k in K3_ORDER],
            "n": [k3["blocks"][k]["n"] for k in K3_ORDER],
            "overall": ov["rate"], "overall_wilson": ES.wilson(ov["k"], ov["n"]),
            "policy": summary["final_policy"]}


def data_f5(summary: dict) -> list:
    rows = []
    arms_present(summary)   # one Fixed-DR block without the other is refused
    for p in _arm_policies(summary, "T-final"):
        counts = [p["outcomes"][k] for k, _l, _c in OUTCOMES]
        if p["outcomes"].get("other", 0):
            raise FigError(f"F5: {p['method']} s{p['seed']} has 'other' outcomes the figure has no colour for")
        rows.append({"method": p["method"], "seed": p["seed"], "n": p["n"], "counts": counts})
    return rows


def _final(df: pd.DataFrame) -> pd.DataFrame:
    """T-final's rows, with Fixed-DR's from T-final-F (same table, D-192)."""
    return df[df["block"].isin(("T-final", ARM_BLOCK["T-final"])) & (df["tag"] == "")]


def data_f6(df: pd.DataFrame) -> dict:
    fin = _final(df)
    out = {"ecdf": {}, "cum": {}, "limit": sorted(set(fin["force_limit_n"].astype(float)))}
    if len(out["limit"]) != 1:
        raise FigError(f"F6: more than one force limit in T-final: {out['limit']}")
    kmax = int(fin["steps"].max())
    ks = np.arange(1, kmax + 1)
    for m in arms_in(df):
        g = fin[fin["method"] == m]
        ok = g[g[KEY] == 1]
        f = np.sort(ok["force_max_filtered_n"].to_numpy(dtype=float))
        out["ecdf"][m] = {"x": f, "y": np.arange(1, len(f) + 1) / len(f)}
        seeds = {}
        for s, gs in g.groupby("seed", sort=True):
            st = gs.loc[gs[KEY] == 1, "steps"].to_numpy(dtype=int)
            seeds[int(s)] = np.array([(st <= k).sum() for k in ks]) / len(gs)
        out["cum"][m] = {"k": ks, "seeds": seeds, "mean": np.mean(list(seeds.values()), axis=0)}
    return out


def data_f7(curves: pd.DataFrame) -> dict:
    out = {}
    for m in ARMS:
        g = curves[curves["method"] == m]
        its = None
        seeds = {}
        for s, gs in g.groupby("seed", sort=True):
            gs = gs.sort_values("iteration")
            it = gs["iteration"].to_numpy(dtype=int)
            if its is None:
                its = it
            elif not np.array_equal(it, its):
                raise FigError(f"F7: {m} s{s} logs other iterations than the first seed")
            seeds[int(s)] = gs["Episode/success_rate"].to_numpy(dtype=float)
        if not seeds and m == "fixeddr":
            continue   # optional arm: extract_training took all five runs or none
        if not seeds:
            raise FigError(f"F7: no {m} curves")
        out[m] = {"it": its, "seeds": seeds, "mean": np.mean(list(seeds.values()), axis=0)}
    return out


def data_f8(curves: pd.DataFrame, sidecars: dict) -> dict:
    """Same normalisation as plot_autodr_run.fig_stufen: (v - centre)/(max - centre)
    per side; the floor (v - f0)/(top - f0). Read from each run's sidecar."""
    out = {}
    g = curves[curves["method"] == "autodr"]
    for s, gs in g.groupby("seed", sort=True):
        ad = sidecars[("autodr", int(s))]["autodr"]
        dims = {d["name"]: d for d in ad["dims"]}
        fl = ad["floor"]
        gs = gs.sort_values("iteration")
        it = gs["iteration"].to_numpy(dtype=int)
        runs = {}
        for name, _label in F8_PANELS:
            if name == "start_height_floor":
                v = gs["dr/start_height_floor"].to_numpy(dtype=float)
                runs[name] = {"hi": (v - fl["f0"]) / (fl["top"] - fl["f0"])}
                continue
            d = dims[name]
            v = gs[f"dr/{name}_hi"].to_numpy(dtype=float)
            sides = {"hi": (v - d["centre"]) / (d["hi_max"] - d["centre"])}
            if not d["one_sided"]:
                v = gs[f"dr/{name}_lo"].to_numpy(dtype=float)
                sides["lo"] = (v - d["centre"]) / (d["lo_max"] - d["centre"])
            runs[name] = sides
        out[int(s)] = {"it": it, "panels": runs}
    return out


def data_f9(df: pd.DataFrame) -> dict:
    fin = _final(df)
    le, te = ES.BIN_EDGES["lat_r_mm"], ES.BIN_EDGES["tilt_deg"]
    out = {"lat_edges": list(le), "tilt_edges": list(te), "methods": {}}
    for m in arms_in(df):
        g = fin[fin["method"] == m]
        li = bin_index(g["lat_r_mm"], le)
        ti = bin_index(g["tilt_deg"], te)
        rate = np.full((len(te), len(le)), np.nan)
        rows = np.zeros((len(te), len(le)), dtype=int)
        for a in range(len(te)):
            for b in range(len(le)):
                sel = (ti == a) & (li == b)
                rows[a, b] = g.loc[sel, "row_id"].nunique()
                if sel.any():
                    rate[a, b] = float(g.loc[sel, KEY].mean())
        out["methods"][m] = {"rate": rate, "rows": rows}
    return out


def data_f10(df: pd.DataFrame, t1_mm: float) -> dict:
    fin = _final(df)
    # Beyond the largest reading of a SUCCESS no curve moves any more.
    top = float(fin.loc[fin[KEY] == 1, "ip_max_mm"].max())
    taus =np.unique(np.concatenate([np.linspace(0.0, max(top, t1_mm) * 1.05, 400), [t1_mm]]))
    out = {"tau": taus, "t1": t1_mm, "methods": {}}
    for m in arms_in(df):
        g = fin[fin["method"] == m]
        ok = g[KEY].to_numpy(dtype=int) == 1
        ip = g["ip_max_mm"].to_numpy(dtype=float)
        out["methods"][m] = np.array([(ok & (ip < t)).sum() for t in taus]) / len(g)
    return out


# ---------------------------------------------------------------------------
# training data: extraction (reads git-ignored rt_logs) and loading
# ---------------------------------------------------------------------------

def _env_value(path: pathlib.Path, key: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"^{re.escape(key)}:\s*(.*)$", line)
        if m:
            return m.group(1).strip().strip("'\"")
    raise FigError(f"{path.name}: no top-level key {key!r}")


def extract_training(src: pathlib.Path, out: pathlib.Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    have = [run for run in FIXED_RUNS.values() if (src / run).is_dir()]
    if have and len(have) != len(FIXED_RUNS):
        raise FigError(f"Fixed-DR training runs: only {have} under {src}, want all {list(FIXED_RUNS.values())}")
    rows = []
    nodr_env = {}

    def _curve(m, s, run):
        with open(src / run / f"{run}_scalars.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("Episode/success_rate", "") == "":
                    continue
                rows.append({"method": m, "seed": s, "iteration": int(float(r["step"])),
                             **{t: r.get(t, "") for t in TRAINING_TAGS}})

    for (m, s), run in sorted(TRAINING_RUNS.items()):
        d = src / run
        _curve(m, s, run)
        shutil.copyfile(d / f"{run}_autodr_1499.json", out / f"{m}_s{s}_autodr_1499.json")
        if m == "nodr":
            nodr_env[s] = {k: _env_value(d / f"{run}_env.yaml", k) for k in NODR_ENV_KEYS}
    # After the pair, so the pair's rows keep their place in curves.csv.
    for (m, s), run in sorted(FIXED_RUNS.items()) if have else ():
        _curve(m, s, run)
    first = next(iter(nodr_env.values()))
    for s, v in nodr_env.items():
        if v != first:
            raise FigError(f"No-DR seed {s} trained with other values than seed 1: {v} vs {first}")
    if first["dr_mode"] != "off":
        raise FigError(f"No-DR env.yaml says dr_mode {first['dr_mode']!r}, not 'off'")
    with open(out / "curves.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "seed", "iteration", *TRAINING_TAGS], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    # Start height of No-DR: the range its own provider held at the end.
    last = {}
    for r in rows:
        if r["method"] == "nodr" and (r["seed"] not in last or r["iteration"] > last[r["seed"]]["iteration"]):
            last[r["seed"]] = r
    sh = sorted({(float(r["dr/start_height_floor"]), float(r["dr/start_height_hi"])) for r in last.values()})
    if len(sh) != 1:
        raise FigError(f"No-DR seeds end on different start-height ranges: {sh}")
    nodr = {"source": "RT-207s1..s5 env.yaml + last scalars row", "env": first,
            "lat_r_mm": 1000.0 * float(first["start_lateral_offset"]),
            "yaw_abs_deg": math.degrees(abs(float(first["fixture_yaw_rad"]))),
            "tilt_deg": math.degrees(float(first["fixture_tilt_rad"])),
            "start_height_m": list(sh[0])}
    for k in ("fixture_yaw_noise_rad", "fixture_tilt_noise_rad"):
        if float(first[k]) != 0.0:
            raise FigError(f"No-DR trained with {k} = {first[k]}: the F3 marker would be a band, not a value")
    with open(out / "nodr_training.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nodr, fh, indent=2, sort_keys=True)
    return {"rows": len(rows), "nodr": nodr}


def load_training(folder: pathlib.Path):
    curves = pd.read_csv(folder / "curves.csv")
    sidecars = {(m, s): json.loads((folder / f"{m}_s{s}_autodr_1499.json").read_text(encoding="utf-8"))
                for (m, s) in TRAINING_RUNS}
    nodr = json.loads((folder / "nodr_training.json").read_text(encoding="utf-8"))
    return curves, sidecars, nodr


# ---------------------------------------------------------------------------
# DRAW functions
# ---------------------------------------------------------------------------

def _style(ax, ylabel: str = "", xlabel: str = ""):
    from matplotlib.ticker import ScalarFormatter
    # Numeric axes only; an axis with text ticks (set_xticks with labels) keeps them.
    for axis in (ax.xaxis, ax.yaxis):
        if type(axis.get_major_formatter()) is ScalarFormatter:
            axis.set_major_formatter(_de_formatter())
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)


def _legend_methods(fig, loc="lower center", extra=(), marker=True, methods=METHODS):
    """``marker=False`` for figures that draw lines only: the legend shows what
    the plot shows. ``methods``: the arms the figure draws, in ARMS order."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=METHOD_COLOR[m], marker="o" if marker else None, linewidth=2, markersize=5,
                label=METHOD_LABEL[m])
         for m in methods] + list(extra)
    fig.legend(handles=h, loc=loc, ncol=len(methods) + len(extra), frameon=False, fontsize=8)


def draw_f1(plt, d):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.0), sharey=True)
    offs = np.linspace(-0.16, 0.16, 5)
    arms = list(d["T-final"])
    for ax, block in zip(axes, ("T-nominal", "T-final")):
        for j, m in enumerate(arms):
            r = d[block][m]
            for i, (rate, w) in enumerate(zip(r["rates"], r["wilson"])):
                x = j + offs[i % len(offs)]
                ax.errorbar(x, 100 * rate, yerr=[[100 * (rate - w[0])], [100 * (w[1] - rate)]], fmt="o",
                            color=METHOD_COLOR[m], markersize=3.5, elinewidth=1, capsize=0,
                            markeredgecolor="white", markeredgewidth=0.4)
            ax.plot([j - 0.3, j + 0.3], [100 * r["mean"]] * 2, color=INK, linewidth=1.4)
            ax.annotate(f"{_num(100 * r['mean'], '.1f')} %", (j + 0.32, 100 * r["mean"]), fontsize=7, color=INK,
                        va="center")
        ax.set_xticks(range(len(arms)), [METHOD_LABEL[m] for m in arms])
        ax.set_xlim(-0.5, len(arms) - 0.3)
        ax.set_title(block, fontsize=9, color=INK)
        _style(ax, "Erfolgsrate in %" if block == "T-nominal" else "")
    axes[0].set_ylim(0, 102)
    fig.tight_layout()
    return fig


def draw_f2(plt, rows):
    # Two rows = the pre-registered K1/K2 figure, unchanged. With K5 each row
    # names its own comparison, and the axis label turns general.
    pair_only = len(rows) == 2
    fig, axes = plt.subplots(len(rows), 1, figsize=(FIG_W, 1.2 * len(rows)))
    for ax, r in zip(axes, rows):
        lo = min(0.0, r["welch"][0], r["boot"][0])
        hi = max(0.0, r["welch"][1], r["boot"][1])
        pad = 0.15 * (hi - lo)
        ax.axvline(0.0, color=INK, linewidth=0.9)
        ax.plot([100 * v for v in r["welch"]], [0.15, 0.15], color=INK, linewidth=3.2, solid_capstyle="butt",
                label="Welch-95-%-Intervall")
        ax.plot([100 * v for v in r["boot"]], [-0.15, -0.15], color=INK_MUTED, linewidth=1.2,
                label="Bootstrap-Intervall (Gegenprobe)")
        ax.plot([100 * r["diff"]], [0.15], "o", color="white", markeredgecolor=INK, markersize=6)
        ax.set_xlim(100 * (lo - pad), 100 * (hi + pad))
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        what = "Differenz" if pair_only else r["label"]
        key = r["key"].split("_")[0]
        ax.set_title(f"{r['block']} ({key}): {what} {_num(100 * r['diff'], '+.1f')} Prozentpunkte",
                     fontsize=8.5, color=INK, loc="left")
        _style(ax)
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("Erfolgsrate AutoDR − No-DR in Prozentpunkten" if pair_only
                        else "Differenz der Erfolgsraten in Prozentpunkten", color=INK, fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08 if pair_only else 0.045, 1, 1))
    return fig


def draw_f3(plt, d, nodr):
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 4.6), sharey=True)
    for ax, (col, label, fac) in zip(axes.flat, F3_PANELS):
        p = d[col]
        if col == "start_height_m":
            a, b = nodr["start_height_m"]
            ax.axvspan(fac * a, fac * b, color=GRID, alpha=0.7, linewidth=0)
            ax.text(fac * a, 3, " No-DR-Training", fontsize=7, color=INK_MUTED, va="bottom")
        else:
            v = nodr[col]
            ax.axvline(v, color=INK_MUTED, linewidth=1.0, linestyle="--")
            ax.text(v, 3, " No-DR-Training", fontsize=7, color=INK_MUTED, va="bottom")
        for m in p["methods"]:
            for rates in p["methods"][m]["seeds"].values():
                ax.plot(p["x"], [100 * r for r in rates], color=METHOD_COLOR[m], linewidth=0.8, alpha=0.35)
            ax.plot(p["x"], [100 * r for r in p["methods"][m]["mean"]], color=METHOD_COLOR[m], linewidth=2.0,
                    marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_ylim(0, 102)
        _style(ax, "Erfolgsrate in %" if ax in axes[:, 0] else "", label)
    _legend_methods(fig, methods=tuple(d[F3_PANELS[0][0]]["methods"]))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def draw_f4(plt, d):
    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    x = np.arange(len(d["keys"]))
    lo, hi = d["overall_wilson"]
    ax.axhspan(100 * lo, 100 * hi, color=GRID, linewidth=0)
    ax.axhline(100 * d["overall"], color=INK_MUTED, linewidth=1.0, linestyle="--")
    # Label above the band where no interval reaches above it; looked for in the
    # data, not fixed at a position (RT-225: between tilt_hi and start_height_hi).
    tops = [w[1] for w in d["wilson"]]
    gaps = [i for i in range(len(tops) - 1) if max(tops[i], tops[i + 1]) < hi]
    ax.text((gaps[0] + 0.5) if gaps else (len(x) - 1) / 2, 100 * hi + 0.1, f"T-final gesamt {_num(100 * d['overall'], '.1f')} % (Wilson-Band)", fontsize=7,
            color=INK_MUTED, ha="center", va="bottom")
    c = METHOD_COLOR[d["policy"]["method"]]
    for xi, rate, w in zip(x, d["rates"], d["wilson"]):
        ax.errorbar(xi, 100 * rate, yerr=[[100 * (rate - w[0])], [100 * (w[1] - rate)]], fmt="o", color=c,
                    markersize=6, elinewidth=1.2, capsize=0, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(x, [K3_LABEL[k] for k in d["keys"]])
    ymin = 100 * min(w[0] for w in d["wilson"])
    ax.set_ylim(math.floor(ymin / 5) * 5, 101)
    _style(ax, "Erfolgsrate in %")
    fig.tight_layout()
    return fig


def draw_f5(plt, rows):
    # 3.4 in holds the ten pair bars; each further bar (Fixed-DR) adds its pitch.
    fig, ax = plt.subplots(figsize=(FIG_W, 3.4 + 0.25 * max(0, len(rows) - 10)))
    # Top to bottom in list order, with a gap of 0.6 bar pitches where the
    # method changes, so the two groups read as two groups.
    pos, p = [], 0.0
    for i, r in enumerate(rows):
        if i > 0 and r["method"] != rows[i - 1]["method"]:
            p += 0.6
        pos.append(p)
        p += 1.0
    y = pos[-1] - np.array(pos)
    for yi, r in zip(y, rows):
        left = 0.0
        for (key, label, colr), cnt in zip(OUTCOMES, r["counts"]):
            w = 100 * cnt / r["n"]
            ax.barh(yi, w, left=left, color=colr, edgecolor="white", linewidth=1.0, height=0.7, label=label)
            if w >= 8:
                ax.text(left + w / 2, yi, f"{w:.0f}", ha="center", va="center", fontsize=7, color=INK)
            left += w
    ax.set_yticks(y, [f"{METHOD_LABEL[r['method']]} s{r['seed']}" for r in rows])
    ax.set_xlim(0, 100)
    _style(ax, "", "Anteil der Episoden auf T-final in %")
    ax.grid(False)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h[:len(OUTCOMES)], lab[:len(OUTCOMES)], loc="lower center", ncol=len(OUTCOMES),
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def draw_f6(plt, d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 2.9))
    for m in d["ecdf"]:
        e = d["ecdf"][m]
        a1.step(e["x"], 100 * e["y"], where="post", color=METHOD_COLOR[m], linewidth=1.6)
    lim = d["limit"][0]
    a1.axvline(lim, color=INK, linewidth=0.9, linestyle="--")
    a1.text(lim, 50, f" Abbruch-\n grenze {_num(lim, 'g')} N", fontsize=7, color=INK, va="center")
    _style(a1, "Anteil der Erfolge in %", "Maximalkraft je Erfolg in N")
    for m in d["ecdf"]:
        c = d["cum"][m]
        for v in c["seeds"].values():
            a2.plot(c["k"], 100 * v, color=METHOD_COLOR[m], linewidth=0.7, alpha=0.35)
        a2.plot(c["k"], 100 * c["mean"], color=METHOD_COLOR[m], linewidth=1.8)
    a2.set_ylim(0, 102)
    _style(a2, "Anteil erfolgreich bis Schritt k in %", "Policy-Schritt k")
    _legend_methods(fig, marker=False, methods=tuple(d["ecdf"]))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def draw_f7(plt, d):
    fig, ax = plt.subplots(figsize=(FIG_W, 2.8))
    # All thin seed lines first, then both means: no method's seeds may cover
    # the other method's mean.
    for m in d:
        c = d[m]
        for v in c["seeds"].values():
            ax.plot(c["it"], 100 * v, color=METHOD_COLOR[m], linewidth=0.6, alpha=0.3)
    for m in d:
        ax.plot(d[m]["it"], 100 * d[m]["mean"], color=METHOD_COLOR[m], linewidth=1.8)
    ax.set_ylim(0, 102)
    _style(ax, "Erfolgsrate im Training in %", "PPO-Iteration")
    _legend_methods(fig, marker=False, methods=tuple(d))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def draw_f8(plt, d):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 3.9), sharex=True, sharey=True)
    c = METHOD_COLOR["autodr"]
    for ax, (name, label) in zip(axes.flat, F8_PANELS):
        for run in d.values():
            p = run["panels"][name]
            ax.plot(run["it"], 100 * p["hi"], color=c, linewidth=0.9, alpha=0.8)
            if "lo" in p:
                ax.plot(run["it"], 100 * p["lo"], color=c, linewidth=0.9, alpha=0.8, linestyle="--")
        ax.set_title(label, fontsize=8.5, color=INK)
        ax.set_ylim(-5, 108)
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("PPO-Iteration", color=INK, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Anteil am Maximum in %", color=INK, fontsize=9)
    h = [Line2D([], [], color=c, linewidth=1.2, label="obere Grenze (bzw. Boden)"),
         Line2D([], [], color=c, linewidth=1.2, linestyle="--", label="untere Grenze")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def draw_f9(plt, d):
    arms = list(d["methods"])
    fig, axes = plt.subplots(1, len(arms), figsize=(FIG_W, 3.4), sharey=True)
    le, te = d["lat_edges"], d["tilt_edges"]
    xl = [f"{_num(bin_lower(le, i), 'g')}–{_num(le[i], 'g')}" for i in range(len(le))]
    yl = [f"{_num(bin_lower(te, i), 'g')}–{_num(te[i], 'g')}" for i in range(len(te))]
    im = None
    for ax, m in zip(axes, arms):
        r = d["methods"][m]
        im = ax.imshow(100 * r["rate"], origin="lower", cmap=HEAT_CMAP, vmin=0, vmax=100, aspect="auto")
        for a in range(len(te)):
            for b in range(len(le)):
                v = r["rate"][a, b]
                txt = "–" if np.isnan(v) else f"{100 * v:.0f}"
                ax.text(b, a + 0.12, txt, ha="center", va="center", fontsize=7.5,
                        color="white" if (not np.isnan(v) and v > 0.6) else INK)
                ax.text(b, a - 0.25, f"n={r['rows'][a, b]}", ha="center", va="center", fontsize=6.5,
                        color="white" if (not np.isnan(v) and v > 0.6) else INK_MUTED)
        # Three panels leave ~0.35 in per cell: 7 pt labels run into each other,
        # so they stand at 45 deg there (6.5 pt, the n= size the user set).
        three = len(arms) > 2
        ax.set_xticks(range(len(le)), xl, fontsize=6.5 if three else 7, rotation=45 if three else 0,
                      ha="right" if three else "center", rotation_mode="anchor")
        ax.set_yticks(range(len(te)), yl, fontsize=7)
        ax.set_title(METHOD_LABEL[m], fontsize=9, color=INK)
        ax.set_xlabel("Seitlicher Versatz in mm", color=INK, fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK_MUTED, length=0)
    axes[0].set_ylabel("Kippung in °", color=INK, fontsize=9)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.035, pad=0.02)
    cb.set_label("Erfolgsrate in %", color=INK, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED)
    cb.outline.set_visible(False)
    return fig


def draw_f10(plt, d):
    fig, ax = plt.subplots(figsize=(FIG_W, 2.8))
    for m in d["methods"]:
        ax.plot(d["tau"], 100 * d["methods"][m], color=METHOD_COLOR[m], linewidth=1.8)
    ax.axvline(d["t1"], color=INK, linewidth=0.9, linestyle="--")
    ax.text(d["t1"], 50, f" t1 = {_num(d['t1'], '.3f')} mm\n (algorithmische Schwelle,\n keine Zulässigkeitsgrenze)",
            fontsize=7, color=INK, va="center")
    ax.set_ylim(0, 102)
    ax.set_xlim(0, float(d["tau"][-1]))
    _style(ax, "Erfolgsrate in %", "Erfolg nur gezählt bei maximaler Durchdringung < τ, τ in mm")
    _legend_methods(fig, marker=False, methods=tuple(d["methods"]))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


NAMES = {"F1": "F1_erfolg_je_seed", "F2": "F2_differenz", "F3": "F3_stoergroessen", "F4": "F4_grenzbloecke",
         "F5": "F5_endgruende", "F6": "F6_kraft_und_dauer", "F7": "F7_lernkurven", "F8": "F8_autodr_grenzen",
         "F9": "F9_heatmap_versatz_kippung", "F10": "F10_durchdringung"}


def t1_from_results(results: pathlib.Path) -> float:
    vals = {json.loads(p.read_text(encoding="utf-8"))["interpenetration"]["t1_mm"]
            for p in sorted((results / "T-final").glob("*/episodes_meta.json"))}
    if len(vals) != 1:
        raise FigError(f"T-final episodes_meta.json files disagree on t1_mm (or none): {sorted(vals)}")
    return float(next(iter(vals)))


def draw_all(results: pathlib.Path, training: pathlib.Path, out: pathlib.Path, only=None) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    except ValueError as exc:
        raise FigError(f"font {FONT!r} not installed; matplotlib would fall back silently") from exc
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.family": "sans-serif", "font.sans-serif": [FONT]})
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(results / "episodes_all.csv", keep_default_na=False, dtype={"tag": str, "boundary_key": str})
    curves, sidecars, nodr = load_training(training)
    jobs = {
        "F1": lambda: draw_f1(plt, data_f1(summary)),
        "F2": lambda: draw_f2(plt, data_f2(summary)),
        "F3": lambda: draw_f3(plt, data_f3(summary), nodr),
        "F4": lambda: draw_f4(plt, data_f4(summary)),
        "F5": lambda: draw_f5(plt, data_f5(summary)),
        "F6": lambda: draw_f6(plt, data_f6(df)),
        "F7": lambda: draw_f7(plt, data_f7(curves)),
        "F8": lambda: draw_f8(plt, data_f8(curves, sidecars)),
        "F9": lambda: draw_f9(plt, data_f9(df)),
        "F10": lambda: draw_f10(plt, data_f10(df, t1_from_results(results))),
    }
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key, job in jobs.items():
        if only and key not in only:
            continue
        fig = job()
        for ext in ("pdf", "png"):
            p = out / f"{NAMES[key]}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight", metadata={"CreationDate": None} if ext == "pdf" else None)
            written.append(p)
        plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Self-test (D-080): checks on built data; each mutation of THIS file's source
# must flip exactly its declared checks.
# ---------------------------------------------------------------------------

def _synthetic():
    """Two seeds per method, 20 rows per policy and block; T-grenze 7 x 4 rows
    for autodr s1. Known answers are written next to the checks."""
    rows = []
    n = 20
    k_by = {("autodr", 1): 18, ("autodr", 2): 16, ("nodr", 1): 6, ("nodr", 2): 4}
    for (m, s), k in k_by.items():
        for block in ("T-final", "T-nominal", "T-auswahl"):
            kk = k if block != "T-nominal" else 20 - s   # 19 / 18: K2 has a variance
            for i in range(n):
                ok = int(i < kk)
                abort = int((not ok) and i == n - 1)
                rows.append({"block": block, "method": m, "seed": s, "tag": "", "row_id": i, "boundary_key": "",
                             "outcome": "success" if ok else ("force_abort" if abort else "timeout"),
                             "success_train": ok, "success_strict": ok, "force_abort": abort,
                             "steps": 10 + i if ok else 256, "force_max_filtered_n": 5.0 + i,
                             "force_max_raw_n": 5.0 + i, "ip_max_mm": 0.01 * i, "force_limit_n": 30.0,
                             # lat 0..28.5 mm, tilt 0..9.5 deg; start height only in 0.04..0.12
                             "lat_r_mm": 1.5 * i, "yaw_abs_deg": 0.25 * i, "tilt_deg": 0.5 * i,
                             "start_height_m": 0.03 + 0.004 * i, "friction": 0.14})
    # Fixed-DR (D-192) in its own blocks: T-final-F 12/20 and 10/20, T-nominal-F 17/20 and 16/20.
    for s, k in ((1, 12), (2, 10)):
        for block, kk in (("T-final-F", k), ("T-nominal-F", 18 - s)):
            for i in range(n):
                ok = int(i < kk)
                rows.append({"block": block, "method": "fixeddr", "seed": s, "tag": "", "row_id": i,
                             "boundary_key": "", "outcome": "success" if ok else "timeout",
                             "success_train": ok, "success_strict": ok, "force_abort": 0,
                             "steps": 10 + i if ok else 256, "force_max_filtered_n": 5.0 + i,
                             "force_max_raw_n": 5.0 + i, "ip_max_mm": 0.01 * i, "force_limit_n": 30.0,
                             "lat_r_mm": 1.5 * i, "yaw_abs_deg": 0.25 * i, "tilt_deg": 0.5 * i,
                             "start_height_m": 0.03 + 0.004 * i, "friction": 0.14})
    for j, key in enumerate(K3_ORDER):
        for i in range(4):
            ok = int(not (key == "tilt_hi" and i == 0))
            rows.append({"block": "T-grenze", "method": "autodr", "seed": 1, "tag": "", "row_id": 4 * j + i,
                         "boundary_key": key, "outcome": "success" if ok else "timeout", "success_train": ok,
                         "success_strict": ok, "force_abort": 0, "steps": 20, "force_max_filtered_n": 5.0,
                         "force_max_raw_n": 5.0, "ip_max_mm": 0.0, "force_limit_n": 30.0, "lat_r_mm": 30.0,
                         "yaw_abs_deg": 0.0, "tilt_deg": 0.0, "start_height_m": 0.05, "friction": 0.14})
    df = pd.DataFrame(rows)
    return df, ES.summarize(df)


def _synthetic_curves():
    rows = []
    for m in ARMS:   # Fixed-DR curves too (D-192); F8 reads autodr only
        for s in (1, 2):
            for it in range(3):
                rows.append({"method": m, "seed": s, "iteration": it,
                             "Episode/success_rate": 0.1 * it + (0.1 if s == 2 else 0.0),
                             "dr/lat_r_hi": 0.015 * it, "dr/yaw_lo": -0.0436 * it, "dr/yaw_hi": 0.0436 * it,
                             "dr/tilt_hi": 0.0873 * it, "dr/start_height_hi": 0.02 + 0.05 * it,
                             "dr/start_height_floor": -0.03 + 0.025 * it, "dr/friction_lo": 0.14 - 0.03 * it,
                             "dr/friction_hi": 0.14 + 0.03 * it})
    dims = [{"name": "lat_r", "centre": 0.0, "hi_max": 0.03, "lo_max": 0.0, "one_sided": True},
            {"name": "yaw", "centre": 0.0, "hi_max": 0.0872, "lo_max": -0.0872, "one_sided": False},
            {"name": "tilt", "centre": 0.0, "hi_max": 0.1746, "lo_max": 0.0, "one_sided": True},
            {"name": "start_height", "centre": 0.02, "hi_max": 0.12, "lo_max": 0.02, "one_sided": True},
            {"name": "friction", "centre": 0.14, "hi_max": 0.20, "lo_max": 0.08, "one_sided": False}]
    side = {"autodr": {"dims": dims, "floor": {"f0": -0.03, "top": 0.02, "n_steps": 5}}}
    return pd.DataFrame(rows), {("autodr", s): side for s in (1, 2)}


def _write_rt_logs(root: pathlib.Path, bad_seed: int | None = None, runs=None):
    for (m, s), run in (runs or TRAINING_RUNS).items():
        d = root / run
        d.mkdir(parents=True)
        with open(d / f"{run}_scalars.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "wall_time", *TRAINING_TAGS])
            for it in range(2):
                w.writerow([it, 0, 0.5, "", "", "", "", 0.12, 0.02, "", ""])
            w.writerow([1, 0, "", "", "", "", "", "", "", "", ""])   # a */time row
        (d / f"{run}_autodr_1499.json").write_text("{}", encoding="utf-8")
        tilt = "0.01" if (m == "nodr" and s == bad_seed) else "0.0"
        (d / f"{run}_env.yaml").write_text(
            f"dr_mode: 'off'\nstart_lateral_offset: 0.0\nfixture_yaw_rad: 0.0\nfixture_yaw_noise_rad: 0.0\n"
            f"fixture_tilt_rad: {tilt}\nfixture_tilt_noise_rad: 0.0\n", encoding="utf-8")


CHECKS = (
    "F1: T-final autodr mean = (0.9 + 0.8) / 2 and dots carry summary's Wilson",
    "F2: T-nominal row reads K2, T-final row reads K1",
    "F3: empty start-height bins dropped (5 of 10 edges hold rows)",
    "F3: lateral bin 24-30 mm mean over seeds = autodr (0.5 + 0.0) / 2",
    "F4: keys in the fixed order, tilt_hi 3/4, overall band = Wilson(18, 20)",
    "F5: every policy's segments sum to its n",
    "F6: cumulative share at the last step = the success rate; ECDF over successes only",
    "F7: mean training curve = mean of the seeds",
    "F8: a boundary at its maximum normalises to 1.0 (lat_r, yaw both sides, floor)",
    "F9: cell (tilt 0-2, lat 0-6) pooled rate = 1.0 over 4 distinct rows",
    "F10: rate at the largest tau = success rate; rate at tau 0 = 0",
    "colour follows the method (AutoDR = #3B6FD4, Fixed-DR = #199e70)",
    "extraction refuses No-DR seeds that trained with different values",
    "figure numbers: decimal comma and real minus in text and on numeric tick labels",
    "F1: Fixed-DR from its own blocks, T-final mean (0.6 + 0.5) / 2, T-nominal (0.85 + 0.8) / 2",
    "Fixed-DR is optional: without its blocks F1 draws the two pre-registered methods",
    "one Fixed-DR block without the other is refused",
    "F2: K5_nominal and K5 rows follow K2/K1 and read summary's K5; without Fixed-DR two rows",
    "F3: Fixed-DR lateral bin 12-18 mm mean over seeds = (1.0 + 0.5) / 2",
    "F5: Fixed-DR bars follow the pair's, from T-final-F (s1: 12 successes of 20)",
    "F6: Fixed-DR from T-final-F: s1 share at the last step = 0.6, ECDF over its 22 successes",
    "F9: Fixed-DR panel, cell (tilt 0-2, lat 0-6) pooled rate = 1.0 over 4 distinct rows",
    "F10: Fixed-DR curve from T-final-F: rate at the largest tau = 0.55, at tau 0 = 0",
    "extraction takes all five Fixed-DR runs (RT-231s1..s5) and refuses four",
    "F7: Fixed-DR mean curve when its curves exist; without them F7 keeps the pair",
)


def _run_checks(ns) -> dict:
    res = {}
    df, s = _synthetic()
    curves, side = _synthetic_curves()

    def _check(i, fn):
        # A mutation that makes a data function RAISE counts as a red check.
        try:
            res[CHECKS[i]] = bool(fn())
        except Exception:   # noqa: BLE001
            res[CHECKS[i]] = False

    def c0():
        a = ns["data_f1"](s)["T-final"]["autodr"]
        pol = [p for p in s["policies"] if p["block"] == "T-final" and p["method"] == "autodr" and p["seed"] == 1][0]
        return abs(a["mean"] - 0.85) < 1e-12 and a["wilson"][0] == pol["success_train_wilson"]

    def c1():
        d2 = ns["data_f2"](s)
        return d2[0]["block"] == "T-nominal" and d2[0]["diff"] == s["K2"]["diff"] and d2[1]["diff"] == s["K1"]["diff"]

    def c2():
        sh = ns["data_f3"](s)["start_height_m"]
        return len(sh["x"]) == 5 and all(v > 0 for v in sh["n"])

    def c3():
        # lat 24..28.5 mm = rows 16..19: s1 succeeds on 16, 17 (0.5), s2 on none (0.0)
        return abs(ns["data_f3"](s)["lat_r_mm"]["methods"]["autodr"]["mean"][-1] - 0.25) < 1e-12

    def c4():
        d4 = ns["data_f4"](s)
        return (d4["keys"] == list(K3_ORDER) and abs(d4["rates"][3] - 0.75) < 1e-12
                and np.allclose(d4["overall_wilson"], ES.wilson(18, 20)))

    def c6():
        d6 = ns["data_f6"](df)
        c = d6["cum"]["autodr"]
        return (abs(c["seeds"][1][-1] - 0.9) < 1e-12 and np.all(np.diff(c["mean"]) >= 0)
                and len(d6["ecdf"]["nodr"]["x"]) == 10)

    def c8():
        p = ns["data_f8"](curves, side)[1]["panels"]
        return (abs(p["lat_r"]["hi"][-1] - 1.0) < 1e-9 and abs(p["yaw"]["lo"][-1] - 1.0) < 0.01
                and abs(p["yaw"]["hi"][-1] - 1.0) < 0.01 and abs(p["start_height_floor"]["hi"][-1] - 1.0) < 1e-9)

    def c9():
        # lat 1.5*i < 6 -> i 0..3; tilt 0.5*i < 2 -> i 0..3: 4 rows, all successes for both seeds
        a = ns["data_f9"](df)["methods"]["autodr"]
        return a["rows"][0, 0] == 4 and abs(a["rate"][0, 0] - 1.0) < 1e-12

    def c10():
        m = ns["data_f10"](df, 0.2)["methods"]["autodr"]
        return abs(m[-1] - 0.85) < 1e-12 and m[0] == 0.0

    _check(0, c0)
    _check(1, c1)
    _check(2, c2)
    _check(3, c3)
    _check(4, c4)
    _check(5, lambda: all(sum(r["counts"]) == r["n"] for r in ns["data_f5"](s)))
    _check(6, c6)
    _check(7, lambda: np.allclose(ns["data_f7"](curves)["autodr"]["mean"], [0.05, 0.15, 0.25]))
    _check(8, c8)
    _check(9, c9)
    _check(10, c10)
    _check(11, lambda: ns["METHOD_COLOR"] == {"autodr": "#3B6FD4", "nodr": "#D97706", "fixeddr": "#199e70"})

    def c13():
        from matplotlib.figure import Figure
        ax = Figure().add_subplot()
        ax.plot([0.0, 1.0], [0.0, 0.5])
        ax.set_ylim(0.0, 0.5)
        ns["_style"](ax)
        ticks = ax.yaxis.get_major_formatter().format_ticks([0.0, 0.25, 0.5])
        return ns["_num"](-1.66, "+.1f") == "−1,7" and ticks == ["0,00", "0,25", "0,50"]

    _check(13, c13)

    def c14():
        f = ns["data_f1"](s)
        return (abs(f["T-final"]["fixeddr"]["mean"] - 0.55) < 1e-12
                and abs(f["T-nominal"]["fixeddr"]["mean"] - 0.825) < 1e-12)

    no_f = df[~df["block"].isin(("T-final-F", "T-nominal-F"))]
    half = df[df["block"] != "T-nominal-F"]
    _check(14, c14)
    _check(15, lambda: sorted(ns["data_f1"](ES.summarize(no_f))["T-final"]) == ["autodr", "nodr"])

    def c16():
        try:
            ns["data_f1"](ES.summarize(half))
            return False
        except Exception as exc:   # noqa: BLE001 - FigError of the executed copy
            return "Fixed-DR" in str(exc)

    _check(16, c16)

    def c17():
        d2 = ns["data_f2"](s)
        return ([r["key"] for r in d2] == ["K2", "K1", "K5_nominal", "K5"]
                and d2[3]["diff"] == s["K5"]["diff"] and d2[2]["welch"] == s["K5_nominal"]["welch_ci"]
                and d2[3]["label"] == "AutoDR − Fixed-DR"
                and len(ns["data_f2"](ES.summarize(no_f))) == 2)

    _check(17, c17)
    # lat 12..16.5 mm = rows 8..11: Fixed-DR s1 (12 successes) all, s2 (10) rows 8, 9.
    _check(18, lambda: abs(ns["data_f3"](s)["lat_r_mm"]["methods"]["fixeddr"]["mean"][2] - 0.75) < 1e-12)

    def c19():
        r5 = ns["data_f5"](s)
        return ([r["method"] for r in r5] == ["autodr", "autodr", "nodr", "nodr", "fixeddr", "fixeddr"]
                and r5[4]["counts"][0] == 12 and sum(r5[4]["counts"]) == 20)

    _check(19, c19)

    def c20():
        d6 = ns["data_f6"](df)
        return (abs(d6["cum"]["fixeddr"]["seeds"][1][-1] - 0.6) < 1e-12
                and len(d6["ecdf"]["fixeddr"]["x"]) == 22)

    _check(20, c20)

    def c21():
        f = ns["data_f9"](df)["methods"]["fixeddr"]
        return f["rows"][0, 0] == 4 and abs(f["rate"][0, 0] - 1.0) < 1e-12

    _check(21, c21)

    def c22():
        f = ns["data_f10"](df, 0.2)["methods"]["fixeddr"]
        return abs(f[-1] - 0.55) < 1e-12 and f[0] == 0.0

    _check(22, c22)

    def c24():
        d7 = ns["data_f7"](curves)
        pair = ns["data_f7"](curves[curves["method"] != "fixeddr"])
        return np.allclose(d7["fixeddr"]["mean"], [0.05, 0.15, 0.25]) and sorted(pair) == ["autodr", "nodr"]

    _check(24, c24)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        _write_rt_logs(t / "rt", bad_seed=3)
        try:
            ns["extract_training"](t / "rt", t / "out")
            res[CHECKS[12]] = False
        except Exception as exc:   # noqa: BLE001 - FigError of the executed copy
            res[CHECKS[12]] = "trained with other values" in str(exc)
        _write_rt_logs(t / "rt_ok")
        try:
            got = ns["extract_training"](t / "rt_ok", t / "out_ok")
            res[CHECKS[12]] = res[CHECKS[12]] and got["rows"] == 2 * len(TRAINING_RUNS)
        except Exception:   # noqa: BLE001
            res[CHECKS[12]] = False
        # Fixed-DR training runs (D-192): all five -> extracted; four -> refused.
        _write_rt_logs(t / "rt_f", runs={**TRAINING_RUNS, **FIXED_RUNS})
        try:
            got = ns["extract_training"](t / "rt_f", t / "out_f")
            res[CHECKS[23]] = got["rows"] == 2 * (len(TRAINING_RUNS) + len(FIXED_RUNS))
        except Exception:   # noqa: BLE001
            res[CHECKS[23]] = False
        shutil.rmtree(t / "rt_f" / FIXED_RUNS[("fixeddr", 5)])
        try:
            ns["extract_training"](t / "rt_f", t / "out_f4")
            res[CHECKS[23]] = False
        except Exception as exc:   # noqa: BLE001
            res[CHECKS[23]] = res[CHECKS[23]] and "Fixed-DR" in str(exc)
    return res


MUTATIONS = (
    ("f1-mean-drops-last-seed", '"mean": float(np.mean(rates))}', '"mean": float(np.mean(rates[:-1]))}',
     (CHECKS[0], CHECKS[14])),
    ("f2-blocks-swapped", '(("K2", "T-nominal"), ("K1", "T-final"))', '(("K1", "T-nominal"), ("K2", "T-final"))',
     (CHECKS[1], CHECKS[17])),
    # An empty bin has rate None: the mean over it raises, so the whole of
    # data_f3 goes red -- both F3 checks.
    ("f3-keeps-empty-bins", "keep = [i for i, n in enumerate(ns) if n > 0]",
     "keep = list(range(len(ns)))", (CHECKS[2], CHECKS[3], CHECKS[18])),
    ("f3-mean-is-first-seed", "np.mean([seeds[k][j] for k in keys]))",
     "np.mean([seeds[k][j] for k in keys[:1]]))", (CHECKS[3], CHECKS[18])),
    ("f4-band-from-wilson-n-minus-1", '"overall_wilson": ES.wilson(ov["k"], ov["n"])',
     '"overall_wilson": ES.wilson(ov["k"], ov["n"] - 1)', (CHECKS[4],)),
    ("f5-drops-timeouts", '    ("timeout", "Zeitlimit", "#ec835a"),\n', "", (CHECKS[5], CHECKS[19])),
    ("f6-counts-all-episodes-in-ecdf", "ok = g[g[KEY] == 1]\n        f = np.sort",
     "ok = g\n        f = np.sort", (CHECKS[6], CHECKS[20])),
    ("f7-mean-is-max", '"mean": np.mean(list(seeds.values()), axis=0)}\n    return out\n\n\ndef data_f8',
     '"mean": np.max(list(seeds.values()), axis=0)}\n    return out\n\n\ndef data_f8', (CHECKS[7], CHECKS[24])),
    ("f8-lo-side-over-hi-max", 'sides["lo"] = (v - d["centre"]) / (d["lo_max"] - d["centre"])',
     'sides["lo"] = (v - d["centre"]) / (d["hi_max"] - d["centre"])', (CHECKS[8],)),
    ("f9-counts-episodes-not-rows", 'rows[a, b] = g.loc[sel, "row_id"].nunique()',
     "rows[a, b] = int(sel.sum())", (CHECKS[9], CHECKS[21])),
    ("f10-counts-every-episode", "(ok & (ip < t)).sum()", "(ip < t).sum()", (CHECKS[10], CHECKS[22])),
    ("colours-swapped", 'METHOD_COLOR = {"autodr": "#3B6FD4", "nodr": "#D97706", "fixeddr": "#199e70"}',
     'METHOD_COLOR = {"autodr": "#3B6FD4", "nodr": "#199e70", "fixeddr": "#D97706"}', (CHECKS[11],)),
    ("extraction-drops-the-equality-guard", "        if v != first:\n",
     "        if False:\n", (CHECKS[12],)),
    ("figure-numbers-keep-the-point", 'return s.replace(".", ",")', "return s", (CHECKS[13],)),
    ("style-keeps-the-default-tick-formatter", "            axis.set_major_formatter(_de_formatter())\n",
     "            pass\n", (CHECKS[13],)),
    # Fixed-DR (D-192): its own blocks, optional, both or none.
    # data_f1 then finds no Fixed-DR run and raises: both F1 checks go red.
    ("f1-reads-fixed-dr-from-the-pair-block", "_policies(summary, ARM_BLOCK[block])", "_policies(summary, block)",
     (CHECKS[0], CHECKS[14], CHECKS[19])),
    ("fixed-dr-always-required", "    if not blocks:\n        return METHODS", "    if False:\n        return METHODS",
     (CHECKS[15], CHECKS[17])),
    ("half-fixed-dr-accepted", "    if blocks == set(ARM_BLOCK.values()):", "    if blocks:",
     (CHECKS[16],)),
    ("f2-k5-rows-swapped", '(("K5_nominal", "T-nominal"), ("K5", "T-final"))',
     '(("K5", "T-nominal"), ("K5_nominal", "T-final"))', (CHECKS[1], CHECKS[17])),
    ("f3-ignores-fixed-dr", 'k.startswith(m + "_s")) for m in arms_present(summary)}',
     'k.startswith(m + "_s")) for m in METHODS}', (CHECKS[18],)),
    ("f5-ignores-fixed-dr", "    for p in _arm_policies(summary, \"T-final\"):", "    for p in _policies(summary, \"T-final\"):",
     (CHECKS[19],)),
    ("final-ignores-the-fixed-dr-block", 'df["block"].isin(("T-final", ARM_BLOCK["T-final"]))',
     'df["block"].isin(("T-final",))', (CHECKS[20], CHECKS[21], CHECKS[22])),
    ("f9-ignores-fixed-dr", 'list(te), "methods": {}}\n    for m in arms_in(df):', 'list(te), "methods": {}}\n    for m in METHODS:',
     (CHECKS[21],)),
    ("f10-ignores-fixed-dr", '"t1": t1_mm, "methods": {}}\n    for m in arms_in(df):',
     '"t1": t1_mm, "methods": {}}\n    for m in METHODS:', (CHECKS[22],)),
    ("extraction-takes-four-fixed-dr-runs", "    if have and len(have) != len(FIXED_RUNS):", "    if False:",
     (CHECKS[23],)),
    ("f7-ignores-fixed-dr", "    for m in ARMS:\n        g = curves[", "    for m in METHODS:\n        g = curves[",
     (CHECKS[24],)),
)
_SELF_TEST_MARKER = "# Self-test (D-080): checks on built data"


def self_test() -> int:
    print(f"[eval_figures] marker: {SCRIPT_MARKER}")
    src = pathlib.Path(__file__).read_text(encoding="utf-8")

    def _exec(text):
        ns = {"__name__": "eval_figures_under_test", "__file__": __file__}
        exec(compile(text, __file__, "exec"), ns)  # noqa: S102 - the point of the self-test
        return ns

    base = _run_checks(_exec(src))
    bad = 0
    for name in CHECKS:
        ok = base.get(name) is True
        print(f"[eval_figures]   {'PASS' if ok else 'FAIL'}  {name}")
        bad += not ok
    cut = src.index(_SELF_TEST_MARKER)
    head, tail = src[:cut], src[cut:]
    for mname, old, new, want in MUTATIONS:
        if head.count(old) != 1:
            print(f"[eval_figures]   mutation {mname}: anchor matches {head.count(old)} times -- FAIL")
            bad += 1
            continue
        got = _run_checks(_exec(head.replace(old, new) + tail))
        flipped = {k for k, v in got.items() if v is not True}
        ok = flipped == set(want)
        print(f"[eval_figures]   mutation {mname}: {'PASS' if ok else 'FAIL'}"
              + ("" if ok else f" (flipped {sorted(flipped)}, declared {sorted(want)})"))
        bad += not ok
    undefended = [c for c in CHECKS if not any(c in w for *_, w in MUTATIONS)]
    print(f"[eval_figures]   every check is flipped by some mutation: "
          f"{'PASS' if not undefended else 'FAIL ' + str(undefended)}")
    bad += bool(undefended)
    total = len(CHECKS) + len(MUTATIONS) + 1
    print(f"[eval_figures] self-test: {total - bad}/{total}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--extract-training", type=str, default=None, metavar="RT_LOGS")
    ap.add_argument("--results", type=str, default="results/eval")
    ap.add_argument("--training", type=str, default="results/training")
    ap.add_argument("--out", type=str, default="docs/figures/eval")
    ap.add_argument("--only", type=str, default="", help="comma list, e.g. F1,F9")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    print(f"[eval_figures] marker: {SCRIPT_MARKER}")
    if args.extract_training:
        got = extract_training(pathlib.Path(args.extract_training), pathlib.Path(args.training))
        print(f"[eval_figures] extracted {got['rows']} iteration rows -> {args.training}; No-DR: {got['nodr']}")
        return 0
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    for p in draw_all(pathlib.Path(args.results), pathlib.Path(args.training), pathlib.Path(args.out), only):
        print(f"[eval_figures] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
