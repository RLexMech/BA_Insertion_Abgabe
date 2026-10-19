# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Render evaluation figures from summary.json; F6/F9/F10 also read episodes_all.csv.
Offline matplotlib; output PDF, SVG and 300 dpi PNG. Statistical helpers
and bin rules come from eval_stats.py. DATA and DRAW functions are separate.

F1-F5: seed success, AutoDR-NoDR differences, disturbance sweeps, selected
policy boundary tests and outcome shares. F6: successful peak-force ECDF
and success-by-step fraction of all episodes. F7-F10: training success,
normalised AutoDR boundaries, lateral/tilt cells with counts, and success
under penetration thresholds (IndustReal Fig. 3). F11/F12 compare range
edges/OOD and OOD-minus-own-T-final (Newcombe per seed).
F7-F12 were added after the results were known; they are descriptive,
not selection criteria. F2 contrasts use Welch and bootstrap intervals.

H1-H4 compare arms with individual seeds (Peng 2018 Fig. 7,
Weissgerber 2015, IndustReal Fig. 4, AutoMate S18). H5/H6 use iterations
only when steps per iteration agree, with the factor labelled.
H7/H8 supplement all-seed results with the policy selected by T-auswahl;
best-of-five is a different algorithm (Agarwal 2021).
H12 reads force-blind results from results/eval_rlgames_kraft.
H14 uses all 15 policies in RT-264/265 tilt/friction sweeps.
H15 reads training/penetration.csv at H15_EPOCHS: success below t1 and
penetration >=t1 among successes; the latter requires success at every
point and names excluded seeds. H18 shows OOD; H19/H20 restrict to policies
with own main success >=LEARNED_MIN; an arm's mean requires at least
LEARNED_MEAN_MIN_N qualifying policies.
H22 includes all policies. Paired figures show reference/test dots; H19/H20
use the PAIR_YAXIS truncation guard.

Seed means use pointwise 95% percentile-bootstrap intervals; single-policy
rates use Wilson intervals. Exceptions: H8 quartiles show spread, F5/H4
stacks, F9 heatmap and F8 single-run lines. Show seeds as dots or
thin lines; table/text SD remains a different summary.

Fixed-DR is included when T-nominal-F and T-final-F exist, except F8.
Training axes use each run's agent/env YAML through run_axis.
For rl_games, --arm ARM=PREFIX selects runs; F8 additionally requires
<prefix><N>_autodr_final.json. OOD figures require T-ood data.

Usage:
    .venv-analysis/Scripts/python.exe scripts/eval_figures.py --extract-training rt_logs
    .venv-analysis/Scripts/python.exe scripts/eval_figures.py --results results/eval
    .venv-analysis/Scripts/python.exe scripts/eval_figures.py --extract-training rt_logs --arm autodr=RT-242s --arm nodr=RT-244s --arm fixeddr=RT-243s --training results/training_rlgames
    .venv-analysis/Scripts/python.exe scripts/eval_figures.py --results daten/endstudie_rl_games/evaluation --training daten/endstudie_rl_games/training_auszug --out docs/figures/eval_rlgames
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

SCRIPT_MARKER = "eval_figures-2026-10-07c"
REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ES = _load_module("eval_stats", REPO / "scripts" / "eval_stats.py")
KEY = ES.KEY
# CURVE AXIS: training curves over total env steps, one home for the rule.
run_axis = _load_module("run_axis", REPO / "scripts" / "tools" / "run_axis.py")
XLABEL_TRAINING = "Policy-Schritte in Mio."   # F7, F8


def _mio_axis(ax):
    """x ticks in millions with the decimal comma, no "1e6" offset.
    Call after ``_style``, which only replaces a plain ScalarFormatter."""
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _de_digits(f"{v / 1e6:g}")))
# H5/H6/H9/H11: the training round. The thesis word is "Epoche",
# also used by rl_games (``max_epochs``); rsl_rl calls the same round an iteration.
# The code keeps its column name ``iteration``.
def xlabel_epoch(factor: int) -> str:
    return f"Epoche (je {_thousands(factor)} Policy-Schritte)"
# Colour follows the method. Fixed-DR uses #199e70 because the
# lighter #1baf7a has only 2.82:1 contrast on white. For the #199e70 / #D97706
# pair, the checked colour-vision distances were 9.0 (protan) and 7.4 (tritan);
# labels and distinct markers provide additional identification.
METHOD_COLOR = {"autodr": "#3B6FD4", "nodr": "#D97706", "fixeddr": "#199e70"}
METHOD_LABEL = {"autodr": "Auto-DR", "nodr": "No-DR", "fixeddr": "Fixed-DR"}
# a second encoding beside the colour, because
# No-DR and Fixed-DR have the same lightness in grey (L* 59,9 / 57,9).
METHOD_MARKER = {"autodr": "o", "nodr": "^", "fixeddr": "s"}
# the words of the figures, as the thesis
# text uses them (Kap. 4.3.2, Tab. 4-1, Tab. 4-13). One home; the other plot scripts import them.
TEST_MAIN = "Haupttest"
TEST_NOMINAL = "Test ohne Streuung"
YLABEL_RATE = "Erfolgsrate in %"
YLABEL_TRAINING_RATE = "Erfolgsrate im Training in %"
YLABEL_OPENING = "Öffnung der Grenzen in %"   # 0 = Nennwert, 100 = Maximum (T2)
POLICY_DOT_LABEL = "Punkt: Policy je Seed"
# The ablation pair of H9 / H11 / H12 (one home; plot_mlp_vs_lstm.py reads it):
# the reference arm keeps the Auto-DR blue, the ablation takes the reference
# palette's magenta (dark step) and a dashed line as second encoding. Pair
# validated with the dataviz validator, light mode: CVD dE 13.6, normal-vision
# dE 26.3, contrast >= 3:1 (2026-10-03).
ABLATION_COLOR = "#d55181"
# the two arms of the force-sensor test, (colour, line style, label).
FORCE_ARM_STYLE = {"autodr": (METHOD_COLOR["autodr"], "-", "Kraft in der Beobachtung"),
                   "autodr_nf": (ABLATION_COLOR, "--", "ohne Kraft (nur der Critic sieht sie)")}
# The pre-registered pair; every figure needs both. Fixed-DR is optional: it
# is drawn where its blocks exist (ARM_BLOCK), else the figure keeps two methods.
METHODS = ("autodr", "nodr")
ARMS = (*METHODS, "fixeddr")
# Fixed-DR plays the same tables in its own blocks.
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
# The five Fixed-DR training runs (RT-231): F7
# only, extracted when all five folders exist, refused when some do.
FIXED_RUNS = {("fixeddr", s): f"RT-231s{s}" for s in range(1, 6)}
# H5's return: the env's undiscounted reward sum per episode, mean over the
# episodes ending at one reset (insertion_env.py "Episode_Reward/total"); the
# rl_games observer averages it over the iteration. rl_games' own rewards/iter
# (last <=100 games) has the same level and twice the step-to-step noise.
RETURN_TAG = "Episode_Reward/total"
TRAINING_TAGS = ("Episode/success_rate", "dr/lat_r_hi", "dr/yaw_lo", "dr/yaw_hi", "dr/tilt_hi",
                 "dr/start_height_hi", "dr/start_height_floor", "dr/friction_lo", "dr/friction_hi",
                 RETURN_TAG, "dr/friction_mean")
# No-DR's training values of the F3 axes: env.yaml keys (RT-207s<N>_env.yaml).
NODR_ENV_KEYS = ("dr_mode", "start_lateral_offset", "fixture_yaw_rad", "fixture_yaw_noise_rad",
                 "fixture_tilt_rad", "fixture_tilt_noise_rad")

# F3 panels: episodes_all column, axis label, factor to the axis unit.
F3_PANELS = (
    # Labels for the disturbance dimensions.
    ("lat_r_mm", "Startversatz in mm", 1.0),
    # Thesis words (Tab. 4-1), 2026-10-11; two lines, in one the H2 panel titles ran into each other.
    ("yaw_abs_deg", "Betrag der Gierung\nder Aufnahme in °", 1.0),
    ("tilt_deg", "Kippung der\nAufnahme in °", 1.0),
    ("start_height_m", "Starthöhe in mm", 1000.0),
    ("friction", "Reibwert", 1.0),
)
# F4: the seven T-grenze keys (eval_table_boundary_seed10.json order).
K3_ORDER = ("lat_r_hi", "yaw_lo", "yaw_hi", "tilt_hi", "start_height_hi", "friction_lo", "friction_hi")
K3_LABEL = {"lat_r_hi": "Versatz\nmax", "yaw_lo": "Gier\nmin", "yaw_hi": "Gier\nmax",
            "tilt_hi": "Kippung\nmax", "start_height_hi": "Starthöhe\nmax",
            "friction_lo": "Reibung\nmin", "friction_hi": "Reibung\nmax"}
# Words as F3_PANELS; yaw is signed here, so no "Betrag".
# Thesis words (Tab. 4-1); plot_autodr_run.fig_stufen reads them too.
F8_PANELS = (("start_height_floor", "Starthöhe, untere Grenze\n(Startcurriculum)"), ("lat_r", "Startversatz"),
             ("yaw", "Gierung"), ("tilt", "Kippung"), ("start_height", "Starthöhe, obere Grenze"),
             ("friction", "Reibwert"))
FIG_W = 6.3   # inches, ~16 cm text width
PNG_DPI = 300   # print resolution; PDF and SVG are vector
# No timestamps in the files: two runs on the same input write the same bytes.
SAVE_META = {"pdf": {"CreationDate": None}, "svg": {"Date": None}, "png": None}
# The thesis body is Helvetica (thesis/includes/preamble.tex: helvet + \sfdefault);
# Arial has Helvetica-compatible metrics and must be installed.
FONT = "Arial"
# Legend, axis titles and tick numbers as large as the Word body text, Arial 12
# holds only while the PNG stands in Word at its own 16 cm width.
FONT_PT = 12


class FigError(RuntimeError):
    """The input is insufficient for the requested figure."""


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
    """Lower edge of bin i. Every F3/F9 table is equal-width at its bottom, so
    bin 0 is as wide as bin 1: lateral, tilt and |yaw| start at 0, friction at
    the band bottom 0.08."""
    return float(edges[i - 1]) if i > 0 else float(edges[0]) - (float(edges[1]) - float(edges[0]))


def _de_digits(s: str) -> str:
    """Decimal comma, as the thesis text sets it (thesis/includes/preamble.tex, siunitx)."""
    return s.replace(".", ",")


def _num(v: float, fmt: str) -> str:
    """A number for figure text: decimal comma and a real minus sign (U+2212)."""
    return _de_digits(format(v, fmt)).replace("-", "−")


def _yerr(rate: float, w) -> list:
    """Error-bar lengths below/above ``rate`` for its Wilson interval ``w``, in %.
    statsmodels returns the lower bound at k = 0 as ~2.2e-19, not 0 (RT-232:
    Fixed-DR s1/s2, 0 of 1000), and matplotlib refuses any negative length.
    Rounding is set to 0; a bound really beyond the rate is refused."""
    lo, hi = rate - w[0], w[1] - rate
    if min(lo, hi) < -1e-12:
        raise FigError(f"rate {rate} lies outside its Wilson interval {w}")
    return [[100 * max(0.0, lo)], [100 * max(0.0, hi)]]


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
    """METHODS, plus Fixed-DR when BOTH of its blocks hold runs.
    ``blocks``: the blocks Fixed-DR runs sit in. One block without the other
    is refused, never drawn half. Only the two -F blocks determine inclusion;
    Fixed-DR also plays T-auswahl, T-ood and (if selected) T-grenze."""
    blocks = set(blocks) & set(ARM_BLOCK.values())
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
                             "mean": float(np.mean(rates)), "ci": [float(v) for v in _boot_ci(rates, "F1")]}
    return out


def data_f2(summary: dict) -> list:
    rows = []
    for kname, block in (("K2", "T-nominal"), ("K1", "T-final")):
        k = summary[kname]
        if k.get("block") != block:
            raise FigError(f"F2: {kname} is not about {block}")
        rows.append({"block": block, "key": kname, "diff": k["diff"], "label": "Auto-DR − No-DR",
                     "welch": list(k["welch_ci"]), "boot": list(k["bootstrap_percentile_ci"])})
    # K5, only when the Fixed-DR blocks exist; eval_stats writes
    # {"missing": ...} otherwise, and an "error" is not drawn over.
    if arms_present(summary) == ARMS:
        for kname, block in (("K5_nominal", "T-nominal"), ("K5", "T-final")):
            k = summary[kname]
            if "diff" not in k or k.get("blocks", [None])[0] != block:
                raise FigError(f"F2: {kname} not computed for {block}: {k}")
            rows.append({"block": block, "key": kname, "diff": k["diff"], "label": "Auto-DR − Fixed-DR",
                         "welch": list(k["welch_ci"]), "boot": list(k["bootstrap_percentile_ci"])})
    return rows


def data_f3(summary: dict) -> dict:
    bins = summary["bins_T_final"]
    out = {}
    for col, _label, fac in F3_PANELS:
        if col not in next(iter(bins.values())):
            continue   # a summary.json from older runs holds no friction bins
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
            mean, std = _mean_std([seeds[k] for k in keys], "F3")
            lo, hi = _boot_ci([seeds[k] for k in keys], "H2")
            per[m] = {"seeds": seeds, "mean": mean.tolist(), "std": std.tolist(), "ci": [lo.tolist(), hi.tolist()]}
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
    pol = summary["final_policy"]
    return {"keys": list(K3_ORDER), "rates": [k3["blocks"][k]["rate"] for k in K3_ORDER],
            "wilson": [k3["blocks"][k]["wilson"] for k in K3_ORDER],
            "n": [k3["blocks"][k]["n"] for k in K3_ORDER],
            "overall": ov["rate"], "overall_wilson": ES.wilson(ov["k"], ov["n"]),
            # K3's overall rate is the final policy's own T-final run (
            # T-final-F for a Fixed-DR policy); the label names that block.
            "reference_block": ES.final_block(pol["method"]), "policy": pol}


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
    """T-final's rows, with Fixed-DR's from T-final-F (same table)."""
    return df[df["block"].isin(("T-final", ARM_BLOCK["T-final"])) & (df["tag"] == "")]


def _mean_std(stack, fig: str):
    """Mean and sample standard deviation (ddof=1) over
    axis 0 = the seeds. The figures draw the mean and seeds, not the SD; the SD stays for the printed numbers and the tables."""
    a = np.asarray(stack, dtype=float)
    if a.shape[0] < 2:
        raise FigError(f"{fig}: {a.shape[0]} seed; the spread band needs two or more")
    return a.mean(axis=0), a.std(axis=0, ddof=1)


def _boot_ci(stack, fig: str):
    """95 % percentile bootstrap interval of the mean over axis 0 = the seeds
    (H2, H5, H6). Resamples, seed and level from eval_stats (one home).
    Columns where every seed is equal get that value as both bounds."""
    from scipy import stats as st
    a = np.asarray(stack, dtype=float)
    if a.shape[0] < 2:
        raise FigError(f"{fig}: {a.shape[0]} seed; the interval needs two or more")
    ci = st.bootstrap((a,), np.mean, axis=0, vectorized=True, n_resamples=ES.BOOT_RESAMPLES,
                      confidence_level=1.0 - ES.ALPHA, method="percentile",
                      rng=np.random.default_rng(ES.BOOT_SEED)).confidence_interval
    return np.asarray(ci.low), np.asarray(ci.high)


# F6(a) evaluates every seed's ECDF on one shared force grid, so seeds with
# different success counts can be averaged point by point. Rounded to 6
# decimals so a grid point equals its decimal value exactly.
F6_FORCE_STEP_N = 0.05


def data_f6(df: pd.DataFrame) -> dict:
    fin = _final(df)
    out = {"ecdf": {}, "cum": {}, "limit": sorted(set(fin["force_limit_n"].astype(float)))}
    if len(out["limit"]) != 1:
        raise FigError(f"F6: more than one force limit in T-final: {out['limit']}")
    kmax = int(fin["steps"].max())
    ks = np.arange(1, kmax + 1)
    fmax = float(fin.loc[fin[KEY] == 1, "force_max_filtered_n"].max())
    grid = np.round(np.arange(0.0, math.ceil(fmax) + F6_FORCE_STEP_N / 2, F6_FORCE_STEP_N), 6)
    for m in arms_in(df):
        g = fin[fin["method"] == m]
        # Per seed: share of THAT seed's successes with peak force <= x.
        ok = g[g[KEY] == 1]
        ecdf, no_success = {}, []
        for s, gs in g.groupby("seed", sort=True):
            f = np.sort(ok.loc[ok["seed"] == s, "force_max_filtered_n"].to_numpy(dtype=float))
            if len(f) == 0:
                # A seed without one success has no force ECDF (RT-232: Fixed-DR
                # s1/s2, 0 of 1000). (a) averages the
                # seeds that have one and names the others; (b) keeps them all.
                no_success.append(int(s))
                continue
            ecdf[int(s)] = np.searchsorted(f, grid, side="right") / len(f)
        mean, std = _mean_std(list(ecdf.values()), "F6")
        out["ecdf"][m] = {"x": grid, "seeds": ecdf, "mean": mean, "std": std, "no_success": no_success,
                          "ci": _boot_ci(list(ecdf.values()), "F6")}
        seeds = {}
        for s, gs in g.groupby("seed", sort=True):
            st = gs.loc[gs[KEY] == 1, "steps"].to_numpy(dtype=int)
            seeds[int(s)] = np.array([(st <= k).sum() for k in ks]) / len(gs)
        mean, std = _mean_std(list(seeds.values()), "F6")
        out["cum"][m] = {"k": ks, "seeds": seeds, "mean": mean, "std": std, "ci": _boot_ci(list(seeds.values()), "F6")}
    return out


def _seed_curves(curves: pd.DataFrame, col: str, fig: str) -> dict:
    """One ``curves.csv`` column per arm: the seeds' curves, their mean and
    sample std (F7, H5, H6). ``x`` = total env steps, ``it`` = the
    iteration; the seeds of one arm must share one grid, arms may differ (each
    run its own factor). A seed with an empty cell is refused, never averaged."""
    out = {}
    for m in ARMS:
        g = curves[curves["method"] == m]
        xs = None
        seeds = {}
        for s, gs in g.groupby("seed", sort=True):
            gs = gs.sort_values("env_steps")
            x = gs["env_steps"].to_numpy(dtype=np.int64)
            if xs is None:
                xs, its = x, gs["iteration"].to_numpy(dtype=np.int64)
            elif not np.array_equal(x, xs):
                raise FigError(f"{fig}: {m} s{s} logs other env steps than the first seed")
            v = gs[col].to_numpy(dtype=float)
            if not np.isfinite(v).all():
                raise FigError(f"{fig}: {m} s{s} has {int((~np.isfinite(v)).sum())} empty {col} cells")
            seeds[int(s)] = v
        if not seeds and m == "fixeddr":
            continue   # optional arm: extract_training took all five runs or none
        if not seeds:
            raise FigError(f"{fig}: no {m} curves")
        mean, std = _mean_std(list(seeds.values()), fig)
        out[m] = {"x": xs, "it": its, "seeds": seeds, "mean": mean, "std": std,
                  "ci": _boot_ci(list(seeds.values()), fig)}
    return out


def data_f7(curves: pd.DataFrame) -> dict:
    """Training success rate per arm (F7 over env steps, H6 over the iteration)."""
    return _seed_curves(curves, "Episode/success_rate", "F7")


def data_h5(curves: pd.DataFrame) -> dict:
    """Training return per arm (H5); a return has no 0..1 bound, so no clipping."""
    return _seed_curves(curves, RETURN_TAG, "H5")


def iteration_factor(curves: pd.DataFrame) -> int:
    """Env steps per iteration, ONE value over every run, or FigError. H5/H6
    draw over the iteration; that axis is fair only when every run
    collects the same samples per iteration.
    rl_games writes ``env_steps = epoch x horizon_length x num_envs`` with the
    epoch from 1; rsl_rl's ``(it + 1) x F`` from it 0 is refused."""
    it = curves["iteration"].to_numpy(dtype=np.int64)
    st = curves["env_steps"].to_numpy(dtype=np.int64)
    if len(it) == 0 or (it < 1).any() or (st % np.maximum(it, 1)).any():
        raise FigError("H5/H6: env_steps is not iteration x one factor (rsl_rl counts (it + 1) x F); "
                       "draw over env steps")
    factors = sorted(set((st // it).tolist()))
    if len(factors) != 1:
        raise FigError(f"H5/H6: the runs collect {factors} env steps per iteration; "
                       f"an iteration axis would compare unequal samples")
    return int(factors[0])


def data_f8(curves: pd.DataFrame, sidecars: dict) -> dict:
    """Same normalisation as plot_autodr_run.fig_stufen: (v - centre)/(max - centre)
    per side; the floor (v - f0)/(top - f0). Read from each run's sidecar."""
    out = {}
    g = curves[curves["method"] == "autodr"]
    for s, gs in g.groupby("seed", sort=True):
        side = sidecars.get(("autodr", int(s)))
        if side is None:
            raise FigError(f"F8: no AutoDR state for autodr s{s}")
        ad = side["autodr"]
        dims = {d["name"]: d for d in ad["dims"]}
        fl = ad["floor"]
        gs = gs.sort_values("env_steps")
        x = gs["env_steps"].to_numpy(dtype=np.int64)
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
        out[int(s)] = {"x": x, "panels": runs}
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
    out = {"tau": taus, "t1": t1_mm, "methods": {}, "ci": {}}
    for m in arms_in(df):
        # Per seed first, then the mean over the seeds and its interval:
        # the policies are not pooled into one sample.
        seeds = []
        for _s, g in fin[fin["method"] == m].groupby("seed", sort=True):
            ok = g[KEY].to_numpy(dtype=int) == 1
            ip = g["ip_max_mm"].to_numpy(dtype=float)
            seeds.append(np.array([(ok & (ip < t)).sum() for t in taus]) / len(g))
        out["methods"][m] = np.mean(seeds, axis=0)
        out["ci"][m] = _boot_ci(seeds, "F10")
        out.setdefault("seeds", {})[m] = seeds   # drawn as thin lines
    return out


# F11/F12: per dimension the scale into the tick unit, the unit, decimals.
F11_UNIT = {"lat_r": (1000.0, " mm", 0), "yaw": (180.0 / math.pi, "°", 1), "tilt": (180.0 / math.pi, "°", 0),
            "start_height": (1000.0, " mm", 0), "friction": (1.0, "", 2)}


def data_f11(summary: dict, sidecar: dict) -> dict:
    """The final policy's T-grenze block (x1.0) beside its T-ood block (the
    sidecar's factor per block). The x1.0 value is read back from the OOD
    sidecar -- centre + (physical - centre) / factor -- never typed. Both rates
    are judged against the policy's own T-final run (eval_stats.final_block);
    K3's and the OOD reference must be the same count, else one line cannot
    serve both. Ported from p6-TestAusserhalb 7d008ff1/db85986c; the OOD
    numbers sit in the same summary.json here."""
    k3 = summary["K3"]
    if "blocks" not in k3:
        raise FigError(f"F11: K3 not computed: {k3}")
    pol = summary["final_policy"]
    o = summary.get("OOD", {}).get("policies", {}).get(f"{pol['method']}_s{pol['seed']}")
    if o is None:
        raise FigError(f"F11: final policy {pol} has no T-ood run: {sorted(summary.get('OOD', {}).get('policies', {}))}")
    ov, ref = k3["overall"], o["reference"]
    if (ov["k"], ov["n"]) != (ref["k"], ref["n"]):
        raise FigError(f"F11: the two T-final references differ: {ov} vs {ref}")
    side = {b["key"]: b for b in sidecar["blocks"]}
    for name, keys in (("K3", k3["blocks"]), ("OOD", o["blocks"]), ("sidecar", side)):
        if set(keys) != set(K3_ORDER):
            raise FigError(f"F11: {name} keys {sorted(keys)} != {sorted(K3_ORDER)}")
    out = {"keys": list(K3_ORDER), "edge": [], "physical": [], "labels": [], "grenze": [], "ood": [],
           "factor": sidecar["factor"], "overall": ref["rate"], "overall_wilson": ES.wilson(ref["k"], ref["n"]),
           "reference_block": ref["block"], "policy": pol}
    for key in K3_ORDER:
        b = side[key]
        dim = key.rsplit("_", 1)[0]
        c = sidecar["centres"][dim]
        edge = c + (b["physical"] - c) / b["factor"]
        scale, unit, nd = F11_UNIT[dim]
        # A block whose factor is not the table's (start height x1.0) says so on its own line.
        cap = "" if b["factor"] == sidecar["factor"] else f"\n(×{_num(b['factor'], '.1f')})"
        out["edge"].append(edge)
        out["physical"].append(b["physical"])
        out.setdefault("factors", []).append(b["factor"])
        out["labels"].append(f"{K3_LABEL[key]}\n{_num(edge * scale, f'.{nd}f')} → "
                             f"{_num(b['physical'] * scale, f'.{nd}f')}{unit}{cap}")
        out["grenze"].append({"rate": k3["blocks"][key]["rate"], "wilson": k3["blocks"][key]["wilson"]})
        out["ood"].append({"rate": o["blocks"][key]["rate"], "wilson": o["blocks"][key]["wilson"]})
    return out


def data_f12(summary: dict, sidecar: dict) -> dict:
    """Every seed's T-ood block against ITS OWN T-final:
    the difference with its Newcombe interval, one dot per seed (F1's dot plot,
    Weissgerber 2015), and the mean of the seed differences as a short line.
    One group per arm that played T-ood (AutoDR and Fixed-DR). All
    numbers are summary.json's `OOD` (eval_stats), none recomputed."""
    o = summary.get("OOD", {})
    across = o.get("across_seeds", {})
    arms = [m for m in ARMS if m in across]
    if not arms or set(across) - set(ARMS):
        raise FigError(f"F12: T-ood methods {sorted(across)}, want some of {list(ARMS)}")
    side = {b["key"]: b for b in sidecar["blocks"]}
    if set(side) != set(K3_ORDER):
        raise FigError(f"F12: sidecar keys {sorted(side)} != {sorted(K3_ORDER)}")
    out = {"keys": list(K3_ORDER), "labels": [], "arms": {}}
    for key in K3_ORDER:
        b = side[key]
        scale, unit, nd = F11_UNIT[key.rsplit("_", 1)[0]]
        out["labels"].append(f"{K3_LABEL[key]}\n{_num(b['physical'] * scale, f'.{nd}f')}{unit}\n"
                             f"(×{_num(b['factor'], '.1f')})")
    for m in arms:
        a = across[m]
        pols = [o["policies"][f"{m}_s{s}"] for s in a["seeds"]]
        for name, keys in [("across", a["blocks"])] + [(f"{m} s{p['policy']['seed']}", p["blocks"]) for p in pols]:
            if set(keys) != set(K3_ORDER):
                raise FigError(f"F12: {name} keys {sorted(keys)} != {sorted(K3_ORDER)}")
        out["arms"][m] = {"seeds": list(a["seeds"]),
                          "diffs": [[p["blocks"][key]["diff_vs_final"] for p in pols] for key in K3_ORDER],
                          "cis": [[p["blocks"][key]["newcombe_ci"] for p in pols] for key in K3_ORDER],
                          "mean": [a["blocks"][key]["diff_mean"] for key in K3_ORDER]}
    return out


# H18 (thesis 5.2.2): the T-ood blocks outside the Prüfbereich.
# The start-height block stays at factor 1.0, so it is not
# outside and is left out. Words as F3_PANELS.
H18_KEYS = tuple(k for k in K3_ORDER if k != "start_height_hi")
# "Start-/versatz" in two lines: at 12 pt it ran into its neighbours under seven bars.
H18_WORD = {"lat_r": "Start-\nversatz", "yaw": "Gierung", "tilt": "Kippung", "friction": "Reibwert"}
H18_ARMS = ("autodr", "fixeddr")


def data_h18(summary: dict, sidecar: dict) -> dict:
    """H18: per T-ood block outside the Prüfbereich (H18_KEYS) each seed's
    ABSOLUTE success rate, the mean over the seeds and its 95 % percentile
    bootstrap interval, for Auto-DR and Fixed-DR; per arm the mean of
    its seeds' own main-test rate (the OOD reference, T-final / T-final-F). Rates are summary.json's `OOD` (eval_stats), none recomputed.
    Refused: a block not at factor 1.5, an arm or seed missing, unequal
    test-case counts (the figure names ONE count)."""
    o = summary.get("OOD", {})
    across = o.get("across_seeds", {})
    if set(H18_ARMS) - set(across):
        raise FigError(f"H18: T-ood methods {sorted(across)}, want {list(H18_ARMS)}")
    side = {b["key"]: b for b in sidecar["blocks"]}
    if set(side) != set(K3_ORDER):
        raise FigError(f"H18: sidecar keys {sorted(side)} != {sorted(K3_ORDER)}")
    out = {"keys": list(H18_KEYS), "labels": [], "arms": {}}
    for key in H18_KEYS:
        b = side[key]
        if b["factor"] != 1.5:
            raise FigError(f"H18: block {key} has factor {b['factor']}, not 1.5")
        dim = key.rsplit("_", 1)[0]
        scale, unit, nd = F11_UNIT[dim]
        v = b["physical"] * scale
        sign = "+" if dim == "yaw" and v > 0 else ""
        out["labels"].append(f"{H18_WORD[dim]}\n{sign}{_num(v, f'.{nd}f')}{unit}")
    ns = set()
    for m in H18_ARMS:
        a = across[m]
        pols = [o["policies"][f"{m}_s{s}"] for s in a["seeds"]]
        for p in pols:
            if set(H18_KEYS) - set(p["blocks"]):
                raise FigError(f"H18: {m} s{p['policy']['seed']} lacks blocks {sorted(set(H18_KEYS) - set(p['blocks']))}")
            ns |= {p["blocks"][k]["n"] for k in H18_KEYS}
        rates = np.array([[p["blocks"][k]["rate"] for k in H18_KEYS] for p in pols], dtype=float)
        lo, hi = _boot_ci(rates, "H18")
        wilson = np.array([[p["blocks"][k]["wilson"] for k in H18_KEYS] for p in pols], dtype=float)
        out["arms"][m] = {"seeds": list(a["seeds"]), "rates": rates, "wilson": wilson, "mean": rates.mean(axis=0),
                          "ci": (lo, hi), "ref_mean": float(np.mean([p["reference"]["rate"] for p in pols])),
                          "ref_block": sorted({p["reference"]["block"] for p in pols})}
    if len(ns) != 1:
        raise FigError(f"H18: test cases per policy and block differ: {sorted(ns)}")
    out["n"] = ns.pop()
    return out


# H19/H20: T-ood and the doubled observation errors, only the policies
# that learned the task. The rule reads each policy's OWN main test, never the
# block shown. Arms with fewer than five such policies
# get no mean and no interval (2-4 runs individually).
LEARNED_MIN = 0.90
LEARNED_MEAN_MIN_N = 5
H20_WORD = {"pn2": "Lagefehler\nder Aufnahme", "gx2": "Greifversatz", "py2": "Gierfehler\nder Aufnahme"}
# Rate axis (lower end, tick step) of thesis 5-9 (H19) and 5-11 (H20): an
# exception to the usual zero baseline for bars; every other bar figure keeps (0, 20).
PAIR_YAXIS = {"H19": (50, 10), "H20": (80, 5)}


def data_learned(policies: dict, keys, labels, fig: str, min_rate: float = LEARNED_MIN) -> dict:
    """H19/H20: per arm (H18_ARMS) the policies whose own main-test rate
    (``reference``: T-final, Fixed-DR T-final-F) is >= min_rate (LEARNED_MIN), sorted by
    seed; per block their rate and Wilson interval as summary.json holds them;
    the mean over the seeds and its bootstrap interval only for an arm
    with >= LEARNED_MEAN_MIN_N such policies. ``left_out`` names the others with
    their main-test rate, ``ref_range`` spans the shown policies' main tests.
    Refused: an arm missing or without a learned policy, a policy without a
    block, unequal test-case counts."""
    per = {}
    for p in policies.values():
        per.setdefault(p["policy"]["method"], []).append(p)
    if set(H18_ARMS) - set(per):
        raise FigError(f"{fig}: methods {sorted(per)}, want {list(H18_ARMS)}")
    out = {"keys": list(keys), "labels": list(labels), "arms": {}, "left_out": {}}
    ns, refs = set(), []
    for m in H18_ARMS:
        ps = sorted(per[m], key=lambda q: q["policy"]["seed"])
        keep = [p for p in ps if p["reference"]["rate"] >= min_rate]
        out["left_out"][m] = [(p["policy"]["seed"], p["reference"]["rate"]) for p in ps if p not in keep]
        if not keep:
            raise FigError(f"{fig}: no {m} policy reaches {min_rate} in its own main test")
        for p in keep:
            if set(keys) - set(p["blocks"]):
                raise FigError(f"{fig}: {m} s{p['policy']['seed']} lacks blocks {sorted(set(keys) - set(p['blocks']))}")
            ns |= {p["blocks"][k]["n"] for k in keys}
        rates = np.array([[p["blocks"][k]["rate"] for k in keys] for p in keep], dtype=float)
        a = {"seeds": [p["policy"]["seed"] for p in keep], "rates": rates,
             "wilson": np.array([[p["blocks"][k]["wilson"] for k in keys] for p in keep], dtype=float),
             "ref": [p["reference"]["rate"] for p in keep]}
        if len(keep) >= LEARNED_MEAN_MIN_N:
            a["mean"] = rates.mean(axis=0)
            a["ci"] = _boot_ci(rates, fig)
        out["arms"][m] = a
        refs += a["ref"]
    if len(ns) != 1:
        raise FigError(f"{fig}: test cases per policy and block differ: {sorted(ns)}")
    out["n"] = ns.pop()
    out["ref_range"] = (min(refs), max(refs))
    return out


def data_h19(summary: dict, sidecar: dict) -> dict:
    """H19: H18's six T-ood blocks and labels (data_h18 checks factor 1.5 and
    the keys), only the learned policies (data_learned)."""
    d18 = data_h18(summary, sidecar)
    return data_learned(summary["OOD"]["policies"], d18["keys"], d18["labels"], "H19")


def _noise_blocks(summary: dict, fig: str):
    """summary.json's NOISE policies, the three keys (H7_NOISE order) and their
    tick labels with the doubled value from the block's ``changed`` entry
    (m -> mm, rad -> deg); a block whose value differs between policies is refused."""
    nz = summary.get("NOISE")
    if not isinstance(nz, dict) or "policies" not in nz:
        raise FigError(f"{fig}: summary.json has no NOISE result: {nz}")
    keys = [t for t, _l in H7_NOISE]
    labels = []
    for t in keys:
        vals = {tuple(sorted(p["blocks"][t]["changed"].items())) for p in nz["policies"].values() if t in p["blocks"]}
        if len(vals) != 1 or len(next(iter(vals))) != 1:
            raise FigError(f"{fig}: block {t} changes {sorted(vals)}, want one value")
        (name, v), = next(iter(vals))
        txt = f"{_num(1000 * v, '.0f')} mm" if name.endswith("_m") else f"{_num(np.degrees(v), '.2f')}°"
        labels.append(f"{H20_WORD[t]}\n±{txt}")
    return nz["policies"], keys, labels


def data_h20(summary: dict) -> dict:
    """H20: the three noise blocks, only the learned policies (data_learned)."""
    return data_learned(*_noise_blocks(summary, "H20"), "H20")


def data_h22(summary: dict) -> dict:
    """H22 (appendix to H20): the three noise blocks for every policy
    (data_learned without a rule), in draw_h18's form: per arm the mean of its
    policies' own main test as ``ref_mean``. An arm without a mean (fewer than
    LEARNED_MEAN_MIN_N policies) is refused."""
    d = data_learned(*_noise_blocks(summary, "H22"), "H22", min_rate=0.0)
    for m, a in d["arms"].items():
        if "mean" not in a:
            raise FigError(f"H22: {m} has {len(a['seeds'])} policies, a mean needs {LEARNED_MEAN_MIN_N}")
        a["ref_mean"] = float(np.mean(a["ref"]))
    return d


def _add_pairs(d: dict, policies: dict, fig: str) -> dict:
    """The pair form (H18, H19, H20, H22): per shown policy the Wilson interval of its OWN main
    test from summary.json's k and n (``ref_wilson``), and for an arm with a mean
    the mean of those main tests with the same bootstrap as the block mean
    (``ref_mean``, ``ref_ci``); ``n_ref`` = the main test's test cases."""
    n_ref = set()
    for m, a in d["arms"].items():
        refs = [policies[f"{m}_s{s}"]["reference"] for s in a["seeds"]]
        n_ref |= {r["n"] for r in refs}
        a["ref_wilson"] = np.array([ES.wilson(r["k"], r["n"]) for r in refs], dtype=float)
        if "mean" in a:
            a["ref_mean"] = float(np.mean(a["ref"]))
            lo, hi = _boot_ci(np.asarray(a["ref"], dtype=float)[:, None], fig)
            a["ref_ci"] = (float(lo[0]), float(hi[0]))
    if len(n_ref) != 1:
        raise FigError(f"{fig}: main-test sizes differ: {sorted(n_ref)}")
    d["n_ref"] = n_ref.pop()
    return d


def data_h20_pairs(summary: dict) -> dict:
    """H20 as pairs: data_h20 plus each shown policy's own main test (_add_pairs).
    Every block replays the main test's 1000 rows, so each pair meets the same rows."""
    return _add_pairs(data_h20(summary), summary["NOISE"]["policies"], "H20")


def data_h19_pairs(summary: dict, sidecar: dict) -> dict:
    """H19 in H20's pair form: the learned policies on the T-ood blocks."""
    return _add_pairs(data_h19(summary, sidecar), summary["OOD"]["policies"], "H19")


def data_h18_pairs(summary: dict, sidecar: dict) -> dict:
    """H18 (appendix to H19) in the pair form: every policy, no rule (min_rate 0)."""
    d18 = data_h18(summary, sidecar)
    return _add_pairs(data_learned(summary["OOD"]["policies"], d18["keys"], d18["labels"], "H18", min_rate=0.0),
                      summary["OOD"]["policies"], "H18")


def data_h22_pairs(summary: dict) -> dict:
    """H22 (appendix to H20) in the pair form: every policy (data_h22's checks)."""
    return _add_pairs(data_h22(summary), summary["NOISE"]["policies"], "H22")


def data_h4(summary: dict) -> list:
    """F5's outcome counts pooled over the seeds of each arm: the main-text
    form, one bar per method (seeds x rows episodes each)."""
    rows = data_f5(summary)
    out = []
    for m in [m for m in ARMS if any(r["method"] == m for r in rows)]:
        rs = [r for r in rows if r["method"] == m]
        out.append({"method": m, "seeds": len(rs), "n": sum(r["n"] for r in rs),
                    "counts": [sum(r["counts"][i] for r in rs) for i in range(len(OUTCOMES))]})
    return out


# H7: the three observation-noise blocks (check_eval_conditions.NOISE_TAGS, each
# x2 of one training value), in figure order, with the words of the figure.
H7_NOISE = (("pn2", "Lage\nTasche"), ("gx2", "Greif-\nversatz"), ("py2", "Gier\nTasche"))
# H8 draws the best policy of the arms that learn the task. No-DR's best (RT-246:
# 163 of 1000 on T-auswahl) succeeds mostly on the easy rows, so its successes'
# force is a selected subset, not comparable.
H8_ARMS = ("autodr", "fixeddr")


def best_policies(summary: dict) -> dict:
    """{arm: seed}: the pre-registered selection rule (eval_stats.select_final: highest T-auswahl rate, then smaller force p95,
    then smaller seed) applied to ONE arm's runs. A supplementary view:
    the best of five is another algorithm than the method (Agarwal et al. 2021), so it never replaces the all-seed comparison (H1, H2)."""
    out = {}
    for m in ARMS:
        mine = [p for p in summary["policies"] if p["method"] == m]
        if any(p["block"] == "T-auswahl" for p in mine):
            out[m] = int(ES.select_final(mine)["seed"])
    return out


def data_h7(summary: dict, sidecar: dict) -> dict:
    """Each arm's best policy (best_policies) on the noise blocks and the T-ood
    blocks, with its own T-final rate as reference; all rates from
    summary.json's NOISE / OOD (eval_stats), none recomputed. An arm whose best
    policy played neither is left out and named (No-DR: no such runs)."""
    best = best_policies(summary)
    noise = summary.get("NOISE", {}).get("policies", {})
    ood = summary.get("OOD", {}).get("policies", {})
    side = {b["key"]: b for b in sidecar["blocks"]}
    if set(side) != set(K3_ORDER):
        raise FigError(f"H7: sidecar keys {sorted(side)} != {sorted(K3_ORDER)}")
    out = {"arms": {}, "left_out": {}, "noise_labels": [lab for _t, lab in H7_NOISE], "ood_labels": [],
           "factor": sidecar["factor"]}
    for key in K3_ORDER:
        f = side[key]["factor"]
        out["ood_labels"].append(K3_LABEL[key] + ("" if f == sidecar["factor"] else "\n(nicht erweitert)"))
    for m, seed in best.items():
        pkey = f"{m}_s{seed}"
        if pkey not in noise and pkey not in ood:
            out["left_out"][m] = seed
            continue
        if pkey not in noise or pkey not in ood:
            raise FigError(f"H7: {pkey} played {'T-ood' if pkey in ood else 'the noise blocks'} only")
        nz, oo = noise[pkey], ood[pkey]
        if (nz["reference"]["k"], nz["reference"]["n"]) != (oo["reference"]["k"], oo["reference"]["n"]):
            raise FigError(f"H7: {pkey} has two T-final references: {nz['reference']} vs {oo['reference']}")
        for name, keys, want in (("noise", nz["blocks"], [t for t, _l in H7_NOISE]), ("OOD", oo["blocks"], K3_ORDER)):
            if set(keys) != set(want):
                raise FigError(f"H7: {pkey} {name} blocks {sorted(keys)} != {sorted(want)}")
        out["arms"][m] = {"seed": seed, "reference": nz["reference"]["rate"], "reference_n": nz["reference"]["n"],
                          "reference_wilson": ES.wilson(nz["reference"]["k"], nz["reference"]["n"]),
                          "noise": [nz["blocks"][t]["rate"] for t, _l in H7_NOISE],
                          "noise_n": [nz["blocks"][t]["n"] for t, _l in H7_NOISE],
                          "noise_wilson": [nz["blocks"][t]["wilson"] for t, _l in H7_NOISE],
                          "ood": [oo["blocks"][k]["rate"] for k in K3_ORDER],
                          "ood_n": [oo["blocks"][k]["n"] for k in K3_ORDER],
                          "ood_wilson": [oo["blocks"][k]["wilson"] for k in K3_ORDER]}
    if not out["arms"]:
        raise FigError(f"H7: no best policy played the noise and T-ood blocks: {best}")
    return out


def data_h8(df: pd.DataFrame, policies: dict) -> dict:
    """Peak filtered contact force of the SUCCESSFUL trials of each given
    policy ({arm: seed}) on T-final (Fixed-DR: T-final-F), per bin of each
    disturbance: median and quartiles by the env's nearest-rank rule
    (eval_stats.pct_nearest_rank, so a figure number and a log number mean the
    same). Bins without table rows are dropped (as F3); a kept bin without a
    success is refused, never drawn as 0."""
    fin = _final(df)
    lim = sorted(set(fin["force_limit_n"].astype(float)))
    if len(lim) != 1:
        raise FigError(f"H8: more than one force limit in T-final: {lim}")
    groups = {}
    for m, seed in policies.items():
        g = fin[(fin["method"] == m) & (fin["seed"] == seed)]
        if g.empty:
            raise FigError(f"H8: no T-final run of {m} s{seed}")
        groups[m] = g
    out = {"limit": lim[0], "policies": dict(policies), "panels": {}}
    for col, _label, fac in F3_PANELS:
        if col not in fin.columns:
            continue
        edges = ES.BIN_EDGES[col]
        rows = next(iter(groups.values()))
        keep = [i for i in range(len(edges)) if (bin_index(rows[col], edges) == i).any()]
        per = {}
        for m, g in groups.items():
            won = g[g[KEY] == 1]
            idx = bin_index(won[col], edges)
            f = won["force_max_filtered_n"].to_numpy(dtype=float)
            q = {"median": [], "q25": [], "q75": [], "n": []}
            for i in keep:
                v = f[idx == i]
                if len(v) == 0:
                    raise FigError(f"H8: {m} s{policies[m]} has no success in {col} bin {i}")
                q["median"].append(ES.pct_nearest_rank(v, 0.5))
                q["q25"].append(ES.pct_nearest_rank(v, 0.25))
                q["q75"].append(ES.pct_nearest_rank(v, 0.75))
                q["n"].append(int(len(v)))
            per[m] = q
        out["panels"][col] = {"x": [fac * 0.5 * (bin_lower(edges, i) + edges[i]) for i in keep],
                              "lower": [fac * bin_lower(edges, i) for i in keep],
                              "upper": [fac * float(edges[i]) for i in keep], "methods": per}
    return out


# H12's two conditions, the FORCE block labels of eval_stats.
H12_CONDITION = (("T-final", "Lagefehler wie im Training"), ("pn2", "Lagefehler ×2"))


def data_h12(summary: dict) -> dict:
    """H12: per force arm and block the per-seed rates and the arm mean,
    read from summary.json ``FORCE`` (eval_stats.force_blocks); only the
    bootstrap interval of the mean is computed here. A summary without
    ``FORCE``, or a block it marks missing, is refused."""
    fo = summary.get("FORCE")
    if not isinstance(fo, dict) or "blocks" not in fo:
        raise FigError(f"H12: summary.json has no FORCE result: {fo}")
    out = {}
    for label, _name in H12_CONDITION:
        b = fo["blocks"].get(label, {})
        if "per_seed" not in b:
            raise FigError(f"H12: FORCE block {label} not computed: {b}")
        for arm in ES.FORCE_ARMS:
            ps = b["per_seed"][arm]
            rates = [p["rate"] for p in ps]
            out.setdefault(arm, {})[label] = {"seeds": [p["seed"] for p in ps], "rates": rates,
                                              "mean": b[f"mean_{arm}"],
                                              "ci": [float(v) for v in _boot_ci(rates, "H12")]}
    return out


# H13: the doubled observation errors for every
# policy that played them, in H1's form: per arm each seed's own untagged main
# test (hollow; Fixed-DR: T-final-F) once, then the blocks (filled).
# Replaces the thesis' hand-made figure 5-7 (BA_Anpassungen_Statistik.md A8).


def data_h13(summary: dict) -> dict:
    """H13: per arm and noise block (H7_NOISE) the per-seed rates beside the
    rates of each seed's own untagged main test, read from summary.json
    ``NOISE`` (eval_stats.noise_blocks); only the mean and its bootstrap
    interval are computed here. A summary without ``NOISE``, or a
    policy without one of the three blocks, is refused."""
    nz = summary.get("NOISE")
    if not isinstance(nz, dict) or "policies" not in nz:
        raise FigError(f"H13: summary.json has no NOISE result: {nz}")
    per = {}
    for p in nz["policies"].values():
        per.setdefault(p["policy"]["method"], []).append(p)
    out = {}
    for m in [a for a in ARMS if a in per]:
        ps = sorted(per[m], key=lambda p: p["policy"]["seed"])
        missing = [(p["policy"]["seed"], t) for p in ps for t, _l in H7_NOISE if t not in p["blocks"]]
        if missing:
            raise FigError(f"H13: {m} lacks noise blocks (seed, tag): {missing}")
        seeds = [p["policy"]["seed"] for p in ps]

        def one(rates):
            return {"seeds": seeds, "rates": rates, "mean": float(np.mean(rates)),
                    "ci": [float(v) for v in _boot_ci(rates, "H13")]}

        out[m] = {"ref": one([p["reference"]["rate"] for p in ps]),
                  **{t: one([p["blocks"][t]["rate"] for p in ps]) for t, _l in H7_NOISE}}
    return out


# H14 (RT-264/265): the robustness curve -- every final-study policy over the
# tilt sweep and the friction sweep (8 steps each, the same 125 rows in every
# step, only the swept quantity nailed; own root results/eval_rlgames_sweep/).
# F7's curve form over the steps; the test range (centre to the
# factor-1.0 steps of the sweep sidecar) shaded. Panel: block, column, unit of
# the sidecar's physical value -> axis unit, axis label.
H14_PANELS = (("T-sweep-tilt", "tilt_deg", math.degrees, "Kippung in °"),
              ("T-sweep-fric", "friction", float, "Reibwert"))
H14_RANGE_LABEL = "grauer Bereich = Prüfbereich"


def data_h14(df: pd.DataFrame, sidecars: dict) -> dict:
    """H14: per sweep panel the steps sorted by the value read back from the
    rows (one value per step, 3 decimals: the applied value carries ~1e-6
    float noise), the test range (sidecar centre to the factor-1.0 steps) and
    per arm each seed's success rate (``success_train``) per step, their mean
    and its pointwise bootstrap interval. Refused: a step set other than the
    sidecar's, a step with two values, a seed missing a step."""
    out = {}
    for block, col, unit, xlabel in H14_PANELS:
        g = df[(df["block"] == block) & (df["tag"] == "")]
        side = sidecars[block]
        steps = {b["key"]: b for b in side["blocks"]}
        if set(g["boundary_key"]) != set(steps):
            raise FigError(f"H14: {block} steps {sorted(set(g['boundary_key']))} != sidecar {sorted(steps)}")
        xs = {}
        for k, gk in g.groupby("boundary_key"):
            v = set(gk[col].astype(float).round(3))
            if len(v) != 1:
                raise FigError(f"H14: {block} step {k} holds {len(v)} values of {col}: {sorted(v)[:4]}")
            xs[k] = v.pop()
        order = sorted(xs, key=xs.get)
        edges = [xs[k] for k in order if steps[k]["factor"] == 1.0] + [unit(side["centres"][side["dim"]])]
        arms = {}
        for m in [a for a in ARMS if a in set(g["method"])]:
            gm = g[g["method"] == m]
            rate = gm.groupby(["seed", "boundary_key"])["success_train"].mean().unstack("boundary_key")
            if rate.isna().any().any() or list(rate.columns.sort_values()) != sorted(order):
                raise FigError(f"H14: {block} {m}: a seed lacks a step")
            seeds = {int(s): rate.loc[s, order].to_numpy(dtype=float) for s in sorted(rate.index)}
            stack = list(seeds.values())
            arms[m] = {"seeds": seeds, "mean": np.mean(stack, axis=0), "ci": _boot_ci(stack, "H14")}
        out[block] = {"x": [xs[k] for k in order], "keys": order, "range": (min(edges), max(edges)),
                      "xlabel": xlabel, "n": int(g.groupby(["method", "seed", "boundary_key"]).size().min()),
                      "arms": arms}
    return out


# H15: epochs per point. The share of the successes needs successes in the point:
# Fixed-DR s5 has about 10 per epoch, so one epoch is too thin a point; ten epochs
# pool 600-6000 episodes per seed and point (prototype 2026-10-07).
H15_EPOCHS = 10


def data_h15(pen: pd.DataFrame, block: int = H15_EPOCHS) -> dict:
    """Per arm and seed, per ``block`` epochs (pooled episodes of THAT seed only,
    never across seeds): ``clean`` = successes below t1 / episodes, ``share_ge`` =
    successes at or above t1 / successes (NaN where the block has no success).
    ``x`` = the block's last epoch. Every seed must cover the same epochs.
    Per arm then the mean over the seeds and its bootstrap interval:
    ``clean`` over every seed; ``share`` over the seeds with a success at EVERY
    point (as F6: a seed without one has no share), the others in ``left_out``."""
    t1 = sorted(set(pen["t1_mm"].astype(float)))
    if len(t1) != 1:
        raise FigError(f"H15: the runs log different t1 {t1}")
    out, grid = {"t1": t1[0], "block": block, "arms": {}}, None
    for m in ARMS:
        g = pen[pen["method"] == m]
        if g.empty:
            continue
        seeds = {}
        for s, gs in g.groupby("seed", sort=True):
            if grid is None:
                grid = sorted(gs["epoch"])
            elif sorted(gs["epoch"]) != grid:
                raise FigError(f"H15: {m} s{s} covers other epochs than the first run")
            b = gs.assign(b=(gs["epoch"] - 1) // block).groupby("b").agg(
                x=("epoch", "max"), n=("episodes", "sum"), k=("successes", "sum"), kg=("successes_ge_t1", "sum"))
            n, k, kg = (b[c].to_numpy(dtype=float) for c in ("n", "k", "kg"))
            seeds[int(s)] = {"x": b["x"].to_numpy(dtype=int), "clean": (k - kg) / n, "successes": k,
                             "share_ge": np.divide(kg, k, out=np.full_like(k, np.nan), where=k > 0)}
        clean = [v["clean"] for v in seeds.values()]
        used = [s for s, v in seeds.items() if np.isfinite(v["share_ge"]).all()]
        share = [seeds[s]["share_ge"] for s in used]
        out["arms"][m] = {"x": next(iter(seeds.values()))["x"], "seeds": seeds,
                          "clean": {"mean": np.mean(clean, axis=0), "ci": _boot_ci(clean, "H15"), "n": len(clean)},
                          "share": {"mean": np.mean(share, axis=0), "ci": _boot_ci(share, "H15"), "n": len(used),
                                    "left_out": [s for s in seeds if s not in used]}}
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


def _log_factor(d: pathlib.Path, run: str) -> tuple:
    """(num_steps_per_env, num_envs) when the run has no ``<run>_agent.yaml``
    (RT-206/207/231): rsl_rl's console prints
    ``Total timesteps: (it + 1) x F`` after every iteration; every printed value
    must be a multiple k x F of the first, and F / num_envs (env.yaml) a whole number."""
    log = next((p for p in (d / f"{run}.txt", d / f"{run}.short.txt") if p.is_file()), None)
    if log is None:
        raise FigError(f"{d}: no {run}_agent.yaml and no console log to read the env-step factor from")
    totals = [int(v) for v in re.findall(r"Total timesteps: (\d+)", log.read_text(encoding="utf-8", errors="replace"))]
    n_envs = run_axis._scene_num_envs(d / f"{run}_env.yaml")
    if not totals or totals[0] % n_envs or any(t % totals[0] for t in totals):
        raise FigError(f"{log}: 'Total timesteps' lines {totals[:3]}.. are not k x one factor over {n_envs} envs")
    return totals[0] // n_envs, n_envs


def extract_training(src: pathlib.Path, out: pathlib.Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    have = [run for run in FIXED_RUNS.values() if (src / run).is_dir()]
    if have and len(have) != len(FIXED_RUNS):
        raise FigError(f"Fixed-DR training runs: only {have} under {src}, want all {list(FIXED_RUNS.values())}")
    rows = []
    nodr_env = {}

    def _curve(m, s, run):
        # each run's own factor, from the params files beside its scalars.csv.
        agent = src / run / f"{run}_agent.yaml"
        factor = (run_axis.read_factor(agent, src / run / f"{run}_env.yaml") if agent.is_file()
                  else _log_factor(src / run, run))
        with open(src / run / f"{run}_scalars.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("Episode/success_rate", "") == "":
                    continue
                it = int(float(r["step"]))
                rows.append({"method": m, "seed": s, "iteration": it, "env_steps": run_axis.env_steps(it, *factor),
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
    return _write_training(out, rows, nodr_env, "RT-207s1..s5 env.yaml + last scalars row")


def _write_training(out: pathlib.Path, rows: list, nodr_env: dict, source: str) -> dict:
    """The shared tail of both extractions: ``curves.csv`` and No-DR's training
    values (F3's marker) from its env.yaml and the last scalars row."""
    first = next(iter(nodr_env.values()))
    for s, v in nodr_env.items():
        if v != first:
            raise FigError(f"No-DR seed {s} trained with other values than seed 1: {v} vs {first}")
    if first["dr_mode"] != "off":
        raise FigError(f"No-DR env.yaml says dr_mode {first['dr_mode']!r}, not 'off'")
    with open(out / "curves.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "seed", "iteration", "env_steps", *TRAINING_TAGS],
                           lineterminator="\n")
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
    nodr = {"source": source, "env": first,
            "lat_r_mm": 1000.0 * float(first["start_lateral_offset"]),
            "yaw_abs_deg": math.degrees(abs(float(first["fixture_yaw_rad"]))),
            "tilt_deg": math.degrees(float(first["fixture_tilt_rad"])),
            "start_height_m": list(sh[0])}
    for k in ("fixture_yaw_noise_rad", "fixture_tilt_noise_rad"):
        if float(first[k]) != 0.0:
            raise FigError(f"No-DR trained with {k} = {first[k]}: the F3 marker would be a band, not a value")
    # Friction of No-DR: the value its provider held, as logged
    # (dr/friction_mean, rl_games runs); one value over every row or refused.
    fr = sorted({float(r["dr/friction_mean"]) for r in rows
                 if r["method"] == "nodr" and r.get("dr/friction_mean") not in ("", None)})
    if fr:
        if len(fr) > 1:
            raise FigError(f"No-DR trained over {len(fr)} friction values {fr[:3]}...: "
                           f"the F3 marker would be a band, not a value")
        nodr["friction"] = fr[0]
    with open(out / "nodr_training.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nodr, fh, indent=2, sort_keys=True)
    return {"rows": len(rows), "nodr": nodr}


def _tag_column(header, tag: str) -> str:
    """The CSV column of a training tag: its rsl_rl name, else rl_games'
    ``Episode/<name>`` (rl_games logs every env key there; the candidate order
    of plot_autodr_run.need()). "" when the run did not log it."""
    for c in (tag, "Episode/" + tag):
        if c in header:
            return c
    return ""


def read_rl_games_run(folder: pathlib.Path, run: str) -> list:
    """One rl_games run (``<run>_scalars.csv`` + ``_agent.yaml`` + ``_env.yaml``
    in ``folder``): one row per ``Episode/success_rate`` row, in epoch order.
    ``iteration`` = the scalars step, which for rl_games' ``Episode/*`` tags is
    the epoch counted from 1; ``env_steps`` by tools/run_axis.py's rl_games rule
    (every TRAINING_TAGS column is an ``Episode/*`` tag). A tag the run
    did not log (No-DR has no lat/yaw/tilt/friction boundary) stays "". The one
    home of the per-run reading; plot_seed_curves.py calls it too."""
    try:
        lib, (horizon, envs) = run_axis.read_run_factor(folder / f"{run}_agent.yaml", folder / f"{run}_env.yaml")
        if lib != "rl_games":
            raise FigError(f"{run}: {lib} run; this reader takes rl_games runs only")
        frames = horizon * envs
        rows, ep_pairs = [], []
        with open(folder / f"{run}_scalars.csv", newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            cols = {t: _tag_column(rd.fieldnames, t) for t in TRAINING_TAGS}
            for r in rd:
                step = int(float(r["step"]))
                if r.get("info/epochs", ""):
                    ep_pairs.append((step, float(r["info/epochs"])))
                if r.get(cols["Episode/success_rate"], "") == "":
                    continue
                rows.append({"iteration": step,
                             "env_steps": run_axis.rl_games_x("Episode/success_rate", step, frames),
                             **{t: r.get(c, "") if c else "" for t, c in cols.items()}})
        run_axis.check_rl_games_epochs(ep_pairs, frames)
    except SystemExit as exc:   # run_axis refuses by SystemExit; the caller handles FigError
        raise FigError(f"{run}: {exc}") from None
    return sorted(rows, key=lambda r: r["iteration"])


def _rl_games_runs(folder: pathlib.Path, prefix: str) -> list:
    """Every ``<prefix><N>`` with a ``_scalars.csv`` in ``folder``, in seed order."""
    runs = sorted((p.name[: -len("_scalars.csv")] for p in folder.glob(f"{prefix}*_scalars.csv")),
                  key=lambda r: int(r[len(prefix):]))
    if len(runs) < 2:
        raise FigError(f"{folder}: {len(runs)} run(s) of {prefix}; the spread band needs two or more")
    return runs


def extract_training_rl_games(src: pathlib.Path, out: pathlib.Path, arms: dict) -> dict:
    """rl_games seed chains. ``arms`` = {arm: run prefix}, e.g. autodr
    RT-242s, nodr RT-244s, fixeddr RT-243s (their PC-side home is
    rl_games/run_eval.ps1); seed N's files are ``src/<prefix>/<prefix>N_*``.
    AutoDR and No-DR are required (F7, F8, F3's marker), Fixed-DR optional.
    The AutoDR states for F8 are copied when all of them exist."""
    if set(arms) - set(ARMS) or not set(METHODS) <= set(arms):
        raise FigError(f"--arm: got {sorted(arms)}; want {list(METHODS)}, optionally fixeddr")
    out.mkdir(parents=True, exist_ok=True)
    rows, nodr_env, last_epoch = [], {}, {}
    for m in ARMS:
        if m not in arms:
            continue
        folder = src / arms[m]
        for run in _rl_games_runs(folder, arms[m]):
            s = int(run[len(arms[m]):])
            got = read_rl_games_run(folder, run)
            rows += [{"method": m, "seed": s, **r} for r in got]
            if m == "nodr":
                nodr_env[s] = {k: _env_value(folder / f"{run}_env.yaml", k) for k in NODR_ENV_KEYS}
            if m == "autodr":
                last_epoch[s] = got[-1]["iteration"]
    res = _write_training(out, rows, nodr_env, f"{arms['nodr']}{min(nodr_env)}..s{max(nodr_env)} env.yaml + last scalars row")
    res["sidecars"] = copy_rl_games_sidecars(src / arms["autodr"], out, arms["autodr"], last_epoch)
    res["penetration_rows"] = extract_penetration(src, out, arms)
    return res


# H15: the interpenetration over the training. The per-episode
# record ``<run>_episodes.csv`` (check_episode_log.py) is git-ignored; this keeps
# per run and epoch the three counts H15 needs. All runs or none (as the sidecars).
PEN_FILE = "penetration.csv"


def extract_penetration(src: pathlib.Path, out: pathlib.Path, arms: dict) -> int:
    """``<prefix>N_episodes.csv`` + ``_episodes_meta.json`` of every run -> one
    row per run and epoch: episodes ended, successes, successes with
    ``ip_max_mm >= t1`` (t1 from the run's own meta; at or above it no success
    can fire at that step, so a success above t1 crossed it earlier in the
    episode). Epoch = iteration + 1, the scalars' epoch from 1 (H6's axis).
    Returns the row count; 0 when no run has the record."""
    rows, have, want = [], [], []
    for m in ARMS:
        if m not in arms:
            continue
        folder = src / arms[m]
        for run in _rl_games_runs(folder, arms[m]):
            want.append(run)
            ep, meta = folder / f"{run}_episodes.csv", folder / f"{run}_episodes_meta.json"
            if not (ep.is_file() and meta.is_file()):
                continue
            have.append(run)
            t1 = float(json.loads(meta.read_text(encoding="utf-8"))["interpenetration"]["t1_mm"])
            d = pd.read_csv(ep, usecols=["iteration", "outcome", "ip_max_mm"])
            if (d["iteration"] < 0).any():
                raise FigError(f"{ep.name}: iteration -1 (train.py did not hand the rollout length over)")
            ok = d["outcome"] == "success"
            g = pd.DataFrame({"epoch": d["iteration"] + 1, "ok": ok,
                              "ge": ok & (d["ip_max_mm"] >= t1)}).groupby("epoch")
            n, k, kg = g.size(), g["ok"].sum(), g["ge"].sum()
            rows += [{"method": m, "seed": int(run[len(arms[m]):]), "epoch": int(e), "episodes": int(n[e]),
                      "successes": int(k[e]), "successes_ge_t1": int(kg[e]), "t1_mm": t1} for e in n.index]
    if not have:
        return 0
    if len(have) != len(want):
        raise FigError(f"episodes.csv for {have} only, want all {want} or none")
    pd.DataFrame(rows).to_csv(out / PEN_FILE, index=False, lineterminator="\n")
    return len(rows)


def copy_rl_games_sidecars(folder: pathlib.Path, out: pathlib.Path, prefix: str, last_epoch: dict) -> list:
    """F8's AutoDR state per seed: ``<prefix><N>_autodr_final.json`` = the
    checkpoint's ``env_state`` plus its file name under ``checkpoint``.
    All seeds or none; each must name its own run and the epoch its curve ends
    on. Copied as ``autodr_s<N>_autodr_ep<E>.json``, never under rsl_rl's
    ``_1499`` name. Returns the copies; none -> [] (F8 waits)."""
    found = {s: folder / f"{prefix}{s}_autodr_final.json" for s in sorted(last_epoch)}
    have = {s: p for s, p in found.items() if p.is_file()}
    if not have:
        return []
    if len(have) != len(found):
        raise FigError(f"AutoDR state for seeds {sorted(have)} only, want all {sorted(found)} or none")
    written = []
    for s, p in have.items():
        side = json.loads(p.read_text(encoding="utf-8"))
        if side.get("seed") != s or f"__{prefix}{s}_" not in side.get("run", ""):
            raise FigError(f"{p.name}: seed {side.get('seed')} / run {side.get('run')!r} is not {prefix}{s}")
        m = re.search(r"_ep_(\d+)_rew_", side.get("checkpoint", ""))
        if not m or int(m.group(1)) != last_epoch[s]:
            raise FigError(f"{p.name}: checkpoint {side.get('checkpoint')!r} is not epoch {last_epoch[s]}, "
                           f"where the training curve ends")
        dst = out / f"autodr_s{s}_autodr_ep{m.group(1)}.json"
        shutil.copyfile(p, dst)
        written.append(dst)
    return written


def load_training(folder: pathlib.Path):
    curves = pd.read_csv(folder / "curves.csv")
    # rsl_rl writes autodr_s<N>_autodr_1499.json, rl_games autodr_s<N>_autodr_ep<E>.json.
    sidecars = {}
    for p in sorted(folder.glob("autodr_s*_autodr_*.json")):
        s = int(re.match(r"autodr_s(\d+)_autodr_", p.name).group(1))
        if ("autodr", s) in sidecars:
            raise FigError(f"{folder}: two AutoDR states for autodr s{s}")
        sidecars[("autodr", s)] = json.loads(p.read_text(encoding="utf-8"))
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
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, labelsize=FONT_PT)   # numbers black
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=FONT_PT)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=FONT_PT)


def _legend_methods(fig, loc="lower center", extra=(), marker=True, methods=METHODS, ax=None, ncol=None):
    """``marker=False`` for figures that draw lines only: the legend shows what
    the plot shows. ``methods``: the arms the figure draws, in ARMS order.
    ``ax``: a framed legend inside that axes instead of one row below the figure.
    ``ncol``: columns of the row below (default: every entry in one row)."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=METHOD_COLOR[m], marker=METHOD_MARKER[m] if marker else None, linewidth=1.6,
                markersize=5, markeredgecolor="white", markeredgewidth=0.5, label=METHOD_LABEL[m])
         for m in methods] + list(extra)
    if ax is not None:
        ax.legend(handles=h, loc=loc, frameon=True, fontsize=FONT_PT, edgecolor=GRID, framealpha=0.9)
        return
    return fig.legend(handles=h, loc=loc, ncol=ncol or len(methods) + len(extra), frameon=False, fontsize=FONT_PT)


# Two columns below a curve figure: the arms, then the thin line and the band.
# Three or more ran past the 6.3 in figure width (PNG 2503 px wide).
CURVE_LEGEND_NCOL = 2


def _curve_extra(n_seeds: int) -> tuple:
    """the two legend entries every mean-over-seeds curve figure adds --
    what a thin line is and what the band is (pointwise)."""
    return (_seed_line_handle(), _band_patch(n_seeds))


def _title(target, what: str, n_seeds: int):
    """Short title for an axes or a figure containing several panels."""
    # Keep the title clear of the axis labels and plot.
    # the figure draws the seeds themselves too (_seed_lines).
    text = f"{what} — {n_seeds} Seeds und ihr Mittel mit 95-%-KI"
    if hasattr(target, "suptitle"):
        target.suptitle(text, fontsize=10, fontweight="bold", color=INK)
    else:
        target.set_title(text, fontsize=10, fontweight="bold", color=INK, pad=18)


def _seed_count(d: dict, fig: str) -> int:
    """The one seed count a title can name; methods that differ are refused."""
    n = {len(v["seeds"]) for v in d.values()}
    if len(n) != 1:
        raise FigError(f"{fig}: the methods have different seed counts {sorted(n)}; one title cannot name them")
    return n.pop()


def draw_f1(plt, d):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.0), sharey=True)
    offs = np.linspace(-0.16, 0.16, 5)
    arms = list(d["T-final"])
    for ax, block in zip(axes, ("T-nominal", "T-final")):
        for j, m in enumerate(arms):
            r = d[block][m]
            for i, (rate, w) in enumerate(zip(r["rates"], r["wilson"])):
                x = j + offs[i % len(offs)]
                ax.errorbar(x, 100 * rate, yerr=_yerr(rate, w), fmt="o",
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
    axes[-1].set_xlabel("Erfolgsrate Auto-DR − No-DR in Prozentpunkten" if pair_only
                        else "Differenz der Erfolgsraten in Prozentpunkten", color=INK, fontsize=9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08 if pair_only else 0.045, 1, 1))
    return fig


def draw_f3(plt, d, nodr):
    panels = [pn for pn in F3_PANELS if pn[0] in d]
    fig, axes, spare, first_col = _panel_grid(plt, len(panels))
    for ax, (col, label, fac) in zip(axes, panels):
        p = d[col]
        if col == "start_height_m":
            a, b = nodr["start_height_m"]
            ax.axvspan(fac * a, fac * b, color=GRID, alpha=0.7, linewidth=0)
            ax.text(fac * a, 3, " No-DR-Training", fontsize=7, color=INK_MUTED, va="bottom")
        elif col in nodr:
            v = nodr[col]
            ax.axvline(v, color=INK_MUTED, linewidth=1.0, linestyle="--")
            ax.text(v, 3, " No-DR-Training", fontsize=7, color=INK_MUTED, va="bottom")
        # The seeds first, then all bands, then all means (as F7).
        for m in p["methods"]:
            _seed_lines(ax, p["x"], p["methods"][m]["seeds"], METHOD_COLOR[m])
        for m in p["methods"]:
            c = p["methods"][m]
            _band(ax, p["x"], c["ci"], METHOD_COLOR[m])
        for m in p["methods"]:
            ax.plot(p["x"], [100 * r for r in p["methods"][m]["mean"]], color=METHOD_COLOR[m], linewidth=1.6,
                    marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_ylim(0, 102)
        _x_range_ticks(ax, col, p)
        _style(ax, "Erfolgsrate in %" if ax in first_col else "", label)
    first = d[F3_PANELS[0][0]]["methods"]
    n_seeds = _seed_count(first, "F3")
    _title(fig, "Erfolgsrate je Störgröße", n_seeds)
    # No panel has a free corner (AutoDR runs along the top, No-DR and the
    # training marks along the bottom); inside the middle of a panel read badly
    # below the panels in two columns, or the free sixth cell.
    extra = _curve_extra(n_seeds)
    if spare is not None:
        # The sixth cell is narrow: the band text breaks before "des" (as H2).
        band = extra[1]
        band.set_label(band.get_label().replace(" des ", "\ndes ", 1).replace(" über ", "\nüber ", 1))
        _legend_methods(fig, loc="center", methods=tuple(first), ax=spare, extra=extra)
        fig.tight_layout()
    else:
        _legend_methods(fig, methods=tuple(first), extra=extra, ncol=CURVE_LEGEND_NCOL)
        fig.tight_layout(rect=(0, 0.10, 1, 1))
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
    label = f"{d['reference_block']} gesamt {_num(100 * d['overall'], '.1f')} % (Wilson-Band)"
    if gaps:
        ax.text(gaps[0] + 0.5, 100 * hi + 0.1, label, fontsize=7, color=INK_MUTED, ha="center", va="bottom")
    c = METHOD_COLOR[d["policy"]["method"]]
    for xi, rate, w in zip(x, d["rates"], d["wilson"]):
        ax.errorbar(xi, 100 * rate, yerr=_yerr(rate, w), fmt="o", color=c,
                    markersize=6, elinewidth=1.2, capsize=0, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(x, [K3_LABEL[k] for k in d["keys"]])
    ymin = 100 * min(w[0] for w in d["wilson"])
    ax.set_ylim(math.floor(ymin / 5) * 5, 101)
    _style(ax, "Erfolgsrate in %")
    if gaps:
        fig.tight_layout()
        return fig
    # Every neighbour pair reaches above the band (RT-252: four blocks at
    # 150/150): no free spot above it, so the label goes below the figure.
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=INK_MUTED, linestyle="--", label=label)], loc="lower center",
               frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 3.1))
    # Both panels as F7: the seeds first, then all bands, then all
    # means, using thinner lines for the individual seeds.
    for ax, part, x in ((a1, "ecdf", "x"), (a2, "cum", "k")):
        for m in d[part]:
            c = d[part][m]
            _seed_lines(ax, c[x], c["seeds"], METHOD_COLOR[m])
        for m in d[part]:
            c = d[part][m]
            _band(ax, c[x], c["ci"], METHOD_COLOR[m])
        for m in d[part]:
            c = d[part][m]
            ax.plot(c[x], 100 * c["mean"], color=METHOD_COLOR[m], linewidth=1.44)
    lim = d["limit"][0]
    a1.axvline(lim, color=INK, linewidth=0.9, linestyle="--")
    a1.text(lim, 50, f" Abbruch-\n grenze {_num(lim, 'g')} N", fontsize=7, color=INK, va="center")
    a1.set_ylim(0, 102)
    # Seeds without a success are not in (a); the panel says so, low and left
    # of the force line (every ECDF is above 85 % past 20 N there -- RT-232;
    # upper left ran into the curves at 100 %, lower right into the line).
    notes = [f"{METHOD_LABEL[m]}: {', '.join(f's{s}' for s in c['no_success'])} ohne Erfolg,\n"
             # The legend's seed count is (b)'s; (a)'s band of such an arm rests on fewer (Kritik A 2026-10-06).
             f"Kurve und Band über {len(c['seeds'])} Seeds" for m, c in d["ecdf"].items() if c["no_success"]]
    if notes:
        a1.text(0.97 * lim, 3, "\n".join(notes), fontsize=6.5, color=INK_MUTED, ha="right", va="bottom")
    _style(a1, "Anteil der Erfolge in %", "Maximalkraft je Erfolg in N")
    a2.set_ylim(0, 102)
    _style(a2, "Anteil erfolgreich bis Schritt k in %", "Policy-Schritt k")
    n_seeds = _seed_count(d["cum"], "F6")
    _title(fig, "Kraft und Dauer", n_seeds)
    _legend_methods(fig, marker=False, methods=tuple(d["ecdf"]), extra=_curve_extra(n_seeds),
                    ncol=CURVE_LEGEND_NCOL)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


def _band(ax, x, ci, colour, rate=True):
    """The 95 % bootstrap interval of the seed mean (``_boot_ci``) as a shaded
    band. A rate is drawn in %;
    ``rate=False`` (a return, a length) draws the band as it is."""
    lo, hi = (np.asarray(v, dtype=float) for v in ci)
    k = 100.0 if rate else 1.0
    ax.fill_between(x, k * lo, k * hi, color=colour, alpha=0.2, linewidth=0)


# every mean over the seeds shows its seeds too -- thin lines for a curve,
# small dots for a per-bin value (Colas et al. 2019: below ten runs, show every
# run beside the mean; Weissgerber et al. 2015 for the dots). Drawn first, so the
# band and the mean lie on top.
SEED_LINE = {"linewidth": 0.6, "alpha": 0.35}


def _seed_lines(ax, x, seeds, colour, rate=True, linestyle="-"):
    """One thin line per seed (``seeds``: {seed: curve} or a list of curves)."""
    k = 100.0 if rate else 1.0
    for v in (seeds.values() if isinstance(seeds, dict) else seeds):
        ax.plot(x, k * np.asarray(v, dtype=float), color=colour, linestyle=linestyle, zorder=1, **SEED_LINE)


def _mean_line(ax, x, y, m, j: int = 0):
    """The mean curve of arm ``m`` with its marker at every
    tenth of its points, arm ``j`` shifted by a third of that step so equal curves keep apart (D6)."""
    step = max(len(x) // 10, 1)
    ax.plot(x, y, color=METHOD_COLOR[m], linewidth=1.44, marker=METHOD_MARKER[m], markersize=4,
            markeredgecolor="white", markeredgewidth=0.5, markevery=(step // 2 + j * step // 3, step), zorder=3)


def _seed_line_handle():
    """Legend entry of ``_seed_lines``."""
    from matplotlib.lines import Line2D
    return Line2D([], [], color=INK_MUTED, linewidth=0.8, alpha=0.6, label=SEED_LINE_LABEL)


def _panel_grid(plt, n: int):
    """F3/H2: one panel per disturbance. Four fit 2 x 2; five take 3 x 2 and
    the sixth cell carries the legend (2 x 3 had no room for 12 pt tick numbers).
    Returns (fig, panels, spare cell or None, the first-column panels)."""
    rows, cols = (2, 2) if n <= 4 else (3, 2)
    fig, axes = plt.subplots(rows, cols, figsize=(FIG_W, 2.2 * rows), sharey=True)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n], (flat[n] if n < len(flat) else None), list(axes[:, 0])


# x ticks of the per-disturbance panels (F3, H2, H8), in the axis unit: the whole
# tested range from its first to its last bin edge, start and end labelled, in
# even steps (the tilt axis showed only "5", the end of the
# range was not readable). Lateral, yaw, tilt, start height step = one bin.
PANEL_TICK_STEP = {"lat_r_mm": 6.0, "yaw_abs_deg": 1.0, "tilt_deg": 2.0, "start_height_m": 50.0, "friction": 0.03}


def _x_range_ticks(ax, col: str, p):
    """Ticks over the whole bin range, a little room at both ends, so No-DR's
    line at 0 does not hide in the axis; ``lower``/``upper`` are already in
    the axis unit."""
    lo, hi = p["lower"][0], p["upper"][-1]
    step = PANEL_TICK_STEP[col]
    ax.set_xticks(np.round(np.arange(lo, hi + step / 2, step), 6))
    pad = 0.04 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)


def _arm_x(p, j: int, n: int):
    """H8: the bin centres of arm ``j`` of ``n``, set a small step apart (the
    one home of that step)."""
    w = p["upper"][0] - p["lower"][0]
    return np.asarray(p["x"]) + (j - (n - 1) / 2) * 0.14 * w


# H2: Auto-DR leaves the figure for a table
# (scripts/tools/table_h2_autodr.py); its values sat at 100 % where no one could read them.
H2_ARMS = ("nodr", "fixeddr")


def _seed_dots(ax, x, seeds, colour):
    """H2: one dot per seed and bin at the bin centre ``x`` -- every value
    where it belongs, overlaps accepted -- so a mean between two
    groups of runs shows where the runs really are. Nearly opaque ("vollere Farbe")."""
    for v in seeds.values():
        ax.scatter(x, 100 * np.asarray(v, dtype=float), s=9, color=colour, alpha=0.9, linewidths=0, zorder=2)


def _error_bars(ax, p, j: int, n: int, centre, lo, hi, colour, marker="o"):
    """H8: one arm's line through ``centre`` with a bar from ``lo`` to ``hi``
    at each bin, arm ``j`` of ``n`` set a small step beside the bin centre, so
    the bars of two arms at the same value stay apart. Values in the axis unit."""
    x = _arm_x(p, j, n)
    ax.errorbar(x, centre, yerr=[centre - lo, hi - centre], fmt="none", ecolor=colour, elinewidth=1.0,
                capsize=2.0, zorder=3)
    _arm_line(ax, x, centre, colour, marker)


def _arm_line(ax, x, centre, colour, marker):
    """H2/H8: one arm's mean as a line with a marker at each bin."""
    ax.plot(x, centre, color=colour, linewidth=1.6, marker=marker, markersize=3.5, markeredgecolor="white",
            markeredgewidth=0.5, zorder=4)


def _bar_handle(label: str, color: str = INK_MUTED):
    from matplotlib.lines import Line2D
    return Line2D([], [], color=color, marker="|", markersize=9, markeredgewidth=1.0, linestyle="none",
                  label=label)


# H2/H5/H6 legend: the interval's name, one home for both forms. A band
# over a curve holds point by point, not for the whole curve at once.
CI_LABEL = "95-%-KI des Mittelwerts (Bootstrap)"   # KI, not spelled out
CI_LABEL_CURVE = CI_LABEL.replace("(Bootstrap)", "(Bootstrap, punktweise)")
SEED_LINE_LABEL = "dünne Linie: ein Trainingslauf (Seed)"


def _band_patch(n_seeds: int):
    """Legend entry of ``_band`` (F3, F6, F7, F10, H5/H6, H9/H11 in plot_mlp_vs_lstm.py)."""
    from matplotlib.patches import Patch
    # short form; method and "punktweise" go to the caption.
    # Two lines: at 12 pt one line runs past the 6.3 in figure width.
    return Patch(facecolor=INK_MUTED, alpha=0.2,
                 label=f"Band: 95-%-KI\ndes Mittelwerts über {n_seeds} Seeds")


# 5-6 / 5-c: thin seed lines only for these arms; Auto-DR's and
# No-DR's runs lie close together, their band says enough. Fixed-DR splits into
# runs near 0 % and near 100 %, which only the single lines show.
SEED_LINE_ARMS = ("fixeddr",)
# Height of the training-curve plot area in inches.
CURVE_AXES_H = 2.2


def _arm_seed_handle(m: str):
    """Legend entry of one arm's thin seed lines (SEED_LINE_ARMS)."""
    from matplotlib.lines import Line2D
    return Line2D([], [], color=METHOD_COLOR[m], linewidth=0.8, alpha=0.7, label=f"{METHOD_LABEL[m]} einzelne Seeds")


def _training_extra(d: dict, fig_key: str) -> tuple:
    """Legend entries after the arms of a training-curve figure (F7, H5/H6)."""
    return (*(_arm_seed_handle(m) for m in SEED_LINE_ARMS if m in d), _band_patch(_seed_count(d, fig_key)))


def _legend_room(fig, leg, ax=None, axes_h=None):
    """Room below the plot for the legend ``leg`` as measured, not guessed (as
    draw_pairs). With ``axes_h`` (inches) the figure height is set so that
    ``ax`` is that tall; the width stays. Extra height spreads over the rows of
    panels, so each step adds the shortfall once per row."""
    for _ in range(8):
        fig.canvas.draw()
        fig.tight_layout(rect=(0, leg.get_window_extent().height / fig.bbox.height + 0.01, 1, 1))
        if axes_h is None:
            return
        fig.canvas.draw()
        have = ax.get_window_extent().height / fig.dpi
        if abs(have - axes_h) < 0.005:
            return
        rows = len({round(a.get_position().y0, 3) for a in fig.axes if a.axison})
        fig.set_figheight(fig.get_figheight() + rows * (axes_h - have))
    raise FigError(f"plot area did not reach {axes_h} in (now {have:.3f} in)")


def _wilson_handle():
    """Legend entry of the Wilson bars on one policy's rates (H3, H7)."""
    return _bar_handle("Strich: 95-%-KI (Wilson)", INK)   # the strokes are INK (H3, H7)


def draw_f7(plt, d):
    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    # The seeds first (only SEED_LINE_ARMS), then all bands,
    # then all means: no method's band may cover the other method's mean.
    for m in d:
        if m in SEED_LINE_ARMS:
            _seed_lines(ax, d[m]["x"], d[m]["seeds"], METHOD_COLOR[m])
    for m in d:
        _band(ax, d[m]["x"], d[m]["ci"], METHOD_COLOR[m])
    for j, m in enumerate(d):
        _mean_line(ax, d[m]["x"], 100 * d[m]["mean"], m, j)   # lw 1.8 x 0.8 (20 % thinner)
    ax.set_ylim(0, 102)
    _style(ax, YLABEL_TRAINING_RATE, XLABEL_TRAINING)
    _mio_axis(ax)
    # No title in the figure: the caption carries it.
    # Below the figure in two columns, as F6: inside, lower right was empty only
    # while every curve sat above 40 %; Fixed-DR's band reaches 5 % (RT-231).
    leg = _legend_methods(fig, methods=tuple(d), extra=_training_extra(d, "F7"), ncol=CURVE_LEGEND_NCOL)
    _legend_room(fig, leg, ax, CURVE_AXES_H)
    return fig


# F8's two limits (as in plot_autodr_run.fig_stufen): own
# symbol each, and the lower one in grey -- both are Auto-DR, so a second method
# colour would claim a method that is not there (D1).
F8_HI = dict(marker="o", markersize=2.5, markevery=(0.0, 0.1))
F8_LO = dict(color=INK_MUTED, marker="^", markersize=3.5, markerfacecolor="none", markeredgewidth=0.9,
             markevery=(0.05, 0.1))


def draw_f8(plt, d):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 3.9), sharex=True, sharey=True)
    c = METHOD_COLOR["autodr"]
    for ax, (name, label) in zip(axes.flat, F8_PANELS):
        for run in d.values():
            p = run["panels"][name]
            # Rule D6: the two limits lie on top of each other; each carries its own
            # symbol, evenly spaced and offset. Dashed is reserved for ablations (D4).
            # A one-sided panel has nothing to tell apart: plain line.
            ax.plot(run["x"], 100 * p["hi"], color=c, linewidth=0.9, alpha=0.8, **(F8_HI if "lo" in p else {}))
            if "lo" in p:
                ax.plot(run["x"], 100 * p["lo"], linewidth=0.9, alpha=0.8, **F8_LO)
        ax.set_title(label, fontsize=8.5, color=INK)
        ax.set_ylim(-5, 108)
        _style(ax)
        _mio_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel(XLABEL_TRAINING, color=INK, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Anteil am Maximum in %", color=INK, fontsize=9)
    h = [Line2D([], [], color=c, linewidth=1.2, label="obere Grenze (bzw. Boden)", **F8_HI),
         Line2D([], [], linewidth=1.2, label="untere Grenze", **F8_LO)]
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
        # so they stand at 45 deg there (6.5 pt, the n= label size).
        three = len(arms) > 2
        ax.set_xticks(range(len(le)), xl, fontsize=6.5 if three else 7, rotation=45 if three else 0,
                      ha="right" if three else "center", rotation_mode="anchor")
        ax.set_yticks(range(len(te)), yl, fontsize=7)
        ax.set_title(METHOD_LABEL[m], fontsize=9, color=INK)
        ax.set_xlabel(F3_PANELS[0][1], color=INK, fontsize=9)   # F9's offset axis = F3's (one home)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelcolor=INK, length=0)
    axes[0].set_ylabel("Kippung in °", color=INK, fontsize=9)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.035, pad=0.02)
    cb.set_label("Erfolgsrate in %", color=INK, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED, labelcolor=INK)
    cb.outline.set_visible(False)
    return fig


def draw_f10(plt, d):
    fig, ax = plt.subplots(figsize=(FIG_W, 2.8))
    # The seeds first, then all bands, then all means (as F7).
    for m in d["methods"]:
        _seed_lines(ax, d["tau"], d["seeds"][m], METHOD_COLOR[m])
    for m in d["methods"]:
        _band(ax, d["tau"], d["ci"][m], METHOD_COLOR[m])
    for m in d["methods"]:
        ax.plot(d["tau"], 100 * d["methods"][m], color=METHOD_COLOR[m], linewidth=1.8)
    ax.axvline(d["t1"], color=INK, linewidth=0.9, linestyle="--")
    # At 75 %: between Fixed-DR (~57 %) and AutoDR (~90 % and up) right of t1;
    # at 50 % the Fixed-DR curve ran through it (RT-232).
    ax.text(d["t1"], 75, f" t1 ={_num(d['t1'], '.3f')} mm\n (algorithmische Schwelle,\n keine Zulässigkeitsgrenze)",
            fontsize=7, color=INK, va="center")
    ax.set_ylim(0, 102)
    ax.set_xlim(0, float(d["tau"][-1]))
    _style(ax, "Erfolgsrate in %", "Erfolg nur gezählt bei maximaler Durchdringung < τ, τ in mm")
    n_seeds = {len(v) for v in d["seeds"].values()}
    if len(n_seeds) != 1:
        raise FigError(f"F10: the methods have different seed counts {sorted(n_seeds)}")
    _legend_methods(fig, marker=False, methods=tuple(d["methods"]), extra=_curve_extra(n_seeds.pop()),
                    ncol=CURVE_LEGEND_NCOL)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig


def draw_f11(plt, d):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(FIG_W, 3.8))
    x = np.arange(len(d["keys"]))
    lo, hi = d["overall_wilson"]
    ax.axhspan(100 * lo, 100 * hi, color=GRID, linewidth=0)
    ax.axhline(100 * d["overall"], color=INK_MUTED, linewidth=1.0, linestyle="--")
    c = METHOD_COLOR[d["policy"]["method"]]
    for xi, g, o in zip(x, d["grenze"], d["ood"]):
        ax.plot([xi - 0.16, xi + 0.16], [100 * g["rate"], 100 * o["rate"]], color=INK_MUTED, linewidth=0.8)
        for dx, e, face in ((-0.16, g, "white"), (0.16, o, c)):
            ax.errorbar(xi + dx, 100 * e["rate"], yerr=_yerr(e["rate"], e["wilson"]),
                        fmt="o", color=c, markerfacecolor=face, markeredgecolor=c, markeredgewidth=1.2,
                        markersize=6, elinewidth=1.2, capsize=0)
    ax.set_xticks(x, d["labels"])
    ymin = 100 * min(e["wilson"][0] for e in d["grenze"] + d["ood"])
    ax.set_ylim(math.floor(ymin / 5) * 5, 101)
    _style(ax, "Erfolgsrate in %")
    ax.tick_params(axis="x", labelsize=7)
    h = [Line2D([], [], color=c, marker="o", linestyle="none", markerfacecolor="white", markeredgewidth=1.2,
                label="an der Grenze (×1,0)"),
         Line2D([], [], color=c, marker="o", linestyle="none", label=f"außerhalb (×{_num(d['factor'], '.1f')})"),
         Line2D([], [], color=INK_MUTED, linestyle="--",
                label=f"{d['reference_block']} {_num(100 * d['overall'], '.1f')} % (Wilson-Band)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def draw_f12(plt, d):
    from matplotlib.lines import Line2D
    arms = list(d["arms"])
    fig, ax = plt.subplots(figsize=(FIG_W, 3.9))
    # One cluster per arm and key; the seeds of a cluster side by side.
    centre = (np.arange(len(arms)) - (len(arms) - 1) / 2) * (0.4 if len(arms) > 1 else 0.0)
    ax.axhline(0.0, color=INK, linewidth=0.9)
    for j, m in enumerate(arms):
        a = d["arms"][m]
        offs = np.linspace(-0.14, 0.14, len(a["seeds"])) if len(a["seeds"]) > 1 else np.zeros(1)
        for xi, (diffs, cis, mean) in enumerate(zip(a["diffs"], a["cis"], a["mean"])):
            for dx, v, ci in zip(offs, diffs, cis):
                ax.errorbar(xi + centre[j] + dx, 100 * v, yerr=_yerr(v, ci), fmt="o", color=METHOD_COLOR[m],
                            markersize=4, elinewidth=1, capsize=0, markeredgecolor="white", markeredgewidth=0.6)
            ax.plot([xi + centre[j] - 0.17, xi + centre[j] + 0.17], [100 * mean] * 2, color=INK, linewidth=1.4)
    ax.set_xticks(np.arange(len(d["keys"])), d["labels"])
    ax.set_xlim(-0.6, len(d["keys"]) - 0.4)
    _style(ax, "Differenz zum eigenen T-final\nin Prozentpunkten")
    ax.tick_params(axis="x", labelsize=7)
    h = [Line2D([], [], color=METHOD_COLOR[m], marker="o", linestyle="none", markeredgecolor="white",
                label=f"{METHOD_LABEL[m]} je Seed ({', '.join(str(s) for s in d['arms'][m]['seeds'])}), "
                      f"Newcombe-95-%-Intervall")
         for m in arms]
    h += [Line2D([], [], color=INK, linewidth=1.4, label="Mittel der Seeds je Methode")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


# Main-text figures H1-H4: no new colours (the method hues and the status
# palette, both validated 2026-09-23); a second condition is a hollow bar.
# Hollow = the reference test, filled = the test under study.
H_CONDITION = (("T-nominal", f"hohl: {TEST_NOMINAL}"), ("T-final", f"voll: {TEST_MAIN}"))
H_RUN_LABEL = POLICY_DOT_LABEL
# The mean above each bar: two numbers per group must not run into each other.
BAR_NUMBER_PT = FONT_PT


def _h_legend(fig, handles, ncol, **kw):
    return fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=FONT_PT, **kw)


def _mean_bar(ax, x, r, c, ls, filled, w, y_num=104):
    """H1's bar: the mean over the seeds (``r``: {"rates", "mean", "ci"}) as a
    hollow or filled bar, one dot per seed, the bootstrap interval of the mean
    beside the dots, the mean as a number in one row above the plot (a
    number on the bar top ran into the seed dots above a low mean). ``r["mean"]``
    None (an arm below LEARNED_MEAN_MIN_N policies): the dots only, no
    bar, interval or number."""
    dx = np.linspace(-0.08, 0.08, len(r["rates"])) if len(r["rates"]) > 1 else np.zeros(1)
    ax.scatter(x + dx, [100 * v for v in r["rates"]], s=14, color=INK, edgecolors="white", linewidths=0.5,
               zorder=3)
    if r.get("mean") is None:
        return
    ax.bar(x, 100 * r["mean"], width=w, color=c if filled else "white", edgecolor=c, linewidth=1.4,
           linestyle=ls, zorder=2)
    lo, hi = r["ci"]
    ax.errorbar(x + w / 2 - 0.05, 100 * r["mean"], yerr=[[100 * (r["mean"] - lo)], [100 * (hi - r["mean"])]],
                fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=3)   # INK: grey vanished on the filled bar
    ax.text(x, y_num, f"{_num(100 * r['mean'], '.1f')} %", ha="center", va="bottom", fontsize=BAR_NUMBER_PT,
            color=INK)


def _bars_axes(ax, ymin=0, step=20):
    """Rate axis of H1's form; ``ymin``/``step`` only for PAIR_YAXIS (5-9, 5-11).
    The room above 100 % (number row) is the same share of the axis for every ymin."""
    ax.set_ylim(ymin, 100 + 12 * (100 - ymin) / 100)
    ax.set_yticks(range(ymin, 101, step))
    _style(ax, YLABEL_RATE)
    ax.grid(axis="x", visible=False)


def _bars_legend(fig, hollow: str | None = None, filled: str | None = None):
    """Legend of H1's form, below the figure, room measured. Without ``hollow``
    and ``filled`` the tick labels name the tests and the legend leaves them
    out (Art 4)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    tests = ([] if hollow is None else
             [Patch(facecolor="white", edgecolor=INK_MUTED, linewidth=1.4, label=hollow),
              Patch(facecolor=INK_MUTED, edgecolor=INK_MUTED, label=filled)])
    # Without the test entries both columns are wide ones: closer together, else 1901 px.
    kw = {} if hollow is not None else {"columnspacing": 1.0}
    leg = _h_legend(fig, tests + [Line2D([], [], color=INK, marker="o", linestyle="none", markersize=4,
                                         label=H_RUN_LABEL),
                                  Line2D([], [], linestyle="none", label="Zahl über dem Balken: Mittelwert"),
                                  # Two lines: at 12 pt one line runs past the figure width.
                                  _bar_handle("Strich: " + CI_LABEL.replace(" des ", "\ndes ", 1), INK)], 2, **kw)
    _legend_room(fig, leg)


def _condition_bars(plt, groups, conditions):
    """H1's form: per group a hollow bar (first condition) beside a filled one
    (second), bar = mean over the seeds, one dot per seed, the mean as a number
    above. ``groups``: [(colour, edge line style, tick label, {block: {"rates",
    "mean"}})]; ``conditions``: the two (block, legend text) pairs."""
    fig, ax = plt.subplots(figsize=(FIG_W, 3.3))
    w = 0.34
    for j, (c, ls, _tick, by_block) in enumerate(groups):
        for k, (block, _label) in enumerate(conditions):
            _mean_bar(ax, j + (k - 0.5) * (w + 0.06), by_block[block], c, ls, bool(k), w)
    ax.set_xticks(range(len(groups)), [g[2] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    _bars_axes(ax)
    _bars_legend(fig, conditions[0][1], conditions[1][1])
    return fig


def _ref_block_bars(plt, d, blocks, ymin=0, step=20):
    """Art 4 in H1's form: one panel per arm, its
    own main test once as the hollow bar, then one filled bar per test block.
    ``d``: {arm: {"ref": r, block: r}} with r as in _mean_bar; ``blocks``:
    [(block key, tick label)] -- the labels of the pair form, with their values.
    The tick labels name the tests under every panel, so neither an axis title
    nor hollow/filled in the legend."""
    # Each panel taller than H1's single one: its own tick labels take three lines.
    fig, axes = plt.subplots(len(d), 1, figsize=(FIG_W, 3.4 * len(d)), sharex=True)
    axes = np.atleast_1d(axes)
    w = 0.6
    for ax, (m, a) in zip(axes, d.items()):
        c = METHOD_COLOR[m]
        for k, key in enumerate(["ref"] + [b for b, _l in blocks]):
            _mean_bar(ax, k, a[key], c, "-", k > 0, w, y_num=100 + 4 * (100 - ymin) / 100)
        _bars_axes(ax, ymin, step)
        ax.set_title(f"{METHOD_LABEL[m]} ({len(a['ref']['rates'])} Policies)", loc="left", fontsize=FONT_PT,
                     color=INK, pad=22)
    axes[-1].set_xticks(range(len(blocks) + 1), [TEST_MAIN] + [lab for _b, lab in blocks])
    axes[-1].set_xlim(-0.6, len(blocks) + 0.6)
    for ax in axes:
        ax.tick_params(axis="x", labelbottom=True)   # sharex hides them on the upper panels
    _bars_legend(fig)
    return fig


def _pairs_to_bars(d: dict) -> dict:
    """The pair data (data_learned + _add_pairs) in _ref_block_bars' form: per
    arm its policies' own main test (``ref``) and each block, the mean and its
    bootstrap interval as data_learned / _add_pairs computed them, None for an
    arm without a mean."""
    out = {}
    for m, a in d["arms"].items():
        has = "mean" in a
        arm = {"ref": {"rates": list(a["ref"]), "mean": a["ref_mean"] if has else None,
                       "ci": a["ref_ci"] if has else None}}
        for j, k in enumerate(d["keys"]):
            arm[k] = {"rates": list(a["rates"][:, j]), "mean": float(a["mean"][j]) if has else None,
                      "ci": (float(a["ci"][0][j]), float(a["ci"][1][j])) if has else None}
        out[m] = arm
    return out


def _pair_floor(bars: dict, ymin: float, fig: str) -> None:
    """A shortened rate axis (PAIR_YAXIS) must not cut a dot or an interval:
    every rate and interval end of _pairs_to_bars' form at or above ``ymin`` %."""
    for m, arm in bars.items():
        for key, r in arm.items():
            low = min(list(r["rates"]) + ([r["ci"][0]] if r["ci"] is not None else []))
            if 100 * low < ymin:
                raise FigError(f"{fig}: {m} {key} reaches {_num(100 * low, '.1f')} %, below the axis start {ymin} %")


def draw_pair_bars(plt, d, fig):
    """H18, H19, H20, H22 (thesis A-2, 5-9, 5-11, A-3) in H1's form, one panel per
    arm; the rate axis from PAIR_YAXIS. Prints
    the caption note on the policies left out and on the axis start."""
    ymin, step = PAIR_YAXIS.get(fig, (0, 20))
    gone = "; ".join(f"{METHOD_LABEL[m]} Seed {', '.join(str(s) for s, _r in lst)} ({TEST_MAIN} "
                     + " / ".join(f"{_num(100 * r, '.1f')}" for _s, r in lst) + " %)"
                     for m, lst in d["left_out"].items() if lst)
    print(f"[eval_figures] caption: not shown: {gone or 'none'}")
    bars = _pairs_to_bars(d)
    _pair_floor(bars, ymin, fig)
    if ymin:
        print(f"[eval_figures] caption: {fig} y axis starts at {ymin} %")
    return _ref_block_bars(plt, bars, list(zip(d["keys"], d["labels"])), ymin, step)


def draw_h1(plt, d):
    return _condition_bars(plt, [(METHOD_COLOR[m], "-", METHOD_LABEL[m], {b: d[b][m] for b, _ in H_CONDITION})
                                 for m in d["T-final"]], H_CONDITION)


def draw_h2(plt, d, nodr):
    """Mean over the seeds as a line (markers at the bin centres), the 95 %
    percentile bootstrap interval of that mean as a band. Line, band and one dot per seed all at the bin centre,
    no arm shifted. Only H2_ARMS
    (Auto-DR: its own table)."""
    from matplotlib.lines import Line2D
    panels = [pn for pn in F3_PANELS if pn[0] in d]
    fig, axes, spare, first_col = _panel_grid(plt, len(panels))
    methods = tuple(m for m in d[F3_PANELS[0][0]]["methods"] if m in H2_ARMS)
    for ax, (col, label, fac) in zip(axes, panels):
        p = d[col]
        if col != "start_height_m" and col in nodr:
            # No-DR saw only this value; the start height it trained over its whole range (caption).
            # Named in the legend: Fixed-DR's band (0-95 %) leaves no free spot for a text in the panel.
            # Dotted, not dashed: dashed means "Ablation".
            ax.axvline(nodr[col], color=INK_MUTED, linewidth=1.0, linestyle=":")
        for m in methods:
            c = p["methods"][m]
            x = np.asarray(p["x"])   # the bin centre itself, no step between the arms
            _seed_dots(ax, x, c["seeds"], METHOD_COLOR[m])
            _band(ax, x, c["ci"], METHOD_COLOR[m])
            _arm_line(ax, x, 100 * np.asarray(c["mean"]), METHOD_COLOR[m], METHOD_MARKER[m])
        ax.set_ylim(-2, 103)
        _x_range_ticks(ax, col, p)
        _style(ax, YLABEL_RATE if ax in first_col else "", label)
    n_seeds = _seed_count({m: d[F3_PANELS[0][0]]["methods"][m] for m in methods}, "H2")
    handles = ([Line2D([], [], color=METHOD_COLOR[m], linewidth=1.6, marker=METHOD_MARKER[m], markersize=3.5,
                       markeredgecolor="white", label=METHOD_LABEL[m]) for m in methods]
               + [_band_patch(n_seeds),
                  Line2D([], [], color=INK_MUTED, marker="o", markersize=3.5, alpha=0.9, linestyle="none",
                         label=H_RUN_LABEL),
                  Line2D([], [], color=INK_MUTED, linewidth=1.0, linestyle=":",
                         label="gepunktet: Wert von No-DR\nim Training")])
    if spare is not None:
        spare.legend(handles=handles, loc="center", frameon=False, fontsize=FONT_PT)
        fig.tight_layout()
    else:
        _h_legend(fig, handles, len(handles))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def draw_h3(plt, d):
    from matplotlib.patches import Patch
    c = METHOD_COLOR[d["policy"]["method"]]
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
    x = np.arange(len(d["keys"]))
    w = 0.36
    for xi, g, o in zip(x, d["grenze"], d["ood"]):
        for e, dx, face in ((g, -(w + 0.04) / 2, "white"), (o, (w + 0.04) / 2, c)):
            ax.bar(xi + dx, 100 * e["rate"], width=w, color=face, edgecolor=c, linewidth=1.4, zorder=2)
            # one policy's rate carries its Wilson interval.
            ax.errorbar(xi + dx, 100 * e["rate"], yerr=_yerr(e["rate"], e["wilson"]), fmt="none", ecolor=INK,
                        elinewidth=0.9, capsize=2.0, zorder=3)
        # Only the bars that carry the message are numbered: the ones beyond the range.
        ax.text(xi + (w + 0.04) / 2, 100 * o["wilson"][1] + 1.5, f"{_num(100 * o['rate'], '.1f')} %", ha="center",
                va="bottom", fontsize=7.5, color=INK)
    # A block that keeps factor 1,0 (start height) says so in words.
    labels = [lab.split("\n(×")[0] + ("" if f == d["factor"] else "\n(nicht erweitert)")
              for lab, f in zip(d["labels"], d["factors"])]
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    _style(ax, "Erfolgsrate in %")
    ax.grid(axis="x", visible=False)
    _h_legend(fig, [Patch(facecolor="white", edgecolor=c, linewidth=1.4, label="am Rand des Trainingsbereichs"),
                    Patch(facecolor=c, edgecolor=c,
                          label=f"{_num(100 * (d['factor'] - 1), '.0f')} % über den Trainingsbereich hinaus"),
                    _wilson_handle()], 3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def draw_h4(plt, rows):
    fig, ax = plt.subplots(figsize=(FIG_W, 2.4))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        left = 0.0
        for (_key, label, colr), cnt in zip(OUTCOMES, r["counts"]):
            w = 100 * cnt / r["n"]
            ax.barh(yi, w, left=left, color=colr, edgecolor="white", linewidth=1.0, height=0.62, label=label)
            if w >= 8:
                ax.text(left + w / 2, yi, f"{_num(w, '.1f')} %", ha="center", va="center", fontsize=8, color=INK)
            left += w
    ax.set_yticks(y, [f"{METHOD_LABEL[r['method']]}\n({r['seeds']} × {r['n'] // r['seeds']} Versuche)" for r in rows])
    ax.set_xlim(0, 100)
    _style(ax, "", "Anteil der Versuche mit voller Störung in %")
    ax.grid(False)
    # The legend names only the outcomes that occur.
    h, _lab = ax.get_legend_handles_labels()
    seen = [i for i in range(len(OUTCOMES)) if any(r["counts"][i] for r in rows)]
    _h_legend(fig, [h[i] for i in seen], len(seen))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def _thousands(n: int) -> str:
    """Digit groups with a narrow no-break space, as siunitx sets them (32 768)."""
    return format(int(n), ",").replace(",", " ")


def _draw_training(plt, d, factor: int, ylabel: str, rate: bool):
    """H5/H6: per arm the mean over the seeds as a line, the 95 % percentile
    bootstrap interval of that mean as a band, over the
    training iteration. The seeds first as thin lines, then
    all bands, then all means (as F7)."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    k = 100.0 if rate else 1.0
    for m in d:
        if m in SEED_LINE_ARMS:   # as F7
            _seed_lines(ax, d[m]["it"], d[m]["seeds"], METHOD_COLOR[m], rate=rate)
    for m in d:
        _band(ax, d[m]["it"], d[m]["ci"], METHOD_COLOR[m], rate=rate)
    for j, m in enumerate(d):
        _mean_line(ax, d[m]["it"], k * d[m]["mean"], m, j)
    if rate:
        ax.set_ylim(0, 102)
        ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(0, max(int(v["it"][-1]) for v in d.values()) + 1)
    _style(ax, ylabel, xlabel_epoch(factor))
    leg = _h_legend(fig, [Line2D([], [], color=METHOD_COLOR[m], linewidth=1.6, marker=METHOD_MARKER[m],
                                 markersize=5, markeredgecolor="white", markeredgewidth=0.5, label=METHOD_LABEL[m])
                          for m in d] + list(_training_extra(d, "H5/H6")), CURVE_LEGEND_NCOL)
    _legend_room(fig, leg, ax, CURVE_AXES_H)
    return fig


def draw_h5(plt, d, factor: int):
    return _draw_training(plt, d, factor, "Return je Episode (Reward-Summe)", rate=False)


def draw_h6(plt, d, factor: int):
    return _draw_training(plt, d, factor, YLABEL_TRAINING_RATE, rate=True)


def draw_h7(plt, d):
    """Two panels: the best policies under doubled observation errors (their
    full-disturbance rate first, as reference), and 50 % beyond the training
    range. One filled bar per arm, the rate in whole % above it."""
    from matplotlib.patches import Patch
    arms = list(d["arms"])
    groups = (["volle\nStörung", *d["noise_labels"]], d["ood_labels"])
    vals = ([[d["arms"][m]["reference"], *d["arms"][m]["noise"]] for m in arms],
            [d["arms"][m]["ood"] for m in arms])
    wils = ([[d["arms"][m]["reference_wilson"], *d["arms"][m]["noise_wilson"]] for m in arms],
            [d["arms"][m]["ood_wilson"] for m in arms])
    titles = ("Beobachtungsfehler × 2",
              f"{_num(100 * (d['factor'] - 1), '.0f')} % über den Trainingsbereich hinaus")
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.4), sharey=True,
                             gridspec_kw={"width_ratios": [len(groups[0]), len(groups[1])]})
    w = 0.8 / len(arms)
    for ax, labels, v, wl, title in zip(axes, groups, vals, wils, titles):
        x = np.arange(len(labels))
        for j, m in enumerate(arms):
            xs = x + (j - (len(arms) - 1) / 2) * w
            ax.bar(xs, [100 * r for r in v[j]], width=w * 0.92, color=METHOD_COLOR[m], zorder=2)
            for xi, r, wi in zip(xs, v[j], wl[j]):
                # each rate with its Wilson interval, the number above the bar's top.
                ax.errorbar(xi, 100 * r, yerr=_yerr(r, wi), fmt="none", ecolor=INK, elinewidth=0.9, capsize=1.8,
                            zorder=3)
                ax.text(xi, 100 * wi[1] + 1.0, _num(100 * r, ".0f"), ha="center", va="bottom", fontsize=6.5,
                        color=INK)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 20))
        _style(ax, "Erfolgsrate in %" if ax is axes[0] else "")
        ax.grid(axis="x", visible=False)
    _h_legend(fig, [Patch(color=METHOD_COLOR[m], label=f"{METHOD_LABEL[m]} s{d['arms'][m]['seed']}") for m in arms]
              + [_wilson_handle(), Patch(color="none", label="je Methode die beste von 5 Policies (gewählt auf T-auswahl)")],
              2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def draw_h8(plt, d):
    """Peak force of the successful trials per bin, one panel per disturbance:
    median as a line, the quartiles (the middle half of the trials) as an
    error bar at each bin, each policy set slightly apart, the force limit as
    a dashed line."""
    from matplotlib.lines import Line2D
    panels = [pn for pn in F3_PANELS if pn[0] in d["panels"]]
    fig, axes, spare, first_col = _panel_grid(plt, len(panels))
    lim = d["limit"]
    pols = list(d["policies"])
    for ax, (col, label, _fac) in zip(axes, panels):
        p = d["panels"][col]
        for j, m in enumerate(pols):
            q = p["methods"][m]
            _error_bars(ax, p, j, len(pols), np.asarray(q["median"]), np.asarray(q["q25"]), np.asarray(q["q75"]),
                        METHOD_COLOR[m])
        ax.axhline(lim, color=INK_MUTED, linewidth=1.0, linestyle="--")
        ax.set_ylim(0, 1.1 * lim)
        _x_range_ticks(ax, col, p)
        _style(ax, "Spitzenkraft in N" if ax in first_col else "", label)
    handles = ([Line2D([], [], color=METHOD_COLOR[m], linewidth=1.6, marker="o", markersize=3.5,
                       markeredgecolor="white", label=f"{METHOD_LABEL[m]} s{s}: Median")
                for m, s in d["policies"].items()]
               + [_bar_handle("Balken: die Hälfte der Versuche\nliegt darin (je ein Viertel\ndarunter und darüber)"),
                  Line2D([], [], color=INK_MUTED, linewidth=1.0, linestyle="--",
                         label=f"Kraftgrenze {_num(lim, '.0f')} N")])
    if spare is not None:
        spare.legend(handles=handles, loc="center", frameon=False, fontsize=8)
        fig.tight_layout()
    else:
        _h_legend(fig, handles, len(handles))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def draw_h12(plt, d):
    """H1's form for the force-sensor test: per arm the pocket belief
    error as in training (hollow) beside the doubled one (filled). The arms
    carry the H11 pair (FORCE_ARM_STYLE): colour, and a dashed edge for the
    ablation."""
    return _condition_bars(plt, [(*FORCE_ARM_STYLE[m], d[m]) for m in d], H12_CONDITION)


def draw_h13(plt, d, labels):
    """H1's form per arm: the arm's own main test once
    (hollow), then each block with one observation error doubled (filled).
    ``labels``: _noise_blocks' tick labels, in H7_NOISE order."""
    return _ref_block_bars(plt, d, list(zip([t for t, _l in H7_NOISE], labels)))


def draw_h14(plt, d):
    """F7's form per sweep panel (seeds, band, mean), one marker
    per step, the test range shaded. One title, one legend below."""
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, len(d), figsize=(FIG_W, 3.0), sharey=True)
    n_seeds = None
    for ax, p in zip(axes, d.values()):
        ax.axvspan(*p["range"], color=GRID, alpha=0.7, linewidth=0, zorder=0)
        n_seeds = _seed_count(p["arms"], "H14")
        for m, a in p["arms"].items():
            _seed_lines(ax, p["x"], a["seeds"], METHOD_COLOR[m])
        for m, a in p["arms"].items():
            _band(ax, p["x"], a["ci"], METHOD_COLOR[m])
        for m, a in p["arms"].items():
            ax.plot(p["x"], 100 * a["mean"], color=METHOD_COLOR[m], linewidth=1.44, marker="o", markersize=3.0,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.set_xticks(p["x"])
        ax.set_ylim(0, 102)
        _style(ax, "Erfolgsrate in %" if ax is axes[0] else "", p["xlabel"])
        ax.tick_params(axis="x", labelsize=7)
    _title(fig, "Erfolg über Kippung und Reibwert", n_seeds)
    arms = tuple(next(iter(d.values()))["arms"])
    _legend_methods(fig, marker=False, methods=arms, ncol=CURVE_LEGEND_NCOL,
                    extra=(*_curve_extra(n_seeds), Patch(facecolor=GRID, alpha=0.7, label=H14_RANGE_LABEL)))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    return fig


# H18: comparison of the out-of-range blocks:
# shape and line style carry the arm in greyscale too.
H18_MARKER = METHOD_MARKER   # old name, read by plot_kraft_durchdringung.py


def draw_h15(plt, d, factor: int):
    """H6's form (seeds as thin lines, 95 % band, mean line), two
    panels over the epoch. The share panel draws only the seeds its mean uses; a
    note names an arm whose share stands on fewer seeds."""
    from matplotlib.lines import Line2D
    arms = list(d["arms"])
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 4.6), sharex=True)
    for ax, key, field in zip(axes, ("clean", "share"), ("clean", "share_ge")):
        for m in arms:
            a = d["arms"][m]
            used = [s for s in a["seeds"] if s not in a[key].get("left_out", [])]
            _seed_lines(ax, a["x"], [a["seeds"][s][field] for s in used], METHOD_COLOR[m])
        for m in arms:
            _band(ax, d["arms"][m]["x"], d["arms"][m][key]["ci"], METHOD_COLOR[m])
        for m in arms:
            a = d["arms"][m]
            ax.plot(a["x"], 100 * a[key]["mean"], color=METHOD_COLOR[m], linewidth=1.44, marker="o", markersize=3.0,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        _style(ax)
    t1 = _num(d["t1"], ".3f")
    axes[0].set_ylabel("Erfolg mit Durchdringung\n< t1 in % der Episoden", color=INK, fontsize=9)
    axes[1].set_ylabel("Erfolge mit Durchdringung\n≥ t1 in % der Erfolge", color=INK, fontsize=9)
    axes[0].set_ylim(0, 102)
    axes[0].set_yticks(range(0, 101, 20))
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlim(0, max(int(a["x"][-1]) for a in d["arms"].values()) + 2)
    axes[1].set_xlabel(xlabel_epoch(factor), color=INK, fontsize=9)
    n = _seed_count({m: {"seeds": a["seeds"]} for m, a in d["arms"].items()}, "H15")
    h = [Line2D([], [], color=METHOD_COLOR[m], linewidth=1.6, label=METHOD_LABEL[m]) for m in arms]
    fig.legend(handles=h + list(_curve_extra(n)), loc="lower center", bbox_to_anchor=(0.5, 0.045),
               ncol=CURVE_LEGEND_NCOL, frameon=False, fontsize=8)
    short = [f"{METHOD_LABEL[m]} {a['share']['n']} von {n} Läufen ({len(a['share']['left_out'])} ohne Erfolg)"
             for m, a in d["arms"].items() if a["share"]["left_out"]]
    note = f"Punkt = {d['block']} Epochen;  t1 = {t1} mm (algorithmische Schwelle, keine Zulässigkeitsgrenze)"
    if short:
        note += "\nUnten gemittelt über: " + "; ".join(short)
    fig.text(0.5, 0.0, note, ha="center", va="bottom", fontsize=7.5, color=INK)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig


NAMES = {"F1": "F1_erfolg_je_seed", "F2": "F2_differenz", "F3": "F3_stoergroessen", "F4": "F4_grenzbloecke",
         "F5": "F5_endgruende", "F6": "F6_kraft_und_dauer", "F7": "F7_lernkurven", "F8": "F8_autodr_grenzen",
         "F9": "F9_heatmap_versatz_kippung", "F10": "F10_durchdringung", "F11": "F11_ood",
         "F12": "F12_ood_seeds", "H1": "H1_methodenvergleich", "H2": "H2_erfolg_ueber_stoerung",
         "H3": "H3_rand_und_ausserhalb", "H4": "H4_endgruende", "H5": "H5_return_im_training",
         "H6": "H6_erfolg_im_training", "H7": "H7_beste_policy_rauschen_ood", "H8": "H8_kraft_ueber_stoerung",
         "H12": "H12_kraft_lagefehler", "H13": "H13_beobachtungsfehler_doppelt", "H14": "H14_robustheitskurve",
         "H15": "H15_durchdringung_im_training", "H18": "H18_ausserhalb_pruefbereich",
         "H19": "H19_ausserhalb_gelernte_policies", "H20": "H20_beobachtungsfehler_gelernte_policies",
         "H22": "H22_beobachtungsfehler_alle_policies"}
# the force-ablation root (results/eval_rlgames_kraft/) holds the T-final
# runs of two AutoDR arms only; there H12 is the one figure.
FORCE_ROOT_ONLY = "force-ablation root: only H12"


def t1_from_results(results: pathlib.Path) -> float:
    vals = {json.loads(p.read_text(encoding="utf-8"))["interpenetration"]["t1_mm"]
            for p in sorted((results / "T-final").glob("*/episodes_meta.json"))}
    if len(vals) != 1:
        raise FigError(f"T-final episodes_meta.json files disagree on t1_mm (or none): {sorted(vals)}")
    return float(next(iter(vals)))


def _ood_sidecar_path() -> pathlib.Path:
    """The sidecar of the table T-ood plays, from eval_collect's block line
    (the one home; rl_games/run_eval.ps1 reads the same line)."""
    ec = _load_module("eval_collect", REPO / "scripts" / "eval_collect.py")
    return (ec.TABLE_DIR / ec.BLOCKS["T-ood"][0]).with_suffix(".json")


def _ood_inputs(results: pathlib.Path, sidecar_path: pathlib.Path) -> dict:
    """The OOD sidecar -- refused unless every T-ood run played exactly the
    table that sidecar describes (and at least one did)."""
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    shas = {json.loads(p.read_text(encoding="utf-8"))["table_sha256"]
            for p in sorted((results / "T-ood").glob("*/eval_manifest.json"))}
    if shas != {sidecar["sha256"]}:
        raise FigError(f"F11/F12: T-ood manifests played {sorted(shas)}, sidecar {sidecar_path.name} "
                       f"is {sidecar['sha256']}")
    return sidecar


def _sweep_inputs(sweep: pathlib.Path) -> dict:
    """H14: per sweep block (H14_PANELS) the sidecar of the table it plays,
    named by eval_collect's block line (one home) -- refused unless every run
    of the block played exactly that table, as ``_ood_inputs``."""
    ec = _load_module("eval_collect", REPO / "scripts" / "eval_collect.py")
    out = {}
    for block, *_rest in H14_PANELS:
        p = (ec.TABLE_DIR / ec.BLOCKS[block][0]).with_suffix(".json")
        side = json.loads(p.read_text(encoding="utf-8"))
        shas = {json.loads(m.read_text(encoding="utf-8"))["table_sha256"]
                for m in sorted((sweep / block).glob("*/eval_manifest.json"))}
        if shas != {side["sha256"]}:
            raise FigError(f"H14: {block} manifests played {sorted(shas)}, sidecar {p.name} is {side['sha256']}")
        out[block] = side
    return out


def _figure_keys(available: dict, only) -> list:
    """The figures to draw, in NAMES order. ``available``: key -> "" when its
    input exists, else why not. Without ``only`` a figure without input is left
    out with one printed line; ``only`` naming one (or an unknown key) is refused."""
    if only:
        unknown = sorted(set(only) - set(NAMES))
        if unknown:
            raise FigError(f"--only: unknown figure(s) {unknown}; known {list(NAMES)}")
        missing = {k: available[k] for k in only if available[k]}
        if missing:
            raise FigError(f"--only: no input for {missing}")
        return [k for k in NAMES if k in only]
    for k in NAMES:
        if available[k]:
            print(f"[eval_figures] {k} left out: {available[k]}")
    return [k for k in NAMES if not available[k]]


def draw_all(results: pathlib.Path, training: pathlib.Path, out: pathlib.Path, only=None,
             sweep: pathlib.Path | None = None) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    except ValueError as exc:
        raise FigError(f"font {FONT!r} not installed; matplotlib would fall back silently") from exc
    plt.rcParams.update({"font.size": FONT_PT, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "eval_figures",   # fixed element ids: SVG bytes repeat
                         "font.family": "sans-serif", "font.sans-serif": [FONT]})
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    # Optional episode-level data used by F6, F9, F10 and H8.
    episodes = results / "episodes_all.csv"
    df = (pd.read_csv(episodes, keep_default_na=False, dtype={"tag": str, "boundary_key": str})
          if episodes.is_file() else None)
    force_root = "FORCE" in summary
    if force_root:
        # no training curves belong to this root; nothing below reads them.
        curves, sidecars, nodr, factor, why_not_h56 = None, {}, None, None, FORCE_ROOT_ONLY
    else:
        curves, sidecars, nodr = load_training(training)
        try:
            factor = iteration_factor(curves)
            why_not_h56 = ""
        except FigError as exc:
            factor, why_not_h56 = None, str(exc)
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
        "F11": lambda: draw_f11(plt, data_f11(summary, _ood_inputs(results, _ood_sidecar_path()))),
        "F12": lambda: draw_f12(plt, data_f12(summary, _ood_inputs(results, _ood_sidecar_path()))),
        "H1": lambda: draw_h1(plt, data_f1(summary)),
        "H2": lambda: draw_h2(plt, data_f3(summary), nodr),
        "H3": lambda: draw_h3(plt, data_f11(summary, _ood_inputs(results, _ood_sidecar_path()))),
        "H4": lambda: draw_h4(plt, data_h4(summary)),
        "H5": lambda: draw_h5(plt, data_h5(curves), factor),
        "H6": lambda: draw_h6(plt, data_f7(curves), factor),
        "H7": lambda: draw_h7(plt, data_h7(summary, _ood_inputs(results, _ood_sidecar_path()))),
        "H8": lambda: draw_h8(plt, data_h8(df, {m: s for m, s in best_policies(summary).items() if m in H8_ARMS})),
        "H12": lambda: draw_h12(plt, data_h12(summary)),
        "H13": lambda: draw_h13(plt, data_h13(summary), _noise_blocks(summary, "H13")[2]),
        # One form for all four: per arm its own main test, then the blocks, H1's bars.
        "H18": lambda: draw_pair_bars(plt, data_h18_pairs(summary, _ood_inputs(results, _ood_sidecar_path())), "H18"),
        "H19": lambda: draw_pair_bars(plt, data_h19_pairs(summary, _ood_inputs(results, _ood_sidecar_path())), "H19"),
        "H20": lambda: draw_pair_bars(plt, data_h20_pairs(summary), "H20"),
        "H22": lambda: draw_pair_bars(plt, data_h22_pairs(summary), "H22"),
        "H14": lambda: draw_h14(plt, data_h14(
            pd.read_csv(sweep / "episodes_all.csv", keep_default_na=False, dtype={"tag": str, "boundary_key": str}),
            _sweep_inputs(sweep))),
        "H15": lambda: draw_h15(plt, data_h15(pd.read_csv(training / PEN_FILE)), factor),
    }
    available = {k: "" for k in NAMES}
    if force_root:
        available = {k: ("" if k == "H12" else FORCE_ROOT_ONLY) for k in NAMES}
    else:
        if not sidecars:
            available["F8"] = f"no AutoDR state in {training}"
        if not (results / "T-ood").is_dir():
            available["F11"] = available["F12"] = available["H3"] = available["H7"] = available["H18"] = (
                available["H19"]) = f"no T-ood block in {results}"
        if not any(p["block"] == "T-auswahl" for p in summary["policies"]):
            available["H7"] = available["H8"] = f"no T-auswahl block in {results}: no best policy per arm"
        available["H5"] = available["H6"] = why_not_h56
        if RETURN_TAG not in curves.columns:
            available["H5"] = f"no {RETURN_TAG} column in {training / 'curves.csv'} (older extraction)"
        available["H12"] = "no FORCE key in summary.json (no autodr_nf runs)"
        if "NOISE" not in summary:
            available["H13"] = available["H20"] = available["H22"] = "no NOISE key in summary.json (no pn2/gx2/py2 runs)"
        if sweep is None or not (sweep / "episodes_all.csv").is_file():
            available["H14"] = f"no episodes_all.csv in the sweep root --sweep {sweep} (RT-264/265)"
        available["H15"] = why_not_h56
        if not (training / PEN_FILE).is_file():
            available["H15"] = f"no {PEN_FILE} in {training} (extract with --extract-training ... --arm)"
        if df is None:
            available["F6"] = available["F9"] = available["F10"] = available["H8"] = f"no {episodes}"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key in _figure_keys(available, only):
        fig = jobs[key]()
        for ext in ("pdf", "svg", "png"):
            p = out / f"{NAMES[key]}.{ext}"
            fig.savefig(p, dpi=PNG_DPI, bbox_inches="tight", metadata=SAVE_META[ext])
            written.append(p)
        plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Self-test: checks on built data; each mutation of THIS file's source
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
                             # lat 0..28.5 mm, tilt 0..9.5 deg; start height 0.030..0.258 m, i.e. the five 50 mm bins above H_min, none below
                             # friction 0.081..0.195: four rows in each of the five bins
                             "lat_r_mm": 1.5 * i, "yaw_abs_deg": 0.25 * i, "tilt_deg": 0.5 * i,
                             "start_height_m": 0.03 + 0.012 * i, "friction": 0.081 + 0.006 * i})
    # Fixed-DR in its own blocks: T-final-F 12/20 and 10/20, T-nominal-F 17/20 and 16/20.
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
                             "start_height_m": 0.03 + 0.012 * i, "friction": 0.081 + 0.006 * i})
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
    for m in ARMS:   # Fixed-DR curves too; F8 reads autodr only
        for s in (1, 2):
            for it in range(3):
                rows.append({"method": m, "seed": s, "iteration": it, "env_steps": (it + 1) * 64,
                             "Episode/success_rate": 0.1 * it + (0.1 if s == 2 else 0.0),
                             "dr/lat_r_hi": 0.015 * it, "dr/yaw_lo": -0.0436 * it, "dr/yaw_hi": 0.0436 * it,
                             "dr/tilt_hi": 0.0873 * it, "dr/start_height_hi": 0.02 + 0.05 * it,
                             "dr/start_height_floor": -0.03 + 0.025 * it, "dr/friction_lo": 0.14 - 0.03 * it,
                             "dr/friction_hi": 0.14 + 0.03 * it,
                             # H5: seeds 10 apart, negative at it 0 (a return is not clipped)
                             RETURN_TAG: -50.0 + 100.0 * it + (10.0 if s == 2 else 0.0), "dr/friction_mean": 0.14})
    dims = [{"name": "lat_r", "centre": 0.0, "hi_max": 0.03, "lo_max": 0.0, "one_sided": True},
            {"name": "yaw", "centre": 0.0, "hi_max": 0.0872, "lo_max": -0.0872, "one_sided": False},
            {"name": "tilt", "centre": 0.0, "hi_max": 0.1746, "lo_max": 0.0, "one_sided": True},
            {"name": "start_height", "centre": 0.02, "hi_max": 0.12, "lo_max": 0.02, "one_sided": True},
            {"name": "friction", "centre": 0.14, "hi_max": 0.20, "lo_max": 0.08, "one_sided": False}]
    side = {"autodr": {"dims": dims, "floor": {"f0": -0.03, "top": 0.02, "n_steps": 5}}}
    return pd.DataFrame(rows), {("autodr", s): side for s in (1, 2)}


# each arm's synthetic num_steps_per_env, all with num_envs 4.
ARM_STEPS = {"autodr": 16, "nodr": 128, "fixeddr": 32}


def _write_rt_logs(root: pathlib.Path, bad_seed: int | None = None, runs=None):
    for (m, s), run in (runs or TRAINING_RUNS).items():
        d = root / run
        d.mkdir(parents=True)
        with open(d / f"{run}_scalars.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "wall_time", *TRAINING_TAGS])
            for it in range(2):
                w.writerow([it, 0, 0.5, "", "", "", "", 0.12, 0.02, "", "", "", ""])
            w.writerow([1, 0, *[""] * len(TRAINING_TAGS)])   # a */time row
        (d / f"{run}_autodr_1499.json").write_text("{}", encoding="utf-8")
        tilt = "0.01" if (m == "nodr" and s == bad_seed) else "0.0"
        (d / f"{run}_env.yaml").write_text(
            f"dr_mode: 'off'\nstart_lateral_offset: 0.0\nfixture_yaw_rad: 0.0\nfixture_yaw_noise_rad: 0.0\n"
            f"fixture_tilt_rad: {tilt}\nfixture_tilt_noise_rad: 0.0\nscene:\n  num_envs: 4\n", encoding="utf-8")
        (d / f"{run}_agent.yaml").write_text(f"seed: {s}\nnum_steps_per_env: {ARM_STEPS[m]}\n", encoding="utf-8")


# rl_games fixture: each arm its own horizon_length, all with 4 envs.
RLG_HORIZON = {"autodr": 8, "nodr": 16, "fixeddr": 32}
RLG_PREFIX = {"autodr": "RA", "nodr": "RN", "fixeddr": "RF"}


def _rlg_value(tag: str, e: int, s: int, m: str = "autodr", nodr_friction_step: float = 0.0) -> float:
    if tag == "dr/friction_mean":   # No-DR's provider holds the centre; the others move
        return 0.14 + (nodr_friction_step * e if m == "nodr" else 0.001 * e)
    return {"Episode/success_rate": 0.1 * e + 0.05 * s, "dr/lat_r_hi": 0.01 * e, "dr/yaw_lo": -0.02 * e,
            "dr/yaw_hi": 0.02 * e, "dr/tilt_hi": 0.05 * e, "dr/start_height_hi": 0.02 + 0.05 * e,
            "dr/start_height_floor": -0.03 + 0.025 * e, "dr/friction_lo": 0.14 - 0.01 * e,
            "dr/friction_hi": 0.14 + 0.01 * e, RETURN_TAG: 100.0 * e + 10.0 * s}[tag]


def _write_rlg_logs(root: pathlib.Path, bad_epochs: bool = False, nodr_friction_step: float = 0.0):
    """Two seeds per arm, epochs 1..3, as rl_games writes them: ``info/epochs``
    at step (epoch - 1) x F (``bad_epochs``: at epoch x F), every env key under
    ``Episode/`` at step = epoch. No-DR logs no lat/yaw/tilt/friction boundary,
    but its return and its constant friction (``nodr_friction_step``: not constant)."""
    for m, prefix in RLG_PREFIX.items():
        d = root / prefix
        d.mkdir(parents=True)
        frames = RLG_HORIZON[m] * 4
        tags = (("Episode/success_rate", "dr/start_height_hi", "dr/start_height_floor", RETURN_TAG, "dr/friction_mean")
                if m == "nodr" else TRAINING_TAGS)
        cols = [t if t.startswith("Episode/") else "Episode/" + t for t in tags]
        for s in (1, 2):
            run = f"{prefix}{s}"
            with open(d / f"{run}_scalars.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["step", "wall_time", "info/epochs", *cols])
                for e in (1, 2, 3):
                    w.writerow([e * frames if bad_epochs else (e - 1) * frames, 0, e, *[""] * len(cols)])
                    w.writerow([e, 0, "", *[_rlg_value(t, e, s, m, nodr_friction_step) for t in tags]])
            (d / f"{run}_env.yaml").write_text(
                "dr_mode: 'off'\nstart_lateral_offset: 0.0\nfixture_yaw_rad: 0.0\nfixture_yaw_noise_rad: 0.0\n"
                "fixture_tilt_rad: 0.0\nfixture_tilt_noise_rad: 0.0\nscene:\n  num_envs: 4\n", encoding="utf-8")
            (d / f"{run}_agent.yaml").write_text(
                f"params:\n  seed: {s}\n  config:\n    horizon_length: {RLG_HORIZON[m]}\n", encoding="utf-8")


CHECKS = (
    "F1: T-final autodr mean = (0.9 + 0.8) / 2 and dots carry summary's Wilson",
    "F2: T-nominal row reads K2, T-final row reads K1",
    "F3: empty start-height bins dropped (5 of 10 edges hold rows)",
    "F3: lateral bin 24-30 mm mean over seeds = autodr (0.5 + 0.0) / 2",
    "F4: keys in the fixed order, tilt_hi 3/4, overall band = Wilson(18, 20)",
    "F5: every policy's segments sum to its n",
    "F6: cumulative share at the last step = the success rate; ECDF over successes only (No-DR mean at 8 N = 5/6)",
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
    "F6: Fixed-DR from T-final-F: s1 share at the last step = 0.6, ECDF over its successes (mean at 14 N = 11/12)",
    "F9: Fixed-DR panel, cell (tilt 0-2, lat 0-6) pooled rate = 1.0 over 4 distinct rows",
    "F10: Fixed-DR curve from T-final-F: rate at the largest tau = 0.55, at tau 0 = 0",
    "extraction takes all five Fixed-DR runs (RT-231s1..s5) and refuses four",
    "F7: Fixed-DR mean curve when its curves exist; without them F7 keeps the pair",
    "F7: stored sample std over the seeds (ddof=1; printed, not drawn)",
    "F3: stored sample std over the seeds (ddof=1), lateral bin 24-30 mm autodr",
    "F6: stored sample std over the seeds (ddof=1), force ECDF and share by step k",
    "F1/F4 error bar: Wilson rounding at 0 of n (2.2e-19) draws length 0; a bound beyond the rate is refused",
    "F6: a Fixed-DR seed with no success leaves (a), is named there, and stays in (b) at share 0",
    "extraction: every arm's env_steps = (iteration + 1) x its own num_steps_per_env x num_envs",
    "F7: two seeds of one arm on different env-step grids are refused",
    "Fixed-DR runs in T-auswahl and T-ood beside its two blocks still draw three arms",
    "F11: x1.0 value = centre + (physical - centre) / the block's own factor; tilt 3/4 beside 2/4; "
    "two different T-final references are refused",
    "F12: per arm one dot per seed = its OOD difference with ITS Newcombe interval; line = across-seed "
    "diff mean; Fixed-DR beside AutoDR",
    "F11/F12: T-ood runs that played another table than the sidecar, or none, are refused",
    "rl_games extraction: env_steps = epoch x horizon_length x num_envs; Episode/dr/* read as dr/*; "
    "No-DR start height = its last [floor, hi]; info/epochs off the factor refused",
    "rl_games AutoDR state: all seeds or none, each naming its own run and the curve's last epoch; copied as _ep<E>",
    "load_training finds rsl_rl (_1499) and rl_games (_ep<E>) AutoDR states; F8 names a seed without one",
    "--only naming a figure without input or an unknown key is refused; without --only it is left out",
    "F4: the reference line names the final policy's own T-final block (T-final-F for Fixed-DR)",
    "H4: one bar per arm = F5's counts summed over its seeds (AutoDR 34 of 40, Fixed-DR 22 of 40)",
    # the final result figures H5/H6 and the friction panel.
    "H5: mean return = mean of the seeds (-45, 55, 155), sample std 10 / sqrt(2), negative values kept",
    "H5/H6: iteration axis needs ONE env-step factor over every run (epoch x F); two factors or rsl_rl's "
    "(it + 1) x F are refused",
    "F3/H2 friction panel: five bins of eval_stats' edges, bin 0 from 0.08 (not 0), x = midpoints, 4 rows each",
    "rl_games extraction: No-DR friction marker = its one logged dr/friction_mean (0.14); a second value is refused",
    # best policy per arm (H7, H8).
    "best_policies: the pre-registered rule on EACH arm's T-auswahl runs (autodr s1, nodr s1); no T-auswahl, no entry",
    "H7: each arm's best policy reads its own NOISE / OOD blocks; an arm without such runs is left out and named",
    "H8: per bin the nearest-rank median and quartiles of the SUCCESSES' peak force (lat 0-6: 6 N; 24-30: 21 N)",
    # H12, the force-sensor test.
    "H12: per arm and block the seed rates and the arm mean of summary's FORCE (autodr_nf pn2 mean "
    "(0.85 + 0.80 + 0.88) / 3); the arms carry the H11 pair (Auto-DR blue solid, #d55181 dashed)",
    "H12: a summary without FORCE, or with a FORCE block marked missing, is refused",
    # H2/H5/H6 draw the bootstrap interval of the seed mean.
    "bootstrap interval: five seeds (0, 0, 0.993, 0.996, 0.056) = own percentile reference within 0.01; "
    "a column of equal seeds gets that value",
    "H2/H5/H6 read the interval: two seeds -> [lower seed, upper seed] (H6, H5 autodr; H2 lateral bin 24-30 mm)",
    "H3/H7: each single-policy rate carries its Wilson interval (H7 reference = Wilson(18, 20), noise and OOD "
    "blocks their own)",
    "F1/F6/F10/H12: the seed mean carries its bootstrap interval (F1 and F6(b) autodr two seeds -> [0.8, 0.9]; "
    "F10 Fixed-DR and H12 inside [lowest, highest seed] around the mean)",
    # the seeds themselves are drawn beside the mean.
    "F10: one curve per seed is kept for the thin lines; their mean = the drawn mean (AutoDR, Fixed-DR)",
    # H13, the doubled observation errors.
    "H13: per arm (ARMS order) the seeds sorted, each seed's own reference beside its pn2/gx2/py2 rate "
    "(fixeddr s1, s2: ref [0.0, 0.9], gx2 [0.01, 0.81]); a policy without a block, or no NOISE, is refused",
    # RT-264/265: H14, the robustness curve.
    "H14: steps sorted by value (friction 0.08, 0.20, 0.26 from keys hi_x1, hi_x2, lo_x1), test range = centre "
    "to the factor-1.0 steps (tilt 0-10, friction 0.08-0.20), mean = mean of the seeds; a step set other than "
    "the sidecar's is refused",
    # H15, the interpenetration over the training.
    "H15: per seed and 10 epochs, clean = (successes - successes >= t1) / episodes (autodr s1 0.6), share = "
    "successes >= t1 / successes (0.25), no success -> NaN (nodr s2 block 2); the share mean leaves that seed "
    "out and names it (nodr: 2 of 3, left_out [2]), clean keeps it; two t1 values are refused",
    # H19/H20, only the learned policies.
    "H19/H20: data_learned keeps the policies whose own main test is >= 0.90 (autodr s1-s5 incl. s4 at "
    "exactly 0.90; fixeddr s2, s4, not s1 0.0 / s3 0.5), sorted by seed, rule on the reference not on the "
    "block (autodr s5 block 0.6 stays); mean only for autodr (5), none for fixeddr (2); left_out and ref_range "
    "right; an arm without a learned policy is refused",
    # H22, all policies of the noise blocks.
    "H22: data_learned with min_rate 0 keeps every policy (fixeddr s1 at 0.0 too), both arms get a mean, "
    "ref_mean = mean of the arm's own main tests (fixeddr 0.586); an arm under LEARNED_MEAN_MIN_N is refused",
    # H20 as pairs.
    "H20 pairs: each shown policy's own main test carries Wilson(k, n) of its reference (autodr s1 990/1000), "
    "autodr gets ref_mean 0.97 with a bootstrap interval around it, fixeddr (2 policies) gets none",
    # shortened axis of 5-9/5-11.
    "PAIR_YAXIS floor: a dot (0.79) or an interval end (0.795) under the axis start 80 % is refused with the figure "
    "and arm named; dots and interval ends at 0.80 and above pass, an arm without a mean (ci None) too",
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
        # No-DR successes: s1 forces 5..10 N (6), s2 5..8 N (4). At 8 N: 4/6 and 4/4.
        e = d6["ecdf"]["nodr"]
        at8 = int(np.flatnonzero(e["x"] == 8.0)[0])
        return (abs(c["seeds"][1][-1] - 0.9) < 1e-12 and np.all(np.diff(c["mean"]) >= 0)
                and abs(e["mean"][at8] - 5 / 6) < 1e-12)

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
                and d2[3]["label"] == "Auto-DR − Fixed-DR"
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
        # Fixed-DR successes: s1 forces 5..16 N (12), s2 5..14 N (10). At 14 N: 10/12 and 10/10.
        e = d6["ecdf"]["fixeddr"]
        at14 = int(np.flatnonzero(e["x"] == 14.0)[0])
        return (abs(d6["cum"]["fixeddr"]["seeds"][1][-1] - 0.6) < 1e-12
                and abs(e["mean"][at14] - 11 / 12) < 1e-12)

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
    # Seeds 1 and 2 differ by 0.1 at every iteration: sample std = 0.1 / sqrt(2).
    _check(25, lambda: np.allclose(ns["data_f7"](curves)["autodr"]["std"], [0.1 / math.sqrt(2)] * 3))
    # Same bin as check 3: seeds 0.5 and 0.0 -> sample std 0.5 / sqrt(2).
    _check(26, lambda: abs(ns["data_f3"](s)["lat_r_mm"]["methods"]["autodr"]["std"][-1] - 0.5 / math.sqrt(2)) < 1e-12)

    def c27():
        # ECDF No-DR at 8 N: 4/6 and 1 -> std (1/3)/sqrt(2); share at the last step
        # AutoDR: 0.9 and 0.8 -> std 0.1/sqrt(2).
        d6 = ns["data_f6"](df)
        e = d6["ecdf"]["nodr"]
        at8 = int(np.flatnonzero(e["x"] == 8.0)[0])
        return (abs(e["std"][at8] - (1 / 3) / math.sqrt(2)) < 1e-12
                and abs(d6["cum"]["autodr"]["std"][-1] - 0.1 / math.sqrt(2)) < 1e-12)

    _check(27, c27)

    def c28():
        # RT-232: Fixed-DR s1, 0 of 1000 -> statsmodels' lower bound 2.168e-19.
        if ns["_yerr"](0.0, [2.168404344971009e-19, 0.003826758485555125])[0][0] != 0.0:
            return False
        try:
            ns["_yerr"](0.5, [0.6, 0.7])
            return False
        except Exception:   # noqa: BLE001 - FigError of the executed copy
            return True

    _check(28, c28)

    def c29():
        # A third Fixed-DR seed in T-final-F with 0 of 20 (RT-232 s1/s2 pattern).
        zero = df[(df["block"] == "T-final-F") & (df["seed"] == 1)].copy()
        zero["seed"], zero["success_train"], zero["success_strict"], zero["outcome"] = 3, 0, 0, "timeout"
        d6 = ns["data_f6"](pd.concat([df, zero], ignore_index=True))
        e, c = d6["ecdf"]["fixeddr"], d6["cum"]["fixeddr"]
        at14 = int(np.flatnonzero(e["x"] == 14.0)[0])
        return (e["no_success"] == [3] and sorted(e["seeds"]) == [1, 2] and abs(e["mean"][at14] - 11 / 12) < 1e-12
                and sorted(c["seeds"]) == [1, 2, 3] and c["seeds"][3][-1] == 0.0)

    _check(29, c29)

    def c31():
        other = curves.copy()
        sel = (other["method"] == "autodr") & (other["seed"] == 2)
        other.loc[sel, "env_steps"] = 2 * other.loc[sel, "env_steps"]
        try:
            ns["data_f7"](other)
            return False
        except Exception as exc:   # noqa: BLE001 - FigError of the executed copy
            return "env steps" in str(exc)

    _check(31, c31)

    def c32():
        extra = df[df["method"] == "fixeddr"].copy()
        extra["block"] = extra["block"].map({"T-final-F": "T-auswahl", "T-nominal-F": "T-ood"})
        return ns["arms_present"](ES.summarize(pd.concat([df, extra], ignore_index=True))) == ns["ARMS"]

    _check(32, c32)
    import contextlib
    import io
    import tempfile

    def c33():
        # p6 c13's sidecar: start height capped at x1.2, the rest x1.5.
        phys = {"lat_r_hi": 0.045, "yaw_lo": -0.1309, "yaw_hi": 0.1309, "tilt_hi": 0.2618,
                "start_height_hi": 0.14, "friction_lo": 0.05, "friction_hi": 0.23}
        side13 = {"factor": 1.5, "centres": {"lat_r": 0.0, "yaw": 0.0, "tilt": 0.0, "start_height": 0.02, "friction": 0.14},
                  "blocks": [{"key": k, "physical": phys[k], "factor": 1.2 if k == "start_height_hi" else 1.5}
                             for k in K3_ORDER]}
        kk = {k: 2 if k == "tilt_hi" else 4 for k in K3_ORDER}
        p1 = {"policy": {"method": "autodr", "seed": 1}, "reference": {"block": "T-final", "k": 18, "n": 20, "rate": 0.9},
              "blocks": {k: {"k": kk[k], "n": 4, "rate": kk[k] / 4, "wilson": ES.wilson(kk[k], 4)} for k in K3_ORDER}}
        d = ns["data_f11"](dict(s, OOD={"policies": {"autodr_s1": p1}}), side13)
        i = {k: K3_ORDER.index(k) for k in K3_ORDER}
        # start height 0.02 + 0.12 / 1.2 = 0.12; friction 0.14 - 0.09 / 1.5 = 0.08
        ok = (abs(d["edge"][i["start_height_hi"]] - 0.12) < 1e-12 and abs(d["edge"][i["friction_lo"]] - 0.08) < 1e-12
              and abs(d["grenze"][i["tilt_hi"]]["rate"] - 0.75) < 1e-12 and d["ood"][i["tilt_hi"]]["rate"] == 0.5
              and np.allclose(d["overall_wilson"], ES.wilson(18, 20)))
        try:
            bad = dict(p1, reference={"block": "T-final", "k": 17, "n": 20, "rate": 0.85})
            ns["data_f11"](dict(s, OOD={"policies": {"autodr_s1": bad}}), side13)
            return False
        except Exception:   # noqa: BLE001 - FigError of the executed copy
            return ok

    _check(33, c33)

    def c34():
        # T-ood rows through the REAL eval_stats: AutoDR s1 tilt 2/4 (rest 4/4), s2 1/4 (3/4)
        # against T-final 18/20, 16/20; Fixed-DR s1 1/4 (3/4), s2 0/4 (2/4) against T-final-F 12/20, 10/20.
        rows = []
        for m, seed, k_tilt, k_rest in (("autodr", 1, 2, 4), ("autodr", 2, 1, 3), ("fixeddr", 1, 1, 3), ("fixeddr", 2, 0, 2)):
            for j, key in enumerate(K3_ORDER):
                for r in range(4):
                    ok = int(r < (k_tilt if key == "tilt_hi" else k_rest))
                    rows.append({"block": "T-ood", "method": m, "seed": seed, "tag": "", "row_id": 4 * j + r,
                                 "boundary_key": key, "outcome": "success" if ok else "timeout",
                                 "success_train": ok, "success_strict": ok, "force_abort": 0, "steps": 20,
                                 "force_max_filtered_n": 5.0})
        fin = df[df["block"].isin(("T-final", "T-final-F")) & df["method"].isin(("autodr", "fixeddr"))]
        o = ES.ood_blocks(pd.concat([fin, pd.DataFrame(rows)], ignore_index=True))
        side12 = {"blocks": [{"key": k, "physical": 0.1, "factor": 1.5} for k in K3_ORDER]}
        d = ns["data_f12"]({"OOD": o}, side12)
        t = K3_ORDER.index("tilt_hi")
        from statsmodels.stats.proportion import confint_proportions_2indep as cp
        a, f = d["arms"].get("autodr", {}), d["arms"].get("fixeddr", {})
        ref_a = [cp(2, 4, 18, 20, method="newcomb", compare="diff"), cp(1, 4, 16, 20, method="newcomb", compare="diff")]
        return (list(d["arms"]) == ["autodr", "fixeddr"] and a["seeds"] == [1, 2] and f["seeds"] == [1, 2]
                and np.allclose(a["diffs"][t], [0.5 - 0.9, 0.25 - 0.8], atol=1e-12)
                and np.allclose(a["cis"][t], ref_a, atol=1e-12)
                and abs(a["mean"][t] - ((0.5 - 0.9) + (0.25 - 0.8)) / 2) < 1e-12
                and np.allclose(f["diffs"][t], [0.25 - 0.6, 0.0 - 0.5], atol=1e-12)
                and abs(f["mean"][t] - ((0.25 - 0.6) + (0.0 - 0.5)) / 2) < 1e-12)

    _check(34, c34)

    def c35():
        with tempfile.TemporaryDirectory() as tmp:
            t = pathlib.Path(tmp)
            sc = t / "table.json"
            sc.write_text(json.dumps({"sha256": "aa"}), encoding="utf-8")
            (t / "T-ood").mkdir()
            try:
                ns["_ood_inputs"](t, sc)   # no run at all
                return False
            except Exception:   # noqa: BLE001
                pass
            for run, sha in (("r1", "aa"), ("r2", "bb")):
                (t / "T-ood" / run).mkdir()
                (t / "T-ood" / run / "eval_manifest.json").write_text(json.dumps({"table_sha256": sha}), encoding="utf-8")
            try:
                ns["_ood_inputs"](t, sc)   # r2 played another table
                return False
            except Exception:   # noqa: BLE001
                pass
            (t / "T-ood" / "r2" / "eval_manifest.json").write_text(json.dumps({"table_sha256": "aa"}), encoding="utf-8")
            return ns["_ood_inputs"](t, sc) == {"sha256": "aa"}

    _check(35, c35)

    def c36():
        with tempfile.TemporaryDirectory() as tmp:
            t = pathlib.Path(tmp)
            _write_rlg_logs(t / "rlg")
            got = ns["extract_training_rl_games"](t / "rlg", t / "o", dict(RLG_PREFIX))
            cv = pd.read_csv(t / "o" / "curves.csv")
            nodr = json.loads((t / "o" / "nodr_training.json").read_text(encoding="utf-8"))
            want = cv["iteration"] * cv["method"].map(RLG_HORIZON) * 4
            a2 = cv[(cv["method"] == "autodr") & (cv["seed"] == 2)].sort_values("iteration")
            ok = (got["rows"] == 18 and got["sidecars"] == [] and bool((cv["env_steps"] == want).all())
                  and np.allclose(a2["dr/lat_r_hi"], [0.01, 0.02, 0.03])
                  and np.allclose(nodr["start_height_m"], [-0.03 + 0.075, 0.02 + 0.15]))
            _write_rlg_logs(t / "bad", bad_epochs=True)
            try:
                ns["extract_training_rl_games"](t / "bad", t / "o_bad", dict(RLG_PREFIX))
                return False
            except Exception as exc:   # noqa: BLE001
                return ok and "info/epochs" in str(exc)

    _check(36, c36)

    def c37():
        with tempfile.TemporaryDirectory() as tmp:
            t = pathlib.Path(tmp)
            f, out = t / "RA", t / "o"
            f.mkdir()
            out.mkdir()
            last = {1: 3, 2: 3}

            def put(seed, run_seed=None, ep=3):
                (f / f"RA{seed}_autodr_final.json").write_text(json.dumps(
                    {"seed": seed, "run": f"09-28_10-45-20__RA{run_seed or seed}_autodr_seed{seed}",
                     "checkpoint": f"last_ur5e_insertion_ep_{ep}_rew_1.5.pth", "autodr": {}}), encoding="utf-8")

            def refused():
                try:
                    ns["copy_rl_games_sidecars"](f, out, "RA", last)
                    return False
                except Exception:   # noqa: BLE001
                    return True

            none = ns["copy_rl_games_sidecars"](f, out, "RA", last) == []
            put(1)
            partial = refused()
            put(2)
            names = sorted(p.name for p in ns["copy_rl_games_sidecars"](f, out, "RA", last))
            put(2, run_seed=1)
            other_run = refused()
            put(2, ep=2)
            other_epoch = refused()
            return (none and partial and other_run and other_epoch
                    and names == ["autodr_s1_autodr_ep3.json", "autodr_s2_autodr_ep3.json"])

    _check(37, c37)

    def c38():
        with tempfile.TemporaryDirectory() as tmp:
            t = pathlib.Path(tmp)
            pd.DataFrame({"method": ["autodr"], "seed": [1]}).to_csv(t / "curves.csv", index=False)
            (t / "nodr_training.json").write_text("{}", encoding="utf-8")
            for name, v in (("autodr_s1_autodr_1499", 1), ("autodr_s2_autodr_ep100", 2), ("nodr_s1_autodr_1499", 3)):
                (t / f"{name}.json").write_text(json.dumps({"autodr": v}), encoding="utf-8")
            _c, got, _n = ns["load_training"](t)
        ok = got == {("autodr", 1): {"autodr": 1}, ("autodr", 2): {"autodr": 2}}
        try:
            ns["data_f8"](curves, {k: v for k, v in side.items() if k != ("autodr", 2)})
            return False
        except Exception as exc:   # noqa: BLE001
            return ok and "autodr s2" in str(exc)

    _check(38, c38)

    def c39():
        fk = ns["_figure_keys"]
        av = {k: "" for k in ns["NAMES"]}
        av["F8"] = "no AutoDR state"

        def refused(only):
            try:
                fk(av, only)
                return False
            except Exception:   # noqa: BLE001
                return True

        with contextlib.redirect_stdout(io.StringIO()):
            free = fk(av, None)
        return (refused({"F8"}) and refused({"F99"}) and fk(av, {"F12", "F1"}) == ["F1", "F12"]
                and free == [k for k in ns["NAMES"] if k != "F8"])

    _check(39, c39)
    _check(40, lambda: ns["data_f4"](s)["reference_block"] == "T-final"
           and ns["data_f4"](dict(s, final_policy=dict(s["final_policy"], method="fixeddr")))["reference_block"]
           == "T-final-F")

    def c41():
        h = ns["data_h4"](s)
        by = {r["method"]: r for r in h}
        return ([r["method"] for r in h] == ["autodr", "nodr", "fixeddr"] and by["autodr"]["n"] == 40
                and by["autodr"]["seeds"] == 2 and by["autodr"]["counts"][0] == 34 and by["fixeddr"]["counts"][0] == 22
                and all(sum(r["counts"]) == r["n"] for r in h))

    _check(41, c41)

    def c42():
        a = ns["data_h5"](curves)["autodr"]
        return np.allclose(a["mean"], [-45.0, 55.0, 155.0]) and np.allclose(a["std"], [10.0 / math.sqrt(2)] * 3)

    _check(42, c42)

    def c43():
        def refused(cv):
            try:
                ns["iteration_factor"](cv)
                return False
            except Exception:   # noqa: BLE001 - FigError of the executed copy
                return True

        ok = curves.copy()
        ok["iteration"] = ok["iteration"] + 1   # rl_games: epoch from 1, env_steps = epoch x F
        ok["env_steps"] = ok["iteration"] * 64
        two = ok.copy()
        sel = two["method"] == "fixeddr"
        two.loc[sel, "env_steps"] = two.loc[sel, "iteration"] * 128
        return ns["iteration_factor"](ok) == 64 and refused(two) and refused(curves)

    _check(43, c43)

    def c44():
        f = ns["data_f3"](s)["friction"]
        return (len(f["x"]) == 5 and f["n"] == [4] * 5 and abs(f["lower"][0] - 0.08) < 1e-12
                and abs(f["x"][0] - 0.092) < 1e-12 and abs(f["upper"][-1] - 0.2) < 1e-12)

    _check(44, c44)

    def c45():
        with tempfile.TemporaryDirectory() as tmp:
            t = pathlib.Path(tmp)
            _write_rlg_logs(t / "rlg")
            ns["extract_training_rl_games"](t / "rlg", t / "o", dict(RLG_PREFIX))
            nodr = json.loads((t / "o" / "nodr_training.json").read_text(encoding="utf-8"))
            _write_rlg_logs(t / "two", nodr_friction_step=0.01)
            try:
                ns["extract_training_rl_games"](t / "two", t / "o2", dict(RLG_PREFIX))
                return False
            except Exception as exc:   # noqa: BLE001
                return nodr.get("friction") == 0.14 and "friction" in str(exc)

    _check(45, c45)
    # T-auswahl: autodr s1 18/20 beats s2 16/20; nodr s1 6/20 beats s2 4/20; Fixed-DR plays none.
    _check(46, lambda: ns["best_policies"](s) == {"autodr": 1, "nodr": 1})

    def c47():
        # autodr s1 is the best AutoDR policy; give it (and a decoy s2) noise and OOD blocks.
        def blocks(keys, rate):
            return {k: {"rate": rate, "n": 150, "wilson": [rate - 0.05, rate + 0.04]} for k in keys}

        ref = {"block": "T-final", "k": 18, "n": 20, "rate": 0.9, "tag": ""}
        noise = {"autodr_s1": {"reference": ref, "blocks": blocks(("pn2", "gx2", "py2"), 0.8)},
                 "autodr_s2": {"reference": ref, "blocks": blocks(("pn2", "gx2", "py2"), 0.1)}}
        ood = {"autodr_s1": {"reference": ref, "blocks": blocks(K3_ORDER, 0.7)},
               "autodr_s2": {"reference": ref, "blocks": blocks(K3_ORDER, 0.2)}}
        side = {"factor": 1.5, "blocks": [{"key": k, "factor": 1.0 if k == "start_height_hi" else 1.5} for k in K3_ORDER]}
        d = ns["data_h7"](dict(s, NOISE={"policies": noise}, OOD={"policies": ood}), side)
        a = d["arms"].get("autodr", {})
        return (list(d["arms"]) == ["autodr"] and d["left_out"] == {"nodr": 1} and a["seed"] == 1
                and a["noise"] == [0.8] * 3 and a["ood"] == [0.7] * 7 and a["reference"] == 0.9
                and d["ood_labels"][K3_ORDER.index("start_height_hi")].endswith("(nicht erweitert)"))

    _check(47, c47)

    def c48():
        # autodr s1 on T-final: forces 5 + i, successes i < 18, lat = 1.5 i.
        # lat 0-6: i 0..3 -> 5, 6, 7, 8 -> nearest-rank median 6, q25 5, q75 7.
        # lat 24-30: i 16..19, successes 16, 17 only -> 21, 22 -> median 21 (failures 23, 24 left out).
        p = ns["data_h8"](df, {"autodr": 1})["panels"]["lat_r_mm"]["methods"]["autodr"]
        return (p["median"][0] == 6.0 and p["q25"][0] == 5.0 and p["q75"][0] == 7.0
                and p["median"][-1] == 21.0 and p["n"][0] == 4 and p["n"][-1] == 2)

    _check(48, c48)
    # eval_stats' own force-root frame (seeds 1-3 per arm, T-final and pn2).
    s_force = ES.summarize(ES._force_frame())
    s_force_untagged = ES.summarize(ES._force_frame(tags=("",)))

    def c49():
        d = ns["data_h12"](s_force)
        nf = d["autodr_nf"]["pn2"]
        st = ns["FORCE_ARM_STYLE"]
        return (sorted(d) == ["autodr", "autodr_nf"] and nf["seeds"] == [1, 2, 3]
                and np.allclose(nf["rates"], [0.85, 0.80, 0.88], atol=1e-12)
                and abs(nf["mean"] - (0.85 + 0.80 + 0.88) / 3) < 1e-12
                and abs(d["autodr"]["T-final"]["mean"] - 0.98) < 1e-12
                and st["autodr"][:2] == ("#3B6FD4", "-") and st["autodr_nf"][:2] == ("#d55181", "--"))

    _check(49, c49)

    def c50():
        for bad in (s, s_force_untagged):
            try:
                ns["data_h12"](bad)
                return False
            except Exception as exc:   # noqa: BLE001 - FigError of the executed copy
                if "H12" not in str(exc):
                    return False
        return True

    _check(50, c50)

    def c51():
        a = np.array([[0.0, 0.4], [0.0, 0.4], [0.993, 0.4], [0.996, 0.4], [0.056, 0.4]])
        lo, hi = ns["_boot_ci"](a, "test")
        rng = np.random.default_rng(7)
        ref = np.percentile(a[:, 0][rng.integers(0, 5, (20000, 5))].mean(1), [2.5, 97.5])
        return np.allclose([lo[0], hi[0]], ref, atol=0.01) and lo[1] == hi[1] == 0.4

    _check(51, c51)

    def c52():
        def two(seeds, ci):
            a = np.asarray(list(seeds.values()), dtype=float)
            return len(a) == 2 and np.allclose(ci[0], a.min(0)) and np.allclose(ci[1], a.max(0))
        lat = ns["data_f3"](s)["lat_r_mm"]["methods"]["autodr"]
        return (two(ns["data_f7"](curves)["autodr"]["seeds"], ns["data_f7"](curves)["autodr"]["ci"])
                and two(ns["data_h5"](curves)["autodr"]["seeds"], ns["data_h5"](curves)["autodr"]["ci"])
                and two(lat["seeds"], lat["ci"]) and lat["ci"][0][-1] == 0.0 and lat["ci"][1][-1] == 0.5)

    _check(52, c52)

    def c53():
        def blocks(keys, rate):
            return {k: {"rate": rate, "n": 150, "wilson": [rate - 0.05, rate + 0.04]} for k in keys}
        ref = {"block": "T-final", "k": 18, "n": 20, "rate": 0.9, "tag": ""}
        noise = {"autodr_s1": {"reference": ref, "blocks": blocks(("pn2", "gx2", "py2"), 0.8)}}
        ood = {"autodr_s1": {"reference": ref, "blocks": blocks(K3_ORDER, 0.7)}}
        side = {"factor": 1.5, "blocks": [{"key": k, "factor": 1.5} for k in K3_ORDER]}
        a = ns["data_h7"](dict(s, NOISE={"policies": noise}, OOD={"policies": ood}), side)["arms"]["autodr"]
        return (np.allclose(a["reference_wilson"], ES.wilson(18, 20))
                and np.allclose(a["noise_wilson"], [[0.75, 0.84]] * 3)
                and np.allclose(a["ood_wilson"], [[0.65, 0.74]] * 7))

    _check(53, c53)

    def c54():
        def inside(ci, rates, mean):
            lo, hi = (float(v) for v in ci)
            return min(rates) - 1e-12 <= lo < hi <= max(rates) + 1e-12 and lo <= mean <= hi
        f1 = ns["data_f1"](s)["T-final"]["autodr"]
        cum = ns["data_f6"](df)["cum"]["autodr"]
        f10 = ns["data_f10"](df, 0.2)
        fx = df[(df["block"] == "T-final-F") & (df["tag"] == "")]
        f10_seeds = [float((g[KEY] == 1).mean()) for _s, g in fx.groupby("seed")]
        nf = ns["data_h12"](s_force)["autodr_nf"]["pn2"]
        return (np.allclose(f1["ci"], [0.8, 0.9]) and np.allclose([cum["ci"][0][-1], cum["ci"][1][-1]], [0.8, 0.9])
                and inside([f10["ci"]["fixeddr"][0][-1], f10["ci"]["fixeddr"][1][-1]], f10_seeds,
                           f10["methods"]["fixeddr"][-1])
                and inside(nf["ci"], nf["rates"], nf["mean"]))

    _check(54, c54)

    def c55():
        f10 = ns["data_f10"](df, 0.2)
        ok = True
        for m, block in (("autodr", "T-final"), ("fixeddr", "T-final-F")):
            n = df[(df["block"] == block) & (df["tag"] == "") & (df["method"] == m)]["seed"].nunique()
            sd = f10["seeds"][m]
            ok = ok and len(sd) == n and np.allclose(np.mean(sd, axis=0), f10["methods"][m])
        return ok

    _check(55, c55)

    def c56():
        def one(m, seed, ref, x, tags=tuple(t for t, _l in H7_NOISE)):
            return {"policy": {"method": m, "seed": seed}, "reference": {"rate": ref},
                    "blocks": {t: {"rate": x + [u for u, _l in H7_NOISE].index(t) / 100} for t in tags}}
        # Fixed-DR s2 first: the seeds must come out sorted.
        # (_boot_ci needs two seeds per arm.)
        noise = {"fixeddr_s2": one("fixeddr", 2, 0.9, 0.8), "fixeddr_s1": one("fixeddr", 1, 0.0, 0.0),
                 "autodr_s1": one("autodr", 1, 0.99, 0.9), "autodr_s2": one("autodr", 2, 0.97, 0.7)}
        d = ns["data_h13"](dict(s, NOISE={"policies": noise}))
        f = d["fixeddr"]
        ok = (list(d) == ["autodr", "fixeddr"] and f["ref"]["seeds"] == [1, 2]
              and np.allclose(f["ref"]["rates"], [0.0, 0.9]) and np.allclose(f["gx2"]["rates"], [0.01, 0.81])
              and abs(f["gx2"]["mean"] - 0.41) < 1e-12 and np.allclose(d["autodr"]["py2"]["rates"], [0.92, 0.72]))
        noise["fixeddr_s1"] = one("fixeddr", 1, 0.0, 0.0, tags=("pn2", "gx2"))
        for bad in (dict(s, NOISE={"policies": noise}), {k: v for k, v in s.items() if k != "NOISE"}):
            try:
                ns["data_h13"](bad)
                return False
            except Exception as exc:   # noqa: BLE001 - FigError of the executed copy
                ok = ok and "H13" in str(exc)
        return ok

    _check(56, c56)

    def c57():
        # Per block: key -> (factor, value in the axis unit). Keys sort
        # alphabetically in another order than by value (friction).
        steps = {"T-sweep-tilt": {"tilt_hi_x0.5": (0.5, 5.0), "tilt_hi_x1": (1.0, 10.0), "tilt_hi_x2": (2.0, 20.0)},
                 "T-sweep-fric": {"friction_hi_x1": (1.0, 0.20), "friction_hi_x2": (2.0, 0.26),
                                  "friction_lo_x1": (1.0, 0.08)}}
        col = {"T-sweep-tilt": "tilt_deg", "T-sweep-fric": "friction"}
        # success per (method, seed): 4 rows per step, k of them successes, k = 4 - step index (value order).
        rows = []
        for block, st_ in steps.items():
            for m, s, drop in (("autodr", 1, 0), ("autodr", 2, 1), ("nodr", 1, 3), ("nodr", 2, 4)):
                for i, (k, (_f, v)) in enumerate(sorted(st_.items(), key=lambda kv: kv[1][1])):
                    kk = max(0, 4 - i - drop)
                    rows += [{"block": block, "tag": "", "boundary_key": k, "method": m, "seed": s,
                              "success_train": int(j < kk), "tilt_deg": 0.0, "friction": 0.14,
                              col[block]: v + 1e-7} for j in range(4)]
        sdf = pd.DataFrame(rows)
        side = {"T-sweep-tilt": {"dim": "tilt", "centres": {"tilt": 0.0, "friction": 0.14},
                                 "blocks": [{"key": k, "factor": f} for k, (f, _v) in steps["T-sweep-tilt"].items()]},
                "T-sweep-fric": {"dim": "friction", "centres": {"tilt": 0.0, "friction": 0.14},
                                 "blocks": [{"key": k, "factor": f} for k, (f, _v) in steps["T-sweep-fric"].items()]}}
        d = ns["data_h14"](sdf, side)
        t, f = d["T-sweep-tilt"], d["T-sweep-fric"]
        ok = (np.allclose(f["x"], [0.08, 0.20, 0.26]) and np.allclose(t["x"], [5.0, 10.0, 20.0])
              and np.allclose(t["range"], (0.0, 10.0)) and np.allclose(f["range"], (0.08, 0.20))
              # autodr at the first step: s1 4/4, s2 3/4 -> mean 0.875; nodr at the last: 0, 0.
              and abs(f["arms"]["autodr"]["mean"][0] - 0.875) < 1e-12 and f["arms"]["nodr"]["mean"][-1] == 0.0
              and list(f["arms"]["autodr"]["seeds"]) == [1, 2] and t["n"] == 4)
        try:
            ns["data_h14"](sdf[sdf["boundary_key"] != "tilt_hi_x2"], side)
            return False
        except Exception as exc:   # noqa: BLE001 - FigError of the executed copy
            return ok and "H14" in str(exc)

    _check(57, c57)

    def c58():
        # Per epoch: 10 episodes, 8 successes, 2 of them >= t1; nodr s2 has no
        # success in epochs 11-20.
        rows = [{"method": m, "seed": s, "epoch": e, "episodes": 10,
                 "successes": 0 if (m, s) == ("nodr", 2) and e > 10 else 8,
                 "successes_ge_t1": 0 if (m, s) == ("nodr", 2) and e > 10 else 2, "t1_mm": 0.2938}
                for m in METHODS for s in (1, 2, 3) for e in range(1, 21)]
        pen = pd.DataFrame(rows)
        d = ns["data_h15"](pen)
        a, nd = d["arms"]["autodr"], d["arms"]["nodr"]
        s1, n2 = a["seeds"][1], nd["seeds"][2]
        ok = (list(s1["x"]) == [10, 20] and np.allclose(s1["clean"], 0.6) and np.allclose(s1["share_ge"], 0.25)
              and np.isnan(n2["share_ge"][1]) and n2["clean"][1] == 0.0 and list(d["arms"]) == list(METHODS)
              and nd["share"]["left_out"] == [2] and nd["share"]["n"] == 2 and np.allclose(nd["share"]["mean"], 0.25)
              # clean keeps s2: block 2 = (0.6 + 0 + 0.6) / 3 = 0.4
              and nd["clean"]["n"] == 3 and np.allclose(nd["clean"]["mean"], [0.6, 0.4]))
        try:
            ns["data_h15"](pen.assign(t1_mm=[0.2938] * (len(pen) - 1) + [0.3]))
            return False
        except Exception as exc:   # noqa: BLE001 - FigError of the executed copy
            return ok and "t1" in str(exc)

    _check(58, c58)

    def c59():
        def pol(m, seed, ref, rate):
            return {"policy": {"method": m, "seed": seed}, "reference": {"rate": ref},
                    "blocks": {k: {"rate": rate, "n": 150, "wilson": [rate - 0.05, min(rate + 0.05, 1.0)]}
                               for k in ("a", "b")}}
        # Out of seed order on purpose; autodr s4 sits exactly on the rule,
        # autodr s5 has a block rate under the rule (the rule must not read it).
        pols = {"fixeddr_s3": pol("fixeddr", 3, 0.5, 0.4), "fixeddr_s4": pol("fixeddr", 4, 0.98, 0.7),
                "fixeddr_s1": pol("fixeddr", 1, 0.0, 0.0), "fixeddr_s2": pol("fixeddr", 2, 0.95, 0.9),
                **{f"autodr_s{s}": pol("autodr", s, r, b) for s, r, b in
                   ((5, 0.93, 0.6), (1, 0.99, 0.95), (2, 0.97, 0.9), (3, 0.95, 0.85), (4, 0.90, 0.92))}}
        d = ns["data_learned"](pols, ["a", "b"], ["A", "B"], "H19")
        a, f = d["arms"]["autodr"], d["arms"]["fixeddr"]
        ok = (a["seeds"] == [1, 2, 3, 4, 5] and f["seeds"] == [2, 4] and "mean" in a and "mean" not in f
              and np.allclose(a["mean"], [0.844, 0.844]) and np.allclose(f["rates"][:, 0], [0.9, 0.7])
              and d["left_out"] == {"autodr": [], "fixeddr": [(1, 0.0), (3, 0.5)]}
              and np.allclose(d["ref_range"], (0.90, 0.99)) and d["n"] == 150)
        bad = {k: v for k, v in pols.items() if k not in ("fixeddr_s2", "fixeddr_s4")}
        try:
            ns["data_learned"](bad, ["a", "b"], ["A", "B"], "H19")
            return False
        except Exception as exc:   # noqa: BLE001 - FigError of the executed copy
            return ok and "H19" in str(exc) and "fixeddr" in str(exc)

    _check(59, c59)

    def c60():
        def pol(m, seed, ref, rate):
            return {"policy": {"method": m, "seed": seed}, "reference": {"rate": ref},
                    "blocks": {t: {"rate": rate, "n": 1000, "wilson": [max(rate - 0.03, 0.0), min(rate + 0.03, 1.0)],
                                   "changed": {c: v}} for t, c, v in
                               (("pn2", "pos_noise_m", 0.01), ("gx2", "grasp_x_m", 0.006), ("py2", "yaw_noise_rad", 0.0447))}}
        refs = {"autodr": (0.99, 0.98, 0.97, 0.96, 0.95), "fixeddr": (0.0, 0.0, 0.98, 0.99, 0.96)}
        pols = {f"{m}_s{s}": pol(m, s, r, r) for m, rs in refs.items() for s, r in zip((5, 1, 3, 2, 4), rs)}
        d = ns["data_h22"]({"NOISE": {"policies": pols}})
        f = d["arms"]["fixeddr"]
        ok = (f["seeds"] == [1, 2, 3, 4, 5] and "mean" in f and np.isclose(f["ref_mean"], 0.586)
              and np.isclose(d["arms"]["autodr"]["ref_mean"], 0.97) and d["n"] == 1000
              and d["labels"][0].endswith("±10 mm"))
        few = {k: v for k, v in pols.items() if k != "fixeddr_s1"}
        try:
            ns["data_h22"]({"NOISE": {"policies": few}})
            return False
        except Exception as exc:   # noqa: BLE001 - FigError of the executed copy
            return ok and "H22" in str(exc) and "fixeddr" in str(exc)

    _check(60, c60)

    def c61():
        def pol(m, seed, k):
            # The block sits 0.05 under the main test, so a mean read off the block is caught.
            ref, rate = k / 1000, max(k / 1000 - 0.05, 0.0)
            return {"policy": {"method": m, "seed": seed}, "reference": {"rate": ref, "k": k, "n": 1000},
                    "blocks": {t: {"rate": rate, "n": 1000, "wilson": [max(rate - 0.03, 0.0), min(rate + 0.03, 1.0)],
                                   "changed": {c: v}} for t, c, v in
                               (("pn2", "pos_noise_m", 0.01), ("gx2", "grasp_x_m", 0.006), ("py2", "yaw_noise_rad", 0.0447))}}
        ks = {"autodr": (990, 980, 970, 960, 950), "fixeddr": (0, 0, 980, 990, 50)}
        pols = {f"{m}_s{s}": pol(m, s, k) for m, kk in ks.items() for s, k in zip((1, 2, 3, 4, 5), kk)}
        d = ns["data_h20_pairs"]({"NOISE": {"policies": pols}})
        a, f = d["arms"]["autodr"], d["arms"]["fixeddr"]
        return (np.allclose(a["ref_wilson"][0], ES.wilson(990, 1000)) and f["seeds"] == [3, 4]
                and np.allclose(f["ref_wilson"][1], ES.wilson(990, 1000)) and np.isclose(a["ref_mean"], 0.97)
                and a["ref_ci"][0] < 0.97 < a["ref_ci"][1] and "ref_ci" not in f)

    _check(61, c61)

    def c62():
        def bars(rates, ci):
            return {"autodr": {"ref": {"rates": [0.99, 0.98], "mean": 0.985, "ci": (0.98, 0.99)},
                               "pn2": {"rates": rates, "mean": float(np.mean(rates)), "ci": ci}},
                    "fixeddr": {"ref": {"rates": [0.9, 0.8], "mean": None, "ci": None}}}

        ns["_pair_floor"](bars([0.80, 0.90], (0.80, 0.90)), 80, "H20")
        for low in (bars([0.79, 0.90], (0.80, 0.90)), bars([0.85, 0.90], (0.795, 0.90))):
            try:
                ns["_pair_floor"](low, 80, "H20")
                return False
            except Exception as exc:   # noqa: BLE001 - FigError of the executed copy
                if "H20" not in str(exc) or "autodr" not in str(exc):
                    return False
        return True

    _check(62, c62)
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
        # Fixed-DR training runs: all five -> extracted; four -> refused.
        _write_rt_logs(t / "rt_f", runs={**TRAINING_RUNS, **FIXED_RUNS})
        try:
            got = ns["extract_training"](t / "rt_f", t / "out_f")
            res[CHECKS[23]] = got["rows"] == 2 * (len(TRAINING_RUNS) + len(FIXED_RUNS))
        except Exception:   # noqa: BLE001
            res[CHECKS[23]] = False
        # three arms, three factors (ARM_STEPS x 4 envs); each row on its own.
        try:
            cv = pd.read_csv(t / "out_f" / "curves.csv")
            want = (cv["iteration"] + 1) * cv["method"].map(ARM_STEPS) * 4
            res[CHECKS[30]] = set(cv["method"]) == set(ARMS) and bool((cv["env_steps"] == want).all())
        except Exception:   # noqa: BLE001
            res[CHECKS[30]] = False
        shutil.rmtree(t / "rt_f" / FIXED_RUNS[("fixeddr", 5)])
        try:
            ns["extract_training"](t / "rt_f", t / "out_f4")
            res[CHECKS[23]] = False
        except Exception as exc:   # noqa: BLE001
            res[CHECKS[23]] = res[CHECKS[23]] and "Fixed-DR" in str(exc)
    return res


MUTATIONS = (
    ("f1-mean-drops-last-seed", '"mean": float(np.mean(rates)), "ci"', '"mean": float(np.mean(rates[:-1])), "ci"',
     (CHECKS[0], CHECKS[14])),
    ("f2-blocks-swapped", '(("K2", "T-nominal"), ("K1", "T-final"))', '(("K1", "T-nominal"), ("K2", "T-final"))',
     (CHECKS[1], CHECKS[17])),
    # An empty bin has rate None, which _mean_std turns into NaN without
    # raising: only the bin-count check sees the kept empty bins.
    ("f3-keeps-empty-bins", "keep = [i for i, n in enumerate(ns) if n > 0]",
     "keep = list(range(len(ns)))", (CHECKS[2],)),
    # One seed left: the spread helper refuses, the whole of data_f3 goes red.
    ("f3-band-from-first-seed-only", '_mean_std([seeds[k] for k in keys], "F3")',
     '_mean_std([seeds[k] for k in keys[:1]], "F3")', (CHECKS[2], CHECKS[3], CHECKS[18], CHECKS[26], CHECKS[44], CHECKS[52])),
    ("f4-band-from-wilson-n-minus-1", '"overall_wilson": ES.wilson(ov["k"], ov["n"])',
     '"overall_wilson": ES.wilson(ov["k"], ov["n"] - 1)', (CHECKS[4],)),
    ("f5-drops-timeouts", '    ("timeout", "Zeitlimit", "#ec835a"),\n', "", (CHECKS[5], CHECKS[19], CHECKS[41])),
    ("f6-counts-all-episodes-in-ecdf", "ok = g[g[KEY] == 1]\n        ecdf, no_success = {}, []",
     "ok = g\n        ecdf, no_success = {}, []", (CHECKS[6], CHECKS[20], CHECKS[27], CHECKS[29])),
    # The spread helper _mean_std is the one home of F3/F6/F7 band numbers.
    ("spread-mean-is-max", "return a.mean(axis=0), a.std(axis=0, ddof=1)",
     "return a.max(axis=0), a.std(axis=0, ddof=1)",
     (CHECKS[3], CHECKS[6], CHECKS[7], CHECKS[18], CHECKS[20], CHECKS[24], CHECKS[29], CHECKS[42])),
    ("spread-std-population-not-sample", "return a.mean(axis=0), a.std(axis=0, ddof=1)",
     "return a.mean(axis=0), a.std(axis=0, ddof=0)", (CHECKS[25], CHECKS[26], CHECKS[27], CHECKS[42])),
    ("f8-lo-side-over-hi-max", 'sides["lo"] = (v - d["centre"]) / (d["lo_max"] - d["centre"])',
     'sides["lo"] = (v - d["centre"]) / (d["hi_max"] - d["centre"])', (CHECKS[8],)),
    ("f9-counts-episodes-not-rows", 'rows[a, b] = g.loc[sel, "row_id"].nunique()',
     "rows[a, b] = int(sel.sum())", (CHECKS[9], CHECKS[21])),
    ("f10-counts-every-episode", "(ok & (ip < t)).sum()", "(ip < t).sum()", (CHECKS[10], CHECKS[22], CHECKS[54])),
    ("colours-swapped", 'METHOD_COLOR = {"autodr": "#3B6FD4", "nodr": "#D97706", "fixeddr": "#199e70"}',
     'METHOD_COLOR = {"autodr": "#3B6FD4", "nodr": "#199e70", "fixeddr": "#D97706"}', (CHECKS[11],)),
    ("extraction-drops-the-equality-guard", "        if v != first:\n",
     "        if False:\n", (CHECKS[12],)),
    ("figure-numbers-keep-the-point", 'return s.replace(".", ",")', "return s", (CHECKS[13],)),
    ("style-keeps-the-default-tick-formatter", "            axis.set_major_formatter(_de_formatter())\n",
     "            pass\n", (CHECKS[13],)),
    # Fixed-DR: its own blocks, optional, both or none.
    # data_f1 then finds no Fixed-DR run and raises: both F1 checks go red.
    ("f1-reads-fixed-dr-from-the-pair-block", "_policies(summary, ARM_BLOCK[block])", "_policies(summary, block)",
     (CHECKS[0], CHECKS[14], CHECKS[19], CHECKS[41], CHECKS[54])),
    ("fixed-dr-always-required", "    if not blocks:\n        return METHODS", "    if False:\n        return METHODS",
     (CHECKS[15], CHECKS[17])),
    ("half-fixed-dr-accepted", "    if blocks == set(ARM_BLOCK.values()):", "    if blocks:",
     (CHECKS[16],)),
    # a Fixed-DR run in T-auswahl / T-ood would count as a third -F block.
    ("selection-block-counts-as-fixed-dr-block", "    blocks = set(blocks) & set(ARM_BLOCK.values())",
     "    blocks = set(blocks)", (CHECKS[32],)),
    ("f2-k5-rows-swapped", '(("K5_nominal", "T-nominal"), ("K5", "T-final"))',
     '(("K5", "T-nominal"), ("K5_nominal", "T-final"))', (CHECKS[1], CHECKS[17])),
    ("f3-ignores-fixed-dr", 'k.startswith(m + "_s")) for m in arms_present(summary)}',
     'k.startswith(m + "_s")) for m in METHODS}', (CHECKS[18],)),
    ("f5-ignores-fixed-dr", "    for p in _arm_policies(summary, \"T-final\"):", "    for p in _policies(summary, \"T-final\"):",
     (CHECKS[19], CHECKS[41])),
    ("final-ignores-the-fixed-dr-block", 'df["block"].isin(("T-final", ARM_BLOCK["T-final"]))',
     'df["block"].isin(("T-final",))',
     (CHECKS[6], CHECKS[10], CHECKS[20], CHECKS[21], CHECKS[22], CHECKS[27], CHECKS[29], CHECKS[54], CHECKS[55])),
    ("f9-ignores-fixed-dr", 'list(te), "methods": {}}\n    for m in arms_in(df):', 'list(te), "methods": {}}\n    for m in METHODS:',
     (CHECKS[21],)),
    ("f10-ignores-fixed-dr", '"t1": t1_mm, "methods": {}, "ci": {}}\n    for m in arms_in(df):',
     '"t1": t1_mm, "methods": {}, "ci": {}}\n    for m in METHODS:', (CHECKS[22], CHECKS[54], CHECKS[55])),
    ("extraction-takes-four-fixed-dr-runs", "    if have and len(have) != len(FIXED_RUNS):", "    if False:",
     (CHECKS[23],)),
    ("f7-ignores-fixed-dr", "    for m in ARMS:\n        g = curves[", "    for m in METHODS:\n        g = curves[",
     (CHECKS[24],)),
    ("yerr-keeps-rounding", "[[100 * max(0.0, lo)], [100 * max(0.0, hi)]]", "[[100 * lo], [100 * hi]]",
     (CHECKS[28],)),
    ("f6-refuses-a-seed-without-success", "                no_success.append(int(s))\n                continue",
     '                raise FigError("F6: no success")', (CHECKS[29],)),
    # No-DR and Fixed-DR read the AutoDR run's params files.
    ("one-arm-with-the-other-arms-factor",
     'factor = (run_axis.read_factor(agent, src / run / f"{run}_env.yaml") if agent.is_file()',
     'factor = (run_axis.read_factor(*(src / TRAINING_RUNS[("autodr", s)] / f"{TRAINING_RUNS[(\'autodr\', s)]}_{k}.yaml"'
     ' for k in ("agent", "env"))) if agent.is_file()', (CHECKS[30],)),
    ("f7-drops-the-grid-guard", "            elif not np.array_equal(x, xs):", "            elif False:",
     (CHECKS[31],)),
    # F11/F12 (ported from p6 7d008ff1 / db85986c).
    # The start-height cap (x1.2) read as x1.5: its x1.0 value becomes 0.10 m.
    ("f11-edge-ignores-the-block-factor", 'edge = c + (b["physical"] - c) / b["factor"]',
     'edge = c + (b["physical"] - c) / sidecar["factor"]', (CHECKS[33],)),
    ("f11-two-references-pass", 'if (ov["k"], ov["n"]) != (ref["k"], ref["n"]):', "if False:", (CHECKS[33],)),
    # The rate mean (about 0.6) drawn on a difference axis.
    ("f12-mean-reads-the-rate-mean", '"mean": [a["blocks"][key]["diff_mean"] for key in K3_ORDER]',
     '"mean": [a["blocks"][key]["mean"] for key in K3_ORDER]', (CHECKS[34],)),
    ("f12-interval-is-wilson", '[p["blocks"][key]["newcombe_ci"] for p in pols]',
     '[p["blocks"][key]["wilson"] for p in pols]', (CHECKS[34],)),
    ("f12-drops-fixed-dr", "arms = [m for m in ARMS if m in across]", "arms = [m for m in METHODS if m in across]",
     (CHECKS[34],)),
    ("ood-table-guard-off", 'if shas != {sidecar["sha256"]}:', "if False:", (CHECKS[35],)),
    # rl_games extraction: rsl_rl's (it + 1) x F applied to an epoch.
    ("rlg-x-by-rsl-rl-rule", 'run_axis.rl_games_x("Episode/success_rate", step, frames)', "(step + 1) * frames",
     (CHECKS[36],)),
    # Without the Episode/ candidate No-DR's start height reads "" and the extraction raises.
    ("rlg-drops-episode-prefix", 'for c in (tag, "Episode/" + tag):', "for c in (tag,):", (CHECKS[36], CHECKS[45])),
    ("rlg-epoch-guard-off", "        run_axis.check_rl_games_epochs(ep_pairs, frames)\n", "", (CHECKS[36],)),
    ("rlg-sidecar-run-guard-off", 'if side.get("seed") != s or f"__{prefix}{s}_" not in side.get("run", ""):',
     "if False:", (CHECKS[37],)),
    ("rlg-sidecars-partial-accepted", "    if len(have) != len(found):", "    if False:", (CHECKS[37],)),
    ("rlg-sidecar-epoch-unchecked", "if not m or int(m.group(1)) != last_epoch[s]:", "if not m:", (CHECKS[37],)),
    ("load-training-reads-only-1499", 'folder.glob("autodr_s*_autodr_*.json")', 'folder.glob("autodr_s*_autodr_1499.json")',
     (CHECKS[38],)),
    ("f8-missing-seed-unnamed", 'side = sidecars.get(("autodr", int(s)))', 'side = sidecars[("autodr", int(s))]',
     (CHECKS[38],)),
    ("only-unavailable-passes", "missing = {k: available[k] for k in only if available[k]}", "missing = {}",
     (CHECKS[39],)),
    ("f4-reference-ignores-fixed-dr", '"reference_block": ES.final_block(pol["method"])', '"reference_block": "T-final"',
     (CHECKS[40],)),
    ("h4-takes-the-first-seed-only", 'rs = [r for r in rows if r["method"] == m]',
     'rs = [r for r in rows if r["method"] == m][:1]', (CHECKS[41],)),
    # H5/H6 and the friction panel.
    ("h5-band-clipped-like-a-rate", '    return _seed_curves(curves, RETURN_TAG, "H5")',
     '    return {m: dict(v, mean=np.clip(v["mean"], 0.0, 1.0)) for m, v in _seed_curves(curves, RETURN_TAG, "H5").items()}',
     (CHECKS[42],)),
    ("iteration-axis-accepts-two-factors", "    if len(factors) != 1:", "    if False:", (CHECKS[43],)),
    ("bin-zero-starts-at-0", "else float(edges[0]) - (float(edges[1]) - float(edges[0]))", "else 0.0", (CHECKS[44],)),
    ("nodr-friction-band-accepted", "        if len(fr) > 1:", "        if False:", (CHECKS[45],)),
    # the best of all 15 runs handed to every arm (Fixed-DR then gets autodr's seed 1).
    ("best-policy-over-all-arms", '        mine = [p for p in summary["policies"] if p["method"] == m]',
     '        mine = list(summary["policies"])', (CHECKS[46], CHECKS[47])),
    ("h7-reads-the-next-seed", '        pkey = f"{m}_s{seed}"', '        pkey = f"{m}_s{seed + 1}"', (CHECKS[47], CHECKS[53])),
    ("h8-counts-failures", "            won = g[g[KEY] == 1]", "            won = g", (CHECKS[48],)),
    # H12.
    # The other arm's mean then lies outside this arm's interval: check 54 sees it too.
    ("h12-mean-of-the-other-arm", '"mean": b[f"mean_{arm}"],', '"mean": b[f"mean_{ES.FORCE_ARMS[0]}"],',
     (CHECKS[49], CHECKS[54])),
    # pn2 drawn from the T-final rates; a missing pn2 block then passes too.
    ("h12-every-block-reads-T-final", 'b = fo["blocks"].get(label, {})', 'b = fo["blocks"].get("T-final", {})',
     (CHECKS[49], CHECKS[50])),
    ("h12-accepts-a-missing-block", '        if "per_seed" not in b:\n            raise FigError(f"H12',
     '        if False:\n            raise FigError(f"H12', (CHECKS[50],)),
    ("h12-ablation-takes-the-autodr-blue", 'ABLATION_COLOR = "#d55181"', 'ABLATION_COLOR = METHOD_COLOR["autodr"]',
     (CHECKS[49],)),
    # the bootstrap interval.
    ("boot-ci-at-90-percent", 'confidence_level=1.0 - ES.ALPHA, method="percentile"',
     'confidence_level=0.90, method="percentile"', (CHECKS[51],)),
    ("h2-ci-is-mean-plus-sd", 'lo, hi = _boot_ci([seeds[k] for k in keys], "H2")', "lo, hi = mean - std, mean + std",
     (CHECKS[52],)),
    ("h5-h6-ci-is-mean-plus-sd", '"ci": _boot_ci(list(seeds.values()), fig)}', '"ci": (mean - std, mean + std)}',
     (CHECKS[52],)),
    ("h7-reference-wilson-from-n-minus-1", 'ES.wilson(nz["reference"]["k"], nz["reference"]["n"])',
     'ES.wilson(nz["reference"]["k"], nz["reference"]["n"] - 1)', (CHECKS[53],)),
    ("h7-noise-reads-the-ood-wilson", '"noise_wilson": [nz["blocks"][t]["wilson"] for t, _l in H7_NOISE]',
     '"noise_wilson": [oo["blocks"][K3_ORDER[0]]["wilson"] for t, _l in H7_NOISE]', (CHECKS[53],)),
    ("f1-ci-is-the-lowest-seed", '"ci": [float(v) for v in _boot_ci(rates, "F1")]', '"ci": [min(rates), min(rates)]',
     (CHECKS[54],)),
    ("f6-ci-collapses-on-the-mean", '"ci": _boot_ci(list(seeds.values()), "F6")}', '"ci": (mean, mean)}',
     (CHECKS[54],)),
    ("f10-ci-collapses-on-the-mean", 'out["ci"][m] = _boot_ci(seeds, "F10")',
     'out["ci"][m] = (out["methods"][m], out["methods"][m])', (CHECKS[54],)),
    ("h12-ci-spans-everything", '"ci": [float(v) for v in _boot_ci(rates, "H12")]', '"ci": [0.0, 1.0]',
     (CHECKS[54],)),
    ("f10-keeps-only-the-first-seed", 'out.setdefault("seeds", {})[m] = seeds   #',
     'out.setdefault("seeds", {})[m] = seeds[:1]   #', (CHECKS[55],)),
    # H13.
    ("h13-reference-is-the-pn2-block", '"ref": one([p["reference"]["rate"] for p in ps])',
     '"ref": one([p["blocks"]["pn2"]["rate"] for p in ps])', (CHECKS[56],)),
    ("h13-seeds-in-file-order", 'ps = sorted(per[m], key=lambda p: p["policy"]["seed"])', "ps = per[m]",
     (CHECKS[56],)),
    ("h13-accepts-a-missing-block", '        if missing:\n            raise FigError(f"H13',
     '        if False:\n            raise FigError(f"H13', (CHECKS[56],)),
    # RT-264/265: H14.
    ("h14-steps-in-key-order", "order = sorted(xs, key=xs.get)", "order = sorted(xs)", (CHECKS[57],)),
    ("h14-range-without-the-centre", ' + [unit(side["centres"][side["dim"]])]', "", (CHECKS[57],)),
    ("h14-accepts-another-step-set", 'if set(g["boundary_key"]) != set(steps):', "if False:", (CHECKS[57],)),
    # H15.
    ("h15-clean-counts-every-success", '"clean": (k - kg) / n,', '"clean": k / n,', (CHECKS[58],)),
    ("h15-share-over-episodes", "np.divide(kg, k, out=", "np.divide(kg, n, out=", (CHECKS[58],)),
    ("h15-share-keeps-seeds-without-success", 'used = [s for s, v in seeds.items() if np.isfinite(v["share_ge"]).all()]',
     "used = list(seeds)", (CHECKS[58],)),
    # H19/H20.
    ("learned-rule-strict", 'keep = [p for p in ps if p["reference"]["rate"] >= min_rate]',
     'keep = [p for p in ps if p["reference"]["rate"] > min_rate]', (CHECKS[59], CHECKS[60])),
    ("learned-rule-reads-the-block", 'keep = [p for p in ps if p["reference"]["rate"] >= min_rate]',
     'keep = [p for p in ps if p["blocks"][keys[0]]["rate"] >= min_rate]', (CHECKS[59], CHECKS[61])),
    ("learned-mean-for-two", "if len(keep) >= LEARNED_MEAN_MIN_N:", "if len(keep) >= 2:",
     (CHECKS[59], CHECKS[60], CHECKS[61])),
    ("learned-accepts-an-empty-arm", "        if not keep:\n            raise FigError(",
     "        if False:\n            raise FigError(", (CHECKS[59],)),
    # H22.
    ("h22-keeps-the-rule", 'd = data_learned(*_noise_blocks(summary, "H22"), "H22", min_rate=0.0)',
     'd = data_learned(*_noise_blocks(summary, "H22"), "H22")', (CHECKS[60],)),
    ("h22-accepts-an-arm-without-a-mean", '        if "mean" not in a:\n            raise FigError(f"H22',
     '        if False:\n            raise FigError(f"H22', (CHECKS[60],)),
    # H20 as pairs.
    ("h20-ref-wilson-from-n-minus-1", 'ES.wilson(r["k"], r["n"]) for r in refs',
     'ES.wilson(r["k"], r["n"] - 1) for r in refs', (CHECKS[61],)),
    ("h20-ref-mean-is-the-block-mean", 'a["ref_mean"] = float(np.mean(a["ref"]))\n            lo, hi',
     'a["ref_mean"] = float(np.mean(a["mean"]))\n            lo, hi', (CHECKS[61],)),
    # PAIR_YAXIS floor.
    ("pair-floor-ignores-the-interval", 'low = min(list(r["rates"]) + ([r["ci"][0]] if r["ci"] is not None else []))',
     'low = min(r["rates"])', (CHECKS[62],)),
    ("pair-floor-never-refuses", "if 100 * low < ymin:", "if False:", (CHECKS[62],)),
)
_SELF_TEST_MARKER = "# Self-test: checks on built data"


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
    ap.add_argument("--results", type=str, default=str(REPO / "daten" / "endstudie_rl_games" / "evaluation"))
    ap.add_argument("--training", type=str, default=None)
    ap.add_argument("--out", type=str, default="docs/figures/eval")
    ap.add_argument("--only", type=str, default="", help="comma list, e.g. F1,F9")
    ap.add_argument("--sweep", type=str, default=None,
                    help="root of the tilt/friction sweep (RT-264/265), H14 only")
    ap.add_argument("--arm", action="append", default=[], metavar="ARM=PREFIX",
                    help="with --extract-training: an rl_games seed chain, e.g. autodr=RT-242s (repeat per arm)")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.training is None:
        args.training = "results/training" if args.extract_training else str(
            REPO / "daten" / "endstudie_rl_games" / "training_auszug")
    print(f"[eval_figures] marker: {SCRIPT_MARKER}")
    if args.arm and not args.extract_training:
        raise SystemExit("--arm belongs to --extract-training")
    if args.extract_training:
        if args.arm:
            arms = dict(a.partition("=")[::2] for a in args.arm)
            if len(arms) != len(args.arm) or not all(arms.values()):
                raise SystemExit(f"--arm {args.arm}: want <arm>=<prefix>, each arm once")
            got = extract_training_rl_games(pathlib.Path(args.extract_training), pathlib.Path(args.training), arms)
            side = [p.name for p in got["sidecars"]] or "none -- F8 waits for the AutoDR state (Part C)"
            print(f"[eval_figures] AutoDR states: {side}")
            print(f"[eval_figures] {PEN_FILE}: {got['penetration_rows'] or 'none -- no episodes.csv, H15 waits'} rows")
        else:
            got = extract_training(pathlib.Path(args.extract_training), pathlib.Path(args.training))
        print(f"[eval_figures] extracted {got['rows']} iteration rows -> {args.training}; No-DR: {got['nodr']}")
        return 0
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    for p in draw_all(pathlib.Path(args.results), pathlib.Path(args.training), pathlib.Path(args.out), only,
                      pathlib.Path(args.sweep) if args.sweep else None):
        print(f"[eval_figures] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
