# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Compute evaluation statistics from ``episodes_all.csv``.

Headline key is ``success_train``. Library functions, no own interval code:

  Wilson          statsmodels proportion_confint(method="wilson")
  Newcombe (K3)   statsmodels confint_proportions_2indep(method="newcomb", compare="diff")
  K1/K2 headline  scipy ttest_ind(equal_var=False).confidence_interval  (Welch)
  K1/K2 beside    scipy bootstrap(paired=False, method="percentile")    (fixed seed)
  K5              the same two, AutoDR (T-final / T-nominal) - Fixed-DR (T-final-F /
                  T-nominal-F); a reading (above_0 / below_0 / contains_0), no "met"
  BONFERRONI      the same Welch call at 1 - ALPHA / k over K1, K2, K5, K5_nominal;
                  cross-check beside the pre-registered 95 % reading, no "met"

The unit of the method comparison is the seed: each policy contributes one rate. Percentiles of force, steps and interpenetration
use the env's nearest-rank rule (`insertion_env._pct`), so an evaluation result and a
training-log result mean the same. The interpenetration threshold t1 is read from
the runs' ``episodes_meta.json`` (one value over all blocks, else refused).

Writes ``summary.json`` (sorted keys; two runs on the same input are byte-equal)
and ``summary_tables.csv`` (one row per policy and block).

    python scripts/eval_stats.py --results results/eval
    python scripts/eval_stats.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

SCRIPT_MARKER = "eval_stats-2026-10-06"
ALPHA = 0.05
BOOT_RESAMPLES = 10000
BOOT_SEED = 20260921
KEY = "success_train"

# Use the env's bin edges and first-upper-edge rule; the last bin includes larger values.
# Friction uses five equal bins over 0.08..0.20.
BIN_EDGES = {
    "lat_r_mm": (6.0, 12.0, 18.0, 24.0, 30.0),
    "yaw_abs_deg": (1.0, 2.0, 3.0, 4.0, 5.0),
    "tilt_deg": (2.0, 4.0, 6.0, 8.0, 10.0),
    "start_height_m": (-0.020, -0.010, 0.0, 0.010, 0.020, 0.070, 0.120, 0.170, 0.220, 0.270),
    "friction": (0.104, 0.128, 0.152, 0.176, 0.200),
}


def _load_module(name: str, path: pathlib.Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The noise tags and their widths have ONE home: check_eval_conditions.NOISE_TAGS.
CEC = _load_module("check_eval_conditions", pathlib.Path(__file__).resolve().parent / "check_eval_conditions.py")
# the reference of the OOD and noise blocks is the policy's UNTAGGED
# T-final run of the same results folder -- one code commit for the whole
# rl_games evaluation (was the tagged repeat "p6" on p6-TestAusserhalb).
NOISE_REFERENCE_TAG = ""


class StatsError(RuntimeError):
    """The input is insufficient for the requested statistic."""


def wilson(k: int, n: int) -> list:
    lo, hi = proportion_confint(int(k), int(n), alpha=ALPHA, method="wilson")
    return [float(lo), float(hi)]


def pct_nearest_rank(values, q: float) -> float | None:
    """`insertion_env._pct`: sorted[min(n-1, ceil(q*n)-1)]; None when empty."""
    v = sorted(float(x) for x in values)
    if not v:
        return None
    return v[min(len(v) - 1, int(math.ceil(q * len(v))) - 1)]


def policy_rows(df: pd.DataFrame, t1_mm: float | None = None) -> list:
    """One dict per (block, method, seed, tag): rates with Wilson, outcomes,
    force, steps and interpenetration over the SUCCESSES only."""
    out = []
    for (block, method, seed, tag), g in df.groupby(["block", "method", "seed", "tag"], sort=True):
        n = int(len(g))
        k = int(g[KEY].sum())
        k_strict = int(g["success_strict"].sum())
        k_abort = int(g["force_abort"].sum())
        ok = g[g[KEY] == 1]
        ip = ok["ip_max_mm"]
        out.append({
            "block": block, "method": method, "seed": int(seed), "tag": tag, "n": n,
            "success_train": k / n, "success_train_k": k, "success_train_wilson": wilson(k, n),
            "success_strict": k_strict / n, "success_strict_wilson": wilson(k_strict, n),
            "force_abort_rate": k_abort / n, "force_abort_wilson": wilson(k_abort, n),
            "outcomes": {
                "success": int(((g["outcome"] == "success") & (g["force_abort"] == 0)).sum()),
                "success_with_force_violation": int(((g["outcome"] == "success") & (g["force_abort"] == 1)).sum()),
                "force_abort": int((g["outcome"] == "force_abort").sum()),
                "timeout": int((g["outcome"] == "timeout").sum()),
                "other": int((g["outcome"] == "other").sum()),
            },
            "successes_force_filtered_n": {
                "median": float(ok["force_max_filtered_n"].median()) if len(ok) else None,
                "p95": pct_nearest_rank(ok["force_max_filtered_n"], 0.95),
                "max": float(ok["force_max_filtered_n"].max()) if len(ok) else None,
            },
            "successes_steps": {
                "median": float(ok["steps"].median()) if len(ok) else None,
                "p95": pct_nearest_rank(ok["steps"], 0.95),
                "max": int(ok["steps"].max()) if len(ok) else None,
            },
            # `ip_max_mm` is the maximum over ALL policy steps of the episode; the
            # success gate checks the same SAPU value only in the success step
            # (insertion_math: interpen < interpen_thresh = t1). n_ge_t1 counts the
            # successes that reached t1 earlier in contact (IndustReal Fig. 3 view, F10).
            "successes_ip_max_mm": {
                "median": float(ip.median()) if len(ok) else None,
                "p95": pct_nearest_rank(ip, 0.95),
                "max": float(ip.max()) if len(ok) else None,
                "n_ge_t1": int((ip >= t1_mm).sum()) if t1_mm is not None else None,
            },
        })
    return out


def t1_from_results(results: pathlib.Path, blocks) -> float:
    """t1 (mm) from ``<results>/<block>/*/episodes_meta.json`` of the given blocks;
    the same value in every file, else refused."""
    vals = {json.loads(p.read_text(encoding="utf-8"))["interpenetration"]["t1_mm"]
            for b in sorted(blocks) for p in sorted((results / b).glob("*/episodes_meta.json"))}
    if len(vals) != 1:
        raise StatsError(f"episodes_meta.json files disagree on t1_mm (or none): {sorted(vals)}")
    return float(next(iter(vals)))


def method_diff(a, b, names=("autodr", "nodr")) -> dict:
    """``names[0] - names[1]`` over the per-seed rates (K1/K2: AutoDR - No-DR,
    K5: AutoDR - Fixed-DR). Welch is the headline (Colas et al. 2018 Sec. 6);
    the percentile bootstrap stands beside it (Sec. 3.2)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise StatsError(f"need >= 2 seeds per method, got {len(a)} and {len(b)}")
    welch = st.ttest_ind(a, b, equal_var=False)
    w_lo, w_hi = welch.confidence_interval(confidence_level=1.0 - ALPHA)

    def _diff(x, y, axis=-1):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    boot = st.bootstrap((a, b), _diff, paired=False, vectorized=True,
                        n_resamples=BOOT_RESAMPLES, confidence_level=1.0 - ALPHA,
                        method="percentile", rng=np.random.default_rng(BOOT_SEED))
    return {
        "n_seeds": [int(len(a)), int(len(b))],
        f"mean_{names[0]}": float(a.mean()), f"mean_{names[1]}": float(b.mean()),
        f"sd_{names[0]}": float(a.std(ddof=1)), f"sd_{names[1]}": float(b.std(ddof=1)),
        "diff": float(a.mean() - b.mean()),
        "welch_ci": [float(w_lo), float(w_hi)], "welch_df": float(welch.df),
        "bootstrap_percentile_ci": [float(boot.confidence_interval.low),
                                    float(boot.confidence_interval.high)],
        "bootstrap": {"resamples": BOOT_RESAMPLES, "seed": BOOT_SEED, "method": "percentile"},
    }


def seed_rates(pol: list, block: str, method: str) -> list:
    """Per-seed rates of one method in one block, untagged runs only, seeds sorted."""
    rows = [p for p in pol if p["block"] == block and p["tag"] == "" and p["method"] == method]
    return [p[KEY] for p in sorted(rows, key=lambda p: p["seed"])]


def method_rates(pol: list, block: str) -> tuple[list, list]:
    return seed_rates(pol, block, "autodr"), seed_rates(pol, block, "nodr")


# K5: AutoDR's rates from
# the block K1/K2 already read, Fixed-DR's from its own -F block (same table).
K5_BLOCKS = (("K5", "T-final", "T-final-F"), ("K5_nominal", "T-nominal", "T-nominal-F"))


def k5_reading(ci) -> str:
    """The three pre-registered readings for the evaluation; no "met"."""
    lo, hi = ci
    return "above_0" if lo > 0.0 else ("below_0" if hi < 0.0 else "contains_0")


def k5(pol: list, blocks: set) -> dict:
    out = {}
    for kname, ablock, fblock in K5_BLOCKS:
        if not {ablock, fblock} <= blocks:
            out[kname] = {"missing": f"needs {ablock} and {fblock} runs"}
            continue
        a = seed_rates(pol, ablock, "autodr")
        f = seed_rates(pol, fblock, "fixeddr")
        try:
            d = method_diff(a, f, names=("autodr", "fixeddr"))
        except StatsError as exc:
            out[kname] = {"error": str(exc), "blocks": [ablock, fblock]}
            continue
        d["blocks"] = [ablock, fblock]
        d["reading"] = k5_reading(d["welch_ci"])
        out[kname] = d
    return out


# Bonferroni cross-check of the method differences (Patterson et al. 2024,
# Sec. 4.5, p. 32): k = the method differences actually computed, each Welch
# interval at 1 - ALPHA / k. Report only -- K1/K2 "met" and the K5 reading stay
# on the pre-registered 95 % interval. The
# family boundary (these four, not FORCE) is own choice, named there.
METHOD_FAMILY = ("K1", "K2", "K5", "K5_nominal")


def family_rates(pol: list, kname: str, d: dict) -> tuple[list, list]:
    if kname in ("K1", "K2"):
        return method_rates(pol, d["block"])
    ablock, fblock = d["blocks"]
    return seed_rates(pol, ablock, "autodr"), seed_rates(pol, fblock, "fixeddr")


def bonferroni(pol: list, out: dict) -> dict:
    fam = [k for k in METHOD_FAMILY if "diff" in out.get(k, {})]
    level = 1.0 - ALPHA / len(fam)
    res = {"family": fam, "k": len(fam), "confidence_level": level}
    for kname in fam:
        a, b = family_rates(pol, kname, out[kname])
        lo, hi = st.ttest_ind(a, b, equal_var=False).confidence_interval(confidence_level=level)
        res[kname] = {"welch_ci": [float(lo), float(hi)], "reading": k5_reading((lo, hi))}
    return res


def final_block(method: str) -> str:
    """The block in which ``method`` plays the T-final table: Fixed-DR in its own
    T-final-F, the other two in T-final. Fixed-DR is a
    candidate and plays T-grenze, T-ood and the noise blocks, so its K3, OOD and
    noise reference is its untagged T-final-F run."""
    return "T-final-F" if method == "fixeddr" else "T-final"


def select_final(pol: list) -> dict:
    """On T-auswahl: highest rate; tie -> smaller force
    p95 over the successes; tie -> smaller seed number. Candidates are every
    method that played T-auswahl (AutoDR, No-DR AND Fixed-DR). A tie that
    survives all three (same seed number in two methods) is not settled by the plan."""
    cand = [p for p in pol if p["block"] == "T-auswahl" and p["tag"] == ""]
    if not cand:
        raise StatsError("no T-auswahl runs: the final policy cannot be selected")
    inf = float("inf")
    key = lambda p: (-p[KEY], p["successes_force_filtered_n"]["p95"] if p["successes_force_filtered_n"]["p95"] is not None else inf, p["seed"])
    ranked = sorted(cand, key=key)
    if len(ranked) > 1 and key(ranked[0]) == key(ranked[1]):
        raise StatsError(f"selection tie the plan does not settle: {ranked[0]['method']} s{ranked[0]['seed']} "
                         f"and {ranked[1]['method']} s{ranked[1]['seed']}")
    best = ranked[0]
    return {"method": best["method"], "seed": best["seed"], "rate": best[KEY],
            "force_p95": best["successes_force_filtered_n"]["p95"], "candidates": len(cand)}


def k3_blocks(df: pd.DataFrame, final: dict) -> dict:
    """Each boundary block of the FINAL policy against its overall T-final rate
    (``final_block``: T-final-F for Fixed-DR), Newcombe difference (block - overall)."""
    sel = lambda b: df[(df["block"] == b) & (df["method"] == final["method"])
                       & (df["seed"] == final["seed"]) & (df["tag"] == "")]
    g, f = sel("T-grenze"), sel(final_block(final["method"]))
    if g.empty or f.empty:
        raise StatsError(f"K3 needs T-grenze AND {final_block(final['method'])} of the final policy")
    k_all, n_all = int(f[KEY].sum()), int(len(f))
    out = {}
    for key, b in g.groupby("boundary_key", sort=True):
        k, n = int(b[KEY].sum()), int(len(b))
        lo, hi = confint_proportions_2indep(k, n, k_all, n_all, method="newcomb", compare="diff", alpha=ALPHA)
        out[key] = {"k": k, "n": n, "rate": k / n, "wilson": wilson(k, n),
                    "diff_vs_final": k / n - k_all / n_all, "newcombe_ci": [float(lo), float(hi)]}
    return {"overall": {"k": k_all, "n": n_all, "rate": k_all / n_all}, "blocks": out,
            "met": all(v["newcombe_ci"][1] >= 0.0 for v in out.values())}


def ood_blocks(df: pd.DataFrame) -> dict:
    """each OOD block against the
    SAME policy's untagged T-final rate, Newcombe difference (block - T-final).
    DESCRIPTIVE: check whether performance degrades gradually or
    collapses beyond the band; there is no pass line, so there is no `met`.

    Every T-ood policy is judged alone (the five AutoDR seeds and
    the five Fixed-DR seeds, whose reference is T-final-F, ``final_block``). Across the
    seeds of one method: the per-seed rates with mean, SD (ddof=1), min and
    max -- K4's form and IndustReal's "mean +- SD over 5 seeds". No interval
    over seeds: Welch (K1) compares two methods; no source in the repo gives
    one for the seeds of ONE method (named gap).
    """
    g = df[df["block"] == "T-ood"]
    pols = sorted((m, int(s)) for m, s in set(zip(g["method"], g["seed"])))
    per = {}
    for m, s in pols:
        f = df[(df["block"] == final_block(m)) & (df["method"] == m) & (df["seed"] == s) & (df["tag"] == "")]
        if f.empty:
            raise StatsError(f"{final_block(m)} of {m} s{s} has no untagged run; the OOD reference needs it")
        k_all, n_all = int(f[KEY].sum()), int(len(f))
        out = {}
        for key, b in g[(g["method"] == m) & (g["seed"] == s)].groupby("boundary_key", sort=True):
            k, n = int(b[KEY].sum()), int(len(b))
            # Keyword order differs from k3_blocks on purpose: each self-test
            # mutation anchor must match exactly one call.
            lo, hi = confint_proportions_2indep(k, n, k_all, n_all, compare="diff", method="newcomb", alpha=ALPHA)
            out[key] = {"k": k, "n": n, "rate": k / n, "wilson": wilson(k, n),
                        "diff_vs_final": k / n - k_all / n_all, "newcombe_ci": [float(lo), float(hi)]}
        per[f"{m}_s{s}"] = {"policy": {"method": m, "seed": s},
                            "reference": {"block": final_block(m), "tag": "", "k": k_all, "n": n_all,
                                          "rate": k_all / n_all},
                            "blocks": out}
    return {"policies": per, "across_seeds": _across_seeds(per), "descriptive": True}


def noise_blocks(df: pd.DataFrame) -> dict:
    """each observation-noise
    x2 run (T-final table, tag in ``CEC.NOISE_TAGS``) against the SAME
    policy's T-final run tagged ``NOISE_REFERENCE_TAG`` (untagged) -- same
    table, same commit, only the noise differs. Newcombe difference per seed,
    the across-seed form of ``ood_blocks``. DESCRIPTIVE, no pass line.
    Fixed-DR's noise runs and their reference sit in T-final-F (
    ``final_block``)."""
    f = df[df["block"].isin(("T-final", "T-final-F"))]
    noisy = f[f["tag"].isin(list(CEC.NOISE_TAGS))]
    pols = sorted((m, int(s)) for m, s in set(zip(noisy["method"], noisy["seed"])))
    per = {}
    for m, s in pols:
        g = f[(f["block"] == final_block(m)) & (f["method"] == m) & (f["seed"] == s)]
        ref = g[g["tag"] == NOISE_REFERENCE_TAG]
        if ref.empty:
            raise StatsError(f"{final_block(m)} of {m} s{s} has noise runs but no {NOISE_REFERENCE_TAG!r} reference")
        k_all, n_all = int(ref[KEY].sum()), int(len(ref))
        out = {}
        for tag, b in g[g["tag"].isin(list(CEC.NOISE_TAGS))].groupby("tag", sort=True):
            k, n = int(b[KEY].sum()), int(len(b))
            # Keyword order differs from k3/ood on purpose (unique mutation anchors).
            lo, hi = confint_proportions_2indep(k, n, k_all, n_all, alpha=ALPHA, method="newcomb", compare="diff")
            out[tag] = {"k": k, "n": n, "rate": k / n, "wilson": wilson(k, n), "changed": CEC.NOISE_TAGS[tag],
                        "diff_vs_final": k / n - k_all / n_all, "newcombe_ci": [float(lo), float(hi)]}
        per[f"{m}_s{s}"] = {"policy": {"method": m, "seed": s},
                            "reference": {"block": final_block(m), "tag": NOISE_REFERENCE_TAG, "k": k_all, "n": n_all,
                                          "rate": k_all / n_all},
                            "blocks": out}
    return {"policies": per, "across_seeds": _across_seeds(per), "descriptive": True}


def _across_seeds(per: dict) -> dict:
    """Per method, over its seeds: each block's rates with mean, SD (ddof=1),
    min, max and the mean of the per-seed differences (K4's form, IndustReal's
    "mean +- SD over 5 seeds"; no interval, see ``ood_blocks``)."""
    across = {}
    for method in sorted({p["policy"]["method"] for p in per.values()}):
        mine = [p for p in per.values() if p["policy"]["method"] == method]
        keys = sorted(mine[0]["blocks"])
        for p in mine:
            if sorted(p["blocks"]) != keys:
                raise StatsError(f"{method} s{p['policy']['seed']} holds blocks {sorted(p['blocks'])}, "
                                 f"s{mine[0]['policy']['seed']} holds {keys}")
        blocks = {}
        for key in keys:
            rates = [p["blocks"][key]["rate"] for p in mine]
            diffs = [p["blocks"][key]["diff_vs_final"] for p in mine]
            blocks[key] = {"rates": rates, "mean": float(np.mean(rates)),
                           "sd": float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
                           "min": min(rates), "max": max(rates),
                           "diffs_vs_final": diffs, "diff_mean": float(np.mean(diffs))}
        across[method] = {"seeds": [p["policy"]["seed"] for p in mine], "blocks": blocks}
    return across


# the force-sensor ablation root (results/eval_rlgames_kraft/). The
# two arms, with force minus without; the blocks are the T-final table untagged
# and tagged pn2.
FORCE_ARMS = ("autodr", "autodr_nf")
FORCE_BLOCKS = (("T-final", ""), ("pn2", "pn2"))


def force_root(df: pd.DataFrame) -> bool:
    return FORCE_ARMS[1] in set(df["method"])


def force_blocks(pol: list) -> dict:
    """per block the per-seed rates of both arms with Wilson, the
    difference of the arm means ``autodr - autodr_nf`` in K1's form (Welch over
    the SEED rates, bootstrap beside it, ``method_diff``), and per seed the loss
    T-final -> pn2 (rate pn2 - rate T-final, the ``diff_vs_final`` of NOISE
    without its interval). DESCRIPTIVE: 3 seeds per arm, no ``met``."""
    out = {"arms": list(FORCE_ARMS), "descriptive": True, "blocks": {}, "loss_T-final_to_pn2": {}}
    per = {}
    for label, tag in FORCE_BLOCKS:
        per[label] = {m: sorted((p for p in pol if p["block"] == "T-final" and p["tag"] == tag
                                 and p["method"] == m), key=lambda p: p["seed"]) for m in FORCE_ARMS}
        if not all(per[label].values()):
            out["blocks"][label] = {"missing": f"needs T-final tag {tag!r} runs of both arms {list(FORCE_ARMS)}"}
            continue
        d = method_diff([p[KEY] for p in per[label][FORCE_ARMS[0]]], [p[KEY] for p in per[label][FORCE_ARMS[1]]],
                        names=FORCE_ARMS)
        d["tag"] = tag
        d["per_seed"] = {m: [{"seed": p["seed"], "k": p["success_train_k"], "n": p["n"], "rate": p[KEY],
                              "wilson": p["success_train_wilson"]} for p in per[label][m]] for m in FORCE_ARMS}
        out["blocks"][label] = d
    for m in FORCE_ARMS:
        ref = {p["seed"]: p[KEY] for p in per["T-final"][m]}
        noisy = {p["seed"]: p[KEY] for p in per["pn2"][m]}
        if not noisy:
            continue
        if sorted(noisy) != sorted(ref):
            raise StatsError(f"{m}: pn2 seeds {sorted(noisy)} and T-final seeds {sorted(ref)} differ")
        loss = [noisy[s] - ref[s] for s in sorted(ref)]
        out["loss_T-final_to_pn2"][m] = {"seeds": sorted(ref), "loss": loss, "mean": float(np.mean(loss))}
    return out


def bin_rates(df: pd.DataFrame, column: str, edges) -> list:
    """`_rate_by_bin` rule, with a Wilson interval per bin; empty bins kept."""
    idx = np.searchsorted(np.asarray(edges, dtype=float), df[column].to_numpy(dtype=float), side="right")
    idx = np.minimum(idx, len(edges) - 1)
    out = []
    for i, upper in enumerate(edges):
        sel = df[KEY].to_numpy()[idx == i]
        n, k = int(len(sel)), int(sel.sum())
        out.append({"upper": float(upper), "n": n, "k": k,
                    "rate": (k / n) if n else None, "wilson": wilson(k, n) if n else None})
    return out


def summarize(df: pd.DataFrame, t1_mm: float | None = None) -> dict:
    pol = policy_rows(df, t1_mm)
    out = {"marker": SCRIPT_MARKER, "headline_key": KEY, "alpha": ALPHA, "policies": pol, "t1_mm": t1_mm,
           "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": __import__("scipy").__version__,
                        "statsmodels": __import__("statsmodels").__version__}}
    blocks = set(df["block"])
    for kname, block in (("K1", "T-final"), ("K2", "T-nominal")):
        if block not in blocks:
            out[kname] = {"missing": f"no {block} runs"}
            continue
        if force_root(df):
            out[kname] = {"missing": "force-ablation root: no nodr arm; see FORCE"}
            continue
        a, b = method_rates(pol, block)
        try:
            d = method_diff(a, b)
        except StatsError as exc:
            out[kname] = {"error": str(exc), "block": block}
            continue
        d["block"] = block
        d["met"] = (d["welch_ci"][0] > 0.0) if kname == "K1" else not (d["welch_ci"][1] < 0.0)
        out[kname] = d
    out.update(k5(pol, blocks))
    if any("diff" in out.get(k, {}) for k in METHOD_FAMILY):
        out["BONFERRONI"] = bonferroni(pol, out)
    if "T-auswahl" in blocks:
        out["final_policy"] = select_final(pol)
        if {"T-grenze", "T-final"} <= blocks:
            try:
                out["K3"] = k3_blocks(df, out["final_policy"])
            except StatsError as exc:
                # e.g. T-grenze ran another policy than the rule selects. Kept in
                # the file and turned into exit 1 by main(), never dropped.
                out["K3"] = {"error": str(exc)}
    if "T-ood" in blocks:
        try:
            out["OOD"] = ood_blocks(df)
        except StatsError as exc:
            out["OOD"] = {"error": str(exc)}
    if set(df.loc[df["block"].isin(("T-final", "T-final-F")), "tag"]) & set(CEC.NOISE_TAGS):
        try:
            out["NOISE"] = noise_blocks(df)
        except StatsError as exc:
            out["NOISE"] = {"error": str(exc)}
    if force_root(df):
        try:
            out["FORCE"] = force_blocks(pol)
        except StatsError as exc:
            out["FORCE"] = {"error": str(exc)}
    k4 = {}
    # K4 per method, Fixed-DR from its T-final-F.
    for m in ("autodr", "nodr", "fixeddr"):
        r = [p[KEY] for p in pol if p["block"] == final_block(m) and p["tag"] == "" and p["method"] == m]
        if r:
            k4[m] = {"rates": r, "min": min(r), "max": max(r),
                     "sd": float(np.std(r, ddof=1)) if len(r) > 1 else None}
    out["K4"] = k4
    # Fixed-DR plays T-final's table in its own block T-final-F; eval_collect
    # keeps each method in its own block, so the keys stay unique.
    fin = df[df["block"].isin(("T-final", "T-final-F")) & (df["tag"] == "")]
    out["bins_T_final"] = {
        f"{m}_s{s}": {c: bin_rates(g, c, e) for c, e in BIN_EDGES.items()}
        for (m, s), g in fin.groupby(["method", "seed"], sort=True)
    }
    return out


def write_outputs(results: pathlib.Path, summary: dict) -> None:
    with open(results / "summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    cols = ("block", "method", "seed", "tag", "n", "success_train", "success_train_k",
            "success_strict", "force_abort_rate")
    pd.DataFrame([{c: p[c] for c in cols} for p in summary["policies"]]).to_csv(
        results / "summary_tables.csv", index=False, lineterminator="\n")


# ---------------------------------------------------------------------------
# Self-test: checks on built data; each mutation of THIS file's source
# must flip exactly its declared checks.
# ---------------------------------------------------------------------------

N_EP = 100
K_AUTODR = (90, 92, 94, 96, 98)       # mean 0.94
K_NODR = (40, 42, 44, 46, 90)         # mean 0.524, skewed: Welch != pooled, BCa != percentile
# Fixed-DR: T-final-F mean 0.74 -> K5 interval wholly above 0;
# T-nominal-F mean 0.928 -> K5_nominal interval contains 0.
K_FIXED = {"T-final-F": (70, 72, 74, 76, 78), "T-nominal-F": (80, 90, 95, 99, 100)}
# Friction 0.08..0.20 in 0.01 steps (T-final's band); 0.20 is the last edge itself.
FRICTION_VALUES = tuple(round(0.08 + 0.01 * j, 2) for j in range(13))
# Interpenetration: row i of T-final / T-nominal / T-auswahl / *-F has ip_max_mm = 0.01 * i;
# t1 as in episodes_meta.json (INTERPEN_THRESH = 0.5 * PLAY_X).
T1_TEST = 0.2938


def _frame() -> pd.DataFrame:
    rows = []
    for method, ks in (("autodr", K_AUTODR), ("nodr", K_NODR)):
        for seed, k in enumerate(ks, start=1):
            for block in ("T-final", "T-nominal", "T-auswahl"):
                for i in range(N_EP):
                    ok = int(i < k)
                    # autodr: two successes per seed hit the force limit -> strict differs
                    viol = int(ok and method == "autodr" and i < 2)
                    rows.append({"block": block, "method": method, "seed": seed, "tag": "",
                                 "row_id": i, "boundary_key": "", "outcome": "success" if ok else "timeout",
                                 "success_train": ok, "success_strict": ok * (1 - viol), "force_abort": viol,
                                 "steps": 40 + i, "force_max_filtered_n": 31.0 if viol else 10.0 + seed,
                                 "lat_r_mm": (i % 31) * 1.0, "yaw_abs_deg": (i % 6) * 1.0,
                                 "tilt_deg": (i % 11) * 1.0, "start_height_m": 0.02 + (i % 10) * 0.01, "ip_max_mm": 0.01 * i,
                                 "friction": FRICTION_VALUES[i % 13]})
    for block, ks in K_FIXED.items():
        for seed, k in enumerate(ks, start=1):
            for i in range(N_EP):
                ok = int(i < k)
                rows.append({"block": block, "method": "fixeddr", "seed": seed, "tag": "",
                             "row_id": i, "boundary_key": "", "outcome": "success" if ok else "timeout",
                             "success_train": ok, "success_strict": ok, "force_abort": 0,
                             "steps": 40 + i, "force_max_filtered_n": 10.0 + seed,
                             "lat_r_mm": (i % 31) * 1.0, "yaw_abs_deg": (i % 6) * 1.0,
                             "tilt_deg": (i % 11) * 1.0, "start_height_m": 0.02 + (i % 10) * 0.01, "ip_max_mm": 0.01 * i,
                             "friction": FRICTION_VALUES[i % 13]})
    # Boundary block of the final policy (autodr s5): 2 keys x 50 rows.
    for j in range(100):
        rows.append({"block": "T-grenze", "method": "autodr", "seed": 5, "tag": "", "row_id": j,
                     "boundary_key": "lat_r_hi" if j < 50 else "yaw_lo", "outcome": "success",
                     "success_train": int(j < 45 or j >= 50), "success_strict": int(j < 45 or j >= 50),
                     "force_abort": 0, "steps": 50, "force_max_filtered_n": 10.0,
                     "lat_r_mm": 30.0, "yaw_abs_deg": 0.0, "tilt_deg": 0.0, "start_height_m": 0.05, "ip_max_mm": 0.0})
    # OOD blocks: autodr s5 lat_r_hi 40/50, yaw_lo 50/50; autodr s4
    # lat_r_hi 30/50, yaw_lo 45/50 (two seeds stand for the five). fixeddr
    # s2 lat_r_hi 20/50, yaw_lo 40/50, against its T-final-F 72/100.
    for method, seed, k_lat, k_yaw in (("autodr", 5, 40, 50), ("autodr", 4, 30, 45), ("fixeddr", 2, 20, 40)):
        for j in range(100):
            ok = int(j < k_lat) if j < 50 else int(j - 50 < k_yaw)
            rows.append({"block": "T-ood", "method": method, "seed": seed, "tag": "", "row_id": j,
                         "boundary_key": "lat_r_hi" if j < 50 else "yaw_lo", "outcome": "success" if ok else "timeout",
                         "success_train": ok, "success_strict": ok, "force_abort": 0, "steps": 50,
                         "force_max_filtered_n": 10.0, "lat_r_mm": 45.0, "yaw_abs_deg": 0.0, "tilt_deg": 0.0,
                         "start_height_m": 0.05, "ip_max_mm": 0.0})
    return pd.DataFrame(rows)


def _noise_frame(df: pd.DataFrame, drop_ref_seed: int | None = None) -> pd.DataFrame:
    """The rl_games T-final folder: beside each autodr seed 4, 5's
    untagged run (96/100, 98/100 in _frame) three x2 runs of 100 rows:
    s4 pn2 80, gx2 96, py2 85; s5 pn2 90, gx2 95, py2 93 (no force violations).
    fixeddr s2 pn2 60 in T-final-F, beside its untagged 72/100 there.
    ``drop_ref_seed`` removes that seed's untagged T-final run."""
    out = df
    if drop_ref_seed is not None:
        out = df[~((df["block"] == "T-final") & (df["method"] == "autodr") & (df["seed"] == drop_ref_seed)
                   & (df["tag"] == ""))]
    rows = []
    for seed, ks in ((4, (80, 96, 85)), (5, (90, 95, 93))):
        for tag, k in zip(("pn2", "gx2", "py2"), ks):
            for i in range(N_EP):
                ok = int(i < k)
                rows.append({"block": "T-final", "method": "autodr", "seed": seed, "tag": tag, "row_id": i,
                             "boundary_key": "", "outcome": "success" if ok else "timeout", "success_train": ok,
                             "success_strict": ok, "force_abort": 0, "steps": 50, "force_max_filtered_n": 10.0,
                             "lat_r_mm": 0.0, "yaw_abs_deg": 0.0, "tilt_deg": 0.0, "start_height_m": 0.05, "ip_max_mm": 0.0})
    for i in range(N_EP):
        ok = int(i < 60)
        rows.append({"block": "T-final-F", "method": "fixeddr", "seed": 2, "tag": "pn2", "row_id": i,
                     "boundary_key": "", "outcome": "success" if ok else "timeout", "success_train": ok,
                     "success_strict": ok, "force_abort": 0, "steps": 50, "force_max_filtered_n": 10.0,
                     "lat_r_mm": 0.0, "yaw_abs_deg": 0.0, "tilt_deg": 0.0, "start_height_m": 0.05, "ip_max_mm": 0.0})
    return pd.concat([out, pd.DataFrame(rows)], ignore_index=True)


def _fixed_final_frame(df: pd.DataFrame) -> pd.DataFrame:
    """fixeddr s3 plays T-auswahl at 99/100 (above autodr s5's 98) and
    T-grenze as the final policy (lat_r_hi 45/50, yaw_lo 50/50); its overall
    T-final rate is its T-final-F run, 74/100 in _frame."""
    rows = []
    for i in range(N_EP):
        ok = int(i < 99)
        rows.append({"block": "T-auswahl", "method": "fixeddr", "seed": 3, "tag": "", "row_id": i,
                     "boundary_key": "", "outcome": "success" if ok else "timeout", "success_train": ok,
                     "success_strict": ok, "force_abort": 0, "steps": 40 + i, "force_max_filtered_n": 13.0,
                     "lat_r_mm": 0.0, "yaw_abs_deg": 0.0, "tilt_deg": 0.0, "start_height_m": 0.05, "ip_max_mm": 0.0})
    for j in range(100):
        ok = int(j < 45 or j >= 50)
        rows.append({"block": "T-grenze", "method": "fixeddr", "seed": 3, "tag": "", "row_id": j,
                     "boundary_key": "lat_r_hi" if j < 50 else "yaw_lo", "outcome": "success" if ok else "timeout",
                     "success_train": ok, "success_strict": ok, "force_abort": 0, "steps": 50,
                     "force_max_filtered_n": 10.0, "lat_r_mm": 30.0, "yaw_abs_deg": 0.0, "tilt_deg": 0.0,
                     "start_height_m": 0.05, "ip_max_mm": 0.0})
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


# the force-ablation root, T-final table only, seeds 1-3 per arm.
K_FORCE = {("autodr", ""): (99, 98, 97), ("autodr", "pn2"): (95, 94, 90),
           ("autodr_nf", ""): (98, 99, 96), ("autodr_nf", "pn2"): (85, 80, 88)}


def _force_frame(tags=("", "pn2")) -> pd.DataFrame:
    rows = []
    for (method, tag), ks in K_FORCE.items():
        if tag not in tags:
            continue
        for seed, k in enumerate(ks, start=1):
            for i in range(N_EP):
                ok = int(i < k)
                rows.append({"block": "T-final", "method": method, "seed": seed, "tag": tag, "row_id": i,
                             "boundary_key": "", "outcome": "success" if ok else "timeout", "success_train": ok,
                             "success_strict": ok, "force_abort": 0, "steps": 50, "force_max_filtered_n": 10.0,
                             "lat_r_mm": 0.0, "yaw_abs_deg": 0.0, "tilt_deg": 0.0, "start_height_m": 0.05,
                             "ip_max_mm": 0.0, "friction": 0.1})
    return pd.DataFrame(rows)


def _wilson_ref(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def _welch_ref(a, b, alpha=ALPHA):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = st.t.ppf(1 - alpha / 2, df) * math.sqrt(va + vb)
    d = a.mean() - b.mean()
    return d - h, d + h


def _boot_ref(a, b, reps=20000, seed=7):
    """Own percentile bootstrap, independent resampling per group (reference only)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = a[rng.integers(0, len(a), (reps, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (reps, len(b)))].mean(1)
    return np.percentile(da - db, [2.5, 97.5])


CHECKS = (
    "wilson k=0,5,10 of 10 = reference formula",
    "policy rate = k / n",
    "K1 diff = 0.416 and Welch CI = reference formula",
    "K1 bootstrap CI = own percentile reference within 0.005",
    "K1 met, K2 met (same data in both blocks)",
    "K1 reads success_train, not success_strict",
    "final policy = autodr s5 (highest rate)",
    "K3: lat_r_hi (45/50) below overall 98/100 -> Newcombe = library call, met False",
    "bins follow _rate_by_bin: value 30 lands in the LAST lateral bin",
    "nearest-rank p95 of 1..20 = 19",
    "summary.json is byte-equal on a second run",
    "K5 diff = autodr T-final - fixeddr T-final-F = 0.20 and Welch CI = reference formula",
    "K5 reading follows the Welch CI: above_0 on T-final, contains_0 on T-nominal, below_0 on (-0.3, -0.1)",
    "bins_T_final holds Fixed-DR's T-final-F bins beside AutoDR/No-DR (same table)",
    "OOD: autodr s5 lat_r_hi 40/50 vs its untagged T-final 98/100 -> Newcombe = library call, no 'met'",
    "OOD per seed: autodr s4 lat_r_hi 30/50 vs ITS OWN T-final 96/100 -> Newcombe = library call",
    "OOD across seeds: lat_r_hi rates [0.6, 0.8] -> mean, SD (ddof=1), min, max, diff mean = numpy reference",
    "OOD refuses a T-ood policy without an untagged T-final run (s4), it is not skipped",
    "NOISE per seed: s4 pn2 80/100 and s5 py2 93/100 vs their own untagged T-final -> Newcombe; no 'met'",
    "NOISE across seeds: pn2 rates [0.8, 0.9] -> mean, SD (ddof=1), diff mean = numpy reference",
    "NOISE refuses a policy whose untagged T-final is missing (s4), it is not skipped",
    "noise-tagged T-final runs change neither K1 nor the OOD reference",
    # Fixed-DR is a candidate and plays T-grenze, T-ood and the noise blocks.
    "a Fixed-DR seed that tops T-auswahl (99/100) is the final policy; K3 reads its T-final-F 74/100",
    "OOD: fixeddr s2 lat_r_hi 20/50 vs ITS T-final-F 72/100 -> Newcombe = library call",
    "NOISE: fixeddr s2 pn2 60/100 vs its untagged T-final-F 72/100 -> Newcombe = library call",
    "K4: Fixed-DR's five T-final-F rates beside AutoDR / No-DR",
    # friction bins, own construction.
    "friction bins: five equal bins over 0.08..0.20 (width 0.024), 0.20 lands in the LAST bin",
    # Episode interpenetration for successful trials only.
    "successes_ip_max_mm: nodr s1 T-final (40/100, ip 0.00..0.39 on the successes) -> median 0.195, "
    "p95 0.37, max 0.39, n_ge_t1 10; t1_mm in the summary",
    # the force-sensor ablation (autodr - autodr_nf, T-final and pn2).
    "FORCE T-final: per-seed k/n + Wilson = reference; diff = mean(autodr) - mean(autodr_nf); "
    "Welch = reference formula over the SEED rates; bootstrap = own reference within 0.005",
    "FORCE pn2: diff = 0.93 - 0.8433 and Welch = reference formula over the seed rates of tag pn2",
    "FORCE loss T-final -> pn2 per seed = rate pn2 - rate T-final, equal to NOISE diffs_vs_final",
    "FORCE root: K1 and K2 are 'missing' (no nodr arm), not an error; NOISE holds both arms",
    "a root without autodr_nf has no FORCE key (standard summary.json)",
    "FORCE with untagged runs only: pn2 'missing', T-final computed, no loss",
    # Patterson et al. 2024 Sec. 4.5: Bonferroni cross-check of the method differences.
    "BONFERRONI: family K1, K2, K5, K5_nominal (k = 4); each Welch CI = reference formula at ALPHA / 4, "
    "K1's lower bound below its 95 % one, 'met' unchanged; readings above_0 x3, contains_0 on K5_nominal",
)


def _run_checks(ns) -> dict:
    res = {}
    res[CHECKS[0]] = all(np.allclose(ns["wilson"](k, 10), _wilson_ref(k, 10), atol=1e-9) for k in (0, 5, 10))
    df = _frame()
    s = ns["summarize"](df, t1_mm=T1_TEST)
    p = [x for x in s["policies"] if x["block"] == "T-final" and x["method"] == "autodr" and x["seed"] == 1][0]
    res[CHECKS[1]] = abs(p["success_train"] - 0.90) < 1e-12
    a = [k / N_EP for k in K_AUTODR]
    b = [k / N_EP for k in K_NODR]
    k1 = s["K1"]
    res[CHECKS[2]] = abs(k1["diff"] - 0.416) < 1e-12 and np.allclose(k1["welch_ci"], _welch_ref(a, b), atol=1e-9)
    res[CHECKS[3]] = np.allclose(k1["bootstrap_percentile_ci"], _boot_ref(a, b), atol=0.005)
    res[CHECKS[4]] = k1["met"] is True and s["K2"]["met"] is True
    res[CHECKS[5]] = abs(k1["mean_autodr"] - 0.94) < 1e-12
    res[CHECKS[6]] = s["final_policy"]["method"] == "autodr" and s["final_policy"]["seed"] == 5
    ref = confint_proportions_2indep(45, 50, 98, 100, method="newcomb", compare="diff")
    k3 = s.get("K3", {})
    res[CHECKS[7]] = ("blocks" in k3 and k3["met"] is False
                      and np.allclose(k3["blocks"]["lat_r_hi"]["newcombe_ci"], ref, atol=1e-12))
    lb = s["bins_T_final"]["autodr_s1"]["lat_r_mm"]
    # values 0..30 (i % 31): 30 -> last bin; 24..29 -> last bin as well.
    res[CHECKS[8]] = lb[-1]["n"] == sum(1 for i in range(N_EP) if (i % 31) >= 24)
    res[CHECKS[9]] = ns["pct_nearest_rank"](range(1, 21), 0.95) == 19
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        ns["write_outputs"](t, s)
        first = (t / "summary.json").read_bytes()
        ns["write_outputs"](t, ns["summarize"](df, t1_mm=T1_TEST))
        res[CHECKS[10]] = first == (t / "summary.json").read_bytes()
    f = [k / N_EP for k in K_FIXED["T-final-F"]]
    k5, k5n = s.get("K5", {}), s.get("K5_nominal", {})
    res[CHECKS[11]] = ("diff" in k5 and abs(k5["diff"] - 0.20) < 1e-12
                       and np.allclose(k5["welch_ci"], _welch_ref(a, f), atol=1e-9)
                       and k5["mean_fixeddr"] == float(np.mean(f)))
    res[CHECKS[12]] = (k5.get("reading") == "above_0" and k5n.get("reading") == "contains_0"
                       and ns["k5_reading"]((-0.3, -0.1)) == "below_0")
    bf = s.get("BONFERRONI", {})
    fam = {"K1": (a, b), "K2": (a, b), "K5": (a, f), "K5_nominal": (a, [k / N_EP for k in K_FIXED["T-nominal-F"]])}
    res[CHECKS[34]] = (bf.get("family") == list(fam) and bf.get("k") == 4
                       and all(np.allclose(bf[k]["welch_ci"], _welch_ref(*ab, alpha=ALPHA / 4), atol=1e-9)
                               for k, ab in fam.items())
                       and bf["K1"]["welch_ci"][0] < k1["welch_ci"][0] and k1["met"] is True
                       and [bf[k]["reading"] for k in fam] == ["above_0", "above_0", "above_0", "contains_0"])
    bt = s["bins_T_final"]
    res[CHECKS[13]] = (sorted(bt) == sorted([f"autodr_s{i}" for i in range(1, 6)] + [f"nodr_s{i}" for i in range(1, 6)]
                                            + [f"fixeddr_s{i}" for i in range(1, 6)])
                       and bt["fixeddr_s1"]["lat_r_mm"][-1]["n"] == lb[-1]["n"])
    ref_ood = confint_proportions_2indep(40, 50, 98, 100, method="newcomb", compare="diff")
    ood = s.get("OOD", {})
    o5 = ood.get("policies", {}).get("autodr_s5", {})
    res[CHECKS[14]] = ("blocks" in o5 and "met" not in ood and o5["reference"]["n"] == 100
                       and o5["reference"]["tag"] == ""
                       and np.allclose(o5["blocks"]["lat_r_hi"]["newcombe_ci"], ref_ood, atol=1e-12))
    ref_s4 = confint_proportions_2indep(30, 50, 96, 100, method="newcomb", compare="diff")
    o4 = ood.get("policies", {}).get("autodr_s4", {})
    res[CHECKS[15]] = ("blocks" in o4 and o4["reference"]["k"] == 96
                       and np.allclose(o4["blocks"]["lat_r_hi"]["newcombe_ci"], ref_s4, atol=1e-12))
    acr = ood.get("across_seeds", {}).get("autodr", {})
    r = [0.6, 0.8]
    lat = acr.get("blocks", {}).get("lat_r_hi", {})
    res[CHECKS[16]] = (acr.get("seeds") == [4, 5] and np.allclose(lat.get("rates", []), r, atol=1e-12)
                       and abs(lat["mean"] - float(np.mean(r))) < 1e-12
                       and abs(lat["sd"] - float(np.std(r, ddof=1))) < 1e-12
                       and lat["min"] == 0.6 and lat["max"] == 0.8
                       and abs(lat["diff_mean"] - ((0.6 - 0.96) + (0.8 - 0.98)) / 2) < 1e-12)
    # s4 played T-ood but its T-final is missing: the whole OOD block is an error.
    no4 = df[(df["block"] == "T-ood") | ((df["block"] == "T-final") & (df["method"] == "autodr") & (df["seed"] == 5))]
    o_no4 = ns["summarize"](no4).get("OOD", {})
    res[CHECKS[17]] = "error" in o_no4 and "autodr s4" in o_no4["error"]
    sn = ns["summarize"](_noise_frame(df))
    nz = sn.get("NOISE", {})
    n4 = nz.get("policies", {}).get("autodr_s4", {})
    n5 = nz.get("policies", {}).get("autodr_s5", {})
    ref_n4 = confint_proportions_2indep(80, 100, 96, 100, method="newcomb", compare="diff")
    ref_n5 = confint_proportions_2indep(93, 100, 98, 100, method="newcomb", compare="diff")
    res[CHECKS[18]] = ("blocks" in n4 and "met" not in nz and n4["reference"]["tag"] == ""
                       and sorted(n4["blocks"]) == ["gx2", "pn2", "py2"]
                       and np.allclose(n4["blocks"]["pn2"]["newcombe_ci"], ref_n4, atol=1e-12)
                       and np.allclose(n5["blocks"]["py2"]["newcombe_ci"], ref_n5, atol=1e-12))
    pn = nz.get("across_seeds", {}).get("autodr", {}).get("blocks", {}).get("pn2", {})
    r = [0.8, 0.9]
    res[CHECKS[19]] = (len(pn.get("rates", [])) == 2 and np.allclose(pn["rates"], r, atol=1e-12)
                       and abs(pn["sd"] - float(np.std(r, ddof=1))) < 1e-12
                       and abs(pn["diff_mean"] - ((0.8 - 0.96) + (0.9 - 0.98)) / 2) < 1e-12)
    nz4 = ns["summarize"](_noise_frame(df, drop_ref_seed=4)).get("NOISE", {})
    res[CHECKS[20]] = "error" in nz4 and "autodr s4" in nz4["error"]
    o5n = sn.get("OOD", {}).get("policies", {}).get("autodr_s5", {})
    res[CHECKS[21]] = (sn["K1"].get("diff") == k1["diff"] and "reference" in o5n
                       and o5n["reference"] == o5.get("reference"))
    sf = ns["summarize"](_fixed_final_frame(df))
    ref_k3f = confint_proportions_2indep(45, 50, 74, 100, method="newcomb", compare="diff")
    k3f = sf.get("K3", {})
    res[CHECKS[22]] = (sf["final_policy"]["method"] == "fixeddr" and sf["final_policy"]["seed"] == 3
                       and "blocks" in k3f and k3f["overall"]["k"] == 74
                       and np.allclose(k3f["blocks"]["lat_r_hi"]["newcombe_ci"], ref_k3f, atol=1e-12))
    of2 = ood.get("policies", {}).get("fixeddr_s2", {})
    ref_of2 = confint_proportions_2indep(20, 50, 72, 100, method="newcomb", compare="diff")
    res[CHECKS[23]] = ("blocks" in of2 and of2["reference"]["block"] == "T-final-F" and of2["reference"]["k"] == 72
                       and np.allclose(of2["blocks"]["lat_r_hi"]["newcombe_ci"], ref_of2, atol=1e-12))
    nf2 = nz.get("policies", {}).get("fixeddr_s2", {})
    ref_nf2 = confint_proportions_2indep(60, 100, 72, 100, method="newcomb", compare="diff")
    res[CHECKS[24]] = ("blocks" in nf2 and nf2["reference"]["block"] == "T-final-F" and nf2["reference"]["k"] == 72
                       and sorted(nf2["blocks"]) == ["pn2"]
                       and np.allclose(nf2["blocks"]["pn2"]["newcombe_ci"], ref_nf2, atol=1e-12))
    k4f = s["K4"].get("fixeddr", {})
    res[CHECKS[25]] = k4f.get("rates") == [k / N_EP for k in K_FIXED["T-final-F"]]
    fb = s["bins_T_final"]["autodr_s1"].get("friction", [])
    vals = [FRICTION_VALUES[i % 13] for i in range(N_EP)]
    lo = (-math.inf, 0.104, 0.128, 0.152, 0.176)
    hi = (0.104, 0.128, 0.152, 0.176, math.inf)
    res[CHECKS[26]] = (len(fb) == 5 and np.allclose([b["upper"] for b in fb], 0.08 + 0.024 * np.arange(1, 6), atol=1e-12)
                       and [b["n"] for b in fb] == [sum(1 for v in vals if a <= v < z) for a, z in zip(lo, hi)])
    pn1 = [x for x in s["policies"] if x["block"] == "T-final" and x["method"] == "nodr" and x["seed"] == 1][0]
    ipd = pn1["successes_ip_max_mm"]
    res[CHECKS[27]] = (abs(ipd["median"] - 0.195) < 1e-12 and abs(ipd["p95"] - 0.37) < 1e-12
                       and abs(ipd["max"] - 0.39) < 1e-12 and ipd["n_ge_t1"] == 10 and s["t1_mm"] == T1_TEST)
    sk = ns["summarize"](_force_frame())
    fo = sk.get("FORCE", {})
    fb = fo.get("blocks", {})
    rate = lambda m, t: [k / N_EP for k in K_FORCE[(m, t)]]
    ft = fb.get("T-final", {})
    res[CHECKS[28]] = ("welch_ci" in ft and "met" not in fo
                       and [(x["seed"], x["k"], x["n"]) for x in ft["per_seed"]["autodr_nf"]] == [(1, 98, 100), (2, 99, 100), (3, 96, 100)]
                       and all(np.allclose(x["wilson"], _wilson_ref(x["k"], x["n"]), atol=1e-9)
                               for m in FORCE_ARMS for x in ft["per_seed"][m])
                       and abs(ft["diff"] - (np.mean(rate("autodr", "")) - np.mean(rate("autodr_nf", "")))) < 1e-12
                       and np.allclose(ft["welch_ci"], _welch_ref(rate("autodr", ""), rate("autodr_nf", "")), atol=1e-9)
                       and np.allclose(ft["bootstrap_percentile_ci"], _boot_ref(rate("autodr", ""), rate("autodr_nf", "")),
                                       atol=0.005))
    fp = fb.get("pn2", {})
    res[CHECKS[29]] = ("welch_ci" in fp and fp["tag"] == "pn2"
                       and abs(fp["diff"] - (0.93 - (0.85 + 0.80 + 0.88) / 3)) < 1e-12
                       and np.allclose(fp["welch_ci"], _welch_ref(rate("autodr", "pn2"), rate("autodr_nf", "pn2")), atol=1e-9))
    lo_ = fo.get("loss_T-final_to_pn2", {})
    nz_k = sk.get("NOISE", {}).get("across_seeds", {})
    res[CHECKS[30]] = (sorted(lo_) == sorted(FORCE_ARMS) and all(
        lo_[m]["seeds"] == [1, 2, 3]
        and np.allclose(lo_[m]["loss"], np.subtract(rate(m, "pn2"), rate(m, "")), atol=1e-12)
        and np.allclose(lo_[m]["loss"], nz_k[m]["blocks"]["pn2"]["diffs_vs_final"], atol=1e-12)
        for m in FORCE_ARMS))
    res[CHECKS[31]] = ("missing" in sk.get("K1", {}) and "error" not in sk.get("K1", {})
                       and "missing" in sk.get("K2", {}) and sorted(nz_k) == sorted(FORCE_ARMS))
    res[CHECKS[32]] = "FORCE" not in s and "FORCE" not in sn and "diff" in k1
    fu = ns["summarize"](_force_frame(tags=("",))).get("FORCE", {})
    res[CHECKS[33]] = ("missing" in fu.get("blocks", {}).get("pn2", {})
                       and "welch_ci" in fu.get("blocks", {}).get("T-final", {})
                       and fu.get("loss_T-final_to_pn2") == {})
    return res


MUTATIONS = (
    ("rate-denominator-n-minus-1",
     '"success_train": k / n, "success_train_k"', '"success_train": k / (n - 1), "success_train_k"',
     # The selection keeps its order (every rate scales by the same 100/99).
     # FORCE reads the same policy rates; so does BONFERRONI.
     (CHECKS[1], CHECKS[2], CHECKS[3], CHECKS[5], CHECKS[11], CHECKS[25],
      CHECKS[28], CHECKS[29], CHECKS[30], CHECKS[34])),
    ("welch-pooled-variance",
     "welch = st.ttest_ind(a, b, equal_var=False)", "welch = st.ttest_ind(a, b, equal_var=True)",
     # FORCE goes through the same method_diff.
     (CHECKS[2], CHECKS[28], CHECKS[29])),
    ("bootstrap-falls-back-to-bca",
     '                        method="percentile", rng=', "                        rng=",
     (CHECKS[3],)),
    ("headline-reads-strict",
     'KEY = "success_train"', 'KEY = "success_strict"',
     # OOD and NOISE too: their T-final references (autodr s4, s5) have two strict failures each.
     (CHECKS[1], CHECKS[2], CHECKS[3], CHECKS[5], CHECKS[7], CHECKS[11],
      CHECKS[14], CHECKS[15], CHECKS[16], CHECKS[18], CHECKS[19], CHECKS[34])),
    ("wilson-becomes-normal",
     'alpha=ALPHA, method="wilson")', 'alpha=ALPHA, method="normal")',
     # FORCE's per-seed Wilson is the policy row's.
     (CHECKS[0], CHECKS[28])),
    ("k3-compares-the-wrong-way",
     'k, n, k_all, n_all, method="newcomb"', 'k_all, n_all, k, n, method="newcomb"',
     (CHECKS[7], CHECKS[22])),
    ("bins-take-the-edge-into-the-lower-bin",
     'side="right")', 'side="left")',
     (CHECKS[8],)),
    ("bootstrap-seed-from-the-clock",
     "rng=np.random.default_rng(BOOT_SEED))", "rng=np.random.default_rng())",
     (CHECKS[10],)),
    ("k1-demands-half-a-rate",
     '(d["welch_ci"][0] > 0.0)', '(d["welch_ci"][0] > 0.5)',
     # BONFERRONI's check also holds that K1's "met" is unchanged.
     (CHECKS[4], CHECKS[34])),
    ("selection-picks-the-lowest-rate",
     "key = lambda p: (-p[KEY],", "key = lambda p: (p[KEY],",
     # K3 then finds no T-grenze run of the selected policy and records an error.
     (CHECKS[6], CHECKS[7], CHECKS[22])),
    ("p95-drops-the-ceil",
     "int(math.ceil(q * len(v))) - 1", "int(q * len(v))",
     # The ip p95 of 40 successes moves from index 37 to 38.
     (CHECKS[9], CHECKS[27])),
    # K5.
    ("k5-swaps-the-arms",
     "d = method_diff(a, f, names=", "d = method_diff(f, a, names=",
     (CHECKS[11], CHECKS[12])),
    # Both K5 entries become errors (no rows), so BONFERRONI's family shrinks to k = 2.
    ("k5-takes-nodr-as-the-fixed-arm",
     'f = seed_rates(pol, fblock, "fixeddr")', 'f = seed_rates(pol, fblock, "nodr")',
     (CHECKS[11], CHECKS[12], CHECKS[34])),
    ("k5-reads-autodr-from-the-F-block",
     'a = seed_rates(pol, ablock, "autodr")', 'a = seed_rates(pol, fblock, "autodr")',
     (CHECKS[11], CHECKS[12], CHECKS[34])),
    # BONFERRONI reads through the same helper.
    ("k5-reading-looks-at-the-upper-bound",
     'return "above_0" if lo > 0.0', 'return "above_0" if hi > 0.0',
     (CHECKS[12], CHECKS[34])),
    ("k5-below-reading-never-fires",
     '("below_0" if hi < 0.0', '("below_0" if hi < -1.0',
     (CHECKS[12],)),
    ("bins-ignore-the-fixed-dr-block",
     'fin = df[df["block"].isin(("T-final", "T-final-F")) & (df["tag"] == "")]',
     'fin = df[(df["block"] == "T-final") & (df["tag"] == "")]',
     (CHECKS[13],)),
    # the OOD and noise blocks.
    ("ood-compares-the-wrong-way",
     'k, n, k_all, n_all, compare="diff"', 'k_all, n_all, k, n, compare="diff"',
     (CHECKS[14], CHECKS[15], CHECKS[23])),
    # Every seed read against the FIRST seed's T-final (s4): s5 now 96/100.
    ("ood-reference-from-the-first-policy",
     '(df["method"] == m) & (df["seed"] == s) & (df["tag"] == "")]',
     '(df["method"] == m) & (df["seed"] == pols[0][1]) & (df["tag"] == "")]',
     (CHECKS[14], CHECKS[16], CHECKS[23])),
    # The noise runs of the same seed would join the OOD reference.
    ("ood-reference-pools-the-tags",
     '(df["seed"] == s) & (df["tag"] == "")]', '(df["seed"] == s)]',
     (CHECKS[21],)),
    ("ood-skips-a-policy-without-reference",
     '        if f.empty:\n            raise StatsError(f"{final_block(m)} of {m} s{s} has no untagged',
     '        if f.empty:\n            continue\n            raise StatsError(f"{final_block(m)} of {m} s{s} has no untagged',
     (CHECKS[17],)),
    # One helper serves both: the SD mutation reaches OOD and NOISE.
    ("across-sd-ddof-0",
     "float(np.std(rates, ddof=1))", "float(np.std(rates, ddof=0))",
     (CHECKS[16], CHECKS[19])),
    ("noise-reference-pools-the-tags",
     'ref = g[g["tag"] == NOISE_REFERENCE_TAG]', "ref = g",
     # FORCE's loss is held against NOISE's diffs_vs_final.
     (CHECKS[18], CHECKS[19], CHECKS[20], CHECKS[24], CHECKS[30])),
    ("noise-compares-the-wrong-way",
     'k, n, k_all, n_all, alpha=ALPHA', 'k_all, n_all, k, n, alpha=ALPHA',
     (CHECKS[18], CHECKS[24])),
    ("selection-leaves-out-fixed-dr",
     'cand = [p for p in pol if p["block"] == "T-auswahl" and p["tag"] == ""]',
     'cand = [p for p in pol if p["block"] == "T-auswahl" and p["tag"] == "" and p["method"] != "fixeddr"]',
     (CHECKS[22],)),
    ("fixed-dr-reference-read-from-T-final",
     'return "T-final-F" if method == "fixeddr" else "T-final"', 'return "T-final"',
     # Fixed-DR s2 finds no T-final run: OOD and NOISE become errors as a whole.
     (CHECKS[14], CHECKS[15], CHECKS[16], CHECKS[18], CHECKS[19], CHECKS[21],
      CHECKS[22], CHECKS[23], CHECKS[24], CHECKS[25])),
    ("noise-ignores-the-F-block",
     'f = df[df["block"].isin(("T-final", "T-final-F"))]', 'f = df[df["block"] == "T-final"]',
     (CHECKS[24],)),
    ("k4-leaves-out-fixed-dr",
     'for m in ("autodr", "nodr", "fixeddr"):', 'for m in ("autodr", "nodr"):',
     (CHECKS[25],)),
    ("noise-skips-a-policy-without-reference",
     "        if ref.empty:\n            raise StatsError(",
     "        if ref.empty:\n            continue\n            raise StatsError(",
     (CHECKS[20],)),
    ("friction-bins-lose-the-band-top",
     '"friction": (0.104, 0.128, 0.152, 0.176, 0.200),', '"friction": (0.104, 0.128, 0.152, 0.176),',
     (CHECKS[26],)),
    ("ip-stats-over-all-episodes",
     'ip = ok["ip_max_mm"]', 'ip = g["ip_max_mm"]',
     (CHECKS[27],)),
    # the unit of the arm comparison is the SEED; pooled episodes
    # (0/1 per row) give the same mean but a far narrower Welch interval.
    ("force-welch-over-pooled-episodes",
     "d = method_diff([p[KEY] for p in per[label][FORCE_ARMS[0]]], [p[KEY] for p in per[label][FORCE_ARMS[1]]],",
     "d = method_diff(*[sum(([1.0] * p['success_train_k'] + [0.0] * (p['n'] - p['success_train_k'])"
     " for p in per[label][m]), []) for m in FORCE_ARMS],",
     (CHECKS[28], CHECKS[29])),
    ("force-swaps-the-arms",
     "[p[KEY] for p in per[label][FORCE_ARMS[0]]], [p[KEY] for p in per[label][FORCE_ARMS[1]]]",
     "[p[KEY] for p in per[label][FORCE_ARMS[1]]], [p[KEY] for p in per[label][FORCE_ARMS[0]]]",
     (CHECKS[28], CHECKS[29])),
    ("force-loss-the-wrong-way",
     "loss = [noisy[s] - ref[s]", "loss = [ref[s] - noisy[s]",
     (CHECKS[30],)),
    ("force-pn2-reads-the-untagged-runs",
     'FORCE_BLOCKS = (("T-final", ""), ("pn2", "pn2"))', 'FORCE_BLOCKS = (("T-final", ""), ("pn2", ""))',
     (CHECKS[29], CHECKS[30], CHECKS[33])),
    ("k1-errors-in-the-force-root",
     '        if force_root(df):\n            out[kname] = {"missing"',
     '        if False:\n            out[kname] = {"missing"',
     (CHECKS[31],)),
    # FORCE in every root -> summary.json without autodr_nf changes.
    ("force-key-in-every-root",
     '    if force_root(df):\n        try:\n            out["FORCE"]',
     '    if True:\n        try:\n            out["FORCE"]',
     (CHECKS[32],)),
    # Patterson et al. 2024 Sec. 4.5: without the division the cross-check is the 95 % interval again.
    ("bonferroni-forgets-to-divide",
     "level = 1.0 - ALPHA / len(fam)", "level = 1.0 - ALPHA",
     (CHECKS[34],)),
)
# Mutations apply to the code ABOVE this marker only -- the table itself
# carries every anchor a second time.
_SELF_TEST_MARKER = "# Self-test: checks on built data"


def self_test() -> int:
    print(f"[eval_stats] marker: {SCRIPT_MARKER}")
    src = pathlib.Path(__file__).read_text(encoding="utf-8")

    def _exec(text):
        ns = {"__name__": "eval_stats_under_test", "__file__": __file__}
        exec(compile(text, __file__, "exec"), ns)  # noqa: S102 - the point of the self-test
        return ns

    base = _run_checks(_exec(src))
    bad = 0
    for name in CHECKS:
        ok = base.get(name) is True
        print(f"[eval_stats]   {'PASS' if ok else 'FAIL'}  {name}")
        bad += not ok
    cut = src.index(_SELF_TEST_MARKER)
    head, tail = src[:cut], src[cut:]
    for mname, old, new, want in MUTATIONS:
        if head.count(old) != 1:
            print(f"[eval_stats]   mutation {mname}: anchor matches {head.count(old)} times -- FAIL")
            bad += 1
            continue
        got = _run_checks(_exec(head.replace(old, new) + tail))
        flipped = {k for k, v in got.items() if v is not True}
        ok = flipped == set(want)
        print(f"[eval_stats]   mutation {mname}: {'PASS' if ok else 'FAIL'}"
              + ("" if ok else f" (flipped {sorted(flipped)}, declared {sorted(want)})"))
        bad += not ok
    undefended = [c for c in CHECKS if not any(c in w for *_, w in MUTATIONS)]
    print(f"[eval_stats]   every check is flipped by some mutation: "
          f"{'PASS' if not undefended else 'FAIL ' + str(undefended)}")
    bad += bool(undefended)
    total = len(CHECKS) + len(MUTATIONS) + 1
    print(f"[eval_stats] self-test: {total - bad}/{total}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--results", type=str, default="results/eval")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    results = pathlib.Path(args.results)
    df = pd.read_csv(results / "episodes_all.csv", keep_default_na=False,
                     dtype={"tag": str, "boundary_key": str})
    summary = summarize(df, t1_mm=t1_from_results(results, set(df["block"])))
    write_outputs(results, summary)
    errors = 0
    for k in ("K1", "K2", "K3"):
        v = summary.get(k)
        if isinstance(v, dict) and "met" in v:
            print(f"[eval_stats]   {k} met: {v['met']}")
        elif isinstance(v, dict):
            print(f"[eval_stats]   {k} NOT COMPUTED: {v}")
            errors += "error" in v
    for k, _, _ in K5_BLOCKS:
        v = summary.get(k, {})
        if "reading" in v:
            print(f"[eval_stats]   {k} reading: {v['reading']} (Welch {v['welch_ci'][0]:+.4f} .. {v['welch_ci'][1]:+.4f})")
        else:
            print(f"[eval_stats]   {k} NOT COMPUTED: {v}")
            errors += "error" in v
    v = summary.get("BONFERRONI", {})
    for k in v.get("family", []):
        lo, hi = v[k]["welch_ci"]
        print(f"[eval_stats]   {k} Bonferroni cross-check (k = {v['k']}, level {v['confidence_level']:.4f}): "
              f"{v[k]['reading']} (Welch {lo:+.4f} .. {hi:+.4f})")
    for name in ("OOD", "NOISE"):
        v = summary.get(name)
        if isinstance(v, dict) and "error" in v:
            print(f"[eval_stats]   {name} NOT COMPUTED: {v['error']}")
            errors += 1
        elif isinstance(v, dict):
            for p in v["policies"].values():
                print(f"[eval_stats]   {name} (descriptive, no pass line): {len(p['blocks'])} blocks of "
                      f"{p['policy']['method']} s{p['policy']['seed']} vs T-final tag {p['reference']['tag']!r}")
    v = summary.get("FORCE")
    if isinstance(v, dict) and "error" in v:
        print(f"[eval_stats]   FORCE NOT COMPUTED: {v['error']}")
        errors += 1
    elif isinstance(v, dict):
        for label, b in v["blocks"].items():
            if "welch_ci" in b:
                print(f"[eval_stats]   FORCE {label} (descriptive): {FORCE_ARMS[0]} - {FORCE_ARMS[1]} = {b['diff']:+.4f} "
                      f"(Welch {b['welch_ci'][0]:+.4f} .. {b['welch_ci'][1]:+.4f}, seeds {b['n_seeds']})")
            else:
                print(f"[eval_stats]   FORCE {label} NOT COMPUTED: {b}")
    print(f"[eval_stats] wrote {results / 'summary.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
