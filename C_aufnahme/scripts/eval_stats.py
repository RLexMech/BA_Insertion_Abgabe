# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Every number of the evaluation, from ``episodes_all.csv`` (eval_plan.md section 5,
section 8b step 2). Laptop only, in ``.venv-analysis`` (D-189).

Headline key is ``success_train``. Library functions, no own interval code:

  Wilson          statsmodels proportion_confint(method="wilson")
  Newcombe (K3)   statsmodels confint_proportions_2indep(method="newcomb", compare="diff")
  K1/K2 headline  scipy ttest_ind(equal_var=False).confidence_interval  (Welch)
  K1/K2 beside    scipy bootstrap(paired=False, method="percentile")    (fixed seed)
  K5              the same two, AutoDR (T-final / T-nominal) - Fixed-DR (T-final-F /
                  T-nominal-F); a reading (above_0 / below_0 / contains_0), no "met"

The unit of the method comparison is the SEED: each policy is one rate
(literature check 2026-09-21). Percentiles of force and steps use the env's
nearest-rank rule (`insertion_env._pct`), so a thesis number and a training-log
number mean the same.

Writes ``summary.json`` (sorted keys; two runs on the same input are byte-equal)
and ``summary_tables.csv`` (one row per policy and block).

    .venv-analysis/Scripts/python.exe scripts/eval_stats.py --results results/eval
    .venv-analysis/Scripts/python.exe scripts/eval_stats.py --self-test
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

SCRIPT_MARKER = "eval_stats-2026-09-23b"
ALPHA = 0.05
BOOT_RESAMPLES = 10000
BOOT_SEED = 20260921
KEY = "success_train"

BIN_EDGES = {
    "lat_r_mm": (6.0, 12.0, 18.0, 24.0, 30.0),
    "yaw_abs_deg": (1.0, 2.0, 3.0, 4.0, 5.0),
    "tilt_deg": (2.0, 4.0, 6.0, 8.0, 10.0),
    "start_height_m": (-0.020, -0.010, 0.0, 0.010, 0.020, 0.040, 0.060, 0.080, 0.100, 0.120),
}


class StatsError(RuntimeError):
    """The input cannot answer the question the plan asks."""


def wilson(k: int, n: int) -> list:
    lo, hi = proportion_confint(int(k), int(n), alpha=ALPHA, method="wilson")
    return [float(lo), float(hi)]


def pct_nearest_rank(values, q: float) -> float | None:
    """`insertion_env._pct`: sorted[min(n-1, ceil(q*n)-1)]; None when empty."""
    v = sorted(float(x) for x in values)
    if not v:
        return None
    return v[min(len(v) - 1, int(math.ceil(q * len(v))) - 1)]


def policy_rows(df: pd.DataFrame) -> list:
    """One dict per (block, method, seed, tag): rates with Wilson, outcomes, force and steps over the SUCCESSES only (eval_plan.md section 4)."""
    out = []
    for (block, method, seed, tag), g in df.groupby(["block", "method", "seed", "tag"], sort=True):
        n = int(len(g))
        k = int(g[KEY].sum())
        k_strict = int(g["success_strict"].sum())
        k_abort = int(g["force_abort"].sum())
        ok = g[g[KEY] == 1]
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
        })
    return out


def method_diff(a, b, names=("autodr", "nodr")) -> dict:
    """``names[0] - names[1]`` over the per-seed rates (K1/K2: AutoDR - No-DR, K5: AutoDR - Fixed-DR)."""
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


K5_BLOCKS = (("K5", "T-final", "T-final-F"), ("K5_nominal", "T-nominal", "T-nominal-F"))


def k5_reading(ci) -> str:
    """The three pre-registered readings of eval_plan.md section 1.4; no "met"."""
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


def select_final(pol: list) -> dict:
    """eval_plan.md section 2 B, on T-auswahl: highest rate; tie -> smaller force p95 over the successes; tie -> smaller seed number."""
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
    """Each boundary block of the FINAL policy against its overall T-final rate, Newcombe difference (block - overall)."""
    sel = lambda b: df[(df["block"] == b) & (df["method"] == final["method"])
                       & (df["seed"] == final["seed"]) & (df["tag"] == "")]
    g, f = sel("T-grenze"), sel("T-final")
    if g.empty or f.empty:
        raise StatsError("K3 needs T-grenze AND T-final of the final policy")
    k_all, n_all = int(f[KEY].sum()), int(len(f))
    out = {}
    for key, b in g.groupby("boundary_key", sort=True):
        k, n = int(b[KEY].sum()), int(len(b))
        lo, hi = confint_proportions_2indep(k, n, k_all, n_all, method="newcomb", compare="diff", alpha=ALPHA)
        out[key] = {"k": k, "n": n, "rate": k / n, "wilson": wilson(k, n),
                    "diff_vs_final": k / n - k_all / n_all, "newcombe_ci": [float(lo), float(hi)]}
    return {"overall": {"k": k_all, "n": n_all, "rate": k_all / n_all}, "blocks": out,
            "met": all(v["newcombe_ci"][1] >= 0.0 for v in out.values())}


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


def summarize(df: pd.DataFrame) -> dict:
    pol = policy_rows(df)
    out = {"marker": SCRIPT_MARKER, "headline_key": KEY, "alpha": ALPHA, "policies": pol,
           "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": __import__("scipy").__version__,
                        "statsmodels": __import__("statsmodels").__version__}}
    blocks = set(df["block"])
    for kname, block in (("K1", "T-final"), ("K2", "T-nominal")):
        if block not in blocks:
            out[kname] = {"missing": f"no {block} runs"}
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
    if "T-auswahl" in blocks:
        out["final_policy"] = select_final(pol)
        if {"T-grenze", "T-final"} <= blocks:
            try:
                out["K3"] = k3_blocks(df, out["final_policy"])
            except StatsError as exc:
                out["K3"] = {"error": str(exc)}
    k4 = {}
    for m in ("autodr", "nodr"):
        r = [p[KEY] for p in pol if p["block"] == "T-final" and p["tag"] == "" and p["method"] == m]
        if r:
            k4[m] = {"rates": r, "min": min(r), "max": max(r),
                     "sd": float(np.std(r, ddof=1)) if len(r) > 1 else None}
    out["K4"] = k4
    fin = df[df["block"].isin(("T-final", "T-final-F")) & (df["tag"] == "")]
    out["bins_T_final"] = {
        f"{m}_s{s}": {c: bin_rates(g, c, e) for c, e in BIN_EDGES.items()}
        for (m, s), g in fin.groupby(["method", "seed"], sort=True)
    }
    out["bins_not_computed"] = {"friction": "no friction bin edges are defined"}
    return out


def write_outputs(results: pathlib.Path, summary: dict) -> None:
    with open(results / "summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    cols = ("block", "method", "seed", "tag", "n", "success_train", "success_train_k",
            "success_strict", "force_abort_rate")
    pd.DataFrame([{c: p[c] for c in cols} for p in summary["policies"]]).to_csv(
        results / "summary_tables.csv", index=False, lineterminator="\n")


# Self-test (D-080): checks on built data; each mutation of THIS file's source

N_EP = 100
K_AUTODR = (90, 92, 94, 96, 98)
K_NODR = (40, 42, 44, 46, 90)
K_FIXED = {"T-final-F": (70, 72, 74, 76, 78), "T-nominal-F": (80, 90, 95, 99, 100)}


def _frame() -> pd.DataFrame:
    rows = []
    for method, ks in (("autodr", K_AUTODR), ("nodr", K_NODR)):
        for seed, k in enumerate(ks, start=1):
            for block in ("T-final", "T-nominal", "T-auswahl"):
                for i in range(N_EP):
                    ok = int(i < k)
                    viol = int(ok and method == "autodr" and i < 2)
                    rows.append({"block": block, "method": method, "seed": seed, "tag": "",
                                 "row_id": i, "boundary_key": "", "outcome": "success" if ok else "timeout",
                                 "success_train": ok, "success_strict": ok * (1 - viol), "force_abort": viol,
                                 "steps": 40 + i, "force_max_filtered_n": 31.0 if viol else 10.0 + seed,
                                 "lat_r_mm": (i % 31) * 1.0, "yaw_abs_deg": (i % 6) * 1.0,
                                 "tilt_deg": (i % 11) * 1.0, "start_height_m": 0.02 + (i % 10) * 0.01})
    for block, ks in K_FIXED.items():
        for seed, k in enumerate(ks, start=1):
            for i in range(N_EP):
                ok = int(i < k)
                rows.append({"block": block, "method": "fixeddr", "seed": seed, "tag": "",
                             "row_id": i, "boundary_key": "", "outcome": "success" if ok else "timeout",
                             "success_train": ok, "success_strict": ok, "force_abort": 0,
                             "steps": 40 + i, "force_max_filtered_n": 10.0 + seed,
                             "lat_r_mm": (i % 31) * 1.0, "yaw_abs_deg": (i % 6) * 1.0,
                             "tilt_deg": (i % 11) * 1.0, "start_height_m": 0.02 + (i % 10) * 0.01})
    for j in range(100):
        rows.append({"block": "T-grenze", "method": "autodr", "seed": 5, "tag": "", "row_id": j,
                     "boundary_key": "lat_r_hi" if j < 50 else "yaw_lo", "outcome": "success",
                     "success_train": int(j < 45 or j >= 50), "success_strict": int(j < 45 or j >= 50),
                     "force_abort": 0, "steps": 50, "force_max_filtered_n": 10.0,
                     "lat_r_mm": 30.0, "yaw_abs_deg": 0.0, "tilt_deg": 0.0, "start_height_m": 0.05})
    return pd.DataFrame(rows)


def _wilson_ref(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def _welch_ref(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = st.t.ppf(1 - ALPHA / 2, df) * math.sqrt(va + vb)
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
)


def _run_checks(ns) -> dict:
    res = {}
    res[CHECKS[0]] = all(np.allclose(ns["wilson"](k, 10), _wilson_ref(k, 10), atol=1e-9) for k in (0, 5, 10))
    df = _frame()
    s = ns["summarize"](df)
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
    res[CHECKS[8]] = lb[-1]["n"] == sum(1 for i in range(N_EP) if (i % 31) >= 24)
    res[CHECKS[9]] = ns["pct_nearest_rank"](range(1, 21), 0.95) == 19
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        ns["write_outputs"](t, s)
        first = (t / "summary.json").read_bytes()
        ns["write_outputs"](t, ns["summarize"](df))
        res[CHECKS[10]] = first == (t / "summary.json").read_bytes()
    f = [k / N_EP for k in K_FIXED["T-final-F"]]
    k5, k5n = s.get("K5", {}), s.get("K5_nominal", {})
    res[CHECKS[11]] = ("diff" in k5 and abs(k5["diff"] - 0.20) < 1e-12
                       and np.allclose(k5["welch_ci"], _welch_ref(a, f), atol=1e-9)
                       and k5["mean_fixeddr"] == float(np.mean(f)))
    res[CHECKS[12]] = (k5.get("reading") == "above_0" and k5n.get("reading") == "contains_0"
                       and ns["k5_reading"]((-0.3, -0.1)) == "below_0")
    bt = s["bins_T_final"]
    res[CHECKS[13]] = (sorted(bt) == sorted([f"autodr_s{i}" for i in range(1, 6)] + [f"nodr_s{i}" for i in range(1, 6)]
                                            + [f"fixeddr_s{i}" for i in range(1, 6)])
                       and bt["fixeddr_s1"]["lat_r_mm"][-1]["n"] == lb[-1]["n"])
    return res


MUTATIONS = (
    ("rate-denominator-n-minus-1",
     '"success_train": k / n, "success_train_k"', '"success_train": k / (n - 1), "success_train_k"',
     (CHECKS[1], CHECKS[2], CHECKS[3], CHECKS[5], CHECKS[11])),
    ("welch-pooled-variance",
     "welch = st.ttest_ind(a, b, equal_var=False)", "welch = st.ttest_ind(a, b, equal_var=True)",
     (CHECKS[2],)),
    ("bootstrap-falls-back-to-bca",
     '                        method="percentile", rng=', "                        rng=",
     (CHECKS[3],)),
    ("headline-reads-strict",
     'KEY = "success_train"', 'KEY = "success_strict"',
     (CHECKS[1], CHECKS[2], CHECKS[3], CHECKS[5], CHECKS[7], CHECKS[11])),
    ("wilson-becomes-normal",
     'alpha=ALPHA, method="wilson")', 'alpha=ALPHA, method="normal")',
     (CHECKS[0],)),
    ("k3-compares-the-wrong-way",
     'k, n, k_all, n_all, method="newcomb"', 'k_all, n_all, k, n, method="newcomb"',
     (CHECKS[7],)),
    ("bins-take-the-edge-into-the-lower-bin",
     'side="right")', 'side="left")',
     (CHECKS[8],)),
    ("bootstrap-seed-from-the-clock",
     "rng=np.random.default_rng(BOOT_SEED))", "rng=np.random.default_rng())",
     (CHECKS[10],)),
    ("k1-demands-half-a-rate",
     '(d["welch_ci"][0] > 0.0)', '(d["welch_ci"][0] > 0.5)',
     (CHECKS[4],)),
    ("selection-picks-the-lowest-rate",
     "key = lambda p: (-p[KEY],", "key = lambda p: (p[KEY],",
     (CHECKS[6], CHECKS[7])),
    ("p95-drops-the-ceil",
     "int(math.ceil(q * len(v))) - 1", "int(q * len(v))",
     (CHECKS[9],)),
    ("k5-swaps-the-arms",
     "d = method_diff(a, f, names=", "d = method_diff(f, a, names=",
     (CHECKS[11], CHECKS[12])),
    ("k5-takes-nodr-as-the-fixed-arm",
     'f = seed_rates(pol, fblock, "fixeddr")', 'f = seed_rates(pol, fblock, "nodr")',
     (CHECKS[11], CHECKS[12])),
    ("k5-reads-autodr-from-the-F-block",
     'a = seed_rates(pol, ablock, "autodr")', 'a = seed_rates(pol, fblock, "autodr")',
     (CHECKS[11], CHECKS[12])),
    ("k5-reading-looks-at-the-upper-bound",
     'return "above_0" if lo > 0.0', 'return "above_0" if hi > 0.0',
     (CHECKS[12],)),
    ("k5-below-reading-never-fires",
     '("below_0" if hi < 0.0', '("below_0" if hi < -1.0',
     (CHECKS[12],)),
    ("bins-ignore-the-fixed-dr-block",
     'fin = df[df["block"].isin(("T-final", "T-final-F")) & (df["tag"] == "")]',
     'fin = df[(df["block"] == "T-final") & (df["tag"] == "")]',
     (CHECKS[13],)),
)
_SELF_TEST_MARKER = "# Self-test (D-080): checks on built data"


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
    summary = summarize(df)
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
    print(f"[eval_stats] wrote {results / 'summary.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
