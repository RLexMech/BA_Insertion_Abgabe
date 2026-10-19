"""H23 / H24: force and interpenetration over ALL trials (successes and failures),
Auto-DR against Fixed-DR, the main test beside the robustness blocks.

  H23  thesis 5.2.2: the three doubled observation errors (pn2, gx2, py2), H20's
       policies (``eval_figures.data_h20``). Every block replays the main test's
       1000 rows; only one observation error differs.
  H24  thesis 5.2.1: the six T-ood blocks at 1.5x (H18_KEYS), H19's policies
       (``eval_figures.data_h19``). The blocks play their OWN table (150 rows each,
       the boundary table's independent stream, ``make_eval_table.py`` kind ood),
       so the main test is a reference with other test cases.

Both use the policies with >= 90 % in their own main test. Per policy and
group, over ALL its trials: the peak filtered force (``force_max_filtered_n``, the
signal the force abort reads) and the largest interpenetration over the episode
(``ip_max_mm``): median, quartiles and the 5 / 95 % values by the env's nearest-rank
rule (``eval_stats.pct_nearest_rank``). The bars are the spread of the trials, not
intervals; no mean over the seeds (Fixed-DR has two policies). Above the
force limit: per policy the trials whose peak force reached it (``force_abort``,
read from the force as ``eval_stats`` counts it). The main-test group is shaded
(the training range as a grey area). The success counts and the abort
counts are held against summary.json (``NOISE`` / ``OOD`` and ``policies``); a
mismatch stops the figure.

Input: ``daten/endstudie_rl_games/evaluation/`` (summary.json, episodes_all.csv, T-ood manifests).
Reuses ``eval_figures`` / ``eval_stats`` and changes nothing there. ``.venv-analysis``:

    .venv-analysis/Scripts/python.exe scripts/plot_kraft_durchdringung.py [--only H23|H24]
    .venv-analysis/Scripts/python.exe scripts/plot_kraft_durchdringung.py --self-test

Writes ``docs/figures/eval_rlgames/H23_beobachtungsfehler_kraft_durchdringung.*`` and
``H24_ausserhalb_kraft_durchdringung.*`` (pdf, svg, png).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import eval_figures as EF  # noqa: E402
from eval_figures import FigError  # noqa: E402

ES = EF.ES
NAMES = {"H23": "H23_beobachtungsfehler_kraft_durchdringung", "H24": "H24_ausserhalb_kraft_durchdringung"}
RESULTS = REPO / "daten" / "endstudie_rl_games" / "evaluation"
OUT = REPO / "docs" / "figures" / "eval_rlgames"
MAIN = "main"   # group key of the main test (T-final / T-final-F, untagged)
OOD_BLOCK = "T-ood"
FORCE = "force_max_filtered_n"
ABORT = "force_abort"   # eval_collect: peak filtered force >= limit
SHADE = "#ebebeb"       # Shading for the main-test group
PANELS = ((FORCE, "Spitzenkraft je Versuch (geglättet)", "Kraft in N"),
          ("ip_max_mm", "Größte Durchdringung je Versuch", "Durchdringung in mm"))
QUANTILES = (("p05", 0.05), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("p95", 0.95))
XLABEL = {"H23": "Verdoppelter Beobachtungsfehler (je Block einer)",
          "H24": "Störgröße beim 1,5-fachen Abstand ihrer Grenze vom Nennwert"}
SHADE_LABEL = {"H23": "Haupttest (Beobachtungsfehler wie im Training)",
               "H24": "Haupttest (alle Störgrößen im Prüfbereich)"}


def _rows(df: pd.DataFrame, kind: str, m: str, s: int, key: str) -> pd.DataFrame:
    """The trials of policy (m, s) in one group. The main test: its untagged run
    in ``final_block`` (eval_stats' NOISE and OOD reference). ``noise``: the same
    block, tagged with the key. ``ood``: block T-ood, ``boundary_key`` = key."""
    pol = (df["method"] == m) & (df["seed"] == s)
    if key == MAIN:
        return df[pol & (df["block"] == ES.final_block(m)) & (df["tag"] == ES.NOISE_REFERENCE_TAG)]
    if kind == "noise":
        return df[pol & (df["block"] == ES.final_block(m)) & (df["tag"] == key)]
    return df[pol & (df["block"] == OOD_BLOCK) & (df["boundary_key"] == key) & (df["tag"] == "")]


def _summary_aborts(policies: list, m: str, s: int, block: str, tag: str) -> tuple:
    """(aborts, n) of one run as eval_stats wrote it (force_abort_rate * n)."""
    hit = [p for p in policies if (p["method"], p["seed"], p["block"], p["tag"]) == (m, s, block, tag)]
    if len(hit) != 1:
        raise FigError(f"summary.json policies: {len(hit)} entries for {m} s{s} {block} {tag!r}")
    return round(hit[0]["force_abort_rate"] * hit[0]["n"]), hit[0]["n"]


def data_kd(learned: dict, section: dict, policies: list, df: pd.DataFrame, kind: str, fig: str) -> dict:
    """Per arm and shown policy (``learned``: data_h20 / data_h19), per group (main
    test, then ``learned['keys']``): the QUANTILES of each panel column over ALL
    trials and the count of trials at the force limit. ``section``: summary.json
    NOISE / OOD ``policies``; ``policies``: summary.json ``policies``. Refused: a
    success or abort count that differs from summary.json, an empty group,
    unequal test-case counts within a group, more than one force limit."""
    keys = [MAIN] + list(learned["keys"])
    out = {"keys": keys, "labels": ["Haupttest"] + list(learned["labels"]), "left_out": learned["left_out"],
           "arms": {}}
    ns, limits = {k: set() for k in keys}, set()
    for m, a in learned["arms"].items():
        arm = {"seeds": list(a["seeds"]), "aborts": [],
               "panels": {c: {q: [] for q, _v in QUANTILES} for c, *_r in PANELS}}
        for s in a["seeds"]:
            pol = section[f"{m}_s{s}"]
            row = {c: {q: [] for q, _v in QUANTILES} for c, *_r in PANELS}
            aborts = []
            for key in keys:
                b = _rows(df, kind, m, s, key)
                if b.empty:
                    raise FigError(f"{fig}: {m} s{s} {key} has no trials")
                won = b[b[EF.KEY].astype(int) == 1]
                want = pol["reference"] if key == MAIN else pol["blocks"][key]
                if (len(won), len(b)) != (want["k"], want["n"]):
                    raise FigError(f"{fig}: {m} s{s} {key} has {len(won)}/{len(b)} successes, "
                                   f"summary.json {want['k']}/{want['n']}")
                ns[key].add(len(b))
                limits |= set(b["force_limit_n"].astype(float))
                aborts.append(int(b[ABORT].astype(int).sum()))
                if key == MAIN or kind == "noise":
                    tag = ES.NOISE_REFERENCE_TAG if key == MAIN else key
                    if (aborts[-1], len(b)) != _summary_aborts(policies, m, s, ES.final_block(m), tag):
                        raise FigError(f"{fig}: {m} s{s} {key} has {aborts[-1]} trials at the force limit, "
                                       f"summary.json {_summary_aborts(policies, m, s, ES.final_block(m), tag)}")
                for c, *_r in PANELS:
                    v = b[c].astype(float).tolist()
                    for q, p in QUANTILES:
                        row[c][q].append(ES.pct_nearest_rank(v, p))
            if kind == "ood":
                # summary.json pools the T-ood run over all its blocks (start height too)
                every = df[(df["method"] == m) & (df["seed"] == s) & (df["block"] == OOD_BLOCK) & (df["tag"] == "")]
                got = (int(every[ABORT].astype(int).sum()), len(every))
                if got != _summary_aborts(policies, m, s, OOD_BLOCK, ""):
                    raise FigError(f"{fig}: {m} s{s} T-ood has {got} trials at the force limit, "
                                   f"summary.json {_summary_aborts(policies, m, s, OOD_BLOCK, '')}")
            arm["aborts"].append(aborts)
            for c in row:
                for q in row[c]:
                    arm["panels"][c][q].append(row[c][q])
        for c in arm["panels"]:
            arm["panels"][c] = {q: np.asarray(v, dtype=float) for q, v in arm["panels"][c].items()}
        arm["aborts"] = np.asarray(arm["aborts"], dtype=int)
        out["arms"][m] = arm
    if any(len(v) != 1 for v in ns.values()):
        raise FigError(f"{fig}: test cases per policy differ within a group: {ns}")
    if len(limits) != 1:
        raise FigError(f"{fig}: more than one force limit: {sorted(limits)}")
    out["n"] = {k: v.pop() for k, v in ns.items()}
    out["limit"] = limits.pop()
    return out


def data_h23(summary: dict, df: pd.DataFrame) -> dict:
    return data_kd(EF.data_h20(summary), summary["NOISE"]["policies"], summary["policies"], df, "noise", "H23")


def data_h24(summary: dict, df: pd.DataFrame, sidecar: dict) -> dict:
    return data_kd(EF.data_h19(summary, sidecar), summary["OOD"]["policies"], summary["policies"], df, "ood", "H24")


def _slots(arms: dict) -> dict:
    """x offset in a group per (arm, policy index): the arms side by side in
    H20's order, half an empty slot between them, over 0.92 of the group (H24's
    seven groups: one digit of the abort count per slot must stay apart)."""
    total = sum(len(a["seeds"]) for a in arms.values()) + 0.5 * (len(arms) - 1)
    step, j, pos = 0.92 / total, 0.0, {}
    for m, a in arms.items():
        for i in range(len(a["seeds"])):
            pos[(m, i)] = -0.46 + step * (j + 0.5)
            j += 1
        j += 0.5
    return pos


def _gone(d: dict) -> str:
    return "; ".join(f"{EF.METHOD_LABEL[m]} Seed {', '.join(str(s) for s, _r in lst)} (Haupttest "
                     + " / ".join(EF._num(100 * r, ".1f") for _s, r in lst) + " %)"
                     for m, lst in d["left_out"].items() if lst)


def draw(plt, d, fig_key: str):
    """Two stacked panels, one group per test; per policy a median point, a thick
    bar over the middle half and a thin one over 90 % of its trials; the trials at
    the force limit as a count above it; the main test shaded."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 1, figsize=(EF.FIG_W, 6.0), sharex=True)
    x = np.arange(len(d["keys"]))
    pos = _slots(d["arms"])
    lim = d["limit"]
    for ax, (col, title, ylabel) in zip(axes, PANELS):
        ax.axvspan(-0.5, 0.5, color=SHADE, linewidth=0, zorder=0)
        for m, a in d["arms"].items():
            c, mk, p = EF.METHOD_COLOR[m], EF.H18_MARKER[m], a["panels"][col]
            for i in range(len(a["seeds"])):
                xs = x + pos[(m, i)]
                ax.vlines(xs, p["p05"][i], p["p95"][i], color=c, linewidth=0.7, alpha=0.7, zorder=2)
                ax.vlines(xs, p["q25"][i], p["q75"][i], color=c, linewidth=2.2, zorder=3)
                ax.scatter(xs, p["median"][i], s=18, marker=mk, color=c, edgecolors="white", linewidths=0.5,
                           zorder=4, clip_on=False)
                if col == FORCE:
                    for xj, k in zip(xs, a["aborts"][i]):
                        ax.text(xj, lim + 0.6, str(k), ha="center", va="bottom", fontsize=7,
                                color=c if k else EF.INK_MUTED)
        if col == FORCE:
            ax.axhline(lim, color=EF.INK_MUTED, linewidth=1.0, linestyle="--", zorder=1)
            ax.set_ylim(0, lim + 3.6)
            ax.set_yticks(np.arange(0, lim + 0.1, 5))
        else:
            ax.set_ylim(bottom=0)
        ax.set_xticks(x, d["labels"])
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_title(title, loc="left", fontsize=9, color=EF.INK, pad=10 if col == FORCE else 6)
        EF._style(ax, ylabel)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", visible=False)
    axes[1].set_xlabel(XLABEL[fig_key], color=EF.INK, fontsize=9, loc="right")
    grey = EF.INK_MUTED
    handles = [Line2D([], [], color=EF.METHOD_COLOR[m], marker=EF.H18_MARKER[m], markersize=5,
                      markeredgecolor="white", linestyle="none",
                      label=f"{EF.METHOD_LABEL[m]}, {len(a['seeds'])} Policies") for m, a in d["arms"].items()]
    handles += [Line2D([], [], color=grey, marker="o", markersize=4, linestyle="none",
                       label="Punkt = Median aller Versuche einer Policy"),
                Line2D([], [], color=grey, linewidth=2.2, label="dicker Strich = mittlere Hälfte der Versuche"),
                Line2D([], [], color=grey, linewidth=0.7, label="dünner Strich = 90 % der Versuche (5 bis 95 %)"),
                Line2D([], [], color=grey, linewidth=1.0, linestyle="--",
                       label=f"Kraftgrenze {EF._num(lim, '.0f')} N; Zahl darüber = Versuche\n"
                             f"der Policy, die sie erreichten"),
                Patch(facecolor=SHADE, linewidth=0, label=SHADE_LABEL[fig_key])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False,
               fontsize=8)
    rule = f"Gezeigt: Policies mit mindestens {EF._num(100 * EF.LEARNED_MIN, '.0f')} % im eigenen Haupttest."
    gone = _gone(d)
    if gone:
        rule += f" Nicht gezeigt: {gone}."
    every = "Alle Versuche, Erfolge und Fehlversuche."
    if fig_key == "H24":
        blocks = {d["n"][k] for k in d["keys"][1:]}
        tests = (f"Haupttest: {EF._thousands(d['n'][MAIN])} Testfälle aus dem ganzen Prüfbereich. Je Block "
                 f"{EF._thousands(blocks.pop())} andere Testfälle: eine Störgröße außerhalb, die übrigen im Prüfbereich.")
    else:
        tests = (f"Je Gruppe dieselben {EF._thousands(d['n'][MAIN])} Testfälle; je Block ist ein "
                 f"Beobachtungsfehler verdoppelt.")
    note = f"{tests}\n{every} {rule}"
    fig.text(0.5, 0.0, note, ha="center", va="bottom", fontsize=7.5, color=EF.INK)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    return fig


def render(only=None) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(RESULTS / "episodes_all.csv", keep_default_na=False, dtype={"tag": str, "boundary_key": str})
    data = {"H23": lambda: data_h23(summary, df),
            "H24": lambda: data_h24(summary, df, EF._ood_inputs(RESULTS, EF._ood_sidecar_path()))}
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for key in only or NAMES:
        plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": NAMES[key],
                             "font.family": "sans-serif", "font.sans-serif": [EF.FONT]})
        fig = draw(plt, data[key](), key)
        for ext in ("pdf", "svg", "png"):
            p = OUT / f"{NAMES[key]}.{ext}"
            fig.savefig(p, dpi=EF.PNG_DPI, bbox_inches="tight", metadata=EF.SAVE_META[ext])
            written.append(p)
        plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# self-test: stub summaries + episode tables with known quantiles

CHANGED = {"pn2": {"obs_noise_pocket_pos_std_m": 0.01}, "gx2": {"grasp_obs_offset_x_m": 0.006},
           "py2": {"pocket_obs_yaw_rad": 0.0446804}}
LEARNED = {("autodr", s) for s in range(1, 6)} | {("fixeddr", 3), ("fixeddr", 4)}


def _trials(block, m, s, tag, key="", force=None, ip=None):
    """20 trials: 18 successes at 10..27 N and 0.1..1.8 mm, 2 failures at 99 N /
    9.9 mm, both at the force limit. Nearest rank over all 20: p05 = 1st, q25 = 5th,
    median = 10th, q75 = 15th, p95 = 19th -> 10 / 14 / 19 / 24 / 99 N and
    0.1 / 0.5 / 1.0 / 1.5 / 9.9 mm (successes only: median 18 N, p95 27 N).
    ``force`` / ``ip`` overwrite every trial (decoys)."""
    out = []
    for i in range(20):
        f = force if force is not None else (10.0 + i if i < 18 else 99.0)
        out.append({"block": block, "method": m, "seed": s, "tag": tag, "boundary_key": key, EF.KEY: int(i < 18),
                    FORCE: f, "ip_max_mm": ip if ip is not None else (0.1 * (i + 1) if i < 18 else 9.9),
                    "force_limit_n": 30.0, ABORT: int(f >= 30.0)})
    return out


def _pol_entry(m, s, block, tag, aborts, n):
    return {"method": m, "seed": s, "block": block, "tag": tag, "n": n, "force_abort_rate": aborts / n}


def _stub_noise(k_override=None, abort_override=None):
    """Five Auto-DR and five Fixed-DR policies, Fixed-DR s3/s4 learned (18/20)."""
    pols, rows, plist = {}, [], []
    for m in ("autodr", "fixeddr"):
        for s in range(1, 6):
            ok = (m, s) in LEARNED
            k = 18 if ok else 0
            blocks = {t: {"changed": CHANGED[t], "k": k, "n": 20, "rate": k / 20, "wilson": [0.0, 1.0]} for t in CHANGED}
            pols[f"{m}_s{s}"] = {"policy": {"method": m, "seed": s}, "blocks": blocks,
                                 "reference": {"block": ES.final_block(m), "tag": "", "k": k, "n": 20, "rate": k / 20}}
            if ok:
                for t in [""] + list(CHANGED):
                    rows += _trials(ES.final_block(m), m, s, t)
                    plist.append(_pol_entry(m, s, ES.final_block(m), t, 2, 20))
    if k_override:
        pols[k_override]["blocks"]["gx2"]["k"] = 17
    if abort_override:
        next(p for p in plist if (p["method"], p["seed"], p["tag"]) == abort_override)["force_abort_rate"] = 1 / 20
    return {"NOISE": {"policies": pols}, "policies": plist}, pd.DataFrame(rows)


def _stub_ood():
    """The learned policies with a main test and two T-ood blocks. Decoys that must
    not be read: a third T-ood block and a tagged main-test run, both at 50 N / 5 mm
    (the third block counts for summary.json's pooled T-ood abort total only)."""
    learned = {"keys": ["tilt_hi", "friction_hi"], "labels": ["Kippung", "Reibwert"], "left_out": {},
               "arms": {m: {"seeds": [s for mm, s in sorted(LEARNED) if mm == m]} for m in ("autodr", "fixeddr")}}
    section, rows, plist = {}, [], []
    for m, s in sorted(LEARNED):
        rows += _trials(ES.final_block(m), m, s, "")
        rows += _trials(ES.final_block(m), m, s, "pn2", force=50.0, ip=5.0)
        for k in learned["keys"]:
            rows += _trials(OOD_BLOCK, m, s, "", k)
        rows += _trials(OOD_BLOCK, m, s, "", "lat_r_hi", force=50.0, ip=5.0)
        section[f"{m}_s{s}"] = {"reference": {"k": 18, "n": 20},
                                "blocks": {k: {"k": 18, "n": 20} for k in learned["keys"] + ["lat_r_hi"]}}
        plist += [_pol_entry(m, s, ES.final_block(m), "", 2, 20), _pol_entry(m, s, OOD_BLOCK, "", 2 + 2 + 20, 60)]
    return learned, section, plist, pd.DataFrame(rows)


def self_test() -> int:
    res = {}
    want = {FORCE: (10.0, 14.0, 19.0, 24.0, 99.0), "ip_max_mm": (0.1, 0.5, 1.0, 1.5, 9.9)}

    def quantiles_ok(d):
        return all(np.allclose(a["panels"][c][q], v) for a in d["arms"].values() for c, vs in want.items()
                   for q, v in zip([q for q, _p in QUANTILES], vs))

    summary, df = _stub_noise()
    d = data_h23(summary, df)
    res["H23: shown policies = H20's rule (Auto-DR 1-5, Fixed-DR 3, 4; Fixed-DR 1, 2, 5 left out)"] = (
        d["arms"]["autodr"]["seeds"] == [1, 2, 3, 4, 5] and d["arms"]["fixeddr"]["seeds"] == [3, 4]
        and [s for s, _r in d["left_out"]["fixeddr"]] == [1, 2, 5])
    res["H23: quantiles over ALL trials incl. the failures, nearest rank (p05..p95 = 10/14/19/24/99 N)"] = (
        quantiles_ok(d))
    res["H23: groups = main test + pn2, gx2, py2; n, limit and 2 aborts per group read"] = (
        d["keys"] == [MAIN, "pn2", "gx2", "py2"] and set(d["n"].values()) == {20} and d["limit"] == 30.0
        and all((a["aborts"] == 2).all() for a in d["arms"].values()))
    for name, kw, needle in (("a success count", {"k_override": "autodr_s2"}, "autodr s2 gx2 has 18/20 successes"),
                             ("an abort count", {"abort_override": ("fixeddr", 4, "py2")}, "fixeddr s4 py2 has 2 trials")):
        summary, df = _stub_noise(**kw)
        label = f"H23: {name} that differs from summary.json stops the figure"
        try:
            data_h23(summary, df)
            res[label] = False
        except FigError as exc:
            res[label] = needle in str(exc)
    learned, section, plist, df = _stub_ood()
    d = data_kd(learned, section, plist, df, "ood", "H24")
    res["H24: main test untagged, T-ood rows by boundary_key (decoys at 50 N / 5 mm not read)"] = (
        quantiles_ok(d) and d["keys"] == [MAIN, "tilt_hi", "friction_hi"])
    for name, ok in res.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad = [n for n, ok in res.items() if not ok]
    print(f"[plot_kraft_durchdringung] self-test {len(res) - len(bad)}/{len(res)}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--only", nargs="+", choices=sorted(NAMES))
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    for p in render(a.only):
        print(f"[plot_kraft_durchdringung] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
