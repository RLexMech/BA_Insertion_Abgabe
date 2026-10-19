"""H21: one T-ood case at friction 0.23 (table row 906) over time, Auto-DR seed 2
(time limit) against seed 4 (success): true insertion depth and smoothed force
(thesis 5.2.1; RT-271).

Source: RT-271 (`rt_logs/RT-271_expectation.md`, OUTCOME), `rl_games/play.py
--trace-obs` on the four-row view table of T-ood friction_hi, env 0 = row 906,
FIRST episode only. Both values come from the trace's `truth` block, read
before the observation noise: depth = -tip_true_m z (pocket frame,
`insertion_math.part_tip_pose`), force = |force_ema_n| (the EMA the force abort
reads); a step with `wrench_valid` false carries no force. Two steps, because
the traces are git-ignored and the figure must redraw from a tracked file:

    .venv-analysis/Scripts/python.exe scripts/plot_trace_verklemmen.py --extract
        rt_logs/RT-271/RT-271_s{2,4}_trace.json -> results/traces/RT-271_zeile906.csv
    .venv-analysis/Scripts/python.exe scripts/plot_trace_verklemmen.py
        -> docs/figures/eval_rlgames/H21_verklemmen_zeile906.{pdf,svg,png}

Reference lines: the success depth and the pocket entrance (depth 0), both from
the run's own manifest (`results/eval_rlgames/T-ood/autodr_s2/eval_manifest.json`
`env_cfg.fixed_asset.success_depth`; the depth is measured from the entrance).
One trace per seed: an exemplary case, no rate. Built 2026-10-09.
"""
import argparse
import csv
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

NAME = "H21_verklemmen_zeile906"
ENV = 0          # view table row 0 = T-ood table row 906 (RT-271 expectation, step 2)
SEEDS = ("2", "4")
TRACE = REPO / "rt_logs" / "RT-271" / "RT-271_s{}_trace.json"
DATA = REPO / "results" / "traces" / "RT-271_zeile906.csv"
MANIFEST = REPO / "results" / "eval_rlgames" / "T-ood" / "autodr_s2" / "eval_manifest.json"
OUT = REPO / "docs" / "figures" / "eval_rlgames"
COLUMNS = ("seed", "step", "t_s", "depth_mm", "force_n", "done")


def extract() -> None:
    rows = []
    for s in SEEDS:
        tr = json.loads(pathlib.Path(str(TRACE).format(s)).read_text(encoding="utf-8"))
        if tr["num_envs"] != 4 or "truth" not in tr:
            raise SystemExit(f"[{NAME}] seed {s}: expected the 4-env RT-271 trace with a truth block")
        dt, truth, done = float(tr["step_dt_s"]), tr["truth"], tr["done"]
        end = next((k for k, d in enumerate(done) if d[ENV]), None)
        if end is None:
            raise SystemExit(f"[{NAME}] seed {s}: env {ENV} never finished its first episode")
        for k in range(end + 1):
            f = truth["force_ema_n"][k][ENV]
            valid = bool(truth["wrench_valid"][k][ENV])
            rows.append([s, k, f"{k * dt:.6f}", f"{-1000.0 * truth['tip_true_m'][k][ENV][2]:.4f}",
                         f"{math.sqrt(sum(v * v for v in f)):.4f}" if valid else "", int(bool(done[k][ENV]))])
        print(f"[{NAME}] seed {s}: first episode {end + 1} steps, end depth {rows[-1][3]} mm")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"[{NAME}] wrote {DATA} ({len(rows)} rows)")


def draw() -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import eval_figures as EF
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": NAME, "font.family": "sans-serif", "font.sans-serif": [EF.FONT]})
    with open(DATA, encoding="utf-8") as fh:
        data = list(csv.DictReader(fh))
    if tuple(data[0]) != COLUMNS:
        raise EF.FigError(f"{DATA} has columns {tuple(data[0])}, expected {COLUMNS}")
    cfg = json.loads(MANIFEST.read_text(encoding="utf-8"))["env_cfg"]
    seat_mm = 1000.0 * float(cfg["fixed_asset"]["success_depth"])
    cap_s = float(cfg["episode_length_s"])
    # Not dashed: dashed means "Ablation" (Grafik_Regelwerk D4). Two lines of one method:
    # the one under study in the method colour, the reference in grey (D7), 2026-10-11.
    style = {"2": dict(ls="-", color=EF.METHOD_COLOR["autodr"], label="Auto-DR Seed 2 (Zeitlimit)"),
             "4": dict(ls="-", color=EF.INK_MUTED, label="Auto-DR Seed 4 (Erfolg)")}
    fig, (ax_d, ax_f) = plt.subplots(2, 1, sharex=True, figsize=(EF.FIG_W, 3.9))
    for s in SEEDS:
        r = [x for x in data if x["seed"] == s]
        t = [float(x["t_s"]) for x in r]
        kw = dict(lw=1.6, **style[s])
        ax_d.plot(t, [float(x["depth_mm"]) for x in r], **kw)
        ax_f.plot(t, [float(x["force_n"]) if x["force_n"] else float("nan") for x in r], **kw)
    # Seat label above its line, entrance label below its line: seed 2 rests just above 0.
    for y, text, va in ((seat_mm, f"Erfolgstiefe {EF._num(seat_mm, '.0f')} mm", "bottom"),
                        (0.0, "Taschen-Eingang", "top")):
        ax_d.axhline(y, color=EF.INK_MUTED, lw=0.8, ls=":", zorder=1)
        ax_d.text(cap_s, y - 1.5 if va == "top" else y, text, color=EF.INK, fontsize=8, ha="right", va=va)   # C5
    ax_d.set_title("Einführtiefe", loc="left", fontsize=9, color=EF.INK)
    ax_f.set_title("Kraft am Werkzeug (geglättet)", loc="left", fontsize=9, color=EF.INK)
    EF._style(ax_d, ylabel="Tiefe in mm")
    EF._style(ax_f, ylabel="Kraft in N", xlabel="Zeit in der Episode in s")
    ax_d.set_xlim(0, cap_s)
    ax_f.set_ylim(bottom=0)
    h, l = ax_d.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "svg", "png"):
        p = OUT / f"{NAME}.{ext}"
        fig.savefig(p, dpi=EF.PNG_DPI, bbox_inches="tight", metadata=EF.SAVE_META[ext])
        written.append(p)
    plt.close(fig)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract", action="store_true", help="step 1: RT-271 traces -> results/traces CSV")
    a = ap.parse_args()
    if a.extract:
        extract()
    else:
        for p in draw():
            print(f"[{NAME}] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
