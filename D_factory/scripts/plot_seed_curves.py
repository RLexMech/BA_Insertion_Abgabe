# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""F7 of ``eval_figures.py`` (training success over total env steps, mean line,
+-1 sample std band over the seeds) for rl_games seed chains, BEFORE any
evaluation exists. Laptop only, in ``.venv-analysis`` (D-189).

A thin caller: the band numbers (``_mean_std``), the drawing (``draw_f7``),
the font setup and the arm colours are ``eval_figures.py``'s; the x axis is
``tools/run_axis.py``'s (D-195 CURVE AXIS, rl_games rule: ``Episode/*`` at
step = epoch -> x = epoch x horizon_length x num_envs). What is own: reading
the seeds of an arm from ``rt_logs/<prefix>/<prefix><N>_scalars.csv`` with
their ``_agent.yaml`` / ``_env.yaml``, because ``eval_figures.extract_training``
is bound to the ten frozen rsl_rl runs (RT-206/207) and needs No-DR.

    .venv-analysis/Scripts/python.exe scripts/plot_seed_curves.py --arm autodr=RT-242s --out rt_logs/RT-242s
    ... --arm autodr=RT-242s --arm fixeddr=RT-243s   (both arms, once RT-243s exists)
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

TAG = "Episode/success_rate"


def load_arm(rt_logs: pathlib.Path, prefix: str) -> dict:
    """One arm: every ``<prefix><N>`` under ``rt_logs/<prefix>/``, in seed order."""
    folder = rt_logs / prefix
    runs = sorted(p.name[: -len("_scalars.csv")] for p in folder.glob(f"{prefix}*_scalars.csv"))
    if len(runs) < 2:
        raise EF.FigError(f"{folder}: {len(runs)} run(s) with a scalars.csv; the band needs two or more")
    xs, seeds = None, {}
    for run in runs:
        lib, (horizon, envs) = EF.run_axis.read_run_factor(folder / f"{run}_agent.yaml", folder / f"{run}_env.yaml")
        if lib != "rl_games":
            raise EF.FigError(f"{run}: {lib} run; this caller reads rl_games runs only")
        frames = horizon * envs
        epochs, rates, ep_pairs = [], [], []
        with open(folder / f"{run}_scalars.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                step = int(float(r["step"]))
                if r.get("info/epochs", ""):
                    ep_pairs.append((step, float(r["info/epochs"])))
                if r.get(TAG, ""):
                    epochs.append(step)
                    rates.append(float(r[TAG]))
        EF.run_axis.check_rl_games_epochs(ep_pairs, frames)
        order = np.argsort(epochs)
        x = np.array([EF.run_axis.rl_games_x(TAG, epochs[i], frames) for i in order], dtype=np.int64)
        if xs is None:
            xs = x
        elif not np.array_equal(x, xs):
            raise EF.FigError(f"{run} logs other env steps than {runs[0]}")
        seeds[int(run[len(prefix):])] = np.asarray(rates, dtype=float)[order]
    mean, std = EF._mean_std(list(seeds.values()), "F7")
    return {"x": xs, "seeds": seeds, "mean": mean, "std": std, "runs": runs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", required=True, metavar="ARM=PREFIX",
                    help=f"arm ({', '.join(EF.ARMS)}) = run prefix under --rt-logs, e.g. autodr=RT-242s")
    ap.add_argument("--rt-logs", default=str(REPO / "rt_logs"))
    ap.add_argument("--out", required=True, help="folder for <prefixes>_F7_lernkurven.pdf/.png")
    args = ap.parse_args()
    d, prefixes = {}, []
    for spec in args.arm:
        arm, _, prefix = spec.partition("=")
        if arm not in EF.ARMS or not prefix:
            raise SystemExit(f"--arm {spec!r}: want <{'|'.join(EF.ARMS)}>=<run prefix>")
        d[arm] = load_arm(pathlib.Path(args.rt_logs), prefix)
        prefixes.append(prefix)
        a = d[arm]
        print(f"[plot_seed_curves] {arm}: {', '.join(a['runs'])}; {len(a['x'])} points, x {a['x'][0]}..{a['x'][-1]}; "
              f"last mean {a['mean'][-1]:.4f} std {a['std'][-1]:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.findfont(font_manager.FontProperties(family=EF.FONT), fallback_to_default=False)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.family": "sans-serif", "font.sans-serif": [EF.FONT]})
    fig = EF.draw_f7(plt, {m: d[m] for m in EF.ARMS if m in d})
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"{'_'.join(prefixes)}_{EF.NAMES['F7']}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", metadata={"CreationDate": None} if ext == "pdf" else None)
        print(f"[plot_seed_curves] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
