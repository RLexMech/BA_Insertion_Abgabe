# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Success rate over START HEIGHT, one bar per PROBE RUN.

WHAT IT DRAWS
-------------
One bar per ``demo_metrics.json``, at the FIXED start height that run was
measured at, with the episode count under the bar. This is the
generalisation probe: a policy trained on starts sampled in [-30, +30] mm
is replayed at heights it never saw, one run per height, and the bars show
where it stops working.

WHY THIS IS NOT ``plot_start_height_bins.py``. That script draws the six
bins INSIDE one run. Its bin edges are the environment's and stop at
+30 mm (``insertion_env.py:677``), and the last bin catches everything at
or above the last edge (``insertion_env.py:2056``). A run started at
+40 mm therefore lands entirely in a bin LABELLED "+20..+30 mm" -- one bar,
wrong label. Across-run heights need their own instrument, and this is it.

INPUT
-----
``--metrics FILE`` once per run, in the order the bars should appear.
Each file must be a FIXED-height run: ``start_tip_above_entrance_mm`` and
``start_tip_above_entrance_low_mm`` equal. A sampled run is refused --
its episodes have no single height, so a single bar would state something
false. ``--label`` may be given once per file for the legend line; without
it the height is the label.

Offline, no Isaac: ``json`` + ``matplotlib``.

    python scripts/plot_probe_start_heights.py \
        --metrics run_h30/demo_metrics.json --metrics run_h40/demo_metrics.json \
        --run RT-157 --out docs/Blockberichte/RT-157_probe_starthoehen.pdf
    python scripts/plot_probe_start_heights.py --self-test

``--self-test`` builds SYNTHETIC metrics files in a temporary directory,
checks the reader on a good file, on a sampled file (must be refused) and
on a file without a success rate (must be refused), draws once to a
temporary path, and never writes under ``docs/``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

SCRIPT_MARKER = "plot_probe_start_heights-2026-09-05"

# The training distribution of the policy this probe replays. Drawn as a
# shaded band so a reader sees at a glance which bars are in-distribution.
# NOT a threshold and not read from the files: it is the caller's context,
# passed in, with this pair only as the default for the RT-156 policy.
TRAINED_RANGE_MM = (-30.0, 30.0)


class ProbeError(Exception):
    """A metrics file that cannot be drawn as one bar. Always fatal."""


def read_probe(path: pathlib.Path) -> dict:
    """Read ONE fixed-height run. Raises ProbeError on anything ambiguous."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"{path}: not readable as JSON ({exc})") from exc

    high = data.get("start_tip_above_entrance_mm")
    low = data.get("start_tip_above_entrance_low_mm")
    if high is None:
        raise ProbeError(
            f"{path}: no 'start_tip_above_entrance_mm'. A home-pose run has no "
            "start height to plot."
        )
    # Guard on EQUALITY, not on presence: a sampled run carries both keys and
    # would otherwise be drawn at its upper bound, which no episode used.
    if low is not None and float(low) != float(high):
        raise ProbeError(
            f"{path}: start height is SAMPLED in [{low}, {high}] mm, not fixed. "
            "One bar cannot stand for a range -- use plot_start_height_bins.py "
            "for a sampled run."
        )

    rate = data.get("success_rate_recent")
    if rate is None:
        raise ProbeError(
            f"{path}: no 'success_rate_recent'. The run wrote no metrics window "
            "(the env dumps every 250 episodes) -- it was too short."
        )

    episodes = data.get("recent_window_episodes")
    counted = data.get("episodes")
    if counted is not None and episodes is not None:
        n = min(int(counted), int(episodes))
    else:
        n = int(counted or episodes or 0)

    forces = data.get("force_norm_n") or {}
    return {
        "path": path,
        "height_mm": float(high),
        "success": float(rate),
        "episodes": n,
        "force_p95_n": forces.get("p95_n"),
        "abort_rate": data.get("force_abort_rate"),
        "clamp_m": (data.get("rl_placeholders") or {}).get("osc_pos_clamp_m"),
    }


def draw(probes: list[dict], out: pathlib.Path, run: str, note: str | None,
         trained: tuple[float, float], no_title: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heights = [p["height_mm"] for p in probes]
    rates = [p["success"] * 100.0 for p in probes]
    labels = [f"{h:+.0f}" for h in heights]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    lo, hi = trained
    inside = [lo <= h <= hi for h in heights]
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    if any(inside):
        ax.axvspan(-0.5, sum(inside) - 0.5, color="#cfe3f5", alpha=0.7, zorder=0,
                   label=f"trainiert: {lo:+.0f} bis {hi:+.0f} mm")

    colours = ["#3b7dbd" if i else "#c4732a" for i in inside]
    bars = ax.bar(range(len(probes)), rates, color=colours, width=0.62, zorder=3)

    for i, (bar, p) in enumerate(zip(bars, probes)):
        ax.text(bar.get_x() + bar.get_width() / 2, rates[i] + 2.0,
                f"{rates[i]:.0f} %", ha="center", va="bottom", fontsize=10, zorder=4)
        ax.text(i, -7.0, f"n = {p['episodes']}", ha="center", va="top", fontsize=8,
                color="0.35")

    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Feste Starthöhe der Spitze über der Öffnungsebene [mm]")
    ax.set_ylabel("Erfolgsrate [%]")
    ax.set_ylim(-14, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.6, len(probes) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", zorder=0)
    if any(inside):
        # NOT "lower left": that is where the per-bar episode counts sit,
        # and the legend covered the n of the first bar (observed 2026-09-06).
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    if not no_title:
        title = f"{run}: Erfolgsrate je fester Starthöhe" if run else \
            "Erfolgsrate je fester Starthöhe"
        if note:
            title += f"\n{note}"
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _self_test() -> int:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and cond

    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)

        good = d / "good.json"
        good.write_text(json.dumps({
            "start_tip_above_entrance_mm": 40.0,
            "start_tip_above_entrance_low_mm": 40.0,
            "success_rate_recent": 0.62,
            "episodes": 812,
            "recent_window_episodes": 2000,
            "force_norm_n": {"p95_n": 24.5},
            "force_abort_rate": 0.0,
            "rl_placeholders": {"osc_pos_clamp_m": 0.07},
        }), encoding="utf-8")
        p = read_probe(good)
        check("fixed-height file is read", p["height_mm"] == 40.0 and p["episodes"] == 812)
        check("force and clamp are carried through",
              p["force_p95_n"] == 24.5 and p["clamp_m"] == 0.07)

        sampled = d / "sampled.json"
        sampled.write_text(json.dumps({
            "start_tip_above_entrance_mm": 30.0,
            "start_tip_above_entrance_low_mm": -30.0,
            "success_rate_recent": 0.997,
            "episodes": 124006,
        }), encoding="utf-8")
        try:
            read_probe(sampled)
            check("a SAMPLED run is refused", False)
        except ProbeError:
            check("a SAMPLED run is refused", True)

        norate = d / "norate.json"
        norate.write_text(json.dumps({
            "start_tip_above_entrance_mm": 60.0,
            "start_tip_above_entrance_low_mm": 60.0,
        }), encoding="utf-8")
        try:
            read_probe(norate)
            check("a run without a success rate is refused", False)
        except ProbeError:
            check("a run without a success rate is refused", True)

        broken = d / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        try:
            read_probe(broken)
            check("unreadable JSON is refused", False)
        except ProbeError:
            check("unreadable JSON is refused", True)

        out = d / "fig.png"
        draw([p, {**p, "height_mm": 60.0, "success": 0.0, "episodes": 400}],
             out, "SELBSTTEST", "synthetisch", TRAINED_RANGE_MM, False)
        check("draw writes a file (synthetic, temp dir)", out.exists() and out.stat().st_size > 0)

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics", type=pathlib.Path, action="append", default=[],
                    help="demo_metrics.json of ONE fixed-height probe run; repeat per run")
    ap.add_argument("--run", default="", help="run label for the title, e.g. RT-157")
    ap.add_argument("--out", type=pathlib.Path, help="output path (.pdf or .png)")
    ap.add_argument("--note", default=None, help="text under the title")
    ap.add_argument("--trained-range-mm", type=float, nargs=2, default=list(TRAINED_RANGE_MM),
                    metavar=("LOW", "HIGH"), help="training start range, shaded (default -30 30)")
    ap.add_argument("--no-title", action="store_true",
                    help="omit the in-figure title (the report caption names it)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()
    if not a.metrics or a.out is None:
        ap.error("--metrics (at least one) and --out are required")

    print(f"[plot_probe_start_heights] marker: {SCRIPT_MARKER}")
    try:
        probes = [read_probe(p) for p in a.metrics]
    except ProbeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    probes.sort(key=lambda p: p["height_mm"])
    for p in probes:
        force = "?" if p["force_p95_n"] is None else f"{p['force_p95_n']:.2f} N"
        print(f"  {p['height_mm']:+6.1f} mm  n={p['episodes']:6d}  "
              f"{p['success'] * 100:6.1f} %  force p95 {force}  "
              f"abort {p['abort_rate']}  clamp {p['clamp_m']}")

    draw(probes, a.out, a.run, a.note, tuple(a.trained_range_mm), a.no_title)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
