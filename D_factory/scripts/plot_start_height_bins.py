# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Success rate over START HEIGHT, one bar per bin, episodes per bin printed.

WHAT IT DRAWS
-------------
The ``success_by_start_height_bin`` table of one ``demo_metrics.json``
(plan step C / SBC, D-165): six bins of 10 mm over the opening plane, the
trailing-window success rate of each as a bar, the number of episodes that
fell in the bin under the bar. It draws exactly that table -- height only,
no tilt, no overlay -- because the table is the SBC readout (success in the
inside bins with 0 at +30 mm is IndustReal's "partially-inserted" overfit,
success at +30 mm is the claim the run has to earn) and the reader must see
how many episodes each percentage stands on.

INPUT
-----
``--metrics FILE`` is either a ``demo_metrics.json`` itself or any text file
that CONTAINS the table (an ``rt_logs`` dump, ``rt_logs/inbox.txt``). The
table is located by its key and decoded with ``json``; if the file holds it
more than once the LAST occurrence is used (a run's dump ends with its final
window). A table with a bin count other than six is refused: the bins are the
env's, not this script's, and a different count means a different env.

Offline, no Isaac: ``json`` + ``matplotlib``.

    python scripts/plot_start_height_bins.py --metrics rt_logs/inbox.txt --run RT-138 \
        --out docs/figures/RT-138_success_by_start_height.png
    python scripts/plot_start_height_bins.py --self-test

``--self-test`` draws a SYNTHETIC table (marked as such in its title) to a
temporary file, checks the extractor on JSON-in-text and on a refused row
count, and never writes under ``docs/``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

KEY = "success_by_start_height_bin"
EXPECTED_BINS = 6

# One hue, one series (dataviz: sequential blue; text wears text tokens).
BAR = "#3B6FD4"
INK = "#1F2933"
INK_MUTED = "#6B7280"
GRID = "#E5E7EB"


def extract_table(text: str, key: str = KEY) -> list[dict]:
    """The LAST ``"<key>": [...]`` array in ``text``, decoded with json."""
    marker = f'"{key}"'
    pos = text.rfind(marker)
    if pos < 0:
        raise ValueError(f"{key!r} not found")
    start = text.index("[", pos)
    arr, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(arr, list):
        raise ValueError(f"{key!r} is not a list")
    if len(arr) != EXPECTED_BINS:
        raise ValueError(f"{key!r} has {len(arr)} bins, expected {EXPECTED_BINS}")
    for row in arr:
        for field in ("start_mm_to", "episodes", "success_rate"):
            if field not in row:
                raise ValueError(f"bin row lacks {field!r}: {row}")
    return arr


def bin_label(row: dict) -> str:
    lo, hi = row.get("start_mm_from"), row["start_mm_to"]
    if lo is None:
        return f"< {hi:+.0f}"
    return f"{lo:+.0f} .. {hi:+.0f}"


def draw(table: list[dict], run: str, out: pathlib.Path, window_note: str | None = None,
         title: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [bin_label(r) for r in table]
    # The env's last bin is open at the top; every fixed +30 mm start lands there.
    labels[-1] = f"≥ {table[-1]['start_mm_to']:+.0f}"
    rates = [0.0 if r["success_rate"] is None else 100.0 * r["success_rate"] for r in table]
    counts = [int(r["episodes"]) for r in table]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    x = range(len(table))
    ax.bar(x, rates, width=0.55, color=BAR, zorder=3)
    for i, (rate, n, row) in enumerate(zip(rates, counts, table)):
        txt = "–" if row["success_rate"] is None else f"{rate:.0f} %"
        ax.text(i, rate + 2.0, txt, ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{lab}\nn = {n}" for lab, n in zip(labels, counts)], fontsize=8.5, color=INK)
    # Head room for the value label of a 100 % bar; the ticks stop at 100.
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Erfolgsrate [%]", color=INK, fontsize=9)
    ax.set_xlabel("Starthöhe der Spitze über der Öffnungsebene [mm]", color=INK, fontsize=9)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    if title:
        # A thesis figure carries its title in the \caption; --no-title
        # leaves the frame out so the caption is not said twice.
        text = f"{run}: Erfolgsrate je Starthöhen-Stufe"
        if window_note:
            text += f"  ({window_note})"
        ax.set_title(text, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _self_test() -> int:
    rc = 0

    def check(name: str, ok: bool) -> None:
        nonlocal rc
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        rc |= 0 if ok else 1

    synthetic = [
        {"start_mm_from": None, "start_mm_to": -20.0, "episodes": 300, "success_rate": 0.8},
        {"start_mm_from": -20.0, "start_mm_to": -10.0, "episodes": 310, "success_rate": 0.4},
        {"start_mm_from": -10.0, "start_mm_to": 0.0, "episodes": 320, "success_rate": 0.2},
        {"start_mm_from": 0.0, "start_mm_to": 10.0, "episodes": 330, "success_rate": 0.35},
        {"start_mm_from": 10.0, "start_mm_to": 20.0, "episodes": 340, "success_rate": None},
        {"start_mm_from": 20.0, "start_mm_to": 30.0, "episodes": 400, "success_rate": 0.22},
    ]
    blob = "noise before\n" + json.dumps({"x": 1, KEY: synthetic[:1] * 6}) + "\nlater dump\n" \
        + json.dumps({"other": 2, KEY: synthetic, "z": 3}) + "\ntrailing text"
    got = extract_table(blob)
    check("extractor finds the LAST table inside surrounding text", got == synthetic)
    check("extractor reads a bare demo_metrics.json", extract_table(json.dumps({KEY: synthetic})) == synthetic)
    try:
        extract_table(json.dumps({KEY: synthetic[:5]}))
        check("a five-bin table is refused", False)
    except ValueError:
        check("a five-bin table is refused", True)
    try:
        extract_table("nothing here")
        check("a missing table is refused", False)
    except ValueError:
        check("a missing table is refused", True)
    check("bin labels: open bottom, closed middle", bin_label(synthetic[0]) == "< -20" and bin_label(synthetic[1]) == "-20 .. -10")
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "synthetic.png"
        draw(synthetic, "SYNTHETIC self-test", out, window_note="erfunden")
        check("draw writes a PNG (synthetic, temp dir)", out.is_file() and out.stat().st_size > 1000)
    print("SELF-TEST PASSED" if rc == 0 else "SELF-TEST FAILED")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=pathlib.Path, help="demo_metrics.json or a text file containing the table")
    ap.add_argument("--run", default="", help="run label for the title, e.g. RT-138")
    ap.add_argument("--out", type=pathlib.Path, help="PNG path, e.g. docs/figures/RT-138_success_by_start_height.png")
    ap.add_argument("--window-note", default=None, help="text after the title, e.g. 'Iteration 1499, 2000-Episoden-Fenster'")
    ap.add_argument("--no-title", action="store_true", help="omit the in-figure title (thesis: the caption names it)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.metrics is None or a.out is None:
        ap.error("--metrics and --out are required (or --self-test)")
    table = extract_table(a.metrics.read_text(encoding="utf-8", errors="replace"))
    draw(table, a.run, a.out, a.window_note, title=not a.no_title)
    for r in table:
        rate = "–" if r["success_rate"] is None else f"{100 * r['success_rate']:5.1f} %"
        print(f"  {bin_label(r):>12} mm  n={int(r['episodes']):5d}  {rate}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
