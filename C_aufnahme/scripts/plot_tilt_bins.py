# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Success rate over FIXTURE TILT, one bar per bin, episodes per bin printed.

WHAT IT DRAWS
-------------
The ``success_by_tilt_bin`` table of one ``demo_metrics.json`` (D-037,
``insertion_env.py`` ``_success_by_tilt_bin``): five bins of 2 deg over the
sampled fixture tilt magnitude (edges 2/4/6/8/10 deg, the last bin open at
the top), the trailing-window success rate of each as a bar, the number of
episodes that fell in the bin under the bar. Height only vs tilt only: this
is the TILT twin of ``plot_start_height_bins.py`` and draws exactly that
table -- no height, no overlay -- because the mean success rate cannot
show whether the LARGE angles were learned or only the small ones, and the
reader must see how many episodes each percentage stands on.

THE EDGES ARE THE ENV'S, AND THEY MOVED (2026-09-12, user). They were
3/6/9/12/15 deg, D-037's superseded ladder; the last edge is now the AutoDR
tilt ceiling of 10 deg (``autodr.DR_DIMS``), the rule D-178 (7) states and
``scripts/check_autodr.py`` holds. THE BIN COUNT DID NOT MOVE -- five then,
five now -- so every figure this script has already drawn keeps its own
numbers and its own caption. A figure drawn from a run BEFORE that date
carries the old edges and must not be relabelled.

A run with ``fixture_tilt_noise_rad`` = 0.0873 (5 deg) fills the 0-2, 2-4
and 4-6 deg bins ONLY; the two upper bins read ``episodes 0, rate None`` and
are drawn as an empty slot with a dash, which is correct, not a defect.
(That field is the STATIC tilt route; under ``dr_mode='autodr'`` the env
refuses it non-zero and the `tilt` boundary draws the angle instead.)

INPUT
-----
``--metrics FILE`` is either a ``demo_metrics.json`` itself or any text file
that CONTAINS the table (an ``rt_logs`` dump, ``rt_logs/inbox.txt``). The
table is located by its key and decoded with ``json``; if the file holds it
more than once the LAST occurrence is used (a run's dump ends with its final
window). A table with a bin count other than five is refused: the bins are
the env's, not this script's, and a different count means a different env.

``--out`` names the PNG; the PDF sibling (same stem, ``.pdf``) is written
beside it, the thesis copy (same convention as ``plot_run_curves.py``).

Offline, no Isaac: ``json`` + ``matplotlib``.

    python scripts/plot_tilt_bins.py --metrics docs/figures/RT-172_demo_metrics.json --run RT-172 \
        --window-note "Iteration 2000, 2000-Episoden-Fenster" \
        --out docs/figures/RT-172_success_by_tilt.png
    python scripts/plot_tilt_bins.py --self-test

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

KEY = "success_by_tilt_bin"
EXPECTED_BINS = 5

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
        for field in ("tilt_deg_from", "tilt_deg_to", "episodes", "success_rate"):
            if field not in row:
                raise ValueError(f"bin row lacks {field!r}: {row}")
    return arr


def bin_label(row: dict) -> str:
    return f"{row['tilt_deg_from']:.0f} .. {row['tilt_deg_to']:.0f}"


def draw(table: list[dict], run: str, out: pathlib.Path, window_note: str | None = None,
         title: bool = True) -> list[pathlib.Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [bin_label(r) for r in table]
    # The env's last bin is open at the top.
    labels[-1] = f"≥ {table[-1]['tilt_deg_from']:.0f}"
    rates = [0.0 if r["success_rate"] is None else 100.0 * r["success_rate"] for r in table]
    counts = [int(r["episodes"]) for r in table]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    x = range(len(table))
    ax.bar(x, rates, width=0.55, color=BAR, zorder=3)
    for i, (rate, row) in enumerate(zip(rates, table)):
        txt = "–" if row["success_rate"] is None else f"{rate:.0f} %"
        ax.text(i, rate + 2.0, txt, ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{lab}\nn = {n}" for lab, n in zip(labels, counts)], fontsize=8.5, color=INK)
    # Head room for the value label of a 100 % bar; the ticks stop at 100.
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Erfolgsrate [%]", color=INK, fontsize=9)
    ax.set_xlabel("Kippung der Aufnahme [°]", color=INK, fontsize=9)
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
        text = f"{run}: Erfolgsrate je Kippungs-Stufe"
        if window_note:
            text += f"  ({window_note})"
        ax.set_title(text, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for p in (out.with_suffix(".png"), out.with_suffix(".pdf")):
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    return written


def _self_test() -> int:
    rc = 0

    def check(name: str, ok: bool) -> None:
        nonlocal rc
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        rc |= 0 if ok else 1

    # The env's CURRENT edges (2/4/6/8/10 deg, `_tilt_bin_edges_deg`). The
    # rates and counts are invented; the bounds are not, so this fixture
    # cannot quietly describe a table shape the env stopped producing.
    synthetic = [
        {"tilt_deg_from": 0.0, "tilt_deg_to": 2.0, "episodes": 1010, "success_rate": 0.95},
        {"tilt_deg_from": 2.0, "tilt_deg_to": 4.0, "episodes": 990, "success_rate": 0.4},
        {"tilt_deg_from": 4.0, "tilt_deg_to": 6.0, "episodes": 0, "success_rate": None},
        {"tilt_deg_from": 6.0, "tilt_deg_to": 8.0, "episodes": 0, "success_rate": None},
        {"tilt_deg_from": 8.0, "tilt_deg_to": 10.0, "episodes": 0, "success_rate": None},
    ]
    blob = "noise before\n" + json.dumps({"x": 1, KEY: synthetic[:1] * 5}) + "\nlater dump\n" \
        + json.dumps({"other": 2, KEY: synthetic, "z": 3}) + "\ntrailing text"
    got = extract_table(blob)
    check("extractor finds the LAST table inside surrounding text", got == synthetic)
    check("extractor reads a bare demo_metrics.json", extract_table(json.dumps({KEY: synthetic})) == synthetic)
    try:
        extract_table(json.dumps({KEY: synthetic[:4]}))
        check("a four-bin table is refused", False)
    except ValueError:
        check("a four-bin table is refused", True)
    try:
        extract_table(json.dumps({KEY: synthetic + synthetic[:1]}))
        check("a six-bin table (the height table's count) is refused", False)
    except ValueError:
        check("a six-bin table (the height table's count) is refused", True)
    try:
        extract_table("nothing here")
        check("a missing table is refused", False)
    except ValueError:
        check("a missing table is refused", True)
    try:
        extract_table(json.dumps({KEY: [{"tilt_deg_from": 0.0, "episodes": 1, "success_rate": 1.0}] * 5}))
        check("a row without tilt_deg_to is refused", False)
    except ValueError:
        check("a row without tilt_deg_to is refused", True)
    check("bin label: closed bins read 'from .. to'", bin_label(synthetic[1]) == "2 .. 4")
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "synthetic.png"
        written = draw(synthetic, "SYNTHETIC self-test", out, window_note="erfunden")
        check("draw writes a PNG (synthetic, temp dir)", out.is_file() and out.stat().st_size > 1000)
        pdf = out.with_suffix(".pdf")
        check("draw writes the PDF sibling", pdf.is_file() and pdf.stat().st_size > 1000 and pdf in written)
    print("SELF-TEST PASSED" if rc == 0 else "SELF-TEST FAILED")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=pathlib.Path, help="demo_metrics.json or a text file containing the table")
    ap.add_argument("--run", default="", help="run label for the title, e.g. RT-172")
    ap.add_argument("--out", type=pathlib.Path, help="PNG path, e.g. docs/figures/RT-172_success_by_tilt.png (PDF sibling is written too)")
    ap.add_argument("--window-note", default=None, help="text after the title, e.g. 'Iteration 2000, 2000-Episoden-Fenster'")
    ap.add_argument("--no-title", action="store_true", help="omit the in-figure title (thesis: the caption names it)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.metrics is None or a.out is None:
        ap.error("--metrics and --out are required (or --self-test)")
    table = extract_table(a.metrics.read_text(encoding="utf-8", errors="replace"))
    written = draw(table, a.run, a.out, a.window_note, title=not a.no_title)
    for r in table:
        rate = "–" if r["success_rate"] is None else f"{100 * r['success_rate']:5.1f} %"
        print(f"  {bin_label(r):>8} deg  n={int(r['episodes']):5d}  {rate}")
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
