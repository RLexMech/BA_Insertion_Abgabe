r"""Table to figure H2: Auto-DR's success in the main test per
value range of the five disturbances. Auto-DR left the figure:
at 100 % its values could not be read there. Same numbers H2 drew: the seed mean,
its 95 % percentile bootstrap interval, the lowest and highest seed.

Generate the CSV first, then the Word table (requires python-docx):

  python scripts/tools/table_h2_autodr.py --csv
      reads daten/endstudie_rl_games/evaluation/summary.json through eval_figures.data_f3 (the
      data function H2 draws, never recomputed here); writes
      docs/tabellen/Tab_H2_autodr_stoergroessen.csv
  python scripts/tools/table_h2_autodr.py
      (python-docx) turns it into
      docs/tabellen/Tab_H2_autodr_stoergroessen.docx


"""
import argparse
import csv
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "tabellen" / "Tab_H2_autodr_stoergroessen"
ARM = "autodr"


def de(x: float, fmt: str = ".1f") -> str:
    return format(x, fmt).replace(".", ",")


def write_csv(results: pathlib.Path) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    import eval_figures as EF
    d = EF.data_f3(json.loads((results / "summary.json").read_text(encoding="utf-8")))
    rows = []
    for col, label, _fac in EF.F3_PANELS:
        if col not in d:
            continue
        p = d[col]
        a = p["methods"][ARM]
        seeds = list(a["seeds"].values())
        for i, (lo_b, hi_b) in enumerate(zip(p["lower"], p["upper"])):
            vals = [s[i] for s in seeds]
            lo, hi = a["ci"][0][i], a["ci"][1][i]
            if not math.isfinite(lo) and max(vals) == min(vals):
                lo = hi = vals[0]   # every seed equal: the interval is that value (_boot_ci's docstring)
            rows.append([label.replace("\n", " "), f"{lo_b:g}", f"{hi_b:g}", p["n"][i], len(seeds),
                         f"{100 * a['mean'][i]:.2f}", f"{100 * lo:.2f}", f"{100 * hi:.2f}",
                         f"{100 * min(vals):.2f}", f"{100 * max(vals):.2f}"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Störgröße", "Bereich von", "Bereich bis", "Testfälle je Policy", "Seeds",
                    "Mittel in %", "KI unten in %", "KI oben in %", "Seed min in %", "Seed max in %"])
        w.writerows(rows)
    print(f"[table_h2_autodr] {len(rows)} rows from {results / 'summary.json'}; wrote {OUT.with_suffix('.csv')}")


def write_docx() -> None:
    from docx import Document
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from table_h6_training import _table
    with open(OUT.with_suffix(".csv"), encoding="utf-8") as f:
        src = list(csv.reader(f))
    seeds = {r[4] for r in src[1:]}
    if len(seeds) != 1:
        raise SystemExit(f"[table_h2_autodr] rows with different seed counts {seeds}")
    rows = [["Störgröße", "Wertebereich", "Testfälle je Policy", "Mittelwert in %", "95-%-KI in %",
             "Seeds von – bis in %"]]
    last = None
    for r in src[1:]:
        lo_b, hi_b = (de(float(v), "g") for v in r[1:3])
        rows.append(["" if r[0] == last else r[0], f"{lo_b} – {hi_b}", r[3], de(float(r[5])),
                     f"{de(float(r[6]))} – {de(float(r[7]))}", f"{de(float(r[8]))} – {de(float(r[9]))}"])
        last = r[0]
    doc = Document()
    doc.add_paragraph(f"Auto-DR im Haupttest: Erfolgsrate je Wertebereich der fünf Störgrößen, Mittelwert über "
                      f"{seeds.pop()} Seeds mit 95-%-KI (Bootstrap) und Spanne der Seeds.")
    _table(doc, rows, 9)
    doc.save(OUT.with_suffix(".docx"))
    print(f"[table_h2_autodr] {len(rows) - 1} rows; wrote {OUT.with_suffix('.docx')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", action="store_true", help="step 1: summary.json -> CSV")
    ap.add_argument("--results", default=str(REPO / "daten" / "endstudie_rl_games" / "evaluation"))
    a = ap.parse_args()
    if a.csv:
        write_csv(pathlib.Path(a.results))
    else:
        write_docx()
    return 0


if __name__ == "__main__":
    sys.exit(main())
