r"""Appendix tables to figure H6 (training success over the epoch): the seed mean
with its 95 % bootstrap interval per arm, and per seed the epoch from which the
training success stays at or above 95 % until the last epoch.

Two steps, because no one Python here has both libraries (as table_ood_newcombe.py):

  .venv-analysis\Scripts\python.exe scripts/tools/table_h6_training.py --csv
      reads results/training_rlgames/curves.csv through eval_figures.data_f7
      (the data function H6 draws, never recomputed here); writes
      docs/tabellen/Tab_A_H6_lernkurve.csv        every epoch, mean + interval + seeds
      docs/tabellen/Tab_A_H6_95_prozent.csv       per seed the "stays >= 95 %" epoch
  python scripts/tools/table_h6_training.py
      (system Python, python-docx) turns both CSVs into one
      docs/tabellen/Tab_A_H6_lernkurve.docx; the curve table every STEP epochs.

"Reaches 95 %" is read as "stays at or above 95 % from this epoch to the last":
every seed starts above 95 % in epochs 1-2 (the start phase), so a first crossing
says nothing. Built 2026-10-09.

Third table (thesis 5.2.1, 2026-10-09): per Auto-DR seed the "stays >= 95 %" epoch
from above, the epoch from which the AutoDR friction boundary ``dr/friction_hi``
(curves.csv) stays at its ceiling ``hi_max`` (``autodr_s<N>_autodr_ep100.json``
``dims``), the epochs at that ceiling, and the success in the T-ood blocks
friction_hi / friction_lo (``results/eval_rlgames/episodes_all.csv``). Step 1
writes ``docs/tabellen/Tab_A_reibung_seeds.csv``, step 2 its ``.docx``.
"""
import argparse
import csv
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_CURVE = REPO / "docs" / "tabellen" / "Tab_A_H6_lernkurve"
OUT_95 = REPO / "docs" / "tabellen" / "Tab_A_H6_95_prozent"
OUT_FRIC = REPO / "docs" / "tabellen" / "Tab_A_reibung_seeds"
TRAINING = REPO / "results" / "training_rlgames"
EPISODES = REPO / "results" / "eval_rlgames" / "episodes_all.csv"
THRESHOLD = 0.95
STEP = 5   # docx rows: epoch 1 and every STEP-th epoch


def de(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def write_csv(curves: pathlib.Path) -> None:
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(REPO / "scripts"))
    import eval_figures as EF
    d = EF.data_f7(pd.read_csv(curves))
    arms = list(d)
    its = d[arms[0]]["it"]
    for m in arms:
        if not np.array_equal(d[m]["it"], its):
            raise SystemExit(f"[table_h6_training] {m} logs other epochs than {arms[0]}")
    head = ["Epoche"]
    for m in arms:
        lab = EF.METHOD_LABEL[m]
        head += [f"{lab} Mittel", f"{lab} KI unten", f"{lab} KI oben"] + [f"{lab} Seed {s}" for s in d[m]["seeds"]]
    OUT_CURVE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CURVE.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for i, it in enumerate(its):
            row = [int(it)]
            for m in arms:
                a = d[m]
                lo, hi = a["ci"][0][i], a["ci"][1][i]
                seeds = [v[i] for v in a["seeds"].values()]
                if not np.isfinite(lo) and max(seeds) == min(seeds):
                    lo = hi = seeds[0]   # every seed equal: the interval is that value (_boot_ci's docstring)
                row += [f"{100 * a['mean'][i]:.2f}", f"{100 * lo:.2f}", f"{100 * hi:.2f}"] + [f"{100 * v:.2f}" for v in seeds]
            w.writerow(row)
    with open(OUT_95.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Methode", "Seed", "ab Epoche", "Policy-Schritte in Mio.", "Epoche 100 in %"])
        for m in arms:
            a = d[m]
            for s, v in a["seeds"].items():
                k = len(v)
                while k > 0 and v[k - 1] >= THRESHOLD:
                    k -= 1
                ep = "" if k == len(v) else int(a["it"][k])
                steps = "" if k == len(v) else f"{a['x'][k] / 1e6:.2f}"
                w.writerow([EF.METHOD_LABEL[m], s, ep, steps, f"{100 * v[-1]:.2f}"])
    print(f"[table_h6_training] arms {arms}, epochs {int(its[0])}..{int(its[-1])}; "
          f"wrote {OUT_CURVE.with_suffix('.csv')} and {OUT_95.with_suffix('.csv')}")


def write_friction_csv(curves: pathlib.Path, episodes: pathlib.Path) -> None:
    """Stdlib only. Reads OUT_95's CSV, so it runs after write_csv."""
    import json
    with open(OUT_95.with_suffix(".csv"), encoding="utf-8") as f:
        learned = {r["Seed"]: r["ab Epoche"] for r in csv.DictReader(f) if r["Methode"] == "Auto-DR"}
    fh = {}
    with open(curves, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["method"] == "autodr" and r["dr/friction_hi"]:
                fh.setdefault(r["seed"], {})[int(r["iteration"])] = float(r["dr/friction_hi"])
    ood = {}
    with open(episodes, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["block"] == "T-ood" and r["method"] == "autodr" and r["boundary_key"] in ("friction_hi", "friction_lo"):
                g = ood.setdefault((r["seed"], r["boundary_key"]), [0, 0, set()])
                g[0] += int(float(r["success_train"]))
                g[1] += 1
                g[2].add(round(float(r["friction"]), 2))
    vals = {k: v[2] for k, v in ood.items()}
    if any(len(v) != 1 for v in vals.values()):
        raise SystemExit(f"[table_h6_training] a T-ood friction block holds more than one friction value: {vals}")
    mu = {key: next(iter(vals[(s, key)])) for s, key in vals}
    rows = []
    for s in sorted(fh, key=int):
        st = json.loads((TRAINING / f"autodr_s{s}_autodr_ep100.json").read_text(encoding="utf-8"))
        dims = (st.get("autodr") or st)["dims"]
        top = next(d["hi_max"] for d in dims if d["name"] == "friction")
        curve = fh[s]
        last = max(curve)
        first = next(e for e in sorted(curve) if all(curve[k] >= top - 1e-6 for k in curve if k >= e))
        if curve[first] < top - 1e-6:
            raise SystemExit(f"[table_h6_training] seed {s}: friction_hi never stays at {top}")
        hi, lo = ood[(s, "friction_hi")], ood[(s, "friction_lo")]
        rows.append([s, learned[s], first, last - first + 1, last,
                     hi[0], hi[1], f"{100 * hi[0] / hi[1]:.1f}", lo[0], lo[1], f"{100 * lo[0] / lo[1]:.1f}", top])
    with open(OUT_FRIC.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Seed", "Erfolg >= 95 % ab Epoche", "Reibgrenze am Maximum ab Epoche",
                    "Epochen am Maximum", "Epochen gesamt",
                    f"T-ood Reibwert {mu['friction_hi']} Erfolge", "von", "in %",
                    f"T-ood Reibwert {mu['friction_lo']} Erfolge", "von", "in %", "Maximum der Reibgrenze"])
        w.writerows(rows)
    print(f"[table_h6_training] friction table: {len(rows)} Auto-DR seeds, ceiling {rows[0][-1]}; "
          f"wrote {OUT_FRIC.with_suffix('.csv')}")


def _table(doc, rows, size):
    from docx.shared import Pt
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for c, v in zip(t.add_row().cells, row):
            c.text = v
            for run in c.paragraphs[0].runs:
                run.font.size = Pt(size)
                run.font.bold = i == 0


def write_docx() -> None:
    from docx import Document
    with open(OUT_CURVE.with_suffix(".csv"), encoding="utf-8") as f:
        curve = list(csv.reader(f))
    with open(OUT_95.with_suffix(".csv"), encoding="utf-8") as f:
        top = list(csv.reader(f))
    head = curve[0]
    arms = [h[: -len(" Mittel")] for h in head if h.endswith(" Mittel")]
    rows = [["Epoche"] + [f"{a}: Mittel in % [95-%-KI]" for a in arms]]
    for r in curve[1:]:
        ep = int(r[0])
        if ep != 1 and ep % STEP:
            continue
        cells = [str(ep)]
        for a in arms:
            j = head.index(f"{a} Mittel")
            cells.append(f"{de(float(r[j]))} [{de(float(r[j + 1]))}; {de(float(r[j + 2]))}]")
        rows.append(cells)
    doc = Document()
    doc.add_paragraph(f"Erfolgsrate im Training je Epoche: Mittel über die Seeds und 95-%-Konfidenzintervall "
                      f"(Bootstrap, punktweise); Epoche 1 und jede {STEP}. Epoche.")
    _table(doc, rows, 8)
    doc.add_paragraph("")
    doc.add_paragraph(f"Epoche, ab der die Erfolgsrate im Training bis Epoche 100 bei mindestens "
                      f"{de(100 * THRESHOLD)} % bleibt; leer = nie bis zum Ende.")
    _table(doc, [top[0]] + [[c if c else "nicht erreicht" for c in r[:4]] + [de(float(r[4]))] for r in top[1:]], 9)
    doc.save(OUT_CURVE.with_suffix(".docx"))
    print(f"[table_h6_training] {len(rows) - 1} + {len(top) - 1} rows; wrote {OUT_CURVE.with_suffix('.docx')}")
    with open(OUT_FRIC.with_suffix(".csv"), encoding="utf-8") as f:
        fr = list(csv.reader(f))
    h = fr[0]
    mu_hi = h[5].split()[-2].replace(".", ",")
    mu_lo = h[8].split()[-2].replace(".", ",")
    rows = [["Seed", "Erfolg ≥ 95 % ab Epoche", f"Reibwert {fr[1][-1].replace('.', ',')} ab Epoche",
             "Epochen mit maximalem Reibwert", f"Erfolg bei Reibwert {mu_hi}", f"Erfolg bei Reibwert {mu_lo}"]]
    for r in fr[1:]:
        rows.append([r[0], r[1], r[2], f"{r[3]} von {r[4]}",
                     f"{r[5]} von {r[6]} ({de(float(r[7]))} %)", f"{r[8]} von {r[9]} ({de(float(r[10]))} %)"])
    doc = Document()
    doc.add_paragraph("Auto-DR je Seed: Lerndauer, Zeitpunkt und Dauer des Trainings mit maximalem Reibwert "
                      "und Erfolg außerhalb des Prüfbereichs (T-ood) am oberen und unteren Rand des Reibwerts.")
    _table(doc, rows, 9)
    doc.save(OUT_FRIC.with_suffix(".docx"))
    print(f"[table_h6_training] {len(rows) - 1} rows; wrote {OUT_FRIC.with_suffix('.docx')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", action="store_true", help="step 1 (.venv-analysis): curves.csv -> CSV")
    ap.add_argument("--curves", default=str(REPO / "results" / "training_rlgames" / "curves.csv"))
    a = ap.parse_args()
    if a.csv:
        write_csv(pathlib.Path(a.curves))
        write_friction_csv(pathlib.Path(a.curves), EPISODES)
    else:
        write_docx()
    return 0


if __name__ == "__main__":
    sys.exit(main())
