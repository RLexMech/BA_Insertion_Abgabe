# Evaluation Experiment C

Umfang: Haupttest und Test ohne Streuung des C-Endstands für Auto-DR, No-DR
und Fixed-DR, entsprechend Abbildung B-1 in Anhang B.1 der finalen Bachelorarbeit
(gedruckte Seite 105). Vier Blöcke, insgesamt 30 Modellläufe und 30.000 Testfälle.

| Block | Modelle | Breite | Fälle je Modell |
|---|---|---|---|
| T-final | RT-206s1–s5, RT-207s1–s5 | target | 1000 |
| T-final-F | RT-231s1–s5 | target | 1000 |
| T-nominal | RT-206s1–s5, RT-207s1–s5 | nominal | 1000 |
| T-nominal-F | RT-231s1–s5 | nominal | 1000 |

Alle Läufe verwenden die unveränderte `scripts/eval/eval_table_seed10.csv`,
`model_1499.pt`, Play-Seed 100, `env.start_floor_m=0.020`, `env.dr_mode=table`
und den C-Beobachtungsmodus `force`. `play.py` setzt 1000 Umgebungen und beendet
nach dem ersten abgeschlossenen Versuch jeder Umgebung. Kein `--max-steps`.

`scripts/run_eval.ps1` prüft Modellprüfsummen und vorhandene Ausgaben und
verlangt einen unveränderten Git-Stand. Ergebnisse:
`results/eval/<Block>/<Methode>_s<Seed>/`. Wiederholungen benötigen einen neuen
`-Rt` und einen neuen `-ResultsRoot`. `-DryRun` prüft die Aufrufe ohne Simulation.

```powershell
python -B scripts/eval_collect.py --results results/eval
```

Der Collector prüft die vorhandenen Läufe dieser vier Blöcke und zeigt pro
Block und Methode Modellzahl, Testfallzahl und Erfolgsrate. Er schreibt
`summary_tables.csv` (je Modell), `episodes_all.csv` (Erstepisoden) und
`collect_report.json` (Prüfbefunde). Ein Haupttest allein kann separat geprüft
werden; das ist keine Bestätigung, dass auch der Test ohne Streuung gelaufen ist.

| Methode | Haupttest | Test ohne Streuung |
|---|---:|---:|
| Auto-DR | 97,02 % | 97,80 % |
| No-DR | 26,24 % | 99,46 % |
| Fixed-DR | 57,38 % | 59,06 % |

Referenz sind die vorhandenen Daten unter
`../auswertung/daten/experiment_C_rsl_rl/evaluation/`. Die GUI-Vorschau mit
vier Fällen ersetzt keinen Messlauf. `eval_table_seed20.csv` dient ausschließlich
dem internen Selbsttest von `check_eval_conditions.py`, nicht einem weiteren
Evaluationsblock.
