# Evaluationsplan Experiment C

Wiederholung der 41 archivierten C-Evaluationsläufe. Originaldaten:
`../auswertung/daten/experiment_C_rsl_rl/evaluation/`.
Die Wiederholung erzeugt neue Ergebnisse; die Archivdaten bleiben erhalten.

| Block | Policies | Tabelle | Breite | Fälle je Policy |
|---|---|---|---|---|
| T-auswahl | RT-206s1–s5, RT-207s1–s5 | eval_table_seed30.csv | target | 1000 |
| T-final | RT-206s1–s5, RT-207s1–s5 | eval_table_seed10.csv | target | 1000 |
| T-final-F | RT-231s1–s5 | eval_table_seed10.csv | target | 1000 |
| T-nominal | RT-206s1–s5, RT-207s1–s5 | eval_table_seed10.csv | nominal | 1000 |
| T-nominal-F | RT-231s1–s5 | eval_table_seed10.csv | nominal | 1000 |
| T-grenze | RT-206s1 | eval_table_boundary_seed10.csv | target | 1050 |

Alle Tabellen liegen unter `scripts/eval/` und bleiben bytegleich. Alle Läufe
verwenden `model_1499.pt`, Play-Seed 100, `env.start_floor_m=0.020`,
`env.dr_mode=table` und den C-Beobachtungsmodus `force`. `play.py` setzt die
Anzahl der Umgebungen auf die Zeilenzahl und beendet den Lauf nach dem ersten
abgeschlossenen Versuch jeder Umgebung. Kein `--max-steps` für Messläufe.

`scripts/run_eval.ps1` prüft die Modellprüfsummen und vorhandene Ausgaben,
verlangt einen unveränderten Git-Stand und versieht Wiederholungen mit einem
eigenen RT-Tag. Ergebnisse: `results/eval/<Block>/<Methode>_s<Seed>/`.
Für eine erneute Wiederholung desselben Blocks einen neuen `-Rt` und
`-ResultsRoot results/eval_wiederholung` angeben. `-DryRun` prüft die Auflösung
der Modelle und Befehle ohne Simulation; dabei sind lokale Änderungen erlaubt.

```powershell
python scripts/eval_collect.py --results results/eval
```

Der Collector prüft Tabellen, Bedingungen, Policy-Zuordnung und Manifestdaten
und erzeugt `episodes_all.csv` sowie `collect_report.json`.
Eine GUI-Vorschau mit vier Tabellenzeilen dient nur dem Ansehen; aufgrund der
anderen Parallelisierung ist sie kein Ersatz für die archivierten Messläufe.
