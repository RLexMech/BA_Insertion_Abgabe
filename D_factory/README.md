# Experiment D

Die Befehle im Ordner `D_factory` in der eingerichteten Isaac-Lab-Umgebung
mit installiertem Paket `source/insertion` ausführen.

## Auto-DR starten

```powershell
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=autodr
```

## Fixed-DR starten

```powershell
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=fixed_dr
```

## No-DR starten

```powershell
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=off
```

## Modell aufrufen

Beispiel für ein Auto-DR-Modell mit gespeichertem Auto-DR-Zustand.
`C:\Pfad\zum\modell.pth` durch den eigenen Modellpfad ersetzen.

```powershell
python scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --kit_args=--/app/vulkan=false --num_envs 4 --seed 1 --max-steps 300 --autodr-state --checkpoint "C:\Pfad\zum\modell.pth" env.dr_mode=autodr
```

Für Fixed-DR beziehungsweise No-DR `--autodr-state` weglassen und
`env.dr_mode=fixed_dr` beziehungsweise `env.dr_mode=off` verwenden.


## Evaluationstabellen

Alle Tabellen liegen einmalig in [scripts/eval/](scripts/eval/), jeweils als
CSV mit zugehöriger JSON-Metadatendatei.

| Verwendung | Tabelle |
|---|---|
| Nominal / final, einschließlich Fixed-DR | [eval_table_seed10_c18.csv](scripts/eval/eval_table_seed10_c18.csv) |
| Auswahl | [eval_table_seed30_c18.csv](scripts/eval/eval_table_seed30_c18.csv) |
| Technikprüfung | [eval_table_seed20_c18.csv](scripts/eval/eval_table_seed20_c18.csv) |
| Grenzfälle | [eval_table_boundary_seed10_c18.csv](scripts/eval/eval_table_boundary_seed10_c18.csv) |
| OOD | [eval_table_ood_seed10_c18.csv](scripts/eval/eval_table_ood_seed10_c18.csv) |

Ein Modell auf der finalen Tabelle auswerten (Modellpfad ersetzen):

```powershell
python scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 100 --checkpoint "C:\Pfad\zum\modell.pth" --eval-table scripts/eval/eval_table_seed10_c18.csv --eval-width target env.start_floor_m=0.020 env.pocket_obs_yaw_rad=0.0223402
```

Die protokollierte Evaluation benötigt ein Git-Repository mit sauberem
Arbeitsstand und einem Commit einschließlich [eval_plan.md](eval_plan.md).
Diese Datei bleibt als Grundlage des Evaluationsnachweises enthalten.
Der Serienstarter [run_eval.ps1](scripts/rl_games/run_eval.ps1) erwartet die
historischen Laufnamen RT-242/243/244 und die Checkpoints der Epoche 100.
[eval_collect.py](scripts/eval_collect.py) sammelt die Ergebnisse,
[eval_stats.py](scripts/eval_stats.py) berechnet die Statistik
(zusätzlich benötigt: NumPy, pandas, SciPy und statsmodels).

## Grundlegende Fügeprüfungen

- [scripted_insert.py](scripts/scripted_insert.py): skriptgesteuerte Fügebewegung.
- [seat_probe.py](scripts/seat_probe.py): Kontaktprüfung in vorgegebenen Einstecktiefen.
- [check_seated_success.py](scripts/check_seated_success.py): Erfolgs- und Tiefenprüfung.
- [check_fixture_mesh.py](scripts/check_fixture_mesh.py) und [check_insertion_sdf.py](scripts/check_insertion_sdf.py): Mesh- und Abstandsprüfung.

Die verfügbaren Testoptionen zeigt jeweils `python scripts/<Dateiname>.py --help`.
