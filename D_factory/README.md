# Experiment D

Die Befehle in PowerShell im Ordner `D_factory` mit aktivem
`env_isaaclab` und installiertem Paket `source/insertion` ausführen.
Die Trainingsbefehle entsprechen den Endstand-Läufen RT-242/243/244:
256 Umgebungen, 100 Epochen, hier jeweils Seed 1. Für die fünf Seeds
`--seed` und die Endziffer in `RT_NAME` gemeinsam auf 1 bis 5 setzen.

## Auto-DR: 100 Epochen

```powershell
$env:RT_NAME = "RT-242s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

## Fixed-DR: 100 Epochen

```powershell
$env:RT_NAME = "RT-243s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

## No-DR: 100 Epochen

```powershell
$env:RT_NAME = "RT-244s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Neue Läufe speichern Checkpoints unter `logs/rl_games/ur5e_insertion/<Lauf>/nn/`,
Trainingskennzahlen als `scalars.csv` sowie Konfigurationen unter `params/`.

## Auto-DR-Modell aus Epoche 100 aufrufen

`<AutoDR-Laufordner>` durch den gewünschten Laufordner ersetzen.
Der Befehl wählt dessen Checkpoint aus Epoche 100 und lädt den gespeicherten
Auto-DR-Zustand. Die Wiedergabe endet nach 300 Schritten.

```powershell
$lauf = ".\logs\rl_games\ur5e_insertion\<AutoDR-Laufordner>"
$modelle = @(Get-ChildItem -LiteralPath "$lauf\nn" -File | Where-Object { $_.Name -match '^last_ur5e_insertion_ep_100_rew_[-+0-9.eE]+\.pth$' })
if ($modelle.Count -ne 1) { throw "Es muss genau ein Checkpoint aus Epoche 100 vorhanden sein." }
$modell = $modelle[0].FullName
python scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --kit_args=--/app/vulkan=false --num_envs 4 --seed 1 --max-steps 300 --autodr-state --checkpoint "$modell" env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Die historischen `.pth`-Modelle sind nicht im Git-Repository enthalten.
Für dieses Beispiel einen eigenen 100-Epochen-Lauf oder einen separat
bereitgestellten Auto-DR-Checkpoint verwenden.

## Vorhandene Logs und Ergebnisse

Die archivierten Daten liegen zentral im Abgabeordner:

- [Trainingsdaten](../auswertung/daten/endstudie_rl_games/trainingslaeufe/):
  RT-242s (Auto-DR), RT-243s (Fixed-DR), RT-244s (No-DR), jeweils Seeds 1–5
  mit `scalars.csv`, Konfigurationen und Metriken.
- [Evaluationsergebnisse und Logs](../auswertung/daten/endstudie_rl_games/evaluation/):
  Episodentabellen, Manifeste, Prüfsummen und Konsolenlogs.

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
