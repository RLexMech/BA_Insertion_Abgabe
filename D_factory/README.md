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

## Evaluation starten und auswerten

Für die Serien müssen die 15 Epoche-100-Modelle der Läufe RT-242s1–5,
RT-243s1–5 und RT-244s1–5 unter `logs/rl_games/ur5e_insertion/` liegen.
Mit `-RunRoot "<Ordner mit den 15 Trainingsläufen>"` lässt sich ein anderer
Speicherort verwenden. Git muss sauber und [eval_plan.md](eval_plan.md)
committet sein. Jeder Aufruf benötigt eine noch unbenutzte RT-Nummer.

```powershell
# Haupttests: alle drei Methoden, je fünf Seeds.
.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-801
if ($LASTEXITCODE -ne 0) { throw "Evaluation abgebrochen" }
.\scripts\rl_games\run_eval.ps1 -Block T-final-F -Rt RT-802
if ($LASTEXITCODE -ne 0) { throw "Evaluation abgebrochen" }
.\scripts\rl_games\run_eval.ps1 -Block T-nominal -Rt RT-803
if ($LASTEXITCODE -ne 0) { throw "Evaluation abgebrochen" }
.\scripts\rl_games\run_eval.ps1 -Block T-nominal-F -Rt RT-804
if ($LASTEXITCODE -ne 0) { throw "Evaluation abgebrochen" }
# OOD: Auto-DR und Fixed-DR.
.\scripts\rl_games\run_eval.ps1 -Block T-ood -Rt RT-805
if ($LASTEXITCODE -ne 0) { throw "Evaluation abgebrochen" }
# Verdoppelte Beobachtungsfehler: beide Methoden, jeweils pn2, gx2, py2.
$rt = 806
foreach ($block in @('T-final', 'T-final-F')) {
    foreach ($noise in @('pn2', 'gx2', 'py2')) {
        .\scripts\rl_games\run_eval.ps1 -Block $block -Noise $noise -Rt "RT-$rt"
        if ($LASTEXITCODE -ne 0) { throw "Evaluation abgebrochen: RT-$rt" }
        $rt++
    }
}
```

`-DryRun` prüft Tabellen, Git-Stand und Modellpfade ohne Simulation.
Vorhandene Evaluations- und Ergebnisordner werden nicht überschrieben.
Geprüfte Läufe werden nach `results/eval/<Block>/<Methode>_s<Seed>[_<Tag>]/`
kopiert; `-ResultsRoot` ändert dieses Sammelverzeichnis.

Ergänzende Prüfungen: `-Block technik` (Wiederholung), `-Block T-auswahl`
(alle 15 Policies), `-Block T-grenze -Final fixeddr_s4` (Beispiel: die in den
archivierten Daten ausgewählte Policy). Jeweils eine eigene `-Rt` angeben.

Für die Statistik eine separate Python-Umgebung anlegen:

```powershell
python -m venv "..\..\insertion-auswertung-env"
$evalPython = "..\..\insertion-auswertung-env\Scripts\python.exe"
& $evalPython -m pip install -r scripts/eval/requirements.txt
python scripts/eval_collect.py --results results/eval
if ($LASTEXITCODE -ne 0) { throw "Evaluationsdaten nicht konsistent" }
& $evalPython scripts/eval_stats.py --results results/eval
```

Erzeugt `episodes_all.csv`, `collect_report.json`, `summary.json` und
`summary_tables.csv`. Kraft und Durchdringung werden über alle Versuche
ausgegeben, die Dauer über erfolgreiche Versuche. Die zusätzliche
Einzelpolicy-Auswahl benötigt `T-auswahl` und den anschließenden Grenztest;
ohne diese Blöcke entfällt dieser Teil der Ausgabe.

Archivierte Ergebnisse ohne Simulation erneut auswerten (einmalig kopieren):

```powershell
$ziel = ".\results\archiv_pruefung"
if (Test-Path -LiteralPath $ziel) { throw "Ziel existiert bereits" }
New-Item -ItemType Directory -Path ".\results" -Force | Out-Null
Copy-Item -LiteralPath "..\auswertung\daten\endstudie_rl_games\evaluation" -Destination $ziel -Recurse
python scripts/eval_collect.py --results "$ziel"
if ($LASTEXITCODE -ne 0) { throw "Evaluationsdaten nicht konsistent" }
& $evalPython scripts/eval_stats.py --results "$ziel"
```

Für Beobachtungs-/Kontaktanalysen den obigen Modellaufruf um
`--trace-obs results/trace.json --trace-steps 300` ergänzen und zuvor
`New-Item -ItemType Directory -Path results -Force` ausführen.
`python scripts/trace_first_episode.py results/trace.json`,
`python scripts/trace_outcome_split.py results/trace.json` und
`python scripts/summarize_trace.py results/trace.json` werten die Aufzeichnung aus.

## Grundlegende Fügeprüfungen

- [scripted_insert.py](scripts/scripted_insert.py): skriptgesteuerte Fügebewegung.
- [seat_probe.py](scripts/seat_probe.py): Kontaktprüfung in vorgegebenen Einstecktiefen.
- [check_seated_success.py](scripts/check_seated_success.py): Erfolgs- und Tiefenprüfung.
- [check_fixture_mesh.py](scripts/check_fixture_mesh.py) und [check_insertion_sdf.py](scripts/check_insertion_sdf.py): Mesh- und Abstandsprüfung.

Die verfügbaren Testoptionen zeigt jeweils `python scripts/<Dateiname>.py --help`.
