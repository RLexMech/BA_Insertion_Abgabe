# Experiment D

Alle Befehle sind PowerShell (Anaconda PowerShell) auf dem Trainings-PC.
Jeder Abschnitt setzt den Block **0** voraus. Ein Fenster beendet sich nach
`--max-steps` selbst; vorher abbrechen mit Strg + C.

## 0. Umgebung und Ordner (einmal pro Fenster)

```powershell
conda activate env_isaaclab
cd "C:\Users\Simon\Desktop\Alexander_Pett\Abgabe\BA_Insertion_Abgabe\D_factory"
python -m pip install -e source/insertion
git lfs pull
```

Auf einem anderen Rechner den Pfad im `cd`-Befehl anpassen. Bei einer
ZIP-Abgabe mit vollständigen Modelldateien entfällt `git lfs pull`.

Regeln für jede Wiedergabe mit Fenster:

- `--kit_args=--/app/vulkan=false` immer mitgeben. Mit Vulkan stürzt auf dem
  Trainings-PC jedes Isaac-Fenster beim Start ab (Treiber 610.88,
  `simulation_app.py:602 _prepare_ui`, Zugriffsverletzung).
- Kein `--headless` (sonst kein Fenster) und kein `--real-time` (das Fenster ruckelt).
- Läuft gleichzeitig ein Training: nur eine Umgebung (siehe Hinweise unten).

## 1. Modell wählen und GUI-Kopie anlegen

Die 15 Modelle aus Epoche 100 liegen unter `models/<Trainingslauf>/nn/`:

| Methode | Trainingslauf (`$lauf`) | `$modus` |
|---|---|---|
| Auto-DR | `RT-242s1` bis `RT-242s5` | `autodr` |
| Fixed-DR | `RT-243s1` bis `RT-243s5` | `fixed_dr` |
| No-DR | `RT-244s1` bis `RT-244s5` | `off` |

Nur `$lauf` ändern; `$modus` folgt daraus. Das Beispiel wählt Auto-DR, Seed 2.

```powershell
$lauf = "RT-242s2"
$modus = @{ "RT-242" = "autodr"; "RT-243" = "fixed_dr"; "RT-244" = "off" }[$lauf.Substring(0, 6)]
$original = @(Get-ChildItem ".\models\*__${lauf}_*\nn\last_ur5e_insertion_ep_100_rew_*.pth" -File)
if ($original.Count -ne 1) { throw "Erwartet genau ein Modell fuer $lauf, gefunden: $($original.Count)" }
$modell = ".\models_view\$lauf\nn\ep100_view.pth"
python -B scripts/prepare_view.py checkpoint "$($original[0].FullName)" "$modell"
if ($LASTEXITCODE -ne 0) { throw "View-Vorbereitung fehlgeschlagen" }
```

Warum eine Kopie: Das Isaac-Fenster lädt ein älteres NumPy. Die Originale
enthalten NumPy-2-Objekte und scheitern dort mit `No module named 'numpy._core'`.
[prepare_view.py](scripts/prepare_view.py) wandelt nur diese Objekte in
Tensoren um, vergleicht danach alle Tensoren bytegenau mit dem Original und
lässt das Original unverändert. Eine vorhandene Kopie wird geprüft und weiterverwendet.

## 2. Ansehen wie im Training

Die Wiedergabe zieht jede Episode aus den Grenzen, die der Lauf bis Epoche 100
erreicht hat (`--autodr-state`), mit den Trainingsflags des Laufs.

```powershell
python -u scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --kit_args=--/app/vulkan=false --num_envs 4 --seed 1 --max-steps 1500 --autodr-state --checkpoint "$modell" "env.dr_mode=$modus" env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Erwartet im Fenster: `[rl_games] insertion package imported from: ...\D_factory\source\insertion\insertion`
und `[rl_games-play] AutoDR state loaded (...)`. Eine Episode dauert höchstens 256 Schritte.

## 3. Ansehen unter den Testbedingungen der Arbeit

Die Tests der Arbeit nutzen feste Falltabellen mit 1000 Fällen. Zum Ansehen
schneidet der erste Block die ersten vier Fälle heraus (Umgebung i = Fall i).
Die Originaltabellen bleiben unverändert.

```powershell
python -B scripts/prepare_view.py table scripts/eval/eval_table_seed10_c18.csv results/view_tables/final_r0_n4.csv --first-row 0 --count 4
if ($LASTEXITCODE -ne 0) { throw "Tabellenvorbereitung fehlgeschlagen" }
python -B scripts/prepare_view.py table scripts/eval/eval_table_ood_seed10_c18.csv results/view_tables/ood_r0_n4.csv --first-row 0 --count 4
if ($LASTEXITCODE -ne 0) { throw "Tabellenvorbereitung fehlgeschlagen" }
$test = @("-u", "scripts/rl_games/play.py", "--task", "Ur5e-Insertion-Direct-v0", "--kit_args=--/app/vulkan=false", "--num_envs", "4", "--seed", "100", "--max-steps", "1500", "--checkpoint", "$modell", "env.dr_mode=table", "env.obs_wrench_mode=force_prevact", "env.start_floor_m=0.020")
```

Dann **genau einen** der folgenden Aufrufe ausführen. Die Werte stammen aus
[run_eval.ps1](scripts/rl_games/run_eval.ps1) (Basis Zeile 24–25, Fehler Zeile 29–31).

```powershell
# T-final: Prüfbereich (Zielbreite).
python @test "env.dr_table_path=results/view_tables/final_r0_n4.csv" env.dr_table_width=target env.pocket_obs_yaw_rad=0.0223402

# T-nominal: dieselben Fälle auf die Nennbreite abgebildet.
python @test "env.dr_table_path=results/view_tables/final_r0_n4.csv" env.dr_table_width=nominal env.pocket_obs_yaw_rad=0.0223402

# pn2: Positionsrauschen der Aufnahme verdoppelt (10 statt 5 mm).
python @test "env.dr_table_path=results/view_tables/final_r0_n4.csv" env.dr_table_width=target env.pocket_obs_yaw_rad=0.0223402 env.obs_noise_pocket_pos_std_m=0.01

# gx2: beobachteter Greifversatz verdoppelt (6 statt 3 mm).
python @test "env.dr_table_path=results/view_tables/final_r0_n4.csv" env.dr_table_width=target env.pocket_obs_yaw_rad=0.0223402 env.grasp_obs_offset_x_m=0.006

# py2: beobachteter Gierfehler der Aufnahme verdoppelt (0.0446804 statt 0.0223402 rad).
python @test "env.dr_table_path=results/view_tables/final_r0_n4.csv" env.dr_table_width=target env.pocket_obs_yaw_rad=0.0446804

# T-ood: Fälle außerhalb des Prüfbereichs (die ersten vier: größerer seitlicher Versatz).
python @test "env.dr_table_path=results/view_tables/ood_r0_n4.csv" env.dr_table_width=target env.dr_table_allow_outside=True env.pocket_obs_yaw_rad=0.0223402
```

Andere Fälle: `--first-row` zählt ab 0 ohne Kopfzeile. In der OOD-Tabelle
beginnen die sieben Gruppen zu je 150 Fällen bei 0 (seitlicher Versatz),
150 (Gierwinkel negativ), 300 (Gierwinkel positiv), 450 (Verkippung),
600 (Starthöhe), 750 (Reibung niedrig) und 900 (Reibung hoch). Für jede
Auswahl einen neuen Dateinamen unter `results/view_tables/` wählen.

Das ist eine Ansicht, keine Messung: Eine Umgebung wiederholt ihren Fall nach
jedem Reset, und 1500 Schritte ergeben keine Erfolgsquote. Die Zahlen der
Arbeit kommen aus den [Evaluationsläufen](#evaluationsläufe).

## 4. Training selbst starten

Die Befehle entsprechen den Läufen RT-242/243/244 der Arbeit: 256 Umgebungen,
100 Epochen, ohne Fenster. Ein Lauf dauert etwa **2 h 15 min** (RT-242s1
startete 10:45, RT-242s2 direkt danach 13:01). Für die Seeds 1 bis 5 `--seed`
und die Endziffer in `RT_NAME` gemeinsam ändern. Je Lauf einen Befehl ausführen.

```powershell
# Auto-DR
$env:RT_NAME = "RT-242s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402

# Fixed-DR
$env:RT_NAME = "RT-243s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402

# No-DR
$env:RT_NAME = "RT-244s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Ein neuer Lauf legt `logs/rl_games/ur5e_insertion/<Datum>__<RT_NAME>_.../` an:
Checkpoints unter `nn/` (alle 25 Epochen, Epoche 100 als
`last_ur5e_insertion_ep_100_rew_*.pth`), Kennzahlen in `scalars.csv`,
Konfiguration unter `params/`. Die mitgelieferten Modelle unter `models/`
bleiben unberührt.

Den eigenen Lauf ansehen: statt Block 1 diesen Block ausführen, danach
Abschnitt 2 oder 3 unverändert.

```powershell
$lauf = "RT-242s1"
$modus = @{ "RT-242" = "autodr"; "RT-243" = "fixed_dr"; "RT-244" = "off" }[$lauf.Substring(0, 6)]
$original = @(Get-ChildItem ".\logs\rl_games\ur5e_insertion\*__${lauf}_*\nn\last_ur5e_insertion_ep_100_rew_*.pth" -File | Sort-Object LastWriteTime -Descending)
if ($original.Count -lt 1) { throw "Kein Epoche-100-Checkpoint fuer $lauf unter logs gefunden" }
$modell = ".\models_view\eigen_$lauf\nn\ep100_view.pth"
python -B scripts/prepare_view.py checkpoint "$($original[0].FullName)" "$modell"
if ($LASTEXITCODE -ne 0) { throw "View-Vorbereitung fehlgeschlagen" }
```

Bei mehreren eigenen Läufen mit demselben `RT_NAME` nimmt der Block den
neuesten. Für einen zweiten eigenen Lauf denselben Namen nur verwenden,
wenn die vorige Kopie unter `models_view\eigen_<Lauf>\` vorher gelöscht wird;
sonst lehnt `prepare_view.py` die abweichende Datei ab.

## Hinweise

- **Während ein Training läuft:** in Abschnitt 2 `--num_envs 1` setzen; in
  Abschnitt 3 eine Tabelle mit `--count 1` schneiden und `"--num_envs", "1"` in `$test`.
- Isaac legt bei der Wiedergabe einen Ordner `replay_h...` neben der Kopie an.
  `models_view/` und `results/view_tables/` sind lokale Hilfsdateien.
- **Prüfstand (18.10.2026):** Abschnitt 2 lief auf dem Trainings-PC mit
  RT-242s2 aus einer gleichwertigen numpy-freien Kopie (mit Fenster) und ohne
  Fenster bis `success_rate 1.0`. Noch ungeprüft auf dem Trainings-PC:
  `prepare_view.py`, Abschnitt 2 für Fixed-DR und No-DR, Abschnitt 3 und
  das Ansehen eines eigenen Laufs.

## Evaluationsläufe

Diese Aufrufe führen die vollständigen Tests **ohne Fenster** mit den
Originalmodellen unter `models/` aus. Einzeln ausführen; erforderlich sind
ein sauberer Git-Stand und noch unbenutzte RT-Nummern. Ergebnisse:
`results/eval/<Block>/<Methode>_s<Seed>/` (bei Zusatztests mit entsprechendem Suffix).

```powershell
# Prüfbereich: Auto-DR / No-DR und Fixed-DR.
.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-801
.\scripts\rl_games\run_eval.ps1 -Block T-final-F -Rt RT-802

# Nennbedingungen: Auto-DR / No-DR und Fixed-DR.
.\scripts\rl_games\run_eval.ps1 -Block T-nominal -Rt RT-803
.\scripts\rl_games\run_eval.ps1 -Block T-nominal-F -Rt RT-804

# OOD: Auto-DR und Fixed-DR.
.\scripts\rl_games\run_eval.ps1 -Block T-ood -Rt RT-805

# Verdoppelte Beobachtungsfehler: Positionsrauschen, Greifversatz, Gierfehler.
.\scripts\rl_games\run_eval.ps1 -Block T-final -Noise pn2 -Rt RT-806
.\scripts\rl_games\run_eval.ps1 -Block T-final -Noise gx2 -Rt RT-807
.\scripts\rl_games\run_eval.ps1 -Block T-final -Noise py2 -Rt RT-808
.\scripts\rl_games\run_eval.ps1 -Block T-final-F -Noise pn2 -Rt RT-809
.\scripts\rl_games\run_eval.ps1 -Block T-final-F -Noise gx2 -Rt RT-810
.\scripts\rl_games\run_eval.ps1 -Block T-final-F -Noise py2 -Rt RT-811

# Auswahl, Grenzfälle der ausgewählten Policy und Technikprüfung.
.\scripts\rl_games\run_eval.ps1 -Block T-auswahl -Rt RT-812
.\scripts\rl_games\run_eval.ps1 -Block T-grenze -Final fixeddr_s4 -Rt RT-813
.\scripts\rl_games\run_eval.ps1 -Block technik -Rt RT-814
```
