# Experiment C

Jeden grauen Block **komplett** kopieren und in die Anaconda PowerShell einfügen.
Alle Blöcke im **selben Fenster** ausführen, denn spätere Blöcke nutzen das in Schritt 2 gewählte Modell.

## 1. Starten

Block unverändert einfügen. Den Ordner bei Bedarf an den eigenen Rechner anpassen.

```powershell
conda activate env_isaaclab
cd "C:\Users\Simon\Desktop\Alexander_Pett\Abgabe\BA_Insertion_Abgabe\C_aufnahme"
python -m pip install -e source/insertion
git lfs pull
```

## 2. Modell wählen

Für ein anderes Modell nur `RT-206s1` in der ersten Zeile ändern, dann den ganzen Block einfügen.
Möglich sind `RT-206s1` bis `RT-206s5` (Auto-DR), `RT-231s1` bis `RT-231s5` (Fixed-DR), `RT-207s1` bis `RT-207s5` (No-DR).
Die 15 Originalmodelle liegen unter `models/<Trainingsordner>/model_1499.pt` und werden direkt geladen.

```powershell
$lauf = "RT-206s1"
$modell = (Get-ChildItem ".\models\*__${lauf}_*\model_1499.pt").FullName
```

## 3. Modell ansehen

Block unverändert einfügen. Ein Fenster mit vier Robotern öffnet sich; die Wiedergabe läuft auf der GPU für 300 Schritte mit `dr_mode=off`.

```powershell
python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --kit_args=--/app/vulkan=false --device cuda:0 --num_envs 4 --seed 1 --max-steps 300 --checkpoint "$modell" env.dr_mode=off env.start_floor_m=0.020
```

Zufällige Bedingungen im vollständigen C-Trainingsbereich: `env.dr_mode=off` durch `env.dr_mode=fixed_dr` ersetzen.
Der C-Player lädt den gespeicherten Auto-DR-Zustand nicht automatisch; für die Testbedingungen der Arbeit Schritt 4 verwenden.

## 4. Ansehen im Haupttest der Arbeit

Block unverändert einfügen. Jeder der vier Roboter bekommt einen festen Testfall aus der Arbeit. Das ist eine kurze Vorschau: Sie endet automatisch, sobald alle vier Roboter ihren ersten Versuch beendet haben. Die vollständige Evaluation steht in Schritt 6.

```powershell
python -B scripts/view_table.py scripts/eval/eval_table_seed10.csv results/view_tables/haupttest.csv --count 4
python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --kit_args=--/app/vulkan=false --device cuda:0 --num_envs 4 --seed 100 --checkpoint "$modell" --eval-table results/view_tables/haupttest.csv --eval-width target --eval-tag "gui-$(Get-Date -Format yyyyMMdd-HHmmss)" env.start_floor_m=0.020
```

Test ohne Streuung: im zweiten Befehl `--eval-width nominal` setzen.
Andere vier Fälle: im ersten Befehl z. B. `--first-row 4` ergänzen und in beiden Befehlen einen neuen Ausgabedateinamen statt `haupttest.csv` verwenden.
Eigene Bedingungen: an den Befehl aus Schritt 3 z. B. `--fixture-offset 0.01 0 --fixture-tilt 2 --fixture-yaw 1` anhängen (Versatz in Metern, Winkel in Grad).

## 5. Selbst trainieren (optional)

Block unverändert einfügen. Trainiert Auto-DR mit Seed 1 für 1500 Iterationen.

```powershell
$env:RT_NAME = "RT-206s1"
python scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

Fixed-DR: `RT-231s1` und `env.dr_mode=fixed_dr`. No-DR: `RT-207s1` und `env.dr_mode=off`.
Ansehen: in Schritt 2 `.\models\` durch `.\logs\rsl_rl\ur5e_insertion\` ersetzen und den gewünschten Trainingsordner wählen, falls mehrere Läufe denselben Namen tragen.

## 6. Evaluation der Arbeit wiederholen

Block unverändert einfügen. Das ist der Haupttest von Auto-DR und No-DR (10 Modelle), ohne Fenster.

```powershell
.\scripts\run_eval.ps1 -Block T-final -Rt RT-902
```

Die übrigen C-Blöcke zu Abbildung B-1 (Anhang B.1), ebenfalls direkt kopierbar:

| Test | Aufruf |
|---|---|
| Haupttest Fixed-DR | `.\scripts\run_eval.ps1 -Block T-final-F -Rt RT-903` |
| Test ohne Streuung Auto-DR / No-DR | `.\scripts\run_eval.ps1 -Block T-nominal -Rt RT-904` |
| Test ohne Streuung Fixed-DR | `.\scripts\run_eval.ps1 -Block T-nominal-F -Rt RT-905` |

Ergebnisse: `results/eval/`. Für eine erneute Wiederholung neue `-Rt`-Nummern und `-ResultsRoot results/eval_wiederholung` verwenden.
Die vollständigen Trainingslogs liegen unter `training_logs/<RT>/`.


## 7. Ergebnisse prüfen

Nach den gewünschten Evaluationsläufen diesen Block einfügen. Er prüft die
Testbedingungen und zeigt die Erfolgsrate je Methode und Testblock direkt an.
Für eine Wiederholung mit anderem Ergebnisordner den Pfad entsprechend ändern.

```powershell
python -B scripts/eval_collect.py --results results/eval
```

Erwartete Haupttestwerte der Bachelorarbeit über jeweils fünf Modelle:
Auto-DR **97,02 %**, No-DR **26,24 %**, Fixed-DR **57,38 %**.
Der vollständige Wiederholungslauf vom 19.10.2026 hat diese Werte für jeden Seed bestätigt.
Ausgewertet wird ausschließlich der erste abgeschlossene Versuch jeder Umgebung.
`VERDICT OK` bestätigt die geprüften Testbedingungen; die Erfolgsraten stehen darunter.
Je Modell: `results/eval/summary_tables.csv`; Einzelversuche: `results/eval/episodes_all.csv`;
Prüfprotokoll: `results/eval/collect_report.json`.

Die ursprünglichen Ergebnisse der Arbeit liegen unter
`../auswertung/daten/experiment_C_rsl_rl/evaluation/`.
Die dort vorhandene `summary_tables.csv` enthält die archivierte Auswertung.
Dieses Archiv einschließlich zusätzlicher historischer Versuche bleibt unverändert.
Die hier wiederholbare Evaluation umfasst ausschließlich Haupttest und Test ohne Streuung.

Die verbliebenen `check_*.py` prüfen Mathematik, Auto-DR, Kontaktgeometrie,
Erfolgskriterium und Ergebnisdaten. Sie werden für die obigen Modellaufrufe
nicht zusätzlich benötigt. Frühere Entwicklungs- und Diagnoseversuche sind
nicht Bestandteil dieses Endstands.
