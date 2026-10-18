# Experiment C – RSL-RL

## 1. Starten

In der eingerichteten Isaac-Lab-Umgebung ausführen; den Ordner bei Bedarf anpassen.

```powershell
conda activate env_isaaclab
cd "C:\Users\Simon\Desktop\Alexander_Pett\Abgabe\BA_Insertion_Abgabe\C_aufnahme"
python -m pip install -e source/insertion
git lfs pull
```

## 2. Die 15 Endmodelle

Originalgewichte nach 1500 Iterationen: `models/<Trainingsordner>/model_1499.pt`.
Auto-DR: `RT-206s1` bis `RT-206s5`; No-DR: `RT-207s1` bis `RT-207s5`;
Fixed-DR: `RT-231s1` bis `RT-231s5`. Für ein anderes Modell nur `$lauf` ändern.

```powershell
$lauf = "RT-206s1"
$modell = (Get-ChildItem ".\models\*__${lauf}_*\model_1499.pt").FullName
```

Vier Umgebungen, 300 Schritte, GUI auf der GPU, C-Grundbedingungen (`dr_mode=off`):

```powershell
python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --device cuda:0 --num_envs 4 --seed 1 --max-steps 300 --checkpoint "$modell" env.dr_mode=off env.start_floor_m=0.020
```

## 3. Ansehen unter den Testbedingungen der Arbeit

Vier Fälle des Haupttests. Die Policy bleibt unverändert; `--first-row` wählt
andere Zeilen der Tabelle (0–996).

```powershell
python -B scripts/view_table.py scripts/eval/eval_table_seed10.csv results/view_tables/haupttest.csv --first-row 0 --count 4
python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --device cuda:0 --num_envs 4 --seed 100 --max-steps 300 --checkpoint "$modell" env.dr_mode=table env.dr_table_path=results/view_tables/haupttest.csv env.dr_table_width=target env.start_floor_m=0.020
```

Andere Testfälle: Tabelle und Ausgabedateiname im ersten Befehl sowie
`env.dr_table_path` im zweiten anpassen. Für den Nominaltest nur
`env.dr_table_width=nominal` setzen.

| Test | Tabelle |
|---|---|
| Haupttest / Nominaltest | `scripts/eval/eval_table_seed10.csv` |
| Auswahltest | `scripts/eval/eval_table_seed30.csv` |
| Grenztest | `scripts/eval/eval_table_boundary_seed10.csv` |

Eigene Bedingungen: im Aufruf aus Abschnitt 2 beispielsweise
`--fixture-offset 0.01 0 --fixture-tilt 2 --fixture-yaw 1` ergänzen
(Versatz in Metern, Winkel in Grad). `env.dr_mode=fixed_dr` zieht zufällige
Bedingungen im vollständigen C-Trainingsbereich. `autodr` allein lädt den
gespeicherten AutoDR-Endzustand nicht; die Testtabellen sind für Vergleiche vorgesehen.

## 4. Die Evaluationsläufe wiederholen

Alle Befehle ohne GUI; jeder Block verwendet die zugehörigen Modelle automatisch.

```powershell
.\scripts\run_eval.ps1 -Block T-auswahl -Rt RT-901
.\scripts\run_eval.ps1 -Block T-final -Rt RT-902
.\scripts\run_eval.ps1 -Block T-final-F -Rt RT-903
.\scripts\run_eval.ps1 -Block T-nominal -Rt RT-904
.\scripts\run_eval.ps1 -Block T-nominal-F -Rt RT-905
.\scripts\run_eval.ps1 -Block T-grenze -Rt RT-906
python scripts/eval_collect.py --results results/eval
```

Ergebnisse: `results/eval/`. Erneute Wiederholung: neue RT-Nummern und
`-ResultsRoot results/eval_wiederholung` verwenden. Nur Befehle prüfen: `-DryRun`.
Trainingslogs: `training_logs/<RT>/`; Herkunft und Prüfstand: [BESTAND.md](BESTAND.md).
