# Auswertung

`daten/experiment_C_rsl_rl/` enthält die Trainings- und Evaluationsdaten von Experiment C, `daten/endstudie_rl_games/` die von Experiment D. Die Auswertungsskripte liegen unter `scripts/`.

Benötigt werden Python, die folgenden Pakete und die Schriftart Arial. Im Hauptordner des Repositorys beginnen:

```powershell
cd auswertung
python -m pip install numpy pandas scipy statsmodels matplotlib
```

Alle folgenden Befehle im Ordner `auswertung/` ausführen. Die Abbildungen werden unter `Auswertung_Abbildungen/` neben dem Repository gespeichert.

## Experiment C

```powershell
python scripts/eval_figures.py --results daten/experiment_C_rsl_rl/evaluation --training daten/experiment_C_rsl_rl/training_auszug --out ../../Auswertung_Abbildungen/C --only H1,F7
```

## Experiment D

```powershell
python scripts/eval_figures.py --results daten/endstudie_rl_games/evaluation --training daten/endstudie_rl_games/training_auszug --out ../../Auswertung_Abbildungen/D --only H1,H2,H6,H18,H19,H20,H22
```

## Vergleich und ergänzende Abbildungen

```powershell
python scripts/plot_rslrl_vs_rlgames.py --pair H16 --out ../../Auswertung_Abbildungen/Vergleich
python scripts/plot_mlp_vs_lstm.py --out ../../Auswertung_Abbildungen/D
python scripts/plot_fortsetzung_fixeddr.py --out ../../Auswertung_Abbildungen/D
```

Weitere Optionen zeigt das jeweilige Skript mit `--help`. Für die Word-Tabellen unter `scripts/tools/table_*.py` wird zusätzlich `python-docx` benötigt.
