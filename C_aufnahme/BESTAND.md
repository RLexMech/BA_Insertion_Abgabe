# Bestand Experiment C

- `models/`: 15 RSL-RL-Endmodelle `model_1499.pt` mit Konfigurationen und AutoDR-Zustand. `models/manifest.json` ordnet Methode und Seed zu.
- `training_logs/<RT>/`: Konsolenlogs, Lernkurven, Episodendaten und Trainingskennzahlen.
- `source/insertion/insertion/tasks/direct/insertion/assets/`: USD-/OBJ-Dateien und Asset-Metadaten.
- `../auswertung/daten/experiment_C_rsl_rl/`: historische Evaluationsdaten und weitere C-Versuche.
- `BESTAND.json`: Dateizuordnung, Prüfsummen und Paketversionen; für RT-206s2, RT-206s3 und RT-206s4 auch die Abschnitte der zusammengefassten Quelllogs.

Trainingsumgebung: Isaac Sim 5.1.0.0, Isaac Lab 2.3.2.post1, RSL-RL 3.0.1, PyTorch 2.7.0+cu118 und NumPy 2.4.6.

Aufrufe für Training, Wiedergabe und Evaluation stehen in [README.md](README.md). Die Evaluationsbedingungen beschreibt [eval_plan.md](eval_plan.md).
