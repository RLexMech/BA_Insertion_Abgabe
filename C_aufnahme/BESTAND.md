# Bestand Experiment C, 18.10.2026

Quelle: `C_Bestand_20261018_222256_056264.zip` vom Trainings-PC,
Originalprojekt `Phase5_Robustheit_v1`. Alle übertragenen Dateien wurden gegen
die im ZIP enthaltenen SHA256 geprüft. `BESTAND.json` dokumentiert die
übernommenen Dateien, Quellen, Prüfsummen und gemeldeten Paketversionen.

- `models/`: 15 unveränderte RSL-RL-Endmodelle `model_1499.pt`, jeweils mit
  `params/env.yaml`, `params/agent.yaml` und `autodr_1499.json`.
  Die Modellprüfsummen stimmen mit den 41 archivierten Evaluationsmanifesten
  überein; `models/manifest.json` ordnet Methode und Seed zu.
- `training_logs/<RT>/`: vollständiger Konsolenlauf mit Iterationen 0–1499 und
  Exitcode 0 sowie `scalars.csv`, `episodes.csv`, `episodes_meta.json` und
  `demo_metrics.json` aus dem jeweiligen Lauf.
- `source/insertion/insertion/tasks/direct/insertion/assets/`: sieben aktive
  USD-/OBJ-/Begleitdateien, bytegleich zu den entsprechenden D-Assets.
- Historische Evaluationsdaten und frühere C-Versuche bleiben unter
  `../auswertung/daten/experiment_C_rsl_rl/`.

RT-206s2, s3 und s4 lagen als angehängte Konsolenlogs mit mehreren Seeds vor.
Übernommen wurde je der vollständige Abschnitt vom Laufkopf bis zum Exitcode
des passenden Seeds. Bei s4 heißt die Quelldatei `RT-206s2.txt`; ihr letzter
Laufkopf benennt Seed 4. Die Originalbytes der Abschnitte bleiben unverändert;
Quellhash und Bytebereich stehen in `BESTAND.json`. Bereits vorhandene kurze
Logauszüge im Auswertungsarchiv bleiben unverändert.

Der Abgabe-Code wurde mit dem übertragenen Original verglichen. Die
Simulationslogik stimmt überein; die bereits gesetzten Abgabe-Standardwerte
(-30 mm Startboden, drei Stall-Puffer, 256 Umgebungen, Seed 1) entsprechen den
archivierten Trainingsaufrufen. Kommentare und Meldungstexte sind gekürzt.
Hinzu kommen Paketierung, Dokumentation und Aufrufhilfen. Keine Gewichte wurden
konvertiert. Die RSL-RL-Checkpoints enthalten keine NumPy-Pickle-Verweise;
eine D-spezifische View-Konvertierung ist deshalb nicht vorgesehen.

Die 41 archivierten Auswertungen wurden auf einer Arbeitskopie mit dem
C-Collector geprüft: 41.050 Erstepisoden, `VERDICT OK`. Dateien, Python-Syntax,
GUI-Tabellenauswahl und die 41 aufgelösten Evaluationsbefehle wurden lokal
geprüft. Isaac Sim und PyTorch sind auf diesem Prüf-PC nicht installiert;
eine neue GPU-/GUI-Wiedergabe und neue Evaluationen sind hier nicht erfolgt.

Auf dem Trainings-PC gemeldet: Isaac Sim 5.1.0.0, Isaac Lab 2.3.2.post1,
RSL-RL 3.0.1, PyTorch 2.7.0+cu118, NumPy 2.4.6. Dies ist eine Bestandsaufnahme,
keine neu erzeugte vollständige Environment-Sperrdatei.
