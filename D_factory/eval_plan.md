# Evaluation – Experiment D

Kurzüberblick zur durchgeführten Endstudie. Maßgeblich sind Kapitel 4.3
und die Ergebnisse in Kapitel 5 der finalen Bachelorarbeit.

- **Vergleich:** Auto-DR, Fixed-DR und No-DR mit jeweils fünf Trainings-Seeds,
  100 Epochen und 256 Umgebungen; Auswertung eingefrorener Policies.
- **Haupttests:** je 1.000 gemeinsame Testfälle im Prüfbereich und unter
  Nennbedingungen für alle 15 Policies.
- **Zusatztests:** Auto-DR und Fixed-DR außerhalb des Prüfbereichs
  (7 × 150 Fälle) sowie mit verdoppelten Beobachtungsfehlern (3 × 1.000 Fälle).
- **Kennzahlen:** geometrischer Fügeerfolg; Kraft und Durchdringung über alle
  Versuche, Dauer für erfolgreiche Versuche.
- **Statistik:** Wilson-Intervalle je Policy und Welch-Intervalle über die
  Seed-Erfolgsraten; Bootstrap und Bonferroni-Korrektur ergänzend.

Testtabellen: [scripts/eval/](scripts/eval/).
Archivierte Trainingsdaten, Evaluationslogs und Ergebnisse:
[auswertung/daten/endstudie_rl_games/](../auswertung/daten/endstudie_rl_games/).
Startbefehle: [README.md](README.md).

Die Aussagen beziehen sich auf die Simulation; ein Nachweis an realer
Hardware ist damit nicht verbunden.
