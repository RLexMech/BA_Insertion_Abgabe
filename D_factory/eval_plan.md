# Evaluation – Experiment D

- **Training:** Auto-DR, Fixed-DR und No-DR; jeweils fünf Seeds,
  100 Epochen und 256 Umgebungen.
- **Haupttests:** je 1.000 gemeinsame Testfälle im Prüfbereich und unter
  Nennbedingungen für alle 15 Policies.
- **Zusatztests:** Auto-DR und Fixed-DR außerhalb des Prüfbereichs
  (7 × 150 Fälle) und mit verdoppelten Beobachtungsfehlern (3 × 1.000 Fälle).
- **Kennzahlen:** Fügeerfolgsrate; Kraft und Durchdringung über alle
  Versuche, Dauer für erfolgreiche Versuche.
- **Statistik:** Wilson-Intervalle je Policy, Welch-Intervalle über die
  Seed-Erfolgsraten; ergänzend Bootstrap und Bonferroni-Korrektur.

[Testtabellen](scripts/eval/) ·
[Trainingsdaten, Evaluationslogs und Ergebnisse](../auswertung/daten/endstudie_rl_games/)
