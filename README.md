# Bachelorarbeit – RL-Fügeaufgabe mit UR5e in Isaac Lab: die 4 Stände

Diese Repo enthält die 4 Stände des Experiments. Jeder Stand ist ein Punkt, an dem die Arbeit in die nächste Stufe übergeht.

| Stufe | Inhalt | Roboter | Ordner | Tag im Original-Repo |
|---|---|---|---|---|
| A | Zylinder Peg-in-Hole | UR10e | [A_zylinder/](A_zylinder/) | `stufe-A-zylinder` in [ur10e-peg-insertion-rl](https://github.com/RLexMech/ur10e-peg-insertion-rl) |
| B | Quader Peg-in-Hole + Generalisierung | UR10e | [B_quader/](B_quader/) | `stufe-B-quader` in [ur10e-peg-insertion-rl](https://github.com/RLexMech/ur10e-peg-insertion-rl) |
| C | echtes Teil in konstruierter Aufnahme, realitätsnahe Szene | UR5e | [C_aufnahme/](C_aufnahme/) | `stufe-C-aufnahme` in [Ur5e_ContactRich_Insertion](https://github.com/RLexMech/Ur5e_ContactRich_Insertion) |
| D | Annäherung an die Factory-Studie | UR5e | folgt | `stufe-D-factory` (folgt) |

## Wo was steht

- [CHECKPOINTS.md](CHECKPOINTS.md): wo jede Stufe endet, wo die nächste beginnt und warum.
- [STAENDE.md](STAENDE.md): pro Stufe Regler, Aktions- und Beobachtungsraum, Reward, Streuung, PPO, Ergebnis, dazu der Befehl, um den Stand auf den Trainingsrechner zu holen.

## Was in den Ordnern liegt

Jeder Ordner ist eine Kopie des Original-Repos am jeweiligen Tag (`git archive`), gekürzt auf:
- Code (`source/`, `scripts/`)
- Entscheidungen (`DECISIONS.md`), Probleme (`PROBLEMS.md`), Übergabe (`HANDOFF*.md`)
- A/B: Ergebnis-Dokumente (`docs/*.md`)
- C: Laufprotokolle (`rt_logs/`), Belege der Streuwerte, CAD der Aufnahme (`CAD/*.stp`)

Nicht enthalten:
- die Git-Geschichte
- Folien, PDFs, Bilder, Thesis-Entwürfe
- trainierte Modelle
- die USD-Dateien der Szene (sie werden auf dem Trainingsrechner erzeugt und sind in keinem Git-Stand)

Die volle Geschichte liegt in den Original-Repos. Die Tags zeigen dort auf genau diese Stände.

## Hinweise zum Lesen

- A und B sind die Vorstudie (D-043 im Repo `Ur5e_ContactRich_Insertion`). C und D sind die eigentliche Aufgabe.
- Viele Streuwerte sind im Code `0` oder `off`. Sie wurden erst beim Lauf über die Befehlszeile gesetzt (`env.…=…`). Die Befehle stehen in STAENDE.md.
- Verweise wie `D-177` zeigen auf Einträge in `DECISIONS.md` des jeweiligen Ordners, `RT-206` auf Dateien in `rt_logs/`.
