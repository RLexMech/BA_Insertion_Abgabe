# Bachelorarbeit – RL-Fügeaufgabe mit UR5e in Isaac Lab: die 4 Stände

Diese Repo enthält die 4 Stände des Experiments. Jeder Stand ist ein Punkt, an dem die Arbeit in die nächste Stufe übergeht.

| Stufe | Inhalt | Roboter | Ordner | Tag im Original-Repo |
|---|---|---|---|---|
| A | Zylinder Peg-in-Hole | UR10e | [A_zylinder/](A_zylinder/) | `stufe-A-zylinder` in [ur10e-peg-insertion-rl](https://github.com/RLexMech/ur10e-peg-insertion-rl) |
| B | Quader Peg-in-Hole + Generalisierung | UR10e | [B_quader/](B_quader/) | `stufe-B-quader` in [ur10e-peg-insertion-rl](https://github.com/RLexMech/ur10e-peg-insertion-rl) |
| C | echtes Teil in konstruierter Aufnahme, realitätsnahe Szene | UR5e | [C_aufnahme/](C_aufnahme/) | `stufe-C-aufnahme` in [Ur5e_ContactRich_Insertion](https://github.com/RLexMech/Ur5e_ContactRich_Insertion) |
| D | Annäherung an die Factory-Studie, Endkonfiguration (Kap. 4.5) | UR5e | [D_factory/](D_factory/) | `stufe-D-factory` in [Ur5e_ContactRich_Insertion](https://github.com/RLexMech/Ur5e_ContactRich_Insertion) |

Dazu [auswertung/](auswertung/): die Daten der zitierten Läufe und die Skripte der Abbildungen von Kapitel 5.

## Wo was steht

- [CHECKPOINTS.md](CHECKPOINTS.md): wo jede Stufe endet, wo die nächste beginnt und warum.
- [STAENDE.md](STAENDE.md): pro Stufe Regler, Aktions- und Beobachtungsraum, Reward, Streuung, PPO, Ergebnis, dazu der Befehl, um den Stand auf den Trainingsrechner zu holen.
- [auswertung/README.md](auswertung/README.md): welcher Lauf welche Zahl und welche Abbildung liefert, und mit welchem Befehl.

## Was in den Ordnern liegt

Jeder Stufen-Ordner ist eine Kopie des Original-Repos an einem Commit (`git archive`), gekürzt auf:
- Code (`source/`, `scripts/`)
- Entscheidungen (`DECISIONS.md`), Probleme (`PROBLEMS.md`), Übergabe (`HANDOFF*.md`)
- A/B: Ergebnis-Dokumente (`docs/*.md`)
- C/D: Laufprotokolle (`rt_logs/`, nur die Erwartungs- und Urteilsdateien), Plan der Evaluation (`eval_plan.md`), Belege der Streuwerte, CAD der Aufnahme (`CAD/*.stp`)

Bei B sind abweichend vom Original-Tag die drei Streuwerte der Taschenlage direkt auf den dokumentierten Endstand gesetzt; siehe [B_quader/README.md](B_quader/README.md). Bei C sind die gemeinsamen Studienwerte (256 Envs, untere Starthöhengrenze −30 mm, drei Stall-Puffer und Standard-Seed 1) voreingestellt; die drei Trainingsbedingungen und Seeds 1–5 sind in [C_aufnahme/README.md](C_aufnahme/README.md) beschrieben.

Welcher Commit das ist, steht in [STAENDE.md](STAENDE.md) („Ordner hier“). Bei C und D ist das der **Endstand**: der letzte Commit, auf dem alle in der Arbeit genannten Läufe der Stufe laufen (Training und Tests). Einzelne Erwartungsdateien kommen aus einem späteren Commit. Sie stehen in der Karte der Stufe.

`auswertung/` enthält Kopien aus Commits und aus den gitignorierten Laufordnern. Jede Datei steht mit Quelle und sha256 in [auswertung/HERKUNFT.csv](auswertung/HERKUNFT.csv).

Nicht enthalten:
- die Git-Geschichte
- Folien, PDFs, Bilder, Videos, Thesis-Entwürfe
- trainierte Modelle (die sha256 jeder getesteten Policy steht in ihrer `eval_manifest.json` unter `auswertung/daten/…/evaluation/`)
- die USD-Dateien der Szene (sie werden auf dem Trainingsrechner erzeugt und sind in keinem Git-Stand)

Die volle Geschichte liegt in den Original-Repos. Die Tags zeigen dort auf die Stände in [STAENDE.md](STAENDE.md).

## Hinweise zum Lesen

- A und B sind die Vorstudie (D-043 im Repo `Ur5e_ContactRich_Insertion`). C und D sind die eigentliche Aufgabe.
- Viele Streuwerte wurden ursprünglich erst beim Lauf über die Befehlszeile gesetzt (`env.…=…`). Die Befehle stehen in STAENDE.md. Bei B sind die drei Endeinstellungen der Taschenlage in dieser Abgabefassung bereits im Code aktiviert.
- Verweise wie `D-177` zeigen auf Einträge in `DECISIONS.md` des jeweiligen Ordners, `RT-206` auf Dateien in `rt_logs/`.
- Entscheidungen nach dem Endstand D (D-207 bis D-230, zum Beispiel D-225: keine finale Policy) stehen nur im Original-Repo (`DECISIONS.md` auf `ff2aadd8`).
