# Auswertung – Daten und Skripte der Abbildungen (Kapitel 5 und Anhang)

Dieser Ordner enthält die Daten der zitierten Läufe und die Skripte, aus denen die Abbildungen und Zahlen von Kapitel 5 entstehen.
Grundlage ist die Datenauswahl `Abgabe\Abbildungen\Quelldaten\` vom 2026-10-11. Aufbau und Namen der Ordner sind gleich geblieben.

## Herkunft

Jede Datei steht mit Quelle und sha256 in [HERKUNFT.csv](HERKUNFT.csv). Es gibt 4 Arten:
- `commit`: Die Datei ist byte-gleich mit der Datei im Commit (Repo `Ur5e_ContactRich_Insertion`, Arbeitskopie `p5-rlgames`).
- `gitignoriert`: Die Datei ist in keinem Commit (`rt_logs/` ist gitignoriert). Sie ist byte-gleich kopiert. Die sha256 steht in der Tabelle.
- `zip`: Vollständige Trainings-Konsolenlogs aus den übertragenen ZIP-Archiven. `repo` nennt die ZIP-Datei, `pfad_im_repo` den Eintrag darin. Die Dateien wurden bytegleich übernommen und den vorhandenen Konfigurationen und Messdaten zugeordnet.
- `erzeugt`: Nur `daten/endstudie_rl_games/evaluation/episodes_all.csv`. Die Datei war nie committet. Sie entsteht mit `scripts/eval_collect.py` aus den Testdaten hier und ist byte-gleich mit der Datei in `Quelldaten`.

`.gitattributes` im Wurzelordner setzt `auswertung/** -text`. Git ändert hier also keine Zeilenenden, und die sha256 bleibt auf jedem Rechner gleich.

## Code

- `scripts/` = Kopie aus Commit `ff2aadd8` („Commit the thesis evaluation scripts and their training extracts“): `eval_collect.py`, `eval_stats.py`, alle `plot_*.py`, `tools/plot_*.py`, `tools/table_*.py`.
  Ausnahme: `eval_figures.py` kommt aus `597ff0b7`. Mit dieser Fassung entstehen die Bilder der Arbeit (v14). `ff2aadd8` ändert danach nur die Zahl über dem Balken („M = 99,2 %“, 9 pt).
  Dazu die Dateien, die diese Skripte laden: `check_eval_conditions.py`, `check_autodr.py`, `tools/run_axis.py`, die Testtabellen `scripts/eval/eval_table_*`.
- Python mit `matplotlib`, `numpy`, `pandas`, `scipy`, `statsmodels`; für `tools/table_*.py` zusätzlich `python-docx`. Schrift Arial muss installiert sein (`eval_figures.py` bricht sonst ab).
- Alle Befehle unten in diesem Ordner (`auswertung/`) ausführen. `Q` = `daten`.
- `--out` immer auf einen eigenen Ordner setzen. Sonst schreibt das Skript nach `docs/figures/…` unter diesem Ordner.

## Ordner

```
daten/
  endstudie_rl_games/            Endstudie (Kap. 5.2-5.4), rl_games, 100 Epochen
    evaluation/                  alle Test-Läufe der Endstudie, dazu summary.json und episodes_all.csv
    training_auszug/             aus trainingslaeufe/ abgeleitet (curves.csv, penetration.csv, fortsetzung_RT-253.csv)
    trainingslaeufe/<Lauf>/      je Trainingslauf: scalars.csv, env.yaml, agent.yaml, AutoDR-Zustand, demo_metrics.json, Konsolenlog (*.txt)
    testtabelle/                 Testtabelle des Tests außerhalb des Prüfbereichs (+ Sidecar-JSON)
    spuren/                      RT-271 (Spur des Verklemmens) und D8 (Konsolenlog der Bildfolge Abb. 5-4)
  experiment_C_rsl_rl/           Experiment C (Kap. 5.1.2), rsl_rl, 1500 Iterationen
    evaluation/                  alle Test-Läufe (RT-222..225, RT-232), dazu summary.json
    training_auszug/             aus trainingslaeufe/ abgeleitet (curves.csv)
    trainingslaeufe/<Lauf>/      je Lauf ein Ordner; *.txt = Konsolenlog (liefert 4096 Schritte je Iteration)
    spuren/                      RT-272b (Gier-Spur der C-Policy vor dem Kontakt)
```

## Welcher Lauf ist was

| Lauf (RT) | Bedingung | Seeds | Teil | Ordner |
|---|---|---|---|---|
| RT-138 | PD-Regler im Gelenkraum, Startband −30…+30 mm | 1 | Tab. 5-1 | `experiment_C_rsl_rl/trainingslaeufe/RT-138/` |
| RT-156 | OSC, Startband −30…+30 mm | 1 | Tab. 5-1 | `experiment_C_rsl_rl/trainingslaeufe/RT-156/` |
| RT-195 | mit Rauschen auf der Beobachtung, 99,8 % | 1 | Kap. 5.1.2 | `experiment_C_rsl_rl/trainingslaeufe/RT-195/` |
| RT-189s1 | Auto-DR ohne Startcurriculum, 3000 Iterationen | 1 | Kap. 5.1.2, Gl. 5.1 | `experiment_C_rsl_rl/trainingslaeufe/RT-189s1/` |
| RT-201 | Auto-DR, ein Lauf | – | Abb. 5-3 | `experiment_C_rsl_rl/trainingslaeufe/RT-201/` |
| RT-206s1…s5 | Auto-DR | 1–5 | Experiment C, Abb. 5-5 (s1) | `experiment_C_rsl_rl/trainingslaeufe/` |
| RT-207s1…s5 | No-DR | 1–5 | Experiment C | `experiment_C_rsl_rl/trainingslaeufe/` |
| RT-231s1…s5 | Fixed-DR | 1–5 | Experiment C | `experiment_C_rsl_rl/trainingslaeufe/` |
| RT-222…225, RT-232 | Tests von Experiment C | – | Anhang (Studie C) | `experiment_C_rsl_rl/evaluation/` |
| RT-272b | Gier-Spur der Policy RT-206s1 | – | Kap. 5.1.2 | `experiment_C_rsl_rl/spuren/RT-272b/` |
| RT-242s1…s5 | Auto-DR (mit LSTM) | 1–5 | Endstudie, Abb. 5-5 (s1) | `endstudie_rl_games/trainingslaeufe/RT-242s/` |
| RT-243s1…s5 | Fixed-DR | 1–5 | Endstudie | `endstudie_rl_games/trainingslaeufe/RT-243s/` |
| RT-244s1…s5 | No-DR | 1–5 | Endstudie | `endstudie_rl_games/trainingslaeufe/RT-244s/` |
| RT-247…250 | Haupttest und Test ohne Streuung | – | Kap. 5.3 | `endstudie_rl_games/evaluation/` |
| RT-254…259 | Test außerhalb des Prüfbereichs, erhöhter Beobachtungsfehler | – | Kap. 5.4 | `endstudie_rl_games/evaluation/` |
| RT-253s1, s2, s5 | Fixed-DR, Fortsetzung bis Epoche 250 | 1, 2, 5 | Anhang A | `endstudie_rl_games/trainingslaeufe/RT-253s/` |
| RT-260s1…s5 | Auto-DR ohne LSTM (nur MLP), Ablation | 1–5 | Abb. A-1 | `endstudie_rl_games/trainingslaeufe/RT-260s/` |
| RT-271 | Spur des Verklemmens (Seed 2 und 4, Zeile 906) | – | Kap. 5.4.1 | `endstudie_rl_games/spuren/RT-271/` |
| D8 | Wiedergabe von RT-243s4 für die Bildfolge | – | Abb. 5-4 | `endstudie_rl_games/spuren/D8/` |

**Test-Ordner in `evaluation/`** (Name je Lauf: `<bedingung>_s<seed>[_<tag>]`):

| Ordner | Test in der Arbeit (Kap. 4.3.2) |
|---|---|
| `T-final` (Auto-DR, No-DR), `T-final-F` (Fixed-DR) | Haupttest, 1000 Testfälle |
| `T-nominal`, `T-nominal-F` | Test ohne Streuung |
| `T-ood` | Test außerhalb des Prüfbereichs (je Block 150 Testfälle) |
| `T-final/*_pn2`, `*_gx2`, `*_py2` | Test mit erhöhtem Beobachtungsfehler: Lagefehler der Aufnahme, Greifversatz, Gierfehler der Aufnahme (je ×2) |
| `T-auswahl`, `T-grenze`, `technik` | nicht in den Abbildungen |

## Kette: Rohdaten → Zahlen → Abbildung

1. Test-Läufe → `episodes_all.csv`: `python scripts/eval_collect.py --results <evaluation>`
2. → `summary.json` (Wilson, Bootstrap, Welch, Newcombe): `python scripts/eval_stats.py --results <evaluation>`
   (Schritte 1 und 2 schreiben in den `--results`-Ordner. Am 2026-10-12 geprüft für Schritt 1 auf einer Kopie von `daten/endstudie_rl_games/evaluation`: `episodes_all.csv` und `collect_report.json` byte-gleich.)
3. Trainingsläufe → `training_auszug/curves.csv`:
   - Endstudie: `python scripts/eval_figures.py --extract-training daten/endstudie_rl_games/trainingslaeufe --arm autodr=RT-242s --arm nodr=RT-244s --arm fixeddr=RT-243s --training <ziel>`
   - Experiment C: `python scripts/eval_figures.py --extract-training daten/experiment_C_rsl_rl/trainingslaeufe --training <ziel>`
4. Abbildungen: Tabelle unten.

## Abbildung → Skript → Daten

| Abb. | Befehl (in `auswertung/`) |
|---|---|
| 5-3 | `python scripts/plot_autodr_run.py --scalars daten/experiment_C_rsl_rl/trainingslaeufe/RT-201/RT-201_scalars.csv --metrics …/RT-201_demo_metrics.json --autodr …/RT-201_autodr_833.json --run RT-201 --agent-yaml …/RT-201_agent.yaml --env-yaml …/RT-201_env.yaml --out-dir <ziel> --only stufen` |
| 5-a | `python scripts/eval_figures.py --results daten/experiment_C_rsl_rl/evaluation --training daten/experiment_C_rsl_rl/training_auszug --out <ziel> --only H1` |
| 5-c | wie 5-a, `--only F7` |
| 5-5 | `python scripts/plot_rslrl_vs_rlgames.py --pair H16 --out <ziel>` – liest feste Pfade (siehe unten) |
| 5-6 | `python scripts/eval_figures.py --results daten/endstudie_rl_games/evaluation --training daten/endstudie_rl_games/training_auszug --out <ziel> --only H6` |
| 5-7 | wie 5-6, `--only H1` |
| 5-8 | wie 5-6, `--only H2` |
| 5-9 | wie 5-6, `--only H19` (liest auch `scripts/eval/eval_table_ood_seed10_c18.json`) |
| 5-11 | wie 5-6, `--only H20` |
| A-1 | `python scripts/plot_mlp_vs_lstm.py --rt-logs daten/endstudie_rl_games/trainingslaeufe --out <ziel>` |
| A-2 | wie 5-6, `--only H18` |
| A-3 | wie 5-6, `--only H22` |
| H25 (Fortsetzung Fixed-DR, Anhang A) | `python scripts/plot_fortsetzung_fixeddr.py` – liest feste Pfade (siehe unten) |
| 5-4, Bildfolge C | `scripts/tools/plot_d8_kontakt.py`, `scripts/tools/plot_c_ausrichten.py` – lesen die Videos (`.mkv`). Die Videos sind nicht in dieser Abgabe. |

`eval_figures.py` druckt für 5-9, 5-11, A-2 und A-3 je eine Zeile `caption:` (nicht gezeigte Policies). Sie gehört in die Bildunterschrift.

**Feste Pfade.** `plot_rslrl_vs_rlgames.py` und `plot_fortsetzung_fixeddr.py` haben keinen Daten-Parameter. Sie lesen relativ zu `auswertung/`. Vor dem Lauf diese Kopien anlegen:

| Pfad, den das Skript liest | Kopie von |
|---|---|
| `docs/figures/RT-206s1_scalars.csv` | `daten/experiment_C_rsl_rl/trainingslaeufe/RT-206s1/RT-206s1_scalars.csv` |
| `rt_logs/RT-242s/RT-242s1_scalars.csv` | `daten/endstudie_rl_games/trainingslaeufe/RT-242s/RT-242s1_scalars.csv` |
| `results/training_rlgames/curves.csv` | `daten/endstudie_rl_games/training_auszug/curves.csv` |
| `results/training_rlgames/fortsetzung_RT-253.csv` | `daten/endstudie_rl_games/training_auszug/fortsetzung_RT-253.csv` |

## Geprüft am 2026-10-12

- Jede Datei aus `Quelldaten` hat hier eine Kopie mit gleichem Inhalt (1413 Dateien; 919 davon unterscheiden sich nur in den Zeilenenden, weil `Quelldaten` aus der Arbeitskopie mit CRLF stammt).
- Abbildungen neu gebaut mit den Befehlen oben, sha256 gegen `Abgabe\Abbildungen\Abb_*.png`:
  alle 13 byte-gleich (5-3, 5-5, 5-6, 5-7, 5-8, 5-9, 5-11, 5-a, 5-c, A-1, A-2, A-3, H25).
- `check_kap5_werte.py` (Original-Repo, `scripts/tools/`) gegen die Daten hier: Die Anker gehören zu `Bachelorarbeit_Alex_v5.docx`. Mit v5 entstehen 203 Prüfzeilen, identisch mit dem Lauf im Original-Repo. Mit v14 bricht das Skript beim ersten geänderten Satz ab (Absatz Tab. 5-1).

## Hinweise

- Experiment C hat keine `agent.yaml`. Den Faktor 4096 Policy-Schritte je Iteration liest `eval_figures._log_factor` aus den Zeilen `Total timesteps` im Konsolenlog jedes Laufs.
- `episodes_all.csv` fehlt für Experiment C. Die Abbildungen 5-a und 5-c brauchen sie nicht.
- Die 15 D-Endmodelle werden unter `D_factory/models/` im übergeordneten Repository über Git LFS verwaltet. Ihre SHA256 steht in den jeweiligen `eval_manifest.json`-Dateien.
- Nicht enthalten: Videos (`.mkv`), `episodes.csv` je Trainingslauf.
