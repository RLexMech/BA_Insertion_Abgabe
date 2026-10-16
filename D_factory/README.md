# Experiment D – Endkonfiguration der Fügeaufgabe

Dieser Ordner enthält die Abgabefassung von Experiment D aus der Bachelorarbeit.
Grundlage ist `c1a635e969f72efd5b9062d2c8994befd29ff100` aus
`Ur5e_ContactRich_Insertion` (Tag `stufe-D-factory`). Die Kommentare wurden
gekürzt, die gemeinsamen Studienwerte als Standard gesetzt und die Tests daran
angepasst. Die Abgabefassung ist deshalb keine unveränderte Kopie des Tags.

## Hauptstudie

| Bedingung | Läufe | Einstellung |
|---|---|---|
| Auto-DR | RT-242s1–s5 | `env.dr_mode=autodr` |
| Fixed-DR | RT-243s1–s5 | `env.dr_mode=fixed_dr` |
| No-DR | RT-244s1–s5 | `env.dr_mode=off` |

Jede Bedingung wird mit Seeds 1–5 jeweils neu trainiert. Das Budget beträgt
**100 Epochen bei 256 parallelen Umgebungen**. No-DR behält die Anpassung der
Starthöhe, die Streuung der Aufnahmeposition und das Beobachtungsrauschen bei.

## Einstellungen im Code

| Größe | Endstand D |
|---|---|
| Roboter / Regler | UR5e, OSC, 120 Hz Physik und 15 Hz Aktionen |
| Aktion | 6 Lageänderungen, EMA 0,2, Grenzen 0,02 m und 0,097 rad je Achse und Schritt |
| Beobachtung | `force_prevact`: 34 Werte, einschließlich Kraft und voriger geglätteter Aktion |
| Lernverfahren | PPO mit `rl_games` |
| Netz | LSTM 2×1024 mit Layer-Normalisierung, MLP 512/128/64; separater Critic mit wahrem Zustand |
| Budget / Standard-Seed | 100 Epochen, 256 Umgebungen, Seed 1 |
| Startanpassung | `start_floor_m=-0.030`, `autodr_stall_buffers=3` |
| Standardbedingung | `dr_mode=off`; Auto-DR und Fixed-DR ausdrücklich wählen |
| Aufnahmeposition | ±50 mm in x, y und z |
| Obergrenze der Starthöhe | 270 mm |
| Beobachtungsfehler | Aufnahme ±5 mm je Achse, Griff ±3 mm, Gier ±1,28° (`0.0223402` rad), Kraft σ=3,5 N |
| Aktionsänderungsstrafe | Aus (`action_rate_scale=0.0`) |
| Kraftabbruch / Episodenlimit | 30 N / 256 Schritte, etwa 17,07 s |

Die übrigen PPO-, Geometrie-, Reward- und Erfolgsparameter stehen unverändert in
`source/insertion/insertion/tasks/direct/insertion/`. Die aktive
Trainingskonfiguration ist `agents/rl_games_ppo_cfg.yaml`.

## Training

Voraussetzungen: eingerichtete Isaac-Lab-/rl_games-Umgebung, die Szenen-USDs auf
dem Trainingsrechner und die Installation dieser Task-Version mit
`pip install -e source/insertion`. Die folgenden Befehle werden aus `D_factory`
ausgeführt. Die drei Beispiele verwenden Seed 1; für die Studie jeweils mit
Seeds 1–5 ausführen.

```powershell
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=autodr
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=fixed_dr
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=off
```

Die gemeinsamen Endeinstellungen brauchen keine zusätzlichen Startargumente.
`scripts/rsl_rl/` enthält frühere Entwicklungswege; die Hauptstudie D verwendet
`scripts/rl_games/`.

## Auswertung und Zusatzversuche

Die historischen Auswertungs-Manifeste nennen `c1a635e9`. Die zugehörigen Daten
liegen im Abgabe-Repo unter `auswertung/daten/endstudie_rl_games/`.
`scripts/rl_games/run_eval.ps1` startet die vorgesehenen Testblöcke mit den
zugehörigen Checkpoints und Tabellen. Es setzt `env.start_floor_m=0.020`
ausdrücklich, damit die Auswertung keine Trainings-Startanpassung verwendet.
Bei direkten Aufrufen von `scripts/rl_games/play.py --eval-table` oder
`--dr-table` denselben Override angeben. Wird `start_tip_above_entrance`
geändert, müssen beide Werte gleich sein.

Die Fortsetzung RT-253 bis 250 Epochen und die MLP-Ablation RT-260 sind
Zusatzversuche; sie ändern nicht die Standards der 100-Epochen-Hauptstudie.
Laufnotizen in `rt_logs/` dokumentieren auch solche Zusatzversuche.
Trainierte Modelle und Szenen-USDs sind nicht Bestandteil dieses Repos.

## Prüfungen

```powershell
python scripts/check_mutation_anchors.py
python scripts/check_mutation_anchors.py --self-test
python scripts/selftest_checks.py --offline
```

Die Offline-Tests prüfen unter anderem Mathematik, Quelltext-Verknüpfungen und
gezielt eingebaute Fehler. Sie ersetzen keinen Lauf in Isaac Sim.
Messskripte setzen ihre rauschfreien Bedingungen ausdrücklich, unabhängig von
den Trainingsstandards.
