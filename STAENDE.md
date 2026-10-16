# Stände – Faktenkarten der 4 Stufen

Jede Karte beschreibt den Stand am **Ende** der Stufe, gelesen aus dem Code am Tag.
Die Übergänge und Gründe stehen in [CHECKPOINTS.md](CHECKPOINTS.md).
„Befehlszeile“ heißt: Der Wert steht im Code anders und wurde nur beim Lauf gesetzt.

Pfade in den Karten:
- A/B: `T` = `source/proxytask/proxytask/tasks/direct/proxytask/`
- C/D: `T` = `source/insertion/insertion/tasks/direct/insertion/`

---

## Stufe A – Zylinder Peg-in-Hole

| | |
|---|---|
| Repo / Tag | `ur10e-peg-insertion-rl` / `stufe-A-zylinder` = `a7c7ebb` (Branch `demo-insertion-sprint`) |
| Ordner hier | [A_zylinder/](A_zylinder/) |
| Ergebnis | Ø25 mm: „jeder Versuch“ nach < 3 min (1024 Envs). Ø28 mm / 1,0 mm Spiel: 100 % in 600 Iterationen. Nur beobachtet, kein Log (`docs/demo_sprint_results.md:241-257`). |
| Trainingsbefehl | nicht gespeichert. Basis: `scripts\rsl_rl\train.py --task=Template-Proxytask-Direct-v0 --headless` (`HANDOFF.md:136`). Konflikt in der Doku: 4096 oder 1024 Envs. |
| Roboter / Regler | UR10e (`UR10e_CFG`), **Gelenk-Deltas** → Gelenk-Positionsziele. Kein OSC, kein IK (`T/proxytask_env.py:296-304`). |
| Aktion | 6-D, 0,02 rad pro Schritt, Clip ±1, Klemmung an den Gelenkgrenzen, kein EMA. 60 Hz, 240 Schritte = 4 s. |
| Beobachtung | 19: Gelenkpos 0:6, Gelenkgeschw. 6:12, Spitze–Bohrung 12:15, EE-Quaternion 15:19. Kein Rauschen. |
| Reward | approach −2, Tiefenfortschritt +100, Erfolg +10, Aktion −0,01, falsche Tiefe −20, Ausrichtung −2 (`T/proxytask_env_cfg.py:82-103`). |
| Erfolg | Tiefe ≥ 25 mm UND Stift in der Bohrung UND Ausrichtung ≥ 0,99. |
| Streuung | nur Gelenke ±0,01 rad pro Episode (≈ 4 mm an der Spitze). Lage des Lochs fest. |
| PPO | rsl_rl, `T/agents/rsl_rl_ppo_cfg.py`. Cartpole-Vorlage von Isaac Lab (byte-gleich mit Repo `proxytask-transfer`). Geändert: Netz [32,32] → [128,128], Obs-Normalisierung an, max. 1500 Iterationen. Sonst Vorlage: lr 1e-3 adaptiv, γ 0,99, λ 0,95, Clip 0,2, 5 Epochen, 4 Minibatches, 16 Schritte/Env, Entropie 0,005. |
| Hinweis | D-012 nennt Factory als Vorlage für die Struktur, D-013 wollte die Factory-PPO-Werte. Im Code steht die Cartpole-Vorlage. |
| Seitenäste | `increment-3-contact-obs` (Kraft in der Beobachtung, nie lauffähig), `demo-reward-literature` (Reward aus der Literatur, nie trainiert). Beide nicht gemergt. |
| Vorläufer | Repo `peg_insert` (14.07.): manager-based, skrl, UR10, Diff-IK. Verworfen, nichts übernommen. |

**Stand auf den Trainings-PC holen (PowerShell):**
```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\proxytask_gen
git clone --filter=blob:none --sparse --branch stufe-A-zylinder https://github.com/RLexMech/ur10e-peg-insertion-rl.git Stufe_A_zylinder
cd Stufe_A_zylinder
git sparse-checkout set --no-cone "/pyproject.toml" "/scripts/**" "/source/**" "!**/CLAUDE.md"
git rev-parse HEAD
```
Erwartet: `a7c7ebb1bd7fe27d9c391cbd5b181c69fe5a6a9d`. Die USD-Dateien sind nicht in Git. Sie liegen in der alten Kopie unter `C:\Users\Simon\Desktop\Alexander_Pett\temp\proxytask_gen\proxytask` und müssen nach `source\proxytask\proxytask\tasks\direct\proxytask\assets` kopiert werden.

---

## Stufe B – Quader Peg-in-Hole + Generalisierung

| | |
|---|---|
| Repo / Tag | `ur10e-peg-insertion-rl` / `stufe-B-quader` = `97df8bf` (Linie `square-peg-insertion` → `generalisation-probe` → `generalisation-angle-probe`) |
| Ordner hier | [B_quader/](B_quader/): Grundlage `97df8bf3`; die drei Streuwerte der Taschenlage sind in dieser Abgabefassung direkt als Endeinstellungen voreingestellt. |
| Ergebnis | Endstand D-038: 99,19 % (Trainingsfenster 5461 Episoden, 4096 Envs, Stopp bei 99 %), Replay 99,57 % (703 Episoden) (`DECISIONS.md:1727-1740`). Zwischenstände in [CHECKPOINTS.md](CHECKPOINTS.md). |
| Trainingsbefehl | nicht gespeichert. Rekonstruiert: `--num_envs 4096 --resume --stop-at-success-rate 0.99 env.fixture_pos_noise_xy=0.02 env.fixture_tilt_noise_rad=0.1745 env.fixture_yaw_noise_rad=0.7854`, `PROXYTASK_POCKET_SIDE_MM=32`. |
| Roboter / Regler / Aktion | wie A (UR10e, Gelenk-Deltas, 6-D, 0,02 rad, kein EMA). |
| Beobachtung | 25: A-Kanäle + cos/sin 4φ 19:21 + Taschen-Quaternion 21:25. Kein Rauschen. |
| Reward | wie A + Gier −2,0. Erfolg: Tiefe ≥ 25 mm UND alle 4 Ecken in der Tasche UND Ausrichtung ≥ 0,99. |
| Geometrie | Stift 30×30×50 mm, Tasche 32×32×35 mm → 1,0 mm Spiel je Achse. |
| Streuung (pro Episode) | Gelenke ±0,01 rad, Handgelenk 3 ±45°. Abgabefassung: Aufnahme xy ±0,02 m, Neigung 0–10°, Gier ±45° direkt als Standardwerte im Code. Im Original-Tag alle `0.0`, dort über die Befehlszeile gesetzt. ±0,05 m gehört zur früheren Stufe D-034. |
| PPO | wie A (keine Änderung zwischen `a7c7ebb` und `97df8bf`). |
| Nicht zitieren | Lauf `2026-08-06_18-43-17_s30b32rxy50` (verworfen). |

**Stand auf den Trainings-PC holen (PowerShell):**
```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\proxytask_gen
git clone --filter=blob:none --sparse --branch stufe-B-quader https://github.com/RLexMech/ur10e-peg-insertion-rl.git Stufe_B_quader
cd Stufe_B_quader
git sparse-checkout set --no-cone "/pyproject.toml" "/scripts/**" "/source/**" "!**/CLAUDE.md"
git rev-parse HEAD
```
Erwartet: `97df8bf385e6138ba30b37d56852fb411fe6f5ce`. USD-Dateien aus `C:\Users\Simon\Desktop\Alexander_Pett\temp\proxytask_gen\generalisation-angle-probe` nach `source\proxytask\proxytask\tasks\direct\proxytask\assets` kopieren.

---

## Stufe C – Aufnahme in realitätsnaher Szene (UR5e)

| | |
|---|---|
| Repo / Endstand | `Ur5e_ContactRich_Insertion` / `153b8b4` (24.09., Linie `p2-rl-code` → … → `p5-kraftsensor`) |
| Tag | `stufe-C-aufnahme` = `5a6cc8e` (15.09.) = Trainingscode von RT-206s/RT-207s. Der Tag bleibt dort. |
| Ordner hier | [C_aufnahme/](C_aufnahme/) = `153b8b4`. Dazu aus `ff2aadd8`: `rt_logs/RT-232_expectation.md`, `rt_logs/RT-272_expectation.md` (Läufe nach dem Commit). |
| Code der Läufe (Kopfzeile `[rt_log] git:`) | Training RT-206s/RT-207s `5a6cc8e`, RT-231s `153b8b4`. Tests RT-222…225, RT-232 und Spur RT-272b `8299485d` (`code_commit` in jeder `eval_manifest.json`). |
| Abstand zum Endstand | `git diff --stat 5a6cc8e 153b8b4 -- source`: 4 Dateien, +408/−29. Commits: `5f97b08c`, `deb51c02`, `500f59dd` (Modus `dr_mode=table` für die Tests), `3c6dabcd` (Modus `dr_mode=fixed_dr`, D-192). `git diff --stat 8299485d 153b8b4 -- source` enthält nur `3c6dabcd`. |
| Endpolicy | RT-206s1 (AutoDR, Seed 1, `model_1499.pt`), gewählt in D-190. Training: 97,3 %. |
| Trainingsbefehl | `.\scripts\rt_log.ps1 RT-206s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3` (`rt_logs/RT-206_expectation.md`). No-DR (RT-207): `env.dr_mode=off`. Fixed-DR (RT-231): `env.dr_mode=fixed_dr` (`rt_logs/RT-231_expectation.md`). |
| Evaluation | Code `8299485d`, `play.py --eval-table`, deterministisch, 1000 Envs, je 1 Episode. RT-224 T-final: AutoDR s1 **0,982**, AutoDR-Mittel 0,970, No-DR-Mittel 0,262. RT-222 T-nominal 0,987, RT-223 T-auswahl 0,979, RT-225 T-grenze PASS (`rt_logs/VERDICTS.md:77-81`). Fixed-DR-Arm: Training RT-231, Test RT-232, T-final-F je Seed 0 / 0 / 0,925 / 0,995 / 0,949 (Urteile in `D_factory/rt_logs/VERDICTS.md:82-83`). Daten: `auswertung/daten/experiment_C_rsl_rl/`. |
| Roboter / Regler | UR5e + angeschweißtes Werkzeug, Schwerkraft an. **OSC** (Isaac Lab `OperationalSpaceController`), kp 100 N/m / 30, kritisch gedämpft (`T/insertion_env_cfg.py:213, 246-248`). |
| Aktion | 6-D Posen-Delta (0:3 m, 3:6 rad), Clip ±1, 0,02 m / 0,097 rad pro Schritt, kein EMA. Neigungskegel 8,52°, Box ±0,08 m. 15 Hz, 256 Schritte = 17 s. |
| Beobachtung | 28: Gelenkpos 0:6, Gelenkgeschw. 6:12, tip_rel 12:15, EE-Quat 15:19, cos/sin φ 19:21, Taschen-Quat 21:25, Kraft 25:28. Actor = Critic. MLP 128×128, **kein LSTM, keine vorige Aktion**. |
| Rauschen auf der Beobachtung | Taschenlage ±5 mm pro Episode, Kraft σ 3,5 N pro Schritt, Griff ±3 mm pro Episode. |
| Reward | SDF-Kerne × SAPU: engaged 1,0, success 1,0, progress 100/m, time −1/256, action_rate −0,0034, Erfolgs-Pauschale, Abbruch −1,0. |
| Erfolg / Ende | Tiefe 33–36 mm UND Durchdringung < 0,29 mm UND Spitze im Taschenvolumen. Ende bei Erfolg, Timeout oder Kraft > 30 N. |
| Streuung (pro Episode) | AutoDR (erst über die Befehlszeile an): seitlicher Start 0–30 mm, Gier ±5°, Neigung 0–10°, Starthöhe bis 120 mm, Reibung 0,08–0,20. Aufnahme xy ±5 mm fest. |
| Geometrie | Spiel in der Sim 0,5876 mm quer / 1,60 mm längs, in echt 0,4 / 0,9 mm (D-087). Aufnahme aus `CAD/Aufnahme_real_v1.stp`. |
| PPO | rsl_rl, `T/agents/rsl_rl_ppo_cfg.py`, Werte wie A/B. 256 Envs × 16 Schritte = 4096 Samples pro Iteration. |
| Wechsel in der Stufe | Gelenk-Deltas 60 Hz → OSC 15 Hz (D-177, 03.09.). AutoDR (D-178..181). Rauschen auf der Beobachtung (D-182/183/185). Kraftgrenze 300 → 50 → 30 N. Grenze RT-180: Urteile davor gelten nicht als Beleg. |
| Seitenäste | `p3-rl-osc` (eigener Impedanzregler, nie gelaufen), `p5-osc-rot-gain` (Rotationssteifigkeit, gemergt), `p5-reward` (Ausrichtungsterm, verworfen), `p1-gains` (UR10e-Gains, ersetzt durch D-105). Test außerhalb des Trainingsbereichs: `p6-TestAusserhalb`, RT-226..230 (nur 15° Neigung fällt ab, −9,3 Punkte). |

**Stand auf den Trainings-PC holen (PowerShell):**
```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion
git clone --filter=blob:none --sparse --branch stufe-C-aufnahme https://github.com/RLexMech/Ur5e_ContactRich_Insertion.git Stufe_C_aufnahme
cd Stufe_C_aufnahme
git sparse-checkout set --no-cone "/rt_logs/**" "/scripts/**" "/source/**" "!**/CLAUDE.md"
git rev-parse HEAD
Copy-Item -Recurse ..\Phase5_Robustheit_v1\source\insertion\insertion\tasks\direct\insertion\assets source\insertion\insertion\tasks\direct\insertion\assets
```
Erwartet: `5a6cc8e3f0f40e34952f5092e742ad9a28e1cff1`. Die Läufe von C nutzten `pip install -e source\insertion`. Es gibt immer nur eine solche Installation.

**Endstand C (Fixed-DR-Training und Tests) holen (PowerShell):**
```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion
git clone --filter=blob:none --sparse --no-checkout https://github.com/RLexMech/Ur5e_ContactRich_Insertion.git Stufe_C_endstand
cd Stufe_C_endstand
git sparse-checkout set --no-cone "/rt_logs/**" "/scripts/**" "/source/**" "!**/CLAUDE.md"
git checkout --detach 153b8b432b92b3218a392c5d1a1f377861603563
git rev-parse HEAD
```
Erwartet: `153b8b432b92b3218a392c5d1a1f377861603563` (auf dem Laptop am 2026-10-12 geprüft: Klon von GitHub, `git rev-parse HEAD` gibt diesen Wert). Assets wie oben kopieren.

---

## Stufe D – Annäherung an Factory, Endkonfiguration (Kap. 4.5)

| | |
|---|---|
| Repo / Tag | `Ur5e_ContactRich_Insertion` / `stufe-D-factory` = `c1a635e9` (30.09., Linie `p5_mitAktion` → `p5-rlgames`) |
| Ordner hier | [D_factory/](D_factory/) = `c1a635e9`. `rt_logs/` nur mit den Dateien der zitierten Läufe: aus dem Commit `VERDICTS.md` und RT-233, 235, 236, 238, 239, 242, 243; aus `ff2aadd8` RT-244 (Fassung mit Ergebnis), 247, 253, 254, 260, 271. |
| Code der Läufe (Kopfzeile `[rt_log] git:`) | Training RT-242s `e10c18b`, RT-243s `9219ab0`, RT-244s `7cd3992`. Tests (alle Blöcke der Endstudie) `c1a635e9` (`code_commit` in jeder `eval_manifest.json`). Fortsetzung RT-253s `c1a635e`. MLP-Ablation RT-260s `ddc9d0b`. Spur RT-271: s2 `13072c3`, s4 `8be1e99`. D8: kein Commit im Log (Wiedergabe von RT-243s4, 03.10.). |
| Abstand zum Endstand | `git diff --stat e10c18bb c1a635e9 -- source scripts/rl_games`: `play.py`, `run_eval.ps1` (Evaluation) und in `source/` nur D-206 (3), die Option `dr_table_allow_outside` (Standard aus, außerhalb von `dr_mode=table` verweigert). `9219ab0` und `7cd3992` haben in `source/` denselben Abstand. `ddc9d0b`: `git diff c1a635e9 ddc9d0b -- source scripts/rl_games` ist leer. RT-271 läuft auf späterem Code (u. a. `00d01d49`: `--trace-obs` in `play.py`). |
| Ergebnis Training (Epoche 100) | Mittel über 5 Seeds: Auto-DR 99,4 %, No-DR 99,7 %, Fixed-DR 40,9 % (Tab. A-3, `auswertung/daten/endstudie_rl_games/training_auszug/curves.csv`). |
| Ergebnis Haupttest (RT-247/248, je 1000 Testfälle) | Auto-DR 997 / 989 / 989 / 991 / 992. Fixed-DR 0 / 0 / 982 / 995 / 78. No-DR 99 / 155 / 178 / 98 / 96 (Seeds 1–5, Tab. A-4, `…/evaluation/summary.json`). Keine finale Policy (D-225, nur im Original-Repo). |
| Trainingsbefehl | `.\scripts\rt_log.ps1 RT-242s1 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402` (`rt_logs/RT-242_expectation.md`). Fixed-DR (RT-243): `env.dr_mode=fixed_dr`. No-DR (RT-244): `env.dr_mode=off`. MLP-Ablation (RT-260): dazu `"~agent.params.network.rnn" "~agent.params.config.central_value_config.network.rnn"`. Fortsetzung (RT-253): `--checkpoint <ep_100> --max_iterations 250`. |
| Testbefehle | `.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-247`, `-Block T-final-F -Rt RT-248`, `-Block T-nominal -Rt RT-249`, `-Block T-nominal-F -Rt RT-250` (`rt_logs/RT-247_expectation.md:78-81`). `-Block T-ood -Rt RT-254`, `-Block T-final -Rt RT-255 -Noise pn2` / `RT-256 -Noise gx2` / `RT-257 -Noise py2` (`rt_logs/RT-254_expectation.md:83-86`). Fixed-DR für T-ood und Rauschen: `rt_log.ps1`-Schleife RT-258/259 (`rt_logs/RT-254_expectation.md:108`). |
| Roboter / Regler | wie C: OSC, kp 100 / 30 (`T/insertion_env_cfg.py:250-251`), `control_mode = "osc"` (`T/insertion_env_cfg.py:213`). OSC setzt 8 Physikschritte je Aktion (`osc_decimation`, `T/insertion_env_cfg.py:339`, angewandt in `T/insertion_env_cfg.py:1952-1958`): 120 Hz Physik, 15 Hz Policy, 256 Schritte = 17,1 s. |
| Aktion | 6-D, Clip ±1, **EMA 0,2** (`T/insertion_env_cfg.py:272`), 0,02 m / 0,097 rad (`T/insertion_env_cfg.py:260-261`). Box ±0,100 m (`:293`), z 0,2915 m (`:316`), Kegel 8,52° (`:321`), Gier-Band ±12,30° (`:330`). |
| Beobachtung | 34 = C-Kanäle + vorige, geglättete Aktion 28:34. Nur über die Befehlszeile (`env.obs_wrench_mode=force_prevact`, Standard `"force"` in `T/insertion_env_cfg.py:236`). Critic: dieselben 34 aus dem wahren Zustand (`T/insertion_env.py:2226-2246`). **LSTM 2×1024** + MLP [512,128,64] (`T/agents/rl_games_ppo_cfg.yaml:32, 41-47`). |
| Rauschen auf der Beobachtung | Taschenlage ±5 mm je Achse (`T/insertion_env_cfg.py:1233`, Name `std`, wirkt als Halbbreite der Gleichverteilung), Griff ±3 mm (`:1267`), Kraft σ 3,5 N je Schritt. Gier-Glaubensfehler ±1,28° je Episode nur über die Befehlszeile (`env.pocket_obs_yaw_rad=0.0223402`, Standard 0 in `:1278`). |
| Reward / Erfolg | wie C, aber Aktionsrate 0 (`T/insertion_env_cfg.py:1456`). SDF mit 64 000 Punkten (`:1486`). Abbruch bei 30 N. |
| Streuung | Aufnahme ±0,05 m in x/y/z, alle Arme (`T/insertion_env_cfg.py:791, 800`). Starthöhe bis 0,270 m (`T/autodr.py:458`), Reibung 0,08–0,20 (`T/autodr.py:464`), Home 0,400 m (`T/insertion_tasks_cfg.py:119`). Arme: Auto-DR / Fixed-DR / No-DR. |
| PPO | **rl_games**, `T/agents/rl_games_ppo_cfg.yaml` = Factory-YAML von Isaac Lab 2.3.2. Geändert: max_epochs 200 → 100 (`:74`), save_frequency 100 → 25 (`:76`), minibatch 512 → 1024 (`:83`). γ 0,995 (`:67`), λ 0,95 (`:68`), lr 1e-4 adaptiv, KL 0,008 (`:69-72`), horizon 128 (`:82`), 4 Mini-Epochen (`:84`), Entropie 0 (`:79`), Critic lr 1e-4 (`:93`). |
| Evaluation | `scripts/rl_games/run_eval.ps1` + `scripts/rl_games/play.py`, deterministisch, Tabelle `scripts/eval/eval_table_seed10_c18.csv` (T-final), T-ood `eval_table_ood_seed10_c18.csv`. Daten: `auswertung/daten/endstudie_rl_games/`. |

**Tabellen der Arbeit → Code** (geprüft am 2026-10-12 gegen `D_factory/` und `RT-242s1_env.yaml` / `_agent.yaml`). Die Werte der Tabellen 4-10, 4-11, 4-12, 4-13, 4-14 und 4-15 stehen in den Zeilen oben oder in diesen Zeilen:
- Tab. 4-11: Antriebe ohne Steifigkeit (`T/insertion_env.py:1649-1652`), Aufnahme kinematisch (`T/insertion_env.py:1560-1562`), Schnittkraft mit Abzug der Gewichtskraft (`T/insertion_env.py:2141-2149`), Glättung 0,25 (`T/insertion_env_cfg.py:1286`), Schwerkraft und TGS 16 / 1 (`T/ur5e_cfg.py:189, 199-200`), dt 1/120 (`T/insertion_env_cfg.py:620`).
- Tab. 4-14: Kernbreiten und b (`T/insertion_tasks_cfg.py:1623, 891, 1520, 1627-1629`), δth 0,2938 mm (`T/insertion_tasks_cfg.py:1591`), Mindesttiefe 10,8 mm (`T/insertion_env_cfg.py:1319`), w_p 100 (`T/insertion_env_cfg.py:1334`), Zeitstrafe (`T/insertion_env_cfg.py:1393, 1401`), Kraftabbruch (`T/insertion_env_cfg.py:1080, 1106`).
- Tab. 4-15: alle Werte in `T/agents/rl_games_ppo_cfg.yaml` (Zeilen oben), 256 Envs über `--num_envs 256`.

Abweichungen zwischen Arbeit und Code (die Arbeit wurde nicht geändert):
1. Tab. 4-8 nennt das Spiel von C (0,588 / 1,600 mm) als **Gesamtspiel**, das von B (1,0 mm) **je Seite**. Je Seite wären es bei C 0,294 / 0,800 mm.
2. Kap. 4.5.5 sagt, abweichend von Factory seien nur Epochen und Mini-Batch-Größe. Im Code weichen auch das Checkpoint-Intervall (25 statt 100) und die Zahl der Envs (256 statt 128) ab (Kommentare in `T/agents/rl_games_ppo_cfg.yaml:76, 83`).
3. Tab. 4-15 nennt für den Critic eine feste Lernrate 1e-4. Die YAML setzt auch für den Critic `lr_schedule: adaptive` mit KL 0,008 (`:94-95`). Ob rl_games 1.6.1 sie anpasst, ist nicht geprüft.
4. Tab. 4-13 nennt den Gierfehler nur beim Quaternion der Aufnahme. Er dreht auch die Lage des Bezugspunkts und cos φ / sin φ (`T/insertion_env.py:2092-2113`). Der Text in Kap. 4.5.3 sagt das richtig.
5. `RT-242s1_env.yaml` zeigt `decimation: 2` und `observation_space: 28`. Grund: `scripts/rl_games/train.py` schreibt die YAML vor dem Aufbau der Umgebung. `RT-242s1_demo_metrics.json` zeigt die echten Werte: 15 Hz, 17,07 s, `force_prevact-34`.

Nicht geprüft: Verhalten der Bibliotheken (Anpassung der Lernrate, Start-log σ), die SDF-Auflösung in den USD-Dateien.

**Stand auf den Trainings-PC holen (PowerShell):**
```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion
git clone --filter=blob:none --sparse --branch stufe-D-factory https://github.com/RLexMech/Ur5e_ContactRich_Insertion.git Stufe_D_factory
cd Stufe_D_factory
git sparse-checkout set --no-cone "/rt_logs/**" "/scripts/**" "/source/**" "!**/CLAUDE.md"
git rev-parse HEAD
Copy-Item -Recurse ..\Phase5_RlGames_v1\source\insertion\insertion\tasks\direct\insertion\assets source\insertion\insertion\tasks\direct\insertion\assets
```
Erwartet: `c1a635e969f72efd5b9062d2c8994befd29ff100`. Auf dem Laptop am 2026-10-12 geprüft: Klon von GitHub mit `--branch stufe-D-factory`, `git rev-parse HEAD` gibt diesen Wert. Die Läufe von D nutzten `pip install -e source\insertion` aus `Phase5_RlGames_v1`. Es darf nur eine solche Installation geben (RT-242a–d waren ungültig, weil `insertion` aus einem anderen Klon kam, `rt_logs/VERDICTS.md:93`).
