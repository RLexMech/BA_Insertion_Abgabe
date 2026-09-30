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
| Ordner hier | [B_quader/](B_quader/) |
| Ergebnis | Endstand D-038: 99,19 % (Trainingsfenster 5461 Episoden, 4096 Envs, Stopp bei 99 %), Replay 99,57 % (703 Episoden) (`DECISIONS.md:1727-1740`). Zwischenstände in [CHECKPOINTS.md](CHECKPOINTS.md). |
| Trainingsbefehl | nicht gespeichert. Rekonstruiert: `--num_envs 4096 --resume --stop-at-success-rate 0.99 env.fixture_pos_noise_xy=0.02 env.fixture_tilt_noise_rad=0.1745 env.fixture_yaw_noise_rad=0.7854`, `PROXYTASK_POCKET_SIDE_MM=32`. |
| Roboter / Regler / Aktion | wie A (UR10e, Gelenk-Deltas, 6-D, 0,02 rad, kein EMA). |
| Beobachtung | 25: A-Kanäle + cos/sin 4φ 19:21 + Taschen-Quaternion 21:25. Kein Rauschen. |
| Reward | wie A + Gier −2,0. Erfolg: Tiefe ≥ 25 mm UND alle 4 Ecken in der Tasche UND Ausrichtung ≥ 0,99. |
| Geometrie | Stift 30×30×50 mm, Tasche 32×32×35 mm → 1,0 mm Spiel je Achse. |
| Streuung (pro Episode) | Gelenke ±0,01 rad, Handgelenk 3 ±45°. Befehlszeile: Aufnahme xy ±0,02 m (Endstand) bzw. ±0,05 m (D-034), Neigung 0–10°, Gier ±45°. Im Code alle `0.0`. |
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
| Repo / Tag | `Ur5e_ContactRich_Insertion` / `stufe-C-aufnahme` = `5a6cc8e` (Linie `p2-rl-code` → … → `p5-kraftsensor`) |
| Ordner hier | [C_aufnahme/](C_aufnahme/) |
| Endpolicy | RT-206s1 (AutoDR, Seed 1, `model_1499.pt`), gewählt in D-190. Training: 97,3 %. |
| Trainingsbefehl | `.\scripts\rt_log.ps1 RT-206s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3` (`rt_logs/RT-206_expectation.md`) |
| Evaluation | Code `8299485d`, `play.py --eval-table`, deterministisch, 1000 Envs, je 1 Episode. RT-224 T-final: AutoDR s1 **0,982**, AutoDR-Mittel 0,970, No-DR-Mittel 0,262. RT-222 T-nominal 0,987, RT-223 T-auswahl 0,979, RT-225 T-grenze PASS (`rt_logs/VERDICTS.md:77-81`). Fixed-DR-Arm: RT-231/232. |
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

---

## Stufe D – Annäherung an Factory (vorläufig bis RT-244)

| | |
|---|---|
| Repo / Tag | `Ur5e_ContactRich_Insertion` / `stufe-D-factory` = **offen** (Linie `p5_mitAktion` → `p5-rlgames`) |
| Läufe bisher | RT-242s1..s5 AutoDR: 98,96–99,70 % (Epoche 91–100). RT-243s1..s5 Fixed-DR: s3 99,22 %, s4 99,63 %, s1/s2 0 %, s5 steigt noch. RT-244s1..s5 No-DR: läuft. |
| Roboter / Regler | wie C (OSC, kp 100/30, 15 Hz, 256 Schritte). |
| Aktion | 6-D, Clip ±1, **EMA 0,2**, 0,02 m / 0,097 rad, Box ±0,100 m / z 0,2915 m, Kegel 8,52°, Gier-Band ±12,30°. |
| Beobachtung | 34 = C-Kanäle + vorige Aktion 28:34 (EMA-Zustand). Nur über die Befehlszeile (`env.obs_wrench_mode=force_prevact`). Critic mit wahrer Taschenlage. **LSTM 2×1024** + MLP [512,128,64]. |
| Rauschen auf der Beobachtung | wie C + Gier-Glaubensfehler ±1,28° pro Episode (Befehlszeile). |
| Reward / Erfolg | wie C, aber action_rate 0. |
| Streuung | wie C, dazu Start bis 0,270 m, Aufnahme ±0,05 m in x/y/z (alle Arme), Home 0,400 m. Arme: AutoDR / Fixed-DR / No-DR. |
| PPO | **rl_games**, `T/agents/rl_games_ppo_cfg.yaml` = Factory-YAML von Isaac Lab 2.3.2. Geändert: max_epochs 200 → 100, save_frequency 100 → 25, minibatch 512 → 1024. γ 0,995, lr 1e-4 adaptiv (KL 0,008), horizon 128, 4 Mini-Epochen, Entropie 0, Critic lr 1e-4. |
| Evaluation | noch nicht gebaut (geplant in D-206). |

Karte, Tag und Pull-Befehl werden nach RT-244 und der Evaluation festgelegt.
