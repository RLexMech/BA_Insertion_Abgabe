# Checkpoints – wo jede Stufe endet und wo die nächste beginnt

Diese Datei beschreibt den Ausgangspunkt (Cartpole-Vorlage → A) und die 3 Übergänge A→B, B→C und C→D.
Für jeden Übergang steht hier: der letzte Stand der alten Stufe, der erste Stand der neuen Stufe, der Grund und was sich geändert hat.
Die vollen Werte jeder Stufe stehen in [STAENDE.md](STAENDE.md).

So liest man die Änderungstabellen:
- Die Spalte „neu“ zeigt den Stand am **Ende** der neuen Stufe.
- „Art“: **geplant** = vor dem Beginn der Stufe entschieden. **Reaktion** = während der Stufe als Antwort auf ein Ergebnis eingeführt.
- „Beleg“: die Entscheidung (D-Nummer) oder der Commit (SHA), der die Änderung trägt.

Alle Belege sind Commits (SHA) oder Entscheidungen (D-Nummer).
- A und B: Repo `RLexMech/ur10e-peg-insertion-rl`, `DECISIONS.md`.
- C und D: Repo `RLexMech/Ur5e_ContactRich_Insertion`, `DECISIONS.md`.

## Zeitleiste

| Stufe | Beginn | Ende (Tag) | Ergebnis am Ende |
|---|---|---|---|
| A Zylinder | `2420516` (27.07.) | `stufe-A-zylinder` = `a7c7ebb` (27.07.) | Zylinder gefügt, Reward-Hacks behoben |
| B Quader | `e18a4b8` (28.07.) | `stufe-B-quader` = `97df8bf` (16.08.) | 99,19 % mit Versatz ±2 cm, Neigung 0–10°, Gier ±45° |
| C Aufnahme | `a9f2d9cf` (30.08.) | `stufe-C-aufnahme` = `5a6cc8e` (15.09.) | Evaluation 98,2 % (RT-224, AutoDR s1) |
| D Factory | `83f6033c` (25.09.) | `stufe-D-factory` = **offen bis RT-244** | vorläufig, siehe STAENDE.md |

---

## Checkpoint 0: Cartpole-Vorlage → A

A beginnt mit der Direct-Vorlage von Isaac Lab (Cartpole, rsl_rl). Die Vorlage liegt unverändert im Repo `RLexMech/proxytask-transfer` (25.07.). Sie kam mit `3f81a96` in `ur10e-peg-insertion-rl` (D-021).

| Größe | Cartpole-Vorlage | A | Beleg |
|---|---|---|---|
| Aufgabe | Wagen balanciert Stab | UR10e fügt Zylinder in Bohrung | – |
| Aufbau der Umgebung | Direct-Umgebung | übernommen | `3f81a96` |
| Bibliothek | rsl_rl, PPO | übernommen | `3f81a96` |
| Takt der Policy | 60 Hz (1/120 s, Faktor 2) | übernommen | `proxytask_env_cfg.py` |
| Parallele Umgebungen | 4096 | Code 128, Läufe 1024 | `proxytask_env_cfg.py`, `demo_sprint_results.md:242` |
| Episodenlänge | 5 s | 4 s (ab `a3807db`, vorher 5 s) | `a3807db` |
| Aktion | 1 Kraft auf den Wagen, Skala 100 N | 6 Änderungen der Gelenkwinkel, 0,02 rad je Schritt | `proxytask_env_cfg.py` |
| Beobachtung | 4 (Wagen, Stab) | 19: Gelenkwinkel (6), Gelenkgeschwindigkeiten (6), Spitze zur Bohrung (3), Quaternion (4) | `proxytask_env_cfg.py` |
| Reward | Überleben +1, Abbruch −2, Stabwinkel −1, Wagengeschw. −0,01, Stabgeschw. −0,005 | Annäherung −2, Tiefenfortschritt +100, Erfolg +10, Aktion −0,01, Tiefe außerhalb der Bohrung −20, Ausrichtung −2; Abbruch unter der Platte kostet die Strafen der restlichen Schritte | `proxytask_env_cfg.py:82-103`, `a3807db`, `c53ebbc` |
| Streuung beim Start | Stabwinkel ±0,25·π rad (±45°) | Gelenkwinkel ±0,01 rad | Cartpole: `proxytask_env.py` multipliziert den Bereich mit π |
| PPO: Lernrate, γ, λ, Clip, Entropie, Epochen, Mini-Batches, Schritte | 1e-3 adaptiv, 0,99, 0,95, 0,2, 0,005, 5, 4, 16 | übernommen | `agents/rsl_rl_ppo_cfg.py` |
| Netz Actor/Critic | 2 × 32, ELU | 2 × 128, ELU | `agents/rsl_rl_ppo_cfg.py` |
| Normierung der Beobachtung | aus | ein | `agents/rsl_rl_ppo_cfg.py` |
| Iterationen, Obergrenze | 150 | 1500 (gelaufen: 600) | `agents/rsl_rl_ppo_cfg.py` |

Hinweis: D-012 nennt Factory als Vorlage für die Struktur, D-013 wollte die PPO-Werte von Factory. Im Code steht die Cartpole-Vorlage.

---

## Checkpoint 1: A → B (Zylinder → Quader)

**Letzter Stand A:** `a7c7ebb` (27.07.). Das Verhalten steht seit `c53ebbc` fest (alter Tag `demo-insertion-working`). Danach kamen nur noch Werkzeuge und Berichte.

**Was A gezeigt hat** (`docs/demo_sprint_results.md` @ `a7c7ebb`):
- Die Policy lernt in dieser Umgebung. Ø25 mm (2,5 mm Spiel): „jeder Versuch“ nach unter 3 Minuten bei 1024 Envs. Ø28 mm (1,0 mm Spiel): 100 % in 600 Iterationen.
- Zwei Reward-Hacks wurden gefunden und behoben:
  1. Der Stift fuhr über die Tischkante und erreichte die Tiefe von unten. Behoben in `a3807db`.
  2. Der Stift lag quer über der Bohrung. Behoben in `c53ebbc`.
- Grenze: Die Ergebnisse sind nur beobachtet (Simulation und TensorBoard). Kein Log ist gespeichert. Die Lage des Lochs war fest.

**Erster Stand B:** `e18a4b8` (28.07.), „Recast the task geometry for the square peg and pocket“. Der Branch `square-peg-insertion` zweigt bei `a7c7ebb` ab.

**Warum:** D-029 (28.07.). Beim Zylinder ist die Drehung um die eigene Achse egal. Die Policy lernt also keine Ausrichtung. Der Quader macht die Drehung zu einer echten Aufgabe. Das ist die kleinste Änderung, die das erreicht.

**Was sich am Übergang ändert:**

| Größe | A | B | Art | Beleg |
|---|---|---|---|---|
| Fügeobjekt | Zylinder Ø25 / Ø28 mm | Quader 30 × 30 × 50 mm | geplant | D-029 |
| Gegenstück | Bohrung Ø30 mm | Tasche mit quadratischem Querschnitt, Kante 45 → 36 → 32 mm, 35 mm tief (1,0 mm Spiel je Achse bei 32 mm) | geplant | D-029, D-033 |
| Beobachtung | 19 | + Drehwinkel als (cos 4φ, sin 4φ) → 21 | geplant | `ef995e8` |
| | | + Orientierung der Tasche als Quaternion → 25 | Reaktion | D-037, `7728a07` |
| Reward | 6 Terme | + Term für den Drehwinkel (Gewicht 2,0) → 7 | geplant | D-029, `ef995e8` |
| Erfolgsbedingung | Spitze in der Bohrung | alle 4 Ecken in der Tasche | geplant | `ef995e8` |
| Startlage | Gelenke ±0,01 rad | + letztes Handgelenk ±45° (Drehung um die Fügeachse) | geplant | D-029 |
| Lage der Tasche | fest | Versatz ±5 cm (allein), Kippung bis 10°, Gier ±45°; Endstand kombiniert: ±2 cm + 10° + ±45° | Reaktion (D-035: feste Policy 0 % bei +2 cm) | D-034, D-036, D-037, D-038 |
| Parallele Umgebungen | 1024 | 1024 → 4096 (Endstand) | – | D-038 |
| Regelung, Aktion, Takt, Episodenlänge, PPO | Gelenk-Deltas, 0,02 rad, 60 Hz, 4 s, rsl_rl | unverändert | – | `git diff a7c7ebb 97df8bf -- agents` leer |

---

## Checkpoint 2: B → C (Quader-Vorstudie → echtes Teil, UR5e)

**Letzter Stand B:** `97df8bf` (16.08.). Das letzte Ergebnis steht in `e12e1b8` (D-038).

**Was B gezeigt hat:** Die Versätze kamen einzeln, danach zusammen:
1. Quader fest: 99,15 % (Tasche 32 mm).
2. D-035: Die feste Policy scheitert bei +2 cm Versatz (0 %). Sie drückt auf die alte Stelle.
3. D-034: Lage der Aufnahme pro Episode ±5 cm → 99,65–99,95 %. Replay: 100 % bei 0, ±2 und ±5 cm.
4. D-036/D-037: Neigung der Tasche bis 10° (Beobachtung 21 → 25).
5. D-038: Gier ±45°. Endstand mit allem zusammen: **99,19 %**, Replay 99,57 % (703 Episoden).

Grenze: Die Kombination lief mit ±2 cm, nicht mit ±5 cm. Die Streuwerte sind nur über die Befehlszeile gesetzt (im Code steht `0.0`). Kein vollständiger Trainingsbefehl ist gespeichert.

**Erster Stand C:** `a9f2d9cf` (30.08., Branch `p2-rl-code`). Hier läuft `train.py` zum ersten Mal mit der echten Aufnahme und dem echten Reward (RT-104, Messung des Durchsatzes). Der erste echte PPO-Lauf ist RT-105. Die neue Repo beginnt schon am 16.08. (`8f208e25`). Bis zum 30.08. wurde dort die Szene gebaut, ohne RL-Training.

**Warum:**
- D-043 (16.08.): Zylinder und Quader sind nur eine Vorstudie. Die Aufgabe der Arbeit ist das echte Teil mit dem UR5e.
- D-042 (16.08.): Was in der Vorstudie geprüft ist, wird mit einem kurzen Eintrag übernommen.
- D-061/D-063 (17.08.): Zuerst kommt die Szene. Das Paket der Vorstudie wird als `insertion` kopiert.
- Für den UR5e gibt es keine eigene Entscheidung. Er ist der Roboter der echten Zelle (D-058). D-092 schreibt seine Konfiguration aus, D-105 schaltet die Schwerkraft an.

**Was sich am Übergang ändert:**

| Größe | B | C | Art | Beleg |
|---|---|---|---|---|
| Roboter | UR10e | UR5e | geplant | D-043, D-058, D-092 |
| Fügeobjekt | Quader | Werkzeugkörper aus Sauggreifer und Fügeteil (143,5 × 90 mm) | geplant | D-088, D-121 |
| Gegenstück | Tasche | Aufnahme aus CAD, etwas mehr Spiel als die echte: 0,59 / 1,60 mm statt 0,4 / 0,9 mm; frei stehend | geplant; frei stehend: Reaktion | D-087, D-121, D-156 |
| Kollisionsmodell | Stift: Box (konvexe Hülle = exakt); Tasche: 9 Quader als Box-Kollider | Aufnahme als SDF-Netz | geplant | `scripts/add_fixture_collision.py` |
| Regelung | Gelenkraum (Gelenk-Deltas auf PD-Antriebe) | Impedanzregler im Arbeitsraum (OSC), kp 100 / 30 | Reaktion (RT-144a: im Gelenkraum alle 6 Gelenke in Sättigung, ≥ 250 N) | D-177 |
| Aktion | 6 Gelenk-Deltas, 0,02 rad | 6-D Posen-Delta, 0,02 m / 0,097 rad, Kegel 8,52°, Box ±0,08 m | Reaktion (mit OSC) | D-177 |
| Takt | 60 Hz | 15 Hz (1/120 s, Faktor 8, wie Factory) | Reaktion | D-177 |
| Startlage | über der Öffnung | in der Tasche (−30 mm), Curriculum über die Starthöhe bis 120 mm | Reaktion | `e6e7d874`, D-178..181 |
| Episodenlänge | 4 s (240 Schritte) | 256 Schritte: 4,3 s bei 60 Hz, 17,1 s bei 15 Hz | geplant | D-113 |
| Beobachtung | 25 (cos 4φ) | 28: cos φ statt cos 4φ, dazu geglättete, tarierte Kraft | geplant | D-107, D-114 |
| Reward | 7 Terme | neu aufgebaut (SDF-Kerne, Boni, Fortschritt, Zeit, Aktionsrate) | geplant | D-109 |
| Erfolgs- und Abbruchbedingungen | 4 Ecken in der Tasche, Tiefe ≥ 25 mm | Tiefe 33–36 mm, Durchdringung < 0,29 mm, im Taschenvolumen; Abbruch bei 30 N | geplant | D-113 |
| Rauschen der Beobachtung | keins | Taschenlage ±5 mm, Griff ±3 mm (je Episode), Kraft σ 3,5 N (je Schritt) | geplant | D-182, D-183 |
| Streuung | Versatz, Kippung, Gier (Befehlszeile) | AutoDR: seitlicher Start, Gier ±5°, Neigung 0–10°, Starthöhe, Reibung; Aufnahme ±5 mm | geplant | D-178..181 |
| Trainingsbedingungen | eine | No-DR, AutoDR, Fixed-DR | geplant | D-178, D-192 |
| Auswertung | Training und Sichtprüfung | mehrere Seeds, eingefrorene Policies, gleiche Testfälle | geplant | `eval_plan.md`, RT-222..225 |
| Parallele Umgebungen | 4096 | 256 | – | RT-206 |
| Bibliothek und Netz | rsl_rl, MLP 128 × 128 | unverändert (nur `experiment_name`) | – | D-116 |

---

## Checkpoint 3: C → D (Robustheit → Annäherung an Factory)

**Letzter Stand C:**
- Code der Endpolicy: `5a6cc8e` (15.09.), Lauf RT-206s1, gewählt in D-190.
- Evaluation: Code `8299485d`, RT-222 bis RT-225. RT-224: AutoDR s1 0,982, No-DR-Mittel 0,262.
- Danach, noch in C: D-192 Fixed-DR-Arm (`3c6dabcd`, RT-231/232). Letzter Commit von C: `e5ee168e` (25.09., D-193).

**Was C gezeigt hat:** Mit OSC, AutoDR und Rauschen auf der Beobachtung fügt die Policy das echte Teil robust. Mit AutoDR schafft sie 98,2 %, ohne Randomisierung nur 26 %.

**Erster Stand D:** `83f6033c` (25.09., Branch `p5_mitAktion`), „Add obs mode force_prevact“. Der erste Lauf ist RT-233.

**Warum:** Eine Kette von Entscheidungen. Am Anfang stand kein gescheiterter Lauf, sondern ein Abgleich von Code und Factory:
1. D-193: Der Beobachtung fehlt die vorige Aktion. Factory hat sie. RT-233 prüft das.
2. D-194: Die Policy sieht Drehungen exakt. Das ist unrealistisch. Neu: Gier-Glaubensfehler ±1,28° pro Episode.
3. D-195: Alle Arme müssen sowieso neu trainiert werden. Darum kommen LSTM, vorige Aktion und ein eigener Critic dazu, wie bei Factory.
4. D-196/D-197: Die Form der Aktion wird wie bei Factory: EMA 0,2, Schritte 0,02 m / 0,097 rad.
5. D-198/D-200: Die Lernrate von rsl_rl hing am Boden (RT-238s1) oder an der Decke (RT-239s1). Darum Wechsel auf rl_games mit der Factory-YAML.
6. D-201 bis D-205: Budget 100 Epochen, Start bis 0,270 m, Streuung der Aufnahme ±0,05 m, Home 0,400 m.

**Was sich am Übergang ändert** (C-Ende → D heute):

| Größe | C | D (vorläufig) | Art | Beleg |
|---|---|---|---|---|
| Beobachtung | 28 | + letzte Aktion (EMA-Zustand) → 34 | geplant | D-193, D-195 |
| Glaubensfehler | Taschenlage, Griff | + Gier der Tasche ±1,28° je Episode | geplant | D-194 |
| Netz des Actors | MLP 128 × 128 | LSTM 2 × 1024 vor MLP [512, 128, 64] | geplant | D-195 |
| Critic | gleiche Beobachtung wie der Actor | eigenes Netz; ohne Rauschen, Taschenlage aus dem wahren Zustand | geplant | D-195, `insertion_env.py:2226-2246` |
| Aktion | 0,02 m / 0,097 rad je Schritt, ohne Glättung | gleiche Schritte, mit Glättung (EMA 0,2), Gier-Band ±12,3° | Reaktion (RT-235s1/RT-236s1) | D-196, D-197 |
| Bibliothek und PPO | rsl_rl, Cartpole-Werte | rl_games mit der Factory-Konfiguration | Reaktion (Lernrate am Boden / an der Decke, RT-238s1/RT-239s1) | D-198, D-200 |
| Streuung der Aufnahme | ±5 mm in der Ebene | ±0,05 m in drei Raumrichtungen, wie PegInsert | geplant | D-203; Factory `fixed_asset_init_pos_noise` |
| Obere Starthöhe | 0,120 m | 0,270 m | Reaktion | D-202 |
| Grundstellung und Zielbereich des Reglers | Home 0,150 m, Box ±0,08 m / z 0,1435 m | Home 0,400 m, Box ±0,100 m / z 0,2915 m | Reaktion | D-204 |
| Trainingsbudget | 1500 Iterationen × 16 Schritte × 256 Envs ≈ 6,1 Mio. Schritte | 100 Epochen × 128 × 256 ≈ 3,3 Mio. Schritte | Reaktion | D-201, D-205 |
| Reward | Aktionsrate −0,0034 | Aktionsrate 0 | – | `insertion_env_cfg.py` |
| Auswertung | rsl_rl-Evaluation (T-nominal, T-final, T-grenze) | für rl_games noch nicht gebaut | – | D-206 |
| Regelung, Takt, Erfolg | OSC 15 Hz, kp 100 / 30 | unverändert | – | – |

**Stand von D heute:** noch offen. RT-244 (No-DR, 5 Seeds) läuft. Die Evaluation für rl_games ist geplant (D-206), aber noch nicht gebaut. Der Tag `stufe-D-factory` kommt danach.
