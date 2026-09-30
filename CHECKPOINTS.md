# Checkpoints – wo jede Stufe endet und wo die nächste beginnt

Diese Datei beschreibt die 3 Übergänge A→B, B→C und C→D.
Für jeden Übergang steht hier: der letzte Stand der alten Stufe, der erste Stand der neuen Stufe, der Grund und was sich geändert hat.
Die vollen Werte jeder Stufe stehen in [STAENDE.md](STAENDE.md).

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

| | A (Ende) | B (Beginn) |
|---|---|---|
| Teil | Zylinder Ø25/Ø28 mm in Bohrung Ø30 mm | Quader 30×30×50 mm in Tasche 32×32×35 mm (1,0 mm Spiel je Achse) |
| Beobachtung | 19 | 21 (+ cos 4φ, sin 4φ) |
| Reward | 6 Terme | + Gier-Term (Gewicht 2,0) |
| Startstreuung | Gelenke ±0,01 rad | + Handgelenk 3 ±45° |
| Regler, Aktion, PPO | Gelenk-Deltas, 0,02 rad, rsl_rl | unverändert |

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

| | B (Ende) | C (Beginn) |
|---|---|---|
| Roboter | UR10e | UR5e |
| Teil / Aufnahme | Quader, Tasche aus dem Generator | echtes Teil 143,5 × 90 mm, Aufnahme aus CAD. Spiel in der Sim 0,59 mm quer / 1,60 mm längs, in echt 0,4 / 0,9 mm (D-087, D-088, D-121) |
| Beobachtung | 25 (cos 4φ) | 28 (cos φ, + 3 Kraftkanäle, D-107, D-114) |
| Reward | 7 Terme der Vorstudie | neu hergeleitet: SDF-Kerne, Boni, Kraftstrafe (D-109) |
| Streuung | Versatz, Neigung, Gier an | alles 0 |
| Regler, Aktion | Gelenk-Deltas, 0,02 rad, 60 Hz | unverändert (D-108) |
| PPO | rsl_rl, Cartpole-Vorlage | unverändert (nur `experiment_name`) |

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

| | C (Ende) | D (vorläufig) |
|---|---|---|
| Aktion | 6-D, Clip ±1, kein EMA | 6-D, Clip ±1, **EMA 0,2**, Gier-Band ±12,30° |
| Beobachtung | 28, keine vorige Aktion | **34** (+ vorige Aktion = EMA-Zustand), Gier-Glaubensfehler ±1,28° |
| Netz | MLP 128×128, Actor = Critic | **LSTM 2×1024** + MLP [512,128,64], eigener Critic mit wahrer Taschenlage |
| PPO | rsl_rl, Cartpole-Vorlage | **rl_games**, Factory-YAML (100 Epochen, Minibatch 1024) |
| Streuung | Start bis 0,120 m, Aufnahme ±5 mm | Start bis 0,270 m, Aufnahme ±0,05 m in x/y/z, Home 0,400 m |
| Reward | action_rate 0,0034 | action_rate 0 |
| Regler, Erfolg | OSC 15 Hz, kp 100/30 | unverändert |

**Stand von D heute:** noch offen. RT-244 (No-DR, 5 Seeds) läuft. Die Evaluation für rl_games ist geplant (D-206), aber noch nicht gebaut. Der Tag `stufe-D-factory` kommt danach.
