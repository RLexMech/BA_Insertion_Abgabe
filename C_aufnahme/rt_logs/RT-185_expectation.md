# RT-185 — expectation, written BEFORE the run (2026-09-12)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

## Kontext

Laufplan-Schritt 4 ist seit D-187 auf H_min reduziert. Die Messvorschrift
steht in `C:\Users\PCUser\.claude\plans\distributed-stargazing-deer.md`,
Abschnitt 4.2, Zeile 227 (gelesen heute): eine Leiter `start_tip_above_entrance`
abwärts 30, 20, 10, 5 mm, mit zwei Pflichtkriterien, (a) Löser sauber
(`start_pose_solve_flags_dropped_envs = 0` bei ≥ 4096 Episoden) und (b) keine
Kontaktkraft beim Reset (Kraft in Schritt 0-2 ≈ 0 N). **Dieser Lauf ist NUR
die oberste Sprosse, 30 mm.** 20/10/5 mm bekommen eigene RT-Nummern.

30 mm war vor Phase 5 die feste Starthöhe (D-163, Zeile 5954,
`rt_logs/VERDICTS.md` RT-120/RT-121). Ein Fehlschlag hier wäre ein Befund
über den Phase-5-Umbau (Scheibe statt Rechteck, D-178; Rauschmodell, D-182;
Netzreparatur, offen bis D-187), nicht über die Höhe selbst — deshalb ist
diese Sprosse zuerst dran, nicht die niedrigste.

## NACHTRAG 2026-09-13 — vor dem Start gegen den Code gelesen

1. **Die Befehle gelten unverändert.** `zero_agent.py` hat die drei
   Ecken-Flags (`:102`, `:115`, `:127`), setzt Beobachtungsrauschen und
   Greif-Versatz auf 0 (`:180-182`) und druckt die TARIERTE Kraft
   (`:89-95`). Der Metrik-Dump feuert bei jedem gemeinsamen Reset der 1024
   Envs (`insertion_env.py:3181`, Intervall 250 < 1024), und
   `start_pose_solve_flags_dropped_envs` ist ein Summenzähler
   (`insertion_env.py:2809`). Der Dump nach Schritt 1024 trägt also 4096
   Episoden.
2. **Die Rechnung „Teil A" nennt veraltete Eingaben** (decimation 2,
   60 Hz). Der Startbericht von RT-189s1 zeigt decimation 8, Policy 15 Hz,
   Episode 256 Schritte. Das Ergebnis 256 Schritte je Episode bleibt gleich,
   darum bleibt `--max-steps 1088`.
3. **KONFLIKT beim Kriterium (b), NICHT aufgelöst.** Diese Datei setzt PASS
   bei „deutlich unter 8.0834 N". D-179 (4) sagt „Kraft ≈ 0 N".
   `zero_agent.py:259-261` zählt gegen die L8-Freiluft-Toleranz 1 N
   (`>1N`-Spalte). Drei Schwellen für einen Punkt. Das Urteil nennt alle
   drei Lesungen; der Nutzer entscheidet, welche gilt.
4. **Die Sprossen 20 / 10 / 5 mm laufen mit diesem Skript NICHT.**
   `--start-tip-above-entrance-mm` setzt nur `start_tip_above_entrance`;
   `start_tip_above_entrance_low` bleibt 0.030 m (Cfg-Default). Unter 30 mm
   ist dann low > high, und `insertion_env.py:660-665` wirft `ValueError`.
   Vor der nächsten Sprosse braucht `zero_agent.py` eine Änderung. 30 mm ist
   nicht betroffen (low = high).
6. **NACH RT-185a (Absturz) REPARIERT, 2026-09-13.** `zero_agent.py` setzt
   bei `--fixture-tilt`/`--fixture-yaw` jetzt `fixture_pos_noise_xy`,
   `fixture_tilt_noise_rad` und `fixture_yaw_noise_rad` auf 0 (Muster
   `play.py:270-273`; Nutzer wählte die feste Ecke, Option 1). Benannter
   Verlust: die Aufnahme verschiebt sich nicht mehr um bis zu 5 mm; der Start
   bleibt taschenrelativ. Punkt 4 ist damit auch behoben:
   `--start-tip-above-entrance-mm` setzt die untere Kante gleich. Im Log
   muss zusätzlich die Zeile `[zero_agent] probe angle set:` stehen.
   Offline: `check_env_wiring.py` 324/324 und `--self-test` COUNTER-PROOF
   PASSED (Laptop). Isaac: UNVERIFIED.
5. Der Abschnitt „Was dieser Lauf nicht beantworten kann" unten ist durch den
   Nachtrag vom 2026-09-12 überholt: die Ecke IST abgedeckt.

## Eine Ausführungslücke — GESCHLOSSEN, bevor der Lauf startet

**NACHTRAG 2026-09-12, nach dem Schreiben dieser Datei.** Die unten
beschriebene Lücke ist BEHOBEN. `scripts/zero_agent.py` hat drei neue
Flags bekommen, nach dem Muster der Überschreibungen, die es schon hatte
(`--start-tip-above-entrance-mm`, `--osc-kp-pos`, `--home-pose`):
`--start-lateral-offset-mm`, `--fixture-tilt` und `--fixture-yaw`. Die
beiden Winkelnamen und die Gradeinheit sind von `play.py:48,60` übernommen,
die 15-Grad-Sperre von `play.py:266-268`, und die Felder sind die
DETERMINISTISCHEN `fixture_tilt_rad` (`insertion_env_cfg.py:683`) und
`fixture_yaw_rad` (`:689`) — nicht die Rausch-Felder, die das Cfg mit einem
festen Winkel ohnehin nicht mischen lässt. Die Befehle unten fahren damit
die vom Plan verlangte Ecke. `check_env_wiring.py` läuft danach grün,
Exit 0. Der Text darunter bleibt als Befund stehen, weil er erklärt, warum
die Flags nötig waren.

## Der Befund, der zu den drei Flags geführt hat

Plan 4.2 verlangt die Prüfung bei `start_lateral_offset` 0,030 m, Kipp 10°,
Gier 5° (korrigiert, siehe unten) — zusammen die ungünstigste Ecke der
DR-Box, nicht ihre Mitte. Ich habe `scripts/zero_agent.py` vollständig
gelesen (223 Zeilen): die Argumente sind `--disable_fabric`, `--num_envs`,
`--task`, `--max-steps`, `--start-tip-above-entrance-mm`, `--home-pose`,
`--osc-kp-pos`, `--print-force`, plus die Standard-`AppLauncher`-Flags
(`zero_agent.py:28-101`). Kein `hydra_task_config`, kein
`parse_known_args`, kein `sys.argv`-Umbau — `args_cli = parser.parse_args()`
ist die einzige Argument-Verarbeitung (`zero_agent.py:103`). Es ruft
`parse_env_cfg(args_cli.task, device=..., num_envs=..., use_fabric=...)`
(`zero_agent.py:124-126`); die Signatur von `parse_env_cfg`
(`C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\utils\parse_cfg.py:120-122`)
nimmt NUR `task_name`, `device`, `num_envs`, `use_fabric` — keinen
beliebigen Feld-Override. `scripts/rsl_rl/train.py` dagegen entkoppelt
Hydra-Args ausdrücklich (`train.py:77` `parse_known_args()`, `:83-84`
`sys.argv`-Umbau, `:139` Import, `:215`
`@hydra_task_config(args_cli.task, args_cli.agent)`) — genau die Maschinerie,
die `zero_agent.py` nicht hat.

**Folge:** `zero_agent.py` kann in seinem heutigen Stand `start_lateral_offset`,
`fixture_tilt_rad` und `fixture_yaw_rad` NICHT setzen — nur
`start_tip_above_entrance` (über `--start-tip-above-entrance-mm`). Ohne
Code-Änderung (ausdrücklich außerhalb des Auftrags für diese Datei) läuft
dieser Lauf zwangsläufig bei den Cfg-Standardwerten:
`start_lateral_offset = 0.0` (`insertion_env_cfg.py:484`),
`fixture_tilt_rad = 0.0` (`:683`), `fixture_yaw_rad = 0.0` (`:689`) — der
MITTE der DR-Box, nicht ihrer Ecke. **Das ist eine echte Lücke, keine
Vereinfachung:** ein PASS bei 0/0/0 Seitversatz/Kipp/Gier sagt wenig über
den Löser oder die Reset-Kraft bei 30 mm Versatz und 10° Kippung, dem Fall,
den RT-134/RT-137/D-163 als kritisch benennen. Ich löse das nicht selbst
auf (das wäre eine Code-Entscheidung); ich flagge es hier, VOR dem Lauf,
und schreibe unten, was der Lauf bei dieser Einschränkung tatsächlich
beantwortet und was nicht.

## Ein zweiter, bereits aufgelöster Widerspruch: Gier 10° → 5°

Plan 4.2 (Grill-Sitzung 2026-09-10) schreibt "Gier 10° fest". Das ist
ÜBERHOLT: D-178 (Titel, `DECISIONS.md:7355`, wörtlich "yaw 5 deg"),
entschieden 2026-09-11 — nach dem Plandatum — setzt die Gier-Decke auf
`0.08726646259971648` rad. Der Code trägt exakt diese Zahl mit der
Quellmarke `_D178` bei `autodr.py:431` ("`[TESTWERT] Phase-5 ceiling,
decided not derived -- D-178 (5)`") und `autodr.py:438`
(`DimSpec("yaw", "rad", -0.08726646259971648, 0.08726646259971648, False,
_D178)`). Nachgerechnet: `0.08726646259971648 * 180 / pi = 5.00000...`.
**Nimm 5°, nicht 10°** — wobei diese Zahl wegen der oben beschriebenen
Lücke ohnehin nicht über `zero_agent.py` gesetzt werden kann.

## Marker- und Code-Stand

Laptop und `origin/p5-robustheit`: **`7c49c8dd`** (`git rev-parse HEAD` und
`git rev-parse origin/p5-robustheit`, beide geprüft heute, identisch). Der
Trainings-PC lief RT-184 auf einem älteren Stand und braucht einen Pull.
Weil das Schreiben und Committen dieser Erwartungsdatei den Tip von
`p5-robustheit` verschiebt, ist `7c49c8dd` eine UNTERGRENZE, kein exakter
Zielwert — das echte Tor ist der Marker im Log, nicht diese Zahl.

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
git fetch origin
git checkout p5-robustheit
git pull origin p5-robustheit
git rev-parse HEAD
```

Erwartet: die ausgegebene SHA ist `7c49c8dd...` ODER ein Nachfolge-Commit
auf `p5-robustheit` (Vorfahr-Prüfung `git merge-base --is-ancestor 7c49c8dd
HEAD`, Exit 0). Ein FEHLER (Exit ungleich 0) heißt: der Pull hat den
falschen Branch oder einen älteren Stand gebracht — Lauf ungültig, nichts
danach wird gewertet (Präzedenzfall RT-151b, `rt_logs/VERDICTS.md`,
2026-09-05 14:51:11, UEBERHOLT wegen falschem Marker).

`SCRIPT_MARKER` / `CODE_MARKER`: für `zero_agent.py` selbst ist mir kein
env-Marker-Print bekannt, der die Commit-SHA im Log selbst trägt (anders
als z.B. `-2026-09-12a` bei `check_seated_success.py`, RT-180c). Das
Marker-Tor läuft für diesen Lauf ausschließlich über die gepullte
`git rev-parse HEAD`-Ausgabe oben, VOR dem Start notiert. **Ob
`zero_agent.py` selbst einen Startup-Marker druckt: unbekannt** — im Log
nachsehen und im Log-Kopf festhalten, nicht annehmen.

## Die Rechnung für Teil A (Episodenzahl)

`episode_steps = ceil(episode_length_s / (sim.dt * decimation))`
(`insertion_env_cfg.py:129`, Formel; `:1192` derselbe Ausdruck als Feld,
niemals ein zweites Mal getippt). Werte: `episode_length_s = 256 / 60`
(`:138`), `sim.dt = 1/120` (`:524`, `SimulationCfg(dt=1/120,
render_interval=decimation)`), `decimation = 2` (`:122`).

```
episode_steps = ceil((256/60) / (1/120 * 2)) = ceil((256/60) / (1/60))
              = ceil(256/60 * 60) = ceil(256.0) = 256
```

Bei 1024 Envs (`--num_envs 1024`) braucht Teil A ≥ 4096 Episoden, also
≥ 4 abgeschlossene Episoden je Env (`1024 * 4 = 4096`, Plan-Kommentar
"1024 Envs, ≥ 4 Resets", Zeile 227). `zero_agent.py` hat keine
`rsl_rl`-Rahmenlogik (`init_at_random_ep_len` gehört zu `train.py:725`,
nicht zu diesem Skript), also laufen alle 1024 Envs synchron und
terminieren gemeinsam alle 256 Schritte:

```
Schritte für 4 Episoden je Env = 4 * 256 = 1024
```

`--max-steps 1024` liefert also genau `floor(1024/256) * 1024 = 4096`
abgeschlossene Episoden — die Untergrenze aus dem Plan, ohne Marge. Ich
nehme `--max-steps 1088` (4 volle Zyklen plus 64 Schritte Marge, damit ein
Log, das eine Zeile zu spät beginnt, die vierte Reset-Zeile trotzdem noch
zeigt); die gezählten ABGESCHLOSSENEN Episoden bleiben bei 4096, weil die
fünfte erst bei Schritt 1280 fällt.

`num_steps_per_env = 16` (`agents/rsl_rl_ppo_cfg.py:27`) ist hier
NICHT die maßgebliche Größe — das ist eine rsl_rl-PPO-Iterationsgröße, und
`zero_agent.py` läuft ohne rsl_rl-Runner (kein `train.py`-Import, keine
`OnPolicyRunner`-Instanz irgendwo in der Datei). Sie wird nur zur Kontrolle
genannt: würde man dieselbe Messung stattdessen über `train.py` fahren
(was hier NICHT vorgeschlagen wird, weil eine Policy — und sei sie frisch
initialisiert — keine "zero actions" mehr sind), bräuchte man
`ceil(1024 / 16) = 64` Iterationen für dieselben 1024 Schritte.

## Teil A — Löser-Sauberkeit bei 30 mm

Zweck: `start_pose_solve_flags_dropped_envs` über ≥ 4096 Episoden lesen.
Fährt die vom Plan verlangte Ecke: H = 30 mm, Start-Radius 30 mm,
Kipp 10°, Gier 5° (nicht 10°, siehe den Widerspruch unten). Möglich durch
die drei neuen Flags, siehe den Nachtrag oben.

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
.\scripts\rt_log.ps1 RT-185a python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --start-tip-above-entrance-mm 30 --start-lateral-offset-mm 30 --fixture-tilt 10 --fixture-yaw 5 --max-steps 1088
```

Danach lesen (PowerShell, kein grep/sed/head):

```powershell
Get-Content logs\demo_metrics.json | Select-String "start_pose_solve"
```

## Teil B — Kontaktkraft beim Reset bei 30 mm

Zweck: Kraft in den ersten Schritten nach dem Reset lesen, mit
`--print-force`. Gleiche Ecke wie Teil A.

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
.\scripts\rt_log.ps1 RT-185b python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --start-tip-above-entrance-mm 30 --start-lateral-offset-mm 30 --fixture-tilt 10 --fixture-yaw 5 --max-steps 6 --print-force
```

`--max-steps 6` reicht: die Ausgabe soll nur die ersten Schritte nach dem
einen Reset zeigen, den `env.reset()` vor der Schleife erzwingt
(`zero_agent.py:168`).

**Wonach Kriterium (b) tatsächlich gelesen wird, weil `--print-force` die
Schritte 0-2 NICHT einzeln als "Schritt 0/1/2" druckt:** Der Zähler
`steps` startet bei 0 (`zero_agent.py:170`), wird VOR dem Druck um 1 erhöht
(`env.step()` bei `:177`, `steps += 1` bei `:178`, `print(...)` erst danach
bei `:193-199`) — die erste gedruckte Zeile trägt also `step 1`, nicht
`step 0`. Der Kommentar sagt das selbst: "the first printed line is one
full env step AFTER the reset, never the reset step itself"
(`zero_agent.py:180-182`). Der Reset-Schritt selbst (Isaac Labs "Schritt
0") wird von diesem Instrument NIE einzeln gedruckt. **Kriterium (b) wird
deshalb an den ERSTEN DREI gedruckten Zeilen (`step 1`, `step 2`, `step
3`) abgelesen**, als die dem Reset nächstliegende Näherung, die dieses
Werkzeug liefert — nicht an einer Zeile, die wörtlich "step 0" heißt.

## PASS/FAIL-Punkte

Kein `judge_run` in `zero_agent.py` — das Skript hat keine eingebaute
PASS/FAIL-Logik, nur den Startup-Report und die `--print-force`-Zeilen.
Die folgenden Punkte sind aus dem Plan (4.2, Zeile 227) und den Werten,
die oben mit Datei:Zeile belegt sind, nicht aus einem `judge_run`-Text:

**Teil A — P1 (a) Löser sauber.** PASS: `start_pose_solve_flags_dropped_envs`
liest 0 in `logs\demo_metrics.json`, geschrieben nach ≥ 4096 abgeschlossenen
Episoden (siehe Rechnung oben; `_metrics_dump_every` schreibt periodisch,
`insertion_env.py:1231`, der letzte Schreibvorgang vor Programmende zählt).
FAIL: Wert > 0 — dann als ANTEIL der 4096 Episoden berichten, weil die
Scheibe `r = R·√u` zieht (`insertion_math`, Quelle des Musters: Plan-Kommentar
Zeile 227) und der Rand selten ist; bei Seitversatz 0 (siehe Lücke) ist ein
FAIL hier ohnehin nicht auf die Rand-Fälle zurückführbar, die der Plan
eigentlich prüfen wollte.

**Teil B — P1 (b) keine Kontaktkraft beim Reset.** PASS: `force p50` und
`max` in den Zeilen `step 1`, `step 2`, `step 3` liegen deutlich unter dem
Eigengewicht des Werkzeugs, 8.0834 N (`rt_logs/VERDICTS.md`, RT-180c,
2026-09-12, "roh 8.0834 N = Werkzeuggewicht"). FAIL: Kraft in dieser
Größenordnung (grob 8 N oder mehr) — das Teil steht beim Reset schon auf
dem Block (RT-134, RT-137, D-163).

## Pins

**DIE DREI ÜBERSCHREIBUNGSZEILEN MÜSSEN IM LOG STEHEN** (2026-09-12, neu).
Fehlt eine, lief der alte Stand des Skripts und der Lauf mass die Mitte
statt der Ecke:

```
[zero_agent] start_lateral_offset overridden: 30.000 mm start-disk radius (THIS RUN ONLY).
[zero_agent] fixture_tilt_rad overridden: 10.0000 deg = 0.174533 rad (static probe tilt, THIS RUN ONLY).
[zero_agent] fixture_yaw_rad overridden: 5.0000 deg = 0.087266 rad (static probe yaw, THIS RUN ONLY).
```


Keine `params/env.yaml`-Pins für diesen Lauf: `zero_agent.py` ruft
`train.py` nicht auf und schreibt keine `params/env.yaml` (die Datei
entsteht in `train.py`, nicht im Zero-Agent-Pfad — kein Aufruf von
`dump_yaml` o.ä. in `zero_agent.py` gefunden). Die einzigen Werte, die
dieser Lauf schreibt, sind die `demo_metrics.json`-Felder oben und die
`--print-force`-Konsolenzeilen; keine JSON-Pins außerhalb von
`start_pose_solve_flags_dropped_envs` werden hier geprüft.

`start_tip_above_entrance_mm` (Konsolenzeile aus
`zero_agent.py:157-159`): erwartet `+30.0 mm`, aus dem übergebenen
`--start-tip-above-entrance-mm 30`.

## Was dieser Lauf nicht beantworten kann

* **Nichts über die Ecke der DR-Box.** Wegen der oben beschriebenen Lücke
  in `zero_agent.py` läuft dieser Lauf bei Seitversatz 0, Kipp 0°, Gier 0°
  — nicht bei den vom Plan verlangten 0,030 m / 10° / 5°. Ein PASS hier
  ist KEIN Beleg, dass der Löser oder die Reset-Kraft auch in der Ecke
  sauber sind; das bleibt unbeantwortet, bis entweder `zero_agent.py`
  einen Override bekommt oder ein anderer Weg (z. B. `train.py` mit
  Hydra-Overridern, aber dann keine "zero actions" mehr) gewählt wird —
  eine Entscheidung, die diese Datei nicht trifft.
* **Nichts über die Sprossen 20, 10 und 5 mm.** Eigene RT-Nummern.
* **Nichts über Erfolgsrate oder Lernen.** `zero_agent.py` treibt keine
  Policy; es gibt keinen Reward-Term, der hier zählt.
* **Nichts über das Netzloch.** Das ist mit D-187 abgeschlossen.
* **Ob `zero_agent.py` selbst einen Marker im Log druckt:** unbekannt,
  siehe oben — im Log nachsehen.
