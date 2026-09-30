# RT-184 — expectation, written BEFORE the run (2026-09-12)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

## Kontext

D-153 (`DECISIONS.md:5122-5136`) hält zwei Dinge fest: die Aufnahme-Mesh hat
ein 1.5-mm-Loch aus drei ungepaarten Kanten (`ZERO-EDGE COUNT 3`), heute
erneut belegt durch RT-183 (`rt_logs/VERDICTS.md`, letzte Zeile,
2026-09-12 14:20:25, git `379aab2`, PASS/exit 1 wie von D-153 gewollt) an
x 35.9863..37.4512 mm, y 79.6347..79.7500 mm, z 0.0000 mm. D-153 sagt
wörtlich: „What is STILL not measured is whether reaching the hole does any
harm; the instrument for that is `stage1_lateral_y_mm.over_reach`." Genau
diese Lücke schließt dieser Lauf — nicht mit dem Episodenzähler, sondern mit
dem SDF direkt an derselben Geometrie.

Der Erfolgstest selbst ist laut D-153 bewiesen sauber (die seated Pose liegt
weit von der Kante). Was NICHT bewiesen ist: ob der SAPU-Term beim Anflug
(Stufe 1, oberhalb der Eingangsebene) über das Loch hinweg irgendeinen
Sprung oder Vorzeichenwechsel zeigt. Das ist die eine Frage dieses Laufs.

## Die Geometrie, heute nachgerechnet

- `HOLE_REACH_OFFSET_Y = POCKET_HOLE_MIN_Y - PART_BBOX_M[1] / 2.0`
  (`insertion_tasks_cfg.py:1521`), mit `POCKET_HOLE_MIN_Y = 0.0796347`
  (`insertion_tasks_cfg.py:1517`, M: RT-94) → 79.6347 − 71.75 = 7.8847 mm
  (Docstring `insertion_tasks_cfg.py:1522` nennt exakt diesen Wert; die
  71.75 mm folgen aus `PART_BBOX_M[1] / 2.0` und stehen dort nicht
  nochmals ausgeschrieben).
- Stufe 1 erlaubt laut Docstring `insertion_tasks_cfg.py:1526`
  `POCKET_STAGE1_Y_RANGE[1] - PART_BBOX_M[1] / 2 = 8.7000 mm`.
- Belastungsfenster: 8.7000 − 7.8847 = **0.8153 mm** breit
  (D-153, `DECISIONS.md:5134`, identisch).
- `STAGE1_DEPTH = 0.015` m (`insertion_tasks_cfg.py:674`) — Stufe 1 reicht
  in z von 0 bis 15 mm ÜBER der Eingangsebene. Das Loch liegt bei
  z = 0.0000 mm, an der Sohle von Stufe 1.

**Warum nicht `over_reach`.** `stage1_lateral_y_mm` (`insertion_env.py:3640`)
und darin der Schlüssel `over_reach` (`insertion_env.py:3409`,
`"over_reach": sum(1 for v in values if v >= reach)`) zählt Episoden mit
`|y| >= HOLE_REACH_OFFSET_Y`, solange `_max_stage1_lat_y`
(`insertion_env.py:938`, akkumuliert `insertion_env.py:1960-1961`) im
z-Band 0..15 mm lag. Das Tor ist NUR das z-Band, nicht die Geometrie von
Stufe 1 — bei einem Startradius von 30 mm (`lat_r` in `autodr.py`, DR_DIMS,
nicht neu nachgemessen hier) liegt der Anfangsversatz oft schon über
7.8847 mm, sodass der Zähler im ersten Schritt jeder Episode feuert. Er
belegt, dass die Belastung eintritt — nicht, dass sie schadet. Deshalb
misst dieser Lauf den SDF direkt an einem festen Höhen-/Seitenraster,
statt Episoden zu zählen.

## Marker- und Code-Stand

- Laptop: `origin/p5-robustheit` = **`09add92c3645e79f5f03ca00a3a7f29a0521f309`**
  (`git rev-parse HEAD` und `git rev-parse origin/p5-robustheit`, beide heute
  geprüft, identisch).
- Trainings-PC lief RT-183 auf `379aab2` — braucht einen Pull.
- `SCRIPT_MARKER = "check_seated_success-2026-09-12a"`
  (`scripts/check_seated_success.py:97`).
- `CODE_MARKER = "insertion-osc-2026-09-03a"` (`insertion_env.py:73`).
- Fehlt einer der beiden Marker im Log oder weicht er ab: der Lauf ist
  ungültig, nichts danach wird gewertet (Präzedenzfall RT-151b,
  `rt_logs/VERDICTS.md:219-220`, verworfen als UEBERHOLT wegen Marker
  `-2026-09-03d` statt des geforderten `-2026-09-05`).

**Pull- und Verify-Befehl, Trainings-PC (PowerShell), einfügefertig:**

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
git pull origin p5-robustheit
git rev-parse HEAD
```

Erwartete SHA nach dem Pull: **mindestens `09add92c`**; der Tip ist der
Commit dieser Datei, `bd97f276`, oder neuer. Zwischen `09add92c` und dem Tip
ändert sich NUR Dokumentation — `git diff --name-only 09add92c..bd97f276`
listet allein diese Erwartungsdatei, und `scripts/check_seated_success.py`
ist seit `5e8c3aa` unverändert. Eine spätere Doku-Zeile darf den Lauf also
nicht blockieren. **Das echte Tor ist der Marker in der Log-Zeile**, siehe
oben: liest er nicht `check_seated_success-2026-09-12a`, ist der Lauf
ungültig.

## The one hypothesis

**Der SDF verhält sich über das Loch hinweg stetig: `sdf_mean_outside_mm`
zeigt bei jeder gemessenen Höhe (0, 5, 10, 15 mm) über die Seitenversätze
0, 4, 7, 7.8847, 8.2, 8.7 mm einen glatten Verlauf, ohne Sprung und ohne
Vorzeichenwechsel beim Überschreiten von 7.8847 mm — insbesondere bei Höhe
0 mm, wo das Loch selbst liegt.** Trifft das zu, schadet das Loch dem
SAPU-Faktor beim Anflug nicht, und die Netzreparatur in Laufplan-Schritt
4.1 ist NICHT belegt.

**Was ein Befund wäre:** ein Sprung oder Vorzeichenwechsel in
`sdf_mean_outside_mm` genau zwischen 7.0 und 8.7 mm, der bei den Höhen 0
und 4 mm (den sauberen, weit vom Fenster entfernten Stützstellen)
fehlt. Vergleichsmaßstab ist die Schrittweite zwischen den sauberen
Stützstellen 0, 4 und 7 mm: ein Sprung im Fenster 7.8847..8.7 mm muss
deutlich größer sein als die dortige Änderung, sonst ist er kein Befund,
sondern gewöhnliches Kurvenrauschen.

## Command (Trainings-PC, PowerShell)

Trainings-PC-Wurzel:
`C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`

```
.\scripts\rt_log.ps1 RT-184 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 0,5,10,15 --curve-lateral-y-mm 0,4,7,7.8847,8.2,8.7 --out rt_logs/RT-184_curve.json
```

Alle sechs Seitenversätze (0..8.7 mm) und alle vier Höhen (0..15 mm) liegen
innerhalb der halben `osc_pos_clamp_m`-Box (0.08 m = 80 mm,
`insertion_env_cfg.py:252`) — die Box zieht hier nichts nach innen; die
RT-151b-Falle (Klemmbox kleiner als der kommandierte Versatz,
`rt_logs/VERDICTS.md:221`) greift bei diesem Raster nicht.

`--curve-settle-steps` wird nicht übergeben, Default 4
(`scripts/check_seated_success.py:1415`); `--solve-tol-mm` wird nicht
übergeben, Default 0.05 mm (`scripts/check_seated_success.py:1442`). Zwei
mit vier Höhen mal sechs Seiten sind 24 Punkte × 4 Settle-Schritte = 96
Schritte, deutlich unter der 256-Schritt-Episode
(`scripts/check_seated_success.py:1420` nennt dieselbe Rechnung für die
eigene Standardbelegung).

## `--reward-curve` ist eine MESSUNG, kein `judge_run`-Test

`scripts/check_seated_success.py:69`: „THE REWARD CURVE IS A FOURTH MODE
AND NOT A TEST." Dieser Modus ruft `judge_run`
(`scripts/check_seated_success.py:178`) NICHT auf — die dortigen P1..P9-
Punkte gehören zu den anderen drei Modi (seated / `--shallow` /
`--lateral`) und sind für RT-184 nicht einschlägig. Statt PASS/FAIL-Punkten
aus `judge_run` gibt es die Vorbedingungen, die die MESSUNG selbst gültig
machen — wörtlich aus dem SUMMARY-Block und dem `metrics`-Dict
(`scripts/check_seated_success.py:1759-1851`), in der Reihenfolge, in der
das Skript sie berechnet:

1. **`all_solves_converged`** (`solve_converged` je Zeile,
   `scripts/check_seated_success.py:1728`) — PASS: `True` auf JEDER Zeile
   und im SUMMARY-Ende. FAIL: eine Zeile mit `NOT CONVERGED` — die IK hat
   die kommandierte Pose nicht erreicht, die Zeile misst nichts.
2. **`all_poses_held`** (`pose_held` je Zeile,
   `scripts/check_seated_success.py:1693,1725`) — PASS: `pose_drift_mm`
   auf JEDER Zeile ≤ 0.05 mm (`--solve-tol-mm`-Default) und im
   SUMMARY-Ende `all_poses_held True`. FAIL: eine Zeile mit
   `POSE NOT HELD` — genau die RT-151b-Falle
   (`rt_logs/VERDICTS.md:221-238`): die Settle-Schritte haben die Pose
   verschoben, die gemessene Zeile ist nicht die kommandierte.
3. **`reset_during_sweep` / `sweep_end_reason`**
   (`scripts/check_seated_success.py:1804-1805`) — PASS: `False` /
   `None`, kein Reset unterbricht das Raster. FAIL: `True` mit einem
   Grund ≠ `success_termination` — jede Zeile NACH dem Reset läse dann
   die Home-Pose, nicht das Raster.
4. **`verdict_ok`** (`scripts/check_seated_success.py:1759-1760,1806`,
   gedruckt als `VERDICT: REWARD_CURVE_MEASURED` bzw.
   `REWARD_CURVE_UNUSABLE`, `:1849-1850`) — die UND-Verknüpfung der drei
   obigen. PASS: `REWARD_CURVE_MEASURED`, exit 0. FAIL:
   `REWARD_CURVE_UNUSABLE`, exit 1 — dann zählt keine Zeile der
   Hypothese, unabhängig vom SDF-Verlauf.

Diese vier Punkte sind Vorbedingungen der Messung, keine Aussage über das
Loch selbst — die Hypothese wird erst DANACH an `sdf_mean_outside_mm`
geprüft.

## Pins, aus `rt_logs/RT-184_curve.json` gelesen

- `marker` = `"check_seated_success-2026-09-12a"`
  (`scripts/check_seated_success.py:1763`).
- `fixture_pos_noise_xy_m` = `0.0` (`scripts/check_seated_success.py:1775`,
  gesetzt an `:1945`, unbedingt für alle Modi — Kommentar
  `:1938-1940`: „It applies to all three modes").
- `obs_noise_pocket_pos_std_m` = `0.0` (`scripts/check_seated_success.py:1776`,
  gesetzt an `:1956`).
- `force_obs_noise_std_n` = `0.0` (`scripts/check_seated_success.py:1777`,
  gesetzt an `:1957`).
- `grasp_obs_offset_x_m` = `0.0` (`scripts/check_seated_success.py:1778`,
  gesetzt an `:1958`). Die vier Streufelder liegen bei 0.0, obwohl die
  cfg-Defaults es NICHT sind (`fixture_pos_noise_xy = 0.005`,
  `insertion_env_cfg.py:653`; `obs_noise_pocket_pos_std_m = 0.0025`,
  `:1065`; `force_obs_noise_std_n = 3.5`, `:1066`;
  `grasp_obs_offset_x_m = 0.003`, `:1074`) — das Skript zwingt sie für
  JEDEN seiner drei Modi (inkl. `--reward-curve`) unbedingt auf 0, laut
  eigenem Kommentar an derselben Stelle.
- `all_solves_converged` = `true`, `all_poses_held` = `true`,
  `reset_during_sweep` = `false`, `verdict_ok` = `true` (siehe oben).
- Je Zeile in `curve` (`scripts/check_seated_success.py:1714-1741`):
  `sdf_mean_outside_mm` (Liste, ein Wert je Env) und `interpen_max_mm`
  (dieselbe Form) — beide Schlüssel stehen wörtlich im Zeilen-Dump
  (`:1732-1733`).
- `curve_heights_mm` = `[0.0, 5.0, 10.0, 15.0]`, `curve_lateral_y_mm` =
  `[0.0, 4.0, 7.0, 7.8847, 8.2, 8.7]` (`scripts/check_seated_success.py:1780-1781`,
  Echo der `--curve-heights-mm` / `--curve-lateral-y-mm`-Eingabe).

## Was dieser Lauf nicht beantworten kann

- Nichts über eine Policy und nichts über Training — es sind
  Teleport-Posen (`_solve_to`, reine DLS-IK,
  `scripts/check_seated_success.py:1708` laut Dokstring-Referenz).
- Nichts darüber, ob Warp Kanten nach Index oder nach Position paart —
  D-153 nennt das offen (`DECISIONS.md:5153-5156`: 903 authored,
  9 nach Weld, davon 3 das hier gemessene Dreieck).
- Nichts über den Bereich unter der Eingangsebene (Stufe 2). Dort hält
  `PLAY_Y / 2` das Teil auf `0.8 mm` (Docstring
  `insertion_tasks_cfg.py:1525`: „stage 2 allows PLAY_Y / 2 = 0.8 mm").
  Dieser Lauf sweept nur Höhen ≥ 0 (oberhalb der Eingangsebene, Stufe 1),
  wie es die eigene Gitterregel verlangt
  (`scripts/check_seated_success.py:626-633`: seitliche Versätze nur bei
  Höhen ≥ 0, sonst Kraft-Abbruch an der Stufe-2-Wand).
- Nichts über `over_reach` selbst — der Zähler wird hier bewusst NICHT
  gelesen; sein Torzustand (nur z-Band, nicht Stufe-1-Geometrie) bleibt
  unangetastet und ungeprüft in diesem Lauf.
- Nichts über einen Sprung außerhalb des gemessenen Rasters (z. B.
  zwischen 15 mm und höher, oder feiner als 0.1 mm Auflösung zwischen
  7.8847 und 8.2 mm) — das Raster ist ein endliches Sample, kein
  stetiger Scan.
