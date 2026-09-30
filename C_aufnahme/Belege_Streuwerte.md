# Belege_Streuwerte

Je gestreute Größe eine Zeile: der Wert, **die Quelle**, und **was die
Quelle wörtlich sagt**. Zweck: beim Schreiben der Bachelorarbeit den Beleg
zu jeder Zahl in einem Griff finden.

Regeln für diese Datei:

- **Eine Zeile je Größe.** Versätze und Rauschen. Seit 2026-09-12 dazu
  zwei ABGELEITETE Abschnitte: C „Reglerzahlen" (Zahlen, die aus den
  Streuwerten folgen) und D „Bins" (die Kanten, mit denen ein Lauf die
  Streuung berichtet). Beide sind Kopien wie alles hier — die Heimat steht
  daneben.
- **Die Zahl wohnt woanders.** Die Spalte „Wert" ist eine KOPIE, und die
  Spalte daneben sagt, wo die Zahl wirklich steht. Wird die Zahl geändert,
  wird sie dort geändert und hier nachgezogen — nie umgekehrt.
- **Ohne Quelle keine Quelle.** Steht in der Spalte „Quelle" ein `—`, dann
  gibt es keine. Das ist der Normalfall und wird nicht beschönigt. Eine
  Zeile mit `—` darf nie als belegt gelesen werden.
- **„Was die Quelle sagt" ist ein Zitat oder eine Fundstelle**, kein
  Referat. Abschnitt, Tabelle, Abbildung oder `datei:zeile`.
- **Anker ist nicht Herleitung.** Ein Wert, der nur INNERHALB fremder Werte
  liegt, ist angelehnt, nicht begründet. Das steht dann so da.
- Eintragen, sobald ein Wert festgezurrt wird. Nicht am Ende sammeln.

Verwandt, nicht hier doppelt gepflegt:

- `InBachelorErwähnen.md` — Grenzen, die in die Arbeit MÜSSEN (Prosa).
- `docs/reference/literature_check_randomisierungsbereiche_2026-09-09.md` —
  die Prüfung, die diese Fragen gestellt hat.
- `docs/reference/literature_check_reibung_paarung_2026-09-11.md` — die
  Reibwert-Prüfung.
- `DECISIONS.md` D-178 bis D-183 — die Entscheidungen vom 2026-09-11;
  D-185 (2026-09-12) bindet das `noise_cfg` des Rauschmodells, ohne das
  jeder Lauf schon im Konstruktor stirbt.
- `source/insertion/insertion/tasks/direct/insertion/autodr.py` `DR_DIMS` —
  die Heimat der AutoDR-Grenzen im Code.

**Heimat der Zahlen, Stand 2026-09-12:** Die Werte unten sind ENTSCHIEDEN
(D-178 bis D-183) und GEBAUT — `5a7d295` (Scheibe, Starthöhe, Boxen, Gier,
Reibung), `c6e8ed8` (Rauschen, Greif-Versatz), `d0fe39b` (Reibungs-Marker),
`ee97ed3` (D-185, der Stopper). Ab jetzt ist der CODE die Heimat, nicht mehr
der D-Eintrag; die Spalte daneben nennt die Datei. Offen bleibt allein H_min
(Messung, Plan-Schritt 4.2) und damit die Höhen-Bins (Commit C).

**Gebaut heißt offline grün auf dem Laptop. NICHTS davon ist je unter Isaac
gelaufen** — kein Lauf hat mit diesen Zahlen eine Episode gerechnet. Jede
Aussage über Verhalten ist `UNVERIFIED`.

Stand dieser Datei: 2026-09-12.

---

## A. Die AutoDR-Größen — sieben Grenzen

Heimat im Code: `autodr.py` `DR_DIMS`. Die Decken sind GESETZT, nicht
hergeleitet. Regel aus der Klärung vom 2026-09-10: die Decke IST das Ziel.
Unter AutoDR ist die Decke eine Klemme; was die Arbeit als Robustheit
berichtet, ist die Stelle, an der jede Grenze am Ende steht (D-178,
Rationale). Die sieben Grenzen: `lat_r_hi`, `yaw_lo`, `yaw_hi`, `tilt_hi`,
`start_height_hi`, `friction_lo`, `friction_hi`.

| # | Größe | Wert (Kopie) | Quelle | Was die Quelle sagt | Stand |
|---|---|---|---|---|---|
| A1 | Versatz-Radius, Kreisscheibe, einseitig | 0 … 0,030 m; Ziehung `r = R·√u`, `φ = 2π·v` (flächengleich) | **—** für die Zahl. Heimat: `autodr.py` `DR_DIMS` Zeile `lat_r` (gebaut `5a7d295`); D-178 (1), (5) | keine Herleitung. Prüfung 09.09: acht Arbeiten, null Herleitungen. Prüfung 10.09: keine Regel „Versatz = N × Spiel". Anker: ±3 mm in der Hand (Factory/FORGE), ±10 mm (IndustReal), ±20 mm (Factory/AutoMate) | GESETZT (Ziel, Klärung 2026-09-10), nicht hergeleitet. ~~**Vorbedingung:** Netzreparatur~~ **entfaellt 2026-09-12, D-187**: das Loch (0,0845 mm²) liegt unter dem Abtastabstand des SDF-Terms (0,901 mm bei 64000 Punkten), RT-183 und RT-184. Bins (6, 12, 18, 24, 30) mm, D-178 (7) |
| A2 | ~~Versatz y~~ | entfällt | D-178 (2) | — | In A1 aufgegangen. Die Frage „x anders als y" ist zurückgezogen, nicht beantwortet (D-178 (4)) |
| A3 | Gierung | ±0,08726646259971648 rad (5°) | **—** für die Zahl (Nutzer 2026-09-11, D-178 (5)). Heimat: `autodr.py` `DR_DIMS` Zeile `yaw` (gebaut `5a7d295`) | keine. Anker: ±5° (IndustReal/AutoMate), [0, 15°] einseitig (Factory GearMesh), ±10° (Beltran-Hernandez). Proxy-±45° ist per D-110 (5) NICHT erblich | GESETZT, „erweitern bei Bedarf". Bins (1, 2, 3, 4, 5)°, D-178 (7). **Geometrische Grenze (Handrechnung 2026-09-11, Plan-Schritt 0.1):** Das Teil passt nur bis **0,2349° (0,004100 rad)** Gierung in die Öffnung. Modell wie § „Kippgrenzen": Rechteck, scharfe Ecken. Die Querachse bindet: L·sin ψ + B·cos ψ ≤ Öffnung quer. L = 143,50 mm (`insertion_tasks_cfg.py:1492`), B = 90,00 mm (`:1362`), Öffnung 90,5876 × 145,1 mm (`:694-695`). Die Längsachse bindet erst bei 1,0335°. Mit 8 mm Eckenradius an Teil UND Tasche: 0,2644°. Der Eckenradius des Teils ist **unbekannt**. Folge: das ist keine Einführgrenze, sondern die Genauigkeit, die die Policy vor dem Sitz selbst erreichen muss. Sie spricht weder für 5° noch für 10°. Die Nasen binden nicht (Nase 17 mm breit in 47 mm Auskerbung, Tex § Tascheninnenkontur) |
| A4 | Kippbetrag, einseitig | 0 … 0,17453292519943295 rad (10°) | `docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.tex` § „Kippgrenzen (CAD-exakt)" für den Kegel; die Decke selbst: **—** (Ziel, D-178 (5)). Heimat der Decke: `autodr.py` `DR_DIMS` Zeile `tilt` (gebaut `5a7d295`) | 8,52° ist die CAD-**Einführgrenze** am Eintritt („Einfädeln (nur vordere Kante drin)"). Der Regler-Kegel misst gegen die Taschenachse (`insertion_math.py:1472-1485`) | GESETZT. 10° liegt 1,48° ÜBER dem Kegel: bei voller Taschenkippung startet das Werkzeug außerhalb, der Kegel zieht das erste Ziel hinein — angenommen, D-178 (6). Die NVIDIA-Linie würfelt **gar keine** Kippung, die Kategorie hat also kein Vorbild |
| A5 | Starthöhe, einseitig | H_min … 0,120 m. **H_min = 0,020 m, GEMESSEN 2026-09-14** (RT-197a/b: Löser 0 abgeworfen über 4096 Episoden, Reset-Kraft 0,000 N, Ecke Radius 30 mm / Kipp 10° / Gier 5°; 10 und 5 mm FAIL, RT-198b/RT-199b: das Teil wird aus dem Rand gedrückt). Vorher `[TESTWERT]` 0,030 m. | H_min: Messung (Plan-Schritt 4.2), gelaufen 2026-09-14, Urteile in `rt_logs/VERDICTS.md`, Eintrag `docs/decisions_inbox.md` „H_min is MEASURED"; Heimat der Zahl: `insertion_tasks_cfg.RUNG0_START_TIP_ABOVE_ENTRANCE` = 0,020 m über `insertion_env_cfg.start_tip_above_entrance`. Decke: **—** (Ziel, D-179). Heimat der Decke: `autodr.py` `DR_DIMS` Zeile `start_height` (gebaut `5a7d295`); Heimat von H_min: `insertion_env_cfg.start_tip_above_entrance` — seit 2026-09-14 steht dort `RUNG0_START_TIP_ABOVE_ENTRANCE` = 0,020 m, das gemessene H_min | Linien-Vergleich, nur Anker: Factory/AutoMate/FORGE ziehen [0,037; 0,057] m über der Spitze des festen Teils (`factory_tasks_cfg.py:112-113`, FORGE Tab. II) | GESETZT (Decke), H_min GEMESSEN 2026-09-14 = 0,020 m. H_min = kleinste Sprosse mit sauberem Start-Löser UND ohne Kontaktkraft beim Reset (D-179 (4)); dazu seit RT-199c die Spitzenhöhe nach dem Reset gegen den Befehl (die EMA-Kraft klingt auch bei Ausstoß ab). Unter 0 geht es nicht: ein Start IN der Tasche ist verweigert (`insertion_env.py:648-662`), weil der Start-Löser nur die Position kommandiert — `InBachelorErwähnen.md`, 2026-09-10 |
| A5b | Start-Boden (SBC Schritt 0), untere Bandkante zu Laufbeginn | -0,030 m, 5 Stufen bis H_min (0,020 m) | `insertion_env_cfg.start_floor_m`, `start_floor_steps`; Inbox 'The start floor (SBC step 0) is built' (2026-09-14) | **-** fuer die Zahl (Nutzer 2026-09-13/14: -0,030 = das Band von RT-191/193/194, dort Bin -20..-30 mm = 1,0; 5 Stufen = 10 mm je Stufe, eine je Bin-Kante). Form: IndustReal Abschnitt IV.G, `z_low` mit Erfolg angehoben | GESETZT, UNVERIFIED (RT-200) |
| A6 | Reibung | 0,08 … 0,20, Mitte 0,14 (Haft = Gleit) | `docs/reference/literature_check_reibung_paarung_2026-09-11.md`; Heimat: `insertion_tasks_cfg.CONTACT_FRICTION` (Mitte, gebaut `5a7d295`) und `autodr.py` `DR_DIMS` Zeile `friction` (Band); D-181 | DuPont Zytel/Minlon Design Guide Module II, Tab. 39, S. 101: Zytel auf Delrin, ohne Schmierung — statisch 0,13–0,20, dynamisch 0,08–0,11. DuPont Delrin Design Guide, Tab. 9, S. 24: Delrin 500 auf Zytel 101 — statisch 0,10, dynamisch 0,20. (Seiten gelesen vom Recherche-Agenten, nicht hier erneut) | GESETZT. Band = Spanne aller Primärwerte, kein eigenes ±%. **Paarung PA6 (Teil) gegen POM (Aufnahme) ist eine Annahme** (Nutzer 2026-09-11). Die Primärwerte gelten für **PA66**, für PA6 gegen POM gibt es keinen. Die zwei Tabellen widersprechen sich (Konflikt C2). Die Web-Werte 0,20–0,32 stehen auf den zitierten Seiten NICHT (C1) → nicht zitierbar |

---

## B. Streuung, die mitläuft, aber nicht von AutoDR geführt wird

| # | Größe | Wert (Kopie) | Heimat der Zahl | Quelle | Stand |
|---|---|---|---|---|---|
| B1 | Position der Aufnahme, xy | 0,005 m | `insertion_env_cfg.py:636` | D-034 — **in dieser Sitzung nicht gelesen**, Inhalt hier daher nicht behauptet | offen: Eintrag nachtragen, sobald D-034 gelesen ist |
| B2 | Gelenkrauschen beim Reset | 0,0 rad (AUS) | `insertion_env_cfg.py:353` | **—** | Der Kommentar bei `:327` notiert 0,005 rad (±0,29°) für den Fall, dass es wieder eingeschaltet wird. Herkunft dieser Zahl nicht geprüft |
| B3 | Kipp-Azimut | gleichverteilt 0 … 360° | Phase-5-Plan § 1 | Plan § 1: „Die Aufnahme kippt in alle Richtungen." Eigene Festlegung, keine Literatur | gesetzt. Kein AutoDR-Puffer, bewusst |
| B4a | Rauschen der Taschenlage | gleichverteilt [−0,005; 0,005] m je Achse, einmal je Episode, auf `tip_rel` (Kanäle 12:15). Das Feld heißt weiter `_std_m`, trägt aber die Halbbreite (Nutzer 2026-09-15) | `insertion_env_cfg.obs_noise_pocket_pos_std_m`; `obs_noise.py` `UniformNoiseCfg`; p1-thesis `docs/decisions_inbox.md` 2026-09-15, ersetzt den Wert aus D-182 | TacSL, arXiv:2408.06506v2, Anhang C, Tab. V: Socket-Beobachtungsrauschen [−0,005; 0,005] m, gleichverteilt | GESETZT (Nutzer 2026-09-15), offline geprüft, auf dem Trainings-PC NICHT verifiziert. **Lücken:** TacSL nennt den Ziehzeitpunkt nicht, „einmal je Episode“ stammt aus FORGE Anhang A; TacSL-Code nicht gelesen; σ der Gleichverteilung 2,89 mm (eigene Umrechnung) liegt über den 2,5 mm, ab denen FORGE (Abschn. V-A) instabiles Training meldet. **Vorher:** Gauß σ = 0,0025 m nach FORGE Tab. II (D-182, RT-201..203); der FORGE-Code erbt Factorys 0,001 m (`factory_env_cfg.py:46`). Form aus D-111 (3): fester Versatz je Episode. **Weg (Bibliothek):** `DirectRLEnvCfg.observation_noise_model` (`C:\IsaacLab\source\isaaclab\isaaclab\envs\direct_rl_env_cfg.py:172`), gebaut in `direct_rl_env.py:210-213`, angewendet in `:410-415` NACH `_get_observations` nur auf `obs_buf["policy"]` — Reward, Abbruch und Erfolg lesen per Bauart den sauberen Kanal. Bias-Funktion ist die Bibliotheks-`uniform_noise` mit `operation="abs"` (`noise_model.py:68-69`); ohne `abs` summiert der Bias über Resets (`noise_model.py:174`, `noise_cfg.py:29`). **Eigenbau (benannt):** `obs_noise.InsertionObsNoise` legt `_bias` vorab als `(num_envs, OBS_DIM)` an — die Bibliotheksklasse kann keinen Bias je Kanal (`noise_model.py:157`) — baut die Kanalmasken aus `OBS_SLICES` und bindet `noise_cfg = None`, weil `cfg.validate()` das geerbte `MISSING` sonst vor der Physik abbricht (D-185) |
| B4b | Rauschen der Kraft | Gauß σ = 3,5 N, je Schritt, auf `force` (Kanäle 25:28) | `insertion_env_cfg.force_obs_noise_std_n` (gebaut `c6e8ed8`); D-182 | UR5e-Datenblatt (`docs/reference/Datenblaetter/ur5e-datasheet.pdf`), „Force sensing, tool flange/torque sensor", Force x-y-z: „Precision ± 3.5 N", „Accuracy ± 4.0 N" | GESETZT, EIN (Nutzer 2026-09-11). **Annahme:** ±3,5 N ist als σ gelesen; das Datenblatt sagt nicht, ob es ein σ oder ein Maximum ist. Zum Vergleich FORGE Tab. II: „Force Noise: 1 N", je Zeitschritt. **Weg:** derselbe Bibliotheks-Einhängepunkt wie B4a, hier der Je-Schritt-Kanal (`noise_cfg` des inneren `NoiseModelWithAdditiveBiasCfg`). **Offen:** das Rauschen wird NACH der Kraft-EMA addiert, die Policy sieht also volles weißes Rauschen auf einem sonst geglätteten Signal — `HANDOFF-RL.md` § Open, Punkt 000 (4), braucht eine `/decision` |
| B4c | Rauschen des Drehmoments | Gauß σ = **0,0 N·m (AUS)**, je Schritt, auf `torque` (Kanäle 28:31, nur Modus `wrench`; im Modus `force` wird ein σ ≠ 0 abgelehnt) | `insertion_env_cfg.torque_obs_noise_std_nm` (Branch `p5-kraftsensor`, 2026-09-13); D-188 | UR5e-Datenblatt (`docs/reference/Datenblaetter/ur5e-datasheet.pdf`), „Force sensing, tool flange/torque sensor", Torque x-y-z: „Range ± 10.0 Nm", „Precision ± 0.2 Nm", „Accuracy ± 0.3 Nm" (pdftotext 2026-09-13) | AUFGEWEITET für RT-196 (Nutzer 2026-09-13, nach RT-194: die Kraft rauscht mit 3,5 N, das Moment mit 0 — ein sauberer Seitenkanal, den der Sensor nicht liefert): σ = 0,2 N·m als Lauf-Override `env.torque_obs_noise_std_nm=0.2`, dieselbe σ-Lesart wie B4b (Annahme, das Datenblatt sagt nicht σ). Der cfg-Standard bleibt 0,0, weil der Modus `force` ein σ ≠ 0 ablehnt; Heimat der Zahl: `docs/decisions_inbox.md`, Eintrag 'The torque channels get the datasheet precision as their noise' (2026-09-13). Vorher: OFFEN, AUS (Nutzer 2026-09-13, Park-Diagnose). **Nicht** 3,5 N auf N·m übertragen. Weg wie B4b (Je-Schritt-Kanal), Maske aus `insertion_math.noise_masks(mode="wrench")`. Prüfblock A/B laufen mit σ = 0 |
| B5 | Greif-Versatz | gleichverteilt ±0,003 m, kurze Teilachse (x), je Episode, **nur in der Beobachtung** | `insertion_env_cfg.grasp_obs_offset_x_m` (gebaut `c6e8ed8`); D-183 | Factory `held_asset_pos_noise` für den Peg `[0.003, 0.0, 0.003]` (`factory_tasks_cfg.py:123`), gleichverteilt (`factory_env.py:763-769`). FORGE, arXiv:2408.04587v2, Anhang A, Tab. II: „Held: x,y (rel): [−3, 3] mm". Die Achse: Nutzer 2026-09-11 (längs zentriert der Greifer von selbst) | GESETZT. Der Versatz steckt nur in `tip_rel`; Reward, Abbruch und Regler lesen die echte Lage. **Physik-Lücke:** Masse, Schwerpunkt und Hebel bleiben an der Nennlage. Überschreibt D-126 (d), weil der Laufzeit-Weg am geschweißten Teil gemessen tot ist (RT-51/RT-52) |

---

## C. Reglerzahlen — was aus den Streuwerten FOLGT

Keine Streuwerte, sondern die Klemmen, die die Streuung umschließen müssen.
Sie stehen hier, weil sie sich mit jeder Decke oben mitbewegen. Die Formel
ist die SCHEIBEN-Formel und sie hat EINE Heimat: den Box-Block in
`scripts/check_env_wiring.py` (`_reach_xy` / `_reach_z`; D-180). Winkel
10° = Kippdecke,
`h` = Starthöhen-Decke 0,120 m, `r` = Versatz-Radius 0,030 m.

| # | Größe | Wert (Kopie) | Heimat der Zahl | Woher der Wert kommt | Stand |
|---|---|---|---|---|---|
| C1 | Klemmbox x/y, Halbweite `osc_pos_clamp_m` | 0,08 m | `insertion_env_cfg.osc_pos_clamp_m` | Scheibe: Reichweite `r + h*sin(theta)` = 0,030 + 0,120*sin 10° = 0,050838 m, plus EIN Aktionsschritt 0,02 m = 0,070838 m. Luft 9,162 mm | GESETZT (D-180). Der alte Gleichheits-Check „x/y ist Factorys 0,05" ist ersetzt durch eine Obergrenze: Luft höchstens `XY_CLAMP_AIR_MAX_M` = 0,010 m (getippte Konstante in `check_env_wiring.py`). Die Gierung fällt aus der Formel heraus, weil die Scheibe drehsymmetrisch um die Taschenachse ist (D-178, Warrant berichtigt 2026-09-12) |
| C2 | Klemmbox z, Halbweite `osc_pos_clamp_z_m` | 0,1435 m | `insertion_env_cfg.osc_pos_clamp_z_m` | Scheibe: `h*cos(theta) + r*sin(theta)` = 0,123386 m, plus Schritt = 0,143386 m. Abstand der Klemme zur Reichweite: 0,114 mm | GESETZT (D-180). Nach OBEN bleiben zum gemessenen Home-Punkt 0,145879 m (`RT143_HOME_TIP_M` in `check_env_wiring.py`, RT-143) noch 2,379 mm. Ob das reicht, ist NICHT gemessen |
| C3 | Aktions-Schrittlimit `osc_pos_step_limit_m` | 0,02 m | `insertion_env_cfg.osc_pos_step_limit_m` | unverändert. D-166 hat die Zahl als KRAFT-Wert gesetzt (`Lambda*kp*Delta`), nicht als Geometriewert; in C1 und C2 ist sie nur der Summand | unverändert (D-180 bestätigt sie ausdrücklich) |
| C4 | Regler-Kegel `osc_tilt_clamp_rad` | 8,52° | `insertion_env_cfg.osc_tilt_clamp_rad` | CAD-Einführgrenze, `docs/Geometrie/Geometrie_Fuegeteil_Aufnahme.tex` § „Kippgrenzen (CAD-exakt)" | Seit B7 Teil 1 misst `clamp_tilt_to_cone` gegen die TASCHENACHSE, nicht gegen die Welt-Senkrechte (Docstring von `clamp_tilt_to_cone` in `insertion_math.py`): eine gekippte Aufnahme kostet 0° Kegel. Die Kippdecke 10° liegt trotzdem 1,48° darüber, weil der Start-IK nur die Position kommandiert — das Werkzeug startet bei voller Taschenkippung außerhalb des Kegels, und der Kegel zieht das erste Ziel hinein (D-178 (6), angenommen) |

---

## D. Bins — wie ein Lauf die Streuung berichtet

Bin-Kanten sind SPROSSEN, keine hergeleiteten Größen (D-178 (7), Muster
D-173). Lesehilfe, nicht neu erfinden: `_rate_by_bin`
(`insertion_env._rate_by_bin`) macht EINEN Bin je Kante, und der letzte Bin nimmt
alles ab der letzten Kante (Prüfregel L-07).

| # | Tabelle | Kanten (Kopie) | Heimat der Zahl | Stand |
|---|---|---|---|---|
| D1 | Erfolg je Versatz-Radius | (6, 12, 18, 24, 30) mm, fünf Bins je 6 mm | `insertion_env._lateral_bin_edges_mm` | Letzte Kante = Decke `lat_r` 30 mm. Gebunden durch einen Check: `check_autodr.py` „the last LATERAL bin edge is the lat_r ceiling" |
| D2 | Erfolg je Gierung | (1, 2, 3, 4, 5)°, fünf Bins je 1° | `insertion_env._yaw_bin_edges_deg` | Letzte Kante = Decke 5°, ebenfalls durch einen Check gebunden (`check_autodr.py`) |
| D3 | Erfolg je Kippung | (2, 4, 6, 8, 10)°, fünf Bins je 2° | `insertion_env._tilt_bin_edges_deg` | Letzte Kante = Decke `tilt` 10°, durch einen Check gebunden (`check_autodr.py` „the last TILT bin edge is the tilt magnitude ceiling“, beide Seiten aus der Quelle gelesen). GEÄNDERT 2026-09-12 (Nutzer), D-178 (7). Vorher (3, 6, 9, 12, 15)° aus der superseded D-037-Leiter: bei Decke 10° blieb der letzte Bin (ab 12°) LEER und der vierte (9–12°) war nur bis 10° besetzt |
| D4 | Erfolg je Starthöhe | (-20, -10, 0, 10, 20, 40, 60, 80, 100, 120) mm, ZEHN Bins | `insertion_env._start_height_bin_edges_m` | GESETZT 2026-09-14 mit H_min (Inbox „The start floor (SBC step 0) is built“, Punkt 4): fünf 10-mm-Bins unter +20 mm für die Boden-Phase, fünf 20-mm-Bins bis zur Decke 0,120 m. In RT-201/202 (`demo_metrics.json`) sind die fünf unteren Bins leer (Phase dr, Boden steht). `scripts/plot_start_height_bins.py` weist jede Tabelle mit ≠ 6 Bins ab und kann RT-201/202 daher NICHT zeichnen — Skript nicht nachgezogen (offen). Vorher (bis 2026-09-14): (-20, -10, 0, 10, 20, 30) mm, sechs Bins |

---

## E. Was noch keine Zeile hat

Aus § 7 der Prüfung vom 09.09 — die Linie streut das, wir nicht, und es ist
bisher weder abgelehnt noch übernommen:

Aufnahme-Höhe z (±50 mm), Teilmasse (±5 g, die einzige echte
Je-Reset-Streuung der Linie), Regler-Steifigkeiten (≈[401, 797]),
Aktions-EMA, Aktionsschwellen, Kraft-Totzone (ändert sich MITTEN in der
Episode), gewürfelte-und-beobachtete Kraftschwelle, Quaternion-Vorzeichen.

~~Und eine Leseschuld: **Akkaya Anhang B, Tabellen 9–12** — die ADR-eigene
Parameterliste. Bis heute hat sie hier niemand gelesen.~~ **ERLEDIGT, geprüft
2026-09-12:** die Recherche vom 2026-09-10 hat Anhang B vollständig aus dem
arXiv-PDF gelesen — Tabellen 9–12, dazu Tabelle 7, § 9.3.1 und Anhang C.1
(`docs/reference/literature_check_versatz_spiel_verhaeltnis_2026-09-10.md`
§ 6, Quellentabelle `:60`). Sie berichtigt dabei die Prüfung vom 09.09: die
fünf Kategorien sind die Zeilengruppen von Tabelle 9 allein (`:311-316`).
