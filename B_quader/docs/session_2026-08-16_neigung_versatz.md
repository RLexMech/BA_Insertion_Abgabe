# Sessionbericht 2026-08-16 — Geneigte, versetzte und gedrehte Tasche

Stand nach der Arbeitssession vom 16.08.2026. Dieser Bericht fasst zusammen,
was vor der Session nicht funktionierte, was jetzt funktioniert, welche
Code-Anpassungen dafür notwendig waren und welche Erkenntnisse daraus für die
eigentliche Zielaufgabe folgen. Alle Zahlen stammen aus Messläufen auf der
Trainingsmaschine (RTX 3080) und wurden aus `demo_metrics.json`-Dateien bzw.
Konsolenausgaben gelesen, nicht geschätzt.

---

## 1. Ausgangslage (Beginn der Session)

- Vorhandene Politik: `2026-08-06_21-19-45_s30b32rxy50`, Checkpoint
  `model_3950.pt`, Beobachtungsraum **21 Kanäle**.
- Diese Politik setzt den quadratischen Zapfen (30×30 mm) zuverlässig in die
  gerade, unverkippte Tasche (32×32 mm, 1,0 mm Spiel pro Achse) ein:
  99,65 % im Training, 100 % im Replay, auch bei ±5 cm Taschenversatz.
- Der Messcode für geneigte Taschen (D-036) war geschrieben, aber noch nie in
  Isaac ausgeführt (Status UNVERIFIED).
- **Nicht vorhanden:** jede Fähigkeit, eine geneigte Tasche zu treffen.

## 2. Was vorher nicht ging und jetzt läuft

| Fähigkeit | Vorher | Nachher (gemessen) |
|---|---|---|
| Gerade Tasche, kein Versatz | 100 % | 100 % (Regressionstest bestanden) |
| Tasche 5° geneigt | 50 % | ≥ 98 % (Neigungs-Bin 3–6°) |
| Tasche 10° geneigt | Einsetzen misslingt | 97,9 % (Bin 9–12°), 99,3 % gesamt |
| Neigung 0–10° **und** Versatz ±2 cm kombiniert | nie trainiert | 99 % (Stop-Kriterium erreicht) |
| Zusätzlich Taschendrehung ±45° um die Tisch-z-Achse (D-038) | durch Code-Sperre abgelehnt | 99,2 % über 5 461 Episoden; Replay 99,6 % über 703 Episoden |

Die Kombinationsstufe mit ±5 cm Versatz wurde bewusst zurückgestellt; die
±2-cm-Stufe genügte für die weiteren Schritte.

## 3. Kernbefund: Die Politik war blind für die Neigung

Der wichtigste Befund der Session, und der Grund für die zentrale
Code-Anpassung:

- Die Baseline-Messung (D-036, erstmals heute auf der Trainingsmaschine
  ausgeführt) ergab: 0° → 100 % Erfolg (mittlere Maximaltiefe 29,3 mm),
  5° → 50 % Erfolg (17,9 mm). Die Messkette selbst wurde per Rücklesen der
  Prim-Pose verifiziert (`measured tilt about y: +5.000 deg ... (OK)`,
  Fehler 0,0000°). Der Einbruch ist also eine Eigenschaft der Politik,
  kein Messartefakt.
- Ursache: **partielle Beobachtbarkeit.** Alle 21 Beobachtungskanäle waren
  entweder Roboterzustand oder wurden *relativ zur Tasche* gemessen
  (Spitze-zum-Eingang im Taschen-Koordinatensystem). Zwei Episoden mit
  unterschiedlich geneigter Tasche erzeugten dadurch **identische
  Beobachtungen**, verlangten aber unterschiedliche Gelenkbewegungen. Eine
  solche Aufgabe kann keine Politik lernen, unabhängig vom Training.

**Regel daraus:** Bevor auf eine neue Variation trainiert wird, ist zu prüfen,
ob die Variation in der Beobachtung überhaupt sichtbar ist.

## 4. Code-Anpassung 1: Beobachtungsraum 21 → 25 Kanäle (D-037)

**Klarstellung, da für die Zielaufgabe relevant:** Es kamen **4 Kanäle** hinzu,
nicht 6. Der Beobachtungsraum hat jetzt **25 Kanäle**, nicht 26.

- Zwischenzeitlich stand ein 6-Kanal-Vorschlag im Raum (6D-Rotations-
  Repräsentation nach Zhou et al., CVPR 2019). Die Literaturrecherche ergab
  jedoch: Die 6D-Repräsentation betrifft Netz-**Ausgänge** (Regression von
  Rotationen). Für Netz-**Eingänge** ist das Quaternion der Praxisstandard.
- Direkter Präzedenzfall: GenPiH (arXiv 2504.04148) — gleicher Roboter
  (UR10e), gleiches Framework (Isaac Lab), gleicher Algorithmus (PPO) — gibt
  die Lochorientierung als Quaternion in die Beobachtung und trainiert damit
  Einsetzen bis Roll/Pitch/Yaw ±25°.
- Umsetzung: Das Taschen-Quaternion (wxyz) wird als Kanäle 21:25 angehängt.
  Kein bestehender Kanalindex verschiebt sich. Das Vorzeichen ist auf w ≥ 0
  kanonisiert, damit jede Orientierung genau eine Kodierung hat (q und −q
  beschreiben dieselbe Drehung).
- **Wichtig für die Zielaufgabe:** Dieses Quaternion trägt bereits *jede*
  Taschenorientierung — Neigung *und* Drehung um die Tisch-z-Achse (Yaw).
  Für die Yaw-Erweiterung (Abschnitt 7) wurden **keine weiteren Kanäle**
  benötigt; es bleibt bei 25. Diese Vorhersage wurde noch am selben Tag durch
  das D-038-Ergebnis bestätigt.

## 5. Code-Anpassung 2: Checkpoint-Verbreiterung für den Warmstart

Problem: Die alte Politik hat eine Eingangsschicht für 21 Kanäle. Neu von Null
trainieren würde das gelöste Einsetzen wegwerfen.

Lösung: `scripts/expand_checkpoint_obs.py` verbreitert einen
rsl_rl-Checkpoint von 21 auf 25 Eingänge:

- Gewichtsmatrizen der ersten Schicht: 4 Nullspalten anhängen. Die neuen
  Kanäle sind damit anfangs wirkungslos; das Verhalten der Politik ist nach
  der Operation bitgenau unverändert (verifiziert: 100 % bei 0°, ~50 % bei 5°,
  identisch zur 21-Kanal-Baseline).
- Normalisierer (`EmpiricalNormalization`): neue Kanäle mit Mittelwert 0,
  Varianz 1, Standardabweichung 1 auffüllen.
- Adam-Optimiererzustand: geleert (Momente passen nicht mehr zur neuen Form).

**Gelernte Lektion (PROBLEMS.md):** Der erste Wurf sortierte die Tensoren nach
*Form* und brach korrekt ab, weil `actor_obs_normalizer._mean` die Form (1, 21)
hat — von einer Gewichtsmatrix mit einem Ausgang nicht unterscheidbar. Eine
Null-Auffüllung des Normalisierer-Mittelwerts hätte alle 25 Kanäle falsch
skaliert. Fix: Zuordnung nach *Name* (führenden Unterstrich abstreifen);
unbekannte Tensoren mit alter Breite führen zum harten Abbruch statt zu einer
stillen Fehlbehandlung.

## 6. Training: Leiter mit Warmstart — und ein lehrreicher Kollaps

Trainiert wurde nie von Null, sondern immer per Warmstart vom letzten
funktionierenden Stand (Curriculum-Leiter, Muster aus D-034). Neigung pro
Episode: Betrag gleichverteilt in [0, max], Richtung gleichverteilt über 360°.
Stop-Kriterium: mittlere Erfolgsrate ≥ 99 % (automatischer Stopp in train.py,
neu eingebaut: `--stop-at-success-rate`).

| Stufe | Ergebnis |
|---|---|
| 0–5° (aus 21-Kanal-Politik, verbreitert) | 99,2 % nach 18 914 Episoden |
| 5° → **15°** (Sprung ×3) | **Kollaps:** A/B-Test bei 5° ergab alt 99 %, neu **0 %** |
| 5° → 10° (aus der 5°-Politik) | 99,3 % nach 2 081 Episoden; Bins: 0–3° 100 %, 3–6° 99,7 %, 6–9° 98,7 %, 9–12° 97,9 % |
| 10° + Versatz ±2 cm kombiniert | 99 % (Stop-Kriterium erreicht) |

**Lektion:** Die Schrittweite der Leiter entscheidet. Der Sprung von 5° auf 15°
(Faktor 3) hat eine funktionierende Politik vollständig zerstört; der Sprung
von 5° auf 10° (Faktor 2) kostete nur 2 081 Episoden. Als Ziel wurden 10°
festgelegt; 15° ist für die Aufgabe unnötig.

Für die Kombinationsstufe war **keine Code-Änderung** notwendig: Der
Reset-Pfad setzt Versatz und Orientierung aus demselben Posen-Schreibbefehl
zusammen, und die Beobachtung enthält beides (Versatz über Kanäle 12:15,
Orientierung über 21:25). Auch keine erneute Checkpoint-Verbreiterung — die
Kanalzahl blieb 25.

## 7. Drehung um die Tisch-z-Achse (Yaw, D-038)

Letzte Generalisierungsachse vor der Zielaufgabe. Zwei Punkte machten sie
billiger als die Neigung:

- **±45° ist der volle Bereich.** Die quadratische Tasche hat vierzählige
  Symmetrie (C4): Alle 90° wiederholt sich die Pose. Größere Winkel wären
  stille Duplikate kleinerer; play.py lehnt sie deshalb ab.
- **Keine neuen Beobachtungskanäle.** Das Taschen-Quaternion (Kanäle 21:25)
  trägt den Yaw bereits — die Vorhersage aus Abschnitt 4 hat sich bestätigt,
  es bleibt bei 25 Kanälen.

Code-Anpassungen (Commit `439c74a`): Die Sperre aus der D-036/D-037-Zeit wurde
ersetzt; Yaw aktiviert jetzt denselben Taschen-Koordinaten-Pfad wie die
Neigung. Die Spawn-Orientierung komponiert R_z(Yaw)·R_y(Neigung) in derselben
Reihenfolge wie der randomisierte Reset. play.py erhielt `--fixture-yaw`
(kombinierbar mit Versatz- und Neigungs-Probe). Die Neigungs-Bin-Statistik
misst jetzt den Polarwinkel der Taschen-z-Achse (acos(r22)) statt des
Quaternion-Betrags — sonst würde eine gedrehte Tasche fälschlich als geneigt
verbucht.

**Ergebnis (verifiziert auf der Trainingsmaschine):** Warmstart aus der
Versatz+Neigungs-Politik, `fixture_yaw_noise_rad = 0.7854` (±45°) zusätzlich,
4096 Umgebungen. Stopp beim 99-%-Kriterium: 99,19 % über ein Fenster von
5 461 Episoden, mittlere Maximaltiefe 28,5 mm, Neigungs-Bins 99,6 / 99,2 /
98,9 / 98,9 % (0–12°) — kein Randkollaps. Ein interaktives Replay mit vollem
Trainingsrauschen ergab 99,57 % über 703 Episoden.

**Bemerkenswert:** Der Sprung von Yaw 0 auf ±45° in einer Stufe verursachte
*keinen* Kollaps — anders als der Neigungssprung 5° → 15°. Erklärung: Der
relative Gierwinkel zwischen Zapfen und Tasche war von Anfang an trainiert
(±45° Reset-Rauschen auf dem Handgelenk seit der ersten Politik); die gedrehte
Tasche verschiebt nur den Zielwinkel, verlangt aber keine neuartige Bewegung.
Die Leiterregel aus Abschnitt 6 gilt also für *wirklich neue* Variationen;
bereits in anderer Form gelernte Variation verträgt große Sprünge.

Offen geblieben (bewusst): Die deterministischen Nullschuss-Proben bei festen
Yaw-Winkeln (10 / 22,5 / 45°) wurden nach einem uneindeutigen 93-%-Replay
abgebrochen; stattdessen wurde direkt trainiert. `--fixture-yaw` bleibt als
Messinstrument verfügbar.

## 8. Nebenbefunde und kleinere Anpassungen

- **Trainingstempo:** Die Umgebungszahl wurde von 128 auf 4096 erhöht.
  Messung: 0,61 s pro Iteration ≈ 107 000 Simulationsschritte/s. Der Engpass
  ist damit nicht mehr die Datensammlung. Hinweis: Episoden-bis-99 %-Zahlen
  sind zwischen Läufen mit unterschiedlicher Umgebungszahl nicht direkt
  vergleichbar.
- **"Drücken auf einer Stelle" erklärt (Messung, Fix noch offen):** Die
  Gelenk-Sollwerte sind nur auf Gelenkgrenzen begrenzt, nicht auf die
  Ist-Position. Bei blockiertem Kontakt läuft der Sollwert davon
  (Integrator-Wind-up). Gemessen an der 10°-Politik: fehlgeschlagene Episoden
  tragen im Mittel 60,6 Regler-Schritte Sollwert-Rückstand, erfolgreiche 12,8
  (Faktor 4,7; Maximum 134). Die inzwischen mitgeschriebene Verteilung der
  finalen D-038-Politik trennt sauber: Erfolgs-p95 0,50 rad gegen
  Fehlschlag-Mittel 1,87 rad (44 Fehlschläge). Ein Begrenzungsband zwischen
  diesen Werten ist damit ableitbar — der Eingriff bleibt bewusst ein
  separater, einzelner Schritt.
- **Benennungsschema für Läufe:** neu ausgeschrieben mit Einheiten, z. B.
  `peg30mm_pocket32mm_offset20mm_tilt0-10deg`. Damit ist im TensorBoard ohne
  Vorwissen erkennbar, welche Kurve welchen Aufbau zeigt.
- **Replay-Videos:** Jede Probe-Pose schreibt jetzt in einen eigenen
  Videoordner; vorher überschrieben sich aufeinanderfolgende Proben stumm.
- **Erfolg je Neigungs-Bin:** `demo_metrics.json` enthält jetzt
  `success_by_tilt_bin` (3°-Bins), damit ein 99 %-Mittel nicht verdecken
  kann, dass die großen Winkel schlecht sind.

## 9. Erkenntnisse für die Zielaufgabe (Checkliste)

1. **Beobachtbarkeit zuerst prüfen.** Jede unabhängig variierte Größe der
   Aufgabe muss in der Beobachtung stehen. Relativmessungen zur Tasche
   verstecken die Taschenpose vollständig.
2. **Orientierung als Quaternion, 4 Kanäle.** Praxisstandard für Eingänge
   (GenPiH); 6D nur für Ausgänge. Für diese Aufgabe: **25 Kanäle genügen** —
   durch das Yaw-Ergebnis (D-038) bestätigt, das Quaternion trug die Drehung
   ohne jede weitere Kanaländerung.
3. **Warmstart statt Neutraining.** Checkpoint verbreitern (Nullspalten,
   Normalisierer neutral, Optimierer leeren) erhält Gelerntes exakt.
   Tensor-Zuordnung dabei nach Name, nie nach Form.
4. **Leiterstufen klein halten — bei wirklich neuer Variation.** Verdopplung
   des Neigungsparameters funktionierte; Verdreifachung zerstörte die Politik.
   Der Yaw-Sprung 0 → ±45° in einer Stufe kollabierte dagegen nicht, weil der
   relative Gierwinkel längst trainiert war. Entscheidend ist, ob die
   Variation eine neuartige Bewegung verlangt, nicht die Zahl an sich.
5. **Erst messen, dann trainieren.** Die 50 %-Baseline bei 5° hat den ganzen
   Aufwand begründet und den Regressionstest geliefert; Ergebnisse werden von
   der Platte gelesen (`demo_metrics.json`), nicht aus Prosa.
6. **Mehr Umgebungen = fast freies Tempo**, solange der Grafikspeicher reicht
   (4096 auf einer RTX 3080 problemlos).

## 10. Offene Punkte

- Kombinationsstufe mit ±5 cm Versatz (zurückgestellt, Befehl liegt bereit).
- Wind-up-Begrenzungsband: Bandgrenzen jetzt ableitbar (zwischen 0,50 und
  1,87 rad), Eingriff noch nicht gemacht.
- Deterministische Yaw-Nullschuss-Proben (10 / 22,5 / 45°) nie gemessen;
  `--fixture-yaw` bleibt als Messinstrument verfügbar.

## 11. Heutige Commits (dieses Repo)

| Commit | Inhalt |
|---|---|
| `e783b00` | D-036-Nachtrag (verifiziert) und D-037-Entscheidung in DECISIONS.md |
| `4b43c64` | Eigener Videoordner je Probe-Pose in play.py |
| `521919b` | Checkpoint-Verbreiterung: Tensor-Zuordnung nach Name statt Form |
| `b1c002e` | PROBLEMS.md-Eintrag zum Abbruch der Verbreiterung |
| `3760133` | Messung des Gelenk-Sollwert-Rückstands (Wind-up) |
| `fc9e185` | Ausgeschriebenes Namensschema für Läufe |
| `5ca2349` | Rückstands-Verteilung (p95/Max) je Ausgang statt nur Mittelwert |
| `439c74a` | Yaw als messbare Probe freigeschaltet (D-038): Sperre ersetzt, `--fixture-yaw`, Bin-Statistik über Polarwinkel |
| `1f71614` | Literaturnotizen zu Curriculum-Strategien (Factory/GenPiH: gemeinsam randomisieren; IndustReal: erfolgsbasierte Stufen) |
| `e12e1b8` | Verifiziertes Ergebnis der Kombination Versatz+Neigung+Yaw in D-038 nachgetragen |
