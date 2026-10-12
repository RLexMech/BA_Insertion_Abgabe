# Evaluationsplan — fünf AutoDR- und fünf No-DR-Policies

Geändert 2026-09-21 nach dem Plan-Commit `a87cfa9a`, VOR dem ersten Evaluationslauf: Hauptzahl `success_train` statt `success_strict` (Nutzerentscheidung, Literaturbefund).
Zweite Änderung 2026-09-21, ebenfalls VOR dem ersten Lauf (Nutzerentscheidung „Weg 1, Option A"): K1/K2 rechnen mit dem Welch-Intervall über die Seed-Raten, der Percentile-Bootstrap steht als Gegenprobe daneben; der eigene Zwei-Ebenen-Bootstrap und G7 entfallen. Beleg: `docs/reference/literature_check_rl_seed_evaluation_2026-09-21.md`.
Dritte Änderung 2026-09-22, nach dem Technik-Block und VOR dem ersten echten Block (Nutzer): fester Abspiel-Seed 100 für alle zehn Policies (§ 3.2).
Vierte Änderung 2026-09-23, NACH allen Evaluationsläufen RT-222..RT-225 und VOR dem ersten Fixed-DR-Training (D-192, Nutzer): dritter Arm Fixed-DR, Frage K5 (§ 1.4, § 3.1, § 3.2, § 5). Nachträglich ergänzt; die Arbeit nennt K5 so. K1–K4 und „Robustheitsziel erreicht" bleiben unverändert.

Stand 2026-09-21, Worktree `p5-kraftsensor`. NICHTS ist gebaut. Alles UNVERIFIED bis
zum Trainings-PC-Lauf. Jede Zahl unten trägt ihre Fundstelle; „UNBEKANNT" heißt: nicht gelesen
oder nicht messbar am Laptop.

## Kontext

Zehn eingefrorene Policies liegen vor (`model_1499.pt`, Code `5a6cc8e`): AutoDR RT-206s1..s5,
No-DR RT-207s1..s5. Bisher gibt es nur Trainingszahlen. Die Arbeit braucht einen ehrlichen
Nachweis: Ist eine Policy robust, unter welchen Bedingungen, wo sind die Grenzen. Die Grundidee
bleibt: vorab gezogene gemeinsame Testfälle, Tabelle mit Hash in git, getrennte Übungstabelle,
Ergebnis je Episode, Prüfung der angewendeten Werte. Neu: klare Zielaussage, vorab festgelegte
Auswahlregel, Erfolgszahl nach Trainingsdefinition als Hauptzahl, Statistik über die Seeds (Welch), Vorab-Festlegung
in `eval_plan.md`.

---

## 1. Robustheitsziel

### 1.1 Störgrößen (Heimat der Zahlen: `autodr.DR_DIMS`, `Belege_Streuwerte.md`, D-178..D-183)

| Größe | Art | Einheit | Verteilung im Test | Bereich | Begründung (Stand der Belege) |
|---|---|---|---|---|---|
| Versatz `lat_r`,`lat_phi` | physisch | m | flächengleich auf Kreisscheibe (`r=R√u`) | 0..0,030 | ENTSCHIEDEN, nicht hergeleitet (D-178). Literatur-Anker ±3..±20 mm; 30 mm liegt darüber. `[TESTWERT]` |
| Gier `yaw` | physisch | rad | gleichverteilt | ±0,0873 (5°) | ENTSCHIEDEN (Nutzer 2026-09-11). Anker ±5° IndustReal/AutoMate. `[TESTWERT]` |
| Kippung `tilt` + Azimut | physisch | rad | Betrag gleichverteilt, Richtung 0..2π | 0..0,1745 (10°) | ENTSCHIEDEN. CAD-Fügegrenze 8,52°; die Decke liegt 1,48° darüber (Annahme D-178 (6)). `[TESTWERT]` |
| Starthöhe | physisch | m | gleichverteilt | 0,020..0,120 | Unterkante GEMESSEN (RT-197a/b), Decke ENTSCHIEDEN (D-179). `[TESTWERT]` |
| Reibung | physisch | 1 | gleichverteilt, statisch = kinetisch | 0,08..0,20 | DuPont-Tabellen PA66/POM; Paarung PA6/POM ist ANNAHME (D-181) |
| Lage der Aufnahme xy | physisch | m | gleichverteilt je Achse | ±0,005 | `cfg:725`; D-034 nicht gelesen → Beleg UNBEKANNT |
| Gelenkrauschen | physisch | rad | — | 0 (AUS) | Spalten existieren, Skala 0 |
| Greif-Fehler `grasp_obs_x` | je Episode konstant, NUR Beobachtung | m | gleichverteilt | ±0,003 | Factory/FORGE ±3 mm (D-183) |
| Taschenposition (Beob.) | je Episode konstant, NUR Beobachtung | m | gleichverteilt je Achse | ±0,005 | TacSL Tab. V (Belege B4a) |
| Kraft (Beob.) | je Schritt, NUR Beobachtung | N | Gauß | σ 3,5 | UR5e-Datenblatt „Precision ±3.5 N"; dass das ein σ ist, ist ANNAHME |

Abhängigkeiten, die der Code erzwingt: Regler-Box folgt aus Versatz, Kippung, Starthöhe (D-180,
`check_env_wiring.py:4918`); Regler-Kippkegel 8,52° < Kipp-Decke 10°; Reibung wird auf Roboter UND
Aufnahme geschrieben, Paarwert gilt nur bei PhysX-Kombinationsmodus `average` (UNGEPRÜFT).

Urteil über die Belege: vier der fünf Decken sind Entscheidungen, keine Herleitungen. Das ist
zulässig, wenn die Arbeit es so sagt: „Zielbereich = festgelegter Prüfbereich". Keine neue
Literaturrecherche nötig; eine wesentliche Begründung fehlt nicht, sie ist als Annahme benannt.

### 1.2 Zielbereich

Zielbereich = die Maxima aus `DR_DIMS`, alle Größen unabhängig gezogen wie oben. Er stand VOR den
Läufen fest (D-178/179/181) und hängt nicht davon ab, wie weit ein Lauf geöffnet hat.
Zusatz-Dokumentation (gelesen aus `rt_logs/RT-206s*/…_scalars.csv`): alle fünf AutoDR-Läufe
haben alle Grenzen am Maximum, erstmals bei It. 732 / 900 / 810 / 768 / 936. Zielbereich und
erreichter Trainingsbereich sind also gleich. Der Testbereich wird nie nachträglich geändert.

### 1.3 Zulässige Aussage

1. **Mittlere Erfolgswahrscheinlichkeit** unter der Störverteilung aus 1.1, in der Simulation
   (Isaac Sim 5.1, starre Greiferkette, dieses Kontaktmodell). Die 1000 Zeilen sind eine
   Zufallsstichprobe dieser Verteilung; die Aussage gilt der Verteilung, nicht der Tabelle.
2. **Verhalten an Grenzen:** je eine Grenze festgenagelt, Rest gezogen (Grenztabelle, § 3).
3. **Nicht untersucht:** Realität, weicher Sauger, Bereiche außerhalb 1.1, Kombination mehrerer
   Grenzen zugleich, andere Bauteile, Kraftspitzen unter 1/15 s (siehe § 4).

### 1.4 Erfolgskriterien — OHNE Prozentschwelle (Nutzer 2026-09-21: es gibt keine fachliche Anforderung)

Kein absoluter Prozentwert gilt als „bestanden". Die Kriterien sind VERGLEICHE mit Intervall.
Sie stehen vor dem finalen Lauf fest. Alle nutzen die Erfolgsrate nach Trainingsdefinition
(`success_train`, § 4); K1, K2 und K3 rechnen damit. Die Kraftabbruchrate wird je Policy getrennt
mit Wilson-Intervall berichtet, ohne Urteil.

| # | Frage | Kriterium erfüllt, wenn |
|---|---|---|
| K1 | Macht AutoDR robuster als No-DR? | Differenz AutoDR − No-DR auf T-final: das 95-%-Welch-Intervall der Differenz (§ 5) liegt ganz über 0 |
| K2 | Kostet die Robustheit Nominalleistung? | Differenz AutoDR − No-DR auf T-nominal: das Intervall liegt NICHT ganz unter 0 |
| K3 | Bricht die finale Policy an einer einzelnen Grenze ein? | Für keinen der 7 Grenzblöcke liegt das Intervall (Block − Gesamtrate T-final, Newcombe-Differenz zweier Wilson-Intervalle) ganz unter 0 |
| K4 | Hängt das Ergebnis am Trainings-Seed? | Nur Bericht: alle zehn Einzelraten, Spanne und Standardabweichung je Methode. Kein Urteil |
| K5 (nachträglich, 2026-09-23) | Liegt der Nutzen am Wachsen der Breite? | Kein Bestehen. Bericht: Differenz AutoDR − Fixed-DR auf T-final, 95-%-Welch-Intervall (§ 5, Punkt 7) |

K5, vorab festgelegt vor dem ersten Fixed-DR-Training. Hypothese: AutoDR erreicht auf T-final eine
höhere Erfolgsrate als Fixed-DR. Richtung aus DORAEMON (Tiboni et al. 2024) Tab. 1; die
Zeilenbeschriftung wird vor einem Zitat am Original geprüft (D-192). Lesart, alle drei Fälle vorab:
- Intervall ganz über 0 → „Im Zielbereich ist AutoDR robuster als Fixed-DR" (diese Aufgabe,
  1500 It., 5 Seeds).
- Intervall ganz unter 0 → „Fixed-DR ist robuster als AutoDR."
- Intervall enthält 0 → „Kein Unterschied gezeigt." NICHT: „gleich gut".
T-nominal: dieselbe Differenz mit Intervall daneben, dieselben drei Lesarten, kein eigenes Kriterium.
K5 geht NICHT in „Robustheitsziel erreicht" ein.

Aussage „Robustheitsziel erreicht" = K1 UND K2 UND K3. Das heißt wörtlich: „AutoDR ist im
Zielbereich robuster als No-DR, ohne nominalen Verlust, und keine Einzelgrenze fällt heraus."
Es heißt NICHT „robust genug für den Einsatz" — dafür fehlt die Anforderung. Die absolute
Erfolgsrate der finalen Policy steht mit Wilson-Intervall daneben, ohne Urteil.
Grenze der Kriterien: ein Intervall, das 0 ausschließt, ist inhaltlich ein Test auf 5-%-Niveau.
Bei 5 Seeds ist die Stichprobe klein (Agarwal et al. 2021: „3-5 runs is a small sample size");
Colas et al. 2018 messen für den Bootstrap bei N = 5 etwa 10 % Fehlalarme statt 5 %. Deshalb
Welch als Hauptintervall. Das steht so in der Arbeit (§ 5).

---

## 2. Zwei getrennte Fragen

**A — Methodenvergleich.** AutoDR gegen No-DR: Erfolgsrate (`success_train`) je Policy, Mittel und Streuung
über die fünf Trainings-Seeds, Differenz in Prozentpunkten (§ 5).

**B — Finale Policy.** Auswahlregel, VORAB fest:
- Kandidaten: die zehn `model_1499.pt`. Keine Checkpoint-Auswahl innerhalb eines Laufs.
- Auswahl auf einer eigenen **Auswahltabelle (Tabellen-Seed 30)**, Zielbereich, 1000 Zeilen.
- Regel: höchste Erfolgsrate (`success_train`). Gleichstand → kleineres Kraft-p95 unter den
  Erfolgen. Noch Gleichstand → kleinere Seed-Nummer.
- Die finale Tabelle (Seed 10) sieht die Auswahl nie. Keine zusätzlichen Trainingsläufe.

Tabellen: Seed 20 = Übung (nur Technik prüfen), Seed 30 = Auswahl, Seed 10 = final.

---

## 3. Fairness und Testmatrix

### 3.1 Fairness (gelesen 2026-09-21)

- Die zwei Trainingsbefehle unterscheiden sich in GENAU einem Feld: `env.dr_mode=autodr` / `off`
  (`rt_logs/RT-206_expectation.md:53`, `RT-207_expectation.md:26`; `diff` der beiden
  `params/env.yaml`: nur `log_dir` und `dr_mode`).
- Budget gleich: 256 Envs × 16 Schritte × 1500 It. = 6 144 000 Env-Schritte.
- PPO, Netz (128/128 ELU), Beobachtung (`force`, 28 Kanäle), Aktion, Reward, Abbruch: gleich.
  Kein `dr_mode`-Zweig in `_get_observations`, `_get_rewards`, `_get_dones`.
- Beide haben dieselbe Start-Boden-Stufe und öffnen die Starthöhe bis 0,120 m.
- „No-DR" heißt NICHT „keine Streuung": Starthöhe, Aufnahme-Lage ±5 mm und Beobachtungsrauschen
  sind auch dort an. Aus sind Versatz, Gier, Kippung, Reibungsband. So benennen.
- Zulässig: Aussage AutoDR gegen No-DR (K1). Aussage AutoDR gegen feste DR NUR über K5 und nur in
  dessen Lesart (§ 1.4). Fixed-DR ist nachträglich ergänzt (D-192, 2026-09-23).
- Fixed-DR (D-192): Befehl wie RT-206, nur `env.dr_mode=fixed_dr`. Dieselbe Start-Boden-Stufe wie
  AutoDR; ab H_min alle fünf Größen fest auf dem Zielbereich (Breite von `FixedWidth.at_max`).
  ZWEI Unterschiede zu AutoDR, nicht einer: (1) die Breite wächst nicht; (2) keine Randstichprobe —
  AutoDR setzt in Phase dr die Hälfte der Episoden auf eine Grenze (`autodr.P_BOUNDARY` 0,5),
  Fixed-DR zieht gleichverteilt wie in der Literatur (DORAEMON). So benennen.
- **Zu prüfen (V1):** Reibwert im No-DR-Training. Erwartet 0,14 (D-111-Schreibzugriff); die
  `env.yaml` zeigt Material-Standardwerte 0,5. Aus dem Startbericht von RT-207s1 lesen.

### 3.2 Testmatrix (alle zehn Policies, deterministische Aktion, 1 Episode je Zeile)

| Lauf | Tabelle | Breite | Zweck | Läufe |
|---|---|---|---|---|
| T-nominal | Seed 10 | Versatz/Gier/Kippung/Reibung auf Mitte, Starthöhe 0,020..0,120 (= No-DR-Trainingsverteilung) | gemeinsame Nominalbedingung | 10 |
| T-auswahl | Seed 30 | Zielbereich | Auswahl finale Policy | 10 |
| T-final | Seed 10 | Zielbereich | Hauptzahl A und B | 10 |
| T-grenze | Grenztabelle Seed 10: 7 Blöcke × 150 Zeilen, je Block eine der 7 Grenzen festgenagelt (`apply_boundary`), Rest gezogen | Zielbereich | Verhalten an Grenzen (K3) | 1 (nur finale Policy) |
| Technik | Seed 20 | beide | § 6 | 3 |
| T-nominal-F (nachträglich) | Seed 10 | wie T-nominal | K5 nominal | 5 (nur Fixed-DR) |
| T-final-F (nachträglich) | Seed 10 | Zielbereich | K5 | 5 (nur Fixed-DR) |

Fixed-DR (nachträglich, 2026-09-23): gleiche Tabellen (Hash), Abspiel-Seed 100. Die AutoDR- und
No-DR-Läufe von T-nominal und T-final werden NICHT wiederholt. Fixed-DR ist kein Kandidat für die
finale Policy (D-190 bleibt). Code-Stand (Nutzer 2026-09-23, „A"): die Fixed-DR-Läufe spielen
auf `8299485d`, dem Stand von RT-222..RT-225. Der Trainings-PC schaltet dafür zurück; so gilt
`eval_collect.py` („ein code_commit") ohne Änderung. Dass `play.py --eval-table` auf diesem Stand
die Fixed-DR-Checkpoints lädt, ohne `FixedDR` zu kennen: UNVERIFIED bis zum ersten Lauf.

Reihenfolge fest: Technik → T-nominal → T-auswahl → (Auswahl notieren, committen) → T-final →
T-grenze. **Umfang: 34 Läufe** (vorher 36; die zwei G7-Halbläufe entfallen), dazu nachträglich 10 Fixed-DR-Läufe. Ein „Lauf" ist EIN `play.py`-Aufruf: 1000 Envs parallel, jede spielt
EINE Episode, höchstens 256 Schritte (17 s simulierte Zeit), dann Ende. Kein Training.
Laufzeit je Lauf UNBEKANNT; der erste Techniklauf misst sie.

Warum jeder Block nötig ist:
- T-final (10): die Hauptzahl. Ohne alle zehn gibt es keinen Methodenvergleich.
- T-nominal (10): K2 und die Ursachenprüfung. Ohne ihn ist unklar, ob ein schlechtes T-final an
  der Störung liegt oder an der Policy selbst.
- T-auswahl (10): nur so bleibt T-final unberührt von der Auswahl. Streichbar NUR, wenn die
  finale Policy ohne Leistungsblick festgelegt wird (z. B. „AutoDR Seed 1"); dann entfällt aber
  der Anspruch „beste Policy".
- T-grenze (1): K3 gilt der finalen Policy. Alle zehn = optional (O6).
- Technik (3): Erstlauf, Wiederholung (G6, `--eval-tag`), ein Lauf in Nominalbreite. G4 braucht
  keinen eigenen Lauf. G7 entfällt (Kopfzeile).

**Abspiel-Seed (Nutzer 2026-09-22):** alle zehn Policies spielen jeden Block mit demselben
`--seed 100`. 100 ist kein Trainings-Seed (1–5, 20, 42) und kein Tabellen-Seed; die Zahl selbst
ist eine freie Wahl ohne Quelle. Der Seed steuert nur den Zufall beim Abspielen
(Beobachtungsrauschen), nicht die Testfälle; gleicher Seed = gleicher Ausgang (RT-220: 0 von
1000 Zeilen anders). Der Technik-Block lief mit `--seed 1` und prüft nur Technik.

Ein kleines PowerShell-Skript `scripts/run_eval.ps1` startet einen Block (10 Läufe) mit einem
Befehl über `rt_log.ps1`; die zehn Checkpoint-Pfade stehen in einer Liste im Skript.

Schlechtes T-nominal löst die Differentialdiagnose aus (Messung → Prädikat → Regler …). Es
beweist allein keinen Eval-Fehler und keinen Policy-Fehler.

---

## 4. Erfolg und Belastung

**Definitionen (aus dem Code, nicht neu erfunden):**
- Kraftsignal: Gelenk-Reaktionskraft am `tool_link` (`body_incoming_joint_wrench_b`,
  `insertion_env.py:2005`), Rahmen des Elternkörpers, ungedreht; Gewicht vor dem Filter abgezogen;
  EMA-Faktor 0,25 (`cfg:1200`); EIN Wert je Policy-Schritt (15 Hz; Physik 120 Hz).
- Kraftgrenze: 30 N, Vergleich `>=` (`insertion_math.py:1223`). Im Training: Episode endet
  (`truncated`), Zahlung −1, Erfolg gewinnt im selben Schritt. Die 30 N sind eine Schranke der
  eigenen Simulation, KEINE Schadensgrenze (Inbox Z. 3934-3970). Die Arbeit behauptet keinen
  Bauteilschutz.
- Erfolg (Training): Tiefe 33..36 mm, Durchdringung < 0,2938 mm, im Taschenvolumen. Keine Haltezeit.
- Zeitlimit 256 Schritte = 17,07 s.

**Kennzahlen je Episode (aus `episodes.csv`, erste Episode je Env, `env_id` = Zeilen-ID):**
- `success_train` = `outcome == success`. **Hauptzahl.** Erfolg ist geometrisch definiert, wie im
  Training; Kraft ist Abbruchgrenze, kein Bestehenskriterium.
- `success_strict` = `outcome == success` UND `force_max_filtered_n < 30`. Ergänzend. Nur hier gilt:
  Erfolg und Kraftverletzung im selben Schritt → Misserfolg.
- Endgrund: Erfolg / davon Erfolg mit Kraftverletzung im selben Schritt / Kraftabbruch / Zeitlimit / sonstig
  („sonstig" > 0 = Defektsignal).
- `steps`, `force_max_filtered_n`, `force_max_raw_n`, `ip_max_mm`.

**Gegen Irreführung:** Kraft und Dauer NUR über Erfolge berichten (Median, p95, Max) und
getrennt je Endgrund. Kein Mittel über alle Episoden: ein früher Abbruch ist weder schonend noch
schnell. Dauer zusätzlich als „Anteil erfolgreich bis Schritt k".

**Spitzen zwischen Policy-Schritten:** heute NICHT erfasst — 7 von 8 Physik-Schritten liest niemand
(`episodes_meta.json` sagt das selbst). Kleinste Änderung (Priorität 2): in `_apply_action` je
Physik-Schritt das ungefilterte Maximum mitschreiben, neue Spalte `force_max_substep_n`, NUR
Protokoll, nicht Abbruch. Dann misst der Übungslauf das Verhältnis Substep-Max zu Schritt-Max.
Ob die Wrench-Daten je Substep frisch sind: UNGEPRÜFT → Identitätstest. Ohne diese Änderung steht
die Lücke als Grenze in der Arbeit.

---

## 5. Statistik

5 Trainingsläufe je Methode (Stichprobe des Lernverfahrens) × 1000 Episoden je Policy
(Stichprobe der Störverteilung). Episoden sind KEINE Wiederholungen des Lernverfahrens. Wie in der
Literatur ist der Seed die Einheit: jede Policy wird erst zu EINER Rate gemittelt, nur die Seeds
werden verglichen (Agarwal et al. 2021 S. 4 Abschn. 3; Colas et al. 2018 S. 3; Beleg-Datei oben).

Bericht:
1. Alle zehn Erfolgsraten (`success_train`), je mit Wilson-95-%-Intervall. `success_strict` bekommt
   nur je Policy eine Rate mit Wilson-Intervall; alle Wilson- und Newcombe-Intervalle der
   Hauptaussage rechnen mit `success_train`.
2. Je Methode Mittel und Standardabweichung über die 5 Seeds.
3. Differenz der Mittel in Prozentpunkten.
4. **Intervall der Differenz (K1, K2):** Eingabe sind die 5 + 5 Seed-Raten (`success_train`).
   - HAUPTZAHL: Welch-95-%-Intervall der Differenz der Mittel,
     `scipy.stats.ttest_ind(autodr, nodr, equal_var=False).confidence_interval(0.95)`. Colas et al.
     2018 empfehlen Welch vor dem Bootstrap bei kleinem N (S. 19, Abschn. 6).
   - GEGENPROBE daneben: Percentile-Bootstrap der Differenz,
     `scipy.stats.bootstrap((autodr, nodr), statistic, paired=False, method="percentile")`, fester
     Zufalls-Seed. `method` ausdrücklich setzen: scipys Standard ist BCa (geprüft an scipy 1.18.1).
     Verfahren nach Colas et al. 2018 S. 9-10 Abschn. 3.2.
   - Grenze, in der Arbeit genannt: 5 Seeds sind eine kleine Stichprobe; der Bootstrap hat bei
     N = 5 etwa 10 % Fehlalarme statt 5 % (Colas S. 16 Abschn. 5.1). Kein p-Wert als Hauptzahl.
   - Episoden werden NICHT neu gezogen: das steht in keiner gelesenen Quelle (NOT FOUND).
5. Finale Policy: Wilson-95-%-Intervall auf T-final. Annahmen: Zeilen unabhängig gleichverteilt
   gezogen (ja, `default_rng`), Envs teilen keine Physik (parallel, getrennt), Policy vorab gewählt
   ohne T-final (ja, § 2). Genauigkeit bei 1000: ±1,2 pp bei 0,96; ±3,1 pp bei 0,5. Null Ausfälle
   in 1000 schließen nur Ausfallraten über ca. 0,3 % aus („Dreierregel"); seltenere Ausfälle bleiben offen.
6. Bins je Größe (bestehende Kanten aus `Belege_Streuwerte.md` § D) und Grenzblöcke: je mit Wilson.
7. **K5 (nachträglich, 2026-09-23):** wie Punkt 4, Eingabe 5 AutoDR- + 5 Fixed-DR-Seed-Raten
   (`success_train`), auf T-final und T-nominal. Die AutoDR-Raten sind dieselben wie in K1. Keine
   Korrektur für Mehrfachvergleiche; in der Arbeit als Grenze genannt.

---

## 6. Technische Gültigkeit (alles auf der Übungstabelle Seed 20)

| # | Prüfung | Bestanden, wenn |
|---|---|---|
| G1 | Tabellen-Hash | Log-Zeile = Laptop-Hash = `Get-FileHash` |
| G2 | Zuordnung | 1000 Episoden mit `episode == 0`, `env_id` genau 0..999, jede einmal |
| G3 | Angewendete Werte | `applied.csv` (ALLE Zeilen, SI-Einheiten, Taschenrahmen; Reibung aus PhysX zurückgelesen) = Tabelle abgebildet; Toleranz 1e-6 |
| G4 | Reset sauber | zweite Episode derselben Zeile im selben Lauf = erste (Ausgang gleich); `wrench_valid` im ersten Schritt 0 |
| G5 | Policy eingefroren | Hash über den ganzen `state_dict()` von `actor_obs_normalizer` vor und nach dem Lauf gleich (Manifest-Felder `normalizer_sha256_before/after`; die Namen `_mean/_var` stehen nicht in der Quell-Liste, Inbox Schritt C). Quelle gelesen: `get_inference_policy` → `eval()`, `act_inference` = Netzausgang (Inbox, G5-Eintrag) |
| G6 | Wiederholbarkeit | gleiche Policy, gleiche Tabelle, gleicher Seed, zweimal: Anteil Zeilen mit anderem Ausgang berichten. Keine Bit-Gleichheit gefordert |
| G7 | ENTFÄLLT (2026-09-21) | Keine gelesene Quelle prüft Reihenfolge oder Env-Zahl; jeder echte Lauf hat genau 1000 Envs, die Env-Zahl kann also keinen Vergleich verfälschen. Ob ein Ergebnis vom Env-Platz abhängt, bleibt UNGEPRÜFT und steht als Grenze in der Arbeit |
| G8 | Festlegung | `--seed` immer gesetzt (sonst still 42), `env.dr_mode=table` für BEIDE Methoden, kein `start_floor_m` |

**Rauschen:** heute globaler Zufallsstrom (`randn_like`), keine Spalte. Wiederholbar je Testfall,
Schritt und Kanal ginge über ein vorab gezogenes Rausch-Feld (1000 × 256 × 6 Werte, aus dem
Tabellen-Seed), nur im Tabellen-Modus, in `obs_noise.py`. Aufwand ca. 1 Tag mit Checks; Gewinn:
engerer Paarvergleich. → OPTIONAL (O2). Standard: unabhängiges Rauschen gleicher Verteilung,
dokumentiert; sein Einfluss steckt in G6.

**Ausschluss von Testfällen:** Standard KEINER. Einzige vorab erlaubte Regel: Startzustand mit
Durchdringung ≥ 0,2938 mm VOR der ersten Aktion (policy-unabhängig, für alle zehn gleich). Der
Übungslauf zählt solche Zeilen. Schwierig oder „Referenzregler scheitert" ist kein Grund.

---

## 7. Vorab-Festlegung und Misserfolg

- Nach Freigabe: dieser Plan → `eval_plan.md` (Worktree-Wurzel), Commit VOR dem ersten Lauf auf
  Seed 30 oder 10. Das Auswahlergebnis kommt als eigener Commit dazu, VOR T-final.
- `eval_manifest.json` je Lauf (schreibt `play.py`): Plan-Commit, Code-Commit, SHA-256 von Checkpoint
  und Tabelle, Breite, vollständige Env-Cfg, Isaac-Sim/Lab/rsl_rl-Version, Aktionsmodus, Rauschwerte,
  Zeitlimit, `num_envs`, Seed.
- Ziel verfehlt: dominante Fehlerarten aus Endgrund × Bins × Grenzblöcke nennen; EINE kleinste
  Änderung vorschlagen (Reihenfolge der Differentialdiagnose, Reward zuletzt). Die Eval macht keine
  Policy robuster.
- Wird nach T-final etwas angepasst, ist Seed 10 Entwicklungsinformation. Bestätigung nur auf
  neuer Tabelle (Seed 11). Höchstens EINE solche Runde in dieser Arbeit.

---

## 8. Änderungen, Reihenfolge, Abschluss

| # | Datei | Änderung | Hypothese | Offline-Check + Mutation |
|---|---|---|---|---|
| 0 | `eval_plan.md` | Plan einfrieren | — | — |
| A | neu `scripts/make_eval_table.py`; `scripts/eval/*.csv` (Seeds 10, 20, 30, Grenztabelle); `.gitignore`-Ausnahme; `.gitattributes -text` | Tabellen | gleiche Bytes auf beiden Rechnern | `--self-test`; Mutation: Spalte im Kopf tauschen → `RowTable.from_csv` lehnt ab |
| B1 | `insertion_env_cfg.py` | `dr_mode='table'`, `dr_table_path`, `dr_table_width` ∈ {target, nominal}; `validate_rl_config` lehnt Boden/Curriculum/fehlenden Pfad ab | falsche Kombination startet nie | `check_env_wiring.py`; Mutation: Ablehnung entfernen |
| B2 | `insertion_env.py` | Tabellenzweig in `_draw_reset_conditions` (:2739) statt `torch.rand`; `FixedWidth`-Provider; Env i ↔ Zeile i; `applied.csv` beim ersten Reset | Env spielt genau die Zeile | `check_autodr.py` neuer Fall mit Form (1000,16) (L-14); Mutation: Zeile i+1 |
| C | `scripts/rsl_rl/play.py` | `--eval-table`, `--eval-width`; Ordner `eval_<breite>_<sha8>`; Stopp nach 1. Episode aller Envs; `eval_manifest.json` | Lauf endet selbst, Manifest vollständig | `check_env_wiring.py`-Pin; Mutation: Manifest-Feld streichen |
| D1 | neu `scripts/check_eval_conditions.py` | G1–G3 | angewendet = Tabelle | `--self-test`; Mutation: Wert +1e-3 |
| D2 | neu `scripts/eval_collect.py`, `eval_stats.py`, `eval_figures.py`, `eval_report.py`, `requirements-analysis.txt` (§ 8b) | § 4 Kennzahlen (`success_train` als Hauptschlüssel), Wilson, Newcombe, Welch-Intervall + Percentile-Bootstrap über die Seed-Raten (§ 5); liest `episodes.csv` (Fenster: `episode == 0`, keine eigene Fensterlogik, L-15) | eine Routine, eine Zahl | `--self-test` (Wilson k=0/5/10 von 10; Welch gegen eine von Hand gerechnete Referenz); Mutation: Nenner n−1 |
| E | (Priorität 2) `insertion_env.py` | `force_max_substep_n` | Spitzen werden sichtbar | Identitätstest Substep-Max ≥ Schritt-Max |
| F | `scripts/CLAUDE.md`, Inbox, `InBachelorErwähnen.md`, `Belege_Streuwerte.md`, `selftest_checks.py`-Liste | Doku | — | — |

Wiederverwendet: `RowTable`, `FixedWidth`, `apply_boundary`, `map_unit_to_bounds` (`autodr.py`),
`episodes.csv`/`episodes_meta.json` (kein neues `eval.json`), `check_episode_log.py`,
`scripts/tools/checks.py`, `trace_first_episode.py` als Muster für „erste Episode je Env".
NICHT in diesem Plan: Nachfüllen, `GridTable`, Rasterkarten, `collect_results.py`, `run_study.ps1`.

**Abschlusskriterien:** G1–G8 (ohne G7) auf Seed 20 belegt → `eval_plan.md` mit K1-K4 committet →
T-nominal, T-auswahl, Auswahl-Commit, T-final, T-grenze gelaufen, alle Manifeste vollständig →
`eval_stats.py` schreibt EINE Ergebnisdatei → Laufkatalog-Abschnitt und `InBachelorErwähnen`-Zeilen.

## 8b. Automatische Auswertung und Grafiken (läuft NUR am Laptop, nie am Trainings-PC)

**Grundsatz:** Jede Zahl und jede Abbildung der Arbeit entsteht per Befehl aus den gespeicherten
Ergebnisdateien. Nichts wird von Hand übertragen.

**Eingabe:** `results/eval/<block>/<policy>/` mit `episodes.csv`, `episodes_meta.json`,
`applied.csv`, `eval_manifest.json` (vom Trainings-PC kopiert; je ca. 100 KB). Die Dateien kommen
in git (Ausnahme in `.gitignore`), sie sind der Beleg der Arbeit.

**Drei Skripte, eine Kette:**
1. `scripts/eval_collect.py` — prüft alle Manifeste (gleicher Plan-Commit, gleicher Code-Commit,
   richtiger Tabellen-Hash je Block, 1000 Zeilen, jede einmal) und baut EINE lange Tabelle
   `results/eval/episodes_all.csv`: Methode, Seed, Block, Zeilen-ID, Endgrund, `success_strict`,
   `success_train`, Schritte, Kräfte, angewendete Störwerte. Bricht mit Exit ≠ 0 ab, wenn ein
   Manifest nicht passt (Gleichheit prüfen, nicht „nicht leer").
2. `scripts/eval_stats.py` — schreibt `results/eval/summary.json` und `summary_tables.csv`
   (`success_train` als Hauptschlüssel; K1–K4, alle Intervalle, Bins, Grenzblöcke, Kraft und Dauer
   über Erfolge; `success_strict` je Policy ergänzend).
3. `scripts/eval_figures.py` — schreibt PDF (Vektor) und PNG nach `docs/figures/eval/`.
Ein Befehl für alles: `.venv-analysis/Scripts/python.exe scripts/eval_report.py` (gebaut 2026-09-23;
läuft auf einer Kopie der vier Blöcke, `technik/` bleibt außen vor).

**Bibliotheken und welche Funktion welches Intervall liefert:**

| Größe | Funktion | Passt zur Methodik? |
|---|---|---|
| Wilson-Intervall einer Rate | `statsmodels.stats.proportion.proportion_confint(k, n, method="wilson")` | ja, direkt |
| Differenz zweier unabhängiger Raten (K3) | `statsmodels.stats.proportion.confint_proportions_2indep(..., method="newcomb")` | ja (Newcombe-Wilson) |
| Mittel, Standardabweichung über 5 Seeds | `numpy` (`ddof=1`) | ja |
| Intervall der Differenz (K1, K2), Hauptzahl | `scipy.stats.ttest_ind(..., equal_var=False).confidence_interval(0.95)` | ja (Welch; Colas et al. 2018 S. 19) |
| Intervall der Differenz (K1, K2), Gegenprobe | `scipy.stats.bootstrap(..., paired=False, method="percentile")` | ja (Colas et al. 2018 S. 9-10; `method` ausdrücklich, Standard ist BCa) |
| Tabellen, Gruppieren, Bins | `pandas` | — |
| Grafiken | `matplotlib` (wie die bestehenden `plot_*.py`) | — |

seaborn ist nicht nötig; matplotlib reicht und ist schon im Repo in Gebrauch.

**Umgebung:** `.venv-analysis` am Laptop, feste Versionen in `requirements-analysis.txt`
(D-189). Installiert und importiert am 2026-09-21.

**Abbildungen (alle ohne Titel, deutsche Achsen, farbenblind-sichere Palette, PDF + PNG).**
Gebaut 2026-09-23 (`scripts/eval_figures.py`, Liste ÜBERSCHRIEBEN nach Nutzerfreigabe; Quellen,
Farbprüfung und Bauschritte: `docs/eval_figures_plan.md`). Farben: AutoDR `#3B6FD4`, No-DR `#D97706`.

| # | Inhalt | Beantwortet |
|---|---|---|
| F1 | Erfolgsrate je Seed als **Punkt** mit Wilson-Intervall (Weissgerber 2015: bei kleinem n jeden Punkt zeigen), T-nominal neben T-final; Methodenmittel als Linie | K1, K2, K4 |
| F2 | Differenz AutoDR − No-DR: Welch-Intervall dick, Bootstrap dünn, je eine Zeile für T-nominal und T-final, Nulllinie | K1, K2 |
| F3 | Erfolgsrate über jeder Störgröße, **4 Felder** (Nutzer 2026-09-23; Reibung hat keine Bin-Kanten): Mittel dick, Seeds dünn, leere Bins weggelassen, **No-DR-Trainingswert markiert** (Versatz, Gier, Kippung je 0; Starthöhe 20–120 mm, `results/training/nodr_training.json`) | wo No-DR bricht, wo AutoDR hält |
| F4 | Finale Policy: 7 Grenzblöcke mit Wilson-Intervall, Gesamtrate T-final als Linie **mit Wilson-Band** | K3 |
| F5 | Endgründe je Policy auf T-final, gestapelt (Erfolg / Erfolg mit Kraftverletzung / Kraftabbruch / Zeitlimit), Status-Palette des `dataviz`-Skills | Fehlerarten, Timeout-Befund |
| F6 | (a) nur Erfolge: Verteilungskurve der Maximalkraft mit Abbruchgrenze aus `episodes_meta.json`; (b) „Anteil erfolgreich bis Schritt k" über ALLE Episoden — zeigt auch, ob No-DR bei Schritt 256 noch zulegt (deutet an, beweist nicht) | Belastung, Dauer, Zeitfenster |

Nachträglich, NACH Kenntnis der Ergebnisse ergänzt (2026-09-23) — beschreibend, ohne Kriterium, ändern
K1–K3 nicht:

| # | Inhalt | Vorbild |
|---|---|---|
| F7 | Lernkurven (Trainings-Erfolgsrate je PPO-Iteration), Seeds dünn, Mittel dick; jede Methode auf IHRER Trainingsverteilung gemessen | Henderson 2018, Peng 2018 |
| F8 | AutoDR-Grenzen über dem Training, 0 = Mitte, 1 = Maximum, 5 Seeds | OpenAI 2019 (ADR) |
| F9 | Heatmap Versatz × Kippung, T-final, Seeds zusammengefasst, Rate und Zeilenzahl je Zelle; Achsenpaar per Regel VOR Ansicht festgelegt | Mehta 2019 Fig. 5 |
| F10 | Erfolgsrate, wenn nur Episoden mit maximaler Durchdringung < τ zählen; t1 aus `episodes_meta.json` markiert | IndustReal 2023 Fig. 3 |

Vor dem ersten Grafik-Code: Skill `dataviz` laden.

**Reproduzierbarkeit, geprüft:**
- Fester Bootstrap-Seed; `eval_report.py` zweimal → `summary.json` bytegleich.
- `report_manifest.json`: SHA-256 jeder Eingabedatei, Bibliotheks-Versionen, Git-Commit.
- `--self-test` auf einem kleinen künstlichen Ergebnisordner mit bekannter Antwort
  (z. B. Methode A 90/100, B 50/100 → bekannte Wilson-Grenzen, Differenz 40 pp).
  Mutationen (D-080): Nenner n−1; Welch mit gleichen Varianzen (`equal_var=True`);
  Bootstrap ohne `method` (fällt auf BCa); `success_strict` statt `success_train`.
- Jede Zahl im Thesis-Text verweist auf einen Schlüssel in `summary.json`.

## 9. Optionale Verbesserungen

| # | Was | Aufwand | Gewinn |
|---|---|---|---|
| O1 | Substep-Kraftmaximum (Zeile E) | ½ Tag + 1 Übungslauf | beantwortet die Spitzen-Frage mit einer Messung statt einer Grenze |
| O2 | Wiederholbares Rauschen je Testfall | ca. 1 Tag | engerer Paarvergleich; G6 trennt PhysX von Rauschen |
| O3 | Zweite finale Tabelle (Seed 11), 1000 Zeilen | 10 Läufe | halbiert die Varianz der Hauptzahl; prüft Tabellen-Abhängigkeit |
| O4 | Rasterkarten (Laufplan Schritt 9) | groß (Stunden je Karte) | Bild der Grenzen in 2D; nicht nötig für die Kernaussage |
| O5 | Weglass-Evals (eine Größe auf Mitte) | 5 Läufe je Policy | zeigt, welche Störung am meisten kostet |
| O6 | T-grenze für alle zehn Policies | 9 Läufe mehr | Grenzverhalten auch im Methodenvergleich |

## Verifikation (Ende-zu-Ende)

Laptop: je geänderte Datei nur ihr Check (`make_eval_table.py --self-test`, `check_autodr.py`,
`check_env_wiring.py`, `check_eval_conditions.py --self-test`, `eval_stats.py --self-test`).
Trainings-PC (PowerShell, über `.\scripts\rt_log.ps1 RT-<N> …`): Übungslauf AutoDR Seed 1 auf Seed 20
→ G1–G8 (ohne G7) aus Dateien lesen (`episodes.csv`, `applied.csv`, `eval_manifest.json`) → `/rt-check` gegen
die vorher geschriebene `rt_logs/RT-<N>_expectation.md`.
