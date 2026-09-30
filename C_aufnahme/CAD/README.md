# CAD/ — STEP-Quelldateien der Szene

STEPs werden committet; die daraus erzeugten USDs bleiben gitignored und
entstehen auf dem Trainings-PC.

## Maßgeblich: `Step/Greifer_Bauteil/greifer_bauteil_asm.stp`

Export 2026-08-22 19:45, Creo, `ap214_is`, `Export assembly as: Single File`.
**Das ist die Quelle für Greifer und Fügeteil.** Eine einzige importierbare
Datei mit beiden Körpern, bereits relativ zum Flanschursprung platziert.

Am 2026-08-22 an der Datei selbst nachgerechnet:

| Prüfung | Ergebnis |
|---|---|
| Einheiten | mm |
| `MANIFOLD_SOLID_BREP` | 2 — beide Körper vorhanden |
| `ADVANCED_FACE` | 1409 |
| `NEXT_ASSEMBLY_USAGE_OCCURRENCE` / `ITEM_DEFINED_TRANSFORMATION` | 2 / 2 — Struktur erhalten |
| `GEOMETRIC_SET` | 0 — keine Bezugsachsen als Drahtgeometrie |
| `VOLUME_MEASURE` | 5 — Validation Properties erhalten |

Platzierungen im Flanschsystem, beide gegen die Schwerpunkte der Datei
gegengerechnet (MATCH auf 12 Stellen):

```
FUEGETEIL    t = (0.050, -0.070, 126.746951) mm    R = 180° um Y
GREIFER_V1   t = (0, 0, 0)                         R = X/Z getauscht, Y negiert
```

Ursprung = Flanschanlagefläche, +Z vom Roboter weg (D-068). Die benannten
Koordinatensysteme `GREIFER_MITTE`/`FUEGETEIL_MITTE` sind in diesem Export
nicht mehr enthalten — die Zahlen sind aber identisch zum Export von 18:27,
der sie noch hatte, also hat sich der Ursprung nicht bewegt.

### Import-Einstellungen (HOOPS)

Siehe `step_import_settings.md` in diesem Ordner. Kurz:
`Meters Per Unit = Millimeters`, `Override Up-Axis = Z-up`,
`Convert Curves = aus`, `Instancing Style = None`,
`Composition Style = None`, `Tessellation Level = ExtraHigh`.

Die Tessellation ist kein Nebenschauplatz: das reale Spiel beträgt 0,2 mm auf
der kurzen Achse. Der Tessellationsfehler ist nach der Konversion zu
**messen**, nicht anzunehmen.

## Alle Dateien und ihre Gültigkeit

| Datei | Stand | Bezugssystem | gilt |
|---|---|---|---|
| `Step/Greifer_Bauteil/greifer_bauteil_asm.stp` | 19:45 | Flansch, +Z vom Roboter weg | **maßgeblich** |
| `Aufnahme_real_v1_mm.stp` | 2026-08-24 | Stufe-2-Öffnungsebene, siehe unten | **maßgeblich für den Import** |
| `Aufnahme_real_v1.stp` | 11:42 | dito | Original-Export, **Einheiten-Etikett falsch** — siehe unten |
| `Step/fuegeteil_prt.stp` | 18:27 | Teilesystem, 180° um Y zur Baugruppe | Archiv — Geometrie steckt in der Baugruppe |
| `Step/greifer_v1_prt.stp` | 18:34 | Teilesystem, eingedreht | Archiv — dito |
| `Step/Fuegeteil_Neuer.stp` | 18:54 | 180° um X gegen die anderen, mit Drahtgeometrie | **Archiv, nicht verwenden** |
| `fuegeteil.stp` | 16:03 | älteres Teilesystem, mit Drahtgeometrie | **Archiv, nicht verwenden** |
| `Greifer_Standart_V5.stp` | 11:42 | UNGEPRÜFT | überholt durch die Baugruppe |
| `*.prt.*`, `*.asm.*`, `*_out.log.*` | div. | — | Creo-Nativstände und Exportlogs, keine Pipeline-Quelle |

**Achtung:** die Archiv-Dateien liegen in drei verschiedenen Bezugssystemen.
Die verifizierten Transformationen gelten **nur** für die maßgebliche
Baugruppendatei. Wer eine Archivdatei verwendet, muss ihr System neu prüfen.

### Einheiten-Defekt in `Aufnahme_real_v1.stp` (gemessen 2026-08-24)

Der Creo-Export trägt **Millimeter-Zahlen mit einem Zoll-Etikett**. Beides ist
gemessen, nicht vermutet:

- Zeile 2682: `#3129=(CONVERSION_BASED_UNIT('INCH',#3128)...)`, und `#3128` ist
  `LENGTH_MEASURE(2.54E1)`. Zeile 3366 weist genau dieses `#3129` als globale
  Längeneinheit zu. Die Datei erklärt sich also als Zoll.
- Alle 1529 Koordinatenwerte gemessen: Median **45,29382**, das ist die
  Taschenwand ±45,294 mm. Maximum 100,45. Die Zahlen sind Millimeter.
- `Step/Greifer_Bauteil/greifer_bauteil_asm.stp` enthält kein einziges `INCH`.
  Dessen Import lief korrekt (RT-8 … RT-12).

Folge: jeder Import skaliert um Faktor 25,4. Gemessen in RT-27 (Etikett Zoll
übernommen) und RT-30 (3810 × 4803,14 × 1549,4 mm statt 150 × 189,1 × 61).

**Reparatur:** `Aufnahme_real_v1_mm.stp` ist eine Kopie mit **einer** geänderten
Stelle — Zeile 3366 zeigt jetzt auf `#3127`, die Millimeter-Einheit, die Creo
selbst schon in die Datei geschrieben hat. Alle Koordinaten sind bit-gleich.
Das Original bleibt unverändert liegen.

**Nicht mit repariert:** Die Flächen- und Volumeneinheiten (`#3162`, `#3171`,
`#3184`, `#3193`) tragen weiter das Zoll-Etikett. Sie speisen nur die
`geometric validation property`-Einträge, nicht die Geometrie. Wer diese Werte
zitiert, muss sie neu prüfen.

**Der saubere Weg bleibt offen:** ein Neu-Export aus Creo mit Einheit
Millimeter. Dann fällt `Aufnahme_real_v1_mm.stp` weg.

### Aufnahme: nur eine Version

`prt0001aufnahme_real_v2.prt.1` ist **dieselbe** Aufnahme wie
`Aufnahme_real_v1.stp`. Das "v2" im Dateinamen ist kein zweiter Entwurf.
Nutzer-Entscheid 2026-08-24. Die daraus abgeleiteten Konstanten (Wände
±72,55 / ±45,294, `STAGE1_DEPTH`, `INSERTION_PLANE_Z`) gelten damit **ohne
Vorbehalt**.

Nicht davon erfasst: das Spiel auf der langen Achse. Das bleibt eine
Messfrage und steht unverändert offen (HANDOFF-SZENE.md, Open).

| Datei | Was | Herkunft | Einheiten | Ursprung |
|---|---|---|---|---|
| `Aufnahme_real_v1.stp` | Aufnahme-Insert (Tasche), außen 150 × 189,1 × 61 mm | Kopie aus `docs/Geometrie/` (eigenes Creo-Modell, 2026-08-19) | Zahlen mm, Etikett ZOLL — siehe „Einheiten-Defekt“ | Mitte der Stufe-2-Öffnungsebene. **Achszuordnung:** +X = kurze Richtung, Wände bei **±45,294 mm**; +Y = LANGE Richtung, Wände bei **±72,55 mm**, und +Y ist die 7,9-mm-Rand-Seite; +Z = aus der Öffnung heraus nach oben. Das CAD-Frame ist achsgleich mit dem Zellen-Frame (spawnt mit Identitätsrotation). |

## Was aus dem Fügeteil-Solid direkt ablesbar ist

Aus den `geometric validation property`-Einträgen der Datei (Creo-Export,
AP214), also keine Schätzung:

- Volumen des Solids: **416 403,68 mm³**
- Schwerpunkt relativ zum Dateiursprung: **(1,4140 / −0,5173 / −1,7741) mm**

Der Schwerpunkt ist NICHT der Bounding-Box-Mittelpunkt — er bestätigt den
mittigen Ursprung also weder noch widerlegt er ihn, er zeigt nur, dass der
Ursprung nahe der Teilemitte liegt. Die Roh-Punkt-Extraktion der
CARTESIAN_POINTs bleibt wie gehabt unbrauchbar für eine Bounding Box
(Kontrollpunkte + gebundene Hilfsgeometrie sprengen die Hülle: X 282,5 /
Y 170,2 / Z 130,6 mm gegen die tatsächlichen 143,50 × 96,41 × 50,506 mm, aus der ersten USD-Konversion gemessen, D-075 — 96,41 quer inklusive der Nasen).
**Bounding Box und Achszuordnung des Fügeteils bleiben UNKNOWN**, bis die
erste Konversion bzw. ein echtes CAD-Werkzeug sie liefert.

## Zellen-Achsen (D-060, Nutzerentscheidung 2026-08-19)

Gilt für die ganze Szene, nicht nur für die Aufnahme — hier wiederholt, weil
das Fehlen dieser Angabe schon einmal zu einer falschen Achszuordnung geführt
hat (Quelle: `insertion_tasks_cfg.py`, Frame-Block ab Zeile 415):

- **+X** vom Roboter weg zur Aufnahme, also zur Rückwand des Testadapters.
- **+Y** nach LINKS vom Roboter aus. Die Aufnahme steht rechts, ihre laterale
  Mitte ist daher negativ. Das 7,9-mm-Randende zeigt nach +Y.
- **+Z** nach oben.
- Die **lange Achse** der Tasche (145,1 mm) und damit die lange Achse des
  Fügeteils (143,50 mm, D-075) laufen **lateral, also in Y** — nicht in X.

## Quellenrang für das Fügeteil

Weil `fuegeteil.stp` DEHN's eigenes CAD ist, gilt D-057 ungeschmälert: **für
die Teilegeometrie führt CAD, die Messung verifiziert nur.** Konkret geprüft
2026-08-22 an der Gesamthöhe entlang der Fügeachse:

| Quelle | Höhe |
|---|---|
| Original-CAD (aus der Baugruppe, 2 × 25,253) | **50,506 mm** — gilt |
| Produktdatenblatt, 43,5 + 7 | 50,5 mm |
| Messschieber vor Ort, 2026-08-17 | 50,7 mm — überholt, Messfehler |

CAD und Datenblatt decken sich auf 6 µm. Die 0,194 mm des Messschieberwerts
liegen in derselben Größenordnung wie das 0,2-mm-Spiel der kurzen Achse, an
dem das Erfolgskriterium hängt — deshalb notiert statt weggerundet (D-073).

Bekannter offener Konflikt: Stufe-2-Öffnung im CAD 145,1 mm lang vs. 143,84 mm
aus Teil + gemessenem Spiel. Wird vor Ort nachgemessen; bis dahin gilt das CAD.

Die alten Fundorte (`Bild/900360.stp`, `docs/Geometrie/Aufnahme_real_v1.stp`)
bleiben unangetastet liegen; Skripte zeigen auf `CAD/`.
