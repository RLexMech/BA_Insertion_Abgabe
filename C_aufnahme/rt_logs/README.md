# rt_logs — Konsolen-Ausgaben vom Trainings-PC

Der Weg vom Trainings-PC zum Laptop ist von Hand. Absichtlich: Git hat das
früher gemacht und vier Runden gekostet, ohne das eigentliche Problem zu
lösen — der Log soll **nicht ins Kontextfenster** des Hauptchats.

## Trainings-PC

Jeden Befehl über den Wrapper laufen lassen (PowerShell, conda-Env aktiv —
kein `bash`, das ist WSL ohne conda):

    .\scripts\rt_log.ps1 RT-6 python scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless

Der Wrapper schreibt Zeitstempel, Befehl, Commit-Hash, die volle Ausgabe und
`exit code: N` nach `rt_logs/<NAME>.txt` und legt den Inhalt am Ende in die
Zwischenablage. Gleicher NAME hängt an dieselbe Datei an.

Kurzer Test des Wrappers, ohne Isaac:

    .\scripts\rt_log.ps1 PROBE cmd /c echo hallo

## Laptop

1. Text in `rt_logs/inbox.txt` einfügen (Texteditor) und speichern.
2. `/rt-check` aufrufen. Ohne Argument nimmt die Skill `rt_logs/inbox.txt`;
   ein Pfad als Argument überschreibt das.

Die Skill holt den Kopf mit `filter_rt_log.py --head` (Zeitstempel, Befehl,
Commit) und schreibt die Erwartung hin, BEVOR der Log angesehen wird.

Dann fragt sie das Skript nach den Belegzeilen — je Erwartungspunkt ein
`--expect "<punkt>"`. Der Rohtext wird nie gelesen; zurück kommen nur die
wenigen Originalzeilen mit `L<nr>`, dazu Exit-Code, Anomalien und Marker.
Geurteilt wird Punkt für Punkt: ERFÜLLT / VERLETZT / NICHT BELEGT. Das
Urteil landet als eine Zeile in `VERDICTS.md`.

Die Ausgabe ist gedeckelt und wächst nicht mit dem Log: ein 177-kB-Dump
liefert dieselben ~15 Zeilen wie ein 8-kB-Dump. Einen Subagenten gibt es
seit 2026-08-24 nicht mehr — seine Grundlast war ein Vielfaches des Logs.

## Was hier versioniert ist

Nur `README.md` und `VERDICTS.md`. Die Log-Dateien selbst sind gitignored und
werden nicht aufbewahrt — `VERDICTS.md` ist der Nachweis, was gelaufen ist
und wie es beurteilt wurde.
