# Experiment A – Zylinder

Zylinder Ø28 × 50 mm, feste Bohrung Ø30 mm. Ausgewählt ist der Scratch-Lauf vom 27.07.2026 um 03:16.
Die Beurteilung in der Arbeit erfolgte durch Sichtprüfung.

## 1. Starten

Den ganzen Block in der Anaconda PowerShell auf dem Trainings-PC ausführen.
Beim Wechsel zwischen A und B erneut ausführen: Beide verwenden das Paket `proxytask`.

```powershell
conda activate env_isaaclab
cd "C:\Users\Simon\Desktop\Alexander_Pett\Abgabe\BA_Insertion_Abgabe\A_zylinder"
python -m pip install --no-deps -e source/proxytask
Remove-Item Env:PROXYTASK_* -ErrorAction SilentlyContinue
```

## 2. Originaldateien einmalig übernehmen

Nur nötig, solange Modell und Assets noch nicht in der Abgabe liegen:

```powershell
python scripts/import_original.py
```

Das Script kopiert aus `proxytask_gen/proxytask` den Lauf `2026-07-27_03-16-21_d28_d28_scratch` und die beiden Hauptassets `ur10e_peg_d28.usd` und `tisch.usd`. Vorhandene Parameter, Metriken und Asset-Prüfberichte werden mitkopiert. Die Originale bleiben unverändert; SHA-256 und Herkunft stehen anschließend unter `models/2026-07-27_03-16-21_d28_d28_scratch/HERKUNFT.json`.

## 3. Abspielen

```powershell
python -u scripts/rsl_rl/play.py --task Template-Proxytask-Direct-v0 --kit_args=--/app/vulkan=false --num_envs 4 --seed 1 --real-time --max-steps 1500 --checkpoint "models/2026-07-27_03-16-21_d28_d28_scratch/model_599.pt"
```

Vier Roboter, automatisches Ende nach 1500 Steuerschritten. Für dauerhafte Wiedergabe `--max-steps 1500` weglassen. Wiedergabemetriken landen im Unterordner `replay`; die ursprünglichen Trainingsmetriken bleiben erhalten.

Voraussetzung ist die vorhandene Umgebung mit Isaac Sim 5.1, Isaac Lab 2.3.2.post1 und RSL-RL 3.0.1. Kein ONNX-Export oder Neutraining ist für die Wiedergabe nötig. Die GUI-Wiedergabe dieses Abgabestands ist noch am Trainings-PC zu bestätigen.
