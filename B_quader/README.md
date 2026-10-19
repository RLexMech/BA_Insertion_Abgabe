# Experiment B – Quader

Quader 30 × 30 × 50 mm, Tasche 32 mm; je Episode ±2 cm Versatz, 0–10° Neigung und ±45° Gier.
Die Beurteilung in der Arbeit erfolgte durch Sichtprüfung.

## 1. Starten

Den ganzen Block in der Anaconda PowerShell auf dem Trainings-PC ausführen.
Beim Wechsel zwischen A und B erneut ausführen: Beide verwenden das Paket `proxytask`.

```powershell
conda activate env_isaaclab
cd "C:\Users\Simon\Desktop\Alexander_Pett\Abgabe\BA_Insertion_Abgabe\B_quader"
python -m pip install --no-deps -e source/proxytask
Remove-Item Env:PROXYTASK_* -ErrorAction SilentlyContinue
```

## 2. Originaldateien einmalig übernehmen

Nur nötig, solange Modell und Assets noch nicht in der Abgabe liegen:

```powershell
python scripts/import_original.py
```

Das Script kopiert aus `proxytask_gen/generalisation-angle-probe` den Lauf `2026-08-16_15-12-40_peg30mm_pocket32mm_offset20mm_tilt0-10deg_yaw+-45deg` und die beiden Hauptassets `ur10e_peg_s30.usd` und `tisch_square_b32.usd`. Vorhandene Parameter, Metriken und Asset-Prüfberichte werden mitkopiert. Die Originale bleiben unverändert; SHA-256 und Herkunft stehen anschließend unter `models/2026-08-16_15-12-40_peg30mm_pocket32mm_offset20mm_tilt0-10deg_yaw+-45deg/HERKUNFT.json`.

## 3. Abspielen

```powershell
python -u scripts/rsl_rl/play.py --task Template-Proxytask-Direct-v0 --kit_args=--/app/vulkan=false --num_envs 4 --seed 1 --real-time --max-steps 1500 --checkpoint "models/2026-08-16_15-12-40_peg30mm_pocket32mm_offset20mm_tilt0-10deg_yaw+-45deg/model_7093.pt"
```

Vier Roboter, automatisches Ende nach 1500 Steuerschritten. Für dauerhafte Wiedergabe `--max-steps 1500` weglassen. Wiedergabemetriken landen im Unterordner `replay`; die ursprünglichen Trainingsmetriken bleiben erhalten.

Voraussetzung ist die vorhandene Umgebung mit Isaac Sim 5.1, Isaac Lab 2.3.2.post1 und RSL-RL 3.0.1. Kein ONNX-Export oder Neutraining ist für die Wiedergabe nötig. Die GUI-Wiedergabe dieses Abgabestands ist noch am Trainings-PC zu bestätigen.
