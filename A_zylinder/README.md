# Experiment A – Zylinder

Befehle aus `A_zylinder` mit eingerichteter Isaac-Lab-Umgebung, installiertem
Paket `source/proxytask` und den benötigten Szenen-Assets ausführen.

## Training starten

```powershell
python scripts/rsl_rl/train.py --task Template-Proxytask-Direct-v0 --headless --seed 1
```

## Modell aufrufen

Den Modellpfad ersetzen.

```powershell
python scripts/rsl_rl/play.py --task Template-Proxytask-Direct-v0 --num_envs 4 --checkpoint "C:\Pfad\zum\modell.pt"
```
