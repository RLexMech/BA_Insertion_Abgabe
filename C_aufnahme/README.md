# Experiment C

Befehle aus `C_aufnahme` mit eingerichteter Isaac-Lab-Umgebung, installiertem
Paket `source/insertion` und den benötigten Szenen-Assets ausführen.

## Auto-DR starten

```powershell
python scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=autodr
```

## Fixed-DR starten

```powershell
python scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=fixed_dr
```

## No-DR starten

```powershell
python scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=off
```

## Modell aufrufen

Den Modellpfad ersetzen. Beispiel einer Wiedergabe ohne Domain Randomization.

```powershell
python scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --seed 1 --max-steps 300 --checkpoint "C:\Pfad\zum\modell.pt" env.dr_mode=off
```
