# Experiment A – Zylinder

Den ganzen Block in der Anaconda PowerShell ausführen:

```powershell
conda activate env_isaaclab
cd "C:\Users\Simon\Desktop\Alexander_Pett\Abgabe\BA_Insertion_Abgabe\A_zylinder"
git lfs pull
python -m pip install --no-deps -e source/proxytask
Remove-Item Env:PROXYTASK_* -ErrorAction SilentlyContinue
python -u scripts/rsl_rl/play.py --task Template-Proxytask-Direct-v0 --kit_args=--/app/vulkan=false --num_envs 4 --seed 1 --real-time --max-steps 1500 --checkpoint "models/2026-07-27_03-16-21_d28_d28_scratch/model_599.pt"
```

Vier Roboter, automatisches Ende nach 1500 Steuerschritten. Für dauerhafte Wiedergabe `--max-steps 1500` weglassen.
