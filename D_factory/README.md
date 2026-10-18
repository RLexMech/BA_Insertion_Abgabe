# Experiment D

Alle Blöcke nacheinander in die Anaconda PowerShell einfügen.

## 1. Starten

```powershell
conda activate env_isaaclab
cd "C:\Users\Simon\Desktop\Alexander_Pett\Abgabe\BA_Insertion_Abgabe\D_factory"
python -m pip install -e source/insertion
git lfs pull
```

## 2. Modell wählen

In der ersten Zeile den Lauf eintragen: `RT-242s1` bis `s5` (Auto-DR),
`RT-243s1` bis `s5` (Fixed-DR), `RT-244s1` bis `s5` (No-DR).

```powershell
$lauf = "RT-242s2"
$modus = @{ "RT-242" = "autodr"; "RT-243" = "fixed_dr"; "RT-244" = "off" }[$lauf.Substring(0, 6)]
$original = (Get-ChildItem ".\models\*__${lauf}_*\nn\last_ur5e_insertion_ep_100_rew_*.pth")[0].FullName
$modell = ".\models_view\$lauf\nn\ep100_view.pth"
python -B scripts/prepare_view.py checkpoint "$original" "$modell"
```

## 3. Ansehen wie im Training

```powershell
python -u scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --kit_args=--/app/vulkan=false --num_envs 4 --seed 1 --max-steps 1500 --autodr-state --checkpoint "$modell" "env.dr_mode=$modus" env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

## 4. Ansehen im Haupttest der Arbeit

```powershell
python -B scripts/prepare_view.py table scripts/eval/eval_table_seed10_c18.csv results/view_tables/haupttest.csv --count 4
python -u scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --kit_args=--/app/vulkan=false --seed 100 --max-steps 1500 --checkpoint "$modell" --dr-table results/view_tables/haupttest.csv env.obs_wrench_mode=force_prevact env.start_floor_m=0.020 env.pocket_obs_yaw_rad=0.0223402
```

Verdoppelte Beobachtungsfehler: im zweiten Befehl ändern.

| Test | Änderung |
|---|---|
| Lagefehler verdoppelt | `env.obs_noise_pocket_pos_std_m=0.01` anhängen |
| Greifversatz verdoppelt | `env.grasp_obs_offset_x_m=0.006` anhängen |
| Gierfehler verdoppelt | `env.pocket_obs_yaw_rad=0.0446804` |

## 5. Selbst trainieren

Dauer etwa 2 h 15 min.

```powershell
$env:RT_NAME = "RT-242s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Fixed-DR: `RT-243s1` und `env.dr_mode=fixed_dr`. No-DR: `RT-244s1` und `env.dr_mode=off`.
Ansehen: in Schritt 2 `.\models\` durch `.\logs\rl_games\ur5e_insertion\` ersetzen.

## 6. Evaluation der Arbeit wiederholen

Haupttest von Auto-DR und No-DR (10 Modelle), ohne Fenster:

```powershell
.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-801
```

Jeder Aufruf braucht eine neue `-Rt`-Nummer. Die anderen Tests der Arbeit:

| Test | `-Block` |
|---|---|
| Haupttest Fixed-DR | `T-final-F` |
| Test ohne Streuung | `T-nominal`, `T-nominal-F` |
| Test außerhalb des Prüfbereichs | `T-ood` |
| Test mit verdoppeltem Beobachtungsfehler | `T-final` und `T-final-F`, je mit `-Noise pn2`, `gx2`, `py2` |

Ergebnisse: `results/eval/`.
