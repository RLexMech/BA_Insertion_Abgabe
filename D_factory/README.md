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

In der ersten Zeile den Lauf eintragen:

| Methode | Lauf |
|---|---|
| Auto-DR | `RT-242s1` bis `RT-242s5` |
| Fixed-DR | `RT-243s1` bis `RT-243s5` |
| No-DR | `RT-244s1` bis `RT-244s5` |

```powershell
$lauf = "RT-242s2"
$modus = @{ "RT-242" = "autodr"; "RT-243" = "fixed_dr"; "RT-244" = "off" }[$lauf.Substring(0, 6)]
$original = (Get-ChildItem ".\models\*__${lauf}_*\nn\last_ur5e_insertion_ep_100_rew_*.pth" -File)[0].FullName
$modell = ".\models_view\$lauf\nn\ep100_view.pth"
python -B scripts/prepare_view.py checkpoint "$original" "$modell"
```

## 3. Ansehen wie im Training

```powershell
python -u scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --kit_args=--/app/vulkan=false --num_envs 4 --seed 1 --max-steps 1500 --autodr-state --checkpoint "$modell" "env.dr_mode=$modus" env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

## 4. Ansehen unter den Testbedingungen der Arbeit

Einmal vorbereiten:

```powershell
python -B scripts/prepare_view.py table scripts/eval/eval_table_seed10_c18.csv results/view_tables/final_r0_n4.csv --first-row 0 --count 4
python -B scripts/prepare_view.py table scripts/eval/eval_table_ood_seed10_c18.csv results/view_tables/ood_r0_n4.csv --first-row 0 --count 4
$test = @("-u", "scripts/rl_games/play.py", "--task", "Ur5e-Insertion-Direct-v0", "--kit_args=--/app/vulkan=false", "--num_envs", "4", "--seed", "100", "--max-steps", "1500", "--checkpoint", "$modell", "env.dr_mode=table", "env.obs_wrench_mode=force_prevact", "env.start_floor_m=0.020")
```

Dann eine Bedingung wählen:

```powershell
# Prüfbereich (T-final)
python @test "env.dr_table_path=results/view_tables/final_r0_n4.csv" env.dr_table_width=target env.pocket_obs_yaw_rad=0.0223402

# Nennbedingungen (T-nominal)
python @test "env.dr_table_path=results/view_tables/final_r0_n4.csv" env.dr_table_width=nominal env.pocket_obs_yaw_rad=0.0223402

# Positionsrauschen verdoppelt (pn2)
python @test "env.dr_table_path=results/view_tables/final_r0_n4.csv" env.dr_table_width=target env.pocket_obs_yaw_rad=0.0223402 env.obs_noise_pocket_pos_std_m=0.01

# Greifversatz verdoppelt (gx2)
python @test "env.dr_table_path=results/view_tables/final_r0_n4.csv" env.dr_table_width=target env.pocket_obs_yaw_rad=0.0223402 env.grasp_obs_offset_x_m=0.006

# Gierfehler verdoppelt (py2)
python @test "env.dr_table_path=results/view_tables/final_r0_n4.csv" env.dr_table_width=target env.pocket_obs_yaw_rad=0.0446804

# Außerhalb des Prüfbereichs (T-ood)
python @test "env.dr_table_path=results/view_tables/ood_r0_n4.csv" env.dr_table_width=target env.dr_table_allow_outside=True env.pocket_obs_yaw_rad=0.0223402
```

## 5. Selbst trainieren

Ein Lauf dauert etwa 2 h 15 min. Einen der drei Blöcke wählen:

```powershell
# Auto-DR
$env:RT_NAME = "RT-242s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

```powershell
# Fixed-DR
$env:RT_NAME = "RT-243s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

```powershell
# No-DR
$env:RT_NAME = "RT-244s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Den eigenen Lauf danach ansehen: diesen Block statt Schritt 2, dann Schritt 3 oder 4.

```powershell
$lauf = "RT-242s1"
$modus = @{ "RT-242" = "autodr"; "RT-243" = "fixed_dr"; "RT-244" = "off" }[$lauf.Substring(0, 6)]
$original = (Get-ChildItem ".\logs\rl_games\ur5e_insertion\*__${lauf}_*\nn\last_ur5e_insertion_ep_100_rew_*.pth" -File | Sort-Object LastWriteTime -Descending)[0].FullName
$modell = ".\models_view\eigen_$lauf\nn\ep100_view.pth"
python -B scripts/prepare_view.py checkpoint "$original" "$modell"
```

## 6. Evaluation der Arbeit wiederholen (ohne Fenster)

```powershell
.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-801
.\scripts\rl_games\run_eval.ps1 -Block T-final-F -Rt RT-802
.\scripts\rl_games\run_eval.ps1 -Block T-nominal -Rt RT-803
.\scripts\rl_games\run_eval.ps1 -Block T-nominal-F -Rt RT-804
.\scripts\rl_games\run_eval.ps1 -Block T-ood -Rt RT-805
.\scripts\rl_games\run_eval.ps1 -Block T-final -Noise pn2 -Rt RT-806
.\scripts\rl_games\run_eval.ps1 -Block T-final -Noise gx2 -Rt RT-807
.\scripts\rl_games\run_eval.ps1 -Block T-final -Noise py2 -Rt RT-808
.\scripts\rl_games\run_eval.ps1 -Block T-final-F -Noise pn2 -Rt RT-809
.\scripts\rl_games\run_eval.ps1 -Block T-final-F -Noise gx2 -Rt RT-810
.\scripts\rl_games\run_eval.ps1 -Block T-final-F -Noise py2 -Rt RT-811
.\scripts\rl_games\run_eval.ps1 -Block T-auswahl -Rt RT-812
.\scripts\rl_games\run_eval.ps1 -Block T-grenze -Final fixeddr_s4 -Rt RT-813
.\scripts\rl_games\run_eval.ps1 -Block technik -Rt RT-814
```

Ergebnisse: `results/eval/<Block>/<Methode>_s<Seed>/`.
