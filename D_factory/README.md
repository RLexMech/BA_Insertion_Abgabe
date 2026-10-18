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
$original = (Get-ChildItem ".\models\*__${lauf}_*\nn\last_ur5e_insertion_ep_100_rew_*.pth" -File)[0].FullName
$modell = ".\models_view\$lauf\nn\ep100_view.pth"
python -B scripts/prepare_view.py checkpoint "$original" "$modell"
```

Ein eigener Lauf aus Schritt 5: in der dritten Zeile `.\models\` durch
`.\logs\rl_games\ur5e_insertion\` ersetzen.

## 3. Ansehen wie im Training

```powershell
python -u scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --kit_args=--/app/vulkan=false --num_envs 4 --seed 1 --max-steps 1500 --autodr-state --checkpoint "$modell" "env.dr_mode=$modus" env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

## 4. Ansehen unter den Testbedingungen der Arbeit

```powershell
python -B scripts/prepare_view.py table scripts/eval/eval_table_seed10_c18.csv results/view_tables/final_r0_n4.csv --first-row 0 --count 4
python -u scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --kit_args=--/app/vulkan=false --num_envs 4 --seed 100 --max-steps 1500 --checkpoint "$modell" env.dr_mode=table env.dr_table_path=results/view_tables/final_r0_n4.csv env.dr_table_width=target env.obs_wrench_mode=force_prevact env.start_floor_m=0.020 env.pocket_obs_yaw_rad=0.0223402
```

Das zeigt den Prüfbereich (T-final). Andere Bedingungen im zweiten Befehl:

| Bedingung | Änderung |
|---|---|
| Nennbedingungen (T-nominal) | `env.dr_table_width=nominal` |
| Positionsrauschen ×2 (pn2) | `env.obs_noise_pocket_pos_std_m=0.01` anhängen |
| Greifversatz ×2 (gx2) | `env.grasp_obs_offset_x_m=0.006` anhängen |
| Gierfehler ×2 (py2) | `env.pocket_obs_yaw_rad=0.0446804` |
| Außerhalb des Prüfbereichs (T-ood) | Tabelle `eval_table_ood_seed10_c18.csv`, `env.dr_table_allow_outside=True` anhängen |

## 5. Selbst trainieren

Dauer etwa 2 h 15 min.

```powershell
$env:RT_NAME = "RT-242s1"
python scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --seed 1 --max_iterations 100 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Fixed-DR: `RT-243s1` und `env.dr_mode=fixed_dr`. No-DR: `RT-244s1` und `env.dr_mode=off`.
Anderer Seed: Endziffer und `--seed` gemeinsam ändern.

## 6. Evaluation der Arbeit wiederholen (ohne Fenster)

```powershell
.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-801
```

Weitere Blöcke: `T-final-F`, `T-nominal`, `T-nominal-F`, `T-ood`, `T-auswahl`,
`technik`, `T-grenze -Final fixeddr_s4`; Rauschen mit `-Noise pn2|gx2|py2`.
Jeder Aufruf braucht eine neue `-Rt`-Nummer. Ergebnisse: `results/eval/`.
