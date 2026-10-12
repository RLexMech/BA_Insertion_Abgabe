# RT-243s1..s5 -- expectation, written BEFORE the run (2026-09-28): the Fixed-DR arm, five seeds, 100 epochs, on D-204's code (rl_games)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Templates: `rt_logs/RT-242_expectation.md`, section RT-242s1..s5 (the
AutoDR arm, same command except `env.dr_mode`), and
`rt_logs/RT-231_expectation.md` (the Fixed-DR arm on rsl_rl). Unchanged
sections point there; only what differs is written out.

## Context

D-192 / D-195: three training arms (AutoDR, No-DR, Fixed-DR), five seeds
each, the same budget. RT-242s1..s5 is the AutoDR arm on D-204's code. This
chain is the Fixed-DR arm: `dr_mode=fixed_dr` (`autodr.FixedDR`,
`autodr.py:1360`; wired `insertion_env.py:578-592`): AutoDR's start-floor
phase line for line, then every quantity at its `FixedWidth.at_max` width,
drawn uniformly (no boundary episodes), to the end. Two differences to
AutoDR, not one (`eval_plan.md` 3.1): the width does not grow, and no
boundary sampling.

New against RT-231 (the last Fixed-DR arm): rl_games instead of rsl_rl
(D-200), 100 epochs instead of 1500 iterations (D-201), D-204's task
(start ceiling 0.270 m, fixture spread 0.05 m in x, y and z, home 0.415 m),
LSTM + a_(t-1) (D-195). RT-231 is a hint only, not a bar: s1 and s2 there
never learned (`VERDICTS.md` 2026-09-24).

**Gate before the start (user's choice, not code):** D-195 binds Fixed-DR to
the AutoDR arm's budget. If RT-242s falsifies D-201 (a seed at `all_at_max`
after epoch 80, or still rising at 100) the user may change the budget, and
this chain with 100 epochs would no longer be the same budget. Read
RT-242s P3/P5 first, then start this chain.

## Code state

Clone `Phase5_RlGames_v1`, branch `p5-rlgames`; the SHA is the commit that
adds this file, read back from `origin/p5-rlgames` and given in chat.
Against the RT-242s commit: this file only (no code change). `CODE_MARKER`
= `insertion-osc-2026-09-03a` (`insertion_env.py:76`), the same as RT-242s
-- the marker does NOT tell the arms apart. The gate is the git line AND the
provider line AND the run folder (P1).

New under Isaac on THIS path: `FixedDR` under rl_games (its state goes into
the checkpoint through `RlGamesGpuEnvAutoDR.get_env_state`,
`scripts/rl_games/train.py:185-190`, inner format `fixed-dr-1`,
`autodr.FixedDR.FORMAT`), and `FixedDR` with the 0.270 m ceiling. Offline:
`check_autodr.py` covers FixedDR (floor equal to AutoDR, at_max bounds, no
boundary env, state round trip; `check_autodr.py:1270-1330`).

## The one hypothesis

**On D-204's code under rl_games, every Fixed-DR seed runs 100 epochs with
exit 0, its floor reaches H_min, and from that moment the five quantities
stay at full width: no `[autodr]` line for one of the seven boundaries
after the transition, `Episode/dr/bounds_version` constant and
`Episode/dr/all_at_max` 1 to epoch 100.**

Falsified by: a Traceback or a missing epoch 100; no `reached H_min, phase
dr` line; an `[autodr]` line for `lat_r_hi`, `yaw_lo`, `yaw_hi`, `tilt_hi`,
`start_height_hi`, `friction_lo` or `friction_hi`; `bounds_version` moving
after `Episode/dr/phase` turned 1.

The success rate is NOT part of the hypothesis. Whether Fixed-DR learns
this task in 100 epochs is unknown; it is reported (P5).

## Command

Training PC, PowerShell, `env_isaaclab` active, GPU free, AFTER the RT-242s
zip is made. Pull first; the expected SHA is given in chat.

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
git pull
git rev-parse HEAD
```

The five seeds as one chain (RT-242s1..s5 with ONLY `env.dr_mode=autodr`
-> `env.dr_mode=fixed_dr` and the RT name):

```powershell
.\scripts\rt_log.ps1 RT-243s1 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-243s2 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 2 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-243s3 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 3 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-243s4 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 4 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-243s5 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 5 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Wall time: unknown for this arm; RT-242s's own epoch time is the hint once
it is read. RT-243s1 is its own smoke: look at its log once `dr/phase`
turned 1 and stop the chain on a Traceback.

After the chain, ONE zip (before any replay of these runs):

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
foreach ($n in 1..5) {
  $t = "RT-243s$n"
  $dirs = @(Get-ChildItem -Directory "logs\rl_games\ur5e_insertion\*__${t}_*")
  if ($dirs.Count -ne 1) { Write-Host "$t : expected ONE run folder, found $($dirs.Count)"; continue }
  $run = $dirs[0].FullName
  python scripts/check_episode_log.py --run-dir $run | Out-File -Encoding utf8 "rt_logs\${t}_eplog.txt"
  foreach ($f in "scalars.csv", "demo_metrics.json", "episodes.csv", "episodes_meta.json") { if (Test-Path "$run\$f") { Copy-Item -LiteralPath "$run\$f" "rt_logs\${t}_$f" } }
  Copy-Item -LiteralPath "$run\params\agent.yaml" "rt_logs\${t}_agent.yaml"
  Copy-Item -LiteralPath "$run\params\env.yaml" "rt_logs\${t}_env.yaml"
  Get-ChildItem -LiteralPath "$run\nn" | Select-Object Name, Length | Out-File -Encoding utf8 "rt_logs\${t}_nn.txt"
}
$files = (Get-ChildItem rt_logs\RT-243s*).FullName
New-Item -ItemType Directory -Force C:\Users\Simon\Desktop\Alexander_Pett\Zips | Out-Null
Compress-Archive -LiteralPath $files -DestinationPath C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-243s.zip -Force
Get-Item C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-243s.zip
```

## Points -- every seed (N = 1..5)

* **P1 setup** (the gate; any line wrong = void, precedent RT-151b): as
  RT-242s P1, with these lines changed: `provider: FixedDR, 8 boundaries,
  bounds_version 0` (`insertion_env.py:5062`); `dr_mode: fixed_dr`; run
  folder
  `logs\rl_games\ur5e_insertion\<mm-dd_HH-MM-SS>__RT-243sN_fixed_dr_offset50mm_currOFF_start+20mm_floor-30mm_seedN`
  (tag code `scripts/rl_games/train.py:121-168`). A folder or provider line
  with `autodr` = wrong run, nothing below is judged.
* **P1b** as RT-242s P1b.
* **P2 console block**: as RT-242s P2 (`Learning iteration 1/100` ..
  `100/100`, zero `block NOT printed` lines).
* **P3 fixed width** (the hypothesis), from `RT-243sN_scalars.csv` on the
  epoch axis and the console: exactly one `reached H_min, phase dr` line
  (`autodr.py:965`) and the first epoch with `Episode/dr/phase` 1
  (reported); after it no `[autodr]` line for one of the seven;
  `Episode/dr/bounds_version` constant from that epoch to 100 (value
  reported; RT-231: 5, a hint); `Episode/dr/all_at_max` 1 from that epoch
  on; every `Episode/dr/<key>` of the seven at its maximum: `lat_r_hi`
  0.030, `yaw_lo` -0.0873, `yaw_hi` 0.0873, `tilt_hi` 0.1745,
  `start_height_hi` 0.270, `friction_lo` 0.08, `friction_hi` 0.20
  (`autodr.DR_DIMS`, `autodr.py:449-463`); every `Episode/dr/<key>_fill` of
  the seven 0 for the whole run (`autodr.py:1299`).
* **P4 learning**: as RT-242s P4.
* **P5 flat at 100?** as RT-242s P5 (D-201: "a No-DR or Fixed-DR arm still
  rising at its end" is its own falsifier), plus `Episode/success_rate` at
  epoch 100. Reported beside RT-242sN of the same seed -- a hint, no bar.
* **P6 end**: as RT-242s P6 (`MAX EPOCHS NUM!`, the five `.pth`, an
  `[autodr-state] saved into the checkpoint` line at every save,
  `[scalars] wrote`, `[rt_log] exit code: 0`).
* **P7 episode record**: as RT-242s P7.
* **P8 D-204 readings**: as RT-242s P8.

## Pins (read from the files, not from the log prose)

* `RT-243sN_agent.yaml`: the RT-242s pins, unchanged (`params.seed: N`,
  `config.max_epochs: 100`, `config.save_frequency: 25`, ...).
* `RT-243sN_env.yaml`: `dr_mode: fixed_dr`; everything else as the RT-242s
  pins. Against `RT-242sN_env.yaml` of the same seed the diff is ONLY
  `dr_mode` and the log folder (`eval_plan.md` 3.1 fairness rule).
* `RT-243sN_demo_metrics.json`: `dr_mode: fixed_dr`; `dr_state.mode`:
  `fixed_dr` (`FixedDR.as_dict`, `autodr.py:1403-1406`);
  `fixture_pos_noise_xy_m: 0.05`, `fixture_pos_noise_z_m: 0.05`,
  `home_tip_above_entrance_mm: 415.0`.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| 100 epochs, exit 0, P3 holds | hypothesis holds for this seed | next seed |
| an `[autodr]` line for one of the seven, or `bounds_version` moves after the transition | FixedDR does not hold its width; the arm is not Fixed-DR | stop the chain, diagnose; no seed counts |
| `dr/phase` 0 through 100 | the floor phase never finished; the seed never trained at full width | report; K5 reads it as a Fixed-DR outcome |
| success stays low at full width | full width from H_min on may be too hard for this budget -- the DORAEMON pattern | report; this IS a result for K5, not a fault |
| P5 still rising at 100 | D-201's Fixed-DR clause | report; user decides |
| Traceback or NaN | not a result | stop the chain; hand the Traceback back |

## What this run cannot answer

* K5 itself: no rl_games evaluation path (D-200 plan, "NOT BUILT").
* Whether growth or boundary sampling causes a difference to AutoDR --
  Fixed-DR changes both at once (`eval_plan.md` 3.1).
* Whether the training success rate is a test success rate.
* The No-DR arm (D-195): not in this chain.
* Whether RT-243sN and RT-242sN are equal in the floor phase beyond what
  the csv rows show (GPU determinism not established).

## OUTCOME (2026-09-29, files `rt_logs/RT-243s/`)

* P1 all five: git `9219ab0` (the chat SHA), import from
  `Phase5_RlGames_v1`, marker, `provider: FixedDR, 8 boundaries,
  bounds_version 0`, `dr_mode: fixed_dr`, folders
  `..._RT-243sN_fixed_dr_offset50mm_currOFF_start+20mm_floor-30mm_seedN`.
  P1b: 0.247..0.290 deg. PASS.
* P2: 100 blocks each, zero `block NOT printed`. PASS.
* **P3 (the hypothesis): HOLDS, all five.** One `reached H_min` line each;
  `dr/phase` 1 from epoch 13 / 14 / 16 / 14 / 16; after it
  `bounds_version` 5 constant to 100, `all_at_max` 1 in every epoch, the
  seven bounds at their maximum (0.03, -0.0873, 0.0873, 0.1745, 0.270,
  0.08, 0.20), every `_fill` 0 over the whole run; zero `[autodr]` lines
  for the seven boundaries (only `start_height_floor`, 7-8 lines).
* P4: zero non-finite cells; `last_lr` 1.3e-5..2.25e-4, never at a bound;
  `cval_lr` 1e-4 throughout.
* P5 (means 81..90 -> 91..100) and success at 100:
  s1 `success_rate` 0.000 -> 0.000, length 254.9 -> 255.0, reward 49.1 -> 49.3;
  s2 0.000 -> 0.000, 254.7 -> 255.0, 47.0 -> 47.4;
  s3 0.9921 -> 0.9922, 58.5 -> 56.6, 524.2 -> 533.2;
  s4 0.9957 -> 0.9963, 51.4 -> 48.5, 547.1 -> 559.0;
  **s5 0.0053 -> 0.0286 (0.056 at 100), 253.5 -> 249.7, 53.7 -> 70.2: still
  rising at its end** -- D-201's Fixed-DR clause; D-205 reads it now.
  Epochs 90..99 outcomes (`episodes.csv`): s1 1285 timeout; s2 1281
  timeout, 3 force_abort; s3 5803 success; s4 6725 success; s5 1209
  timeout, 101 success. s1/s2 at 100: `mean_max_depth_mm` 0.02 / 0.0,
  `engaged` 0 -- they stay out of the pocket until the time limit.
  s3 / s4 first reach 0.9 at epoch 54 / 48.
* P6: 100/100, `MAX EPOCHS NUM!`, ep 25/50/75/100 + `ur5e_insertion.pth`,
  exit 0, no Traceback. PASS.
* P7: all five `VERDICT: COMPLETE`, rollout 128. PASS.
* Pins: `env.yaml` against RT-242sN differs ONLY in `dr_mode`, `seed` and
  `log_dir` lines (fairness rule holds); `agent.yaml` only in
  `full_experiment_name`.
* Wall time 1.26 / 1.26 / 2.08 / 2.10 / 1.53 h (`wall_time`, last minus
  first row); the seeds that fail are the fast ones, cause not measured.
* Reading: table row "success stays low at full width" for s1, s2, s5 --
  a K5 result, not a fault. RT-231 (rsl_rl, other setup) had the same
  seeds s1 and s2 at 0.
