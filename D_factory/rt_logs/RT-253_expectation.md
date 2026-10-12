# RT-253 -- expectation, written BEFORE the run (2026-10-01): Fixed-DR s1, s2, s5 resumed from `ep_100` to epoch 250 (D-209, exploratory)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Template: `rt_logs/RT-243_expectation.md` (the same arm, seeds and
overrides; OUTCOME). Only what differs is written out.

## Context

D-209 (user 2026-10-01). RT-243s1, s2 never entered the pocket in 100
epochs (success 0, `mean_max_depth_mm` 0.02 / 0.0 at 100); s5 was still
rising (success 0.0053 -> 0.0286 over epochs 81..90 -> 91..100, 0.056 at
100). The evaluation confirmed it (T-final-F .000 / .000 / .078). This run
asks: do the SAME seeds learn with 150 more epochs? Exploratory; outside
K1-K5 and the final-policy selection (D-205, D-208).

## Code state

**NO pull.** Training PC on `c1a635e969f72efd5b9062d2c8994befd29ff100`
(the evaluation clone; RT-252 is not started). Between RT-243's `9219ab0`
and `c1a635e9` the training path under `dr_mode=fixed_dr` is unchanged:
`git diff 9219ab0..c1a635e9 -- source/ scripts/rl_games/train.py` touches
only table mode (`allow_outside`, D-206 (3)). Gates as RT-243 P1: git line
`c1a635e`, `imported from: ...Phase5_RlGames_v1\...`, `provider: FixedDR,
8 boundaries, bounds_version 0`, `dr_mode: fixed_dr`.

Resume mechanics (read in code, not run):
* `--checkpoint <path>` -> rl_games `load_checkpoint`
  (`scripts/rl_games/train.py:317-320`); `restore` sets `epoch_num` and
  `frame` from the file (`a2c_common.py:635-640` at `6b3534f2`), loads the
  optimizer and hands `env_state` to `set_env_state`
  (`train.py:193-215`): same seed required, else `SystemExit`.
* `--max_iterations 250` -> `max_epochs` 250 (`train.py:314-316`); the loop
  stops at `epoch_num >= max_epochs` (`a2c_common.py:1134`) -> epochs
  101..250.
* `last_lr` restarts at the yaml `learning_rate` 1e-4 (`a2c_common.py:263`,
  not in the checkpoint). Named in D-209, not changed.

## The one hypothesis

**150 more epochs rescue only the seed that had begun to learn: at epochs
241..250 the training `Episode/success_rate` mean of s5 is above its
91..100 mean 0.0286, while s1 and s2 stay at 0.000 (their 91..100 value).**

The two reference values are RT-243's own measurements (OUTCOME, P5). Any
other combination falsifies it; each seed is reported on its own.

## Command (training PC, `env_isaaclab` active, GPU free)

Pre-check (runs nothing):

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
git rev-parse HEAD
git status --short --untracked-files=no
$L='logs\rl_games\ur5e_insertion'; foreach ($s in 1,2,5) { $c=@(Get-ChildItem "$L\*__RT-243s${s}_*\nn" -File | Where-Object { $_.Name -match '^last_ur5e_insertion_ep_100_rew_[-+0-9.eE]+\.pth$' }); "s$s found $($c.Count): $($c.FullName)"; "s$s old RT-253 folders: $(@(Get-ChildItem "$L\*__RT-253s${s}_*" -Directory -ErrorAction SilentlyContinue).Count)" }
```

Expected: HEAD `c1a635e969f72efd5b9062d2c8994befd29ff100`; status prints
nothing; per seed `found 1` with `...RT-243s1_fixed_dr_...\nn\last_ur5e_insertion_ep_100_rew_49.403553.pth`,
`...RT-243s2_...rew_47.91025.pth`, `...RT-243s5_...rew_87.923416.pth`, and
`old RT-253 folders: 0`. Anything else: do not run; paste it.

The three seeds as one chain (same window):

```powershell
foreach ($s in 1,2,5) { $c=(Get-ChildItem "$L\*__RT-243s${s}_*\nn" -File | Where-Object { $_.Name -match '^last_ur5e_insertion_ep_100_rew_[-+0-9.eE]+\.pth$' }).FullName; .\scripts\rt_log.ps1 "RT-253s$s" python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed $s --checkpoint $c --max_iterations 250 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402 }
```

The overrides are RT-243's, word for word (`rt_logs/RT-243_expectation.md`
command); new are only `--checkpoint` and `--max_iterations 250`.
Wall time: about 1.9 / 1.9 / 2.3 h (RT-243: 1.26 / 1.26 / 1.53 h per 100
epochs, x1.5); longer if s5 learns (learning seeds s3, s4 took 2.1 h per
100). Whole chain about 6-7 h.

After the chain, ONE zip:

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
foreach ($n in 1,2,5) {
  $t = "RT-253s$n"
  $dirs = @(Get-ChildItem -Directory "logs\rl_games\ur5e_insertion\*__${t}_*")
  if ($dirs.Count -ne 1) { Write-Host "$t : expected ONE run folder, found $($dirs.Count)"; continue }
  $run = $dirs[0].FullName
  python scripts/check_episode_log.py --run-dir $run | Out-File -Encoding utf8 "rt_logs\${t}_eplog.txt"
  foreach ($f in "scalars.csv", "demo_metrics.json", "episodes.csv", "episodes_meta.json") { if (Test-Path "$run\$f") { Copy-Item -LiteralPath "$run\$f" "rt_logs\${t}_$f" } }
  Copy-Item -LiteralPath "$run\params\agent.yaml" "rt_logs\${t}_agent.yaml"
  Copy-Item -LiteralPath "$run\params\env.yaml" "rt_logs\${t}_env.yaml"
  Get-ChildItem -LiteralPath "$run\nn" | Select-Object Name, Length | Out-File -Encoding utf8 "rt_logs\${t}_nn.txt"
}
$files = (Get-ChildItem rt_logs\RT-253s*).FullName
Compress-Archive -LiteralPath $files -DestinationPath C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-253s.zip -Force
Get-Item C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-253s.zip
```

## Points -- every seed (N = 1, 2, 5)

* **P1 setup** (gate): as RT-243 P1; run folder
  `...\<mm-dd_HH-MM-SS>__RT-253sN_fixed_dr_offset50mm_currOFF_start+20mm_floor-30mm_seedN`;
  `[INFO]: Loading model checkpoint from: ...RT-243sN_...\nn\last_ur5e_insertion_ep_100_rew_<R>.pth`.
* **P1c resume** (new): one `[autodr-state] loaded from the checkpoint
  (run ..., seed N): bounds_version 5` line (RT-243 OUTCOME P3: 5 after
  the floor phase); NO `reached H_min, phase dr` line (the phase is
  loaded); the first console block reads `Learning iteration 101/250`.
  A `SystemExit` about seed or format = the resume refused, nothing judged.
* **P2 console block**: `Learning iteration 101/250` .. `250/250`, zero
  `block NOT printed` lines.
* **P3 fixed width**: as RT-243 P3 over epochs 101..250: `bounds_version`
  5 constant, `all_at_max` 1, the seven bounds at their maximum, zero
  `[autodr]` lines for the seven.
* **P4 learning (the hypothesis)**: from `RT-253sN_scalars.csv` on the
  epoch axis: `Episode/success_rate` mean over 241..250 per seed, against
  0.000 / 0.000 / 0.0286 (RT-243 91..100). Reported beside it: the first
  epoch with success > 0, `mean_max_depth_mm` at 250, `last_lr` range.
* **P6 end**: `MAX EPOCHS NUM!`; `nn` holds `ep_125`, `ep_150`, `ep_175`,
  `ep_200`, `ep_225`, `ep_250` (`save_frequency` 25) and `last_..._ep_250_...`;
  `[scalars] wrote`; `[rt_log] exit code: 0`; no Traceback.
* **P7 episode record**: `check_episode_log` VERDICT COMPLETE (iteration
  numbering after a resume: record what it prints).

## Pins

* `RT-253sN_agent.yaml`: `params.seed: N`, `config.max_epochs: 250`,
  `load_checkpoint: true`, `load_path` = the `ep_100` file; everything else
  as `rt_logs/RT-243s/RT-243sN_agent.yaml` (diff: only these and the name).
* `RT-253sN_env.yaml`: against `RT-243sN_env.yaml` only the log folder.

## Pre-registered readings

| Outcome | Reading |
|---|---|
| s5 up, s1/s2 at 0 | hypothesis holds: more epochs help a seed that has begun; seeds that never enter the pocket stay stuck |
| s1 or s2 above 0 | the budget, not only the seed, limits Fixed-DR here; D-205's 100 epochs cost these seeds |
| s5 not above 0.0286 | s5's rise at 100 was not a trend |
| resume refused / Traceback | not a result |

## What this run cannot answer

* Any evaluation number: the new checkpoints are not played (D-209 keeps
  them outside the evaluation); a replay or evaluation needs its own
  decision.
* K5 at a larger budget: AutoDR and No-DR stay at 100 epochs.
* Whether an uninterrupted 250-epoch run would equal this resume (`last_lr`
  restart, environments reset).
* Why exactly s1 and s2 fail (only the reading "they stay out of the
  pocket", RT-243 OUTCOME).

## CORRECTION, before the run (2026-10-01)

First try: all three calls stopped in argparse, `train.py: error: argument
--checkpoint: expected one argument`, `[rt_log] exit code: 2`, `0 neue
Ordner` -- the loop's `$c` was empty, nothing trained. The pre-check had
also printed `found 0` (console in `C:\Users`, relative `$L`). The
wildcard-in-the-middle path (`$L\*__RT-243sN_*\nn`) is replaced by the
two-step search `run_eval.ps1` and the RT-251 loop use (`-LiteralPath $L
-Directory -Filter`, then `nn`), with an absolute `$L`, and the chain runs
only if all three checkpoints are found. `rt_log.ps1` APPENDS to an
existing log, so the three failed logs are renamed first
(`RT-253sN.try1-no-checkpoint.*`). Everything else in this file stands.

## OUTCOME (2026-10-01, files `rt_logs/RT-253s/`)

Started 01:40 / 03:19 / 04:55 (after the CORRECTION above); run time about
1.6 / 1.6 / 3.4 h (s5 slower once it learns, as RT-243s3/s4).

**Hypothesis: HOLDS.** Training `Episode/success_rate`, mean over epochs
241..250 (`RT-253sN_scalars.csv`, epoch axis): **s1 0.0000, s2 0.0000,
s5 0.9941** (reference RT-243 91..100: 0.000 / 0.000 / 0.0286).

* P1: all three `[rt_log] git: c1a635e`, `imported from:
  ...Phase5_RlGames_v1\source\insertion\insertion`, `dr_mode: fixed_dr`,
  folders `10-01_..__RT-253sN_fixed_dr_offset50mm_currOFF_start+20mm_floor-30mm_seedN`.
  The provider line reads `bounds_version 5`, not RT-243's 0: the startup
  report prints it AFTER the state load (log line 130 after line 93), so it
  shows the loaded state -- consistent with P1c, not a fault.
* P1c: `Loading model checkpoint from: ...RT-243sN_...\nn\last_..._ep_100_...`;
  `[autodr-state] loaded from the checkpoint (run '<RT-243sN folder>', seed
  N): bounds_version 5`; no `reached H_min` line; first block `Learning
  iteration 101/250`. PASS.
* P2: 150 blocks each (101..250), zero `block NOT printed`. PASS.
* P3: `bounds_version` 5, `all_at_max` 1, `dr/phase` 1 in every epoch;
  zero `[autodr]` lines. PASS.
* P4 (the hypothesis), beside it:
  * s1, s2: success 0 in every epoch; `mean_max_depth_mm` 0.00 / 0.01 at
    the end, episode length 255.0 / 254.3 -- they still never enter the
    pocket. `episodes.csv`: s1 19199 timeout + 5 force_abort, s2 19196 +
    16, zero successes.
  * s5: from 0.20 at epoch 103 to 0.99 at epoch 135; 10-epoch mean >= 0.5
    from epoch 118, >= 0.9 from epoch 131; length 47.7 and depth 34.76 mm
    at the end. Epochs 101/102 read 1.000 / 0.987: censoring right after
    the environment reset (iteration 0 of `episodes.csv` holds 6 episodes,
    all success, before any timeout could end) -- not a reading.
  * The jump 0.056 (epoch 100) -> 0.20 (epoch 103) right after the resume
    is NOT explained: the `last_lr` restart at 1e-4 (D-209) and the
    synchronous reset of all environments are both possible; not separated.
  * `last_lr` 8.8e-6..6.7e-5 (s1, s2), 1.3e-5..6.7e-5 (s5); zero
    non-finite cells.
* P6: `MAX EPOCHS NUM!`, `ep_125` .. `ep_250` + `last_..._ep_250_...`,
  `[scalars] wrote`, `[rt_log] exit code: 0`, no Traceback / SystemExit.
  PASS.
* P7: `check_episode_log` VERDICT COMPLETE, all three (iterations counted
  from 0 again after the resume: 1..149 / 1..149 / 0..149).
* Reading (pre-registered table, first row): more epochs rescue a seed
  that has begun to learn (s5 now ~0.99 in training); seeds that never
  enter the pocket stay stuck (s1, s2 at 0 after 250 epochs). For Fixed-DR
  here the seed decides more than the budget. Training numbers only: the
  `ep_250` checkpoints are not evaluated (D-209).

### Addendum after the OUTCOME (2026-10-01): where s1, s2 sit

The user watched RT-253s2 (`ep_250`, view copy, `--autodr-state` under
`fixed_dr`): the part goes behind and under the pocket; the gripper, which
has no collider (D-077), hangs inside the fixture. The tripwire agrees
(`Episode/below_fixture_rate`, must read 0; inbox 2026-09-15): mean over
101..250 s1 0.9974, s2 0.9936, s5 0.0001; RT-243 (1..100) s1 0.6374, s2
0.6647, s5 0.3778 (last 10: 0.98 / 0.99 / 0.31). AutoDR RT-242s1..s5: 0 at
the end (s2 up to 0.22 on the way). The path is unpaid since 2026-09-15 but
not blocked; `InBachelorErwähnen.md` row 2026-10-01.
