# RT-206a, RT-206s1..s5 -- restarted study, Condition 2 (AutoDR), written BEFORE the runs (2026-09-15)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

## Context

User decision 2026-09-15 (`docs/decisions_inbox.md`, "Seed study restart:
1000 fixed iterations, a 30 N force abort, every seed from scratch"): all
seeds restart from scratch, exactly 1500 PPO iterations each (raised from 1000 the same day), force abort of
an episode at the cfg default (lowered to 30 N the same day). The same code
state carries the pocket-gate floor (`docs/decisions_inbox.md`, "The pocket
gate gets a floor", RT-201s3). RT-201, RT-201s2, RT-201s3 are record only.

## Code state

Branch `p5-kraftsensor`, the commit that adds this file. Its SHA is read back
from `origin/p5-kraftsensor` at handover and given in chat with the pull
command. Every log's `[rt_log] git:` line must match it before any point
below is judged.

Changes against RT-201s3 (`5b7e67b`), all in that commit:
- `insertion_math.in_pocket_volume` and its use as `in_pocket` in
  `_get_dones` (floor = fixture underside, 46 mm);
- `below_fixture_rate` in the per-iteration log and in `demo_metrics.json`;
- `force_abort_f_max_n` 50.0 -> 30.0 (`insertion_env_cfg.py`);
- the episode record `episodes.csv` + `episodes_meta.json`
  (`docs/decisions_inbox.md`, "Interpenetration record for the thesis");
  `train.py` hands `num_steps_per_env` to the env for its iteration column.
Offline: `check_insertion_math.py` 294 PASS + counter-proof, `check_env_wiring.py` 358 PASS + counter-proof, `check_insertion_sdf.py --self-test` 35/35, `check_episode_log.py --self-test` 17/17, `check_mutation_anchors.py` 489/489.
Run `rt_logs/RT-208_expectation.md` (the pose ladder) BEFORE the smoke.

## The one hypothesis

**Under the 2026-09-15 code state every AutoDR seed completes 1500 iterations
with exit 0, and no seed is paid for depth below the fixture.**

Falsified by: a Traceback or a missing 1500th iteration; `Episode_Reward/
engaged` window means above 20 per episode in any 50-iteration window
(RT-201s3 hump 38-61, RT-201s2 at most 9.3) while `below_fixture_rate` > 0.

## Commands

Training PC root `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`.
One job at a time. Smoke FIRST:

```powershell
.\scripts\rt_log.ps1 RT-206a python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations 30 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

Then the five seeds, one after another:

```powershell
.\scripts\rt_log.ps1 RT-206s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-206s2 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 2 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-206s3 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 3 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-206s4 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 4 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-206s5 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 5 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

No `--stop-when-dr-max`: that is the rule. No `env.force_abort_f_max_n`: the
cfg default is the value.

## Points -- smoke RT-206a (builds and wires, answers nothing about learning)

* **S1** `[rt_log] git:` = handover SHA; `Environment seed : 1`.
* **S2** startup report line `termination ... force abort at 30.0 N`.
* **S3** the iteration block prints `below_fixture_rate` (as
  `Mean episode below_fixture_rate` or the key name rsl_rl gives it).
* **S4** 30 iterations (every env resets at least once: an episode is 256
  steps = 16 iterations, `Laufplan_Phase5.md` step 3), no Traceback, `[scalars] wrote`, `[rt_log] exit code: 0`
  (the exit-code line was missing in older chains -- if missing again, S4 is
  judged on the absence of a Traceback and on `[scalars] wrote`, and says so).
* **S5** the line `[insertion] episode record: ... (13 columns)`, then on the
  training PC `python scripts/check_episode_log.py --find-run RT-206a`:
  every C1-C16 PASS and `VERDICT: COMPLETE`; its summary lines are reported.

## Points -- per seed RT-206s1..s5

* **P1 setup**: git SHA; seed; `force abort at 30.0 N`; `provider: AutoDR,
  8 boundaries`; `START FLOOR` line; folder `..._RT-206sN_autodr_..._seedN`.
* **P2 floor**: `reached H_min, phase dr` line; `dr/phase` 0 -> 1 iteration
  (a per-seed result).
* **P3 boundaries at 1500**: `dr/all_at_max`, `dr/bounds_version`, each
  boundary value at iteration 1499 -- REPORTED, no bar (named cost of the
  fixed budget).
* **P4 six curves** in order (reward, surrogate, value, entropy; EV/KL not
  logged) with trend; `Policy/mean_noise_std` not below 0.2.
* **P5 success**: `Episode/success_rate` at 1499 and its last-50 mean;
  `dr/train_success_regular_fresh`; the four bin tables of
  `demo_metrics.json`.
* **P6 end**: `Learning iteration 1499/1500`, `model_1499.pt`, `[scalars] wrote`,
  exit code; files handed back: `scalars.csv`, `demo_metrics.json`,
  `autodr_1499.json`, `params/env.yaml`, `episodes.csv`, `episodes_meta.json`,
  the short log, and the output of
  `python scripts/check_episode_log.py --find-run RT-206sN` (VERDICT COMPLETE).
* **P7 physics** -- the point this restart exists for:
  `below_fixture_rate` per iteration (curve) and in `demo_metrics.json`
  (`recent`, `episodes`); `Episode/max_depth_max_mm` must stay <= 46 mm
  (the gate's floor); `Episode_Reward/engaged` window means (compare
  RT-201s2's 3-9 and RT-201s3's 38-61); `interpen_max_mm` p50/p95/p99/max
  and `over_thresh`.
* **P8 force under the new limit**: `force_abort_rate` curve and final value,
  `force_norm_n` p50/p95/max and `over_f_max`, `Episode/force_max_p95_n`
  curve. A hint only: RT-201 family p95 30.7-30.9 N under 50 N.
* **P9 stagnation**: hold lines, `dr/*_holds`, any `STALL` note.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| 1500 iterations, exit 0, `below_fixture_rate` 0 in the final window, engaged never above 20 | hypothesis holds for this seed | next seed |
| `below_fixture_rate` > 0 but engaged stays at the RT-201s2 level | the part still gets around and under the free-standing fixture (RT-201s3r), the reward no longer pays it; FINDING for the thesis, not a stop | report the rate |
| engaged window mean > 20 | a paid exploit remains; the fix did not close it | stop the series, diagnose before the next seed |
| `force_abort_rate` climbs and success stalls below 0.9 | the 30 N limit forbids the solution for this setup (CLAUDE.md hacking checklist: abort as the brake) | report to the user; the limit is the user's decision |
| `dr/all_at_max` 0 at 1499 | the seed did not reach the DR ceilings inside the budget | reported per seed, no rerun (rule) |
| a run stops before 1499 without Traceback (RT-201s2 pattern) | not a result | tell the user; rerun the same seed from scratch |

## What these runs cannot answer

* Which way the part gets under the fixture -- answered by RT-201s3r
  (around it; `rt_logs/VERDICTS.md`).
* Whether 30 N is a damage limit -- it is not; no source for one exists.
* The test success rate -- that is the evaluation table (Laufplan steps 8/12),
  not built.
* Condition 1 -- `rt_logs/RT-207_expectation.md`.
