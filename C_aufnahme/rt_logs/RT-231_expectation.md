# RT-231s1..s5 -- Fixed-DR training arm (D-192), written BEFORE the runs (2026-09-23)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Number: first written as RT-228 (`d74018df`, 2026-09-23 19:54). 17 minutes
later `origin/p6-TestAusserhalb` took RT-228 as well (`76ed9e0e`, OOD and
T-final repeat, run and judged there as `RT-228s<N>f` / `RT-228s<N>o`).
The Fixed-DR training had not started, so it moved to RT-231 (user "B",
2026-09-24): RT-226..RT-230 are p6's (`git ls-tree` over every
`origin/*` branch, 2026-09-24). "RT-228" in p5 commits before the rename
means this file.

## Context

D-192: a third training arm, Fixed-DR, so that the evaluation can ask K5
(`eval_plan.md` section 1.4, pre-registered in `bd0e86e1` before this run).
The arm is `dr_mode=fixed_dr` (`autodr.FixedDR`, `3c6dabcd`): AutoDR's
start-floor phase line for line, then every quantity at the
`FixedWidth.at_max` width, drawn uniformly (no boundary episodes), fixed to
the end. Template for everything not named here: `rt_logs/RT-206_expectation.md`.

## Code state

Branch `p5-kraftsensor`, the commit that adds this file (code `3c6dabcd`).
Its SHA is read back from `origin/p5-kraftsensor` at handover and given in
chat with the pull command. Every log's `[rt_log] git:` line must match it
before any point below is judged.

`CODE_MARKER = "insertion-osc-2026-09-03a"` (`insertion_env.py:76`), the
same as RT-206 -- the marker does NOT distinguish this code from RT-206's.
The gate is therefore the git line AND the provider line (P1).

Changes against RT-206 (`5a6cc8e`), read from `git diff 5a6cc8e 3c6dabcd --
source scripts/rsl_rl` (2026-09-23): `train.py` unchanged. `insertion_env.py`
/ `insertion_env_cfg.py` / `autodr.py`: the evaluation table (every new
line gated on `dr_mode == 'table'` or `self._eval_table`, or a print), the
episode log's `getattr(self._dr, "phase", "fixed")`, and the Fixed-DR arm.
The autodr/off training path is unchanged by that reading -- read, not run.
Offline (laptop, `3c6dabcd`): `check_autodr.py` 170 PASS + counter-proof
PASSED; `check_env_wiring.py` 389 PASS + counter-proof PASSED;
`check_mutation_anchors.py` every anchor once.

## The one hypothesis

**Under `3c6dabcd` every Fixed-DR seed completes 1500 iterations with exit 0,
its floor reaches H_min, and from that moment the five quantities stay at
their full width: no `[autodr]` line after the transition and
`dr/bounds_version` constant to iteration 1499.**

Falsified by: a Traceback or a missing 1500th iteration; no
`reached H_min, phase dr` line; any `[autodr]` line for one of the seven
boundaries (`lat_r_hi`, `yaw_lo`, `yaw_hi`, `tilt_hi`, `start_height_hi`,
`friction_lo`, `friction_hi`); `dr/bounds_version` changing after
`dr/phase` turned 1.

The success rate is NOT part of the hypothesis. Whether Fixed-DR learns
the task in 1500 iterations is unknown for this setup; it is reported.

## Commands

Training PC root
`C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`
(rt_log header of RT-206s1). Pull and verify first; the expected SHA is
given in chat.

```powershell
git pull; git rev-parse --short HEAD
```

Watcher in a SECOND PowerShell window, on for the whole series
(`PROBLEMS.md` 2026-09-22, the orphan `hub.exe` holds rt_log's pipe):

```powershell
while ($true) { Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('hub.exe','omni.telemetry.transmitter.exe') } | Where-Object { -not (Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue) } | ForEach-Object { "$(Get-Date -Format HH:mm:ss) stop $($_.Name) $($_.ProcessId)"; Stop-Process -Id $_.ProcessId }; Start-Sleep 15 }
```

The five seeds, one after another (RT-206s1..s5 with ONLY
`env.dr_mode=autodr` -> `env.dr_mode=fixed_dr`):

```powershell
.\scripts\rt_log.ps1 RT-231s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations 1500 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3; .\scripts\rt_log.ps1 RT-231s2 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 2 --max_iterations 1500 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3; .\scripts\rt_log.ps1 RT-231s3 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 3 --max_iterations 1500 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3; .\scripts\rt_log.ps1 RT-231s4 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 4 --max_iterations 1500 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3; .\scripts\rt_log.ps1 RT-231s5 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 5 --max_iterations 1500 env.dr_mode=fixed_dr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

No smoke run: the build path is RT-206's, the new code is one provider
class. RT-231s1 is its own smoke -- look at its log once `dr/phase` turned
1 (RT-206s1: iteration 64; a hint only) and stop the chain on a Traceback.
Wall time: RT-206s1..s5 took 3.57-3.79 h each (`scalars.csv` wall_time),
so about 18-19 h for the chain -- a hint from the old run, not a promise.
No `--stop-when-dr-max`: the rule is the fixed budget (as RT-206). Under
Fixed-DR it would stop the run right after the transition, because
`all_at_max` turns true there.

## Points -- per seed RT-231s1..s5

* **P1 setup**: `[rt_log] git:` = handover SHA; `Environment seed : N`;
  `force abort at 30.0 N`; `provider: FixedDR, 8 boundaries,
  bounds_version 0`; the `START FLOOR` report line with `dr_mode=fixed_dr`;
  run folder `..._RT-231sN_fixed_dr_..._floor-30mm_seedN` (tag from
  `train.py:334-335`). A folder or provider line with `autodr` = wrong run,
  nothing below is judged.
* **P2 floor**: exactly one `reached H_min, phase dr` line; the `dr/phase`
  0 -> 1 iteration (per seed, reported). Beside it, REPORTED ONLY: whether
  `scalars.csv` rows before that iteration equal RT-206sN's
  (`rt_logs/RT-206sN/RT-206sN_scalars.csv`) -- D-192 claims the floor phase
  is AutoDR's line for line; equal rows would show it, different rows are
  NOT a FAIL (the code commit differs, GPU determinism over that span is
  not established), they are named.
* **P3 fixed width** (the hypothesis): after the transition no `[autodr]`
  line at all; `dr/bounds_version` constant from the transition to 1499;
  `dr/all_at_max` 1 from the transition on; every `dr/<key>` of the seven
  at its maximum (0.03 / -0.0873 / 0.0873 / 0.1745 / 0.12 / 0.08 / 0.20,
  `autodr.DR_DIMS` via `FixedWidth.at_max`); every `dr/<key>_fill` of the
  seven 0 for the whole run.
* **P4 six curves** in order (reward, surrogate, value, entropy; EV/KL not
  logged) with trend; `Policy/mean_noise_std` not below 0.2.
* **P5 success**: `Episode/success_rate` at 1499 and its last-50 mean;
  `dr/train_success_regular_fresh`; the bin tables of `demo_metrics.json`.
  Reported against RT-206sN and RT-207sN -- a hint, no bar.
* **P6 end**: `Learning iteration 1499/1500`, `model_1499.pt`,
  `[scalars] wrote`, `[rt_log] exit code: 0`; files handed back:
  `scalars.csv`, `demo_metrics.json`, `autodr_1499.json`, `params/env.yaml`,
  `episodes.csv`, `episodes_meta.json`, the short log, and the output of
  `python scripts/check_episode_log.py --find-run RT-231sN` (VERDICT
  COMPLETE).
* **P7 physics**: `below_fixture_rate`, `Episode/max_depth_max_mm` <= 46 mm,
  `Episode_Reward/engaged` window means, `interpen_max_mm` p50/p95/p99/max
  -- as RT-206 P7.
* **P8 force**: `force_abort_rate` curve and final value,
  `Episode/force_max_p95_n` curve.

## Pins (read from the files, not from the log prose)

* `params/env.yaml`: `dr_mode: fixed_dr`, `start_floor_m: -0.03`,
  `autodr_stall_buffers: 3`. Against `rt_logs/RT-206sN/RT-206sN_env.yaml` the
  diff is ONLY `log_dir` and `dr_mode` (the `eval_plan.md` 3.1 fairness
  rule).
* `autodr_1499.json`: `"format": "autodr-sidecar-1"` (`train.py`
  `SIDECAR_FORMAT`), `"seed": N`, and inside `"autodr"`:
  `"format": "fixed-dr-1"` (`autodr.FixedDR.FORMAT`), `"floor": {"f0": -0.03,
  "top": 0.02, "n_steps": 5}`, `"steps"` of each of the seven = 10
  (`autodr.DELTA_STEPS`) and of `start_height_floor` = 5.
* `demo_metrics.json`: `"dr_mode": "fixed_dr"`; `dr_state.mode` =
  `"fixed_dr"` (`FixedDR.as_dict`).

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| 1500 iterations, exit 0, P3 holds | hypothesis holds for this seed | next seed |
| an `[autodr]` line for one of the seven, or `bounds_version` moves after the transition | FixedDR does not hold its width under Isaac; the arm is not Fixed-DR | stop the chain, diagnose; no seed counts |
| no transition to phase dr within 1500 | the floor phase did not finish; the seed never trained at full width | report; no rerun (budget rule); K5 reads it as a Fixed-DR outcome |
| `force_abort_rate` climbs and success stalls | full width from H_min on may be too hard for this budget -- the DORAEMON pattern | report; this IS a result for K5, not a fault |
| a run stops before 1499 without Traceback | not a result | tell the user; rerun the same seed from scratch |

## What these runs cannot answer

* K5 itself -- that needs the evaluation blocks T-nominal-F and T-final-F
  (`eval_plan.md` 3.2) and the commit decision (HANDOFF-RL.md NEXT 1d).
* Whether growth or boundary sampling causes a difference -- Fixed-DR
  changes both at once (`eval_plan.md` 3.1).
* Whether the training success rate is a test success rate -- it carries
  PPO's exploration noise; the plan forbids reading it so.
* Whether the RT-206 and RT-231 floor phases are identical beyond what P2
  reports.
