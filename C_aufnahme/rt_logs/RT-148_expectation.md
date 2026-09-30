# RT-148 — expectation, written BEFORE the run (2026-09-04)

**The first PPO run under the OSC controller.** Everything before this
date that trained ran under `joint_pd`. This is a BASELINE: it is not
tuned, and no hyperparameter is changed for it.

## Configuration — every number named, none of them varied

| field | value | home |
|---|---|---|
| `control_mode` | `osc` | Decision (1)/(6), inbox "Audit 2026-09-03 (a)" |
| `osc_kp_pos` | 100.0 | **D-166** — derived, `kp <= 20/(8.82*0.02)` = 113.4 |
| `osc_pos_step_limit_m` | 0.02 | placeholder, Factory's |
| policy rate | 15 Hz (`osc_decimation` 8) | placeholder, Factory's |
| controller rate | 120 Hz (every physics step) | Factory's form, verified |
| episode | 256 control steps = 17.067 s | D-113 (3) |
| `curriculum_enabled` | False | frozen: the first PPO run trains WITHOUT the ladder |
| start | rung 0, tip +0.030 m above the entrance | `RUNG0_START_TIP_ABOVE_ENTRANCE` |
| `force_abort_f_max_n` | 300.0 | **D-167** — kept for this run on purpose |
| `num_envs` | 128 | cfg default |
| PPO | `num_steps_per_env` 16, `num_mini_batches` 4, `lr` 1e-3 | `rsl_rl_ppo_cfg.py` |
| seed | 42 | |

## Two runs, in this order — do not skip a

`demo_metrics.json` is written at the END. A long run that is interrupted
leaves nothing to read, and nobody has measured what one iteration of
this env costs.

**RT-148a — cost and smoke, 50 iterations:**

    .\scripts\rt_log.ps1 RT-148a python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 128 --headless --seed 42 --max_iterations 50

**RT-148b — the baseline, only after a is judged.** Its
`--max_iterations` comes from a's measured seconds per iteration and is
filled in then, not now.

## Numbered expectation — RT-148a

1. exit 0, no traceback, and `demo_metrics.json` is written.
2. The startup report prints `control_mode osc`, kp pos 100.0, policy
   15.0 Hz / controller 120.0 Hz, episode 256 steps = 17.0667 s,
   `curriculum_enabled False`, and a rung-0 start of +30 mm.
3. **`force_norm_n` p95 stays BELOW 20 N.** This is the load-bearing
   point of the run: it is D-166's bound read back. The predicted ceiling
   is `Lambda * kp * step_limit` = 17.6 N at the reset pose.
4. **`force_abort_rate` = 0.0.** D-167's prediction: the steady push
   cannot reach 300 N, so the abort cannot fire from contact.
5. `mean_max_depth_mm` is GREATER THAN 0 at the end of the 50
   iterations — the policy at least reaches the entrance plane.
6. Entropy has not collapsed to ~0 within 50 iterations.
7. The run reports seconds per iteration, so `[measure]` wall-clock and a
   defensible `max_iterations` for RT-148b can be computed.

`success_rate_recent` is READ, not expected. 50 untuned iterations are
not a success claim in either direction.

## What would make each point wrong

1. -> a crash. Nothing about the hypothesis; fix and rerun.
2. -> the cfg did not reach the env. The run is VOID, not a finding.
3. -> **D-166's bound was computed at the wrong pose.** `Lambda` = 8.82 kg
   was read at the reset pose ONLY, and the margin is 13 % (the push
   passes 20 N as soon as `Lambda` passes 10.0 kg). Then `kp` moves, not
   the band. This is the entry's own Accepted risk coming true.
4. -> transients exceed the steady push by more than 17x, which nobody has
   measured. That would make D-167's guard live after all and is a real
   finding, not a failure.
5. -> see the pre-registered suspects below. Judge 5 ONLY together with
   them; do not reach for the learning rate.
6. -> exploration died in 50 iterations, which under an untuned baseline
   points at the reward scale, not at `entropy_coef` (which is inside
   D-117's study and must not be touched outside it).

## Pre-registered suspects, IN THIS ORDER

Written now so that no explanation can be chosen after the fact.

1. **`mean_max_depth_mm` stays 0 -> the shaping signal is dead at the
   start (D-161's mechanism).** At +30 mm the SDF kernel is UNREAD:
   RT-119 measured `kernel_sum` 1.46e-04 at +20 mm and 2.29e-08 at
   +50 mm, and +30 mm lies between with no run on it. D-165's depth
   progress term cannot help during the approach either — it pays the
   increment of the GATED maximum depth, and depth is negative until the
   tip is below the entrance. So the whole 30 mm of approach is carried by
   the coarse kernel alone.
2. **Forces above 20 N -> D-166's `Lambda` was read at one pose.**
3. **The action oscillates -> the missing EMA.** Factory and AutoMate
   smooth the action with `ema_factor` 0.2 (`factory_env.py:213`,
   `factory_env_cfg.py:51`, `assembly_env_cfg.py:48`); we have no
   smoothing at all and never decided to have none.
4. **It plateaus with a standing offset -> the anchoring (NEXT 0b).**
   We apply the delta to the CURRENT pose; IndustReal Sec. V-D replaced
   exactly that form because of "substantial steady-state error". RT-147
   measured that the arm never arrives.

**NOT the learning rate first.** CLAUDE.md's rule: on grad-norm or NaN
trouble, check action limits, gains and timestep before touching `lr`.

## The six curves, in the fixed order

Read `Episode Reward -> Policy Loss -> Value Loss -> Entropy ->
Explained Variance -> KL`, each with its trend, and only then judge.
Named red flags for this run: entropy early to 0; value loss diverging
with explained variance to 0 (D-116: rsl_rl has no value normalisation).

## What this run does NOT decide

- The episode length. It MEASURES how much of the 256 steps the policy
  uses; the decision follows the reading (Factory 10 s, AutoMate 5 s,
  ours 17.07 s).
- The action-rate penalty's weight. Its meaning changed with the
  controller — under `osc` the delta is a velocity, so `0.1 *
  ||a_t - a_(t-1)||` now prices acceleration — and it has not been
  re-read. Baseline first.
- The observation channels (§ 6 (5), concept stream).
- `F_max`. D-167 keeps 300 N deliberately so this run can measure the
  undisturbed force distribution.
