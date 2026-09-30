# RT-108 — expectation, written BEFORE the run

Written 2026-08-30 on the dev laptop. CLAUDE.md § Hyperparameter requires the
expectation to exist before the log does; `/rt-check` judges the log against
THIS file and nothing else.

**This file was RT-105's until 2026-08-30.** That run was started outside
`rt_log.ps1`, produced no log, and is retired without a verdict in
`VERDICTS.md`. The number is spent. The run then became RT-107, and RT-107 was
in turn spent on the 15-minute reward-hack probe (user, 2026-08-30), so the
full run is RT-108. Two things changed
with the renumber, and both are in the table below: the start-pose noise is
back to 0.0 (user), so this rung randomises the FIXTURE only.

## What the run is

The FIRST real PPO run of the insertion task. One fixed rung 0, no ladder
(`curriculum_enabled = False`, run tag `currOFF`). Its purpose is NOT a
success rate — it is the LEARNING CURVES that D-110 (3) derives
`rung_step_sizes` from, plus the first live reading of the D-153 exposure
instrument.

**Amended 2026-08-31, before the run.** RT-107 measured a reward hack and a
force-ceiling farm; D-157 and D-158 repair the task definition, and both land
in this run (D-158 states why the "one change per run" rule does not govern
this phase). Two points, P8 and P9, are added below and P3's number moved.

**Amended a second time 2026-08-31, still before the run (D-160).** The
force channel now carries a GRAVITY TARE: the welded tool's own weight
(0.824 kg, 8.083 N) is subtracted from the joint wrench before the EMA.
Until RT-115 the channel read contact PLUS that constant, because D-114
imported FORGE's force route while this repo keeps gravity on at the robot
and FORGE does not. Two points move with it, and NOTHING else in this file
changes:

* **P9's readings shift by ~8 N.** `force_norm_n` p50/p95/p99 are now
  contact only. RT-107's distribution (p50 79.4 N, p95 87.6 N against a
  100 N limit) is doubly untransferable — a hacking policy AND an untared
  channel. The instruction not to predict an abort rate stands unchanged.
* **A new free reading, no threshold attached.** With the tare on, a resting
  or free-flying arm should read near 0 N, not ~8 N. If `force_norm_n` p50
  sits near 8 N across the run, the tare is not doing what RT-115 measured
  and that is worth reporting before any other force number is read.

The four noise fields below are UNCHANGED since git `9e1ebcd` (re-checked
against `insertion_env_cfg.py` on 2026-08-31, after D-160 landed), as are
`force_abort_f_max_n = 60.0`, `abort_payment = -1.0` and all four agent
hyperparameters.

Configuration under test (all four from `insertion_env_cfg.py`, values
unchanged since git `9e1ebcd`):

| field | value |
|---|---|
| `fixture_pos_noise_xy` | 0.005 m |
| `reset_joint_noise` | **0.0** |
| `reset_yaw_noise` | **0.0** |
| `fixture_yaw_noise_rad` | 0.0 |
| `fixture_tilt_noise_rad` | 0.0 |

ONE thing varies in this run: the fixture's x and y. The start pose is
identical in every episode. That is deliberate (user, 2026-08-30) and it
BOUNDS WHAT THE RUN CAN CLAIM — see "What this run cannot answer" below.

Agent config is the file's own default, unchanged: `num_steps_per_env = 16`,
`max_iterations = 1500`, `learning_rate = 1.0e-3`, `num_mini_batches = 4`.
NOTHING is tuned in this run. It is the baseline every later change is read
against, so a hyperparameter edit here would destroy its only purpose.

## Arithmetic, so the log can be read against numbers

**The env count is 1024, not 128 (user, 2026-08-31, before the run).** The
whole block below was rewritten for it; the 128-env numbers this file
carried until now are void, not merely scaled. The change is allowed and
does not touch D-117 (c): that rule forbids the hyperparameter STUDY from
varying the env count, and names the first main run's configuration as the
reference the study then holds fixed. This run IS that first main run, so it
sets the value rather than varying one. Every later study run must use 1024.

The reason for 1024, and it is not throughput: **RT-107 found its reward
exploit at episode 68 960.** At 128 envs this run reaches only ~12 000
episodes in total, so a late-emerging exploit of that same class could not
appear at all — the run would end long before the point where RT-107's
policy changed behaviour, and a clean P8 would prove nothing about it. At
1024 envs the run passes that episode count with margin. 1024 is also
RT-107's own env count, which makes the two runs comparable.

* 256 control steps per episode / 16 steps per iteration = **16 iterations
  per episode**; at 1024 envs that is **64 episodes per iteration**.
* 1500 iterations = 24 000 control steps per env = **96 000 episodes**
  (93.75 per env). Past RT-107's 68 960 at iteration ~1078.
* Trailing window = **2000 episodes**, full at iteration **~31**;
  `demo_metrics.json` is rewritten every **250** episodes, so the FIRST dump
  lands at iteration **~4**. RT-104's impossible expectation (a metrics file
  after 5 iterations) was a 128-env artefact and does not apply here — at
  this env count a file after 5 iterations is expected, not impossible.
  D-157's prose "visible from iteration ~4" matches this env count.
* **NO wall-clock band is predicted.** RT-104's 72–219 ms per RL step was
  measured at 128 envs and does NOT transfer to 1024; no run at 1024 envs
  has a recorded wall clock in `rt_logs/VERDICTS.md`. The duration is a
  READING taken from this log, and it becomes the band later runs are
  compared against. Quoting "30–50 min" here would be an invented number.

## Points, each PASS/FAIL on its own

1. **P1 — it starts and survives.** Exit 0, no `nan` anywhere in the log, no
   traceback. Read the exit-code trap in HANDOFF-RL: `exit code: 0` alone is
   NOT proof; the log must also carry no traceback.
2. **P2 — the run is the configured one.** The startup report prints
   `reset_joint_noise = 0.0`, `reset_yaw_noise = 0.0`,
   `fixture_pos_noise_xy = 0.005`, `fixture_yaw_noise_rad = 0.0`, and the run
   folder carries `_currOFF`.
3. **P3 — the placeholders are shouted.** `PLACEHOLDER VALUES IN USE` appears
   with `abort_payment = -1.0` and `force_abort_f_max_n` at D-158's value
   (60.0, and the cfg field is its only home). This run's success rate is
   only readable together with them.
4. **P4 — a metrics file exists.** `demo_metrics.json` in the run folder,
   with `episodes` >= 250 and `recent_window_episodes` > 0.
5. **P5 — the D-153 exposure is measured.** `stage1_lateral_y_mm` is present
   and NOT `{"episodes": 0}`. It carries `reach_offset_mm = 7.8847`,
   `stage1_allows_mm = 8.7000`, and an `over_reach` count.
   **EXPECTED: `over_reach = 0`** and `max_mm` well below 7.8847 — the
   configured 0.005 m fixture offset cannot put the part at the hole.
   `over_reach > 0` does NOT fail the run; it REOPENS D-153 with a number,
   which is exactly what the instrument was built for.
6. **P6 — the six metrics exist and are readable in order.** Episode Reward,
   Policy Loss, Value Loss, Entropy, Explained Variance, KL. TensorBoard
   under `logs/rsl_rl/ur5e_insertion/<run>/`.
7. **P7 — the jitter gets a number.** The user watched the retired RT-105 in
   the viewport and reported the arm jittering and drifting UP and toward the
   robot base early on. `joint_target_lag_rad` in `demo_metrics.json` is the
   instrument that decides whether that is harmless start-of-training noise
   or the D-037 wind-up. Read `max` and `failure_p95` DIVIDED BY
   `action_scale` (0.02 rad): the quotient is the number of control steps the
   policy must spend unwinding before the arm moves at all. NO threshold is
   set here, because none is derived — this point is a READING, and its
   result is what a threshold would later be argued from.
8. **P8 — the D-157 tripwire reads zero.** `demo_metrics.json` carries
   `success_depth_invariant_violations`; both `recent` and `cumulative` MUST
   be 0. The same number is on the TensorBoard curve. Anything above 0 voids
   the success rate in that same file — it is RT-107 coming back, and the run
   is not read further. Paired reading: at any `success_rate_recent > 0`,
   `mean_max_depth_mm` MUST be > 0. RT-107's signature was 0.9765 at 0.0.
9. **P9 — the 60 N limit gets a number, not a forecast.** Read
   `force_abort_rate` and the `force_norm_n` spread (p50 / p95 / p99 /
   `over_f_max`) together. **NO abort rate is predicted here** (D-158): the
   RT-107 distribution that would support a prediction belongs to the hacking
   policy and does not transfer. A high abort rate is a reading that feeds
   the force decision, not a failed run.

   *When these can be read at all:* `demo_metrics.json` is written only on
   episode end, so the measured dump rule applies (HANDOFF-RL.md): at least
   16 iterations (one full episode), AND `_ep_count` reaching a multiple of
   the 250-episode dump interval. At 1024 envs the second condition is the
   binding one and lands at iteration ~4 — see the arithmetic block. (The
   "16 iterations at 128 envs" wording this line carried until 2026-08-31
   was the 128-env form of the same rule.)

## What this run cannot answer, stated before anyone reads a number into it

* **Nothing about yaw.** `reset_yaw_noise = 0.0` and
  `fixture_yaw_noise_rad = 0.0`, so every episode starts inside the part's
  free-yaw window (0.0041 rad). D-038's objection applies in full: a high
  success rate here would say nothing about turning the part.
* **Nothing about start-pose robustness.** `reset_joint_noise = 0.0`, so the
  arm begins every episode in the same joint vector.
* **Nothing about tilt.** Both tilt fields are 0.0.

## What is NOT expected, stated so it is not read as failure

* **No success-rate target.** Rung 0 has never been trained. Any rate is a
  measurement, not a grade. `success_rate_recent = 0.0` after 1500
  iterations is a legitimate outcome and would say the reward shaping needs
  work — it would not say the run failed.
* **No judgement on hyperparameters from this run alone.** Candidate order
  if the curves ask for it (CLAUDE.md): learning rate -> clip range -> batch
  size, ONE change per run, expectation written first.

## Red flags to report unprompted (CLAUDE.md § Analyse-Regeln)

* Entropy collapsing toward 0 early -> exploration dead.
* Grad-norm pinned at the clamp or NaN -> check action limits, gains,
  timestep FIRST, never the learning rate.
* Value Loss diverging / Explained Variance -> 0 -> value-function drift;
  rsl_rl has no value normalisation (D-116), so that observation point
  reopens.
* **Wall clock: no band, so no red flag from it.** No run at 1024 envs has a
  recorded duration, so nothing here can be "far outside" anything. Report
  the measured wall clock as a plain reading; it becomes the band later runs
  are judged against.
