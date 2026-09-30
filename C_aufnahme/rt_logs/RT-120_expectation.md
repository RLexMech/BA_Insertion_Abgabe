# RT-120 — expectation, written BEFORE the run

Written 2026-08-31 on the dev laptop, git `7ec93a7`. CLAUDE.md
§ Hyperparameter requires the expectation to exist before the log does;
`/rt-check` judges the log against THIS file and nothing else.

## What the run is

The SAME run as RT-108 with exactly ONE difference: where the episode starts.

* RT-108 started every episode at the home pose, +165 mm above the stage-2
  opening plane. RT-119 measured the whole D-109 reward there: **3.05e-23**.
  The policy learned "touch nothing" and retreated upwards (D-161).
* RT-120 starts every episode at **-26 mm**, i.e. 26 mm INSIDE a 36 mm
  pocket, one coarse kernel margin above the seat (D-162). RT-119 measured
  `kernel_sum` 0.183 at -20 mm and 0.583 at the seat, so the reward here has
  a magnitude PPO can see.

Everything else is unchanged and must stay unchanged: 1024 envs,
`curriculum_enabled = False`, `rung_step_sizes` unset, `fixture_pos_noise_xy
= 0.005`, all four reset noises as in RT-108, and the file's own agent
defaults (`num_steps_per_env = 16`, `max_iterations = 1500`,
`learning_rate = 1.0e-3`, `num_mini_batches = 4`). NOTHING is tuned here.

## THE QUESTION THIS RUN ANSWERS

**Can this policy learn at all, given a reward with a gradient?**

It does NOT answer whether the policy can insert from a realistic stand-off.
The part starts 7 mm from the success band, so a success rate here is a
statement about the last 7 mm and nothing else. It must never be quoted as
the task's success rate.

## Arithmetic, so the log can be read against numbers

Unchanged from RT-108 because the env count and the episode length are
unchanged: 16 iterations per episode, 64 episodes per iteration, 1500
iterations = 96 000 episodes. Trailing window 2000 episodes, full at
iteration ~31; first `demo_metrics.json` dump at iteration ~4.

**NO wall-clock band is predicted, and this run is SLOWER than RT-108 by an
unknown amount.** Every reset now runs up to 120 IK iterations
(`start_pose_solve_last_iterations` in the metrics file says how many were
actually needed). The cost is a READING taken from this log, not a
prediction.

## Points, each PASS/FAIL on its own

1. **P1 — it starts and survives.** Exit 0, no `nan` anywhere, no traceback.
   The exit-code trap applies: `exit code: 0` alone is NOT proof.
2. **P2 — the code on the training PC is this code.** The startup report
   prints `insertion-rung0-start-2026-08-31-p1` and the rt_log header prints
   git `7ec93a7`. An older marker voids everything below.
3. **P3 — the run is the configured one, and the start pose is the new one.**
   The startup line `[insertion] rung-0 start pose: tool point -26.000 mm
   above the stage-2 opening plane` appears; the run folder carries
   `start-26mm` AND `currOFF`; the startup report prints
   `reset_joint_noise = 0.0`, `reset_yaw_noise = 0.0`,
   `fixture_pos_noise_xy = 0.005`.
4. **P4 — THE START POSE ACTUALLY LANDS.** In `demo_metrics.json`:
   `start_tip_above_entrance_mm = -26.0`,
   `start_pose_solve_unconverged_resets = 0`, and
   `start_pose_solve_worst_residual_mm` below the tolerance
   `start_pose_solve_tol_mm` (0.05). Anything else means the episodes did NOT
   start where the run says they did, and every number below is void.
   **NO warning line `START POSE SOLVE DID NOT CONVERGE` in the log.**
5. **P5 — the startup report agrees with itself.** `tip insertion depth`
   reads about **+0.026 m** with the expectation printed beside it, and the
   flange standoff matches its own printed expectation. RT-108's `-0.165 m`
   here would mean the solve never ran.
6. **P6 — the D-157 tripwire reads zero.** `success_depth_invariant_violations`
   is 0 for BOTH `recent` and `cumulative`. **Read this BEFORE any success
   rate.** Anything above 0 voids the success numbers.
7. **P7 — THE LEARNING QUESTION.** All six metrics in order, each with a
   trend: Episode Reward → Policy Loss → Value Loss → Entropy → Explained
   Variance → KL. **No threshold is set here.** What is being read is whether
   the episode reward MOVES at all against RT-108's flat run, and whether
   entropy falls in a way that says the policy committed to something.
   `success_rate_recent` and `mean_max_depth_mm` are read AFTER P6 and after
   the six curves, never first.
8. **P8 — the arm no longer runs away.** D-161's second finding is unfixed on
   purpose: the joint-target integrator has no clamp against the reset pose.
   `joint_target_lag_rad` (`max` and `failure_p95`, divided by
   `action_scale` = 0.02) says how many control steps the policy spends
   unwinding. **A READING, no threshold.** If the drift survives a reward
   that has a gradient, the clamp becomes its own change.
9. **P9 — the force channel.** `force_norm_n` p50/p95/p99 under the gravity
   tare (D-160). RT-119 measured 0.002 N in free air; RT-108 averaged 42.9 N
   while the replay showed free air, and that is an OPEN question (D-161).
   Here the part IS in contact with the pocket from step 0, so a nonzero
   reading is expected and no abort rate is predicted. Report
   `force_abort_rate` beside it, never folded into the success rate.
10. **P10 — the D-153 exposure.** `stage1_lateral_y_mm` present and not
    `{"episodes": 0}`. `over_reach = 0` expected. `over_reach > 0` does not
    fail the run; it reopens D-153 with a number.

## What this run cannot answer

* Nothing about the approach. The part starts inside the pocket.
* Nothing about yaw (`reset_yaw_noise = 0.0`, D-038).
* Nothing about `rung_step_sizes`. The ladder is off and stays off; the step
  sizes come from the learning curves this run produces, not from this run's
  configuration.
