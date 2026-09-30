# RT-134 — expectation, written BEFORE the run

Written 2026-09-02 on the dev laptop, git `ae50431`. CLAUDE.md
§ Hyperparameter requires the expectation to exist before the log does;
`/rt-check` judges the log against THIS file and nothing else.

## What the run is

**RT-131 repeated with exactly ONE change: D-164.** The force abort left
`terminated` and is now `truncated`, so rsl_rl adds `gamma*V(s)` back for it
instead of forfeiting the whole remaining return. `compute_dones` also makes
the two channels exclusive (`& ~success_now`), which closes the pre-existing
overlap of a success landing on the timeout step.

NOTHING ELSE MOVED. Same seed 42, same 1024 envs, same rung-0 start at
+30 mm, `curriculum_enabled = False`, `rung_step_sizes` unset,
`fixture_pos_noise_xy = 0.005`, the same reset noises, the same agent
defaults (`num_steps_per_env = 16`, `max_iterations = 1500`,
`learning_rate = 1.0e-3`, `num_mini_batches = 4`, `gamma = 0.99`), the same
`force_abort_f_max_n = 300.0`, the same `abort_payment = -1.0`, the same
kernel widths. This is an A/B against RT-131, not a new configuration.

## THE QUESTION THIS RUN ANSWERS

**Does the policy keep trying to insert once an abort no longer costs the
whole remaining episode?** RT-131's measured failure is the baseline and the
comparison is direct, iteration for iteration.

## The RT-131 baseline, from its own curves — the numbers to beat

* `force_abort_rate`: climbs to 1.0 by iteration ~40, collapses to 0 between
  iterations 40 and 70, stays 0 through 299.
* `mean_max_depth_mm`: peaks at ~0.4 mm around iteration ~17, is 0 from
  iteration ~70 onward.
* `Train/mean_reward`: rises to 57.7 by iteration ~150, then flat. 57.66 at
  iteration 299.
* `Policy/mean_noise_std`: ~0.2–0.35 during iterations 40–70, 0.0051 at 299.
* `success_rate`: flat 0 throughout.
* `force_norm_n`: p50 39.3 N, p95 41.9 N, max 42.9 N, `over_f_max` 0.

## Arithmetic that motivated the change — NOT a prediction of this run

Computed offline from the committed constants: holding the start distance for
a full episode pays 58.89; a force abort at step 20 paid 3.60 under RT-131's
pricing, i.e. −55 against a +5.72 gain for the whole 30 mm → 5 mm approach.
D-164 removes the −54 forfeit and leaves the written −1.0. No claim is made
here about what reward level results.

## The commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v1
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash given in the handover message. The CODE change of
this run is commit **`ae50431`** (`git log --oneline ae50431 -1` must be in
the history); commits after it add only this expectation file. If `ae50431`
is NOT an ancestor of HEAD, STOP and report the hash.

```
.\scripts\rt_log.ps1 RT-134 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless
```

Let it run **past iteration 150** at minimum — RT-131's collapse finished by
70 and its reward plateaued by 150, so anything shorter cannot be compared.
Full 1500 preferred. Stop with Ctrl+Break (Windows), not Ctrl+C.

After the run, paste `demo_metrics.json` from the run folder into
`rt_logs/inbox.txt` with the log, and bring the same six TensorBoard curves
plus `force_abort_rate` and `mean_max_depth_mm`.

## Points, each PASS/FAIL on its own

1. **P1 — it starts and survives.** Exit 0 or a clean Ctrl+Break, no `nan`,
   no traceback. The exit-code trap applies.
2. **P2 — the code is this code.** `[rt_log]` header shows git `ae50431`; the
   startup report prints `insertion-rung0-start-2026-08-31-p1` and the force
   limit as **300.0 N** with its `[placeholder]` mark;
   `demo_metrics.json` carries `force_abort_f_max_n` = 300.0 and
   `abort_payment` = −1.0 in `rl_placeholders`.
3. **P3 — nothing else moved.** Run folder carries `start+30mm`, `currOFF`,
   `seed42`; the report prints `reset_joint_noise = 0.0`,
   `reset_yaw_noise = 0.0`, `fixture_pos_noise_xy = 0.005`;
   `curriculum_enabled` false and `rung_step_sizes` null in the metrics.
4. **P4 — the start pose lands.** `start_tip_above_entrance_mm` = 30.0,
   `start_pose_solve_unconverged_resets` = 0,
   `start_pose_solve_worst_residual_mm` below 0.05.
5. **P5 — the D-157 tripwire reads zero.**
   `success_depth_invariant_violations` 0 for `recent` AND `cumulative`.
   **Read this BEFORE any success rate.**
6. **P6 — THE D-164 CHECK, and it is TWO-SIDED.** Read
   `force_abort_rate` and `mean_max_depth_mm` as curves over iterations,
   against the RT-131 baseline above.
   * **The pass condition:** `force_abort_rate` does NOT collapse to 0 by
     iteration 70, and `mean_max_depth_mm` stays above 0 past iteration 100.
     The policy keeps making contact.
   * **The named failure mode**, predicted by the four sources in
     `docs/reference/literature_check_abort_return_forfeiture_2026-09-02.md`
     (DeepMimic Table 5, ET-MDP Prop. 1, Kobayashi 2023 §V,
     Beltran-Hernandez Table II): `force_abort_rate` parks near 1.0 with a
     high `force_norm_n` p95. That is ramming become cheap, and it FAILS this
     point just as hard as a collapse to 0 does.
   * Anything between the two is the outcome D-164 wants and is read, not
     scored.
7. **P7 — the six metrics, in order, each with its trend:** Episode Reward →
   Policy Loss → Value Loss → Entropy → Explained Variance → KL. No
   threshold. Red flags per CLAUDE.md. `Policy/mean_noise_std` is read
   alongside entropy, because RT-131's judgement turned on it.
   `success_rate_recent` is read only AFTER P5 and after the six curves.
8. **P8 — the force reading under the 300 N window.** `force_norm_n`
   p50/p95/p99, `max_n`, `over_f_max` and `force_abort_rate`, all READ. The
   RT-131 comparison is the point: p95 was 41.9 N when the policy had stopped
   touching. A p95 that now sits in the 250–300 N class means the window is
   being used as a ceiling and feeds P6's failure mode.
9. **P9 — depth against reward.** `mean_max_depth_mm` beside the episode
   reward. Reward rising while depth stays flat is the RT-107 hack signature;
   reward rising WITH depth is the healthy case.
10. **P10 — the arm does not run away.** `joint_target_lag_rad` `max` and
    `failure_p95`, divided by `action_scale` 0.02 = control steps of
    unwinding. A reading; D-161's second finding is unfixed on purpose.
11. **P11 — the D-153 exposure.** `stage1_lateral_y_mm` present and not
    `{"episodes": 0}`; `over_reach = 0` expected, `> 0` reopens D-153 with a
    number rather than failing the run.

## What this run cannot answer

* Whether the part enters TILTED: still no part-attitude log. The env logs
  the FIXTURE tilt bin. A `play.py` replay is the instrument, own RT number.
* Nothing about the approach plateau. The kernel widths are untouched and
  closed by D-109 point (9); the +5.72 approach gain is the SECOND hypothesis
  and is not tested here.
* Nothing about the SUCCESS lump's undiscounted defect (~2.6x at
  `gamma = 0.99`). Named in D-164, own entry, not changed.
* Nothing about yaw (`reset_yaw_noise = 0.0`), the ladder (off), or the real
  cell's rear wall (D-156, block off).
