# RT-138 — expectation, written BEFORE the run

Written 2026-09-02 on the dev laptop, after the RT-137 mid-run reading
(iteration 241: progress 0, part on the outer rim 16.6 mm beside the axis)
and the user's decision to run plan step C. `/rt-check` judges the log
against THIS file and nothing else.

## What the run is

**RT-137 repeated with ONE change: the start height is sampled** (SBC,
IndustReal sec. IV.G). `env.start_tip_above_entrance_low=-0.030` draws each
episode's start uniformly in [−30, +30] mm above the opening plane. Half the
episodes begin INSIDE the pocket, on the axis, up to 3 mm above the success
band. D-165 (`w_depth_progress = 100`), the per-term reward log, seed 42,
1024 envs, 5 mm fixture noise, 300 N / −1.0 abort, the kernels, the D-164
`truncated` abort: all as in RT-137.

Home of the change: `docs/decisions_inbox.md`, entry "Step C: the rung-0
start height is SAMPLED ..." (2026-09-02).

## THE QUESTION THIS RUN ANSWERS — and the honest prior

Does a policy that sometimes starts inside learn to finish from inside, and
does anything of that reach the OUTSIDE start? Two readings, in this order:

1. The inside bins (`success_by_start_height_bin`, starts below 0): the
   progress term now has something to pay. Success here is the EASY end.
2. The top bin (starts at +20..+30 mm and above, RT-137's task): success
   here is the claim. **The stated prior is the IndustReal "partially-
   inserted" overfit: success in the low bins, 0 in the top bin, the part
   still on the outer rim for the outside starts.** That outcome is the
   A/B the plan asks for; it is not a failure of the run.

## The RT-137 baseline (iteration 241)

`Mean reward` 52.97; `reward_terms_recent_mean` kernels 56.39 / time −0.996
/ action_rate −2.45 / progress 0 / total 52.94; `mean_max_depth_mm` 0.0;
`force_abort_rate` 0.0; `force_norm_n` p50 102.3 / p95 133.1 / p99 157.2 /
max 190.0; `stage1_lateral_y_mm` p50 16.59 / max 26.36, `over_reach`
2000/2000; `joint_target_lag_rad` max 0.0787 (3.94 steps); value loss
0.0271; entropy −17.91; noise std 0.02.

## The commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v1
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash in the handover message. NOT `9c0a2ad`: that
first attempt died at cfg merge (Isaac Lab's hydra refuses a float over a
`None` default, PROBLEMS.md 2026-09-02); the fix makes the OFF value the
upper bound itself and is the commit after it.

```
.\scripts\rt_log.ps1 RT-138 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless env.start_tip_above_entrance_low=-0.030
```

Past iteration 250 at minimum (RT-137 was read at 241). Paste
`demo_metrics.json` and the last iteration block into `rt_logs/inbox.txt`;
bring the six curves AND `Episode_Reward/progress`, `mean_max_depth_mm`,
`start_height_mean_mm`, `force_abort_rate`.

## Points, each PASS/FAIL on its own

1. **P1 — starts and survives.** No `nan`, no traceback, NONE of the four
   start-range refusals fires (all orientation noises are 0.0 in the cfg;
   `fixture_pos_noise_xy` 0.005 is allowed). A hand stop is not a finding.
2. **P2 — the code is this code.** `[rt_log]` git hash from the handover;
   startup prints `rung-0 start height SAMPLED per episode (SBC, plan step
   C): uniform in [-30.000, +30.000] mm`; `insertion-rung0-start-2026-08-31-p1`;
   `cfg.w_depth_progress` = 100.0 `[proxy]`.
3. **P3 — the range is on disk.** Run folder tag `start-30..+30mm`;
   `demo_metrics.json` `start_tip_above_entrance_mm` 30.0 AND
   `start_tip_above_entrance_low_mm` −30.0; `success_by_start_height_bin`
   present with 6 rows (edges −20/−10/0/10/20/30 mm), every row's
   `episodes` > 0 (the draw is uniform, an empty bin means it is not).
   `start_height_mean_mm` on the curve near 0 (centre of the range; the
   fixed start would read a flat +30.0).
4. **P4 — start pose, now at heights down to −30 mm.**
   `start_pose_solve_unconverged_resets` 0, worst residual below 0.05 mm.
   The dls solve has only been driven to negative heights by RT-119 /
   RT-133 (seated, −34 mm) as a teleport; this is its first use as a
   sampled reset. A nonzero unconverged count is a finding on its own.
5. **P5 — D-157 tripwire.** `success_depth_invariant_violations` 0 / 0.
   Read BEFORE any success rate.
6. **P6 — THE SEED HOLDS (the run's own counter-proof).** The do-nothing
   floor of `mean_max_depth_mm` under Uniform[−30, +30] is
   E[max(0, −h)] = **7.5 mm**. If `Episode_Reward/progress` at the FIRST
   iterations reads ≈ 0.75 per episode (= 100 × 0.0075) with
   `mean_max_depth_mm` ≈ 7.5 and nothing else moving, the teleport is being
   paid as progress and the seed is NOT working — FAIL, regardless of what
   the policy does later. Expected instead: progress at iteration 0 well
   below 0.75, and every later progress value read as the policy's own.
7. **P7 — the per-term account still adds up.** `reward_terms_recent_mean
   .total` ≈ rsl_rl `Mean reward` (RT-137: 52.94 vs 52.97). `time` ≈ −1.0
   for full episodes, shorter in magnitude where episodes end early
   (success or abort); `abort` = −1.0 × `force_abort_rate`; `success_lump`
   > 0 iff `success_rate` > 0.
8. **P8 — THE SBC READING, two-sided.** `success_by_start_height_bin`:
   * inside bins (start below 0) with success > 0 = the progress term found
     something to pay and the policy finished from inside;
   * top bin (20..30+) success = the task; 0 there with success in the low
     bins = the NAMED prior (partially-inserted overfit), read as "C teaches
     the inside, not yet the entry".
   * `mean_max_depth_mm` read AGAINST the 7.5 mm floor: only the excess is
     learned depth.
   * The named failure mode: success with `stage1_lat_y` p50 above
     `stage1_allows_mm` (8.7) or with P5 > 0 — depth paid where the D-157
     box should have refused it. FAIL.
9. **P9 — the six metrics, in order, with trend:** Episode Reward → Policy
   Loss → Value Loss → Entropy → Explained Variance → KL.
   `Policy/mean_noise_std` beside entropy. Value loss is the one to watch
   (D-116, no value normalisation): the return now has TWO populations
   (inside starts that can reach the lump, outside starts that cannot), so
   a value loss well above RT-137's 0.0271 is expected and is a reading,
   not a red flag by itself; explained variance is what says whether the
   critic separates them.
10. **P10 — force under the 300 N window.** p50/p95/p99/max, `over_f_max`,
    `force_abort_rate`, against RT-137's 102.3 / 133.1 / 157.2 / 190.0 /
    0 / 0.0. Inside starts with 0.8 mm play WILL touch walls; an abort rate
    > 0 is expected early and is read against its trend, not scored.
11. **P11 — where the outside starts go.** `stage1_lateral_y_mm` p50 / max
    and `over_reach` (RT-137: 16.59 / 26.36 / 2000). The metric is over ALL
    episodes; a p50 that DROPS below RT-137's says the outside starts moved
    too. Read, not scored.
12. **P12 — the arm does not run away.** `joint_target_lag_rad` max and
    `failure_p95` (RT-137: 0.0787 / 0.99 steps). A reading.

## What this run cannot answer

* The right value of `low` or its schedule — one run at one value sizes
  nothing (inbox entry, "NOT decided").
* Whether the tilt seen at iteration 50 of RT-137 is reward-driven or the
  elbow's arc — still no per-step tip trace.
* Part attitude at the end of an episode — still no instrument.
