# RT-137 — expectation, written BEFORE the run

Written 2026-09-02 on the dev laptop. Plan `so-was-ich-jetzt-virtual-goblet.md`,
steps B + A. `/rt-check` judges the log against THIS file and nothing else.
**Run only after RT-136 has PASSED** — P1 there is the JIT gate for the
same code.

## What the run is

**RT-134 repeated with TWO changes, one of them an instrument:**

* **B (instrument, no reward value changes):** the reward is logged per
  term — `Episode_Reward/kernels, engaged, success, progress, time,
  action_rate, success_lump, abort, total` — one TensorBoard curve each,
  the per-episode SUM; `demo_metrics.json` carries
  `reward_terms_recent_mean`.
* **A (D-165, the hypothesis):** the proxy's depth-progress term,
  `w_depth_progress = 100.0` per metre of NEW gated max depth, SAPU-scaled,
  outside the lump. Zero anywhere the part has not entered.

Same seed 42, same 1024 envs, same +30 mm fixed start, same 5 mm fixture
noise, same 300 N / −1.0 abort, same kernels, same D-164 `truncated` abort.

## THE QUESTION THIS RUN ANSWERS — and the honest prior

Does anything ENTER once entry pays? D-165 states its own limit: the term
pays nothing until a corner is in, and from RT-134's 4.88 mm median miss
the window is 0.8 mm x 0.29 mm. **The expected outcome of A alone is
RT-134's pose again** (`mean_max_depth_mm` ~0, `Episode_Reward/progress`
~0). That is not a failure of the run; it is the A/B the plan asks for
before the start-distribution change (plan step C). What this run MUST
deliver either way is the per-term account of where RT-134's 55.94 came
from.

## The RT-134 baseline (iteration 191, `demo_metrics.json` at 13 500 episodes)

`Train/mean_reward` 55.94; `mean_max_depth_mm` 0.0; `force_abort_rate` 0.0;
`force_norm_n` p50 132.6 / p95 166.2 / p99 179.2 / max 205.0, `over_f_max`
0; `stage1_lateral_y_mm` p50 4.88 / max 9.34, `over_reach` 72/2000;
`joint_target_lag_rad` max 0.0772 (3.86 steps), `failure_p95` 0.0153;
`Policy/mean_noise_std` 0.02; value loss 0.0061; entropy −16.4461.

## The commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v1
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash from the handover; `28e0d74` in the history.

```
.\scripts\rt_log.ps1 RT-137 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless
```

Past iteration 150 at minimum; RT-134's snapshot was 191, so 200+ makes
the comparison direct. Paste `demo_metrics.json` and the iteration block
into `rt_logs/inbox.txt`; bring the six curves AND every
`Episode_Reward/*` curve, plus `force_abort_rate` and `mean_max_depth_mm`.

## Points, each PASS/FAIL on its own

1. **P1 — starts and survives.** No `nan`, no traceback; a hand stop is
   not a finding. The JIT trap was RT-136's P1; a JIT error here after
   RT-136 passed is a NEW finding (different call path: batched, 1024
   envs).
2. **P2 — the code is this code.** `[rt_log]` header git hash from the
   handover; startup report prints `insertion-rung0-start-2026-08-31-p1`,
   300.0 N `[placeholder]`, and `cfg.w_depth_progress` = **100.0** with its
   `[proxy]` mark; `demo_metrics.json` `proxy_task_values` carries the same.
3. **P3 — nothing else moved.** Run folder `start+30mm`, `currOFF`,
   `seed42`; reset noises 0.0 / 0.0 / 0.005; `curriculum_enabled` false,
   `rung_step_sizes` null; `abort_payment` −1.0.
4. **P4 — start pose.** 30.0 mm, `unconverged_resets` 0, worst residual
   below 0.05 mm.
5. **P5 — D-157 tripwire.** `success_depth_invariant_violations` 0 / 0.
   Read BEFORE any success rate.
6. **P6 — THE INSTRUMENT (B) IS SOUND. This is the point the run must
   pass regardless of A.**
   * `demo_metrics.json` has `reward_terms_recent_mean` with `episodes`
     > 0 and all nine keys (`kernels, engaged, success, progress, time,
     action_rate, success_lump, abort, total`), none `null`.
   * **Identity:** `reward_terms_recent_mean.total` equals rsl_rl's
     `Mean reward` of the same region to within the window difference
     (both are per-episode undiscounted returns; RT-134's 55.94 is the
     order). A `total` off by more than a few units means the rows do not
     add up to what is paid, and the instrument FAILS.
   * `time` ≈ −1.0 per episode (256 x −1/256; slightly less in magnitude
     for episodes that end early). `abort` = −1.0 x `force_abort_rate`
     (0 if no aborts). `success_lump` = 0 while `success_rate` = 0.
7. **P7 — THE D-165 READING, two-sided like RT-134's P6.**
   * `Episode_Reward/progress` over iterations and
     `reward_terms_recent_mean.progress`: **0 throughout** = RT-134's pose
     again, the STATED prior, read as "A alone does not move the part";
     **> 0 at any iteration** = a corner entered — read `mean_max_depth_mm`
     beside it (progress = 100 x max_depth in metres, so 0.1 per
     millimetre of mean new depth).
   * The named failure mode: `progress` > 0 with `success_depth_invariant
     _violations` > 0 or with `stage1_lat_y` p50 above `stage1_allows_mm`
     — depth being paid where the D-157 box should have refused it. FAIL.
8. **P8 — the six metrics, in order, with trend:** Episode Reward →
   Policy Loss → Value Loss → Entropy → Explained Variance → KL.
   `Policy/mean_noise_std` beside entropy. Value loss is the one to watch:
   the progress row adds variance to the return if it fires (D-116, no
   value normalisation). `success_rate_recent` only after P5.
9. **P9 — where RT-134's 55.94 went.** With B the account is READ, not
   computed: `kernels` (RT-134's back-solved ~0.23/step x 255 ≈ 59 is the
   order), `time` ≈ −1.0, `action_rate` (the number that explains 55.94 <
   57.66 if it does), `engaged`/`success`/`lump` 0.
10. **P10 — force under the 300 N window.** p50/p95/p99/max, `over_f_max`,
    `force_abort_rate`, against RT-134's 132.6 / 166.2 / 179.2 / 205.0.
11. **P11 — depth against reward.** Reward rising with depth flat is the
    RT-107 signature; with the progress row logged, reward rising while
    `progress` stays 0 and `kernels` rises is the plateau being climbed
    — read, not scored.
12. **P12 — the arm does not run away.** `joint_target_lag_rad` max and
    `failure_p95` (RT-134: 3.86 / 0.76 steps). A reading.
13. **P13 — D-153 exposure.** `stage1_lateral_y_mm` present; `over_reach`
    (RT-134: 72/2000). `> 0` keeps D-153 open with its number.

## What this run cannot answer

* Whether the policy would learn the last millimetres from INSIDE — that
  is plan step C (start-height sampling), its own inbox entry and run.
* The size of `w_depth_progress` (100 is `[proxy]`); one run at one value
  sizes nothing.
* Part attitude at the end of an episode — still no instrument.
