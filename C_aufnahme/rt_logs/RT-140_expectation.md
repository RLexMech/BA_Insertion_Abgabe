# RT-140 — expectation (the FRESH contact-penalty run; text = the RT-139 fresh expectation as committed in 28524a8, re-filed unchanged for RT-140 on 2026-09-03 after RT-139 became the resume run)

Written 2026-09-03 on the dev laptop, after the RT-138 verdict
(`rt_logs/VERDICTS.md`, 2026-09-02 14:11:35). This file was OVERWRITTEN:
the earlier RT-139 (a resume of RT-138) was cancelled by the user the same
day, before it ran. `/rt-check` judges the log against THIS file and
nothing else.

## What the run is

**RT-138 repeated with ONE change: a per-step contact-force penalty.**
FORGE form, `-0.2/N above 15 N`, every step, not SAPU-scaled, beside the
unchanged 300 N abort. Home: `docs/decisions_inbox.md`, entry "The contact
force is priced per step ..." (2026-09-03). Everything else as RT-138:
fresh run, Uniform[−30, +30] mm start, D-165 progress term, per-term log,
seed 42, 1024 envs, 1500 iterations, 5 mm fixture noise. RT-138 is the A/B.

## The RT-138 baseline (iteration 1499, 2000-episode window)

`Mean reward` 282.67; success recent 0.3735 / cumulative 0.2036;
`mean_max_depth_mm` 20.78; `force_abort_rate` 0.641; `force_norm_n` p50
304.9 / p95 338.6 / p99 361.8 / max 396.3, success_p95 299.7, `over_f_max`
1282/2000; `success_by_start_height_bin` 82 / 40 / 18 / 36 / 27 / 22 %
(deep → top); `stage1_lateral_y_mm` p50 0.0 / max 17.8 / `over_reach` 5;
`joint_target_lag_rad` max 1.227 / failure_p95 0.359; value loss 2261.6;
entropy −3.74; noise std 0.21; `Episode_Reward/total` 219.56 vs
`reward_terms_recent_mean.total` 216.16.

## The arithmetic this run is read against (per episode)

Hard pressing as in RT-138 (305 N, 13 steps): contact ≈ −754 against an
expected lump of +207 (37 % × 560). Gentle insertion as the script did
(~100 N over ~10 contact steps): ≈ −170 against +560 on success.
Standstill at the start: ≈ +59, no penalty. Ordering the reward now
states: gentle success > standstill > hard pressing.

## Points, each PASS/FAIL on its own

1. **P1 — the code is this code.** `[rt_log] git:` = the hash in the
   handover; startup prints `contact penalty (FORGE form, 2026-09-03):
   -0.2/N above 15.0 N, every step, not SAPU-scaled ... ON`; the placeholder
   block lists `contact_penalty_threshold_n = 15.0`; `SAMPLED per episode
   ... uniform in [-30.000, +30.000] mm`; `insertion-rung0-start-2026-08-31-p1`.
2. **P2 — the term is live.** `Episode_Reward/contact` exists on the curve
   and is NEGATIVE from the first iteration on (the RT-138 policy shape
   presses from step 1; a flat 0.0 = the term is fed zeros = FAIL).
   `demo_metrics.json` `reward_terms_recent_mean.contact` present.
3. **P3 — THE TWO-SIDED READING (D-164's pass condition, RT-131's failure
   mode).** `force_abort_rate` must NOT collapse to 0 WITH
   `mean_max_depth_mm` at the 7.5 mm floor and success 0 — that is the
   standstill (Factory Table IV, RT-131) and FAILS the run. The other side
   is RT-138 itself: abort rate ≥ 0.6 with p50 ≥ 300 N = the penalty did
   not bite at 0.2 → the decade row 0.02 is NOT the answer, the scale
   goes UP, not down (read, then decide).
4. **P4 — THE CLAIM: force comes down without success going away.**
   `force_norm_n` p95 below 300 N (RT-138: 338.6) AND `success_rate_recent`
   not below 0.37 (RT-138). Both, or the point is VERLETZT with the two
   numbers quoted. p50 / p99 / max / `over_f_max` / `force_abort_rate`
   read beside them.
5. **P5 — the SBC table.** `success_by_start_height_bin` all six rows,
   top bin (20..30) against 22 %, deep bin against 82 %. Read, and the top
   bin is the thesis number.
6. **P6 — D-157 tripwire.** `success_depth_invariant_violations` 0 / 0.
7. **P7 — the per-term account.** `reward_terms_recent_mean.total` vs
   `Mean reward`, both quoted; the RT-138 gap (216 vs 283) is still
   unexplained and is reported, not scored. `contact` ≈ −0.2 × Σ max(0,
   F−15) per episode: sanity-read against force p50 and episode length.
8. **P8 — the six curves, in order, with trend:** Episode Reward →
   Policy Loss → Value Loss → Entropy → Explained Variance → KL, plus
   `Policy/mean_noise_std`. Value loss against 2261 (D-116). Entropy
   falling toward 0 with noise std well under 0.21 = exploration dying,
   red flag on its own.
9. **P9 — the arm.** `joint_target_lag_rad` max / failure_p95 vs 1.227 /
   0.359; `stage1_lateral_y_mm` p50 / max / `over_reach` vs 0.0 / 17.8 / 5.
   Readings.

## The commands (training PC, repo root, conda env active)

CORRECTED 2026-09-03 after RT-139a/b both failed on argparse: `zero_agent.py`
has NO hydra override mechanism (`env.foo=bar`), only its own named flags
(`--start-tip-above-entrance-mm`, `--print-force`, ...) — checked against
its source, no `contact_penalty_threshold_n` flag exists there. Only
`train.py` accepts hydra overrides. The smoke test drops the (unsupported)
start-height override; the guard test uses `train.py --max_iterations 1`,
which builds the env (where the guard runs) and fails at construction,
before any real training compute, if the guard does not fire.

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v1
git pull
git rev-parse --short HEAD          # expected: the hash in the handover
.\scripts\rt_log.ps1 RT-139a python scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --max-steps 10 --headless
.\scripts\rt_log.ps1 RT-139b python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --max_iterations 1 env.contact_penalty_threshold_n=400.0
.\scripts\rt_log.ps1 RT-140 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless env.start_tip_above_entrance_low=-0.030
```

RT-139a: the smoke — must print the contact-penalty startup line and exit
0. RT-139b: the guard — a threshold above F_max must be REFUSED (non-zero
exit, message names `contact_penalty_threshold_n` and
`force_abort_f_max_n`); exit 0 here = the guard is not wired = stop.
CORRECTED 2026-09-03 after RT-139b: hydra type-checks the override against
the `float` annotation, so `400` (int) dies in hydra before the guard runs;
the value must be written `400.0`.

Bring: the `[rt_log]` header, the startup block (placeholder lines, the
contact line), the first iteration block, the last iteration block,
`demo_metrics.json`.

## What this run cannot answer

* The right scale — one value (FORGE's) at one threshold (the user's).
* Whether the gentle path is FOUND or only priced: needs the tip trace
  (still no instrument).
* The RT-138 P7 gap and the undiscounted lump (D-164's consequence).
