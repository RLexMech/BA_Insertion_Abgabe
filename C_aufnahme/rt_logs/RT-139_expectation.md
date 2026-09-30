# RT-139 — expectation, written BEFORE the run

Written 2026-09-03 on the dev laptop, after the RT-138 verdict
(`rt_logs/VERDICTS.md`, 2026-09-02 14:11:35). This file was OVERWRITTEN
TWICE: the first RT-139 (a plain resume of RT-138) was cancelled by the
user on 2026-09-03; the second (a FRESH run with the contact penalty) was
replaced the same night, before it started, by the user's decision to
RESUME RT-138 AND add the penalty. `/rt-check` judges the log against
THIS file and nothing else.

## What the run is

**RT-138 CONTINUED from its own last checkpoint, with ONE reward change:
a per-step contact-force penalty.** FORGE form, `-0.2/N above 15 N`,
every step, not SAPU-scaled, beside the unchanged 300 N abort. Home:
`docs/decisions_inbox.md`, entry "The contact force is priced per step
..." (2026-09-03). The resume reading of D-110 (1) is the inbox entry
"RT-139 continues RT-138 from its own checkpoint ..." (2026-09-03): fresh
training is about the PROXY checkpoint, not about resuming an own run.

Checkpoint: `logs/rsl_rl/ur5e_insertion/09-02_14-20-45_offset5mm_currOFF_start-30..+30mm_seed42/model_1499.pt`
(training PC, given by the user 2026-09-03). rsl_rl `OnPolicyRunner.load()`
restores policy, optimizer and the iteration counter. Everything else as
RT-138: Uniform[−30, +30] mm start, D-165 progress term, per-term log,
seed 42, 1024 envs, 5 mm fixture noise. `--max_iterations 500`.

This is NOT the A/B the fresh run would have been. It is a BEFORE/AFTER:
RT-138's iteration 1499 is the "before", this run's last iteration the
"after". What the resume costs, stated up front: the loaded policy has
almost no exploration left (noise std 0.21, entropy −3.74) and its value
function is calibrated to the OLD reward — a value-loss jump in the first
iterations is expected, not a defect.

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
states: gentle success > standstill > hard pressing. The loaded policy
starts at "hard pressing", i.e. at the BOTTOM of that ordering: its first
`Mean reward` must be far below 282.67.

## Points, each PASS/FAIL on its own

1. **P1 — the code is this code, and the checkpoint is THAT checkpoint.**
   `[rt_log] git:` = the hash in the handover; startup prints `contact
   penalty (FORGE form, 2026-09-03): -0.2/N above 15.0 N, every step, not
   SAPU-scaled ... ON`; the placeholder block lists
   `contact_penalty_threshold_n = 15.0`; `SAMPLED per episode ... uniform
   in [-30.000, +30.000] mm`; `insertion-rung0-start-2026-08-31-p1`. The
   resume line names `09-02_14-20-45_offset5mm_currOFF_start-30..+30mm_seed42`
   and `model_1499.pt`. A NEW run folder is created (the RT-138 folder is
   not written into).
2. **P1b — the iteration counter (UNVERIFIED semantics, read, then
   decide).** The first iteration block's number: 1500 = the counter was
   restored and `--max_iterations 500` means "500 more" (run ends at
   1999); 0 = the counter was NOT restored (then the policy weights may
   still be loaded — P2 decides). If the run STOPS at once because
   1500 ≥ 500, that is the third reading: `--max_iterations` is a total,
   and the next command needs 2000.
3. **P2 — the term is live AND the policy is the loaded one.**
   `Episode_Reward/contact` exists and is STRONGLY NEGATIVE in the FIRST
   iteration (the loaded policy presses at ~305 N from step 1; per the
   arithmetic, ≈ −700 per episode; a flat 0.0 = fed zeros = FAIL). First
   iteration `Mean reward` far below 282.67 (the penalty bites the old
   habit). If instead the first iteration reads like a FRESH policy
   (reward ≈ +59 standstill, contact ≈ 0, depth 0) the checkpoint did not
   load = FAIL on P1.
4. **P3 — THE TWO-SIDED READING (D-164's pass condition, RT-131's failure
   mode).** `force_abort_rate` must NOT collapse to 0 WITH
   `mean_max_depth_mm` at the 7.5 mm floor and success 0 — that is the
   standstill (Factory Table IV, RT-131) and FAILS the run; for a loaded
   policy this reads as "the penalty taught it to retreat and stop". The
   other side is RT-138 itself: abort rate ≥ 0.6 with p50 ≥ 300 N at the
   END = the penalty did not bite at 0.2 → the scale goes UP, not down.
5. **P4 — THE CLAIM: force comes down without success going away.**
   Last iteration: `force_norm_n` p95 below 300 N (RT-138: 338.6) AND
   `success_rate_recent` not below 0.37 (RT-138). Both, or the point is
   VERLETZT with the two numbers quoted. p50 / p99 / max / `over_f_max` /
   `force_abort_rate` read beside them.
6. **P5 — the SBC table.** `success_by_start_height_bin` all six rows,
   top bin (20..30) against 22 %, deep bin against 82 %. Read, and the top
   bin is the thesis number. `force_abort_by_start_height_bin` (new key,
   commit beddc6e, UNVERIFIED) present beside it, same six edges.
7. **P6 — D-157 tripwire.** `success_depth_invariant_violations` 0 / 0.
8. **P7 — the per-term account.** `reward_terms_recent_mean.total` vs
   `Mean reward`, both quoted; the RT-138 gap (216 vs 283) is still
   unexplained and is reported, not scored. `contact` ≈ −0.2 × Σ max(0,
   F−15) per episode: sanity-read against force p50 and episode length.
9. **P8 — the six curves, in order, with trend:** Episode Reward →
   Policy Loss → Value Loss → Entropy → Explained Variance → KL, plus
   `Policy/mean_noise_std`. Value loss: a JUMP in the first iterations is
   EXPECTED (old value function, new reward); what is scored is whether it
   comes back down by the end (against 2261). Entropy / noise std: the
   loaded policy starts at −3.74 / 0.21; if they do not RISE at all while
   force stays ≥ 300 N, exploration is dead and the fresh run is plan B.
10. **P9 — the arm.** `joint_target_lag_rad` max / failure_p95 vs 1.227 /
    0.359; `stage1_lateral_y_mm` p50 / max / `over_reach` vs 0.0 / 17.8 /
    5. Readings.

## The commands (training PC, repo root, conda env active)

RT-139a (smoke, `zero_agent.py`) and RT-139b (guard, `train.py
--max_iterations 1 env.contact_penalty_threshold_n=400.0`) are DONE and
PASS (`rt_logs/VERDICTS.md`, 2026-09-03 01:03:34 and 01:36:44); the smoke
ran on 28524a8, the env code is unchanged since. `--load_run` is a regex
in Isaac Lab's `get_checkpoint_path` and `sort_alpha=False` picks the
newest match by mtime; the full folder name matches itself.
`--checkpoint model_1499.pt` pins the file.

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v1
git pull
git rev-parse --short HEAD          # expected: the hash in the handover
.\scripts\rt_log.ps1 RT-139 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --resume --load_run 09-02_14-20-45_offset5mm_currOFF_start-30..+30mm_seed42 --checkpoint model_1499.pt --max_iterations 500 env.start_tip_above_entrance_low=-0.030
```

Bring: the `[rt_log]` header, the startup block (placeholder lines, the
contact line, the resume/load line), the first iteration block, the last
iteration block, `demo_metrics.json`.

## What this run cannot answer

* The right scale — one value (FORGE's) at one threshold (the user's).
* Whether the gentle path is FOUND or only priced: needs the tip trace
  (still no instrument).
* What a FRESH policy would do under the same penalty (the cancelled A/B).
* The RT-138 P7 gap and the undiscounted lump (D-164's consequence).
