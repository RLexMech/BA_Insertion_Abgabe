# RT-156 — expectation, written BEFORE the run

Written 2026-09-05 on the dev laptop, while RT-154 runs. `/rt-check` judges
the log against THIS file and nothing else. Plan:
`C:\Users\PCUser\.claude\plans\c-users-pcuser-desktop-claude-bachelor-piped-brooks.md`
(approved 2026-09-05), hypothesis H-A.

## The one hypothesis

**"The RT-138 recipe transfers to the OSC controller."** RT-138 is the only
PPO run that ever reached success on the real part (37.35 % recent, top bin
+20..+30 mm 22.4 %, `docs/figures/RT-138_demo_metrics.json`) and it differs
from every 0 % run in ONE thing: the start height is SAMPLED uniform in
[−30, +30] mm (IndustReal sec. IV.G, `env.start_tip_above_entrance_low=-0.030`).
Every fixed-start run — RT-131, RT-134, RT-137 (joint_pd) and RT-148b (osc) —
read 0.0. The OSC controller has never trained with the sampled start.

Three flags move, one hypothesis. Why the bundle is one hypothesis and not
three: `contact` and `far` read EXACTLY 0.0 over 2000 episodes in RT-148b
(`RT-148b_demo_metrics.json`, `reward_terms_recent_mean`), so switching them
off changes nothing for the behaviour class RT-148b showed. They are switched
off because (a) RT-138 ran without both, and (b) the contact threshold 15 N
lies UNDER the seat force the scripted OSC insertion needed (RT-150: max
16.72 N) — a deep start that pushes to the seat would pay 0.2/N for doing
the task. The inbox entry that owns the 0.2 is itself marked "MEASURED TWICE
AND FAILED BOTH WAYS" (RT-139/140).

1024 envs, not 128: RT-138's count. The comparison target of this run is
RT-138, not RT-154. Whether the batch size alone changes sigma is RT-154's
open question and is NOT answered here.

Everything else as RT-154: osc, kp 100, 15 Hz, 256 steps (17.07 s),
`action_rate_scale` 0.0034 (D-168), seed 42, fixture noise 5 mm xy,
`w_depth_progress` 100 `[proxy]`, 300 N abort as `truncated`.

Prerequisites: RT-154 judged (sigma trend known), RT-155 PASS (success
wiring under OSC).

## Commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash in the handover message (`a123c4a` if nothing is
pushed in between).

```
.\scripts\rt_log.ps1 RT-156 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max_iterations 1500 env.start_tip_above_entrance_low=-0.030 env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0
```

Hydra floats need the decimal point (`0.0`, not `0`; PROBLEMS.md RT-139b).
Run time: UNKNOWN — 1024 envs have never run under OSC. RT-104 measured
72–219 ms per RL step at 128 envs under joint_pd; RT-148b 3.69 s per
iteration at 128 envs under OSC. Read the seconds per iteration off the
first block and report them.

Paste the log (`rt_logs/RT-156.short.txt`) and `demo_metrics.json` into
`rt_logs/inbox.txt`. First block (iteration 0), the block nearest 500, and
the last block are all needed.

## Points, each PASS/FAIL on its own

1. **P1 — starts and survives.** Exit 0, no traceback, no `nan`, none of
   the four start-range refusals fires. 1500/1500 iterations, or a hand
   stop that is named as such.
2. **P2 — the code and the flags are on disk.** `[rt_log]` header shows
   the handover hash. Startup prints `rung-0 start height SAMPLED per
   episode (SBC, plan step C): uniform in [-30.000, +30.000] mm`; the
   contact line ends `*** OFF, scale 0 -- the RT-138 reward ***`; the
   leash line ends `*** OFF, scale 0 -- the RT-140 reward ***`
   (`insertion_env.py:3045-3061`); the `--- controller ---` block shows
   mode `osc` / kp 100 and no `DRIVES NOT INERT`. `demo_metrics.json`: `start_tip_above_entrance_low_mm`
   −30.0, `num_envs` 1024, `control_mode` osc, `policy_rate_hz` 15.
3. **P3 — the range is on disk.** `success_by_start_height_bin` has 6 rows
   and every row's `episodes` > 0; `start_height_mean_mm` on the curve near
   0 (fixed start reads +30.0).
4. **P4 — start pose down to −30 mm under OSC.**
   `start_pose_solve_unconverged_resets` 0, worst residual < 0.05 mm
   (RT-138: 0.050 mm). The IK reset is controller-independent, so a change
   here is a finding on its own.
5. **P5 — D-157 tripwire.** `success_depth_invariant_violations` 0 / 0,
   read BEFORE any success rate.
6. **P6 — the seed is not paid as progress.** `Episode_Reward/progress` at
   iteration 0 well below 0.75 (= 100 × E[max(0, −h)] = 100 × 7.5 mm), as
   in RT-138 P6.
7. **P7 — THE HYPOTHESIS, deep bins.** `success_by_start_height_bin`, the
   bin −30…−20 mm: **PASS ≥ 0.50** (RT-138: 0.825), **FAIL < 0.10**,
   between = the run does not decide. Read at the last block. If the run is
   stopped early, read the latest block and say so.
   * The 0.50 line is MINE: RT-138's 0.825 minus a margin for the softer
     controller (kp 100 pushes at most ~18–24 N where the stiff PD pushed
     305 N). Named as an order-of-magnitude line, not a measured threshold.
8. **P8 — the top bin is READ, not predicted.** Bin +20…+30 mm success
   (RT-138: 0.224). This number decides the branch in the plan:
   * deep bins learn, top bin ≈ 0 → the lateral/yaw blindness above the
     entrance (RT-151c/RT-153) is confirmed as the next blocker → H-B.
   * top bin > 0.20 → the reward is sufficient; next is the curriculum.
   * all bins 0 → controller/physics under OSC first, not the reward.
9. **P9 — sigma survives.** `Mean action noise std` at iteration 500
   > 0.30 (the RT-154 line, same ownership: MINE, order of magnitude).
   If RT-154 has already shown sigma dying under 0.0034 at 128 envs, this
   point is read against RT-154, not against 0.30, and the difference is
   the batch size — report, do not conclude.
10. **P10 — the account adds up.** `reward_terms_recent_mean.total` ≈
    rsl_rl `Mean reward` (RT-138: 216.16 vs 220, gap named there);
    `contact` and `far` rows EXACTLY 0.0 (scale 0); `success_lump` > 0 iff
    `success_rate` > 0; `abort` = −1.0 × `force_abort_rate`.
11. **P11 — force under OSC with deep starts.** `force_norm_n` p95 < 25 N
    (RT-148a read p99 22.4 / max 24.2 N as the OSC ceiling; RT-150 seated
    at 16.72 N). p95 above 25 N = the stall-force model (Lambda·kp·Delta)
    does not hold in contact — a finding for D-166/D-167, not a reward
    finding. `force_abort_rate` expected 0.0 (D-167: the 300 N abort cannot
    fire under kp 100).
12. **P12 — the six curves, in order, with trend.** Episode Reward →
    Policy Loss → Value Loss → Entropy → Explained Variance → KL. EV and KL
    are NOT logged by rsl_rl 3.0.1 (VERDICTS RT-148b NACHTRAG 2 (b)); say
    "not logged" for both, read `Loss/learning_rate` as the indirect KL
    readout. Value loss: two return populations (inside starts reach the
    lump, outside starts do not), so a value loss far above RT-148b's
    0.0549 is expected (RT-138: 2261) and is a reading, not a red flag.

## What would make each point wrong

1. → crash or flag typo. Fix, rerun, same number only if nothing ran.
2. → no pull, or a flag not applied (RT-139b-class Hydra error). Rerun.
3./4. → instrument or IK finding, stops the reading of 7–12.
5. → P5 > 0 voids every success number in the run (RT-107 class).
7. → FAIL: the deep-start seating that the stiff PD achieved does not
   transfer to kp 100. Next suspect is the controller (D-166 `F_search`,
   kp), NOT the reward: RT-150 seated only with Delta at the step limit.
8. → nothing; it is the branch variable.
9. → see RT-154; if sigma dies here but not in RT-154, the batch size is
   the confound and gets its own run.

## Read, NOT expected

`mean_max_depth_mm` against the 7.5 mm do-nothing floor (only the excess is
learned); `stage1_lateral_y_mm` p50/max and `over_reach` (RT-148b: 51.3 mm /
1994); `mean_success_episode_steps` (RT-138: 18.1 — the deep starts fall
in); `action_abs_max_mean`, `action_sat_frac` (D-168 tripwires, first
baseline under 1024 envs); seconds per iteration.

## What this run cannot answer

* The right value or schedule of `low` — one value, one run.
* Whether the top-bin number, if > 0, comes from the reward or from the
  walls guiding a part that happens to be centred (fixture noise 5 mm is
  uniform, so ~6 % of episodes start within ±0.29 mm laterally by chance).
  A per-episode lateral-error-at-entry trace does not exist.
* Yaw (RT-152 deferred).
