# RT-107 — expectation, written BEFORE the run

The 15-minute REWARD-HACK PROBE (user, 2026-08-30): a short run before the
full one, read only for signs that the policy is farming the reward instead of
inserting the part. Written on the dev laptop before the run exists;
`/rt-check` judges the log against THIS file and nothing else.

## Why this run exists

`insertion_math.compute_rewards_insertion` carries its own exploit analysis in
its docstring. It names five exploits as dissolved and **one as NOT defeated**:

> "**Existing instead of solving** — NOT defeated, named. Since everything is
> positive, merely existing is rewarded."

Two more are open elsewhere, both recorded, neither instrumented:

* **The clamp exploit** (D-152, and the same gap for the fixture in D-153):
  `insertion_sdf.clamp_outside` clamps a far-outside point reading negative to
  zero, "so a bad pose scores like a seated one. No instrument for this exists
  yet."
* **Engaged-parking.** D-109 (5) pays `w_engaged` **per timestep with no
  latch**, and the same docstring notes the two bonuses are about two thirds
  of the episode return. A policy that reaches engaged depth and STAYS there
  collects that bonus every step without ever entering the success band.

Fifteen minutes is enough to see the shape of the reward curve against the
depth curve. It is NOT enough to judge learning — that is RT-108's job.

## Configuration

Identical to RT-108, so the probe and the full run are the same experiment:
`fixture_pos_noise_xy = 0.005`, `reset_joint_noise = 0.0`,
`reset_yaw_noise = 0.0`, both fixture-orientation noises 0.0,
`curriculum_enabled = False`, block off. Agent config untouched.
**Nothing is tuned. A hyperparameter change here would make RT-108
incomparable to it.**

## The numbers the hacks are read against

| quantity | value | home |
|---|---|---|
| seat depth | 36.00 mm | `POCKET_SEAT_DEPTH` |
| engaged depth (30 %) | **10.80 mm** | `ENGAGED_DEPTH` |
| success band | 33.00 – 36.00 mm | `SEATED_SUCCESS_DEPTH` … seat |
| SAPU discard threshold | 0.2938 mm interpenetration | `INTERPEN_THRESH` |
| action scale | 0.02 rad | `action_scale` |

## Points

1. **P1 — it runs.** Exit 0 or a clean Ctrl+Break, no `nan`, no traceback.
2. **P2 — a metrics file exists.** At 1024 envs `demo_metrics.json` is first
   written around iteration 4, so fifteen minutes is far more than enough.
   If it is missing, every point below is unreadable and the probe failed.
3. **P3 — "existing instead of solving".** Read Episode Reward against
   `mean_max_depth_mm`. **The hack signature is reward rising while depth
   stays flat.** Reward rising WITH depth is the healthy case.
4. **P4 — engaged-parking.** `mean_max_depth_mm` settling near **10.80 mm**
   and stopping there, with `success_rate_recent` at 0, is the signature. Any
   plateau elsewhere is not this hack.
5. **P5 — pressing instead of searching (D-037).** `joint_target_lag_rad`,
   `max` and `failure_p95` DIVIDED BY 0.02: the quotient is how many control
   steps the policy spends unwinding before the arm moves at all. This is also
   the number that answers what the user saw in the viewport — jitter and
   drift up and toward the base.
6. **P6 — force farming.** `force_norm_n` (p95, p99, max) and
   `force_abort_rate`. Read against `f_max_n`, which is the INVENTED 50 N
   placeholder — a high abort rate here says as much about the placeholder as
   about the policy.
7. **P7 — the D-153 exposure.** `stage1_lateral_y_mm.over_reach` is expected
   to be **0**. Non-zero reopens D-153 with a number.

## What NO result here decides

* **Not a learning judgement.** Fifteen minutes at rung 0 cannot say whether
  the task is learnable. A flat reward is not a failure and a rising reward is
  not a success.
* **Not a hyperparameter signal.** CLAUDE.md's order stands: identity tests,
  teleport test, THEN tuning. This probe changes nothing.
* **Not proof that a hack is absent.** The clamp exploit (D-152) has no
  instrument at all. This probe can only catch the hacks whose signature
  reaches `demo_metrics.json`; it cannot see one that does not.

## If a hack shows

Report it, do not fix it in the same breath. One hypothesis per change
(CLAUDE.md), and a reward change before RT-108 makes the two runs
incomparable — so the finding is written down first and the fix is its own
step.
