# RT-141 — expectation (contact scale 0.02 AND the new distance leash)

Written 2026-09-03 on the dev laptop, after the RT-139 and RT-140 verdicts
(`rt_logs/VERDICTS.md`, 2026-09-03). `/rt-check` judges the log against
THIS file and nothing else.

## THE READING RULE, first, because both earlier readings got it wrong

Every row of `reward_terms_recent_mean` is an EPISODE SUM. Two runs whose
episodes have different lengths cannot be compared row against row. Divide
every row by the episode length, and take that length from the `time` row:

    steps_per_episode = time / -0.00390625        (-1/T at T = 256)

Applied to the two runs this expectation is read against:

| | RT-138 | RT-139 | RT-140 |
|---|---|---|---|
| `time` | −0.049205 | −0.051648 | −0.988912 |
| ⇒ steps/episode | 12.60 | 13.22 | **253.16** |
| `action_rate` sum | −1.7143 | −1.8086 | −9.9578 |
| ⇒ per step | 0.1361 | 0.1368 | **0.0393** |
| `contact` sum | — | −437.58 | −31.04 |
| ⇒ per step | — | −33.10 | −0.1226 |
| `kernels` sum | 3.2896 (RT-139) | 3.2896 | 4.3637 |
| ⇒ per step | — | 0.2488 | **0.01724** |

So RT-140 was **3.5x smoother per step**, not jerkier: its −9.96 is large
only because its episodes ran 19x longer. Any point below that quotes a
per-step number quotes it this way.

## What the run is

**RT-140 repeated with TWO changes.**

1. `env.contact_penalty_scale=0.02` — the decade sensitivity row named in
   `docs/decisions_inbox.md`, entry "The contact force is priced per step
   ..." (2026-09-03). Hydra override only, no code.
2. **NEW CODE: a dense distance leash**, reward row `far`:
   `-far_penalty_scale * max(0, sdf_dist - far_penalty_leash_m)`, every
   step, not SAPU-scaled, not part of `step_task`. Leash 0.17321 m
   (MEASURED = `KERNEL_MARGIN_COARSE`, the SDF distance at the home pose),
   scale 5.7734 per metre (DERIVED, `1/KERNEL_MARGIN_COARSE`, in
   `RL_PLACEHOLDERS`). Home: `docs/decisions_inbox.md`, entry "The reward
   pays for the flight ..." (2026-09-03).

Everything else as RT-140: fresh run, Uniform[−30, +30] mm start, D-165
progress term, per-term log, seed 42, 1024 envs, 1500 iterations
(`agents/rsl_rl_ppo_cfg.py` `max_iterations`), 5 mm fixture noise.
RT-138 (no contact penalty) and RT-140 (contact penalty at 0.2, no leash)
are the two A/Bs.

**DELIBERATE DEVIATION from "one hypothesis per change"** (user decision
2026-09-03, "beides gemeinsam passt"). It stays attributable because
`contact` and `far` are SEPARATE `REWARD_TERMS` rows: the log says which
term paid what, per step. Any verdict that cannot name which row moved is
not a verdict.

## Why the leash exists, with the number the run is read against

RT-140's `kernels` 0.01724/step inverts through the coarse kernel
`1/(2*cosh(17.2809*d)+2)` to **d = 233 mm** — further out than the home
pose (173.21 mm). It could sit there for free: the kernel's gradient is
−0.00029 per millimetre at 233 mm and −0.00103 at the rung-0 start, while
one contact step at RT-139's mean excess (165.5 N) cost 33.1 at scale 0.2.
One touch was worth 32 metres of flight.

Per step after the change (computed offline from the committed constants;
this is the only home for these four numbers):

| behaviour | kernels | far | net per step | over 253 steps |
|---|---|---|---|---|
| flee to 233 mm | +0.0172 | −0.3452 | **−0.3280** | −83 |
| stand at the rung-0 start (30 mm) | +0.2339 | 0.0 | **+0.2339** | +59 |
| sit at the home pose (173.21 mm) | +0.0455 | 0.0 | **+0.0455** | +11.5 |

And the contact side at 0.02: a contact step at 165 N excess costs −3.31
instead of −33.10, i.e. it wipes out 14 steps of standing still instead of
141. A gentle insertion (~100 N over ~10 contact steps) costs ~−17 against
a success lump of ~+560.

Ordering after both changes: gentle success > standstill > flight. Before
them the flight was the cheapest thing on the board.

**Stated openly: the same ordering argument already held at scale 0.2 and
RT-140 still fled.** The ordering of final policies is not what these runs
turn on; the exploration path is. Only the run answers it.

## The RT-140 baseline (iteration 1499, 2000-episode window)

success recent 0.0 / cumulative 0.00037; `mean_max_depth_mm` 7.535;
`force_abort_rate` 0.0075; `force_norm_n` p50 25.4 / p95 195.9 / p99 280.2
/ max 750.7, `over_f_max` 15/2000; `success_by_start_height_bin` 0 % in all
six; `stage1_lateral_y_mm` p50 0.756 / p95 7.18 / max 72.44 /
`over_reach` 74; `joint_target_lag_rad` max 0.4067 / failure_p95 0.0907;
noise std 0.08; entropy −6.97; `reward_terms_recent_mean.total` −37.43.

The RT-138 baseline, for the PASS condition: success recent 0.3735;
`force_norm_n` p95 338.6; `force_abort_rate` 0.641; top start-height bin
22 %; value loss 2261.6.

## Points, each PASS/FAIL on its own

1. **P1 — the code is this code.** `[rt_log] git:` must be **d480c72 or
   a descendant of it** — d480c72 is the commit that introduced the
   `far` term, and every commit after it on this branch is
   documentation only, so it does not move the env code. The exact head
   is named in the handover beside the commands; a hash that is NOT a
   descendant of d480c72 means the training PC did not pull. The startup block must print, on their own lines:
   `contact penalty (FORGE form, 2026-09-03): -0.02/N above 15.0 N ... ON`
   AND
   `distance leash (2026-09-03): -5.7734/m beyond 0.17321 m SDF distance
   ... ON`.
   The `RL_PLACEHOLDERS` block must list `far_penalty_scale` beside
   `contact_penalty_threshold_n`, `force_abort_f_max_n`, `abort_payment`
   and `engaged_depth_m`. Marker `insertion-rung0-start-2026-08-31-p1`;
   `SAMPLED per episode ... uniform in [-30.000, +30.000] mm`.
2. **P2 — both new terms are live.** `Episode_Reward/contact` AND
   `Episode_Reward/far` exist on the curve, and
   `reward_terms_recent_mean` carries both keys. `far` must be NEGATIVE at
   least in the early iterations (a fresh policy leaves the leash before it
   learns not to); a flat 0.0 for the whole run means either the policy
   never left the leash — which P4 will confirm or deny through the depth
   and success rows — or the term is fed a constant. Distinguish the two
   with `kernels` per step: below 0.0455 the part IS outside the home-pose
   distance, so a `far` of 0 there is a WIRING failure and the run is void.
3. **P3 — the two failure sides. Each is a FAIL on its own.**
   * **Side A, it presses again (RT-138 returns):** `force_norm_n` p50
     ≥ 250 N AND `force_abort_rate` ≥ 0.5. Verdict: the contact scale is
     not the lever, and the FORM changes next — F_max, or observing the
     threshold as FORGE does. No intermediate scale values. NOT simply a
     higher threshold: see "What this run cannot answer".
   * **Side B, it stands still or flees again (RT-140 returns):**
     `success_rate_recent` 0 in ALL six `success_by_start_height_bin` rows
     AND `mean_max_depth_mm` at the 7.5 mm start floor AND `kernels` per
     step below 0.234. Then read the `far` row PER STEP to say WHICH
     failure it is:
     - `far` per step clearly negative → the leash fired and lost. The
       term is not the lever, and the next change is a FORM change on the
       contact side, not another scale.
     - `far` per step ≈ 0 with `kernels` per step between 0.0455 and
       0.234 → the policy parked just INSIDE the leash. Then the LEASH
       DISTANCE is the wrong number, not the term, and the next change is
       the leash, not the scale.
4. **P4 — THE CLAIM: force comes down without success going away.**
   `force_norm_n` p95 below 300 N (RT-138: 338.6) AND
   `success_rate_recent` not below 0.37 (RT-138). Both, or the point is
   VERLETZT with the two numbers quoted. p50 / p99 / max / `over_f_max` /
   `force_abort_rate` read beside them.
5. **P5 — the SBC table.** `success_by_start_height_bin`, all six rows.
   Deep bin against RT-139's 92 %, top bin (20..30 mm) against RT-138's
   22 %. The top bin is the thesis number.
6. **P6 — D-157 tripwire.** `success_depth_invariant_violations` 0 / 0.
7. **P7 — the per-term account.** `reward_terms_recent_mean.total` vs
   rsl_rl's `Mean reward`, both quoted, and every row divided by the
   episode length before it is compared to RT-139 or RT-140. Sanity: `far`
   per step ≈ −5.7734 × (mean SDF distance − 0.17321) over the steps spent
   outside; `contact` per step ≈ −0.02 × mean excess newtons. The RT-138
   P7 gap (216 vs 283) is still unexplained and is reported, not scored.
8. **P8 — the six curves, in order, with trend:** Episode Reward → Policy
   Loss → Value Loss → Entropy → Explained Variance → KL, plus
   `Policy/mean_noise_std`. Value loss against RT-138's 2261 (D-116 —
   rsl_rl has no value normalisation, and the leash adds a term of order
   −0.3/step, so a rise is expected and its SIZE is the reading). Entropy
   against RT-140's −6.97 and noise std against its 0.08: entropy falling
   toward 0 with noise std well under 0.21 is exploration dying, a red flag
   on its own.
9. **P9 — the arm.** `joint_target_lag_rad` max / success_p95 /
   failure_p95 against RT-139's 0.990 / 0.563 / 0.320. NOTE, so the number
   is not misread: RT-139's lag is LARGER on the successes than on the
   failures, so the wind-up is how the policy inserts, not why it presses —
   an anti-wind-up clamp is NOT indicated. `stage1_lateral_y_mm` p50 / p95
   / max / `over_reach` against 0.756 / 7.18 / 72.44 / 74, read with its
   caveat: the metric is a PER-EPISODE MAX and `in_stage1` is a pure
   z-window with NO lateral gate, so a large max is one episode carrying
   the part off the fixture footprint at mouth height, not a wall-to-wall
   bounce. Readings, not gates.

## The commands (training PC, repo root, conda env active)

The env code CHANGED since RT-140, so the smoke and the guard run first.
`zero_agent.py` cannot show either penalty line (it is a measurement env,
`rl_terms_enabled` False), so both short runs use `train.py
--max_iterations 1`, which builds the env — where the guard runs — and
fails at construction, before any real training compute.

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v1
git pull
git rev-parse --short HEAD
.\scripts\rt_log.ps1 RT-141a python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --max_iterations 1
.\scripts\rt_log.ps1 RT-141b python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --max_iterations 1 env.far_penalty_scale=-1.0
.\scripts\rt_log.ps1 RT-141 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless env.start_tip_above_entrance_low=-0.030 env.contact_penalty_scale=0.02
```

* **RT-141a, the smoke:** must print BOTH penalty lines (P1) and exit 0.
* **RT-141b, the guard:** a negative leash scale must be REFUSED — non-zero
  exit, message naming `far_penalty_scale`. Exit 0 here means the guard is
  not wired: STOP, do not start the long run. The value must be written
  `-1.0`, not `-1`: hydra type-checks the override against the `float`
  annotation and an int dies before the guard runs (RT-139b's failure).
* **RT-141, the run:** 1500 iterations, the cfg default.

Bring back: the `[rt_log]` header, the startup block (placeholder lines,
the contact line, the leash line), the first iteration block, the last
iteration block, `demo_metrics.json`.

## What this run cannot answer

* **The contact threshold, and the contact THRESHOLD is NOT the obvious next
  change any more. `docs/reference/literature_check_contact_force_sliding_search_2026-09-03.md`
  (2026-09-03) puts the published search-phase band at 1-20 N -- FORGE
  trains at [5, 10] N, Inoue at 20 N, Lee & Pham at 1 N and halt at 50 N --
  so 15 N sits INSIDE the literature and 300 N sits more than an order of
  magnitude ABOVE all of it. RT-129's 50-170 N is the SCRIPTED CONTROLLER's
  force at its own gains, which `InBachelorErwaehnen.md` (2026-09-01)
  already marks as not the task's. The two numbers are therefore not the
  same kind of number, and the conflict is NAMED, not resolved: raising the
  threshold is no longer supported, and lowering F_max needs a decision on
  which of the two is the reference.
* **Whether the gentle path is FOUND or only priced.** Still no per-step
  trace of the part or the tip; every position instrument in the env is a
  per-episode maximum.
* **The leash DISTANCE.** One value, derived from the home pose. P3 side B
  says how to tell a wrong distance from a wrong term.
* **The FORGE mitigation we skip:** the threshold randomised per env and
  put into the observation. D-114 named it and we do not do it.
* **Whether all 28 observation dims earn their place.** Separate task;
  `pocket_quat` (4 dims) is constant by config today and `yaw_cos_sin`
  (2 dims) is a function of two other channels — neither needs a run.
* The RT-138 P7 gap and the undiscounted success lump (D-164's named
  consequence).
