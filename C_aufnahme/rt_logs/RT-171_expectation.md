# RT-171 — expectation, written BEFORE the run (2026-09-06)

Written on the dev laptop. `/rt-check` judges the log against THIS file and
nothing else. RT-171 is the alignment-term run on `p5-reward` in
`Phase3_Implementierung_v3`. Its CONTROL is RT-172 (`RT-172_expectation.md`):
same checkpoint, same task, same three overrides, ONE reward row more. RT-172
must be finished and judged before this run starts -- both because the two
are one comparison and because only one editable `insertion` install can
exist (`HANDOFF-RL.md` § training PC), and RT-172 runs off `_v2`.

## The one hypothesis

**"A potential-based alignment row makes the RT-156 policy learn a TILTED
pocket where the reward as it stands (RT-172) does not."** RT-170 BEFUND 2
measured the kernel nearly blind to tilt (1.02e-5 per degree against
2.35e-4 per mm); RT-158 measured the failing part TILTED (mouth mode
15.34 deg, rim mode 5.42 deg). The row pays for righting and charges for
tilting, and nothing else -- discounted over an episode it telescopes to
`gamma^T Phi(s_T) - Phi(s_0)` (Ng, Harada & Russell 1999), so it cannot
change which policy is optimal; it changes the GRADIENT the policy sees
while tilted.

Falsifier, stated before the run: if the 3-6 deg bin of this run is not
better than RT-172's, the missing tilt gradient was not the blocker, and the
diagnosis goes back to point 3 of the differential diagnosis (action and
controller -- the stiffness UNDER contact is still unmeasured, RT-168
GRENZE). The weight decade {0.1, 10.0} is NOT in this run (§ below).

## The one change

`w_tilt` 0.0 -> 1.0, i.e. the cfg default of `p5-reward`
(`insertion_env_cfg.py`, `w_tilt`), with `kernel_a_tilt` = arccosh(10) /
15.34 deg = 11.1799 /rad (`insertion_tasks_cfg.KERNEL_A_TILT`, margin =
RT-158 mouth-mode median, user's choice 2026-09-06 over the CAD cone) and
`shaping_gamma` = 0.99 read from `agents/rsl_rl_ppo_cfg.py`. The row is
`Episode_Reward/tilt_shaping` on the curve and `tilt_shaping` in
`reward_terms_recent_mean`.

Everything else EXACTLY as RT-172: the RT-156 checkpoint `model_250.pt`
resumed, `env.fixture_tilt_noise_rad=0.0873` (5 deg), start band +30..+50
mm, `osc_pos_clamp_m` 0.07, `contact_penalty_scale` 0.0 and
`far_penalty_scale` 0.0 (the two RT-156 overrides, NOT cfg defaults), osc,
kp 100, kp_rot 30, 15 Hz, 256 steps, 1024 envs, seed 42, force abort 50 N
as `truncated`, `--max_iterations 2000`. NAMED: the resume starts a critic
that has never seen the row; Ng's theorem says the optimal value shifts by
exactly `+Phi(s)`, so the critic has to learn one more smooth function of
the observation. Expect a value-loss transient at the start (P9).

Laptop state before the run (all UNVERIFIED under Isaac): `py_compile` OK on
every edited file; `scripts/check_insertion_math.py` 243 checks green (214
before), its `--self-test` counter-proof green with three new mutations
(`tilt-sign-flipped`, `tilt-potential-not-squashed`,
`tilt-terminal-not-zeroed`) and two cfg mutations; `check_env_wiring.py`
156 green; `check_seated_success.py --self-test` 76/76;
`selftest_checks.py --offline` PASS (all seven counter-proofs green, 86 mutations in the math one).

## Commands (training PC, PowerShell, repo root, conda env active)

The `_v3` clone must exist (`HANDOFF-RL.md` § training PC). Then, ONLY after
RT-172 is finished and judged:

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v3
git pull
git rev-parse --short HEAD
```

Expected HEAD: the SHA in the handover message (read back from
`git ls-remote origin p5-reward` on the laptop AFTER the push).

The checkpoint lives in `_v2`'s log tree and rsl_rl looks for it under the
CURRENT repo root (`logs/rsl_rl/ur5e_insertion/`), so copy the run folder --
copy, never move, `_v2` stays runnable:

```
New-Item -ItemType Directory -Force logs\rsl_rl\ur5e_insertion
Copy-Item -Recurse "..\Phase3_Implementierung_v2\logs\rsl_rl\ur5e_insertion\09-05_19-45-03_offset5mm_currOFF_start-30..+30mm_seed42" "logs\rsl_rl\ur5e_insertion\"
Test-Path "logs\rsl_rl\ur5e_insertion\09-05_19-45-03_offset5mm_currOFF_start-30..+30mm_seed42\model_250.pt"
```

Must read `True`. Then move the editable install to `_v3` (Isaac Lab
template form, UNVERIFIED on this machine; `pip show insertion` is the
truth):

```
python -m pip install -e source\insertion
pip show insertion
```

`Editable project location` must end in `Phase3_Implementierung_v3`. Paste
that line into the handover. Then the run:

```
.\scripts\rt_log.ps1 RT-171 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max_iterations 2000 --resume --load_run 09-05_19-45-03 --checkpoint model_250.pt env.fixture_tilt_noise_rad=0.0873 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.osc_pos_clamp_m=0.07 env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0
```

No `env.w_tilt` override: 1.0 is the branch default and the startup report
prints it (P2). `--load_run` takes the TIMESTAMP only; `train.py
--checkpoint` the FILE NAME. Hydra floats need the decimal point.

Paste `rt_logs/RT-171.short.txt` and the run's `demo_metrics.json` into
`rt_logs/inbox.txt` on the laptop. Run folder tag expected:
`..._offset5mm_tilt0-5deg_currOFF_start+30..+50mm_seed42` (the tag does not
carry `w_tilt`; P2 reads the startup report instead).

## PASS / FAIL lines, in order

1. **P1 — the reset is contact-free.** As RT-172 P1: `force_abort_rate` in
   the first block is NOT 1.0 and the step-2 force is of the reset
   transient's order (RT-149: 2.021 N), not tens of N. If P1 fails, nothing
   below is read.
2. **P2 — the run is what it says.** `pip show insertion` location ends in
   `_v3`; run folder tag contains `tilt0-5deg` AND `start+30..+50mm`;
   startup report prints the line `alignment shaping (RT-171,
   potential-based): 0.99 * Phi(s') - Phi(s), Phi = 1.0 * sech(11.1799/rad
   * theta) vs the POCKET axis (margin 15.34 deg, ...) ... ON`; the
   `Loading model checkpoint from:` line ends in
   `09-05_19-45-03_offset5mm_currOFF_start-30..+30mm_seed42\model_250.pt`;
   `reward_terms_recent_mean` in `demo_metrics.json` has a `tilt_shaping`
   key.
3. **P3 — the account adds up.** `contact` and `far` EXACTLY 0.0; `abort` =
   -1.0 x `force_abort_rate`; `success_lump` > 0 iff `success_rate` > 0.
4. **P4 — THE IDENTITY of the row, read on the LOGGED (undiscounted)
   per-episode sum.** Summed plainly, `gamma * Phi(s_t+1) - Phi(s_t)` over
   an episode of L steps is `-(1 - gamma) * sum_{1..L-1} Phi_t + gamma *
   Phi_L - Phi_0`, and on the success step `Phi_L` is zeroed. With `w_tilt`
   1.0 and Phi in [0, 1] that is bounded in **[-3.56, +0.99]** for every
   episode (L <= 256), and it is NEGATIVE for every episode that does not
   end more upright than it began. PREDICTION, per outcome: a success at
   step L, aligned throughout, reads about `-0.01 * L - 1.0` (L = 60:
   -1.6); a timeout held aligned reads about -3.55; a timeout ending
   tilted at the margin reads about `-(0.01 * 255 * Phi_mean) - Phi_0 +
   0.1`. So the WINDOW MEAN of `tilt_shaping` must lie in **[-3.6, 0.0]**.
   A POSITIVE window mean, or a mean below -3.6, is not a reward finding --
   it is the first-step phantom (`Phi(s_0)` read from a stale reset
   observation, `HANDOFF-RL.md` § Open 2d (iii)) or a sign fault, and P5
   is NOT read until it is explained. The tilted-pocket start makes
   `Phi_0` itself vary: the part starts world-vertical while the pocket is
   tilted 0..5 deg, so `Phi_0` in [0.61, 1.0] -- the bound above already
   covers that.
5. **P5 — THE BRANCH, read at the LAST block against RT-172's LAST block,
   `success_by_tilt_bin` (trailing window), same bins:**
   * this run's 3-6 deg bin exceeds RT-172's 3-6 deg bin by >= 0.10 while
     the 0-3 deg bin is not lower by more than 0.05 -> the missing tilt
     gradient WAS the blocker; the row does what it is for. Next: the
     weight decade (RT-171b `env.w_tilt=0.1`, predicted FAIL arm below the
     0.3516 bound; RT-171c `env.w_tilt=10.0`).
   * within +-0.05 of RT-172 in both bins -> no measurable effect at
     w = 1.0. Not yet a falsification: run the decade first; if 10.0 does
     not move it either, the falsifier above fires.
   * this run's 3-6 deg bin BELOW RT-172's by >= 0.10 -> the row hurts.
     Named candidate: it charges the D-163 edge-first approach (the CAD
     cone allows 8.52 deg of deliberate lean, which the row prices at
     `1 - Phi(8.52 deg)` = 0.63 w until righted). Read
     `reward_terms_recent_mean.tilt_shaping` beside the success rate over
     the blocks before concluding.
   * RT-172's own P5 read "BOTH bins under 0.50" (forgetting / broken
     resume) -> this run has no control and is read as P2/P4 only.
6. **P6 — force is READ, not predicted.** `force_norm_n` p95 and
   `force_abort_rate` over the last window, beside RT-172's and RT-156's
   (p95 27.27 N).
7. **P7 — sigma survives.** `Mean action noise std` at the last block >
   0.30; under 0.05 = the RT-148b collapse, then P5 is read as unexplored.
8. **P8 — the solver still lands.** `start_pose_solve_unconverged_resets`
   0, `start_pose_solve_worst_residual_mm` under 0.05 mm.
9. **P9 — the critic absorbs the row.** `Value function loss` in the first
   1-3 blocks is ABOVE RT-172's at the same iterations (the resumed critic
   has never seen `+Phi(s)`), and by iteration 200 it is back within 2x of
   RT-172's. `Explained variance` must not sit at 0 after iteration 200
   (D-116, value drift). A critic that never recovers makes P5 a
   value-function finding, not a reward finding.
10. **P10 — no hacking.** `Episode Reward` rising while `success_rate` is
    flat AND `tilt_shaping` is the term that rises -> the row is being
    farmed, which the telescoping forbids unless the phantom of P4 is
    present. Report the per-term rows over the blocks; do not conclude.

## What this run does NOT answer

* The weight: one arm of the decade. RT-171b/c (`env.w_tilt=0.1` / `10.0`)
  are the sensitivity rows of D-109 (9)(iii); 0.1 is the predicted FAIL arm.
* The margin: 15.34 deg is one run's median (RT-158, 1024 envs, one
  checkpoint). The CAD-cone alternative (8.52 deg) is not run.
* Yaw: `fixture_yaw_noise_rad` stays 0.0; the row is yaw-invariant by
  construction (checked offline) and RT-152 stays a thesis-must-state
  limitation.
* Training from scratch under the new reward: this is a resume, chosen to
  pair with RT-172. A from-scratch run is a different question.
* The `shaping_gamma` second home: the env reads gamma from the agent cfg at
  import; a hydra override of `agent.algorithm.gamma` would not reach it.
  Not exercised here (no such override).
