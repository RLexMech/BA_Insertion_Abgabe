# RT-131 — expectation, written BEFORE the run

Written 2026-09-01 on the dev laptop. CLAUDE.md § Hyperparameter requires
the expectation to exist before the log does; `/rt-check` judges the log
against THIS file and nothing else. **Runs only after RT-130 PASSED.**

## What the run is

The first PPO run under the REVISED reward (inbox entry
"Reward-Ueberarbeitung", 2026-09-01: coarse margin 173.21 mm, time penalty
−1/256 per step, action-rate penalty 0.1, success TERMINATES with payout)
and the first from the D-163 start pose (+30 mm above the stage-2 opening
plane, in free air). Rung 0, ladder off. Seed 42 (rsl_rl runner default,
`RslRlOnPolicyRunnerCfg.seed`).

Two user decisions of 2026-09-01 are in this run and are READ, not judged:

* `force_abort_f_max_n` = **300 N**, an observation window, still a
  placeholder (inbox entry "The tilted scripted insertion is SPENT ...").
  D-158's concern — the policy pressing against the limit as a ceiling
  (RT-107, p95 87.6 N under 100 N) — is accepted knowingly.
* D-163's open reward question — one SDF distance pays a deliberate tilt
  LESS — is accepted, not solved ("vielleicht lernt er ja trotzdem").

Unchanged and to stay unchanged: 1024 envs, `curriculum_enabled = False`,
`rung_step_sizes` unset, `fixture_pos_noise_xy = 0.005`, the reset noises
as in RT-108/RT-120, and the agent file's own defaults
(`num_steps_per_env = 16`, `max_iterations = 1500`,
`learning_rate = 1.0e-3`, `num_mini_batches = 4`). NOTHING is tuned.

## THE QUESTION THIS RUN ANSWERS

**Does the policy learn to insert from 30 mm above the opening under a
reward that has a gradient over the whole travel — and does it find the
tilted entry on its own?** The second half has NO instrument yet: the env
logs the FIXTURE tilt bin per episode, not the PART's attitude (checked
2026-09-01, `insertion_env.py` `_recent_tilt_deg` is the pocket tilt). So
the tilt half can only be read from a replay (`play.py`) after the run,
and that is a separate RT number.

## Arithmetic

Unchanged from RT-120: 16 iterations per episode at 256 steps, 64 episodes
per iteration, 1500 iterations = 96 000 episodes if every episode runs to
the timeout. Success terminations and force aborts SHORTEN episodes, so the
episode count will be HIGHER and is a reading, not a prediction. Trailing
window 2000 episodes. **No wall-clock band is predicted.**

## The commands (training PC, repo root, conda env active)

```
git pull
git rev-parse --short HEAD
```

Expected HEAD: given in the handover message.

```
.\scripts\rt_log.ps1 RT-131 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless
```

After the run, paste `demo_metrics.json` from the run folder into
`rt_logs/inbox.txt` with the log.

## Points, each PASS/FAIL on its own

1. **P1 — it starts and survives.** Exit 0 (or a clean Ctrl+Break), no
   `nan`, no traceback. The exit-code trap applies.
2. **P2 — the code is this code.** The startup report prints
   `insertion-rung0-start-2026-08-31-p1` and the `[rt_log]` header prints
   the git hash from the handover. The startup report prints the force
   limit as **300.0 N** with its `[placeholder]` mark, and
   `demo_metrics.json` carries `force_abort_f_max_n` = 300.0 in
   `RL_PLACEHOLDERS`.
3. **P3 — the run is the configured one.** Startup line
   `[insertion] rung-0 start pose: tool point +30.000 mm above the stage-2
   opening plane`; the run folder carries `start+30mm`, `currOFF`,
   `seed42`; the report prints `reset_joint_noise = 0.0`,
   `reset_yaw_noise = 0.0`, `fixture_pos_noise_xy = 0.005`.
4. **P4 — the start pose lands.** `start_tip_above_entrance_mm = 30.0`,
   `start_pose_solve_unconverged_resets = 0`,
   `start_pose_solve_worst_residual_mm` below `start_pose_solve_tol_mm`
   (0.05); no `START POSE SOLVE DID NOT CONVERGE` line.
5. **P5 — the episodes do NOT die on the first steps.** RT-120's failure
   signature was mean failing-episode length 3.6 steps with
   `force_abort_rate` 1.0, from a start inside the pocket. Here the part
   starts in free air, so the mean episode length in the first metrics
   dump must be well above that class. This is the D-163 check.
6. **P6 — the D-157 tripwire reads zero.**
   `success_depth_invariant_violations` is 0 for `recent` AND `cumulative`.
   **Read this BEFORE any success rate.**
7. **P7 — THE LEARNING QUESTION.** All six metrics, in order, each with
   its trend: Episode Reward → Policy Loss → Value Loss → Entropy →
   Explained Variance → KL. No threshold. Red flags per CLAUDE.md: entropy
   to zero early; grad-norm pinned at the clamp; explained variance to 0
   (D-116, no value normalisation). The success metric is
   `success_rate_recent` and it now counts SUCCESS TERMINATIONS per
   episode, read only after P6 and after the six curves.
8. **P8 — the force reading under the 300 N window.** `force_norm_n`
   p50/p95/p99 and `force_abort_rate`, both READ, no threshold. The one
   named pattern to look for: the distribution parking just under the
   limit as RT-107 did under 100 N. If p95 sits in the 250–300 N class the
   window is being used as a ceiling; if it sits in the class RT-129's
   scripted controller needed (50–170 N) or below, it is not. Either way
   it is the first input to any later `F_max` proposal — the ladder in the
   inbox entry is the source for that proposal, not this file.
9. **P9 — depth against reward.** `mean_max_depth_mm` read beside the
   episode reward. Reward rising while depth stays flat is the RT-107
   hack signature; reward rising WITH depth is the healthy case. Engaged
   parking near 10.8 mm with `success_rate_recent` 0 is the other named
   signature (RT-107 expectation, P4).
10. **P10 — the arm does not run away.** `joint_target_lag_rad` `max` and
    `failure_p95` (divided by `action_scale` 0.02 = control steps of
    unwinding). A reading; D-161's second finding is unfixed on purpose.
11. **P11 — the D-153 exposure.** `stage1_lateral_y_mm` present and not
    `{"episodes": 0}`; `over_reach = 0` expected, and `> 0` reopens D-153
    with a number rather than failing the run.

## Readings taken OUT of the run — no threshold

* R1: the SAPU discard share, if the metrics carry it — how often the
  interpenetration filter zeroed the task return. RT-129 read 0.71 mm at
  ~310 N in two scripted envs, so the window can reach SAPU territory.
* R2: episodes per iteration and the mean successful-episode length — the
  time penalty's first live effect.
* R3: where the aborts happen (depth at abort), if the metrics carry it.
  If they do not, that is the first logging gap to close before any
  `F_max` proposal.

## What this run cannot answer

* Whether the part enters TILTED: no part-attitude log (see above). A
  replay with `play.py` is the instrument, own RT number.
* Nothing about yaw (`reset_yaw_noise = 0.0`).
* Nothing about `rung_step_sizes`; the ladder is off.
* Nothing about the real cell's rear wall (D-156, block off).
