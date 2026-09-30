# RT-191 — expectation, written BEFORE the run (2026-09-13)

Written on the dev laptop by the main chat. `/rt-check` judges the log
against THIS file. **UNVERIFIED** until the training PC has run it.

## What this is

Option B1 of the parking diagnosis (chat, 2026-09-13). RT-XXX (verdict
line in `rt_logs/VERDICTS.md`, 2026-09-13 11:54:35) showed: with the three
observation-noise fields at 0 the policy finds the pocket by iteration ~200
and then hovers ENGAGED at 13 mm, success 0 — the pattern `Laufplan_Phase5.md`
6b names. 6b's own rule: start at differential-diagnosis point 4, the reset
distribution, NOT the reward.

**One change against RT-XXX:** the start band reaches into the pocket,
`env.start_tip_above_entrance_low=-0.030` (the cfg's own hydra example,
`insertion_env_cfg.py:424`). That needs `dr_mode=off` (a band is refused
under autodr), so this is a No-DR run. Noise stays 0. Seed 20 (probe seed,
`Laufplan_Phase5.md`). 256 envs. 300 iterations.

This CHANGES THE START DISTRIBUTION. A start inside the pocket is a
curriculum, not the Phase-5 target distribution (D-179's start height is
one-sided from H_min upward). The question is only whether the seat is
reachable for this reward once episodes begin inside.

## The question

Does the hover at 13 mm break when some episodes start inside the pocket?

## Command (training PC, PowerShell, conda env `env_isaaclab`)

Root: `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`

```
git pull --ff-only origin p5-robustheit
```

```
.\scripts\rt_log.ps1 RT-191 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 20 --max_iterations 300 env.start_tip_above_entrance_low=-0.030 env.obs_noise_pocket_pos_std_m=0.0 env.force_obs_noise_std_n=0.0 env.grasp_obs_offset_x_m=0.0
```

After the run (the shutdown may hang after `Training time`; close the
window then and run the shortener by hand):

```
python scripts/shorten_rt_log.py rt_logs\RT-191.txt --out rt_logs\RT-191.short.txt
```

## PASS / FAIL lines

**P1 — code stand.** `[rt_log] git:` is the SHA handed over in chat, and
`git diff --quiet cf7ab14 <SHA> -- source/` on the laptop shows only the
`max_depth_max_mm` log key (commit `f1983815`).

**P2 — it starts as intended.** No `ValueError`, no Traceback. The startup
report reads `obs_noise_pocket_pos_std_m = 0.0`, `grasp_obs_offset_x_m =
0.0`, `dr_mode: off`. The run folder tag carries `start-30..+30mm` and
`seed20`. `start_height_mean_mm` on the curve sits near 0, not at 30.

**P3 — the new key prints.** The line `Mean episode max_depth_max_mm`
appears in every iteration block, as the LAST `Mean episode` line.

**P4 — the discriminating point, read at iteration 150 and 300:**
`success_rate`, `mean_max_depth_mm`, `max_depth_max_mm`,
`Episode_Reward/engaged`, `Episode_Reward/success_lump`.

- **Outcome 1, the hover breaks:** `success_rate` > 0.5 by 300 and
  `max_depth_max_mm` >= 33. Then the seat is reachable for this reward with
  an in-pocket curriculum. Phase 5 has no such curriculum, so B2 (a reward
  that breaks the hover on its own) becomes the design question.
- **Outcome 2, the hover stays:** `success_rate` < 0.1 at 300 while
  `mean_max_depth_mm` sits at 10–15 and `engaged` above half the return.
  Then the curriculum does not carry, and B2 moves up to the next step.
- **Outcome 3, in between:** report the curve, conclude nothing; a longer
  cap is the next question, not a change.

Note on reading: episodes that START below 0 mm seed `_max_depth` with
their start depth (`insertion_env.py:2786`), so `mean_max_depth_mm` rises
at once by construction. Only `success_rate` and `max_depth_max_mm` >= 33
show a SEAT.

**P5 — six metrics with trend.** Episode Reward → Policy Loss → Value Loss
→ Entropy → Explained Variance → KL. EV and KL not logged (D-116).

**P6 — red flags, self-reported.** `mean_noise_std` (RT-XXX ended at 0.38),
`force_abort_rate`, `interpen_max_max_mm`, NaN.

## What this run cannot answer

- Anything about noise: it runs at 0.
- Anything about Phase 5's own start distribution.
- Whether B2 is needed: only whether the reward can seat at all on this code.

## Status

**UNVERIFIED.**
