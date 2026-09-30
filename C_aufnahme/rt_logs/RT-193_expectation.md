# RT-193 — expectation, written BEFORE the run (2026-09-13)

Branch `p5-kraftsensor`. **UNVERIFIED.** The `wrench`-mode twin of RT-191:
same command, same seed, same noise-off start band, ONE change — the
observation mode. Requested by the user through the p5-robustheit diagnosis
chat (2026-09-13). RT-191 and RT-192 are taken; RT-193 is the next free
number (p5-robustheit `VERDICTS.md` read from origin, 2026-09-13).

## The one question

**"With the three torque channels in the observation, does the same run reach
the same success as RT-191 — faster, slower, or not at all?"**

## Baseline RT-191 (p5-robustheit, verdict commit `21946aea`, run git `afddfb6`)

```
python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 20 --max_iterations 300 env.start_tip_above_entrance_low=-0.030 env.obs_noise_pocket_pos_std_m=0.0 env.force_obs_noise_std_n=0.0 env.grasp_obs_offset_x_m=0.0
```

| RT-191 | value |
|---|---|
| success_rate, iteration 11 | 0.5739 |
| success_rate, iteration 299 | 0.9903 |
| max_depth_max_mm | 36.0 |
| std | 0.82 |
| iteration time | 6.0 s -> 10.0 s |

(numbers as reported by the diagnosis chat; the verdict line on
p5-robustheit is the home.)

## What is NOT identical to RT-191, named

1. **The mode** — the one intended change: `env.obs_wrench_mode=wrench`,
   31 channels, torque sigma 0 (cfg default, D-188).
2. **Row 0 of the force channel.** This branch's commit `07dbff4a` makes the
   wrench EMA skip the reset step (inbox entry 2026-09-13). RT-191 ran WITHOUT
   it: its row 0 carried 0.25 x the old episode's wrench. This is a second
   difference and it is in BOTH channels' first row. Not removable without
   reverting a correctness fix; named here so a difference in the first
   iterations is not read as the torque's doing alone.
3. **Code base.** RT-191 ran at `afddfb6` (p5-robustheit); this branch forks
   at `d9205bf0` and cherry-picks `f1983815` (the `max_depth_max_mm` log
   key, log only), so the SAME keys are printed. Nothing else of
   p5-robustheit after `d9205bf0` is code.
4. The normaliser sees 31 inputs instead of 28; the actor's first layer is
   128 x 31. Same seed does not mean the same weights.

## Command (training PC, PowerShell, clone `Phase5_Robustheit_v1`)

Only after RT-192 (the implementation check) — or, if the user wants the
comparison first, after RT-191 has finished and nothing else runs.

```
git fetch origin
git checkout p5-kraftsensor
git pull
git rev-parse HEAD
```

Expected HEAD: the SHA in the chat handover (the commit carrying this file).

```
.\scripts\rt_log.ps1 RT-193 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 20 --max_iterations 300 env.start_tip_above_entrance_low=-0.030 env.obs_noise_pocket_pos_std_m=0.0 env.force_obs_noise_std_n=0.0 env.grasp_obs_offset_x_m=0.0 env.obs_wrench_mode=wrench
```

Hand over: the short log, `demo_metrics.json` of the run, and
`export_tb_scalars.py` CSV if it runs (`docs/figures/RT-193_scalars.csv`).

## Points

* **P1 setup.** exit 0, `git:` = handed-over SHA, startup report
  `obs mode wrench-31`, `observation_space` 31, torque sigma 0.0, pocket
  sigma 0.0, force sigma 0.0, grasp 0.0, start band low -0.030. Otherwise
  RUN INVALID.
* **P2 the six curves, in the fixed order** (Episode Reward, Policy Loss,
  Value Loss, Entropy, Explained Variance, KL), each with its trend, before
  any judgement; all finite.
* **P3 success.** `success_rate` at iteration 299 and the FIRST iteration at
  which it passes 0.90, both beside RT-191's 0.9903 and its own first-0.90
  iteration (read from the RT-191 log, not from prose).
* **P4 depth.** `max_depth_max_mm` and its std beside 36.0 / 0.82.
* **P5 cost.** iteration time beside 6.0 -> 10.0 s (31 inputs, same envs).
* **P6 the torque itself.** `torque_norm_nm` (p50/p95/max, N m) and
  `wrench_valid_frac` from the log: the channel the policy was given, as a
  magnitude — the diagnosis put the parking-phase moment at ~0.03 N m.

## Pre-registered readings (no bar is invented)

| Outcome | Reading | Next |
|---|---|---|
| SAME | success at 299 within one seed's spread of 0.9903 (spread unknown: one seed each — say so) | The torque neither helps nor hurts at noise 0. The question moves to noise ON. |
| FASTER | first-0.90 iteration clearly earlier than RT-191's | Repeat with a second seed before any claim. |
| WORSE | success at 299 clearly below, or no rise | Read P6 first: a torque channel with |tau| ~ 0 is an input the normaliser inflates. Then the differences 2-4 above. |

One seed against one seed decides nothing on its own. The seed spread of
this setup is not measured. Whatever falls out, the next step is a second
seed, not a conclusion.
