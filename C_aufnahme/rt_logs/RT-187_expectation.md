# RT-187 — expectation, written BEFORE the run (2026-09-12)

Written on the dev laptop, worktree `p5-robustheit`. Instrument commit
`e7b85a2d`. `/rt-check` judges the logs against THIS file and nothing else.

Two arms, `RT-187s20e256` and `RT-187s20e512`: **the same training run at 256
and at 512 environments.** 256 is the reference, decided by the user
2026-09-12 ("256 als erster Trainingsdurchlauf als Referenz, und dann
schauen welches bessere Ergebnisse liefert").

This REPLACES the replay probe in `rt_logs/RT-186_expectation.md`, which is
parked. The solver question (16 vs 192 position iterations) is **not**
answered here and stays open in `docs/decisions_inbox.md`.

## Why this run exists

Two threads meet in it.

1. **The environment count has never been chosen for this setup.** The cfg
   default is 128 (`insertion_env_cfg.py:828`); every training run so far
   passed `--num_envs 1024` (RT-107, RT-181). Neither number was measured
   against an alternative. The published practice for contact-rich
   insertion is 128: the Factory paper trains "each policy using 128
   parallel simulation environments", Isaac Lab's own Factory and Automate
   configs set `num_envs=128`, and IndustReal uses 128.
2. **The documented reason to use fewer is the GPU contact buffer**, not
   solver accuracy — IsaacGymEnvs `docs/factory.md` names "Reduce the number
   of environments" as the remedy when penetration appears. Since `e7b85a2d`
   the interpenetration is readable on its own, so this run reads it for free
   in both arms instead of arguing about it.

## The one hypothesis

**"512 environments reach the stop bar in less wall-clock time than 256, and
neither arm pays for it with more interpenetration."**

Two ways it is falsified, and both are real outcomes, not failures:

* 256 reaches the bar sooner in wall-clock time → the larger batch does not
  pay for its cost per iteration on this card.
* 512 shows materially more interpenetration than 256 → the contact-buffer
  mechanism the Factory docs describe bites between these two counts, and the
  env count is a physics decision, not only a throughput decision.

## What "better" means here — decided BEFORE the run

"Better" is ambiguous and the three readings disagree, so all three are
recorded and the **wall-clock** one decides. Reason: the practical question
is which setting reaches a trained policy sooner on THIS machine.

| reading | how it is computed | favours, a priori |
|---------|--------------------|-------------------|
| **W — wall clock** *(the decider)* | iterations to the bar × mean `Iteration time` of that arm | unknown, that is the point |
| S — sample efficiency | iterations to the bar × `num_envs` × 16 = environment steps | 256, if both learn per-update equally |
| I — per iteration | iterations to the bar | 512, it sees 2× the data per update |

`16` is `num_steps_per_env` (`rsl_rl_ppo_cfg.py:27`).

**The bar** is `Laufplan_Phase5.md:12`, `success_rate >= 0.995`. A second,
softer reading at `>= 0.90` is recorded as well, because a run that never
reaches 0.995 within the cap still gives a comparable number at 0.90 and the
comparison must not collapse if neither arm saturates.

## The one change between the arms

`--num_envs`, and nothing else. Same seed, same code, same overrides, same
`--max_iterations`. `Laufplan_Phase5.md:13` ("all 5 seeds of a step get the
same number of iterations") is the same rule applied to two arms.

Training seed **20**, because `Laufplan_Phase5.md:9` reserves seed 20 for
probe runs and this is a probe: it picks a setting, it is not a result.

**No confound from the metrics window.** `window = max(2000, num_envs *
metrics_window_iterations * 16 / max_episode_length)`
(`insertion_env.py:1055-1056`), with `metrics_window_iterations = 20`
(`insertion_env_cfg.py:514`) and `max_episode_length` 256:

* 256 envs: `256 * 20 * 16 / 256 = 320` → the floor wins, window **2000**.
* 512 envs: `512 * 20 * 16 / 256 = 640` → the floor wins, window **2000**.

Both arms average their success rate and their interpenetration over 2000
episodes. The rolling rate is therefore directly comparable, and a maximum
over the window is not biased by arm.

## `--max_iterations 1500`

The cfg default (`rsl_rl_ppo_cfg.py:28`). It is a technical safety, not a
budget: **an old run never sets a number for this setup**
(`Laufplan_Phase5.md:14`), and no fresh-from-scratch run at 256 or 512 envs
exists. Hand stop is allowed once BOTH arms have passed the bar, and the
verdict then reads the iteration numbers, not the stop.

If an arm does not reach 0.90 within 1500 iterations, say so and judge on the
curve; do not extend one arm and not the other.

## Commands (training PC, PowerShell, conda env `env_isaaclab`)

Root: `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`

### Step 0 — pull and confirm the instrument is present

```
git pull --ff-only origin p5-robustheit
```

```
git merge-base --is-ancestor e7b85a2d HEAD; if ($?) { "instrument present" } else { "INSTRUMENT MISSING -- STOP" }
```

No SHA is written here to compare against: a SHA naming the commit that
carries this file invalidates itself the moment the file is edited (the
RT-184 lesson, `e791ca29`). The ancestry of `e7b85a2d` is the claim that
matters — without it there is no `interpen_max_mm` key.

If it prints anything but `instrument present`, stop.

### Arm 1 — 256 environments (the reference, run this first)

```
.\scripts\rt_log.ps1 RT-187s20e256 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 20 --max_iterations 1500
```

```
Copy-Item logs\demo_metrics.json logs\RT-187e256_demo_metrics.json
```

### Arm 2 — 512 environments

```
.\scripts\rt_log.ps1 RT-187s20e512 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 512 --headless --seed 20 --max_iterations 1500
```

```
Copy-Item logs\demo_metrics.json logs\RT-187e512_demo_metrics.json
```

**The two arms run ONE AFTER THE OTHER, never at the same time.** The
operator measured this on 2026-09-12 with RT-181: a second run alongside
pushed the iteration time from a median of 13.9 s to 37.1 s, about 2.7× for
two runs, so two at once give FEWER iterations per hour than two in sequence
(`docs/decisions_inbox.md`, "Seeds run ONE AT A TIME"). A parallel second arm
would also destroy the wall-clock comparison this run is about.

**No `env.*` override is passed.** The cfg defaults are the current stand:
`dr_mode` "off" (`insertion_env_cfg.py:634`), `curriculum_enabled` False
(`:996`), and the Phase-5 scatter fields at their decided values. This is a
setting probe on the CURRENT configuration, not a replay of an older one.
`env.osc_pos_clamp_m` is never passed (`Laufplan_Phase5.md:19-24`).

### After each arm

Export the scalars so the iteration numbers are read from a file and not
from the console:

The run folder's name starts with the timestamp the arm's log prints in its
`[rt_log]` header. `--load-run` takes a folder-name PREFIX and must match
exactly one folder (`export_tb_scalars.py:223`), so the timestamp alone is
enough.

```
python scripts/export_tb_scalars.py --load-run <timestamp-of-that-arm> --out docs/figures/RT-187e256_scalars.csv
```

```
python scripts/export_tb_scalars.py --load-run <timestamp-of-that-arm> --out docs/figures/RT-187e512_scalars.csv
```

`--log-root` defaults to `logs/rsl_rl/ur5e_insertion` (`:224`), which is
where both arms land, so it is not passed. Copy both CSVs to the laptop
under the same names.

## The break-even, written down BEFORE arm 2 runs

Arm 1 is running as this is written. The operator read **about 5 s per
iteration at 256 environments** (2026-09-12, from the arm's own log). One
iteration is `num_envs * num_steps_per_env` = `256 * 16` = **4096
environment steps**, so the throughput is **819 environment steps per
second**. This is the reference, and it is the only throughput number this
setup has.

**Two predictions, both falsifiable, registered here before arm 2:**

1. **The iteration time of arm 2.** If the card is saturated at 256, arm 2
   holds 819 steps/s and needs `8192 / 819` = **10.0 s per iteration**. A
   clearly SHORTER time means there was parallel headroom at 256 and the
   card was not the limit. A LONGER time than 10.0 s means 512 costs more
   than linearly — contention, not headroom.
2. **The break-even for the wall-clock verdict.** W favours 512 exactly
   when

   `(iterations_512 / iterations_256) * (time_512 / time_256) < 1`

   A perfect batch would halve the iterations and double the time, leaving
   W unchanged — the neutral case. So 512 wins only by being sublinear in
   time, or better than linear in iterations, or both. Compute this ratio
   explicitly in the verdict; do not eyeball the two numbers.

**A hint, and it is only a hint.** RT-181 read a 13.9 s median at 1024
envs, which is 1179 steps/s — HIGHER than arm 1's 819. That points at
headroom at 256. It does NOT set a number here: RT-181 ran on code stand
`23a7aa0` as a 30-iteration no-DR probe, and `Laufplan_Phase5.md:14` says
an old run never sets a number for this setup. It is written down so the
verdict can say whether the hint held.

## PASS / FAIL lines, in order

Every point names the FILE it is read from. Nothing is judged from the
console, and nothing from prose.

**P1 — both arms actually ran the requested count.** Each log's `[rt_log]
cmd:` header carries `--num_envs 256` / `--num_envs 512`, and the startup
report's env count agrees. An arm whose header and report disagree is VOID.

**P2 — the six metrics, in order, per arm.** From the scalars CSV, per
`CLAUDE.md`: Episode Reward → Policy Loss → Value Loss → Entropy →
Explained Variance → KL, each with a trend. Explained Variance and KL are
not logged by rsl_rl 3.0.1 (D-116); the substitute is `Loss/learning_rate`
under the adaptive KL controller, as RT-174 and RT-176 did it. **No verdict
before all six are written down.**

**P3 — the bar, per arm.** From the scalars CSV: the FIRST iteration whose
`Episode/success_rate` is `>= 0.90`, and the first `>= 0.995`. Write both
numbers for both arms. If an arm never reaches a bar within 1500, write
"not reached within 1500" — that is a number too.

**P4 — the wall clock, per arm.** The mean `Iteration time` from the log,
via `scripts/filter_rt_log.py --series "Iteration time"`, times P3's
iteration count. This is reading W and it DECIDES the comparison. Report the
mean and the spread of the iteration times, not the mean alone: a mean over a
run whose times drift is not a rate.

**P4b — `nvidia-smi` is NOT evidence for P4, and may not be used as such.**
Observed by the operator during arm 1 on 2026-09-12: 3823-3887 MiB used of
the card's 10053 MiB (`rt_logs/RT-179/RT-179.txt:21`), i.e. **38 %**, with
`utilization.gpu` reading 46 % at start and then 98 %.

* The MEMORY figure is usable and is worth recording: 38 % at 256 envs says
  512 fits with headroom. That is a capacity statement, and it is why arm 2
  is expected to start at all.
* The UTILISATION figure is not usable for the throughput question.
  `utilization.gpu` is the fraction of the sample period in which ANY kernel
  was resident — a time occupancy, not a throughput. A single small kernel
  holds it at 98 % on a card that is nowhere near its limit. "98 % therefore
  the card is saturated therefore 512 cannot help" is exactly the inference
  this line forbids.

P4's `Iteration time` is the only admissible measurement of throughput here.

**P5 — S and I, per arm.** `iterations × num_envs × 16` and `iterations`
alone. Recorded even when they disagree with W. If W and S point opposite
ways, say so explicitly — that is the interesting outcome, not a problem.

**P6 — the interpenetration, per arm.** From
`logs/RT-187e256_demo_metrics.json` and `..._e512_...`: `interpen_max_mm`
present with `episodes > 0` and `measured` not `false`, then `mean_mm`,
`p95_mm`, `max_mm` and `over_thresh` against `episodes`. This is the
contact-buffer question. **Named limit: with one arm per count there is no
run-to-run spread, so only a LARGE difference may be called a finding, and
the verdict must say that no spread was measured.**

**P7 — the reset-step check.** The start band is +30..+50 mm above the
opening, so the part is far from the fixture at reset. If `p50_mm` is
non-zero by a similar amount in BOTH arms, suspect the reset step rather than
the physics: `force_norm_n` is already known to read a reset transient
(2.021 N in every env, RT-149 DETAIL (c)). Named risk, not a prediction.

**P8 — the red flags, self-reported per `CLAUDE.md`.** Entropy to 0 early;
grad norm pinned or NaN; success collapsing after a change; Value Loss
diverging. Report them whether or not they change the verdict.

**P9 — no reward hacking in either arm.** A rising Episode Reward with a
flat success rate is a hacking finding. If it appears, break out
`Episode_Reward/*` per term before any other judgement.

**P10 — the log is complete.** Each arm's log ends with an exit line. A
missing exit line does not by itself fail an arm (RT-178 and RT-181 both
lack one) but must be stated.

## What this run cannot answer

* **One seed per arm.** No run-to-run spread is measured, so a small
  difference in either direction means nothing. If the two arms land close,
  the honest verdict is "not separated by this run", and the next step is a
  second seed per arm, not a decision.
* **It does not answer the solver question.** Both arms run the authored
  16 / 1 position and velocity iterations (`ur5e_cfg.py:182-183`, D-127).
  16 vs 192 stays open.
* **It says nothing about 128 or 1024.** Only the two counts that ran.
  Extrapolating the wall-clock trend past them is not supported.
* **One GPU.** The RTX 3080 with 10053 MB (`rt_logs/RT-179/RT-179.txt:21`).
  The throughput ranking is a property of this card.
* **The bar is the TRAINING rate with exploration**, a rolling window of
  2000 episodes (`insertion_env.py:1055-1056`), not a test rate. Two arms
  are compared on the same instrument, which is what makes it fair, but the
  number is not an evaluation result.
* **Nothing here is a thesis number.** A probe on training seed 20 that
  picks a setting.

## Status

**UNVERIFIED.** Nothing in this file has run.
