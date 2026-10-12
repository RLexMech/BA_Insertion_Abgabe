# RT-239s1 -- expectation, written BEFORE the run (2026-09-27): RT-238s1's setup with D-198 (4 mini-batches instead of 32), 400 iterations

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Number: highest over every `origin/*` branch is RT-238 (`git ls-tree -r
--name-only` over `rt_logs`, 2026-09-27). Template for everything not named
here: `rt_logs/RT-238_expectation.md` (same clone, same arm, same flags, same
seed) and, through it, RT-236, RT-235 and RT-233. Only what differs is
stated below.

## Context

RT-238s1 (D-197 pilot) PASSED (`rt_logs/VERDICTS.md` 2026-09-27 02:54:12):
floor phase done at 23, full DR width at 361, success END WINDOW 393..399
0.911 and still rising. The adaptive learning rate sat at its 1e-5 floor
in 119/200 iterations of 0..199 and 177/200 of 200..399, median 1e-5 in
both halves (recounted 2026-09-27 from `rt_logs/RT-238s1/RT-238s1_
scalars.csv`, `Loss/learning_rate`, test `lr <= 1.001e-5`, scratch
`lr_floor_count.py`). D-198 (`DECISIONS.md`) sets `num_mini_batches` 32 ->
4, Isaac Lab's rsl_rl default: 20 instead of 160 rate adjustments per
iteration. This run is RT-238s1 with exactly that one change.

## Code state

Clone `Phase5_MitAktion_v1` (as RT-238). It needs a PULL: RT-238s1 ran
`5512e64` (`rt_logs/RT-238s1/RT-238s1.txt` L4). Branch `p5_mitAktion`, the
commit that adds this file; its SHA is read back from
`origin/p5_mitAktion` at handover and given in chat. Every log's `[rt_log]
git:` line must match it before any point below is judged.

Training path against `5512e64` (`git diff --stat 5512e64..HEAD -- source
scripts/rsl_rl`): ONE file, `agents/rsl_rl_ppo_cfg.py`, 5 insertions, 2
deletions. `ba599e00`: `PPORunnerMlpAblationCfg.__post_init__` sets
`self.algorithm.num_mini_batches = 4` (was 32; `:93`); comment and
docstring. The LSTM cfg inherits it. Env, `insertion_math`, `train.py`
unchanged.

Offline (laptop): `check_env_wiring.py` red on the old cfg (2 checks), then
433 PASS; `check_mutation_anchors.py` 598 anchors match; 4 named mutations
each flip exactly their declared checks (scratch runner, not the full
sweep).

`CODE_MARKER = "insertion-osc-2026-09-03a"` (`insertion_env.py:76`),
unchanged -- it does NOT tell this code from RT-238's. The log prints no
mini-batch count (RT-238s1 log: no line holds `mini_batch`). The gate is
the git line, the `[rt_log] cmd:` line and the pin in `params/agent.yaml`
(P1).

## The one hypothesis

**With 4 mini-batches the adaptive learning rate leaves the floor: in
iterations 200..399 `Loss/learning_rate` sits at 1e-5 in FEWER than
RT-238s1's 177 of 200 iterations.**

Counted as for RT-238s1 (`lr <= 1.001e-5`, same script). Same setup, same
seed, one change (D-198 Attribution), so the difference is attributed to
the mini-batch count -- for this one seed.

Falsified by: 177 or more of 200 at the floor (D-198 "What would make it
wrong": the collapse is then not caused by the update count). Not testable:
a Traceback or a stop before 399.

## Commands

Training PC, PowerShell, conda env `env_isaaclab` active. Pull, then verify:

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_MitAktion_v1
git pull --ff-only origin p5_mitAktion
git rev-parse HEAD
git status --porcelain --untracked-files=no
pip show insertion
```

Expected: HEAD = the SHA from chat; `git status` no output; `Editable project
location` ends in `Phase5_MitAktion_v1\source\insertion`.

Watcher in a second window: as RT-233 (`rt_logs/RT-233_expectation.md`
§ Commands, the `while ($true)` line).

The run (ONE seed). No `--max_iterations`: 400 from the cfg (as RT-238):

```powershell
.\scripts\rt_log.ps1 RT-239s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --agent rsl_rl_rnn_cfg_entry_point --num_envs 256 --headless --seed 1 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Wall time: unknown for this setup. RT-238s1 ran 7.30 h (`HANDOFF-RL.md`
START HERE, `wall_time`); the update now takes 20 larger gradient steps
instead of 160 smaller ones over the same samples. NOT a bar.

## Points -- RT-239s1

* **P1 setup** (the gate; any line wrong = void, nothing below judged --
  precedent RT-151b): everything RT-238 P1 names, with `RT-239s1` in place
  of `RT-238s1` (run folder
  `..._RT-239s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1`),
  the D-197 startup lines unchanged. PLUS, read after the run:
  `params/agent.yaml` `algorithm.num_mini_batches: 4`. The log alone
  cannot show it.
* **P1b the band's premise**: as RT-238 P1b (`reset yaw vs seat` max
  <= 0.011 deg). The reset path is unchanged since RT-238s1 (0.005 deg,
  its L80) -- a reference, not the bar.
* **P2 floor**: as RT-238 P2 (`dr/phase` 0 -> 1 iteration, reported
  beside RT-238s1's 23).
* **P3 boundaries**: as RT-238 P3 (`dr/all_at_max` first iteration,
  reported beside RT-238s1's 361).
* **P4 curves**: as RT-238 P4. THE HYPOTHESIS lives here:
  `Loss/learning_rate` at 1e-5 per half (0..199, 200..399), median and max
  per half, min / max / 399. Beside it RT-238s1: 119/200 and 177/200,
  median 1e-5 in both halves. Also count the iterations at the 1e-2
  ceiling (rsl_rl's cap, `HANDOFF-RL.md` RT-236s1 LEARNING RATE).
  `Policy/mean_noise_std` with its trend (RT-238s1 0.99 -> 0.87).
  `Episode_Reward/action_rate` 0.0 in every row, as RT-238.
* **P5 success**: as RT-238 P5 (steps 0 and 399, minimum, END WINDOW
  393..399, 50-it means, bins of `demo_metrics.json`). Beside RT-238s1's
  END WINDOW 0.911: same setup, one change, one seed each -- a reading of
  the lever, no bar.
* **P6 end**: as RT-238 P6 with `RT-239s1`; hand back the same files,
  including the `check_episode_log.py --find-run RT-239s1` output (missing
  for RT-238s1).
* **P7 physics**: as RT-233 P7, reported.
* **P8 force and action**: as RT-238 P8. Watched, carried over from
  RT-238s1: `action_sat_frac` 0.32 -> 0.58 and `action_abs_max_mean` 2.9
  -> 5.2 (the D-168 drift pair); force max p95 24.1 N and
  `Episode/force_abort_rate` 0.015 over 393..399 (`HANDOFF-RL.md` START
  HERE, RT-238s1 DONE). Reported, no bar.
* **Reported, no bar**: the later startup reports' `yaw vs seat` (mid-
  episode, ACTUAL pose); RT-238s1 read 21.6 / 24.0 / 21.7 deg (L332, L645,
  L871), past the 12.30 deg band (the band clamps the TARGET, L-13).

## Pins (read from the files, not from the log prose)

* `params/agent.yaml`: `algorithm.num_mini_batches: 4` (was 32 in
  `rt_logs/RT-238s1/RT-238s1_agent.yaml`); everything else as RT-238s1's
  file: `num_steps_per_env: 128`, `max_iterations: 400`, `save_interval:
  50`, `num_learning_epochs: 5`, `learning_rate: 0.001`, `schedule:
  adaptive`, `desired_kl: 0.01`, `rnn_type: lstm`, `rnn_hidden_dim: 256`,
  `rnn_num_layers: 2`, `seed: 1`. Source: `agents/rsl_rl_ppo_cfg.py`.
* `params/env.yaml`: as RT-238 (no env change).
* `demo_metrics.json` `rl_placeholders`: as RT-238.
* `autodr_399.json`: `"format": "autodr-sidecar-1"`, `"seed": 1`.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| exit 0 at 399, lr at floor in < 177/200 of 200..399 | the update count drove the collapse (this seed) | report success beside RT-238s1; user decides the next lever (more iterations if success still rises at 399) |
| exit 0 at 399, lr at floor in >= 177/200 of 200..399 | D-198 does not free the rate; the collapse has another cause | stop; user decides (D-198 options C / D have no source) |
| lr often at the 1e-2 ceiling and success falls | the rate now overshoots | report; user decides |
| success END WINDOW clearly below RT-238s1's 0.911 with the rate off the floor | fewer, larger steps learn less per iteration (D-198, What would make it wrong) | report; user decides |
| `action_sat_frac` climbing past RT-238s1's 0.58 | the D-168 drift grows with the faster rate | report; user decides |
| P1b as RT-238 (between 0.011 and 12.30 deg, or about 180 deg) | as RT-238 | as RT-238 |
| Traceback `CUDA out of memory` in the PPO update | a recurrent mini-batch now holds 8 x the samples | stop; hand the Traceback back |
| a stop before 399 without Traceback | not a result | tell the user; rerun the same seed from scratch |

## What this run cannot answer

* Whether the effect holds across seeds: one seed against one seed.
* Whether more iterations would still help: success was still rising at
  399 in RT-238s1; this run keeps 400.
* Whether the hectic approach, the park and the big yaw turn are gone:
  that needs the replay of `model_399.pt`.
* What the LSTM contributes (MLP ablation arm, which shares the 4), the
  test success rate and the force learning effect (evaluation side not
  ready, D-195 seam 6).
