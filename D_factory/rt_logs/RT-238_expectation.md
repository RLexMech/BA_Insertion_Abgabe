# RT-238s1 -- expectation, written BEFORE the run (2026-09-27): the D-195 pilot with D-197 (Factory's steps 0.02 m / 0.097 rad, action EMA 0.2, action-rate weight 0.0; yaw band 12.30 deg), 400 iterations

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Number: highest over every `origin/*` branch is RT-237 (`git ls-tree -r
--name-only` over `rt_logs`, 2026-09-27; RT-237 was withdrawn unrun).
Template for everything not named here: `rt_logs/RT-236_expectation.md`
(same clone, same arm, same flags, same seed) and, through it, RT-235 and
RT-233. Only what differs is stated below.

## Context

RT-236s1 (D-196 form: 0.01 m / 1 deg steps, band 7.74 deg) learned slowly
and was stopped by the user at ~it 170 ("the policy is useless"); no
pre-registered end, no VERDICTS line yet (`HANDOFF-RL.md` START HERE,
RT-236s1 STOPPED). Mid-run readings off screenshots: `dr/phase` 1 only at
~143 (RT-235s1: 34), noise std rising, the part parks above the pocket.
D-197 (`DECISIONS.md`) takes Factory's step thresholds back TOGETHER with
Factory's action EMA, and (user 2026-09-27, "A, like Factory") Factory's
action-rate weight 0.0; the D-196 yaw band stays at its rule's width. User
2026-09-27: D-197 first, before the mini-batch lever; train 400 iterations.
This is that run.

## Code state

Clone `Phase5_MitAktion_v1` (as RT-236). It needs a PULL: RT-236s1 ran
`ae5fd00` (`rt_logs/RT-236s1/RT-236s1.txt`, `[rt_log] git:`). Branch
`p5_mitAktion`, the commit that adds this file; its SHA is read back from
`origin/p5_mitAktion` at handover and given in chat. Every log's
`[rt_log] git:` line must match it before any point below is judged.

Training path against `ae5fd006` (`git diff --stat ae5fd006..5e4fd9cf --
source scripts/rsl_rl`: `insertion_env.py`, `insertion_env_cfg.py`,
`scripted_policy.py` (docstring only); three commits, offline-checked, none
run under Isaac):

* `2a3ab8dd` cfg: `osc_pos_step_limit_m = 0.02`, `osc_rot_step_limit_rad =
  0.097`, `osc_pos_clamp_m = 0.080`, `osc_pos_clamp_z_m = 0.1435`,
  `osc_yaw_clamp_rad = math.radians(12.30)` (were 0.01 / 1 deg / 0.070 /
  0.1335 / 7.74 deg). Startup lines relabelled D-197.
* `eb03028f` the action EMA: cfg `action_ema_factor = 0.2` (+
  `RL_PLACEHOLDERS`); env `_ema_actions` updated in `_pre_physics_step` by
  `insertion_math.ema_update(self._ema_actions, clamped, 0.2)` before the
  mode branch; the OSC delta reads the EMA state; `_reset_idx` zeroes it;
  `force_prevact`'s `prev_action` IS the EMA state (was the raw draw). New
  startup line `action EMA:`. `ema_update` itself ran under TorchScript
  since D-114 (force channel); the new call site has not.
* `5e4fd9cf` cfg: `action_rate_scale = 0.0` (was 0.0034, D-168), Factory's
  `action_grad_penalty_scale` (`factory_tasks_cfg.py:71`). A REWARD change:
  the term is still measured on the raw draw and logged, it pays nothing.

Offline (laptop): `check_env_wiring.py` 433 PASS; `check_mutation_anchors.py`
all anchors match; 29 named mutations on step, box, band, EMA and the rate
weight each flip exactly their declared checks (scratch runner, not the
full sweep). `insertion_math.py` unchanged.

`CODE_MARKER = "insertion-osc-2026-09-03a"` (`insertion_env.py:76`),
unchanged -- it does NOT tell this code from RT-236's. The gate is the git
line, the `[rt_log] cmd:` line and the D-197 startup lines (P1).

## The one hypothesis

**The D-197 form trains at the 400-iteration budget: the run reaches
iteration 399 with exit 0 and leaves the AutoDR floor phase (`dr/phase`
reaches 1).**

Same learning event as RT-236 (defined by code: `ADVANCE_AT = 0.80`,
`curriculum.py:53`, via `autodr.py:743`). The iteration where it happens
is REPORTED beside RT-235s1's 34 (`VERDICTS.md` 2026-09-26 13:16:47) and
RT-236s1's ~143 (screenshot, approximate); neither is a bar for this setup.

Falsified by: a Traceback or a stop before 399; `dr/phase` 0 at every step.

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

The run (ONE seed). No `--max_iterations`: 400 from the cfg (as RT-236):

```powershell
.\scripts\rt_log.ps1 RT-238s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --agent rsl_rl_rnn_cfg_entry_point --num_envs 256 --headless --seed 1 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Wall time: unknown for this setup. RT-236s1 ran 73.5 s per iteration over
it 0..8 (`HANDOFF-RL.md` START CHECK RT-236s1), about 8.2 h for 400; the
EMA adds one tensor line per step. NOT a bar.

## Points -- RT-238s1

* **P1 setup** (the gate; any line wrong = void, nothing below judged --
  precedent RT-151b): everything RT-236 P1 names, with `RT-238s1` in place
  of `RT-236s1` (run folder
  `..._RT-238s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1`).
  The D-197 lines of the controller report (`insertion_env.py`
  `_print_controller_report`) replace RT-236's D-196 lines:
  * `step limits:  0.02 m, 0.097 rad (5.56 deg) per policy step
    [placeholder, Factory thresholds, D-197]`;
  * `action EMA:   0.2 on the new clamped draw, applied = EMA state [...];
    obs prev_action = EMA state; action-rate term measured on the RAW draw,
    weight 0.0 (Factory 0.0)` (NEW);
  * the reward line ends `action-rate penalty -0.0 * ||a_t - a_(t-1)||`
    (`insertion_env.py`, reward block of the startup report);
  * `tip clamp:    +-0.08 m x/y [...] and +-0.1435 m z [...]`;
  * `yaw band:     +-12.30 deg about THIS EPISODE'S SEATED orientation [...]`;
  * `reset yaw vs seat: max |yaw| ... deg over 256 envs` -- present, not
    `UNREAD`; its VALUE is P1b;
  * `Decision (4) reading: ... * 0.02 m) = ...` (the step factor is 0.02);
  * the placeholder block lists `action_ema_factor = 0.2`.
* **P1b the band's premise**: `max |yaw|` of `reset yaw vs seat` <=
  **0.011 deg** keeps the band's rule with the measured home (band 12.30
  deg minus need 5.4508 + 1.28 + 5.5577 = 12.2885 deg; start yaw 5.4508
  from the RT-236 ADDENDUM, computed by scratch `p1b_rt238.py`); <= 0.024
  deg against the closed-form need (`check_env_wiring` detail line). The
  reset path is unchanged since RT-236s1 (only `_reset_idx` gained the EMA
  zeroing); RT-236s1 read 0.005 deg (its L79) -- a reference, not the bar.
  0.011 .. 12.30 deg: valid, the band is short of its rule by the excess
  (user decides); about 180 deg or >= 12.30 deg: wrong frame or home --
  stop, void.
* **P2 floor**: as RT-236 P2 (`dr/phase` 0 -> 1 iteration, reported).
* **P3 boundaries**: as RT-236 P3.
* **P4 curves**: as RT-236 P4, including `Loss/learning_rate` min / max /
  399 and the 1e-5 counts per half (the 32 mini-batches are unchanged; the
  lr question stays open beside D-197). `Policy/mean_noise_std` with its
  trend: nothing taxes it any more (rate weight 0.0). Reported, no bar.
  `Episode_Reward/action_rate` must read 0.0 in every row (weight 0.0); a
  nonzero value means the cfg did not load.
* **P5 success**: as RT-236 P5 (steps 0 and 399, minimum, END WINDOW
  393..399, bins of `demo_metrics.json`).
* **P6 end**: as RT-236 P6 with `RT-238s1` (`model_0..399.pt` every 50,
  `autodr_399.json`, `[scalars] wrote`, exit 0; hand back `scalars.csv`,
  `demo_metrics.json`, `autodr_399.json`, `params/`, `episodes.csv`,
  `episodes_meta.json`, the short log, `check_episode_log.py --find-run
  RT-238s1`).
* **P7 physics**: as RT-233 P7, reported.
* **P8 force and action**: as RT-236 P8 (force max mean / p95 and
  `Episode/force_abort_rate`, means over 0..6 and 393..399). Watched item
  for D-197's "what would make it wrong": the EMA lag could raise the
  force-abort rate. Plus the D-168 drift pair `action_abs_max_mean` and
  `action_sat_frac` (curve, 0..6 and 393..399): with the rate weight at 0.0
  nothing prices the raw command. Reported, no bar.

## Pins (read from the files, not from the log prose)

* `params/agent.yaml`: as RT-236 (`num_steps_per_env: 128`,
  `max_iterations: 400`, `save_interval: 50`, lstm 256 x 2, 32
  mini-batches). No change.
* `params/env.yaml`: `osc_pos_step_limit_m: 0.02`,
  `osc_rot_step_limit_rad: 0.097`, `osc_pos_clamp_m: 0.08`,
  `osc_pos_clamp_z_m: 0.1435`, `osc_yaw_clamp_rad: 0.21467549799530256`
  (`math.radians(12.30)`, printed by `p1b_rt238.py` and by the
  `check_env_wiring` detail line), `action_ema_factor: 0.2` (NEW),
  `action_rate_scale: 0.0` (was 0.0034),
  `osc_tilt_clamp_rad: 0.14870205226991687` (unchanged); plus RT-235's pins
  (`obs_wrench_mode: force_prevact`, `observation_space: 34`,
  `state_space: 34`, `pocket_obs_yaw_rad: 0.0223402`, `dr_mode: autodr`,
  `seed: 1`). Source: `insertion_env_cfg.py`.
* `demo_metrics.json` `rl_placeholders`: the osc values above and
  `action_ema_factor: 0.2` (it joined `RL_PLACEHOLDERS`, `eb03028f`;
  `insertion_env.py` writes the dict from that table).
* `autodr_399.json`: `"format": "autodr-sidecar-1"`, `"seed": 1`.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| exit 0 at 399, `dr/phase` reaches 1 | the D-197 form trains at this budget | replay `model_399.pt` (hectic approach? big yaw turn? park?); then the other seeds and arms, each its own expectation |
| exit 0 at 399, `dr/phase` 0 throughout | D-197 does not free the floor phase | stop; user decides (next lever: mini-batches 32 -> 4) |
| `Episode/force_abort_rate` clearly above RT-235s1's over the same steps | reads as the EMA lag (D-197, What would make it wrong) | report; user decides |
| `Loss/learning_rate` at 1e-5 in most of the run | the collapsed adaptive rate is still there under D-197 | report beside success; the mini-batch lever stays next |
| `action_sat_frac` climbing while `Policy/mean_noise_std` looks healthy | the D-168 named failure: exploration dead inside the clamp, now with no rate price | report; user decides |
| P1b between 0.011 and 12.30 deg | the home is not exactly seated; band short of its rule | valid; user decides |
| P1b about 180 deg, or >= 12.30 deg | wrong frame or home | stop, void; hand the startup block back |
| Traceback naming `ema_update`, `_ema_actions` or a shape in `_pre_physics_step` | seam 2 wiring, untested under Isaac | stop; hand the Traceback back |
| a stop before 399 without Traceback | not a result | tell the user; rerun the same seed from scratch |

## What this run cannot answer

* Which change did what: against RT-236s1 the step, the EMA, the band
  width and the rate weight change together; against RT-235s1 the band, the
  EMA, the rate weight and the budget (D-197 Attribution). A side-by-side
  is a reading, not a comparison.
* Whether the lr collapse (32 mini-batches x 5 epochs) slows learning: it is
  kept here; only the mini-batch lever run tests it.
* Whether the hectic approach, the park and the big yaw turn are gone: no
  training tag measures them; that needs the replay of `model_399.pt`.
* Whether dropping the rate weight (0.0034 -> 0.0) matters on its own: it
  changes together with the rest (D-197 (iii)).
* What the LSTM contributes (MLP ablation arm), the test success rate and
  the force learning effect (evaluation side not ready, D-195 seam 6).
