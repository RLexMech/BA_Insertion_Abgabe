# RT-236s1 -- expectation, written BEFORE the run (2026-09-26): the D-195 pilot again, now with D-196 (yaw band, 1 cm / 1 deg steps, smaller box) and the 400-iteration budget

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Number: highest over every `origin/*` branch is RT-235 (`git ls-tree -r
--name-only` over `rt_logs`, 2026-09-26). Template for everything not named
here: `rt_logs/RT-235_expectation.md` (same clone, same arm, same flags, same
seed). Only what differs from it is stated below.

## Context

RT-235s1 (D-195 pilot, LSTM, AutoDR seed 1, 225 it) passed its gate and
trained (`rt_logs/VERDICTS.md` 2026-09-26 13:16:47: exit 0, 4.09 h, phase dr
at 34). Two things followed from it before this run:

* D-196 (`DECISIONS.md`): the replay of `model_224.pt` showed a hectic
  approach, a long park and a large yaw turn into a jam. The user chose (A) a
  yaw band and (B) smaller per-step limits, no reward change.
* D-195 BUDGET: 225 iterations were too few (success still rising,
  `dr/all_at_max` never reached, `HANDOFF-RL.md` item 2a); the user set 400.

D-196 changes the action interface, so the pilot is repeated before the
other arms. This is that repeat.

## Code state

Clone `Phase5_MitAktion_v1` (as RT-235). It needs a PULL: RT-235s1 ran
`6792b78` (`rt_logs/RT-235s1/RT-235s1.txt`, `[rt_log] git:`). Branch
`p5_mitAktion`, the commit that adds this file; its SHA is read back from
`origin/p5_mitAktion` at handover and given in chat. Every log's
`[rt_log] git:` line must match it before any point below is judged.

What changed on the training path against `6792b78` (`git diff --stat
6792b78..8856a4c1 -- source scripts/rsl_rl`: only these four commits; all
offline-checked, none run under Isaac):

* `318b66cc` `insertion_math.clamp_yaw_to_band`: the commanded tool yaw is
  pulled back into a band about THIS episode's SEATED orientation (pocket *
  `SEATED_TOOL_QUAT_LOCAL`). Checked only under the numpy stand-in; it has
  NEVER been compiled by `torch.jit.script` (the laptop has no torch).
* `90922b4a` `insertion_env_cfg.py`: `osc_pos_step_limit_m = 0.01`,
  `osc_rot_step_limit_rad = math.radians(1.0)`, `osc_pos_clamp_m = 0.070`,
  `osc_pos_clamp_z_m = 0.1335` (were 0.02 / 0.097 / 0.08 / 0.1435).
* `defab1d3` `insertion_env.py` `_apply_osc`: `seat_quat_w =
  insertion_math.quat_mul(self._fixture_quat, self._seat_quat_local)`, the
  band call after the tilt cone and before the box; cfg
  `osc_yaw_clamp_rad = math.radians(7.74)` (D-196 seam 0: worst start yaw +
  belief error + one rotation step). Startup report: new lines `yaw band:`
  and `reset yaw vs seat:`.
* `dbb42c78` `agents/rsl_rl_ppo_cfg.py`: `PPORunnerMlpAblationCfg.
  max_iterations = 400` (the LSTM cfg inherits it).

Offline (laptop): `check_env_wiring.py` 425 PASS; `check_insertion_math.py`
all new checks PASS, 9 OLD reds unchanged (the `validate_rl_config` guard
tests); `check_mutation_anchors.py` 587 anchors match; named mutations each
flip exactly their declared checks (scratch runners, not the full sweep).

`CODE_MARKER = "insertion-osc-2026-09-03a"` (`insertion_env.py:76`),
unchanged -- it does NOT tell this code from RT-235's. The gate is the git
line, the `[rt_log] cmd:` line and the D-196 startup lines (P1).

## The one hypothesis

**The D-196 form trains at the 400-iteration budget: the run reaches
iteration 399 with exit 0 and leaves the AutoDR floor phase (`dr/phase`
reaches 1).**

Why this: the failure D-196 names for itself is "steps so small that the
part cannot reach the pocket" (D-196, What would make it wrong), and a band
too narrow would block the yaw alignment. Leaving the floor needs the
AutoDR rule's success threshold at the floor boundary (`ADVANCE_AT = 0.80`,
`curriculum.py:53`, via `autodr.py:743`), as in RT-235 -- a learning event
defined by code, not by a number typed here. RT-235s1's step 34 is a
different setup and sets no bar.

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

The run (ONE seed). No `--max_iterations`: the 400 come from the cfg
(`train.py:245-247` keeps the cfg value when the flag is absent):

```powershell
.\scripts\rt_log.ps1 RT-236s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --agent rsl_rl_rnn_cfg_entry_point --num_envs 256 --headless --seed 1 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Wall time: unknown for this setup. RT-235s1 took 4.09 h for 225 iterations
(`VERDICTS.md` 2026-09-26 13:16:47); scaled by 400/225 that is 7.3 h, which
ASSUMES D-196 leaves the cost per iteration unchanged -- not measured, not a
bar.

## Points -- RT-236s1

* **P1 setup** (the gate; any line wrong = the run is void, nothing below is
  judged -- precedent RT-151b, `rt_logs/VERDICTS.md` 2026-09-05 14:51:11):
  everything RT-235 P1 names, with `RT-236s1` in place of `RT-235s1` (git =
  handover SHA; cwd; cmd flags; robot asset; `force_prevact-34`; layout
  `28:34 prev_action`; the `pocket-yaw belief error` line; `Actor RNN` /
  `Critic RNN` `LSTM(34, 256, num_layers=2)`; MLP first layer
  `in_features=256`; NO `obs_groups` UserWarning; `force abort at 30.0 N`;
  `provider: AutoDR, 8 boundaries, bounds_version 0`; the `START FLOOR`
  line; run folder `..._RT-236s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1`).
  NEW, the D-196 lines of the controller report (`insertion_env.py`
  `_print_controller_report`):
  * `step limits:  0.01 m, 0.017453292519943295 rad (1.00 deg) per policy
    step [placeholder, own, D-196]`;
  * `tip clamp:    +-0.07 m x/y [...] and +-0.1335 m z [...]`;
  * `yaw band:     +-7.74 deg about THIS EPISODE'S SEATED orientation [...]`;
  * `reset yaw vs seat: max |yaw| ... deg over 256 envs (env 0 ...)` --
    present and not `UNREAD`. Its VALUE is P1b;
  * `Decision (4) reading: ... * 0.01 m) = ...` (the step factor is 0.01).
* **P1b the band's premise** (NEW): at DR width 0 (AutoDR starts every
  boundary at its centre) the reset yaw vs seat IS the home tool's deviation
  from the seated orientation. The band covers it up to band minus need =
  0.1350884841043611 - 0.13471429610827554 rad = **0.021 deg** (the two
  numbers of the `check_env_wiring` detail line "the yaw band holds the worst
  start yaw ..."). Readings: `max |yaw|` <= 0.021 deg: the rule holds;
  0.021 .. 7.74 deg: the run stays valid, the band is short of its rule by
  the excess (user decides on D-196); about 180 deg or >= 7.74 deg: the frame
  or the home is wrong, every first target is turned -- stop, void.
* **P2 floor**: `reached H_min, phase dr` line; `dr/phase` 0 -> 1 iteration.
  Reported, no bar.
* **P3 boundaries**: first iteration with `dr/all_at_max` = 1, and its value
  at 399. Reported, no bar (the 400 follow the user's own extrapolation
  "full width near ~375" from RT-235s1, one seed, another setup).
* **P4 curves** in order (reward, surrogate, value, entropy) with trend;
  `Policy/mean_noise_std` not below 0.2; `Loss/learning_rate`: min, max,
  value at 399, and how many steps sit at 1e-5 in steps 0..199 and 200..399
  (halves, a split, not a bar).
* **P5 success**: `Episode/success_rate` at steps 0 and 399, its minimum and
  where, the curve, and the END WINDOW mean over 393..399 (D-195 END
  WINDOW); `dr/train_success_regular_fresh`; the bin tables of
  `demo_metrics.json` (`success_by_tilt_bin`, `success_by_yaw_bin`,
  `success_by_lateral_bin`, `success_by_start_height_bin`).
* **P6 end**: `Learning iteration 399/400`; checkpoints `model_0/50/100/150/
  200/250/300/350/399.pt` (`save_interval: 50`, `params/agent.yaml`; saves at
  `it % 50 == 0` and at the end, D-195 FORCE); `autodr_399.json`;
  `[scalars] wrote`; `[rt_log] exit code: 0`; `wall_time` at step 399 of
  `scalars.csv`. Files handed back: `scalars.csv`, `demo_metrics.json`,
  `autodr_399.json`, `params/env.yaml`, `params/agent.yaml`, `episodes.csv`,
  `episodes_meta.json`, the short log, and the output of
  `python scripts/check_episode_log.py --find-run RT-236s1`.
* **P7 physics**: as RT-233 P7, reported, no bar.
* **P8 force and action**: as RT-233 P8, plus the means of
  `Episode/force_max_mean_n`, `Episode/force_max_p95_n` and
  `Episode/force_abort_rate` over steps 0..6 and 393..399. Reported, no bar
  (the curriculum caveat of RT-235 still holds).
* **P9 tilt shaping** (NEW, the user's watch item, D-196 "Consequence NOT
  acted on"): [PREMISE VOID, found 2026-09-26 at the start check, AFTER
  the run began: the term is OFF by config (`w_tilt` 0.0 since D-172; log
  L169 `*** OFF, w_tilt 0 ***`), so `Episode_Reward/tilt_shaping` is 0.0
  by construction (it is 0.0000 in it 0..8, as in all of RT-235s1). P9
  reports only `success_by_tilt_bin` at the end. `pruefregeln.md` L-16.
  It read:] with a 1 deg rotation step the tilt potential (D-171) moves
  about 5.6 x less per step than at 0.097 rad, while the per-step contact
  penalty stays. Report `Episode_Reward/tilt_shaping` beside
  `Episode_Reward/total` (curve, value at 0 and 393..399 mean) and
  `success_by_tilt_bin` at the end. Reported, no bar -- no tag isolates the
  effect.

## Pins (read from the files, not from the log prose)

* `params/agent.yaml`: `num_steps_per_env: 128`, `max_iterations: 400`,
  `save_interval: 50`; the rest as RT-235 (`ActorCriticRecurrent`, lstm, 256,
  2 layers, `[128, 128]`, 32 mini-batches, obs_groups). Source:
  `agents/rsl_rl_ppo_cfg.py`.
* `params/env.yaml` (keys as `RT-235s1_env.yaml:117-125` dumps them):
  `osc_pos_step_limit_m: 0.01`, `osc_rot_step_limit_rad:
  0.017453292519943295`, `osc_pos_clamp_m: 0.07`, `osc_pos_clamp_z_m:
  0.1335`, `osc_tilt_clamp_rad: 0.14870205226991687` (unchanged),
  `osc_yaw_clamp_rad: 0.1350884841043611` (NEW); plus RT-235's pins
  (`obs_wrench_mode: force_prevact`, `observation_space: 34`, `state_space:
  34`, `pocket_obs_yaw_rad: 0.0223402`, `dr_mode: autodr`, `seed: 1`).
  Source: `insertion_env_cfg.py`; values by `python -c "import math;
  print(math.radians(1.0), math.radians(7.74))"`.
* `demo_metrics.json` `rl_placeholders`: the same five osc values, and
  `osc_yaw_clamp_rad` present (it joined `RL_PLACEHOLDERS`, `defab1d3`);
  `"pocket_obs_yaw_rad": 0.0223402`, `"obs_version": "force_prevact-34"`.
* `autodr_399.json`: `"format": "autodr-sidecar-1"`, `"seed": 1`.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| exit 0 at 399, `dr/phase` reaches 1 | the D-196 form trains at this budget | replay `model_399.pt` (does the hectic approach / the big yaw turn go?); then s2..s5 and the other arms, each its own expectation |
| exit 0 at 399, `dr/phase` 0 throughout | the D-196 limits keep the policy on the floor | stop; user decides (D-196 reopened) |
| P1b `max |yaw|` between 0.021 and 7.74 deg | the home is not exactly the seated orientation; the band is short of its rule | run stays valid; user decides whether the band moves |
| P1b about 180 deg, or >= 7.74 deg | wrong frame or wrong home: every first target is turned | stop, void; hand the startup block back |
| Traceback at import naming `clamp_yaw_to_band`, `rotate_into_frame` or TorchScript | the new function never ran under `torch.jit.script` | stop; hand the Traceback back |
| Traceback naming `seat_quat_w`, `_seat_quat_local` or a shape in `quat_mul` | seam 2 wiring, untested under Isaac | stop; hand the Traceback back |
| `Loss/learning_rate` at 1e-5 in most of 200..399 | reads as a collapsed adaptive rate | report beside the success curve; user decides |
| [VOID 2026-09-26, see P9: the term is off, 0.0 by construction] `Episode_Reward/tilt_shaping` near 0 while `success_by_tilt_bin` falls in the high bins | reads as the user's point 5 (weaker per-step tilt pull) | report; user decides (reward is off-limits by the user) |
| a stop before 399 without Traceback | not a result | tell the user; rerun the same seed from scratch |

## What this run cannot answer

* Whether the hectic approach and the large yaw turn are gone: no training
  tag measures yaw excursion or contact speed (`HANDOFF-RL.md` item 2a,
  "Still not measured"). That needs the replay of `model_399.pt`.
* Which of the changes did what: (A) band, (B) steps and the budget 225 ->
  400 change together (D-196 Attribution). A side-by-side with RT-235s1 is
  a reading, not a comparison.
* [VOID 2026-09-26, see P9: there is no tilt pull, the term is off.]
  Whether the weaker per-step tilt pull (P9) matters: no tag isolates it.
* What the LSTM contributes: the MLP ablation arm (same 128 / 400 / 32).
* The test success rate and the force learning effect (`model_50` vs
  `model_399` on the same table, D-195 FORCE): the evaluation side is not
  ready (`run_eval.ps1` has no `--agent`, D-195 seam 6 open).
* Whether the kp bound matters now: `osc_kp_pos` stays 100 against a bound of
  about 211 at the new step (D-166 inline note); the run prints the bound,
  it does not test a second kp.

## ADDENDUM before any result (2026-09-26): P1b checked against RT-235s1

Written before any RT-236s1 number was read. Nothing above changes except
the P1b bar, sharpened here. The gate accepts the handover SHA `ae5fd006`
OR any later commit whose `git diff --stat ae5fd006..<log sha>` touches
`rt_logs/` only (this addendum is such a commit; no new pull is needed).

* RT-235s1 ran the SAME reset path (D-196 touches no reset code, `git diff
  6792b78..8856a4c1` above). Its FIRST startup report is the reset pose at
  DR width 0: `EE world quat` env 0 / env 1 (`RT-235s1.txt:209`, `:255`),
  `peg yaw phi (folded)` -0.001 / -0.000 deg (`:223`, `:269`). From the two
  quaternions, with `insertion_math`'s own functions (scratch
  `p1b_from_rt235.py`): reset yaw vs seat **+0.00135 / +0.00006 deg**.
* The `reset yaw noise: ... on wrist_3_joint` lines (`:230`, `:276`, e.g.
  -0.108 deg) are a JOINT angle, `_joint_targets - home_pose` on wrist_3
  (`insertion_env.py` startup block), not the tool yaw: the start solve moves
  the tip by the fixture offset, base and wrist turn together, and the tool
  yaw stays at the reading above.
* SHARPER BAR. The home is not exactly the seated orientation in TILT
  either (w about 0.0008). With the measured quaternion as home, the worst
  start yaw at the DR ceilings is 5.4508 / 5.4501 deg (same script), not the
  closed form's 5.4386: the band's need is then 5.4508 + 1.28 + 1.0 = 7.7308
  deg <= 7.74 deg, air 0.009 deg. So P1b reads: `max |yaw|` <= **0.009 deg**
  keeps the rule with this home (was 0.021, which ignored the tilt part);
  the RT-235s1 readings sit below it. Between 0.009 and 7.74 deg the reading
  in P1b's table above applies unchanged.
* NEW REPORTED READING (no bar): the LATER startup reports of RT-235s1, taken
  mid-training with a free yaw, read `peg yaw phi (folded)` -0.220 / +9.487
  (`:470`, `:516`), -10.108 / -0.764 (`:778`, `:824`), -22.007 / -5.682 deg
  (`:998`, `:1044`) -- in the pocket frame, folded to +-45 deg. RT-236s1
  prints the same lines; P1 adds them beside these. The band clamps the
  TARGET, so the actual pose may lag or overshoot it; a reading well past
  7.74 deg is reported, not judged.
