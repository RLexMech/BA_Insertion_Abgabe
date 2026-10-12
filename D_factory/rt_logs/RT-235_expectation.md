# RT-235s1 -- expectation, written BEFORE the run (2026-09-26): the D-195 pilot, AutoDR seed 1 with the LSTM, a_(t-1) and the D-194 yaw belief error

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Number: highest over every `origin/*` branch is RT-234 (`git ls-tree -r
--name-only` over `rt_logs`, 2026-09-26). RT-234 was the same pilot at 188
iterations; the user raised the budget by 20 % before any result, so RT-234
is superseded and a started RT-234s1 is stopped and void. A new number,
because `rt_log.ps1` appends to an existing log. Template for everything not
named here: `rt_logs/RT-233_expectation.md` (same clone, same AutoDR arm,
same flags). Only what differs from it is stated below.

## Context

D-195 (`DECISIONS.md`) turns the actor into rsl_rl's recurrent actor-critic
(LSTM 2 x 256 before the MLP), keeps a_(t-1) in the observation
(`force_prevact`, 34 channels), gives the critic its own group from the TRUE
state, and trains everything with D-194 on (the per-episode pocket-yaw
belief error). Rollout 128, 225 iterations, 32 mini-batches (D-195, B1). Budget 225 x
128 = 28,800 steps per env = 1.2 x the old 1500 x 16 (user 2026-09-26,
D-195 BUDGET).
D-195 CONDITIONS: "the first seed of the first arm is its own pilot" -- no
separate smoke run. This is that seed. Its log answers three questions:
does it learn, does the adaptive learning rate collapse, what is the wall
time. What comes out is the result (user 2026-09-26: no time).

## Code state

Clone `Phase5_MitAktion_v1` on the training PC (made for RT-233, recipe in
`rt_logs/RT-233_expectation.md` § Commands). It needs a PULL: it last ran
`823c1be` (RT-233s1 log, `[rt_log] git:`). Branch `p5_mitAktion`, the commit
that adds this file; its SHA is read back from `origin/p5_mitAktion` at
handover and given in chat. Every log's `[rt_log] git:` line must match it
before any point below is judged.

What changed against RT-233's code (all offline-checked, none run under
Isaac):

* D-194 (`e69309d2`, `96ea52bd`, `bab25b28`): the believed pocket frame,
  table column 16 `pocket_obs_yaw`, cfg `pocket_obs_yaw_rad` (default 0.0 =
  off). This run switches it on: `env.pocket_obs_yaw_rad=0.0223402`, the
  value `Belege_Streuwerte.md` B6 names ("der Lauf setzt 0.0223402 rad =
  1,28°"). The 17-wide row changes the RNG stream (HANDOFF item 1 (a)).
* D-195 seam 1 (`834a605d`): `--agent rsl_rl_rnn_cfg_entry_point` ->
  `PPORunnerLstmCfg` (`agents/rsl_rl_ppo_cfg.py`): extends
  `PPORunnerMlpAblationCfg` (`num_steps_per_env = 128`, `max_iterations =
  225` since the budget commit, `algorithm.num_mini_batches = 32`, `obs_groups = {"policy":
  ["policy"], "critic": ["critic"]}`); policy
  `RslRlPpoActorCriticRecurrentCfg`, `rnn_type="lstm"`, `rnn_hidden_dim=256`,
  `rnn_num_layers=2`, MLP fields read from `PPORunnerCfg.policy` ([128, 128],
  elu, both normalisers on, init noise 1.0).
* D-195 seam 2 (`1f6df5f6`): `_get_observations` returns `{"policy": obs,
  "critic": critic}`; the critic's `tip_rel`, `yaw_cos_sin`, `pocket_quat`
  come from `_true_pocket_blocks` (zero yaw error, nominal tip offset); noise
  on `policy` only (Isaac Lab `direct_rl_env.py:412-415`);
  `resolve_obs_layout` writes `state_space = 34`.
* Seams 3/4 (`2ee87f35`, `3e358c98`) touch `play.py` and
  `checkpoint_width.py` only -- not on the training path.

Offline (laptop, numpy stand-in): `check_env_wiring.py` 419 PASS, every new
mutation flips exactly its declared checks (budget: `iterations-back-to-188`) (scratch runner, not the full
sweep); `check_mutation_anchors.py` 568 anchors match; `checkpoint_width.py
--self-test` 19/19. `insertion_math.py` unchanged by D-195.

`CODE_MARKER = "insertion-osc-2026-09-03a"` (`insertion_env.py:76`),
unchanged -- it does NOT tell this code from RT-233's. The gate is the git
line, the `[rt_log] cmd:` line and the network lines (P1).

## The one hypothesis

**The D-195 form trains at this budget: the run reaches iteration 224 with
exit 0 and leaves the AutoDR floor phase (`dr/phase` reaches 1).**

Why this and not "success at the end above success at the start": the
start floor begins 30 mm INSIDE the pocket, so success is high from the
first iteration and dips when the floor rises (RT-233s1, MLP:
`Episode/success_rate` 0.848 at step 0, minimum 0.494 at step 142, 0.982 at
1499). Leaving the floor phase needs the AutoDR buffer rule's success
threshold at the floor boundary (`advance_at = ADVANCE_AT`,
`autodr.py:743`; `ADVANCE_AT = 0.80`, `curriculum.py:53`; the floor is the
eighth key `start_height_floor`, same rule) -- a learning event defined by
existing code, not by a number typed here. RT-233s1 left it at step 67.

Falsified by: a Traceback or a stop before 224; `dr/phase` 0 at every step.

The learning rate and the wall time (D-195's other two pilot questions)
are REPORTED, without a bar. The adaptive rate moves every iteration
between rsl_rl's bounds, 1e-5 and 1e-2 (rsl_rl 3.0.1, the training PC's
version per `results/eval/*/eval_manifest.json` `versions`,
`algorithms/ppo.py:282/284`; `desired_kl = 0.01`, `PPORunnerCfg`) [It read
"rsl_rl 3.1.2 `algorithms/ppo.py:279-282`"; corrected 2026-09-26 at the
start, same bounds]: in
RT-233s1 it sat at 1e-5 at step 0 and in 45 of 1500 steps, and read 2.25e-5
at 1499 -- one value at one step says nothing, so no single-step bar is
set.

No MLP run under D-194 exists, so "learns clearly worse than an MLP" (D-195,
What would make it wrong) is NOT judged here; that needs the ablation arm.

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

The run (ONE seed, the pilot). No `--max_iterations`: the 225 come from the
cfg (`train.py:245-246` keeps the cfg value when the flag is absent):

```powershell
.\scripts\rt_log.ps1 RT-235s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --agent rsl_rl_rnn_cfg_entry_point --num_envs 256 --headless --seed 1 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Wall time: unknown for this setup. RT-206sN (MLP, 16-step rollout, 24,000
steps per env) took 3.57-3.79 h; this run has 1.2 x the steps and an LSTM --
no number follows from that.

## Points -- RT-235s1

* **P1 setup** (the gate; any line wrong = the run is void, nothing below is
  judged -- precedent RT-151b, `rt_logs/VERDICTS.md` 2026-09-05 14:51:11):
  `[rt_log] git:` = handover SHA; `cwd:` ends in `Phase5_MitAktion_v1`;
  `[rt_log] cmd:` carries `--agent rsl_rl_rnn_cfg_entry_point` and
  `env.pocket_obs_yaw_rad=0.0223402`; `[insertion] robot asset:` under
  `Phase5_MitAktion_v1`; `obs mode force_prevact-34: ...`; `observation
  layout (built 34-dim of 34, mode force_prevact, ...): ... | 28:34
  prev_action`; `pocket-yaw belief error env0..` line with `+-0.0223402 rad`
  and values within +-0.0223402, not all 0; THE NETWORK (rsl_rl 3.0.1
  `actor_critic_recurrent.py:70-71, 82-83` prints [It read "3.1.2
  `:72-87`"; corrected 2026-09-26]): `Actor RNN:` and `Critic RNN:`
  both `LSTM(34, 256, num_layers=2)`, `Actor MLP:` and `Critic MLP:` first
  layer `Linear(in_features=256, out_features=128, ...)`; NO `UserWarning`
  line about `obs_groups` (RT-233s1 had two, log lines 285/287 -- their
  absence shows the cfg's `obs_groups` took); `force abort at 30.0 N`;
  `provider: AutoDR, 8 boundaries, bounds_version 0`; the `START FLOOR` line;
  run folder `..._RT-235s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1`.
  Seed from `params/env.yaml` and the sidecar (the log has no `Environment
  seed : N` line, RT-233 interim, `HANDOFF-RL.md`).
* **P2 floor**: `reached H_min, phase dr` line; `dr/phase` 0 -> 1 iteration.
  Reported, no bar (a 128-step iteration is 8 old iterations of env steps;
  no old number is a reference).
* **P3 boundaries**: first iteration with `dr/all_at_max` = 1, and its value
  at 224. Reported, no bar.
* **P4 curves** in order (reward, surrogate, value, entropy) with trend;
  `Policy/mean_noise_std` not below 0.2; `Loss/learning_rate` over the whole
  run: min, max, value at 224, and how many steps sit at 1e-5 in steps
  0..111 and 112..224 (halves, a split, not a bar).
* **P5 success**: `Episode/success_rate` at steps 0 and 224, its minimum
  and where, and the curve;
  `dr/train_success_regular_fresh`; the bin tables of `demo_metrics.json`.
  A last-N-iteration mean is NOT defined here: the old last-50 window
  (steps 1450..1499 of a 16-step run) has no agreed 128-step counterpart yet
  (open, the user decides before RT-235s2).
* **P6 end**: `Learning iteration 224/225` (rsl_rl `on_policy_runner.py`
  log line `Learning iteration {it}/{tot_iter}`), `model_224.pt` (rsl_rl
  saves `model_{it}.pt` every `save_interval` = 50 and at the end:
  `model_0/50/100/150/200/224.pt`), `autodr_224.json` (`train.py`
  `_autodr_sidecar_path`), `[scalars] wrote`, `[rt_log] exit code: 0`;
  `wall_time` at step 224 of `scalars.csv` = the pilot's wall time. Files
  handed back: `scalars.csv`, `demo_metrics.json`, `autodr_224.json`,
  `params/env.yaml`, `params/agent.yaml` (NEW, the agent cfg is new),
  `episodes.csv`, `episodes_meta.json`, the short log, and the output of
  `python scripts/check_episode_log.py --find-run RT-235s1`.
* **P7 physics** and **P8 force and action**: as RT-233 P7/P8, reported, no
  bar.

## Pins (read from the files, not from the log prose)

* `params/agent.yaml`: `num_steps_per_env: 128`, `max_iterations: 225`,
  `algorithm.num_mini_batches: 32`, `policy.class_name:
  ActorCriticRecurrent`, `policy.rnn_type: lstm`, `policy.rnn_hidden_dim:
  256`, `policy.rnn_num_layers: 2`, `policy.actor_hidden_dims: [128, 128]`,
  `obs_groups` policy -> [policy], critic -> [critic]. Source: the cfg
  classes above.
* `params/env.yaml`: `obs_wrench_mode: force_prevact`, `observation_space:
  34`, `state_space: 34` (NEW, `resolve_obs_layout`), `pocket_obs_yaw_rad:
  0.0223402`, `dr_mode: autodr`, `start_floor_m: -0.03`,
  `autodr_stall_buffers: 3`, `seed: 1`. `state_space: 34` is UNVERIFIED (the
  yaml is dumped after `gym.make`, as `observation_space` in RT-233's pins).
* `demo_metrics.json`: `"obs_wrench_mode": "force_prevact"`, `"obs_version":
  "force_prevact-34"`, `"pocket_obs_yaw_rad": 0.0223402`
  (`insertion_env.py:4480`), `"dr_mode": "autodr"`.
* `autodr_224.json`: `"format": "autodr-sidecar-1"`, `"seed": 1`.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| exit 0 at 224, `dr/phase` reaches 1 | the D-195 form trains at this budget | RT-235s2..s5 and the other arms, each its own expectation |
| exit 0 at 224, `dr/phase` 0 throughout | the form does not leave the floor at 225 x 128 | stop; user decides (D-195 reopened) |
| `Loss/learning_rate` at 1e-5 in most of steps 112..224 | reads as a collapsed adaptive rate | report the curve beside the success curve; user decides |
| `obs_groups` UserWarning in the log, or `Actor MLP` first layer `in_features=34` | the MLP cfg or the default groups ran, not D-195's | void; check `--agent` in `[rt_log] cmd:` and HEAD |
| Traceback naming `critic`, `state_space`, `obs_groups` or `ActorCriticRecurrent` | seam 1/2 wiring, untested under Isaac | stop; hand the Traceback back |
| a stop before 224 without Traceback | not a result | tell the user; rerun the same seed from scratch |

## What this run cannot answer

* What the LSTM contributes: that needs the MLP ablation arm (same 128 /
  225 / 32, `--agent rsl_rl_mlp_ablation_cfg_entry_point`), D-195 "Owed to
  the results".
* Whether the memory is used at test time: `play.py
  --memory-reset-every-step` on the evaluation table, a later run.
* Any comparison with RT-206 / RT-233: D-194 and D-195 change together
  (D-195 (3)).
* The test success rate: training success carries PPO's exploration noise;
  the T-final evaluation needs the evaluation side (`run_eval.ps1` has no
  `--agent` yet, the tables are 16 columns wide -- HANDOFF item 1).
* Whether the critic group equals the policy group numerically when both
  belief errors are 0: not measured anywhere.
* How much less force the policy needs at the end than at the start: the
  training force tags mix learning with the curriculum (the start is easy,
  the end is full width). RT-233s1 (MLP) read `Episode/force_max_mean_n`
  12.79 N over its first 50 steps and 12.13 N over its last 50,
  `Episode/force_max_p95_n` 25.39 -> 25.27 N. A clean answer needs an early
  and the final checkpoint on the SAME evaluation table (open, next chat).

## ADDENDUM before any result (2026-09-26)

Written while RT-235s1 runs, before any of its numbers were read. Two
readings are added; nothing above changes. [Correction, ADDENDUM 2: it was
not running yet.]

* **End window** (user, D-195 END WINDOW): the final reading of a run is
  the mean over steps 218..224 of `scalars.csv` (the last 7 iterations).
  P5 adds `Episode/success_rate` over 218..224 beside step 224.
* **Force, start against end** (user 2026-09-26): P8 adds the means of
  `Episode/force_max_mean_n`, `Episode/force_max_p95_n` and
  `Episode/force_abort_rate` over steps 0..6 and 218..224. Reported, no
  bar; the reading carries the curriculum caveat above.

## ADDENDUM 2 before the start (2026-09-26)

RT-235s1 had NOT started at `1b14d923`: the training PC still ran the
RT-233 chain (s4, stopped by the user). It starts at the SHA given in chat,
read back from `origin/p5_mitAktion`; P1's `[rt_log] git:` line is held
against THAT SHA. The training path is the same as at `1b14d923`: under
`source/` only `insertion_env.py` changed (`APPLIED_LOG_COLUMNS`,
`_write_applied_csv`, `_applied_log_meta`, seam 7 (4) `ffa0a2a3`), and
`applied.csv` is written under `dr_mode='table'` only (`check_env_wiring`
"applied.csv is written in table mode only ..."). The rest is evaluation
scripts, tables and docs (`git diff --stat 1b14d923 <SHA>`). Commands and
points above are unchanged.
