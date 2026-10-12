# RT-233s1..s5 -- AutoDR with the previous action in the observation (D-193 falsification), written BEFORE the runs (2026-09-26)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Number: highest over every `origin/*` branch is RT-232 (`git ls-tree -r
--name-only` over `rt_logs`, 2026-09-25). Written in the main chat with the
`rt-expectation` skill. Template for everything not named here:
`rt_logs/RT-231_expectation.md` (five seeds in one chain, watcher) and
`rt_logs/RT-206_expectation.md` (the AutoDR points).

## Context

D-193 (branch `p5-kraftsensor`, `DECISIONS.md`) keeps the previous action
a_(t-1) OUT of the policy observation and names this run as the one that
would falsify it (user 2026-09-25: five seeds, not one). The literature check
(`docs/reference/literature_check_prev_action_in_observation_2026-09-25.md`
on `p5-kraftsensor`) leaves two reasons open why a_(t-1) could matter here:
the action-rate reward term reads it (D-193 (3), (8)), and FORGE adds it to
infer unobserved dynamics (Sec. II-B) -- AutoDR randomises friction, which
the actor does not see (D-193 (7)(ii)). This run changes ONE thing against
RT-206s1..s5: `env.obs_wrench_mode=force_prevact`.

## Code state

**NEW CLONE on the training PC: `Phase5_MitAktion_v1`**, its own folder beside
`Phase5_Robustheit_v1`, which stays as it is. Branch `p5_mitAktion`, the
commit that adds this file. Its SHA is read back from `origin/p5_mitAktion`
at handover and given in chat. Every log's `[rt_log] git:` line must match it
before any point below is judged.

Code against the evaluated AutoDR arm: `p5_mitAktion` = `e5ee168e`
(`p5-kraftsensor`, D-193 entry) + `83f6033c` (the new mode). `git diff
5a6cc8e e5ee168e -- source scripts/rsl_rl/train.py` (2026-09-25): the
evaluation table and Fixed-DR, every new line gated on `dr_mode ==
'table'` / `'fixed_dr'` or `self._eval_table` -- the autodr training path is
unchanged by that reading (read, not run; RT-231 names the same reading).
`83f6033c`: a third observation mode, `force_prevact`, default stays
`force`:

* `insertion_math.py`: `OBS_MODES = ("force", "wrench", "force_prevact")`,
  `PREV_ACTION_WIDTH = 6`, `obs_slices("force_prevact")` = the 28-wide table
  plus `prev_action` 28:34, `assemble_observation_prevact`.
* `insertion_env.py` `_get_observations`: under `force_prevact` the cat is the
  force cat plus `prev_action = self._prev_actions` -- the raw previous
  action the action-rate reward compares against (`_pre_physics_step`),
  zeroed by `_reset_idx`. No noise on it (`noise_masks` puts sigma on
  `force` and `tip_rel` only).

Offline (laptop, numpy stand-in, `83f6033c`): `check_env_wiring.py` 393 PASS;
the 10 new or touched wiring mutations and both new math mutations flip
exactly their declared checks; `check_mutation_anchors.py` 533 anchors match.
`check_insertion_math.py` carries 9 red cfg-guard checks that are ALREADY
red on `p5-kraftsensor` `e5ee168e` (not touched here), so its full
`--self-test` refuses to start; the new math checks are green.

`CODE_MARKER = "insertion-osc-2026-09-03a"` (`insertion_env.py:76`),
unchanged -- the marker does NOT tell this code from RT-206's. The gate is
the git line, the asset path (the editable install) and the layout line
(P1).

## The one hypothesis

**Appending a_(t-1) to the observation does not change the final training
success: the Welch 95 % interval of the difference between the five
RT-233sN and the five RT-206sN last-50 means of `Episode/success_rate`
contains 0.**

Reference, read from `rt_logs/RT-206sN/RT-206sN_scalars.csv` on the laptop
(2026-09-25; mean of `Episode/success_rate` over `step` 1450..1499, n = 50
each): s1 0.9706, s2 0.9599, s3 0.9594, s4 0.9745, s5 0.9384 -- mean
0.9606, sd 0.0140.

Last-50 mean: the mean of `Episode/success_rate` over the rows with `step`
1450..1499 of `scalars.csv`. Welch: the t-interval with Welch-Satterthwaite
degrees of freedom, 95 %, on the five means per arm.

Falsified by: the interval lies wholly above 0 or wholly below 0.

Resolution: with sd 0.014 per arm and 5 seeds, the interval half-width is
roughly 2 percentage points; a smaller effect is not visible here. A
contained 0 shows "no difference detected", NOT equality (Colas et al.
2018; `InBachelorErwähnen.md` 2026-09-23, K2 row).

## Commands

Training PC, PowerShell, conda env `env_isaaclab` active. Clone recipe of
the p6 clone (`origin/p6-TestAusserhalb:HANDOFF-RL.md`, "THE
`p6-TestAusserhalb` CLONE", tested 2026-09-23), with the branch and folder
changed, no checkpoint copy, and no `checkout -B <sha>` line: the clone
takes the branch tip, which IS the commit that adds this file. The expected
SHA is given in chat:

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion
git clone --filter=blob:none --sparse -b p5_mitAktion https://github.com/RLexMech/Ur5e_ContactRich_Insertion.git Phase5_MitAktion_v1
cd Phase5_MitAktion_v1
git sparse-checkout set --no-cone "/.claude/**" "/rt_logs/**" "/scripts/**" "/source/**" "!**/CLAUDE.md"
git rev-parse HEAD
git status --porcelain --untracked-files=no
Copy-Item -Recurse ..\Phase5_Robustheit_v1\source\insertion\insertion\tasks\direct\insertion\assets source\insertion\insertion\tasks\direct\insertion\assets
Test-Path source\insertion\insertion\tasks\direct\insertion\assets\Werkzeug
Test-Path source\insertion\insertion\tasks\direct\insertion\assets\Robot\ur5e_tool.usd
python -m pip install -e source\insertion
pip show insertion
```

Expected: HEAD = the SHA from chat; `git status` no output; both `Test-Path`
`True`; `Editable project location` ends in
`Phase5_MitAktion_v1\source\insertion`. THE INSTALL SWITCH takes the editable
`insertion` away from `Phase5_Robustheit_v1`; back = the same `pip install -e
source\insertion` inside that folder.

Watcher in a SECOND PowerShell window, on for the whole series (as RT-231):

```powershell
while ($true) { Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('hub.exe','omni.telemetry.transmitter.exe') } | Where-Object { -not (Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue) } | ForEach-Object { "$(Get-Date -Format HH:mm:ss) stop $($_.Name) $($_.ProcessId)"; Stop-Process -Id $_.ProcessId }; Start-Sleep 15 }
```

The five seeds, one after another (RT-206s1..s5 with ONLY
`env.obs_wrench_mode=force_prevact` added):

```powershell
.\scripts\rt_log.ps1 RT-233s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact; .\scripts\rt_log.ps1 RT-233s2 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 2 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact; .\scripts\rt_log.ps1 RT-233s3 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 3 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact; .\scripts\rt_log.ps1 RT-233s4 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 4 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact; .\scripts\rt_log.ps1 RT-233s5 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 5 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact
```

No smoke run: RT-233s1 is its own smoke -- look at its first minutes (P1)
and stop the chain on a Traceback. Wall time: RT-206s1..s5 took 3.57-3.79 h
each (`wall_time` in their `scalars.csv`), so about 18-19 h for the chain --
a hint from the old run, not a promise. No `--stop-when-dr-max`: fixed
budget, as RT-206.

## Points -- per seed RT-233s1..s5

* **P1 setup** (the gate; any line wrong = the run is void, nothing below is
  judged -- precedent RT-151b, `rt_logs/VERDICTS.md` 2026-09-05 14:51:11):
  `[rt_log] git:` = handover SHA; `cwd:` ends in `Phase5_MitAktion_v1`;
  `[insertion] robot asset:` path under `Phase5_MitAktion_v1` (the install
  moved); `obs mode force_prevact-34: torque read and logged, NOT in the
  policy observation`; `observation layout (built 34-dim of 34, mode
  force_prevact, D-107/D-114/D-188): ... | 25:28 force | 28:34 prev_action`;
  `Actor MLP` AND critic first layer `Linear(in_features=34, ...)`;
  `Environment seed : N`; `force abort at 30.0 N`; `provider: AutoDR, 8
  boundaries, bounds_version 0`; the `START FLOOR` line; run folder
  `..._RT-233sN_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seedN`.
* **P2 floor**: `reached H_min, phase dr` line; `dr/phase` 0 -> 1 iteration,
  reported beside RT-206sN's.
* **P3 boundaries**: first iteration with `dr/all_at_max` = 1, and its value
  at 1499 -- REPORTED beside RT-206sN (first `all_at_max` 1 at 732 / 900 /
  810 / 768 / 936, `RT-206sN_scalars.csv`), no bar.
* **P4 six curves** in order (reward, surrogate, value, entropy; EV/KL not
  logged) with trend; `Policy/mean_noise_std` not below 0.2.
* **P5 success** (the hypothesis): `Episode/success_rate` at 1499 and its
  last-50 mean; `dr/train_success_regular_fresh`; the bin tables of
  `demo_metrics.json`. After s5: the Welch interval as defined above.
* **P6 end**: `Learning iteration 1499/1500`, `model_1499.pt`,
  `[scalars] wrote`, `[rt_log] exit code: 0`; files handed back:
  `scalars.csv`, `demo_metrics.json`, `autodr_1499.json`, `params/env.yaml`,
  `episodes.csv`, `episodes_meta.json`, the short log, and the output of
  `python scripts/check_episode_log.py --find-run RT-233sN` (VERDICT
  COMPLETE).
* **P7 physics**: `below_fixture_rate`, `Episode/max_depth_max_mm` <= 46 mm,
  `Episode_Reward/engaged` window means, `interpen_max_mm` p50/p95/p99/max
  -- as RT-206 P7.
* **P8 force and action**: `force_abort_rate` curve and final value,
  `Episode/force_max_p95_n` curve; `Episode_Reward/action_rate` window mean
  beside RT-206sN (the term a_(t-1) should let the policy lower, D-193 (3)),
  `Episode/action_abs_max_mean`.

## Pins (read from the files, not from the log prose)

* `demo_metrics.json`: `"obs_wrench_mode": "force_prevact"`,
  `"obs_version": "force_prevact-34"` (RT-206sN: `"force"`, `"force-28"`);
  `"dr_mode": "autodr"`.
* `params/env.yaml`: `obs_wrench_mode: force_prevact`, `dr_mode: autodr`,
  `start_floor_m: -0.03`, `autodr_stall_buffers: 3`. Against
  `rt_logs/RT-206sN/RT-206sN_env.yaml` the diff is ONLY `log_dir`,
  `obs_wrench_mode` and `observation_space`. `observation_space` is expected
  34 (the yaml is dumped after `gym.make`, `train.py:537` vs `:684`, and
  `resolve_obs_layout` writes it); no non-force run left an `env.yaml` in
  `rt_logs`, so 34 there is UNVERIFIED -- if it reads 28, report it; the
  gate is the log's `in_features=34`.
* `autodr_1499.json`: `"format": "autodr-sidecar-1"`, `"seed": N`, provider
  format as RT-206sN's file.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| all five seeds 1500 it, exit 0, Welch interval contains 0 | no training-side effect of a_(t-1) detected (not equality); D-193 stands on the training side | T-final evaluation of the five policies (own RT) before any thesis sentence |
| interval wholly above 0 | a_(t-1) raises training success: D-193's cost bound (3) does not hold | T-final evaluation; D-193 reopened, user decides |
| interval wholly below 0 | a_(t-1) lowers training success | T-final evaluation; report |
| Traceback at env build naming `obs_wrench_mode` or a width | the old install or the old code ran | stop the chain; check `pip show insertion` and HEAD |
| a run stops before 1499 without Traceback | not a result | tell the user; rerun the same seed from scratch |

## What these runs cannot answer

* D-193's own falsifier -- that is the T-final success rate
  (`eval_plan.md`), which needs the evaluation of the five policies on the
  seed-10 table (own RT, after this one).
* Which of the two mechanisms (the action-rate term, or inferring hidden
  friction) causes a difference, if one shows -- this run changes the
  observation once and serves both.
* Whether a memory (LSTM, observation stack) would do more than one step of
  a_(t-1) -- not tested.
* Effects below about 2 percentage points (see Resolution).
* Whether the training success rate is a test success rate -- it carries
  PPO's exploration noise.
