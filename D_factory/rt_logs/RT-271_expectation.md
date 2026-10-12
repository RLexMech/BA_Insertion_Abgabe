# RT-271 -- expectation, written BEFORE the run (2026-10-09): replay trace of Auto-DR s2 and s4 on four T-ood friction_hi rows (`rl_games/play.py --trace-obs`, 2 runs)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Template: `rt_logs/RT-270_expectation.md` (same clone, same checkpoints, same
overrides). Only what differs is written out. A VIEW with a trace, not an
evaluation: no manifest, no `eval_collect`, no number for the thesis tables.

## Context

T-ood block `friction_hi` (friction 0.23 = table value 1.25, x1.5 of the
Prüfbereich edge 0.20; sidecar `scripts/eval/eval_table_ood_seed10_c18.json`
block 6): Auto-DR s2 87/150, s4 138/150; failures end at the time limit
(`InBachelorErwähnen.md` row 2026-10-09). User view 2026-10-09 (s2, window,
same four rows): the part sits almost fully in the pocket, tilted against the
walls; "a small wiggle would seat it" (`HANDOFF-RL.md` § Live gaps, first
item). The eval record holds no per-episode depth (`episodes_meta.json`
columns), so "almost seated" is not measured. This run measures it.

Rows 906, 907, 908, 910 of `eval_table_ood_seed10_c18.csv` (sha `02adce21...`):
all four timeouts for s2 in T-ood; for s4 906/908/910 success (26/43/128 steps),
907 timeout (`results/eval_rlgames/episodes_all.csv`, block T-ood, `row_id`).
Start heights 62 / 244 / 22 / 99 mm (same file, `start_height_m`).

## Marker- und Code-Stand

Clone `Phase5_RlGames_v1`, branch `p5-rlgames`. RT-270 runs on `2054d682`.
**A pull IS needed, and only AFTER RT-270 has finished** (a pull inside the
block gives its later runs another `code_commit`; `eval_collect.py` refuses
mixed `code_commit`). `git diff --stat 2054d682..00d01d49`:
`scripts/rl_games/play.py` (+117/-2, the trace), `scripts/check_env_wiring.py`
(1 line, the refusal pin), `HANDOFF-RL.md`, `scripts/CLAUDE.md`. Env unchanged.
Expected HEAD after the pull: the commit that adds THIS file (SHA given in
chat). `CODE_MARKER` = `insertion-osc-2026-09-03a` (`insertion_env.py:76`).
`play.py` has no `SCRIPT_MARKER`; its gate is the `[trace] wrote` line, which
only the new code prints.

Gates per run (each from the rt_log):
* `[rt_log] git: <first 7 of HEAD>`
* `[rl_games] insertion package imported from: C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1\source\insertion\insertion`
* `[rl_games-play] view table: ...ood_friction_hi_s2.csv, 4 rows = envs, width target (NOT an evaluation)` (`play.py`, `--dr-table` branch)
* `[eval-table] ... rows 4 ... Values outside [0, 1] allowed (OOD table): True.` (`insertion_env.py:5113-5118`)
* `[trace] wrote 300 steps x 4 envs to ...RT-271_s<N>_trace.json`
* `[rt_log] exit code: 0`, no Traceback.

## The one hypothesis

**In every first-episode timeout of the two traces, the true depth at the
last step of that episode lies between the engaged depth 10.8 mm and the
success depth 33 mm.**

Depth = `-tip_true_m[step][env][2]` (pocket frame, `insertion_math.py`
`part_tip_pose`: `depth = -tip_rel[:, 2]`). Engaged depth `engaged_depth_m`
0.0108, success depth `fixed_asset.success_depth` 0.033 (both from
`results/eval_rlgames/T-ood/autodr_s2/eval_manifest.json` `env_cfg`, same env).
FALSE if any first-episode timeout ends shallower than 10.8 mm (the part is
not in the pocket: not jammed in it).

## Command (training PC, `env_isaaclab` active, AFTER RT-270)

### Step 1 -- pull and verify

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
git pull
git rev-parse HEAD
git status --short --untracked-files=no
```

Expected: the SHA given in chat; status prints nothing.

### Step 2 -- the view table (four rows; idempotent)

```powershell
New-Item -ItemType Directory -Force C:\Users\Simon\Desktop\Alexander_Pett\view\RT-242s2_ep100 | Out-Null
(Get-FileHash scripts\eval\eval_table_ood_seed10_c18.csv).Hash.Substring(0,8)
$l = Get-Content scripts\eval\eval_table_ood_seed10_c18.csv; $l[0], $l[907], $l[908], $l[909], $l[911] | Set-Content -Encoding ascii C:\Users\Simon\Desktop\Alexander_Pett\view\RT-242s2_ep100\ood_friction_hi_s2.csv
Import-Csv C:\Users\Simon\Desktop\Alexander_Pett\view\RT-242s2_ep100\ood_friction_hi_s2.csv | Select-Object friction
```

Expected: `02ADCE21`, then four lines `1.25`. (`$l[k+1]` = data row k; rows
906/907/908/910.)

### Step 3 -- the two traces (headless, original checkpoints)

```powershell
.\scripts\rt_log.ps1 RT-271_s2 python -u scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 100 --checkpoint C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1\logs\rl_games\ur5e_insertion\09-28_13-01-32__RT-242s2_autodr_offset50mm_currOFF_start+20mm_floor-30mm_seed2\nn\last_ur5e_insertion_ep_100_rew_559.6284.pth --dr-table C:\Users\Simon\Desktop\Alexander_Pett\view\RT-242s2_ep100\ood_friction_hi_s2.csv --trace-obs rt_logs/RT-271_s2_trace.json --trace-steps 300 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402 env.dr_table_allow_outside=true
```

```powershell
.\scripts\rt_log.ps1 RT-271_s4 python -u scripts/rl_games/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 100 --checkpoint C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1\logs\rl_games\ur5e_insertion\09-28_17-38-01__RT-242s4_autodr_offset50mm_currOFF_start+20mm_floor-30mm_seed4\nn\last_ur5e_insertion_ep_100_rew_563.57416.pth --dr-table C:\Users\Simon\Desktop\Alexander_Pett\view\RT-242s2_ep100\ood_friction_hi_s2.csv --trace-obs rt_logs/RT-271_s4_trace.json --trace-steps 300 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402 env.dr_table_allow_outside=true
```

Checkpoint paths = `checkpoint_path` of
`results/eval_rlgames/T-ood/autodr_s{2,4}/eval_manifest.json`. 300 steps
cover the first episode (cap 256 steps, `episode_cap_steps`); env i replays
row i in every episode (`insertion_env.py:2975-2980`).

## Post-run: one zip (training PC)

```powershell
New-Item -ItemType Directory -Force C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-271 | Out-Null
Copy-Item rt_logs\RT-271_s2*, rt_logs\RT-271_s4* C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-271
(Get-ChildItem C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-271).Count
Compress-Archive -Path C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-271\* -DestinationPath C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-271.zip -Force
Get-Item C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-271.zip
```

Expected: count `6` (per run `.txt`, `.short.txt`, `_trace.json`).

## PASS/FAIL points

No check script has a `judge_run` for a trace. The points are the gates
above, then on the laptop (`rt_logs/RT-271/`):

1. `python scripts/summarize_trace.py rt_logs/RT-271/RT-271_s2_trace.json`
   (and `_s4`) runs without error and lists 4 envs. FAIL = the trace format
   differs from rsl_rl's; nothing below is read.
2. `python scripts/trace_outcome_split.py rt_logs/RT-271/RT-271_s2_trace.json`
   (and `_s4`) splits the first episodes into success / timeout and prints the
   `truth` block. FAIL = no `truth` block (the env buffers were not found).
3. The hypothesis read: per first-episode timeout, depth at its last step
   from `truth.tip_true_m` (reader for this one number: to be written on the
   laptop, offline, stdlib).

## Pins (from the trace JSON, each of the 2)

| field | expected | from |
|---|---|---|
| `num_envs` | 4 | the view table's rows |
| `obs_wrench_mode` / `obs_version` | `force_prevact` / `force_prevact-34` | override; `insertion_math.obs_version` |
| `step_dt_s` | 0.0666... | `eval_manifest.json` `step_dt_s` |
| `force_obs_noise_std_n` / `obs_noise_pocket_pos_std_m` | 3.5 / 0.005 | `eval_manifest.json` `noise` |
| `grasp_obs_offset_x_m` | 0.003 | same |
| `row0_is_unnoised` | `true` | `play.py` (DirectRLEnv applies noise in `step()` only) |
| `len(done)` | 300 | `--trace-steps` |
| `truth` present | yes | `play.py` guard `_peg_body_idx` + `_force_tared_raw` |

## Pre-registered readings

| Outcome | Reading |
|---|---|
| every timeout ends at 10.8-33 mm depth | the part is in the pocket and stops short of the seat: jamming, as the view showed |
| a timeout ends < 10.8 mm | that case is not jammed in the pocket (stuck at the entrance or beside it) |
| a row's outcome differs from T-ood (`episodes_all.csv`) | expected possible: the noise draws differ with 4 envs (see below); the trace still shows that episode's motion |
| s4 success: tilt/depth shows a back-off or re-tilt before seating | described, not judged (no criterion was set before the run) |
| a gate / Traceback / missing `[trace] wrote` | not a result |

## What this run cannot answer

* Whether a trace episode IS the T-ood episode of that row: the table fixes
  pose and friction, but the observation noise (pocket position, force) comes
  from the run RNG; with 4 envs instead of 1050 the draws differ.
* Why s2 learned no recovery move: two policies, four rows, no training data.
* Any rate: 4 rows per policy; the rates stay those of T-ood.
* The contact geometry (which edge touches which wall): only the SAPU
  interpenetration maximum is logged, no contact points.
* Anything on a real robot.

## OUTCOME

JUDGED 2026-10-09 (laptop, files `rt_logs/RT-271/`). **Hypothesis FALSE.**

Gates: both runs `[rt_log] git: 13072c3`, `imported from ...Phase5_RlGames_v1\
source\insertion\insertion`, `view table ... 4 rows`, `[eval-table] ... OOD
table): True`, `[trace] wrote 300 steps x 4 envs`, `exit code: 0`. The s4 log
also holds a first attempt on the OLD clone (`git: 8be1e99`, `play.py: error:
unrecognized arguments: --trace-obs ...`, exit 2) before the pull; not used.
Points 1 and 2 PASS (`summarize_trace.py`, `trace_outcome_split.py` read both
traces, 4 envs, `truth` block present).

First episodes (`summarize_trace.py` done steps, `trace_outcome_split.py`):

| row | s2 | s4 |
|---|---|---|
| 906 | timeout (255) | success (25) |
| 907 | success (55) -- T-ood: timeout | success (57) -- T-ood: timeout |
| 908 | timeout (255) | success (199) -- T-ood: 43 |
| 910 | timeout (255) | success (35) -- T-ood: 128 |

Point 3, the hypothesis: true tip at the end of the three s2 timeouts
(`truth.tip_true_m`, `trace_outcome_split` "tip end" z): -6.940 / -3.419 /
-5.316 mm, i.e. depth 6.9 / 3.4 / 5.3 mm -- all BELOW the engaged depth
10.8 mm. Pre-registered reading: not jammed in the pocket; the part is stuck
at the entrance. What the trace shows there (described, not judged): tilted
3.6-7.8 deg beyond yaw at the end, lateral 2.8-3.4 mm, `parked_in_contact`
3 of 3 (z travel 0.05-0.17 mm over the last 20 steps), true force 10.1-11.6 N
(mean, steps 20..end), SAPU interpenetration > 0 in 0.27-0.88 of the steps,
OSC target 18.6-20.0 mm below the tool body (z of `err mean`; the step limit
is 0.02 m, `osc_pos_step_limit_m`). s4 row 908, its one long success (199
steps): OSC target 9.2 mm below the body (`err mean` z), true force mean
4.2 N.

Correction to the user view of 2026-10-09: "almost fully in the pocket" is not
what the truth says; the part entered 3-7 mm of the 36 mm pocket
(`depth_max` 0.036, `eval_manifest.json`). Not changed: tilted, stationary,
in contact, below the abort force.

Counter-test asked by the user (low friction, rest drawn), from existing data
(`results/eval_rlgames/episodes_all.csv`, no new run): T-ood friction_lo
(0.05) s2 149/150, s4 148/150; T-final friction <= 0.10 s2 160/161, s4
161/161. s2 is as good as the others at low friction.
