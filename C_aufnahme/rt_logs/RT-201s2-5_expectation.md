# RT-201s2..s5 -- Condition 2 (AutoDR), seeds 2-5, written BEFORE the run (2026-09-15)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

## Context

RT-201 (`rt_logs/RT-201_expectation.md`) is seed 1 of Condition 2 (AutoDR,
floor -0.030): per `rt_logs/VERDICTS.md` (2026-09-14 04:27) it reached
`[stop-criterion] target reached at iteration 850`, all seven boundaries at
max (lat_r 588, friction_hi 590, tilt 591, friction_lo 596, yaw_lo 597,
yaw_hi 602, start_height_hi 623 -- `all_at_max` at 623), fresh
0.9955 >= 0.995 bar, checkpoint/sidecar saved at 833. RT-201's own closing
line: "Ein Seed." `Laufplan_Phase5.md`, "Rules for every run", was REPLACED
2026-09-14 (same replacement RT-202s2-5 ran under): every seed of both
conditions stops on its own `--stop-when-dr-max 0.995`, cap 3000; Condition
2 seeds 2-5 run one after another as `RT-201s2..s5` with the RT-201 command
and only `--seed` changed. This file registers those four before the first
one starts.

## Marker- and code-stand

Branch `p5-kraftsensor`. `git log --oneline -5` (laptop, this worktree):

```
3a346934 Log the per-episode force maximum per iteration: force_max_mean_n, force_max_p95_n, force_max_max_n
a49d8c02 Record RT-205: 1500 fixed iterations cut the insertion force by about a quarter against RT-201's stop
9c960ffd Belege_Streuwerte D4: the height bins are ten (-20..120 mm) since the H_min commit, not six
aead0e1e Keep the RT-201/202/203 demo_metrics and AutoDR sidecars under docs/figures
6f327cdd Rewrite KONZEPTPHASE.md as the consolidated concept as implemented; archive the 2026-08 block plan
```

`origin/p5-kraftsensor` = `3a346934` (short `3a34693`) at file-write time.
Verify before handing over the command:

```powershell
git -C C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1 rev-parse --short HEAD
```

If it does not read `3a34693`, pull and re-verify:

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
git pull
git rev-parse --short HEAD
```

Expected SHA after pull: `3a34693`. Each RT-201sN log's `[rt_log] git:` line
must match this SHA before any point below is judged -- a mismatch
invalidates the run (RT-151b precedent, `rt_logs/VERDICTS.md`,
2026-09-05 14:51:11).

The only code change reaching this run since RT-201 (`d316f72`) is
`3a346934`: three LOG-ONLY per-iteration curves added at
`source/insertion/insertion/tasks/direct/insertion/insertion_env.py:3342-3350`
(comment "Force instrument" region above line 3342), merged into the same
`self.extras["log"]` dict RT-201 already read from --
`force_max_mean_n`, `force_max_p95_n`, `force_max_max_n` (mean / p95 / last
element of the sorted `_recent_force_norm` buffer). This is a difference to
seed 1 (RT-201's own scalars.csv has no force curve at all), not a change
to physics, reward, or RNG -- nothing else changed between `d316f72` and
`3a346934` per the commit list above (the two commits in between,
`a49d8c02` and `9c960ffd`, touch `docs/` only).

`SCRIPT_MARKER` / `CODE_MARKER`: unchanged from RT-201/RT-202s2-5 --
`--stop-when-dr-max` under `dr_mode=autodr` plus start floor,
`env.autodr_stall_buffers=3` (`autodr-3` sidecar, hold/STALL lines),
`[autodr-state]` on resume.

## The one hypothesis

**Seeds 2-5 also reach `[stop-criterion] target reached` with all seven
AutoDR boundaries at max before the 3000-iteration cap, each at its own
stop iteration -- and RT-201's 850 does NOT predict any of them.**

RT-205 (`rt_logs/VERDICTS.md`, 2026-09-14 17:35, BEFUND 0) ran the identical
seed and command as RT-201 except the stop flag, and its curves already
diverge from RT-201 from iteration 49 (Reward 313.07 vs. 338.72) -- same
seed, same command, different trajectory. Runs are therefore not
bit-reproducible on this stack; RT-201's stop iteration (850) is the outcome
of one seed, not a pass bar or a prediction for seeds 2-5.

## Command

Run these four, one after another (training PC root
`C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`),
verifying the SHA once before the first and re-checking `[rt_log] git:`
per log:

```powershell
.\scripts\rt_log.ps1 RT-201s2 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 2 --max_iterations 3000 --stop-check-every 50 --stop-when-dr-max 0.995 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-201s3 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 3 --max_iterations 3000 --stop-check-every 50 --stop-when-dr-max 0.995 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-201s4 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 4 --max_iterations 3000 --stop-check-every 50 --stop-when-dr-max 0.995 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-201s5 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 5 --max_iterations 3000 --stop-check-every 50 --stop-when-dr-max 0.995 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

The user starts seed 2 now; seeds 3-5 later, one job at a time (training PC
runs one job at a time).

## Points, per seed (RT-201's form)

Applied identically to RT-201s2, RT-201s3, RT-201s4, RT-201s5.

* **P1 setup**: `git:` = `3a34693`; `START FLOOR` line; `provider: AutoDR,
  8 boundaries`; band table start_height [-0.030, +0.020], others [c, c];
  run folder `..._RT-201sN_autodr_..._start+20mm_floor-30mm_seedN` (seed
  number in place of RT-201's seed 1).
* **P2 the floor moves**: `[autodr] start_height_floor: rate ... -> expand`
  lines, the last carrying `-- reached H_min, phase dr`; `dr/phase` 0 -> 1;
  record each seed's own transition iteration -- RT-201 needed 55, this is
  a per-seed result, not a bar.
* **P3 the seven open**: after the transition `dr/lat_r_hi`, `dr/yaw_lo/hi`,
  `dr/tilt_hi`, `dr/start_height_hi`, `dr/friction_lo/hi` leave their
  centres; `[autodr]` expand lines for them; `dr/flags_dropped` recorded,
  not predicted (RT-201: 31).
* **P4 the six curves** in the fixed order (reward, surrogate, value,
  entropy; EV/KL not logged), with trends; `Policy/mean_noise_std` not
  below 0.2.
* **P5 success**: `dr/train_success_regular_fresh` >= 0.995 over 2000 in
  the fresh window of the final `bounds_version`; `Episode/success_rate`;
  `success_by_start_height_bin` (10 bins) and the lateral/yaw/tilt bins once
  those boundaries open.
* **P6 stop**: `[stop-criterion] target reached` with `all boundaries at
  max True`; `autodr_<it>.json` format `autodr-3`, 8 keys.
* **P7 physics**: `force_abort_rate`, `max_depth_max_mm` (> 37 = the
  under-fixture path, RT-202 finding, FINDING), interpen.
* **P8 stagnation**: hold lines and `dr/*_holds`; a `STALL` note is a
  review trigger, read `_last_rate` and the height/lateral/yaw/tilt bins
  before any change.
* **P9 force instrument (new against RT-201, `3a346934`)**: `scalars.csv`
  carries three additional per-iteration tags, `Episode/force_max_mean_n`,
  `Episode/force_max_p95_n`, `Episode/force_max_max_n`, and the console
  block for each logged iteration prints them alongside the existing
  `Episode/*` lines. LOG-ONLY per the commit and the code read above --
  their presence/absence is the point, not a value prediction (RT-201 has
  no such column to compare against).

## Files to hand back per seed

Same list and provenance as RT-202s2-5 (`rt_logs/RT-202s2-5_expectation.md`,
verified there against `train.py`, `insertion_env.py`, `rt_log.ps1`):

* **`params/env.yaml`**, **`params/agent.yaml`** (`train.py:680-681`) --
  config provenance: `dr_mode=autodr`, `start_floor_m`,
  `autodr_stall_buffers`, seed.
* **`scalars.csv`** (`train.py:851-852`) -- the six curves (P4), AutoDR
  boundary/hold tags, success/depth/force curves, and now the three P9
  force tags.
* **`autodr_<it>.json`** (`_autodr_sidecar_path`, `train.py:183-200`,
  written alongside each checkpoint, `train.py:585-595`) -- AutoDR end
  state (`format`, `bounds_version`, boundaries, `phase`) for P1/P2/P3/P6/P8
  and any resume.
* **`demo_metrics.json`** (`_write_metrics`, `insertion_env.py:3759`, path
  at `:3769`, called from `:3416`) -- `success_by_*_bin` (P5),
  `force_abort_rate`, `max_depth_max_mm`, interpenetration, force/torque
  percentiles (P7).
* **the final `model_<it>.pt`** -- checkpoint at the stop iteration or the
  cap.
* **`RT-201sN.short.txt`** -- full log at `rt_logs/RT-201sN.txt`, shortened
  via `shorten_rt_log.py`, both copied into the run folder
  (`rt_log.ps1:135-150`).

On the laptop these land at `rt_logs/RT-201/RT-201sN_<name>` (one subfolder
per seed) for `params/env.yaml`, `params/agent.yaml`, `autodr_<it>.json`,
`demo_metrics.json`, `model_<it>.pt`, `RT-201sN.short.txt`; `scalars.csv`
additionally goes to `docs/figures/RT-201sN_scalars.csv`.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| target reached at all four seeds | the floor+7-boundary machine is not a seed-1 fluke; the four stop iterations (beside RT-201's 850) are the spread the thesis reports for Condition 2 | close Condition 2, compare its spread against Condition 1 (RT-202s2-5) |
| a seed hits the 3000 cap without `all boundaries at max True` | that seed did not reach the target inside budget; read its `_fill`, `_last_rate` and bin tables before ranking it against the other three | per-boundary diagnosis before any change |
| a `STALL` note on any seed | differential diagnosis point 8 (RT-201's P8/P9), a review trigger, not itself a verdict | read `_last_rate` and the relevant bin table for that seed before any change |
| `max_depth_max_mm` > 37 on any seed | FINDING per P7 (RT-202 already showed this path under No-DR), independent of the target-reached question | read interpenetration alongside it |
| `Policy/mean_noise_std` < 0.2 on any seed | differential diagnosis point 5 (exploration), a review trigger for that seed | -- |
| force curves (P9) present but trend differs materially seed to seed | descriptive only -- report mean/p95/max across the run per seed, no claim about cause without a matched-iteration comparison (RT-205's own caveat: trajectory divergence from it 49 means raw-iteration comparison across seeds needs the same care) | -- |

## Was dieser Lauf nicht beantworten kann

* Ob RT-201 (850) und RT-201s2..s5 aus derselben Verteilung stammen, ist
  eine statistische Frage, die dieser Lauf allein nicht beantwortet (fuenf
  Punkte, keine Signifikanzaussage) -- und RT-205 zeigt, dass selbst
  identischer Seed + Befehl keine identische Trajektorie liefert.
* Der Vergleich Condition 1 (RT-202s2-5) gegen Condition 2 (dieser Lauf)
  ist eine Auswertung NACH allen acht Laeufen, nicht Teil dieser Erwartung
  pro Seed.
* Nichts zur Wrench-Beobachtung (RT-195/196/203) -- dieser Lauf bleibt auf
  force-obs, AutoDR, Boden.
* Die 1000-Episoden-Eval-Tabelle (Laufplan Schritt 8) ist NICHT gebaut;
  dieser Lauf liefert nur die Checkpoints, die sie spaeter als Input
  braucht.
* Ob das Kraft-Sinken aus RT-205 (laengeres Training senkt die
  Einfuehrkraft) sich in den vier neuen Seeds wiederholt, ist eine Frage
  fuer die Checkpoint-Abspielreihe, nicht fuer die neuen P9-Kurven allein
  (die messen nur den laufenden Trainings-Verlauf, keine feste
  Eval-Episodenmenge).
