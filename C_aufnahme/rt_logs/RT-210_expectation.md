# RT-210 — solver stability probe for thesis chapter 4.4, written BEFORE the run (2026-09-22)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

## Context

Chapter 4.4 needs one statement: is the solver setting the code actually
trains and evaluates with — PhysX TGS, `dt = 1/120`, OSC decimation 8,
robot articulation 16 position / 1 velocity iterations
(`source/insertion/insertion/tasks/direct/insertion/ur5e_cfg.py:182-183`,
read: `solver_position_iteration_count=16` at line 182,
`solver_velocity_iteration_count=1` at line 183) — numerically stable on
hard contact. `rt_logs/RT-186_expectation.md` designed a 16-vs-192 probe on
2026-09-12 but it was never run (parked twice, see its own header); the
D-115 question ("16 vs Isaac Lab Factory's 192 position iterations") is
still open in `docs/decisions_inbox.md`. RT-209 (2026-09-22, PASS per
`rt_logs/VERDICTS.md` 2026-09-22 11:56:42) already ran RT-206s1 through
1000 hard-contact rows at 16/1 and is reused here as the 16/1 arm and as
the noise floor. **This is a solver stability check, not a parameter
study; the 16/1 setting stays unless it clearly fails, and the user decides
that — this file proposes no change.**

## Vehicle, and why not something cheaper

* `scripts/check_scripted_insert.py` is UNFIT: it is an offline frame-math
  check of `scripted_policy.py` with no physics (`scripts/CLAUDE.md`:
  "offline check of `scripted_policy.py`; every frame hop against a known
  rotation ... 72 checks / 41 mutations").
* `scripts/scripted_insert.py` is UNFIT: it has no hydra
  (`rt_logs/RT-186_expectation.md` § "Vehicle", point 2: "it has no hydra
  (`scripted_insert.py:220`) ... so the solver arm is impossible without a
  code change"), and the D-108 scripted-descent gate was dropped
  2026-08-29, so whether it still seats at all is unknown.
* Chosen: the existing evaluation replay of RT-209
  (`rt_logs/RT-209_expectation.md`) — `play.py` on checkpoint
  `.../09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1/model_1499.pt`
  against `scripts/eval/eval_table_seed20.csv`, `--eval-width target`,
  `--seed 1`: 1000 fixed table rows including max tilt/lateral/yaw corners.
  `success_rate_recent` 0.982 (`results/eval/technik/autodr_s1/demo_metrics.json`
  key `success_rate_recent`) means real hard contact happens on nearly every
  row. RT-209's exact command form and the training-PC root
  `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`
  are reused. The folder `eval_target_dd424c93` already exists there
  (RT-209 occupied it), so RT-210c/d each need `--eval-tag` — `play.py`
  REFUSES a folder that already holds `episodes.csv`
  (`scripts/rsl_rl/play.py:517-519`: "REFUSED: ... already holds
  episodes.csv. Pass --eval-tag for a repeat").

## Marker- und Code-Stand

Worktree `p5-kraftsensor`, branch `p5-kraftsensor`. Laptop `HEAD` =
`6fc6565e7d4a000f443c178fb8f3b362f1e69239`, verified pushed:
`git ls-remote origin p5-kraftsensor` → same SHA
(`6fc6565e7d4a000f443c178fb8f3b362f1e69239 refs/heads/p5-kraftsensor`).

Checked: `git diff --stat 79189e3..6fc6565e -- scripts source` → only
`scripts/CLAUDE.md`, `scripts/check_eval_conditions.py` (+14/-3) and
`scripts/list_usd_prims.py` (+7, the solver-attribute printing this run
needs) changed. **No file under `source/` differs between RT-209's code
stand and this one** — RT-210c is expected to reproduce RT-209's physics
exactly, module the seed/table being identical and the tag being the only
difference. `scripts/eval/eval_table_seed20.csv` is byte-identical too
(`git diff --stat 79189e3..6fc6565e -- scripts/eval/eval_table_seed20.csv`
→ empty, no commit touches it in that range) — its RT-209 hash
`dd424c934d50e3452e98e98dde499acf337267700f7c7f82161a694df58b7edb` still
applies.

`SCRIPT_MARKER`/`CODE_MARKER`: `play.py` prints no named marker string
(`rt_logs/RT-209_expectation.md` § "Marker- und Code-Stand": "No
`SCRIPT_MARKER`/`CODE_MARKER` print exists for `play.py` itself"). The gate
for RT-210c/d is the same one RT-209 used: the `[eval-table]` line
(`insertion_env.py:4902-4908`) carrying the table's sha256, and
`eval_manifest.json`'s `code_commit` = `6fc6565e7d4a000f443c178fb8f3b362f1e69239`
/ `code_dirty` = `false`. For RT-210a/b `list_usd_prims.py` itself is the
marker: it must be the version that prints `artPosIters`/`artVelIters`/
`rbPosIters`/`rbVelIters` (added in commit `6fc6565e`, `scripts/list_usd_prims.py:105-108`);
an older clone silently prints only mass/inertia/COM and the run is void
(same precedent as RT-151b, `rt_logs/VERDICTS.md` 2026-09-05 14:51:11,
discarded for a marker mismatch).

**The guard specific to RT-210d**: its startup report line
(`insertion_env.py:5102-5145`, printed as
`solver iterations (D-115 verify): /World/envs/env_0/Robot/root_joint:
solverPositionIterationCount=..., solverVelocityIterationCount=...`) must
read `solverPositionIterationCount=32`. RT-207s1's log shows the un-overridden
line reads 16 (`rt_logs/RT-207s1/RT-207s1.txt:139`:
"solverPositionIterationCount=16, solverVelocityIterationCount=1") and the
Fixture root carries no authored value at all
(`rt_logs/RT-207s1/RT-207s1.txt:140`: "NO authored value on any prim below
-- the PhysX SDK default applies and is NOT readable here"). If RT-210d's
line still reads 16, the hydra override was ignored and the arm is VOID —
fall back to editing `ur5e_cfg.py:182` for that arm only, note the edit in
the log, revert afterwards (same fallback `rt_logs/RT-186_expectation.md`
names, never tried).

**The cap check**: `insertion_env_cfg.py:591` is
`sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)`
— no `PhysxCfg` is passed, so Isaac Lab's own default applies:
`max_position_iteration_count: int = 255`
(`C:\IsaacLab\source\isaaclab\isaaclab\sim\simulation_cfg.py:76`, read on
the laptop). 32 is far under that cap, so the override is not silently
clamped by the sim-level ceiling — only the guard line above can show
whether hydra actually reached the field.

## The one hypothesis

**16/1 is stable on this hard-contact table (exits 0, guard line reads 16,
no non-finite value in `episodes.csv`, no Traceback / PhysX error), and
32/1 changes interpenetration and force peaks by no more than the
RT-209-vs-RT-210c repeat spread.**

Falsified if: the 16/1 run does not exit 0, or the guard line does not read
16, or `episodes.csv` contains a non-finite cell, or a Traceback / PhysX
error line appears in the RT-210c log — OR if `interpen_max_mm` / the force
fields in RT-210d differ from RT-210c by more than the RT-209-vs-RT-210c
spread.

## Known confound — name it, do not resolve it here

RT-206s1 was **trained** at 16/1. At 32/1 the policy acts on physics it has
never seen during training. A difference between RT-210c and RT-210d can
therefore be **policy mismatch**, not solver quality — this run cannot
separate the two. Additionally, the fixture authors no solver iterations at
all (RT-207s1 log line 140, above); how PhysX combines the robot's 16 (or
32) with the fixture's PhysX-SDK-default count in one contact pair is NOT
established anywhere in this repo — do not claim an answer to that from
this run.

## Pre-run block (PowerShell, training PC, root
`C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`)

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1

git branch --show-current
git fetch origin p5-kraftsensor
git merge --ff-only FETCH_HEAD
# STOP, no run, if this fails

git rev-parse HEAD
# expected: c8891e2482e5a33979b0b015125b1ec4b85c4b05 (or newer: git diff --stat 6fc6565e..HEAD -- scripts source must be empty)
git status --short
# expected: empty

Test-Path "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt"
# expected: True (same checkpoint RT-209 used)
```

## RT-210a — shipped ur5e.usd, before our override (UNVERIFIED script path)

```powershell
.\scripts\rt_log.ps1 RT-210a python -u scripts/list_usd_prims.py --usd ISAAC_NUCLEUS_DIR/Robots/UniversalRobots/ur5e/ur5e.usd
```

If a prim prints no `artPosIters`/`artVelIters`/`rbPosIters`/`rbVelIters`
bit at all, that means "no authored value in the file" (PhysX default
applies, unreadable) — record it as "not authored", never invent a number
for the SDK default.

## RT-210b — the asset the env actually loads

```powershell
.\scripts\rt_log.ps1 RT-210b python -u scripts/list_usd_prims.py --usd "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\source\insertion\insertion\tasks\direct\insertion\assets\Robot\ur5e_tool.usd"
```

This is a flattened copy of `ur5e.usd` plus the welded tool
(`scripts/author_tool_ur5e.py:893-896`: "Flatten the shipped asset into an
in-memory layer ... the shipped UR5e references other layers (possibly on
Nucleus) and the result has to stand alone on the training machine";
`:910-913`: "Second flatten, at export: it absorbs the reference ... so the
written robot USD does not depend on a second file at run time"). Same
robot asset path RT-209's log printed:
`[insertion] robot asset: ...\assets\Robot\ur5e_tool.usd`
(`rt_logs/RT-209/RT-209.txt:47`).

## RT-210c — replay at 16/1 (no override, tag `solver16`)

```powershell
.\scripts\rt_log.ps1 RT-210c python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt" --eval-table scripts/eval/eval_table_seed20.csv --eval-width target --eval-tag solver16
```

Resulting folder (`eval_dirname("target", "dd424c93...", "solver16")`,
`scripts/tools/eval_manifest.py:81-93`): `eval_target_dd424c93_solver16`.

```powershell
Copy-Item rt_logs\RT-210c.txt "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_target_dd424c93_solver16\RT-210c.txt"
```

Copy back to the laptop: the training-PC folder
`...\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_target_dd424c93_solver16`
(with its `demo_metrics.json`, `episodes.csv`, `episodes_meta.json`,
`eval_manifest.json`, `applied.csv`, `applied_meta.json`, `RT-210c.txt`)
into `results\solver_probe\RT-210c\` in the worktree; the full log also
into `rt_logs\RT-210c\RT-210c.txt`.

## RT-210d — replay with the 32 override (tag `solver32`), UNVERIFIED hydra path

```powershell
.\scripts\rt_log.ps1 RT-210d python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt" --eval-table scripts/eval/eval_table_seed20.csv --eval-width target --eval-tag solver32 env.robot_cfg.spawn.articulation_props.solver_position_iteration_count=32
```

Resulting folder: `eval_target_dd424c93_solver32`.

```powershell
Copy-Item rt_logs\RT-210d.txt "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_target_dd424c93_solver32\RT-210d.txt"
```

Copy back to the laptop: the same six files into
`results\solver_probe\RT-210d\`, the full log into `rt_logs\RT-210d\RT-210d.txt`.

**Both c and d MUST be checked against the guard line before anything else
is read** — if RT-210d's line reads 16 instead of 32, the arm is VOID and
none of its numbers may be compared against RT-210c.

## PASS/FAIL points, in script order

**RT-210a/RT-210b (`list_usd_prims.py`) — no `judge_run`, this script has
none. Read directly:**
* Script prints `[list_usd_prims] <resolved path>` then one line per prim;
  a prim carries a solver bit only where `HasAuthoredValue()` is true
  (`scripts/list_usd_prims.py:110-112`). Exit code must be 0. A prim with
  none of the four attributes is reported as "not authored", never as 0.

**RT-210c/RT-210d (`play.py --eval-table`) — no `judge_run` in `play.py`
either (no `check_*.py` runs this file). Read directly, in the order the
script would hit them:**
* Run must exit 0 (no Traceback).
* The guard line `solver iterations (D-115 verify): .../Robot/root_joint:
  solverPositionIterationCount=...` must read 16 (RT-210c) / 32 (RT-210d).
  Wrong number = VOID, per the guard above.
* `eval_manifest.json`: `code_commit` = `6fc6565e7d4a000f443c178fb8f3b362f1e69239`,
  `code_dirty` = `false`, `envs_finished_first_episode` = `1000`,
  `table_sha256` = `dd424c934d50e3452e98e98dde499acf337267700f7c7f82161a694df58b7edb`,
  `tag` = `"solver16"` / `"solver32"`.
* `episodes.csv`: no non-finite (`NaN`/`inf`) cell in any column
  (`EPISODE_LOG_COLUMNS`, `insertion_env.py:84-88`: `run, seed, iteration,
  env_id, episode, outcome, steps, ip_max_mm, ip_steps_ge_t1,
  force_max_filtered_n, force_max_raw_n, bounds_version, dr_phase`).
* No PhysX warning/error line in the run's own log beyond the boilerplate
  `enable_external_forces_every_iteration` line RT-209 also printed
  (`rt_logs/RT-209/RT-209.txt:32`-ish) — any NEW warning/error text is
  reported, not silently dropped.

There is **no existing numeric threshold for "stable"** anywhere in this
repo for this setup. None is invented here. "Stable for 16/1" is defined
purely by the four bullets above (exit 0, guard=16, no non-finite, no
Traceback/PhysX error); the magnitude of interpenetration and force is
**reported and compared** against RT-209 and RT-210d, never thresholded.
**This is the gap named explicitly, not closed by this file.**

## What to record per replay arm (RT-209, RT-210c, RT-210d), from files

1. **Interpenetration** — `interpen_max_mm` block in `demo_metrics.json`
   (`mean_mm`, `p50_mm`, `p95_mm`, `p99_mm`, `max_mm`, `thresh_mm`,
   `over_thresh`) and per-episode `ip_max_mm` in `episodes.csv`. This is
   the **Warp SAPU reading, not a PhysX reading**
   (`episodes_meta.json` → `"interpenetration"` → `"method"`: "Warp mesh
   query, insertion_sdf.SdfDistanceQuery.interpen_distances ... not a PhysX
   reading").
2. **Jitter / explosion** — **no dedicated instrument exists.** Say so.
   Readable proxies only: any non-finite value in `episodes.csv`,
   `force_max_raw_n` vs `force_max_filtered_n` per episode (a raw spike far
   above the filtered value hints at a single-step transient), and the
   outcome counts (`success` / `force_abort` / `timeout` / `other`, tallied
   from `episodes.csv`'s `outcome` column).
3. **Force peaks at tool_link** — `force_max_raw_n` per episode (max, p99
   over the episodes) and the `force_norm_n` block in `demo_metrics.json`
   (`mean_n`, `p50_n`, `p95_n`, `p99_n`, `max_n`, `success_p95_n`,
   `failure_p95_n`, `f_max_n`, `over_f_max`).
4. **NaN / robot or OSC failure** — the env has **no NaN guard**: `grep -n
   "isnan\|isfinite" source/insertion/insertion/tasks/direct/insertion/insertion_env.py`
   returns no match (checked). Say so. Readable instead: exit code /
   Traceback in the log, non-finite cells in `episodes.csv`,
   `start_pose_solve_unconverged_resets` (RT-209's value: `0`, from
   `results/eval/technik/autodr_s1/demo_metrics.json`), and any PhysX
   warning/error line in the log.

Also `wall_time_s` from `eval_manifest.json` (RT-209's value: `420.3147518634796`
s, `results/eval/technik/autodr_s1/eval_manifest.json` key `wall_time_s`) —
read for RT-210c and RT-210d the same way, to see the cost of 32.

## RT-209's numbers — the noise floor RT-210c is compared against

Read from `results/eval/technik/autodr_s1/demo_metrics.json` and
`episodes.csv` (commands: `python -c "import json;
print(json.load(open('results/eval/technik/autodr_s1/demo_metrics.json')))"`
and `head`/`Select-String` on `episodes.csv`), verified here:

* `interpen_max_mm` = `{"episodes": 2000, "mean_mm": 0.18858376749710715,
  "p50_mm": 0.15414506196975708, "p95_mm": 0.4194676876068115, "p99_mm":
  0.7096901535987854, "max_mm": 2.0943135023117065, "thresh_mm":
  0.2938000000000038, "over_thresh": 411}`.
* `force_norm_n` = `{"episodes": 2000, "mean_n": 13.203893066942692,
  "p50_n": 11.843729019165039, "p95_n": 26.144535064697266, "p99_n":
  33.285400390625, "max_n": 42.37588119506836, "success_p95_n":
  24.77456283569336, "failure_p95_n": 41.862449645996094, "f_max_n": 30.0,
  "over_f_max": 42}`.
* `start_pose_solve_unconverged_resets` = `0`.
* `wall_time_s` = `420.3147518634796`.
* `success_rate_recent` = `0.982`.

Note this `demo_metrics.json` is the FIXED-path file the env keeps writing
over the whole run and reflects a rolling 2000-episode window, not the
1000-row table 1:1 (1000 envs each finishing at least one episode while
slower envs keep some faster ones repeating — the same mechanism RT-209's
G4 read; `rt_logs/VERDICTS.md` 2026-09-22 11:56:42: "offen: G4 959/999
gleich"). The per-episode `episodes.csv` rows are the exact 1000-row-table
record and are what RT-210c/d's per-episode comparison should use first.

**RT-210c vs RT-209** gives the repeat spread (same code, same table, same
seed, same checkpoint, differing only in the eval-tag / run instance and
the check-script-only commits between them). **No 16-vs-32 difference
(RT-210d vs RT-210c) may be called material unless it exceeds
max(|RT-210c − RT-209|) on each of the numbers above.**

## Pins (read from each arm's `eval_manifest.json`, `MANIFEST_FIELDS`,
`scripts/tools/eval_manifest.py:30-55`)

| Field | RT-210c expected | RT-210d expected |
|---|---|---|
| `checkpoint_path` | ends `...RT-206s1_...\model_1499.pt` | same |
| `table_sha256` | `dd424c934d50e3452e98e98dde499acf337267700f7c7f82161a694df58b7edb` | same |
| `table_rows` | `1000` | `1000` |
| `width` | `"target"` | `"target"` |
| `tag` | `"solver16"` | `"solver32"` |
| `seed` | `1` | `1` |
| `num_envs` | `1000` | `1000` |
| `code_commit` | `6fc6565e7d4a000f443c178fb8f3b362f1e69239` | same |
| `code_dirty` | `false` | `false` |
| `envs_finished_first_episode` | `1000` | `1000` |
| `wall_time_s` | UNKNOWN before the run, compare to RT-209's `420.3` s | UNKNOWN before the run, compare to RT-210c |
| `normalizer_sha256_before`/`_after` | non-null, equal | non-null, equal |

`fixture_pos_noise_xy_m`, `obs_noise_pocket_pos_std_m`,
`force_obs_noise_std_n`, `grasp_obs_offset_x_m` are not pinned here: this
run does not touch noise widths (only `--eval-tag` and the solver override
differ from RT-209's command), so their values are whatever `--eval-table`
sets from the table row (`applied.csv` per row), same as RT-209 — no new
number is claimed for them.

## What this run cannot answer

* **Whether the 16/1 setting stays** — that decision is the user's, not
  this file's. This run only reports whether 16/1 exits clean and how far
  32/1 moves the two magnitudes above the repeat spread.
* **Physical validity of the interpenetration reading** — `ip_max_mm` is a
  Warp SAPU reading against a non-watertight fixture mesh (`episodes_meta.json`
  → `"mesh_note"`: "the fixture mesh is not watertight (RT-64)"), not a
  PhysX contact reading. This run does not add a PhysX-side interpenetration
  instrument.
* **How PhysX combines two different solver iteration counts on one
  contact pair** (robot 16 or 32, fixture unauthored/SDK default) — not
  established anywhere in this repo (see "Known confound" above).
* **Whether an outcome difference in RT-210d is solver quality or policy
  mismatch** — the checkpoint was trained at 16/1 only (see "Known
  confound").
* **Any thesis-grade jitter/explosion instrument** — none exists; only the
  proxy readings listed under "What to record", point 2, are available.
* **A general 16-vs-192-iteration question** (the original D-115 conflict,
  Factory paper vs Isaac Lab's `factory_env_cfg.py`) — this run tests 16 vs
  32 only, on one checkpoint, one table, one GPU.
* **`over_f_max` / `over_thresh` as pass/fail bars** — `f_max_n` (30 N) and
  `thresh_mm` (`INTERPEN_THRESH`, algorithmic acceptance threshold, D-106
  (3)) are reward/termination constants, not stability thresholds; they are
  reported as counts, not judged against a stability criterion.
