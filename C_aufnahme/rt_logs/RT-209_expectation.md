# RT-209 — first practice run of the evaluation path (eval_plan.md § 3.2 "Technik", § 6 G1–G8 without G7, § "Verifikation"). Written BEFORE the run (2026-09-21)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

## Context

`eval_plan.md` (worktree root, committed; last commit touching it
`6f92d325bb9de81963e4ba6a5262dd292ad01110`, checked with
`git log -1 --format=%H -- eval_plan.md`; the untracked `Pläne/eval_plan.md`
is a stale foreign copy, not the plan) lays
out the evaluation of the ten frozen policies (`model_1499.pt`, code `5a6cc8e`):
five AutoDR (`RT-206s1..s5`), five No-DR (`RT-207s1..s5`). Everything the plan
needs to run — `RowTable`/`FixedWidth` (dr_mode='table'), `play.py --eval-table`,
`check_eval_conditions.py` (G1–G3), `eval_manifest.py`, `eval_collect.py` — has
so far only been checked offline (`--self-test`, `check_env_wiring.py`,
`check_autodr.py`). Nothing has run on the training PC yet. `eval_plan.md`
§ "Verifikation" calls for exactly one thing first: "Übungslauf AutoDR Seed 1
auf Seed 20 → G1–G8 aus Dateien lesen → `/rt-check`". RT-209 is that run — the
Erstlauf of the three technique runs § 3.2 lists (Erstlauf, Wiederholung G6
with `--eval-tag`, one run in nominal width; G7 is dropped, plan header line 2);
only the Erstlauf is RT-209, see
"What this run cannot answer"). Its only job is to prove the path runs
end-to-end once; it carries no success-rate claim (§ 1.4: "Kein absoluter
Prozentwert gilt als bestanden").

## Marker- und Code-Stand

Worktree `p5-kraftsensor`, branch `p5-kraftsensor`, laptop `HEAD` =
`79189e32c67597f65f714212a96d6552bd1a9618` (checked: `git rev-parse HEAD`,
equal to `origin/p5-kraftsensor` per the task brief). `git status --short` on
the laptop shows only `?? "Pläne/eval_plan.md"` (untracked, foreign file, not
touched here) — no tracked file is dirty.

RT-206s1 trained on `5a6cc8e` (`rt_logs/RT-206s1/RT-206s1.short.txt` line 4:
`[rt_log] git: 5a6cc8e`). Checked: `git merge-base --is-ancestor 5a6cc8e
79189e32c67597f65f714212a96d6552bd1a9618` → exit 0, ancestor confirmed.
Checked: `git diff --stat 5a6cc8e 79189e32c67597f65f714212a96d6552bd1a9618 --
source scripts` → 22 files changed, 7463 insertions(+), 44 deletions(-),
including new `scripts/tools/eval_manifest.py` (215 lines) and changes to
`autodr.py`, `insertion_env.py` (+253/-…), `insertion_env_cfg.py` (+74/-…).
The training PC's clone is on `5a6cc8e` or older (its exact branch/commit is
UNKNOWN from the laptop) and MUST move to `79189e32...` before this run —
`env.dr_mode=table`, `play.py --eval-table`, `check_eval_conditions.py` and
`eval_manifest.py` do not exist at `5a6cc8e`.

No `SCRIPT_MARKER`/`CODE_MARKER` print exists for `play.py` itself (checked:
`grep -n SCRIPT_MARKER scripts/rsl_rl/play.py` — no match). The marker for
this run is the `[eval-table]` line the env prints
(`source/insertion/insertion/tasks/direct/insertion/insertion_env.py:4902-4908`,
checked) and the `code_commit`/`code_dirty` fields `eval_manifest.json`
records (`scripts/tools/eval_manifest.py:32-33`, `MANIFEST_FIELDS`). Both are
read from files after the run, not printed under a named marker string — if
either is missing or `code_dirty` is not `false`, the run is invalid, same
precedent as RT-151b (`rt_logs/VERDICTS.md`, 2026-09-05 14:51:11, discarded as
ÜBERHOLT for a marker mismatch).

## The one hypothesis

**The evaluation path (`env.dr_mode=table`, `play.py --eval-table`) runs the
first AutoDR-seed-1 checkpoint through all 1000 rows of the Seed-20 practice
table end to end without crashing, and every file it writes
(`episodes.csv`, `applied.csv`, `applied_meta.json`, `eval_manifest.json`)
lets `check_eval_conditions.py` confirm G1a/G2/G3 and lets the plan's other
files answer G4, G5, G8 (G6, G7 need runs this file does not cover; G7 is
dropped per the task brief).**

## Pre-run block (PowerShell, training PC, root
`C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`)

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1

# 1. which branch is checked out here (UNKNOWN from the laptop)
git branch --show-current

# 2. pull to the SHA this run needs
git fetch origin p5-kraftsensor
git merge --ff-only FETCH_HEAD
# STOP, no run, if this fails (diverged history / wrong branch checked out)

# 3. verify the SHA and a clean tree -- eval_collect.py refuses code_dirty != false
#    ("Refuses with exit 1 ... a dirty clone", scripts/eval_collect.py:16-18;
#    the manifest field itself, scripts/tools/eval_manifest.py:33)
git rev-parse HEAD
# expected: 79189e32c67597f65f714212a96d6552bd1a9618
git status --short
# expected: empty (no tracked-file changes)

# 4. checkpoint must exist -- NOT verified from the laptop
Test-Path "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt"

# 5. table hash on the training PC, for G1c
Get-FileHash -Algorithm SHA256 scripts\eval\eval_table_seed20.csv
# expected (lowercased, check_eval_conditions.py:157 lowercases the --filehash
# argument itself): dd424c934d50e3452e98e98dde499acf337267700f7c7f82161a694df58b7edb
```

## Command

```powershell
.\scripts\rt_log.ps1 RT-209 python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt" --eval-table scripts/eval/eval_table_seed20.csv --eval-width target
```

Notes on the command, each checked against `scripts/rsl_rl/play.py`:
* `--num_envs` is NOT passed: `--eval-table` sets `env_cfg.scene.num_envs` to
  the table's row count itself (play.py:316-319) and REFUSES if `--num_envs`
  is given and disagrees.
* `--eval-width target` is the default already (`choices=("target",
  "nominal")`, default `"target"`, play.py:110-115) but is spelled out per the
  task brief ("width `target`").
* No `--eval-tag`: this is the Erstlauf, not the repeat (G6).
* No `env.start_floor_m` override (G8): `dr_mode='table'` REFUSES unless
  `start_floor_m == start_tip_above_entrance`
  (`insertion_env_cfg.py:1914-1924`), and both default to the same constant
  `RUNG0_START_TIP_ABOVE_ENTRANCE = 0.020`
  (`insertion_env_cfg.py:481,407`, `insertion_tasks_cfg.py:1685`) — leaving
  both at default satisfies the guard; passing RT-206s1's training value
  `env.start_floor_m=-0.030` would violate it and REFUSE the run.
* No `env.autodr_stall_buffers` override: that field only matters under
  `dr_mode=autodr` (RT-206s1's training mode); `--eval-table` forces
  `env_cfg.dr_mode = "table"` itself (play.py:320), so the AutoDR-only field
  is inert here.
* No `env.dr_mode=...` override: `--eval-table` sets it (play.py:320); passing
  a conflicting one would be double-set, not tested here — omitted.
* `obs_wrench_mode` is NOT overridden: RT-206s1 trained with
  `obs_wrench_mode: force` (`rt_logs/RT-206s1/RT-206s1_env.yaml:116`), which
  is also the cfg default (`obs_wrench_mode: str = "force"`,
  `insertion_env_cfg.py:232`) — no override needed, and the checkpoint-width
  guard (`play.py:575-578`, D-188) would REFUSE a mismatched one anyway.
* `--headless`: every prior `play.py` run in `rt_logs/*_expectation.md` on
  this machine uses it (e.g. `RT-149`, `RT-158`, `RT-186_expectation.md`);
  1000 parallel envs are not meant to be watched.
* `--fixture-offset`/`-tilt`/`-yaw`, `--max-steps`, `--trace-obs`, `--video`
  are refused in combination with `--eval-table` (play.py:302-306) and are
  therefore absent.

## PASS/FAIL points, in script order

**G1–G3 — verbatim `CHECK_NAMES` of `scripts/check_eval_conditions.py:89-99`,
run after the folder is copied to the laptop:**

Right after the run, on the training PC, put the log INTO the evaluation
folder, so ONE folder goes to the laptop (user, 2026-09-22):

```powershell
Copy-Item rt_logs\RT-209.txt "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_target_dd424c93\RT-209.txt"
```

On the laptop the log is then copied once more to `rt_logs\RT-209\RT-209.txt`
(the path `/rt-check` and `filter_rt_log.py` use). An extra file in the run
folder is harmless: `eval_collect.py` iterates only sub-folders (line 237)
and reads the run's files by name.

Laptop layout (`scripts/eval_collect.py` docstring, `BLOCKS["technik"]`,
line 55): copy the training PC folder
`...\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_target_dd424c93`
to `results\eval\technik\autodr_s1\` in the worktree; the full log goes to
`rt_logs\RT-209\RT-209.txt` (pattern of the check's docstring line 30). The
copy step itself is MANUAL; no script does it.

```powershell
python scripts/check_eval_conditions.py --run-dir results/eval/technik/autodr_s1 --table scripts/eval/eval_table_seed20.csv --log rt_logs/RT-209/RT-209.txt --filehash <the Get-FileHash hex of the pre-run block>
.venv-analysis/Scripts/python.exe scripts/eval_collect.py --results results/eval
```

* **G1a table bytes = sidecar = manifest = applied_meta** — PASS iff the
  table file's SHA-256 equals `eval_table_seed20.json`'s `sha256`,
  `eval_manifest.json`'s `table_sha256`, and `applied_meta.json`'s
  `table.sha256`, all `dd424c93...`. FAIL means one of those four disagrees
  with the committed table.
* **G1b the run log's [eval-table] line carries the same hash** — needs
  `--log <RT-209 log file>`; PASS iff every `[eval-table]` line the run
  printed (`insertion_env.py:4902-4908`) carries `sha256 dd424c93...`. FAIL
  (or `None`/INCOMPLETE without `--log`) means the running env did not see
  the committed table.
* **G1c the training PC's Get-FileHash is the same hash** — needs
  `--filehash <hex from the pre-run block above>`; PASS iff it equals
  `dd424c93...` (case-insensitive, `check_eval_conditions.py:157` lowercases
  it). FAIL/INCOMPLETE means the file that ran on the training PC was not
  read back correctly, or the flag was omitted.
* **G2a one episode-0 row per env id 0..N-1** — PASS iff `episodes.csv` has
  exactly one `episode == "0"` row per `env_id` in `0..999`. FAIL means an
  env skipped or duplicated its first episode.
* **G2b N = table rows = manifest num_envs = envs that finished** — PASS iff
  table rows (1000), `eval_manifest.json`'s `num_envs` and
  `envs_finished_first_episode` all equal 1000. FAIL means the run ended
  before every env finished (crash, timeout) or `--num_envs` mismatch slipped
  through.
* **G3a applied.csv has the agreed columns and one row per table row, in
  order** — PASS iff `applied.csv`'s header equals `APPLIED_COLUMNS`
  (`check_eval_conditions.py:59-64`, sourced from
  `insertion_env.APPLIED_LOG_COLUMNS`, `insertion_env.py:97-100`) and has
  1000 rows with `row_id` `"0"`..`"999"` in order.
* **G3b the env's bands are the plan's bands for this width** — PASS iff
  `applied_meta.json`'s `bounds` match the check's own `BANDS["target"]`
  literal (`eval_plan.md` § 1.1) within `1e-9`, and
  `fixture_pos_noise_xy_m`/`grasp_obs_offset_x_m` match `0.005`/`0.003`
  (`eval_plan.md` § 1.1 rows "Lage der Aufnahme xy", "Greif-Fehler").
* **G3c every applied value = the table row mapped, within 1e-6** — PASS iff
  `map_row` (the check's own float64 re-derivation, NOT an import of
  `autodr`/`insertion_math`) matches every `applied.csv` value within `1e-6`
  for all 1000 rows.
* **G3d friction read back from PhysX = commanded, on both assets** — PASS
  iff the `friction_robot_*`/`friction_pocket_*` (or equivalent
  `FRICTION_READBACKS`) columns in `applied.csv` equal `friction_cmd` within
  `1e-6` for all 1000 rows.

**G4 — NOT a `check_eval_conditions.py` name; read directly from this run's
own `episodes.csv` (columns `env_id`, `episode`, `outcome`,
`insertion_env.py:84-88` `EPISODE_LOG_COLUMNS`), the same pairing
`eval_collect.py:159-162` computes (`first`/`second` dicts, `g4_repeats`,
`g4_same_outcome`) — `eval_collect.py` reports it as a share once the folder
sits at `results/eval/technik/autodr_s1/`:**
* Outcome half — OBSERVE, not a pre-registered PASS: whether any env
  finishes its table row before the slowest envs (`play.py:766-776`: the loop
  ends only once every env has posted one `episode == "0"` row; envs that
  finish earlier keep replaying and accumulate `episode == "1"`, `"2"`, ...
  rows meanwhile). If `episodes.csv` holds any `episode == "1"` rows, report
  `g4_repeats` = count and `g4_same_outcome` = how many match their
  `episode == "0"` outcome for the same `env_id`. If it holds none, G4's
  outcome half is UNANSWERED by this run, say so plainly — it is not a FAIL.
* `wrench_valid` half — UNANSWERABLE by RT-209: it is only written by
  `--trace-obs` (`play.py:693,700,749`), which `play.py:302-306` REFUSES in
  combination with `--eval-table`. Record this as a limit, not as a missing
  PASS.

**G5 — `eval_manifest.json` fields `normalizer_sha256_before`/
`normalizer_sha256_after`
(`scripts/tools/eval_manifest.py:30-51` `MANIFEST_FIELDS`, comment "G5"):**
PASS iff the two hashes are equal non-null strings (`_normalizer_sha256`,
`play.py:581` before load-time inference use, `play.py:841` after the loop —
both from `runner.load`'s in-memory state, no second file). Whether
`get_inference_policy` → `eval()` and `act_inference` = mean action holds for
rsl_rl 3.0.1 is to be read from the training PC's installed source
(`eval_plan.md` § 6 G5) — UNKNOWN from the laptop, not readable here.

**G8 — read from the command line and the manifest, no
`check_eval_conditions.py` name:** PASS iff (a) `--seed 1` was passed
(checked above it is; `play.py:298-299` REFUSES an unset `--seed` under
`--eval-table`), (b) `eval_manifest.json`'s `width` is `"target"`, and (c) no
`start_floor_m` override was given (checked above; the guard at
`insertion_env_cfg.py:1918-1924` would otherwise have REFUSED the run before
it started, so the run existing at all is itself G8-evidence for that half).

**G6, G7 — not answerable by RT-209** (per the task brief: G7 dropped
entirely; G6 needs the repeat run under `--eval-tag`, a separate RT).

## Pins (from `eval_manifest.json`'s `MANIFEST_FIELDS`,
`scripts/tools/eval_manifest.py:30-51`, and `applied_meta.json`)

| Field | Source | Expected |
|---|---|---|
| `checkpoint_path` | `eval_manifest.json` | ends `...RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt` |
| `checkpoint_sha256` | `eval_manifest.json` | UNKNOWN before the run (`_em.sha256_file(resume_path)`, `play.py:819`) — record it, do not predict it |
| `table_path` / `table_sha256` | `eval_manifest.json` | `.../eval_table_seed20.csv` / `dd424c934d50e3452e98e98dde499acf337267700f7c7f82161a694df58b7edb` |
| `table_rows` | `eval_manifest.json` | `1000` |
| `width` | `eval_manifest.json` | `"target"` |
| `tag` | `eval_manifest.json` | `""` |
| `task` | `eval_manifest.json` | `"Ur5e-Insertion-Direct-v0"` |
| `seed` | `eval_manifest.json` | `1` |
| `num_envs` | `eval_manifest.json` | `1000` |
| `code_commit` | `eval_manifest.json` (`git_commits`, `eval_manifest.py:103-121`) | `79189e32c67597f65f714212a96d6552bd1a9618` |
| `code_dirty` | `eval_manifest.json` | `false` |
| `plan_commit` | `eval_manifest.json` (`git log -1 --format=%H -- eval_plan.md`, `eval_manifest.py:119`) | `6f92d325bb9de81963e4ba6a5262dd292ad01110` (laptop, same command at `79189e32`). `null` is allowed by `MAY_BE_NONE` (sparse clone, `eval_manifest.py:59-63`), but `eval_collect.py` refuses a missing plan commit (docstring line 18) -> `null` = OBSERVE and report, the collector will then exit 1 |
| `obs_wrench_mode` | `eval_manifest.json` | `"force"` |
| `control_mode` | `eval_manifest.json` | `"osc"` (`rt_logs/RT-206s1/RT-206s1_env.yaml:115`) |
| `action_mode` | `eval_manifest.json` | literal string `"runner.get_inference_policy (rsl_rl): eval mode, act_inference = actor output, no sampling"` (`play.py:830`) |
| `noise.fixture_pos_noise_xy` etc. | `eval_manifest.json` `noise` dict | read off `_cfg` at run time, cross-check against § 1.1 bands, not predicted here |
| `versions.isaacsim`/`isaaclab`/`rsl_rl` | `eval_manifest.json` (`_versions`, `play.py:268-277`) | either a version string or `"UNKNOWN: <reason>"` (`metadata.PackageNotFoundError`) — BOTH are legal outcomes per `_versions`'s own docstring; record which, do not treat `UNKNOWN` as a failure |
| `normalizer_sha256_before` / `_after` | `eval_manifest.json` | non-null, equal (G5, above) |
| `policy_steps_run` | `eval_manifest.json` | UNKNOWN before the run (depends on how long the slowest env takes; capped implicitly by `episode_length_buf` / `max_episode_length`) |
| `envs_finished_first_episode` | `eval_manifest.json` | `1000` |
| `wall_time_s` | `eval_manifest.json` (`play.py:844`, `time.time() - _eval_t0`) | UNKNOWN — this is the § 3.2 "Laufzeit je Lauf UNBEKANNT; der erste Techniklauf misst sie" measurement. Read it directly from this field; cross-check against the rt_log's own timestamps (`[rt_log] 2026-09-21 ...` start line vs. the `[play] evaluation complete` / process-exit line in `rt_logs\RT-209.txt`) as a second source, not the primary one |
| `env_cfg.dr_table_width` | `eval_manifest.json` `env_cfg` dict | `"target"` |

## Wall-clock time

Primary source: `eval_manifest.json`'s `wall_time_s` field (exact, computed
in-process, `play.py:844`). Secondary/cross-check source: the timestamp on
`rt_log.ps1`'s own start line in `rt_logs\RT-209.txt` (`[rt_log] <date>
<time> cwd: ...`) against the file's mtime or its last printed line's
timestamp (rt_log.ps1 does not stamp every line, only the header — see
`scripts/rt_log.ps1`). If the two disagree by more than a few seconds, prefer
`wall_time_s` and note the discrepancy; it isolates play.py's own loop from
process startup/teardown that the outer rt_log wrapper also times.

## What this run cannot answer

* **G6 (Wiederholbarkeit)** — needs a second run of the same policy on the
  same table under `--eval-tag`; not part of RT-209.
* **G7** — dropped (`eval_plan.md` header line 2, § 6 row G7). Whether an
  outcome depends on the env slot stays UNCHECKED.
* **G4's `wrench_valid` half** — `--trace-obs` is refused together with
  `--eval-table` (`play.py:302-306`), so whether the wrench read is `False`
  exactly on the reset step cannot be shown by this run; only the "second
  episode's outcome repeats the first" half is checkable, and only if some
  envs happen to finish early enough to accumulate a second episode before
  the slowest envs end the loop.
* **The repeat run and the nominal-width run** of the three "Technik" runs
  (§ 3.2: Erstlauf, Wiederholung, one Nominalbreite run) — neither is RT-209;
  RT-209 is the Erstlauf only.
* **Any success-rate claim** — `eval_plan.md` § 1.4 fixes no percentage
  threshold for this or any run; RT-209 is a technical hypothesis about the
  path running end to end, not a policy judgement.
* **`rsl_rl`'s `get_inference_policy`/`act_inference` semantics** (G5's second
  half) — must be read from the training PC's installed `rsl_rl` 3.0.1
  source, not visible from the laptop.
* **`eval_collect.py`'s cross-run checks** (two folders same policy without a
  tag, code/plan-commit agreement across a block) — with one folder they have
  nothing to compare; the collector still runs G1a/G2/G3 and the per-run
  refusals on it.
