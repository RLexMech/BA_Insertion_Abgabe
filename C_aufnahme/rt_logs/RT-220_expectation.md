# RT-220 — the G6 repeat of the technique run (eval_plan.md § 3.2 "Technik", § 6 G6). Written BEFORE the run (2026-09-22)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

## Context

RT-209 (the Erstlauf) ran the AutoDR-seed-1 checkpoint (`RT-206s1`, `model_1499.pt`)
through all 1000 rows of the Seed-20 practice table once, end to end
(`results/eval/technik/autodr_s1/`, `rt_logs/RT-209/`). Verdict PASS, corrected
2026-09-22 (`rt_logs/VERDICTS.md`, 2026-09-22 11:56:42; `HANDOFF-RL.md` § Open
NEXT 1–2). Its own "What this run cannot answer" section named the repeat
explicitly: *"G6 (Wiederholbarkeit) — needs a second run of the same policy on
the same table under `--eval-tag`; not part of RT-209."* `HANDOFF-RL.md` § Open
NEXT, last line before the numbered list closes: *"Then: G6 repeat
(`--eval-tag`) and the nominal-width run."* `eval_plan.md` § 3.2 lists three
technique runs: "Technik (3): Erstlauf, Wiederholung (G6, `--eval-tag`), ein
Lauf in Nominalbreite" (G7 dropped, plan header line 2). RT-220 is
that Wiederholung — the SAME command as RT-209, same policy, same table, same
seed, only `--eval-tag rep2` added, so the two runs can sit side by side in one
`results/eval/technik/` block without overwriting each other (`play.py:512-521`
REFUSES a second run into a folder that already holds `episodes.csv` unless
`--eval-tag` is given). `eval_plan.md` § 6 G6: *"gleiche Policy, gleiche
Tabelle, gleicher Seed, zweimal: Anteil Zeilen mit anderem Ausgang berichten.
Keine Bit-Gleichheit gefordert."* — G6 is a report, not a pass/fail line
(confirmed in code, see "PASS/FAIL points" below).

The tag `rep2` is not named anywhere in `eval_plan.md`; the plan only requires
"a tag" for the repeat. `rep2` is chosen here because it is the exact example
`scripts/tools/eval_manifest.py`'s own self-test uses for a tagged folder name
(`scripts/tools/eval_manifest.py:183`: `eval_dirname("nominal", sha.upper(),
"rep2") == "eval_nominal_abababab_rep2"`) — reusing an already-exercised,
already-valid tag string (`_TAG` pattern: letters, digits, `-` only,
`eval_manifest.py`, checked against the self-test case) rather than inventing
a new one.

## Marker- und Code-Stand

Worktree `p5-kraftsensor`, branch `p5-kraftsensor`. Laptop `HEAD` =
`6fc6565e7d4a000f443c178fb8f3b362f1e69239` (checked: `git rev-parse HEAD`).
`git status --short` shows only `?? "Pläne/eval_plan.md"` (untracked, foreign
file, untouched) and `?? results/` (untracked results tree, RT-209's output
copied in — not a tracked-file change) — no tracked file is dirty.

The training PC ran RT-209 at `79189e32c67597f65f714212a96d6552bd1a9618`
(RT-209's own expectation and `results/eval/technik/autodr_s1/eval_manifest.json`
field `code_commit`, read directly: `"79189e32c67597f65f714212a96d6552bd1a9618"`).
Checked here: `git diff --stat 79189e32c67597f65f714212a96d6552bd1a9618 HEAD --
source scripts` →

```
 scripts/CLAUDE.md                |  3 ++-
 scripts/check_eval_conditions.py | 14 +++++++++++---
 scripts/list_usd_prims.py        |  7 +++++++
 3 files changed, 20 insertions(+), 4 deletions(-)
```

`source/` has ZERO changes between the two commits; nothing that runs on the
training PC (`play.py`, `insertion_env.py`, `insertion_env_cfg.py`,
`eval_manifest.py`, `autodr.py`) moved. The three changed files are all
laptop-side: `check_eval_conditions.py` (the G1b fix — `SCRIPT_MARKER` moved
`check_eval_conditions-2026-09-21a` → `check_eval_conditions-2026-09-22a`,
`git diff` line `-SCRIPT_MARKER = "check_eval_conditions-2026-09-21a"` /
`+SCRIPT_MARKER = "check_eval_conditions-2026-09-22a"`; this is the fix
`HANDOFF-RL.md` § Open NEXT 1 already describes and RT-209's corrected verdict
already used), `list_usd_prims.py` (unrelated tooling), `scripts/CLAUDE.md`
(prose). `eval_collect.py` did not change at all in this diff — its marker
stays `eval_collect-2026-09-21a` (`scripts/eval_collect.py:44`, checked).
**Conclusion: the training PC must NOT pull.** It stays on
`79189e32c67597f65f714212a96d6552bd1a9618`; pulling to `6fc6565e...` would
change nothing `play.py` reads or writes for this run, and pulling
unnecessarily is not this run's job.

No `SCRIPT_MARKER`/`CODE_MARKER` print exists for `play.py` itself (checked
again: `grep -n SCRIPT_MARKER scripts/rsl_rl/play.py` — no match, same as
RT-209). The marker for this run is, as in RT-209, the `[eval-table]` line the
env prints (`insertion_env.py:4902-4908`) and the `code_commit`/`code_dirty`
fields in `eval_manifest.json` (`scripts/tools/eval_manifest.py:32-33`). If
either is missing, or `code_commit` is not `79189e32c67597f65f714212a96d6552bd1a9618`,
or `code_dirty` is not `false`, the run is invalid — same precedent as
RT-151b (`rt_logs/VERDICTS.md`, 2026-09-05 14:51:11, discarded as ÜBERHOLT for
a marker mismatch).

The laptop-side checker that will judge this run's files carries the fixed
marker: `check_eval_conditions-2026-09-22a` (`scripts/check_eval_conditions.py:46`,
checked at `HEAD = 6fc6565e...`). It accepts one-or-more `[eval-table]` lines
in the log, all carrying the table hash (`check_eval_conditions.py:150-157`,
`bool(log_shas) and all(h == table_sha for h in log_shas)`), not the old
single-line rule — relevant because `report_at_steps = (2, 60, 180, 258)`
(`eval_manifest.json` field `env_cfg.report_at_steps`, RT-209's own file, read
above) means the env will again print four `[eval-table]` lines in this run's
log, same as RT-209.

## The one hypothesis

**Running the identical AutoDR-seed-1 checkpoint through the identical
Seed-20 table a second time, under `--eval-tag rep2`, produces a second,
separate evaluation folder (`eval_target_dd424c93_rep2`) whose G1–G3 technical
checks PASS exactly as RT-209's did, and whose per-row outcomes — compared to
RT-209's episode-0 outcomes by `eval_collect.py`'s `g6_report` — are reported,
not judged, as a share of rows with a different outcome.**

## Marker/Pull-Verifikation — kein Pull nötig

No pull command is given because none is needed: the diff above shows
`source/` and `scripts/rsl_rl/play.py` unchanged since `79189e32...`, and the
training PC's clone is already there (RT-209 verified this same SHA on the
same machine). The pre-run block below re-verifies the SHA and a clean tree
directly, rather than trusting that nothing moved between runs.

## Pre-run block (PowerShell, training PC, root
`C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`)

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1

# 1. verify the SHA and a clean tree -- NO pull, NO fetch: this run needs the
#    exact commit RT-209 already ran on, nothing source-side changed since
git rev-parse HEAD
# expected: 79189e32c67597f65f714212a96d6552bd1a9618
git status --short
# expected: empty (no tracked-file changes)

# 2. checkpoint must still exist -- same file RT-209 used
Test-Path "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt"

# 3. the tagged output folder must NOT exist yet -- play.py:517-520 REFUSES a
#    run into a folder that already holds episodes.csv
Test-Path "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_target_dd424c93_rep2\episodes.csv"
# expected: False

# 4. table hash on the training PC, for G1c -- cheap, included again rather
#    than assumed to carry over from RT-209 (per task brief)
Get-FileHash -Algorithm SHA256 scripts\eval\eval_table_seed20.csv
# expected (lowercased, check_eval_conditions.py:157 lowercases the --filehash
# argument itself): dd424c934d50e3452e98e98dde499acf337267700f7c7f82161a694df58b7edb
```

## Command

```powershell
.\scripts\rt_log.ps1 RT-220 python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt" --eval-table scripts/eval/eval_table_seed20.csv --eval-width target --eval-tag rep2
```

Only change from RT-209's command: `--eval-tag rep2` appended. Checked against
`scripts/rsl_rl/play.py`: `--eval-tag` defaults to `""`, type `str`
(`play.py:116-120`, help text: "Suffix of the evaluation folder, for a REPEAT
of the same policy on..."); it sets `replay_dirname =
_em.eval_dirname(args_cli.eval_width, _table_sha, args_cli.eval_tag)`
(`play.py:516`), which — per `eval_dirname`
(`scripts/tools/eval_manifest.py:81-93`) — appends `_rep2` to the same
`eval_target_dd424c93` name RT-209 used, giving
`eval_target_dd424c93_rep2` (verified: `eval_dirname` checked line by line,
`_TAG` pattern accepts `rep2`). Every other flag/omission reasoning from
RT-209's expectation (`--num_envs` absent, `--eval-width target` spelled out
though default, no `env.start_floor_m`/`env.autodr_stall_buffers`/`env.dr_mode`
overrides, `obs_wrench_mode` not overridden, `--headless`, no
`--fixture-offset`/`-tilt`/`-yaw`/`--max-steps`/`--trace-obs`/`--video`)
applies unchanged — same checkpoint, same table, same width, same code.

## PASS/FAIL points, in script order

**G1–G3 — verbatim `CHECK_NAMES` of `scripts/check_eval_conditions.py:90-99`,
run after the folder is copied to the laptop, same nine names and same
PASS/FAIL criteria as RT-209 (unchanged file except the G1b rule fix already
folded into this run's marker `check_eval_conditions-2026-09-22a`):**

Right after the run, on the training PC, put the log INTO the evaluation
folder, so ONE folder goes to the laptop (user rule, 2026-09-22):

```powershell
Copy-Item rt_logs\RT-220.txt "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_target_dd424c93_rep2\RT-220.txt"
```

On the laptop the log is copied once more to `rt_logs\RT-220\RT-220.txt`. The
training PC folder `...\eval_target_dd424c93_rep2` is copied to
`results\eval\technik\autodr_s1_rep2\` in the worktree — the tag becomes the
folder suffix per `_FOLDER = re.compile(r"^(autodr|nodr)_s(\d+)(?:_([A-Za-z0-9-]+))?$")`
(`scripts/eval_collect.py:58`), so `autodr_s1_rep2` parses as method `autodr`,
seed `1`, tag `rep2`. The copy step is MANUAL; no script does it.

```powershell
python scripts/check_eval_conditions.py --run-dir results/eval/technik/autodr_s1_rep2 --table scripts/eval/eval_table_seed20.csv --log rt_logs/RT-220/RT-220.txt --filehash <the Get-FileHash hex of the pre-run block>
.venv-analysis/Scripts/python.exe scripts/eval_collect.py --results results/eval
```

* **G1a table bytes = sidecar = manifest = applied_meta** — PASS iff the
  table file's SHA-256 equals `eval_table_seed20.json`'s `sha256`,
  `eval_manifest.json`'s `table_sha256`, and `applied_meta.json`'s
  `table.sha256`, all `dd424c93...`. FAIL means one of those four disagrees
  with the committed table.
* **G1b the run log's [eval-table] line carries the same hash** — needs
  `--log <RT-220 log file>`; PASS iff every `[eval-table]` line the run
  printed carries `sha256 dd424c93...` (at least one line, all matching —
  the fixed rule). FAIL (or INCOMPLETE without `--log`) means the running env
  did not see the committed table.
* **G1c the training PC's Get-FileHash is the same hash** — needs
  `--filehash <hex from the pre-run block above>`; PASS iff it equals
  `dd424c93...`. FAIL/INCOMPLETE means the file that ran on the training PC
  was not read back correctly, or the flag was omitted.
* **G2a one episode-0 row per env id 0..N-1** — PASS iff `episodes.csv` has
  exactly one `episode == "0"` row per `env_id` in `0..999`. FAIL means an
  env skipped or duplicated its first episode.
* **G2b N = table rows = manifest num_envs = envs that finished** — PASS iff
  table rows (1000), `eval_manifest.json`'s `num_envs` and
  `envs_finished_first_episode` all equal 1000. FAIL means the run ended
  before every env finished, or `--num_envs` mismatch slipped through.
* **G3a applied.csv has the agreed columns and one row per table row, in
  order** — PASS iff `applied.csv`'s header equals `APPLIED_COLUMNS`
  (`insertion_env.APPLIED_LOG_COLUMNS`) and has 1000 rows with `row_id`
  `"0"`..`"999"` in order.
* **G3b the env's bands are the plan's bands for this width** — PASS iff
  `applied_meta.json`'s `bounds` match the check's own `BANDS["target"]`
  literal within `1e-9`, and `fixture_pos_noise_xy_m`/`grasp_obs_offset_x_m`
  match `0.005`/`0.003`.
* **G3c every applied value = the table row mapped, within 1e-6** — PASS iff
  `map_row` matches every `applied.csv` value within `1e-6` for all 1000
  rows.
* **G3d friction read back from PhysX = commanded, on both assets** — PASS
  iff `friction_robot_*`/`friction_pocket_*` in `applied.csv` equal
  `friction_cmd` within `1e-6` for all 1000 rows.

**G6 — NOT one of `CHECK_NAMES`; computed by `eval_collect.py`'s `g6_report`
(`scripts/eval_collect.py:171-190`), reported in `collect_report.json`'s
`"g6"` list once both `autodr_s1` and `autodr_s1_rep2` sit under
`results/eval/technik/`:**
* For `(method="autodr", seed="1", width="target")` with `tags=["", "rep2"]`:
  `g6_report` intersects the two runs' `row_id` sets (`common = sorted(set(a)
  & set(b))`, `eval_collect.py:185`) — expected 1000, since both runs cover
  all 1000 table rows — and counts `diff = sum(1 for k in common if a[k] !=
  b[k])`. PASS/FAIL: **there is none.** `g6_report`'s own docstring:
  *"share of rows whose outcome differs. Reported, no pass line"*
  (`eval_collect.py:172-173`); `eval_plan.md` § 6 G6: *"Anteil Zeilen mit
  anderem Ausgang berichten. Keine Bit-Gleichheit gefordert."* This run's job
  is to produce `rows` = 1000 and `rows_with_other_outcome` = **REPORTED**,
  not predicted. Do NOT predict the number here.
* The only prior hint available is a DIFFERENT quantity, named as such, not
  as an expectation for G6: RT-209's own `collect_report.json` (read from
  `HANDOFF-RL.md` § Open NEXT 3, `rt_logs/VERDICTS.md` 2026-09-22 11:56:42)
  gives 959 of 999 within-RT-209 repeats with the same outcome — that is
  `eval_collect.py`'s G4 (`g4_repeats`/`g4_same_outcome`,
  `eval_collect.py:159-162`), envs that happened to finish early and replay a
  second table row WITHIN THE SAME RUN, same random-noise stream continuing.
  G6 here compares episode 0 of TWO SEPARATE runs (RT-209 vs. RT-220), each
  with its own env reset and its own draw from the global noise stream
  (`eval_plan.md` § 6 "Rauschen": "heute globaler Zufallsstrom"). The two
  numbers measure different things and are not expected to match.

## Pins (from `eval_manifest.json`'s `MANIFEST_FIELDS`,
`scripts/tools/eval_manifest.py:30-51`, and `applied_meta.json`)

| Field | Source | Expected |
|---|---|---|
| `checkpoint_path` | `eval_manifest.json` | ends `...RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt` (same as RT-209) |
| `checkpoint_sha256` | `eval_manifest.json` | equals RT-209's own value, read directly from `results/eval/technik/autodr_s1/eval_manifest.json`: `b7e4ca87516d332a2e5ed4aea1e8f0040cc9ba3cf61f26940af8c47523d66394` — same checkpoint file, same hash function (`play.py:819`), no reason for it to differ |
| `table_path` / `table_sha256` | `eval_manifest.json` | `.../eval_table_seed20.csv` / `dd424c934d50e3452e98e98dde499acf337267700f7c7f82161a694df58b7edb` |
| `table_rows` | `eval_manifest.json` | `1000` |
| `width` | `eval_manifest.json` | `"target"` |
| `tag` | `eval_manifest.json` | `"rep2"` |
| `task` | `eval_manifest.json` | `"Ur5e-Insertion-Direct-v0"` |
| `seed` | `eval_manifest.json` | `1` |
| `num_envs` | `eval_manifest.json` | `1000` |
| `code_commit` | `eval_manifest.json` | `79189e32c67597f65f714212a96d6552bd1a9618` — MUST equal RT-209's; `eval_collect.py:204-205` refuses a block whose runs disagree on `code_commit` |
| `code_dirty` | `eval_manifest.json` | `false` |
| `plan_commit` | `eval_manifest.json` | `6f92d325bb9de81963e4ba6a5262dd292ad01110` (same as RT-209, read from RT-209's own manifest above — the last commit touching `eval_plan.md` has not changed) |
| `obs_wrench_mode` | `eval_manifest.json` | `"force"` |
| `control_mode` | `eval_manifest.json` | `"osc"` |
| `action_mode` | `eval_manifest.json` | literal string `"runner.get_inference_policy (rsl_rl): eval mode, act_inference = actor output, no sampling"` |
| `noise.*` | `eval_manifest.json` `noise` dict | same values as RT-209's manifest (`obs_noise_pocket_pos_std_m` 0.005, `force_obs_noise_std_n` 3.5, `torque_obs_noise_std_nm` 0.0, `grasp_obs_offset_x_m` 0.003, `fixture_pos_noise_xy` 0.005, `reset_joint_noise` 0.0) — same cfg, no override |
| `versions.isaacsim`/`isaaclab`/`rsl_rl`/`torch` | `eval_manifest.json` | expected same as RT-209 (isaacsim `5.1.0.0`, isaaclab `2.3.2.post1`, rsl_rl `3.0.1`, torch `2.7.0+cu118`) since the training PC did not pull or reinstall — UNVERIFIED until read, record whatever it says; RT-209's `torch +cu118` vs. `CLAUDE.md`/`DECISIONS.md` `+cu128` conflict (`HANDOFF-RL.md` § Open NEXT 4) is UNRESOLVED and out of scope here |
| `normalizer_sha256_before` / `_after` | `eval_manifest.json` | non-null, equal to each other (G5); NOT necessarily equal to RT-209's own before/after hash — same checkpoint load, so expected equal to RT-209's `4015a57384bad94e1159e60e9937e96aff07559b0785c1ebd993df1d0afd9959`, but read and compare, do not assume |
| `policy_steps_run` | `eval_manifest.json` | UNKNOWN before the run — RT-209 recorded `255`; may repeat or differ, record what is read |
| `envs_finished_first_episode` | `eval_manifest.json` | `1000` |
| `wall_time_s` | `eval_manifest.json` | UNKNOWN before the run; RT-209 recorded `420.3147518634796` s as a cross-check reference point only, not a prediction |
| `env_cfg.dr_table_width` | `eval_manifest.json` `env_cfg` dict | `"target"` |
| `env_cfg.report_at_steps` | `eval_manifest.json` `env_cfg` dict | `[2, 60, 180, 258]` (same as RT-209, cfg default, unchanged — read from RT-209's own manifest) |

## What this run cannot answer

* **G7** — dropped (`eval_plan.md` header line 2, § 6 row G7). Whether an
  outcome depends on the env slot stays UNCHECKED, same as RT-209.
* **The nominal-width run** of the three "Technik" runs — neither RT-209 nor
  RT-220 covers it; still open per `HANDOFF-RL.md` § Open NEXT ("Then: G6
  repeat (--eval-tag) and the nominal-width run").
* **Why RT-209's G4 gave 959/999 same-outcome, not 999/999** — the candidate
  named in the plan (§ 6 "Rauschen": global noise stream) is NOT examined by
  RT-220; RT-220 measures a different quantity (across-run G6), not the
  within-run G4 cause.
* **G4's `wrench_valid` half** — `--trace-obs` is refused together with
  `--eval-table` (`play.py:302-306`), unchanged from RT-209.
* **Any success-rate claim** — `eval_plan.md` § 1.4 fixes no percentage
  threshold; RT-220, like RT-209, is a technical hypothesis about the path,
  not a policy judgement.
* **`rsl_rl`'s `get_inference_policy`/`act_inference` semantics** (G5's
  code-reading half) — already read once for RT-209 from the training PC's
  installed source; not re-read here, and RT-220 does not change it.
* **Whether the torch `+cu118` vs. `+cu128` conflict is real** — carried over
  from RT-209 (`HANDOFF-RL.md` § Open NEXT 4), UNRESOLVED, not this run's
  job to resolve, only to record what `versions.torch` reads.
* **A bit-identical repeat** — `eval_plan.md` § 6 G6 explicitly does not
  require it ("Keine Bit-Gleichheit gefordert"); RT-220 cannot and does not
  attempt to show determinism at the bit level, only outcome-level agreement,
  reported not judged.
