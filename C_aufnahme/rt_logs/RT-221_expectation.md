# RT-221 — the nominal-width technique run (eval_plan.md § 3.2 "Technik", § 6). Written BEFORE the run (2026-09-22)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Written in the main chat, not by the `rt-expectation` agent (user,
2026-09-22: the agent cost ~100k tokens per file). Structure and every
unchanged line come from `rt_logs/RT-220_expectation.md` /
`rt_logs/RT-209_expectation.md`; only what this run changes is argued here.

## Context

RT-209 (Erstlauf, target width) is PASS (`rt_logs/VERDICTS.md`, 2026-09-22
11:56:42). RT-220 is the G6 repeat. `eval_plan.md` § 3.2 names the third
technique run: "Technik (3): Erstlauf, Wiederholung (G6, `--eval-tag`), ein
Lauf in Nominalbreite". RT-221 is that run: same policy (RT-206s1
`model_1499.pt`), same table (`scripts/eval/eval_table_seed20.csv`), same
seed 1, width `nominal`.

**The tag `nom` is OUR OWN construction (named gap).** The plan gives a tag
only to the repeat. But `eval_collect.py` keys duplicates on
`(block, method, seed, tag)` WITHOUT the width (`scripts/eval_collect.py:208-213`),
so a second untagged `autodr_s1` folder in block `technik` would be refused
as "the same policy and tag". The training-PC folder alone would not collide
(`eval_nominal_dd424c93` vs `eval_target_dd424c93`, `eval_dirname`,
`scripts/tools/eval_manifest.py:81-92`); the laptop folder would.

## What "nominal" means here (read, not assumed)

* Plan: § 3.2 row T-nominal, "Versatz/Gier/Kippung/Reibung auf Mitte,
  Starthöhe 0,020..0,120 (= No-DR-Trainingsverteilung)".
* Env: `insertion_env.py:552-554` — under `dr_table_width == "nominal"` only
  the `start_height` dimension stays in `FixedWidth.at_max`; the others sit on
  their centres (`lat_r` 0, `yaw` 0, `tilt` 0, friction
  `insertion_tasks_cfg.CONTACT_FRICTION` = 0.14, `insertion_tasks_cfg.py:1396`).
  Cfg comment: `insertion_env_cfg.py:713-714`.
* Check: `BANDS["nominal"]` in `scripts/check_eval_conditions.py:77-83`:
  `lat_r (0,0)`, `yaw (0,0)`, `tilt (0,0)`, `start_height (0.020, 0.120)`,
  `friction (0.140, 0.140)`.
* The three agree on paper. NOT checkable offline: whether
  `applied_meta.json` `bounds` (`_live_bounds()`, `insertion_env.py:3994`)
  carries ALL five keys under `nominal`, as G3b demands
  (`set(got_b) == set(want_b)`, `check_eval_conditions.py:176`). That is
  exactly what G3b of this run tests.

## Marker- und Code-Stand

Training PC stays on `79189e32c67597f65f714212a96d6552bd1a9618` (RT-209's
`code_commit`). **NO pull**: `eval_collect.py:204-205` refuses a block whose
runs disagree on `code_commit`. `git diff --stat 79189e32 HEAD -- source
scripts` (laptop, 2026-09-22): only `scripts/CLAUDE.md`,
`scripts/check_eval_conditions.py`, `scripts/list_usd_prims.py` — nothing
`play.py` runs. Laptop checker marker: `check_eval_conditions-2026-09-22a`.

## The one hypothesis

**The eval path runs the same checkpoint on the same table in nominal width
end to end, and G1–G3 PASS — in particular G3b (the env's nominal bands =
the plan's) and G3c (every applied value = the row mapped in nominal width).**

## Pre-run block (PowerShell, training PC)

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
git rev-parse HEAD
# expected: 79189e32c67597f65f714212a96d6552bd1a9618
git status --short
# expected: empty
Test-Path "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt"
# expected: True
Test-Path "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_nominal_dd424c93_nom\episodes.csv"
# expected: False
Get-FileHash -Algorithm SHA256 scripts\eval\eval_table_seed20.csv
# expected: DD424C934D50E3452E98E98DDE499ACF337267700F7C7F82161A694DF58B7EDB
```

## Command

```powershell
.\scripts\rt_log.ps1 RT-221 python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt" --eval-table scripts/eval/eval_table_seed20.csv --eval-width nominal --eval-tag nom
```

Changes against RT-209: `--eval-width nominal` (`choices=("target",
"nominal")`, `play.py:110-115`) and `--eval-tag nom`. Every other flag and
omission as argued in `RT-209_expectation.md` § Command.

## Post-run

Training PC, right after the run (user rule, 2026-09-22):

```powershell
Copy-Item rt_logs\RT-221.txt "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_nominal_dd424c93_nom\RT-221.txt"
```

Laptop: the folder's CONTENT goes to `results\eval\technik\autodr_s1_nom\`
(`_FOLDER` regex, `eval_collect.py:58`); the log is copied again to
`rt_logs\RT-221\RT-221.txt`.

```powershell
python scripts/check_eval_conditions.py --run-dir results/eval/technik/autodr_s1_nom --table scripts/eval/eval_table_seed20.csv --log rt_logs/RT-221/RT-221.txt --filehash <the Get-FileHash hex of the pre-run block>
.venv-analysis/Scripts/python.exe scripts/eval_collect.py --results results/eval
```

## PASS/FAIL points, in script order (CHECK_NAMES verbatim)

1. **G1a table bytes = sidecar = manifest = applied_meta** — all `dd424c93…`.
2. **G1b the run log's [eval-table] line carries the same hash** — at least
   one line, every line `dd424c93…`, and every line says `width nominal`
   (`insertion_env.py:4906`; the width is read by eye, the check does not
   test it).
3. **G1c the training PC's Get-FileHash is the same hash**.
4. **G2a one episode-0 row per env id 0..N-1** — 1000.
5. **G2b N = table rows = manifest num_envs = envs that finished** — 1000.
6. **G3a applied.csv has the agreed columns and one row per table row, in order**.
7. **G3b the env's bands are the plan's bands for this width** — against
   `BANDS["nominal"]`; FAIL if `bounds` misses a key or a centre differs.
8. **G3c every applied value = the table row mapped, within 1e-6** — in
   nominal width: lateral 0, yaw 0, tilt 0 (pocket quaternion identity),
   friction 0.14, start height mapped over 0.020..0.120.
9. **G3d friction read back from PhysX = commanded, on both assets** — 0.14.

`eval_collect.py`: VERDICT OK, 0 problems, with three folders in `technik`
(`autodr_s1`, `autodr_s1_rep2` if RT-220 is back, `autodr_s1_nom`).

## Pins (`eval_manifest.json`)

| Field | Expected |
|---|---|
| `width` / `env_cfg.dr_table_width` | `"nominal"` |
| `tag` | `"nom"` |
| `seed` / `num_envs` / `table_rows` / `envs_finished_first_episode` | `1` / `1000` / `1000` / `1000` |
| `table_sha256` | `dd424c934d50e3452e98e98dde499acf337267700f7c7f82161a694df58b7edb` |
| `checkpoint_sha256` | RT-209's `b7e4ca87516d332a2e5ed4aea1e8f0040cc9ba3cf61f26940af8c47523d66394` |
| `code_commit` / `code_dirty` | `79189e32c67597f65f714212a96d6552bd1a9618` / `false` |
| `plan_commit` | `6f92d325bb9de81963e4ba6a5262dd292ad01110` |
| `normalizer_sha256_before` = `_after` | equal (G5); RT-209 had `4015a573…` — compare, do not assume |
| `env_cfg.start_floor_m` = `start_tip_above_entrance` | `0.02` = `0.02` (G8) |
| `wall_time_s`, `policy_steps_run` | UNKNOWN; RT-209: 420.3 s, 255 steps — reference only |
| `versions.torch` | record; the `+cu118` vs `+cu128` conflict (`HANDOFF-RL.md` § Open) stays open |

## What this run cannot answer

* G6 in nominal width (no repeat of RT-221).
* Any target-vs-nominal result: one seed, the practice table, § 1.4 has no
  percentage line. The success rate is read and recorded, not judged.
* G4's `wrench_valid` half (`--trace-obs` is refused with `--eval-table`).
* G7 (dropped).
