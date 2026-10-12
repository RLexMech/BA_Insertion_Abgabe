# RT-222 — block T-nominal, all ten policies (eval_plan.md § 3.2). Written BEFORE the run (2026-09-22)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Written in the main chat with the `rt-expectation` skill. Template:
`rt_logs/RT-221_expectation.md`. Unchanged sections point there; only what
this block changes is argued here.

## Context

The technique block is accepted (`HANDOFF-RL.md` § Open, 2026-09-22: RT-209
PASS, RT-220 G6 = 0 of 1000 rows differ, RT-221 nominal width G1a/G2/G3
PASS). `eval_plan.md` § 3.2 fixes the order "Technik → T-nominal → T-auswahl
→ … ". T-nominal is next: table seed 10, nominal width, all ten frozen
policies (`model_1499.pt`, AutoDR RT-206s1..s5, No-DR RT-207s1..s5), play seed
100 (§ 3.2 "Abspiel-Seed", commit `391c7ae4`). Purpose (§ 3.2): K2 and the
cause check for T-final.

New: the block runs through `scripts/run_eval.ps1` (commit `f90d50c1`), one
command for ten `play.py` runs, each through `rt_log.ps1`.

## What "nominal" means here

Unchanged from `RT-221_expectation.md` § "What 'nominal' means here". RT-221
ran nominal width once and G3b/G3c PASSED there (`HANDOFF-RL.md` § Open).

## Block → table, width, names (read, not assumed)

* `scripts/eval_collect.py:51`: `"T-nominal": ("eval_table_seed10.csv", "nominal")`.
  `run_eval.ps1` reads this line; it does not type the table.
* Table `scripts/eval/eval_table_seed10.csv`, laptop `Get-FileHash`
  `1613635BD5F942E355D2E413DEBA25A5E97B368DCE4798AD9030ED89CF27ABA3`
  = sidecar `eval_table_seed10.json` `"sha256"`, `n_rows` 1000. The file is
  `-text` in git (`git ls-files --eol`: `i/lf w/lf attr/-text`), so the
  training-PC bytes are the laptop bytes.
* Eval folder per policy: `<run folder>\eval_nominal_1613635b`
  (`eval_dirname`, `scripts/tools/eval_manifest.py:81-93`, no tag).
* Ten run folders: `log_dir` in `rt_logs/RT-206s<N>/RT-206s<N>_env.yaml`
  and `rt_logs/RT-207s<N>/RT-207s<N>_env.yaml`, copied into the
  `$Policies` list of `run_eval.ps1`. Exception RT-206s2..s4: their
  `env.yaml` names the folder `…RT-206s1…`, but on the training PC the
  folders are `…RT-206s2/s3/s4…` (user, 2026-09-22, after the first
  `run_eval.ps1` stopped with "no checkpoint"). Timestamps as in `env.yaml`;
  the dry run's checkpoint check confirms each path. `_autodr_` and
  `_seed<N>` are unchanged, and those two are what
  `identity_from_checkpoint` reads (`eval_collect.py:79-85`).
* rt_log names `RT-222_<autodr|nodr>_s<N>` (10 logs, one per run).

## Marker- und Code-Stand

Clone `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`,
branch `p5-kraftsensor`. **ONE pull, before the block, never during it.**
Expected HEAD after the pull: the commit that fixes the RT-206s2..s4 folder
names in `run_eval.ps1` and this file (after `d3bfef0c`); the chat hands over
the full SHA.

`git diff --stat 42b0b65a 391c7ae4 -- source scripts` (laptop, 2026-09-22):
empty. So `play.py`, the env and the tools are byte-equal to the technique
block; `f90d50c1` adds only `scripts/run_eval.ps1`, this commit only this
file. `plan_commit` becomes `391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3`
(last commit touching `eval_plan.md`; `eval_manifest.py:31`).

Laptop checker marker: `check_eval_conditions-2026-09-22a`
(`scripts/check_eval_conditions.py:46`).

The RT-210 session works in the same training-PC folder. If it pulls during
the block, HEAD moves and `run_eval.ps1` stops before the next run (by
design). Its untracked files do not stop the block: the clean check is
`git status --porcelain --untracked-files=no`, the same rule as the
manifest's `code_dirty` (`eval_manifest.py:120`).

## The one hypothesis

**All ten policies play the seed-10 table in nominal width end to end
through `run_eval.ps1`, and every run passes G1–G3; `eval_collect.py`
accepts block T-nominal with one `code_commit`, one `plan_commit` and ten
folders.**

## Pre-run block (PowerShell, training PC)

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
git pull origin p5-kraftsensor
git rev-parse HEAD
# expected: the SHA from the chat
.\scripts\run_eval.ps1 -Block T-nominal -Rt RT-222 -DryRun
# expected: "pre-block checks: ... -- OK", ten DRY RUN lines, "DRY RUN done", exit 0
```

## Command

```powershell
.\scripts\run_eval.ps1 -Block T-nominal -Rt RT-222
```

Each of the ten runs is (printed by the dry run):
`python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 100 --checkpoint <run folder>\model_1499.pt --eval-table scripts/eval/eval_table_seed10.csv --eval-width nominal`.
Changes against RT-221: table seed 10 instead of 20, `--seed 100` instead of
1, no `--eval-tag`, ten checkpoints instead of one. Every other flag and
omission as argued in `RT-209_expectation.md` § Command.

Wall time: UNKNOWN for this table. RT-221 (nominal, seed-20 table) took 85 s;
reference only.

## Post-run

`run_eval.ps1` already puts the full and short log, `table_filehash.txt` and
`head.txt` into each eval folder. Nothing to copy by hand on the training PC.

Laptop: each folder's CONTENT goes to `results\eval\T-nominal\<autodr|nodr>_s<N>\`
(`_FOLDER`, `eval_collect.py:58`); each log again to
`rt_logs\RT-222\RT-222_<autodr|nodr>_s<N>.txt`. Then, per run:

```powershell
python scripts/check_eval_conditions.py --run-dir results/eval/T-nominal/autodr_s1 --table scripts/eval/eval_table_seed10.csv --log rt_logs/RT-222/RT-222_autodr_s1.txt --filehash 1613635BD5F942E355D2E413DEBA25A5E97B368DCE4798AD9030ED89CF27ABA3
```

(same for the nine others), then

```powershell
.venv-analysis/Scripts/python.exe scripts/eval_collect.py --results results/eval
```

Known: the foreign folder `results/eval/technik/eval_target_dd424c93_solver16`
(RT-210 session) breaks the collector's `technik` block (`HANDOFF-RL.md`
§ Open). It is not mine and is not moved; the T-nominal verdict is read from
the T-nominal part of `collect_report.json`.

## PASS/FAIL points, in script order (CHECK_NAMES verbatim), per run

1. **G1a table bytes = sidecar = manifest = applied_meta** — all `1613635b…`.
2. **G1b the run log's [eval-table] line carries the same hash** — at least
   one line, every line `1613635b…`, width `nominal` by eye.
3. **G1c the training PC's Get-FileHash is the same hash** — from
   `table_filehash.txt` in the eval folder.
4. **G2a one episode-0 row per env id 0..N-1** — 1000.
5. **G2b N = table rows = manifest num_envs = envs that finished** — 1000.
6. **G3a applied.csv has the agreed columns and one row per table row, in order**.
7. **G3b the env's bands are the plan's bands for this width** — against
   `BANDS["nominal"]` (`check_eval_conditions.py:77`).
8. **G3c every applied value = the table row mapped, within 1e-6** — lateral
   0, yaw 0, tilt 0, friction 0.14, start height mapped over 0.020..0.120.
9. **G3d friction read back from PhysX = commanded, on both assets** — 0.14.

Block: `run_eval.ps1` ends with "block T-nominal done: 10 of 10 runs, exit 0
each". `eval_collect.py`: no T-nominal problem, ten folders, one
`code_commit`, one `plan_commit`.

## Pins (`eval_manifest.json`, each of the ten)

| Field | Expected |
|---|---|
| `width` / `env_cfg.dr_table_width` | `"nominal"` |
| `tag` | `""` |
| `seed` / `num_envs` / `table_rows` / `envs_finished_first_episode` | `100` / `1000` / `1000` / `1000` |
| `table_sha256` | `1613635bd5f942e355d2e413deba25a5e97b368dce4798ad9030ed89cf27aba3` |
| `checkpoint_sha256` | autodr_s1: RT-209's `b7e4ca87516d332a2e5ed4aea1e8f0040cc9ba3cf61f26940af8c47523d66394`; the nine others UNKNOWN — record |
| `code_commit` / `code_dirty` | the SHA from the chat, the SAME in all ten; `false` |
| `plan_commit` | `391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3` |
| `normalizer_sha256_before` = `_after` | equal (G5) |
| `env_cfg.start_floor_m` = `start_tip_above_entrance` | `0.02` = `0.02` (G8) |
| `head.txt` | = `code_commit` |
| `wall_time_s`, `policy_steps_run`, success rates | UNKNOWN — record, do not judge |

## What this run cannot answer

* Any robustness claim: nominal width is the No-DR training distribution
  (§ 3.2). T-nominal only gives K2's input and the cause check for T-final.
* Whether one method is better: § 1.4 has no percentage line; the rates are
  recorded, the statistics (`eval_stats.py`) run later over T-final.
* G6 for these ten (no repeat in this block).
* G4's cause (959/999 in the technique block; `HANDOFF-RL.md` § Open item 3).
* The torch `+cu118` vs `+cu128` conflict — recorded, not resolved.
* Start penetration before the first action — not written in any file
  (thesis limit, `HANDOFF-RL.md` § Open).

## CORRECTION, after the first attempt (2026-09-22)

First attempt on `0112be18`: run 1 (`autodr_s1`) printed `[play] evaluation
complete: all 1000 envs finished their first episode after 202 steps` and
the manifest line, then never ended (no `[rt_log] exit code` line). The
user stopped it with Ctrl+C; runs 2..10 never started. RT-209's log ends
the same way, so the hang is not new; a single run hid it, the block runner
waits on it.

Fix (one change): `scripts/rsl_rl/play.py` ends the process with
`os._exit(0)` right after the manifest in an evaluation run, skipping
`env.close()` and `simulation_app.close()`. Which of the two hangs is NOT
measured. Pinned by `check_env_wiring.py` "the evaluation ends the process
right after the manifest", mutation `eval-run-waits-for-the-shutdown`
(laptop: 388/388 checks, counter-proof PASSED).

Consequences for this file:
* Expected HEAD / `code_commit`: the commit that carries this fix; the chat
  hands over the SHA. All ten runs are repeated on it, `autodr_s1` too
  (`eval_collect.py:204-205` refuses two code_commits in one block).
* The first attempt's `autodr_s1` folder is renamed `eval_nominal_1613635b_hung`
  and its logs `RT-222hung_autodr_s1*`, by hand on the training PC. Not data.
* New PASS point per run: the rt_log ends with `[rt_log] exit code: 0` and
  `run_eval.ps1` moves on to the next run by itself.
