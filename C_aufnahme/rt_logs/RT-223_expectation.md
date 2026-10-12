# RT-223 — block T-auswahl, all ten policies (eval_plan.md § 3.2). Written BEFORE the run (2026-09-22)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Written in the main chat with the `rt-expectation` skill. Template:
`rt_logs/RT-222_expectation.md` (incl. its CORRECTION section). Every section
not argued here is unchanged from there.

## Context

`eval_plan.md` § 3.2 order: Technik → T-nominal → T-auswahl → (selection
noted, committed) → T-final. T-auswahl follows T-nominal (RT-222) directly on
the training PC, before RT-222 is judged on the laptop (user, 2026-09-22).
Risk named: if RT-222 shows a defect of the evaluation itself, RT-223 is
repeated. Purpose (§ 3.2): selection of the final policy, so T-final stays
untouched by the selection.

## Block → table, width, names

* `scripts/eval_collect.py:52`: `"T-auswahl": ("eval_table_seed30.csv", "target")`.
* Table `scripts/eval/eval_table_seed30.csv`, laptop `Get-FileHash`
  `D26610DC6838B4B2E2B453E267468A898DDE7F20BB2AF8291047AD9CF50A0ECE`
  = sidecar `eval_table_seed30.json` `"sha256"`, `n_rows` 1000; `-text` in git.
* Eval folder per policy: `<run folder>\eval_target_d26610dc`.
* Same ten run folders and checkpoints as RT-222 (`run_eval.ps1` `$Policies`).
* rt_log names `RT-223_<autodr|nodr>_s<N>`.

## Marker- und Code-Stand

**NO pull.** The training PC stays on `8299485d0a60dd0cca01a7d84946f441a96c1100`,
the commit RT-222 runs on. This file is committed on the laptop only as the
before-the-run record; it changes no code. So RT-222 and RT-223 share one
`code_commit` (`eval_collect.py:202-205` compares it over ALL runs).
`plan_commit` stays `391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3`.

The `hub.exe` watcher from the RT-222 session (second window) stays on:
Isaac's `hub.exe` outlives each run and holds rt_log's output pipe (measured
2026-09-22: stopping it released RT-222 run 1 at once).

## The one hypothesis

**All ten policies play the seed-30 table in target width end to end through
`run_eval.ps1`, every run passes G1–G3, and `eval_collect.py` accepts block
T-auswahl with the same `code_commit` as T-nominal.**

## Command (training PC, after RT-222 printed "block T-nominal done")

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
git rev-parse HEAD
# expected: 8299485d0a60dd0cca01a7d84946f441a96c1100
.\scripts\run_eval.ps1 -Block T-auswahl -Rt RT-223 -DryRun
# expected: "table scripts/eval/eval_table_seed30.csv, width target, seed 100", "DRY RUN done"
.\scripts\run_eval.ps1 -Block T-auswahl -Rt RT-223
```

Each run: `play.py ... --seed 100 --checkpoint <run folder>\model_1499.pt
--eval-table scripts/eval/eval_table_seed30.csv --eval-width target`.

## Post-run

As RT-222, with `results\eval\T-auswahl\<autodr|nodr>_s<N>\`,
`rt_logs\RT-223\`, `--table scripts/eval/eval_table_seed30.csv` and
`--filehash D26610DC6838B4B2E2B453E267468A898DDE7F20BB2AF8291047AD9CF50A0ECE`.

## PASS/FAIL points, per run (CHECK_NAMES verbatim)

As RT-222 points 1–9, with hash `d26610dc…` and G3b/G3c against
`BANDS["target"]` (`check_eval_conditions.py:70`) instead of `"nominal"`.
Plus: each rt_log ends with `[rt_log] exit code: 0`; the block ends with
"block T-auswahl done: 10 of 10 runs, exit 0 each".

## Pins (`eval_manifest.json`, each of the ten)

| Field | Expected |
|---|---|
| `width` / `tag` | `"target"` / `""` |
| `seed` / `num_envs` / `table_rows` / `envs_finished_first_episode` | `100` / `1000` / `1000` / `1000` |
| `table_sha256` | `d26610dc6838b4b2e2b453e267468a898dde7f20bb2af8291047ad9cf50a0ece` |
| `checkpoint_sha256` | = the same policy's value in RT-222 |
| `code_commit` / `code_dirty` | `8299485d0a60dd0cca01a7d84946f441a96c1100` / `false` |
| `plan_commit` | `391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3` |
| `normalizer_sha256_before` = `_after` | equal (G5) |
| `head.txt` | `8299485d0a60dd0cca01a7d84946f441a96c1100` |
| `wall_time_s`, success rates | UNKNOWN — record, do not judge |

## What this run cannot answer

* The final policy: the selection rule is applied on the laptop afterwards
  and committed BEFORE T-final.
* The method comparison (Hauptzahl A/B): that is T-final (§ 3.2).
* Anything about RT-222 — its check comes separately.
* Which Isaac call spawns `hub.exe`, and why training runs did not hang.

## CORRECTION, after the first attempt (2026-09-22)

Only run 1 (`autodr_s1`) ran: manifest written (`wall_time_s` 848), then the
run hung on `hub.exe` (watcher was off) and runs 2..10 never started. The
folder has no log, `head.txt` or `table_filehash.txt`. Not data.

Repeat the WHOLE block, still on `8299485d` (no pull, no code change, so the
code_commit stays equal to RT-222). Before it, on the training PC, move the
first attempt aside (rename, never delete):

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
Rename-Item "logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_target_d26610dc" "eval_target_d26610dc_hung"
Get-ChildItem rt_logs -Filter 'RT-223_*' | Rename-Item -NewName { $_.Name -replace '^RT-223_', 'RT-223hung_' }
```

Then the watcher in a second window, then the Command section unchanged.
Expected: `autodr_s1` gives the same episodes as the first attempt (same
policy, table, seed; RT-220 G6 = 0 of 1000 rows differ).
