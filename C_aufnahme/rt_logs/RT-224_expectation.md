# RT-224 — block T-final, all ten policies (eval_plan.md § 3.2). Written BEFORE the run (2026-09-22)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Written in the main chat with the `rt-expectation` skill. Template:
`rt_logs/RT-223_expectation.md`. Only what changes is argued here.

## Context

T-nominal (RT-222) and T-auswahl (RT-223) are PASS, both on `8299485d`
(`rt_logs/VERDICTS.md`). The selection is recorded BEFORE this run, as
`eval_plan.md` § 3.2 demands: **D-190, final policy AutoDR seed 1**. T-final
is the main number: block A (method comparison, K1) and the absolute rate of
the final policy.

## Block → table, width, names

* `scripts/eval_collect.py:53`: `"T-final": ("eval_table_seed10.csv", "target")`.
* Table `scripts/eval/eval_table_seed10.csv`, hash
  `1613635BD5F942E355D2E413DEBA25A5E97B368DCE4798AD9030ED89CF27ABA3`
  — the SAME table as T-nominal, but target width, so the eval folder is
  `<run folder>\eval_target_1613635b` and cannot collide with T-nominal's
  `eval_nominal_1613635b`.
* rt_log names `RT-224_<autodr|nodr>_s<N>`; laptop folders
  `results\eval\T-final\<autodr|nodr>_s<N>`.

## Marker- und Code-Stand

**NO pull.** The training PC stays on
`8299485d0a60dd0cca01a7d84946f441a96c1100`, so all three blocks share one
`code_commit` (`eval_collect.py:202-205` compares it over ALL runs).
`plan_commit` stays `391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3`.

The watcher stays on: an orphan `hub.exe` OR
`omni.telemetry.transmitter.exe` holds rt_log's pipe after every run
(`PROBLEMS.md` 2026-09-22).

## The one hypothesis

**All ten policies play the seed-10 table in target width end to end, every
run passes G1–G3, and `eval_collect.py` accepts T-nominal + T-auswahl +
T-final with one `code_commit`; `eval_stats.py` then computes K1.**

## Command (training PC)

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
git rev-parse HEAD
# expected: 8299485d0a60dd0cca01a7d84946f441a96c1100
.\scripts\run_eval.ps1 -Block T-final -Rt RT-224 -DryRun
# expected: "table scripts/eval/eval_table_seed10.csv, width target, seed 100",
#           "eval folder name eval_target_1613635b", "DRY RUN done"
.\scripts\run_eval.ps1 -Block T-final -Rt RT-224
```

Watcher in the second window (both process names, `PROBLEMS.md`).

## Post-run — the exact steps (they worked for RT-222 and RT-223)

**Training PC**, watcher stopped, packs the ten eval folders under the names
the collector wants:

```powershell
$r='C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion'; $o='C:\Users\Simon\Desktop\RT-224_T-final'; New-Item -ItemType Directory -Force $o | Out-Null; Get-ChildItem $r -Directory | ForEach-Object { $e = Join-Path $_.FullName 'eval_target_1613635b'; if (Test-Path $e) { $m = if ($_.Name -match '_autodr_') {'autodr'} else {'nodr'}; $s = [regex]::Match($_.Name,'seed(\d+)$').Groups[1].Value; Copy-Item $e (Join-Path $o "${m}_s$s") -Recurse } }; Get-ChildItem $o -Directory | ForEach-Object { $_.Name + $(if (Test-Path "$($_.FullName)\head.txt") {''} else {'  (no head.txt)'}) }; Compress-Archive "$o\*" "$o.zip"
```

Expected: ten names `autodr_s1`..`nodr_s5`, none with `(no head.txt)`.

**Laptop**, in this worktree, after unpacking the zip to `<X>`:

```powershell
Get-ChildItem <X> -Directory | ForEach-Object { Copy-Item $_.FullName (Join-Path 'results\eval\T-final' $_.Name) -Recurse }
Get-ChildItem results\eval\T-final -Directory | ForEach-Object { Get-ChildItem $_.FullName -Filter 'RT-224_*' | Copy-Item -Destination rt_logs\RT-224 }
```

Then the nine G-points per run (all ten must say `VERDICT: PASS`, exit 0):

```powershell
Get-ChildItem results\eval\T-final -Directory | ForEach-Object { $n = $_.Name; $fh = (Get-Content "$($_.FullName)\table_filehash.txt").Trim(); $out = python scripts/check_eval_conditions.py --run-dir "results/eval/T-final/$n" --table scripts/eval/eval_table_seed10.csv --log "rt_logs/RT-224/RT-224_$n.txt" --filehash $fh 2>&1; "$n  exit $LASTEXITCODE  " + (($out | Select-String -Pattern 'VERDICT').Line -join '') }
```

Then the manifest pins of all ten (expect `pin mismatches: 0`):

```powershell
$bad = 0; Get-ChildItem results\eval\T-final -Directory | ForEach-Object { $m = Get-Content "$($_.FullName)\eval_manifest.json" -Raw | ConvertFrom-Json; $h = (Get-Content "$($_.FullName)\head.txt").Trim(); $ok = ($m.seed -eq 100) -and ($m.width -eq 'target') -and ($m.tag -eq '') -and ($m.code_dirty -eq $false) -and ($m.code_commit -eq '8299485d0a60dd0cca01a7d84946f441a96c1100') -and ($m.plan_commit -eq '391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3') -and ($m.num_envs -eq 1000) -and ($m.envs_finished_first_episode -eq 1000) -and ($m.normalizer_sha256_before -eq $m.normalizer_sha256_after) -and ($h -eq $m.code_commit) -and ($m.table_sha256 -eq '1613635bd5f942e355d2e413deba25a5e97b368dce4798ad9030ed89cf27aba3'); if (-not $ok) { $bad++; "PIN MISMATCH: $($_.Name)" } }; "pin mismatches: $bad"
```

Collector and statistics run on a COPY that holds only the three real blocks,
because `results/eval/technik/` carries two foreign folders and older
commits and makes the collector refuse the whole folder:

```powershell
$t = Join-Path $env:TEMP 'rt224c'; if (Test-Path $t) { Remove-Item -Recurse -Force $t }; New-Item -ItemType Directory $t | Out-Null; 'T-nominal','T-auswahl','T-final' | ForEach-Object { Copy-Item "results\eval\$_" (Join-Path $t $_) -Recurse }; .venv-analysis/Scripts/python.exe scripts/eval_collect.py --results $t; .venv-analysis/Scripts/python.exe scripts/eval_stats.py --results $t
```

Expected: `VERDICT OK`, 30000 rows, ONE `code_commit`; then `K1 met: ...`
and `K2 met: True` (K2's numbers must not move -- T-nominal is unchanged).
Read K1 out of `summary.json`, not out of the console prose.

## PASS/FAIL points, per run (CHECK_NAMES verbatim)

As RT-222 points 1–9, hash `1613635b…`, G3b/G3c against `BANDS["target"]`
(`check_eval_conditions.py:70`). Block: "block T-final done: 10 of 10 runs,
exit 0 each"; collector over all three blocks: no problem outside `technik`.

## Pins (`eval_manifest.json`, each of the ten)

| Field | Expected |
|---|---|
| `width` / `tag` | `"target"` / `""` |
| `seed` / `num_envs` / `table_rows` / `envs_finished_first_episode` | `100` / `1000` / `1000` / `1000` |
| `table_sha256` | `1613635bd5f942e355d2e413deba25a5e97b368dce4798ad9030ed89cf27aba3` |
| `checkpoint_sha256` | = the same policy's value in RT-222 and RT-223 |
| `code_commit` / `code_dirty` / `head.txt` | `8299485d…` / `false` / `8299485d…` |
| `plan_commit` | `391c7ae4…` |
| `normalizer_sha256_before` = `_after` | equal (G5) |
| `wall_time_s`, success rates | UNKNOWN — record. T-auswahl (target, other table) took 388–881 s; reference only |

## Expectation about the numbers, written BEFORE the run

K1 is judged only here. On the SELECTION table (RT-223, other rows, same
width) AutoDR ran 0.953–0.979 and No-DR 0.196–0.315. That is a different
table and no evidence for T-final; it is written down so the run cannot be
re-read as a surprise afterwards. K1 is met when the Welch interval of the
difference lies wholly above 0 (`eval_plan.md` § 1.4).

## What this run cannot answer

* K3: that is T-grenze, final policy only (`run_eval.ps1` refuses that block
  — it is one policy; a single `rt_log.ps1` command).
* Whether the selected policy is the best on the final table: the selection
  is fixed by D-190 and is not revisited.
* Sim-to-real anything: every number is simulation.
