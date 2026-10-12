# RT-232 — blocks T-nominal-F and T-final-F, the five Fixed-DR policies (eval_plan.md § 3.2). Written BEFORE the run (2026-09-24)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Written in the main chat with the `rt-expectation` skill. Templates:
`rt_logs/RT-224_expectation.md` (ten-policy block, pins, post-run) and
`rt_logs/RT-225_expectation.md` (single `rt_log.ps1` calls instead of
`run_eval.ps1`). Only what changes is argued here.

Number: RT-226..RT-230 are p6's, RT-231 is the Fixed-DR training
(`git log --all` and `git ls-tree` over every `origin/*` branch,
2026-09-24: highest RT-231). One number for both blocks; the ten rt_log names
carry the block.

## Context

RT-231s1..s5 trained the Fixed-DR arm (D-192): PASS on its hypothesis
(`rt_logs/VERDICTS.md`, 2026-09-24 02:23:49). K5 (`eval_plan.md` § 1.4,
pre-registered before RT-231 in `bd0e86e1`) needs these five policies played
on the SAME seed-10 table as AutoDR in T-final (RT-224) and T-nominal
(RT-222): blocks T-final-F and T-nominal-F, 5 runs each (`eval_plan.md` § 3.2
table, rows "T-nominal-F (nachträglich)" / "T-final-F (nachträglich)").
The AutoDR and No-DR runs are NOT repeated (§ 3.2).

Why single commands and not `run_eval.ps1`: `run_eval.ps1` refuses `*-F`
(HANDOFF-RL.md, chat 3), and on `8299485d` it knows only the ten AutoDR/No-DR
checkpoints. User choice "A" (HANDOFF-RL.md NEXT 1d): one `rt_log.ps1` +
`play.py` call per policy, like RT-225, one loop per block. The loop repeats
by hand what `run_eval.ps1` does after each run on `8299485d`
(`scripts/run_eval.ps1:150-156`): full + short log, `table_filehash.txt`,
`head.txt` into the eval folder.

## Block → table, width, names

| Block | Table (`eval_collect.py:61-62`) | Width | Eval folder | rt_log names | Laptop folders |
|---|---|---|---|---|---|
| T-nominal-F | `scripts/eval/eval_table_seed10.csv` | `nominal` | `eval_nominal_1613635b` | `RT-232_nominal_fixeddr_s1`..`s5` | `results\eval\T-nominal-F\fixeddr_s1`..`s5` |
| T-final-F | `scripts/eval/eval_table_seed10.csv` | `target` | `eval_target_1613635b` | `RT-232_final_fixeddr_s1`..`s5` | `results\eval\T-final-F\fixeddr_s1`..`s5` |

* Table SHA-256 `1613635bd5f942e355d2e413deba25a5e97b368dce4798ad9030ed89cf27aba3`,
  1000 data rows: laptop `sha256sum` of `git show 8299485d:scripts/eval/eval_table_seed10.csv`
  and of the working copy, both the same (2026-09-24); `wc -l` = 1001 with
  the header. The SAME table as RT-222/RT-224.
* Eval folder name from `eval_manifest.eval_dirname` on `8299485d`
  (`scripts/tools/eval_manifest.py:81-93`, `eval_<width>_<sha8>`, no tag).
* Laptop folder names from `eval_collect.py:68`
  `_FOLDER = ^(autodr|nodr|fixeddr)_s(\d+)(?:_([A-Za-z0-9-]+))?$`; the
  collector holds the name against the checkpoint folder's `_fixed_dr_` tag
  (`eval_collect.py:67`) and refuses `fixeddr` outside the two `-F` blocks
  (`:115`).
* Checkpoints, the five RT-231 run folders (from
  `rt_logs/RT-231sN/RT-231sN_folder.txt`, written by the hand-back command on
  the training PC), each `\model_1499.pt`:
  * s1 `09-24_02-23-58__RT-231s1_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed1`
  * s2 `09-24_05-34-28__RT-231s2_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed2`
  * s3 `09-24_08-07-07__RT-231s3_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed3`
  * s4 `09-24_11-18-12__RT-231s4_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed4`
  * s5 `09-24_14-51-40__RT-231s5_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed5`

## Marker- und Code-Stand

**The training PC switches BACK to `8299485d0a60dd0cca01a7d84946f441a96c1100`**
(`eval_plan.md` § 3.2: "die Fixed-DR-Läufe spielen auf `8299485d`"), because
`eval_collect.py:218-221` refuses runs that disagree on `code_commit`, and
RT-222..RT-225 ran there. It is on `153b8b4` now (RT-231 rt_log headers).
Detached checkout, no pull; afterwards `git checkout -` returns it.

* env `CODE_MARKER = "insertion-osc-2026-09-03a"` (`insertion_env.py:76` on
  `8299485d`, `git show`), the same string as on `153b8b4` — the marker does
  NOT tell the two commits apart. The gate is `head.txt` and the manifest's
  `code_commit`.
* `plan_commit` = last commit on `8299485d` that touched `eval_plan.md` =
  `391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3` (`git log -1 --format=%H
  8299485d -- eval_plan.md`; `eval_manifest.py:31`).
* check script (laptop, afterwards) `SCRIPT_MARKER =
  "check_eval_conditions-2026-09-22a"` (`scripts/check_eval_conditions.py:46`).

A wrong commit voids the run; nothing after it is judged (precedent RT-151b,
`rt_logs/VERDICTS.md` 2026-09-05 14:51:11).

**Why `8299485d` can load a Fixed-DR checkpoint — by code reading only.**
`play.py` on `8299485d` sets `env_cfg.dr_mode = "table"` itself under
`--eval-table` (`play.py:320`), reads no `autodr_*.json` sidecar (no
`sidecar`/`autodr` hit in the file), and loads only the weights
(`runner.load(resume_path)`, `:579`) after the width guard (`:575-578`).
Fixed-DR trained with the same observation layout as AutoDR (both default
`obs_wrench_mode`; RT-231 changed only `dr_mode`). `eval_plan.md` § 3.2 marks
this UNVERIFIED until the first run — this run is that check.

The watcher stays on (`PROBLEMS.md` 2026-09-22: orphan `hub.exe` /
`omni.telemetry.transmitter.exe`).

## The one hypothesis

**On `8299485d` all five Fixed-DR checkpoints play the seed-10 table end to
end in nominal and in target width, each of the ten runs passes G1–G3, and
`eval_collect.py` accepts the six blocks together with one `code_commit`, so
`eval_stats.py` computes K5 and K5_nominal.**

Falsified by: a `[play] REFUSED` line or a Traceback at `runner.load` (the
checkpoint does not load on the old code); a run that ends before 1000 envs
finished; any G point FAIL; the collector naming a second `code_commit` or
refusing a `fixeddr` folder.

## Command (training PC)

Watcher in a second window first:

```powershell
while ($true) { Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('hub.exe','omni.telemetry.transmitter.exe') } | Where-Object { -not (Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue) } | ForEach-Object { "$(Get-Date -Format HH:mm:ss) stop $($_.Name) $($_.ProcessId)"; Stop-Process -Id $_.ProcessId }; Start-Sleep 15 }
```

Pre-checks, one at a time:

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
git status --porcelain --untracked-files=no
# expected: no output (code_dirty=false; the checkout needs it too)
git checkout --detach 8299485d0a60dd0cca01a7d84946f441a96c1100
git rev-parse HEAD
# expected: 8299485d0a60dd0cca01a7d84946f441a96c1100
Get-FileHash -Algorithm SHA256 .\scripts\eval\eval_table_seed10.csv | Format-List Hash
# expected: 1613635BD5F942E355D2E413DEBA25A5E97B368DCE4798AD9030ED89CF27ABA3
```

Existence check (checkpoint there, no eval folder, no rt_log file — rt_log
APPENDS, `play.py` refuses an existing `episodes.csv`):

```powershell
$r='C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1'; $L="$r\logs\rsl_rl\ur5e_insertion"; $F=@('09-24_02-23-58__RT-231s1_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed1','09-24_05-34-28__RT-231s2_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed2','09-24_08-07-07__RT-231s3_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed3','09-24_11-18-12__RT-231s4_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed4','09-24_14-51-40__RT-231s5_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed5'); $bad=0; foreach ($i in 0..4) { $s=$i+1; $d="$L\$($F[$i])"; if (-not (Test-Path "$d\model_1499.pt")) { "s$s no checkpoint"; $bad++ }; foreach ($w in 'nominal','target') { if (Test-Path "$d\eval_${w}_1613635b") { "s$s eval_${w}_1613635b exists"; $bad++ } }; foreach ($b in 'nominal','final') { if (Test-Path "$r\rt_logs\RT-232_${b}_fixeddr_s$s.txt") { "s$s rt_log RT-232_${b}_fixeddr_s$s exists"; $bad++ } } }; "problems: $bad"
```

Expected: `problems: 0`.

Block T-nominal-F (five runs, one after another):

```powershell
$r='C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1'; $L="$r\logs\rsl_rl\ur5e_insertion"; $F=@('09-24_02-23-58__RT-231s1_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed1','09-24_05-34-28__RT-231s2_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed2','09-24_08-07-07__RT-231s3_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed3','09-24_11-18-12__RT-231s4_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed4','09-24_14-51-40__RT-231s5_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed5'); foreach ($i in 0..4) { $s=$i+1; $n="RT-232_nominal_fixeddr_s$s"; $d="$L\$($F[$i])"; .\scripts\rt_log.ps1 $n python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 100 --checkpoint "$d\model_1499.pt" --eval-table scripts/eval/eval_table_seed10.csv --eval-width nominal; $e="$d\eval_nominal_1613635b"; if (Test-Path $e) { Copy-Item "$r\rt_logs\$n.txt" $e -Force; if (Test-Path "$r\rt_logs\$n.short.txt") { Copy-Item "$r\rt_logs\$n.short.txt" $e -Force }; (Get-FileHash -Algorithm SHA256 "$r\scripts\eval\eval_table_seed10.csv").Hash | Set-Content "$e\table_filehash.txt" -Encoding ascii; (git -C $r rev-parse HEAD) | Set-Content "$e\head.txt" -Encoding ascii; "$n done" } else { "$n NO EVAL FOLDER -- STOP"; break } }
```

Block T-final-F — the same loop, only `nominal` → `target` in width and
folder, and the name `final`:

```powershell
$r='C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1'; $L="$r\logs\rsl_rl\ur5e_insertion"; $F=@('09-24_02-23-58__RT-231s1_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed1','09-24_05-34-28__RT-231s2_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed2','09-24_08-07-07__RT-231s3_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed3','09-24_11-18-12__RT-231s4_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed4','09-24_14-51-40__RT-231s5_fixed_dr_offset5mm_currOFF_start+20mm_floor-30mm_seed5'); foreach ($i in 0..4) { $s=$i+1; $n="RT-232_final_fixeddr_s$s"; $d="$L\$($F[$i])"; .\scripts\rt_log.ps1 $n python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 100 --checkpoint "$d\model_1499.pt" --eval-table scripts/eval/eval_table_seed10.csv --eval-width target; $e="$d\eval_target_1613635b"; if (Test-Path $e) { Copy-Item "$r\rt_logs\$n.txt" $e -Force; if (Test-Path "$r\rt_logs\$n.short.txt") { Copy-Item "$r\rt_logs\$n.short.txt" $e -Force }; (Get-FileHash -Algorithm SHA256 "$r\scripts\eval\eval_table_seed10.csv").Hash | Set-Content "$e\table_filehash.txt" -Encoding ascii; (git -C $r rev-parse HEAD) | Set-Content "$e\head.txt" -Encoding ascii; "$n done" } else { "$n NO EVAL FOLDER -- STOP"; break } }
```

Smoke: the first run (`RT-232_nominal_fixeddr_s1`) IS the loading check. If
its log shows `[play] REFUSED` or a Traceback, stop — nothing else runs.

Pack, then return the clone to its branch:

```powershell
$L='C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion'; $o='C:\Users\Simon\Desktop\RT-232'; New-Item -ItemType Directory -Force "$o\T-nominal-F","$o\T-final-F" | Out-Null; Get-ChildItem $L -Directory | Where-Object { $_.Name -like '*__RT-231s*_fixed_dr_*' } | ForEach-Object { $s = [regex]::Match($_.Name,'seed(\d+)$').Groups[1].Value; if (Test-Path "$($_.FullName)\eval_nominal_1613635b") { Copy-Item "$($_.FullName)\eval_nominal_1613635b" "$o\T-nominal-F\fixeddr_s$s" -Recurse }; if (Test-Path "$($_.FullName)\eval_target_1613635b") { Copy-Item "$($_.FullName)\eval_target_1613635b" "$o\T-final-F\fixeddr_s$s" -Recurse } }; Get-ChildItem $o -Directory -Recurse -Depth 1 | Where-Object { $_.Name -like 'fixeddr_s*' } | ForEach-Object { "$($_.Parent.Name)\$($_.Name)" + $(if (Test-Path "$($_.FullName)\head.txt") {''} else {'  (no head.txt)'}) }; Compress-Archive "$o\*" "$o.zip" -Force
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1; git checkout -; git rev-parse --short HEAD
```

Expected: ten lines `T-nominal-F\fixeddr_s1`..`T-final-F\fixeddr_s5`, none
with `(no head.txt)`; after `git checkout -` the short HEAD of the branch
(`153b8b4` or later).

Expected per eval folder: `applied.csv`, `applied_meta.json`, `episodes.csv`,
`eval_manifest.json`, `RT-232_<block>_fixeddr_sN.txt` (+ `.short.txt`),
`table_filehash.txt`, `head.txt`.

## Post-run — laptop, in this worktree

The zip arrives in `C:\Users\PCUser\Downloads\RT-232.zip`:

```powershell
$x = Join-Path $env:TEMP 'rt232x'; if (Test-Path $x) { Remove-Item -Recurse -Force $x }; Expand-Archive 'C:\Users\PCUser\Downloads\RT-232.zip' $x; foreach ($b in 'T-nominal-F','T-final-F') { New-Item -ItemType Directory -Force "results\eval\$b" | Out-Null; Get-ChildItem "$x\$b" -Directory | ForEach-Object { Copy-Item $_.FullName "results\eval\$b\$($_.Name)" -Recurse } }; New-Item -ItemType Directory -Force rt_logs\RT-232 | Out-Null; Get-ChildItem results\eval\T-nominal-F,results\eval\T-final-F -Recurse -Filter 'RT-232_*' | Copy-Item -Destination rt_logs\RT-232\; Get-ChildItem rt_logs\RT-232 | Select-Object Name
```

The nine G points per run (all ten `VERDICT: PASS`, exit 0):

```powershell
foreach ($p in @(@('T-nominal-F','nominal'),@('T-final-F','final'))) { Get-ChildItem "results\eval\$($p[0])" -Directory | ForEach-Object { $n = $_.Name; $fh = (Get-Content "$($_.FullName)\table_filehash.txt").Trim(); $out = python scripts/check_eval_conditions.py --run-dir "results/eval/$($p[0])/$n" --table scripts/eval/eval_table_seed10.csv --log "rt_logs/RT-232/RT-232_$($p[1])_$n.txt" --filehash $fh 2>&1; "$($p[0])/$n  exit $LASTEXITCODE  " + (($out | Select-String -Pattern 'VERDICT').Line -join '') } }
```

The manifest pins (expect `pin mismatches: 0`):

```powershell
$bad = 0; foreach ($p in @(@('T-nominal-F','nominal'),@('T-final-F','target'))) { Get-ChildItem "results\eval\$($p[0])" -Directory | ForEach-Object { $m = Get-Content "$($_.FullName)\eval_manifest.json" -Raw | ConvertFrom-Json; $h = (Get-Content "$($_.FullName)\head.txt").Trim(); $ok = ($m.seed -eq 100) -and ($m.width -eq $p[1]) -and ($m.tag -eq '') -and ($m.code_dirty -eq $false) -and ($m.code_commit -eq '8299485d0a60dd0cca01a7d84946f441a96c1100') -and ($m.plan_commit -eq '391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3') -and ($m.num_envs -eq 1000) -and ($m.envs_finished_first_episode -eq 1000) -and ($m.normalizer_sha256_before -eq $m.normalizer_sha256_after) -and ($h -eq $m.code_commit) -and ($m.table_sha256 -eq '1613635bd5f942e355d2e413deba25a5e97b368dce4798ad9030ed89cf27aba3'); if (-not $ok) { $bad++; "PIN MISMATCH: $($p[0])/$($_.Name)" } } }; "pin mismatches: $bad"
```

Then the whole chain, one command (`eval_report.py` stages the four blocks
plus both `-F` blocks on a temp copy, never `technik`; `eval_report.py:9-15`):

```powershell
.venv-analysis/Scripts/python.exe scripts/eval_report.py
```

Expected: collector `VERDICT OK`, **41050 rows** (31050 of RT-225 + 5 × 1000
+ 5 × 1000), ONE `code_commit`. K1, K2, K3 must not move against
`rt_logs/VERDICTS.md` RT-224/RT-225 — their files are untouched. **Read K5 and
K5_nominal out of `results/eval/summary.json`** (keys `K5`, `K5_nominal`,
field `reading` ∈ `above_0` / `below_0` / `contains_0`, `eval_stats.py:153-159`),
not out of the console prose.

## PASS/FAIL points, per run (CHECK_NAMES verbatim, `scripts/check_eval_conditions.py:90-100`)

As RT-222/RT-224 points 1–9, hash `1613635b…`, 1000 rows; G3b/G3c against
`BANDS["nominal"]` for T-nominal-F and `BANDS["target"]` for T-final-F.

| # | Name |
|---|---|
| 1 | `G1a table bytes = sidecar = manifest = applied_meta` |
| 2 | `G1b the run log's [eval-table] line carries the same hash` |
| 3 | `G1c the training PC's Get-FileHash is the same hash` |
| 4 | `G2a one episode-0 row per env id 0..N-1` |
| 5 | `G2b N = table rows = manifest num_envs = envs that finished` |
| 6 | `G3a applied.csv has the agreed columns and one row per table row, in order` |
| 7 | `G3b the env's bands are the plan's bands for this width` |
| 8 | `G3c every applied value = the table row mapped, within 1e-6` |
| 9 | `G3d friction read back from PhysX = commanded, on both assets` |

Block sentences for the verdict: "block T-nominal-F done: 5 of 5 runs, exit 0
each"; "block T-final-F done: 5 of 5 runs, exit 0 each"; collector over six
blocks: no problem.

## Pins (`eval_manifest.json`, each of the ten)

| Field | Expected | Where the value comes from |
|---|---|---|
| `width` / `tag` | `"nominal"` (T-nominal-F) or `"target"` (T-final-F) / `""` | `eval_collect.py:61-62` |
| `seed` | `100` | `eval_plan.md` § 3.2 "Abspiel-Seed 100" (Fixed-DR paragraph) |
| `num_envs` / `table_rows` / `envs_finished_first_episode` | `1000` / `1000` / `1000` | 1000 data rows; `play.py:319` sets num_envs to them |
| `table_sha256` | `1613635bd5f942e355d2e413deba25a5e97b368dce4798ad9030ed89cf27aba3` | laptop SHA-256, see above |
| `checkpoint_sha256` | UNKNOWN — record; must be the SAME for seed N in T-nominal-F and T-final-F | one `model_1499.pt` per seed |
| `code_commit` / `code_dirty` / `head.txt` | `8299485d…` / `false` / `8299485d…` | § 3.2, RT-222..RT-225 |
| `plan_commit` | `391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3` | `git log -1 8299485d -- eval_plan.md` |
| `normalizer_sha256_before` = `_after` | equal (G5) | RT-209 rule |
| `episode_cap_steps` / `control_mode` / `obs_wrench_mode` | `256` / `osc` / `force` | `results/eval/T-final/autodr_s1/eval_manifest.json`, same env |
| `wall_time_s`, success rates | UNKNOWN — record | T-final (RT-224) took 225.7–431.3 s per run; another policy, reference only |

## Expectation about the numbers, written BEFORE the run

The K5 hypothesis and its three readings are pre-registered in
`eval_plan.md` § 1.4 (`bd0e86e1`, before RT-231). Nothing is added here.

Written down so the run cannot be re-read as a surprise — a hint, not a
prediction: in TRAINING, Fixed-DR's success over the last 50 iterations was
s1 0.00, s2 0.00, s3 0.94, s4 0.99, s5 0.95 (`rt_logs/RT-231sN/RT-231sN_scalars.csv`,
`Episode/success_rate`, iterations 1450–1499). Training rates carry PPO's
exploration noise and another start distribution; the plan forbids reading
them as test rates. **No seed is dropped from K5** — its input is "5 AutoDR- + 5
Fixed-DR-Seed-Raten" (`eval_plan.md` § 5 point 7); s1 and s2 are Fixed-DR
outcomes like the others.

## What this run cannot answer

* Whether growth or boundary sampling causes a K5 difference: Fixed-DR changes
  both at once (`eval_plan.md` § 3.1, "ZWEI Unterschiede").
* Whether Fixed-DR would learn with more iterations or other seeds: 1500
  iterations, 5 seeds is the budget.
* K1–K4: untouched; the four earlier blocks are not replayed.
* Why s1 and s2 did not learn in training: not examined here (s2's entropy
  fell 8.5 → 1.1, s1's stayed at 7.3 — two different pictures).
* Sim-to-real anything: every number is simulation.

## Follow-up, not part of this run

* `eval_collect.py` self-test holds a CONSTRUCTED RT-231 folder name
  (`eval_collect.py:399-401`); replace it with the real s1 name above — own
  commit, `check_*` self-test of that file only.
