# RT-225 — block T-grenze, final policy only (eval_plan.md § 3.2). Written BEFORE the run (2026-09-22)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Written in the main chat with the `rt-expectation` skill. Template:
`rt_logs/RT-224_expectation.md`. Only what changes is argued here.

## Context

T-grenze is the last block of `eval_plan.md` § 3.2 and the only evidence for
**K3** (§ 1.4): "Für keinen der 7 Grenzblöcke liegt das Intervall (Block −
Gesamtrate T-final, Newcombe-Differenz zweier Wilson-Intervalle) ganz unter 0."

Two things make this block different from RT-222/223/224:

* **One policy, not ten.** K3 is about the final policy: **AutoDR seed 1**
  (D-190, selected on T-auswahl before T-final ran). `run_eval.ps1` REFUSES
  this block on purpose (`scripts/run_eval.ps1:82`:
  `if ($Block -in @('T-grenze', 'technik')) { Stop-Block ... }`), so this is a
  single `rt_log.ps1` call, and the three files `run_eval.ps1` normally drops
  into the eval folder (the log, `table_filehash.txt`, `head.txt`) are written
  by hand here.
* **Another table.** `eval_collect.py:54`:
  `"T-grenze": ("eval_table_boundary_seed10.csv", "target")`.
  The table has **1050 rows** (7 blocks × 150), not 1000
  (`scripts/eval/eval_table_boundary_seed10.json`: `"rows_per_block": 150`,
  seven `blocks` entries `lat_r_hi, yaw_lo, yaw_hi, tilt_hi, start_height_hi,
  friction_lo, friction_hi`). `play.py:319` sets
  `env_cfg.scene.num_envs = _table_rows`, so **num_envs = 1050** without a
  `--num_envs` flag.

K3 cannot be computed before T-final exists: `eval_stats.py:164` measures each
boundary block AGAINST the overall T-final rate of the same policy.

## Block → table, width, folder names

* Table `scripts/eval/eval_table_boundary_seed10.csv`, SHA-256
  `03623206a216bde98e1d15dc11097281e8de1d985a6458ce341c2b0f586fb1a8`
  (computed on the laptop from the committed bytes; the training PC's
  `Get-FileHash` must return the same — that IS G1c).
* Eval folder name `eval_target_03623206`
  (`scripts/tools/eval_manifest.py:81-93`, `eval_<width>_<sha8>`, no tag).
* rt_log name `RT-225`; laptop folder `results\eval\T-grenze\autodr_s1`
  (`eval_collect.py:58` `_FOLDER` = `^(autodr|nodr)_s(\d+)(?:_([A-Za-z0-9-]+))?$`).

## Marker- und Code-Stand

**NO pull.** The training PC stays on
`8299485d0a60dd0cca01a7d84946f441a96c1100`, like RT-222/223/224, because
`eval_collect.py:202-205` compares `code_commit` over ALL runs of all blocks.
`plan_commit` stays `391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3`.

Markers expected in the log:

* env `CODE_MARKER = "insertion-osc-2026-09-03a"`
  (`source/insertion/insertion/tasks/direct/insertion/insertion_env.py:76`,
  printed at `:4847`).
* check script (laptop, afterwards) `SCRIPT_MARKER =
  "check_eval_conditions-2026-09-22a"` (`scripts/check_eval_conditions.py:46`).

A wrong marker voids the run; nothing after it is judged (precedent RT-151b,
`rt_logs/VERDICTS.md` 2026-09-05 14:51:11).

The watcher stays on: an orphan `hub.exe` OR
`omni.telemetry.transmitter.exe` holds rt_log's pipe after every run
(`PROBLEMS.md` 2026-09-22).

## The one hypothesis

**AutoDR seed 1 plays all 1050 rows of the boundary table in target width end
to end, the run passes G1–G3, and its `eval_manifest.json` carries the same
`code_commit`, `plan_commit`, `checkpoint_sha256` and `seed` as its T-final
run, so `eval_collect.py` accepts all four blocks together and
`eval_stats.py` computes K3.**

## Command (training PC)

Watcher in a second window first (both process names, `PROBLEMS.md`
2026-09-22). Then, in one PowerShell window:

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
git rev-parse HEAD
# expected: 8299485d0a60dd0cca01a7d84946f441a96c1100
git status --porcelain --untracked-files=no
# expected: no output (that is what code_dirty=false means)
Get-FileHash -Algorithm SHA256 .\scripts\eval\eval_table_boundary_seed10.csv | Format-List Hash
# expected: 03623206A216BDE98E1D15DC11097281E8DE1D985A6458CE341C2B0F586FB1A8
Test-Path 'C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_target_03623206'
# expected: False  (an existing folder would be a second run mixed in)
Test-Path 'C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\rt_logs\RT-225.txt'
# expected: False  (rt_log APPENDS to an existing file)
```

The run:

```powershell
.\scripts\rt_log.ps1 RT-225 python scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --seed 100 --checkpoint C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt --eval-table scripts/eval/eval_table_boundary_seed10.csv --eval-width target
```

Right after the run (this replaces what `run_eval.ps1` does for the ten-policy
blocks — log, table hash and HEAD into the eval folder, so the laptop copies
ONE folder):

```powershell
$e = 'C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\eval_target_03623206'; $r = 'C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1'; Copy-Item "$r\rt_logs\RT-225.txt" "$e\RT-225.txt" -Force; if (Test-Path "$r\rt_logs\RT-225.short.txt") { Copy-Item "$r\rt_logs\RT-225.short.txt" "$e\RT-225.short.txt" -Force }; (Get-FileHash -Algorithm SHA256 "$r\scripts\eval\eval_table_boundary_seed10.csv").Hash | Set-Content "$e\table_filehash.txt" -Encoding ascii; (git -C $r rev-parse HEAD) | Set-Content "$e\head.txt" -Encoding ascii; Compress-Archive "$e" 'C:\Users\Simon\Desktop\RT-225_T-grenze.zip' -Force; Get-ChildItem $e | Select-Object Name
```

Expected in the folder: `applied.csv`, `applied_meta.json`, `episodes.csv`,
`eval_manifest.json`, `RT-225.txt`, `table_filehash.txt`, `head.txt`.

## Post-run — laptop, in this worktree

The zip arrives in `C:\Users\PCUser\Downloads\RT-225_T-grenze.zip`:

```powershell
$z = 'C:\Users\PCUser\Downloads\RT-225_T-grenze.zip'; $x = Join-Path $env:TEMP 'rt225x'; if (Test-Path $x) { Remove-Item -Recurse -Force $x }; Expand-Archive $z $x; New-Item -ItemType Directory -Force results\eval\T-grenze | Out-Null; Copy-Item (Join-Path $x 'eval_target_03623206') results\eval\T-grenze\autodr_s1 -Recurse; New-Item -ItemType Directory -Force rt_logs\RT-225 | Out-Null; Copy-Item results\eval\T-grenze\autodr_s1\RT-225*.txt rt_logs\RT-225\; Get-ChildItem results\eval\T-grenze\autodr_s1 | Select-Object Name
```

The nine G points:

```powershell
$fh = (Get-Content 'results\eval\T-grenze\autodr_s1\table_filehash.txt').Trim(); python scripts/check_eval_conditions.py --run-dir results/eval/T-grenze/autodr_s1 --table scripts/eval/eval_table_boundary_seed10.csv --log rt_logs/RT-225/RT-225.txt --filehash $fh; "exit $LASTEXITCODE"
```

Collector and statistics on a COPY that holds only the four real blocks
(`results/eval/technik/` carries foreign folders and older commits and makes
the collector refuse the whole folder):

```powershell
$t = Join-Path $env:TEMP 'rt225c'; if (Test-Path $t) { Remove-Item -Recurse -Force $t }; New-Item -ItemType Directory $t | Out-Null; 'T-nominal','T-auswahl','T-final','T-grenze' | ForEach-Object { Copy-Item "results\eval\$_" (Join-Path $t $_) -Recurse }; .venv-analysis/Scripts/python.exe scripts/eval_collect.py --results $t; .venv-analysis/Scripts/python.exe scripts/eval_stats.py --results $t
```

Expected: `VERDICT OK`, 31050 rows (30000 + 1050), ONE `code_commit`; then K3
for the seven boundary keys. K1 and K2 must not move — the T-nominal,
T-auswahl and T-final files are untouched. **Read K3 out of `summary.json`,
not out of the console prose.**

## PASS/FAIL points, in script order (CHECK_NAMES verbatim,
`scripts/check_eval_conditions.py:90-99`)

| # | Name | PASS when | FAIL means |
|---|---|---|---|
| 1 | `G1a table bytes = sidecar = manifest = applied_meta` | all four carry `03623206a216…fb1a8` | a different table was played |
| 2 | `G1b the run log's [eval-table] line carries the same hash` | at least one `[eval-table]` line, EVERY such line the same hash | the env read another file |
| 3 | `G1c the training PC's Get-FileHash is the same hash` | `--filehash` equals it | the clone's table differs from the committed one |
| 4 | `G2a one episode-0 row per env id 0..N-1` | exactly one `episode == 0` row for each of 0..1049 | rows missing or doubled |
| 5 | `G2b N = table rows = manifest num_envs = envs that finished` | 1050 = 1050 = 1050 = 1050 | not every row was played to the end |
| 6 | `G3a applied.csv has the agreed columns and one row per table row, in order` | 17 columns, 1050 rows, `row_id` ascending | the writer or the order changed |
| 7 | `G3b the env's bands are the plan's bands for this width` | `applied_meta.json` bands = `BANDS["target"]` (`check_eval_conditions.py:70-77`) | the env ran another width |
| 8 | `G3c every applied value = the table row mapped, within 1e-6` | worst deviation < 1e-6 | the mapping in the env is not the plan's |
| 9 | `G3d friction read back from PhysX = commanded, on both assets` | read-back equals `friction_cmd` on robot and fixture | the material did not take the value |

Block sentence for the verdict: "block T-grenze done: 1 of 1 run, exit 0";
collector over all four blocks: no problem outside `technik`.

## Pins (`results/eval/T-grenze/autodr_s1/eval_manifest.json`)

| Field | Expected | Where the value comes from |
|---|---|---|
| `width` / `tag` | `"target"` / `""` | `eval_collect.py:54` |
| `seed` | `100` | `eval_plan.md` § 3.2 "Abspiel-Seed"; same as RT-222/223/224 |
| `num_envs` / `table_rows` / `envs_finished_first_episode` | `1050` / `1050` / `1050` | `eval_table_boundary_seed10.json` `n_rows`, `play.py:319` |
| `table_sha256` | `03623206a216bde98e1d15dc11097281e8de1d985a6458ce341c2b0f586fb1a8` | laptop SHA-256 of the committed CSV |
| `checkpoint_sha256` | `b7e4ca87516d332a2e5ed4aea1e8f0040cc9ba3cf61f26940af8c47523d66394` | `results/eval/T-auswahl/autodr_s1/eval_manifest.json` (same file must be replayed) |
| `code_commit` / `code_dirty` / `head.txt` | `8299485d…` / `false` / `8299485d…` | no pull; RT-222/223/224 |
| `plan_commit` | `391c7ae47d1d6e42a2ce1ccda40f53458aaefdb3` | RT-222/223/224 manifests |
| `normalizer_sha256_before` = `_after` | equal (G5) | RT-209 rule |
| `episode_cap_steps` / `control_mode` / `obs_wrench_mode` | `256` / `osc` / `force` | T-auswahl manifest, same env |
| `wall_time_s` | UNKNOWN — record it | T-auswahl autodr_s1 took 644.3 s at 1000 envs; 1050 envs is a DIFFERENT table, reference only |
| success rates | UNKNOWN — record | — |

## Expectation about the numbers, written BEFORE the run

None. K3 asks whether any one boundary block falls out, and no boundary block
has ever been played. Writing a guess here would only make the outcome
re-readable as a confirmation. What IS fixed beforehand: K3 is met when, for
all seven keys, the Newcombe interval (block rate − overall T-final rate of
the same policy) is NOT wholly below 0 (`eval_plan.md` § 1.4).

## What this run cannot answer

* K1 and K2: those are T-final (RT-224) and T-nominal (RT-222). This run
  touches neither file.
* Whether another policy would be steadier at a boundary: one policy runs
  here. All ten is option O6 in `eval_plan.md` § 9 and is not done.
* Which of the seven boundaries is "the" weak one in a causal sense: the
  boundary blocks nail ONE dimension each and draw the rest, so a low block
  rate names a correlation, not a cause.
* Sim-to-real anything: every number is simulation.
* The start-state penetration before the first action is still not recorded
  (`InBachelorErwähnen.md`, 2026-09-22).
