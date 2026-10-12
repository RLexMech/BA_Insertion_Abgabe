# RT-254 -- expectation, written BEFORE the run (2026-10-01): T-ood and the noise blocks pn2 / gx2 / py2 for AutoDR (RT-254..RT-257, `run_eval.ps1`) and Fixed-DR (RT-258, RT-259, rt_log loops), 40 runs

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Templates: `rt_logs/RT-247_expectation.md` (chain of `run_eval.ps1`
blocks on the same clone; OUTCOME PASS) and `rt_logs/RT-251_expectation.md`
(Fixed-DR as an `rt_log.ps1` loop; OUTCOME PASS). Only what differs is
written out.

## Context

D-206 (3): T-ood (the boundary table with its nailed cells moved outwards,
x1.5) and the observation-noise blocks (x2 of one training width each) for
AutoDR. D-208 point 2: Fixed-DR plays the same blocks, as rt_log loops on
the unchanged clone. Order D-208 point 3: after T-grenze, which is judged
(RT-252, `rt_logs/RT-252_expectation.md` OUTCOME). The PC was checked
2026-10-01 before this file: no `eval_target_02adce21*` and no
`eval_target_d4b97259_*` folder exists in any RT-242/243/244 run (user
listing in chat). These blocks are DESCRIPTIVE: no pass line
(`eval_stats.py` `ood_blocks`, `noise_blocks`).

## Block -> table, width, names

* T-ood: `scripts/eval_collect.py:70` (BLOCKS, same line at `c1a635e9`):
  `"T-ood": ("eval_table_ood_seed10_c18.csv", "target")`. Table sha256
  `02adce218ae2c98fba2b0eb1431ded8e4aa1fb897ada4b12e5837d61996e64f6`
  (= sidecar `eval_table_ood_seed10_c18.json` `sha256`, laptop `sha256sum`,
  and the file at `c1a635e9`), **1050 rows** (`n_rows`), 7 blocks x 150,
  factor 1.5, `start_height_hi` capped at 1.0 (sidecar `factor_cap`).
  Eval folder `eval_target_02adce21` (`scripts/tools/eval_manifest.py:81`
  `eval_dirname`). `run_eval.ps1:124-126`: autodr s1..s5; `:201` adds
  `--eval-ood`.
* Noise: `-Block T-final -Noise <tag>` (`run_eval.ps1:105`, `:128-129`):
  autodr s1..s5 on `eval_table_seed10_c18.csv`, sha256
  `d4b972597b357844d895fc43b03e776537cee45bc7b5fa1910ddfd19f8954f6d`, 1000
  rows; eval folder `eval_target_d4b97259_<tag>`. Overrides
  `run_eval.ps1:80-84`, `:202-204`: pn2 `env.obs_noise_pocket_pos_std_m=0.01`
  on top of the yaw belief; gx2 `env.grasp_obs_offset_x_m=0.006` on top;
  py2 `env.pocket_obs_yaw_rad=0.0446804` REPLACES `0.0223402`.
* Names: AutoDR `RT-254_autodr_s<N>` (T-ood), `RT-255_autodr_s<N>_pn2`,
  `RT-256_autodr_s<N>_gx2`, `RT-257_autodr_s<N>_py2` (`run_eval.ps1:158-159`).
  Fixed-DR `RT-258_fixeddr_s<N>` (T-ood), `RT-259_fixeddr_s<N>_<tag>`.
* Fixed-DR loop = `run_eval.ps1:197-204` argument for argument (order:
  `--eval-tag`, `--eval-ood`, `force_prevact`, yaw or py2, pn2/gx2), and
  `:217-222` after each run. Checkpoints as RT-251: s1 `rew_49.403553`,
  s2 `rew_47.91025`, s3 `rew_534.6653`, s4 `rew_570.71265`, s5
  `rew_87.923416` (`rt_logs/RT-243s/RT-243s<N>_nn.txt`; pattern
  `run_eval.ps1:73`, only `ep_100` matches).
* RT numbers: highest in the repo is RT-253 (`rt_logs/`, checked
  2026-10-01); RT-254..RT-259 are free.

## Marker- und Code-Stand

**NO pull.** Training PC on `c1a635e969f72efd5b9062d2c8994befd29ff100`.
`git diff --stat c1a635e9 HEAD -- scripts/rl_games/ source/ scripts/eval/
scripts/rt_log.ps1` is empty (laptop, 2026-10-01): play, env, tables and
rt_log are the PC's. Gates as RT-247: `[rt_log] git: c1a635e`,
`imported from: ...Phase5_RlGames_v1\source\insertion\insertion`,
`code_dirty` false, `plan_commit` `4b4457144e2c7bedcf22b432ba8e5988e6aa9c7a`.
T-ood runs: `[eval-table] ... 02adce21... Values outside [0, 1] allowed
(OOD table): True`. Noise runs: `... d4b97259... (OOD table): False`.

## The one hypothesis

**All 40 runs (AutoDR and Fixed-DR, T-ood and pn2 / gx2 / py2) play their
table end to end, every run passes G1a-G3e, and every manifest carries
`code_commit` `c1a635e9...`, its table hash, its tag, the tag's doubled
noise width, and `dr_table_allow_outside` true exactly in the ten T-ood
runs.**

Falsified by: a `STOP:` line; a run without `[rt_log] exit code: 0`; any
`CHECK_NAMES` FAIL; another `code_commit`, table hash or noise width. The
rates are recorded, not judged.

## Command (training PC, `env_isaaclab` active, GPU free)

### Step 1 -- AutoDR, dry run of the four blocks

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
git rev-parse HEAD
git status --short --untracked-files=no
.\scripts\rl_games\run_eval.ps1 -Block T-ood -Rt RT-254 -DryRun
.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-255 -Noise pn2 -DryRun
.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-256 -Noise gx2 -DryRun
.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-257 -Noise py2 -DryRun
```

Expected: HEAD `c1a635e969f72efd5b9062d2c8994befd29ff100`; status prints
nothing; T-ood `table scripts/eval/eval_table_ood_seed10_c18.csv, seed 100,
5 run(s)`, `Get-FileHash 02adce21...`, DRY RUN lines with `--eval-ood`;
each noise block `... eval_table_seed10_c18.csv, seed 100, 5 run(s), noise
<tag>`, `Get-FileHash d4b97259...`, DRY RUN lines with `--eval-tag <tag>`
and the tag's override; `DRY RUN done` four times. A `STOP:` line = do not
run; paste it.

### Step 2 -- AutoDR chain (20 runs)

```powershell
.\scripts\rl_games\run_eval.ps1 -Block T-ood -Rt RT-254; if ($LASTEXITCODE -eq 0) { .\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-255 -Noise pn2 }; if ($LASTEXITCODE -eq 0) { .\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-256 -Noise gx2 }; if ($LASTEXITCODE -eq 0) { .\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-257 -Noise py2 }; "chain exit $LASTEXITCODE"
```

Expected: four `block ... done: 5 of 5 runs, exit 0 each`, `chain exit 0`.

### Step 3 -- Fixed-DR pre-check (runs nothing)

```powershell
$r='C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1'; cd $r; $L="$r\logs\rl_games\ur5e_insertion"; $H=(git rev-parse HEAD); $bad=0; foreach ($s in 1..5) { $d=@(Get-ChildItem -LiteralPath $L -Directory -Filter "*__RT-243s${s}_*"); if ($d.Count -ne 1) { "s$s run folders: $($d.Count)"; $bad++; continue }; $c=@(Get-ChildItem -LiteralPath "$($d[0].FullName)\nn" -File | Where-Object { $_.Name -match '^last_ur5e_insertion_ep_100_rew_[-+0-9.eE]+\.pth$' }); if ($c.Count -ne 1) { "s$s checkpoints: $($c.Count)"; $bad++ } else { "s$s $($c[0].Name)" }; foreach ($e in 'eval_target_02adce21','eval_target_d4b97259_pn2','eval_target_d4b97259_gx2','eval_target_d4b97259_py2') { if (Test-Path "$($d[0].FullName)\$e") { "s$s $e exists"; $bad++ } }; foreach ($n in "RT-258_fixeddr_s$s","RT-259_fixeddr_s${s}_pn2","RT-259_fixeddr_s${s}_gx2","RT-259_fixeddr_s${s}_py2") { if (Test-Path "$r\rt_logs\$n.txt") { "$n rt_log exists"; $bad++ } } }; "HEAD $H"; "tracked changes: $(@(git status --porcelain --untracked-files=no).Count)"; "ood table $((Get-FileHash -Algorithm SHA256 "$r\scripts\eval\eval_table_ood_seed10_c18.csv").Hash)"; "final table $((Get-FileHash -Algorithm SHA256 "$r\scripts\eval\eval_table_seed10_c18.csv").Hash)"; "problems: $bad"
```

Expected: five lines `s<N> last_ur5e_insertion_ep_100_rew_<R>.pth` with the
five names above; `HEAD c1a635e969f72efd5b9062d2c8994befd29ff100`;
`tracked changes: 0`;
`ood table 02ADCE218AE2C98FBA2B0EB1431DED8E4AA1FB897ADA4B12E5837D61996E64F6`;
`final table D4B972597B357844D895FC43B03E776537CEE45BC7B5FA1910DDFD19F8954F6D`;
`problems: 0`. Anything else: do not run; paste the output.

### Step 4 -- Fixed-DR loop (20 runs: 5 T-ood, then 5 each pn2, gx2, py2), same window

```powershell
$jobs=@(); foreach ($sd in 1..5) { $jobs += ,@("RT-258_fixeddr_s$sd", $sd, 'scripts/eval/eval_table_ood_seed10_c18.csv', '', 'eval_target_02adce21') }; foreach ($tg in 'pn2','gx2','py2') { foreach ($sd in 1..5) { $jobs += ,@("RT-259_fixeddr_s${sd}_$tg", $sd, 'scripts/eval/eval_table_seed10_c18.csv', $tg, "eval_target_d4b97259_$tg") } }; foreach ($j in $jobs) { $n=$j[0]; $sd=$j[1]; $tab=$j[2]; $tag=$j[3]; $d=(Get-ChildItem -LiteralPath $L -Directory -Filter "*__RT-243s${sd}_*").FullName; $c=(Get-ChildItem -LiteralPath "$d\nn" -File | Where-Object { $_.Name -match '^last_ur5e_insertion_ep_100_rew_[-+0-9.eE]+\.pth$' }).FullName; if ((git rev-parse HEAD) -ne $H -or @(git status --porcelain --untracked-files=no).Count -ne 0) { "STOP: HEAD moved or tracked change before $n"; break }; $a=@('python','-u','scripts/rl_games/play.py','--task','Ur5e-Insertion-Direct-v0','--headless','--seed','100','--checkpoint',$c,'--eval-table',$tab,'--eval-width','target'); if ($tag) { $a+=@('--eval-tag',$tag) } else { $a+='--eval-ood' }; $a+='env.obs_wrench_mode=force_prevact'; if ($tag -eq 'py2') { $a+='env.pocket_obs_yaw_rad=0.0446804' } else { $a+='env.pocket_obs_yaw_rad=0.0223402' }; if ($tag -eq 'pn2') { $a+='env.obs_noise_pocket_pos_std_m=0.01' }; if ($tag -eq 'gx2') { $a+='env.grasp_obs_offset_x_m=0.006' }; & .\scripts\rt_log.ps1 $n @a; $code=$LASTEXITCODE; $e="$d\$($j[4])"; if (-not (Test-Path $e)) { "STOP: $n exit $code and no eval folder"; break }; Copy-Item "$r\rt_logs\$n.txt" $e -Force; if (Test-Path "$r\rt_logs\$n.short.txt") { Copy-Item "$r\rt_logs\$n.short.txt" $e -Force }; [IO.File]::WriteAllText("$e\table_filehash.txt", "$((Get-FileHash -Algorithm SHA256 "$r\$tab").Hash)`n"); [IO.File]::WriteAllText("$e\head.txt", "$(git rev-parse HEAD)`n"); "$n exit $code"; if ($code -ne 0) { "STOP: $n exit $code"; break } }
```

Expected: twenty lines `RT-258_fixeddr_s<N> exit 0` /
`RT-259_fixeddr_s<N>_<tag> exit 0`, no `STOP:`.

Variable names: PowerShell names are case-insensitive (`$T` = `$t`), so
the loop uses `$tab` / `$tag` / `$sd` (found by the laptop dry build).
Laptop dry build 2026-10-01 (fake run tree, `rt_log.ps1` replaced by a
printer, nothing played): 20 commands, argument order as
`run_eval.ps1:197-204`, only `ep_100` picked (an `ep_250` file and the
`rew__` twin were present), files land in the four expected folders with
the right table's hash. Longest PC path of a new file: 249 characters
(`RT-243s2` folder + `eval_target_d4b97259_py2\RT-259_fixeddr_s2_py2.short.txt`;
computed from the manifests' `checkpoint_path`), under 260.

Wall time: UNKNOWN for these tables. For orientation only (other tables):
AutoDR target runs 278-391 s (RT-246); Fixed-DR s1/s2 163-173 s, s3-s5
378-386 s (RT-251). Steps 2 and 4 together: roughly 3.5-4.5 h.

If a run prints `[rl_games-play] eval manifest:` and does not move on
within a minute: the `PROBLEMS.md` watcher (2026-09-22), as RT-246.

## Post-run: one zip (training PC)

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
New-Item -ItemType Directory -Force C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-254 | Out-Null
foreach ($p in 'RT-242','RT-243') { foreach ($s in 1..5) { $run = (Get-ChildItem logs\rl_games\ur5e_insertion -Directory -Filter "*__${p}s${s}_*").FullName; foreach ($e in 'eval_target_02adce21','eval_target_d4b97259_pn2','eval_target_d4b97259_gx2','eval_target_d4b97259_py2') { Copy-Item -LiteralPath "$run\$e" -Destination "C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-254\${p}s${s}_$e" -Recurse -Force } } }
(Get-ChildItem C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-254 -Directory).Count
Compress-Archive -Path C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-254\* -DestinationPath C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-254.zip -Force
Get-Item C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-254.zip
```

Expected: the count line prints `40`.

## PASS/FAIL points, per run (CHECK_NAMES verbatim, `check_eval_conditions.py:117-130`)

The ten points G1a-G3e as RT-247. T-ood: hash `02adce21...`, G3e
"values outside [0, 1] only in the cells an OOD sidecar nails, and the
run's switch matches" with kind `ood` and switch True. Noise: hash
`d4b97259...`, G3b against `BANDS["target"]` with the tag's width from
`NOISE_TAGS` (`check_eval_conditions.py:101-104`). Laptop, per folder:
`python scripts/check_eval_conditions.py --run-dir rt_logs/RT-254/<folder> --table scripts/eval/<table> --log <folder>/<run name>.txt --filehash <content of table_filehash.txt>`.
Plus each rt_log ends with `[rt_log] exit code: 0`; G5
`normalizer_sha256_before` = `_after`, non-null, and equal to the same
seed's T-final / T-final-F run; G8 seed 100, width target, `dr_mode`
table.

## Pins (`eval_manifest.json`)

| field | T-ood (10) | noise (30) |
|---|---|---|
| `width` / `tag` | `"target"` / `""` | `"target"` / `pn2` \| `gx2` \| `py2` |
| `seed` / `num_envs` / `table_rows` / `envs_finished_first_episode` | 100 / 1050 x4 | 100 / 1000 x4 |
| `table_sha256` | `02adce21...64f6` | `d4b97259...4f6d` |
| `dr_table_allow_outside` | `true` | `false` |
| `noise.obs_noise_pocket_pos_std_m` | 0.005 | pn2 0.01, else 0.005 |
| `noise.grasp_obs_offset_x_m` | 0.003 | gx2 0.006, else 0.003 |
| `noise.pocket_obs_yaw_rad` | 0.0223402 | py2 0.0446804, else 0.0223402 |
| `checkpoint_sha256` | = the same seed's T-auswahl run (RT-246 / RT-251) | same |
| `code_commit` / `code_dirty` / `plan_commit` | `c1a635e9...` / `false` / `4b445714...` | same |
| `obs_wrench_mode` | `force_prevact` | same |
| `head.txt` | `c1a635e969f72efd5b9062d2c8994befd29ff100` | same |
| `wall_time_s`, rates | record | record |

Training-width values (0.005 / 0.003 / 0.0223402): RT-252 manifest `noise`
block; `check_eval_conditions.py` `NOISE_TAGS` doubles them.

## Laptop after the checks

Copy to `results/eval_rlgames/T-ood/<method>_s<N>/`,
`results/eval_rlgames/T-final/autodr_s<N>_<tag>/`,
`results/eval_rlgames/T-final-F/fixeddr_s<N>_<tag>/` (D-208 point 2: the
Fixed-DR noise runs sit in T-final-F). `eval_collect.py` (expected VERDICT
OK, 49050 + 10 x 1050 + 30 x 1000 = 89550 rows), `eval_stats.py`: K1-K5
unchanged number for number; `OOD` and `NOISE` sections appear.

## What this run cannot answer

* Any pass/fail: OOD and noise are descriptive (`eval_plan.md` 3.2,
  `eval_stats.py` `ood_blocks`, `noise_blocks`).
* `eval_report.py` reads T-ood for no method yet (HANDOFF open point); the
  numbers come from `summary.json` until that is built.
* The start-height block of T-ood is not beyond training (cap 1.0,
  `InBachelorErwähnen.md` 2026-09-30); the tilt block starts outside the
  controller cone (same file).
* Whether a Fixed-DR seed that never learned (s1, s2) says anything under
  noise: it fails on the clean table already (RT-248).

## OUTCOME (2026-10-02, after the run; judged in the main chat)

Files: `rt_logs/RT-254/` (two zips instead of one: `RT-254a.zip` = 20
AutoDR folders, judged while the Fixed-DR loop ran; `RT-254b.zip` = 20
Fixed-DR folders). Copies in `results/eval_rlgames/T-ood/<method>_s<N>/`,
`T-final/autodr_s<N>_<tag>/`, `T-final-F/fixeddr_s<N>_<tag>/`.
Step 1 dry run: only the py2 block was seen in full (others: no STOP
reported by the user; `run_eval.ps1` re-checks before every block).
Step 2: `chain exit 0`. Step 3 pre-check exactly as expected
(`problems: 0`). Step 4: 20 runs, all eval folders present.

**Hypothesis: PASS.**

* Per folder (40 of 40, scratchpad checker): `check_eval_conditions.py`
  VERDICT PASS (0 failed); `[rt_log] exit code: 0` once; `[rt_log] git:
  c1a635e`; `imported from: ...Phase5_RlGames_v1\source\insertion\insertion`;
  `(OOD table): True` in the 10 T-ood runs, `False` in the 30 noise runs; no
  Traceback; `head.txt` `c1a635e9...`.
* Pins: tag, width target, seed 100, N 1050 (T-ood) / 1000 (noise),
  `table_sha256` `02adce21...` / `d4b97259...`, `code_commit` `c1a635e9...`,
  `code_dirty` false, `plan_commit` `4b445714...`, `force_prevact`, noise
  widths exactly the doubled one per tag (0.01 / 0.006 / 0.0446804, others at
  training width), `dr_table_allow_outside` true exactly in T-ood;
  `checkpoint_sha256` and normalizer hash (G5, before = after) equal to the
  same seed's T-final / T-final-F run.
* Wall time: AutoDR 377-405 s; Fixed-DR s1/s2 162-178 s, s3-s5 379-404 s.
* `eval_collect.py`: VERDICT OK, 89550 rows (as expected). `eval_stats.py`:
  K1-K5, K5_nominal and `final_policy` unchanged; the AutoDR OOD/NOISE
  entries unchanged by the Fixed-DR runs. Reference of a Fixed-DR policy =
  its T-final-F run (s4: 995).

**Numbers (descriptive, no pass line).** T-ood, successes of 150 per block
(reference = T-final(-F) of 1000):

| policy | ref | lat_r_hi | yaw_lo | yaw_hi | tilt_hi | start_h_hi | fric_lo | fric_hi |
|---|---|---|---|---|---|---|---|---|
| autodr s1 | 997 | 147 | 149 | 148 | 146 | 150 | 149 | 136 |
| autodr s2 | 989 | 148 | 150 | 149 | 136 | 147 | 149 | 87 |
| autodr s3 | 989 | 148 | 142 | 147 | 133 | 148 | 148 | 132 |
| autodr s4 | 991 | 150 | 148 | 148 | 138 | 149 | 148 | 138 |
| autodr s5 | 992 | 149 | 148 | 149 | 137 | 147 | 149 | 128 |
| fixeddr s1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fixeddr s2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fixeddr s3 | 982 | 139 | 147 | 143 | 117 | 148 | 148 | 136 |
| **fixeddr s4 (final)** | 995 | 148 | 148 | 150 | **85** | 150 | 150 | 135 |
| fixeddr s5 | 78 | 7 | 3 | 9 | 0 | 5 | 18 | 4 |

Noise, successes of 1000 (difference to the policy's untagged T-final(-F)):
AutoDR pn2 951 / 943 / 848 / 901 / 867 (mean -0.090), gx2 994 / 985 / 988 /
984 / 980 (-0.005), py2 995 / 987 / 987 / 991 / 991 (-0.001). Fixed-DR s3,
s4: pn2 888, 915 (-0.094, -0.080), gx2 971, 988, py2 967, 983; s1, s2 0
everywhere; s5 59-86.

Reading: the final policy Fixed-DR s4 drops hardest at 15 deg tilt (85/150);
that block starts outside the controller cone (`InBachelorErwähnen.md`
2026-09-30), so the number goes next to that sentence. Doubled pocket
position noise (pn2) is the only noise that costs clearly, for both arms.
