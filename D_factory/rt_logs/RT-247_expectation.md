# RT-247 -- expectation, written BEFORE the run (2026-09-30): D-206 plan step 4 on rl_games -- T-final (RT-247), T-final-F (RT-248), T-nominal (RT-249), T-nominal-F (RT-250), 30 runs on the seed-10 table

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

One file for four RT numbers: `rl_games/run_eval.ps1` names each run
`<Rt>_<method>_s<N>` and refuses an existing rt_log file
(`run_eval.ps1:158`, `:177`), and `-Rt` must be `RT-<number>` (`:102`).
T-final and T-nominal play the same ten policies, so one `-Rt` for both
would collide. RT-248, RT-249 and RT-250 are judged in THIS file.

Templates: `rt_logs/RT-246_expectation.md` (T-auswahl on the same clone
state; OUTCOME PASS) and `rt_logs/RT-245_expectation.md` (G1-G8, laptop
checks). Only what differs is written out.

## Context

`Pläne/D-206_Bauplan.md`, "Ablauf am Trainings-PC" step 4. The final
policy is chosen and committed BEFORE this block (D-207, AutoDR s2,
commit `6f0faa25`), so T-final never sees the selection (`eval_plan.md`
2 B). These four blocks carry question A: K1 (T-final), K2 (T-nominal),
K5 / K5_nominal (the `-F` blocks against their plain block,
`eval_stats.py` `K5_BLOCKS`). T-grenze (step 5) and T-ood + noise (step 6)
come after.

## Block -> table, width, names

* `scripts/eval_collect.py:58,60,70,71` (the same four lines at
  `c1a635e9`, `git show c1a635e9:scripts/eval_collect.py`):
  T-nominal and T-nominal-F -> `eval_table_seed10_c18.csv`, `nominal`;
  T-final and T-final-F -> the same table, `target`.
* Table `scripts/eval/eval_table_seed10_c18.csv`, sha256
  `d4b972597b357844d895fc43b03e776537cee45bc7b5fa1910ddfd19f8954f6d`
  (= sidecar `eval_table_seed10_c18.json` `sha256`, laptop `sha256sum`, and
  the file at `c1a635e9`), 1000 rows.
* Eval folders: `<run folder>\eval_target_d4b97259` (T-final, T-final-F),
  `<run folder>\eval_nominal_d4b97259` (T-nominal, T-nominal-F).
* Policies (`run_eval.ps1:72-73`, `:121-133`): T-final, T-nominal =
  `*__RT-242s<N>_*` (autodr) + `*__RT-244s<N>_*` (nodr), 10 runs;
  T-final-F, T-nominal-F = `*__RT-243s<N>_*` (fixeddr), 5 runs. Each
  `nn\last_ur5e_insertion_ep_100_rew_<R>.pth`. The five Fixed-DR files
  exist once each without `rew__` (`rt_logs/RT-243s/RT-243s<N>_nn.txt`:
  `rew_49.403553`, `rew_47.91025`, `rew_534.6653`, `rew_570.71265`,
  `rew_87.923416`).
* rt_log names: `RT-247_<autodr|nodr>_s<N>`, `RT-248_fixeddr_s<N>`,
  `RT-249_<autodr|nodr>_s<N>`, `RT-250_fixeddr_s<N>`.

## Marker- und Code-Stand

**NO pull.** The training PC stays on
`c1a635e969f72efd5b9062d2c8994befd29ff100` (RT-245, RT-246). Since then
only laptop files moved (`eval_collect.py` B1, docs, results); the
training PC does not need them. Expected `plan_commit`
`4b4457144e2c7bedcf22b432ba8e5988e6aa9c7a` (manifest of RT-245 / RT-246).
Gates as RT-246 (git line `c1a635e`, `imported from:
...Phase5_RlGames_v1\...`, `[eval-table] ... d4b97259... Values outside
[0, 1] allowed (OOD table): False`, `code_dirty` false).

## The one hypothesis

**All 30 runs play the seed-10 table end to end through
`rl_games/run_eval.ps1`, every run passes G1a-G3e, every manifest carries
`code_commit` `c1a635e9...`, and each AutoDR / No-DR policy has the same
`checkpoint_sha256` in T-final and T-nominal as in RT-246 (10 of 10).**

Falsified by: a `STOP:` line; a run without `[rt_log] exit code: 0`; any
`CHECK_NAMES` FAIL; another `code_commit`; a checkpoint hash of an
AutoDR / No-DR policy that differs from RT-246 (`rt_logs/RT-246_expectation.md`
OUTCOME, Pins).

## Command (training PC, `env_isaaclab` active, GPU free)

Dry run of all four blocks first:

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
git rev-parse HEAD
git status --short --untracked-files=no
.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-247 -DryRun
.\scripts\rl_games\run_eval.ps1 -Block T-final-F -Rt RT-248 -DryRun
.\scripts\rl_games\run_eval.ps1 -Block T-nominal -Rt RT-249 -DryRun
.\scripts\rl_games\run_eval.ps1 -Block T-nominal-F -Rt RT-250 -DryRun
```

Expected: HEAD `c1a635e969f72efd5b9062d2c8994befd29ff100`; status prints
nothing; per block `table scripts/eval/eval_table_seed10_c18.csv, seed 100,
10 / 5 / 10 / 5 run(s)`, `table Get-FileHash d4b97259...`, `pre-block
checks: ... -- OK`, the `DRY RUN:` lines, `DRY RUN done`. A `STOP:` line =
do not run; paste it.

Then the four blocks in one chain; a block with exit != 0 stops the chain:

```powershell
.\scripts\rl_games\run_eval.ps1 -Block T-final -Rt RT-247; if ($LASTEXITCODE -eq 0) { .\scripts\rl_games\run_eval.ps1 -Block T-final-F -Rt RT-248 }; if ($LASTEXITCODE -eq 0) { .\scripts\rl_games\run_eval.ps1 -Block T-nominal -Rt RT-249 }; if ($LASTEXITCODE -eq 0) { .\scripts\rl_games\run_eval.ps1 -Block T-nominal-F -Rt RT-250 }; "chain exit $LASTEXITCODE"
```

Wall time: T-final about 10 x 6.3 min (RT-246 target runs 278-391 s,
same policies, another table); T-final-F, T-nominal, T-nominal-F:
UNKNOWN (RT-245 has one nominal run, 142.8 s, AutoDR s1 only). Whole
chain: UNKNOWN, roughly 2-3 h.

The final line of each block prints `results\eval\<Block>\...` as
target folder (`run_eval.ps1:239`); on the laptop the root is
`results\eval_rlgames\` (D-206 B1). Cosmetic, no action.

If a run prints `[rl_games-play] eval manifest:` and does not move on
within a minute: the `PROBLEMS.md` watcher (2026-09-22), as RT-246.

## Post-run: one zip (training PC)

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
New-Item -ItemType Directory -Force C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-247 | Out-Null
foreach ($p in @('RT-242s1','RT-242s2','RT-242s3','RT-242s4','RT-242s5','RT-243s1','RT-243s2','RT-243s3','RT-243s4','RT-243s5','RT-244s1','RT-244s2','RT-244s3','RT-244s4','RT-244s5')) { $run = (Get-ChildItem logs\rl_games\ur5e_insertion -Directory -Filter "*__${p}_*").FullName; foreach ($w in @('target','nominal')) { Copy-Item -LiteralPath "$run\eval_${w}_d4b97259" -Destination "C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-247\${p}_eval_${w}_d4b97259" -Recurse -Force } }
Compress-Archive -Path C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-247\* -DestinationPath C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-247.zip -Force
Get-Item C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-247.zip
```

Expected: 30 folders in the zip (15 policies x 2 widths). A missing
folder makes `Copy-Item` print an error; paste it.

## PASS/FAIL points, per run (CHECK_NAMES verbatim, `check_eval_conditions.py:117-130`, laptop version)

The ten points G1a-G3e (G3e passes a plain table by construction, `:127-129`), hash `d4b97259...`; G3b/G3c against
`BANDS["target"]` (RT-247, RT-248) or `BANDS["nominal"]` (RT-249,
RT-250). Laptop, per folder:
`python scripts/check_eval_conditions.py --run-dir rt_logs/RT-247/<prefix>_eval_<width>_d4b97259 --table scripts/eval/eval_table_seed10_c18.csv --log <folder>/<RT>_<m>_s<N>.txt --filehash <content of table_filehash.txt>`.
Plus: each rt_log ends with `[rt_log] exit code: 0`; the console ends
with `chain exit 0`. G5 per run: `normalizer_sha256_before` = `_after`,
non-null. G8: seed 100, width as the block, `dr_mode` table.

## Pins (`eval_manifest.json`, each of the 30)

| field | expected |
|---|---|
| `width` / `tag` | `"target"` (RT-247/248) or `"nominal"` (RT-249/250) / `""` |
| `seed` / `num_envs` / `table_rows` / `envs_finished_first_episode` | 100 / 1000 / 1000 / 1000 |
| `table_sha256` | `d4b972597b357844d895fc43b03e776537cee45bc7b5fa1910ddfd19f8954f6d` |
| `checkpoint_path` | ends `...RT-24{2,3,4}s<N>_...\nn\last_ur5e_insertion_ep_100_rew_<R>.pth` (no `rew__`) |
| `checkpoint_sha256` | autodr / nodr: = RT-246 per policy; fixeddr: record, and the same in RT-248 and RT-250 per seed |
| `code_commit` / `code_dirty` / `plan_commit` | `c1a635e9...` / `false` / `4b445714...` |
| `obs_wrench_mode` / `noise.pocket_obs_yaw_rad` | `force_prevact` / 0.0223402 (`run_eval.ps1:76-77`) |
| `head.txt` | `c1a635e969f72efd5b9062d2c8994befd29ff100` |
| `wall_time_s`, success rates | record, do not judge here |

## Laptop after the checks

Copy each folder to `results/eval_rlgames/<Block>/<method>_s<N>/`, then
`.venv-analysis`: `eval_collect.py --results results/eval_rlgames`
(expected VERDICT OK, one `code_commit`, 43000 rows = 13000 + 30000) and
`eval_stats.py --results results/eval_rlgames` (K1, K2, K5, K5_nominal
computed; K3 still missing: no T-grenze).

## What this run cannot answer

* K3: T-grenze with the final policy AutoDR s2 (D-207), plan step 5.
* T-ood and the noise blocks `pn2`/`gx2`/`py2`: plan step 6.
* Why a Fixed-DR seed fails: RT-243s1, s2 had success 0 in training, s5
  0.056 (`HANDOFF-RL.md` NEXT 3). Their rates here are recorded, not
  explained.
* Whether RT-244s trained as its expectation says: its own judgement,
  separate (not yet done on the laptop).
* Whether the policy uses its LSTM memory: not in D-206's scope.

## CORRECTION, before the run (2026-09-30)

D-208: Fixed-DR is a candidate. The chain RT-247..RT-250 starts only AFTER
RT-251 (Fixed-DR T-auswahl) is judged and the selection over 15 is
committed in D-208. The Context sentence "The final policy is chosen and
committed BEFORE this block (D-207, AutoDR s2, commit `6f0faa25`)" is
superseded: the final policy is D-208's. Nothing else in this file changes.

## OUTCOME (2026-10-01, after the run; judged in the main chat)

Files: `rt_logs/RT-247/` (zip `RT-247.zip`, 30 eval folders = 15 policies x
2 widths, under their training-run prefix). Console: `chain exit 0` (user).
The chain started AFTER D-208's selection was committed (`0a4bc721`), as
the CORRECTION above requires.

**Hypothesis: PASS** (RT-247, RT-248, RT-249, RT-250).

* Gates, all 30: `[rt_log] git: c1a635e`; `imported from:
  ...Phase5_RlGames_v1\source\insertion\insertion`; `[eval-table]` with
  `d4b97259...`; `Values outside [0, 1] allowed (OOD table): False`;
  `head.txt` `c1a635e9...`; `table_filehash.txt` = `d4b97259...`;
  `evaluation complete: all 1000 envs finished their first episode`;
  `[rt_log] exit code: 0`; no Traceback.
* `check_eval_conditions.py`: VERDICT PASS, all ten `CHECK_NAMES` PASS, in
  all 30 folders.
* Pins, all 30: width as the block, `tag` "", seed 100, 1000/1000/1000/1000,
  `table_sha256` `d4b97259...`, `code_commit` `c1a635e9...`, `code_dirty`
  false, `plan_commit` `4b445714...`, `force_prevact`, yaw 0.0223402,
  checkpoint `...\nn\last_ur5e_insertion_ep_100_rew_<R>.pth`. G5 equal,
  non-null. **`checkpoint_sha256` = the same policy's T-auswahl manifest
  (RT-246 / RT-251) in 30 of 30.**
* Steps to the last first episode: target 255 in all 15; nominal 56-133
  for eight AutoDR / No-DR policies, 255 for the rest (autodr s5, nodr s4,
  all five Fixed-DR). Run time 87-425 s.
* `eval_collect.py`: technik + T-auswahl + the four blocks, 0 problems,
  48000 rows, VERDICT OK, one `code_commit`.
* `eval_stats.py` (`summary.json`, marker `eval_stats-2026-09-30b`),
  `success_train`:
  * T-final: AutoDR .997/.989/.989/.991/.992 (mean 0.9916, sd 0.0033);
    No-DR .099/.155/.178/.098/.096 (mean 0.1252, sd 0.0386).
    **K1 met:** diff +0.8664, Welch [+0.8186, +0.9142] (df 4.06), bootstrap
    [+0.8342, +0.8946].
  * T-nominal: AutoDR 1.000/1.000/.998/.996/.999 (0.9986); No-DR
    1.000/1.000/.999/.997/1.000 (0.9992). **K2 met:** diff -0.0006, Welch
    [-0.0028, +0.0016] (df 7.55), bootstrap [-0.0022, +0.0010].
  * T-final-F: Fixed-DR .000/.000/.982/.995/.078 (mean 0.411, sd 0.528).
    **K5 contains_0:** diff +0.5806, Welch [-0.0752, +1.2364] (df 4.00),
    bootstrap [+0.1854, +0.9758].
  * T-nominal-F: Fixed-DR .000/.000/.987/.998/.184 (0.4338, sd 0.5155).
    **K5_nominal contains_0:** diff +0.5648, Welch [-0.0753, +1.2049].
* Reported, not judged: No-DR on T-final fails by timeout (817-895 per
  seed), force aborts 1-11. Fixed-DR s1, s2 end every episode as timeout
  even on T-nominal (1000/1000); s5 811 timeouts on T-nominal. The final
  policy Fixed-DR s4: 995/1000 on T-final-F, 998/1000 on T-nominal-F.
* Next: T-grenze with `-Final fixeddr_s4` (K3), then T-ood + noise.
