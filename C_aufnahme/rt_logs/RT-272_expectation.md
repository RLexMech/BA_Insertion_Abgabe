# RT-272 -- expectation, written BEFORE the run (2026-10-11): the Experiment C final policy RT-206s1 replayed on one T-final row, recorded, with an observation trace

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Templates: `rt_logs/RT-269_expectation.md` § RT-269b (ffmpeg gdigrab block),
`rt_logs/RT-232_expectation.md` (clone `Phase5_Robustheit_v1`, detach to
`8299485d`, return with `git checkout -`). Only what differs is written out.

## Context

* Thesis (Word `Bachelorarbeit_Alex_v_10.docx`, 2026-10-11) 5.1.3 states that
  in Experiment C the part was aligned to the pocket BEFORE contact, and gives
  this as the reason for the yaw error of Experiment D. `HANDOFF-RL.md:542`:
  no data in the repo, recording only, user decides. User 2026-10-11 (chat):
  record it.
* Why C could align early: the C observation carries `yaw_cos_sin` and
  `pocket_quat` exactly; only `tip_rel` and `force` are noised (D-194
  Context, read 2026-09-26).
* Policy: RT-206s1 `model_1499.pt`, the C final policy (D-190). Checkpoint
  sha256 `b7e4ca87516d332a2e5ed4aea1e8f0040cc9ba3cf61f26940af8c47523d66394`
  (`results/eval/T-final/autodr_s1/eval_manifest.json`, `checkpoint_sha256`).
* Case: T-final table `scripts/eval/eval_table_seed10.csv` (sha256
  `1613635b...`, equal at `8299485d` and on `p5-rlgames`, laptop
  2026-10-11), data row 6. In RT-222's T-final run of this policy
  (`results/eval/T-final/autodr_s1/applied.csv` row_id 6,
  `episodes.csv` env_id 6 episode 0): `yaw_rad` -0.0849910006 (-4.87 deg),
  `start_height_m` 0.10953021, tilt 0.20 deg (from `pocket_q*`), lateral
  26.7 mm, outcome `success` in 28 steps, `ip_max_mm` 0.147246,
  `force_max_raw_n` 16.4205.
  Chosen by: success in RT-222, |yaw| >= 4 deg, start >= 80 mm, tilt <= 3 deg
  (22 rows qualify); row 6 has the smallest tilt of the rows with
  |yaw| >= 4.87 deg (6, 258, 299, 131). One case, picked for a clear view, not
  a sample.

## Code state and markers

* Clone `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`,
  detached at `8299485d0a60dd0cca01a7d84946f441a96c1100` (the code of
  RT-222..RT-225 and RT-232, D-190; `eval_manifest.json` `code_commit`).
  No pull: the commit is in that clone since RT-232.
* `git log 391c7ae4..origin/p5-kraftsensor -- source scripts/rsl_rl` names only
  `8299485d` and `3c6dabcd` (Fixed-DR mode, D-192), so `8299485d` is the last
  evaluation state before Fixed-DR training code.
* CODE_MARKER `insertion-osc-2026-09-03a`
  (`insertion_env.py:76` at `8299485d`), printed as `code marker: ...` in the
  startup report (`insertion_env.py:4847`, report steps `(2, 60, 180, 258)`,
  `insertion_env_cfg.py:908`). `scripts/rsl_rl/play.py` at `8299485d` has no
  SCRIPT_MARKER.
* THE IMPORT TRAP: the training PC's editable `insertion` install points to
  `Phase5_MitAktion_v1` since RT-233 (`RT-233_expectation.md:103-110`), and
  `play.py` at `8299485d` puts no clone on `sys.path`. The command therefore
  sets `$env:PYTHONPATH` to this clone's `source\insertion` for this session
  and restores it afterwards. Gate P0 checks the
  `[insertion] robot asset:` line (`insertion_env.py:1596`); a path under any
  other clone = RUN INVALID.
* Outputs stay out of the frozen run folder: `play.py` writes `exported\` next
  to the checkpoint (`play.py:587`) and `replay_h...\` under its folder
  (`play.py:471-521`), so the run plays a COPY in
  `C:\Users\Simon\Desktop\Alexander_Pett\view\RT-272\`.

## Run form

* `--num_envs 1` with a one-row table: env 0 sits at the world origin, so the
  camera of `videos/README.md` (`env.viewer.eye=[0.99,0.672,0.947]`,
  `env.viewer.lookat=[0.413,0.095,0.369]`, set 2026-09-15 on `5a6cc8e`, the C
  scene) shows it. `FIXTURE_POS` is equal at `8299485d` and HEAD
  (`insertion_tasks_cfg.py:71`).
* Table mode by Hydra, not `--eval-table`: `--eval-table` writes an eval
  folder with a manifest (`play.py:295-322`), this is a view.
  `env.dr_mode=table env.dr_table_path=rt_logs/RT-272_row6.csv
  env.dr_table_width=target`; the path is relative with forward slashes, so
  Hydra sees no backslash or drive colon. `validate_table_mode`
  (`insertion_env_cfg.py:1884` at `8299485d`) needs `start_floor_m` equal to
  `start_tip_above_entrance`: the command sets neither (as RT-222).
  Env i plays row i in every episode (`insertion_env.py:4906`), so every
  episode is row 6.
* `--seed 100` as RT-222. Deterministic actor (rsl_rl inference policy).
* `--trace-obs rt_logs/RT-272_trace.json --trace-steps 256`: the trace ends
  the run (`play.py:754-755`). At 28 steps per episode (RT-222 row 6) that is
  about 8 episodes; the real count is in the trace.
* Recording: ffmpeg of the Python env, gdigrab, `-t 300`, into
  `C:\Users\Simon\Desktop\Alexander_Pett\Bilder\RT-272_C_ausrichten.mkv`.
  300 s is a cap, not a prediction: the run time of this setup is unknown.

## The one hypothesis

**In every complete episode of the trace, the C policy has done at least half
of its yaw turn before the first contact: (phi_start - phi_contact) /
(phi_start - phi_end) >= 0.5.**

Reading rule, written now:
* episodes split at `done`; only episodes with a start and an end in the trace
  count; the outcome comes from `replay_h...\episodes.csv` (the env's episode
  record, `insertion_env.py:79-84` and `1137-1148` at `8299485d`, written into
  `env_cfg.log_dir`). NOT checked that a play run writes it; if it is missing,
  the outcome is read from the trace (`done` and the last `truth.tip_true_m`).
* phi = atan2(`yaw_cos_sin`[1], `yaw_cos_sin`[0]) per step (trace slice
  `yaw_cos_sin`, `insertion_math.py:780`, "cos phi / sin phi -- NOT cos 4
  phi"; the part's x-axis yaw in the pocket frame, `insertion_math.py:932`;
  not noised, D-194).
* phi_start = first step of the episode; phi_end = last step of a SUCCESSFUL
  episode; phi_contact = first step with `truth.interpen_max_m` > 0.
* The 0.5 is OWN CONSTRUCTION: no source gives a threshold for "aligned before
  contact". The share itself is reported for every episode, whatever the
  verdict.
* The recording is the second reading: the part turns above the rim, not
  while it touches it.

FALSE if the share is below 0.5 in any complete successful episode, or if no
complete successful episode exists in the trace.

## Command

Training PC, PowerShell, conda env `env_isaaclab` active, no training run in
parallel. Three blocks, one after the other.

**Block 1 -- clone check and detach** (as RT-232):

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1; git rev-parse --short HEAD; git status --porcelain --untracked-files=no; git checkout --detach 8299485d0a60dd0cca01a7d84946f441a96c1100; git rev-parse HEAD
```

Expected: first line = the branch HEAD to return to (write it down); NO
status output; last line `8299485d0a60dd0cca01a7d84946f441a96c1100`. Status
output = stop, nothing else runs.

**Block 2 -- checkpoint copy and one-row table:**

```powershell
$r = "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1"; $v = "C:\Users\Simon\Desktop\Alexander_Pett\view\RT-272"; New-Item -ItemType Directory -Force $v | Out-Null; Copy-Item -LiteralPath "$r\logs\rsl_rl\ur5e_insertion\09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1\model_1499.pt" "$v\model_1499.pt"; Get-FileHash -Algorithm SHA256 "$v\model_1499.pt" | Format-List Hash; Get-Content "$r\scripts\eval\eval_table_seed10.csv" | Select-Object -Index 0,7 | Set-Content -Encoding ascii "$r\rt_logs\RT-272_row6.csv"; Get-Content "$r\rt_logs\RT-272_row6.csv"; Get-FileHash -Algorithm SHA256 "$r\rt_logs\RT-272_row6.csv" | Format-List Hash
```

Expected: checkpoint hash
`B7E4CA87516D332A2E5ED4AEA1E8F0040CC9BA3CF61F26940AF8C47523D66394`; the CSV
prints two lines, the header `lat_r,lat_phi,yaw,...,grasp_obs_x` and a row
starting `0.7905134972617673,0.5316016891970812,0.01303743209502628`. Table
hash expected `6487396E4F3BEC2971C8618E04B2A585C3F307352DD267E6CEA7EFD88A5779D2`
(laptop, same two lines with CRLF; if only this hash differs while the two
lines match, the line ending differs -- not a stop).

**Block 3 -- recording and replay:**

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1; $oldpp = $env:PYTHONPATH; $env:PYTHONPATH = "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\source\insertion"; $bild = "C:\Users\Simon\Desktop\Alexander_Pett\Bilder"; New-Item -ItemType Directory -Force $bild | Out-Null; $ff = python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"; Write-Host "ffmpeg: $ff"; if (-not (Test-Path $ff)) { $env:PYTHONPATH = $oldpp; throw "ffmpeg not found" }; $rec = Start-Process -FilePath $ff -ArgumentList "-y","-f","gdigrab","-framerate","30","-t","300","-i","desktop","-vf","scale=1600:-2","-pix_fmt","yuv420p","$bild\RT-272_C_ausrichten.mkv" -PassThru -WindowStyle Hidden; .\scripts\rt_log.ps1 RT-272 python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 1 --seed 100 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\view\RT-272\model_1499.pt" --trace-obs rt_logs/RT-272_trace.json --trace-steps 256 env.dr_mode=table env.dr_table_path=rt_logs/RT-272_row6.csv env.dr_table_width=target "env.viewer.eye=[0.99,0.672,0.947]" "env.viewer.lookat=[0.413,0.095,0.369]"; $rec.WaitForExit(); $env:PYTHONPATH = $oldpp; Get-Item "$bild\RT-272_C_ausrichten.mkv"
```

Keep the Isaac window in front and do not cover it. If the view fails on the
two `env.viewer.*` flags, run without them (RT-269's fallback).

**Block 4 -- pack, then return the clone:**

```powershell
$r = "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1"; $v = "C:\Users\Simon\Desktop\Alexander_Pett\view\RT-272"; $z = "C:\Users\Simon\Desktop\Alexander_Pett\Zips"; New-Item -ItemType Directory -Force $z | Out-Null; $files = @("$r\rt_logs\RT-272.txt", "$r\rt_logs\RT-272.short.txt", "$r\rt_logs\RT-272_trace.json", "$r\rt_logs\RT-272_row6.csv", "C:\Users\Simon\Desktop\Alexander_Pett\Bilder\RT-272_C_ausrichten.mkv") + (Get-ChildItem $v -Directory -Filter "replay_*" | ForEach-Object { $_.FullName }); Compress-Archive -LiteralPath $files -DestinationPath "$z\RT-272.zip" -Force; Get-Item "$z\RT-272.zip"; cd $r; git checkout -; git rev-parse --short HEAD
```

Expected: the zip exists; the last line is the branch HEAD from block 1.

## Gates (in log order)

* P0 `[rt_log] git: 8299485`; `[insertion] robot asset:` under
  `...\Phase5_Robustheit_v1\source\insertion\...`; `code marker:
  insertion-osc-2026-09-03a`. Any other clone or marker = RUN INVALID.
* P1 `[INFO]: Loading model checkpoint from:
  C:\Users\Simon\Desktop\Alexander_Pett\view\RT-272\model_1499.pt`
  (`play.py:559`); no `[play] REFUSED` (the width guard, `play.py:575-578`),
  no Traceback.
* P2 `[eval-table] rt_logs\RT-272_row6.csv  rows 1  sha256` + the block-2
  table hash in lower case + `width target. Env i plays row i ...`
  (`insertion_env.py:4904-4908`; the path is `str(pathlib.Path)`, so Windows
  prints a backslash).
* P3 `[trace] wrote 256 steps x 1 envs to ...\rt_logs\RT-272_trace.json`
  (`play.py:809`); `[rt_log] exit code: 0`.
* P4 `RT-272_C_ausrichten.mkv` exists and is larger than 1 MB.

## Pins (from `RT-272_trace.json`, fields written at `play.py:665-676`)

| Field | Expected | Source |
|---|---|---|
| `num_envs` | 1 | `--num_envs 1` |
| `obs_wrench_mode` | `force` | `insertion_env_cfg.py:232` at `8299485d` |
| `obs_noise_pocket_pos_std_m` | 0.005 (uniform half width, not a sigma) | `insertion_env_cfg.py:1162-1174` at `8299485d`, equal at `5a6cc8e:1158` |
| `force_obs_noise_std_n` | 3.5 | `insertion_env_cfg.py:1175` at `8299485d` |
| `grasp_obs_offset_x_m` | 0.003 | `insertion_env_cfg.py:1208` at `8299485d` |
| `fixture_pos_noise_xy_m` | 0.005 | `insertion_env_cfg.py:741` at `8299485d` |
| `torque_obs_noise_std_nm` | 0.0 | `insertion_env_cfg.py:1182` at `8299485d` |
| `slices.yaw_cos_sin` | [19, 21] | `insertion_math.py:780` at `8299485d` |
| `truth` block | present | `play.py:685-694` (guarded) |

## What this run cannot answer

* One policy (s1), one case (row 6), one play seed. It does not show that "the
  C policies" align early in general; the thesis sentence may only claim what
  this case shows.
* It is no causal test of "exact yaw observation -> early alignment". That
  needs the same training with a yaw error, which is Experiment D, a different
  setup (D-200: no numbers side by side with the final study).
* `interpen_max_m` = 0 does not prove free (`play.py:705`, truth_frames). A contact earlier than the first > 0 step would put
  phi_contact later and make the share LARGER, i.e. biased toward the
  hypothesis. The recording is the check on that.
* Episode count and run time are unknown before the run.

## OUTCOME

2026-10-12. RT-272 died at the Hydra merge, before any env existed (log tail
pasted by the user): `ValueError: [Config]: Incorrect type under namespace:
/dr_table_path. Expected: <class 'NoneType'>, Received: <class 'str'>.`,
`[rt_log] exit code: 1`. Cause: Isaac Lab's `update_class_from_dict` refuses a
value whose type differs from the default's (`C:\IsaacLab\source\isaaclab\isaaclab\utils\dict.py:153-157`),
and `dr_table_path` defaults to `None` (`insertion_env_cfg.py:716` at
`8299485d`). Known since RT-138 (`PROBLEMS.md`, 2026-09-02); missed when this
file was written -> `docs/reference/pruefregeln.md` L-23. No policy step ran:
the hypothesis is untouched. Repeated as RT-272b below.

## CORRECTION, after the failed start (2026-10-12): RT-272b, the same yaw and start height as a probe pose

* Why not the table: table mode is reachable only through `--eval-table`
  (`play.py:295-322` at `8299485d`), which refuses `--trace-obs`
  (`play.py:302`) and stops after the first episode. So RT-272b uses the
  existing generalisation probe (D-038): `--fixture-yaw=-4.87`
  (`play.py:420-441`) and the start height fixed at row 6's 0.1095 m by three
  FLOAT overrides over float defaults, which the merge accepts:
  `env.start_tip_above_entrance=0.1095 env.start_tip_above_entrance_low=0.1095
  env.start_floor_m=0.1095` (low = high: fixed start, `insertion_env_cfg.py:447`;
  floor = high: no floor, `insertion_env_cfg.py:481`; a floor below the start
  together with a fixture angle under `dr_mode='off'` is refused,
  `insertion_env.py:428-440`).
* `dr_mode` stays `off` (default, `insertion_env_cfg.py:704`): lateral 0,
  tilt 0, friction `CONTACT_FRICTION` 0.14 (`insertion_tasks_cfg.py:1396`).
  `--fixture-yaw` forces `fixture_pos_noise_xy` to 0 (`play.py:434-435`).
* So the case is NOT row 6 any more. Same: yaw -4.87 deg, start 109.5 mm,
  observation noise. Different: lateral 0 instead of 26.7 mm, friction 0.14
  instead of 0.108, tilt 0 instead of 0.20 deg, fixture xy noise 0.
* Names: `RT-272b` (rt_log APPENDS to `RT-272.txt`), trace
  `rt_logs/RT-272b_trace.json`, recording `RT-272b_C_ausrichten.mkv`.
  The checkpoint copy of block 2 is reused; the one-row table is not used.
* Hypothesis and reading rule: unchanged.
* Gates: P0, P1, P3, P4 as above with the new names. P2 is replaced by
  `[INFO] --fixture-yaw: fixture pose noise forced to 0 for this deterministic
  probe.` (`play.py:434`) and `[INFO] GENERALISATION PROBE: fixture yawed by
  -4.87 deg about env z ...` (`play.py:439`). An `[eval-table]` line = wrong
  run.
* Pins: as above, except `fixture_pos_noise_xy_m` = 0.0 (forced by the probe).
* Cannot answer, in addition: this is not a T-final case; no number of this
  run goes next to the T-final results.

**Block 3b -- recording and replay** (the clone must still be detached at
`8299485d`; the first line checks it):

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1; git rev-parse HEAD; (Get-FileHash -Algorithm SHA256 "C:\Users\Simon\Desktop\Alexander_Pett\view\RT-272\model_1499.pt").Hash; $oldpp = $env:PYTHONPATH; $env:PYTHONPATH = "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\source\insertion"; $bild = "C:\Users\Simon\Desktop\Alexander_Pett\Bilder"; New-Item -ItemType Directory -Force $bild | Out-Null; $ff = python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"; Write-Host "ffmpeg: $ff"; if (-not (Test-Path $ff)) { $env:PYTHONPATH = $oldpp; throw "ffmpeg not found" }; $rec = Start-Process -FilePath $ff -ArgumentList "-y","-f","gdigrab","-framerate","30","-t","300","-i","desktop","-vf","scale=1600:-2","-pix_fmt","yuv420p","$bild\RT-272b_C_ausrichten.mkv" -PassThru -WindowStyle Hidden; .\scripts\rt_log.ps1 RT-272b python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 1 --seed 100 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\view\RT-272\model_1499.pt" --fixture-yaw=-4.87 --trace-obs rt_logs/RT-272b_trace.json --trace-steps 256 env.start_tip_above_entrance=0.1095 env.start_tip_above_entrance_low=0.1095 env.start_floor_m=0.1095 "env.viewer.eye=[0.99,0.672,0.947]" "env.viewer.lookat=[0.413,0.095,0.369]"; $rec.WaitForExit(); $env:PYTHONPATH = $oldpp; Get-Item "$bild\RT-272b_C_ausrichten.mkv"
```

Expected first two lines: `8299485d0a60dd0cca01a7d84946f441a96c1100` and
`B7E4CA87516D332A2E5ED4AEA1E8F0040CC9BA3CF61F26940AF8C47523D66394`. Another
HEAD = run block 1 first.

**Block 4b -- pack, then return the clone:**

```powershell
$r = "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1"; $v = "C:\Users\Simon\Desktop\Alexander_Pett\view\RT-272"; $z = "C:\Users\Simon\Desktop\Alexander_Pett\Zips"; New-Item -ItemType Directory -Force $z | Out-Null; $files = @("$r\rt_logs\RT-272.txt", "$r\rt_logs\RT-272b.txt", "$r\rt_logs\RT-272b.short.txt", "$r\rt_logs\RT-272b_trace.json", "C:\Users\Simon\Desktop\Alexander_Pett\Bilder\RT-272b_C_ausrichten.mkv") + (Get-ChildItem $v -Directory -Filter "replay_*" | ForEach-Object { $_.FullName }); Compress-Archive -LiteralPath $files -DestinationPath "$z\RT-272.zip" -Force; Get-Item "$z\RT-272.zip"; cd $r; git checkout -; git rev-parse --short HEAD
```

### OUTCOME RT-272b

2026-10-12, log start `2026-10-12 00:24:19`, `git: 8299485`, cmd = block 3b
(`filter_rt_log.py --head`); expectation committed before it (`7b52a4d4`,
00:07:02). Files in `rt_logs/RT-272/` (zip from the training PC).

Gates: **PASS technically.**
* P0 PASS: `[insertion] robot asset:` under `Phase5_Robustheit_v1` (L346);
  `code marker: insertion-osc-2026-09-03a` (L371). The PYTHONPATH won.
* P1 PASS: `[INFO]: Loading model checkpoint from:
  C:\Users\Simon\Desktop\Alexander_Pett\view\RT-272\model_1499.pt` (L367); no
  `[play]` line, no Traceback.
* P2 PASS: one `[INFO] --fixture-yaw` and one `[INFO] GENERALISATION PROBE`
  line (`--events "[INFO]"`); no `[eval-table]` line.
* P3 PASS: `[trace] wrote 256 steps x 1 envs` (L1014), exit 0 (L1016).
* P4 PASS: `RT-272b_C_ausrichten.mkv` 14 500 471 bytes, 300 s, 1600x900.
* Pins: all as written (trace JSON: `fixture_pos_noise_xy_m` 0.0,
  `obs_noise_pocket_pos_std_m` 0.005, `force_obs_noise_std_n` 3.5,
  `grasp_obs_offset_x_m` 0.003, `torque_obs_noise_std_nm` 0.0,
  `obs_wrench_mode` force, `slices.yaw_cos_sin` [19, 21], `truth` present).
* Episodes (`replay_h+110mm_yaw-04.9deg/episodes.csv`): 9, all `success`,
  28-29 steps; the trace holds exactly these 9, all complete.

Hypothesis: **FALSE by the rule as written.**
`python scripts/trace_yaw_before_contact.py rt_logs/RT-272/RT-272b_trace.json`:
share at the first `interpen_max_m` > 0 row, episodes 0-8:
0.962, 0.057, 0.126, 1.055, 0.139, 1.001, 0.000, 0.299, 0.974.
Five of nine are below 0.5.
* Episode 6 is an instrument artifact the rule did not exclude: interpen
  0.020 mm on the episode's FIRST row, at z +109.5 mm with 0.0 N (the reset
  row carries the old episode's buffer). An error of this expectation, found
  after the result; it does not change the verdict (1, 2, 4, 7 stay below).
* Episodes 1, 2, 4, 7: first interpen at row 16, z +8.0 to +8.2 mm,
  0.22-0.34 mm, 0.4-0.6 N: a light touch 8 mm above the opening, with 6-30 %
  of the turn done.
* DESCRIPTIVE, NOT the rule (chosen after the result): the first row with
  |force_tared_raw_n| >= 1 N is row 17 (z +2.0 to +2.8 mm) in 8 of 9 episodes,
  row 18 in episode 8; the share there is 0.53-0.87 (0.974 in episode 8).
* Episode 0 row by row (`--rows`): yaw error 4.86 deg at the start, 2.45 at
  z +87 mm, back to 5.38 at z +21 mm, then 3.24 (z +15), 1.09 (z +9),
  0.65 (z +2, 24.2 N), about 0 from z -8 mm on.

Reading: the C policy does most of its 5 deg turn in the last ~20 mm above
the opening and finishes it in contact. "Aligned BEFORE contact" does not
hold in this case, neither by the rule nor by the force onset. One policy,
one pose (not row 6, see the correction).

Counter-check from existing data (2026-10-12, user asked whether C aligns
MORE before contact than D): the same reader on the Experiment D trace
RT-271 s4 (Auto-DR s4, rl_games, T-ood friction_hi row 906 in env 0, start
+62 mm, friction 0.23), `python scripts/trace_yaw_before_contact.py
rt_logs/RT-271/RT-271_s4_trace.json --env 0`: 9 complete episodes with a
turn of 4.85-5.26 deg (the observed yaw carries D-194's per-episode belief
error, a constant offset that cancels in the share). Share at the first
|F_raw| >= 1 N row: 0.676, 0.857, 0.563, 1.044, 1.073, 0.623, 1.060, 0.473,
0.324 (C, RT-272b: 0.53-0.97). The light touch at z +8 mm shows up in D too
(episodes 0, 1). Seven of D's nine first rows carry the stale reset
interpen as well. So on this measure C and D look alike; the data does NOT
show that C aligns more before contact than D. Different case, policy and
setup (start 62 vs 110 mm, friction 0.23 vs 0.14, LSTM vs MLP): a
description, not a comparison test.

Recording: the camera of `videos/README.md` shows the whole robot; a 5 deg
turn of the part is not visible at that distance (contact sheet of 1 frame
per 5 s and a 5 fps crop, laptop ffmpeg 9.0.1, 2026-10-12). The simulation is
on screen from about 30 s to about 95 s of the file.
