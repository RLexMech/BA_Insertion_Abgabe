# RT-242a / b / b2 / c / d -- expectation, written BEFORE the run (2026-09-28): the D-204 probes before any training -- home tip, start IK at the high and the low corner, reset force

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Number: no `RT-242` under `rt_logs/` on any `origin/*` branch (`git grep`,
2026-09-28). Templates: `rt_logs/RT-185_expectation.md` (the zero_agent recipe,
parts A/B, and its Nachtrag), `rt_logs/RT-197_199_expectation.md` (256 envs,
4160 steps), `rt_logs/RT-143_expectation.md` (`--home-pose`). Plan:
`Pläne/D-202_D-203_Bauplan.md` step 7; decision D-204, "Verification".

## Context

D-204 raised the home to 0.400 m above the block top (tip 0.415 m nominal),
the start ceiling to 0.270 m, the box to z 0.2915 / x-y 0.100 m, and the true
fixture spread to 0.05 m in x, y and z. None of it has run under Isaac. Plan
step 7 asks for five probes before any training. Reading the recipes against
today's code found three gaps; all three are closed in `scripts/zero_agent.py`
(commit before this file, user "a", 2026-09-28):

1. **RT-143 never measured the home pose.** Its first reading came after one
   full control step, and the home stood outside the tip clamp box, so the
   arm was already moving down. Proof inside RT-143 itself: the same first
   line read -145.879 mm at kp 100 and -108.569 mm at kp 500 (VERDICTS before
   `a7b67909`, row 172: "Tiefe -145.879 -> -50.015 mm ueber Schritt 1-20
   (kp 100) und -108.569 -> -50.007 mm ueber Schritt 1-5 (kp 500)"). A pose at
   rest does not depend on kp. The flange line of the same run (0.297879 m at
   t = 0.133 s) sits the same 19.121 mm under 0.317 m: the same moment. The
   one clean reading of the old home is RT-46, flange 0.317026 m against
   0.3170 m nominal (`insertion_env.py:5658`). So the "19.1 mm deficit" that
   D-204 (1), `insertion_tasks_cfg.py:112`, `check_env_wiring.py:75-88` and
   the plan's section 2 carry as a possible home error is, by RT-143's own
   numbers, the first step of the clamp pull. Those texts get their
   correction after RT-242a, with the measured number.
   The new home repeats the trap: tip 0.415 m against a box top of at most
   0.2915 + 0.05 = 0.3415 m. Fix: `--print-force` now prints a `step    0`
   line from the reset observation. `env.reset()` writes the joints, runs
   `sim.forward()` and then reads the observation
   (`C:\IsaacLab\source\isaaclab\isaaclab\envs\direct_rl_env.py:315-331`;
   the joint write invalidates the body poses, `articulation.py:606-614`), so
   this depth is the written pose before any physics step.
2. **`--home-pose` left the fixture spread live.** The entrance moves up to
   50 mm in z per episode; the home stands still. The depth would read against
   a moving plane, and `check_env_wiring`'s bound needs the tip above the
   NOMINAL entrance. Fix: `--home-pose` sets `fixture_pos_noise_xy` and
   `fixture_pos_noise_z` to 0 (the same assignments as the probe-angle block).
3. **b, b2, c and d did not build.** `--start-tip-above-entrance-mm` set the
   start and its low edge, not `start_floor_m` (cfg default 0.020 m,
   `insertion_env_cfg.py:510`). At 270 mm the floor switched ON and the env
   refused it beside a lateral offset or an angle under `dr_mode='off'`
   (`insertion_env.py:436-455`); at -30 mm the floor lay above the start
   (`:409-414`). Fix: the flag sets `start_floor_m` equal ("Equal = no
   floor", the env's own message). Laptop proof: the env's floor block,
   cut from `insertion_env.py` and executed on the five probe configs
   (scratch script, not committed): old settings -> a OK, b/b2/c/d
   `ValueError`; new settings -> all five OK, `floor_on=False`.

Laptop checks on the fix: `py_compile` OK, `check_mutation_anchors` 600/600,
`check_env_wiring` 435/435, `check_env_wiring --self-test` COUNTER-PROOF
PASSED.

## Code state

Clone `Phase5_RlGames_v1`, branch `p5-rlgames`; the SHA is the commit that
adds this file, read back from `origin/p5-rlgames` and given in chat. Since
`0f8660e6` (the D-204 docs commit) the parallel chat changed only
`scripts/check_workcell_geometry.py` + docs (`4daa71b6`); this chat changes
only `scripts/zero_agent.py`. `CODE_MARKER = "insertion-osc-2026-09-03a"`
(`insertion_env.py:76`), unchanged; `zero_agent.py` has no `SCRIPT_MARKER`.
The gate is the `[rt_log] git:` line plus the new print lines below; a log
without them ran the old script = RUN INVALID (precedent RT-151b).

## The one hypothesis

**At D-204's code, the home tip stands above 0.3415 m over the nominal
entrance before any physics step, and the start IK solves every episode clean
at the high corner (with and without the probe angles) and at the low corner,
with no contact force at the high corner's reset.**

## Commands

Training PC, PowerShell, `env_isaaclab` active, GPU free. The five runs are
independent; run all, then zip. Wall time: a and d are seconds after the Isaac
start; b, b2, c 4160 steps x 256 envs each, duration unknown.

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
git pull
git rev-parse HEAD
.\scripts\rt_log.ps1 RT-242a python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --home-pose --print-force --max-steps 6
.\scripts\rt_log.ps1 RT-242d python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --start-tip-above-entrance-mm 270 --start-lateral-offset-mm 30 --fixture-tilt 10 --fixture-yaw 5 --max-steps 6 --print-force
.\scripts\rt_log.ps1 RT-242b python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --start-tip-above-entrance-mm 270 --start-lateral-offset-mm 30 --fixture-tilt 10 --fixture-yaw 5 --max-steps 4160
Copy-Item logs\demo_metrics.json rt_logs\RT-242b_demo_metrics.json
.\scripts\rt_log.ps1 RT-242b2 python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --start-tip-above-entrance-mm 270 --start-lateral-offset-mm 30 --max-steps 4160
Copy-Item logs\demo_metrics.json rt_logs\RT-242b2_demo_metrics.json
.\scripts\rt_log.ps1 RT-242c python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --start-tip-above-entrance-mm -30 --max-steps 4160
Copy-Item logs\demo_metrics.json rt_logs\RT-242c_demo_metrics.json
New-Item -ItemType Directory -Force C:\Users\Simon\Desktop\Alexander_Pett\Zips | Out-Null
Compress-Archive -LiteralPath rt_logs\RT-242a.txt, rt_logs\RT-242a.short.txt, rt_logs\RT-242d.txt, rt_logs\RT-242d.short.txt, rt_logs\RT-242b.txt, rt_logs\RT-242b.short.txt, rt_logs\RT-242b_demo_metrics.json, rt_logs\RT-242b2.txt, rt_logs\RT-242b2.short.txt, rt_logs\RT-242b2_demo_metrics.json, rt_logs\RT-242c.txt, rt_logs\RT-242c.short.txt, rt_logs\RT-242c_demo_metrics.json -DestinationPath C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-242.zip -Force
Get-Item C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-242.zip
```

Why these numbers: 256 envs and 4160 steps = RT-197..199's form (user
2026-09-14; 16 episodes x 256 envs = 4096, plus 64 margin). An episode is 256
control steps (`episode_steps`, `insertion_env_cfg.py:1385`, kept under OSC by
`resolve_control_mode`, `:1950`); zero_agent has no random episode start, so
all envs reset together at step 256, 512, ... 4096. The dump fires at every
common reset: window `max(2000, 256*20*16/256)` = 2000
(`insertion_env.py:1320-1322`, `metrics_window_iterations` 20), interval
2000 // 8 = 250 (`:1521`), trigger `_ep_count % 250 < n` with n = 256 (`:3725`).
`logs\demo_metrics.json` is the fallback path of a run without `log_dir`
(`:1509`, `:4356`), overwritten by every run, hence the copy after each.
RT-242a runs at the cfg kp 100 (`osc_kp_pos`, `insertion_env_cfg.py:250`);
the plan's "RT-143b form" means the flags `--home-pose --print-force`, not
kp 500 -- step 0 does not depend on kp.

## Points -- every run (P0 gate)

`[rt_log] git:` = the chat SHA; `[rt_log] cmd:` = the command above; `code
marker: insertion-osc-2026-09-03a`; the obs-noise pins as in RT-197..199
(pocket, force and grasp offset 0). No Traceback, no `ValueError`. Missing
exit line = a note, not a FAIL (open since RT-192b).

## Points -- RT-242a (home tip)

* **a1 setup**: the lines `[zero_agent] --home-pose: start_tip_above_entrance
  = None (bare home pose, RT-46 form)` and `[zero_agent] --home-pose:
  fixture_pos_noise_xy 0.05 -> 0.0, fixture_pos_noise_z 0.05 -> 0.0 (depth
  against the NOMINAL entrance, THIS RUN ONLY).` Missing = old script, RUN
  INVALID.
* **a2 the home tip**: the line `[zero_agent] step    0 | depth p50 ... | min
  ... | max ... | reset observation, before any physics step (no force read)`.
  Home tip above the nominal entrance = -(depth p50) / 1000 m. **PASS: > 0.3415
  m** (box half 0.2915, `insertion_env_cfg.py:316`, + z spread 0.05,
  `:792`; D-204 Verification). min = max expected (4 envs, spread 0, reset
  noise 0, `insertion_env_cfg.py:410-411`); a spread = a finding.
  Nominal: -415.000 mm (`WORKCELL_HOME_TIP_ABOVE_ENTRANCE`,
  `insertion_tasks_cfg.py:1109`). The difference to it is REPORTED; no bar
  (unknown for the new joint angles). The number goes into
  `check_env_wiring.MEASURED_HOME_TIP_M` (`check_env_wiring.py:88`).
* **a3 the pull, a reading**: steps 1-6 read a depth that rises from step 1 on
  toward the box top, -291.5 mm (the clamp holds the target inside
  `osc_pos_clamp_z_m` around the entrance). Force above 1 N in the first steps
  is the arm accelerating (RT-143: 1.943 / 1.225 N), not contact. If step 1
  still reads the step-0 depth, the pull did not act -- a finding, not a FAIL
  of a2.

## Points -- RT-242d (reset force, high corner)

* **d1 setup**: `start_tip_above_entrance overridden: +270.0 mm above the
  stage-2 opening plane (low edge and start_floor_m set equal: fixed start, no
  floor).`; the three corner lines (30.000 mm radius, 10.0000 deg, 5.0000
  deg); `[zero_agent] probe angle set: fixture_pos_noise_xy 0.05 -> 0.0,
  fixture_pos_noise_z 0.05 -> 0.0, ...`. The line `[insertion] START FLOOR
  (SBC step 0)` must be ABSENT. Otherwise RUN INVALID.
* **d2 reset force** (D-179 (4) criterion (b), read as RT-197b was): PASS when
  all three RT-185 readings AND the RT-199c reading hold -- `force p50`
  0.000 N and `max` about 0 N in the lines `step 1` .. `step 3`, `>1N 0/256`,
  and the depth against the command: `step    0` depth p50 about -270 mm
  (the written pose; pocket frame, so the tilt does not change it), steps 1-6
  without a jump or a drift of the RT-198b/RT-199b kind (mm per step). The
  drift is REPORTED; RT-197b read 0.1 mm over 6 steps at 20 mm. If the
  readings disagree, the verdict names each; the user decides.

## Points -- RT-242b / b2 / c (start IK, 4096 episodes each)

* **setup b**: as d1. **setup b2**: the height line and `start_lateral_offset
  overridden: 30.000 mm`; NO `probe angle set` line (the spread stays live).
  **setup c**: `start_tip_above_entrance overridden: -30.0 mm ... (low edge and
  start_floor_m set equal: fixed start, no floor).`; no corner line. All three:
  `[insertion] START FLOOR (SBC step 0)` ABSENT.
* **solver clean** (D-179 (4) criterion (a), plan step 7): PASS when
  `start_pose_solve_flags_dropped_envs` = 0 in the run's
  `RT-242<x>_demo_metrics.json` with `episodes` >= 4096. FAIL: > 0, reported
  as a share of `episodes` (RT-185 text). Also REPORTED:
  `start_pose_solve_unconverged_resets`, `start_pose_solve_worst_residual_mm`
  against `start_pose_solve_tol_mm` 0.05 (`start_pose_solve_tol_m`,
  `insertion_env_cfg.py:588`); RT-197a's worst was 0.0447 mm at 20 mm.
* **the reach, a reading**: the `[zero_agent] --max-steps 4160 reached after
  4160 steps; stopping.` line. b2 and c keep the fixture spread live, so their
  arm reaches up to 50 mm further in every axis than in b.

Any FAIL -> back to the laptop before training (plan step 7).

## Pins (read from the files, not from the log prose)

In `RT-242b_demo_metrics.json`, `RT-242b2_demo_metrics.json`,
`RT-242c_demo_metrics.json`:

| key | b | b2 | c | source |
|---|---|---|---|---|
| `dr_mode` | `off` | `off` | `off` | cfg `:738`, zero_agent sets none |
| `start_tip_above_entrance_mm` | 270.0 | 270.0 | -30.0 | the flag |
| `start_floor_m` | `null` | `null` | `null` | floor off (`insertion_env.py:4416`) -- the fix |
| `start_lateral_offset_m` | 0.03 | 0.03 | 0.0 | the flag / cfg `:580` |
| `fixture_tilt_rad` | 0.174533 | 0.0 | 0.0 | the flag |
| `fixture_yaw_rad` | 0.087266 | 0.0 | 0.0 | the flag |
| `fixture_pos_noise_xy_m` | 0.0 | 0.05 | 0.05 | probe block / cfg `:783` |
| `fixture_pos_noise_z_m` | 0.0 | 0.05 | 0.05 | probe block / cfg `:792` |
| `obs_noise_pocket_pos_std_m` | 0.0 | 0.0 | 0.0 | zero_agent pin |
| `force_obs_noise_std_n` | 0.0 | 0.0 | 0.0 | zero_agent pin |
| `grasp_obs_offset_x_m` | 0.0 | 0.0 | 0.0 | zero_agent pin |
| `home_tip_above_entrance_mm` | 415.0 | 415.0 | 415.0 | nominal chain, `insertion_tasks_cfg.py:1109` |

RT-242a and RT-242d write no dump that counts (no episode ends in 6 steps).

## What this run cannot answer

* Anything about learning: RT-242s1..s5.
* The flat coarse reward kernel at 270 mm (`InBachelorErwähnen.md`
  2026-09-28): computed, not measured here.
* The IK at the corners BETWEEN the probed ones, and the rare disk edge: the
  lateral offset is a disk `r = R*sqrt(u)`, so r = 30 mm itself is rare
  (RT-185 text).
* Whether the reset force is clean at -30 mm: the floor phase starts there
  ON the axis; no force probe is asked for it (plan step 7).
* Whether the old home was 19 mm low: the old home no longer exists. RT-242a
  measures the new one.

## CORRECTION, after the run (2026-09-28)

All five runs are INVALID (`rt_logs/VERDICTS.md`, 2026-09-28; files in
`rt_logs/RT-242/`). `zero_agent.py` imported the editable `insertion` install,
and that install is bound to `Phase5_MitAktion_v1` (D-200 (1): only the
rl_games scripts put THIS clone's `source/insertion` first,
`scripts/rl_games/train.py:109`). Evidence: `[insertion] robot asset:
...\Phase5_MitAktion_v1\source\insertion\...` and `osc_pos_clamp_m = 0.08`,
`osc_pos_clamp_z_m = 0.1435` in RT-242b2/c; RT-242a/b/d stopped with
`AttributeError: 'InsertionEnvCfg' object has no attribute
'fixture_pos_noise_z'` (the old cfg has no such field). The gap in this file:
the P0 gate had no import line, so it could not see the wrong package; the
code-state section did not ask which folder `insertion` comes from.
Fix: `zero_agent.py` inserts `<clone>\source\insertion` first and prints
`[zero_agent] insertion package imported from: ...`. The re-run takes new
letters (a number is never re-issued), below.

## RT-242e / f / g / h / i -- added BEFORE the re-run (2026-09-28)

Status: **UNVERIFIED.** Same probes, same points, same pins as above; only
the letters, the import gate and the SHA change.

| new | = old | probe |
|---|---|---|
| RT-242e | a | home tip, `step    0` |
| RT-242f | b | IK high corner with the probe angles |
| RT-242g | b2 | IK high corner without angles, spread live |
| RT-242h | c | IK low corner -30 mm |
| RT-242i | d | reset force high corner |

**Code state**: SHA = the commit that adds this section, read back from
`origin/p5-rlgames`, given in chat. Since `99b1340d` only
`scripts/zero_agent.py` (import) and docs changed. [2026-09-28, later
commit: the same import line went into 17 other scripts (`HANDOFF-RL.md`,
NEXT 1); `zero_agent.py` and the env are unchanged, so RT-242e..i may run
on that SHA too.]

**P0 gate, added to every run** (before any other point):
`[zero_agent] insertion package imported from:
C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1\source\insertion\insertion`
and `[insertion] robot asset:` under the same `Phase5_RlGames_v1` folder. In
f/g/h/i the startup report must read `osc_pos_clamp_m = 0.1` and
`osc_pos_clamp_z_m = 0.2915` (`insertion_env_cfg.py:293`, `:316`). Any
`Phase5_MitAktion_v1` path = RUN INVALID.

Wall time: RT-242b2 ran 4160 steps x 256 envs in about 7.5 min, RT-242c in
about 9 min (header times; old env, indicative only).

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
git pull
git rev-parse HEAD
.\scripts\rt_log.ps1 RT-242e python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --home-pose --print-force --max-steps 6
.\scripts\rt_log.ps1 RT-242i python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --start-tip-above-entrance-mm 270 --start-lateral-offset-mm 30 --fixture-tilt 10 --fixture-yaw 5 --max-steps 6 --print-force
.\scripts\rt_log.ps1 RT-242f python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --start-tip-above-entrance-mm 270 --start-lateral-offset-mm 30 --fixture-tilt 10 --fixture-yaw 5 --max-steps 4160
if (Test-Path logs\demo_metrics.json) { Move-Item -Force logs\demo_metrics.json rt_logs\RT-242f_demo_metrics.json }
.\scripts\rt_log.ps1 RT-242g python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --start-tip-above-entrance-mm 270 --start-lateral-offset-mm 30 --max-steps 4160
if (Test-Path logs\demo_metrics.json) { Move-Item -Force logs\demo_metrics.json rt_logs\RT-242g_demo_metrics.json }
.\scripts\rt_log.ps1 RT-242h python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --start-tip-above-entrance-mm -30 --max-steps 4160
if (Test-Path logs\demo_metrics.json) { Move-Item -Force logs\demo_metrics.json rt_logs\RT-242h_demo_metrics.json }
$files = (Get-ChildItem rt_logs\RT-242e*, rt_logs\RT-242f*, rt_logs\RT-242g*, rt_logs\RT-242h*, rt_logs\RT-242i*).FullName
New-Item -ItemType Directory -Force C:\Users\Simon\Desktop\Alexander_Pett\Zips | Out-Null
Compress-Archive -LiteralPath $files -DestinationPath C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-242e.zip -Force
Get-Item C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-242e.zip
```

`Move-Item` instead of `Copy-Item`: a run that dies before its first dump
must not hand over the previous run's `logs\demo_metrics.json` under its own
name (RT-242 left one behind from RT-242c).

---

# RT-242s1..s5 -- expectation, written BEFORE the run (2026-09-28): the AutoDR arm, five seeds, 100 epochs, on D-204's code

Status: **UNVERIFIED.** Nothing in this section has run under Isaac.

Template: RT-240s1 and RT-240s2..s5 (`rt_logs/RT-240_expectation.md`).
Unchanged sections point there; only what differs is written out. Plan:
`Pläne/D-202_D-203_Bauplan.md` step 8.

## Context

RT-240s1 (the same command, old code) reached `Episode/dr/all_at_max` 1 at
epoch 55 and was stopped at ~107 (its OUTCOME, `rt_logs/RT-240_expectation.md`).
D-201 then set the budget to 100 epochs, checkpoints every 25, and named
what would make that wrong: "an RT-242 seed reaching `bounds_version` 75
late (after about epoch 80) or still rising in `rewards/iter` / falling in
`episode_lengths/iter` at epoch 100". Since RT-240s1 the task got harder
(D-204, probes PASSED as RT-242e..i): start ceiling 0.120 -> 0.270 m (AutoDR
height step 0.025 m), true fixture spread 0.005 m x/y -> 0.05 m x, y AND z
(live from epoch 1, not an AutoDR quantity), box 0.08 / 0.1435 -> 0.100 /
0.2915 m, home 0.415 m. A known risk against the bar: the coarse reward
kernel is flat at 270 mm (`InBachelorErwähnen.md` 2026-09-28; computed, not
measured). These five seeds are D-201's test and the AutoDR arm (D-195:
five seeds).

## Code state

Clone `Phase5_RlGames_v1`, branch `p5-rlgames`; the SHA is the commit that
adds this section, read back from `origin/p5-rlgames` and given in chat.
Since RT-240s1 (`de2c319`) under `scripts/rl_games` and `source`: the D-201
yaml (`max_epochs` 100, `save_frequency` 25, `rl_games_ppo_cfg.yaml:74-76`),
the D-204 env (`818f9af4`, `ce0cacc4`, `f3c5c417`), `play.py` views, and
comment-only line refs in `train.py`. `CODE_MARKER` =
`insertion-osc-2026-09-03a` (`insertion_env.py:76`), unchanged. New under
Isaac on THIS path: the AutoDR provider with the 0.270 m ceiling and the
fixture z column (RT-242e..i ran `dr_mode='off'`).

## The one hypothesis

**On D-204's code, every seed reaches the AutoDR ceiling
(`Episode/dr/all_at_max` = 1) at or before epoch 80 of 100 (D-201).**

Falsified by: a seed whose first epoch with `Episode/dr/all_at_max` 1 is
after 80, or never comes. D-201's second clause (still rising at 100) is P5,
reported for the user's judgement; D-201 gives it no number.

## Command

Training PC, PowerShell, `env_isaaclab` active, GPU free. The five seeds as
one chain (the user's RT-240 form); the chain can be stopped after any seed.
Wall time: RT-240s1 read 88.4-88.6 s per epoch at its end (`Iteration time`),
so about 2.5 h per seed and about 12 h for five; not measured on this code.

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
git pull
git rev-parse HEAD
.\scripts\rt_log.ps1 RT-242s1 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-242s2 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 2 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-242s3 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 3 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-242s4 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 4 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-242s5 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 5 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

After the chain, ONE zip (before any replay of these runs):

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
foreach ($n in 1..5) {
  $t = "RT-242s$n"
  $dirs = @(Get-ChildItem -Directory "logs\rl_games\ur5e_insertion\*__${t}_*")
  if ($dirs.Count -ne 1) { Write-Host "$t : expected ONE run folder, found $($dirs.Count)"; continue }
  $run = $dirs[0].FullName
  python scripts/check_episode_log.py --run-dir $run | Out-File -Encoding utf8 "rt_logs\${t}_eplog.txt"
  foreach ($f in "scalars.csv", "demo_metrics.json", "episodes.csv", "episodes_meta.json") { if (Test-Path "$run\$f") { Copy-Item -LiteralPath "$run\$f" "rt_logs\${t}_$f" } }
  Copy-Item -LiteralPath "$run\params\agent.yaml" "rt_logs\${t}_agent.yaml"
  Copy-Item -LiteralPath "$run\params\env.yaml" "rt_logs\${t}_env.yaml"
  Get-ChildItem -LiteralPath "$run\nn" | Select-Object Name, Length | Out-File -Encoding utf8 "rt_logs\${t}_nn.txt"
}
$files = (Get-ChildItem rt_logs\RT-242s*).FullName
New-Item -ItemType Directory -Force C:\Users\Simon\Desktop\Alexander_Pett\Zips | Out-Null
Compress-Archive -LiteralPath $files -DestinationPath C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-242s.zip -Force
Get-Item C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-242s.zip
```

## Points -- every seed (N = 1..5)

* **P1 setup** (the gate; any line wrong = void, precedent RT-151b): git =
  the chat SHA; cmd = the seed's command; `[rl_games] insertion package
  imported from: ...\Phase5_RlGames_v1\source\insertion\insertion`; `code
  marker: insertion-osc-2026-09-03a`; `[rl_games] env.shaping_gamma =
  0.995`; run folder
  `logs\rl_games\ur5e_insertion\<mm-dd_HH-MM-SS>__RT-242sN_autodr_offset50mm_currOFF_start+20mm_floor-30mm_seedN`
  (tag code `scripts/rl_games/train.py:121-168`; `offset50mm` is the new
  x/y spread, the z spread has no tag); `provider: AutoDR, 8 boundaries,
  bounds_version 0`; `START FLOOR (SBC step 0): -30.0 mm -> H_min +20.0 mm
  in 5 rungs`; controller lines as RT-242f L77-L80: `step limits: 0.02 m,
  0.097 rad`, `tip clamp: +-0.1 m x/y ... and +-0.2915 m z`, `yaw band:
  +-12.30 deg`; `action_ema_factor = 0.2`. The network lines (`Asymmetric-
  learning : True`, `RunningMeanStd: (34,)`, the 1024 LSTM) as RT-240s1 P1.
* **P1b** reported as RT-240s1 P1b (the first `reset yaw vs seat` line, void
  only at about 180 deg or >= 12.30 deg).
* **P2 console block**: `Learning iteration 1/100` .. `100/100`, ZERO
  `[rl_games-console] block NOT printed` lines (as RT-240s1 P2).
* **P3 the ceiling** (the hypothesis), from `RT-242sN_scalars.csv`, rows on
  the epoch axis (`Episode/*`, `*/iter`; `Pläne/D-200_Bauplan.md` part 2
  (6)): the first epoch with `Episode/dr/all_at_max` 1 -- PASS at <= 80; its
  `Episode/dr/bounds_version` there and at 100 (RT-240s1: 75, D-201); the
  first epoch with `Episode/dr/phase` 1 and the `reached H_min, phase dr`
  line; whether `all_at_max` falls back to 0 after it first reached 1.
* **P4 learning**, reported: all losses finite in every epoch (a NaN or inf =
  FAIL); `info/last_lr` min, max, and the count of epochs at exactly 1e-6 or
  1e-2 (RT-240s1: none over 1..106); `losses/cval_loss` at epochs 1, 25,
  50, 75, 100; `info/cval_lr` 1e-4 in every row.
* **P5 flat at 100?** (D-201's second clause), reported, no bar: means of
  `rewards/iter`, `episode_lengths/iter` and `Episode/success_rate` over
  epochs 81..90 and 91..100 (windows own, no source; RT-240s1 was read the
  same way over 56..65 vs 91..100 in `HANDOFF-RL.md`). The user judges
  "still rising".
* **P6 end**: `Learning iteration 100/100`; `MAX EPOCHS NUM!`;
  `RT-242sN_nn.txt` lists `last_ur5e_insertion_ep_{25,50,75,100}_rew_*.pth`
  and `ur5e_insertion.pth` (D-201); an `[autodr-state] saved into the
  checkpoint` line at every save; `[scalars] wrote`; `[rt_log] exit code: 0`.
* **P7 episode record**: `RT-242sN_eplog.txt` ends `VERDICT: COMPLETE`;
  `rollout_steps_per_iteration: 128` in `episodes_meta.json`.
* **P8 D-204 readings**, reported, no bar: `success_by_start_height_bin` in
  `demo_metrics.json` (edges -0.020 .. 0.020 m in 10 mm below H_min, then
  0.070, 0.120, 0.170, 0.220, 0.270 m; `insertion_env.py:1376-1377`); `stage1_lateral_y_mm.over_reach` (D-203: a metric, not a
  gate); the force keys of RT-240s1 P8 over epochs 1..7 and 94..100.

## Pins (read from the files, not from the log prose)

* `RT-242sN_agent.yaml`: `params.seed: N`; `config.max_epochs: 100`;
  `config.save_frequency: 25`; `config.minibatch_size: 1024`;
  `config.central_value_config.minibatch_size: 1024`;
  `config.horizon_length: 128`; `config.gamma: 0.995`;
  `config.learning_rate: 0.0001`; `network.rnn.units: 1024`. Source:
  `agents/rl_games_ppo_cfg.yaml`.
* `RT-242sN_env.yaml`: `dr_mode: autodr`, `obs_wrench_mode: force_prevact`,
  `start_floor_m: -0.03`, `autodr_stall_buffers: 3`, `pocket_obs_yaw_rad:
  0.0223402`, `seed: N`, `fixture_pos_noise_xy: 0.05`,
  `fixture_pos_noise_z: 0.05`, `osc_pos_clamp_m: 0.1`,
  `osc_pos_clamp_z_m: 0.2915` (`insertion_env_cfg.py:783`, `:792`, `:293`,
  `:316`).
* `RT-242sN_demo_metrics.json`: `fixture_pos_noise_xy_m: 0.05`,
  `fixture_pos_noise_z_m: 0.05`, `home_tip_above_entrance_mm: 415.0`,
  `dr_mode: autodr`.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| all five: `all_at_max` 1 by 80, P5 flat | D-201's budget holds on D-204's task | replay and the cross-seed reading (own expectation) |
| a seed reaches 1 after 80, or never | D-201 is wrong for this task (its own clause) | report the epochs; user decides on the budget |
| `all_at_max` 1 in time, but P5 still rising | D-201's second clause | report; user decides |
| `dr/phase` 0 through 100 | the floor phase is never left (new: the -30 mm floor under the 0.05 xyz spread) | report; user decides |
| stall on one boundary (its `_hi`/`_lo` stops below max) | name the boundary; `start_height` is the suspect (flat kernel) | report; user decides |
| Traceback or NaN | not a result | stop the chain; hand the Traceback back |

## What this run cannot answer

* Test success: no rl_games evaluation path (D-200 plan, "NOT BUILT").
* Which D-204 change made the task harder or easier: ceiling, spread and
  home change at once against RT-240s1.
* The No-DR and Fixed-DR arms (D-195): not in this chain.
* The cross-seed statistics: own expectation before they are read.

## OUTCOME RT-242s1..s5 (2026-09-28, files `rt_logs/RT-242s/`)

* P1 all five: git `e10c18b` (= the commit of this section), import from
  `Phase5_RlGames_v1`, marker, `shaping_gamma 0.995`, `provider: AutoDR, 8
  boundaries, bounds_version 0`, START FLOOR, step limits, clamp 0.1 /
  0.2915, yaw band 12.30, EMA 0.2, folder tags as expected. P1b: first
  `reset yaw vs seat` max 0.247..0.290 deg. PASS.
* P2: 100 blocks 1/100..100/100, zero `block NOT printed`. PASS.
* **P3 (the hypothesis): FALSIFIED.** First epoch with `all_at_max` 1:
  s1 74, s2 **88**, s3 80, s4 73, s5 **81**. `bounds_version` 75 there and
  at 100 in all five; never falls back. `dr/phase` 1 from epoch 13 / 14 /
  16 / 14 / 16. All seven bounds at their maximum at epoch 100.
* P4: zero non-finite cells; `last_lr` 1.3e-5..2.25e-4, never at 1e-6 or
  1e-2; `cval_lr` 1e-4 throughout; `cval_loss` @1/25/50/75/100 falls from
  1.35-2.19 to 0.019-0.024.
* P5 (means 81..90 -> 91..100): `rewards/iter` s1 560.4 -> 565.1, s2 552.6
  -> 552.1, s3 555.2 -> 565.3, s4 553.7 -> 565.0, s5 551.7 -> 561.7;
  `episode_lengths/iter` s1 47.1 -> 45.2, s2 50.0 -> 49.6, s3 49.8 -> 46.4,
  s4 48.0 -> 45.8, s5 49.7 -> 46.9; `Episode/success_rate` s1 .9963 ->
  .9953, s2 .9916 -> .9929, s3 .9825 -> .9896, s4 .9898 -> .9936, s5 .9943
  -> .9970. Reward and length still move in s3..s5; success is flat near
  0.99. The user judges "still rising".
* P6: 100/100, `MAX EPOCHS NUM!`, ep 25/50/75/100 + `ur5e_insertion.pth`,
  `[scalars] wrote`, exit 0, no Traceback. The best-reward file
  `ur5e_insertion.pth` was last written at epoch ~100 (s1), ~33 (s2),
  ~95 (s3), ~93 (s4), ~22 (s5) -> in s2 and s5 it is an EARLY policy from
  narrow bounds, not the full-width one. PASS.
* P7: all five `VERDICT: COMPLETE`, `rollout_steps_per_iteration: 128`. PASS.
* Pins: agent.yaml and env.yaml as listed, all five; demo_metrics
  `fixture_pos_noise_xy_m` / `_z_m` 0.05, home 415.0 mm, `dr_mode autodr`.
* Wall time 2.23-2.36 h per seed.
* Reading (table above): "a seed reaches 1 after 80" -> D-201 is wrong for
  this task by its own clause; user decides on the budget.
* Added 2026-09-28 (same data): the epoch BEFORE each first full epoch holds
  a fraction of `all_at_max` (the switch happens inside it): s1 73 0.602,
  s2 87 0.078, s3 79 0.609, s4 72 0.062, s5 80 0.547. So s5 reached the
  ceiling during epoch 80; by P3's wording (value 1) it is 81. s2 is late
  under either reading -- the FAIL does not hang on s5.
