# RT-175 — expectation, written BEFORE the run (2026-09-07)

Written on the dev laptop. `/rt-check` judges the log against THIS file and
nothing else. Third axis of the user's target set (tilt / lateral / yaw):
the pocket YAW, +-6 deg (user, 2026-09-07: "minimal, max. 6 deg").

## The one hypothesis

**"The policy learns a pocket yawed by up to +-6 deg on top of tilt 0..8 deg
and a lateral start of +-6 mm, under the reward as it stands."**

## What is NOT known when this run starts, named

* **RT-174 is UNJUDGED.** Its files (full log, `demo_metrics.json`, CSV)
  had not reached the laptop when this run was handed over; the user
  reported it "successful" in chat and the folder name carries `lat6mm`.
  This run resumes RT-174's LAST checkpoint. If RT-174 turns out to be
  P5 branch 2 (lateral not learned), RT-175 inherits an unlearned lateral
  start and its yaw answer is confounded. `/rt-check RT-174` FIRST when
  the files arrive; that verdict decides how P5 below is read.
* **The yaw reward landscape has never been measured** (RT-152 skipped,
  `InBachelorErwähnen.md` 2026-09-05). Height and x/y were swept (RT-151c,
  RT-153); the rotation about the pocket axis was not.
* **The success gate's yaw window is the square proxy's.** `YAW_WINDOW_RAD`
  = +-3.96 deg (`insertion_tasks_cfg.py:289`, a 30 mm square in a 32 mm
  pocket); the startup report prints "yaw window: OPEN -- the proxy's C4
  window does not apply to a part that fits one way round" (D-121). At a
  pocket yaw of 6 deg the part must rotate by at least ~2 deg to satisfy
  the four-corner gate in the yawed pocket frame; what the REAL part's
  window is stands in no file.

## The one change

`env.fixture_yaw_noise_rad` 0.0 -> 0.1047 rad (6.00 deg; the tag rounds to
`yaw+-6deg`, `train.py:224`). Per episode, uniform in +-6 deg about the env
z axis through the pocket centre (D-038, `insertion_env_cfg.py:563-572`).
The per-env fixture quaternion carries tilt AND yaw, so the pocket frame,
the start solver goal, observation channels 19:21 (`yaw_cos_sin`, the
part's x-axis about the POCKET z-axis) and 21:25 (`pocket_quat`) all follow
the yawed pocket. The guard at `insertion_env.py:427-434` refuses yaw noise
only with a NEGATIVE start low; the band is +30..+50 mm, so it passes.

Everything else exactly as RT-174: tilt 0..8 deg (`0.1396`), lateral
`start_lateral_offset` 0.006, starts +30..+50 mm, tip box 0.07, contact 0,
far 0, 1024 envs, seed 42, force abort 50 N, `fixture_pos_noise_xy` 0.005.
Reward unchanged (D-172). Base: RT-174's folder (timestamp resolved on the
training PC as the newest run folder whose name contains `lat6mm`) and its
newest `model_*.pt`; the handover command prints both -- READ THEM OFF THE
LOG HEADER, they are not in this file.

Readout by construction: there is NO `success_by_yaw_bin` and NO
`success_by_lateral_bin`. Yaw is read only through the overall rate and the
tilt bins. `demo_metrics.json` carries `fixture_yaw_noise_rad` = 0.1047 as
the identity check.

## Budget

`--max_iterations 500` (mine, labelled; ~4.2 h at ~30.5 s/iteration).
Additive on resume: the first line reads `Learning iteration N/N+500` with
N = the loaded RT-174 iteration. Checkpoints every 50. 1024 envs (D-117).

## Commands (training PC, PowerShell, repo root, conda env `env_isaaclab`)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: the SHA in the handover message.

```
$run = Get-ChildItem logs\rsl_rl\ur5e_insertion -Directory | Where-Object { $_.Name -like '*lat6mm*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1; $ts = $run.Name.Substring(0,14); $ck = (Get-ChildItem $run.FullName -Filter model_*.pt | Sort-Object { [int]($_.BaseName -replace 'model_','') } | Select-Object -Last 1).Name; "RT-174 folder: $($run.Name)  load_run: $ts  checkpoint: $ck"
.\scripts\rt_log.ps1 RT-175 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max_iterations 500 --resume --load_run $ts --checkpoint $ck env.fixture_yaw_noise_rad=0.1047 env.fixture_tilt_noise_rad=0.1396 env.start_lateral_offset=0.006 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.osc_pos_clamp_m=0.07 env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0
```

The first line prints the resolved folder, timestamp and checkpoint; the
`[rt_log] cmd:` header of the log records the expanded values. Run folder
tag expected to contain `tilt0-8deg`, `yaw+-6deg`, `start+30..+50mm` and
`lat6mm`.

## PASS / FAIL lines, in order

1. **P1 — the reset is contact-free.** First block `force_abort_rate` NOT
   1.0; startup report step 2: `|prim - buffer|` ~1e-6, `max joint
   deviation` ~0, and the `entrance buffer` line now shows a NON-zero
   `yaw` for env 0/1 (RT-173/174 printed `yaw +0.0000 rad`).
2. **P2 — the run is what it says.** Folder tag contains `yaw+-6deg` AND
   `lat6mm` AND `tilt0-8deg`; startup report "fixture pose noise (D-034):
   xy +-0.005 m, yaw +-0.1047 rad (ACTIVE ...)"; "fixture tilt noise:
   0..8.00 deg ACTIVE"; "start lateral offset ... +-6.000 mm"; the
   `[INFO]: Loading model checkpoint from:` line names the RT-174 folder
   (`lat6mm` in its name) and the checkpoint the handover line printed;
   `demo_metrics.json` `fixture_yaw_noise_rad` 0.1047,
   `start_lateral_offset_m` 0.006, `fixture_tilt_noise_rad` 0.1396.
3. **P3 — the account adds up.** `contact`, `far` EXACTLY 0.0; `abort` =
   -1.0 x `force_abort_rate`; `success_lump` > 0 iff `success_rate` > 0.
4. **P4 — the first blocks are READ.** First-block `success_rate` beside
   RT-174's LAST-block rate (the loaded policy on the same task minus
   yaw); the drop is the yaw cost at iteration 0. Iteration at which the
   block rate first exceeds 0.90.
5. **P5 — THE BRANCH, at the LAST block, read AFTER RT-174's verdict:**
   * `success_rate_recent` >= 0.95 AND every populated tilt bin >= 0.90
     -> the unchanged reward pays for +-6 deg yaw on top of tilt and
     lateral; the three-axis target set is reached at these magnitudes.
     The next question is the per-axis instrument (bins for lateral and
     yaw) and the play (D-087), not the reward.
   * `success_rate_recent` <= 0.50 -> yaw is the blocker under this
     reward. Suspects, pre-registered: (i) the unmeasured yaw landscape
     (RT-152); (ii) the square gate's yaw window (D-121). Read the failure
     off a `play.py --trace-obs` replay WITH all eight `env.*` overrides
     (`trace_outcome_split.py` prints the yaw error); do not guess.
   * Between the lines -> report the numbers and the trend.
   * If RT-174 was itself branch 2, P5 here is CONFOUNDED and is reported
     as such, not concluded.
6. **P6 — force and reach are READ.** `force_norm_n` p95 / max,
   `force_abort_rate`, `stage1_lateral_y_mm` max / `over_reach`, beside
   RT-174's values.
7. **P7 — sigma survives.** `Mean action noise std` at the last block
   > 0.30.
8. **P8 — the solver still lands.** `start_pose_solve_unconverged_resets`
   0, `start_pose_solve_worst_residual_mm` < 0.05 (8 deg + lateral: RT-174's
   value, not yet read).

## What this run does NOT answer

* Per-yaw and per-lateral success (no bin tables); the cross of the three
  axes; per-bin START values.
* The real part's yaw window (D-121 reopened, not recomputed).
* Anything above 8 deg tilt, 6 mm lateral, 6 deg yaw.
