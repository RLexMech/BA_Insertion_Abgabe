# RT-174 — expectation, written BEFORE the run (2026-09-07)

Written on the dev laptop. `/rt-check` judges the log against THIS file and
nothing else. The overnight run after RT-172 (tilt 0..5 deg, P5 branch 1)
and RT-173 (tilt 0..8 deg, P5 branch 1, `VERDICTS.md` 2026-09-07).

## The one hypothesis

**"The policy learns a LATERAL start offset of up to 6 mm on top of the
0..8 deg tilted pocket, under the reward as it stands."** No run has ever
trained lateral recovery: D-170 measured that the start solver puts every
episode on the pocket axis to within 0.035 mm, whatever `fixture_pos_noise_xy`
says. `start_lateral_offset` (D-170) is the field for it; it exists, is
wiring-checked offline (`check_env_wiring.py`), and has NEVER run on the
training PC (RT-161..163 were handed over, no log came back). This run is
its first measurement and the second axis of the user's target set
(tilt / lateral / yaw, 2026-09-07); yaw stays 0.

## The changes against the checkpoint, named

Base: RT-172's `model_600.pt` (folder
`09-06_18-57-23_offset5mm_tilt0-5deg_currOFF_start+30..+50mm_seed42`), the
same checkpoint RT-173 started from. NOT the RT-173 folder: RT-173 was
stopped by hand at about iteration 624, and with `save_interval` 50 its only
checkpoint is `model_600.pt` = one update of fine-tuning, so there is nothing
to gain from it. RT-173 measured that this checkpoint re-learns 0..8 deg in
three iterations (0.9059 at 601, 0.9980 at 603), so the tilt half of this run
has RT-173 as its baseline.

1. `env.fixture_tilt_noise_rad` 0.0873 -> 0.1396 (0..8 deg; RT-173's value,
   measured 1.0 / 1.0 / 0.9852 per bin).
2. `env.start_lateral_offset` 0.0 -> 0.006 m (USER-SET 2026-09-07; the top
   rung of the RT-161..163 ladder that never ran). Uniform +-6 mm on each
   pocket axis, drawn per env at every reset, added to the solver goal in
   the POCKET frame (D-170 (1)); the height command is untouched. Below the
   7.8847 mm `HOLE_REACH_OFFSET_Y` ceiling (D-170 (4)), so the shoulder-hole
   exposure cannot come from the START pose; it can still come from the
   policy's own motion, which is what `stage1_lateral_y_mm.over_reach`
   counts (RT-173 read 1 of 2000, max 9.33 mm; RT-172 0; RT-156 11).

Two changes on top of the checkpoint, NAMED. The tilt half is measured
(RT-173); the lateral half is what this run asks. There is NO
`success_by_lateral_bin` (not built); the lateral half is read only through
the overall rate and the tilt bins.

Everything else exactly as RT-172/RT-173: starts sampled +30..+50 mm
(`start_lateral_offset` is REFUSED with a negative low, D-170 (3) -- the band
is non-negative, so the guard passes), tip box 0.07, `contact_penalty_scale`
0.0, `far_penalty_scale` 0.0, 1024 envs, seed 42, force abort 50 N as
`truncated`, `fixture_pos_noise_xy` 0.005 (still cancelled at the start by
the solver, D-170), `fixture_yaw_noise_rad` 0.0. Reward unchanged (D-172).

## Budget

`--max_iterations 800` (mine, labelled: overnight; 800 x ~30.5 s = 6.8 h,
RT-173 read 30.2-31.2 s/iteration). Additive on resume -> first line
`Learning iteration 600/1400`. Checkpoints every 50 iterations. Env count
1024 (D-117 (c)).

## Commands (training PC, PowerShell, repo root, conda env `env_isaaclab`)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: the SHA in the handover message (read back from
`git ls-remote origin p4-reward` on the laptop AFTER the push).

```
.\scripts\rt_log.ps1 RT-174 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max_iterations 800 --resume --load_run 09-06_18-57-23 --checkpoint model_600.pt env.fixture_tilt_noise_rad=0.1396 env.start_lateral_offset=0.006 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.osc_pos_clamp_m=0.07 env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0
```

`--load_run` takes the TIMESTAMP only; `train.py --checkpoint` takes the
FILE NAME. Hydra floats need the decimal point. Run in `env_isaaclab`.

After the run (or a hand stop, which writes NO short log and NO clipboard):
`rt_logs\RT-174.txt` (full log), `logs\demo_metrics.json` (overwritten by
the next run -- copy first), and the scalars CSV via `export_tb_scalars.py
--run-dir <folder>`. Run folder tag expected to contain `tilt0-8deg`,
`lat6mm` and `start+30..+50mm`.

## PASS / FAIL lines, in order

1. **P1 — the reset is contact-free.** First block `force_abort_rate` NOT
   1.0; the pose checks in the startup report show `|prim - buffer|` ~1e-6
   and `max joint deviation` ~0 at step 2. The lateral offset does not
   change the vertical clearance (the tool axis is vertical at the start,
   RT-173 step-2 report: z-axis . (0,0,-1) = 0.999999), so RT-173's P1
   estimate (1.8 mm margin at the lowest draw, largest lever) carries.
   The startup report's env 0/1 `XY offset to entrance` must now read up to
   ~8.5 mm (6 mm per axis, plus the 5 mm fixture noise the report measures
   in the WORLD frame) instead of RT-173's <= 5 mm.
2. **P2 — the run is what it says.** Folder tag contains `tilt0-8deg`,
   `lat6mm` AND `start+30..+50mm`; startup report prints "rung-0 start
   lateral offset" ACTIVE with +-6.000 mm (`insertion_env.py:480-482`),
   "tip clamp: +-0.07 m", "fixture tilt noise (D-037): magnitude 0..8.00
   deg ... ACTIVE"; `[INFO]: Loading model checkpoint from:` ends in
   `09-06_18-57-23_offset5mm_tilt0-5deg_currOFF_start+30..+50mm_seed42\model_600.pt`;
   first block `Learning iteration 600/1400`; `demo_metrics.json`
   `start_lateral_offset_m` = 0.006.
3. **P3 — the account adds up.** `contact` and `far` EXACTLY 0.0; `abort`
   = -1.0 x `force_abort_rate`; `success_lump` > 0 iff `success_rate` > 0.
4. **P4 — the first blocks are READ.** First-block `success_rate` beside
   RT-173's 0.1190 (same checkpoint, no lateral) and the iteration at which
   the block rate first exceeds 0.90 (RT-173: iteration 601). A slower
   climb is the lateral cost; no per-lateral start value exists.
5. **P5 — THE BRANCH, read at the LAST block:**
   * `success_rate_recent` >= 0.95 AND every populated tilt bin >= 0.90 ->
     the unchanged reward pays for a 6 mm lateral start on top of 8 deg;
     the next axis is yaw (D-172 ladder continues, RT-152's unmeasured
     landscape named first).
   * `success_rate_recent` <= 0.50 -> the lateral start is the blocker
     under this reward (RT-173 read 0.996 without it). The failure mode
     is then read off a `play.py --trace-obs` replay WITH the same
     `env.*` overrides (RT-172 expectation, Replay NACHTRAG), not
     guessed; RT-151c/RT-153 measured NO usable lateral gradient near the
     centre, which is the pre-registered suspect.
   * Between the lines -> report the numbers and the trend; do not
     conclude.
6. **P6 — force and reach are READ.** `force_norm_n` p95 / max,
   `force_abort_rate` (RT-173: 25.99 / 38.46 N, 0.0) and
   `stage1_lateral_y_mm` max / `over_reach` (RT-173: 9.33 mm, 1). A rising
   `over_reach` is the D-153 exposure; it does not fail the run but must be
   reported beside the rate.
7. **P7 — sigma survives.** `Mean action noise std` at the last block
   > 0.30 (RT-173 ended at 0.57).
8. **P8 — the solver still lands OFF-AXIS.** `start_pose_solve_unconverged_resets`
   0 and `start_pose_solve_worst_residual_mm` under 0.05 (level 0.0354,
   5 deg 0.0400, 8 deg 0.0416 -- rising with tilt; the off-axis goal is new).

## What this run does NOT answer

* Per-lateral-offset success: there is NO `success_by_lateral_bin` table.
  Building one (pattern `_rate_by_bin`, `insertion_env.py:2120`) is the
  first instrument task before a lateral ladder, and it is NOT in this run.
* Yaw: `fixture_yaw_noise_rad` stays 0.0. The yaw reward landscape is
  unmeasured (RT-152 skipped) and the success gate's yaw window is the
  square proxy's (`YAW_WINDOW_RAD` +-3.96 deg, "OPEN" for the real part).
* The tilt x lateral x height cross; the per-bin START values.
* Anything above 8 deg (controller cone 8.52 deg).
