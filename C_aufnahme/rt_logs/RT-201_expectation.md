# RT-201 -- the first AutoDR run WITH the start floor, written BEFORE the run (2026-09-14)

Branch `p5-kraftsensor`, code `8555dbf3` (floor build `4ee1abfe` + two verdict
lines). **UNVERIFIED as a learning run**: RT-200a/b showed only that the
floor builds and wires. RT-200c (resume) was SKIPPED by the user -- a resume
of this run is therefore unproven; if it dies, it restarts, it does not
resume.

## The one question

**"Does a policy that starts inside the pocket (floor -30 mm) climb to H_min
and then open all seven AutoDR boundaries to their ceilings?"** RT-189s1
(fixed start +30 mm, no floor) reached 0 success in 3000 iterations; RT-191
(band -30..+30, No-DR) reached 0.99 in 300.

## Design (user, 2026-09-14: "sauber 201 als ersten Lauf mit seed 1")

`dr_mode=autodr`, floor -0.030 m, 5 rungs, H_min 0.020, seed 1, 256 envs, obs
mode `force` (cfg default; the wrench question RT-195/196 is judged in the
other session), observation noise ON (cfg defaults: pocket 0.0025 m, force
3.5 N, grasp 0.003 m), `--stop-when-dr-max 0.995` and cap 3000 iterations
like RT-189s1, `--stop-check-every 50`. One change against RT-189s1: the
floor (and H_min 20 instead of 30 mm as the centre).

Expected duration: UNKNOWN for this setup. Floor phase ~5 x 32 iterations
if every buffer advances (RT-188 fill rate, all boundary episodes on one
key); the seven boundaries then fill in parallel; the fresh window needs
2000 regular episodes after the LAST move. RT-194 ran 6.7-10.6 s/it.

## Points

* **P1 setup**: `git:` = handed-over SHA; `START FLOOR` line; `provider:
  AutoDR, 8 boundaries`; band table start_height [-0.030, +0.020], others
  [c, c]; run folder `..._RT-201_autodr_..._start+20mm_floor-30mm_seed1`.
* **P2 the floor moves**: `[autodr] start_height_floor: rate ... -> expand`
  lines, five of them, the fifth carrying `-- reached H_min, phase dr`;
  `dr/start_height_floor` -0.030 -> +0.020 in the CSV, `dr/phase` 0 -> 1,
  `dr/bounds_version` 5 at the transition. Every boundary episode until then
  is on the floor (`dr/*_fill` of the seven stays 0 in phase floor).
* **P3 the seven open**: after the transition `dr/lat_r_hi`, `dr/yaw_lo/hi`,
  `dr/tilt_hi`, `dr/start_height_hi`, `dr/friction_lo/hi` leave their centres;
  `[autodr]` expand lines for them; `dr/flags_dropped` stays 0.
* **P4 the six curves** in the fixed order (reward, surrogate, value,
  entropy; EV/KL not logged), with trends; `Policy/mean_noise_std` must NOT
  collapse below 0.2 before the boundaries are open (RT-189s1 died at it
  215 with std < 0.2).
* **P5 success**: `dr/train_success_regular_fresh` over the run;
  `Episode/success_rate`; `success_by_start_height_bin` in demo_metrics with
  the 10 bins; `success_by_lateral/yaw/tilt_bin` once those open.
* **P6 stop**: `[stop-criterion] target reached` with `all boundaries at max
  True` and fresh success >= 0.995 -- or the cap with the bounds where they
  stand; `autodr_<it>.json` with `"format": "autodr-2"`.
* **P7 physics**: `force_abort_rate`, `max_depth_max_mm` (> 37 = the
  under-fixture path, RT-193 it 200, FINDING), interpen.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| target reached, all seven at max | the floor unlocks AutoDR; the No-DR twin (seed 1, `dr_mode=off`, same floor) is the next run | RT-202 = No-DR twin |
| floor reaches H_min, some boundaries stall below max at the cap | the floor works; which boundary stalls is the finding (read its `_fill` and its bin table) | per-boundary diagnosis before any change |
| floor never leaves -0.030 with boundary success >= 0.80 | a wiring defect in record/update for the floor key | read `dr/start_height_floor_fill` against 240 |
| success collapses after the transition (phase dr) | catastrophic forgetting at the phase switch (D-110) | read the fresh window right after `bounds_version` 5 |
| exploration dead (std < 0.2) before the transition | RT-189s1's failure returns despite the floor | differential diagnosis point 5 |

## Added BEFORE the start (2026-09-14, user proposal 1 and 2): stagnation readings

The run was NOT started on d46a2463; it starts on the commit that adds the
stall readings (`autodr-3` sidecar), with `env.autodr_stall_buffers=3`
(the reviewer's proposal: three full buffers in a row without a move; one
buffer of one key needs 240 boundary flags, so in phase dr with seven keys
sharing the boundary episodes that is roughly 3 x 200 iterations).

* **P8 hold lines**: a full buffer between the thresholds now writes
  `[autodr] <key>: rate 0.4000 over 240 -> hold <v> -> <v>` (before this
  commit it wrote nothing). `dr/<key>_last_rate` is the rate of the last
  FULL buffer (-1 before the first), `dr/<key>_holds` the full buffers in a
  row without a real move (clamped_zero counts, clamped_max does not).
* **P9 the STALL note**: a hold line ending in `-- STALL n full buffers
  without a move (review bar 3)` with n >= 3 is a REVIEW TRIGGER, not a
  change: the bounds and `dr/bounds_version` must be unchanged on that
  line. Reading: that key sits between 0.10 and 0.80 for ~600 iterations;
  read its `_last_rate` curve and its bin table before touching anything.
* **Proposal 2 (stop rule) needed no build**: `_criterion_dr_max` already
  requires all boundaries at max AND >= 2000 regular episodes in the fresh
  window of the CURRENT bounds_version AND their rate >= 0.995
  (`scripts/rsl_rl/train.py`, `[stop-criterion]` line).

One seed decides nothing on its own.
