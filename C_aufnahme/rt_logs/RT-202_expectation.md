# RT-202 -- the No-DR twin of RT-201, written BEFORE the run (2026-09-14)

Branch `p5-kraftsensor`, code `145eaed9` or later (RT-201 ran on `d316f72e`;
the commits since touch check scripts only). RT-201 (`rt_logs/VERDICTS.md`)
reached all seven AutoDR ceilings at iteration 850 with the floor.

## The one question

**"Does the same floor, on the No-DR branch (two keys: the floor, then
`start_height_hi` up to 0.120 m), reach its target -- and in how many
iterations against RT-201's 850?"** RT-191 (band -30..+30 mm, no floor,
No-DR) reached 0.99 in 300 iterations.

## Design (user, 2026-09-14: RT-202 = No-DR twin)

Identical to RT-201 except `dr_mode=off`: floor -0.030 m, 5 rungs, H_min
0.020, seed 1, 256 envs, obs `force`, noise ON (cfg defaults),
`env.autodr_stall_buffers=3`, `--stop-when-dr-max 0.995`, cap 3000,
`--stop-check-every 50`. Under `off` the fixture stays at tilt 0 / yaw 0 and
`start_lateral_offset` 0 (the env refuses otherwise);
`fixture_pos_noise_xy` 0.005 stays live as in RT-191..194.

## Points

* **P1 setup**: `git:` = handed-over SHA; `dr_mode off`; `provider:
  AutoDR, 2 boundaries`; band table start_height [-0.030, +0.020] max
  [+0.020, +0.120]; START FLOOR line; folder `..._RT-202_offset5mm_currOFF_start+20mm_floor-30mm_seed1`.
* **P2 the floor moves**: five `[autodr] start_height_floor ... expand`
  lines, the fifth `-- reached H_min, phase dr`; `dr/phase` 0 -> 1 at
  `bounds_version` 5. RT-201 needed 55 iterations for this.
* **P3 the ceiling opens**: `[autodr] start_height_hi ... expand` lines,
  0.020 -> 0.120 in ten moves; `dr/bounds_version` ends at 15 if nothing
  ever shrinks.
* **P4 the six curves** as in RT-201; `Policy/mean_noise_std` not below 0.2.
* **P5 success**: `dr/train_success_regular_fresh` >= 0.995 over 2000;
  `success_by_start_height_bin` 10 bins, the five above +20 mm filled.
* **P6 stop**: `[stop-criterion] target reached` with `all boundaries at
  max True`; `autodr_<it>.json` format `autodr-3`, 2 keys.
* **P7 physics**: `force_abort_rate`, `max_depth_max_mm` (> 37 = FINDING),
  interpen.
* **P8 stagnation**: hold lines and `dr/*_holds`; a `STALL` note is a
  review trigger, read `_last_rate` and the height bins before any change.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| target reached | the floor works on both branches; iterations vs 850 is the comparison the thesis states | the wrench question (RT-195/196, other session) or the SBC step 1 plan |
| floor reaches H_min, ceiling stalls below 0.120 | the height ceiling alone is the hard part on No-DR | read the height bins and `_last_rate` |
| success collapses after the transition | D-110 at the phase switch | read the fresh window after `bounds_version` 5 |
| exploration dead (std < 0.2) | differential diagnosis point 5 | -- |

One seed decides nothing on its own.
