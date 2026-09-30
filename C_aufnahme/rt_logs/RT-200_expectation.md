# RT-200a / RT-200b / RT-200c -- smoke of the START FLOOR (SBC step 0), written BEFORE the runs (2026-09-14)

Branch `p5-kraftsensor`. **UNVERIFIED** -- the floor has never run under
Isaac. Build: `Pläne/SBC_Schritt0_Entwurf.md` § 0.1b, plan
`synthetic-marinating-mitten.md`. Two iterations each, nothing else running.
These runs answer "does it build and wire", NOT "does the floor learn".

## The three runs

| RT | what | command tail |
|---|---|---|
| RT-200a | autodr + floor | `env.dr_mode=autodr env.start_floor_m=-0.030 --stop-when-dr-max 0.8` |
| RT-200b | off + floor (No-DR branch) | `env.dr_mode=off env.start_floor_m=-0.030 --stop-when-dr-max 0.8` |
| RT-200c | resume of RT-200a from its `model_1.pt` | `--resume --load_run <RT-200a timestamp> --checkpoint model_1.pt` plus the RT-200a tail |

All: `--num_envs 256 --headless --seed 20 --max_iterations 2 --stop-check-every 1`,
obs mode `force` (cfg default), pull SHA in the handover.

## P1 setup (each run) -- otherwise RUN INVALID

* `git:` = handed-over SHA; no Traceback/ValueError.
* startup line `[insertion] START FLOOR (SBC step 0): ... floor -30.000 mm ... 5 rungs`.
* startup report: `dr_mode: autodr` (a) / `off` (b); `provider: AutoDR, 8 boundaries` (a) / `2 boundaries` (b);
  band table: `start_height  [-0.030000, +0.020000]`, every other listed
  quantity `[c, c]` (a: lat_r/yaw/tilt `[0,0]`, friction `[0.14,0.14]`; b: only
  start_height is listed); line `START FLOOR (SBC step 0): -30.0 mm -> H_min
  +20.0 mm in 5 rungs; now -30.0 mm, phase floor`.
* run folder name carries `floor-30mm` and `start+20mm` (the high is 0.020 now).

## P2 the machine

* `[stop-criterion] iteration 1/2: ... all boundaries at max False`.
* `scalars.csv`: `dr/start_height_floor` = -0.030, `dr/start_height_floor_fill` >= 0,
  `dr/phase` = 0, `dr/flags_dropped` = 0, `dr/bounds_version` = 0 (two
  iterations cannot fill 240 flags: 256 envs x ~0.5 boundary share x
  episodes finished in 2 iterations).
* `autodr_1.json` (sidecar): `"format": "autodr-2"`, `"floor": {"f0": -0.03,
  "top": 0.02, "n_steps": 5}`, keys list ends with `start_height_floor` (a: 8
  keys, b: 2 keys).
* `demo_metrics.json`: `start_floor_m` -0.03, `start_floor_steps` 5,
  `dr_state.phase` "floor", `success_by_start_height_bin` has 10 bins and the
  episodes sit in the bins from `< -20` to `10..20` only.
* (c) resume: `[autodr-state] loaded ... bounds_version 0, all_at_max False,
  phase floor`, no refusal.

## P3 the physics is unchanged

* `force_abort_rate` well below 1.0 (RT-120 shape absent: the start is on the
  axis, angles 0), `start_pose_solve_flags_dropped_envs` 0,
  `start_pose_solve_unconverged_resets` 0.
* Exit line may be missing (open since RT-192b); the `Training time:` line
  and `[scalars] wrote` line stand in.

## Readings

| Outcome | Reading | Next |
|---|---|---|
| all P1-P3 green in a, b, c | the floor is wired; the real run follows (its own expectation, 600+ iterations, `--stop-when-dr-max`) | write RT-201 expectation |
| a KeyError / ValueError at build | a wiring miss the offline checks did not see | fix, re-run the same smoke |
| (b) refuses `--stop-when-dr-max` | the train.py predicate did not see the floor | fix train.py |
| (c) refuses the sidecar | the `autodr-2` block differs from the configured floor | read both, fix the loader |
