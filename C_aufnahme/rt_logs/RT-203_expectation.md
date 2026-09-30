# RT-203 -- RT-201 repeated with the wrench observation and torque noise, written BEFORE the run (2026-09-14)

Branch `p5-kraftsensor`, code `04df4754`. **UNVERIFIED.** RT-201
(`rt_logs/VERDICTS.md`, 2026-09-14 04:27 entry) reached
`[stop-criterion] target reached at iteration 850`, all seven boundaries at
max, fresh 2000/2000 at 0.9955, `bounds_version` 75, force-28, obs mode
`force`. This is NOT a `/decision`. The user explicitly marked this an open
question (2026-09-14): the run decides whether wrench is used further, no
D-entry follows from it alone.

## The one question

**"Does the same floor-unlocked AutoDR machine that reached its target in
850 iterations under `force-28` (RT-201) reach the same target under
`wrench-31` with torque noise 0.2 Nm, and does the RT-195/196 under-pocket
depth path (`max_depth_max_mm` > 37 mm) reappear here?"**

## Design -- exactly ONE change against RT-201

Everything from RT-201 (`rt_logs/RT-201_expectation.md`) unchanged:
`dr_mode=autodr`, floor -0.030 m, 5 rungs, H_min 0.020, seed 1, 256 envs,
observation noise ON (cfg defaults: pocket 0.0025 m, force 3.5 N, grasp
0.003 m), `env.autodr_stall_buffers=3`, `--stop-when-dr-max 0.995`, cap
3000, `--stop-check-every 50`.

The one change: `env.obs_wrench_mode=wrench env.torque_obs_noise_std_nm=0.2`
(RT-201 ran `force`, torque_obs_noise_std_nm 0.0 per
`rt_logs/RT-201/RT-201.short.txt:91`).

Marker: `git:` = `04df4754` or later; if the training PC is behind, pull
first:
```
git -C <root> fetch origin p5-kraftsensor
git -C <root> checkout p5-kraftsensor
git -C <root> pull origin p5-kraftsensor
git -C <root> rev-parse HEAD
```
Expected SHA after pull: `04df4754` (or a later commit on
`origin/p5-kraftsensor` -- verify with the same `git rev-parse HEAD`
before handing the run over; a diverging SHA voids the run per the
RT-151b precedent, `rt_logs/VERDICTS.md`).

Run folder name: expected the same tag body as RT-201, just the run number
swapped -- `..._RT-203_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1`.
The folder tag does NOT carry the obs mode (`scripts/rsl_rl/train.py:450`
comment shows only the `RT-<n>_<...>` pattern from the passed-in tag; no
`obs_wrench_mode` read feeds the folder name -- grepped, no match). The obs
mode is distinguishable only from the startup print and `demo_metrics.json`,
not from the folder name.

## Command (PowerShell, training PC)

```
.\scripts\rt_log.ps1 RT-203 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations 3000 --stop-check-every 50 --stop-when-dr-max 0.995 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=wrench env.torque_obs_noise_std_nm=0.2
```

## Points, in the order RT-201/RT-202 read them (same convention, no
automated `judge_run` exists for train runs -- these are read by hand
per `/rt-check` against `rt_logs/VERDICTS.md` RT-201/RT-202 entries)

* **P1 setup**: `git:` = `04df4754` or the pulled SHA; startup print line
  (`insertion_env.py:4584-4587`) reads `obs mode wrench-31: torque 28:31 IN
  the policy observation; torque_obs_noise_std_nm 0.2 (0 = off, OPEN per
  D-188); the wrench EMA skips the reset step and advances once per step`;
  `provider: AutoDR, 8 boundaries`; band table `start_height [-0.030,
  +0.020]`, others at centre; `demo_metrics.json` shows `obs_wrench_mode
  wrench`, `torque_obs_noise_std_nm 0.2`, pocket/force/grasp sigmas at cfg
  defaults; run folder `..._RT-203_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1`;
  no width-mismatch refusal, no Traceback, no NaN.
* **P2 the floor moves**: five `[autodr] start_height_floor: rate ... ->
  expand` lines, the fifth carrying `-- reached H_min, phase dr`;
  `dr/start_height_floor` -0.030 -> +0.020, `dr/phase` 0 -> 1,
  `dr/bounds_version` 5 at the transition. RT-201 needed 55 iterations for
  this -- read RT-203's iteration against 55, not against a new guess.
* **P3 the seven open**: after the transition `dr/lat_r_hi`,
  `dr/yaw_lo/hi`, `dr/tilt_hi`, `dr/start_height_hi`, `dr/friction_lo/hi`
  leave their centres; `[autodr]` expand lines for each; `dr/flags_dropped`
  stays low (RT-201: 31 single cases, no pattern -- read RT-203's count the
  same way, not against a fixed bar).
* **P4 the six curves** (reward, surrogate, value, entropy; EV/KL not
  logged in this rsl_rl version per RT-193/194/195/196), with trend;
  `Policy/mean_noise_std` must not collapse below 0.2 before the boundaries
  open (RT-189s1 died at it 215 with std < 0.2; RT-201 stayed >= 0.76,
  minimum before the transition 0.95).
* **P5 success**: `dr/train_success_regular_fresh`; `Episode/success_rate`;
  `success_by_start_height_bin` (10 bins, the five below +20 mm empty until
  the floor phase ends); `success_by_lateral/yaw/tilt_bin` once those open.
* **P6 stop**: `[stop-criterion] target reached` with `all boundaries at
  max True` and fresh success >= 0.995 -- or the cap (3000) with the bounds
  where they stand; `autodr_<it>.json` `"format": "autodr-3"` (same sidecar
  version as RT-201/RT-202, `env.autodr_stall_buffers=3` on the command
  line).
* **P7 physics**: `force_abort_rate`; `max_depth_max_mm` -- the RT-195/196
  under-pocket path is `> 37` (RT-195: 393 iterations, peak 144.4 mm;
  RT-196 wrench+torque 0.2, No-DR, seed 20: 224 iterations, peak 143.8 mm;
  RT-194 wrench, torque 0, seed 20: 0 iterations; RT-201 force, AutoDR,
  seed 1: 36.0 mm end value, no reported > 37 excursion) -- read whether it
  appears here, under AutoDR + wrench + torque noise, a combination none
  of RT-194/195/196/201 ran; interpen max / p99 (RT-201: 3.4 mm / 0.94 mm,
  an open, not-investigated finding -- read the same fields here, do not
  resolve it in this run).
* **P8 stagnation**: hold lines (`[autodr] <key>: rate ... over 240 ->
  hold ...`), `dr/<key>_last_rate`, `dr/<key>_holds`; a `STALL` note
  (n >= 3) is a review trigger, not a change -- bounds and
  `dr/bounds_version` must stay put on that line (RT-201: 8 hold lines,
  max 1 hold per boundary, no STALL).
* **P9 torque channel in the loop**: `torque_norm_nm` p50/p95/max and
  `wrench_valid_frac` from `demo_metrics.json`, read against RT-196
  (wrench, torque 0.2, No-DR, seed 20, 600 it: p50 0.31, p95 0.77, max
  1.90, valid 0.949) and RT-194 (wrench, torque 0, No-DR, seed 20: p50
  0.28, p95 0.80, max 1.73, valid 0.948) -- RT-203 is the first of these
  under AutoDR (moving boundaries change the contact distribution the
  torque channel sees), so a difference from RT-196 does not by itself
  say AutoDR vs. wrench; read `exploration std < 0.2` (differential
  diagnosis point 5) the same way as P4.

## Pins (from `demo_metrics.json`, field names as the run writes them)

* `obs_wrench_mode`: `"wrench"` -- command-line override, one change
  against RT-201's `"force"`.
* `torque_obs_noise_std_nm`: `0.2` -- command-line override
  (`insertion_env_cfg.py:1145` default is `0.0`; RT-196 used the same
  0.2 value, cited there as the datasheet-precision decision,
  `docs/decisions_inbox.md` "The torque channels get the datasheet
  precision as their noise").
* `fixture_pos_noise_xy_m` / `obs_noise_pocket_pos_std_m`: cfg default
  0.0025 m (unchanged from RT-201, not read here -- read back from the
  JSON, not assumed).
* `force_obs_noise_std_n`: cfg default 3.5 N (unchanged from RT-201).
* `grasp_obs_offset_x_m` / grasp noise: cfg default 0.003 m (unchanged
  from RT-201).
* `dr_mode`: `"autodr"`, `start_floor_m`: `-0.030`, `autodr_stall_buffers`:
  `3` -- all unchanged from RT-201, read back to confirm the ONE change
  did not silently touch anything else.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| target reached faster than 850 it, all seven at max, no > 37 mm depth path, torque stats close to RT-196 | wrench carries no cost here either; still one seed, not a verdict for wrench | a second seed before any claim |
| target reached slower than 850 it or a boundary stalls (STALL note) | AutoDR interacts with the wrench channel or the added noise dimension; read which boundary and its `_last_rate`/bin table | per-boundary diagnosis, same as RT-201's own stall branch |
| > 37 mm depth path reappears (as in RT-195/196) | the under-pocket path is tied to noise/torque, not to dr_mode -- occurs under AutoDR too | compare against RT-201's clean 36.0 mm end value; still not investigated as a cause |
| success collapses after the floor->dr transition | catastrophic forgetting at the phase switch (D-110), same check as RT-201 P5 | read the fresh window right after `bounds_version` 5 |
| exploration dead (std < 0.2) before the transition | RT-189s1's failure mode returns despite floor + wrench | differential diagnosis point 5 |

One seed decides nothing on its own -- neither for nor against using the
wrench channel. This run is an open question, not a decision; no
`/decision` and no D-entry follow from RT-203 alone.

## What this run cannot answer

* Whether wrench helps or hurts AutoDR convergence in general -- one seed,
  one direction of comparison (RT-203 vs. RT-201), no repeated seed.
* Whether the > 37 mm depth path (if it appears or not) is caused by
  wrench, by AutoDR's moving boundaries, by noise, or by their
  interaction -- RT-194/195/196 varied wrench under No-DR only; RT-201
  varied the floor under force-28 only; no prior run varies both at once
  except this one, so the cause cannot be isolated from RT-203 alone.
* The interpen finding open since RT-193/194 (max 3.4 mm / p99 0.94 mm in
  RT-201) -- this run reads the same fields but does not investigate the
  cause.
* Whether the torque channel's value changes under moving lateral/yaw/tilt
  boundaries (never tested before this run) -- RT-196's torque stats are
  the only wrench+noise reference, but under a FIXED band, not AutoDR's
  expanding one.
* Anything about `play.py` replay behaviour, checkpoint width handling, or
  a resumed run -- RT-203 is a single `train.py` invocation, not a replay.
