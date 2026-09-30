# RT-195t -- expectation, written BEFORE the run (2026-09-14)

Branch `p5-kraftsensor`. **UNVERIFIED.** A play trace, no training. It asks
where the impossible depth of RT-193 / RT-195 / RT-196 comes from
(`max_depth_max_mm` plateaus at 143.5..144.4 mm for up to 393 iterations; the
pocket floor is at 36 mm; verdicts RT-195 and RT-196 in `VERDICTS.md`).

## The one question

**"When the leading tool point goes deeper than the underside of the
cut-out (`POCKET_ASSET_BOTTOM_Z = -0.046`) while laterally inside the pocket
walls, did it get there AROUND the free-standing cut-out, or THROUGH its
floor?"**

## What is known before the run (read in code, 2026-09-14)

* The plateau value equals `osc_pos_clamp_z_m = 0.1435` (the z half-width
  of the tip clamp box around the entrance) -- nothing but the clamp stops
  the part there.
* `spawn_workcell_block: false` in all three runs: the cut-out stands free;
  below -0.046 m is air.
* The depth gate is lateral only (`insertion_math.in_pocket_cross_section`,
  `|x| < POCKET_WALL_X`, `|y| < POCKET_WALL_Y`); no lower bound.
* Outer body of the cut-out: x -0.0848..+0.0652 m, y -0.0887..+0.1005 m
  (`POCKET_LOCAL_X_RANGE` / `_Y_RANGE`). The clamp box is +-0.08 m in x/y,
  so only the +x side leaves room beside the body (15 mm). Whether the part
  fits through that gap: UNKNOWN.
* `play.py` uses `get_inference_policy` = the mean action, no exploration
  noise. The training events may need that noise. Unknown.

## Design

* Checkpoint: RT-195 `model_200.pt` (inside its deep window, It 47..439;
  `mean_max_depth_mm` 45.4 at It 200 -- the highest share of deep episodes in
  any of the three runs). `save_interval = 50`, so the file exists.
* RT-195's run folder got NO RT name from the code (`4d1860f` has no
  `RT_NAME`); the user added "195" to the folder name by hand on the
  training PC (chat, 2026-09-14). Exact new name not seen; the filter is
  `*195*` and `$run` must print one folder.
* Current code, RT-195's start band restored by hydra: the default
  `start_tip_above_entrance` is 0.020 now (was 0.030 in RT-195, demo_metrics
  `start_tip_above_entrance_mm` 30.0). `start_floor_m` defaults to the same
  constant and must equal the high edge, or `__init__` refuses the static
  band. Noise, obs mode (`force`), dr_mode (off) are cfg defaults, as in
  RT-195.
* 64 envs, 512 steps (two full episodes at most; successes end earlier).

## Command (training PC, PowerShell, clone `Phase5_Robustheit_v1`, NOTHING else running)

```
$run = (Get-ChildItem logs\rsl_rl\ur5e_insertion -Directory -Filter "*195*").FullName
$run
.\scripts\rt_log.ps1 RT-195t python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 64 --headless --checkpoint "$run\model_200.pt" --trace-obs rt_logs/RT-195t_trace.json --trace-steps 512 env.start_tip_above_entrance=0.030 env.start_tip_above_entrance_low=-0.030 env.start_floor_m=0.030
```

`$run` must print exactly ONE folder. Hand over: `rt_logs\RT-195t.txt` and
`rt_logs\RT-195t_trace.json`.

## Points

* **P1 setup.** No Traceback; `git:` = the handed-over SHA; the loaded
  checkpoint path is the RT-195 folder (`09-13_22-38..`, "195" in the name) `\model_200.pt`; obs mode `force-28`;
  startup report shows the start band -0.030..+0.030 and NO start floor;
  `workcell block: *** OFF`. Trace: `truth.tip_true_m` has 512 rows x 64 envs.
  Otherwise RUN INVALID.
* **P2 event count.** Event E = a step with `tip_true_m` z < -0.046 AND
  |x| < 0.0452938 AND |y| < 0.07255. Count the envs with at least one E.
* **P3 path of each E-env** (first E only), from the last step with
  z >= -0.036 up to the first E step:
  * AROUND = at least one step in that stretch lies outside the outer body
    (x > 0.0652 or x < -0.0848 or y > 0.1005 or y < -0.0887);
  * THROUGH = every step of that stretch lies inside |x| < 0.0453 and
    |y| < 0.0726.
  Also read `truth.interpen_max_m` over the stretch (> 0 = touching; 0 does
  NOT prove free) and `done` (reset steps).
* **P4 reset step.** Is any first E on a step whose previous `done` is True
  (a reset row)?

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| E > 0, all AROUND | cause 1: the part leaves beside the free-standing cut-out and comes back under it; a scene artefact of the missing block | decide with the user: lower bound on the depth gate (floor z) vs. block on vs. clamp box -- a decision, not a fix in this entry |
| E > 0, all THROUGH, interpen > 0 in the stretch | cause 2: the part is pushed through the 10 mm floor | read the force on those steps; check the fixture collider (SDF) on the training PC |
| E > 0, THROUGH, interpen 0 throughout | cause 3 (collider hole) or a missed Warp query -- not separable here | read the fixture USD collision on the training PC |
| E > 0, first E on a reset row only | cause 4: a stale read at reset | read `_solve_start_pose` against the reset-step trap |
| E = 0 | the mean policy of It 200 does not take the path; the events need exploration noise. Nothing about causes 1-3 is shown | NOT a pass. Options: model_100 / model_300, or a stochastic replay (needs a code change, own decision) |
| E > 0, mixed AROUND and THROUGH | two paths | report both counts; no single cause |

One trace of one checkpoint. It shows a path, not a rate.
