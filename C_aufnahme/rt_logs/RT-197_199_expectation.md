# RT-197 / RT-198 / RT-199 -- H_min ladder, rungs 20 / 10 / 5 mm, written BEFORE the runs (2026-09-13)

Branch `p5-kraftsensor`. **UNVERIFIED.** Same script, same corner, same two
criteria as RT-185 (`rt_logs/RT-185_expectation.md`, incl. its Nachtrag
2026-09-13 point 6: the fix `ae007356` is in this branch). Only the height
changes. RT-185a/b (30 mm) run FIRST; a rung below a failed rung is not run.

## Rule (D-179 (4))

H_min is the LOWEST rung that meets BOTH:

* (a) solver clean: `start_pose_solve_flags_dropped_envs` = 0 in the dump
  after step 1024 (4096 episodes, 1024 envs), reported as a share;
* (b) no contact force at reset: tared force at steps 0-2 about 0 N.
  THREE readings exist (RT-185 Nachtrag point 3: "clearly below 8.0834 N",
  "about 0 N", the `>1N` column of `zero_agent.py`). The verdict names all
  three; the user decides which one holds. Not decided here.

ENVS (user 2026-09-14): 256, not 1024. Criterion (a) needs 4096 episodes = 16 per env; the a-runs get `--max-steps 4160` (16*256 + 64 margin, RT-185's own pattern 4*1024 + 64). The dump fires at every common reset (interval 250 < 256). Applies to RT-185a too when it runs now. b-runs stay at 6 steps.

Corner: radius 30 mm, tilt 10 deg, yaw 5 deg, fixture noise forced to 0 by
the script (named loss: the fixture does not shift by up to 5 mm).

| RT | height | a: solver (4160 steps, 256 envs) | b: reset force (6 steps, --print-force) |
|---|---|---|---|
| RT-185 | 30 mm | RT-185a | RT-185b |
| RT-197 | 20 mm | RT-197a | RT-197b |
| RT-198 | 10 mm | RT-198a | RT-198b |
| RT-199 | 5 mm | RT-199a | RT-199b |

0 mm only if 5 mm is clean (D-179 (4)); it gets its own number then.

## P1 setup, every run

`git:` = the handed-over SHA; log carries `[zero_agent] probe angle set:`,
the three corner overrides, `start_tip_above_entrance` AND `_low` both at
the rung's height (`low = high`, zero-width band), obs noise and grasp offset
0. Otherwise RUN INVALID. Exit line may be missing (open since RT-192b).

## Pre-registered readings

| Outcome | Reading |
|---|---|
| 30, 20, 10, 5 all clean | H_min = 5 mm; run 0 mm next (own RT). |
| rung k clean, rung k-1 fails (a) | H_min = rung k; the failing rung's `flags_dropped` share goes into the verdict. |
| rung k clean, rung k-1 fails (b) only | H_min = rung k under the reading the user picks; the other two readings are named. |
| 30 mm fails | finding about the Phase-5 rebuild (RT-185 text), no H_min; stop the ladder. |

H_min then goes to its one home `insertion_env_cfg.start_tip_above_entrance`
(D-179 (3)), row A5 of `Belege_Streuwerte.md`, and the SBC floor rungs
(`Pläne/SBC_Schritt0_Entwurf.md` § 0.1a point 3) are computed from it.
