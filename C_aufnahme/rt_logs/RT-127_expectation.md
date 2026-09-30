# RT-127 — expectation, written BEFORE the run

Written 2026-09-01 on the dev laptop, git `596da8c` ("tilt_insert: print
the step-1 tilt reference"), pushed to `origin/p3-rl-code`. `/rt-check`
judges the log against THIS file and nothing else.

## What the run is

A MEASUREMENT, not a fix. RT-124 (tilt fan) and RT-125 (straight
control) both died at `depth p50 0.000 mm`. RT-126 silenced the tilt
controller (`--tilt-gain 0 --tilt-tol-deg 200`) and the descent came
back to 19.07-21.12 mm. So the blocker sits in the tilt reference angle
itself. This run reads that angle out at step 1 and nothing else.

```
git pull
git rev-parse --short HEAD          # expected: 596da8c
.\scripts\rt_log.ps1 RT-127 python -u scripts/tilt_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 16 --headless --tilt-deg 0 --upright-depth-mm 0 --f-abort 300 --max-steps 900
```

`--tilt-deg 0` with `--upright-depth-mm 0` means the COMMANDED tilt is
zero at every depth. The tilt error is therefore the measured angle
itself, with nothing else mixed in.

## Points, each PASS/FAIL on its own

1. **P1 — the run starts and the step-1 block prints.** No traceback, no
   `nan`, and the three step-1 lines appear: `step 1 residuals`,
   `step 1 TILT REFERENCE`, `step 1 tilt gate`. A run that reaches the
   end without these lines is a FAIL of this point, whatever else it
   says.
2. **P2 — the code is this code.** The `[rt_log]` header prints
   `596da8c`. A different hash means the pull did not land (RT-116,
   RT-121 both failed this way) and the rest is not judged.
3. **P3 — the commanded tilt is zero.** `commanded p50` on the TILT
   REFERENCE line reads `+0.0000 deg`. If it does not, the arguments did
   not do what this file assumes and no reading is taken.
4. **P4 — signed and error agree.** With P3 held, `error p50` equals
   `signed p50` to the printed digits. A mismatch means `tilt_schedule`
   or the error term is not what the code reads to me, and that is the
   next thing to look at, not the angle.

## The reading, and the rule written BEFORE the number arrives

R1 is `signed p50` on the `step 1 TILT REFERENCE` line.

* **|R1| at or above 170 deg** — the axis-flip hypothesis is CONFIRMED:
  the part z-axis points down while the pocket z-axis points up, the
  0.2 deg gate can never open, and the controller drives the wrist into
  the block. Then, and only then, build the fix as the module already
  does it for yaw (`latch_yaw_target`,
  `scripted_policy.py:106`): latch the step-1 tilt as the reference and
  command tilt as a DIFFERENCE from it. Offline checks plus one
  mutation, per D-080.
* **|R1| at or below 0.2 deg** (inside the gate) — the hypothesis is
  REFUTED. The gate is open at step 1, so something else stops the
  descent. No rebuild. Next measurement to be chosen from the log.
* **Anything between** — the hypothesis is NOT confirmed as written.
  Report the number, change nothing, pick the next measurement.

R2 is the `step 1 tilt gate` count. Under the flip case it reads
`16 of 16` outside. Under the refuted case it reads `0 of 16`. This is a
cross-check on R1, not an independent point.

## What this run does NOT decide

The straight descent still stops at 19-21 mm on the 300 N abort with
zero clean insertions (RT-69, RT-126). That is a SEPARATE blocker in the
descent path, and `seat_probe` (RT-81/82) already showed the pocket
accepts the seat at 8 N. RT-127 says nothing about it. F_max stays
unmeasurable and `engaged_depth_m` stays open until one clean insertion
exists. The suction-cup datasheet value (upper F_max anchor) stays
`[Datenblatt pending]` and no run can close it.
