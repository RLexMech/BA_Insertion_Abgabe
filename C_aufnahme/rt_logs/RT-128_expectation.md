# RT-128 — expectation, written BEFORE the run

Written 2026-09-01 on the dev laptop, after the tilt latch landed.
`/rt-check` judges the log against THIS file and nothing else.

## What changed and why

RT-127 measured the tilt reference at `signed p50 -180.0000 deg`, 16 of 16
envs outside the 0.2 deg gate. The tool axis is held ANTIPARALLEL to the
pocket axis, so a controller steering against the pocket axis carries a
permanent 180 deg error. The fix is the pattern the module already uses for
the yaw (`latch_yaw_target`): latch the step-1 attitude per env and command
the DIFFERENCE from it — `latch_tilt_reference` / `tilt_since_reference`.
The wrapped difference was lifted out of `yaw_error` into
`signed_angle_difference` so both callers share one home; the reference sits
exactly ON the branch cut, which is where an unwrapped subtraction is
guaranteed to be wrong.

Offline, ALREADY VERIFIED on the laptop:
`python scripts/check_scripted_insert.py --self-test` → 99 checks /
51 mutations, COUNTER-PROOF PASSED. `python scripts/selftest_checks.py
--offline` → `[selftest] offline: PASS`. The two existing yaw mutations now
flip the tilt checks too (2 and 4 declared), which is the shared home proving
itself.

## The run

```
git pull
git rev-parse --short HEAD
```

Expected HEAD: given in the handover message.

```
.\scripts\rt_log.ps1 RT-128 python -u scripts/tilt_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 17 --headless --tilt-sweep-deg -8 8 --upright-depth-mm 8 --f-abort 300 --max-steps 900
```

17 envs, NOT 16: an even count skips the straight 0 deg case, which RT-124
already lost once. With 17 the sweep is -8 .. +8 in 1 deg steps and env 8
carries the straight insertion.

## Points, each PASS/FAIL on its own

1. **P1 — the run starts and the step-1 block prints.** No traceback, no
   `nan`; `step 1 residuals`, `step 1 TILT REFERENCE`, `step 1 tilt gate`
   all present. The env code must COMPILE under real torch.jit first — an
   import-time error here is a FAIL of this point.
2. **P2 — the code is this code.** The `[rt_log]` header prints the git hash
   from the handover and the script line prints
   `marker: tilt_insert-2026-09-01b`.
3. **P3 — the latch does its job at step 1.** On the TILT REFERENCE line,
   `signed p50` still reads about -180 deg (the attitude did not change; only
   the reference did), and `since latched reference p50` reads
   `+0.0000 deg`. That pair IS the fix.
4. **P4 — the gate opens.** `step 1 tilt gate` reads a count WELL BELOW 17.
   Envs whose commanded start tilt is not yet reached may legitimately be
   held, so `17 of 17` is the failure and `0 of 17` is not required.
5. **P5 — the descent happens.** The per-env table reports `depth p50`
   greater than 0.000 mm for the majority of envs. RT-126 with the
   controller silenced reached 19.07-21.12 mm; anything in that class or
   better clears this point, anything at 0.000 mm does not.
6. **P6 — the tilt column is now readable.** The depth profile's `tilt max`
   column reports values in the single-degree class, not ~180. This is the
   relative tilt, per the new docstring.

## Readings taken OUT of the run — no threshold is set here

* R1: peak depth per start tilt. Does the tilted entry get deeper than the
  straight one? That is D-071's claim and it has never been measured.
* R2: peak force per start tilt, and the number of CLEAN episodes. F_max's
  lower bound needs at least one clean insertion; with zero clean episodes
  it stays unmeasurable.
* R3: the capture depth read-off, if any clean episode exists.

## What this run does NOT decide

The 19-21 mm jam on the STRAIGHT descent (RT-69, RT-126) is a separate
blocker and this change does not touch it. If P5 passes but every episode
still ends on the 300 N abort with 0 clean, the latch is fixed and the jam is
not — those are two results, not one. The suction-cup datasheet value stays
`[Datenblatt pending]` and no run can close it.
