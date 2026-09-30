# RT-168 — expectation, written BEFORE the run (2026-09-06)

**RENUMBERED TWICE, 2026-09-06: RT-160, then RT-167, now RT-168.** RT-160 was issued twice: to the
resumed training run of 16:15:10 (judged in `VERDICTS.md`, cited by D-170)
and, an hour later and from the same stale ledger, to this probe. The
training run keeps the number because it is the one with a verdict line.
The probe's two CRASHED attempts stay recorded as RT-160 in `PROBLEMS.md`
and in the probe's own comments -- that is what their log headers say, and
rewriting them would break the trail to the actual logs. Only the run that
has NOT happened yet moves here.

**Question.** Is `osc_kp_rot` too soft to pull a jammed, tilted part upright?

The tilt clamp is 8.52° (`osc_tilt_clamp_rad`, CAD, Decision (3)). Against it
the RT-158 trace splits without overlap:

| group | tilt median | over the clamp |
|---|---|---|
| success (n=135) | 2.23° | **0 of 135** |
| mode A rim (n=71) | 10.22° | **71 of 71** |
| mode B mouth (n=50) | 15.32° | **46 of 50** |

Over the clamp the commanded target is more upright than the part, every
physics step, for the rest of the episode — and the part stays tilted. The
restoring moment `Lambda_rot · osc_kp_rot · theta_err` at `osc_kp_rot = 30`
is 0.203 N·m (mode A) and 0.812 N·m (mode B), with the measured
`Lambda_rot` max 0.22806 kg·m² the startup report prints. Nobody has checked
those two numbers against what they must overcome.

**Why not another RT-159.** RT-159 swapped the gain to 100 under a policy
trained at 30; the first-episode rate fell 0.5273 → 0.0039. That measures
the swap, not the gain. This probe removes the policy: it teleports to the
jam and holds a ZERO action, which under OSC *is* the standing "straighten
up" command (target = cone-clamped current pose).

**Command** (training PC, clone `Phase3_Implementierung_v3`, after pulling
`p5-osc-rot-gain` at the SHA handed over in chat). CORRECTED 2026-09-06: this
line said `a9a2ff5`, which is the probe's FIRST commit -- it predates both
crash fixes (`PROBLEMS.md`, argument parsing and the env import path). Running
that SHA reproduces the two crashes and measures nothing. The SHA to pull is
read back from `origin/p5-osc-rot-gain` and never copied from here.

    .\scripts\rt_log.ps1 RT-168 python -u scripts/tilt_recovery_probe.py --task Ur5e-Insertion-Direct-v0 --num_envs 16 --headless --mode a --kp-rot 30 --kp-rot 50 --kp-rot 100 --kp-rot 200 --out rt_logs/RT-168_metrics.json

16 envs: every env is teleported to the SAME pose, so the spread across envs
is solver and contact noise only. It is not a sample of anything, and the
report takes medians, not means.

## Points

1. **Mechanics.** exit 0, no traceback, the metrics file is written.
2. **The gain really changed per rung.** `osc_kp_rot` in each rung of the
   metrics file is 30 / 50 / 100 / 200. If all four read 30, the rebuild via
   `_build_task_space_controller` did not take and the run is worthless.
3. **The teleport landed.** `solve_converged` true and `teleport_tilt_deg`
   within 0.3° of 10.22 in every rung. A rung that did not converge is NOT a
   gain result and must be reported as such, never averaged in.
4. **No termination during the hold.** `terminated_during_hold` false. A
   termination resets that env inside `step()`, so anything read after it is
   the post-reset pose, not the jam (the reset-step trap).
5. **THE OPEN QUESTION — no number is predicted.** `tilt_after_hold_deg`
   and `tilt_min_during_hold_deg` per gain. Registering a value here would
   be inventing one. The three readings and what each means:
   * **the tilt falls to ≈ 8.52° at 30 as well** → the stiffness is NOT what
     holds the part. `osc_kp_rot` leaves the suspect list, and the work goes
     back to reward and start-height coverage.
   * **the tilt only falls above some gain** → the stiffness IS the jam. The
     next step is then a TRAINING run at that gain, never a replay, and the
     gain gets its own `/decision` (next free number D-170).
   * **the tilt falls at no gain, not even 200** → geometry or friction
     holds the part and no gain change can help. That would also retire the
     whole `osc_kp_rot` line.
6. **A control that must hold.** The reset pose reads `reset_tilt_deg` ≈ 0.
   If the arm is already tilted at reset, the pocket frame and the tool
   frame do not agree and every angle in this run is wrong.

## What the run does NOT decide

* **It does not recommend a gain.** Even a clean "200 rights it" is a
  reading at ONE pose, with ONE tilt axis, under a zero action. The derived
  band from the D-166 rule applied to rotation is 44–65, and that derivation
  rests on an INVENTED moment budget (20 N at half the part width) and on
  `osc_rot_step_limit_rad`, itself a Factory placeholder. Nothing here turns
  a gain into a measured number.
* **It says nothing about training.** No policy is in the loop.
* **It says nothing about mode B** unless `--mode b` is run as well.
* **The tilt axis is commanded, not measured.** `--tilt-axis x` tips the
  part along its LONG axis. Which axis the real failures tilt about is
  computable from the RT-158 trace and has not been done.

## Afterwards

`/rt-check` against these points, then `rt_logs/RT-168_metrics.json` to the
laptop. Numbers are read from that file, never from the console prose.
