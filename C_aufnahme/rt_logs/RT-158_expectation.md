# RT-158 — expectation, written BEFORE the run (2026-09-06)

**Question.** RT-157 found ONE parked failure pose, identical at +40 and
+60 mm: `tip_rel` x +7.79 / y −15.9 / z +4.36 mm, yaw error 6.8°, standing
force ~15.4 N, 331 cases, spread under 0.5 mm. The user reads it in the
viewer as the part hanging on the pocket rim by one corner. The trace could
not confirm that, because it carried yaw but no tilt.

This run repeats the +40 mm probe with `ee_quat` in the trace, so the tilt is
measured instead of read off a screenshot. It is the RT-149b line of
`Pläne/Plan-Merge.md`.

It is also the first run under D-169's 50 N abort, so it doubles as that
entry's verification.

**Command** (training PC, after pulling this branch):

    .\scripts\rt_log.ps1 RT-158 python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --checkpoint C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2\logs\rsl_rl\ur5e_insertion\09-05_19-45-03_offset5mm_currOFF_start-30..+30mm_seed42\model_250.pt --trace-obs rt_logs/RT-158_trace.json --trace-steps 288 env.start_tip_above_entrance=0.040 env.start_tip_above_entrance_low=0.040 env.osc_pos_clamp_m=0.07

256 envs, not 1024: `ee_quat` adds four floats per env per step, and the
1024-env RT-157h40t trace already weighed 90 MB. At 256 the file is ~30 MB
and still yields ~66 of the rim pose (264 per 1024 at this height).

## Points

1. **Mechanics.** exit 0, no traceback, `[trace] wrote 288 steps x 256 envs`.
2. **The trace carries the new channel.** `ee_quat` present, and
   `slices` contains `"ee_quat": [15, 19]`. Without both, points 5–7 are
   NOT BELEGT, not failed.
3. **Same condition as RT-157h40t.** Start pose `+40.000 mm above the
   stage-2 opening plane`, `osc_pos_clamp_m = 0.07`, `model_250.pt`.
4. **D-169 verification.** `force_abort_f_max_n` prints as 50.0 and
   `force_abort_rate` is 0.0. RT-157 measured the largest per-episode force
   maximum at this height as 43.88 N, so a 50 N limit must not fire. If it
   DOES fire, D-169 is wrong and must be re-opened — that is the one
   outcome that falsifies it.
5. **The result reproduces.** First-episode success rate (`trace_first_episode.py`)
   0.582 ± 0.06. RT-157h40t read 596/1024 = 0.5820; at 256 envs the binomial
   standard deviation is 3.1 %, so the band is two sigma. Outside it →
   the 50 N abort or the sampling changed the behaviour, and that is a
   finding, not a pass.
6. **The rim pose reproduces.** Among the timeouts, the cluster at lateral
   ≥ 12 mm ends at x +7.8 ± 0.5 / y −15.9 ± 0.7 mm.
7. **THE OPEN QUESTION — no prediction is registered.** `rot_beyond_yaw_end_deg`
   for that cluster: the rotation the yaw channel cannot account for. Both
   answers are informative and neither is a failure:
   * **near 0°** → the part stands FLAT beside the pocket, not tilted. The
     moment-channel idea (obs 28 → 31, planned for branch `p5-wrench-obs`)
     then targets the wrong defect and must be reconsidered before it is
     built.
   * **clearly above 0°** → the part is tilted, the user's viewer reading is
     confirmed as a measurement, and the moment channels have a measured
     motivation.

   Writing a number here would be inventing one. The threshold used to
   COUNT tilted cases is 2° (`trace_outcome_split.py`, `tilt_deg`), chosen
   as roughly the yaw noise the successful episodes show
   (`yaw_err_max_deg` p75 2.50°) — a floor, not a physical bound.

## What the run does NOT decide

* About which axis the part tilts. `quat_angle_deg` gives the magnitude of
  the rotation, not its axis, and which local axis of the EE frame runs
  along the insertion direction is not established anywhere in this repo.
* Anything about heights other than +40 mm.
* Anything about training. This is a replay of a frozen checkpoint under a
  deterministic inference policy (RT-154p NACHTRAG (a)).

## Afterwards

`/rt-check` against these points, then the trace file (~30 MB) to the laptop
and `python scripts/trace_outcome_split.py rt_logs/RT-158_trace.json`.
