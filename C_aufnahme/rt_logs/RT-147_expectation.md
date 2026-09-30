# RT-147 — expectation, written BEFORE the run (2026-09-04)

## What this run decides

One question, two incompatible answers:

**Is the OSC pose delta a VELOCITY command (per second) or a STEP size
(per control step)?**

Everything downstream hangs on it: the episode length, the kp choice, the
F_max reading, and whether the policy rate matters at all under
re-anchoring.

## The command

    .\scripts\rt_log.ps1 RT-147 python -u scripts/tilt_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 17 --headless --tilt-axis y --tilt-sweep-deg -8 8 --upright-depth-mm 8 --stop-depth-mm 35 --f-abort 300 --max-steps 3600 --control-mode osc --osc-kp-pos 100 --decimation 2 --episode-steps 1024 --out rt147_osc100_60hz_1024steps.json

## The comparison partner

RT-144b (2026-09-04 10:47:07, git `ebdd2a6`, `rt_logs/VERDICTS.md`):
osc, kp 100, decimation 8 (15 Hz), episode 256 steps = 17.067 s,
0 of 51 clean, depth p50 11-18 mm per tilt bin.

RT-147 holds EVERYTHING equal except the rate:

| | RT-144b | RT-147 |
|---|---|---|
| control mode | osc | osc |
| kp | 100 | 100 |
| descend rate | 0.5 mm / control step | 0.5 mm / control step |
| start pose | home pose (165 mm above) | home pose (165 mm above) |
| **seconds per episode** | **17.067** | **17.067** (1024 * dt * 2) |
| **policy rate** | **15 Hz** | **60 Hz** |

`--episode-steps 1024` is what holds the seconds equal; without it
decimation 2 gives 4.267 s and the run is RT-144d again (see the RT-146
NACHTRAG).

## The two outcomes

1. **Delta is a VELOCITY** (the model in HANDOFF-RL § Open NEXT 0):
   `v = Delta * sqrt(kp) / (2*zeta)`, per SECOND. The rate does not enter.
   -> depth p50 lands in RT-144b's 11-18 mm band, clean stays at 0.

2. **Delta is a STEP SIZE**: four times the control steps in the same
   seconds means four times the descent.
   -> depth p50 far above 18 mm, the 35 mm stop depth is reached, clean
   episodes appear.

These are far apart. There is no reading where both fit.

## Numbered expectation

1. exit 0, no traceback.
2. The startup report prints `control_mode osc`, `kp 100`, and a control
   rate of 60.0 Hz.
3. Depth p50 per tilt bin lands in or near RT-144b's 11-18 mm band
   (outcome 1), NOT above 30 mm (outcome 2).
4. Clean episodes stay at 0 of the run's episodes (follows from 3; the
   35 mm stop depth is not reached).
5. Episodes per env is about 3 (3600 control steps / 1024 per episode),
   so roughly 51 episodes over 17 envs -- the same count as RT-144b.

## What would make each point wrong

1. -> a crash; nothing to do with the hypothesis.
2. -> the flags did not reach the cfg; the run is void, not a finding.
3. -> the model is wrong. That is a REAL result, not a failure: it would
   mean the rate is a lever after all, and the RT-144 reading would have
   to be redone at equal seconds for every row.
4. -> follows 3; judge it together with 3, never alone.
5. -> `--episode-steps` did not take effect. Check the printed episode
   length before judging 3 or 4.

## Not decided by this run

- The kp comparison (RT-144b vs RT-144c) is a separate confound: kp 500
  travels 2.24x faster and may simply beat the clock. Named in
  HANDOFF-RL § Open NEXT 0; NOT answered here.
- The published-band question for F_max. RT-144c's 2.0209 N is close to
  `Lambda * kp * Delta` = 2.21 N (analysis, UNVERIFIED), which would make
  it a readout of our own gain, not a task force.
- The good policy rate as such. That is the literature question in
  `docs/reference/literature_check_policy_rate_osc_2026-09-04.md`.
