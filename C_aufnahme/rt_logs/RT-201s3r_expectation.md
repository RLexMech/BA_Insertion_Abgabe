# RT-201s3r -- replay of RT-201s3 model_200: which way does the part get under the fixture? Written BEFORE the run (2026-09-15)

Status: **UNVERIFIED.** A measurement, not a training run. Optional for the
restart, but it is the one measurement that turns the mechanism in
`docs/decisions_inbox.md` ("The pocket gate gets a floor") from a derivation
into a finding.

## Context

RT-201s3 booked depths up to 146.7 mm (pocket 36 mm) in 719 of 785
iterations; the hump of `mean_max_depth_mm` sits at iterations 150-250. The
CAD outline and the clamp box leave only the way THROUGH the 10 mm floor
(derived, not measured). `model_200.pt` is inside the hump.

## Code state

The handover SHA of `rt_logs/RT-206_expectation.md`. The replay runs on the
NEW code, which does not change the physics or the policy input; the gate
floor changes only what is booked. `env.force_abort_f_max_n=50.0` restores
the limit RT-201s3 trained under, so the 30 N abort does not end the pushes
that the replay exists to watch.

## The one hypothesis

**The part reaches the space under the fixture by passing through the pocket
floor: in the trace, the true tip passes z = -36 mm and z = -46 mm (pocket
frame, below the entrance) while it stays inside the opening, |x| < 45.3 mm
and |y| < 72.6 mm.**

Falsified by: an env whose true tip is below -46 mm and whose lateral
position left the opening (|x| >= 45.3 or |y| >= 72.6 mm) on the way down;
or no env below -46 mm at all in 512 steps (then the hypothesis is not
tested, and the reading is "not reproduced at 16 envs").

## Command

Training PC root `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`:

```powershell
.\scripts\rt_log.ps1 RT-201s3r python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 16 --headless --checkpoint C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-15_07-37-52__RT-201s3_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed3\model_200.pt --trace-obs rt_logs\RT-201s3r_trace.json --trace-steps 512 env.force_abort_f_max_n=50.0
```

Hand back: `rt_logs\RT-201s3r_trace.json` and the short log.

## Points

* **R1** `[rt_log] git:` = handover SHA; checkpoint path printed; startup
  report `force abort at 50.0 N`.
* **R2** the trace carries the `truth` block (`tip_true_m`,
  `interpen_max_m`, `force_ema_n`) for 16 envs x 512 steps.
* **R3** per env: minimum true tip z; for every env below -46 mm, the step
  it passes -36 and -46 mm, |x| and |y| at those steps, `interpen_max_m`
  in the steps before and after, and the force EMA norm there.
* **R4** `demo_metrics.json` of the replay, if written (>= 50 episodes):
  `below_fixture_rate`.

## What it cannot answer

* WHY PhysX lets the part through (SDF thin wall, offsets, solver) -- it
  shows the path, not the collider's internals.
* The tip in the trace is read once per policy step (15 Hz); a pass inside
  one step shows as a jump, not a path.
* Whether the play env matches RT-201s3's DR state at iteration 200 exactly:
  play uses the cfg defaults (start +20 mm, bounds at the centre, fixture
  noise from the cfg), which is the state RT-201s3 had at iteration 200
  (`dr/bounds_version` 5, floor done).
