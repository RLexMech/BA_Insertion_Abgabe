# RT-196 -- expectation, written BEFORE the run (2026-09-13)

Branch `p5-kraftsensor`. **UNVERIFIED.** RT-194 with ONE change: the torque
channels get their own noise, sigma 0.2 N m per step (inbox entry "The
torque channels get the datasheet precision as their noise", 2026-09-13;
`Belege_Streuwerte.md` B4c). User decision after RT-194.

## The one question

**"Does the wrench-31 policy still reach RT-194's success (0.999 at 599,
0.90 first at 239) when the torque channels carry the datasheet noise?"**

The moment in contact is 0.28 N m p50 / 0.80 p95 (RT-194 demo_metrics);
the sigma is 0.2 N m. The noise is of the SAME order as the signal. If the
channel's value in RT-194 came from its cleanness, this run shows it.

## Design

RT-194's command plus `env.torque_obs_noise_std_nm=0.2`. Same seed 20, 256
envs, 600 iterations, start band -0.030, pocket 0.0025 m, force 3.5 N,
grasp 0.003 m (cfg defaults). Nothing else.

The torque noise is added where the force noise is added: AFTER the EMA
(the open point of B4b applies unchanged).

## Command (training PC, after RT-195, nothing else running)

```
.\scripts\rt_log.ps1 RT-196 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 20 --max_iterations 600 env.start_tip_above_entrance_low=-0.030 env.obs_wrench_mode=wrench env.torque_obs_noise_std_nm=0.2
```

Hand over: the run folder's `demo_metrics.json`, `scalars.csv`, `RT-196.txt`.

## Points

* **P1 setup.** exit 0 (or the log's last lines; the exit line is missing
  since RT-192b -- open), `git:` = handed-over SHA, `obs mode wrench-31`,
  startup report `torque_obs_noise_std_nm 0.2`, `in_features=31`, pocket
  0.0025, force 3.5, grasp 0.003, low -0.030, dr off. demo_metrics
  `torque_obs_noise_std_nm` 0.2. Otherwise RUN INVALID.
* **P2 the six curves**, fixed order, trends, finite (EV/KL not logged).
* **P3 success** at 599 and first >= 0.50 / 0.90, beside RT-194 (0.999 /
  it 0 / it 239) AND beside RT-195 (force-28, the no-torque baseline).
* **P4 depth** `max_depth_max_mm` / `mean_max_depth_mm`; any value > 37 =
  the under-fixture path (RT-193 it 200), FINDING.
* **P5** `torque_norm_nm` p50/p95 (the TRUE moment, logged before the
  noise) beside RT-194's 0.28 / 0.80 -- the signal did not change, only
  what the policy sees.
* **P6** iteration time beside RT-194 (6.7 -> 10.6 s).

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| like RT-194 (0.90 near it 239, ~0.999 at 599) | the policy does not depend on a clean moment; whatever the channel gives survives its noise | RT-195 decides whether it gives anything at all |
| slower than RT-194 but reaches ~0.99 | the noise costs learning time, not the solution | second seed before any claim |
| clearly below RT-194, near RT-195 | the RT-194 gain (if RT-195 shows one) came from the clean channel; under sensor noise the moment adds nothing | the hope is falsified at this sigma; the sigma reading itself stays an assumption |
| below RT-195 | a noised near-zero input hurts: the normaliser inflates it | read the free-air torque against the sigma; consider noise before the EMA (B4b open point) |

One seed against one seed decides nothing on its own.
