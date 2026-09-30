# RT-194 / RT-195 — expectation, written BEFORE the runs (2026-09-13)

Branch `p5-kraftsensor`. **UNVERIFIED.** The pair that tests the user's
hope for the torque channel (chat, 2026-09-13): "a one-edge contact shows
in the sign of the moment, and the policy moves over". That can only show
where the force channel is ambiguous — under observation NOISE. RT-193
(noise 0) is the control that the channel does not hurt; this pair is the
test that it helps. Numbers RT-194 and RT-195 are free on both branches'
`VERDICTS.md`, `HANDOFF-RL.md` and `rt_logs/` (origin, 2026-09-13).

## The one question

**"With the cfg's own observation noise ON (pocket 0.0025 m, force 3.5 N,
grasp 0.003 m; torque 0, D-188), does the wrench-31 policy reach a higher
success rate than the force-28 policy — same seed, same start band, same
iterations?"**

## Design — two runs, one difference

|  | RT-194 | RT-195 |
|---|---|---|
| obs mode | `wrench` (31) | `force` (28, cfg default) |
| noise | cfg defaults (ON) | cfg defaults (ON) |
| start band | `start_tip_above_entrance_low=-0.030` (RT-191's curriculum start) | same |
| seed / envs / iterations | 20 / 256 / 600 | 20 / 256 / 600 |
| dr_mode | off (cfg default; a band is refused under autodr) | off |

600 iterations is OUR CHOICE, not a sourced number: RT-191 needed ~200
iterations to pass 0.90 at noise 0 (its log; read the exact one from the
RT-191 verdict on p5-robustheit), and noise is expected to slow learning by
an unknown factor. Twice RT-191's budget is a guess that fits the user's
absence (~5 h for RT-193 + this pair at 6-10 s/it). Both runs get the SAME
budget, so the comparison holds whatever the absolute value.

There is NO earlier run with noise ON and this start band: RT-189s1
(noise on, parked at 13 mm) started at +30 mm fixed under AutoDR. RT-195 is
therefore the baseline of RT-194, and nothing else is.

## Commands (training PC, PowerShell, clone `Phase5_Robustheit_v1`)

Run AFTER RT-193, strictly one at a time (RT-182 trap). The chain below does
that: PowerShell runs them in order.

```
.\scripts\rt_log.ps1 RT-194 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 20 --max_iterations 600 env.start_tip_above_entrance_low=-0.030 env.obs_wrench_mode=wrench
```

```
.\scripts\rt_log.ps1 RT-195 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 20 --max_iterations 600 env.start_tip_above_entrance_low=-0.030
```

Hand over per run: the short log, `demo_metrics.json` of the run folder,
and the last line of the full log (the `[rt_log] exit code:` line, which
RT-192b..e did not carry — open).

## Points

* **P1 setup, both.** exit 0, `git:` = the handed-over SHA, startup report:
  pocket sigma 0.0025, force sigma 3.5, grasp 0.003, start band low -0.030,
  `dr_mode: off`. RT-194: `obs mode wrench-31`, torque sigma 0.0, actor
  `in_features=31`. RT-195: `force-28`, `in_features=28`. Otherwise RUN
  INVALID.
* **P2 the six curves, both, fixed order** (Episode Reward, Policy Loss,
  Value Loss, Entropy, Explained Variance, KL), each with its trend, all
  finite, BEFORE any judgement.
* **P3 success, side by side.** `success_rate` at iteration 599, and the
  FIRST iteration at which each passes 0.50 and 0.90 (none = "never").
* **P4 depth.** `max_depth_max_mm` and `mean_max_depth_mm` at the end,
  both runs. A run that holds ~13 mm is the RT-189 parking pattern.
* **P5 the torque itself (RT-194).** `torque_norm_nm` p50/p95/max and
  `wrench_valid_frac` from `demo_metrics.json`. If p95 is of the order of
  the free-air 0.03-0.14 N m of RT-192c, the channel carried little in
  contact either — read P3 with that in mind.
* **P6 cost.** iteration time of both, side by side.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| RT-194 clearly above RT-195 at 599 AND earlier to 0.50 | the moment breaks the one-edge ambiguity under noise (the hope holds, one seed) | second seed for BOTH before any claim; then AutoDR with wrench |
| both alike (within what one seed can show — spread unknown) | the moment adds nothing the noised position channel does not already give at this band | the hope is not falsified, only unshown: the next lever is the noise-ON parking case at +30 mm fixed (RT-189's setup) with wrench |
| RT-194 clearly below | the extra channel hurts: normaliser inflates a near-zero input, or 31 inputs slow the same budget | read P5 first; then the reset-row difference (RT-193 expectation, point 2) |
| both parked at ~13 mm, success ~0 | the start band alone does not carry noise-ON learning in 600 it; the mode is not readable from this pair | the curriculum question goes back to p5-robustheit; this pair is void for the torque question |

One seed against one seed decides nothing on its own. Whatever falls out,
the next step is a second seed, not a conclusion.
