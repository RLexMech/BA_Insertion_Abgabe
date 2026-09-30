# RT-148b expectation — the first LONG PPO run under OSC

Written 2026-09-04 on the dev laptop, BEFORE the run, git `a609472`.
No code change against RT-148a. This is the same configuration, run long.

## Command

    .\scripts\rt_log.ps1 RT-148b python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 128 --headless --seed 42 --max_iterations 4000

`--max_iterations 4000` comes from RT-148a's ~3.5 s per iteration, i.e.
~3.9 h. That seconds-per-iteration figure is a 50-iteration sample and may
drift.

## Why no code change first

RT-148a ran 50 iterations, about 3 minutes. Entropy moved 8.48 -> 8.24.
The policy is still essentially random. Every live observation the user
made while watching the viewer on 2026-09-04 (yaw drifting further and
further, the part swinging laterally above the fixture) is what a random
action sequence MUST look like under a controller whose action is a
velocity (RT-147). Judging the controller, the reward weights or the yaw
freedom on that state would be judging noise. This run buys the state on
which such a judgement is possible.

## Numbered expectation

1. exit 0, no traceback, `demo_metrics.json` written with `control_mode`
   `osc` and `policy_rate_hz` 15.0.
2. The startup report is identical to RT-148a's: kp pos 100.0, kp rot 30.0,
   drives inert, episode 256 steps = 17.0667 s, `curriculum_enabled False`,
   rung-0 start +30 mm.
3. **`force_norm_n` p95 stays BELOW 20 N over the whole run.** RT-148a
   measured p95 18.387 N at p99 22.4 N / max 24.2 N over 50 iterations.
   D-166's bound is read back here on a policy that has actually moved.
4. **`force_abort_rate` = 0.0, `over_f_max` 0** (D-167).
5. **Entropy falls clearly below 8.24.** If it is still ~8.4 after 4000
   iterations, nothing is learning and every other number in this run is
   meaningless.
6. **THE LOAD-BEARING POINT: `over_reach` falls below RT-148a's 154 of
   250.** The lateral gate `in_pocket_cross_section` must open at least
   sometimes. While it stays shut, `progress`, `engaged` and `success`
   cannot pay at all, and the only paying rows are `kernels` against
   `time`, `action_rate`, `contact` and `far`.
7. `mean_max_depth_mm` rises above RT-148a's 0.0124 mm. This is point 6
   restated in metres; it is listed separately because the gated depth is
   the number the thesis reports.

## Read, NOT expected

`success_rate_recent`; the `action_rate` share of the reward account
(-55.660 per episode in RT-148a, larger than the whole positive shaping);
seconds per iteration.

## NOT measurable in this run

**The yaw drift the user saw has no metric.** No per-episode part attitude
is logged during training (`HANDOFF-RL.md` § Live gaps). This run can
neither confirm nor refute it. The instrument is a `play.py` replay of the
resulting checkpoint, or a new logged channel — neither exists yet.

## What would make each point wrong

1. -> a crash. Nothing about the hypothesis; fix and rerun.
2. -> the run is not the configuration this file names. Void it.
3. -> D-166's `F_search = 20 N` derivation of `osc_kp_pos` does not hold
   under a moving policy. Reopen D-166, not the run.
4. -> a 300 N event under a 100 N/m gain would mean the force reading, not
   the controller, is wrong.
5. -> exploration died or never started. Check `entropy_coef` and the
   reward scale before anything else.
6. and 7. -> the reward has NO PATH from the start pose to the pocket. That
   is then the finding, and the next question is the lateral gradient of
   `kernels` above the entrance plane, not the hyperparameters.
