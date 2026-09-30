# RT-154 expectation — does the sigma tax fall away when its scale does?

Written 2026-09-05 on the dev laptop, BEFORE the run. `/rt-check` judges the
log against THIS file and nothing else. It is a direct A/B against RT-148b:
same task, same `--num_envs 128`, same `--seed 42`. **One change in the whole
repo touches the reward: `action_rate_scale` 0.1 -> 0.0034 (D-168).** The two
new `extras["log"]` curves are instruments and change no number the policy
sees.

## The one hypothesis

D-168: the action-rate term reads the SAMPLED action
(`insertion_env.py:870,883`), so its expected cost is a tax on the exploration
noise sigma and on nothing else:

```
a_t - a_(t-1) ~ N(0, 2 sigma^2 I_6)
E||.|| = sigma * sqrt(2) * E[chi_6],  E[chi_6] = 2.3501
per episode = scale * sigma * (1*2.3501 + 255*3.3235) = scale * sigma * 849.8
```

At the old 0.1 and start sigma 1.0 that is **85 per episode** against a total
kernel income of **57**. RT-148b measured the consequence: sigma 0.99 -> 0.01
before iteration 500, 97 % of the return gain from this one term
(`VERDICTS.md` lines 188, 195). The formula is not a guess — it already fits
RT-148b's own numbers: at the measured sigma 0.65 it predicts 55.2 and the log
read 57.32 (96 %).

At the new 0.0034 the same tax is **2.89 per episode at sigma 1.0**, 5.07 % of
57. That is the budget D-168 wrote down: S = 1.0 (`init_noise_std`), X = 5 %.

**What this run does NOT test:** whether the task can be solved. The kernel is
flat near the axis (RT-151c, RT-153), so it cannot be. The potential-based
rebuild is a separate hypothesis and is not in this run.

## Commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash in the handover message.

```
.\scripts\rt_log.ps1 RT-154 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 128 --headless --seed 42 --max_iterations 1000
```

`--max_iterations 1000`, not RT-148b's 4000: the falsifier fires at iteration
500 (that is where RT-148b's sigma was already 0.01), and 1000 gives it a
factor-two margin. RT-148b measured its OWN rate — 04:06:18 for 4000
iterations at these same 128 envs, i.e. 3.69 s/iteration — so 1000 is
about 62 minutes instead of four hours. Nothing in the hypothesis needs
the other 3000.

`--num_envs 128` is RT-148b's env count, checked against the log rather
than remembered: `VERDICTS.md:187` names "128envs", and the same entry's
8,192,000 steps / 32,000 episodes only come out at 128
(128 * 16 * 4000 = 8,192,000). It is also the cfg default
(`insertion_env_cfg.py:647`), which is why RT-148a's parameter table
lists it as "cfg default". The runs BEFORE RT-148a (RT-131 to RT-141)
all passed `--num_envs 1024` explicitly; that is a different batch size
and a different run family.

IT MUST NOT MOVE IN THIS RUN. The env count sets the batch
(num_envs * num_steps_per_env), the batch sets the gradient variance,
and the gradient variance is one of the things that decides how sigma
behaves. RT-148b is the ONLY run that measured the sigma collapse this
change is meant to fix, so its 128 is the baseline the whole expectation
compares against. Raising it to 1024 would change two things at once and
make P3 unreadable. Whether a larger batch keeps sigma alive on its own
is a REAL and separate question — and its own run.

Paste the log into `rt_logs/inbox.txt` after clearing it, and
`demo_metrics.json` with it. The FIRST block (iteration 0) and the LAST block
must both be in the paste — P3 needs the trend, not the endpoint.

## Points, each PASS/FAIL on its own

1. **P1 — the run starts and survives.** Exit 0, no traceback, no `nan`.
   1000/1000 iterations. `demo_metrics.json` present in the paste.
2. **P2 — the code is this code.** `[rt_log]` header shows the handover hash.
   `git rev-parse` printed the same hash before the run. Without this, every
   number below is unattributable.
3. **P3 — THE DISCRIMINATING MEASUREMENT: the sigma trajectory.**
   `Mean action noise std` in the console block, `Policy/mean_noise_std` on
   the TensorBoard curve. Baseline RT-148b: 0.99 at start, 0.65 at iteration
   49, **0.01 before iteration 500**.
   * **PASS: sigma > 0.30 at iteration 500.**
   * **FAIL: sigma <= 0.05 at iteration 500** — the term was not the driver.
   * Between 0.05 and 0.30 the run does not decide and must say so.
   The 0.30 line is MINE, an order-of-magnitude line, not a measured
   threshold, and the verdict must name it as such. Report sigma at
   iterations 0, 49, 100, 250, 500, 750, 1000 — the SHAPE decides, not one
   point.
4. **P4 — the action-rate account matches the prediction.**
   Per iteration, `Episode_Reward/action_rate` against
   `-0.0034 * 849.8 * sigma = -2.889 * sigma`. Report the pair at the same
   seven iterations. The measured value should be MORE negative than the
   prediction; the gap is the deterministic policy's own motion, which is
   what the term is supposed to price. RT-148b's gap at sigma 0.01 was 2.84
   (predicted 0.85, measured 3.69).
   * A gap that GROWS strongly while sigma is flat = command chatter, i.e.
     the second named risk of D-168. Cross-read `force_norm_n` p95 and
     `force_abort_rate` before calling it.
   * A measured value SMALLER in magnitude than the prediction means the
     formula or the scale is not what the run executed. Then P3 is void.
5. **P5 — the raw-action tripwires (new, D-168).** `action_abs_max_mean` and
   `action_sat_frac` on the TensorBoard curves. **This is their first
   execution; they have never run under Isaac.**
   * They must be PRESENT and non-`nan`. Absent = the log block was not
     rebuilt, and the D-168 risk is unmeasured for this run.
   * `action_sat_frac` climbing toward 1.0 while sigma looks healthy is the
     named failure: the mean has drifted outside [-1, 1], every draw
     saturates the clamp, exploration is dead by a second route. NO
     threshold is pre-registered — nothing has ever measured this quantity,
     so the first reading is a baseline, not a test. Report it and say so.
   * `action_abs_max_mean`: RT-148b's replay (RT-149) saw `|a|` up to 3.5
     under the OLD scale. A value far above that under the new scale is a
     finding.
6. **P6 — success stays 0. This is the PREDICTION, not the hope.**
   `success_rate` 0.0000 for the whole run, `mean_max_depth_mm` on the floor.
   RT-151c and RT-153 measured the near-centre lateral slope at 0.0007-0.0073
   of `g_v`, i.e. no usable lateral signal. Removing the sigma tax cannot
   create one. **A non-zero success rate here would falsify RT-151c/RT-153,
   not confirm D-168**, and would have to be chased before anything else is
   believed.
7. **P7 — the kernel is the identity anchor.** `Episode_Reward/kernels` in
   the 55-58 band, as in RT-148b (55.19 at iteration 49, 56.42 at 3999). This
   run changed nothing in the kernel. A value outside that band means the
   0.0034 edit touched something it had no business touching, and P3 to P6
   are void.
8. **P8 — the six curves, in order, each with a trend.** Episode Reward ->
   Policy Loss -> Value Loss -> Entropy -> Explained Variance -> KL. First
   block and last block both required.
   * Episode Reward: RT-148b went -3.40 -> +51.73, and 97 % of that was the
     rate term falling away. Under 0.0034 that source of gain is worth 2.89
     at most, so a comparable rise here CANNOT come from it. If Episode
     Reward rises far beyond ~57 - 2.9 = 54, name where it comes from before
     reading it as progress.
   * Entropy: 6-D Gaussian, `6*(0.5*ln(2*pi*e) + ln sigma)`. At sigma 1.0
     that is 8.51, at 0.30 it is +1.29, at 0.01 it is -19.12 (RT-148b read
     -19.65). It is a second, independent readout of P3 — the two must agree.
   * Value Loss / Explained Variance: rsl_rl has no value normalisation
     (D-116). A diverging value loss reopens that observation point and is
     reported, not fixed in this run.
   * KL: reported. `learning_rate` is adaptive; a rail-pinned KL is a
     finding about the LR schedule, not about D-168.

## What would make each point wrong

1. -> crash or flag typo. Fix, rerun, same RT number only if nothing ran.
2. -> no pull. Rerun, nothing else.
3. -> if sigma still collapses, **D-168 is falsified.** The action-rate term
   was not the driver. Next suspects, in order: `entropy_coef` (0.005,
   `rsl_rl_ppo_cfg.py:42`) and the value layer (D-116). NOT another reward
   weight.
4. -> read `insertion_env_cfg.py` `action_rate_scale` in the run's own config
   dump before blaming the formula.
5. -> a missing curve is a wiring failure, not a result. Check
   `extras["log"]` was rebuilt; the run's numbers stay usable for P3/P4 but
   the D-168 risk is then explicitly UNMEASURED in the verdict.
6. -> see point text. A success here is a bigger finding than the run.
7. -> revert and re-measure before anything else is read.
8. -> nothing; each curve is reported with its trend and judged on the
   CLAUDE.md red-flag list.

## Read, NOT expected

`force_norm_n` p50/p95/p99 and `force_abort_rate` (D-167 wants the
distribution; no prediction). `stage1_lat_y_max_mm`. `over_reach`.
`joint_target_lag`. `start_height_mean_mm` (should read a flat +30.0 at
rung 0). Seconds per iteration. `episodes_to_threshold` (must stay null,
see P6).

## What this run cannot answer

* **Nothing about whether the task is solvable.** The kernel is unchanged and
  flat. That is hypothesis 2 and its own run.
* Nothing about yaw — RT-152 is still open and the yaw axis is UNMEASURED.
* Nothing about the FORGE action EMA (`factory_env.py:213`). D-168 option (a)
  was rejected for this step, not refuted; if sigma survives here it stays a
  live alternative for the action layer.
* Nothing about `force_abort_f_max_n` = 300 N: under OSC the abort cannot
  fire (D-167), so this run measures the force distribution, not the guard.
* Nothing about a sigma FLOOR. rsl_rl has none, and none was added. If sigma
  drifts down slowly over 1000 iterations without hitting 0.01, that is a
  partial result and must be reported as one.
