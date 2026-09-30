# RT-187s20e256r — expectation, written BEFORE the run (2026-09-12)

Written on the dev laptop. `/rt-check` judges the log against THIS file.
**UNVERIFIED** until the training PC has run it. A MEASUREMENT
(`check_seated_success.py --reward-curve`), not a check and not a training run.

## Why

RT-187s20e256b (`09-12_17-06-42_offset5mm_currOFF_start+30mm_seed20`,
`model_1999.pt`, 256 envs, seed 20) was replayed as RT-187s20e256p
(`play.py --trace-obs`, 8 envs, git `ed3b3f8`). The trace was evaluated on the
laptop with `summarize_trace.py`, `trace_first_episode.py`,
`trace_outcome_split.py` and an inline read of `ee_quat` per step:

- first episode: success 0 / timeout 8 / censored 0 of 8; no done before step
  255, so no force abort (force max 16.5 N against 50 N);
- median over 8 envs: tilt 0.1 deg at step 0, 6.3 deg at step 8, 8.5 deg at
  step 20, 8.66 deg from step 50 to 254 (the controller cone is 8.52 deg);
- tip z 30.0 mm at step 0, 10.7 mm from step 8 on; `reached_below_entrance`
  0 of 8; `parked_in_contact` 8 of 8; lateral end median 7.43 mm (read from
  the NOISED `tip_rel` channel, sigma 2.5 mm per episode);
- lean direction the same in all 8 envs: tool z-axis xy component about
  (-0.07, -0.133), i.e. about 7.7 deg as a tilt about pocket x and about
  4.0 deg about pocket y;
- user, by eye in `play.py`: one lug ("Nase") of the part rests on the pocket
  rim, and the policy then stops.

This pose was seen before (RT-141, RT-158 mode A) on other checkpoints.
RT-170 measured the reward nearly blind to tilt at h -6 mm for 1-2 deg
(`VERDICTS.md`, RT-170 BEFUND 2). Tilt 7.7 deg at h +10 mm is NOT measured.

## The question

Is the rim-parked pose a place where the reward gives no reason to leave?
Concretely: at the SAME height, does the upright pose pay clearly more per
step than the tilted pose?

## Command (training PC, repo root)

```
.\scripts\rt_log.ps1 RT-187s20e256r python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-tilt-axis x --curve-tilt-deg 0 -7.7 7.7 --curve-heights-mm 30 15 10 5 0 -15 -34 --out rt_logs/RT-187s20e256r_metrics.json
```

Grid: 3 tilts x 7 heights = 21 points x 4 settle steps = 84 of 256 steps.
Tilt 0 first. `-7.7` is the policy's main lean component; `+7.7` is the other
lean direction as contrast (the underside is not symmetric).

## Named gaps

- The policy leans about BOTH pocket axes (7.7 deg x, 4.0 deg y). The ladder
  can tilt about ONE axis. The y component is not reproduced.
- The sign of `-7.7` is derived from `signed_tilt_angle` (`atan2(-a_y, a_z)`)
  on the trace's `ee_quat` against the reset baseline; the log's own
  `got` tilt column must confirm it. If `got` reads the opposite sign, the
  `+7.7` descent is the policy's pose instead.
- Lateral is 0 in every row. The policy's lateral offset is not reproduced.
- The D-080 counter-proof of the curve (`--curve-coarse-margin-mm 10`) is not
  re-run here.

## Expectations

**E0 (mechanical).** The run reaches the SUMMARY block. `exit 1` with
`all_poses_held False` is EXPECTED in contact rows (RT-164..RT-167 precedent)
and is not a crash. The tilt-0 descent ends by `success_termination` at
-34 mm or runs to the end.

**E1 (upright curve).** Tilt 0: `kernel_sum` rises monotonically from +30 mm
to -15 mm.

**E2 (the tilted descent stops).** At least one of -7.7 / +7.7 does not reach
-15 mm: it ends in `force_abort` or its achieved depth stays above the
entrance plane. Where it stops is read off `got_depth_mm`; the trace median
was +10.7 mm above.

**E3 (the discriminating point).** Per-step `reward` at h +10 mm, tilt 0
against the tilted row of the SAME height (the `got` tilt closest to the
policy's pose):

- **H1, reward blind:** difference < 1e-3 per step. Then the reward gives no
  pull from the parked pose back to upright; over the ~245 remaining steps
  that is < 0.25, below the episode's total time bill of 1.0. The parking is
  then a REWARD finding (no gradient out of the plateau).
- **H2, reward sees it:** difference >= 1e-3 per step. Then the reward does
  prefer upright, and the parking is NOT a reward problem first: suspects 3
  (action/controller: the policy drives to the 8.52 deg cone in 20 steps) and
  5 (exploration) of the differential diagnosis come before any reward change.

The 1e-3 threshold is our own choice, set here before the run: it is the
per-step size at which the difference over the remaining episode would reach
the same order as the time penalty. It is not derived from a source.

**E4 (order test, arithmetic after the run).** Parked return is bounded by
256 x (per-step reward at the parked row). Seated return is read from the
tilt-0 -34 mm row (`success_lump` + step reward). Expected: seated >> parked.
If NOT, the reward order is broken and that outranks E3.

Nothing in this file changes the reward. Any reward change is a separate
VORSCHLAG after the verdict.
