# RT-153 expectation — the x slice, both signs, out to the OSC clamp

Written 2026-09-05 on the dev laptop, BEFORE the run. It is the x twin of
RT-151c, which swept y only. `/rt-check` judges the two logs against THIS file
and nothing else.

## Why x at all

RT-149 replayed `model_3999.pt` over 8 envs and measured x roaming both signs
(`rt_logs/VERDICTS.md` line 205 (b) and (f)): env 3 walks x -49 -> +33 mm,
envs 0/6/7 swing x -35 .. +39 mm, and **5 of 8 envs reach |x| 48 to 49.9 mm**
-- the `osc_pos_clamp_m` = 50 mm clamp, and past `POCKET_WALL_X` = 45.2938 mm
(`insertion_tasks_cfg.py:694`), where the D-157 gate `in_pocket_cross_section`
closes on x.

The reward curve has never swept x. Until this change x was pinned to zero
inside the solve (`d_pocket[:, 0] = -tip_rel[:, 0]`) and no flag could move
it. So the landscape the policy is standing in along x is unmeasured.

## What the code does that it did not do before

Marker `check_seated_success-2026-09-05c`. Four changes, all offline-tested
(`--self-test` 76/76, the x block is 14 of those lines, and the by-hand
counter-proof in `scripts/` reddened exactly its own 5 cases and nothing else).
**None of the four has run under Isaac. This run is their first execution.**

1. New flag `--curve-lateral-x-mm`, default `[0.0]`. The default reproduces
   the pre-2026-09-05c curve row for row.
2. `curve_grid` returns `(h, y, x, tilt)`. x is the INNERMOST loop; the tilt
   stays outermost, unchanged.
3. `_solve_to` takes `goal_lateral_x`, default `0.0`. The default is exactly
   the old hard-wired zero, so the seated identity test and both counter-proofs
   are untouched.
4. `curve_lateral_error` applies BOTH existing rules to x as well: no non-zero
   x below the entrance plane (the stage-2 walls hold the part across the
   short axis to `PLAY_X` / 2 = 0.2938 mm), and no |x| past the OSC clamp
   half-width (the box is componentwise). No new rule was invented; only the
   axis is new.

The per-row `curve` line, the JSON row and the `SUMMARY row` line all carry
`x` and `got_x_mm`, and `pose_drift_mm` now takes the max over x as well.

## The grid

Heights **30.0 and 16.0 mm** above the entrance plane -- the two heights
RT-151c measured, so the `x 0` rows are a direct identity anchor against a
run that already passed. Both are above the block top face
(`STAGE1_DEPTH` = 15 mm, `insertion_tasks_cfg.py:674`), so every x is free
air at every height. h +45 is dropped: `g_v` is defined between +16 and +30,
and the episode budget is spent on x instead.

x offsets, **21 values, both signs**:

```
-49.9 -48 -45 -35 -29.79 -25 -15 -5 -2 -1 0 1 2 5 10.29 15 25 35 45 48 49.9
```

All |x| <= 50 mm, so none is refused and none can be clamped. The landmarks
are geometry, not taste:

* **+-45.2938 mm** = `POCKET_WALL_X`, the D-157 gate. 45 is inside it,
  48 and 49.9 are outside. This is the only crossing in the grid.
* **+10.2938 mm / -29.7938 mm** = the stage-1 free travel in x, MY OWN
  arithmetic from `POCKET_STAGE1_X_RANGE` (`insertion_tasks_cfg.py:703`)
  minus `PART_BODY_X` / 2 (`:1362`). The file states the y twin (8.7 mm,
  `:1521`) but never states these two, so they are derived here and named as
  derived. They are NOT load-bearing for any point below; they are sample
  positions.
* x is ASYMMETRIC in this fixture (`POCKET_STAGE1_X_RANGE` and
  `POCKET_LOCAL_X_RANGE` are both off-centre), which is why both signs are
  swept. y was swept on one sign only because y is not.
* 1 and 2 mm mirror RT-151c: the near-centre interval is where the slope
  decides, and it is the interval nobody has measured.

2 x 21 = 42 points x 4 settle steps = 168 of the 256 step episode.

## Commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash in the handover message.

```
.\scripts\rt_log.ps1 RT-153 python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 30.0 16.0 --curve-lateral-x-mm -49.9 -48 -45 -35 -29.79 -25 -15 -5 -2 -1 0 1 2 5 10.29 15 25 35 45 48 49.9 --out rt_logs/RT-153_curve.json
```

```
.\scripts\rt_log.ps1 RT-153-cp python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 30.0 16.0 --curve-lateral-x-mm -49.9 -48 -45 -35 -29.79 -25 -15 -5 -2 -1 0 1 2 5 10.29 15 25 35 45 48 49.9 --curve-coarse-margin-mm 10 --out rt_logs/RT-153_curve_counterproof.json
```

Paste the SHORT log of each run into `rt_logs/inbox.txt`, after clearing what
is in there now. Two pastes, nothing else.

## The discard rule — before any point is read

A row with `held False`, or `interpen` above `INTERPEN_THRESH` = 0.2938 mm
(`insertion_tasks_cfg.py:1568`, half `PLAY_X`), is DISCARDED, not interpreted.
A run whose `sweep_end_reason` is `force_abort` is unusable from that point on.

## Points, each PASS/FAIL on its own

1. **P1 — both legs start and survive.** Exit 0, no traceback, no `nan`.
   42 `curve  h` lines per leg, heights outer and x inner, in the commanded
   order. `reset_during_sweep` false, `sweep_end_reason` null, `verdict_ok`
   true, last line `VERDICT: REWARD_CURVE_MEASURED`. SUMMARY block present
   with one `SUMMARY grid`, 42 `SUMMARY row` and one `SUMMARY end` line.
2. **P2 — the code is this code.** `[rt_log]` header shows the handover hash;
   the log prints `marker: check_seated_success-2026-09-05c` (the `c` is the
   point -- RT-151c ran `-2026-09-05b`); `SUMMARY grid` shows
   `laterals_x_mm` equal to the commanded list, `laterals_mm [0.0]`,
   `tilts_deg [0.0]`, `kernel_margin_coarse_mm` 173.21; the cp leg prints
   `COUNTER-PROOF -- coarse margin forced to 10.0 mm, a_coarse = 299.32 /m`.
3. **P3 — the pose gate, now on three axes.** Every row: `held True`,
   `drift` <= 0.050 mm, `got_x_mm` within 0.05 mm of the commanded x on all
   four envs, `got_y_mm` within 0.05 mm of 0, `got_depth_mm` within 0.05 mm
   of -h. `SUMMARY end` shows `all_poses_held True`. This is the first time
   the drift arithmetic reads a NON-ZERO x, and the first time a commanded x
   passes through `_solve_to` at all.
4. **P4 — no impossible pose.** `interpen_max_mm` 0.000 on every row (free
   air: the lowest height is 1 mm above the block face) and `force_max_n`
   < 1 N (RT-151c read <= 0.64 N over comparable free-air poses).
5. **P5 — instrument identity against RT-151c.** Same code path, same pinned
   pose, `fixture_pos_noise_xy` 0. The `x 0` rows must reproduce RT-151c's
   `y 0` rows to 1e-4:
   (i) h +30 / x 0 `kernel_sum` = 2.235047e-01
   (ii) h +16 / x 0 `kernel_sum` = 2.380458e-01
   (iii) h +16 > h +30 at x 0.
   A miss here means the 2026-09-05c edit moved a number it had no business
   moving -- most likely the new `goal_lateral_x` default is not 0 in the path
   that runs -- and P6 to P9 are void.
6. **P6 — rows that MUST read zero, and the ONE gate that must flip.**
   On every row: `engaged` False, `success` False, `max_depth_mm` 0.000;
   terms `engaged`, `success`, `success_lump`, `abort`, `action_rate` = 0;
   `progress` = 0 by pinning (`env_cfg.w_depth_progress = 0.0`, D-165);
   `time` = -0.00391. So `reward` = `kernels` - 0.00391 on every row.
   **`in_pocket` is the exception and it is a prediction:** True on every row
   with |x| <= 45 mm, and **False on the four rows |x| 48 and 49.9**, because
   `in_pocket_cross_section` tests `|x| < POCKET_WALL_X` = 45.2938 mm
   (`insertion_math.py`, the D-157 gate). If those four read True the gate is
   not reading x and RT-149's finding (f) is wrong.
7. **P7 — the shape.** `kernel_sum` NON-INCREASING as |x| grows, at both
   heights, on BOTH sides of zero, and the maximum of each height's row is at
   x 0. Everything else here is unpredicted and every outcome is a finding:
   * whether the two sides are SYMMETRIC. The geometry is not (see the grid),
     so an asymmetry is expected but its size is not predicted. Report the
     pair difference `|ks(+x) - ks(-x)|` at 5, 15, 25, 35, 45 mm.
   * whether anything changes at the +-45.2938 mm wall crossing. The kernel
     reads the SDF and knows nothing about the D-157 gate, so a step there
     would be a finding about the SDF, not about the gate.
   * a value that RISES with |x| anywhere is the strongest possible finding.
8. **P8 — the near-centre slope, same rule as RT-151c.** Per height, from
   consecutive rows: `slope(x_a -> x_b) = [ks(x_a) - ks(x_b)] / (x_b - x_a)`,
   as a fraction of `g_v = [ks(h +16, x 0) - ks(h +30, x 0)] / 14 mm`
   (RT-151c measured g_v = 1.0387e-03 /mm; recompute it from THIS run's own
   two anchor rows, do not carry the old number in).
   THE DECIDING NUMBERS, named now: `x 0 -> 1`, `x 0 -> -1`, `x 0 -> 5` and
   `x 0 -> -5` as a fraction of g_v, at both heights -- eight values.
   Rule, the SAME self-set lines RT-151c used so the two axes compare:
   below 0.01 on all eight -> **the reward carries no usable lateral signal
   near the centre in x either**; at or above 0.05 -> the near-centre x
   signal exists and exploration is the suspect, not the reward; between
   0.01 and 0.05 the run does not decide and says so. The thresholds are
   MINE, order-of-magnitude lines, and the verdict must name them as such.
   PRE-REGISTERED EXPECTATION: below 0.01, i.e. the same answer y gave
   (RT-151c measured 0.0007 to 0.0073 over six such values). If x comes out
   at or above 0.05 while y did not, the kernel is anisotropic and that is a
   bigger finding than either single result.
9. **P9 — the paid counter-proof (D-080), per leg.** With the coarse margin
   forced to 10 mm, EVERY row's `kernel_sum` on the cp leg is at least 10x
   smaller than the same row on the main leg (RT-151c collapsed h +16 x 0 to
   4.52e-04 and h +30 x 0 to 8.26e-06). The sign in P7 must survive the
   mutation wherever the cp value is still resolvable (> 1e-4); below that
   the column is noise and is not read. If the column does not collapse, the
   curve is not reading the kernel it claims to read and P7/P8 are void.

## What would make each point wrong

1. -> a crash or a flag typo. Fix, rerun. A grid REFUSAL here would be a bug
   in the new x guard, not in the grid: every commanded |x| is <= 49.9 mm and
   both heights are positive.
2. -> no pull, or the old marker `-2026-09-05b`: rerun, nothing else.
3. -> `held False` on an x row while RT-151c held every y row means the OSC
   clamp is not the only thing moving a pose, or the x column of the IK
   command does not do what the y column does. Name it and stop; do not read
   a reward off a drifting pose.
4. -> `interpen` > 0.2938 mm in free air: the block's top face is not at
   +15 mm in the USD as the cfg says. A scene finding; name it and stop.
5. -> the `goal_lateral_x` default is leaking into the x 0 rows, or the edit
   changed the solve. Revert and re-measure before anything else is read.
6. -> any nonzero term: read the cited code line before reading the number.
   `in_pocket` True at |x| 49.9 means the D-157 gate does not read x.
7. -> nothing; every outcome is a finding. A mixed sign (falls, then rises)
   needs the four envs checked against each other in the `SUMMARY row`; if
   they disagree the SDF query is noisy at that pose and the row is UNKLAR.
8. -> if `g_v` from this run's own anchors is ~0, the fraction has no
   denominator and that is a stronger finding than either branch. Name it.
9. -> see point text.

## Read, NOT expected

`sdf` at every x -- no prediction. `force` per row in detail. Seconds per
point. The per-env spread of `kernel_sum` (should be identical to ~1e-6).

## What this run cannot answer

* Nothing about what the POLICY does with the landscape. 42 poses, no policy.
* Nothing about yaw (RT-149 held +25 deg) -- that is RT-152.
* Nothing about x and y TOGETHER: y is pinned to 0 here, so a diagonal
  offset is unmeasured. RT-149's envs held both at once.
* Nothing below the entrance plane with an x offset (refused on purpose).
* Nothing about tilt: the teleport commands the part upright.
* **Nothing above +-50 mm in any axis.** The guard refuses it. RT-149's envs
  sat AT that clamp, so the landscape just past it stays unmeasured with this
  instrument.
* Nothing at h +45: dropped from the grid to pay for the x sweep. RT-151c
  measured it in y.
