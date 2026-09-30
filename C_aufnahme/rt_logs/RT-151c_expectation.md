# RT-151c expectation — the lateral slice, re-run inside the clamp box

Written 2026-09-05 on the dev laptop, BEFORE the run. It replaces RT-151b's
two outermost rows, which were never at the commanded pose, and it replaces
RT-151b's P8 criterion, which the correction showed to be non-discriminating.
`/rt-check` judges the two logs against THIS file and nothing else.

## Why RT-151b needs a re-run at all

RT-151b commanded y 51.3 and 60.0 mm and measured 50.320 and 52.454 mm, with
`solve_converged True` and `residual 0.000 mm` on both. The residual is taken
inside `_solve_to` right after the last IK write, BEFORE any physics; the four
settle steps then run `_apply_osc` -> `clamp_tip_in_box` with
`osc_pos_clamp_m` = 0.05 m (`insertion_env_cfg.py:240`, `insertion_env.py:1007`),
which pulls a tip commanded outside the +-50 mm box back toward the entrance.
Rows at y <= 45 mm sit inside the box and landed exactly; only the two outside
it drifted. `PROBLEMS.md`, row 2026-09-05.

Two consequences, both already on record in `rt_logs/VERDICTS.md`:

* **RT-151b P8 is withdrawn.** Its "gradient exists, g_l/g_v = 0.1153" read the
  y 51.3 row. On the valid rows alone (y <= 45) the same arithmetic gives
  g_l = 1.03716e-04 /mm, g_v = 1.03862e-03 /mm, ratio **0.0999** against the
  self-set line 0.100 -- the verdict flips on the contaminated point, by 0.1 %.
  A criterion that a single bad row can turn over decides nothing.
* **The height axis has the same problem.** The box is componentwise, so
  RT-121/RT-123's sweeps from +165 mm were pulled DOWN while they settled. The
  "+165 mm reads 0.0455" anchor is not a held pose. NOT re-measured here: the
  script now refuses heights outside the box, so this tool cannot measure it.

RT-151b's rows at y <= 45 and h 30/16 are NOT discarded. They are the
prediction this run must reproduce (P5, P7).

## What the code does that it did not do before

Marker `check_seated_success-2026-09-05b`. Three changes, all offline-tested
(`--self-test` 49/49 plus 11 new guard lines, `selftest_checks.py --offline`
PASS, both D-080 mutations proven to redden exactly their own cases). **None
of the three has run under Isaac. This run is their first execution.**

1. `SUMMARY row` carries `got_y_mm` and `got_depth_mm` for all envs, plus
   `drift` and `held`. RT-151b's SUMMARY carried `residual`/`conv` only, which
   is why the miss survived the short log.
2. New `pose_drift_mm` per row = max over envs of `|got_y - commanded_y|` and
   `|got_depth + commanded_h|`, read AFTER the settle. Above `--solve-tol-mm`
   the row sets `pose_held` False, prints `POSE NOT HELD`, and
   `all_poses_held` joins `verdict_ok`.
3. `curve_lateral_error` refuses heights AND laterals outside
   `osc_pos_clamp_m`, called a second time in `main` once `env_cfg` exists so
   this run's configured value is read, still before `gym.make`.

## The grid

Heights **45.0 / 30.0 / 16.0 mm** above the entrance plane. All three inside
the +-50 mm box. Together they cover RT-149's observed band (the policy roamed
z 12-47 mm), and they give the vertical reference at three points instead of
two. 16.0 mm is 1 mm above the block's top face
(`STAGE1_DEPTH` = 15 mm, `insertion_tasks_cfg.py:674`), so every lateral is
free air at every height.

Laterals **0 1 2 5 8.7 15 25 35 45 mm**. All <= 50 mm, so none is refused and
none can be clamped. 1 and 2 mm are new: RT-151b's coarsest information is
exactly where it matters most, between 0 and 5 mm, and that interval carried
the flattest slope it measured. 8.7 mm is the stage-1 free travel; 45 mm is
RT-151b's outermost row that actually held its pose.

3 x 9 = 27 points x 4 settle steps = 108 of the 256 step episode.

## Commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash in the handover message.

```
.\scripts\rt_log.ps1 RT-151c python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 45.0 30.0 16.0 --curve-lateral-y-mm 0 1 2 5 8.7 15 25 35 45 --out rt_logs/RT-151c_curve.json
.\scripts\rt_log.ps1 RT-151c-cp python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 45.0 30.0 16.0 --curve-lateral-y-mm 0 1 2 5 8.7 15 25 35 45 --curve-coarse-margin-mm 10 --out rt_logs/RT-151c_curve_counterproof.json
```

Paste the SHORT log of each run into `rt_logs/inbox.txt`, after deleting the
RT-148b block that is in there now. Two pastes, nothing else.

## The discard rule — before any point is read

A row with `held False`, or `interpen` above `INTERPEN_THRESH` = 0.2938 mm
(`insertion_tasks_cfg.py:1568`, half `PLAY_X`), is DISCARDED, not interpreted.
A run whose `sweep_end_reason` is `force_abort` is unusable from that point on.
New this time: `held` is printed, so the discard rule no longer needs the JSON.

## Points, each PASS/FAIL on its own

1. **P1 — both legs start and survive.** Exit 0, no traceback, no `nan`.
   27 `curve  h` lines per leg, heights outer and laterals inner, in the
   commanded order. Header `3 heights x 9 laterals = 27 points x 4 settle
   steps = 108 of the 256 step episode`. `reset_during_sweep` false,
   `sweep_end_reason` null, `verdict_ok` true, last line
   `VERDICT: REWARD_CURVE_MEASURED`. SUMMARY block present with one
   `SUMMARY grid`, 27 `SUMMARY row` and one `SUMMARY end` line.
2. **P2 — the code is this code.** `[rt_log]` header shows the handover hash;
   the log prints `marker: check_seated_success-2026-09-05b` (the `b` is the
   point -- RT-151b ran `-2026-09-05`); `curve_heights_mm` /
   `curve_lateral_y_mm` equal the commanded lists; `curve_tilt_deg` = `[0.0]`;
   `kernel_margin_coarse_mm` = 173.21; the cp leg prints `COUNTER-PROOF --
   coarse margin forced to 10.0 mm, a_coarse = 299.32 /m`.
3. **P3 — THE NEW GATE, and it is read from the SUMMARY block alone.** Every
   row: `held True`, `drift` <= 0.050 mm, `got_y_mm` within 0.05 mm of the
   commanded y on all four envs, `got_depth_mm` within 0.05 mm of -h.
   `SUMMARY end` shows `all_poses_held True`. This is the point RT-151b could
   not state and its SUMMARY could not carry.
4. **P4 — no impossible pose.** `interpen_max_mm` 0.000 on every row (free air
   at every height, the lowest is 1 mm above the block face) and
   `force_max_n` < 1 N (RT-151b read <= 0.64 N over the same free-air poses).
5. **P5 — instrument identity against RT-151b's VALID rows.** Same code path,
   same pinned pose, `fixture_pos_noise_xy` 0.
   (i) h +30 / y 0 `kernel_sum` = 2.235053e-01 within 1e-4, and within 1e-3 of
   RT-123's 0.2235.
   (ii) h +16 / y 0 = 2.380460e-01 within 1e-4.
   (iii) the y 0 rows are monotone in height: h +16 > h +30 > h +45.
   A miss on (i) or (ii) means the 2026-09-05b edit changed a number it had no
   business changing, and P7/P8 are void.
6. **P6 — rows that MUST read zero.** On every row: `engaged` False,
   `success` False, `max_depth_mm` 0.000, `in_pocket` True (x is 0 and every
   |y| <= 45 mm is inside the D-157 box, +-72.55 mm); terms `engaged`,
   `success`, `success_lump`, `abort`, `action_rate` = 0; `progress` = 0 by
   pinning (`env_cfg.w_depth_progress = 0.0`, D-165); `time` = -0.00391. So
   `reward` = `kernels` - 0.00391 on every row.
7. **P7 — the shape, and the rows RT-151b already measured.** `kernel_sum`
   NON-INCREASING in y at all three heights. At h +30 and h +16 the seven
   shared laterals must REPRODUCE RT-151b to 1e-4:

   | y mm | h +30 | h +16 |
   |---|---|---|
   | 0 | 2.235053e-01 | 2.380460e-01 |
   | 5 | 2.234770e-01 | 2.380078e-01 |
   | 8.7 | 2.234111e-01 | 2.379233e-01 |
   | 15 | 2.231749e-01 | 2.376441e-01 |
   | 25 | 2.223929e-01 | 2.368049e-01 |
   | 35 | 2.210069e-01 | 2.354065e-01 |
   | 45 | 2.189354e-01 | 2.333788e-01 |

   New and unpredicted: the whole h +45 row, and y 1 and 2 at every height.
   A value that RISES with y anywhere is the strongest possible finding.
8. **P8 — THE LOCAL SLOPE PROFILE, replacing the single ratio.** The 0.1 x g_v
   rule from RT-151b is declared NON-DISCRIMINATING here and is NOT the
   decider: it returns 0.0999 on the valid rows and 0.1153 on the contaminated
   ones, on either side of its own line. Recording it is fine; deciding on it
   is not.

   What is read instead, per height, from consecutive rows:
   `slope(y_a -> y_b) = [ks(y_a) - ks(y_b)] / (y_b - y_a)`, and the same as a
   fraction of `g_v = [ks(h +16, y 0) - ks(h +30, y 0)] / 14 mm`.

   PRE-REGISTERED PREDICTION, from RT-151b's valid h +16 rows:

   | y mm | slope /mm | as x g_v |
   |---|---|---|
   | 0 -> 5 | 7.640e-06 | 0.0074 |
   | 5 -> 8.7 | 2.284e-05 | 0.0220 |
   | 8.7 -> 15 | 4.432e-05 | 0.0427 |
   | 15 -> 25 | 8.392e-05 | 0.0808 |
   | 25 -> 35 | 1.398e-04 | 0.1346 |
   | 35 -> 45 | 2.028e-04 | 0.1952 |

   The profile rises by a factor 27 across the range, so no single average
   describes it. THE DECIDING NUMBER, named now: the near-centre slope
   `y 0 -> 1` and `y 0 -> 5` as a fraction of g_v, at every height. Rule:
   below 0.01 at all three heights -> **the reward carries no usable lateral
   signal near the centre**, and a policy starting centred feels essentially
   nothing pulling it back. At or above 0.05 -> the near-centre signal exists
   and exploration is the suspect, not the reward. Between 0.01 and 0.05 the
   run does not decide and says so. The three thresholds are MINE, not from
   any source; they are order-of-magnitude lines and the verdict names them
   as such. y 0 -> 1 has never been measured at any height.
9. **P9 — the paid counter-proof (D-080), per leg.** With the coarse margin
   forced to 10 mm, EVERY row's `kernel_sum` on the cp leg is at least 10x
   smaller than the same row on the main leg (RT-151b measured 526x at h +16
   y 0 and 27038x at h +30 y 0). The sign in P7 must survive the mutation
   wherever the cp value is still resolvable (> 1e-4); below that the column
   is noise and is not read. If the column does not collapse, the curve is
   not reading the kernel it claims to read and P7/P8 are void.

## What would make each point wrong

1. -> a crash or a flag typo. Fix, rerun. A grid REFUSAL here would be a bug
   in the new guard, not in the grid: every commanded value is inside +-50 mm.
2. -> no pull, or the old marker `-2026-09-05`: rerun, nothing else.
3. -> `held False` on any row means the clamp is not the only thing that moves
   a pose during the settle, and the mechanism in `PROBLEMS.md` is incomplete.
   Name it and stop; do not read the reward off a drifting pose again.
   `all_poses_held True` while a `got_y` is visibly off means the drift
   arithmetic or its tolerance is wrong -- the gate's first real execution.
4. -> `interpen` > 0.2938 mm in free air: the block's top face is not at
   +15 mm in the USD as the cfg says. A scene finding; name it and stop.
5. -> the 2026-09-05b edit moved a number. Revert and re-measure before
   anything else is read.
6. -> any nonzero: read the cited code line before reading the number.
7. -> nothing; every outcome is a finding. A mixed sign (falls, then rises)
   needs the four envs checked against each other in the SUMMARY row; if they
   disagree the SDF query is noisy at that pose and the row is UNKLAR.
8. -> if g_v itself is ~0 (h +45, +30, +16 read the same kernel at y 0), the
   height gradient is flat in this band, the fraction has no denominator, and
   that is a stronger finding than either branch. Name it.
9. -> see point text.

## Read, NOT expected

`sdf` at every offset -- no prediction. `force` per row in detail. Seconds per
point. The per-env spread of `kernel_sum` (should be identical to ~1e-6).
The 0.1 x g_v ratio on the valid rows: recorded for continuity with RT-151b,
not used to decide.

## What this run cannot answer

* Nothing about what the POLICY does with the landscape. 27 poses, no policy.
* Nothing about WHY RT-148b's policy sat at 52.5 mm, where this run shows the
  reward is LOWEST. That contradiction is open and this grid does not touch it.
* Nothing about x offsets (RT-149: 5/8 envs at |x| 48-49.9 mm) -- RT-153.
  Nothing about yaw (RT-149 held 25 deg) -- RT-152.
* Nothing below the entrance plane with an offset (refused on purpose).
* Nothing about tilt: the teleport commands the part upright.
* Nothing about NEGATIVE y: free travel there is 0.8 mm in stage 1, and the
  grid runs +y only. RT-149's y -52 mm side stays unmeasured.
* **Nothing above +50 mm in any axis.** The new guard refuses it, so RT-123's
  +165 mm anchor cannot be re-measured with this tool as it stands. Re-opening
  that needs a different instrument, not a wider grid.
