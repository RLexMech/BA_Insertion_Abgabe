# RT-151 expectation — the lateral reward slice (two grids, two counter-proofs)

Written 2026-09-05 on the dev laptop, BEFORE the run. Plan:
`Pläne/Plan-Merge.md` Schritt 2, row RT-151; grid and rules from
`Pläne/Plan1.md` "RT-150 — der seitliche Reward-Schnitt" (the number moved
to RT-151 when RT-150 became the scripted seat). No measurement-code change: marker
`check_seated_success-2026-09-05` (= the 2026-09-03d code of RT-142's fourth fix plus a SUMMARY print block, no measurement change), which
has never run on the training PC in curve mode with
laterals — RT-135 ran the lateral grid on the 2026-09-02 marker).
`/rt-check` judges the four logs against THIS file and nothing else.

## What the run is

A MEASUREMENT, not a test: `check_seated_success.py --reward-curve`
teleports the part to a grid of (height above the entrance plane, pocket-y
offset), holds 4 zero-action settle steps per point and prints the env's
own caches: `sdf` (mean outside distance), `kernel_sum`, every reward term
per step, `engaged` / `success` / `in_pocket`, `interpen`, `force`.

THE QUESTION. RT-148b's policy ends 51.3 mm beside the pocket on the
block's top face (VERDICTS RT-148b NACHTRAG (4)); RT-149 shows it sliding
between y -52 and +9 mm at z 12-47 mm. Does the D-109 reward FALL when the
part moves sideways at those heights — a gradient a policy could follow
back — or is it flat there? Decides "no lateral gradient, the reward must
change" against "the gradient exists, exploration died first"
(Plan-Merge stop rule; RT-152/153 only if this shows no gradient).

Priors, both measured: RT-135 (h +1 mm, y 0..8 mm) kernel_sum fell
0.24625 -> 0.24611, 5.82e-5 over 4.88 mm, monotone; RT-123 (y 0) 0.2235 at
h +30, 0.0455 at h +165, 0.5853 at the seat. Nothing above 8 mm lateral or
at 30 mm height with an offset has ever been measured.

## Why two grids and not one

The block's top face is STAGE1_DEPTH = 15 mm above the entrance plane
(`insertion_tasks_cfg.py:674`). Stage 1 is 153.0 mm long in y against the
part's 143.5 mm, and it is ASYMMETRIC about the stage-2 origin:
-72.55..+80.45 mm (`:704`); the part's half-length is 71.75 mm, so the
free travel is +8.7 mm in +y and 0.8 mm in -y. A part below +15 mm with
y beyond +8.7 mm is INSIDE the block material — D-160 cause 1, the
impossible pose. Plan1 Fassung 1 had that error; the grid below does not.

- **a) inside stage 1:** heights 30.0 / 5.0 mm, y 0 1 2 4 6 8 mm. At h +5
  the lower 10 mm of the part sit in stage 1; at y 8 the +y face is
  0.7 mm from the stage-1 wall. Extends RT-135 (h +1) to h +5 and +30.
- **b) above the top face:** heights 30.0 / 16.0 mm, y 0 5 8.7 15 25 35
  45 51.3 60 mm. 16.0 mm is 1 mm above the face: free air for every
  offset. 51.3 mm is RT-148b's measured resting offset.

The h +30 / y 0 row appears in BOTH runs — the same pose, twice.

## Commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash in the handover message.

```
.\scripts\rt_log.ps1 RT-151a python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 30.0 5.0 --curve-lateral-y-mm 0 1 2 4 6 8 --out rt_logs/RT-151a_curve.json
.\scripts\rt_log.ps1 RT-151a-cp python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 30.0 5.0 --curve-lateral-y-mm 0 1 2 4 6 8 --curve-coarse-margin-mm 10 --out rt_logs/RT-151a_curve_counterproof.json
.\scripts\rt_log.ps1 RT-151b python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 30.0 16.0 --curve-lateral-y-mm 0 5 8.7 15 25 35 45 51.3 60 --out rt_logs/RT-151b_curve.json
.\scripts\rt_log.ps1 RT-151b-cp python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 30.0 16.0 --curve-lateral-y-mm 0 5 8.7 15 25 35 45 51.3 60 --curve-coarse-margin-mm 10 --out rt_logs/RT-151b_curve_counterproof.json
```

After each run the SHORT log is on the clipboard: paste it into
`rt_logs/inbox.txt`. Four pastes, nothing else (the JSONs stay on the
training PC as fallback; RT-135's were missing and P2/P4/P7 could not be
judged then — the SUMMARY block closes that gap).

## The discard rule — before any point is read

A row whose `interpen` column exceeds `INTERPEN_THRESH` = 0.2938 mm
(`insertion_tasks_cfg.py:1568`, half `PLAY_X`) is an IMPOSSIBLE POSE: the
solver put the part into material. The row is DISCARDED, not interpreted
— its `sdf`, `kernel_sum` and terms describe a buried part. A run whose
`sweep_end_reason` is `force_abort` is unusable from that point on (every
later row reads the post-reset home pose).

## Points, each PASS/FAIL on its own

1. **P1 — the four legs start and survive.** Exit 0 each, no traceback,
   no `nan`. a: 12 `curve  h` lines (30.0 x 6, then 5.0 x 6). b: 18 lines
   (30.0 x 9, then 16.0 x 9). Heights outer, laterals inner, in the
   commanded order. Header line `2 heights x 6 laterals = 12 points x 4
   settle steps = 48 of the 256 step episode` (b: `9 laterals = 18
   points ... 72`). `reset_during_sweep` false, `sweep_end_reason` null,
   `verdict_ok` true, last line `VERDICT: REWARD_CURVE_MEASURED`. The
   SUMMARY block is present (new 2026-09-05): one `SUMMARY grid` line, one
   `SUMMARY row` line per point with ALL FOUR envs' `kernel_sum` and `sdf_mm`,
   one `SUMMARY end` line; it carries the JSON fields every point below
   names, so the pasted SHORT log (`rt_logs/RT-151x.short.txt`, what
   `rt_log.ps1` puts on the clipboard) is enough — the JSON files are the
   fallback, not the handover.
2. **P2 — the code is this code.** `[rt_log]` header shows the handover
   hash; the log prints `marker: check_seated_success-2026-09-05`; JSON
   `curve_heights_mm` / `curve_lateral_y_mm` equal the commanded lists;
   `curve_tilt_deg` = `[0.0]`; `kernel_margin_coarse_mm` = 173.21 on the
   main legs; the counter-proof legs print `COUNTER-PROOF -- coarse
   margin forced to 10.0 mm, a_coarse = 299.32 /m` (acosh(10)/0.010).
3. **P3 — the pose is the commanded pose.** Every row: `solve_converged`
   true, `solve_residual_mm` < 0.05, `achieved_lateral_y_mm` (env 0, the
   `(got ...)` field) within 0.05 mm of the commanded y,
   `achieved_depth_mm` = -h within 0.05 mm (above the plane is a NEGATIVE
   depth: -30.0 / -5.0 / -16.0).
4. **P4 — no row is an impossible pose.** `interpen` <= 0.2938 mm on EVERY
   row of a and b (discard rule above). Sharper, from geometry: rows at
   h +30 and h +16 (free air) read `interpen` 0.000 and `force` < 1 N
   (RT-121 free air <= 0.56 N, RT-135 < 1 N). Rows at h +5: `force` < 1 N
   too — the part hangs 5 mm above the stage-1 floor with >= 0.7 mm to
   the wall. A force in the tens of N on an h +5 row is wall contact and
   that row is not free.
5. **P5 — instrument identity, two parts.**
   (i) The h +30 / y 0 row of run a and of run b: `kernel_sum` equal to
   each other within 1e-4, and equal to RT-123's 0.2235 within 1e-3
   (same code, same pinned pose, no reset noise: `fixture_pos_noise_xy`
   0, script pins it).
   (ii) The y 0 rows are the pure height curve: `kernel_sum` at h +5 >
   at h +16 > at h +30 (closer to the seat pays more; RT-123 was
   monotone in height; RT-135 read 0.2462 at h +1, so h +5 lies between
   0.2235 and 0.2462).
6. **P6 — rows that MUST read zero, and why.** On every row of every leg:
   `engaged` False and `success` False (the part is above the plane;
   ENGAGED_DEPTH = 0.3 x 36 mm = 10.8 mm below it, `:1675`);
   `max_depth_mm` 0.0 (the gated depth buffer books nothing above the
   plane); terms `engaged`, `success`, `success_lump`, `abort` = 0;
   `action_rate` = 0 (zero actions from a zero-cleared buffer);
   `progress` = 0 BY PINNING (`env_cfg.w_depth_progress = 0.0` at
   `check_seated_success.py:1504`, D-165) — not a finding; `time` =
   -0.00391 (-1/256) on every row. `in_pocket` True on every row: x is
   0 and every |y| <= 60 mm lies inside the D-157 box (+-72.55 mm,
   `POCKET_WALL_Y`); this gate does NOT see the block face. So the
   ONLY term that can move across a row is `kernels`, and `reward` =
   `kernels` - 0.00391 on every row.
7. **P7 — THE NUMBER, read, and the one shaped expectation (sign).**
   Per leg and per height, `kernel_sum` against y, verbatim into the
   verdict. Shaped: `kernel_sum` is NON-INCREASING in y at EVERY height
   (h +30, +5, +16). The SDF mean outside distance to the seated goal
   must grow with lateral error. A value that RISES with y at h +30 or
   h +16 is the strongest possible finding: the reward PAYS the part for
   moving away from the pocket at the heights where RT-148b/149 live —
   and no exploration argument saves it. Also read: at which y the fall
   flattens (the 25..60 mm rows of b against the 0..8.7 rows).
8. **P8 — the decision threshold, OWN CONSTRUCTION, stated now.** No
   source gives a "followable gradient" criterion; PPO's gradient does
   not see the constant time term at all. Two references, both from THIS
   run or a measured prior, are written down so the verdict cannot pick
   one afterwards:
   (R1, vertical) the height gain per mm at y 0 from the same leg:
   g_v = [kernel_sum(h +16, y 0) - kernel_sum(h +30, y 0)] / 14 mm (run b);
   g_v(a) = [ks(h +5) - ks(h +30)] / 25 mm (run a).
   (R2, prior lateral) RT-135's 5.82e-5 / 4.88 mm = 1.19e-5 per mm at h +1.
   The lateral gain per mm at h +16: g_l = [ks(y 0) - ks(y 51.3)] / 51.3 mm.
   Rule: g_l < 0.1 x g_v -> "flat sideways relative to down" (the policy
   gains ten times more per mm by descending than by centring; a lateral
   error is nearly free). g_l >= 0.1 x g_v -> "gradient exists". The
   factor 10 is MINE, an order-of-magnitude line, not a criterion from
   the literature; the verdict names it as such. Plan1's comparison
   against the time step 0.0039 is kept as a third, weaker reference only
   (RT-135's whole 8 mm slide was 0.015 x that step).
9. **P9 — the paid counter-proof (D-080), per leg.** With the coarse
   margin forced to 10 mm, EVERY row's `kernel_sum` on the cp leg is at
   least 10x SMALLER than the same row on its main leg (RT-135 measured
   18x at sdf 14.26 mm, h +1; at h +30 the coarse kernel 1/cosh(299 x
   0.039) is ~1e-5, so the ratio there is far larger). The sign in P7
   must survive the mutation only where the value is still resolvable
   (> 1e-4); below that the column is noise and is not read. If the
   column does NOT collapse, the curve is not reading the kernel it
   claims to read and P7 is void.

## What would make each point wrong

1. -> a crash, a flag typo, or a grid refusal (`curve_lateral_error` only
   refuses laterals with NEGATIVE heights; none here). Fix, rerun.
2. -> no pull, or an old marker: rerun, nothing else.
3. -> the IK did not reach the pose: the row is not evidence; if it is
   an h +5 / y 8 row, the 0.7 mm clearance was too tight for the solver
   — drop y 8, keep the rest.
4. -> `interpen` > 0.2938 on an h +5 row: the part met the stage-1 wall
   or floor — geometry error in THIS file (the +8.7 mm free travel or
   the 15 mm face height), not the reward. On an h +16 row: the block's
   top face is not at +15 mm in the USD as the cfg says — a scene
   finding, name it, stop the reward reading for run b.
5. -> (i) unequal across a/b: a reset randomisation the script does not
   pin; (i) off RT-123: the kernel constants changed since 2026-09-01
   and every prior anchor in this file is stale. (ii) non-monotone in
   height at y 0: the instrument itself is broken, P7/P8 void.
6. -> any nonzero: the gate or the pin is not what the cited line says;
   read the code line before reading the number.
7. -> nothing; every outcome is a finding. Only a mixed sign (falls, then
   rises) needs the per-env JSON: check that the 4 envs agree; if they
   do not, the SDF query is noisy at that pose and the row is UNKLAR.
8. -> if g_v itself is ~0 (h +30 and h +16 read the same kernel): the
   height gradient is flat in this band and the lateral question is
   moot — a stronger finding than either branch, name it.
9. -> see point text.

## Read, NOT expected

`sdf` at every offset — no prediction (the mean outside distance over
the SAPU point set of a 143 mm part is not a function of the offset
magnitude alone, and nobody has computed it for these poses).
`force` at h +5 rows in detail; seconds per point; the per-env spread
of `kernel_sum` (4 envs, should be identical to ~1e-6 with noise pinned).

## What this run cannot answer

* Nothing about what the POLICY does with the gradient. It measures the
  landscape at 30 poses and nothing else.
* Nothing about x offsets (RT-149: 5/8 envs at |x| 48-49.9 mm) — that is
  RT-153, new code. Nothing about yaw (RT-149: held 25 deg) — RT-152.
* Nothing below the plane with an offset (refused on purpose).
* Nothing about tilt: the teleport commands the part upright; RT-149's
  z 12.5 mm at y -31 mm with 1-1.6 N stays open.
* Nothing about NEGATIVE y: the free travel there is 0.8 mm, and the
  grid runs +y only. RT-149's y -52 mm side of the slide is not measured
  here; if run b shows a gradient in +y, the -y side is a separate grid
  at h +16 only (free air) — not pencilled, name it in the verdict.
