# RT-135 — expectation, written BEFORE the run

Written 2026-09-02 on the dev laptop, after `/rt-check` RT-134 (FAIL, the
part parks 4.88 mm beside the stage-2 axis at depth 0). Plan
`so-was-ich-jetzt-virtual-goblet.md`, step D. `/rt-check` judges the log
against THIS file and nothing else.

## What the run is

A MEASUREMENT, not training and not an identity test: the RT-123 reward
curve instrument (`check_seated_success.py --reward-curve`) with the new
`--curve-lateral-y-mm` grid. ONE height, 1 mm above the entrance plane (free
air, so nothing can abort the sweep), EIGHT pocket-frame y offsets from 0 to
8 mm, 4.88 mm among them — RT-134's median landing offset.

THE QUESTION. How much does the D-109 reward change when the part slides
sideways at depth 0? The SDF response to a LATERAL offset has never been
measured; every earlier curve (RT-119/121/123) swept height at lateral 0.
The plan's arithmetic ESTIMATES the change at ~4e-4 per step for 4.88 mm
(one tenth of the time penalty) — an estimate from `sqrt(36^2 + 4.88^2)`,
which the mesh-based mean-outside distance need not follow. This run replaces
the estimate with the number. Nothing here judges the task.

## The commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v1
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash given in the handover message.

```
.\scripts\rt_log.ps1 RT-135 python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 1.0 --curve-lateral-y-mm 0 1 2 3 4 4.88 6 8 --out rt135_lateral_curve.json
.\scripts\rt_log.ps1 RT-135 python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 1.0 --curve-lateral-y-mm 0 1 2 3 4 4.88 6 8 --curve-coarse-margin-mm 10 --out rt135_lateral_curve_counterproof.json
```

Paste both logs AND both JSON files into `rt_logs/inbox.txt`.

## Points, each PASS/FAIL on its own

1. **P1 — both legs start and survive.** Exit 0 each, no traceback, no
   `nan`. Eight `curve  h +1.0 mm | y ...` lines per leg, in the commanded
   order 0, 1, 2, 3, 4, 4.88, 6, 8.
2. **P2 — the code is this code.** `[rt_log]` header shows the git hash from
   the handover; the log prints `marker: check_seated_success-2026-09-02`;
   the JSON carries `curve_lateral_y_mm` = `[0, 1, 2, 3, 4, 4.88, 6, 8]` and
   `curve_heights_mm` = `[1.0]`.
3. **P3 — the pose is the commanded pose.** Every row: `solve_converged`
   true, `solve_residual_mm` below 0.05, and `achieved_lateral_y_mm` (env 0,
   the `(got ...)` field on the printed line) within 0.05 mm of the commanded
   y. `achieved_depth_mm` about −1.0 on every row (1 mm ABOVE the plane is a
   NEGATIVE depth).
4. **P4 — the two gated readings at the RT-134 pose.** `in_pocket` true on
   every row (|y| ≤ 8 mm is far inside the D-157 box, ±72.55 mm) and
   `max_depth_mm` 0.0 on every row (above the plane the depth metric books
   nothing). Anything else means the box gate or the depth buffer is not what
   `insertion_env.py:1102-1109` says.
5. **P5 — THE NUMBER, read.** Main leg: `kernel_sum` per y, and the
   difference `kernel_sum(y=0) − kernel_sum(y=4.88)`. Written down verbatim
   into the D-entry for the depth-progress term. No threshold.
   * The one shaped expectation: `kernel_sum` does NOT increase with |y|
     (non-increasing from y = 0 to y = 8). The SDF distance to the seated
     goal must grow with lateral error; a value that RISES with y means the
     kernel is not reading the lateral at all and the row is a FAIL.
6. **P6 — free air.** `force_n` below 1 N on every row of the main leg
   (RT-121 measured ≤ 0.56 N in free air, RT-132 < 0.01 N). A force in the
   tens of N means the pose is in contact and the sdf row is not free-air.
7. **P7 — the sweep is whole.** `reset_during_sweep` false,
   `sweep_end_reason` null, `verdict_ok` true, the log ends with
   `VERDICT: REWARD_CURVE_MEASURED` on both legs.
8. **P8 — the paid counter-proof (D-080).** Second leg, coarse margin
   forced to 10 mm: EVERY `kernel_sum` on the eight rows is below 1e-3
   (arithmetic: a = acosh(10)/0.010 = 299 /m at an sdf near 36 mm gives
   e ≈ exp(−10.8) ≈ 2e-5; the mid and fine kernels are spent there). The
   main leg's values sit near 0.2 (RT-123 measured 0.2235 at 30 mm and the
   seat at 0.5853; 1 mm above the plane lies between). If the counter-proof
   does NOT collapse the column, the curve is not reading the kernel it
   claims to read.

## What this run cannot answer

* Nothing about what a POLICY does with the gradient — it measures the
  reward landscape at eight poses and nothing else.
* Nothing below the plane: the grid refuses a lateral together with a
  negative height on purpose (`curve_lateral_error`).
* Nothing about tilt: the teleport commands the part upright.
