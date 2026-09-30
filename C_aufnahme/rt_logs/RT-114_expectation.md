# RT-114 — expectation, written BEFORE the run

Written 2026-08-31 on the dev laptop. `CLAUDE.md` requires the expectation to
exist before the log does; `/rt-check` judges the log against THIS file and
nothing else.

## What the run is

The lateral counter-proof of `check_seated_success.py`, RE-RUN with a
REACHABLE probe pose. It is the ONE discriminating measurement demanded after
D-159 was reverted (two failed attempts, `CLAUDE.md` § Change discipline).

RT-112 and RT-113 both aborted on settle step 1 with kN-scale forces and
8–16 mm of solver interpenetration. The offline re-derivation (user,
2026-08-31) found the cause: the probe teleported the part UPRIGHT to
y = 109.0 mm at 34 mm depth, and an upright part of 143.5 mm long-axis extent
centred there still covers the fixture block's outer material out to
`POCKET_LOCAL_Y_RANGE[1]` = 100.45 mm over the full depth. The forces were
correct physics for an impossible pose. Nothing about D-157 was measured.

ONE thing changed for RT-114: the probe offset. It is now computed from the
geometry constants, not typed —
`POCKET_LOCAL_Y_RANGE[1] + 0.5*hypot(PART_BBOX_M[0], PART_BBOX_M[1]) + 5 mm`
= **191.89 mm**, free air for any yaw. Second, smaller change:
`fixture_pos_noise_xy` is pinned to 0.0, because this is an identity test.

Code under test: `scripts/check_seated_success.py`, marker
`check_seated_success-2026-08-31f`. Offline before the run:
`--self-test` 34/34, `scripts/selftest_checks.py --offline` PASS.

## Geometry, so the log can be read against numbers

| quantity | value | home |
|---|---|---|
| block outer +y material | 100.45 mm | `POCKET_LOCAL_Y_RANGE[1]` |
| part long-axis extent | 143.50 mm | `PART_BBOX_M[1]` |
| bare clearance (upright) | 172.20 mm | computed in the script |
| **commanded offset** | **191.89 mm** | computed in the script |
| pocket wall (L2's bar) | 72.55 mm | `POCKET_WALL_Y` |
| commanded depth | 34.0 mm | `--depth-mm` default |
| success band | 33.0–36.0 mm | `depth_min` / `depth_max` |
| force abort limit | 60.0 N | `force_abort_f_max_n` |
| settle steps | 30 | `--settle-steps` default |
| solve tolerance | 0.05 mm | `--solve-tol-mm` default |

At 34 mm depth the part's tip sits at world z +0.0285 m; the table plate is at
world z −0.024 m. Roughly 52 mm of air below it. Nothing may touch anything.

## Points, each PASS/FAIL on its own

1. **P1 — it starts and survives.** Exit 0, no traceback, no `nan`. `exit
   code: 0` alone is NOT proof; the log must also carry no traceback.
2. **P2 — the run is the configured one.** The mode line reads
   `LATERAL COUNTER-PROOF (y = 191.89 mm, block outer edge 100.45 mm, wall
   72.55 mm)`, commanded depth 34.0 mm, band 33.0–36.0 mm. The startup report
   prints `fixture_pos_noise_xy = 0.000` and `inactive`.
3. **P3 — the teleport converges.** `solve residual` < 0.05 mm and the word
   `converged`. That is judged point L/P1 in the script itself.
4. **P4 — NO abort.** The line `ABORT at settle step` must NOT appear. All
   30 settle steps run. This is the point RT-112 and RT-113 failed.
5. **P5 — the forces are small.** `force N:` **< 1.0 N** in every env. The
   pose is free air; anything larger is not a resting contact and is the
   discriminating signal (see below).
6. **P6 — every L point PASSes.** L2 (outside the wall), L3 (projected depth
   inside the band), L4 (success predicate silent), L5 (engaged bonus
   silent), L6 (`max_depth` metric books nothing), L7 (paid reward equals the
   D-109 formula).
7. **P7 — the verdict.** `VERDICT: LATERAL_CONTROL_HOLDS`.
8. **P8 — the metrics file exists and agrees.** `check_seated_success_metrics.json`
   carries `verdict_ok: true`, `commanded_lateral_y_mm` ≈ 191.89,
   `fixture_pos_noise_xy_m: 0.0`, `terminated_any: false`, `in_region` all
   false, `engaged` all false, `max_depth_metric_mm` all 0.0. Numbers are read
   from this file, never from the prose in the log.

## What this run decides

* **All points PASS** → cause rank 1 (impossible probe pose) is confirmed.
  RT-112/RT-113 said nothing about the sim. The D-157 gate is demonstrated:
  the paying predicates stay silent beside the pocket. Handoff point 3 (the
  "wrapper" question) is answered NO — its condition never triggered. The
  path to RT-108 is free.
* **P5 fails: large forces in free air** → cause rank 2 (a solver/contact
  fault) becomes the live hypothesis, because the pose can no longer explain
  the force. THEN, and only then, a physics diagnosis and the wrapper
  question reopen.
* **P4 fails but P3 also fails** → the IK could not reach 191.89 mm; that is
  a reach limit, not a physics finding, and the offset is re-derived, not the
  sim doubted.

## What this run cannot answer, stated before anyone reads a number into it

* **Nothing about RT-107's real hack pose.** RT-107's 109 mm is a POLICY
  offset, reached by a part that was free to tilt. This run commands an
  upright part and cannot reproduce it. Characterising the real pose is a
  separate measurement with its own RT number.
* **Nothing about the seated case.** Only `--lateral` runs here. The seated
  and `--shallow` counter-proofs are unchanged and not re-run.
* **Nothing about training.** No policy, no reward tuning, 4 envs, one
  teleport.

## Red flags to report unprompted

* Any `ABORT at settle step` line — read the printed pre-step force, tip_rel,
  depth and interpenetration before anything else.
* `interpen max mm` above the SAPU threshold (0.2938 mm) in free air.
* `solve residual` converging but `lateral y mm` far from 191.89 — the
  command and the achieved pose disagree.
