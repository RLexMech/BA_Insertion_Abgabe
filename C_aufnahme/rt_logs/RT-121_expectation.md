# RT-121 — expectation, written BEFORE the run

Written 2026-09-01 on the dev laptop, git `2f39886` (plus this file's own
commit). `/rt-check` judges the log against THIS file and nothing else.

## What the run is

A MEASUREMENT, not a test and not training: `check_seated_success.py
--reward-curve` (the RT-119 instrument, unchanged code) over a NEW height
grid, plus its paid counter-proof. It executes step 1 of the approved plan
(`plan-erstellen-wie-die-cozy-mccarthy.md`): before the rung-0 start becomes
a sampled interval `U[z_low, +30 mm]` (IndustReal §IV.G form), the two
unknowns must be read from the sim:

1. `kernel_sum` at **+30 mm** — D-163 states this value is UNMEASURED
   (RT-119 read +20 mm and +50 mm, never +30).
2. The **contact boundary**: the deepest height whose settled force is still
   free-air (< 1 N, the L8 tolerance). The plan's selection rule derives
   `z_low` from it. RT-120 proved −26 mm is beyond it (force p50 93.5 N).

The height ORDER is deliberate: high → low, the risky deep points LAST. If a
deep point trips the 60 N force abort, the episode resets and every LATER
point would silently measure the home pose; running deep points last keeps
the free-air points clean. A poisoned point shows its own oversized residual.

## The commands (training PC, repo root, conda env active)

```
git pull
git rev-parse --short HEAD
```

Expected HEAD: the commit that carries this file (given in the handover
message; it is ahead of `2f39886` by doc-only commits — the instrument code
itself last changed before RT-119 and is identical).

```
.\scripts\rt_log.ps1 RT-121 python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 50 30 20 15 10 5 0 -5 -10 -20 -26 --out rt121_curve.json
.\scripts\rt_log.ps1 RT-121 python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 50 30 20 15 10 5 0 -5 -10 -20 -26 --curve-coarse-margin-mm 200 --out rt121_curve_counterproof.json
```

Both use `RT-121`, so they share one log file. Separate `--out` names so the
counter-proof does not overwrite the measurement JSON. Settle steps stay at
the default 4 (11 points x 4 = 44 << 256, no mid-curve truncation).

## Points, each PASS/FAIL on its own

1. **P1 — both runs start and survive.** Exit 0 each, no traceback, no
   `nan`. The exit-code trap applies: `exit code: 0` alone is NOT proof —
   the per-height lines must exist.
2. **P2 — the code is this code.** The `[rt_log]` header prints the git
   hash matching the handover. An older hash voids everything below.
3. **P3 — the grid is the commanded grid.** Eleven per-height lines, in the
   order 50, 30, 20, 15, 10, 5, 0, −5, −10, −20, −26 mm. Each point's IK
   residual is under the 0.05 mm tolerance. A point with a blown residual is
   VOID (its pose is not the commanded pose), and every point AFTER an
   abort/reset is void too.
4. **P4 — the instrument reproduces RT-119 at the shared heights.**
   `kernel_sum` within the same order of magnitude as RT-119:
   +50 mm ≈ 2.29e-08, +20 mm ≈ 1.46e-04, +10 mm ≈ 2.04e-03,
   0 mm ≈ 1.64e-02, −20 mm ≈ 0.183. Fixture noise (±5 mm xy) moves the
   pose per reset, so exact equality is not expected; a different ORDER OF
   MAGNITUDE at any shared height fails this point.
5. **P5 — the counter-proof pays.** With `--curve-coarse-margin-mm 200`,
   the kernel value at +50 mm rises by ORDERS OF MAGNITUDE over the plain
   run's +50 mm value (RT-119's pattern: 165 mm went 3e-23 → 0.0648).
   If it does not move, the curve is not reading the kernel it claims.
6. **P6 — free air is free air.** At the heights ABOVE the (unknown)
   contact boundary, settled force reads < 1 N (RT-115 measured 0.002 N
   tared free air). Points at +50 and +30 mm MUST be free air — the part
   hangs 15+ mm above the stage-1 rim there (D-163). Force ≥ 1 N at +30 mm
   fails this point AND overturns the plan's premise for `z_high`.

## Readings taken OUT of the run (no thresholds, they decide the next step)

* **R1:** `kernel_sum` at +30 mm — the number D-163 flagged. It calibrates
  how weak the top-of-range reward is against the −1.0 abort payment.
* **R2:** the contact boundary height — first height (top-down) where force
  ≥ 1 N. The plan's rule: `z_low` = the deepest measured height that is
  still free-air. If even 0 mm shows contact, the sampled range covers only
  free air, and WHY the reset touches becomes its own diagnosis step
  (plan, step 1, IF-branch).
* **R3:** force at −26 mm. If it is LOW here, the curve teleport does NOT
  reproduce the RT-120 reset mechanism (curve: 120 IK iterations per point;
  training reset: converged in 5) — that discrepancy would be its own open
  question and goes to the handoff, not silently past.
* **R4:** force at −5/−10/−20 mm — where inside the pocket contact begins,
  for the later ladder's lower rungs.

## What this run cannot answer

* Nothing about learning — no policy runs here.
* Nothing about the tilt strategy — the teleport commands an upright part.
* Nothing about `rung_step_sizes` (D-110 (3): from learning curves, which
  do not exist yet).
