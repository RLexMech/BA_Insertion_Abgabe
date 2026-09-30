# RT-123 — expectation, written BEFORE the run

Written 2026-09-01 on the dev laptop, after the reward revision landed
(inbox entry "Reward-Ueberarbeitung" 2026-09-01, p1-konzept-messung;
code commits c265783 / 44441ed / d3171b1). `/rt-check` judges the log
against THIS file and nothing else.

## What the run is

A MEASUREMENT, not training: the RT-119/121 instrument re-run under the
REVISED reward (margin_coarse = 173.21 mm, time penalty -1/256,
action-rate penalty, success terminates with payout). Two legs:

```
git pull
git rev-parse --short HEAD
```

Expected HEAD: given in the handover message.

```
.\scripts\rt_log.ps1 RT-123 python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 165 50 30 20 10 0 -10 -20 -26 -34 --out rt123_curve.json
.\scripts\rt_log.ps1 RT-123 python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 165 50 30 20 10 0 -10 -20 -26 -34 --curve-coarse-margin-mm 10 --out rt123_curve_counterproof.json
```

The counter-proof is TURNED AROUND vs RT-121: the live margin is now
wide, so the paid mutation NARROWS it back to 10 mm and must collapse
the far values again.

## Points, each PASS/FAIL on its own

1. **P1 — both runs start and survive.** Exit 0 each, no traceback, no
   `nan`; a per-height line for every commanded height. NOTE: the env
   code must first COMPILE under real torch.jit (the laptop shim stubs
   jit out) — an import-time TypeError here is a FAIL of this point.
2. **P2 — the code is this code.** The `[rt_log]` header prints the
   git hash from the handover.
3. **P3 — the grid is the commanded grid.** Ten per-height lines in
   order; every IK residual under 0.05 mm.
4. **P4 — the far field is alive.** Main leg: `kernel_sum` at 165 mm
   in the 0.05 order of magnitude (offline shim measured 0.0516); at
   +30 mm around 0.23 (RT-121 counter-proof at margin 200 read 0.2297;
   173.21 mm must land near it, same order). Monotone rise toward the
   seat.
5. **P5 — the counter-proof pays.** With `--curve-coarse-margin-mm 10`
   the 165 mm value collapses by many orders of magnitude (offline
   shim: 3.6e-22 class).
6. **P6 — the near field is unchanged.** At -20 mm and the seat the
   kernel_sum stays in the RT-119 order (0.18 / 0.58 class): mid and
   fine kernels untouched.
7. **P7 — the seated exit is the success exit.** The seated
   teleport point(s) terminate via SUCCESS (not force abort, not
   timeout), and the reward column shows the payout arithmetic the
   script now prints.

## Readings taken OUT of the run

* R1: kernel_sum at 165 / 30 / 0 mm under the live margin — the new
  gradient the PPO probe will see.
* R2: the reward column at the seat with the payout — the number the
  value function must span.
