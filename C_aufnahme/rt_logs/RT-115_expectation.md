# RT-115 — expectation, written BEFORE the run

Written 2026-08-31 on the dev laptop. `/rt-check` judges the log against THIS
file and nothing else.

## What the run is

The same lateral counter-proof as RT-114, re-run with the GRAVITY TARE on. It
decides ONE thing: does the force channel now read the contact load alone?

RT-114 was sound in every geometric point (all L points PASS, no abort,
`LATERAL_CONTROL_HOLDS`) but read **8.08 N in free air** — the part hanging
121.6 mm from the block with zero interpenetration. That is the welded tool's
own weight (0.824 kg x 9.81 = 8.083 N), already measured in RT-59 at ratio
1.000, and it is in the signal because `body_incoming_joint_wrench_b` is the
wrench the parent link applies to the child. FORGE never sees it: Factory and
FORGE spawn robot and held asset with `disable_gravity=True`. This repo keeps
gravity on by supervisor decision.

ONE thing changed: `cfg.force_gravity_tare = True` subtracts
`-mass * gravity`, rotated into the parent link's frame, before the EMA.

Code under test: marker `check_seated_success-2026-08-31g`, env marker
unchanged in kind. Offline before the run: `check_insertion_math --self-test`
COUNTER-PROOF PASSED, `check_seated_success --self-test` 37/37,
`selftest_checks --offline` PASS.

## The sign is NOT claimed here

The tare's direction was derived from the API docstring ("wrench applied from
body parent to child body"), not measured. This run is what decides it, and
both outcomes are readable in the log:

| tared \|F\| in free air | meaning |
|---|---|
| ~0 N | the sign is right |
| ~16.17 N | the sign is flipped (the tare was ADDED) — flip and re-run |
| ~8.08 N | the tare never ran — check the startup report line |

The run prints the UNTARED wrench beside the tared one for exactly this.

## Points, each PASS/FAIL on its own

1. **P1 — it starts and survives.** Exit 0, no traceback, no `nan`.
2. **P2 — the tare is really on.** The startup report line `force tare:  ON`
   appears, naming the world hold force `[0.0, 0.0, 8.0834]` N, the frame
   `wrist_3_link` (index 6), and "Free air must now read ~0 N, not 8.08 N".
3. **P3 — the geometry is unchanged from RT-114.** Mode line reads
   `y = 191.89 mm`, commanded depth 34.0 mm, band 33.0–36.0 mm,
   `fixture_pos_noise_xy = 0.000`, solve residual < 0.05 mm and `converged`.
4. **P4 — NO abort.** No `ABORT at settle step` line; all 30 settle steps run.
5. **P5 — the tared force is ~0.** `force N:` **< 1.0 N** in every env. This
   is the whole run.
6. **P6 — the control still shows the weight.** `force N RAW (untared
   wrench, the control):` **≈ 8.08 N** in every env. If the raw line were
   also ~0, the tare would not be what changed the reading and P5 would prove
   nothing.
7. **P7 — every L point PASSes**, L8 included.
8. **P8 — the verdict.** `VERDICT: LATERAL_CONTROL_HOLDS`.
9. **P9 — the metrics file agrees.** `check_seated_success_metrics.json`:
   `verdict_ok: true`, `force_n` all < 1.0, `force_n_raw_untared` all ≈ 8.08,
   `terminated_any: false`. Numbers from the file, never from the prose.

## What this run cannot answer

* **Nothing about the inertial term.** The tare removes `mass * gravity`, not
  `mass * acceleration`. This run settles with zero action, so acceleration is
  ~0 and the term is invisible here by construction. A moving policy will see
  it, and no static tare can remove it.
* **Nothing about the seated case.** Only `--lateral` runs. The seated and
  `--shallow` counter-proofs must be re-run separately before their old force
  numbers are quoted again.
* **Nothing about the 60 N abort limit.** The limit is unchanged and is still
  the `[placeholder]` D-114 wants measured. After the tare the same 60 N
  admits ~8 N MORE real contact than before. That is a consequence to record,
  not a result of this run.

## Red flags to report unprompted

* `force tare:  OFF` in the startup report — the run tested nothing.
* Tared force ~16.17 N — flipped sign, see the table above.
* Raw force NOT ~8.08 N — then the baseline itself moved and the comparison
  to RT-114 is not controlled.
* Any `ABORT at settle step` line.
