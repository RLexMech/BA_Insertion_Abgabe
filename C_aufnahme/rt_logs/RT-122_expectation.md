# RT-122 — expectation, written BEFORE the run

Written 2026-09-01 on the dev laptop. `/rt-check` judges the log against
THIS file and nothing else.

## What the run is

THE DISCRIMINATING MEASUREMENT for the RT-120 / RT-121 force conflict.
It is NOT training and NOT the curve teleport — it is the env's OWN reset
path (D-162 dls-IK) with ZERO actions:

* RT-120 (training, start −26 mm): force p50 **93.5 N**, every episode
  force-aborted after mean 3.6 steps.
* RT-121 (curve teleport, −26 mm, zero actions): force **0.17 N**.

Same commanded height, two instruments, a factor of ~550 between them.
D-163's mechanism sentence ("the reset teleports into contact") hangs on
this. Two candidates, decided here and not by plausibility:

* **(a)** the untrained policy's exploration rams the part (RT-120's
  `joint_target_lag` max 0.154 rad supports this);
* **(b)** the env reset lands differently than the curve's script-local IK
  teleport (curve: up to 120 IK iterations per point; training reset:
  converged in 5 — noted as reading R3 of RT-121).

`zero_agent.py` now takes `--start-tip-above-entrance-mm` (goes through
the real reset path) and `--print-force` (per-step depth + tared force
norm across envs). 1024 envs — the SAME reset population as RT-120.
20 steps — RT-120's episodes died after 3.6, so the window is ample.
`rl_terms_enabled` is False in this script (measurement env), so there is
NO force abort: even a 93 N contact keeps stepping and stays readable.

## The commands (training PC, repo root, conda env active)

```
git pull
git rev-parse --short HEAD
```

Expected HEAD: the commit named in the handover message.

```
.\scripts\rt_log.ps1 RT-122 python scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --start-tip-above-entrance-mm=-26 --print-force --max-steps 20
```

## Points, each PASS/FAIL on its own

1. **P1 — it starts and survives.** Exit 0 (tag `zero_agent`), no
   traceback, no `nan`. The exit-code trap applies: the 20 per-step lines
   must exist.
2. **P2 — the code is this code.** The `[rt_log]` header prints the git
   hash from the handover. An older hash voids everything below.
3. **P3 — the run is the configured one.** The log shows, in order:
   `start_tip_above_entrance overridden: -26.0 mm`, the env's startup
   line `rung-0 start pose: tool point -26.000 mm above the stage-2`,
   and NO `START POSE SOLVE DID NOT CONVERGE` warning anywhere.
4. **P4 — the start pose lands.** Step 1's `depth p50` reads about
   **+26 mm** (tip 26 mm inside the pocket; IK tolerance 0.05 mm, physics
   settle may move it a little). A depth near 0 or near +165 mm means the
   override never reached the reset and every force line is void.

## THE DISCRIMINATOR (P5) — decision rule fixed BEFORE the numbers

Read `force p50` over steps 5–20 (the EMA needs a few steps to fill;
RT-120's own p50 was a max over ≤4-step episodes of the same EMA, so
early steps can only UNDERstate, never overstate).

* **Outcome A — the reset itself is in contact:** force p50 in the tens
  of newtons (order of RT-120's 93.5 N). Candidate **(b)** confirmed:
  the env reset ≠ the curve teleport, RT-121's curve does NOT measure
  the reset path, and the `z_low` selection rule must be re-measured
  THROUGH the reset path (a `zero_agent` height series), not from the
  RT-121 curve. D-163's mechanism sentence stands.
* **Outcome B — the reset is clean:** force p50 < 1 N (the L8 free-air
  tolerance) across all steps and the `>1N` count stays near
  0/1024. Candidate **(a)** confirmed: RT-120's forces came from the
  policy's own actions, RT-121's curve is valid for `z_low`, and D-163's
  mechanism sentence gets an inline correction. Plan step 2 may proceed
  with `z_low` from the RT-121 curve.
* **Anything between** (p50 between 1 N and ~30 N, or a large `>1N`
  count at low p50): UNKLAR — no z_low decision, own diagnosis step
  (`/mattpocock-skills:diagnosing-bugs`), one hypothesis per change.

No outcome is predicted. That is the point of the run.

## Readings taken OUT of the run (no thresholds)

* **R1:** the depth drift over 20 steps under zero actions — does the
  part sink, hold, or get pushed out.
* **R2:** the `max` force column — a single-env outlier at low p50 would
  point at a per-env geometry effect (fixture xy noise ±5 mm), not at
  the reset mechanism itself.

## What this run cannot answer

* Nothing about learning — no policy runs here.
* If Outcome A: not WHERE the contact boundary of the reset path lies —
  that needs its own height series through `zero_agent`.
* Nothing about the tilt strategy or `rung_step_sizes`.
