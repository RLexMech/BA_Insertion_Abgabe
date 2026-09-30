# RT-155 — expectation, written BEFORE the run

Written 2026-09-05 on the dev laptop, while RT-154 runs. `/rt-check` judges
the log against THIS file and nothing else.

## What the run is

The teleport success identity test (`CLAUDE.md` differential diagnosis
step 2), **for the first time under the OSC controller**. Every PASS of the
seated leg so far (RT-109/110, RT-133, RT-136) ran under `joint_pd`; the
controller switch to `osc` landed 2026-09-03 (`52c0853`). The script sets
no `control_mode`, so it inherits the cfg default `osc`, kp 100, 15 Hz.
Only RT-142a (2026-09-03, FAIL for an angle bug) showed `success_termination`
firing under OSC, as a side reading.

Why now: RT-156 (SBC start under OSC) will be judged on `success_rate` and
`success_lump`. Without this run a zero there cannot be told apart from a
success predicate that does not fire under OSC.

Nothing in the repo changed for this run. Same commit as RT-154.

## Commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: `a123c4a` (the RT-154 commit; nothing newer is pushed).

```
.\scripts\rt_log.ps1 RT-155 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless
.\scripts\rt_log.ps1 RT-155 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --shallow
```

Both legs write to the same short log; paste it once into `rt_logs/inbox.txt`.

## Points, each PASS/FAIL on its own

1. **P1 — both legs start and survive.** Exit 0 each, no traceback, no
   `nan`.
2. **P2 — the code is this code.** `[rt_log]` header `a123c4a`;
   `marker: check_seated_success-2026-09-05c`; the startup report's
   `--- controller ---` block prints mode `osc`, `osc_kp_pos` 100.0, and
   does NOT print `*** DRIVES NOT INERT ***`.
3. **P3 — THE OSC QUESTION.** Seated leg: `VERDICT: SEATED_IDENTITY_HOLDS`;
   achieved depth in the 33.0–36.0 mm band (RT-133/136: 34.0 mm under
   joint_pd); the episode terminates on the SUCCESS exit, `force abort`
   silent; the paid reward equals the formula including the payout lump
   (the script's P8 line, its own tolerance).
   * FAIL here = the reward/termination wiring under OSC is broken and
     RT-156 must not run until it is fixed.
4. **P4 — the shallow control holds.** `VERDICT: SHALLOW_CONTROL_HOLDS`;
   neither predicate fires at ~15 mm; no payout; no termination.
5. **P5 — the settle does not move the seated part.** Under OSC the four
   settle steps run through `_apply_osc` with zero action (target =
   current pose, RT-145). The seated pose lies inside the ±50 mm tip box,
   so `clamp_tip_in_box` must be the identity there. Achieved depth after
   the settle within 0.5 mm of the commanded seat depth. RT-151b showed
   what the clamp does OUTSIDE the box; this point reads that it does
   nothing inside.

## Readings, no threshold

* R1: the tared seated force under OSC (RT-133 class: below 1 N).
* R2: `kernels` at the physical seat (RT-123: 0.5853, sdf 0.46 mm — NOT
  1.0, the peak is at sdf 0). A value far from 0.58 is a finding about the
  SDF query, not about the controller.

## What this run cannot answer

* Nothing about learning, exploration or the approach.
* Nothing about the ±50 mm clamp on the HEIGHT sweeps (RT-121/123 anchor,
  VERDICTS 2026-09-05 "NICHT NACHGEMESSEN") — a separate run.
