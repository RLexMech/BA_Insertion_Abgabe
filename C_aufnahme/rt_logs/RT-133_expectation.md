# RT-133 — expectation, written BEFORE the run

Written 2026-09-01 on the dev laptop. `/rt-check` judges the log against
THIS file and nothing else. **This is RT-130's own retry after a fix; the
number RT-130 stays spent (FAIL is on record, `rt_logs/VERDICTS.md`).**

## What changed and why

RT-130 (2026-09-01 23:15:28) ran the same two commands below and its
seated leg FAILED: `SEATED_IDENTITY_FAILED`, `achieved depth mm:
[-165.0, -165.0, -165.0, -165.0]` despite the IK solve converging to
0.000 mm. `zero_agent.py --print-force` (RT-132) independently confirmed
the env's own reset lands correctly at both −30 mm and −34 mm and that its
reset transient reads 2.021 N — the exact force RT-130 printed for the
seated state. Diagnosis (`PROBLEMS.md`, 2026-09-01): the settle loop's
`env.step()` call, on the step that terminates, runs `_reset_idx` INSIDE
that call (the documented contract `dones -> rewards -> resets ->
observations`), so the returned `obs` already belongs to the NEXT
episode — the bare home pose, +165 mm above the entrance. This is the
"Reset-Step-Falle" CLAUDE.md names, invisible here until the 2026-09-01
revision made success terminate the seated run for the first time.

Fix: on a terminating settle step, `depths`/`lat_y_mm`/`force_n`/
`max_depth_mm` are now read from the PRE-step captures the abort branch
already used, not from the post-reset `obs`/`_max_depth`. The
no-termination path (shallow, lateral) is unchanged — it never hit this
bug. `force_raw_n` (the untared control print) shares the same bug and is
left open on purpose (named in a code comment, no cache exists for it,
not gated by any judged point).

Offline, ALREADY VERIFIED on the laptop (does NOT exercise the fixed
path — needs a real termination under real Isaac):
`python scripts/check_seated_success.py --self-test` → 41/41.
`python scripts/selftest_checks.py --offline` → PASS.
Marker bumped `check_seated_success-2026-09-01a` -> `...b`.

## The commands (training PC, repo root, conda env active) — identical to RT-130

```
git pull
git rev-parse --short HEAD
```

Expected HEAD: given in the handover message.

```
.\scripts\rt_log.ps1 RT-133 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --out rt133_seated.json
.\scripts\rt_log.ps1 RT-133 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --shallow --out rt133_shallow.json
```

## Points, each PASS/FAIL on its own

1. **P1 — both legs start and survive.** Exit 0 each (Ctrl+Break, not
   Ctrl+C, if it must be stopped early — Ctrl+C drops the exit line), no
   traceback, no `nan`.
2. **P2 — the code is this code.** The `[rt_log]` header prints the git
   hash from the handover; the script prints
   `marker: check_seated_success-2026-09-01b` (the `b`, not RT-130's
   `a`); the startup report prints the force limit as 300.0 N.
3. **P3 — THE FIXED NUMBER.** `achieved depth mm` reads close to
   34,000 (band 33.0–36.0 mm), not −165.0. This is the one line that
   failed in RT-130 and is the discriminator for this run.
4. **P4 — seated leg, the rest of the script's own points.** IK
   converged; success predicate fires in every env; engaged bonus fires;
   SDF mean-outside under the rigid-shift bound; the paid reward equals
   the D-109 formula INCLUDING the payout lump; episode TERMINATES on the
   success exit, not the force abort. VERDICT line
   `SEATED_IDENTITY_HOLDS`.
5. **P5 — shallow leg unchanged.** Every depth below the band, predicate
   silent, episode does NOT terminate, reward below the peak-minus-one
   ceiling, identity holds. VERDICT line `SHALLOW_CONTROL_HOLDS`. This
   leg already PASSED in RT-130 and must not regress.

## Readings taken OUT of the run — no threshold

* R1: the seated force under the tare, now read from the pre-step
  capture. RT-130 printed 2.02 N (the reset transient, wrong); a correct
  reading should sit near the RT-81 UNTARED figure minus the ~8 N tare,
  i.e. close to 0 N, or whatever the pre-step contact force actually is
  at the settle step that terminates.
* R2: the paid seated reward with the payout, as printed.

## What this run does NOT decide

Nothing about the approach, the tilt, or learning. It only re-establishes
that the success wiring is right before RT-134 (the PPO run, formerly
planned as RT-131 — renumber only if RT-131 is still free when this runs)
trains on it.
