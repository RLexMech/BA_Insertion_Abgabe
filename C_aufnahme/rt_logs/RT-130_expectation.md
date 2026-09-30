# RT-130 — expectation, written BEFORE the run

**SPENT. RT-130 ran and FAILED** (2026-09-01 23:15:28, `rt_logs/VERDICTS.md`):
seated leg `SEATED_IDENTITY_FAILED`, a bug in this script, not the env
(diagnosis + fix: `PROBLEMS.md` 2026-09-01, `scripts/check_seated_success.py`
marker `...09-01b`). The retry with the SAME two commands is
`rt_logs/RT-133_expectation.md`, not this number — numbers are never
re-issued. Kept below as the historical record.

Written 2026-09-01 on the dev laptop. `/rt-check` judges the log against
THIS file and nothing else.

## What the run is

The SEATED TELEPORT IDENTITY TEST and its paid counter-proof, re-run under
the revised reward (inbox entry "Reward-Ueberarbeitung", 2026-09-01) and
under the new abort window (`force_abort_f_max_n` = 300 N, user-set
2026-09-01, inbox entry "The tilted scripted insertion is SPENT ..."). It is
item 2 of `HANDOFF-RL.md` § "What the next runs owe": the seated mode has
NOT been run since the reward revision, and its P4 is REVERSED there —
the episode must now TERMINATE on the success exit. RT-109/RT-110
(2026-08-31, git `33025fc`) are the last seated/shallow runs and predate
both the revision and the gravity tare.

RT-123 (2026-09-01 13:11:05) already re-measured the reward CURVE under
the revised reward and PASSED, so the curve is NOT re-run here.

Nothing is tuned. The only number that moved since RT-123 is the abort
limit, and a seated part rests at the tool's own weight (RT-81: 8.08 N
untared), so the limit cannot fire in either leg.

## The commands (training PC, repo root, conda env active)

```
git pull
git rev-parse --short HEAD
```

Expected HEAD: given in the handover message.

```
.\scripts\rt_log.ps1 RT-130 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --out rt130_seated.json
.\scripts\rt_log.ps1 RT-130 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --shallow --out rt130_shallow.json
```

## Points, each PASS/FAIL on its own

1. **P1 — both legs start and survive.** Exit 0 each, no traceback, no
   `nan`. The env must COMPILE under real torch.jit; an import-time error
   is a FAIL of this point. The exit-code trap applies: `exit code: 0`
   alone is not proof, the VERDICT line must be there.
2. **P2 — the code is this code.** The `[rt_log]` header prints the git
   hash from the handover; the script prints
   `marker: check_seated_success-2026-09-01a`; the startup report prints
   the force limit as 300.0 N with its `[placeholder]` mark beside it.
3. **P3 — seated leg, the script's own P1..P3.** IK converged, every
   depth inside the 33.0–36.0 mm band, the success predicate fires in
   every env.
4. **P4 — seated leg, THE REVERSED POINT.** The script's P4 reads
   `terminated True, success exit True`, and its P5 reads PASS (no force
   abort). A seated run that keeps stepping is the failure here.
5. **P5 — seated leg, P6..P8.** Engaged bonus fires; SDF mean-outside
   under the rigid-shift bound; the paid reward equals the D-109 formula
   INCLUDING the payout lump (this is the first run that would catch a
   wrong payout multiplier — P8 compares the paid reward against the
   formula with the lump). VERDICT line `SEATED_IDENTITY_HOLDS`.
6. **P6 — shallow leg, S2..S6.** Every depth below the band, predicate
   silent, episode does NOT terminate (neither exit may fire at 15 mm),
   reward below the peak-minus-one ceiling, identity holds. VERDICT line
   `SHALLOW_CONTROL_HOLDS`.

## Readings taken OUT of the run — no threshold

* R1: the seated force under the tare (`|F|` in the P1 detail line).
  RT-81 read 8.08 N UNTARED; a tared reading near 0 N is what D-160 says.
  Anything near 8 N means the tare is not in this path.
* R2: the paid seated reward with the payout, as printed — the number the
  PPO value function has to span (RT-123 printed 566.2 for the curve's
  seat point; this is the seated mode's own reading of the same lump).

## What this run does NOT decide

Nothing about the approach, the tilt, or learning. It says only that the
success wiring is right before RT-131 trains on it.
