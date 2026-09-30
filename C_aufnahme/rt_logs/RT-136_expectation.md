# RT-136 — expectation, written BEFORE the run

Written 2026-09-02 on the dev laptop, after the per-term reward log
(commit `017149d`) and D-165 (commit `28e0d74`). `/rt-check` judges the
log against THIS file and nothing else.

## What the run is

The seated identity test, re-run because the reward function was SPLIT
(`compute_reward_terms_insertion` + the summing wrapper) and gained the
D-165 progress row. Two things are new on Isaac and were proven only under
the laptop's numpy stand-in: (1) the split function compiles under real
`torch.jit.script` (a scripted function calling a scripted function, a
`torch.stack` of a list, `torch.sum(dim=0)`); (2) with `w_depth_progress`
PINNED to 0.0 by the script, the paid reward still equals `expected_reward`
at the seat (P8 identity) — i.e. the split changed the arithmetic by nothing.

The step order (`_get_dones → _get_rewards → _reset_idx`) also runs the new
per-episode accumulation for the first time on Isaac.

## The commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v1
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash given in the handover message. `28e0d74` must be
in the history (`git log --oneline 28e0d74 -1`).

```
.\scripts\rt_log.ps1 RT-136 python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless
.\scripts\rt_log.ps1 RT-136 python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --shallow
```

Same two legs as RT-133 (its PASS is the baseline: `achieved depth 34.0 mm`,
`SEATED_IDENTITY_HOLDS`, `SHALLOW_CONTROL_HOLDS`).

## Points, each PASS/FAIL on its own

1. **P1 — both legs start and survive.** Exit 0 each, no traceback, no
   `nan`. **The JIT trap is the point of this run**: an import-time or
   first-call `RuntimeError` from `torch.jit` naming
   `compute_reward_terms_insertion` or `compute_rewards_insertion` is a
   FAIL of this point and stops everything downstream.
2. **P2 — the code is this code.** `[rt_log]` header shows the git hash
   from the handover; `marker: check_seated_success-2026-09-02`; the
   startup report's proxy-value block lists `cfg.w_depth_progress` with its
   `[proxy]` mark — and, because the script PINS it, the printed value is
   **0.0**, not 100.0.
3. **P3 — the seated identity holds under the split.** Seated leg:
   `VERDICT: SEATED_IDENTITY_HOLDS`; achieved depth in the 33.0–36.0 mm
   band (RT-133: 34.0 mm); P8 line reports the paid reward equal to the
   formula (the script's own tolerance), payout included.
4. **P4 — the shallow control holds.** `VERDICT: SHALLOW_CONTROL_HOLDS`;
   neither predicate fires at 15 mm; no payout.
5. **P5 — the success exit, not the abort.** Seated leg terminates on the
   success predicate; `force abort` silent; resting force far below 300 N
   (RT-133 class, i.e. below 1 N after the gravity tare).

## What this run cannot answer

* Nothing about the size or effect of the progress term — it is pinned
  to 0 here by design. RT-137 reads it.
* Nothing about `Episode_Reward/<term>` curves — they go to rsl_rl's
  logger, which this script does not run. RT-137 reads them.
