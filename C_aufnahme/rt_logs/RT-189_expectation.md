# RT-189 — expectation, written BEFORE the run (2026-09-12)

Written on the dev laptop by the main chat. `/rt-check` judges the logs
against THIS file. **UNVERIFIED** until the training PC has run it.

## What this is

`Laufplan_Phase5.md` step 11: the AutoDR main run on the wide fixture.
**One RT number for all five training seeds** (`Laufplan_Phase5.md:16-18`):
`RT-189s1` runs now, `RT-189s2` … `RT-189s5` later under THIS file.

Decided by the user 2026-09-12:

* **Seed 1 first, not a seed-20 probe.** If it works it already counts as the
  first of the five; if it does not, something must change and seed 1 reruns
  anyway, so a probe would have saved nothing in either case.
* **Cap `--max_iterations 3000`, resume if needed.**
* **256 environments** (safety margin, chosen before RT-188).

The smoke run RT-188 (`rt_logs/VERDICTS.md`, PASS) is the precondition: the
AutoDR chain starts on this code at 256 envs, fills as computed and writes its
sidecar.

## The two conditions under which seed 1 counts as one of the five

1. **The code stand is frozen.** The last commit touching `source/` or
   `scripts/` is `e7b85a2d` (read on the laptop 2026-09-12:
   `git log -1 -- source/ scripts/` on origin, and
   `git diff e7b85a2d HEAD -- source/ scripts/` empty at `05ca1cc2`). **All
   five seeds must run on a SHA whose `source/` and `scripts/` equal
   `e7b85a2d`'s.** No SHA of this file is pinned — it would invalidate itself
   on the next doc commit (the RT-184 lesson, `e791ca29`).
2. **Every seed gets the same cap and the same resume rule.** "All 5 seeds of
   a step get the same number of iterations" (`Laufplan_Phase5.md:13`). Under
   a stop rule a seed may stop earlier; the CAP is what must be equal.

**A consequence to know before starting:** the start-height lower edge is the
placeholder `start_tip_above_entrance` = 0.030 m, because H_min is not
measured (Laufplan step 4; `[TESTWERT]` in `autodr.DR_DIMS`). If H_min is
measured later and the centre changes, that is a config change and the seed
group run so far no longer belongs to the new setting.

## The one hypothesis

**"From zero, on the rebuilt code at 256 environments, AutoDR opens all seven
boundaries to their maximum and the fresh regular success rate reaches 0.995
within 3000 iterations."**

## Pre-registered branches — read at the end, from files

* **A — the stop rule fired.** `[stop-criterion] target reached at iteration
  N; stopping.` in the log, a `model_N.pt` with its `autodr_N.json`. Seeds 2–5
  run the identical command.
* **C — the cap was reached with boundaries opening.** `dr/bounds_version` > 0
  at the end but `dr/all_at_max` 0. Resume seed 1 (rule below). Seeds 2–5 later
  get the same total cap.
* **B — nothing opened.** `dr/bounds_version` still 0 at the cap. **No
  resume.** Run the project's Differentialdiagnose and start at point 4,
  reset distribution and curriculum rung — `Laufplan_Phase5.md` step 6b says
  so for exactly this pattern, and the reward is the last suspect. The known
  suspect: at width 0 every boundary episode starts at the fixed +30 mm centre,
  the start RT-187 (100 iterations) and RT-188 (30 iterations) never succeeded
  from; and every run that ever learned this task started from a checkpoint
  whose start band reached −30 mm (`docs/decisions_inbox.md`, the from-zero
  entry). That points at the start band, not at a proven cause.

## The arithmetic behind 3000 — measured, not assumed

RT-188 measured 1.070 flags per boundary per iteration at 256 envs, so one
buffer fill (240) takes **~224 iterations**. Every boundary needs
`DELTA_STEPS` = 10 advancing fills, then the stop rule needs a fresh window of
2000 regular episodes, 8 per iteration → 250 iterations.

* **Floor: 10 × 224 + 250 ≈ 2494 iterations.** 3000 leaves **~506 iterations
  ≈ 2.3 fills of slack** — for all seven boundaries together, since they fill
  in parallel.
* **So 3000 only suffices if almost every fill advances.** A fill advances
  only at boundary success ≥ 0.80; a fill below 0.10 retreats. Any boundary
  with more than ~2 non-advancing fills pushes the run past the cap → branch C.
* **Two effects move the floor, in opposite directions:** successful episodes
  end early, so episodes per iteration RISE and fills get faster; and every
  real bound move CLEARS the fresh window (`insertion_env.py:2928`), so the
  250-iteration refill restarts after the last move. Neither is quantified here.
* **Wall clock:** at RT-188's ~5.13 s per iteration, 3000 iterations ≈ **4.3 h**.
  Whether the iteration time changes as the boundaries open is unknown.

## What the stop rule does, read in `train.py:743-775`

`train.py` runs `runner.learn` in blocks of `--stop-check-every` (default 50)
and checks the criterion after each block, printing one `[stop-criterion]`
line. On "fire" it saves `model_<current_learning_iteration>.pt` explicitly,
which also writes the sidecar. Running to the cap, the help text states the
last checkpoint is named `model_<updates − blocks + 1>` because rsl_rl 3.0.1
repeats one index per block: 3000 updates in 60 blocks → **`model_2941.pt`**.
That name is read from the help text, not observed — confirm it in the folder
listing before any resume.

## The resume rule — for branch C only

What a resume under AutoDR requires, read in `train.py:565-615`:

* `autodr_<it>.json` must lie beside the checkpoint, or the run is REFUSED
  (a trained policy on width-0 boundaries is a warm start, not a resume).
* The sidecar's seed must equal `--seed`, or the run is REFUSED.
* `load_state_dict` restores boundaries, steps and buffers.
* `--max_iterations` is ADDITIVE to the loaded state (`VERDICTS.md:301`,
  RT-172 correction).
* `--load_run` takes the folder TIMESTAMP (it is a regex); `--checkpoint`
  takes the FILENAME for `train.py` (`HANDOFF-RL.md:1694`).

**What a resume does NOT restore: the fresh success window.**
`_fresh_successes` lives only in the env (`insertion_env.py:1177`); the
sidecar does not carry it. After a resume the stop rule cannot fire until
2000 new regular episodes have arrived — **≥ 250 iterations at 256 envs.**

**NAMED CONFLICT with `Laufplan_Phase5.md` step 7**, which resumes in blocks
of 50 iterations (= `save_interval`). That rule was written for the No-DR
runs. Under `--stop-when-dr-max`, a 50-iteration resume can never fire the
stop rule, because the fresh window cannot refill in 50 iterations. **A resume
block under AutoDR must be at least ~250 iterations for the stop rule to be
reachable at all.** The block size is the user's decision; this file only
states the floor.

Resume command template — the three values in angle brackets come from the
folder listing after the run and cannot be known now:

```
.\scripts\rt_log.ps1 RT-189s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations <BLOCK> --stop-when-dr-max 0.995 --resume --load_run <TIMESTAMP> --checkpoint <model_N.pt> env.dr_mode=autodr
```

Same log name `RT-189s1`: a resume is the same run, so appending to its log
file is intended.

## Commands (training PC, PowerShell, conda env `env_isaaclab`)

Root: `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`

```
git pull --ff-only origin p5-robustheit
```

```
git diff --quiet e7b85a2d HEAD -- source/ scripts/; if ($LASTEXITCODE -eq 0) { "code stand OK" } else { "CODE CHANGED -- STOP" }
```

```
.\scripts\rt_log.ps1 RT-189s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations 3000 --stop-when-dr-max 0.995 env.dr_mode=autodr
```

After the run:

```
Copy-Item logs\demo_metrics.json logs\RT-189s1_demo_metrics.json
```

```
Get-ChildItem logs\rsl_rl\ur5e_insertion -Directory | Sort-Object LastWriteTime | Select-Object -Last 1 | ForEach-Object { $_.Name; Get-ChildItem $_.FullName -File | Sort-Object LastWriteTime | Select-Object -Last 6 Name }
```

```
python scripts/export_tb_scalars.py --load-run <TIMESTAMP> --out docs/figures/RT-189s1_scalars.csv
```

## PASS / FAIL lines, in order

**P1 — the code stand.** The log header's `[rt_log] git:` SHA satisfies
`git diff --quiet e7b85a2d <SHA> -- source/ scripts/` on the laptop. If not,
seed 1 does not belong to this group — VOID, not FAIL.

**P2 — it starts as intended.** No `ValueError`, no Traceback. The startup
report shows `dr_mode: autodr`, 7 boundaries, `bounds_version 0`, 256 envs.
The run folder name ends in `seed1`.

**P3 — the stop rule is armed.** A `[stop-criterion]` line after every
50-iteration block. None at all = the rule never ran = FAIL.

**P4 — does AutoDR open anything.** From the scalars CSV: the FIRST iteration
at which `dr/bounds_version` leaves 0, and its value at the end. This point
decides branch B against A/C.

**P5 — the six metrics, in order, with a trend.** Episode Reward → Policy
Loss → Value Loss → Entropy → Explained Variance → KL. EV and KL are not
logged by rsl_rl 3.0.1 (D-116); name the substitute used. No verdict before
all six are written down.

**P6 — the end state, from the last sidecar.** `autodr_<last>.json`: `steps`
for all seven boundaries, `all_at_max`, `bounds_version`. From the log's last
block: `dr/train_success_regular_fresh`, `dr/success_rate_boundary`,
`dr/fresh_n`.

**P7 — which branch, stated with its evidence.** A: the `target reached`
line and the saved file. C: `bounds_version` > 0, `all_at_max` 0 at the cap.
B: `bounds_version` 0 at the cap.

**P8 — ENGAGED-HOVER, carried over from `Laufplan_Phase5.md` step 6b.** That
rule was written for the No-DR budget run; it is applied here unchanged and
the user may drop it. Read at iteration 250 and at the end, from the CSV: the
share of `Episode_Reward/engaged` in the total reward, `mean_max_depth_mm`,
`success_rate`. The pattern it names: engaged above half the return, depth
well short of a seat, success 0.

**P9 — red flags, self-reported.** `action_sat_frac` trend (D-168 tripwire;
RT-188 read 0.324 → 0.342 over 30 iterations); entropy toward 0; value loss
diverging; NaN.

**P10 — the instruments, read not judged.** `interpen_max_mm` tail,
`force_abort_rate`, `stage1_lateral_y_mm.over_reach`, from
`logs/RT-189s1_demo_metrics.json`. Note: that file is the LAST 250-episode
dump, not necessarily the exact run end.

**P11 — the log is complete.** Exit line present, or its absence stated.

## What this run cannot answer

* **Anything about seeds 2–5.** One seed; no spread.
* **H_min.** The start-height lower edge is a placeholder.
* **The test rate.** `dr/train_success_regular_fresh` is a training rate
  under exploration noise; the evaluation table is step 12.
* **Whether 256 was the right count.** RT-187 measured throughput only; the
  env-count question stays unanswered.
* **Nothing here is a thesis number** until all five seeds and the eval exist.

## Status

**UNVERIFIED.**
