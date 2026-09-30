# RT-179 — expectation, written BEFORE the run (2026-09-10)

The first AutoDR run that can GROW. RT-178 proved the provider is wired and
reports itself honestly; it could not move a single boundary, and its own
expectation file said so in advance as arithmetic. This run asks the next
question and only that one:

**Does AutoDR actually open the bands, or does it sit at width 0?**

Code `17a7f92`. **The code is byte-identical to RT-178's `307f764`** —
`git diff --stat 307f764 17a7f92 -- source/ scripts/` is empty; the two
commits in between are documentation. So every difference between RT-178 and
RT-179 is the budget and the stop rule, nothing else.

The six ceilings are UNCHANGED and all six still carry `[TESTWERT]`. The
decision that would have fixed them is ON HOLD (`docs/decisions_inbox.md`,
2026-09-10), and the literature check of 2026-09-10 found no rule relating a
start offset to the clearance. That is deliberate: this run is what produces
the end-state table the ceilings will eventually be argued against.

## The one hypothesis

**With the bands allowed to open, the policy learns at width 0 first, the
boundaries then advance, and the run either reaches all eleven maxima or
names the boundary that brakes.** Every line below is readable off the log,
off `demo_metrics.json`, or off the `autodr_<it>.json` chain.

## The budget, and what the arithmetic says it can buy

MEASURED at RT-178, not estimated: `dr_state.fill` grew to 152 .. 182 per
boundary over 60 iterations, mean 166.7, sum 1834 — that is **2.78 flags per
boundary per iteration**, against the plan's estimate of 2.91.

    first buffer full   = 240 / 2.78          = 87 iterations
    steps to a ceiling  = DELTA_STEPS         = 10 advances
    bare minimum        = 10 * 87             = 870 iterations

So 1500 iterations is about **1.7x the bare minimum**, and the bare minimum
assumes every single full buffer produces an ADVANCE. It will not: while the
success rate is still low, a full buffer means `mean <= retreat_at` = a
RETREAT, clamped at width 0 (`clamped_zero`), and the buffer clears and
refills. Every such buffer costs ~87 iterations and moves nothing.

**The one unknown that decides this run:** at which iteration the boundary
success rate first reaches `advance_at` = 0.80. RT-177 reached 0.80 at
iteration 464 on the FULL fixed ranges (plan, context paragraph — a written
number, not a measurement made here). Width 0 is strictly easier than that,
so it should come sooner, but by how much is UNKNOWN and this file does not
guess. If it comes at ~200, roughly 15 buffer fills remain against the 10
needed — tight, and it works. If it comes at ~600, it does not.

The 2.78 is also a LOWER bound: it was measured on a random policy whose
every episode ran the full 256 steps (983040 steps / 3751 episodes = 262).
Successful episodes end early, which RAISES the flag rate. So the count
above is pessimistic, in our favour.

**Run time:** RT-177 did 1108 iterations in about 9 h (plan, context
paragraph), i.e. ~29 s per iteration. 1500 iterations is therefore roughly
**12 h**. The friction write measured at RT-178 costs 0.478 ms mean
(L597) and does not change that.

## Command (training PC, PowerShell, conda env `env_isaaclab`)

```
.\scripts\rt_log.ps1 RT-179 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max_iterations 1500 --stop-when-dr-max 0.80 env.dr_mode=autodr env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0
```

Identical to RT-178 except `--max_iterations 60` -> `1500` and the added
`--stop-when-dr-max 0.80`. The 0.80 is the plan's number (§ 4) and is passed
explicitly because `train.py` refuses to default it. `--stop-check-every`
stays at its default 50.

**CHECKPOINT NAMES WILL LOOK WRONG, AND THAT IS KNOWN.** The stop rule runs
the block loop, and rsl_rl 3.0.1 repeats one iteration index per block
boundary (`train.py --stop-check-every` help; the InBachelorErwähnen row of
2026-09-10). The number of gradient updates is still 1500; the final
checkpoint is named `model_<updates - blocks + 1>`. Do not read a checkpoint
number as an update count.

## PASS / FAIL lines, in order

**P1 — IT STARTS AND THE STOP RULE IS ARMED.** No refusal of
`--stop-when-dr-max`, and the startup report is RT-178's: `provider: AutoDR,
11 boundaries, bounds_version 0`, six `now` pairs at width 0, the six `max`
pairs unchanged, the wiring line, six `[TESTWERT-check]` lines. FAIL on any
difference from RT-178 here — the code is identical, so a difference means
the command changed something it should not have.

**P2 — SUCCESS APPEARS AT WIDTH 0.** The fresh regular-episode success rate
rises above 0 and keeps rising. Read `dr/train_success_regular_fresh` and
`demo_metrics` `success_rate_recent`. RT-178 measured 0.0 over 60 iterations
with `buf_s` eleven times 0. FAIL if it is still 0.0 at iteration 300: then
nothing downstream can happen and the run is answering a different question
than the one it was started for.

**P3 — THE FIRST `[autodr]` LINE FIRES, AND ROUGHLY WHEN COMPUTED.**
EXPECTED at or after **iteration 87**, from the measured 2.78. Read as a
MEASUREMENT, not a threshold. Much earlier means the flag rate rose (check
the episode length); much later means boundary sampling is not reaching the
envs. READ THE DIRECTION: an early line is a `clamped_zero` retreat and is
NOT a defect.

**P4 — THE FIRST REAL ADVANCE. THIS IS THE RUN'S QUESTION.**
`bounds_version` goes above 0 on a boundary that actually MOVED, and the
`[autodr]` line says advance. Record the iteration and the boundary. A run
that ends with `bounds_version` still 0 answers the question with NO, and
that is a result, not a failure of the run.

**P5 — THE BANDS OPEN.** `dr/<key>_hi` climbs and `dr/<key>_lo` falls; they
are not flat. Check all eleven, not one.

**P6 — THE FLAG RATE RISES ABOVE 2.78.** Once episodes end early the fill
per iteration must go UP. If it does not while the success rate climbs, the
two readings contradict each other and one of them is wrong.

**P7 — THE STOP RULE FIRES, OR THE BACKSTOP DOES.** Both are readable
results. If it fires: all eleven boundaries at maximum AND fresh regular
success >= 0.80, and the log says so. If the backstop ends the run: report
how far each boundary got. FAIL only if the run stops for a third reason.

**P8 — THE SIDECAR CHAIN IS COMPLETE.** An `autodr_<it>.json` beside every
`model_<it>.pt`. Keys per the CORRECTED list (RT-178 P5 correction): `format
"autodr-1"`, `keys`, `dims`, `steps`, `buf_n`, `buf_s`, `dropped`,
`bounds_version`, `buffer_m`, `p_boundary`, `advance_at`, `retreat_at`. NOT
`mode`/`bounds`/`fill` — those are the telemetry dict's.

**P9 — THE KNOWN CLAMP DEFECT SHOWS UP, AND IS NOT READ AS A POLICY
RESULT.** This is EXPECTED, written here in advance so it cannot be
discovered as a surprise: `start_height.hi_max` = 0.050 m leaves the
controller's z box **exactly 0.000000 m** of headroom, and `tilt.hi_max` =
8.00 deg leaves the cone **0.52 deg** (InBachelorErwähnen, 2026-09-10). So
as `start_height_hi` opens toward its maximum, a boundary env starts ON the
clamp wall and its target is clamped from step 1. Consequence, decided
before the numbers exist: **a success rate at the top start-height boundary
or at full tilt is NOT a statement about the policy** in this run. Expect
`start_height_hi` to be a candidate brake in P10.

**P10 — THE BRAKE IS NAMED.** From the end-state table: which boundary did
NOT reach its maximum, and how far did it get. Report all eleven, plus
`dr/flags_dropped` (unconverged start poses, RT-178 measured 0).

**P11 — NOTHING BREAKS ONCE THE BANDS ARE OPEN.** No NaN, no traceback,
`mean_max_depth_mm` finite. Watch two numbers that RT-172..RT-176 tracked:
`force_norm_n.over_f_max` and `stage1_lateral_y_mm.over_reach` (the D-153
shoulder-hole exposure). RT-178 read 2 and 1579 on a random policy; these are
expected to FALL as the policy learns, as they did over RT-172..RT-176.

## What this run cannot answer

* Whether the six ceilings are RIGHT. They are `[TESTWERT]`, the literature
  gives no rule, and this run is one of the inputs to that argument, not
  its answer.
* Anything with a confidence interval. One seed. The plan wants five.
* Anything about the narrow fixture, which is a different geometry.
* Whether a boundary that reached its maximum is SOLVED at that maximum —
  P9 says why, for at least two of the eleven.
