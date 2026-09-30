# RT-176 — expectation, written BEFORE the run (2026-09-07)

Written on the dev laptop. `/rt-check` judges the log against THIS file and
nothing else. This is a PROBE RUN in the sense of D-174: one seed, `--resume`,
hand stop allowed. It is not an acceptance run and carries no thesis number.

## The one hypothesis

**"RT-175 was stopped before it was finished, and the two per-axis tables it
could not write will show that the lateral start and the pocket yaw are
learned across their whole draw — not only near the axis."**

## Why this run and not the next rung

Two reasons, both from RT-175's own numbers and neither of them a new idea.

1. **RT-175 was NOT saturated at its hand stop.** `RT-175_scalars.csv`, rows
   1450..1679: Episode Reward 318.8 -> 605.6 and still climbing; Value Loss
   ends at 1436, twice RT-174's 706; Entropy RISES 0.935 -> 1.767 and sigma
   0.5661 -> 0.5951. Every one of those says the run was mid-learning, not
   converged. Its 0.9925 is a lower bound, not an end state.
2. **RT-175 ran on `1f9eba1`, which predates D-173** (`b95fb77`;
   `git merge-base --is-ancestor` says NO). It therefore has neither
   `success_by_lateral_bin` nor `success_by_yaw_bin`. RT-174 has neither
   either. The current stage — tilt 0..8 deg + lateral 6 mm + yaw 6 deg —
   has never been read per axis by anything. D-173 point 7 states the rule
   this run satisfies.

A start-height rung on top would change an axis in the same run in which the
two tables appear for the first time. If a bin then reads low, nothing would
say whether that is the new height or the new instrument. One change per run:
here the change is the INSTRUMENT, and the configuration is frozen.

## The one change

**None to the environment.** Every `env.*` override is RT-175's, verbatim.
What changes is the code the run stands on: `bdacbb3` instead of `1f9eba1`,
which brings D-173's two tables and D-174's explicit `seed = 42`.

`seed = 42` is a provenance change, not a value change: `PPORunnerCfg` now
spells out the value it used to inherit from `RslRlOnPolicyRunnerCfg`
(IsaacLab 2.3.2, `rl_cfg.py:141`, "Default is 42"). The command still passes
`--seed 42`, so the run is bit-for-bit the same draw RT-175 had.

Resumed from RT-175's newest checkpoint, `model_1650.pt`
(`save_interval = 50`, run reached 1679). `--max_iterations` is ADDITIVE, so
`--max_iterations 300` targets 1950 and runs 300 iterations.

## PASS / FAIL lines, in order

1. **P1 — the reset is contact-free.** First block `force_abort_rate` 0.0000;
   startup report step 2 `|prim-buffer|` below 1e-5 m, max joint deviation
   0.000000 rad; the "entrance buffer" line shows a NON-ZERO yaw in both
   report envs (RT-175 read +0.0682 / -0.0601 rad).
2. **P2 — the run is what it says.** Folder tag carries `tilt0-8deg`,
   `yaw+-6deg`, `lat6mm`, `start+30..+50mm`, `seed42`. The
   `[INFO]: Loading model checkpoint from:` line ends on
   `09-07_09-03-58_..._lat6mm_seed42\model_1650.pt`. First iteration line
   reads `Learning iteration 1650/1950`.
3. **P3 — the account adds up.** `contact` 0.0 and `far` 0.0 exactly;
   `abort` = -1.0 x `force_abort_rate`; `success_lump` > 0 iff success > 0;
   `success_depth_invariant_violations` 0.
4. **P4 — the two NEW tables exist and are populated.**
   `demo_metrics.json` carries `success_by_lateral_bin` (edges 2/4/6/8 mm)
   and `success_by_yaw_bin` (edges 2/4/6/8 deg), and in each the bins up to
   the drawn maximum have `episodes` > 0. **A table that is absent, or whose
   bins are all empty, is a FAIL of this run's whole purpose** — it means the
   training PC did not pull `bdacbb3`.
5. **P5 — THE BRANCH, read at the LAST block:**
   * Every populated lateral bin >= 0.90 AND every populated yaw bin >= 0.90
     -> the reward carries both axes across their whole draw, not only near
     the centre. The stage is then measured and D-174's acceptance run is the
     next thing to fire, not another rung.
   * Any populated bin <= 0.50 while `success_rate_recent` >= 0.95 -> the
     overall rate was hiding a corner. That is the finding D-173 was built to
     produce; the next run is a rung on THAT axis, not a new one.
   * Between the lines -> report the numbers and the trend; do not conclude.
6. **P6 — force and reach, against RT-175.** `force_norm_n` p95 <= 27 N.
   `over_f_max` and `force_abort_rate`: RT-175 read 1 and 0.0005, the first
   of the series. **Named prediction: with 300 more iterations these fall
   back to 0.** If they do not, the abort is a property of the yaw axis and
   not a transient, and that is a finding.
   `stage1_lateral_y_mm.over_reach`: RT-175 read 37 against RT-174's 2. It is
   reported next to the rate, not interpreted.
7. **P7 — the run finishes learning.** Episode Reward flattens; Value Loss
   falls below RT-175's 1436; Entropy and sigma stop rising. Sigma stays
   above 0.30 at the LAST block. If Reward is still climbing at 1950, the run
   needed more budget and says so.

## What this run cannot answer

* **Nothing about seeds.** One seed, seed 42. D-174 says a probe run carries
  no thesis number, and this is a probe run.
* **Nothing about a start height above +50 mm.** The band is unchanged, and
  the height table's top bin is open, so it cannot resolve +30..+50 mm.
* **Nothing about RT-174's or RT-175's own missing tables.** Those two runs
  stay permanently unresolved on their own axes; this run measures the
  configuration, not those runs.
