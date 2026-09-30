# RT-177s42 — expectation, written BEFORE the run (2026-09-08)

Written on the dev laptop. `/rt-check` judges the log against THIS file and
nothing else.

This is the **first acceptance run of the project** under D-174: a fixed
budget, **no hand stop**, and a pre-registered branch. It covers **seed 42
only** — the one seed that D-174 point 5 requires to start from random
weights, without `--resume`. Seeds 1 and 2 run later and get their own
expectation. **The stage is closed only when all three have run.**

Log name for `rt_log.ps1`: `RT-177s42`.

## The one hypothesis

**"The reward solves the full draw from zero. The hand-built ladder
RT-156 → RT-172 → RT-173 → RT-174 → RT-175 → RT-176 does not carry."**

D-174 point 5 assumes the opposite outcome is just as informative: a
failure from zero **is a finding**. It would make the curriculum
load-bearing rather than decorative — something the thesis then has to
justify instead of merely describing.

## Why this run

The whole chain reached the current stage by hand-laddering: six runs, each
resuming the previous checkpoint, each adding one axis. Nobody has ever
asked whether that was necessary. RT-177s42 asks it, and it is the cheapest
possible way to ask: the environment is untouched, so the only difference
against the chain is the ladder itself.

**Seed 42 is deliberately the from-zero seed.** The chain also ran on 42.
So from-zero-42 against chain-42 differs in exactly one thing: the resume.

### What the two precedents actually say — neither settles it

* **RT-156 points toward "yes".** Fresh, no `--resume`, OSC, seed 42,
  1024 envs: 0.997 by iteration 282. **But** it ran the *easy* task — no
  tilt, no lateral offset, no yaw, and a start band of −30..+30 mm. It is
  not this task.
* **RT-140 points toward "no".** Fresh, success **0.0000** at iteration
  1499, depth 7.487 mm = the floor. **But** it ran under **joint PD**, not
  OSC, with the **contact penalty ON** (git `f6625aa`). Different
  controller *and* different reward. It proves a from-zero run *can* fail;
  it is **not** evidence about this configuration.

No run from zero at the full draw has ever been measured. That is the gap
this run fills.

## The one change

**None to the environment.** Every `env.*` override is RT-176's line,
character for character. Removed are only `--resume`, `--load_run` and
`--checkpoint`.

Code: whatever `origin/p4-reward` carries on the night of the run. The SHA
is deliberately **not** pinned here — see the command block below for why.
What is pinned is the condition: `git diff 864cdf1 HEAD -- source/ scripts/`
must be **empty**, i.e. bit for bit the code RT-176 ran on. Verified on the
laptop for every SHA from `864cdf1` up to this file's own commit; every
commit in between touches documents only.

## The budget is a TIME CHOICE, not a derivation

`--max_iterations 1200`. This number is **not derived from anything.** No
run from zero at the full draw exists, so there is no measurement to size
it from. It is chosen to fit one night:

1200 iterations × ~30 s ≈ **10 h**. The 30 s is measured, not guessed —
the `wall_time` column of all six committed scalars CSVs reads 27.1 to
30.6 s per iteration (RT-156 27.1 · RT-172 30.2 · RT-173 30.6 · RT-174
30.4 · RT-175 30.4 · RT-176 30.0), and the run logs quote the same
("Iteration time: 28.81s" RT-175, "31.35s" RT-176).

That is all the number says. It is **not** a claim that 1200 is enough.

`--max_iterations` is **absolute** here. Additivity is a property of
`--resume`, and there is no resume in this run.

**No hand stop** (D-174 point 3). The run ends when the budget ends.

## Command (training PC, PowerShell, conda env `env_isaaclab`)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

**Do not compare the SHA to a string.** Every documentation commit moves
it, this file included, so a pinned SHA is stale the moment it is written.
The binding condition is functional — run it and expect **no output**:

```
git diff 864cdf1 HEAD -- source/ scripts/
```

Empty means HEAD carries bit for bit the code RT-176 ran on, whatever the
SHA happens to be. At the time of writing that was `7efa8ea`. The log
header will show whatever HEAD then is, and that is **not** a mismatch
against RT-176's `864cdf1`.

```
.\scripts\rt_log.ps1 RT-177s42 python scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max_iterations 1200 env.fixture_yaw_noise_rad=0.1047 env.fixture_tilt_noise_rad=0.1396 env.start_lateral_offset=0.006 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.osc_pos_clamp_m=0.07 env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0
```

## PASS / FAIL lines, in order

1. **P1 — the reset is contact-free.** First block `force_abort_rate`
   0.0000; startup report step 2 `|prim-buffer|` below 1e-5 m, max joint
   deviation 0.000000 rad; the "entrance buffer" line shows a NON-ZERO yaw
   in both report envs; `start_pose_solve_unconverged_resets` 0.

2. **P2 — the run started from zero, and the log must PROVE it.**
   * **No** `[INFO]: Loading model checkpoint from:` line anywhere in the
     log. **A checkpoint line means the run is invalid and must be
     repeated** — it would be a resumed run wearing this run's name.
   * First iteration line reads `Learning iteration 0/1200`. *(Derived,
     not quoted: `train.py:398` passes `num_learning_iterations =
     max_iterations` and a fresh runner starts at 0. No previous log in
     this repo carries the string, so this is a prediction.)*
   * Folder tag carries `tilt0-8deg`, `yaw+-6deg`, `lat6mm`,
     `start+30..+50mm`, `seed42`.

3. **P3 — the account adds up.** `contact` 0.0 and `far` 0.0 exactly;
   `abort` = −1.0 × `force_abort_rate`; `success_lump` > 0 iff success > 0;
   `success_depth_invariant_violations` 0.

4. **P4 — all three bin tables are populated.** `success_by_tilt_bin`,
   `success_by_lateral_bin` (edges 2/4/6/8 mm) and `success_by_yaw_bin`
   (edges 2/4/6/8 deg). The code carries D-173, so **a missing table means
   the training PC did not pull the current code** and the run is
   worthless.

5. **P5 — THE BRANCH, read at the LAST block.** Read
   `success_rate_recent`. **These lines are BRANCH TRIGGERS for this run,
   not acceptance thresholds.** The project has no decided success
   threshold, and this file does not create one.
   * **≥ 0.95 → branch 1: the ladder does NOT carry.** The reward solves
     the full draw from zero. Next: seeds 1 and 2, same budget.
   * **≤ 0.10 → branch 2: the ladder DOES carry.** The curriculum becomes
     a finding the thesis must justify, not a convenience. Seeds 1 and 2
     still run — the finding needs n = 3 exactly as much.
   * **between → branch 3:** report the curve shape and conclude nothing.
     The six metrics carry that reading (`CLAUDE.md` §
     `Training & Log-Analyse`), and a reward still climbing at the end
     means too little budget, not a weak reward.

6. **P6 — force, against RT-176.** Final window `force_abort_rate` 0.0,
   `over_f_max` 0, `force_norm_n` p95 ≤ 27 N, max under the 50 N line
   (D-169).
   **Named prediction, from RT-176's trace:** aborts cluster early and
   then fall to exactly 0.0000. RT-176 carried an abort in 49 of its 300
   iteration rows, all in the first third, the last at iteration 1766.
   **Aborts in the last fifth contradict the user's thesis of 2026-09-07
   and are a finding.**

7. **P7 — STRICKEN 2026-09-08, AFTER the run, by a general rule change.**
   Recorded as struck rather than deleted, because it was removed after
   the user reported a result and a silently deleted point is
   indistinguishable from one tuned to fit.

   Original text: *"did the run finish learning? Pull the scalars CSV.
   Expect: Episode Reward flattens, Entropy and sigma stop rising, sigma
   above 0.30 at the last block, Value Loss falling."*

   **Why it goes:** it duplicated a repo-level rule into a run-level file.
   `CLAUDE.md` § `Training & Log-Analyse` already demands all six metrics
   with their trend for *every* log analysis, and already names dying
   entropy and a diverging value loss as red flags to report unasked. P7
   added no obligation; it only added a gate that no log could ever close
   on its own — the scalars CSV always arrives later, so P7 turned four
   runs in a row (RT-174, RT-175, RT-176, RT-177) into UNKLAR for a reason
   that was never a reward defect.

   **Nothing is lost.** The six metrics are still reported, still with
   trend, still before any verdict — they just no longer carry a PASS/FAIL
   mark. Saturation stays a *reading*, not a gate.

   **The judgement this run still owes:** if Episode Reward was still
   climbing at the hand stop, say so plainly. That is not a FAIL of the
   reward and must never be reported as one — the answer is more budget,
   not a different reward.

   This applies from RT-178 onward: **no expectation file gets a P7.**

## What this run cannot answer

* **Nothing about the acceptance run as a whole.** One of three seeds.
  D-174 point 6 wants all three values plus the spread.
* **Nothing that separates "reward too weak" from "budget too small"** in
  branch 3. The budget is unmeasured by construction — see above.
* **Nothing about the narrow fixture** (D-087, stage 2). That scene does
  not exist yet; the plan is `docs/Umbau/Umbauplan_Enge_Aufnahme.md` on
  `p1-konzept-messung`.
* **Nothing about start heights above +50 mm.** The band is unchanged and
  the height table's top bin is open at the top.
* **The asymmetry inside D-174's own seed set.** Point 5 puts one seed on
  from-zero and leaves the other two open. If seeds 1 and 2 resume from a
  seed-42 checkpoint, the three are **not three samples of one procedure**
  — they are one from-zero run plus two continuations of a shared history.
  Recorded here, **not solved**: the rl-code stream decides this before
  seeds 1 and 2 run.
