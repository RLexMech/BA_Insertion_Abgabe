# HANDOFF — Stream rl-code (only this stream writes here)

> # !!! STOP — READ THIS BEFORE ANYTHING ELSE !!!
>
> ## Eingang vom Konzept-Strang, 2026-09-06 — THE ACTION SPACE CHANGED
>
> **Everything below this box that says "six joint deltas", "PD position
> control", "no success termination" or "60 Hz" is STALE.** Written by the
> concept stream (`p1-konzept-messung`, commit `355e493`). This box is
> INPUT, not a decision, and it changes nothing else in this file.
>
> - **D-108 (1) FELL on 2026-09-03.** The action is a 6-D pose delta under
>   task-space impedance (Isaac Lab `OperationalSpaceController`), not six
>   joint deltas under PD position control. Gravity stays ON; the PhysX
>   drives go inert (stiffness and damping 0), which breaks the LETTER of
>   D-105 for those two values.
> - **The switch IS numbered since 2026-09-10: D-177.** It was the inbox
>   candidate "Audit 2026-09-03 (a)" with six decision branches, which
>   stays its source. **The supervisor question of branch (2) is still
>   UNANSWERED**, so the drive-value half is not fixed in the concept —
>   the code runs OSC as its default regardless.
> - **D-108 IS marked since 2026-09-10** (this branch): the entry carries
>   an inline correction "DECISION (1) OVERTURNED 2026-09-03" naming the
>   OSC cfg, the inert drives and the D-105 letter break, and pointing
>   here and at the `p4-konzept` inbox entry. It still has NO D-number.
> - **Success TERMINATES the episode** since 2026-09-01, with an
>   equivalent-return payout (inbox "Reward-Ueberarbeitung" (7)). It is an
>   own construction with no published precedent.
> - **The force abort is `truncated` and BOOTSTRAPPED** since 2026-09-04
>   (D-164). Under OSC it is no longer the brake against ramming (D-167).
> - **The policy rate is 15 Hz.** D-024's 60 Hz is the controller loop only.
>   The 256-step episode therefore lasts four times as long, and its
>   derivation is OWED.
> - **Unchanged:** the 28 observation channels, the success criteria, the
>   geometry, the stack and the RL library.
>
> **Today's state lives in two sheets** on `origin/p1-konzept-messung`
> (`355e493`):
>
> - `docs/Blockberichte/Gesamtbericht_Konzeptphase_Codestrang.pdf` — ONLY
>   the current concept, no history. **Read this one to build.**
> - `docs/Blockberichte/Gesamtbericht_Konzeptphase.pdf` — the path, with a
>   section on the controller switch and an overturn table.
> - `HANDOFF-KONZEPT.md` on that branch — open points and what comes next.
>
> Fetch them with `git fetch origin && git merge origin/p1-konzept-messung`.
>
> **Why this hits the rl-code stream:** the concept sheets now MATCH what
> you built — that was not true before today. Also recorded, and
> deliberately NOT weighed by the concept stream: `HANDOFF-RL.md` warns
> "the reward is under DIAGNOSIS" on RT-148b (0 % success), while the LATER
> run RT-156 of the same day reached 99.7 %. Both findings stand. **You own
> that resolution.**


Last updated: 2026-09-12, dev laptop, branch `p5-robustheit` (worktree
`Thesis Workflow Phase 5/p5-robustheit`, branched from `p5-reward` at `5b58d14`).
Read the branch head with `git ls-remote origin p5-robustheit`, never from
this file.

**NOTHING IN THE PHASE-5 BUILD HAS EVER RUN UNDER ISAAC.** Every offline
check is green and every runtime claim is UNVERIFIED. The next thing that
reaches the training PC is `Laufplan_Phase5.md` step 3 — see § Open, NEXT.

The older `p4-reward` / `p5-osc-rot-gain` lines that stood here are history;
read a head with `git ls-remote origin <branch>`. The training PC clone
(`Phase3_Implementierung_v2`) tracks `p4-reward` and the two heads agree, so
a plain `git pull` there is correct. `p5-wrench-obs` is SUPERSEDED and
carries no work.

**This file is 686 lines against the 250-350 the handoff rule names.** The
overflow is old NEXT detail and Frozen prose, not this session's material;
moving it verbatim into `docs/archive/` is owed and has not been done.

Scope of this stream: the RL code for the UR5e insertion — observation,
action, reward, termination, curriculum (concept blocks 3–7). Scene work
belongs to stream szene-umbau; concept decisions to `p4-konzept`. Rules for
this stream: the worktree-local `CLAUDE.md` (gitignored by design,
`.gitignore:82`), present in `p4-reward`; copy it by hand into any new
worktree.

**THE LOUDEST WARNING:** the reward is under DIAGNOSIS, not repair.
RT-148b (4000 PPO iterations under osc, 32,000 episodes) ended at 0 %
success, and the controller is NOT the cause: RT-150 proved kp 100 seats
the part when the pose delta is at the OSC step limit. The active plan is
`Pläne/Plan-Merge.md`; touch no reward term, weight or observation channel
before RT-151 (lateral reward slice) has a verdict. Every run must state
mode, kp, rate AND episode seconds (RT-144/146/147 trap: a delta under osc
is a VELOCITY, and `--max-steps` counts control steps of a 17.07 s episode).

History: everything before session 17 lives in
`docs/archive/HANDOFF-RL_Archiv_2026-08-30.md`; the reward rework of
2026-09-01, the contact-penalty / distance-leash analysis (RT-139..RT-141)
and the RT-131 finding were moved verbatim to
`docs/archive/HANDOFF-RL_Archiv_2026-09-03.md`; the RT-143..RT-147 NEXT
narrative to `docs/archive/HANDOFF-RL_Archiv_2026-09-05.md`. Nothing there
is a next step; its commands must not be run again.

## Arbeitsplan

Two levels. Neither is re-derived here — each row points at its home.

**Level 1 — build phases P1–P7.** They own the gates. Home:
`docs/Blockberichte/Gesamtbericht_Konzeptphase_Codestrang.pdf`
§ "Umsetzung in Phasen". P1 belongs to stream szene-umbau. P2 needs the
training PC.

**Level 2 — code milestones M0–M2.** This file owns them.

- M0.4 module cut — DONE. M1 pure math — DONE.
- M2.0–M2.4b — DONE AND VERIFIED on Isaac (verdicts in `rt_logs/VERDICTS.md`).
- M2.5 curriculum, M2.6 report/metrics, M2.7 checks — NOT STARTED. Start
  height is a cfg field since 2026-08-31 (rung 0); the ladder itself is not.
- **The action space is REOPENED (2026-09-03):** D-108 fell; the controller
  is the OSC of the inbox entry "Audit 2026-09-03 (a)". The discriminating
  measurement (RT-144) is owed before any further PPO run.

**Level 3 — Phase 5, the robustness study.** Plan:
`C:\Users\PCUser\.claude\plans\fragen-in-chat-plan-parallel-creek.md`.
**Run order and run rules: `Laufplan_Phase5.md` (2026-09-11) — it replaces
plan § 4's stop bar and § 9.**
Branch `p5-robustheit` (from `p5-reward` @ `5b58d14`). Three phases; this
line says which one we are in.

- **Phase A — the AutoDR core, offline. DONE (2026-09-09).** `autodr.py`
  (11 boundaries, stdlib) + `scripts/check_autodr.py` (115 checks, 91
  mutations, both D-080 directions). Three critic rounds: 5.5 → 7.0 → 7.5
  against a bar of 8.5. The bar was NOT reached in the three rounds the
  plan allows, but both items the last round called blocking are fixed and
  re-measured. Commits `acdd7bf` → `15bb17b` → `ca698ac` → `b8eb749`.
  Laptop only — no simulator evidence for any of it.
- **Phase B — the reset path in the env. IN PROGRESS.** B1 `3679dec`,
  B2 `1d11778` + round-3 fixes `36db764`, B3 `a671aa6` + round-2 fixes
  `dbbad59` (D-176), B4 friction per reset `8261c84` + round-2 fixes
  `a596d35`, B5 the AutoDR loop `1020e73` + round-2 fixes `9404224`,
  B6 train.py under AutoDR, B7 the clamp box (frame AND value) and the
  unconverged-IK boundary flag `15e3d0f` + round-3 fixes. **B7 IS DONE**;
  the smoke run is next. `dr_table_path` / `dr_refill` belong to play.py
  (plan step 2) and `fixture_variant` needs the eng-USD (plan step 3);
  neither is owed before the smoke run, which is why B6 did not take them.
  Check counts after Commit A0 (D-184): env wiring 299 / 295 mutations,
  insertion math 254 / 92 mutations, autodr 115 / 91, all three
  counter-proofs green, plus `check_mutation_anchors` 401 anchors / 0 stale
  (measured 2026-09-11).
  Laptop only, no simulator.
  B7 round 3 scored LOGIK 6.5 / CHECKS **4.0 (6 of 15)** / EHRLICHKEIT 9.0
  (30 of 32 claims true) / SCOPE 9.0 -- and 14 of 15 after the fixes. Four
  of the nine misses were the shape round 2 had just repaired one line
  along: a pinned NAME, a pinned KEY, a pinned CALL, a one-sided `>=`.
  B6 round 1 scored LOGIK 6.5 / CHECKS **0.0 (0 of 15)** / EHRLICHKEIT 7.5
  (31 of 36 claims true) / SCOPE 9.0. All fourteen round-1 checks were
  substring pins, and every one of the critic's fifteen damage mutations
  kept the pinned text and changed its meaning -- the same failure B4
  round 1 had. Round 2 replaced them with sixteen STRUCTURAL checks (call
  arguments, the guard a statement lives under, called-vs-mentioned,
  statement order, branch selection, the denominator behind a pinned key),
  fixed seven real defects and corrected six false claims. Details in
  `docs/decisions_inbox.md`, entry "B6 round 2". Commit `13a73c6`.
  B7 IS COMPLETE, all three parts, plus a critic round and its round-2 fixes.
  Commits `179de98` (parts 1+3), `d532037`, `78806ab`
  (`scripts/check_mutation_anchors.py`), `c131e32` (part 2) and the round-2
  commit. NEXT is the smoke run.
  B7 round 1 scored LOGIK 7.0 / CHECKS 8.0 -- but only **10 of 15**
  semantically, two of the twelve "catches" were `check_mutation_anchors`
  tripping on its own anchor line -- / EHRLICHKEIT 6.5 (24 of 32 claims
  true) / SCOPE 8.5. Round 2 fixed a REAL frame defect (the z rule read a
  pocket-frame band against a world-aligned box; the world-z requirement is
  0.070694 m, so the shipped 0.070 box was 0.694 mm short of its own rule),
  added the missing UPPER bound (a metre/millimetre slip passed all 673
  checks), closed the four gaps the critic's own mutations walked through --
  all four in part 3, all four keeping every line round 1 had pinned -- and
  corrected eight false claims. `osc_pos_clamp_z_m` was **0.10 m** and was
  DECIDED, not derived (user, 2026-09-10, reaffirmed after the 0.0707
  alternative was put with its arithmetic): the start band is expected to
  rise toward the home pose and the number should not move twice.
  **Superseded 2026-09-11 by D-180: it is 0.1435 m, from the disk formula
  with the 0.120 m height ceiling — the clamp table above owns the number.** Details in
  `docs/decisions_inbox.md`, entry "B7 round 2".
  The general lesson, B4's and B6's in a third form: a structural check that
  pins the STATEMENT the author wrote does not pin the INVARIANT that
  statement exists to establish. Line numbers, key names and assignment
  targets are all statements.
  Part 1: `clamp_tilt_to_cone` now measures the 8.52 deg CAD limit from the
  POCKET AXIS, not the world vertical -- a frame correction, not a new
  number, and an upright reference reproduces the old body bit for bit
  (0.000e+00 over five orientations). Part 3: an unconverged start solve
  withdraws BOTH the boundary flag and the bounds stamp, per env, and the
  count is published as `dr/start_solve_flags_dropped`. Part 2, the clamp
  box VALUE, is the open one: the binding axis is z with margin 0.000000 m,
  and the margin has NO source -- Factory's own ratios are 2.0 / 0.88 /
  1.11. The proposal (0.070 m = `start_height.hi_max` + one action step) is
  OUR construction and is marked as such. Details in the inbox entry
  "B7 parts 1 and 3".
  B5 round 1 scored LOGIK 7.0 / CHECKS 7.3 (11 of 15) / EHRLICHKEIT 8.0
  (20 of 25) / SCOPE 8.5. Round 2 fixed the one real defect (per-element
  device syncs in the `record` loop), added five checks against the four
  mutations that walked past, corrected five false comment claims, and
  handed two NUMBER decisions back to the user (the clamp box, and the
  unconverged-IK question) -- both below. Details in
  `docs/decisions_inbox.md`, entry "B5 round 2".
- **B4 round 1 scored CHECKS 0/15 under the rubric and the round-2 commit is
  the answer.** The critic wrote fifteen damage mutations of its own, ran
  them, and NONE was caught. Cause, and it is the general lesson: every B4
  check was exact `ast.unparse` equality on an expression one of B4's own
  declared mutations edited, so the closed set defended itself and nothing
  outside it. Device, dtype, statement ORDER, call arguments, loop placement
  and the arithmetic BEHIND a pinned key name were all undefended. Fifteen
  checks were added against exactly those, one mutation each.
  Round-1 sub-scores: LOGIK 5.0, CHECKS 0.0, EHRLICHKEIT 7.3 (22/30 true),
  SCOPE 8.0.
- **B4, what it is and what it is not.** `_apply_friction` draws ONE
  friction per env per reset from the LIVE AutoDR band, static = kinetic,
  onto robot AND fixture, by editing two cached CPU material buffers in the
  resetting rows and pushing the WHOLE buffer plus the subset of ids
  (`envs/mdp/events.py:283`). Cost is measured around the whole block and
  reported as `dr/friction_write_ms_mean` / `_max` plus
  `demo_metrics.json` `friction_write_ms`, to be read against the iteration
  time of the SAME log — that number decides whether plan §6 keeps this path
  or falls back to the redraw-per-boundary-move alternative.
  ~~Friction is the ONLY one of the six quantities wired to the provider;
  lat_x, lat_y, yaw, tilt and start_height still read the static cfg
  fields, and the startup report says so.~~ **Superseded by B5 below: all
  six read the live band through `_live_bounds()`, and `_apply_friction`
  reads it through that seam too.**
  **NOTHING HERE HAS RUN IN A SIMULATOR.** The call form was read out of
  `C:\IsaacLab`, never executed. Evidence owed from the smoke run: the
  `friction after first reset env0..3` line (PhysX vs commanded, per env)
  and the `friction write cost mean/max ms` line.
  Correction made during B4: the claim "the per-episode redraw is FORGE's"
  was FALSE. FORGE randomises friction at `mode="startup"`
  (`forge_env_cfg.py:51-84`), one draw per env for the whole run, fixed
  asset only, and its dynamic friction is pinned at 0.25 while static gets
  the band — so static != kinetic there. The per-reset schedule and the
  tied coefficients are own construction; recorded in
  `docs/decisions_inbox.md` (2026-09-10) and `InBachelorErwähnen.md`.
- **B5 CLOSED THE LOOP, and it was three holes, not one.** The user chose
  option A on 2026-09-10: B5 takes all three rather than splitting them,
  because each one alone leaves a build that reads as a working AutoDR run.
  (a) `boundary_assignment` / `apply_boundary` had no caller, so no env was
  ever a boundary env. (b) FIVE of the six quantities never read the
  provider — `lat_x`, `lat_y`, `yaw`, `tilt`, `start_height` read the static
  cfg fields, which AutoDR mode refuses non-zero. Only friction (B4) read
  `bounds()`. (c) `record()` / `update()` were never called.
  Now: one seam `InsertionEnv._live_bounds()` answers for all six;
  `self._dr.bounds()` is called in exactly two places, the seam and the
  startup report (which only prints); the draw marks and nails boundary
  envs and stamps `bounds_version` per env; `_log_finished_episodes`
  records, updates, prints the `[autodr]` lines and merges
  `self._dr.scalars()` plus `dr/train_success_regular_fresh`,
  `dr/fresh_n`, `dr/fresh_min`, `dr/success_rate_boundary`,
  `dr/friction_mean` into `extras["log"]`; `demo_metrics.json` gains
  `dr_fresh` and `friction_applied`.
  **TWO LATENT DEFECTS WENT WITH IT**, both inside the code B5 was already
  rewriting; the commit message of `1020e73` names only the first, and its
  "five of the eleven boundaries" is wrong -- five of the six QUANTITIES is
  NINE of the eleven BOUNDARIES. (2) `_map_start_conditions` did
  `float(self.cfg.start_tip_above_entrance)` unconditionally, so the
  home-pose setting (`None`) raised `TypeError` at the first reset; the seam
  answers with `_start_tip_height` there. (1) Plan risk R1, one level down:
  the tilt quaternion was built behind `cfg.fixture_tilt_noise_rad > 0.0`, a field
  AutoDR mode refuses non-zero — the pocket would have yawed and never
  tilted for the whole run while `dr/tilt_hi` climbed. Fourth reach flag
  `_tilt_reach` replaces it. No run has ever used `dr_mode='autodr'`, so no
  result is affected.
  **`dr_mode='off'` KEEPS ITS DISTRIBUTION, NOT ITS TRAJECTORY. MEASURED,**
  2e6 float32 draws per row, 2026-09-10: `tilt` and `start_height` (band and
  width 0) are bit-identical, 0 of 2e6 differ. `lat_x`/`lat_y` and `yaw`
  moved from `(2u-1)*h` to `-h + u*2h` and differ in ABOUT HALF the draws
  (51-57 %, stream-dependent -- the maxima reproduce, the count does not),
  by at most 4.66e-10 m and 7.45e-09 rad. Physically nothing; what it
  costs is exact replay of a past run. The symmetric form was rejected on a
  measurement, and the first wording of this line was wrong: over the 616
  reachable `(lo, hi)` pairs in float32, `lo + u*(hi-lo)` misses `lo` in 0
  and `hi` in 151 (worst 2.98e-8); the symmetric form misses `lo` in 215 and
  `hi` in 132. Exact at ONE end always, not at both. A nailed upper boundary
  lands within 3e-10 m of a 6 mm edge.
  **NOTHING HAS RUN IN A SIMULATOR.** The startup report now stamps itself
  `NOT VERIFIED IN A SIMULATOR`, and that line is a check.
- ~~**BLOCKING BEFORE THE SMOKE RUN, and it is a NUMBER decision, not code:
  the controller's clamps do not contain the AutoDR reach.**~~ **CLOSED
  2026-09-11 by D-180 and commit `5a7d295`** — the user took the number
  decision, the boxes were resized for the DISK, and the containment check
  was rewritten to read `autodr` bounds instead of the static constants.
  The table below is the CURRENT state, recomputed 2026-09-12 from the code
  (`insertion_env_cfg` fields `osc_pos_clamp_m`, `osc_pos_clamp_z_m`,
  `osc_tilt_clamp_rad`; formula `_reach_xy` / `_reach_z` in
  `scripts/check_env_wiring.py`), NOT the 2026-09-10 defect table:

  | what | value | reach it must contain | margin |
  |---|---|---|---|
  | `osc_pos_clamp_m` (tip box half width around the entrance) | 0.080 m | disk reach `r + h*sin(tilt)` = 0.050838 m plus one action step 0.02 m = 0.070838 m | **+0.009162 m**, and the check now CAPS that air at `XY_CLAMP_AIR_MAX_M` = 0.010 m |
  | `osc_pos_clamp_z_m` (tip box half height) | 0.1435 m | `h*cos(tilt) + r*sin(tilt)` = 0.123386 m plus one step = 0.143386 m | **+0.000114 m**; above the box 0.002379 m are left to the measured home tip 0.145879 m |
  | `osc_tilt_clamp_rad` (tool cone, measured from THIS EPISODE'S POCKET AXIS since B7 part 1) | 8.5200 deg | `tilt.hi_max` 10.0000 deg | **-1.4800 deg**, ACCEPTED, not a defect: the start solver commands position only, so at full pocket tilt the tool starts outside the cone and the cone pulls the first target in (D-178 (6)). Expected in the log: a small force peak at step 1 on `tilt_hi` boundary envs |

  `r` = `lat_r.hi_max` 0.030 m, `h` = `start_height.hi_max` 0.120 m,
  `tilt` = `tilt.hi_max` 10 deg, all from `autodr.DR_DIMS`. The yaw drops
  out of the sideways reach because the DISK is rotationally symmetric
  about the pocket axis (D-178, warrant corrected 2026-09-12).
  **Both box numbers are DECIDED, not derived** (D-180): the air is set, and
  what the check defends is the upper bound on it. `Belege_Streuwerte.md`
  § C owns the copy of these numbers.
  **UNVERIFIED:** none of this has run under Isaac. What the old entry
  described (a boundary env starting the tip EXACTLY on the box face at
  `start_height_hi`, so the OSC target is clamped in +z from step 1) was
  measured on the OLD 0.05 m box and is history, kept here only so nobody
  re-derives it.
- **OPEN, needs a written decision before the boundaries carry weight: may
  an UNCONVERGED start-pose solve book a boundary flag?** `_solve_start_pose`
  counts non-convergence in a GLOBAL `_start_solve_unconverged` and prints
  "does NOT start where the config says" once per run — there is no per-env
  residual, so a boundary env whose IK missed still records on its boundary.
  Before B5 that polluted a metric; after B5 it MOVES the distribution, and
  the low `start_height` edge (toward 0.020 m) is where the solver has least
  room. Either mask `record` with a per-env residual, or state in the thesis
  that the buffers measure COMMANDED conditions.
- **EXPECT `dr/train_success_regular_fresh` TO READ -1.0 FOR MOST OF THE
  RUN** (arithmetic from plan § 1, not measured): at 1024 envs the fresh
  window needs about 62 move-free iterations to reach 2000, while a buffer
  fills roughly every 7.5 iterations across the eleven boundaries. Every
  real move clears the window. That is consistent with the stop rule, which
  needs `all_at_max` anyway, but the curve will saw-tooth and read -1.0
  until the boundaries settle. Note also that at 1024 envs
  `maxlen == window == _fresh_min == 2000`, so "minimum" IS "completely
  full"; at 4096 envs they diverge (5120 vs 2000) and the reported rate then
  averages up to 5120 episodes while still gated at 2000. The key's meaning
  moves with `num_envs`.
- **NOT WIRED AND NOT OWNED BY ANY B-STEP YET — the same question as B5's,
  one file over. `train.py` has nothing of plan § 10's train.py line.**
  Measured 2026-09-10: `grep dr_mode scripts/rsl_rl/train.py` returns
  NOTHING. So (a) no `autodr` / `fix-eng` run tag — two runs with different
  `dr_mode` land in folder names that cannot be told apart (plan risk R5);
  (b) no `--stop-when-dr-max`, so the § 4 stop rule has no reader even
  though the env now publishes `dr/all_at_max`,
  `dr/train_success_regular_fresh` and `dr/fresh_n`; (c) no
  `autodr_<it>.json` beside each checkpoint and no `--resume` load, so an
  interrupted AutoDR run silently restarts at width 0 (plan risk R13).
  Plan § 10's own order puts train.py BEFORE the smoke run. Say whether B6
  takes it beside the CFG switches, or whether it becomes B7.
- **Phase B, the original scope line.** Split drawing the reset
  values from applying them; fix `_tilt_rot`/`_start_lat_off` so a run can
  start at width 0 and open later; friction per reset; DR metrics merged
  into `extras["log"]` instead of replacing it; the CFG switches. Ends in a
  smoke run on the training PC.
- **Phase C — the evaluation chain.** Tables, `play.py` eval with refill,
  statistics, `results/`, plots, the fixture switch, DORAEMON.

Open from the Phase-A critic rounds, none of it blocking Phase B:
`GridTable` validates none of the plan-§8 structure its docstring asserts
(those invariants arrive with `make_grid_table.py` in Phase C);
`check_autodr.py` does not use the `scripts/tools/checks.py` harness that
plan §10 names, following its `check_curriculum_ladder.py` sibling instead;
`record()` is a per-episode scalar call, so the env loops in Python over
FINISHED envs (≈2 calls per step at the plan's own rate) — worth one line
in the env-wiring note, not a batched API.

**The critic score is a rubric since 2026-09-09, not one opinion number.**
The old prompt asked for a single 0-10 against a bar of 8.5. Calibration:
that same prompt scores `C:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\events.py`
— shipped upstream Isaac Lab, not our code — at **4.1**, BELOW every score it
ever gave us (Phase A 5.5/7.0/7.5, B2 5.5/5.5). The number measured how much
an adversarial reader can list, not quality, so it could never be compared
against a bar. Replaced by four sub-scores, three of them counts: LOGIC
(defects found), CHECKS (`10 * caught / 15` damage mutations the critic
applies and runs), HONESTY (`10 * true / checked` behaviour claims in the
comments), SCOPE. The prompt is not a repo file; it is rebuilt per round.

Phase-B step B2 round 3 under that rubric: **L 9.0 / C 8.7 / H 8.0 / S 6.0,
TOTAL 7.9**, no blocking defect. All five findings were re-read in the source
and fixed in the follow-up commit: three comments that claimed more than the
code delivers (the `never from the static fields` line in the cfg, the
`SIX BOUNDARY-OWNED` scope of the autodr refusal, `provenance_lines`'
"no AutoDR run can start without"); `demo_metrics.json` key `dr_bounds` held
the whole provider state and wrote `bounds_version` a second time, now one
key `dr_state`; and the two mutations that SURVIVED (a `dr_mode` refusal
naming one mode instead of excluding the known ones, and the `start_height`
centre read off `_low`) are each pinned by a check plus its own mutation.
`check_env_wiring.py` is 182 checks, counter-proof green. Laptop only.

Open after Phase-B step B2 (`bounds_max` + `dr_mode`, three critic rounds).
NONE of these is due at B2 — B2 wires the measurement FRAME, not the
values — but each goes wrong the moment the values land, so they are owed
before the smoke run:

- ~~**The tip clamp box does not know the start-height boundary.**~~
  **CLOSED 2026-09-11 by D-180 / `5a7d295`:** the box block now reads
  `_autodr_reach` (`check_env_wiring.py`, `_reach_xy` / `_reach_z`), i.e.
  the BOUNDARIES,
  not `RUNG0_START_TIP_ABOVE_ENTRANCE`. Numbers in the clamp table above.
  (What it said: the check tested `osc_pos_clamp_m` 0.05 against the static
  0.030 while `start_height.hi_max` was already 0.050 — same class of defect
  as B2's `_tilt_rot`.)
- **The D-153 hole sum is still unimplemented** (plan §7): `lat_y.hi_max`
  (0.006) + `fixture_pos_noise_xy` (0.005) = 11.0 mm against
  `HOLE_REACH_OFFSET_Y` = 7.8847 mm. With today's `DR_DIMS` it would fail.
  `validate_rl_config` is its home.
- `_start_height_bin_edges_m` tops out at 0.030 while the boundary reaches
  0.050; the top bin becomes an unbounded catch-all.
- `compare_runs.py:342` groups fair comparisons on
  `(clearance_mm, yaw_deg, noise_rad)`. `dr_mode` is recorded in
  `demo_metrics.json` since B2 but is not in that key, so an AutoDR run and
  a No-DR run still land in one head-to-head group.
- `play.py:244-306` zeroes the four static angle fields for its probe modes
  and never touches `dr_mode`. Inert at width 0, a trap in Phase C.
- `check_env_wiring.py` has NO coverage gate (`check_autodr.py` does: "every
  check is defended by some mutation"). Adding one touches all 182 checks,
  most of them older than B2 — its own change, not B2's.

## Frozen (settled — do not re-litigate; content lives at the target)

- **D-number rule:** before taking a D-number, read the highest one on
  `main` (`git show main:DECISIONS.md`), not the highest on this branch.
- **The controller switch is DECIDED, not pending the measurement**
  (`docs/decisions_inbox.md` on `p4-konzept`, entry "Audit 2026-09-03 (a)",
  Decision (1)–(6), grilled, candidate). The measurement is the proof.
  Do not re-argue joint deltas; do not use `variable_kp`. `joint_pd` exists
  ONLY as the PD half of RT-144.
- **Gravity stays ON (D-105 core).** Inert drives under OSC break the
  LETTER of D-105 for stiffness and damping — that is the OPEN supervisor
  question of Decision (2). D-105 itself is NOT edited by this stream.
- **Tilt cone 8.52 deg from CAD** (`Geometrie_Fuegeteil_Aufnahme.tex:264`),
  Decision (3); yaw free (D-106). Not a placeholder.
- **The five numbers** (`a_fine`, `interpen_thresh`, `engaged_depth_m`,
  `abort_payment` form, success band): concept commit `77f5b3c`, landed in
  `f1051b3`. Cite, do not re-argue.
- **D-152** SDF on-surface tolerance `3e-5` m is DERIVED. **D-153** the
  fixture keeps its 1.5 mm hole. **D-154** 64,000 SDF samples. **D-155** P4
  judges the p99. **D-156** workcell block OFF (D-125 parked). **D-157**
  lateral gate in the paying predicates. **D-158** force-abort limit is
  user-set, in `RL_PLACEHOLDERS`. **D-161** the reward is zero at the home
  pose (why the first PPO run learned nothing). **D-164** the force abort
  IS `truncated` (decided 2026-09-02, BUILT — `insertion_math.py:1038`,
  `truncated = (timeout | force_abort) & (~success_now)`; the earlier
  "code NOT built" in this file was wrong, corrected 2026-09-04).
- **Never shape production torch code around the offline stand-in** (user,
  2026-08-31): the SHIM grows the operation, not the math module.
- Inbox "The SAPU port needs no second Warp kernel"; "The quaternion is
  converted at the Warp boundary"; "H1 is answered…" (`action_scale = 0.02`
  confirmed by RT-84 — for the joint_pd chain only); "The Warp meshes come
  from the USD as OBJ…"; "The lug/notch pair does not bound the lateral
  play"; "SAPU scales the task return only"; "The force-sensor link is
  `tool_link`, measured" (RT-59); "The first PPO run trains WITHOUT the
  ladder"; "The tilted scripted insertion is SPENT …" (its F_max and
  capture-depth rules are withdrawn; the instrument itself is reused for
  RT-144 as a controller comparison, not for those rules).
- D-066 one handoff per stream. D-042 proxy-settled choices get a short
  adoption entry. D-080 / D-081 named checks with counter-proof mutations;
  exit code set BEFORE `simulation_app.close()`.
- `docs/reference/literature_check_isaac_coding_rules_2026-08-24.md` —
  DirectRLEnv step contract, lifecycle rules, dependency pins.
- `docs/reference/literature_check_contact_force_sliding_search_2026-09-03.md`
  — the published search-phase force band is 1–20 N; the abort limit is 50 N
  since D-169 (2026-09-06), so still outside it but no longer by an order of
  magnitude. The band is the SEARCH force, the limit is the JOINING force;
  which of the two numbers is OUR reference is not decided.

**The training PC's repo root** (so no command ever ships with a `<path>`
placeholder). NEW folder since 2026-09-03 (user); `Phase3_Implementierung_v1`
(branch `p3-rl-code`) is a read-only archive once the editable install points
here (`pip show insertion`). The clone tracks `p4-reward` — RT-157 ran off
`a840b91` and `b95593e`, both heads of that branch, so the `p4-rl-code` this
paragraph used to name is stale:

    C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2

The asset folder the env resolves (`insertion_tasks_cfg.py:435`):

    C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2\source\insertion\insertion\tasks\direct\insertion\assets\Werkzeug

Sparse-checkout set of that clone (its only home): `.claude rt_logs scripts
source`. `insertion` is an editable pip install bound to that folder
(`pip show insertion`, `Editable project location` is the truth). The USDs
and OBJs are in NO git tree — copy the `assets` tree, never re-run the STEP
pipeline. Every `rt_log.ps1` line handed to the user is COMPLETE and
paste-ready.

**THE `p5-robustheit` CLONE (user, 2026-09-10). Its own folder, NOT a branch
switch of `_v2`:**

    C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1

Same sparse-checkout set (`.claude rt_logs scripts source`), same editable
install rule, and the `assets` tree is COPIED from `_v2`, never rebuilt (0
asset files are in git -- `git ls-files .../assets` returns nothing). Handed
over 2026-09-10 for the RT-178 smoke run at
`3a65fb1188f0ec574b07a3fb9cb7910b1e709fbc`. THE INSTALL SWITCH IS THE ONE
IRREVERSIBLE-LOOKING STEP: only one editable `insertion` install can exist, so
`pip install -e source\insertion` here takes it away from `_v2` until it is
moved back. The blocker that rule was written for (RT-152, the yaw sweep) is
SKIPPED by user decision 2026-09-06; RT-177 has no entry in `VERDICTS.md`, so
whether `_v2` still owes a run is NOT established here.

**A THIRD clone is planned, not yet made.** It belongs to `p5-reward`
(user, 2026-09-06; this REPLACES the earlier plan that gave `_v3` to
`p5-osc-rot-gain`):

    C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v3

Same sparse-checkout set (`.claude rt_logs scripts source`), same editable
install rule, and the `assets` tree is COPIED from `_v2`, never rebuilt.

Branch `p5-reward` was created 2026-09-06 from `p4-reward` at `87fa4e0` and
pushed; `git ls-remote origin refs/heads/p5-reward` read
`87fa4e0988eff552c2f6db80de1e94259eb39207` back. Its purpose is the ALIGNMENT
TERM -- the potential-based tilt shaping proposed after RT-170 BEFUND 2. The
proposal, with its derivation and its three-trajectory accounting, is the plan
file `weiter-aus-dem-vorigen-squishy-bird.md` (laptop only,
`C:\Users\PCUser\.claude\plans\`, NOT in git); it is reviewed with Fable 5.1
before any code is written. The laptop worktree for the branch exists since
2026-09-06: `C:\Users\PCUser\Desktop\Claude\Bachelor\Thesis Workflow Phase 5\p5-reward`.
The branch carries the RT-171 build since the same evening (the SHA is the
one read back from `git ls-remote origin refs/heads/p5-reward` in the
handover); `p4-reward`'s documentation commits up to `3612660` are merged
into it, so the two handoffs agree up to that point.

`p5-osc-rot-gain` keeps its branch on origin and carries no work: RT-168
closed `osc_kp_rot` as a suspect (§ Open NEXT point 1), so the clone that
paragraph asked for is not needed. The RT-158 tilt medians it used to restate
live in `rt_logs/VERDICTS.md`, 2026-09-06.

WHY it needs its own branch and its own clone: a reward change retrains from
scratch, so `_v2` must stay runnable. Only ONE editable `insertion` install
can exist, so `_v3` must not be installed before the `_v2` measurements
(RT-152, the yaw sweep) are done -- that is how `_v1` became an archive.

**`p5-wrench-obs` was DELETED 2026-09-06** (user's decision; `git push
origin --delete p5-wrench-obs` and `git branch -D p5-wrench-obs` run on the
laptop, `git ls-remote origin refs/heads/p5-wrench-obs` reads nothing back).
Its last commit `69cc504` is still reachable by SHA, so the delete is
reversible as long as this line stands -- do not delete this line. It carried
no work. Its one purpose was `OBS_SLICES` 28 -> 31, the three moment channels
(`insertion_math.py:762`), motivated by RT-157 BEFUND 4: the three FORCE
channels do not encode WHERE the part is caught (rim pose 15.9 mm off in -y
reads -0.49 N, a clean insertion reads -0.41 N). That change is DEFERRED, not
dropped, for two reasons read off the code. First, the tilt information is
already in the observation: `ee_quat` 15:19 is a full quaternion, and with
`fixture_tilt_rad = 0.0` (`insertion_env_cfg.py:554`) and every reset spread
at 0 (:325, :326) "rotation since reset" IS "tilt against the pocket" -- a
moment channel adds no new information, only another form. Second, the
controller suspect above is cheaper to test and comes first. If the channels
do come back, the better channel is probably the TILT DIRECTION IN THE POCKET
FRAME, two numbers, after the `yaw_cos_sin` pattern (`insertion_math.py:788`)
-- not the moment. Reversing D-114 needs its own `/decision`; none is written.

## RT ledger

- **Next free RT number: RT-178.** **RT-177 is REGISTERED and runs
  tonight (2026-09-08):** the **first acceptance run of the project** under
  D-174 -- fixed budget, no hand stop, branch written down first.
  Expectation `rt_logs/RT-177_expectation.md`, log name `RT-177s42`. The
  code SHA is deliberately NOT pinned (every doc commit moves it); the
  binding condition is that `git diff 864cdf1 HEAD -- source/ scripts/`
  prints nothing. It covers **seed 42 only**, the seed D-174 point 5 puts on
  random weights: no `--resume`, no `--load_run`, no `--checkpoint`, and
  every `env.*` is RT-176's line character for character
  (`git diff 864cdf1 HEAD -- source/ scripts/` is empty, so it is
  bit for bit the code RT-176 ran on). **Hypothesis:** the reward solves
  the full draw from zero and the hand-built ladder RT-156 -> RT-172 ->
  RT-173 -> RT-174 -> RT-175 -> RT-176 does not carry. **The opposite
  outcome is equally a finding** and makes the curriculum load-bearing
  rather than decorative. Branch at the last block, pre-registered as a
  branch TRIGGER and not an acceptance threshold (the project has no
  decided success threshold): `success_rate_recent` >= 0.95 -> ladder does
  not carry, run seeds 1 and 2; <= 0.10 -> ladder carries, still run seeds
  1 and 2 because the finding needs n = 3; between -> report the curve,
  conclude nothing. **Budget `--max_iterations 1200` is a TIME choice, not
  a derivation** -- no run from zero at the full draw has ever been
  measured; 1200 x ~30 s ~= 10 h = one night, and the 30 s is measured off
  the `wall_time` column of all six committed scalars CSVs (27.1 to
  30.6 s/iteration). It is NOT a claim that 1200 suffices; a reward still
  climbing at 1199 means too little budget, not a weak reward.
  `--max_iterations` is ABSOLUTE here, additivity belongs to `--resume`.
  **Two precedents, neither settling it:** RT-156 (fresh, OSC, seed 42)
  reached 0.997 by iteration 282 but on the EASY task -- no tilt, no
  lateral, no yaw, band -30..+30 mm; RT-140 (fresh) read success 0.0000 at
  iteration 1499 but under JOINT PD with the contact penalty ON, so a
  different controller AND a different reward. **Named limit:** D-174's own
  seed set is asymmetric -- point 5 puts one seed on from-zero and leaves
  the other two open, so if seeds 1 and 2 resume from a seed-42 checkpoint
  the three are not three samples of one procedure. Recorded, not solved;
  the rl-code stream decides it before seeds 1 and 2 run.
- **RT-176 is CLOSED (ran 2026-09-07 15:42:56, git `864cdf1`, folder
  `09-07_15-43-06_offset5mm_tilt0-8deg_yaw+-6deg_currOFF_start+30..+50mm_lat6mm_seed42`,
  judged 2026-09-07, `VERDICTS.md` four lines): PASS, P5 branch 1.** The
  frozen RT-175 configuration on newer code -- **no `env.*` argument
  changed at all**, the change was the INSTRUMENT: `b95fb77` (D-173)
  brought `success_by_lateral_bin` and `success_by_yaw_bin`, and RT-176 is
  the first run that could fill them. `success_rate_recent` 0.9975, and no
  occupied bin of any of the three tables falls below 0.99 (lowest 0.9924,
  yaw 4-6 deg) -- the overall rate hides no corner, which is what D-173 was
  built to test. RT-175's single force abort is gone (`over_f_max` 0,
  `force_abort_rate` 0.0, last abort at iteration 1766 then 183 iterations
  at exactly zero), and `over_reach` falls back to 6 from 37. **The only
  run of the series to spend its full budget** (1650..1949, exit 0);
  Episode Reward flattened (+50.5 over the first 100 iterations, +6.4 over
  the last 100), Value Loss 1053.7 under RT-175's 1436, Entropy and sigma
  both turned over. Documents: `docs/runs/RT-176.md`, report section 9,
  and the new `docs/Blockberichte/Bericht_Laufkatalog.tex`.
  **PERMANENT GAP, not repairable:** RT-174 has no lateral table and
  RT-175 no yaw table -- both ran before D-173, and RT-176 measures the
  CONFIGURATION, not those runs.
- **RT-174 is CLOSED (ran 2026-09-07
  01:27:10, git `35d91b6`, folder
  `09-07_01-27-19_offset5mm_tilt0-8deg_currOFF_start+30..+50mm_lat6mm_seed42`,
  judged 2026-09-07, `VERDICTS.md` five lines):** the overnight lateral run
  (`env.start_lateral_offset=0.006` on tilt 0..8 deg, resume RT-172's
  `model_600.pt`), expectation `rt_logs/RT-174_expectation.md`. Verdict
  **UNKLAR, but P5 is decided: BRANCH 1** -- `success_rate_recent` 1.0 and
  ALL THREE populated tilt bins 1.0 (n 779 / 719 / 502) over 768806
  episodes; RT-173's eight failures in the 6-9 deg bin are gone. The
  unchanged reward pays a 6 mm lateral start on top of 8 deg tilt. The run
  is UNKLAR and not PASS for three named gaps, none of them a result: P7
  (sigma at the LAST block) is proven only for the first blocks because no
  scalars CSV is exported yet; the `[INFO]: Loading model checkpoint from:`
  line did not come back from the filter; and P1's "XY offset up to ~8.5 mm"
  cannot be asked with two report envs (step 2 read 2.720 and 0.648 mm).
  ONE EXPECTATION WAS CORRECTED BEFORE THE FILTER RAN and the correction is
  the reader's warning: the file asked for `Learning iteration 600/1400` at
  `--max_iterations 800`; the run used **1500**, so the additive line is
  600/2100 and L284 reads exactly that. Budget change by the user, not a
  code finding. **THE RED NUMBER, reported not interpreted:**
  `stage1_lateral_y_mm` max climbs 0 -> 9.33 -> 13.18 mm and `over_reach`
  0 -> 1 -> 2 across RT-172 / RT-173 / RT-174. Force went the other way:
  p95 24.34 N against RT-173's 25.99 N.
  **RT-175 is CLOSED (ran 2026-09-07 09:03:49, git `1f9eba1`, folder
  `09-07_09-03-58_offset5mm_tilt0-8deg_yaw+-6deg_currOFF_start+30..+50mm_lat6mm_seed42`,
  resume RT-174's `model_1450.pt`, 1450->1950, hand stop at 1651; judged
  2026-09-07, `VERDICTS.md` six lines):** the yaw run
  (`env.fixture_yaw_noise_rad=0.1047`, +-6 deg, on top of tilt 0..8 deg and
  lateral 6 mm), expectation `rt_logs/RT-175_expectation.md`. Verdict
  **UNKLAR for the same two gaps as RT-174 (P7 at the last block, the
  checkpoint load line), but P5 is BRANCH 1 and NOT confounded** -- RT-174
  was judged branch 1 first, which is the condition the expectation put on
  reading P5. `success_rate_recent` 0.9925, tilt bins 0.9934 / 0.9933 /
  0.9900 over 82004 episodes. **THE UNCHANGED REWARD NOW CARRIES ALL THREE
  AXES OF THE TARGET SET AT ONCE:** tilt 0..8 deg, lateral +-6 mm, pocket
  yaw +-6 deg. The yaw really is in the pose, not just in a flag: the
  `entrance buffer` line reads yaw +0.0682 / -0.0601 rad against +0.0000
  in RT-173/RT-174. **TWO NUMBERS WENT THE WRONG WAY, reported not
  interpreted:** the first force abort of the lineage (max 53.59 N,
  `over_f_max` 1, rate 0.0005; RT-172/173/174 all read 0 in their own
  windows). CORRECTED the same day, measured: `over_f_max` is a TRAILING
  WINDOW count, not a cumulative one (`insertion_env.py:2323`), so "1"
  means one episode of the LAST 2000. And it was not one event: the block
  rate reads 0.0000 -> 0.0005 (L705) -> 0.0000 again (L1220) -> 0.0010
  (L5756) -> 0.0005 at the end. Several aborts, and the PEAK is late.
  This also answers the user's thesis of 2026-09-07 ("the aborts would
  fall with longer training"): over these 201 iterations the direction is
  UP, not down. Not refuted -- 201 iterations is short, the yaw axis was
  new in this run, and the full per-iteration curve lives in
  `RT-175_scalars.csv`, which is not exported yet, and `over_reach` 37 of 2000 against RT-174's 2, with the
  lateral p99 now 9.33 mm, i.e. ABOVE the 7.8847 mm reach mark instead of
  below it. Registered as an OBSERVATION, not a finding: successful
  episodes take 28.35 steps against RT-174's 16.28, and `kernels`,
  `action_rate` and `time` all move the same way. Compatible with "the
  policy searches longer"; the duration is measured, the cause is not.
  **Newest checkpoint on the training PC: `model_1650.pt` (user,
  2026-09-07).**
  **RT-173 is CLOSED (ran 2026-09-07 00:51:57, git `35d91b6`, PASS, P5
  branch 1, `VERDICTS.md` two lines):** tilt 0..8 deg on RT-172's
  `model_600.pt`, bins 1.0 / 1.0 / 0.9852, all 8 failures of the window in
  the 6-8 deg bin, user stop at ~624/1100. Log came by CHAT, not by file;
  scalars CSV now on file (`docs/figures/RT-173_scalars.csv`, 25 iteration
  rows 600..624). **Documentation chain DONE 2026-09-07:**
  `docs/runs/RT-173.md`, four curves with the RT-172 overlay, report
  section "Lauf RT-173", `InBachelorErwaehnen.md` row (the ladder ends at
  the 8.52 deg cone). Still missing: the full log as a file.
  RT-169 and RT-170 are SPENT (verdicts
  2026-09-06, the tilt-x reward-curve sweeps). **RT-171 is RETIRED, never
  ran** (the alignment-term run, dropped by D-172; the number stays out of
  use because it is referenced in four files). **RT-172 is CLOSED
  (ran 2026-09-06 18:57:14, git `3612660`, judged 2026-09-06/07,
  `VERDICTS.md` seven lines):** the tilt fine-tune of the RT-156
  checkpoint (`env.fixture_tilt_noise_rad=0.0873`, SAMPLED +30..+50 mm
  start, tip box 0.07, resume `model_250.pt`), expectation
  `rt_logs/RT-172_expectation.md` with two NACHTRAG blocks. Result: P5
  BRANCH 1, both tilt bins 1.0 (0-3 deg n 1224, 3-6 deg n 776), user stop
  at iteration 632/2250. Documented: `docs/runs/RT-172.md`, report section
  "Lauf RT-172", figures `docs/figures/RT-172_*`. See § Open 2d. The line
  below is the STALE predecessor, kept for the trail.
- *(stale)* **Next free RT number: RT-169.** (Three earlier lines here -- "RT-160",
  "RT-161" and "RT-168" -- were stale in turn and are replaced by this one.
  RT-157 and RT-158 are CLOSED; RT-157 covers four heights plus the
  RT-157h40t trace, RT-158 is PASS with a NACHTRAG and shows the failing
  part is TILTED.)
- **RT-160 WAS ISSUED TWICE, 2026-09-06 -- read this before citing it.** Two
  sessions took it from the stale ledger above within one hour. It belongs to
  the RESUMED TRAINING RUN of 16:15:10 (`VERDICTS.md`; the run D-170 cites for
  the 0.0354 mm start residual), because that is the one with a verdict line.
  The righting probe that also carried the label moved on -- twice, see the
  next entry. Its two CRASHED attempts keep the old RT-160 label in
  `PROBLEMS.md` and in `tilt_recovery_probe.py`'s comments -- their log
  headers really do say RT-160, and rewriting them would cut the trail to the
  actual logs. Each of those places now carries the disambiguation in line.
  THE CAUSE WAS THIS FILE: nobody updated the ledger after spending a number,
  so the next reader took a spent one. Spend a number here in the same commit
  that hands the command over.
- **THE FIX FOR RT-160 THEN COLLIDED A SECOND TIME, SAME DAY.** The righting
  probe was moved to RT-167 without checking that number was free either --
  it was not: RT-167 was already the training PC's own name for a finished
  `check_seated_success --reward-curve` run (its log names itself
  `RT-167.txt`, judged FAIL/mechanical in `VERDICTS.md`). That run keeps
  RT-167, same rule as above: a result beats a plan. The righting probe is
  now **RT-168** (`rt_logs/RT-168_expectation.md`); it RAN and is CLOSED, see the ledger entry below.
  Two renumberings of the same probe in one day is itself the finding: fixing
  a stale ledger entry by hand, under time pressure, reproduces the defect
  it was fixing. The number is spent here, in this sentence, in the same
  commit as the rename -- not left for the next reader to infer.
- **RT-161, RT-162, RT-163 are SPENT, not free.** Handed over in chat on
  2026-09-06 as the lateral-offset ladder under D-170
  (`env.start_lateral_offset` 0.002 / 0.004 / 0.006, replays of the RT-160
  checkpoint). No log ever came back and there is no verdict line, so they
  measure nothing -- but they are NOT to be re-issued, because the exact
  commands are out there and a returning log would collide a second time.
- **RT-164, RT-165, RT-166, RT-167 are CLOSED** (verdicts 2026-09-06), the
  tilt ladder of the reward curve via `check_seated_success --reward-curve`.
  RT-164 (x) and RT-165 (y) are FAIL on the INSTRUMENT, not on the reward: the
  height grid stopped at -10 mm while `engaged_depth_m` is 10.8 mm, so
  `engaged` never fired at ANY tilt, 0 deg included, and the gate question was
  never put. RT-166 repeated x with heights to -20 mm: `engaged` fires from
  -15 mm at 0 deg, and `POSE NOT HELD` turns out to depend on DEPTH AND TILT
  TOGETHER, not tilt alone. RT-167 narrowed the grid (0-3 deg, -5..-15 mm) and
  sharpened the threshold: 0 deg and 1 deg hold through -15 mm, 2 deg breaks
  between -5 and -8 mm, 3 deg breaks already at -5 mm, the shallowest point
  tested. All four exit 1 on the script's own `verdict_ok False`
  (`all_poses_held False`), which is a measurement failure, not a crash.
- **RT-159 is CLOSED: FAIL** (verdict `VERDICTS.md` 2026-09-06). It repeated
  RT-158 with `env.osc_kp_rot=100.0`; the first-episode rate fell from
  0.5273 to 1/256 = 0.0039, and the failure is a THIRD mode (255 timeouts
  parked at median `z_end` +7.72 mm ABOVE the opening, `reached_below_entrance`
  0 of 255). It shows the rotation stiffness cannot be swapped under a
  trained policy. It does NOT show 30 is the better value: a policy TRAINED
  at another gain has never been measured.
- **RT-168 is CLOSED: PASS mechanically, and it RETIRES `osc_kp_rot`**
  (verdicts `VERDICTS.md` 2026-09-06, five lines; git `27c28bb`, 16 envs,
  mode a, 18:00:31). Renumbered twice -- see the two collision entries above:
  RT-160, then RT-167, now RT-168. It is the FIRST run of
  `scripts/tilt_recovery_probe.py` that reached the simulation loop; the two
  earlier attempts died on defects of the script itself and both are in
  `PROBLEMS.md`. The probe is no longer UNVERIFIED as far as running goes.
  Answer: the tilt fell to the 8.52 deg clamp at every gain (8.5212 /
  8.5210 / 8.5208 / 8.5206 at 30 / 50 / 100 / 200) and the gain only changed
  the settling SPEED. Its own confound, and it is the reason no gain number
  comes out of this run: `force_n` 0.0003-0.0007 N and `tip_z` +17.5..+21.5 mm
  above the opening plane -- NO contact, so a FREE part was righted, not a
  jammed one. See § Open NEXT 1.
- *(historic, kept for the trail)* **Next free RT number: RT-157.** RT-154 is RUNNING on the training PC
  (2026-09-05, git `a123c4a`, expectation on file). RT-155 (seated identity
  under OSC) and RT-156 (SBC start under OSC) have expectation files
  written BEFORE their runs (`rt_logs/RT-155_expectation.md`,
  `RT-156_expectation.md`); neither has run. RT-152 (yaw slice) stays
  DEFERRED and keeps no number reservation — it takes the next free one
  when it runs. RT-151c and RT-153 are CLOSED (`VERDICTS.md` 2026-09-05).
- Judged in `rt_logs/VERDICTS.md`, 2026-09-04/05, all under osc kp 100:
  RT-143a/b (FAIL: tip-box clamp, not droop), RT-144a–d (K1 comparison;
  b vs c is a travel-time race, not a kp finding), RT-145 (PASS, hold),
  RT-146 (FAIL, episode-seconds trap), RT-147 (PASS, delta is a velocity),
  RT-148a/b (PPO 50 / 4000 iterations, 0 % success), RT-149 (replay trace
  of the RT-148b checkpoint, DETAIL block), RT-150 (PASS, scripted seat).
- RT-142's fourth fix (git `0d787a3`, tilt-ladder code now under marker
  `check_seated_success-2026-09-05`) has NOT run; it measures the tilt
  ladder of the reward curve and is parked, not dropped.
- Every spent or retired number (RT-59 … RT-150) is judged in
  `rt_logs/VERDICTS.md`. Never re-issue a number, never re-run an archived
  command. RT-60..RT-64 are in the SCENE stream's ledger.

## State (current, one line per fact)

- **THE CURRENT REWARD LEARNS A 0..5 DEG TILTED POCKET, measured** (RT-172,
  git `3612660`, training PC, fine-tune of RT-156 `model_250.pt`, starts
  +30..+50 mm, user stop at 632/2250): success 1.0 in both tilt bins
  (n 1224 / 776), force p95 26.54 N, 0 aborts at 50 N, sigma 0.57. What it
  does not say: § Open 2d. Home of the numbers: `docs/figures/RT-172_*`,
  `docs/runs/RT-172.md`, `rt_logs/VERDICTS.md`.
- **`--max_iterations` is ADDITIVE on `--resume`** (RT-172 log: "Learning
  iteration 250/2250"). Budget a resume as loaded + N.
- **The height-bin table cannot resolve a band above +30 mm**
  (`insertion_env.py:715`, last bin open at the top): a run that starts
  +30..+50 mm lands in ONE bin. Instrument gap, UNCHANGED code.
- **A `play.py` replay rebuilds the env cfg from DEFAULTS** (hydra); every
  `env.*` override of the training command must be repeated, tilt noise
  included (`rt_logs/RT-172_expectation.md`, Replay NACHTRAG).
- **THE TILT CLAMP SEPARATES SUCCESS FROM FAILURE, measured** (RT-158
  trace, laptop, `scripts/trace_tilt_steering.py`). Against
  `osc_tilt_clamp_rad` = 8.52 deg: 0 of 135 successes over it, 71 of 71
  mode-A and 46 of 50 mode-B failures over it; tilt medians 2.23 / 10.22 /
  15.32 deg. Every failure therefore runs a permanent "straighten up"
  command that fails. Numbers: `rt_logs/VERDICTS.md`, ACHSEN-MESSUNG.
- **THE MOMENT CHANNELS ARE RULED OUT, measured.** The angle between the
  commanded lateral action and the deep side of the tilt is tightly
  distributed in every group (p25-p75 spread ~3 deg: success 33.4, mode A
  65.3, mode B 48.2 deg median), and 0 of 253 failures drive AWAY. A policy
  that cannot see the tilt would spread over 0-180 deg. `ee_quat` already
  carries the information and the policy uses it, so `OBS_SLICES` 28 -> 31
  would add a form, not a fact. `p5-wrench-obs` is superseded.
  [INLINE CORRECTION 2026-09-13, D-188: this line holds for the TILT --
  `ee_quat` shows it -- and NOT for the CONTACT LOCATION. A moment is
  `r x F` and carries the lever arm; neither the three force channels
  (RT-157 BEFUND 4) nor `ee_quat` do. D-188 adds the three torque
  channels as the OPTIONAL mode `obs_wrench_mode=wrench` (31 wide);
  `force` (28) stays the default. Branch `p5-kraftsensor`.]
- **The failure is a frozen weak-push state, measured.** Mean command
  magnitude 1.192 (success) against 0.312 (A) and 0.460 (B); motion over the
  same 20 steps 0.847 / 0.012 / 0.043 mm; command straightness
  |mean a|/mean|a| 0.59 / 0.975 / 0.993, so it is ONE steady direction, not
  cancelled chatter.
- **`osc_kp_rot` is the open suspect and is UNMEASURED.** Restoring moment
  `Lambda_rot * kp * theta_err` at 30 is 0.203 N*m (mode A) and 0.812 N*m
  (mode B), with `Lambda_rot` max 0.22806 kg m^2 from the startup report.
  Applying the D-166 rule shape to rotation gives a band of 44-65 -- but its
  moment budget (20 N at half the part width) is an OWN CONSTRUCTION with no
  source, and `osc_rot_step_limit_rad` is a Factory placeholder. 30 sits
  under that band, 100 (RT-159) above it; 50 has never been run.
- **`scripts/tilt_recovery_probe.py` is UNVERIFIED.** Self-test 26/26 on the
  laptop, `py_compile` OK, and all 31 of its Isaac accesses read back against
  `insertion_env.py` / `insertion_env_cfg.py` / `seat_probe.py`. It has never
  reached the simulation loop. `torch_shim` grew `.clone()` and
  `torch.cross` for it; `check_insertion_math` 214/214 and
  `selftest_checks.py --offline` still PASS after that change.
- **`scripts/trace_tilt_steering.py`** -- self-test 23/23, four D-080
  mutations. Offline, stdlib, imports the first-episode rule from
  `trace_outcome_split.py`. Its frame precondition is
  `fixture_yaw_rad = fixture_yaw_noise_rad = 0`; it cannot read that from a
  trace and says so.
- **RT-157 is documented** in `docs/Blockberichte/Bericht_Trainingsdokumentation.tex`
  section 3, with `RT-157_probe_starthoehen.pdf` beside it (drawn by
  `scripts/plot_probe_start_heights.py` from the four committed
  `docs/figures/RT-157h*_demo_metrics.json`).

- **Log handover is ONE paste since 2026-09-05:** `rt_log.ps1` puts
  `rt_logs/<NAME>.short.txt` on the clipboard (`scripts/shorten_rt_log.py`,
  self-test 29/29 laptop; verified on the training PC by the RT-151a
  rerun, git `4c4d44b`: every number identical to the full log, JSON not
  needed). `check_seated_success --reward-curve` prints a SUMMARY block
  with all envs per row; marker `check_seated_success-2026-09-05`.
  Reusing a NAME appends to the same full log and the short log then
  carries every run under that name (RT-151a-cp ran twice, both in one paste).

**Verified on Isaac (training PC), all under `joint_pd`:**

- SDF distance half: RT-90, git `3dbb21c`. SAPU pair: RT-95/96, `a70a86e`.
  64,000 samples + p99: RT-100/101, `1c79b11`.
- Real reward + termination + teleport identity: RT-102/103, `05fdfb3`,
  D-109 identity EXACT. Seated + shallow: RT-109/110, `33025fc`. Lateral
  counter-proof: RT-114/115 (D-160).
- Free-standing scene + live fixture noise: RT-106, `964ca2f`.
- Curriculum-OFF switch trains: RT-104, `a9f2d9c` (72–114 ms per RL step at
  128 envs).
- Reward curve at the home pose: RT-119, `92fd6d3` (D-161); upright ladder
  RT-123 PASS.
- PPO under joint_pd: RT-138 (SBC start, 37 % success, forces just under
  the 300 N abort), RT-139/140/141 FAIL (contact penalty: either no effect or
  0 % success). Numbers: `rt_logs/VERDICTS.md`,
  `docs/figures/RT-13[89]_demo_metrics.json`, `RT-140_demo_metrics.json`.
  Reading rule for `reward_terms_recent_mean`: rows are EPISODE SUMS, divide
  by `time / -0.00390625` (archive 2026-09-03).
- Drives (RT-45, RT-46): the welded USD carries stiffness 9400.5 / damping
  0.378 / maxForce 150 (shoulder, elbow) and 28 (wrist); the arm holds the
  home pose under them at 0.317026 m flange standoff (t = 3.0 s).

**Verified on Isaac (training PC) under `osc`, kp 100, 15 Hz, 256-step
episode (evidence: `rt_logs/VERDICTS.md`, run + git SHA):**

- Controller chain runs; drives read back inert (0/0), maxForce 150/28,
  Lambda_max 8.82 kg at the reset pose, cond(J) 7.78, Decision (4) bound
  kp <= 113.4 N/m at F_search 20 N (RT-143a/b, `8856900`).
- Hold: started 30 mm above the entrance the depth stays at -29.987 mm for
  200 steps under zero action, force decays 1.516 -> 0.000 N by step 20
  (RT-145). Outside the +-50 mm tip box the clamp pulls the target to the
  box edge (RT-143's "115 mm sink"); every training episode starts INSIDE
  the box (`RUNG0_START_TIP_ABOVE_ENTRANCE` 0.030 m).
- The pose delta is a VELOCITY: `v = Delta*sqrt(kp)/(2*zeta)` per second,
  independent of the policy rate (RT-147, `22a551b`; 60 Hz repeat of
  RT-144b with the same 17.07 s gave the same depth). Stall push =
  `Lambda*kp*Delta`: 17.64 N at Delta 0.02 m.
- **kp 100 SEATS when Delta is at the step limit** (RT-150, `114815d`,
  user viewer check the same day): `scripted_insert.py --control-mode osc
  --osc-kp-pos 100 --descend-rate 0.02 --max-step 0.02`, 32/32 episodes
  DESCEND to the pocket floor, depth p50 36.003 mm (seat threshold 33 mm),
  force max 16.72 N < 17.64 N — the reset-pose Lambda reads back in
  contact. `scripted_insert.py`'s DEFAULT rates (0.5 / 2 mm per control
  step) are joint_pd step sizes and cannot finish inside the episode under
  osc.
- PPO: RT-148a (50 it., `c7c3853`) and RT-148b (4000 it., 4:06 h, 32,000
  episodes, `3989ff9`): success 0.0, `mean_max_depth_mm` 0.0,
  `over_reach` 1994/2000, sigma 0.65 -> 0.01 before iteration 500.
  Numbers: `docs/figures/RT-148b_demo_metrics.json` (action_rate window
  mean -3.907; console at iteration 3999 -3.69 — two windows, same run).
  Findings and four corrections: VERDICTS RT-148b + NACHTRAG 2026-09-05
  (N1: `action_rate` is 96 % a tax on the exploration noise at sigma 0.65).
- Replay of `model_3999.pt` (RT-149, `29d0177`, `play.py --trace-obs`,
  `scripts/summarize_trace.py`): the episode is 255 policy steps; the
  policy is NOT frozen (1/8 envs stall at the -50 mm clamp with action -> 0,
  3/8 run a ~64-step limit cycle); `force_norm_n` p50 2.021 N IS the
  reset-row force in all envs (episode max = reset transient); during the
  episode force p50 <= 0.05 N; no +15 mm rest; yaw jumps 25 deg then holds;
  x reaches the +-50 mm clamp in 5/8 envs. `rt_logs/RT-149_trace.json` is
  gitignored (1.3 MB); the summary lines are in VERDICTS DETAIL RT-149.

**Built 2026-09-03 (git `52c0853`); the chain is Isaac-verified above,
individual items below are as written:**

- `cfg.control_mode` = `"osc"` (default) | `"joint_pd"`; resolved by
  `insertion_env_cfg.resolve_control_mode` BEFORE `super().__init__`
  (decimation, episode seconds, render interval). Unknown mode refuses.
- OSC: `pose_abs`, `impedance_mode fixed`, full inertial decoupling,
  `gravity_compensation True` with gravity ON, `nullspace none`; controlled
  body `tool_link` (Jacobian row block `body - 1`); efforts via
  `set_joint_effort_target` every physics step; zero efforts + zero delta at
  reset. Pattern: `run_osc.py`, `test_operational_space.py:350-358`,
  `task_space_actions.py:648-651`.
- Drives under osc: stiffness AND damping 0.0 on all three actuator blocks,
  written into `cfg.robot_cfg.actuators` in `_setup_scene` before the
  Articulation is built; maxForce, limits, armature, friction USD. The
  startup report READS the gains back from PhysX and prints `*** DRIVES NOT
  INERT ***` if they are not 0.
- Action: `clamp(a) * (osc_pos_step_limit_m, osc_rot_step_limit_rad)`, held
  over the decimation window, target = current pose + delta re-anchored each
  physics step; rotation delta PRE-multiplied (env frame, Factory);
  `insertion_math.clamp_tilt_to_cone` (8.52 deg, measured from THIS
  EPISODE'S POCKET AXIS since B7 part 1 — NOT from the world vertical) then
  `clamp_tip_in_box` (±`osc_pos_clamp_m` in x/y and ±`osc_pos_clamp_z_m` in
  z around this episode's `_entrance_pos`, acting on the LEADING TOOL
  POINT).
- Placeholders (D-149 form, in `RL_PLACEHOLDERS`, printed + in
  `demo_metrics.json`): `osc_kp_pos` 100, `osc_kp_rot` 30,
  `osc_pos_step_limit_m` 0.02, `osc_rot_step_limit_rad` 0.097,
  `osc_pos_clamp_m` 0.08 and `osc_pos_clamp_z_m` 0.1435 (both D-180,
  `5a7d295`; 0.05 / 0.10 until then), `osc_decimation` 8. Damping ratio 1.0
  is a RULE (critical), not a placeholder. The cone is CAD.
- Episode under osc: 256 STEPS kept (D-113 (3)), i.e. 17.07 s at 15 Hz;
  `time_penalty_per_step` -1/256 unchanged. Re-deriving the length is OWED
  (audit section 6 (5)) — not done.
- Startup report `--- controller ---` block (both modes): mode, gains,
  limits, the per-joint drive table (USD type, stiffness, damping,
  maxForce read back), Lambda eigenvalues (translational / rotational, env
  0, at the reset pose), the Jacobian condition number, the gravity
  torques vs maxForce, and the Decision (4) bound `kp <= 20 N /
  (Lambda_max * step)`.
- Instruments: `tilt_insert.py` (marker `tilt_insert-2026-09-03a`) and
  `scripted_insert.py` (`scripted_insert-2026-09-03a-h1`) take
  `--control-mode`, `--osc-kp-pos`, `--osc-kp-rot`, `--decimation`; under
  osc the pocket command goes pocket → env → action
  (`scripted_policy.osc_action_from_pocket_command`), no IK. `tilt_insert`
  logs all six joint torques per control step (`applied_torque` vs
  `joint_effort_limits`: per-joint peak, saturation at 0.99·maxForce, depth
  profile, per-step trace in the JSON). `zero_agent.py` takes `--home-pose`
  and `--osc-kp-pos`.
- `_max_target_lag` reads 0 under osc (no integrator); it is a joint_pd
  diagnostic only.
- Offline evidence (laptop, 2026-09-03): `check_insertion_math` 214 +
  counter-proof, `check_env_wiring` 156 + counter-proof (11 new mutations),
  `check_scripted_insert` 108 / 56 mutations, `check_seated_success
  --self-test` 49/49, `check_curriculum_ladder` 29, `selftest_checks
  --offline` PASS.
- Code marker: `insertion-osc-2026-09-03a`.

**Env state (unchanged by the switch):**

- Observation 0:28 (`insertion_math.OBS_SLICES`), force block 25:28
  EMA-smoothed, gravity-tared. `pocket_quat` is constant (all fixture
  orientation noise 0). `obs_noise_enabled` is read nowhere.
- Reward: D-109 kernels + 2026-09-01 revision + D-165 progress; `contact`
  and `far` are removed from the code (D-184).
  Success terminates with the payout lump; the force abort is
  `truncated` (D-164, BUILT — `insertion_math.py:1038`).
- `curriculum_enabled = False`; rung 0 start `+0.030 m` with
  `start_tip_above_entrance_low` (SBC, Uniform[low, high]); fixture noise
  5 mm xy.
- `RL_PENDING` = `rung_step_sizes` only. `RL_PLACEHOLDERS` = `abort_payment`,
  `force_abort_f_max_n` (value in the cfg; D-169, lowered 2026-09-15, inbox "Seed study restart"), `engaged_depth_m`, plus the six
  OSC fields above.
- `demo_metrics.json` is written only on episode end (>= 16 iterations at
  128 envs, `_ep_count` a multiple of `_metrics_dump_every`); it now carries
  `control_mode` and `policy_rate_hz`.

**Phase cut 2 → 3 → 4:**

- Phase 2 frozen by the tags `phase-2-abschluss-*`. Phase 3 branches start
  at the phase-2 heads. `p4-rl-code` was branched from `p3-rl-code` at
  `0d787a3` (2026-09-03) and is pushed; the training PC runs it from the
  new sparse clone `Phase3_Implementierung_v2`.
- The merge to `main` is DEFERRED (deliberate deviation from D-065): three
  streams are ahead of main and `DECISIONS.md` is main-only. Own task,
  nothing else in flight.

## Open

### NEXT, in order — nothing skips

**START HERE (2026-09-12, chat 3). Read this block, then act. Everything
below it is history and detail; open it only when a line here points there.**

* **2026-09-14, `p5-kraftsensor`: RT-201 REACHED ITS TARGET -- the floor
  unlocks AutoDR.** autodr + floor -0.030, seed 1: floor on H_min at it 55,
  all seven ceilings by it 623, `[stop-criterion] target reached` at it 850
  (fresh 0.9955 over 2000), 1 h 59 min; RT-189s1 without the floor had 0
  success in 3000. Verdict line in `rt_logs/VERDICTS.md`, CSV in
  `docs/figures/RT-201_scalars.csv`. Also built the same day: stagnation
  readings (`dr/<key>_last_rate`, `dr/<key>_holds`, hold lines, STALL note
  via `env.autodr_stall_buffers`, sidecar `autodr-3`; inbox "AutoDR
  stagnation readings") and the check speed-up (wiring counter-proof 52 s,
  autodr 29 s; RULE: run only the check that pins the changed file).
  NEXT: RT-202 = the No-DR twin (`rt_logs/RT-202_expectation.md`). Open:
  RT-199e (optional), the autodr-3 resume path (RT-200c skipped),
  interpen max 3.4 mm (as RT-193/194), tilt bin 8-10 deg at 0.984 is the
  weakest bin.

* **2026-09-14, `p5-kraftsensor`: H_min = 20 mm MEASURED and the START
  FLOOR (SBC step 0) is BUILT, UNVERIFIED.** H_min: RT-197a/b clean in the
  full corner, 10 and 5 mm pushed out of the rim (RT-198b, RT-199b; centre
  clean, RT-199c) -- inbox "H_min is MEASURED"; `RUNG0_START_TIP_ABOVE_ENTRANCE`
  is 0.020 now (a No-DR run without an override starts at +20 mm). Floor:
  inbox "The start floor (SBC step 0) is built", `Pläne/SBC_Schritt0_Entwurf.md`
  § 0.1b, plan `synthetic-marinating-mitten.md`; cfg `start_floor_m` /
  `start_floor_steps`, key `start_height_floor`, sidecar `autodr-2`,
  No-DR branch on the same provider. NEXT: RT-200a/b/c smoke
  (`rt_logs/RT-200_expectation.md`, 2 iterations each) after RT-196 (torque
  noise, running 2026-09-14 morning) -- then the first floored AutoDR run
  with its own expectation. Open: RT-199e (offset without tilt, explains the
  push), the missing `[rt_log] exit code:` line, the under-fixture depth path.
* **BRANCH `p5-kraftsensor` (2026-09-13, plan J):** the three torque
  channels as observation mode `wrench` (D-188), the wrench EMA skips the
  reset step and advances once per step (inbox entry 2026-09-13 -- this
  changes row 0 of the `force` channel for EVERY policy), Pruefblock B is
  `probe_wrench_bodies.py --tare-check` (RT-192 expectation; pushed
  2026-09-13, origin/p5-kraftsensor at `%s` (rename commit),
  code state `971f32c0`). Plan:
  `Pläne/PlanKraft plus Drehmoment in der Beobachtung.md`. NOT merged into
  `p5-robustheit`; the training PC runs the Rauschen-aus-Lauf from
  `p5-robustheit` FIRST, then RT-192 (never two runs at once, RT-182).

* Judged today: RT-180 FAIL (L3, clamp pull), RT-180b FAIL (L8 only; L3 and
  D-157 HOLD at one settle step), RT-181 UNKLAR (P4 needs more logs, P8 no
  number). RT-182 is DROPPED: user measured two runs at once = 37 s instead
  of 13.9 s per iteration; seeds run one at a time (`docs/decisions_inbox.md`,
  last entry). Verdict lines: `rt_logs/VERDICTS.md`, last four.
* **THE CLAMP TASK IS DONE AND VERIFIED (2026-09-12, chat 3). D-186,
  RT-180c PASS.** The `--lateral` branch of `check_seated_success.py` widens
  `env_cfg.osc_pos_clamp_m` to `max(clamp_m, |lateral_y_m| + 0.005)` before
  `gym.make`: 0.08 m -> 0.1968892059209824 m at the default offset. The z
  clamp, seated, shallow and training are untouched. Marker
  `check_seated_success-2026-09-12a`. Laptop: `--self-test` 81/81,
  `selftest_checks.py --offline` PASS.
  * **RT-180c: all eight points PASS, 22 of 22 pins met**, read from
    `rt_logs/RT-180c/RT-180c_lateral.json` (git-ignored, the user pasted the
    JSON). `osc_pos_clamp_m` 0.1968892059209824, `settle_steps` 30,
    `lateral_y_mm` 191.88926 mm against a commanded 191.88921 mm, depth
    34.0002 mm, `force_n` 0.00036 N at `force_n_raw_untared` 8.08344 N.
    Verdict line in `rt_logs/VERDICTS.md`, last line.
  * **THE 18.66 mm GAP IS CLOSED.** It was the clamp pull, not a different
    reference point: at the widened box the same command lands 0.00007 mm
    from its target. One settle step under the 0.08 m box had therefore
    already moved the tip 18.66 mm, which is also why RT-180b's raw force
    stood above the tool weight. PROBLEMS.md row, 2026-09-12, top of the
    table.
  * **D-157 is now exercised AND at rest on the OSC env.** RT-180b showed
    it at one step with the part still moving; RT-180c shows it after 30
    steps at 0.00036 N.
  * CORRECTED, and worth not repeating: "RT-180b raw 11.6 N" was wrong in
    the first version of D-186 and of the RT-180c expectation. 11.6 N raw
    (3.03 N tared) belongs to the SECOND probe at TWO settle steps. RT-180b
    at one step read 1.569 N tared, 8.28 N raw. Both entries carry an
    inline correction.
  * LIMIT: the `[rt_log]` head was never handed over for RT-180c, so its
    `git:` line is not evidenced. The marker in the JSON pins the script
    version.
* **STEP 4 OF THE RUN PLAN, first question ANSWERED (2026-09-12, RT-183).**
  The plan's step 4.1 asks for a mesh repair. Before building anything for
  it, the defect was re-measured against TODAY's asset, because rule L-09
  forbids reading an old run as a current state and the asset lives outside
  git. **The hole is still there, at the same address.** RT-183, git
  `379aab2`, verdict line in `rt_logs/VERDICTS.md`: G1 903/757 PASS, G2
  frame guard PASS with every error 0.0000 mm, `ZERO-EDGE COUNT 3` at
  x 35.9863..37.4512, y 79.6347..79.7500, z 0.0000 mm, verdict
  `OPEN_EDGES_NOT_CLEAR`, exit 1 — which D-153 wants. D-153 carries the
  re-measurement line.
  * Both code constants are confirmed against today's numbers:
    `POCKET_HOLE_MIN_Y` 79.6347 mm, and `HOLE_REACH_OFFSET_Y` = 79.6347 −
    71.75 = 7.8847 mm. Stage 1 allows 8.7000 mm, so the reachable window is
    0.8153 mm wide. The plan's 30 mm radius is 3.80x the reach offset.
  * **STILL OPEN, and it is the next question: does reaching the hole do
    any harm?** D-153 calls the approach "NOT provably safe", which is an
    exposure, not a measured failure. The instrument is
    `stage1_lateral_y_mm.over_reach` (`insertion_env.py:3409`). Nobody has
    run it above the reach offset, because `start_lateral_offset` defaults
    to 0.0 (`insertion_env_cfg.py:479`) and D-178 (5) forbids a larger
    radius until the mesh is repaired and re-measured. That is a circle:
    the permission needs the repair, the repair needs the evidence. Break
    it deliberately or repair without evidence — user decision, not made.
  * NOTE on the run: RT-183 had NO committed expectation file. The three
    outcomes were written in chat before the run and the verdict line says
    so. From now on the `rt-expectation` subagent writes the file first.
* **DOES THE HOLE HARM ANYTHING? RT-184 says: not through the reward, and
  the reason is RESOLUTION (2026-09-12).** `--reward-curve`, grid of four
  heights (0, 5, 10, 15 mm) x six lateral offsets (0, 4, 7, 7.8847, 8.2,
  8.7 mm), so three offsets below the 7.8847 mm reach and three inside the
  0.8153 mm exposure window. Expectation `rt_logs/RT-184_expectation.md`,
  verdict line in `rt_logs/VERDICTS.md`, table in
  `rt_logs/RT-184/sdf_table.csv` (git-ignored).
  * `sdf_mean_outside_mm` rises SMOOTHLY across the boundary at all four
    heights. The slope change across it is 1.27x to 1.39x; the change
    between the two CLEAN support points below it (0->4 vs 4->7) is 1.98x.
    The crossing is therefore less of a change than the clean part of the
    curve — no jump, no sign flip.
  * `interpen_max_mm` is exactly 0.0 in all 24 rows.
  * Pins all met: `all_solves_converged` and `all_poses_held` true, worst
    `pose_drift_mm` 0.00077 against a 0.05 mm tolerance, `verdict_ok` true,
    `reset_during_sweep` false, marker `check_seated_success-2026-09-12a`,
    four scatter fields 0.0.
  * **THE LIMIT, and it is the real result.** The hole is a triangle of
    1.4649 x 0.1153 mm, area 0.0845 mm2. The reward samples the part with
    `sdf_num_sample_points = 64000` (`insertion_env_cfg.py:1268`), which
    over the part's bounding-box area of 51907 mm2 is 1.233 points per mm2,
    a mean spacing of 0.901 mm. Expected sample points over the hole per
    pose: **0.10**. The hole is an order of magnitude below the sampling
    resolution of the very term it could corrupt. So RT-184 does not show
    the hole is harmless; it shows the reward cannot resolve it.
  * **DECIDED (user, 2026-09-12): take both out. D-187.** The mesh repair is
    dropped from run-plan step 4.1 — with the teleport test before and after
    it, because there is no "after" — and the D-178 (5) radius ban is lifted,
    so a run may draw `lat_r` up to its 0.030 m ceiling. The mesh stays
    unrepaired and `check_fixture_mesh.py` keeps exiting 1 on this asset, the
    way D-153 wanted. Carried into `Laufplan_Phase5.md` step 4,
    `Belege_Streuwerte.md` row A1, the ceiling comment at
    `insertion_env_cfg.py:455-466`, D-178 (5) and D-153's reopening
    condition, plus a row in `InBachelorErwähnen.md`.
  * `stage1_lateral_y_mm.over_reach` stays in the metrics and stays worth
    reading, but D-187 says in as many words it is NOT a gate.
  * **STEP 4 IS NOW ONLY H_min** (`distributed-stargazing-deer.md` 4.2) then
    Commit C. That is the next piece of work for option A.
* Second open decision, user not yet answered: first training run -- A =
  finish Laufplan step 4 first, then No-DR budget run seed 20 (Claude's
  pick); B = a long No-DR smoke on `23a7aa0` now. Ask once, one line.
* Another chat edits `insertion_env.py`, `insertion_env_cfg.py`,
  `check_env_wiring.py`, `tilt_recovery_probe.py` and the three expectation
  files in THIS worktree (critic round 2 repairs). Do not touch those; commit
  only your own paths with explicit `git add <file>`; never stage the
  `Pläne/*` deletions; commit with `git commit -F <file>`.
* Local is 11+ commits ahead of `origin/p5-robustheit`; the training PC is
  on `23a7aa0`. Nothing below 23a7aa0 changed the env, so RT-180c can run
  after a pull of the script change only.

000. **RT-180 IS JUDGED (2026-09-12, chat 2): FAIL on L2/L3 of the lateral
    run; seated, shallow and curve all points met; the three pins HOLD.**
    Verdict line in `rt_logs/VERDICTS.md`, files in `rt_logs/RT-180/`
    (git-ignored). Read that line before anything else.
    * Lateral pose is PULLED during the settle: after 1 settle step the tip
      reads y 173.227 mm / depth +33.972 mm (log L756, inside the band);
      after 30 it reads 84.3 mm / -50.1 mm. Mechanism read off
      `insertion_env.py:1521-1550`: target = current pose + delta, THEN the
      box clamp, so a zero action outside the 0.08 m box is not "stay put".
      The probe's smallest LEGAL y (`check_seated_success.py:398-406`,
      100.45 mm + half part width) is already above the clamp, so every
      legal `--lateral` value is outside the box. The curve mode guards this
      (`:597-606`), the lateral mode does not.
    * **RT-112/RT-113 do NOT cover D-157 on this code** (PD control, no OSC,
      no clamp box then). D-157 has never passed on the OSC env. The test is
      OPEN, not done.
    * OPEN, reported not interpreted: at step 2 the tip reads y 173.227 mm
      while 191.889 mm was commanded (IK residual 0.0003 mm, converged).
      18.66 mm gap, ratio 1.108 = no unit. Unknown whether the command is a
      different point than the tip, or one settle step already pulled.
    * Named, explained, not a finding: seated `force_n_raw_untared` reads
      0.0 (tared 2.02 N) because the print reads the NEXT episode on a
      terminating step (`check_seated_success.py:2262-2267`, its own text).
    * The expectation file names code `9f37a07`; the run ran `23a7aa0`
      (harmless: docstring + the four scatter keys in the curve dump, which
      CLOSES the expectation's own "named gap" — that paragraph is stale).
    * NEXT for the lateral line: `rt_logs/RT-180b_expectation.md`, lateral
      only, `--settle-steps 1`, the eight L-points verbatim, the 173/192 gap
      as "read only". No code change. Written expectation before the run.
    * **RT-181 and RT-182 are NOT judged yet.** Logs come as @-files into
      `Downloads`; the rt-check skill copies them to `rt_logs/RT-18x/`.
      Rule broken once this session and not to repeat: the raw log was
      read with grep before the skill was read. Filter only.
    * Critic round 2 runs in ANOTHER chat (user, 2026-09-12). Do not start
      it here.
    * RE-CHECKED 2026-09-12 (chat 3, on the user's question): the verdict
      holds point by point against the four JSONs, the two startup-report
      lines (OFF bias line L140/422/700/984, grasp zero row L141/423/701/985)
      and both markers. ONE GAP in the record, not in the verdict: only run 1
      carries `[rt_log] exit code` (L282). Runs 2-4 end at their VERDICT line
      with no exit line; `rt_log.ps1:96-98` writes it only when the pipeline
      returns, so each of the three was cut off after the VERDICT. Cause:
      the user stopped them by hand after the VERDICT line (user,
      2026-09-12). Not a hang. The
      lateral run's exit status 1 was therefore never seen; its VERDICT line
      L840 is the script's own FAIL. No verdict changes.
    * **RT-180b RAN (2026-09-12, chat 3; verdict line in `rt_logs/VERDICTS.md`,
      Kit-log time 13:09:19, the `[rt_log]` head was never handed over).**
      `--settle-steps 1`: L3 HOLDS (33.972 mm, |y| 173.227 mm, the RT-180
      L756 numbers to the digit) and L4/L5/L6 hold -- **D-157 is exercised
      on the OSC env for the first time.** L8 FAILS: 1.569 N tared, 8.28 N
      raw. A second probe, same command with `--settle-steps 2` (user,
      JSON pasted, no log): |y| 179.07 mm, depth 24.86 mm, tared 3.03 N,
      raw 11.6 N. Raw ABOVE the tool weight (11.6 > 8.08) means the tool is
      accelerating (9 mm up, 6 mm sideways in one step); per-step tared
      force 6.3 N then 7.5 N, growing. **The tare is fine at rest (RT-180,
      0.04 N at 30 steps); L8 fails on motion, not on the tare.** So the
      lateral mode can never satisfy L3 (needs step 1) and L8 (needs rest)
      in one run under the 0.08 m clamp. Code question, proposed as A: the
      `--lateral` branch sets `env_cfg.osc_pos_clamp_m` wide enough for
      191.9 mm, the way it already pins the four scatter fields
      (`check_seated_success.py:1867-1880`); needs `/decision` and an
      RT-180c expectation. B: leave it, two half proofs. User has not
      chosen yet. The 191.889 vs 173.227 gap is still "read only".

00. **RUN `Laufplan_Phase5.md` STEP 3 ON THE TRAINING PC.** Code:
    `origin/p5-robustheit` = `23a7aa0`. Pull and check the SHA first.
    Three RT numbers, each with its expectation already written:
    `rt_logs/RT-180_expectation.md` (the owed teleport test, four modes),
    `rt_logs/RT-181_expectation.md` (No-DR smoke alone, training seed 20,
    30 iterations), `rt_logs/RT-182_expectation.md` (five at once, seeds 1-5).
    Each file carries its own commands. `dr_mode` defaults to `off`, so NO
    hydra override is passed.
    Three traps, all real: never pass `env.osc_pos_clamp_m` (0.07 < the
    0.070838 m the disk start needs, and NOTHING checks it at runtime — the
    offline check reads the source default and never sees a hydra override;
    the user declined a runtime guard 2026-09-12);
    `env.contact_penalty_scale` / `env.far_penalty_scale` now kill a run,
    those fields left the code with D-184; and the startup report's bias line
    must be compared at step 2 AGAINST step 258 only — 60 and 180 are inside
    the same episode and are identical by construction.

01. **BEFORE `/rt-check` — FOUR REPAIRS CRITIC ROUND 2 ASKS FOR.** Round 2
    scored **7.0 / 10** against a bar of 8 and is the LAST round the rubric
    allows. Full record: `docs/decisions_inbox.md`, "2026-09-12 — Critic
    round 2". Round 1 (6.5) is the entry above it.
    * **THE LOGS MUST BE THE FULL FILES, NOT THE CLIPBOARD**, for any run
      taken before `shorten_rt_log.py` was fixed. The clipboard carries the
      SHORT log, which dropped every repeated startup report — including the
      count-258 bias line that RT-181 P4/P6 and RT-182 P5/P6 read. The
      shortener now keeps the step, bias and grasp lines (self-test 33/33,
      counter-proved), but a log already produced does not gain them.
    * **RT-182's Q rule needs the overlap window.** The five windows start by
      hand, so `Training time` measures a stagger, not throughput. Read the
      `Iteration time` median over the fully overlapping iterations instead.
    * **P4 cannot tell `abs` from `add` as written.** Replace it with the
      pooled rule: `mean|v|(258) / mean|v|(2)` over all six logs — about 1.0
      for `abs`, about 1.41 for `add`.
    * **The three expectation files name `9f37a07`; the code is `23a7aa0`.**
      Line numbers shifted +12 (`check_seated_success.py`) and +6
      (`insertion_env.py`), and RT-180's "A gap, named" paragraph is now
      false. Mark them "(Erwartung korrigiert)".
    **AND A WARRANT THAT WAS WRONG:** "2, 60 and 180 are identical by
    construction" does NOT hold under `train.py`, which passes
    `init_at_random_ep_len=True` (`:725`) so rsl_rl randomises every env's
    episode counter at start. A line that changes before count 258 is an early
    reset, not a finding. Count 258 is still past every first reset, with
    margin zero.

02. **THREE `CLAUDE.md` LINES ARE OWED BY THE USER** (Claude does not edit a
    CLAUDE.md; the 2026-09-12 exemption covered six named places and is spent).
    `scripts/CLAUDE.md`: the `tilt_insert.py` entry needs the three scatter
    pins, and the `check_env_wiring.py` entry needs the five-script pin check,
    the four magnitude ceilings, both metrics dumps and the bias read-back.
    The task `CLAUDE.md`: the `insertion_env.py` entry needs that its 12:15
    docstring now cites D-183.

### Live gaps and reminders

- **A SIXTH script reads the pose out of `obs` and pins nothing:**
  `scripts/tilt_recovery_probe.py`. It is also missing from `scripts/CLAUDE.md`
  and from the pin check's script list.
- **Three of the new checks let a sensible mutation through** (critic round 2,
  rank 5): the pin check does not require the pin BEFORE `gym.make`; nothing
  pins where `_obs_calls` may be written; the report check tests strings, so
  swapping the observation slice prints zeros forever and still passes.
- **The three magnitude ceilings are 14-15x the value** — trip wires for a
  1000x slip, not a 10x one. `D106_DEPTH_BAND` 0.003 m is the tighter
  neighbour for the pocket bias.
- **This file is 3-5x the size `/handoff` allows** (roughly 250-350 lines for
  a code stream). The 2026-09-12 session moved only its own record out, into
  `docs/decisions_inbox.md`. An archive pass into
  `docs/archive/HANDOFF-RL_Archiv_<date>.md` is owed and was NOT done, because
  doing it in a hurry loses facts.
- **Force noise is added AFTER the EMA** (`ft_smoothing_factor` 0.25), about
  2.6x the quantity D-182 cites. NEEDS a `/decision`: accept and write it
  down, or move the channel.
- `obs_noise.py` is only parsed as AST offline and never executed; the `(3,)`
  path cannot be tested under the numpy stand-in.
- The three trace analysers still call their numbers "end pose". The trace
  header records the four scatter widths since `1f4e6f6`, the names do not.
- `fixture_pos_noise_xy` is unpinned in `seat_probe.py`, `zero_agent.py` and
  `scripted_insert.py` (default 0.005). Left open on purpose. Note while
  deciding: the start-pose solver runs LAST in `_reset_idx` and cancels the
  lateral part of it — RT-160, worst residual 0.0354 mm over 17780 episodes
  (`insertion_env_cfg.py:441-447`) — so for the START POSE it changes nothing.
- `DimSpec.bind` makes the lower range check vacuous for a one-sided quantity.
  A named risk until H_min is measured.
- The `Verification status` line of D-178, D-179, D-180, D-182 and D-183 still
  reads `NOT BUILT, UNVERIFIED` although Commit A and Commit B are in.

00. **B7 IS SCORED AND REPAIRED (2026-09-10, round 3).** The critic ran over
    `15e3d0f` with fifteen FRESH mutations: LOGIK **6.5** / CHECKS **4.0**
    (6 of 15) / EHRLICHKEIT **9.0** (30 of 32) / SCOPE **9.0**, bar 8.5.
    Eleven new checks and eleven new mutations later, **14 of the 15 are
    caught**. Numbers, the nine misses and the one deliberate non-fix
    (`osc_pos_step_limit_m` -- the sufficiency rule adds the number it would
    have to police) are in the inbox entry "B7 round 3". Two false claims
    corrected in place: `check_autodr` runs **91** mutations, not 87, and a
    band at 0.110 m needs **0.1302** m.
    Green, laptop only: `check_env_wiring` 305 / 300, `check_insertion_math`
    265 / 99, `check_autodr` 115 / 91, `check_mutation_anchors` 413 anchors /
    0 stale, all three counter-proofs PASSED.
    **NEXT IS THE RT-178 SMOKE RUN, handed over 2026-09-10** at
    `3a65fb1188f0ec574b07a3fb9cb7910b1e709fbc`, into its OWN clone
    `Phase5_Robustheit_v1` (user, 2026-09-10 -- NOT a branch switch of `_v2`;
    the clone paragraph above owns the path and the install warning).
    `env.dr_mode=autodr` is the only override the DR side needs (all six
    centres are inside their limits, and `insertion_env.py:351-381` REFUSES
    the six boundary-owned static fields, so RT-177's tilt/yaw/lateral/start
    overrides must all come off). `--max_iterations 60` because
    `save_interval = 50` is what makes the `autodr_50.json` sidecar appear. What the
    constraints do NOT settle: RT-171..RT-177 all carry
    `env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0` (the cfg
    defaults are 0.2 and 1/KERNEL_MARGIN_COARSE, so dropping them turns two
    reward terms back on; **Closed 2026-09-11 by D-184: both terms are
    removed from the code.**) and `env.osc_pos_clamp_m=0.07` (B7 had since made
    x/y Factory's 0.05 and z 0.10, so carrying it over contradicted the
    decision that just shipped). Asked in chat 2026-09-10; not answered yet.
    **Overtaken 2026-09-11 by D-180: x/y is 0.08 m and z 0.1435 m, so 0.07 is
    now too SMALL, not too large — the rule and the reason are in point 000
    and in `Laufplan_Phase5.md` § Rules.**

0. **RT-174 IS JUDGED (2026-09-07): P5 BRANCH 1. THE UNCHANGED REWARD PAYS
   A 6 mm LATERAL START ON TOP OF 8 deg TILT.** `success_rate_recent` 1.0,
   all three populated tilt bins 1.0 (n 779 / 719 / 502), 768806 episodes;
   RT-173's eight failures in the 6-9 deg bin are GONE. Force fell
   (p95 24.34 N against 25.99 N). Cost of the lateral start, measured: the
   first block reads 0.0758 against RT-173's 0.1190 from the SAME
   checkpoint, and the 0.90 line falls one iteration later (602 against
   601). That is the whole price. Ledger and `VERDICTS.md` (five lines)
   carry the numbers; do not restate them here.
   **THE THREE OPEN GAPS, none of them a result** -- the run reads UNKLAR
   only because of them: (i) **the scalars CSV is not exported**, so P7
   (sigma at the LAST block) and the six-metric line are NOT provable; the
   export command is in the chat handover of 2026-09-07 and names the run
   folder in full. (ii) the `[INFO]: Loading model checkpoint from:` line
   did not come back from `filter_rt_log.py`. (iii) P1's "XY offset up to
   ~8.5 mm" cannot be asked with two report envs (step 2 read 2.720 and
   0.648 mm; the 7.8-10.9 mm at steps 60/180 are the policy's own motion).
   **THE RED NUMBER, reported not interpreted:** `stage1_lateral_y_mm` max
   climbs 0 -> 9.33 -> 13.18 mm and `over_reach` 0 -> 1 -> 2 across
   RT-172 / RT-173 / RT-174. D-153 exposure rises with the lateral start.
   **THE DOWNLOADS TRAP, resolved -- read this before trusting a file
   name:** until 2026-09-07 10:00 the laptop's `Downloads/RT-174_scalars.csv`
   was BYTE-IDENTICAL to `RT-173_scalars.csv` (md5
   `c0d065d8324b73271d88599e12fce713`, iteration rows 600..624) because the
   same export ran before RT-174 started, and `demo_metrics (5).json` read
   `start_lateral_offset_m` 0.0, i.e. RT-173. The real files arrived later
   (`RT-174.txt`, 32742 lines; `demo_metrics (6).json` with
   `start_lateral_offset_m` 0.006). **Check the identity field, never the
   file name.**
   ORDER NOW: (a) **DONE 2026-09-07: `/rt-check RT-175`**, branch 1, not
   confounded -- see the ledger entry. (b) The documentation chain for
   RT-174 AND RT-175 (run file, `plot_run_curves.py`, report section) once
   the two CSVs are on the laptop; both tilt charts are already drawn
   (`docs/figures/RT-17{4,5}_success_by_tilt.*`). RT-173's chain is DONE.
   The CSVs are also what closes P7 and the six-metric line on BOTH runs.
   (c) **DONE 2026-09-07 (D-173): `success_by_lateral_bin` and
   `success_by_yaw_bin` are wired.** Both go through `_rate_by_bin`, both
   buffers are filled in `_log_finished_episodes` beside the tilt line, and
   both keys are in `demo_metrics.json`. Edges: lateral (2, 4, 6, 8) mm on
   the HYPOT of the two pocket-axis components, yaw (2, 4, 6, 8) deg on the
   MAGNITUDE; last bin open at the top; the edges are ladder rungs and D-173
   says so. Offline green: `check_env_wiring.py` ALL 161 CHECKS PASSED
   (5 new), `--self-test` COUNTER-PROOF PASSED with the two new mutations
   `yaw-bin-table-reads-the-tilt-buffer` and `lateral-buffer-never-filled`,
   `selftest_checks.py --offline` PASS. **UNVERIFIED:** no run has produced
   either table. The FIRST run that does is the one that makes RT-174's
   answer readable per offset; RT-174 and RT-175 themselves can never get
   one, and that is recorded, not repaired.
   The user's target set (2026-09-07): random tilt + lateral + yaw
   (<= ~6 deg), then reduced play (D-087).
   BEFORE any yaw run: RT-152's landscape is unmeasured and the gate's
   yaw window is the square proxy's (§ 2).
   **SAC: the user wants a SAC comparison run; D-120 (supervisor,
   2026-08-28) says NO SAC training run, written comparison only. A
   conflict to resolve by decision, not by running.**
   **SEEN BY EYE, NOT MEASURED (user, 2026-09-07 ~02:00, `play.py` replay
   of RT-172 `model_600.pt` with tilt 0..8 deg and lateral 6 mm, 16
   envs):** one episode touched the rim, lifted, re-approached, lifted
   again, rotated about z, aligned, then inserted cleanly. A retry-and-
   search behaviour under free yaw. Nothing on file. To make it a
   finding: the three-step trace recipe (`play.py --trace-obs` WITH the
   same `env.*` overrides, `summarize_trace.py`, `trace_outcome_split.py`)
   on the RT-174 checkpoint -- rule or one-off.
   **MODE A, still the user's priority (2026-09-07 ~02:10).** In the same
   replay the rim-parked, frozen failure (RT-158: part over the 8.52 deg
   clamp, 10.22 deg median, ~15 N, no motion) was seen "often". Step (1)
   is now answered: training on the lateral start took the 6-9 deg tilt bin
   from 0.9852 to 1.0, so mode A is at most rare in the TRAINED
   distribution -- but the replay ran on the RT-172 checkpoint, NOT this
   one, and nothing counted it. Step (2) stands: trace the RT-174
   checkpoint (recipe above) and COUNT mode A per first episode
   (`trace_outcome_split.py`); (3) only then the fallback of § 2d, the
   alignment term, with the trace count as its target.
1. **RT-168 IS CLOSED (2026-09-06). `osc_kp_rot` IS RETIRED AS A SUSPECT.**
   The righting probe ran (`VERDICTS.md`, five lines, git `27c28bb`). The
   tilt fell to the 8.52 deg clamp at EVERY gain -- 8.5212 / 8.5210 / 8.5208 /
   8.5206 deg at 30 / 50 / 100 / 200. A 6.7-fold gain change moves the end
   state by 0.00065 deg; it only changes the SPEED (38 steps against 20).
   "The rotation stiffness is too soft to command the recovery" is therefore
   FALSE, measured. Do not re-open it without a new measurement.
   **THE PROBE'S OWN CONFOUND, and it is reported, not buried:** `force_n`
   read 0.0003-0.0007 N at the end of every rung and `tip_z` +17.5..+21.5 mm
   ABOVE the opening plane. There was NO CONTACT. The real mode-A failure
   reads 15.4 N. Copying the jam's TIP POSE does not reproduce the jam, so
   the probe righted a FREE part. "Stiffness UNDER contact" is therefore
   still unmeasured, and this probe cannot ask it. Both registered readings
   that survive (1 and 3) point the same way: the line is closed.
2. **THE DIRECTION REVIEW IS SETTLED (user, 2026-09-06): the CURRICULUM RUNG
   IS THE NEXT TRAINING RUN.** The reasons, each read back off a file:
   * The +40 mm failures live OUTSIDE the trained band. D-163 sets the
     rung-0 start to +30 mm as the UPPER bound, RT-156 trained -30..+30 mm,
     and RT-157h30 read 100 % (613 episodes) at a fixed +30 mm. Modes A and B
     were only ever measured at +40 mm and +60 mm. Inside the band no tilt
     failure has ever been observed. The tilt line was chasing a
     generalisation gap.
   * The plan's own branch, registered BEFORE the run
     (`rt_logs/RT-156_expectation.md:100`): "top bin > 0.20 -> the reward is
     sufficient; next is the curriculum." RT-156 read **0.9813**, 4.9 times
     the trigger. RT-156's FAIL sits on P11 (force p95 27.27 N against a
     25 N line) and that line is 50 N since D-169; the branch trigger is
     untouched by it.
   * `fixture_tilt_noise_rad` = 0.0 (`insertion_env_cfg.py:607`),
     `fixture_yaw_noise_rad` = 0.0 (:572), `curriculum_enabled` = False
     (:838). The TILTED APPROACH -- the task D-163 and the thesis are about
     -- is not being trained at all yet.
2b. **CORRECTION, SAME DAY (RT-170, 18:02:50, git `27c28bb`): the CURRICULUM
   IS NOT THE NEXT RUN AFTER ALL, AND IT CANNOT BE.** Point 2 above was
   written before RT-170's verdict landed. Two reasons, both read off files:
   * **The ladder cannot be switched on.** `rung_step_sizes` is `None` and
     carries `[open] D-110 point (3)`; `curriculum.Ladder` refuses to build
     without it, and `_check_rl_config` raises `RlConfigIncomplete` either
     way (set while OFF, or missing while ON). D-110 (3) says the step sizes
     come from THIS project's own learning curves, which do not exist. The
     final-rung ranges are `[CAD pending]` / `[verify on site]` (D-110 (4)).
     Nobody can start this run today without inventing two numbers.
   * **The reward is nearly blind to tilt, measured** (RT-170 BEFUND 2).
     At h -6 mm `kernel_sum` falls 2.482100e-01 (1.0 deg) -> 2.481998e-01
     (2.0 deg) = 1.02e-5 per degree, against 2.35e-4 per mm in depth. Per
     ACTION STEP (0.097 rad = 5.56 deg against 0.02 m) descending is worth
     ~82x more than righting. Recomputed independently here:
     20 mm * 2.35e-4 = 4.70e-3 against 5.56 deg * 1.02e-5 = 5.67e-5, ratio
     82.9. Turning tilt noise on before this is fixed trains a task the
     reward does not pay for. THE REWARD IS NOW THE NAMED SUSPECT.
     **TWO CORRECTIONS, same day, read off the code, both owned by
     `rt_logs/VERDICTS.md` 2026-09-06 -- do not restate the numbers here:**
     (i) the 8.11 deg `gate_min_alignment` gate is NOT in the paying path.
     It lives only in the `_peg_geometry` INSTRUMENT and in a check-script
     branch. The paying predicates check depth, interpenetration and the
     D-157 box, and NO angle. So there is no reward cliff to ramp; the
     missing gradient is real, the cliff is not. This is a SHAPING problem,
     not a specification problem. (ii) the 82x is the KERNEL share alone --
     D-165's progress term pays 425x more per 20 mm step inside the box.
2d. **THE ALIGNMENT TERM IS DROPPED (D-172, 2026-09-07); what follows is
   its review, kept as the FALLBACK for a rung that fails under the
   unchanged reward.** Original framing: RT-152 IS SKIPPED; THE ALIGNMENT
   TERM IS NEXT (user decision,
   2026-09-06, overriding the merge order of the same day: review, then
   build). RT-152 stays a thesis-must-state limitation
   (`InBachelorErwähnen.md`, 2026-09-05) and keeps no number. The term goes
   on `p5-reward` in `Phase3_Implementierung_v3` as RT-171: potential-based
   tilt shaping, `F = gamma*Phi(s') - Phi(s)`, `Phi = w*sech(a*theta)`
   against the POCKET axis, `a` from the D-109 (9) rule. Plan file
   `weiter-aus-dem-vorigen-squishy-bird.md` (laptop, not in git), § 9 holds
   the review.
   **REVIEW DONE 2026-09-06 (Fable 5.1), read off the code, no run.** The
   proposal stands with seven corrections:
   (i) SIGN. Seated = part +z ANTIPARALLEL to pocket +z, so
   `cos theta = -axes_in_pocket[:, 2, 2]` (the minus of
   `insertion_env.py:1238`). The plan's "+z against +z" is wrong by a sign.
   (ii) MARGIN -- DECIDED 15.34 deg (user, 2026-09-06), D-171. With margin 8.52 deg the pull per
   0.097 rad step is 0.194 w at 8.52 deg but 0.018 w at 15.34 deg, the
   RT-158 mouth-mode median, i.e. below the 0.08/step contact penalty at
   the pose the policy actually fails in. 8.52 deg was a FREE part's rest
   (RT-168 confound). Alternative: margin 15.34 deg (RT-158 measurement):
   pull 0.108 w at 15.34 deg, 0.370 w at 8.52 deg.
   (iii) RESET. Isaac Lab runs NO forward between `_reset_idx` and
   `_get_observations` (`direct_rl_env.py:398-410`), so `Phi(s_0)` must not
   be read from `body_quat_w` in `_reset_idx` (the documented stale case,
   `insertion_env.py:1673`). CORRECTED AT BUILD TIME: the reset commands
   NO orientation at all (`_solve_start_pose`: "the command carries a ZERO
   rotation delta"), so there is no sampled tilt to read. Built instead:
   `Phi(s_0)` is taken from the RESET OBSERVATION -- `_get_observations`
   writes `_tilt_phi_prev` from the very quaternion the policy is shown,
   on every call, so the first step of an episode reads the s_0 the policy
   conditions on, stale or not. Whatever staleness that observation has,
   `Phi(obs_0)` does not depend on this episode's actions, so the
   telescoped sum stays action-independent. RT-171 P4 bounds the logged
   row to catch a phantom anyway. Masking the first step instead is
   WRONG: the sum then keeps `-Phi(s_1)`, which pays 0.72 w for one 0.097
   rad tilt on step 0.
   (iv) GAMMA. The env needs gamma; its one home is
   `rsl_rl_ppo_cfg.py:47`. BUILT: `cfg.shaping_gamma` defaults to
   `PPORunnerCfg().algorithm.gamma` read at import from that file, the
   guard refuses a value outside (0, 1], and the startup report prints it.
   NAMED GAP, not closed: a hydra override of `agent.algorithm.gamma` does
   not reach the env; such a run must set `env.shaping_gamma` too.
   (v) LUMP. The row stays OUT of `step_task` (`insertion_math.py:667`),
   like `progress`, or the success step pays `-w * steps_remaining`.
   (vi) The four-trajectory ordering test of CLAUDE.md § Testing did NOT
   exist (`check_insertion_math.py` had only the undiscounted lump identity
   at :809). BUILT, discounted, five synthetic trajectories (expert / hover
   / jam / idle / tilt-cycle) at w_tilt 0.1, 1.0, 10.0 -- see the RT-171
   block of that script.
   (vii) RT-168's "38 steps" are CONTROL steps (`tilt_recovery_probe.py:316`,
   one `env.step` per hold step), so the lower bound 0.3516 stands and the
   plan's UNKNOWN (a) is closed.
   Numbers recomputed independently: a = 20.129 /rad, Phi(8.52) = 0.100,
   one righting step at the cone edge pays 0.522 w, gamma^256 = 0.0763,
   lump at step 50 = 3.0 * 205 = 615 (peak kernel sum 1.0 + two bonuses).
   RULING on the three-axis clause: DISCHARGED for this ONE row, as it was
   for D-168 -- the row does not touch the kernel, targets the axis RT-170
   measured, and is yaw-invariant. The kernel rebuild stays blocked.
   Everything UNVERIFIED until RT-171 runs.
   **BUILT 2026-09-06 on `p5-reward`, laptop, UNVERIFIED under Isaac.**
   Files: `insertion_tasks_cfg.py` (`TILT_MARGIN_RAD`, `KERNEL_A_TILT`),
   `insertion_math.py` (`tilt_cos_theta`, `tilt_potential`, the
   `tilt_shaping` row -- last in `REWARD_TERMS`, outside `step_task`,
   `Phi(s') = 0` on the success step), `insertion_env_cfg.py` (`w_tilt`
   -- default 0.0 SINCE D-172, was 1.0 as built; `env.w_tilt=1.0`
   revives the term --, `kernel_a_tilt`, `shaping_gamma`, three guards),
   `insertion_env.py` (`_last_tilt_phi` from `_get_dones`,
   `_tilt_phi_prev` from `_get_observations`, the reward call, a startup
   line), `check_insertion_math.py` (29 new checks: measurement,
   potential, row, telescoping, farming, the discounted ordering test;
   three new source mutations and two cfg mutations),
   `check_env_wiring.py` (one mutation repointed),
   `check_seated_success.py` (pins `w_tilt = 0.0` like
   `w_depth_progress`). Laptop results: `check_insertion_math.py` 243
   green (214 before) and its counter-proof green; `check_env_wiring.py`
   156 green and its counter-proof green; `check_seated_success.py
   --self-test` 76/76. Expectation BEFORE the run:
   `rt_logs/RT-171_expectation.md` -- the run is the RT-172 recipe on the
   `_v3` clone plus the row, and it waits for RT-172 to finish (one
   editable install). Decision: D-171.
   **RT-172 MEASURED THE BASELINE (2026-09-06, judged 2026-09-06/07): P5
   BRANCH 1, read off `docs/figures/RT-172_demo_metrics.json` and
   `VERDICTS.md`.** The RT-156 checkpoint, fine-tuned under the reward AS
   IT STANDS (contact 0, far 0, no alignment term) with the pocket tilted
   0..5 deg and starts sampled +30..+50 mm, reads success 1.0 in BOTH
   tilt bins (0-3 deg n 1224, 3-6 deg n 776), cumulative 0.9912 over
   281768 episodes, force p95 26.54 N (RT-156 27.27), 0 aborts at 50 N,
   sigma 0.81 -> 0.57, user stop at iteration 632/2250 with the curve
   saturated from ~450. The pre-registered branch text therefore applies
   VERBATIM: "the current reward pays enough for a 5 deg pocket through
   the success lump and D-165's progress term. The alignment term
   (RT-171) then needs a motivation other than RT-170's kernel ratio, and
   the next question is the magnitude ladder, not the reward."
   WHAT THIS DOES NOT SAY, read off the instruments: (a) P4 (was the
   checkpoint worse at 3-6 deg BEFORE fine-tuning) is NOT BELEGT -- the
   tilt-bin table exists only in the end window, not per block; so the
   0.0070 -> 0.5058 first two blocks are not a per-tilt start value.
   (b) tilt x height is not crossed, and the height table has ONE open
   bin for the whole +30..+50 mm band (P4b NACHTRAG). (c) 5 deg is one
   rung without a source of ours (2c); nothing above 5 deg is measured,
   and the controller cone is 8.52 deg. (d) The mouth-mode failures of
   RT-158 (15.34 deg part tilt) were measured at +40 mm on a LEVEL pocket
   with the -30..+30 mm checkpoint; RT-172 trains +30..+50 mm and reads
   1.0 there, so those modes are gone under fine-tuning within the band
   -- whether the alignment term would help a HARDER rung is untested.
   RULED: D-172 -- RT-171 dropped, ladder next, rung value user-set.
2c. **THE TILT MAGNITUDE HAS NO SOURCE OF OURS.** The env comment
   "Ladder of D-037: 5 deg, then 15 deg" cited
   [QUOTE DEAD 2026-09-12: that sentence no longer exists anywhere in the
   repo. It sat on `fixture_tilt_noise_rad` and was rewritten to name
   D-178's 10 deg one-sided ceiling; the old line number `:606` had also
   drifted. THE POINT BELOW STILL STANDS -- the tilt magnitude still has no
   source of ours, and D-178 set 10 deg by user decision, not by
   measurement.]
   the OLD REPO's D-037 -- this repo has no D-037 (`grep "^## D-037"`
   returns nothing). Old D-037 picked 5 deg because old D-036 measured the
   PROXY policy at 100 % at 0 deg and 50 % at 5 deg. That is a proxy
   number and a proxy reason: different contour, ~7x tighter play, ONE valid
   yaw pose. It is a PATTERN, not a source. What this task needs is
   D-110 (4)'s cell-derived range, and that measurement does not exist.
   One hard bound of OURS does exist: `osc_tilt_clamp_rad` = 8.52 deg from
   CAD ("Einfaedeln"). A 15 deg fixture tilt is outside what the controller
   may even command, so the old ladder's second rung is not usable here.

3. **THE 44-65 BAND FOR `osc_kp_rot` IS NOT A NUMBER. Do not quote it.**
   Broken down 2026-09-06:
   * The moment budget (20 N at half the part width) has NO SOURCE. The 20 N
     of D-166 is a SEARCH force for pushing DOWN. Nothing says a righting
     moment equals that force times a lever. Own construction.
   * The denominator is a placeholder: `osc_rot_step_limit_rad` = 0.097
     carries "[placeholder] ... Factory's rot_action_threshold".
   * The lower end does not recompute. With `Lambda_rot` 0.22806 and 0.097
     the denominator is 0.022122. `POCKET_WALL_Y` 0.07255 gives 65.6 (the
     upper end checks out); `POCKET_WALL_X` 0.0452938 gives 40.9 and
     `POCKET_NOTCH_X` gives 42.6. Neither is 44, and NO file holds the
     derivation. The position rule itself is fine (20/(8.82*0.02) = 113.4).

Older framing (kept for detail): `Pläne/Plan-Merge.md`. The Plan-Merge STOP RULE HAS FIRED
and its diagnosis phase is now PARTLY CLOSED. RT-151c (y) and RT-153 (x)
both measured the same answer: no usable lateral gradient near the
centre. RT-152 (yaw) is the one axis still unmeasured and is DEFERRED,
not cancelled — see 2.

The "no reward code until all three axes are measured" clause is
DISCHARGED for one specific change and one only: D-168 (2026-09-05)
changes `action_rate_scale` 0.1 -> 0.0034. That is not a landscape
change. RT-151c/RT-153 measured what the reward pays for POSE; D-168
fixes what it charges for NOISE, which is a different failure and was
measured by RT-148b, not by the curve sweeps. The kernel is untouched.
The potential-based rebuild of the kernel — the change the three-axis
clause is actually about — stays blocked and stays a separate
hypothesis.

1. **RT-154 — the sigma-tax run. Code is BUILT, the run is due.**
   `rt_logs/RT-154_expectation.md` holds the command, the budget and the
   PASS/FAIL lines; it was written before the run. One change:
   `action_rate_scale` 0.1 -> 0.0034 (D-168), plus two new instruments
   (`action_abs_max_mean`, `action_sat_frac`) that change no number the
   policy sees. Laptop state: `py_compile` OK on both edited files,
   `scripts/selftest_checks.py --offline` PASS, derivation recomputed
   independently (`E[chi_6]` = 2.3500, episode factor 849.8, old scale
   -> 85.0, new -> 2.889 = 5.07 % of 57). **UNVERIFIED — nothing has run
   under Isaac.** The discriminating measurement is the sigma trajectory:
   PASS above 0.30 at iteration 500, FAIL at or below 0.05. Success is
   PREDICTED to stay 0; the kernel is still flat.

   RT-153 is CLOSED — see `rt_logs/VERDICTS.md`, the 2026-09-05 entry.

2. **RT-152 — the yaw slice. DEFERRED 2026-09-05, behind RT-154.**
   Not cancelled and not answered: yaw is the ONE axis of the reward
   landscape with no measurement at all, and RT-149 showed the policy
   holding +25 deg of it. Logged as a thesis-must-state limitation
   (`InBachelorErwähnen.md`, 2026-09-05). Why behind RT-154: the sweep
   measures a landscape, and RT-148b showed the policy never got to
   explore that landscape at all. Fixing the exploration first is the
   cheaper order. CORRECTED 2026-09-05: this does NOT need
   new mathematics. `insertion_math.yaw_cos_sin` (`:788`) is already "the
   yaw of the part's own x-axis about the pocket z-axis", observation
   channels 19:21, and `scripted_policy.yaw_error` (`:165`) gives the
   signed wrapped difference. Both are self-tested. What is missing is
   wiring only: a `--curve-yaw-deg` flag, `d_pocket[:, 5]`, and a SECOND
   drift measure in DEGREES with its own tolerance — it must NOT be folded
   into `pose_drift_mm`, which is millimetres, or the gate hides an
   implicit length scale. `TILT_AXES` = {"x": 0, "y": 1} and
   `signed_tilt_angle` stay untouched: that function measures the tilt of
   the part's z-axis and is invariant under rotation about it, so it can
   never express yaw. The pocket z-axis itself is fine and in daily use
   (`d_pocket[:, 2]`, `:1875`) — it is the rotation ABOUT it that has no
   sweep.
3. **Blockbericht Reward** (`docs/Blockberichte/Blockbericht_Reward.tex`,
   Plan-Merge Schritt 1, outline in `Pläne/Plan1.md` Deliverable 1) — after
   1 and 2; it must cite files, not prose. Step 0 is now CLOSED: the
   RT-148b metrics file, log excerpt and TensorBoard tag list are all in.
4. **Reopen D-116.** rsl_rl 3.0.1 logs NEITHER explained variance NOR KL —
   measured, not assumed: the RT-148b run's TensorBoard has 30 scalar tags
   and neither appears (nor clip_fraction, nor grad_norm). `CLAUDE.md`
   § Training demands all six metrics in order; two of the six do not
   exist. `Loss/learning_rate` IS logged and the config is
   `schedule="adaptive"`, `desired_kl=0.01`
   (`rsl_rl_ppo_cfg.py:44,48`), so the learning-rate trace is an INDIRECT
   KL readout — the exact switching rule is UNVERIFIED, rsl_rl is not on
   the laptop.
5. Then, concept stream, not here: `F_search` inside the 1–20 N band
   (Decision (4) input; kp 100 is IN by the rule and SEATS by RT-150),
   episode length at 15 Hz (audit section 6 (5)), observation channels.

### Waiting on the user / supervisor

- **Decision (2) supervisor question:** "Gravity stays ON. The controller
  becomes task-space impedance in torque mode. For that the drive
  stiffness and damping must be 0; maxForce stays. Agreed?" — to be asked
  WITH the RT-143/RT-144 numbers. Until answered nothing in the concept is
  fixed; the code default is osc regardless (Decision (1)/(6)).

### Live gaps and reminders (not next steps, must not be lost)

- **The published rate and episode anchors** (read at source 2026-09-04,
  full table in `docs/reference/literature_check_policy_rate_osc_2026-09-04.md`):
  Isaac Lab Factory `decimation = 8` at `dt = 1/120` -> 15 Hz policy,
  120 Hz controller (`factory_env_cfg.py:72,99`), episode
  `peg_insert` 10.0 s / `gear_mesh` 20.0 s / `nut_thread` 30.0 s
  (`:195,202,209`); AutoMate assembly 15 Hz, episode 5.0 s
  (`assembly_env_cfg.py:72,105`, with the 10.0 s line commented out above
  it). WE RUN 17.07 s — 1.7x Factory's peg insertion, 3.4x AutoMate's.
  The difference is a CONVENTION one: they fix the SECONDS and let the
  step count follow, D-113 fixes 256 STEPS and lets the seconds follow.
  IndustReal states 60 Hz control / 120 Hz physics and is the only one of
  the six that JUSTIFIES its rate (deployment aliasing and compute, not
  physics); FORGE states 15 Hz policy / 1000 Hz controller without a
  reason. No source gives a LOWER bound, and none ties a bound to contact
  stiffness or solver substeps.
- **Factory and AutoMate smooth the action with an EMA, `ema_factor = 0.2`
  (`factory_env.py:213`, `factory_env_cfg.py:51`, `assembly_env_cfg.py:48`).
  WE HAVE NO SMOOTHING.** Never decided, never noticed until 2026-09-04.
- **The "we run 60 Hz (D-024)" line in
  `docs/reference/literature_check_reward_2026-08-26.md` § 6 is stale, not
  wrong.** It was written before the controller switch (2026-09-03): under
  `joint_pd` `decimation = 2` is 60 Hz, under `osc` `osc_decimation = 8`
  is 15 Hz. Both were true at their time. Cite the mode with the number.
- **The `--episode-steps` flag on `tilt_insert.py` is UNVERIFIED beyond
  RT-147.** It overrides `cfg.episode_steps` before `gym.make` and is safe
  only because that script sets `rl_terms_enabled = False`, which skips
  `validate_rl_config`'s `episode_steps` / `time_penalty_per_step`
  identity check. Never copy it into a training script without also
  rewriting `time_penalty_per_step`.
- ~~**The tilt cone is measured from the WORLD vertical**, not the pocket
  axis.~~ **CLOSED since Phase 5 step B7 part 1, struck 2026-09-12.**
  `clamp_tilt_to_cone` measures against THIS EPISODE'S POCKET AXIS —
  `insertion_math.py:1441-1443` and `:1473-1475` (`ref = axes_from_quat(
  ref_quat)`), header "WHY THE POCKET AND NOT THE WORLD" at `:1450`, and
  `Belege_Streuwerte.md` C4 says the same. Two other places in THIS file
  already said so (`§ Frozen` and `§ State`); this line was the third home of
  one fact and contradicted the other two.
- **`gravity_compensation` uses PhysX's forces for the whole articulation
  including the welded tool.** Whether that is exact for the fixed-joint
  chain is what RT-143 P3 measures; no source says so.
- **Under zero action the OSC has no restoring spring ONLY INSIDE the tip
  box** (target = current pose each physics step). OUTSIDE the box
  `clamp_tip_in_box` makes the target a fixed point 50 mm above the
  entrance and the OSC pulls hard toward it — that, not droop, is what
  RT-143 measured (VERDICTS NACHTRAG 2026-09-04); the no-spring hold is
  measured by RT-145 (§ State).
- **`applied_torque` under joint_pd is the actuator model's estimate**, not
  a PhysX read-back (`articulation.py:_apply_actuator_model`). Named in
  RT-144.
- **Scripted descend rates are per CONTROL step and were sized for
  joint_pd.** Under osc they are velocities (§ State); `tilt_insert.py`'s
  `--episode-steps` (verified by RT-147) and `scripted_insert.py`'s
  `--descend-rate 0.02 --max-step 0.02` (RT-150) are the working forms.
  `--decimation` never holds a descent rate equal under osc, it only
  changes the episode seconds (RT-144d/146, `PROBLEMS.md`).
- **Observation question (archive 2026-09-03):** `joint_pos`/`joint_vel`/
  `ee_quat` (16 dims) are proxy carry-over with no channel reason;
  `yaw_cos_sin` is a re-encoding of quaternions; `pocket_quat` is constant;
  `ee_quat` is not sign-canonicalised; ~~D-114's noise-injected force channel
  is not implemented~~ — **IMPLEMENTED since D-182**, a per-step Gaussian of
  3.5 N on the force channels (`insertion_env_cfg.py:1020`,
  `insertion_math.noise_masks`), struck 2026-09-12. Section 6 (5) re-derives
  the observation after the controller measurement.
- **`play.py --checkpoint` wants a full path, `train.py` a filename**
  (`PROBLEMS.md`, RT-140p, repeated in RT-149's first attempt). No
  per-episode PART attitude is logged in training; `play.py --trace-obs`
  + `scripts/summarize_trace.py` is the instrument (no tilt in the trace
  yet — add `ee_quat` 15:19 if RT-149's 12.5 mm at y -31 mm needs it).
- **`force_norm_n` in `demo_metrics.json` is an EPISODE MAXIMUM and reads
  the RESET TRANSIENT** (2.021 N in every env, RT-149 DETAIL (c)); it says
  nothing about task contact. Instrument defect, not fixed; home VERDICTS.
- **Whether rsl_rl 3.0.1 logs explained variance / KL to TensorBoard is
  UNKNOWN** (rsl_rl is not on the laptop; the console prints neither).
  § Open NEXT 2 reads the tag list. Value drift (D-116) is not excluded.
- **Anchoring form (Factory: target = current pose + delta) vs IndustReal
  PLAI (target = previous target + delta)** — concept-stream question,
  sourced in `docs/reference/literature_check_policy_rate_osc_2026-09-04.md`
  § 2.2; not a blocker for measurements on the current controller.
- **`compute_rewards_insertion`'s docstring attributes -50 to
  Beltran-Hernandez; the paper pays -10** (D-114 carries the correction;
  the docstring does not). The SUCCESS lump is undiscounted at γ = 0.99
  (factor ~2.6, "slightly preferred" is wrong) — own entry, not started.
- **The action-rate scale 0.1 is FORGE's**, sensitivity list; under osc the
  action is a pose delta and the term's meaning changed with it — nobody
  has re-read it.
- **Physics anchors (audit K4):** solver 16/1, contact/rest offset PhysX
  default, unauthored inertia — force peaks are not separable from solver
  artefacts in any run, PD or OSC.
- **H3** (fine reorientation near the wall) has no instrument. **The sign AT
  the surface is unmeasured** (D-152/D-153). **P5 has no emptiness guard**
  in `check_insertion_sdf.py`. **`add_fixture_collision.py:160`** duplicates
  the edge-pairing rule. **Exit-code trap:** an import-time failure bypasses
  the D-081 handler in every Isaac script (`PROBLEMS.md`).
- **Two R2 design decisions have no D-number:** the observation-only
  information boundary and the `dls` IK route (now the reset IK only).
- ~~**Observation-noise channels are undecided** (D-111 (3)); irrelevant while
  the switch is off (supervisor question F1).~~ **CLOSED by D-182, struck
  2026-09-12.** D-182 corrects D-111 (3) in its own words: the FORM stands, a
  per-step force channel is added, and the activation NO LONGER WAITS for
  supervisor question F1. Both sigmas default non-zero and ON
  (`insertion_env_cfg.py:1019-1020`).
- **Global `HANDOFF.md` does not yet point at this stream** (edited only on
  `main`, D-066).
- **`insertion_math.py` docstrings quote three numbers D-121 superseded**
  (prose only). **The real fixture has NO bounding-box orientation check.**
  **Petrovic 40-Hz warning vs. our 60 Hz (D-024)** — now moot for the
  policy (15 Hz placeholder) but the CONTROLLER runs at 120 Hz.

## Eingang vom Konzept-Strang (live constraints; full texts in the archive)

Block 9 (hyperparameter study, D-117…D-120) starts only AFTER the first
full training run is evaluated (D-117 a). Until then "do not break":

1. `scripts/rsl_rl/train.py` lines 169/174 (the two logging print lines)
   stay as they are — D-119 makes them load-bearing.
2. `success_rate_recent` in `demo_metrics.json` keeps its name and meaning
   (D-118's objective).
3. `scripts/compare_runs.py` stays runnable (no Isaac, no GPU).
4. `--stop-at-success-rate` stays (D-118's tie-breaker).
5. Hydra CLI overrides for the agent config: `[verify on site]`.
6. The forbidden list (D-117 c): the study never varies `action_scale`,
   reward weights/kernels, curriculum rungs, friction, the 256-step
   episode, or the env count. (`action_scale` is the joint_pd scale; the
   OSC step limits are placeholders OUTSIDE the study until decided.)

Open numbers this stream produces: `[measure]` wall-clock of ONE training
run at the final env count; `[measure]` `max_iterations`; `[verify on
site]` the Hydra key path and `ray/tuner.py --run_mode local` (D-119);
`[verify on site]` whether `optuna` installs in the Isaac Sim Python env.

Do not change `use_clipped_value_loss` or `entropy_coef` outside the study
(both are contradicted by literature and are IN the study, D-117).

## Rules for this file

- This file answers ONE question: what must a fresh session know to
  continue correctly. It is not a diary.
- Run verdicts → `rt_logs/VERDICTS.md`. Solved problems → `PROBLEMS.md`.
  Decisions → `DECISIONS.md` / `docs/decisions_inbox.md`. Session
  narrative is NOT written anywhere in this file — git history and
  `docs/archive/` exist.
- A closed item is DELETED here, not marked CLOSED; its content lives at
  its home. An item needed against re-derivation goes to § `Frozen` as one
  line + pointer.
- At the end of every session: re-read the whole file and overwrite what
  the session made wrong — never only append.
