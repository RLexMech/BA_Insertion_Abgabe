# HANDOFF — Stream rl-code (only this stream writes here)

Last updated: 2026-09-28, dev laptop, branch `p5-rlgames` (worktree
`Thesis Workflow Phase 5/p5-rlgames`, branched from `p5_mitAktion`
`c92ad4b4`). Read the branch head with `git ls-remote origin p5-rlgames`,
never from this file. `p5_mitAktion` and `p5-kraftsensor` keep their own
copies of this file; this branch does not merge them.

**THIS BRANCH = D-200: training on rl_games with Factory/FORGE's yaml,
decided value by value (`Pläne/D-200_Bauplan.md` "Values"); the env is
unchanged; the rsl_rl path stays.** The evaluated AutoDR
arm (RT-222..RT-225 on `8299485d`, K1/K2/K3 met) is frozen. Start at § Open,
NEXT, "START HERE (p5-rlgames)".

Scope of this stream: the RL code for the UR5e insertion — observation,
action, reward, termination, curriculum (concept blocks 3–7). Scene work
belongs to stream szene-umbau; concept decisions to `p4-konzept`. Rules for
this stream: the worktree-local `CLAUDE.md` (gitignored by design,
`.gitignore:82`), present in `p4-reward`; copy it by hand into any new
worktree.

History: everything before session 17 lives in
`docs/archive/HANDOFF-RL_Archiv_2026-08-30.md`; the reward rework of
2026-09-01, the contact-penalty / distance-leash analysis (RT-139..RT-141)
and the RT-131 finding were moved verbatim to
`docs/archive/HANDOFF-RL_Archiv_2026-09-03.md`; the RT-143..RT-147 NEXT
narrative to `docs/archive/HANDOFF-RL_Archiv_2026-09-05.md`. Nothing there
is a next step; its commands must not be run again.

The 2026-09-28 archive pass moved the old header warnings, the Arbeitsplan,
the old clone paragraphs, the RT ledger, § State and the older § Open blocks
VERBATIM to `docs/archive/HANDOFF-RL_Archiv_2026-09-28.md`; its header says
what is where.

## Arbeitsplan

Milestones M0-M2 and Phase 5 steps A/B/C as written up to 2026-09-12:
`docs/archive/HANDOFF-RL_Archiv_2026-09-28.md`, § Arbeitsplan. The current
work is § Open, START HERE (p5-rlgames).

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

**The training PC's clone of this branch** (so no command ever ships with a
`<path>` placeholder):

    C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1

Clone recipe (sparse set, `assets` copied from `Phase5_MitAktion_v1`, NO
`pip install -e`): `rt_logs/RT-240_expectation.md` § 1.

Older training-PC clone paragraphs (`Phase3_Implementierung_v2`,
`Phase5_Robustheit_v1`, the planned `_v3`, `p5-reward`, `p5-osc-rot-gain`):
`docs/archive/HANDOFF-RL_Archiv_2026-09-28.md`, § Frozen.

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

- **Next free RT number: RT-245** (2026-09-29: RT-244 is the highest over
  every `origin/*` branch). RT-241 is skipped: `Pläne/D-199_Bauplan.md`
  names it as a D-199 pilot and it has no verdict.

The ledger as it stood on 2026-09-26:
`docs/archive/HANDOFF-RL_Archiv_2026-09-28.md`, § RT ledger. Verdicts:
`rt_logs/VERDICTS.md`.

## State (current, one line per fact)

The old State list (RT-172 period and older):
`docs/archive/HANDOFF-RL_Archiv_2026-09-28.md`, § State. The current state
is in § Open, START HERE (p5-rlgames).

## Open

### NEXT, in order — nothing skips

**START HERE (p5-rlgames, 2026-09-27). Read this block, then act.**

* D-200 (`DECISIONS.md`): training on rl_games with Factory/FORGE's yaml,
  decided value by value 2026-09-27 (plan, section "Values"; user "1B",
  "2A"). Plan and what is built: `Pläne/D-200_Bauplan.md`; cost map:
  `docs/reference/rl_games_switch_cost_2026-09-27.md`. Laptop checks done:
  yaml = Isaac Lab's Factory yaml except `name`, `max_epochs`, both
  `minibatch_size` (`diff --strip-trailing-cr`), both scripts compile,
  `rt_log.ps1` parses and copies a log into a new `logs/rl_games` run
  folder (scratch test).
* STATE on the training PC (`rt_logs/RT-240_expectation.md`, RESULT
  sections; `VERDICTS.md`): clone `Phase5_RlGames_v1`; rl_games `6b3534f2`
  was already installed; `ray` 2.52.1 added (freeze gate: only new packages;
  PROBLEMS.md); measured stack = the expectation's "Setup result, step 1".
  RT-240a trained 4 epochs, exit 0, checkpoint + AutoDR state, losses finite
  (FAIL only on the console block); RT-240b replayed it, PASS. 62-88 s per
  epoch, GPU peak 4.7 of 10 GB -> no memory reason for the network fallback
  (D-200 (2)); more envs possible, not decided (folder `CLAUDE.md` §8).
* UNVERIFIED under Isaac: the console-block fix (`8611fcd0`), D-200 (7),
  the rollout length for the episode record (`75f2d105`), and the replay
  folder fix of the migration audit (below). RT-240c/d checks all three.
* MIGRATION AUDIT DONE 2026-09-27 (rsl_rl path against rl_games path, code
  first). One real gap: `scripts/rl_games/play.py` gave the env the
  TRAINING run folder; RT-240b wrote into RT-240a's record (its L303).
  Fixed: `<run>\replay_h<mm>`, as `scripts/rsl_rl/play.py` (plan part 3).
  Corrected readings, no code: RT-240s1 P1b (count 2 = before any action on
  rsl_rl, after ONE on rl_games), P8 (`action_sat_frac` / `action_abs_max_mean`
  read the clamped action: 0 / 1 by construction), csv step axes (plan part
  2 (6), four axes), console blocks count from 1 (part 2 (7)), NOT BUILT
  list completed (part 2), `rt-check` labels for rl_games, L-19, L-20.
* CHANGE LIST FOR THE THESIS (user 2026-09-27, "maximal wichtig"):
  `docs/Aenderungen_seit_Planungsstand_2026-09-26.md` -- what changed since
  the planning state `ae5fd006`, with a "Warum" section. EVERY important
  change from now on gets a row under "Nachträge" (date, old -> new,
  source, one sentence why). Information only; Astra makes the figures.
* RT-240s1 STOPPED by the user at epoch ~107 (2026-09-28): outcome in
  `rt_logs/RT-240_expectation.md` (RT-240s2..s5 section, OUTCOME), files in
  `rt_logs/RT-240s1/`. `all_at_max` from epoch 55, lr never at a bound.
  Still rising at 100: `rewards/iter` 600.6 -> 623.5, `episode_lengths/iter`
  33.1 -> 28.9 (means over epochs 56-65 vs 91-100). D-201: budget 100,
  checkpoint every 25 (yaml). User was offered Factory's 200; kept 100 so far.
* VIEWS 2026-09-28 (RT-240s1 epoch-100 policy, no training; impressions, no
  counts): inserts almost always inside the trained band; with the true
  fixture spread 30 mm instead of 5 mm clearly worse -> reads as the actor
  locating the pocket from its joint positions (Factory's actor has none,
  `factory_env_cfg.py:77`); with the yaw belief x3 it jams before stage 2
  (estimate: stage 2 needs yaw within ~0.24 deg; torque not observed); after
  a failed try it often parks outside and does not retry; with `--stochastic`
  (the only change) it retries a little more -> partly a deterministic rest
  point, partly no learned retry. View commands:
  memory "Replay a policy after a run"; play options `--dr-table`,
  `--autodr-state` (`654f233c`, `d967de56`, UNVERIFIED beyond the user's eyes).
* D-202 + D-203 BUILT OFFLINE 2026-09-28 (D-204; plan and BUILD RESULT:
  `Pläne/D-202_D-203_Bauplan.md`); probes PASSED under Isaac (RT-242e..i),
  no training yet. Start ceiling 0.270 m
  (AutoDR step 25 mm, bins 50 mm); box 0.100 / 0.2915 m; home 0.400 m above
  the block top (tip 0.415 m nominal, same arm family; screener gate over the
  block along the whole reset descent); true fixture spread 0.05 m in x, y AND
  z (table column 17, `_c18` eval tables, `applied.csv` `pocket_dz_m`). Not
  added: start-yaw noise (the pocket yaw is it), physical grasp offset (D-183
  stays). Checks: anchors 600, `check_env_wiring` 435 + counter-proof,
  `check_autodr` 172 + counter-proof, `check_eval_conditions` 25/25,
  `make_eval_table` 23/23, eval pipeline self-tests green (`.venv-analysis`),
  `check_insertion_math` only its 9 old reds. OPEN from the build: the coarse
  reward kernel is flat at 270 mm (`InBachelorErwähnen.md` 2026-09-28).
  `check_env_wiring.MEASURED_HOME_TIP_M` = 0.415 (RT-242e).
* NEXT, in order:
  1. DONE 2026-09-28: plan step 7 PASSED as RT-242e..i (`rt_logs/VERDICTS.md`;
     expectation `rt_logs/RT-242_expectation.md`). Home tip measured 0.415 m
     (= nominal) -> `check_env_wiring.MEASURED_HOME_TIP_M`; start IK clean
     at 270/30/10/5, 270/30 with the 0.05 xyz spread, and -30 mm (4096
     episodes each); reset force 0.000 N at the high corner. FINDING: RT-143
     never measured the home pose (first line one step into the clamp
     pull); the "19.1 mm deficit" texts carry inline corrections (D-180,
     D-204 (1), `insertion_tasks_cfg.py`, `check_env_wiring.py`, plan).
     Fixes on the way (PROBLEMS 2026-09-28): `zero_agent.py` height flag sets
     `start_floor_m` equal; `--home-pose` spread off + `step 0` line;
     RT-242a..d INVALID because `zero_agent.py` imported the env of
     `Phase5_MitAktion_v1` (the editable install), fixed like
     `rl_games/train.py:109`. The same import line (`sys.path.insert` of
     `<clone>/source/insertion` + `[<tag>] insertion package imported
     from: ...`) went into the 17 other scripts that import `insertion`
     (2026-09-28; 15 under `scripts/` incl. `probe_assets.py`, plus
     `rsl_rl/train.py`, `play.py`); SKIPPED: `diagnose_tool_candidate.py`
     (SPENT, user). Line refs into seat_probe, verify_fixture_usd and
     rsl_rl train/play moved by 5; the refs into rsl_rl train/play that
     were already stale before that point at their code again (`053ced57`,
     comments only, mapped by content). Checks: py_compile, anchors 600,
     `check_env_wiring` 435 + self-test. UNVERIFIED under Isaac until a
     training-PC run of one of them prints `Phase5_RlGames_v1`.
  2. Then the five seeds RT-242s1..s5, budget per D-201. Expectation WRITTEN
     2026-09-28 (`rt_logs/RT-242_expectation.md`, section RT-242s1..s5; one
     chain, ~12 h): hypothesis = every seed `all_at_max` 1 by epoch 80
     (D-201's own clause). RAN 2026-09-28, FAIL on the hypothesis: first
     `all_at_max` 1 at 74/88/80/73/81 (s2, s5 late); everything else PASS
     (`VERDICTS.md`, OUTCOME in the expectation, Laufkatalog § RT-242s).
     D-201 falsified by its own clause; budget stays 100 for every arm
     (D-205, user 2026-09-28).
     `ur5e_insertion.pth` of s2/s5 is from epoch ~33/~22: view `ep_100`.
  3. Then the Fixed-DR arm RT-243s1..s5 (same command, only
     `env.dr_mode=fixed_dr`). Expectation WRITTEN 2026-09-28
     (`rt_logs/RT-243_expectation.md`). Gate passed: budget 100 (D-205).
     RAN 2026-09-28/29, hypothesis PASS (width held, all five). Learning:
     s3 0.992, s4 0.996; s1, s2 0 (timeouts, depth 0); s5 0.056 at 100 and
     STILL RISING -> D-201's Fixed-DR clause, user reads it (D-205).
     Figure: `docs/Blockberichte/RT-242s_RT-243s_F7_lernkurven.pdf`.
  4. The No-DR arm RT-244s1..s5 (same command, only `env.dr_mode=off`).
     Expectation WRITTEN 2026-09-29 (`rt_logs/RT-244_expectation.md`).
  5. Evaluation scope DECIDED 2026-09-30 (D-206): only the three rl_games
     arms, `ep_100`, T-final + T-grenze; OOD by D-191 (port from
     `origin/p6-TestAusserhalb`), AutoDR only; no MLP ablation, no mass, no
     real fixture; rear wall optional at the nominal fixture pose. BUILD
     PLAN approved 2026-09-30: `Pläne/D-206_Bauplan.md` (part A = training
     PC, one commit C; part B = laptop only; part C = docs, incl. the D-206
     inline corrections). PART A + C BUILT 2026-09-30 (plan, "Bau-Ergebnis";
     D-206 corrections 1-3): `rl_games/play.py --eval-table`,
     `rl_games/run_eval.ps1`, the OOD port (`allow_outside`, table
     `eval_table_ood_seed10_c18.csv`, start height x1.0), `T-ood` in BLOCKS.
     Offline checks green; UNVERIFIED under Isaac. Laptop dry run of
     `rl_games/run_eval.ps1` on `4b445714` against a fake run root: technik,
     T-auswahl, T-final-F, T-grenze, T-ood, T-final py2/pn2 all "checks
     passed"; five refusals STOP as meant (no/extra `-Final`, `-Noise` off
     T-final, bad `-Rt`, missing checkpoint + two folders for one seed).
     Expectation WRITTEN: `rt_logs/RT-245_expectation.md`. NEXT: RT-245 technik
     (3 runs, RT-242s1, `rl_games/run_eval.ps1 -Block technik`) after
     RT-244 has ended -- do not pull on the training PC before; then read
     G1-G8, the G5 name, the run time, and compare both `ep_100` files.
     Part B (laptop) can be built while RT-244 runs; B1 (`nn` identity +
     tag rule) is needed before `eval_collect.py` reads RT-245. B2 G3e +
     `py2`, B3 stats, B4 figures, B5 `.venv-analysis`.
* Not built: plan part 2, "NOT BUILT" (the one list).

Older START HERE blocks (`p5_mitAktion` 2026-09-26; `p5-kraftsensor`
2026-09-12 chat 3, with its numbered items):
`docs/archive/HANDOFF-RL_Archiv_2026-09-28.md`, § Open.

### Live gaps and reminders

- **Owed to Astra, for the writing assistant** (user 2026-09-28): the
  information for (1) D-202 and D-203, (2) a new `07_Parameter.md`, (3) a
  new `04_Ergebnisse_und_Abbildungen.md` with the rsl_rl results. It goes
  into `docs/Aenderungen_seit_Planungsstand_2026-09-26.md`; Astra updates
  `C:\Users\PCUser\Desktop\Claude\Bachelor\ausformulieren\unterlagen\` (not
  git; `04`/`07` as of 2026-09-26) herself. This stream does not edit those
  files. Stale in the list: D-202/D-203/D-204 still read "NICHT unter Isaac
  gelaufen" (RT-242e..i PASS since), and § 8 names only `Kapitelplanung\`.
- **`check_insertion_math.py` has 9 red cfg-guard checks** (its `_Cfg` stub
  lacks `dr_table_path` and friends; red on `p5-kraftsensor` `e5ee168e` too,
  2026-09-25). Its full `--self-test` refuses to start on a red baseline;
  named mutations were run by a scratch runner instead. Not fixed.
- **Stale pointer:** `insertion_env_cfg.py:252` cites Factory
  `factory_env_cfg.py:53` for the step limits; the values stand at `:56-57`
  (`:53` is `pos_action_bounds`).
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

Older numbered NEXT items (00, 0, 1-3, the older framing and its 1-5;
2026-09-05..10): `docs/archive/HANDOFF-RL_Archiv_2026-09-28.md`, § Open.

### Waiting on the user / supervisor

- **Decision (2) supervisor question:** "Gravity stays ON. The controller
  becomes task-space impedance in torque mode. For that the drive
  stiffness and damping must be 0; maxForce stays. Agreed?" — to be asked
  WITH the RT-143/RT-144 numbers. Until answered nothing in the concept is
  fixed; the code default is osc regardless (Decision (1)/(6)).

The second live-gap list (2026-09-04..12, controller-switch period, NOT
re-checked since): `docs/archive/HANDOFF-RL_Archiv_2026-09-28.md`, § Open.

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
