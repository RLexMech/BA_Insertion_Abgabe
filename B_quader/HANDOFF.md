# HANDOFF — single entry point

Last updated: 2026-08-16 (end of day), dev PC. This day closed the whole
orientation/generalisation section: D-036 measured and VERIFIED, D-037
(pocket tilt in the observation + curriculum) and D-038 (pocket yaw) decided,
trained and VERIFIED. All measurements were executed by the user on the
training machine and read from disk (`demo_metrics.json`). Human-readable
summary of the day with all numbers:
`docs/session_2026-08-16_neigung_versatz.md` (also delivered as PDF).

The previous entry (2026-08-06 night, dev PC) covered the full D-034 arc:
BLOCKED resolution, commissioning, randomised curriculum, generalisation
curve.

Update this file at the end of every work session: what changed, what is
verified, what is next. When updating, do not only append — re-read the whole
file and correct what the session's findings made wrong.

## Current state, one paragraph

Active branch: **`generalisation-angle-probe`**, fully pushed
(`generalisation-probe` holds the finished translation half,
`square-peg-insertion` the solved fixed-pose task). **The task is solved on the target geometry
AND robust to fixture translation.** The 30 × 30 × 50 mm square peg is inserted
into the 32 × 32 × 35 mm blind pocket — 1.0 mm clearance per axis, ±3.96°
free-yaw window — by the D-034 policy `2026-08-06_21-19-45_s30b32rxy50`
(trained with ±5 cm per-episode fixture-pose randomisation) at **99.65 %**
recent training success and **100 % replay success at every measured fixture
offset (0, ±2, ±5 cm; 250–1250 episodes per point)**. The negative control
holds: the old fixed-pose policy scores 0.39 % at +5 cm. D-035's question is
closed — the earlier 0 %-at-2-cm failure was location-specific and D-034
removes it across the trained range. The whole chain ran on 2026-08-06: the
confounded BLOCKED probe verdict was overturned from its own saved JSONs
(single-waypoint marginal-contact artefact at depth 0.0, PROBLEMS.md), the
offset-0 probe on the kinematic table is PASSABLE, the noise-0 regression
replay of the old checkpoint reached 99.76 % (kinematic table changes nothing
for the policy), and the randomised curriculum ran b = 45 → 36 → 32 at
99.85 / 99.95 / 99.65 % with near-lossless transfer (2052 episodes to 90 % at
the target rung vs 77 789 when trained at a fixed pose). Sample cost of the
randomisation at stage 0: factor 1.22. Fixed-pose baseline and its
commissioning: `docs/square_task_commissioning.md`. **The orientation half is
DONE (2026-08-16): the final D-038 policy handles per-episode pocket offset
(±2 cm), tilt (0–10°, random direction) and yaw (±45°) COMBINED at 99.19 %
recent success over a 5461-episode window (interactive replay with full
training noise: 99.57 % over 703 episodes), per-tilt-bin rates
0.996/0.992/0.989/0.989 with no edge collapse, code marker
`square-peg-2026-08-16-d038`. Observation is 25-dim (pocket quaternion in
channels 21:25, D-037); ±45° yaw is the full range by C4 symmetry. The
generalisation section of the proxytask is closed; next is the real task.**

## Verified on the training machine (2026-08-06)

Full numbers and commands: `docs/square_task_commissioning.md`. In brief:

- **Assets.** Table generated from nine box colliders at b = 45 and b = 32
  (10/10 self-checks each, contract bbox to 6.1e-09 m, zero composition arcs);
  peg authored with a `local_bbox_size_m` of 0.030 × 0.030 × 0.050 m, i.e. no
  45° mesh rotation, and cuboid inertias consistent with 1040 kg/m³ ABS.
- **Env.** Gate closed at the home pose at b = 32, both asset checks PASS
  against their generator sidecars, 21-dim observation, and channels 19:21
  reproducing (cos 4φ, sin 4φ) to four digits.
- **Reward.** 41/41 under real PyTorch.
- **Passability.** PASSABLE on both rungs; at b = 32 the yaw control reacts at
  6.30× its own contact-free noise, which is the measurement D-029 risk 1
  required — the squareness is physically effective.
- **Training (fixed pose).** Stage 0 (b = 45) 99.9 %; b = 36 100 % over 751
  replay episodes; target rung (b = 32) **99.15 %**, 77 789 episodes to the
  90 % threshold against 17 974 at stage 0.
- **Training (D-034, ±5 cm randomised).** Same ladder: b = 45 99.85 %
  (21 869 episodes to 90 %), b = 36 99.95 % (2069), b = 32 **99.65 %** (2052).
  Generalisation curve of the final policy: 100 % at 0, ±2, ±5 cm; fixed-pose
  policy at +5 cm: 0.39 % (control). Details in "Next steps" item 4's table
  below (to be folded into docs/square_task_commissioning.md or a follow-on
  doc when written up for the thesis).
- **D-029 risk 5 is refuted by measurement.** The 13 mm gravity droop is 13× the
  1.0 mm clearance and was the ranked first suspect, with D-014/OSC as the
  remedy. Joint-delta actions on the shipped PD gains reach 99.15 % through it.
  OSC was not required for this task; D-014 is not thereby retired.

Older verified baselines (increments 1–2, demo sprint, toolchain, two-machine
rule) are unchanged and listed in `docs/project_overview.md` and the increment
tags.

## Where the approach itself went wrong on 2026-08-06 (kept as lesson; the finding itself is since resolved — see below)

The BLOCKED finding below is confounded **by our own sequence**: the handoff's
commissioning order was (a) noise-0 regression first, (b) passability probe
second — what actually ran was the probe first, at +3 cm, on the kinematic
table. That changed two variables against the morning's PASSABLE in one
measurement (fixture offset AND static→kinematic swap), which is precisely
the one-change-per-step rule this project exists to enforce. Second lesson:
the probe operates in a different regime (hard-written joint states) than the
policy (position controller); a probe verdict was implicitly treated as a
statement about the policy regime. Third: the D-035 lesson (one run, one
file) had to be re-learned — the probe's default JSON path overwrote the
morning's PASS and then its own BLOCKED. The next session should re-check the
plan below against these three points before running anything.

## BLOCKED finding resolved (2026-08-06, read from the saved JSONs)

Measurement steps 1–2 of the previous plan are DONE, both read at zero cost
from files already on the training machine:

- `…\assets\Robot\passability_kinematic_offset30_BLOCKED.json` (the renamed
  default file — it was moved next to the peg USD, not copied to the repo
  root) and `passability_kinematic_offset0.json` (repo root of the training
  clone; the offset-0 probe had already been run).
- **Offset chain exonerated:** at +3 cm the centred column sat 0.74 mm from
  the pocket centre (`tip_xy_offset_m` ≈ 0.0007, inside the 1.0 mm
  clearance) and descended to the full 27 mm at ~1.05× its contact-free
  baseline. No `warm-up IK failed` anywhere.
- **Kinematic swap exonerated (probe regime):** offset 0 on the kinematic
  table is PASSABLE with clean separation — centred 1.05× vs controls
  4.24×/5.70×, verdict stable for any factor in [1.05, 4.24]; qualitatively
  the same picture as the morning's static-table PASS (1.05× vs 6.30×).
- **The BLOCKED verdict rests on ONE waypoint:** at exactly depth 0.0 the
  centred pass spiked once to 0.0116 rad (4.26× baseline) and was quiet at
  every deeper waypoint. A blocked pocket would react persistently and
  increasingly with depth; a single non-monotonic spike is not obstruction.
  The verdict's own stability interval was [4.03, 4.26] — a 5 % band, i.e.
  exactly the CLAUDE.md case "the threshold is the only thing between the
  data and the opposite conclusion".
- **Both runs are anomalous at exactly depth 0.0, in opposite directions:**
  offset 0 has damped motion there (0.0019 < 0.0027 baseline, tip_xy drops
  to 0.08 mm), offset 30 the spike. The commanded IK cannot differ there —
  the 0.05 rad branch-continuity gate in `ik_for_depth` rejects flips as
  unreachable, and neither JSON lists unreachable depths — so the anomaly
  is in the simulation response. **Mechanism CONFIRMED by the control-pass
  cross-check** (training machine, 2026-08-06): at depth 0.0 the peg bottom
  rim is exactly coplanar with the plate top — the PhysX marginal-contact
  edge case (zero separation to the four plate boxes around the opening).
  All six passes across both runs are anomalous at exactly depth 0.0 and
  nowhere adjacent — both offset controls drop to 0.0034 rad (neighbours
  0.0038–0.0059) with tip_xy pinned at the commanded 4.00 mm, both yaw
  controls drop to ~0.0035 rad with tip_xy at 2–6 µm (zero sag: the turned
  rim rests on the plate), centred at offset 0 is supported (0.0019). The
  +3 cm centred pass is the only one that kicked instead (0.0116) —
  marginal contact is discontinuous, support vs ejection depends on the
  arm configuration. Logged as a PROBLEMS.md row (2026-08-06); the probe's
  depth-0 handling (persistence criterion) is still to be implemented, see
  next steps.

## Completed 2026-08-06 (measurements on the training machine)

1. ~~Zero-cost cross-check of the depth-0 anomaly~~ **DONE**: all six passes
   anomalous at exactly depth 0.0 — mechanism CONFIRMED, see the resolution
   section above. The fix belongs in the probe (depth-0 handling), not in
   the env.
2. ~~Noise-0 regression replay of the b = 32 checkpoint~~ **DONE** (2026-08-06,
   training machine): **0.9976 over 1250 episodes**, mean max depth 29.0 mm,
   `fixture_pos_noise_xy_m: 0.0`, code marker `square-peg-2026-08-06b-d034`,
   read from `…\2026-08-06_02-25-24_s30b32\replay\demo_metrics.json`. On the
   level of the 99.15 % training result → **the kinematic table does not
   change the physics for the policy; D-034 commissioning is complete and
   training is RELEASED.** This run also verifies the plain `replay/` metrics
   path and the ed40da6 progress print (both previously UNVERIFIED).
3. **Curriculum with randomisation — DONE.**
   Stage 0: run `2026-08-06_20-33-51_s30b45rxy50`
   (fresh, 1024 envs, 500 iterations, ~4.5 min), **99.85 % recent** /
   93.2 % cumulative, 29.5 mm mean depth, **21 869 episodes to the 90 %
   threshold vs 17 974 without randomisation (factor 1.22)** — read from
   the run's demo_metrics.json, config fields verified (b = 45,
   fixture_pos_noise_xy_m 0.05). D-034 criterion ≥ 95 % met. b = 36 rung
   DONE (resume, iterations 500→1498): **99.95 % recent** / 99.65 %
   cumulative, 29.2 mm mean depth, **2069 episodes to the 90 % threshold**
   over 903 276 episodes total — near-lossless transfer from b = 45; this
   also recovers the sample-efficiency figure the first curriculum round
   lost. **Target rung b = 32 DONE** (resume from the b = 36 rxy50 run,
   stopped early at ~99 % stable): **99.65 % recent** / 98.2 % cumulative,
   29.2 mm mean depth, **2052 episodes to the 90 % threshold vs 77 789 at
   the fixed pose** — the randomised curriculum transfers between rungs
   almost losslessly instead of relearning at the target rung. All numbers
   read from the run directories' demo_metrics.json. **D-034's success
   criterion is met on the target geometry.**

   **Generalisation curve DONE** (2026-08-06 late evening, training machine,
   all points read from their own `replay_dx…\demo_metrics.json`; policy
   `2026-08-06_21-19-45_s30b32rxy50`, checkpoint model_3950):

   | Fixture offset (x) | Episodes | Success | Mean max depth |
   |---|---|---|---|
   | 0 (trained centre) | 1000 | **100 %** | 28.0 mm |
   | +2 cm (D-035 failure point) | 500 | **100 %** | 29.3 mm |
   | −2 cm | 250 | **100 %** | 29.6 mm |
   | +5 cm (training-range edge) | 1250 | **100 %** | 29.3 mm |
   | −5 cm | 501 | **100 %** | 28.3 mm |

   Control (offset really applied, same code path): the FIXED-pose policy
   `2026-08-06_02-25-24_s30b32` at +5 cm scores **0.39 % over 512 episodes,
   0.1 mm mean depth** — visually it presses the peg onto the trained
   position (watched live). The D-035 question is closed by measurement:
   the failure was location-specific, and D-034's randomisation removes it
   across the whole trained range. NOTE: the play.py offset limit was
   raised 0.05 → 0.15 m per axis (2026-08-06, UNVERIFIED) after the
   geometry check the old limit demanded: the joint-space reset spawns the
   peg tip 150 mm above the plate top at any xy offset (no spawn collision
   possible), and the pocket moves with the asset, so the binding
   constraint is pocket-to-base distance ([0.265, 0.585] m at ±15 cm).
   Out-of-training-range points (e.g. ±7, ±10, ±15 cm) are now runnable;
   at +15 cm in y the pocket sits 26.5 cm from the base, so a failure
   there can be reach/configuration rather than generalisation — watch
   the replay, don't read only the rate.

## Completed 2026-08-16 (day, both machines): D-036 measured, D-037 + D-038 trained

Full narrative and all numbers: `docs/session_2026-08-16_neigung_versatz.md`;
decisions: DECISIONS.md D-037 and D-038 (both with verified addenda). In brief:

- **D-036 VERIFIED** on the training machine: tilt read-back exact
  (`measured tilt about y: +5.000 deg … (OK)`, error 0.0000°). Baseline of
  the 21-dim policy: 100 % at tilt 0 (29.3 mm mean max depth), **50 % at 5°**
  (17.9 mm) — a policy property, not a measurement artefact.
- **Root cause: partial observability.** All 21 channels are robot state or
  measured relative to the pocket, so two differently-tilted pockets produced
  identical observations. Fix (D-037): pocket quaternion (wxyz, w ≥ 0)
  appended as channels 21:25 — 4 channels, quaternion input per GenPiH
  (arXiv 2504.04148), NOT the 6D representation (that is for outputs).
- **Checkpoint widener** `scripts/expand_checkpoint_obs.py` (21 → 25:
  zero weight columns, neutral normaliser stats, optimizer state cleared;
  tensors matched by NAME, not shape — PROBLEMS.md row for the shape-sort
  abort). Verified: the widened checkpoint reproduces the baseline exactly.
- **Curriculum (all rungs stopped by `--stop-at-success-rate 0.99`):**
  0–5° → 99.2 % (18 914 episodes); 5°→15° in one jump COLLAPSED (A/B at 5°:
  old 99 %, new 0 %); 5°→10° → 99.3 % (2081 episodes); +offset ±2 cm → 99 %;
  +yaw ±45° (D-038) → 99.19 %. The yaw jump 0→±45° in ONE step did NOT
  collapse — relative yaw was pre-trained via the ±45° wrist reset noise;
  the ladder rule applies to genuinely new variation only.
- **Speed:** 4096 envs instead of 128 → 0.61 s/iteration (~107 k steps/s) on
  the RTX 3080. Episodes-to-threshold numbers are not comparable across
  different env counts.
- **Wind-up measured** (explains the "pressing on one spot" failures):
  joint-target lag, failures 60.6 vs successes 12.8 control steps
  (10° policy); final D-038 policy separates cleanly at success-p95 0.50 rad
  vs failure-mean 1.87 rad (44 failures) — a clamp band between them is
  derivable. Not yet implemented (deliberately a separate single change).
- **Tooling:** explicit run names (`peg30mm_pocket32mm_offset20mm_tilt0-10deg…`),
  per-probe-pose video folders, `success_by_tilt_bin` in demo_metrics.json
  (bins via pocket-z polar angle acos(r22), so yaw cannot masquerade as tilt),
  `--fixture-yaw` probe flag (cap 45°), early-stop flag in train.py.

## Completed 2026-08-16 (morning, dev PC — review only, no measurements)

Code review of the D-036 tilt increment (`f85c517`). Result: **the transforms
are correct, the reporting around them was not.** Fix committed as `fc18189`,
pushed.

- **The maths holds.** `_peg_geometry`'s two transforms are the same rotation:
  `tip_local @ R` equals `R^T v` for a row vector, and the axes use
  `R^T @ peg_axes`. `depth = -tip_rel[2]` is algebraically the old
  `plate_top_z - tip_z` because `OPENING_ENTRANCE_POS[2]` **is** `PLATE_TOP_Z`
  (both 0.755, `proxytask_tasks_cfg.py` lines 69 and 77) — the tilt branch is
  not an approximation of the old expression, it is the same number rotated.
  The `below_plate` rewrite matches the world-frame test exactly. Checked
  numerically at 0, 2, −5, 10 and 15° (scratch script, not committed): a peg
  aligned with the TILTED pocket reads `alignment` +1 and `phi` 0, which is
  the property the gate and the align/yaw reward terms need.
- **The tilt axis was well chosen** (worth keeping for the write-up): the
  rotation is about env y through the pocket centre, and the UR10e base sits
  at env-local x = 0, y = +0.415 m — i.e. exactly ON that axis. The plate
  under the robot base therefore does not move at all under tilt, so no
  base-vs-plate interference is possible. The plate's x edges sweep
  0.250 × sin(tilt) = 65 mm at the 15° cap, which is what the code comment
  claims.
- **Gap 1, closed (the important one).** Nothing measured whether the tilt
  reached the sim. It enters ONLY as the spawn orientation, and the existing
  pose check reads `root_pos_w` — which a rotation about the asset origin
  leaves untouched, so `|prim - buffer|` read ~0 whether the tilt landed or
  silently failed. A no-tilt run would have printed "ACTIVE +5.00 deg" and
  produced a plausible success rate measured against a pocket frame that does
  not physically exist. The startup report now reads `root_quat_w` back,
  recovers the angle as `2·atan2(qy, qw)` (quaternion sign normalised first),
  prints it against the commanded value, and prints
  `*** TILT MISMATCH … every pocket-frame number below is VOID ***` when they
  disagree by more than 1e-4 rad. `|(qx, qz)|` is printed too: nonzero means
  the rotation is not about env y.
- **Gap 2, closed.** The per-env peg block of the startup report still
  measured in the WORLD frame (`depth`, `phi`, the four corners, `alignment`,
  `gate`) while the policy, the reward and `demo_metrics.json` consume the
  pocket frame — two numbers for one quantity, and the printed pair is the one
  a reader trusts. It now applies the same `R^T`; at tilt 0 the branch is the
  pre-D-036 expression. Added a `peg TIP rel. entrance` line (these ARE
  observation channels 12:15), and the home-pose expectations now scale with
  the tilt (depth × cos, tip offset h·sin, h = 0.100 m) — printing the
  untilted expectation under tilt would read as a fault where there is none.
  At 15° the report now expects −96.593 mm depth and a 25.882 mm tip offset.

## Next steps

1. **Start the real task** (the generalisation section is closed). Entry
   point: the "Real task (planning notes, 2026-08-06)" section below —
   feasibility via scripted DISASSEMBLY from the assembled end pose
   (AutoMate-style), CAD of the rectangular guide piece is available.
   First concrete step: define the asset/geometry plan and the passability
   check for the real part before any env code.
2. **Wind-up clamp band** (the "pressing on one spot" failure): clamp the
   joint-target lag as ONE change, band chosen between success-p95 0.50 rad
   and failure-mean 1.87 rad (measured on the final D-038 policy), then
   re-measure the SAME policy with and without the clamp — success rate up
   and failure lag down confirms the mechanism; unchanged numbers refute it.
3. **Thesis write-up**: fold the D-034 curve and the D-036/D-037/D-038 arc
   into `docs/square_task_commissioning.md` (or a sibling doc); sources of
   record are the run directories' `demo_metrics.json` and
   `docs/session_2026-08-16_neigung_versatz.md`.

## Real task (planning notes, 2026-08-06)

- **Feasibility check via disassembly, not assembly.** For the real part
  (rectangular guide piece, CAD available), the pre-training passability check
  is planned AutoMate-style: spawn the part in its assembled end pose and
  script the *extraction*; a collision-free extraction path proves the
  insertion path exists, and the reversed trajectory doubles as a reference
  solution. This sidesteps having to script the corner-first insertion
  strategy even if the real clearance is tight. Precedent: AutoMate
  (arXiv:2407.08028, 100 disassembly paths per assembly), Assemble Them All
  (arXiv:2211.03977); see
  `docs/reference/literature_check_methodology_2026-08-06.md` Part 2.
- Simulator-physics validation is inherited from the proxytask (decided
  2026-08-06, user): no additional contact-physics validation beyond published
  practice. The per-task commissioning chain for *new own artifacts* (assets
  vs. generator sidecars, gate, observation identity, reward algebra) is not
  covered by this and stays — it exists because it caught real defects twice
  (PROBLEMS.md 2026-07-26, 2026-08-06).

## Known open items (do not lose)

- **Probe depth-0 handling** (demoted from next steps, still open): implement
  the persistence criterion in `verify_peg_passability.py`'s `judge` (a
  reaction must hold over ≥ 2 consecutive waypoints; single-sample excursions
  at a zero-separation boundary are the artefact regime, PROBLEMS.md
  2026-08-06), then rerun the +3 cm probe
  (`--json passability_kinematic_offset30_v2.json`), expected PASSABLE.
- **±5 cm combined rung deferred by choice** (2026-08-16): the combined
  policy was trained at offset ±2 cm only; the K2 command (resume with
  `env.fixture_pos_noise_xy=0.05`) is in the session transcript /
  session report if the wider range is ever needed.
- **Deterministic yaw zero-shot probes never measured** (D-038 addendum):
  the probe series at 10/22.5/45° was cut short after one ambiguous 93 %
  replay in favour of training directly. `play.py --fixture-yaw` (cap 45°,
  composes with offset and tilt) remains available as the instrument.
- **Episodes-to-threshold comparability:** runs from 2026-08-16 afternoon
  onward use 4096 envs (earlier: 128, one 1024 run on 2026-08-06). The
  episode counts to a success threshold are not comparable across env
  counts; note it wherever curves are compared for the thesis.

- **Run `2026-08-06_18-43-17_s30b32rxy50` is DISCARDED.** A 251-episode
  rxy50 attempt at b = 32 started 18:43, before the commissioning release,
  aborted almost immediately (0 % success, 0.0 mm depth — uninformative at
  that length; its 2250-episode noise-0 replay of the barely-trained
  checkpoint is likewise 0 %). Attribution was asked three times and never
  answered; the folder stays on disk, its numbers are never to be cited.
  The proper curriculum (20:33 onward) supersedes it.
- **Replay-watching (nearest-orientation question) is still carried over**:
  watch the rxy50 policy replays to see whether it rotates to the nearest
  of the four insertable orientations or a preferred one.

- **The D-033 entry in DECISIONS.md is incomplete.** It ends inside "Options
  considered"; `Decision`, `Rationale` and `Sources` were never written. The
  decision is implemented and verified, but the entry must be completed from the
  implementation before anything in it is cited. Flagged in place.
- **CLAUDE.md § "Two-repo sync workflow" is out of date.** The executable copy on
  the training machine is no longer the manually synchronised
  `…\proxytask_gen\proxytask` but a git clone at
  `…\proxytask_gen\proxytask_quader\proxytask_quader`, with `pip install -e`
  pointing at it. Sync is `git pull`, which removes the stale-copy failure mode
  the section was written for.
- **One rule is waiting for the user's decision on CLAUDE.md** (the file is the
  user's): *a negative control is scaled to the geometry it is meant to
  discriminate against, and a control that cannot physically react must not
  produce a verdict — its silence is not evidence.* Two wrong verdicts have now
  come from this mechanism (2026-07-26, 2026-08-06). A second candidate rule
  about boundary checks and numeric precision was deliberately **not** proposed
  for CLAUDE.md; it lives in PROBLEMS.md only. A third candidate from this
  session (contact verdicts need persistence over consecutive samples, not a
  single excursion at a zero-separation boundary) likewise lives in
  PROBLEMS.md (2026-08-06) pending the user's call. **A fourth candidate from
  2026-08-16, the same family as the first:** *an identity check must exercise
  every degree of freedom the change introduces; a check that cannot fail is
  not evidence.* The D-036 pose check read position only, and a rotation about
  the asset origin does not move the position — so it would have reported
  agreement for an untilted pocket in a run labelled "tilt ACTIVE". Not
  written to CLAUDE.md or PROBLEMS.md: no wrong verdict was actually produced
  (the review caught it before the first run), so there is no symptom to log
  as a problem. The user's call whether it earns a rule.
- ~~play.py metrics fix~~ **fully verified 2026-08-06**: offset variant,
  plain `replay/` path and the ed40da6 progress print all exercised on the
  training machine during the noise-0 replay and the generalisation curve.
- **Passability JSON hygiene — guard implemented** (`163c8f2`): the script
  now refuses to overwrite an existing default file (UNVERIFIED — every run
  since has passed an explicit `--json`). Historical note: the morning's
  static-table PASS JSON was lost to the old behaviour; only its numbers in
  `docs/square_task_commissioning.md` survive. The saved BLOCKED file is
  `…\assets\Robot\passability_kinematic_offset30_BLOCKED.json` (training
  machine). The curve metrics have `.curve_2026-08-06.json` backups next to
  each `demo_metrics.json` IF the user ran the backup command; confirm
  before overwriting anything in that run directory.
- **Latent trap in the fixture cfg:** `fixture_cfg.init_state.pos` is the
  module constant `FIXTURE_POS`, while play.py's `--fixture-offset` shifts only
  `cfg.fixture_pos` (the spawn translation). Nothing currently writes the
  rigid object's default root state, so the mismatch is inert — but any future
  code path that resets rigid objects to their default state would silently
  teleport the fixture back to the unshifted pose. Align init_state with
  fixture_pos in `_setup_scene` when next touching the env. **Since D-036 the
  trap has an ORIENTATION half as well:** `init_state.rot` is the default
  identity while the spawn orientation is what carries the tilt, so the same
  hypothetical default-state write would silently un-tilt the pocket too. The
  new quaternion read-back (`fc18189`) would catch it — but only at the
  report steps, not continuously.
- **The startup report's body-vs-plate clearance loop is world-frame on
  purpose, and is now labelled as such.** It compares body world z against a
  constant `plate_top_z` and assumes the plate centred at env-local (0, 0).
  Two things break that: under tilt the plate top falls by x·tan(tilt), and
  under `--fixture-offset` the plate moves while the test does not (that
  second one is PRE-D-036 and was never noticed — the offset curve's reports
  printed slightly wrong clearance notes throughout). It is a robot-vs-table
  sanity check, not a task measurement, so it was left approximate and
  marked `WORLD-FRAME, indicative only under tilt` rather than rewritten.
  Fix it properly if it ever has to carry a verdict.
- **Cosmetic under tilt:** the whole table rotates, legs included, so on one
  side the legs sink into the ground plane and on the other they lift off.
  Both are static/kinematic colliders, so this is visual only — but do not
  report it as a bug when watching a tilted replay.
- **The dev-PC working copy is a FRESH clone** at
  `C:\Users\PCUser\Desktop\Claude\Generalisierung-Angle-Probe` (2026-08-16),
  not whatever path earlier sessions used. It had no git identity, so
  `user.name` / `user.email` were set locally to
  `Alexander Pett <alexander.pett99@gmail.com>` to match the existing
  history. Pre-commit hooks are NOT installed in this clone.
- **Checkpoint 5 (ρ₀) is still open.** The uniformity of φ over ±45° after reset
  is supported only by the offline sampler (20 000 samples, p95 = 42.7°) and by
  two in-sim samples. The histogram was to be taken from a training run and was
  not; it is not load-bearing for the result, but it is unfinished.
- **The b = 36 sample-efficiency figure is lost** — its metrics file was
  overwritten by a replay before it was read. Only the replay measurement
  (100 % over 751 episodes) survives. Recoverable at best from TensorBoard
  events in that run directory.
- **Literature citations need verification before thesis use.** Whitney 1982,
  Zhou et al. 2019, Ravindran & Barto, Wang/Walters/Platt and NVIDIA
  Factory/IndustReal were all cited from model memory. Look each up and confirm
  claim and bibliographic data.
- Force/torque observations remain unused. D-005 names wrist F/T as part of the
  observation space; the task was solved purely kinematically. This is a
  deviation from the project's own specification and must either be implemented
  (`increment-3-contact-obs`) or justified in the thesis.
- Peg friction remains unset (no measured value for ABS; guessing is forbidden).
  Fixture friction 0.75 per spec. The 50 N force limit is TBD.
- `ur10e_peg_s30.candidate.usd` and the CAD-era strays (`tisch.usd.bak`,
  `tisch.fixed.usd`, `tisch.fix_units.json`, `tisch.verify.json`,
  `Tisch.verify.json`) can be cleaned up.
- Cosmetic and known-harmless: the `ActorCritic … state_dependent_std` warning,
  and the `ee_joint` "disjointed body transforms" warning (D-020 addendum 1).
- Old round-peg checkpoints cannot be replayed on this branch; use
  `demo-insertion-sprint`.
- `.claude/settings.local.json` is untracked — local-only, leave untracked.
- The next free decision ID is **D-039**. D-034 (fixture-pose randomisation),
  D-035 (translation probe), D-036 (pocket tilt in the pocket frame, VERIFIED
  2026-08-16), D-037 (pocket orientation in the observation + tilt curriculum,
  verified addendum) and D-038 (pocket yaw, verified addendum) are written;
  note the file order is chronological (D-035 precedes D-034).

## When to read what

| You are about to… | Read |
|---|---|
| do anything | CLAUDE.md (rules) + this file |
| trust or cite any result of the square task | **docs/square_task_commissioning.md** — every checkpoint with its command and its decisive number |
| understand the pivot's decisions | DECISIONS.md **D-029**, then **D-033** (note its entry is incomplete) |
| need project context, roadmap | docs/project_overview.md (pre-pivot, read as historical) |
| implement env / reward / curriculum | docs/task_specification.md (square addendum) |
| generate or verify the square table | docs/asset_contract_tisch.md § **D-033 addendum** |
| debug a failure | PROBLEMS.md first — known problems are not re-solved |
| compare or replay runs | scripts/compare_runs.py header comments |
| see the demo sprint's results | docs/demo_sprint_results.md |
| map Factory PegInsert parameters | docs/factory_mapping.md |
