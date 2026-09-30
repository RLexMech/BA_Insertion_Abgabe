# Design Decision Log

Running log of all thesis-relevant design decisions. Every entry is written at the
moment the decision is made. Language: English (thesis itself is written in German).

## Entry template

```
## D-NNN: <title>
- Date: YYYY-MM-DD
- Status: accepted | proposed (pending verification) | superseded by D-MMM
- Context: <problem and constraints>
- Options considered: <option A / option B / ...>
- Decision: <what was chosen>
- Rationale: <why, with technical arguments>
- Sources: <papers, documentation pages, measurements>
```

---

## D-001: Target software stack — Isaac Lab 2.3 + Isaac Sim 5.x

- Date: 2026-07-23
- Status: accepted
- Context: The initial task specification named Isaac Lab 3.0 (beta) with Isaac
  Sim 6.0.1. Isaac Lab 3.0 is a ground-up architectural overhaul distributed as a
  beta release; API names and module locations differ from the 2.x line. The
  installed environment on the training machine is Isaac Lab 2.3.
- Options considered: (a) Isaac Lab 3.0 beta + Isaac Sim 6.0.1; (b) Isaac Lab 2.3 +
  Isaac Sim 5.x.
- Decision: Isaac Lab 2.3 + Isaac Sim 5.x. Exact installed versions are to be
  recorded on the training machine in Phase 1.
- Rationale: A stable release line offers substantially better reproducibility for a
  bachelor thesis than a beta: mature documentation, functioning reference
  environments (including Factory), and community-validated configurations. The 3.0
  migration risk (renamed modules, backend-specific implementations) adds effort
  without scientific benefit for this task.
- Sources: Isaac Lab release notes and migration guide (isaac-sim.github.io/IsaacLab).
- Addendum (2026-07-24, verified on training machine): exact installed versions —
  Isaac Sim 5.1.0-rc.19+release.26219.9c81211b.gl; Isaac Lab 2.3.2; Python 3.11.15
  (conda env `env_isaaclab`); NVIDIA GeForce RTX 3080, driver 591.86 (driver
  supports CUDA up to 13.1); PyTorch 2.7.0+cu118 (bundled CUDA 11.8 runtime — the
  version mismatch with `nvidia-smi` is expected, as the driver value is an upper
  bound). Installation confirmed by running the unmodified Isaac-Ant-v0 example
  via rsl_rl in headless mode (command recorded in CLAUDE.md); GPU rendering also
  confirmed working.

## D-002: Two-machine development model and verification rule

- Date: 2026-07-23
- Status: accepted
- Context: Development happens on a Windows PC without Isaac Sim/Lab; execution and
  training happen on a separate Windows machine with an RTX-class GPU (12–24 GB).
  Claude Code will later run directly on the training machine.
- Decision: Code and documentation are authored on the dev PC and transferred via a
  private GitHub repository in small incremental commits. No artifact is labeled
  "tested"/"verified" until it has actually run on the training machine; until then
  it carries the label UNVERIFIED.
- Rationale: Prevents silent divergence between claimed and actual behavior — a
  reproducibility requirement — and makes each pulled increment independently
  testable on the training machine.
- Sources: —

## D-003: Action space — incremental task-space pose deltas with compliant control

- Date: 2026-07-23
- Status: proposed (pending implementation verification)
- Context: Contact-rich insertion requires compliant behavior to avoid excessive
  contact forces and to enable search behavior. Candidate action spaces: joint-space
  targets, task-space absolute targets, task-space incremental deltas.
- Decision: The policy outputs 6-DoF incremental task-space pose deltas
  (Δx, Δy, Δz, Δroll, Δpitch, Δyaw), executed by a compliant task-space controller.
  The concrete Isaac Lab 2.3 realization for the UR10e — Operational Space
  Controller vs DifferentialIKController combined with implicit actuator
  stiffness/damping — is an open sub-decision to be resolved against the 2.3 API
  before environment coding.
- Rationale: Literature reports 3–4× lower sample complexity for task-space
  impedance-style control compared with joint-space control in contact-rich
  insertion; incremental actions generalize better than absolute targets and avoid
  encoding task-specific biases.
- Sources: Varin et al., action-space comparison (IAS TU Darmstadt); IndustReal
  (Tang et al., 2023); Isaac Lab Factory environment documentation. Indexed in
  docs/reference/research_cylindertask.md §4–5.

## D-004: Reward — dense staged reward first, SDF-based reward as upgrade path

- Date: 2026-07-23
- Status: accepted
- Context: Sparse rewards do not converge reliably on contact-rich insertion.
  Candidates: keypoint-based dense reward (Factory default), staged dense reward
  (approach/alignment/insertion/success with force and torque penalties), SDF-based
  alignment reward (IndustReal).
- Decision: Start with the staged dense reward; implement and validate each term
  against a scripted insertion before training. Adopt the SDF-based reward as a
  documented upgrade if convergence stalls or reward exploitation occurs.
- Rationale: The staged reward is simple to implement and debug term-by-term, which
  suits the phased validation plan. IndustReal reports 88.6 % success for SDF-based
  reward vs 1.8–54.2 % for keypoint-based variants, so the upgrade path is kept
  explicit; it is deferred because SDF computation adds implementation risk at
  project start.
- Sources: IndustReal (Tang et al., 2023); reward-design comparison studies indexed
  in docs/reference/research_cylindertask.md §6, §10.

## D-005: End-effector configuration — rigid peg at flange, F/T observation

- Date: 2026-07-23
- Status: accepted
- Context: The task assumes a pre-grasped peg. Modeling a gripper adds asset and
  contact complexity without contributing to the research question. Jamming
  detection and contact-aware behavior require force information.
- Decision: The peg is modeled as a rigid body fixed to the UR10e flange (no gripper
  model). The observation vector includes the 6-D wrist force/torque signal.
- Rationale: Removes irrelevant complexity while keeping all observations available
  on the real robot (the UR10e provides an integrated wrist F/T sensor), preserving
  the optional sim-to-real stretch goal without privileged information.
- Sources: IndustReal and Factory both use fixed-peg configurations for the
  insertion subtask; docs/reference/research_cylindertask.md §3.

## D-006: Curriculum — hybrid of peg-diameter stages and sampling-based initial-state randomization

- Date: 2026-07-23
- Status: proposed (pending literature verification)
- Context: The task specification defines a 3-stage curriculum over peg diameter
  (hole fixed at Ø 30 mm; stage 1: Ø 28 mm peg, 1.0 mm radial clearance). The
  research synthesis recommends a 4-level curriculum over initial-state error,
  clearance, and friction with sampling-based progression (IndustReal), which
  reports 88.6 % success vs 32.4 % for a standard curriculum.
- Decision: Hybrid approach: clearance progression via the peg-diameter stages from
  the task specification, combined with the sampling-based initial-state
  randomization and progression logic (increase the lower bound of the sampling
  range on success-rate threshold; retreat on failure). Before implementation, this
  combination is to be verified against current literature; stage-2/3 peg diameters
  and progression thresholds are open sub-decisions, to be informed by physics
  tests of the minimal stable clearance in simulation.
- Rationale: Preserves the geometry-based difficulty axis required by the task
  specification while adopting the empirically superior progression mechanism from
  the literature.
- Sources: The_Task.txt (task specification); IndustReal sampling-based curriculum;
  docs/reference/research_cylindertask.md §7, §10.

## D-007: Documentation format and language

- Date: 2026-07-23
- Status: accepted
- Context: Thesis-relevant decisions must be traceable and citable. Candidates:
  one file per decision (ADR style), a single running log, chapter-mirrored docs.
- Decision: Single running DECISIONS.md with a fixed entry template. Code,
  comments, and documentation in English; the thesis itself is written in German.
- Rationale: A single chronological log with template-enforced structure keeps the
  overhead low for a single-author project while remaining citable; English
  documentation matches the robotics literature and simplifies source alignment.
- Sources: —

## D-008: Repository structure — rules-only CLAUDE.md, HANDOFF.md entry point, problem log

- Date: 2026-07-23
- Status: accepted
- Context: External practitioner guidance on working with Claude Code in RL
  projects (docs/reference/workflow_tips.txt) recommends a specific document
  structure: a short rules-only instruction file, a handoff file as the single
  session entry point (current state, next 1-3 steps, read-map), a running
  problem log, and long reference material loaded per-phase only. The initial
  CLAUDE.md mixed rules with project narrative (~4 pages).
- Options considered: (a) keep the monolithic CLAUDE.md; (b) split per the
  guidance into CLAUDE.md (rules, <= 1 page), HANDOFF.md (state + next steps +
  read-map), PROBLEMS.md (symptom/cause/fix log), docs/project_overview.md
  (narrative, roadmap, hyperparameters).
- Decision: Option (b). Additional working rules adopted from the same source:
  one hypothesis per change; debugging order setup -> scripted solvability ->
  PPO tuning; explicit exploitability check on every reward edit; no functional
  claim without command + output; root-cause stop after 2 failed fix attempts;
  runs write machine-readable metrics files; end-of-session HANDOFF.md update.
- Rationale: Instruction-following degrades with instruction-file length; rules
  placed close to the working context are followed more reliably. A handoff file
  makes multi-session work (and the dev/training machine handover of D-002)
  reproducible. A problem log prevents re-solving known failures and doubles as
  thesis evidence. The one-hypothesis and measurement-first rules keep RL
  debugging attributable, which is a methodological requirement for the thesis.
- Sources: docs/reference/workflow_tips.txt (practitioner guidance, 2026-07);
  consistent with D-002 (UNVERIFIED rule) and D-007 (documentation duty).

## D-009: Tooling — project skills and session-end hook enforce the documentation duty

- Date: 2026-07-23
- Status: accepted
- Context: The documentation rules of D-007/D-008 (decision log entries at the
  moment of decision, one problem-log row per solved problem, end-of-session
  HANDOFF.md update) rely on being remembered under working pressure, which is
  exactly when they are most likely to be skipped. Claude Code offers two
  enforcement mechanisms: skills (reusable workflow instructions, invocable
  explicitly or triggered by context) and hooks (commands the harness itself
  executes on lifecycle events, independent of model or user attention).
- Options considered: (a) rely on CLAUDE.md rules alone; (b) encode the three
  recurring documentation workflows as project skills and add a harness-level
  reminder hook.
- Decision: Option (b). Three project skills in `.claude/skills/`: `/decision`
  (append a DECISIONS.md entry per template), `/problem` (add a PROBLEMS.md row
  and check for an implied permanent rule), `/handoff` (end-of-session HANDOFF.md
  update plus commit check). One `SessionEnd` hook in `.claude/settings.json`
  that emits a reminder if HANDOFF.md was not modified on the current day.
  A `/hypothesis` skill (one-hypothesis-per-change ritual) is deferred until
  training runs exist (Phase 2).
- Rationale: Skills move workflow knowledge from prose rules into executable,
  versioned procedures that trigger contextually; the hook removes the last
  single point of failure (human memory) for the handoff update because the
  harness executes it unconditionally. Partially resolves open decision 1
  (workflow).
- Sources: Claude Code documentation (skills, hooks); D-007, D-008.

## D-010: Documentation — INDEX.md as mandatory section map for docs/reference/

- Date: 2026-07-23
- Status: accepted
- Context: The per-section reading rule for long reference docs (D-008) requires
  knowing in advance which section is relevant. Without a map, section selection
  is blind: cross-section dependencies are missed (e.g. the action space of
  research_cylindertask.md §4 presupposes the controller of §5), sections can be
  misidentified, and document-internal caveats (Franka-specific parameters,
  7-DOF observation layouts) are encountered only by chance. The rationale for
  the repository structure itself is preserved independently in D-008, so the
  map does not need to duplicate source content.
- Options considered: (a) keep blind per-section selection; (b) allow loading
  reference docs whole when uncertain; (c) create docs/reference/INDEX.md — a
  one-time full read of each reference doc distilled into a section table with
  line anchors, project-phase mapping, dependencies, and caveat markers,
  consulted before every section access.
- Decision: Option (c). CLAUDE.md rule extended accordingly. The index carries a
  snapshot date and the rule that an edit to a source doc must update the index
  in the same commit; reference docs are otherwise frozen sources.
- Rationale: Option (a) leaves cross-reference and misidentification risks
  unmitigated; option (b) violates the context-budget rationale of D-008. The
  index costs one full read at creation time and ~40 lines per subsequent
  access, while making section selection informed instead of blind. Creating
  the index already corrected one latent error: a privileged-observation caveat
  initially attributed to §3 (which is explicitly non-privileged and consistent
  with D-005) in fact belongs to §13 (Factory environments).
- Sources: docs/reference/workflow_tips.txt (context-budget guidance); D-005,
  D-008; Claude documentation best practices on strategic file selection.

## D-011: Tooling — /tagesbericht skill for the personal daily journal

- Date: 2026-07-23
- Status: accepted
- Context: Alexander keeps a personal daily journal of what was accomplished
  each working day, separate from the repo-internal English documentation.
  Reconstructing the day by hand duplicates information that git history and
  HANDOFF.md already contain, and the summary format was being re-specified
  ad hoc each time.
- Options considered: (a) continue writing the journal entry manually;
  (b) extend /handoff to also emit a German summary; (c) a separate
  /tagesbericht skill that outputs a chat-only German summary (Erledigt heute /
  Stand / Als Nächstes) built from the day's git log and HANDOFF.md.
- Decision: Option (c). Chat output only, no file writes; the journal itself
  lives outside the repository. The skill warns if the day's work contains
  decisions or solved problems missing from DECISIONS.md / PROBLEMS.md.
- Rationale: Keeps /handoff single-purpose (repo state, English) per D-008
  while removing manual reconstruction effort; chat-only output was chosen
  because the journal is maintained outside the repo. Extends the tooling
  approach of D-009.
- Sources: D-007 (language split), D-008, D-009.

## D-012: Isaac Lab workflow — Direct, with Factory PegInsert as template

- Date: 2026-07-24
- Status: proposed (pending verification)
- Context: Isaac Lab 2.3 offers two task-authoring workflows. Manager-based
  environments decompose the task into configurable managers (observations,
  actions, rewards, randomization); Direct environments implement the full logic
  in a single class inheriting from `DirectRLEnv`. The contact-rich insertion
  physics is the highest-risk part of the project, and Isaac Lab 2.3 ships
  `Isaac-Factory-PegInsert-Direct-v0`, whose contact-solver settings, material
  properties, and timestep/control-frequency configuration are already tuned for
  insertion. The available UR10/UR10e reference environments (Reach, Stack-Cube)
  are manager-based but do not involve contact-rich insertion.
- Options considered: (a) Manager-based — modular, cleaner to document, component
  swapping during prototyping; (b) Direct — full control in one class, and the
  Factory PegInsert environment can serve as a template.
- Decision: Direct workflow. The environment is developed using
  `Isaac-Factory-PegInsert-Direct-v0` as the structural template: physics
  configuration, robot/actuator setup pattern, and training infrastructure are
  reused; reward structure, observation vector, and randomization are replaced
  per D-004/D-005/D-006 (the Factory keypoint reward, privileged observations,
  and aggressive randomization are explicitly not copied).
- Rationale: The manager-based workflow's modularity primarily benefits
  multi-developer projects; this is a single-author thesis. Reusing a
  contact-physics configuration that is already validated for insertion removes
  the largest source of physics-tuning risk, whereas a manager-based rebuild
  would recreate that configuration from scratch without scientific benefit. The
  Direct workflow also gives fine-grained control over environment logic, which
  suits the term-by-term reward validation plan (D-004).
- Sources: Isaac Lab 2.3 documentation, "Task Design Workflows" and "Available
  Environments" (isaac-sim.github.io/IsaacLab, v2.3.0); docs/reference/
  research_cylindertask.md §13 (reuse/not-copy analysis of Factory).

## D-013: RL library — rsl_rl

- Date: 2026-07-24
- Status: accepted
- Context: Isaac Lab 2.3 supports rsl_rl, rl_games, skrl, and Stable-Baselines3.
  The algorithm is fixed to PPO. Selection criteria for a bachelor thesis:
  simplicity, logging, reproducibility (open decision 2). The Factory
  environments ship with an rl_games PPO configuration; the Phase 1 installation
  check on the training machine ran rsl_rl (Isaac-Ant-v0, verified 2026-07-24).
- Options considered: (a) rsl_rl — PPO-focused, fastest in the Isaac Lab
  benchmark, vectorized, minimal but sufficient surface; (b) rl_games — Factory's
  shipped PPO config directly reusable, but weakly documented with a convoluted
  configuration format; (c) skrl — best documentation and broadest algorithm
  support, marginally slower, breadth not needed for PPO-only use;
  (d) Stable-Baselines3 — no vectorized training, slowest by a factor of ~1.6,
  excluded.
- Decision: rsl_rl. The PPO hyperparameters of the Factory PegInsert rl_games
  configuration are translated into the rsl_rl configuration format rather than
  adopting rl_games.
- Rationale: rsl_rl covers exactly the required algorithm with the smallest
  configuration surface, which serves reproducibility and thesis documentation;
  it is the fastest of the supported libraries in Isaac Lab's own benchmark; and
  it is the only library already verified end-to-end on the training machine
  (D-001 addendum), so the toolchain risk is zero. The sole advantage of
  rl_games — direct reuse of the Factory PPO file — is outweighed by its weak
  documentation; hyperparameter translation is a bounded, documentable step.
- Sources: Isaac Lab 2.3 documentation, "Reinforcement Learning Frameworks"
  comparison (isaac-sim.github.io/IsaacLab, v2.3.0); D-001 addendum
  (verified Ant run via rsl_rl on the training machine, 2026-07-24).

## D-014: Controller realization — OperationalSpaceController

- Date: 2026-07-24
- Status: proposed (pending verification)
- Context: D-003 fixes the action space to incremental task-space pose deltas
  executed by a compliant controller and leaves the Isaac Lab 2.3 realization
  open: OperationalSpaceController (OSC) vs DifferentialIKController combined
  with implicit actuator stiffness/damping. The Isaac Lab 2.3 API defines
  DifferentialIKController as purely kinematic (joint-position output, no
  impedance regulation), while OperationalSpaceController provides impedance
  control with configurable stiffness/damping, motion- and wrench-control axes,
  gravity compensation, and joint-effort output.
- Options considered: (a) DifferentialIKController + implicit actuator gains —
  simpler, but compliance arises only indirectly from the joint-level PD
  actuator model, with no task-space impedance shaping; (b) OSC — direct
  task-space impedance control matching the τ = J^T (K_p Δx + K_d Δv) structure
  recommended for contact-rich insertion.
- Decision: OperationalSpaceController. Stiffness and damping gains for the
  UR10e are measured in simulation (Phase 2 physics validation), not copied from
  Franka reference values (stiffness 800 N/m, damping 40 Ns/m).
- Rationale: Contact-rich insertion requires defined compliant behavior in task
  space to bound contact forces and enable search behavior; only the OSC
  provides task-space impedance regulation in the 2.3 API. The diff-IK variant
  reduces compliance to joint-level actuator gains, which cannot shape
  directional task-space stiffness and would confound the force-limit validation
  (50 N criterion, Phase 2). This resolves the open sub-decision of D-003.
- Sources: Isaac Lab 2.3 API documentation, `isaaclab.controllers`
  (OperationalSpaceController, DifferentialIKController); docs/reference/
  research_cylindertask.md §4–§5 (impedance-control recommendation and
  sample-efficiency comparison); D-003.

## D-015: Git checkpoint model — verified milestones as recoverable baselines

- Date: 2026-07-24
- Status: accepted
- Context: The commit rules (CLAUDE.md, D-002) require small verified commits but
  do not define how working states are staged during active coding, when edits
  are exploratory and not yet verified on the training machine. Open decision 1
  (workflow) included settling this checkpoint model.
- Options considered: (a) commit only verified states — loses intermediate work;
  (b) commit freely without marking — verified baselines become indistinguishable
  from transitional states; (c) transitional edits with explicitly marked,
  recoverable milestones.
- Decision: Option (c). Edits are transitional until verified working. A verified
  state is saved as a milestone: pushed to the remote as the new baseline and
  marked recoverable (git tag or branch), and coding continues from there.
  Important previous states must remain recoverable at all times; milestones are
  never deleted or rebased away.
- Rationale: Separates exploration from verified baselines without losing either:
  the milestone marking makes every verified state citable and restorable (a
  reproducibility requirement per D-002), while transitional commits keep
  incremental work transferable between the two machines.
- Sources: — (organizational; consistent with D-002 and the CLAUDE.md git rules).

## D-016: Fixture-USD import conventions — SI meters, origin at bore entrance, Z-up

- Date: 2026-07-24
- Status: accepted (asset verification itself pending on the training machine)
- Context: The table/hole fixture ("Tisch2") is authored in Creo and imported via
  STEP → USD. Three silent failure modes exist: unit mismatch (STEP metadata is
  typically millimetres; a known Isaac Sim 5.0/5.1 defect double-applies the
  unit-resolving scale when `metersPerUnit != 1`), origin mismatch (after import
  the USD origin is the CAD world origin, not the bore — the more probable
  misalignment source than units), and axis mismatch (Isaac Lab's mesh converter
  emits Y-up USD; Isaac Sim stages are Z-up).
- Options considered: (a) trust CAD export settings; (b) enforce invariants on
  the saved USD with a post-import verification procedure.
- Decision: Option (b). Invariants on the fixture USD: `metersPerUnit = 1.0`
  with real-world dimensions (plate thickness reads 0.055 m, bore diameter
  0.030 m in-stage); default-prim origin coincident with the bore-entrance datum
  (or a constant CAD-origin-to-bore offset carried explicitly in the scene
  config); up-axis verified +Z. Any ~1000× scaling is corrected at the USD
  level, then saved.
- Rationale: Post-import measurement against known nominals catches all three
  failure modes deterministically regardless of exporter behavior; standardizing
  `metersPerUnit = 1.0` sidesteps the documented 5.x double-scaling defect.
- Sources: Isaac Sim documentation (units, up-axis conventions); Isaac Lab asset
  pipeline docs; merged from the external component log (D-1, D-4, D-5), see
  D-016…D-020 provenance note in
  `Bachelor/Projekt/design_decisions_component1_2.md`.

## D-017: Task frame at the bore entrance, +z along the insertion axis

- Date: 2026-07-24
- Status: accepted
- Context: Observations are defined relative to the known hole pose (D-005), and
  insertion depth must be a well-defined scalar for termination and reward.
- Options considered: (a) express poses in world coordinates; (b) a dedicated
  hole-fixed reference frame at the bore entrance, +z directed into the bore.
- Decision: Option (b). The frame defines the task frame for observations
  (EE pose relative to it) and insertion depth (displacement along its +z).
  Caveat: STEP datum frames generally do not survive tessellation as transform
  prims — after import, verify an `Xform` exists at the bore entrance and
  re-author it if not.
- Rationale: Decouples policy input from fixture placement in world coordinates;
  the 25 mm success criterion stays well defined regardless of fixture
  orientation; independent of the world up-axis.
- Sources: Merged from the external component log (D-2); consistent with D-005
  and the task specification's termination criteria.

## D-018: Asset-to-configuration-class mapping

- Date: 2026-07-24
- Status: accepted
- Context: The scene holds three assets (UR10e, peg, table/hole fixture); Isaac
  Lab 2.3 offers ArticulationCfg, RigidObjectCfg, and static AssetBaseCfg.
- Options considered: mapping alternatives, including configuring the fixture as
  an articulation or importing the peg as a file asset.
- Decision: UR10e → `ArticulationCfg` (derived from shipped `UR10e_CFG`); peg →
  spawned `CylinderCfg` primitive (radius 0.014 m, height 0.050 m, stage 1);
  fixture → static `AssetBaseCfg` with `UsdFileCfg` (authored USD), carrying the
  collision geometry and serving as the contact-sensor filter target
  (`filter_prim_paths_expr`). Only the fixture requires file import.
- Rationale: The robot is the sole articulated body; the fixture is static and
  non-articulated; the peg needs no external file, so creating it procedurally
  keeps the import surface (and its failure modes, D-016) confined to one asset.
- Sources: Isaac Lab 2.3 API documentation (assets, sensors); merged from the
  external component log (D-3).
- Addendum (2026-07-26): **no physical joint is authored between robot and
  fixture**, although the robot is mounted on the fixture plate. The question was
  raised explicitly, with the Isaac Sim Robot Assembler proposed as the mechanism.
  It does not apply: the documentation states that static assemblies such as a
  robot permanently fixed to a table do not need the tool, which is intended for
  assets that move while the timeline plays, and that it removes the
  `ArticulationRoot` from the attached robot — which would remove joint control,
  actuators and the RL interface altogether. Both bodies are already immovable by
  different mechanisms: the fixture is a static collider with no rigid body
  (this entry), and the articulation root is world-fixed
  (`fix_root_link = True`, plus the asset's own `root_joint`). Measured: the base
  holds env-local `(0, 0.190, 0.755)` unchanged over 3 s.
  The connection is instead expressed in configuration: `task_geometry.py` derives
  the robot base height from the plate top, so the robot stands on the fixture by
  construction rather than by two literals that happen to match. Consequence to
  carry forward: if the fixture pose is ever randomized, the robot does **not**
  follow automatically — both must be moved together, or a joint introduced then.
- Sources (addendum): Isaac Sim 5.1 documentation, "Robot Assembler for Asset
  Assembly"; increment-1 startup report, training machine, 2026-07-26.
- Addendum 2 (2026-07-26): **the peg mapping above is superseded by D-019.**
  This entry maps the peg to a spawned `CylinderCfg` primitive; D-019, written
  the same day once the failure modes were known, settles the peg as a welded
  link inside the robot USD. The two stood in unresolved tension until the peg
  increment forced the question, and D-019 governs: nothing spawns the peg at
  scene-build time, so there is no peg spawn cfg at all, and `HeldAssetCfg`
  carries the peg's data for the authoring script rather than for a spawner.
  `docs/factory_mapping.md` is corrected in the same commit. The fixture and
  robot mappings in this entry are unaffected. The first addendum's reference to
  `task_geometry.py` still holds; the file was renamed to
  `proxytask_tasks_cfg.py` (D-027) without changing the derivation it describes.
- Addendum 3 (2026-08-06): **the fixture mapping is superseded by D-034.** The
  fixture is no longer a static `AssetBaseCfg` collider but a kinematic
  `RigidObjectCfg`, because D-035 measured that the fixed pose must become a
  per-episode runtime quantity and a static collider has no writable pose.
  Addendum 1's consequence ("if the fixture pose is ever randomized, the robot
  does not follow automatically — both must be moved together") is resolved
  deliberately in the *other* direction: the robot base stays world-fixed and
  only the fixture moves. The base does not rest on the plate by contact (it
  is world-fixed by `fix_root_link`, addendum 1), so a plate sliding a few
  centimetres under it has no physical effect; the pocket moving relative to
  the trained arm configuration is the entire point of the randomisation. The
  robot and peg mappings in this entry are unaffected.

## D-019: Peg attachment realization — welded rigid link inside the robot USD

- Date: 2026-07-24
- Status: **verified** (training machine, 2026-07-26, marker
  `increment2-peg-2026-07-26a`). The decisive question was whether PhysX keeps a
  fixed-joint child as a distinct articulation link or collapses it into the
  parent, which would have left the contact sensor without a target. It does
  not collapse: `body_names` reads
  `[..., 'wrist_3_link', 'peg_link']`, eight bodies against the shipped seven,
  and `peg_link` carries its own index 7. Measured at the home pose with the
  arm written kinematically: `flange -> peg origin` 0.000000 m and
  `tip along tool axis` +0.050000 m, i.e. the weld holds the peg exactly at the
  flange with its tip exactly one peg length along the tool direction.
  Independent cross-check under gravity at 3 s: the flange stands 13.415 mm off
  the bore axis while the tip stands 15.624 mm off; the tool is tilted by
  arccos(0.999024) = 2.53°, and 0.050 m · sin(2.53°) = 2.21 mm accounts for the
  2.209 mm difference. Peg length, direction and attachment are therefore
  confirmed by a second, independent route rather than by the authoring script's
  own numbers. Stable at 1, 16 and 128 environments.
- Context: D-005 fixes "rigid peg at the flange, no gripper" but not the
  mechanical realization in the simulator. Candidates: separate `RigidObjectCfg`
  peg joined to the wrist by a PhysX fixed joint at scene-build time, or the peg
  modeled as an additional rigid link welded into a re-authored robot USD.
- Options considered: (a) fixed joint to a separate rigid object; (b) welded
  link inside the robot USD (part of the articulation).
- Decision: Option (b), using the gripper-less UR10e configuration (no Robotiq
  variant). The contact sensor then targets the peg link of the articulation.
  The re-authored USD path enters `ur10e_peg_insertion_cfg.py` via
  `PEG_WELDED_UR10E_USD_PATH`.
- Rationale: Kinematic reparenting under an articulation is unreliable, and a
  scene-build fixed joint adds a failure class ("no bodies defined at
  body0/body1 … Failed to create articulation") without benefit; a welded link
  makes the peg's mass/inertia part of the articulation and gives the contact
  sensor a stable target. A gripper variant would add unused joints (D-005).
- Sources: Merged from the external component log (D-6); realizes D-005.
- Addendum (2026-07-26, peg increment): the mechanism, now that it is written.
  `scripts/author_peg_ur10e.py` flattens the shipped UR10e USD into a private
  copy and authors, under the default prim `/ur10e`: a `peg_link` Xform placed
  at the flange's world transform, its geometry as a child, `RigidBodyAPI` and
  `MassAPI` on the link, and a `UsdPhysics.FixedJoint` named `peg_weld` with
  identity local frames on both sides. A rigid body may not be nested inside
  another rigid body, so `peg_link` is a sibling of `wrist_3_link` rather than
  its child; the script refuses to run if the default prim itself carries
  `RigidBodyAPI`. Three realization details are decisions in their own right:
  - **The geometry is a tessellated mesh, not a `UsdGeom.Cylinder` gprim.**
    PhysX has no native cylinder collision shape; Isaac Sim covers cylinders
    through a custom-geometry path whose setting the authoring script cannot
    verify. A mesh removes that dependency and makes `MeshCollisionAPI`
    unambiguously applicable. Measured cost at 64 segments: the convex hull is
    inscribed, so the effective radius is 0.0169 mm under nominal at facet
    midpoints — 1.7 % of the 1.0 mm radial clearance. The script computes and
    prints this rather than assuming it.
  - **Collision approximation is `convexHull`, not the fixture's `none`.** This
    is a constraint, not a preference: PhysX accepts exact triangle meshes only
    for static colliders, and the peg is part of a dynamic articulation. The peg
    is convex, so the hull loses nothing beyond the faceting above.
  - **`MassAPI:diagonalInertia` is the about-COM value 8.23e-6 kg m².** The
    task specification also lists 2.82e-5 about the flange; the two agree via
    Steiner (8.23e-6 + m·(L/2)² = 2.82e-5), which the script asserts at startup.
    Writing the flange value would have overstated transverse inertia 3.4-fold.
  Naming correction: this entry announced `PEG_WELDED_UR10E_USD_PATH` in
  `ur10e_peg_insertion_cfg.py`. Neither exists. The path is resolved at spawn
  time by `resolve_peg_robot_usd_path()` in `proxytask_tasks_cfg.py`, with the
  override `PROXYTASK_UR10E_PEG_USD`, mirroring the fixture resolver — resolving
  at module import would raise on any machine without the asset and take task
  registration down with it. Status stays `proposed`: nothing here has run on
  the training machine, and the claim that PhysX keeps `peg_link` as a distinct
  articulation body rather than collapsing the fixed joint is exactly what the
  startup report's `peg_body` line tests.
- Sources (addendum): `scripts/author_peg_ur10e.py`; offline geometry check of
  the mesh generator on the dev PC (closed surface, bounding box
  0.028 × 0.028 × 0.050 m, positive divergence-theorem volume matching the
  inscribed 64-gon prism); docs/task_specification.md stage-1 peg values.

## D-020: UR10e articulation cfg — derived from shipped UR10e_CFG, two variants

- Date: 2026-07-24
- Status: proposed (pending verification)
- Context: The task needs a UR10e ArticulationCfg with contact reporting and a
  control mode matching D-014 (OSC, joint-effort output). The shipped
  `UR10e_CFG` carries validated per-group implicit-PD gains but
  `activate_contact_sensors=False`, and nonzero PD gains fight effort commands.
  This entry consolidates the robot-cfg decisions referenced as "D-8…D-12" in
  `ur10e_peg_insertion_cfg.py`, whose defining document was never created.
- Options considered: (a) re-author from a raw usd_path; (b) derive from the
  shipped `UR10e_CFG`, overriding only task-required fields.
- Decision: Option (b), realized in `ur10e_peg_insertion_cfg.py` (currently
  outside the repo, to be moved in): `activate_contact_sensors=True`; nested
  `rigid_props.replace(disable_gravity=True, max_depenetration_velocity=5.0)`
  (disable_gravity is an explicit modeling choice, to revisit against the
  F/T-realism requirement); two variants — `UR10E_PEG_INSERTION_CFG` with
  shipped PD gains for position-target bring-up, and
  `UR10E_PEG_INSERTION_TORQUE_CFG` with arm stiffness/damping = 0 for the OSC
  effort pipeline, following the Factory PegInsert pattern.
- Rationale: Deriving preserves the validated actuator and solver settings while
  keeping every task-specific delta visible and reviewable; the zero-gain
  variant is required because the implicit actuator PD would superimpose
  position-tracking torques on the OSC effort command (Factory zeroes arm gains
  for the same reason); wholesale replacement of nested cfgs would silently
  reset shipped fields to class defaults.
- Sources: Isaac Lab 2.3 API documentation (ArticulationCfg, ImplicitActuatorCfg);
  Factory PegInsert source (IsaacLab tag v2.3.2, factory_env_cfg.py); D-014.
- Addendum (2026-07-26, verified on training machine via
  `scripts/probe_assets.py`): the premise holds — `isaaclab_assets.robots.
  universal_robots` does ship `UR10e_CFG`, with `usd_path`
  `{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd`, alongside
  `UR10e_ROBOTIQ_2F_85_CFG` and `UR10e_ROBOTIQ_GRIPPER_CFG` (both irrelevant per
  D-019, which selects the gripper-less variant). Option (b) therefore stands
  without a fallback. Asset facts recorded for the environment code: default prim
  `/ur10e`; seven rigid bodies `base_link, shoulder_link, upper_arm_link,
  forearm_link, wrist_1_link, wrist_2_link, wrist_3_link`; six revolute joints
  named exactly `shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
  wrist_1_joint, wrist_2_joint, wrist_3_joint`; `root_joint` is a
  `PhysicsFixedJoint` against the world, so the base is fixed by the asset itself;
  `ee_joint` is a marker-only fixed joint on `wrist_3_link` at local position
  (0, 1.3e-8, 0), i.e. the flange frame coincides with the `wrist_3_link` origin
  and carries no offset. `ur10e_instanceable.usd` does not exist.
- Addendum 2 (2026-07-26, `probe_assets.py` section 4 on the training machine —
  the measured API surface the environment code is written against, replacing
  field names previously taken from model memory):
  - `UsdFileCfg` has **no `mesh_collision_props` field**. Its fields are `func,
    visible, semantic_tags, copy_from_source, mass_props, deformable_props,
    rigid_props, collision_props, activate_contact_sensors, scale,
    articulation_props, fixed_tendons_props, spatial_tendons_props,
    joint_drive_props, visual_material_path, visual_material, usd_path, variants`.
    `MeshCollisionPropertiesCfg` exists as a separate class whose field is
    `mesh_approximation_name`, not `mesh_approximation`. **Consequence to carry
    forward:** the exact-triangle-mesh collision approximation is not reachable
    through `UsdFileCfg`, so the blind bore is spawned with `collision_props` only
    and may be sealed by a convex hull. That is invisible until a peg exists, so it
    is deferred to the peg increment deliberately rather than forgotten.
  - `ArticulationCfg` fields: `class_type, prim_path, spawn, init_state,
    collision_group, debug_vis, articulation_root_prim_path,
    soft_joint_pos_limit_factor, actuators`. `InitialStateCfg`: `pos, rot, lin_vel,
    ang_vel, joint_pos, joint_vel`. `fix_root_link` confirmed present on
    `ArticulationRootPropertiesCfg`.
  - `ArticulationData` carries **both** spellings — `body_pos_w` / `body_quat_w`
    and `body_link_pos_w` / `body_link_quat_w`, plus `root_pos_w`, `root_quat_w`,
    `default_joint_pos`, `default_root_state`. The 2.x renaming worry is moot; use
    `body_pos_w`.
  - Signatures: `spawn_from_usd(prim_path, cfg, translation=None,
    orientation=None)`; `write_joint_state_to_sim(position, velocity,
    joint_ids=None, env_ids=None)`; `write_root_pose_to_sim(root_pose,
    env_ids=None)`; `find_bodies(name_keys, preserve_order=False) -> (ids, names)`;
    `clone_environments(copy_from_source=False)`;
    `filter_collisions(global_prim_paths=None)`.
  - Verbatim `UR10e_CFG`: `prim_path` is `MISSING` and must be set;
    `spawn.copy_from_source=True`; `spawn.rigid_props.disable_gravity=True`
    already, matching this entry's decision; `articulation_props`
    `solver_position_iteration_count=16`, `enabled_self_collisions=False`,
    `fix_root_link=None`; `init_state.joint_pos` keyed by joint name; three implicit
    actuator groups — `shoulder` (`shoulder_.*`, stiffness 1320.0, damping
    72.6636085), `elbow` (`elbow_joint`, 600.0, 34.64101615), `wrist` (`wrist_.*`,
    216.0, 29.39387691), all with `effort_limit=None` and `velocity_limit=None`, so
    limits come from the USD. The articulation root API sits on
    `/ur10e/root_joint`, not on `/ur10e`.
  - Deviation flagged for increment 1 only: the commissioning scene overrides
    `disable_gravity` to `False`, so the implicit actuators must hold the home pose
    against gravity with no action applied. That is the intended observable and an
    implicit gain test; the `disable_gravity=True` choice of this entry and the
    zero-gain variant both belong to the OSC increment.

## D-021: Project structure — external template project at the repository root, package `proxytask`

- Date: 2026-07-25
- Status: accepted (generation itself pending on the training machine)
- Context: The environment code needs a location from which Isaac Lab can load
  it via `gym.register`. Isaac Lab 2.3 documents two options and ships a
  template generator (`isaaclab.bat --new`): an *internal task* inside the Isaac
  Lab repository, or an *external project* — a self-contained, pip-installable
  extension. The documentation recommends the external variant and disables the
  internal one for pip installations. A complication: the generator runs
  `git init` in the target directory, while this project already has a
  repository carrying the decision log, problem log, and handoff history with a
  private remote attached.
- Options considered: (a) internal task inside the Isaac Lab tree — obscures
  project visibility and complicates Isaac Lab version updates; (b) external
  project as a second, separate repository — separates code from the decisions
  that justify it; (c) external project generated into a temporary directory,
  its `.git` discarded, contents placed at the root of the existing repository.
- Decision: Option (c). The generated Isaac Lab structure (`source/`,
  `scripts/`, `pyproject.toml`, `setup.py`) becomes the repository root layout;
  the documentation files (CLAUDE.md, HANDOFF.md, DECISIONS.md, PROBLEMS.md,
  `docs/`) remain alongside it. Generator answers: external project, Direct
  workflow (D-012), rsl_rl/PPO (D-013). Extension, module, and task are named
  `proxytask`; the registered task ID follows the Isaac Lab convention
  (`<Name>-Direct-v0`). Generation runs on the training machine, since the
  generator requires an Isaac Lab installation (D-002).
- Rationale: The external project keeps the code independent of the Isaac Lab
  version and installable as an Omniverse extension, as recommended. Placing it
  in the existing repository keeps code and the decisions that motivate it in a
  single traceable history — a reproducibility requirement for the thesis —
  and preserves the attached remote without a history migration; discarding the
  generated `.git` costs nothing, as it contains only the template commit. The
  short package name was chosen because the name repeats three times in the
  nested path (`source/<name>/<name>/tasks/direct/<name>/`); with the longer
  candidate the absolute path reached roughly 196 of the 260 characters
  permitted by default on Windows, leaving little margin for the `agents/`
  subdirectory and OneDrive synchronization. Lowercase follows PEP 8 and avoids
  case-sensitivity defects between Windows and Linux.
- Sources: Isaac Lab 2.3 documentation, "Create new project or task" and
  "Project Structure" (isaac-sim.github.io/IsaacLab, v2.3.1); D-002 (two-machine
  model), D-012 (Direct workflow), D-013 (rsl_rl).

## D-022: Bore task frame realized as configuration constants, not as a USD prim

- Date: 2026-07-26
- Status: **verified** (training machine, 2026-07-26, increment-1 startup report).
  With the home pose commanded, the flange stands `0.000782 m` from the bore in XY
  and `0.148859 m` above the plate top against a nominal `0.150`, with the tool
  axis at `z · (0,0,−1) = 0.999997`. The env-local realization is confirmed by a
  four-environment run: env 0 and env 1 report identical env-local values to the
  last digit while their world positions differ by exactly `env_spacing = 3.0`
  (bore at world y `−1.725` and `+1.275`), which is the error a single-environment
  run cannot expose. See D-026 for why the steady-state reading differs.
- Context: D-017 defines a task frame at the bore entrance with +z along the
  insertion axis, and anticipates that STEP datum frames generally do not survive
  tessellation as transform prims. That caveat materialized: `list_usd_prims.py`
  run against `Tisch.usd` on the training machine (2026-07-25) returned only
  `/World/tisch5`, its `Looks` scope, and the mesh child — no datum-derived
  `Xform`. Creo's STEP export was configured with `Datums` and `Extended Datums`
  enabled under AP242, and the coordinate system was nevertheless absent from the
  STEP file itself. The open question was therefore how to realize the frame.
- Options considered: (a) author the `Xform` into the USD with a script
  (`add_bore_task_frame.py` was written for this; its execution on the training
  machine did not produce output and the approach was abandoned rather than
  debugged further, since the alternatives are cheaper); (b) re-export the STEP
  with the bore coordinate system as the part default so the USD origin coincides
  with the bore — rejected, because that system's +z points into the plate, so the
  table imports inverted, and correcting it requires a 180° rotation about X,
  which mirrors the Y axis, plus a compensating +0.755 m translation in Z; two
  interacting corrections, one of them in the axis used for robot placement;
  (c) keep the already-verified upright `Tisch.usd` unmodified and carry the
  CAD-origin-to-bore offset explicitly in the scene configuration.
- Decision: Option (c), which D-016 already admits as an alternative
  ("…or a constant CAD-origin-to-bore offset carried explicitly in the scene
  config"). No `Xform` is authored into the fixture USD; readers should not
  search for one. The fixture is a static asset (D-018), so its pose does not
  change at runtime and the bore entrance is a constant.

  All values below are **relative to the environment origin**, not absolute world
  coordinates. Isaac Lab places each parallel environment at its own
  `scene.env_origins` (a grid offset in X and Y); asset state returned by the API
  is in absolute world coordinates, so runtime comparisons must add or subtract
  `env_origins`. Treating these constants as absolute would make the bore pose
  correct only for environment 0.

  - Fixture spawn: `pos = (0.0, 0.0, 0.700)`, `rot = (1.0, 0.0, 0.0, 0.0)`
    (identity; the asset is already upright), which places the leg underside on
    the ground plane, since the asset's own origin sits at the plate underside
    with the legs extending to −0.700 m.
  - Plate top surface: `z = 0.755`.
  - Bore entrance, i.e. task frame origin: `(0.0, -0.225, 0.755)`.
  - Task frame orientation: +z antiparallel to the environment's +Z, expressible
    as a constant quaternion `(0.0, 1.0, 0.0, 0.0)` (180° about X) where a full
    pose is required.
  - Insertion depth reduces to `0.755 - z_peg_tip`, a scalar subtraction; no
    quaternion arithmetic and no sign inversion are involved.
- Rationale: The approach leaves the verified asset byte-identical, so the
  `verify_fixture_usd.py` result (metersPerUnit = 1.0, up-axis Z, no residual
  millimetre scaling, total height 0.755 m) remains valid without re-running the
  import chain. It eliminates a class of orientation and mirroring defects that
  option (b) introduces precisely in the axis along which the manipulator is to be
  mounted, and it is consistent with D-005, under which the hole pose is known
  rather than observed. The offsets are measurements, not estimates: the −Y
  direction was confirmed against the imported asset in the Isaac Sim viewport
  (2026-07-26), and the 0.225 m offset, the 0.055 m plate thickness, and the
  0.030 m bore diameter were read from the Creo source. Consequence to record:
  should fixture-pose randomization ever be introduced, these constants must be
  replaced by values derived from the asset's actual transform.
- Sources: Measurements — `verify_fixture_usd.py` and `list_usd_prims.py` output
  on the training machine (2026-07-25/26), Creo dimension queries on the source
  model; D-016 (import invariants, which sanctions the explicit-offset variant),
  D-017 (task frame definition and its datum caveat), D-018 (fixture as a static
  asset), D-005 (hole pose is known).

## D-023: UR10e base placement — 415 mm from the bore along −Y, on the fixture plate

- Date: 2026-07-26
- Status: **verified** (training machine, 2026-07-26). The placement was exercised
  implicitly, as planned, by assembling the scene and commanding the arm to the
  bore. Measured: base env-local `(0.0, 0.190, 0.755)`, unchanged over 3 s and
  identical across environments; the base sits on the plate top by construction
  (`task_geometry.py`), 110 mm from its centre to the plate edge and 415 mm to the
  bore; the flange reaches the bore to `0.78 mm`. The open wrist-singularity
  caveat is answered: `q5 = ±90°` exactly, so `|sin q5| = 1.0`, the maximum
  distance from a wrist singularity, in every IK branch.
  One qualification, recorded rather than smoothed over: the *first* home pose
  selected under D-025 put the upper arm through the tabletop. The placement is
  reachable, but not from every IK branch — see the D-025 addendum.
- Context: The manipulator is mounted on the fixture plate, whose top face lies
  at env-local z = 0.755, and the bore entrance is at env-local
  (0.0, −0.225, 0.755) per D-022. The base position determines whether the
  insertion target falls in a well-conditioned region of the UR10e workspace,
  and it is bounded by the plate itself. Relevant measured and documented
  quantities: plate extents x ∈ [−0.250, +0.250] and y ∈ [−0.300, +0.300] from
  the verified `Tisch.usd` bounding box; UR10e base diameter 190 mm, so the base
  center can be at most y = +0.205 for the base to rest fully on the plate;
  UR10e working radius 1300 mm; a restricted near-field of approximately 200 mm
  around the base's vertical center axis, within which shoulder and axis
  configurations cannot be approached flexibly and tangential motion is
  problematic.
- Options considered: (a) base at the geometric maximum y = +0.205, flush with
  the plate edge, giving 430 mm but no material for the mounting flange and
  fasteners; (b) base at y = +0.190 with a 15 mm edge margin, giving 415 mm;
  (c) enlarge the plate in Y to permit a greater distance; (d) mount the
  manipulator beside the fixture on a separate pedestal instead of on the plate.
- Decision: Option (b). Base center at env-local (0.0, +0.190, 0.755); distance
  to the bore entrance 415 mm, purely along −Y, with base and bore both on the
  x = 0 centerline. Option (c) is rejected: the fixture is already verified, and
  altering it entails a new CAD export, re-import, re-verification and
  re-derivation of the D-022 constants against a need that has not been
  demonstrated. Option (d) is retained in reserve as the remedy should the
  working distance prove insufficient, since relocating the manipulator, not
  enlarging the workpiece table, is how physical cells resolve this.
- Rationale: At 415 mm the target clears the documented 200 mm restricted
  near-field by approximately a factor of two while remaining far inside the
  1300 mm working radius, so the configuration sits neither in the base
  singularity zone nor near the reach limit. Placing base and bore both at x = 0
  puts the approach in the arm's sagittal plane, which avoids the asymmetry that
  base rotation would otherwise introduce into an otherwise symmetric task. The
  15 mm edge margin leaves material for the mounting flange and fastening.
  Caveat: the 200 mm figure bounds the base near-field only. Wrist singularities
  depend on the concrete pose, and a downward-pointing tool at base height is a
  configuration that warrants inspection; the datasheet figure is therefore
  necessary but not sufficient, which is why the status remains `proposed`.
- Sources: Universal Robots UR10e user manual (working radius; restricted
  near-field around the base),
  https://jk.de/media/73/32/10/1718317162/Universal_Robots_UR10e_User_Manual_de_Global.pdf;
  plate extents measured via `verify_fixture_usd.py` on the training machine
  (2026-07-25); D-022 (bore entrance and plate-top constants), D-018 (fixture as
  a static asset), D-020 (UR10e articulation cfg).

## D-024: Control rate — 60 Hz, i.e. sim dt 1/120 s with decimation 2

- Date: 2026-07-26
- Status: proposed (pending verification — no environment has yet been stepped at
  this rate; the value is a documented reconciliation, not a measurement).
- Context: Two project documents state incompatible control rates.
  `docs/project_overview.md` and `docs/plan_alignment.md` name 60 Hz control at a
  physics dt of 1/120 s, which implies decimation 2. `docs/factory_mapping.md`
  adopts Factory PegInsert's `decimation = 8`, which at dt 1/120 s yields 15 Hz.
  The environment cfg cannot hold both, and the contradiction was never resolved
  in writing.
- Options considered: (a) dt 1/120 s, decimation 2 (60 Hz); (b) dt 1/120 s,
  decimation 8 (15 Hz); (c) leave it open and carry a placeholder until the
  controller and reward exist.
- Decision: Option (a). `sim.dt = 1/120`, `decimation = 2`, control rate 60 Hz.
- Rationale: Factory's `decimation = 8` is not transferable in isolation, because
  it is paired with a substantially smaller physics timestep; transplanting the
  decimation while keeping our dt silently changes the control rate by a factor of
  four. The 60 Hz figure is the one the project's own planning documents commit
  to, and it is the value against which the OSC gains (D-014) will be measured.
  Option (c) was rejected because a placeholder would have to be revisited in the
  same breath as the controller, and the contradiction would survive into that
  step unrecorded.
- Consequence to record: if contact-rich insertion later proves unstable at
  dt 1/120 s, the correct response is to reduce dt and re-derive decimation from
  the 60 Hz target, not to raise decimation.
- Sources: `docs/project_overview.md` (num_envs / control rate / dt / action
  scale); `docs/plan_alignment.md`; `docs/factory_mapping.md` (decimation 8,
  marked adopt/UNVERIFIED); user decision, 2026-07-26.

## D-025: Home pose — computed offline by closed-form IK, stored as a constant

- Date: 2026-07-26
- Status: proposed (pending verification — the joint values have not yet been
  loaded into a simulator; the IK itself is verified, see below).
- Context: The UR10e needs a defined initial joint configuration. None was
  specified anywhere in the document set. D-023's placement of the base 415 mm
  from the bore is `proposed` and explicitly flags that "wrist singularities
  depend on the concrete pose, and a downward-pointing tool at base height is a
  configuration that warrants inspection" — so the home pose is also the
  measurement that answers D-023.
- Options considered: (a) the asset's shipped `default_joint_pos`; (b) a generic
  neutral pose unrelated to the bore; (c) joint angles computed so the flange
  stands above the bore with the tool axis pointing down.
- Decision: Option (c). Target in the robot base frame: position
  (0.0, -0.415, 0.150) m, orientation diag(1, -1, -1) — the same
  180-deg-about-X convention D-022 records for the task frame, so the later
  insertion controller sees zero orientation error at the home pose. The flange
  standoff above the plate top is 0.150 m. Of the eight IK branches, the one
  selected satisfies, in order: all joints within +-pi, q3 < 0, maximal
  |sin q5|, smallest ||q||. The angles are computed offline by
  `scripts/tools/compute_home_pose.py` and transcribed by hand into the robot
  cfg, keyed by joint name rather than by position.
- Selected values (UR10e table, DH base frame as given): shoulder_pan +1.137749,
  shoulder_lift +0.887344, elbow -2.265118, wrist_1 +2.948571, wrist_2 +1.570796,
  wrist_3 -0.433047 rad.
- Rationale: Storing the pose as a constant keeps it reviewable and citable and
  leaves the environment without a runtime IK dependency. Name-keyed assignment is
  the cheapest defence against a joint-order mismatch between the DH table and the
  asset. Options (a) and (b) were rejected because neither exercises the bore
  approach, which is precisely what D-023 leaves open.
- Verification (dev PC, 2026-07-26): 200 random FK -> IK -> FK round-trips pass at
  tolerance 1e-9 for both the UR10e and UR10 DH tables; every branch reproduces
  the pose it was solved for, and the reference configuration is recovered among
  the branches each time. Independently, the joint offsets read out of
  `ur10e.usd` reproduce the assumed UR10e DH table exactly (d1 0.1807,
  a2 -0.6127, a3 -0.57155, d4 0.17415, d5 -0.11985, d6 0.11655), so the table is
  confirmed against the asset and not only internally consistent.
- Addendum (2026-07-26, measured on the training machine — **the selected branch
  is replaced**): the values above were loaded into the simulator and failed for
  two independent reasons, both now corrected in `ur10e_cfg.py`.
  - The convention hazard this entry flagged in advance is real. With the
    original angles the tool pointed down correctly (`z · (0,0,−1) = 0.999997`)
    and the standoff was right, but the flange sat at y = +0.41494 relative to
    the base where the bore is at −0.415 — same magnitude, opposite sign. The USD
    `base_link` frame is rotated by π about z relative to the DH base frame.
  - **The selected branch put the upper arm through the tabletop.** Measured
    `forearm_link` at (−0.118, −0.167, 0.452): inside the plate footprint and
    0.25 m below the plate underside, with `shoulder_pan` held 6.6° off target by
    contact and not converging over 3 s (6.538° → 6.572° → 6.593°). The selection
    criteria listed above never included plate clearance, because the fixture
    carried no collision when they were applied.
  - `scripts/tools/screen_home_pose_branches.py` (new) screens all eight branches
    against the plate volume, after validating the kinematic chain against the
    measured body positions to 0.06 mm. It also establishes the frame convention
    empirically rather than by assumption: the DH chain is driven by the USD joint
    angles directly and its output is rotated by π about z, and the USD link
    frames do not map 1:1 onto DH frames (the three wrist links correspond to DH
    frames 4, 5, 6; DH frame 3 has no USD counterpart).
  - Four branches keep the whole arm at z ≥ 0.905 m, so the flange is the lowest
    point and the arm reaches down from above. Among those, this entry's own
    remaining criteria (all |q| < π, q3 < 0, max |sin q5|, min ‖q‖) select
    **branch 3**: shoulder_pan +2.003843, shoulder_lift −1.911842, elbow
    −2.265118, wrist_1 +2.606164, wrist_2 −1.570796, wrist_3 +0.433047 rad.
  - Rule carried forward: plate clearance is a selection criterion for any future
    home pose, ranked before the pose-quality criteria. A branch that reaches the
    target is not usable if the arm passes through the fixture.
- Finding relevant to D-023: the wrist-singularity margin |sin q5| is 1.0000 for
  all eight branches and for both DH tables — q5 lands on exactly +-90 deg, the
  maximum possible distance from a wrist singularity. The shoulder margin is
  radius / d4 = 0.415 / 0.17415 = 2.38. D-023's open wrist caveat is thereby
  answered analytically; the simulator run confirms rather than discovers it.
- Open caveat: the UR DH base frame and the URDF/USD `base_link` frame differ by
  a rotation of pi about z. Redefining the base yaw shifts q1 by pi and leaves
  q2..q6 unchanged, so both q1 families are printed by the tool. Which one the
  asset uses is settled by the first environment run, not by argument.
- Sources: published Universal Robots UR10e DH parameters; `ur10e.usd` joint
  offsets read via `scripts/probe_assets.py` (training machine, 2026-07-26);
  D-022 (bore entrance, task-frame orientation), D-023 (base placement).

## D-026: Increment-1 environment semantics — commissioning stub with a numeric startup report

- Date: 2026-07-26
- Status: verified (training machine, 2026-07-26; the non-headless look and the
  five-iteration training run are still outstanding and are recorded as such).
- Context: Increment 1 assembles the first real scene — fixture plus UR10e at the
  D-022/D-023 constants — with no peg, gripper, controller or reward. Its purpose
  is not to learn anything but to make the placement checkable. That requires
  deciding what the environment does in the absence of a task, and how the check
  is performed.
- Options considered: (a) a visual check in the viewport; (b) an environment that
  already carries a provisional reward and controller, so the increment ends
  closer to the real task; (c) a stub with zero reward and unapplied actions,
  whose only behaviour is a numeric report of the quantities D-022/D-023 assert.
- Decision: Option (c), realized as follows. Actions are declared but not applied
  (`_apply_action` is a no-op); reward is identically zero; termination is timeout
  only. `action_space = 6` and `observation_space = 12` (joint positions and
  velocities) are deliberate interim values that carry no claim about the final
  observation design, which remains open decision 5 and is constrained by D-005.
  The scene uses `num_envs = 128` and `env_spacing = 3.0`; the plate spans
  0.6 m in y, so 3.0 m leaves ample separation while keeping 128 environments
  within a workable footprint. A startup report prints the placement quantities at
  steps 2, 60 and 180 after reset (0.03 s, 1 s, 3 s at the D-024 rate of 60 Hz),
  for environment 0 **and** environment 1.
- Rationale: A viewport check cannot distinguish a correct placement from one that
  is wrong by millimetres, and it cannot detect the specific failure this increment
  exists to exclude — constants applied as absolute world coordinates instead of
  env-local ones, which is invisible with a single environment and is why the
  report covers two. Option (b) was rejected because it would confound a placement
  error with a reward or controller error, against the project rule of changing one
  thing at a time. Reporting at several times rather than once is a direct
  consequence of a misdiagnosis during this increment: the single early print
  landed 33 ms after reset, and an unsettled transient was read as contact with the
  plate.
- Measured consequence, accepted rather than tuned away: with gravity enabled and
  the shipped implicit-PD gains, the arm does not hold the commanded pose exactly.
  It converges to a steady offset — `shoulder_lift` 0.03404 rad, `elbow`
  0.02185 rad, `wrist_1` 0.01168 rad — which with the shipped stiffnesses (1320,
  600, 216) corresponds to holding torques of 44.9, 13.1 and 2.5 Nm, all in the
  direction of gravity. The readings at 1 s and 3 s agree to the sixth decimal, so
  this is a converged equilibrium, not a transient. A proportional controller
  cannot hold a loaded pose without gravity compensation; this is the expected
  behaviour of the configuration D-020 flagged as a deliberate commissioning
  deviation, and it is precisely the implicit gain test that deviation was for.
  The acceptance criteria are therefore read in two parts: the kinematic claim of
  D-022/D-023 is judged on the reading before gravity acts (XY 0.78 mm, standoff
  0.1489 m), and the steady-state droop (XY 13.4 mm, standoff 0.1326 m, largest
  joint deviation 1.95 deg) is recorded as a property of the uncompensated gains.
  Gains are not raised to meet the criterion: physical parameters are measured, not
  fitted, and D-020 already provides for zeroing the arm gains in the OSC increment,
  where the controller assumes responsibility for holding the pose.
- Sources: increment-1 startup report, training machine, 2026-07-26 (marker
  `increment1-scene-2026-07-26e`, runs with `--num_envs 1` and `--num_envs 4`);
  D-020 addendum 2 (measured API surface, gravity deviation and shipped actuator
  gains); D-022, D-023 (the constants under test); D-024 (60 Hz); D-025 addendum
  (home pose and the branch that had to be replaced); D-005 (observation
  constraints); open decision 5 (observation vector layout).

## D-027: Task geometry moved into `proxytask_tasks_cfg.py` with a fixed/held asset split

- Date: 2026-07-26
- Status: accepted
- Context: `task_geometry.py` held the D-022/D-023 placement constants as plain
  module values. Factory keeps the equivalent data in `factory_tasks_cfg.py` as
  `@configclass` dataclasses (`FixedAssetCfg`, `HeldAssetCfg`), and
  `docs/factory_mapping.md` records that structure as "adopt". The conversion was
  deferred out of increment 1 for a concrete reason: `ur10e_cfg` imports the
  geometry module, so the geometry module must not import back.
- Options considered: (a) keep plain constants and add peg values to them;
  (b) replace them with config classes; (c) keep the constants and add the config
  classes alongside, in one module that stays a dependency leaf.
- Decision: Option (c). `task_geometry.py` is renamed (git mv, history preserved)
  to `proxytask_tasks_cfg.py`, keeps every constant verbatim, gains the bore and
  stage-1 peg constants, and adds `FixedAssetCfg` / `HeldAssetCfg` plus
  `resolve_peg_robot_usd_path()`. Its only imports are `os` and
  `isaaclab.utils.configclass`, so it remains a leaf and the cycle cannot arise.
  Verified by search, not assumed: the module's only importers are `ur10e_cfg`,
  `proxytask_env_cfg` and `proxytask_env`; the numpy-only offline tools do not
  touch it and are therefore unaffected by the new isaaclab dependency.
- Rationale: The constants are what the startup report and the offline tools
  read, and turning them into class attributes would have bought nothing but an
  extra indirection; the config classes are what the peg's fixed/held split
  needs. Keeping both in one leaf module satisfies the Factory structure without
  the import cycle that blocked the conversion. Derived values stay derived:
  `RADIAL_CLEARANCE` is computed from the two diameters rather than repeated as
  1.0 mm, in the same spirit as deriving the robot base height from the plate top.
- Deliberate asymmetry: `resolve_fixture_usd_path()` stays in
  `proxytask_env_cfg.py`. `scripts/verify_fixture_spawn.py` imports it from
  there, and moving a verified function to make two resolvers look alike would
  have broken a working script for cosmetic symmetry. The reason is recorded in
  the new resolver's docstring so the split does not read as an oversight.
- Sources: `docs/factory_mapping.md` §1 (asset config pattern, "adopt");
  Isaac Lab v2.3.2 `factory_tasks_cfg.py` (structure only); HANDOFF open item
  carried since 2026-07-26.

## D-028: Peg/bore passability measured by a self-calibrating penetration probe

- Date: 2026-07-26
- Status: **verified** (training machine, 2026-07-26, verdict PASSABLE). See the
  measured result and the protocol correction at the end of this entry.
- Context: The peg increment's headline question is whether a Ø28 mm peg can
  occupy the Ø30 mm blind bore of the fixture's exact triangle-mesh collider.
  Nothing solid has ever been put into that hole, and D-020 addendum 2 deferred
  the convex-hull worry to precisely this point. Two properties of the existing
  setup rule out the obvious test: the shipped PD gains droop about 13 mm at
  steady state (D-026), an order of magnitude larger than the 1.0 mm radial
  clearance, so a servoed descent would measure the gains; and writing joint
  states kinematically teleports the arm regardless of collision, so a completed
  descent is not by itself evidence of anything.
- Options considered: (a) servoed descent under the shipped gains; (b) kinematic
  descent, treating a completed sweep as success; (c) kinematic descent with a
  free physics step at each waypoint, judged against a baseline measured where
  contact is geometrically impossible, plus a deliberately misaligned control
  pass.
- Decision: Option (c), implemented as `scripts/verify_peg_passability.py`. At
  each 1 mm waypoint the arm is written and held for four steps, then one step
  runs with nothing written; overlap with plate material appears there as joint
  motion and a peg velocity spike. The descent starts at the home pose, 100 mm
  above the plate, and those contact-free waypoints define the threshold
  (5× the observed maximum, with an absolute floor). A second pass offsets the
  peg axis by 4 mm, so its edge misses the Ø30 mm bore and it *must* react.
  Verdicts: PASSABLE, BLOCKED, COLLISION_ABSENT (the control failed to react,
  so the collider is inert and a clean centred pass would prove nothing), or
  INCONCLUSIVE. Metrics are written to JSON per the CLAUDE.md rule.
- Rationale: The negative control is what makes a clean result meaningful; without
  it, "the peg went in" is indistinguishable from "nothing collides with anything".
  Calibrating on the robot's own contact-free noise avoids inventing a force or
  deviation threshold, which the project rules forbid guessing. The script drives
  the real `ProxytaskEnv` rather than building its own scene, so no untested
  scene-construction code sits on the critical path and the geometry measured is
  the one training will use; the env is not modified. IK branches are selected by
  continuity with the previous waypoint rather than by index, because branch
  ordering depends on the target and an index valid at the home pose need not stay
  valid 100 mm lower.
- Offline evidence (dev PC, no Isaac): 128 of 128 waypoints solve in both passes;
  worst FK position error 3e-16 m; largest joint step 2.1 mrad against a 50 mrad
  continuity guard; and the first waypoint reproduces the training-machine-verified
  home pose of `ur10e_cfg.py` to 4.9e-07 rad, which validates the Rz(π) frame
  convention against a measured artifact rather than only internally.
- Sources: D-026 (measured droop); D-020 addendum 2 (deferred convex-hull
  question); `scripts/tools/screen_home_pose_branches.py` (the p_env = BASE +
  Rz(π)·p_DH convention, established by matching measured body positions);
  docs/task_specification.md (25 mm success depth, 30 mm blind bore).
- **Measured result (training machine, 2026-07-26): verdict PASSABLE.** All
  deviations are the free-step maximum joint deviation in rad.

  | Quantity | Value | Ratio to own contact-free noise |
  |---|---|---|
  | Centred pass, contact-free (≥ 5 mm clear) | 0.00272775 | 1.00 |
  | Offset pass, contact-free (≥ 5 mm clear) | 0.00272775 | 1.00 |
  | Centred pass, inside the bore to 25 mm | 0.00286937 | **1.05** |
  | Offset pass (4 mm), at the entrance | 0.01303017 | **4.78** |

  The peg descended to 27 mm — past the 25 mm success threshold, short of the
  30 mm bottom — with the free-step reaction indistinguishable from the
  contact-free case, while the misaligned control rose to 4.78× from 1 mm depth
  onward, which is the plate top surface the offset peg's edge must strike. The
  verdict is stable for any reaction factor between 1.05 and 4.78; the value
  used, 2.0, sits near the middle of that interval, so the conclusion does not
  rest on the constant.

  The two contact-free maxima being **equal to six significant figures** is what
  makes the argument close: it rules out the one alternative explanation, namely
  that the offset pass's larger reading came from the different gravity torque
  of a laterally shifted arm rather than from contact. A 4 mm shift across a
  415 mm reach leaves the free-step noise unchanged.

  This also closes, by measurement rather than by argument, the question
  deferred in **D-020 addendum 2**: the blind bore is not sealed by a convex
  hull. The fixture's exact triangle mesh (`add_fixture_collision.py`,
  approximation `none`) admits the Ø28 mm peg at the planned 1.0 mm radial
  clearance.
- **Protocol correction, recorded because the first run failed on it.** The
  entry above originally judged a reaction by a preset threshold of 5× the
  centred pass's contact-free maximum, and both passes were calibrated on that
  one baseline. The first measurement returned COLLISION_ABSENT from data that
  says the opposite: the control reached 4.78× and missed the cut by 4.6 %.
  Two defects, one of substance and one of method:
  - **Substance:** calibrating the offset pass on the centred pass's noise
    confounds contact with the configuration change the offset itself causes.
    Each pass is now calibrated against its own contact-free waypoints, and the
    verdict rests on the paired comparison of two runs differing by 4 mm of
    lateral position and nothing else.
  - **Method:** the factor 5 was chosen before any data existed, which is the
    same error as guessing a physical parameter. An acceptance threshold has to
    be derived from the measured noise it must discriminate against. The script
    now reports the interval of factors that yield the same verdict, so the
    reader sees the margin instead of trusting a number.

  Lowering the threshold until the run passed was available and was rejected:
  the fix had to be the missing measurement, not a more permissive cut.

---

## D-029: Square-peg pivot — 30×30×50 mm prism into a 32×32×35 mm blind pocket

- Date: 2026-07-28
- Status: proposed (pending verification on the training machine)
- Revised in part by D-033 (2026-08-05): the **asset production** (Creo import →
  script generator, hence also the file name and the collision-approximation
  requirement) and the **curriculum axis** (peg side → pocket side) are
  superseded. The geometry, gate, observation, reward, reset distribution and
  the target rung (30 mm peg in a 32 mm pocket) stand unchanged.
- Context: The demo sprint solved the round Ø25 mm peg task. A rotationally
  symmetric peg leaves the yaw degree of freedom task-irrelevant: the goal set
  is SO(2)-symmetric about the insertion axis, the 6-DOF arm is redundant for
  the 5-dimensional task, and nothing about orientation search is learned or
  measurable. Replacing the cylinder with a square prism makes yaw
  task-relevant (goal symmetry shrinks to the cyclic group C4), which is the
  thesis's next increment. Branch `square-peg-insertion`, cut from
  `demo-insertion-sprint` @ a7c7ebb. A numbering note: D-030 through D-032
  are already referenced by the increment-3 branch; the next free ID on this
  branch is **D-033**.
- Options considered:
  (a) separate container fixture with a square opening standing on the table —
  rejected by the user in favour of (b);
  (b) re-import the same table with a square pocket cut at the old bore
  centre — chosen; one asset, one spawn path, all verified table numbers keep
  their meaning;
  (c) stage the MDP changes (observation/reward/reset) after a geometry-only
  etappe — rejected: a square peg without yaw randomization, yaw observation
  and yaw reward is behaviourally the round task with a different mesh, so
  the intermediate stage would verify nothing the demo sprint has not.
- Decision:
  - **Geometry.** Peg 30×30×50 mm square prism (ABS, ρ = 1040 kg/m³:
    m = ρa²L = 46.80 g, I_axial = ma²/6 = 7.02e-6 kg·m²,
    I_trans,COM = m(a²+L²)/12 = 1.326e-5 kg·m²; Steiner to the flange as
    before). Pocket 32×32 mm, 35 mm deep, blind, centred where the bore was;
    success depth stays 25 mm. Clearance 1.0 mm per axis at φ = 0; free-yaw
    window ±(asin(b/(a√2)) − 45°) = ±3.96° about each of the four C4
    orientations; tilt window ≈ 2.3°.
  - **Asset convention.** The re-imported table (`tisch_square.usd`) carries
    its origin ON the opening plane at the pocket centre. The derivation
    chain in `proxytask_tasks_cfg.py` is flipped accordingly: FIXTURE_POS =
    OPENING_ENTRANCE_POS = (0, −0.225, 0.755); the plate top is the fixture
    z. The env-cfg resolver accepts `tisch_square.usd` ONLY — the round
    `tisch.usd` is deliberately not a fallback, because silently spawning it
    would void every result while everything still runs.
  - **Gate.** Lateral containment is a four-corner test: the corners of the
    tip cross-section, c_i = tip ± (a/2)x̂ ± (a/2)ŷ, must each lie within the
    pocket half-side on both XY axes (Chebyshev). One test subsumes offset,
    yaw and tilt; a centre-distance test would call the geometrically
    impossible 45° pose "inside". Pocket-volume and alignment conditions stay.
  - **Observation.** 21-dim: the demo layout plus (cos 4φ, sin 4φ) of the peg
    yaw. C4-invariant by construction — the four insertable orientations are
    one point in observation space, not four separate lessons — continuous
    (no ±45° seam), and (1, 0) at every goal orientation.
  - **Reward.** Seventh term −w_yaw·(1 − cos 4φ)/2 with w_yaw = 2.0, derived:
    unwinding 45° over ~40 steps yields ~0.04·w_yaw per step against ~0.01 of
    action cost — parity at w_yaw = 0.25, safety factor 8, an order of
    magnitude under the depth term. Closes the yaw-parking attractor
    (vertical, centred, 45° turned: approach ≈ 0, align ≈ 0, insertion
    impossible — the exact yaw analogue of the sideways-parking attractor the
    demo sprint met). The out-of-bounds termination price includes the yaw
    term (termination neutrality, see the WP0 correction).
  - **Reset distribution.** Per-joint noise: ±45° (0.7854 rad) uniform on
    wrist_3, ±0.01 rad on the other five. wrist_3 is a pure tool-axis yaw
    actuator at the home pose, so the wide noise moves φ and nothing else.
    Without it every start lies inside the ±3.96° free-yaw window, the policy
    never needs to turn, and a 100 % success rate would be a non-result.
  - **Curriculum** (pocket fixed at 32 mm; PROXYTASK_PEG_SIDE_MM, bound
    < 32): 22.6 → 4.7 mm/±45° (stage 0: diagonal ≤ pocket side, C4 window
    inactive — separates "reward/obs works" from "yaw is hard") · 24 →
    4.0/±25.5° · 26 → 3.0/±15.5° · 28 → 2.0/±8.9° · 30 → 1.0/±3.96°
    (target). Assets named `ur10e_peg_s{mm:g}.usd`; run tags `s{mm:g}`.
  - **Deletions.** The cylinder authoring path, `--segments`, the faceting
    metric, `add_bore_task_frame.py`, and the legacy
    `PROXYTASK_PEG_DIAMETER_MM` variable (now a hard error) are removed.
    Old branches preserve the round task.
- Rationale: The square task is the smallest change that makes orientation
  search real: same arm, same table, same reward skeleton, one new goal
  symmetry. Encoding the symmetry in the observation ((cos 4φ, sin 4φ))
  rather than asking the policy to discover it follows the MDP-homomorphism
  argument (state abstraction over the goal's symmetry group); the C4
  window/corner-gate numbers are derived, not tuned. The box mesh is authored
  axis-aligned to the flange x/y — NOT as a 4-segment cylinder, which yields
  a 45°-rotated square and two cancelling yaw zero points between mesh and
  pocket (ranked risk 2).
- Ranked risks (mitigations in place): (1) a convex-hull collider seals the
  non-convex blind pocket and mimics an exploration failure —
  `--approximation none` is contractual and the passability yaw control
  detects it; (2) two yaw zero points — axis-aligned mesh plus the φ report
  line; (3) sham success without wide yaw noise — reset design above;
  (4) jump size at 1.0 mm/±3.96° — curriculum stage 0 first; (5) 13.4 mm
  gravity droop = 13× clearance (D-026) — first suspect on failure, answer
  would be D-014/OSC, not reward changes; (6) yaw parking — w_yaw, asserted
  offline; (7) time budget (≥ 40 of 240 steps for a 45° turn) — measured
  before any PPO tuning.
- Sources: geometry and formulas as in this entry (derivations in
  `proxytask_tasks_cfg.py` comments); offline assertions
  `scripts/check_demo_reward_math.py` (41 checks PASS on the dev PC, numpy
  stand-in); reset spread `scripts/reset_noise_spread.py` (lateral 0.00 mm
  under pure wrist_3 noise; folded-yaw p95 42.7° vs the 3.96° window).
  Everything else UNVERIFIED until the training-machine checkpoint chain runs.

---

## D-033: Generated box-composed pocket table; the curriculum runs on the pocket

- Date: 2026-08-05
- Status: proposed (pending verification on the training machine)
- Context: D-029 assumed the square pocket would arrive as a Creo re-import
  saved as `tisch_square.usd`. That path carries the whole import chain
  (metersPerUnit, flattening, instancing, referenced-context checks) plus a
  mandatory `--approximation none` collision step, and it makes every pocket
  size a manual CAD round trip. It also placed the curriculum on the **peg**
  side length — but the peg is welded into the robot USD (D-019), so each rung
  needs its own authored robot asset, and the peg cannot change within a
  training session at all. Separately, D-029's ranked risk 1 (a convex hull
  seals the non-convex blind pocket and mimics an exploration failure) remained
  a *detectable* risk rather than an excluded one.
- Options considered:
  (a) Creo re-import per D-029 — rejected: one CAD round trip per curriculum
  rung, and the sealed-pocket risk survives as something to test for rather
  than something that cannot happen;
  (b) generate the table from a script as a single mesh — rejected: a
  non-convex mesh reintroduces exactly the approximation question the import
  path had, only in Python;
  (c) generate the table as a composition of axis-aligned boxes, each a
  primitive collider — chosen;
  (d) keep the peg as the curriculum axis — rejected, see Decision.

> **NOTE (2026-08-06): this entry is incomplete.** It ends here; `Decision`,
> `Rationale` and `Sources` were never written. The decision itself is in
> force and is implemented — the generator, the pocket-side curriculum
> variable and the sidecar asset check all exist and are verified (see
> `docs/square_task_commissioning.md`) — but the entry must be completed from
> the implementation before any of it is cited in the thesis.

- Addendum (2026-08-06, verified on the training machine): **the rung ladder
  was shortened to 45 → 36 → 32.** The ladder recorded in
  `proxytask_tasks_cfg.py` was 45 → 40 → 36 → … → 32. The rung at 40 mm was
  skipped after quantifying what each rung actually demands. Since the reset
  yaw is uniform over ±45° and the free-yaw window follows from the geometry,
  the share of episodes that cannot be solved without rotating is 0 % at
  b = 45, 43 % at b = 40, 71 % at b = 36 and 91 % at b = 32. At b = 40 the
  majority of episodes (57 %) still admit insertion at the reset orientation,
  so that rung largely repeats the preceding task rather than adding the
  capability under test. Jumping from 45 straight to 32 was rejected for the
  opposite reason (D-029 risk 4): a failure there could not be attributed
  between the size of the jump and an insufficient `reward_w_yaw`, which would
  make the run unreadable. b = 36 forces rotation in the clear majority of
  episodes while keeping the lateral tolerance at three times the target's.
  Measured outcome: stage 0 at 99.9 % recent success, b = 36 at 100 % over 751
  replay episodes, and the target rung b = 32 at **99.15 %** with 77 789
  episodes to the 90 % threshold against 17 974 at stage 0.
- Decision:
  - **Asset production.** `scripts/author_tisch_square.py` composes the table
    from nine axis-aligned cuboids: four plate pieces tiling the plate around
    the pocket, the pocket floor, four legs. Each is a `UsdGeom.Cube` carrying
    its own `UsdPhysics.CollisionAPI`. No mesh, no approximation step, no
    external references, one file. The script verifies the stage it authored by
    measuring the prims back (bbox, opening measured between the wall prims,
    depth, collider count, op-free default prim) and writes the result to a
    `.author.json` sidecar; `scripts/check_tisch_geometry.py` checks the same
    box layout offline, without Isaac.
  - **Curriculum axis.** The **pocket** is the curriculum variable, the peg
    stays at 30 mm. Rungs (clearance per axis / free-yaw window):
    45 mm → 7.5 mm/±45° (stage 0: the 42.43 mm diagonal fits at any yaw, C4
    inactive) · 40 → 5.0/±25.5° · 36 → 3.0/±13.1° · 34 → 2.0/±8.3° ·
    32 → 1.0/±3.96° (the D-029 target, unchanged). Set per run via
    `PROXYTASK_POCKET_SIDE_MM`; `PROXYTASK_PEG_SIDE_MM` remains for special
    experiments. Each rung is its own (resume) run, so no single run sees a
    changing MDP.
  - **Name is contract, and the asset is checked against the cfg.** Assets are
    `tisch_square_b{mm:g}.usd`, run tags are `s{peg}b{pocket}` (e.g. `s30b45`),
    and the metrics file records `pocket_side_m`. The resolver refuses a
    size-less `tisch_square.usd`. Because `PROXYTASK_TISCH_USD` can bypass the
    filename entirely, the startup report additionally compares the generator's
    sidecar (`pocket_side_m`, `pocket_depth_m`) against the configured fixed
    asset — the fixture counterpart of the peg check D-028's context
    introduced, now warranted because the fixture is what varies.
  - **Unchanged.** The env (four-corner gate, 21-dim observation, yaw reward,
    ±45° reset), the asset origin convention, the spawn translation
    (0, −0.225, 0.755), the bbox 0.500 × 0.600 × 0.755 m over z ∈ −0.755…0,
    and therefore `verify_fixture_usd.py` and `verify_fixture_spawn.py`.
  - **Not in scope.** Per-episode pose randomisation of the fixture (the
    Factory pattern) and a tilted pocket are a later increment, **D-034**. The
    spawn stays a static collider (D-018).
- Rationale: A box is an analytic collider in PhysX — the contact is computed
  against the primitive, not against a hull fitted to a mesh — so a composition
  of boxes has no approximation stage in which the pocket could be closed. This
  converts D-029's risk 1 from *mitigated and tested for* to *impossible by
  construction*, which is the stronger form of the same argument the four-corner
  gate uses against a centre-distance test. Because the pocket walls are each a
  single flat box face, the sliding surfaces have no internal seams; the only
  seams are the four corners, where a real pocket has corners.
  Moving the curriculum onto the pocket follows from where the difficulty
  actually lives: both the lateral clearance (b − a)/2 and the yaw window
  asin(b/(a√2)) − 45° depend on the *pair*, so either side can carry the rung —
  but the peg is welded into the robot USD and needs a re-authored robot per
  value, while a pocket is a regenerated table in seconds. Stage 0 at b = 45 mm
  is chosen so the diagonal fits (C4 inactive), which separates "reward and
  observation work" from "yaw is hard" exactly as D-029's 22.6 mm peg rung did,
  without a second robot asset. Costing this is one geometric consequence worth
  naming: a wider pocket also means a longer unguided drop to the floor, so the
  depth term is easier to satisfy at stage 0 than at the target rung — stage
  results are not comparable across rungs, which is why each rung is its own run
  with its own metrics file.
- Ranked risks: (1) box-collider edges — the pocket mouth is a sharp box edge
  rather than a chamfered CAD edge, so contact at the opening may differ from
  the imported asset; measured directly by `verify_peg_passability.py`, and on
  any anomaly the edge is the first suspect, not the reward; (2) the visuals are
  cuboids, so the table looks coarse — accepted deliberately, because collision
  and visual are the same prims and therefore cannot disagree ("looks right,
  collides wrong" is excluded); (3) the stage-0 yaw control would have reported
  `SQUARENESS_ABSENT` for a correct asset, since the 45° control assumes the
  diagonal cannot pass — fixed by deactivating that control where the diagonal
  fits, and reported as `yaw_control: inactive (diagonal fits)`;
  (4) non-stationarity across rungs — avoided by one run per rung.
- Sources: geometry and formulas as in this entry, derived in
  `proxytask_tasks_cfg.py`; box layout asserted offline by
  `scripts/check_tisch_geometry.py` (exact tiling, zero pairwise overlap, open
  pocket mouth, contract bbox, for 32/34/36/40/45/60 mm — PASS on the dev PC);
  rung table reproduced independently from the derived constants; asset contract
  in `docs/asset_contract_tisch.md` (D-033 addendum). USD collision schema and
  Isaac Lab collider documentation: https://isaac-sim.github.io/IsaacLab/
  (version picker 2.3) per the CLAUDE.md reference rule. Everything touching
  Isaac is UNVERIFIED until the training-machine chain runs.

## D-035: Generalisation probe by fixture translation — separate replay runs via a play.py offset flag

- Date: 2026-08-06
- Status: proposed (pending verification)
- Context: The b = 32 policy reached 99.15 % recent success (run
  2026-08-06_02-25-24_s30b32) with the fixture at one fixed pose throughout
  training. Whether the policy learned the pocket-relative task or memorised
  one joint-space location is undecided and determines how the result must be
  qualified in the thesis (HANDOFF.md next step 1). The observation, gate and
  reward are already formulated relative to the pocket opening
  (`_entrance_local` from `cfg.opening_entrance_pos`), so they remain correct
  under a pure translation; the robot base position and the home joint
  configuration are independent constants and stay unmoved, forcing the policy
  into joint configurations it never trained on.
- Options considered: (a) separate replay runs, one per offset, with a
  `--fixture-offset DX DY` CLI flag in `scripts/rsl_rl/play.py`; (b) a
  scheduled in-run position switch (e.g. 150 episodes at one offset, then 150
  at the mirrored offset) inside the environment; (c) a 45° pocket rotation
  instead of a translation.
- Decision: (a). `play.py` gains `--fixture-offset DX DY` (metres, capped at
  0.05 m per axis), which shifts `env_cfg.fixture_pos` and
  `env_cfg.opening_entrance_pos` together — they are the same point by
  construction — with z untouched, before the environment is created. Each
  offset run writes its metrics into its own directory
  (`replay_dx{±NNNN}mm_dy{±NNNN}mm`) instead of the shared `replay/`. The
  environment code is not modified; the policy is evaluated exactly as
  trained, without retraining. Offset series: mirrored pairs per axis, ±2 cm
  then ±5 cm, each axis separately. Abort criterion per offset: 0 % success
  and ≈0 mean insertion depth after ~100 episodes ends the run as a recorded
  negative; otherwise ~750 episodes, matching the b = 36 replay sample.
- Rationale: (b) requires new episode-scheduling logic inside the environment
  — code that would itself need commissioning — and merges two positions into
  one metrics file, which would have to be separated after the fact; this is
  the same metric-mixing failure mode that cost the b = 36 training figure
  (PROBLEMS.md 2026-08-06). Replays without retraining are cheap, so one run
  per offset yields the same information with zero environment changes and
  one unambiguous success rate per position. Mirrored offsets retain the
  discriminating idea behind (b): a policy that memorised the trained location
  fails on the far side of the pocket, and passing both signs on both axes is
  the stronger evidence. (c) was deferred deliberately: under C4 symmetry a
  45° pocket rotation is either unobservable (task unsolvable) or a
  relabelling of the four target orientations (passes trivially); it audits
  the implementation, not the policy, and belongs to D-034. The per-offset
  metrics directory closes the replay-overwrite failure mode for a series of
  replays on the same checkpoint.
- Sources: HANDOFF.md next step 1 (2026-08-06) for the probe requirement and
  the deferral argument for the rotation; PROBLEMS.md 2026-08-06 (replay
  overwrote training metrics) for the per-offset directory; constants audited
  in `proxytask_tasks_cfg.py` (`ROBOT_BASE_POS` independent of `FIXTURE_POS`
  in x/y) and `proxytask_env.py` (entrance-relative observation/gate/reward,
  z-only `depth`/`below_plate`). UNVERIFIED: authored on the dev PC; verified
  only when the offset series has run on the training machine.
- Addendum (2026-08-06, training machine): **probe run and decided at the
  first offset.** At (+0.02, 0.00) m the policy failed every episode (0 %
  success) and pressed the peg onto the *trained* pocket location, offset
  from the actual pocket by the full fixture offset, identically across
  episodes. That is the signature of a memorised joint-space trajectory, not
  of a pocket-relative skill; the mirrored and ±5 cm runs were dropped as
  no longer discriminating (user decision, 2026-08-06). The mechanism — the
  pocket-relative channels 12:15 are informationally redundant given the
  absolute joint channels when the fixture never moves, so training induces
  no dependence on them — and the consequence are recorded in D-034, which
  this measurement converts from insurance into a necessary correction.
  Verdict on the headline number: the 99.15 % of run
  `2026-08-06_02-25-24_s30b32` measures the task *at the trained fixture
  pose only* and must be qualified as such in the thesis.

## D-034: Per-episode fixture-pose randomisation via a kinematic rigid-body fixture

- Date: 2026-08-06
- Status: proposed (pending verification)
- Context: The D-035 probe measured 0 % success at a 2 cm fixture offset: the
  b = 32 policy presses the peg onto the trained location regardless of where
  the pocket is, i.e. it memorised a joint-space trajectory. The mechanism is
  identifiable in the observation design: with the fixture at one fixed pose,
  the pocket-relative channels 12:15 are a deterministic function of the
  absolute joint channels 0:6 (forward kinematics), carry zero exclusive
  information during training, and the network has no pressure to use them.
  The Factory template trains its PegInsert with the socket pose randomised
  per episode (fixed_asset_init_pos_noise = [0.05, 0.05, 0.05] m,
  fixed_asset_init_orn_range_deg = 360, isaaclab_tasks v2.3.2,
  factory_tasks_cfg.py, verified in source 2026-08-06), which makes the
  goal-relative observation the only source of pocket information. The
  project fixture, however, was a static collider (D-018) cloned identically
  into every env (replicate_physics), with no runtime pose to write.
- Options considered: (a) per-env fixed fixture positions at scene build
  (breaks the joint-goal confound for the shared policy, but requires
  disabling physics replication or per-env spawning, and gives no per-episode
  variety, blocking the later yaw randomisation); (b) fixture as a KINEMATIC
  rigid object, root pose teleported at episode reset (Factory pattern);
  (c) scheduled in-run position switches (rejected in D-035 already: merges
  positions into one metrics file, needs env scheduling logic).
- Decision: (b). The fixture spawns as before into env_0, then
  define_rigid_body_properties applies a RigidBodyAPI with
  kinematic_enabled=True — necessarily the define_ variant: the generated
  table carries no RigidBodyAPI and the spawner's modify_ path is a silent
  no-op on such a prim (verified in isaaclab v2.3.2 sim/schemas/schemas.py:
  modify_rigid_body_properties returns False when the API is absent). The
  asset is wrapped as a RigidObject, registered with the scene, and
  _reset_idx samples per reset a uniform xy offset (cfg.fixture_pos_noise_xy,
  default 0) and a yaw (cfg.fixture_yaw_noise_rad, fixed 0 this increment)
  and writes entrance buffer and fixture root pose from the same tensor. The
  entrance became a per-env (N,3)+yaw buffer consumed by observation, gate
  and reward. The robot base and home joint configuration stay fixed
  (deliberate inversion of D-018 addendum 1's "move both together" — the
  pocket moving relative to the trained arm configuration is the point). The
  observation layout stays unchanged in this increment (one hypothesis per
  change: randomisation alone must force the network onto channels 12:15).
  Run tags append rxy{mm} and the metrics file records both noise fields.
  Training plan: primary path is a fresh curriculum (b = 45 -> 36 -> 32) with
  xy noise 0.05 m from stage 0; a resume of the memorised b = 32 checkpoint
  under randomisation runs additionally as a fine-tuning-vs-retraining
  comparison, not as a gate. Pre-registered criteria: recovered means recent
  success >= 0.95 on the rung; failed means < 0.5 after the ~18k episodes
  stage 0 needed. Commissioning before any training: entrance-buffer identity
  check in the startup report, regression replay of the b = 32 checkpoint at
  noise 0 (must reproduce ~99 %; on failure the kinematic contact is the
  first suspect, not the policy), and verify_peg_passability.py
  --fixture-offset at a shifted pose. Yaw randomisation and the tilted pocket
  remain this entry's second half and are NOT enabled: channels 19:21 measure
  the peg yaw in world frame and the corner gate assumes an axis-aligned
  pocket; both are marked in code and must become pocket-relative first.
- Rationale: A kinematic rigid body is immovable by contact forces (PhysX
  drives it) yet has a writable root pose, which is exactly the combination
  the randomisation needs; it is the mechanism of the template this project
  adopted (D-012), works under physics replication, and carries the second
  half (yaw/tilt) without further structural change. Per-episode sampling
  makes the confound-free information in channels 12:15 exclusive: no
  function of the joint state alone predicts the pocket pose any more, so a
  policy that solves the randomised task must read the goal-relative
  channels — which is the property D-035 showed the fixed-pose training
  never induced. The noise-0 regression replay separates "refactor changed
  the physics" from "policy needs the randomisation" with one measurement.
- Sources: D-035 measurement (replay run of 2026-08-06, replay_dx+0020mm
  directory, 0 % success); isaaclab v2.3.2 source: factory_tasks_cfg.py
  (socket randomisation values) and sim/schemas/schemas.py
  (modify-vs-define rigid-body behaviour), both read 2026-08-06;
  docs/factory_mapping.md (observation comparison, v2.3.2 provenance);
  Isaac Lab 2.3 documentation (RigidObject, write_root_pose_to_sim), per the
  CLAUDE.md reference rule. UNVERIFIED: authored on the dev PC; verified
  when the commissioning chain and the first randomised training run on the
  training machine.

---

## D-036: Pocket tilt as fixture root-pose rotation, measured in the pocket frame

- Date: 2026-08-07
- Status: proposed (pending verification)
- Context: The orientation half of the generalisation question (D-034's second
  half). The user directed (2026-08-07) that the pocket be tilted out of the
  horizontal plane ("angewinkelt"), first as a pure evaluation probe of the
  trained rxy50 policy on branch generalisation-angle-probe, training only if
  the probe fails. The measurement layer assumed an upright pocket throughout:
  the four-corner gate tests world-xy containment, depth is measured along
  world z, alignment against world-down, and observation channels 12:15 are a
  world-frame offset. At a tilt of only 5 deg a world-frame gate misjudges the
  lateral position by about depth * sin(tilt), up to ~3 mm over the 35 mm
  pocket depth -- three times the 1.0 mm clearance -- so success rates measured
  without a frame change would be artefacts of the measurement, not properties
  of the policy.
- Options considered: (a) author a new table asset with a tilted pocket cut
  into a level plate (requires oriented box colliders, a new asset contract
  and one generated asset per angle; the axis-aligned-boxes-only construction
  argument of D-033 no longer applies); (b) rotate the WHOLE existing fixture
  about its origin -- which by construction (D-033) is the pocket centre on
  the opening plane -- via the root pose, and transform the measurement layer
  into the pocket frame; (c) tilt the fixture but leave the measurement layer
  world-frame (rejected outright: metrics wrong by more than the clearance,
  see Context).
- Decision: (b). One static tilt angle per run (cfg.fixture_tilt_rad), applied
  as the spawn orientation about the env y axis through the pocket centre;
  positive tips the +x plate side down. The kinematic fixture keeps its spawn
  pose because the D-034 reset write only runs when a noise range is > 0, and
  combining tilt with the D-034 noise ranges is rejected in __init__ (the
  reset write composes a yaw-only quaternion and would silently drop the
  tilt). _peg_geometry transforms the tip-to-entrance offset and the peg axes
  by R^T into the pocket frame; gate, depth, alignment, phi and observation
  channels 12:15 are computed from the transformed quantities, so they measure
  against the tilted pocket. At tilt 0 the transform object is None and every
  expression is the exact pre-D-036 code, keeping the noise-0 replay
  byte-identical as the regression case. compute_rewards is untouched: its
  approach term uses only the tip-entrance distance, which is
  rotation-invariant, and every other term consumes the already-transformed
  inputs -- the 41 reward tests remain valid. play.py exposes the probe as
  --fixture-tilt DEG (capped at 15 deg; beyond that the plate edge sweep and
  the reachable alignment envelope are unchecked), composes with
  --fixture-offset, forces the pose noise to 0, writes one replay directory
  per pose (replay[_dx..][_tilt..]) and the metrics file records
  fixture_tilt_rad. A tilt RANDOMISATION for training is deliberately out of
  scope of this entry: it requires per-env fixture quaternions in the
  measurement transform (the static R cannot carry per-episode angles), the
  same plumbing the still-open yaw half needs.
- Rationale: The kinematic-fixture root pose is exactly the mechanism D-034
  introduced for pose variation, so no new asset and no new asset contract is
  needed; one flag covers every angle, where option (a) costs one authored and
  commissioned USD per angle. Rotating about the asset origin keeps the pocket
  centre fixed, so tilt composes cleanly with the translation probe and the
  entrance buffer needs no correction. The y axis is chosen over x because the
  robot base sits on the plate at x = 0, y = +0.190: a y-axis rotation leaves
  plate height unchanged along the x = 0 line (the base neighbourhood moves
  only by its +-0.095 m x-extent times sin(tilt), and the fixture-vs-base
  contact pair is kinematic-vs-fixed, which produces no forces on the arm),
  whereas an x-axis rotation of +5 deg would raise the plate under the base by
  ~36 mm into the base body. Measuring in the pocket frame is the only option
  whose error at tilt 0 is exactly zero by construction rather than "small".
- Sources: D-033 (asset origin convention), D-034 (kinematic fixture,
  reset-write mechanism, entrance buffer); geometry constants in
  proxytask_tasks_cfg.py (PLATE_HALF_EXTENTS, ROBOT_BASE_OFFSET_Y,
  OPENING_TO_BASE_DISTANCE), read 2026-08-07; gate error bound
  depth * sin(tilt) with POCKET_DEPTH = 0.035 m against SIDE_CLEARANCE =
  0.001 m. UNVERIFIED: authored on the dev PC; verified when (1) a noise-0,
  tilt-0 replay of 2026-08-06_21-19-45_s30b32rxy50 reproduces ~99.65 % (the
  transform-off regression), and (2) a --fixture-tilt run starts, the startup
  report prints the tilt line, and the replay writes plausible
  demo_metrics.json with fixture_tilt_rad set, on the training machine.
- Addendum (2026-08-16, VERIFIED on the training machine): both conditions are
  met and D-036 is no longer UNVERIFIED. Noise-0, tilt-0 replay of
  2026-08-06_21-19-45_s30b32rxy50 / model_3950: **100.0 % over 804 episodes**,
  29.3 mm mean maximum depth, and the report's read-back line prints
  `measured tilt about y: +0.000 deg vs commanded +0.000 deg, |err| 0.0000 deg
  (OK)`. The tilt-5 run prints the same line at +5.000 deg with |err| 0.0000
  and scores **50.0 % over 500 episodes** at 17.9 mm mean maximum depth. The
  tilt therefore lands in the prim, the pocket-frame measurement is sound, and
  the drop from 100 % to 50 % is a property of the policy, not of the
  measurement. This is the baseline D-037 is measured against. Runs executed in
  a fresh clone of this branch on the training machine at
  temp\proxytask_gen\generalisation-angle-probe; the USD assets are gitignored
  and were copied from the previous folder.

---

## D-037: Training on a randomly oriented pocket — per-env tilt and the pocket quaternion in the observation

- Date: 2026-08-16
- Status: proposed (pending verification)
- Context: D-036 measured the orientation half of the generalisation question
  and answered it: the D-034 policy scores 100 % at tilt 0 and **50 % at 5 deg**
  (804 / 500 episodes, tilt read back from the prim at |err| 0.0000 deg). The
  task is physically passable at 5 deg — half the episodes insert — so the
  failure is the policy, not the geometry, and the remedy is training rather
  than a mechanism change. Two things block a training run. (1) The tilt is one
  static rotation matrix built in __init__ and one spawn orientation, both
  carrying a single angle for all envs; a per-episode angle needs a per-env
  quaternion, which is the same plumbing the still-open yaw half needs.
  (2) **The policy cannot observe the pocket orientation at all.** Observation
  channels 12:15 are pocket-frame, channels 15:19 are the EE quaternion in the
  WORLD frame, and nothing else carries the fixture pose. Under a per-episode
  tilt two episodes with different angles are therefore indistinguishable in
  observation space while requiring different joint motions — a partially
  observable task in which PPO can only converge to the average.
- Options considered: For the observation — (a) add the pocket orientation as a
  quaternion (4 channels); (b) add it as the 6D representation of Zhou et al.,
  i.e. two columns of the rotation matrix (6 channels); (c) add only the pocket
  z axis (3 channels); (d) re-express the EE quaternion in the pocket frame,
  keeping 21 channels, and let the network infer the tilt from joint angles
  plus the pocket-frame quaternion. For the warm start — (e) surgery on
  model_3950 (new input columns zeroed) or (f) training from scratch. For the
  curriculum — (g) a ladder of tilt magnitudes or (h) the full range at once.
- Decision: (a) + (e) + (g). Four channels are appended at indices 21:25
  carrying the pocket quaternion in the env frame with the sign canonicalised
  to w >= 0; observation_space goes 21 -> 25 and no existing channel index
  moves. The static _tilt_rot becomes a per-env (N, 3, 3) built from a per-env
  fixture quaternion that _reset_idx samples and writes to the kinematic
  fixture. Per episode the tilt magnitude is drawn uniformly from [0, theta_max]
  and its direction uniformly over 360 deg in the env xy plane, i.e. the pocket
  tips in an arbitrary direction rather than about y only. Ladder:
  theta_max = 5 deg warm-started from model_3950, then theta_max = 15 deg
  resumed from it. Success is additionally counted per tilt bin (0-3, 3-6, 6-9,
  9-12, 12-15 deg) in demo_metrics.json. Training stops automatically when the
  trailing-window success rate over ALL angles reaches 99 %, evaluated between
  50-iteration blocks, with max_iterations kept as the backstop. Reward terms,
  PPO hyperparameters, reset noise and the xy pose randomisation (held at 0 for
  this round) are unchanged. play.py's --fixture-tilt DEG keeps its present
  meaning — one deterministic angle about y for all envs — so the post-training
  curve is measured with the same instrument that produced the 100 % / 50 %
  baseline. The yaw noise of D-034 stays rejected in __init__ until it is
  measured in its own step, even though the per-env quaternion could now carry
  it.
- Rationale: The quaternion is what the closest published precedent uses.
  GenPiH trains a peg-in-hole policy with PPO in Isaac Lab **on a UR10e**, with
  the hole pose randomised over RPY [-25, 25] deg — beyond the 15 deg cap here
  — and gives the policy the hole position plus its orientation as a
  quaternion, reaching near 100 % over more than 8000 hole poses with hardware
  validation. Option (b) was the initial proposal on the strength of Zhou et
  al., who show quaternions are discontinuous and 6D representations learn
  better; that result concerns rotations produced as network OUTPUTS, where the
  double cover forces the map to be discontinuous. As an INPUT the objection
  reduces to q ~ -q, which canonicalising the sign removes, and within +-15 deg
  the quaternion never approaches the antipode (w >= 0.991). Four channels also
  subsume the yaw half at no extra cost, which was the only advantage (b) had
  over (c) — a second observation change and a second checkpoint surgery are
  avoided. Option (d) keeps 21 channels and a free warm start, but requires the
  network to recover the fixture orientation by implicitly comparing forward
  kinematics against a pocket-frame quaternion; the information is present but
  the inference is not one an MLP should be asked to make when four explicit
  channels cost nothing. NVIDIA's own Factory peg-insert env is not a
  counter-example: it never rotates the fixed asset and gives the fixed-asset
  quaternion to the CRITIC only, so its actor is blind by construction in a
  setting where blindness is free. The warm start (e) is worth its ~20 lines
  because it is self-checking: the new columns are zeroed, so the expanded
  policy must reproduce the measured 100 % at tilt 0 and 50 % at 5 deg exactly,
  and any error in the surgery — including in the observation normaliser, whose
  running mean/var buffers must be extended with (0, 1) since normalisation is
  on for both actor and critic — shows up as a changed number against a
  baseline that is already on disk. The ladder (g) follows D-034, where the
  rungs transferred near-losslessly (2052 episodes to 90 % at the target rung
  against 77 789 from scratch). Arbitrary tilt direction rather than the
  measured y axis is the user's decision, taken with the trade-off stated: it
  is closer to the real task and to GenPiH, and step 7 keeps the comparison
  honest by evaluating about y. The mean-based stop criterion is likewise the
  user's decision over a worst-bin criterion; the per-bin counts are written
  regardless, so a premature stop is visible in the metrics file rather than
  hidden. The reachability of a 15 deg pocket was NOT pre-measured (declined):
  the 12-15 deg bin makes it observable within minutes of the second rung, and
  the spawn is safe by geometry — the plate half-extent of 0.250 m lifts the
  far edge by 0.250 * sin(15 deg) = 65 mm against a 150 mm home standoff.
- Sources: D-034 (kinematic fixture, reset write, curriculum transfer), D-036
  (pocket-frame measurement, tilt read-back, the 100 % / 50 % baseline), D-005
  (observation restricted to joint states, EE pose relative to the KNOWN hole
  pose, and wrist F/T — the pocket orientation is part of the known hole pose,
  so the new channels are not privileged information). Zhou et al., "On the
  Continuity of Rotation Representations in Neural Networks", CVPR 2019,
  arXiv:1812.07035. "A General Peg-in-Hole Assembly Policy Based on Domain
  Randomized Reinforcement Learning" (GenPiH), arXiv:2504.04148, read
  2026-08-16: UR10e, Isaac Lab, PPO, hole pose randomised RPY [-25, 25] deg,
  observation = hole position + hole orientation as a quaternion + last action.
  Isaac Lab Factory factory_env.py / factory_env_cfg.py, read 2026-08-16:
  obs_order = [fingertip_pos_rel_fixed, fingertip_quat, ee_linvel, ee_angvel],
  fixed_quat in state_order only, no fixed-asset orientation randomisation.
  IsaacGymEnvs shadow_hand.py: goal quaternion and the relative quaternion
  quat_mul(object_rot, quat_conjugate(goal_rot)) both in the observation.
  Geometry constants from proxytask_tasks_cfg.py (PLATE_HALF_EXTENTS 0.250 x
  0.300, HOME_STANDOFF_Z 0.150), read 2026-08-16. UNVERIFIED: authored on the
  dev PC; verified when (1) --fixture-tilt 5 on the rewritten per-env transform
  still scores ~50 %, (2) the expanded checkpoint reproduces 100 % at tilt 0 and
  ~50 % at 5 deg, and (3) both ladder rungs reach their stop criterion with the
  per-bin counts written, on the training machine.

---

## D-038: Pocket yaw generalisation — probe first, +-45 deg is the full range

- Date: 2026-08-16
- Status: proposed (pending verification)
- Context: After D-037 the policy handles pocket tilt (0-10 deg, with +-2 cm
  offset). The last generalisation axis before the real task is rotation of the
  pocket about the env z axis (yaw). A guard from the D-036/D-037 era rejected
  any yaw because the then-static rotation matrix could not carry per-episode
  angles and no yawed pocket had ever been measured. Since D-037 the rotation
  matrices and the observation (channels 21:25) are built per env from the full
  fixture quaternion, so the machinery can carry yaw; only the guard, the
  static-probe path, and the metrics kept it out.
- Options considered: (a) train directly with yaw noise; (b) measure the
  existing policy at fixed yaw angles first (zero-shot probe), then train only
  if needed; (c) keep yaw out of scope.
- Decision: (b) probe first, with +-45 deg as the target range. Code changes:
  yaw activates the pocket-frame path in _tilt_active; the deterministic/random
  guard generalises to "one probe angle XOR one noise range"; static spawn
  orientation composes R_z(yaw) @ R_y(tilt); play.py gains --fixture-yaw
  (capped at 45 deg); the per-bin tilt statistic now measures the pocket-z
  polar angle via acos(r22) instead of the quaternion magnitude, so yaw cannot
  masquerade as tilt; train.py tags yaw noise as "yaw+-Xdeg" (the range is
  symmetric, unlike the 0-to-max tilt magnitude).
- Rationale: The square pocket has C4 symmetry: poses repeat every 90 deg, so
  +-45 deg covers every physically distinct yaw and larger probe angles would
  silently duplicate smaller ones (play.py rejects them). Probe-before-train
  because the policy has plausibly already learned this task: wrist_3 reset
  noise is +-45 deg since the beginning, phi is measured relative to the pocket
  frame since D-036, and a yawed pocket only relocates phi = 0 -- the zero-shot
  probe measures whether that argument holds instead of assuming it.
- Sources: C4 symmetry encoding and pocket-frame measurement (D-030, D-036,
  D-037); reset noise ranges (proxytask_env_cfg.py). UNVERIFIED: authored on
  the dev PC (py_compile plus a numeric self-test of the quaternion
  composition and both angle read-backs, max err < 1e-14). Verified when, on
  the training machine: (1) --fixture-yaw 0 reproduces the K1 baseline, (2)
  the startup report prints "measured yaw about z ... (OK)" at a nonzero
  probe angle, (3) the zero-shot rates at 10 / 22.5 / 45 deg are on disk.
- Addendum (2026-08-16, verified on training machine): the probe series was cut
  short by choice -- after one ambiguous 93 % replay the user opted to train
  directly, so the zero-shot yaw rates and the deterministic --fixture-yaw
  read-back remain unmeasured (the flag stays available). The combined rung,
  warm-started from the offset20mm/tilt0-10deg policy with
  fixture_yaw_noise_rad = 0.7854 on top (4096 envs), stopped at the 99 %
  criterion; demo_metrics.json (code_marker square-peg-2026-08-16-d038):
  recent success 0.9919 over a 5461-episode window, mean max depth 28.5 mm,
  per-tilt-bin rates 0.996 / 0.992 / 0.989 / 0.989 (0-12 deg), no edge
  collapse. An interactive replay with full training noise scored 0.9957 over
  703 episodes. No collapse at the +-45 deg yaw jump, consistent with the
  rationale (wrist reset noise +-45 deg since the start). Target-lag split for
  the parked anti-wind-up decision: success p95 0.50 rad vs failure mean
  1.87 rad (44 failures) -- a clamp band between them is now derivable.
