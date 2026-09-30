# Project Overview

Narrative reference for the UR10e cylindrical peg-in-hole RL project. Rules live in
CLAUDE.md; current state and next steps live in HANDOFF.md. This file changes
rarely (roadmap or stack changes only).

## Canonical software stack

| Component | Version | Status |
|---|---|---|
| Isaac Sim | 5.x (exact version to be recorded on training machine) | pinned (D-001) |
| Isaac Lab | 2.3 | pinned (D-001) |
| RL algorithm | PPO | fixed |
| RL library | rsl_rl | pinned (D-013) |
| OS (both machines) | Windows | fixed |

## Task summary

Full spec: [task_specification.md](task_specification.md). Key facts:

- Robot: UR10e (6-DOF). Peg rigidly fixed to the end-effector flange, no gripper
  model; wrist F/T is part of the observation (D-005).
- Hole: circular, Ø 30.0 mm, depth 30 mm, aluminum/steel (friction ≈ 0.75), fixed base.
- Peg: cylinder, 50 mm length, ABS (ρ ≈ 1.04 g/cm³). Stage 1: Ø 28.0 mm
  (1.0 mm radial clearance). Later stages: curriculum decision D-006.
- Table: height 0.75 m, plate 55 mm, legs 0.70 m.
- Termination — success: peg tip ≥ 25 mm deep. Failure: timeout 2000 steps; contact
  force > 50 N (TBD, revalidate against measured force distributions); no progress
  for 500 steps.
- Scope: simulation first; sim-to-real on a real UR10e is an optional stretch goal —
  hence the no-privileged-observations rule.

## Decisions made (rationale in DECISIONS.md)

- D-001 stack 2.3 + 5.x · D-002 two-machine split + UNVERIFIED rule ·
  D-003 incremental 6-DoF task-space pose deltas with compliant control
  (realized via D-014) · D-004 dense staged reward, SDF (IndustReal)
  as upgrade path · D-005 rigid peg + F/T, no gripper · D-006 hybrid curriculum
  (pending verification) · D-007 single DECISIONS.md, docs English / thesis German ·
  D-008 repository structure per workflow-tips benchmark · D-012 Direct workflow,
  Factory PegInsert template · D-013 rsl_rl · D-014 OperationalSpaceController ·
  D-015 git checkpoint model (verified milestones as recoverable baselines).

## Open decisions (resolve as DECISIONS.md entries before coding the affected part)

1. ~~Isaac Lab workflow~~ — resolved: Direct, Factory PegInsert as template (D-012).
2. ~~RL library~~ — resolved: rsl_rl (D-013).
3. ~~Controller implementation for D-003~~ — resolved: OSC (D-014).
4. Curriculum details for D-006 (stage-2/3 clearances, progression thresholds).
5. Exact observation vector layout for the 6-DOF UR10e (do not cite the synthesis'
   "26 dims"; its terms sum to 31 for the UR10e — see plan_alignment.md).

## Roadmap (3 months, near-full-time)

- Phase 0 (dev PC): specification, decision log, repo skeleton. Done.
- Phase 1 (training machine): record exact installed versions; run an unmodified
  Isaac Lab 2.3 example to confirm the installation; resolve open decisions 1–3.
- Phase 2: environment — UR10e + table + hole + peg assets; scripted (non-RL)
  insertion validates physics: contact force < 50 N, penetration < 0.5 mm,
  10/10 successful scripted insertions before any RL training.
- Phase 3: observations, actions, reward; training with zero randomization as
  sanity check (expect > 80 % success quickly).
- Phase 4: curriculum stage 1 training; then full hybrid curriculum.
- Phase 5: domain randomization (friction, mass, damping) and robustness evaluation
  (≥ 100-case test suite: success rate, insertion time, max contact force).
- Phase 6: evaluation experiments for the thesis, writing support.

Each phase ends with an explicit success criterion checked on the training machine
before the next phase starts.

## Starting hyperparameters (from the research synthesis — starting points, not results)

gamma 0.99 · lam 0.95 · lr 2.5e-4 · clip 0.2 · rollout 2048 · minibatch 64 ·
epochs 10 · entropy 0.01 · value coef 0.5 · grad clip 0.5 · MLP [256, 128], ELU ·
obs/return normalization on · num_envs 128–256 (GPU-dependent) · control 60 Hz ·
physics dt 1/120 s · action scale 0.025. Tuning heuristics and failure-mode table:
[reference/research_cylindertask.md](reference/research_cylindertask.md) §8, §9, §12.

## Additional documents

- [plan_alignment.md](plan_alignment.md) — verification that the project follows
  the research-synthesis workflow, incl. documented deviations. Update when a
  deviation is added or resolved.
- Projektleitfaden.docx — illustrative German phase workbook (checklists,
  NVIDIA-doc links, success criteria, evidence log). Derived document; on conflict,
  CLAUDE.md and DECISIONS.md are canonical. Regenerate via
  tools/build_projektleitfaden.js when phases or decisions change.
- [reference/workflow_tips.txt](reference/workflow_tips.txt) — external developer
  guidance the repository structure is benchmarked against (D-008).
