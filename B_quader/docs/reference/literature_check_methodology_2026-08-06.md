# Literature check — methodological order when tackling a new assembly-RL task (2026-08-06)

Question: when published groups approach a NEW contact-rich assembly task, what
comes first — a written method design ("solution concept"), or something before
it? Produced by a web-research pass on 2026-08-06; only actually-retrieved
documents are cited (ar5iv/HTML full texts except where noted). Bibliographic
data needs the standard final check before thesis citation.

## Canonical documented order

Across the NVIDIA assembly line (Factory → IndustReal → AutoMate → FORGE) and
best-practice guides:

1. **Task/asset definition** — geometries, clearances, controllers, task
   formalization (all four NVIDIA papers).
2. **Simulator/environment validation** — contact physics accurate and stable,
   assets physically simulatable, *before* any learning (Factory Sec. III:
   8-scene contact-physics test suite; AutoMate: depenetration to guarantee
   simulatability; SB3 guide: `check_env` + random-action debugging; Ibarz et
   al.: environment-verification lessons).
3. **Non-learned feasibility evidence** — classical controllers or
   scripted/reversed trajectories demonstrating the sim supports the motion
   (Factory: seven classical controllers incl. impedance control before RL;
   AutoMate: simulated disassembly paths, doubling as demonstrations). Present
   in the simulation-infrastructure papers, absent in pure policy-learning
   papers (FORGE, Luo et al.).
4. **Observation/action/reward specification + randomization ranges** (explicit
   ordered list in FORGE).
5. **RL training** — Factory frames it explicitly as proof-of-concept that the
   *validated* simulation enables learning, noting that model-free RL "reveals
   and exploits any inaccuracies or instabilities in the simulator".
6. **Evaluation/ablation**, then sim-to-real (IndustReal: PLAI deployment).

## Verdict on "solution concept first"

Published practice does **not** start with the method design. A step precedes
it: validating that the simulated task is physically well-posed. However, the
papers that skip an explicit validation step (IndustReal, FORGE) **inherit it**
— they build directly on Factory's already-validated simulation methods, so
validation happened once upstream rather than being absent.

Mapping to this project: the proxytask commissioning chain *is* the upstream
validation (Factory's role); the real-part task inherits it the way IndustReal
inherits Factory. What the new task still owes before training is the
asset-specific feasibility evidence (AutoMate's depenetration/disassembly
role): the real part geometry, UR5e and gripper each need their own
passability/scripted-insertion check. Writing the solution concept between CAD
measurement and commissioning is therefore consistent with published practice,
provided the concept marks measurement-derived values (force thresholds,
reward weights) as explicit placeholders filled by the commissioning
measurements — a two-pass concept: draft → review → measure → finalize.

## Scripted/classical feasibility baselines: standard or not?

Mixed. Present in Factory (classical controllers + physics test suite) and
AutoMate (disassembly paths); absent in IndustReal, FORGE, Luo et al. (only a
kinematics-waypoint baseline). Best-practice guides (SB3, Ibarz et al.)
recommend at least random-action environment checks and knowing
classical-controller performance. A scripted insertion as formal step is a
documented practice of the simulation-validation-oriented papers, not a
universal field standard.

## Per-source summary

| Source | What it documents about order |
|---|---|
| Factory — Narang et al., RSS 2022, arXiv:2205.03532 | Physics-first: contact validation (8 scenes) → classical controllers → RL as proof-of-concept. |
| IndustReal — Tang et al., RSS 2023, arXiv:2305.17110 | Builds on Factory's validated sim; no scripted baseline; decomposition → training (SAPU/SDF/SBC) → deployment. |
| AutoMate — Tang et al., 2024, arXiv:2407.08028 | Feasibility-first: depenetration to 0.5 mm clearance → simulated disassembly paths → RL with imitation reward. |
| FORGE — Noseworthy et al., 2024, arXiv:2408.04587 | Ordered pipeline sim config → assets → obs/action → reward → randomization → PPO; no separate validation step (inherits Factory). |
| Luo et al., 2019, arXiv:1903.01066 | Minimal kinematics baseline + iLQG variants only. |
| Ibarz et al., "How to Train Your Robot with Deep RL", IJRR 2021, arXiv:2102.02915 | Prototype in sim first, verify setup, debug reward, know classical performance, sanity-check data before scaling. |
| Xu et al., peg-in-hole survey, 2019, arXiv:1904.05240 | Model-based and learning as coexisting alternatives; no canonical ordered pipeline. |
| Stable Baselines3 "RL Tips" (docs) | check_env → random-action debugging → simplified problem first → tuning → multi-seed evaluation. |
| Elguea-Aguinaco et al., RCIM 2023 | Metadata only — full text not retrieved; no pipeline claim cited. |

## Part 2 (same day) — feasibility checks and per-task simulator re-validation

Follow-up research pass on two sharper questions.

### Q1: Geometric/physical feasibility checks before policy learning — who else does it?

**Verdict: documented in the AutoMate lineage and the assembly-planning
literature; absent from typical single-task peg-in-hole RL papers.**

- **AutoMate** (arXiv:2407.08028, verified in HTML body): depenetration to a
  guaranteed 0.5 mm radial clearance ("translate each vertex along its closest
  face normal until achieving a radial clearance of 0.5 mm"), then **100
  successful disassembly paths per assembly** with a low-level controller —
  empirical feasibility proof doubling as reversed demonstrations.
- **Assemble Them All** — Tian, Xu, Li, Luo, Sueda, Li, Willis, Matusik,
  SIGGRAPH Asia 2022, arXiv:2211.03977 (abstract verified): assembly
  feasibility established via physics-based **disassembly** simulation;
  "thousands of physically valid industrial assemblies" curated this way.
  Assembly-planning paper, not policy learning; the principle AutoMate builds
  on.
- **MatchMaker** — 2025, arXiv:2503.05887 (HTML verified): dedicated
  clearance-specification stage before training (occupancy grids, removal of
  grids within clearance distance); explicitly criticizes prior datasets as
  "not entirely penetration-free... incompatible with many high accuracy
  simulators."
- **IndustReal** (ar5iv verified): SAPU's interpenetration check (1000 sampled
  points, max penetration depth, reward down-weighting) runs **during
  training** — an anti-exploit mechanism, not a pre-training feasibility gate.
- **Isaac Gym Factory docs** (factory.md, verified): asset *quality guidance*
  (SDF resolution 256–512, triangle density, contact-offset formula) and
  reactive troubleshooting, but no formal pre-import feasibility protocol.
- No standalone peg-in-hole RL paper was found that documents a scripted
  passability check before training as an explicit methodological step.

### Q2: Do works that inherit a validated simulator re-validate it for their own assets?

**Verdict: no — inherited validation is taken on trust; per-asset
re-validation is essentially undocumented.**

- **Factory** (ar5iv verified) validated once: contact-force norms against the
  real-world Daily Interactive Manipulation dataset (MMD 0.01269 vs ~0.1–1.0
  for unrelated tasks), joint torques checked against collaborative-robot
  ranges, plus the explicit philosophy that successfully training model-free
  RL is itself "important qualitative evidence of simulator robustness".
- **IndustReal**: no documented re-validation for its NIST assets; residual
  inaccuracy handled at training time via SAPU and SDF reward.
- **FORGE** (HTML verified): "All policies are trained using the Factory
  simulation methods within IsaacGym" — no mesh-quality, penetration, or force
  sanity checks for its own geometries; compensates via dynamics
  randomization.

### Consequence for this project

The proxytask commissioning chain (per-asset checks against generator
sidecars, gate, observation identity, reward algebra, scripted passability)
does per-task what the published descendants of Factory take on trust. Running
the same chain again for the real part, UR5e and gripper is therefore **more
rigorous than documented standard practice** — citable as a deliberate
methodological addition (motivated by Factory's own RL-exploits-simulator
argument and IndustReal's SAPU), not as reproduction of a standard. The
feasibility probe after CAD measurement has direct precedent in
AutoMate/MatchMaker/Assemble Them All.

## Sources

- Factory: https://arxiv.org/abs/2205.03532 (https://ar5iv.labs.arxiv.org/html/2205.03532)
- IndustReal: https://ar5iv.labs.arxiv.org/html/2305.17110
- AutoMate: https://ar5iv.labs.arxiv.org/html/2407.08028
- FORGE: https://ar5iv.labs.arxiv.org/html/2408.04587
- Luo et al.: https://ar5iv.labs.arxiv.org/html/1903.01066
- Ibarz et al.: https://ar5iv.labs.arxiv.org/html/2102.02915
- Xu et al.: https://arxiv.org/abs/1904.05240
- SB3 RL Tips: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- Elguea-Aguinaco et al. (metadata only): https://www.sciencedirect.com/science/article/pii/S0736584522001995
- Assemble Them All: https://arxiv.org/abs/2211.03977
- MatchMaker: https://arxiv.org/html/2503.05887v1
- Isaac Gym Factory docs: https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/factory.md
