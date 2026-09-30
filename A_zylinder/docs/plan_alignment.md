# Alignment: Project vs. Research Synthesis

Verification (2026-07-23) that the project structure follows the workflow of the
research synthesis (docs/reference/research_cylindertask.md) in principle, with all
deviations documented as deliberate decisions. Robot substitution (UR10e instead of
Franka) is acknowledged throughout; the synthesis itself lists the UR10e as a valid
alternative.

## Points of exact alignment

| Research synthesis section | Project counterpart | Status |
|---|---|---|
| §1 Architecture: policy → delta pose → compliant task-space controller → contact loop | D-003 (incremental 6-DoF pose deltas, compliant control) | aligned |
| §3 Observations: proprioception + relative pose + F/T, no privileged information | D-005 + "no privileged observations" rule (CLAUDE.md) | aligned |
| §4 Action space: incremental, scale 0.025 | D-003 + hyperparameter block (CLAUDE.md) | aligned |
| §6 Dense staged reward (approach/alignment/insertion/success + force/torque penalties) | D-004, starting point | aligned |
| §7 Sampling-based curriculum progression (IndustReal) | D-006 hybrid | aligned |
| §8 PPO hyperparameters | Adopted 1:1 as starting points (gamma 0.99, lr 2.5e-4, rollout 2048, MLP [256,128], ELU, 60 Hz, dt 1/120) | aligned |
| §9 Implementation sequence | Roadmap phases: env setup → scripted insertion as physics validation (force < 50 N, penetration < 0.5 mm, 10/10 scripted insertions) → verify observations/reward terms individually → training without randomization as sanity check (> 80 %) → curriculum → domain randomization → 100-case evaluation suite | aligned |
| §12/§9 Debugging knowledge and failure modes | Referenced from CLAUDE.md (§8, §9, §12 pointers) | aligned |
| §15 Timeline, 12 weeks | Roadmap "3 months, near-full-time" | aligned |

## Deliberate deviations (documented, no plan breaks)

1. **Reward: staged-dense first, SDF only as upgrade path.** The synthesis names
   the SDF-based reward the single most important implementation decision (§10:
   88.6 % vs 1.8–54.2 %). The project nevertheless starts with the staged reward
   from §6 — rationale in D-004 (term-by-term debuggability, lower implementation
   risk at project start). If convergence stalls, SDF is the defined next step,
   exactly as the synthesis prescribes as the fix for reward exploitation.
2. **Curriculum axis.** The synthesis varies clearance directly (1.0 → 0.5 mm)
   plus friction over 4 levels; the project varies peg diameter with a fixed hole —
   mathematically the same clearance progression, differently parameterized.
   Deviations: friction is fixed by the material (≈ 0.75) instead of being a
   curriculum variable, and 3 stages instead of 4. Both marked
   "pending verification" in D-006.
3. **Workflow decision (cloning Factory) still open.** The synthesis says to copy
   `factory_env_cfg.py` directly. The project keeps this as open decision 1 because
   the synthesis was written against an unclear Isaac Lab version and Factory must
   first be inspected as it exists in 2.3. This is not a deviation from the plan
   but the revalidation the synthesis itself demands.
4. **Robot/geometry.** UR10e instead of Franka (explicitly listed as an alternative
   in the synthesis), 30 mm hole instead of 10 mm. Scaled, principle identical.

## Findings about the synthesis itself

- The observation vector in the synthesis is internally inconsistent: it states
  "26 dimensions", but the listed terms sum to 7+7+3+4+3+3+6 = 33 (Franka). For the
  UR10e the same layout yields 6+6+3+4+3+3+6 = 31. Do not cite the "26" when
  resolving open decision 5.
- §9 contains a hard criterion adopted into the roadmap: 10/10 successful
  insertions with a scripted controller before any RL training (Step 2.1).
