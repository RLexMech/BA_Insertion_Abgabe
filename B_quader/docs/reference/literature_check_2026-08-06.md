# Literature check — RL peg-in-hole approach (2026-08-06)

Purpose: verify, against actually-retrieved sources, whether the planned approach
for the real-part insertion task (curriculum, wrist F/T as observation and
reward penalty, simulated-force clamping, non-cylindrical parts, F/T ablation)
matches published practice. Produced by a web-research pass on 2026-08-06; every
claim below cites only pages that were actually fetched or returned in search
results — nothing is from model memory. Bibliographic data still needs a final
check against the arXiv pages before thesis citation (same rule as HANDOFF's
open item on citations).

## Claim 1 — Curriculum learning is standard practice

**Verdict: partially supported — common, but not universal; the flagship
Isaac-Gym-lineage works are split.**

- **IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to
  Reality.** Tang, Lin, Akinola, Handa, Sukhatme, Ramos, Fox, Narang. RSS 2023,
  arXiv:2305.17110. Proposes a **sampling-based curriculum (SBC)** as a core
  contribution for insertion tasks (verified from the arXiv abstract page).
- **AutoMate: Specialist and Generalist Assembly Policies over Diverse
  Geometries.** Tang, Akinola, Xu, Wen, Handa, Van Wyk, Fox, Sukhatme, Ramos,
  Narang. RSS 2024, arXiv:2407.08028. Sampling-based curriculum: the minimum
  initial plug height increases across stages (verified from arXiv HTML full
  text).
- **Counter-examples (tight clearance WITHOUT curriculum):**
  - **Factory: Fast Contact for Robotic Assembly.** Narang et al. (NVIDIA/UW),
    RSS 2022, arXiv:2205.03532. RL policies on ISO-standard clearances
    (**0.104 mm** for 4 mm pegs, ISO 965 nut-bolt) with initial-state
    randomization but **no staged curriculum** (verified from ar5iv full text).
  - **FORGE: Force-Guided Exploration for Robust Contact-Rich Manipulation
    under Uncertainty.** Noseworthy, Tang, Wen, Handa, Kessens, Roy, Fox,
    Ramos, Narang, Akinola. RSS 2024 / IEEE RA-L, arXiv:2408.04587. Peg
    insertion, nut threading, gear meshing with dynamics + initial-state
    randomization, **no curriculum** (verified from arXiv HTML full text).

Implication: a clearance curriculum is defensible and well-precedented
(IndustReal/AutoMate), but it cannot be claimed as *required* — Factory and
FORGE solve sub-mm tasks without one. Whether it is needed in our setup is a
measurement (direct training at target difficulty vs. curriculum), and the
comparison must run at the final randomization level, since curriculum need is
a function of clearance × initial-state spread (this is exactly what SBC
staffs).

## Claim 2 — F/T as observation AND reward penalty is common

**Verdict: partially supported — the combination exists, but the dominant
NVIDIA sim-to-real line uses NEITHER.**

- **Both (obs + penalty): FORGE** (above). Policy observes estimated EE contact
  force (noisy, 1 N Gaussian noise); reward contains an explicit
  excessive-force penalty `R = −β·max(0, ‖F_ee‖ − F_th)` with β = 0.2 (peg) /
  0.05 (gear, nut), computed on ground-truth force. Closest published precedent
  for the planned setup.
- **Observation-only (representation): Making Sense of Vision and Touch.** Lee
  et al. (Stanford), arXiv:1907.13098 (journal version of the ICRA 2019 best
  paper). Wrist F/T fused into a learned multimodal representation used as
  policy input; no force reward penalty reported in search results.
- **Neither: Factory (RSS 2022), IndustReal (RSS 2023), AutoMate (RSS 2024).**
  AutoMate's observation is joint angles + EE pose + goal pose only, reward has
  no force penalty (verified from full text); Factory's best screw policy
  excluded force obs and had no force penalty (verified from ar5iv).

Flag: wrist F/T in observations is *less* standard in the Isaac Gym/Lab
assembly lineage than assumed — that lineage mostly relies on proprioception +
known goal pose. FORGE is the strongest citation for our design. This nuances
docs/reference/research_cylindertask.md §6, which presents force/torque
penalties as standard.

## Claim 3 — Clamping/filtering simulated F/T against contact-force spikes

**Verdict: not found as a documented, named practice in the papers checked.**

- **FORGE** is the closest: Gaussian noise (1 N) on the force observation and a
  separation of noisy-obs vs. ground-truth-penalty forces, but the full text
  (searched for "filter/EMA/clip/spike") documents **no spike filtering or
  clamping** of simulated forces.
- **Factory** smooths *actions* (action-gradient penalty, EMA on Place
  actions), not forces.
- Isaac Gym docs and IsaacLab discussion threads (e.g. isaac-sim/IsaacLab
  discussion #2697, Isaac Gym force-sensor docs) acknowledge unreliable/spiky
  contact-force readouts and community workarounds (body force sensors instead
  of the net contact force tensor), but this is forum-level, not citable.

Thesis consequence: present F/T clamping/filtering as an own engineering
measure motivated by known simulator artifacts (supervisor independently
suggested clamping), not as established literature practice. Low-pass filtering
of *real* F/T sensors at deployment is common but a different claim.

## Claim 4 — Non-cylindrical / keyed insertion with RL

**Verdict: supported.**

- **AutoMate** (RSS 2024): 100 geometrically diverse assemblies, many with
  symmetry-breaking features; specialist policies solve 80 assemblies at ≥80 %
  sim success, zero-shot sim-to-real.
- **Siemens gear-assembly benchmark**, used in **Reinforcement Learning on
  Variable Impedance Controller for High-Precision Robotic Assembly** (Luo et
  al., arXiv:1903.01066): includes a **square-hole small gear**, gear-on-shaft
  insertion and gear-teeth meshing — keyed/rectangular insertion predates
  IndustReal.
- **PolyFit: A Peg-in-hole Assembly Framework for Unseen Polygon Shapes via
  Sim-to-real Adaptation** (arXiv:2312.02531): explicitly targets polygonal
  pegs including unseen shapes.

The square/rectangular part is well within published territory.

## Claim 5 — Ablations with vs. without F/T observations

**Verdict: supported — but the direction of the effect is task-dependent.**

- **Factory** (RSS 2022, verified from ar5iv): force observations ablated for
  the Screw subpolicy **hurt performance** — 50.26 % success with force obs vs.
  77.60 % without.
- **Making Sense of Vision and Touch** (Lee et al.): removing the
  force/haptic modality degrades peg-insertion completion — F/T **helps** when
  pose uncertainty makes contact events informative.
- **FORGE**: motivates force observation specifically under significant pose
  uncertainty; reports training instability above σ = 2.5 mm pose noise.

Pattern: F/T observations pay off when the goal pose is uncertain; with a
precisely known hole pose the Factory result warns they may add noise without
benefit. In our setup the goal pose is known and made artificially uncertain by
spawn randomization — a with/without-F/T ablation is therefore a genuinely open
question and a defensible thesis experiment.

## Summary of deviations from the literature mainstream

1. F/T in observations + soft force penalty is precedented (FORGE) but not the
   mainstream of the Isaac assembly lineage.
2. Simulated-F/T clamping has no citable published precedent — frame as an
   engineering measure.
3. Curriculum is well-precedented but not necessary per Factory/FORGE — phrase
   thesis claims accordingly, and measure its benefit in our own setup.

## Sources

- Factory: https://arxiv.org/abs/2205.03532 (full text: https://ar5iv.labs.arxiv.org/html/2205.03532)
- IndustReal: https://arxiv.org/abs/2305.17110
- AutoMate: https://arxiv.org/html/2407.08028v2
- FORGE: https://arxiv.org/html/2408.04587v2 (project page: https://noseworm.github.io/forge/)
- Making Sense of Vision and Touch: https://arxiv.org/abs/1907.13098
- Variable Impedance Controller / Siemens benchmark: https://arxiv.org/pdf/1903.01066
- PolyFit: https://arxiv.org/pdf/2312.02531
- Isaac Gym force-sensor docs: https://docs.robotsfan.com/isaacgym/programming/forcesensors.html
- IsaacLab discussion #2697: https://github.com/isaac-sim/IsaacLab/discussions/2697
