# Literature check — target-pose uncertainty and in-hand error (2026-08-14)

Purpose: verify, against actually-retrieved sources, whether the current
observation design ("EE pose relative to a KNOWN hole pose") remains defensible
once (a) the fixture pose is randomized per episode and (b) the part can shift
slightly in the gripper (in-hand/grasp pose error), and how published insertion
works handle both. Produced by a web-research pass on 2026-08-14; every claim
below cites only pages that were actually fetched — nothing is from model
memory. Where a detail sits in an appendix or paywalled version that was not
retrieved, it is marked "NOT RETRIEVED" with a pointer. Bibliographic data
(exact venues, page numbers) still needs a final check against the arXiv pages
before thesis citation (same rule as the 2026-08-06 literature check).

Questions asked of each work:

1. Is the (noisy/nominal) target pose fed into the observation, and is noise
   added to it during training?
2. Is the in-hand/grasp error observed, estimated, or only randomized?
3. What are the actual reward terms, and are they computed on ground-truth sim
   state (privileged) or on the noisy observed pose?

## Solution 1 — FORGE (closest template)

**Noseworthy, Tang, Wen, Handa, Kessens, Roy, Fox, Ramos, Narang, Akinola.
"FORGE: Force-Guided Exploration for Robust Contact-Rich Manipulation under
Uncertainty." arXiv:2408.04587 (v2, 2025-01-02).** The arXiv abstract page
lists no venue; the 2026-08-06 check recorded RSS 2024 / IEEE RA-L —
NOT RETRIEVED, confirm venue before citing.

- **Task/clearance:** 8 mm round peg into socket with 0.5 mm diametral
  clearance; gear meshing (0.5 mm); M16 nut threading; snap-fit; planetary
  gearbox (verified from arXiv HTML full text).
- **Target pose in observation:** yes — the policy observes a "noisy estimate
  of the fixed part's pose" plus noisy EE pose/velocity and an estimated
  contact force. **Noise on the target pose is the core mechanism:** Gaussian,
  σ = 2.5 mm on the fixed part's position (perturbation direction uniform on
  the unit sphere). EE position noise 0.25 mm, force noise 1 N. Training was
  found **unstable above σ = 2.5 mm** pose noise; real-world tests up to 5 mm
  error degrade gracefully.
- **In-hand error:** explicitly NOT observed — "We do not include pose or
  velocity of the held part because it can move in the gripper and be
  difficult to track." Instead the held part's in-gripper pose is randomized
  at initialization (x, y ±3 mm; z 10–20 mm task-dependent, Table II) and the
  policy is forced to be robust to it. Gearbox experiments used a
  predetermined grasp location "with small noise from placement error".
- **Reward (all terms on ground-truth sim state, i.e. privileged):**
  1. Keypoint distance through a logistic kernel
     K_{a,b}(d) = (e^{-ad} + b + e^{ad})^{-1}; nut threading uses a
     coarse-to-fine sum K^coarse + K^fine (per-task a, b in Table II, e.g.
     peg: (50, 2) coarse, (100, 0) fine).
  2. Discrete bonuses R_bonus = 1_place + 1_success (on ground-truth poses).
  3. Excessive-force penalty R = −β·max(0, ‖F_ee‖ − F_th), β = 0.2 (peg),
     0.05 (gear/nut) — computed on **ground-truth** force while the policy
     observes the **noisy** force.
  4. Success-prediction penalty R^ET = −|a^ET − y_t| against the ground-truth
     success label.
- **Verdict for our question:** strongest precedent. Noisy-but-observed target
  pose + unobserved-but-randomized in-hand error + fully privileged
  ground-truth reward is exactly the asymmetric pattern we would need; it also
  bounds the usable noise level (σ ≤ 2.5 mm) and motivates F/T observations
  under pose uncertainty.

## Solution 2 — AutoMate

**Tang, Akinola, Xu, Wen, Handa, Van Wyk, Fox, Sukhatme, Ramos, Narang.
"AutoMate: Specialist and Generalist Assembly Policies over Diverse
Geometries." RSS 2024, arXiv:2407.08028 (v2).**

- **Task/clearance:** 100 plug-socket assemblies; 1 mm diametral clearance in
  sim, 0.5–1.0 mm real (3D-print overextrusion) (verified from arXiv HTML).
- **Target pose in observation:** yes — actor observes joint angles, current
  EE pose, and the EE **goal pose**; critic additionally observes velocities
  and the **current plug pose** (privileged, asymmetric actor-critic). Noise
  IS added: "We apply noise to all socket-pose observations", ±2 mm per axis
  (Table V per the fetched text; exact table NOT RETRIEVED in full — check
  Appendix Table V of arXiv:2407.08028). Real-world deployment adds ±10 mm xy
  and 15±5 mm z socket perturbation with ±2 mm observation noise per axis.
- **In-hand error:** randomized, not observed — "the plug is randomly
  initialized in the robot gripper" (ranges in Table II, NOT RETRIEVED —
  check appendix); real experiments note human lead-through grasping adds
  1–3 mm / 5–10° perturbation that the policies tolerate.
- **Reward:** R_t = ω^B·R^B_t + ω^I·R^I_t.
  - Baseline R^B: distance-to-goal penalty, simulation-error (interpenetration)
    penalty, per-timestep task-difficulty reward, success bonus — computed on
    the **ground-truth plug pose** (privileged; the plug pose is a
    critic-only observation).
  - Imitation R^I_t = max_i (1 − tanh(dissimilarity between the executed EE
    path and demonstration path i)), via dynamic time warping or signature
    transforms.
  - Weights ω^B, ω^I only matched "to the same order of magnitude"; no
    numeric values given in the retrieved text.
  - Sampling-based curriculum on the minimum initial plug height.
- **Verdict:** confirms the pattern at scale: goal pose in the observation
  with ±2 mm noise, in-hand randomization without observation, ground-truth
  reward. No F/T at all — proprioception + noisy goal pose suffices at ~1 mm
  clearance, i.e. our clearance regime.

## Solution 3 — IndustReal

**Tang, Lin, Akinola, Handa, Sukhatme, Ramos, Fox, Narang. "IndustReal:
Transferring Contact-Rich Assembly Tasks from Simulation to Reality."
RSS 2023, arXiv:2305.17110.**

- **Task/clearance:** round/rectangular pegs (8–16 mm, clearances 0.5–0.6 mm),
  gears (0.5 mm), NEMA connectors (verified from arXiv HTML).
- **Target pose in observation:** yes — observations consist "exclusively of
  joint angles, gripper/object poses, and/or target poses" (socket pose
  observed). Noise added during training: "for the Insert policies, we
  introduced observation noise" (ranges in Table VII — NOT RETRIEVED, check
  appendix of arXiv:2305.17110). At deployment the target observation carries
  U[−2, 2] mm x/y noise; the real socket pose comes from a coarse RGB
  perception pipeline plus manually recorded targets.
- **In-hand error:** not handled algorithmically in the retrieved text; the
  pegs start in the gripper, and the paper reports real engagement failures
  were "almost exclusively due to slip between the gripper and object" — i.e.
  in-hand error is an acknowledged failure mode, not a modeled one.
- **Reward (general form G = Π w_h · (Σ_t Σ_i w_{d_i} R_{d_i}(t) + Σ_j
  w_{s_j} R_{s_j})):**
  1. SDF-based dense reward −log(Σ_i φ(x_i)/N) over points sampled on the
     plug, φ = signed distance to the socket mesh (ground-truth geometry).
  2. SAPU (simulation-aware policy update): returns discarded if max
     interpenetration > 1 mm, else weighted by 1 − tanh(d_ip/ε_d).
  3. Engagement and success terminal bonuses (magnitudes in Tables IV/VI —
     NOT RETRIEVED).
  4. Sampling-based curriculum on initial height.
  All rewards computed on ground-truth sim state; asymmetric critic (velocity
  critic-only).
- **Verdict:** direct precedent for "noisy observed target pose + privileged
  reward"; also a warning that unmodeled in-hand slip becomes the dominant
  real-world failure mode when it is neither observed nor randomized.

## Solution 4 — Factory

**Narang, Storey, Akinola, Macklin, Reist, Wawrzyniak, Guo, Moravanszky,
State, Lu, Handa, Fox. "Factory: Fast Contact for Robotic Assembly."
RSS 2022, arXiv:2205.03532.**

- **Task/clearance:** nut-and-bolt (ISO 965, M4–M20), round/rectangular
  peg-in-hole (0.104 mm for the 4 mm peg class), connectors, gears (verified
  from ar5iv full text).
- **Target pose in observation:** object poses (hand, nut, bolt) are observed
  directly from sim ground truth; **no observation noise** is added. Target
  states are encoded in the reward, not as a separate goal-pose input.
- **In-hand error:** randomized, not observed — nut-in-gripper yaw randomized
  ±180° (Place) / ±15° (Screw), fingertip offsets ±3 mm; each subpolicy's
  initial-state distribution spans the final states of the preceding one to
  absorb compounding error. Real-world in-hand pose estimation is deferred to
  citation of visuotactile sensing work.
- **Reward:** keypoint-based distances on ground-truth state throughout —
  Pick: ‖k_nut − k_ee‖ over 2–4 collinear keypoints + grasp-lift bonus;
  Place: keypoint distance between nut and bolt axes (aligned = nut base 1 mm
  above bolt top, success < 0.8 mm) + action-gradient penalty β‖a_t − a_{t−1}‖,
  β = 0.1; Screw: sum of two keypoint distances (nut-to-bolt-base,
  EE-to-nut) + early termination.
- **Verdict:** the sim-only baseline of the lineage: clean observations, all
  robustness from initial-state (incl. in-hand) randomization. Shows that
  in-hand randomization without observing it predates the sim-to-real papers.

## Solution 5 — Making Sense of Vision and Touch

**Lee, Zhu, Zachares, Tan, Srinivasan, Savarese, Fei-Fei, Garg, Bohg.
"Making Sense of Vision and Touch." ICRA 2019 (best paper); journal version
arXiv:1907.13098 (IJRR — venue string NOT RETRIEVED, confirm before citing).**

- **Task/clearance:** square, triangular, semicircular, hexagonal 3D-printed
  pegs, "nominal clearance of around 2 mm"; peg rigidly mounted to the EE (no
  gripper, hence no in-hand error by construction) (verified from ar5iv full
  text of 1907.13098).
- **Target pose in observation:** the policy does NOT receive an explicit
  hole pose — input is a learned multimodal representation of RGB (128×128),
  depth, the last 32 F/T readings, and proprioception (EE pose/velocity from
  FK). Target localization is implicit in vision. Box position is randomized
  at episode start during representation training; whether the RL episodes
  randomize the hole pose was NOT RETRIEVED — check Sec. "Experiments" of
  arXiv:1907.13098.
- **Reward (staged, on ground-truth sim state; s = peg-to-hole relative
  position):** reaching c_r(1 − tanh λ‖s‖)(1 − s_ψ); alignment
  1 + c_a(1 − ‖s‖₂/‖ε₁‖₂)(1 − s_ψ/ε_ψ) inside threshold ε₁; insertion
  2 + c_i(h_d − ‖s_z‖) once descending; completion bonus 5 when
  h_d − |s_z| ≤ ε₂. Constants c_r, c_a, c_i NOT RETRIEVED.
- **Verdict:** the alternative design point — drop the target pose from the
  observation entirely and let vision+touch find it. Privileged staged reward
  on true relative pose regardless. Not our template (we have no camera), but
  the right citation for "if pose uncertainty grows beyond noise-robustness,
  the literature switches modality rather than widening the noise".

## Solution 6 — Inoue et al. 2017

**Inoue, De Magistris, Munawar, Yokoya, Tachibana. "Deep Reinforcement
Learning for High Precision Assembly Tasks." IROS 2017, arXiv:1708.04033.**

- **Task/clearance:** cylindrical peg-in-hole at 10 and 20 µm clearance —
  below the robot's own ±60 µm accuracy (verified from ar5iv full text).
- **Target pose in observation:** the hole pose is treated as **unknown up to
  a bounded error** (up to 3 mm initial offset, plate tilted up to 1.6°). The
  search-phase state is s = [Fx, Fy, Fz, Mx, My, P̃x, P̃y], where P̃ are
  encoder positions deliberately **rounded to a grid** so the policy cannot
  rely on precise position — robustness against unknown offset is built into
  the observation encoding. Insertion phase: s = [0, 0, Fz, Mx, My, 0, 0].
- **In-hand error:** not handled — "we suppose that the peg is already
  grasped and in contact with the hole plate."
- **Reward:** search success r = 1.0 − k/k_max (k = steps used, k_max = 100);
  search failure penalty r = −(d − d₀)/(D − d₀) for distance d beyond safe
  bound d₀; insertion penalty r = −(Z − z)/Z against goal depth Z. Computed
  on the real robot's measured state (no sim; d and z from encoders).
- **Verdict:** the F/T-driven extreme: when target error (3 mm) is orders of
  magnitude above clearance (10 µm), position input is nearly useless and
  force becomes the observation. Frames our regime (mm clearance, mm
  uncertainty) as the benign case where a noisy pose observation still works.

## Solution 7 — Schoettler et al. 2020

**Schoettler, Nair, Luo, Bahl, Aparicio Ojea, Solowjow, Levine. "Deep
Reinforcement Learning for Industrial Insertion Tasks with Visual Inputs and
Natural Rewards." arXiv:1906.05841 (venue: IROS 2020 per common listing —
NOT RETRIEVED, confirm before citing).**

- **Task/clearance:** real USB, D-sub, and waterproof Model-E connector
  insertion; "errors as small as ±1 mm can lead to consistent failure"
  (verified from ar5iv full text).
- **Target pose in observation:** the goal position x* is known to the
  controller (P-controller starts 5 cm above it); robustness is tested by
  adding **±1 mm perturbations of the goal location** at evaluation, not by
  training-time noise (training-time goal noise NOT RETRIEVED — check Sec. VI
  of arXiv:1906.05841). Observations per variant: 32×32 grayscale image
  only; or EE Cartesian position + vertical force f_z.
- **In-hand error:** avoided by hardware — "a gripper designed to ensure a
  failure free, fully automated training process" (connector fixtured in the
  gripper); F/T bias recalibrated before each rollout.
- **Reward variants:**
  1. Sparse: r = 1 on detected insertion signal, else 0.
  2. Goal-image: negative ℓ₁ image distance to a goal image.
  3. Dense baseline: r_t = −α‖x_t − x*‖₁ − β/(‖x_t − x*‖₂ + ε) − φ·f_z with
     α = 100, β = 0.002, φ = 0.1 (sign flips after insertion).
  4. Residual RL: u_t = π_H(s_t) + π_θ(s_t) with π_H = −k_p(x_t − x*),
     k_p = [1, 1, 0.3].
  All computed against the known goal x* on the real robot (no privileged sim
  state exists; the "true" pose is the calibrated one).
- **Verdict:** real-robot evidence that a nominal known goal + small (±1 mm)
  unmodeled goal error is survivable when contact feedback (or vision) is in
  the loop; also the cautionary case that goal-relative dense rewards break
  exactly when the goal is wrong — their sparse/vision rewards were motivated
  by that.

## Synthesis

**(i) Is "known hole pose in observation + noise on it" the standard
pattern?** Yes — for the Isaac-Gym/Lab assembly lineage it is *the* pattern.
FORGE, AutoMate, and IndustReal all keep the (nominal) target pose in the
actor observation and corrupt it with training-time noise (FORGE: Gaussian
σ = 2.5 mm; AutoMate: ±2 mm/axis; IndustReal: Table VII ranges, ±2 mm x/y at
deployment), while every reward term is computed on ground-truth sim state.
Privileged reward + non-privileged (noisy) observation is standard asymmetric
practice, made explicit by AutoMate's critic-only plug pose and FORGE's
noisy-force-obs vs. ground-truth-force-penalty split. Our current design
(exact hole pose in the observation, reward on true state) is therefore one
noise term away from the published standard — the defensible upgrade is to
keep the observation but corrupt it, not to remove it. FORGE's σ ≤ 2.5 mm
stability bound and Inoue's regime (error ≫ clearance ⇒ position obs useless)
bracket where the pattern stops working: noise up to roughly 2–5× the
clearance is published territory; beyond that the literature switches to
force-led search (Inoue) or vision (Lee, Schoettler goal-image).

**(ii) Closest template for UR5e + suction + in-hand variation:** FORGE. It
is the only checked work that simultaneously (a) noises the observed fixture
pose, (b) explicitly refuses to observe the held part because "it can move in
the gripper", randomizing its in-gripper pose (±3 mm) instead, and (c) keeps
a fully privileged reward incl. a force penalty on ground-truth force. That
is precisely the suction-cup situation: the part's pose relative to the
flange is nominal-plus-error and untracked. AutoMate is the secondary
template for the no-F/T variant (its policies tolerate 1–3 mm / 5–10°
lead-through grasp perturbation with proprioception + noisy goal pose only).
IndustReal is the negative example: unrandomized in-hand slip became its
dominant real failure mode.

**(iii) Open questions for our own measurement, not the literature:**

1. The actual in-hand error distribution of the two-suction-cup grip
   (translation and tilt of the part relative to the flange, and whether it
   drifts under contact forces). No checked paper measures suction grasps;
   all magnitudes above are parallel-jaw numbers.
2. The actual fixture-pose error after spawn randomization + any real
   locating scheme — this sets the required observation-noise level, and
   whether we land inside FORGE's stable regime (σ ≤ 2.5 mm at 0.5 mm
   clearance; our 1.0 mm clearance may tolerate proportionally more, but that
   scaling is not published).
3. Whether our policy needs F/T once pose noise is injected. Factory (force
   obs hurt at known pose) vs. FORGE (force obs motivated by pose noise)
   predict opposite outcomes; the 2026-08-06 check already flagged this as a
   genuine ablation.
4. Whether reward-on-true-pose plus observation-on-noisy-pose trains stably
   in *our* env at *our* noise level — FORGE's instability threshold is
   task-specific and must be re-measured.

## Sources (retrieved 2026-08-14)

- FORGE: https://arxiv.org/html/2408.04587v2 and https://arxiv.org/abs/2408.04587
- AutoMate: https://arxiv.org/html/2407.08028v2
- IndustReal: https://arxiv.org/html/2305.17110
- Factory: https://ar5iv.labs.arxiv.org/html/2205.03532
- Making Sense of Vision and Touch (journal): https://ar5iv.labs.arxiv.org/html/1907.13098 (conference abstract: https://arxiv.org/abs/1810.10191)
- Inoue et al. 2017: https://ar5iv.labs.arxiv.org/html/1708.04033
- Schoettler et al.: https://ar5iv.labs.arxiv.org/html/1906.05841
