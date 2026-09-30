# Curriculum strategies for randomization difficulty — research note (2026-08-16)

Purpose: compare four strategies for scheduling randomization difficulty when
training a contact-rich insertion policy, against what published work and
released code actually do. Produced by a web-research pass on 2026-08-16; every
claim cites only pages or repository files that were actually fetched — nothing
is from model memory. Claims that could not be confirmed in a primary source
are labeled UNVERIFIED. Bibliographic data still needs a final check against
the publisher pages before thesis citation (same rule as
docs/reference/literature_check_2026-08-06.md).

Strategy labels used throughout:

- **(a) sequential manual rungs** — separate warm-started fine-tuning runs,
  each widening one randomization range; a human advances the ladder
  (this project's current approach).
- **(b) scheduled in-run curriculum** — difficulty increases at fixed
  iterations/episodes inside one run.
- **(c) adaptive in-run curriculum** — difficulty widens automatically when a
  success-rate statistic passes a threshold (IndustReal SBC, OpenAI ADR).
- **(d) full randomization from scratch** — the final ranges from step one,
  no curriculum.

## 1. Short answer

Published contact-rich insertion work uses either (c) or (d); no surveyed
paper uses (b), and none uses (a) as a published method — (a) is this
project's human-in-the-loop rendering of (c), with the same update rule
(advance when success passes a threshold) executed manually between runs.
The two direct comparisons that exist both favor threshold-driven adaptation:
IndustReal's ablation (SBC 88.6 % vs. 66.8 % no-curriculum vs. 32.4 % naive
curriculum) and OpenAI's ADR-vs-fixed-randomization result. They also show
that (d) is viable when the task plus randomization is within reach (Factory
and GenPiH train jointly with no curriculum at all), and that a *naive*
curriculum can be worse than none. For this project: keep the manual ladder
for the few remaining rungs, but run the (d) control (final ranges from
scratch) as the thesis comparison, and adopt a minimal SBC-style in-run
widening only if further widening stalls or the number of rungs grows.

## 2. Per-source findings

### 2.1 Factory (RSS 2022) — strategy (d)

Narang, Storey, Akinola, Macklin, Reist, Wawrzyniak, Guo, Moravanszky, State,
Lu, Handa, Fox: *Factory: Fast Contact for Robotic Assembly.* RSS 2022,
arXiv:2205.03532.

- **Paper (verified, ar5iv full text):** no curriculum or staged difficulty.
  RL training randomizes initial states directly at full range every episode;
  the Screw subpolicy's Table III lists e.g. hand angle ±90°, nut-in-gripper
  yaw ±15°, bolt position ±10 mm; reported screw success 85.6 %. Quote: "At
  the start of each Screw episode, the Franka hand and nut were reset to a
  stable grasp pose, randomized relative to the top of the bolt." Consistent
  with the earlier check in
  docs/reference/literature_check_2026-08-06.md (Claim 1).
- **Isaac Lab port (verified, isaac-sim/IsaacLab, `main` branch as of
  2026-08-16, `source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py`):**
  no curriculum logic; fixed-asset and held-asset pose noise
  (`fixed_asset_init_pos_noise`, `fixed_asset_init_orn_range_deg`,
  `held_asset_pos_noise`, `hand_init_pos_noise`, `hand_init_orn_noise`) is
  applied at full configured range on every reset. `curr_successes` /
  `ep_succeeded` are tracked for logging only. (Checked on `main`, not
  re-checked against the `v2.3.0` tag — UNVERIFIED for the exact tag, but the
  AutoMate check in §2.6 was done on `v2.3.0` directly.)

### 2.2 IndustReal (RSS 2023) — strategy (c), on one variable only

Tang, Lin, Akinola, Handa, Sukhatme, Ramos, Fox, Narang: *IndustReal:
Transferring Contact-Rich Assembly Tasks from Simulation to Reality.*
RSS 2023, arXiv:2305.17110.

- **What SBC schedules (verified, ar5iv full text):** the initial plug height
  relative to the socket. Initial height is sampled from
  `Uniform[z_low, z_high]`; `z_high` stays constant at 10 mm above the hole
  top, while `z_low` starts 10 mm *below* the hole top (partially engaged =
  easy) and is raised as performance improves. Paper text: advance `z_low` by
  Δz_i = 5 mm when mean success exceeds 80 %, retreat by Δz_d = 3 mm when it
  falls below 10 %, else hold. Crucially, the policy always samples the whole
  interval up to the moving bound — difficulty is widened, not shifted.
- **Everything else is NOT curriculated (verified, same source):** socket pose
  randomization (±10 cm position; rotation noise in the released config
  `socket_rot_noise: [0.0, 0.0, 0.0872665]`, i.e. ±5° yaw) and plug pose
  noise (±10 mm) are fixed at full range from the start of training.
- **Code (verified, isaac-sim/IsaacGymEnvs, `main`):**
  - `isaacgymenvs/tasks/industreal/industreal_task_pegs_insert.py` — applies
    the curriculum each update via
    `self.curr_max_disp = algo_utils.get_new_max_disp(curr_success=..., ...)`
    and rescales reward with
    `sbc_rew_scale = algo_utils.get_curriculum_reward_scale(...)`, applied
    asymmetrically (`rew/scale` if negative, `rew*scale` if positive) so that
    reward magnitudes stay comparable across curriculum stages.
  - `isaacgymenvs/tasks/industreal/industreal_algo_utils.py` — the update rule
    verbatim:

    ```python
    def get_new_max_disp(curr_success, cfg_task, curr_max_disp):
        """Update max downward displacement of plug at beginning of episode,
        based on success rate."""
        if curr_success > cfg_task.rl.curriculum_success_thresh:
            new_max_disp = max(
                curr_max_disp + cfg_task.rl.curriculum_height_step[0],
                cfg_task.rl.curriculum_height_bound[0],
            )
        elif curr_success < cfg_task.rl.curriculum_failure_thresh:
            new_max_disp = min(
                curr_max_disp + cfg_task.rl.curriculum_height_step[1],
                cfg_task.rl.curriculum_height_bound[1],
            )
        else:
            new_max_disp = curr_max_disp
        return new_max_disp
    ```

  - `isaacgymenvs/cfg/task/IndustRealTaskPegsInsert.yaml` — released values:
    `initial_max_disp: 0.01`, `curriculum_success_thresh: 0.75`,
    `curriculum_failure_thresh: 0.5`,
    `curriculum_height_step: [-0.005, 0.003]`,
    `curriculum_height_bound: [-0.01, 0.01]`.
  - **Discrepancy (both sides verified):** the paper text states thresholds
    80 % / 10 %; the released config ships 0.75 / 0.5. Cite the paper values
    as the method and note the config if code-level detail is needed.

### 2.3 GenPiH (arXiv 2025) — strategy (d)

Liu, Kramberger, Bodenhagen: *A General Peg-in-Hole Assembly Policy Based on
Domain Randomized Reinforcement Learning.* arXiv:2504.04148, 2025.
(Peer-review status UNVERIFIED — treat as preprint until checked.)

- **Verified (arXiv HTML full text):** hole pose is randomized at full range
  for every environment from the start — x ∈ [−0.2, 0.2] m,
  y ∈ [−0.26, 0.26] m, z ∈ [0, 0.16] m, RPY ∈ [−25°, 25°] — with no
  curriculum or staged training mentioned. PPO in Isaac Lab, 8192 parallel
  environments, "nearly 100 % success" across the randomized poses within
  100 epochs (simulation only). The policy observes the hole pose directly
  (position and orientation in the observation vector), which is the same
  design choice as this project's pocket-pose observation. Which RL library
  implements their PPO is not stated (UNVERIFIED).
- Relevance: the closest published setup to this project (UR10e, Isaac Lab,
  PPO, observable hole pose, ±25° orientation randomization) trained jointly
  with no curriculum and reports near-saturated success in simulation.

### 2.4 OpenAI ADR (2019) — strategy (c), all parameters

OpenAI: Akkaya et al.: *Solving Rubik's Cube with a Robot Hand.*
arXiv:1910.07113, 2019. (Preprint; no peer-reviewed venue — cite as such.)

- **Mechanism (verified, ar5iv full text):** each randomization parameter λ_i
  has a uniform range (φ_i^L, φ_i^H). A fraction of episodes is used for
  boundary evaluation: one dimension is pinned to its current lower or upper
  bound, performance is accumulated in per-boundary buffers, and when average
  performance exceeds a threshold t_H the boundary is pushed outward by a
  step Δ; when it falls below t_L, pulled inward. "ADR automatically generates
  a distribution over randomized environments of ever-increasing difficulty."
- **ADR vs. fixed full randomization (verified):** on block reorientation,
  manual (fixed) domain randomization reached 2.7 ± 1.1 real-robot successes
  vs. 32.0 ± 6.4 for the largest ADR run. Section 8.2: "The larger the fixed
  randomization entropy, the longer it takes to train from scratch. We
  hypothesize that for a sufficiently difficult task and randomization
  entropy, training from scratch becomes infeasible altogether." This is the
  strongest published statement against strategy (d) at high difficulty.

### 2.5 DeXtreme (2022/2023) — vectorized ADR, one paragraph

Handa, Allshire, Makoviychuk, Petrenko, Singh, Liu, Makoviichuk, Van Wyk,
Zhurkevich, Sundaralingam, Narang, Lafleche, Fox, State: *DeXtreme: Transfer
of Agile In-hand Manipulation from Simulation to Reality.* arXiv:2210.13702;
ICRA 2023 per the fetched page (venue UNVERIFIED against proceedings).
Verified from the ar5iv full text: they implement a GPU-vectorized ADR in
which 40 % of the parallel environments are dedicated to boundary evaluation;
a range is widened when mean consecutive successes at the boundary exceed
t_H = 20 and tightened below t_L = 5, with per-parameter step sizes Δ chosen
individually ("This trades off more tuning work for more stable training").
Relevance here: ADR-style adaptation is implementable inside a massively
parallel GPU simulator of the Isaac lineage, at the cost of sacrificing a
fraction of environments to boundary probes.

### 2.6 Isaac Lab 2.3 — curriculum support and rsl_rl warm start

- **Manager-based workflow (verified, v2.3.0 API docs,
  `source/api/lab/isaaclab.managers.html`):** `CurriculumManager` ("Manager to
  implement and execute specific curricula") with `CurriculumTermCfg` exists
  for `ManagerBasedRLEnv`. The v2.3.0 API page documents no curriculum
  facility for `DirectRLEnv`; in the direct workflow a curriculum must be
  hand-rolled in the env.
- **Direct-workflow precedent (verified, isaac-sim/IsaacLab tag `v2.3.0`):**
  the AutoMate port ships exactly such a hand-rolled SBC in a `DirectRLEnv`:
  `source/isaaclab_tasks/isaaclab_tasks/direct/automate/assembly_env.py`
  calls, gated by `cfg_task.if_sbc`,
  `automate_algo.get_new_max_disp(curr_success=torch.count_nonzero(self.ep_succeeded)/self.num_envs, ...)`,
  and `automate_algo_utils.py` contains the same threshold rule as IndustReal
  (`curriculum_success_thresh` / `curriculum_failure_thresh`, tensorized
  per-assembly bounds). So the reference implementation for an in-run
  adaptive curriculum in this project's env class exists inside Isaac Lab
  2.3 itself.
- **rsl_rl resume (verified, leggedrobotics/rsl_rl `main`,
  `rsl_rl/runners/on_policy_runner.py`):** `OnPolicyRunner.load()` restores
  the model weights, the algorithm state (optimizer, normalizer/RND state via
  `self.alg.load(...)`) and `current_learning_iteration` from the checkpoint.
  It restores nothing about the environment: randomization ranges come solely
  from the env config of the new run. This is what makes the manual ladder
  (a) work — each rung is a fresh env config on top of restored weights — and
  it equally means an adaptive curriculum's current range would need to be
  persisted by the env itself (e.g. in the metrics file) to survive a resume.

### 2.7 Narvekar et al. (JMLR 2020) — taxonomy

Narvekar, Peng, Leonetti, Sinapov, Taylor, Stone: *Curriculum Learning for
Reinforcement Learning Domains: A Framework and Survey.* JMLR 21(181), 2020.
Verified from the arXiv full text (2003.04960): curriculum learning is
decomposed into task generation, sequencing, and transfer learning
(Sec. 3.2); sequencing is classified as *static* ("generated in its entirety
before training") vs. *adaptive* ("influenced by properties that can only be
measured during learning, such as the learning progress by the agent")
(Sec. 3.4); task generation should "avoid negative transfer, where using a
task for transfer hurts performance" (Sec. 4.1). In this vocabulary,
strategies (b) are static sequences, (c) are adaptive sequencing, and the
manual ladder (a) is adaptive sequencing with a human executing the update
rule between runs. An explicit quantitative statement that overly large
difficulty jumps hinder learning was NOT found in the fetched text
(UNVERIFIED — do not cite the survey for that specific claim; use the ADR
Sec. 8.2 quote and this project's own 5°→15° collapse instead).

## 3. Comparative evidence

Direct comparisons between the strategies are rare; the two that exist:

1. **IndustReal ablation (verified, ar5iv, 5 seeds × 1000 trials):**

   | Training regime | Success | Engagement | Pos. error |
   |---|---|---|---|
   | No curriculum (final difficulty from scratch, ≈ strategy d) | 66.8 ± 5.8 % | 89.2 ± 5.0 % | 10.7 ± 2.7 mm |
   | "Standard" curriculum (range shifted harder, not widened) | 32.4 ± 1.8 % | 46.0 ± 3.6 % | 18.2 ± 0.4 mm |
   | Sampling-based curriculum (strategy c) | 88.6 ± 2.4 % | 96.6 ± 2.3 % | 3.8 ± 0.8 mm |

   Two readings matter for this project: (i) the adaptive curriculum beats
   training at final difficulty from scratch, and (ii) a curriculum that
   *shifts* the sampled range toward harder states instead of *widening* it
   is substantially worse than no curriculum at all — the policy forgets the
   easy states it no longer sees. The manual ladder here widens ranges
   (each rung's range contains the previous one), so it is on the SBC side
   of this distinction.

2. **OpenAI ADR vs. fixed randomization (verified):** §2.4 — adaptive
   expansion reached roughly an order of magnitude more real-robot successes
   than a fixed manually-tuned randomization, and the paper argues training
   from scratch at large fixed randomization entropy scales poorly.

Against this stand the (d) data points: Factory (85.6 % screw success,
sub-millimeter clearances, no curriculum), FORGE (no curriculum; verified in
docs/reference/literature_check_2026-08-06.md), and GenPiH (±25° hole
orientation trained jointly, near-saturated sim success with the hole pose in
the observation). The consistent pattern: when the policy directly observes
the randomized variable and the task remains solvable across the range,
joint training works; curricula earn their complexity when initial-state
difficulty (engagement depth, tight clearance, high randomization entropy)
makes the reward signal too sparse from scratch. No surveyed insertion work
uses fixed-iteration schedules (b); difficulty updates are always
performance-gated where a curriculum is used at all.

## 4. Recommendation for this project

Current state (session 2026-08-16): rungs verified at 99–99.3 % — base →
offset ±5 cm → tilt ±5° → ±10° → combined offset ±2 cm + tilt 0–10°; the
direct 5°→15° tilt jump collapsed to 0 % success while the 5°→10° step
succeeded; yaw ±45° is next and may be near-solved zero-shot (wrist reset
noise has been ±45° throughout, `reset_yaw_noise = 0.7854` in
`source/proxytask/proxytask/tasks/direct/proxytask/proxytask_env_cfg.py`).

**Recommendation: keep the manual ladder for the remaining rungs; add the
strategy-(d) control run; defer an in-run adaptive curriculum unless
widening stalls.** Reasoning:

- The ladder already implements the update rule the literature supports
  (advance on success ≥ threshold, widen rather than shift), and it has
  produced a verified checkpoint and metrics file at every rung — directly
  usable as thesis evidence. Only one or two rungs remain (yaw, then
  combined), so the amortized benefit of automating the schedule is small.
- The 5°→15° collapse is this project's own measurement of the ADR Sec. 8.2
  claim at rung granularity: even warm-started, too large a difficulty step
  destroys performance. It argues for *keeping steps small*, not necessarily
  for automating them.
- **The missing experiment is the (d) baseline**: one run at the final
  combined ranges from scratch. IndustReal's no-curriculum baseline reached
  66.8 %, GenPiH solved ±25° jointly, and the pocket pose is observable in
  this env — the ladder cannot be claimed as *necessary* without this
  control (same conclusion as literature_check Claim 1). Whatever the
  outcome, it is a result: either the ladder is justified by a measured gap,
  or joint training suffices and the thesis reports that with data.
- If the ladder does stall (a rung that cannot reach threshold with an
  acceptable step size) or more variables must be widened than expected, the
  minimal adaptive mechanism in this `DirectRLEnv` is small and has a direct
  in-family precedent (AutoMate port, §2.6): the env already maintains
  `self._recent_successes` (trailing deque, `proxytask_env.py`) and
  per-episode tilt binning. A minimal SBC-style implementation would (i) hold
  a `self._curr_tilt_bound` initialized at the last verified range, (ii) on
  each metrics-window boundary apply the IndustReal threshold rule (widen
  `fixture_tilt_noise_rad` by a small step when the recent rate ≥ 0.95,
  shrink when < 0.5, hold otherwise), (iii) keep sampling the tilt uniformly
  over the whole `[0, bound]` interval (widen, never shift — the IndustReal
  ablation's decisive point; the current per-episode sampling already does
  this), and (iv) write the current bound into the metrics JSON every window,
  both for auditability and because `OnPolicyRunner.load()` restores no env
  state across resumes (§2.6). Two verification consequences must be
  accepted: the recent-success rate becomes a statistic over a *moving*
  difficulty distribution (the existing per-tilt-bin reporting mitigates
  this), and the `train.py` early stop at mean success 0.99 must additionally
  require that the bound has reached its final value, otherwise the run
  stops early at an easy intermediate stage.

Honest trade-off summary: the manual ladder costs one launch per rung and a
human decision, but every checkpoint is independently verified and the
thesis narrative maps one-to-one onto runs. An adaptive curriculum is one
run and one launch, but its schedule becomes part of the system under test —
it needs its own identity checks (does the bound move only when the
threshold logic says so; is the metrics window long enough to be a stable
gate), and a bug in the gate silently changes what the run trained on.

## 5. For the thesis (citable literature)

Suitable for a German bachelor-thesis chapter on curriculum learning and
domain randomization (verify bibliographic data on the publisher page before
citing; academic-style caveats included):

- **Narvekar et al., JMLR 21(181), 2020** — peer-reviewed journal survey;
  the source for terminology (task sequencing, transfer, static vs. adaptive
  curricula). Safe as the framework citation.
- **Tang et al., IndustReal, RSS 2023 (arXiv:2305.17110)** — peer-reviewed;
  the SBC method and the only insertion-specific curriculum ablation found.
  Core citation for the curriculum discussion.
- **Narang et al., Factory, RSS 2022 (arXiv:2205.03532)** — peer-reviewed;
  counter-example: contact-rich assembly RL without curriculum.
- **Tang et al., AutoMate, RSS 2024 (arXiv:2407.08028)** — peer-reviewed;
  SBC generalized over geometries (verified earlier in
  literature_check_2026-08-06.md); its Isaac Lab 2.3 port is the code
  precedent for a DirectRLEnv curriculum.
- **OpenAI (Akkaya et al.), arXiv:1910.07113, 2019** — preprint only, no
  peer-reviewed venue; widely cited. Usable for the ADR mechanism and the
  adaptive-vs-fixed comparison, but label it as a preprint and do not rest a
  load-bearing claim on it alone.
- **Handa et al., DeXtreme (arXiv:2210.13702)** — cite for vectorized ADR in
  GPU simulation; confirm the ICRA 2023 venue in the proceedings first
  (UNVERIFIED here).
- **Liu et al., GenPiH, arXiv:2504.04148, 2025** — closest setup to this
  project; peer-review status UNVERIFIED — cite as preprint, and only for
  what its text supports (joint full-range training with observable hole
  pose in simulation).
- Isaac Lab 2.3 documentation and the isaac-sim/IsaacLab v2.3.0 sources are
  citable as software references (documentation/version, not literature).
