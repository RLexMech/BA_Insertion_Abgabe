# Research Synthesis — Contact-Rich Cylindrical Peg-in-Hole with RL in Isaac Lab

> Provenance: text extracted from "Research Cylindertask (1) (1).pdf" (AI-assisted
> literature synthesis, 25 pages). Section numbers (§) referenced from CLAUDE.md and
> DECISIONS.md refer to the numbered sections of this document. Written against
> Isaac Lab 2.x-era material and a Franka baseline; all API snippets and
> Franka-specific values (7-DOF observation layout, effort limits) require
> adaptation to the UR10e (6-DOF) and validation against Isaac Lab 2.3 (D-001).
> Claims cite sources as inline markers (arxiv, isaac-sim, ...); verify before
> citing in the thesis.

PAGES: 25
--- PAGE 1 ---
Based on my comprehensive research, I can now provide you with a detailed,
practical implementation guide. Let me synthesize the most reproducible approach
from the literature and Isaac Lab's existing infrastructure.
The Most Reproducible RL Approach for a Contact-Rich Cylindrical Peg-in-Hole
Task in NVIDIA Isaac Lab
Executive Summary
The single most reproducible approach for your task is NVIDIA's Factory PegInsert
environment in Isaac Lab, enhanced with techniques from the IndustReal paper.
This combination provides:isaac-sim.github+2
 Ready-to-use environment with contact-rich physics already configured
 Proven sim-to-real transfer with 82-99% success rates on similar tasks
 PPO-compatible implementation already working in Isaac Lab
 Complete source code available in the Isaac Lab repository
 Bachelor-thesis-scale complexity when simplified appropriately
The key insight from multiple sources is that task-space impedance control
dramatically outperforms joint-space control for contact-rich insertion tasks, reducing
sample complexity by 3-4×.ar5iv.labs.arxiv+1
1. Recommended System Architecture
Control Flow Diagram
text
┌────────────────────────────────────────────────────
─────────┐
│ OBSERVATION VECTOR │
│ [joint_pos (7), joint_vel (7), ee_pos_rel (3), │
│ ee_quat_rel (4), ee_vel (6), force_torque (6)] │
└────────────────────────────────────────────────────
─────────┘
↓
┌────────────────────────────────────────────────────
─────────┐
│ PPO POLICY │
│ Input: 26-dim observation → MLP (256, 128) → │
--- PAGE 2 ---
│ Output: 6-dim action (Δx, Δy, Δz, Δroll, Δpitch, Δyaw) │
└────────────────────────────────────────────────────
─────────┘
↓
┌────────────────────────────────────────────────────
─────────┐
│ TASK-SPACE IMPEDANCE CONTROLLER │
│ Action scale: 0.025 (±2.5% of workspace per step) │
│ Stiffness: 800 N/m (position), 40 Nm/rad (orientation) │
│ Damping: 40 Ns/m (critical damping) │
└────────────────────────────────────────────────────
─────────┘
↓
┌────────────────────────────────────────────────────
─────────┐
│ FRANKA/UR10 │
│ Control frequency: 60-100 Hz │
│ Physics timestep: 1/120 Hz (Isaac Sim default) │
└────────────────────────────────────────────────────
─────────┘
↓
┌────────────────────────────────────────────────────
─────────┐
│ CONTACT INTERACTION │
│ Peg-hole clearance: 0.5-1.0 mm │
│ Friction: 0.75 (calibrated to real materials) │
│ Solver iterations: 4-8 (contact stability) │
└────────────────────────────────────────────────────
─────────┘
↓
New observation (loop)
Why this architecture:
--- PAGE 3 ---
 Impedance control provides inherent compliance, preventing excessive
contact forces and enabling "search" behavior through
contactar5iv.labs.arxiv+1
 Incremental actions (delta commands) generalize better than absolute
targets and avoid encoding task-specific biasesisaac-sim+1
 60-100 Hz control matches real robot capabilities while avoiding aliasingarxiv
 Task-space commands simplify the learning problem by abstracting away
inverse kinematicsias.informatik.tu-darmstadt
2. Recommended Task Definition
Robot
 Franka Emika Panda (7-DOF) - default in Isaac Lab Factory
environmentsisaac-sim.github+1
 Alternative: UR10e (6-DOF) if you have access to one for sim-to-realisaac-sim
 Why Franka: Built-in impedance control, widely used in research, excellent
Isaac Lab support
Peg Geometry
 Cylindrical peg: 10 mm diameter, 50 mm length
 Material: ABS plastic (density ~1.04 g/cm³)
 Attached to: End-effector (rigid connection, no gripper needed for thesis)
Hole Geometry
 Circular hole: 10.5-11.0 mm diameter (0.5-1.0 mm clearance)
 Depth: 30 mm (sufficient for "fully inserted" detection)
 Material: Aluminum or steel (higher friction ~0.75)
 Mounted on: Fixed base (no movement during insertion)
Initial State Randomization
Based on IndustReal and Isaac Lab Factory:isaac-sim.github+2
Parameter Curriculum Level 1 Level 2 Level 3 Level 4
Lateral position error (x, y) ±5 mm ±10 mm ±15 mm ±20 mm
Height above hole (z) 20-30 mm 30-40 mm 40-50 mm 50-60 mm
Orientation error (roll, pitch) ±2° ±5° ±8° ±10°
--- PAGE 4 ---
Parameter Curriculum Level 1 Level 2 Level 3 Level 4
Orientation error (yaw) ±5° ±15° ±30° ±45°
Peg-hole clearance 1.0 mm 0.75 mm 0.5 mm 0.5 mm
Friction coefficient 0.5 0.6 0.7 0.75
Episode Termination
 Success: Peg tip reaches 25 mm depth (out of 30 mm hole)
 Failure (timeout): 2000 steps (~20-30 seconds at 60-100 Hz)
 Failure (crash): Contact force > 50 N (safety limit)
 Failure (jamming): No progress for 500 consecutive steps
Source: Success criteria from IndustReal, timeout from Factory PegInsertarxiv+1
3. Recommended Observation Vector
Minimal State-Based (Simulation-Only Proof-of-Concept)
python
# Total: 26 dimensions
observations = {
# Proprioceptive (robot state)
"joint_pos": robot_joint_positions, # 7 dims (Franka)
"joint_vel": robot_joint_velocities, # 7 dims
# Task-relevant relative state
"ee_pos_rel": peg_tip_to_hole_center, # 3 dims (x, y, z)
"ee_quat_rel": peg_to_hole_orientation, # 4 dims (quaternion)
"ee_vel_lin": end_effector_linear_vel, # 3 dims
"ee_vel_ang": end_effector_angular_vel, # 3 dims
# Contact information (critical for insertion)
"force_torque": ee_force_torque_sensor, # 6 dims (Fx, Fy, Fz, Tx, Ty, Tz)
}
--- PAGE 5 ---
Why this works:
 Relative pose (peg-to-hole) is the most informative signal for insertionarxiv+1
 Force/torque enables contact-aware behavior and jamming detectionacm+1
 No privileged information - all observations available on real robots with F/T
sensor
For Sim-to-Real Transfer
Add observation noise during training to match real sensor characteristics:isaac-
sim+1
python
# Add to observation config
"joint_pos_noise": UniformNoise(n_min=-0.001, n_max=0.001), # ±1 mm
"joint_vel_noise": UniformNoise(n_min=-0.01, n_max=0.01), # ±0.01 m/s
"force_torque_noise": UniformNoise(n_min=-0.5, n_max=0.5), # ±0.5 N
For Vision-Based Extension (Future Work)
Replace relative pose with depth/RGB observations:github+1
 Depth image: 96×96 from wrist-mounted camera
 RGB image: 96×96 (optional, for texture/feature detection)
 Require visual encoder (CNN or ViT) to process images
Not recommended for Bachelor thesis - adds significant complexity and training
time.
4. Recommended Action Space
Task-Space Impedance Control (Strongly Recommended)
python
# 6-dimensional action space
actions = {
"delta_x": Δx, # -1 to 1 → scaled to ±0.025 m
"delta_y": Δy, # -1 to 1 → scaled to ±0.025 m
"delta_z": Δz, # -1 to 1 → scaled to ±0.025 m
"delta_roll": Δroll, # -1 to 1 → scaled to ±0.025 rad (~1.4°)
"delta_pitch": Δpitch, # -1 to 1 → scaled to ±0.025 rad
--- PAGE 6 ---
"delta_yaw": Δyaw, # -1 to 1 → scaled to ±0.025 rad
}
Action interpretation:
 Actions are incremental targets for the impedance controller
 Controller computes: τ = K_p * (x_target - x_current) + K_d * (v_target -
v_current)
 Default gains: K_p = 800 N/m (position), K_d = 40 Ns/m (damping)
Why impedance control:
1. 3-4× faster learning than joint torque or PD controlar5iv.labs.arxiv+1
2. Inherent compliance prevents damage and enables search behavior
3. Task-space abstraction simplifies inverse kinematics learning
4. Proven sim-to-real transfer in IndustRealarxiv
Alternative: Joint-Space Control (Not Recommended)
python
# Only if impedance control unavailable
actions = {
"joint_pos_target": [7 dims], # Relative joint positions
}
Disadvantages:
 Requires learning inverse kinematics implicitly
 Less compliant, more prone to jamming
 2-3× slower convergenceias.informatik.tu-darmstadt
5. Recommended Controller
Primary: Task-Space Impedance Controller
Implementation in Isaac Lab:
python
# From Isaac-Deploy-GearAssembly configuration [web:19]
actuators = {
"arm": ImplicitActuatorCfg(
--- PAGE 7 ---
joint_names_expr=["joint_.*"],
effort_limit=87.0,
velocity_limit=2.0,
stiffness=800.0, # Calibrated to match real behavior
damping=40.0, # Critical damping
),
}
Control loop:
1. Policy outputs delta pose [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
2. Impedance controller computes joint torques:
text
τ = J^T * (K_p * Δx + K_d * Δv)
where J is the Jacobian
3. Torques applied at 100 Hz (Franka default)
Why Not Other Controllers?
Sample Sim-to-
Controller Compliance Recommendation
Efficiency Real
Impedance (task- Proven
Excellent High ✓ Use this
space) arxiv
Inverse Dynamics Good Medium Theoretical Alternative
Joint PD Moderate Low Possible Avoid
Direct Torque Poor Very Low Difficult Avoid
Source: Empirical comparison inar5iv.labs.arxiv+1
6. Recommended Reward Function
Dense, Staged Reward Design
Based on IndustReal, S2P, and comparison study:arxiv+2
python
def compute_reward(peg_pos, hole_pos, peg_quat, hole_quat,
insertion_depth, force, torque, success):
--- PAGE 8 ---
# === Phase 1: Approach (peg above hole) ===
distance_xy = sqrt((peg_pos[0] - hole_pos[0])**2 +
(peg_pos[1] - hole_pos[1])**2)
distance_z = peg_pos[2] - hole_pos[2] # Height above hole
r_approach = -1.0 * distance_xy - 0.5 * max(0, distance_z - 0.03)
# Penalize being far from hole center and too high
# === Phase 2: Alignment (peg near hole entrance) ===
orientation_error = acos(2 * (peg_quat · hole_quat)**2 - 1)
r_alignment = -2.0 * orientation_error # Penalize misalignment
# === Phase 3: Contact & Insertion ===
# Reward insertion progress (dense signal)
r_insertion = 5.0 * insertion_depth # 5 reward per meter inserted
# Penalize excessive forces (safety + realism)
force_magnitude = sqrt(force[0]**2 + force[1]**2 + force[2]**2)
r_force_penalty = -0.1 * max(0, force_magnitude - 10.0) # Soft limit at 10N
# Penalize excessive torques
torque_magnitude = sqrt(torque[0]**2 + torque[1]**2 + torque[2]**2)
r_torque_penalty = -0.1 * max(0, torque_magnitude - 1.0) # Soft limit at 1Nm
# === Phase 4: Success Bonus (sparse) ===
r_success = 100.0 * success # Large bonus for full insertion
# === Total Reward ===
total_reward = (r_approach + r_alignment + r_insertion +
--- PAGE 9 ---
r_force_penalty + r_torque_penalty + r_success)
return total_reward
Reward Components Explained
Component Type Weight Purpose Source
Distance (xy) Dense -1.0 Guide to hole center arxiv+1
Height (z) Dense -0.5 Prevent hovering too high arxiv
Orientation Dense -2.0 Align peg with hole arxiv+1
Insertion depth Dense +5.0 Encourage progress arxiv+1
Force penalty Dense -0.1 Prevent jamming acm+1
Torque penalty Dense -0.1 Prevent twisting acm
Success bonus Sparse +100 Define task completion arxiv+1
Reward Scaling Guidelines
 Dense rewards should sum to ~1-10 per step during normal operation
 Sparse success bonus should be ~10-100× the typical per-step reward
 Penalty terms should be small enough to not dominate, but large enough to
shape behavior
 Tune empirically: Start with suggested weights, adjust if policy gets stuck
Common Reward Exploits to Avoid
1. Policy learns to push sideways instead of inserting
o Fix: Add lateral force penalty: -0.2 * sqrt(Fx² + Fy²)
2. Policy "vibrates" to accumulate reward
o Fix: Add action smoothness penalty: -0.01 * ||a_t - a_{t-1}||²
3. Policy gives up after initial contact
o Fix: Increase insertion depth reward weight or add progress bonus
7. Recommended Curriculum
Four-Level Curriculum with Progression Logic
Based on IndustReal's Sampling-Based Curriculum and Factory:arxiv+1
--- PAGE 10 ---
python
class CurriculumLevel:
def __init__(self, name, lateral_error, height_range, orientation_error,
clearance, friction, success_threshold=0.8):
self.name = name
self.lateral_error = lateral_error # ±mm
self.height_range = height_range # (min, max) in mm
self.orientation_error = orientation_error # ±degrees
self.clearance = clearance # mm
self.friction = friction
self.success_threshold = success_threshold
curriculum = [
CurriculumLevel("L1_Easy",
lateral_error=5,
height_range=(20, 30),
orientation_error=2,
clearance=1.0,
friction=0.5),
CurriculumLevel("L2_Medium",
lateral_error=10,
height_range=(30, 40),
orientation_error=5,
clearance=0.75,
friction=0.6),
CurriculumLevel("L3_Hard",
lateral_error=15,
height_range=(40, 50),
--- PAGE 11 ---
orientation_error=8,
clearance=0.5,
friction=0.7),
CurriculumLevel("L4_Expert",
lateral_error=20,
height_range=(50, 60),
orientation_error=10,
clearance=0.5,
friction=0.75),
]
Progression Logic (Sampling-Based)
python
def update_curriculum(current_level, success_rate, z_low):
"""
Sampling-based curriculum from IndustReal [web:24].
Key insight: Sample from entire range, but increase lower bound.
"""
if success_rate > 0.8:
# Advance: increase difficulty
z_low += 5 # mm (increase minimum height)
if z_low >= current_level.height_range[1]:
# Promote to next level
current_level_idx += 1
z_low = curriculum[current_level_idx].height_range[0]
elif success_rate < 0.1:
# Retreat: decrease difficulty
z_low -= 3 # mm (gentler than advancement)
if z_low < curriculum[current_level_idx].height_range[0]:
--- PAGE 12 ---
current_level_idx = max(0, current_level_idx - 1)
return current_level_idx, z_low
Why this works:
 Prevents overfitting to easy initial states (common failure mode)
 Maintains exploration by sampling from full range
 Dynamic adjustment based on performance prevents getting stuck
8. Recommended PPO Configuration
Starting Hyperparameters
Based on Isaac Lab Factory, IndustReal, and action space comparison:isaac-sim+3
python
ppo_config = {
# === Core RL Parameters ===
"gamma": 0.99, # Discount factor (standard for manipulation)
"lam": 0.95, # GAE lambda (standard)
"learning_rate": 2.5e-4, # PPO default, proven in Factory
"clip_range": 0.2, # PPO clip (prevents large updates)
# === Training Schedule ===
"rollout_length": 2048, # Steps per environment per update
"num_envs": 256, # Parallel environments (GPU-dependent)
"batch_size": 64, # Minibatch size (for 2048 rollouts: 32 batches)
"epochs": 10, # PPO epochs per update
# === Loss Weights ===
"entropy_coeff": 0.01, # Encourage exploration (reduce over time)
"value_loss_coeff": 0.5, # Standard PPO value
"max_grad_norm": 0.5, # Gradient clipping (stability)
--- PAGE 13 ---
# === Network Architecture ===
"policy_net": [256, 128], # 2-layer MLP (standard for manipulation)
"value_net": [256, 128], # Shared architecture
"activation": "elu", # Or "tanh" (Factory default)
# === Normalization ===
"normalize_obs": True, # Critical for stable training
"normalize_returns": True, # Advantage normalization
"clip_obs": 5.0, # Prevent outliers
# === Control Parameters ===
"control_frequency": 60, # Hz (matches real robot)
"physics_timestep": 1/120, # Isaac Sim default
"action_scale": 0.025, # ±2.5% of workspace per step
}
Evidence-Based vs. Reasonable Starting Points
Parameter Source Confidence
PPO default ias.informatik.tu-darmstadt,
learning_rate: 2.5e-4 High
Factory isaac-sim
PPO default ias.informatik.tu-darmstadt,
rollout_length: 2048 High
tested in Factory
num_envs: 256 Factory default isaac-sim, IndustReal arxiv Medium
Gear Assembly isaac-sim, matches 2.5° joint
action_scale: 0.025 High
motion
control_frequency: 60
IndustReal deployment arxiv High
Hz
entropy: 0.01 Standard SB3 PPO, may need decay Medium
network: Common in manipulation arxiv+1 Medium
Tuning Guidelines
1. If training is unstable:
--- PAGE 14 ---
o Reduce learning rate to 1e-4
o Increase clip_range to 0.3
o Reduce action_scale to 0.02
2. If convergence is slow:
o Increase num_envs to 512 (if GPU memory allows)
o Increase entropy to 0.02 (more exploration)
o Try epochs: 5 → 10 → 15
3. If policy gets stuck:
o Check reward scaling (may be too sparse)
o Verify curriculum progression (may be too fast)
o Add action noise during training
9. Implementation Sequence
Step-by-Step Order with Success Criteria
Phase 1: Environment Setup (Week 1-2)
Step 1.1: Create the environment
 Clone Isaac Lab repository
 Navigate to
source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/factory/
 Copy factory_env_cfg.py to new file peg_insert_simple_cfg.py
 Modify: Replace complex gear/nut geometry with simple cylinder and circular
hole
Success criterion: Environment launches without errors, Franka arm appears in
viewer
Step 1.2: Verify robot control without RL
 Run deterministic script to move robot to pre-insertion pose
 Verify: No physics errors, robot reaches target within 5 mm
Success criterion: Deterministic motion works, robot doesn't crash into table
Step 1.3: Verify peg and hole physics
 Spawn peg and hole in scene
 Manually drop peg into hole (scripted)
--- PAGE 15 ---
 Verify: Contact forces appear, peg settles in hole without excessive
penetration
Success criterion: Contact forces < 50 N, penetration < 0.5 mm
Phase 2: Baseline Controller (Week 2-3)
Step 2.1: Implement deterministic insertion
 Scripted policy: Move to hole → Lower → Insert
 No randomization, perfect alignment
 Verify: 100% success with scripted motion
Success criterion: 10/10 successful insertions with scripted controller
Step 2.2: Implement observations
 Add observation terms from Section 3
 Print observations to console during scripted insertion
 Verify: Observations change as expected (e.g., distance decreases)
Success criterion: Observation values match expected physics (within 10%)
Step 2.3: Implement action mapping
 Map policy output → impedance controller targets
 Test with random actions (no RL yet)
 Verify: Robot moves in response to actions, no instabilities
Success criterion: Random actions produce smooth motion, no physics errors
Phase 3: Reward and Training (Week 3-5)
Step 3.1: Implement reward function
 Add reward computation from Section 6
 Log reward components separately during scripted insertion
 Verify: Each term behaves as expected (e.g., r_insertion increases with depth)
Success criterion: Total reward increases during scripted successful insertion
Step 3.2: Train with NO randomization (sanity check)
 Disable all state randomization (perfect initial pose)
 Train PPO for 10,000 steps
--- PAGE 16 ---
 Expected: Policy should learn quickly (easy task)
Success criterion: Success rate > 80% after 5,000 steps
Step 3.3: Add curriculum (Level 1)
 Enable Level 1 randomization only (±5 mm, 20-30 mm height)
 Train for 50,000 steps
 Expected: Policy learns but slower than no-randomization case
Success criterion: Success rate > 70% on Level 1 evaluation
Phase 4: Robustness and Generalization (Week 5-8)
Step 4.4: Add full curriculum
 Enable all 4 curriculum levels
 Implement progression logic from Section 7
 Train for 200,000+ steps
 Expected: Policy gradually improves on harder levels
Success criterion: Success rate > 60% on Level 3, > 40% on Level 4
Step 4.5: Add domain randomization
 Randomize: friction (0.5-0.75), mass (±10%), damping (±20%)
 Train for additional 100,000 steps
 Expected: Policy becomes more robust to parameter variations
Success criterion: Success rate drops < 15% when evaluating on randomized
parameters
Step 4.6: Evaluate robustness
 Create evaluation suite with 100 test cases
 Vary: initial pose, friction, mass
 Measure: Success rate, average insertion time, max contact force
Success criterion: Success rate > 80% on test suite, avg insertion time < 10
seconds
Debugging Decision Tree
text
--- PAGE 17 ---
┌────────────────────────────────────────────────────
─────────┐
│ Training Diagnostics │
└────────────────────────────────────────────────────
─────────┘
│
▼
┌───────────────────────────────┐
│ Reward does not increase? │
└───────────────────────────────┘
│
┌───────────────┴───────────────┐
│ Yes │ No
▼ ▼
┌──────────────────┐ ┌──────────────────────┐
│ Check reward │ │ Insertion succeeds │
│ implementation: │ │ during training? │
│ - Log each term │ └──────────────────────┘
│ - Verify scales │ │
│ - Test with │ ┌─────────┴─────────┐
│ scripted policy│ │ Yes │ No
└──────────────────┘ ▼ ▼
┌──────────────┐ ┌────────────────┐
│ Policy │ │ Reward │
│ learns to │ │ increases but │
│ exploit │ │ task fails │
└──────────────┘ └────────────────┘
│ │
┌───────────────┴─────────┐ │
│ Why? │ ▼
--- PAGE 18 ---
│ 1. Exploits reward │ ┌──────────────┐
│ 2. Contact unstable │ │ Policy │
│ 3. Action scale too │ │ pushes into │
│ large │ │ hole but │
└─────────────────────────┘ │ jams │
└──────────────┘
│
┌─────────────┴─────────────┐
│ Why? │
│ 1. Insufficient compliance│
│ 2. No search behavior │
│ 3. Reward guides wrong │
└───────────────────────────┘
Specific Failure Modes and Solutions
Problem Likely Cause Fix Source
Reward Reward function
Log reward terms separately,
doesn't bug or physics arxiv
test with scripted policy
increase error
Poor reward
shaping (e.g.,
Policy exploits Add force penalty, use SDF-
rewards force acm+1
reward based reward arxiv
instead of
progress)
Insufficient
Policy gets Increase entropy, reduce ias.informatik.tu-
exploration or
stuck at edge stiffness, add action noise darmstadt
compliance
Policy only Overfitting to
Increase randomization, use
succeeds from training arxiv
curriculum
one pose distribution
Catastrophic
Policy
forgetting or Reduce learning rate, increase ias.informatik.tu-
collapses after
too-large clip_range darmstadt
improvement
updates
--- PAGE 19 ---
Problem Likely Cause Fix Source
Solver iterations
Contact Increase
too low or
physics solver_position_iteration_count isaac-sim
timestep too
unstable to 8-16
high
Training
Distribution shift Add more randomization, use
succeeds, eval arxiv
or overfitting leaky curriculum
fails
10. Most Important Conclusions
Single Most Reproducible Paper/Project
IndustReal: Transferring Contact-Rich Assembly from Simulation to
Realityarxiv
 Complete sim-to-real pipeline with 83-99% success rates
 Open-source code (IndustRealLib)
 Detailed implementation notes and troubleshooting
Single Most Useful Open-Source Repository
NVIDIA Isaac Lab - Factory PegInsert Environmentisaac-sim.github+1
 Production-ready environment already in Isaac Lab
 PPO training scripts included
 Sim-to-real deployment guide availableisaac-sim
Single Most Practical Architecture
Task-space impedance control with incremental actionsar5iv.labs.arxiv+1
 3-4× faster learning than joint-space control
 Inherent compliance prevents damage
 Proven in sim-to-real transferarxiv
Single Most Important Implementation Decision
Use dense, staged reward with SDF-based alignment metricarxiv
 SDF reward achieves 88.6% success vs 1.8-54.2% for keypoint-based
 Dense insertion-depth reward prevents early convergence to local optima
 Force/torque penalties prevent jamming and damage
--- PAGE 20 ---
Single Biggest Mistake to Avoid
Training with curriculum that starts too hard or advances too fastarxiv
 Common failure: Policy overfits to partially-inserted state
 Solution: Use sampling-based curriculum with gradual lower-bound increase
 Evidence: Standard curriculum achieved 32.4% success, sampling-based
achieved 88.6%
11. Top 10 Ranked Resources
Ran Reproducibilit Cod Relevanc
Title Type Source
k y e e
IndustReal Paper +
1 ★★★★★ ✓ ★★★★★ arxiv
(2023) Code
Isaac Lab
Environme
2 Factory ★★★★★ ✓ ★★★★★ isaac-sim.github+1
nt
PegInsert
Isaac Lab
Gear
3 Tutorial ★★★★★ ✓ ★★★★☆ isaac-sim
Assembly
Tutorial
Action
Space ias.informatik.tu-
4 Paper ★★★★☆ ✓ ★★★★☆
Comparison darmstadt
(2019)
DRL Peg-in-
5 Hole UR5 Code ★★★★☆ ✓ ★★★★☆ github
(GitHub)
S2P:
Separate
6 Primitive Paper ★★★☆☆ ✗ ★★★★☆ arxiv
Policy
(2025)
Reward
7 Design for Paper ★★★☆☆ ✗ ★★★☆☆ acm
Insertion
--- PAGE 21 ---
Ran Reproducibilit Cod Relevanc
Title Type Source
k y e e
Catalyst-RL
8 Peg-in-Hole Code ★★★☆☆ ✓ ★★★☆☆ github
Tutorial
Berkeley
www2.eecs.berkel
9 Force Paper ★★☆☆☆ ✗ ★★★☆☆
ey
Control RL
Reddit/Foru
10 m Anecdotal ★☆☆☆☆ ✗ ★★☆☆☆ Various
Discussions
12. Practical Lessons from Engineers
Common Failure Modes (From Isaac Lab Users)
1. GPU memory overflow in contact-rich environments
o Fix: Reduce num_envs from 256 to 128 or 64
o Fix: Reduce solver_position_iteration_count from 8 to 4
o Source:isaac-sim
2. Policy learns to push through objects (interpenetration)
o Fix: Use SAPU (Simulation-Aware Policy Update) from IndustReal
o Fix: Increase contact solver iterations
o Source:arxiv
3. PPO doesn't converge on contact-rich tasks
o Likely cause: Reward too sparse or action space too large
o Fix: Use dense insertion-depth reward, reduce action scale
o Source:arxiv+1
4. Sim-to-real gap larger than expected
o Likely cause: Joint friction not modeled in simulation
o Fix: Add joint friction randomization (0.3-0.7 Nm)
o Fix: Use PLAI (Policy-Level Action Integrator) at deployment
o Source:isaac-sim+1
--- PAGE 22 ---
What Actually Works (From Implementation Experience)
 PPO is sufficient for peg insertion (no need for SAC/TD3)isaac-sim+1
 Impedance control > torque control for sample efficiencyias.informatik.tu-
darmstadt
 Curriculum learning is necessary for contact-rich tasksarxiv
 Simple motion primitives help but aren't required with good rewardarxiv
 Start near the hole (not from random workspace) for faster convergencearxiv
 End-to-end RL is feasible but hierarchical approaches converge fasterarxiv
13. Isaac Lab Factory Investigation
What Can Be Reused
From Factory PegInsert environment:isaac-sim.github+1
1. Physics configuration:
o Contact solver settings (already tuned for insertion)
o Friction and material properties
o Timestep and control frequency
2. Robot setup:
o Franka model with impedance control
o Joint limits and actuator configuration
3. Training infrastructure:
o PPO configuration files
o Parallel environment setup
o Checkpointing and logging
What Should NOT Be Copied
1. Complex reward structure - Factory uses keypoint-based rewards which
underperform SDFarxiv
2. Privileged observations - Some Factory environments use ground-truth
object poses not available on real robots
3. Overly aggressive randomization - Factory may use wider ranges than
needed for thesis
Recommended Modifications
--- PAGE 23 ---
1. Simplify geometry: Replace gear/nut with cylinder/circle
2. Use SDF-based reward instead of keypoint-basedarxiv
3. Add force/torque observations for contact-aware behavior
4. Reduce observation space to only real-robot-available sensors
5. Implement curriculum progression from IndustRealarxiv
Factory vs. Your Task
Aspect Factory PegInsert Your Simplified Version
Object geometry Complex (gears, nuts) Simple (cylinder)
Clearance 0.5-0.6 mm 0.5-1.0 mm (easier)
Observations Privileged + real Real-only
Reward Keypoint-based SDF-based (recommended)
Control Impedance Impedance (same)
Complexity High (industrial) Medium (thesis-scale)
Conclusion: Factory provides excellent starting point, but simplify for thesis scope.
14. Limitations and Uncertainties
What Is Directly Supported by Sources
 ✓ Task-space impedance control outperforms joint controlar5iv.labs.arxiv+1
 ✓ SDF-based reward achieves 88.6% successarxiv
 ✓ Sampling-based curriculum prevents overfittingarxiv
 ✓ Isaac Lab Factory PegInsert works with PPOisaac-sim.github+1
 ✓ Impedance control enables sim-to-real transferarxiv
What Is Inferred from Multiple Sources
 → Action scale 0.025 is reasonable (from gear assembly and action space
study )isaac-sim+1
 → 256 environments is feasible (Factory default, but may need reduction
)isaac-sim
 → 60 Hz control is sufficient (IndustReal deployment, PPO study )arxiv+1
What Are Recommended Engineering Decisions
--- PAGE 24 ---
 Use 4-level curriculum (not 2 or 6) - balances complexity and gradual
progression
 Start with Franka (not UR10) - better Isaac Lab integration, easier debugging
 Use force/torque observations - critical for contact-aware behavior, available
on most research robots
 Train for 200,000+ steps - based on IndustReal training curvesarxiv
What Remains Uncertain
 Exact success rate achievable - depends on your specific geometry,
randomization ranges, and tuning
 Training time - depends on GPU (RTX 3090: ~12-24 hours, but may
vary)isaac-sim
 Sim-to-real transfer - your specific robot and sensors may have different
characteristics than Franka + RealSense
 Generalization to different geometries - not tested in sources, would require
additional experiments
15. Final Practical Advice
Timeline Estimate (Bachelor Thesis)
Week Milestone Deliverable
1-2 Environment setup Working simulation with Franka + peg + hole
3-4 Baseline controller Scripted insertion, reward debugging
5-6 Initial RL training Policy succeeds on Level 1 curriculum
7-8 Full curriculum Policy succeeds on all levels
9-10 Domain randomization Robust policy across parameter variations
11-12 Evaluation + thesis 100-test evaluation, thesis writing
Minimum Viable Thesis
If time is limited, focus on:
1. Simplified environment: Cylinder + circular hole (no complex geometry)
2. Basic curriculum: 2 levels instead of 4
3. Evaluation: Success rate vs randomization magnitude
4. Bonus: Sim-to-real if you have access to real robot
--- PAGE 25 ---
Stretch Goals
If ahead of schedule:
1. Vision-based observations: Replace relative pose with depth images
2. Generalization study: Test on different peg shapes (square, triangle)
3. Sim-to-real: Deploy to real Franka or UR10
4. Comparison study: PPO vs SAC vs TD3
Bottom line: Start with Isaac Lab Factory PegInsert, simplify the geometry, add
IndustReal's SDF reward and curriculum, train with PPO using impedance control.
This approach has been proven to work in simulation and real-world deployment,
making it the most reproducible path for your Bachelor thesis.
