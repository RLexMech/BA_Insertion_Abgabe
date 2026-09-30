# Task Specification — Peg-in-Hole Insertion (UR10e)

Canonical task specification. Derived from the original The_Task.txt; the software
stack section was corrected per decision D-001 (Isaac Lab 2.3 instead of 3.0 beta).

> **2026-07-28 — square-peg pivot (D-029).** The current task, on branch
> `square-peg-insertion`, is the **square** variant specified in the addendum
> at the end of this file. The cylindrical sections below are retained as the
> historical baseline the demo sprint verified; where they conflict with the
> addendum, the addendum wins.

## Objective

Develop a practical, reproducible, technically detailed approach for solving a
contact-rich peg-in-hole insertion task with reinforcement learning:

- The robot already holds a simple cylindrical peg (pre-grasped).
- The robot must insert the cylinder into a hole.
- The task is performed in simulation.
- The robot starts above the hole with a positional and possibly rotational error.
- The insertion requires contact interaction, alignment, searching, and handling of
  jamming/contact forces.
- Long-term goal: a robust policy that generalizes to different initial
  perturbations and potentially different simple geometries.
- Scope: simulation first; deployment to a real UR10e is an optional stretch goal.

## Software stack (per D-001)

1. Simulator: NVIDIA Isaac Sim 5.x (exact installed version to be recorded on the
   training machine).
2. RL/robot-learning framework: NVIDIA Isaac Lab 2.3.
3. RL algorithm: PPO (RL library selection is an open documented decision).
4. Reproducibility note: any code or configuration taken from Isaac Lab 3.0
   material must be revalidated against the 2.3 API before use.

## Robot

UR10e (6-DOF). Peg rigidly attached to the end-effector flange; no gripper model
(D-005). Wrist force/torque signal available as observation.

## Table asset

1. Plate thickness: 55 mm
2. Table height: 0.755 m (measured from the Creo source; supersedes the
   earlier 0.75 m placeholder, confirmed via `verify_fixture_usd.py` against
   `Tisch.usd`, 2026-07-25)
3. Leg length: 0.70 m

## Hole geometry (constant across curriculum)

1. Circular hole: 30.0 mm diameter
2. Depth: 30 mm (sufficient for "fully inserted" detection)
3. Material: aluminum or steel (friction ≈ 0.75)
4. Mounted on a fixed base (no movement during insertion)

## Peg geometry

Cylindrical peg, 50 mm length, ABS plastic (density ≈ 1.04 g/cm³), attached to the
end-effector (pre-grasped). Peg diameter and clearance vary across the curriculum
(hole diameter fixed at 30.0 mm; see D-006). Mass and inertia computed via
V = πr²L, m = Vρ, I_axial = ½mr², I_trans(COM) = (m/12)(3r² + L²),
I_trans(grip) = I_trans(COM) + m(L/2)².

Stage 1 — training-friendly (curriculum start):

- Peg diameter: 28.0 mm
- Radial clearance: 1.0 mm (diametral clearance 2.0 mm)
- Mass: 32.02 g
- I_axial: 3.14e-6 kg·m²
- I_transversal (COM): 8.23e-6 kg·m²
- I_transversal (grip point, Steiner): 2.82e-5 kg·m²

Stage 2 and 3 diameters: open sub-decision of D-006, to be informed by simulation
tests of the minimal stable clearance.

## Episode termination

1. Success: peg tip reaches 25 mm depth (of 30 mm hole depth); 25 mm of peg length
   remains protruding above the hole opening at success.
2. Failure (timeout): 2000 steps (≈ 20–30 s at 60–100 Hz control frequency).
3. Failure (crash): contact force > 50 N (safety limit, TBD — to be revalidated
   against measured contact force distributions for the 30 mm hole geometry).
4. Failure (jamming): no progress for 500 consecutive steps.

## Primary reference

NVIDIA Isaac Lab documentation, including the reference architecture:
https://isaac-sim.github.io/IsaacLab/main/source/refs/reference_architecture/index.html

## Addendum (2026-07-28, D-029): square peg, square pocket

The task pivots from the rotationally symmetric cylinder to a square prism, so
that the tool-axis yaw becomes task-relevant (goal symmetry SO(2) → C4).

### Geometry

| Quantity | Value |
|---|---|
| Peg | 30 × 30 × 50 mm square prism, ABS (ρ ≈ 1040 kg/m³), welded to the flange |
| Peg mass / inertia | m = ρa²L = 46.80 g; I_axial = ma²/6 = 7.02e-6 kg·m²; I_trans(COM) = m(a²+L²)/12 = 1.326e-5 kg·m² |
| Pocket | 32 × 32 mm, 35 mm deep, **blind**, cut into the table plate at the old bore centre |
| Clearance | 1.0 mm per axis at φ = 0 |
| Free-yaw window | ±3.96° about each of the four C4 orientations (asin(32/(30√2)) − 45°) |
| Tilt window | ≈ 2.3° over the 50 mm peg |
| Success | tip 25 mm deep with the four-corner containment gate satisfied |

### Curriculum (pocket fixed at 32 mm, peg side varies)

| Side [mm] | Clearance/axis [mm] | Yaw window |
|---|---|---|
| 22.6 | 4.7 | ±45° (diagonal ≤ pocket: C4 constraint inactive — stage 0) |
| 24 | 4.0 | ±25.5° |
| 26 | 3.0 | ±15.5° |
| 28 | 2.0 | ±8.9° |
| 30 (target) | 1.0 | ±3.96° |

Set via `PROXYTASK_PEG_SIDE_MM` (must be < 32); the old
`PROXYTASK_PEG_DIAMETER_MM` raises a hard error.

### MDP deltas against the demo sprint

- Observation 21-dim: demo layout + (cos 4φ, sin 4φ) of the peg yaw
  (C4-invariant, continuous, (1, 0) at every goal orientation).
- Reward: seventh term −w_yaw (1 − cos 4φ)/2, w_yaw = 2.0 (derivation in
  D-029); the out-of-bounds price includes it.
- Reset: ±45° uniform on wrist_3 (pure tool-axis yaw at home), ±0.01 rad on
  the other five joints.
- Gate: four-corner Chebyshev containment of the tip cross-section, replacing
  the radial test.

Everything not listed here (episode length 4 s / 240 steps, action scale,
60 Hz control, table numbers, home pose, robot base placement) is unchanged
from the verified demo sprint.
