# Task Specification — Cylindrical Peg-in-Hole Insertion (UR10e)

Canonical task specification. Derived from the original The_Task.txt; the software
stack section was corrected per decision D-001 (Isaac Lab 2.3 instead of 3.0 beta).

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
