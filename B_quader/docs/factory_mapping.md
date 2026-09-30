# Factory PegInsert → this project — parameter mapping

Concrete side-by-side of the Isaac Lab `Isaac-Factory-PegInsert-Direct-v0`
template and this project's UR10e peg-insertion environment. It records, for
every relevant parameter, whether we **adopt** it unchanged, **adapt** it, or
**keep our own** value (deliberately not taken from Factory).

Realizes D-012 (Direct workflow with Factory as structural template); the
per-topic decisions live in D-003/D-004/D-005/D-006/D-014/D-016…D-020.

Provenance:
- Factory column is quoted verbatim from the Isaac Lab source at tag **v2.3.2**
  (`source/isaaclab_tasks/isaaclab_tasks/direct/factory/`:
  `factory_env_cfg.py`, `factory_tasks_cfg.py`), inspected 2026-07-25.
- Our column is the planned/decided value with its deciding decision entry.
  Every "adopt"/"adapt" value is a **starting point and UNVERIFIED** until it
  runs on the training machine (D-002); gains and physical parameters are
  measured in simulation, not copied (CLAUDE.md).

---

## 1. Structure and infrastructure — adopt as-is

| Aspect | Factory (v2.3.2) | This project | Handling |
|---|---|---|---|
| Workflow / base class | `DirectRLEnvCfg` | `DirectRLEnv(Cfg)` | **Adopt** — confirms D-012 |
| File layout | `*_env_cfg.py`, `*_env.py`, `*_control.py`, `*_tasks_cfg.py`, `agents/` | same layout | **Adopt** |
| Contact solver / friction / timestep | tuned for insertion | reuse | **Adopt** (§13) |
| Parallel-env / checkpoint / logging infra | rl_games PPO + parallel scene | reuse pattern | **Adopt** |
| Asset config pattern | `FixedAssetCfg` / `HeldAssetCfg` (usd_path, diameter, height, base_height, friction, mass) | same two-class split: hole = fixed, peg = held | **Adopt** (structure only; values below) |

## 2. Environment parameters — adapt

| Parameter | Factory (v2.3.2) | This project | Decision |
|---|---|---|---|
| Robot | Franka, 7-DOF + 2-finger gripper | UR10e, 6-DOF, no gripper, peg welded | D-005/D-019/D-020 |
| `reset_joints` | 7 values (Franka) | 6 values (UR10e) | D-020 |
| `action_space` | 6 (task-space) | 6 (task-space Δpose) | **same count** — D-003 |
| Controller | impedance (`factory_control.py`) | OperationalSpaceController | D-014 |
| `default_task_prop_gains` | `[100,100,100,30,30,30]` | starting point; **measure UR10e gains in sim** | D-014 |
| `decimation` | 8 | 8 (starting point) | adopt, UNVERIFIED |
| `episode_length_s` | 10.0 (PegInsert) | 10.0 (starting point) | adopt, UNVERIFIED |
| `observation_space` | 21 (`fingertip_pos_rel_fixed`, `fingertip_quat`, `ee_linvel`, `ee_angvel`) | ~24, real-only, **+ wrist F/T** | D-005 |
| `state_space` (privileged critic) | 72 (held/fixed asset ground-truth poses) | **not used** — no privileged obs | D-005 |
| Reward | keypoint-based (`keypoint_scale = 0.15`) | staged dense → SDF upgrade; **not keypoint** | D-004 |
| Randomization ranges | industrial / wide | reduced to thesis scope | D-006 |
| `success_threshold` / `engage_threshold` | `0.04` (frac. socket height) / `0.9` | adopt structure; success = 25 mm insertion depth | D-017, task spec |

## 3. Geometry — kept as ours (deliberately NOT taken from Factory)

The task geometry is defined by this project's specification and CAD asset, not
by Factory. Factory's 8 mm gear/nut/peg dimensions are **reference only**.

| Quantity | Factory (v2.3.2) | This project (kept) | Decision |
|---|---|---|---|
| Table / fixture | `factory_hole_8mm.usd` (generic socket) | **our CAD table `tisch5`**, static `AssetBaseCfg` + `UsdFileCfg` | D-016/D-018 |
| Hole (fixed asset) Ø | `0.0081 m` (`Hole8mm`) | **Ø 0.030 m** (in-stage nominal) | D-006/D-016 |
| Hole / socket height | `0.025 m` | our bore depth (from CAD) | D-016 |
| Peg (held asset) Ø | `0.007986 m` (`Peg8mm`) | **Ø 0.028 m** (tessellated mesh, radius 0.014 m, welded into the robot USD) | D-019 |
| Peg height / length | `0.050 m` | **0.050 m** (stage 1) | D-018 |
| Radial clearance | ≈ 0.057 mm (from the two Ø above) | **1.0 mm** (stage 1) | D-006 |
| Peg realization | separate held rigid body | welded rigid link in the robot USD | D-019 |

**Peg realization, reconciled.** D-018 originally mapped the peg to a spawned
`CylinderCfg` primitive, which contradicted D-019's welded link; the row above
carried both readings until the peg increment. D-019 governs (D-018 addendum 2):
nothing spawns the peg, `HeldAssetCfg` carries its data for the authoring script,
and the collision approximation differs by necessity between the two assets —
the fixture is a static collider and takes the exact triangle mesh, while the peg
belongs to a dynamic articulation, for which PhysX accepts only convex shapes.

**Clearance reality-check.** Factory's actual radial clearance derived from the
source diameters is ≈ 0.057 mm (Ø 0.0081 − Ø 0.007986 = 0.114 mm diametral).
This contradicts `research_cylindertask.md` §13, which states a Factory
clearance of 0.5–0.6 mm; the source value governs (measure-from-source rule).
The relative conclusion is unaffected: our 1.0 mm radial clearance is looser
than Factory's, i.e. our task is mechanically easier.

---

## One-line summary

Adopt Factory's **structure, physics config, and infrastructure**; adapt the
**robot, controller, observations, and reward**; keep **our own geometry**
(table, hole Ø, peg Ø, length, 1.0 mm clearance). Only the Factory *environment
scaffolding* is reused — the physical task remains as planned.
