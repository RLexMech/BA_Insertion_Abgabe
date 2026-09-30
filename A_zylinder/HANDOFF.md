# HANDOFF — single entry point

Last updated: 2026-07-26 (fifth session of the day), dev PC, with the full peg
verification sequence run on the training machine. Update this file at the end of
every work session: what changed, what is verified, what is next. When updating,
do not only append — re-read the whole file and correct what the session's
findings made wrong.

## Current state

Phase 0 is complete and tagged. **Increment 1 is complete and verified**, merged
into `main` (`8259b30`) and tagged `increment-1-verified`. **Increment 2 (the
peg) is complete and verified on the training machine** on the branch
`increment-2-peg`, which is pushed but **not yet merged** — merging and tagging
is step 1 below and the only thing left of this increment.

Acceptance criteria, measured (marker `increment1-scene-2026-07-26e`):

| # | Criterion | Result |
|---|---|---|
| 1 | task still registered | PASS — env loads in every run |
| 2 | base env-local (0, 0.190, 0.755), stable | PASS, unchanged over 3 s |
| 3 | joints = commanded ±0.5° | 0.000° kinematic / 1.95° steady state |
| 4 | tool axis · (0,0,−1) ≥ 0.999 | PASS — 1.000000 |
| 5 | XY ≤ 5 mm, standoff 0.150 ±5 mm | 0.000 mm / 0.150000 m kinematic |
| 6 | env-local identical, world differs by `env_spacing` | PASS, exact |
| 7 | no exception | PASS, including 5 training iterations |

Criteria 3 and 5 are read in two parts by decision (D-026): the kinematic claim of
D-022/D-023 is judged before gravity acts, and the steady-state droop (1.95°,
13.4 mm) is recorded as a measured property of the uncompensated shipped gains,
not fitted away. D-022, D-023 are now `verified`; D-025 carries an addendum
replacing its selected IK branch.

### This session (fourth, 2026-07-26): the environment code, and three asset defects

The environment was written and verified, but most of the session went into
defects that were invisible until the scene actually ran. Each has a PROBLEMS row.

- **The unit scale did not survive being referenced.** The 0.001 compensation sat
  in the asset's root `xformOpOrder`; Isaac Lab's spawner authors
  `[translate, orient, scale]` on the *referencing* prim, which replaces that order
  instead of merging. The table spawned 1000× oversized. Independently visible in
  the GUI, where the UR10e had to be scaled up 1000× to match. Fixed by wrapping
  the geometry under a clean, op-free default prim (`fix_fixture_asset.py`).
- **Collision properties reached nothing.** The geometry sat behind an
  `instanceable` prim, and USD schema edits cannot enter an instance prototype, so
  `collision_props` silently did nothing and the fixture had no collision at all.
  De-instanced, then `add_fixture_collision.py` applies `CollisionAPI` and
  `MeshCollisionAPI` with `approximation = none` — the exact triangle mesh, which
  also settles the blind-bore question deferred from D-020 addendum 2.
- **The D-025 home pose ran the upper arm through the tabletop.** Its selection
  criteria never included plate clearance, because the fixture had no collision
  when they were applied. `screen_home_pose_branches.py` validates the kinematic
  chain against measured body positions to 0.06 mm, then screens all eight
  branches; four keep the whole arm above the plate, and D-025's own remaining
  criteria select branch 3.

Two process lessons, both now fixed in the tooling rather than in anyone's memory:
`verify_fixture_usd.py` held the asset at `asset contract: PASS` for a full day
because it only ever tested the direct-open case — the one usage that never
occurs. The asset contract now requires the referenced-context check. And the
startup report printed once, 33 ms after reset, so an unsettled transient was
misread as plate contact; it now prints at 0.03 s, 1 s and 3 s and tests the plate
volume rather than comparing heights.

Two milestones exist as recoverable baselines per D-015:

- **`phase0-complete`** — annotated tag on `74ad251`, pushed. The state before
  any environment code: toolchain verified, fixture imported, geometry fixed as
  D-022/D-023 constants.
- **`increment-1-verified`** — annotated tag on `8259b30`, pushed. The commissioned
  scene: fixture and UR10e at the D-022/D-023 constants, all seven acceptance
  criteria passing.

`main` has moved past the tag: it is at `90e00b4`, one documentation-only commit
that reconciled stale statements in this file. `increment-1-scene` is fully merged
and pushed. Increment 2 work sits on `increment-2-peg`, cut from `90e00b4` and
pushed with an upstream. All three branches track a remote, so `git pull` works
on any of them, and the training machine reaches this session's work with
`git fetch && git checkout increment-2-peg`.

### Fixture asset: repaired and verified (2026-07-26, third session)

The blocking problem this file previously described is **solved**. Both defects are
rows in PROBLEMS.md, and the operative import rule now lives in
`docs/asset_contract_tisch.md` — read that before ever touching the asset again.

What was wrong, in order: (1) `Tisch.usd` was a 4 KB wrapper whose geometry sat
behind a **payload** to a sibling `tisch5.usd`, which a rename had broken, so a
spawn would have produced a silently **invisible** fixture; (2) the STEP was
briefly re-imported with the **bore** coordinate system, which lands the table
upside down — the variant D-022 explicitly rejected; (3) the correct re-import
arrived as a **millimetre stage** (`metersPerUnit = 0.001`), which Isaac Lab
ignores, so the fixture would have spawned 1000× oversized.

Final verified state, `verify_fixture_usd.py` on the training machine:
`D-016 invariants: PASS`, `orientation: upright`, `composition arcs: 1
(1 internal, 0 external)`, `asset contract: PASS`. Concretely `metersPerUnit`
1.0, up-axis Z, extents 0.500 × 0.600 × 0.755 m, bbox z range −0.700 … +0.055 m
(so the origin sits at the plate underside, matching the D-022 spawn constant),
and the one reference arc is internal (`/tisch/Prototypes/tisch`, converter
instancing) so the file is genuinely self-contained.

Two consequences for the environment code, both non-obvious:

- **Prim paths differ between imports** — the 2026-07-25 asset used `/World` with
  `/World/tisch5`; the repaired one uses `/tisch` with an internal reference to
  `/tisch/Prototypes/tisch`. Spawn through the **default prim**; do not hard-code
  either path.
- **The file name's casing is `tisch.usd` on disk**, while the docs say
  `Tisch.usd`. Windows resolves both, Linux would not. The asset-path resolver in
  `proxytask_env_cfg.py` should try both spellings rather than assume one.

New tooling from this repair, both UNVERIFIED-free now (they ran on the training
machine): `scripts/fix_stage_units.py` converts a millimetre stage to metres via
the two coupled edits — `metersPerUnit` to 1.0 **plus** a compensating 0.001 scale
placed first in `xformOpOrder` — backing up the input and replacing it only after
a fresh-stage contract check. `verify_fixture_usd.py` now separates stage units
from physical metres, derives an orientation verdict from the bbox z range
(extents alone cannot tell upright from inverted), and classifies composition arcs
as internal or external.

### Toolchain (training machine)

- Phase 1 installation check (2026-07-24): unmodified Isaac-Ant-v0 trained
  headless via rsl_rl, GPU rendering confirmed. Versions recorded as a D-001
  addendum — Isaac Sim 5.1.0-rc.19+release.26219.9c81211b.gl, Isaac Lab 2.3.2,
  Python 3.11.15, RTX 3080, driver 591.86, PyTorch 2.7.0+cu118. Canonical run
  command in CLAUDE.md.
- Generated project registers and runs: `pip install -e source\proxytask`
  succeeded, `list_envs.py` lists **`Template-Proxytask-Direct-v0`** (entry point
  `proxytask.tasks.direct.proxytask.proxytask_env:ProxytaskEnv`), and the
  placeholder Cartpole scene renders non-headless.
- Headless rsl_rl baseline (2026-07-25): the user ran
  `scripts\rsl_rl\train.py --task=Template-Proxytask-Direct-v0 --headless
  --max_iterations 5` and confirmed iteration/reward output. Recorded on the
  user's word; the console output was not captured here, an explicit user call
  that deviates from the usual command-plus-output rule. Note the evidence gap if
  this is ever questioned.

### Repository and two-machine setup

- Repository restructured per docs/reference/workflow_tips.txt (D-008); private
  GitHub remote `RLexMech/ur10e-peg-insertion-rl`.
- D-021 template merged into this repository (2026-07-25): pushed from the
  training-machine folder to a bridge repo `RLexMech/proxytask-transfer`, then its
  files — not its git history — merged into this repo's root. `.gitignore` merged,
  keeping the template's blanket "no USD in the repo" rule; USD assets are stored
  outside git entirely, with no small-asset exception (user decision).
- Two-machine workflow (CLAUDE.md § "Two-repo sync workflow"): this repo is the
  source of truth for code; the training machine's
  `C:\Users\Simon\Desktop\Alexander_Pett\temp\proxytask_gen\proxytask` stays the
  executable copy and is kept in sync by copy-paste; `proxytask-transfer` stays
  alive as a diff channel. Practical consequence seen this session: any script
  written here must be copied over manually before it can run — this caused two
  false "it does not work" detours.

### Fixture asset (`Tisch`)

- STEP import: the HOOPS converter crashes when Isaac Sim is launched via the
  isaaclab launcher; starting from the native **isaacsim** install folder works
  (PROBLEMS.md, 2026-07-25). No permanent CLAUDE.md rule added (user decision).
- Import arrived at 0.001× scale and was corrected via the `Scale:unitsResolve`
  xformOp, then saved as `Tisch.usd` under
  `...\proxytask\source\proxytask\proxytask\tasks\direct\proxytask\assets\Tisch\`.
  **Correction (2026-07-26): it was not saved flattened.** See the blocking
  problem above — the geometry lives behind a broken payload.
- **D-016 invariants pass** (`verify_fixture_usd.py`, training machine,
  2026-07-25): `metersPerUnit = 1.0`, `up_axis = Z`, no residual millimetre
  scaling. Bounding box 0.500 × 0.600 × 0.755 m.
- All geometry confirmed against the Creo source: plate thickness 55 mm, total
  height 755 mm, leg length 700 mm, bore Ø30 mm, bore a **blind hole 30 mm deep**
  (so the peg cannot fall through, and the hole bottom is a physical stop 5 mm
  beyond the 25 mm success threshold). The documented nominal height was a rounded
  0.750 m placeholder and was corrected to 0.755 m in
  `docs/task_specification.md` and in the verify script's `NOMINALS_M`.
- The Creo bore coordinate system is **not** in the USD, and not in the STEP file
  either, despite `Datums`/`Extended Datums` under AP242 (PROBLEMS.md,
  2026-07-26). D-017 anticipated this.

### Task geometry — env-local constants (D-022, D-023)

All values are relative to the **environment origin**, not absolute world
coordinates: Isaac Lab offsets each parallel environment by `scene.env_origins`,
so treating them as absolute would make the bore pose correct only for
environment 0.

| Quantity | Value |
|---|---|
| Fixture spawn | `pos = (0.0, 0.0, 0.700)`, `rot = (1, 0, 0, 0)` |
| Plate top surface | `z = 0.755` |
| Bore entrance (task frame origin) | `(0.0, -0.225, 0.755)` |
| Insertion direction | −Z; depth = `0.755 − z_peg_tip` |
| UR10e base center | `(0.0, +0.190, 0.755)`, 415 mm from the bore along −Y |

The import rule that produces an asset matching these constants — export from the
**part's default coordinate system**, not the bore coordinate system, and save
flattened — is in `docs/asset_contract_tisch.md`. Read that before re-importing;
using the bore coordinate system makes the table import upside down, which D-022
explicitly rejected.

No `Xform` for the bore is authored into the fixture, so do not go looking for one.
`scripts/add_bore_task_frame.py` is the abandoned alternative — it never produced
output on the training machine and is not part of the pipeline. (The older claim
that the asset stays "byte-identical" no longer holds: it was re-imported and
unit-corrected on 2026-07-26 and re-verified afterwards, which is what the
verification now rests on.)

### Increment 1 preparation (earlier sessions, branch `increment-1-scene`)

The approved plan for increment 1 is: replace the Cartpole placeholder with
`Tisch.usd` as a static fixture plus the UR10e at the D-022/D-023 constants, arm
in a tool-down home pose over the bore, and nothing else — no peg, no gripper, no
controller, no reward. Actions are declared but not applied; reward is zero;
dones are timeout only; a one-time startup report prints the numbers that make
D-022/D-023 checkable. Control rate 60 Hz (D-024). None of the environment code
is written yet.

- **`scripts/probe_assets.py`** (new) — one read-only probe answering everything
  the scene code depends on. Run twice on the training machine, 2026-07-26.
  Verified findings are folded into the D-020 addendum: `UR10e_CFG` **is**
  shipped (my expectation that it was not was wrong), default prim `/ur10e`,
  joint names exactly as the DH order assumes, `ee_body_name = wrist_3_link`, and
  `ee_joint` sits at local position (0, 1.3e-8, 0) on `wrist_3_link` — so the
  flange carries **no** offset from the readable body pose. `ur10e_instanceable.
  usd` does not exist.
- **`scripts/tools/compute_home_pose.py`** (new) — closed-form 8-branch UR IK,
  numpy only, no Isaac import, Python 3.11 compatible. **Verified on the dev PC**:
  200 random FK -> IK -> FK round-trips at tolerance 1e-9 for both the UR10e and
  UR10 tables, every branch reproducing its target and the reference
  configuration recovered each time. The USD joint offsets independently
  reproduce the assumed UR10e DH table, so the table is confirmed against the
  asset. Selected home pose and the reasoning are recorded as D-025.
- **D-023 gain:** the wrist-singularity margin |sin q5| is 1.0000 in all eight
  branches and both DH tables (q5 = +-90 deg exactly); shoulder margin
  radius/d4 = 2.38. D-023's open wrist caveat is answered analytically; the
  simulator run now only has to confirm it. (It has since: D-023 is `verified`
  per the acceptance table above.)
- **`scripts/probe_assets.py` section 4 has now run** (training machine,
  2026-07-26), after two runs in which it produced no output at all — the cause was
  a stale manual copy of the script, not a defect in the section. The full measured
  API surface is recorded as **D-020 addendum 2** and the environment code is to be
  written against it, not against memory. The script gained `--only-api` (sections
  1–3 download robot USDs from Nucleus and dominate the runtime; section 4 is pure
  introspection) and a startup banner naming the sections a given copy contains, so
  a stale copy is now visible in the first line of output.

  The single most consequential finding: **`UsdFileCfg` has no
  `mesh_collision_props` field**, so the exact triangle mesh the blind bore needs is
  not reachable that way. Risk 3 of the old plan materialized and its fallback
  applies — `collision_props` only, blind-bore passability deferred to the peg
  increment, where a convex hull sealing the bore would actually be observable.

### Design references

- Factory PegInsert source inspected from the real IsaacLab v2.3.2 GitHub source
  (not from research_cylindertask.md prose). Confirms D-012 structurally:
  `FactoryEnvCfg` extends `DirectRLEnvCfg`. Full adopt/adapt/keep-ours comparison
  in `docs/factory_mapping.md`, including the correction that Factory's actual
  radial clearance is ≈0.057 mm, not the 0.5–0.6 mm stated in
  research_cylindertask.md §13 — the source value governs. Our own geometry
  (Ø30 mm bore, Ø28 mm peg, 1.0 mm radial clearance) stays as planned; only the
  environment scaffolding is reused.
- Settled workflow decisions: D-012 Direct workflow, D-013 rsl_rl, D-014
  OperationalSpaceController, D-015 git checkpoint model.

## Increment 2 (peg) — complete and verified on the training machine

**The peg fits the bore.** `verify_peg_passability.py`, training machine
2026-07-26, verdict **PASSABLE**: the Ø28 mm welded peg descended to 27 mm in
the Ø30 mm blind bore — past the 25 mm success threshold, short of the 30 mm
bottom — with a free-step reaction of 1.05× its own contact-free noise, i.e.
touching nothing, while a deliberately 4 mm-misaligned control rose to 4.78×
from 1 mm depth onward. Both passes' contact-free maxima are equal to six
significant figures (0.00272775 rad), which is what excludes the one competing
explanation, that the control's reaction came from the shifted arm's gravity
rather than from contact. Full numbers and the protocol correction are in D-028.

This closes by measurement the question deferred in D-020 addendum 2: **the
blind bore is not sealed by a convex hull.** The fixture's exact triangle mesh
admits the peg at the planned 1.0 mm radial clearance.

D-019 is `verified`: PhysX keeps `peg_link` as a distinct articulation body
rather than collapsing the fixed joint, so the contact sensor has a stable
target — the condition under which the welded variant was chosen. Measured
`flange -> peg origin` 0.000000 m and `tip along tool axis` +0.050000 m; and
under gravity the tip's extra 2.209 mm of lateral offset is exactly
0.050 · sin(2.53°), confirming length and direction by a second route.

Runs completed: `author_peg_ur10e.py` (13/13 PASS), `zero_agent` at 16 and 128
environments (no exception, env-local values identical while world origins
differ by `env_spacing`), `verify_peg_passability.py` (PASSABLE).

Still open, deliberately: no reward, no controller, no observation change, and
no ContactSensor object — only its spawn-time prerequisite
(`activate_contact_sensors`, confirmed as a declared cfg field in the report).

### What was written (dev PC)

Branch **`increment-2-peg`**, cut from `main` at `90e00b4`. Increment 1 is closed
and untouched. The bore was confirmed visually in Isaac Sim on 2026-07-26, which
cleared the gate the previous handoff put first; the plan is
`C:\Users\alexp\.claude\plans\cozy-crunching-charm.md`.

Written on the dev PC and since verified on the training machine (above):

- `task_geometry.py` → **`proxytask_tasks_cfg.py`** (git mv, history preserved),
  keeping every constant and adding the bore/peg values, `FixedAssetCfg` /
  `HeldAssetCfg`, and `resolve_peg_robot_usd_path()` (D-027). `RADIAL_CLEARANCE`
  is derived from the two diameters rather than repeated.
- **`scripts/author_peg_ur10e.py`** writes `peg_link`, its geometry, mass/inertia
  and the `peg_weld` fixed joint into a flattened copy of the shipped UR10e USD
  (D-019 addendum). Output `assets/Robot/ur10e_peg.usd`, outside git.
- **`scripts/verify_peg_passability.py`** measures the headline question with a
  self-calibrating penetration probe and a deliberately misaligned negative
  control (D-028).
- The env points the robot spawn at the peg USD at scene-build time and reports
  the peg tip, its insertion depth, and a wrong-side authoring check along the
  tool axis. New marker `increment2-peg-2026-07-26a`.

Three plan deviations, each with its reason recorded in the commit and the
decision entries: the peg geometry is a tessellated mesh rather than a
`UsdGeom.Cylinder` gprim (PhysX has no native cylinder collider); its collision
approximation is `convexHull` rather than the fixture's `none` (PhysX allows
exact triangle meshes only for static colliders, and the peg is dynamic); and
`resolve_fixture_usd_path()` stayed in `proxytask_env_cfg` because
`verify_fixture_spawn.py` imports it from there.

Two offline checks did run, and both matter:

- The mesh generator produces a closed surface, bounding box exactly
  0.028 × 0.028 × 0.050 m, positive divergence-theorem volume matching the
  inscribed 64-gon prism (which is what would expose inverted winding).
  Faceting error 0.0169 mm, 1.7 % of the radial clearance.
- The descent path solves 128 of 128 waypoints in both passes with a worst FK
  error of 3e-16 m, and **its first waypoint reproduces the verified home pose
  to 4.9e-07 rad** — the frame convention is therefore validated against a
  training-machine measurement, not merely self-consistent.

One arithmetic error was caught before it reached the asset: a draft had the
peg's diagonal inertia at 2.618e-5 (Steiner mis-applied by a factor of ten).
The value written is the task specification's about-COM 8.23e-6, and the
authoring script asserts the Steiner identity at startup.

## Next steps (1–3)

1. **Close increment 2.** Nothing is left to measure; the branch is verified and
   pushed. Merge and tag on the dev PC, following D-015 exactly as increment 1
   was closed:

   ```
   git checkout main && git merge --no-ff increment-2-peg
   git tag -a increment-2-verified -m "Peg welded and bore passability measured"
   git push && git push --tags
   ```

   This is deliberately left undone rather than assumed: closing a milestone is
   the user's call, and the branch state is complete either way.

2. **Contact sensor and the observation vector.** The spawn-time prerequisite is
   set and confirmed; `peg_link` is a real articulation body, so a
   `ContactSensorCfg` now has a stable target. This is where the 6-D wrist
   force/torque signal of D-005 enters, which makes it the natural moment to
   settle **open decision 5**, the observation vector layout — the interim
   `observation_space = 12` carries no claim (D-026). The 50 N force limit is
   still TBD and only becomes measurable once contact forces are read.

3. **Controller increment (D-014, OperationalSpaceController).** The first
   increment where actions are actually applied. Note the standing consequence
   of D-020: the arm gains go to zero there, because the controller takes over
   holding the pose — the 1.96° gravity droop measured in D-026 is a property of
   the uncompensated shipped gains and disappears with the controller, so it
   must not be "fixed" beforehand.

## When to read what

| You are about to… | Read |
|---|---|
| do anything | CLAUDE.md (rules) + this file |
| need project context, roadmap, hyperparameters | docs/project_overview.md |
| touch a settled design question | DECISIONS.md (D-001…D-008, not re-litigated) |
| implement env / reward / curriculum | docs/task_specification.md + only the current phase's section of docs/reference/research_cylindertask.md |
| debug a failure | PROBLEMS.md first — known problems are not re-solved |
| check conformity with the research plan | docs/plan_alignment.md |
| work through the phases by hand (human-facing) | docs/Projektleitfaden.docx |
| audit the repository structure | docs/reference/workflow_tips.txt (benchmark; re-check until every point holds) |
| map Factory PegInsert parameters onto our task | docs/factory_mapping.md (D-012/D-021) |
| place assets or compute insertion depth | the constants table above (D-022, D-023) |
| import, re-export or verify `Tisch.usd` | docs/asset_contract_tisch.md — read it **before** touching the asset |

## Known open items (not next steps, do not lose)

- Force limit 50 N is TBD — revalidate against measured contact-force
  distributions for the 30 mm geometry (Phase 2).
- Stage-2/3 peg diameters: open sub-decision of D-006.
- Observation vector layout for the UR10e: open decision 5.
- **Reward implementation pattern for Phase 2:** when the real reward is
  written (replacing increment 1's zero reward), follow the template's
  pattern of a freestanding `@torch.jit.script` function outside the env
  class, as `compute_rewards` did for Cartpole before it was deleted here
  (increment 1 has no reward to compute, so nothing to JIT yet). The Isaac
  Lab tutorial motivates this by performance at high `num_envs`, not by
  style — do not inline the reward as a plain method.
- `scripts/add_bore_task_frame.py` produced no console output on the training
  machine and was abandoned rather than diagnosed. If USD authoring is ever
  needed again, that thread is unresolved.
- **`CreateJoint - found a joint with disjointed body transforms ... /Robot/
  joints/ee_joint`** appears at every simulation start in the peg increment.
  It names the *shipped* asset's marker joint, whose offset D-020 addendum 1
  measured at 1.3e-08 m — not the authored `peg_weld`, which produces no
  warning at all. Whether it also occurred in increment 1 is unknown: that log
  was not kept. No measured effect — the kinematics match increment 1 to within
  6 µm (XY 0.000776 vs 0.000782, standoff 0.148857 vs 0.148859). Recorded as an
  observation, not a defect; revisit only if flange contact ever behaves oddly.
- **`getAttributeCount called on non-existent path
  /World/envs/env_9/Robot/wrist_3_link/collisions/wrist3`** in the 16-environment
  training run — one environment out of sixteen, at the flange collision geometry.
  **Not reproduced in the peg increment** at 1, 16 or 128 environments, so it is
  non-deterministic rather than resolved. Left here because the flange is where
  contact happens; if it returns once the contact sensor is live, it matters.
- `ActorCritic.__init__ got unexpected arguments: ['state_dependent_std']` — the
  PPO cfg passes a field this rsl_rl version ignores. Cosmetic now; remove or
  confirm when the hyperparameters are actually tuned.
- **Peg friction is deliberately unset.** `HeldAssetCfg.friction` is `None`: no
  measured value exists for the ABS peg, and guessing physical parameters is
  forbidden. No physics material is authored on the peg in this increment.
  Measure it in the contact increment, where it becomes observable.
- **No action writes a joint target**, so the implicit actuators' target comes
  from wherever the USD drives left it. `_reset_idx` now writes
  `set_joint_position_target` defensively (guarded, because that method was not
  probed either). If the report shows the guard's warning line, the arm is
  holding position only by luck and this needs a measured replacement.
- The UR DH base frame and the USD `base_link` frame may differ by pi about z.
  `compute_home_pose.py` prints both q1 families; only q1 differs. If the first
  environment run shows the tool pointing down correctly but the XY offset
  mirrored, switch families — do not re-derive (D-025 caveat).
- The home pose folds the arm tightly (elbow -129.8 deg) because the bore is only
  415 mm from the base, and the arm stands on the same plate that spans
  y in [-0.300, +0.300]. Whether the elbow sweeps into the plate is visible only
  in the non-headless run; neither the IK nor the startup report can show it.
- Asset naming: the fixture is **`Tisch`** conceptually, but the file on disk is
  **`tisch.usd`, lowercase**, and its internal prim path is `/tisch` with an
  internal reference to `/tisch/Prototypes/tisch`. The 2026-07-25 asset used
  `/World` and `/World/tisch5`; older entries (D-016/D-017, PROBLEMS.md) say
  `tisch5` or "Tisch2". Nothing was rewritten retroactively, so **do not hard-code
  a prim path** — spawn through the default prim, and let the asset-path resolver
  accept both filename spellings.
- Stray files in the asset folder from the repair: `tisch.usd.bak` (keep until
  increment 1 passes), plus `tisch.fixed.usd`, `tisch.fix_units.json`,
  `tisch.verify.json` and yesterday's `Tisch.verify.json`. Clean up the latter four
  once the increment is verified.
- The source STEP file on the dev PC is `Creo/tisch5.stp` (42 KB) — the earlier
  `tisch2.stp` name in this file was wrong. Decided 2026-07-26: CAD sources stay
  outside git like the USDs; `*.stp`/`*.step` are now in `.gitignore`. The file
  itself remains on the dev PC as the import source.
- `.claude/settings.local.json` is untracked — local-only harness settings, leave
  untracked.
