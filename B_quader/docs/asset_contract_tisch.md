# Asset contract — `Tisch.usd`

One page, only checkable numbers. Read this **before** re-importing or re-exporting
the fixture; it is the operative form of D-016, D-022 and D-023, which state the
same facts in prose and are easy to miss during an import.

Authority: D-022 (bore as configuration constants), D-016 (import invariants),
D-023 (base placement). If this file and those entries ever disagree, the
DECISIONS entries win and this file is wrong.

## Import rule (the one that has been gotten wrong twice)

Export the STEP from Creo with the **part's default coordinate system**.

**Not** the bore coordinate system. Its +z points into the plate, so the table
imports upside down; correcting that needs a 180° rotation about X *plus* a
compensating +0.755 m translation in Z, otherwise the table ends up below the
ground plane. D-022 considered and rejected exactly this variant. The bore is not
represented in the USD at all — it lives in the scene config, see below.

Isaac Sim's CAD import authors millimetre numbers and declares the stage in
millimetres (`metersPerUnit = 0.001`). Isaac Lab does not honour that metadata, so
the asset must be converted to a metre stage. This needs **two coupled edits**:
set `metersPerUnit` to 1.0 *and* apply a compensating 0.001 uniform scale
(`Scale:unitsResolve`). Setting only the metadata produces a 500 m table.
`scripts/fix_stage_units.py` performs both.

**That is necessary but not sufficient** (measured 2026-07-26, PROBLEMS.md). The
unit scale must **not** end up on the default prim. Isaac Lab's spawner authors
`[translate, orient, scale]` on the *referencing* prim, and USD composition lets
that **replace** the referenced `xformOpOrder` rather than merge with it — the
unit scale is silently dropped and the fixture spawns 1000× oversized. The
geometry must therefore sit under a **clean, op-free default prim**, with the
unit scale one level down. `scripts/fix_fixture_asset.py` produces that shape.

The asset must also carry **no `instanceable` prims**: USD schema edits cannot
reach inside an instance prototype, so `collision_props` silently does nothing
and the fixture spawns without collision.

Both defects are invisible in the viewport, because the asset is correctly
proportioned either way, and both are invisible to a direct-open check.
**Every asset check must be run in the referenced context** — that omission is
why a broken asset held `asset contract: PASS` for a full day.

Save **flattened** (`Flatten` / `Export`), never `Save As`. A non-flattened save
writes a thin wrapper whose payload points at a sibling file; renaming or moving
the asset then breaks the reference, the spawn still succeeds via the default
prim, and the fixture is silently invisible. That failure occurred on 2026-07-26
(PROBLEMS.md).

## Expected state of the saved file

| Property | Expected value | Status |
|---|---|---|
| `metersPerUnit` | 1.0 | verified 2026-07-25 |
| Up axis | Z | verified 2026-07-25 |
| BBox extents | 0.500 × 0.600 × 0.755 m | verified 2026-07-25 |
| BBox z range | −0.700 … +0.055 m | verified 2026-07-26 on the re-import |
| Asset origin | plate underside, legs extending to −0.700 m | verified 2026-07-26 |
| Default prim | set, and the sole root child | verified 2026-07-26 (`/tisch`) |
| External composition arcs | none | verified 2026-07-26 (`external_arcs: []`) |

Prim paths differ between imports: the 2026-07-25 asset used `/World` with a
`/World/tisch5` child, the 2026-07-26 re-import uses `/tisch` with an internal
reference to `/tisch/Prototypes/tisch`. Do not hard-code either. Spawn through the
default prim and let `UsdFileCfg` resolve it.

An **internal** reference arc (empty `assetPath`) is how the CAD converter
expresses instancing. It was recorded here as "harmless"; **that was wrong**.
It is harmless for *composition* — the asset stays self-contained — but fatal for
*schema edits*: the prim it creates is `instanceable`, and `collision_props` then
reaches nothing. De-instance it (`fix_fixture_asset.py`). An **external** arc is a
separate defect: it makes the asset depend on a second file, which produced the
invisible fixture on 2026-07-26.

The z range is the cheapest orientation check: upright puts the plate top at
+0.055 and the legs at −0.700. An upside-down asset shows −0.055 … +0.700.
Bounding-box *extents* alone cannot distinguish the two.

Confirm with **both** of these — the first alone is not enough, see above:

```powershell
python scripts\verify_fixture_usd.py --usd <path>\tisch.usd
python scripts\verify_fixture_spawn.py
```

`verify_fixture_spawn.py` is the one that decides: its variant **D** spawns the
asset through Isaac Lab's own spawner and must read `metres (correct)`, and the
`modify_collision_properties` warning must be absent.

## What the USD deliberately does not contain

No bore task frame, no `Xform` for it — do not go looking for one, and do not
author one in. The STEP file carried no datum frame despite `Datums` /
`Extended Datums` under AP242 (D-017, PROBLEMS.md 2026-07-26), and
`scripts/add_bore_task_frame.py` was abandoned rather than debugged.

The bore is realized as constants in the scene configuration instead. All values
are **relative to the environment origin**, never absolute world coordinates
(Isaac Lab offsets each parallel environment by `scene.env_origins`):

| Quantity | Value |
|---|---|
| Fixture spawn | `pos = (0.0, 0.0, 0.700)`, `rot = (1, 0, 0, 0)` — identity, the asset is already upright |
| Plate top surface | `z = 0.755` |
| Bore entrance (task frame origin) | `(0.0, -0.225, 0.755)` |
| Task frame orientation | `(0.0, 1.0, 0.0, 0.0)`, i.e. 180° about X |
| Insertion direction | −Z; depth = `0.755 − z_peg_tip` |
| UR10e base center | `(0.0, +0.190, 0.755)`, 415 mm from the bore along −Y |

Geometry read from the Creo source: plate thickness 55 mm, total height 755 mm,
leg length 700 mm, bore Ø30 mm as a **blind hole 30 mm deep**. The −Y direction
of the bore was confirmed in the Isaac Sim viewport on 2026-07-26.

Consequence to keep in mind: because the bore sits on the −Y half of a plate that
spans y ∈ [−0.300, +0.300], a mirrored import would place it at +0.225. The plate
is symmetric in Y, so such a case is repairable by flipping the sign of both the
bore and the base constant together — but it must be measured in the viewport,
not assumed.

## Collision

Spawn without `rigid_props` → static collider, no rigid-body dynamics (D-018).
Collision approximation must be an exact triangle mesh, not the default convex
hull: a convex hull seals the blind bore, and the defect would only surface an
increment later when the peg refuses to enter. Static bodies are the one case in
which PhysX permits an exact triangle mesh.

## Addendum (2026-08-05, D-033) — the generator replaces the import

**The square table is no longer imported from Creo.** It is generated by
`scripts/author_tisch_square.py` as a composition of nine axis-aligned boxes
(four plate pieces around the pocket, the pocket floor, four legs), each a
`UsdGeom.Cube` with its own `UsdPhysics.CollisionAPI`.

This supersedes the import-specific parts of the D-029 addendum below. What
changes:

- **Points 2 and 3 of that addendum no longer apply as written.** The file name
  now carries the pocket size — **`tisch_square_b{mm}.usd`**, e.g.
  `tisch_square_b45.usd` — because the pocket became the curriculum variable
  (D-033) and a size-less name could hold any rung. The resolver accepts the
  size for the configured run only, and rejects both the size-less
  `tisch_square.usd` and the round `tisch.usd`.
- **The approximation question disappears.** Boxes are native PhysX primitive
  colliders, so there is no mesh approximation step to get wrong.
  `add_fixture_collision.py --approximation none` is not part of the square
  chain at all, and the sealed-pocket failure mode is impossible by
  construction rather than checked for.
- **Point 1 stands unchanged.** The generator authors the same origin
  convention, bbox and spawn translation, so `verify_fixture_usd.py` and
  `verify_fixture_spawn.py` expect exactly the values tabulated below.

What is still contract, and is now asserted by the generator itself before it
writes (PASS/FAIL, mirrored into a `.author.json` sidecar): metersPerUnit 1.0,
up axis Z, an op-free default prim, the 0.500 × 0.600 × 0.755 m bbox over
z ∈ −0.755 … 0.000, the opening measured back from the wall prims, the pocket
depth, nine collision prims, no `MeshCollisionAPI`, no instanceable prims.
`scripts/check_tisch_geometry.py` checks the same box layout offline, without
Isaac, so a geometry slip is catchable on the dev PC.

The 45°-yaw control in `verify_peg_passability.py` now switches itself off for
pockets of 42.43 mm and wider, where the peg's diagonal fits and the turned peg
is *supposed* to enter — including the curriculum's stage 0 at 45 mm. It
reports `yaw_control: inactive (diagonal fits)` there instead of
`SQUARENESS_ABSENT`.

## Addendum (2026-07-28, D-029) — square re-import: `tisch_square.usd`

> **Superseded in part by the D-033 addendum above.** The origin convention and
> the expected file state (point 1) still hold; the import procedure, the file
> name (point 2) and the collision approximation (point 3) do not.


The square pivot re-imports the table with a **32 × 32 mm, 35 mm deep blind
pocket** cut at the old bore centre. Everything above about units, flattening,
instancing, op-free default prims and referenced-context checks applies
unchanged to the new import. Three things are different, and all three are
load-bearing:

1. **New origin convention.** The Creo export places the part origin **on the
   opening plane at the pocket centre** (the old asset had it at the plate
   underside). Expected state of the saved file therefore changes to:

   | Property | Expected value |
   |---|---|
   | BBox extents | 0.500 × 0.600 × 0.755 m (unchanged) |
   | BBox z range | **−0.755 … 0.000 m** (upright; inverted reads 0.000 … +0.755) |
   | Asset origin | opening plane, pocket centre — coincides with the task frame |
   | Spawn translation | `(0.0, −0.225, 0.755)` — fixture position IS the pocket entrance now |

   `verify_fixture_usd.py` encodes the new z range; the old round asset now
   deliberately reads `unrecognized` there.

2. **File name is part of the contract.** Save as **`tisch_square.usd`**. The
   env resolver accepts this name only (plus the `PROXYTASK_TISCH_USD`
   override) and does NOT fall back to the round `tisch.usd` sitting in the
   same folder: silently spawning the round table under the square task would
   reproduce the stale-asset failure mode — everything runs, every result void.

3. **Collision approximation `none` is mandatory, not preferred.** The blind
   pocket is a **non-convex** feature of the plate. A convex-hull collider
   seals it flat, and a sealed pocket during training is indistinguishable
   from an exploration failure — the most expensive misdiagnosis available.
   Run `add_fixture_collision.py --approximation none` and then
   `verify_peg_passability.py`: its centred pass reacting from ~0 mm depth is
   the sealed-pocket signature, and its 45°-yaw control (diagonal 42.4 mm
   against the 32 mm opening) must react from ~0 mm or the verdict is
   `SQUARENESS_ABSENT`.

The pocket itself is still not represented as a task frame in the USD; it
lives in `proxytask_tasks_cfg.py` (POCKET_SIDE, POCKET_DEPTH,
OPENING_ENTRANCE_POS), which is also where the derivation chain now starts
from the opening plane rather than from the plate underside.
