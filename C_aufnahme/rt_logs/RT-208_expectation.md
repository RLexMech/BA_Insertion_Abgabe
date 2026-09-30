# RT-208 / RT-208m -- does the interpenetration reading react correctly at known poses? Written BEFORE the run (2026-09-15)

Status: **UNVERIFIED.** Nothing in this file has run.

## Context

User, 2026-09-15: the thesis will evaluate interpenetration, so the reading
must be made plausible first. The fixture mesh is not watertight (RT-64) and
the sign of the Warp query near the fixture surface was never measured. What
exists: RT-95/96 (sign far outside, by construction) and RT-184 (24 separated
poses at the rim, `interpen_max_mm` exactly 0.0). Nothing ever put the part
INTO the fixture by a known amount. Home of the scope:
`docs/decisions_inbox.md`, "Interpenetration record for the thesis".

## Code state

Branch `p5-kraftsensor`, the commit that adds this file; SHA given in chat
with the pull command. `scripts/check_insertion_sdf.py --pose-ladder`, marker
`check_insertion_sdf-2026-09-15a-poseladder`. No Isaac, no stage: Warp +
trimesh + torch in the training PC's Isaac Python, seconds.

## The one hypothesis

**The SAPU reading reacts to known overlaps: it never falls as the part moves
further into a wall or the floor, it sees every overlap the ladder builds, and
it reads exactly 0 where the part is measured to be clear of the fixture.**

Falsified by `VERDICT: READING_SUSPECT` (any of Q1, Q2, Q3 FAIL). The
counter-proof RT-208m must print `MUTATION VERDICT: PASS`.

## Commands

Training PC root `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`.

```powershell
.\scripts\rt_log.ps1 RT-208 python -u scripts\check_insertion_sdf.py --pose-ladder --seed 0
```

```powershell
.\scripts\rt_log.ps1 RT-208m python -u scripts\check_insertion_sdf.py --pose-ladder --seed 0 --mutate-sign
```

## Points

* **R1** marker line `check_insertion_sdf-2026-09-15a-poseladder`; the
  constants line prints `PLAY_X=0.0005876...`, `INTERPEN_THRESH=0.0002938...`,
  `SEATED_TOOL_QUAT_LOCAL=(0.0, 0.0, 1.0, 0.0)`; 64000 sample points.
* **R2 frame chain**: in every `floor` row, `lowest z[mm]` = -(depth) to
  within 0.03 mm (the tip is the part's leading point, D-152 tolerance). In
  the lateral rows, `lowest z` = -33.000 mm.
* **R3 Q1, Q2, Q3** each PASS; `VERDICT: READING_PLAUSIBLE`, exit 0.
* **R4 geometry table** -- REPORTED, no bar: `n of m rows match the
  planar-wall reading`, every DEVIATION row with its direction and offset,
  and the two `centre read at` lines.
* **R5 RT-208m**: Q3 FAIL and `MUTATION VERDICT: PASS`.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| READING_PLAUSIBLE, all rows match | the reading reacts as a planar-wall model predicts; the named assumptions hold for these poses | start the smoke RT-206a |
| READING_PLAUSIBLE, some rows DEVIATION | the reading is monotone and sees overlap, but the local geometry differs from the planar model (lugs, part centre offset, chamfers) | read the deviating rows and the centre lines before the smoke; a real offset is a geometry finding, not a code stop |
| floor rows read 0 past +0.2 mm | the floor is not seen: open mesh or sign error at the floor | STOP; the episode record would under-count |
| Q3 FAIL without mutation | a clear part reads overlap: sign flipped somewhere | STOP |
| R2 off by more than 0.03 mm | the pose chain (seat quaternion, tip offset, frame) is not the env's | STOP; the ladder then measures the wrong pose |
| RT-208m prints MUTATION VERDICT: FAIL | the ladder cannot catch a flipped sign | the probe proves nothing; fix before judging RT-208 |

## What this cannot answer

* Tilted or yawed poses; the ladder is translations at the seated orientation.
* The reading between two policy steps (the env reads at 15 Hz).
* Whether PhysX lets the part reach these overlaps -- this is the Warp query
  alone, not a simulation.
