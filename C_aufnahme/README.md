# UR5e Contact-Rich Insertion

Bachelor thesis project. Goal: contact-rich insertion with a Universal Robots UR5e.

## Status (v2, 2026-09-06)

Every number below names the run that produced it. Nothing here is claimed
without a training-PC run; see `rt_logs/VERDICTS.md`.

- **Scene and assets stand.** Real part and fixture from CAD, workcell
  geometry checked (`scripts/check_workcell_geometry.py`).
- **Controller is OSC**, not joint PD (D-108 fell 2026-09-03). A scripted
  descent seats the part at 36.0 mm depth under 16.7 N (RT-150).
- **The success predicate fires under OSC** — verified by teleport into the
  seated pose (RT-155).
- **A PPO policy inserts the part.** RT-156, 1024 envs, start height drawn
  from −30…+30 mm; top bin (+20…+30 mm) 98.1 %.
- **Generalisation is measured, not assumed.** RT-157 replayed that policy
  from four fixed start heights: 100 / 71.5 / 76.8 / 93.0 % at
  +30/+40/+50/+60 mm. Not monotonic. Where a trace exists, the
  censoring-free rate (first episode per env) is lower: 58.2 % at +40 mm,
  92.8 % at +60 mm. No trace was taken at +30 and +50 mm.
- **The failures are two fixed poses**, identical at every height, both
  parked in contact: one hanging on the pocket rim, one jammed in the mouth.
  The three force channels do NOT encode which way the part is caught.
- **Force abort is 50 N** since D-169 — the first value in this field taken
  off a measurement instead of a guess.

Open: why +40 mm is worse than +60 mm. Next measurement adds `ee_quat` to the
replay trace to settle whether the rim pose is tilted.

## Structure

- `source/insertion/` — the Isaac Lab task (env, cfg, math, scripted policy).
- `scripts/` — entry points and offline checks; index in `scripts/CLAUDE.md`.
- `DECISIONS.md`, `PROBLEMS.md`, `rt_logs/VERDICTS.md` — the record.
- `docs/` — reports, per-run pages, figures, literature checks.

USDs and OBJs are in no git tree. The training PC runs a sparse clone;
`HANDOFF-RL.md` owns its path and checkout set.
