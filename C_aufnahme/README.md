# UR5e Contact-Rich Insertion

Bachelor thesis project. Goal: contact-rich insertion with a Universal Robots UR5e.

## Abgabefassung: bestätigter Endstand C

Grundlage ist `153b8b4` aus `Ur5e_ContactRich_Insertion`. Lange Codekommentare
wurden gekürzt und veraltete Testkonfigurationen berichtigt. Die gemeinsamen
Einstellungen der Studienläufe sind direkt voreingestellt:

- 256 parallele Umgebungen, 1500 PPO-Iterationen.
- `start_floor_m=-0.030`, `autodr_stall_buffers=3`.
- Standard-Seed 1; für die Studie jeweils Seeds 1 bis 5 verwenden.
- `dr_mode=off` bleibt der Standard für No-DR; Auto-DR und Fixed-DR werden
  ausdrücklich gewählt. Auch No-DR verwendet die Starthöhen-Anpassung.

Aus `C_aufnahme` mit eingerichteter Isaac-Lab-Umgebung und den Szenen-Assets:

```powershell
python scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=off
python scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=autodr
python scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --headless --seed 1 env.dr_mode=fixed_dr
```

Die Befehle stehen für RT-207, RT-206 und RT-231; jeden mit Seeds 1–5 ausführen.
Die übrigen Einstellungen bleiben erhalten: OSC, 28 Beobachtungswerte,
MLP 128/128, Aktionsänderungsstrafe 0,0034 und Kraftabbruch bei 30 N.

Für tabellenbasierte Auswertung mit `play.py --eval-table` zusätzlich
`env.start_floor_m=0.020` angeben: Die Auswertung verwendet keine Trainings-
Starthöhen-Anpassung. Bei geänderter `start_tip_above_entrance` müssen beide
Werte gleich sein. `run_eval.ps1` setzt die 0,020 m bereits ausdrücklich.
Die historischen Auswertungen liefen auf `8299485d`; diese Abgabefassung
ist durch die beschriebenen Anpassungen kein unveränderter Original-Commit.

Die folgenden Abschnitte beschreiben den historischen Entwicklungsstand.

## Branch `p5-kraftsensor` — what was reached (2026-09-23)

Numbers from `rt_logs/VERDICTS.md` and `results/eval/summary.json`.

- **Ten policies trained on the same budget:** AutoDR RT-206s1..s5 and
  No-DR RT-207s1..s5, 1500 iterations each, force observation
  (`obs_wrench_mode=force`, 28 channels).
- **Evaluation built and run, all four blocks PASS** on one code commit
  (`8299485d`): T-nominal RT-222, T-auswahl RT-223, T-final RT-224,
  T-grenze RT-225; G1–G3 checked per run, collector over all blocks OK.
- **K1 met:** on T-final AutoDR 0.970 vs No-DR 0.262, difference +70.8
  percentage points, Welch 95 % [+66.3, +75.3]. No-DR fails by timeout,
  not by force.
- **K2 and K3 met, both only just:** nominal difference −1.7 points (Welch
  upper bound +0.04 points; the bootstrap disagrees); boundary block
  `tilt_hi` upper bound +0.19 points. Final policy AutoDR seed 1 (D-190),
  98.2 % on T-final.
- **Figures F1–F10 and a one-command report** (`scripts/eval_report.py`,
  `docs/figures/eval/`); limits for the thesis text in
  `InBachelorErwähnen.md`.

**Next branch: `p6-TestAusserhalb`** (worktree `Thesis Workflow Phase 6/p6-TestAusserhalb`):
the out-of-distribution test at ×1.5 of the target range (D-055), a film strip
and a force trace — plan `docs/eval_extra_runs_plan.md`. It needs new code, so
it lives on its own branch; `p5-kraftsensor` keeps the evaluated state.

## Status v2 (2026-09-06) — SUPERSEDED, history only (e.g. the force abort is 30 N today, `episodes_meta.json`)

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
