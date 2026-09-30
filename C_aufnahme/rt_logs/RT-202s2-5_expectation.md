# RT-202s2..s5 -- Condition 1 (No-DR), seeds 2-5, written BEFORE the run (2026-09-14)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

## Context

RT-202 (`rt_logs/RT-202_expectation.md`) is seed 1 of Condition 1 (No-DR,
wide fixture): it registered the No-DR twin of RT-201's floor run and, per
`rt_logs/VERDICTS.md` (2026-09-14 13:51), reached its target at iteration
392 (correcting a dip cause noted in RT-201). One seed decides nothing on
its own (RT-202's own closing line). `Laufplan_Phase5.md`, "Rules for every
run", was REPLACED 2026-09-14 (user, after the supervisor meeting): the
fixed "all 5 seeds get the same iteration budget" rule is struck; every
seed of both conditions now stops on its own
`--stop-when-dr-max 0.995` (under No-DR with the floor: the two floor/height
keys), cap 3000 (`Laufplan_Phase5.md` lines 13-17). Condition 1 steps 5-7
are REPLACED the same day (`Laufplan_Phase5.md` lines 67-69): no shared
budget run, seeds 2-5 run as `RT-202s2..s5` with the RT-202 command and only
`--seed` changed. RT-202s2..s5 is that: the remaining four seeds of Condition 1,
run one after another because the training PC runs one job at a time.

## Marker- and code-stand

Branch `p5-kraftsensor`. RT-202 ran on `145eaed9` or later. This run file
does not itself pin a new SHA -- the command is byte-identical to RT-202's
except `--seed`, so whatever commit carried RT-202 carries RT-202s2..s5. Before
handing over the command, verify with `git -C <root> rev-parse HEAD` (or
`--short`) against the last commit this session pushed, and read
`[rt_log] git:` from each RT-202sN log against that SHA. If the training PC
is behind: `git pull` then `git rev-parse HEAD`, expected SHA = whatever
`origin/p5-kraftsensor` reads at hand-over time (not yet known at file-write
time -- fill in before running).

`SCRIPT_MARKER` / `CODE_MARKER`: `scripts/CLAUDE.md` records that since
2026-09-14 `rsl_rl/train.py` accepts `--stop-when-dr-max` under
`dr_mode=autodr` OR a start floor, and `[autodr-state]` prints the phase on
resume -- this is the marker RT-202 already ran under. Each RT-202sN log
must show the same `[rt_log] git:` line; a mismatch across the four seeds
invalidates comparison between them (RT-151b precedent, `rt_logs/VERDICTS.md`,
2026-09-05 14:51:11).

## The one question

**Do seeds 2-5 reproduce RT-202's outcome (floor reached H_min, then
`start_height_hi` opened to 0.120 m, fresh success >= 0.995 before the
3000-iteration cap) -- and at what iteration does each seed stop?**

RT-202 (seed 1) stopped at iteration 392. That number is the outcome of ONE
seed, not a prediction for seeds 2-5: per-seed stop iteration is a RESULT to
record, not a pass bar to hold seeds 2-5 against. The pass bar is the
outcome itself (target reached before the cap), read independently per seed.

## Design

Identical to RT-202 except `--seed`. `dr_mode=off`, floor -0.030 m, 5 rungs,
H_min 0.020 (RT-202 design line), 256 envs, obs `force`, noise ON (cfg
defaults), `env.autodr_stall_buffers=3`, `--stop-when-dr-max 0.995`, cap
3000, `--stop-check-every 50`. No mass randomization (Condition 1 = No-DR;
mass belongs to the DR conditions per the Laufplan, not checked further
here since RT-202's own design line already fixes noise ON / DR off for
this branch).

## Command

Run these four, one after another (training PC root
`C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`):

```powershell
.\scripts\rt_log.ps1 RT-202s2 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 2 --max_iterations 3000 --stop-check-every 50 --stop-when-dr-max 0.995 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-202s3 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 3 --max_iterations 3000 --stop-check-every 50 --stop-when-dr-max 0.995 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-202s4 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 4 --max_iterations 3000 --stop-check-every 50 --stop-when-dr-max 0.995 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-202s5 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 5 --max_iterations 3000 --stop-check-every 50 --stop-when-dr-max 0.995 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

Verify before each: `[rt_log] git:` line in the resulting log matches the
expected SHA (see "Marker- and code-stand").

## Points, per seed (RT-202's form)

Applied identically to RT-202s2, RT-202s3, RT-202s4, RT-202s5.

* **P1 setup**: `git:` = handed-over SHA; `dr_mode off`; `provider: AutoDR,
  2 boundaries`; band table start_height [-0.030, +0.020] max
  [+0.020, +0.120]; START FLOOR line; folder
  `..._RT-202sN_offset5mm_currOFF_start+20mm_floor-30mm_seedN` (seed number
  in place of RT-202's seed 1).
* **P2 the floor moves**: five `[autodr] start_height_floor ... expand`
  lines, the fifth `-- reached H_min, phase dr`; `dr/phase` 0 -> 1 at
  `bounds_version` 5. RT-201 needed 55 iterations, RT-202 (seed 1) reached
  this within its 392-iteration run -- record each seed's own iteration.
* **P3 the ceiling opens**: `[autodr] start_height_hi ... expand` lines,
  0.020 -> 0.120 in ten moves; `dr/bounds_version` ends at 15 if nothing
  ever shrinks.
* **P4 the six curves** as in RT-201/RT-202; `Policy/mean_noise_std` not
  below 0.2.
* **P5 success**: `dr/train_success_regular_fresh` >= 0.995 over 2000;
  `success_by_start_height_bin` 10 bins, the five above +20 mm filled.
* **P6 stop**: `[stop-criterion] target reached` with `all boundaries at
  max True`; `autodr_<it>.json` format `autodr-3`, 2 keys.
* **P7 physics**: `force_abort_rate`, `max_depth_max_mm` (> 37 = FINDING),
  interpen.
* **P8 stagnation**: hold lines and `dr/*_holds`; a `STALL` note is a
  review trigger, read `_last_rate` and the height bins before any change.

## Files to hand back per seed

Verified against code before listing (`scripts/rsl_rl/train.py`,
`insertion_env.py`, `scripts/rt_log.ps1`). Each RT-202sN run produces one
folder under `logs/rsl_rl/<experiment>/<timestamp>__RT-202sN_...`:

* **`params/env.yaml`**, **`params/agent.yaml`** -- written by
  `dump_yaml(os.path.join(log_dir, "params", "env.yaml"/"agent.yaml"), ...)`
  (`train.py:680-681`). Feeds: config provenance -- confirms `dr_mode`,
  `start_floor_m`, `autodr_stall_buffers`, seed and every other env/agent
  setting actually used, independent of the command line.
* **`scalars.csv`** -- written at the end of the run via
  `_exp.write_csv(pathlib.Path(log_dir) / "scalars.csv", _tags, _rows)`
  (`train.py:851-852`), a wide CSV of every TensorBoard scalar
  (`scripts/CLAUDE.md`, `export_tb_scalars.py` entry). Feeds: the six curves
  (P4), AutoDR boundary/hold tags, success/depth/force curves for
  `plot_run_curves.py`.
* **`autodr_<it>.json`** -- the AutoDR state sidecar, path built by
  `_autodr_sidecar_path` (`train.py:183-200`, e.g. `.../model_1500.pt` ->
  `.../autodr_1500.json`), written alongside each checkpoint save
  (`train.py:585-595`, `_save_with_autodr`). Feeds: AutoDR end state
  (`format`, `bounds_version`, boundaries, `phase`) for P1/P2/P3/P6/P8 and
  for any resume.
* **`demo_metrics.json`** -- written by `_write_metrics`
  (`insertion_env.py:3759`, path `pathlib.Path(log_dir) / "demo_metrics.json"`
  at `insertion_env.py:3769`, called from `insertion_env.py:3416`). Feeds:
  `success_by_start_height_bin` (P5), `force_abort_rate`, `max_depth_max_mm`,
  interpenetration (P7) -- the source for `plot_start_height_bins.py` and
  `plot_tilt_bins.py`.
* **the final `model_<it>.pt`** -- the rsl_rl checkpoint at the stop
  iteration (or the cap). Feeds: the eval checkpoint for the later 1000-
  episode table (Laufplan step 8, not built yet -- see below) and for any
  `play.py --trace-obs` diagnostic.
* **`RT-202sN.short.txt`** -- `rt_log.ps1` writes the full log to
  `rt_logs/RT-202sN.txt`, shortens it via `shorten_rt_log.py` to
  `rt_logs/RT-202sN.short.txt`, and (since 2026-09-13, `rt_log.ps1:135-150`)
  copies BOTH the full and short log into the one run folder created since
  the run started (guard: exactly one new folder, else a printed note, no
  copy). Feeds: the run narrative and `[rt_log] git:` marker check
  alongside the numeric artefacts.

On the laptop these land at `rt_logs/RT-202/RT-202sN_<name>` (one
subfolder per seed) for `params/env.yaml`, `params/agent.yaml`,
`autodr_<it>.json`, `demo_metrics.json`, `model_<it>.pt`,
`RT-202sN.short.txt`; `scalars.csv` additionally goes to
`docs/figures/RT-202sN_scalars.csv` (matching the existing
`RT-<N>_scalars.csv` naming used by `export_tb_scalars.py`,
`scripts/CLAUDE.md`).

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| all four seeds reach target before the cap | the floor+ceiling machine is not a seed-1 fluke on No-DR; the four stop iterations (beside RT-202's 392) are the spread the thesis reports | close Condition 1, move to the DR condition or the wrench question |
| a seed hits the 3000 cap without `all boundaries at max True` | that seed did not reach the target inside budget; read its height bins and `_last_rate` before ranking it against the other three | review whether it is a seed-specific stall or a shared bottleneck across seeds |
| a `STALL` note on any seed | differential diagnosis point 8 (per RT-202's P8); a review trigger, not itself a verdict | read `_last_rate` and the height bins for that seed before any change |
| `max_depth_max_mm` > 37 on any seed | FINDING per P7, independent of the target-reached question | read the interpenetration alongside it |
| `Policy/mean_noise_std` < 0.2 on any seed | differential diagnosis point 5 (exploration), a review trigger for that seed | -- |

## Was dieser Lauf nicht beantworten kann

* Die 1000-Episoden-Eval-Tabelle (Laufplan Schritt 8, Table-Seed 10) ist
  NICHT gebaut; dieser Lauf hängt nicht von ihr ab und liefert nur die
  vier Checkpoints, die sie später als Input braucht.
* Ob die vier Seeds untereinander konsistent sind (Varianz der
  Stop-Iteration, Konsistenz der Erfolgsraten) ist eine Auswertung NACH
  allen vier Läufen, nicht Teil dieser Erwartung pro Seed.
* Nichts zur DR-Bedingung (Condition 2) oder zur Wrench-Beobachtung
  (RT-195/196/203) -- dieser Lauf bleibt auf Condition 1, No-DR, force-obs.
* Ob RT-202s Iteration 392 und RT-202s2..s5 aus derselben Verteilung
  stammen, ist eine statistische Frage, die dieser Lauf allein nicht
  beantwortet (fünf Punkte, keine Signifikanzaussage).
