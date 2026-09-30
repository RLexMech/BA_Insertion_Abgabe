# RT-205 -- RT-201 with NO stop criterion, fixed 1500 iterations, written BEFORE the run (2026-09-14)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

## Context

RT-201 (seed 1, AutoDR, floor -0.030) reached `[stop-criterion] target
reached at iteration 850` with all seven boundaries at max and fresh success
0.9955 over 2000 regular episodes (`rt_logs/VERDICTS.md`, 2026-09-14 04:27
entry). Its own CSV and sidecar physically end earlier -- `docs/figures/RT-201_scalars.csv`
and the handed-over sidecar both stop at iteration 833, not 850; that gap
between the printed stop line and the last written row is named here, not
resolved. RT-202 (No-DR twin, same seed 1) ran number-for-number identical
to RT-201 up to iteration 189 (VERDICTS.md, RT-202 entry, BEFUND 1) --
established fact, not re-derived here.

The Laufplan's stop rule for the five-seed study is `--stop-when-dr-max
0.995`, cap 3000, for EVERY seed of both conditions (`Laufplan_Phase5.md:124`,
also `Laufplan_Phase5.md:15`). RT-205 is a deliberate, named exception to
that rule: it is a probe, not a study seed, and the study seeds
(RT-201s2..s5, RT-202s2..s5) keep `--stop-when-dr-max` unchanged. RT-205
uses seed 1 on purpose (user, 2026-09-14) so its trajectory up to RT-201's
stop can be read against RT-201's own CSV.

## The one hypothesis

**After the policy reaches the target that stops RT-201 at iteration 850,
the contact force keeps falling with more training -- so at iteration 1500
the force distribution (`demo_metrics.json` `force_norm_n`) sits below
RT-201's end values (p50 15.06, p95 30.68, p99 37.69, max 59.54, N,
`rt_logs/RT-201/RT-201_demo_metrics.json`).**

## Marker- and code-stand

Branch `p5-kraftsensor`. Laptop HEAD and `origin/p5-kraftsensor` HEAD are
both `e5cbd0d0` (checked: `git rev-parse HEAD` = `git rev-parse
origin/p5-kraftsensor` = `e5cbd0d004a0f8ba48313e58064d752cc4978fcc`, working
tree clean). This is the same commit RT-202s2..s5 and RT-201s2..s5 are
registered against (`e5cbd0d0` "Keep the D-169 force-abort limit for the
five-seed study and write down why" -- docs-only, `DECISIONS.md` /
`InBachelorErwähnen.md` / `docs/decisions_inbox.md`, no code file touched;
`git show --stat e5cbd0d0`). No pull needed if the training PC is already
on this commit; if `git -C <root> rev-parse HEAD` on the training PC prints
anything other than `e5cbd0d004a0f8ba48313e58064d752cc4978fcc`, pull first:

```powershell
git -C C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1 fetch origin p5-kraftsensor
git -C C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1 checkout p5-kraftsensor
git -C C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1 pull origin p5-kraftsensor
git -C C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1 rev-parse HEAD
```

Expected SHA after pull: `e5cbd0d004a0f8ba48313e58064d752cc4978fcc` (or a
later commit on `origin/p5-kraftsensor` -- verify with the same command
before handing the run over; a diverging SHA voids the run per the RT-151b
precedent, `rt_logs/VERDICTS.md`, 2026-09-05 14:51:11, verworfen als
UEBERHOLT wegen falschem Marker).

`SCRIPT_MARKER` / `CODE_MARKER`: `insertion_env.py:74` sets `CODE_MARKER =
"insertion-osc-2026-09-03a"`; the run prints it (`insertion_env.py:4406`,
`print(f"code marker: {CODE_MARKER}")`) and it also lands in
`demo_metrics.json` under `"code_marker"` (`insertion_env.py:4013`). This
is the same marker RT-201/RT-202/RT-203 already ran under (`scripts/CLAUDE.md`
records no CODE_MARKER change since 2026-09-03; the `SCRIPT_MARKER` entry
`scripts/CLAUDE.md` records is the 2026-09-14 `train.py` change that
accepts `--stop-when-dr-max` under `dr_mode=autodr` OR a start floor and
prints `[autodr-state]` on resume -- unchanged since RT-201). The `[rt_log]
git:` line in the resulting log must read `e5cbd0d0...`; a mismatch
invalidates the run before any point below is read.

## Verified about the stop mechanism (read, not assumed)

`scripts/rsl_rl/train.py:29` defines `--max_iterations` (default `None`,
overridden here to 1500 on the command line). `--stop-when-dr-max` and
`--stop-at-success-rate` (`train.py:34-58`) both default to `None`.
`train.py:783-784`:

```
if args_cli.stop_at_success_rate is None and _dr_bar is None:
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
```

Neither flag is passed in RT-205's command, so `_dr_bar` (`train.py:719`,
`= args_cli.stop_when_dr_max`) is `None` and this branch runs: ONE
`runner.learn()` call for the full 1500 iterations, no block loop, no
`_criterion_dr_max` check, no `[stop-criterion]` line at all. The run
either finishes all 1500 iterations or crashes; there is no early exit path
in this branch.

Checkpoint interval: `agents/rsl_rl_ppo_cfg.py:32`, `save_interval = 50` --
unchanged from every prior RT-20x run cited above. `runner.save` is wrapped
by `_save_with_autodr` (`train.py:593-595`) whenever `env.unwrapped._dr` is
not `None`, which it is here (`env.dr_mode=autodr`). The wrapper
(`train.py:205-236`) writes the checkpoint FIRST, then its sidecar, so
every `model_<it>.pt` this run writes -- every interval save AND the final
save at 1500 -- gets a sibling `autodr_<it>.json` (`_autodr_sidecar_path`,
`train.py:183-202`, pattern `model_<it>.pt -> autodr_<it>.json`). Expected
checkpoints: `model_50.pt` .. `model_1500.pt` in steps of 50 (rsl_rl 3.0.1
saves on `it % save_interval == 0`, per `train.py:64-68`'s own comment on
the block-loop indexing -- that comment describes the block-loop case;
RT-205 runs the single-call branch, so the plain rsl_rl schedule applies
without the block-loop's index-repeat quirk), each with its `autodr_<it>.json`
carrying `"format": "autodr-3"` (`env.autodr_stall_buffers=3` is set on the
command line, same sidecar version as RT-201/RT-202/RT-203).

## Command (PowerShell, training PC root
`C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`)

```powershell
.\scripts\rt_log.ps1 RT-205 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations 1500 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

## PASS/FAIL points

No `judge_run` exists for `train.py` runs (same convention as
RT-201/RT-202/RT-203: read by hand against `rt_logs/VERDICTS.md`). Read in
this order:

* **P1 setup**: `[rt_log] git:` = `e5cbd0d0...`; `code marker:
  insertion-osc-2026-09-03a`; `provider: AutoDR, 8 boundaries`; band table
  `start_height [-0.030, +0.020]`, others at centre; run folder tag
  `..._RT-205_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1`; no
  width-mismatch refusal, no Traceback, no NaN; NO `[stop-criterion]` line
  anywhere in the log (verified above -- its absence is expected, not a
  fault).
* **P2 the floor moves**: same five `[autodr] start_height_floor: rate ...
  -> expand` lines as RT-201, the fifth carrying `-- reached H_min, phase
  dr`, at the same iterations RT-201 used (it 3/13/22/37/55) if determinism
  holds -- RT-202 matched RT-201 number-for-number up to it 189, so a
  divergence before it 55 would itself be a finding.
* **P3 the seven open**: RT-201's own corrected timeline (VERDICTS.md
  KORREKTUR) is that `bounds_version` stays 5 until it 222/223, then the
  seven reach their ceilings by it 588-623. Read RT-205's `dr/*_fill` and
  `[autodr]` expand lines against those iterations, not a new guess.
* **P4 the six curves** (reward, surrogate, value, entropy; EV/KL not
  logged in this rsl_rl version). `Policy/mean_noise_std` must not
  collapse below 0.2 before the boundaries open (RT-189s1 died at it 215
  with std < 0.2; RT-201 stayed >= 0.76, minimum before the transition
  0.95).
* **P5 success**: `Episode/success_rate`, `dr/train_success_regular_fresh`,
  `success_by_start_height/lateral/yaw/tilt_bin` in `demo_metrics.json`.
  Compare the trajectory up to it 833 against `docs/figures/RT-201_scalars.csv`
  step by step; the FIRST differing step (if any) is a reading, not a
  fault -- determinism across two separate processes is not guaranteed and
  is not assumed here.
* **P6 no stop line, cap reached**: the run trains the full 1500 iterations
  in one `runner.learn()` call and exits 0; `model_1500.pt` and
  `autodr_1500.json` (`"format": "autodr-3"`) exist.
* **P7 physics**: `force_abort_rate`, `max_depth_max_mm` (RT-201 ended at
  36.0 mm, no `> 37` excursion), interpen max/p99 (RT-201: 3.4 mm / 0.94
  mm, an open, not-investigated finding -- read the same fields here, do
  not resolve it in this run).
* **P8 stagnation**: hold lines, `dr/<key>_last_rate`, `dr/<key>_holds`; a
  STALL note (n >= 3) is a review trigger, not a change (RT-201: 8 hold
  lines, max 1 hold per boundary, no STALL).
* **P9 the hypothesis instrument -- force at the end**: `demo_metrics.json`
  `force_norm_n` (p50/p95/p99/max/success_p95/over_f_max) at the end of
  the 1500-iteration run, read against RT-201's end values (p50 15.06, p95
  30.68, p99 37.69, max 59.54, success_p95 30.51, over_f_max 3,
  `rt_logs/RT-201/RT-201_demo_metrics.json`). This is instrument (A). Only
  ONE `demo_metrics.json` survives per run (it is rewritten periodically),
  so RT-205's file gives only the 1500-iteration snapshot, not a curve.

## Instrument (B): replay series over training time

`demo_metrics.json` is periodically overwritten and only the LAST one
survives, so instrument (A) alone cannot show force falling WITH training
-- it can only compare two single points (833 vs 1500). Instrument (B)
replays the SAVED checkpoints under one fixed, dr-off condition so each
point is read from its own fresh `demo_metrics.json`.

Checkpoints to replay: the saved checkpoint nearest RT-201's stop (833 ->
nearest multiple of 50 below is `model_800.pt`; RT-205 saves every 50
iterations per P6 above), `model_1000.pt`, `model_1250.pt`, and the last
(`model_1500.pt`). The run folder's TIMESTAMP is unknown until the run
happens -- this is the one allowed placeholder, named here:
`<RUN_FOLDER>` is the folder `rt_log.ps1` creates under
`logs/rsl_rl/<experiment_name>/`, printed at the start of the RT-205 log
and carrying the `RT-205` tag. Fill it in from the run log before pasting
these commands.

```powershell
python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --max-steps 2560 --checkpoint "logs\rsl_rl\<experiment_name>\<RUN_FOLDER>\model_800.pt" env.dr_mode=off env.start_floor_m=0.12 env.start_tip_above_entrance=0.12 env.start_tip_above_entrance_low=0.02 env.start_lateral_offset=0.03 env.fixture_yaw_noise_rad=0.08726646259971647 env.fixture_tilt_noise_rad=0.17453292519943295
```

```powershell
python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --max-steps 2560 --checkpoint "logs\rsl_rl\<experiment_name>\<RUN_FOLDER>\model_1000.pt" env.dr_mode=off env.start_floor_m=0.12 env.start_tip_above_entrance=0.12 env.start_tip_above_entrance_low=0.02 env.start_lateral_offset=0.03 env.fixture_yaw_noise_rad=0.08726646259971647 env.fixture_tilt_noise_rad=0.17453292519943295
```

```powershell
python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --max-steps 2560 --checkpoint "logs\rsl_rl\<experiment_name>\<RUN_FOLDER>\model_1250.pt" env.dr_mode=off env.start_floor_m=0.12 env.start_tip_above_entrance=0.12 env.start_tip_above_entrance_low=0.02 env.start_lateral_offset=0.03 env.fixture_yaw_noise_rad=0.08726646259971647 env.fixture_tilt_noise_rad=0.17453292519943295
```

```powershell
python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --headless --num_envs 256 --max-steps 2560 --checkpoint "logs\rsl_rl\<experiment_name>\<RUN_FOLDER>\model_1500.pt" env.dr_mode=off env.start_floor_m=0.12 env.start_tip_above_entrance=0.12 env.start_tip_above_entrance_low=0.02 env.start_lateral_offset=0.03 env.fixture_yaw_noise_rad=0.08726646259971647 env.fixture_tilt_noise_rad=0.17453292519943295
```

All four hydra overrides above are field names read and confirmed in
`insertion_env_cfg.py`: `start_floor_m` (:481, hydra form `env.start_floor_m=`),
`start_tip_above_entrance` (:406), `start_tip_above_entrance_low` (:447),
`start_lateral_offset` (:551, a RADIUS, refused if negative), the yaw/tilt
noise fields (:735, :775, both default 0.0, opt-in via hydra). The
fixture-yaw and fixture-tilt values above (0.08726646259971647 rad = 5 deg,
0.17453292519943295 rad = 10 deg) are the values already used in this
worktree's own play.py override convention (radian forms of round-degree
figures) -- they are named here as the FULL-RANGE scatter so every
checkpoint replay sees the same distribution; if a different scatter is
wanted, that is a separate decision, not made here. `dr_mode=off` does NOT
touch friction: `CONTACT_FRICTION = 0.14` (`insertion_tasks_cfg.py:1396`)
is a task-geometry constant, not an AutoDR-managed quantity, so every
replay runs at friction 0.14 regardless of `dr_mode` -- named so it is not
silently assumed to vary.

**Replay-directory collision, read from `play.py`:** `log_dir =
os.path.dirname(resume_path)` (`play.py:364`) -- the CHECKPOINT's own
directory, which is the SAME training-run folder for every one of
`model_800.pt` / `model_1000.pt` / `model_1250.pt` / `model_1500.pt`.
`replay_dirname` (`play.py:394-403`) is built ONLY from `_h =
env_cfg.start_tip_above_entrance` and the `--fixture-offset` /
`--fixture-tilt` / `--fixture-yaw` CLI args -- it does NOT read the
checkpoint's iteration number at all. With the SAME `env.start_tip_above_entrance=0.12`
on every one of the four commands above, `replay_dirname` is
`replay_h+120mm` every time, and `env_cfg.log_dir = os.path.join(log_dir,
replay_dirname)` (`play.py:404`) is therefore the IDENTICAL path for all
four replays. **This collides**: the second replay's `demo_metrics.json`
overwrites the first's, the third overwrites the second, and so on --
`play.py`'s own comment at :380-383 names exactly this failure mode for
"successive probe runs on the same checkpoint" but does not cover
successive checkpoints of the same run at the same probe pose, which is
RT-205's case. **The user must copy `demo_metrics.json` out of
`<RUN_FOLDER>\replay_h+120mm\` (to e.g. `rt_logs/RT-205/RT-205_replay_800.json`,
`..._1000.json`, `..._1250.json`, `..._1500.json`) immediately after each
replay, before running the next one.**

## Pins (from `demo_metrics.json`, field names as the run writes them)

* `dr_mode`: `"autodr"` -- command-line override, unchanged from RT-201.
* `start_floor_m`: `-0.030` -- command-line override, unchanged from
  RT-201.
* `autodr_stall_buffers`: `3` -- command-line override, unchanged from
  RT-201.
* `fixture_pos_noise_xy_m` / `obs_noise_pocket_pos_std_m`: cfg default
  0.0025 m (unchanged from RT-201, read back from the JSON, not assumed).
* `force_obs_noise_std_n`: cfg default 3.5 N (unchanged from RT-201).
* `grasp_obs_offset_x_m` / grasp noise: cfg default 0.003 m (unchanged
  from RT-201).
* `code_marker`: `"insertion-osc-2026-09-03a"` (`insertion_env.py:4013`,
  `:74`) -- must match every prior RT-20x run cited above.
* `force_norm_n`: the P9 instrument -- p50/p95/p99/max/success_p95/over_f_max,
  read at the end of the 1500-iteration run, per `_force_stats`
  (`insertion_env.py:3594-3633`) over the trailing window (`_recent_force_norm`,
  `insertion_env.py:1290`, sized by `metrics_window_iterations`,
  `insertion_env.py:1194-1195` -- not a fixed 2000; read the window size
  this run actually used off the same JSON rather than assuming RT-201's).

## Pre-registered reading

The force falling further after it 850 is read DESCRIPTIVELY, with ONE
seed -- no bar is invented for "falls" (L-03, `docs/reference/pruefregeln.md`).
Report the p50/p95/p99/max at 1500 (instrument A) and the four replay
points at ~800/1000/1250/1500 (instrument B) together with the ratio
against RT-201's own end values, not against a threshold decided here.
Watch alongside the force reading:

* `Episode/success_rate` stays >= 0.995 after it 833, or degrades;
* `Policy/mean_noise_std` / entropy trend after it 833 (does exploration
  keep shrinking, or has it already bottomed out at RT-201's stop);
* `dr.all_at_max()` stays `True` for the whole 833-1500 stretch (a
  regression here would mean a boundary moved back, which nothing in this
  design should trigger);
* `Episode/force_abort_rate`;
* whether `max_depth_max_mm` crosses 37 mm at any point after it 833 (the
  RT-193/RT-195/RT-196 under-fixture path, RT-193 it 200, an open finding
  -- RT-201 itself stayed clean at 36.0 mm).

## What this run cannot answer

* Whether the force actually falls WITH training or the two instruments
  (A: one end snapshot; B: four replay points, each with PPO exploration
  present or absent depending on `play.py`'s own sampling mode -- not
  checked here) merely show sampling noise across four points. One seed,
  no repeat.
* Whether replay (B)'s numbers are comparable to RT-201's own training-time
  numbers at all: RT-201's `force_norm_n` is measured DURING training
  under the moving AutoDR distribution and PPO exploration noise; RT-205's
  replay points are measured under a FIXED, dr-off, full-range scatter with
  whatever noise mode `play.py` applies. This is a deliberate, named
  choice (one fixed condition so every checkpoint sees the same
  distribution), not an attempt to reproduce RT-201's own training
  distribution.
* Why RT-201's printed stop line says iteration 850 while its CSV and
  sidecar physically end at 833 -- named above, not resolved by this run.
* Whether a boundary can silently regress after `all_at_max()` (D-110-style
  forgetting) over a LONGER horizon than 1500 iterations -- this run caps
  at 1500 by design and says nothing about iteration 1501 onward.
* Any general claim about wrench-mode, torque noise, or the five-seed
  study's own seeds -- RT-205 is `force`-mode, seed 1, a probe outside the
  study's own stop-rule convention (Laufplan_Phase5.md:124), not a study
  seed and not a repeat of RT-203's wrench question.
* Whether the replay-directory collision named above (P9 / instrument B)
  is fixed by a future `play.py` change -- as read today the four replays
  overwrite each other's `demo_metrics.json` unless copied out by hand
  between runs; this file names the workaround, it does not change the
  script.
