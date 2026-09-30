# RT-157 — expectation, written BEFORE the run

Written 2026-09-05 on the dev laptop, after RT-156 was judged. `/rt-check`
judges the logs against THIS file and nothing else.

## The one hypothesis

**"The RT-156 policy keeps inserting from start heights it never trained
on, and there is a height at which it stops."** RT-156 trained on starts
sampled uniform in [−30, +30] mm and reached 98.1 % in its top bin
(+20…+30 mm, `docs/figures/RT-156_demo_metrics.json`). Nothing above
+30 mm was ever seen. This run replays ONE checkpoint at four FIXED
heights — +30 (in distribution, the anchor), +40, +50, +60 mm — and reads
where the success rate breaks.

Four runs, not one. The env's start-height table has fixed bin edges that
stop at +30 mm (`insertion_env.py:677`) and its last bin catches everything
at or above the last edge (`insertion_env.py:2056`), so a single sampled
run over [+40, +60] mm would put every episode into one bin LABELLED
"+20…+30 mm". The height must therefore be fixed per run, and the four
runs feed one figure.

## The one deviation from the trained controller, named

`osc_pos_clamp_m` goes from 0.05 m to **0.07 m** for all four runs.

Why it must move: the clamp keeps the tool point inside an axis-aligned
box of that half-width around the pocket entrance, component-wise, z
included (`insertion_env.py:1030` → `insertion_math.py:1339`). At 0.05 m a
start at +60 mm is OUTSIDE the box: the first `_apply_action` clamps the
target to +50 mm and the controller drags the part down before the policy
acts. RT-151b measured exactly this on the lateral axis (commanded 60 mm,
reached 52.454 mm; `rt_logs/VERDICTS.md` 2026-09-05 15:07:33), and the
follow-up entry there states the box clamps height the same way —
NOT YET MEASURED for z, so this run is also its first reading.

Why 0.07 and not 0.08: the smallest round value that contains the +60 mm
start and the −36 mm seat with margin, so the LATERAL loosening (±50 →
±70 mm) that comes with it is as small as possible. The number is MINE,
chosen for this probe; it is not derived and not a decision.

Accepted cost, stated: the plant is not bit-identical to the one the
policy trained under. The clamp is not in the observation
(`insertion_math.OBS_SLICES`), so the policy is not told about it, but its
learned behaviour was shaped in a world where the target got clamped at
±50 mm. Whether the z clamp ever bound during training is NOT MEASURED.
The +30 mm run carries the same 0.07 m, so all four points share one
instrument; the +30 mm point is therefore NOT directly comparable to
RT-156's top bin, and any difference between them is this deviation plus
the fixed-versus-sampled start.

## The episode budget

The run length is set in STEPS, not episodes, and `--max-steps` is what
stops `play.py` (added 2026-09-06; before that only `--trace-obs` or
`--video` ever ended the loop).

The env writes `demo_metrics.json` every `max(50, window // 8)` finished
episodes, and `window = max(2000, …)` has a hard floor
(`insertion_env.py:646`, `:732`), so the derived interval is 250
episodes. At a SMALL env count that binds hard: a first attempt at 100
envs × 768 steps needed those 768 steps only to force 250 episodes out of
a failing height, and handed the fast height 3700 episodes it did not
need. A large env count removes the problem instead of working around
it.

The run is therefore sized so that ONE pass gives one attempt per env:
**1024 envs × 288 steps**. 1024 is the user's number and the point of it —
1024 parallel attempts, resolved in a single sweep, the same env count
RT-156 trained at.

* **failing height** (every episode runs the full 256-step cap): exactly
  **1024 finished episodes**, one per env. This is the height the probe
  exists to measure, and it is the one that gets the full sample.
* **fast height**: envs reset and start again, so roughly 8400 episodes
  are finished; the trailing window keeps the last 2000 and the figure
  prints `n` under the bar.

288 steps, not 256: the cap is 256 control steps, and `play.py` breaks on
the step counter (`scripts/rsl_rl/play.py:476-478`), so a break at exactly
256 risks cutting the episode-end bookkeeping of the very episodes being
counted. 32 steps of margin cost nothing and remove the edge.

The dump interval stays the derived 250 and is NOT overridden. A
`metrics_dump_every` cfg field was tried on 2026-09-06 and removed the
same day: declared `int | None = None`, it made every hydra override die
on the type check (RT-157h30 run 2, exit 1, PROBLEMS.md 2026-09-06). It
is not needed at this size anyway — 1024 episodes finish at once.

No trace. `--trace-obs` was the only self-stop `play.py` had, so a probe
that wants nothing but `demo_metrics.json` had to write a trace nobody
reads: 1024 × 288 × 15 numbers, roughly 84 MB per run and 336 MB for the
four. `--max-steps N` now stops the loop on its own, counted for every
step, and the four runs write no trace at all.

## Code changed for this run (laptop, offline, all three checked)

1. `scripts/rsl_rl/play.py` — the replay directory name now carries the
   start height (`replay_h+40mm`). Without it the four runs overwrite each
   other's `demo_metrics.json` inside one `replay/`, which is the exact
   failure that file's own comment warns about. Nothing else in the repo
   reads that directory name (checked 2026-09-05).
2. `scripts/plot_probe_start_heights.py` — NEW. Draws one bar per
   fixed-height run. `--self-test` passes 6/6; the mutation
   (`!=` → `>` in the equality guard) breaks it, exit 1.
3. `scripts/rsl_rl/play.py` — `--max-steps N` stops the loop and exits.
   Before it, only `--trace-obs` or `--video` ever ended the run, so a
   probe that wants nothing but `demo_metrics.json` had to write ~84 MB
   of trace per run. Default `None` keeps the old behaviour.

## Commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: the hash in the handover message.

Four runs, one per height. `<CKPT>` is the one checkpoint, identical in
all four:

```
C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2\logs\rsl_rl\ur5e_insertion\09-05_19-45-03_offset5mm_currOFF_start-30..+30mm_seed42\model_250.pt
```

Then, per height h ∈ {0.030, 0.040, 0.050, 0.060}:

```
.\scripts\rt_log.ps1 RT-157h30 python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --checkpoint "<CKPT>" --max-steps 288 env.start_tip_above_entrance=0.030 env.start_tip_above_entrance_low=0.030 env.osc_pos_clamp_m=0.07
```

…and the same three more times with `0.040`, `0.050`, `0.060` and the
matching `RT-157h40/50/60` names.

Run time: UNKNOWN. RT-156p did 512 steps × 8 envs; this is 288 × 1024, a
factor of 72 in env-steps. Read the wall time off the first run before
starting the other three.

## Points, each PASS/FAIL on its own

1. **P1 — all four start and finish.** Exit 0, no traceback, no `nan`,
   none of the four start-range refusals fires, `[play] stopping after 288
   steps (--max-steps)` in each log.
2. **P2 — the flags are on disk, per run.** Startup prints
   `rung-0 start pose: tool point +XX.000 mm above the stage-2 opening
   plane` with XX = the commanded height, and NOT the "SAMPLED per
   episode" line (a fixed start must read as fixed). `demo_metrics.json`:
   `start_tip_above_entrance_mm` = `start_tip_above_entrance_low_mm` = the
   height; `rl_placeholders.osc_pos_clamp_m` = 0.07; `num_envs` 1024;
   `control_mode` osc.
3. **P3 — four separate metrics files exist.** One
   `replay_h+30mm / _h+40mm / _h+50mm / _h+60mm` directory per run, each
   with its own `demo_metrics.json`. If two runs share a directory the
   whole probe is void (RT-107 class).
4. **P4 — the start pose is actually reached.**
   `start_pose_solve_unconverged_resets` 0 and
   `start_pose_solve_worst_residual_mm` < 0.05 in ALL FOUR. A failure here
   at +60 mm would mean the IK, not the policy, is the limit, and it stops
   the reading of 6–9.
5. **P5 — D-157 tripwire.** `success_depth_invariant_violations` 0 / 0 in
   all four, read BEFORE any success rate.
6. **P6 — THE ANCHOR.** +30 mm success rate **≥ 0.80**. This is the
   in-distribution height. Below 0.80 the probe is not measuring
   generalisation but the clamp change or the fixed-versus-sampled start,
   and points 7–8 must not be read as generalisation.
   * The 0.80 line is MINE: RT-156's top bin read 0.981 under the 0.05 m
     clamp and a sampled start, and I allow 18 points for the two named
     differences. Order-of-magnitude line, not a measured threshold.
7. **P7 — THE HYPOTHESIS.** The success rate is **monotonically
   non-increasing** over +30 → +40 → +50 → +60 mm, within 0.05 per step.
   A rise of more than 0.05 with height is a finding against the
   instrument, not a better policy, and must be reported as such.
8. **P8 — where it breaks, READ not predicted.** The first height whose
   success rate falls below 0.50, and the value at +60 mm. This is the
   branch variable and has no PASS line:
   * all four ≥ 0.50 → the policy generalises past the clamp radius; the
     next question is how far, not whether.
   * a clean drop at one height → that height is the reach limit of the
     learned behaviour; report it with the +60 value beside it.
   * +40 already ≈ 0 → the policy learned a start-height-specific
     descent, and the RT-156 result does not transfer at all.
9. **P9 — force under the raised clamp.** `force_norm_n` p95 per run,
   read against RT-156's 27.269 N. A p95 that GROWS with start height is
   the finding D-166/D-167 needs (the `Lambda·kp·Delta` ceiling of 17.6 N
   already failed in RT-156). `force_abort_rate` expected 0.0 in all four;
   an abort at 300 N would be new.
10. **P10 — the clamp is no longer the wall.** `stage1_lateral_y_mm` max
    per run. Under 0.05 m the lateral values piled at ~50 mm (RT-148b:
    52.56 mm at iteration 500, VERDICTS 2026-09-04). At 0.07 m a pile at
    ~70 mm would say the policy is again riding the box; a max well under
    70 mm says the box is not what limits it.

## What would make each point wrong

1. → crash or flag typo. Fix, rerun.
2. → a flag not applied (Hydra float without the decimal point,
   PROBLEMS.md RT-139b). Rerun; the reading is void until it prints.
3. → the `play.py` change did not land. The four numbers would then all
   come from the LAST run. Void.
4. → IK or reach finding, not a policy finding.
6. → the anchor fell: the probe measures the deviation, not the height.
   Everything after it is uninterpretable.
7. → a non-monotone series points at the instrument: fixture noise, too
   few episodes at one height, or a metrics file from the wrong run.

## Read, NOT expected

`mean_max_depth_mm` and `mean_success_episode_steps` per height (a higher
start must cost more steps); `episodes` per run; wall time of the first
run.

## What this run cannot answer

* Whether the 0.07 m clamp changes the behaviour at +30 mm — the
  comparison to RT-156's top bin confounds clamp and start distribution.
  Separating them needs a fifth run at 0.05 m, which is NOT part of this.
* Anything about the real fixture: the pocket in simulation is wider than
  the rig (play 1.60 mm lengthways / 0.59 mm across against 0.9 / 0.4 mm,
  D-087).
* Whether a policy TRAINED with higher starts would do better. This
  replays one checkpoint; it does not train.
