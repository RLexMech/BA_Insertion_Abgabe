# RT-188 — expectation, written BEFORE the run (2026-09-12)

Written on the dev laptop by the main chat. `/rt-check` judges the log
against THIS file. **UNVERIFIED** until the training PC has run it.

## What this is

The **AutoDR smoke run** of `Laufplan_Phase5.md` step 10 ("Smoke, 30
iterations — boundaries move, fill rates, `autodr_<it>.json`"), on the wide
fixture, at **256 environments**, training seed **20** (probe seed,
`Laufplan_Phase5.md:9`). Log name `RT-188s20`.

Decided by the user 2026-09-12: work with 256 environments "to be safe" and
go straight to the first wide-range AutoDR run.

## Deviations from the Laufplan, named — not hidden

1. **Condition 1 (steps 5–9: No-DR budget run, seeds 1–5, eval, grid) has
   not run.** The Laufplan puts AutoDR after it. User decision.
2. **H_min is not measured** (step 4; RT-185 is only its 30 mm rung and has
   no `VERDICTS.md` line). Under `dr_mode=autodr` the start-height boundary
   opens upward from its centre, and that centre is
   `start_tip_above_entrance` = `RUNG0_START_TIP_ABOVE_ENTRANCE` = **0.030 m**
   (`insertion_tasks_cfg.py:1669`), marked `[TESTWERT]` in `autodr.DR_DIMS`
   ("lo = the centre H_min, not yet measured -- D-179 (4)"). Nothing refuses a
   `[TESTWERT]` at start (grep over `source/` and `scripts/rsl_rl/`: no
   raise/exit on it). So the run starts; its start-height lower edge is a
   placeholder, not H_min.
3. **256 instead of the 1024 every earlier AutoDR estimate assumed.** This
   slows the AutoDR clock — see "What 256 costs" below.

## Why the run can start at all — the static-DR guard, checked

`insertion_env.py:382-410` raises `ValueError` under `dr_mode='autodr'` if any
of `fixture_yaw_noise_rad`, `fixture_tilt_noise_rad`, `fixture_tilt_rad`,
`fixture_yaw_rad`, `start_lateral_offset` is non-zero, or if
`start_tip_above_entrance_low` differs from `start_tip_above_entrance`. With
the cfg defaults all five are 0.0 and both heights read 30.0 mm — read from
RT-187's two `demo_metrics.json` files, not assumed. `fixture_pos_noise_xy`
(default 0.005), the joint noise and the tilt azimuth are NOT in that set and
stay live by design (the guard's own message says so). **So no `env.*`
override is needed except `env.dr_mode=autodr`.** The two penalty flags of the
RT-178/RT-179 commands no longer exist and are NOT passed.

## What 30 iterations at 256 environments can and cannot show

From `autodr.py:222-249` (`BUFFER_M` 240, `P_BOUNDARY` 0.5, `DELTA_STEPS` 10)
and seven boundaries (`lat_r_hi`, `yaw_lo`, `yaw_hi`, `tilt_hi`,
`start_height_hi`, `friction_lo`, `friction_hi`), with the file's own formula
(`autodr.py:236-241`: episodes per iteration = envs × 16 / 256):

| envs | episodes / it | flags / boundary / it | it per buffer fill |
|------|---------------|-----------------------|--------------------|
| 256 | 16 | 1.143 | **210** |

30 iterations give about **34 flags per boundary of 240.** **No buffer can
fill, so no boundary can move in this smoke.** The Laufplan's "boundaries
move" is NOT testable at 256 in 30 iterations; it is dropped here, not
failed. This is an estimate from the formula; it assumes full-length
episodes, which holds while no episode succeeds.

## The command (training PC, PowerShell, conda env `env_isaaclab`)

Root: `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`

```
git pull --ff-only origin p5-robustheit
```

```
git merge-base --is-ancestor e7b85a2d HEAD; if ($?) { "instrument present" } else { "INSTRUMENT MISSING -- STOP" }
```

```
.\scripts\rt_log.ps1 RT-188s20 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 20 --max_iterations 30 env.dr_mode=autodr
```

```
Copy-Item logs\demo_metrics.json logs\RT-188_demo_metrics.json
```

```
Get-ChildItem logs\rsl_rl\ur5e_insertion -Directory | Sort-Object LastWriteTime | Select-Object -Last 1 | ForEach-Object { Get-ChildItem $_.FullName -File | Select-Object Name }
```

## PASS / FAIL lines, in order

**P1 — it starts.** No `ValueError` from the static-DR guard, no Traceback.
The startup report shows AutoDR active and prints the seven boundaries.

**P2 — the seven fill keys exist.** The log carries `dr/<boundary>_fill` for
all seven boundary names above, plus `dr/bounds_version`, `dr/all_at_max`,
`dr/flags_dropped` (`autodr.py:853-859`). Fewer than seven = FAIL (guard on
equality, not on non-emptiness).

**P3 — the fills are sane.** At the last iteration every `_fill` is > 0 and
< 240. Record all seven values beside the estimate of ~34. The estimate is a
reading, not a gate: a large, uniform deviation means the episode arithmetic
is wrong and must be reported, not tuned away.

**P4 — nothing moved, nothing dropped.** `dr/bounds_version` 0,
`dr/all_at_max` 0, `dr/flags_dropped` 0. A non-zero `bounds_version` after
~34 flags per boundary would be a bookkeeping fault.

**P5 — the sidecars are written.** The run folder holds `autodr_<it>.json`
next to each `model_<it>.pt` (`train.py:159-160`), at least for the first and
the last checkpoint.

**P6 — the instruments ride along.** `logs/RT-188_demo_metrics.json` carries
`dr_mode "autodr"`, a non-null `dr_state`, and `interpen_max_mm` with
`episodes > 0`. Read, do not judge: the policy is untrained.

**P7 — the log is complete.** Exit line present or its absence stated.

## What the MAIN run after this smoke still needs — open, not decided here

1. **The iteration cap** (`Laufplan_Phase5.md`, table "Open": user, later).
   A derived FLOOR, not a budget: every boundary needs `DELTA_STEPS` = 10
   advancing fills and then a fresh window of 2000 regular episodes (8 per
   iteration at 256). **10 × 210 + 250 = 2350 iterations, about 3.3 h at the
   measured ~5 s per iteration.** A cap below that makes
   `--stop-when-dr-max` unreachable by construction. Real runs need more,
   because a fill only advances at boundary success ≥ 0.80.
2. **Seeds.** `Laufplan_Phase5.md` step 11 says seeds 1–5 for the main run;
   this smoke uses probe seed 20.

## What 256 costs, from this session's own measurements

AutoDR's clock is buffer fills, and a fill counts episodes. RT-187 measured
~5 s per iteration at 256 and ~8 s at 512:

| envs | it per fill | wall clock per fill |
|------|-------------|---------------------|
| 256 | 210 | 17.5 min |
| 512 | 105 | 14.0 min |

256 costs about 25 % more wall-clock time per fill than 512. Recorded as the
price of the safety margin the user chose, not as an objection.

## Named risk — the from-zero finding

At width 0 every boundary episode starts at the centre: +30 mm, no tilt, no
yaw, no lateral offset, friction 0.14, plus the live 5 mm fixture noise. That
is the start RT-187 ran from zero for 100 iterations without a single
success. Until the policy reaches `advance_at` 0.80 on boundary episodes, a
full buffer RETREATS, clamped at zero, and nothing opens. RT-179 ended that
way (`bounds_version 0` after 1500 iterations) — a hint only: it ran on
`35a6e97`, without the penalty removal and without the noise model
(`docs/decisions_inbox.md`, rebuild-boundary entry). Not a reason to skip the
smoke; the reason the main run must be watched for `bounds_version` staying 0.

## What this run cannot answer

* Whether AutoDR opens anything — 30 iterations cannot fill a buffer.
* Whether the reward learns under AutoDR — untrained.
* Anything about H_min — the centre is a placeholder.

## Status

**UNVERIFIED.**
