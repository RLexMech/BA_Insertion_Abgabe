# RT-178 — expectation, written BEFORE the run (2026-09-10)

A SMOKE RUN, not a result. It buys the evidence the code has been claiming on
a laptop since step B5: that `dr_mode='autodr'` starts, that every reset maps
its draw onto the live band, and that the checkpoint sidecar is written. No
success rate is a criterion here, and none is read as one.

Clone `Phase5_Robustheit_v1`, branch `p5-robustheit`, code
`307f764100da93873a14295ba62fc3776a3485ad`. Nothing of B7 has ever run under
Isaac; the z clamp is a runtime path and is UNVERIFIED until this run.

## The one hypothesis

**The AutoDR provider comes up wired and reports itself honestly.** Every
line below is readable off the log or off `autodr_50.json`. If P1..P6 hold,
B5/B6/B7 are wired; if P7 holds as written, the loop is merely UNFED, not
broken.

## The budget, and what it cannot buy

`--max_iterations 60`, `save_interval = 50`, so `model_50.pt` and its sibling
`autodr_50.json` (`train.py:159-176`) both exist when the run ends.

**MEASURED BEFOREHAND, and it decides what this run can prove:** one boundary
buffer holds `BUFFER_M = 240` flags. Per iteration each env takes
`num_steps_per_env = 16` policy steps and an episode caps at 256 steps
(`episode_length_s = 256/60` at 60 Hz), so

    episodes      = 1024 * 16 * N / 256           = 64 * N
    per boundary  = episodes * p_boundary / 11    = 2.909 * N

| N   | episodes | flags per boundary | buffer 240 |
|-----|----------|--------------------|------------|
| 60  |     3840 |              174.5 | NOT full   |
| 85  |     5440 |              247.3 | full       |
| 100 |     6400 |              290.9 | full       |

So at 60 iterations **no boundary can complete a buffer**, `update()` returns
nothing, and there is no `[autodr]` line and no `bounds_version` bump. That is
arithmetic, not a forecast. The one way it changes: episodes that END EARLY
(force abort) raise the episode count, and a random policy may abort often. So
the count is a LOWER bound on episodes and `dr/<key>_fill` is the reading that
settles it, never the expectation here.

**`--max_iterations 100` would close the loop once** (one full buffer per
boundary, one `[autodr]` line per boundary) at a cost of about two thirds more
run time and a second sidecar `autodr_100.json`. Whether that is worth it is
the user's call; this file registers the 60 as commanded.

## Command (training PC, PowerShell, conda env `env_isaaclab`)

```
.\scripts\rt_log.ps1 RT-178 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max_iterations 60 env.dr_mode=autodr env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0
```

`env.dr_mode=autodr` is the ONLY override the DR side needs: the six centres
come from the task config. RT-177's `fixture_yaw_noise_rad`,
`fixture_tilt_noise_rad`, `start_lateral_offset`, `start_tip_above_entrance`
and `start_tip_above_entrance_low` are gone ON PURPOSE -- `insertion_env.py:361`
RAISES on that combination, because an AutoDR boundary already owns each of
them. The two reward-scale zeros are the standing condition of RT-171..RT-177
and travel unchanged. `env.osc_pos_clamp_m=0.07` is deliberately NOT carried
over: B7 split the box per axis, x/y is Factory's 0.05 and z is 0.10.

## PASS / FAIL lines, in order

**P1 — IT STARTS.** No `ValueError` naming `dr_mode='autodr' together with
static randomisation`. FAIL means a cfg default is non-zero that this mode
requires at zero, and the message names it.

**P2 — THE PROVIDER REPORTS ITSELF.** `provider: AutoDR, 11 boundaries,
bounds_version 0`. Eleven, not twelve: `tilt_lo` has no boundary
(one-sided). FAIL on any other count.

**P3 — THE BAND TABLE IS WIDTH 0 AND THE MAXIMA ARE THE REAL ONES.** The six
rows print `now [lo, hi]` and `max [lo, hi]`. Every `now` pair must be
lo == hi (AutoDR starts at width 0), and the `max` column must read

    lat_x        [-0.006000, +0.006000]
    lat_y        [-0.006000, +0.006000]
    yaw          [-0.104720, +0.104720]
    tilt         [+0.000000, +0.139626]
    start_height [+0.020000, +0.050000]
    friction     [+0.200000, +0.600000]

FAIL if a `now` pair has width, or if a `max` pair differs from `DR_DIMS`.

**P4 — THE WIRING LINE AND THE PROVENANCE LINES.** `ALL SIX QUANTITIES ARE
WIRED (Phase 5 step B5)` is present, and every dim whose `source` carries
`[TESTWERT]` prints a `[TESTWERT-check]` line. FAIL if the wiring line is
absent; a missing TESTWERT line means a placeholder lost its mark.

**P5 — THE SIDECAR EXISTS AND IS COMPLETE.**

> **CORRECTION 2026-09-10, after the run.** The key list below is WRONG and
> must not be copied into the next expectation file. `mode`, `bounds` and
> `fill` are the keys of `AutoDR.as_dict()` (`autodr.py:781`), the TELEMETRY
> dict that reaches `demo_metrics.json` as `dr_state`. The sidecar is written
> by `AutoDR.state_dict()` (`autodr.py:809`) and carries `format: "autodr-1"`,
> `keys`, `dims`, `steps`, `buf_n`, `buf_s`, `dropped`, `bounds_version`,
> `buffer_m`, `p_boundary`, `advance_at`, `retreat_at`. Those twelve are what
> `load_state_dict()` (`autodr.py:843-885`) actually demands, and RT-178's
> `autodr_50.json` carries all twelve. P5 is ERFUELLT. The list below was
> written from a sibling file instead of from the serialiser -- see
> `docs/reference/pruefregeln.md` L-06.
 After iteration 50,
`autodr_50.json` sits beside `model_50.pt` and carries `mode: "autodr"`,
`bounds_version`, `bounds`, `steps`, `fill`, `buffer_m: 240`,
`p_boundary: 0.5`, `advance_at: 0.8`, `retreat_at: 0.1`. FAIL if the file is
missing or a key is absent -- resume without it is refused, so a missing
sidecar ends the AutoDR line here.

**P6 — THE BUFFERS FILL, AND ROUGHLY AS COMPUTED.** `dr/<key>_fill` climbs
for all eleven keys. EXPECTED about **145** per boundary at iteration 50 and
about **175** at iteration 60, from the table above. Read as a MEASUREMENT,
not a threshold: a much higher number means episodes ended early (check the
abort rate), a much lower one means boundary sampling is not reaching the
envs and P2 lied.

**P7 — AND NOTHING MOVES, WHICH IS THE POINT.** `bounds_version` stays **0**,
no `[autodr]` line appears, `dr/<key>_hi` is flat. This is EXPECTED at this
budget and is not a defect. If an `[autodr]` line DOES appear, it is a
`clamped_zero` on a random policy (rate <= 0.10 with `steps = 0`, so no number
changes and no version bump) -- read the line, do not treat it as a surprise.
A `bounds_version` above 0 at 60 iterations means a buffer filled AND a bound
moved; then P6's episode count has to explain it before anything else is
believed.

**P8 — THE FRICTION READ-BACK.** The startup report prints the commanded
friction beside PhysX's own read-back for envs 0..3, after a reset. They must
agree. This is the per-reset friction claim of step B4 paying its first
evidence in a simulator.

**P9 — B7 PART 3 REPORTS ITSELF.** `demo_metrics.json` carries
`start_pose_solve_flags_dropped_envs` and `start_pose_solve_unconverged_resets`.
EXPECTED 0 or small. THE DIAGNOSTIC THAT MATTERS: dropped envs climbing while
`dr/<key>_fill` flattens is not a hard boundary -- it is a start pose the IK
cannot reach.

**P10 — THE Z CLAMP DOES NOT BLOW UP.** No NaN in the actions, no immediate
force abort on every env, `mean_max_depth_mm` finite. This is the first time
`osc_pos_clamp_z_m = 0.10` and the pocket-frame cone have ever run.

## What this run cannot answer

* Whether AutoDR EXPANDS. That needs a full buffer, i.e. about 85 iterations
  (table above), and a policy good enough to reach 0.80 at a boundary.
* Whether the policy learns anything. Random weights, 60 iterations, no
  criterion here reads a success rate.
* Whether the box and the cone are RIGHT in physics -- only that they run.
  The B7 arithmetic is pinned offline against `autodr.bounds_max()`; a
  simulator cannot check a containment rule, only survive it.
* Anything about the geometry transfer, which is a different task entirely.
