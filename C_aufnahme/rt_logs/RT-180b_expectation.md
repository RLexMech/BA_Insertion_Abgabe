# RT-180b — expectation, written BEFORE the run (2026-09-12)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

THE LATERAL COUNTER-PROOF OF RT-180, RE-RUN WITH ONE SETTLE STEP. RT-180
(`rt_logs/VERDICTS.md`, 2026-09-12 11:59:53) FAILED its lateral run on L3:
after the default 30 settle steps the tip read |y| 84.3 mm and depth
-50.1 mm instead of the commanded 191.889 mm / 34.0 mm. The mechanism was
read off the code, not the log: `_apply_osc` builds the target as the
CURRENT pose plus the action delta (`insertion_env.py:1522-1524`) and THEN
clamps it into the box around the entrance (`insertion_env.py:1542-1550`,
half-width `osc_pos_clamp_m = 0.08` m, `insertion_env_cfg.py:252`). A zero
action outside the box is therefore not "stay put"; it is "move to the box
edge". The probe's smallest LEGAL lateral offset
(`lateral_probe_min_y_m`, `check_seated_success.py:398-406`: 100.45 mm plus
half the part width) already lies outside the 80 mm box, so EVERY legal
`--lateral` value is pulled during the settle. The curve mode refuses such
offsets up front (`curve_lateral_error`, `check_seated_success.py:597-606`);
the lateral mode has no such guard.

**One settle step is the smallest number the script allows that still
judges the teleported pose.** `--settle-steps 0` would skip the loop
(`check_seated_success.py:2162`) and judge `obs` from `env.reset()`
(`:1964`), which is the HOME pose, not the teleport. The IK teleport writes
the joint state directly (`:2107`) and calls no `env.step`, so the
observation calls of this run are exactly two: the reset (count 1) and the
single settle step (count 2). Count 2 is a `report_at_steps` entry
(`insertion_env_cfg.py:796`; call site `insertion_env.py:1848`), so the
startup report prints INSIDE the very `_get_observations` call whose `obs`
the script then judges (`check_seated_success.py:2181`, `:2239-2247`,
`terminated_any` is `False` beside the pocket). **The report's pose line and
the JSON therefore describe the same buffer at the same instant.** That is
the identity this run adds.

No code change. Code on the training PC: `23a7aa0` (RT-180's `git:` line).
The laptop is at `dc91f96`, `origin/p5-robustheit` at `4725840`; the diff
`23a7aa0..dc91f96` touches only `HANDOFF-RL.md`, `docs/decisions_inbox.md`,
`rt_logs/VERDICTS.md` and `scripts/shorten_rt_log.py` (read today,
`git diff --stat`) — neither `insertion_env.py` nor
`check_seated_success.py`. **No pull is needed; the `git:` line must read
`23a7aa0`.** Markers, both must appear: `check_seated_success-2026-09-05c`
(`check_seated_success.py:97`), `insertion-osc-2026-09-03a`
(`insertion_env.py:73`).

## The one hypothesis

**With one settle step the tip still stands where the teleport put it — beside
the block at seat depth — so L3 holds, and L4/L5/L6 test D-157 on the OSC env
for the first time.** RT-112/RT-113 tested D-157 under PD control without a
clamp box (`HANDOFF-RL.md` § Open, 000); on this env D-157 has never been
exercised, because RT-180's tip was already 50 mm above the entrance when
the predicate was read.

## Command (training PC, PowerShell, conda env `env_isaaclab`)

```
.\scripts\rt_log.ps1 RT-180b python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --lateral --settle-steps 1 --out rt_logs/RT-180b_lateral.json
```

Own log NAME `RT-180b`, so `rt_log.ps1` writes a fresh `rt_logs\RT-180b.txt`
instead of appending a fifth run to `RT-180.txt`. Hand over the full log
and `rt_logs\RT-180b_lateral.json`.

Everything else stays at RT-180's defaults (`rt_logs/RT-180_expectation.md`,
table under "Command"): band 33.0-36.0 mm, commanded depth 34.0 mm,
commanded lateral offset 191.889 mm (computed, `check_seated_success.py:409-419`),
pocket wall 72.55 mm, `--solve-steps 120`, `--solve-tol-mm 0.05`. The ONLY
flag that differs from RT-180's run 3 is `--settle-steps 1` (default 30,
`check_seated_success.py:1377`).

## What the pose must read, and where the number comes from

RT-180's lateral run printed the count-2 report at log L629; inside it, after
exactly ONE settle step, the pose lines read (RT-180 log, via
`filter_rt_log.py`, 2026-09-12):

```
L756:   peg TIP rel. entrance: [-5.2e-05, 0.173227, -0.033972]  (env-local; pocket upright -- this is observation channels 12:15)
L764:   tip insertion depth:   +0.033972 m
```

Same code, same teleport, same single step (the script sets no seed of its
own; the env's `cfg.seed` is whatever the cfg carries): **the expected
reading is |y| ≈ 173.2 mm and depth ≈ 33.97 mm.** Whether it is the SAME to
the printed digits is a determinism reading, recorded, not judged.

**Read only, not interpreted (carried over from RT-180, still open):** the
IK converged to 0.0003 mm residual on a commanded y of 191.889 mm, yet the
count-2 line reads 173.227 mm — an 18.66 mm gap, ratio 1.108, no unit. This
run CANNOT tell whether one settle step already pulled the tip 18.66 mm
toward the box edge or whether the command refers to a different point than
`peg TIP`. Reading the pose at step 0 would need a print before the first
`env.step`, which is a code change and out of scope. The number is recorded
again; nothing below depends on it.

## PASS / FAIL lines, in order

The eight point names are the script's own (`judge_run`,
`check_seated_success.py`), copied from `rt_logs/RT-180/RT-180_lateral.json`
`points[]`, not paraphrased. Verdict line `LATERAL_CONTROL_HOLDS`; exit code
`0` iff the verdict holds (`check_seated_success.py:2391`, `:2401`).

1. **P1 the kinematic solve converged** — PASS: `solve_converged: true`,
   residual under 0.05 mm (RT-180 read 0.000309 mm). Unchanged by the
   settle count: the residual is taken after the IK, before the settle.
2. **L2 every env really stands OUTSIDE the pocket wall (|y| > 72.55 mm)** —
   PASS: `lateral_y_mm` all near 173.2 mm (NOT near the commanded 191.889 —
   see "Read only" above). FAIL: any |y| ≤ 72.55 mm.
3. **L3 the projected depth is INSIDE the 33.0-36.0 mm band (the hack pose,
   or nothing is exercised)** — PASS: `achieved_depth_mm` all near 33.97 mm.
   **This is the point RT-180 failed and the point this run exists for.**
   FAIL: any depth outside 33.0-36.0 mm; then L4-L6 prove nothing again, and
   the next step is a code question (a step-0 read or a lateral guard), not
   another settle count.
4. **L4 the success predicate does NOT fire beside the pocket (D-157)** —
   PASS: `in_region` all `false`, detail `0/4 in region`. **First real
   exercise of D-157 on the OSC env.** FAIL: any `true` — the predicate
   accepts the hack pose and RT-107's exploit is open.
5. **L5 the engaged bonus does NOT fire beside the pocket (D-157)** — PASS:
   `engaged` all `false`, `0/4 engaged`. FAIL: any `true`.
6. **L6 the depth metric books nothing beside the pocket** — PASS:
   `max_depth_metric_mm` all `0.0`, detail `booked: []`. In RT-180 this was
   trivially true (tip 50 mm ABOVE the entrance); here it is the real test.
7. **L7 the paid reward equals the D-109 formula on the printed numbers** —
   PASS: `reward` and `reward_expected` agree within 0.001 per env.
   `action_rate_norm` reads `0.0` (zero action after a reset that zeroed the
   previous-action buffer, `check_seated_success.py:2283`).
8. **L8 the force in FREE AIR is under 1.0 N (the gravity tare removed the
   tool's own weight)** — PASS: `force_n` all < 1.0 N and
   `force_n_raw_untared` near 8.08 N (RT-180 read 8.083 N at 30 steps; the
   raw wrench is a live articulation read, not smoothed). NOTE for the
   judge: `force_n` is the EMA `_force_smooth`, cleared to zero at reset and
   updated ONCE here with factor `ft_smoothing_factor = 0.25`
   (`insertion_env.py:1825-1827`, `insertion_env_cfg.py:1058`), so it reads
   0.25 × the tared raw value of this step. A number SMALLER than RT-180's
   0.04 N is the EMA, not a better tare. A FAIL with a value near 2.0 N
   (0.25 × 8.08) would mean the tare did not apply on the first step — that
   is a finding; a FAIL with a doubled raw value is a sign error.

**Pins, unchanged from RT-180 and read from the FILE, not the log prose:**
`fixture_pos_noise_xy_m`, `obs_noise_pocket_pos_std_m`, `force_obs_noise_std_n`,
`grasp_obs_offset_x_m` all `0.0` in `RT-180b_lateral.json`; the report
prints the OFF form `observation bias (D-182): OFF -- both sigmas are 0.0,
so resolve_obs_noise_model returned None and no model exists` and the zero
row `grasp belief error env0..3 (D-183, +-0.0 m ...): [0.0, 0.0, 0.0, 0.0]`
(opposite rules, `rt_logs/RT-180_expectation.md` § 3). A built-model bias
line or a non-zero grasp entry is a FAIL regardless of L2-L8.

**The identity this run adds (I1), judged, not merely recorded:** the
count-2 report's `peg TIP rel. entrance: [x, y, z]` line (the `y` and `-z`
in metres) must equal the JSON's `lateral_y_mm[0]` / `achieved_depth_mm[0]`
for env 0 to the printed precision (report prints 6 decimals in m, the JSON
full float in mm). Same `_get_observations` call, same buffer — a mismatch
means the judge reads a different quantity than the report, and every
"the observation IS the pose" statement of RT-180 loses its witness.

## What this run cannot answer

* **The 18.66 mm gap** between the commanded 191.889 mm and the count-2
  reading 173.227 mm. Recorded twice, explained by nothing. A step-0 read
  needs a code change.
* **Whether 30 settle steps would hold ANY lateral pose.** By construction
  they cannot for a legal `--lateral` value on this clamp (all legal offsets
  lie outside the 80 mm box). Whether the lateral mode should refuse
  out-of-box offsets like the curve mode does is a code decision, not a
  measurement; it is not made here.
* **Nothing about D-157 under motion.** One zero-action step beside the
  block. The predicate is read at rest at the hack pose, which is what
  D-157 specifies; a policy sliding along the wall is a different test.
* **Nothing about training, the reward over an episode, or the noise model
  when it is on** — as RT-180.
