# RT-180 — expectation, written BEFORE the run (2026-09-12)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

**(Erwartung korrigiert 2026-09-12, vor dem Urteil.)** The commit name and
every line number that moved with it were corrected BEFORE any log was read.
The verdict therefore carries `(Erwartung korrigiert)`, the mark
`.claude/skills/rt-check/SKILL.md` step 4b prescribes. What changed: the SHA
(next paragraph), the shifted line numbers, and the claim in
§ `2. The four scatter fields in the metrics dump` about the curve dump, which
was true of `9f37a07` and false of the commit that ran.

THE TELEPORT SUCCESS IDENTITY TEST, step 2 of the strict order in `CLAUDE.md`
§ Code, and the run `Laufplan_Phase5.md` § Order step 4 names ("Net repair ...
teleport test before and after"). It is the FIRST run of any kind to reach the
training PC after the Phase-5 build, and the first since the observation noise
model (D-182) and the grasp belief error (D-183) went into the env.

Clone `Phase5_Robustheit_v1`, branch `p5-robustheit`, code
`origin/p5-robustheit` = **`23a7aa0`** — the commit the operator pulled and
ran. This file first named `9f37a07`, the commit BEFORE the pin and dump
repairs. Two commits landed after `23a7aa0` and neither touches the env or
this script: `git diff --stat 23a7aa0..4725840 -- source/ scripts/` reports
exactly one file, `scripts/shorten_rt_log.py` (45 insertions, 10 deletions).
So `23a7aa0` is the code that ran. The SHA correction moves line numbers:
`git diff 9f37a07..23a7aa0 --stat` gives `scripts/check_seated_success.py |
12 +` (12 lines inserted after old line 1704, `"mode": "reward_curve",` — so
every cited line from old 1705 on moves **+12**) and `insertion_env.py |
10 +-` (a 2-line comment replaced by 8 at old 1729-1730 — so every cited line
from old 1731 on moves **+6**). Both shifts were checked line by line against
`git show 23a7aa0:<path>`, not only by arithmetic. Script marker
`check_seated_success-2026-09-05c` (`scripts/check_seated_success.py:97`), env
marker `insertion-osc-2026-09-03a` (`insertion_env.py:73`). Both must appear in
the log; if either differs, the training PC is not on this commit and nothing
below is judged.

## The one hypothesis

**With the three Phase-5 scatter sources pinned to zero, the env still
reproduces the teleport success identity — and the pins actually hold, so the
observation IS the pose.** Everything below is readable off the log or off the
four JSON files. No success rate, no training claim, and no claim about the
reward over an episode is made here.

## Command (training PC, PowerShell, conda env `env_isaaclab`)

```
.\scripts\rt_log.ps1 RT-180 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --out rt_logs/RT-180_seated.json
.\scripts\rt_log.ps1 RT-180 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --shallow --out rt_logs/RT-180_shallow.json
.\scripts\rt_log.ps1 RT-180 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --lateral --out rt_logs/RT-180_lateral.json
.\scripts\rt_log.ps1 RT-180 python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 30.0 5.0 --out rt_logs/RT-180_curve.json
```

Four runs, four JSON files, one log NAME — `rt_log.ps1` appends the same name
into one file, so all four land in the RT-180 log in this order.

The settings none of the four commands passes, and what they therefore are.
Every value is read from source; none is typed on the command line.

| setting | value | where it comes from |
|---------|-------|---------------------|
| success band | 33.0 – 36.0 mm | `insertion_env_cfg.py:1228-1229` -> `insertion_tasks_cfg.py:884` = `POCKET_SEAT_DEPTH` (`:868`, = `-POCKET_FLOOR_Z`, `:696` = -0.036 m) minus `D106_DEPTH_BAND` (`:878` = 0.003 m) |
| commanded seated depth | 34.0 mm | `--depth-mm` default, `check_seated_success.py:1273` |
| commanded shallow depth | 15.0 mm | hard-wired for `--shallow`, `check_seated_success.py:1815` |
| commanded lateral offset | 191.889 mm | computed, not typed: `lateral_probe_default_y_m` (`check_seated_success.py:409-419`) = `POCKET_LOCAL_Y_RANGE[1]` 0.10045 m (`insertion_tasks_cfg.py:699`) + half the `PART_BBOX_M` footprint diagonal (`:1497`) + `LATERAL_PROBE_MARGIN_M` 0.005 m (`check_seated_success.py:395`) |
| pocket wall, long axis | 72.55 mm | `POCKET_WALL_Y` = 0.07255 m (`insertion_tasks_cfg.py:695`), x1000 at `check_seated_success.py:1834` |
| `--solve-steps` | 120 | `check_seated_success.py:1375` |
| `--settle-steps` | 30 | `check_seated_success.py:1377` |
| `--solve-tol-mm` | 0.05 mm | `check_seated_success.py:1380` — the IK gate AND the curve's `pose_drift_mm` gate |
| `--curve-settle-steps` | 4 | `check_seated_success.py:1353` |
| `sdf_slack_mm` (P7) | 0.5 mm | `check_seated_success.py:193` |
| `free_air_force_tol_n` (L8) | 1.0 N | `check_seated_success.py:200` |

## PASS / FAIL lines, in order

The names below are the script's own (`judge_run`,
`check_seated_success.py:178-369`); the wording is copied, not paraphrased.
Every mode prints `P1` first, then its own family.

### Run 1 — seated (no flag). Verdict line `SEATED_IDENTITY_HOLDS`.

1. **P1 the kinematic solve converged** — PASS: the printed solve residual is
   under 0.05 mm. FAIL: an unconverged teleport is not a seated state and, in
   the script's own words, "nothing below is judged from it".
2. **P2 every achieved depth is inside the 33.0-36.0 mm band** — PASS: all four
   envs near the commanded 34.0 mm. FAIL lists the envs outside.
3. **P3 the success predicate fires in every env** — PASS: `in_region` 4/4.
   FAIL: the D-106 predicate does not fire at a physically seated pose, i.e.
   the depth gate or the SAPU interpenetration gate is wrong.
4. **P4 the episode TERMINATES at the seat (the success exit)** — PASS:
   `terminated_any` True AND `success_term_any` True; the log carries a
   `SUCCESS TERMINATION at settle step <i>` line with the payout multiplier.
   FAIL: the seated state is not absorbing.
5. **P5 the exit is the SUCCESS, not the force abort** — PASS: `force_abort_any`
   False. FAIL: P4 alone would have passed on an abort, which is why both
   points exist.
6. **P6 the engaged bonus fires (depth past 30 % of the seat)** — PASS: 4/4
   engaged.
7. **P7 the sdf mean-outside stays under the rigid-shift bound (band_max -
   depth + 0.5 mm)** — PASS: per env, `sdf <= (36.0 - depth) + 0.5` mm. A FAIL
   also falsifies the NAMED ASSUMPTION in `_get_dones` that the fixture OBJ's
   frame is the entrance-anchored pocket frame.
8. **P8 the paid reward equals the D-109 formula on the printed numbers** —
   PASS: every env within the absolute 1e-3 bar, against a number in the
   hundreds (the payout dominates the line). FAIL prints
   `env i: paid X vs formula Y`.

### Run 2 — `--shallow` (15 mm). Verdict line `SHALLOW_CONTROL_HOLDS`.

1. **P1 the kinematic solve converged** — as above.
2. **S2 every achieved depth stays below the band (33.0 mm)** — PASS: all four
   near 15 mm. This is the paid counter-proof that the band is a LOWER bound
   as well as an upper one.
3. **S3 the success predicate does NOT fire** — PASS: `in_region` 0/4.
4. **S4 the episode does not terminate** — PASS: `terminated_any` False. FAIL
   means either that the success termination leaked past the band, or that a
   force abort fired in a pose that carries no contact load.
5. **S5 the reward stays below `reward_peak - 1.0` (no success bonus outside
   the band)** — the ceiling is `1.0 + w_engaged + w_success - 1.0`, computed
   from THIS run's cfg at `check_seated_success.py:1930` and printed inside
   the point's own name. **The number is unknown for this setup until the log
   prints it.** PASS: no env at or above it.
6. **S6 the paid reward equals the D-109 formula on the printed numbers** — as
   P8.

### Run 3 — `--lateral` (seated depth, beside the pocket). Verdict line `LATERAL_CONTROL_HOLDS`.

Eight points, not six: this family runs **L2 through L8**.

1. **P1 the kinematic solve converged** — as above.
2. **L2 every env really stands OUTSIDE the pocket wall (|y| > 72.55 mm)** —
   PASS: all four near the commanded 191.889 mm. FAIL lists the envs still
   over the pocket, and then nothing below tests D-157 at all — that was
   RT-112/RT-113.
3. **L3 the projected depth is INSIDE the 33.0-36.0 mm band (the hack pose, or
   nothing is exercised)** — PASS: the axis projection reads a perfect seat
   while the part stands beside the block. FAIL means the hack pose was never
   reached, and L4/L5 then prove nothing.
4. **L4 the success predicate does NOT fire beside the pocket (D-157)** — PASS:
   0/4 in region. This is the point RT-107's hack would break.
5. **L5 the engaged bonus does NOT fire beside the pocket (D-157)** — PASS: 0/4
   engaged.
6. **L6 the depth metric books nothing beside the pocket** — PASS: every
   `max_depth` entry 0.0.
7. **L7 the paid reward equals the D-109 formula on the printed numbers** — as
   P8.
8. **L8 the force in FREE AIR is under 1.0 N (the gravity tare removed the
   tool's own weight)** — PASS: |F| near 0 on the TARED channel while the
   printed RAW untared wrench still shows the tool's own weight. A FAIL with a
   DOUBLED raw value is a sign error in the tare; a FAIL with a noisy tared
   value is the force-noise pin — see below.

### Run 4 — `--reward-curve --curve-heights-mm 30.0 5.0`. Verdict line `REWARD_CURVE_MEASURED`.

**This mode judges nothing about the task.** Its own help text
(`check_seated_success.py:1254`): "Nothing here judges the task; the verdict is
made by /rt-check against an expectation written BEFORE the run." Its
`verdict_ok` (`check_seated_success.py:1697-1699`) is only whether the teleport
reached the commanded grid: `all_converged AND all_poses_held AND (no reset, or
the reset was a success termination)`.

The grid: 2 heights x 1 lateral y x 1 lateral x x 1 tilt — the three list
defaults are `[0.0]` (`check_seated_success.py:1294`, `:1308`, `:1322`) =
**2 points** x 4 settle steps = **8 steps of the 256-step episode**. Both
heights are POSITIVE, i.e. ABOVE the entrance, so no point sits in the success
band and **no reset is expected**.

What it MEASURES per point — env 0 on the `curve` line, all envs in the
`SUMMARY row` block and in the JSON: SDF distance, `kernel_sum` alone, both
bonus predicates, `in_pocket`, force, the per-step reward, the per-term row,
and the achieved y / x / depth with `pose_drift_mm`.

1. **`all_solves_converged` True** — the IK reached both commanded heights.
2. **`all_poses_held` True (tol 0.050 mm)** — the line the three pins pay for;
   see the next section.
3. **`reset_during_sweep` False, `sweep_end_reason` null** — neither 30 mm nor
   5 mm above the entrance is a seat, so nothing may terminate. A force abort
   or a timeout here leaves the curve unusable.
4. **`curve_coarse_margin_mm` null and `kernel_margin_coarse_mm` 173.21 mm** —
   the shipped coarse width is what is being measured, not a counter-proof
   width. `KERNEL_MARGIN_COARSE = 0.17321` (`insertion_tasks_cfg.py:1605`).
5. **Both `kernel_sum` values finite, and the 5 mm one larger than the 30 mm
   one** — a MEASUREMENT, not a threshold. No magnitude is predicted here:
   **unknown for this setup**, because this code has never run.

The paid counter-proof of this mode — `--curve-coarse-margin-mm 10`, which must
collapse the 165 mm value into the ~1e-22 range — is **NOT part of RT-180**: no
command above carries it, and the commanded heights do not include 165 mm.

## What is NEW and UNPROVEN in this run

Three things have never executed. RT-180 is their first proof.

### 1. The three pins, `scripts/check_seated_success.py:1878-1880`

```
env_cfg.obs_noise_pocket_pos_std_m = 0.0
env_cfg.force_obs_noise_std_n = 0.0
env_cfg.grasp_obs_offset_x_m = 0.0
```

The paragraph above them (`check_seated_success.py:1868-1877`) gives the
reason: this script reads the pose OUT OF THE OBSERVATION
(`_slice(obs, "tip_rel")` -> depth / lat_y / lat_x -> `pose_drift_mm`) and
gates it at `solve_tol_mm`. Its words: "An identity test needs the observation
to BE the pose." The comment above them had claimed the fixture jitter was "the
only reset randomisation the env still has", and `c6e8ed8` had made that false:
the three Phase-5 fields default ON.

The unpinned defaults, for scale:

| field | default | file:line |
|-------|---------|-----------|
| `obs_noise_pocket_pos_std_m` | 0.0025 m = 2.5 mm | `insertion_env_cfg.py:1041` |
| `force_obs_noise_std_n` | 3.5 N | `insertion_env_cfg.py:1042` |
| `grasp_obs_offset_x_m` | 0.003 m = 3 mm | `insertion_env_cfg.py:1050` |

**WHAT A FAILED PIN WOULD LOOK LIKE — so that it is not mistaken for a pose
that really moved.**

* **Curve mode.** `pose_drift_mm` is
  `max(|got_y - y_cmd|, |got_x - x_cmd|, |got_depth + h|)`
  (`check_seated_success.py:1626-1630`) and the gate is 0.05 mm
  (`:1631`). A 2.5 mm per-episode pocket bias, or a 3 mm grasp offset on the
  part's short axis, is one to two orders ABOVE that gate. The signature of a
  FAILED PIN, as opposed to a moved pose:
  * the same row reads `residual X.XXX mm conv True` — the IK closed its own
    error, and the drift exists only in the observation;
  * the drift is **the same on both rows**, of the order 2.5 mm or 3 mm,
    because `_bias` and `_grasp_obs_off` are written in `reset()` and nowhere
    else and both curve points sit inside ONE episode;
  * `POSE NOT HELD` fires while `got_depth_mm`, `got_y_mm` and `got_x_mm` are
    each offset by a CONSTANT per env.
  A pose that really moved looks different: the settle steps and
  `clamp_tip_in_box` move it by a row-dependent amount. RT-151b's case had
  `residual 0.000 mm` on rows that had slid 0.98 and 7.55 mm — different per
  row, not constant.
* **Lateral mode.** L3 gates the observed depth inside a 3 mm-wide band around
  the commanded 34.0 mm, and a 2.5 mm bias can push it out. L8 gates the
  observed force at 1.0 N against a 3.5 N sigma; that one fails loudly. A
  lateral run that fails L3 and/or L8 while L2, L4, L5 and L6 pass is a pin
  failure, not a D-157 finding.
* **Seated mode is the LEAST sensitive, and that is not a defence.** When the
  episode terminates, `tip_rel`, the force and `max_depth` are read from
  PRE-STEP captures (`_peg_geometry()` and `_force_smooth`,
  `check_seated_success.py:2240-2242`), i.e. from physics, not from the
  observation. So a failed pin can pass the seated run and still be there.
  **The seated verdict alone does not clear the pins. The curve run and the
  lateral run do.**

### 2. The four scatter fields in the metrics dump

`check_seated_success.py:2332` and `:2342-2344` write, into the JSON of the
seated / shallow / lateral runs — and `:1713-1716` writes the same four keys,
same spelling, into the curve run's JSON:

```
"fixture_pos_noise_xy_m", "obs_noise_pocket_pos_std_m",
"force_obs_noise_std_n", "grasp_obs_offset_x_m"
```

Each key is the cfg field's OWN name, and each value READS the cfg rather than
repeating a literal 0.0. **All four must read 0.0 in `RT-180_seated.json`,
`RT-180_shallow.json`, `RT-180_lateral.json` AND `RT-180_curve.json`.**

The evidence that the scatter was off is THE FILE, not the log prose. Before
this commit the dump named only the fixture jitter, so a run with the
observation noise or the grasp belief error left ON would have written a
metrics file indistinguishable from one where they were off. Numbers are read
from files.

**The curve dump carries them too, since `23a7aa0`. This file said the
opposite until 2026-09-12; that was true of `9f37a07` and is false of the
commit that ran.** The reward-curve mode builds its OWN metrics dict
(`check_seated_success.py:1699-1746`), and that dict now names all four keys
at `check_seated_success.py:1713-1716` — `git show
23a7aa0:scripts/check_seated_success.py | grep -n "grasp_obs_offset_x_m"`
returns `1716`, beside `:1713` `fixture_pos_noise_xy_m`, `:1714`
`obs_noise_pocket_pos_std_m` and `:1715` `force_obs_noise_std_n`. So
`RT-180_curve.json` IS readable as evidence that the scatter was off, on the
same four keys as the other three files, and it is the mode that needs it
most: the curve is the run that reads the pose out of the observation.
`pose_drift_mm` / `all_poses_held` remain the second, independent reading of
the same fact, not the only one.

### 3. The two startup-report lines

`InsertionEnv._print_startup_report` (`insertion_env.py:4035`) is called from
`_get_observations` whenever `_obs_calls` hits one of
`report_at_steps = (2, 60, 180, 258)` (`insertion_env_cfg.py:796`;
call site `insertion_env.py:1848`). The IK teleport does NOT call `env.step` —
it writes the joint state directly (`check_seated_success.py:2083-2113`) — so
the observation calls of these runs are one reset plus the settle steps:
**at most 31 in the seated / shallow / lateral runs (30 settle steps) and 9 in
the curve run (2 points x 4).** So **only count 2 fires, and the report prints
exactly once per run.** Counts 60 / 180 / 258 are out of reach at this budget,
and the report's own "the pair to compare is step 2 against step 258"
instruction therefore CANNOT be followed in RT-180 — that pair needs a run past
the 256-step episode cap (`episode_length_s = 256/60`,
`insertion_env_cfg.py:138`).

**The observation-bias line — and the PASS is the OFF line, not a row of
zeros.** `obs_noise.resolve_obs_noise_model` (`obs_noise.py:204-222`) returns
`None` when BOTH sigmas are 0.0, and a `None` model is never built by the base
class. The env branches on exactly that (`insertion_env.py:4431-4441`):

* **PASS — the line that must appear:**
  `observation bias (D-182): OFF -- both sigmas are 0.0, so
  resolve_obs_noise_model returned None and no model exists`
  (`insertion_env.py:4433-4434`).
* **FAIL:** an `observation bias env0..3 (D-182, tip_rel <lo>:<hi>, sigma ... m,
  drawn per EPISODE): [[...]]` line (`insertion_env.py:4439-4441`). Even a row
  of ZEROS on that line is a FAIL, because it means the model was BUILT: a
  zero-width model still draws `randn_like` every step and so moves the global
  RNG stream, and the switched-off env is then no longer the un-noised env bit
  for bit (`obs_noise.py:211-216`).

**The grasp-belief line behaves by the opposite rule and always prints**
(`insertion_env.py:4442-4445`) — it has no OFF branch:

* **PASS:** `grasp belief error env0..3 (D-183, +-0.0 m on the part's short
  axis, drawn per EPISODE): [0.0, 0.0, 0.0, 0.0]`. Here a row of zeros IS the
  pass, because the value is `(2u - 1) * grasp_obs_offset_x_m`
  (`insertion_env.py:2228-2230`) and that factor is pinned to 0.0.
* **FAIL:** any non-zero entry, or `+-0.003 m` in the header.

The two lines are therefore judged by OPPOSITE rules and must not be confused:
the OFF line for the bias, a zero row for the grasp error.

## What this run cannot answer

* **Nothing about training.** No policy is loaded, no gradient is taken, no
  success rate is produced. Four teleports are not a run.
* **Nothing about the reward's behaviour over an EPISODE.** Every reward number
  here is a per-step value at a teleported pose. The reward-ordering test
  (expert / hover / wedged / idle), the discounted shaping balance and the
  payout's dominance over T steps are not touched.
* **Nothing about whether the noise model works WHEN IT IS ON.** This run
  switches all three sources OFF by construction. That the bias is re-drawn per
  episode rather than accumulated (the `operation="abs"` claim), that the
  2.5 mm sigma reaches the right observation slice, that the 3.5 N force noise
  and the +-3 mm belief error are wired at all — none of it is tested here. The
  only thing proven is that the OFF path is genuinely off. The ON-path check is
  owed by `Laufplan_Phase5.md` § Order step 3 ("noise model built, ... grasp
  offset in the startup report, pocket bias does not grow across resets") and
  needs a run that crosses a reset, i.e. past 256 control steps.
* **Nothing about the coarse kernel's decay.** Two points at 30 mm and 5 mm are
  not a curve; the shipped 173.21 mm width is not probed anywhere near the
  165 mm reset stand-off, and the `--curve-coarse-margin-mm 10` counter-proof
  is not run.
* **Nothing about the reset distribution, AutoDR, the disk start or H_min.**
  `fixture_pos_noise_xy` is pinned to 0.0 (`check_seated_success.py:1867`),
  `curriculum_enabled` is False, and `start_tip_above_entrance` is pinned to
  None (`:1898`, the 165 mm home pose) — so the Phase-5 reset path is
  deliberately not exercised.
* **Nothing about `w_depth_progress` or `w_tilt`.** Both are pinned to 0.0
  (`check_seated_success.py:1887`, `:1892`) because `expected_reward` does not
  model them. They are verified offline and on a training curve, not here.
* **Nothing about the OSC clamp box, the contact model, or force limits under
  load.** The lateral pose is free air and the seated pose is a kinematic
  teleport; no policy ever drives into the pocket in this run.
