# RT-142 — expectation (the TILT LADDER: does the reward rise from the crooked rim-rest to the seat?)

Written 2026-09-03 on the dev laptop, after the RT-141 verdict and the
RT-141p replay (`rt_logs/VERDICTS.md`, 2026-09-03). `/rt-check` judges the
log against THIS file and nothing else.

## Why this measurement exists

RT-141 (contact scale 0.02 + distance leash) ended with success 0.0 in all
six start-height bins, depth at the 7.3 mm floor, `far` 0.0 exactly, and
the replay showed the SAME end pose in all 16 envs: the part lies CROOKED
on the fixture rim, one edge in the opening, wrist strongly bent. That is
step 1 of the strategy the code names (`insertion_tasks_cfg.py:1624`:
"approach tilted, touch an edge, align on it, then push") — and the policy
never does steps 2 and 3. The user confirmed 2026-09-03 that the tilted
approach is WANTED, so a tilt penalty is off the table.

Whether the reward even RISES from that crooked rim pose to the seat has
never been measured: every existing curve (RT-123, RT-135) is UPRIGHT.
This run measures it. It JUDGES NOTHING about the task; the verdict on
which term is the lever comes from reading the rows.

## What the run is

`check_seated_success.py --reward-curve` with the NEW tilt ladder
(`--curve-tilt-deg`, `--curve-tilt-axis`; marker
`check_seated_success-2026-09-03`): tilt OUTERMOST, one descent from
+30 mm to the seat per tilt, a jam ends THAT tilt's descent only. Every
row prints the per-step reward TERMS (the same names as
`Episode_Reward/<term>` in the training log, read off the env's own
`_episode_sums` across the last settle step), the SDF distance, the
interpenetration, the force, the achieved tilt (signed about the chosen
axis, and the unsigned axis angle), and the IK residuals in mm and deg.

Two runs, one per tilt axis:

* **RT-142a — about pocket y (the LONG axis):** the part leans ACROSS its
  width. The lean the RT-141p screenshot shows, and the axis with the
  0.5876 mm play (D-106 across-tilt bound at full depth: 0.46 deg).
* **RT-142b — about pocket x (the SHORT axis):** the part leans along its
  LONG side. Play 1.60 mm (D-106 long-tilt bound ~2.5 deg).

Tilts 0, 5, 10, 20 deg. Heights +30, +20, +10, +5, +3, 0, -5, -10, -20,
-34 mm (tip = the part bottom on the tool axis, `peg_tip_offset`; a
NEGATIVE height is a depth). `--num_envs 4` like RT-123/RT-135; env 0
carries the printed line, all four go to the JSON. Fixture noise stays
the cfg default (5 mm xy), so the four envs are four pocket positions.

The env code is 609df52's (what RT-141 trained on); only the check
script changed. No smoke run needed: the check builds the env itself and
exits non-zero on an import failure (`isaac_exit`).

## Offline before the run (done on the laptop, 2026-09-03)

`python scripts/check_seated_success.py --self-test` → **49/49**, marker
`check_seated_success-2026-09-03`; the new case "curve_grid puts the tilt
OUTERMOST" PASSES with its mutation (tilt loop moved inside changes the
order). Run once against the worktree and once against a clean HEAD
checkout: both 49/49.

## The first attempt (RT-142a/b, 2026-09-03 17:17, git 306d273): FAIL, instrument defect

Both runs exit 1, `REWARD_CURVE_UNUSABLE`: the achieved tilt read
-180/+180 deg at EVERY commanded tilt, and the force abort ended every
descent at +30 mm with interpenetration up to 19.5 mm. Cause: the tilt was
measured on the welded PART body (`_peg_body_idx`), whose frame stands
180 deg to the pocket axis when the part is upright; the solver chased a
180 deg "error" and spun the tool into the fixture. **First fix attempt (marker `...-2026-09-03b`), ALSO FAILED**: switched
the read to the EE body / `ee_quat`, assuming that would read upright as
~0 deg. Ran on the training PC (git 5e4443c, three attempts) and read
180 deg there too, on both axes, all four envs, refused cleanly by a
guard rather than spinning the tool. The offline self-test
(`check_scripted_insert.py`) confirms `axis_tilt_angle(q, q) == 0` on
hand-built quaternions, so the formula is not at fault: the home pose
genuinely stands 180 deg to `pocket_quat` on this rig, on EITHER body,
and RT-123's whole upright curve was measured at exactly that pose with
no tilt logic at all. Also found in this pass: the tilt column's default
(`goal_tilt_rad=0.0`) had been active on EVERY `_solve_to` call, including
the seated identity test and both counter-proofs, which never asked for
tilt control before -- a second, until-now-unrun regression.

**Second fix (marker `...-2026-09-03c`), git 162d3fa: PHYSICALLY WORKED,
still exit 1.** `goal_tilt_rad` now defaults to `None`, tilt column left
untouched for the seated test and both counter-proofs. The ladder reads
the reset baseline and commands `baseline + radians(tilt_deg)`. Both
training-PC runs reached the actual physics: tilt 0 descended all the way
to `success_termination` at -34 mm (interpenetration/force normal
throughout), and the tilted descents jammed only partway down instead of
at +30 mm. But `VERDICT: REWARD_CURVE_UNUSABLE` on both anyway: at tilt 0
the measured angle flips between `-180.00` and `+180.00` (the same
physical orientation on the two sides of `atan2`'s branch cut), and the
UNWRAPPED difference read a spurious 360 deg residual -- `NOT CONVERGED`
on every row past the first few heights, despite the pose being correctly
reached.

**Third fix (marker `...-2026-09-03d`), THIS is what the commands below
run.** A `_wrap_pi` helper normalises every angle DIFFERENCE into
`(-pi, pi]` before it becomes an IK command or a residual: applied to the
IK's tilt command, the printed/JSON residual, and the baseline's own
per-env spread and mean (a plain `.mean()` across the branch cut has the
identical failure mode one step earlier, so the baseline now uses a
circular mean, `atan2(mean sin, mean cos)`). Checked by hand against the
exact failing case (`-180 - (+180)` now wraps to `0`, not `-360`) before
this fix shipped. The points below are unchanged; P1 carries the new
marker and P3 should now show real residuals, not branch-cut artefacts.

## Points, each PASS/FAIL on its own

1. **P1 — the code is this code.** `[rt_log] git:` is the commit named in
   the handover (or a descendant); `marker: check_seated_success-2026-09-03d`
   in the JSON and on the startup line. The line `tilt baseline at reset
   (obs ee_quat vs pocket_quat, signed): y ... deg (spread ...), x ... deg
   (spread ...)` prints, with BOTH spreads under 1 deg — the guard's own
   proof that the four envs share one baseline pose. The curve header prints
   `TILT LADDER, tilt = [0.0, 5.0, 10.0, 20.0] deg ON TOP OF the <baseline>
   deg reset baseline, about pocket y` (RT-142a) / `x` (RT-142b) and the term list
   `['kernels', 'engaged', 'success', 'progress', 'time', 'action_rate',
   'success_lump', 'abort', 'contact', 'far']`.
2. **P2 — the upright descent is RT-123, row for row.** Tilt 0 rows:
   `kernel_sum` at +30 mm ≈ 0.2235 (RT-123: 0.2235), at 0 mm within a few
   % of RT-123's 0 mm row, at -34 mm ≈ 0.5853 with `success True`,
   `success_termination` on that point and the payout in `reward`
   (RT-123: 566.2, now at T=256 and the D-165 term it may differ — QUOTE
   it). `far` row 0.0 at every tilt-0 height (all heights are inside the
   173.21 mm leash), `contact` 0.0 above the plane. If the tilt-0 rows do
   not reproduce RT-123, the ladder is not measuring the reward RT-141
   trained on, and P3/P4 are void.
3. **P3 — the IK reached every pose.** Every row `residual` ≤ 0.05 mm AND
   ≤ 0.05 deg, no `NOT CONVERGED`. A tilt the arm cannot hold (joint
   limits at 20 deg are possible) is reported as NOT CONVERGED and its
   row is VOID, not evidence — say which rows.
4. **P4 — THE READING: the crooked rim pose against the seat.** For each
   tilt ≠ 0, take the row nearest the rim rest (height 0 or the last row
   before the jam) and quote, per step: `kernels`, `contact`, `progress`,
   the `reward` column, `sdf`, `interpen`, `force`. Then the SAME numbers
   for tilt 0 at the same height, and for tilt 0 at the seat. The
   question, and the only one this run answers: **is there a height and a
   tilt where the per-step reward is HIGHER than at any upright pose the
   policy would have to pass through on the way down?** If a crooked row
   near 0 mm pays more per step than the upright rows between it and the
   seat (contact included), the crooked rest is a local optimum OF THE
   REWARD, and the term that makes it one is named by the row (kernel
   paying the crooked proximity, or contact pricing the upright descent).
   If every upright row below it pays MORE, the reward is monotone and
   the rest is an EXPLORATION failure, not a reward failure — a different
   next change (not a reward term).
5. **P5 — where the crooked descents END.** `sweep_end_by_tilt` in the
   JSON and the `RESET during ...` lines: per tilt, the height and the
   reason (`force_abort` = the crooked part jammed; `success_termination`
   would be a SURPRISE at any tilt ≥ 5 deg — the pocket walls cannot hold
   a part 5 deg crooked at 33–36 mm depth (D-106), so that would be a
   predicate defect and its own finding). Quote force and interpen on the
   jam row. Readings, not gates.
6. **P6 — the SDF under tilt.** `sdf` of the tilted rows vs the upright
   row at the same height. The kernel reads `mean_outside_distance` over
   the part's sampled points; a lean lowers some points and raises others,
   so the mean may barely move — or fall, if the dipping edge enters the
   opening. This is the number that says whether the kernel CAN tell
   crooked from upright at all. Reading, not gate.
7. **P7 — the term identity.** On every row NOT flagged `RESET ON THIS
   POINT`, `reward` (the env's paid step value) must equal the sum of the
   ten printed terms to 1e-4 (both read from the same step; a mismatch is
   a wiring defect in the check, not in the env). On a reset row the
   terms are the PREVIOUS settle step's (the sums are zeroed on the
   ending step) and the identity is not expected.

## The commands (training PC, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v1
git pull
git rev-parse --short HEAD
.\scripts\rt_log.ps1 RT-142a python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 30 20 10 5 3 0 -5 -10 -20 -34 --curve-tilt-deg 0 5 10 20 --curve-tilt-axis y --out rt142a_tilt_ladder_y.json
.\scripts\rt_log.ps1 RT-142b python scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --reward-curve --curve-heights-mm 30 20 10 5 3 0 -5 -10 -20 -34 --curve-tilt-deg 0 5 10 20 --curve-tilt-axis x --out rt142b_tilt_ladder_x.json
```

Bring back: both `[rt_log]` headers, both full curve outputs (every
`curve` and `terms` line, every `RESET during` line, the VERDICT line and
exit code), and both JSON files (`rt142a_tilt_ladder_y.json`,
`rt142b_tilt_ladder_x.json`, written to the repo root).

## What this run cannot answer

* **The exploration path.** A monotone reward does not make the policy
  find it; P4's "exploration failure" branch names a different problem,
  not a solution.
* **The actual tilt of RT-141's end pose.** Nothing logs the part's tilt
  per episode in training; the ladder brackets it with 5/10/20 deg, it
  does not measure it. If the reading turns on the exact angle, a
  per-episode tilt metric is the next instrument.
* **The lateral position of the rim rest.** The ladder is at y = 0 (the
  pocket axis); the crooked rest may sit off-axis. `--curve-lateral-y-mm`
  is refused below the plane, so an off-axis crooked row needs a
  separate decision.
