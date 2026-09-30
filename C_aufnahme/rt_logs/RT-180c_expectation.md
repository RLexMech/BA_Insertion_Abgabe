# RT-180c — expectation, written BEFORE the run (2026-09-12)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

THE LATERAL COUNTER-PROOF, THIRD ATTEMPT, WITH A WIDENED CLAMP BOX AND THE
SETTLE COUNT BACK AT 30. RT-180 (30 settle steps) FAILED L2/L3: the tip was
dragged from the commanded 191.889 mm to 84.3 mm / -50.1 mm. RT-180b (1
settle step) HELD the pose — L3 and D-157 passed for the first time on the
OSC env — but FAILED L8: 1.569 N tared against a 1.0 N gate, raw 8.28 N
against the tool's own 8.083 N weight, so the part was still moving when the
predicate was read.
The two runs are the two ends of the settle range and both fail, because the
pull is present at EVERY step: `_apply_osc` builds the target as the current
pose plus the action delta and clamps it into the box around the entrance
afterwards (`insertion_env.py:1521-1550`), and every LEGAL lateral offset
(>= 100.45 mm + half the part width = 172.2 mm) lies outside the shipped
0.08 m half-width (`insertion_env_cfg.py:252`).

**The code change (D-186, commit `5e8c3aa`): the `--lateral` branch of
`check_seated_success.py` now widens `env_cfg.osc_pos_clamp_m` to
`max(clamp_m, |lateral_y_m| + 0.005)` before `gym.make`.** With the default
offset that is `0.08 m -> 0.1968892059209824 m`. The z half-width
`osc_pos_clamp_z_m` is NOT touched. Seated and shallow runs keep 0.08 m.
The metrics dict gained `osc_pos_clamp_m` and `settle_steps`, both read back
from the cfg and the parsed argument.

Marker bumped: `check_seated_success-2026-09-05c` ->
**`check_seated_success-2026-09-12a`** (`check_seated_success.py:97`). The
env is UNCHANGED — `git diff --stat 23a7aa0..5e8c3aa` touches no file under
`source/` — so `insertion-osc-2026-09-03a` must still appear
(`insertion_env.py:73`).

**A PULL IS REQUIRED.** The training PC is on `23a7aa0`; the script change
is not there. See § Command for the pull, the verify and the expected SHA.

## The one hypothesis

**With the box widened past the commanded offset, 30 settle steps hold the
teleported pose AND let the motion die out, so L3 and L8 pass in the same
run for the first time.** RT-180b proved the pose is reachable; RT-180
proved 30 steps are enough to settle the force. Only the clamp stood between
them.

## The discriminating measurement this run adds

RT-180 and RT-180b both reported a gap: the IK converged to 0.0003 mm on a
commanded y of 191.889 mm, yet the first post-step reading was 173.227 mm —
18.66 mm short, ratio 1.108, no unit. Two candidate causes were on the table
and neither run could separate them:

* **(A) one settle step already pulled the tip toward the 80 mm box edge.**
  Then the widened box removes the pull and this run reads y about 191.9 mm.
* **(B) the commanded point and the reported `peg TIP` are different
  points.** Then the box is irrelevant and this run reads y about 173.2 mm
  again, at the SAME 18.66 mm offset.

**This run separates them.** The outcome is recorded either way; (B) is a
separate finding and does not by itself fail the run, because L2 and L3 are
judged on the ACHIEVED pose, not on the command.

## Command (training PC, PowerShell, conda env `env_isaaclab`)

Pull first, then verify the SHA, then run. Three commands, in this order:

```
git -C C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1 pull --ff-only
```

```
git -C C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1 rev-parse HEAD
```

That must print the SHA of the branch tip pushed from the laptop. **Until the
laptop pushes, the pull brings nothing and the run must not start.** The real
gate is the marker line in the log: if it does not read
`check_seated_success-2026-09-12a`, the run used the OLD script and is void
(precedent: RT-151b, `rt_logs/VERDICTS.md` 2026-09-05 14:51:11, thrown out as
`UEBERHOLT` on exactly this).

```
.\scripts\rt_log.ps1 RT-180c python -u scripts/check_seated_success.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --lateral --out rt_logs/RT-180c_lateral.json
```

`--settle-steps` is NOT passed: the default is 30
(`check_seated_success.py:1377`), which is what this run wants. Everything
else stays at RT-180's defaults: band 33.0-36.0 mm, commanded depth 34.0 mm,
commanded lateral offset 191.889 mm (computed,
`check_seated_success.py:409-419`), pocket wall 72.55 mm, `--solve-steps 120`,
`--solve-tol-mm 0.05`. Hand over the full log AND
`rt_logs\RT-180c_lateral.json`.

## PASS / FAIL lines, in order

The eight point names are the script's own (`judge_run`), carried over from
`rt_logs/RT-180b_expectation.md` verbatim. Verdict line
`LATERAL_CONTROL_HOLDS`; exit code `0` iff the verdict holds
(`check_seated_success.py:2391`, `:2401`).

1. **P1 the kinematic solve converged** — PASS: `solve_converged: true`,
   residual under 0.05 mm (RT-180 read 0.000309 mm). The residual is taken
   after the IK, before the settle, so the clamp cannot touch it.
2. **L2 every env really stands OUTSIDE the pocket wall (|y| > 72.55 mm)** —
   PASS: `lateral_y_mm` all well above 72.55 mm. Expected near 191.9 mm
   under cause (A), near 173.2 mm under cause (B); both pass L2. FAIL: any
   |y| <= 72.55 mm — that would mean the widened box did not reach the cfg.
3. **L3 the projected depth is INSIDE the 33.0-36.0 mm band** — PASS:
   `achieved_depth_mm` all near 33.97 mm. **RT-180 failed this at 30 steps;
   RT-180b passed it at 1 step. This run claims 30 steps now pass too.**
   FAIL: a depth near -50 mm means the pull is still there and the
   assignment to `env_cfg.osc_pos_clamp_m` did not take effect — check the
   JSON key first, not the physics.
4. **L4 the success predicate does NOT fire beside the pocket (D-157)** —
   PASS: `in_region` all `false`, detail `0/4 in region`. RT-180b passed
   this; here it must hold after 30 steps of settled, contact-free hold.
5. **L5 the engaged bonus does NOT fire beside the pocket (D-157)** — PASS:
   `engaged` all `false`, `0/4 engaged`.
6. **L6 the depth metric books nothing beside the pocket** — PASS:
   `max_depth_metric_mm` all `0.0`, detail `booked: []`.
7. **L7 the paid reward equals the D-109 formula on the printed numbers** —
   PASS: `reward` and `reward_expected` agree within 0.001 per env.
8. **L8 the force in FREE AIR is under 1.0 N** — PASS: `force_n` all
   < 1.0 N and `force_n_raw_untared` near 8.08 N. **This is the point
   RT-180b failed** (1.569 N tared, raw 8.28 N, i.e. 0.2 N above the
   8.083 N weight). At 30 steps the EMA `_force_smooth`
   (`ft_smoothing_factor` 0.25, `insertion_env.py:1825-1827`) is converged,
   so `force_n` should read what RT-180 read at 30 steps: about 0.04 N. A
   raw value still above the 8.083 N weight means the pose is being pushed
   even inside the widened box — that is a new finding, and the next question is then
   the OSC gain, not the clamp.

## The two new pins, read from the FILE (`RT-180c_lateral.json`)

* `osc_pos_clamp_m` must read **`0.1968892059209824`**. `0.08` would mean the
  assignment never ran; anything else means the offset or the margin moved.
* `settle_steps` must read **`30`**.

Both keys are NEW in this run. Their absence means the training PC ran the
old script — the same failure the marker gate catches.

**Pins, unchanged from RT-180b and read from the FILE, not the log prose:**
`fixture_pos_noise_xy_m`, `obs_noise_pocket_pos_std_m`,
`force_obs_noise_std_n`, `grasp_obs_offset_x_m` all `0.0`; the report prints
the OFF form `observation bias (D-182): OFF -- both sigmas are 0.0, so
resolve_obs_noise_model returned None and no model exists` and the zero row
`grasp belief error env0..3 (D-183, +-0.0 m ...): [0.0, 0.0, 0.0, 0.0]`. A
built-model bias line or a non-zero grasp entry is a FAIL regardless of
L2-L8.

**The mode line must name the widened box** (`--lateral` branch of
`check_seated_success.py`):
`LATERAL COUNTER-PROOF (y = 191.89 mm, block outer edge 100.45 mm, wall
72.55 mm, OSC tip clamp widened to 196.89 mm)`.

## What this run cannot answer

* **Nothing about training.** The widened box exists for this probe only;
  the training runs keep `osc_pos_clamp_m = 0.08`, and RT-180 remains the
  only measurement of what that box does to an out-of-box pose.
* **Nothing about D-157 under motion.** Zero action beside the block; the
  predicate is read at rest, which is what D-157 specifies.
* **Nothing about the reward over an episode, or about the noise model when
  it is on** — as RT-180 and RT-180b.

## CORRECTION, after the run (2026-09-12)

Two fixes to this file, both made after RT-180c ran, neither one changing
what the run was asked to show:

1. **The 11.6 N was misattributed.** It belongs to a SECOND probe at TWO
   settle steps (tared 3.03 N, raw 11.6 N), not to RT-180b. RT-180b at one
   settle step read 1.569 N tared and 8.28 N raw. `HANDOFF-RL.md`, the
   RT-180b paragraph, names both runs and is the source. The argument is
   unchanged: raw above the 8.083 N tool weight at either count means motion.
2. **The L8 expectation named "about 0.04 N" and the run read 0.00036 N.**
   That is a factor of about 111, not a unit factor. It is explained and it
   is not a defect: RT-180's 0.04 N was measured while the tip was still
   being dragged toward the box edge, so the EMA still carried motion. Here
   the pose stands to 0.00007 mm and the tared force falls to the numerical
   floor. The gate was `< 1.0 N` and it is met with a margin of 2800.

## RESULT (2026-09-12): PASS, all eight points, 22 of 22 pins

Read from `rt_logs/RT-180c/RT-180c_lateral.json` (git-ignored). The
discriminating measurement came out as cause **(A)**: `lateral_y_mm`
191.88926 mm against a commanded 191.88921 mm, a gap of 0.00007 mm. The
18.66 mm was the clamp pull. Cause (B), a different reference point, is
refuted. The verdict line is in `rt_logs/VERDICTS.md`.

**LIMIT: the `[rt_log]` head was not handed over.** The run's `git:` line is
therefore not evidenced. The marker `check_seated_success-2026-09-12a` in
the JSON is what pins the script version, and it is correct.
