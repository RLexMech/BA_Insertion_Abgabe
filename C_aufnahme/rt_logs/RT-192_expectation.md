# RT-192 — expectation, written BEFORE the run (2026-09-13)

Written on the dev laptop, branch `p5-kraftsensor`. `/rt-check` judges the
logs and the JSON files against THIS file. **UNVERIFIED** until the training
PC has run it. Five short commands, no training beyond two PPO updates. This
is Prüfblock B of the Auftrag (`Pläne/Kontaktbeobachtung_Implementierungsplan_Claude.md`
§ 6) plus the PPO wiring, i.e. the IMPLEMENTATION check of D-188. It says
nothing about whether `wrench` learns better.

RT-190 is DROPPED (`d9205bf0`). RT-191 is TAKEN by the parallel chat on
`p5-robustheit` (curriculum run, 300 iterations, seed 20, 2026-09-13); this
block was RT-191 until then and is RT-192 now (checked against the
p5-robustheit worktree 2026-09-13).

## The one question

**"Do the three torque channels carry the joint wrench's moment — right
frame, right sign, right reference point, tared, into channels 28:31 — and
does the rest of the pipeline (reset, double read, PPO, checkpoints) hold
with them?"**

## What changed against RT-189s1pf (code, branch `p5-kraftsensor`)

* `insertion_math.py`: `obs_slices(mode)`, `obs_dim(mode)`,
  `gravity_tare_torque`, `assemble_observation_wrench`, `noise_masks` with
  `torque_std_nm` + `mode`.
* `insertion_env_cfg.py`: `obs_wrench_mode = "force"` (default, 28) /
  `"wrench"` (31); `torque_obs_noise_std_nm = 0.0`; `resolve_obs_layout`.
* `insertion_env.py`: wrench read `0:6`; torque tare with the lever
  `body_com_pos_w[tool] - body_pos_w[parent]`; `_torque_smooth`,
  `_torque_tared_raw`, `_wrench_valid`; **the wrench EMA skips the reset
  step and advances once per step** (inbox entry 2026-09-13) — this changes
  row 0 of the `force` channel for EVERY policy, RT-189s1's included.
* `play.py` / `train.py --resume`: checkpoint width guard
  (`scripts/tools/checkpoint_width.py`), refusal with a readable line.
* `probe_wrench_bodies.py --tare-check`, `trace_outcome_split.py`
  (torque in the truth report, marker `trace_outcome_split-2026-09-13b`).

## Named assumptions this run measures

1. Reference point of the moment = PARENT link origin (`wrist_3_link`), the
   Isaac test's (`test_articulation.py:1871-1882`). Point 1 tests it against
   the child origin.
2. Sign of the weight moment = the measured FORCE sign (raw ≈ +hold,
   RT-115/RT-180c), not the Isaac test's `m*g`. Points 1 and 2.
3. rsl_rl 3.0.1 checkpoint layout `model_state_dict` / `actor.0.weight`.
   Command 5.

## Commands (training PC, PowerShell)

Root: `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`
(the `p5-robustheit` clone, HANDOFF-RL.md "THE p5-robustheit CLONE"). ONLY
after RT-191 (the curriculum run from `p5-robustheit`) has FINISHED (never two runs
at once, RT-182). The branch is switched IN THAT CLONE and switched back
afterwards; the editable install stays where it is (same folder).

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1
git fetch origin
git checkout p5-kraftsensor
git pull
git rev-parse HEAD
```

Code state: `971f32c05fc3f07c1f85901a6c36bc05d053b51c` (commit 2b, the last code
change). Expected HEAD = the commit that carries THIS file, one ahead of it;
its SHA is in the chat handover and in `HANDOFF-RL.md` START HERE. Anything
else: STOP, do not run.

```
.\scripts\rt_log.ps1 RT-192a python -u scripts/probe_wrench_bodies.py --tare-check --headless --json rt_logs/RT-192a_tare_check.json
```

```
.\scripts\rt_log.ps1 RT-192b python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 64 --headless --max_iterations 2 env.obs_wrench_mode=wrench env.dr_mode=autodr
```

```
.\scripts\rt_log.ps1 RT-192c python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --trace-obs rt_logs/RT-192c_trace.json --trace-steps 32 env.obs_wrench_mode=wrench env.dr_mode=autodr
```

(no `--load_run`: play.py takes the NEWEST run folder, which is RT-192b's.)

```
.\scripts\rt_log.ps1 RT-192d python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --load_run "09-12_21-58-11" --trace-obs rt_logs/RT-192d_trace.json --trace-steps 32 env.dr_mode=autodr
```

```
.\scripts\rt_log.ps1 RT-192e python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --load_run "09-12_21-58-11" --trace-obs rt_logs/RT-192e_trace.json --trace-steps 32 env.dr_mode=autodr env.obs_wrench_mode=wrench
```

Then back:

```
git checkout p5-robustheit
```

Hand over: the five short logs, `rt_logs/RT-192a_tare_check.json`, and
`python scripts/trace_outcome_split.py rt_logs/RT-192c_trace.json` plus the
same for `RT-192d_trace.json` (run on the training PC or paste the JSONs).

## Points (all pre-registered; a number is PASS only against its own line)

**RT-192a — Prüfblock B, `wrench_tare_check.json`:**

* **P1 setup.** `[rt_log] exit code: 0`, `git:` = the SHA above, startup
  report prints `obs mode wrench-31`, torque sigma 0.0. Otherwise RUN INVALID.
* **P2 rest (point 1).** `1_rest.fits` has EXACTLY ONE entry and it is
  `parent_origin+`; `env_tare_fits` true; `torque_tared_env_norm_nm` ≤
  max(0.005, 1 % of `candidates.parent_origin+.pred_norm_nm`) in 4/4 envs;
  `force_tared_norm_n` < 0.01 N in 4/4 (RT-180c form). `|f_raw|` ≈ 8.0834 N
  (ratio 0.99..1.01).
  * `fits` = `parent_origin-` → the sign convention is the Isaac test's, our
    torque tare is inverted: FINDING, flip the sign in
    `gravity_tare_torque`, re-run a only.
  * `fits` = `child_origin±` → the reference point is the child origin:
    FINDING, the lever in `insertion_env.py` becomes
    `body_com_pos_w[tool] - body_pos_w[tool]`, re-run a only.
  * `fits` empty → the reference point is neither: FINDING, no further
    building; print the numbers.
* **P3 known load (point 2).** `2_load.force_fits` true with `force_sign`
  equal to the sign found in P2; `torque_fit` = `parent_origin+` (ratio
  0.97..1.03, cos ≥ 0.999 in 4/4); `obs_channel_equals_ema` true (channels
  28:31 moved by exactly the EMA delta, |diff| ≤ 1e-6, and 25:28 likewise).
  Expected magnitudes, as a plausibility line only: |ΔF| ≈ 5 N, |Δτ| ≈
  5 N × |r_tip| with r_tip ≈ 0.15 m ⇒ ≈ 0.7-0.8 N m (the exact value is the
  JSON's `pred_torque_nm`).
* **P4 partial reset (point 3).** All six flags in `3_reset` true: flag
  only on env 0, env 0 buffers AND obs channels 25:31 zero on the reset
  step, `valid` = [False, True, True, True] on that step and all True on
  the next, other envs' torque unchanged (≤ 1 %), env 0 EMA restarts at
  0.25 × its tared raw on the next step.
* **P5 double read (point 4).** All five `buffers_byte_identical` true.
* **Verdict line:** `TARE_CHECK_PASS`, exit 0. Any FAIL: the point names
  the defect; fix that one thing, re-run a only.

**RT-192b — two PPO updates in `wrench` mode:**

* **P6.** exit 0, startup report `obs mode wrench-31`, `observation_space`
  31, two iterations logged with FINITE reward / losses / entropy (no nan,
  no inf), a checkpoint `model_1.pt` (or the final index the runner writes)
  in the new run folder under `logs/rsl_rl/ur5e_insertion/`. No number of
  the learning curve is read: 2 iterations say nothing.

**RT-192c — replay of that checkpoint in `wrench` mode:**

* **P7.** exit 0, loads without a width refusal, trace JSON carries
  `obs_wrench_mode = "wrench"`, `obs_version = "wrench-31"`, key
  `torque_nm` with 32 rows × 4 envs × 3, `slices.torque = [28, 31]`, and
  the truth block with `torque_tared_raw_nm`, `torque_ema_nm`,
  `wrench_valid`. `trace_outcome_split` prints `raw tau mean ... Nm` per
  env. Row 0: `wrench_valid` all False (the reset step); rows 1..: True.
  Rows 1..31: `torque_nm` == `truth.torque_ema_nm` exactly (sigma 0).

**RT-192d — the OLD policy (28) in `force` mode on the new code:**

* **P8.** exit 0, loads (28 = 28, no refusal), trace `obs_version =
  "force-28"`, NO `torque_nm` key, truth block DOES carry the three torque
  keys (filled in every mode). NAMED: row 0 force is 0.0 now (the reset
  mask), so this replay is not bit-identical to RT-189s1pf — first-step
  actions may differ; not a defect, the inbox entry owns it.

**RT-192e — the OLD policy against `wrench` = the refusal:**

* **P9.** the log carries one line `[play] REFUSED: checkpoint (...) was
  trained on a 28-wide observation, but the env builds 31
  (obs_wrench_mode='wrench'). It matches obs_wrench_mode='force': pass
  env.obs_wrench_mode=force ...`, exit code 2, NO `size mismatch`
  traceback from `runner.load`. A `size mismatch` traceback instead means
  the guard did not fire (the checkpoint layout assumption 3 is wrong):
  FINDING, print the checkpoint's keys.

## Stop rule (Auftrag § 6)

P1-P9 green ⇒ the implementation is DONE and verified for its own claims;
nothing more is run for it. No seeds, no long runs. Whether `wrench` helps
is the NEXT experiment, its own expectation file.

## NACHTRAG 2026-09-13 — first RT-192a run (marker `-13a`, code `f85e6493`): TARE_CHECK_FAIL, two of four points

Read from `RT-192a_tare_check.json` (pasted in chat; `[rt_log]` head not
handed over, so the git line is not evidenced — the marker pins the script).

* P4 (partial reset) and P5 (double read): **PASS**, all flags true.
* P2 (rest): tare right — raw torque 0.00686 N m, tared 9e-7 N m, raw force
  8.0834 N, tared 8e-7 N. But `fits` = [`parent_origin+`, `child_origin+`]:
  the two origins COINCIDE and the gravity lever is parallel to the tool
  axis, so the test could not separate them. My test design, not the code.
  Sign +: pinned.
* P3 (load), force: |ΔF| 5.002 N (ratio 1.0004), direction cos −1.000. The
  sign convention is MEASURED: incoming wrench = parent reaction = MINUS the
  load, the same as gravity (raw = +hold = −m g). My expectation had the
  load sign wrong; the code is consistent.
* P3, torque: **FINDING.** |Δτ| = 1.276 N m about −y against a predicted
  0.760 N m: a lever of 0.255 m where the tip sits 0.152 m from the link
  origin. The moment is NOT taken about the link origin, OR the application
  point is not where `positions=` says. 0.103 m is unexplained. cos −1
  (consistent with the measured sign).
* Channels 28:31 moved exactly by the EMA delta: the wiring is right.

**Re-run of RT-192a only, marker `probe_wrench_bodies-2026-09-13b`:** the same
5 N at TWO points (tip, and tip − 0.100 m up the axis). Pre-registered:

* `lever_diff_m` = 0.100 ± 3 % in 4/4 ⇒ the application point IS honoured
  and `reference_point_d_m` names the reference point (metres up the tool
  axis from the link origin; expected 0 if the env's assumption holds,
  ≈ −0.103 if the first run's number stands). Then read it against
  `com_parent_in_parent_m` / `com_child_in_parent_m` in the JSON.
* `lever_diff_m` ≈ 0 ⇒ the application point is ignored (the load acts at
  one fixed point); then nothing about the reference point is known yet.
* Either way P2 now passes on sign and tare alone; the env's reference
  point assumption is judged by `torque_fit == "parent_origin"`.
  `torque_fit == None` with a clean lever difference means the env's lever
  must be corrected by d: a code change, one commit, one more run of a.

Command (same clone, after pull):

```
.\scripts\rt_log.ps1 RT-192a2 python -u scripts/probe_wrench_bodies.py --tare-check --headless --json rt_logs/RT-192a2_tare_check.json
```

## NACHTRAG 2026-09-13 — RT-192a2 (marker `-13b`, git `6426120`): reference point MEASURED, exit 1 by design

`[rt_log]` head handed over: git 6426120 matches. Points 1, 3, 4 PASS.
Point 2: force ratio 1.0004 / 1.000, cos 1.000 at both points (sign
convention now in the prediction). `lever_diff_m` 0.100 = dz in 4/4: the
application point IS honoured. Levers 0.2552 m (tip) and 0.1552 m (tip −
0.1): **the moment is taken about a point 0.1032 m up the tool axis from
the link origin** (`reference_point_d_m` 0.1032, 4/4). From the JSON:
`tool_origin_in_parent_m` ≈ 0 (origins coincide), `com_parent_in_parent_m`
z −0.0229, `com_child_in_parent_m` z +0.1032 — the reference is the MIRROR
of the tool COM through the joint origin. Noted, not explained. Neither
candidate fitted (`torque_fit` null) → FAIL as pre-registered.

**Fix (one thing):** `InsertionEnvCfg.torque_ref_offset_parent_m =
(0.0, 0.0, -0.1032)` m in the parent frame; the env's lever is now
`COM − (parent origin + R_parent · offset)`. x of the offset is NOT
measured (an x load cannot see it).

**RT-192a3, marker `probe_wrench_bodies-2026-09-13c`:** three loads —
x at the tip, x at tip − 0.1 m, y at the tip — and a third candidate
`measured_ref` (the cfg offset). Pre-registered: `torque_fit` =
`measured_ref` with ratio 0.97..1.03 and cos ≥ 0.999 at all three loads;
`reference_point_x_from_y_load_m` ≈ 0 (|x| < 0.003 m) — if not, the x
component of the offset is that number and one more run follows. One
startup report only (`report_at_steps = (2,)`), so the short log stays
short.

```
.\scripts\rt_log.ps1 RT-192a3 python -u scripts/probe_wrench_bodies.py --tare-check --headless --json rt_logs/RT-192a3_tare_check.json
```

## NACHTRAG 2026-09-13 — RT-192a3 (marker `-13c`, git `13ee31a`): the 0.1032 m was an ARTEFACT of the test; WITHDRAWN

With the cfg offset the "measured lever" at the same tip load jumped from
0.2552 m to 0.4086 m, the lever difference from 0.100 to 0.124 m, and the
ratio against every candidate stayed ~1.6-2.7. A geometry cannot do that.
Cause, read off the a2 log: under the 5 N load the OSC let the arm move
by tens of degrees (second startup report: 67 deg tilt, 0.10 m sideways).
My comparison "tared after minus tared before" therefore carried the change
of the GRAVITY moment with the pose (COM lever 0.103 m x 8.08 N x sin(tilt)
~ 0.7 N m), and the tare with a shifted reference changed that term again.
Point 3 failed for the same reason (env 1 still drifting under the y load).
`torque_ref_offset_parent_m` is back at (0, 0, 0). The y-load number
(x 0.015 m) is void for the same reason.

What stands from a1..a3: force magnitude and frame (ratio 1.000), sign
convention (incoming = minus load, as gravity), rest tare (1e-6 N m),
reset mask, once-guard, obs 28:31 == EMA delta.

**RT-192a4, marker `probe_wrench_bodies-2026-09-13d`:** during every load the
JOINTS ARE PINNED (`write_joint_state_to_sim` each physics substep, the
seat_probe form), so the pose and the gravity moment are identical before
and after, and the RAW torque difference is the load moment alone; no tare
enters. The load is removed and the OSC settles before point 3.
Pre-registered: `pose_drift_m` < 1e-4 m at every load; `lever_diff_m` =
0.100 ± 3 %; `reference_point_d_m` names the reference (0 ± 0.005 m = the
link origin, the env's current assumption → `torque_fit` = `parent_origin`
or `measured_ref`, both the origin now, PASS); any other d = the number to
put into `torque_ref_offset_parent_m`, one more run.

```
.\scripts\rt_log.ps1 RT-192a4 python -u scripts/probe_wrench_bodies.py --tare-check --headless --json rt_logs/RT-192a4_tare_check.json
```

## NACHTRAG 2026-09-13 — RT-192a4 (marker `-13d`, git `4b660b2`): pinning by state overwrite is no equilibrium; WITHDRAWN

Point 1 PASS again. Point 2 under the overwritten joint state: force ratio
0.33 / 0.33 / 0.50, cos −0.24 / 0.69 / −1.00 — even the FORCE is wrong, so
the reaction wrench read while the state is rewritten every substep is not
a static reaction. My test, not the physics. Levers 0.134 / 0.068 / 0.079 m
are void.

Why the OSC could not hold the pose in a1..a3 (read off `_apply_osc`): with
a zero action the target is re-anchored to the CURRENT pose every physics
step, so a constant load drags the arm without limit (a2: 67 deg tilt,
0.10 m sideways at step 60).

**RT-192a5, marker `probe_wrench_bodies-2026-09-13e`:** the probe runs the
env in `control_mode="joint_pd"`. The USD drives hold the joints with their
authored stiffness (only the `osc` branch zeroes them, insertion_env.py:1416),
so the load gives a small STATIC deflection and a real reaction; the wrench
read, the tare and the obs wiring are the same code in both modes. Deltas
are RAW (same pose before/after up to the small deflection). Pre-registered:
`joint_drift_rad` small at every load (reported, no bar); force ratio
0.97..1.03 with cos ≥ 0.999 at all three loads; `lever_diff_m` = 0.100 ± 3 %;
`reference_point_d_m` names the reference (0 ± 0.005 m = the link origin =
PASS with `torque_fit` parent_origin/measured_ref; anything else = the number
for `torque_ref_offset_parent_m`, one more run). Points 3 and 4 as before.

```
.\scripts\rt_log.ps1 RT-192a5 python -u scripts/probe_wrench_bodies.py --tare-check --headless --json rt_logs/RT-192a5_tare_check.json
```

## NACHTRAG 2026-09-13 — RT-192a5 (marker `-13e`, git `cb2248e`, joint_pd): the reference point IS 0.1032 m up the arm

Static pose, raw deltas. Force ratio 0.9997 / 0.9998 / 1.0002, cos 1.000 at
all three loads. Levers: tip 0.2551 m, tip − 0.1 m 0.1552 m (difference
0.0999 = dz: application point honoured), y load at the tip 0.2552 m (no
sideways offset). `reference_point_d_m` = 0.1031..0.1032 in 4/4: **the
moment is taken about a point 0.1032 m up the tool axis from the link
origin.** Same number as a2; a3/a4 were artefacts. The mirror of the tool
COM (+0.1032), noted, not explained. Point 1 residual 0.00028 N m under the
joint_pd droop (|f_tared| 0.0028 N) — PASS. Points 3/4: lines not in the
handed-over excerpt; read from the full log before the verdict.

`torque_ref_offset_parent_m = (0, 0, -0.1032)` is set from THIS run.

**RT-192a6 (same marker `-13e`, only the cfg constant changed):**
pre-registered `torque_fit` = `measured_ref`, ratio 0.97..1.03 at all three
loads; points 1, 3, 4 PASS; verdict TARE_CHECK_PASS, exit 0. Then b..e.

```
.\scripts\rt_log.ps1 RT-192a6 python -u scripts/probe_wrench_bodies.py --tare-check --headless --json rt_logs/RT-192a6_tare_check.json
```

## NACHTRAG 2026-09-13 — RT-192a7 (marker `-13f`, git `4d1860f`): TARE_CHECK_PASS

All four points PASS (console excerpt, `[tare]` lines; exit line not in the
excerpt, JSON written). Point 2: force ratio 0.9997/0.9998/1.0002, cos 1.000;
`measured_ref` ratio 0.9996/0.9997/0.9998, cos 1.000 at all three loads;
lever diff 0.0999 m; d 0.1031 m. Points 3 and 4 all true. Prüfblock B is
DONE for its own claims. Next: b..e at this code state, then RT-193.
