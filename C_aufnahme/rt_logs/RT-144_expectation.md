# RT-144 — expectation (THE K1 MEASUREMENT: scripted insertion, joint PD vs OSC, with joint torques)

Written 2026-09-03 on the dev laptop, BEFORE any run. `/rt-check` judges the
log against THIS file and nothing else. Spec: `docs/decisions_inbox.md`
(branch `p4-konzept`), entry "Audit 2026-09-03 (a)", "Discriminating
measurement"; reason: `docs/reference/konzept_audit_proxy_vs_real_2026-09-03.md`
K1 and section 6 ("Die EINE Messung, die K1 entscheidet").

**Run RT-143 first.** RT-144 needs P1 and P2 of RT-143 (inert drives,
Lambda printed) and is not worth a training-PC hour if the arm falls in
free air.

## What this measures

The RT-129 form — `tilt_insert.py`, 17 envs, start tilt −8..+8 deg about
the pocket y axis, ramped upright by 8 mm of depth, descent stopped at 35 mm
(the seat stop RT-129 used), abort limit opened to 300 N so the curve is not
truncated — run THREE times on the same commit, the same part, the same
start height (the bare home pose; the script pins
`start_tip_above_entrance = None`) and the same tilt sweep:

| run | control_mode | kp pos | note |
|---|---|---|---|
| RT-144a | `joint_pd` | – (USD drives 9400.5 / 0.378) | today's controller, the PD half |
| RT-144b | `osc` | 100 [placeholder] | the decided controller |
| RT-144c | `osc` | 500 [placeholder] | the second placeholder value |

NEW in the instrument (marker `tilt_insert-2026-09-03a`): the joint torques
of all six joints EVERY control step (`robot.data.applied_torque`, the
actuator model's applied effort after its clip — under joint_pd that is the
implicit drive formula stiffness·err + damping·err_vel, an ESTIMATE of what
PhysX applied; under osc the OSC effort target after the maxForce clip),
the per-joint peak per episode, the saturation count (|tau| ≥ 0.99 maxForce,
maxForce READ off `joint_effort_limits`), a per-joint torque-vs-depth
profile, and a per-step trace in the JSON. No run before this one logged a
joint torque (audit K1: "Was NICHT gemessen ist: Gelenkmomente").

The K1 rule, quoted from the audit, section 6: compare the peak force of the
CLEAN insertions. "Liegt OSC im Literaturband und PD nicht, ist K1
bewiesen." The literature search-phase band is 1–20 N
(`docs/reference/literature_check_contact_force_sliding_search_2026-09-03.md`);
RT-129 read 50–170 N in its clean episodes under PD (7 of 9 deep envs; F_max
lower bound 252.55 N over all 27 clean).

## Points, each PASS/FAIL on its own

1. **P1 — the code is this code, in the right mode.** `[rt_log] git:` is the
   handover commit or a descendant; `marker: tilt_insert-2026-09-03a`;
   `code marker: insertion-osc-2026-09-03a`; the script line
   `[tilt_insert] control_mode joint_pd: ...` (a) / `control_mode osc: ...
   OSC kp pos 100.0` (b) / `... 500.0` (c). The drive table: (a) NON-ZERO
   stiffness and damping on every joint, PER-JOINT DIFFERENT (the USD's own
   authored values, `stiffness=None` in `ur5e_cfg.py:212-231`); (b)/(c)
   `0.0000` / `0.0000` on every joint; maxForce `150.00 / 150.00 / 150.00 /
   28.00 / 28.00 / 28.00` in ALL THREE. `*** DRIVES NOT INERT ***` absent
   in b/c. Any traceback = FAIL.

   **DO NOT compare (a) against RT-45's `9400.5010 / 0.3780`** (corrected
   2026-09-04, user). Two reasons, both fatal to that comparison: (i) RT-45
   was `probe_assets`, a STATIC read of the AUTHORED USD value, and USD
   authors angular drive gains PER DEGREE, while this table is a PhysX
   runtime read-back PER RADIAN — the two differ by exactly 180/pi = 57.2958
   (9400.5009765625 * 57.2958 = 538609.03, read back 538609.0625; damping
   0.37800323963165283 * 57.2958 = 21.6580, read back 21.6580); (ii) RT-45
   probed ONE joint, `shoulder_pan`, and never claimed the six are equal —
   they are not (authored per degree: shoulder_pan 9400.5, shoulder_lift
   10020.9, elbow 10230.2, wrist_1/2 3940.6, wrist_3 1000.1). Judge (a) on
   POSE AND TRACKING STABILITY instead: `max joint deviation` and the
   descent's own residuals across the step-2/60/180 report blocks.
2. **P2 — Lambda and cond(J) are pose properties, not controller
   properties.** The `Lambda (env 0, at the reset pose = home pose)` MAX
   values and the `Jacobian condition number` agree across a/b/c to the
   printed precision (same pose, same PhysX). A difference is a wiring
   defect in the report, not a physics finding. Quote them once.
3. **P3 — every run completes and reports.** The line `[tilt_insert] N
   episodes, C clean, A force-aborted`, the `start tilt -> peak depth /
   force / interpen` table, the depth profile, the `joint torques` table
   (six rows) and the `torque profile over depth` block, `metrics ->
   <path>`, exit code 0. Under OSC the 900 control steps are 60 s of sim
   time (15 Hz) — fewer episodes than RT-129 (which had 51 in 15 s at
   60 Hz) is EXPECTED and not a failure; `N >= 17` (every env finished at
   least once) is the bar.
4. **P4 — THE K1 READING.** Per run, quote: `F_max LOWER BOUND from clean
   insertions: <x> N (over <C> clean episodes)`, the `peak force p50` of the
   `start tilt` table per env, and the force profile's `force max` in the
   1–35 mm bins. Then the rule: **OSC (b or c) clean peaks inside 1–20 N
   AND PD (a) clean peaks above it → K1 PROVEN.** Both above → the
   controller MODE is not the lever at these gains (then P5/P6 say whether
   saturation or the placeholder kp is). Zero clean episodes in a run → that
   run's bound is UNMEASURABLE and it says so; do not read a bound off an
   aborted run (the `WARNING: ... TRUNCATED` line).
5. **P5 — the torques, all six joints.** Per run, the six-row table:
   `maxForce`, `peak |tau|`, `(% of maxForce)`, `peak clean`, `saturated
   steps`. Quote every row. READINGS: (i) whether ANY joint saturates under
   PD during the clean insertions (audit K1: "Ob die 300 N der Effort-Deckel
   sind (150 / 28 an den Antrieben ...) oder die Kontaktphysik, ist heute
   nicht entscheidbar" — this is the number that decides it); (ii) the
   wrist joints (maxForce 28) against the shoulder/elbow (150); (iii) under
   OSC the same, plus the gravity-compensation torques the startup report
   prints next to maxForce. A joint saturated for most of the descent under
   PD is the "effort cap" answer; none saturated is the "contact physics"
   answer.
6. **P6 — kp 100 vs 500 (b vs c).** Same readings. Reading, not gate: peak
   force, clean count, depth reached, saturation. Compare BOTH with the
   `Decision (4) reading: kp <= ...` bound the startup report prints (P2 of
   RT-143): a run whose kp sits above that bound is expected to exceed
   F_search = 20 N by construction, and that is a check of the derivation,
   not a surprise.
7. **P7 — the tilt instrument still works under OSC.** `step 1 TILT
   REFERENCE: ... since latched reference p50 +0.0000 deg` and the depth
   profile's `tilt max` column in the single-degree class (RT-128 P3/P6),
   and `depth p50 > 0` for the majority of envs (RT-128 P5) in b/c. The
   OSC tilt cone (8.52 deg) must not clip the commanded ±8 deg: if `tilt max`
   at 0–1 mm reads below 8 in b/c but 8.04 in a (RT-129), the cone is
   biting on the commanded tilt and that is a finding against
   `osc_tilt_clamp_rad`'s frame, not against the task.

## Named confounds — say them in the verdict, do not resolve them there

* **Control rate.** a runs at 60 Hz (D-024), b/c at 15 Hz (placeholder). The
  scripted descent is 0.5 mm per control step, so the OSC descends four
  times slower. A slower descent ALONE lowers peak force. The discriminating
  extra run, if P4 turns on it: RT-144d = b with `--decimation 2` (both
  modes at 60 Hz).
* **Action semantics.** Under OSC the constant descend command is a creep
  (target re-anchored every physics step), under PD an integrated target
  (the wind-up the audit names). That IS the difference under test; it is
  not separable from "the controller".
* **Torque source.** `applied_torque` under joint_pd is the actuator
  model's estimate of the implicit drive, not a PhysX read-back. Saturation
  read off it is the model's clip, which uses the same maxForce.
* **The tilt cone clamps at 8.52 deg from the WORLD vertical**
  (`clamp_tilt_to_cone`, `osc_tilt_clamp_rad`), and this sweep commands
  +-8 deg. Under OSC the outer sweep points sit 0.52 deg inside the clamp,
  so any added orientation error is silently eaten; under joint_pd no such
  clamp exists. Read the ACHIEVED tilt per env, never the commanded one.

* **Physics anchors (audit K4) untouched:** solver 16/1, contact/rest
  offsets at PhysX default, unauthored inertia. Force peaks may be solver
  artefacts in ALL THREE runs alike.

## The commands (training PC, repo root, conda env active, AFTER RT-143)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: given in the handover message (read back from origin).

```
.\scripts\rt_log.ps1 RT-144a python -u scripts/tilt_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 17 --headless --tilt-axis y --tilt-sweep-deg -8 8 --upright-depth-mm 8 --stop-depth-mm 35 --f-abort 300 --max-steps 900 --control-mode joint_pd --out rt144a_pd.json
.\scripts\rt_log.ps1 RT-144b python -u scripts/tilt_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 17 --headless --tilt-axis y --tilt-sweep-deg -8 8 --upright-depth-mm 8 --stop-depth-mm 35 --f-abort 300 --max-steps 900 --control-mode osc --osc-kp-pos 100 --out rt144b_osc100.json
.\scripts\rt_log.ps1 RT-144c python -u scripts/tilt_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 17 --headless --tilt-axis y --tilt-sweep-deg -8 8 --upright-depth-mm 8 --stop-depth-mm 35 --f-abort 300 --max-steps 900 --control-mode osc --osc-kp-pos 500 --out rt144c_osc500.json
```

Optional, only if P4 needs the rate ruled out:

```
.\scripts\rt_log.ps1 RT-144d python -u scripts/tilt_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 17 --headless --tilt-axis y --tilt-sweep-deg -8 8 --upright-depth-mm 8 --stop-depth-mm 35 --f-abort 300 --max-steps 900 --control-mode osc --osc-kp-pos 100 --decimation 2 --out rt144d_osc100_60hz.json
```

Bring back: every `[rt_log]` header, the full startup report of each run
(controller block, drive table, Lambda / cond(J) / gravity-torque lines),
the whole `[tilt_insert]` tail (episode line, start-tilt table, depth
profile, joint-torque table, torque profile, capture read-offs, F_max
lower-bound line, WARNING line if any, metrics path, exit code), and the
JSON files (`rt144a_pd.json`, `rt144b_osc100.json`, `rt144c_osc500.json`,
written to the repo root).

## What this run does NOT decide

* The supervisor question of Decision (2). The numbers go WITH the question.
* `F_max` — entry (b) of the same audit derives it AFTER this measurement,
  from the clean insertion under the NEW controller, and that derivation is
  a concept-stream decision.
* The kp value — Decision (4) derives it from Lambda (RT-143 P2) and
  F_search; this run only shows what 100 and 500 do.
* Whether a POLICY inserts gently under the OSC. This is a scripted
  controller; "the action space can produce the contact phase" is what it
  can show, "the policy will" is not.
