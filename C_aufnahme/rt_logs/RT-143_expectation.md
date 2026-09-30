# RT-143 — expectation (droop and free-space tracking under the OSC; RT-46 and RT-84 forms)

Written 2026-09-03 on the dev laptop, BEFORE any run. `/rt-check` judges the
log against THIS file and nothing else. Spec: `docs/decisions_inbox.md`
(branch `p4-konzept`), entry "Audit 2026-09-03 (a)", Decision (1)–(6);
reason: `docs/reference/konzept_audit_proxy_vs_real_2026-09-03.md` K1 and
section 6.

## What changed (built, offline-checked, NOT run on Isaac)

`cfg.control_mode = "osc"` is the new default: the action is a 6-D pose
delta, the target is `current pose + clamped delta` re-anchored EVERY physics
step (Factory form), tracked by Isaac Lab's `OperationalSpaceController`
(`pose_abs`, impedance `fixed`, full inertial decoupling, gravity
compensation ON with gravity ON, nullspace `none`). The PhysX drives are
inert under it (stiffness AND damping 0 on all three actuator blocks;
maxForce, limits, armature, friction stay USD). Two clamps on the target: the
leading tool point inside a ±0.05 m box around the pocket entrance
[placeholder, Factory], the tool axis inside an 8.52 deg cone from vertical
(CAD). Yaw free. Policy rate 15 Hz (decimation 8) [placeholder, Factory];
the episode keeps its 256-step count, so it is 17.07 s under OSC. kp position
100 / rotation 30 [placeholders], damping critical. `control_mode =
"joint_pd"` is the old D-108 chain, kept as the PD half of RT-144.

Offline evidence (laptop, 2026-09-03): `check_insertion_math` 214 checks +
counter-proof PASSED, `check_env_wiring` 156 + counter-proof PASSED,
`check_scripted_insert` 108 / 56 mutations PASSED, `selftest_checks
--offline` PASS. NOTHING here has run on Isaac; the first Isaac contact of
the OSC path is RT-143a.

## What this measures

1. **Droop (RT-46 form)** — zero actions at the HOME pose. Under the OSC a
   zero delta means "target = current pose" every physics step, so the only
   things holding the arm are the gravity compensation (from PhysX) and the
   critical damping. Any error in the compensation integrates into a DRIFT
   that nothing corrects. RT-46 read the flange standoff at 0.317026 m
   against 0.3170 nominal at t = 3.000 s under the USD drives; this is the
   same reading under the new controller.
2. **Free-space tracking (RT-84 form)** — `scripted_insert --free-space
   --aim-sweep x`: hold the part above the opening, command a lateral offset
   of the play's order, read the steady residual against `PLAY_X / 2`
   (0.2938 mm). RT-84 read p95 0.0461 mm under PD. Decision (4) names this
   the bandwidth CHECK that comes after the gains, not their source.

Both at kp 100 (cfg default) and kp 500 (the second placeholder run).

## Points, each PASS/FAIL on its own

1. **P1 — the code is this code, and the drives are inert.** `[rt_log]
   git:` is the commit from the handover (or a descendant); the startup
   report prints `code marker: insertion-osc-2026-09-03a` and the block
   `--- controller (inbox entry 'Audit 2026-09-03 (a)') ---` with
   `control_mode:   osc`. The drive table reads stiffness `0.0000` and damping
   `0.0000` on ALL SIX joints and maxForce `150.00` on the first three /
   `28.00` on the wrist three (RT-45's USD values). The line `*** DRIVES NOT
   INERT ***` must be ABSENT. Any traceback = FAIL (an import-time error is a
   FAIL of this point).
2. **P2 — Lambda and the condition number print.** Lines `Lambda (env 0, at
   the reset pose = home pose): translational eigenvalues [...] kg, MAX
   <x> kg; rotational ...` and `Jacobian condition number (env 0, tool_link):
   <c>` with finite numbers, plus `Decision (4) reading: kp <= ... = <k>`.
   QUOTE all three numbers; they are the inputs to the kp derivation and to
   Decision (5)'s reading. No threshold — these are readings. (`UNREAD --`
   on either line is a FAIL of this point.)
3. **P3 — the arm holds in free air (droop, RT-143a/b).** With
   `--print-force`, every `[zero_agent] step` line must read `force p50` and
   `max` below 1 N (the L8 free-air tolerance of `check_seated_success.py`,
   the threshold zero_agent already uses). Quote `Z standoff (flange)` at
   steps 2, 60 and 180 (the report prints `t = ... s`; under OSC step 180 is
   12.0 s, not 3.0 s) and `max joint deviation` at each, next to RT-46's
   0.317026 m. The DRIFT `standoff(180) - standoff(2)` is a READING with no
   threshold — it is what Decision (2) must be answered with.
4. **P4 — the tracking verdict (RT-143c/d).** `[scripted_insert] H1 steady
   error p95 ...` and the verdict `H1_TRACKS` (p95 below the 0.2938 mm
   tolerance, `PLAY_X / 2`). `H1_MISSES` is a FAIL of this point;
   `H1_RESET_FIRED` means an episode ended inside the run and the point is
   VOID (say so; 250 steps < 256 must hold, the script refuses otherwise).
   Quote settle step and steady max per env against RT-84 (settle 0–1,
   p95 0.0461 mm).
5. **P5 — kp 100 vs kp 500.** Same readings, both values. Reading, not gate:
   which one drifts less (P3) and settles faster (P4). Do NOT derive kp from
   this — Decision (4) derives it from Lambda and F_search, and P2 prints
   that bound; quote it next to 100 and 500.

## Named confounds this run carries

* **The policy rate is 15 Hz here and 60 Hz in RT-46/RT-84.** The scripted
  commands are per control step (2 mm clip), so the free-space approach is
  four times slower in wall-clock terms. The residual is a steady-state
  reading and does not depend on it; the settle STEP does. `--decimation 2`
  exists on `scripted_insert.py` for a same-rate re-run if P4 turns on it.
* **The delta is re-anchored every physics step**, so a constant scripted
  command is a creep, not a setpoint — the proportional lateral term still
  converges (error → 0 → delta → 0).
* **`gravity_compensation` reads PhysX's `get_gravity_compensation_forces`
  for the WHOLE articulation** including the welded tool and part. Whether
  that is exact for a welded fixed-joint chain is the thing P3 measures.

## The commands (training PC, repo root, conda env active)

The clone is the NEW folder `Phase3_Implementierung_v2` on branch `p4-rl-code`
(sparse clone, assets copied, editable install re-pointed -- see
`HANDOFF-RL.md` § Frozen). Then the usual pull:

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: given in the handover message (read back from origin).

```
.\scripts\rt_log.ps1 RT-143a python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --home-pose --print-force --max-steps 200
.\scripts\rt_log.ps1 RT-143b python -u scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless --home-pose --print-force --max-steps 200 --osc-kp-pos 500
.\scripts\rt_log.ps1 RT-143c python -u scripts/scripted_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 16 --headless --free-space --aim-sweep x --max-steps 250
.\scripts\rt_log.ps1 RT-143d python -u scripts/scripted_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 16 --headless --free-space --aim-sweep x --max-steps 250 --osc-kp-pos 500
```

Bring back: all four `[rt_log]` headers, the full startup report of each run
(the controller block, the drive table, the Lambda / condition-number lines,
the three `-- env 0 --` blocks at steps 2/60/180), every `[zero_agent] step`
line of a/b, and the whole `[scripted_insert]` tail of c/d (H1 table, steady
error, verdict, exit code).

## What this run does NOT decide

* Nothing about contact: that is RT-144.
* Not the supervisor question of Decision (2) — gravity ON with inert
  drives breaks the letter of D-105; the numbers here go to the supervisor
  WITH the question.
* Not the values of the placeholders. Every OSC number prints under `***
  PLACEHOLDER VALUES IN USE ***` and travels in `demo_metrics.json` /
  the metrics JSON as such.
