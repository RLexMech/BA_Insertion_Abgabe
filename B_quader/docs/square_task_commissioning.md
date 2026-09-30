# Square task — commissioning protocol on the training machine

**Status: checkpoints 1–4 and 6–9 verified on the training machine on
2026-08-06. The curriculum was run to its end on the same day: stage 0
(b = 45) at 99.9 %, b = 36 at 100 % in replay, and the D-029 target rung
(b = 32, 1.0 mm clearance, ±3.96° yaw window) at 99.15 % — see §7. Checkpoint 5
(reset distribution) is open.** This document records what was measured, with
which command, so the results are read from numbers rather than from prose.
It covers the branch `square-peg-insertion` (D-029 geometry pivot, D-033
generator and pocket curriculum). Everything here supersedes the UNVERIFIED
labels that branch carried until this session.

Machine and toolchain: Windows 11 (26200), Ryzen 9 5900X, RTX 3080, driver
591.86, Isaac Sim 5.1.0-rc.19 / Isaac Lab 2.3.2, PyTorch 2.7.0+cu118, physics
step 1/120 s, control step 1/60 s (D-024). Code marker on every report:
`square-peg-2026-08-05a-d033`.

## 1. Summary

| # | Checkpoint | Decisive measurement | Verdict |
|---|---|---|---|
| 1 | Table generated, not imported | 10/10 self-checks at b = 45 and b = 32; contract bbox 0.500 × 0.600 × 0.755 m, max. error 6.1e-09 m; 0 composition arcs | PASS |
| 2 | Peg authored | `local_bbox_size_m` 0.030 × 0.030 × 0.050 m (not 0.042 → no 45° mesh rotation); 13/13 checks | PASS |
| 3 | Containment gate | Gate `closed` at the home pose in every env; both asset checks PASS against their generator sidecars | PASS |
| 4 | Observation | 21-dim layout; channels 19:21 reproduce (cos 4φ, sin 4φ) to four digits | PASS |
| 5 | Reset distribution ρ₀ | Only 2 envs printed; in-sim uniformity of φ not established | **open** |
| 6 | Reward and gate algebra | 41/41 under real PyTorch after the float64 fix | PASS |
| 7 | Passability, b = 32 | Centred peg reaches 25 mm at 1.05× its own contact-free noise; yaw control reacts at 6.30×, lateral at 4.24× | PASS |
| 7 | Passability, b = 45 | Lateral control 3.92× at the derived 10.5 mm offset; yaw correctly inactive | PASS |

## 2. Assets (checkpoints 1 and 2)

The table is now composed from nine axis-aligned boxes, each a native PhysX
primitive collider (D-033). Both curriculum rungs were generated:

```
python scripts\author_tisch_square.py --pocket-side-mm 45
python scripts\author_tisch_square.py --pocket-side-mm 32
```

Measured at b = 32: plate z −0.055 … 0.000 m (55 mm), pocket floor
−0.055 … −0.035 m, so the pocket is 35 mm deep measured from the opening
plane; opening ±0.016 m in x and y, i.e. **1.00 mm clearance per axis**
against the 30 mm peg. At b = 45 the clearance is 7.50 mm per axis. The
overall bounding box is identical at both rungs, 0.500 × 0.600 × 0.755 m,
and the pocket sits deliberately off-centre in y (75 mm from the south edge),
which places it at world y = −0.225 once spawned.

`verify_fixture_usd.py` measured the contract bbox with a maximum absolute
error of 6.1e-09 m against nominal — five orders below the 1 mm tolerance —
reported `orientation: upright`, z range −0.755 … 0.000 m, and **zero
composition arcs**. The generated asset is therefore self-contained; the
external-reference failure mode of the CAD era (PROBLEMS.md, 2026-07-26)
cannot recur on this path. `verify_fixture_spawn.py` measured all four spawn
routes (direct, referencing prim, parent, Isaac Lab spawner) as
`metres (correct)` with the plate top at z = 0.755 m and the legs on the
ground plane, which closes the unit-scale failure mode as well.

The peg (`author_peg_ur10e.py`) passed 13/13. The decisive field is
`local_bbox_size_m` = 0.030 × 0.030 × 0.050 m: a 4-segment cylinder, the
defect D-029 warns about, would have produced 0.042 m across the diagonal.
The authored physics is internally consistent rather than merely accepted:
mass 0.04680 kg over 45 cm³ gives 1040 kg/m³ (ABS), and the principal
inertias 1.3260e-05 and 7.0202e-06 kg m² are exactly m/12·(a² + L²) and
m/12·2a² — the cuboid formulae, not the cylinder ones. The centre of mass
lies at half length and the geometry spans the flange plane to +50 mm.

## 3. Gate, observation and reset (checkpoints 3–5)

From the startup report of `zero_agent.py --num_envs 16`:

- Both asset checks PASS and cite their sidecars, including the fixture check
  introduced by D-033 (`tisch_square_b32.author.json`,
  `author_tisch_square-2026-08-05b`). A table generated for the wrong rung
  would be caught here rather than silently trained on.
- The gate is `closed` at the home pose in every environment, with its reason
  printed (corners outside, depth −0.087 m, alignment +0.9995).
- Peg kinematics are exact: flange to peg origin 0.000000 m, tip at
  +0.050000 m along the tool axis.

**Checkpoint 4 was verified algebraically, not through the (1, 0) special
case.** The reset noise is always active, so no pre-noise snapshot exists.
The C4 encoding was instead checked against the reported φ:

| Env | φ (folded) | 4φ | expected (cos, sin) | reported |
|---|---|---|---|---|
| 0 | −42.484° | −169.94° | (−0.9847, −0.1745) | (−0.9846, −0.1748) |
| 1 | −17.782° | −71.13° | (+0.3233, −0.9463) | (+0.3235, −0.9462) |

The residual is the printing of φ to three decimals. φ tracks the `wrist_3`
deviation (42.10° and 17.63°), which confirms in simulation the pure-yaw
actuator claim that `reset_noise_spread.py` derived by forward kinematics.

**Checkpoint 5 remains open.** The startup report prints environments 0 and 1
only; two samples say nothing about uniformity of φ over ±45°. The evidence
so far is the offline sampler (20 000 samples, p95 = 42.7°). The histogram
should be taken from the stage-0 smoke run, where thousands of resets occur.
This is recorded as open rather than inferred from the sampler alone.

## 4. Gravity droop is time-dependent (D-026)

Four report lines fell outside their stated expectations in the
`zero_agent.py` run: Z standoff 0.1372 m against 0.150 ± 0.005, insertion
depth −0.0873 m against −0.100 ± 0.005, XY offset 0.0112 m against ≤ 0.005,
tip Chebyshev 0.0101 m against ≤ 0.0010. The shortfall of 12.8 mm matches the
13.4 mm droop measured in D-026 to within 0.6 mm.

That the cause is compliance and not sampling follows from the report itself:
`joint_pos commanded` is identical across environments and therefore excludes
the reset noise, while the deviations at `shoulder_lift` (−0.025 rad) and
`elbow` (−0.016 rad) exceed the 0.01 rad noise bound on those joints.

The `verify_peg_passability.py` run then showed the same quantities *inside*
their expectations — standoff 0.1484 m, XY offset 0.0026 m. The difference is
the sampling time: that report prints at t = 0.033 s, the `zero_agent` one at
t = 1.0 s and t = 3.0 s. **The droop develops within the first second and is
then static** (the t = 1 s and t = 3 s values are identical). The "expect"
annotations in the startup report are formulated for the gravity-free nominal
pose and should be labelled as such.

Consequences: the paired passability comparison is unaffected, since all three
passes carry the same droop. For training, the tip starts up to 16 mm off the
pocket centre while the pocket half-side is 16 mm, so the descent carries a
lateral correction task; this enters the time-budget measurement of
checkpoint 8. Per D-026 and D-014 the answer to the droop is OSC, never a
reward edit.

## 5. Reward algebra and one harness defect (checkpoint 6)

`check_demo_reward_math.py` first reported **37/41** under real PyTorch after
41/41 under the numpy stand-in on the dev PC. The four failures were exactly
the boundary-*inclusive* success cases, while all four just-outside negative
controls kept rejecting correctly — a pattern that indicates the test
fixture, not the gate.

All four cases use the pose literal `z = 0.730`. Real PyTorch defaults to
float32 and stores that as 0.7300000190734863, so the depth at the threshold
came out 1.9e-08 m short and `depth >= 0.025` turned False. The shortfall was
measured, not inferred, and lies five orders of magnitude below the 1.0 mm
clearance the gate has to discriminate against. The fix sets the real-torch
path to float64 so both backends test the same algebra rather than the same
rounding; no threshold and no assertion was changed. Re-run on the training
machine: **41/41**. Logged in PROBLEMS.md with the resulting rule.

## 6. Passability (checkpoint 7)

`verify_peg_passability.py` at `PROXYTASK_POCKET_SIDE_MM=32`, three passes
differing only in lateral position and yaw:

| Pass | Ratio to its own contact-free noise | First reaction |
|---|---|---|
| centred | 1.05× | none; clean to 27 mm |
| 4 mm lateral offset | 4.24× | −1 mm (plate top) |
| 45° yaw | **6.30×** | +3 mm |

`reached_success_depth: True` — the centred peg passes the 25 mm success
depth through 1.0 mm of clearance without touching anything. The three passes
share an identical contact-free baseline of 0.00272894 rad, which excludes the
configuration-change explanation that produced a wrong verdict on
2026-07-26. The verdict holds for any reaction factor between 1.05 and 4.24;
2.0 was used, i.e. the middle of the interval rather than its edge.

The 45° yaw control produces the *strongest* reaction of the three. This is
the measurement D-029 risk 1 required: the squareness is physically effective,
and `yaw_control: active` rather than `SQUARENESS_ABSENT`. That the reaction
begins at +3 mm rather than at 0 mm is consistent with the same joint
compliance that produces the droop — the controller continues to command
downward while torque builds.

### 6.1 Stage 0 and a defect in the negative control

The same run at `PROXYTASK_POCKET_SIDE_MM=45` first returned
**COLLISION_ABSENT**. The cause was the control, not the collider: the offset
was fixed at 4 mm while stage 0 has 7.5 mm of clearance, so the offset peg
(corner Chebyshev 4 + 15 = 19 mm against a 22.5 mm half-pocket) fits the
opening and cannot reach a wall. The yaw control deactivates itself at that
rung by design, so no control discriminated — and the script nevertheless
issued a confident verdict.

The discriminating measurement was one run with `--offset 0.010`: 3.23×, first
reaction at −1 mm, PASSABLE. The offset is now derived as clearance + 3 mm.
The margin is measured rather than chosen, and the measurements show that
overlap, not pocket width, sets the reaction:

| Rung | Offset | Overlap | Reaction |
|---|---|---|---|
| b = 45 | 10.0 mm | 2.5 mm | 3.23× |
| b = 45 | 10.5 mm (derived) | 3.0 mm | 3.92× |
| b = 32 | 4.0 mm (derived) | 3.0 mm | 4.24× |

At equal overlap the reactions agree to within 8 % across a 40 % difference in
pocket width, and within a rung the reaction rises monotonically with overlap.
An offset inside the clearance now yields INCONCLUSIVE instead of
COLLISION_ABSENT, and the stability interval is null when no control
discriminates. Re-run after the change: b = 45 gave 3.92× and PASSABLE, and
b = 32 reproduced the pre-change values bit-identically. Logged in PROBLEMS.md.

### 6.2 The gate is open at the home pose at stage 0

At b = 45 the startup report shows `gate: OPEN` with the tip 100 mm above the
plate, because the peg lies inside the 45 mm pocket footprint even at 14° of
reset yaw. This does not violate the reward semantics: every consumer of
`gate` is additionally conditioned on depth, which is negative above the
plate. `depth_gated` cannot raise `max_depth` above 0, `success` fails on the
depth term, and `misplaced_penalty` is zero above the plate in either branch.
The acceptance criterion "gate CLOSED at the home pose" was formulated for
b = 32, where the reset yaw pushes the corners out, and should read: closed at
the target rung, and free to be open at wide rungs because depth blocks any
payout.

## 7. Stage 0 trained (checkpoints 8 and 9)

Run `2026-08-06_00-58-29_s30b45`, 1024 environments, 500 iterations,
215 760 episodes:

| Quantity | Value |
|---|---|
| Success rate, trailing 2000 episodes | **99.9 %** |
| Success rate, cumulative | 94.6 % |
| Episodes to the 90 % threshold | 17 974 |
| Mean maximum depth | 29.5 mm |

The stage-0 criterion in HANDOFF.md is a recent success rate above 0.9; the
measured value is 0.999. The mean maximum depth of 29.5 mm sits above the
25 mm success threshold and below the 35 mm pocket bottom, so the policy
inserts rather than bottoming out. The run directory carries the rung
(`_s30b45`) and the metrics file records `pocket_side_m: 0.045`. The 17 974
episodes to threshold are the sample-efficiency reference against which later
rungs are read.

**What this result does not show.** At b = 45 the free-yaw window is the full
±45°: the 42.43 mm diagonal fits the 45 mm opening, so the peg enters at any
orientation. Stage 0 is therefore, in physical terms, the *cylinder* task
performed with a square peg — the cross-section's symmetry has no consequence.
The replay confirms this: the policy aims slightly but does not consistently
align to one of the four insertable orientations, which is the expected
behaviour when only the reward term −w_yaw(1 − cos 4φ)/2 encourages alignment
and geometry does not require it. The 99.9 % therefore validate approach,
descent and depth control, not the square-specific capability.

For the same reason checkpoint 5 gains nothing here: every reset yaw fits, so
a high success rate says nothing about the coverage of ρ₀.

Yaw first binds below the peg diagonal of 42.43 mm. The free-yaw window per
rung, from the curriculum table in `proxytask_tasks_cfg.py`:

| Pocket b | Clearance | Free-yaw window |
|---|---|---|
| 45 mm | 7.5 mm | ±45° (unbounded) |
| 40 mm | 5.0 mm | ±25.5° |
| 36 mm | 3.0 mm | ±13.1° |
| 32 mm (target) | 1.0 mm | ±3.96° |

Yaw first binds below the peg diagonal, which makes the rungs below 42.43 mm
the genuine test of the pivot. Rungs are not comparable head-to-head; each is
its own resume run with its own metrics file.

### 7.1 The curriculum was shortened

The rung at b = 40 was skipped. The criterion was the share of episodes whose
reset yaw falls outside the free-yaw window and which therefore cannot be
solved without rotating; the reset yaw is uniform over ±45°:

| Pocket b | Clearance | Yaw window | Episodes forcing rotation |
|---|---|---|---|
| 45 mm | 7.5 mm | ±45° | 0 % |
| 40 mm | 5.0 mm | ±25.5° | 43 % |
| 36 mm | 3.0 mm | ±13.1° | 71 % |
| 32 mm | 1.0 mm | ±3.96° | 91 % |

At b = 40 the peg still enters without any rotation in 57 % of episodes, so
that rung largely repeats the preceding task. The ladder run was therefore
45 → 36 → 32 rather than the 45 → 40 → 36 → 32 recorded in
`proxytask_tasks_cfg.py`.

### 7.2 b = 36

Resumed from the stage-0 checkpoint, iterations 500 to 1498. The training
metrics file was destroyed before it was read: `play.py` wrote the replay's
metrics over it (see PROBLEMS.md and the fix in `play.py`). What survives is
the replay of the final policy, `model_1498.pt`, at b = 36: **100 % over 751
episodes** at 29.6 mm mean maximum depth, measured without exploration noise.
The sample-efficiency figure for this rung is lost.

### 7.3 b = 32 — the D-029 target rung

Run `2026-08-06_02-25-24_s30b32`, resumed from the b = 36 policy, 1024
environments, 264 759 episodes:

| Quantity | Value |
|---|---|
| Success rate, trailing 2000 episodes | **99.15 %** |
| Success rate, cumulative | 73.9 % |
| Episodes to the 90 % threshold | 77 789 |
| Mean maximum depth | 28.8 mm |

This is the geometry of D-029: 1.0 mm of clearance per axis and a ±3.96°
free-yaw window. Because 91 % of episodes start outside that window, a 99.15 %
success rate is not reachable without systematically rotating into one of the
four insertable orientations. Unlike stage 0, this run therefore demonstrates
the square-specific capability.

The 77 789 episodes to threshold, against 17 974 at stage 0, quantify the cost
of the tightening — a factor of 4.3, and that despite starting from an already
competent b = 36 policy rather than from scratch. The cumulative rate of 73.9 %
reflects the long sub-threshold phase.

**D-029 risk 5 did not materialise.** The 13 mm gravity droop is thirteen times
the 1.0 mm clearance and was ranked the first suspect should training fail,
with D-014/OSC as the remedy. With joint-delta actions on the shipped PD gains
the policy commands through that droop to 99.15 %. OSC was not required for
this task. This does not retire D-014 — the droop remains real and a
force-controlled variant may still be preferable — but the premise that it
blocks insertion at 1.0 mm clearance is refuted by measurement.

## 8. Open items from this session

- Checkpoint 5 (ρ₀ uniformity) is unverified in simulation; take the
  histogram from the stage-0 smoke run.
- The `expect` annotations in the startup report describe the gravity-free
  nominal pose and should say so.
- The startup report's acceptance line "gate CLOSED at the home pose" applies
  to the target rung only; see §6.2.
- `ur10e_peg_s30.candidate.usd` remains beside the peg asset as an
  intermediate file, together with the stray CAD-era files listed in
  HANDOFF.md.
- The executable copy on the training machine is now a git clone of this
  repository rather than the manually synchronised folder described in
  CLAUDE.md § "Two-repo sync workflow"; that section is out of date.
