# Demo sprint — course of work and results

**Status: proposed / partially verified. Nothing here is a baseline for
increments 1–3.** This branch (`demo-insertion-sprint`, cut from
`increment-2-verified` = `90621a8`) is a deliberately simplified detour taken
on the night of 2026-07-26/27 to reach a first learning curve before a
deadline. It changes the peg geometry, the action space, the observation and
the reward against the increment line, so none of its numbers transfer to
increments 1–3, and none of increments 1–3's acceptance criteria apply here.

The controller decision D-014 (OperationalSpaceController) is **not**
implemented on this branch. Actions are joint-position deltas on the shipped
PD gains, so the 13.4 mm gravity droop measured in D-026 is present throughout
and the policy has to command through it.

## 1. What was built, in order

| Step | Change | Where |
|---|---|---|
| 1 | Separate clone + branch from `increment-2-verified` | `Keine 3`, `demo-insertion-sprint` |
| 2 | Peg Ø28 → Ø25 mm (radial clearance 1.0 → 2.5 mm) | `proxytask_tasks_cfg.py` |
| 3 | Joint-delta actions applied (0.02 rad/step, clamped to soft limits) | `proxytask_env.py` |
| 4 | 19-dim observation, joint velocities as finite differences | `proxytask_env.py` |
| 5 | Dense staged reward + success metric to disk | `proxytask_env.py` |
| 6 | Four reward defects found and closed (§3) | `proxytask_env.py` |
| 7 | Peg diameter parameterised as the single curriculum knob | `proxytask_tasks_cfg.py` |
| 8 | Per-run metrics, trailing window, sample-efficiency figure | `proxytask_env.py`, `compare_runs.py` |

Deliberately **not** done, and why: no OSC (D-014 belongs to its own
increment), no force/torque reward terms (this branch carries no wrench
channel, and the 50 N limit is still open), no contact sensor, no
sampling-based initial-state curriculum in the IndustReal sense — only a
uniform joint-space reset noise.

## 2. Measured on the training machine

Each figure below comes from console output of a run on the training machine.

**Peg asset, Ø25 mm rung.** `author_peg_ur10e.py`: 13/13 checks PASS,
`local_bbox_size_m` `[0.025, 0.025, 0.050]`, mass 0.02553 kg, Steiner check
consistent. `verify_peg_passability.py --offset 0.006`: verdict **PASSABLE** —
the centred peg reached 27 mm at 1.05× its own contact-free noise while the
6 mm offset control rose to 6.55×, so the verdict holds for any reaction
factor between 1.05 and 6.55 and is not threshold-dependent.

**Finite-differenced joint velocities work.** Resting joint speed reads
9.1e-05 rad/s at t = 1 s and exactly 0.0 at t = 3 s, against the raw solver
channel's standing ≈ 0.05 rad/s phantom (D-030 addendum 2). The
open-decision-5 criterion (< 1 mm/s at the flange) is met by a wide margin
**without** changing the solver configuration, so this branch's physics is
bit-identical to the configuration increments 1–3 were measured under.
This is the cheapest known answer to that defect and is worth carrying back.

**Asset/constants consistency check.** The startup report compares the
authored mesh against the configuration and prints
`asset check: PASS — authored geometry matches cfg`. It caught nothing here
only because it was ported *after* the first run; the mechanism itself comes
from increment 3 (`4908adb`, `708062e`).

**Gravity droop is the dominant lateral error.** Tip XY offset to the bore
after settling: 13.1–16.0 mm across environments, consistent with D-026's
13.4 mm plus reset noise. Against 2.5 mm of clearance this is roughly six
times the gate, against the Ø28 rung's 1.0 mm it is thirteen times, and
against a Ø29 rung's 0.5 mm it would be twenty-seven times.

**float32 margin at scale** (computed, `reset_noise_spread.py` and the grid
arithmetic): at 4096 environments with 3.0 m spacing the grid is 64×64, so env
origins reach 96 m from the centre, where one float32 ULP is 11.4 µm. That is
1.1 % of the Ø28 rung's 1.0 mm gate and 2.3 % of a Ø29 rung's 0.5 mm. This
answers the recheck D-030 addendum asked for; it is not a factor yet, but the
margin is no longer three orders of magnitude.

**What reset_joint_noise means in millimetres** (`reset_noise_spread.py`,
20 000 samples, verified FK):

| noise [rad] | per joint | lateral mean | p95 | max | tilt p95 |
|---|---|---|---|---|---|
| 0.01 | ±0.57° | 4.3 mm | 7.7 mm | 11.4 mm | 1.2° |
| 0.02 | ±1.15° | 8.6 mm | 15.4 mm | 21.9 mm | 2.3° |
| 0.05 | ±2.86° | 21.4 mm | 38.5 mm | 53.4 mm | 5.8° |
| 0.10 | ±5.73° | 42.8 mm | 76.8 mm | 110.7 mm | 11.6° |

Side finding: this confirmed the DH-frame caveat D-025 left open. The DH base
frame is rotated by π about z against the USD `base_link` frame. With the
rotation applied the calculation reproduces the training-machine report
exactly (0.000 mm lateral, 100 mm above the plate); without it the home tip
lands mirrored, 830 mm from the bore.

## 3. The main result: four reward defects, each found by watching the robot

This is the part with thesis value. All four were **invisible in the reward
curve** and visible only in the simulation, and each was the rational optimum
of the reward as written at the time.

1. **Diving under the plate.** `depth` was `plate_top_z − tip_z` with no upper
   bound. Ten millimetres below the plate it reads 65 mm — past the 25 mm
   success threshold — while the distance to the bore entrance is 55 mm,
   *smaller* than the 83 mm at the home pose. That paid the full depth reward
   plus the success bonus, about 16.5 points, for going under the table. Fix:
   the gate became a bore-*volume* test, and depth outside it is charged
   (`w_misplaced`), because closing the gate alone left the space merely
   unrewarded rather than unattractive.
2. **Suicide.** Terminating on leaving the arena would have been worth far
   more than finishing an episode — escaping a 200-step remainder of a
   negative reward beats any bonus in the task. Fix: the out-of-bounds
   termination is priced at exactly the per-step cost the remaining steps
   would have accrued, so leaving is neither a shortcut nor a punishment.
3. **Parking across the bore.** A distance-only approach term is maximised by
   putting the tip on the hole *from any direction*. Every environment in a
   1024-env run converged on reaching in sideways with the peg lying almost
   flat, scoring an approach term of about zero — the best value reachable
   without inserting — and stopped there. Leaving that pose costs before it
   pays, because standing the peg up moves the tip away from the bore. Fix:
   an alignment term plus an alignment condition in the gate. Recorded frames
   show the arm *does* hold a downward orientation mid-episode and then
   rotates into the flat pose, so this was a preference, not an inability.
4. **A transposed rotation matrix**, caught before it ever ran, by the offline
   check (`check_demo_reward_math.py`): the ported batched quaternion helper
   returned Rᵀ, which would have rotated every peg tip wrongly while still
   producing entirely plausible positions.

Defects 1–3 correspond to the exploits `research_cylindertask.md` §6 lists,
and defect 3 is its exploit no. 1 ("policy learns to push sideways instead of
inserting") in a form the source does not describe: it names a lateral force
penalty as the fix, which needs a wrench channel; here it was answered
structurally, by separating the lateral and height terms and gating on
orientation.

The reward follows the staged structure D-004 selected (approach, alignment,
insertion, success) and **not** Factory's keypoint reward, which D-004 rejects
citing IndustReal's 1.8–54.2 % success for keypoint variants against 88.6 %
for SDF. Two documented deviations from §6 remain: no force/torque penalties
(no wrench on this branch), and dense terms summing to roughly 0.2 per step
against §6's guideline of 1–10.

`scripts/check_demo_reward_math.py` covers all of this offline — 30 checks,
no Isaac and no PyTorch needed — and each exploit above has an assertion.

## 3a. The reward function as it now stands

Implemented as a free-standing `@torch.jit.script` function outside the env
class, which is the pattern the earlier increments left for the first real
reward. Every term below is per step; the peg tip position, the insertion
depth and the gate all come from one shared geometry helper, so the reward,
the termination and the observation cannot disagree about where the peg is.

**The gate** is the condition under which the peg counts as *in the hole*, and
three things must hold at once: laterally within the radial clearance,
`depth ≤ bore_depth + 1 mm`, and `alignment ≥ 0.99`. Each of the three closes
a specific exploit from §3 — the lateral test alone allowed tunnelling beside
the bore, the depth bound alone was what the under-the-plate dive abused, and
the alignment condition rules out a peg lying across the hole. The 1 mm
tolerance on depth covers float32 (the bore bottom itself computes to just
over 30 mm and was being excluded) and the small penetration PhysX permits at
the stop.

| Term | Formula | Weight | Why this weight |
|---|---|---|---|
| Approach | `−w · ‖tip − bore‖` | 2.0 | Starting value. Gives ≈ −0.17 per step at the home pose, so it guides without dominating the depth term. Not tuned. |
| Depth progress | `+w · Δ(max depth so far)`, gated | 100.0 | Scaled so the full 25 mm insertion is worth 2.5 points, i.e. an order above the per-step approach cost. Potential-based on the running maximum, which is what makes gate-farming worthless. |
| Success | `+w` once, on `depth ≥ 25 mm` and gate | 10.0 | Four times the total depth reward, so success dominates partial insertion without swamping the shaping. Well below §6's suggested 100 because the episode terminates on success here. |
| Alignment | `−w · (1 − cos θ)` | 2.0 | Set equal to the approach weight deliberately: the flat parked pose then costs 2.0 per step against an approach gain of at most ≈ 0.17, so the local optimum of §3.3 is destroyed by a factor of ten. |
| Misplaced depth | `−w · depth outside the gate` | 20.0 | Chosen so the space under the plate (≈ 55 mm of depth) costs 1.43 per step against 0.16 at the home pose — worse by a factor of nine, not merely unrewarded. |
| Action | `−w · ‖a‖²` | 0.01 | Small by design: it exists to discourage flailing, and at 6 saturated joints it costs 0.06, a third of the approach term. |
| Out of bounds | `(recurring terms) × remaining steps` | — | Not a weight but a price: exactly what the rest of the episode would have cost. Derived, not chosen, and that is what makes early termination neutral instead of profitable. |

**Honest status of these numbers.** They are *reasoned starting values*, not
an empirical optimum. None was obtained by a sweep. Each was derived by
computing what the competing behaviour would score and setting the weight so
the intended behaviour wins by a stated margin — the margins above (ten-fold,
nine-fold, four-fold) are the actual design criterion. That the first
configuration trained successfully means the margins were sufficient, not that
they are optimal, and no claim is made that a different set would be worse.

### Why the reward was changed, in order

The sequence matters more than the endpoint, because each change was forced by
an observation rather than chosen in advance.

1. **Initial design** — approach, gated depth progress, success bonus, action
   penalty. The two exploits anticipated in advance were gate-farming
   (answered by the potential-based maximum) and success by tunnelling
   (answered by requiring the lateral gate for success).
2. **After the first training run** — the arm searched under the table. Cause:
   unbounded depth below the plate scored better than the home pose *and*
   counted as success. Added the bore-volume gate; then found that this alone
   only made the state unrewarded while the approach term still preferred it,
   so added `w_misplaced`.
3. **Same change, second-order effect** — terminating on out-of-bounds created
   a suicide incentive worth more than any bonus. Priced the termination.
4. **After the second training run** — every environment parked across the
   bore. Cause: a distance-only approach term has no notion of direction.
   Added the alignment term and the alignment condition in the gate.

Steps 2–4 were all discovered by **watching the simulation**, not by reading
the reward curve, which rose smoothly throughout. That is the single most
transferable lesson from this sprint.

## 3b. Hyperparameters

### Environment

| Parameter | Value | Origin |
|---|---|---|
| Control rate | 60 Hz (`decimation = 2`, `dt = 1/120`) | Inherited, D-024 |
| Episode length | 4.0 s = 240 steps | Reduced from 5.0 s after observing the arm spend the tail searching. The descent is ≈ 125 mm (100 mm standoff + 25 mm insertion), which 240 steps cover with room. |
| Action space | 6 joint-position deltas | The NVIDIA UR10-Reach pattern. Chosen over OSC (D-014) purely to remove the unmeasured-gains risk on a deadline. |
| `action_scale` | 0.02 rad/step | ≈ 1.15° per joint per step. Two steps suffice to command through the 13.4 mm droop, so the policy has authority without being able to jump the clearance in one step. Not swept. |
| Observation | 19-dim | D-030's layout minus the wrench and contact channels, which have no consumer without force terms. |
| `reset_joint_noise` | 0.01 rad | ≈ 4.3 mm lateral scatter at the tip (measured, §2). Deliberately small for the first runs; **this is the value the robustness sweep exists to raise**, and results at this setting do not demonstrate generalisation. |
| Solver | unchanged (`enable_external_forces_every_iteration = False`) | Kept identical to the configuration increments 1–3 were measured under, so the physics stays comparable. The velocity defect was answered by finite differences instead. |

### PPO (rsl_rl)

Everything not listed was left at the template's value; only three fields were
changed, each for a stated reason.

| Parameter | Value | Origin |
|---|---|---|
| `actor/critic_hidden_dims` | `[128, 128]` | **Changed** from `[32, 32]`, which was sized for the template's Cartpole task. Insertion needs more capacity than a 4-dim balancing problem. Not swept — the smaller network was never tried on this task. |
| `actor/critic_obs_normalization` | `True` | **Changed** from `False`. The 19-dim observation mixes radians, metres and a unit quaternion; without normalisation the metre-scale channels (millimetre magnitudes) would be swamped by the radian-scale ones. |
| `max_iterations` | 1500, run at 600 | **Changed** from the template's 150. 600 was chosen after observing that success arrives well inside it; comparisons must fix it, since an unequal budget makes runs incomparable. |
| `num_steps_per_env` | 16 | Template default, unchanged. |
| `learning_rate` | 1.0e-3, `schedule = adaptive` | Template default. The adaptive schedule with `desired_kl = 0.01` retunes it during training, so a fixed starting value carries little risk. |
| `gamma` / `lam` | 0.99 / 0.95 | Template defaults. At 240-step episodes, γ = 0.99 gives a horizon of ≈ 100 steps, comfortably shorter than an episode. |
| `entropy_coef` | 0.005 | Template default. Not raised despite the local optima of §3 — those were fixed in the reward rather than explored around, which is the more reliable of the two. |
| `clip_param` / `max_grad_norm` | 0.2 / 1.0 | Template defaults, standard PPO. |
| Environments | 4096 | User's standard. Drives the metrics window sizing (§5) and the float32 margin (§2). |

**Honest status.** No hyperparameter sweep was run. Three fields were changed
with a reason each, the rest are the template's defaults inherited from
`docs/reference/research_cylindertask.md` §8, which is itself described there
as a starting point tuned for the Factory/Franka setup rather than for this
robot. The task solved at the first configuration tried, so nothing forced a
sweep — and consequently nothing is known about how sensitive the result is to
any of these values. §8's tuning guidance was never needed and remains
available if a harder rung stalls.

## 4. Training results

**Ø25 mm rung (2.5 mm clearance).** The policy inserts the peg on every
attempt after under three minutes of training at 1024 environments. State
tagged `demo-insertion-working` (`c53ebbc`).

**Ø28 mm rung (1.0 mm clearance, the task specification's stage 1).** Reaches
100 % within 600 iterations, both from scratch and by fine-tuning the Ø25
policy.

> **Evidence status.** Both result lines above rest on the user's observation
> of the running simulation and of the TensorBoard curve; the console output
> and the per-run `demo_metrics.json` were not captured into this repository.
> This deviates from the CLAUDE.md rule that a functional claim needs a
> command plus its output, and it is recorded as a gap rather than glossed
> over — the same call the toolchain section makes for the 2026-07-25 rsl_rl
> baseline. To close it, run `python scripts/compare_runs.py` and paste its
> table here; the numbers are then read from disk, which is what the project
> requires of a result.

**Not yet run:** the Ø29 mm rung (0.5 mm clearance), and the robustness sweep
over `reset_joint_noise` at Ø28.

### What the results do and do not show

They show that at 1.0–2.5 mm radial clearance the task is solvable within
600 iterations under the shipped PD gains, with a dense staged reward and
joint-delta actions — i.e. **without** the operational-space controller D-014
assumes, and despite a droop thirteen times the clearance at the Ø28 rung.

They do **not** yet show robustness. The reset noise in these runs was
±0.01 rad, about 4.3 mm of lateral scatter at the tip. §7 of the research
synthesis names overfitting to easy initial states as the common failure mode,
and D-006 puts the sampling-based initial-state randomisation ahead of the
geometric axis in importance (IndustReal: 88.6 % against 32.4 % for a standard
curriculum). A policy that inserts reliably from nearly the same start pose
every time has not been shown to generalise. **The robustness sweep is
therefore the load-bearing experiment, not the next diameter.**

They also do not separate the two configurations that both reach 100 %. Final
success rate cannot: what distinguishes them is how much experience each
needed. `episodes_to_threshold` (first episode at which the trailing rate held
90 %) was added for exactly this and is reported by `compare_runs.py` as the
`to 90%` column.

## 5. Tooling produced (all runnable without Isaac)

- `scripts/check_demo_reward_math.py` — 30 offline checks of the reward and
  the quaternion helper, including one assertion per exploit above. Caught
  defect 4.
- `scripts/compare_runs.py` — ranks runs by success rate then sample
  efficiency, names the best checkpoint with a ready `play.py` command, groups
  runs sharing a configuration (the only fair head-to-head comparisons), and
  recovers runs predating the per-run metrics file from TensorBoard.
- `scripts/reset_noise_spread.py` — converts `reset_joint_noise` into
  millimetres at the peg tip via the verified FK.

## 6. Open items

- The evidence gap in §4: no metrics table has been read from disk.
- Robustness across `reset_joint_noise` is unmeasured at every rung.
- The Ø29 mm rung (0.5 mm clearance) is untried. Against it: faceting error
  0.0175 mm (3.5 % of the clearance), float32 11.4 µm at 4096 envs (2.3 %),
  and the 13.4 mm droop (27×). If it fails, the droop is the first suspect and
  the answer is D-014, not a reward change.
- The reward's dense terms sit six times below §6's scaling guideline. A
  restructured version following §6 term by term is parked, deliberately not
  runnable, on branch `demo-reward-literature` — worth finishing as an A/B
  comparison against this reward rather than as a blind replacement.
- Whether the finite-difference velocity fix should be carried back into
  increment 3, where the same defect is still open, is a real decision and
  belongs in DECISIONS.md if taken.
