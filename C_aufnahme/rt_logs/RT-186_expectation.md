# RT-186 — expectation, written BEFORE the run (2026-09-12)

> **DEAD 2026-09-12, NOT RUN — do not revive without rewriting it.** The
> user ruled the same day that nothing built before the rebuild carries a
> number for this code (anchor `864cdf1`, `docs/decisions_inbox.md`). This
> file's entire vehicle is a replay of the RT-176 checkpoint, which is
> pre-rebuild, so the per-field "code stand" comparison below does not save
> it: the objection is methodological, not field by field. Kept as a record
> of the design and of the handling traps, not as a plan.
>
> **Originally PARKED, NOT RUN.** The user redirected the question the same
> day: instead of a replay probe at 128 vs 1024 environments, the practical
> question is which environment count TRAINS better, compared at 256 and 512.
> That is `rt_logs/RT-187_expectation.md`. Nothing in this file is wrong and
> nothing is deleted — the four arms, the code-stand comparison and the
> handling traps stay valid if the replay probe is ever wanted. The solver
> question (16 vs 192 position iterations) is NOT answered by RT-187 and
> stays open in `docs/decisions_inbox.md`.

Written on the dev laptop, worktree `p5-robustheit`, code stand `cd496b06`
(pushed to origin before this file was written). `/rt-check` judges the logs
against THIS file and nothing else.

Four arms, `RT-186a` … `RT-186d`. This is a **physics probe**, not a training
run: no policy is learned, no thesis number is produced. It answers one open
question in `docs/decisions_inbox.md`, entry 2026-09-12 ("16 solver iterations
vs Isaac Lab Factory's 192").

## Why this run exists

The inbox entry names a conflict that nothing in this repo has measured:

* Factory paper and IsaacGymEnvs `FactoryBase.yaml`: **16** position
  iterations, `dt = 1/60`, 2 substeps.
* Isaac Lab 2.3 `factory_env_cfg.py` and `assembly_env_cfg.py`: **192**
  position iterations, `dt = 1/120`, with the comment "Important to avoid
  interpenetration".
* Our robot articulation authors **16 / 1** (`ur5e_cfg.py:182-183`, measured
  RT-50, recorded D-127). The fixture authors **nothing** — RT-181's log line
  128 reads "NO authored value on any prim below -- the PhysX SDK default
  applies and is NOT readable here".
* Our `sim` is `SimulationCfg(dt=1/120)` with **no `PhysxCfg` at all**
  (`insertion_env_cfg.py:524`), so Isaac Lab's defaults apply. Its global cap
  `max_position_iteration_count` defaults to **255**
  (`C:\IsaacLab\source\isaaclab\isaaclab\sim\simulation_cfg.py:76`, read on
  the laptop), so 192 is reachable on the articulation alone and no
  `PhysxCfg` has to be introduced for this probe.
* The documented reason to use FEWER environments is the **GPU contact
  buffer**, not solver accuracy: IsaacGymEnvs `docs/factory.md` lists
  "Reduce the number of environments" as the remedy when penetration appears.
  Training here runs `--num_envs 1024` (RT-107, RT-181); the cfg default is
  128 (`insertion_env_cfg.py:828`).

Until commit `e7b85a2d` (2026-09-12) none of this could be judged on anything
but the reward, which the same interpenetration depth already scales (SAPU,
D-109). That commit made the depth readable on its own:
`extras["log"]["interpen_max_mean_mm"]` and `["interpen_max_max_mm"]` on the
curve, and `interpen_max_mm` in `logs/demo_metrics.json` with the keys
`episodes, mean_mm, p50_mm, p95_mm, p99_mm, max_mm, thresh_mm, over_thresh`.

## The one hypothesis

**"At the setting we train on — 1024 environments, 16 position iterations on
the robot articulation, no `PhysxCfg` — the SAPU interpenetration of an
inserting policy stays below the SAPU threshold, and neither raising the
environment count from 128 to 1024 nor raising the position iterations from
16 to 192 moves the distribution beyond the run-to-run spread."**

**The run-to-run spread of this setup is UNKNOWN.** It is therefore measured
in the same batch (arm d) instead of being assumed. No difference between two
arms may be called material before that spread has been read. This is a
comparison RULE, decided here; it is not a measurement and no threshold
number is invented for it.

## Vehicle, and why not something cheaper

The question is about physics, so the arms must produce episodes in which the
part is actually pressed into the pocket. Three candidates were weighed:

1. **A short fresh training run.** Rejected: an untrained policy flails in
   free space. Both settings would read near zero and prove nothing.
2. **`scripted_insert.py`.** Attractive (deterministic trajectory, no policy
   variance) but rejected for two reasons: it has **no hydra**
   (`scripted_insert.py:220`, "this entry point has no hydra … parse_args()
   rejects a bare 'env.x=y' token with exit code 2"), so the solver arm is
   impossible without a code change; and the D-108 scripted-descent gate was
   dropped on 2026-08-29, so whether it still seats at all is unknown.
3. **Replay of the RT-176 policy with `play.py`.** Chosen. `play.py` HAS
   hydra (`play.py:185`, `@hydra_task_config`), and RT-176 reached
   `success_rate_recent` 0.9975 (`HANDOFF-RL.md`, "RT-176 is CLOSED"), so
   nearly every episode seats the part.

## The checkpoint's code stand, and where it differs from ours

The checkpoint was trained on `864cdf1` (2026-09-07). This run stands on
`cd496b06` (2026-09-12). In between, `insertion_env.py` grew by 1622 lines,
`insertion_env_cfg.py` by 572 and `insertion_math.py` by 414, and `autodr.py`
and `obs_noise.py` appeared. "The dimensions match" is NOT enough, so the
whole acting chain was diffed on the laptop, `git show 864cdf1:` against the
worktree. What follows is read, not argued.

**Identical — the policy sees and commands the same things:**

* `OBS_SLICES` byte for byte: joint_pos 0:6, joint_vel 6:12, tip_rel 12:15,
  ee_quat 15:19, yaw_cos_sin 19:21, pocket_quat 21:25, force 25:28.
* `assemble_observation` — the function BODY diffed to zero lines.
* `OBS_DIM` 28, `action_space` 6.
* `actor_hidden_dims` / `critic_hidden_dims` `[128, 128]`, `activation "elu"`.
* `action_scale` 0.02, `decimation` 2, `osc_decimation` 8,
  `control_mode "osc"`, `osc_kp_pos` 100.0, `osc_kp_rot` 30.0,
  `osc_tilt_clamp_rad` 8.52 deg, `episode_length_s` 256/60.
* Observation noise is off **bit for bit**, not merely zero-width: with both
  sigmas at 0 `resolve_obs_noise_model` returns `None` and no model is built
  (`obs_noise.py:204-222`), so `randn_like` is never drawn and the global RNG
  stream is the un-noised one. The two pins in the override block are what
  buy that.

Every config field of `insertion_env_cfg.py` was compared by AST (89 fields
then, 91 now). **Exactly two values changed**, six were removed and eight
added. The removed six are the two penalty terms and the old noise pair; the
added eight are `dr_mode` (off), the three scatter fields (pinned to 0), and
`kernel_a_tilt` / `w_tilt` / `shaping_gamma` — all reward-side, which a
replay does not act on. `report_at_steps` gained a fourth entry, logging
only.

**THE ONE REAL DIFFERENCE, and it is a behaviour difference: the command box
is wider than the one the policy trained in.**

| axis | RT-176 ran at | this run runs at |
|------|---------------|------------------|
| x, y | 0.07 m (`env.osc_pos_clamp_m=0.07` on its command line) | **0.08 m** (`osc_pos_clamp_m`, the default) |
| z | 0.07 m (one scalar clamped all three axes) | **0.1435 m** (`osc_pos_clamp_z_m`, a field that did not exist) |

`insertion_env.py:1557-1559` now clamps per axis; on `864cdf1` one scalar
clamped all three. The box is therefore WIDER in every axis, so the policy's
commands are cut LESS than they were in training. It cannot be matched back:
`Laufplan_Phase5.md:19-24` forbids passing `env.osc_pos_clamp_m` at all, and
D-180 owns both values.

**What that does to this probe, stated plainly:**

* It does **not** confound the comparisons. The box is the same in all four
  arms, so a vs b, c vs b and d vs b each still isolate their one factor.
* It **does** qualify P9, the absolute reading. A wider box permits a larger
  commanded step, and a larger step toward a seated part is exactly what
  produces interpenetration. P9's number is "this policy under THIS box",
  not "RT-176 as it trained".

The checks above are four file reads and an AST comparison, not a load test.
The load line in arm b's log is what closes it (P1 below).

## The four arms, and the order they run in

Run **b and d first** — they establish the spread. Only then a, then c.

| arm | `--num_envs` | position iterations | seed | what it answers |
|-----|--------------|---------------------|------|-----------------|
| b | 1024 | 16 (unchanged) | 42 | the setting we train on |
| d | 1024 | 16 (unchanged) | 20 | the run-to-run spread of b |
| a | 128 | 16 (unchanged) | 42 | does the environment count matter |
| c | 1024 | **192** | 42 | do 16 iterations suffice |

One factor per comparison: **a vs b** is the environment count alone,
**c vs b** is the iteration count alone, **d vs b** is the seed alone.

## The override block, and what was deliberately removed from it

All four arms carry RT-175's spread — the distribution the RT-176 policy was
trained on — with three deliberate edits:

* **`env.osc_pos_clamp_m` is NOT passed.** `Laufplan_Phase5.md:19-24` forbids
  it: older run commands carry `0.07`, which is smaller than the 0.070838 m
  the disk start demands, so a copied command silently cuts the box at the
  disk rim. D-180 owns the value.
* **`env.contact_penalty_scale` and `env.far_penalty_scale` are NOT passed.**
  Both fields no longer exist — `grep -rn` over `source/` returns zero hits
  (Laufplan step 1 removed the terms). Passing them would fail the run.
* **The four Phase-5 scatter fields are pinned to zero.** They did not exist
  when RT-176 trained, and their current defaults are non-zero:
  `fixture_pos_noise_xy` 0.005 (`insertion_env_cfg.py:658`),
  `obs_noise_pocket_pos_std_m` 0.0025 (`:1070`),
  `force_obs_noise_std_n` 3.5 (`:1071`),
  `grasp_obs_offset_x_m` 0.003 (`:1079`). Left at their defaults the policy
  would act on observations it never saw, and a behaviour change would be
  confounded with the physics change this probe is about.

`dr_mode` defaults to `"off"` (`insertion_env_cfg.py:634`) and
`curriculum_enabled` to `False` (`:996`), so neither is passed.

The common override block, verbatim:

```
env.fixture_yaw_noise_rad=0.1047 env.fixture_tilt_noise_rad=0.1396 env.start_lateral_offset=0.006 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.fixture_pos_noise_xy=0.0 env.obs_noise_pocket_pos_std_m=0.0 env.force_obs_noise_std_n=0.0 env.grasp_obs_offset_x_m=0.0
```

Arm **c alone** appends:

```
env.robot_cfg.spawn.articulation_props.solver_position_iteration_count=192
```

The path: `insertion_env_cfg.py:541` holds
`robot_cfg: ArticulationCfg = UR5E_HOME_CFG`, and `ur5e_cfg.py:162-183` is
`ArticulationCfg(spawn=sim_utils.UsdFileCfg(… articulation_props=
ArticulationRootPropertiesCfg(solver_position_iteration_count=16, …)))`.

**THIS HYDRA PATH IS UNVERIFIED.** Nobody has passed a nested spawn property
on the command line in this repo. If hydra refuses it, the fallback is to
edit `ur5e_cfg.py:182` to `192` for arm c only, note the edit in the log, and
revert it afterwards. A refused override that is silently ignored would be
worse than a crash — P6 below is the guard against exactly that.

## Why 800 steps

`_metrics_dump_every = max(50, window // 8)` (`insertion_env.py:1241`) and
`window = max(2000, num_envs * metrics_window_iterations * 16 /
max_episode_length)` (`:1055-1056`), with `metrics_window_iterations = 20`
(`insertion_env_cfg.py:514`) and `max_episode_length` 256.

* 128 envs: `128 * 20 * 16 / 256 = 160` → the floor wins, window **2000**.
* 1024 envs: `1024 * 20 * 16 / 256 = 1280` → the floor wins, window **2000**.

So the window is 2000 episodes in **every** arm. There is no sample-size
confound between a and b, which matters because a maximum over a larger
window is expected to be larger for that reason alone.

The dump fires every 250 finished episodes. RT-175 measured
`mean_success_episode_steps` 28.35, so a seated episode ends far before the
256-step cap. At 128 envs, 800 steps give roughly 128 × 800/28 ≈ 3600
episodes — the window fills several times over. 800 is generous on purpose:
the cost of a short arm is a void run.

## Two handling traps — read before running anything

1. **`logs/demo_metrics.json` is a FIXED path** (`insertion_env.py:1229`) and
   every run overwrites it. After EACH arm the file must be copied before the
   next arm starts, or its numbers are gone.
2. **`play.py --checkpoint` wants a FULL path** (`train.py` wants a
   filename) — `HANDOFF-RL.md:1694`, the lesson of RT-140p and RT-149.

## Commands (training PC, PowerShell, conda env `env_isaaclab`)

Root: `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`

### Step 0 — pull, and find the checkpoint

```
git pull --ff-only origin p5-robustheit
git rev-parse HEAD
```

Do NOT compare this against a SHA written here: a SHA naming the commit
that carries this file invalidates itself the moment the file is edited
(the RT-184 lesson, commit `e791ca29`). Check the thing that actually
matters instead -- that the pulled code carries the interpenetration
instrument:

```
git merge-base --is-ancestor e7b85a2d HEAD; if ($?) { "instrument present" } else { "INSTRUMENT MISSING -- STOP" }
```

If it prints anything but `instrument present`, stop: without `e7b85a2d`
there is no `interpen_max_mm` key and every arm is void.

```
Get-ChildItem "logs\rsl_rl\ur5e_insertion\09-07_15-43-06_offset5mm_tilt0-8deg_yaw+-6deg_currOFF_start+30..+50mm_lat6mm_seed42\model_*.pt" | Sort-Object LastWriteTime | Select-Object -Last 3 Name, LastWriteTime
```

**The exact filename is UNKNOWN to the laptop and is NOT written here.**
RT-176 ran iterations 1650..1949 with `save_interval = 50`, so the highest
`model_*.pt` is what the last line prints. The operator read it on 2026-09-12: **`model_1949.pt`**, after copying
the run folder over from another location. It is already substituted below.

**If the folder does not exist, the whole run is BLOCKED.** Do not substitute
another run's folder. Report it back and stop — a different policy is a
different experiment.

### Arm b — the setting we train on (run this first)

```
.\scripts\rt_log.ps1 RT-186b python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max-steps 800 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-07_15-43-06_offset5mm_tilt0-8deg_yaw+-6deg_currOFF_start+30..+50mm_lat6mm_seed42\model_1949.pt" env.fixture_yaw_noise_rad=0.1047 env.fixture_tilt_noise_rad=0.1396 env.start_lateral_offset=0.006 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.fixture_pos_noise_xy=0.0 env.obs_noise_pocket_pos_std_m=0.0 env.force_obs_noise_std_n=0.0 env.grasp_obs_offset_x_m=0.0
```

```
Copy-Item logs\demo_metrics.json logs\RT-186b_demo_metrics.json
```

### Arm d — the same thing on another seed (the spread)

```
.\scripts\rt_log.ps1 RT-186d python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 20 --max-steps 800 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-07_15-43-06_offset5mm_tilt0-8deg_yaw+-6deg_currOFF_start+30..+50mm_lat6mm_seed42\model_1949.pt" env.fixture_yaw_noise_rad=0.1047 env.fixture_tilt_noise_rad=0.1396 env.start_lateral_offset=0.006 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.fixture_pos_noise_xy=0.0 env.obs_noise_pocket_pos_std_m=0.0 env.force_obs_noise_std_n=0.0 env.grasp_obs_offset_x_m=0.0
```

```
Copy-Item logs\demo_metrics.json logs\RT-186d_demo_metrics.json
```

### Arm a — 128 environments

```
.\scripts\rt_log.ps1 RT-186a python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 128 --headless --seed 42 --max-steps 800 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-07_15-43-06_offset5mm_tilt0-8deg_yaw+-6deg_currOFF_start+30..+50mm_lat6mm_seed42\model_1949.pt" env.fixture_yaw_noise_rad=0.1047 env.fixture_tilt_noise_rad=0.1396 env.start_lateral_offset=0.006 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.fixture_pos_noise_xy=0.0 env.obs_noise_pocket_pos_std_m=0.0 env.force_obs_noise_std_n=0.0 env.grasp_obs_offset_x_m=0.0
```

```
Copy-Item logs\demo_metrics.json logs\RT-186a_demo_metrics.json
```

### Arm c — 192 position iterations

```
.\scripts\rt_log.ps1 RT-186c python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max-steps 800 --checkpoint "C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1\logs\rsl_rl\ur5e_insertion\09-07_15-43-06_offset5mm_tilt0-8deg_yaw+-6deg_currOFF_start+30..+50mm_lat6mm_seed42\model_1949.pt" env.fixture_yaw_noise_rad=0.1047 env.fixture_tilt_noise_rad=0.1396 env.start_lateral_offset=0.006 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.fixture_pos_noise_xy=0.0 env.obs_noise_pocket_pos_std_m=0.0 env.force_obs_noise_std_n=0.0 env.grasp_obs_offset_x_m=0.0 env.robot_cfg.spawn.articulation_props.solver_position_iteration_count=192
```

```
Copy-Item logs\demo_metrics.json logs\RT-186c_demo_metrics.json
```

Paste each log into `rt_logs/inbox.txt` on the laptop, one arm at a time,
and paste the four `RT-186*_demo_metrics.json` files with them.

## PASS / FAIL lines, in order

Every point names the FILE and KEY it is read from. Nothing is judged from
the console.

**P1 — the checkpoint loaded.** Arm b's log carries a
`Loading model checkpoint from: …` line naming the RT-176 folder and the
`model_1949.pt` file from step 0. Without it the arm is VOID: the four dimension
values read on the laptop are an argument, not a load test.

**P2 — the arm actually inserted.** `logs/RT-186<arm>_demo_metrics.json`,
`success_rate_recent >= 0.90` in every arm. An arm below that did not seat
the part and therefore measured nothing about the collider: it is **VOID,
not FAIL**, and the comparison it feeds is not made. (0.90 and not RT-176's
0.9975, because the four pinned scatter fields and the replay path are not
RT-176's exact conditions. It is a floor for "the probe did its job", not a
performance bar.)

**P3 — the instrument is populated.** `interpen_max_mm` is present in all
four JSONs with `episodes > 0` and `measured` NOT `false`. A `measured:
false` means `rl_terms_enabled` was off and the SAPU query never ran; the
arm is then VOID.

**P4 — the reset-step check.** The start is +30..+50 mm above the opening
(`env.start_tip_above_entrance` 0.050 / `_low` 0.030), so the part is
nowhere near the fixture when an episode begins. If `interpen_max_mm.p50_mm`
is non-zero in EVERY arm by a similar amount, suspect the reset step, not
the physics: `force_norm_n` is already known to read a reset transient
(`HANDOFF-RL.md`, 2.021 N in every env, RT-149 DETAIL (c)) and the same
buffer shape could do it here. This is a NAMED RISK, not a prediction.

**P5 — the spread, read BEFORE a and c are judged.** From
`RT-186b_demo_metrics.json` and `RT-186d_demo_metrics.json`, write down the
pair of `mean_mm` and the pair of `p95_mm`. The larger of the two absolute
differences is **the spread**. No later comparison may be called material
unless it exceeds it. Write the two pairs into the verdict verbatim.

**P6 — arm c really changed the solver.** Arm c's startup report must show
the robot articulation at **192** position iterations, the same line RT-50
and D-127 read (`solver iterations (D-115 verify): …
solverPositionIterationCount=…`). If it still reads 16, the hydra override
was ignored and arm c is VOID — judge nothing from it and fall back to the
`ur5e_cfg.py:182` edit. **This point is why arm c cannot be judged on its
numbers alone.**

**P7 — the environment count.** `a` vs `b`, on `mean_mm` and `p95_mm` of
`interpen_max_mm`. The hypothesis holds if the difference does not exceed
P5's spread. It is FALSIFIED — the GPU contact buffer matters at our
env count — if **b** reads materially HIGHER than **a**.

**P8 — the iteration count.** `c` vs `b`, same two keys. The hypothesis
holds if the difference does not exceed P5's spread. It is FALSIFIED — 16
iterations are not enough — if **c** reads materially LOWER than **b**.

**P9 — the absolute question, independent of all comparisons.**
`interpen_max_mm.over_thresh` against `episodes` in arm b. `thresh_mm` is
printed in the same dict (it is `cfg.interpen_thresh`, owned by D-109, and
is NOT restated here). A large `over_thresh` at the setting we train on
would mean the SAPU filter is discarding a large share of episodes, which
is a finding about the collider whatever a, c and d say.

**P10 — the log is complete.** Each arm's log ends with an exit line. A
missing exit line does not by itself fail an arm (RT-178 and RT-181 both
lack one), but it must be stated in the verdict.

## What this run cannot answer

* **Only the robot side changes.** The fixture authors no iteration count
  (RT-181 line 128), so arm c raises the articulation's number while the
  fixture keeps the PhysX SDK default, which this repo has never read. A
  null result in P8 does NOT mean "iterations do not matter"; it means
  "raising the ROBOT articulation from 16 to 192 did not matter here".
* **One policy, one geometry, one GPU.** The RTX 3080 has 10053 MB
  (`rt_logs/RT-179/RT-179.txt:21`). The contact-buffer mechanism the Factory
  docs describe is a property of the card and the scene; another card could
  read differently.
* **`max_depenetration_velocity` is unset and stays unset**
  (`ur5e_cfg.py`, the comment above `articulation_props`). D-159 set it to
  5.0, RT-113 showed no effect, the user reverted it. This probe does not
  touch it, and "unset" is not the same as 0.0.
* **No PhysX contact-buffer warning is expected or checked.** RT-181's log
  shows none at 1024 envs, but whether PhysX would print one at all in this
  configuration is untested. A silent buffer is not an empty buffer.
* **The policy ran in a wider command box than it learned in** (the table
  above: x/y 0.07 -> 0.08 m, z 0.07 -> 0.1435 m). Constant across the four
  arms, so the comparisons hold, but the absolute level in P9 is not the
  level RT-176 itself produced.
* **The run folder was copied onto the training PC from elsewhere** (user,
  2026-09-12). P1's load line is therefore not a formality: it is the only
  evidence that the file actually replayed is the RT-176 policy.
* **Nothing here is a thesis number.** Four probe runs, one seed pair, one
  checkpoint. The D-115 conflict between 16 and 192 stays written as a
  conflict; this run says only what OUR scene does.

## Status

**UNVERIFIED.** Nothing in this file has run. Written on the laptop against
code stand `8cb3a2b1` or later; the instrument commit is `e7b85a2d`.
