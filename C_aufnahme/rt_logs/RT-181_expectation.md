# RT-181 — expectation, written BEFORE the run (2026-09-12)

**A SMOKE RUN, NOT A RESULT.** `Laufplan_Phase5.md:36` calls step 3 the
"No-DR smoke, 30 iterations, not a result." No success rate is a criterion
here and none is read as one. Status: UNVERIFIED until the training PC.

**(Erwartung korrigiert 2026-09-12, vor dem Urteil.)** Corrected BEFORE any
log was read, so the verdict carries `(Erwartung korrigiert)`, the mark
`.claude/skills/rt-check/SKILL.md` step 4b prescribes. What changed: the SHA
and the line numbers that moved with it; P4's criterion, which could not tell
`abs` from `add` and is replaced by a pooled magnitude rule; the false
"identical by construction" warrant for counts 60 and 180; the literal
expected in `params/env.yaml` (P5); the checkpoint claim; and a new P9 that
states the start distribution this run actually uses.

Code: `origin/p5-robustheit` = **`23a7aa0`** — the commit the operator pulled
and ran. This file first named `9f37a07`, the commit BEFORE the pin and dump
repairs. Two commits landed after `23a7aa0` and neither touches the env:
`git diff --stat 23a7aa0..4725840 -- source/ scripts/` reports exactly one
file, `scripts/shorten_rt_log.py` (45 insertions, 10 deletions). So `23a7aa0`
is the code that ran. The correction moves line numbers:
`git diff 9f37a07..23a7aa0 --stat` gives `insertion_env.py | 10 +-` — a
2-line comment replaced by 8 at old 1729-1730, so every cited
`insertion_env.py` line from old 1731 on moves **+6** (checked line by line
against `git show 23a7aa0:<path>`, not only by arithmetic). No other file
cited below changed between the two commits. **NOTHING in this
build has ever run under Isaac.** The observation noise model (D-182), the
grasp belief offset (D-183), the fourth startup-report count and the removal
of the two penalty terms (D-184) are all read off the source, never executed.

`dr_mode` defaults to `"off"` (`insertion_env_cfg.py:629`), so **no hydra
override is passed at all** and no AutoDR provider is built.

RT-181 is part (a) of step 3: ONE run alone, training seed 20. Part (b),
five seeds at once, is RT-182, and the parallel decision belongs there.

## The one hypothesis

**The new build starts, builds the observation noise model, and prints what
it claims.** Every line below is readable off the log or off
`params/env.yaml`. If P1..P7 hold, the D-182 / D-183 / D-184 wiring is real
rather than source-read; if P4 fails, `operation="abs"` is not doing what
`obs_noise.py`'s header says it does.

## Command (training PC, PowerShell, conda env `env_isaaclab`)

Read `nvidia-smi` at the START (before the run) and at the END, per
`Laufplan_Phase5.md:39-40`:

```
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

```
.\scripts\rt_log.ps1 RT-181s20 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 20 --max_iterations 30
```

Training seed 20, because `Laufplan_Phase5.md:9` reserves seed 20 for probe
runs. Name `RT-181s20` follows `Laufplan_Phase5.md:16-18` (one RT number per
step, seed as suffix), and `rt_log.ps1:51` puts the log at
`rt_logs\RT-181s20.txt`.

**THE TWO COMMAND-LINE TRAPS, and why this command carries neither.**

1. **`env.osc_pos_clamp_m` is NOT passed** — `Laufplan_Phase5.md:19-24`:
   the cfg default is the only source of the box half-width, and the 0.07 m
   that older RT commands carry is SMALLER than the 0.070838 m the disk start
   demands, so a copied command would silently cut the box at the disk rim.
2. **`env.contact_penalty_scale` / `env.far_penalty_scale` are NOT passed** —
   those fields left the code with Commit A0 (`Laufplan_Phase5.md:33-34`,
   D-184), so an old command carrying them now FAILS in hydra before physics
   starts: the keys are not in the node registered at
   `isaaclab_tasks/utils/hydra.py:58`, which is built from `env_cfg.to_dict()`
   at `hydra.py:49`. Verified on the laptop today:

   ```
   $ grep -rn "contact_penalty_scale\|far_penalty_scale" source/ scripts/
   $ echo $?
   1
   ```

   Empty result, exit status 1 — neither name occurs anywhere in `source/`
   or `scripts/`. (The names still occur in `DECISIONS.md`,
   `docs/decisions_inbox.md` and the `docs/runs/RT-*.md` files; those are
   history, not code.) The exact hydra error text is **unknown for this
   setup**.

## The budget, and what it buys

`--max_iterations 30`. Neither `--stop_at_success_rate` nor
`--stop-when-dr-max` is passed, so `train.py:725` takes the single-call
branch: one `runner.learn(num_learning_iterations=30, ...)`, no block loop,
iteration indices **0..29** (L-01).

`Laufplan_Phase5.md:45-46` gives the reason 30 is enough, verbatim:

> Why 30 is enough: 16 steps per iteration, an episode ends at 256 steps
> = 16 iterations, so every env resets at least once.

Re-read, not re-derived: `num_steps_per_env = 16`
(`agents/rsl_rl_ppo_cfg.py:27`); the episode cap is
`ceil(episode_length_s / (sim.dt * decimation))` with `episode_length_s =
256/60` (`insertion_env_cfg.py:138`), `sim.dt = 1/120`
(`insertion_env_cfg.py:519`) and `decimation = 2`
(`insertion_env_cfg.py:122`) = **256** control steps
(`insertion_env_cfg.py:1163`). 256 / 16 = 16 iterations; 30 > 16.

30 iterations = 30 * 16 = **480 control steps**, so `_obs_calls` passes the
fourth report count 258 (`insertion_env_cfg.py:796`). Episodes:
1024 * 16 * 30 / 256 = **1920**, and that is a LOWER bound (L-04) — an
episode that ends early on a force abort raises the count.

## PASS / FAIL lines, in order

**P1 — IT STARTS AT ALL.** No traceback, no `TypeError: Missing values
detected` out of `cfg.validate()` (`direct_rl_env.py:90`). This is the one
that pays for D-185: `InsertionObsNoiseCfg` binds
`noise_cfg: NoiseCfg | None = None` (`obs_noise.py:196`) precisely because an
inherited `MISSING` would kill every run in the constructor. FAIL here means
the binding is not enough and the message names the field.

**P2 — THE NOISE MODEL EXISTS.** The startup report must print the
**observation bias** line in its MODEL-BUILT form
(`insertion_env.py:4439-4441`):

```
observation bias env0..3 (D-182, tip_rel 12:15, sigma 0.0025 m, drawn per EPISODE): [[...], [...], [...], [...]]
```

and must NOT print the `OFF -- both sigmas are 0.0` form
(`insertion_env.py:4433-4434`). Both sigmas are at their cfg DEFAULTS and
both are non-zero — `obs_noise_pocket_pos_std_m = 0.0025`
(`insertion_env_cfg.py:1041`), `force_obs_noise_std_n = 3.5`
(`insertion_env_cfg.py:1042`) — so `resolve_obs_noise_model`
(`obs_noise.py:218-221`) returns a cfg, not `None`, and
`DirectRLEnv.__init__` builds the model. FAIL (the `OFF` line) means the
assignment at `insertion_env.py:146` did not reach the base class.

Four rows of three numbers each: `_n_obs = min(4, num_envs)`
(`insertion_env.py:4430`) and the `tip_rel` slice is `(12, 15)`
(`insertion_math.py:778`) of `OBS_DIM = 28` (`insertion_math.py:784`).

**P3 — THE GRASP BELIEF ERROR IS PRINTED AND INSIDE ITS BOX.** The line
(`insertion_env.py:4442-4445`) reads

```
grasp belief error env0..3 (D-183, +-0.003 m on the part's short axis, drawn per EPISODE): [v0, v1, v2, v3]
```

with `grasp_obs_offset_x_m = 0.003` (`insertion_env_cfg.py:1050`). The draw
is `(2u - 1) * 0.003` with `u` uniform on [0, 1]
(`insertion_env.py:2228-2230`), so **every printed value must satisfy
|v| <= 0.003 exactly** — a hard bound, not a tolerance. FAIL: a value outside
±0.003, or four identical values across four envs (a per-env draw that is not
per-env), or a value that is identical at step 2 and step 258 in every env
(no re-draw).

**P4 — THE POCKET BIAS IS RE-DRAWN, NOT ACCUMULATED. THIS IS THE CHECK THAT
MATTERS — AND IT CANNOT BE DECIDED FROM RT-181 ALONE.** The per-episode bias
is written ONLY in the noise model's `reset()`, as
`self._bias[env_ids] = func(self._bias[env_ids], cfg)` (`obs_noise.py:51-58`,
`insertion_env.py:4419-4425`). `report_at_steps = (2, 60, 180, 258)`
(`insertion_env_cfg.py:796`).

**The two formulas, read from the library.** `gaussian_noise` in
`C:\IsaacLab\source\isaaclab\isaaclab\utils\noise\noise_model.py:92-97`:

```
if cfg.operation == "add":
    return data + cfg.mean + cfg.std * torch.randn_like(data)
...
elif cfg.operation == "abs":
    return cfg.mean + cfg.std * torch.randn_like(data)
```

With `mean = 0.0`, a buffer holding draw `x1` that is reset once more holds
`x1 + x2` under `add` and `x2` under `abs`. **Exactly one reset happens
before count 2:** `RslRlVecEnvWrapper.__init__` calls `env.reset()`
(`isaaclab_rl/rsl_rl/vecenv_wrapper.py:66`), `DirectRLEnv.reset` calls
`_reset_idx` (`direct_rl_env.py:315`), which calls
`self._observation_noise_model.reset(env_ids)` (`direct_rl_env.py:625`); the
two observation calls before the first step are that reset's own return
(`direct_rl_env.py:331`) and the runner's `get_observations()`. So count 2 is
`x1` under BOTH operations, and count 258 is `x1 + x2` under `add` against
`x2` under `abs`.

**WHY THE OLD CRITERION WAS NO CRITERION.** This file used to ask only that
"the step-258 rows differ from the step-2 rows". `x1 + x2` differs from `x1`
exactly as much as `x2` does, so that test passes under BOTH operations and
decides nothing. It is replaced:

**THE POOLED MAGNITUDE RULE.** Pool the printed bias values of all SIX logs
of step 3 — `RT-181s20` plus `RT-182s1`..`RT-182s5`. Each printout gives
4 envs (`_n_obs = min(4, num_envs)`, `insertion_env.py:4430`) x 3 channels
(the `tip_rel` slice `(12, 15)`, `insertion_math.py:778`) = 12 numbers, so
6 x 12 = **72** numbers at count 2 and 72 at count 258. Compute

```
R = mean|v|(258) / mean|v|(2)
```

* `abs` re-draws from the same distribution, so **R ~ 1.0**.
* `add` sums two independent draws — `x1 + x2 ~ N(0, sigma*sqrt(2))` — so
  **R ~ sqrt(2) = 1.414**.

**THE BAR IS 1.22, and here is where it comes from.** `R` is a ratio of two
means of 72 half-normal magnitudes; its spread has to be computed, not
asserted. Monte Carlo on this laptop, 2026-09-12, 200 000 trials, seed
20260912, with the count-2 draw SHARED between numerator and denominator
under `add` (it is the same `x1`, so the two are correlated):

| n pooled | `abs`: mean (sd) | `add`: mean (sd) | P(`abs` >= 1.22) | P(`add` < 1.22) |
|---|---|---|---|---|
| 72 (all six logs) | 1.008 (0.128) | 1.421 (0.133) | 0.058 | 0.056 |
| 12 (one log alone) | 1.051 (0.345) | 1.454 (0.352) | 0.263 | 0.262 |

1.22 is the bar because at 72 pooled numbers that is where the two error
rates meet, at about 6 % each way. **The 12-number row is the reason this
point is not judged inside RT-181: one log alone misreads the operation about
one time in four.** RT-181 contributes 12 of the 72; the P4 verdict is
written once, after RT-182's five logs exist.

* **PASS:** `R` < 1.22 over the pooled 72.
* **FAIL:** `R` >= 1.22 — the `add`-instead-of-`abs` failure that
  `obs_noise.py`'s header calls load-bearing (`obs_noise.py:49-58`): the
  pocket offset random-walks away over a run instead of being re-drawn per
  episode.
* **The rule errs one way only.** An env that reset TWICE before count 258
  (an early force abort) holds `x1 + x2 + x3` under `add` and still one draw
  under `abs`. Extra resets can only push `R` UP, never down, so they cannot
  manufacture a PASS.

**WHAT COUNTS 60 AND 180 ARE — NOT "identical by construction".** This file
used to say the printouts at 2, 60 and 180 all sit inside the first episode
and are identical by construction. **That is FALSE under `train.py`.**
`grep -n init_at_random_ep_len scripts/rsl_rl/train.py` returns line 725,
`runner.learn(num_learning_iterations=agent_cfg.max_iterations,
init_at_random_ep_len=True)`, and rsl_rl 3.0.1 then randomises every env's
episode counter before the first step:
`self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
high=int(self.env.max_episode_length))` (`on_policy_runner.py:66-69`, read on
this laptop under
`Thesis Workflow Phase 5\p5_robustheit_astra\.venv-review\Lib\site-packages\rsl_rl\runners\`,
version pinned by `rsl_rl_lib-3.0.1.dist-info`). With `max_episode_length =
256` the draw `b` is uniform on 0..255, and the time-out rule is
`episode_length_buf >= max_episode_length - 1` (`insertion_env.py:1890`), so
an env that drew `b` first resets at step `255 - b`. Envs therefore reset at
different times, and **a bias row that has changed at count 60 or count 180
is an early reset, not a finding.** Count 258 still lies past every env's
first reset, but **with margin zero**: the env that draws `b = 0` resets at
step 255, i.e. at observation call 257 — one call before the report at 258.
Count 2 itself is the runner constructor, before any step at all.

**What the rule does NOT give.** It is a two-hypothesis discriminator, not a
measurement. The bias distribution itself — that it is Gaussian, that its
sigma is 0.0025 m — stays **unknown for this setup**. The DECLARED sigma is
0.0025 m (`GaussianNoiseCfg(mean=0.0, std=bias_std, operation="abs")`,
`obs_noise.py:146-148`; sigma from `insertion_env_cfg.py:1041`); 3 sigma =
0.0075 m, and the chance that at least one of 12 draws lands outside
±3 sigma is 0.0319 (`1 - 0.9973002039367398**12`), so a single large value is
not by itself a finding.

**P5 — `params/env.yaml` IS WRITTEN AND CARRIES THE TWO SIGMAS.**
`train.py:621` dumps it to
`logs/rsl_rl/ur5e_insertion/<MM-DD_HH-MM-SS>_<tags>_seed20/params/env.yaml`
(`experiment_name = "ur5e_insertion"`, `agents/rsl_rl_ppo_cfg.py:38`;
`log_dir` built at `train.py:259` and `train.py:412-419`). Read it with

```
Select-String -Path logs\rsl_rl\ur5e_insertion\*seed20\params\env.yaml -Pattern "obs_noise_pocket_pos_std_m|force_obs_noise_std_n|grasp_obs_offset_x_m|observation_noise_model|dr_mode"
```

PASS: the file exists and carries `obs_noise_pocket_pos_std_m: 0.0025`,
`force_obs_noise_std_n: 3.5`, `grasp_obs_offset_x_m: 0.003` and
**`dr_mode: 'off'` — WITH the single quotes.** PyYAML quotes the string
because a bare `off` would round-trip as the boolean False:
`python -c "import yaml; print(repr(yaml.dump({'dr_mode':'off'})))"` prints
`"dr_mode: 'off'\n"` (run on this laptop, 2026-09-12). This file expected the
bare literal `dr_mode: off`; that literal never appears, and its absence is
NOT a FAIL.
FAIL: the run dies IN `dump_yaml` — which is what D-182's own verification
status is waiting on ("whether this cfg class survives hydra's `to_dict` and
the `params/env.yaml` dump is read off the source, not run. The dump itself
is the training PC's answer", `obs_noise.py:88-90`). **This run is that
answer.**

**The `observation_noise_model` key is a SEPARATE reading and its expected
content is unknown for this setup.** Two facts, both read today: hydra
serialises the cfg at REGISTRATION time (`hydra.py:49`), and at that moment
`observation_noise_model` is still the `DirectRLEnvCfg` default `None`
(`direct_rl_env_cfg.py:172`) — `InsertionEnv.__init__` only assigns it at
`insertion_env.py:146`, inside `gym.make` (`train.py:482`), which runs BEFORE
the dump at `train.py:621`. Whether the dumped object is the same instance
the env mutated — and therefore whether `observation_noise_model` appears in
the YAML as an `InsertionObsNoiseCfg` or as `null` — depends on whether
gymnasium copies the `cfg` keyword argument. **gymnasium is not installed on
the laptop** (`python -c "import gymnasium"` -> `ModuleNotFoundError: No
module named 'gymnasium'`), so this was NOT verified. Record what the key
reads; neither value is a FAIL.

**P6 — THE STARTUP REPORT FIRES FOUR TIMES AND SAYS `dr_mode: off`.** Four
blocks, labelled `step 2 of (2, 60, 180, 258)`, `step 60 of ...`,
`step 180 of ...`, `step 258 of ...` (`insertion_env.py:4058`), each carrying
the LOG line `dr_mode: off` — unquoted here, because this is
`print(f"dr_mode: {cfg.dr_mode}")` (`insertion_env.py:4084`) and not the YAML
dump of P5 — and no AutoDR band table. This file cited
`insertion_env.py:4430` for that print; at both SHAs that line is
`_t_lo, _t_hi = insertion_math.OBS_SLICES["tip_rel"]`, so the citation was
wrong independently of the shift. FAIL if the fourth block is missing:
`_obs_calls` counts the WHOLE run (`insertion_env.py:298-299`,
`insertion_env.py:1762`), 480 > 258, so at 30 iterations it must appear.

**DO NOT READ THE BLOCK HEADER'S `t = ... s after reset` AS A PER-ENV TIME.**
It is `self._obs_calls * cfg.sim.dt * cfg.decimation`
(`insertion_env.py:4050`, printed at `:4058`) — the RUN's elapsed sim time
since its first observation call. Under `init_at_random_ep_len=True` (P4) the
envs reset at different steps, so no env is `t` seconds past ITS OWN reset.
The number says nothing about any env's episode phase and must not be used as
if it did.

**P7 — IT FINISHES 30 ITERATIONS AND SAYS HOW LONG IT TOOK.** The last
learning block is labelled `Learning iteration 29/30` (L-01), and
`Training time: <x> seconds` is printed (`train.py:775`; the clock starts at
`train.py:519`, after the env is built, so Isaac boot is excluded). **Record
`Training time` and the per-iteration `Iteration time: <x>s` series — RT-182
divides by them.** Read the series, not one value: in RT-179 iteration 0 cost
22.36 s and iteration 1 cost 16.75 s (`rt_logs/RT-179/RT-179.txt:371` and
`rt_logs/RT-179/RT-179.txt:443`), so iteration 0 carries a warm-up and is
excluded from any rate (L-05). What a solo 1024-env iteration costs in THIS
build is **unknown for this setup** — that is what this run measures.

**P8 — THE VRAM AND UTILISATION READINGS EXIST.** `nvidia-smi` before and
after, both pasted into the log by hand. PASS is simply: two readings exist.
There is no threshold — **the per-process VRAM of one 1024-env run in this
build is unknown for this setup, and RT-181 is the run that establishes it.**
The only grounded ceiling: the training PC's card is an NVIDIA GeForce
RTX 3080 with **10053 MB** of GPU memory (`rt_logs/RT-179/RT-179.txt:21`,
same machine — cwd
`C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`,
`rt_logs/RT-179/RT-179.txt:2`). RT-179 recorded no per-process VRAM number,
and it ran a DIFFERENT configuration (`env.dr_mode=autodr` plus the two
penalty scales on the command line, `rt_logs/RT-179/RT-179.txt:3`) — so it
sets no number here, only the card.

**P9 — THE START DISTRIBUTION IS THE ZERO-WIDTH ONE (point added
2026-09-12).** This file never stated what a `dr_mode='off'` run starts from,
so there was nothing to falsify. The cfg defaults, read from
`git show 23a7aa0:source/insertion/insertion/tasks/direct/insertion/insertion_env_cfg.py`:

| quantity | default | file:line |
|----------|---------|-----------|
| lateral start radius `start_lateral_offset` | **0.0 m** — a point, not a disk | `insertion_env_cfg.py:479` |
| start height `start_tip_above_entrance` | `RUNG0_START_TIP_ABOVE_ENTRANCE` = **0.030 m** | `insertion_env_cfg.py:384-386`; `insertion_tasks_cfg.py:1669` |
| start-height LOWER bound `start_tip_above_entrance_low` | the SAME constant, so **low == high**, zero width | `insertion_env_cfg.py:425-427` |
| yaw noise `reset_yaw_noise` | **0.0 rad** | `insertion_env_cfg.py:360` |
| joint noise `reset_joint_noise` | **0.0 rad** | `insertion_env_cfg.py:359` |
| static fixture tilt `fixture_tilt_rad` | **0.0 rad** | `insertion_env_cfg.py:678` |
| fixture tilt noise `fixture_tilt_noise_rad` | **0.0 rad** | `insertion_env_cfg.py:703` |
| fixture yaw noise `fixture_yaw_noise_rad` | **0.0 rad** | `insertion_env_cfg.py:663` |
| fixture position noise `fixture_pos_noise_xy` | **0.005 m — the one scatter that is NOT zero** | `insertion_env_cfg.py:653` |

The log lines that show it, PASS meaning each reads as written:

* `[insertion] rung-0 start pose: tool point +30.000 mm above the stage-2
  opening plane ...` (`insertion_env.py:729-735`).
* `[insertion] rung-0 start lateral offset OFF (start_lateral_offset = 0.0):
  every episode starts ON the pocket axis, which CANCELS the lateral part of
  fixture_pos_noise_xy and reset_joint_noise (D-170).`
  (`insertion_env.py:745-748`).
* **NO** `rung-0 start height SAMPLED per episode` line
  (`insertion_env.py:750-755`). It is gated on `_start_sampling`, which is
  `_low < float(cfg.start_tip_above_entrance)` (`insertion_env.py:655-659`)
  and is False when low == high.
* `fixture pose noise (D-034): xy +-0.005 m, yaw +-0.0000 rad (ACTIVE --
  kinematic fixture teleported per reset)` (`insertion_env.py:4127-4129`).
* `fixture tilt (D-036 probe): +0.00 deg ...` (`insertion_env.py:4073`),
  `fixture yaw (D-038 probe): +0.00 deg ...` (`:4075`) and
  `fixture tilt noise (D-037): magnitude 0..0.00 deg ...` (`:4077`).

FAIL: any of those carries a non-zero width, or the `start height SAMPLED`
line appears. Then this is not the zero-width start the smoke run is supposed
to be, and nothing measured here compares with any later run.

**What that means for the run, said plainly:** every episode starts at the
same commanded pose — 30 mm above the entrance, on the pocket axis, upright.
The fixture is teleported +-5 mm per reset, but the start pose is solved ON
the pocket axis, so that jitter does not become a lateral start error (the
`(D-170)` clause in the line above says so). **A run from one fixed start
pose is not evidence about the task's difficulty**, which is a second reason
no success rate is read here.

## What this run cannot answer

* **Nothing about learning.** 30 iterations, fresh weights. `success_rate`,
  `mean_max_depth_mm` and every `Episode_Reward/<term>` will be printed and
  none of them is a criterion. `Laufplan_Phase5.md:36` says it: not a result.
* **Nothing about engaged-hover.** `Laufplan_Phase5.md:59-90` (step 6b) makes
  the engaged share an ABORT criterion for the BUDGET run, not for a smoke
  run, and 30 iterations is far too short to see the pattern. Precedent, named
  as a HINT with its setup difference in the same breath: in RT-179
  `Episode_Reward/engaged` held 183.6460 of `Mean reward` 231.39 (= 0.794) at
  iteration 196, with `success_rate 0.0000` and `mean_max_depth_mm 14.5132`
  (`rt_logs/RT-179/RT-179.txt:14696`, `:14703`, `:14705`, `:14707`, `:14717`),
  while at iteration 147 the same share was 33.5035 / 91.75 = 0.365
  (block-end rows, parsed today; the rows at a block START are buffer-refill
  artefacts, L-05). **RT-179 is NOT a number for this setup:** it carries the
  OLD six dims / eleven boundaries — a `lat_x`/`lat_y` box at ±0.006 m, yaw
  ±0.10471975511965977, tilt 0..0.13962634015954636, a TWO-sided start height
  0.02..0.05 centred 0.03 and friction centred **0.4** over 0.2..0.6
  (`rt_logs/RT-179/RT-179_autodr_1470.json`, read today) — whereas this build
  has five dims / seven boundaries: `lat_r` 0..0.030 (a 30 mm disk RADIUS),
  yaw ±0.08726646259971648, tilt 0..0.17453292519943295, a ONE-sided
  `start_height` on the still unmeasured H_min, and friction 0.08..0.20
  centred on `CONTACT_FRICTION = 0.14` (`autodr.py:433-454`;
  `insertion_tasks_cfg.py:1396`). **A clean 30-iteration smoke run is NOT
  evidence against engaged-hover.**
* **Nothing about the bias spread as a NUMBER.** See P4: two reset-straddling
  printouts settle the shape, never the distribution.
* **Nothing about parallel throughput.** That is RT-182's question, and it
  needs RT-181's `Iteration time` as its denominator.
* **Nothing about H_min, the net repair, or the teleport success test.** Those
  are `Laufplan_Phase5.md:47-48` (step 4), after this.
* **Which checkpoint files land — KNOWN since 2026-09-12, and the answer is
  NOT `model_30.pt`.** rsl_rl 3.0.1 IS on this laptop
  (`Thesis Workflow Phase 5\p5_robustheit_astra\.venv-review\Lib\site-packages\`,
  `rsl_rl_lib-3.0.1.dist-info`), and `rsl_rl/runners/on_policy_runner.py` was
  read. The loop is `for it in range(start_iter, tot_iter)` with
  `start_iter = self.current_learning_iteration` = 0 (`:58`, `:95-97`), so
  `it` runs **0..29** (L-01). Inside the loop, `if it % self.save_interval ==
  0` saves `model_{it}.pt` (`:159-160`), and `save_interval = 50`
  (`agents/rsl_rl_ppo_cfg.py:29`), so only `it = 0` matches. AFTER the loop,
  `on_policy_runner.py:174-175` saves
  `model_{self.current_learning_iteration}.pt` with no condition but
  `log_dir is not None and not self.disable_logs`, and
  `disable_logs = self.is_distributed and self.gpu_global_rank != 0` (`:51`)
  is False on one GPU. `current_learning_iteration` is assigned `it` at
  `:153`, so its final value is **29**. Expect `model_0.pt` and
  **`model_29.pt`**; at `--max_iterations 30` no `model_30.pt` is ever
  written. This file said the schedule was unknown because rsl_rl was not
  installed; that was wrong. Still not load-bearing: `params/env.yaml` is
  written before training starts (`train.py:621`).
