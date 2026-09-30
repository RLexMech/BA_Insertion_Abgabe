# RT-182 — expectation, written BEFORE the run (2026-09-12)

**A SMOKE RUN, NOT A RESULT.** `Laufplan_Phase5.md:36` calls step 3 the
"No-DR smoke, 30 iterations, not a result." Nothing about learning is read
here. Status: UNVERIFIED until the training PC.

**(Erwartung korrigiert 2026-09-12, vor dem Urteil.)** Corrected BEFORE any
log was read, so the verdict carries `(Erwartung korrigiert)`, the mark
`.claude/skills/rt-check/SKILL.md` step 4b prescribes. What changed: the SHA
and the `insertion_env.py` line numbers that moved with it; P6's criterion,
which could not tell `abs` from `add`, replaced by the pooled magnitude rule
that RT-181 P4 owns; the false "identical by construction" warrant for counts
60 and 180; and the `dr_mode` literal expected in `params/env.yaml` (P5).

Code: `origin/p5-robustheit` = **`23a7aa0`** — the commit the operator pulled
and ran. This file first named `9f37a07`, the commit BEFORE the pin and dump
repairs. Two commits landed after `23a7aa0` and neither touches the env:
`git diff --stat 23a7aa0..4725840 -- source/ scripts/` reports exactly one
file, `scripts/shorten_rt_log.py` (45 insertions, 10 deletions). So `23a7aa0`
is the code that ran. `git diff 9f37a07..23a7aa0 --stat` gives
`insertion_env.py | 10 +-` — a 2-line comment replaced by 8 at old
1729-1730 — so every cited `insertion_env.py` line from old 1731 on moves
**+6** (checked line by line against `git show 23a7aa0:<path>`, not only by
arithmetic). No other file cited below changed between the two commits.
NOTHING in this build has ever run under Isaac. `dr_mode` defaults to `"off"`
(`insertion_env_cfg.py:629`), so no hydra override is passed.

RT-182 is part (b) of step 3: FIVE runs at once, training seeds 1..5.
Part (a) — one run alone, training seed 20 — is RT-181, and its
`Training time` is the DENOMINATOR of the rule below. **RT-181 must be
finished and read before RT-182 is judged.**

## The one hypothesis

**Five 1024-env runs fit on this GPU at once and are worth running that way.**
That is a throughput-and-capacity question, not a correctness question:
whether the build works at all is RT-181's hypothesis and is not re-asked
here. The rule that decides it is fixed BEFORE the run and is
`Laufplan_Phase5.md:41-43`, verbatim:

> Rule, set before: five together give more iterations per hour than one alone
> AND none crashes → the seeds run in parallel. Otherwise test two at once,
> same rule.

Why it matters beyond convenience: `Laufplan_Phase5.md:13` requires all five
seeds of a step to get the same number of iterations and forbids a seed
running on alone, so the five seeds of every later condition are a block. How
that block is scheduled is decided here, once.

## Command (training PC, PowerShell, conda env `env_isaaclab`)

Read `nvidia-smi` BEFORE any window is started (the baseline) and again while
all five are running, per `Laufplan_Phase5.md:39-40`:

```
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

Five PowerShell windows, started as close together as possible, seeds 1..5:

```
.\scripts\rt_log.ps1 RT-182s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 1 --max_iterations 30
.\scripts\rt_log.ps1 RT-182s2 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 2 --max_iterations 30
.\scripts\rt_log.ps1 RT-182s3 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 3 --max_iterations 30
.\scripts\rt_log.ps1 RT-182s4 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 4 --max_iterations 30
.\scripts\rt_log.ps1 RT-182s5 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 5 --max_iterations 30
```

Training seeds 1..5, per `Laufplan_Phase5.md:9`. One NAME per seed, so
`rt_log.ps1:51` writes five separate logs `rt_logs\RT-182s1.txt` ..
`rt_logs\RT-182s5.txt` — the same NAME would APPEND five runs into one file
(`rt_log.ps1:21`).

**THE TWO COMMAND-LINE TRAPS, and why none of the five carries either.**

1. **`env.osc_pos_clamp_m` is NOT passed** — `Laufplan_Phase5.md:19-24`: the
   cfg default is the only source of the box half-width, and the 0.07 m that
   older RT commands carry is SMALLER than the 0.070838 m the disk start
   demands.
2. **`env.contact_penalty_scale` / `env.far_penalty_scale` are NOT passed** —
   removed from the code with Commit A0 (`Laufplan_Phase5.md:33-34`, D-184).
   Verified on the laptop today:

   ```
   $ grep -rn "contact_penalty_scale\|far_penalty_scale" source/ scripts/
   $ echo $?
   1
   ```

   Empty result, exit status 1 — neither name occurs in `source/` or
   `scripts/`. An old command carrying them fails in hydra: the keys are not
   in the node registered at `isaaclab_tasks/utils/hydra.py:58`, built from
   `env_cfg.to_dict()` at `hydra.py:49`. The exact error text is **unknown for
   this setup**. Copying a command from `docs/runs/RT-172.md`..`RT-176.md`
   five times is exactly how this trap gets sprung.

## PASS / FAIL lines, in order

**P1 — ALL FIVE START AND ALL FIVE FINISH. THIS IS A PRECONDITION OF THE
RULE, NOT A SEPARATE NICE-TO-HAVE.** Each of the five logs ends with
`Learning iteration 29/30` (L-01: 30 iterations are numbered 0..29) and
`[rt_log] exit code: 0` (`rt_log.ps1:96`). FAIL: any window dies — a CUDA
out-of-memory, a driver reset, an Isaac Sim startup failure, or a silent
window. **One crash in five settles the rule against parallel five by itself**
("AND none crashes", `Laufplan_Phase5.md:41-42`) and sends the answer to the
fallback in P2c, regardless of how good the throughput number looks.

**P2 — THE THROUGHPUT RULE. THIS IS THE DECISION.**

*The two numbers that get divided, and where each is read:*

* **T_solo** — `Training time: <x> seconds`, one line, from
  `rt_logs\RT-181s20.txt` (printed at `train.py:775`; the clock starts at
  `train.py:519`, after the env is built, so Isaac boot is excluded).
* **T_par** — the LARGEST `Training time: <x> seconds` of the five logs
  `rt_logs\RT-182s1.txt` .. `rt_logs\RT-182s5.txt`. The largest, not the mean:
  the block is done when the slowest seed is done.

```
Select-String -Path rt_logs\RT-181s20.txt,rt_logs\RT-182s*.txt -Pattern "Training time:"
```

*The quotient:*

```
Q = T_par / T_solo
```

*The rule, worked out from `Laufplan_Phase5.md:41-43`.* Five together produce
5 * 30 = **150** iterations in T_par. One alone produces 30 iterations in
T_solo, so five one after another take 5 * T_solo. "More iterations per hour"
is therefore

```
150 / T_par  >  150 / (5 * T_solo)     <=>     T_par < 5 * T_solo     <=>     Q < 5
```

* **PASS (Q < 5 AND P1 holds): the seeds run in parallel, all five at once.**
* **FAIL (Q >= 5, or any crash in P1):** five at once buys nothing. Then, per
  `Laufplan_Phase5.md:42-43`, **test TWO at once and apply the same rule** —
  with 2 in place of 5, i.e. two together pass iff
  T_par(2 seeds) < 2 * T_solo. That second measurement is its own RT number
  and is not part of RT-182.

*The cross-check, same rule on a different instrument.* `Iteration time: <x>s`
is printed per iteration (present as `rt_logs/RT-179/RT-179.txt:371`,
`:443`). Compare the MEDIAN over iterations 1..29 of the five parallel logs
against the median over iterations 1..29 of RT-181. **Iteration 0 is excluded
from both** — it carries a warm-up: in RT-179 iteration 0 cost 22.36 s and
iteration 1 cost 16.75 s (`rt_logs/RT-179/RT-179.txt:371` and `:443`), a
different configuration but the same instrument, and L-05 forbids reading a
run's first value as its steady state. The two instruments must agree on the
SIDE of 5; if they do not, `Training time` decides, because it is the number
the scheduling question is actually about.

**What Q will be is unknown for this setup.** Neither T_solo nor T_par exists
yet, and no old run sets either: RT-179 ran one process with
`env.dr_mode=autodr` and the two now-deleted penalty scales
(`rt_logs/RT-179/RT-179.txt:3`) and never ran concurrently with anything.

**P3 — THE FIVE FIT IN VRAM.** The card is an NVIDIA GeForce RTX 3080 with
**10053 MB** of GPU memory (`rt_logs/RT-179/RT-179.txt:21`, same training PC —
cwd `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`,
`rt_logs/RT-179/RT-179.txt:2`). That is the only hard ceiling that exists
before the run.

PASS: the `memory.used` reading taken while all five are running stays below
10053 MB and no log carries a CUDA out-of-memory. The per-process budget is

```
(10053 MB - baseline) / 5
```

with `baseline` = the `memory.used` reading taken BEFORE any window starts
(desktop, browser, whatever else holds VRAM). With a baseline of 0 that is
**2010.6 MB** per process, which is the most optimistic form of the budget and
not a prediction. **What ONE 1024-env run of this build actually uses is
unknown for this setup** — RT-181's P8 is the run that establishes it, which
is why `Laufplan_Phase5.md:37` orders (a) before (b). If RT-181's reading
already exceeds one fifth of the free memory, RT-182 can be expected to fail
P1 on memory and the two-at-once fallback is the next test.

FAIL means P1 fails with it; record both readings either way, because the
per-process number is what sizes every later block of five seeds.

**P4 — THE FIVE RUNS DO NOT OVERWRITE EACH OTHER ON DISK.** Three separate
collision surfaces, all three expected to hold, each checked:

* **Run folder.** `log_dir` is a timestamp to the SECOND (`train.py:259`) plus
  a `seed<N>` suffix (`train.py:412-413`, `train.py:417-418`). Five windows
  started inside one second would share the timestamp; the seed suffix is what
  keeps them apart. PASS: five distinct directories under
  `logs\rsl_rl\ur5e_insertion\`, one per seed. FAIL: fewer than five.
* **`demo_metrics.json`.** Written to `Path(log_dir) / "demo_metrics.json"`
  when `cfg.log_dir` is set and to the FIXED path `logs/demo_metrics.json`
  when it is not (`insertion_env.py:3439-3440`, fallback set at
  `insertion_env.py:1219`). `train.py:430` sets it. PASS: five files, one per
  run folder. FAIL: one file at `logs\demo_metrics.json` — then the five runs
  overwrote each other and every later block of five seeds has the same bug.
* **The clipboard.** `rt_log.ps1:129` puts each run's short log on the
  clipboard when that run ends, so with five windows only the LAST one to
  finish is on the clipboard. Not a defect — but the four other short logs
  must be taken from disk (`rt_logs\RT-182s<N>.short.txt`,
  `rt_log.ps1:111`), not from the clipboard.

**P5 — THE FOUR STEP-3 READINGS REPEAT IN EVERY SEED.** `Laufplan_Phase5.md:43-44`
lists them; each is already a PASS line of RT-181, so here the question is
only whether they survive five-way contention and differ across seeds:

1. **Noise model built.** Every one of the five logs carries the MODEL-BUILT
   form of the bias line (`insertion_env.py:4439-4441`), never the
   `OFF -- both sigmas are 0.0` form (`insertion_env.py:4433-4434`). Both
   sigmas are at their non-zero cfg defaults —
   `obs_noise_pocket_pos_std_m = 0.0025` (`insertion_env_cfg.py:1041`),
   `force_obs_noise_std_n = 3.5` (`insertion_env_cfg.py:1042`) — so
   `resolve_obs_noise_model` (`obs_noise.py:218-221`) returns a cfg in all
   five.
2. **`params/env.yaml` written.** Five files, each carrying
   `obs_noise_pocket_pos_std_m: 0.0025`, `force_obs_noise_std_n: 3.5`,
   `grasp_obs_offset_x_m: 0.003` and **`dr_mode: 'off'` — WITH the single
   quotes**, because PyYAML quotes the string so it does not round-trip as the
   boolean False (`python -c "import yaml;
   print(repr(yaml.dump({'dr_mode':'off'})))"` prints `"dr_mode: 'off'\n"`,
   run on the laptop 2026-09-12; RT-181 P5 owns the check). This file expected
   the bare literal `dr_mode: off`; that literal never appears and its absence
   is NOT a FAIL. `train.py:621` writes the file. What the
   `observation_noise_model` key itself reads is **unknown for this setup** —
   see RT-181 P5; record it, do not judge it.
3. **Grasp offset in the startup report.** The line
   `grasp belief error env0..3 (D-183, +-0.003 m ...)`
   (`insertion_env.py:4442-4445`) appears in all five, every printed value
   satisfying |v| <= 0.003 exactly, because the draw is `(2u - 1) * 0.003`
   with `u` uniform on [0, 1] (`insertion_env.py:2228-2230`,
   `grasp_obs_offset_x_m = 0.003` at `insertion_env_cfg.py:1050`).
4. **Pocket bias does not grow across resets.** See P6.

**THE CROSS-SEED READING, which only RT-182 can give.** Five different
training seeds must produce five DIFFERENT step-2 bias rows and five different
grasp-offset rows. PASS: the seed-1..seed-5 rows differ from one another.
FAIL: two seeds print identical rows — then the draws do not depend on
`cfg.seed` and every "five seeds" statement later in Phase 5 is five copies of
one run.

**P6 — THE POCKET BIAS IS RE-DRAWN, NOT ACCUMULATED. THE CRITERION IS POOLED
OVER SIX LOGS, SO IT IS NOT DECIDED INSIDE RT-182 EITHER.**
`report_at_steps = (2, 60, 180, 258)` (`insertion_env_cfg.py:796`). The
per-episode bias is written ONLY in the noise model's `reset()`
(`obs_noise.py:51-58`, `insertion_env.py:4419-4425`), and the episode cap is
256 control steps — `ceil(episode_length_s / (sim.dt * decimation))` with
`episode_length_s = 256/60` (`insertion_env_cfg.py:138`), `sim.dt = 1/120`
(`insertion_env_cfg.py:519`), `decimation = 2` (`insertion_env_cfg.py:122`),
pinned at `insertion_env_cfg.py:1163`.

**THE OLD CRITERION DECIDED NOTHING AND IS REPLACED.** This file used to ask
only that each log's step-258 rows "differ from that log's step-2 rows".
`gaussian_noise` returns `data + mean + std*randn` under `add` and
`mean + std*randn` under `abs`
(`C:\IsaacLab\source\isaaclab\isaaclab\utils\noise\noise_model.py:92-97`),
so count 258 holds `x1 + x2` under `add` and `x2` under `abs` — and both
differ from the count-2 value `x1`. The test passed under either operation.

**The replacement is RT-181 P4's POOLED MAGNITUDE RULE, which owns it.**
`R = mean|v|(258) / mean|v|(2)` over the **72** numbers of all SIX step-3 logs
(`RT-181s20` plus these five; 4 envs x 3 channels x 6 runs), against a bar of
**1.22** derived there from a 200 000-trial Monte Carlo — `abs` gives
`R ~ 1.0`, `add` gives `R ~ sqrt(2) = 1.414`. **RT-182 supplies 60 of the 72
numbers and still cannot settle it alone:** 12 numbers from a single log
misread the operation about one time in four (the table in RT-181 P4). Read
the step-2 and step-258 bias printouts out of all six logs, pool them, and
write the P6 verdict once.

**"2, 60 and 180 are identical by construction" was FALSE and is gone.**
`train.py:725` passes `init_at_random_ep_len=True`
(`grep -n init_at_random_ep_len scripts/rsl_rl/train.py`), so rsl_rl 3.0.1
randomises every env's episode counter before the first step
(`on_policy_runner.py:66-69`) and the envs reset at different times. **A bias
row that has changed at count 60 or count 180 is an early reset, not a
finding.** Count 258 still lies past every env's first reset, but with
**margin zero**: with the time-out rule
`episode_length_buf >= max_episode_length - 1` (`insertion_env.py:1890`) and
`max_episode_length = 256`, the env that draws `b = 0` first resets at step
255 — one observation call before the report at 258. The full derivation is
in RT-181 P4 and is not repeated here.

**THE `t = ... s after reset` IN EVERY BLOCK HEADER IS NOT A PER-ENV TIME.**
It is `self._obs_calls * cfg.sim.dt * cfg.decimation`
(`insertion_env.py:4050`), the run's own elapsed sim time. With envs
resetting at different steps it says nothing about any env's episode phase.
RT-181 P6 owns this warning.

**WHAT RT-182 ADDS ON ITS OWN IS NOT P6.** It is the cross-seed reading in
P5 — five different step-2 bias rows and five different grasp-offset rows —
and that one IS decidable inside RT-182. The magnitude of the bias, its
distribution and its sigma stay **unknown for this setup**: the DECLARED
sigma is 0.0025 m (`obs_noise.py:146-148`, sigma from
`insertion_env_cfg.py:1041`), 3 sigma = 0.0075 m, and with 12 numbers per
printout the chance that at least one lands outside ±3 sigma is 0.0319
(`1 - 0.9973002039367398**12`), so a single large value is not a finding.

The fourth report requires `_obs_calls` to pass 258
(`insertion_env.py:298-299`, `insertion_env.py:1762`); 30 iterations at
`num_steps_per_env = 16` (`agents/rsl_rl_ppo_cfg.py:27`) is 480 control steps,
so it fires. `Laufplan_Phase5.md:45-46`, verbatim:

> Why 30 is enough: 16 steps per iteration, an episode ends at 256 steps
> = 16 iterations, so every env resets at least once.

256 / 16 = 16 iterations; 30 > 16.

## What this run cannot answer

* **Nothing about learning.** Five times 30 iterations on fresh weights.
  `Episode/success_rate`, `mean_max_depth_mm` and every
  `Episode_Reward/<term>` are printed in all five and none of them is a
  criterion. `Laufplan_Phase5.md:36`: not a result. In particular, five seeds
  agreeing on a bad success rate at iteration 29 is not a seed study.
* **Nothing about engaged-hover — and a clean run here is NOT evidence against
  it.** The engaged-share abort is step 6b (`Laufplan_Phase5.md:59-90`) and
  belongs to the BUDGET run. As a HINT with its setup difference named in the
  same breath: RT-179's `Episode_Reward/engaged` first held more than half the
  return late — 183.6460 of `Mean reward` 231.39 (= 0.794) at iteration 196,
  with `success_rate 0.0000` and `mean_max_depth_mm 14.5132`
  (`rt_logs/RT-179/RT-179.txt:14696`, `:14703`, `:14705`, `:14707`, `:14717`),
  against 33.5035 / 91.75 = 0.365 at iteration 147 (block-end rows, parsed
  today; rows at a block START are buffer-refill artefacts, L-05). **RT-179
  sets no number for this setup:** it carries the OLD six dims / eleven
  boundaries — `lat_x`/`lat_y` box ±0.006 m, yaw ±0.10471975511965977, tilt
  0..0.13962634015954636, a TWO-sided start height 0.02..0.05 centred 0.03,
  friction centred **0.4** over 0.2..0.6
  (`rt_logs/RT-179/RT-179_autodr_1470.json`, read today) — while this build has
  five dims / seven boundaries: `lat_r` 0..0.030 (a 30 mm disk RADIUS), yaw
  ±0.08726646259971648, tilt 0..0.17453292519943295, a ONE-sided
  `start_height` on the still unmeasured H_min, friction 0.08..0.20 centred on
  `CONTACT_FRICTION = 0.14` (`autodr.py:433-454`;
  `insertion_tasks_cfg.py:1396`). 30 << 196 either way.
* **Nothing about long-run throughput.** Q is measured over 30 iterations from
  a cold start. Whether the same quotient holds over a run of hundreds of
  iterations — thermal throttling, VRAM fragmentation, a growing replay of
  logging — is **unknown for this setup**.
* **Nothing about TWO at once.** If P2 fails, the fallback measurement
  (`Laufplan_Phase5.md:42-43`) is a separate run with its own RT number.
* **Nothing about whether the build is correct.** That is RT-181's question.
  If RT-181 failed, RT-182 measures the throughput of a broken build and its
  Q is worthless.
* **Nothing about AutoDR.** `dr_mode: off` in all five
  (`insertion_env_cfg.py:629`); no provider, no boundaries, no
  `autodr_<it>.json`. The AutoDR smoke is step 10
  (`Laufplan_Phase5.md:103`).
