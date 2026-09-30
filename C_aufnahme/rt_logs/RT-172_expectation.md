# RT-172 — expectation, written BEFORE the run (2026-09-06)

Written on the dev laptop while the user is away. `/rt-check` judges the log
against THIS file and nothing else. RT-171 is RESERVED for the alignment-term
run on `p5-reward` (`HANDOFF-RL.md` § Open 2d) and is not touched here.

## The one hypothesis

**"The RT-156 policy (`model_250.pt`) learns a TILTED pocket by fine-tuning
under the reward as it stands."** RT-170 BEFUND 2 measured the kernel nearly
blind to tilt (1.02e-5 per degree against 2.35e-4 per mm); the handoff named
the reward the suspect and proposed an alignment term (RT-171). Nobody has
measured what the CURRENT reward does with a tilted pocket. This run is that
measurement, and it is the BASELINE RT-171 will be compared against: same
checkpoint, same task, one reward term more. Without this run RT-171 has no
control.

Why fine-tune, not train from scratch: RT-160 resumed the same checkpoint at
a fixed +40 mm and read success 0.9860 over 2000 episodes after 50
iterations (`VERDICTS.md` 2026-09-06 16:15:10). The resume path works and is
cheap. A from-scratch run under a reward already under suspicion would cost a
night and answer a different question.

## The one change

`env.fixture_tilt_noise_rad` 0.0 -> 0.0873 rad (5 deg). Per episode each env
draws a tilt MAGNITUDE uniform in [0, 5 deg] and a tilt DIRECTION uniform
over 360 deg (`insertion_env.py:1715-1730`, D-037). The pocket frame, the
start-pose solver (`_solve_start_pose`, pocket-frame goal, `:1826`) and
observation channels 21:25 all follow the tilted fixture.

**Start height: SAMPLED +30..+50 mm (user, 2026-09-06), NOT the -30..+30 mm
band RT-156 trained on.** Read off the code: `insertion_env.py:422-440`
REFUSES a negative `start_tip_above_entrance_low` together with
`fixture_tilt_noise_rad` (the start IK commands position only; an
un-commanded orientation inside the pocket is the RT-120 wall contact). So
the old band cannot stay. The user chose +30..+50 mm: the lower bound is
D-163's rung 0 (USER-SET), the upper bound covers the +40 mm dip. WITHOUT
tilt this policy was measured at those heights: +30 mm 100 % (RT-157h30,
613 ep.), +40 mm 58.2 % first-episode (RT-157h40t), +50 mm 76.8 %
(RT-157h50). So the height half of this run has a baseline per bin
(`success_by_start_height_bin`, 10 mm bins), and the tilt half has none.
NAMED: this is TWO changes on top of the checkpoint (band + tilt), not one;
the per-height and per-tilt bins separate them only marginally, never
jointly. The cross (tilt x height) is NOT read out by any instrument.

**`osc_pos_clamp_m` 0.05 -> 0.07.** The +50 mm start sits ON the edge of the
default +-50 mm tip box; outside it the clamp pulls the target to the box
edge (RT-143's "115 mm sink"). Every +40/+50/+60 mm probe of this checkpoint
ran at 0.07 (RT-157h40/h50/h60, RT-158, RT-159), and at +30 mm the 0.07 box
read 100 % (RT-157h30), so the wider box is measured harmless for the
policy. Third change, same value as every prior use.

Everything else exactly as the checkpoint saw it: osc, kp 100, kp_rot 30,
15 Hz, 256 steps (17.07 s), 1024 envs, seed 42, `fixture_pos_noise_xy`
0.005, `action_rate_scale` 0.0034 (D-168), `contact_penalty_scale` 0.0 and
`far_penalty_scale` 0.0 (the two RT-156 overrides -- RT-156p FAILED because a
replay dropped them; they are NOT the cfg defaults, `insertion_env_cfg.py:1041,
:1073`), force abort 50 N as `truncated` (D-169/D-164).

## Why 5 deg, stated plainly

**There is NO source of ours for the magnitude** (`HANDOFF-RL.md` § Open 2c:
D-110 (4)'s cell-derived range does not exist; the env comment cites the OLD
repo's D-037, a proxy pattern). 5 deg is that pattern's first rung. The one
bound of OURS: `osc_tilt_clamp_rad` = 8.52 deg (CAD, "Einfaedeln",
`insertion_env_cfg.py:245`) is the most the TOOL may tilt from vertical. A
pocket tilted 5 deg leaves 3.52 deg of approach tilt for the D-163 strategy;
8 deg would leave 0.52 deg and 15 deg (the old ladder's second rung) is
outside what the controller may command. 5 deg is a BUDGET number, labelled
USER-SET/pattern, not derived. If the user wants another value, only this
number and the tag change.

Readout by construction: `success_by_tilt_bin` has edges 3/6/9/12/15 deg
(`insertion_env.py:703`), so this run fills the 0-3 and 3-6 deg bins ONLY;
the three upper bins read `episodes 0, rate None` and that is correct, not a
defect.

## Budget

`--max_iterations 2000` (user, 2026-09-06: "give it more time to learn", raised
from 1000 the same evening; the user is away for a long stretch).
How rsl_rl 3.0.1 counts this on a resume is UNVERIFIED (rsl_rl is not on the
laptop): if the flag is ADDITIONAL to the loaded iteration 250 the run does
2000 iterations; if it is ABSOLUTE it does 1750. RT-160 printed "Iter 50/300"
with `--max_iterations 300`, which reads like a counter restarted at 0, but
the log is not in the repo. Either way the run does at least 1750 iterations
-- read the first "Learning iteration" line and report which it is. Time:
RT-156 ran 282 iterations between 19:44:54 and before 00:00:27 at 1024
envs, so at most ~54 s per iteration; 2000 iterations are therefore at most
~30 h. RT-148b ran 4000 iterations in one go, so the length itself is not new. Read the seconds per iteration off the first block and report them.
Checkpoints every 50 iterations (`save_interval`, `rsl_rl_ppo_cfg.py:20`),
so a hand stop loses at most 50.

**Env count stays 1024, deliberately.** D-117 (c) forbids varying the env
count between compared runs, and RT-156 / RT-160 / this run are one
comparison. 2048 envs have never run under OSC on that machine (memory
UNKNOWN) -- a crash at start would waste the whole absence. And the batch
size question (does it alone move sigma?) is RT-154's, still open; it must
not be folded into this run.

## Commands (training PC, PowerShell, repo root, conda env active)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: the SHA in the handover message (read back from
`git ls-remote origin p4-reward` on the laptop AFTER the push).

```
.\scripts\rt_log.ps1 RT-172 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max_iterations 2000 --resume --load_run 09-05_19-45-03 --checkpoint model_250.pt env.fixture_tilt_noise_rad=0.0873 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.osc_pos_clamp_m=0.07 env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0
```

`--load_run` takes the TIMESTAMP only (it is a regex, `parse_cfg.py`); the
folder is `09-05_19-45-03_offset5mm_currOFF_start-30..+30mm_seed42` (the
RT-157/RT-158 checkpoint path). `train.py --checkpoint` takes the FILE NAME,
`play.py --checkpoint` the full path -- do not mix them. Hydra floats need the
decimal point (`0.0`, not `0`; `PROBLEMS.md` RT-139b).

Paste `rt_logs/RT-172.short.txt` (on the clipboard when the run ends) and the
run's `demo_metrics.json` into `rt_logs/inbox.txt` on the laptop. Run folder
tag expected: `..._offset5mm_tilt0-5deg_currOFF_start+30..+50mm_seed42`.

## PASS / FAIL lines, in order

1. **P1 — the reset is contact-free.** `force_abort_rate` in the first
   iteration block is NOT 1.0 (RT-120 failure shape), and the startup report
   force at step 2 is of the order of the reset transient (RT-149 measured
   2.021 N as the episode maximum in free flight), not tens of N.
   ESTIMATE, not measured: with the fixture tilted 5 deg the stage-1 rim on
   the high side rises by about 6-8 mm (sin 5 deg times 72-95 mm of fixture
   half-extent) against the 15 mm the +30 mm start clears (`STAGE1_DEPTH`
   0.015); the vertical part should hang ~7 mm free at the LOWEST draw and
   more above it. If P1 fails, the start
   height for tilted rungs needs its own derivation and NOTHING below is read.
2. **P2 — the run is what it says.** Run folder tag contains `tilt0-5deg`
   AND `start+30..+50mm` (not `start-30..+30mm`); startup report prints
   "tip clamp: +-0.07 m"; startup report line
   "fixture tilt noise (D-037): magnitude 0..5.00 deg ... ACTIVE"; the
   `[INFO]: Loading model checkpoint from:` line ends in
   `09-05_19-45-03_offset5mm_currOFF_start-30..+30mm_seed42\model_250.pt`.
3. **P3 — the account adds up.** `reward_terms_recent_mean` rows `contact`
   and `far` EXACTLY 0.0; `abort` = -1.0 x `force_abort_rate` (D-164
   shape); `success_lump` > 0 iff `success_rate` > 0.
4. **P4 — the untrained policy is worse at larger tilt, at the START.**
   PREDICTION (mine, order of magnitude): in the FIRST block's
   `success_by_tilt_bin` the 0-3 deg bin reads higher than the 3-6 deg bin.
   RT-158 measured the successful episodes' own part tilt at a 2.23 deg
   median, so a pocket within 3 deg is inside what the policy already does;
   3-6 deg is not. If the two bins read equal at iteration 0 the tilt does
   not matter to this policy and the whole RT-170 line is weaker than
   stated.
4b. **P4b — the height half reproduces its baseline at the START.** First
   block `success_by_start_height_bin`: the 30-40 mm bin above the 40-50 mm
   bin is NOT predicted (RT-157 read 100 / 58 / 77 % at 30 / 40 / 50, not
   monotone). Read only: both bins populated (~half the episodes each), and
   the LAST block's 40-50 mm bin >= 0.90 -- RT-160 reached 0.9860 at a
   fixed +40 mm after 50 iterations, so a band that does not get there in
   2000 says the tilt costs the height, and P5 is read with that in mind.
   **NACHTRAG 2026-09-06, read off the code AFTER the run started and BEFORE
   any log came back (Fable 5.1, laptop): P4b CANNOT be read by this
   instrument.** `insertion_env.py:715` fixes the height bin edges at
   `(-0.020, -0.010, 0.0, 0.010, 0.020, 0.030)` m and the last bin is open
   at the top, so EVERY start of this run (+30..+50 mm) lands in the one
   ">= +30 mm" bin. There are no 30-40 and 40-50 mm bins. The table
   reduces to the overall rate, exactly as its docstring says for the
   fixed start. What survives of P4b: the last bin holds ~all episodes,
   and its LAST-block rate is the height half's number, unresolved by
   height. The prediction is NOT changed here; the readout is corrected.
   The tilt bins (P4, P5) are unaffected: their edges are 3/6/9/12/15 deg.
5. **P5 — THE BRANCH, read at the LAST block (`success_by_tilt_bin`,
   trailing window):**
   * 3-6 deg bin >= 0.90 AND 0-3 deg bin >= 0.95 -> the current reward pays
     enough for a 5 deg pocket through the success lump and D-165's
     progress term. The alignment term (RT-171) then needs a motivation
     other than RT-170's kernel ratio, and the next question is the
     magnitude ladder, not the reward.
   * 3-6 deg bin <= 0.50 while 0-3 deg stays >= 0.90 -> the tilt IS the
     blocker under this reward. RT-171 has its baseline and its target
     number.
   * BOTH bins fall under 0.50 -> forgetting or a broken resume, NOT a
     reward finding; compare with RT-160 (0.9860 after 50 iterations) and
     check P2 before anything else.
   * Between the lines -> report the two numbers and the trend over the
     blocks; do not conclude.
6. **P6 — force is READ, not predicted.** `force_norm_n` p95 and
   `force_abort_rate` over the last window; RT-160 read a max of 51.64 N
   against the 50 N line. Report both beside the RT-156 values (p95
   27.27 N).
7. **P7 — sigma survives.** `Mean action noise std` at the last block
   > 0.30 (RT-156 trend 1.00 -> 0.83 at iteration 282). Under 0.05 = the
   RT-148b collapse; then P5 is read as unexplored, not as learned.
8. **P8 — the solver still lands.** `start_pose_solve_unconverged_resets`
   0 and `start_pose_solve_worst_residual_mm` under the 0.05 mm tolerance
   (RT-160: 0.0354 mm on the level pocket). A tilted goal that stops
   converging is an instrument finding and blocks P5.

## Replay of the RT-172 checkpoint (NACHTRAG 2026-09-06, read off the code, no run)

`play.py` builds the env cfg through `hydra_task_config` (`play.py:185`) from
the cfg DEFAULTS plus the command line -- it does NOT read the run's saved
cfg. The defaults are `fixture_tilt_noise_rad` 0.0, `osc_pos_clamp_m` 0.05,
`start_tip_above_entrance` / `_low` the -30..+30 mm band, and the two
penalty scales at their NON-zero defaults (`insertion_env_cfg.py:1041,
:1073`; RT-156p FAILED on exactly this). So a replay that leaves the
overrides off measures a LEVEL pocket in the old band. The six
`env.*=...` overrides of the training command must be repeated verbatim.
The line `env_cfg.fixture_tilt_noise_rad = 0.0` at `play.py:306` sits
INSIDE the `--fixture-yaw` branch (`:289`), and the `--fixture-tilt` branch
(`:270-273`) zeroes only pos and yaw noise; neither fires on a plain
`--load_run` replay. `--fixture-tilt <deg>` is the DETERMINISTIC single-pose
probe and is a different measurement from the sampled 0..5 deg.

## What this run does NOT answer

* The tilt MAGNITUDE ladder (D-110 (4)): 5 deg is one rung, pattern-sourced.
* Yaw: `fixture_yaw_noise_rad` stays 0.0. RT-152 (the yaw slice) is still
  the next MEASUREMENT on this branch and is unaffected.
* The curriculum: `curriculum_enabled` stays False, `rung_step_sizes` stays
  `None` (§ Open 2b). This is a fixed-rung run, and the tag says `currOFF`.
* Anything at +40 mm: modes A and B live outside the trained band and are
  not probed here.
