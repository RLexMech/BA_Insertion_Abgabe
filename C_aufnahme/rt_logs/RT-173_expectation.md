# RT-173 — expectation, written BEFORE the run (2026-09-07)

Written on the dev laptop. `/rt-check` judges the log against THIS file and
nothing else. Second rung of the tilt magnitude ladder under D-172; the
first rung was RT-172 (`rt_logs/RT-172_expectation.md`, P5 branch 1).

## The one hypothesis

**"The RT-172 policy (`model_600.pt`) learns a pocket tilted up to 8 deg by
fine-tuning under the reward as it stands."** RT-172 answered this for
0..5 deg with 1.0 in both bins. D-172 (3) makes the ladder the next
measurement and leaves the rung value to the user; the user set 8 deg on
2026-09-07. The one bound of ours: `osc_tilt_clamp_rad` = 8.52 deg
(`insertion_env_cfg.py:245`, CAD). At 8 deg the D-163 approach strategy
keeps 0.52 deg of commandable tilt; this is the last rung inside the cone.

## The one change

`env.fixture_tilt_noise_rad` 0.0873 -> 0.1396 rad (8.00 deg; `math.degrees`
= 7.9985, the tag rounds to `tilt0-8deg`, `train.py:220`). Magnitude uniform
in [0, 8 deg], direction uniform over 360 deg (`insertion_env.py:1715`).

Everything else exactly as RT-172 saw it: resume from the RT-172 run folder
`09-06_18-57-23_offset5mm_tilt0-5deg_currOFF_start+30..+50mm_seed42`,
checkpoint `model_600.pt` (user read it off the training PC 2026-09-07;
the last one before the stop at 632), starts sampled +30..+50 mm, tip box
0.07, `contact_penalty_scale` 0.0, `far_penalty_scale` 0.0, 1024 envs,
seed 42, force abort 50 N as `truncated`. Reward unchanged (D-172).

Readout by construction: `success_by_tilt_bin` edges 3/6/9/12/15 deg
(`insertion_env.py:703`). Uniform [0, 8] fills 0-3 (37.5 %), 3-6 (37.5 %)
and 6-9 (25 %, ~500 of 2000 window episodes); 9-12 and 12-15 stay
`episodes 0, rate None`, which is correct. The 6-9 deg bin is the NEW bin
of this rung; it holds only 6-8 deg, not 6-9.

**Two instrument limits, carried from RT-172, not fixed:** the tilt-bin
table exists only in the end window (no per-block start value, P4 there
NOT BELEGT); the height-bin table has ONE open bin above +30 mm, so the
+30..+50 mm band is not resolved by height (RT-172 P4b NACHTRAG).

## Budget

`--max_iterations 500` (mine, labelled: RT-172 saturated ~200 iterations
after its start; 500 leaves room for a slower rung and costs ~4.2 h at the
30.2 s/iteration RT-172 measured). `--max_iterations` is ADDITIVE on a
resume (RT-172, "Learning iteration 250/2250"), so the first line must read
`Learning iteration 600/1100`. Checkpoints every 50 iterations; a hand stop
loses at most 50. Env count stays 1024 (D-117 (c)).

## Commands (training PC, PowerShell, repo root, conda env `env_isaaclab`)

```
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase3_Implementierung_v2
git pull
git rev-parse --short HEAD
```

Expected HEAD: the SHA in the handover message (read back from
`git ls-remote origin p4-reward` on the laptop AFTER the push).

```
.\scripts\rt_log.ps1 RT-173 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 1024 --headless --seed 42 --max_iterations 500 --resume --load_run 09-06_18-57-23 --checkpoint model_600.pt env.fixture_tilt_noise_rad=0.1396 env.start_tip_above_entrance=0.050 env.start_tip_above_entrance_low=0.030 env.osc_pos_clamp_m=0.07 env.contact_penalty_scale=0.0 env.far_penalty_scale=0.0
```

`--load_run` takes the TIMESTAMP only; `train.py --checkpoint` takes the
FILE NAME. Hydra floats need the decimal point. Run in `env_isaaclab`, not
`(base)` -- RT-172's first two attempts died there on the `isaaclab` import.

Paste `rt_logs/RT-173.short.txt` and the run's `demo_metrics.json` into
`rt_logs/inbox.txt` on the laptop. Run folder tag expected:
`..._offset5mm_tilt0-8deg_currOFF_start+30..+50mm_seed42`.

## PASS / FAIL lines, in order

1. **P1 — the reset is contact-free.** `force_abort_rate` in the first
   block is NOT 1.0, and the startup-report force at step 2 is of the order
   of the reset transient (RT-149: 2.021 N), not tens of N. ESTIMATE, not
   measured: at 8 deg the stage-1 rim on the high side rises by
   sin 8 deg x 72..95 mm = 10.0..13.2 mm against the 15 mm the +30 mm start
   clears (`STAGE1_DEPTH` 0.015). That is 1.8 mm of margin at the LOWEST
   draw on the largest lever -- tighter than RT-172's ~7 mm. If P1 fails,
   the start height for this rung needs its own derivation and NOTHING
   below is read.
2. **P2 — the run is what it says.** Folder tag contains `tilt0-8deg` AND
   `start+30..+50mm`; startup report prints "tip clamp: +-0.07 m" and
   "fixture tilt noise (D-037): magnitude 0..8.00 deg ... ACTIVE"; the
   `[INFO]: Loading model checkpoint from:` line ends in
   `09-06_18-57-23_offset5mm_tilt0-5deg_currOFF_start+30..+50mm_seed42\model_600.pt`;
   first block reads `Learning iteration 600/1100`.
3. **P3 — the account adds up.** `reward_terms_recent_mean` rows `contact`
   and `far` EXACTLY 0.0; `abort` = -1.0 x `force_abort_rate`;
   `success_lump` > 0 iff `success_rate` > 0.
4. **P4 — the first block is READ, not judged.** First block
   `success_rate` and `force_abort_rate`, beside RT-172's 0.0070 / 0.0095.
   No per-bin start value exists (instrument limit above).
5. **P5 — THE BRANCH, read at the LAST block (`success_by_tilt_bin`,
   trailing window):**
   * 6-9 deg bin >= 0.90 AND 0-3 and 3-6 deg >= 0.95 -> the unchanged
     reward pays enough up to 8 deg; the ladder has reached the cone and
     the next question is the cone itself (`osc_tilt_clamp_rad`), not the
     reward.
   * 6-9 deg bin <= 0.50 while 0-3 deg stays >= 0.90 -> the rung FAILS
     under the unchanged reward. This is the measured failure D-172 (2)
     names as the condition for reopening the alignment term; the
     fallback in `HANDOFF-RL.md` § Open 2d then has its target number.
   * ALL bins fall under 0.50 -> forgetting or a broken resume, NOT a
     rung finding; check P2 and compare RT-172 (1.0 / 1.0 at 632).
   * Between the lines -> report the three numbers and the trend over
     the blocks; do not conclude.
6. **P6 — force is READ.** `force_norm_n` p95 / max and `force_abort_rate`
   over the last window, beside RT-172 (26.54 / 39.40 N, 0.0).
7. **P7 — sigma survives.** `Mean action noise std` at the last block
   > 0.30 (RT-172 ended at 0.57). Under 0.05 = collapse; P5 then reads as
   unexplored.
8. **P8 — the solver still lands.** `start_pose_solve_unconverged_resets`
   0 and `start_pose_solve_worst_residual_mm` under 0.05 (RT-172: 0.0400
   mm at 5 deg; a rising residual with tilt is an instrument finding).

## What this run does NOT answer

* Anything above 8 deg: the controller cone is 8.52 deg; a rung beyond it
  needs a cone decision first, not a run.
* The tilt x height cross, and the per-bin START value (instrument limits).
* Yaw (`fixture_yaw_noise_rad` stays 0.0; RT-152 unmeasured) and the
  curriculum (`curriculum_enabled` False, `rung_step_sizes` None).
