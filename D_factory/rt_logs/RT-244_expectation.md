# RT-244s1..s5 -- expectation, written BEFORE the run (2026-09-29): the No-DR arm, five seeds, 100 epochs, on D-204's code (rl_games)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Templates: `rt_logs/RT-242_expectation.md` (section RT-242s1..s5) and
`rt_logs/RT-243_expectation.md`. Unchanged sections point there; only what
differs is written out.

## Context

D-195: three arms, five seeds each, one budget (100 epochs, D-205).
AutoDR ran as RT-242s, Fixed-DR as RT-243s. This chain is the third arm,
No-DR: `dr_mode=off` WITH the start floor (`insertion_env.py:458-471`): an
AutoDR provider with ONE dimension, `start_height` -- first the floor
(-30 mm up to H_min), then the start-height ceiling alone, up to 0.270 m
(`autodr.DR_DIMS`, `autodr.py:457`). The same curriculum as the other arms
for the start height (user 2026-09-14, `insertion_env_cfg.py` comment at
`start_floor_m`). "No-DR" is NOT "no spread" (`eval_plan.md` 3.2): the
start height grows, the fixture spread 0.05 m in x, y, z
(`insertion_env_cfg.py:783`, `:792`) and the observation noise stay on.
Off: lateral offset, yaw, tilt, the friction band (one-shot D-111 value).

## Code state

Clone `Phase5_RlGames_v1`, branch `p5-rlgames`; the SHA is the commit that
adds this file, given in chat. Training code unchanged since RT-242s
(`git diff e10c18b..HEAD -- source scripts/rl_games scripts/rt_log.ps1`
empty, 2026-09-28). `CODE_MARKER` = `insertion-osc-2026-09-03a`
(`insertion_env.py:76`). The gate is the git line, the provider line and
the run folder (P1).

## The one hypothesis

**On D-204's code under rl_games, every No-DR seed runs 100 epochs with
exit 0, its floor reaches H_min, and afterwards only `start_height_hi`
moves: lateral offset, yaw, tilt and friction keep their cfg values for
the whole run.**

Falsified by: a Traceback or a missing epoch 100; no `reached H_min` line;
any `Episode/dr/` column of `lat_r`, `yaw`, `tilt` or `friction` in
`scalars.csv` (the one-dimension provider has none), or an `[autodr]` line
for one of them; the `friction per reset: OFF` line missing.

Whether `start_height_hi` reaches 0.270 m, and when, is reported (P3), not
part of the hypothesis: no No-DR run on this task has run before.

## Command

Training PC, PowerShell, `env_isaaclab` active, GPU free.

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
git pull
git rev-parse HEAD
```

The five seeds as one chain (RT-242s with ONLY `env.dr_mode=autodr` ->
`env.dr_mode=off` and the RT name):

```powershell
.\scripts\rt_log.ps1 RT-244s1 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-244s2 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 2 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-244s3 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 3 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-244s4 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 4 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402; .\scripts\rt_log.ps1 RT-244s5 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 5 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402
```

Wall time unknown for this arm (RT-242s 2.2-2.4 h, RT-243s 1.3-2.1 h per
seed). After the chain: the RT-243s zip block with `RT-243s` -> `RT-244s`.

## Points -- every seed (N = 1..5)

* **P1 setup** (the gate): as RT-242s P1, with these lines changed:
  `dr_mode: off`; `provider: AutoDR, 2 boundaries, bounds_version 0`
  (`start_height_floor` + `start_height_hi`, `autodr.AutoDR.n_boundaries`);
  the `START FLOOR` line; `friction per reset: OFF` (`insertion_env.py:5287`);
  run folder
  `logs\rl_games\ur5e_insertion\<mm-dd_HH-MM-SS>__RT-244sN_offset50mm_currOFF_start+20mm_floor-30mm_seedN`
  (no dr_mode tag under `off`, `scripts/rl_games/train.py:127-128`). A
  folder with `autodr` or `fixed_dr`, or a provider with 8 boundaries =
  wrong run.
* **P1b, P2**: as RT-242s.
* **P3 the start height** (hypothesis part and report): one `reached H_min`
  line and the first epoch with `Episode/dr/phase` 1; no `Episode/dr/`
  column for `lat_r`, `yaw`, `tilt`, `friction`; REPORTED: the first
  epoch with `Episode/dr/start_height_hi` = 0.270 and `all_at_max` 1, and
  `bounds_version` at 100.
* **P4, P6, P7**: as RT-242s.
* **P5 flat at 100?** as RT-242s P5, plus `Episode/success_rate` at 100;
  beside RT-242sN / RT-243sN of the same seed -- a hint, no bar.
* **P8**: as RT-242s P8.

## Pins (read from the files, not from the log prose)

* `RT-244sN_env.yaml`: `dr_mode: 'off'` (or `off`); against
  `RT-242sN_env.yaml` the diff is ONLY `dr_mode`, `seed` and `log_dir`
  lines (fairness rule, `eval_plan.md` 3.1).
* `RT-244sN_agent.yaml`: against RT-242sN only `full_experiment_name`.
* `RT-244sN_demo_metrics.json`: `dr_mode: off`; `fixture_pos_noise_xy_m`
  and `_z_m` 0.05; `home_tip_above_entrance_mm` 415.0.

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| 100 epochs, exit 0, P3 holds | hypothesis holds for this seed | next seed |
| a column or `[autodr]` line of lat_r / yaw / tilt / friction | the arm is not No-DR | stop the chain, diagnose |
| `start_height_hi` never reaches 0.270 | the No-DR arm trains on a lower ceiling than the other two | report; user decides |
| still rising at 100 | D-201's clause for the No-DR arm | report; user decides (D-205) |
| Traceback or NaN | not a result | stop the chain; hand the Traceback back |

## What this run cannot answer

* K1 / K5: no rl_games evaluation path (D-200, "NOT BUILT").
* Whether the fixture spread alone (0.05 m, live in all three arms) is
  what makes the task hard -- no arm switches it off.
* Test success: the training success rate carries the exploration noise.

## OUTCOME (judged 2026-10-02, files `rt_logs/RT-244s/`; ran 2026-09-29)

* P1 all five: git `7cd3992` (= the commit of this file), cmd = the seed's
  command, import from `Phase5_RlGames_v1`, marker, `shaping_gamma 0.995`,
  `provider: AutoDR, 2 boundaries, bounds_version 0`, START FLOOR line
  (`dr_mode=off`), `friction per reset: OFF`, D-111 friction 0.14 written
  once, step limits 0.02 m / 0.097 rad, clamp 0.1 / 0.2915, yaw band 12.30,
  EMA 0.2, `Asymmetric-learning : True`, `build mlp: 1024` twice,
  `RunningMeanStd: (34,)`; folders
  `..._RT-244sN_offset50mm_currOFF_start+20mm_floor-30mm_seedN` (no dr_mode
  tag). P1b: 0.247..0.290 deg. PASS.
* P2: 100 blocks 1/100..100/100 each, zero `block NOT printed`. PASS.
* **P3 (the hypothesis): HOLDS, all five.** One `reached H_min` line each;
  `dr/phase` 1 from epoch 13 / 14 / 16 / 14 / 16 (same as RT-242s,
  RT-243s). `[autodr]` lines only for `start_height_floor` (7-8) and
  `start_height_hi` (94-105). No boundary column of `lat_r`, `yaw`, `tilt`,
  `friction` (no `_lo` / `_hi` / `_fill` / `_holds` / `_last_rate`; RT-242s1
  has all of them). Wording note: the falsifier text "any `Episode/dr/`
  column of ... friction" also catches three env REPORTER columns that every
  arm logs (`insertion_env.py:3600-3607`, `:3712`): `dr/friction_mean` 0.14
  in every row (= the D-111 value), `dr/friction_write_ms_mean` / `_max` 0.0
  in every row (= never written, comment `insertion_env.py:3598-3599`). They
  show friction stayed fixed; they are no provider boundary. REPORTED:
  first epoch with `start_height_hi` 0.270 and `all_at_max` 1 = 40 / 41 /
  41 / 35 / 43 (AutoDR RT-242s: 74 / 88 / 80 / 73 / 81); `bounds_version`
  15 there and at 100; `all_at_max` never falls back; floor at 100 = 0.02 m
  (H_min).
* P4: zero non-finite cells; `last_lr` 1.98e-5..2.25e-4, never at 1e-6 or
  1e-2; `cval_lr` 1e-4 in every row; `cval_loss` @1/25/50/75/100 falls from
  1.35-2.19 to 0.009-0.017.
* P5 (means 81..90 -> 91..100; success at 100):
  s1 `success_rate` .9945 -> .9935 (.9970), length 46.6 -> 46.0, reward 572.7 -> 575.6;
  s2 .9997 -> .9980 (.9942), 46.5 -> 48.4, 575.5 -> 571.6;
  s3 .9984 -> .9971 (.9970), 47.2 -> 47.1, 574.4 -> 574.2;
  s4 .9969 -> .9972 (.9980), 47.2 -> 47.0, 573.7 -> 576.6;
  s5 .9983 -> .9974 (.9986), 48.8 -> 47.5, 566.2 -> 572.5.
  Success flat near 0.997; reward moves by -3.9..+6.3. Beside the same seed
  (hint, no bar): AutoDR RT-242sN 91..100 .9953 / .9929 / .9896 / .9936 /
  .9970; Fixed-DR RT-243sN .000 / .000 / .9922 / .9963 / .0286. These are
  TRAINING rates on each arm's own spread -- not comparable across arms.
* P6: 100/100, `MAX EPOCHS NUM!`, `[autodr-state] saved into the
  checkpoint` 26-33 times, `[scalars] wrote`, exit 0, no Traceback; `nn`:
  ep 25/50/75/100 + `ur5e_insertion.pth` (+ the `rew__..._` copy of ep_100,
  as RT-243s). PASS.
* P7: all five `VERDICT: COMPLETE`, `rollout_steps_per_iteration: 128`. PASS.
* P8 (reported, no bar): `success_by_start_height_bin` -- bins below H_min
  hold 0 episodes (floor done), bins 20..270 mm 0.990-1.000 in all five;
  `stage1_lateral_y_mm.over_reach` 2 / 36 / 21 / 3 / 102. Force keys NOT
  read (no bar; RT-242s / RT-243s did not read them either).
* Pins: `env.yaml` against RT-242sN differs ONLY in `dr_mode` (`'off'`)
  -- no `seed` / `log_dir` line differs (same seed; `log_dir` not in the
  file), fairness rule holds; `agent.yaml` only in `full_experiment_name`;
  `demo_metrics.json` `dr_mode off`, noise xy / z 0.05, home 415.0 mm.
* Wall time 2.10 / 2.13 / 2.17 / 2.21 / 2.16 h (`wall_time`, last minus
  first row).
* Reading (table above): row 1 for all five -- hypothesis holds. The
  ceiling 0.270 is reached (no "lower ceiling" row). "Still rising": success
  is flat, reward moves a little as in RT-242s; D-205 already holds the
  budget at 100 for every arm -- no new decision.
* The evaluation of these runs (`ep_100`) already exists: T-auswahl RT-246
  (success .103 / .142 / .163 / .078 / .087 at full width, timeouts),
  T-final / T-nominal RT-247..RT-250.
