# RT-260a / RT-260s1..s5 -- expectation, written BEFORE the run (2026-10-03): the MLP ablation of D-214 -- RT-242s without the LSTM

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

Template: `rt_logs/RT-242_expectation.md`, section RT-242s1..s5. Unchanged
sections point there; only what differs is written out.

## Context

D-195 gave the policy an LSTM because the task holds hidden quantities that
stay constant within an episode (pocket position bias, grasp offset,
pocket-yaw belief error, friction), and named the ablation it owes to the
results chapter: the same AutoDR arm, MLP versus LSTM, both with a_(t-1)
and D-194, same seeds. D-206 point 4 dropped it; D-214 (2026-10-03, user)
brings it back. The LSTM side exists: RT-242s1..s5 (AutoDR, `e10c18b`;
OUTCOME in `rt_logs/RT-242_expectation.md`: first epoch with
`Episode/dr/all_at_max` 1 at 74 / 88 / 80 / 73 / 81, success over epochs
91..100 .9953 / .9929 / .9896 / .9936 / .9970). This chain is the MLP side:
the RT-242s command plus two Hydra deletions that remove the `rnn` block of
the actor and of the separate critic. The previous action stays in the
observation (it is the EMA state, D-193 (1), D-214 Rationale), so exactly
one thing changes: the memory.

## Code state

Clone `Phase5_RlGames_v1`, branch `p5-rlgames`. The training PC stands on
`c1a635e` (the evaluation commit). The training path is identical to
RT-242s' `e10c18b`: `git diff e10c18b..c1a635e -- source` touches only the
evaluation table mode (`allow_outside`, `dr_table_allow_outside`, D-206
(3)), and `git diff c1a635e..origin/p5-rlgames -- source scripts/rl_games
scripts/rt_log.ps1` is empty (laptop, 2026-10-03). A pull is still asked
for, so that the `[rt_log] git:` line names the commit that adds this file
(SHA given in chat). `CODE_MARKER` = `insertion-osc-2026-09-03a`
(`insertion_env.py:76`), unchanged.

How the MLP arises, checked offline only (D-214 Rationale): rl_games
`6b3534f2` builds a recurrent net only when the key exists
(`rl_games/algos_torch/network_builder.py:517`, `self.has_rnn = 'rnn' in
params`); without it the MLP input is the observation size and the builder
prints `build mlp: <input size>` (`:98`). Isaac Lab replaces the rl_games
agent dict by Hydra's result (`isaaclab_tasks/utils/hydra.py:95-96`); a
hydra-core 1.3.2 compose test deleted both blocks (control kept them).
NOT checked: the Hydra version on the training PC, and the run under Isaac
-- RT-260a is that check.

## The one hypothesis

**Without the LSTM (the RT-242s command otherwise), the MLP reaches the
AutoDR ceiling later than the LSTM of the same seed: the first epoch with
`Episode/dr/all_at_max` 1 lies after RT-242sN's (74 / 88 / 80 / 73 / 81),
or never comes within 100 epochs, in at least 4 of the 5 seeds.**

This is D-195's claim (memory is needed for the hidden per-episode
constants) put into one training number. Falsified by: 2 or more seeds with
the ceiling at or before the same seed's RT-242s epoch. RT-260a has its own,
technical hypothesis (below); it is no part of this one.

RT-260a: **both networks build as plain MLPs on the 34 inputs (`build mlp:
34` twice, no 1024 line), and 4 epochs run with finite losses and exit 0.**

## Commands

Training PC, PowerShell, `env_isaaclab` active, GPU free.

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
git pull
git rev-parse HEAD
```

Expected: the SHA given in chat. Then the smoke run (about 6 min; RT-242s
read 2.23-2.36 h per 100 epochs):

```powershell
.\scripts\rt_log.ps1 RT-260a python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402 "~agent.params.network.rnn" "~agent.params.config.central_value_config.network.rnn" agent.params.config.max_epochs=4
Select-String -Path rt_logs\RT-260a.txt -Pattern "build mlp|Asymmetric|RunningMeanStd|imported from|code marker|Learning iteration 4/4|Traceback|exit code"
```

Paste the `Select-String` output into the chat. ONLY after the RT-260a
gate passed, the five seeds as one chain (the RT-242s chain with the two
deletions and the RT name):

```powershell
.\scripts\rt_log.ps1 RT-260s1 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402 "~agent.params.network.rnn" "~agent.params.config.central_value_config.network.rnn"; .\scripts\rt_log.ps1 RT-260s2 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 2 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402 "~agent.params.network.rnn" "~agent.params.config.central_value_config.network.rnn"; .\scripts\rt_log.ps1 RT-260s3 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 3 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402 "~agent.params.network.rnn" "~agent.params.config.central_value_config.network.rnn"; .\scripts\rt_log.ps1 RT-260s4 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 4 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402 "~agent.params.network.rnn" "~agent.params.config.central_value_config.network.rnn"; .\scripts\rt_log.ps1 RT-260s5 python -u scripts/rl_games/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 5 env.dr_mode=autodr env.start_floor_m=-0.030 env.autodr_stall_buffers=3 env.obs_wrench_mode=force_prevact env.pocket_obs_yaw_rad=0.0223402 "~agent.params.network.rnn" "~agent.params.config.central_value_config.network.rnn"
```

Wall time for the MLP: unknown (a smaller network; the environment step is
the same). After the chain, ONE zip -- the RT-242s block with `RT-242s` ->
`RT-260s` (`rt_logs/RT-242_expectation.md`, section Command), written out:

```powershell
cd C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_RlGames_v1
foreach ($n in 1..5) {
  $t = "RT-260s$n"
  $dirs = @(Get-ChildItem -Directory "logs\rl_games\ur5e_insertion\*__${t}_*")
  if ($dirs.Count -ne 1) { Write-Host "$t : expected ONE run folder, found $($dirs.Count)"; continue }
  $run = $dirs[0].FullName
  python scripts/check_episode_log.py --run-dir $run | Out-File -Encoding utf8 "rt_logs\${t}_eplog.txt"
  foreach ($f in "scalars.csv", "demo_metrics.json", "episodes.csv", "episodes_meta.json") { if (Test-Path "$run\$f") { Copy-Item -LiteralPath "$run\$f" "rt_logs\${t}_$f" } }
  Copy-Item -LiteralPath "$run\params\agent.yaml" "rt_logs\${t}_agent.yaml"
  Copy-Item -LiteralPath "$run\params\env.yaml" "rt_logs\${t}_env.yaml"
  Get-ChildItem -LiteralPath "$run\nn" | Select-Object Name, Length | Out-File -Encoding utf8 "rt_logs\${t}_nn.txt"
}
$files = (Get-ChildItem rt_logs\RT-260*).FullName
New-Item -ItemType Directory -Force C:\Users\Simon\Desktop\Alexander_Pett\Zips | Out-Null
Compress-Archive -LiteralPath $files -DestinationPath C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-260.zip -Force
Get-Item C:\Users\Simon\Desktop\Alexander_Pett\Zips\RT-260.zip
```

## Points -- RT-260a (smoke, the gate for the chain)

* **A1 setup**: as RT-242s P1 (git = the chat SHA; import from
  `...\Phase5_RlGames_v1\source\insertion\insertion`; `code marker:
  insertion-osc-2026-09-03a`; `env.shaping_gamma = 0.995`; `provider:
  AutoDR, 8 boundaries, bounds_version 0`; START FLOOR line; step limits,
  clamp, yaw band, EMA 0.2; run folder
  `..._RT-260a_autodr_offset50mm_currOFF_start+20mm_floor-30mm_seed1`).
* **A2 the network** (the RT-260a hypothesis): `build mlp: 34` exactly twice
  (actor and critic), no `build mlp: 1024` (RT-244s OUTCOME P1 read `build
  mlp: 1024` twice for the LSTM); `Asymmetric-learning : True`;
  `RunningMeanStd: (34,)`. A `1024` line or a single `build mlp` line =
  the deletion did not act; the chain does not start.
* **A3 run**: `Learning iteration 1/4` .. `4/4`, zero `block NOT printed`,
  no `Traceback`, losses finite, `[rt_log] exit code: 0`.

## Points -- every seed RT-260sN (N = 1..5)

* **P1 setup**: as RT-242s P1, with the network lines of A2 instead of
  "the 1024 LSTM"; run folder
  `..._RT-260sN_autodr_offset50mm_currOFF_start+20mm_floor-30mm_seedN`.
* **P1b, P2, P4, P6, P7**: as RT-242s.
* **P3 the ceiling** (the hypothesis): as RT-242s P3, and beside it the
  same seed's RT-242s epoch (74 / 88 / 80 / 73 / 81). Per seed: later /
  same / earlier / never.
* **P5**, reported, no bar: as RT-242s P5 (means of `rewards/iter`,
  `episode_lengths/iter`, `Episode/success_rate` over 81..90 and 91..100),
  beside RT-242sN of the same seed. These are TRAINING rates on the same
  arm and spread, so the two networks may be set side by side here (unlike
  across arms).
* **P8**: as RT-242s P8.

## Pins (read from the files, not from the log prose)

* `RT-260sN_agent.yaml` against `RT-242sN_agent.yaml`: the `network.rnn`
  block and the `config.central_value_config.network.rnn` block are
  MISSING, `full_experiment_name` differs; every other line equal
  (`max_epochs: 100`, `minibatch_size: 1024`, `gamma: 0.995`,
  `learning_rate: 0.0001`, `mlp.units: [512, 128, 64]`, `seq_length: 128`
  -- unused without an RNN). RT-260a: `max_epochs: 4`.
* `RT-260sN_env.yaml` against `RT-242sN_env.yaml`: equal except
  `dr_table_allow_outside: false` (a field added after `e10c18b`,
  `insertion_env_cfg.py`, D-206 (3)); any other differing line = a
  different environment.
* `RT-260sN_demo_metrics.json`: `dr_mode: autodr`, `fixture_pos_noise_xy_m`
  and `_z_m` 0.05, `home_tip_above_entrance_mm` 415.0 (as RT-242s).

## Pre-registered readings

| Outcome | Reading | Next |
|---|---|---|
| A2 fails (`1024` line, or one `build mlp`) | the deletion did not reach rl_games | stop; hand the log back; no chain |
| >= 4 of 5 seeds later or never | the LSTM speeds up training on this task (D-195 supported for training) | the thesis figure MLP vs LSTM |
| <= 1 seed later | memory is not needed for training progress here; D-195's rationale is not supported in training | report; the thesis says so |
| 2 or 3 seeds later | no difference shown (n = 5) | report as such |
| a seed never reaches the ceiling | the MLP fails on this spread for that seed | report with its last bounds |
| Traceback or NaN | not a result | stop the chain; hand the Traceback back |

## What this run cannot answer

* Test success of the MLP: no evaluation run is planned (D-214). The
  LSTM's test rates (RT-247..RT-250) have no MLP partner.
* Whether the LSTM helps the Fixed-DR or No-DR arm: only AutoDR is run.
* Which hidden quantity the memory estimates: the ablation removes the
  memory as a whole.
* Seed spread beyond n = 5.

## OUTCOME RT-260a / RT-260s1..s5 (judged 2026-10-03, files `rt_logs/RT-260s/`; ran 2026-10-03 01:39 .. 11:36)

Read by `judge_rt260*.py` (scratch, one pass over every file); LSTM numbers
from `rt_logs/RT-242s/` the same way. They match the RT-242s OUTCOME above
(ceiling epochs, cval_loss 1.35-2.19 -> 0.019-0.024).

* **RT-260a: PASS.** A1: git `ddc9d0b` (= the commit of this file), import
  from `Phase5_RlGames_v1`, marker, `shaping_gamma 0.995`, `provider:
  AutoDR, 8 boundaries, bounds_version 0`, START FLOOR, step limits, clamp
  0.1 / 0.2915, yaw band 12.30, EMA 0.2, folder
  `10-03_01-40-09__RT-260a_autodr_offset50mm_currOFF_start+20mm_floor-30mm_seed1`.
  **A2: `build mlp: 34` exactly twice, no `1024` line**, `Asymmetric-learning
  : True`, `RunningMeanStd: (34,)`. A3: `Learning iteration` 1..4/4, zero
  `block NOT printed`, `MAX EPOCHS NUM!`, exit 0, no Traceback. (The one
  `inf` hit in every log is the config line `Observations clipping: inf`.)
* P1 all five: as A1/A2 (git `ddc9d0b`, `build mlp: 34` twice), folders
  `..._RT-260sN_autodr_offset50mm_currOFF_start+20mm_floor-30mm_seedN`.
  P1b: first `reset yaw vs seat` 0.246..0.292 deg. PASS.
* P2: 100 blocks 1/100..100/100 each, zero `block NOT printed`. PASS.
* **P3 (the hypothesis): HOLDS, 5 of 5.** First epoch with
  `Episode/dr/all_at_max` 1, MLP vs LSTM (RT-242sN): s1 **86** vs 74, s2
  **95** vs 88, s3 **85** vs 80, s4 **97** vs 73, s5 **84** vs 81 -- later
  in every seed (+12 / +7 / +5 / +24 / +3 epochs). `bounds_version` 75 there
  and at 100; never falls back. The epoch before holds a fraction (s1 0.555,
  s2 0.461, s3 0.938, s4 0.273, s5 0.125); read with the switch inside that
  epoch (85 / 94 / 84 / 96 / 83) every seed is still later than the LSTM
  under the same reading. `dr/phase` 1 from epoch 18 / 16 / 16 / 16 / 18
  (LSTM 13 / 14 / 16 / 14 / 16).
* P4: zero non-finite cells. `info/last_lr` 1.00e-4 .. 1.71e-3 in all five
  (never below the start value; LSTM 1.3e-5 .. 2.25e-4), never at 1e-6 or
  1e-2. `info/cval_lr` 1e-4 in every row. `losses/cval_loss` @1/25/50/75/100
  (loss rows sit at frame step e x 32768): s1 1.942 / 0.189 / 0.106 / 0.063
  / 0.040, s2 2.355 / 0.220 / 0.104 / 0.080 / 0.057, s3 1.631 / 0.244 /
  0.110 / 0.055 / 0.045, s4 2.164 / 0.196 / 0.094 / 0.081 / 0.069, s5 1.372
  / 0.236 / 0.136 / 0.067 / 0.049 (LSTM at 100: 0.019-0.024).
* P5 (means 81..90 -> 91..100, MLP | LSTM same seed): `Episode/success_rate`
  s1 .9914 -> .9961 | .9963 -> .9953; s2 .9871 -> .9890 | .9916 -> .9929;
  s3 .9930 -> .9963 | .9825 -> .9896; s4 .9839 -> .9788 | .9898 -> .9936;
  s5 .9893 -> .9943 | .9943 -> .9970. `rewards/iter` 91..100: 554.2 / 525.6
  / 554.1 / 518.2 / 554.1 | 565.1 / 552.1 / 565.3 / 565.0 / 561.7 (MLP lower
  in all five). `episode_lengths/iter` 91..100: 49.8 / 57.6 / 48.6 / 60.8 /
  49.4 | 45.2 / 49.6 / 46.4 / 45.8 / 46.9 (MLP longer in all five).
  Success: MLP higher in s1, s3, lower in s2, s4, s5 -- no direction.
* P6: 100/100, `MAX EPOCHS NUM!`, `nn`: ep 25/50/75/100 (+ the `rew__..._`
  twin of ep_100) + `ur5e_insertion.pth`, `[autodr-state] saved into the
  checkpoint` 21 / 16 / 19 / 15 / 20 times, `[scalars] wrote`, exit 0, no
  Traceback. PASS.
* P7: all five `VERDICT: COMPLETE`, `rollout_steps_per_iteration: 128`. PASS.
* P8 (reported, no bar): `success_by_start_height_bin` -- the five bins
  below H_min hold 0 episodes (floor done); bins 20..270 mm 0.976-1.000;
  lowest s4 (0.976-0.987). `stage1_lateral_y_mm.over_reach` 243 / 336 / 294
  / 331 / 227 (RT-242s not read). Force keys not read.
* Pins: `agent.yaml` against RT-242sN differs in exactly the 15 lines of the
  two `rnn` blocks (missing) and `full_experiment_name`; `env.yaml` only in
  `dr_table_allow_outside: false` (as predicted); `demo_metrics.json`
  `dr_mode autodr`, noise xy / z 0.05, home 415.0 mm.
* Wall time 1.92 / 1.96 / 1.95 / 1.92 / 1.95 h per seed (`wall_time`, last
  minus first row; LSTM RT-242s 2.23-2.37 h, same reading).
* Reading (table above): ">= 4 of 5 seeds later" -> **the LSTM speeds up
  training on this task; D-195 supported for training.** In numbers: the MLP
  reaches the full AutoDR range 3-24 epochs later, ends with a lower return
  and longer episodes in all five seeds, and its training success at full
  width (91..100) is 0.979-0.996 against 0.990-0.997 -- the same band, no
  ordering. What the memory buys here is speed (curriculum and insertion),
  not a higher end success in training. Test success: not measured (D-214).
