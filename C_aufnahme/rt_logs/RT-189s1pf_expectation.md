# RT-189s1pf — expectation, written BEFORE the run (2026-09-13)

Written on the dev laptop. `/rt-check` judges the log and the trace against
THIS file. **UNVERIFIED** until the training PC has run it. A replay, not a
training run. It answers ONE question and changes NOTHING the policy sees.

REWRITTEN 2026-09-13 (same day, before any run). The first version switched
the force observation noise off. That changes the policy's input, so it would
no longer measure why the policy parks under its own conditions (user review).
This version keeps policy, noise and command identical to RT-189s1p and only
records extra, un-noised channels.

## Why

RT-189s1p (`rt_logs/VERDICTS.md`, 2026-09-13 line) called the parked part
"in contact". The inline correction on that line shows the force channel was
noise only (|F| 5.47..5.88 N per env = E|N(0,3.5² I₃)| 5.585 N). Whether the
parked part touches the fixture is OPEN, and so is what the controller tries
to do there.

## The one question

**"In the RT-189s1 final policy's parked phase: is there a force on the part,
does the part move, and how far is the clamped OSC target from the tool
body?"**

NOT answered by this run: whether the part is BLOCKED. Contact alone does not
prove a blockage — the part may rest on the fixture and still slide. That
needs a targeted correction motion out of the parked pose, a separate run
(outcome K below).

## What changed against RT-189s1p

* **Code only, measurement only:**
  * `insertion_env.py`: buffer `_force_tared_raw` = the tared force BEFORE the
    EMA and the observation noise; cleared on reset like `_force_smooth`.
    Nothing in observation, reward or termination reads it.
  * `scripts/rsl_rl/play.py`: under `--trace-obs` a `truth` block per step,
    read at the same moment as the observation row: `force_tared_raw_n`,
    `force_ema_n` (`_force_smooth`), `tip_true_m` (no grasp offset, no pocket
    noise), `body_pose_w`, `osc_delta`, `osc_target_pose_w`,
    `interpen_max_m`. Frames are written into the file (`truth_frames`).
  * `scripts/trace_outcome_split.py`: prints the truth block (marker
    `trace_outcome_split-2026-09-13a`).
* **Same as RT-189s1p:** policy (`--load_run 09-12_21-58-11`, newest
  checkpoint), 8 envs, 512 steps, `env.dr_mode=autodr`, all noise at the cfg
  values (force σ 3.5 N, pocket σ 2.5 mm).
* **Not available:** contact forces or points between part and fixture. The
  env has no `ContactSensor`. `interpen_max_m` > 0 means touching; 0 does NOT
  prove free.

## Named limits of the channels

* The tared joint wrench is not a pure contact force: while the part moves it
  carries dynamic parts. Read it in the nearly still parked window only.
* `osc_target_pose_w` comes from the LAST physics substep of the previous
  policy step; `body_pose_w` is read after that substep. The error
  target − body is therefore one substep old, not simultaneous.
* `osc_delta` is a pose OFFSET of the target per policy step, re-applied
  against the current pose at every physics step. It is not a promised motion
  per step.
* Row 0 is the reset step (reset-step trap, project CLAUDE.md). The window
  starts at step 20.

## Command (training PC, PowerShell)

Pull first (code changed):

```
git pull
git rev-parse HEAD
```

Expected SHA: see the chat handover (the commit that carries this file).

```
.\scripts\rt_log.ps1 RT-189s1pf python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 8 --headless --load_run "09-12_21-58-11" --trace-obs rt_logs/RT-189s1pf_trace.json --trace-steps 512 env.dr_mode=autodr
```

## Points

* **P1 setup.** `[rt_log] exit code: 0`, no Traceback, `git:` line = the
  handed-over SHA. Trace carries `truth` with all 7 keys and
  `force_obs_noise_std_n` = 3.5. Otherwise: RUN INVALID.
* **P2 same behaviour as RT-189s1p.** First episode per env: 8/8 timeout,
  0/8 below the entrance. The policy input is unchanged, so this is expected
  up to simulator nondeterminism; if it fails, report it and judge no further.
* **P3 the numbers**, per env, steps 20..end, from
  `python scripts/trace_outcome_split.py rt_logs/RT-189s1pf_trace.json`:
  raw and EMA mean force vector and norm, true tip end and travel per axis,
  mean error target − body and norm, interpenetration max and share > 0.

## Pre-registered readings (no numeric bar is fixed in advance)

No threshold for "near zero" exists in this repo, and none is invented here.
The reading compares each quantity against its own scale in the same trace:
force against the policy's commanded push estimate (a calculation, ≈ 0.47 N
from Λ_max 9.4002 kg · kp 100 · δ 0.5 mm, Λ at the RESET pose — own
construction, named), error against `osc_delta`, travel against the error.
If a case is not clear on these scales, it is reported as UNCLEAR.

| Outcome | Reading | Next step |
|---|---|---|
| F — free | no force beyond the push scale, interpen 0, error ≈ `osc_delta` and no standing error | First find out WHY the policy learned to stay here. No reward, start or exploration change yet. |
| K — contact, standing error | force on the push scale or above AND/OR interpen > 0, a standing error target − body, travel ≈ 0 | Next run: a targeted correction motion out of the parked pose — can the pose be left mechanically? Only then: does the policy fail to use it? |
| UNCLEAR | neither pattern in ≥ 6 of 8 envs ("6 of 8" own construction, not a test) | Report per env; no next step picked. |

If K is followed by a working correction that the policy does not use, THAT
is the basis to change exploration, start distribution or reward — one at a
time.
