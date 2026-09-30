# RT-150 expectation — scripted seat at osc kp 100, Delta at the step limit

Written 2026-09-05 on the dev laptop, BEFORE the run. Plan:
`Pläne/Plan-Merge.md` Schritt 2 (item from Plan 2). No code change:
`scripted_insert.py` has carried `--control-mode` / `--osc-kp-pos` since
marker `scripted_insert-2026-09-03a-h1`. Under osc the descent has NEVER
run; only `--free-space` did (RT-143c/d).

## Command

    .\scripts\rt_log.ps1 RT-150 python -u scripts/scripted_insert.py --task Ur5e-Insertion-Direct-v0 --num_envs 16 --headless --control-mode osc --osc-kp-pos 100 --descend-rate 0.02 --max-step 0.02 --max-steps 512 --out rt_logs/RT-150_metrics.json

Bring back `rt_logs/RT-150.txt` AND `rt_logs/RT-150_metrics.json`
(small: 32 per-episode rows).

## Why these two numbers, and why they are not new

`--descend-rate 0.02 --max-step 0.02` = `osc_pos_step_limit_m` (0.02 m,
`insertion_env_cfg.py`, RL_PLACEHOLDERS). Under OSC the pose delta is a
VELOCITY command, `v = Delta * sqrt(kp) / (2 zeta)` per second
(RT-147, CONFIRMED, `HANDOFF-RL.md` § Open 0). The script's defaults
(0.5 mm and 2 mm per control step) were written for the joint_pd chain,
where a delta is a step. Under kp 100, zeta 1.0 (`osc_damping_ratio`):

| Delta | v = Delta*10/2 | 43 mm standoff->seat | stall push Lambda*kp*Delta |
|---|---|---|---|
| 0.0005 (default) | 2.5 mm/s | 17.2 s > 17.07 s episode | 0.44 N |
| 0.002 | 10 mm/s | 4.3 s | 1.76 N |
| **0.02 (limit)** | **100 mm/s** | **0.43 s** | **17.64 N** |

Lambda_max 8.82 kg is the reset-pose value (RT-143, `HANDOFF-RL.md` § Open 0).
The default rate cannot finish inside the episode -- the RT-144b/RT-147 trap
("the 17 s were never the task's"). The limit is what a policy has; that is
the question Plan 2 asks: does kp 100 seat when Delta is at the stop?

Episode: `resolve_control_mode` pins 256 control steps = 17.0667 s at 15 Hz.
With `rl_terms_enabled False` the env terminates on NOTHING but the timeout
(`insertion_env.py:1394-1402`): no success exit, no env force abort. So
512 steps = exactly 2 episodes per env = 32 episodes, and DESCEND keeps
pushing into the floor after the seat at the bounded stall force above.
The script's own ABORT_FORCE phase arms at `force_abort_f_max_n` = 300 N.

## Numbered expectation

1. exit 0; `marker: scripted_insert-2026-09-03a-h1`; the line
   `control_mode osc: pose-delta action, OSC kp pos 100.0`; the episode line
   `256 control steps = 17.0667 s`; `32 episodes over 512 steps`.
2. **No force abort:** `final_phase_counts` has no `ABORT_FORCE` (0 of 32);
   every episode ends in `DESCEND` (ALIGN -> DESCEND opens within the
   first ~2 s: 20 mm approach at up to 50 mm/s plus a 5 mm lateral fixture
   noise at 0.5 gain).
3. **THE LOAD-BEARING POINT -- two outcomes, both named before the run:**
   `peak_depth_m` p50 against `SEATED_SUCCESS_DEPTH` = 0.033 m
   (`POCKET_SEAT_DEPTH` 0.036 - `D106_DEPTH_BAND` 0.003).
   (A) p50 >= 0.033 -> kp 100 seats when Delta is at the limit. The reward
       audit stands on a controller that can do the task.
   (B) p50 < 0.033 -> kp 100 cannot seat even at full Delta. Then the
       controller (kp / F_search, Decision (4)) comes before any reward
       word -- Plan-Merge stop rule.
   No prediction which. Priors, different instruments: RT-144b tilt_insert
   kp 100 at creep rate 11-18 mm; RT-74 scripted_insert joint_pd 19.5 mm
   at ~204 N (friction, no jump).
4. **The RT-147 force law read in contact:** `peak_force_n` max <= 24.2 N
   (RT-148a's measured worst case, implied Lambda 12.1 kg). A max <= 17.64 N
   would additionally confirm the reset-pose Lambda 8.82 kg in contact.
   `over_f_max` / ABORT_FORCE stay 0 (point 2 restated in newtons).
5. **If (B): where and why it stops.** The force-vs-depth table shows the
   force at the deepest bin at the ceiling (>= 15 N): the controller pushes
   at its maximum and the part does not move -- a friction/jam stall, not a
   speed problem. `peak_axis_tilt_rad` p50 read in degrees: > 1 deg says
   geometric jam, < 0.3 deg says friction.
6. Depth spread: `peak_depth_m` p95 - p50 < 3 mm. Sixteen envs with the
   same command and only 5 mm fixture noise should stop at about the same
   depth; a wide spread says the lateral hold at 0.5 gain does not hold
   during a 100 mm/s descent (then RT-150b at Delta 0.005).

## Read, NOT expected

`first_step_residuals` (lateral p50 vs the 5 mm noise, yaw ~0); the
force-vs-depth table in full; seconds per step; `peak_force_n` p50.

## What would make each point wrong

1. -> a crash or wrong flag names; fix the command, rerun.
2. -> force above 300 N under a 100 N/m gain: the force reading or the law,
   not the controller; also possible: the ALIGN gate never opens (lateral
   tolerance 0.2 mm unreachable at velocity semantics) and no episode ever
   descends -- then the `final phases` line shows ALIGN and point 3 is void.
3. -> nothing; both branches are findings.
4. -> the stall force exceeds Lambda*kp*Delta: Lambda in contact is larger
   than at the reset pose, or the OSC is not acceleration-decoupled in
   contact. Reopen RT-147's CONFIRMED line, not the run.
5. -> force at the stall depth well below the ceiling with the part not
   moving: the controller is not pushing -- a target-clamp or phase problem
   in the script, not a physics finding.
6. -> see point text.

## Limits

A scripted controller, not the policy; one Delta; one kp; 16 envs; no
tilt schedule (that was tilt_insert's, SPENT). Says nothing about the
reward. If (A), RT-151 (lateral reward slice) is next; if (B), the concept
stream owns the next step.
