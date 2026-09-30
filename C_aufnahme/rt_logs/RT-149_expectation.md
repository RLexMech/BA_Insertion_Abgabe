# RT-149 expectation — replay trace of the RT-148b checkpoint

Written 2026-09-05 on the dev laptop, BEFORE the run. Plan:
`Pläne/Plan-Merge.md` Schritt 2. Code: `scripts/rsl_rl/play.py --trace-obs`
(UNVERIFIED, laptop only; syntax checked, never executed under Isaac).

## Command

    .\scripts\rt_log.ps1 RT-149 python -u scripts/rsl_rl/play.py --task Ur5e-Insertion-Direct-v0 --num_envs 8 --headless --seed 42 --load_run 09-04_14-02-30_offset5mm_currOFF_start+30mm_seed42 --checkpoint model_3999.pt --trace-obs rt_logs/RT-149_trace.json --trace-steps 512

Bring back `rt_logs/RT-149.txt` AND `rt_logs/RT-149_trace.json`.
(`--load_run` / `--checkpoint` are rsl_rl's own flags, `scripts/rsl_rl/cli_args.py:31-32`.)

## What it reads

Per step, per env, the RAW observation slices the policy already sees
(`insertion_math.OBS_SLICES`): `tip_rel` 12:15 (pocket frame, entrance
plane z = 0, y = lateral), `yaw_cos_sin` 19:21, `force` 25:28 (EMA, tared),
plus the sampled action and `done`. No env change, no reward change.

## Numbered expectation

1. exit 0, trace file written, 512 steps x 8 envs, `done` fires exactly
   at step 256 and 512 for every env (time-out at 256 steps, no force
   abort, no success).
2. **Frozen policy:** over steps 20..255 the action changes by less than
   0.05 per step in every channel (RT-148b sigma 0.01, action_rate
   -3.69 per episode).
3. **Lateral y stalls at the clamp:** `tip_rel.y` reaches |y| in
   [48, 54] mm and then stays within +-1 mm (RT-148b
   `stage1_lateral_y_mm` p50 51.289, p95 51.744, max 53.570).
4. **THE +15 mm PREDICTION (VERDICTS NACHTRAG (4)):** once |y| > 8.7 mm
   the part rests ON the block top: `tip_rel.z` settles in [+13, +18] mm
   and never goes below 0 in any env. If `tip_rel.z` sits far above +18 mm
   the part hovers in free air and never touched the block; if it reads
   below 0 anywhere the "gate never opened" reading of RT-148b was right
   after all and NACHTRAG (4) is withdrawn.
5. **Yaw drifts at a constant rate:** `phi = atan2(sin, cos)` changes
   monotonically over the episode with a per-step slope that varies by
   less than 20 % between steps 50 and 250 (RT-148p viewer: "dreht weiter
   um z"). Read the slope in deg/s as a NUMBER.
6. **Force per step (the only valid K3 replacement):** if point 4 holds,
   `||force||` while resting is > 0.5 N and roughly constant (the block
   carries part of the tool weight under a still-pushing OSC target);
   while moving in free air it is < 0.5 N. A per-step force that is
   0.0 N throughout means free air, and NACHTRAG (3)'s "not proven"
   resolves to "no contact".
7. `tip_rel.x` stays inside +-45.29 mm (POCKET_WALL_X) in every env. If it
   does not, `in_pocket_cross_section` failed on x, not on y, and §6 of
   the report changes.

## Read, NOT expected

The per-episode `demo_metrics.json` the replay writes; seconds per step.

## What would make each point wrong

1. -> crash or wrong flag names; fix the command, rerun.
2. -> the checkpoint is not RT-148b's or the policy is not frozen; then
   the -3.69 action_rate reading is wrong.
3. -> the 51 mm is not the clamp (VERDICTS "UNBEWIESENE VERMUTUNG"); the
   stall has another cause, to be named.
4. -> see point text: hover vs. below-plane are DIFFERENT findings.
5. -> yaw is not free-running; then D-106 "yaw free" is not the mechanism
   of the observed rotation.
6. -> contact is not the mechanism of the +15 mm rest.
7. -> the lateral-gate finding moves from y to x.

## Limits

Eight envs, one seed, one checkpoint. Says nothing about the reward
gradient; that is RT-151.
