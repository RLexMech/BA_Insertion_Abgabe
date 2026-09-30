# RT-207s1..s5 -- restarted study, Condition 1 (No-DR), written BEFORE the runs (2026-09-15)

Status: **UNVERIFIED.** Nothing in this file has run under Isaac.

## Context and code state

Same decision, same code state and same handover SHA as
`rt_logs/RT-206_expectation.md` (read its "Context" and "Code state"; not
restated here). The only difference: `env.dr_mode=off`. Under off, the start
floor still moves (two keys: `start_height_floor`, `start_height_hi`), as in
RT-202. Run AFTER RT-206a has passed its smoke points; the order of the ten
seeds between RT-206 and RT-207 is the user's.

## The one hypothesis

**Under the 2026-09-15 code state every No-DR seed completes 1500 iterations
with exit 0, and no seed is paid for depth below the fixture.**
Falsified exactly as in `RT-206_expectation.md`.

## Commands

Training PC root `C:\Users\Simon\Desktop\Alexander_Pett\temp\Contactrich_Insertion\Phase5_Robustheit_v1`.
One job at a time.

```powershell
.\scripts\rt_log.ps1 RT-207s1 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 1 --max_iterations 1500 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-207s2 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 2 --max_iterations 1500 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-207s3 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 3 --max_iterations 1500 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-207s4 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 4 --max_iterations 1500 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

```powershell
.\scripts\rt_log.ps1 RT-207s5 python -u scripts/rsl_rl/train.py --task Ur5e-Insertion-Direct-v0 --num_envs 256 --headless --seed 5 --max_iterations 1500 env.dr_mode=off env.start_floor_m=-0.030 env.autodr_stall_buffers=3
```

## Points per seed

P1-P9 of `RT-206_expectation.md`, with these changes: P1 `dr_mode off` and
`provider: AutoDR, 2 boundaries`; P3 reads only `start_height_floor` and
`start_height_hi` (RT-202: height 0.02 -> 0.12 between 190 and 292); P5 has
no lateral/yaw/tilt bins to judge (they stay at the centre); the sidecar is
`autodr_1499.json` with two keys. Pre-registered readings: the same table.
