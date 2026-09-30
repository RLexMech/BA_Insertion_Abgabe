# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""THE RIGHTING PROBE: at gain k, does the controller pull a JAMMED, TILTED
part back upright -- with no policy in the loop at all?

WHY THIS FILE EXISTS
--------------------
RT-158 measured the tilt of every first episode at the +40 mm start
(``rt_logs/VERDICTS.md``, 2026-09-06). Against the tilt clamp of 8.52 deg
(``osc_tilt_clamp_rad``, CAD-derived, Decision (3)) the split is total:

    success      median  2.23 deg    0 of 135 over the clamp
    mode A rim   median 10.22 deg   71 of  71 over the clamp
    mode B mouth median 15.32 deg   46 of  50 over the clamp

Over the clamp the commanded target is MORE UPRIGHT than the part, every
physics step, for the rest of the episode -- and the part stays tilted. The
restoring moment is ``Lambda_rot * osc_kp_rot * theta_err``; at
``osc_kp_rot = 30`` that is 0.203 Nm (mode A) and 0.812 Nm (mode B), using
the measured ``Lambda_rot`` max of 0.22806 kg m^2 the startup report prints.
Nobody has ever checked those two numbers against what they must overcome,
and ``osc_kp_rot`` is the one number on that path with no source of ours
(``insertion_env_cfg.py:218`` says so itself).

WHY NOT ANOTHER RT-159
----------------------
RT-159 swapped the gain to 100 UNDER a policy trained at 30 and the success
rate fell from 0.5273 to 0.0039. That result is real but it cannot answer
this question: the policy saw a machine it had never learned on, so the run
measures the swap, not the gain. This probe removes the policy entirely.

WHAT IT DOES
------------
Per (gain, tilt) rung:

1. RESET, then rebuild the task-space controller with THIS rung's
   ``osc_kp_rot`` (``_build_task_space_controller``, the env's own method --
   the gain is not patched into the controller behind its back).
2. TELEPORT the arm to the jam: a commanded tip offset in the pocket frame
   AND a commanded tilt, solved kinematically the way ``seat_probe.py``
   does it, with no ``env.step`` inside the solve (RT-80).
3. HOLD with a ZERO action for ``--hold-steps`` control steps, and read the
   tilt every step.

A ZERO ACTION IS THE WHOLE EXPERIMENT, and that is the point. Under OSC the
target is re-anchored every physics step as ``current pose + delta`` and then
clamped to the cone (``insertion_env.py:1030-1035``). With delta = 0 the
commanded target IS the cone-clamped current pose -- exactly the standing
"straighten up" command a jammed episode gets. So the probe reproduces the
failing command without inventing one.

WHAT IT REPORTS, AND WHAT IT REFUSES TO DECIDE
----------------------------------------------
Per rung: the tilt at teleport, after the hold, the minimum reached, the
contact force, the lateral offset and the depth. It sets NO threshold. What
counts as "rights itself" is a decision; this run reports the trajectory the
decision needs. In particular it does NOT recommend a gain.

READING IT
----------
    tilt falls to the clamp at 30 as well  -> the stiffness is NOT the jam.
                                              Look at reward and coverage.
    tilt only falls above some k           -> the stiffness IS the jam, and
                                              the next step is a TRAINING run
                                              at that k, never a replay.
    tilt falls at no gain                  -> geometry or friction holds it;
                                              a gain change cannot help.

THE TOOL AXIS CONVENTION IS READ, NOT ASSUMED. ``clamp_tilt_to_cone``
(``insertion_math.py:1372``) takes the body's +z as the tool axis and
``-axis_z`` as the cosine of the tilt from vertical, so the tool +z points
DOWN. This script uses ``axes_from_quat`` the same way and prints the reset
axis so a reader can see it.

USAGE (training PC)
-------------------
    .\scripts\rt_log.ps1 RT-168 python -u scripts/tilt_recovery_probe.py ^
        --task Ur5e-Insertion-Direct-v0 --num_envs 16 --headless ^
        --kp-rot 30 --kp-rot 50 --kp-rot 100 --kp-rot 200 ^
        --tilt-deg 10.2 --tilt-deg 15.3

    python scripts/tilt_recovery_probe.py --self-test    (laptop, no Isaac)

THE RT NUMBER MOVED, THE CRASH HISTORY DID NOT (2026-09-06). This probe was
built under RT-160 and crashed twice under that label. RT-160 turned out to be
double-issued -- the resumed training run of 16:15:10 had already taken it and
is the one with a verdict line -- so the probe's FIRST REAL RUN moved to
RT-167, and then AGAIN to RT-168: RT-167 turned out to be already spent too,
on a finished check_seated_success reward-curve run (its own log names
itself RT-167.txt). Every "RT-160 died on ..." below is left alone
on purpose: that is what those log headers actually say, and the self-test
checks named after them are anchors to real crashes.

The defaults are the two poses RT-157/RT-158 measured, in millimetres of
``tip_rel``: mode A ``x +7.79 / y -15.90 / z +4.36`` at 10.2 deg, mode B
``x +0.25 / y -4.70 / z -1.85`` at 15.3 deg. Pass ``--mode b`` for the
second; the numbers live in ``JAM_POSES`` below with their source.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys

SCRIPT_MARKER = "tilt_recovery_probe-2026-09-06a"

# The two parked failure poses, in millimetres of `tip_rel` (pocket frame) and
# degrees of tool-axis tilt. SOURCE: rt_logs/VERDICTS.md, RT-157h40t/h60
# BEFUND 2 (the x/y split) and the 2026-09-06 ACHSEN-MESSUNG (the tilt, from
# the RT-158 trace). NOT rounded here -- rounding a measured pose hides which
# digits were measured.
JAM_POSES = {
    "a": {"x_mm": 7.79, "y_mm": -15.90, "z_mm": 4.36, "tilt_deg": 10.22},
    "b": {"x_mm": 0.25, "y_mm": -4.70, "z_mm": -1.85, "tilt_deg": 15.32},
}


# The env modules this script imports, as (dotted path, file under
# ``source/insertion/insertion/``). RT-160 died on a dotted path that does not
# exist, and nothing offline could have caught it -- now the self-test can.
INSERTION_MODULES = (
    ("insertion.tasks.direct.insertion.insertion_math",
     "tasks/direct/insertion/insertion_math.py"),
    ("insertion.tasks.direct.insertion.scripted_policy",
     "tasks/direct/insertion/scripted_policy.py"),
)


# --------------------------------------------------------------------------
# THE PURE HALF -- offline testable, no Isaac (CLAUDE.md, Code Style)
# --------------------------------------------------------------------------

def tilt_delta_pocket(torch, tool_axis_p, tilt_rad: float, about: str):
    """Axis-angle delta, in the POCKET frame, that swings the tool axis from
    where it is to ``tilt_rad`` off the pocket's own axis.

    ``tool_axis_p`` is (N, 3), the tool's +z expressed in the pocket frame; at
    an upright reset it is (0, 0, -1) -- the tool points DOWN the pocket.

    The delta is the SMALLEST rotation between the two directions
    (``n = a x d``, angle = ``atan2(|n|, a.d)``), so it adds no yaw of its
    own -- the same construction ``clamp_tilt_to_cone`` uses, in the other
    direction. Two directions that already agree give a zero delta rather
    than a NaN axis.
    """
    down = torch.zeros_like(tool_axis_p)
    down[:, 2] = -1.0
    s, c = math.sin(tilt_rad), math.cos(tilt_rad)
    des = torch.zeros_like(tool_axis_p)
    if about == "x":
        # Rodrigues about +x: (0,0,-1) -> (0, +sin, -cos)
        des[:, 1] = s
        des[:, 2] = -c
    elif about == "y":
        # about +y: (0,0,-1) -> (-sin, 0, -cos)
        des[:, 0] = -s
        des[:, 2] = -c
    else:
        raise ValueError(f"about must be 'x' or 'y', got {about!r}")
    a = tool_axis_p / torch.linalg.norm(tool_axis_p, dim=-1).reshape(-1, 1)
    n = torch.cross(a, des, dim=-1)
    n_norm = torch.linalg.norm(n, dim=-1)
    dot = torch.sum(a * des, dim=-1)
    ang = torch.atan2(n_norm, torch.clamp(dot, min=-1.0, max=1.0))
    safe = torch.clamp(n_norm, min=1.0e-9).reshape(-1, 1)
    return torch.where((n_norm > 1.0e-9).reshape(-1, 1), n / safe * ang.reshape(-1, 1),
                       torch.zeros_like(n))


def tilt_from_axis_deg(torch, tool_axis_p):
    """Tilt [deg] of the tool axis off the pocket axis, (N,) from (N, 3)."""
    a = tool_axis_p / torch.linalg.norm(tool_axis_p, dim=-1).reshape(-1, 1)
    return torch.acos(torch.clamp(-a[:, 2], min=-1.0, max=1.0)) * (180.0 / math.pi)


def _self_test() -> int:
    """Runs on the laptop: real torch when it imports, the numpy stand-in
    otherwise (``scripts/tools/torch_shim.py``)."""
    shim_path = pathlib.Path(__file__).resolve().parent / "tools" / "torch_shim.py"
    spec = importlib.util.spec_from_file_location("torch_shim", shim_path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    torch, _ = shim.load(announce=False)

    def axis(vals):
        # NO explicit dtype: torch_shim builds float64 and load() puts REAL
        # torch on float64 too, so both paths test the same algebra
        # (torch_shim.py:28-32). Pinning float32 here would break that.
        return torch.tensor([list(v) for v in vals])

    def rodrigues(v, aa):
        """Rotate v by the axis-angle aa, both (N,3) -- the check's OWN
        implementation, so the code under test is not verified against
        itself."""
        ang = torch.linalg.norm(aa, dim=-1).reshape(-1, 1)
        k = aa / torch.clamp(ang, min=1.0e-12)
        c, s = torch.cos(ang), torch.sin(ang)
        return (v * c + torch.cross(k, v, dim=-1) * s
                + k * torch.sum(k * v, dim=-1).reshape(-1, 1) * (1.0 - c))

    up = axis([(0.0, 0.0, -1.0)] * 3)
    checks = []
    checks.append(("an upright axis reads 0 deg tilt",
                   abs(float(tilt_from_axis_deg(torch, up)[0])) < 1e-4))
    lean = axis([(0.0, math.sin(math.radians(10.0)), -math.cos(math.radians(10.0)))])
    checks.append(("a 10 deg lean reads 10 deg",
                   abs(float(tilt_from_axis_deg(torch, lean)[0]) - 10.0) < 1e-3))
    # The delta really lands on the commanded tilt, checked by an independent
    # Rodrigues rotation rather than by re-running the same formula.
    for about in ("x", "y"):
        for deg in (5.0, 10.22, 15.32, 25.0):
            d = tilt_delta_pocket(torch, up.clone(), math.radians(deg), about)
            got = float(tilt_from_axis_deg(torch, rodrigues(up.clone(), d))[0])
            checks.append((f"delta about {about} reaches {deg} deg", abs(got - deg) < 1e-3))
    # From an ALREADY tilted pose the delta must close the REMAINING gap --
    # this is the iteration the solve loop relies on.
    d = tilt_delta_pocket(torch, lean.clone(), math.radians(15.32), "x")
    got = float(tilt_from_axis_deg(torch, rodrigues(lean.clone(), d))[0])
    checks.append(("delta from a tilted pose closes the gap", abs(got - 15.32) < 1e-3))
    # Already there -> zero delta, not a NaN axis.
    d0 = tilt_delta_pocket(torch, lean.clone(), math.radians(10.0), "x")
    checks.append(("no delta when already at the tilt",
                   float(torch.linalg.norm(d0, dim=-1)[0]) < 1e-6))
    checks.append(("and it is not NaN", float(d0[0][0]) == float(d0[0][0])))
    # MUTATION (D-080) 1: the two axes must NOT give the same delta. If the
    # `about` branch were dropped, every rung would tilt the same way and the
    # long-axis jam (mode B) could never be reproduced.
    dx = tilt_delta_pocket(torch, up.clone(), math.radians(10.0), "x")
    dy = tilt_delta_pocket(torch, up.clone(), math.radians(10.0), "y")
    checks.append(("tilting about x and about y differ",
                   float(torch.linalg.norm(dx - dy, dim=-1)[0]) > 1e-3))
    # MUTATION (D-080) 2: the minimal rotation adds NO yaw. A delta with a
    # z component would twist the part and the yaw channel would move.
    checks.append(("the delta carries no yaw component", abs(float(dx[0][2])) < 1e-9))
    checks.append(("neither does the y one", abs(float(dy[0][2])) < 1e-9))
    # MUTATION (D-080) 3: a bad axis name must RAISE, not silently pick one.
    try:
        tilt_delta_pocket(torch, up.clone(), 0.1, "z")
        checks.append(("a bad axis name is refused", False))
    except ValueError:
        checks.append(("a bad axis name is refused", True))
    # The two documented jam poses are the ones the docstring quotes.
    checks.append(("mode A pose is the measured one",
                   JAM_POSES["a"] == {"x_mm": 7.79, "y_mm": -15.90, "z_mm": 4.36,
                                      "tilt_deg": 10.22}))
    checks.append(("mode B pose is the measured one",
                   JAM_POSES["b"] == {"x_mm": 0.25, "y_mm": -4.70, "z_mm": -1.85,
                                      "tilt_deg": 15.32}))
    # MUTATION (D-080) 5: every dotted import path must name a file that
    # really exists in this repo. RT-160 died on
    # `insertion.tasks.insertion_tasks_cfg`, one level too shallow; changing
    # any path below to a wrong one must break this check.
    _pkg = pathlib.Path(__file__).resolve().parent.parent / "source" / "insertion" / "insertion"
    for _dotted, _rel in INSERTION_MODULES:
        checks.append((f"{_dotted} exists on disk", (_pkg / _rel).is_file()))
        checks.append((f"{_dotted} matches its file path",
                       _dotted == "insertion." + _rel[:-3].replace("/", ".")))
    checks.append(("the path RT-160 died on really is absent",
                   not (_pkg / "tasks" / "insertion_tasks_cfg.py").is_file()))

    # MUTATION (D-080) 4: RT-160 died on "unrecognized arguments: --headless".
    # The launcher's flags reach this parser too, so the FIRST parse must
    # tolerate them -- parse_known_args, never parse_args. Swapping it back
    # makes this check raise SystemExit.
    try:
        _pre, _rest = _build_parser().parse_known_args(["--headless", "--self-test"])
        checks.append(("a launcher flag does not break the first parse", _pre.self_test is True))
        checks.append(("and the launcher flag is handed on, not swallowed", "--headless" in _rest))
    except SystemExit:
        checks.append(("a launcher flag does not break the first parse", False))

    bad = [n for n, ok in checks if not ok]
    if bad:
        print(f"[tilt_recovery_probe] self-test: FAIL {bad}")
        return 1
    print(f"[tilt_recovery_probe] self-test: {len(checks)}/{len(checks)} checks passed "
          f"({SCRIPT_MARKER})")
    return 0


# --------------------------------------------------------------------------
# THE ISAAC HALF
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Righting probe: does the controller pull a jammed tilted part upright?"
    )
    p.add_argument("--task", type=str, default="Ur5e-Insertion-Direct-v0")
    p.add_argument("--disable_fabric", action="store_true", default=False,
                   help="Disable fabric and use USD I/O operations.")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--self-test", action="store_true",
                   help="Run the offline arithmetic checks and exit. No Isaac.")
    p.add_argument("--mode", choices=sorted(JAM_POSES), default="a",
                   help="Which measured jam pose to teleport to (default: a, the rim pose).")
    p.add_argument("--kp-rot", type=float, action="append", default=None, metavar="K",
                   help="Repeat once per gain to sweep. Default: 30 50 100 200.")
    p.add_argument("--tilt-deg", type=float, action="append", default=None, metavar="DEG",
                   help="Repeat once per commanded tilt. Default: the --mode pose's own tilt.")
    p.add_argument("--tilt-axis", choices=("x", "y"), default="x",
                   help="Pocket axis the tilt is about (default: x, which tips along the LONG axis).")
    p.add_argument("--tip-x-mm", type=float, default=None, help="Override the pose's tip_rel x.")
    p.add_argument("--tip-y-mm", type=float, default=None, help="Override the pose's tip_rel y.")
    p.add_argument("--tip-z-mm", type=float, default=None,
                   help="Override the pose's tip_rel z (POSITIVE is above the opening plane).")
    p.add_argument("--solve-steps", type=int, default=60,
                   help="Kinematic teleport iterations. No env.step inside (RT-80).")
    p.add_argument("--solve-tol-mm", type=float, default=0.5)
    p.add_argument("--solve-tol-deg", type=float, default=0.3)
    p.add_argument("--hold-steps", type=int, default=40,
                   help="Control steps with a ZERO action after the teleport, tilt read every step.")
    p.add_argument("--out", type=str, default="tilt_recovery_metrics.json")
    return p


def main(args) -> int:  # pragma: no cover -- needs Isaac
    # THE IMPORT BLOCK MIRRORS seat_probe.py:150-164. RT-160 (2026-09-06)
    # died here on `insertion.tasks.insertion_tasks_cfg`, a path that does not
    # exist -- `tasks/` holds only `direct/` -- and the module was not even
    # used. The two bare imports are REGISTRATION side effects: without them
    # `gym.make` cannot find the task id. `INSERTION_MODULES` pins the dotted
    # paths so the offline self-test catches a wrong one on the laptop.
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: PLC0415, E501
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms  # noqa: PLC0415

    import isaaclab_tasks  # noqa: PLC0415, F401
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    import insertion.tasks  # noqa: PLC0415, F401
    from insertion.tasks.direct.insertion import (  # noqa: PLC0415
        insertion_math,
        scripted_policy as sp,
    )

    gains = args.kp_rot if args.kp_rot else [30.0, 50.0, 100.0, 200.0]
    pose = dict(JAM_POSES[args.mode])
    tilts = args.tilt_deg if args.tilt_deg else [pose["tilt_deg"]]
    x_mm = pose["x_mm"] if args.tip_x_mm is None else args.tip_x_mm
    y_mm = pose["y_mm"] if args.tip_y_mm is None else args.tip_y_mm
    z_mm = pose["z_mm"] if args.tip_z_mm is None else args.tip_z_mm

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs,
                            use_fabric=not args.disable_fabric)
    # A MEASUREMENT env, the state seat_probe.py and scripted_insert.py run in.
    env_cfg.rl_terms_enabled = False
    # PINNED (2026-09-12). The three Phase-5 scatter fields default ON, and
    # `obs_noise.resolve_obs_noise_model` does NOT consult `rl_terms_enabled`
    # -- the noise model is built in a measurement env too, and Isaac Lab
    # applies it to `obs_buf["policy"]` in `step()`. This probe reads the
    # pocket quaternion, the tip pose and the contact force straight out of
    # that buffer (`obs_dict["policy"]` below), and the whole reading is a
    # DISPLACEMENT of a few millimetres and a force of a few newtons: 2.5 mm of
    # per-episode pocket bias and 3.5 N of per-step jitter are the same size as
    # the effect. An identity measurement needs the observation to BE the pose.
    env_cfg.obs_noise_pocket_pos_std_m = 0.0
    env_cfg.force_obs_noise_std_n = 0.0
    env_cfg.grasp_obs_offset_x_m = 0.0
    # The teleport below commands the pose; a rung-0 reset would move the pose
    # this script claims to command. Pinned, not assumed (seat_probe.py:209).
    env_cfg.start_tip_above_entrance = None
    # This probe is only meaningful under the task-space controller: the whole
    # experiment is what a ZERO action means there. Fail loudly, never quietly
    # measure joint_pd and call it a gain reading.
    if getattr(env_cfg, "control_mode", "osc") != "osc":
        raise SystemExit(f"[tilt_recovery_probe] control_mode is "
                         f"{env_cfg.control_mode!r}, not 'osc'. A zero action does not "
                         "command a righting under joint_pd, so this probe would measure "
                         "nothing. Aborting rather than reporting.")

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    device, robot, n_env = u.device, u.robot, u.num_envs
    body_idx = u._peg_body_idx
    if body_idx is None:
        raise SystemExit(f"[tilt_recovery_probe] no body {u.cfg.peg_body_name!r}: no welded tool.")
    n_joints = len(robot.joint_names)
    jacobi_idx = body_idx - 1 if robot.is_fixed_base else body_idx

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True,
                                    ik_method="dls", ik_params={"lambda_val": 0.05}),
        num_envs=n_env, device=device,
    )
    zero_action = torch.zeros(n_env, u.cfg.action_space, device=device)
    clamp_deg = math.degrees(float(u.cfg.osc_tilt_clamp_rad))

    print("[tilt_recovery_probe] ---- does the controller right a jammed tilted part? ----")
    print(f"[tilt_recovery_probe] marker: {SCRIPT_MARKER}")
    print(f"[tilt_recovery_probe] jam pose '{args.mode}': tip_rel x {x_mm:+.2f} / "
          f"y {y_mm:+.2f} / z {z_mm:+.2f} mm, tilt about pocket {args.tilt_axis}")
    print(f"[tilt_recovery_probe] gains {gains}, tilts {tilts} deg, "
          f"tilt clamp {clamp_deg:.2f} deg, hold {args.hold_steps} steps with a ZERO action")
    print("[tilt_recovery_probe] NOTE: this run sets NO threshold and recommends NO gain.")

    def tool_axis_pocket(pocket_quat):
        """Tool +z (points DOWN, insertion_math.py:1372) in the pocket frame."""
        axis_w = insertion_math.axes_from_quat(robot.data.body_quat_w[:, body_idx])[:, :, 2]
        return quat_apply_inverse(pocket_quat, axis_w)

    rungs = []
    for kp in gains:
        for tilt_deg in tilts:
            obs_dict, _ = env.reset()
            # seat_probe.py:170's `_slice`, inline: OBS_SLICES is the one
            # home for a channel offset. There is no `slice_obs` helper --
            # the hasattr guard that used to stand here was guessing.
            pocket_quat = obs_dict["policy"][
                :, slice(*insertion_math.OBS_SLICES["pocket_quat"])].clone()

            # THE GAIN, through the env's own builder so the damping (critical,
            # 2*sqrt(kp)) is rebuilt with it instead of going stale.
            u.cfg.osc_kp_rot = float(kp)
            u._build_task_space_controller()

            reset_axis = tool_axis_pocket(pocket_quat)
            reset_tilt = float(tilt_from_axis_deg(torch, reset_axis).max().item())

            # ------------------------------------------------------- SOLVE
            # Kinematic only: no env.step inside the loop (RT-80, seat_probe).
            err_mm = err_deg = None
            for _ in range(int(args.solve_steps)):
                with torch.no_grad():
                    *_, tip_rel = u._peg_geometry()
                    d_pocket = torch.zeros(n_env, 6, device=device)
                    d_pocket[:, 0] = (x_mm / 1000.0) - tip_rel[:, 0]
                    d_pocket[:, 1] = (y_mm / 1000.0) - tip_rel[:, 1]
                    d_pocket[:, 2] = (z_mm / 1000.0) - tip_rel[:, 2]
                    axis_p = tool_axis_pocket(pocket_quat)
                    d_pocket[:, 3:6] = tilt_delta_pocket(
                        torch, axis_p, math.radians(float(tilt_deg)), args.tilt_axis)
                    err_mm = float(torch.linalg.norm(d_pocket[:, 0:3], dim=-1).max().item() * 1000.0)
                    err_deg = float(torch.linalg.norm(d_pocket[:, 3:6], dim=-1).max().item()
                                    * 180.0 / math.pi)

                    root_pose_w = robot.data.root_pose_w
                    cmd_base = sp.command_to_base_frame(d_pocket, pocket_quat, root_pose_w[:, 3:7])
                    ee_pose_w = robot.data.body_pose_w[:, body_idx]
                    ee_pos_b, ee_quat_b = subtract_frame_transforms(
                        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                        ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                    jac = robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, :n_joints]
                    ik.set_command(cmd_base, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
                    # .clone() + no_grad are ONE fix, not two (RT-78,
                    # seat_probe.py:325): an inference tensor in the
                    # articulation buffers survives the next reset.
                    q_des = ik.compute(ee_pos_b, ee_quat_b, jac, robot.data.joint_pos).clone()
                    # The env's own reset triple: state, drive target, and the
                    # integrator buffer, so a zero action means HOLD.
                    robot.write_joint_state_to_sim(q_des, torch.zeros_like(q_des))
                    robot.set_joint_position_target(q_des)
                    u._joint_targets[:] = q_des
                if err_mm < args.solve_tol_mm and err_deg < args.solve_tol_deg:
                    break

            # Measured AGAIN after the last write: inside the loop the residual
            # describes the pose BEFORE that iteration's teleport.
            with torch.no_grad():
                *_, tip_rel = u._peg_geometry()
                axis_p = tool_axis_pocket(pocket_quat)
                start_tilt = float(tilt_from_axis_deg(torch, axis_p).median().item())
                err_mm = float(torch.linalg.norm(
                    torch.stack((tip_rel[:, 0] - x_mm / 1000.0,
                                 tip_rel[:, 1] - y_mm / 1000.0,
                                 tip_rel[:, 2] - z_mm / 1000.0), dim=-1),
                    dim=-1).max().item() * 1000.0)
            converged = err_mm < args.solve_tol_mm and abs(start_tilt - tilt_deg) < args.solve_tol_deg

            # -------------------------------------------------------- HOLD
            # Zero action. Under OSC that IS the standing "straighten up"
            # command: target = current pose, then clamped to the cone.
            trail, terminated_any = [], False
            for _ in range(int(args.hold_steps)):
                with torch.no_grad():
                    obs_dict, _, terminated, _, _ = env.step(zero_action)
                    if bool(terminated.any()):
                        terminated_any = True
                        break
                    trail.append(float(tilt_from_axis_deg(
                        torch, tool_axis_pocket(pocket_quat)).median().item()))
            with torch.no_grad():
                obs = obs_dict["policy"]
                sl = insertion_math.OBS_SLICES
                tip = obs[:, slice(*sl["tip_rel"])]
                frc = obs[:, slice(*sl["force"])]
                end_tilt = float(tilt_from_axis_deg(torch, tool_axis_pocket(pocket_quat))
                                 .median().item())
                force_n = float(insertion_math.force_magnitude(frc).median().item())
                lat_mm = float(torch.linalg.norm(tip[:, 0:2], dim=-1).median().item() * 1000.0)
                z_out_mm = float(tip[:, 2].median().item() * 1000.0)

            rungs.append({
                "osc_kp_rot": float(kp),
                "commanded_tilt_deg": float(tilt_deg),
                "tilt_axis": args.tilt_axis,
                "reset_tilt_deg": reset_tilt,
                "teleport_tilt_deg": start_tilt,
                "solve_residual_mm": err_mm,
                "solve_converged": bool(converged),
                "terminated_during_hold": terminated_any,
                "tilt_after_hold_deg": end_tilt,
                "tilt_min_during_hold_deg": min(trail) if trail else None,
                "tilt_trail_deg": trail,
                "force_n": force_n,
                "lateral_mm": lat_mm,
                "tip_z_mm": z_out_mm,
                "righting_moment_nm_at_teleport": None,
            })

    out = pathlib.Path(args.out)
    out.write_text(json.dumps({
        "run": "tilt_recovery_probe (righting probe, no policy)",
        "marker": SCRIPT_MARKER,
        "task": args.task, "num_envs": n_env,
        "mode": args.mode, "tip_rel_mm": {"x": x_mm, "y": y_mm, "z": z_mm},
        "tilt_clamp_deg": clamp_deg,
        "osc_kp_pos": float(u.cfg.osc_kp_pos),
        "osc_damping_ratio": float(u.cfg.osc_damping_ratio),
        "hold_steps": int(args.hold_steps),
        "rungs": rungs,
    }, indent=2), encoding="utf-8")

    print(f"\n[tilt_recovery_probe] tilt clamp {clamp_deg:.2f} deg -- a rung that RIGHTS the "
          f"part ends at or under it")
    print(f"[tilt_recovery_probe] {'kp_rot':>8} {'cmd':>7} {'teleport':>9} {'after':>7} "
          f"{'min':>7} {'force':>8} {'lat':>7}")
    for r in rungs:
        flag = "" if r["solve_converged"] else "   TELEPORT DID NOT CONVERGE -- not a result"
        if r["terminated_during_hold"]:
            flag += "   TERMINATED during hold"
        mn = r["tilt_min_during_hold_deg"]
        print(f"[tilt_recovery_probe] {r['osc_kp_rot']:8.1f} {r['commanded_tilt_deg']:7.2f} "
              f"{r['teleport_tilt_deg']:9.2f} {r['tilt_after_hold_deg']:7.2f} "
              f"{(mn if mn is not None else float('nan')):7.2f} {r['force_n']:8.2f} "
              f"{r['lateral_mm']:7.2f}{flag}")
    print(f"[tilt_recovery_probe] metrics -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    # TWO parses, and the order is forced. `--self-test` must work on the
    # laptop, where `isaaclab.app` does not import at all, so nothing Isaac
    # may be touched before that flag is read. Hence parse_known_args first:
    # it ignores the launcher's own flags instead of rejecting them.
    _PRE, _ = _build_parser().parse_known_args()
    if _PRE.self_test:
        sys.exit(_self_test())

    # AppLauncher BEFORE every other import, torch included (CLAUDE.md).
    from isaaclab.app import AppLauncher  # noqa: E402

    # RT-160 (2026-09-06) died here with "unrecognized arguments: --headless":
    # the launcher's flags (--headless, --device, --enable_cameras, ...) are
    # ADDED to this parser, they are not a second parser. seat_probe.py:135
    # is the working form and this is it.
    _parser = _build_parser()
    AppLauncher.add_app_launcher_args(_parser)
    ARGS = _parser.parse_args()

    app_launcher = AppLauncher(ARGS)
    simulation_app = app_launcher.app

    _TOOLS = pathlib.Path(__file__).resolve().parent / "tools"
    _spec = importlib.util.spec_from_file_location("isaac_exit", _TOOLS / "isaac_exit.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["isaac_exit"] = _mod
    _spec.loader.exec_module(_mod)
    exit_with = _mod.exit_with

    _code = 0
    try:
        _code = main(ARGS)
    except SystemExit as _e:  # a fail-fast message is a result, not a crash
        print(str(_e))
        _code = 1
    except Exception:  # noqa: BLE001 -- the traceback IS the log
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        _code = 1
    # D-081: (app, code, tag) -- exit BEFORE the shutdown, because
    # simulation_app.close() does not return (RT-22).
    exit_with(simulation_app, _code, tag="tilt_recovery_probe")
