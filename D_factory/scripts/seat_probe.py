# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""The PHYSICS identity test for the insertion contact."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import pathlib
import sys
import traceback

from isaaclab.app import AppLauncher

_TOOLS = pathlib.Path(__file__).resolve().parent / "tools"
_spec = importlib.util.spec_from_file_location("isaac_exit", _TOOLS / "isaac_exit.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["isaac_exit"] = _mod
_spec.loader.exec_module(_mod)
exit_with = _mod.exit_with

SCRIPT_MARKER = "seat_probe-2026-08-29b"

parser = argparse.ArgumentParser(
    description="Physics identity test: teleport the part to a seated pose and read the contact."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--out",
    type=str,
    default="seat_probe_metrics.json",
    help="Metrics file. Numbers are read from files, never from prose.",
)

probe = parser.add_argument_group("probe (measurement settings, not task decisions)")
probe.add_argument(
    "--depths-mm",
    type=float,
    nargs="+",
    default=[5.0, 10.0, 15.0, 20.0, 24.0, 28.0, 33.0, 36.0],
    help=(
        "Commanded depths below the stage-2 opening plane, in mm. The default "
        "ladder brackets what the descent runs reached (RT-74/RT-76 stall near "
        "19.5-24.2 mm), the success depth (33 mm) and the measured floor "
        "(36 mm), so the rung where the contact changes is inside the range "
        "rather than at its edge."
    ),
)
probe.add_argument(
    "--solve-steps",
    type=int,
    default=120,
    help=(
        "Kinematic IK iterations per rung. Each iteration TELEPORTS, so contact "
        "cannot accumulate into the solve -- the arm is put where the geometry "
        "says, not where the physics allows. Convergence is reported per rung; "
        "a rung that did not converge is a finding and is NOT read as a "
        "contact result."
    ),
)
probe.add_argument(
    "--settle-steps",
    type=int,
    default=30,
    help=(
        "Control steps with a ZERO action after the teleport, before anything "
        "is read. Zero action means the drives hold the teleported target, so "
        "this is the physics answering. Required by the reset-step trap: link "
        "and sensor data in the same step as a state write are stale."
    ),
)
probe.add_argument("--ik-lambda", type=float, default=0.05, help="dls damping (Isaac Lab default is 0.01).")
probe.add_argument(
    "--solve-tol-mm",
    type=float,
    default=0.05,
    help=(
        "Kinematic convergence tolerance for the SOLVE phase, in mm. It judges "
        "the IK only -- it is not a task tolerance and nothing physical is "
        "compared against it."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

try:
    import json  # noqa: E402

    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402

    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
    from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

    import isaaclab_tasks  # noqa: F401, E402
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "source" / "insertion"))
    import insertion  # noqa: E402
    print(f"[seat_probe] insertion package imported from: {pathlib.Path(insertion.__file__).resolve().parent}")
    import insertion.tasks  # noqa: F401, E402
    from insertion.tasks.direct.insertion import (  # noqa: E402
        insertion_math,
        insertion_tasks_cfg,
        scripted_policy as sp,
    )
except BaseException:
    traceback.print_exc()
    exit_with(simulation_app, 1, tag="seat_probe-import")


def _slice(obs: torch.Tensor, name: str) -> torch.Tensor:
    """Read one observation block by NAME, from the one table that owns it."""
    lo, hi = insertion_math.OBS_SLICES[name]
    return obs[:, lo:hi]


def _percentiles(values: list) -> dict:
    """p50 / max over a list, or an explicit empty marker."""
    if not values:
        return {"n": 0, "p50": None, "max": None}
    s = sorted(values)
    return {
        "n": len(s),
        "p50": s[max(0, min(len(s) - 1, int(round(0.5 * (len(s) - 1)))))],
        "max": s[-1],
    }


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.rl_terms_enabled = False

    # Keep diagnostic measurements independent of training defaults.
    env_cfg.obs_wrench_mode = "force"
    env_cfg.start_floor_m = 0.020
    env_cfg.autodr_stall_buffers = 0
    env_cfg.pocket_obs_yaw_rad = 0.0
    env_cfg.obs_noise_pocket_pos_std_m = 0.0
    env_cfg.force_obs_noise_std_n = 0.0
    env_cfg.grasp_obs_offset_x_m = 0.0

    env_cfg.start_tip_above_entrance = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.robot
    num_envs = unwrapped.num_envs

    peg_body_idx = unwrapped._peg_body_idx
    if peg_body_idx is None:
        raise SystemExit(
            f"[seat_probe] the robot carries no body '{unwrapped.cfg.peg_body_name}', so there is no part "
            "to seat. This test is meaningless without the welded tool -- author it with "
            "scripts/author_tool_ur5e.py first. Bodies present: " + str(robot.body_names)
        )
    n_joints = len(robot.joint_names)
    jacobi_idx = peg_body_idx - 1 if robot.is_fixed_base else peg_body_idx

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": args_cli.ik_lambda},
        ),
        num_envs=num_envs,
        device=device,
    )

    seat_depth_m = float(insertion_tasks_cfg.POCKET_SEAT_DEPTH)
    success_depth_m = float(insertion_tasks_cfg.SEATED_SUCCESS_DEPTH)

    print("[seat_probe] ---- physics identity test: does the pocket accept the seat? ----")
    print(f"[seat_probe] marker: {SCRIPT_MARKER}")
    print(f"[seat_probe] tool body: {unwrapped.cfg.peg_body_name} (index {peg_body_idx}), "
          f"jacobian row block {jacobi_idx}, fixed base {robot.is_fixed_base}")
    print(f"[seat_probe] measured floor {seat_depth_m * 1000.0:.1f} mm, "
          f"success at {success_depth_m * 1000.0:.1f} mm, "
          f"per-side cross play {insertion_tasks_cfg.PLAY_X * 500.0:.4f} mm")
    print(f"[seat_probe] ladder: {args_cli.depths_mm} mm   "
          f"solve {args_cli.solve_steps} teleport iterations, "
          f"settle {args_cli.settle_steps} control steps with a ZERO action")
    print("[seat_probe] NOTE: this run sets NO force threshold. What counts as an acceptable "
          "force at a seated pose is a decision, not a measurement.")

    zero_action = torch.zeros(num_envs, unwrapped.cfg.action_space, device=device)
    rungs = []

    for depth_mm in args_cli.depths_mm:
        target_rel_z = -float(depth_mm) / 1000.0

        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        pocket_quat = _slice(obs, "pocket_quat").clone()

        solved_err_mm = None
        for _ in range(int(args_cli.solve_steps)):
            with torch.no_grad():
                *_, tip_rel = unwrapped._peg_geometry()

                d_pocket = torch.zeros(num_envs, 6, device=device)
                d_pocket[:, 0] = -tip_rel[:, 0]
                d_pocket[:, 1] = -tip_rel[:, 1]
                d_pocket[:, 2] = target_rel_z - tip_rel[:, 2]
                solved_err_mm = float(
                    torch.linalg.norm(d_pocket[:, 0:3], dim=-1).max().item() * 1000.0
                )

                root_pose_w = robot.data.root_pose_w
                cmd_base = sp.command_to_base_frame(d_pocket, pocket_quat, root_pose_w[:, 3:7])

                ee_pose_w = robot.data.body_pose_w[:, peg_body_idx]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, :n_joints]
                ik.set_command(cmd_base, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
                joint_pos_des = ik.compute(
                    ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos
                ).clone()

                robot.write_joint_state_to_sim(joint_pos_des, torch.zeros_like(joint_pos_des))
                robot.set_joint_position_target(joint_pos_des)
                unwrapped._joint_targets[:] = joint_pos_des

            if solved_err_mm is not None and solved_err_mm < float(args_cli.solve_tol_mm):
                break

        with torch.no_grad():
            *_, tip_rel = unwrapped._peg_geometry()
            solved_err_mm = float(torch.linalg.norm(
                torch.stack((tip_rel[:, 0], tip_rel[:, 1], tip_rel[:, 2] - target_rel_z), dim=-1),
                dim=-1,
            ).max().item() * 1000.0)

        converged = solved_err_mm is not None and solved_err_mm < float(args_cli.solve_tol_mm)

        terminated_any = False
        for _ in range(int(args_cli.settle_steps)):
            with torch.no_grad():
                obs_dict, _, terminated, truncated, _ = env.step(zero_action)
                obs = obs_dict["policy"]
                if bool(terminated.any()):
                    terminated_any = True
                    break

        with torch.no_grad():
            tip_rel = _slice(obs, "tip_rel")
            force = _slice(obs, "force")
            achieved_mm = [float(v) * -1000.0 for v in tip_rel[:, 2].tolist()]
            lateral_mm = [
                float(v) * 1000.0
                for v in torch.linalg.norm(tip_rel[:, 0:2], dim=-1).tolist()
            ]
            force_n = [float(v) for v in insertion_math.force_magnitude(force).tolist()]

        rungs.append({
            "commanded_depth_mm": float(depth_mm),
            "solve_converged": bool(converged),
            "solve_residual_mm": solved_err_mm,
            "terminated_during_settle": terminated_any,
            "achieved_depth_mm": _percentiles(achieved_mm),
            "lateral_offset_mm": _percentiles(lateral_mm),
            "force_n": _percentiles(force_n),
        })

    metrics = {
        "run": "seat_probe (physics identity test)",
        "marker": SCRIPT_MARKER,
        "task": args_cli.task,
        "num_envs": num_envs,
        "seat_depth_mm": seat_depth_m * 1000.0,
        "success_depth_mm": success_depth_m * 1000.0,
        "play_x_per_side_mm": float(insertion_tasks_cfg.PLAY_X) * 500.0,
        "probe": {
            "depths_mm": list(args_cli.depths_mm),
            "solve_steps": int(args_cli.solve_steps),
            "settle_steps": int(args_cli.settle_steps),
            "solve_tol_mm": float(args_cli.solve_tol_mm),
            "ik_lambda": float(args_cli.ik_lambda),
        },
        "rungs": rungs,
    }
    out = pathlib.Path(args_cli.out)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[seat_probe] commanded -> achieved depth, and the force at the seat:")
    for r in rungs:
        flag = "" if r["solve_converged"] else "  IK DID NOT CONVERGE -- not a contact result"
        end = "  TERMINATED during settle" if r["terminated_during_settle"] else ""
        print(f"[seat_probe]   commanded {r['commanded_depth_mm']:6.1f} mm  "
              f"achieved p50 {r['achieved_depth_mm']['p50']:7.3f} mm  "
              f"force p50 {r['force_n']['p50']:8.2f} N  max {r['force_n']['max']:8.2f} N  "
              f"lateral p50 {r['lateral_offset_mm']['p50']:6.3f} mm  "
              f"solve residual {r['solve_residual_mm']:7.3f} mm{flag}{end}")
    print(f"[seat_probe] metrics -> {out.resolve()}")
    print("[seat_probe] READ IT LIKE THIS: an achieved depth that tracks the commanded one at a low "
          "force means the geometry ACCEPTS the seat and the jam is in the descent. An achieved "
          "depth that stops following, at a force that climbs, means the COLLIDER forbids it and no "
          "controller can pass the D-108 gate as the asset stands.")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_with(simulation_app, 1, tag="seat_probe-main")
    exit_with(simulation_app, 0, tag="seat_probe")
