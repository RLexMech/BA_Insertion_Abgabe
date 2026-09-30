# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""The PHYSICS identity test for the insertion contact. It does not descend.

WHY THIS FILE EXISTS
--------------------
`CLAUDE.md` § Code names a strict order and says never to break it:

    1. Identitätstests für Physik, Observations und Reward-Terme.
    2. Scripted solvability.
    3. PPO-Tuning.

Steps 1 for OBSERVATIONS and REWARD exist and are green (`check_env_wiring`,
`check_insertion_math`, `check_demo_reward_math`). For the PHYSICS of the
insertion contact there was none. `scripts/CLAUDE.md` indexed 34 scripts and not
one put the part in the pocket and asked the simulation whether the seat is
free -- so every run from RT-68 to RT-76 measured a CONTROLLER against a contact
nobody had verified. This is that missing test.

WHAT IT DOES, AND WHY IT IS NOT scripted_insert AGAIN
-----------------------------------------------------
`scripted_insert.py` DESCENDS: 0.5 mm per control step through the env's action
interface, so what it measures is the whole chain -- controller, integrator,
drives, contact. A jam anywhere in that chain looks the same from outside.

This script TELEPORTS. It solves the arm kinematically to a commanded depth,
writes the joint state straight into the sim (the env's OWN reset mechanism,
`insertion_env.py:1038`), and only then lets the physics settle and speak. The
descent is removed from the question, so what is left is the contact alone.

Run as a LADDER of commanded depths. Both outcomes answer the question:

  * force stays low up to the success depth -> the geometry ACCEPTS the seat.
    The jam is in the descent path, and the controller work resumes knowing the
    target is reachable.
  * force explodes at the same depth the descent stalls at -> the COLLIDER
    forbids the seat. No controller and no policy can ever succeed, the D-108
    gate cannot be passed as the asset stands, and the work moves to the asset
    (the 9 boundary edges of RT-61, the SDF sign, the voxel pitch against the
    0.2938 mm per-side play).

IT SETS NO THRESHOLD. What counts as "low force" at a seated pose is a decision
nobody has taken; this run reports the distribution the decision needs.

THE RESET-STEP TRAP (`CLAUDE.md` § Error Handling) is why SETTLE exists at all:
sensor and link data in the same step as a reset are stale, so nothing is read
until the arm has been left alone for `--settle-steps` control steps, and that
count is printed with every number it produced.
"""

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

# GUARDED for the same reason scripted_insert.py guards: RT-66 died on an
# import-time TorchScript failure whose traceback printed while the wrapper
# still reported exit code 0. A failing run that reports success is the one
# outcome D-081 exists to prevent.
try:
    import json  # noqa: E402

    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402

    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
    from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

    import isaaclab_tasks  # noqa: F401, E402
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

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
    """Read one observation block by NAME, from the one table that owns it.

    Same helper and same reason as `scripted_insert.py`: `OBS_SLICES` is the
    single home of the channel layout, and indexing by name keeps this script
    correct when a block moves.
    """
    lo, hi = insertion_math.OBS_SLICES[name]
    return obs[:, lo:hi]


def _percentiles(values: list) -> dict:
    """p50 / max over a list, or an explicit empty marker.

    Deliberately narrower than `scripted_insert.py`'s: this run has one sample
    per env per rung, so p90/p95 over a handful of envs would be dressing.
    """
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
    # A MEASUREMENT env, the same state zero_agent.py and scripted_insert.py
    # run in: nothing learns here, and it is the only state that builds while
    # the RL_PENDING values are unset.
    env_cfg.rl_terms_enabled = False

    # PINNED (2026-09-12). The three Phase-5 scatter fields default ON, and
    # `obs_noise.resolve_obs_noise_model` does NOT consult `rl_terms_enabled`
    # -- the noise model is built in a measurement env too, and Isaac Lab
    # applies it to `obs_buf["policy"]` in `step()`. This probe reads its
    # depth and its contact force OUT OF THAT BUFFER (`_slice(obs, "tip_rel")`
    # and `_slice(obs, "force")` below), so the scatter would land straight in
    # the measurement: 2.5 mm of per-episode pocket bias on the commanded
    # depth and 3.5 N of per-step jitter on the very force this script exists
    # to read. An identity test needs the observation to BE the pose.
    env_cfg.obs_noise_pocket_pos_std_m = 0.0
    env_cfg.force_obs_noise_std_n = 0.0
    env_cfg.grasp_obs_offset_x_m = 0.0

    # THE START POSE STAYS THE HOME POSE (D-161, 2026-08-31). The cfg default
    # is now rung 0 -- the env solves an IK at every reset and starts the tool
    # point inside the pocket. This run does its OWN teleport and reports
    # against the 165 mm home stand-off, so a rung-0 reset would move the pose
    # this script claims to command. Pinned here, not assumed.
    env_cfg.start_tip_above_entrance = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.robot
    num_envs = unwrapped.num_envs

    # ---------------------------------------------------------------- fail fast
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

        # Every rung starts from a clean reset, so a rung cannot inherit the
        # pose -- or the contact -- the previous one ended in.
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        # Read ONCE, here: the fixture does not move inside a rung, and the
        # solve below never steps the env, so no later observation exists to
        # read it from.
        pocket_quat = _slice(obs, "pocket_quat").clone()

        # ---------------------------------------------------------- SOLVE
        # KINEMATIC ONLY, and since RT-80 it really is one.
        #
        # WHAT RT-80 MEASURED, and it is a defect of this script, not of the
        # asset: the loop called ``env.step`` after every write, and
        # ``DirectRLEnv.step`` runs ``decimation`` physics substeps
        # (direct_rl_env.py:370). So the contact pushed the arm away after each
        # teleport and the solver chased a target the physics kept undoing.
        # Every rung from 15 mm on -- the depths where the part is inside the
        # pocket -- reported ``IK DID NOT CONVERGE`` at forces up to 15262 N and
        # lateral offsets up to 62.9 mm, while 5 mm and 10 mm, where nothing
        # touches at 8.1 N, converged. ``env.step`` could additionally
        # TERMINATE and reset an env mid-solve (``terminated = success |
        # below_plate``, insertion_env.py:971; the reset happens inside
        # ``step()``, direct_rl_env.py:396-398), leaving one env at the home
        # pose while the others were in the pocket -- and ``solved_err_mm``
        # takes the max over envs, so that one env failed the whole rung.
        #
        # NO STEP IS NEEDED. ``write_joint_position_to_sim`` sets
        # ``_body_link_pose_w.timestamp = -1.0`` (articulation.py:605-610), and
        # the next read of ``body_pose_w`` re-runs the forward kinematics
        # (articulation_data.py:590-592). The pose is fresh with no physics at
        # all. This is the same mechanism ``DirectRLEnv.reset`` relies on.
        solved_err_mm = None
        for _ in range(int(args_cli.solve_steps)):
            with torch.no_grad():
                # From the env's OWN geometry, not from an observation: there is
                # no step in this loop to produce one. seat_probe is explicitly
                # outside the D-108 information boundary (check_env_wiring.py
                # says so and why) -- it is a probe, not the gate.
                *_, tip_rel = unwrapped._peg_geometry()

                # The wanted step, in the POCKET frame: null both lateral
                # components and go to the rung depth. Rotation is left alone --
                # the part is already parallel and this test is about position.
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
                # ``.clone()`` and ``no_grad`` above are one fix, not two.
                # RT-78 died here: under ``inference_mode`` the solver's
                # output is an INFERENCE TENSOR, it lands in the
                # articulation's buffers, and the next ``env.reset()``
                # cannot overwrite them (articulation.py:578). The clone
                # additionally stops the env's integrator buffer from
                # aliasing the tensor handed to the sim.
                joint_pos_des = ik.compute(
                    ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos
                ).clone()

                # THE TELEPORT, and it is the env's own reset triple
                # (insertion_env.py:1038-1041 plus the target buffer). All three
                # parts are needed: the state so the arm IS there, the drive
                # target so it is not immediately pulled back, and the env's
                # integrator buffer so the next zero action means "hold" rather
                # than "return to where you were".
                robot.write_joint_state_to_sim(joint_pos_des, torch.zeros_like(joint_pos_des))
                robot.set_joint_position_target(joint_pos_des)
                unwrapped._joint_targets[:] = joint_pos_des

            if solved_err_mm is not None and solved_err_mm < float(args_cli.solve_tol_mm):
                break

        # THE RESIDUAL IS MEASURED AGAIN, after the last write. Inside the loop
        # it is computed BEFORE the teleport of that iteration, so the pose the
        # settle actually starts from was never the pose the residual described.
        # One extra read closes that gap; it costs a forward kinematics call.
        with torch.no_grad():
            *_, tip_rel = unwrapped._peg_geometry()
            solved_err_mm = float(torch.linalg.norm(
                torch.stack((tip_rel[:, 0], tip_rel[:, 1], tip_rel[:, 2] - target_rel_z), dim=-1),
                dim=-1,
            ).max().item() * 1000.0)

        converged = solved_err_mm is not None and solved_err_mm < float(args_cli.solve_tol_mm)

        # --------------------------------------------------------- SETTLE
        # No more writing. Zero action, so the drives hold the teleported
        # target and everything that happens now is the contact.
        terminated_any = False
        for _ in range(int(args_cli.settle_steps)):
            with torch.no_grad():
                obs_dict, _, terminated, truncated, _ = env.step(zero_action)
                obs = obs_dict["policy"]
                if bool(terminated.any()):
                    terminated_any = True
                    # A termination RESETS that env inside step(), so anything
                    # read after this is the post-reset pose, not the seat. Stop
                    # and report it -- a success at the seated pose is itself
                    # the answer this run is looking for.
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

    # PRINTED AS WELL AS WRITTEN. RT-70 taught this: a metrics file whose path
    # is printed but whose contents are not proves nothing in a log, and the log
    # is what /rt-check reads.
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
    # D-081: set the exit status BEFORE the shutdown. close() does not return
    # (RT-22), so anything after it never decides anything.
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_with(simulation_app, 1, tag="seat_probe-main")
    exit_with(simulation_app, 0, tag="seat_probe")
