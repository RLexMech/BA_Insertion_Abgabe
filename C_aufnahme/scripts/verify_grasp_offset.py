# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure whether the welded part can take a per-reset flange offset (D-070)."""

from __future__ import annotations

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

parser = argparse.ArgumentParser(description="Probe a runtime flange-to-part offset on the welded tool (D-070).")
parser.add_argument("--task", type=str, required=True, help="Task name, e.g. Ur5e-Insertion-Direct-v0.")
parser.add_argument(
    "--offset",
    type=float,
    required=True,
    help="Offset in metres written to tool_weld physics:localPos0 x, env 0 only. 0.0 is the negative control.",
)
parser.add_argument("--steps", type=int, default=10, help="Sim steps after the write before the second read.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.num_envs = 2

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import insertion.tasks  # noqa: F401, E402

TAG = "verify_grasp_offset"
MARKER = "verify_grasp_offset-2026-08-27a"
MOVE_TOL_M = 1.0e-6


def rel_part_pose(env) -> torch.Tensor:
    """Part (tool_link) position in the flange (wrist_3) frame, per env, metres."""
    robot = env.unwrapped.robot
    flange_idx = robot.find_bodies("wrist_3_link")[0][0]
    part_idx = robot.find_bodies("tool_link")[0][0]
    fl_pos = robot.data.body_pos_w[:, flange_idx]
    fl_quat = robot.data.body_quat_w[:, flange_idx]
    pt_pos = robot.data.body_pos_w[:, part_idx]
    d = pt_pos - fl_pos
    w, x, y, z = fl_quat.unbind(-1)
    cw, cx, cy, cz = w, -x, -y, -z
    t = 2.0 * torch.cross(torch.stack([cx, cy, cz], dim=-1), d, dim=-1)
    return d + cw.unsqueeze(-1) * t + torch.cross(torch.stack([cx, cy, cz], dim=-1), t, dim=-1)


def main() -> int:
    print(f"[{TAG}] marker: {MARKER}")
    print(f"[{TAG}] offset to write: {args_cli.offset:+.6f} m on env 0 tool_weld physics:localPos0.x")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=2)
    env = gym.make(args_cli.task, cfg=env_cfg)
    with torch.inference_mode():
        env.reset()
        zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(2):
            env.step(zero)

    before = rel_part_pose(env).clone()
    for i in range(2):
        print(f"[{TAG}] BEFORE env {i}: part in flange frame "
              f"({before[i, 0]:+.9f}, {before[i, 1]:+.9f}, {before[i, 2]:+.9f}) m")

    from pxr import Usd, UsdPhysics, Gf  # noqa: PLC0415
    from isaaclab.sim.utils.stage import get_current_stage  # noqa: PLC0415
    stage = get_current_stage()
    robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if not robot_root or not robot_root.IsValid():
        print(f"[{TAG}] FAIL: /World/envs/env_0/Robot not found")
        return 1
    joint_prim = None
    for p in Usd.PrimRange(robot_root):
        if p.GetName() == "tool_weld" and p.IsA(UsdPhysics.FixedJoint):
            joint_prim = p
            break
    if joint_prim is None:
        print(f"[{TAG}] FAIL: no FixedJoint named tool_weld under env 0's robot")
        return 1
    attr = joint_prim.GetAttribute("physics:localPos0")
    old_val = attr.Get()
    print(f"[{TAG}] joint: {joint_prim.GetPath()}, physics:localPos0 before write: {old_val}")
    attr.Set(Gf.Vec3f(float(args_cli.offset), 0.0, 0.0))
    print(f"[{TAG}] wrote physics:localPos0 = ({args_cli.offset}, 0, 0) on env 0")

    with torch.inference_mode():
        env.reset()
        for _ in range(int(args_cli.steps)):
            env.step(zero)

    after = rel_part_pose(env).clone()
    delta = (after - before).norm(dim=-1)
    for i in range(2):
        print(f"[{TAG}] AFTER  env {i}: part in flange frame "
              f"({after[i, 0]:+.9f}, {after[i, 1]:+.9f}, {after[i, 2]:+.9f}) m, "
              f"|delta| = {delta[i]:.9f} m")

    moved0 = bool(delta[0] > MOVE_TOL_M)
    moved1 = bool(delta[1] > MOVE_TOL_M)

    if args_cli.offset == 0.0:
        ok = not moved0 and not moved1
        print(f"[{TAG}] NEGATIVE CONTROL: {'PASS -- both envs bit-still' if ok else 'FAIL -- the measurement itself moves the part'}")
        return 0 if ok else 1

    print(f"[{TAG}] RESULT env 0 (written): {'MOVED' if moved0 else 'NOT MOVED'} "
          f"({delta[0]:.9f} m vs written {abs(args_cli.offset):.6f} m)")
    print(f"[{TAG}] RESULT env 1 (untouched): {'MOVED -- the write LEAKED across envs' if moved1 else 'still -- no leak'}")
    if moved0:
        print(f"[{TAG}] => a runtime USD write on the weld joint reaches PhysX; the D-070 "
              "offset can use this route (per-env values need per-env writes).")
    else:
        print(f"[{TAG}] => the joint frame was consumed at articulation parse time; this "
              "route cannot apply a runtime offset; a different mechanism is required.")
    return 0


if __name__ == "__main__":
    _code = 1
    try:
        _code = main()
        if _code is None:
            _code = 1
    except SystemExit as _exc:
        _code = int(_exc.code or 0)
    except BaseException:
        traceback.print_exc()
        _code = 1
    exit_with(simulation_app, _code, tag=TAG)
