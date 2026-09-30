# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Probe the robot and fixture assets before the first environment increment.

UNVERIFIED — authored on the dev PC (no Isaac installation); run on the
training machine:

    cd C:\\Isaaclab
    conda activate env_isaaclab
    .\\isaaclab.bat -p <proxytask>\\scripts\\probe_assets.py --usd <path\\to\\Tisch.usd>

Answers three questions that the environment code depends on and that cannot be
decided from the dev PC:

  1. Which UR articulation configs does ``isaaclab_assets`` actually ship, and
     which USD does each point at? DECISIONS D-018/D-020 assume a shipped
     ``UR10e_CFG``; this reports whether that is true. UR10 and UR10e differ in
     link lengths, so the answer selects the DH table used by
     ``scripts/tools/compute_home_pose.py``.
  2. For each candidate robot USD: default prim, joint prims with their body
     relations, rigid-body links, and where ``PhysicsArticulationRootAPI`` sits.
     CAVEAT: the joint order printed here is stage-traversal order. The binding
     order is whatever ``Articulation.joint_names`` reports at runtime; that is
     what the environment startup report prints, and that is the authority.
  3. For the fixture stage (--usd): its ``defaultPrim``. ``UsdFileCfg`` spawns by
     adding a reference, which resolves to nothing when no default prim is set.

Read-only: opens stages, writes nothing, modifies nothing.
"""

from __future__ import annotations

import argparse
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe robot and fixture assets.")
parser.add_argument(
    "--usd",
    type=str,
    default=None,
    help="Optional fixture USD (Tisch.usd) to report the default prim for.",
)
parser.add_argument(
    "--robot-usd",
    type=str,
    action="append",
    default=None,
    help="Extra robot USD path to probe. Repeatable; added to the built-in candidates.",
)
parser.add_argument(
    "--only-api",
    action="store_true",
    help=(
        "Run section 4 only. Sections 1-3 open USDs from the Nucleus cloud, which is "
        "the slow part; section 4 is pure introspection and needs no asset."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Both isaaclab_assets and pxr are importable only after the app is up.
from pxr import Tf, Usd, UsdPhysics  # noqa: E402

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def report_shipped_cfgs() -> list[str]:
    """List UR-related ArticulationCfg objects shipped by isaaclab_assets.

    Returns the usd_path of every config found, for use as probe candidates.
    """
    rule("1. Shipped configs in isaaclab_assets.robots.universal_robots")

    print(f"ISAAC_NUCLEUS_DIR    = {ISAAC_NUCLEUS_DIR}")
    print(f"ISAACLAB_NUCLEUS_DIR = {ISAACLAB_NUCLEUS_DIR}")
    print()

    try:
        from isaaclab_assets.robots import universal_robots as ur_module
    except ImportError as exc:
        print(f"FAIL: cannot import isaaclab_assets.robots.universal_robots: {exc}")
        return []

    print(f"module file: {ur_module.__file__}")
    found: list[str] = []
    names = sorted(n for n in dir(ur_module) if n.endswith("_CFG"))
    if not names:
        print("FAIL: module exports no *_CFG names.")
        return []

    for name in names:
        cfg = getattr(ur_module, name)
        usd_path = getattr(getattr(cfg, "spawn", None), "usd_path", None)
        print(f"  {name:<28} usd_path = {usd_path}")
        if usd_path:
            found.append(usd_path)

    has_ur10e = any(n.upper().startswith("UR10E") for n in names)
    print()
    print(f"  -> UR10e_CFG shipped: {'YES' if has_ur10e else 'NO'}")
    if not has_ur10e:
        print("  -> Outcome B or C: define our own ArticulationCfg (plan section 2.1).")
    return found


def probe_robot_stage(usd_path: str) -> None:
    """Print the articulation structure of a robot USD."""
    print()
    print("-" * 78)
    print(f"USD: {usd_path}")
    print("-" * 78)

    # A missing layer raises rather than returning None, and a probe must never
    # abort the run just because one speculative candidate does not exist.
    try:
        stage = Usd.Stage.Open(usd_path)
    except Tf.ErrorException as exc:
        print(f"  not openable (asset missing or path wrong): {str(exc).strip().splitlines()[-1]}")
        return
    if stage is None:
        print("  not openable (asset missing or path wrong)")
        return

    default_prim = stage.GetDefaultPrim()
    print(f"  defaultPrim: {default_prim.GetPath() if default_prim else 'NONE (!)'}")

    joints: list[tuple[str, str, str, str]] = []
    links: list[str] = []
    art_roots: list[str] = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_roots.append(path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.append(path)
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            # localPos0/localRot0 is the joint frame relative to body0. For a
            # marker-only fixed joint such as ur10e's ee_joint (body1 empty)
            # this is the flange offset from wrist_3_link -- the number needed
            # to compare the IK target (DH frame 6, i.e. the flange) against
            # the only pose the environment can actually read, which is the
            # rigid body wrist_3_link.
            local_pos = joint.GetLocalPos0Attr().Get()
            local_rot = joint.GetLocalRot0Attr().Get()
            joints.append(
                (
                    path,
                    prim.GetTypeName(),
                    str(body0[0]) if body0 else "-",
                    str(body1[0]) if body1 else "-",
                    local_pos,
                    local_rot,
                )
            )

    print(f"  articulation root API on: {art_roots if art_roots else 'NONE (!)'}")

    print(f"  joints ({len(joints)}, stage-traversal order — NOT necessarily binding order):")
    for path, type_name, body0, body1, local_pos, local_rot in joints:
        print(f"    {path}")
        print(f"        type={type_name}  body0={body0}  body1={body1}")
        print(f"        localPos0={local_pos}  localRot0={local_rot}")

    print(f"  rigid-body links ({len(links)}):")
    for path in links:
        print(f"    {path}")


def probe_fixture_stage(usd_path: str) -> None:
    rule("3. Fixture stage")
    resolved = pathlib.Path(usd_path).resolve()
    print(f"USD: {resolved}")
    if not resolved.is_file():
        print("  FAIL: file does not exist on this machine.")
        return

    try:
        stage = Usd.Stage.Open(str(resolved))
    except Tf.ErrorException as exc:
        print(f"  FAIL: could not open stage: {str(exc).strip().splitlines()[-1]}")
        return
    if stage is None:
        print("  FAIL: could not open stage.")
        return

    default_prim = stage.GetDefaultPrim()
    if default_prim:
        print(f"  defaultPrim: {default_prim.GetPath()}  (type {default_prim.GetTypeName()})")
        print("  -> UsdFileCfg reference will resolve.")
    else:
        print("  defaultPrim: NONE")
        print("  -> UsdFileCfg reference resolves to nothing; the inner prim must be")
        print("     spawned explicitly, or the stage re-saved with a default prim set.")

    print("  root-level prims:")
    for prim in stage.GetPseudoRoot().GetChildren():
        print(f"    {prim.GetPath()}  (type {prim.GetTypeName()})")


def report_api_surface() -> None:
    """Dump the real field names and signatures the environment code is written against.

    The dev PC has no Isaac installation, so every cfg field and method signature
    used in the environment would otherwise be written from memory and validated
    by crashing on the training machine, one guess per round trip. One
    introspection pass replaces that.
    """
    rule("4. API surface (field names and signatures used by the environment)")

    import dataclasses
    import inspect
    import pprint

    def dump_fields(label: str, dotted: str) -> None:
        module_name, _, attr = dotted.rpartition(".")
        try:
            module = __import__(module_name, fromlist=[attr])
            obj = getattr(module, attr)
        except (ImportError, AttributeError) as exc:
            print(f"  {label:<34} MISSING ({exc})")
            return
        try:
            names = [f.name for f in dataclasses.fields(obj)]
        except TypeError:
            print(f"  {label:<34} not a dataclass; dir(): {sorted(n for n in dir(obj) if not n.startswith('_'))}")
            return
        print(f"  {label:<34} {names}")

    print("-- config classes --")
    for label, dotted in [
        ("SimulationCfg", "isaaclab.sim.SimulationCfg"),
        ("InteractiveSceneCfg", "isaaclab.scene.InteractiveSceneCfg"),
        ("DirectRLEnvCfg", "isaaclab.envs.DirectRLEnvCfg"),
        ("ArticulationCfg", "isaaclab.assets.ArticulationCfg"),
        ("ArticulationCfg.InitialStateCfg", "isaaclab.assets.ArticulationCfg.InitialStateCfg"),
        ("AssetBaseCfg", "isaaclab.assets.AssetBaseCfg"),
        ("UsdFileCfg", "isaaclab.sim.UsdFileCfg"),
        ("RigidBodyPropertiesCfg", "isaaclab.sim.RigidBodyPropertiesCfg"),
        ("ArticulationRootPropertiesCfg", "isaaclab.sim.ArticulationRootPropertiesCfg"),
        ("CollisionPropertiesCfg", "isaaclab.sim.CollisionPropertiesCfg"),
        ("MeshCollisionPropertiesCfg", "isaaclab.sim.MeshCollisionPropertiesCfg"),
        ("ImplicitActuatorCfg", "isaaclab.actuators.ImplicitActuatorCfg"),
    ]:
        dump_fields(label, dotted)

    print()
    print("-- signatures --")
    try:
        from isaaclab.assets import Articulation
        from isaaclab.envs import DirectRLEnv
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim.spawners.from_files import spawn_from_usd

        for label, fn in [
            ("spawn_from_usd", spawn_from_usd),
            ("Articulation.write_joint_state_to_sim", Articulation.write_joint_state_to_sim),
            ("Articulation.write_root_pose_to_sim", Articulation.write_root_pose_to_sim),
            ("Articulation.find_bodies", Articulation.find_bodies),
            ("InteractiveScene.clone_environments", InteractiveScene.clone_environments),
            ("InteractiveScene.filter_collisions", InteractiveScene.filter_collisions),
        ]:
            print(f"  {label}{inspect.signature(fn)}")
    except (ImportError, AttributeError, ValueError) as exc:
        print(f"  signature dump failed: {exc}")

    print()
    print("-- ArticulationData attributes (pose/state names differ across 2.x) --")
    try:
        from isaaclab.assets.articulation import ArticulationData

        wanted = ("pos_w", "quat_w", "joint_pos", "joint_vel", "body_", "root_", "default_")
        names = sorted(
            n for n in dir(ArticulationData) if not n.startswith("_") and any(w in n for w in wanted)
        )
        pprint.pprint(names, width=100, compact=True)
    except ImportError as exc:
        print(f"  ArticulationData import failed: {exc}")

    print()
    print("-- DirectRLEnv step/counter attributes --")
    try:
        from isaaclab.envs import DirectRLEnv as _Env

        names = sorted(
            n for n in dir(_Env) if not n.startswith("_") and ("step" in n or "count" in n or "episode" in n)
        )
        print(f"  {names}")
    except ImportError as exc:
        print(f"  DirectRLEnv import failed: {exc}")

    print()
    print("-- UR10e_CFG, verbatim (source of actuator gains and joint limits) --")
    try:
        from isaaclab_assets.robots.universal_robots import UR10e_CFG

        pprint.pprint(UR10e_CFG, width=110)
    except ImportError as exc:
        print(f"  UR10e_CFG import failed: {exc}")


def main() -> None:
    # Staleness detector. This script is copied to the training machine by hand, and a
    # stale copy has twice looked like a script that silently skips a section. If the
    # banner below is absent from a run, the copy over there predates that section.
    print("[probe_assets] sections available: 1 shipped cfgs, 2 robot USDs, 3 fixture stage, 4 API surface")
    if args_cli.only_api:
        print("[probe_assets] --only-api: skipping sections 1-3")
        try:
            report_api_surface()
        except Exception as exc:  # noqa: BLE001
            print(f"API surface probe failed: {type(exc).__name__}: {exc}")
        return

    try:
        candidates = report_shipped_cfgs()
    except Exception as exc:  # noqa: BLE001
        print(f"shipped-config probe failed: {type(exc).__name__}: {exc}")
        candidates = []

    # Candidates the plan expects to exist; duplicates are dropped below.
    candidates += [
        f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
        f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e_instanceable.usd",
        f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10/ur10_instanceable.usd",
    ]
    if args_cli.robot_usd:
        candidates += args_cli.robot_usd

    seen: set[str] = set()
    unique = [c for c in candidates if not (c in seen or seen.add(c))]

    rule("2. Robot USD candidates")
    for usd_path in unique:
        try:
            probe_robot_stage(usd_path)
        except Exception as exc:  # noqa: BLE001 - a probe must never abort on one candidate
            print(f"  probe failed: {type(exc).__name__}: {exc}")

    if args_cli.usd is not None:
        try:
            probe_fixture_stage(args_cli.usd)
        except Exception as exc:  # noqa: BLE001
            rule("3. Fixture stage")
            print(f"  probe failed: {type(exc).__name__}: {exc}")
    else:
        rule("3. Fixture stage")
        print("skipped — pass --usd <path\\to\\Tisch.usd> to probe it")

    try:
        report_api_surface()
    except Exception as exc:  # noqa: BLE001
        print(f"API surface probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
    simulation_app.close()
