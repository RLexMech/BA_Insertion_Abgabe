# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline check of the WIRING between the env and the pure math (M2.3/M2.4)."""

from __future__ import annotations

import ast
import concurrent.futures
import importlib.util
import math
import pathlib
import re
import sys
import types

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "source" / "insertion" / "insertion" / "tasks" / "direct" / "insertion"
MATH_SRC = PKG / "insertion_math.py"
OBS_NOISE_SRC = PKG / "obs_noise.py"
ENV_SRC = PKG / "insertion_env.py"
CFG_SRC = PKG / "insertion_env_cfg.py"
TASKS_SRC = PKG / "insertion_tasks_cfg.py"
TRAIN_SRC = REPO / "scripts" / "rsl_rl" / "train.py"
ZERO_SRC = REPO / "scripts" / "zero_agent.py"
RANDOM_SRC = REPO / "scripts" / "random_agent.py"
SCRIPTED_SRC = REPO / "scripts" / "scripted_insert.py"
TILT_SRC = REPO / "scripts" / "tilt_insert.py"
SEAT_SRC = REPO / "scripts" / "seat_probe.py"
SEATED_SRC = REPO / "scripts" / "check_seated_success.py"
RECOVERY_SRC = REPO / "scripts" / "tilt_recovery_probe.py"
PPO_SRC = PKG / "agents" / "rsl_rl_ppo_cfg.py"
INIT_SRC = PKG / "__init__.py"
COMPARE_SRC = REPO / "scripts" / "compare_runs.py"
AUTODR_SRC = PKG / "autodr.py"
PLAY_SRC = REPO / "scripts" / "rsl_rl" / "play.py"
RLG_PLAY_SRC = REPO / "scripts" / "rl_games" / "play.py"
EVAL_MANIFEST_SRC = REPO / "scripts" / "tools" / "eval_manifest.py"

MEASURED_HOME_TIP_M = 0.415

XY_CLAMP_AIR_MAX_M = 0.010

YAW_CLAMP_AIR_MAX_RAD = math.radians(0.1)

FORBIDDEN_TARGET_TRUTH: tuple[str, ...] = (
    "_entrance_pos",
    "_entrance_base",
    "_tilt_rot",
    "_tilt_rot_t",
    "_fixture_quat",
    "_fixture_yaw",
    "_peg_geometry",
)

ENV_LOCAL_TO_SLICE = {
    "joint_pos": "joint_pos",
    "joint_vel_fd": "joint_vel",
    "tip_rel": "tip_rel",
    "ee_quat_w": "ee_quat",
    "yaw_cs": "yaw_cos_sin",
    "pocket_quat": "pocket_quat",
    "force": "force",
    "torque": "torque",
    "prev_action": "prev_action",
}

_SHIM = REPO / "scripts" / "tools" / "torch_shim.py"
_spec = importlib.util.spec_from_file_location("torch_shim", _SHIM)
_shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shim)
torch, _IS_REAL_TORCH = _shim.load()
sys.modules.setdefault("torch", torch)


def _read(path: pathlib.Path, mutations: tuple[tuple[str, str], ...]) -> str:
    """Source text, with each mutation required to match EXACTLY once."""
    src = path.read_text(encoding="utf-8")
    for old, new in mutations:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"mutation {old!r} matches {n} times in {path.name}, expected 1")
        src = src.replace(old, new)
    return src


_TREES: dict = {}
_WALKS: dict = {}


def _parse(src_path: pathlib.Path, muts: tuple[tuple[str, str], ...]) -> ast.AST:
    """``ast.parse(_read(path, mutations))``, cached per (path, mutations)."""
    key = (str(src_path), tuple(muts))
    tree = _TREES.get(key)
    if tree is None:
        tree = ast.parse(_read(src_path, muts), str(src_path))
        _TREES[key] = tree
    return tree


def _walk(node: ast.AST):
    """``ast.walk`` with its result list cached per node object."""
    hit = _WALKS.get(id(node))
    if hit is None or hit[0] is not node:
        hit = (node, list(ast.walk(node)))
        _WALKS[id(node)] = hit
    return iter(hit[1])


_UNPARSED: dict = {}


def _unparse(node: ast.AST) -> str:
    """``ast.unparse`` cached per node object -- the other half of a build."""
    hit = _UNPARSED.get(id(node))
    if hit is None or hit[0] is not node:
        hit = (node, ast.unparse(node))
        _UNPARSED[id(node)] = hit
    return hit[1]


def _autodr_reach(autodr_mut: tuple[tuple[str, str], ...] = ()) -> dict:
    """``AutoDR.bounds_max()`` -- the widest band a run could EVER draw from."""
    _spec = importlib.util.spec_from_file_location(
        "_check_autodr_for_wiring", REPO / "scripts" / "check_autodr.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ns = _mod.load_autodr_src(_read(AUTODR_SRC, autodr_mut))
    dims = ns["DR_DIMS"]
    centres = {d.name: (d.lo_max if d.one_sided else 0.5 * (d.lo_max + d.hi_max))
               for d in dims}
    return ns["AutoDR"](ns["bind_centres"](centres)).bounds_max()


def load_obs_slices(mutations: tuple[tuple[str, str], ...] = ()) -> tuple[dict, int]:
    """``OBS_SLICES`` and ``OBS_DIM``, by executing the whole math module."""
    src = _read(MATH_SRC, mutations)
    ns: dict = {"torch": torch, "math": math, "__name__": "insertion_math"}
    exec(compile(src, str(MATH_SRC), "exec"), ns)  # noqa: S102 - the point of the file
    return ns["OBS_SLICES"], ns["OBS_DIM"], ns


def _func(tree: ast.AST, name: str, cls: str | None = None) -> ast.FunctionDef:
    """The named function, optionally inside the named class. Hard stop if absent."""
    scope: ast.AST = tree
    if cls is not None:
        for node in _walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                scope = node
                break
        else:
            raise SystemExit(f"class {cls} not found")
    for node in _walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"function {name} not found in {cls or 'module'}")


def _cat_operand_lists(fn: ast.FunctionDef) -> list[list[str]]:
    """EVERY ``torch.cat((...))`` tuple in ``fn``, in source order (D-188: the observation is built by one of two cats, chosen by the mode)."""
    out: list[list[str]] = []
    for node in _walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cat"
            and node.args
            and isinstance(node.args[0], ast.Tuple)
        ):
            out.append([e.id for e in node.args[0].elts if isinstance(e, ast.Name)])
    return out


def _cat_operands(fn: ast.FunctionDef) -> list[str]:
    """Names passed to the ``torch.cat`` tuple inside ``fn``, in order."""
    for node in _walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cat"
            and node.args
            and isinstance(node.args[0], ast.Tuple)
        ):
            return [e.id for e in node.args[0].elts if isinstance(e, ast.Name)]
    return []


def _identifiers(node: ast.AST) -> set[str]:
    """Every name and attribute used in the subtree -- docstrings excluded."""
    out: set[str] = set()
    for n in _walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def _ema_update_targets(fn: ast.FunctionDef) -> set[str]:
    """The FIRST argument of every ``ema_update(...)`` call in ``fn``, as source text -- i.e."""
    out: set[str] = set()
    for node in _walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "ema_update" and node.args):
            out.add(_unparse(node.args[0]))
    return out


def _class_attr_int(tree: ast.AST, cls: str, attr: str) -> int | None:
    """An ``attr = <int>`` assignment directly on the named class body."""
    for node in _walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for stmt in node.body:
                targets = []
                if isinstance(stmt, ast.Assign):
                    targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    targets = [stmt.target.id]
                if attr in targets and isinstance(stmt.value, ast.Constant):
                    return stmt.value.value
    return None


def _module_const(tree: ast.AST, name: str):
    """A module-level constant, by executing ONLY its own assignment node."""
    for node in _walk(tree):
        target = None
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            target = names[0] if names else None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        if target == name:
            ns: dict = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<cfg>", "exec"), ns)  # noqa: S102
            return ns.get(name)
    return None


def _module_chain(tree: ast.AST, names: tuple[str, ...], seed: dict | None = None) -> dict:
    """Execute the named module-level assignments, IN THE GIVEN ORDER, together."""
    wanted = set(names)
    found: dict[str, ast.stmt] = {}
    for node in _walk(tree):
        target = None
        if isinstance(node, ast.Assign):
            ids = [t.id for t in node.targets if isinstance(t, ast.Name)]
            target = ids[0] if ids else None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        if target in wanted and target not in found:
            found[target] = node
    ns: dict = dict(seed or {})
    for name in names:
        node = found.get(name)
        if node is None:
            continue
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<tasks>", "exec"), ns)  # noqa: S102
        except Exception:  # noqa: BLE001 - an unresolvable name IS the finding
            continue
    return {k: v for k, v in ns.items() if k in wanted}


def _module_expr(tree: ast.AST, name: str) -> str | None:
    """The right-hand side of a module-level ``name = ...``, as source text."""
    for node in _walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return _unparse(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return _unparse(node.value)
    return None


def _fold_number(node: ast.AST):
    """A literal arithmetic expression's value, or ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _fold_number(node.operand)
        return None if inner is None else (inner if isinstance(node.op, ast.UAdd) else -inner)
    if isinstance(node, ast.BinOp):
        left, right = _fold_number(node.left), _fold_number(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Div):
            return None if right == 0 else left / right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
    return None


def _fold_radians(node: ast.AST | None):
    """``math.radians(<number>)`` or a plain literal number, in radians; else ``None``."""
    if (isinstance(node, ast.Call) and _unparse(node.func) == "math.radians"
            and len(node.args) == 1 and not node.keywords):
        deg = _fold_number(node.args[0])
        return None if deg is None else math.radians(deg)
    return None if node is None else _fold_number(node)


def _pocket_obs_yaw_rad() -> float:
    """D-194's pocket-yaw belief error of the run, in radians."""
    _spec = importlib.util.spec_from_file_location(
        "_check_eval_conditions_for_wiring", REPO / "scripts" / "check_eval_conditions.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return float(_mod.POCKET_OBS_YAW_RAD)


def _class_attr_expr(tree: ast.AST, cls: str, attr: str) -> ast.AST | None:
    """The right-hand SIDE of ``attr = ...`` on the named class body."""
    for node in _walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for stmt in node.body:
                targets = []
                if isinstance(stmt, ast.Assign):
                    targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    targets = [stmt.target.id]
                if attr in targets:
                    return stmt.value
    return None


def _class_attr_kwargs(tree: ast.AST, cls: str, attr: str) -> dict[str, str]:
    """``attr = Something(k=<expr>, ...)`` on the class body, as ``{k: source}``."""
    node = _class_attr_expr(tree, cls, attr)
    if not isinstance(node, ast.Call):
        return {}
    return {kw.arg: _unparse(kw.value) for kw in node.keywords if kw.arg}


def _literal(node: ast.AST | None):
    """A literal value, or ``None`` for anything this cannot read."""
    if isinstance(node, ast.Constant):
        return node.value
    if (isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return None


def _class_attr_const(tree: ast.AST, cls: str, attr: str):
    """An ``attr = <literal>`` assignment on the named class body, or ``None``."""
    for node in _walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for stmt in node.body:
                targets = []
                if isinstance(stmt, ast.Assign):
                    targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    targets = [stmt.target.id]
                if attr in targets:
                    value = _literal(stmt.value)
                    if value is not None:
                        return value
    return None


def _class_bases(tree: ast.AST, cls: str) -> list[str] | None:
    """The named class's base classes, unparsed; ``None`` when it is absent."""
    for node in _walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return [_unparse(b) for b in node.bases]
    return None


def _method_self_assigns(tree: ast.AST, cls: str, method: str) -> dict[str, ast.AST]:
    """``self.<x> = <expr>`` inside ``cls.method`` as ``{\"self.<x>\": expr}``; empty when the class or the method is absent (a renamed class must read as a red check, not stop the whole run)."""
    for node in _walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == method:
                    return {
                        _unparse(s.targets[0]): s.value
                        for s in ast.walk(fn)
                        if isinstance(s, ast.Assign) and len(s.targets) == 1
                        and _unparse(s.targets[0]).startswith("self.")
                    }
    return {}


def _gym_register_kwargs(tree: ast.AST) -> dict[str, str]:
    """The ``kwargs={...}`` of the ONE ``gym.register`` call, ``{key: source}``."""
    calls = [n for n in _walk(tree) if isinstance(n, ast.Call) and _unparse(n.func) == "gym.register"]
    if len(calls) != 1:
        return {}
    for kw in calls[0].keywords:
        if kw.arg == "kwargs" and isinstance(kw.value, ast.Dict):
            return {k.value: _unparse(v) for k, v in zip(kw.value.keys, kw.value.values)
                    if isinstance(k, ast.Constant)}
    return {}


def _string_constants(node: ast.AST) -> list[str]:
    """Every string constant in the subtree, docstrings included."""
    return [n.value for n in _walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _guarded_call(fn: ast.FunctionDef, call: str, flag: str) -> tuple[bool, bool]:
    """``(the call happens, it happens under an if that tests ``flag``)``."""
    called = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == call
        for n in _walk(fn)
    )
    guarded = False
    for node in _walk(fn):
        if isinstance(node, ast.If) and flag in _identifiers(node.test):
            body_calls = {
                n.func.id
                for stmt in node.body
                for n in _walk(stmt)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            if call in body_calls:
                guarded = True
    return called, guarded


def _marker_under_flag(fn: ast.FunctionDef, marker: str, flag: str) -> tuple[bool, bool]:
    """``(the marker string is in the function, it is ONLY under ``flag``)``."""
    present = marker in _string_constants(fn)
    if not present:
        return False, False
    under = any(
        isinstance(node, ast.If)
        and flag in _identifiers(node.test)
        and marker in [c for stmt in node.body for c in _string_constants(stmt)]
        for node in _walk(fn)
    )
    return present, under


def _mentions(node: ast.AST, flag: str) -> bool:
    """The flag appears as an identifier OR as a string constant in the subtree."""
    if flag in _identifiers(node):
        return True
    return any(
        isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == flag
        for n in _walk(node)
    )


def _refuses_on_flag(tree: ast.AST, flag: str) -> bool:
    """Some ``if`` mentioning ``flag`` whose body raises."""
    for node in _walk(tree):
        if isinstance(node, ast.If) and _mentions(node.test, flag):
            if any(isinstance(n, ast.Raise) for stmt in node.body for n in _walk(stmt)):
                return True
    return False


def _cli_flags(tree: ast.AST) -> set[str]:
    """Every option string handed to an ``add_argument`` call in the module."""
    flags: set[str] = set()
    for node in _walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        flags.add(a.value)
    return flags


def _cli_default(tree: ast.AST, flag: str):
    """The ``default=`` of the ``add_argument`` call that declares ``flag``."""
    for node in _walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if flag not in names:
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                return kw.value.value
    return None


def _assign_precedes_call(fn: ast.FunctionDef, attr: str, call_attr: str) -> bool:
    """Every ``<x>.<attr> = ...`` in ``fn`` sits ABOVE every ``<y>.<call_attr>(...)``."""
    assigns = [
        n.lineno
        for n in _walk(fn)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute) and t.attr == attr
    ]
    calls = [
        n.lineno
        for n in _walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == call_attr
    ]
    return bool(assigns) and bool(calls) and max(assigns) < min(calls)


def _gym_make_line(tree: ast.AST) -> int | None:
    """The line of this script's FIRST ``gym.make(...)`` call, or ``None``."""
    lines = [
        n.lineno
        for n in _walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "make"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "gym"
    ]
    return min(lines) if lines else None


def _sets_attr_none(tree: ast.AST, attr: str) -> bool:
    """Some ``<something>.<attr> = None`` BEFORE this script's ``gym.make``."""
    _make = _gym_make_line(tree)
    if _make is None:
        return False
    for node in _walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if node.value.value is None and node.lineno < _make:
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr == attr:
                        return True
    return False


def _sets_attr_zero(tree: ast.AST, attr: str) -> bool:
    """Some ``<something>.<attr> = 0.0`` BEFORE this script's ``gym.make``."""
    _make = _gym_make_line(tree)
    if _make is None:
        return False
    for node in _walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if node.value.value == 0.0 and node.value.value is not False:
                if node.lineno >= _make:
                    continue
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr == attr:
                        return True
    return False


def _sets_flag_false(tree: ast.AST, flag: str) -> bool:
    """Some ``<something>.<flag> = False`` at any depth."""
    for node in _walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if node.value.value is False:
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr == flag:
                        return True
    return False


FLAG = "rl_terms_enabled"

EPISODE_LOG_COLUMNS_AGREED = (
    "run", "seed", "iteration", "env_id", "episode", "outcome", "steps",
    "ip_max_mm", "ip_steps_ge_t1", "force_max_filtered_n", "force_max_raw_n",
    "bounds_version", "dr_phase",
)

APPLIED_LOG_COLUMNS_AGREED = (
    "row_id", "lat_x_m", "lat_y_m", "start_height_m", "yaw_rad",
    "pocket_qw", "pocket_qx", "pocket_qy", "pocket_qz",
    "pocket_dx_m", "pocket_dy_m", "pocket_dz_m", "grasp_obs_x_m", "pocket_obs_yaw_rad",
    "friction_cmd", "friction_robot_min", "friction_robot_max",
    "friction_fixture_min", "friction_fixture_max",
)
APPLIED_LOG_SOURCES_AGREED = (
    "torch.arange(n, device=self.device, dtype=torch.float64)",
    "self._start_lat_off[:, 0].to(torch.float64)",
    "self._start_lat_off[:, 1].to(torch.float64)",
    "self._start_height.to(torch.float64)",
    "self._fixture_yaw.to(torch.float64)",
    "self._fixture_quat[:, 0].to(torch.float64)",
    "self._fixture_quat[:, 1].to(torch.float64)",
    "self._fixture_quat[:, 2].to(torch.float64)",
    "self._fixture_quat[:, 3].to(torch.float64)",
    "_pocket_d[:, 0].to(torch.float64)",
    "_pocket_d[:, 1].to(torch.float64)",
    "_pocket_d[:, 2].to(torch.float64)",
    "self._grasp_obs_off[:, 0].to(torch.float64)",
    "self._pocket_obs_yaw.to(torch.float64)",
    "self._friction_applied.to(torch.float64)",
    "_fric[0].min(dim=1).values",
    "_fric[0].max(dim=1).values",
    "_fric[1].min(dim=1).values",
    "_fric[1].max(dim=1).values",
)


def build_checks(
    math_mut: tuple[tuple[str, str], ...] = (),
    env_mut: tuple[tuple[str, str], ...] = (),
    cfg_mut: tuple[tuple[str, str], ...] = (),
    tasks_mut: tuple[tuple[str, str], ...] = (),
    train_mut: tuple[tuple[str, str], ...] = (),
    agent_mut: tuple[tuple[str, str], ...] = (),
    scripted_mut: tuple[tuple[str, str], ...] = (),
    tilt_mut: tuple[tuple[str, str], ...] = (),
    seat_mut: tuple[tuple[str, str], ...] = (),
    seated_mut: tuple[tuple[str, str], ...] = (),
    ppo_mut: tuple[tuple[str, str], ...] = (),
    compare_mut: tuple[tuple[str, str], ...] = (),
    autodr_mut: tuple[tuple[str, str], ...] = (),
    obs_noise_mut: tuple[tuple[str, str], ...] = (),
    recovery_mut: tuple[tuple[str, str], ...] = (),
    play_mut: tuple[tuple[str, str], ...] = (),
    init_mut: tuple[tuple[str, str], ...] = (),
    rlg_play_mut: tuple[tuple[str, str], ...] = (),
) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    slices, obs_dim, math_ns = load_obs_slices(math_mut)
    env_tree = _parse(ENV_SRC, env_mut)
    cfg_tree = _parse(CFG_SRC, cfg_mut)
    tasks_tree = _parse(TASKS_SRC, tasks_mut)
    ppo_tree = _parse(PPO_SRC, ppo_mut)
    compare_tree = _parse(COMPARE_SRC, compare_mut)
    train_tree = _parse(TRAIN_SRC, train_mut)
    zero_tree = _parse(ZERO_SRC, agent_mut)
    tilt_tree = _parse(TILT_SRC, tilt_mut)
    random_tree = _parse(RANDOM_SRC, agent_mut)
    noise_tree = _parse(OBS_NOISE_SRC, obs_noise_mut)

    names = list(slices)
    has_force = "force" in slices
    check("the force block is the LAST block in the table",
          has_force and names[-1] == "force" and slices["force"][1] == obs_dim,
          f"{names[-1] if names else '<empty>'} .. {obs_dim}")
    try:
        wslices = math_ns["obs_slices"]("wrench")
        wdim = math_ns["obs_dim"]("wrench")
    except Exception:  # noqa: BLE001 - a broken table is a finding, not a crash
        wslices, wdim = {}, None
    wnames = list(wslices)
    check("the wrench layout is the force layout plus torque appended LAST",
          bool(wslices) and wnames[:-1] == names and wnames[-1] == "torque"
          and all(wslices[k] == v for k, v in slices.items())
          and wslices["torque"] == (obs_dim, obs_dim + 3) and wdim == obs_dim + 3,
          f"{wnames} .. {wdim}")
    try:
        pslices = math_ns["obs_slices"]("force_prevact")
        pdim = math_ns["obs_dim"]("force_prevact")
    except Exception:  # noqa: BLE001 - a broken table is a finding, not a crash
        pslices, pdim = {}, None
    pnames = list(pslices)
    check("the force_prevact layout is the force layout plus prev_action appended LAST",
          bool(pslices) and pnames[:-1] == names and pnames[-1] == "prev_action"
          and all(pslices[k] == v for k, v in slices.items())
          and pslices["prev_action"] == (obs_dim, obs_dim + 6) and pdim == obs_dim + 6,
          f"{pnames} .. {pdim}")

    _declared_expr = _class_attr_expr(cfg_tree, "InsertionEnvCfg", "observation_space")
    _declared_src = _unparse(_declared_expr) if _declared_expr is not None else None
    declared = _class_attr_int(cfg_tree, "InsertionEnvCfg", "observation_space")
    check("observation_space is the full OBS_SLICES width",
          _declared_src == "insertion_math.OBS_DIM" or declared == obs_dim,
          f"cfg {_declared_src!r}, OBS_DIM {obs_dim}")
    _resolve_obs = _func(cfg_tree, "resolve_obs_layout")
    _resolve_obs_src = _unparse(_resolve_obs)
    check("resolve_obs_layout writes observation_space from insertion_math.obs_dim(mode)",
          "width = insertion_math.obs_dim(mode)" in _resolve_obs_src
          and "cfg.observation_space = width" in _resolve_obs_src, "")
    _init_fn_obs = _func(env_tree, "__init__", "InsertionEnv")
    check("the observation layout is resolved BEFORE the base class reads the cfg",
          _assign_precedes_call(_init_fn_obs, "_obs_mode", "__init__"))

    obs_fn = _func(env_tree, "_get_observations", "InsertionEnv")
    _asm = [n for n in _walk(env_tree) if isinstance(n, ast.FunctionDef) and n.name == "_assemble_obs"]
    _asm_fn = _asm[0] if len(_asm) == 1 else ast.parse("def _absent():\n    pass\n").body[0]
    _cats = [[ENV_LOCAL_TO_SLICE.get(n, n) for n in c] for c in _cat_operand_lists(_asm_fn)]
    wanted = names
    check("the observation is assembled in the OBS_SLICES order",
          wanted in _cats, f"cats {_cats} vs table {wanted}")
    check("the wrench observation is the force cat plus torque",
          (wanted + ["torque"]) in _cats, f"cats {_cats}")
    check("the force_prevact observation is the force cat plus prev_action",
          (wanted + ["prev_action"]) in _cats, f"cats {_cats}")
    check("the observation has exactly three cats, one per mode",
          len(_cats) == 3, f"{len(_cats)} cats")
    check("prev_action is bound from self._ema_actions, the applied (smoothed) action",
          "prev_action = self._ema_actions" in _unparse(_asm_fn), "")

    _rets = [n for n in _walk(obs_fn) if isinstance(n, ast.Return)]
    check("_get_observations returns exactly the policy and the critic group",
          len(_rets) == 1 and _unparse(_rets[0].value) == "{'policy': obs, 'critic': critic}",
          str([_unparse(r.value) for r in _rets]))
    _asm_calls = {
        _unparse(a.targets[0]): [_unparse(x) for x in a.value.args]
        for a in _walk(obs_fn) if isinstance(a, ast.Assign) and len(a.targets) == 1
        and isinstance(a.value, ast.Call) and _unparse(a.value.func) == "self._assemble_obs"
    }
    _pol_args = ["joint_pos", "joint_vel_fd", "tip_rel", "ee_quat_w", "yaw_cs",
                 "pocket_quat", "force", "torque"]
    _true_of = {"tip_rel": "tip_rel_true", "yaw_cs": "yaw_cs_true", "pocket_quat": "pocket_quat_true"}
    _true_assign = [_unparse(a) for a in _walk(obs_fn) if isinstance(a, ast.Assign)
                    and isinstance(a.value, ast.Call)
                    and _unparse(a.value.func) == "self._true_pocket_blocks"]
    check("the critic group differs from the policy group only in the three pocket blocks",
          _asm_calls == {"obs": _pol_args, "critic": [_true_of.get(a, a) for a in _pol_args]}
          and _true_assign == ["tip_rel_true, yaw_cs_true, pocket_quat_true = self._true_pocket_blocks()"],
          f"{_asm_calls} / {_true_assign}")
    _tpb = [n for n in _walk(env_tree) if isinstance(n, ast.FunctionDef) and n.name == "_true_pocket_blocks"]
    _tpb_bpf = [a for a in (_walk(_tpb[0]) if len(_tpb) == 1 else []) if isinstance(a, ast.Assign)
                and isinstance(a.value, ast.Call)
                and _unparse(a.value.func) == "insertion_math.believed_pocket_frame"]
    _tpb_tgt = ([_unparse(e) for e in _tpb_bpf[0].targets[0].elts]
                if len(_tpb_bpf) == 1 and isinstance(_tpb_bpf[0].targets[0], ast.Tuple) else [])
    check("the critic's pocket frame is believed_pocket_frame at ZERO error",
          len(_tpb_bpf) == 1 and [_unparse(x) for x in _tpb_bpf[0].value.args]
          == ["self._fixture_quat", "torch.zeros(self.num_envs, device=self.device)"]
          and len(_tpb_tgt) == 2,
          str([_unparse(a) for a in _tpb_bpf]))
    _tpb_ptp = [n for n in (_walk(_tpb[0]) if len(_tpb) == 1 else []) if isinstance(n, ast.Call)
                and _unparse(n.func) == "insertion_math.part_tip_pose"]
    check("the critic's tip uses the NOMINAL tip offset in that true frame",
          len(_tpb_ptp) == 1 and len(_tpb_ptp[0].args) == 6 and bool(_tpb_tgt)
          and _unparse(_tpb_ptp[0].args[2]) == "self._tip_offset_local"
          and _unparse(_tpb_ptp[0].args[5]) == _tpb_tgt[0],
          str([_unparse(n) for n in _tpb_ptp]))
    check("resolve_obs_layout writes state_space with the same width",
          "cfg.state_space = width" in _resolve_obs_src, "")

    ids = _identifiers(obs_fn)
    check("the wrench is read through body_incoming_joint_wrench_b",
          "body_incoming_joint_wrench_b" in ids,
          f"{sorted(ids & {'body_incoming_joint_wrench_b', 'body_link_state_w'})}")
    _ema_targets = _ema_update_targets(obs_fn)
    check("the force block goes through insertion_math.ema_update",
          "self._force_smooth" in _ema_targets, str(sorted(_ema_targets)))
    _obs_unparsed = _unparse(obs_fn)
    check("all six wrench columns are read (0:6), torque = columns 3:6",
          "self._force_body_idx, 0:6]" in _obs_unparsed
          and "torque_raw = wrench_raw[:, 3:6]" in _obs_unparsed, "")
    check("the torque block goes through insertion_math.gravity_tare_torque with the COM lever",
          "gravity_tare_torque" in ids and "body_com_pos_w" in ids, "")
    check("the torque block goes through insertion_math.ema_update",
          "self._torque_smooth" in _ema_targets, str(sorted(_ema_targets)))
    reset_fn = _func(env_tree, "_reset_idx", "InsertionEnv")
    reset_src = ast.dump(reset_fn)
    check("the force EMA buffer is cleared in _reset_idx",
          "_force_smooth" in reset_src, "")
    check("the torque EMA buffer is cleared in _reset_idx",
          "_torque_smooth" in reset_src, "")
    obs_src = _unparse(obs_fn)
    check("the reset flag _wrench_fresh is raised in _reset_idx",
          "_wrench_fresh" in reset_src, "")
    check("the wrench EMA skips envs _reset_idx just touched (keep = ~_wrench_fresh)",
          "keep = ~self._wrench_fresh" in obs_src
          and "self._wrench_fresh.fill_(False)" in obs_src
          and "self._wrench_valid.copy_(keep)" in obs_src, "")
    _once = [n for n in _walk(obs_fn)
             if isinstance(n, ast.If) and "common_step_counter" in _identifiers(n.test)]
    check("the wrench EMA advances once per common_step_counter value",
          len(_once) == 1 and "_wrench_step" in _identifiers(_once[0].test) if _once else False,
          f"{len(_once)} guarded if-block(s)")
    check("the C4 yaw encoding is gone from the observation",
          not {"cos4phi", "sin4phi"} & ids,
          f"{sorted({'cos4phi', 'sin4phi'} & ids)}")
    check("the observation comes from insertion_math, not from local pose math",
          {"part_tip_pose", "yaw_cos_sin", "believed_pocket_frame"} <= ids,
          f"missing {sorted({'part_tip_pose', 'yaw_cos_sin', 'believed_pocket_frame'} - ids)}")

    _bias_model_cfgs = [
        n for n in _walk(noise_tree)
        if isinstance(n, ast.Call)
        and _unparse(n.func).endswith("NoiseModelWithAdditiveBiasCfg")
    ]

    def _inner_cfg(call: ast.Call, arg: str, cls: str) -> dict | None:
        """The keywords of the ``cls`` noise cfg bound to ``arg``."""
        for kw in call.keywords:
            if (kw.arg == arg and isinstance(kw.value, ast.Call)
                    and _unparse(kw.value.func).endswith(cls)):
                return {k.arg: _unparse(k.value) for k in kw.value.keywords}
        return None

    _one_model = len(_bias_model_cfgs) == 1
    _bias_kw = (_inner_cfg(_bias_model_cfgs[0], "bias_noise_cfg", "UniformNoiseCfg")
                if _one_model else None)
    _step_kw = (_inner_cfg(_bias_model_cfgs[0], "noise_cfg", "GaussianNoiseCfg")
                if _one_model else None)
    check("the per-episode bias is drawn ABSOLUTE, not added onto the last one",
          _bias_kw is not None and _step_kw is not None
          and _bias_kw.get("operation") == "'abs'"
          and _bias_kw.get("n_max") is not None
          and _step_kw.get("operation") != "'abs'"
          and _step_kw.get("std") is not None,
          f"{len(_bias_model_cfgs)} model cfg(s); bias {_bias_kw}, per-step {_step_kw}")
    check("the pocket bias is uniform in [-bias_std, +bias_std]",
          _bias_kw is not None
          and _bias_kw.get("n_min") == "-bias_std"
          and _bias_kw.get("n_max") == "bias_std",
          f"bias {_bias_kw}")
    _bound = _class_attr_expr(noise_tree, "InsertionObsNoiseCfg", "noise_cfg")
    _bound_src = _unparse(_bound) if _bound is not None else None
    check("the inherited MISSING noise_cfg is bound, so cfg.validate() passes",
          _bound is not None and _bound_src != "MISSING",
          f"noise_cfg bound to {_bound_src!r} (None = not bound at all)")
    _noise_ids = _identifiers(noise_tree)
    check("the bias buffer and its component count are sized up front",
          {"_bias", "_num_components", "noise_masks", "obs_dim"} <= _noise_ids,
          f"missing {sorted({'_bias', '_num_components', 'noise_masks', 'obs_dim'} - _noise_ids)}")
    _nm_calls = [n for n in _walk(noise_tree) if isinstance(n, ast.Call)
                 and _unparse(n.func).endswith("noise_masks")]
    _nm_kw = {k.arg: _unparse(k.value) for k in _nm_calls[0].keywords} if len(_nm_calls) == 1 else {}
    check("noise_masks receives torque_std_nm and mode from the noise cfg, by keyword",
          _nm_kw.get("torque_std_nm") == "float(noise_model_cfg.torque_std_nm)"
          and _nm_kw.get("mode") == "str(noise_model_cfg.obs_mode)",
          f"{len(_nm_calls)} call(s), keywords {_nm_kw}")
    _validate_src = _unparse(_func(cfg_tree, "validate_rl_config"))
    check("the torque sigma is in the cfg's negative-scatter sweep",
          "'torque_obs_noise_std_nm'" in _validate_src, "")
    check("the torque sigma is a labelled placeholder",
          "torque_obs_noise_std_nm" in (_module_const(cfg_tree, "RL_PLACEHOLDERS") or {}), "")
    _resolver = _func(noise_tree, "resolve_obs_noise_model")
    _res_assigns = {
        t.id: _unparse(a.value)
        for a in _walk(_resolver) if isinstance(a, ast.Assign)
        for t in a.targets if isinstance(t, ast.Name)
    }
    _off_ifs = [
        n for n in _walk(_resolver) if isinstance(n, ast.If)
        and any(isinstance(r, ast.Return) and isinstance(r.value, ast.Constant)
                and r.value.value is None for r in _walk(n))
    ]
    _off_test = _unparse(_off_ifs[0].test) if len(_off_ifs) == 1 else None
    check("both sigmas at zero switch the observation noise model off entirely",
          _off_test == "pocket == 0.0 and force == 0.0 and (torque == 0.0)"
          and _res_assigns.get("pocket") == "float(cfg.obs_noise_pocket_pos_std_m)"
          and _res_assigns.get("force") == "float(cfg.force_obs_noise_std_n)"
          and _res_assigns.get("torque") == "float(cfg.torque_obs_noise_std_nm)",
          f"{len(_off_ifs)} off-branch(es), test {_off_test!r}, from {_res_assigns}")

    _GRASP = "_grasp_obs_off"
    _GRASP_ALLOWED = ("__init__", "_get_observations", "_print_startup_report",
                      "_reset_idx", "_write_applied_csv")
    _grasp_users = sorted(
        n.name for n in _walk(env_tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _GRASP in _identifiers(n)
    )
    check("the grasp offset reaches the OBSERVATION and no paying path",
          _grasp_users == sorted(_GRASP_ALLOWED),
          f"used in {_grasp_users}, allowed {sorted(_GRASP_ALLOWED)}")
    _grasp_writes = {
        name: [_unparse(a) for a in _walk(_func(env_tree, name, "InsertionEnv"))
               if isinstance(a, ast.Assign)
               and any(_GRASP in _unparse(t) for t in a.targets)]
        for name in ("_reset_idx", "_get_observations")
    }
    _grasp_src = _grasp_writes["_reset_idx"][0] if _grasp_writes["_reset_idx"] else ""
    check("the grasp offset is MAPPED from its own table column, never drawn beside the row",
          len(_grasp_writes["_reset_idx"]) == 1
          and "_reset_unit" in _grasp_src
          and "grasp_obs_x" in _grasp_src
          and "rand" not in _grasp_src
          and not _grasp_writes["_get_observations"],
          str(_grasp_writes))

    _PYAW = "_pocket_obs_yaw"
    _PYAW_ALLOWED = ("__init__", "_get_observations", "_print_startup_report",
                     "_reset_idx", "_write_applied_csv")
    _pyaw_users = sorted(
        n.name for n in _walk(env_tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _PYAW in _identifiers(n)
    )
    check("the pocket-yaw belief error reaches the OBSERVATION and no paying path",
          _pyaw_users == sorted(_PYAW_ALLOWED),
          f"used in {_pyaw_users}, allowed {sorted(_PYAW_ALLOWED)}")
    _pyaw_writes = {
        name: [_unparse(a) for a in _walk(_func(env_tree, name, "InsertionEnv"))
               if isinstance(a, ast.Assign)
               and any(_PYAW in _unparse(t) for t in a.targets)]
        for name in ("_reset_idx", "_get_observations")
    }
    _pyaw_src = _pyaw_writes["_reset_idx"][0] if _pyaw_writes["_reset_idx"] else ""
    check("the pocket-yaw belief error is MAPPED from its own table column, never drawn beside the row",
          len(_pyaw_writes["_reset_idx"]) == 1
          and "_reset_unit" in _pyaw_src
          and "'pocket_obs_yaw'" in _pyaw_src
          and "2.0 * " in _pyaw_src and "- 1.0" in _pyaw_src
          and "cfg.pocket_obs_yaw_rad" in _pyaw_src
          and "rand" not in _pyaw_src
          and not _pyaw_writes["_get_observations"],
          str(_pyaw_writes))
    _bpf = [a for a in _walk(obs_fn) if isinstance(a, ast.Assign)
            and isinstance(a.value, ast.Call)
            and _unparse(a.value.func) == "insertion_math.believed_pocket_frame"]
    _bpf_tgt = []
    if len(_bpf) == 1 and isinstance(_bpf[0].targets[0], ast.Tuple):
        _bpf_tgt = [_unparse(e) for e in _bpf[0].targets[0].elts]
    _bpf_args = [_unparse(x) for x in _bpf[0].value.args] if len(_bpf) == 1 else []
    _ptp = [n for n in _walk(obs_fn) if isinstance(n, ast.Call)
            and _unparse(n.func) == "insertion_math.part_tip_pose"]
    _ptp_frame = _unparse(_ptp[0].args[5]) if len(_ptp) == 1 and len(_ptp[0].args) == 6 else None
    check("tip_rel and the yaw come from the BELIEVED pocket frame",
          _bpf_args == ["self._fixture_quat", "self._pocket_obs_yaw"]
          and len(_bpf_tgt) == 2 and _ptp_frame == _bpf_tgt[0],
          f"believed_pocket_frame{tuple(_bpf_args)} -> {_bpf_tgt}, part_tip_pose frame {_ptp_frame}")
    _pq_assigns = [a for a in _walk(obs_fn) if isinstance(a, ast.Assign)
                   and "pocket_quat" in {_unparse(e) for t in a.targets
                                         for e in (t.elts if isinstance(t, ast.Tuple) else [t])}]
    check("the pocket_quat channel carries the BELIEVED pocket quaternion",
          len(_pq_assigns) == 1 and _bpf and _pq_assigns[0] is _bpf[0]
          and _bpf_tgt[1:] == ["pocket_quat"],
          f"{len(_pq_assigns)} assignment(s) of pocket_quat, believed targets {_bpf_tgt}")
    _tct = [n for n in _walk(obs_fn) if isinstance(n, ast.Call)
            and _unparse(n.func) == "insertion_math.tilt_cos_theta"]
    _prot = [_unparse(a.value) for a in _walk(obs_fn) if isinstance(a, ast.Assign)
             and any(_unparse(t) == "pocket_rot" for t in a.targets)]
    check("the tilt potential keeps the TRUE pocket frame",
          len(_tct) == 1 and _unparse(_tct[0].args[1]) == "pocket_rot"
          and _prot == ["self._tilt_rot if self._tilt_rot is not None "
                        "else self._pocket_rot_identity"],
          f"tilt_cos_theta frame {[_unparse(n.args[1]) for n in _tct]}, pocket_rot = {_prot}")
    check("the pocket-yaw belief error is in the cfg's negative-scatter sweep",
          "'pocket_obs_yaw_rad'" in _validate_src, "")
    _pyaw_default = _class_attr_const(cfg_tree, "InsertionEnvCfg", "pocket_obs_yaw_rad")
    check("the pocket-yaw belief error defaults to 0.0223402 rad",
          _pyaw_default == 0.0223402 and not isinstance(_pyaw_default, bool),
          f"default {_pyaw_default!r}")

    _pocket_sigma = _class_attr_const(cfg_tree, "InsertionEnvCfg",
                                      "obs_noise_pocket_pos_std_m")
    _seat_depth = _module_chain(
        tasks_tree, ("POCKET_FLOOR_Z", "POCKET_SEAT_DEPTH")
    ).get("POCKET_SEAT_DEPTH")
    check("the pocket bias half width is TacSL's 5 mm and stays under the seat depth",
          _pocket_sigma == 0.005
          and _seat_depth is not None and 0.0 < _pocket_sigma < float(_seat_depth),
          f"half width {_pocket_sigma} m vs POCKET_SEAT_DEPTH {_seat_depth} m")
    _force_sigma = _class_attr_const(cfg_tree, "InsertionEnvCfg",
                                     "force_obs_noise_std_n")
    _f_abort = _class_attr_const(cfg_tree, "InsertionEnvCfg", "force_abort_f_max_n")
    check("the force sigma is the UR5e datasheet's 3.5 N and stays under the abort force",
          _force_sigma == 3.5
          and _f_abort is not None and 0.0 < _force_sigma < float(_f_abort),
          f"sigma {_force_sigma} N vs force_abort_f_max_n {_f_abort} N")
    _grasp_off = _class_attr_const(cfg_tree, "InsertionEnvCfg", "grasp_obs_offset_x_m")
    _wall_x = _module_const(tasks_tree, "POCKET_WALL_X")
    check("the grasp belief error is Factory's 3 mm and stays inside the short-axis pocket wall",
          _grasp_off == 0.003
          and _wall_x is not None and 0.0 < _grasp_off < float(_wall_x),
          f"offset {_grasp_off} m vs POCKET_WALL_X {_wall_x} m")

    report_fn = _func(env_tree, "_print_startup_report", "InsertionEnv")
    marker_tests = [n.test for n in _walk(report_fn)
                    if isinstance(n, ast.IfExp)
                    and isinstance(n.body, ast.Constant)
                    and isinstance(n.body.value, str)
                    and ("NOT BUILT" in n.body.value or "MISSING" in n.body.value)]
    name_compared = [t for t in marker_tests
                     if any(isinstance(c, ast.Constant) and isinstance(c.value, str)
                            for c in _walk(t))]
    check("the layout report derives its NOT-BUILT marker from the built width",
          len(marker_tests) == 1 and not name_compared,
          f"{len(marker_tests)} marker(s), {len(name_compared)} compared to a string")
    check("no block is marked missing by name in the layout report",
          not any("scene stream S2" in c for c in _string_constants(report_fn)),
          "")

    init_fn = _func(env_tree, "__init__", "InsertionEnv")
    called, guarded = _guarded_call(init_fn, "validate_rl_config", "_" + FLAG)
    check("the env validates the rl config in __init__", called)
    check("the guard sits behind the rl_terms_enabled switch", guarded)

    dones_fn = _func(env_tree, "_get_dones", "InsertionEnv")
    rewards_fn = _func(env_tree, "_get_rewards", "InsertionEnv")
    dones_ids = _identifiers(dones_fn)
    rewards_ids = _identifiers(rewards_fn)

    check("the termination is insertion_math.compute_dones",
          "compute_dones" in dones_ids)
    check("the below_plate exit is gone from the termination",
          "below_plate" not in dones_ids)
    check("the reward is insertion_math.compute_reward_terms_insertion",
          "compute_reward_terms_insertion" in rewards_ids)
    proxy_reward_defs = [
        n for n in env_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_rewards"
    ]
    check("the proxy reward function is gone from the env",
          not proxy_reward_defs)
    check("the reward pays zero in a measurement env",
          "_rl_terms_enabled" in rewards_ids)
    check("the force abort is off in a measurement env",
          "_rl_terms_enabled" in dones_ids)
    check("the abort reads the EMA the policy sees",
          "_force_smooth" in dones_ids
          and "body_incoming_joint_wrench_b" not in dones_ids)
    sdf_guarded = any(
        isinstance(node, ast.If)
        and "_rl_terms_enabled" in _identifiers(node.test)
        and any("SdfDistanceQuery" in _identifiers(stmt) for stmt in node.body)
        for node in _walk(init_fn)
    )
    check("the SDF queries are built in __init__ behind the switch", sdf_guarded)
    sapu_kw = any(
        isinstance(n, ast.Call) and any(k.arg == "mesh_obj_path" for k in n.keywords)
        for n in _walk(init_fn)
    )
    check("the SAPU query asks the FIXTURE mesh",
          sapu_kw and "resolve_pocket_obj_path" in _identifiers(init_fn))
    check("the goal pose composes the seated tool rotation",
          "SEATED_TOOL_QUAT_LOCAL" in _identifiers(init_fn)
          and "_seat_quat_local" in dones_ids)
    stray_weights = []
    for node in _walk(cfg_tree):
        if isinstance(node, ast.ClassDef) and node.name == "InsertionEnvCfg":
            for stmt in node.body:
                names = []
                if isinstance(stmt, ast.Assign):
                    names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    names = [stmt.target.id]
                stray_weights += [n for n in names if n.startswith("reward_w_")]
    check("the cfg no longer defines a proxy reward weight",
          not stray_weights, f"{stray_weights}")
    log_fn = _func(env_tree, "_log_finished_episodes", "InsertionEnv")
    metrics_fn0 = _func(env_tree, "_write_metrics", "InsertionEnv")
    check("the per-iteration log carries the abort rate",
          "force_abort_rate" in _string_constants(log_fn))
    check("demo_metrics.json carries the abort rate",
          "force_abort_rate" in _string_constants(_func(env_tree, "_write_metrics", "InsertionEnv")))
    check("the per-iteration log carries the interpenetration maximum",
          {"interpen_max_mean_mm", "interpen_max_max_mm"} <= set(_string_constants(log_fn)))
    check("demo_metrics.json carries the interpenetration tail",
          "interpen_max_mm" in _string_constants(metrics_fn0))
    _ip_present, _ip_gated = _marker_under_flag(
        log_fn, "interpen_max_mean_mm", "_rl_terms_enabled")
    check("the interpenetration curve is gated on rl_terms_enabled",
          _ip_present and _ip_gated, f"present={_ip_present} gated={_ip_gated}")
    check("the per-iteration log carries the force maximum",
          {"force_max_mean_n", "force_max_p95_n", "force_max_max_n"} <= set(_string_constants(log_fn)))

    dones_fn = _func(env_tree, "_get_dones", "InsertionEnv")
    exposure_line = min(
        (n.lineno for n in _walk(dones_fn)
         if isinstance(n, ast.Assign) and "_max_stage1_lat_y" in _identifiers(n)),
        default=None,
    )
    sdf_line = min(
        (n.lineno for n in _walk(dones_fn)
         if isinstance(n, ast.Assign) and "_sdf_query" in _identifiers(n)),
        default=None,
    )
    check("the stage-1 exposure is recorded in _get_dones",
          exposure_line is not None, f"line {exposure_line}")
    check("the exposure is recorded above the reward-side work",
          exposure_line is not None and sdf_line is not None
          and exposure_line < sdf_line,
          f"exposure {exposure_line} vs SDF query {sdf_line}")
    check("the stage-1 band reads STAGE1_DEPTH, not a typed number",
          "STAGE1_DEPTH" in _identifiers(dones_fn))
    check("the exposure buffer is cleared in _reset_idx",
          "_max_stage1_lat_y" in _identifiers(_func(env_tree, "_reset_idx", "InsertionEnv")))
    stage1_fn = _func(env_tree, "_stage1_lateral_stats", "InsertionEnv")
    check("the exposure is judged against HOLE_REACH_OFFSET_Y",
          "HOLE_REACH_OFFSET_Y" in _identifiers(stage1_fn))
    check("the exposure statistic counts the episodes that reached the hole",
          "over_reach" in _string_constants(stage1_fn))
    check("demo_metrics.json carries the stage-1 exposure",
          "stage1_lateral_y_mm" in _string_constants(metrics_fn0)
          and "_stage1_lateral_stats" in _identifiers(metrics_fn0))
    check("the per-iteration log carries the stage-1 exposure",
          "stage1_lat_y_max_mm" in _string_constants(log_fn))

    def _call_args(fn, name: str) -> list[str]:
        for n in _walk(fn):
            if isinstance(n, ast.Call):
                f = n.func
                fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                if fname == name:
                    return [_unparse(a) for a in n.args]
        return []

    _gate_args = _call_args(dones_fn, "in_pocket_cross_section")
    check("the lateral gate is insertion_math's, not a second inline copy",
          "in_pocket_cross_section" in dones_ids)
    check("the gate reads the pocket walls, not a new tolerance",
          _gate_args[1:] == ["insertion_tasks_cfg.POCKET_WALL_X",
                             "insertion_tasks_cfg.POCKET_WALL_Y"],
          f"{_gate_args}")
    check("the success predicate is handed the lateral gate",
          "in_pocket" in _call_args(dones_fn, "in_success_region"),
          f"{_call_args(dones_fn, 'in_success_region')}")
    check("the engaged predicate is handed the lateral gate",
          "in_pocket" in _call_args(dones_fn, "is_engaged"),
          f"{_call_args(dones_fn, 'is_engaged')}")
    check("the per-iteration log carries the depth invariant",
          "success_depth_invariant_violations" in _string_constants(log_fn))
    check("demo_metrics.json carries the depth invariant",
          "success_depth_invariant_violations" in _string_constants(metrics_fn0))
    check("the pocket gate has a floor at the fixture underside, not a new tolerance",
          _call_args(dones_fn, "in_pocket_volume")
          == ["cross_section", "depth", "-insertion_tasks_cfg.POCKET_ASSET_BOTTOM_Z"],
          f"{_call_args(dones_fn, 'in_pocket_volume')}")
    check("the per-iteration log carries the below-fixture rate",
          "below_fixture_rate" in _string_constants(log_fn))
    check("demo_metrics.json carries the below-fixture rate",
          "below_fixture_rate" in _string_constants(metrics_fn0))
    check("the unclean-step counter reads the SAPU rule on this step's interpenetration",
          _call_args(dones_fn, "interpen_unclean")
          == ["interpen_max", "float(self.cfg.interpen_thresh)"],
          f"{_call_args(dones_fn, 'interpen_unclean')}")
    _reset_src = _unparse(_func(env_tree, "_reset_idx", "InsertionEnv"))
    check("the unclean-step counter is zeroed per episode",
          "self._interpen_unclean_steps[env_ids] = 0\n" in _reset_src + "\n")
    check("the raw force maximum is zeroed per episode",
          "self._max_force_raw_norm[env_ids] = 0.0" in _reset_src)
    check("the episode row is written from the harvest",
          "_append_episode_rows" in _identifiers(
              _func(env_tree, "_log_finished_episodes", "InsertionEnv")))
    check("episodes.csv carries exactly the agreed columns",
          _module_const(env_tree, "EPISODE_LOG_COLUMNS") == EPISODE_LOG_COLUMNS_AGREED,
          f"{_module_const(env_tree, 'EPISODE_LOG_COLUMNS')}")
    check("train.py hands the rollout length to the env",
          "env_cfg.rollout_steps_per_iteration = agent_cfg.num_steps_per_env"
          in _unparse(train_tree))

    _rewards_fn = _func(env_tree, "_get_rewards", "InsertionEnv")
    _init_fn = _func(env_tree, "__init__", "InsertionEnv")
    check("every reward term is accumulated per env",
          "_episode_sums" in _identifiers(_rewards_fn)
          and "REWARD_TERMS" in _identifiers(_init_fn))

    def _clears_episode_sums(fn) -> bool:
        for node in _walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = _unparse(node.targets[0])
                if (target.startswith("self._episode_sums[")
                        and isinstance(node.value, ast.Constant)
                        and node.value.value == 0.0):
                    return True
        return False

    check("the term sums are cleared at episode end", _clears_episode_sums(log_fn))
    check("the per-episode log carries the reward terms",
          "Episode_Reward/" in _string_constants(log_fn))
    check("demo_metrics.json carries the reward terms",
          "reward_terms_recent_mean" in _string_constants(metrics_fn0))

    _rew_args = _call_args(_rewards_fn, "compute_reward_terms_insertion")
    check("the reward is handed the depth progress the metric produced",
          "self._last_depth_progress" in _rew_args, f"{_rew_args[:9]}")
    _progress_writes = [
        n for n in _walk(dones_fn)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and _unparse(n.targets[0]) == "self._last_depth_progress"
        and not isinstance(n.value, ast.Call)
    ]
    check("the depth progress is the gated max-depth gain, not the raw depth",
          bool(_progress_writes)
          and all("_max_depth" in _identifiers(n.value) for n in _progress_writes),
          f"{[_unparse(n.value) for n in _progress_writes]}")
    check("the progress weight is labelled [proxy]",
          "cfg.w_depth_progress" in (_module_const(cfg_tree, "PROXY_TASK_VALUES") or {}))

    placeholders = _module_const(cfg_tree, "RL_PLACEHOLDERS") or {}

    def _placeholder_is_set(name: str) -> bool:
        """Does the cfg field hold a value at all -- literal OR derived?"""
        node = _class_attr_expr(cfg_tree, "InsertionEnvCfg", name)
        if node is None:
            return False
        if isinstance(node, ast.Constant) and node.value is None:
            return False
        return _class_attr_const(cfg_tree, "InsertionEnvCfg", name) is not None or not isinstance(
            node, (ast.Constant,)
        )

    unset = [n for n in placeholders if not _placeholder_is_set(n)]
    check("the cfg gives every placeholder a real value",
          bool(placeholders) and not unset, f"unset {unset}")

    report_fn = _func(env_tree, "_print_startup_report", "InsertionEnv")
    check("the startup report reads RL_PLACEHOLDERS",
          "RL_PLACEHOLDERS" in _identifiers(report_fn))
    check("the startup report shouts that the numbers are provisional",
          any("PLACEHOLDER VALUES IN USE" in c for c in _string_constants(report_fn)))

    metrics_fn = _func(env_tree, "_write_metrics", "InsertionEnv")
    check("demo_metrics.json carries the placeholders",
          "rl_placeholders" in _string_constants(metrics_fn)
          and "RL_PLACEHOLDERS" in _identifiers(metrics_fn))
    check("demo_metrics.json carries the scene state",
          "spawn_workcell_block" in _string_constants(metrics_fn0))
    check("demo_metrics.json carries the ladder state",
          "curriculum_enabled" in _string_constants(metrics_fn)
          and "rung_step_sizes" in _string_constants(metrics_fn))
    _scatter = ("obs_noise_pocket_pos_std_m", "force_obs_noise_std_n",
                "grasp_obs_offset_x_m", "pocket_obs_yaw_rad")
    _metrics_strings = _string_constants(metrics_fn)
    check("demo_metrics.json carries the observation noise and the grasp scatter",
          all(n in _metrics_strings for n in _scatter),
          f"missing {[n for n in _scatter if n not in _metrics_strings]}")
    check("every scatter number is a labelled placeholder",
          all(n in placeholders for n in _scatter),
          f"missing {[n for n in _scatter if n not in placeholders]}")

    proxy = _module_const(cfg_tree, "PROXY_TASK_VALUES") or {}
    proxy_keys = _module_const(cfg_tree, "PROXY_METRICS_KEYS") or {}
    report_ids = _identifiers(report_fn)
    report_strs = _string_constants(report_fn)
    metrics_ids = _identifiers(metrics_fn)
    metrics_strs = _string_constants(metrics_fn)

    check("the cfg carries a proxy-value table", bool(proxy), f"{len(proxy)} entries")
    dangling = sorted(k for k, path in proxy_keys.items() if path not in proxy)
    check("every proxy metrics key points at an entry that explains it",
          bool(proxy_keys) and not dangling, f"dangling {dangling}")
    absent_keys = sorted(k for k in proxy_keys if k not in metrics_strs)
    check("every proxy metrics key is really written to demo_metrics.json",
          not absent_keys, f"absent {absent_keys}")

    restating = sorted(p for p, mark in proxy.items()
                       if re.search(r"\d\.\d", str(mark)))
    check("no proxy mark restates its own number", not restating,
          f"restating {restating}")

    check("the startup report resolves PROXY_TASK_VALUES",
          "resolve_proxy_task_values" in report_ids)
    check("the startup report shouts that the scoring is proxy",
          any("PROXY TASK VALUES IN USE" in c for c in report_strs))
    check("demo_metrics.json carries the proxy values",
          "proxy_task_values" in metrics_strs
          and "resolve_proxy_task_values" in metrics_ids)
    check("demo_metrics.json says which of its own keys are proxy",
          "proxy_metrics_keys" in metrics_strs
          and "PROXY_METRICS_KEYS" in metrics_ids)

    check("the report does not print the proxy clearance as a task fact",
          "SIDE_CLEARANCE" not in report_ids)
    check("the report does not print the proxy yaw window as a task fact",
          "YAW_WINDOW_RAD" not in report_ids)
    check("the report prints the MEASURED play instead",
          "PLAY_X" in report_ids and "PLAY_Y" in report_ids)
    check("demo_metrics.json carries the measured play",
          "play_x_mm" in metrics_strs and "play_y_mm" in metrics_strs)
    check("the report title does not name the proxy branch",
          not any("square-peg-insertion" in c for c in report_strs))

    check("train.py refuses a measurement env", _refuses_on_flag(train_tree, FLAG))
    tag_parts = [
        _unparse(n.args[0])
        for n in _walk(train_tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "append"
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "parts"
        and n.args
    ]
    check("the run tag carries the seed",
          any("seed" in t and "agent_cfg.seed" in t for t in tag_parts),
          f"{tag_parts}")
    check("the run tag names the block only when it is ON",
          any(t == "'blockON'" or t == '"blockON"' for t in tag_parts)
          and not any("blockOFF" in t for t in tag_parts),
          f"{tag_parts}")
    check("the run tag does NOT carry the env count",
          not any("num_envs" in t for t in tag_parts), f"{tag_parts}")
    resume_calls = [
        n for n in _walk(train_tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "get_checkpoint_path"
    ]
    check("the resume picks the newest run by time, not by name",
          bool(resume_calls) and all(
              any(kw.arg == "sort_alpha" and kw.value.value is False for kw in c.keywords)
              for c in resume_calls),
          f"{len(resume_calls)} call(s)")
    check("the run tag does NOT assert the proxy peg and pocket sizes",
          not any("held_asset" in t or "fixed_asset" in t
                  or "peg" in t or "pocket" in t for t in tag_parts),
          f"{tag_parts}")
    check("zero_agent builds a measurement env", _sets_flag_false(zero_tree, FLAG))
    check("random_agent builds a measurement env", _sets_flag_false(random_tree, FLAG))

    scripted_tree = _parse(SCRIPTED_SRC, scripted_mut)
    check("scripted_insert builds a measurement env", _sets_flag_false(scripted_tree, FLAG))

    seat_tree = _parse(SEAT_SRC, seat_mut)
    check("seat_probe builds a measurement env", _sets_flag_false(seat_tree, FLAG))
    seat_calls = {
        n.func.attr for n in _walk(seat_tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    check("seat_probe teleports instead of descending",
          "write_joint_state_to_sim" in seat_calls,
          "write_joint_state_to_sim" if "write_joint_state_to_sim" in seat_calls else "MISSING")
    check("seat_probe holds the teleported pose with a drive target",
          "set_joint_position_target" in seat_calls)

    seat_inference = sorted({
        n.attr for n in _walk(seat_tree)
        if isinstance(n, ast.Attribute) and n.attr == "inference_mode"
    })
    check("seat_probe never runs under inference_mode",
          not seat_inference,
          "torch.no_grad()" if not seat_inference else "uses torch.inference_mode()")

    _teleport_loops = [
        n for n in _walk(seat_tree)
        if isinstance(n, (ast.For, ast.While))
        and any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == "write_joint_state_to_sim"
            for c in _walk(n)
        )
    ]
    _inner = [
        n for n in _teleport_loops
        if not any(o is not n and o in _walk(n) for o in _teleport_loops)
    ]
    _solve_steps = sorted({
        f"{c.func.value.id}.{c.func.attr}"
        for loop in _inner for c in _walk(loop)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
        and c.func.attr == "step" and isinstance(c.func.value, ast.Name)
    })
    check("seat_probe solves kinematically, without stepping the sim",
          bool(_inner) and not _solve_steps,
          "no step in the teleport loop" if bool(_inner) and not _solve_steps
          else (f"steps: {_solve_steps}" if _inner else "NO TELEPORT LOOP FOUND"))

    used = _identifiers(scripted_tree)
    touched = sorted(n for n in FORBIDDEN_TARGET_TRUTH if n in used)
    check("the scripted insertion never touches the target truth",
          not touched,
          f"touched {touched}" if touched else "none of " + str(len(FORBIDDEN_TARGET_TRUTH)))

    check("the scripted insertion reads the observation through OBS_SLICES",
          "OBS_SLICES" in used)
    named_blocks = {
        n.value for n in _walk(scripted_tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in slices
    }
    check("every observation block it reads is a key of that table",
          named_blocks and named_blocks <= set(slices),
          f"reads {sorted(named_blocks)}")

    check("the scripted gate can raise the abort limit for one run",
          "--f-abort" in _cli_flags(scripted_tree))
    check("the abort override reaches the cfg before the env is built",
          _assign_precedes_call(_func(scripted_tree, "main"), "force_abort_f_max_n", "make"))

    scripted_calls = {
        n.func.attr for n in _walk(scripted_tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    check("the scripted gate measures the axis tilt",
          "axis_tilt_angle" in scripted_calls)
    _metrics_keys: set = set()
    for node in _walk(_func(scripted_tree, "main")):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if any(isinstance(t, ast.Name) and t.id == "metrics" for t in node.targets):
                _metrics_keys = {
                    k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
    check("the axis tilt reaches the metrics file",
          "peak_axis_tilt_rad" in _metrics_keys,
          f"metrics keys: {len(_metrics_keys)}")
    check("the scripted gate bins the force by depth",
          "bin_force_by_depth" in scripted_calls)
    check("the force-depth profile reaches the metrics file",
          "force_by_depth" in _metrics_keys)
    _printed = "".join(
        _unparse(n) for n in _walk(_func(scripted_tree, "main"))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
    )
    check("the force-depth profile is PRINTED, not only written",
          "force max" in _printed and "force_mean_n" in _printed)

    _depths = _module_chain(
        tasks_tree, ("POCKET_FLOOR_Z", "POCKET_SEAT_DEPTH", "D106_DEPTH_BAND", "SEATED_SUCCESS_DEPTH")
    )
    _fixed_kwargs = _class_attr_kwargs(cfg_tree, "InsertionEnvCfg", "fixed_asset")

    check("the gate depth is wired to the real fixture's seat depth",
          _fixed_kwargs.get("depth") == "insertion_tasks_cfg.POCKET_SEAT_DEPTH",
          f"depth={_fixed_kwargs.get('depth')!r}")
    _seat_expr = _module_expr(tasks_tree, "POCKET_SEAT_DEPTH")
    check("the seat depth is derived from the measured POCKET_FLOOR_Z",
          _seat_expr == "-POCKET_FLOOR_Z",
          f"POCKET_SEAT_DEPTH = {_seat_expr}")
    check("the seat depth is the pocket floor, sign flipped",
          _depths.get("POCKET_SEAT_DEPTH") is not None
          and _depths.get("POCKET_FLOOR_Z") is not None
          and abs(_depths["POCKET_SEAT_DEPTH"] + _depths["POCKET_FLOOR_Z"]) < 1e-12,
          f"{_depths.get('POCKET_SEAT_DEPTH')} vs {_depths.get('POCKET_FLOOR_Z')}")

    check("the success depth is wired to the D-106 seated band",
          _fixed_kwargs.get("success_depth") == "insertion_tasks_cfg.SEATED_SUCCESS_DEPTH",
          f"success_depth={_fixed_kwargs.get('success_depth')!r}")
    _band_expr = _module_expr(tasks_tree, "SEATED_SUCCESS_DEPTH")
    check("the success depth is the seat depth minus the D-106 band",
          _band_expr == "POCKET_SEAT_DEPTH - D106_DEPTH_BAND",
          f"SEATED_SUCCESS_DEPTH = {_band_expr}")
    check("the seated band comes out at 33 mm",
          _depths.get("SEATED_SUCCESS_DEPTH") is not None
          and abs(_depths["SEATED_SUCCESS_DEPTH"] - 0.033) < 1e-12,
          f"{_depths.get('SEATED_SUCCESS_DEPTH')} m "
          f"(band {_depths.get('D106_DEPTH_BAND')} m)")

    _reward_nums = _module_chain(
        tasks_tree,
        ("POCKET_WALL_X", "POCKET_OPENING_X", "PART_BODY_X", "PLAY_X",
         "INTERPEN_THRESH", "KERNEL_A_FINE",
         "POCKET_FLOOR_Z", "POCKET_SEAT_DEPTH",
         "ENGAGED_DEPTH_FRACTION", "ENGAGED_DEPTH"),
        seed={"math": math},
    )

    _thresh_expr = _module_expr(tasks_tree, "INTERPEN_THRESH")
    check("the SAPU threshold is derived from the cross play",
          _thresh_expr == "0.5 * PLAY_X",
          f"INTERPEN_THRESH = {_thresh_expr}")
    check("the SAPU threshold comes out at half the play",
          _reward_nums.get("INTERPEN_THRESH") is not None
          and abs(_reward_nums["INTERPEN_THRESH"] - 0.0002938) < 1e-12,
          f"{_reward_nums.get('INTERPEN_THRESH')} m")

    _fine_expr = _module_expr(tasks_tree, "KERNEL_A_FINE")
    check("the fine kernel width is derived from the cross play",
          _fine_expr == "math.acosh(10.0) / PLAY_X",
          f"KERNEL_A_FINE = {_fine_expr}")
    check("the fine kernel width comes out at the decided 5094",
          _reward_nums.get("KERNEL_A_FINE") is not None
          and abs(_reward_nums["KERNEL_A_FINE"] - 5094.0) < 1.0,
          f"{_reward_nums.get('KERNEL_A_FINE')}")

    _engaged_expr = _module_expr(tasks_tree, "ENGAGED_DEPTH")
    check("the engaged depth is derived from the stage-2 seat depth",
          _engaged_expr == "ENGAGED_DEPTH_FRACTION * POCKET_SEAT_DEPTH",
          f"ENGAGED_DEPTH = {_engaged_expr}")
    check("the engaged depth comes out at 10.8 mm",
          _reward_nums.get("ENGAGED_DEPTH") is not None
          and abs(_reward_nums["ENGAGED_DEPTH"] - 0.0108) < 1e-12,
          f"{_reward_nums.get('ENGAGED_DEPTH')} m "
          f"(fraction {_reward_nums.get('ENGAGED_DEPTH_FRACTION')})")

    def _cfg_field_expr(name: str) -> str | None:
        node = _class_attr_expr(cfg_tree, "InsertionEnvCfg", name)
        return None if node is None else _unparse(node)

    for _field, _want in (
        ("engaged_depth_m", "insertion_tasks_cfg.ENGAGED_DEPTH"),
        ("interpen_thresh", "insertion_tasks_cfg.INTERPEN_THRESH"),
        ("kernel_a_fine", "insertion_tasks_cfg.KERNEL_A_FINE"),
        ("depth_min", "insertion_tasks_cfg.SEATED_SUCCESS_DEPTH"),
        ("depth_max", "insertion_tasks_cfg.POCKET_SEAT_DEPTH"),
    ):
        _got = _cfg_field_expr(_field)
        check(f"cfg.{_field} is wired to the derived constant",
              _got == _want, f"{_field} = {_got}")

    check("the real success weight is D-109's 1.0, not the proxy's",
          _class_attr_const(cfg_tree, "InsertionEnvCfg", "w_success") == 1.0,
          f"w_success = {_class_attr_const(cfg_tree, 'InsertionEnvCfg', 'w_success')}")

    _len_expr = _class_attr_expr(cfg_tree, "InsertionEnvCfg", "episode_length_s")
    _episode_s = _fold_number(_len_expr) if _len_expr is not None else None
    _dec = _class_attr_int(cfg_tree, "InsertionEnvCfg", "decimation")
    _sim = _class_attr_expr(cfg_tree, "InsertionEnvCfg", "sim")
    _dt = None
    if isinstance(_sim, ast.Call):
        for _kw in _sim.keywords:
            if _kw.arg == "dt":
                _dt = _fold_number(_kw.value)
    _steps = None
    if _episode_s is not None and _dec and _dt:
        _steps = math.ceil(_episode_s / (_dt * _dec))
    check("the episode cap is D-113's 256 control steps",
          _steps == 256,
          f"{_steps} steps (episode_length_s={_episode_s}, dt={_dt}, decimation={_dec})")
    check("the episode length is written as a ratio, not a rounded decimal",
          isinstance(_len_expr, ast.BinOp),
          _unparse(_len_expr) if _len_expr is not None else "<absent>")
    _report_fn = _func(env_tree, "_print_startup_report", "InsertionEnv")
    _report_attrs = {
        n.attr for n in _walk(_report_fn) if isinstance(n, ast.Attribute)
    }
    check("the startup report prints the episode cap in STEPS",
          "max_episode_length" in _report_attrs,
          f"episode_length_s also printed: {'episode_length_s' in _report_attrs}")
    _report_expr = _class_attr_expr(cfg_tree, "InsertionEnvCfg", "report_at_steps")
    _report_counts = [
        n for n in (_fold_number(e) for e in getattr(_report_expr, "elts", ()))
        if n is not None
    ]
    _late_counts = [c for c in _report_counts if _steps is not None and c > _steps]
    check("the startup report is taken at least once past the episode cap",
          bool(_late_counts),
          f"report_at_steps={_report_counts}, cap={_steps} steps, past it: {_late_counts}")
    _init_fn = _func(env_tree, "__init__", "InsertionEnv")
    _obs_fn = _func(env_tree, "_get_observations", "InsertionEnv")

    def _obs_call_writes(scope: ast.AST) -> tuple[set[int], set[int]]:
        plain = {
            n.lineno for n in _walk(scope) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "_obs_calls"
                    for t in n.targets)
        }
        aug = {
            n.lineno for n in _walk(scope) if isinstance(n, ast.AugAssign)
            and isinstance(n.target, ast.Attribute)
            and n.target.attr == "_obs_calls"
        }
        return plain, aug

    _oc_plain, _oc_aug = _obs_call_writes(env_tree)
    _oc_init_plain, _oc_init_aug = _obs_call_writes(_init_fn)
    _oc_obs_plain, _oc_obs_aug = _obs_call_writes(_obs_fn)
    check("_obs_calls is written in exactly two places: = in __init__, += in _get_observations",
          len(_oc_plain) == 1 and _oc_plain == _oc_init_plain and not _oc_init_aug
          and len(_oc_aug) == 1 and _oc_aug == _oc_obs_aug and not _oc_obs_plain,
          f"assigns at {sorted(_oc_plain)} (__init__ {sorted(_oc_init_plain)}), "
          f"increments at {sorted(_oc_aug)} (_get_observations {sorted(_oc_obs_aug)})")

    _scene_fn = _func(env_tree, "_setup_scene", "InsertionEnv")
    _blk_present, _blk_guarded = _marker_under_flag(
        _scene_fn, "/World/envs/env_0/Block", "spawn_workcell_block")
    check("the workcell block spawns ONLY under cfg.spawn_workcell_block",
          _blk_present and _blk_guarded,
          f"prim path in _setup_scene: {_blk_present}, under the flag: {_blk_guarded}")
    _blk_default = _class_attr_const(cfg_tree, "InsertionEnvCfg", "spawn_workcell_block")
    check("the workcell block defaults to OFF",
          _blk_default is False,
          f"spawn_workcell_block={_blk_default!r}")
    check("the startup report prints the workcell block state",
          "spawn_workcell_block" in _identifiers(_report_fn),
          f"report mentions the flag: {'spawn_workcell_block' in _identifiers(_report_fn)}")

    _init_fn = _func(env_tree, "__init__", "InsertionEnv")
    _randomised = [
        n for n in _walk(_init_fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "randomised" for t in n.targets)
    ]
    _rand_ids = _identifiers(_randomised[0].value) if _randomised else set()
    _z_maps = [
        _unparse(n) for n in _walk(_func(env_tree, "_reset_idx", "InsertionEnv"))
        if isinstance(n, ast.AugAssign) and _unparse(n.target) == "entrance[:, 2]"
    ]
    check("the reset moves the entrance along z by the z spread, from its own table column",
          len(_z_maps) == 1 and "self._col['fixture_noise_z']" in _z_maps[0]
          and _z_maps[0].endswith("* self.cfg.fixture_pos_noise_z"), str(_z_maps))
    check("the deterministic-vs-randomised guard counts fixture_pos_noise_z",
          "fixture_pos_noise_z" in _rand_ids, f"randomised reads {sorted(_rand_ids)}")
    check("the deterministic-vs-randomised guard counts fixture_pos_noise_xy",
          "fixture_pos_noise_xy" in _rand_ids,
          f"randomised reads {sorted(_rand_ids & {'fixture_pos_noise_xy', 'fixture_tilt_noise_rad', 'fixture_yaw_noise_rad'})}")

    _experiment = _class_attr_const(ppo_tree, "PPORunnerCfg", "experiment_name")
    _compare_default = _cli_default(compare_tree, "--experiment")
    check("the experiment folder is not the proxy demo sprint's",
          isinstance(_experiment, str) and _experiment != "demo_insertion",
          f"experiment_name={_experiment!r}")
    check("compare_runs reads the same experiment folder train.py writes",
          _experiment is not None and _compare_default == _experiment,
          f"{_compare_default!r} vs {_experiment!r}")

    init_tree = _parse(INIT_SRC, init_mut)
    _reg = _gym_register_kwargs(init_tree)
    check("the default agent entry point stays the 16-step MLP PPORunnerCfg",
          _reg.get("rsl_rl_cfg_entry_point", "").endswith(".rsl_rl_ppo_cfg:PPORunnerCfg'"),
          str(_reg.get("rsl_rl_cfg_entry_point")))
    check("the LSTM agent is registered as rsl_rl_rnn_cfg_entry_point",
          _reg.get("rsl_rl_rnn_cfg_entry_point", "").endswith(".rsl_rl_ppo_cfg:PPORunnerLstmCfg'"),
          str(_reg.get("rsl_rl_rnn_cfg_entry_point")))
    check("the MLP ablation agent is registered as rsl_rl_mlp_ablation_cfg_entry_point",
          _reg.get("rsl_rl_mlp_ablation_cfg_entry_point", "").endswith(
              ".rsl_rl_ppo_cfg:PPORunnerMlpAblationCfg'"),
          str(_reg.get("rsl_rl_mlp_ablation_cfg_entry_point")))
    _abl = "PPORunnerMlpAblationCfg"
    check("the LSTM cfg extends the ablation cfg, which extends PPORunnerCfg",
          _class_bases(ppo_tree, "PPORunnerLstmCfg") == [_abl]
          and _class_bases(ppo_tree, _abl) == ["PPORunnerCfg"],
          f"{_class_bases(ppo_tree, 'PPORunnerLstmCfg')} / {_class_bases(ppo_tree, _abl)}")
    _abl_post = _method_self_assigns(ppo_tree, _abl, "__post_init__")
    _b1 = (_class_attr_const(ppo_tree, _abl, "num_steps_per_env"),
           _class_attr_const(ppo_tree, _abl, "max_iterations"),
           _literal(_abl_post.get("self.algorithm.num_mini_batches")))
    check("the ablation cfg sets rollout 128, 400 iterations, 4 mini-batches (D-195 B1, BUDGET, D-198)",
          _b1 == (128, 400, 4), str(_b1))
    try:
        _b0 = (_class_attr_const(ppo_tree, "PPORunnerCfg", "num_steps_per_env"),
               _class_attr_const(ppo_tree, "PPORunnerCfg", "max_iterations"),
               int(_class_attr_kwargs(ppo_tree, "PPORunnerCfg", "algorithm")["num_mini_batches"]))
    except (KeyError, ValueError):
        _b0 = (None, None, None)
    _b_ok = all(isinstance(x, int) for x in _b0 + _b1)
    check("the B1 budget is the user's 51,200 steps per env (400 x 128, D-195 BUDGET)",
          _b_ok and _b1[1] * _b1[0] == 51200,
          f"{_b1} vs {_b0}")
    check("the ablation cfg keeps the base cfg's mini-batch count, Isaac Lab's 4 (D-198)",
          _b_ok and _b1[2] == _b0[2], f"{_b1} vs {_b0}")
    _groups = _class_attr_expr(ppo_tree, _abl, "obs_groups")
    check("the ablation cfg sends the env's critic group to the critic (D-195)",
          _groups is not None
          and _unparse(_groups) == "{'policy': ['policy'], 'critic': ['critic']}",
          _unparse(_groups) if _groups is not None else "no obs_groups")
    _lstm_fn = [n for n in _walk(ppo_tree) if isinstance(n, ast.ClassDef) and n.name == "PPORunnerLstmCfg"]
    check("the LSTM cfg's __post_init__ runs the ablation cfg's first",
          bool(_lstm_fn) and "super().__post_init__()" in [
              _unparse(s) for f in _lstm_fn[0].body if isinstance(f, ast.FunctionDef)
              and f.name == "__post_init__" for s in f.body], "")
    _lstm_pol = _method_self_assigns(ppo_tree, "PPORunnerLstmCfg", "__post_init__").get("self.policy")
    _lstm_kw = ({k.arg: _unparse(k.value) for k in _lstm_pol.keywords if k.arg}
                if isinstance(_lstm_pol, ast.Call) else {})
    check("the LSTM policy is rsl_rl's recurrent actor-critic, lstm 2 x 256 (D-195 A1)",
          isinstance(_lstm_pol, ast.Call)
          and _unparse(_lstm_pol.func) == "RslRlPpoActorCriticRecurrentCfg"
          and {k: v for k, v in _lstm_kw.items() if k.startswith("rnn_")}
          == {"rnn_type": "'lstm'", "rnn_hidden_dim": "256", "rnn_num_layers": "2"},
          str(_lstm_kw))
    _mlp_kw = _class_attr_kwargs(ppo_tree, "PPORunnerCfg", "policy")
    check("the LSTM policy's MLP part is read from the MLP policy, field by field",
          bool(_mlp_kw) and {k: v for k, v in _lstm_kw.items() if not k.startswith("rnn_")}
          == {k: f"self.policy.{k}" for k in _mlp_kw},
          f"{sorted(_lstm_kw)} vs {sorted(_mlp_kw)}")

    _start_expr = _class_attr_expr(cfg_tree, "InsertionEnvCfg", "start_tip_above_entrance")
    _start_src = _unparse(_start_expr) if _start_expr is not None else None
    check("the rung-0 start height is derived, not typed",
          _start_src is not None and "RUNG0_START_TIP_ABOVE_ENTRANCE" in _start_src,
          f"start_tip_above_entrance = {_start_src}")
    _chain = _module_chain(
        tasks_tree, ("STAGE1_DEPTH", "RUNG0_START_TIP_ABOVE_ENTRANCE")
    )
    _rim = _chain.get("STAGE1_DEPTH")
    _rung0 = _chain.get("RUNG0_START_TIP_ABOVE_ENTRANCE")
    check("the rung-0 start height clears the fixture's stage-1 rim",
          isinstance(_rim, (int, float)) and isinstance(_rung0, (int, float))
          and _rung0 > _rim,
          f"start {_rung0} m vs stage-1 rim {_rim} m")

    _reset_fn = _func(env_tree, "_reset_idx", "InsertionEnv")
    _solve_lines = [
        n.lineno for n in _walk(_reset_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_solve_start_pose"
    ]
    check("_reset_idx solves the start pose",
          bool(_solve_lines), f"call lines {_solve_lines}")
    _fixture_lines = [
        n.lineno for n in _walk(_reset_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("write_root_pose_to_sim", "write_root_velocity_to_sim")
    ]
    check("the start pose is solved AFTER the fixture pose is written",
          bool(_solve_lines) and bool(_fixture_lines)
          and min(_solve_lines) > max(_fixture_lines),
          f"solve {_solve_lines} vs fixture {_fixture_lines}")

    _solve_fn = _func(env_tree, "_solve_start_pose", "InsertionEnv")
    _solve_calls = {
        n.func.attr for n in _walk(_solve_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    check("the start-pose solve never steps the sim",
          not (_solve_calls & {"step", "step_sim_no_action", "render"}),
          f"stepping calls {sorted(_solve_calls & {'step', 'step_sim_no_action', 'render'})}")
    _solve_writes = {
        sub.attr
        for n in _walk(_solve_fn) if isinstance(n, ast.Assign)
        for target in n.targets
        for sub in _walk(target) if isinstance(sub, ast.Attribute)
    }
    check("the start pose seeds the joint-target integrator",
          "_joint_targets" in _solve_writes,
          f"written: {sorted(w for w in _solve_writes if w.startswith('_'))}")

    seated_tree = _parse(SEATED_SRC, seated_mut)
    recovery_tree = _parse(RECOVERY_SRC, recovery_mut)
    _pinned = {
        "scripted_insert": _sets_attr_none(scripted_tree, "start_tip_above_entrance"),
        "seat_probe": _sets_attr_none(seat_tree, "start_tip_above_entrance"),
        "check_seated_success": _sets_attr_none(seated_tree, "start_tip_above_entrance"),
    }
    check("the teleport runs pin the start pose back to the home pose",
          all(_pinned.values()),
          ", ".join(f"{k}={v}" for k, v in _pinned.items()))
    _SCATTER = ("obs_noise_pocket_pos_std_m", "force_obs_noise_std_n",
                "grasp_obs_offset_x_m", "pocket_obs_yaw_rad")
    _scatter_pinned = {
        name: sorted(f for f in _SCATTER if not _sets_attr_zero(tree, f))
        for name, tree in (("scripted_insert", scripted_tree),
                           ("seat_probe", seat_tree),
                           ("check_seated_success", seated_tree),
                           ("zero_agent", zero_tree),
                           ("tilt_insert", tilt_tree),
                           ("tilt_recovery_probe", recovery_tree))
    }
    check("the identity runs pin all four observation scatter fields to zero",
          not any(_scatter_pinned.values()),
          "; ".join(f"{k} misses {v}" for k, v in _scatter_pinned.items() if v)
          or f"all {len(_scatter_pinned)} pin {list(_SCATTER)}")

    _DUMPED_SCATTER = {"fixture_pos_noise_xy": "fixture_pos_noise_xy_m",
                       "fixture_pos_noise_z": "fixture_pos_noise_z_m",
                       **{_f: _f for _f in _SCATTER}}
    _identity_dump: dict[str, str] = {}
    _curve_dump: dict[str, str] = {}
    for _node in _walk(seated_tree):
        if not (isinstance(_node, ast.Assign) and isinstance(_node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == "metrics"
                        for t in _node.targets)):
            continue
        _items = {k.value: _unparse(v)
                  for k, v in zip(_node.value.keys, _node.value.values)
                  if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if "identity test" in _items.get("run", ""):
            _identity_dump = _items
        elif "reward curve" in _items.get("run", ""):
            _curve_dump = _items

    def _dump_gaps(dump: dict[str, str]) -> list[str]:
        return sorted(
            _key for _field, _key in _DUMPED_SCATTER.items()
            if f"env_cfg.{_field}" not in dump.get(_key, "")
        )

    _dump_missing = _dump_gaps(_identity_dump)
    check("the seated metrics dump names every pinned scatter field",
          bool(_identity_dump) and not _dump_missing,
          f"missing: {_dump_missing}" if _dump_missing
          else f"carried: {sorted(_DUMPED_SCATTER.values())}")
    _curve_missing = _dump_gaps(_curve_dump)
    check("the reward-curve metrics dump names every pinned scatter field",
          bool(_curve_dump) and not _curve_missing,
          f"missing: {_curve_missing}" if _curve_missing
          else f"carried: {sorted(_DUMPED_SCATTER.values())}")

    _env_strings = {
        n.value for n in _walk(env_tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    check("demo_metrics.json carries the start height",
          "start_tip_above_entrance_mm" in _env_strings, "")

    _draw_fn = _func(env_tree, "_draw_reset_conditions", "InsertionEnv")
    _map_fn = _func(env_tree, "_map_start_conditions", "InsertionEnv")
    _rand_calls = {
        "_reset_idx": _reset_fn, "_solve_start_pose": _solve_fn,
        "_map_start_conditions": _map_fn, "_draw_reset_conditions": _draw_fn,
    }
    _draws_in = {
        name: sorted(
            c.func.attr for c in _walk(fn)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr in ("rand", "rand_like", "randn", "randint", "normal")
        )
        for name, fn in _rand_calls.items()
    }
    check("ONE place draws the reset conditions -- _draw_reset_conditions",
          _draws_in["_draw_reset_conditions"] == ["rand", "rand"]
          and not any(v for k, v in _draws_in.items()
                      if k != "_draw_reset_conditions"),
          str(_draws_in))
    _height_assigns = [
        n for n in _walk(_map_fn) if isinstance(n, ast.Assign)
        and any(isinstance(sub, ast.Attribute) and sub.attr == "_start_height"
                for target in n.targets for sub in _walk(target))
    ]
    _map_reads_draw = any(
        isinstance(sub, ast.Attribute) and sub.attr == "_reset_unit"
        for sub in _walk(_map_fn)
    )
    _height_from_draw = _map_reads_draw and any(
        "start_height" in _unparse(a.value) for a in _height_assigns
    )
    check("the start height is mapped per env from its own column of the draw",
          _height_from_draw,
          f"_start_height assignments {[a.lineno for a in _height_assigns]}, reads _reset_unit: {_height_from_draw}")
    _solve_height_assigns = [
        n for n in _walk(_solve_fn) if isinstance(n, ast.Assign)
        and any(isinstance(sub, ast.Attribute) and sub.attr in ("_start_height", "_start_lat_off")
                for target in n.targets for sub in _walk(target))
    ]
    check("the solver REALISES the start pose, it does not decide it",
          not _solve_height_assigns,
          f"assignments in _solve_start_pose: {[a.lineno for a in _solve_height_assigns]}")
    _draw_lines = [
        n.lineno for n in _walk(_reset_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_draw_reset_conditions"
    ]
    _map_lines = [
        n.lineno for n in _walk(_reset_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_map_start_conditions"
    ]
    _unit_reads = [
        n.lineno for n in _walk(_reset_fn)
        if isinstance(n, ast.Attribute) and n.attr == "_reset_unit"
    ]
    check("the draw runs before the FIRST consumer of the row",
          bool(_draw_lines) and bool(_unit_reads)
          and max(_draw_lines) < min(_unit_reads),
          f"draw {_draw_lines}, first read {min(_unit_reads) if _unit_reads else None}")
    check("the mapping runs before the solve",
          bool(_map_lines) and bool(_solve_lines)
          and min(_map_lines) < min(_solve_lines),
          f"map {_map_lines}, solve {_solve_lines}")
    _env_init = _func(env_tree, "__init__", "InsertionEnv")
    _unit_cols = [
        n for n in _walk(_env_init)
        if isinstance(n, ast.Assign)
        and any(isinstance(sub, ast.Attribute) and sub.attr == "_reset_unit"
                for t in n.targets for sub in _walk(t))
    ]
    check("the reset row is sized from autodr.TABLE_COLUMNS, never a literal",
          bool(_unit_cols)
          and all("TABLE_COLUMNS" in _unparse(a.value) for a in _unit_cols),
          "; ".join(_unparse(a.value) for a in _unit_cols))

    _dr_assigns = [
        n for n in _walk(_env_init) if isinstance(n, ast.Assign)
        and any(isinstance(sub, ast.Attribute) and sub.attr == "_dr"
                for t in n.targets for sub in _walk(t))
    ]
    check("the bounds provider is built in __init__ and stored on self._dr",
          bool(_dr_assigns)
          and any("autodr.AutoDR" in _unparse(a.value) for a in _dr_assigns)
          and any(_unparse(a.value) == "None" for a in _dr_assigns),
          "; ".join(_unparse(a.value) for a in _dr_assigns))
    _reach_assigns = [
        n for n in _walk(_env_init) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_reach" for t in n.targets)
    ]
    check("the reach comes from bounds_max, never from the width-0 bounds",
          bool(_reach_assigns)
          and all("bounds_max" in _unparse(a.value) for a in _reach_assigns),
          "; ".join(_unparse(a.value) for a in _reach_assigns))

    _EXPECT = {
        ("_angle_reach", "off"):
            "float(cfg.fixture_tilt_rad) != 0.0 or float(cfg.fixture_tilt_noise_rad) "
            "!= 0.0 or float(cfg.fixture_yaw_rad) != 0.0 or "
            "(float(cfg.fixture_yaw_noise_rad) != 0.0)",
        ("_fixture_pose_reach", "off"):
            "float(cfg.fixture_pos_noise_xy) > 0.0 or float(cfg.fixture_pos_noise_z) "
            "> 0.0 or float(cfg.fixture_yaw_noise_rad) > 0.0 or "
            "(float(cfg.fixture_tilt_noise_rad) > 0.0)",
        ("_angle_reach", "dr"): "_yaw_span or _tilt_span",
        ("_fixture_pose_reach", "dr"):
            "_angle_reach or float(cfg.fixture_pos_noise_xy) > 0.0 or "
            "float(cfg.fixture_pos_noise_z) > 0.0",
        ("_yaw_span", "dr"): "_yaw_r[1] - _yaw_r[0] > 0.0",
        ("_tilt_span", "dr"): "_tilt_r[1] - _tilt_r[0] > 0.0",
        ("_yaw_r", "dr"):
            "_reach.get('yaw', (-float(cfg.fixture_yaw_noise_rad), "
            "float(cfg.fixture_yaw_noise_rad)))",
        ("_tilt_r", "dr"): "_reach.get('tilt', (0.0, float(cfg.fixture_tilt_noise_rad)))",
        ("_reach", "dr"): "self._dr.bounds_max()",
    }

    def _named_assign(fn, name):
        return [
            n for n in _walk(fn) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
        ]

    _reach_if = [
        n for n in _walk(_env_init) if isinstance(n, ast.If)
        and "_dr is not None" in _unparse(n.test)
        and any(isinstance(b, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_angle_reach"
                        for t in b.targets)
                for b in _walk(n))
    ]
    _dr_lines, _off_lines = set(), set()
    for _n in _reach_if:
        for _b in _n.body:
            _dr_lines.update(range(_b.lineno, (_b.end_lineno or _b.lineno) + 1))
        for _b in _n.orelse:
            _off_lines.update(range(_b.lineno, (_b.end_lineno or _b.lineno) + 1))

    def _one_in(name, arm):
        """The single assignment to `name` in that arm, or None."""
        want = _dr_lines if arm == "dr" else _off_lines
        found = [a for a in _named_assign(_env_init, name) if a.lineno in want]
        total = len(_named_assign(_env_init, name))
        expected = 2 if name in ("_angle_reach", "_fixture_pose_reach") else 1
        if len(found) != 1 or total != expected:
            return None
        return found[0]

    _pinned = {}
    for (_name, _arm), _want in _EXPECT.items():
        _a = _one_in(_name, _arm)
        _pinned[(_name, _arm)] = (
            None if _a is None else _unparse(_a.value)
        )
    _bad = {f"{n}[{a}]": got for (n, a), got in _pinned.items() if got != _EXPECT[(n, a)]}
    check("the off-mode reaches are the frozen pre-Phase-5 expressions, exactly",
          not any(k.endswith("[off]") for k in _bad),
          "; ".join(f"{k} = {v}" for k, v in _bad.items() if k.endswith("[off]")))
    check("the provider reaches are the exact bounds_max expressions",
          not any(k.endswith("[dr]") for k in _bad),
          "; ".join(f"{k} = {v}" for k, v in _bad.items() if k.endswith("[dr]")))
    _flag_assigns = {
        attr: [
            _unparse(n.value) for n in _walk(_env_init)
            if isinstance(n, ast.Assign)
            and any(isinstance(sub, ast.Attribute) and sub.attr == attr
                    for t in n.targets for sub in _walk(t))
        ]
        for attr in ("_tilt_active", "_fixture_pose_reach")
    }
    check("both B2 flags are assigned the reach itself, once, and nothing else",
          _flag_assigns["_tilt_active"] == ["_angle_reach"]
          and _flag_assigns["_fixture_pose_reach"] == ["_fixture_pose_reach"],
          str(_flag_assigns))
    _static_dicts = [
        a.value for a in _named_assign(_env_init, "_static_dr")
        if isinstance(a.value, ast.Dict)
    ]
    _static_pairs = {
        k.value: _unparse(v)
        for d in _static_dicts for k, v in zip(d.keys, d.values)
        if isinstance(k, ast.Constant)
    }
    _OWNED = {
        "fixture_yaw_noise_rad": "float(cfg.fixture_yaw_noise_rad)",
        "fixture_tilt_noise_rad": "float(cfg.fixture_tilt_noise_rad)",
        "fixture_tilt_rad": "float(cfg.fixture_tilt_rad)",
        "fixture_yaw_rad": "float(cfg.fixture_yaw_rad)",
        "start_lateral_offset": "self._start_lat_half",
    }
    check("the autodr refusal names every boundary-owned static field, and reads it",
          _static_pairs == _OWNED,
          f"got {_static_pairs}")
    _live_assigns = [_unparse(a.value) for a in _named_assign(_env_init, "_live")]
    check("the refusal fires on a NON-ZERO static field",
          _live_assigns == ["{k: v for k, v in _static_dr.items() if v != 0.0}"],
          str(_live_assigns))
    _low_src = [_unparse(a.value) for a in _named_assign(_env_init, "_low_static")]
    _low_ifs = [
        n for n in _walk(_env_init) if isinstance(n, ast.If)
        and "_low_static" in _unparse(n.test)
        and "_live" in _unparse(n)
    ]
    _low_tests = [_unparse(n.test) for n in _low_ifs]
    check("the autodr refusal covers the start-height lower bound too",
          _low_src == ["cfg.start_tip_above_entrance_low"]
          and _low_tests == ["_low_static is not None and float(_low_static) != "
                             "float(cfg.start_tip_above_entrance)"],
          f"source {_low_src}, test {_low_tests}")
    _pose_guards = [
        n for n in _walk(_reset_fn) if isinstance(n, ast.If)
        and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "write_root_pose_to_sim" for c in _walk(n))
    ]
    _cfg_noise_names = ("fixture_pos_noise_xy", "fixture_yaw_noise_rad",
                        "fixture_tilt_noise_rad")
    check("the fixture-pose reset guard reads the reach flag, not the cfg ranges",
          bool(_pose_guards)
          and all("_fixture_pose_reach" in _unparse(g.test) for g in _pose_guards)
          and not any(name in _unparse(g.test)
                      for g in _pose_guards for name in _cfg_noise_names),
          "; ".join(_unparse(g.test) for g in _pose_guards))
    _dr_mode_default = _class_attr_const(cfg_tree, "InsertionEnvCfg", "dr_mode")
    check("dr_mode is a cfg field and defaults to 'off'",
          _dr_mode_default == "off", repr(_dr_mode_default))
    _init_raises = [
        n for n in _walk(_env_init) if isinstance(n, ast.If)
        and any(isinstance(b, ast.Raise) for b in n.body)
    ]
    _mode_guards = [g for g in _init_raises if "dr_mode" in _unparse(g.test)]
    _static_guards = [
        g for g in _init_raises
        if any(isinstance(sub, ast.Name) and sub.id == "_live" for sub in _walk(g.test))
    ]
    _mode_tests = [_unparse(g.test) for g in _mode_guards]
    check("an unknown dr_mode is refused, never run as 'off'",
          _mode_tests == ["str(cfg.dr_mode) != 'off'"],
          "; ".join(_mode_tests))

    _tbl_draw_ifs = [
        n for n in _walk(_draw_fn) if isinstance(n, ast.If)
        and _unparse(n.test) == "self._eval_table_unit is not None"
    ]
    _tbl_draw = [
        ([_unparse(s) for s in n.body], [_unparse(s).split("(")[0] for s in n.orelse])
        for n in _tbl_draw_ifs
    ]
    check("the reset draw copies row i for env i through autodr.table_rows, else torch.rand",
          _tbl_draw == [(
              ["self._reset_unit[idx] = autodr.table_rows(self._eval_table_unit, idx)"],
              ["self._reset_unit[idx] = torch.rand"],
          )],
          str(_tbl_draw))
    _tbl_mode_ifs = [
        n for n in _walk(_env_init) if isinstance(n, ast.If)
        and _unparse(n.test) == "str(cfg.dr_mode) == 'table'"
    ]
    _tbl_bodies = [[_unparse(s) for s in n.body][:2] for n in _tbl_mode_ifs]
    check("table mode validates its own cfg before it reads the table",
          _tbl_bodies.count([
              "validate_table_mode(cfg)",
              "self._eval_table = autodr.RowTable.from_csv(cfg.dr_table_path, "
              "allow_outside=bool(cfg.dr_table_allow_outside))",
          ]) == 1,
          str(_tbl_bodies))
    _tbl_count_tests = [
        _unparse(g.test) for g in _init_raises if "_eval_table" in _unparse(g.test)
    ]
    check("the table must have EXACTLY num_envs rows",
          _tbl_count_tests == ["self._eval_table.n_rows != int(self.num_envs)"],
          str(_tbl_count_tests))
    _tbl_dims = [_unparse(a.value) for a in _named_assign(_env_init, "_table_dims")]
    _tbl_nominal = [
        _unparse(n.test) for n in _walk(_env_init) if isinstance(n, ast.If)
        and any("_table_dims" in _unparse(s) for s in n.body)
        and "dr_table_width" in _unparse(n.test)
    ]
    _tbl_provider = [
        _unparse(n.value) for m in _tbl_mode_ifs for s in m.body for n in _walk(s)
        if isinstance(n, ast.Assign) and _unparse(n.targets[0]) == "self._dr"
    ]
    check("the table provider is FixedWidth.at_max, and 'nominal' keeps the start height alone",
          len(_tbl_dims) == 2 and _tbl_dims[0].startswith("autodr.bind_centres(")
          and _tbl_dims[1] == "tuple((d for d in _table_dims if d.name == 'start_height'))"
          and _tbl_nominal == ["str(cfg.dr_table_width) == 'nominal'"]
          and _tbl_provider == ["autodr.FixedWidth.at_max(_table_dims)"],
          f"dims {_tbl_dims}, nominal {_tbl_nominal}, provider {_tbl_provider}")
    _five_key_dicts = [
        _unparse(a) for c in _walk(_env_init) if isinstance(c, ast.Call)
        and _unparse(c.func).endswith("bind_centres")
        for a in c.args if isinstance(a, ast.Dict) and len(a.keys) == 5
    ]
    check("the table and Fixed-DR providers open around the SAME five centres as AutoDR",
          len(_five_key_dicts) == 3 and len(set(_five_key_dicts)) == 1,
          str(_five_key_dicts))
    _shared_refusal = [
        n for n in _walk(_env_init) if isinstance(n, ast.If)
        and _unparse(n.test) == "str(cfg.dr_mode) in ('autodr', 'fixed_dr', 'table')"
        and any(isinstance(sub, ast.Name) and sub.id == "_live" for sub in _walk(n))
    ]
    check("the static-field refusal covers 'autodr', 'fixed_dr' AND 'table'",
          len(_shared_refusal) == 1, f"{len(_shared_refusal)} block(s)")
    _fd_provider = [
        (_unparse(n.value.func),
         [_unparse(k.value) for k in n.value.keywords if k.arg in ("floor", "stall_buffers")])
        for m in _walk(_env_init) if isinstance(m, ast.If)
        and _unparse(m.test) == "str(cfg.dr_mode) == 'fixed_dr'"
        for s in m.body for n in _walk(s)
        if isinstance(n, ast.Assign) and _unparse(n.targets[0]) == "self._dr"
        and isinstance(n.value, ast.Call)
    ]
    check("the fixed_dr arm builds autodr.FixedDR on the start floor",
          _fd_provider == [("autodr.FixedDR", ["_floor_spec", "int(cfg.autodr_stall_buffers)"])],
          str(_fd_provider))
    _phase_src = [
        _unparse(a.value)
        for a in _named_assign(_func(env_tree, "_append_episode_rows", "InsertionEnv"), "phase")
    ]
    check("the episode log reads the phase with a default, so FixedWidth cannot crash it",
          _phase_src == ["str(getattr(self._dr, 'phase', 'fixed'))", "'none'"],
          str(_phase_src))
    _tbl_report = [
        _unparse(n) for n in _walk(_func(env_tree, "_print_startup_report", "InsertionEnv"))
        if isinstance(n, ast.JoinedStr) and "[eval-table]" in _unparse(n)
    ]
    check("the startup report prints the table's sha256 on the [eval-table] line (G1)",
          len(_tbl_report) == 1 and "self._eval_table.sha256" in _tbl_report[0]
          and "self._eval_table.n_rows" in _tbl_report[0],
          str(_tbl_report)[:200])
    _autodr_only = [
        _unparse(n.test) for n in _walk(_func(env_tree, "_print_startup_report", "InsertionEnv"))
        if isinstance(n, ast.If) and "buffer_m" in _unparse(n)
        and not any(isinstance(s, ast.If) and "buffer_m" in _unparse(s) for s in n.body)
    ]
    check("the AutoDR-only report paragraph is guarded by the provider's type",
          _autodr_only == ["isinstance(self._dr, autodr.AutoDR)"], str(_autodr_only))
    _applied_fn = _func(env_tree, "_write_applied_csv", "InsertionEnv")
    check("applied.csv carries exactly the agreed columns",
          _module_const(env_tree, "APPLIED_LOG_COLUMNS") == APPLIED_LOG_COLUMNS_AGREED,
          f"{_module_const(env_tree, 'APPLIED_LOG_COLUMNS')}")
    _applied_stacks = [
        tuple(_unparse(e) for e in n.args[0].elts)
        for n in _walk(_applied_fn)
        if isinstance(n, ast.Call) and _unparse(n.func) == "torch.stack"
    ]
    check("every applied.csv column reads its agreed source, row id = env id, in column order",
          _applied_stacks == [APPLIED_LOG_SOURCES_AGREED], str(_applied_stacks)[:300])
    _applied_fric = [_unparse(a.value) for a in _named_assign(_applied_fn, "_fric")]
    check("the four friction read-backs come out of PhysX, robot first, fixture second",
          len(_applied_fric) == 1
          and "root_physx_view.get_material_properties()[..., 0:2]" in _applied_fric[0]
          and _applied_fric[0].endswith("for _asset in (self.robot, self._fixture)]"),
          str(_applied_fric)[:300])
    _applied_triggers = [
        (_unparse(n.test), [_unparse(s) for s in n.body])
        for n in _walk(_func(env_tree, "_get_observations", "InsertionEnv"))
        if isinstance(n, ast.If) and "_write_applied_csv" in _unparse(n)
    ]
    _applied_wait = [
        (_unparse(s.test), type(s.body[0]).__name__)
        for s in _applied_fn.body[1:2] if isinstance(s, ast.If)
    ]
    check("applied.csv is written in table mode only, once, and only after a physics step",
          _applied_triggers == [(
              "self._eval_table is not None and (not self._applied_written)",
              ["self._write_applied_csv()"],
          )]
          and _applied_wait == [("int(self.common_step_counter) < 1", "Return")],
          f"{_applied_triggers} / {_applied_wait}")
    _applied_flag = [
        _unparse(n) for n in _walk(_applied_fn)
        if isinstance(n, ast.Assign) and _unparse(n.targets[0]) == "self._applied_written"
    ]
    check("the applied.csv writer raises its own written flag",
          _applied_flag == ["self._applied_written = True"], str(_applied_flag))
    _applied_first_if = [
        s for s in _applied_fn.body if isinstance(s, ast.If) and "log_dir" in _unparse(s.test)
    ][:1]
    check("applied.csv is refused without a run folder, not skipped",
          [(_unparse(s.test), type(s.body[0]).__name__) for s in _applied_first_if]
          == [("not log_dir", "Raise")],
          str([_unparse(s.test) for s in _applied_first_if]))
    play_tree = _parse(PLAY_SRC, play_mut)
    _play_main = _func(play_tree, "main")
    _em_tree = _parse(EVAL_MANIFEST_SRC, ())
    _manifest_fields = _module_const(_em_tree, "MANIFEST_FIELDS") or ()
    _git_keys = [
        tuple(k.value for k in n.value.keys)
        for n in _walk(_func(_em_tree, "git_commits"))
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)
    ]
    _manifest_calls = [
        n for n in _walk(_play_main) if isinstance(n, ast.Call)
        and _unparse(n.func) == "_em.build_manifest"
    ]
    _manifest_kw = [
        ([k.arg for k in n.keywords if k.arg is not None],
         [_unparse(k.value) for k in n.keywords if k.arg is None])
        for n in _manifest_calls
    ]
    check("play.py hands build_manifest EXACTLY the manifest's field list",
          len(_manifest_kw) == 1 and len(_git_keys) == 1
          and _manifest_kw[0][1] == ["_em.git_commits(_repo)"]
          and sorted(_manifest_kw[0][0] + list(_git_keys[0])) == sorted(_manifest_fields)
          and len(_manifest_fields) == 26,
          f"{_manifest_kw} + {_git_keys} vs {len(_manifest_fields)} fields")
    _play_assigns = {
        _unparse(n.targets[0]): _unparse(n.value)
        for n in _walk(_play_main) if isinstance(n, ast.Assign) and len(n.targets) == 1
        and _unparse(n.targets[0]).startswith(("env_cfg.dr_", "_norm_before"))
    }
    check("the evaluation run fixes dr_mode, the table path and the width on the cfg",
          {k: v for k, v in _play_assigns.items() if k.startswith("env_cfg.dr_")} == {
              "env_cfg.dr_mode": "'table'",
              "env_cfg.dr_table_path": "_table_path",
              "env_cfg.dr_table_width": "args_cli.eval_width",
          }, str(_play_assigns))
    _play_num_envs = [
        _unparse(n.value) for n in _walk(_play_main) if isinstance(n, ast.Assign)
        and _unparse(n.targets[0]) == "env_cfg.scene.num_envs"
    ]
    check("the evaluation run takes num_envs from the table's row count",
          _play_num_envs[1:] == ["_table_rows"], str(_play_num_envs))
    _play_if_tests = [_unparse(n.test) for n in _walk(_play_main) if isinstance(n, ast.If)]
    check("an evaluation without --seed is refused (G8)",
          _play_if_tests.count("args_cli.seed is None") == 1, "")
    check("an evaluation folder that already holds episodes.csv is refused",
          _play_if_tests.count(
              "os.path.exists(os.path.join(log_dir, replay_dirname, 'episodes.csv'))") == 1, "")
    _play_stop = [
        n for n in _walk(_play_main) if isinstance(n, ast.If)
        and _unparse(n.test) == "bool(_eval_first_done.all())"
        and any(isinstance(s, ast.Break) for s in n.body)
    ]
    check("the evaluation stops once ALL envs have finished their first episode",
          len(_play_stop) == 1
          and "_eval_first_done |= dones.to(torch.bool)" in _unparse(_play_main), "")
    check("the normaliser is hashed before the run and again for the manifest (G5)",
          _play_assigns.get("_norm_before")
          == "_normalizer_sha256(runner) if _em is not None else None"
          and any(k.arg == "normalizer_sha256_after"
                  and _unparse(k.value) == "_normalizer_sha256(runner)"
                  for n in _manifest_calls for k in n.keywords),
          str(_play_assigns.get("_norm_before")))
    _manifest_blocks = [
        n for n in _walk(_play_main) if isinstance(n, ast.If)
        and _unparse(n.test) == "_em is not None"
        and "_em.write_manifest(" in _unparse(n)
    ]
    check("the evaluation ends the process right after the manifest",
          len(_manifest_blocks) == 1
          and _unparse(_manifest_blocks[0].body[-1]) == "os._exit(0)",
          str([_unparse(b.body[-1]) for b in _manifest_blocks]))
    _mem_assign = [_unparse(a.value) for a in _walk(_play_main) if isinstance(a, ast.Assign)
                   and _unparse(a.targets[0]) == "_mem_reset"]
    _mem_ifs = [n for n in _walk(_play_main) if isinstance(n, ast.If)
                and _unparse(n.test) == "version.parse(installed_version) >= version.parse('4.0.0')"
                and [_unparse(s) for s in n.body] == ["policy.reset(_mem_reset)"]
                and [_unparse(s) for s in n.orelse] == ["policy_nn.reset(_mem_reset)"]]
    check("play.py resets the recurrent memory after every step: done envs, or all under the switch",
          _mem_assign == ["torch.ones_like(dones) if args_cli.memory_reset_every_step else dones"]
          and len(_mem_ifs) == 1, f"{_mem_assign}, {len(_mem_ifs)} reset block(s)")
    check("--memory-reset-every-step exists and is off by default",
          "--memory-reset-every-step" in _cli_flags(play_tree)
          and _cli_default(play_tree, "--memory-reset-every-step") is False, "")
    _am = [_unparse(k.value) for n in _manifest_calls for k in n.keywords if k.arg == "action_mode"]
    check("the manifest's action_mode says which memory reset ran",
          len(_am) == 1 and "args_cli.memory_reset_every_step" in _am[0], str(_am))
    rlg_tree = _parse(RLG_PLAY_SRC, rlg_play_mut)
    _rlg_main = _func(rlg_tree, "main")
    _rlg_manifest_calls = [
        n for n in _walk(_rlg_main) if isinstance(n, ast.Call)
        and _unparse(n.func) == "_em.build_manifest"
    ]
    _rlg_manifest_kw = [
        ([k.arg for k in n.keywords if k.arg is not None],
         [_unparse(k.value) for k in n.keywords if k.arg is None])
        for n in _rlg_manifest_calls
    ]
    check("rl_games play.py hands build_manifest EXACTLY the manifest's field list",
          len(_rlg_manifest_kw) == 1 and len(_git_keys) == 1
          and _rlg_manifest_kw[0][1] == ["_em.git_commits(_repo)"]
          and sorted(_rlg_manifest_kw[0][0] + list(_git_keys[0])) == sorted(_manifest_fields),
          f"{_rlg_manifest_kw} + {_git_keys}")
    _rlg_eval_ifs = [
        n for n in _walk(_rlg_main) if isinstance(n, ast.If)
        and _unparse(n.test) == "args_cli.eval_table is not None"
    ]
    _rlg_eval_assigns = [
        (_unparse(s.targets[0]), _unparse(s.value))
        for n in _rlg_eval_ifs for s in _walk(n)
        if isinstance(s, ast.Assign) and len(s.targets) == 1
        and _unparse(s.targets[0]).startswith(("env_cfg.dr_", "env_cfg.scene.num_envs"))
    ]
    check("rl_games play.py: the evaluation run fixes dr_mode, the table path, the width and the OOD switch",
          len(_rlg_eval_ifs) == 1
          and sorted(a for a in _rlg_eval_assigns if a[0].startswith("env_cfg.dr_")) == sorted([
              ("env_cfg.dr_mode", "'table'"),
              ("env_cfg.dr_table_path", "_table_path"),
              ("env_cfg.dr_table_width", "args_cli.eval_width"),
              ("env_cfg.dr_table_allow_outside", "bool(args_cli.eval_ood)"),
          ]), str(_rlg_eval_assigns))
    check("rl_games play.py: the evaluation run takes num_envs from the table's row count",
          [a for a in _rlg_eval_assigns if a[0] == "env_cfg.scene.num_envs"]
          == [("env_cfg.scene.num_envs", "_table_rows")], str(_rlg_eval_assigns))
    _rlg_eval_tests = [_unparse(s.test) for n in _rlg_eval_ifs for s in _walk(n) if isinstance(s, ast.If)]
    _rlg_refusal_exit = [
        s for n in _rlg_eval_ifs for s in _walk(n) if isinstance(s, ast.If)
        and _unparse(s.test) == "_eval_refusals" and _unparse(s.body[-1]) == "raise SystemExit(2)"
    ]
    check("rl_games play.py: an evaluation without a fixed seed or a named checkpoint is refused (G8)",
          all(t in _rlg_eval_tests for t in ("args_cli.seed is None", "args_cli.seed == -1",
                                             "not args_cli.checkpoint"))
          and len(_rlg_refusal_exit) == 1, str(_rlg_eval_tests))
    _rlg_flag_loops = sorted(
        (tuple(e.value for e in s.iter.elts if isinstance(e, ast.Constant)), _unparse(s.body[0].test))
        for n in _rlg_eval_ifs for s in _walk(n)
        if isinstance(s, ast.For) and isinstance(s.iter, ast.Tuple)
        and s.body and isinstance(s.body[0], ast.If)
    )
    check("rl_games play.py: the views, sampling, a step cap and a picked checkpoint are refused with --eval-table",
          _rlg_flag_loops == sorted([
              (("dr_table", "max_steps"), "getattr(args_cli, _flag) is not None"),
              (("use_last_checkpoint", "use_pretrained_checkpoint", "autodr_state", "video", "stochastic"),
               "getattr(args_cli, _flag)"),
          ]), str(_rlg_flag_loops))
    _rlg_orelse = [
        _unparse(s.test) for n in _rlg_eval_ifs for s in n.orelse
        if isinstance(s, ast.If) and _unparse(s.body[-1]) == "raise SystemExit(2)"
    ]
    check("rl_games play.py: --eval-width / --eval-tag / --eval-ood without --eval-table are refused",
          _rlg_orelse == ["args_cli.eval_tag or args_cli.eval_width != 'target' or args_cli.eval_ood"],
          str(_rlg_orelse))
    _rlg_dir_assigns = [
        (_unparse(n.targets[0]), _unparse(n.value)) for n in _walk(_rlg_main)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and _unparse(n.targets[0]) in ("replay_dirname", "env_cfg.log_dir")
    ]
    _rlg_if_tests = [_unparse(n.test) for n in _walk(_rlg_main) if isinstance(n, ast.If)]
    check("rl_games play.py: the evaluation writes into <run>/eval_<width>_<sha8>[_<tag>] and never into an old one",
          ("replay_dirname", "_em.eval_dirname(args_cli.eval_width, _table_sha, args_cli.eval_tag)")
          in _rlg_dir_assigns
          and [a for a in _rlg_dir_assigns if a[0] == "env_cfg.log_dir"]
          == [("env_cfg.log_dir", "os.path.join(log_dir, replay_dirname)")]
          and _rlg_if_tests.count(
              "os.path.exists(os.path.join(log_dir, replay_dirname, 'episodes.csv'))") == 1,
          str(_rlg_dir_assigns))
    _rlg_stop = [
        n for n in _walk(_rlg_main) if isinstance(n, ast.If)
        and _unparse(n.test) == "bool(_eval_first_done.all())"
        and any(isinstance(s, ast.Break) for s in n.body)
    ]
    check("rl_games play.py: the evaluation stops once ALL envs have finished their first episode",
          len(_rlg_stop) == 1
          and "_eval_first_done |= dones.to(torch.bool)" in _unparse(_rlg_main), "")
    _rlg_norm_before = [
        (n.lineno, _unparse(n.value)) for n in _walk(_rlg_main)
        if isinstance(n, ast.Assign) and _unparse(n.targets[0]) == "_norm_before"
    ]
    _rlg_restore = [n.lineno for n in _walk(_rlg_main)
                    if isinstance(n, ast.Call) and _unparse(n.func) == "agent.restore"]
    check("rl_games play.py: the normaliser is hashed after the restore and again for the manifest (G5)",
          [v for _l, v in _rlg_norm_before] == ["_normalizer_sha256(agent) if _em is not None else None"]
          and len(_rlg_restore) == 1 and _rlg_norm_before[0][0] > _rlg_restore[0]
          and any(k.arg == "normalizer_sha256_after" and _unparse(k.value) == "_normalizer_sha256(agent)"
                  for n in _rlg_manifest_calls for k in n.keywords),
          str(_rlg_norm_before))
    check("rl_games play.py: the G5 hash reads agent.model.running_mean_std",
          "getattr(getattr(agent, 'model', None), 'running_mean_std', None)"
          in _unparse(_func(rlg_tree, "_normalizer_sha256")), "")
    _rlg_manifest_blocks = [
        n for n in _walk(_rlg_main) if isinstance(n, ast.If)
        and _unparse(n.test) == "_em is not None"
        and "_em.write_manifest(" in _unparse(n)
    ]
    check("rl_games play.py: the evaluation ends the process right after the manifest",
          len(_rlg_manifest_blocks) == 1
          and _unparse(_rlg_manifest_blocks[0].body[-1]) == "os._exit(0)",
          str([_unparse(b.body[-1]) for b in _rlg_manifest_blocks]))
    _rlg_mem_assign = [_unparse(a.value) for a in _walk(_rlg_main) if isinstance(a, ast.Assign)
                       and _unparse(a.targets[0]) == "_mem_reset"]
    _rlg_mem_zero = [_unparse(a) for a in _walk(_rlg_main) if isinstance(a, ast.Assign)
                     and _unparse(a.targets[0]).startswith("s[")]
    check("rl_games play.py resets the recurrent memory after every step: done envs, or all under the switch",
          _rlg_mem_assign == ["torch.ones_like(dones) if args_cli.memory_reset_every_step else dones"]
          and _rlg_mem_zero == ["s[:, _mem_reset, :] = 0.0"], f"{_rlg_mem_assign}, {_rlg_mem_zero}")
    check("rl_games play.py: --memory-reset-every-step exists and is off by default",
          "--memory-reset-every-step" in _cli_flags(rlg_tree)
          and _cli_default(rlg_tree, "--memory-reset-every-step") is False, "")
    _rlg_am = [_unparse(k.value) for n in _rlg_manifest_calls for k in n.keywords if k.arg == "action_mode"]
    check("rl_games play.py: the manifest's action_mode says which memory reset ran",
          len(_rlg_am) == 1 and "args_cli.memory_reset_every_step" in _rlg_am[0], str(_rlg_am))
    _rlg_noise_keys = [
        e.value for n in _rlg_manifest_calls for k in n.keywords
        if k.arg == "noise" and isinstance(k.value, ast.DictComp)
        for e in getattr(k.value.generators[0].iter, "elts", []) if isinstance(e, ast.Constant)
    ]
    check("rl_games play.py: the manifest's noise names the yaw belief and the z spread, all cfg fields",
          {"obs_noise_pocket_pos_std_m", "grasp_obs_offset_x_m", "pocket_obs_yaw_rad",
           "fixture_pos_noise_xy", "fixture_pos_noise_z"} <= set(_rlg_noise_keys)
          and all(_class_attr_expr(cfg_tree, "InsertionEnvCfg", k) is not None for k in _rlg_noise_keys),
          str(_rlg_noise_keys))
    _centre_calls = [
        n for n in _walk(_env_init) if isinstance(n, ast.Call)
        and _unparse(n.func).endswith("bind_centres")
    ]
    _centre_pairs = sorted({
        (k.value, _unparse(v))
        for c in _centre_calls for a in c.args if isinstance(a, ast.Dict)
        for k, v in zip(a.keys, a.values)
        if isinstance(k, ast.Constant)
    })
    _centres = (
        dict(_centre_pairs) if len({k for k, _ in _centre_pairs}) == len(_centre_pairs)
        else _centre_pairs
    )
    check("the AutoDR centres are bound to the five named sources",
          _centres == {
              "lat_r": "0.0",
              "yaw": "0.0",
              "tilt": "0.0",
              "start_height": "float(cfg.start_tip_above_entrance)",
              "friction": "insertion_tasks_cfg.CONTACT_FRICTION",
          },
          str(_centres))

    _floor_default = _class_attr_expr(cfg_tree, "InsertionEnvCfg", "start_floor_m")
    _floor_steps = _class_attr_const(cfg_tree, "InsertionEnvCfg", "start_floor_steps")
    check("start_floor_m is a float cfg field defaulting to -0.030 m, "
          "and start_floor_steps is 5",
          _floor_default is not None
          and ast.literal_eval(_floor_default) == -0.030
          and _floor_steps == 5,
          f"{None if _floor_default is None else _unparse(_floor_default)}, {_floor_steps}")
    _floor_raises = [
        _unparse(n) for n in _walk(_env_init) if isinstance(n, ast.Raise)
        and "start_floor_m" in _unparse(n)
    ]
    check("the floor refuses five ways: above the high, the success band, a second lower edge, "
          "reset noise, and angles or offset under dr_mode='off'",
          len(_floor_raises) == 5
          and any("depth_min" in r for r in _floor_raises)
          and any("two lower" in r for r in _floor_raises)
          and any("reset_joint_noise" in r for r in _floor_raises)
          and any("dr_mode='off'" in r and "RT-120" in r for r in _floor_raises),
          f"{len(_floor_raises)} raises")
    _autodr_calls = [
        n for n in _walk(_env_init) if isinstance(n, ast.Call)
        and _unparse(n.func) == "autodr.AutoDR"
    ]
    _floor_kw = [
        [_unparse(k.value) for k in c.keywords if k.arg == "floor"] for c in _autodr_calls
    ]
    _nodr_dims = [
        _unparse(k.value) for c in _autodr_calls for a in c.args
        if isinstance(a, ast.Call) for k in a.keywords if k.arg == "dims"
    ]
    check("both AutoDR providers take the floor, and the No-DR one carries only start_height",
          len(_autodr_calls) == 2 and _floor_kw == [["_floor_spec"], ["_floor_spec"]]
          and _nodr_dims == ["tuple((d for d in autodr.DR_DIMS if d.name == 'start_height'))"],
          f"{_floor_kw}, dims {_nodr_dims}")
    _stall_kw = [
        [_unparse(k.value) for k in c.keywords if k.arg == "stall_buffers"]
        for c in _autodr_calls
    ]
    _stall_default = _class_attr_const(cfg_tree, "InsertionEnvCfg", "autodr_stall_buffers")
    check("autodr_stall_buffers is an int cfg field defaulting to 3",
          _stall_default == 3, str(_stall_default))
    check("both AutoDR providers take the cfg stall bar",
          _stall_kw == [["int(cfg.autodr_stall_buffers)"]] * 2, str(_stall_kw))
    _report_fn = _func(env_tree, "_print_startup_report", "InsertionEnv")
    _report_loops = [
        _unparse(n.iter) for n in _walk(_report_fn) if isinstance(n, ast.For)
        and "_b[_name]" in _unparse(n)
    ]
    check("the startup report's band table walks the provider's own dims",
          _report_loops == ["(d.name for d in self._dr.dims)"], str(_report_loops))
    _metrics_fn = _func(env_tree, "_write_metrics", "InsertionEnv")
    _metric_keys = {
        k.value for d in _walk(_metrics_fn) if isinstance(d, ast.Dict)
        for k in d.keys if isinstance(k, ast.Constant)
    }
    check("the metrics dump carries start_floor_m and start_floor_steps",
          {"start_floor_m", "start_floor_steps"} <= _metric_keys, "")
    check("the metrics dump carries autodr_stall_buffers",
          "autodr_stall_buffers" in _metric_keys, "")
    _floor_tags = [
        _unparse(n) for n in _walk(train_tree) if isinstance(n, ast.Call)
        and _unparse(n).startswith("parts.append(f'floor")
    ]
    check("the run tag names the floor in its own if, beside the start tag",
          len(_floor_tags) == 1, str(_floor_tags))
    _fric_centre = _module_const(tasks_tree, "CONTACT_FRICTION")
    _fric_band = _autodr_reach(autodr_mut).get("friction")
    check("the contact-friction centre lies inside the DR_DIMS friction band",
          _fric_centre is not None and _fric_band is not None
          and float(_fric_band[0]) < float(_fric_centre) < float(_fric_band[1]),
          f"CONTACT_FRICTION {_fric_centre} vs band {_fric_band}")
    check("dr_mode='autodr' together with a static range is refused",
          bool(_static_guards),
          "; ".join(_unparse(g.test) for g in _static_guards))
    _metrics_fn = _func(env_tree, "_write_metrics", "InsertionEnv")
    _metrics_pairs = {
        k.value: _unparse(v)
        for d in _walk(_metrics_fn) if isinstance(d, ast.Dict)
        for k, v in zip(d.keys, d.values)
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
        and k.value.startswith("dr_")
    }
    check("the metrics dump records which source randomised the run",
          set(_metrics_pairs) == {"dr_mode", "dr_state", "dr_fresh"}
          and _metrics_pairs["dr_mode"] == "str(self.cfg.dr_mode)"
          and _metrics_pairs["dr_state"]
          == "None if self._dr is None else self._dr.as_dict()",
          str(_metrics_pairs))
    _fresh_src = _metrics_pairs.get("dr_fresh", "")
    check("the metrics dump carries the fresh window, its minimum and all_at_max",
          "'n': len(self._fresh_successes)" in _fresh_src
          and "'min': self._fresh_min" in _fresh_src
          and "'all_at_max': self._dr.all_at_max()" in _fresh_src
          and "'boundary_n': len(self._recent_boundary_successes)" in _fresh_src,
          _fresh_src[:90])
    check("the dumped fresh rate is None until the window holds the minimum",
          "if len(self._fresh_successes) >= self._fresh_min else None" in _fresh_src,
          _fresh_src[:90])
    _report_fn = _func(env_tree, "_print_startup_report", "InsertionEnv")
    _report_dr_ifs = [
        _unparse(n.test) for n in _walk(_report_fn) if isinstance(n, ast.If)
        and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "bounds_max" for c in _walk(n))
    ]
    check("the startup report names dr_mode and the provider bounds",
          "dr_mode: " in _string_constants(_report_fn)
          and _report_dr_ifs == ["self._dr is not None"],
          f"{[t for t in _string_constants(_report_fn) if 'dr_mode' in t]}, "
          f"guards {_report_dr_ifs}")
    _wired_lines = [
        t for t in _string_constants(_report_fn) if "ARE WIRED" in t
    ]
    check("the startup report says all five DR quantities are wired",
          len(_wired_lines) == 1
          and "ALL FIVE" in _wired_lines[0]
          and "_live_bounds()" in _wired_lines[0]
          and "BOUNDARY SAMPLING is on" in _wired_lines[0],
          str([t[:70] for t in _wired_lines]))
    check("the provider block stamps itself NOT VERIFIED IN A SIMULATOR",
          any("NOT VERIFIED IN A SIMULATOR" in t
              for t in _string_constants(_report_fn)),
          "")
    _z_cmds = [
        n for n in _walk(_solve_fn) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Subscript) and "d_pocket" in _unparse(t)
                and _unparse(t).endswith("[:, 2]") for t in n.targets)
    ]
    check("the start-pose solve commands the per-env height",
          bool(_z_cmds) and all(
              any(isinstance(sub, ast.Attribute) and sub.attr == "_start_height"
                  for sub in _walk(a.value)) for a in _z_cmds),
          "; ".join(_unparse(a) for a in _z_cmds))
    _init_fn = _func(env_tree, "__init__", "InsertionEnv")
    _guards = [
        n for n in _walk(_init_fn) if isinstance(n, ast.If)
        and any(isinstance(b, ast.Raise) for b in n.body)
    ]
    _band_guard = [
        g for g in _guards
        if "_low" in _unparse(g.test) and "depth_min" in _unparse(g.test)
    ]
    check("the lower start bound is guarded above the success band",
          bool(_band_guard),
          "; ".join(_unparse(g.test) for g in _band_guard))
    _orient_names = ("reset_joint_noise", "reset_yaw_noise", "_angle_reach")
    _orient_guard = [
        g for g in _guards
        if "_low < 0.0" in _unparse(g.test)
        and all(name in _unparse(g.test) for name in _orient_names)
    ]
    check("a start inside the pocket refuses every orientation noise and the angle reach",
          bool(_orient_guard),
          "missing: " + ", ".join(
              name for name in _orient_names
              if not any(name in _unparse(g.test) for g in _guards if "_low < 0.0" in _unparse(g.test))
          ))
    check("the start-pose solve seeds the gated max depth with the start depth",
          "_max_depth" in _solve_writes,
          f"written: {sorted(w for w in _solve_writes if w.startswith('_'))}")
    _map_fn = _func(env_tree, "_map_start_conditions", "InsertionEnv")
    _lat_writes = [
        n for n in _walk(_map_fn) if isinstance(n, ast.Assign)
        and any("_start_lat_off" in _unparse(t) for t in n.targets)
    ]
    _lat_guarded = [
        n for n in _walk(_map_fn) if isinstance(n, ast.If)
        and any(w in _walk(n) for w in _lat_writes)
    ]
    check("the lateral start row is written at EVERY reset, behind no guard",
          len(_lat_writes) == 1 and not _lat_guarded,
          f"{len(_lat_writes)} write(s), guards "
          f"{[_unparse(g.test) for g in _lat_guarded]}")
    _lat_src = _unparse(_lat_writes[0]) if _lat_writes else ""
    check("the lateral draw maps the radius and angle columns onto a disk of the LIVE radius",
          _lat_src == "self._start_lat_off[idx] = insertion_math.disk_offset("
                      "u[:, self._col['lat_r']], u[:, self._col['lat_phi']], "
                      "float(_b['lat_r'][1]))",
          f"write {_lat_src}")
    _lat_resolves = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._start_lat_half" for t in a.targets)
    ]
    _raw_reads = [
        n for n in _walk(env_tree)
        if isinstance(n, ast.Attribute) and n.attr == "start_lateral_offset"
    ]
    check("the env resolves start_lateral_offset once and reads it nowhere else",
          _lat_resolves == ["resolve_start_lateral_offset(cfg.start_lateral_offset)"]
          and len(_raw_reads) == 1,
          f"resolved {_lat_resolves}, raw reads {len(_raw_reads)}")
    _lat_default = _class_attr_expr(cfg_tree, "InsertionEnvCfg", "start_lateral_offset")
    check("start_lateral_offset defaults to a bare number, so BOTH hydra forms merge",
          _lat_default is not None and _unparse(_lat_default) == "0.0",
          _unparse(_lat_default) if _lat_default is not None else "missing")
    _lat_tests = [
        _unparse(n.test) for n in _walk(_env_init) if isinstance(n, ast.If)
        and "_start_lat_half" in _unparse(n.test)
    ]
    check("both lateral guards test the radius itself",
          _lat_tests == ["_low < 0.0 and self._start_lat_half != 0.0",
                         "self._start_lat_half != 0.0"],
          str(_lat_tests))
    _lat_report = [
        _unparse(n) for n in _walk(_env_init)
        if isinstance(n, ast.JoinedStr) and "DISK of radius" in _unparse(n)
    ]
    check("the startup report names the lateral radius in millimetres",
          len(_lat_report) == 1
          and "DISK of radius {self._start_lat_half * 1000.0:.3f} mm" in _lat_report[0],
          str(_lat_report))
    _lat_metric = [
        _unparse(v) for d in _walk(_metrics_fn) if isinstance(d, ast.Dict)
        for k, v in zip(d.keys, d.values)
        if isinstance(k, ast.Constant) and k.value == "start_lateral_offset_m"
    ]
    check("the metrics dump records the lateral radius as one number",
          _lat_metric == ["self._start_lat_half"], str(_lat_metric))
    _pocket_goals = [
        _unparse(a) for a in _walk(_solve_fn) if isinstance(a, ast.Assign)
        and any("d_pocket[:, " in _unparse(t) for t in a.targets)
        and "_start_lat_off" in _unparse(a.value)
    ]
    check("the start-pose goal spends the lateral draw axis for axis",
          _pocket_goals == [
              "d_pocket[:, 0] = self._start_lat_off[:, 0] - tip_rel[:, 0]",
              "d_pocket[:, 1] = self._start_lat_off[:, 1] - tip_rel[:, 1]",
          ],
          str(_pocket_goals))
    _train_src = _unparse(train_tree)
    _lat_tags = [
        _unparse(a) for c in _walk(train_tree)
        if isinstance(c, ast.Call) and _unparse(c.func) == "parts.append"
        for a in c.args
        if isinstance(a, ast.JoinedStr) and "lat" in _unparse(a)
    ]
    _lat_tag_ifs = [
        _unparse(n.test) for n in _walk(train_tree) if isinstance(n, ast.If)
        and any("lat" in _unparse(b) for b in n.body)
        and "_lat" in _unparse(n.test)
    ]
    check("the run tag names the lateral radius",
          "resolve_start_lateral_offset" in _train_src
          and _lat_tags == ["f'lat{_lat * 1000:g}mm'"]
          and _lat_tag_ifs == ["_lat > 0.0"],
          f"{_lat_tags}, guard {_lat_tag_ifs}")
    _fric_fn = _func(env_tree, "_apply_friction", "InsertionEnv")
    _fric_src = _unparse(_fric_fn)
    _fric_calls = [
        n for n in _walk(_reset_fn) if isinstance(n, ast.If)
        and "_apply_friction" in _unparse(n)
    ]
    _fric_reach_assigns = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._friction_reach" for t in a.targets)
    ]
    _fric_reach_src = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_fric" for t in a.targets)
    ]
    check("the per-reset friction write is gated by the REACH, not by today's band",
          len(_fric_calls) == 1
          and _unparse(_fric_calls[0].test) == "ready and self._friction_reach"
          and _fric_reach_src == [
              "self._dr.bounds_max().get('friction', (insertion_tasks_cfg.CONTACT_FRICTION, "
              "insertion_tasks_cfg.CONTACT_FRICTION))"
          ],
          f"guard {[_unparse(c.test) for c in _fric_calls]}, "
          f"reach from {_fric_reach_src}")
    check("dr_mode='off' has no friction reach",
          _fric_reach_assigns == ["_fric[1] - _fric[0] > 0.0", "False"],
          str(_fric_reach_assigns))
    _fric_bounds = [
        _unparse(a.value) for a in _walk(_fric_fn) if isinstance(a, ast.Assign)
        and _unparse(a.targets[0]) == "(lo, hi)"
    ]
    check("the friction draw maps this reset onto the LIVE band",
          _fric_bounds == ["self._live_bounds()['friction']"]
          and "bounds_max" not in _fric_src,
          str(_fric_bounds))
    _fric_mu = [
        _unparse(a.value) for a in _walk(_fric_fn) if isinstance(a, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "mu" for t in a.targets)
    ]
    check("the friction value is mapped off the friction column",
          _fric_mu == ["autodr.map_unit_to_bounds(self._reset_unit[idx]"
                       "[:, self._col['friction']], float(lo), float(hi))"],
          str(_fric_mu))
    _fric_cols = [
        _unparse(a) for a in _walk(_fric_fn) if isinstance(a, ast.Assign)
        and "_buf[idx_cpu" in _unparse(a)
    ]
    check("static and dynamic friction get the SAME drawn value",
          _fric_cols == ["_buf[idx_cpu, :, 0] = _col", "_buf[idx_cpu, :, 1] = _col"],
          str(_fric_cols))
    _fric_cache = [
        _unparse(n) for n in _walk(_env_init) if isinstance(n, ast.For)
        and "_friction_bufs.append" in _unparse(n)
    ]
    _fric_loops = [
        _unparse(n.iter) for n in _walk(_fric_fn) if isinstance(n, ast.For)
    ]
    check("the friction write covers the robot AND the fixture",
          len(_fric_cache) == 1
          and "for _asset in (self.robot, self._fixture)" in _fric_cache[0]
          and _fric_loops == ["self._friction_bufs"],
          f"cached {len(_fric_cache)}, loop {_fric_loops}")
    _fric_pushes = [
        _unparse(n) for n in _walk(_fric_fn) if isinstance(n, ast.Call)
        and _unparse(n.func).endswith("set_material_properties")
    ]
    check("the material push hands over the WHOLE buffer plus the subset of ids",
          _fric_pushes == ["_asset.root_physx_view.set_material_properties(_buf, idx_cpu)"],
          str(_fric_pushes))
    _fric_gets = [
        n for n in _walk(env_tree) if isinstance(n, ast.Call)
        and _unparse(n.func).endswith("get_material_properties")
    ]
    _fric_gets_in_apply = [
        n for n in _walk(_fric_fn) if isinstance(n, ast.Call)
        and _unparse(n.func).endswith("get_material_properties")
    ]
    check("the reset never re-reads the material buffer from PhysX",
          not _fric_gets_in_apply and len(_fric_gets) == 4,
          f"{len(_fric_gets_in_apply)} in _apply_friction, {len(_fric_gets)} in the env")
    _fric_body = list(_fric_fn.body)
    _t0_at = [i for i, n in enumerate(_fric_body)
              if isinstance(n, ast.Assign) and _unparse(n.value) == "time.perf_counter()"]
    _loop_at = [i for i, n in enumerate(_fric_body) if isinstance(n, ast.For)]
    _append_src = [_unparse(n) for n in _walk(_fric_fn)
                   if isinstance(n, ast.Call) and "_friction_write_ms.append" in _unparse(n.func)]
    check("the write-cost timer spans the whole block, both pushes inside",
          len(_t0_at) == 1 and len(_loop_at) == 1 and _t0_at[0] < _loop_at[0]
          and _append_src == ["self._friction_write_ms.append("
                              "(time.perf_counter() - t0) * 1000.0)"],
          f"t0 at {_t0_at}, loop at {_loop_at}, append {_append_src}")
    _fric_applied = [
        _unparse(a) for a in _walk(_fric_fn) if isinstance(a, ast.Assign)
        and any("_friction_applied" in _unparse(t) for t in a.targets)
    ]
    check("the commanded friction is recorded per env, from the same draw",
          _fric_applied == ["self._friction_applied[idx] = mu"],
          str(_fric_applied))
    _report_fn = _func(env_tree, "_print_startup_report", "InsertionEnv")
    _report_src = _unparse(_report_fn)
    _fric_read = [
        _unparse(a.value) for a in _walk(_report_fn) if isinstance(a, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_read" for t in a.targets)
    ]
    _fric_read_f = [
        _unparse(a.value) for a in _walk(_report_fn) if isinstance(a, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_read_f" for t in a.targets)
    ]
    check("the startup report reads the applied friction back PER ENV, both assets",
          _fric_read == ["[float(_fric_r[_e, :, 0].mean()) for _e in range(_n_rep)]"]
          and _fric_read_f == ["[float(_fric_f[_e, :, 0].mean()) for _e in range(_n_rep)]"]
          and "friction after first reset env0.." in _report_src
          and "friction write cost mean/max ms" in _report_src,
          f"robot {_fric_read}, fixture {_fric_read_f}")
    check("the startup report reads the per-episode observation scatter back",
          "_observation_noise_model" in _report_src
          and "_grasp_obs_off" in _report_src
          and "observation bias env0.." in _report_src
          and "grasp belief error env0.." in _report_src,
          f"bias printed: {'observation bias env0..' in _report_src}, "
          f"belief error printed: {'grasp belief error env0..' in _report_src}")
    _report_slice_keys = [
        n.slice.value for n in _walk(_report_fn)
        if isinstance(n, ast.Subscript)
        and _unparse(n.value).endswith("OBS_SLICES")
        and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)
    ]
    _report_bias_reads = [
        _unparse(n) for n in _walk(_report_fn)
        if isinstance(n, ast.Subscript) and _unparse(n.value).endswith("_bias")
    ]
    check("the startup report slices the bias with the tip_rel entry of OBS_SLICES",
          _report_slice_keys == ["tip_rel"]
          and _report_bias_reads == ["_noise._bias[:_n_obs, _t_lo:_t_hi]"],
          f"OBS_SLICES keys in the report: {_report_slice_keys}, "
          f"bias reads: {_report_bias_reads}")
    _prov_loops = [n for n in _walk(_report_fn) if isinstance(n, ast.For)
                   and "provenance_lines" in _unparse(n.iter)]
    _prov_src = ("\n".join(_unparse(s) for s in _prov_loops[0].body)
                 if len(_prov_loops) == 1 else "")
    check("the provenance lines are printed under a neutral prefix, not a [TESTWERT] one",
          len(_prov_loops) == 1 and "TESTWERT" not in _prov_src,
          f"{len(_prov_loops)} loop(s): {_prov_src}")
    _fric_log_keys = {
        k.value for d in _walk(_func(env_tree, "_log_finished_episodes", "InsertionEnv"))
        if isinstance(d, ast.Dict) for k in d.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    check("the write cost reaches the log as mean AND max",
          {"dr/friction_write_ms_mean", "dr/friction_write_ms_max"} <= _fric_log_keys,
          str(sorted(k for k in _fric_log_keys if "friction" in k)))
    _fric_cache_src = _fric_cache[0] if _fric_cache else ""
    _fric_dtype_assigns = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._friction_buf_dtype" for t in a.targets)
    ]
    check("the cached material buffers are CLONED, not aliased",
          ".get_material_properties().clone()" in _fric_cache_src,
          _fric_cache_src.splitlines()[1] if _fric_cache_src else "no cache loop")
    check("the one-shot D-111 write fills the two FRICTION columns",
          "_mats[..., 0] = insertion_tasks_cfg.CONTACT_FRICTION" in _fric_cache_src
          and "_mats[..., 1] = insertion_tasks_cfg.CONTACT_FRICTION" in _fric_cache_src
          and "_mats[..., 2]" not in _fric_cache_src,
          _fric_cache_src)
    check("both material buffers are pinned to ONE dtype at cache time",
          _fric_dtype_assigns == ["next(iter(_dtypes))"]
          and "_dtypes = {_b.dtype for _, _b in self._friction_bufs}" in _unparse(_env_init),
          str(_fric_dtype_assigns))
    _fric_idx_cpu = [
        _unparse(a.value) for a in _walk(_fric_fn) if isinstance(a, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "idx_cpu" for t in a.targets)
    ]
    check("the push index is moved to the HOST as int64",
          _fric_idx_cpu == ["idx.to(device='cpu', dtype=torch.long)"],
          str(_fric_idx_cpu))
    _fric_col_assigns = [
        a for a in _walk(_fric_fn) if isinstance(a, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_col" for t in a.targets)
    ]
    _fric_loop = [n for n in _walk(_fric_fn) if isinstance(n, ast.For)]
    _col_in_loop = [
        a for a in _fric_col_assigns
        if _fric_loop and any(a is b for b in _walk(_fric_loop[0]))
    ]
    check("the drawn column is copied to the host ONCE, outside the asset loop",
          len(_fric_col_assigns) == 1 and not _col_in_loop
          and _unparse(_fric_col_assigns[0].value)
          == "mu.to(device='cpu', dtype=self._friction_buf_dtype).unsqueeze(1)",
          f"{len(_fric_col_assigns)} assign(s), {len(_col_in_loop)} inside the loop")
    _fric_stmts = list(_fric_fn.body)
    _sync_at = [
        i for i, n in enumerate(_fric_stmts) if isinstance(n, ast.If)
        and "cuda" in _unparse(n).lower()
    ]
    _t0_idx = [i for i, n in enumerate(_fric_stmts)
               if isinstance(n, ast.Assign) and _unparse(n.value) == "time.perf_counter()"]
    check("the write-cost timer is fenced against the queued reset work",
          len(_sync_at) == 1 and len(_t0_idx) == 1 and _sync_at[0] < _t0_idx[0]
          and "torch.cuda.synchronize()" in _unparse(_fric_stmts[_sync_at[0]]),
          f"fence at {_sync_at}, t0 at {_t0_idx}")
    _append_stmts = [
        a for a in _walk(_fric_fn)
        if isinstance(a, ast.Expr) and "_friction_write_ms.append" in _unparse(a)
    ]
    _append_in_loop = [
        a for a in _append_stmts
        if _fric_loop and any(a is b for b in _walk(_fric_loop[0]))
    ]
    check("the write cost is appended ONCE per reset call, outside the loop",
          len(_append_stmts) == 1 and not _append_in_loop,
          f"{len(_append_stmts)} append(s), {len(_append_in_loop)} inside the loop")
    _fric_guards = [
        _unparse(n.test) for n in _walk(_fric_fn) if isinstance(n, ast.If)
        and "idx.numel()" in _unparse(n.test)
    ]
    check("the friction write skips only the EMPTY reset",
          _fric_guards == ["idx.numel() == 0"], str(_fric_guards))
    _reset_stmts = list(_reset_fn.body)
    def _stmt_index(pred):
        return [i for i, n in enumerate(_reset_stmts) if pred(_unparse(n))]
    _draw_at = _stmt_index(lambda t: "_draw_reset_conditions" in t)
    _apply_at = _stmt_index(lambda t: "_apply_friction" in t)
    _apply_args = [
        _unparse(n) for n in _walk(_reset_fn) if isinstance(n, ast.Call)
        and _unparse(n.func).endswith("_apply_friction")
    ]
    check("the friction write runs AFTER the draw, on the resetting envs only",
          len(_draw_at) == 1 and len(_apply_at) == 1 and _draw_at[0] < _apply_at[0]
          and _apply_args == ["self._apply_friction(idx)"],
          f"draw at {_draw_at}, apply at {_apply_at}, call {_apply_args}")
    _log_fn_b4 = _func(env_tree, "_log_finished_episodes", "InsertionEnv")
    _fric_log_vals = {
        k.value: _unparse(v) for d in _walk(_log_fn_b4) if isinstance(d, ast.Dict)
        for k, v in zip(d.keys, d.values)
        if isinstance(k, ast.Constant) and str(k.value).startswith("dr/friction_write_ms")
    }
    check("the logged write cost is a mean over the SAMPLES and a true maximum",
          _fric_log_vals.get("dr/friction_write_ms_mean")
          == "sum(self._friction_write_ms) / len(self._friction_write_ms) "
             "if self._friction_write_ms else 0.0"
          and _fric_log_vals.get("dr/friction_write_ms_max")
          == "max(self._friction_write_ms) if self._friction_write_ms else 0.0",
          str(_fric_log_vals))
    _fric_metric = [
        _unparse(v) for d in _walk(_metrics_fn) if isinstance(d, ast.Dict)
        for k, v in zip(d.keys, d.values)
        if isinstance(k, ast.Constant) and k.value == "friction_write_ms"
    ]
    _fric_metric_src = _fric_metric[0] if _fric_metric else ""
    check("demo_metrics records mean, max, the sample count and the window",
          "'mean': sum(self._friction_write_ms) / len(self._friction_write_ms)"
          in _fric_metric_src
          and "'max': max(self._friction_write_ms)" in _fric_metric_src
          and "'samples_in_window': len(self._friction_write_ms)" in _fric_metric_src
          and "'window': self._friction_write_ms.maxlen" in _fric_metric_src,
          _fric_metric_src or "key missing")
    _fric_deque = [
        _unparse(a.value) for a in _walk(_env_init)
        if isinstance(a, ast.AnnAssign) and a.value is not None
        and _unparse(a.target) == "self._friction_write_ms"
    ]
    check("the write-cost window holds 1000 reset calls",
          _fric_deque == ["collections.deque(maxlen=1000)"], str(_fric_deque))
    _fric_applied_init = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._friction_applied" for t in a.targets)
    ]
    check("the commanded-friction buffer starts at the nominal value",
          len(_fric_applied_init) == 1
          and "insertion_tasks_cfg.CONTACT_FRICTION" in _fric_applied_init[0]
          and "torch.full(" in _fric_applied_init[0],
          str(_fric_applied_init))
    _fric_cmd = [
        _unparse(a.value) for a in _walk(_report_fn) if isinstance(a, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_cmd" for t in a.targets)
    ]
    _fric_nrep = [
        _unparse(a.value) for a in _walk(_report_fn) if isinstance(a, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_n_rep" for t in a.targets)
    ]
    check("the report compares PhysX against the COMMANDED buffer, over four envs",
          _fric_cmd == ["[float(v) for v in self._friction_applied[:_n_rep].tolist()]"]
          and _fric_nrep == ["min(4, self.num_envs)"],
          f"cmd {_fric_cmd}, n_rep {_fric_nrep}")
    _fric_head = [
        _unparse(a.value) for a in _walk(_report_fn) if isinstance(a, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_fric_head" for t in a.targets)
    ]
    check("the D-111 friction header depends on the reach",
          len(_fric_head) == 1 and "self._friction_reach" in _fric_head[0]
          and "centre" in _fric_head[0] and "written once" in _fric_head[0],
          str(_fric_head))

    _live_fn = _func(env_tree, "_live_bounds", "InsertionEnv")
    _log_fn = _func(env_tree, "_log_finished_episodes", "InsertionEnv")

    _live_dicts = [
        a.value for a in _walk(_live_fn)
        if isinstance(a, ast.Assign) and isinstance(a.value, ast.Dict)
        and any(isinstance(t, ast.Name) and t.id == "_static" for t in a.targets)
    ]
    _live_keys = [
        [k.value for k in d.keys if isinstance(k, ast.Constant)]
        for d in _live_dicts
    ]
    _autodr_tree = ast.parse(_read(AUTODR_SRC, ()), str(AUTODR_SRC))
    _dim_names = [
        c.args[0].value
        for a in _autodr_tree.body if isinstance(a, ast.AnnAssign | ast.Assign)
        and "DR_DIMS" in _unparse(a.target if isinstance(a, ast.AnnAssign)
                                     else a.targets[0])
        for c in _walk(a) if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name) and c.func.id == "DimSpec"
        and c.args and isinstance(c.args[0], ast.Constant)
    ]
    check("the band seam answers for every DR_DIMS quantity, and only those",
          len(_dim_names) == 5 and len(_live_keys) == 1
          and list(_live_keys[0]) == _dim_names,
          f"{_live_keys} vs {_dim_names}")

    _live_ifs = [
        _unparse(n.test) for n in _walk(_live_fn) if isinstance(n, ast.If)
        and "self._dr.bounds()" in _unparse(n)
    ]
    _live_merge = [
        _unparse(b) for n in _walk(_live_fn) if isinstance(n, ast.If)
        for b in n.body if "self._dr.bounds()" in _unparse(b)
    ]
    _live_returns = [
        _unparse(r.value) for r in _walk(_live_fn) if isinstance(r, ast.Return)
    ]
    check("the band seam reads the PROVIDER first, not the mode string",
          _live_ifs == ["self._dr is not None"]
          and _live_merge == ["_static.update(self._dr.bounds())"]
          and _live_returns == ["_static"],
          f"{_live_ifs}, merge {_live_merge}, returns {_live_returns}")

    _bounds_calls = {}
    for _m in _walk(env_tree):
        if not isinstance(_m, ast.FunctionDef):
            continue
        _n = sum(
            1 for c in _walk(_m)
            if isinstance(c, ast.Call) and _unparse(c).startswith("self._dr.bounds()")
        )
        if _n:
            _bounds_calls[_m.name] = _n
    check("only the seam and the startup report call self._dr.bounds()",
          _bounds_calls == {"_live_bounds": 1, "_print_startup_report": 2},
          str(_bounds_calls))

    _seam_users = sorted(
        _m.name for _m in _walk(env_tree) if isinstance(_m, ast.FunctionDef)
        and any(isinstance(c, ast.Call)
                and _unparse(c).startswith("self._live_bounds()")
                for c in _walk(_m))
    )
    check("every randomised quantity reads the band through the one seam",
          _seam_users == ["_applied_log_meta", "_apply_friction",
                          "_map_start_conditions", "_reset_idx"],
          str(_seam_users))

    _live_src = _unparse(_live_fn)
    _live_pairs = {
        k.value: _unparse(v)
        for d in _live_dicts for k, v in zip(d.keys, d.values)
        if isinstance(k, ast.Constant)
    }
    check("with no provider the seam rebuilds the static cfg bands",
          _live_pairs.get("lat_r") == "(0.0, self._start_lat_half)"
          and _live_pairs.get("yaw") == "(-_yaw, _yaw)"
          and _live_pairs.get("tilt") == "(0.0, _tilt)"
          and _live_pairs.get("start_height") == "(_lo, _hi)"
          and "_yaw = float(self.cfg.fixture_yaw_noise_rad)" in _live_src
          and "_tilt = float(self.cfg.fixture_tilt_noise_rad)" in _live_src,
          str(_live_pairs))
    check("a low bound at or above the high one collapses to the fixed height",
          "_lo = _hi if _lo is None or float(_lo) >= _hi else float(_lo)" in _live_src,
          "")

    _yaw_assigns = [_unparse(a.value) for a in _named_assign(_reset_fn, "yaw")]
    _mag_assigns = [_unparse(a.value) for a in _named_assign(_reset_fn, "mag")]
    check("the pocket yaw is mapped onto the live band from the yaw column",
          _yaw_assigns == ["autodr.map_unit_to_bounds(u[:, self._col['yaw']], "
                           "float(_y_lo), float(_y_hi))"],
          str(_yaw_assigns))
    check("the tilt magnitude is mapped onto the live band from the tilt column",
          _mag_assigns == ["autodr.map_unit_to_bounds(u[:, self._col['tilt']], "
                           "float(_t_lo), float(_t_hi))"],
          str(_mag_assigns))
    _tilt_guards = [
        _unparse(n.test) for n in _walk(_reset_fn) if isinstance(n, ast.If)
        and any(isinstance(a, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "mag" for t in a.targets)
                for a in n.body)
    ]
    check("the tilt quaternion is built behind the REACH, not behind a cfg field",
          _tilt_guards == ["self._tilt_reach"], str(_tilt_guards))
    _tilt_reach_assigns = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._tilt_reach" for t in a.targets)
    ]
    check("the tilt reach asks bounds_max under a provider and the cfg field otherwise",
          _tilt_reach_assigns == ["_tilt[1] - _tilt[0] > 0.0",
                                  "float(cfg.fixture_tilt_noise_rad) > 0.0"]
          and "_tilt = self._dr.bounds_max().get('tilt', (0.0, float(cfg.fixture_tilt_noise_rad)))"
          in _unparse(_env_init),
          str(_tilt_reach_assigns))

    _draw_stmts = list(_draw_fn.body)
    def _draw_at(text):
        return [i for i, s in enumerate(_draw_stmts) if text in _unparse(s)]
    _row_at = _draw_at("self._reset_unit[idx] = torch.rand")
    _guard_at = _draw_at("self._dr is None")
    _assign_at = _draw_at("boundary_assignment")
    _mark_at = _draw_at("self._boundary_of[idx]")
    _nail_at = _draw_at("apply_boundary")
    _stamp_at = _draw_at("self._bounds_stamp[idx]")
    check("the draw marks boundary envs and nails them, in that order",
          [len(x) for x in (_row_at, _guard_at, _assign_at, _mark_at, _nail_at, _stamp_at)]
          == [1, 1, 1, 1, 1, 1]
          and _row_at[0] < _guard_at[0] < _assign_at[0] < _mark_at[0]
          and _mark_at[0] < _nail_at[0] < _stamp_at[0],
          f"row {_row_at} guard {_guard_at} assign {_assign_at} "
          f"mark {_mark_at} nail {_nail_at} stamp {_stamp_at}")
    _assign_calls = [
        _unparse(c) for c in _walk(_draw_fn) if isinstance(c, ast.Call)
        and _unparse(c.func).endswith("boundary_assignment")
    ]
    check("the boundary draw uses two INDEPENDENT columns",
          _assign_calls == ["self._dr.boundary_assignment(_rb[:, 0], _rb[:, 1])"],
          str(_assign_calls))
    _nail_calls = [
        _unparse(c) for c in _walk(_draw_fn) if isinstance(c, ast.Call)
        and _unparse(c.func).endswith("apply_boundary")
    ]
    _rows_assigns = [_unparse(a.value) for a in _named_assign(_draw_fn, "_rows")]
    _sel_assigns = [_unparse(a.value) for a in _named_assign(_draw_fn, "_sel")]
    check("the nail gets the whole row buffer, absolute env ids and one selection",
          _nail_calls == ["autodr.apply_boundary(self._reset_unit, _rows, "
                          "self._boundary_cols[_sel], self._boundary_sides[_sel])"]
          and _rows_assigns == ["idx[_is_b]"]
          and _sel_assigns == ["_which[_is_b]"],
          f"{_nail_calls}, rows {_rows_assigns}, sel {_sel_assigns}")
    _mark_assigns = [
        _unparse(a.value) for a in _walk(_draw_fn) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._boundary_of[idx]" for t in a.targets)
    ]
    check("every resetting env gets a boundary mark, -1 when it is regular",
          _mark_assigns == ["torch.where(_is_b, _which, torch.full_like(_which, -1))"],
          str(_mark_assigns))
    _stamp_assigns = [
        _unparse(a.value) for a in _walk(_draw_fn) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._bounds_stamp[idx]" for t in a.targets)
    ]
    check("the stamp is the provider's CURRENT bounds version",
          _stamp_assigns == ["self._dr.bounds_version"], str(_stamp_assigns))
    _b_cols = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._boundary_cols" for t in a.targets)
    ]
    _b_sides = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._boundary_sides" for t in a.targets)
    ]
    _b_keys = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._boundary_keys" for t in a.targets)
    ]
    check("columns, sides and keys are the provider's, in the provider's order",
          _b_cols == ["torch.tensor(self._dr.boundary_columns, dtype=torch.long, "
                      "device=self.device)", "None"]
          and _b_sides == ["torch.tensor(self._dr.boundary_sides, "
                           "dtype=self._reset_unit.dtype, device=self.device)", "None"]
          and _b_keys == ["tuple(self._dr.keys)", "()"],
          f"cols {_b_cols}, sides {_b_sides}, keys {_b_keys}")
    _b_of_init = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._boundary_of" for t in a.targets)
    ]
    _stamp_init = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._bounds_stamp" for t in a.targets)
    ]
    check("the mark and the stamp are integer buffers, the mark starting at -1",
          _b_of_init == ["torch.full((self.num_envs,), -1, dtype=torch.long, "
                         "device=self.device)"]
          and _stamp_init == ["torch.zeros((self.num_envs,), dtype=torch.long, "
                              "device=self.device)"],
          f"mark {_b_of_init}, stamp {_stamp_init}")

    _log_stmts = list(_walk(_log_fn))
    def _log_line(text):
        return [
            n.lineno for n in _log_stmts
            if isinstance(n, (ast.Expr, ast.Assign, ast.For, ast.If))
            and text in _unparse(n)
        ]
    _record_at = [
        n.lineno for n in _log_stmts if isinstance(n, ast.Call)
        and _unparse(n.func).endswith("_dr.record")
    ]
    _update_at = [
        n.lineno for n in _log_stmts if isinstance(n, ast.Call)
        and _unparse(n.func).endswith("_dr.update")
    ]
    _extend_at = [
        n.lineno for n in _log_stmts if isinstance(n, ast.Call)
        and _unparse(n.func).endswith("_fresh_successes.extend")
    ]
    _clear_at = [
        n.lineno for n in _log_stmts if isinstance(n, ast.Call)
        and _unparse(n.func).endswith("_fresh_successes.clear")
    ]
    check("record runs before update, and the fresh window is filled before it",
          len(_record_at) == len(_update_at) == len(_extend_at) == len(_clear_at) == 1
          and _record_at[0] < _extend_at[0] < _update_at[0] < _clear_at[0],
          f"record {_record_at}, extend {_extend_at}, update {_update_at}, "
          f"clear {_clear_at}")
    _record_calls = [
        _unparse(c) for c in _log_stmts if isinstance(c, ast.Call)
        and _unparse(c.func).endswith("_dr.record")
    ]
    check("the success flag is booked on the boundary that episode ran at",
          _record_calls == ["self._dr.record(self._boundary_keys[_k], bool(_f))"],
          str(_record_calls))
    _fresh_assigns = [_unparse(a.value) for a in _named_assign(_log_fn, "_fresh")]
    check("the fresh window takes regular episodes AND the current stamp only",
          _fresh_assigns == ["~_is_b & (self._bounds_stamp[idx] "
                             "== self._dr.bounds_version)"],
          str(_fresh_assigns))
    _clear_guards = [
        _unparse(n.test) for n in _log_stmts if isinstance(n, ast.If)
        and any(isinstance(b, ast.Expr) and isinstance(b.value, ast.Call)
                and _unparse(b.value.func).endswith("_fresh_successes.clear")
                for b in n.body)
    ]
    check("the fresh window is cleared only when a boundary actually MOVED",
          _clear_guards == ["any((_ev.moved for _ev in _events))"],
          str(_clear_guards))
    _dr_log_vals = {}
    for _a in _walk(_log_fn):
        if not isinstance(_a, ast.Assign) or len(_a.targets) != 1:
            continue
        _t = _a.targets[0]
        if not isinstance(_t, ast.Subscript):
            continue
        if _unparse(_t.value) != "self.extras['log']":
            continue
        if isinstance(_t.slice, ast.Constant) and isinstance(_t.slice.value, str):
            _dr_log_vals[_t.slice.value] = _unparse(_a.value)
    check("the fresh rate averages the FRESH window and nothing else",
          _dr_log_vals.get("dr/train_success_regular_fresh")
          == "sum(self._fresh_successes) / _fresh_n if _fresh_n >= self._fresh_min "
             "else -1.0",
          str(_dr_log_vals.get("dr/train_success_regular_fresh")))
    check("the fresh count and the minimum are their own curves",
          _dr_log_vals.get("dr/fresh_n") == "float(_fresh_n)"
          and _dr_log_vals.get("dr/fresh_min") == "float(self._fresh_min)"
          and [_unparse(a.value) for a in _named_assign(_log_fn, "_fresh_n")]
          == ["len(self._fresh_successes)"],
          f"n {_dr_log_vals.get('dr/fresh_n')}, min {_dr_log_vals.get('dr/fresh_min')}")
    check("the boundary rate averages the boundary window, on its own key",
          _dr_log_vals.get("dr/success_rate_boundary")
          == "sum(self._recent_boundary_successes) / "
             "len(self._recent_boundary_successes) "
             "if self._recent_boundary_successes else 0.0",
          str(_dr_log_vals.get("dr/success_rate_boundary")))
    check("the commanded friction reaches the log from the per-episode buffer",
          _dr_log_vals.get("dr/friction_mean")
          == "sum(self._recent_friction) / len(self._recent_friction) "
             "if self._recent_friction else 0.0"
          and any(_unparse(c) == "self._recent_friction.extend("
                  "self._friction_applied[idx].tolist())"
                  for c in _log_stmts if isinstance(c, ast.Call)),
          str(_dr_log_vals.get("dr/friction_mean")))
    _scalars_calls = [
        _unparse(c) for c in _log_stmts if isinstance(c, ast.Call)
        and "scalars()" in _unparse(c)
        and _unparse(c.func).endswith(".update")
    ]
    check("the provider's scalars are MERGED into the existing log dict",
          _scalars_calls == ["self.extras['log'].update(self._dr.scalars())"],
          str(_scalars_calls))
    _log_dr_guards = [
        _unparse(n.test) for n in _walk(_log_fn) if isinstance(n, ast.If)
        and any(isinstance(c, ast.Call) and "_dr." in _unparse(c)
                for c in _walk(n))
    ]
    check("every provider call in the log sits behind `self._dr is not None`",
          set(_log_dr_guards) == {"self._dr is not None"}
          and len(_log_dr_guards) == 2,
          str(_log_dr_guards))
    _floor_assigns = [_unparse(a.value) for a in _named_assign(_env_init, "window_floor")]
    _fresh_min_assigns = [
        _unparse(a.value) for a in _walk(_env_init) if isinstance(a, ast.Assign)
        and any(_unparse(t) == "self._fresh_min" for t in a.targets)
    ]
    _window_assigns = [_unparse(a.value) for a in _named_assign(_env_init, "window")]
    check("the fresh minimum and the window floor are one number",
          _floor_assigns == ["2000"]
          and _fresh_min_assigns == ["window_floor"]
          and len(_window_assigns) == 1
          and _window_assigns[0].startswith("max(window_floor,"),
          f"floor {_floor_assigns}, min {_fresh_min_assigns}, "
          f"window {_window_assigns}")
    _b5_deques = {
        _unparse(a.target): _unparse(a.value)
        for a in _walk(_env_init) if isinstance(a, ast.AnnAssign) and a.value
        and _unparse(a.target) in (
            "self._fresh_successes", "self._recent_boundary_successes",
            "self._recent_friction",
        )
    }
    check("the three B5 windows share the trailing-window size",
          len(_b5_deques) == 3
          and set(_b5_deques.values()) == {"collections.deque(maxlen=window)"},
          str(_b5_deques))

    _b_win_calls = [
        _unparse(c) for c in _log_stmts if isinstance(c, ast.Call)
        and _unparse(c.func).endswith("_recent_boundary_successes.extend")
    ]
    check("the boundary window is fed by the BOUNDARY mask",
          _b_win_calls == ["self._recent_boundary_successes.extend("
                           "successes[_is_b].tolist())"],
          str(_b_win_calls))
    _fric_win_top = [
        _unparse(b) for b in _log_fn.body
        if "_recent_friction.extend" in _unparse(b)
    ]
    check("the friction window is filled in EVERY mode, outside the provider guard",
          _fric_win_top == ["self._recent_friction.extend("
                            "self._friction_applied[idx].tolist())"],
          str(_fric_win_top))
    _ev_loops = [
        n for n in _walk(_log_fn) if isinstance(n, ast.For)
        and _unparse(n.iter) == "_events"
    ]
    check("every PROCESSED buffer prints its line, moved or not",
          len(_ev_loops) == 1
          and [_unparse(b) for b in _ev_loops[0].body] == ["print(_ev.line())"],
          str([_unparse(b) for b in _ev_loops[0].body] if _ev_loops else []))
    _home_fallbacks = [
        _unparse(a) for a in _walk(_live_fn) if isinstance(a, ast.Assign)
        and "_start_tip_height" in _unparse(a)
    ]
    check("the seam falls back to the home height only when the cfg field IS None",
          _home_fallbacks == ["_hi = self._start_tip_height if _hi is None "
                              "else float(_hi)"],
          str(_home_fallbacks))
    _rec_loops = [
        n for n in _walk(_log_fn) if isinstance(n, ast.For)
        and any(isinstance(c, ast.Call) and _unparse(c.func).endswith("_dr.record")
                for c in _walk(n))
    ]
    _rec_body = [_unparse(b) for b in _rec_loops[0].body] if _rec_loops else []
    _rec_iter = _unparse(_rec_loops[0].iter) if _rec_loops else ""
    _tolists = [
        _unparse(a.value) for a in _walk(_log_fn) if isinstance(a, ast.Assign)
        and _unparse(a.targets[0]) in ("_b_keys", "_b_flags")
    ]
    check("the record loop runs on host values, not on device tensors",
          _rec_iter == "zip(_b_keys, _b_flags)"
          and _rec_body == ["self._dr.record(self._boundary_keys[_k], bool(_f))"]
          and _tolists == ["_b_of[_is_b].tolist()", "successes[_is_b].tolist()"],
          f"iter {_rec_iter}, body {_rec_body}, copies {_tolists}")
    _train_strings = {
        n.value for n in _walk(train_tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    _low_expr = _class_attr_expr(cfg_tree, "InsertionEnvCfg", "start_tip_above_entrance_low")
    _low_src = _unparse(_low_expr) if _low_expr is not None else None
    check("the lower start bound defaults to the derived upper bound, not None",
          _low_src is not None and "RUNG0_START_TIP_ABOVE_ENTRANCE" in _low_src
          and "None" not in _low_src,
          f"start_tip_above_entrance_low = {_low_src}")
    check("demo_metrics.json and the run tag carry the start range",
          "start_tip_above_entrance_low_mm" in _env_strings
          and "success_by_start_height_bin" in _env_strings
          and "start_tip_above_entrance_low" in _train_strings,
          f"env keys {'start_tip_above_entrance_low_mm' in _env_strings}/"
          f"{'success_by_start_height_bin' in _env_strings}, "
          f"train tag {'start_tip_above_entrance_low' in _train_strings}")


    _dr_tag_ifs = [
        n for n in _walk(train_tree) if isinstance(n, ast.If)
        and "dr_mode" in _unparse(n.test)
        and any("parts.append" in _unparse(b) for b in n.body)
    ]
    _dr_tag_test = _unparse(_dr_tag_ifs[0].test) if _dr_tag_ifs else None
    _dr_tag_body = [_unparse(b) for b in _dr_tag_ifs[0].body] if _dr_tag_ifs else []
    check("the run tag names the DR mode, and only when it is not off",
          len(_dr_tag_ifs) == 1
          and _dr_tag_test == "hasattr(env_cfg, 'dr_mode') and str(env_cfg.dr_mode) != 'off'"
          and _dr_tag_body == ["parts.append(str(env_cfg.dr_mode))"],
          f"test {_dr_tag_test}, body {_dr_tag_body}")

    _dr_tag_line = _dr_tag_ifs[0].lineno if _dr_tag_ifs else -1
    _tilt_tag_lines = [
        n.lineno for n in _walk(train_tree)
        if isinstance(n, ast.Call) and "parts.append" in _unparse(n)
        and "tilt0-" in _unparse(n)
    ]
    _tilt_tag_line = min(_tilt_tag_lines) if _tilt_tag_lines else -1
    check("the DR-mode tag is appended before the static-noise tags",
          _dr_tag_line > 0 and _tilt_tag_line > 0 and _dr_tag_line < _tilt_tag_line,
          f"dr tag at {_dr_tag_line}, tilt tag at {_tilt_tag_line}")

    _dr_flag_kw: dict = {}
    for _n in _walk(train_tree):
        if (isinstance(_n, ast.Call) and isinstance(_n.func, ast.Attribute)
                and _n.func.attr == "add_argument"
                and any(isinstance(a, ast.Constant) and a.value == "--stop-when-dr-max"
                        for a in _n.args)):
            _dr_flag_kw = {kw.arg: _unparse(kw.value) for kw in _n.keywords}
    _dr_flag_seen = "--stop-when-dr-max" in _cli_flags(train_tree)
    check("--stop-when-dr-max is a float the run must pass, never defaulted here",
          _dr_flag_seen
          and _dr_flag_kw.get("type") == "float"
          and _dr_flag_kw.get("default") == "None",
          f"declared {_dr_flag_seen}, type {_dr_flag_kw.get('type')}, "
          f"default {_dr_flag_kw.get('default')}")

    _gym_make_lines = [
        n.lineno for n in _walk(train_tree)
        if isinstance(n, ast.Call) and _unparse(n).startswith("gym.make(")
    ]
    _gym_make_line = min(_gym_make_lines) if _gym_make_lines else -1
    _stop_refusals = sorted(
        (_unparse(n.test), n.lineno) for n in _walk(train_tree)
        if isinstance(n, ast.If) and "stop_when_dr_max" in _unparse(n.test)
        and any(isinstance(b, ast.Raise) for b in n.body)
    )
    _want_refusals = sorted([
        "args_cli.stop_when_dr_max is not None and args_cli.stop_at_success_rate is not None",
        "args_cli.stop_when_dr_max is not None and (str(getattr(env_cfg, 'dr_mode', 'off')) "
        "!= 'autodr' and (not _floor_on))",
    ])
    check("both stop-flag refusals run before the env is built",
          [t for t, _ in _stop_refusals] == _want_refusals
          and _gym_make_line > 0
          and all(ln < _gym_make_line for _, ln in _stop_refusals),
          f"{[t for t, _ in _stop_refusals]}, lines "
          f"{[ln for _, ln in _stop_refusals]} vs gym.make at {_gym_make_line}")

    _save_body = [_unparse(b) for b in _func(train_tree, "_save").body]
    check("the sidecar is written after the checkpoint, by the wrapped save",
          _save_body[:1] == ["out = save_fn(path, *args, **kwargs)"]
          and any("json.dump(" in b and "provider.state_dict()" in b for b in _save_body)
          and _save_body[-1:] == ["return out"],
          str(_save_body))

    _wrap_assigns = [
        _unparse(n) for n in _walk(train_tree)
        if isinstance(n, ast.Assign) and _unparse(n.targets[0]) == "runner.save"
    ]
    check("runner.save is wrapped so every checkpoint gets its AutoDR state",
          _wrap_assigns == [
              "runner.save = _save_with_autodr(runner.save, _dr, agent_cfg.seed,"
              " os.path.basename(log_dir))"
          ],
          str(_wrap_assigns))

    _side_src = _unparse(_func(train_tree, "_autodr_sidecar_path"))
    check("the sidecar carries the checkpoint's own iteration number",
          "stem.startswith('model_')" in _side_src
          and "'autodr_' + stem[len('model_'):] + '.json'" in _side_src,
          _side_src.replace("\n", " | ")[:220])

    _resume_refusals = [
        n for n in _walk(train_tree) if isinstance(n, ast.If)
        and _unparse(n.test) == "not os.path.isfile(_side)"
        and any(isinstance(b, ast.Raise) for b in n.body)
    ]
    _dr_loads = [
        _unparse(n) for n in _walk(train_tree)
        if isinstance(n, ast.Call) and "load_state_dict" in _unparse(n)
    ]
    _runner_load_lines = [
        n.lineno for n in _walk(train_tree)
        if isinstance(n, ast.Call) and _unparse(n) == "runner.load(resume_path)"
    ]
    _dr_load_lines = [
        n.lineno for n in _walk(train_tree)
        if isinstance(n, ast.Call) and "load_state_dict" in _unparse(n)
    ]
    _runner_load_line = min(_runner_load_lines) if _runner_load_lines else -1
    _dr_load_line = min(_dr_load_lines) if _dr_load_lines else -1
    check("a resume without the AutoDR state is refused, and the state loads after the policy",
          len(_resume_refusals) == 1
          and _dr_loads == ["_dr.load_state_dict(_doc['autodr'])"]
          and _runner_load_line > 0 and _dr_load_line > _runner_load_line,
          f"refusals {len(_resume_refusals)}, loads {_dr_loads}, "
          f"runner.load@{_runner_load_line} dr.load@{_dr_load_line}")

    _dr_crit = _func(train_tree, "_criterion_dr_max")
    _dr_crit_src = _unparse(_dr_crit)
    _dr_returns = [_unparse(n.value) for n in _walk(_dr_crit)
                   if isinstance(n, ast.Return) and n.value is not None]
    check("the AutoDR stop tests all three conditions of plan section 4",
          any("at_max and enough and (rate >= target)" in r for r in _dr_returns),
          str(_dr_returns))

    _dr_reads_fresh = "_fresh_successes" in _dr_crit_src and "_fresh_min" in _dr_crit_src
    _dr_reads_recent = "_recent_successes" in _dr_crit_src
    check("the AutoDR stop reads the fresh window, never the recent one",
          _dr_reads_fresh and "all_at_max" in _dr_crit_src and not _dr_reads_recent,
          f"fresh {_dr_reads_fresh}, all_at_max "
          f"{'all_at_max' in _dr_crit_src}, recent {_dr_reads_recent}")

    check("an under-full fresh window prints no rate at all",
          "shown = f'{rate:.4f}' if enough else '--'" in _dr_crit_src,
          _dr_crit_src.split("shown = ")[-1].splitlines()[0][:48]
          if "shown = " in _dr_crit_src else "no shown")

    _rate_crit_src = _unparse(_func(train_tree, "_criterion_success_rate"))
    _rate_ok = (
        "_recent_successes" in _rate_crit_src
        and "recent.maxlen // 4" in _rate_crit_src
        and "filled and rate >= target" in _rate_crit_src
        and "_fresh_successes" not in _rate_crit_src
    )
    check("D-037's stop rule still reads every episode and the quarter-window guard",
          _rate_ok,
          f"recent {'_recent_successes' in _rate_crit_src}, "
          f"quarter {'recent.maxlen // 4' in _rate_crit_src}, "
          f"compare {'filled and rate >= target' in _rate_crit_src}, "
          f"fresh leaked {'_fresh_successes' in _rate_crit_src}")

    _loop_ifs = sorted(
        (n.lineno, _unparse(n.test)) for n in _walk(train_tree)
        if isinstance(n, ast.If) and _unparse(n.test).startswith("state ==")
    )
    check("the block loop abandons before it fires",
          [t for _, t in _loop_ifs] == ["state == 'abandon'", "state == 'fire'"],
          str([t for _, t in _loop_ifs]))

    _ladder_guards = [
        n for n in _walk(cfg_tree) if isinstance(n, ast.If)
        and _unparse(n.test) == "ladder_on and str(cfg.dr_mode) != 'off'"
        and any(isinstance(b, ast.Raise) for b in n.body)
    ]
    _validate_fn = _func(cfg_tree, "validate_rl_config")
    _named_assigns = [
        _unparse(n.value) for n in _walk(_validate_fn)
        if isinstance(n, ast.Assign) and _unparse(n.targets[0]) == "named"
    ]
    _dr_in_sweep = len(_named_assigns) == 1 and "'dr_mode'" in _named_assigns[0]
    check("the ladder and AutoDR are refused together, and dr_mode is in the absent-field sweep",
          len(_ladder_guards) == 1 and _dr_in_sweep,
          f"guards {len(_ladder_guards)}, dr_mode in sweep {_dr_in_sweep}")

    _vtm_ns: dict = {}
    try:
        for _want in ("RlConfigIncomplete", "DR_TABLE_WIDTHS", "validate_table_mode"):
            _node = next(
                n for n in cfg_tree.body
                if getattr(n, "name", None) == _want
                or (isinstance(n, ast.AnnAssign) and _unparse(n.target) == _want)
            )
            exec(compile(ast.Module(body=[_node], type_ignores=[]), "<cfg>", "exec"), _vtm_ns)  # noqa: S102
        _vtm = _vtm_ns["validate_table_mode"]
    except Exception:  # noqa: BLE001 - an unbuildable guard IS the finding
        _vtm = None

    def _table_cfg(**over):
        base = dict(dr_mode="table", dr_table_path="scripts/eval/eval_table_seed20.csv",
                    dr_table_width="target", start_floor_m=0.02,
                    start_tip_above_entrance=0.02, dr_table_allow_outside=False)
        base.update(over)
        return types.SimpleNamespace(**base)

    def _table_verdict(**over) -> str:
        if _vtm is None:
            return "no function"
        try:
            _vtm(_table_cfg(**over))
        except Exception as exc:  # noqa: BLE001 - the NAME is what is tested
            return type(exc).__name__
        return "passes"

    _tv_good = [_table_verdict(), _table_verdict(dr_table_width="nominal"),
                _table_verdict(dr_mode="off", dr_table_path=None),
                _table_verdict(dr_mode="autodr", dr_table_path=None, start_floor_m=-0.03)]
    check("table mode: both widths pass, and 'off' / 'autodr' with a floor are left alone",
          _tv_good == ["passes"] * 4, str(_tv_good))
    _tv_path = [_table_verdict(dr_table_path=None), _table_verdict(dr_table_path="  ")]
    check("table mode without a table path is refused",
          _tv_path == ["RlConfigIncomplete"] * 2, str(_tv_path))
    _tv_floor = [_table_verdict(start_floor_m=-0.03),
                 _table_verdict(start_tip_above_entrance=None)]
    check("table mode with a start floor, or without a start height, is refused",
          _tv_floor == ["RlConfigIncomplete"] * 2, str(_tv_floor))
    _tv_width = _table_verdict(dr_table_width="max")
    check("table mode with an unknown width is refused",
          _tv_width == "RlConfigIncomplete", _tv_width)
    _tv_stray = _table_verdict(dr_mode="off")
    check("a table path that no mode reads is refused",
          _tv_stray == "RlConfigIncomplete", _tv_stray)
    _tv_ood = [_table_verdict(dr_table_allow_outside=True),
               _table_verdict(dr_mode="off", dr_table_path=None, dr_table_allow_outside=True)]
    check("the OOD switch passes in table mode and is refused outside it",
          _tv_ood == ["passes", "RlConfigIncomplete"], str(_tv_ood))
    _vtm_calls = [
        n for n in _validate_fn.body
        if isinstance(n, ast.Expr) and _unparse(n.value) == "validate_table_mode(cfg)"
    ]
    check("validate_rl_config calls validate_table_mode unconditionally",
          len(_vtm_calls) == 1, f"{len(_vtm_calls)} top-level call(s)")
    _tbl_defaults = [
        _unparse(_e) if _e is not None else "ABSENT"
        for _e in (_class_attr_expr(cfg_tree, "InsertionEnvCfg", "dr_table_path"),
                   _class_attr_expr(cfg_tree, "InsertionEnvCfg", "dr_table_width"),
                   _class_attr_expr(cfg_tree, "InsertionEnvCfg", "dr_table_allow_outside"))
    ]
    check("the cfg defaults are no table and the target width",
          _tbl_defaults == ["None", "'target'", "False"], str(_tbl_defaults))

    _parent: dict = {}
    for _n in _walk(train_tree):
        for _c in ast.iter_child_nodes(_n):
            _parent[_c] = _n

    def _enclosing_if(node) -> str | None:
        """The test of the nearest ``if`` this node sits in the BODY of."""
        cur = node
        while cur in _parent:
            par = _parent[cur]
            if isinstance(par, ast.If) and any(cur is b for b in par.body):
                return _unparse(par.test)
            cur = par
        return None

    def _calls(scope, name: str) -> list:
        """Every ``ast.Call`` in ``scope`` whose callee unparses to ``name``."""
        return [n for n in _walk(scope)
                if isinstance(n, ast.Call) and _unparse(n.func) == name]

    def _assigned(scope, target: str) -> list:
        """Unparsed right-hand sides of every ``target = ...`` in ``scope``."""
        return [_unparse(n.value) for n in _walk(scope)
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and _unparse(n.targets[0]) == target]

    _dr_crit_fn = _func(train_tree, "_criterion_dr_max")
    _rate_crit_fn = _func(train_tree, "_criterion_success_rate")
    _save_fn = _func(train_tree, "_save")
    _side_fn = _func(train_tree, "_autodr_sidecar_path")
    _main_fn = _func(train_tree, "main")

    _wrap_calls = _calls(train_tree, "_save_with_autodr")
    _wrap_guard = _enclosing_if(_wrap_calls[0]) if _wrap_calls else None
    check("the save wrapper is installed under the provider guard and no other",
          len(_wrap_calls) == 1 and _wrap_guard == "_dr is not None",
          f"installs {len(_wrap_calls)}, guard {_wrap_guard!r}")

    _side_args = {
        id(c): [_unparse(a) for a in c.args] for c in _calls(train_tree, "_autodr_sidecar_path")
    }
    _side_arglists = sorted(v for v in _side_args.values())
    check("the sidecar path is derived from the checkpoint path at both call sites",
          _side_arglists == [["path"], ["resume_path"]],
          str(_side_arglists))

    _split_args = [[_unparse(a) for a in c.args] for c in _calls(_side_fn, "os.path.split")]
    check("the sidecar path splits the checkpoint path itself, not a basename",
          _split_args == [["model_path"]], str(_split_args))

    _atmax_calls = _calls(_dr_crit_fn, "dr.all_at_max")
    _atmax_names = [_unparse(n) for n in _walk(_dr_crit_fn)
                    if isinstance(n, ast.Attribute) and n.attr == "all_at_max"]
    check("the AutoDR stop CALLS all_at_max, it does not merely mention it",
          len(_atmax_calls) == 1 and len(_atmax_names) == 1,
          f"calls {len(_atmax_calls)}, references {len(_atmax_names)}")

    _floor_rhs = _assigned(_dr_crit_fn, "floor")
    check("the fresh-window floor is the env's whole _fresh_min",
          _floor_rhs == ["int(getattr(base, '_fresh_min', 0))"], str(_floor_rhs))

    _dr_rate_rhs = _assigned(_dr_crit_fn, "rate")
    check("the AutoDR stop's rate is the fresh buffer's own mean",
          _dr_rate_rhs == ["sum(fresh) / n_fresh if n_fresh else 0.0"], str(_dr_rate_rhs))

    _while = next((n for n in _walk(_main_fn) if isinstance(n, ast.While)), None)
    _body_src = [_unparse(b) for b in _while.body] if _while is not None else []
    _learn_at = next((i for i, b in enumerate(_body_src) if b.startswith("runner.learn(")), -1)
    _crit_at = next((i for i, b in enumerate(_body_src) if "criterion()" in b), -1)
    check("the criterion is evaluated after the block trains, never before",
          _learn_at >= 0 and _crit_at > _learn_at,
          f"learn at {_learn_at}, criterion at {_crit_at}")

    _rate_rhs = _assigned(_rate_crit_fn, "rate")
    check("D-037's rate divides by the window's contents, not its capacity",
          _rate_rhs == ["sum(recent) / len(recent) if recent else 0.0"], str(_rate_rhs))

    _note_prints = _calls(train_tree, "print")
    _note_call = next((c for c in _note_prints
                       if [_unparse(a) for a in c.args] == ["note"]), None)
    _note_guard = _enclosing_if(_note_call) if _note_call is not None else "MISSING"
    check("the not-a-test-rate note is printed on every path, not only on an early stop",
          _note_call is not None and _note_guard is None,
          f"note guard {_note_guard!r}")

    _bar_if = next((n for n in _walk(_main_fn) if isinstance(n, ast.If)
                    and _unparse(n.test) == "_dr_bar is None"), None)
    _then_crit = [_unparse(x.value) for b in (_bar_if.body if _bar_if else [])
                  for x in _walk(b) if isinstance(x, ast.Assign)
                  and _unparse(x.targets[0]) == "criterion"]
    _else_crit = [_unparse(x.value) for b in (_bar_if.orelse if _bar_if else [])
                  for x in _walk(b) if isinstance(x, ast.Assign)
                  and _unparse(x.targets[0]) == "criterion"]
    check("each stop branch selects its own criterion",
          _then_crit == ["_criterion_success_rate"] and _else_crit == ["_criterion_dr_max"],
          f"no-bar {_then_crit}, bar {_else_crit}")

    _final_saves = [_unparse(c) for c in _walk(_main_fn)
                    if isinstance(c, ast.Call) and "save" in _unparse(c.func)
                    and "final_path" in _unparse(c)]
    check("the early-stop save goes through the wrapped instance attribute",
          _final_saves == ["runner.save(final_path)"], str(_final_saves))

    _final_rhs = _assigned(_main_fn, "final_path")
    check("the early-stop checkpoint is named from rsl_rl's own iteration counter",
          _final_rhs == ["os.path.join(log_dir, f'model_{runner.current_learning_iteration}.pt')"],
          str(_final_rhs))

    _dr_line_rhs = _assigned(_dr_crit_fn, "line")
    _dr_line = _dr_line_rhs[0] if _dr_line_rhs else ""
    check("the AutoDR stop prints the guarded rate, never the raw one",
          "{shown}" in _dr_line and "{rate" not in _dr_line,
          _dr_line[-96:])

    _abandon_if = next((n for n in _walk(_main_fn) if isinstance(n, ast.If)
                        and _unparse(n.test) == "state == 'abandon'"), None)
    _abandon_learns = _calls(_abandon_if, "runner.learn") if _abandon_if else []
    _abandon_rest = [_unparse(n.value) for n in _walk(_abandon_if)
                     if isinstance(n, ast.Assign)
                     and _unparse(n.targets[0]) == "_rest"] if _abandon_if else []
    check("an abandoned criterion trains the rest instead of ending the run",
          len(_abandon_learns) == 1
          and _abandon_rest == ["agent_cfg.max_iterations - trained"],
          f"learn calls {len(_abandon_learns)}, _rest {_abandon_rest}")

    _range_ifs = [n for n in _walk(_main_fn) if isinstance(n, ast.If)
                  and "0.0 <= float(_val) <= 1.0" in _unparse(n.test)
                  and any(isinstance(b, ast.Raise) for b in n.body)]
    _range_loop = next((n for n in _walk(_main_fn) if isinstance(n, ast.For)
                        and "stop_when_dr_max" in _unparse(n.iter)
                        and "stop_at_success_rate" in _unparse(n.iter)), None)
    check("both stop bars are refused outside [0, 1]",
          len(_range_ifs) == 1 and _range_loop is not None,
          f"guards {len(_range_ifs)}, covers both flags {_range_loop is not None}")

    _env_keys = sorted(
        _unparse(k) for c in _calls(_save_fn, "json.dump")
        for a in c.args if isinstance(a, ast.Dict) for k in a.keys
    )
    _seed_refusals = [n for n in _walk(_main_fn) if isinstance(n, ast.If)
                      and _unparse(n.test) == "_doc.get('seed') != agent_cfg.seed"
                      and any(isinstance(b, ast.Raise) for b in n.body)]
    _fmt_refusals = [n for n in _walk(_main_fn) if isinstance(n, ast.If)
                     and _unparse(n.test) == "_doc.get('format') != SIDECAR_FORMAT"
                     and any(isinstance(b, ast.Raise) for b in n.body)]
    _distill_refusals = [n for n in _walk(_main_fn) if isinstance(n, ast.If)
                         and _unparse(n.test)
                         == "_dr is not None and agent_cfg.algorithm.class_name == 'Distillation'"
                         and any(isinstance(b, ast.Raise) for b in n.body)]
    _loads = [_unparse(c) for c in _calls(_main_fn, "_dr.load_state_dict")]
    check("the sidecar carries the run identity and the resume checks it",
          _env_keys == ["'autodr'", "'format'", "'run'", "'seed'"]
          and len(_seed_refusals) == 1 and len(_fmt_refusals) == 1
          and len(_distill_refusals) == 1
          and _loads == ["_dr.load_state_dict(_doc['autodr'])"],
          f"envelope {_env_keys}, seed {len(_seed_refusals)}, format "
          f"{len(_fmt_refusals)}, distill {len(_distill_refusals)}, load {_loads}")

    _eparent: dict = {}
    for _n in _walk(env_tree):
        for _c in ast.iter_child_nodes(_n):
            _eparent[_c] = _n

    def _e_enclosing_if(node) -> str | None:
        cur = node
        while cur in _eparent:
            par = _eparent[cur]
            if isinstance(par, ast.If) and any(cur is b for b in par.body):
                return _unparse(par.test)
            cur = par
        return None

    def _e_calls(scope, name: str) -> list:
        return [n for n in _walk(scope)
                if isinstance(n, ast.Call) and _unparse(n.func) == name]

    _cone_calls = _e_calls(env_tree, "insertion_math.clamp_tilt_to_cone")
    _cone_args = [[_unparse(a) for a in c.args] for c in _cone_calls]
    check("the tilt cone is measured from this episode's pocket axis",
          _cone_args == [["tgt_quat_w", "float(self.cfg.osc_tilt_clamp_rad)",
                          "self._fixture_quat"]],
          str(_cone_args))

    _apply_osc_fn = _func(env_tree, "_apply_osc", "InsertionEnv")
    _yaw_calls = _e_calls(env_tree, "insertion_math.clamp_yaw_to_band")
    _yaw_args = [[_unparse(a) for a in c.args] for c in _yaw_calls]
    _seat_ref = [_unparse(n.value) for n in _walk(_apply_osc_fn)
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and _unparse(n.targets[0]) == "seat_quat_w"]
    check("the yaw band is measured against this episode's SEATED orientation",
          _yaw_args == [["tgt_quat_w", "float(self.cfg.osc_yaw_clamp_rad)", "seat_quat_w"]]
          and _seat_ref == ["insertion_math.quat_mul(self._fixture_quat, self._seat_quat_local)"],
          f"{_yaw_args} / seat_quat_w = {_seat_ref}")
    _orient_seq = sorted(
        (n.lineno, _unparse(n.value.func), _unparse(n.targets[0]))
        for n in _walk(_apply_osc_fn)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.value, ast.Call)
        and _unparse(n.value.func) in ("insertion_math.clamp_tilt_to_cone",
                                          "insertion_math.clamp_yaw_to_band",
                                          "insertion_math.clamp_tip_in_box")
    )
    check("the yaw band is ASSIGNED back, after the tilt cone and before the box",
          [(f, t) for _, f, t in _orient_seq]
          == [("insertion_math.clamp_tilt_to_cone", "tgt_quat_w"),
              ("insertion_math.clamp_yaw_to_band", "tgt_quat_w"),
              ("insertion_math.clamp_tip_in_box", "tgt_pos_w")],
          str([(f, t) for _, f, t in _orient_seq]))

    _box_calls = _e_calls(env_tree, "insertion_math.clamp_tip_in_box")
    _box_args = [[_unparse(a) for a in c.args] for c in _box_calls]
    check("the tip box gets x, y and z half-widths, z from its OWN cfg field",
          _box_args == [["tgt_pos_w", "tgt_quat_w", "self._tip_offset_local", "centre_w",
                         "float(self.cfg.osc_pos_clamp_m)",
                         "float(self.cfg.osc_pos_clamp_m)",
                         "float(self.cfg.osc_pos_clamp_z_m)"]],
          str(_box_args))
    check("the tip box is centred on THIS episode's pocket entrance",
          any(isinstance(n, ast.Assign)
              and [_unparse(t) for t in n.targets] == ["centre_w"]
              and _unparse(n.value) == "self._entrance_pos + self.scene.env_origins"
              for n in _walk(_func(env_tree, "_apply_osc", "InsertionEnv"))))
    _clamp_assigns = sorted(
        (_unparse(n.value.func), _unparse(n.targets[0]))
        for n in _walk(_func(env_tree, "_apply_osc", "InsertionEnv"))
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.value, ast.Call)
        and _unparse(n.value.func) in ("insertion_math.clamp_tilt_to_cone",
                                          "insertion_math.clamp_tip_in_box")
    )
    check("both clamps are ASSIGNED back, not called for a side effect they have not got",
          _clamp_assigns == [("insertion_math.clamp_tilt_to_cone", "tgt_quat_w"),
                             ("insertion_math.clamp_tip_in_box", "tgt_pos_w")],
          str(_clamp_assigns))

    _solve_fn = _func(env_tree, "_solve_start_pose", "InsertionEnv")
    _bad_rhs = [_unparse(n.value) for n in _walk(_solve_fn)
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and _unparse(n.targets[0]) == "_bad"]
    check("the start-solve residual is measured per env, not as a batch maximum",
          _bad_rhs == ["torch.linalg.norm(_goal[idx] - tip_rel[idx, 0:3], dim=-1) >= tol"],
          str(_bad_rhs))

    _goal_rows = {
        _unparse(n.targets[0]): _unparse(n.value)
        for n in _walk(_solve_fn)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and _unparse(n.targets[0]).startswith("_goal[")
    }
    check("the residual's goal is this reset's own lateral offset and height, per axis",
          _goal_rows == {
              "_goal[:, 0]": "self._start_lat_off[:, 0]",
              "_goal[:, 1]": "self._start_lat_off[:, 1]",
              "_goal[:, 2]": "self._start_height",
          },
          str(sorted(_goal_rows.items())))

    _withdrawals = sorted(
        (_unparse(n.targets[0]), _unparse(n.value), _e_enclosing_if(n))
        for n in _walk(_solve_fn)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and _unparse(n.targets[0]) in ("self._boundary_of[_rows]", "self._bounds_stamp[_rows]")
    )
    check("an unconverged start solve withdraws BOTH the boundary flag and the stamp",
          [(t, v) for t, v, _ in _withdrawals] == [
              ("self._boundary_of[_rows]", "-1"),
              ("self._bounds_stamp[_rows]", "-1"),
          ]
          and {g for _, _, g in _withdrawals} == {"_n_bad"},
          str(_withdrawals))

    _rows_rhs = [_unparse(n.value) for n in _walk(_solve_fn)
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and _unparse(n.targets[0]) == "_rows"]
    check("the withdrawn rows are ENV IDS (idx[_bad]), not mask positions",
          _rows_rhs == ["idx[_bad]"], str(_rows_rhs))

    _depth_seed = [n.lineno for n in _walk(_solve_fn)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and _unparse(n.targets[0]) == "self._max_depth[idx]"]
    _bad_line = [n.lineno for n in _walk(_solve_fn)
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and _unparse(n.targets[0]) == "_bad"]
    check("the withdrawal reads the pose the solve actually reached",
          len(_depth_seed) == 1 and len(_bad_line) == 1 and _bad_line[0] > _depth_seed[0],
          f"depth seed at {_depth_seed}, residual at {_bad_line}")

    _loop_lines = {
        line for loop in _walk(_solve_fn) if isinstance(loop, ast.For)
        for n in _walk(loop)
        for line in (getattr(n, "lineno", None),) if line is not None
    }
    _fresh_reads = [
        n.lineno for n in _walk(_solve_fn)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and _unparse(n.value) == "self._peg_geometry()"
        and "tip_rel" in _unparse(n.targets[0])
        and n.lineno not in _loop_lines
    ]
    check("a FRESH _peg_geometry read sits between the solve loop and the depth seed",
          len(_depth_seed) == 1
          and any(_depth_seed[0] > line > max(_loop_lines, default=0) for line in _fresh_reads)
          if _fresh_reads and _depth_seed else False,
          f"fresh reads outside the loop at {_fresh_reads}, loop ends "
          f"{max(_loop_lines, default=0)}, depth seed at {_depth_seed}")

    _dropped_keys = [
        _unparse(n) for n in _walk(env_tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and "flags_dropped" in n.value
    ]
    check("the dropped-booking count reaches the log and demo_metrics.json",
          sorted(set(_dropped_keys)) == ["'dr/start_solve_flags_dropped'",
                                         "'start_pose_solve_flags_dropped_envs'"],
          str(sorted(set(_dropped_keys))))

    _dropped_incr = [_unparse(n.value) for n in _walk(_solve_fn)
                     if isinstance(n, ast.AugAssign)
                     and _unparse(n.target) == "self._start_solve_flags_dropped"]
    check("the dropped-booking count adds the SAME quantity the guard tested",
          _dropped_incr == ["_n_bad"], str(_dropped_incr))

    _n_bad_rhs = [_unparse(n.value) for n in _walk(_solve_fn)
                  if isinstance(n, ast.Assign) and len(n.targets) == 1
                  and _unparse(n.targets[0]) == "_n_bad"]
    check("the dropped-booking count is the MASK's sum, not the size of the reset",
          _n_bad_rhs == ["int(_bad.sum().item())"], str(_n_bad_rhs))

    _dropped_values = []
    for _n in _walk(env_tree):
        if (isinstance(_n, ast.Assign) and len(_n.targets) == 1
                and "flags_dropped" in _unparse(_n.targets[0])):
            _dropped_values.append((_unparse(_n.targets[0]), _unparse(_n.value)))
        elif isinstance(_n, ast.Dict):
            for _k, _v in zip(_n.keys, _n.values):
                if (isinstance(_k, ast.Constant) and isinstance(_k.value, str)
                        and "flags_dropped" in _k.value):
                    _dropped_values.append((_k.value, _unparse(_v)))
    check("both published dropped-booking figures read the withdrawal counter itself",
          sorted(_dropped_values) == [
              ("self._start_solve_flags_dropped", "0"),
              ("self.extras['log']['dr/start_solve_flags_dropped']",
               "float(self._start_solve_flags_dropped)"),
              ("start_pose_solve_flags_dropped_envs",
               "self._start_solve_flags_dropped"),
          ], str(sorted(_dropped_values)))

    def _bin_args(method: str) -> list[str]:
        fn = next((n for n in _walk(env_tree)
                   if isinstance(n, ast.FunctionDef) and n.name == method), None)
        if fn is None:
            return []
        calls = [n for n in _walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "_rate_by_bin"]
        return [_unparse(a) for a in calls[0].args] if calls else []
    _succ_args = _bin_args("_success_by_start_height_bin")
    _abort_args = _bin_args("_force_abort_by_start_height_bin")
    check("the abort table bins the ABORTS over the success table's start-height edges",
          len(_succ_args) >= 3 and len(_abort_args) >= 3
          and _succ_args[0] == "self._recent_successes"
          and _abort_args[0] == "self._recent_aborts"
          and _succ_args[1:3] == _abort_args[1:3]
          == ["self._recent_start_height_m", "self._start_height_bin_edges_m"],
          f"success {_succ_args[:3]} abort {_abort_args[:3]}")
    _metrics_fn = _func(env_tree, "_write_metrics", "InsertionEnv")
    _abort_key_value = None
    for node in _walk(_metrics_fn):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "force_abort_by_start_height_bin":
                    _abort_key_value = _unparse(v)
    check("demo_metrics.json feeds force_abort_by_start_height_bin from the abort table",
          _abort_key_value == "self._force_abort_by_start_height_bin()",
          f"value {_abort_key_value}")

    _lat_args = _bin_args("_success_by_lateral_bin")
    _yaw_args = _bin_args("_success_by_yaw_bin")
    check("the lateral table bins the successes over the lateral buffer and its own edges",
          len(_lat_args) >= 3
          and _lat_args[0] == "self._recent_successes"
          and _lat_args[1] == "self._recent_lateral_mm"
          and _lat_args[2] == "self._lateral_bin_edges_mm",
          f"lateral {_lat_args[:3]}")
    check("the yaw table bins the successes over the yaw buffer and its own edges",
          len(_yaw_args) >= 3
          and _yaw_args[0] == "self._recent_successes"
          and _yaw_args[1] == "self._recent_yaw_deg"
          and _yaw_args[2] == "self._yaw_bin_edges_deg",
          f"yaw {_yaw_args[:3]}")
    _log_fn = _func(env_tree, "_log_finished_episodes", "InsertionEnv")
    _log_src = _unparse(_log_fn) if _log_fn is not None else ""
    check("_log_finished_episodes fills the lateral buffer from _start_lat_off",
          "self._recent_lateral_mm.extend" in _log_src
          and "self._start_lat_off[idx]" in _log_src,
          "extend/_start_lat_off present" if _log_src else "method not found")
    check("_log_finished_episodes fills the yaw buffer from _fixture_yaw",
          "self._recent_yaw_deg.extend" in _log_src
          and "self._fixture_yaw[idx]" in _log_src,
          "extend/_fixture_yaw present" if _log_src else "method not found")
    _lat_key_value = _yaw_key_value = None
    for node in _walk(_metrics_fn):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "success_by_lateral_bin":
                    _lat_key_value = _unparse(v)
                if isinstance(k, ast.Constant) and k.value == "success_by_yaw_bin":
                    _yaw_key_value = _unparse(v)
    check("demo_metrics.json feeds success_by_lateral_bin and success_by_yaw_bin "
          "from those two tables",
          _lat_key_value == "self._success_by_lateral_bin()"
          and _yaw_key_value == "self._success_by_yaw_bin()",
          f"lateral {_lat_key_value} yaw {_yaw_key_value}")

    _tilt_args = _bin_args("_success_by_tilt_bin")
    check("the tilt table bins the successes over the tilt buffer and its own edges",
          len(_tilt_args) >= 3
          and _tilt_args[0] == "self._recent_successes"
          and _tilt_args[1] == "self._recent_tilt_deg"
          and _tilt_args[2] == "self._tilt_bin_edges_deg",
          f"tilt {_tilt_args[:3]}")


    _mode_default = _class_attr_const(cfg_tree, "InsertionEnvCfg", "control_mode")
    check("the cfg defaults to the decided controller (control_mode = osc)",
          _mode_default == "osc", f"control_mode = {_mode_default!r}")
    _cone_expr = _cfg_field_expr("osc_tilt_clamp_rad")
    check("the tilt cone is written as the CAD degrees, not a typed radian",
          _cone_expr == "math.radians(8.52)", f"osc_tilt_clamp_rad = {_cone_expr}")
    _rot_step_expr = _cfg_field_expr("osc_rot_step_limit_rad")
    check("the rotation step limit is written as Factory's typed radian",
          _rot_step_expr == "0.097", f"osc_rot_step_limit_rad = {_rot_step_expr}")
    _reach = _autodr_reach(autodr_mut)
    _half_xy = _class_attr_const(cfg_tree, "InsertionEnvCfg", "osc_pos_clamp_m")
    _half_z = _class_attr_const(cfg_tree, "InsertionEnvCfg", "osc_pos_clamp_z_m")
    _step = _class_attr_const(cfg_tree, "InsertionEnvCfg", "osc_pos_step_limit_m")
    _seat_chain = _module_chain(
        tasks_tree,
        ("STAGE1_DEPTH", "STAGE2_DEPTH", "POCKET_FLOOR_Z", "POCKET_SEAT_DEPTH"),
        seed={"math": math},
    )
    _seat = _seat_chain.get("POCKET_SEAT_DEPTH")
    _tilt = _reach["tilt"][1]
    _h = _reach["start_height"][1]
    _r = _reach["lat_r"][1]
    _reach_xy = _r + _h * math.sin(_tilt)
    _reach_z = _h * math.cos(_tilt) + _r * math.sin(_tilt)
    check("the tip clamp box holds one action step beyond the SIDEWAYS reach",
          None not in (_half_xy, _step) and _half_xy >= _reach_xy + _step,
          f"half_xy {_half_xy} vs world-x reach {_reach_xy:.6f} + step {_step}")
    check("the tip clamp box holds one action step above the highest legal start",
          None not in (_half_z, _step) and _half_z >= _reach_z + _step,
          f"half_z {_half_z} vs world-z reach {_reach_z:.6f} + step {_step}")
    check("the tip clamp box still reaches the seat below the entrance",
          None not in (_half_z, _seat) and _half_z >= float(_seat),
          f"half_z {_half_z} vs seat {_seat}")
    _home_tip = _module_chain(
        tasks_tree,
        ("HOME_STANDOFF_Z", "STAGE1_DEPTH", "WORKCELL_HOME_TIP_ABOVE_ENTRANCE"),
        seed={"math": math},
    ).get("WORKCELL_HOME_TIP_ABOVE_ENTRANCE")
    _home_bound = (min([float(_home_tip)] + ([MEASURED_HOME_TIP_M]
                                              if MEASURED_HOME_TIP_M is not None else []))
                   if _home_tip is not None else None)
    _z_spread = _class_attr_const(cfg_tree, "InsertionEnvCfg", "fixture_pos_noise_z")
    _box_top = (None if None in (_half_z, _z_spread)
                else float(_half_z) + float(_z_spread))
    check("the tip clamp box stays BELOW the home tip standoff",
          None not in (_box_top, _home_bound) and _box_top < _home_bound,
          f"half_z {_half_z} + z spread {_z_spread} = {_box_top} vs bound {_home_bound} "
          f"(nominal {_home_tip}, measured "
          f"{MEASURED_HOME_TIP_M if MEASURED_HOME_TIP_M is not None else 'pending RT-242a'})")
    check("the z half-width is WIDER than the x/y one",
          None not in (_half_xy, _half_z) and _half_z > _half_xy,
          f"half_z {_half_z} vs half_xy {_half_xy}")
    _need_xy = None if _step is None else _reach_xy + _step
    check("the x/y half-width holds the disk reach plus one step and at most the set air",
          None not in (_half_xy, _need_xy)
          and _need_xy <= _half_xy <= _need_xy + XY_CLAMP_AIR_MAX_M,
          f"half_xy {_half_xy} vs need {_need_xy} (reach {_reach_xy:.6f} + step {_step}), "
          f"air at most {XY_CLAMP_AIR_MAX_M}")
    _yaw_lo, _yaw_hi = _reach["yaw"]
    _yaw_start = (max(abs(_yaw_lo), abs(_yaw_hi))
                  + math.atan((1.0 - math.cos(_tilt)) / (2.0 * math.sqrt(math.cos(_tilt)))))
    _rot_step = _fold_radians(_class_attr_expr(cfg_tree, "InsertionEnvCfg", "osc_rot_step_limit_rad"))
    _yaw_band = _fold_radians(_class_attr_expr(cfg_tree, "InsertionEnvCfg", "osc_yaw_clamp_rad"))
    _belief_yaw = _pocket_obs_yaw_rad()
    _need_yaw = None if _rot_step is None else _yaw_start + _belief_yaw + _rot_step
    _yaw_detail = (f"band {_yaw_band} vs need {_need_yaw} (start {math.degrees(_yaw_start):.4f} deg + belief "
                   f"{math.degrees(_belief_yaw):.4f} deg + step {_rot_step})")
    check("the yaw band holds the worst start yaw, the belief error and one rotation step",
          None not in (_yaw_band, _need_yaw) and _yaw_band >= _need_yaw, _yaw_detail)
    check("the yaw band carries at most the set air above its need",
          None not in (_yaw_band, _need_yaw) and _yaw_band <= _need_yaw + YAW_CLAMP_AIR_MAX_RAD,
          f"{_yaw_detail}, air at most {math.degrees(YAW_CLAMP_AIR_MAX_RAD):.2f} deg")
    _play = min(x for x in (_reward_nums.get("PLAY_X"), _reward_nums.get("PLAY_Y"))
                if x is not None) if _reward_nums.get("PLAY_X") is not None else None
    _tol = _class_attr_const(cfg_tree, "InsertionEnvCfg", "start_pose_solve_tol_m")
    check("the start-solve tolerance stays inside the pocket play",
          None not in (_tol, _play) and 0.0 < float(_tol) < float(_play),
          f"start_pose_solve_tol_m {_tol} vs play {_play}")
    _osc_ph = _module_const(cfg_tree, "RL_PLACEHOLDERS") or {}
    _osc_names = ("osc_kp_pos", "osc_kp_rot", "osc_pos_step_limit_m",
                  "osc_rot_step_limit_rad", "osc_pos_clamp_m", "osc_pos_clamp_z_m",
                  "osc_yaw_clamp_rad", "osc_decimation")
    check("every OSC number without a source is a labelled placeholder",
          all(n in _osc_ph for n in _osc_names),
          f"missing {[n for n in _osc_names if n not in _osc_ph]}")

    _init_fn = _func(env_tree, "__init__", "InsertionEnv")
    check("the control mode is resolved BEFORE the base class reads the cfg",
          _assign_precedes_call(_init_fn, "_control_mode", "__init__"))
    check("the observation noise model is set BEFORE the base class reads the cfg",
          _assign_precedes_call(_init_fn, "observation_noise_model", "__init__"))

    _scene_fn = _func(env_tree, "_setup_scene", "InsertionEnv")
    _art_line = None
    for _n in _walk(_scene_fn):
        if isinstance(_n, ast.Call) and isinstance(_n.func, ast.Name) and _n.func.id == "Articulation":
            _art_line = _n.lineno
    _gains_zeroed = False
    for _n in _walk(_scene_fn):
        if isinstance(_n, ast.If) and "_control_mode" in _identifiers(_n.test) and "osc" in _string_constants(_n.test):
            _zeroed = {
                t.attr for st in _walk(_n) if isinstance(st, ast.Assign)
                for t in st.targets if isinstance(t, ast.Attribute)
                and isinstance(st.value, ast.Constant) and st.value.value == 0.0
            }
            if {"stiffness", "damping"} <= _zeroed and _art_line is not None and _n.lineno < _art_line:
                _gains_zeroed = True
    check("under osc the drive stiffness AND damping are zeroed before the Articulation is built",
          _gains_zeroed)

    _apply_fn = _func(env_tree, "_apply_action", "InsertionEnv")
    _apply_calls = {n.func.attr for n in _walk(_apply_fn)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    _apply_branches = any(
        isinstance(n, ast.If) and "_control_mode" in _identifiers(n.test) for n in _walk(_apply_fn)
    )
    check("_apply_action branches on the control mode: OSC per physics step, else the PD target",
          _apply_branches and {"_apply_osc", "set_joint_position_target"} <= _apply_calls,
          f"calls {sorted(_apply_calls)}")

    _osc_fn = _func(env_tree, "_apply_osc", "InsertionEnv")
    _osc_calls = [n.func.attr for n in _walk(_osc_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    _need = ["apply_pose_delta", "clamp_tilt_to_cone", "clamp_tip_in_box", "set_command", "compute",
             "set_joint_effort_target"]
    _order_ok = all(n in _osc_calls for n in _need) and (
        _osc_calls.index("apply_pose_delta") < _osc_calls.index("clamp_tilt_to_cone")
        < _osc_calls.index("clamp_tip_in_box") < _osc_calls.index("set_command")
        < _osc_calls.index("set_joint_effort_target")
    )
    check("_apply_osc: delta, cone, box, then the OSC, then joint EFFORTS -- in that order",
          _order_ok, f"calls {_osc_calls}")

    _build_fn = _func(env_tree, "_build_task_space_controller", "InsertionEnv")
    _osc_kw: dict = {}
    for _n in _walk(_build_fn):
        if isinstance(_n, ast.Call) and isinstance(_n.func, ast.Name) and _n.func.id == "OperationalSpaceControllerCfg":
            _osc_kw = {kw.arg: _unparse(kw.value) for kw in _n.keywords if kw.arg}
    _want_kw = {
        "target_types": "['pose_abs']",
        "impedance_mode": "'fixed'",
        "inertial_dynamics_decoupling": "True",
        "partial_inertial_dynamics_decoupling": "False",
        "gravity_compensation": "True",
        "nullspace_control": "'none'",
    }
    _kw_bad = {k: _osc_kw.get(k) for k, v in _want_kw.items() if _osc_kw.get(k) != v}
    check("the OSC cfg is Decision (2)/(5): pose_abs, fixed, full decoupling, gravity comp ON, nullspace none",
          not _kw_bad, f"off: {_kw_bad}" if _kw_bad else "")

    _pre_fn = _func(env_tree, "_pre_physics_step", "InsertionEnv")
    _pre_ids = _identifiers(_pre_fn)
    check("the OSC delta is scaled by BOTH per-step limits in _pre_physics_step",
          {"_osc_delta", "osc_pos_step_limit_m", "osc_rot_step_limit_rad"} <= _pre_ids)

    _ema_assigns = [n for n in _walk(_pre_fn) if isinstance(n, ast.Assign)
                    and any(_unparse(t) == "self._ema_actions" for t in n.targets)]
    _ema_ok = (len(_ema_assigns) == 1
               and isinstance(_ema_assigns[0].value, ast.Call)
               and _unparse(_ema_assigns[0].value.func) == "insertion_math.ema_update"
               and [_unparse(a) for a in _ema_assigns[0].value.args]
               == ["self._ema_actions", "clamped", "float(self.cfg.action_ema_factor)"])
    check("the action EMA is updated once per step from the CLAMPED draw, the cfg factor on the new sample",
          _ema_ok, " / ".join(_unparse(n) for n in _ema_assigns) or "no assignment to self._ema_actions")
    _clamped_loads = sum(1 for n in _walk(_pre_fn)
                         if isinstance(n, ast.Name) and n.id == "clamped" and isinstance(n.ctx, ast.Load))
    _applied_src = [_unparse(n.value) for n in _walk(_pre_fn) if isinstance(n, ast.Assign)
                    and any(("_osc_delta" in _unparse(t) or "_joint_targets" in _unparse(t))
                            for t in n.targets)]
    _applied_ema = sum(1 for s in _applied_src if "self._ema_actions" in s)
    check("the OSC delta and the joint target read the EMA state; the clamped draw feeds ONLY the EMA",
          _clamped_loads == 1 and _applied_ema == 3,
          f"{_clamped_loads} load(s) of clamped, {_applied_ema} applied assignment(s) read the EMA state")
    _ema_init_src = _unparse(_func(env_tree, "__init__", "InsertionEnv"))
    check("the action EMA state is one row per env over the action width",
          "self._ema_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)"
          in _ema_init_src)
    check("_reset_idx zeroes the action EMA on the reset envs (Factory, factory_env.py:809)",
          "self._ema_actions[env_ids] = 0.0" in _unparse(_func(env_tree, "_reset_idx", "InsertionEnv")))
    _pre_src = _unparse(_pre_fn)
    check("the action-rate measurement stays on the RAW draw, not the EMA state",
          "self._last_action_rate = torch.linalg.norm(self.actions - self._prev_actions, dim=-1)" in _pre_src
          and "self._prev_actions = self.actions.clone()" in _pre_src)
    _ema_expr = _cfg_field_expr("action_ema_factor")
    check("action_ema_factor is Factory's ema_factor, 0.2 (factory_env_cfg.py:51)",
          _ema_expr == "0.2", f"action_ema_factor = {_ema_expr}")
    _rate_expr = _cfg_field_expr("action_rate_scale")
    check("action_rate_scale is Factory's action_grad_penalty_scale, 0.0 (factory_tasks_cfg.py:71)",
          _rate_expr == "0.0", f"action_rate_scale = {_rate_expr}")

    _ctrl_fn = _func(env_tree, "_print_controller_report", "InsertionEnv")
    _ctrl_attrs = {n.attr for n in _walk(_ctrl_fn) if isinstance(n, ast.Attribute)}
    check("the startup report reads the drives BACK from PhysX and prints Lambda and cond(J)",
          {"get_dof_stiffnesses", "get_dof_dampings", "get_dof_max_forces", "eigvalsh", "svdvals"} <= _ctrl_attrs,
          f"missing {sorted({'get_dof_stiffnesses', 'get_dof_dampings', 'get_dof_max_forces', 'eigvalsh', 'svdvals'} - _ctrl_attrs)}")
    check("the startup report prints the yaw band and the reset yaw against the seated orientation",
          {"osc_yaw_clamp_rad", "_seat_quat_local", "rotate_into_frame"} <= _ctrl_attrs,
          f"missing {sorted({'osc_yaw_clamp_rad', '_seat_quat_local', 'rotate_into_frame'} - _ctrl_attrs)}")
    check("the startup report prints the action EMA factor (D-197)",
          "action_ema_factor" in _ctrl_attrs)
    _report_calls = {n.func.attr for n in _walk(_report_fn)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    check("the startup report prints the controller block", "_print_controller_report" in _report_calls)

    def _mode_branch_calls(tree: ast.AST, osc_call: str) -> bool:
        for n in _walk(tree):
            if isinstance(n, ast.If) and "control_mode" in _identifiers(n.test):
                inner = {c.func.attr for c in _walk(n) if isinstance(c, ast.Call)
                         and isinstance(c.func, ast.Attribute)}
                if osc_call in inner and "joint_delta_to_action" in inner:
                    return True
        return False

    check("tilt_insert branches on control_mode: OSC action or the IK chain",
          _mode_branch_calls(tilt_tree, "osc_action_from_pocket_command"))
    check("scripted_insert branches on control_mode: OSC action or the IK chain",
          _mode_branch_calls(scripted_tree, "osc_action_from_pocket_command"))
    _tilt_flags = _cli_flags(tilt_tree)
    check("tilt_insert takes the controller overrides as flags (no hydra here)",
          {"--control-mode", "--osc-kp-pos", "--osc-kp-rot", "--decimation"} <= _tilt_flags)
    _tilt_attrs = {n.attr for n in _walk(tilt_tree) if isinstance(n, ast.Attribute)}
    check("tilt_insert logs the joint torques against maxForce",
          {"applied_torque", "joint_effort_limits"} <= _tilt_attrs)
    check("zero_agent can pin the home pose for the RT-46 droop form",
          "--home-pose" in _cli_flags(zero_tree))

    return results


WRAP_GUARD = ('    if _dr is not None:\n'
              '        runner.save = _save_with_autodr(\n')


MUTATIONS: tuple[tuple[str, dict, tuple[str, ...]], ...] = (
    (
        "the-boundary-window-is-fed-by-the-regular-mask",
        {"env_mut": ((
            "            self._recent_boundary_successes.extend(successes[_is_b].tolist())\n",
            "            self._recent_boundary_successes.extend(successes[~_is_b].tolist())\n",
        ),)},
        ("the boundary window is fed by the BOUNDARY mask",),
    ),
    (
        "the-friction-window-moves-inside-the-provider-guard",
        {"env_mut": ((
            "        self._recent_friction.extend(self._friction_applied[idx].tolist())\n",
            "            self._recent_friction.extend(self._friction_applied[idx].tolist())\n",
        ),)},
        ("the friction window is filled in EVERY mode, outside the provider guard",),
    ),
    (
        "only-moves-get-an-autodr-line",
        {"env_mut": ((
            "            for _ev in _events:\n",
            "            for _ev in _events:\n                if not _ev.moved:\n                    continue\n",
        ),)},
        ("every PROCESSED buffer prints its line, moved or not",),
    ),
    (
        "the-home-pose-fallback-test-is-inverted",
        {"env_mut": ((
            "        _hi = self._start_tip_height if _hi is None else float(_hi)\n",
            "        _hi = self._start_tip_height if _hi is not None else float(_hi)\n",
        ),)},
        ("the seam falls back to the home height only when the cfg field IS None",),
    ),
    (
        "the-record-loop-indexes-device-tensors-again",
        {"env_mut": ((
            "            _b_keys = _b_of[_is_b].tolist()\n"
            "            _b_flags = successes[_is_b].tolist()\n"
            "            for _k, _f in zip(_b_keys, _b_flags):\n"
            "                self._dr.record(self._boundary_keys[_k], bool(_f))\n",
            "            for _j in torch.nonzero(_is_b, as_tuple=False).flatten().tolist():\n"
            "                self._dr.record(self._boundary_keys[int(_b_of[_j])], bool(successes[_j]))\n",
        ),)},
        (
            "the record loop runs on host values, not on device tensors",
            "the success flag is booked on the boundary that episode ran at",
        ),
    ),
    (
        "the-band-seam-invents-a-seventh-quantity",
        {"env_mut": ((
            '            "friction": (\n',
            '            "tilt_azimuth": (0.0, 1.0),\n            "friction": (\n',
        ),)},
        ("the band seam answers for every DR_DIMS quantity, and only those",),
    ),
    (
        "the-band-seam-branches-on-the-mode-string",
        {"env_mut": ((
            "        if self._dr is not None:\n            # WITH A PROVIDER its live bounds win",
            "        if str(self.cfg.dr_mode) == 'autodr':\n            # WITH A PROVIDER its live bounds win",
        ),)},
        ("the band seam reads the PROVIDER first, not the mode string",),
    ),
    (
        "start-floor-default-is-None",
        {"cfg_mut": ((
            "    start_floor_m: float = -0.030\n",
            "    start_floor_m: float | None = None\n",
        ),)},
        ("start_floor_m is a float cfg field defaulting to -0.030 m, "
         "and start_floor_steps is 5",),
    ),
    (
        "floor-accepts-a-second-lower-edge",
        {"env_mut": ((
            "                    raise ValueError(\n"
            "                        f\"start_floor_m ({_floor:+.4f} m) together with a static band \"",
            "                    _unused = (\n"
            "                        f\"start_floor_m ({_floor:+.4f} m) together with a static band \"",
        ),)},
        ("the floor refuses five ways: above the high, the success band, a second lower edge, "
         "reset noise, and angles or offset under dr_mode='off'",),
    ),
    (
        "autodr-provider-drops-the-floor",
        {"env_mut": ((
            "                ),\n                floor=_floor_spec,\n                stall_buffers=int(cfg.autodr_stall_buffers),\n            )\n        elif str(cfg.dr_mode) == \"fixed_dr\":",
            "                ),\n                stall_buffers=int(cfg.autodr_stall_buffers),\n            )\n        elif str(cfg.dr_mode) == \"fixed_dr\":",
        ),)},
        ("both AutoDR providers take the floor, and the No-DR one carries only start_height",),
    ),
    (
        "report-walks-DR_DIM_NAMES-again",
        {"env_mut": ((
            "            for _name in (d.name for d in self._dr.dims):\n",
            "            for _name in autodr.DR_DIM_NAMES:\n",
        ),)},
        ("the startup report's band table walks the provider's own dims",),
    ),
    (
        "stall-bar-defaults-to-zero",
        {"cfg_mut": (("    autodr_stall_buffers: int = 3\n",
                      "    autodr_stall_buffers: int = 0\n"),)},
        ("autodr_stall_buffers is an int cfg field defaulting to 3",),
    ),
    (
        "autodr-provider-drops-the-stall-bar",
        {"env_mut": ((
            "                floor=_floor_spec,\n                stall_buffers=int(cfg.autodr_stall_buffers),\n            )\n        elif str(cfg.dr_mode) == \"fixed_dr\":",
            "                floor=_floor_spec,\n            )\n        elif str(cfg.dr_mode) == \"fixed_dr\":",
        ),)},
        ("both AutoDR providers take the cfg stall bar",),
    ),
    (
        "metrics-drop-the-stall-bar",
        {"env_mut": ((
            '            "autodr_stall_buffers": int(self.cfg.autodr_stall_buffers),\n',
            "",
        ),)},
        ("the metrics dump carries autodr_stall_buffers",),
    ),
    (
        "metrics-drop-the-floor",
        {"env_mut": ((
            '            "start_floor_m": (float(self.cfg.start_floor_m) if self._start_floor_on else None),\n',
            "",
        ),)},
        ("the metrics dump carries start_floor_m and start_floor_steps",),
    ),
    (
        "run-tag-drops-the-floor",
        {"train_mut": ((
            '            parts.append(f"floor{_floor_v * 1000:+g}mm")\n',
            "            pass\n",
        ),)},
        ("the run tag names the floor in its own if, beside the start tag",),
    ),
    (
        "the-band-seam-returns-the-provider-alone",
        {"env_mut": ((
            "            _static.update(self._dr.bounds())\n",
            "            return self._dr.bounds()\n",
        ),)},
        ("the band seam reads the PROVIDER first, not the mode string",),
    ),
    (
        "the-report-reads-the-band-through-the-seam",
        {"env_mut": ((
            '            _lo, _hi = self._dr.bounds()["friction"]\n',
            '            _lo, _hi = self._live_bounds()["friction"]\n',
        ),)},
        ("only the seam and the startup report call self._dr.bounds()",
         "every randomised quantity reads the band through the one seam"),
    ),
    (
        "the-start-mapping-keeps-its-own-band",
        {"env_mut": ((
            "        u = self._reset_unit[idx]\n        _b = self._live_bounds()\n",
            "        u = self._reset_unit[idx]\n"
            "        _b = {'start_height': (0.03, 0.03), 'lat_r': (0.0, 0.0)}\n",
        ),)},
        ("every randomised quantity reads the band through the one seam",),
    ),
    (
        "the-tilt-band-loses-its-one-sidedness",
        {"env_mut": ((
            '            "tilt": (0.0, _tilt),\n',
            '            "tilt": (-_tilt, _tilt),\n',
        ),)},
        ("with no provider the seam rebuilds the static cfg bands",),
    ),
    (
        "the-zero-width-height-band-takes-the-sampling-arm",
        {"env_mut": ((
            "        _lo = _hi if _lo is None or float(_lo) >= _hi else float(_lo)\n",
            "        _lo = _hi if _lo is None or float(_lo) > _hi else float(_lo)\n",
        ),)},
        ("a low bound at or above the high one collapses to the fixed height",),
    ),
    (
        "the-pocket-yaw-is-drawn-off-the-tilt-column",
        {"env_mut": ((
            '                u[:, self._col["yaw"]], float(_y_lo), float(_y_hi)\n',
            '                u[:, self._col["tilt"]], float(_y_lo), float(_y_hi)\n',
        ),)},
        ("the pocket yaw is mapped onto the live band from the yaw column",),
    ),
    (
        "the-tilt-magnitude-is-drawn-off-the-yaw-column",
        {"env_mut": ((
            '                    u[:, self._col["tilt"]], float(_t_lo), float(_t_hi)\n',
            '                    u[:, self._col["yaw"]], float(_t_lo), float(_t_hi)\n',
        ),)},
        ("the tilt magnitude is mapped onto the live band from the tilt column",),
    ),
    (
        "the-tilt-guard-reads-the-cfg-field-again",
        {"env_mut": ((
            "            if self._tilt_reach:\n",
            "            if self.cfg.fixture_tilt_noise_rad > 0.0:\n",
        ),)},
        ("the tilt quaternion is built behind the REACH, not behind a cfg field",),
    ),
    (
        "no-dr-runs-get-a-tilt-reach-anyway",
        {"env_mut": ((
            "            self._tilt_reach = float(cfg.fixture_tilt_noise_rad) > 0.0\n",
            "            self._tilt_reach = True\n",
        ),)},
        ("the tilt reach asks bounds_max under a provider and the cfg field otherwise",),
    ),
    (
        "the-stamp-moves-above-the-nail",
        {"env_mut": (
            ("        _rows = idx[_is_b]\n",
             "        self._bounds_stamp[idx] = self._dr.bounds_version\n"
             "        _rows = idx[_is_b]\n"),
            ("        self._bounds_stamp[idx] = self._dr.bounds_version\n"
             "\n    def _map_start_conditions",
             "\n    def _map_start_conditions"),
        )},
        ("the draw marks boundary envs and nails them, in that order",),
    ),
    (
        "the-boundary-choice-reuses-the-selection-draw",
        {"env_mut": ((
            "        _is_b, _which = self._dr.boundary_assignment(_rb[:, 0], _rb[:, 1])\n",
            "        _is_b, _which = self._dr.boundary_assignment(_rb[:, 0], _rb[:, 0])\n",
        ),)},
        ("the boundary draw uses two INDEPENDENT columns",),
    ),
    (
        "the-nail-uses-local-positions-not-env-ids",
        {"env_mut": ((
            "        _rows = idx[_is_b]\n",
            "        _rows = torch.nonzero(_is_b, as_tuple=False).flatten()\n",
        ),)},
        ("the nail gets the whole row buffer, absolute env ids and one selection",),
    ),
    (
        "regular-envs-keep-last-episodes-boundary-mark",
        {"env_mut": ((
            "            _is_b, _which, torch.full_like(_which, -1)\n",
            "            _is_b, _which, self._boundary_of[idx]\n",
        ),)},
        ("every resetting env gets a boundary mark, -1 when it is regular",),
    ),
    (
        "the-stamp-is-a-constant",
        {"env_mut": ((
            "        self._bounds_stamp[idx] = self._dr.bounds_version\n",
            "        self._bounds_stamp[idx] = 0\n",
        ),)},
        ("the stamp is the provider's CURRENT bounds version",),
    ),
    (
        "the-boundary-keys-are-sorted-out-of-the-provider-order",
        {"env_mut": ((
            "            self._boundary_keys = tuple(self._dr.keys)\n",
            "            self._boundary_keys = tuple(sorted(self._dr.keys))\n",
        ),)},
        ("columns, sides and keys are the provider's, in the provider's order",),
    ),
    (
        "the-boundary-mark-is-a-float-buffer",
        {"env_mut": ((
            "            (self.num_envs,), -1, dtype=torch.long, device=self.device\n",
            "            (self.num_envs,), -1, device=self.device\n",
        ),)},
        ("the mark and the stamp are integer buffers, the mark starting at -1",),
    ),
    (
        "update-runs-before-record",
        {"env_mut": (
            ("            _events = self._dr.update()\n", ""),
            ("            _b_keys = _b_of[_is_b].tolist()\n",
             "            _events = self._dr.update()\n"
             "            _b_keys = _b_of[_is_b].tolist()\n"),
        )},
        ("record runs before update, and the fresh window is filled before it",),
    ),
    (
        "the-flag-is-booked-by-the-loop-counter",
        {"env_mut": ((
            "                self._dr.record(self._boundary_keys[_k], bool(_f))\n",
            "                self._dr.record(self._boundary_keys[_f], bool(_k))\n",
        ),)},
        (
            "the success flag is booked on the boundary that episode ran at",
            "the record loop runs on host values, not on device tensors",
        ),
    ),
    (
        "the-fresh-window-forgets-the-stamp",
        {"env_mut": ((
            "            _fresh = (~_is_b) & (self._bounds_stamp[idx] == self._dr.bounds_version)\n",
            "            _fresh = ~_is_b\n",
        ),)},
        ("the fresh window takes regular episodes AND the current stamp only",),
    ),
    (
        "the-fresh-window-is-cleared-on-any-event",
        {"env_mut": ((
            "            if any(_ev.moved for _ev in _events):\n",
            "            if _events:\n",
        ),)},
        ("the fresh window is cleared only when a boundary actually MOVED",),
    ),
    (
        "the-fresh-rate-averages-the-ordinary-window",
        {"env_mut": ((
            "                sum(self._fresh_successes) / _fresh_n\n",
            "                sum(self._recent_successes) / len(self._recent_successes)\n",
        ),)},
        ("the fresh rate averages the FRESH window and nothing else",),
    ),
    (
        "the-fresh-count-reports-the-minimum",
        {"env_mut": ((
            '            self.extras["log"]["dr/fresh_n"] = float(_fresh_n)\n',
            '            self.extras["log"]["dr/fresh_n"] = float(self._fresh_min)\n',
        ),)},
        ("the fresh count and the minimum are their own curves",),
    ),
    (
        "the-boundary-rate-averages-the-wrong-window",
        {"env_mut": ((
            '            self.extras["log"]["dr/success_rate_boundary"] = (\n'
            "                sum(self._recent_boundary_successes)\n",
            '            self.extras["log"]["dr/success_rate_boundary"] = (\n'
            "                sum(self._fresh_successes)\n",
        ),)},
        ("the boundary rate averages the boundary window, on its own key",),
    ),
    (
        "the-commanded-friction-loses-its-reader-again",
        {"env_mut": ((
            "        self._recent_friction.extend(self._friction_applied[idx].tolist())\n",
            "",
        ),)},
        (
            "the commanded friction reaches the log from the per-episode buffer",
            "the friction window is filled in EVERY mode, outside the provider guard",
        ),
    ),
    (
        "the-provider-scalars-replace-the-log-dict",
        {"env_mut": ((
            '            self.extras["log"].update(self._dr.scalars())\n',
            '            self.extras["log"] = self._dr.scalars()\n',
        ),)},
        ("the provider's scalars are MERGED into the existing log dict",),
    ),
    (
        "the-log-block-drops-the-provider-guard",
        {"env_mut": ((
            "        # and which read 0.0 there.\n        if self._dr is not None:\n",
            "        # and which read 0.0 there.\n        if True:\n",
        ),)},
        ("every provider call in the log sits behind `self._dr is not None`",),
    ),
    (
        "the-fresh-minimum-is-typed-in-again",
        {"env_mut": ((
            "        self._fresh_min = window_floor\n",
            "        self._fresh_min = 2000\n",
        ),)},
        ("the fresh minimum and the window floor are one number",),
    ),
    (
        "the-fresh-window-is-shorter-than-its-own-minimum",
        {"env_mut": ((
            "        self._fresh_successes: collections.deque = collections.deque(maxlen=window)\n",
            "        self._fresh_successes: collections.deque = collections.deque(maxlen=500)\n",
        ),)},
        ("the three B5 windows share the trailing-window size",),
    ),
    (
        "the-dumped-minimum-is-a-literal",
        {"env_mut": ((
            '                    "min": self._fresh_min,\n',
            '                    "min": 2000,\n',
        ),)},
        ("the metrics dump carries the fresh window, its minimum and all_at_max",),
    ),
    (
        "the-dumped-fresh-rate-is-ungated",
        {"env_mut": ((
            "                        if len(self._fresh_successes) >= self._fresh_min\n",
            "                        if self._fresh_successes\n",
        ),)},
        ("the dumped fresh rate is None until the window holds the minimum",),
    ),
    (
        "the-cached-material-buffer-is-aliased",
        {"env_mut": ((
            "            _mats = _asset.root_physx_view.get_material_properties().clone()\n",
            "            _mats = _asset.root_physx_view.get_material_properties()\n",
        ),)},
        ("the cached material buffers are CLONED, not aliased",),
    ),
    (
        "the-d111-write-hits-restitution",
        {"env_mut": ((
            "            _mats[..., 1] = insertion_tasks_cfg.CONTACT_FRICTION  # dynamic\n",
            "            _mats[..., 2] = insertion_tasks_cfg.CONTACT_FRICTION  # dynamic\n",
        ),)},
        ("the one-shot D-111 write fills the two FRICTION columns",),
    ),
    (
        "the-host-copy-dtype-is-typed-in",
        {"env_mut": ((
            "        self._friction_buf_dtype = next(iter(_dtypes))\n",
            "        self._friction_buf_dtype = torch.float32\n",
        ),)},
        ("both material buffers are pinned to ONE dtype at cache time",),
    ),
    (
        "the-push-index-stays-on-the-device",
        {"env_mut": ((
            '        idx_cpu = idx.to(device="cpu", dtype=torch.long)\n',
            '        idx_cpu = idx.to(dtype=torch.long)\n',
        ),)},
        ("the push index is moved to the HOST as int64",),
    ),
    (
        "the-host-copy-goes-back-inside-the-loop",
        {"env_mut": (
            ("        _col = mu.to(device=\"cpu\", dtype=self._friction_buf_dtype).unsqueeze(1)\n", ""),
            ("        for _asset, _buf in self._friction_bufs:\n",
             "        for _asset, _buf in self._friction_bufs:\n"
             "            _col = mu.to(device=\"cpu\", dtype=self._friction_buf_dtype).unsqueeze(1)\n"),
        )},
        ("the drawn column is copied to the host ONCE, outside the asset loop",),
    ),
    (
        "the-write-cost-timer-loses-its-fence",
        {"env_mut": ((
            "        if mu.is_cuda:\n            torch.cuda.synchronize()\n", "",
        ),)},
        ("the write-cost timer is fenced against the queued reset work",),
    ),
    (
        "the-write-cost-is-appended-per-asset",
        {"env_mut": ((
            "            _asset.root_physx_view.set_material_properties(_buf, idx_cpu)\n",
            "            _asset.root_physx_view.set_material_properties(_buf, idx_cpu)\n"
            "            self._friction_write_ms.append((time.perf_counter() - t0) * 1000.0)\n",
        ),)},
        (
            "the write cost is appended ONCE per reset call, outside the loop",
            "the write-cost timer spans the whole block, both pushes inside",
        ),
    ),
    (
        "single-env-resets-skip-the-friction-write",
        {"env_mut": ((
            "        if idx.numel() == 0:\n"
            "            return\n"
            "        # The LIVE band, not the reach: the reach decides whether this method\n",
            "        if idx.numel() <= 1:\n"
            "            return\n"
            "        # The LIVE band, not the reach: the reach decides whether this method\n",
        ),)},
        ("the friction write skips only the EMPTY reset",),
    ),
    (
        "the-friction-write-covers-every-env",
        {"env_mut": ((
            "            self._apply_friction(idx)\n",
            "            self._apply_friction(self.robot._ALL_INDICES)\n",
        ),)},
        ("the friction write runs AFTER the draw, on the resetting envs only",),
    ),
    (
        "the-logged-mean-divides-by-the-window",
        {"env_mut": ((
            "                sum(self._friction_write_ms) / len(self._friction_write_ms)\n"
            "                if self._friction_write_ms\n",
            "                sum(self._friction_write_ms) / self._friction_write_ms.maxlen\n"
            "                if self._friction_write_ms\n",
        ),)},
        ("the logged write cost is a mean over the SAMPLES and a true maximum",),
    ),
    (
        "demo-metrics-reports-the-cheapest-reset-as-the-worst",
        {"env_mut": ((
            '                    "max": max(self._friction_write_ms),\n',
            '                    "max": min(self._friction_write_ms),\n',
        ),)},
        ("demo_metrics records mean, max, the sample count and the window",),
    ),
    (
        "the-write-cost-window-shrinks-to-ten",
        {"env_mut": ((
            "        self._friction_write_ms: collections.deque = collections.deque(maxlen=1000)\n",
            "        self._friction_write_ms: collections.deque = collections.deque(maxlen=10)\n",
        ),)},
        ("the write-cost window holds 1000 reset calls",),
    ),
    (
        "the-commanded-friction-buffer-starts-at-zero",
        {"env_mut": ((
            "        self._friction_applied = torch.full(\n"
            "            (self.num_envs,), float(insertion_tasks_cfg.CONTACT_FRICTION), device=self.device\n"
            "        )\n",
            "        self._friction_applied = torch.zeros(self.num_envs, device=self.device)\n",
        ),)},
        ("the commanded-friction buffer starts at the nominal value",),
    ),
    (
        "the-report-compares-physx-against-itself",
        {"env_mut": ((
            "            _cmd = [float(v) for v in self._friction_applied[:_n_rep].tolist()]\n",
            "            _cmd = [float(v) for v in _read]\n",
        ),)},
        ("the report compares PhysX against the COMMANDED buffer, over four envs",),
    ),
    (
        "the-report-prints-one-env-instead-of-four",
        {"env_mut": ((
            "            _n_rep = min(4, self.num_envs)\n",
            "            _n_rep = min(1, self.num_envs)\n",
        ),)},
        ("the report compares PhysX against the COMMANDED buffer, over four envs",),
    ),
    (
        "the-d111-header-claims-the-centre-was-written",
        {"env_mut": ((
            "            if self._friction_reach\n"
            '            else f"u = {insertion_tasks_cfg.CONTACT_FRICTION} written once, PROVISIONAL/F3"\n',
            "            if False\n"
            '            else f"u = {insertion_tasks_cfg.CONTACT_FRICTION} written once, PROVISIONAL/F3"\n',
        ),)},
        ("the D-111 friction header depends on the reach",),
    ),
    (
        "the-friction-reach-is-read-off-the-live-band",
        {"env_mut": ((
            '            _fric = self._dr.bounds_max().get(\n',
            '            _fric = self._dr.bounds().get(\n',
        ),)},
        (
            "the per-reset friction write is gated by the REACH, not by today's band",
            "only the seam and the startup report call self._dr.bounds()",
        ),
    ),
    (
        "the-friction-guard-forgets-the-reach",
        {"env_mut": ((
            "        if ready and self._friction_reach:\n",
            "        if ready and self._dr is not None:\n",
        ),)},
        ("the per-reset friction write is gated by the REACH, not by today's band",),
    ),
    (
        "no-dr-runs-get-a-friction-reach-anyway",
        {"env_mut": ((
            "            self._friction_reach = False\n",
            "            self._friction_reach = True\n",
        ),)},
        ("dr_mode='off' has no friction reach",),
    ),
    (
        "the-friction-draw-uses-the-widest-band",
        {"env_mut": ((
            '        lo, hi = self._live_bounds()["friction"]\n',
            '        lo, hi = self._dr.bounds_max()["friction"]\n',
        ),)},
        (
            "the friction draw maps this reset onto the LIVE band",
            "every randomised quantity reads the band through the one seam",
        ),
    ),
    (
        "the-friction-draw-reads-the-start-height-column",
        {"env_mut": ((
            '            self._reset_unit[idx][:, self._col["friction"]], float(lo), float(hi)\n',
            '            self._reset_unit[idx][:, self._col["start_height"]], float(lo), float(hi)\n',
        ),)},
        ("the friction value is mapped off the friction column",),
    ),
    (
        "kinetic-friction-drifts-off-static",
        {"env_mut": ((
            "            _buf[idx_cpu, :, 1] = _col  # dynamic\n",
            "            _buf[idx_cpu, :, 1] = _col * 0.8  # dynamic\n",
        ),)},
        ("static and dynamic friction get the SAME drawn value",),
    ),
    (
        "only-the-robot-gets-the-drawn-friction",
        {"env_mut": ((
            "        for _asset, _buf in self._friction_bufs:\n",
            "        for _asset, _buf in self._friction_bufs[:1]:\n",
        ),)},
        ("the friction write covers the robot AND the fixture",),
    ),
    (
        "the-material-push-hands-over-a-sliced-buffer",
        {"env_mut": ((
            "            _asset.root_physx_view.set_material_properties(_buf, idx_cpu)\n",
            "            _asset.root_physx_view.set_material_properties(_buf[idx_cpu], idx_cpu)\n",
        ),)},
        ("the material push hands over the WHOLE buffer plus the subset of ids",),
    ),
    (
        "the-reset-re-reads-the-material-buffer",
        {"env_mut": ((
            "            _buf[idx_cpu, :, 0] = _col  # static\n",
            "            _buf = _asset.root_physx_view.get_material_properties()\n"
            "            _buf[idx_cpu, :, 0] = _col  # static\n",
        ),)},
        ("the reset never re-reads the material buffer from PhysX",),
    ),
    (
        "the-write-cost-is-reported-as-zero",
        {"env_mut": (
            ("        t0 = time.perf_counter()\n", ""),
            ("        self._friction_write_ms.append((time.perf_counter() - t0) * 1000.0)\n",
             "        self._friction_write_ms.append(0.0)\n"),
        )},
        (
            "the write-cost timer spans the whole block, both pushes inside",
            "the write-cost timer is fenced against the queued reset work",
        ),
    ),
    (
        "the-commanded-friction-is-recorded-as-the-band-edge",
        {"env_mut": ((
            "        self._friction_applied[idx] = mu\n",
            "        self._friction_applied[idx] = float(lo)\n",
        ),)},
        ("the commanded friction is recorded per env, from the same draw",),
    ),
    (
        "the-friction-read-back-pools-the-envs-again",
        {"env_mut": ((
            "            _read = [float(_fric_r[_e, :, 0].mean()) for _e in range(_n_rep)]\n",
            "            _read = [float(_fric_r[:, :, 0].mean()) for _e in range(_n_rep)]\n",
        ),)},
        ("the startup report reads the applied friction back PER ENV, both assets",),
    ),
    (
        "the-friction-read-back-drops-the-fixture",
        {"env_mut": ((
            "            _read_f = [float(_fric_f[_e, :, 0].mean()) for _e in range(_n_rep)]\n",
            "            _read_f = list(_read)\n",
        ),)},
        ("the startup report reads the applied friction back PER ENV, both assets",),
    ),
    (
        "the-write-cost-loses-its-maximum",
        {"env_mut": ((
            '            "dr/friction_write_ms_max": (\n',
            '            "dr/friction_write_ms_mean_again": (\n',
        ),)},
        (
            "the write cost reaches the log as mean AND max",
            "the logged write cost is a mean over the SAMPLES and a true maximum",
        ),
    ),
    (
        "the-negative-low-refusal-reads-the-fixture-noise",
        {"env_mut": ((
            "if _low < 0.0 and self._start_lat_half != 0.0:",
            "if _low < 0.0 and float(cfg.fixture_pos_noise_xy) != 0.0:",
        ),)},
        ("both lateral guards test the radius itself",),
    ),
    (
        "the-report-prints-the-radius-in-metres-as-millimetres",
        {"env_mut": ((
            "DISK of radius {self._start_lat_half * 1000.0:.3f} mm",
            "DISK of radius {self._start_lat_half:.3f} mm",
        ),)},
        ("the startup report names the lateral radius in millimetres",),
    ),
    (
        "the-metrics-dump-records-the-radius-as-a-pair",
        {"env_mut": ((
            '"start_lateral_offset_m": self._start_lat_half,',
            '"start_lateral_offset_m": [self._start_lat_half, self._start_lat_half],',
        ),)},
        ("the metrics dump records the lateral radius as one number",),
    ),
    (
        "the-start-pose-goal-spends-x-on-y",
        {"env_mut": ((
            "d_pocket[:, 1] = self._start_lat_off[:, 1] - tip_rel[:, 1]",
            "d_pocket[:, 1] = self._start_lat_off[:, 0] - tip_rel[:, 1]",
        ),)},
        ("the start-pose goal spends the lateral draw axis for axis",),
    ),
    (
        "the-run-tag-guard-tags-a-zero-radius",
        {"train_mut": ((
            "    if _lat > 0.0:",
            "    if _lat >= 0.0:",
        ),)},
        ("the run tag names the lateral radius",),
    ),
    (
        "the-run-tag-names-the-radius-in-metres",
        {"train_mut": ((
            'f"lat{_lat * 1000:g}mm"',
            'f"lat{_lat:g}mm"',
        ),)},
        ("the run tag names the lateral radius",),
    ),
    (
        "the-lateral-write-goes-back-behind-a-cfg-guard",
        {"env_mut": ((
            "        self._start_lat_off[idx] = insertion_math.disk_offset(\n"
            "            u[:, self._col[\"lat_r\"]], u[:, self._col[\"lat_phi\"]], float(_b[\"lat_r\"][1])\n"
            "        )\n",
            "        if self._start_lat_half != 0.0:\n"
            "            self._start_lat_off[idx] = insertion_math.disk_offset(\n"
            "                u[:, self._col[\"lat_r\"]], u[:, self._col[\"lat_phi\"]], float(_b[\"lat_r\"][1])\n"
            "            )\n",
        ),)},
        ("the lateral start row is written at EVERY reset, behind no guard",),
    ),
    (
        "the-disk-radius-and-angle-columns-are-swapped",
        {"env_mut": ((
            'u[:, self._col["lat_r"]], u[:, self._col["lat_phi"]], float(_b["lat_r"][1])',
            'u[:, self._col["lat_phi"]], u[:, self._col["lat_r"]], float(_b["lat_r"][1])',
        ),)},
        ("the lateral draw maps the radius and angle columns onto a disk of the LIVE radius",),
    ),
    (
        "the-disk-reads-the-cfg-radius-not-the-live-band",
        {"env_mut": ((
            'u[:, self._col["lat_r"]], u[:, self._col["lat_phi"]], float(_b["lat_r"][1])',
            'u[:, self._col["lat_r"]], u[:, self._col["lat_phi"]], float(self._start_lat_half)',
        ),)},
        ("the lateral draw maps the radius and angle columns onto a disk of the LIVE radius",),
    ),
    (
        "the-env-reads-the-raw-lateral-field-again",
        {"env_mut": ((
            "            if _low < 0.0 and self._start_lat_half != 0.0:\n",
            "            if _low < 0.0 and float(cfg.start_lateral_offset) != 0.0:\n",
        ),)},
        ("the env resolves start_lateral_offset once and reads it nowhere else",
         "both lateral guards test the radius itself"),
    ),
    (
        "the-lateral-cfg-default-is-a-tuple",
        {"cfg_mut": ((
            "    start_lateral_offset: float = 0.0",
            "    start_lateral_offset: float = (0.0, 0.0)",
        ),)},
        ("start_lateral_offset defaults to a bare number, so BOTH hydra forms merge",),
    ),
    (
        "an-unknown-dr-mode-falls-back-to-off",
        {"env_mut": ((
            '        elif str(cfg.dr_mode) != "off":\n',
            '        elif str(cfg.dr_mode) == "table":\n',
        ),)},
        ("an unknown dr_mode is refused, never run as 'off'",),
    ),
    (
        "the-start-height-centre-is-the-lower-bound",
        {"env_mut": ((
            '            self._dr = autodr.AutoDR(\n                autodr.bind_centres(\n'
            '                    {\n                        "lat_r": 0.0,\n'
            '                        "yaw": 0.0,\n                        "tilt": 0.0,\n'
            '                        "start_height": float(cfg.start_tip_above_entrance),\n',
            '            self._dr = autodr.AutoDR(\n                autodr.bind_centres(\n'
            '                    {\n                        "lat_r": 0.0,\n'
            '                        "yaw": 0.0,\n                        "tilt": 0.0,\n'
            '                        "start_height": float(cfg.start_tip_above_entrance_low),\n',
        ),)},
        ("the AutoDR centres are bound to the five named sources",
         "the table and Fixed-DR providers open around the SAME five centres as AutoDR"),
    ),
    (
        "the-solver-draws-its-own-start-height-again",
        {"env_mut": ((
            "        tol = float(self.cfg.start_pose_solve_tol_m)\n"
            "        n_joints = len(self.robot.joint_names)\n",
            "        self._start_height[idx] = float(self.cfg.start_tip_above_entrance) "
            "* torch.rand(1, device=self.device)\n"
            "        tol = float(self.cfg.start_pose_solve_tol_m)\n"
            "        n_joints = len(self.robot.joint_names)\n",
        ),)},
        ("ONE place draws the reset conditions -- _draw_reset_conditions",
         "the solver REALISES the start pose, it does not decide it"),
    ),
    (
        "the-start-height-is-mapped-off-the-wrong-column",
        {"env_mut": ((
            'u[:, self._col["start_height"]], float(_h_lo), float(_h_hi)',
            'u[:, self._col["friction"]], float(_h_lo), float(_h_hi)',
        ),)},
        ("the start height is mapped per env from its own column of the draw",),
    ),
    (
        "the-draw-runs-after-its-first-consumer",
        {"env_mut": ((
            "        if ready:\n"
            "            self._draw_reset_conditions(idx)\n"
            "\n"
            "        super()._reset_idx(env_ids)\n",
            "        super()._reset_idx(env_ids)\n",
        ), (
            "        if ready:\n"
            "            # Map the start conditions into their per-env buffers FIRST;",
            "        if ready:\n"
            "            self._draw_reset_conditions(idx)\n"
            "            # Map the start conditions into their per-env buffers FIRST;",
        ))},
        ("the draw runs before the FIRST consumer of the row",),
    ),
    (
        "the-reset-row-width-is-typed-in-by-hand",
        {"env_mut": ((
            "            (self.num_envs, len(autodr.TABLE_COLUMNS)), device=self.device",
            "            (self.num_envs, 15), device=self.device",
        ),)},
        ("the reset row is sized from autodr.TABLE_COLUMNS, never a literal",),
    ),
    (
        "yaw-bin-table-reads-the-tilt-buffer",
        {"env_mut": (("            self._recent_successes, self._recent_yaw_deg, "
                      "self._yaw_bin_edges_deg,",
                      "            self._recent_successes, self._recent_tilt_deg, "
                      "self._yaw_bin_edges_deg,"),)},
        ("the yaw table bins the successes over the yaw buffer and its own edges",),
    ),
    (
        "tilt-bin-table-reads-the-yaw-edges",
        {"env_mut": (("            self._recent_successes, self._recent_tilt_deg, "
                      "self._tilt_bin_edges_deg,",
                      "            self._recent_successes, self._recent_tilt_deg, "
                      "self._yaw_bin_edges_deg,"),)},
        ("the tilt table bins the successes over the tilt buffer and its own edges",),
    ),
    (
        "lateral-buffer-never-filled",
        {"env_mut": (("        self._recent_lateral_mm.extend(\n"
                      "            (torch.linalg.norm(self._start_lat_off[idx], dim=-1) "
                      "* 1000.0).tolist()\n        )\n", ""),)},
        ("_log_finished_episodes fills the lateral buffer from _start_lat_off",),
    ),
    (
        "rung0-start-height-typed-out",
        {"cfg_mut": (("    start_tip_above_entrance: float | None = (\n"
                      "        insertion_tasks_cfg.RUNG0_START_TIP_ABOVE_ENTRANCE\n",
                      "    start_tip_above_entrance: float | None = (\n"
                      "        0.030\n"),)},
        ("the rung-0 start height is derived, not typed",),
    ),
    (
        "rung0-start-height-back-inside-the-pocket",
        {"tasks_mut": (("RUNG0_START_TIP_ABOVE_ENTRANCE = 0.020",
                        "RUNG0_START_TIP_ABOVE_ENTRANCE = -0.026"),)},
        ("the rung-0 start height clears the fixture's stage-1 rim",),
    ),
    (
        "start-pose-solve-never-called",
        {"env_mut": (("            self._solve_start_pose(env_ids)\n", ""),)},
        ("_reset_idx solves the start pose",
         "the start pose is solved AFTER the fixture pose is written",
         "the mapping runs before the solve"),
    ),
    (
        "start-pose-solved-before-the-fixture-moves",
        {"env_mut": (("            self._solve_start_pose(env_ids)\n", ""),
                     ("        super()._reset_idx(env_ids)\n",
                      "        super()._reset_idx(env_ids)\n"
                      "        if ready:\n"
                      "            self._solve_start_pose(env_ids)\n"))},
        (
            "the start pose is solved AFTER the fixture pose is written",
            "the mapping runs before the solve",
        ),
    ),
    (
        "start-pose-solve-steps-the-sim",
        {"env_mut": (("                self._joint_targets[idx] = sub\n",
                      "                self.sim.step(render=False)\n"
                      "                self._joint_targets[idx] = sub\n"),)},
        ("the start-pose solve never steps the sim",),
    ),
    (
        "start-pose-leaves-the-integrator-at-home",
        {"env_mut": (("                self._joint_targets[idx] = sub\n", ""),)},
        ("the start pose seeds the joint-target integrator",),
    ),
    (
        "teleport-run-loses-the-home-pose-pin",
        {"scripted_mut": (("    env_cfg.start_tip_above_entrance = None\n", ""),)},
        ("the teleport runs pin the start pose back to the home pose",),
    ),
    (
        "start-height-missing-from-the-metrics-file",
        {"env_mut": (('            "start_tip_above_entrance_mm": (\n', '            "_dropped": (\n'),)},
        ("demo_metrics.json carries the start height",),
    ),
    (
        "start-height-mapped-off-the-bound-not-the-draw",
        {"env_mut": ((
            "autodr.map_unit_to_bounds(\n"
            "            u[:, self._col[\"start_height\"]], float(_h_lo), float(_h_hi)\n"
            "        )",
            "float(_h_hi)",
        ),)},
        ("the start height is mapped per env from its own column of the draw",),
    ),
    (
        "start-pose-solve-commands-the-bound-not-the-draw",
        {"env_mut": (("                d_pocket[:, 2] = self._start_height - tip_rel[:, 2]\n",
                      "                d_pocket[:, 2] = high - tip_rel[:, 2]\n"),)},
        ("the start-pose solve commands the per-env height",),
    ),
    (
        "lower-start-bound-guarded-against-the-floor-not-the-band",
        {"env_mut": (("            if _low <= -float(cfg.depth_min):\n",
                      "            if _low <= -float(cfg.depth_max):\n"),)},
        ("the lower start bound is guarded above the success band",),
    ),
    (
        "off-mode-angle-reach-loses-the-static-yaw-probe",
        {"env_mut": (("                or float(cfg.fixture_yaw_rad) != 0.0\n",
                      ""),)},
        (
            "the off-mode reaches are the frozen pre-Phase-5 expressions, exactly",
        ),
    ),
    (
        "off-mode-pose-reach-loses-the-d034-xy-noise",
        {"env_mut": (("                float(cfg.fixture_pos_noise_xy) > 0.0\n",
                      "                float(cfg.fixture_yaw_noise_rad) > 0.0\n"),)},
        (
            "the off-mode reaches are the frozen pre-Phase-5 expressions, exactly",
        ),
    ),
    (
        "provider-pose-reach-drops-the-un-adapted-xy-noise",
        {"env_mut": (("            _fixture_pose_reach = (_angle_reach or float(cfg.fixture_pos_noise_xy) > 0.0\n",
                      "            _fixture_pose_reach = (_angle_reach\n"),)},
        (
            "the provider reaches are the exact bounds_max expressions",
        ),
    ),
    (
        "provider-angle-reach-asks-only-the-yaw-boundary",
        {"env_mut": (('            _tilt_span = _tilt_r[1] - _tilt_r[0] > 0.0\n',
                      '            _tilt_span = False\n'),)},
        (
            "the provider reaches are the exact bounds_max expressions",
        ),
    ),
    (
        "autodr-refusal-lets-the-tilt-noise-through",
        {"env_mut": (('                "fixture_tilt_noise_rad": float(cfg.fixture_tilt_noise_rad),\n',
                      ''),)},
        (
            "the autodr refusal names every boundary-owned static field, and reads it",
        ),
    ),
    (
        "autodr-refusal-ignores-the-start-height-band",
        {"env_mut": (("            _low_static = cfg.start_tip_above_entrance_low\n",
                      "            _low_static = None\n"),)},
        (
            "the autodr refusal covers the start-height lower bound too",
        ),
    ),
    (
        "metrics-file-cannot-tell-the-two-modes-apart",
        {"env_mut": (('            "dr_mode": str(self.cfg.dr_mode),\n',
                      ''),)},
        (
            "the metrics dump records which source randomised the run",
        ),
    ),
    (
        "startup-report-drops-the-wiring-line",
        {"env_mut": (('            print(f"  ALL FIVE QUANTITIES ARE WIRED (Phase 5 step B5): every reset maps "\n',
                      '            print(f"  the reset draws "\n'),)},
        (
            "the startup report says all five DR quantities are wired",
        ),
    ),
    (
        "startup-report-drops-the-unverified-stamp",
        {"env_mut": ((
            '            print("  NOT VERIFIED IN A SIMULATOR: this build has never run. Read the "\n'
            '                  "[autodr] lines and dr/ curves of THIS run before believing it.")\n',
            "",
        ),)},
        ("the provider block stamps itself NOT VERIFIED IN A SIMULATOR",),
    ),
    (
        "reach-read-from-the-width-0-bounds",
        {"env_mut": (("            _reach = self._dr.bounds_max()\n",
                      "            _reach = self._dr.bounds()\n"),)},
        (
            "the reach comes from bounds_max, never from the width-0 bounds",
            "the provider reaches are the exact bounds_max expressions",
            "only the seam and the startup report call self._dr.bounds()",
        ),
    ),
    (
        "tilt-buffers-sized-from-a-cfg-angle-field",
        {"env_mut": (("        self._tilt_active = _angle_reach\n",
                      "        self._tilt_active = float(cfg.fixture_tilt_noise_rad) != 0.0\n"),)},
        (
            "both B2 flags are assigned the reach itself, once, and nothing else",
        ),
    ),
    (
        "fixture-pose-guard-back-on-the-cfg-ranges",
        {"env_mut": (("        if ready and self._fixture_pose_reach:\n",
                      "        if ready and self.cfg.fixture_pos_noise_xy > 0.0:\n"),)},
        (
            "the fixture-pose reset guard reads the reach flag, not the cfg ranges",
        ),
    ),
    (
        "no-off-mode-provider-default",
        {"env_mut": (("        self._dr = None\n",
                      "        self._dr_provider = None\n"),)},
        (
            "the bounds provider is built in __init__ and stored on self._dr",
        ),
    ),
    (
        "unknown-dr-mode-runs-as-off",
        {"env_mut": (('        elif str(cfg.dr_mode) != "off":\n',
                      '        elif False:\n'),)},
        (
            "an unknown dr_mode is refused, never run as 'off'",
        ),
    ),
    (
        "autodr-merges-with-the-static-ranges",
        {"env_mut": (("            if _live:\n",
                      "            if False:\n"),)},
        (
            "dr_mode='autodr' together with a static range is refused",
        ),
    ),
    (
        "startup-report-hides-the-dr-mode",
        {"env_mut": (('        print(f"dr_mode: {cfg.dr_mode}")\n',
                      '        print(f"dr mode: {cfg.dr_mode}")\n'),)},
        (
            "the startup report names dr_mode and the provider bounds",
        ),
    ),
    (
        "dr-mode-defaults-to-autodr",
        {"cfg_mut": (('    dr_mode: str = "off"\n',
                      '    dr_mode: str = "autodr"\n'),)},
        (
            "dr_mode is a cfg field and defaults to 'off'",
        ),
    ),
    (
        "inside-start-accepts-yaw-noise",
        {"env_mut": (("                or float(cfg.reset_yaw_noise) != 0.0\n"
                      "                or _angle_reach\n",
                      "                or _angle_reach\n"),)},
        ("a start inside the pocket refuses every orientation noise and the angle reach",),
    ),
    (
        "teleport-depth-paid-as-progress",
        {"env_mut": (("            self._max_depth[idx] = (-tip_rel[idx, 2]).clamp(min=0.0)\n", ""),)},
        ("a FRESH _peg_geometry read sits between the solve loop and the depth seed",
         "the start-pose solve seeds the gated max depth with the start depth",
         "the withdrawal reads the pose the solve actually reached"),
    ),
    (
        "lower-start-bound-defaults-to-none",
        {"cfg_mut": (("    start_tip_above_entrance_low: float | None = (\n"
                      "        insertion_tasks_cfg.RUNG0_START_TIP_ABOVE_ENTRANCE\n"
                      "    )\n",
                      "    start_tip_above_entrance_low: float | None = None\n"),)},
        ("the lower start bound defaults to the derived upper bound, not None",),
    ),
    (
        "start-range-missing-from-the-metrics-file",
        {"env_mut": (('            "start_tip_above_entrance_low_mm": (\n', '            "_dropped_low": (\n'),)},
        ("demo_metrics.json and the run tag carry the start range",),
    ),
    (
        "block-tag-back-to-naming-OFF",
        {"train_mut": (('    if hasattr(env_cfg, "spawn_workcell_block") and bool(env_cfg.spawn_workcell_block):\n'
                        '        parts.append("blockON")',
                        '    if hasattr(env_cfg, "spawn_workcell_block") and not bool(env_cfg.spawn_workcell_block):\n'
                        '        parts.append("blockOFF")'),)},
        ("the run tag names the block only when it is ON",),
    ),
    (
        "scene-state-missing-from-the-metrics-file",
        {"env_mut": (('            "spawn_workcell_block": bool(self.cfg.spawn_workcell_block),\n', ""),)},
        ("demo_metrics.json carries the scene state",),
    ),
    (
        "proxy-sizes-back-in-the-run-tag",
        {"train_mut": (("    parts = []",
                        "    parts = []\n"
                        "    parts.append(f'peg{env_cfg.held_asset.side * 1000:g}mm')"),)},
        ("the run tag does NOT assert the proxy peg and pocket sizes",),
    ),
    (
        "seed-drops-out-of-the-run-tag",
        {"train_mut": (('        parts.append(f"seed{agent_cfg.seed}")',
                        '        parts.append("run")'),)},
        ("the run tag carries the seed",),
    ),
    (
        "env-count-back-in-the-run-tag",
        {"train_mut": (("    parts = []",
                        "    parts = []\n"
                        "    parts.append(f'env{env_cfg.scene.num_envs}')"),)},
        ("the run tag does NOT carry the env count",),
    ),
    (
        "resume-sorts-runs-alphabetically-again",
        {"train_mut": (("log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, sort_alpha=False",
                        "log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint"),)},
        ("the resume picks the newest run by time, not by name",),
    ),
    (
        "exposure-moved-below-the-rl-terms-return",
        {"env_mut": (
            ('        in_stage1 = (tip_rel[:, 2] >= 0.0) & (\n            tip_rel[:, 2] <= insertion_tasks_cfg.STAGE1_DEPTH\n        )\n        self._max_stage1_lat_y = torch.maximum(\n            self._max_stage1_lat_y,\n            torch.where(\n                in_stage1, tip_rel[:, 1].abs(), torch.zeros_like(tip_rel[:, 1])\n            ),\n        )\n', ""),
            ("        force_norm = insertion_math.force_magnitude(self._force_smooth)",
             '        in_stage1 = (tip_rel[:, 2] >= 0.0) & (\n            tip_rel[:, 2] <= insertion_tasks_cfg.STAGE1_DEPTH\n        )\n        self._max_stage1_lat_y = torch.maximum(\n            self._max_stage1_lat_y,\n            torch.where(\n                in_stage1, tip_rel[:, 1].abs(), torch.zeros_like(tip_rel[:, 1])\n            ),\n        )\n' + "        force_norm = insertion_math.force_magnitude(self._force_smooth)"),
        )},
        ("the exposure is recorded above the reward-side work",),
    ),
    (
        "stage1-band-typed-by-hand",
        {"env_mut": (("            tip_rel[:, 2] <= insertion_tasks_cfg.STAGE1_DEPTH",
                      "            tip_rel[:, 2] <= 0.015"),)},
        ("the stage-1 band reads STAGE1_DEPTH, not a typed number",),
    ),
    (
        "exposure-not-cleared-on-reset",
        {"env_mut": (("            self._max_stage1_lat_y[env_ids] = 0.0\n", ""),)},
        ("the exposure buffer is cleared in _reset_idx",),
    ),
    (
        "reach-offset-typed-by-hand",
        {"env_mut": (("        reach = insertion_tasks_cfg.HOLE_REACH_OFFSET_Y",
                      "        reach = 0.0078847"),)},
        ("the exposure is judged against HOLE_REACH_OFFSET_Y",),
    ),
    (
        "exposure-missing-from-the-metrics-file",
        {"env_mut": (('            "stage1_lateral_y_mm": self._stage1_lateral_stats(),\n', ""),)},
        ("demo_metrics.json carries the stage-1 exposure",),
    ),
    (
        "block-spawns-without-the-flag",
        {"env_mut": (("        if self.cfg.spawn_workcell_block:",
                      "        if True:"),)},
        ("the workcell block spawns ONLY under cfg.spawn_workcell_block",),
    ),
    (
        "block-default-back-to-ON",
        {"cfg_mut": (("    spawn_workcell_block: bool = False",
                      "    spawn_workcell_block: bool = True"),)},
        ("the workcell block defaults to OFF",),
    ),
    (
        "report-stops-naming-the-scene",
        {"env_mut": (("                 if bool(self.cfg.spawn_workcell_block)",
                      "                 if bool(self.cfg.rl_terms_enabled)"),)},
        ("the startup report prints the workcell block state",),
    ),
    (
        "guard-blind-to-xy-noise-again",
        {"env_mut": (("                or float(cfg.fixture_pos_noise_xy) != 0.0",
                      "                or float(cfg.fixture_yaw_noise_rad) != 0.0"),)},
        ("the deterministic-vs-randomised guard counts fixture_pos_noise_xy",),
    ),
    (
        "guard-blind-to-z-noise",
        {"env_mut": (("                or float(cfg.fixture_pos_noise_z) != 0.0\n",
                      ""),)},
        ("the deterministic-vs-randomised guard counts fixture_pos_noise_z",),
    ),
    (
        "the-fixture-z-spread-is-never-applied",
        {"env_mut": (("            ) * self.cfg.fixture_pos_noise_z\n",
                      "            ) * 0.0\n"),)},
        ("the reset moves the entrance along z by the z spread, from its own table column",),
    ),
    (
        "gate-depth-back-to-the-proxy-constant",
        {"cfg_mut": ((
            "        depth=insertion_tasks_cfg.POCKET_SEAT_DEPTH,",
            "        depth=insertion_tasks_cfg.POCKET_DEPTH,",
        ),)},
        ("the gate depth is wired to the real fixture's seat depth",),
    ),
    (
        "seat-depth-typed-instead-of-derived",
        {"tasks_mut": (("POCKET_SEAT_DEPTH = -POCKET_FLOOR_Z", "POCKET_SEAT_DEPTH = 0.036"),)},
        ("the seat depth is derived from the measured POCKET_FLOOR_Z",),
    ),
    (
        "success-depth-back-to-the-proxy-constant",
        {"cfg_mut": ((
            "        success_depth=insertion_tasks_cfg.SEATED_SUCCESS_DEPTH,",
            "        success_depth=insertion_tasks_cfg.SUCCESS_DEPTH,",
        ),)},
        ("the success depth is wired to the D-106 seated band",),
    ),
    (
        "d106-band-inlined-as-a-literal",
        {"tasks_mut": ((
            "SEATED_SUCCESS_DEPTH = POCKET_SEAT_DEPTH - D106_DEPTH_BAND",
            "SEATED_SUCCESS_DEPTH = 0.033",
        ),)},
        ("the success depth is the seat depth minus the D-106 band",),
    ),
    (
        "d106-band-changed-behind-the-derivation",
        {"tasks_mut": (("D106_DEPTH_BAND = 0.003", "D106_DEPTH_BAND = 0.002"),)},
        ("the seated band comes out at 33 mm",),
    ),
    (
        "sapu-threshold-typed-instead-of-derived",
        {"tasks_mut": (("INTERPEN_THRESH = 0.5 * PLAY_X",
                        "INTERPEN_THRESH = 0.0002938"),)},
        ("the SAPU threshold is derived from the cross play",),
    ),
    (
        "sapu-threshold-takes-the-whole-play",
        {"tasks_mut": (("INTERPEN_THRESH = 0.5 * PLAY_X",
                        "INTERPEN_THRESH = 1.0 * PLAY_X"),)},
        ("the SAPU threshold is derived from the cross play",
         "the SAPU threshold comes out at half the play"),
    ),
    (
        "fine-kernel-width-typed-instead-of-derived",
        {"tasks_mut": (("KERNEL_A_FINE = math.acosh(10.0) / PLAY_X",
                        "KERNEL_A_FINE = 5094.0"),)},
        ("the fine kernel width is derived from the cross play",),
    ),
    (
        "fine-kernel-margin-swapped-for-the-long-play",
        {"tasks_mut": (("KERNEL_A_FINE = math.acosh(10.0) / PLAY_X",
                        "KERNEL_A_FINE = math.acosh(10.0) / PLAY_Y"),)},
        ("the fine kernel width is derived from the cross play",
         "the fine kernel width comes out at the decided 5094"),
    ),
    (
        "engaged-depth-typed-instead-of-derived",
        {"tasks_mut": (("ENGAGED_DEPTH = ENGAGED_DEPTH_FRACTION * POCKET_SEAT_DEPTH",
                        "ENGAGED_DEPTH = 0.0108"),)},
        ("the engaged depth is derived from the stage-2 seat depth",),
    ),
    (
        "engaged-fraction-back-to-factorys-peg-value",
        {"tasks_mut": (("ENGAGED_DEPTH_FRACTION = 0.30",
                        "ENGAGED_DEPTH_FRACTION = 0.10"),)},
        ("the engaged depth comes out at 10.8 mm",),
    ),
    (
        "engaged-depth-field-typed-into-the-cfg",
        {"cfg_mut": (("    engaged_depth_m: float = insertion_tasks_cfg.ENGAGED_DEPTH",
                      "    engaged_depth_m: float = 0.0108"),)},
        ("cfg.engaged_depth_m is wired to the derived constant",),
    ),
    (
        "real-success-weight-back-to-the-proxy-ten",
        {"cfg_mut": (("    w_success: float = 1.0", "    w_success: float = 10.0"),)},
        ("the real success weight is D-109's 1.0, not the proxy's",),
    ),
    (
        "episode-length-as-a-rounded-decimal",
        {"cfg_mut": (("    episode_length_s = 256 / 60", "    episode_length_s = 4.2667"),)},
        ("the episode cap is D-113's 256 control steps",
         "the episode length is written as a ratio, not a rounded decimal"),
    ),
    (
        "episode-length-back-to-the-proxy-240",
        {"cfg_mut": (("    episode_length_s = 256 / 60", "    episode_length_s = 240 / 60"),)},
        ("the episode cap is D-113's 256 control steps",),
    ),
    (
        "startup-report-never-outlives-the-first-episode",
        {"cfg_mut": (("    report_at_steps = (2, 60, 180, 258)",
                      "    report_at_steps = (2, 60, 180)"),)},
        ("the startup report is taken at least once past the episode cap",),
    ),
    (
        "the-seated-dump-drops-the-grasp-belief-error",
        {"seated_mut": ((
            '        "grasp_obs_offset_x_m": float(env_cfg.grasp_obs_offset_x_m),\n'
            '        "pocket_wall_y_mm": wall_y_mm,',
            '        "pocket_wall_y_mm": wall_y_mm,',
        ),)},
        ("the seated metrics dump names every pinned scatter field",),
    ),
    (
        "the-curve-dump-drops-the-grasp-belief-error",
        {"seated_mut": ((
            '        "grasp_obs_offset_x_m": float(env_cfg.grasp_obs_offset_x_m),\n'
            '        "curve_settle_steps": settle,',
            '        "curve_settle_steps": settle,',
        ),)},
        ("the reward-curve metrics dump names every pinned scatter field",),
    ),
    (
        "episode-cap-reported-in-seconds-only",
        {"env_mut": ((
            'print(f"episode cap (D-113 (3)): {self.max_episode_length} control steps "',
            'print(f"episode cap (D-113 (3)): "',
        ),)},
        ("the startup report prints the episode cap in STEPS",),
    ),
    (
        "the-force-depth-binning-is-dropped-from-the-loop",
        {"scripted_mut": ((
            "            sp.bin_force_by_depth(",
            "            _ = sp.force_magnitude if False else None; _skip = (",
        ),)},
        ("the scripted gate bins the force by depth",),
    ),
    (
        "the-force-depth-profile-is-written-but-not-printed",
        {"scripted_mut": ((
            'f"force max {_row[\'force_max_n\']:8.2f} N  mean {_row[\'force_mean_n\']:8.2f} N")',
            'f"")',
        ),)},
        ("the force-depth profile is PRINTED, not only written",),
    ),
    (
        "experiment-folder-back-to-the-demo-sprint",
        {"ppo_mut": (('experiment_name = "ur5e_insertion"', 'experiment_name = "demo_insertion"'),)},
        ("the experiment folder is not the proxy demo sprint's",
         "compare_runs reads the same experiment folder train.py writes"),
    ),
    (
        "lstm-entry-point-misnamed",
        {"init_mut": (('"rsl_rl_rnn_cfg_entry_point"', '"rsl_rl_lstm_cfg_entry_point"'),)},
        ("the LSTM agent is registered as rsl_rl_rnn_cfg_entry_point",),
    ),
    (
        "default-entry-point-trains-the-lstm",
        {"init_mut": (('rsl_rl_ppo_cfg:PPORunnerCfg"', 'rsl_rl_ppo_cfg:PPORunnerLstmCfg"'),)},
        ("the default agent entry point stays the 16-step MLP PPORunnerCfg",),
    ),
    (
        "ablation-entry-point-dropped",
        {"init_mut": (('        "rsl_rl_mlp_ablation_cfg_entry_point": '
                       'f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerMlpAblationCfg",\n', ''),)},
        ("the MLP ablation agent is registered as rsl_rl_mlp_ablation_cfg_entry_point",),
    ),
    (
        "lstm-cfg-skips-the-ablation-cfg",
        {"ppo_mut": (("class PPORunnerLstmCfg(PPORunnerMlpAblationCfg):",
                      "class PPORunnerLstmCfg(PPORunnerCfg):"),)},
        ("the LSTM cfg extends the ablation cfg, which extends PPORunnerCfg",),
    ),
    (
        "rollout-back-to-64",
        {"ppo_mut": (("    num_steps_per_env = 128\n", "    num_steps_per_env = 64\n"),)},
        ("the ablation cfg sets rollout 128, 400 iterations, 4 mini-batches (D-195 B1, BUDGET, D-198)",
         "the B1 budget is the user's 51,200 steps per env (400 x 128, D-195 BUDGET)"),
    ),
    (
        "iterations-back-to-the-pilots-225",
        {"ppo_mut": (("    max_iterations = 400\n", "    max_iterations = 225\n"),)},
        ("the ablation cfg sets rollout 128, 400 iterations, 4 mini-batches (D-195 B1, BUDGET, D-198)",
         "the B1 budget is the user's 51,200 steps per env (400 x 128, D-195 BUDGET)"),
    ),
    (
        "ablation-drops-its-mini-batch-line",
        {"ppo_mut": (("        self.algorithm.num_mini_batches = 4\n", "        pass\n"),)},
        ("the ablation cfg sets rollout 128, 400 iterations, 4 mini-batches (D-195 B1, BUDGET, D-198)",
         "the B1 budget is the user's 51,200 steps per env (400 x 128, D-195 BUDGET)",
         "the ablation cfg keeps the base cfg's mini-batch count, Isaac Lab's 4 (D-198)"),
    ),
    (
        "mini-batches-back-to-d195s-32",
        {"ppo_mut": (("        self.algorithm.num_mini_batches = 4\n",
                      "        self.algorithm.num_mini_batches = 32\n"),)},
        ("the ablation cfg sets rollout 128, 400 iterations, 4 mini-batches (D-195 B1, BUDGET, D-198)",
         "the ablation cfg keeps the base cfg's mini-batch count, Isaac Lab's 4 (D-198)"),
    ),
    (
        "critic-reads-the-policy-group",
        {"ppo_mut": (('"critic": ["critic"]}', '"critic": ["policy"]}'),)},
        ("the ablation cfg sends the env's critic group to the critic (D-195)",),
    ),
    (
        "lstm-cfg-drops-the-super-call",
        {"ppo_mut": (("    def __post_init__(self):\n        super().__post_init__()\n        self.policy = ",
                      "    def __post_init__(self):\n        self.policy = "),)},
        ("the LSTM cfg's __post_init__ runs the ablation cfg's first",),
    ),
    (
        "lstm-becomes-a-gru",
        {"ppo_mut": (('rnn_type="lstm",', 'rnn_type="gru",'),)},
        ("the LSTM policy is rsl_rl's recurrent actor-critic, lstm 2 x 256 (D-195 A1)",),
    ),
    (
        "lstm-copies-the-template-mlp",
        {"ppo_mut": (("actor_hidden_dims=self.policy.actor_hidden_dims,",
                      "actor_hidden_dims=[256, 128, 64],"),)},
        ("the LSTM policy's MLP part is read from the MLP policy, field by field",),
    ),
    (
        "critic-group-not-returned",
        {"env_mut": (('        return {"policy": obs, "critic": critic}\n',
                      '        return {"policy": obs}\n'),)},
        ("_get_observations returns exactly the policy and the critic group",),
    ),
    (
        "critic-reads-the-believed-tip",
        {"env_mut": (("            joint_pos, joint_vel_fd, tip_rel_true, ee_quat_w,",
                      "            joint_pos, joint_vel_fd, tip_rel, ee_quat_w,"),)},
        ("the critic group differs from the policy group only in the three pocket blocks",),
    ),
    (
        "critic-frame-carries-the-yaw-error",
        {"env_mut": (("            self._fixture_quat, torch.zeros(self.num_envs, device=self.device)\n",
                      "            self._fixture_quat, self._pocket_obs_yaw\n"),)},
        ("the critic's pocket frame is believed_pocket_frame at ZERO error",
         "the pocket-yaw belief error reaches the OBSERVATION and no paying path"),
    ),
    (
        "critic-tip-carries-the-grasp-error",
        {"env_mut": (("            self._tip_offset_local,\n            self.scene.env_origins,\n"
                      "            self._entrance_pos,\n            true_rot,\n",
                      "            self._tip_offset_local + self._grasp_obs_off,\n"
                      "            self.scene.env_origins,\n            self._entrance_pos,\n            true_rot,\n"),)},
        ("the critic's tip uses the NOMINAL tip offset in that true frame",
         "the grasp offset reaches the OBSERVATION and no paying path"),
    ),
    (
        "state-space-left-at-zero",
        {"cfg_mut": (("    cfg.state_space = width\n", ""),)},
        ("resolve_obs_layout writes state_space with the same width",),
    ),
    (
        "the-run-reader-is-left-behind-by-the-rename",
        {"compare_mut": (('default="ur5e_insertion"', 'default="insertion"'),)},
        ("compare_runs reads the same experiment folder train.py writes",),
    ),
    (
        "observation-width-drops-the-force-block",
        {"cfg_mut": (("    observation_space = insertion_math.OBS_DIM", "    observation_space = 25"),)},
        ("observation_space is the full OBS_SLICES width",),
    ),
    (
        "layout-marker-hardcoded-on-the-block-name-again",
        {"env_mut": ((
            '''            f"{lo}:{hi} {name}" + ("  <-- NOT BUILT" if lo >= _built else "")''',
            '''            f"{lo}:{hi} {name}" + ("  <-- MISSING (scene stream S2)" if name == "force" else "")''',
        ),)},
        ("the layout report derives its NOT-BUILT marker from the built width",
         "no block is marked missing by name in the layout report"),
    ),
    (
        "abort-table-bins-the-successes",
        {"env_mut": (("            self._recent_aborts, self._recent_start_height_m, self._start_height_bin_edges_m,",
                      "            self._recent_successes, self._recent_start_height_m, self._start_height_bin_edges_m,"),)},
        ("the abort table bins the ABORTS over the success table's start-height edges",),
    ),
    (
        "abort-key-fed-by-the-success-table",
        {"env_mut": (('            "force_abort_by_start_height_bin": self._force_abort_by_start_height_bin(),',
                      '            "force_abort_by_start_height_bin": self._success_by_start_height_bin(),'),)},
        ("demo_metrics.json feeds force_abort_by_start_height_bin from the abort table",),
    ),
    (
        "guard-call-removed",
        {"env_mut": (("            validate_rl_config(self.cfg)", "            pass"),)},
        ("the env validates the rl config in __init__",
         "the guard sits behind the rl_terms_enabled switch"),
    ),
    (
        "guard-not-behind-the-switch",
        {"env_mut": ((
            "        if self._rl_terms_enabled:\n            validate_rl_config(self.cfg)",
            "        validate_rl_config(self.cfg)",
        ),)},
        ("the guard sits behind the rl_terms_enabled switch",),
    ),
    (
        "termination-bypasses-compute-dones",
        {"env_mut": (("insertion_math.compute_dones(", "insertion_math_compute_dones("),)},
        ("the termination is insertion_math.compute_dones",),
    ),
    (
        "below-plate-exit-restored",
        {"env_mut": ((
            "        self._last_success = in_region",
            "        self._last_success = in_region\n"
            "        below_plate = depth > 0.05\n"
            "        terminated = terminated | below_plate",
        ),)},
        ("the below_plate exit is gone from the termination",),
    ),
    (
        "reward-pays-only-the-kernels",
        {"env_mut": ((
            "        terms = insertion_math.compute_reward_terms_insertion(",
            "        terms = insertion_math.kernel_sum(",
        ),)},
        (
            "the reward is insertion_math.compute_reward_terms_insertion",
            "the reward is handed the depth progress the metric produced",
        ),
    ),
    (
        "reward-terms-never-accumulated",
        {"env_mut": ((
            "        for i, name in enumerate(insertion_math.REWARD_TERMS):\n"
            "            self._episode_sums[name] += terms[i]\n",
            "",
        ),)},
        ("every reward term is accumulated per env",),
    ),
    (
        "reward-terms-never-cleared",
        {"env_mut": (("            self._episode_sums[name][idx] = 0.0\n", ""),)},
        ("the term sums are cleared at episode end",),
    ),
    (
        "reward-terms-not-logged",
        {"env_mut": (('"Episode_Reward/" + name', '"EpisodeReward/" + name'),)},
        ("the per-episode log carries the reward terms",),
    ),
    (
        "reward-terms-missing-from-the-metrics-file",
        {"env_mut": ((
            '            "reward_terms_recent_mean": self._reward_terms_stats(),\n',
            "",
        ),)},
        ("demo_metrics.json carries the reward terms",),
    ),
    (
        "reward-not-handed-the-progress",
        {"env_mut": ((
            "            self._last_depth_progress,\n            self._last_tilt_phi,",
            "            torch.zeros_like(self._last_depth_progress),\n            self._last_tilt_phi,",
        ),)},
        ("the reward is handed the depth progress the metric produced",),
    ),
    (
        "progress-computed-from-raw-depth",
        {"env_mut": ((
            "        self._last_depth_progress = self._max_depth - prev_max_depth",
            "        self._last_depth_progress = depth - prev_max_depth",
        ),)},
        ("the depth progress is the gated max-depth gain, not the raw depth",),
    ),
    (
        "progress-weight-unlabelled",
        {"cfg_mut": ((
            '    "cfg.w_depth_progress": (\n'
            '        "[proxy] the square-peg proxy\'s reward_w_depth, paid per metre of "\n'
            '        "NEW gated max depth (D-165). The proxy solved 1 mm clearance with it "\n'
            '        "at 100; nothing on this task has sized it. Zero switches the term "\n'
            '        "off and reproduces the pre-D-165 reward."\n'
            '    ),\n',
            "",
        ),)},
        ("the progress weight is labelled [proxy]",),
    ),
    (
        "proxy-reward-function-restored",
        {"env_mut": ((
            "class InsertionEnv(DirectRLEnv):",
            "def compute_rewards():\n    pass\n\n\nclass InsertionEnv(DirectRLEnv):",
        ),)},
        ("the proxy reward function is gone from the env",),
    ),
    (
        "reward-bypasses-the-switch",
        {"env_mut": ((
            "        if self._peg_body_idx is None or not self._rl_terms_enabled:",
            "        if self._peg_body_idx is None:",
        ),)},
        ("the reward pays zero in a measurement env",),
    ),
    (
        "abort-alive-in-the-measurement-env",
        {"env_mut": ((
            "        if not self._rl_terms_enabled:\n            terminated = torch.zeros_like(time_out)",
            "        if not True:\n            terminated = torch.zeros_like(time_out)",
        ),)},
        ("the force abort is off in a measurement env",),
    ),
    (
        "abort-reads-the-raw-wrench",
        {"env_mut": ((
            "        force_norm = insertion_math.force_magnitude(self._force_smooth)",
            "        force_norm = insertion_math.force_magnitude("
            "self.robot.data.body_incoming_joint_wrench_b[:, self._force_body_idx, 0:3])",
        ),)},
        ("the abort reads the EMA the policy sees",),
    ),
    (
        "sdf-queries-built-unguarded",
        {"env_mut": ((
            "        if self._rl_terms_enabled and self._peg_body_idx is not None:",
            "        if self._peg_body_idx is not None:",
        ),)},
        ("the SDF queries are built in __init__ behind the switch",),
    ),
    (
        "seat-rotation-dropped-from-the-goal-call",
        {"env_mut": ((
            "            self._fixture_quat,\n            self._seat_quat_local,\n"
            "            self._tip_offset_local,\n            float(self.cfg.depth_max),",
            "            self._fixture_quat,\n            self._fixture_quat,\n"
            "            self._tip_offset_local,\n            float(self.cfg.depth_max),",
        ),)},
        ("the goal pose composes the seated tool rotation",),
    ),
    (
        "sapu-queries-the-part-mesh",
        {"env_mut": ((
            "                mesh_obj_path=insertion_paths.resolve_pocket_obj_path(),\n",
            "",
        ),)},
        ("the SAPU query asks the FIXTURE mesh",),
    ),
    (
        "reward-weight-back-in-the-cfg",
        {"cfg_mut": ((
            "    gate_min_alignment = 0.99",
            "    reward_w_yaw = 2.0\n    gate_min_alignment = 0.99",
        ),)},
        ("the cfg no longer defines a proxy reward weight",),
    ),
    (
        "abort-rate-dropped-from-the-log",
        {"env_mut": ((
            "            # the success definition).\n            \"force_abort_rate\": (",
            "            # the success definition).\n            \"abort_rate\": (",
        ),)},
        ("the per-iteration log carries the abort rate",),
    ),
    (
        "abort-rate-dropped-from-the-metrics",
        {"env_mut": ((
            "            # (rl_terms_enabled = False) the abort is off and this reads 0.0.\n"
            "            \"force_abort_rate\": (",
            "            # (rl_terms_enabled = False) the abort is off and this reads 0.0.\n"
            "            \"abort_rate\": (",
        ),)},
        ("demo_metrics.json carries the abort rate",),
    ),
    (
        "interpen-max-dropped-from-the-log",
        {"env_mut": ((
            '            self.extras["log"]["interpen_max_max_mm"] = (',
            '            self.extras["log"]["interpen_worst_mm"] = (',
        ),)},
        ("the per-iteration log carries the interpenetration maximum",),
    ),
    (
        "force-p95-dropped-from-the-log",
        {"env_mut": ((
            '            self.extras["log"]["force_max_p95_n"] = _fv[',
            '            self.extras["log"]["force_worst_n"] = _fv[',
        ),)},
        ("the per-iteration log carries the force maximum",),
    ),
    (
        "interpen-tail-dropped-from-the-metrics",
        {"env_mut": ((
            '            "interpen_max_mm": self._interpen_stats(),',
            '            "interpen_mm": self._interpen_stats(),',
        ),)},
        ("demo_metrics.json carries the interpenetration tail",),
    ),
    (
        "interpen-curve-ungated",
        {"env_mut": ((
            "        if self._rl_terms_enabled and self._recent_interpen_max:",
            "        if self._recent_interpen_max:",
        ),)},
        ("the interpenetration curve is gated on rl_terms_enabled",),
    ),
    (
        "observation-blocks-swapped",
        {"env_mut": ((
            "(joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force), dim=-1",
            "(joint_pos, joint_vel_fd, ee_quat_w, tip_rel, yaw_cs, pocket_quat, force), dim=-1",
        ),)},
        ("the observation is assembled in the OBS_SLICES order",),
    ),
    (
        "c4-yaw-encoding-restored",
        {"env_mut": ((
            "            yaw_cs = insertion_math.yaw_cos_sin(x_axis)",
            "            cos4phi = x_axis\n            yaw_cs = insertion_math.yaw_cos_sin(cos4phi)",
        ),)},
        ("the C4 yaw encoding is gone from the observation",),
    ),
    (
        "wrench-read-from-a-different-body-field",
        {"env_mut": ((
            "self.robot.data.body_incoming_joint_wrench_b[:, self._force_body_idx, 0:6]",
            "self.robot.data.body_link_state_w[:, self._force_body_idx, 0:6]",
        ),)},
        ("the wrench is read through body_incoming_joint_wrench_b",),
    ),
    (
        "force-ema-bypassed",
        {"env_mut": ((
            """                    insertion_math.ema_update(
                        self._force_smooth, force_raw, float(self.cfg.ft_smoothing_factor)
                    ),""",
            "                    force_raw,",
        ),)},
        ("the force block goes through insertion_math.ema_update",),
    ),
    (
        "wrench-reset-flag-never-raised",
        {"env_mut": (("            self._wrench_fresh[env_ids] = True\n", ""),)},
        ("the reset flag _wrench_fresh is raised in _reset_idx",),
    ),
    (
        "wrench-reset-mask-dropped",
        {"env_mut": (("                keep = ~self._wrench_fresh\n",
                      "                keep = torch.ones_like(self._wrench_fresh)\n"),)},
        ("the wrench EMA skips envs _reset_idx just touched (keep = ~_wrench_fresh)",),
    ),
    (
        "wrench-once-guard-removed",
        {"env_mut": (("            if int(self.common_step_counter) != self._wrench_step:\n",
                      "            if True:\n"),)},
        ("the wrench EMA advances once per common_step_counter value",),
    ),
    (
        "wrench-read-drops-the-torque-columns",
        {"env_mut": (("            torque_raw = wrench_raw[:, 3:6]\n",
                      "            torque_raw = wrench_raw[:, 0:3]\n"),)},
        ("all six wrench columns are read (0:6), torque = columns 3:6",),
    ),
    (
        "torque-tare-dropped",
        {"env_mut": (("                torque_raw = insertion_math.gravity_tare_torque(\n",
                      "                torque_raw = torque_raw + 0.0 * insertion_math.gravity_tare(\n"),)},
        ("the torque block goes through insertion_math.gravity_tare_torque with the COM lever",),
    ),
    (
        "torque-ema-bypassed",
        {"env_mut": ((
            """                    insertion_math.ema_update(
                        self._torque_smooth, torque_raw, float(self.cfg.ft_smoothing_factor)
                    ),""",
            "                    torque_raw,",
        ),)},
        ("the torque block goes through insertion_math.ema_update",),
    ),
    (
        "torque-ema-survives-the-reset",
        {"env_mut": (("            self._torque_smooth[env_ids] = 0.0\n", ""),)},
        ("the torque EMA buffer is cleared in _reset_idx",),
    ),
    (
        "wrench-cat-without-torque",
        {"env_mut": (("                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force, torque),\n",
                      "                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force),\n"),)},
        ("the wrench observation is the force cat plus torque",),
    ),
    (
        "wrench-cat-torque-before-force",
        {"env_mut": (("                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force, torque),\n",
                      "                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, torque, force),\n"),)},
        ("the wrench observation is the force cat plus torque",),
    ),
    (
        "prevact-cat-without-prev-action",
        {"env_mut": (("                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force, prev_action),\n",
                      "                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force),\n"),)},
        ("the force_prevact observation is the force cat plus prev_action",),
    ),
    (
        "prevact-reads-self-actions",
        {"env_mut": (("            prev_action = self._ema_actions\n",
                      "            prev_action = self.actions\n"),)},
        ("prev_action is bound from self._ema_actions, the applied (smoothed) action",),
    ),
    (
        "prevact-reads-the-raw-draw",
        {"env_mut": (("            prev_action = self._ema_actions\n",
                      "            prev_action = self._prev_actions\n"),)},
        ("prev_action is bound from self._ema_actions, the applied (smoothed) action",),
    ),
    (
        "prevact-branch-removed",
        {"env_mut": (("        elif self._obs_mode == \"force_prevact\":\n", "        elif False:\n"),
                     ("            prev_action = self._ema_actions\n", "            pass\n"),
                     ("            obs = torch.cat(\n"
                      "                (joint_pos, joint_vel_fd, tip_rel, ee_quat_w, yaw_cs, pocket_quat, force, prev_action),\n"
                      "                dim=-1,\n"
                      "            )\n", ""))},
        ("the force_prevact observation is the force cat plus prev_action",
         "the observation has exactly three cats, one per mode",
         "prev_action is bound from self._ema_actions, the applied (smoothed) action"),
    ),
    (
        "prev-action-slice-overlaps-the-table",
        {"math_mut": (('        out["prev_action"] = (OBS_DIM, OBS_DIM + PREV_ACTION_WIDTH)\n',
                       '        out["prev_action"] = (OBS_DIM - PREV_ACTION_WIDTH, OBS_DIM)\n'),)},
        ("the force_prevact layout is the force layout plus prev_action appended LAST",),
    ),
    (
        "obs-layout-resolved-after-super",
        {"env_mut": (
            ("        self._obs_mode = resolve_obs_layout(cfg)\n", ""),
            ("        super().__init__(cfg, render_mode, **kwargs)\n",
             "        super().__init__(cfg, render_mode, **kwargs)\n"
             "        self._obs_mode = resolve_obs_layout(cfg)\n"),
        )},
        ("the observation layout is resolved BEFORE the base class reads the cfg",),
    ),
    (
        "obs-layout-width-typed",
        {"cfg_mut": (("    width = insertion_math.obs_dim(mode)\n", "    width = 28\n"),)},
        ("resolve_obs_layout writes observation_space from insertion_math.obs_dim(mode)",),
    ),
    (
        "torque-slice-prepended-in-the-table",
        {"math_mut": (('        out["torque"] = (OBS_DIM, OBS_DIM + TORQUE_WIDTH)\n',
                       '        out["torque"] = (OBS_DIM - TORQUE_WIDTH, OBS_DIM)\n'),)},
        ("the wrench layout is the force layout plus torque appended LAST",),
    ),
    (
        "noise-masks-called-without-the-mode",
        {"obs_noise_mut": (("            mode=str(noise_model_cfg.obs_mode),\n", ""),)},
        ("noise_masks receives torque_std_nm and mode from the noise cfg, by keyword",),
    ),
    (
        "bias-buffer-sized-from-obs-dim-again",
        {"obs_noise_mut": (("        _width = insertion_math.obs_dim(str(noise_model_cfg.obs_mode))\n",
                            "        _width = insertion_math.OBS_DIM\n"),)},
        ("the bias buffer and its component count are sized up front",),
    ),
    (
        "torque-sigma-escapes-the-negative-sweep",
        {"cfg_mut": (('                  "torque_obs_noise_std_nm", "grasp_obs_offset_x_m",\n',
                      '                  "grasp_obs_offset_x_m",\n'),)},
        ("the torque sigma is in the cfg's negative-scatter sweep",),
    ),
    (
        "torque-sigma-unlabelled",
        {"cfg_mut": (('    "torque_obs_noise_std_nm": (\n', '    "torque_obs_noise_std_nm_unlabelled": (\n'),)},
        ("the torque sigma is a labelled placeholder",
         "the cfg gives every placeholder a real value"),
    ),
    (
        "force-ema-survives-the-reset",
        {"env_mut": (("            self._force_smooth[env_ids] = 0.0" + chr(10), ""),)},
        ("the force EMA buffer is cleared in _reset_idx",),
    ),
    (
        "pose-math-inlined-again",
        {"env_mut": ((
            "            tip_rel, _depth, x_axis = insertion_math.part_tip_pose(",
            "            tip_rel, _depth, x_axis = self._inline_pose(",
        ),)},
        (
            "the observation comes from insertion_math, not from local pose math",
            "tip_rel and the yaw come from the BELIEVED pocket frame",
        ),
    ),
    (
        "force-block-no-longer-last",
        {"math_mut": ((
            '    "force": (25, 28),        # three force components, EMA-smoothed, D-114\n',
            "",
        ),)},
        ("the force block is the LAST block in the table",
         "the observation is assembled in the OBS_SLICES order",
         "the wrench observation is the force cat plus torque",
         "the force_prevact observation is the force cat plus prev_action"),
    ),
    (
        "placeholder-warning-removed",
        {"env_mut": ((
            """        if RL_PLACEHOLDERS:
            print("*** PLACEHOLDER VALUES IN USE -- every number this run reports "
                  "is provisional ***")
            for name, mark in RL_PLACEHOLDERS.items():
                print(f"    {name} = {getattr(self.cfg, name)}  {mark}")""",
            "        pass",
        ),)},
        ("the startup report reads RL_PLACEHOLDERS",
         "the startup report shouts that the numbers are provisional"),
    ),
    (
        "success-predicate-loses-the-lateral-gate",
        {"env_mut": (("            in_pocket,\n", ""),)},
        ("the success predicate is handed the lateral gate",),
    ),
    (
        "engaged-predicate-loses-the-lateral-gate",
        {"env_mut": ((
            "            depth, float(self.cfg.engaged_depth_m), in_pocket",
            "            depth, float(self.cfg.engaged_depth_m)",
        ),)},
        ("the engaged predicate is handed the lateral gate",),
    ),
    (
        "lateral-gate-inlined-in-the-env",
        {"env_mut": (("insertion_math.in_pocket_cross_section(",
                      "self._inline_cross_section("),)},
        ("the lateral gate is insertion_math's, not a second inline copy",
         "the gate reads the pocket walls, not a new tolerance"),
    ),
    (
        "pocket-gate-floor-moved-off-the-underside",
        {"env_mut": (("            cross_section, depth, -insertion_tasks_cfg.POCKET_ASSET_BOTTOM_Z",
                      "            cross_section, depth, -insertion_tasks_cfg.POCKET_FLOOR_Z"),)},
        ("the pocket gate has a floor at the fixture underside, not a new tolerance",),
    ),
    (
        "below-fixture-rate-dropped-from-the-log",
        {"env_mut": (('            "below_fixture_rate": (',
                      '            "below_floor_share": ('),)},
        ("the per-iteration log carries the below-fixture rate",),
    ),
    (
        "below-fixture-rate-dropped-from-the-metrics",
        {"env_mut": (('            "below_fixture_rate": {',
                      '            "below_floor_share": {'),)},
        ("demo_metrics.json carries the below-fixture rate",),
    ),
    (
        "unclean-counter-reads-the-wrong-threshold",
        {"env_mut": (("            interpen_max, float(self.cfg.interpen_thresh)\n        ).to(torch.int32)",
                      "            interpen_max, float(self.cfg.depth_min)\n        ).to(torch.int32)"),)},
        ("the unclean-step counter reads the SAPU rule on this step's interpenetration",),
    ),
    (
        "unclean-counter-never-zeroed",
        {"env_mut": (("            self._interpen_unclean_steps[env_ids] = 0\n", ""),)},
        ("the unclean-step counter is zeroed per episode",),
    ),
    (
        "raw-force-max-never-zeroed",
        {"env_mut": (("            self._max_force_raw_norm[env_ids] = 0.0\n", ""),)},
        ("the raw force maximum is zeroed per episode",),
    ),
    (
        "episode-row-not-written",
        {"env_mut": (("        self._append_episode_rows(idx, successes, aborts)\n", ""),)},
        ("the episode row is written from the harvest",),
    ),
    (
        "episode-column-dropped",
        {"env_mut": (('    "ip_max_mm", "ip_steps_ge_t1", "force_max_filtered_n", "force_max_raw_n",\n',
                      '    "ip_max_mm", "force_max_filtered_n", "force_max_raw_n",\n'),)},
        ("episodes.csv carries exactly the agreed columns",),
    ),
    (
        "rollout-length-not-handed-over",
        {"train_mut": (("    env_cfg.rollout_steps_per_iteration = agent_cfg.num_steps_per_env\n", ""),)},
        ("train.py hands the rollout length to the env",),
    ),
    (
        "depth-invariant-missing-from-metrics",
        {"env_mut": (('            "success_depth_invariant_violations": {',
                      '            "unused_invariant_key": {'),)},
        ("demo_metrics.json carries the depth invariant",),
    ),
    (
        "placeholders-missing-from-metrics",
        {"env_mut": (('            "rl_placeholders": {', '            "unused_key": {'),)},
        ("demo_metrics.json carries the placeholders",),
    ),
    (
        "ladder-state-missing-from-metrics",
        {"env_mut": (('            "curriculum_enabled": bool(self.cfg.curriculum_enabled),',
                      '            "unused_ladder_key": bool(self.cfg.curriculum_enabled),'),)},
        ("demo_metrics.json carries the ladder state",),
    ),
    (
        "placeholder-unset-in-the-cfg",
        {"cfg_mut": (("    force_abort_f_max_n: float = 30.0",
                      "    force_abort_f_max_n: float | None = None"),)},
        (
            "the cfg gives every placeholder a real value",
            "the force sigma is the UR5e datasheet's 3.5 N and stays under the abort force",
        ),
    ),
    (
        "proxy-table-emptied",
        {"cfg_mut": (("PROXY_TASK_VALUES: dict = {",
                      "PROXY_TASK_VALUES: dict = {}\nUNUSED_PROXY_TABLE: dict = {"),)},
        ("the cfg carries a proxy-value table",
         "every proxy metrics key points at an entry that explains it",
         "the progress weight is labelled [proxy]"),
    ),
    (
        "proxy-metrics-key-points-nowhere",
        {"cfg_mut": (('    "peg_side_m": "cfg.held_asset.side",',
                      '    "peg_side_m": "cfg.held_asset.sidee",'),)},
        ("every proxy metrics key points at an entry that explains it",),
    ),
    (
        "proxy-metrics-key-not-in-the-file",
        {"cfg_mut": (('    "side_clearance_mm": "tasks.SIDE_CLEARANCE",',
                      '    "side_clearance_xx": "tasks.SIDE_CLEARANCE",'),)},
        ("every proxy metrics key is really written to demo_metrics.json",),
    ),
    (
        "a-proxy-mark-restates-its-number",
        {"cfg_mut": (("        \"[proxy] tool-axis alignment floor, calibrated on the square peg. \"",
                      "        \"[proxy] the 0.99 alignment floor, calibrated on the square peg. \""),)},
        ("no proxy mark restates its own number",),
    ),
    (
        "the-report-stops-resolving-the-proxy-table",
        {"env_mut": (("        _proxy = resolve_proxy_task_values(self.cfg)",
                      "        _proxy = []"),)},
        ("the startup report resolves PROXY_TASK_VALUES",),
    ),
    (
        "the-proxy-warning-is-removed",
        {"env_mut": ((
            """            print("*** PROXY TASK VALUES IN USE -- the physics is the real part in the real "
                  "pocket, but these SCORING numbers still describe the square peg ***")""",
            "            pass",
        ),)},
        ("the startup report shouts that the scoring is proxy",),
    ),
    (
        "proxy-values-missing-from-metrics",
        {"env_mut": (('            "proxy_task_values": {',
                      '            "unused_key": {'),)},
        ("demo_metrics.json carries the proxy values",),
    ),
    (
        "metrics-stops-naming-its-own-proxy-keys",
        {"env_mut": (('            "proxy_metrics_keys": PROXY_METRICS_KEYS,',
                      '            "proxy_metrics_keys_unused": None,'),)},
        ("demo_metrics.json says which of its own keys are proxy",),
    ),
    (
        "the-proxy-clearance-comes-back-to-the-report",
        {"env_mut": ((
            'f"(real limit is half the across-play, {insertion_tasks_cfg.PLAY_X/2.0:.6f} m; "',
            'f"(real limit is half the across-play, {insertion_tasks_cfg.SIDE_CLEARANCE:.6f} m; "',
        ),)},
        ("the report does not print the proxy clearance as a task fact",),
    ),
    (
        "the-proxy-yaw-window-comes-back-to-the-report",
        {"env_mut": ((
            'f"~uniform in +-{math.degrees(cfg.reset_yaw_noise):.2f} after)")',
            'f"~uniform in +-{math.degrees(insertion_tasks_cfg.YAW_WINDOW_RAD):.2f} after)")',
        ),)},
        ("the report does not print the proxy yaw window as a task fact",),
    ),
    (
        "the-measured-play-is-half-reported",
        {"env_mut": ((
            'f"Y {wc.PLAY_Y*1000:.4f} mm along the part")',
            'f"Y (not reported) mm along the part")',
        ),)},
        ("the report prints the MEASURED play instead",),
    ),
    (
        "the-measured-play-never-reaches-the-metrics-file",
        {"env_mut": (('            "play_x_mm": insertion_tasks_cfg.PLAY_X * 1000.0,',
                      '            "play_x_unused": insertion_tasks_cfg.PLAY_X * 1000.0,'),)},
        ("demo_metrics.json carries the measured play",),
    ),
    (
        "the-report-title-names-the-proxy-branch-again",
        {"env_mut": ((
            'print("INSERTION STARTUP REPORT -- REAL geometry and physics, PROXY scoring (UNVERIFIED)")',
            'print("INSERTION STARTUP REPORT (square-peg pivot, branch square-peg-insertion, UNVERIFIED)")',
        ),)},
        ("the report title does not name the proxy branch",),
    ),
    (
        "train-accepts-a-measurement-env",
        {"train_mut": ((
            '    if not getattr(env_cfg, "rl_terms_enabled", True):',
            "    if False:",
        ),)},
        ("train.py refuses a measurement env",),
    ),
    (
        "train-tags-a-hand-written-autodr",
        {"train_mut": ((
            "        parts.append(str(env_cfg.dr_mode))\n",
            '        parts.append("autodr")\n',
        ),)},
        ("the run tag names the DR mode, and only when it is not off",),
    ),
    (
        "train-tags-dr-mode-even-when-off",
        {"train_mut": ((
            '    if hasattr(env_cfg, "dr_mode") and str(env_cfg.dr_mode) != "off":\n',
            '    if hasattr(env_cfg, "dr_mode"):\n',
        ),)},
        ("the run tag names the DR mode, and only when it is not off",),
    ),
    (
        "dr-tag-appended-last",
        {"train_mut": (
            ('    if hasattr(env_cfg, "dr_mode") and str(env_cfg.dr_mode) != "off":\n'
             "        parts.append(str(env_cfg.dr_mode))\n", ""),
            ("    if agent_cfg.run_name:\n",
             '    if hasattr(env_cfg, "dr_mode") and str(env_cfg.dr_mode) != "off":\n'
             "        parts.append(str(env_cfg.dr_mode))\n"
             "    if agent_cfg.run_name:\n"),
        )},
        ("the DR-mode tag is appended before the static-noise tags",),
    ),
    (
        "stop-when-dr-max-defaults-to-the-bar",
        {"train_mut": ((
            '    "--stop-when-dr-max",\n    type=float,\n    default=None,\n',
            '    "--stop-when-dr-max",\n    type=float,\n    default=0.8,\n',
        ),)},
        ("--stop-when-dr-max is a float the run must pass, never defaulted here",),
    ),
    (
        "both-stop-criteria-allowed-together",
        {"train_mut": ((
            "    if args_cli.stop_when_dr_max is not None and args_cli.stop_at_success_rate is not None:\n",
            "    if False:\n",
        ),)},
        ("both stop-flag refusals run before the env is built",),
    ),
    (
        "dr-stop-demands-the-wrong-mode",
        {"train_mut": ((
            '        str(getattr(env_cfg, "dr_mode", "off")) != "autodr" and not _floor_on\n',
            '        str(getattr(env_cfg, "dr_mode", "off")) == "autodr" and not _floor_on\n',
        ),)},
        ("both stop-flag refusals run before the env is built",),
    ),
    (
        "sidecar-written-before-the-checkpoint",
        {"train_mut": (
            ("        out = save_fn(path, *args, **kwargs)\n"
             "        side = _autodr_sidecar_path(path)\n",
             "        side = _autodr_sidecar_path(path)\n"),
            ("        return out\n\n    return _save\n",
             "        out = save_fn(path, *args, **kwargs)\n"
             "        return out\n\n    return _save\n"),
        )},
        ("the sidecar is written after the checkpoint, by the wrapped save",),
    ),
    (
        "runner-save-not-wrapped",
        {"train_mut": ((
            "        runner.save = _save_with_autodr(\n"
            "            runner.save, _dr, agent_cfg.seed, os.path.basename(log_dir)\n"
            "        )\n",
            "        pass\n",
        ),)},
        ("runner.save is wrapped so every checkpoint gets its AutoDR state",
         "the save wrapper is installed under the provider guard and no other"),
    ),
    (
        "sidecar-name-loses-the-iteration",
        {"train_mut": ((
            '        return os.path.join(head, "autodr_" + stem[len("model_"):] + ".json")\n',
            '        return os.path.join(head, "autodr_latest.json")\n',
        ),)},
        ("the sidecar carries the checkpoint's own iteration number",),
    ),
    (
        "resume-without-the-autodr-state-runs-anyway",
        {"train_mut": ((
            "            if not os.path.isfile(_side):\n",
            "            if False:\n",
        ),)},
        ("a resume without the AutoDR state is refused, and the state loads after the policy",),
    ),
    (
        "resume-reads-the-state-and-drops-it",
        {"train_mut": ((
            '            _dr.load_state_dict(_doc["autodr"])\n',
            '            _doc["autodr"]\n',
        ),)},
        ("a resume without the AutoDR state is refused, and the state loads after the policy",
         "the sidecar carries the run identity and the resume checks it"),
    ),
    (
        "dr-stop-drops-the-at-max-condition",
        {"train_mut": ((
            '        return ("fire" if (at_max and enough and rate >= target) else "wait"), line\n',
            '        return ("fire" if (enough and rate >= target) else "wait"), line\n',
        ),)},
        ("the AutoDR stop tests all three conditions of plan section 4",),
    ),
    (
        "dr-stop-reads-the-recent-window",
        {"train_mut": ((
            '        fresh = getattr(base, "_fresh_successes", None)\n',
            '        fresh = getattr(base, "_recent_successes", None)\n',
        ),)},
        ("the AutoDR stop reads the fresh window, never the recent one",),
    ),
    (
        "under-full-window-prints-a-rate",
        {"train_mut": ((
            '        shown = f"{rate:.4f}" if enough else "--"\n',
            '        shown = f"{rate:.4f}"\n',
        ),)},
        ("an under-full fresh window prints no rate at all",),
    ),
    (
        "rate-stop-loses-the-quarter-window",
        {"train_mut": ((
            "        filled = len(recent) >= (recent.maxlen // 4)\n",
            "        filled = True\n",
        ),)},
        ("D-037's stop rule still reads every episode and the quarter-window guard",),
    ),
    (
        "loop-fires-before-it-abandons",
        {"train_mut": (
            (
            "            if state == \"fire\":\n"
            "                stopped_early = True\n"
            '                print(f"[stop-criterion] target reached at iteration {trained}; stopping.")\n'
            "                break\n"
            , ""),
            ('            if state == "abandon":\n',
            "            if state == \"fire\":\n"
            "                stopped_early = True\n"
            '                print(f"[stop-criterion] target reached at iteration {trained}; stopping.")\n'
            "                break\n"
             '            if state == "abandon":\n'),
        )},
        ("the block loop abandons before it fires",),
    ),
    (
        "ladder-and-autodr-allowed-together",
        {"cfg_mut": ((
            '    if ladder_on and str(cfg.dr_mode) != "off":\n',
            "    if False:\n",
        ),)},
        ("the ladder and AutoDR are refused together, and dr_mode is in the absent-field sweep",),
    ),
    (
        "dr-mode-not-in-the-absent-field-sweep",
        {"cfg_mut": ((
            '        "w_tilt", "shaping_gamma", "kernel_a_tilt",\n'
            '        "dr_mode",\n',
            '        "w_tilt", "shaping_gamma", "kernel_a_tilt",\n',
        ),)},
        ("the ladder and AutoDR are refused together, and dr_mode is in the absent-field sweep",),
    ),
    (
        "table-mode-floor-refusal-removed",
        {"cfg_mut": ((
            "    if high is None or float(cfg.start_floor_m) != float(high):\n",
            "    if False:\n",
        ),)},
        ("table mode with a start floor, or without a start height, is refused",),
    ),
    (
        "table-mode-missing-path-allowed",
        {"cfg_mut": ((
            "    if path is None or not str(path).strip():\n",
            "    if False:\n",
        ),)},
        ("table mode without a table path is refused",),
    ),
    (
        "table-mode-any-width-allowed",
        {"cfg_mut": ((
            "    if str(cfg.dr_table_width) not in DR_TABLE_WIDTHS:\n",
            "    if False:\n",
        ),)},
        ("table mode with an unknown width is refused",),
    ),
    (
        "table-path-without-table-mode-allowed",
        {"cfg_mut": ((
            "        if path is not None:\n            raise RlConfigIncomplete(\n"
            '                f"dr_table_path is',
            "        if False:\n            raise RlConfigIncomplete(\n"
            '                f"dr_table_path is',
        ),)},
        ("a table path that no mode reads is refused",),
    ),
    (
        "validate-table-mode-never-called",
        {"cfg_mut": ((
            "    validate_table_mode(cfg)\n    if not ladder_on:\n",
            "    if not ladder_on:\n",
        ),)},
        ("validate_rl_config calls validate_table_mode unconditionally",),
    ),
    (
        "table-width-default-nominal",
        {"cfg_mut": ((
            '    dr_table_width: str = "target"\n',
            '    dr_table_width: str = "nominal"\n',
        ),)},
        ("the cfg defaults are no table and the target width",),
    ),
    (
        "ood-switch-default-on",
        {"cfg_mut": ((
            "    dr_table_allow_outside: bool = False\n",
            "    dr_table_allow_outside: bool = True\n",
        ),)},
        ("the cfg defaults are no table and the target width",),
    ),
    (
        "ood-switch-outside-table-mode-allowed",
        {"cfg_mut": ((
            "        if bool(cfg.dr_table_allow_outside):\n",
            "        if False:\n",
        ),)},
        ("the OOD switch passes in table mode and is refused outside it",),
    ),
    (
        "env-drops-the-ood-switch",
        {"env_mut": ((
            "cfg.dr_table_path, allow_outside=bool(cfg.dr_table_allow_outside)\n",
            "cfg.dr_table_path\n",
        ),)},
        ("table mode validates its own cfg before it reads the table",),
    ),
    (
        "env-asks-for-the-next-row",
        {"env_mut": ((
            "autodr.table_rows(self._eval_table_unit, idx)",
            "autodr.table_rows(self._eval_table_unit, (idx + 1) % self.num_envs)",
        ),)},
        ("the reset draw copies row i for env i through autodr.table_rows, else torch.rand",),
    ),
    (
        "table-may-be-longer-than-the-env-count",
        {"env_mut": ((
            "            if self._eval_table.n_rows != int(self.num_envs):\n",
            "            if self._eval_table.n_rows < int(self.num_envs):\n",
        ),)},
        ("the table must have EXACTLY num_envs rows",),
    ),
    (
        "table-mode-skips-its-own-validation",
        {"env_mut": ((
            "            validate_table_mode(cfg)\n            self._eval_table = ",
            "            self._eval_table = ",
        ),)},
        ("table mode validates its own cfg before it reads the table",),
    ),
    (
        "table-provider-at-centre",
        {"env_mut": ((
            "autodr.FixedWidth.at_max(_table_dims)",
            "autodr.FixedWidth.at_centre(_table_dims)",
        ),)},
        ("the table provider is FixedWidth.at_max, and 'nominal' keeps the start height alone",),
    ),
    (
        "nominal-width-keeps-all-five",
        {"env_mut": ((
            "                _table_dims = tuple(d for d in _table_dims if d.name == \"start_height\")\n",
            "                _table_dims = tuple(d for d in _table_dims)\n",
        ),)},
        ("the table provider is FixedWidth.at_max, and 'nominal' keeps the start height alone",),
    ),
    (
        "table-centre-differs-from-autodr",
        {"env_mut": ((
            "            _table_dims = autodr.bind_centres(\n                {\n"
            "                    \"lat_r\": 0.0,\n",
            "            _table_dims = autodr.bind_centres(\n                {\n"
            "                    \"lat_r\": 0.001,\n",
        ),)},
        ("the table and Fixed-DR providers open around the SAME five centres as AutoDR",
         "the AutoDR centres are bound to the five named sources"),
    ),
    (
        "static-refusal-back-to-autodr-only",
        {"env_mut": ((
            '        if str(cfg.dr_mode) in ("autodr", "fixed_dr", "table"):\n',
            '        if str(cfg.dr_mode) == "autodr":\n',
        ),)},
        ("the static-field refusal covers 'autodr', 'fixed_dr' AND 'table'",),
    ),
    (
        "static-refusal-drops-fixed-dr",
        {"env_mut": ((
            '        if str(cfg.dr_mode) in ("autodr", "fixed_dr", "table"):\n',
            '        if str(cfg.dr_mode) in ("autodr", "table"):\n',
        ),)},
        ("the static-field refusal covers 'autodr', 'fixed_dr' AND 'table'",),
    ),
    (
        "fixed-dr-arm-builds-autodr",
        {"env_mut": ((
            "            self._dr = autodr.FixedDR(\n",
            "            self._dr = autodr.AutoDR(\n",
        ),)},
        ("the fixed_dr arm builds autodr.FixedDR on the start floor",
         "both AutoDR providers take the floor, and the No-DR one carries only start_height",
         "both AutoDR providers take the cfg stall bar"),
    ),
    (
        "fixed-dr-arm-drops-the-stall-bar",
        {"env_mut": ((
            "                floor=_floor_spec,\n                stall_buffers=int(cfg.autodr_stall_buffers),\n"
            "            )\n        elif str(cfg.dr_mode) != \"off\":\n",
            "                floor=_floor_spec,\n"
            "            )\n        elif str(cfg.dr_mode) != \"off\":\n",
        ),)},
        ("the fixed_dr arm builds autodr.FixedDR on the start floor",),
    ),
    (
        "fixed-dr-arm-drops-the-floor",
        {"env_mut": ((
            "                floor=_floor_spec,\n                stall_buffers=int(cfg.autodr_stall_buffers),\n"
            "            )\n        elif str(cfg.dr_mode) != \"off\":\n",
            "                floor=None,\n                stall_buffers=int(cfg.autodr_stall_buffers),\n"
            "            )\n        elif str(cfg.dr_mode) != \"off\":\n",
        ),)},
        ("the fixed_dr arm builds autodr.FixedDR on the start floor",),
    ),
    (
        "fixed-dr-centre-differs-from-autodr",
        {"env_mut": ((
            "            self._dr = autodr.FixedDR(\n                autodr.bind_centres(\n"
            "                    {\n                        \"lat_r\": 0.0,\n",
            "            self._dr = autodr.FixedDR(\n                autodr.bind_centres(\n"
            "                    {\n                        \"lat_r\": 0.001,\n",
        ),)},
        ("the table and Fixed-DR providers open around the SAME five centres as AutoDR",
         "the AutoDR centres are bound to the five named sources"),
    ),
    (
        "episode-log-reads-phase-without-default",
        {"env_mut": ((
            'phase = str(getattr(self._dr, "phase", "fixed"))',
            "phase = str(self._dr.phase)",
        ),)},
        ("the episode log reads the phase with a default, so FixedWidth cannot crash it",),
    ),
    (
        "eval-table-line-loses-the-hash",
        {"env_mut": ((
            'f"{self._eval_table.n_rows}  sha256 {self._eval_table.sha256}  "',
            'f"{self._eval_table.n_rows}  "',
        ),)},
        ("the startup report prints the table's sha256 on the [eval-table] line (G1)",),
    ),
    (
        "autodr-paragraph-printed-for-every-provider",
        {"env_mut": ((
            "            if isinstance(self._dr, autodr.AutoDR):\n",
            "            if True:\n",
        ),)},
        ("the AutoDR-only report paragraph is guarded by the provider's type",),
    ),
    (
        "applied-csv-swaps-two-columns",
        {"env_mut": ((
            '    "row_id", "lat_x_m", "lat_y_m", "start_height_m", "yaw_rad",\n',
            '    "row_id", "lat_y_m", "lat_x_m", "start_height_m", "yaw_rad",\n',
        ),)},
        ("applied.csv carries exactly the agreed columns",),
    ),
    (
        "applied-csv-row-id-off-by-one",
        {"env_mut": ((
            "                torch.arange(n, device=self.device, dtype=torch.float64),\n",
            "                torch.arange(n, device=self.device, dtype=torch.float64) + 1.0,\n",
        ),)},
        ("every applied.csv column reads its agreed source, row id = env id, in column order",),
    ),
    (
        "applied-csv-friction-echoes-the-command",
        {"env_mut": ((
            "                _fric[0].min(dim=1).values,\n",
            "                self._friction_applied.to(torch.float64),\n",
        ),)},
        ("every applied.csv column reads its agreed source, row id = env id, in column order",),
    ),
    (
        "applied-csv-reads-the-robot-twice",
        {"env_mut": ((
            "            for _asset in (self.robot, self._fixture)\n        ]\n        _pocket_d",
            "            for _asset in (self.robot, self.robot)\n        ]\n        _pocket_d",
        ),)},
        ("the four friction read-backs come out of PhysX, robot first, fixture second",),
    ),
    (
        "applied-csv-drops-the-pocket-yaw-error",
        {"env_mut": ((
            "                self._pocket_obs_yaw.to(torch.float64),\n",
            "",
        ),)},
        ("every applied.csv column reads its agreed source, row id = env id, in column order",
         "the pocket-yaw belief error reaches the OBSERVATION and no paying path"),
    ),
    (
        "applied-csv-pocket-yaw-reads-the-grasp-buffer",
        {"env_mut": ((
            "                self._pocket_obs_yaw.to(torch.float64),\n",
            "                self._grasp_obs_off[:, 0].to(torch.float64),\n",
        ),)},
        ("every applied.csv column reads its agreed source, row id = env id, in column order",
         "the pocket-yaw belief error reaches the OBSERVATION and no paying path"),
    ),
    (
        "applied-csv-written-before-any-physics-step",
        {"env_mut": ((
            "        if int(self.common_step_counter) < 1:\n",
            "        if int(self.common_step_counter) < 0:\n",
        ),)},
        ("applied.csv is written in table mode only, once, and only after a physics step",),
    ),
    (
        "applied-csv-rewritten-every-step",
        {"env_mut": ((
            "        self._applied_written = True\n",
            "",
        ),)},
        ("the applied.csv writer raises its own written flag",),
    ),
    (
        "applied-csv-skipped-without-a-run-folder",
        {"env_mut": ((
            "            raise RuntimeError(\n                \"dr_mode='table' without cfg.log_dir",
            "            return\n            raise RuntimeError(\n                \"dr_mode='table' without cfg.log_dir",
        ),)},
        ("applied.csv is refused without a run folder, not skipped",),
    ),
    (
        "manifest-loses-a-field",
        {"play_mut": ((
            "            tag=args_cli.eval_tag,\n",
            "",
        ),)},
        ("play.py hands build_manifest EXACTLY the manifest's field list",),
    ),
    (
        "eval-run-leaves-dr-mode-off",
        {"play_mut": ((
            '        env_cfg.dr_mode = "table"\n',
            '        env_cfg.dr_mode = "off"\n',
        ),)},
        ("the evaluation run fixes dr_mode, the table path and the width on the cfg",),
    ),
    (
        "eval-run-waits-for-the-shutdown",
        {"play_mut": ((
            "        sys.stderr.flush()\n        os._exit(0)\n",
            "        sys.stderr.flush()\n",
        ),)},
        ("the evaluation ends the process right after the manifest",),
    ),
    (
        "eval-run-keeps-the-cli-env-count",
        {"play_mut": ((
            "        env_cfg.scene.num_envs = _table_rows\n",
            "",
        ),)},
        ("the evaluation run takes num_envs from the table's row count",),
    ),
    (
        "eval-run-accepts-a-missing-seed",
        {"play_mut": ((
            "        if args_cli.seed is None:\n",
            "        if False:\n",
        ),)},
        ("an evaluation without --seed is refused (G8)",),
    ),
    (
        "eval-run-appends-to-an-old-folder",
        {"play_mut": ((
            '        if os.path.exists(os.path.join(log_dir, replay_dirname, "episodes.csv")):\n',
            "        if False:\n",
        ),)},
        ("an evaluation folder that already holds episodes.csv is refused",),
    ),
    (
        "eval-run-stops-on-the-first-env",
        {"play_mut": ((
            "                if bool(_eval_first_done.all()):\n",
            "                if bool(_eval_first_done.any()):\n",
        ),)},
        ("the evaluation stops once ALL envs have finished their first episode",),
    ),
    (
        "normaliser-hash-before-is-dropped",
        {"play_mut": ((
            "    _norm_before = _normalizer_sha256(runner) if _em is not None else None\n",
            "    _norm_before = None\n",
        ),)},
        ("the normaliser is hashed before the run and again for the manifest (G5)",),
    ),
    (
        "memory-switch-ignored",
        {"play_mut": ((
            "_mem_reset = torch.ones_like(dones) if args_cli.memory_reset_every_step else dones",
            "_mem_reset = dones",
        ),)},
        ("play.py resets the recurrent memory after every step: done envs, or all under the switch",),
    ),
    (
        "memory-never-reset-under-rsl-rl-3",
        {"play_mut": (("                policy_nn.reset(_mem_reset)\n", "                pass\n"),)},
        ("play.py resets the recurrent memory after every step: done envs, or all under the switch",),
    ),
    (
        "memory-switch-on-by-default",
        {"play_mut": (('    default=False,\n    help="D-195 test condition',
                       '    default=True,\n    help="D-195 test condition'),)},
        ("--memory-reset-every-step exists and is off by default",),
    ),
    (
        "manifest-hides-the-memory-reset",
        {"play_mut": (('            + ("recurrent memory cleared EVERY step (--memory-reset-every-step)"\n'
                       '               if args_cli.memory_reset_every_step else "recurrent memory reset on done"),',
                       '            + "recurrent memory reset on done",'),)},
        ("the manifest's action_mode says which memory reset ran",),
    ),
    (
        "rlg-manifest-loses-a-field",
        {"rlg_play_mut": (("            tag=args_cli.eval_tag,\n", ""),)},
        ("rl_games play.py hands build_manifest EXACTLY the manifest's field list",),
    ),
    (
        "rlg-eval-run-leaves-dr-mode-off",
        {"rlg_play_mut": ((
            '        env_cfg.scene.num_envs = _table_rows\n        env_cfg.dr_mode = "table"\n',
            '        env_cfg.scene.num_envs = _table_rows\n        env_cfg.dr_mode = "off"\n',
        ),)},
        ("rl_games play.py: the evaluation run fixes dr_mode, the table path, the width and the OOD switch",),
    ),
    (
        "rlg-play-never-sets-the-ood-switch",
        {"rlg_play_mut": (("        env_cfg.dr_table_allow_outside = bool(args_cli.eval_ood)\n", ""),)},
        ("rl_games play.py: the evaluation run fixes dr_mode, the table path, the width and the OOD switch",),
    ),
    (
        "rlg-eval-run-keeps-the-cli-env-count",
        {"rlg_play_mut": (("        env_cfg.scene.num_envs = _table_rows\n", ""),)},
        ("rl_games play.py: the evaluation run takes num_envs from the table's row count",),
    ),
    (
        "rlg-eval-run-accepts-a-missing-seed",
        {"rlg_play_mut": (("        if args_cli.seed is None:\n", "        if False:\n"),)},
        ("rl_games play.py: an evaluation without a fixed seed or a named checkpoint is refused (G8)",),
    ),
    (
        "rlg-eval-run-accepts-a-random-seed",
        {"rlg_play_mut": (("        if args_cli.seed == -1:\n            _eval_refusals",
                           "        if False:\n            _eval_refusals"),)},
        ("rl_games play.py: an evaluation without a fixed seed or a named checkpoint is refused (G8)",),
    ),
    (
        "rlg-eval-run-samples-the-action",
        {"rlg_play_mut": (('"autodr_state", "video", "stochastic")', '"autodr_state", "video")'),)},
        ("rl_games play.py: the views, sampling, a step cap and a picked checkpoint are refused with --eval-table",),
    ),
    (
        "rlg-eval-flags-without-table-accepted",
        {"rlg_play_mut": ((
            '    elif args_cli.eval_tag or args_cli.eval_width != "target" or args_cli.eval_ood:\n',
            "    elif False:\n",
        ),)},
        ("rl_games play.py: --eval-width / --eval-tag / --eval-ood without --eval-table are refused",),
    ),
    (
        "rlg-eval-run-appends-to-an-old-folder",
        {"rlg_play_mut": ((
            '        if os.path.exists(os.path.join(log_dir, replay_dirname, "episodes.csv")):\n',
            "        if False:\n",
        ),)},
        ("rl_games play.py: the evaluation writes into <run>/eval_<width>_<sha8>[_<tag>] and never into an old one",),
    ),
    (
        "rlg-eval-folder-is-the-replay-folder",
        {"rlg_play_mut": ((
            "        replay_dirname = _em.eval_dirname(args_cli.eval_width, _table_sha, args_cli.eval_tag)\n",
            "",
        ),)},
        ("rl_games play.py: the evaluation writes into <run>/eval_<width>_<sha8>[_<tag>] and never into an old one",),
    ),
    (
        "rlg-eval-run-stops-on-the-first-env",
        {"rlg_play_mut": (("                if bool(_eval_first_done.all()):\n",
                           "                if bool(_eval_first_done.any()):\n"),)},
        ("rl_games play.py: the evaluation stops once ALL envs have finished their first episode",),
    ),
    (
        "rlg-normaliser-hash-before-is-dropped",
        {"rlg_play_mut": (("    _norm_before = _normalizer_sha256(agent) if _em is not None else None\n",
                           "    _norm_before = None\n"),)},
        ("rl_games play.py: the normaliser is hashed after the restore and again for the manifest (G5)",),
    ),
    (
        "rlg-normaliser-reads-the-value-normaliser",
        {"rlg_play_mut": (('"running_mean_std", None)', '"value_mean_std", None)'),)},
        ("rl_games play.py: the G5 hash reads agent.model.running_mean_std",),
    ),
    (
        "rlg-eval-run-waits-for-the-shutdown",
        {"rlg_play_mut": (("        sys.stderr.flush()\n        os._exit(0)\n", "        sys.stderr.flush()\n"),)},
        ("rl_games play.py: the evaluation ends the process right after the manifest",),
    ),
    (
        "rlg-memory-switch-ignored",
        {"rlg_play_mut": (("_mem_reset = torch.ones_like(dones) if args_cli.memory_reset_every_step else dones",
                           "_mem_reset = dones"),)},
        ("rl_games play.py resets the recurrent memory after every step: done envs, or all under the switch",),
    ),
    (
        "rlg-memory-never-reset",
        {"rlg_play_mut": (("                        s[:, _mem_reset, :] = 0.0\n", "                        pass\n"),)},
        ("rl_games play.py resets the recurrent memory after every step: done envs, or all under the switch",),
    ),
    (
        "rlg-memory-switch-on-by-default",
        {"rlg_play_mut": (('    default=False,\n    help="D-195 test condition',
                           '    default=True,\n    help="D-195 test condition'),)},
        ("rl_games play.py: --memory-reset-every-step exists and is off by default",),
    ),
    (
        "rlg-manifest-hides-the-memory-reset",
        {"rlg_play_mut": (('            + ("recurrent memory cleared EVERY step (--memory-reset-every-step)"\n'
                           '               if args_cli.memory_reset_every_step else "recurrent memory reset on done"),',
                           '            + "recurrent memory reset on done",'),)},
        ("rl_games play.py: the manifest's action_mode says which memory reset ran",),
    ),
    (
        "rlg-manifest-noise-loses-the-yaw-belief",
        {"rlg_play_mut": (('"pocket_obs_yaw_rad", "fixture_pos_noise_xy",', '"fixture_pos_noise_xy",'),)},
        ("rl_games play.py: the manifest's noise names the yaw belief and the z spread, all cfg fields",),
    ),
    (
        "wrapper-installed-only-on-resume",
        {"train_mut": ((WRAP_GUARD,
                        '    if _dr is not None and agent_cfg.resume:\n'
                        '        runner.save = _save_with_autodr(\n'),)},
        ("the save wrapper is installed under the provider guard and no other",),
    ),
    (
        "wrapper-installed-when-there-is-no-provider",
        {"train_mut": ((WRAP_GUARD,
                        '    if _dr is None:\n'
                        '        runner.save = _save_with_autodr(\n'),)},
        ("the save wrapper is installed under the provider guard and no other",),
    ),
    (
        "resume-derives-the-sidecar-from-the-new-run",
        {"train_mut": (("            _side = _autodr_sidecar_path(resume_path)\n",
                        "            _side = _autodr_sidecar_path(log_dir)\n"),)},
        ("the sidecar path is derived from the checkpoint path at both call sites",),
    ),
    (
        "sidecar-split-from-a-basename",
        {"train_mut": (("    head, name = os.path.split(model_path)\n",
                        "    head, name = os.path.split(os.path.basename(model_path))\n"),)},
        ("the sidecar path splits the checkpoint path itself, not a basename",),
    ),
    (
        "all-at-max-referenced-not-called",
        {"train_mut": (("        at_max = bool(dr.all_at_max())\n",
                        "        at_max = bool(dr.all_at_max)\n"),)},
        ("the AutoDR stop CALLS all_at_max, it does not merely mention it",),
    ),
    (
        "fresh-floor-quartered",
        {"train_mut": (('        floor = int(getattr(base, "_fresh_min", 0))\n',
                        '        floor = int(getattr(base, "_fresh_min", 0)) // 4\n'),)},
        ("the fresh-window floor is the env's whole _fresh_min",),
    ),
    (
        "fresh-floor-misspelled-into-zero",
        {"train_mut": (('        floor = int(getattr(base, "_fresh_min", 0))\n',
                        '        floor = int(getattr(base, "_fresh_minimum", 0))\n'),)},
        ("the fresh-window floor is the env's whole _fresh_min",),
    ),
    (
        "fresh-rate-taken-from-the-boundary-buffer",
        {"train_mut": (("        rate = (sum(fresh) / n_fresh) if n_fresh else 0.0\n",
                        "        rate = (sum(base._recent_boundary_successes) / n_fresh)"
                        " if n_fresh else 0.0\n"),)},
        ("the AutoDR stop's rate is the fresh buffer's own mean",),
    ),
    (
        "criterion-evaluated-before-the-block-trains",
        {"train_mut": ((
            "            runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=(trained == 0))\n"
            "            trained += chunk\n"
            "            state, line = criterion()\n",
            "            state, line = criterion()\n"
            "            runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=(trained == 0))\n"
            "            trained += chunk\n",
        ),)},
        ("the criterion is evaluated after the block trains, never before",),
    ),
    (
        "d037-rate-divided-by-the-window-capacity",
        {"train_mut": (("        rate = (sum(recent) / len(recent)) if recent else 0.0\n",
                        "        rate = (sum(recent) / recent.maxlen) if recent else 0.0\n"),)},
        ("D-037's rate divides by the window's contents, not its capacity",),
    ),
    (
        "the-not-a-test-rate-note-only-on-an-early-stop",
        {"train_mut": ((
            '            print(f"[stop-criterion] saved {final_path}")\n'
            "        print(note)\n",
            '            print(f"[stop-criterion] saved {final_path}")\n'
            "            print(note)\n",
        ),)},
        ("the not-a-test-rate note is printed on every path, not only on an early stop",),
    ),
    (
        "stop-branches-select-each-others-criterion",
        {"train_mut": (
            ("            criterion = _criterion_success_rate\n",
             "            criterion = _criterion_dr_max\n"),
            ("            criterion = _criterion_dr_max\n"
             "            note = (\n"
             '                "[stop-criterion] NOTE: dr/train_success_regular_fresh is a TRAINING rate "\n',
             "            criterion = _criterion_success_rate\n"
             "            note = (\n"
             '                "[stop-criterion] NOTE: dr/train_success_regular_fresh is a TRAINING rate "\n'),
        )},
        ("each stop branch selects its own criterion",),
    ),
    (
        "early-stop-save-bypasses-the-wrapper",
        {"train_mut": (("            runner.save(final_path)\n",
                        "            type(runner).save(runner, final_path)\n"),)},
        ("the early-stop save goes through the wrapped instance attribute",),
    ),
    (
        "early-stop-checkpoint-named-from-the-block-counter",
        {"train_mut": ((
            '            final_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")\n',
            '            final_path = os.path.join(log_dir, f"model_{trained}.pt")\n'),)},
        ("the early-stop checkpoint is named from rsl_rl's own iteration counter",),
    ),
    (
        "the-log-prints-the-unguarded-rate",
        {"train_mut": (('            f"{n_fresh}/{floor}, fresh success {shown} (bar {target:.4f}, "\n',
                        '            f"{n_fresh}/{floor}, fresh success {rate:.4f} (bar {target:.4f}, "\n'),)},
        ("the AutoDR stop prints the guarded rate, never the raw one",),
    ),
    (
        "abandon-ends-the-run-after-one-block",
        {"train_mut": ((
            "                _rest = agent_cfg.max_iterations - trained\n",
            "                _rest = 0\n"),)},
        ("an abandoned criterion trains the rest instead of ending the run",),
    ),
    (
        "abandon-branch-never-learns",
        {"train_mut": ((
            "                    runner.learn(num_learning_iterations=_rest,"
            " init_at_random_ep_len=False)\n",
            "                    pass\n"),)},
        ("an abandoned criterion trains the rest instead of ending the run",),
    ),
    (
        "the-stop-bar-is-not-range-checked",
        {"train_mut": (("        if _val is not None and not (0.0 <= float(_val) <= 1.0):\n",
                        "        if False:\n"),)},
        ("both stop bars are refused outside [0, 1]",),
    ),
    (
        "resume-accepts-another-runs-seed",
        {"train_mut": (('            if _doc.get("seed") != agent_cfg.seed:\n',
                        "            if False:\n"),)},
        ("the sidecar carries the run identity and the resume checks it",),
    ),
    (
        "distillation-loads-the-teachers-boundaries",
        {"train_mut": ((
            '        if _dr is not None and agent_cfg.algorithm.class_name == "Distillation":\n',
            "        if False:\n"),)},
        ("the sidecar carries the run identity and the resume checks it",),
    ),
    (
        "the-sidecar-envelope-drops-the-seed",
        {"train_mut": (('                    "seed": seed,\n', ""),)},
        ("the sidecar carries the run identity and the resume checks it",),
    ),
    (
        "the-cone-reference-is-pinned-upright",
        {"env_mut": (("            tgt_quat_w, float(self.cfg.osc_tilt_clamp_rad), self._fixture_quat\n",
                      "            tgt_quat_w, float(self.cfg.osc_tilt_clamp_rad),"
                      " torch.zeros_like(self._fixture_quat)\n"),)},
        ("the tilt cone is measured from this episode's pocket axis",),
    ),
    (
        "the-start-residual-is-the-batch-maximum-again",
        {"env_mut": (("            _bad = torch.linalg.norm(_goal[idx] - tip_rel[idx, 0:3], dim=-1) >= tol\n",
                      "            _bad = torch.full((idx.numel(),), err_m >= tol,"
                      " dtype=torch.bool, device=self.device)\n"),)},
        ("the start-solve residual is measured per env, not as a batch maximum",),
    ),
    (
        "only-the-boundary-flag-is-withdrawn",
        {"env_mut": (("                self._bounds_stamp[_rows] = -1\n", ""),)},
        ("an unconverged start solve withdraws BOTH the boundary flag and the stamp",),
    ),
    (
        "the-depth-seed-moves-below-the-residual",
        {"env_mut": (
            ("            self._max_depth[idx] = (-tip_rel[idx, 2]).clamp(min=0.0)\n", ""),
            ("                self._start_solve_flags_dropped += _n_bad\n",
             "                self._start_solve_flags_dropped += _n_bad\n"
             "            self._max_depth[idx] = (-tip_rel[idx, 2]).clamp(min=0.0)\n"),
        )},
        ("the withdrawal reads the pose the solve actually reached",),
    ),
    (
        "the-dropped-booking-count-never-reaches-the-log",
        {"env_mut": (('            self.extras["log"]["dr/start_solve_flags_dropped"] = float(\n',
                      '            self.extras["log"]["dr/unused_scalar"] = float(\n'),)},
        ("the dropped-booking count reaches the log and demo_metrics.json",
         "both published dropped-booking figures read the withdrawal counter itself"),
    ),
    (
        "diagnostic-agents-demand-the-open-values",
        {"agent_mut": (("    env_cfg.rl_terms_enabled = False", "    pass"),)},
        ("zero_agent builds a measurement env", "random_agent builds a measurement env"),
    ),
    (
        "the-scripted-gate-demands-the-open-values",
        {"scripted_mut": (("    env_cfg.rl_terms_enabled = False", "    pass"),)},
        ("scripted_insert builds a measurement env",),
    ),
    (
        "the-seat-probe-demands-the-open-values",
        {"seat_mut": (("    env_cfg.rl_terms_enabled = False", "    pass"),)},
        ("seat_probe builds a measurement env",),
    ),
    (
        "the-seat-probe-descends-instead-of-teleporting",
        {"seat_mut": ((
            "                robot.write_joint_state_to_sim(joint_pos_des, "
            "torch.zeros_like(joint_pos_des))\n",
            "",
        ),)},
        ("seat_probe teleports instead of descending",
         "seat_probe solves kinematically, without stepping the sim"),
    ),
    (
        "the-seat-probe-runs-under-inference-mode",
        {"seat_mut": ((
            "        with torch.no_grad():\n"
            "            tip_rel = _slice(obs, \"tip_rel\")\n",
            "        with torch.inference_mode():\n"
            "            tip_rel = _slice(obs, \"tip_rel\")\n",
        ),)},
        ("seat_probe never runs under inference_mode",),
    ),
    (
        "the-seat-probe-steps-the-sim-inside-the-solve",
        {"seat_mut": ((
            "                unwrapped._joint_targets[:] = joint_pos_des\n",
            "                unwrapped._joint_targets[:] = joint_pos_des\n"
            "                obs_dict, _, _, _, _ = env.step(zero_action)\n",
        ),)},
        ("seat_probe solves kinematically, without stepping the sim",),
    ),
    (
        "the-seat-probe-teleports-without-holding",
        {"seat_mut": ((
            "                robot.set_joint_position_target(joint_pos_des)\n",
            "",
        ),)},
        ("seat_probe holds the teleported pose with a drive target",),
    ),
    (
        "the-abort-override-lands-after-the-env-is-built",
        {"scripted_mut": (
            ("    if args_cli.f_abort is not None:\n"
             "        env_cfg.force_abort_f_max_n = float(args_cli.f_abort)\n", ""),
            ("    env = gym.make(args_cli.task, cfg=env_cfg)\n",
             "    env = gym.make(args_cli.task, cfg=env_cfg)\n"
             "    if args_cli.f_abort is not None:\n"
             "        env_cfg.force_abort_f_max_n = float(args_cli.f_abort)\n"),
        )},
        ("the abort override reaches the cfg before the env is built",),
    ),
    (
        "the-tilt-measurement-is-dropped",
        {"scripted_mut": (("            tilt = sp.axis_tilt_angle(ee_quat, pocket_quat)",
                           "            tilt = force_norm * 0.0"),)},
        ("the scripted gate measures the axis tilt",),
    ),
    (
        "the-tilt-never-reaches-the-metrics-file",
        {"scripted_mut": (('        "peak_axis_tilt_rad": _percentiles(tilts),\n', ""),)},
        ("the axis tilt reaches the metrics file",),
    ),
    (
        "the-abort-flag-is-removed-again",
        {"scripted_mut": (('    "--f-abort",', '    "--not-a-flag",'),)},
        ("the scripted gate can raise the abort limit for one run",),
    ),
    (
        "the-scripted-gate-peeks-at-the-pocket-pose",
        {"scripted_mut": ((
            "            pocket_quat = _slice(obs, \"pocket_quat\")",
            "            pocket_quat = unwrapped._fixture_quat",
        ),)},
        ("the scripted insertion never touches the target truth",),
    ),
    (
        "the-scripted-gate-hardcodes-a-channel-offset",
        {"scripted_mut": ((
            "    lo, hi = insertion_math.OBS_SLICES[name]",
            "    lo, hi = 12, 15",
        ),)},
        ("the scripted insertion reads the observation through OBS_SLICES",),
    ),
    (
        "osc-drives-keep-the-usd-gains",
        {"env_mut": (("                _act.stiffness = 0.0\n                _act.damping = 0.0\n",
                      "                pass\n"),)},
        ("under osc the drive stiffness AND damping are zeroed before the Articulation is built",),
    ),
    (
        "osc-gravity-compensation-off",
        {"env_mut": (("            gravity_compensation=True,\n", "            gravity_compensation=False,\n"),)},
        ("the OSC cfg is Decision (2)/(5): pose_abs, fixed, full decoupling, gravity comp ON, nullspace none",),
    ),
    (
        "osc-nullspace-position",
        {"env_mut": (("            nullspace_control=\"none\",\n", "            nullspace_control=\"position\",\n"),)},
        ("the OSC cfg is Decision (2)/(5): pose_abs, fixed, full decoupling, gravity comp ON, nullspace none",),
    ),
    (
        "tilt-cone-typed-in-radians",
        {"cfg_mut": (("    osc_tilt_clamp_rad: float = math.radians(8.52)\n",
                      "    osc_tilt_clamp_rad: float = 0.1487\n"),)},
        ("the tilt cone is written as the CAD degrees, not a typed radian",),
    ),
    (
        "withdrawal-indexes-local-rows",
        {"env_mut": (("                _rows = idx[_bad]\n",
                      "                _rows = _bad.nonzero().reshape(-1)\n"),)},
        ("the withdrawn rows are ENV IDS (idx[_bad]), not mask positions",),
    ),
    (
        "start-goal-lateral-axes-swapped",
        {"env_mut": (("            _goal[:, 0] = self._start_lat_off[:, 0]\n"
                      "            _goal[:, 1] = self._start_lat_off[:, 1]\n",
                      "            _goal[:, 0] = self._start_lat_off[:, 1]\n"
                      "            _goal[:, 1] = self._start_lat_off[:, 0]\n"),)},
        ("the residual's goal is this reset's own lateral offset and height, per axis",),
    ),
    (
        "dropped-count-books-the-whole-batch",
        {"env_mut": (("                self._start_solve_flags_dropped += _n_bad\n",
                      "                self._start_solve_flags_dropped += int(idx.numel())\n"),)},
        ("the dropped-booking count adds the SAME quantity the guard tested",),
    ),
    (
        "the-post-loop-pose-read-is-dropped",
        {"env_mut": (("\n            *_, tip_rel = self._peg_geometry()\n", "\n"),)},
        ("a FRESH _peg_geometry read sits between the solve loop and the depth seed",),
    ),
    (
        "clamp-box-smaller-than-the-sideways-reach",
        {"cfg_mut": (("    osc_pos_clamp_m: float = 0.100\n", "    osc_pos_clamp_m: float = 0.02\n"),)},
        ("the tip clamp box holds one action step beyond the SIDEWAYS reach",
         "the x/y half-width holds the disk reach plus one step and at most the set air"),
    ),
    (
        "clamp-box-z-back-to-the-xy-width",
        {"cfg_mut": (("    osc_pos_clamp_z_m: float = 0.2915\n",
                      "    osc_pos_clamp_z_m: float = 0.100\n"),)},
        ("the tip clamp box holds one action step above the highest legal start",
         "the z half-width is WIDER than the x/y one"),
    ),
    (
        "z-half-width-slips-to-millimetres",
        {"cfg_mut": (("    osc_pos_clamp_z_m: float = 0.2915\n",
                      "    osc_pos_clamp_z_m: float = 291.5\n"),)},
        ("the tip clamp box stays BELOW the home tip standoff",),
    ),
    (
        "the-fixture-z-spread-lifts-the-box-over-the-home",
        {"cfg_mut": (("    fixture_pos_noise_z = 0.05\n",
                      "    fixture_pos_noise_z = 0.15\n"),)},
        ("the tip clamp box stays BELOW the home tip standoff",),
    ),
    (
        "the-autodr-start-band-grows-past-the-box",
        {"autodr_mut": (('    DimSpec("start_height", "m", 0.0, 0.270, True,\n',
                         '    DimSpec("start_height", "m", 0.0, 0.280, True,\n'),)},
        ("the tip clamp box holds one action step above the highest legal start",),
    ),
    (
        "the-autodr-tilt-maximum-swings-past-the-sideways-box",
        {"autodr_mut": (('    DimSpec("tilt", "rad", 0.0, 0.17453292519943295, True, _D178),\n',
                         '    DimSpec("tilt", "rad", 0.0, 0.9, True, _D178),\n'),)},
        ("the tip clamp box holds one action step beyond the SIDEWAYS reach",
         "the x/y half-width holds the disk reach plus one step and at most the set air",
         "the yaw band holds the worst start yaw, the belief error and one rotation step"),
    ),
    (
        "bounds-max-doubles-every-upper-edge",
        {"autodr_mut": (("        return {d.name: (d.lo_max, d.hi_max) for d in self.dims}\n",
                         "        return {d.name: (d.lo_max, d.hi_max * 2.0) for d in self.dims}\n"),)},
        ("the tip clamp box holds one action step above the highest legal start",
         "the tip clamp box holds one action step beyond the SIDEWAYS reach",
         "the x/y half-width holds the disk reach plus one step and at most the set air",
         "the yaw band holds the worst start yaw, the belief error and one rotation step"),
    ),
    (
        "apply-action-always-joint-pd",
        {"env_mut": (("        if self._control_mode == \"osc\":\n            self._apply_osc()\n            return\n"
                      "        self.robot.set_joint_position_target(self._joint_targets)\n",
                      "        self.robot.set_joint_position_target(self._joint_targets)\n"),)},
        ("_apply_action branches on the control mode: OSC per physics step, else the PD target",),
    ),
    (
        "tip-clamp-skipped",
        {"env_mut": (("        tgt_pos_w = insertion_math.clamp_tip_in_box(\n"
                      "            tgt_pos_w,\n"
                      "            tgt_quat_w,\n"
                      "            self._tip_offset_local,\n"
                      "            centre_w,\n"
                      "            float(self.cfg.osc_pos_clamp_m),\n"
                      "            float(self.cfg.osc_pos_clamp_m),\n"
                      "            float(self.cfg.osc_pos_clamp_z_m),\n"
                      "        )\n", ""),)},
        ("_apply_osc: delta, cone, box, then the OSC, then joint EFFORTS -- in that order",
         "the tip box gets x, y and z half-widths, z from its OWN cfg field",
         "both clamps are ASSIGNED back, not called for a side effect they have not got",
         "the yaw band is ASSIGNED back, after the tilt cone and before the box"),
    ),
    (
        "tip-clamp-z-argument-takes-the-xy-field",
        {"env_mut": (("            float(self.cfg.osc_pos_clamp_z_m),\n",
                      "            float(self.cfg.osc_pos_clamp_m),\n"),)},
        ("the tip box gets x, y and z half-widths, z from its OWN cfg field",),
    ),
    (
        "the-tip-box-is-centred-on-the-nominal-entrance",
        {"env_mut": (("        centre_w = self._entrance_pos + self.scene.env_origins\n",
                      "        centre_w = self._entrance_nominal + self.scene.env_origins\n"),)},
        ("the tip box is centred on THIS episode's pocket entrance",),
    ),
    (
        "clamp-box-z-no-longer-reaches-the-seat",
        {"cfg_mut": (("    osc_pos_clamp_z_m: float = 0.2915\n",
                      "    osc_pos_clamp_z_m: float = 0.03\n"),)},
        ("the tip clamp box holds one action step above the highest legal start",
         "the tip clamp box still reaches the seat below the entrance",
         "the z half-width is WIDER than the x/y one"),
    ),
    (
        "the-box-clamp-is-called-for-its-side-effect",
        {"env_mut": (("        tgt_pos_w = insertion_math.clamp_tip_in_box(\n",
                      "        insertion_math.clamp_tip_in_box(\n"),)},
        ("both clamps are ASSIGNED back, not called for a side effect they have not got",
         "the yaw band is ASSIGNED back, after the tilt cone and before the box"),
    ),
    (
        "the-dropped-count-counts-the-reset-not-the-mask",
        {"env_mut": (("            _n_bad = int(_bad.sum().item())\n",
                      "            _n_bad = int(idx.numel())\n"),)},
        ("the dropped-booking count is the MASK's sum, not the size of the reset",),
    ),
    (
        "the-dropped-log-publishes-the-unconverged-counter",
        {"env_mut": (('            self.extras["log"]["dr/start_solve_flags_dropped"] = float(\n'
                      "                self._start_solve_flags_dropped\n"
                      "            )\n",
                      '            self.extras["log"]["dr/start_solve_flags_dropped"] = float(\n'
                      "                self._start_solve_unconverged\n"
                      "            )\n"),)},
        ("both published dropped-booking figures read the withdrawal counter itself",),
    ),
    (
        "the-xy-clamp-grows-past-the-set-air-ceiling",
        {"cfg_mut": (("    osc_pos_clamp_m: float = 0.100\n",
                      "    osc_pos_clamp_m: float = 0.12\n"),)},
        ("the x/y half-width holds the disk reach plus one step and at most the set air",),
    ),
    (
        "the-start-solve-tolerance-passes-the-play",
        {"cfg_mut": (("    start_pose_solve_tol_m: float = 0.00005\n",
                      "    start_pose_solve_tol_m: float = 0.005\n"),)},
        ("the start-solve tolerance stays inside the pocket play",),
    ),
    (
        "the-step-limit-goes-back-to-d196s-1cm",
        {"cfg_mut": (("    osc_pos_step_limit_m: float = 0.02\n",
                      "    osc_pos_step_limit_m: float = 0.01\n"),)},
        ("the x/y half-width holds the disk reach plus one step and at most the set air",),
    ),
    (
        "the-rotation-step-is-typed-as-converted-degrees",
        {"cfg_mut": (("    osc_rot_step_limit_rad: float = 0.097\n",
                      "    osc_rot_step_limit_rad: float = math.radians(5.5577)\n"),)},
        ("the rotation step limit is written as Factory's typed radian",),
    ),
    (
        "the-osc-delta-bypasses-the-ema",
        {"env_mut": (("            self._osc_delta[:, 0:3] = self._ema_actions[:, 0:3] * "
                      "float(self.cfg.osc_pos_step_limit_m)\n",
                      "            self._osc_delta[:, 0:3] = clamped[:, 0:3] * "
                      "float(self.cfg.osc_pos_step_limit_m)\n"),)},
        ("the OSC delta and the joint target read the EMA state; the clamped draw feeds ONLY the EMA",),
    ),
    (
        "the-ema-weights-the-old-state",
        {"env_mut": (("            self._ema_actions, clamped, float(self.cfg.action_ema_factor)\n",
                      "            clamped, self._ema_actions, float(self.cfg.action_ema_factor)\n"),)},
        ("the action EMA is updated once per step from the CLAMPED draw, the cfg factor on the new sample",),
    ),
    (
        "the-ema-reads-the-unclamped-draw",
        {"env_mut": (("            self._ema_actions, clamped, float(self.cfg.action_ema_factor)\n",
                      "            self._ema_actions, actions, float(self.cfg.action_ema_factor)\n"),)},
        ("the action EMA is updated once per step from the CLAMPED draw, the cfg factor on the new sample",
         "the OSC delta and the joint target read the EMA state; the clamped draw feeds ONLY the EMA"),
    ),
    (
        "the-ema-state-is-sized-by-the-observation",
        {"env_mut": (("        self._ema_actions = torch.zeros(self.num_envs, self.cfg.action_space, "
                      "device=self.device)\n",
                      "        self._ema_actions = torch.zeros(self.num_envs, self.cfg.observation_space, "
                      "device=self.device)\n"),)},
        ("the action EMA state is one row per env over the action width",),
    ),
    (
        "the-ema-survives-the-reset",
        {"env_mut": (("            self._ema_actions[env_ids] = 0.0\n", ""),)},
        ("_reset_idx zeroes the action EMA on the reset envs (Factory, factory_env.py:809)",),
    ),
    (
        "the-rate-measurement-moves-onto-the-ema-state",
        {"env_mut": (("        self._last_action_rate = torch.linalg.norm(self.actions - self._prev_actions, dim=-1)\n",
                      "        self._last_action_rate = torch.linalg.norm(self._ema_actions - self._prev_actions, "
                      "dim=-1)\n"),)},
        ("the action-rate measurement stays on the RAW draw, not the EMA state",),
    ),
    (
        "the-rate-weight-goes-back-to-d168",
        {"cfg_mut": (("    action_rate_scale: float = 0.0\n",
                      "    action_rate_scale: float = 0.0034\n"),)},
        ("action_rate_scale is Factory's action_grad_penalty_scale, 0.0 (factory_tasks_cfg.py:71)",),
    ),
    (
        "the-ema-factor-is-switched-off",
        {"cfg_mut": (("    action_ema_factor: float = 0.2\n",
                      "    action_ema_factor: float = 1.0\n"),)},
        ("action_ema_factor is Factory's ema_factor, 0.2 (factory_env_cfg.py:51)",),
    ),
    (
        "the-report-types-the-ema-factor",
        {"env_mut": (('            print(f"  action EMA:   {cfg.action_ema_factor} on the new clamped draw, ',
                      '            print(f"  action EMA:   0.2 on the new clamped draw, '),)},
        ("the startup report prints the action EMA factor (D-197)",),
    ),
    (
        "the-yaw-band-is-skipped",
        {"env_mut": (("        tgt_quat_w = insertion_math.clamp_yaw_to_band(\n"
                      "            tgt_quat_w, float(self.cfg.osc_yaw_clamp_rad), seat_quat_w\n"
                      "        )\n", ""),)},
        ("the yaw band is measured against this episode's SEATED orientation",
         "the yaw band is ASSIGNED back, after the tilt cone and before the box"),
    ),
    (
        "the-yaw-band-gets-the-bare-pocket-quaternion",
        {"env_mut": (("            tgt_quat_w, float(self.cfg.osc_yaw_clamp_rad), seat_quat_w\n",
                      "            tgt_quat_w, float(self.cfg.osc_yaw_clamp_rad), self._fixture_quat\n"),)},
        ("the yaw band is measured against this episode's SEATED orientation",),
    ),
    (
        "the-seated-reference-drops-the-seat-quaternion",
        {"env_mut": (("        seat_quat_w = insertion_math.quat_mul(self._fixture_quat, self._seat_quat_local)\n",
                      "        seat_quat_w = self._fixture_quat\n"),)},
        ("the yaw band is measured against this episode's SEATED orientation",),
    ),
    (
        "the-yaw-band-runs-after-the-box",
        {"env_mut": (
            ("        tgt_quat_w = insertion_math.clamp_yaw_to_band(\n"
             "            tgt_quat_w, float(self.cfg.osc_yaw_clamp_rad), seat_quat_w\n"
             "        )\n", ""),
            ("        self._osc_target_pose_w[:, 0:3] = tgt_pos_w\n",
             "        tgt_quat_w = insertion_math.clamp_yaw_to_band(\n"
             "            tgt_quat_w, float(self.cfg.osc_yaw_clamp_rad), seat_quat_w\n"
             "        )\n"
             "        self._osc_target_pose_w[:, 0:3] = tgt_pos_w\n"),
        )},
        ("the yaw band is ASSIGNED back, after the tilt cone and before the box",),
    ),
    (
        "the-yaw-band-sits-below-its-sufficiency",
        {"cfg_mut": (("    osc_yaw_clamp_rad: float = math.radians(12.30)\n",
                      "    osc_yaw_clamp_rad: float = math.radians(7.74)\n"),)},
        ("the yaw band holds the worst start yaw, the belief error and one rotation step",),
    ),
    (
        "the-yaw-band-grows-past-the-set-air",
        {"cfg_mut": (("    osc_yaw_clamp_rad: float = math.radians(12.30)\n",
                      "    osc_yaw_clamp_rad: float = math.radians(45.0)\n"),)},
        ("the yaw band carries at most the set air above its need",),
    ),
    (
        "the-autodr-yaw-maximum-grows-past-the-band",
        {"autodr_mut": (('    DimSpec("yaw", "rad", -0.08726646259971648, 0.08726646259971648, False, _D178),\n',
                         '    DimSpec("yaw", "rad", -0.10471975511965977, 0.10471975511965977, False, _D178),\n'),)},
        ("the yaw band holds the worst start yaw, the belief error and one rotation step",),
    ),
    (
        "the-startup-yaw-reads-the-bare-pocket-frame",
        {"env_mut": (("                _seat_w = insertion_math.quat_mul(self._fixture_quat, self._seat_quat_local)\n",
                      "                _seat_w = self._fixture_quat\n"),)},
        ("the startup report prints the yaw band and the reset yaw against the seated orientation",),
    ),
    (
        "control-mode-resolved-after-super",
        {"env_mut": (
            ("        self._control_mode = resolve_control_mode(cfg)\n", ""),
            ("        super().__init__(cfg, render_mode, **kwargs)\n",
             "        super().__init__(cfg, render_mode, **kwargs)\n"
             "        self._control_mode = resolve_control_mode(cfg)\n"),
        )},
        ("the control mode is resolved BEFORE the base class reads the cfg",),
    ),
    (
        "osc-placeholder-unlabelled",
        {"cfg_mut": (("    \"osc_kp_pos\": (\n", "    \"osc_kp_pos_unlabelled\": (\n"),)},
        ("every OSC number without a source is a labelled placeholder",
         "the cfg gives every placeholder a real value"),
    ),
    (
        "tilt-insert-ignores-the-mode",
        {"tilt_mut": (("            if control_mode == \"osc\":\n                # ONE hop",
                       "            if False:\n                # ONE hop"),)},
        ("tilt_insert branches on control_mode: OSC action or the IK chain",),
    ),
    (
        "torque-log-dropped",
        {"tilt_mut": (("            tau = robot.data.applied_torque[:, :n_joints].abs()\n",
                       "            tau = torch.zeros(num_envs, n_joints, device=device)\n"),)},
        ("tilt_insert logs the joint torques against maxForce",),
    ),
    (
        "the-provenance-prefix-calls-every-line-a-testwert",
        {"env_mut": (('                print(f"  [provenance] {_line}")\n',
                      '                print(f"  [TESTWERT-check] {_line}")\n'),)},
        ("the provenance lines are printed under a neutral prefix, not a [TESTWERT] one",),
    ),
    (
        "the-noise-model-is-set-after-the-base-class",
        {"env_mut": ((
            "        cfg.observation_noise_model = obs_noise.resolve_obs_noise_model(cfg)\n"
            "        super().__init__(cfg, render_mode, **kwargs)\n",
            "        super().__init__(cfg, render_mode, **kwargs)\n"
            "        cfg.observation_noise_model = obs_noise.resolve_obs_noise_model(cfg)\n",
        ),)},
        ("the observation noise model is set BEFORE the base class reads the cfg",),
    ),
    (
        "the-pocket-bias-accumulates-over-resets",
        {"obs_noise_mut": ((
            '                    n_min=-bias_std, n_max=bias_std, operation="abs"\n',
            '                    n_min=-bias_std, n_max=bias_std, operation="add"\n',
        ),)},
        ("the per-episode bias is drawn ABSOLUTE, not added onto the last one",),
    ),
    (
        "the-pocket-bias-becomes-one-sided",
        {"obs_noise_mut": ((
            '                    n_min=-bias_std, n_max=bias_std, operation="abs"\n',
            '                    n_min=0.0 * bias_std, n_max=bias_std, operation="abs"\n',
        ),)},
        ("the pocket bias is uniform in [-bias_std, +bias_std]",),
    ),
    (
        "the-abs-marker-moves-to-the-per-step-noise",
        {"obs_noise_mut": ((
            "                noise_cfg=GaussianNoiseCfg(std=step_std),\n"
            "                bias_noise_cfg=UniformNoiseCfg(\n"
            '                    n_min=-bias_std, n_max=bias_std, operation="abs"\n'
            "                ),\n",
            '                noise_cfg=GaussianNoiseCfg(std=step_std, operation="abs"),\n'
            "                bias_noise_cfg=UniformNoiseCfg(\n"
            "                    n_min=-bias_std, n_max=bias_std\n"
            "                ),\n",
        ),)},
        ("the per-episode bias is drawn ABSOLUTE, not added onto the last one",),
    ),
    (
        "the-noise-model-is-never-switched-off",
        {"obs_noise_mut": (("        return None\n", "        pass\n"),)},
        ("both sigmas at zero switch the observation noise model off entirely",),
    ),
    (
        "the-off-switch-needs-only-one-sigma-at-zero",
        {"obs_noise_mut": (("    if pocket == 0.0 and force == 0.0 and torque == 0.0:\n",
                            "    if pocket == 0.0 or force == 0.0 or torque == 0.0:\n"),)},
        ("both sigmas at zero switch the observation noise model off entirely",),
    ),
    (
        "the-inherited-missing-noise-cfg-comes-back",
        {"obs_noise_mut": ((
            "    noise_cfg: NoiseCfg | None = None\n",
            "",
        ),)},
        ("the inherited MISSING noise_cfg is bound, so cfg.validate() passes",),
    ),
    (
        "the-bound-noise-cfg-is-itself-missing",
        {"obs_noise_mut": ((
            "    noise_cfg: NoiseCfg | None = None\n",
            "    noise_cfg: NoiseCfg = MISSING\n",
        ),)},
        ("the inherited MISSING noise_cfg is bound, so cfg.validate() passes",),
    ),
    (
        "the-bias-buffer-is-left-at-the-library-size",
        {"obs_noise_mut": ((
            "        _width = insertion_math.obs_dim(str(noise_model_cfg.obs_mode))\n"
            "        self._bias = torch.zeros(num_envs, _width, device=self._device)\n"
            "        self._num_components = _width\n",
            "",
        ),)},
        ("the bias buffer and its component count are sized up front",),
    ),
    (
        "the-grasp-offset-reaches-the-termination",
        {"env_mut": ((
            "            body_quat,\n            self._tip_offset_local,\n",
            "            body_quat,\n            self._tip_offset_local + self._grasp_obs_off,\n",
        ),)},
        ("the grasp offset reaches the OBSERVATION and no paying path",),
    ),
    (
        "the-grasp-offset-leaks-into-the-instrument",
        {"env_mut": ((
            "        tip_w = peg_pos_w + torch.matmul(peg_axes, self._tip_offset_local)\n",
            "        tip_w = peg_pos_w + torch.matmul(\n"
            "            peg_axes, self._tip_offset_local + self._grasp_obs_off\n"
            "        )\n",
        ),)},
        ("the grasp offset reaches the OBSERVATION and no paying path",),
    ),
    (
        "the-grasp-offset-is-never-actually-applied",
        {"env_mut": ((
            "            self._grasp_obs_off[idx, 0] = (\n"
            "                2.0 * self._reset_unit[idx][:, self._col[\"grasp_obs_x\"]] - 1.0\n"
            "            ) * float(self.cfg.grasp_obs_offset_x_m)\n",
            "            self._grasp_obs_off[idx, 0] = 0.0\n",
        ),)},
        ("the grasp offset is MAPPED from its own table column, never drawn beside the row",),
    ),
    (
        "pocket-yaw-belief-drawn-beside-the-row",
        {"env_mut": ((
            '                2.0 * self._reset_unit[idx][:, self._col["pocket_obs_yaw"]] - 1.0\n',
            "                2.0 * torch.rand(idx.numel(), device=self.device) - 1.0\n",
        ),)},
        (
            "the pocket-yaw belief error is MAPPED from its own table column, never drawn beside the row",
            "ONE place draws the reset conditions -- _draw_reset_conditions",
        ),
    ),
    (
        "pocket-yaw-belief-one-sided",
        {"env_mut": ((
            '                2.0 * self._reset_unit[idx][:, self._col["pocket_obs_yaw"]] - 1.0\n',
            '                self._reset_unit[idx][:, self._col["pocket_obs_yaw"]]\n',
        ),)},
        ("the pocket-yaw belief error is MAPPED from its own table column, never drawn beside the row",),
    ),
    (
        "pocket-yaw-belief-reaches-the-reward",
        {"env_mut": ((
            "            self._tilt_phi_prev,\n",
            "            self._tilt_phi_prev + 0.0 * self._pocket_obs_yaw,\n",
        ),)},
        ("the pocket-yaw belief error reaches the OBSERVATION and no paying path",),
    ),
    (
        "tip-rel-reads-the-true-pocket",
        {"env_mut": (("                belief_rot,\n", "                pocket_rot,\n"),)},
        ("tip_rel and the yaw come from the BELIEVED pocket frame",),
    ),
    (
        "pocket-yaw-belief-zeroed-in-the-observation",
        {"env_mut": ((
            "            self._fixture_quat, self._pocket_obs_yaw\n",
            "            self._fixture_quat, torch.zeros_like(self._pocket_obs_yaw)\n",
        ),)},
        ("tip_rel and the yaw come from the BELIEVED pocket frame",),
    ),
    (
        "pocket-quat-channel-left-true",
        {"env_mut": (
            ("        belief_rot, pocket_quat = insertion_math.believed_pocket_frame(\n",
             "        belief_rot, _believed_quat = insertion_math.believed_pocket_frame(\n"),
            ("        # The force block (D-114).",
             "        pocket_quat = insertion_math.canonicalize_quat(self._fixture_quat)\n\n"
             "        # The force block (D-114)."),
        )},
        ("the pocket_quat channel carries the BELIEVED pocket quaternion",),
    ),
    (
        "tilt-potential-reads-the-belief",
        {"env_mut": ((
            "                    self.robot.data.body_quat_w[:, self._peg_body_idx], pocket_rot\n",
            "                    self.robot.data.body_quat_w[:, self._peg_body_idx], belief_rot\n",
        ),)},
        ("the tilt potential keeps the TRUE pocket frame",),
    ),
    (
        "pocket-yaw-belief-escapes-the-negative-sweep",
        {"cfg_mut": (('                  "pocket_obs_yaw_rad"):\n', "                  ):\n"),)},
        ("the pocket-yaw belief error is in the cfg's negative-scatter sweep",),
    ),
    (
        "pocket-yaw-belief-off-by-default",
        {"cfg_mut": (("    pocket_obs_yaw_rad: float = 0.0223402\n",
                      "    pocket_obs_yaw_rad: float = 0.0\n"),)},
        ("the pocket-yaw belief error defaults to 0.0223402 rad",),
    ),
    (
        "pocket-yaw-belief-missing-from-demo-metrics",
        {"env_mut": (('            "pocket_obs_yaw_rad": self.cfg.pocket_obs_yaw_rad,\n', ""),)},
        ("demo_metrics.json carries the observation noise and the grasp scatter",),
    ),
    (
        "the-grasp-offset-is-mapped-off-the-wrong-column",
        {"env_mut": ((
            "                2.0 * self._reset_unit[idx][:, self._col[\"grasp_obs_x\"]] - 1.0\n",
            "                2.0 * self._reset_unit[idx][:, self._col[\"friction\"]] - 1.0\n",
        ),)},
        ("the grasp offset is MAPPED from its own table column, never drawn beside the row",),
    ),
    (
        "the-grasp-offset-is-drawn-beside-the-row-again",
        {"env_mut": ((
            "                2.0 * self._reset_unit[idx][:, self._col[\"grasp_obs_x\"]] - 1.0\n",
            "                2.0 * torch.rand(int(idx.numel()), device=self.device) - 1.0\n",
        ),)},
        ("ONE place draws the reset conditions -- _draw_reset_conditions",
         "the grasp offset is MAPPED from its own table column, never drawn beside the row"),
    ),
    (
        "one-identity-run-stops-pinning-the-force-sigma",
        {"seat_mut": (("    env_cfg.force_obs_noise_std_n = 0.0\n", ""),)},
        ("the identity runs pin all four observation scatter fields to zero",),
    ),
    (
        "the-scatter-pin-moves-past-gym-make",
        {"seat_mut": (
            ("    env_cfg.grasp_obs_offset_x_m = 0.0\n", ""),
            ("    env = gym.make(args_cli.task, cfg=env_cfg)\n",
             "    env = gym.make(args_cli.task, cfg=env_cfg)\n"
             "    env_cfg.grasp_obs_offset_x_m = 0.0\n"),
        )},
        ("the identity runs pin all four observation scatter fields to zero",),
    ),
    (
        "the-pocket-bias-sigma-slips-to-millimetres",
        {"cfg_mut": (("    obs_noise_pocket_pos_std_m: float = 0.005",
                      "    obs_noise_pocket_pos_std_m: float = 5.0"),)},
        ("the pocket bias half width is TacSL's 5 mm and stays under the seat depth",),
    ),
    (
        "the-force-sigma-slips-past-the-abort-limit",
        {"cfg_mut": (("    force_obs_noise_std_n: float = 3.5",
                      "    force_obs_noise_std_n: float = 3500.0"),)},
        ("the force sigma is the UR5e datasheet's 3.5 N and stays under the abort force",),
    ),
    (
        "the-grasp-belief-error-slips-to-millimetres",
        {"cfg_mut": (("    grasp_obs_offset_x_m: float = 0.003",
                      "    grasp_obs_offset_x_m: float = 3.0"),)},
        ("the grasp belief error is Factory's 3 mm and stays inside the short-axis pocket wall",),
    ),
    (
        "the-friction-centre-leaves-the-dr-band",
        {"tasks_mut": (("CONTACT_FRICTION = 0.14", "CONTACT_FRICTION = 0.4"),)},
        ("the contact-friction centre lies inside the DR_DIMS friction band",),
    ),
    (
        "the-report-stops-reading-the-noise-model",
        {"env_mut": ((
            "        _noise = getattr(self, \"_observation_noise_model\", None)\n",
            "        _noise = None\n",
        ),)},
        ("the startup report reads the per-episode observation scatter back",),
    ),
    (
        "the-report-slices-the-bias-with-the-force-key",
        {"env_mut": ((
            "            _t_lo, _t_hi = insertion_math.OBS_SLICES[\"tip_rel\"]\n",
            "            _t_lo, _t_hi = insertion_math.OBS_SLICES[\"force\"]\n",
        ),)},
        ("the startup report slices the bias with the tip_rel entry of OBS_SLICES",),
    ),
    (
        "the-obs-call-counter-is-reset-per-episode",
        {"env_mut": ((
            "        ready = hasattr(self, \"_joint_targets\")\n",
            "        ready = hasattr(self, \"_joint_targets\")\n        self._obs_calls = 0\n",
        ),)},
        ("_obs_calls is written in exactly two places: = in __init__, += in _get_observations",),
    ),
)


def _run(**kwargs) -> dict:
    return {name: ok for name, ok, _ in build_checks(**kwargs)}


def _run_mutation(index: int) -> tuple[str, dict]:
    """One mutation by its index in ``MUTATIONS`` -- the unit of the pool."""
    name, kwargs, _ = MUTATIONS[index]
    return name, _run(**kwargs)


def _self_test() -> int:
    rc = 0
    print("[check_env_wiring] counter-proof (D-080)")
    baseline = _run()
    if not all(baseline.values()):
        bad = [k for k, v in baseline.items() if not v]
        print(f"  FAIL  baseline is not green, counter-proof is meaningless: {bad}")
        return 1
    print(f"  PASS  baseline: {len(baseline)} checks green")

    results: dict[str, dict] = {}
    with concurrent.futures.ProcessPoolExecutor() as pool:
        for name, got in pool.map(_run_mutation, range(len(MUTATIONS)), chunksize=4):
            results[name] = got
    for name, kwargs, expected in MUTATIONS:
        got = results[name]
        flipped = {k for k in baseline if k not in got or not got[k]}
        unknown = set(expected) - set(baseline)
        missing = set(expected) - flipped
        extra = flipped - set(expected)
        if unknown:
            print(f"  FAIL  {name}: names no existing check {sorted(unknown)}")
            rc = 1
        elif not flipped:
            print(f"  FAIL  {name}: flipped NOTHING -- these checks can never fail")
            rc = 1
        elif missing or extra:
            print(f"  FAIL  {name}: missing {sorted(missing)} extra {sorted(extra)}")
            rc = 1
        else:
            print(f"  PASS  {name}: flips exactly {len(flipped)} declared check(s)")

    print()
    print("COUNTER-PROOF PASSED" if rc == 0 else "COUNTER-PROOF FAILED")
    return rc


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    results = build_checks()
    failures = []
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)
    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print(f"ALL {len(results)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
