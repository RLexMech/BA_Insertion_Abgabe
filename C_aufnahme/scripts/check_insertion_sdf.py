# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Check interpenetration math and mesh distance signs.

--self-test runs offline; --pose-ladder measures known separated and overlapping
poses on the GPU. Distances are sampled mesh estimates, not PhysX penetration.
The fixture is not watertight; check_fixture_mesh.py checks its open edges."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import pathlib
import sys
import types

SCRIPT_MARKER = "check_insertion_sdf-2026-09-15a-poseladder"

_HERE = pathlib.Path(__file__).resolve().parent
_PKG_DIR = (
    _HERE.parent / "source" / "insertion" / "insertion" / "tasks" / "direct" / "insertion"
)
_SDF_SRC = _PKG_DIR / "insertion_sdf.py"
_MATH_SRC = _PKG_DIR / "insertion_math.py"
_TASKS_SRC = _PKG_DIR / "insertion_tasks_cfg.py"

MAX_DIST_M = 1.5
NUM_SAMPLE_POINTS = 64000

ON_SURFACE_TOL_M = 3e-5

PROBE_MULTIPLES = (0.5, 1.0, 2.0, 4.0)
PROBE_ASSERT_FROM = 2.0


IdentityVerdict = collections.namedtuple(
    "IdentityVerdict", "p4 p99 dmax n_above n_points"
)


def percentile(values, q: float) -> float:
    """The q-th percentile with linear interpolation, on a plain list."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    v = sorted(float(x) for x in values)
    if len(v) == 1:
        return v[0]
    idx = (len(v) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (idx - lo)


def judge_identity(abs_dists, tol: float) -> IdentityVerdict:
    """P4 over the identity transform's absolute distances."""
    vals = [abs(float(x)) for x in abs_dists]
    n = len(vals)
    if n == 0:
        raise ValueError("judge_identity needs at least one distance")
    p99 = percentile(vals, 99.0)
    dmax = max(vals)
    n_above = sum(1 for x in vals if x > tol)
    return IdentityVerdict(p99 <= tol, p99, dmax, n_above, n)


def _ident(bulk_value, bulk_n, tail_value, tail_n):
    return [bulk_value] * bulk_n + [tail_value] * tail_n


IDENTITY_CASES = {
    "I1_rare_outliers_pass": (_ident(1e-9, 995, 1e-4, 5), True),
    "I2_many_outliers_fail": (_ident(1e-9, 950, 1e-4, 50), False),
    "I3_rt90_shape_passes": (_ident(5.358e-6, 953, 5.358e-6, 0), True),
    "I4_broad_shift_fails": (_ident(1e-4, 1000, 1e-4, 0), False),
}


def run_identity_cases() -> int:
    """Run IDENTITY_CASES against D-152's tolerance. Returns the mismatches."""
    bad = 0
    for name, (vals, want) in IDENTITY_CASES.items():
        v = judge_identity(vals, ON_SURFACE_TOL_M)
        if v.p4 == want:
            print(f"[check_insertion_sdf]   identity {name}: PASS "
                  f"(P4 {v.p4}, p99 {v.p99:.3e} m, max {v.dmax:.3e} m, "
                  f"{v.n_above} of {v.n_points} above tol)")
        else:
            print(f"[check_insertion_sdf]   identity {name}: FAIL -- "
                  f"expected P4 {want}, got {v.p4} "
                  f"(p99 {v.p99:.3e} m, max {v.dmax:.3e} m)")
            bad += 1
    return bad


LadderRow = collections.namedtuple("LadderRow", "name mult dist dmin dmean asserted")
LadderVerdict = collections.namedtuple(
    "LadderVerdict", "p2 p3 p5 p2_bad p3_bad p5_bad n_asserted n_p3_pairs"
)


def judge_ladder(rows, diameter: float) -> LadderVerdict:
    """Turn the measured ladder into P2, P3 and P5."""
    n_asserted = sum(1 for r in rows if r.asserted)

    p2_bad = [(r.name, r.mult, r.dmin) for r in rows if r.asserted and r.dmin < 0.0]

    p5_bad = [
        (r.name, r.mult, r.dmean)
        for r in rows
        if r.asserted and not (r.dist - diameter <= r.dmean <= r.dist + diameter)
    ]

    by_dir: dict[str, list] = {}
    for r in rows:
        if r.asserted:
            by_dir.setdefault(r.name, []).append(r)
    p3_bad, n_p3_pairs = [], 0
    for name in sorted(by_dir):
        seq = sorted(by_dir[name], key=lambda r: r.mult)
        means = [max(0.0, r.dmean) for r in seq]
        n_p3_pairs += max(0, len(means) - 1)
        if any(b < a for a, b in zip(means, means[1:])):
            p3_bad.append((name, means))

    p2 = (not p2_bad) and n_asserted > 0
    p3 = (not p3_bad) and n_p3_pairs > 0
    p5 = not p5_bad
    return LadderVerdict(p2, p3, p5, p2_bad, p3_bad, p5_bad, n_asserted, n_p3_pairs)


def load_sdf_module(sdf_source: str | None = None, pkg_name: str = "_insertion_pkg"):
    """Import ``insertion_sdf`` with its relative import satisfied."""
    for name in list(sys.modules):
        if name == pkg_name or name.startswith(pkg_name + "."):
            del sys.modules[name]

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(_PKG_DIR)]
    sys.modules[pkg_name] = pkg

    math_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.insertion_math", _MATH_SRC
    )
    math_mod = importlib.util.module_from_spec(math_spec)
    sys.modules[math_spec.name] = math_mod
    math_spec.loader.exec_module(math_mod)

    sdf_mod = types.ModuleType(f"{pkg_name}.insertion_sdf")
    sdf_mod.__file__ = str(_SDF_SRC)
    sdf_mod.__package__ = pkg_name
    sys.modules[sdf_mod.__name__] = sdf_mod
    text = _SDF_SRC.read_text(encoding="utf-8") if sdf_source is None else sdf_source
    exec(compile(text, str(_SDF_SRC), "exec"), sdf_mod.__dict__)
    return sdf_mod


def load_paths_module():
    """Import ``insertion_paths`` -- stdlib only, no Isaac, no SimulationApp."""
    spec = importlib.util.spec_from_file_location(
        "_insertion_paths", _PKG_DIR / "insertion_paths.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def offline_checks(torch, sdf) -> dict[str, bool]:
    """Every check of the pure-torch half. Returns ``{name: passed}``."""
    import numpy as np

    res: dict[str, bool] = {}

    res["C1_imports_without_warp"] = (
        "warp" not in sys.modules and hasattr(sdf, "SdfDistanceQuery")
    )

    ident_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    got = np.asarray(sdf.quat_wxyz_to_xyzw(ident_wxyz))
    ok_ident = np.allclose(got, [[0.0, 0.0, 0.0, 1.0]])
    labelled = torch.tensor([[10.0, 21.0, 32.0, 43.0]])
    ok_order = np.allclose(
        np.asarray(sdf.quat_wxyz_to_xyzw(labelled)), [[21.0, 32.0, 43.0, 10.0]]
    )
    res["C2_quat_reorder"] = bool(ok_ident and ok_order)

    pos = torch.tensor([[0.3, -0.2, 0.5]])
    quat = torch.tensor([[0.7071067811865476, 0.0, 0.0, 0.7071067811865476]])
    rel_pos, rel_quat = sdf.goal_relative_transform(pos, quat, pos, quat)
    res["C3_identity_when_seated"] = bool(
        np.allclose(np.asarray(rel_pos), 0.0, atol=1e-12)
        and np.allclose(np.abs(np.asarray(rel_quat)), [[0.0, 0.0, 0.0, 1.0]], atol=1e-12)
    )

    rng = np.random.default_rng(0)

    def _rand_quat(n):
        q = rng.normal(size=(n, 4))
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    def _apply(q_wxyz, p, v):
        rot = np.asarray(sdf.im.axes_from_quat(torch.tensor(q_wxyz)))
        return np.einsum("bij,bj->bi", rot, v) + p

    n = 64
    p_curr, p_goal = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    q_curr, q_goal = _rand_quat(n), _rand_quat(n)
    pt = rng.normal(size=(n, 3))
    mp = rng.normal(size=(n, 3))

    direct = np.linalg.norm(
        _apply(q_curr, p_curr, pt) - _apply(q_goal, p_goal, mp), axis=1
    )
    rel_pos, rel_quat_xyzw = sdf.goal_relative_transform(
        torch.tensor(p_curr), torch.tensor(q_curr),
        torch.tensor(p_goal), torch.tensor(q_goal),
    )
    rq = np.asarray(rel_quat_xyzw)
    rel_wxyz = np.concatenate([rq[:, 3:4], rq[:, 0:3]], axis=1)
    substituted = np.linalg.norm(
        _apply(rel_wxyz, np.asarray(rel_pos), pt) - mp, axis=1
    )
    res["C4_substitution_preserves_distance"] = bool(
        np.allclose(direct, substituted, atol=1e-10)
    )

    d = torch.tensor([[-3.0, -1e-9, 0.0, 1e-9, 2.5]])
    res["C5_clamp_outside"] = bool(
        np.allclose(np.asarray(sdf.clamp_outside(d)), [[0.0, 0.0, 0.0, 1e-9, 2.5]])
    )

    try:
        paths = load_paths_module()
        paths.default_part_obj_path()
        try:
            paths.resolve_part_obj_path()
        except FileNotFoundError:
            pass
        res["C6_paths_need_no_isaac"] = True
    except Exception as exc:  # noqa: BLE001 - any import-time failure is the finding
        res["C6_paths_need_no_isaac"] = False
        print(f"[check_insertion_sdf]   C6 raised: {type(exc).__name__}: {exc}")
    return res


MUTATIONS = {
    "quat_not_reordered": (
        "return quat[..., [1, 2, 3, 0]]",
        "return quat",
        {"C2_quat_reorder", "C3_identity_when_seated", "C4_substitution_preserves_distance"},
    ),
    "quat_reordered_backwards": (
        "return quat[..., [1, 2, 3, 0]]",
        "return quat[..., [3, 0, 1, 2]]",
        {"C2_quat_reorder", "C3_identity_when_seated", "C4_substitution_preserves_distance"},
    ),
    "clamp_keeps_inside": (
        "return torch.where(signed_dist < 0.0, torch.zeros_like(signed_dist), signed_dist)",
        "return torch.where(signed_dist > 0.0, torch.zeros_like(signed_dist), signed_dist)",
        {"C5_clamp_outside"},
    ),
    "relative_pose_argument_order": (
        "rel_pos, rel_quat = im.relative_pose(goal_pos, goal_quat, part_pos, part_quat)",
        "rel_pos, rel_quat = im.relative_pose(part_pos, part_quat, goal_pos, goal_quat)",
        {"C4_substitution_preserves_distance"},
    ),
}


_CASE_STEP_Z = 0.061000 + 0.050514
_CASE_DISTS = tuple(m * _CASE_STEP_Z for m in PROBE_MULTIPLES)
_CASE_DIAMETER = 0.182600 + 0.248957


def _case_rows(means, mins=None, asserted=(False, False, True, True), name="z-"):
    """Build one direction's ladder from its four measured means."""
    mins = mins if mins is not None else tuple(0.5 * m for m in means)
    return [
        LadderRow(name, mult, dist, dmin, dmean, ok)
        for mult, dist, dmin, dmean, ok
        in zip(PROBE_MULTIPLES, _CASE_DISTS, mins, means, asserted)
    ]


_RT92_MEANS = (0.066, 0.024, 0.048, 0.271)

LADDER_CASES = {
    "L1_rt92_shape_passes": (_case_rows(_RT92_MEANS), (True, True, True)),
    "L2_dip_inside_asserted_fails_p3": (
        _case_rows((0.066, 0.024, 0.271, 0.048)), (True, False, True)),
    "L3_negative_min_fails_p2": (
        _case_rows(_RT92_MEANS, mins=(0.0002, 0.0002, -0.001, 0.135)),
        (False, True, True)),
    "L4_mean_off_band_fails_p5": (
        _case_rows((0.066, 0.024, 0.048, 5.0)), (True, True, False)),
    "L5_empty_ladder_is_not_a_pass": (
        _case_rows(_RT92_MEANS, asserted=(False, False, False, False)),
        (False, False, True)),
    "L6_mutation_shape_is_caught": (
        _case_rows(tuple(-m for m in _RT92_MEANS),
                   mins=tuple(-0.5 * m for m in _RT92_MEANS)),
        (False, True, False)),
}


def run_ladder_cases() -> int:
    """Run LADDER_CASES. Returns the number of cases that did not match."""
    bad = 0
    for name, (rows, want) in LADDER_CASES.items():
        v = judge_ladder(rows, _CASE_DIAMETER)
        got = (v.p2, v.p3, v.p5)
        if got == want:
            print(f"[check_insertion_sdf]   ladder {name}: PASS "
                  f"(P2/P3/P5 = {got}, {v.n_p3_pairs} P3 pairs)")
        else:
            print(f"[check_insertion_sdf]   ladder {name}: FAIL -- "
                  f"expected P2/P3/P5 = {want}, got {got} "
                  f"(p2_bad={v.p2_bad} p3_bad={v.p3_bad} p5_bad={v.p5_bad})")
            bad += 1
    return bad


def run_self_test(torch) -> int:
    base_mod = load_sdf_module()
    base = offline_checks(torch, base_mod)
    print(f"[check_insertion_sdf] marker: {SCRIPT_MARKER}")
    for name, ok in base.items():
        print(f"[check_insertion_sdf]   {name}: {'PASS' if ok else 'FAIL'}")
    failed = [n for n, ok in base.items() if not ok]
    if failed:
        print(f"[check_insertion_sdf] BASE CHECKS FAILED: {failed}")
        return 1

    source = _SDF_SRC.read_text(encoding="utf-8")
    bad = 0
    for name, (old, new, must_break) in MUTATIONS.items():
        if source.count(old) != 1:
            print(f"[check_insertion_sdf]   mutation {name}: ANCHOR NOT UNIQUE "
                  f"({source.count(old)} matches) -- the check cannot be trusted")
            bad += 1
            continue
        mutated = load_sdf_module(source.replace(old, new))
        got = {n for n, ok in offline_checks(torch, mutated).items() if not ok}
        if got == must_break:
            print(f"[check_insertion_sdf]   mutation {name}: PASS "
                  f"(broke exactly {sorted(got)})")
        else:
            print(f"[check_insertion_sdf]   mutation {name}: FAIL -- "
                  f"expected {sorted(must_break)}, broke {sorted(got)}")
            bad += 1

    load_sdf_module()

    bad += run_ladder_cases()

    bad += run_identity_cases()

    bad += run_pose_cases(torch, base_mod)

    bad += run_spacing_cases()

    total = (len(base) + len(MUTATIONS) + len(LADDER_CASES)
             + len(IDENTITY_CASES) + len(SPACING_CASES) + POSE_CASES_COUNT)
    print(f"[check_insertion_sdf] self-test: {total - bad}/{total} "
          f"({len(base)} checks + {len(MUTATIONS)} mutations + "
          f"{len(LADDER_CASES)} ladder cases + {len(IDENTITY_CASES)} identity "
          f"cases + {len(SPACING_CASES)} spacing cases + {POSE_CASES_COUNT} pose cases)")
    return 1 if bad else 0


BOX_PART_M = (0.09000, 0.14350, 0.050506)
PLAY_X_M = 0.0005876
BENCH_POINTS = (1000, 4000, 16000, 64000, 143000)
BENCH_ENVS = 128


def box_area_m2(dims) -> float:
    """Surface area of a box, square metres."""
    x, y, z = dims
    return 2.0 * (x * y + x * z + y * z)


def spacing_from_area(area_m2: float, n: int) -> float:
    """Mean spacing of ``n`` evenly sampled points on ``area_m2``, metres."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return (area_m2 / float(n)) ** 0.5


SPACING_CASES = {
    "S1_d122_1000_is_7.03mm": (1000, 7.03),
    "S2_d122_16000_is_1.76mm": (16000, 1.76),
    "S3_143000_meets_the_cross_play": (143000, 0.5876),
}


def run_spacing_cases() -> int:
    """Reproduce D-122's published table. Returns the number of mismatches."""
    area = box_area_m2(BOX_PART_M)
    bad = 0
    for name, (n, want_mm) in SPACING_CASES.items():
        got_mm = spacing_from_area(area, n) * 1000.0
        if abs(got_mm - want_mm) <= 0.01:
            print(f"[check_insertion_sdf]   spacing {name}: PASS "
                  f"({got_mm:.4f} mm against D-122's {want_mm} mm)")
        else:
            print(f"[check_insertion_sdf]   spacing {name}: FAIL -- "
                  f"expected {want_mm} mm, got {got_mm:.4f} mm")
            bad += 1
    return bad


def run_bench(args) -> int:
    """Time ``signed_distances`` over a ladder of sample counts. No verdict."""
    import time

    import torch

    sdf = load_sdf_module()
    obj_path = args.obj if args.obj is not None else load_paths_module().resolve_part_obj_path()
    query_obj = args.query_obj
    two_meshes = query_obj is not None and query_obj != obj_path
    points = (tuple(int(v) for v in args.bench_points.split(","))
              if args.bench_points else BENCH_POINTS)
    envs = int(args.bench_envs)

    print(f"[check_insertion_sdf] marker: {SCRIPT_MARKER}")
    print("[check_insertion_sdf] MODE: BENCH -- D-122's missing half, the COST. "
          "Nothing is asserted here and there is no verdict; this run measures.")
    print(f"[check_insertion_sdf] sampled obj: {obj_path}")
    print(f"[check_insertion_sdf] queried obj: {query_obj if two_meshes else '(the same file)'}")
    print(f"[check_insertion_sdf] envs {envs} (insertion_env_cfg.py:456), "
          f"warm-up {args.bench_warmup}, timed repeats {args.bench_repeats}")

    box_a = box_area_m2(BOX_PART_M)
    print(f"[check_insertion_sdf] D-122's box area {box_a * 1e4:.1f} cm2 "
          f"(90.00 x 143.50 x 50.506 mm), cross play {PLAY_X_M * 1000:.4f} mm")

    cuda = torch.cuda.is_available()
    if cuda:
        free0, total0 = torch.cuda.mem_get_info()
        print(f"[check_insertion_sdf] device memory before: "
              f"{free0 / 2**20:.0f} MiB free of {total0 / 2**20:.0f} MiB")

    mesh_a = float("nan")
    print("[check_insertion_sdf]        N   returned   build[s]   "
          "ms/call   ms/call/env   box sp[mm]  mesh sp[mm]  vs play   dev MiB used")
    for n in points:
        free_before = None
        if cuda:
            torch.cuda.synchronize()
            free_before = torch.cuda.mem_get_info()[0]
        t0 = time.perf_counter()
        query = sdf.SdfDistanceQuery(
            obj_path=obj_path,
            num_sample_points=n,
            max_dist=MAX_DIST_M,
            device=args.device,
            seed=args.seed,
            mesh_obj_path=query_obj,
        )
        build_s = time.perf_counter() - t0

        pos = torch.zeros((envs, 3), dtype=torch.float32)
        quat = torch.zeros((envs, 4), dtype=torch.float32)
        quat[:, 3] = 1.0

        for _ in range(int(args.bench_warmup)):
            query.signed_distances(pos, quat)
        if cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(int(args.bench_repeats)):
            out = query.signed_distances(pos, quat)
        if cuda:
            torch.cuda.synchronize()
        per_call_ms = (time.perf_counter() - t0) * 1000.0 / float(args.bench_repeats)

        returned = query.num_points
        mesh_a = float(query.sampled_area_m2)
        box_sp = spacing_from_area(box_a, returned) * 1000.0
        mesh_sp = spacing_from_area(mesh_a, returned) * 1000.0
        used = float("nan")
        if cuda:
            used = (free_before - torch.cuda.mem_get_info()[0]) / 2**20

        print(f"[check_insertion_sdf]  {n:7d}   {returned:8d}   {build_s:8.2f}   "
              f"{per_call_ms:7.3f}   {per_call_ms / envs:11.5f}   "
              f"{box_sp:10.3f}  {mesh_sp:11.3f}  {mesh_sp / (PLAY_X_M * 1000):7.2f}x  "
              f"{used:13.0f}")
        print(f"[check_insertion_sdf]      out tensor {tuple(out.shape)} "
              f"{out.numel() * 4 / 2**20:.1f} MiB, "
              f"queries per step {envs * returned}")
        del query, out
        if cuda:
            torch.cuda.empty_cache()

    print(f"[check_insertion_sdf] mesh area MEASURED: {mesh_a * 1e4:.1f} cm2 "
          f"against D-122's box lower bound {box_a * 1e4:.1f} cm2 "
          f"(ratio {mesh_a / box_a:.3f})")
    print("[check_insertion_sdf] BENCH DONE -- no verdict by design (D-122's "
          "count is a decision, and this run supplies its cost column).")
    return 0


POSE_LADDER_CONSTANTS = (
    "PLAY_X", "PLAY_Y", "POCKET_SEAT_DEPTH", "SEATED_SUCCESS_DEPTH",
    "INTERPEN_THRESH", "FLANGE_TO_PART_BOTTOM", "SEATED_TOOL_QUAT_LOCAL",
)
FLOOR_STEPS_MM = (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0)
LATERAL_X_MM = (0.0, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0)
LATERAL_Y_MM = (0.0, 0.4, 0.8, 1.2, 2.0, 3.0)

PoseRow = collections.namedtuple("PoseRow", "direction offset_mm depth_m dx_m dy_m expected_m")
PoseVerdict = collections.namedtuple("PoseVerdict", "q1 q2 q3 q1_bad q2_bad n_match n_rows")


def read_task_constants(text: str | None = None) -> dict:
    """Fold the plain module-level assignments of insertion_tasks_cfg.py."""
    import ast

    tree = ast.parse(text if text is not None else _TASKS_SRC.read_text(encoding="utf-8"))
    ns: dict = {}
    ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}

    def ev(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            return ns[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Tuple):
            return tuple(ev(e) for e in node.elts)
        if isinstance(node, ast.Subscript):
            return ev(node.value)[ev(node.slice)]
        raise ValueError(type(node).__name__)

    for node in tree.body:
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, value = node.target.id, node.value
        if target is None:
            continue
        try:
            ns[target] = ev(value)
        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError):
            ns.pop(target, None)
    missing = [n for n in POSE_LADDER_CONSTANTS if n not in ns]
    if missing:
        raise RuntimeError(f"insertion_tasks_cfg.py: could not fold {missing}")
    return {n: ns[n] for n in POSE_LADDER_CONSTANTS}


def pose_ladder_rows(c: dict) -> list:
    """The ladder's poses and the reading a PLANAR wall would give. Pure."""
    rows = []
    seat = float(c["POCKET_SEAT_DEPTH"])
    band = float(c["SEATED_SUCCESS_DEPTH"])
    for k in FLOOR_STEPS_MM:
        rows.append(PoseRow("floor", k, seat + k / 1000.0, 0.0, 0.0, max(0.0, k / 1000.0)))
    for axis, steps, half in (("x", LATERAL_X_MM, 0.5 * float(c["PLAY_X"])),
                              ("y", LATERAL_Y_MM, 0.5 * float(c["PLAY_Y"]))):
        for sgn, name in ((1.0, axis + "+"), (-1.0, axis + "-")):
            for s in steps:
                d = sgn * s / 1000.0
                rows.append(PoseRow(name, s, band, d if axis == "x" else 0.0,
                                    d if axis == "y" else 0.0, max(0.0, s / 1000.0 - half)))
    return rows


def judge_pose_ladder(rows, readings, far_reading: float, far_disjoint: bool, tol: float) -> PoseVerdict:
    """Q1/Q2/Q3 over the measured ladder, plus the geometry match count. Pure."""
    by_dir: dict = collections.OrderedDict()
    for r, v in zip(rows, readings):
        by_dir.setdefault(r.direction, []).append((r.offset_mm, float(v)))
    q1_bad, q2_bad = [], []
    for name, pts in by_dir.items():
        pts.sort()
        for (o0, v0), (o1, v1) in zip(pts, pts[1:]):
            if v1 < v0 - tol:
                q1_bad.append((name, o0, v0, o1, v1))
        if not pts[-1][1] > tol:
            q2_bad.append((name, pts[-1][0], pts[-1][1]))
    q3 = bool(far_disjoint) and float(far_reading) == 0.0
    n_match = sum(1 for r, v in zip(rows, readings) if abs(float(v) - r.expected_m) <= tol)
    return PoseVerdict(not q1_bad, not q2_bad, q3, q1_bad, q2_bad, n_match, len(rows))


_CASE_C = {"PLAY_X": 0.0005876, "PLAY_Y": 0.0016, "POCKET_SEAT_DEPTH": 0.036,
           "SEATED_SUCCESS_DEPTH": 0.033}


def run_pose_cases(torch=None, sdf=None) -> int:
    """Expectation arithmetic, the verdict rule and the constant fold. Returns mismatches."""
    bad = 0
    rows = pose_ladder_rows(_CASE_C)
    ideal = [r.expected_m for r in rows]

    def row(direction, offset):
        return next(r for r in rows if r.direction == direction and r.offset_mm == offset)

    def case(name, ok):
        nonlocal bad
        print(f"[check_insertion_sdf]   pose {name}: {'PASS' if ok else 'FAIL'}")
        bad += 0 if ok else 1

    case("PR1 floor +0.5 mm expects 0.5 mm, floor -1.0 mm expects 0",
         abs(row("floor", 0.5).expected_m - 0.0005) < 1e-12 and row("floor", -1.0).expected_m == 0.0)
    case("PR2 x- 0.6 mm expects 0.6 - 0.2938 = 0.3062 mm, at dx = -0.6 mm, depth 33 mm",
         abs(row("x-", 0.6).expected_m - 0.0003062) < 1e-12 and row("x-", 0.6).dx_m == -0.0006
         and row("x-", 0.6).depth_m == 0.033)
    case("PR3 y+ 1.2 mm expects 0.4 mm, y+ 0.8 mm expects 0",
         abs(row("y+", 1.2).expected_m - 0.0004) < 1e-12 and row("y+", 0.8).expected_m == 0.0)
    v = judge_pose_ladder(rows, ideal, 0.0, True, ON_SURFACE_TOL_M)
    case("PC1 the planar-wall table passes Q1/Q2/Q3 and matches every row",
         (v.q1, v.q2, v.q3) == (True, True, True) and v.n_match == v.n_rows)
    dip = list(ideal)
    i = rows.index(row("x+", 1.0))
    dip[i] = 0.0
    v = judge_pose_ladder(rows, dip, 0.0, True, ON_SURFACE_TOL_M)
    case("PC2 a reading that falls as the offset grows fails Q1 only", (v.q1, v.q2, v.q3) == (False, True, True))
    v = judge_pose_ladder(rows, [0.0] * len(rows), 0.0, True, ON_SURFACE_TOL_M)
    case("PC3 zero everywhere (overlap never seen) fails Q2 only", (v.q1, v.q2, v.q3) == (True, False, True))
    v = judge_pose_ladder(rows, ideal, 1.2, True, ON_SURFACE_TOL_M)
    case("PC4 a far row that reads non-zero (flipped sign) fails Q3 only", (v.q1, v.q2, v.q3) == (True, True, False))
    v = judge_pose_ladder(rows, ideal, 0.0, False, ON_SURFACE_TOL_M)
    case("PC5 a far row that is not disjoint fails Q3 -- it proves nothing", (v.q1, v.q2, v.q3) == (True, True, False))
    c = read_task_constants()
    case("PK1 the folded PLAY_X is the cross play and INTERPEN_THRESH is half of it",
         abs(c["PLAY_X"] - PLAY_X_M) < 1e-12 and abs(c["INTERPEN_THRESH"] - 0.5 * c["PLAY_X"]) < 1e-15)
    case("PK2 the folded seated tool quaternion is the 180 deg turn about y",
         tuple(c["SEATED_TOOL_QUAT_LOCAL"]) == (0.0, 0.0, 1.0, 0.0))
    case("PK3 a name whose right-hand side cannot be folded is refused, not guessed",
         _refuses("PLAY_X = f(1)\n"))
    ok4 = False
    if torch is not None and sdf is not None:
        try:
            im = sys.modules["_insertion_pkg.insertion_math"]
            kc = dict(_CASE_C, FLANGE_TO_PART_BOTTOM=0.152, SEATED_TOOL_QUAT_LOCAL=(0.0, 0.0, 1.0, 0.0))
            rp, rq = ladder_rel_poses(torch, im, sdf, kc, [(0.033, 0.0006, 0.0), (0.036, 0.0, -0.0008)])
            import numpy as np
            rp = np.asarray(rp.tolist(), dtype=np.float64)
            rq = np.asarray(rq.tolist(), dtype=np.float64)
            want_p = np.array([[0.0006, 0.0, 0.119], [0.0, -0.0008, 0.116]])
            want_q = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            ok4 = (rp.shape == (2, 3) and np.abs(rp - want_p).max() < 1e-6
                   and np.abs(np.abs(rq) - want_q).max() < 1e-6)
        except Exception as exc:
            print(f"[check_insertion_sdf]   pose PK4 raised {type(exc).__name__}: {exc}")
    case("PK4 the pose chain with the env's shapes lands the tool_link at (dx, dy, 0.152 - depth)", ok4)
    return bad


def _refuses(text: str) -> bool:
    try:
        read_task_constants(text)
    except RuntimeError:
        return True
    return False


POSE_CASES_COUNT = 12


def ladder_rel_poses(torch, im, sdf, c: dict, poses) -> tuple:
    """(depth, dx, dy) poses -> SAPU ``(rel_pos, rel_quat_xyzw)``, the env's chain."""
    seat_q = torch.tensor([list(c["SEATED_TOOL_QUAT_LOCAL"])])
    tip_off = torch.tensor([0.0, 0.0, float(c["FLANGE_TO_PART_BOTTOM"])])
    ident_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    zero_p = torch.tensor([[0.0, 0.0, 0.0]])
    rel_p, rel_q = [], []
    for depth, dx, dy in poses:
        entrance = torch.tensor([[float(dx), float(dy), 0.0]])
        goal_pos, goal_quat = im.seated_goal_pose(entrance, ident_q, seat_q, tip_off, float(depth))
        p, q = sdf.goal_relative_transform(goal_pos, goal_quat, zero_p, ident_q)
        rel_p.append(p)
        rel_q.append(q)
    return torch.cat(rel_p, dim=0), torch.cat(rel_q, dim=0)


def _rotate_xyzw(q, v):
    """Rotate (N, 3) points by one xyzw quaternion. numpy."""
    import numpy as np

    u = np.asarray(q[:3], dtype=np.float64)
    w = float(q[3])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def run_pose_ladder(args) -> int:
    import numpy as np
    import torch

    sdf = load_sdf_module()
    im = sys.modules["_insertion_pkg.insertion_math"]
    paths = load_paths_module()
    part_obj = args.obj if args.obj is not None else paths.resolve_part_obj_path()
    fixture_obj = args.query_obj if args.query_obj is not None else paths.resolve_pocket_obj_path()
    c = read_task_constants()
    tol = float(args.on_surface_tol_m)

    print(f"[check_insertion_sdf] marker: {SCRIPT_MARKER}")
    print("[check_insertion_sdf] MODE: POSE LADDER -- part sample points against the FIXTURE "
          "mesh at known separated and overlapping poses")
    print(f"[check_insertion_sdf] sampled obj: {part_obj}")
    print(f"[check_insertion_sdf] queried obj: {fixture_obj}")
    print("[check_insertion_sdf] constants folded from insertion_tasks_cfg.py: "
          + ", ".join(f"{k}={c[k]}" for k in POSE_LADDER_CONSTANTS))
    query = sdf.SdfDistanceQuery(
        obj_path=part_obj,
        num_sample_points=args.num_sample_points,
        max_dist=MAX_DIST_M,
        device=args.device,
        seed=args.seed,
        mesh_obj_path=fixture_obj,
    )
    print(query.describe())
    if args.mutate_sign:
        print("[check_insertion_sdf] *** MUTATION ACTIVE: the signed distance is negated before "
              "the clamp. Q3 MUST fail, or this ladder proves nothing.")

    rows = pose_ladder_rows(c)
    mesh_hi_z = float(query.mesh_bounds[1][2])
    far_depth = -(mesh_hi_z + float(query.extents[2]) + float(query.extents[0]))
    poses = [(r.depth_m, r.dx_m, r.dy_m) for r in rows] + [(far_depth, 0.0, 0.0)]

    rel_p, rel_q = ladder_rel_poses(torch, im, sdf, c, poses)

    signed = query.signed_distances(rel_p, rel_q)
    if args.mutate_sign:
        signed = -signed
    reading = im.max_interpen_dist(im.interpen_from_signed(signed)).cpu().numpy().astype(np.float64)

    pts = np.asarray(query.points.numpy(), dtype=np.float64)
    rp = rel_p.cpu().numpy().astype(np.float64)
    rq = rel_q.cpu().numpy().astype(np.float64)
    low_z = [float((_rotate_xyzw(rq[i], pts) + rp[i])[:, 2].min()) for i in range(len(poses))]
    far_disjoint = low_z[-1] > mesh_hi_z

    print("[check_insertion_sdf] dir    offset[mm]  depth[mm]   dx[mm]   dy[mm]  lowest z[mm]  "
          "reading[mm]  planar-wall[mm]  row")
    for i, r in enumerate(rows):
        match = abs(reading[i] - r.expected_m) <= tol
        print(f"[check_insertion_sdf]  {r.direction:5s} {r.offset_mm:9.2f} {r.depth_m * 1e3:10.3f} "
              f"{r.dx_m * 1e3:8.3f} {r.dy_m * 1e3:8.3f} {low_z[i] * 1e3:12.4f} "
              f"{reading[i] * 1e3:11.4f} {r.expected_m * 1e3:15.4f}  {'match' if match else 'DEVIATION'}")
    print(f"[check_insertion_sdf]  far   depth {far_depth * 1e3:.3f} mm, lowest z {low_z[-1] * 1e3:.4f} mm "
          f"vs fixture top {mesh_hi_z * 1e3:.4f} mm, disjoint {far_disjoint}, reading {reading[-1] * 1e3:.6f} mm")

    for axis, steps in (("x", LATERAL_X_MM), ("y", LATERAL_Y_MM)):
        for s in steps:
            ip = rows.index(next(r for r in rows if r.direction == axis + "+" and r.offset_mm == s))
            im_ = rows.index(next(r for r in rows if r.direction == axis + "-" and r.offset_mm == s))
            if reading[ip] > tol and reading[im_] > tol:
                print(f"[check_insertion_sdf] {axis} centre read at |s| = {s} mm: "
                      f"{(reading[ip] - reading[im_]) * 1e3 / 2.0:+.4f} mm (+{axis} side minus -{axis} side, halved)")

    v = judge_pose_ladder(rows, reading[:-1].tolist(), float(reading[-1]), far_disjoint, tol)
    print(f"[check_insertion_sdf] Q1 the reading never falls as the offset grows: "
          f"{'PASS' if v.q1 else 'FAIL -- ' + repr(v.q1_bad)}")
    print(f"[check_insertion_sdf] Q2 the largest offset of every direction reads an overlap: "
          f"{'PASS' if v.q2 else 'FAIL -- ' + repr(v.q2_bad)}")
    print(f"[check_insertion_sdf] Q3 the far row is disjoint by measurement and reads exactly 0: "
          f"{'PASS' if v.q3 else 'FAIL'}")
    print(f"[check_insertion_sdf] GEOMETRY (reported, not judged): {v.n_match} of {v.n_rows} rows "
          f"match the planar-wall reading within {tol * 1e3:.3f} mm; a DEVIATION row is a finding")
    if args.mutate_sign:
        caught = not v.q3
        print(f"[check_insertion_sdf] MUTATION VERDICT: "
              f"{'PASS -- the ladder catches a flipped sign' if caught else 'FAIL -- the ladder survived its own mutation'}")
        return 0 if caught else 1
    ok = v.q1 and v.q2 and v.q3
    print(f"[check_insertion_sdf] VERDICT: {'READING_PLAUSIBLE' if ok else 'READING_SUSPECT'}")
    return 0 if ok else 1


def run_probe(args) -> int:
    import numpy as np
    import torch

    sdf = load_sdf_module()
    obj_path = args.obj if args.obj is not None else load_paths_module().resolve_part_obj_path()
    query_obj = args.query_obj
    two_meshes = query_obj is not None and query_obj != obj_path

    print(f"[check_insertion_sdf] marker: {SCRIPT_MARKER}")
    print(f"[check_insertion_sdf] sampled obj: {obj_path}")
    print(f"[check_insertion_sdf] queried obj: {query_obj if two_meshes else '(the same file)'}")
    if two_meshes:
        print("[check_insertion_sdf] MODE: SAPU -- the PART's sample points against the "
              "FIXTURE mesh, the configuration insertion_env will build. The queried "
              "mesh's counts are PRINTED, not gated: their identity against RT-86 is "
              "check_fixture_mesh.py's G1 and keeps its one home there.")

    query = sdf.SdfDistanceQuery(
        obj_path=obj_path,
        num_sample_points=args.num_sample_points,
        max_dist=MAX_DIST_M,
        device=args.device,
        seed=args.seed,
        mesh_obj_path=query_obj,
    )
    print(query.describe())
    if args.mutate_sign:
        print("[check_insertion_sdf] *** MUTATION ACTIVE: every distance is negated. "
              "P2 and P5 MUST fail and P4 MUST stay green, or this probe proves nothing.")

    sample_extents = np.asarray(query.extents, dtype=np.float64)
    mesh_extents = np.asarray(query.mesh_extents, dtype=np.float64)

    step = mesh_extents + (sample_extents if two_meshes else 0.0)

    diameter = float(np.linalg.norm(sample_extents))
    if two_meshes:
        diameter += float(np.linalg.norm(mesh_extents))
    print(f"[check_insertion_sdf] sampled extents {sample_extents[0]:.6f} x "
          f"{sample_extents[1]:.6f} x {sample_extents[2]:.6f} m")
    print(f"[check_insertion_sdf] queried extents {mesh_extents[0]:.6f} x "
          f"{mesh_extents[1]:.6f} x {mesh_extents[2]:.6f} m, "
          f"bounds {query.mesh_bounds[0]} .. {query.mesh_bounds[1]} m")
    print(f"[check_insertion_sdf] ladder step per axis {step[0]:.6f} x {step[1]:.6f} "
          f"x {step[2]:.6f} m, P5 slack {diameter:.6f} m")

    pts_np = np.asarray(query.points.numpy(), dtype=np.float64)
    pts_lo, pts_hi = pts_np.min(axis=0), pts_np.max(axis=0)
    mesh_lo = np.asarray(query.mesh_bounds[0], dtype=np.float64)
    mesh_hi = np.asarray(query.mesh_bounds[1], dtype=np.float64)

    def boxes_disjoint(offset) -> bool:
        lo, hi = pts_lo + offset, pts_hi + offset
        return bool(np.any(lo > mesh_hi) or np.any(hi < mesh_lo))

    def distances(offsets: np.ndarray):
        pos = torch.tensor(offsets, dtype=torch.float32)
        quat = torch.zeros((offsets.shape[0], 4), dtype=torch.float32)
        quat[:, 3] = 1.0
        d = query.signed_distances(pos, quat)
        return -d if args.mutate_sign else d

    print(f"[check_insertion_sdf] launching first query: 1 env x "
          f"{query.num_points} points on {query.wp_device!s} "
          f"(torch side: {query.torch_device!r}) ...")
    d0 = distances(np.zeros((1, 3))).cpu().numpy()
    print("[check_insertion_sdf] first query returned.")

    if two_meshes:
        p4 = True
        print("[check_insertion_sdf] P4 identity transform: SKIPPED (sampled mesh != "
              "queried mesh -- the sample points are not on the queried surface, so "
              "'~0 at the identity' is not a claim about this pair)")
        print(f"[check_insertion_sdf] identity-transform distances, REPORTED not judged: "
              f"min {d0.min():.6f} m, mean {d0.mean():.6f} m, max {d0.max():.6f} m, "
              f"negative points {int((d0 < 0.0).sum())} of {d0.size}")
    else:
        _iv = judge_identity(np.abs(d0).ravel().tolist(), args.on_surface_tol_m)
        p4 = _iv.p4
        print(f"[check_insertion_sdf] P4 identity transform: p99 |d| = "
              f"{_iv.p99:.3e} m against tol {args.on_surface_tol_m:g} m -- "
              f"{'PASS' if p4 else 'FAIL'}  "
              f"(max |d| = {_iv.dmax:.3e} m, {_iv.n_above} of {_iv.n_points} "
              f"points above tol; p99 judges because this pair feeds the "
              f"AVERAGING SDF reward -- see judge_identity)")

    if not two_meshes:
        _a0 = np.abs(d0).ravel()
        _worst = int(np.argmax(_a0))
        _pt = pts_np[_worst % query.num_points]
        _vals = _a0.tolist()
        print(f"[check_insertion_sdf] P4 spread over {_a0.size} points: "
              f"median {percentile(_vals, 50.0):.3e} m, "
              f"p99 {_iv.p99:.3e} m, "
              f"max {_iv.dmax:.3e} m, "
              f"points above tol: {_iv.n_above}")
        print(f"[check_insertion_sdf] P4 worst point: index {_worst}, "
              f"|d| = {float(_a0[_worst]):.3e} m, "
              f"xyz = ({_pt[0]:.6f}, {_pt[1]:.6f}, {_pt[2]:.6f}) m")

    rows, offsets, labels = [], [], []
    for axis in range(3):
        for mult in PROBE_MULTIPLES:
            for sgn in (+1.0, -1.0):
                v = np.zeros(3)
                v[axis] = sgn * mult * step[axis]
                offsets.append(v)
                labels.append((axis, sgn, mult, float(abs(v[axis])), boxes_disjoint(v)))
    d = distances(np.asarray(offsets)).cpu().numpy()

    print("[check_insertion_sdf] axis  mult   offset[m]    min d[m]      mean d[m]   "
          " max d[m]  disjoint  asserted")
    for k, (axis, sgn, mult, dist, disjoint) in enumerate(labels):
        row = d[k]
        asserted = mult >= PROBE_ASSERT_FROM and disjoint
        name = f"{'xyz'[axis]}{'+' if sgn > 0 else '-'}"
        print(f"[check_insertion_sdf]  {name}  {mult:4.1f}  {dist:10.6f}  "
              f"{row.min():12.6f}  {row.mean():11.6f}  {row.max():10.6f}   "
              f"{'yes' if disjoint else ' no':>4}      "
              f"{'yes' if asserted else 'no'}")
        rows.append(LadderRow(name, mult, dist, float(row.min()),
                              float(row.mean()), asserted))

    v = judge_ladder(rows, diameter)

    if v.n_asserted == 0:
        print("[check_insertion_sdf] *** NO ROW IS ASSERTED: no offset put the sampled "
              "cloud provably outside the queried mesh, so P2 and P5 are vacuous. "
              "This is a FAILURE of the probe, not a pass. ***")
    print(f"[check_insertion_sdf] rows asserted: {v.n_asserted} of {len(labels)} "
          f"(mult >= {PROBE_ASSERT_FROM} AND bounding boxes disjoint)")

    p2 = v.p2
    print(f"[check_insertion_sdf] P2 sign holds where every point is outside by "
          f"construction: {'PASS' if p2 else 'FAIL -- ' + repr(v.p2_bad)}")

    p3 = v.p3
    print(f"[check_insertion_sdf] P3 mean distance rises over the ASSERTED rows "
          f"({v.n_p3_pairs} consecutive pairs; rows before the cloud clears the "
          f"queried mesh are printed, not judged): "
          f"{'PASS' if p3 else 'FAIL -- ' + repr(v.p3_bad)}")
    if v.n_p3_pairs == 0:
        print("[check_insertion_sdf] *** P3 HAS NO PAIR TO COMPARE: fewer than two "
              "asserted rows in every direction, so the rule is vacuous. This is a "
              "FAILURE of the probe, not a pass. ***")

    p5 = v.p5
    print(f"[check_insertion_sdf] P5 mean distance within one bbox diagonal of the "
          f"translation: {'PASS' if p5 else 'FAIL -- ' + repr(v.p5_bad)}")

    all_ok = p2 and p3 and p4 and p5
    if args.mutate_sign:
        caught = (not p2) and (not p5) and p4
        print(f"[check_insertion_sdf] MUTATION VERDICT: "
              f"{'PASS -- the probe catches a flipped sign' if caught else 'FAIL -- the probe survived its own mutation and proves nothing'}")
        return 0 if caught else 1

    print(f"[check_insertion_sdf] VERDICT: {'SIGN_HOLDS' if all_ok else 'SIGN_SUSPECT'}")
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true",
                        help="Offline: the torch half plus its mutations. No GPU.")
    parser.add_argument("--obj", type=str, default=None,
                        help="Part OBJ, the mesh that is SAMPLED. Default: "
                             "insertion_paths.resolve_part_obj_path(), which needs no Isaac.")
    parser.add_argument("--query-obj", type=str, default=None,
                        help="Mesh that is QUERIED. Default: the sampled one, which is the "
                             "SDF reward term. Point it at the fixture OBJ for the SAPU "
                             "configuration (D-109 (10)): the part's points against the "
                             "fixture's surface. P4 is then skipped and the ladder steps "
                             "along the fixture.")
    parser.add_argument("--num-sample-points", type=int, default=NUM_SAMPLE_POINTS,
                        help="Surface samples. Isaac Lab's default is 1000, for another "
                             "task's env count; stated, never silent.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the surface sampling. Always set (project rule).")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' asks Warp for the preferred device.")
    parser.add_argument("--on-surface-tol-m", type=float, default=ON_SURFACE_TOL_M,
                        help="P4 tolerance: how far a sampled point may read off its "
                             "own surface at the identity transform. Default is the "
                             "success-depth uncertainty, see ON_SURFACE_TOL_M.")
    parser.add_argument("--mutate-sign", action="store_true",
                        help="D-080 counter-proof: negate every distance. P2 and P5 "
                             "must fail, P4 must not.")
    parser.add_argument("--pose-ladder", action="store_true",
                        help="Training PC: the part against the FIXTURE at known separated and "
                             "overlapping poses (floor, x, y, far). Default meshes from "
                             "insertion_paths; --obj / --query-obj override them.")
    parser.add_argument("--bench", action="store_true",
                        help="MEASUREMENT mode, no verdict: time signed_distances over a "
                             "ladder of sample counts at our env count. Supplies the cost "
                             "column D-122 says its decision needs.")
    parser.add_argument("--bench-points", type=str, default=None,
                        help="Comma-separated sample counts for --bench. Default: "
                             f"{','.join(str(v) for v in BENCH_POINTS)} -- the last rung is "
                             "roughly where the spacing meets the cross play.")
    parser.add_argument("--bench-envs", type=int, default=BENCH_ENVS,
                        help="Envs per timed call. Default is the scene's own count, "
                             "insertion_env_cfg.py:456.")
    parser.add_argument("--bench-warmup", type=int, default=3,
                        help="Untimed calls before the timed ones. The first call compiles "
                             "the kernel and allocates; a training step never pays that.")
    parser.add_argument("--bench-repeats", type=int, default=20,
                        help="Timed calls per rung; the reported ms/call is their mean.")
    args = parser.parse_args()

    if args.self_test:
        spec = importlib.util.spec_from_file_location(
            "torch_shim", _HERE / "tools" / "torch_shim.py"
        )
        shim = importlib.util.module_from_spec(spec)
        sys.modules["torch_shim"] = shim
        spec.loader.exec_module(shim)
        torch, _ = shim.load()
        sys.modules.setdefault("torch", torch)
        return run_self_test(torch)

    if args.pose_ladder:
        return run_pose_ladder(args)

    if args.bench:
        return run_bench(args)

    return run_probe(args)


if __name__ == "__main__":
    sys.exit(main())
