# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Draw an evaluation table ONCE, on the laptop, and never overwrite it.

WHY (`eval_plan.md` step A, inbox 2026-09-21 "the table is committed"). The
table IS the identity of the test cases: every policy plays row ``i`` in env
``i``. Whether numpy ``default_rng`` returns the same values on two numpy
versions is UNKNOWN, so the training PC never redraws -- it reads the
committed file, and ``RowTable.from_csv`` hashes its RAW BYTES
(``autodr.py``, ``from_csv``). Hence: LF line endings written as bytes, a
``-text`` line in ``.gitattributes``, and ``repr(float)`` cells so the text
round-trips to the same float64.

THREE KINDS
-----------
``rows``      N rows x ``len(autodr.TABLE_COLUMNS)`` columns, uniform in
              [0, 1): ``default_rng(seed)``.
``boundary``  one block per AutoDR boundary (seven for ``autodr.DR_DIMS``),
              ``--rows-per-block`` rows each. Block ``b`` has the column of
              boundary ``b`` nailed with ``autodr.apply_boundary`` (0.0 = lower
              edge, 1.0 = upper edge); every other column stays drawn. Block of
              a row = ``row_id // rows_per_block``; the key order is
              ``autodr.boundary_keys(DR_DIMS)`` and the sidecar names it.
              The draw uses ``default_rng([seed, BOUNDARY_STREAM])``, NOT
              ``default_rng(seed)``: the latter's first rows ARE the ``rows``
              table of the same seed, and `eval_plan.md` K3 compares a block
              against that table as two INDEPENDENT rates.
``ood``       (D-206 (3); D-055/D-191, ported by hand from
              `origin/p6-TestAusserhalb` `019ae3d8`) NOT drawn: the committed
              ``boundary`` table of the same seed and width, with the nailed
              cell of every block moved OUTWARDS to ``OOD_FACTOR`` times the
              target distance from the centre (capped per key in
              ``OOD_FACTOR_CAP``). Every other cell is byte-for-byte the
              boundary table's, so an OOD block and its T-grenze block differ
              in ONE quantity only. The nailed unit values leave [0, 1]; the
              env reads them only behind ``play.py --eval-ood``.

The layout (columns, boundaries, sides) is READ from ``autodr.py`` through
``check_autodr.load_autodr`` -- nothing of it is typed here.

THE NAME CARRIES THE WIDTH: ``eval_table_seed<S>_c<columns>.csv``,
``eval_table_boundary_seed<S>_c<columns>.csv``,
``eval_table_ood_seed<S>_c<columns>.csv``. D-194 made the row 17 wide;
the 16-column tables (no suffix) stay as the old evaluation's identity and
``RowTable`` refuses them on this code (HANDOFF-RL 2a (b)(1)).

NEVER OVERWRITES. An existing file with the same bytes reports ``identical``
(that is also the way to ask "does this numpy still reproduce the table?");
an existing file with other bytes is refused with a non-zero exit. To redraw
a table, delete it by hand -- and know that its hash is the test's identity.

    python scripts/make_eval_table.py --kind rows --seed 10
    python scripts/make_eval_table.py --kind boundary --seed 10
    python scripts/make_eval_table.py --kind ood --seed 10
    python scripts/make_eval_table.py --self-test
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import pathlib
import sys
import tempfile

import numpy as np

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import check_autodr  # noqa: E402 - owns the one loader of autodr.py

DEFAULT_OUT_DIR = SCRIPTS / "eval"
DEFAULT_ROWS = 1000
DEFAULT_ROWS_PER_BLOCK = 150
BOUNDARY_STREAM = 1


class TableError(RuntimeError):
    """A table that cannot be written as asked. Never a silent fallback."""


OOD_FACTOR = 1.5
OOD_FACTOR_CAP = {"start_height_hi": 1.0}

_ENV_DIR = SCRIPTS.parent / "source" / "insertion" / "insertion" / "tasks" / "direct" / "insertion"


def _tasks_const(name: str) -> float:
    """A LITERAL module-level constant of insertion_tasks_cfg.py (that file imports isaaclab, so it is read, not imported)."""
    tree = ast.parse((_ENV_DIR / "insertion_tasks_cfg.py").read_text(encoding="utf-8"))
    for n in tree.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, (int, float))):
            return float(n.value.value)
    raise TableError(f"insertion_tasks_cfg.py has no literal {name}")


def _cfg_default_expr(field: str) -> ast.expr:
    tree = ast.parse((_ENV_DIR / "insertion_env_cfg.py").read_text(encoding="utf-8"))
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "InsertionEnvCfg":
            for n in cls.body:
                if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == field:
                    return n.value
    raise TableError(f"InsertionEnvCfg has no field {field}")


def _resolve_centre(expr: ast.expr) -> float:
    """The three expression shapes the env's table-centre dict holds; anything else is refused, never guessed."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
        return float(expr.value)
    if (isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name)
            and expr.value.id == "insertion_tasks_cfg"):
        return _tasks_const(expr.attr)
    if (isinstance(expr, ast.Call) and getattr(expr.func, "id", None) == "float" and len(expr.args) == 1
            and isinstance(expr.args[0], ast.Attribute) and getattr(expr.args[0].value, "id", None) == "cfg"):
        return _resolve_centre(_cfg_default_expr(expr.args[0].attr))
    raise TableError(f"table centre expression {ast.unparse(expr)!r} is not one this reader resolves")


def table_centres() -> dict:
    """The centres the env binds for dr_mode='table' -- read from ITS OWN `_table_dims = autodr.bind_centres({...})` (insertion_env.py), the one home."""
    tree = ast.parse((_ENV_DIR / "insertion_env.py").read_text(encoding="utf-8"))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
            and getattr(n.targets[0], "id", None) == "_table_dims"
            and isinstance(n.value, ast.Call) and getattr(n.value.func, "attr", None) == "bind_centres"]
    if len(hits) != 1:
        raise TableError(f"insertion_env.py holds {len(hits)} `_table_dims = autodr.bind_centres(...)`, want 1")
    d = hits[0].value.args[0]
    if not isinstance(d, ast.Dict):
        raise TableError("the table centres are not a dict literal")
    return {k.value: _resolve_centre(v) for k, v in zip(d.keys, d.values)}


def ood_step(dim, side: str, centre: float, factor: float) -> tuple:
    """``(unit, physical)`` of the nailed cell: the physical value sits ``factor`` times the target distance from the centre, on ``side``; the unit value is what the env's OWN maps turn into it (`disk_offset` for the radius, `map_unit_to_bounds` on the table band otherwise)."""
    if dim.name == "lat_r":
        return factor ** 2, factor * dim.hi_max
    if dim.one_sided:
        return factor, centre + factor * (dim.hi_max - centre)
    lo, hi = dim.lo_max, dim.hi_max
    p = (centre - lo) / (hi - lo)
    u = p + factor * (1.0 - p) if side == "hi" else p * (1.0 - factor)
    return u, lo + u * (hi - lo)


def ood_layout(ns: dict) -> list:
    """``[(key, column, unit, physical, factor), ...]`` in boundary order."""
    centres = table_centres()
    dims = {d.name: d for d in ns["DR_DIMS"]}
    out = []
    for key, col, _side in boundary_layout(ns):
        name, side = key.rsplit("_", 1)
        f = OOD_FACTOR_CAP.get(key, OOD_FACTOR)
        u, phys = ood_step(dims[name], side, centres[name], f)
        out.append((key, col, u, phys, f))
    return out


def ood_problems(unit: np.ndarray, base: np.ndarray, rows_per_block: int, ns: dict) -> list:
    """Every way ``unit`` is NOT the boundary table ``base`` with its nailed cells moved outwards."""
    layout = ood_layout(ns)
    if unit.shape != base.shape or unit.shape[0] != len(layout) * rows_per_block:
        return [f"shape {unit.shape} vs boundary {base.shape}, {len(layout)} blocks x {rows_per_block}"]
    problems = []
    for b, (key, col, u, _phys, f) in enumerate(layout):
        sl = slice(b * rows_per_block, (b + 1) * rows_per_block)
        if not np.all(unit[sl, col] == u):
            problems.append(f"block {b} ({key}): column {col} is not all {u}")
        if f > 1.0 and not (u < 0.0 or u > 1.0):
            problems.append(f"block {b} ({key}): nailed value {u} is not outside [0, 1]")
        other = np.ones(unit.shape[1], dtype=bool)
        other[col] = False
        if not np.array_equal(unit[sl][:, other], base[sl][:, other]):
            problems.append(f"block {b} ({key}): a cell other than column {col} differs from the boundary table")
    return problems


def boundary_layout(ns: dict) -> list:
    """``[(key, column, side), ...]`` in ``autodr.boundary_keys`` order."""
    out = []
    for d in ns["DR_DIMS"]:
        for side in d.sides:
            out.append((f"{d.name}_{side}", d.column, 1.0 if side == "hi" else 0.0))
    keys = tuple(k for k, _, _ in out)
    if keys != ns["boundary_keys"](ns["DR_DIMS"]):
        raise TableError(f"boundary order {keys} differs from autodr.boundary_keys")
    return out


def draw_rows(seed: int, n_rows: int, ns: dict) -> np.ndarray:
    if n_rows < 1:
        raise TableError(f"n_rows must be >= 1, got {n_rows}")
    return np.random.default_rng(seed).random((n_rows, len(ns["TABLE_COLUMNS"])))


def draw_boundary(seed: int, rows_per_block: int, ns: dict) -> np.ndarray:
    if rows_per_block < 1:
        raise TableError(f"rows_per_block must be >= 1, got {rows_per_block}")
    layout = boundary_layout(ns)
    n = len(layout) * rows_per_block
    unit = np.random.default_rng([seed, BOUNDARY_STREAM]).random(
        (n, len(ns["TABLE_COLUMNS"]))
    )
    rows = np.arange(n)
    block = rows // rows_per_block
    cols = np.array([c for _, c, _ in layout])[block]
    sides = np.array([s for _, _, s in layout])[block]
    return ns["apply_boundary"](unit, rows, cols, sides)


def boundary_problems(unit: np.ndarray, rows_per_block: int, ns: dict) -> list:
    """Every way ``unit`` is NOT a boundary table. Empty list = it is one."""
    layout = boundary_layout(ns)
    problems = []
    if unit.shape[0] != len(layout) * rows_per_block:
        return [f"{unit.shape[0]} rows, expected {len(layout)}*{rows_per_block}"]
    for b, (key, col, side) in enumerate(layout):
        blk = unit[b * rows_per_block:(b + 1) * rows_per_block]
        if not np.all(blk[:, col] == side):
            problems.append(f"block {b} ({key}): column {col} is not all {side}")
        for c in range(unit.shape[1]):
            if c != col and blk.shape[0] > 1 and np.all(blk[:, c] == blk[0, c]):
                problems.append(f"block {b} ({key}): column {c} is constant, not drawn")
    return problems


def table_bytes(unit: np.ndarray, ns: dict) -> bytes:
    """Header + rows, ``repr(float)`` cells, LF only, UTF-8."""
    lines = [",".join(ns["TABLE_COLUMNS"])]
    lines.extend(",".join(repr(float(v)) for v in row) for row in unit)
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_new(path: pathlib.Path, data: bytes) -> str:
    """``written`` or ``identical``; other bytes already there -> TableError."""
    if path.exists():
        if path.read_bytes() == data:
            return "identical"
        raise TableError(
            f"{path} exists with OTHER bytes -- refused. Its hash is the identity "
            "of the test cases; delete it by hand if a redraw is really meant."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:
        fh.write(data)
    return "written"


def read_back_problems(path: pathlib.Path, unit: np.ndarray, data: bytes, ns: dict,
                       allow_outside: bool = False) -> list:
    """The file through the env's OWN reader against what was drawn. Equality, not non-emptiness."""
    table = ns["RowTable"].from_csv(path, allow_outside=allow_outside)
    problems = []
    sha = hashlib.sha256(data).hexdigest()
    if table.sha256 != sha:
        problems.append(f"sha256 on disk {table.sha256} != drawn {sha}")
    if table.n_rows != unit.shape[0]:
        problems.append(f"{table.n_rows} rows on disk, {unit.shape[0]} drawn")
    elif not np.array_equal(np.array(table.rows), unit):
        problems.append("values on disk differ from the drawn float64 values")
    return problems


def make(kind: str, seed: int, n_rows: int, rows_per_block: int, out_dir: pathlib.Path,
         ns: dict) -> int:
    if kind == "rows":
        unit = draw_rows(seed, n_rows, ns)
        name = f"eval_table_seed{seed}_c{unit.shape[1]}"
        meta = {"kind": kind, "seed": seed, "rng": f"numpy.random.default_rng({seed})"}
    elif kind == "ood":
        bpath = out_dir / f"eval_table_boundary_seed{seed}_c{len(ns['TABLE_COLUMNS'])}.csv"
        bside = json.loads(bpath.with_suffix(".json").read_text(encoding="utf-8"))
        braw = bpath.read_bytes()
        if hashlib.sha256(braw).hexdigest() != bside.get("sha256"):
            raise TableError(f"{bpath} does not hash to its sidecar's sha256")
        rows_per_block = int(bside["rows_per_block"])
        base = np.array(ns["RowTable"].from_csv(bpath).rows, dtype=float)
        if boundary_problems(base, rows_per_block, ns):
            raise TableError(f"{bpath} is not a boundary table of this autodr.py")
        layout = ood_layout(ns)
        unit = base.copy()
        for b, (_k, col, u, _p, _f) in enumerate(layout):
            unit[b * rows_per_block:(b + 1) * rows_per_block, col] = u
        problems = ood_problems(unit, base, rows_per_block, ns)
        if problems:
            raise TableError("not an OOD table: " + "; ".join(problems))
        name = f"eval_table_ood_seed{seed}_c{unit.shape[1]}"
        meta = {
            "kind": kind, "seed": seed,
            "from": {"file": bpath.name, "sha256": bside["sha256"]},
            "rows_per_block": rows_per_block,
            "factor": OOD_FACTOR, "factor_cap": dict(OOD_FACTOR_CAP),
            "centres": table_centres(),
            "source": "D-206 (3), D-055/D-191; x1.5 from IndustReal (Tang et al. 2023) Table X; "
                      "cap: OSC z box, start inside the box (make_eval_table.py OOD_FACTOR_CAP)",
            "blocks": [{"block": b, "key": k, "column": c, "nailed_to": u, "factor": f,
                        "physical": p}
                       for b, (k, c, u, p, f) in enumerate(layout)],
        }
    else:
        unit = draw_boundary(seed, rows_per_block, ns)
        problems = boundary_problems(unit, rows_per_block, ns)
        if problems:
            raise TableError("not a boundary table: " + "; ".join(problems))
        name = f"eval_table_boundary_seed{seed}_c{unit.shape[1]}"
        meta = {
            "kind": kind, "seed": seed,
            "rng": f"numpy.random.default_rng([{seed}, {BOUNDARY_STREAM}])",
            "rows_per_block": rows_per_block,
            "blocks": [{"block": b, "key": k, "column": c, "nailed_to": s}
                       for b, (k, c, s) in enumerate(boundary_layout(ns))],
        }
    data = table_bytes(unit, ns)
    csv_path = out_dir / f"{name}.csv"
    status = write_new(csv_path, data)
    problems = read_back_problems(csv_path, unit, data, ns, allow_outside=(kind == "ood"))
    if problems:
        raise TableError(f"{csv_path}: " + "; ".join(problems))
    sha = hashlib.sha256(data).hexdigest()
    print(f"[make_eval_table] {status}: {csv_path}")
    print(f"[make_eval_table] rows {unit.shape[0]}  columns {unit.shape[1]}  sha256 {sha}")
    if status == "written":
        meta.update({"file": csv_path.name, "n_rows": int(unit.shape[0]),
                     "n_columns": int(unit.shape[1]), "sha256": sha,
                     "numpy": np.__version__})
        side = json.dumps(meta, indent=2).encode("utf-8") + b"\n"
        print(f"[make_eval_table] {write_new(out_dir / f'{name}.json', side)}: "
              f"{out_dir / f'{name}.json'}")
    return 0


def _refuses(fn) -> bool:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the NAME is what is tested
        return type(exc).__name__ in ("AutoDRError", "TableError")
    return False


def _self_test() -> int:
    ns = check_autodr.load_autodr()
    cols = ns["TABLE_COLUMNS"]
    a = table_bytes(draw_rows(10, 50, ns), ns)
    unit_b = draw_boundary(10, 6, ns)
    layout = boundary_layout(ns)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        good = tmp / "good.csv"
        first = write_new(good, a)
        table = ns["RowTable"].from_csv(good)
        head = list(cols)
        head[2], head[3] = head[3], head[2]
        swapped = tmp / "swapped.csv"
        swapped.write_bytes(a.replace(",".join(cols).encode(), ",".join(head).encode(), 1))
        other = table_bytes(draw_rows(20, 50, ns), ns)
        not_nailed = np.random.default_rng([10, BOUNDARY_STREAM]).random(unit_b.shape)
        made = tmp / "made"
        with contextlib.redirect_stdout(io.StringIO()):
            make("rows", 10, 50, 6, made, ns)
            make("boundary", 10, 50, 6, made, ns)
            make("ood", 10, 50, 6, made, ns)
        width = {p.name: len(p.read_bytes().split(b"\n", 1)[0].split(b","))
                 for p in made.glob("*.csv")}
        w = next(iter(width.values()), None)
        made_ood = json.loads((made / f"eval_table_ood_seed10_c{w}.json").read_text(encoding="utf-8"))
        made_bnd = json.loads((made / f"eval_table_boundary_seed10_c{w}.json").read_text(encoding="utf-8"))
        checks = [
            ("same seed, same bytes", a == table_bytes(draw_rows(10, 50, ns), ns)),
            ("another seed, other bytes", a != other),
            ("LF only, no CR anywhere", b"\r" not in a and a.endswith(b"\n")),
            ("header is TABLE_COLUMNS", a.split(b"\n", 1)[0].decode() == ",".join(cols)),
            ("first write says written", first == "written"),
            ("RowTable.from_csv takes the file", table.n_rows == 50),
            ("read-back equals the draw, hash and values",
             read_back_problems(good, draw_rows(10, 50, ns), a, ns) == []),
            ("read-back notices other values",
             read_back_problems(good, draw_rows(20, 50, ns), a, ns) != []),
            ("MUTATION swapped header: from_csv refuses",
             _refuses(lambda: ns["RowTable"].from_csv(swapped))),
            ("same bytes again says identical", write_new(good, a) == "identical"),
            ("other bytes onto an existing file are refused",
             _refuses(lambda: write_new(good, other))),
            ("the refused write left the file alone", good.read_bytes() == a),
            ("seven blocks in autodr.boundary_keys order",
             tuple(k for k, _, _ in layout) == ns["boundary_keys"](ns["DR_DIMS"])
             and len(layout) == 7),
            ("boundary table passes its own verifier", boundary_problems(unit_b, 6, ns) == []),
            ("MUTATION nailing skipped: the verifier names every block",
             len(boundary_problems(not_nailed, 6, ns)) == len(layout)),
            ("wrong row count is named", boundary_problems(unit_b[:-1], 6, ns) != []),
            ("block 0 nails lat_r to 1.0 and leaves yaw drawn",
             layout[0][0] == "lat_r_hi" and np.all(unit_b[:6, layout[0][1]] == 1.0)
             and not np.all(unit_b[:6, layout[1][1]] == unit_b[0, layout[1][1]])),
            ("a lower edge nails to 0.0",
             layout[1][0] == "yaw_lo" and np.all(unit_b[6:12, layout[1][1]] == 0.0)),
            ("boundary stream is NOT the rows stream of the same seed",
             not np.array_equal(not_nailed[:6], draw_rows(10, 6, ns))),
            ("boundary table is accepted by RowTable",
             ns["RowTable"](unit_b.tolist(), "x").n_rows == 6 * len(layout)),
            ("zero rows are refused", _refuses(lambda: draw_rows(10, 0, ns))),
            ("file names carry the width: _c<header columns>",
             set(width.values()) == {w}
             and sorted(width) == [f"eval_table_boundary_seed10_c{w}.csv",
                                   f"eval_table_ood_seed10_c{w}.csv",
                                   f"eval_table_seed10_c{w}.csv"]),
            ("each table's sidecar carries the same name",
             sorted(p.stem for p in made.glob("*.json")) == sorted(p[:-4] for p in width)),
            ("the OOD sidecar names its boundary table by file and hash",
             made_ood["from"] == {"file": f"eval_table_boundary_seed10_c{w}.csv",
                                  "sha256": made_bnd["sha256"]}),
        ]
        centres = table_centres()
        ood = {k: (u, p, f) for k, _c, u, p, f in ood_layout(ns)}
        rad = np.pi / 180.0
        want_u = {"lat_r_hi": 2.25, "yaw_lo": -0.25, "yaw_hi": 1.25, "tilt_hi": 1.5,
                  "start_height_hi": 1.0, "friction_lo": -0.25, "friction_hi": 1.25}
        want_p = {"lat_r_hi": 0.045, "yaw_lo": -7.5 * rad, "yaw_hi": 7.5 * rad, "tilt_hi": 15.0 * rad,
                  "start_height_hi": 0.270, "friction_lo": 0.05, "friction_hi": 0.23}
        base = unit_b
        ood_unit = base.copy()
        for b, (_k, col, u, _p, _f) in enumerate(ood_layout(ns)):
            ood_unit[b * 6:(b + 1) * 6, col] = u
        sh = [k for k, _c, _s in layout].index("start_height_hi")
        stray = ood_unit.copy()
        stray[0, layout[1][1]] += 0.01
        stuck = ood_unit.copy()
        stuck[0:6, layout[0][1]] = 1.0
        pushed = ood_unit.copy()
        pushed[sh * 6:(sh + 1) * 6, layout[sh][1]] = 1.2
        checks += [
            ("table centres read from the env's own bind: friction 0.14, start 0.020, others 0",
             centres == {"lat_r": 0.0, "yaw": 0.0, "tilt": 0.0, "start_height": 0.02, "friction": 0.14}),
            ("OOD unit values per key (x1.5; start_height capped at x1.0)",
             set(ood) == set(want_u) and all(abs(ood[k][0] - want_u[k]) < 1e-12 for k in want_u)),
            ("OOD physical values: 45 mm, +-7.5 deg, 15 deg, 0.270 m, 0.05 / 0.23",
             all(abs(ood[k][1] - want_p[k]) < 1e-12 for k in want_p)),
            ("OOD table from the boundary table passes its own verifier",
             ood_problems(ood_unit, base, 6, ns) == []),
            ("the capped start-height block IS its boundary block (the control)",
             np.array_equal(ood_unit[sh * 6:(sh + 1) * 6], base[sh * 6:(sh + 1) * 6])),
            ("MUTATION a cell outside the nailed column changed: named",
             any("differs from the boundary table" in p for p in ood_problems(stray, base, 6, ns))),
            ("MUTATION nailed cell left at the boundary: named",
             any("is not all" in p for p in ood_problems(stuck, base, 6, ns))),
            ("MUTATION start-height block pushed past its cap: named",
             any("start_height_hi" in p for p in ood_problems(pushed, base, 6, ns))),
            ("OOD rows: refused by the default RowTable, taken with allow_outside",
             _refuses(lambda: ns["RowTable"](ood_unit.tolist(), "x"))
             and ns["RowTable"](ood_unit.tolist(), "x", allow_outside=True).n_rows == 42),
        ]
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"[make_eval_table --self-test] {len(checks) - len(failed)}/{len(checks)}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--kind", choices=("rows", "boundary", "ood"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--rows-per-block", type=int, default=DEFAULT_ROWS_PER_BLOCK)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if args.kind is None or args.seed is None:
        ap.error("--kind and --seed are required (no default seed: a table is an identity)")
    ns = check_autodr.load_autodr()
    try:
        return make(args.kind, args.seed, args.rows, args.rows_per_block, args.out_dir, ns)
    except (TableError, ns["AutoDRError"]) as exc:
        print(f"[make_eval_table] REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
