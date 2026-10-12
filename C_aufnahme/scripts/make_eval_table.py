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

TWO KINDS
---------
``rows``      N rows x 16 columns, uniform in [0, 1): ``default_rng(seed)``.
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

The layout (columns, boundaries, sides) is READ from ``autodr.py`` through
``check_autodr.load_autodr`` -- nothing of it is typed here.

NEVER OVERWRITES. An existing file with the same bytes reports ``identical``
(that is also the way to ask "does this numpy still reproduce the table?");
an existing file with other bytes is refused with a non-zero exit. To redraw
a table, delete it by hand -- and know that its hash is the test's identity.

    python scripts/make_eval_table.py --kind rows --seed 10
    python scripts/make_eval_table.py --kind boundary --seed 10
    python scripts/make_eval_table.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
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
# Second entry of the seed sequence for the boundary table. numpy documents a
# sequence of ints as a seed (SeedSequence entropy); [seed, 1] is a stream
# independent of default_rng(seed).
BOUNDARY_STREAM = 1


class TableError(RuntimeError):
    """A table that cannot be written as asked. Never a silent fallback."""


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


def read_back_problems(path: pathlib.Path, unit: np.ndarray, data: bytes, ns: dict) -> list:
    """The file through the env's OWN reader against what was drawn. Equality, not non-emptiness."""
    table = ns["RowTable"].from_csv(path)
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
        name = f"eval_table_seed{seed}"
        meta = {"kind": kind, "seed": seed, "rng": f"numpy.random.default_rng({seed})"}
    else:
        unit = draw_boundary(seed, rows_per_block, ns)
        problems = boundary_problems(unit, rows_per_block, ns)
        if problems:
            raise TableError("not a boundary table: " + "; ".join(problems))
        name = f"eval_table_boundary_seed{seed}"
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
    problems = read_back_problems(csv_path, unit, data, ns)
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
        # MUTATION (D-080, eval_plan.md row A): two header columns swapped.
        head = list(cols)
        head[2], head[3] = head[3], head[2]
        swapped = tmp / "swapped.csv"
        swapped.write_bytes(a.replace(",".join(cols).encode(), ",".join(head).encode(), 1))
        other = table_bytes(draw_rows(20, 50, ns), ns)
        # MUTATION: a boundary table whose nailing was skipped.
        not_nailed = np.random.default_rng([10, BOUNDARY_STREAM]).random(unit_b.shape)
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
        ]
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"[make_eval_table --self-test] {len(checks) - len(failed)}/{len(checks)}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--kind", choices=("rows", "boundary"))
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
