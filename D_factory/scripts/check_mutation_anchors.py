"""Every mutation anchor in the two big check scripts, swept in one pass.

`check_env_wiring.py --self-test` and `check_insertion_math.py --self-test`
each re-run the whole check suite once per mutation, and a stale anchor is a
hard `SystemExit` on the FIRST one it reaches. That is minutes of waiting to
learn about a single typo, and it hides every stale anchor behind it.

This sweeps them all and lists them together. It applies no check and runs no
suite; it only asks the question `_read` asks -- does this anchor match the
real source EXACTLY once -- for every anchor in both tables.

Two things it does NOT do the naive way:

* Anchors inside ONE mutation are applied in sequence, each against the text
  the previous anchor already changed. That is what `_read` and `load_math`
  do, so a pristine-source count would report false stale AND miss real ones.
  A bad anchor stops that entry: what the text looks like after a mutation
  that did not apply is undefined, and the real code aborts there too.
* The key-to-file map is read out of `check_env_wiring.py` itself, from its
  `_read(<X>_SRC, <key>)` call sites, so re-pointing a key cannot silently
  leave this sweep checking the old file.

Laptop only, stdlib only. `--self-test` for the counter-proof (D-080).
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import pathlib
import sys
from typing import Callable, NamedTuple

SCRIPT_MARKER = "check_mutation_anchors 2026-09-14"
# The call sites that consume an anchor tuple against ONE `*_SRC` file.
# `_parse` is check_env_wiring's cached `ast.parse(_read(...))` (2026-09-14).
READ_FUNCS = ("_read", "_parse")

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO / "scripts"

ENV_WIRING = SCRIPTS_DIR / "check_env_wiring.py"
INSERTION_MATH = SCRIPTS_DIR / "check_insertion_math.py"

UNKNOWN_KEY = -1  # Finding.count when the key maps to no source file at all


class Finding(NamedTuple):
    """One anchor that does not match its source exactly once."""

    table: str
    name: str
    key: str
    path: str
    count: int
    anchor: str

    def line(self) -> str:
        head = (self.anchor.strip().splitlines() or [""])[0][:74]
        if self.count == UNKNOWN_KEY:
            return (f"  UNKNOWN KEY  [{self.table}] {self.name}: "
                    f"key {self.key!r} maps to no source")
        return (f"  STALE  [{self.table}] {self.name} ({self.key} -> {self.path}) "
                f"matches {self.count}x: {head}")


# --------------------------------------------------------------------------
# the pure core
# --------------------------------------------------------------------------

def sweep_pairs(pairs, text: str) -> tuple[int, int, str] | None:
    """Replay `_read`'s loop over one mutation's anchors against one file.

    Returns the FIRST anchor that does not match exactly once, as
    ``(index, count, anchor)``, or None when all of them do. Stops there
    because the text after a mutation that did not apply is undefined.
    """
    for index, (old, new) in enumerate(pairs):
        count = text.count(old)
        if count != 1:
            return (index, count, old)
        text = text.replace(old, new)
    return None


def sweep(entries, key_paths, read_text: Callable[[str], str]) -> list[Finding]:
    """Sweep normalised entries ``(table, name, key, pairs)``.

    ``key_paths`` maps a mutation key to the tuple of source paths that key is
    applied to -- more than one when a key feeds two files (`agent_mut` reaches
    both `zero_agent.py` and `random_agent.py`, and an anchor must be unique in
    each of them, because `_read` runs once per file).
    """
    findings: list[Finding] = []
    for table, name, key, pairs in entries:
        paths = key_paths.get(key)
        if not paths:
            findings.append(Finding(table, name, key, "", UNKNOWN_KEY, ""))
            continue
        for path in paths:
            hit = sweep_pairs(pairs, read_text(path))
            if hit is not None:
                _index, count, anchor = hit
                findings.append(Finding(table, name, key, path, count, anchor))
    return findings


# --------------------------------------------------------------------------
# reading the key-to-file map out of a check script
# --------------------------------------------------------------------------

def read_anchor_calls(tree: ast.AST) -> dict[str, set[str]]:
    """Argument name -> the `*_SRC` names it is `_read` against."""
    direct: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id in READ_FUNCS):
            continue
        if len(node.args) != 2:
            continue
        src_arg, mut_arg = node.args
        if not (isinstance(src_arg, ast.Name) and isinstance(mut_arg, ast.Name)):
            continue
        direct.setdefault(mut_arg.id, set()).add(src_arg.id)
    return direct


def resolve_keys(tree: ast.AST, entry_func: str = "build_checks",
                 suffix: str = "_mut") -> tuple[dict[str, list[str]], list[str]]:
    """Every `*_mut` parameter of `entry_func`, resolved to `*_SRC` names.

    Direct when the parameter reaches `_read` itself; otherwise one level of
    indirection (`build_checks` hands `math_mut` to `load_obs_slices`, and that
    helper's own parameter is what `_read` sees). A parameter that resolves to
    nothing is returned as unresolved -- it is a key this sweep cannot check,
    not something to guess a file for.
    """
    direct = read_anchor_calls(tree)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if entry_func not in funcs:
        raise SystemExit(f"no function {entry_func!r} to read mutation keys from")
    entry = funcs[entry_func]
    keys = [a.arg for a in list(entry.args.args) + list(entry.args.kwonlyargs)
            if a.arg.endswith(suffix)]

    resolved: dict[str, list[str]] = {}
    unresolved: list[str] = []
    for key in keys:
        hit = set(direct.get(key, ()))
        for node in ast.walk(entry):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            # `_read` is the destination, not a step on the way to it. Following
            # it as a helper would hand every key the file of whichever OTHER
            # caller happens to name its parameter the same thing.
            if node.func.id in READ_FUNCS:
                continue
            callee = funcs.get(node.func.id)
            if callee is None:
                continue
            for position, arg in enumerate(node.args):
                if not (isinstance(arg, ast.Name) and arg.id == key):
                    continue
                if position < len(callee.args.args):
                    hit |= direct.get(callee.args.args[position].arg, set())
        if hit:
            resolved[key] = sorted(hit)
        else:
            unresolved.append(key)
    return resolved, unresolved


def table_names(tree: ast.AST) -> set[str]:
    """Module-level names ending in `MUTATIONS`."""
    names: set[str] = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return {n for n in names if n.endswith("MUTATIONS")}


# --------------------------------------------------------------------------
# adapters for the two real check scripts
# --------------------------------------------------------------------------

def _is_local(module, root: pathlib.Path) -> bool:
    """True for a module living under `root`, or a stand-in object with no file.

    The discriminator for the cleanup below. A third-party extension module
    (numpy) carries a `__file__` far outside `root` and MUST stay cached: its
    compiled half refuses a second load in the same process. Anything a check
    script authors itself is either a file beside it (`tools/torch_shim.py`)
    or a bare object published under a borrowed name.
    """
    file = getattr(module, "__file__", None)
    if file is None:
        return True
    try:
        return root in pathlib.Path(file).resolve().parents
    except (OSError, ValueError):
        return False


def _load_module(path: pathlib.Path):
    """Import a check script and undo what it published under other names.

    Not politeness: `check_env_wiring.py` publishes its numpy stand-in under
    the name `torch`, and `check_insertion_math.py` then imports it, believes
    it found real PyTorch and calls `set_default_dtype` on a stand-in that has
    no such method. Loading both in one process needs that name gone again --
    but only the repo's own entries, never the third-party cache.
    """
    root = path.resolve().parent
    before = dict(sys.modules)
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    finally:
        for name in set(sys.modules) - set(before):
            if _is_local(sys.modules[name], root):
                del sys.modules[name]
        sys.modules.update(before)
    return module


def _env_wiring_entries(module, tree):
    """`check_env_wiring.MUTATIONS`: (name, {key: pairs}, expected)."""
    resolved, unresolved = resolve_keys(tree)
    key_paths: dict[str, tuple[str, ...]] = {}
    for key, src_names in resolved.items():
        paths = []
        for src_name in src_names:
            value = getattr(module, src_name, None)
            if value is None:
                unresolved.append(f"{key} -> {src_name} (no such module constant)")
                continue
            paths.append(str(value))
        if paths:
            key_paths[key] = tuple(paths)
    entries = [("MUTATIONS", name, key, pairs)
               for name, muts, _expected in module.MUTATIONS
               for key, pairs in muts.items()]
    return entries, key_paths, unresolved


# `check_insertion_math` keeps its two tables flat, one file each. Registered
# by name so a THIRD table cannot appear without this sweep noticing.
MATH_TABLES = {"MUTATIONS": "MATH_SRC", "CFG_MUTATIONS": "CFG_SRC"}


def _math_entries(module, tree):
    entries = []
    key_paths: dict[str, tuple[str, ...]] = {}
    unresolved: list[str] = []
    for table, src_name in MATH_TABLES.items():
        key_paths[table] = (str(getattr(module, src_name)),)
        for name, pairs, _expected in getattr(module, table):
            entries.append((table, name, table, pairs))
    for extra in sorted(table_names(tree) - set(MATH_TABLES)):
        unresolved.append(f"{extra} (table not registered in MATH_TABLES)")
    return entries, key_paths, unresolved


def run_sweep() -> int:
    """Sweep both check scripts against the real sources on disk."""
    print(f"[check_mutation_anchors] {SCRIPT_MARKER}")
    cache: dict[str, str] = {}

    def read_text(path: str) -> str:
        if path not in cache:
            cache[path] = pathlib.Path(path).read_text(encoding="utf-8")
        return cache[path]

    rc = 0
    total = 0
    for label, script, adapter in (
        ("check_env_wiring", ENV_WIRING, _env_wiring_entries),
        ("check_insertion_math", INSERTION_MATH, _math_entries),
    ):
        tree = ast.parse(script.read_text(encoding="utf-8"), str(script))
        module = _load_module(script)
        entries, key_paths, unresolved = adapter(module, tree)
        anchors = sum(len(pairs) for _t, _n, _k, pairs in entries)
        total += anchors
        print(f"\n{label}: {len(entries)} mutation/key pair(s), {anchors} anchor(s), "
              f"{len(key_paths)} key(s) -> source")
        for key in sorted(key_paths):
            names = ", ".join(pathlib.Path(p).name for p in key_paths[key])
            print(f"    {key:<14} -> {names}")
        for item in unresolved:
            print(f"  UNMAPPED  {label}: {item}")
            rc = 1
        findings = sweep(entries, key_paths, read_text)
        for finding in findings:
            print(finding.line())
        if findings:
            rc = 1
        else:
            print("  PASS  every anchor matches its source exactly once")

    print()
    if rc:
        print("STALE OR UNMAPPED ANCHORS -- fix them before running any --self-test")
    else:
        print(f"ALL {total} ANCHORS MATCH ({SCRIPT_MARKER})")
    return rc


# --------------------------------------------------------------------------
# counter-proof (D-080)
# --------------------------------------------------------------------------

class _FakeMod:
    """A stand-in module with nothing but a `__file__`, for `_is_local`."""

    def __init__(self, file):
        self.__file__ = str(file)


def _self_test() -> int:
    print(f"[check_mutation_anchors] self-test ({SCRIPT_MARKER})")
    files = {
        "a.py": "alpha\nbeta\ngamma\n",
        "b.py": "alpha\ndelta\n",
        "twice.py": "alpha\nalpha\n",
    }
    read = files.__getitem__
    paths = {"k": ("a.py",), "two": ("a.py", "b.py"), "dup": ("twice.py",)}
    checks: list[tuple[str, bool]] = []

    def run(entries):
        return sweep(entries, paths, read)

    # 1 clean table finds nothing
    clean = run([("T", "clean", "k", (("beta\n", "BETA\n"),))])
    checks.append(("a matching anchor is not reported", clean == []))

    # 2 an anchor that matches nothing is caught
    zero = run([("T", "gone", "k", (("epsilon\n", ""),))])
    checks.append(("an anchor matching 0x is reported",
                   len(zero) == 1 and zero[0].count == 0))
    checks.append(("the report names the table, mutation and key",
                   zero[:1] == [Finding("T", "gone", "k", "a.py", 0, "epsilon\n")]))

    # 3 an anchor that matches twice is caught (a mutation that hits two places)
    dup = run([("T", "dup", "dup", (("alpha\n", "A\n"),))])
    checks.append(("an anchor matching 2x is reported",
                   len(dup) == 1 and dup[0].count == 2))

    # 4 THE ORDER CASE: the second anchor exists only after the first applied.
    #   A pristine-source count would call this stale. It is not.
    ordered = run([("T", "ordered", "k", (("beta\n", "BETA\n"), ("BETA\n", "X\n")))])
    checks.append(("an anchor created by the previous one is accepted", ordered == []))

    # 5 the mirror: the first anchor DESTROYS the second. A pristine-source
    #   count would call this fine. It is stale.
    broken = run([("T", "broken", "k", (("beta\n", "BETA\n"), ("beta\n", "X\n")))])
    checks.append(("an anchor destroyed by the previous one is reported",
                   len(broken) == 1 and broken[0].count == 0))

    # 6 an unknown key is reported, not skipped
    unknown = run([("T", "stray", "nope", (("beta\n", ""),))])
    checks.append(("an unmapped key is reported",
                   len(unknown) == 1 and unknown[0].count == UNKNOWN_KEY))
    checks.append(("the unmapped line says so", "UNKNOWN KEY" in unknown[0].line()))

    # 7 two bad anchors in one mutation report ONCE -- the state after a
    #   mutation that did not apply is undefined, so the rest is not judged
    two_bad = run([("T", "two", "k", (("nope1\n", ""), ("nope2\n", "")))])
    checks.append(("a mutation stops at its first bad anchor", len(two_bad) == 1))

    # 8 a key on two files is checked in BOTH
    split = run([("T", "split", "two", (("gamma\n", "G\n"),))])
    checks.append(("a two-file key reports the file that misses it",
                   len(split) == 1 and split[0].path == "b.py" and split[0].count == 0))
    both_ok = run([("T", "both", "two", (("alpha\n", "A\n"),))])
    checks.append(("a two-file key passes when both files match", both_ok == []))

    # 9 sweep_pairs returns the INDEX of the offending anchor
    hit = sweep_pairs((("alpha\n", "A\n"), ("nope\n", "")), files["a.py"])
    checks.append(("the offending anchor index is reported",
                   hit is not None and hit[0] == 1))

    # -- the key map, read out of a synthetic check script ------------------
    src = (
        "def _read(path, mutations):\n"
        "    return path\n"
        "def _parse(src_path, muts):\n"
        "    return _read(src_path, muts)\n"
        "FOO_SRC = 'foo.py'\n"
        "BAZ_SRC = 'baz.py'\n"
        "BAR_SRC = 'bar.py'\n"
        "ZERO_SRC = 'zero.py'\n"
        "RANDOM_SRC = 'random.py'\n"
        "def load_obs_slices(mutations=()):\n"
        "    return _read(BAR_SRC, mutations)\n"
        "def build_checks(foo_mut=(), bar_mut=(), agent_mut=(), lost_mut=(), baz_mut=()):\n"
        "    _read(FOO_SRC, foo_mut)\n"
        "    _parse(BAZ_SRC, baz_mut)\n"
        "    load_obs_slices(bar_mut)\n"
        "    _read(ZERO_SRC, agent_mut)\n"
        "    _read(RANDOM_SRC, agent_mut)\n"
        "MUTATIONS = ()\n"
        "EXTRA_MUTATIONS = ()\n"
    )
    tree = ast.parse(src)
    resolved, unresolved = resolve_keys(tree)
    checks.append(("a key read directly resolves", resolved.get("foo_mut") == ["FOO_SRC"]))
    checks.append(("a key handed to a helper resolves", resolved.get("bar_mut") == ["BAR_SRC"]))
    checks.append(("a key read through the cached _parse resolves to its file, not to 'src_path'",
                   resolved.get("baz_mut") == ["BAZ_SRC"]))
    checks.append(("a key on two files resolves to both",
                   resolved.get("agent_mut") == ["RANDOM_SRC", "ZERO_SRC"]))
    checks.append(("a key that reaches no _read is reported unresolved",
                   unresolved == ["lost_mut"] and "lost_mut" not in resolved))
    checks.append(("every module-level MUTATIONS table is listed",
                   table_names(tree) == {"MUTATIONS", "EXTRA_MUTATIONS"}))

    # the map must FOLLOW the source: re-point foo_mut and the map must move
    repointed = resolve_keys(ast.parse(src.replace("_read(FOO_SRC, foo_mut)",
                                                   "_read(BAR_SRC, foo_mut)")))[0]
    checks.append(("re-pointing a key moves the map with it",
                   repointed.get("foo_mut") == ["BAR_SRC"]))

    # the real registry must cover the real math tables
    math_tree = ast.parse(INSERTION_MATH.read_text(encoding="utf-8"), str(INSERTION_MATH))
    checks.append(("the real check_insertion_math tables are all registered",
                   table_names(math_tree) == set(MATH_TABLES)))

    # -- module-table isolation: a check script that publishes a name must not
    #    leave it behind for the NEXT one (this is the `torch` stand-in case)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        probe = pathlib.Path(tmp) / "anchor_probe_mod.py"
        probe.write_text("import sys\n"
                         "import colorsys  # a stdlib file OUTSIDE the repo\n"
                         "sys.modules.setdefault('anchor_probe_fake_torch', object())\n"
                         "TOKEN = 7\n", encoding="utf-8")
        sys.modules.pop("colorsys", None)
        loaded = _load_module(probe)
        checks.append(("an isolated load still returns the module",
                       getattr(loaded, "TOKEN", None) == 7))
        checks.append(("a name a check script publishes does not leak to the next",
                       "anchor_probe_fake_torch" not in sys.modules))
        checks.append(("the probe module itself is dropped again",
                       "anchor_probe_mod" not in sys.modules))
        # numpy's compiled half refuses a second load, so the cache must survive
        checks.append(("a module from OUTSIDE the repo stays cached",
                       "colorsys" in sys.modules))
        probe_root = probe.resolve().parent
        checks.append(("the local/foreign split is what decides",
                       _is_local(loaded, probe_root)
                       and not _is_local(sys.modules["colorsys"], probe_root)))
        # the real case: `tools/torch_shim.py` must count as local to `scripts/`
        checks.append(("torch_shim counts as local to the check scripts",
                       _is_local(_FakeMod(SCRIPTS_DIR / "tools" / "torch_shim.py"),
                                 SCRIPTS_DIR)))
        checks.append(("a stand-in object with no file counts as local",
                       _is_local(object(), SCRIPTS_DIR)))

    bad = [label for label, ok in checks if not ok]
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print()
    if bad:
        print(f"{len(bad)} SELF-TEST CASE(S) FAILED: {bad}")
        return 1
    print(f"ALL {len(checks)} SELF-TEST CASES PASSED ({SCRIPT_MARKER})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--self-test", action="store_true",
                        help="run the counter-proof instead of sweeping the real sources")
    args = parser.parse_args()
    return _self_test() if args.self_test else run_sweep()


if __name__ == "__main__":
    raise SystemExit(main())
