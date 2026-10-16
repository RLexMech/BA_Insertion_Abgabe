# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""torch, or a numpy stand-in for the elementary tensor algebra (offline)."""

from __future__ import annotations

import ast
import sys

import numpy as np

SHIM_MARKER = "torch-shim-2026-09-09a"


class _ND(np.ndarray):
    """ndarray with the tensor METHODS torch code actually writes."""

    def transpose(self, *axes):  # type: ignore[override]
        if len(axes) == 2 and self.ndim > 2:
            return np.swapaxes(self, axes[0], axes[1]).view(_ND)
        return np.ndarray.transpose(self, *axes).view(_ND)

    def abs(self):
        return np.abs(self).view(_ND)

    def clone(self):
        """torch's ``.clone()``; numpy spells the same thing ``.copy()``."""
        return np.copy(self).view(_ND)

    def long(self):
        """torch's ``.long()``: cast to int64."""
        return np.asarray(self).astype(np.int64).view(_ND)

    def clamp(self, min=None, max=None):  # noqa: A002 - mirrors torch's kwarg names
        """torch's ``.clamp(min, max)``; numpy spells it ``np.clip``."""
        return np.clip(self, min, max).view(_ND)

    def sum(self, dim=None, keepdim=False, axis=None, **kwargs):  # type: ignore[override]
        """torch's ``.sum(dim=...)``; numpy spells the same axis ``axis``."""
        if dim is not None:
            kwargs["axis"] = dim
        elif axis is not None:
            kwargs["axis"] = axis
        if keepdim:
            kwargs["keepdims"] = True
        return np.asarray(np.asarray(self).sum(**kwargs)).view(_ND)


def _nd(x):
    """Every constructed tensor is an ``_ND``, or the override above never runs."""
    return np.asarray(x).view(_ND)


class _Linalg:
    @staticmethod
    def norm(x, dim=None, keepdim=False, axis=None):
        return np.linalg.norm(x, axis=dim if dim is not None else axis, keepdims=keepdim)


class _Jit:
    @staticmethod
    def script(fn):
        """torch.jit.script is a compiler, not semantics -- identity here."""
        return fn


class _TorchShim:
    """Only the operations the offline-checked math actually calls."""

    linalg = _Linalg()
    jit = _Jit()

    @staticmethod
    def stack(seq, dim=0):
        return np.stack(seq, axis=dim).view(_ND)

    @staticmethod
    def cat(seq, dim=0):
        return np.concatenate(seq, axis=dim).view(_ND)

    @staticmethod
    def cross(a, b, dim=-1):
        """torch spells the axis ``dim``; numpy spells it ``axis``."""
        return np.cross(a, b, axis=dim).view(_ND)

    @staticmethod
    def tensor(data, device=None, dtype=None):
        return _nd(np.array(data, dtype=np.float64 if dtype is None else dtype))

    @staticmethod
    def zeros(*shape, device=None):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _nd(np.zeros(shape))

    @staticmethod
    def ones(*shape, device=None):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _nd(np.ones(shape))

    @staticmethod
    def full_like(x, v):
        return _nd(np.full_like(x, v, dtype=np.float64))

    @staticmethod
    def arange(*a, device=None):
        return _nd(np.arange(*a, dtype=np.float64))

    @staticmethod
    def eye(*a, **kw):
        return _nd(np.eye(*a, **kw))

    @staticmethod
    def zeros_like(x):
        return _nd(np.zeros_like(x))

    @staticmethod
    def ones_like(x):
        return _nd(np.ones_like(x))

    exp = staticmethod(np.exp)
    tanh = staticmethod(np.tanh)
    cos = staticmethod(np.cos)
    sin = staticmethod(np.sin)
    atan2 = staticmethod(np.arctan2)
    acos = staticmethod(np.arccos)
    sqrt = staticmethod(np.sqrt)
    abs = staticmethod(np.abs)
    sign = staticmethod(np.sign)
    round = staticmethod(np.round)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    where = staticmethod(np.where)
    matmul = staticmethod(np.matmul)
    randn = staticmethod(np.random.randn)
    isfinite = staticmethod(np.isfinite)

    @staticmethod
    def clamp(x, min=None, max=None):  # noqa: A002 - mirrors torch's kwarg names
        return np.clip(x, min, max)

    @staticmethod
    def sum(x, dim=None):
        return np.sum(x, axis=dim)

    @staticmethod
    def amax(x, dim=None):
        return np.amax(x, axis=dim)

    @staticmethod
    def amin(x, dim=None):
        return np.amin(x, axis=dim)

    @staticmethod
    def all(x, dim=None):
        return np.all(x, axis=dim)

    @staticmethod
    def any(x, dim=None):
        return np.any(x, axis=dim)

    @staticmethod
    def manual_seed(s):
        np.random.seed(s)

    @staticmethod
    def allclose(a, b, atol=1e-8):
        return bool(np.allclose(a, b, atol=atol))

    @staticmethod
    def logical_and(a, b):
        return np.logical_and(a, b)

    @staticmethod
    def logical_or(a, b):
        return np.logical_or(a, b)

    @staticmethod
    def logical_not(a):
        return np.logical_not(a)


JIT_DECORATOR = "@torch.jit.script"


def _jit_decorator_lines(src: str) -> set[int]:
    """1-based line numbers of every decorator that resolves to jit's script."""
    lines: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        for dec in getattr(node, "decorator_list", []):
            expr = dec.func if isinstance(dec, ast.Call) else dec
            chain: list[str] = []
            while isinstance(expr, ast.Attribute):
                chain.append(expr.attr)
                expr = expr.value
            if isinstance(expr, ast.Name):
                chain.append(expr.id)
            if chain and chain[0] == "script" and ("jit" in chain or len(chain) == 1):
                lines.update(range(dec.lineno, (dec.end_lineno or dec.lineno) + 1))
    return lines


def strip_jit_script(src: str) -> str:
    """Remove every ``@torch.jit.script`` decorator line from module source."""
    out, blanked = [], set()
    for i, line in enumerate(src.splitlines(keepends=True), start=1):
        bare = line.strip()
        rest = bare[len(JIT_DECORATOR):]
        if bare.startswith(JIT_DECORATOR) and (not rest or rest[0].isspace() or rest[0] == '#'):
            blanked.add(i)
            out.append(line[len(line.rstrip()):])
            continue
        out.append(line)
    if not blanked:
        raise SystemExit(
            f"strip_jit_script removed no {JIT_DECORATOR!r} line. Either the module "
            "no longer uses it, or it is spelled differently -- and a check that "
            "execs a mutated string is blind under real torch without this strip."
        )
    expected = _jit_decorator_lines(src)
    if blanked != expected:
        survivors = sorted(expected - blanked)
        false_hits = sorted(blanked - expected)
        raise SystemExit(
            "strip_jit_script refuses a partial strip. "
            f"Real jit-script decorators the text rule missed, by line: {survivors}; "
            f"non-decorator lines it blanked (e.g. inside a string), by line: {false_hits}. "
            "A missed decorator recompiles from the file on disk under real torch and "
            "its mutations become silent no-ops -- spell it bare, or reword the string."
        )
    return "".join(out)


def load(announce: bool = True):
    """Return ``(torch_or_stand_in, is_real)``."""
    try:
        import torch  # noqa: PLC0415 - the whole point is the optional import
    except ModuleNotFoundError:
        if announce:
            print("[shim] PyTorch not installed - using the numpy stand-in (algebra only).")
        return _TorchShim(), False
    torch.set_default_dtype(torch.float64)
    if announce:
        print(f"[shim] using real PyTorch {torch.__version__}, default dtype float64.")
    return torch, True


def _self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    t = _TorchShim()
    print(f"[shim] self-test {SHIM_MARKER}")

    z = t.zeros(2, 3)
    check("zeros((2,3)) has shape (2,3)", z.shape == (2, 3), str(z.shape))
    check("tensor is float64", t.tensor([1.0]).dtype == np.float64, str(t.tensor([1.0]).dtype))
    check("zeros((2,3)) tuple form agrees", t.zeros((2, 3)).shape == (2, 3))

    c = t.cat((t.zeros(2, 3), t.ones(2, 4)), dim=-1)
    check("cat((2,3),(2,4), dim=-1) -> (2,7)", c.shape == (2, 7), str(c.shape))
    check("cat keeps the block order", float(c[0, 2]) == 0.0 and float(c[0, 3]) == 1.0)
    check("stack of two (3,) -> (2,3)", t.stack([t.zeros(3), t.ones(3)]).shape == (2, 3))

    _x = t.tensor([[1.0, 0.0, 0.0]])
    _y = t.tensor([[0.0, 1.0, 0.0]])
    check("cross(x, y) is +z", [float(v) for v in t.cross(_x, _y, dim=-1)[0]] == [0.0, 0.0, 1.0])
    check("cross is antisymmetric",
          [float(v) for v in t.cross(_y, _x, dim=-1)[0]] == [0.0, 0.0, -1.0])
    check("cross of a vector with itself is zero",
          float(t.linalg.norm(t.cross(_x, _x, dim=-1), dim=-1)[0]) == 0.0)
    check("cross returns an _ND (so .abs()/.clone() still work)",
          isinstance(t.cross(_x, _y, dim=-1), _ND))
    _orig = t.tensor([[1.0, 2.0]])
    _copy = _orig.clone()
    _copy[0, 0] = 99.0
    check("clone does not alias the original", float(_orig[0, 0]) == 1.0, str(float(_orig[0, 0])))
    check("clone carries the values across", float(_copy[0, 1]) == 2.0)
    check("clone returns an _ND", isinstance(_copy, _ND))

    _u = t.tensor([0.0, 0.4, 0.999999, 1.0])
    _idx = (_u * 11).long()
    check("long() truncates towards zero, so a non-negative product floors",
          [int(v) for v in _idx] == [0, 4, 10, 11], str([int(v) for v in _idx]))
    check("long() returns an _ND, so .clamp() chains off it", isinstance(_idx, _ND))
    check("clamp() as a METHOD bounds both ends",
          [int(v) for v in _idx.clamp(0, 10)] == [0, 4, 10, 10],
          str([int(v) for v in _idx.clamp(0, 10)]))
    check("clamp() as a METHOD leaves the receiver alone",
          [int(v) for v in _idx] == [0, 4, 10, 11])

    check("exp(0) == 1", float(t.exp(t.tensor([0.0]))[0]) == 1.0)
    check("tanh(0) == 0", float(t.tanh(t.tensor([0.0]))[0]) == 0.0)
    check("tanh(1) matches numpy", abs(float(t.tanh(t.tensor([1.0]))[0]) - np.tanh(1.0)) < 1e-15)

    check("acos(1) == 0", abs(float(t.acos(t.tensor([1.0]))[0])) < 1e-15)
    check("acos(0) is pi/2", abs(float(t.acos(t.tensor([0.0]))[0]) - np.pi / 2) < 1e-15)
    check("acos(-1) is pi", abs(float(t.acos(t.tensor([-1.0]))[0]) - np.pi) < 1e-15)

    check("clamp(x, min=0) floors at 0", float(t.clamp(t.tensor([-3.0]), min=0.0)[0]) == 0.0)
    check("clamp(x, max=1) caps at 1", float(t.clamp(t.tensor([3.0]), max=1.0)[0]) == 1.0)

    m = t.tensor([[1.0, 5.0], [9.0, 2.0]])
    check("sum(dim=-1) reduces the last axis", list(t.sum(m, dim=-1)) == [6.0, 11.0],
          str(list(t.sum(m, dim=-1))))
    check("amax(dim=-1) reduces the last axis", list(t.amax(m, dim=-1)) == [5.0, 9.0],
          str(list(t.amax(m, dim=-1))))

    n = t.linalg.norm(t.tensor([[3.0, 4.0]]), dim=-1)
    check("linalg.norm(dim=-1) of (3,4) is 5", abs(float(n[0]) - 5.0) < 1e-12, str(float(n[0])))

    w = t.where(t.tensor([1.0, 0.0]) > 0.5, t.tensor([7.0, 7.0]), t.tensor([0.0, 0.0]))
    check("where picks per element", list(w) == [7.0, 0.0], str(list(w)))

    r = t.matmul(t.stack([t.eye(3), t.eye(3)]), t.tensor([1.0, 2.0, 3.0]))
    check("matmul((2,3,3),(3,)) -> (2,3)", r.shape == (2, 3), str(r.shape))
    check("identity matmul is the vector itself", list(r[0]) == [1.0, 2.0, 3.0])

    nd = t.stack([t.eye(3), t.eye(3)])
    check("_ND.transpose(1,2) swaps the two trailing axes", nd.transpose(1, 2).shape == (2, 3, 3))

    a = t.tensor([[-3.0, 4.0], [0.0, -0.5]])
    check("_ND.abs() is the elementwise magnitude",
          [list(r) for r in a.abs()] == [[3.0, 4.0], [0.0, 0.5]],
          str([list(r) for r in a.abs()]))
    check("a column slice keeps .abs()", float(a[:, 0].abs()[0]) == 3.0)
    check(".abs() returns a tensor, not a bare ndarray", isinstance(a.abs(), _ND),
          type(a.abs()).__name__)

    oblong = t.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    check("transpose(1,2) really swaps, not just returns",
          oblong.transpose(1, 2).shape == (1, 3, 2), str(oblong.transpose(1, 2).shape))
    check("transpose(1,2) moves the elements with the axes",
          float(oblong.transpose(1, 2)[0, 2, 0]) == 3.0)
    for name, made in (("tensor", oblong), ("eye", t.eye(3)), ("zeros", t.zeros(1, 2, 3)),
                       ("ones", t.ones(1, 2, 3)), ("zeros_like", t.zeros_like(oblong)),
                       ("ones_like", t.ones_like(oblong)), ("full_like", t.full_like(oblong, 1.0)),
                       ("arange", t.arange(6.0))):
        check(f"{name}() returns a tensor with torch transpose semantics",
              isinstance(made, _ND), type(made).__name__)

    check("atan2(1,0) is pi/2",
          abs(float(t.atan2(t.tensor([1.0]), t.tensor([0.0]))[0]) - np.pi / 2) < 1e-15)

    try:
        t.svd  # noqa: B018 - the point is the AttributeError
        check("an unimplemented op raises AttributeError", False, "svd resolved")
    except AttributeError:
        check("an unimplemented op raises AttributeError", True)

    NL = chr(10)
    HASH = chr(35)
    jit_src = NL.join([
        "import torch", "", "", "@torch.jit.script", "def f(x):", "    return x",
        "", "", "@torch.jit.script  ", "def g(x):", "    return x", "",
    ])
    stripped = strip_jit_script(jit_src)
    check("strip_jit_script removes every decorator line",
          JIT_DECORATOR not in stripped, repr(stripped))
    check("strip_jit_script keeps the function bodies",
          stripped.count("def f(x):") == 1 and stripped.count("return x") == 2,
          repr(stripped))
    check("strip_jit_script keeps the import", "import torch" in stripped)
    try:
        strip_jit_script(NL.join(["def f(x):", "    return x", ""]))
        stripped_nothing = False
    except SystemExit:
        stripped_nothing = True
    check("strip_jit_script REFUSES source with no decorator to strip",
          stripped_nothing)

    check("strip_jit_script keeps the line count",
          len(stripped.splitlines()) == len(jit_src.splitlines()),
          f"{len(jit_src.splitlines())} -> {len(stripped.splitlines())}")
    check("strip_jit_script leaves every def on its own line number",
          [i for i, l in enumerate(stripped.splitlines()) if l.startswith("def ")]
          == [i for i, l in enumerate(jit_src.splitlines()) if l.startswith("def ")],
          repr(stripped))

    trailing = NL.join([
        "@torch.jit.script", "def f(x):", "    return x", "",
        "@torch.jit.script  %s put back by the mutation" % HASH,
        "def g(x):", "    return x", "",
    ])
    check("strip_jit_script removes a decorator with a trailing comment",
          "@torch.jit.script" not in strip_jit_script(trailing),
          repr(strip_jit_script(trailing)))

    keep = NL.join([
        "@torch.jit.script", "def f(x):", "    return x", "",
        "%s NOT ``@torch.jit.script``, and the reason is MEASURED." % HASH,
        "@torch.jit.script_method", "def g(x):", "    return x", "",
    ])
    kept = strip_jit_script(keep)
    check("strip_jit_script keeps a prose line that only quotes the name",
          "NOT ``@torch.jit.script``" in kept, repr(kept))
    check("strip_jit_script keeps @torch.jit.script_method",
          "@torch.jit.script_method" in kept, repr(kept))

    def refuses(bad_src: str) -> bool:
        try:
            strip_jit_script(bad_src)
            return False
        except SystemExit:
            return True

    check("strip_jit_script REFUSES when '@ torch.jit.script' survives beside a bare one",
          refuses(NL.join([
              "@torch.jit.script", "def f(x):", "    return x", "",
              "@ torch.jit.script", "def g(x):", "    return x", "",
          ])))
    check("strip_jit_script REFUSES when an aliased '@jit.script' survives beside a bare one",
          refuses(NL.join([
              "from torch import jit", "",
              "@torch.jit.script", "def f(x):", "    return x", "",
              "@jit.script", "def g(x):", "    return x", "",
          ])))
    check("strip_jit_script REFUSES to blank a docstring line starting with the decorator text",
          refuses(NL.join([
              '"""doc', "@torch.jit.script is a compiler, not semantics.", '"""', "",
              "@torch.jit.script", "def f(x):", "    return x", "",
          ])))

    obj, is_real = load(announce=False)
    try:
        import torch as _real  # noqa: PLC0415

        check("load() returns real torch when it imports", is_real and obj is _real)
        check("load() sets the default dtype to float64",
              _real.get_default_dtype() == _real.float64)
    except ModuleNotFoundError:
            check("load() falls back to the stand-in when torch is absent",
              (not is_real) and isinstance(obj, _TorchShim))

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    print("nothing to do; run with --self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
