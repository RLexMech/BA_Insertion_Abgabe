"""Offline check of the demo-sprint env's pure-torch pieces (no Isaac needed).

Runs on the dev PC, which has neither Isaac nor PyTorch: the two pieces that
carry real risk -- the batched quaternion-to-axes helper and the reward
function -- use only elementary tensor operations, so they are extracted from
``proxytask_env.py`` by source text and exercised against a numpy stand-in
when PyTorch is absent. With PyTorch installed (training machine) the real
library is used instead.

This is a scaffold for the algebra only. It says nothing about Isaac, the
GPU, the solver or the physics, all of which still require a training-machine
run per CLAUDE.md.

It has already earned its place: on 2026-07-26 it caught the ported
``_axes_from_quat_batched`` returning the transposed rotation matrix, which
would have rotated every peg tip wrongly while still producing plausible
positions -- exactly the class of silent error the increment-3 docstring
warns about.

    python scripts/check_demo_reward_math.py
"""

from __future__ import annotations

import ast
import math
import pathlib
import sys

import numpy as np

ENV_SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "source"
    / "proxytask"
    / "proxytask"
    / "tasks"
    / "direct"
    / "proxytask"
    / "proxytask_env.py"
)


# ---------------------------------------------------------------------------
# torch, or a numpy stand-in covering only the operations under test
# ---------------------------------------------------------------------------

try:
    import torch
except ModuleNotFoundError:

    class _ND(np.ndarray):
        """ndarray whose ``.transpose(a, b)`` swaps two axes, as torch's does."""

        def transpose(self, *axes):  # type: ignore[override]
            if len(axes) == 2 and self.ndim > 2:
                return np.swapaxes(self, axes[0], axes[1]).view(_ND)
            return np.ndarray.transpose(self, *axes).view(_ND)

    class _Linalg:
        @staticmethod
        def norm(x, dim=None, keepdim=False, axis=None):
            return np.linalg.norm(x, axis=dim if dim is not None else axis, keepdims=keepdim)

    class _Jit:
        @staticmethod
        def script(fn):
            return fn

    class _TorchShim:
        linalg = _Linalg()
        jit = _Jit()

        @staticmethod
        def stack(seq, dim=0):
            return np.stack(seq, axis=dim).view(_ND)

        @staticmethod
        def tensor(data, device=None, dtype=None):
            return np.array(data, dtype=np.float64 if dtype is None else dtype)

        @staticmethod
        def zeros(*shape, device=None):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return np.zeros(shape)

        @staticmethod
        def ones(*shape, device=None):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return np.ones(shape)

        eye = staticmethod(np.eye)
        zeros_like = staticmethod(np.zeros_like)
        maximum = staticmethod(np.maximum)
        where = staticmethod(np.where)
        matmul = staticmethod(np.matmul)
        randn = staticmethod(np.random.randn)

        @staticmethod
        def full_like(x, v):
            return np.full_like(x, v, dtype=np.float64)

        @staticmethod
        def clamp(x, min=None, max=None):  # noqa: A002 - mirrors torch's kwarg names
            return np.clip(x, min, max)

        @staticmethod
        def sum(x, dim=None):
            return np.sum(x, axis=dim)

        @staticmethod
        def manual_seed(s):
            np.random.seed(s)

        @staticmethod
        def allclose(a, b, atol=1e-8):
            return bool(np.allclose(a, b, atol=atol))

    torch = _TorchShim()  # type: ignore[assignment]
    print("[check] PyTorch not installed - using the numpy stand-in (algebra only).")
else:
    print(f"[check] using real PyTorch {torch.__version__}.")


# ---------------------------------------------------------------------------
# Extract the functions under test from the env source
# ---------------------------------------------------------------------------

WANTED = {"_axes_from_quat", "_axes_from_quat_batched", "compute_rewards"}
_ns: dict = {"torch": torch}
for _node in ast.parse(ENV_SRC.read_text(encoding="utf-8")).body:
    if isinstance(_node, ast.FunctionDef) and _node.name in WANTED:
        # Drop @torch.jit.script: irrelevant to the numerics, and the stand-in
        # cannot compile it.
        _node.decorator_list = []
        exec(compile(ast.Module(body=[_node], type_ignores=[]), str(ENV_SRC), "exec"), _ns)
_missing = WANTED - set(_ns)
if _missing:
    raise SystemExit(f"could not extract from {ENV_SRC}: {sorted(_missing)}")

_axes_from_quat = _ns["_axes_from_quat"]
_axes_from_quat_batched = _ns["_axes_from_quat_batched"]
compute_rewards = _ns["compute_rewards"]

FAILURES: list[str] = []


def _s(x) -> float:
    """First element as a Python float, for both torch tensors and ndarrays.

    ``float()`` on a one-element array is an error in numpy 2.x, while
    ``.item()`` exists on both -- but only via the underlying array, so go
    through reshape to stay agnostic.
    """
    return float(np.asarray(x).reshape(-1)[0])


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. Batched quaternion helper vs the verified single-quaternion form
# ---------------------------------------------------------------------------

print("1. batched vs single quaternion -> axes")
torch.manual_seed(0)
q = torch.randn(64, 4)
q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
batched = _axes_from_quat_batched(q)
single = torch.stack([_axes_from_quat(q[i]) for i in range(q.shape[0])])
err = float(abs(batched - single).max())
check("batched agrees with single-quaternion form", err < 1e-6, f"max abs diff {err:.3e}")

ident = _axes_from_quat_batched(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))[0]
check("identity quat -> identity matrix", torch.allclose(ident, torch.eye(3), atol=1e-6))

s, c = math.sin(math.pi / 4), math.cos(math.pi / 4)
z_axis = _axes_from_quat_batched(torch.tensor([[c, s, 0.0, 0.0]]))[0][:, 2]
check(
    "90deg about x maps local +z to world -y",
    torch.allclose(z_axis, torch.tensor([0.0, -1.0, 0.0]), atol=1e-6),
    f"got {[round(float(v), 6) for v in z_axis]}",
)

# The exact product _peg_geometry uses: matmul((N,3,3), (3,)) -> (N,3).
tip_offset = torch.tensor([0.0, 0.0, 0.050])
prod = torch.matmul(batched, tip_offset)
check("matmul((N,3,3),(3,)) shape is (N,3)", tuple(prod.shape) == (64, 3), str(tuple(prod.shape)))
prod_ref = torch.stack([single[i] @ tip_offset for i in range(64)])
check("matmul matches per-env matrix-vector product", torch.allclose(prod, prod_ref, atol=1e-6))


# ---------------------------------------------------------------------------
# 2. Reward function, including the three exploits it must defeat
# ---------------------------------------------------------------------------

print("2. compute_rewards")
SUCCESS_DEPTH = 0.025
CLEARANCE = 0.0025  # demo-sprint radial clearance, 25 mm peg in a 30 mm bore
W_APPROACH, W_DEPTH, W_SUCCESS, W_ACTION, W_MISPLACED, W_ALIGN = 2.0, 100.0, 10.0, 0.01, 20.0, 2.0
PLATE_TOP_Z = 0.755
PLATE_BOTTOM_Z = 0.700
BORE_DEPTH = 0.030
MIN_ALIGNMENT = 0.99
bore = torch.tensor([0.0, -0.225, PLATE_TOP_Z])


def step(tip, max_depth, actions=None, remaining=0.0, alignment=1.0):
    """Mirror of the env's _peg_geometry + compute_rewards for one env.

    The gate is the bore-volume test the env applies: laterally inside the
    clearance, no deeper than the blind bore, and pointing into it. Keeping
    this in step with _peg_geometry is what makes the checks below meaningful.
    """
    tip = tip.reshape(1, 3)
    depth = torch.tensor([PLATE_TOP_Z - float(tip[0, 2])])
    align = torch.tensor([alignment])
    xy_inside = torch.linalg.norm(tip[:, :2] - bore[:2], dim=-1) < CLEARANCE
    gate = xy_inside & (depth <= BORE_DEPTH + 0.001) & (align >= MIN_ALIGNMENT)
    below_plate = tip[:, 2] < PLATE_BOTTOM_Z
    actions = torch.zeros(1, 6) if actions is None else actions
    return compute_rewards(
        tip, bore, depth, gate, below_plate, align, torch.tensor([remaining]), actions, max_depth,
        SUCCESS_DEPTH, W_APPROACH, W_DEPTH, W_SUCCESS, W_ACTION, W_MISPLACED, W_ALIGN,
    )


# Exploit A -- gate-farming: descend inside the gate, leave sideways, return
# to the same depth. The return must pay nothing.
md = torch.zeros(1)
r1, md, _ = step(torch.tensor([0.0, -0.225, 0.745]), md)       # 10 mm deep, in gate
_, md, _ = step(torch.tensor([0.02, -0.225, 0.760]), md)       # lifted out, off gate
r2, md, _ = step(torch.tensor([0.0, -0.225, 0.745]), md)       # back to 10 mm
first = _s(r1) + W_APPROACH * 0.010
second = _s(r2) + W_APPROACH * 0.010
check("first entry to 10 mm pays the depth term", abs(first - W_DEPTH * 0.010) < 1e-4,
      f"{first:.4f} vs {W_DEPTH * 0.010:.4f}")
check("re-entry to the SAME depth pays nothing", abs(second) < 1e-6, f"depth component {second:.3e}")
check("max_depth is monotonic", abs(_s(md) - 0.010) < 1e-6, f"max_depth {_s(md):.6f}")

r3, md, _ = step(torch.tensor([0.0, -0.225, 0.740]), md)       # 15 mm
inc = _s(r3) + W_APPROACH * 0.015
check("progress beyond the max pays only the increment", abs(inc - W_DEPTH * 0.005) < 1e-4,
      f"{inc:.4f} vs {W_DEPTH * 0.005:.4f}")

# Exploit B -- tunneling through the plate beside the bore: 30 mm "deep" but
# 20 mm off-axis must not count as success, and must not raise max_depth.
_, md_off, succ_tunnel = step(torch.tensor([0.020, -0.225, 0.725]), torch.zeros(1))
check("deep but off-axis does NOT count as success", not bool(succ_tunnel))
check("off-gate depth does not raise max_depth", abs(_s(md_off)) < 1e-9, f"{_s(md_off):.3e}")
_, _, succ_real = step(torch.tensor([0.0, -0.225, 0.725]), torch.zeros(1))
check("deep and on-axis DOES count as success", bool(succ_real))
_, _, succ_edge = step(torch.tensor([0.0, -0.225, 0.730]), torch.zeros(1))
check("depth exactly at the 25 mm threshold counts", bool(succ_edge))

# Exploit C -- diving under the plate. Directly below the bore the tip is
# CLOSER to the bore entrance (55 mm) than at the home pose (~83 mm), and the
# unbounded depth reads 55 mm, past the 25 mm threshold. Before 2026-07-27
# this paid ~16.5 points for going under the table; a training run was
# observed doing it.
under = torch.tensor([0.0, -0.225, 0.690])  # 10 mm below the plate underside
r_under, md_under, succ_under = step(under, torch.zeros(1))
check("under the plate does NOT count as success", not bool(succ_under))
check("under the plate pays no depth reward", abs(_s(md_under)) < 1e-9, f"max_depth {_s(md_under):.3e}")
r_home, _, _ = step(torch.tensor([0.0, -0.2115, 0.836]), torch.zeros(1))
check("under the plate scores WORSE than the home pose", _s(r_under) < _s(r_home),
      f"under {_s(r_under):.4f} < home {_s(r_home):.4f}")
# The bore bottom is a hard stop at 30 mm; 1 mm past it means through the plate.
_, _, succ_through = step(torch.tensor([0.0, -0.225, 0.724]), torch.zeros(1))
check("deeper than the blind bore does NOT count as success", not bool(succ_through))
_, _, succ_bottom = step(torch.tensor([0.0, -0.225, 0.725]), torch.zeros(1))
check("resting on the bore bottom still counts as success", bool(succ_bottom))

# Exploit D -- suicide: terminating early to stop paying the approach term.
# Leaving the arena must be no better than standing there for the rest of the
# episode, or the policy learns to dive out immediately.
REMAINING = 200.0
r_leave, _, _ = step(under, torch.zeros(1), remaining=REMAINING)
r_stay, _, _ = step(under, torch.zeros(1), remaining=0.0)
cost_of_staying = _s(r_stay) * (1.0 + REMAINING)  # this step plus the rest
check("leaving is not better than staying out the episode", _s(r_leave) <= cost_of_staying + 1e-3,
      f"leave {_s(r_leave):.3f} vs stay {cost_of_staying:.3f}")
check("the out-of-bounds penalty is priced, not punitive",
      abs(_s(r_leave) - cost_of_staying) < 1e-3,
      f"difference {abs(_s(r_leave) - cost_of_staying):.6f}")
# In bounds, the remaining-steps count must change nothing at all.
r_in_0, _, _ = step(torch.tensor([0.0, -0.225, 0.745]), torch.zeros(1), remaining=0.0)
r_in_200, _, _ = step(torch.tensor([0.0, -0.225, 0.745]), torch.zeros(1), remaining=REMAINING)
check("in bounds, remaining steps do not affect the reward", abs(_s(r_in_0) - _s(r_in_200)) < 1e-9)

# Exploit E -- parking the tip on the bore with the peg lying across it. A
# distance-only approach term is maximised from any direction, so this pose
# scored ~0, the best value reachable without inserting, and every arm in a
# 1024-env run converged on it and stopped.
at_bore = torch.tensor([0.0, -0.225, PLATE_TOP_Z])
r_flat, _, succ_flat = step(at_bore, torch.zeros(1), alignment=0.0)     # peg across the hole
r_upright, _, _ = step(at_bore, torch.zeros(1), alignment=1.0)          # peg into the hole
check("lying across the bore scores worse than pointing into it", _s(r_flat) < _s(r_upright),
      f"flat {_s(r_flat):.4f} < upright {_s(r_upright):.4f}")
# The pose the run converged on must also lose to the honest approach from
# above, which is the move it refused to make.
above = torch.tensor([0.0, -0.225, PLATE_TOP_Z + 0.05])
r_above, _, _ = step(above, torch.zeros(1), alignment=1.0)
check("the flat parked pose loses to standing 50 mm above the bore", _s(r_flat) < _s(r_above),
      f"flat {_s(r_flat):.4f} < above {_s(r_above):.4f}")
# Alignment must gate depth too: a peg across the hole cannot enter it.
_, md_flat, succ_flat_deep = step(torch.tensor([0.0, -0.225, 0.730]), torch.zeros(1), alignment=0.5)
check("misaligned depth does NOT count as success", not bool(succ_flat_deep))
check("misaligned depth pays no depth reward", abs(_s(md_flat)) < 1e-9, f"max_depth {_s(md_flat):.3e}")
# A tilt the clearance physically permits (5.7 deg, cos 0.995) must still pass.
_, _, succ_tilt = step(torch.tensor([0.0, -0.225, 0.730]), torch.zeros(1), alignment=0.995)
check("a tilt the clearance permits still counts", bool(succ_tilt))
# Perfect alignment must cost nothing at all.
r_perfect, _, _ = step(above, torch.zeros(1), alignment=1.0)
check("perfect alignment carries no penalty",
      abs(_s(r_perfect) + W_APPROACH * 0.05) < 1e-6,
      f"{_s(r_perfect):.6f} vs {-W_APPROACH * 0.05:.6f}")

# Exploit F -- the approach term must reward getting closer, above the plate.
r_far, _, _ = step(torch.tensor([0.0, -0.125, 0.905]), torch.zeros(1))
r_near, _, _ = step(torch.tensor([0.0, -0.215, 0.775]), torch.zeros(1))
check("closer to the bore scores higher off-gate", _s(r_near) > _s(r_far),
      f"near {_s(r_near):.4f} > far {_s(r_far):.4f}")

r_still, _, _ = step(torch.tensor([0.0, -0.225, 0.905]), torch.zeros(1))
r_moving, _, _ = step(torch.tensor([0.0, -0.225, 0.905]), torch.zeros(1), actions=torch.ones(1, 6))
check("action penalty subtracts w_action * ||a||^2", abs(_s(r_still - r_moving) - W_ACTION * 6) < 1e-5,
      f"delta {_s(r_still - r_moving):.6f} vs {W_ACTION * 6:.6f}")

tips = torch.tensor([[0.0, -0.225, 0.745], [0.02, -0.225, 0.760], [0.0, -0.225, 0.725]])
depths = PLATE_TOP_Z - tips[:, 2]
gates = (torch.linalg.norm(tips[:, :2] - bore[:2], dim=-1) < CLEARANCE) & (depths <= BORE_DEPTH + 0.001)
below = tips[:, 2] < PLATE_BOTTOM_Z
rb, mb, sb = compute_rewards(
    tips, bore, depths, gates, below, torch.ones(3), torch.zeros(3), torch.zeros(3, 6), torch.zeros(3),
    SUCCESS_DEPTH, W_APPROACH, W_DEPTH, W_SUCCESS, W_ACTION, W_MISPLACED, W_ALIGN,
)
check("batched call returns per-env vectors",
      tuple(rb.shape) == (3,) and tuple(mb.shape) == (3,) and tuple(sb.shape) == (3,),
      f"{tuple(rb.shape)}, {tuple(mb.shape)}, {tuple(sb.shape)}")
check("batched success flags are [F, F, T]", [bool(v) for v in sb] == [False, False, True],
      str([bool(v) for v in sb]))

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")



