"""Offline check of the square-peg env's pure-torch pieces (no Isaac needed).

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
    # The shim builds every tensor as float64; real torch defaults to float32.
    # A pose literal like z = 0.730 is then stored as 0.7300000190734863, so the
    # exactly-at-threshold depth comes out 1.9e-8 m short of SUCCESS_DEPTH and
    # the four boundary-INCLUSIVE checks fail for a reason that has nothing to
    # do with the gate. Measured on the training machine 2026-08-05; 1.9e-8 m is
    # five orders below the 1.0 mm clearance the gate discriminates against.
    # Matching the shim keeps both paths testing the same algebra rather than
    # the same rounding -- no threshold is relaxed.
    torch.set_default_dtype(torch.float64)
    print("[check] default dtype set to float64 so boundary poses are exact.")


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
# 2. Reward function, including the exploits it must defeat (one assertion
#    per exploit question in the compute_rewards docstring)
# ---------------------------------------------------------------------------

print("2. compute_rewards")
SUCCESS_DEPTH = 0.025
PEG_SIDE = 0.030   # 30 mm square peg ...
POCKET_SIDE = 0.032  # ... in a 32 mm square pocket: 1.0 mm clearance per axis
W_APPROACH, W_DEPTH, W_SUCCESS, W_ACTION, W_MISPLACED, W_ALIGN, W_YAW = (
    2.0, 100.0, 10.0, 0.01, 20.0, 2.0, 2.0
)
PLATE_TOP_Z = 0.755
PLATE_BOTTOM_Z = 0.700
POCKET_DEPTH = 0.035
MIN_ALIGNMENT = 0.99
entrance = torch.tensor([0.0, -0.225, PLATE_TOP_Z])


def step(tip, max_depth, actions=None, remaining=0.0, alignment=1.0, phi=0.0):
    """Mirror of the env's _peg_geometry + compute_rewards for one env.

    The gate is the four-corner containment test the env applies: every
    corner of the tip cross-section, ``tip_xy +- (side/2) x_peg +-
    (side/2) y_peg`` with the peg axes yawed by ``phi``, inside the pocket's
    half-side on both XY axes (Chebyshev), no deeper than the blind pocket,
    and pointing into it. Keeping this in step with _peg_geometry is what
    makes the checks below meaningful.
    """
    tip = tip.reshape(1, 3)
    depth = torch.tensor([PLATE_TOP_Z - float(tip[0, 2])])
    align = torch.tensor([alignment])
    half = PEG_SIDE / 2.0
    ex = np.array([math.cos(phi), math.sin(phi)]) * half
    ey = np.array([-math.sin(phi), math.cos(phi)]) * half
    tip_xy = np.asarray(tip).reshape(3)[:2] - np.asarray(entrance)[:2]
    cheb = max(
        float(np.abs(tip_xy + sx * ex + sy * ey).max()) for sx in (1, -1) for sy in (1, -1)
    )
    # Encoded as a float comparison so both real torch and the numpy shim
    # produce a boolean tensor (torch.tensor([True]) is float64 in the shim).
    corners_inside = torch.tensor([1.0 if cheb < POCKET_SIDE / 2.0 else 0.0]) > 0.5
    gate = corners_inside & (depth <= POCKET_DEPTH + 0.001) & (align >= MIN_ALIGNMENT)
    below_plate = tip[:, 2] < PLATE_BOTTOM_Z
    cos4phi = torch.tensor([math.cos(4.0 * phi)])
    actions = torch.zeros(1, 6) if actions is None else actions
    return compute_rewards(
        tip, entrance, depth, gate, below_plate, align, cos4phi, torch.tensor([remaining]),
        actions, max_depth,
        SUCCESS_DEPTH, W_APPROACH, W_DEPTH, W_SUCCESS, W_ACTION, W_MISPLACED, W_ALIGN, W_YAW,
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

# Exploit B -- tunneling through the plate beside the pocket: 30 mm "deep" but
# 20 mm off-axis must not count as success, and must not raise max_depth.
_, md_off, succ_tunnel = step(torch.tensor([0.020, -0.225, 0.725]), torch.zeros(1))
check("deep but off-axis does NOT count as success", not bool(succ_tunnel))
check("off-gate depth does not raise max_depth", abs(_s(md_off)) < 1e-9, f"{_s(md_off):.3e}")
_, _, succ_real = step(torch.tensor([0.0, -0.225, 0.725]), torch.zeros(1))
check("deep and on-axis DOES count as success", bool(succ_real))
_, _, succ_edge = step(torch.tensor([0.0, -0.225, 0.730]), torch.zeros(1))
check("depth exactly at the 25 mm threshold counts", bool(succ_edge))

# Exploit C -- diving under the plate. Directly below the pocket the tip is
# CLOSER to the pocket entrance (55 mm) than at the home pose (~83 mm), and the
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
# The pocket bottom is a hard stop at 35 mm; 1 mm past the tolerance means
# through the plate.
_, _, succ_through = step(torch.tensor([0.0, -0.225, 0.718]), torch.zeros(1))
check("deeper than the blind pocket does NOT count as success", not bool(succ_through))
_, _, succ_bottom = step(torch.tensor([0.0, -0.225, 0.720]), torch.zeros(1))
check("resting on the pocket bottom still counts as success", bool(succ_bottom))

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
# The out-of-bounds price must include the yaw term: dying with a yaw error
# would otherwise be cheaper than standing still with it, and the exploit
# would reopen through the new term.
r_leave_yaw, _, _ = step(under, torch.zeros(1), remaining=REMAINING, phi=math.pi / 4)
r_stay_yaw, _, _ = step(under, torch.zeros(1), remaining=0.0, phi=math.pi / 4)
check("the out-of-bounds price includes the yaw term",
      abs(_s(r_leave_yaw) - _s(r_stay_yaw) * (1.0 + REMAINING)) < 1e-3,
      f"difference {abs(_s(r_leave_yaw) - _s(r_stay_yaw) * (1.0 + REMAINING)):.6f}")
# In bounds, the remaining-steps count must change nothing at all.
r_in_0, _, _ = step(torch.tensor([0.0, -0.225, 0.745]), torch.zeros(1), remaining=0.0)
r_in_200, _, _ = step(torch.tensor([0.0, -0.225, 0.745]), torch.zeros(1), remaining=REMAINING)
check("in bounds, remaining steps do not affect the reward", abs(_s(r_in_0) - _s(r_in_200)) < 1e-9)

# Exploit E -- parking the tip on the pocket with the peg lying across it. A
# distance-only approach term is maximised from any direction, so this pose
# scored ~0, the best value reachable without inserting, and every arm in a
# 1024-env run converged on it and stopped.
at_pocket = torch.tensor([0.0, -0.225, PLATE_TOP_Z])
r_flat, _, succ_flat = step(at_pocket, torch.zeros(1), alignment=0.0)     # peg across the hole
r_upright, _, _ = step(at_pocket, torch.zeros(1), alignment=1.0)          # peg into the hole
check("lying across the pocket scores worse than pointing into it", _s(r_flat) < _s(r_upright),
      f"flat {_s(r_flat):.4f} < upright {_s(r_upright):.4f}")
# The pose the run converged on must also lose to the honest approach from
# above, which is the move it refused to make.
above = torch.tensor([0.0, -0.225, PLATE_TOP_Z + 0.05])
r_above, _, _ = step(above, torch.zeros(1), alignment=1.0)
check("the flat parked pose loses to standing 50 mm above the pocket", _s(r_flat) < _s(r_above),
      f"flat {_s(r_flat):.4f} < above {_s(r_above):.4f}")
# Alignment must gate depth too: a peg across the hole cannot enter it.
_, md_flat, succ_flat_deep = step(torch.tensor([0.0, -0.225, 0.730]), torch.zeros(1), alignment=0.5)
check("misaligned depth does NOT count as success", not bool(succ_flat_deep))
check("misaligned depth pays no depth reward", abs(_s(md_flat)) < 1e-9, f"max_depth {_s(md_flat):.3e}")
# A tilt inside the loose 0.99 alignment gate must still pass: the alignment
# gate is a coarse anti-exploit guard (8.1 deg), the exact geometric gating
# is the corner test's job.
_, _, succ_tilt = step(torch.tensor([0.0, -0.225, 0.730]), torch.zeros(1), alignment=0.995)
check("a tilt inside the alignment gate still counts", bool(succ_tilt))
# Perfect alignment must cost nothing at all.
r_perfect, _, _ = step(above, torch.zeros(1), alignment=1.0)
check("perfect alignment carries no penalty",
      abs(_s(r_perfect) + W_APPROACH * 0.05) < 1e-6,
      f"{_s(r_perfect):.6f} vs {-W_APPROACH * 0.05:.6f}")

# Exploit F -- the approach term must reward getting closer, above the plate.
r_far, _, _ = step(torch.tensor([0.0, -0.125, 0.905]), torch.zeros(1))
r_near, _, _ = step(torch.tensor([0.0, -0.215, 0.775]), torch.zeros(1))
check("closer to the pocket scores higher off-gate", _s(r_near) > _s(r_far),
      f"near {_s(r_near):.4f} > far {_s(r_far):.4f}")

r_still, _, _ = step(torch.tensor([0.0, -0.225, 0.905]), torch.zeros(1))
r_moving, _, _ = step(torch.tensor([0.0, -0.225, 0.905]), torch.zeros(1), actions=torch.ones(1, 6))
check("action penalty subtracts w_action * ||a||^2", abs(_s(r_still - r_moving) - W_ACTION * 6) < 1e-5,
      f"delta {_s(r_still - r_moving):.6f} vs {W_ACTION * 6:.6f}")

# Exploit G -- yaw-parking (new with the square peg). Vertical, centred on
# the pocket, turned 45 deg: approach ~0, align ~0 -- under the round task's
# terms indistinguishable from the correct pre-insertion pose, yet insertion
# is geometrically impossible (diagonal 42.4 mm > 32 mm opening). wrist_3 is
# a pure yaw actuator, so correcting costs nothing before it pays; the yaw
# term is what makes the parked pose lose.
r_parked, _, _ = step(at_pocket, torch.zeros(1), phi=math.pi / 4)
r_aligned, _, _ = step(at_pocket, torch.zeros(1), phi=0.0)
check("yaw-parking at 45 deg costs w_yaw per step",
      abs(_s(r_aligned) - _s(r_parked) - W_YAW * 1.0) < 1e-6,
      f"delta {_s(r_aligned) - _s(r_parked):.6f} vs {W_YAW:.6f}")
r_inserting, _, _ = step(torch.tensor([0.0, -0.225, 0.745]), torch.zeros(1), phi=0.0)
check("the parked pose loses to turned-and-inserting", _s(r_parked) < _s(r_inserting),
      f"parked {_s(r_parked):.4f} < inserting {_s(r_inserting):.4f}")
# 45 deg has all four corners outside the pocket at ANY offset: never success.
_, md_45, succ_45 = step(torch.tensor([0.0, -0.225, 0.730]), torch.zeros(1), phi=math.pi / 4)
check("45 deg yaw does NOT count as success at depth", not bool(succ_45))
check("45 deg yaw pays no depth reward", abs(_s(md_45)) < 1e-9, f"max_depth {_s(md_45):.3e}")
# The corner test subsumes offset AND yaw jointly: 0.9/0.9 mm of offset fits
# at phi = 0 (corner Chebyshev 15.9 < 16 mm) but not with 10 deg of yaw on
# top (corner extent 15 * (cos + sin) + 0.9 = 18.3 mm).
_, _, succ_corner = step(torch.tensor([0.0009, -0.225 + 0.0009, 0.730]), torch.zeros(1), phi=0.0)
check("0.9/0.9 mm offset at phi=0 counts as success", bool(succ_corner))
_, _, succ_corner_yaw = step(
    torch.tensor([0.0009, -0.225 + 0.0009, 0.730]), torch.zeros(1), phi=math.radians(10.0)
)
check("the same offset with 10 deg yaw does NOT count", not bool(succ_corner_yaw))
# The free-yaw window at zero offset is +-3.96 deg: just inside passes, just
# outside does not.
_, _, succ_in_window = step(torch.tensor([0.0, -0.225, 0.730]), torch.zeros(1), phi=math.radians(3.5))
check("3.5 deg yaw (inside the window) counts", bool(succ_in_window))
_, _, succ_out_window = step(torch.tensor([0.0, -0.225, 0.730]), torch.zeros(1), phi=math.radians(4.5))
check("4.5 deg yaw (outside the window) does NOT count", not bool(succ_out_window))
# C4 invariance: orientations 90 deg apart are the same task state, so the
# reward must be identical -- the policy is never asked to prefer one of the
# four insertable orientations.
r_phi, _, _ = step(torch.tensor([0.0, -0.225, 0.745]), torch.zeros(1), phi=0.2)
r_phi90, _, _ = step(torch.tensor([0.0, -0.225, 0.745]), torch.zeros(1), phi=0.2 + math.pi / 2)
check("phi and phi + 90 deg earn the identical reward", abs(_s(r_phi) - _s(r_phi90)) < 1e-9,
      f"difference {abs(_s(r_phi) - _s(r_phi90)):.3e}")

tips = torch.tensor([[0.0, -0.225, 0.745], [0.02, -0.225, 0.760], [0.0, -0.225, 0.725]])
depths = PLATE_TOP_Z - tips[:, 2]
# Batched gate mirror at phi = 0: corner Chebyshev collapses to the tip's
# Chebyshev offset plus the half-side. Built as a float list so both real
# torch and the numpy shim yield a boolean tensor.
gates = torch.tensor([
    1.0 if (
        max(abs(float(tips[i][0]) - float(entrance[0])), abs(float(tips[i][1]) - float(entrance[1])))
        + PEG_SIDE / 2.0 < POCKET_SIDE / 2.0
        and float(depths[i]) <= POCKET_DEPTH + 0.001
    ) else 0.0
    for i in range(3)
]) > 0.5
below = tips[:, 2] < PLATE_BOTTOM_Z
rb, mb, sb = compute_rewards(
    tips, entrance, depths, gates, below, torch.ones(3), torch.ones(3), torch.zeros(3),
    torch.zeros(3, 6), torch.zeros(3),
    SUCCESS_DEPTH, W_APPROACH, W_DEPTH, W_SUCCESS, W_ACTION, W_MISPLACED, W_ALIGN, W_YAW,
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



