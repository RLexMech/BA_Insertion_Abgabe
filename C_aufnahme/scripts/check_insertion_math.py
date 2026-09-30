# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Offline check of ``insertion_math.py`` -- the real task's pure math.

WHAT IT CHECKS
--------------
``insertion_math.py`` holds the reward, observation and termination algebra
for the UR5e insertion (D-107, D-109, D-113, D-114). It imports only ``math``
and ``torch``, so this check can execute the WHOLE FILE under the numpy
stand-in (``scripts/tools/torch_shim.py``) -- no Isaac, no torch, no GPU.

That is the one difference to the older ``check_demo_reward_math.py``, which
cherry-picks three functions out of the 1781-line env by AST name. A fixed
name list silently stops covering whatever is added next; running the whole
file cannot.

THE COUNTER-PROOF (D-080)
-------------------------
``--self-test`` is not a second opinion on the same source. It breaks the
source in exactly one way per mutation and requires EXACTLY the named checks
to flip. A mutation that flips nothing proves the checks cannot fail; a
mutation that flips extra checks proves they are not surgical. Both are
reported as failures of the check, not of the math.

    python scripts/check_insertion_math.py
    python scripts/check_insertion_math.py --self-test
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import re
import sys

MATH_SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "source"
    / "insertion"
    / "insertion"
    / "tasks"
    / "direct"
    / "insertion"
    / "insertion_math.py"
)

_SHIM_PATH = pathlib.Path(__file__).resolve().parent / "tools" / "torch_shim.py"
_spec = importlib.util.spec_from_file_location("torch_shim", _SHIM_PATH)
torch_shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(torch_shim)
torch, _IS_REAL_TORCH = torch_shim.load()
# insertion_math.py does `import torch` at the top. Under the stand-in there is
# no such module, so publish the stand-in under that name -- the import machinery
# looks in sys.modules first. Deliberately not setdefault: if real torch is
# installed it is already there and load() returned it.
sys.modules.setdefault("torch", torch)


# ---------------------------------------------------------------------------
# The constants under test. They are NOT defaults of the math module -- the
# module takes every number as an argument and owns none of them. These are
# the D-109 values, written here once so a check reads like the decision.
# ---------------------------------------------------------------------------

# Kernel widths, SOLVED via dm_control tolerance() (Tassa et al. 2018):
# a = arccosh(1/0.1)/margin = 2.9932/margin. D-109 point (9).
# THE COARSE MARGIN IS A REACH, NOT A TOLERANCE (2026-09-01). It was 10 mm,
# typed here as "the start scatter", and the concept stream refuted BOTH halves
# of that: dm_control's margin is the decay reach outward from the tolerance
# zone, and the start scatter is not what the coarse kernel was ever measuring.
# The margin is now the MEASURED SDF distance at the workcell home pose
# (RT-119, 173.21 mm), so a = acosh(10)/0.17321 = 17.28 /m. Home of the
# decision: inbox entry "Reward-Ueberarbeitung" 2026-09-01
# (p1-konzept-messung), point (1). Typed here by hand, like every other
# constant in this block, so the check does not assert the source against
# itself.
A_COARSE, B_COARSE = 17.28, 2.0        # margin 173.21 mm (the measured travel reach)
A_MID, B_MID = 998.0, 2.0              # margin 3 mm     (success band)
# THE CROSS PLAY DOUBLED, 2026-08-28. The part body is 90.00 mm across (read
# out of the part CAD, with the 143.50 mm long-axis control), not the 90.3 mm
# caliper reading of D-057, so the play is 0.5876 mm and not 0.2876 mm. Home of
# the fact: docs/decisions_inbox.md, entry "The part body is 90.00 mm across,
# not 90.3" (stream szene).
#
# NO LONGER PROVISIONAL since 2026-08-30. The comment here used to say the
# re-derivation "belongs to the concept stream and has no D-number yet"; it has
# been taken. Branch p1-konzept-messung, commit 77f5b3c, entry "The D-121
# recomputation: a_fine = 5094, SAPU threshold = 0.2938 mm, sweep updated, the
# 2048 justification re-anchored". Both RULES survived the doubled input -- the
# fine kernel's margin is still the tightest tolerance of the task, and the
# SAPU threshold is still half the play -- so it really is the same formula
# with the corrected input.
#
# THESE TWO LITERALS STAY LITERALS. The env now derives both from PLAY_X
# (``insertion_tasks_cfg.KERNEL_A_FINE`` / ``INTERPEN_THRESH``). A check that
# imported those would assert them against themselves; typing them here is what
# makes this a check. Same reason the fine margin below is its own literal.
A_FINE, B_FINE = 5094.0, 0.0           # margin 0.5876 mm (cross play)
W_ENGAGED, W_SUCCESS = 1.0, 1.0        # D-109 point (5): weight 1.0 each
INTERPEN_THRESH = 0.0002938            # SAPU, D-109 point (10): 0.2938 mm

# STILL NOT A DECIDED VALUE, and the fact that the env now carries the same
# number changes nothing here. On 2026-08-30 the user set ``abort_payment`` to
# -1.0 as a LOUD PLACEHOLDER (RL_PLACEHOLDERS, printed by every run); D-114's
# magnitude is still [open] and still falls together with F_max at the force
# measurement. This check must therefore never assert a number for it -- only
# properties that hold for ANY negative payment. -1.0 is the probe value, and
# it is a coincidence that the placeholder picked the same one.
PROBE_ABORT_PENALTY = -1.0

# The three 2026-09-01 terms are exercised with PROBE values, never with the
# config's own: this module owns no numbers and neither does this block. The
# time penalty's probe is written as -1/256 because the RULE is -1/T
# (Brahmbhatt et al., ICRA 2023, arXiv:2301.12587 Sec. III) and 256 is the
# decided horizon -- but nothing here asserts that the config carries either;
# the guard in insertion_env_cfg.py does that, and its own checks are below.
PROBE_TIME_PENALTY = -1.0 / 256.0
PROBE_ACTION_RATE_SCALE = 0.1
# D-165: the proxy's reward_w_depth, per metre of NEW gated max depth. A probe
# value like the two above -- the checks assert the FORM (per metre, SAPU-
# scaled, outside the lump), not the size, which is a [proxy] placeholder.
PROBE_W_PROGRESS = 100.0
PROBE_EPISODE_STEPS = 256
# RT-171 (2026-09-06): the alignment shaping row. Probe values by the same
# rule -- the checks assert the FORM (sech kernel, Ng row, terminal
# convention, place in the stack, telescoping, discounted ordering). The
# margin literal IS typed here on purpose: the one config check compares the
# task file's text against it, so it must not be imported. gamma 0.99 is the
# probe discount; the config reads its own from agents/rsl_rl_ppo_cfg.py.
PROBE_W_TILT = 1.0
PROBE_TILT_MARGIN_RAD = math.radians(15.34)
PROBE_A_TILT = math.acosh(10.0) / PROBE_TILT_MARGIN_RAD
PROBE_GAMMA = 0.99


def load_math(mutations: tuple[tuple[str, str], ...] = ()) -> dict:
    """Execute the whole math module under the stand-in, optionally mutated.

    Each mutation is (old, new) applied to the SOURCE TEXT and required to
    match exactly once -- a mutation that silently matches nothing would make
    the counter-proof vacuous, which is the failure mode this whole harness
    exists to catch.
    """
    src = MATH_SRC.read_text(encoding="utf-8")
    for old, new in mutations:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"mutation {old!r} matches {n} times in {MATH_SRC.name}, expected 1")
        src = src.replace(old, new)
    # AFTER the mutations, so their counts still address the real file. Without
    # this the counter-proof is blind under real torch: jit.script recompiles
    # from the file on disk and the mutated string never reaches the function
    # (measured RT-108-gate, 2026-08-31). See torch_shim.strip_jit_script.
    src = torch_shim.strip_jit_script(src)
    ns: dict = {"torch": torch, "math": math, "__name__": "insertion_math"}
    exec(compile(src, str(MATH_SRC), "exec"), ns)  # noqa: S102 - the point of the file
    return ns


CFG_SRC = MATH_SRC.parent / "insertion_env_cfg.py"


def load_cfg_guard(mutations: tuple[tuple[str, str], ...] = ()) -> dict:
    """Extract the UNSET guard from insertion_env_cfg.py.

    Cherry-picking by AST name here, not whole-file exec: that module imports
    ``isaaclab``, which the laptop does not have. Only six top-level names
    are wanted and they touch nothing but builtins, so this is the narrow case
    the technique fits -- unlike insertion_math.py, where the whole point was
    to stop cherry-picking.
    """
    import ast

    src = CFG_SRC.read_text(encoding="utf-8")
    for old_text, new_text in mutations:
        n = src.count(old_text)
        if n != 1:
            raise SystemExit(f"mutation {old_text!r} matches {n} times in {CFG_SRC.name}")
        src = src.replace(old_text, new_text)
    wanted = {"RL_PENDING", "RL_PENDING_MARKS", "RL_PLACEHOLDERS",
              "CURRICULUM_PENDING", "RlConfigIncomplete", "validate_rl_config",
              "CONTROL_MODES", "resolve_control_mode",
              "resolve_start_lateral_offset"}
    ns: dict = {}
    for node in ast.parse(src, str(CFG_SRC)).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            name = targets[0] if targets else None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name in wanted:
            ns[name] = None
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(CFG_SRC), "exec"), ns)
    missing = wanted - {k for k, v in ns.items() if v is not None}
    if missing:
        raise SystemExit(f"could not extract from {CFG_SRC.name}: {sorted(missing)}")
    return ns


def _f(x) -> float:
    """First element as a Python float, for torch tensors and ndarrays alike."""
    import numpy as np

    return float(np.asarray(x).reshape(-1)[0])


def _arr(x):
    """As a plain numpy array, for torch tensors and ndarrays alike."""
    import numpy as np

    return np.asarray(x)


def _maxabs(a, b) -> float:
    """Largest absolute difference between two arrays, broadcast if needed."""
    import numpy as np

    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _np_eye(n: int = 1):
    """``n`` stacked 3x3 identity matrices, the reference for a pure rotation."""
    import numpy as np

    return np.broadcast_to(np.eye(3), (n, 3, 3))


def _np_diag(x: float, y: float, z: float):
    """One batched 3x3 diagonal matrix, for the known half-turn rotations."""
    import numpy as np

    return np.diag([x, y, z]).reshape(1, 3, 3)


def _np_row(x: float, y: float, z: float):
    """One batched 3-vector, for comparing a single extracted axis."""
    import numpy as np

    return np.array([[x, y, z]])


def _np_row4(w: float, x: float, y: float, z: float):
    """One batched wxyz quaternion, for comparing against a known rotation."""
    import numpy as np

    return np.array([[w, x, y, z]])


# ===========================================================================
#  The checks. Each returns True when the math is right. Names are the
#  contract the mutation table below refers to.
# ===========================================================================

def _rewards(ns, sdf, interpen=0.0, engaged=False, success=False, force_abort=False,
             action_rate=0.0, steps_remaining=0.0,
             time_penalty=0.0, action_rate_scale=0.0,
             depth_progress=0.0, w_progress=0.0,
             tilt_phi=0.0, tilt_phi_prev=0.0, gamma=0.0):
    """One call of compute_rewards_insertion with the D-109 constants.

    The RT-171 shaping row (2026-09-06) follows the same rule: both potentials
    0 and gamma 0 unless a check hands them in, so the row is 0 - 0 and every
    older number is untouched.

    THE THREE 2026-09-01 TERMS DEFAULT TO OFF -- time penalty 0, action-rate
    scale 0, zero steps remaining -- and that is deliberate. The checks written
    against the D-109 shaping half still say exactly what they said before the
    revision, and every one of the new terms is asserted in its own block with
    its probe value handed in explicitly. A default that quietly folded the
    time penalty into every number would make a broken term look like a
    changed bonus weight.
    """
    one = torch.tensor([1.0])
    zero = torch.tensor([0.0])
    return ns["compute_rewards_insertion"](
        torch.tensor([float(sdf)]),
        torch.tensor([float(interpen)]),
        (one if engaged else zero) > 0.5,
        (one if success else zero) > 0.5,
        (one if force_abort else zero) > 0.5,
        torch.tensor([float(action_rate)]),
        torch.tensor([float(steps_remaining)]),
        torch.tensor([float(depth_progress)]),
        torch.tensor([float(tilt_phi)]),
        torch.tensor([float(tilt_phi_prev)]),
        A_COARSE, B_COARSE, A_MID, B_MID, A_FINE, B_FINE,
        W_ENGAGED, W_SUCCESS, float(w_progress), INTERPEN_THRESH, PROBE_ABORT_PENALTY,
        float(time_penalty), float(action_rate_scale),
        float(gamma),
    )


def build_checks(ns, cfg_mutations: tuple = ()) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    def check(name: str, cond, detail: str = "") -> None:
        out.append((name, bool(cond), detail))

    # =====================================================================
    #  Pose algebra (own construction -- see the module header)
    # =====================================================================
    # Checked against KNOWN rotations, not against a second copy of the same
    # expansion. Two copies agreeing proves only that they were copied; the
    # 2026-07-26 transposed-matrix defect in the proxy survived exactly that
    # kind of agreement until an identity test was written.
    axes_from_quat = ns["axes_from_quat"]
    quat_mul = ns["quat_mul"]
    part_tip_pose = ns["part_tip_pose"]

    R2 = math.sqrt(0.5)
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_x180 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    q_z90 = torch.tensor([[R2, 0.0, 0.0, R2]])
    q_y90 = torch.tensor([[R2, 0.0, R2, 0.0]])

    check("the identity quaternion gives the identity matrix",
          _maxabs(axes_from_quat(q_id), _np_eye()) < 1e-12,
          f"{_maxabs(axes_from_quat(q_id), _np_eye()):.3e}")
    check("180 deg about x flips y and z",
          _maxabs(axes_from_quat(q_x180), _np_diag(1.0, -1.0, -1.0)) < 1e-12)
    # 90 deg about z sends the body x-axis to +y. Column 0 IS that axis; if the
    # three axes were stacked as ROWS instead, column 0 would read (0, -1, 0).
    check("the axes are COLUMNS, not rows",
          _maxabs(_arr(axes_from_quat(q_z90))[:, :, 0], _np_row(0.0, 1.0, 0.0)) < 1e-12,
          str(_arr(axes_from_quat(q_z90))[:, :, 0].round(6).tolist()))
    check("90 deg about y sends body z to +x",
          _maxabs(_arr(axes_from_quat(q_y90))[:, :, 2], _np_row(1.0, 0.0, 0.0)) < 1e-12)
    # Orthonormality over a fixed set -- fixed, not random, because a check
    # that cannot be re-run bit-for-bit is not evidence (cfg.seed rule).
    q_batch = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [R2, 0.0, R2, 0.0],
        [0.0, R2, 0.0, R2],
    ])
    rot = axes_from_quat(q_batch)
    check("every rotation matrix is orthonormal",
          _maxabs(torch.matmul(rot.transpose(1, 2), rot), _np_eye(4)) < 1e-12,
          f"{_maxabs(torch.matmul(rot.transpose(1, 2), rot), _np_eye(4)):.3e}")
    check("axes_from_quat is batched over envs", tuple(rot.shape) == (4, 3, 3),
          str(tuple(rot.shape)))

    check("the identity quaternion is neutral in the product",
          _maxabs(quat_mul(q_z90, q_id), q_z90) < 1e-12)
    # The whole point of quat_mul: composing two quaternions must equal
    # composing their matrices, or a yawed AND tilted pocket lands wrong.
    check("quat_mul composes the same way the matrices do",
          _maxabs(axes_from_quat(quat_mul(q_z90, q_y90)),
                  torch.matmul(axes_from_quat(q_z90), axes_from_quat(q_y90))) < 1e-12)

    # -- part_tip_pose ------------------------------------------------------
    origins = torch.tensor([[0.0, 0.0, 0.0]])
    entrance = torch.tensor([[0.0, 0.0, 0.0]])
    eye1 = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    no_offset = torch.tensor([0.0, 0.0, 0.0])
    down_offset = torch.tensor([0.0, 0.0, -0.1])

    seated = part_tip_pose(torch.tensor([[0.0, 0.0, 0.0]]), q_id, no_offset,
                           origins, entrance, eye1)
    check("a part exactly at the opening reports zero depth",
          abs(_f(seated[1])) < 1e-12, f"{_f(seated[1]):.3e}")
    check("a part exactly at the opening reports zero offset",
          _maxabs(seated[0], _np_row(0.0, 0.0, 0.0)) < 1e-12)

    inserted = part_tip_pose(torch.tensor([[0.0, 0.0, -0.005]]), q_id, no_offset,
                             origins, entrance, eye1)
    check("depth is POSITIVE into the pocket",
          abs(_f(inserted[1]) - 0.005) < 1e-12, f"{_f(inserted[1]):.6f}")

    # The tip is the body pose PLUS the offset rotated by the body quaternion.
    offset_only = part_tip_pose(torch.tensor([[0.0, 0.0, 0.0]]), q_id, down_offset,
                                origins, entrance, eye1)
    check("the tip offset moves the measured point",
          abs(_f(offset_only[1]) - 0.1) < 1e-12, f"{_f(offset_only[1]):.6f}")
    # ... and it is rotated, not added in the env frame: 180 deg about x turns
    # a -z offset into +z, so the same part now reads 0.1 m ABOVE the opening.
    flipped = part_tip_pose(torch.tensor([[0.0, 0.0, 0.0]]), q_x180, down_offset,
                            origins, entrance, eye1)
    check("the tip offset rotates with the part",
          abs(_f(flipped[1]) + 0.1) < 1e-12, f"{_f(flipped[1]):.6f}")

    shifted = part_tip_pose(torch.tensor([[1.0, 2.0, 0.0]]), q_id, no_offset,
                            torch.tensor([[1.0, 2.0, 0.0]]), entrance, eye1)
    check("the env origin is subtracted",
          _maxabs(shifted[0], _np_row(0.0, 0.0, 0.0)) < 1e-12)

    # A pocket yawed 90 deg about z: its x-axis is env +y, its y-axis is env -x.
    # Columns of pocket_rot are those axes in the env frame.
    yawed = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    off_x = part_tip_pose(torch.tensor([[0.01, 0.0, 0.0]]), q_id, no_offset,
                          origins, entrance, yawed)
    # 10 mm along env +x is 10 mm along the pocket's MINUS y.
    check("the offset is projected into the pocket frame",
          _maxabs(off_x[0], _np_row(0.0, -0.01, 0.0)) < 1e-12,
          str(_arr(off_x[0]).round(6).tolist()))
    check("the part axis is projected into the pocket frame",
          _maxabs(off_x[2], _np_row(0.0, -1.0, 0.0)) < 1e-12,
          str(_arr(off_x[2]).round(6).tolist()))

    # -- relative_pose: what lets the SDF query run against ONE static mesh ---
    relative_pose = ns["relative_pose"]
    quat_conjugate = ns["quat_conjugate"]

    check("conjugating twice is the identity",
          _maxabs(quat_conjugate(quat_conjugate(q_z90)), q_z90) < 1e-15)
    check("the conjugate keeps w and flips the vector part",
          _maxabs(quat_conjugate(q_z90), _np_row4(R2, 0.0, 0.0, -R2)) < 1e-15,
          str(_arr(quat_conjugate(q_z90)).round(6).tolist()))

    # A pose relative to ITSELF is the identity pose. This is the check that
    # the whole batched-SDF substitution stands on: if it did not hold, moving
    # the mesh to the goal pose and moving the points into the goal frame would
    # not be the same measurement.
    p_any = torch.tensor([[0.3, -0.2, 0.7]])
    same = relative_pose(p_any, q_z90, p_any, q_z90)
    check("a pose relative to itself has zero offset",
          _maxabs(same[0], _np_row(0.0, 0.0, 0.0)) < 1e-15,
          str(_arr(same[0]).round(9).tolist()))
    check("a pose relative to itself is the identity rotation",
          _maxabs(same[1], _np_row4(1.0, 0.0, 0.0, 0.0)) < 1e-15,
          str(_arr(same[1]).round(9).tolist()))

    zero_p = torch.tensor([[0.0, 0.0, 0.0]])
    from_id = relative_pose(zero_p, q_id, p_any, q_z90)
    check("relative to the identity frame is the pose itself",
          _maxabs(from_id[0], p_any) < 1e-15 and _maxabs(from_id[1], q_z90) < 1e-15)

    # Frame A yawed 90 deg at the origin, B one metre along env +x: in A's own
    # frame that is one metre along MINUS y.
    rel = relative_pose(zero_p, q_z90, torch.tensor([[1.0, 0.0, 0.0]]), q_id)
    check("the relative offset is expressed in frame A",
          _maxabs(rel[0], _np_row(0.0, -1.0, 0.0)) < 1e-12,
          str(_arr(rel[0]).round(6).tolist()))
    # ... and B's orientation seen from A is A's yaw undone.
    check("the relative rotation is A's rotation undone",
          _maxabs(axes_from_quat(rel[1]),
                  axes_from_quat(quat_conjugate(q_z90))) < 1e-12)

    # Two DIFFERENT, non-commuting rotations, compared against an independent
    # reference: the matrix product. Every check above uses the identity or a
    # pose against itself, where conj(a)*b and b*conj(a) agree -- so none of
    # them can see a swapped composition order.
    ab = relative_pose(zero_p, q_z90, zero_p, q_y90)
    check("the relative rotation matches R(a)^T R(b)",
          _maxabs(axes_from_quat(ab[1]),
                  torch.matmul(axes_from_quat(q_z90).transpose(1, 2),
                               axes_from_quat(q_y90))) < 1e-12)

    # The substitution itself, stated as an identity over a point: rotating a
    # point into the goal frame and comparing there must equal comparing in the
    # world frame. Distances are what the SDF measures, so it is enough that
    # the offset LENGTH survives.
    goal_p, goal_q = torch.tensor([[0.10, 0.20, 0.30]]), q_y90
    curr_p, curr_q = torch.tensor([[0.11, 0.19, 0.32]]), q_z90
    r_pos, _r_quat = relative_pose(goal_p, goal_q, curr_p, curr_q)
    world_gap = float(((_arr(curr_p) - _arr(goal_p)) ** 2).sum() ** 0.5)
    frame_gap = float((_arr(r_pos) ** 2).sum() ** 0.5)
    check("the frame change preserves distance (the SDF substitution)",
          abs(world_gap - frame_gap) < 1e-12, f"{world_gap:.9f} vs {frame_gap:.9f}")

    two = part_tip_pose(
        torch.tensor([[0.0, 0.0, -0.005], [0.0, 0.0, -0.010]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        no_offset,
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                      [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
    )
    check("part_tip_pose is batched over envs",
          tuple(two[0].shape) == (2, 3) and tuple(two[1].shape) == (2,),
          f"{tuple(two[0].shape)} {tuple(two[1].shape)}")

    # THE PER-ENV TIP OFFSET (D-183). The grasp offset turns with the part, so
    # the observation hands this function ONE offset PER ENV instead of the
    # single (3,) constant every other caller passes. Checked here before the
    # env is wired, because the naive form does not fail loudly in the general
    # case: ``matmul`` of an (N, 3, 3) against a bare (N, 3) reads the second
    # operand as a MATRIX -- it RAISES for N != 3 and silently returns a
    # (3, 3, 3) block for N == 3. Both shapes must work, and the (3,) path
    # must stay bit-identical, or every existing caller changes meaning.
    def _tp(off, n=2):
        """part_tip_pose at the origin, identity pose and pocket, n envs."""
        z = torch.tensor([[0.0, 0.0, 0.0]] * n)
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)
        eye = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * n)
        return part_tip_pose(z, q, off, z, z, eye)

    try:
        _pe = _tp(torch.tensor([[0.0, 0.0, -0.1], [0.0, 0.0, -0.1]]))
        _pe_ok, _pe_detail = tuple(_pe[0].shape) == (2, 3), str(tuple(_pe[0].shape))
    except Exception as exc:  # noqa: BLE001
        _pe_ok, _pe_detail = False, f"raised {type(exc).__name__}"
    check("part_tip_pose takes ONE tip offset per env, not only a shared (3,)",
          _pe_ok, _pe_detail)
    # ... and each env gets ITS OWN row. A form that broadcast row 0 across the
    # batch would pass the shape check above and hand every env the same grasp
    # error, which is no grasp scatter at all.
    try:
        _sp = _tp(torch.tensor([[0.003, 0.0, 0.0], [-0.003, 0.0, 0.0]]))
        _sp_ok = (abs(_arr(_sp[0])[0, 0] - 0.003) < 1e-12
                  and abs(_arr(_sp[0])[1, 0] + 0.003) < 1e-12)
        _sp_detail = str(_arr(_sp[0]).round(6).tolist())
    except Exception as exc:  # noqa: BLE001
        _sp, _sp_ok, _sp_detail = None, False, f"raised {type(exc).__name__}"
    check("a per-env tip offset moves each env by ITS OWN row", _sp_ok, _sp_detail)
    # D-183's third offline gate: the grasp offset must not reach the YAW the
    # policy reads. ``yaw_cos_sin`` consumes ``x_axis``, and ``x_axis`` is
    # built from the part quaternion and the pocket frame alone -- the tip
    # offset never enters it. That is a property of THIS function, not of the
    # call site, so it is asserted here rather than by reading the env.
    check("the tip offset does not move the part's x-axis (yaw_cos_sin is untouched)",
          bool(_sp_ok and _maxabs(_sp[2], _np_row(1.0, 0.0, 0.0)) < 1e-12),
          str(_arr(_sp[2]).round(6).tolist()) if _sp_ok else "not reached")

    # D-183 CLAIMED "the (3,) path stays bit-identical" and NOTHING tested it
    # until now (critic finding 13, 2026-09-12). Two different claims hide in
    # that sentence, so there are two checks:
    #
    # (1) THE TWO CALL FORMS AGREE. The shared (3,) offset and an (N, 3) whose
    #     rows are all that same offset must give the same numbers -- every
    #     caller that keeps passing the (3,) constant depends on it.
    # (2) THE NEW LINE AGREES WITH THE OLD ONE. The reshape pair replaced a
    #     bare ``torch.matmul(part_axes, tip_offset_local)``, and for the (3,)
    #     form that old expression is still computable HERE (matmul promotes
    #     a vector second operand and drops the added dimension again), so the
    #     two forms can be compared instead of asserted. The reference is
    #     built from the module's OWN ``axes_from_quat`` and
    #     ``rotate_into_frame``: damage shared by both sides cancels, and only
    #     the one line D-183 changed is under test.
    #
    # NOT a tolerance. "Bit-identical" means ``==``, and that is what is
    # written. WHAT THIS PROVES AND WHAT IT DOES NOT: the comparison runs in
    # float64 -- under the numpy stand-in, and under real torch too, because
    # ``torch_shim.load`` sets the default dtype to float64. So it proves the
    # two are the same EXPRESSION. It does not prove that torch's float32
    # CUDA kernels round the (3,)-vector dispatch and the (N, 3, 1)-matrix
    # dispatch to the same bits; that has no offline form and stays the
    # training PC's answer.
    # The pose is deliberately generic -- rotated part, yawed AND tilted
    # pocket, non-zero env origin and entrance -- because at the identity the
    # offset passes through untouched and the comparison proves nothing.
    rotate_into_frame = ns["rotate_into_frame"]
    bi_pos = torch.tensor([[0.31, -0.22, 0.74], [0.30, -0.20, 0.70]])
    bi_quat = torch.tensor([[0.5, 0.5, 0.5, 0.5], [R2, 0.0, R2, 0.0]])
    bi_org = torch.tensor([[1.0, -2.0, 0.5], [1.0, -2.0, 0.5]])
    bi_ent = torch.tensor([[0.10, -0.20, 0.30], [0.12, -0.18, 0.28]])
    bi_rot = axes_from_quat(torch.tensor([[R2, 0.0, 0.0, R2], [0.5, 0.5, 0.5, 0.5]]))
    bi_off3 = torch.tensor([0.02, -0.013, 0.152])
    bi_offn = torch.tensor([[0.02, -0.013, 0.152], [0.02, -0.013, 0.152]])

    bi_shared = part_tip_pose(bi_pos, bi_quat, bi_off3, bi_org, bi_ent, bi_rot)
    try:
        bi_perenv = part_tip_pose(bi_pos, bi_quat, bi_offn, bi_org, bi_ent, bi_rot)
        # All three returns, not only the position: depth and x_axis are what
        # the termination and the yaw encoding read.
        _bi_ok = all(bool((_arr(a) == _arr(b)).all())
                     for a, b in zip(bi_shared, bi_perenv))
        _bi_detail = f"max |d| {_maxabs(bi_shared[0], bi_perenv[0]):.3e}"
    except Exception as exc:  # noqa: BLE001
        _bi_ok, _bi_detail = False, f"raised {type(exc).__name__}"
    check("a (3,) offset and its (N, 3) broadcast give bit-identical results",
          _bi_ok, _bi_detail)

    _old_tip = bi_pos + torch.matmul(axes_from_quat(bi_quat), bi_off3) - bi_org
    _old_rel = rotate_into_frame(bi_rot, _old_tip - bi_ent)
    check("the (3,) path is bit-identical to the pre-D-183 bare matmul",
          bool((_arr(bi_shared[0]) == _arr(_old_rel)).all()),
          f"max |d| {_maxabs(bi_shared[0], _old_rel):.3e}")

    # -- seated_goal_pose: M2.4b step 2, the SDF reward's target -------------
    # Checked as a ROUND TRIP through part_tip_pose, not against a second copy
    # of the same subtraction: the goal pose fed back through the forward
    # function must read exactly seat depth and zero lateral offset.
    #
    # The SEAT ROTATION is part of the contract since RT-102 (2026-08-30):
    # the tool_link frame hangs INVERTED (a 180 deg turn about pocket y,
    # MEASURED), and the first build, which returned the fixture quaternion
    # alone, put the goal flange 125 mm under the entrance and read an sdf
    # mean-outside distance of 25.76 mm at a physically seated pose.
    seated_goal_pose = ns["seated_goal_pose"]
    SEAT_D = 0.036  # probe value; the env's number is POCKET_SEAT_DEPTH
    goal_entrance = torch.tensor([[0.10, -0.20, 0.30]])
    # A LATERAL component on purpose: with a z-only offset a pure yaw cannot
    # tell a rotated offset from an unrotated one.
    goal_off = torch.tensor([0.02, 0.0, -0.1])
    # The measured seat rotation, typed here as a LITERAL on purpose (the env
    # derives it from SEATED_TOOL_QUAT_LOCAL; asserting the import against
    # itself would be blind): 180 deg about y, wxyz.
    q_seat = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    g_pos, g_quat = seated_goal_pose(goal_entrance, q_id, q_seat, goal_off, SEAT_D)
    # R_y(180) sends (0.02, 0, -0.1) to (-0.02, 0, 0.1), so the goal is
    # entrance - (0,0,0.036) - (-0.02, 0, 0.1).
    check("an untilted goal pose subtracts seat depth and the FLIPPED tip offset",
          _maxabs(g_pos, _np_row(0.12, -0.20, 0.164)) < 1e-12,
          str(_arr(g_pos).round(6).tolist()))
    check("the goal orientation composes the seated tool rotation",
          _maxabs(g_quat, q_seat) < 1e-15,
          str(_arr(g_quat).round(6).tolist()))
    # THE RT-102 FALSIFIER, as a check: with the REAL offset sign (+z along
    # the tool axis, tool z pointing down) the goal FLANGE must land ABOVE
    # the entrance. The un-flipped build put it 0.188 m lower.
    g3_pos, _g3_quat = seated_goal_pose(
        goal_entrance, q_id, q_seat, torch.tensor([0.0, 0.0, 0.152]), SEAT_D)
    check("with the real offset sign the goal flange sits ABOVE the entrance",
          float(_arr(g3_pos)[0][2]) > float(_arr(goal_entrance)[0][2]),
          f"flange z {float(_arr(g3_pos)[0][2]):.4f} vs entrance z 0.3000")

    gy_pos, gy_quat = seated_goal_pose(goal_entrance, q_z90, q_seat, goal_off, SEAT_D)
    rt = part_tip_pose(gy_pos, gy_quat, goal_off, origins, goal_entrance,
                       axes_from_quat(q_z90))
    check("the goal pose reports exactly seat depth through part_tip_pose",
          abs(_f(rt[1]) - SEAT_D) < 1e-12, f"{_f(rt[1]):.9f}")
    lat = _arr(rt[0])[0]
    check("the goal tip offset rotates with the pocket",
          abs(float(lat[0])) < 1e-12 and abs(float(lat[1])) < 1e-12,
          str(_arr(rt[0]).round(6).tolist()))

    q_yt = quat_mul(q_z90, q_y90)
    gt_pos, gt_quat = seated_goal_pose(goal_entrance, q_yt, q_seat, goal_off, SEAT_D)
    rt2 = part_tip_pose(gt_pos, gt_quat, goal_off, origins, goal_entrance,
                        axes_from_quat(q_yt))
    check("a yawed and tilted pocket still seats the goal exactly",
          _maxabs(rt2[0], _np_row(0.0, 0.0, -SEAT_D)) < 1e-12,
          str(_arr(rt2[0]).round(6).tolist()))

    g2_pos, g2_quat = seated_goal_pose(
        torch.tensor([[0.0, 0.0, 0.0], [0.10, -0.20, 0.30]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [R2, 0.0, 0.0, R2]]),
        q_seat, goal_off, SEAT_D)
    check("seated_goal_pose is batched over envs",
          tuple(g2_pos.shape) == (2, 3) and tuple(g2_quat.shape) == (2, 4),
          f"{tuple(g2_pos.shape)} {tuple(g2_quat.shape)}")

    squash = ns["squash"]
    kernel_sum = ns["kernel_sum"]
    sapu = ns["sapu_reward_scale"]

    # -- the squashing function itself (D-109 point (6)) --------------------
    # It REPLACES IndustReal's -log mapping rather than guarding it, so the
    # one thing it must do is be finite at a perfect seat.
    at_zero = _f(squash(torch.tensor([0.0]), A_MID, B_MID))
    check("squash is finite at x = 0", math.isfinite(at_zero), f"{at_zero:.6f}")
    check("squash(0) is 1/(2+b)", abs(at_zero - 1.0 / (2.0 + B_MID)) < 1e-12,
          f"{at_zero:.6f} vs {1.0 / (2.0 + B_MID):.6f}")
    check("squash is strictly positive far out",
          _f(squash(torch.tensor([1.0]), A_COARSE, B_COARSE)) > 0.0)
    check("squash is symmetric in x",
          abs(_f(squash(torch.tensor([0.004]), A_MID, B_MID))
              - _f(squash(torch.tensor([-0.004]), A_MID, B_MID))) < 1e-15)

    # -- the three kernels ---------------------------------------------------
    peak = _f(kernel_sum(torch.tensor([0.0]), A_COARSE, B_COARSE, A_MID, B_MID, A_FINE, B_FINE))
    check("three kernels peak at exactly 1.0", abs(peak - 1.0) < 1e-12, f"{peak:.12f}")

    # The Tassa relation the widths were SOLVED from, not tuned: at its own
    # margin a kernel must have fallen to 10 % of dm_control's convention.
    # a = arccosh(1/0.1)/margin, so a*margin = arccosh(10) = 2.9932.
    for name, a, margin in (("coarse", A_COARSE, 0.17321), ("mid", A_MID, 0.003),
                            ("fine", A_FINE, 0.0005876)):
        check(f"{name} width follows a = 2.9932/margin",
              abs(a * margin - math.acosh(10.0)) < 0.01,
              f"a*margin {a * margin:.4f} vs {math.acosh(10.0):.4f}")

    # EXPLOIT: submarining / side-parking / yaw-parking. All three move the
    # part AWAY from the seat on the SDF manifold, so all three die if and
    # only if the reward falls monotonically with the SDF distance.
    ds = [0.0, 0.0001, 0.0005, 0.001, 0.003, 0.01, 0.05]
    vals = [_f(kernel_sum(torch.tensor([d]), A_COARSE, B_COARSE, A_MID, B_MID, A_FINE, B_FINE))
            for d in ds]
    check("kernel sum falls monotonically with SDF distance",
          all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)),
          " > ".join(f"{v:.4f}" for v in vals))

    # EXPLOIT: suicide / early abort. D-109 point (11): the squashed reward is
    # ALWAYS positive, so every step pays and leaving is never free.
    check("the kernel sum is strictly positive everywhere", all(v > 0.0 for v in vals),
          f"min {min(vals):.3e}")

    # NAMED, NOT DEFEATED (D-109 point (11), new exploit class): existing is
    # rewarded. Pinned rather than fixed, so a later change to the shape shows
    # up here instead of passing unnoticed. 50 mm out still pays this much:
    far = vals[-1]
    check("existing far from the seat still pays (known, reported)", far > 0.0,
          f"{far:.3e} at 50 mm -- this is the open exploit class, not a defect")
    # RESTATED 2026-09-01, and the restatement is the whole revision. This bar
    # used to be ``peak / far > 100`` at 50 mm, which the old 10 mm coarse
    # kernel cleared by 22 orders of magnitude -- and that was the defect, not
    # the virtue: a signal that is 1e-22 at the distances an episode actually
    # spans is a flat landscape. With the coarse margin at the measured travel
    # reach the ratio at 50 mm is a single-digit number BY DESIGN. The property
    # worth pinning is the ORDER, not the size of the gap.
    check("the seat still pays strictly more than existing 50 mm out",
          peak > far and peak / far > 3.0, f"ratio {peak / far:.2f}")
    # THE POINT OF THE NEW MARGIN, pinned as its own check: at the workcell
    # home stand-off (165 mm, WORKCELL_HOME_TIP_ABOVE_ENTRANCE) the shaping
    # term must be a number a policy can follow, not a denormal. RT-119
    # measured 3.05e-23 there under the old width; anything of that order is
    # the failure this revision exists to remove.
    at_home = _f(kernel_sum(torch.tensor([0.165]), A_COARSE, B_COARSE,
                            A_MID, B_MID, A_FINE, B_FINE))
    check("the kernel still has a magnitude at the 165 mm home stand-off",
          at_home > 0.01, f"{at_home:.4f} (was 3.05e-23 at the 10 mm margin)")

    # -- SAPU (D-109 point (10)) --------------------------------------------
    check("SAPU scale is exactly 1 at zero interpenetration",
          abs(_f(sapu(torch.tensor([0.0]), INTERPEN_THRESH)) - 1.0) < 1e-15)
    just_under = _f(sapu(torch.tensor([INTERPEN_THRESH * 0.999]), INTERPEN_THRESH))
    check("SAPU scale just under the threshold is 1 - tanh(~1)",
          abs(just_under - (1.0 - math.tanh(0.999))) < 1e-9, f"{just_under:.6f}")
    check("SAPU discards above the threshold",
          _f(sapu(torch.tensor([INTERPEN_THRESH * 1.001]), INTERPEN_THRESH)) == 0.0)
    check("SAPU falls monotonically below the threshold",
          _f(sapu(torch.tensor([0.00002]), INTERPEN_THRESH))
          > _f(sapu(torch.tensor([0.0001]), INTERPEN_THRESH)))

    # -- the assembled reward ------------------------------------------------
    r_seated = _f(_rewards(ns, 0.0, engaged=True, success=True))
    r_engaged = _f(_rewards(ns, 0.0, engaged=True, success=False))
    r_bare = _f(_rewards(ns, 0.0))
    check("the success bonus is paid per step, weight 1.0",
          abs((r_seated - r_engaged) - W_SUCCESS) < 1e-12, f"{r_seated - r_engaged:.6f}")
    check("the engaged bonus is paid per step, weight 1.0",
          abs((r_engaged - r_bare) - W_ENGAGED) < 1e-12, f"{r_engaged - r_bare:.6f}")

    # EXPLOIT: tunneling. The ONE that survives Block 5, and the whole reason
    # SAPU exists. Past the threshold the entire task return is discarded --
    # bonuses included, or tunneling would still pay 2.0 per step.
    r_tunnel = _f(_rewards(ns, 0.0, interpen=INTERPEN_THRESH * 2.0, engaged=True, success=True))
    check("tunneling past the SAPU threshold pays nothing", abs(r_tunnel) < 1e-15,
          f"{r_tunnel:.6e}")
    r_clean = _f(_rewards(ns, 0.0, interpen=0.0, engaged=True, success=True))
    check("a clean seat pays the full return", abs(r_clean - (1.0 + W_ENGAGED + W_SUCCESS)) < 1e-12,
          f"{r_clean:.6f}")
    r_dirty = _f(_rewards(ns, 0.0, interpen=INTERPEN_THRESH * 0.5, engaged=True, success=True))
    check("partial interpenetration scales the return down",
          0.0 < r_dirty < r_clean, f"{r_dirty:.6f} < {r_clean:.6f}")

    # -- the abort payment (D-114) -------------------------------------------
    # The magnitude is decided since 2026-09-01 (one kernel peak) but still
    # unmeasured, so this file keeps asserting only the sign and the
    # composition -- properties that hold for ANY negative payment.
    r_abort = _f(_rewards(ns, 0.001, force_abort=True))
    r_alive = _f(_rewards(ns, 0.001, force_abort=False))
    check("aborting costs the fixed payment",
          abs((r_abort - r_alive) - PROBE_ABORT_PENALTY) < 1e-12, f"{r_abort - r_alive:.6f}")
    check("aborting is strictly worse than not aborting", r_abort < r_alive,
          f"{r_abort:.6f} < {r_alive:.6f}")
    # COMPOSITION, decided here and stated: SAPU scales the TASK return only.
    # Its source (IsaacGymEnvs get_sapu_reward_scale) scales the task reward
    # and knows no abort payment; the abort payment's source
    # (Beltran-Hernandez 2020) knows no SAPU. Each keeps its own semantics,
    # so the payment is added AFTER the scaling and is not discounted by it.
    r_abort_tunnel = _f(_rewards(ns, 0.001, interpen=INTERPEN_THRESH * 2.0, force_abort=True))
    check("the abort payment survives a discarded task return",
          abs(r_abort_tunnel - PROBE_ABORT_PENALTY) < 1e-12, f"{r_abort_tunnel:.6f}")

    # =====================================================================
    #  The 2026-09-01 revision: time penalty, action rate, success payout
    #  (inbox entry "Reward-Ueberarbeitung" 2026-09-01, p1-konzept-messung)
    # =====================================================================

    # -- (6) R_time = -1/T, paid on EVERY step -------------------------------
    r_plain = _f(_rewards(ns, 0.001))
    r_timed = _f(_rewards(ns, 0.001, time_penalty=PROBE_TIME_PENALTY))
    check("the time penalty is subtracted from every step",
          abs((r_timed - r_plain) - PROBE_TIME_PENALTY) < 1e-12,
          f"{r_timed - r_plain:.8f} vs {PROBE_TIME_PENALTY:.8f}")
    # TERMINAL STEPS TOO -- a step that ends the episode still took its time.
    # Both terminal kinds are checked, because they take different branches.
    r_timed_abort = _f(_rewards(ns, 0.001, force_abort=True,
                                time_penalty=PROBE_TIME_PENALTY))
    check("the time penalty is paid on the force-abort step as well",
          abs((r_timed_abort - r_abort) - PROBE_TIME_PENALTY) < 1e-12,
          f"{r_timed_abort - r_abort:.8f}")
    r_succ_free = _f(_rewards(ns, 0.0, engaged=True, success=True))
    r_succ_timed = _f(_rewards(ns, 0.0, engaged=True, success=True,
                               time_penalty=PROBE_TIME_PENALTY))
    check("the time penalty is paid on the success-termination step as well",
          abs((r_succ_timed - r_succ_free) - PROBE_TIME_PENALTY) < 1e-12,
          f"{r_succ_timed - r_succ_free:.8f}")
    # THE SAFETY INEQUALITY THE DECISION STATES, checked rather than trusted:
    # a full episode of time costs exactly one abort payment, so time can never
    # be the cheaper way out. Both sides are probe values here.
    check("a whole episode of time costs exactly one abort payment",
          abs(PROBE_TIME_PENALTY * PROBE_EPISODE_STEPS - PROBE_ABORT_PENALTY) < 1e-12,
          f"{PROBE_TIME_PENALTY * PROBE_EPISODE_STEPS:.6f}")
    # And the inequality that keeps the suicide exploit closed: living pays
    # the kernel, and the kernel at the rung-0 start beats the per-step time
    # bill by orders. 0.165 m is the home stand-off; the rung-0 start is
    # nearer still, so this is the WORST case of the two.
    check("living beats the time penalty even at the home stand-off",
          at_home > abs(PROBE_TIME_PENALTY),
          f"kernel {at_home:.4f} vs time {abs(PROBE_TIME_PENALTY):.6f}")

    # -- (8) the action-rate penalty -----------------------------------------
    r_jerk = _f(_rewards(ns, 0.001, action_rate=0.5,
                         action_rate_scale=PROBE_ACTION_RATE_SCALE))
    check("the action-rate penalty is scale * ||a_t - a_(t-1)||, subtracted",
          abs((r_jerk - r_plain) + PROBE_ACTION_RATE_SCALE * 0.5) < 1e-12,
          f"{r_jerk - r_plain:.8f} vs {-PROBE_ACTION_RATE_SCALE * 0.5:.8f}")
    check("a repeated action pays no action-rate penalty",
          abs(_f(_rewards(ns, 0.001, action_rate=0.0,
                          action_rate_scale=PROBE_ACTION_RATE_SCALE)) - r_plain) < 1e-12)
    check("the action-rate penalty grows with the change",
          _f(_rewards(ns, 0.001, action_rate=1.0, action_rate_scale=PROBE_ACTION_RATE_SCALE))
          < r_jerk)

    # -- (7) success terminates and is paid out ------------------------------
    # THE IDENTITY THE PAYOUT IS FOR: seating at step t and being paid out must
    # equal holding that same state to the timeout. Both sides are computed
    # from the SAME per-step value, so what is under test is the multiplier and
    # nothing else.
    step_seated = _f(_rewards(ns, 0.0, engaged=True, success=True))
    remaining = 200.0
    paid = _f(_rewards(ns, 0.0, engaged=True, success=True, steps_remaining=remaining))
    check("the success payout is the step value times the steps left",
          abs(paid - step_seated * (1.0 + remaining)) < 1e-9,
          f"{paid:.4f} vs {step_seated * (1.0 + remaining):.4f}")
    check("holding to the timeout and being paid out are the same undiscounted return",
          abs(paid - step_seated * (1.0 + remaining)) < 1e-9,
          f"{1.0 + remaining:.0f} steps at {step_seated:.4f}")
    check("no steps left means no lump, only the step itself",
          abs(_f(_rewards(ns, 0.0, engaged=True, success=True, steps_remaining=0.0))
              - step_seated) < 1e-12)
    # THE EXPLOIT THE PAYOUT CLOSES (the 2026-09-01 reward account): parking at
    # "engaged" for the rest of the episode must not beat seating now. Engaged
    # parking pays the engaged step value for every remaining step; seating
    # pays the seated step value for the same steps.
    step_parked = _f(_rewards(ns, 0.0, engaged=True, success=False))
    check("seating and being paid out beats parking at engaged for the same steps",
          step_seated * (1.0 + remaining) > step_parked * (1.0 + remaining),
          f"{step_seated:.4f}/step vs {step_parked:.4f}/step")
    # A TUNNELLED "seat" cannot buy itself an income: SAPU discards the step
    # value, and the lump is computed from the SCALED value.
    check("a tunnelled seat is paid out nothing",
          abs(_f(_rewards(ns, 0.0, interpen=INTERPEN_THRESH * 2.0, engaged=True,
                          success=True, steps_remaining=remaining))) < 1e-15)
    # THE TIE-BREAK, a code-level choice and therefore checked here: success
    # beats the force abort on a shared step -- the lump is paid and the
    # payment is not.
    r_both = _f(_rewards(ns, 0.0, engaged=True, success=True, force_abort=True,
                         steps_remaining=remaining))
    check("success wins the tie against a force abort on the same step",
          abs(r_both - paid) < 1e-9, f"{r_both:.4f} vs {paid:.4f}")
    check("the abort payment is NOT charged on a success step",
          r_both > 0.0 and abs((r_both - paid) - PROBE_ABORT_PENALTY) > 0.5,
          f"{r_both:.4f}")

    # =====================================================================
    #  The alignment shaping row (RT-171, 2026-09-06)
    # =====================================================================
    # Potential-based shaping, F = gamma * Phi(s') - Phi(s) with
    # Phi = w_tilt * sech(a * theta) and theta the part's tilt against the
    # POCKET axis. Three claims, each with its own block: the MEASUREMENT
    # (cos theta, sign and reference frame), the POTENTIAL (the D-109 (9)
    # curve), and the ROW (Ng form, terminal convention, place in the stack).
    # Then the two properties the whole term rests on -- telescoping and the
    # DISCOUNTED success-over-failure ordering -- on synthetic trajectories.
    tilt_cos_theta = ns["tilt_cos_theta"]
    tilt_potential = ns["tilt_potential"]

    # -- the measurement ----------------------------------------------------
    # 180 deg about x turns the tool's +z into env -z: the part hangs straight
    # down an untilted pocket, which IS the seated orientation. The sign is
    # the one the plan review caught ("+z against +z" would read -1 here).
    c_down = _f(tilt_cos_theta(q_x180, eye1))
    check("a part hanging straight down an untilted pocket reads cos theta = +1",
          abs(c_down - 1.0) < 1e-12, f"{c_down:.12f}")
    c_up = _f(tilt_cos_theta(q_id, eye1))
    check("a part pointing straight UP reads cos theta = -1",
          abs(c_up + 1.0) < 1e-12, f"{c_up:.12f}")
    # A pocket tilted by beta about x (columns = its axes in the env frame),
    # the part still vertical: the angle between them is beta.
    beta = PROBE_TILT_MARGIN_RAD
    cb, sb = math.cos(beta), math.sin(beta)
    pocket_x = torch.tensor([[[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]]])
    c_beta = _f(tilt_cos_theta(q_x180, pocket_x))
    check("a pocket tilted by beta with the part still vertical reads cos beta",
          abs(c_beta - cb) < 1e-12, f"{c_beta:.12f} vs {cb:.12f}")
    # The part tilted WITH the pocket (rotate the hanging part by the same
    # beta about x): aligned again. This is the check that pins the POCKET as
    # the reference -- against the world vertical it would read cos beta.
    q_beta_x = torch.tensor([[math.cos(beta / 2.0), math.sin(beta / 2.0), 0.0, 0.0]])
    q_down_tilted = quat_mul(q_beta_x, q_x180)
    c_with = _f(tilt_cos_theta(q_down_tilted, pocket_x))
    check("a part tilted WITH its tilted pocket reads cos theta = +1",
          abs(c_with - 1.0) < 1e-12, f"{c_with:.12f}")
    # Yaw about the pocket axis is invisible to the term (RT-152 stays a
    # separate question): a hanging part yawed 90 deg still reads +1.
    c_yaw = _f(tilt_cos_theta(quat_mul(q_z90, q_x180), eye1))
    check("yaw about the pocket axis leaves cos theta at +1",
          abs(c_yaw - 1.0) < 1e-12, f"{c_yaw:.12f}")
    c_batch = tilt_cos_theta(torch.cat((q_x180, q_id), dim=0),
                             torch.cat((eye1, eye1), dim=0))
    check("tilt_cos_theta is batched over envs",
          tuple(c_batch.shape) == (2,) and abs(_f(c_batch[0:1]) - 1.0) < 1e-12
          and abs(_f(c_batch[1:2]) + 1.0) < 1e-12, str(tuple(c_batch.shape)))

    # -- the potential -------------------------------------------------------
    def _phi(theta_rad: float, w: float = PROBE_W_TILT) -> float:
        return _f(tilt_potential(torch.tensor([math.cos(theta_rad)]), w, PROBE_A_TILT))

    phi_up = _phi(0.0)
    check("Phi is w upright", abs(phi_up - PROBE_W_TILT) < 1e-12, f"{phi_up:.12f}")
    phi_margin = _phi(PROBE_TILT_MARGIN_RAD)
    check("Phi is 0.1 w at the margin (the D-109 (9) rule)",
          abs(phi_margin - 0.1 * PROBE_W_TILT) < 1e-9, f"{phi_margin:.9f}")
    worst_sech = 0.0
    for deg in (5.42, 8.52):
        th = math.radians(deg)
        worst_sech = max(worst_sech, abs(_phi(th) - PROBE_W_TILT / math.cosh(PROBE_A_TILT * th)))
    check("Phi is sech(a * theta) at 5.42 and 8.52 deg", worst_sech < 1e-12, f"{worst_sech:.3e}")
    grid = [_phi(math.radians(d)) for d in range(0, 181, 5)]
    check("Phi falls monotonically from upright to upside down",
          all(a > b for a, b in zip(grid, grid[1:])), f"{grid[0]:.4f} .. {grid[-1]:.2e}")
    pull_near = _phi(math.radians(2.0)) - _phi(math.radians(4.0))
    pull_far = _phi(math.radians(13.0)) - _phi(math.radians(15.0))
    check("the pull grows toward upright (no trap halfway)",
          pull_near > pull_far > 0.0, f"{pull_near:.4f} per 2 deg near, {pull_far:.4f} far")
    check("w_tilt = 0 makes Phi exactly zero", _phi(0.3, w=0.0) == 0.0)
    phi_over = tilt_potential(torch.tensor([1.5, -1.5]), PROBE_W_TILT, PROBE_A_TILT)
    check("a cos theta beyond [-1, 1] is clamped, no NaN",
          abs(_f(phi_over[0:1]) - phi_up) < 1e-12
          and _f(phi_over[1:2]) == _f(phi_over[1:2]) and _f(phi_over[1:2]) < 0.01 * phi_up)
    # The config's width against the literal margin typed here, by text --
    # importing the constant would assert it against itself.
    tasks_src = (MATH_SRC.parent / "insertion_tasks_cfg.py").read_text(encoding="utf-8")
    check("the config's tilt kernel is arccosh(10) over the 15.34 deg RT-158 margin",
          "TILT_MARGIN_RAD = math.radians(15.34)\n" in tasks_src
          and "KERNEL_A_TILT = math.acosh(10.0) / TILT_MARGIN_RAD\n" in tasks_src)

    # -- the row --------------------------------------------------------------
    g = PROBE_GAMMA
    r_plain_tilt = _f(_rewards(ns, 0.001))
    check("tilt args at zero leave the reward bit-identical (the regression switch)",
          _f(_rewards(ns, 0.001, tilt_phi=0.0, tilt_phi_prev=0.0, gamma=g)) == r_plain_tilt)
    d_row = _f(_rewards(ns, 0.001, tilt_phi=0.4, tilt_phi_prev=0.7, gamma=g)) - r_plain_tilt
    check("the row is gamma * Phi(s') - Phi(s)",
          abs(d_row - (g * 0.4 - 0.7)) < 1e-12, f"{d_row:.12f}")
    r_succ = _f(_rewards(ns, 0.0, engaged=True, success=True))
    d_succ = _f(_rewards(ns, 0.0, engaged=True, success=True,
                         tilt_phi=0.4, tilt_phi_prev=0.7, gamma=g)) - r_succ
    check("the success step zeroes Phi(s')", abs(d_succ + 0.7) < 1e-12, f"{d_succ:.12f}")
    r_abort = _f(_rewards(ns, 0.001, force_abort=True))
    d_abort = _f(_rewards(ns, 0.001, force_abort=True,
                          tilt_phi=0.4, tilt_phi_prev=0.7, gamma=g)) - r_abort
    check("a force abort keeps the real Phi(s') (truncation is bootstrapped)",
          abs(d_abort - (g * 0.4 - 0.7)) < 1e-12, f"{d_abort:.12f}")
    r_tun = _f(_rewards(ns, 0.001, interpen=INTERPEN_THRESH * 2.0))
    d_tun = _f(_rewards(ns, 0.001, interpen=INTERPEN_THRESH * 2.0,
                        tilt_phi=0.4, tilt_phi_prev=0.7, gamma=g)) - r_tun
    check("a tunnelled step still pays the row (not SAPU-scaled)",
          abs(d_tun - (g * 0.4 - 0.7)) < 1e-12, f"{d_tun:.12f}")
    # NOT in the lump: with 200 steps left the row still pays -Phi(s) once,
    # not 201 times. Phi(s') = 0 here so the claim is about the lump alone.
    r_lump = _f(_rewards(ns, 0.0, engaged=True, success=True, steps_remaining=200.0))
    d_lump = _f(_rewards(ns, 0.0, engaged=True, success=True, steps_remaining=200.0,
                         tilt_phi=0.0, tilt_phi_prev=0.7, gamma=g)) - r_lump
    check("the row is NOT in the lump", abs(d_lump + 0.7) < 1e-9, f"{d_lump:.9f}")
    terms_fn = ns["compute_reward_terms_insertion"]
    names = ns["REWARD_TERMS"]
    one = torch.tensor([1.0])
    zero = torch.tensor([0.0])
    stack = terms_fn(
        torch.tensor([0.001]), zero, zero > 0.5, zero > 0.5, zero > 0.5, zero, zero, zero,
        torch.tensor([0.4]), torch.tensor([0.7]),
        A_COARSE, B_COARSE, A_MID, B_MID, A_FINE, B_FINE, W_ENGAGED, W_SUCCESS, 0.0,
        INTERPEN_THRESH, PROBE_ABORT_PENALTY, 0.0, 0.0, g,
    )
    check("tilt_shaping is the last row of the stack and reads gamma * Phi(s') - Phi(s)",
          names[-1] == "tilt_shaping" and tuple(stack.shape) == (len(names), 1)
          and abs(_f(stack[len(names) - 1]) - (g * 0.4 - 0.7)) < 1e-12,
          f"{names[-1]} at {len(names) - 1}, {_f(stack[len(names) - 1]):.12f}")

    # -- telescoping ----------------------------------------------------------
    # Sum_t gamma^t F_t == gamma^T Phi(s_T) - Phi(s_0): the discounted shaping
    # depends on the first and the last state only (Ng, Harada & Russell
    # 1999). Righting from the margin over 40 steps, then holding.
    def _row(phi_prev: float, phi_now: float, success: bool = False) -> float:
        base = _f(_rewards(ns, 0.0 if success else 0.03, engaged=success, success=success))
        return _f(_rewards(ns, 0.0 if success else 0.03, engaged=success, success=success,
                           tilt_phi=phi_now, tilt_phi_prev=phi_prev, gamma=g)) - base

    thetas = [PROBE_TILT_MARGIN_RAD * (1.0 - t / 40.0) for t in range(41)] + [0.0] * 20
    phis = [_phi(th) for th in thetas]
    T_len = len(phis) - 1
    disc_sum = sum((g ** t) * _row(phis[t], phis[t + 1]) for t in range(T_len))
    want = (g ** T_len) * phis[-1] - phis[0]
    check("the discounted row over a trajectory telescopes to gamma^T Phi(s_T) - Phi(s_0)",
          abs(disc_sum - want) < 1e-9, f"{disc_sum:.9f} vs {want:.9f}")
    disc_succ = sum((g ** t) * _row(phis[t], phis[t + 1]) for t in range(T_len - 1))
    disc_succ += (g ** (T_len - 1)) * _row(phis[T_len - 1], phis[T_len], success=True)
    check("the discounted row over a trajectory that ends in success telescopes to -Phi(s_0)",
          abs(disc_succ + phis[0]) < 1e-9, f"{disc_succ:.9f} vs {-phis[0]:.9f}")
    # Farming: tilt to the margin and right again, ten times over. Discounted
    # it earns exactly (gamma^T - 1) Phi(s_0) < 0 -- nothing to farm.
    cyc = []
    for _ in range(10):
        cyc += [PROBE_TILT_MARGIN_RAD * k / 5.0 for k in range(5)]
        cyc += [PROBE_TILT_MARGIN_RAD * (5 - k) / 5.0 for k in range(5)]
    cyc.append(0.0)
    cphis = [_phi(th) for th in cyc]
    Tc = len(cphis) - 1
    disc_cyc = sum((g ** t) * _row(cphis[t], cphis[t + 1]) for t in range(Tc))
    check("tilting and righting in a circle earns nothing (discounted sum = (gamma^T - 1) Phi(s_0))",
          abs(disc_cyc - ((g ** Tc) - 1.0) * cphis[0]) < 1e-9 and disc_cyc < 0.0,
          f"{disc_cyc:.9f}")
    # Why the ordering test below is DISCOUNTED: summed plainly, the row
    # charges every aligned step (1 - gamma) Phi, which would read as a
    # standing penalty on the best state and mis-rank a hover.
    undisc_hover = sum(_row(phi_up, phi_up) for _ in range(PROBE_EPISODE_STEPS))
    check("undiscounted, the row charges an aligned hover (1 - gamma) Phi(upright) per step",
          abs(undisc_hover + (1.0 - g) * phi_up * PROBE_EPISODE_STEPS) < 1e-9,
          f"{undisc_hover:.6f}")

    # -- the reward-ordering test, DISCOUNTED, on reference trajectories -------
    # Every successful trajectory must out-earn every unsuccessful one, at
    # every weight of the sensitivity decade. Synthetic inputs, real reward,
    # PROBE constants for the other terms. The trajectories: an EXPERT
    # (aligned descent, seated at step 50), a HOVER (aligned, in the air,
    # timeout), a JAM (tilts to the margin at the mouth, timeout
    # -- RT-158's mode B), an IDLE (never leaves the home distance) and a
    # TILT-CYCLE (tilts and rights every ten steps in the air).
    def _G(steps: list, w: float) -> float:
        total = 0.0
        prev = _phi(steps[0]["theta"], w)
        for t, s in enumerate(steps):
            now = _phi(s["theta"], w)
            r = _f(_rewards(ns, s["sdf"], engaged=s.get("engaged", False),
                            success=s.get("success", False),
                            steps_remaining=s.get("remaining", 0.0),
                            time_penalty=PROBE_TIME_PENALTY,
                            depth_progress=s.get("progress", 0.0), w_progress=PROBE_W_PROGRESS,
                            tilt_phi=now, tilt_phi_prev=prev, gamma=g))
            total += (g ** t) * r
            prev = now
        return total

    T_ep = PROBE_EPISODE_STEPS
    expert = [{"sdf": 0.03 * (1.0 - t / 49.0), "theta": 0.0, "engaged": t >= 35,
               "progress": 0.036 / 50.0} for t in range(50)]
    expert[-1].update({"sdf": 0.0, "success": True, "remaining": float(T_ep - 1 - 49)})
    hover = [{"sdf": 0.03, "theta": 0.0} for _ in range(T_ep)]
    jam = [{"sdf": 0.005, "theta": PROBE_TILT_MARGIN_RAD * min(t, 10) / 10.0}
           for t in range(T_ep)]
    idle = [{"sdf": 0.17321, "theta": 0.0} for _ in range(T_ep)]
    cycle = [{"sdf": 0.03, "theta": PROBE_TILT_MARGIN_RAD * float((t % 20) >= 10)}
             for t in range(T_ep)]
    ordered = True
    detail = []
    for w in (0.1, 1.0, 10.0):
        Ge = _G(expert, w)
        losers = {"hover": _G(hover, w), "jam": _G(jam, w), "idle": _G(idle, w),
                  "cycle": _G(cycle, w)}
        ordered = ordered and all(Ge > v for v in losers.values())
        detail.append(f"w={w}: expert {Ge:.2f} > " + ", ".join(f"{k} {v:.2f}" for k, v in losers.items()))
    check("discounted ordering: the expert beats hover / jam / idle / tilt-cycle at w_tilt 0.1, 1.0, 10.0",
          ordered, "; ".join(detail))
    # The term does what it is for: the jam, which ends tilted, is charged
    # more shaping than the hover, which stays upright. Read as the row alone.
    def _S(steps: list) -> float:
        prev = _phi(steps[0]["theta"])
        total = 0.0
        for t, s in enumerate(steps):
            now = _phi(s["theta"])
            total += (g ** t) * _row(prev, now)
            prev = now
        return total

    s_jam, s_hover = _S(jam), _S(hover)
    check("the jam is charged for its tilt: its discounted shaping is below the hover's",
          s_jam < s_hover, f"jam {s_jam:.4f} vs hover {s_hover:.4f}")

    # =====================================================================
    #  Observation (D-107, D-114)
    # =====================================================================
    slices = ns["OBS_SLICES"]
    check("the observation is 28 channels wide", ns["OBS_DIM"] == 28, str(ns["OBS_DIM"]))

    # The slice table must tile 0..28 exactly: no gap (a channel nothing
    # writes) and no overlap (two blocks fighting over one index).
    covered: list[int] = []
    for lo, hi in slices.values():
        covered.extend(range(lo, hi))
    check("the slice table tiles 0..28 with no gap or overlap",
          sorted(covered) == list(range(ns["OBS_DIM"])) and len(covered) == ns["OBS_DIM"],
          f"{len(covered)} indices, {len(set(covered))} distinct")

    # THE LAYOUT IDENTITY TEST. Every block gets its own marker value, so a
    # swapped cat order and a wrong slice table both show up as a block
    # reading back someone else's marker.
    n = 3
    widths = {k: hi - lo for k, (lo, hi) in slices.items()}
    markers = {k: float(i + 1) for i, k in enumerate(slices)}
    blocks = {k: torch.ones(n, widths[k]) * markers[k] for k in slices}
    obs = ns["assemble_observation"](
        blocks["joint_pos"], blocks["joint_vel"], blocks["tip_rel"], blocks["ee_quat"],
        blocks["yaw_cos_sin"], blocks["pocket_quat"], blocks["force"],
    )
    check("assemble_observation returns (num_envs, 28)", tuple(obs.shape) == (n, 28),
          str(tuple(obs.shape)))
    for name, (lo, hi) in slices.items():
        seg = obs[:, lo:hi]
        check(f"channels {lo}:{hi} carry {name}",
              bool((seg == markers[name]).all()), f"first value {_f(seg)}")

    # -- the two observation modes (D-188) -----------------------------------
    # ``force`` IS the table above; ``wrench`` appends the torque block. The
    # widths are typed literals on purpose (28 is D-114's, 31 = 28 + 3), so
    # this does not assert the module against itself.
    obs_slices_fn, obs_dim_fn = ns["obs_slices"], ns["obs_dim"]
    check("obs mode 'force' is the 28-wide OBS_SLICES table unchanged",
          obs_slices_fn("force") == slices and obs_dim_fn("force") == 28 == ns["OBS_DIM"],
          f"{obs_dim_fn('force')}")
    wslices = obs_slices_fn("wrench")
    check("obs mode 'wrench' is 31 wide", obs_dim_fn("wrench") == 31, f"{obs_dim_fn('wrench')}")
    check("the torque slice is appended DIRECTLY after force, 28:31",
          wslices.get("torque") == (slices["force"][1], slices["force"][1] + 3)
          and list(wslices)[-1] == "torque",
          f"{wslices.get('torque')}")
    check("the wrench table keeps every force-mode slice at its old index",
          all(wslices[k] == v for k, v in slices.items()))
    wcovered: list[int] = []
    for lo, hi in wslices.values():
        wcovered.extend(range(lo, hi))
    check("the wrench table tiles 0..31 with no gap or overlap",
          sorted(wcovered) == list(range(31)) and len(wcovered) == 31,
          f"{len(wcovered)} indices, {len(set(wcovered))} distinct")
    try:
        obs_slices_fn("wrenchh")
        _mode_ok, _mode_detail = False, "an unknown mode was ACCEPTED"
    except ValueError as exc:
        _mode_ok, _mode_detail = True, str(exc)
    check("obs_slices refuses an unknown mode", _mode_ok, _mode_detail)
    check("obs_version names mode and width",
          ns["obs_version"]("force") == "force-28" and ns["obs_version"]("wrench") == "wrench-31")
    # The wrench layout identity: the first 28 channels are assemble_observation's
    # own output and 28:31 carry the torque marker.
    torque_marker = 99.0
    wobs = ns["assemble_observation_wrench"](
        blocks["joint_pos"], blocks["joint_vel"], blocks["tip_rel"], blocks["ee_quat"],
        blocks["yaw_cos_sin"], blocks["pocket_quat"], blocks["force"],
        torch.ones(n, 3) * torque_marker,
    )
    check("assemble_observation_wrench returns (num_envs, 31)", tuple(wobs.shape) == (n, 31),
          str(tuple(wobs.shape)))
    check("the wrench layout's first 28 channels ARE the force layout",
          bool((wobs[:, 0:28] == obs).all()))
    check("channels 28:31 carry torque",
          bool((wobs[:, 28:31] == torque_marker).all()), f"first value {_f(wobs[:, 28:31])}")

    # -- yaw encoding: cos phi / sin phi, NOT cos 4 phi (D-107 point 2) -----
    ycs = ns["yaw_cos_sin"]
    e0 = ycs(torch.tensor([[1.0, 0.0, 0.0]]))
    check("yaw of an unrotated x-axis is (1, 0)",
          abs(_f(e0[:, 0:1]) - 1.0) < 1e-12 and abs(_f(e0[:, 1:2])) < 1e-12)
    e30 = ycs(torch.tensor([[math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0]]))
    check("yaw encoding is (cos phi, sin phi) of the true angle",
          abs(_f(e30[:, 0:1]) - math.cos(math.radians(30.0))) < 1e-12
          and abs(_f(e30[:, 1:2]) - math.sin(math.radians(30.0))) < 1e-12)
    check("the yaw encoding is a unit vector",
          abs(_f(torch.linalg.norm(e30, dim=-1)) - 1.0) < 1e-12)
    # The regression against the proxy: the square peg's cos 4 phi mapped
    # four orientations onto one input. The real part has ONE valid pose, so
    # a 90 deg turn MUST look different.
    e90 = ycs(torch.tensor([[0.0, 1.0, 0.0]]))
    check("a 90 deg turn changes the encoding (no C4 folding)",
          abs(_f(e90[:, 0:1]) - _f(e0[:, 0:1])) > 0.5,
          f"cos {_f(e0[:, 0:1]):.3f} vs {_f(e90[:, 0:1]):.3f}")
    # A part tilted to vertical projects to nothing; the encoding must be a
    # defined value, not a NaN reaching the policy.
    e_deg = ycs(torch.tensor([[0.0, 0.0, 1.0]]))
    check("a degenerate projection gives a defined encoding, not NaN",
          math.isfinite(_f(e_deg[:, 0:1])) and math.isfinite(_f(e_deg[:, 1:2])))

    # -- quaternion canonicalisation (D-037) ---------------------------------
    canon = ns["canonicalize_quat"]
    q = torch.tensor([[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0]])
    c = canon(q)
    check("canonicalize forces w >= 0", bool((c[:, 0] >= 0.0).all()),
          str([round(float(v), 3) for v in c[:, 0]]))
    check("canonicalize is idempotent", bool((canon(c) == c).all()))
    check("canonicalize leaves an already-positive quaternion untouched",
          bool((c[1] == q[1]).all()))
    check("canonicalize negates ALL four components, not just w",
          bool((c[0] == -q[0]).all()), str([round(float(v), 3) for v in c[0]]))
    # w exactly 0 is the 180-degree case: a sign(w) multiply would zero it.
    check("a quaternion with w = 0 survives canonicalisation",
          abs(_f(torch.linalg.norm(c[2:3], dim=-1)) - 1.0) < 1e-12,
          f"norm {_f(torch.linalg.norm(c[2:3], dim=-1)):.6f}")

    # -- EMA (D-114: the force channel is smoothed) --------------------------
    ema = ns["ema_update"]
    prev, new = torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])
    check("alpha = 1 passes the new sample through", abs(_f(ema(prev, new, 1.0)) - 1.0) < 1e-15)
    check("alpha = 0 freezes at the previous sample", abs(_f(ema(prev, new, 0.0))) < 1e-15)
    check("alpha = 0.25 weights the NEW sample by 0.25",
          abs(_f(ema(prev, new, 0.25)) - 0.25) < 1e-15, f"{_f(ema(prev, new, 0.25)):.6f}")
    x = prev
    for _ in range(200):
        x = ema(x, new, 0.25)
    check("the EMA converges to a constant input", abs(_f(x) - 1.0) < 1e-9, f"{_f(x):.9f}")

    # -- observation noise: the two channel masks (D-182) --------------------
    # Our own share of the noise model is exactly these two tensors. The
    # library owns the hook, the reset schedule and both noise functions
    # (``NoiseModelWithAdditiveBias``); ``obs_noise.InsertionObsNoise`` only
    # hands them over. A mask built against the wrong block cannot be seen in
    # a training curve -- the run is simply noisy somewhere else -- so it is
    # checked here, against OBS_SLICES itself and never against typed indices.
    masks = ns["noise_masks"]
    # THE ORDER OF THE TWO SIGMAS IS NOT A CONVENTION ANY MORE (2026-09-12).
    # Both are bare floats, so a POSITIONAL call carries nothing but their
    # order: swapped, the env trains with a 3.5 m pocket bias and 0.0025 N of
    # force jitter and nothing raises -- the L-08 defect, which a sweep for
    # the parameter NAME cannot see. ``noise_masks`` is keyword-only, so the
    # swap cannot be written at all; this check is what holds the ``*`` in
    # place, because a positional call that SUCCEEDS is exactly the state the
    # signature exists to forbid.
    try:
        masks(0.0025, 3.5)
        _kw_ok, _kw_detail = False, "a positional call was ACCEPTED"
    except TypeError as exc:
        _kw_ok, _kw_detail = True, str(exc)
    check("noise_masks refuses positional sigmas, so the two cannot swap silently",
          _kw_ok, _kw_detail)
    # The two probe values are DIFFERENT on purpose: they are what pins each
    # KEYWORD to its block by VALUE below. 0.0025 must come out on tip_rel and
    # 3.5 on force, never the other way round.
    step_std, bias_std = masks(pocket_pos_std_m=0.0025, force_std_n=3.5)
    check("noise_masks returns one std per observation channel",
          tuple(step_std.shape) == (ns["OBS_DIM"],)
          and tuple(bias_std.shape) == (ns["OBS_DIM"],),
          f"{tuple(step_std.shape)} {tuple(bias_std.shape)}")
    _lo_t, _hi_t = slices["tip_rel"]
    _lo_f, _hi_f = slices["force"]
    # COUNTED, not summed: a count of non-zero channels is exact in floating
    # point and a sum of three equal sigmas is not.
    check("the per-episode bias lands on tip_rel and nowhere else",
          bool((_arr(bias_std)[_lo_t:_hi_t] == 0.0025).all())
          and int((_arr(bias_std) != 0.0).sum()) == _hi_t - _lo_t,
          str(_arr(bias_std).round(6).tolist()))
    check("the per-step noise lands on force and nowhere else",
          bool((_arr(step_std)[_lo_f:_hi_f] == 3.5).all())
          and int((_arr(step_std) != 0.0).sum()) == _hi_f - _lo_f,
          str(_arr(step_std).round(6).tolist()))
    # The two must not leak into each other. The pocket bias is a per-EPISODE
    # constant and the force noise is per-STEP, so a sigma that appeared in
    # both masks would change what a channel MEANS, not merely how loud it is.
    check("neither mask touches the other's block",
          int((_arr(bias_std)[_lo_f:_hi_f] != 0.0).sum()) == 0
          and int((_arr(step_std)[_lo_t:_hi_t] != 0.0).sum()) == 0,
          f"bias@force {_arr(bias_std)[_lo_f:_hi_f].tolist()}, "
          f"step@tip {_arr(step_std)[_lo_t:_hi_t].tolist()}")
    # Both sigmas at zero is the OFF state the cfg promises: the env builds no
    # model at all, and the masks it would have built are all zero.
    _z_step, _z_bias = masks(pocket_pos_std_m=0.0, force_std_n=0.0)
    check("both sigmas at zero give all-zero masks (the switched-off state)",
          int((_arr(_z_step) != 0.0).sum()) == 0
          and int((_arr(_z_bias) != 0.0).sum()) == 0)
    # -- the torque sigma (D-188): its own N m, its own block, wrench mode only
    _w_step, _w_bias = masks(pocket_pos_std_m=0.0025, force_std_n=3.5,
                             torque_std_nm=0.2, mode="wrench")
    _lo_q, _hi_q = wslices["torque"]
    check("wrench-mode masks are 31 wide",
          tuple(_w_step.shape) == (31,) and tuple(_w_bias.shape) == (31,),
          f"{tuple(_w_step.shape)} {tuple(_w_bias.shape)}")
    # ONE check pins force AND torque by value: 3.5 on 25:28, 0.2 on 28:31,
    # six non-zero channels and no other. A torque sigma that lands on the
    # force block would overwrite the 3.5 and be seen here.
    check("in wrench mode the per-step noise is 3.5 on force, 0.2 on torque, nowhere else",
          bool((_arr(_w_step)[_lo_f:_hi_f] == 3.5).all())
          and bool((_arr(_w_step)[_lo_q:_hi_q] == 0.2).all())
          and int((_arr(_w_step) != 0.0).sum()) == (_hi_f - _lo_f) + (_hi_q - _lo_q),
          str(_arr(_w_step).round(6).tolist()))
    check("the torque sigma never reaches the bias mask",
          int((_arr(_w_bias) != 0.0).sum()) == _hi_t - _lo_t
          and bool((_arr(_w_bias)[_lo_t:_hi_t] == 0.0025).all()))
    _w0_step, _ = masks(pocket_pos_std_m=0.0, force_std_n=0.0, torque_std_nm=0.0, mode="wrench")
    check("wrench mode with all three sigmas at zero is all-zero, 31 wide",
          tuple(_w0_step.shape) == (31,) and int((_arr(_w0_step) != 0.0).sum()) == 0)
    try:
        masks(pocket_pos_std_m=0.0, force_std_n=0.0, torque_std_nm=0.2, mode="force")
        _tq_ok, _tq_detail = False, "a torque sigma in force mode was ACCEPTED"
    except ValueError as exc:
        _tq_ok, _tq_detail = True, str(exc)
    check("noise_masks refuses a torque sigma in force mode (it would land nowhere)",
          _tq_ok, _tq_detail)

    # =====================================================================
    #  Termination and success (D-113, D-114, D-106)
    # =====================================================================
    # Probe values. F_max is [measure] and the band edges are [CAD pending];
    # nothing below asserts a NUMBER for any of them, only the shape of the
    # rule around them.
    f_max, max_len = 25.0, 256
    depth_min, depth_max, engaged_depth = 0.020, 0.023, 0.005

    dones = ns["compute_dones"]
    fmag = ns["force_magnitude"]

    check("force magnitude is the norm of the three components",
          abs(_f(fmag(torch.tensor([[3.0, 4.0, 0.0]]))) - 5.0) < 1e-12)

    # ------------------------------------------------------------------
    # THE GRAVITY TARE (2026-08-31). The wrench carries the child link's own
    # weight because this repo keeps gravity ON at the robot, while FORGE --
    # the pattern D-114 copied -- spawns robot and held asset with
    # disable_gravity=True and therefore never sees it. These cases pin the
    # RULE offline; whether the real wrench's SIGN matches is a measurement
    # (the free-air probe must read ~0 N) and is not claimed here.
    tare = ns["gravity_tare"]
    _tare_m, _tare_g = 0.824, -9.81
    hold_w = torch.tensor([0.0, 0.0, -_tare_m * _tare_g])          # +8.083 N, world +z
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]])       # no rotation

    def _vec(t):
        return [float(v) for v in t.tolist()[0]]

    check("the tare is a no-op when it is switched off",
          _vec(tare(torch.tensor([[0.0, 0.0, 8.083]]), ident, hold_w, False))
          == [0.0, 0.0, 8.083])

    _t = _vec(tare(torch.tensor([[0.0, 0.0, _tare_m * -_tare_g]]), ident, hold_w, True))
    check("pure tool weight, upright, tares to zero",
          max(abs(v) for v in _t) < 1e-5)

    _t = _vec(tare(torch.tensor([[0.0, 0.0, _tare_m * -_tare_g + 12.0]]), ident, hold_w, True))
    check("a 12 N contact on top of the weight survives the tare",
          abs(_t[2] - 12.0) < 1e-5 and abs(_t[0]) < 1e-9 and abs(_t[1]) < 1e-9)

    # THE ROTATION IS THE POINT. A wrist turned 90 deg about x carries the same
    # world weight in a DIFFERENT body component. Subtracting a constant 8.083
    # from the z channel would leave 8.083 N of phantom force on y here, which
    # is the defect the rotation exists to prevent.
    c = math.cos(math.pi / 4.0)
    q_x90 = torch.tensor([[c, c, 0.0, 0.0]])           # +90 deg about x, wxyz
    # +90 deg about x puts the body y-axis on world +z, so a world +z force
    # reads on the body's +y component. Worked out from axes_from_quat's own
    # convention (axes are COLUMNS), which the checks above already pin.
    raw_x90 = torch.tensor([[0.0, _tare_m * -_tare_g, 0.0]])
    _t = _vec(tare(raw_x90, q_x90, hold_w, True))
    check("a 90 deg wrist turn still tares to zero (the rotation is applied)",
          max(abs(v) for v in _t) < 1e-5)

    # THE MUTATION (D-080): drop the rotation, i.e. subtract the world vector
    # straight from the body-frame reading. The upright case still passes --
    # which is exactly why it alone proves nothing -- and the turned case must
    # break, leaving the full weight behind.
    naive = [raw_x90.tolist()[0][i] - hold_w.tolist()[i] for i in range(3)]
    check("the mutation (no rotation) leaves the full weight on a turned wrist",
          abs(max(abs(v) for v in naive) - _tare_m * -_tare_g) < 1e-5)

    # ------------------------------------------------------------------
    # PRUEFBLOCK A (D-188, 2026-09-13): the torque tare, with the expected
    # moment worked out BY HAND and typed here, not recomputed by the code
    # under test. Lever (0.1, 0, 0.2) m from the reference point to the COM,
    # hold force (0, 0, 8.0834) N: r x F = (0*8.0834 - 0.2*0, 0.2*0 - 0.1*8.0834,
    # 0.1*0 - 0*0) = (0, -0.80834, 0) N m in WORLD. Whether PhysX's reading
    # has this sign and this reference point is NOT claimed here; the
    # --tare-check probe (RT-192) measures it.
    tare_q = ns["gravity_tare_torque"]
    lever = torch.tensor([[0.1, 0.0, 0.2]])
    hold_q = torch.tensor([0.0, 0.0, 8.0834])
    tau_w = [0.0, -0.80834, 0.0]
    check("the torque tare is a no-op when it is switched off",
          _vec(tare_q(torch.tensor([[0.0, 0.0, 0.5]]), ident, hold_q, lever, False))
          == [0.0, 0.0, 0.5])
    _t = _vec(tare_q(torch.tensor([tau_w]), ident, hold_q, lever, True))
    check("the pure weight moment r x F_hold, upright, tares to zero",
          max(abs(v) for v in _t) < 1e-9, str([round(v, 6) for v in _t]))
    _t = _vec(tare_q(torch.tensor([[tau_w[0] + 0.5, tau_w[1], tau_w[2]]]),
                     ident, hold_q, lever, True))
    check("a 0.5 N m contact moment on top of the weight moment survives the torque tare",
          abs(_t[0] - 0.5) < 1e-9 and abs(_t[1]) < 1e-9 and abs(_t[2]) < 1e-9,
          str([round(v, 6) for v in _t]))
    # THE ROTATION IS THE POINT, as for the force. +90 deg about x puts the
    # body z-axis on world -y (axes_from_quat, columns), so the world moment
    # (0, -0.80834, 0) reads on the body's +z component: R^T v, component 2
    # = col2 . v = (0, -1, 0) . (0, -0.80834, 0) = +0.80834.
    raw_q_x90 = torch.tensor([[0.0, 0.0, 0.80834]])
    _t = _vec(tare_q(raw_q_x90, q_x90, hold_q, lever, True))
    check("a 90 deg wrist turn still tares the weight moment to zero (the rotation is applied)",
          max(abs(v) for v in _t) < 1e-9, str([round(v, 6) for v in _t]))
    # The unrotated subtraction would leave |tau_w| on the turned wrist.
    naive_q = [raw_q_x90.tolist()[0][i] - tau_w[i] for i in range(3)]
    check("the mutation (torque not rotated) leaves the full weight moment on a turned wrist",
          abs(max(abs(v) for v in naive_q) - 0.80834) < 1e-9)

    def _d(step, fn, seated=False):
        return dones(torch.tensor([float(step)]), max_len, torch.tensor([float(fn)]),
                     f_max, torch.tensor([1.0 if seated else 0.0]) > 0.5)

    term, trunc, abort = _d(10, 1.0)
    check("mid-episode under the force limit, neither flag is set",
          (not bool(_f(term))) and (not bool(_f(trunc))))

    term, trunc, abort = _d(max_len - 1, 1.0)
    check("the last step sets truncated, not terminated",
          bool(_f(trunc)) and not bool(_f(term)))
    term, trunc, abort = _d(max_len - 2, 1.0)
    check("the step before the last does not truncate", not bool(_f(trunc)))

    # THE FORCE ABORT IS TRUNCATED, NOT TERMINATED (D-164, 2026-09-02), which
    # REVERSES the two flags this check used to assert. It read
    # "bool(term) and not bool(trunc)" and cited D-113; the abort is a
    # designer-imposed safety stop, not a terminal of the task's own MDP, so
    # rsl_rl must bootstrap gamma*V(s) for it. Written in both directions --
    # truncated set AND terminated clear -- because either half alone is
    # satisfied by a constant.
    term, trunc, abort = _d(10, f_max)
    check("force exactly at F_max aborts, as TRUNCATED not terminated",
          bool(_f(trunc)) and not bool(_f(term)) and bool(_f(abort)))
    term, trunc, abort = _d(10, f_max * 0.999)
    check("force just under F_max does not abort",
          not bool(_f(term)) and not bool(_f(trunc)) and not bool(_f(abort)))

    check("compute_dones returns three flags: terminated, truncated, force_abort",
          len(_d(10, 1.0)) == 3)

    # THE SUCCESS TERMINATION (2026-09-01, point (7)), which REVERSES the check
    # that stood here. It used to read "a seated, force-free env does NOT
    # terminate" and cited D-113 (1); the seated state is absorbing now, so the
    # same input must produce the opposite flag. The check is written in both
    # directions -- seated terminates, unseated does not -- because a
    # ``terminated = True`` constant would satisfy the first half alone.
    term, trunc, abort = _d(10, 0.0, seated=True)
    check("a seated, force-free env DOES terminate (the success exit)",
          bool(_f(term)) and not bool(_f(trunc)))
    check("a success termination is not reported as a force abort",
          not bool(_f(abort)))
    term, trunc, abort = _d(10, 0.0, seated=False)
    check("an unseated, force-free env still does not terminate",
          not bool(_f(term)) and not bool(_f(trunc)))
    # The two sources are separable, which is what the third flag is for: the
    # abort rate (D-113 (9)) must not count successes.
    term, trunc, abort = _d(10, f_max, seated=True)
    check("a seated env over F_max terminates and DOES report the abort flag",
          bool(_f(term)) and bool(_f(abort)))
    # D-164 made the two channels exclusive. Without that guard this env would
    # be terminated AND truncated at once, and the wrapper would hand rsl_rl a
    # time_out on a success step -- a bootstrap on top of the success lump.
    check("a seated env over F_max is NOT also truncated", not bool(_f(trunc)))
    term, trunc, abort = _d(max_len - 1, 0.0, seated=True)
    check("a success on the timeout step is terminated, not truncated",
          bool(_f(term)) and not bool(_f(trunc)))

    # -- the pocket cross-section (D-157) ------------------------------------
    # RT-107 measured the hole this closes: success_rate_recent 0.9765 at
    # mean_max_depth_mm 0.0. `depth` is an AXIS PROJECTION and carries no
    # lateral information, so a descent 109 mm BESIDE the fixture books its
    # height loss as insertion depth. The metric already gated on the pocket
    # walls; the two PAYING predicates did not.
    #
    # wall_x / wall_y are probe values, NOT the CAD numbers -- those live in
    # insertion_tasks_cfg.py (POCKET_WALL_X / POCKET_WALL_Y) and arrive here
    # as arguments, like every other geometry number in this file.
    cross = ns["in_pocket_cross_section"]
    wall_x, wall_y = 0.040, 0.070
    hack_y = 0.109                       # the RT-107 hack pose, measured

    def _c(x, y):
        return bool(_f(cross(torch.tensor([[float(x), float(y), 0.0]]), wall_x, wall_y)))

    check("a tip on the pocket axis is inside the cross-section", _c(0.0, 0.0))
    check("a tip past the short wall is outside", not _c(wall_x * 1.5, 0.0))
    check("a tip past the long wall is outside", not _c(0.0, wall_y * 1.5))
    check("the wall itself is OUTSIDE -- the test is strict, as the metric's was",
          (not _c(wall_x, 0.0)) and (not _c(0.0, wall_y)))
    check("the cross-section ignores depth -- it reads x and y only",
          bool(_f(cross(torch.tensor([[0.0, 0.0, 99.0]]), wall_x, wall_y))))
    got_c = _arr(cross(torch.tensor([[0.0, 0.0, 0.0],
                                     [0.0, hack_y, 0.0],
                                     [wall_x * 0.5, wall_y * 0.5, 0.0]]), wall_x, wall_y))
    check("the cross-section is batched, one flag per env",
          [bool(_f(v)) for v in got_c] == [True, False, True])

    # -- the pocket volume (2026-09-15, RT-201s3) ----------------------------
    # The cross-section alone let a tip BELOW the free-standing fixture count
    # as in the pocket: RT-201s3 booked ~144 mm of depth and was paid for it.
    # `under` is a probe value, NOT the CAD number (POCKET_ASSET_BOTTOM_Z).
    vol = ns["in_pocket_volume"]
    under = 0.046

    def _v(cs, depth):
        return bool(_f(vol(torch.tensor([1.0 if cs else 0.0]) > 0.5,
                           torch.tensor([float(depth)]), under)))

    check("a tip in the cross-section above the fixture underside is in the pocket",
          _v(True, 0.030))
    check("a tip in the cross-section BELOW the fixture underside is NOT (RT-201s3)",
          not _v(True, 0.144))
    check("the underside itself is inside -- the floor test is inclusive", _v(True, under))
    check("outside the cross-section is outside at any depth", not _v(False, 0.030))
    got_v = _arr(vol(torch.tensor([1.0, 1.0, 0.0]) > 0.5,
                     torch.tensor([0.030, 0.144, 0.030]), under))
    check("the pocket volume is batched, one flag per env",
          [bool(_f(v)) for v in got_v] == [True, False, False])

    # -- the unclean-step flag (2026-09-15, episodes.csv) ---------------------
    # `t` is a probe value, NOT INTERPEN_THRESH: 0.25 is exact in float32, so
    # the edge case tests the operator and not a rounding of the constant.
    unclean = ns["interpen_unclean"]
    t = 0.25

    def _u(d):
        return bool(_f(unclean(torch.tensor([float(d)]), t)))

    check("a step below the SAPU threshold is clean", not _u(0.125))
    check("a step AT the SAPU threshold is unclean -- the complement of the success filter's <",
          _u(t))
    check("a step above the SAPU threshold is unclean", _u(0.5))
    got_u = _arr(unclean(torch.tensor([0.0, 0.25, 0.75]), t))
    check("the unclean flag is batched, one flag per env",
          [bool(_f(v)) for v in got_u] == [False, True, True])

    # -- the success region --------------------------------------------------
    region = ns["in_success_region"]

    def _r(depth, interpen=0.0, in_pocket=True):
        return bool(_f(region(torch.tensor([float(depth)]), torch.tensor([float(interpen)]),
                              depth_min, depth_max, INTERPEN_THRESH,
                              torch.tensor([1.0 if in_pocket else 0.0]) > 0.5)))

    check("inside the band with a clean seat is a success region", _r(0.0215))
    check("too shallow is not the success region", not _r(depth_min - 0.001))
    check("too deep is not the success region -- it is a band, not a threshold",
          not _r(depth_max + 0.001))
    check("both band edges are inclusive", _r(depth_min) and _r(depth_max))
    check("a tunnelled seat is NOT a success", not _r(0.0215, INTERPEN_THRESH * 2.0))
    check("in the band but OUTSIDE the pocket is NOT a success (D-157)",
          not _r(0.0215, in_pocket=False))

    def _e(depth, in_pocket=True):
        return bool(_f(ns["is_engaged"](torch.tensor([float(depth)]), engaged_depth,
                                        torch.tensor([1.0 if in_pocket else 0.0]) > 0.5)))

    check("engaged is a depth threshold inside the pocket",
          _e(engaged_depth) and not _e(engaged_depth * 0.5))
    check("engaged does NOT fire outside the pocket (D-157)",
          not _e(engaged_depth, in_pocket=False))

    # THE RT-107 REGRESSION. One case, both paying predicates: depth sits in
    # the success band, the part sits 109 mm beside the pocket. Before D-157
    # this scored a success and paid the engaged bonus.
    hack_pocket = cross(torch.tensor([[0.0, hack_y, 0.0]]), wall_x, wall_y)
    check("RT-107 replay: seat depth 109 mm BESIDE the pocket is no success",
          not bool(_f(region(torch.tensor([0.0215]), torch.tensor([0.0]),
                             depth_min, depth_max, INTERPEN_THRESH, hack_pocket))))
    check("RT-107 replay: seat depth 109 mm BESIDE the pocket is not engaged",
          not bool(_f(ns["is_engaged"](torch.tensor([0.0215]), engaged_depth, hack_pocket))))

    # -- ``success_and_hold`` IS GONE (2026-09-01) --------------------------
    # It was episode success = ``in_region & is_last_step``. The success
    # termination makes the seating step the last step, so the conjunction can
    # only ever read False for a successful episode. Deleting the function is
    # the change; asserting that it stayed deleted is what this check is for --
    # a re-added "helpful" hold would silently zero the success metric.
    check("success_and_hold is gone from the math module",
          "success_and_hold" not in ns, str("success_and_hold" in ns))

    # =====================================================================
    #  SAPU reduction (D-109 point (10), ported from IsaacGymEnvs)
    # =====================================================================
    # The Warp array the reduction consumes is BUILT here, not by a second
    # kernel. `get_interpen_dist` differs from the SDF kernel in one line -- it
    # writes only negatives into a zeroed array -- so a clamp reproduces it.
    ifs = ns["interpen_from_signed"]
    signed = torch.tensor([[-0.0003, 0.0, 0.002, 1.5]])
    got_ifs = _arr(ifs(signed))
    check("a penetration survives the clamp unchanged",
          abs(_f(got_ifs[0][0]) + 0.0003) < 1e-15, f"{_f(got_ifs[0][0]):.6f}")
    check("a point exactly ON the surface stays zero",
          abs(_f(got_ifs[0][1])) < 1e-15, f"{_f(got_ifs[0][1]):.3e}")
    check("a clearance becomes zero, not a penetration",
          abs(_f(got_ifs[0][2])) < 1e-15, f"{_f(got_ifs[0][2]):.3e}")
    # The MISS case: our kernel writes +max_dist where the source leaves its
    # seeded 0.0. Both must reach the reduction as the same zero, or the two
    # implementations are not the same array.
    check("a miss (+max_dist) clamps to the source's seeded zero",
          abs(_f(got_ifs[0][3])) < 1e-15, f"{_f(got_ifs[0][3]):.3e}")
    check("the clamp keeps the (envs, points) shape",
          tuple(_arr(ifs(torch.tensor([[0.0, -0.001], [0.002, 0.0]]))).shape) == (2, 2))

    mid = ns["max_interpen_dist"]
    # EVERY fixture below is written in the KERNEL's convention: penetration
    # is NEGATIVE, a point that clears the mesh stays at zero (source:
    # industreal_algo_utils.py, `if signed_dist < 0.0` into a wp.zeros array).
    # An earlier version of this check fed hand-picked POSITIVE penetrations,
    # which is why it stayed green while the function returned zero on every
    # real input.
    #
    # One deep point among many shallow ones IS the tunnelling case. A mean
    # over 1000 sample points would hide it.
    spike = torch.tensor([[0.0, 0.0, 0.0, 0.0, -0.001]])
    check("the reduction takes the MAX, not the mean",
          abs(_f(mid(spike)) - 0.001) < 1e-15, f"{_f(mid(spike)):.6f}")
    untouched = torch.tensor([[0.0, 0.0, 0.0]])
    check("a part that touches nothing reports zero penetration",
          abs(_f(mid(untouched))) < 1e-15, f"{_f(mid(untouched)):.3e}")
    # Positive readings must not survive the negation as penetrations.
    clearance = torch.tensor([[0.002, 0.001, 0.005]])
    check("a positive reading clamps to zero", abs(_f(mid(clearance))) < 1e-15,
          f"{_f(mid(clearance)):.3e}")
    mixed = torch.tensor([[0.002, -0.0003, 0.005]])
    check("clearance elsewhere does not cancel a penetration",
          abs(_f(mid(mixed)) - 0.0003) < 1e-15, f"{_f(mid(mixed)):.6f}")
    batch = mid(torch.tensor([[0.0, 0.0], [0.0, -0.0005], [0.001, 0.001]]))
    check("the reduction is batched over envs", tuple(batch.shape) == (3,), str(tuple(batch.shape)))
    # The whole chain: a clean part keeps its return, one tunnelled point
    # discards it.
    check("a clean part keeps the full SAPU scale",
          abs(_f(sapu(mid(clearance), INTERPEN_THRESH)) - 1.0) < 1e-15)
    deep = torch.tensor([[0.0, 0.0, -INTERPEN_THRESH * 1.5]])
    check("ONE point past the threshold discards the whole env",
          _f(sapu(mid(deep), INTERPEN_THRESH)) == 0.0)

    # =====================================================================
    #  The UNSET guard in insertion_env_cfg.py
    # =====================================================================
    cfg_ns = load_cfg_guard(cfg_mutations)
    validate = cfg_ns["validate_rl_config"]
    Incomplete = cfg_ns["RlConfigIncomplete"]
    pending = cfg_ns["RL_PENDING"]
    marks = cfg_ns["RL_PENDING_MARKS"]
    placeholders = cfg_ns["RL_PLACEHOLDERS"]
    curriculum_pending = cfg_ns["CURRICULUM_PENDING"]

    check("every pending name carries a mark", set(pending) == set(marks),
          f"{sorted(set(pending) ^ set(marks))}")
    # ONE since 2026-08-30. It was five, then four, then three. Every departure
    # was for one of exactly two reasons, and both are asserted from here
    # rather than only counted:
    #   ``force_abort_f_max_n`` (2026-08-28) and ``abort_payment``
    #   (2026-08-30) left because they hold INVENTED numbers, so they must
    #   still be visible -- both moved to RL_PLACEHOLDERS and the checks below
    #   hold them there.
    #   ``force_sensor_body_name`` (RT-59, MEASURED) and ``engaged_depth_m``
    #   (DECIDED 2026-08-30, derived from POCKET_SEAT_DEPTH) left because they
    #   have real values. Such a value belongs in neither table, and putting it
    #   back into RL_PENDING would advertise a protection that no longer
    #   protects anything.
    # The count is asserted so that adding a second without a mark, or dropping
    # the last one quietly, cannot pass.
    check("the one open value is named", len(pending) == 1, str(pending))
    # CURRICULUM_PENDING is documented as a SUBSET of RL_PENDING: its names
    # stay in the big table so a rename is still a code fault and switching
    # the ladder on re-demands them without a second edit. A name that is in
    # the subset but not in the table would be demanded by neither state.
    check("CURRICULUM_PENDING is a subset of RL_PENDING",
          set(curriculum_pending) <= set(pending),
          f"curriculum={sorted(curriculum_pending)} pending={sorted(pending)}")
    check("the measured link name is NOT pending",
          "force_sensor_body_name" not in pending, str(pending))
    # The two names that left on 2026-08-30, asserted rather than assumed. A
    # regression that put either back would make the guard refuse to build an
    # env whose numbers are decided, and the failure would read as a code fault
    # instead of a stale table.
    check("the decided engaged depth is NOT pending",
          "engaged_depth_m" not in pending, str(pending))
    check("the abort payment is a placeholder, not pending",
          "abort_payment" in placeholders and "abort_payment" not in pending,
          f"pending={sorted(pending)} placeholders={sorted(placeholders)}")

    # -- the placeholder table (new 2026-08-28) -----------------------------
    # An UNSET value stops the run; an INVENTED one does not. That makes this
    # table the weaker of the two by construction, so the checks here are about
    # it staying VISIBLE: named, marked as invented, and never quietly counted
    # as pending as well -- which would let the guard appear to protect a field
    # that already holds a number.
    check("F_max is a named placeholder",
          "force_abort_f_max_n" in placeholders, str(sorted(placeholders)))
    check("a placeholder is never also pending",
          not (set(placeholders) & set(pending)),
          str(sorted(set(placeholders) & set(pending))))
    check("every placeholder mark says the number is invented",
          all("[placeholder]" in m for m in placeholders.values()),
          str(sorted(placeholders)))
    # A mark must not carry the VALUE. Both readers -- the startup report and
    # demo_metrics.json -- print the live field beside the mark, so a number in
    # the text is a second home for the same fact and goes stale the first time
    # anything moves the field. RT-69 moves it by CLI flag, and a mark reading
    # "50.0 N" next to a printed 200.0 is the RT-65 defect class again. Dates
    # and D-numbers are not values and stay allowed, so the test is on DIGITS
    # THAT ARE NOT PART OF A DATE OR A D-NUMBER.
    _mark_numbers = {
        name: sorted(set(re.findall(r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])", mark)))
        for name, mark in placeholders.items()
    }
    check("no placeholder mark restates the number it labels",
          not any(_mark_numbers.values()),
          str({k: v for k, v in _mark_numbers.items() if v}))

    class _Cfg:
        pass

    def _double(pending_value=None, placeholder_value=50.0, ladder=True, dr_mode="off"):
        """A cfg stand-in that mirrors the real @configclass.

        Every field EXISTS: pending ones may hold None, placeholder ones hold a
        number. A double that simply omitted the placeholders would trip the
        absent-field path and test a code fault instead of the unset one.

        ``ladder`` is ``curriculum_enabled``. It defaults to True because the
        five historical refusal checks below test the DEMAND side of the
        guard, and since 2026-08-30 the demand for ``rung_step_sizes`` only
        exists while the ladder is on -- with it off, an all-None double is a
        VALID config and there is no refusal to inspect. The ladder-off
        contract has its own named checks further down.
        """
        c = _Cfg()
        c.curriculum_enabled = ladder
        # Phase 5 step B6: the guard reads `dr_mode` for the ladder/AutoDR
        # collision, so the double carries it or every check below tests the
        # absent-field path instead of the one under test. 'off' is the cfg
        # default and the state every check here except the two new ones
        # wants.
        c.dr_mode = dr_mode
        # The three fields the 2026-09-01 reward terms added to the guard's
        # reach. They are neither pending nor placeholders -- both have a
        # source -- but the guard READS them, so a double without them tests
        # the absent-field path instead of the one under test. The values are
        # the decided RELATIONS, not copies of the config: T steps and -1/T.
        c.episode_steps = PROBE_EPISODE_STEPS
        c.time_penalty_per_step = -1.0 / PROBE_EPISODE_STEPS
        c.action_rate_scale = PROBE_ACTION_RATE_SCALE
        # D-165: read by the guard (sign check), so the double carries it.
        c.w_depth_progress = PROBE_W_PROGRESS
        # RT-171: the alignment term's weight, width and gamma are read by
        # the guard (sign / range checks), so the double carries them.
        c.w_tilt = PROBE_W_TILT
        c.kernel_a_tilt = PROBE_A_TILT
        c.shaping_gamma = PROBE_GAMMA
        for name in pending:
            setattr(c, name, pending_value)
        for name in placeholders:
            setattr(c, name, placeholder_value)
        return c

    empty = _double()
    try:
        validate(empty)
        check("an env with NOTHING set is refused", False, "no error raised")
    except Incomplete as exc:
        check("an env with NOTHING set is refused", True, "")
        text = str(exc)
        check("the refusal names every missing value at once",
              all(name in text for name in pending))
        # ASSERTED AGAINST THE TABLE, not against a literal mark string. This
        # line used to read ``"[open]" in text and "[CAD pending]" in text``,
        # and "[CAD pending]" died on 2026-08-30 when ``engaged_depth_m`` was
        # decided -- a check pinned to a string that no mark contains any more
        # is unfailable, which is the failure mode the "[measure]" note already
        # warned about once. Reading the marks out of the table cannot go
        # stale.
        # ``marks.get``, not ``marks[...]``: a mutation may put a name into
        # RL_PENDING that has no mark, and this check must FAIL on that rather
        # than raise KeyError and take the whole run down. A missing mark
        # cannot be in the text, so the sentinel is any string the message
        # cannot contain.
        check("the refusal carries each value's mark",
              all(marks.get(name, "\0") in text for name in pending), text[:70])
    except Exception as exc:  # noqa: BLE001
        check("an env with NOTHING set is refused", False,
              f"raised {type(exc).__name__} instead of RlConfigIncomplete")

    # Both blocks below index ``pending``, so they can only run while the table
    # has an entry. The guard is not decoration: since 2026-08-30 exactly ONE
    # name is left, and the mutation that drops it would otherwise raise
    # IndexError here -- the harness would crash instead of flipping a check,
    # and a crash proves nothing. Under that mutation these checks DISAPPEAR,
    # which the counter-proof counts as a flip and the mutation declares.
    if pending:
        one_left = _double(1.0)
        setattr(one_left, pending[-1], None)
        try:
            validate(one_left)
            check("ONE unset value is enough to refuse", False, "no error raised")
        except Incomplete as exc:
            check("ONE unset value is enough to refuse", True, "")
            # NAMED AGAINST THE OTHER TABLE since 2026-08-30. It used to read
            # ``pending[-1] in text and pending[0] not in text``, which needed
            # TWO pending names to say anything; with one left it asserted that
            # a name is both present and absent and could never pass. The
            # property under test is unchanged -- the refusal names what is
            # missing and nothing else -- so the "nothing else" is now the
            # placeholder table, whose fields all hold values here.
            text = str(exc)
            check("the refusal names only what is actually missing",
                  pending[-1] in text
                  and not any(name in text for name in placeholders),
                  text[:70])
        except Exception:  # noqa: BLE001
            check("ONE unset value is enough to refuse", False, "wrong exception type")

    # A name in RL_PENDING that the cfg does not carry is a code fault, not a
    # missing measurement. Before 2026-08-27 getattr(cfg, name, None) read the
    # two as one, so a typo produced a refusal that told the reader to go and
    # measure a number that already had a home.
    if pending:
        absent_cfg = _double(1.0)
        delattr(absent_cfg, pending[0])
        try:
            validate(absent_cfg)
            check("a field the guard names but the cfg lacks is refused", False,
                  "no error raised")
        except Incomplete as exc:
            text = str(exc)
            check("a field the guard names but the cfg lacks is refused", True, "")
            check("the code-fault refusal names the absent field",
                  pending[0] in text, text[:70])
            check("the code-fault refusal does not blame a measurement",
                  "[open]" not in text and "must be measured" not in text, text[:70])
        except Exception as exc:  # noqa: BLE001
            check("a field the guard names but the cfg lacks is refused", False,
                  f"raised {type(exc).__name__} instead of RlConfigIncomplete")

    # A placeholder set back to None is neither of the two failures above. The
    # number was never measured, so "go and measure it" is wrong about what just
    # happened -- somebody removed a value the env was built to read.
    emptied_cfg = _double(1.0, placeholder_value=None)
    try:
        validate(emptied_cfg)
        check("an emptied placeholder is refused", False, "no error raised")
    except Incomplete as exc:
        text = str(exc)
        check("an emptied placeholder is refused", True, "")
        check("the placeholder refusal names the placeholder",
              all(name in text for name in placeholders), text[:70])
        check("the placeholder refusal does not send anyone off to measure",
              "must be measured" not in text, text[:70])
    except Exception as exc:  # noqa: BLE001
        check("an emptied placeholder is refused", False,
              f"raised {type(exc).__name__} instead of RlConfigIncomplete")

    full = _double(1.0)
    try:
        validate(full)
        check("a fully configured env passes the guard", True)
    except Exception as exc:  # noqa: BLE001
        check("a fully configured env passes the guard", False, str(exc)[:60])

    # -- the ladder-off contract (curriculum_enabled, 2026-08-30) -----------
    # With the ladder OFF exactly two states are legal, and both directions
    # are asserted. Unset passes: this is the first-training configuration
    # (D-110 (3) ring -- the step sizes come from learning curves, which need
    # a run, which must therefore start without them). Set is REFUSED: a step
    # size nothing reads is a number that looks decided and is not -- the old
    # check_seated_success TEST SENTINEL was exactly this shape.
    off_unset = _double(None, ladder=False)
    try:
        validate(off_unset)
        check("a ladder-off env with no step size passes the guard", True)
    except Exception as exc:  # noqa: BLE001
        check("a ladder-off env with no step size passes the guard", False,
              str(exc)[:70])

    off_set = _double(1.0, ladder=False)
    try:
        validate(off_set)
        check("a step size set while the ladder is OFF is refused", False,
              "no error raised")
    except Incomplete as exc:
        text = str(exc)
        check("a step size set while the ladder is OFF is refused", True, "")
        check("the ladder-off refusal names the unread value",
              all(name in text for name in curriculum_pending), text[:70])
    except Exception as exc:  # noqa: BLE001
        check("a step size set while the ladder is OFF is refused", False,
              f"raised {type(exc).__name__} instead of RlConfigIncomplete")

    # -- the ladder/AutoDR collision (Phase 5 step B6, plan section 10) -----
    # They overlap on TWO axes, not three: a rung is a half-width on start
    # height, FIXTURE OFFSET and tilt (`check_curriculum_ladder.AXES`), and
    # AutoDR tracks `start_height` and `tilt` but NOT `fixture_pos_noise_xy`
    # -- its `lat_r` is `start_lateral_offset`, a different quantity.
    # On those two both would write the same reset quantity from their own
    # state and the applied range would be whichever ran last, which is enough
    # to refuse. BOTH DIRECTIONS are asserted,
    # because a guard that refuses everything passes the refusal half on its
    # own: AutoDR with the ladder OFF is the whole Phase 5 study and must pass.
    #
    # NOT LIVE TODAY: `Ladder` is not wired into the env (M2.5). The guard is
    # the thing that keeps the combination from appearing the day it is.
    # FULLY configured (`1.0`, not `None`): with the pending values unset the
    # guard refuses this double several branches EARLIER, and the check below
    # would then pass with the ladder/AutoDR guard deleted. The counter-proof
    # caught exactly that -- reading the code did not.
    both_on = _double(1.0, ladder=True, dr_mode="autodr")
    try:
        validate(both_on)
        check("the ladder and AutoDR together are refused", False, "no error raised")
    except Incomplete as exc:
        text = str(exc)
        check("the ladder and AutoDR together are refused", True, text[:70])
        check("the ladder/AutoDR refusal names both mechanisms",
              "curriculum_enabled" in text and "dr_mode" in text
              and "autodr" in text, text[:90])
    except Exception as exc:  # noqa: BLE001
        check("the ladder and AutoDR together are refused", False,
              f"raised {type(exc).__name__} instead of RlConfigIncomplete")

    autodr_only = _double(None, ladder=False, dr_mode="autodr")
    try:
        validate(autodr_only)
        check("AutoDR with the ladder off passes the guard", True)
    except Exception as exc:  # noqa: BLE001
        check("AutoDR with the ladder off passes the guard", False, str(exc)[:70])

    # -- D-165: the guard refuses a weight that would CHARGE for entering ---
    neg_progress = _double(None, ladder=False)
    neg_progress.w_depth_progress = -1.0
    # The refusal TYPE is asserted, not its text: under the other cfg
    # mutations in the table the same double is refused one guard earlier,
    # and a text match here would make three unrelated mutations flip this.
    try:
        validate(neg_progress)
        check("a negative depth-progress weight is refused", False, "no error raised")
    except Incomplete as exc:
        check("a negative depth-progress weight is refused", True, str(exc)[:70])
    except Exception as exc:  # noqa: BLE001
        check("a negative depth-progress weight is refused", False,
              f"raised {type(exc).__name__} instead of RlConfigIncomplete")

    # -- RT-171: a weight that would PAY for tilting, a gamma that is no
    #    discount. Same shape as the D-165 check: the TYPE is asserted.
    neg_tilt = _double(None, ladder=False)
    neg_tilt.w_tilt = -1.0
    try:
        validate(neg_tilt)
        check("a negative tilt weight is refused", False, "no error raised")
    except Incomplete as exc:
        check("a negative tilt weight is refused", True, str(exc)[:70])
    except Exception as exc:  # noqa: BLE001
        check("a negative tilt weight is refused", False,
              f"raised {type(exc).__name__} instead of RlConfigIncomplete")
    bad_gamma = _double(None, ladder=False)
    bad_gamma.shaping_gamma = 1.5
    try:
        validate(bad_gamma)
        check("a shaping gamma outside (0, 1] is refused", False, "no error raised")
    except Incomplete as exc:
        check("a shaping gamma outside (0, 1] is refused", True, str(exc)[:70])
    except Exception as exc:  # noqa: BLE001
        check("a shaping gamma outside (0, 1] is refused", False,
              f"raised {type(exc).__name__} instead of RlConfigIncomplete")

    # -- D-182 / D-183: a negative SCATTER WIDTH. Nothing downstream would
    #    report it: `std * randn` is symmetric, so a negative sigma draws
    #    exactly the same distribution and the run looks right, while the
    #    uniform grasp draw simply turns inside out. Same shape as the two
    #    checks above -- the refusal TYPE is asserted, not its text, because
    #    other cfg mutations refuse this double one guard earlier.
    neg_sigma = _double(None, ladder=False)
    neg_sigma.force_obs_noise_std_n = -1.0
    try:
        validate(neg_sigma)
        check("a negative observation-scatter width is refused", False, "no error raised")
    except Incomplete as exc:
        check("a negative observation-scatter width is refused", True, str(exc)[:70])
    except Exception as exc:  # noqa: BLE001
        check("a negative observation-scatter width is refused", False,
              f"raised {type(exc).__name__} instead of RlConfigIncomplete")

    # -- batching (house rule: no python loop over envs) ---------------------
    n = 5
    rb = ns["compute_rewards_insertion"](
        torch.tensor([0.0, 0.001, 0.01, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, INTERPEN_THRESH * 2.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]) > 0.5,
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]) > 0.5,
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]) > 0.5,
        torch.tensor([0.0, 0.1, 0.2, 0.0, 0.0]),
        torch.tensor([10.0, 0.0, 0.0, 10.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        A_COARSE, B_COARSE, A_MID, B_MID, A_FINE, B_FINE,
        W_ENGAGED, W_SUCCESS, 0.0, INTERPEN_THRESH, PROBE_ABORT_PENALTY,
        0.0, 0.0, 0.0,
    )
    check("batched call returns one reward per env", tuple(rb.shape) == (n,), str(tuple(rb.shape)))
    check("the batched tunnelling env is the zero one", abs(_f(rb[3:4])) < 1e-15)

    # =====================================================================
    #  D-165: the depth-progress term (2026-09-02, after RT-134/RT-135).
    #  The proxy's `progress = w * (new_max_depth - max_depth)`, handed in
    #  as the metres of NEW gated max depth. FORM checks only; the weight is
    #  a [proxy] placeholder and its size is not asserted anywhere.
    # =====================================================================
    r_plain = _f(_rewards(ns, 0.01))
    check("no new max depth pays no progress",
          abs(_f(_rewards(ns, 0.01, depth_progress=0.0, w_progress=PROBE_W_PROGRESS)) - r_plain) < 1e-15)
    check("the depth progress pays w_progress per metre of new max depth",
          abs(_f(_rewards(ns, 0.01, depth_progress=0.001, w_progress=PROBE_W_PROGRESS))
              - r_plain - PROBE_W_PROGRESS * 0.001) < 1e-12)
    check("the depth progress is discarded by SAPU like the task return",
          abs(_f(_rewards(ns, 0.01, interpen=INTERPEN_THRESH * 2.0, depth_progress=0.001,
                          w_progress=PROBE_W_PROGRESS))) < 1e-15)
    # The lump reads step_task; a one-off progress delta must NOT be bought
    # out for the steps remaining -- that would pay 201x for one millimetre.
    paid_plain = _f(_rewards(ns, 0.0, engaged=True, success=True, steps_remaining=200.0))
    paid_prog = _f(_rewards(ns, 0.0, engaged=True, success=True, steps_remaining=200.0,
                            depth_progress=0.001, w_progress=PROBE_W_PROGRESS))
    check("the depth progress is NOT multiplied into the success payout",
          abs((paid_prog - paid_plain) - PROBE_W_PROGRESS * 0.001) < 1e-9,
          f"delta {paid_prog - paid_plain:.4f}")
    # The regression switch: w = 0 with a live progress tensor is the old reward.
    rb_w0 = ns["compute_rewards_insertion"](
        torch.tensor([0.0, 0.001, 0.01, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, INTERPEN_THRESH * 2.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]) > 0.5,
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]) > 0.5,
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]) > 0.5,
        torch.tensor([0.0, 0.1, 0.2, 0.0, 0.0]),
        torch.tensor([10.0, 0.0, 0.0, 10.0, 0.0]),
        torch.tensor([0.001, 0.002, 0.003, 0.004, 0.005]),
        # RT-171: live potentials with gamma 0 and equal Phi on both sides
        # are the row's own regression switch, 0 - 0 does not enter here.
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        A_COARSE, B_COARSE, A_MID, B_MID, A_FINE, B_FINE,
        W_ENGAGED, W_SUCCESS, 0.0, INTERPEN_THRESH, PROBE_ABORT_PENALTY,
        0.0, 0.0, 0.0,
    )
    check("w_progress = 0 reproduces the previous reward",
          float(torch.amax(torch.abs(rb_w0 - rb))) < 1e-15)

    # =====================================================================
    #  The per-term stack (2026-09-02, after RT-134). The env logs one row
    #  per REWARD_TERMS entry; the table and the stack must agree, and each
    #  row must carry the term it is named after. Five envs: 0 seated
    #  (engaged + success, lump), 1 engaged only, 2 free and making NEW
    #  depth (progress), 3 tunnelled seat, 4 force abort.
    # =====================================================================
    term_names = tuple(ns["REWARD_TERMS"])
    tb = ns["compute_reward_terms_insertion"](
        torch.tensor([0.0, 0.001, 0.01, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, INTERPEN_THRESH * 2.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0]) > 0.5,
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]) > 0.5,
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]) > 0.5,
        torch.tensor([0.0, 0.1, 0.2, 0.0, 0.0]),
        torch.tensor([10.0, 0.0, 0.0, 10.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.001, 0.0, 0.0]),
        # RT-171 potentials: env 1 rights itself (0.5 -> 0.9), env 2 tilts
        # (0.9 -> 0.5), the rest are upright and unchanged (1.0 -> 1.0)
        torch.tensor([1.0, 0.9, 0.5, 1.0, 1.0]),
        torch.tensor([1.0, 0.5, 0.9, 1.0, 1.0]),
        A_COARSE, B_COARSE, A_MID, B_MID, A_FINE, B_FINE,
        W_ENGAGED, W_SUCCESS, PROBE_W_PROGRESS, INTERPEN_THRESH, PROBE_ABORT_PENALTY,
        PROBE_TIME_PENALTY, PROBE_ACTION_RATE_SCALE,
        PROBE_GAMMA,
    )
    check("REWARD_TERMS names every row of the stack, once",
          tuple(tb.shape) == (len(term_names), n) and len(set(term_names)) == len(term_names),
          f"{tuple(tb.shape)} vs {len(term_names)} names")
    # The LABELLING check reads WHICH envs a row is active on, never its
    # value -- the values are every other check's business, and a value
    # assertion here would flip under every arithmetic mutation in the table.
    # Env 3 (the tunnelled seat) is left out on purpose: SAPU zeroes the task
    # rows there, and that zero is the SAPU checks' claim, not this one's.
    row = {name: tb[i] for i, name in enumerate(term_names) if i < int(tb.shape[0])}

    def _col(name: str, env: int) -> float:
        return _f(row[name][env:env + 1])

    def _on(name: str, env: int) -> bool:
        return name not in row or abs(_col(name, env)) > 1e-15

    def _off(name: str, env: int) -> bool:
        return name not in row or abs(_col(name, env)) < 1e-15

    rows_ok = bool(row) and all([
        # kernels: alive on every clean env
        _on("kernels", 0) and _on("kernels", 1) and _on("kernels", 2),
        # engaged: envs 0 and 1 (engaged), not 2 (free) or 4 (abort)
        _on("engaged", 0) and _on("engaged", 1),
        _off("engaged", 2) and _off("engaged", 4),
        # success: env 0 only -- env 1 is engaged WITHOUT success, which is
        # what tells the two bonus rows apart
        _on("success", 0),
        _off("success", 1) and _off("success", 2) and _off("success", 4),
        # progress: env 2 alone made new depth
        _on("progress", 2),
        _off("progress", 0) and _off("progress", 1) and _off("progress", 4),
        # time: one value on every env, terminal ones included
        ("time" not in row) or len({round(_col("time", e), 15) for e in range(n)}) == 1,
        # action_rate: envs 1 and 2 moved, 0 and 4 did not
        _on("action_rate", 1) and _on("action_rate", 2),
        _off("action_rate", 0) and _off("action_rate", 4),
        # success_lump: nothing where there is no success (1, 2, 4)
        _off("success_lump", 1) and _off("success_lump", 2) and _off("success_lump", 4),
        # abort: env 4 alone
        _on("abort", 4),
        _off("abort", 0) and _off("abort", 1) and _off("abort", 2),
        # tilt_shaping (RT-171): env 1 rights itself (0.5 -> 0.9) and is
        # PAID, env 2 tilts (0.9 -> 0.5) and is CHARGED, env 0 pays -Phi(s)
        # on its success step because Phi(s') is zeroed there. Env 4's
        # truncation keeps the real Phi(s'), which is the value block's claim.
        ("tilt_shaping" not in row) or (
            _col("tilt_shaping", 1) > 0.0 and _col("tilt_shaping", 2) < 0.0
            and _col("tilt_shaping", 0) < 0.0),
    ])
    check("the term rows carry the terms they are named after", rows_ok,
          "" if rows_ok else {k: [round(_col(k, e), 4) for e in range(n)] for k in row})


    # =====================================================================
    #  The task-space action (inbox entry "Audit 2026-09-03 (a)")
    # =====================================================================
    # target = current + delta, a box on the leading tool point, a cone on
    # the tool axis. Checked against KNOWN rotations, like the pose algebra
    # above; the controller that consumes the target is Isaac Lab's and is
    # not under test here.
    quat_from_axis_angle = ns["quat_from_axis_angle"]
    quat_rotate = ns["quat_rotate"]
    apply_pose_delta = ns["apply_pose_delta"]
    clamp_tip_in_box = ns["clamp_tip_in_box"]
    tool_axis_tilt = ns["tool_axis_tilt"]
    clamp_tilt_to_cone = ns["clamp_tilt_to_cone"]
    _R2 = math.sqrt(0.5)
    _q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    _q_down = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # 180 deg about x: tool +z points DOWN
    _q_z90 = torch.tensor([[_R2, 0.0, 0.0, _R2]])

    check("a zero axis-angle is the identity quaternion",
          _maxabs(quat_from_axis_angle(torch.zeros(1, 3)), _q_id) < 1e-12)
    check("90 deg about z as an axis-angle is the known quaternion",
          _maxabs(quat_from_axis_angle(torch.tensor([[0.0, 0.0, math.pi / 2]])), _q_z90) < 1e-12)
    check("quat_rotate sends x to y under 90 deg about z",
          _maxabs(quat_rotate(_q_z90, torch.tensor([[1.0, 0.0, 0.0]])), _np_row(0.0, 1.0, 0.0)) < 1e-12)
    _p0 = torch.tensor([[0.1, 0.2, 0.3]])
    _dp = torch.tensor([[0.01, -0.02, 0.005]])
    _pos1, _quat1 = apply_pose_delta(_p0, _q_down, _dp, torch.zeros(1, 3))
    check("a zero rotation delta leaves the orientation bit-identical and adds the translation",
          _maxabs(_quat1, _q_down) < 1e-12 and _maxabs(_pos1, _p0 + _dp) < 1e-12)
    # PRE-multiplied (env frame): +90 deg about env z turns the body x axis to
    # +y even though the body hangs upside down. Post-multiplied (tool frame)
    # it would turn it to -y, because the 180 deg about x flips z.
    _pos2, _quat2 = apply_pose_delta(_p0, _q_down, torch.zeros(1, 3), torch.tensor([[0.0, 0.0, math.pi / 2]]))
    _x_axis2 = _arr(axes_from_quat(_quat2))[:, :, 0]
    check("the rotation delta is applied in the ENV frame (pre-multiplied, Factory)",
          _maxabs(_x_axis2, _np_row(0.0, 1.0, 0.0)) < 1e-12,
          str(_x_axis2.round(6).tolist()))
    check("the rotation delta keeps the tool axis down (rotation about z)",
          abs(float(tool_axis_tilt(_quat2)[0])) < 1e-9)

    _tip_off = torch.tensor([0.0, 0.0, 0.15])
    _centre = torch.zeros(1, 3)
    _inside = torch.tensor([[0.0, 0.0, 0.15]])  # tool down: tip = pos + (0,0,-0.15) = centre
    check("a tip inside the box leaves the body position bit-identical",
          _maxabs(clamp_tip_in_box(_inside, _q_down, _tip_off, _centre, 0.05, 0.05, 0.07),
                  _inside) < 1e-12)
    _out = torch.tensor([[0.07, 0.0, 0.15]])
    _clamped = _arr(clamp_tip_in_box(_out, _q_down, _tip_off, _centre, 0.05, 0.05, 0.07))
    check("a tip outside the box is pulled onto its face, and the BODY moves with it",
          _maxabs(_clamped, _np_row(0.05, 0.0, 0.15)) < 1e-12,
          str(_clamped.round(6).tolist()))
    # -- THE BOX HAS THREE HALF-WIDTHS (Phase 5 step B7) -------------------
    # The z axis carries the whole start band and needs room for one action
    # step above the highest legal start; x and y never did. A single
    # half-width made the two share a number.
    #
    # Each axis is probed at a point that is OUTSIDE the 0.05 box and INSIDE
    # the 0.07 one, so a check can only pass if the third argument reached the
    # right axis. The same displacement is used on all three, which is what
    # makes the pair of expectations (x/y clamped at 0.05, z untouched)
    # separate the axes rather than the magnitudes.
    _mid = torch.tensor([[0.06, 0.0, 0.15]])   # tip at x = +60 mm
    _mid_c = _arr(clamp_tip_in_box(_mid, _q_down, _tip_off, _centre, 0.05, 0.05, 0.07))
    check("a 60 mm tip offset along x is still clamped at the x half-width",
          _maxabs(_mid_c, _np_row(0.05, 0.0, 0.15)) < 1e-12,
          str(_mid_c.round(6).tolist()))
    _mid_y = torch.tensor([[0.0, 0.06, 0.15]])
    _mid_yc = _arr(clamp_tip_in_box(_mid_y, _q_down, _tip_off, _centre, 0.05, 0.05, 0.07))
    check("a 60 mm tip offset along y is still clamped at the y half-width",
          _maxabs(_mid_yc, _np_row(0.0, 0.05, 0.15)) < 1e-12,
          str(_mid_yc.round(6).tolist()))
    # tool down, tip = pos + (0, 0, -0.15): a body at z = 0.21 puts the tip at
    # +60 mm, outside 0.05 and inside 0.07 -- untouched, and the OLD single
    # half-width would have pulled it to 0.05.
    _mid_z = torch.tensor([[0.0, 0.0, 0.21]])
    _mid_zc = _arr(clamp_tip_in_box(_mid_z, _q_down, _tip_off, _centre, 0.05, 0.05, 0.07))
    check("a 60 mm tip offset along z passes, because z has its OWN wider half-width",
          _maxabs(_mid_zc, _np_row(0.0, 0.0, 0.21)) < 1e-12,
          str(_mid_zc.round(6).tolist()))
    _high_z = torch.tensor([[0.0, 0.0, 0.25]])  # tip at +100 mm, past 0.07
    _high_zc = _arr(clamp_tip_in_box(_high_z, _q_down, _tip_off, _centre, 0.05, 0.05, 0.07))
    check("a tip past the z half-width is clamped at z, not at the x/y one",
          _maxabs(_high_zc, _np_row(0.0, 0.0, 0.22)) < 1e-12,
          str(_high_zc.round(6).tolist()))
    # Three EQUAL half-widths must reproduce the pre-B7 body bit for bit. If
    # this ever disagrees at the isotropic point, the change was not a
    # per-axis split but a different clamp.
    _iso = torch.tensor([[0.07, -0.09, 0.23], [0.01, 0.02, 0.16]])
    _iso_q = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    _iso_c = _arr(clamp_tip_in_box(_iso, _iso_q, _tip_off, torch.zeros(2, 3), 0.05, 0.05, 0.05))
    # The OLD body written out longhand in numpy -- one clamp over the whole
    # displacement -- rather than trusted to a paragraph.
    import numpy as _np

    _iso_off = _arr(quat_rotate(_iso_q, _tip_off.reshape(1, 3) + torch.zeros(2, 3)))
    _iso_old = _np.clip(_arr(_iso) + _iso_off, -0.05, 0.05) - _iso_off
    check("three EQUAL half-widths reproduce the single-half-width body bit for bit",
          _maxabs(_iso_c, _iso_old) < 1e-15,
          f"max abs difference {float(_np.abs(_iso_c - _iso_old).max()):.3e}")
    # -- what the round-2 rubric critic walked through ---------------------
    # THE BOX IS MEASURED FROM ITS CENTRE. Every probe above centres it on the
    # ORIGIN, where `tip - centre` and `tip` are the same expression, so the
    # centring itself was never exercised here at all -- it was pinned only as
    # a call-site string in check_env_wiring. The env centres the box on the
    # pocket entrance, a metre from the origin, and dropping the subtraction
    # there moves the box to the world origin and takes the arm with it.
    _off_centre = torch.tensor([[0.4, -0.3, 0.9]])
    _oc_body = torch.tensor([[0.46, -0.3, 1.05]])  # tool down: tip 60 mm +x of the centre
    _oc_c = _arr(clamp_tip_in_box(_oc_body, _q_down, _tip_off, _off_centre, 0.05, 0.05, 0.07))
    check("the box is measured from its CENTRE, not from the world origin",
          _maxabs(_oc_c, _np_row(0.45, -0.3, 1.05)) < 1e-12,
          str(_oc_c.round(6).tolist()))
    # AND EACH AXIS TAKES ITS OWN HALF-WIDTH, proved with THREE DIFFERENT
    # numbers. The probes above pass 0.05, 0.05, 0.07, so x and y share a
    # value and any exchange between them is invisible -- in the check, in the
    # call site (both arguments read `osc_pos_clamp_m`) and therefore
    # everywhere. One displacement, three different clamps, one expectation
    # per axis.
    _tri = torch.tensor([[0.06, 0.06, 0.21]])  # tip at (+60, +60, +60) mm
    _tri_c = _arr(clamp_tip_in_box(_tri, _q_down, _tip_off, _centre, 0.03, 0.05, 0.07))
    check("each axis takes ITS OWN half-width, proved with three DIFFERENT ones",
          _maxabs(_tri_c, _np_row(0.03, 0.05, 0.21)) < 1e-12,
          str(_tri_c.round(6).tolist()))

    check("the tool axis tilt reads 0 for a part hanging straight down",
          abs(float(tool_axis_tilt(_q_down)[0])) < 1e-12)
    check("the tool axis tilt reads 180 deg for a part pointing up",
          abs(float(tool_axis_tilt(_q_id)[0]) - math.pi) < 1e-9)
    _q_tilt8 = quat_mul(quat_from_axis_angle(torch.tensor([[0.0, math.radians(8.0), 0.0]])), _q_down)
    _q_tilt20 = quat_mul(quat_from_axis_angle(torch.tensor([[0.0, math.radians(20.0), 0.0]])), _q_down)
    check("a 20 deg lean about y reads 20 deg",
          abs(float(tool_axis_tilt(_q_tilt20)[0]) - math.radians(20.0)) < 1e-9)
    _cone = math.radians(8.52)
    # AN UPRIGHT POCKET is the reference these four checks were written in
    # (step B7 gave the cone a reference frame; before that it was the world
    # vertical, which is what an upright pocket's own down direction is).
    # `_q_id` is the identity orientation, so its -z IS (0, 0, -1).
    _q_ref_up = _q_id
    check("a lean inside the cone comes back bit-identical",
          _maxabs(clamp_tilt_to_cone(_q_tilt8, _cone, _q_ref_up), _q_tilt8) < 1e-12)
    _q_cl = clamp_tilt_to_cone(_q_tilt20, _cone, _q_ref_up)
    check("a lean beyond the cone is pulled back to EXACTLY the cone angle",
          abs(float(tool_axis_tilt(_q_cl)[0]) - _cone) < 1e-9,
          f"{math.degrees(float(tool_axis_tilt(_q_cl)[0])):.4f} deg")
    _axis_cl = _arr(axes_from_quat(_q_cl))[:, :, 2]
    # A +20 deg lean about +y puts the tool axis at (-sin 20, 0, -cos 20); the
    # pull-back must land it at (-sin 8.52, 0, -cos 8.52): same plane, same
    # side, shorter lean. Any other x would be a rotation AROUND the vertical.
    check("the pull-back moves the tool axis STRAIGHT toward vertical (stays in its own plane)",
          abs(float(_axis_cl[0, 1])) < 1e-9 and abs(float(_axis_cl[0, 0]) + math.sin(_cone)) < 1e-9,
          str(_axis_cl.round(6).tolist()))
    check("the clamped orientation is still a unit quaternion",
          abs(float(torch.linalg.norm(_q_cl, dim=-1)[0]) - 1.0) < 1e-9)
    _q_yawed = quat_mul(_q_z90, _q_down)
    check("yaw stays free: a yawed upright part is not touched by the cone",
          _maxabs(clamp_tilt_to_cone(_q_yawed, _cone, _q_ref_up), _q_yawed) < 1e-12)
    _q_up_cl = clamp_tilt_to_cone(_q_id, _cone, _q_ref_up)
    check("a part pointing straight up is pulled to the cone without NaN",
          bool(torch.all(torch.isfinite(_q_up_cl)))
          and abs(float(tool_axis_tilt(_q_up_cl)[0]) - _cone) < 1e-6)

    # -- the cone's REFERENCE FRAME (Phase 5 step B7) ----------------------
    #
    # The 8.52 deg is a CAD angle BETWEEN part and pocket. Until B7 the clamp
    # measured it from the world vertical, and under AutoDR the pocket tilts
    # to 8.00 deg -- so merely aligning with the pocket spent 8.00 of the 8.52
    # and the policy kept 0.52 deg. The number is unchanged; the frame is now
    # the one the number's source is stated in.

    # (1) AN UPRIGHT REFERENCE REPRODUCES THE OLD CONSTRUCTION, BIT FOR BIT.
    # This is the regression proof and it is written out longhand rather than
    # trusted: the pre-B7 body computed `tilt` from `-axis[:, 2]`, the axis as
    # `(-a_y, a_x, 0)` and the degenerate fallback as world x. If the new body
    # and this one ever disagree at the identity, the change was not a frame
    # correction.
    def _cone_world(q, cone_rad, eps=1.0e-8):
        """The pre-B7 body, verbatim, for comparison only."""
        axis = axes_from_quat(q)[:, :, 2]
        tilt = torch.acos(torch.clamp(-axis[:, 2], min=-1.0, max=1.0))
        excess = torch.clamp(tilt - cone_rad, min=0.0)
        n = torch.stack((-axis[:, 1], axis[:, 0], torch.zeros_like(axis[:, 0])), dim=-1)
        n_norm = torch.linalg.norm(n, dim=-1)
        degenerate = n_norm < eps
        n_safe = torch.where(degenerate.reshape(-1, 1), torch.zeros_like(n), n)
        n_safe[:, 0] = torch.where(degenerate, torch.ones_like(n_norm), n_safe[:, 0])
        n_unit = n_safe / torch.where(degenerate, torch.ones_like(n_norm), n_norm).reshape(-1, 1)
        return quat_mul(quat_from_axis_angle(n_unit * excess.reshape(-1, 1)), q)

    _probe = quat_mul(
        quat_from_axis_angle(torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, math.radians(3.0), 0.0],
            [math.radians(-14.0), 0.0, 0.0],
            [math.radians(5.0), math.radians(-7.0), math.radians(40.0)],
            [0.0, math.radians(179.0), 0.0],
            # STRAIGHT UP, added in round 2. The round-1 entry claimed this
            # set covered "the degenerate straight-up case" and it did not:
            # 179 deg leaves the correction axis at norm 1.7e-02, six orders
            # above the 1e-08 threshold, and the one row that IS degenerate
            # (tilt 0) has excess 0, so the fallback axis cannot reach the
            # output there. Without this row the regression proof never
            # compared the two fallback axes -- world x against the pocket's
            # own x, which are the same vector ONLY at an upright reference.
            # That identity is the whole claim, and it went untested.
            [0.0, math.pi, 0.0],
        ])),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]] * 6),
    )
    _ref_id5 = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 6)
    _new_at_id = clamp_tilt_to_cone(_probe, _cone, _ref_id5)
    _old = _cone_world(_probe, _cone)
    check("an upright reference reproduces the pre-B7 world-vertical clamp exactly",
          _maxabs(_new_at_id, _old) == 0.0,
          f"max abs difference {float(_maxabs(_new_at_id, _old)):.3e} over "
          f"{int(_probe.shape[0])} orientations")

    # (2) A PART ALIGNED WITH A TILTED POCKET IS NOT TOUCHED.
    #
    # THE PROBE TILT IS BEYOND THE CONE ON PURPOSE (20 deg against 8.52), and
    # NOT the AutoDR maximum. At 8.00 deg an aligned part sits inside the
    # WORLD cone as well, so a check there passes under the pre-B7 behaviour
    # too and discriminates nothing at all. The first version of this check
    # probed at 8.00 deg and the counter-proof caught it: a mutation that put
    # the world vertical back left this check green.
    _pocket_probe = math.radians(20.0)
    _q_pocket20 = quat_mul(
        quat_from_axis_angle(torch.tensor([[0.0, _pocket_probe, 0.0]])),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    # The tool aligned with that pocket: the same tilt applied to a part
    # hanging down.
    _q_tool_on_axis = quat_mul(
        quat_from_axis_angle(torch.tensor([[0.0, _pocket_probe, 0.0]])), _q_down)
    check("a part aligned with a tilted pocket is not touched by the cone",
          _maxabs(clamp_tilt_to_cone(_q_tool_on_axis, _cone, _q_pocket20), _q_tool_on_axis) < 1e-12,
          f"that part reads {math.degrees(float(tool_axis_tilt(_q_tool_on_axis)[0])):.4f} deg "
          f"from world against a {math.degrees(_cone):.2f} deg cone -- a world-vertical "
          "clamp would move it")

    # (3) AND AT THE AutoDR MAXIMUM THE PRE-B7 FRAME LEFT 0.52 deg. The study
    # never reaches 20 deg; this is the number the run actually lives with,
    # stated as arithmetic rather than as a story.
    _q_tool_at_max = quat_mul(
        quat_from_axis_angle(torch.tensor([[0.0, math.radians(8.0), 0.0]])), _q_down)
    _left = _cone - float(tool_axis_tilt(_q_tool_at_max)[0])
    check("the pre-B7 frame left only the measured 0.52 deg of the cone",
          abs(math.degrees(_left) - 0.52) < 0.01,
          f"{math.degrees(_left):.4f} deg left of {math.degrees(_cone):.2f}")

    # (4) THE CONE IS STILL A CONE, measured in the new frame: a part leaning
    # 20 deg away from a tilted pocket comes back at exactly the cone angle
    # FROM THE POCKET, not from the world.
    _q_tool_off = quat_mul(quat_from_axis_angle(torch.tensor([[0.0, math.radians(20.0), 0.0]])),
                           _q_tool_on_axis)
    _q_off_cl = clamp_tilt_to_cone(_q_tool_off, _cone, _q_pocket20)
    _pocket_down = -_arr(axes_from_quat(_q_pocket20))[:, :, 2]
    _tool_cl = _arr(axes_from_quat(_q_off_cl))[:, :, 2]
    _rel = math.acos(max(-1.0, min(1.0, float((_tool_cl * _pocket_down).sum(axis=-1)[0]))))
    check("a lean beyond the cone is pulled back to the cone angle FROM THE POCKET",
          abs(_rel - _cone) < 1e-6,
          f"{math.degrees(_rel):.4f} deg from the pocket axis, "
          f"{math.degrees(float(tool_axis_tilt(_q_off_cl)[0])):.4f} deg from world")

    # (5) THE DEGENERATE FALLBACK IS THE POCKET'S OWN X AXIS. World x is only
    # perpendicular to `down` while the pocket stands upright; against a
    # tilted pocket it would tip the part out of its own plane.
    _q_anti = quat_mul(quat_from_axis_angle(torch.tensor([[0.0, _pocket_probe, 0.0]])), _q_id)
    _q_anti_cl = clamp_tilt_to_cone(_q_anti, _cone, _q_pocket20)
    _anti_axis = _arr(axes_from_quat(_q_anti_cl))[:, :, 2]
    _anti_rel = math.acos(max(-1.0, min(1.0, float((_anti_axis * _pocket_down).sum(axis=-1)[0]))))
    check("a part antiparallel to a TILTED pocket axis is pulled to the cone without NaN",
          bool(torch.all(torch.isfinite(_q_anti_cl))) and abs(_anti_rel - _cone) < 1e-6,
          f"{math.degrees(_anti_rel):.4f} deg from the pocket axis")

    # =====================================================================
    #  disk_offset (D-178 (1)): the lateral start offset on a disk
    # =====================================================================
    # A PROBE radius, not the task's ceiling: the module owns no numbers and
    # neither does this block.
    disk_offset = ns["disk_offset"]
    _R = 0.030
    _rim = disk_offset(torch.tensor([1.0]), torch.tensor([0.0]), _R)
    _ring = disk_offset(torch.tensor([1.0, 1.0, 1.0, 1.0]),
                        torch.tensor([0.1, 0.3, 0.6, 0.85]), _R)
    _ring_err = max(abs(float(v) - _R) for v in torch.linalg.norm(_ring, dim=-1))
    # u_r = 1 is the RIM, exactly: sqrt(1.0) is 1.0, so a boundary env nailed
    # to the upper edge of `lat_r` starts ON that edge, not within a rounding
    # error of it. At phi = 0 the whole radius lies on +x.
    check("disk_offset: u_r = 1 lands on the rim at r_max EXACTLY, one (x, y) row per env",
          tuple(_rim.shape) == (1, 2) and float(_rim[0, 0]) == _R and float(_rim[0, 1]) == 0.0
          and tuple(_ring.shape) == (4, 2) and _ring_err < 1e-15,
          f"rim {_arr(_rim).tolist()}, ring |r - R| max {_ring_err:.2e}")
    _axis = disk_offset(torch.tensor([0.0, 0.0]), torch.tensor([0.3, 0.9]), _R)
    check("disk_offset: u_r = 0 is the pocket axis, whatever the angle",
          all(float(v) == 0.0 for v in _arr(_axis).reshape(-1)),
          str(_arr(_axis).tolist()))
    # The angle column spans the WHOLE circle: a quarter turn is +y, a half
    # turn is -x. A map that drew only [0, pi) passes both checks above.
    _turns = disk_offset(torch.tensor([1.0, 1.0]), torch.tensor([0.25, 0.5]), _R)
    check("disk_offset: the angle is 2*pi*u_phi, counter-clockwise from +x",
          abs(float(_turns[0, 0])) < 1e-15 and abs(float(_turns[0, 1]) - _R) < 1e-15
          and abs(float(_turns[1, 0]) + _R) < 1e-15 and abs(float(_turns[1, 1])) < 1e-15,
          str(_arr(_turns).tolist()))
    # UNIFORM OVER THE AREA, deterministically (D-178, Verification status).
    # u_r = linspace(0, 1, n) with n EVEN, so no sample sits on u_r = 0.5. The
    # inner disk of radius R/sqrt(2) holds half the area, so exactly the draws
    # with u_r < 0.5 must land inside it: n/2 of them. `r = R*u_r` instead puts
    # a share of about 0.707 there.
    _n = 1000
    _u = torch.tensor([i / (_n - 1) for i in range(_n)])
    _spread = disk_offset(_u, torch.tensor([0.37] * _n), _R)
    _inner = sum(1 for v in torch.linalg.norm(_spread, dim=-1)
                 if float(v) < _R / math.sqrt(2.0))
    _below_half = sum(1 for v in _u if float(v) < 0.5)
    check("disk_offset is uniform over the AREA: exactly the u_r < 0.5 draws land inside R/sqrt(2)",
          _inner == _below_half == _n // 2,
          f"inside R/sqrt(2): {_inner}, u_r < 0.5: {_below_half}, n/2 = {_n // 2}")

    # =====================================================================
    #  resolve_control_mode (insertion_env_cfg.py)
    # =====================================================================
    resolve_mode = cfg_ns["resolve_control_mode"]
    control_modes = cfg_ns["CONTROL_MODES"]

    class _Sim:
        pass

    def _mode_cfg(mode):
        c = _Cfg()
        c.control_mode = mode
        c.decimation = 2
        c.osc_decimation = 8
        c.episode_steps = 256
        c.episode_length_s = 256 / 60
        c.sim = _Sim()
        c.sim.dt = 1 / 120
        c.sim.render_interval = 2
        return c

    check("the control modes are exactly osc and joint_pd",
          tuple(control_modes) == ("osc", "joint_pd"), str(control_modes))
    _c = _mode_cfg("osc")
    _got = resolve_mode(_c)
    check("osc mode runs at osc_decimation and renders at that interval",
          _got == "osc" and _c.decimation == 8 and _c.sim.render_interval == 8,
          f"decimation {_c.decimation}, render_interval {_c.sim.render_interval}")
    check("the OSC episode keeps its 256-step count (seconds follow the rate)",
          math.ceil(_c.episode_length_s / (_c.sim.dt * _c.decimation)) == 256
          and abs(_c.episode_length_s - 256 * 8 / 120) < 1e-12,
          f"episode_length_s {_c.episode_length_s:.6f}")
    _c = _mode_cfg("joint_pd")
    _got = resolve_mode(_c)
    check("joint_pd mode leaves the D-024 rate fields untouched",
          _got == "joint_pd" and _c.decimation == 2 and _c.sim.render_interval == 2
          and abs(_c.episode_length_s - 256 / 60) < 1e-12)
    try:
        resolve_mode(_mode_cfg("pose_pd"))
        _refused = False
    except Incomplete:
        _refused = True
    check("an unknown control mode is refused", _refused)

    # =====================================================================
    #  resolve_start_lateral_offset (insertion_env_cfg.py, D-178 (3))
    # =====================================================================
    # RUN, not read: this is the one place the field's shape is read. Since
    # D-178 it is ONE number, the disk radius -- the cfg default 0.0, the
    # hydra form `env.start_lateral_offset=<radius>`, a Python number. The
    # (x, y) pair of D-176 is refused BY NAME, never read as one of its entries.
    resolve_lat = cfg_ns["resolve_start_lateral_offset"]

    def _lat_refusal(value) -> str:
        """The refusal text, or "" when the value was accepted."""
        try:
            resolve_lat(value)
        except ValueError as exc:
            return str(exc) or "refused without a message"
        return ""

    check("a number passes through as the radius in metres",
          resolve_lat(0.006) == 0.006 and isinstance(resolve_lat(0.006), float)
          and resolve_lat(0) == 0.0 and isinstance(resolve_lat(0), float)
          and resolve_lat(0.0) == 0.0,
          f"{resolve_lat(0.006)!r}, {resolve_lat(0)!r}")
    # Refused, not repaired. An old pair command read as a radius would run a
    # disk nobody asked for, and the pointer is how the reader finds out why.
    check("a pair or a hydra list is refused, with a pointer to D-178",
          all("D-178" in _lat_refusal(v)
              for v in ([0.002, 0.001], (0.002, 0.001), [0.006])),
          _lat_refusal([0.002, 0.001])[:70])
    check("a negative radius is refused",
          bool(_lat_refusal(-0.002)))
    check("a string and a bool are refused, never coerced",
          bool(_lat_refusal("0.002")) and bool(_lat_refusal(True)))

    return out


# ===========================================================================
#  Counter-proof table (D-080): one mutation, and EXACTLY the checks it must
#  break. Each is a plausible edit, not a syntax error.
# ===========================================================================

MUTATIONS: tuple[tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]], ...] = (
    # -- disk_offset (D-178 (1)) ---------------------------------------------
    (
        # The square root is dropped: r = R*u_r. Rim and axis stay exact and
        # every angle stays right -- only the AREA distribution is wrong, with
        # half of all draws inside R/2, a quarter of the disk.
        "the-disk-radius-loses-its-sqrt",
        (("    r = r_max * torch.sqrt(u_r)\n", "    r = r_max * u_r\n"),),
        ("disk_offset is uniform over the AREA: exactly the u_r < 0.5 draws land inside R/sqrt(2)",),
    ),
    (
        # The angle spans half a circle: every offset lands in the upper half
        # plane, and no radius check can see it.
        "the-disk-angle-covers-half-a-circle",
        (("    phi = 2.0 * math.pi * u_phi\n", "    phi = math.pi * u_phi\n"),),
        ("disk_offset: the angle is 2*pi*u_phi, counter-clockwise from +x",),
    ),
    (
        # cos and sin swap axes: the angle is measured from +y. Every radius
        # stays right; the offset lands at the mirrored angle.
        "the-disk-axes-are-swapped",
        (("    return torch.stack((r * torch.cos(phi), r * torch.sin(phi)), dim=-1)\n",
          "    return torch.stack((r * torch.sin(phi), r * torch.cos(phi)), dim=-1)\n"),),
        (
            "disk_offset: u_r = 1 lands on the rim at r_max EXACTLY, one (x, y) row per env",
            "disk_offset: the angle is 2*pi*u_phi, counter-clockwise from +x",
        ),
    ),
    (
        # The symmetric +-range habit of the D-176 square band carried over:
        # r = R*(2*sqrt(u_r) - 1). The rim still reads R, but u_r = 0 lands on
        # the far rim instead of the axis and the area share goes wrong.
        "the-disk-radius-keeps-the-old-symmetric-map",
        (("    r = r_max * torch.sqrt(u_r)\n",
          "    r = r_max * (2.0 * torch.sqrt(u_r) - 1.0)\n"),),
        (
            "disk_offset: u_r = 0 is the pocket axis, whatever the angle",
            "disk_offset is uniform over the AREA: exactly the u_r < 0.5 draws land inside R/sqrt(2)",
        ),
    ),
    # -- Phase 5 step B7: the cone's reference frame -------------------------
    (
        # The cone is measured from the world vertical again. The signature
        # keeps its third argument and every caller still passes a pocket
        # quaternion, so nothing outside this line looks different.
        "the-cone-ignores-its-reference-frame",
        (("    down = -ref[:, :, 2]",
          "    down = torch.zeros_like(tool) - axes_from_quat("
          "torch.zeros_like(ref_quat) + quat_from_axis_angle("
          "torch.zeros_like(tool)))[:, :, 2]"),),
        (
            "a part aligned with a tilted pocket is not touched by the cone",
            "a lean beyond the cone is pulled back to the cone angle FROM THE POCKET",
            "a part antiparallel to a TILTED pocket axis is pulled to the cone without NaN",
        ),
    ),
    (
        # The degenerate fallback is world x again. It is perpendicular to
        # `down` only while the pocket stands upright; against a tilted pocket
        # it tips the part out of its own plane and the pull-back lands
        # somewhere else.
        "the-degenerate-fallback-is-world-x-again",
        (("    n_safe = torch.where(degenerate.reshape(-1, 1), ref[:, :, 0], n)",
          "    n_safe = torch.where(degenerate.reshape(-1, 1), "
          "torch.zeros_like(n) + torch.tensor([[1.0, 0.0, 0.0]]), n)"),),
        ("a part antiparallel to a TILTED pocket axis is pulled to the cone without NaN",),
    ),
    # -- SAPU: the clamp that replaces the second Warp kernel ----------------
    # Clamping the wrong end is the whole failure mode: it keeps the
    # clearances and throws the penetrations away, which reads as a part that
    # never tunnels.
    (
        "interpen-clamps-the-wrong-end",
        (("return torch.clamp(signed_dist, max=0.0)",
          "return torch.clamp(signed_dist, min=0.0)"),),
        (
            "a penetration survives the clamp unchanged",
            "a clearance becomes zero, not a penetration",
            "a miss (+max_dist) clamps to the source's seeded zero",
        ),
    ),
    # -- seated_goal_pose (M2.4b step 2) -------------------------------------
    # Depth applied along +z instead of -z: the goal floats one seat depth
    # ABOVE the opening, which the SDF reward would then chase.
    (
        "goal-depth-applied-upward",
        (("goal_tip = entrance_pos - seat_depth * pocket_rot[:, :, 2]",
          "goal_tip = entrance_pos + seat_depth * pocket_rot[:, :, 2]"),),
        (
            "an untilted goal pose subtracts seat depth and the FLIPPED tip offset",
            "the goal pose reports exactly seat depth through part_tip_pose",
            "a yawed and tilted pocket still seats the goal exactly",
        ),
    ),
    # The tip offset subtracted in the ENV frame instead of rotated by the
    # goal orientation. With the seat flip in the contract this is loud even
    # in an untilted pocket -- the goal orientation is never the identity.
    (
        "goal-tip-offset-not-rotated",
        (("goal_pos = goal_tip - torch.matmul(axes_from_quat(goal_quat), tip_offset_local)",
          "goal_pos = goal_tip - tip_offset_local"),),
        (
            "an untilted goal pose subtracts seat depth and the FLIPPED tip offset",
            "with the real offset sign the goal flange sits ABOVE the entrance",
            "the goal pose reports exactly seat depth through part_tip_pose",
            "the goal tip offset rotates with the pocket",
            "a yawed and tilted pocket still seats the goal exactly",
        ),
    ),
    # THE RT-102 DEFECT, replayed: the seat rotation dropped, the fixture
    # quaternion alone as the goal orientation. The round trip is BLIND to it
    # (pos and quat stay mutually consistent), which is exactly why the two
    # VALUE checks exist.
    (
        "goal-seat-rotation-dropped",
        (("goal_quat = quat_mul(fixture_quat, seat_quat_local)",
          "goal_quat = fixture_quat"),),
        (
            "an untilted goal pose subtracts seat depth and the FLIPPED tip offset",
            "the goal orientation composes the seated tool rotation",
            "with the real offset sign the goal flange sits ABOVE the entrance",
        ),
    ),
    # -- pose algebra -------------------------------------------------------
    # The first one is not hypothetical: the proxy shipped exactly this damage
    # on 2026-07-26 and only an identity test found it.
    (
        "axes-stacked-as-rows",
        (("return torch.stack([col0, col1, col2], dim=-1)",
          "return torch.stack([col0, col1, col2], dim=1)"),),
        (
            "the axes are COLUMNS, not rows",
            "90 deg about y sends body z to +x",
            "quat_mul composes the same way the matrices do",
            "the relative offset is expressed in frame A",
            "the relative rotation matches R(a)^T R(b)",
            "a 90 deg wrist turn still tares to zero (the rotation is applied)",
            "a 90 deg wrist turn still tares the weight moment to zero (the rotation is applied)",
            "quat_rotate sends x to y under 90 deg about z",
            # RT-171: the tilted-pocket case composes two rotations, so the
            # row/column swap reaches it too (2026-09-06).
            "a part tilted WITH its tilted pocket reads cos theta = +1",
            "a part aligned with a tilted pocket is not touched by the cone",
            "a part antiparallel to a TILTED pocket axis is pulled to the cone without NaN",
        ),
    ),
    (
        "quat-mul-flips-a-cross-term",
        (("aw * bx + ax * bw + ay * bz - az * by",
          "aw * bx + ax * bw + ay * bz + az * by"),),
        (
            "quat_mul composes the same way the matrices do",
            "the relative rotation matches R(a)^T R(b)",
            "a part antiparallel to a TILTED pocket axis is pulled to the cone without NaN",
        ),
    ),
    (
        "tip-offset-added-in-the-env-frame",
        (("    tip_local = body_pos + torch.matmul(\n"
          "        part_axes, tip_offset_local.reshape(-1, 3, 1)\n"
          "    ).reshape(-1, 3) - env_origins",
          "    tip_local = body_pos + tip_offset_local - env_origins"),),
        (
            "the tip offset rotates with the part",
            # The old-form comparison rotates the offset the way the line did
            # before D-183, so an unrotated new line disagrees with it. The
            # (3,)/(N, 3) comparison does NOT flip: both call forms lose the
            # rotation together and still agree with each other.
            "the (3,) path is bit-identical to the pre-D-183 bare matmul",
            # The goal round trip runs THROUGH part_tip_pose, so damage to the
            # forward tip line surfaces there too -- shared home, shared flip.
            # Since the seat flip the goal orientation is never the identity,
            # so the depth check flips as well.
            "the goal pose reports exactly seat depth through part_tip_pose",
            "the goal tip offset rotates with the pocket",
            "a yawed and tilted pocket still seats the goal exactly",
        ),
    ),
    (
        # D-183: the reshape pair goes and the bare (N, 3, 3) @ (N, 3) form
        # comes back. The shared (3,) offset every other caller passes still
        # works -- which is exactly why reading the existing checks cannot
        # catch this -- but a per-env offset now raises.
        "the-per-env-tip-offset-is-read-as-a-matrix",
        (("torch.matmul(\n"
          "        part_axes, tip_offset_local.reshape(-1, 3, 1)\n"
          "    ).reshape(-1, 3)",
          "torch.matmul(part_axes, tip_offset_local)"),),
        (
            "part_tip_pose takes ONE tip offset per env, not only a shared (3,)",
            "a per-env tip offset moves each env by ITS OWN row",
            "the tip offset does not move the part's x-axis (yaw_cos_sin is untouched)",
            # The (N, 3) call raises, so the two call forms cannot be compared
            # at all -- which counts as a flip. The OLD-form comparison does
            # not flip, and that is the point: this mutation restores exactly
            # the pre-D-183 expression, so the (3,) path really is unchanged.
            "a (3,) offset and its (N, 3) broadcast give bit-identical results",
        ),
    ),
    (
        # The tip point stops being made env-local. Plausible in an Isaac Lab
        # env, where world and env frames differ only by the origin and a
        # single-env probe cannot tell them apart. Both call forms lose the
        # term together, so the (3,)/(N, 3) comparison is blind to it; the
        # comparison against the pre-D-183 expression is not.
        "the-env-origin-is-not-subtracted-from-the-tip",
        ((").reshape(-1, 3) - env_origins\n",
          ").reshape(-1, 3)\n"),),
        (
            "the env origin is subtracted",
            "the (3,) path is bit-identical to the pre-D-183 bare matmul",
        ),
    ),
    # rotate_into_frame is shared by part_tip_pose and relative_pose, so one
    # damage to it shows up in both -- which is the point of it having one home.
    (
        "projection-uses-r-not-r-transpose",
        (("torch.sum(rot * v.reshape(-1, 3, 1), dim=1)",
          "torch.sum(rot * v.reshape(-1, 1, 3), dim=2)"),),
        (
            "the offset is projected into the pocket frame",
            "the relative offset is expressed in frame A",
            "a yawed and tilted pocket still seats the goal exactly",
            "a 90 deg wrist turn still tares to zero (the rotation is applied)",
            "a 90 deg wrist turn still tares the weight moment to zero (the rotation is applied)",
        ),
    ),
    (
        "quat-conjugate-flips-w-instead",
        (("return torch.stack((q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]), dim=-1)",
          "return torch.stack((-q[:, 0], q[:, 1], q[:, 2], q[:, 3]), dim=-1)"),),
        (
            "the conjugate keeps w and flips the vector part",
            "a pose relative to itself is the identity rotation",
            "relative to the identity frame is the pose itself",
        ),
    ),
    (
        "relative-rotation-composed-the-wrong-way",
        (("rel_quat = quat_mul(quat_conjugate(quat_a), quat_b)",
          "rel_quat = quat_mul(quat_b, quat_conjugate(quat_a))"),),
        ("the relative rotation matches R(a)^T R(b)",),
    ),
    (
        "relative-offset-taken-from-the-wrong-frame",
        (("rel_pos = rotate_into_frame(axes_from_quat(quat_a), pos_b - pos_a)",
          "rel_pos = rotate_into_frame(axes_from_quat(quat_b), pos_b - pos_a)"),),
        (
            "relative to the identity frame is the pose itself",
            "the relative offset is expressed in frame A",
        ),
    ),
    (
        "depth-sign-flipped",
        (("depth = -tip_rel[:, 2]", "depth = tip_rel[:, 2]"),),
        (
            "depth is POSITIVE into the pocket",
            "the tip offset moves the measured point",
            "the tip offset rotates with the part",
            "the goal pose reports exactly seat depth through part_tip_pose",
        ),
    ),
    (
        "part-axes-left-in-the-world-frame",
        (("axes_in_pocket = torch.matmul(pocket_rot.transpose(1, 2), part_axes)",
          "axes_in_pocket = part_axes"),),
        ("the part axis is projected into the pocket frame",),
    ),
    (
        "kernel-b-typo",
        (("squash(sdf_dist, a_mid, b_mid)", "squash(sdf_dist, a_mid, b_fine)"),),
        ("three kernels peak at exactly 1.0", "a clean seat pays the full return"),
    ),
    (
        "sapu-neutered",
        (("scale = 1.0 - torch.tanh(interpen_dist / interpen_thresh)",
          "scale = torch.ones_like(interpen_dist)"),),
        (
            "SAPU scale just under the threshold is 1 - tanh(~1)",
            "SAPU falls monotonically below the threshold",
            "partial interpenetration scales the return down",
        ),
    ),
    (
        "sapu-discard-dropped",
        (("torch.zeros_like(scale), scale", "scale, scale"),),
        (
            "SAPU discards above the threshold",
            "tunneling past the SAPU threshold pays nothing",
            "the batched tunnelling env is the zero one",
            # D-165: the progress row sits inside the same scale.
            "the depth progress is discarded by SAPU like the task return",
            # Not surprises, consequences: both checks expect the discarded
            # return to be exactly zero.
            "the abort payment survives a discarded task return",
            "ONE point past the threshold discards the whole env",
            # Since 2026-09-01 the payout is computed from the SCALED step
            # value, so a SAPU that no longer discards also buys a tunnelled
            # "seat" a full episode's income. That is a consequence, not a
            # surprise, and it is the loudest one in the table.
            "a tunnelled seat is paid out nothing",
        ),
    ),
    (
        "abort-payment-positive",
        (("abort_pay = torch.where(force_abort & (~success), payment, zero)",
          "abort_pay = torch.where(force_abort & (~success), -payment, zero)"),),
        (
            "aborting costs the fixed payment",
            "aborting is strictly worse than not aborting",
            "the abort payment survives a discarded task return",
        ),
    ),
    (
        "abort-payment-scaled-by-sapu",
        (("        abort_pay,\n        tilt_pay,\n    ], dim=0)",
          "        abort_pay * scale,\n        tilt_pay,\n    ], dim=0)"),),
        ("the abort payment survives a discarded task return",),
    ),
    # -- the per-term stack (2026-09-02) -------------------------------------
    # The table drifts from the stack: a row without a name, or a name
    # without a row, and the env would log a term under the wrong label.
    (
        "reward-terms-table-loses-a-name",
        # REPOINTED 2026-09-06 (RT-171): the LAST name goes, so every other
        # row keeps its index and only the shape check can see the loss.
        (('    "abort",         # abort_payment on the force-abort step\n'
          '    "tilt_shaping",  # gamma * Phi(s\') - Phi(s), Phi = w_tilt * sech(a * theta) vs the POCKET axis; Phi(s\') = 0 on the success step (RT-171)\n)',
          '    "abort",         # abort_payment on the force-abort step\n)'),),
        ("REWARD_TERMS names every row of the stack, once",
         "tilt_shaping is the last row of the stack and reads gamma * Phi(s') - Phi(s)"),
    ),
    # Two rows swapped: the sum is unchanged, so only the labelling check
    # can catch it -- which is exactly why that check has an env that is
    # engaged WITHOUT success.
    (
        "reward-term-rows-swapped",
        (("        engaged_row,\n        success_row,",
          "        success_row,\n        engaged_row,"),),
        ("the term rows carry the terms they are named after",),
    ),
    # -- D-165: the depth-progress term (2026-09-02) -------------------------
    # The term dropped: the metric still climbs, the row still exists, and
    # nothing is paid. Flips the per-metre check, the lump check (delta 0)
    # and the labelling (env 2's progress row goes silent).
    (
        "depth-progress-dropped",
        (("    progress_pay = w_progress * depth_progress * scale",
          "    progress_pay = zero"),),
        (
            "the depth progress pays w_progress per metre of new max depth",
            "the depth progress is NOT multiplied into the success payout",
            "the term rows carry the terms they are named after",
        ),
    ),
    # Paid unscaled: a tunnelled part reads as depth and gets paid for it.
    (
        "depth-progress-skips-sapu",
        (("    progress_pay = w_progress * depth_progress * scale",
          "    progress_pay = w_progress * depth_progress"),),
        ("the depth progress is discarded by SAPU like the task return",),
    ),
    # Folded into the lump: one millimetre on the success step is bought out
    # for every remaining step, ~200x what it is worth.
    (
        "depth-progress-inside-the-lump",
        (("    lump = torch.where(success, step_task * steps_remaining, zero)",
          "    lump = torch.where(success, (step_task + progress_pay) * steps_remaining, zero)"),),
        ("the depth progress is NOT multiplied into the success payout",),
    ),
    # -- the 2026-09-01 terms (inbox entry "Reward-Ueberarbeitung") ----------
    # Each mutation is the plausible edit, not a syntax error: a term dropped,
    # a sign flipped, a multiplier taken from the wrong quantity.
    (
        # The time penalty silently gone. Every step then pays what it paid
        # before the revision, which is exactly the state the two failed PPO
        # runs trained in.
        "time-penalty-dropped",
        (("    time_pay = torch.full_like(step_task, time_penalty_per_step)",
          "    time_pay = torch.zeros_like(step_task)"),),
        (
            "the time penalty is subtracted from every step",
            "the time penalty is paid on the force-abort step as well",
            "the time penalty is paid on the success-termination step as well",
        ),
    ),
    (
        # The action-rate term PAYS for jerk instead of pricing it. The config
        # guard refuses a negative scale; this is the same damage one level
        # down, where no guard can see it.
        "action-rate-penalty-pays-instead-of-charging",
        (("    rate_pay = -action_rate_scale * action_rate",
          "    rate_pay = action_rate_scale * action_rate"),),
        (
            "the action-rate penalty is scale * ||a_t - a_(t-1)||, subtracted",
            "the action-rate penalty grows with the change",
        ),
    ),
    (
        # The payout dropped: success terminates but is paid only its own
        # step. That is the WORST of both schemes -- the episode ends early
        # AND forfeits the remaining income, so seating becomes strictly worse
        # than parking, which is the exploit the payout exists to close.
        "success-payout-dropped",
        (("    lump = torch.where(success, step_task * steps_remaining, zero)",
          "    lump = zero"),),
        (
            "the success payout is the step value times the steps left",
            "holding to the timeout and being paid out are the same undiscounted return",
            # The two tie-break checks do NOT flip here, and that is worth
            # saying: with no lump at all the two sides they compare are the
            # same number again. The tie-break has its own mutation below.
        ),
    ),
    (
        # The payout paid on the UNSCALED task return: a tunnelled seat then
        # buys itself a full episode of income, which is the one exploit SAPU
        # exists for.
        "success-payout-skips-sapu",
        (("    lump = torch.where(success, step_task * steps_remaining, zero)",
          "    lump = torch.where(success, (kernels + engaged_pay + success_pay) * steps_remaining, zero)"),),
        (
            "a tunnelled seat is paid out nothing",
            # The batched env 3 is the tunnelling one and it carries a seat
            # plus steps remaining, so the same damage shows up there too.
            "the batched tunnelling env is the zero one",
        ),
    ),
    (
        # The tie-break inverted: the force abort charged on a step that seated
        # the part. Invisible in every other case, which is why it has its own
        # mutation.
        "tie-break-gives-the-abort-priority",
        (("    abort_pay = torch.where(force_abort & (~success), payment, zero)",
          "    abort_pay = torch.where(force_abort, payment, zero)"),),
        (
            "success wins the tie against a force abort on the same step",
            "the abort payment is NOT charged on a success step",
        ),
    ),
    (
        # Success stops terminating -- D-113 (1) put back by a "helpful" edit.
        # The reward would keep paying the lump every step from then on.
        "success-no-longer-terminates",
        (("    terminated = success_now", "    terminated = success_now & (force_norm < 0.0)"),),
        (
            "a seated, force-free env DOES terminate (the success exit)",
            "a seated env over F_max terminates and DOES report the abort flag",
            "a success on the timeout step is terminated, not truncated",
            # NOT "a seated env over F_max is NOT also truncated": the
            # exclusivity guard reads success_now directly, so it keeps that
            # env out of `truncated` even with `terminated` neutered. The env
            # would then end nowhere at all, which is what the two checks
            # above catch.
        ),
    ),
    (
        # D-164 put back the way it was: the abort returns to `terminated` and
        # rsl_rl forfeits the remaining return again -- the RT-131 defect.
        "abort-terminates-again",
        (("    terminated = success_now", "    terminated = success_now | force_abort"),
         ("    truncated = (timeout | force_abort) & (~success_now)",
          "    truncated = timeout & (~success_now)")),
        (
            "force exactly at F_max aborts, as TRUNCATED not terminated",
        ),
    ),
    (
        # The exclusivity guard dropped: a success landing on the timeout step,
        # or over F_max, would be terminated AND truncated at once, and the
        # wrapper would hand rsl_rl a time_out on a success step -- a bootstrap
        # stacked on top of the success lump.
        "termination-channels-not-exclusive",
        (("    truncated = (timeout | force_abort) & (~success_now)",
          "    truncated = timeout | force_abort"),),
        (
            "a seated env over F_max is NOT also truncated",
            "a success on the timeout step is terminated, not truncated",
        ),
    ),
    (
        # The abort flag reported as `terminated`. Before D-164 that was the
        # UNION and every success would have been counted as a force abort in
        # the D-113 (9) rate; since D-164 `terminated` IS the success, so the
        # rate would count successes and miss every real abort. Both halves
        # are wrong and both are caught.
        "abort-flag-reports-the-union",
        (("    return terminated, truncated, force_abort",
          "    return terminated, truncated, terminated"),),
        (
            "a success termination is not reported as a force abort",
            "force exactly at F_max aborts, as TRUNCATED not terminated",
        ),
    ),
    (
        "obs-blocks-swapped",
        (("(joint_pos, joint_vel, tip_rel, ee_quat, yaw_cs, pocket_quat, force), dim=-1",
          "(joint_pos, joint_vel, tip_rel, ee_quat, pocket_quat, yaw_cs, force), dim=-1"),),
        ("channels 19:21 carry yaw_cos_sin", "channels 21:25 carry pocket_quat"),
    ),
    (
        "yaw-folded-back-to-c4",
        (("phi = torch.atan2(x_axis[:, 1], x_axis[:, 0])",
          "phi = 4.0 * torch.atan2(x_axis[:, 1], x_axis[:, 0])"),),
        (
            "yaw encoding is (cos phi, sin phi) of the true angle",
            "a 90 deg turn changes the encoding (no C4 folding)",
        ),
    ),
    (
        "canon-flips-the-wrong-sign",
        (("flip = q[:, 0:1] < 0.0", "flip = q[:, 0:1] > 0.0"),),
        (
            "canonicalize forces w >= 0",
            "canonicalize leaves an already-positive quaternion untouched",
            "canonicalize negates ALL four components, not just w",
        ),
    ),
    (
        "ema-alpha-on-the-wrong-side",
        (("return alpha * new + (1.0 - alpha) * prev",
          "return alpha * prev + (1.0 - alpha) * new"),),
        (
            "alpha = 1 passes the new sample through",
            "alpha = 0 freezes at the previous sample",
            "alpha = 0.25 weights the NEW sample by 0.25",
        ),
    ),
    (
        # The per-step force noise is pointed at the JOINT POSITIONS. Every
        # shape stays right and the model still runs; the force channel is
        # simply clean and six joint angles are noisy instead. Nothing in a
        # training curve would say so.
        "the-force-mask-lands-on-the-joint-positions",
        (('    lo, hi = OBS_SLICES["force"]', '    lo, hi = OBS_SLICES["joint_pos"]'),),
        ("the per-step noise lands on force and nowhere else",
         "in wrench mode the per-step noise is 3.5 on force, 0.2 on torque, nowhere else"),
    ),
    (
        # The mirror image: the per-EPISODE pocket bias lands on the force
        # block. The pose channels are then clean and the force carries a
        # constant offset for a whole episode -- a bias on the one channel
        # whose purpose is to change WITHIN the episode.
        "the-pocket-bias-lands-on-the-force-block",
        (('    lo, hi = OBS_SLICES["tip_rel"]', '    lo, hi = OBS_SLICES["force"]'),),
        ("the per-episode bias lands on tip_rel and nowhere else",
         "neither mask touches the other's block",
         "the torque sigma never reaches the bias mask"),
    ),
    (
        # THE SWAP ITSELF, written the way it would look inside the function:
        # the pocket sigma on the force block, the force sigma on the pocket
        # block. Both masks keep their shape AND their non-zero count, so the
        # shape check, the leak check and the switched-off check all stay
        # green -- only the two checks that pin a sigma BY VALUE can see it.
        # This is the 3.5 m pose bias the critic finding named, produced
        # without touching a single caller.
        "the-two-sigmas-swap-blocks",
        (("    step_std[lo:hi] = force_std_n\n",
          "    step_std[lo:hi] = pocket_pos_std_m\n"),
         ("    bias_std[lo:hi] = pocket_pos_std_m\n",
          "    bias_std[lo:hi] = force_std_n\n")),
        ("the per-episode bias lands on tip_rel and nowhere else",
         "the per-step noise lands on force and nowhere else",
         "in wrench mode the per-step noise is 3.5 on force, 0.2 on torque, nowhere else",
         "the torque sigma never reaches the bias mask"),
    ),
    (
        # The ``*`` goes and the two sigmas may be passed by position again.
        # The masks themselves are untouched -- what comes back is the ABILITY
        # to write the swap above at a call site, silently, which is the hole
        # L-08 describes. Nothing but the guard check can see this.
        "the-sigmas-can-be-passed-by-position-again",
        (("def noise_masks(\n    *,\n    pocket_pos_std_m: float,\n",
          "def noise_masks(\n    pocket_pos_std_m: float,\n"),),
        ("noise_masks refuses positional sigmas, so the two cannot swap silently",),
    ),
    (
        # The tare's own mutations (D-080). The first is the defect the whole
        # rotation exists to prevent: subtract the world weight straight from a
        # body-frame reading. It is INVISIBLE while the wrist stands upright,
        # which is why the turned case is in the check list at all.
        "gravity-tare-not-rotated",
        (("return force_raw - rotate_into_frame(axes_from_quat(parent_quat_w), weight)",
          "return force_raw - weight"),),
        ("a 90 deg wrist turn still tares to zero (the rotation is applied)",),
    ),
    (
        "gravity-tare-switch-ignored",
        (("    if not enabled:\n        return force_raw",
          "    if False:\n        return force_raw"),),
        ("the tare is a no-op when it is switched off",),
    ),
    # -- the torque tare and the wrench layout (D-188) ----------------------
    (
        # The cross product reversed: F x r instead of r x F. The magnitude is
        # right and the frame is right; only the SIGN of the weight moment is
        # wrong, so the tare ADDS the moment instead of removing it.
        "gravity-tare-torque-cross-reversed",
        (("    moment_w = torch.cross(lever_w, hold, dim=-1)\n",
          "    moment_w = torch.cross(hold, lever_w, dim=-1)\n"),),
        (
            "the pure weight moment r x F_hold, upright, tares to zero",
            "a 0.5 N m contact moment on top of the weight moment survives the torque tare",
            "a 90 deg wrist turn still tares the weight moment to zero (the rotation is applied)",
        ),
    ),
    (
        # The force tare's own defect, on the torque: subtract the world
        # moment straight from the body-frame reading. Invisible upright.
        "gravity-tare-torque-not-rotated",
        (("    return torque_raw - rotate_into_frame(axes_from_quat(parent_quat_w), moment_w)",
          "    return torque_raw - moment_w"),),
        ("a 90 deg wrist turn still tares the weight moment to zero (the rotation is applied)",),
    ),
    (
        "gravity-tare-torque-switch-ignored",
        (("    if not enabled:\n        return torque_raw",
          "    if False:\n        return torque_raw"),),
        ("the torque tare is a no-op when it is switched off",),
    ),
    (
        # The torque block is placed BEFORE the force block instead of after
        # it: 25:28 torque, 28:31 force. Every force-mode reader that says
        # OBS_SLICES["force"] would then read torque in wrench mode.
        "torque-slice-not-appended-at-the-end",
        (('        out["torque"] = (OBS_DIM, OBS_DIM + TORQUE_WIDTH)\n',
          '        out["torque"] = (OBS_DIM - TORQUE_WIDTH, OBS_DIM)\n'),),
        (
            "obs mode 'wrench' is 31 wide",
            "the torque slice is appended DIRECTLY after force, 28:31",
            "the wrench table tiles 0..31 with no gap or overlap",
            "obs_version names mode and width",
            "wrench-mode masks are 31 wide",
            "in wrench mode the per-step noise is 3.5 on force, 0.2 on torque, nowhere else",
            "wrench mode with all three sigmas at zero is all-zero, 31 wide",
        ),
    ),
    (
        # The torque block goes in FRONT of the 28: a checkpoint trained on
        # this would read joint positions where it expects torques.
        "torque-block-prepended-in-the-cat",
        (("        (assemble_observation(joint_pos, joint_vel, tip_rel, ee_quat, yaw_cs, pocket_quat, force),\n"
          "         torque),\n",
          "        (torque,\n"
          "         assemble_observation(joint_pos, joint_vel, tip_rel, ee_quat, yaw_cs, pocket_quat, force)),\n"),),
        (
            "the wrench layout's first 28 channels ARE the force layout",
            "channels 28:31 carry torque",
        ),
    ),
    (
        # The torque sigma is pointed at the FORCE block: 25:28 reads 0.2
        # instead of 3.5 and the torque channels are clean.
        "the-torque-sigma-lands-on-the-force-block",
        (('        lo, hi = slices["torque"]\n',
          '        lo, hi = slices["force"]\n'),),
        ("in wrench mode the per-step noise is 3.5 on force, 0.2 on torque, nowhere else",),
    ),
    (
        # The refusal goes: a torque sigma in force mode is silently dropped,
        # and the run's log says "torque noise 0.2" while the policy sees none.
        "the-torque-sigma-is-ignored-in-force-mode",
        (('    if "torque" not in slices and torque_std_nm != 0.0:\n',
          '    if False:\n'),),
        ("noise_masks refuses a torque sigma in force mode (it would land nowhere)",),
    ),
    (
        "force-abort-strictly-greater",
        (("force_abort = force_norm >= f_max", "force_abort = force_norm > f_max"),),
        (
            "force exactly at F_max aborts, as TRUNCATED not terminated",
            # A force landing exactly on the limit no longer raises the abort
            # flag, so the seated-and-over-limit case loses it too.
            "a seated env over F_max terminates and DOES report the abort flag",
        ),
    ),
    (
        "timeout-off-by-one",
        (("timeout = episode_length_buf >= max_episode_length - 1",
          "timeout = episode_length_buf >= max_episode_length"),),
        ("the last step sets truncated, not terminated",),
    ),
    (
        "success-region-drops-the-interpenetration-filter",
        (("clean = interpen_dist < interpen_thresh", "clean = interpen_dist >= 0.0"),),
        ("a tunnelled seat is NOT a success",),
    ),
    (
        "success-region-loses-its-upper-edge",
        (("not_too_deep = depth <= depth_max", "not_too_deep = depth >= 0.0"),),
        ("too deep is not the success region -- it is a band, not a threshold",),
    ),
    # -- the D-157 lateral gate ---------------------------------------------
    # THE RT-107 DEFECT ITSELF: without the gate the two paying predicates
    # read a pure axis projection, and a descent 109 mm beside the fixture
    # scores a seat. This is the state the code was in on 2026-08-30.
    (
        "success-region-drops-the-pocket-gate",
        (("return deep_enough & not_too_deep & clean & in_pocket",
          "return deep_enough & not_too_deep & clean"),),
        (
            "in the band but OUTSIDE the pocket is NOT a success (D-157)",
            "RT-107 replay: seat depth 109 mm BESIDE the pocket is no success",
        ),
    ),
    (
        "engaged-drops-the-pocket-gate",
        (("return (depth >= engaged_depth) & in_pocket", "return depth >= engaged_depth"),),
        (
            "engaged does NOT fire outside the pocket (D-157)",
            "RT-107 replay: seat depth 109 mm BESIDE the pocket is not engaged",
        ),
    ),
    # The gate widened to include the wall plane. The metric it was lifted out
    # of compared strictly; a <= here would let the two predicates and the
    # depth metric disagree on the boundary env.
    (
        "pocket-gate-includes-the-wall",
        (("return (tip_rel[:, 0].abs() < wall_x) & (tip_rel[:, 1].abs() < wall_y)",
          "return (tip_rel[:, 0].abs() <= wall_x) & (tip_rel[:, 1].abs() <= wall_y)"),),
        ("the wall itself is OUTSIDE -- the test is strict, as the metric's was",),
    ),
    # The long axis read off the DEPTH column instead of y -- the one slip
    # that would leave the gate in place and still pass the RT-107 hack pose,
    # because the hack's lateral offset lives in exactly that column.
    (
        "pocket-gate-reads-depth-as-the-long-axis",
        (("(tip_rel[:, 1].abs() < wall_y)", "(tip_rel[:, 2].abs() < wall_y)"),),
        (
            "a tip past the long wall is outside",
            "the wall itself is OUTSIDE -- the test is strict, as the metric's was",
            "the cross-section ignores depth -- it reads x and y only",
            "the cross-section is batched, one flag per env",
            "RT-107 replay: seat depth 109 mm BESIDE the pocket is no success",
            "RT-107 replay: seat depth 109 mm BESIDE the pocket is not engaged",
        ),
    ),
    # The pocket gate loses its floor again: a tip under the free-standing
    # fixture counts as in the pocket and books depth (RT-201s3, 2026-09-15).
    (
        # The episode record's step counter moves its edge off the success
        # filter's complement: a step AT the threshold stops counting.
        "unclean-flag-drops-the-edge",
        (("    return interpen_dist >= interpen_thresh",
          "    return interpen_dist > interpen_thresh"),),
        ("a step AT the SAPU threshold is unclean -- the complement of the success filter's <",
         "the unclean flag is batched, one flag per env"),
    ),
    (
        "pocket-volume-loses-its-floor",
        (("    return cross_section & (depth <= underside_depth)",
          "    return cross_section"),),
        ("a tip in the cross-section BELOW the fixture underside is NOT (RT-201s3)",
         "the pocket volume is batched, one flag per env"),
    ),
    (
        "sapu-reduction-averages",
        (("return torch.amax(torch.clamp(-per_point_dist, min=0.0), dim=-1)",
          "return torch.sum(torch.clamp(-per_point_dist, min=0.0), dim=-1) / 5.0"),),
        (
            "the reduction takes the MAX, not the mean",
            "clearance elsewhere does not cancel a penetration",
            "ONE point past the threshold discards the whole env",
        ),
    ),
    (
        "sapu-reduction-drops-the-clamp",
        (("torch.amax(torch.clamp(-per_point_dist, min=0.0), dim=-1)",
          "torch.amax(-per_point_dist, dim=-1)"),),
        ("a positive reading clamps to zero", "a clean part keeps the full SAPU scale"),
    ),
    # The damage this one models is the bug that shipped in M1: the kernel
    # reports penetration NEGATIVE, and reading it as positive makes the
    # reduction return zero for every real input -- SAPU and the success
    # filter both silently off, nothing crashing, nothing warning.
    (
        "sapu-reduction-reads-penetration-as-positive",
        (("torch.clamp(-per_point_dist, min=0.0)",
          "torch.clamp(per_point_dist, min=0.0)"),),
        (
            "the reduction takes the MAX, not the mean",
            "a positive reading clamps to zero",
            "clearance elsewhere does not cancel a penetration",
            "a clean part keeps the full SAPU scale",
            "ONE point past the threshold discards the whole env",
        ),
    ),
    (
        # RE-TARGETED 2026-09-02 with the per-term stack: the composition is
        # the three rows now, so "outside the scale" is the two bonus rows
        # taken UNSCALED. A tunnelled seat then pays both bonuses and, through
        # step_task, a full payout on top -- the exploit SAPU exists for.
        "bonuses-outside-the-sapu-scale",
        (("    engaged_row = engaged_pay * scale\n    success_row = success_pay * scale",
          "    engaged_row = engaged_pay\n    success_row = success_pay"),),
        (
            "tunneling past the SAPU threshold pays nothing",
            "a tunnelled seat is paid out nothing",
            "the batched tunnelling env is the zero one",
        ),
    ),
    # -- the task-space action (inbox entry "Audit 2026-09-03 (a)") ----------
    (
        # The rotation delta composed in the TOOL frame instead of the env
        # frame. Same magnitude, opposite sense once the part hangs down.
        "rotation-delta-post-multiplied",
        (("    return pos + delta_pos, quat_mul(quat_from_axis_angle(delta_rot), quat)",
          "    return pos + delta_pos, quat_mul(quat, quat_from_axis_angle(delta_rot))"),),
        ("the rotation delta is applied in the ENV frame (pre-multiplied, Factory)",),
    ),
    (
        # The box clamps the TIP and forgets to carry the body with it: the
        # controller is then handed the tip position as the body target and
        # the part is commanded one tool length too deep.
        "tip-clamp-returns-the-tip",
        (("    return tip_c - offset_w\n", "    return tip_c\n"),),
        # The five B7 rows joined this list when the per-axis checks were
        # written: every one of them reads a BODY position, so dropping the
        # offset moves all of them. Found by the harness's own "extra" report,
        # not by reading.
        ("a tip inside the box leaves the body position bit-identical",
         "a tip outside the box is pulled onto its face, and the BODY moves with it",
         "a 60 mm tip offset along x is still clamped at the x half-width",
         "a 60 mm tip offset along y is still clamped at the y half-width",
         "a 60 mm tip offset along z passes, because z has its OWN wider half-width",
         "a tip past the z half-width is clamped at z, not at the x/y one",
         "three EQUAL half-widths reproduce the single-half-width body bit for bit",
         "the box is measured from its CENTRE, not from the world origin",
         "each axis takes ITS OWN half-width, proved with three DIFFERENT ones"),
    ),
    # -- the three half-widths (Phase 5 step B7) ---------------------------
    # Each of these hands one axis another axis's half-width. That is the
    # whole failure mode of a per-axis split: the code still clamps, still
    # returns a body position, and only the AXIS the number reached is wrong.
    (
        # z falls back to the x/y width -- the pre-B7 behaviour restored on
        # exactly the axis the change was made for. The margin above the
        # highest legal start goes back to zero and nothing else moves.
        "tip-clamp-uses-the-xy-width-on-z",
        (("            torch.clamp(d[:, 2], min=-half_z, max=half_z),\n",
          "            torch.clamp(d[:, 2], min=-half_x, max=half_x),\n"),),
        ("a 60 mm tip offset along z passes, because z has its OWN wider half-width",
         "a tip past the z half-width is clamped at z, not at the x/y one",
         "each axis takes ITS OWN half-width, proved with three DIFFERENT ones"),
    ),
    (
        # THE CENTRE IS DROPPED from the displacement. Invisible in every probe
        # that centres the box on the origin, which was all of them until the
        # round-2 critic wrote this one; in the env the box moves to the world
        # origin and the commanded tip goes with it.
        "the-box-forgets-its-centre",
        (("    d = tip - centre\n", "    d = tip\n"),),
        ("the box is measured from its CENTRE, not from the world origin",),
    ),
    (
        # x is bounded by the y half-width and back. Dormant while the two hold
        # the same number -- which is the state the per-axis split leaves them
        # in, so nothing but three DIFFERENT half-widths can separate them.
        "tip-clamp-swaps-the-x-and-y-widths",
        (("            torch.clamp(d[:, 0], min=-half_x, max=half_x),\n"
          "            torch.clamp(d[:, 1], min=-half_y, max=half_y),\n",
          "            torch.clamp(d[:, 0], min=-half_y, max=half_y),\n"
          "            torch.clamp(d[:, 1], min=-half_x, max=half_x),\n"),),
        ("each axis takes ITS OWN half-width, proved with three DIFFERENT ones",),
    ),
    (
        # x takes the WIDER z width: the sideways box silently grows past
        # Factory's bound, which is the direction that looks harmless.
        "tip-clamp-uses-the-z-width-on-x",
        (("            torch.clamp(d[:, 0], min=-half_x, max=half_x),\n",
          "            torch.clamp(d[:, 0], min=-half_z, max=half_z),\n"),),
        ("a tip outside the box is pulled onto its face, and the BODY moves with it",
         "a 60 mm tip offset along x is still clamped at the x half-width",
         "the box is measured from its CENTRE, not from the world origin",
         "each axis takes ITS OWN half-width, proved with three DIFFERENT ones"),
    ),
    (
        # The same on y. Named separately because x and y hold the SAME number
        # in the cfg: one mutation could never separate them.
        "tip-clamp-uses-the-z-width-on-y",
        (("            torch.clamp(d[:, 1], min=-half_y, max=half_y),\n",
          "            torch.clamp(d[:, 1], min=-half_z, max=half_z),\n"),),
        ("a 60 mm tip offset along y is still clamped at the y half-width",
         "each axis takes ITS OWN half-width, proved with three DIFFERENT ones"),
    ),
    (
        # The cone clamps EVERYTHING to vertical: no tilt survives, D-071's
        # edge-first entry is impossible, and the policy can never lean.
        "cone-clamps-every-lean",
        (("    excess = torch.clamp(tilt - max_tilt_rad, min=0.0)\n",
          "    excess = tilt\n"),),
        # NOT "yaw stays free": a yawed UPRIGHT part has tilt 0, so even
        # "excess = tilt" leaves it alone. The plane check flips instead: the
        # lean collapses to exactly vertical, x = 0 instead of -sin(cone).
        ("a lean inside the cone comes back bit-identical",
         "a lean beyond the cone is pulled back to EXACTLY the cone angle",
         "the pull-back moves the tool axis STRAIGHT toward vertical (stays in its own plane)",
         "a part pointing straight up is pulled to the cone without NaN",
            "a lean beyond the cone is pulled back to the cone angle FROM THE POCKET",
            "a part aligned with a tilted pocket is not touched by the cone",
            "a part antiparallel to a TILTED pocket axis is pulled to the cone without NaN",
            "an upright reference reproduces the pre-B7 world-vertical clamp exactly",
        ),
    ),
    (
        # The correction axis reversed: the pull-back pushes the lean FURTHER.
        "cone-pulls-the-wrong-way",
        # RE-POINTED in step B7: the axis is a `torch.cross` now, because the
        # reference direction is no longer a constant the expression could be
        # written out around.
        (("    n = torch.cross(tool, down, dim=-1)", "    n = torch.cross(down, tool, dim=-1)"),),
        # The straight-up case is NOT in this list: it takes the degenerate
        # fallback axis, which the sign flip does not touch.
        ("a lean beyond the cone is pulled back to EXACTLY the cone angle",
         "the pull-back moves the tool axis STRAIGHT toward vertical (stays in its own plane)",
            "a lean beyond the cone is pulled back to the cone angle FROM THE POCKET",
         "an upright reference reproduces the pre-B7 world-vertical clamp exactly"),
    ),
    # -- RT-171: the alignment shaping row --------------------------------
    (
        # The sign the plan review caught: "+z against +z" pays for an
        # upside-down part and charges the seated one.
        "tilt-sign-flipped",
        (("    return -axes_in_pocket[:, 2, 2]\n", "    return axes_in_pocket[:, 2, 2]\n"),),
        (
            "a part hanging straight down an untilted pocket reads cos theta = +1",
            "a part pointing straight UP reads cos theta = -1",
            "a pocket tilted by beta with the part still vertical reads cos beta",
            "a part tilted WITH its tilted pocket reads cos theta = +1",
            "yaw about the pocket axis leaves cos theta at +1",
            "tilt_cos_theta is batched over envs",
        ),
    ),
    (
        # sech(a x) is 2 * squash(x, a, 0); drop the 2 and the potential peaks
        # at w/2, the margin value is 0.05 w and the D-109 (9) rule is gone.
        "tilt-potential-not-squashed",
        (("    return w_tilt * 2.0 * squash(theta, a_tilt, 0.0)\n",
          "    return w_tilt * squash(theta, a_tilt, 0.0)\n"),),
        (
            "Phi is w upright",
            "Phi is 0.1 w at the margin (the D-109 (9) rule)",
            "Phi is sech(a * theta) at 5.42 and 8.52 deg",
        ),
    ),
    (
        # The absorbing state keeps its potential: the success step then
        # pays gamma * Phi(s') - Phi(s) like any other, and the discounted
        # sum over a successful episode no longer telescopes to -Phi(s_0).
        "tilt-terminal-not-zeroed",
        (("    tilt_phi_next = torch.where(success, zero, tilt_phi)\n",
          "    tilt_phi_next = tilt_phi\n"),),
        (
            "the success step zeroes Phi(s')",
            "the discounted row over a trajectory that ends in success telescopes to -Phi(s_0)",
        ),
    ),
)


# The guard lives in a different file, so it needs its own mutation table.
CFG_MUTATIONS: tuple[tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]], ...] = (
    # -- resolve_start_lateral_offset (D-178 (3)) ---------------------------
    (
        # The radius is read as a diameter: every run draws half the offset
        # its command and its run tag state, and nothing downstream can tell.
        "the-radius-is-read-as-a-diameter",
        (("    radius = float(value)\n", "    radius = float(value) * 0.5\n"),),
        ("a number passes through as the radius in metres",),
    ),
    (
        # A D-176 pair is read as its first entry instead of refused: an old
        # `[0.006, 0.002]` command runs as a 6 mm disk and nobody is told the
        # pair is gone.
        "a-pair-is-read-as-its-first-entry",
        (("    if isinstance(value, (list, tuple)):\n        raise ValueError(\n",
          "    if isinstance(value, (list, tuple)):\n        value = value[0]\n"
          "    if False:\n        raise ValueError(\n"),),
        ("a pair or a hydra list is refused, with a pointer to D-178",),
    ),
    (
        # A negative radius passes. `r * cos(phi)` with a negative r is the
        # same disk turned half way, so nothing downstream notices the sign.
        "a-negative-radius-passes",
        (("    if radius < 0.0:\n", "    if False:\n"),),
        ("a negative radius is refused",),
    ),
    (
        # `True` reaches the number branch (bool IS an int) and becomes a
        # radius of one METRE.
        "a-bool-is-taken-for-a-number",
        (("    if isinstance(value, bool) or not isinstance(value, (int, float)):\n",
          "    if not isinstance(value, (int, float)):\n"),),
        ("a string and a bool are refused, never coerced",),
    ),
    (
        "guard-never-fires",
        (("    if missing:", "    if False:"),),
        (
            "an env with NOTHING set is refused",
            "the refusal names every missing value at once",
            "the refusal carries each value's mark",
            "ONE unset value is enough to refuse",
            "the refusal names only what is actually missing",
        ),
    ),
    (
        # D-165: the sign guard on the progress weight removed.
        "negative-progress-weight-accepted",
        (("    if float(cfg.w_depth_progress) < 0.0:", "    if False:"),),
        ("a negative depth-progress weight is refused",),
    ),
    (
        # RT-171: the sign guard on the tilt weight removed.
        "negative-tilt-weight-accepted",
        (("    if float(cfg.w_tilt) < 0.0:", "    if False:"),),
        ("a negative tilt weight is refused",),
    ),
    (
        # RT-171: the discount range guard removed.
        "shaping-gamma-unbounded",
        (("    if not 0.0 < float(cfg.shaping_gamma) <= 1.0:", "    if False:"),),
        ("a shaping gamma outside (0, 1] is refused",),
    ),
    (
        # D-182 / D-183: the scatter-width sign guard removed. A negative
        # sigma then builds an env, and the Gaussian is symmetric enough that
        # nothing downstream ever complains.
        "negative-scatter-width-slips-through",
        (("        if float(getattr(cfg, _name)) < 0.0:", "        if False:"),),
        ("a negative observation-scatter width is refused",),
    ),
    (
        "guard-inverted",
        # REPOINTED 2026-08-30: the missing-list comprehension reads
        # ``demanded`` (the ladder-filtered view of RL_PENDING) since
        # ``curriculum_enabled`` arrived. Same inversion, same meaning.
        (("in demanded if getattr(cfg, name) is None]",
          "in demanded if getattr(cfg, name) is not None]"),),
        (
            "an env with NOTHING set is refused",
            "the refusal names every missing value at once",
            "the refusal carries each value's mark",
            "the refusal names only what is actually missing",
            "a fully configured env passes the guard",
            # ADDED 2026-08-30 with the one-name table. The inverted guard
            # reads "missing = the names that ARE set", so with a single open
            # value the one-unset case has nothing left to report. At three
            # names the other two still triggered a refusal and this check
            # survived the mutation -- the coverage grew when the table shrank.
            "ONE unset value is enough to refuse",
        ),
    ),
    (
        "guard-cannot-see-an-absent-field",
        (("if not hasattr(cfg, name)]", "if False]"),),
        (
            "a field the guard names but the cfg lacks is refused",
            "the code-fault refusal names the absent field",
            "the code-fault refusal does not blame a measurement",
        ),
    ),
    (
        # The counter-proof for the RT-59 result. The link name is MEASURED, so
        # putting it back into RL_PENDING is the careless edit that matters
        # now: it would read as protection while protecting nothing, and it
        # would send the next reader off to author an asset that RT-59 showed
        # is not needed. Without a mark it also breaks the mark pairing.
        "measured-link-name-put-back-into-rl-pending",
        (('RL_PENDING: tuple[str, ...] = (',
          'RL_PENDING: tuple[str, ...] = ("force_sensor_body_name",'),),
        (
            "every pending name carries a mark",
            "the one open value is named",
            "the measured link name is NOT pending",
            # The ladder-off double leaves the smuggled-in name None, and the
            # name has no mark -- the guard dies on KeyError while composing
            # its refusal, so the ladder-off pass check sees the wrong
            # exception instead of a pass.
            "a ladder-off env with no step size passes the guard",
            "AutoDR with the ladder off passes the guard",
            # D-165: same mechanism -- the negative-weight double dies on the
            # KeyError before the sign guard is reached.
            "a negative depth-progress weight is refused",
            # RT-171 (2026-09-06): the two new guard checks assert the TYPE
            # like the D-165 one and flip with it under this mutation.
            "a negative tilt weight is refused",
            "a shaping gamma outside (0, 1] is refused",
            # D-182: the observation-scatter guard is the same shape as the
            # three above and dies on the same wrecked table before its own
            # branch is reached.
            "a negative observation-scatter width is refused",
            # The last three are not padding. The fixture cfg used below no
            # longer carries a ``force_sensor_body_name`` attribute at all, so
            # re-listing the name sends the guard down its CODE-FAULT path
            # instead of its missing-measurement path, and the wording of every
            # refusal changes with it. That is worth declaring: it is the
            # second, quieter half of the same mistake.
            "an env with NOTHING set is refused",
            "the refusal names every missing value at once",
            "the refusal carries each value's mark",
        ),
    ),
    (
        # F_max has a value now, so the guard has nothing to refuse about it.
        # Listing it as pending anyway would read as protection that is not
        # there -- the ONE shape this split exists to prevent.
        "fmax-back-in-rl-pending",
        (('RL_PENDING: tuple[str, ...] = (',
          'RL_PENDING: tuple[str, ...] = ("force_abort_f_max_n",'),),
        (
            "every pending name carries a mark",
            "the one open value is named",
            "a placeholder is never also pending",
            # The empty double gets its 50 N back from the placeholder loop, so
            # the refusal cannot name it and this check sees the inconsistency
            # from the other side.
            "the refusal names every missing value at once",
            # F_max has no entry in RL_PENDING_MARKS, so a pending name without
            # a mark is exactly what the refusal cannot carry.
            "the refusal carries each value's mark",
        ),
    ),
    (
        # The invented number quietly stops being declared as invented.
        "fmax-no-longer-a-placeholder",
        (('    "force_abort_f_max_n": (', '    "some_other_value": ('),),
        ("F_max is a named placeholder",),
    ),
    (
        "placeholder-mark-no-longer-says-invented",
        (('"[placeholder] no longer INVENTED', '"no longer INVENTED'),),
        ("every placeholder mark says the number is invented",),
    ),
    (
        # The careless edit that matters now, written the way it would really
        # be written: put the value back into the prose "so the reader sees it
        # right away". It reads well and it is a second home for the number.
        "the-value-is-written-back-into-its-own-mark",
        (("[placeholder] no longer INVENTED -- that was",
          "[placeholder] 50.0 N, no longer INVENTED -- that was"),),
        ("no placeholder mark restates the number it labels",),
    ),
    (
        # Somebody empties the placeholder and the env builds anyway, reading
        # None where it expects a force limit.
        "emptied-placeholder-slips-through",
        (("emptied = [name for name in RL_PLACEHOLDERS if getattr(cfg, name) is None]",
          "emptied = []"),),
        (
            "an emptied placeholder is refused",
            "the placeholder refusal names the placeholder",
            "the placeholder refusal does not send anyone off to measure",
        ),
    ),
    (
        # REPOINTED 2026-08-30. It used to delete ``engaged_depth_m``, which is
        # a decided value now and no longer in the list. The mistake it models
        # is unchanged -- somebody tidies an open number out of RL_PENDING --
        # but with one name left the tidy-up EMPTIES the table, so the whole
        # guard stops guarding rather than guarding less. That is the stronger
        # version of the same mistake, and the checks that index ``pending``
        # disappear with it instead of raising IndexError.
        "the-last-open-number-dropped-from-the-list",
        # DISAMBIGUATED 2026-08-30: the bare line '    "rung_step_sizes",'
        # occurs TWICE since CURRICULUM_PENDING exists, and the loader refuses
        # an ambiguous mutation. The full three-line tuple pins the edit to
        # RL_PENDING -- which is also the mistake being modelled: somebody
        # tidies the open number out of the DEMAND table and leaves the
        # subset behind.
        (('RL_PENDING: tuple[str, ...] = (\n    "rung_step_sizes",\n)',
          'RL_PENDING: tuple[str, ...] = (\n)'),),
        (
            "every pending name carries a mark",
            "the one open value is named",
            # The subset now names a value the big table no longer demands.
            "CURRICULUM_PENDING is a subset of RL_PENDING",
            # D-165: the negative-weight double dies on an AttributeError in
            # the emptied table before the sign guard is reached.
            "a negative depth-progress weight is refused",
            # RT-171 (2026-09-06): the two new guard checks assert the TYPE
            # like the D-165 one and flip with it under this mutation.
            "a negative tilt weight is refused",
            "a shaping gamma outside (0, 1] is refused",
            # D-182: same mechanism again -- the emptied table kills the guard
            # before the observation-scatter branch runs.
            "a negative observation-scatter width is refused",
            # With the table empty the guard has nothing to refuse, so the
            # env with NOTHING set builds -- and every check that lived inside
            # that refusal, or indexed the table, is simply gone.
            "an env with NOTHING set is refused",
            "the refusal names every missing value at once",
            "the refusal carries each value's mark",
            "ONE unset value is enough to refuse",
            "the refusal names only what is actually missing",
            "a field the guard names but the cfg lacks is refused",
            "the code-fault refusal names the absent field",
            "the code-fault refusal does not blame a measurement",
            # The doubles build their fields FROM the table, so with it empty
            # they carry no rung_step_sizes at all -- and the ladder-off
            # branch dies on AttributeError instead of judging anything.
            "a ladder-off env with no step size passes the guard",
            "AutoDR with the ladder off passes the guard",
            "a step size set while the ladder is OFF is refused",
            "the ladder-off refusal names the unread value",
        ),
    ),
    (
        # The D-080 counter-proof for the ladder-off REFUSAL: the branch that
        # rejects a set-but-unread step size is deleted, and the old
        # check_seated_success TEST SENTINEL shape slips through again.
        "ladder-off-accepts-a-set-step-size",
        (("    if not ladder_on:", "    if False:"),),
        (
            "a step size set while the ladder is OFF is refused",
            "the ladder-off refusal names the unread value",
        ),
    ),
    (
        # Phase 5 step B6. The ladder/AutoDR guard is deleted: both mechanisms
        # may widen start height, fixture offset and tilt in the same run, and
        # the range an episode gets is whichever wrote last.
        "ladder-and-autodr-accepted-together",
        (('    if ladder_on and str(cfg.dr_mode) != "off":', "    if False:"),),
        (
            "the ladder and AutoDR together are refused",
            "the ladder/AutoDR refusal names both mechanisms",
        ),
    ),
    (
        # The guard drops `ladder_on`, so it refuses AutoDR in EVERY state --
        # including `curriculum_enabled=False`, which is the whole Phase 5
        # study. A guard that refuses everything passes its refusal check on
        # its own; only the pass direction catches this.
        "the-guard-refuses-autodr-with-the-ladder-off-too",
        (('    if ladder_on and str(cfg.dr_mode) != "off":',
          '    if str(cfg.dr_mode) != "off":'),),
        ("AutoDR with the ladder off passes the guard",),
    ),
    (
        # The refusal stops naming the mode it refused. The run is still
        # refused, and the reader is left to guess which of the two
        # mechanisms the message is about.
        "the-ladder-autodr-refusal-hides-the-mode",
        (('            f"curriculum_enabled is True together with dr_mode={cfg.dr_mode!r}. "',
          '            "curriculum_enabled is True together with another mechanism. "'),),
        ("the ladder/AutoDR refusal names both mechanisms",),
    ),
    (
        # The D-080 counter-proof for the ladder-off PASS: the demand filter
        # is dropped, the guard demands the step size in every state, and the
        # first-training configuration (ladder off, value None) cannot build
        # -- the D-110 (3) ring closes again.
        "ladder-off-still-demands-the-step-size",
        (("    demanded = RL_PENDING if ladder_on else tuple(n for n in RL_PENDING if n not in CURRICULUM_PENDING)",
          "    demanded = RL_PENDING"),),
        (
            "a ladder-off env with no step size passes the guard",
            "AutoDR with the ladder off passes the guard",
        ),
    ),
    (
        # The 2026-08-30 counter-proof for the decided engaged depth. Putting
        # it back into RL_PENDING would make the guard refuse to build an env
        # whose number is decided and derived -- protection that protects
        # nothing, and it would send the next reader off to the CAD for a value
        # that already has a home in insertion_tasks_cfg.ENGAGED_DEPTH.
        "decided-engaged-depth-put-back-into-rl-pending",
        (('RL_PENDING: tuple[str, ...] = (',
          'RL_PENDING: tuple[str, ...] = ("engaged_depth_m",'),),
        (
            "every pending name carries a mark",
            "the one open value is named",
            "the decided engaged depth is NOT pending",
            # REPOINTED 2026-09-01. The name went back into RL_PLACEHOLDERS
            # when its fraction was downgraded, so this mutation now puts it in
            # BOTH tables -- the one state the split exists to forbid, and the
            # check that sees it is the overlap one. The two KeyError-side
            # flips this list used to declare are gone with the same move: the
            # double fills every placeholder with a value, so the name is no
            # longer left None and the guard never reaches its refusal.
            "a placeholder is never also pending",
            # The refusal still fires -- rung_step_sizes is unset in the empty
            # double -- but it cannot name the smuggled-in name, which HAS a
            # value from the placeholder loop and has no entry in
            # RL_PENDING_MARKS either. Both refusal-content checks see that.
            "the refusal names every missing value at once",
            "the refusal carries each value's mark",
        ),
    ),
    (
        # The invented abort payment quietly stops being declared as invented,
        # the way F_max could. Two placeholders now, and the table is the only
        # thing that makes either visible in a run's report and metrics file.
        "abort-payment-no-longer-a-placeholder",
        (('    "abort_payment": (', '    "some_other_payment": ('),),
        ("the abort payment is a placeholder, not pending",),
    ),
    # -- resolve_control_mode (inbox entry "Audit 2026-09-03 (a)") ----------
    (
        # The OSC rate is never written: the env runs the OSC at D-024's
        # 60 Hz while the report and the metrics claim the 15 Hz placeholder.
        "osc-decimation-not-applied",
        (("        cfg.decimation = dec\n", "        cfg.decimation = cfg.decimation\n"),),
        ("osc mode runs at osc_decimation and renders at that interval",
         "the OSC episode keeps its 256-step count (seconds follow the rate)"),
    ),
    (
        # The seconds are left at the 60 Hz value: at decimation 8 the episode
        # silently shrinks to 64 steps and the time penalty -1/256 no longer
        # sums to -1.
        "osc-episode-seconds-not-rederived",
        (("        cfg.episode_length_s = int(cfg.episode_steps) * float(cfg.sim.dt) * dec\n",
          "        cfg.episode_length_s = float(cfg.episode_length_s)\n"),),
        ("the OSC episode keeps its 256-step count (seconds follow the rate)",),
    ),
    (
        # A typo in a hydra override runs the OTHER controller.
        "unknown-control-mode-accepted",
        (("    if mode not in CONTROL_MODES:\n", "    if False:\n"),),
        ("an unknown control mode is refused",),
    ),
)


def _run(mutations=(), cfg_mutations=()) -> dict[str, bool]:
    return {name: ok for name, ok, _ in build_checks(load_math(mutations), cfg_mutations)}


def _self_test() -> int:
    """Break the source one way at a time; require exactly the named flips."""
    print("[check_insertion_math] counter-proof (D-080)")
    baseline = _run()
    rc = 0
    if not all(baseline.values()):
        bad = [k for k, v in baseline.items() if not v]
        print(f"  FAIL  baseline is not green, counter-proof is meaningless: {bad}")
        return 1
    print(f"  PASS  baseline: {len(baseline)} checks green")

    for name, muts, expected in MUTATIONS:
        got = _run(muts)
        # A check that DISAPPEARED counts as flipped: a mutation can remove the
        # branch a check lived in, and "it never ran" is not "it passed".
        flipped = {k for k in baseline if k not in got or not got[k]}
        missing = set(expected) - flipped
        extra = flipped - set(expected)
        unknown = set(expected) - set(baseline)
        if unknown:
            print(f"  FAIL  {name}: names no existing check {sorted(unknown)}")
            rc = 1
            continue
        if not flipped:
            print(f"  FAIL  {name}: flipped NOTHING -- these checks can never fail")
            rc = 1
        elif missing or extra:
            print(f"  FAIL  {name}: missing {sorted(missing)} extra {sorted(extra)}")
            rc = 1
        else:
            print(f"  PASS  {name}: flips exactly {len(flipped)} declared check(s)")

    for name, muts, expected in CFG_MUTATIONS:
        got = _run((), muts)
        # A check that DISAPPEARED counts as flipped: a mutation can remove the
        # branch a check lived in, and "it never ran" is not "it passed".
        flipped = {k for k in baseline if k not in got or not got[k]}
        missing = set(expected) - flipped
        extra = flipped - set(expected)
        unknown = set(expected) - set(baseline)
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
    results = build_checks(load_math())
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
