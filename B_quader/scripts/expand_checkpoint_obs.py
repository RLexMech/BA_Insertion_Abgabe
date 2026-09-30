"""Widen a trained rsl_rl checkpoint from a 21-channel to a 25-channel observation.

D-037 appends the pocket quaternion as observation channels 21:25. That changes
the input width of the policy and value networks, so the D-034 checkpoint
``model_3950.pt`` (99.65 % training, 100 % replay at tilt 0, 50 % at 5 deg)
cannot be loaded into the new env as it stands. Training the tilt curriculum
from scratch would throw that policy away; at the fixed pose the same rung cost
77 789 episodes to reach 90 %.

**What this script does.** Every tensor whose input dimension is the OLD
observation width is widened to the new width, and the added entries are
neutral:

- first-layer weight matrices ``(out, 21) -> (out, 25)``: the four new columns
  are ZERO,
- observation-normaliser buffers ``(21,) -> (25,)``: new mean 0, new variance
  and standard deviation 1, so the new channels pass through unscaled and are
  then multiplied by the zero columns above.

The observation enters the network at exactly one place -- the first linear
layer of the actor and of the critic -- so zero columns there make the widened
network compute the **same function** as the original for every input whose
first 21 channels agree. The expanded policy is therefore not "close to" the
old one, it is the old one, and its first measurement is a regression test with
two numbers already on disk: replay at tilt 0 must reproduce 100 %, and
``--fixture-tilt 5`` must reproduce ~50 %. A deviation means this script was
wrong, not that the policy changed.

**Optimizer state is deliberately dropped** (``state`` emptied, param groups
kept). Its Adam moments are per-parameter tensors whose shapes no longer match,
and they are keyed by parameter index rather than by name, so widening them
correctly would mean reconstructing the model's parameter order -- a guess this
script refuses to make. Adam re-initialises its moments lazily on the first
update; the cost is a few iterations of adaptation at the start of a run that is
changing task anyway.

Usage (training machine, inside the Isaac conda env; needs torch, nothing else):

    python scripts/expand_checkpoint_obs.py --self-test
    python scripts/expand_checkpoint_obs.py IN.pt OUT.pt

``--self-test`` builds a synthetic checkpoint with the same structure, runs the
same transform over it and asserts every invariant. Run it first: it separates
"the script is broken" from "this checkpoint has an unexpected layout" before
either can be blamed for a bad replay.

UNVERIFIED: written on the dev PC, which has no torch.
"""

from __future__ import annotations

import argparse
import sys

import torch

OLD_OBS_DEFAULT = 21
NEW_OBS_DEFAULT = 25

# Normaliser buffers, by the suffix of their key, and the value the new entries
# get. mean 0 and var/std 1 make the appended channels pass through unchanged,
# which is what the zero weight columns then rely on.
NORM_FILL = {
    "mean": 0.0,
    "var": 1.0,
    "std": 1.0,
    "running_mean": 0.0,
    "running_var": 1.0,
}


def _fill_for(key: str) -> float | None:
    """Neutral fill value for a normaliser buffer, or None if the key is not one.

    The leading underscore is stripped: rsl_rl's EmpiricalNormalization
    registers its buffers as ``_mean``, ``_var``, ``_std`` and ``_count``
    (observed in model_3950.pt on the training machine, 2026-08-16), while
    other implementations use the bare names.
    """
    leaf = key.rsplit(".", 1)[-1].lstrip("_")
    return NORM_FILL.get(leaf)


def expand_state_dict(state: dict, old_obs: int, new_obs: int, label: str) -> list[str]:
    """Widen every observation-width tensor in ``state`` in place.

    Returns the report lines describing what was changed. Raises ValueError on
    anything ambiguous rather than guessing: a silently wrong widening produces
    a policy that runs and is subtly wrong, which is the failure mode this
    whole project logs problems about.
    """
    pad = new_obs - old_obs
    report: list[str] = []
    for key in list(state.keys()):
        value = state[key]
        if not torch.is_tensor(value):
            continue
        if old_obs not in tuple(value.shape):
            continue

        # Decide by NAME, not by shape. rsl_rl stores the normaliser mean and
        # variance as (1, obs), which is shape-indistinguishable from a weight
        # matrix with one output; sorting them by shape alone would zero-fill a
        # normaliser mean, which rescales every channel rather than only the
        # new ones. Observed on model_3950.pt, 2026-08-16.
        fill = _fill_for(key)
        if fill is not None:
            if value.shape[-1] != old_obs:
                raise ValueError(
                    f"{label}: normaliser buffer '{key}' has shape {tuple(value.shape)}, whose LAST "
                    f"dimension is not the old observation width {old_obs}. Refusing to guess which "
                    "axis to widen."
                )
            shape = list(value.shape)
            shape[-1] = new_obs
            widened = torch.full(shape, fill, dtype=value.dtype)
            widened[..., :old_obs] = value
            state[key] = widened
            report.append(
                f"  {key}: {tuple(value.shape)} -> {tuple(widened.shape)}, {pad} entr(y/ies) = {fill}"
            )
            continue

        # First-layer weights: (out_features, obs).
        if value.dim() == 2 and value.shape[1] == old_obs and "weight" in key:
            widened = torch.zeros(value.shape[0], new_obs, dtype=value.dtype)
            widened[:, :old_obs] = value
            state[key] = widened
            report.append(f"  {key}: {tuple(value.shape)} -> {tuple(widened.shape)}, {pad} zero column(s)")
            continue

        raise ValueError(
            f"{label}: tensor '{key}' has shape {tuple(value.shape)}, which contains the old "
            f"observation width {old_obs}, but it is neither a known normaliser buffer "
            f"({sorted(NORM_FILL)}, with or without a leading underscore) nor a 2-D weight matrix "
            "whose second dimension is the observation. Refusing to guess -- a wrong widening here "
            "produces a policy that runs and is silently wrong. Inspect the checkpoint and extend "
            "this script."
        )
    return report


def expand_checkpoint(ckpt: dict, old_obs: int, new_obs: int) -> list[str]:
    """Widen every state dict inside a loaded checkpoint. Returns report lines."""
    report: list[str] = []
    touched_any = False
    for top_key, value in ckpt.items():
        if not isinstance(value, dict):
            continue
        if top_key == "optimizer_state_dict":
            continue
        tensors = {k: v for k, v in value.items() if torch.is_tensor(v)}
        if not tensors:
            continue
        lines = expand_state_dict(value, old_obs, new_obs, label=top_key)
        if lines:
            touched_any = True
            report.append(f"{top_key}:")
            report.extend(lines)

    if not touched_any:
        raise ValueError(
            f"nothing in this checkpoint had an observation dimension of {old_obs}. Either it was "
            "already expanded, or --old-obs is wrong, or its layout differs from the assumed "
            "rsl_rl one. Nothing was written."
        )

    opt = ckpt.get("optimizer_state_dict")
    if isinstance(opt, dict) and opt.get("state"):
        opt["state"] = {}
        report.append("optimizer_state_dict: per-parameter Adam moments cleared (see module docstring)")
    return report


def _self_test() -> None:
    """Round-trip a synthetic checkpoint and assert every invariant."""
    old, new = 21, 25
    torch.manual_seed(0)
    # Mirrors the layout actually found in model_3950.pt on the training
    # machine (2026-08-16): rsl_rl's EmpiricalNormalization registers _mean,
    # _var and _std with a leading underscore and shape (1, obs) -- the same
    # shape as a one-output weight matrix, which is why the transform sorts
    # tensors by name rather than by shape. The bare-name, 1-D variants are
    # kept alongside so the script stays usable if a future rsl_rl renames them.
    ckpt = {
        "model_state_dict": {
            "actor.0.weight": torch.randn(128, old),
            "actor.0.bias": torch.randn(128),
            "actor.2.weight": torch.randn(128, 128),
            "critic.0.weight": torch.randn(128, old),
            "actor_obs_normalizer._mean": torch.randn(1, old),
            "actor_obs_normalizer._var": torch.rand(1, old) + 0.5,
            "actor_obs_normalizer._std": torch.rand(1, old) + 0.5,
            "actor_obs_normalizer._count": torch.tensor(1234, dtype=torch.long),
            "critic_obs_normalizer._mean": torch.randn(1, old),
            "critic_obs_normalizer._var": torch.rand(1, old) + 0.5,
            "critic_obs_normalizer._std": torch.rand(1, old) + 0.5,
            "legacy_normalizer.mean": torch.randn(old),
            "legacy_normalizer.var": torch.rand(old) + 0.5,
        },
        "optimizer_state_dict": {"state": {0: {"exp_avg": torch.randn(128, old)}}, "param_groups": [{"lr": 1e-3}]},
        "iter": 3950,
    }
    before = {k: v.clone() for k, v in ckpt["model_state_dict"].items()}
    report = expand_checkpoint(ckpt, old, new)
    print("\n".join(report))

    state = ckpt["model_state_dict"]
    checks = 0
    for key in ("actor.0.weight", "critic.0.weight"):
        w = state[key]
        assert w.shape == (128, new), f"{key} shape {tuple(w.shape)}"
        assert torch.equal(w[:, :old], before[key]), f"{key}: original columns changed"
        assert torch.count_nonzero(w[:, old:]) == 0, f"{key}: new columns are not zero"
        checks += 3
    for key, fill in (
        ("actor_obs_normalizer._mean", 0.0),
        ("actor_obs_normalizer._var", 1.0),
        ("actor_obs_normalizer._std", 1.0),
        ("critic_obs_normalizer._mean", 0.0),
        ("critic_obs_normalizer._var", 1.0),
    ):
        b = state[key]
        assert b.shape == (1, new), f"{key} shape {tuple(b.shape)}"
        assert torch.equal(b[..., :old], before[key]), f"{key}: original entries changed"
        assert torch.all(b[..., old:] == fill), f"{key}: new entries are not {fill}"
        checks += 3
    for key, fill in (("legacy_normalizer.mean", 0.0), ("legacy_normalizer.var", 1.0)):
        b = state[key]
        assert b.shape == (new,), f"{key} shape {tuple(b.shape)}"
        assert torch.equal(b[:old], before[key]), f"{key}: original entries changed"
        assert torch.all(b[old:] == fill), f"{key}: new entries are not {fill}"
        checks += 3
    assert state["actor.0.bias"].shape == (128,), "bias must not be touched"
    assert state["actor.2.weight"].shape == (128, 128), "hidden layer must not be touched"
    assert torch.equal(state["actor_obs_normalizer._count"], before["actor_obs_normalizer._count"])
    assert ckpt["optimizer_state_dict"]["state"] == {}, "optimizer moments must be cleared"
    assert ckpt["iter"] == 3950, "iteration counter must survive"
    checks += 4

    # The refusal path is the one that matters most: it is what stopped a
    # (1, 21) normaliser mean from being zero-filled as if it were a weight
    # matrix on 2026-08-16. A transform that silently handles the unknown case
    # is worse than one that stops.
    hostile = {"model_state_dict": {"mystery.tensor": torch.randn(7, old)}}
    try:
        expand_checkpoint(hostile, old, new)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("an unrecognised observation-width tensor must abort, not be guessed at")

    print(f"\nSELF-TEST PASSED ({checks} assertions)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", help="checkpoint to read, e.g. .../model_3950.pt")
    parser.add_argument("output", nargs="?", help="checkpoint to write, e.g. .../model_3950_obs25.pt")
    parser.add_argument("--old-obs", type=int, default=OLD_OBS_DEFAULT)
    parser.add_argument("--new-obs", type=int, default=NEW_OBS_DEFAULT)
    parser.add_argument("--self-test", action="store_true", help="check the transform on a synthetic checkpoint and exit")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if not args.input or not args.output:
        parser.error("input and output are required unless --self-test is given")
    if args.new_obs <= args.old_obs:
        parser.error(f"--new-obs ({args.new_obs}) must exceed --old-obs ({args.old_obs})")

    print(f"reading  {args.input}")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print(f"ERROR: expected a dict checkpoint, got {type(ckpt).__name__}", file=sys.stderr)
        return 1

    print(f"widening observation {args.old_obs} -> {args.new_obs}")
    try:
        report = expand_checkpoint(ckpt, args.old_obs, args.new_obs)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("\n".join(report))

    torch.save(ckpt, args.output)
    print(f"\nwrote    {args.output}")
    print(
        "\nNEXT: this is a claim, not a result. Replay the expanded checkpoint at tilt 0 "
        "(must reproduce 100 %) and at --fixture-tilt 5 (must reproduce ~50 %). Those two "
        "numbers are on disk from D-036 and are the only proof that the widening was neutral."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
