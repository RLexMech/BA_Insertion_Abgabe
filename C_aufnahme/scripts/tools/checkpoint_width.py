# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""Reject actor observation-width mismatch before runner.load.

Force mode needs 28 channels; wrench mode needs 31. For rsl_rl 3.0.1,
read the second dimension of the first 2-D actor.<n>.weight in
model_state_dict. Unknown layouts return None and defer to runner.load.
Do not pad or transfer weights: observation layout and normalisation are
trained together. Stdlib-only shape inspection; --self-test needs no torch.

Usage:
    python scripts/tools/checkpoint_width.py --self-test
"""

from __future__ import annotations

import re
import sys

_ACTOR_WEIGHT = re.compile(r"^actor\.(\d+)\.weight$")


def actor_input_width(state_dict) -> int | None:
    """The observation width the actor's first linear layer expects, or ``None`` when the state dict carries no ``actor.<n>.weight`` of rank 2."""
    found: list[tuple[int, int]] = []
    for key, value in state_dict.items():
        m = _ACTOR_WEIGHT.match(str(key))
        if m is None:
            continue
        shape = tuple(getattr(value, "shape", ()))
        if len(shape) == 2:
            found.append((int(m.group(1)), int(shape[1])))
    if not found:
        return None
    return min(found)[1]


def width_mismatch_message(
    ckpt_width: int | None,
    env_width: int,
    env_mode: str,
    mode_widths: dict[str, int],
    path: str = "",
) -> str | None:
    """``None`` when the checkpoint fits the env (or its width is unknown); otherwise the one line to print before refusing."""
    if ckpt_width is None or ckpt_width == env_width:
        return None
    fitting = sorted(m for m, w in mode_widths.items() if w == ckpt_width)
    where = f" ({path})" if path else ""
    head = (f"checkpoint{where} was trained on a {ckpt_width}-wide observation, "
            f"but the env builds {env_width} (obs_wrench_mode={env_mode!r}).")
    if fitting:
        return head + (f" It matches obs_wrench_mode={fitting[0]!r}: pass "
                       f"env.obs_wrench_mode={fitting[0]} on the command line. "
                       "No padding, no weight transfer.")
    return head + (f" No observation mode builds {ckpt_width} channels "
                   f"({', '.join(f'{m}={w}' for m, w in sorted(mode_widths.items()))}); "
                   "this checkpoint is from another layout of the observation.")


def refusal_for_checkpoint(path: str, env_mode: str, env_width: int,
                           mode_widths: dict[str, int], loader) -> str | None:
    """The whole guard in one call: load ``path`` with ``loader`` (torch.load bound by the caller), read the actor width out of ``model_state_dict``, compare."""
    loaded = loader(path)
    state = loaded.get("model_state_dict") if isinstance(loaded, dict) else None
    if not isinstance(state, dict):
        return None
    return width_mismatch_message(actor_input_width(state), env_width, env_mode,
                                  mode_widths, path)


def _self_test() -> int:
    class _T:
        def __init__(self, *shape):
            self.shape = shape

    modes = {"force": 28, "wrench": 31}
    sd28 = {"actor_obs_normalizer._mean": _T(1, 28), "actor.0.weight": _T(128, 28),
            "actor.0.bias": _T(128), "actor.2.weight": _T(128, 128), "critic.0.weight": _T(128, 28)}
    sd31 = dict(sd28, **{"actor.0.weight": _T(128, 31)})
    checks = [
        ("actor.0.weight (128, 28) reads width 28", actor_input_width(sd28) == 28),
        ("actor.0.weight (128, 31) reads width 31", actor_input_width(sd31) == 31),
        ("the lowest layer index wins whatever the dict order",
         actor_input_width({"actor.2.weight": _T(128, 128), "actor.0.weight": _T(128, 28)}) == 28),
        ("critic keys alone give None", actor_input_width({"critic.0.weight": _T(128, 28)}) is None),
        ("a rank-1 actor tensor is not a width", actor_input_width({"actor.0.weight": _T(128)}) is None),
        ("an empty state dict gives None", actor_input_width({}) is None),
        ("28 into a 28 env: no message",
         width_mismatch_message(28, 28, "force", modes) is None),
        ("unknown width: no message (runner.load speaks)",
         width_mismatch_message(None, 31, "wrench", modes) is None),
        ("28 into a 31 env names obs_wrench_mode=force",
         "env.obs_wrench_mode=force" in (width_mismatch_message(28, 31, "wrench", modes) or "")),
        ("31 into a 28 env names obs_wrench_mode=wrench",
         "env.obs_wrench_mode=wrench" in (width_mismatch_message(31, 28, "force", modes) or "")),
        ("a width no mode builds says so",
         "No observation mode builds 25" in (width_mismatch_message(25, 28, "force", modes) or "")),
        ("the path is quoted when given",
         "(model_10.pt)" in (width_mismatch_message(28, 31, "wrench", modes, "model_10.pt") or "")),
        ("a mismatch never returns None",
         width_mismatch_message(28, 31, "wrench", modes) is not None),
        ("refusal_for_checkpoint refuses a 28 checkpoint in a 31 env",
         refusal_for_checkpoint("m.pt", "wrench", 31, modes,
                                lambda p: {"model_state_dict": sd28}) is not None),
        ("refusal_for_checkpoint lets a 31 checkpoint into a 31 env",
         refusal_for_checkpoint("m.pt", "wrench", 31, modes,
                                lambda p: {"model_state_dict": sd31}) is None),
        ("a checkpoint without model_state_dict is left to runner.load",
         refusal_for_checkpoint("m.pt", "wrench", 31, modes, lambda p: {"iter": 3}) is None),
    ]
    bad = [name for name, ok in checks if not ok]
    if bad:
        print(f"[checkpoint_width] self-test: FAIL {bad}")
        return 1
    print(f"[checkpoint_width] self-test: {len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(_self_test())
    print(__doc__)
