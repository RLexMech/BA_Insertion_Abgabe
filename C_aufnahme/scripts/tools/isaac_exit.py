# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""End an Isaac script with an exit code that survives the shutdown (D-081)."""

from __future__ import annotations

import os
import sys
from typing import Any

ISAAC_EXIT_MARKER = "isaac_exit-2026-08-24a"


def exit_with(simulation_app: Any, code: int, tag: str = "isaac_exit") -> None:
    """Never returns. Ends the process with ``code``, shutdown or no shutdown."""
    code = int(code)
    sys.stdout.flush()
    sys.stderr.flush()
    if code == 0:
        simulation_app.close()
        os._exit(0)
    print(f"[{tag}] exiting {code} WITHOUT the Isaac shutdown: close() does not return "
          f"(measured, RT-22), so a non-zero code cannot be delivered after it.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
