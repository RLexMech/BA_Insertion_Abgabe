# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""End an Isaac script with an exit code that survives the shutdown (D-081).

WHY THIS EXISTS
---------------
MEASURED, run RT-22 (2026-08-23): ``simulation_app.close()`` DOES NOT RETURN.
The run put two print statements after it -- one inside ``except SystemExit``
around the close, one plain, after the ``finally`` block. The counter-proof
finished with PASS, and NEITHER line appeared in the log. The next line in the
log is ``[rt_log] exit code: 0``, written by the wrapper.

Two suspects were eliminated first, in order, rather than one being assumed:

  RT-20  ``python -c "import sys; sys.exit(3)"`` through rt_log.ps1 reports
         ``exit code: 3``. The wrapper reads a non-zero code correctly.
  RT-22  the two lines after close() do not print. So the process ends inside
         close(), and it ends with status 0.

Consequence, and it applies to EVERY Isaac script in this repo, not only the
one that found it: code placed after ``simulation_app.close()`` never runs, and
``raise SystemExit(n)`` there is dead. A failing run that returns 1 from main()
still reports ``exit code: 0``, so the machine-readable half of the two-machine
loop -- the line /rt-check reads -- says the run succeeded.

WHAT THIS DOES ABOUT IT
-----------------------
On success the app is closed normally; whether close() returns is then moot,
because the code to deliver is 0 either way.

On failure the shutdown is SKIPPED and the process exits with the code. That
is a deliberate trade and not a tidy one: Isaac does not get its orderly
teardown. It is taken because the process is ending regardless, the OS reclaims
the GPU context, and the alternative is a failing run that reports success --
which is the defect this whole file exists to remove. Stdout is flushed first,
since ``os._exit`` skips the interpreter's own flush.

HOW FAR THIS GOES ON THE fix_stage_units.py ENTRY -- and where it stops
----------------------------------------------------------------------
The unexplained ``fix_stage_units.py`` entry in HANDOFF-SZENE.md has the same
SHAPE: a passing verdict, exit 0, two expected lines that never print, and an
original that was not replaced. Corrected 2026-08-24, because the first reading
of this section stopped one step too early. It argued the finding cannot apply
because those two prints sit INSIDE main(). What decides it is not where the
PRINTS sit but where the EXCEPTION lands, and that script ran

    try: main()
    finally: simulation_app.close()

So IF anything between the verdict and those prints raised, the exception went
straight into that finally, close() never returned, and the process ended with
no traceback and status 0. The finding therefore DOES explain the SILENCE --
conditionally, on a premise that is still untested.

It explains NOTHING about the cause. Why the replace failed is a separate
question and is still open. The premise ("something raised") is itself
unproven: the process could equally have died inside the copy without an
exception. ``fix_stage_units.py`` now names whichever it is in one run -- the
replace is wrapped and prints the exception type, and the shutdown no longer
eats the code. Nothing here may be quoted as the cause until that run exists.

Usage:

    from tools.isaac_exit import exit_with     # or load it by path
    exit_with(simulation_app, code)
"""

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
