# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# The Isaac Lab project template ends with `from .ui_extension_example import
# *`. That line was copied in phase B but the module it names never was, and
# rightly so: it is a Kit UI panel example with no place in a headless
# training package. Removed 2026-08-18, when the first import of this package
# on a machine WITH Isaac died on it -- the dev laptop cannot import isaaclab
# at all, so the dangling import had stayed invisible since the copy.
