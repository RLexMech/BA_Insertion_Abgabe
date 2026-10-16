# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to print all the available environments in Isaac Lab."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List Isaac Lab environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import insertion.tasks  # noqa: F401


def main():
    """Print all environments registered in `insertion` extension."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "Template-" in task_spec.id and (args_cli.keyword is None or args_cli.keyword in task_spec.id):
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        simulation_app.close()
