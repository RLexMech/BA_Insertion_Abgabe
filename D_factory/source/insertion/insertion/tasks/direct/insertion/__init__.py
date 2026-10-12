# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Ur5e-Insertion-Direct-v0",
    entry_point=f"{__name__}.insertion_env:InsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insertion_env_cfg:InsertionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        # D-195, chosen with `--agent`: the LSTM and its MLP ablation twin.
        "rsl_rl_rnn_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerLstmCfg",
        "rsl_rl_mlp_ablation_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerMlpAblationCfg",
        # D-200: Factory/FORGE's rl_games yaml (only `name` changed), Factory's
        # registration form (isaaclab_tasks direct/factory/__init__.py:20).
        # Used by scripts/rl_games/train.py and play.py (their default --agent).
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)