# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42

    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    experiment_name = "ur5e_insertion"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerMlpAblationCfg(PPORunnerCfg):
    """MLP ablation arm, `--agent rsl_rl_mlp_ablation_cfg_entry_point`."""

    num_steps_per_env = 128
    max_iterations = 400
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.num_mini_batches = 4


@configclass
class PPORunnerLstmCfg(PPORunnerMlpAblationCfg):
    """recurrent actor-critic, `--agent rsl_rl_rnn_cfg_entry_point`."""

    def __post_init__(self):
        super().__post_init__()
        self.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=self.policy.init_noise_std,
            actor_obs_normalization=self.policy.actor_obs_normalization,
            critic_obs_normalization=self.policy.critic_obs_normalization,
            actor_hidden_dims=self.policy.actor_hidden_dims,
            critic_hidden_dims=self.policy.critic_hidden_dims,
            activation=self.policy.activation,
            rnn_type="lstm",
            rnn_hidden_dim=256,
            rnn_num_layers=2,
        )
