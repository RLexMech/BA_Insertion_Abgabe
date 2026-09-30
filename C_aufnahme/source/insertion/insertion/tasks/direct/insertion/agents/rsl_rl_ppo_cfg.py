# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # The policy seed, spelled out so it is READ and not inherited. Until
    # 2026-09-07 this class set no seed, so every run RT-104..RT-175 silently
    # took the base class default (IsaacLab 2.3.2,
    # source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:141, "Default is 42").
    # The VALUE does not change here; only its provenance does. --seed on the
    # command line still overrides it (scripts/rsl_rl/cli_args.py:71-75) and
    # the run folder always carries it (train.py:301-302).
    seed = 42

    # Demo sprint (branch demo-insertion-sprint, 2026-07-26): hidden dims
    # widened from the Cartpole-sized [32, 32] and observation normalization
    # switched on, because the observation -- 28 channels under
    # obs_wrench_mode="force", 31 under "wrench" (D-188; "19-dim" stood here
    # until 2026-09-13 as a proxy leftover) -- mixes radians (joint pos/vel)
    # with metres (tip-to-pocket), quaternions, newtons and newton metres,
    # unlike the template's original single-modality Cartpole observation.
    # This normalisation is what lets N m stand beside N unscaled.
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    # The log folder every run writes into: logs/rsl_rl/<experiment_name>/.
    # It was "demo_insertion" until 2026-08-28 -- the square-peg demo sprint's
    # own folder -- so real UR5e runs would have landed beside the proxy's,
    # with nothing in the path saying which task produced which curve.
    # ``scripts/compare_runs.py`` defaults to this same string and moves with
    # it; ``check_env_wiring.py`` holds the two together.
    # NOTE: this rename touches NO hyperparameter. CLAUDE.md section Code
    # forbids tuning before the gates are closed.
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