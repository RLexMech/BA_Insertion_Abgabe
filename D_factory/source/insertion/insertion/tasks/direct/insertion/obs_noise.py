# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation noise with per-episode bias and per-step jitter."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import (
    GaussianNoiseCfg,
    NoiseCfg,
    NoiseModelCfg,
    NoiseModelWithAdditiveBias,
    NoiseModelWithAdditiveBiasCfg,
    UniformNoiseCfg,
)

from . import insertion_math


class InsertionObsNoise(NoiseModelWithAdditiveBias):
    """Per-channel observation noise with a per-episode bias."""

    def __init__(
        self, noise_model_cfg: InsertionObsNoiseCfg, num_envs: int, device: str
    ):
        step_std, bias_std = insertion_math.noise_masks(
            pocket_pos_std_m=float(noise_model_cfg.pocket_pos_std_m),
            force_std_n=float(noise_model_cfg.force_std_n),
            torque_std_nm=float(noise_model_cfg.torque_std_nm),
            mode=str(noise_model_cfg.obs_mode),
            device=device,
        )
        super().__init__(
            NoiseModelWithAdditiveBiasCfg(
                noise_cfg=GaussianNoiseCfg(std=step_std),
                bias_noise_cfg=UniformNoiseCfg(
                    n_min=-bias_std, n_max=bias_std, operation="abs"
                ),
            ),
            num_envs,
            device,
        )
        _width = insertion_math.obs_dim(str(noise_model_cfg.obs_mode))
        self._bias = torch.zeros(num_envs, _width, device=self._device)
        self._num_components = _width


@configclass
class InsertionObsNoiseCfg(NoiseModelCfg):
    """The two scalars, and nothing else that could become a tensor."""

    class_type: type = InsertionObsNoise

    noise_cfg: NoiseCfg | None = None

    pocket_pos_std_m: float = 0.0
    force_std_n: float = 0.0
    torque_std_nm: float = 0.0
    obs_mode: str = "force"


def resolve_obs_noise_model(cfg) -> InsertionObsNoiseCfg | None:
    """The cfg's two sigmas -> a noise model, or ``None`` for no model at all."""
    pocket = float(cfg.obs_noise_pocket_pos_std_m)
    force = float(cfg.force_obs_noise_std_n)
    torque = float(cfg.torque_obs_noise_std_nm)
    if pocket == 0.0 and force == 0.0 and torque == 0.0:
        return None
    return InsertionObsNoiseCfg(pocket_pos_std_m=pocket, force_std_n=force,
                                torque_std_nm=torque, obs_mode=str(cfg.obs_wrench_mode))
