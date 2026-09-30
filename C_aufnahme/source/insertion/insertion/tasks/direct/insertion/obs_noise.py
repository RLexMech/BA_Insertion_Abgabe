# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

r"""The observation noise model (D-182): pocket-position bias and force noise.

WHAT IS THE LIBRARY'S AND WHAT IS OURS
--------------------------------------
Isaac Lab 2.3.2 owns nearly all of this, and that is why the hook was chosen
over the own form D-111 had built:

* the hook itself -- ``DirectRLEnvCfg.observation_noise_model``
  (``direct_rl_env_cfg.py:172``),
* the construction -- ``class_type(cfg, num_envs, device)``
  (``direct_rl_env.py:210-213``),
* the reset schedule -- ``DirectRLEnv._reset_idx`` calls ``reset(env_ids)``
  (``:624-625``), which ``InsertionEnv._reset_idx`` reaches through its own
  ``super()._reset_idx``,
* the application -- ``:414-415``, AFTER ``_get_observations`` and ONLY to
  ``obs_buf["policy"]``,
* and both noise functions (``utils/noise/noise_model.py``).

That fourth point is the whole argument. Reward, termination and success read
the env's OWN buffers inside ``_get_dones`` / ``_get_rewards``, never
``obs_buf``, so they see the CLEAN values by construction rather than by a
convention somebody has to keep. The own form ``insertion_math.add_obs_offset``
sat inside ``_get_observations``, one method away from those buffers, kept
apart only by hand; it is deleted.

OUR SHARE IS TWO THINGS, and D-182 names both:

1. ``insertion_math.noise_masks`` -- which channel carries which sigma.
2. the PRE-SIZED BIAS BUFFER below.

WHY THE SUBCLASS EXISTS AT ALL
------------------------------
``NoiseModelWithAdditiveBias`` cannot carry a per-channel bias from the first
reset. It creates ``_bias`` as ``(num_envs, 1)`` (``noise_model.py:157``) and
widens it to the channel count only on the first ``__call__``
(``:186-191``) -- but ``reset()`` (``:174``) runs EARLIER: once from
``DirectRLEnv._reset_idx``, and again from ``RslRlVecEnvWrapper``'s
constructor, which calls ``env.reset()`` before any step
(``isaaclab_rl/rsl_rl/vecenv_wrapper.py:66``). A ``(28,)`` std meeting a
one-column buffer breaks there -- on the first reset of every run, before a
single observation exists. Sizing the buffer up front is the entire fix.
``_num_components`` is set with it because that is the flag the library's
widening branch tests (``:186``): leaving it None would let the library
re-sample the buffer we just sized.

THE ``operation="abs"`` IS LOAD-BEARING
---------------------------------------
``reset()`` writes ``self._bias[env_ids] = func(self._bias[env_ids], cfg)``
and ``NoiseCfg.operation`` defaults to ``"add"`` (``noise_cfg.py:29``). With
that default every reset would ADD a fresh draw ONTO the old bias, so the
pocket offset would random-walk away over a run instead of being re-drawn
once per episode -- and a drifting observation bias looks from the outside
exactly like a policy that stops learning. ``"abs"`` makes ``gaussian_noise``
return ``mean + std * randn_like(data)`` and ignore the incoming buffer
(``noise_model.py:96-97``), which is the per-episode draw D-182 asks for.
SINCE 2026-09-15 THE BIAS IS UNIFORM, not Gaussian (user; TacSL Tab. V,
+-5 mm; p1-thesis docs/decisions_inbox.md). ``uniform_noise`` has the same
"abs" branch: it returns ``rand_like(data) * (n_max - n_min) + n_min`` and
ignores the incoming buffer (``noise_model.py:68-69``), so everything above
holds for the uniform draw unchanged.
The per-STEP force noise keeps the additive default on purpose: it is added
to a real reading, not substituted for it.

NAMED SIDE EFFECTS (D-182), so the next reader does not rediscover them
-----------------------------------------------------------------------
* THE FIRST OBSERVATION OF A RUN IS UNNOISED, in three places, because only
  ``step()`` passes the buffer through the hook. ``DirectRLEnv.reset()``
  returns ``self._get_observations()`` directly (``direct_rl_env.py:331``);
  ``RslRlVecEnvWrapper.get_observations()`` calls ``_get_observations()``
  itself (``vecenv_wrapper.py:148``); and ``scripts/rsl_rl/play.py
  --trace-obs`` records raw slices of whatever it was handed, so the first
  row of a trace is clean and the rest are not.
* A PER-STEP ``randn_like`` SHIFTS THE GLOBAL RNG STREAM. A run WITH noise
  therefore draws different reset conditions than one without: the two are
  not the same env with a louder observation, and the no-DR reference run
  has to carry the noise too or it is not a reference.

LEAVING ``noise_cfg`` UNSET IS NOT SAFE. THIS PARAGRAPH SAID IT WAS, AND IT
WAS WRONG (corrected 2026-09-12; the reading it describes is the D-185 bug).
It argued that ``configclass`` turns a MISSING annotation into a
``default_factory`` and that only an explicit ``validate()`` call rejects
MISSING entries (``utils/configclass.py:246``), so an unset field could not
hurt because this class never dereferences it. The missed question is WHO
CALLS ``validate()``: ``DirectRLEnv.__init__`` opens with ``cfg.validate()``
as its FIRST statement (``direct_rl_env.py:90``), so an inherited MISSING
kills EVERY run in the constructor, before physics starts. The class below
therefore BINDS ``noise_cfg: NoiseCfg | None = None``, and its own docstring
carries the full argument. D-185 owns the decision.

STILL UNVERIFIED (D-182, verification status): whether this cfg class
survives hydra's ``to_dict`` and the ``params/env.yaml`` dump is read off the
source, not run. The dump itself is the training PC's answer.
"""

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
    """Per-channel observation noise with a per-episode bias.

    Two channels, one model. ``tip_rel`` carries a UNIFORM bias drawn ONCE
    per episode (a pocket-localisation error, which does not change while the
    part moves); ``force`` carries a Gaussian drawn EVERY step (sensor
    jitter). Every other channel is untouched, which the ``(28,)`` masks say
    by holding zero there -- a zero sigma makes ``std * randn`` exactly zero,
    so the clean value passes through unchanged.

    The cfg carries SCALARS only. The masks are built here, at env-build
    time, because a tensor in the cfg would have to survive hydra's
    ``to_dict`` and the ``params/env.yaml`` every run writes.
    """

    def __init__(
        self, noise_model_cfg: InsertionObsNoiseCfg, num_envs: int, device: str
    ):
        # BY KEYWORD, never by position, and ``noise_masks`` is keyword-only
        # so this is the only way it can be written. Two bare floats passed
        # in the wrong order build a 3.5 m pocket bias and 0.0025 N of force
        # jitter, raise nothing and leave every log line looking plausible --
        # the L-08 defect (``docs/reference/pruefregeln.md``), which a sweep
        # for the parameter name cannot see. Held in place by
        # ``scripts/check_insertion_math.py``, "noise_masks refuses positional
        # sigmas".
        step_std, bias_std = insertion_math.noise_masks(
            pocket_pos_std_m=float(noise_model_cfg.pocket_pos_std_m),
            force_std_n=float(noise_model_cfg.force_std_n),
            torque_std_nm=float(noise_model_cfg.torque_std_nm),
            mode=str(noise_model_cfg.obs_mode),
            device=device,
        )
        # The library's own cfg, built here rather than carried in ours: it
        # holds the two mask TENSORS, so it must not be the object hydra
        # dumps. ``operation="abs"`` on the bias only -- see the header.
        # ``bias_std`` holds the uniform HALF WIDTH (``noise_masks``), so the
        # draw is [-hw, +hw] on tip_rel and exactly 0 on every other channel.
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
        # THE PRE-SIZED BUFFER (the header's second point). The library would
        # allocate (num_envs, 1) and widen it on the first __call__, which is
        # one reset too late. Setting ``_num_components`` here is what tells
        # the library the widening has already happened.
        # The WIDTH IS THE MODE'S (D-188): 28 under ``force``, 31 under
        # ``wrench``. A 28-wide buffer meeting a 31-wide observation would
        # break in the library's own add, one step into the run.
        _width = insertion_math.obs_dim(str(noise_model_cfg.obs_mode))
        self._bias = torch.zeros(num_envs, _width, device=self._device)
        self._num_components = _width


@configclass
class InsertionObsNoiseCfg(NoiseModelCfg):
    """The two scalars, and nothing else that could become a tensor.

    ``noise_cfg`` IS NEVER READ on this class -- this model builds its own
    inner cfg in ``__init__`` from the two scalars below, and
    ``NoiseModel.__init__`` stores THAT one as ``_noise_model_cfg``
    (``noise_model.py:116``). The inherited field is dead weight.

    It is nevertheless bound to ``None`` rather than left inherited, and the
    reason is a hard stop, not tidiness: ``NoiseModelCfg.noise_cfg`` is
    ``MISSING`` (``noise_cfg.py:78``), ``DirectRLEnv.__init__`` opens with
    ``cfg.validate()`` (``direct_rl_env.py:90``), and ``_validate`` walks the
    whole cfg tree and raises ``TypeError: Missing values detected`` on the
    first MISSING it meets (``configclass.py:246-300``). An inherited MISSING
    therefore kills EVERY run in the constructor, before physics starts.
    Measured 2026-09-12 by running that ``_validate`` out of the installed
    tree: MISSING raises, ``None`` passes, a real ``GaussianNoiseCfg`` passes.

    ``None`` rather than a stand-in ``GaussianNoiseCfg(std=0.0)``, and the
    difference is the failure mode. Both silence ``validate()``. But if
    somebody later drops the inner-cfg construction in ``__init__`` and lets
    the base class fall back on this field, ``None`` raises
    ``AttributeError: 'NoneType' object has no attribute 'func'`` at the first
    observation, while a zero-width Gaussian passes the observation through
    UNNOISED and never says a word. The loud failure is the right one.
    Isaac Lab uses the same ``| None`` idiom for "there is no model here"
    (``DirectRLEnvCfg.observation_noise_model``, ``direct_rl_env_cfg.py:172``).
    """

    class_type: type = InsertionObsNoise

    # Bound, not inherited -- see the docstring. Never dereferenced.
    noise_cfg: NoiseCfg | None = None

    # Metres. The HALF WIDTH of the per-EPISODE uniform bias on the
    # pocket-position channels (name kept, user 2026-09-15).
    pocket_pos_std_m: float = 0.0
    # Newtons. The per-STEP Gaussian on the force channels.
    force_std_n: float = 0.0
    # Newton metres. The per-STEP Gaussian on the torque channels (D-188),
    # ``wrench`` mode only; refused by ``noise_masks`` in ``force`` mode.
    torque_std_nm: float = 0.0
    # Which layout the masks and the bias buffer are sized for.
    obs_mode: str = "force"


def resolve_obs_noise_model(cfg) -> InsertionObsNoiseCfg | None:
    """The cfg's two sigmas -> a noise model, or ``None`` for no model at all.

    ONE HOME for the both-zero rule, called from ``InsertionEnv.__init__``
    BEFORE ``super().__init__`` because the base class reads the field inside
    its own constructor.

    ``None`` rather than a zero-width Gaussian, and the difference is not
    cosmetic: a model that exists draws ``randn_like`` every step even at
    sigma 0, which changes nothing in the observation and everything in the
    global RNG stream. The switched-off env has to be the un-noised env bit
    for bit, or the arm a noisy run is compared against is not the arm
    anybody thinks it is.
    """
    pocket = float(cfg.obs_noise_pocket_pos_std_m)
    force = float(cfg.force_obs_noise_std_n)
    torque = float(cfg.torque_obs_noise_std_nm)
    if pocket == 0.0 and force == 0.0 and torque == 0.0:
        return None
    return InsertionObsNoiseCfg(pocket_pos_std_m=pocket, force_std_n=force,
                                torque_std_nm=torque, obs_mode=str(cfg.obs_wrench_mode))
