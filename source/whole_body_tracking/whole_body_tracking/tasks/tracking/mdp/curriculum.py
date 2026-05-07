"""Unified curriculum utilities for fast training.

Contains:
- ScaledUniformNoiseCfg: noise amplitude × env.obs_noise_scale
- curriculum_push_by_setting_velocity: push range × env.push_velocity_scale
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ScaledUniformNoiseCfg:
    """Uniform noise whose amplitude is multiplied by ``env.obs_noise_scale``."""

    n_min: float = -1.0
    n_max: float = 1.0

    def __call__(self, data: torch.Tensor, env) -> torch.Tensor:  # noqa: ANN001
        scale = float(getattr(env, "obs_noise_scale", 1.0))
        lo = self.n_min * scale
        hi = self.n_max * scale
        return data + (hi - lo) * torch.rand_like(data) + lo


def curriculum_push_by_setting_velocity(
    env,  # ManagerBasedEnv
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
) -> None:
    """Push robot with velocity bounds scaled by ``env.push_velocity_scale``."""
    from isaaclab.envs.mdp.events import push_by_setting_velocity

    scale = float(getattr(env, "push_velocity_scale", 1.0))
    scaled_range = {axis: (lo * scale, hi * scale) for axis, (lo, hi) in velocity_range.items()}
    push_by_setting_velocity(env, env_ids, velocity_range=scaled_range)
