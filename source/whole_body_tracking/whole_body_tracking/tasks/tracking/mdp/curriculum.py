"""Unified curriculum utilities for fast training.

Contains:
- ScaledUniformNoiseCfg: uniform noise with runtime scale field.
- curriculum_push_by_setting_velocity: push range × env.push_velocity_scale.
"""
from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg


def curriculum_uniform_noise(data: torch.Tensor, cfg: "ScaledUniformNoiseCfg") -> torch.Tensor:
    """Uniform noise with runtime amplitude scale on top of [n_min, n_max]."""
    scale = float(getattr(cfg, "scale", 1.0))
    lo = cfg.n_min * scale
    hi = cfg.n_max * scale

    if cfg.operation == "add":
        return data + torch.rand_like(data) * (hi - lo) + lo
    elif cfg.operation == "scale":
        return data * (torch.rand_like(data) * (hi - lo) + lo)
    elif cfg.operation == "abs":
        return torch.rand_like(data) * (hi - lo) + lo
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


@configclass
class ScaledUniformNoiseCfg(UniformNoiseCfg):
    """Uniform noise cfg with additional runtime scale field."""

    func = curriculum_uniform_noise
    scale: float = 1.0


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
