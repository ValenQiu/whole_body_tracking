"""curriculum_push_by_setting_velocity: push range × env.push_velocity_scale.

Wraps isaaclab's push_by_setting_velocity so that the velocity bounds are
scaled by the ``push_velocity_scale`` attribute on ``env`` (default 1.0).
When scale=1.0 the behaviour is identical to the original function.
"""
from __future__ import annotations

import torch

from isaaclab.envs.mdp.events import push_by_setting_velocity


def curriculum_push_by_setting_velocity(
    env,  # ManagerBasedEnv
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
) -> None:
    """Push robot with velocity bounds scaled by ``env.push_velocity_scale``."""
    scale = float(getattr(env, "push_velocity_scale", 1.0))
    scaled_range = {axis: (lo * scale, hi * scale) for axis, (lo, hi) in velocity_range.items()}
    push_by_setting_velocity(env, env_ids, velocity_range=scaled_range)
