"""ScaledUniformNoiseCfg: noise amplitude × env.obs_noise_scale (default 1.0).

Drop-in replacement for AdditiveUniformNoiseCfg.  The env is expected to
expose an attribute ``obs_noise_scale`` (float, 0..1).  When the attribute is
absent the class falls back to scale=1.0 (full noise), preserving backward
compatibility with envs that do not set the attribute.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass, field


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
