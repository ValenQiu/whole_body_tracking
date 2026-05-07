"""Runner cfg for fast training (production).

Single public class ``G1FastPPORunnerCfg``, symmetric with ``G1FlatPPORunnerCfg``.

Ablation variants (base / noise-only / push-only / early-stop / full) are archived
in ``rsl_rl_fast_ppo_cfg_ablation.py`` and are not registered as gym tasks.
"""
from isaaclab.utils import configclass
from .rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1FastPPORunnerCfg(G1FlatPPORunnerCfg):
    """Fast training: curriculum obs noise + push; early stopping off (sentinel).

    Sentinel conventions (see ``FastMotionOnPolicyRunner``):
        noise_scale_start < 1.0  → noise curriculum active
        push_scale_start  < 1.0  → push curriculum active
        min_iterations = int(1e9) → early stopping disabled
    """
    experiment_name = "g1_fast"
    noise_warmup_iters: int   = 5000
    noise_scale_start: float  = 0.2
    push_warmup_iters: int    = 8000
    push_scale_start: float   = 0.1
    plateau_window: int       = 1000
    plateau_threshold: float  = 5e-5
    min_iterations: int       = int(1e9)
