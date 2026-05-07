"""Archived ablation runner cfgs (M0–M3 + M_Full).

Not registered in gymnasium — kept for historical experiments and selftest
regression. Production uses ``rsl_rl_fast_ppo_cfg.G1FastPPORunnerCfg`` only.
"""
from isaaclab.utils import configclass
from .rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1FastBasePPORunnerCfg(G1FlatPPORunnerCfg):
    """Base for all fast-training cfgs.

    All features disabled by sentinel values.  Subclasses activate
    individual features by overriding sentinel fields only.
    """
    experiment_name = "g1_fast"
    noise_warmup_iters: int   = 5000
    noise_scale_start: float  = 1.0
    push_warmup_iters: int    = 8000
    push_scale_start: float   = 1.0
    plateau_window: int       = 1000
    plateau_threshold: float  = 5e-5
    min_iterations: int       = int(1e9)


@configclass
class G1FastNoisePPORunnerCfg(G1FastBasePPORunnerCfg):
    """M1: curriculum obs noise only."""
    experiment_name = "g1_fast_noise"
    noise_scale_start: float  = 0.2


@configclass
class G1FastPushPPORunnerCfg(G1FastBasePPORunnerCfg):
    """M2: curriculum push only."""
    experiment_name = "g1_fast_push"
    push_scale_start: float   = 0.1


@configclass
class G1FastEarlyStopPPORunnerCfg(G1FastBasePPORunnerCfg):
    """M3: early stopping only."""
    experiment_name = "g1_fast_earlystop"
    min_iterations: int       = 5000


@configclass
class G1FastFullPPORunnerCfg(G1FastBasePPORunnerCfg):
    """M_Full: noise + push curriculum; early stop off (base sentinel)."""
    experiment_name = "g1_fast_full"
    noise_scale_start: float  = 0.2
    push_scale_start: float   = 0.1
