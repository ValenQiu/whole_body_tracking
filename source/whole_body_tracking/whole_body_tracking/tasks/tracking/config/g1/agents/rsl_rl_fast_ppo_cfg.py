"""Runner cfgs for fast-training tasks.

Design (MimicKit-inspired):
  - No boolean feature flags.  Features are controlled by sentinel hyperparameter
    values so that FastMotionOnPolicyRunner contains zero if-branches.
  - Sentinel conventions:
      noise_scale_start = 1.0   → noise curriculum disabled (schedule is flat)
      push_scale_start  = 1.0   → push  curriculum disabled (schedule is flat)
      min_iterations = int(1e9) → early stopping disabled (never triggers)
  - Each ablation variant is its own cfg subclass that overrides only the
    hyperparameters it activates — exactly like AMPAgent / ADDAgent in MimicKit.
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
    # Noise curriculum: ramp obs_noise_scale from noise_scale_start → 1.0
    # Sentinel: noise_scale_start=1.0  means the schedule is always 1.0 (disabled)
    noise_warmup_iters: int   = 5000
    noise_scale_start: float  = 1.0   # sentinel = disabled
    # Push curriculum: ramp push_velocity_scale from push_scale_start → 1.0
    # Sentinel: push_scale_start=1.0  means the schedule is always 1.0 (disabled)
    push_warmup_iters: int    = 8000
    push_scale_start: float   = 1.0   # sentinel = disabled
    # Plateau early stopping
    # Sentinel: min_iterations=int(1e9) means the check never triggers
    plateau_window: int       = 1000
    plateau_threshold: float  = 5e-5
    min_iterations: int       = int(1e9)  # sentinel = disabled


@configclass
class G1FastNoisePPORunnerCfg(G1FastBasePPORunnerCfg):
    """M1: curriculum obs noise only."""
    experiment_name = "g1_fast_noise"
    noise_scale_start: float  = 0.2   # activate noise curriculum


@configclass
class G1FastPushPPORunnerCfg(G1FastBasePPORunnerCfg):
    """M2: curriculum push only."""
    experiment_name = "g1_fast_push"
    push_scale_start: float   = 0.1   # activate push curriculum


@configclass
class G1FastEarlyStopPPORunnerCfg(G1FastBasePPORunnerCfg):
    """M3: early stopping only."""
    experiment_name = "g1_fast_earlystop"
    min_iterations: int       = 5000  # activate early stopping


@configclass
class G1FastFullPPORunnerCfg(G1FastBasePPORunnerCfg):
    """M_Full / production: all three features active."""
    experiment_name = "g1_fast_full"
    noise_scale_start: float  = 0.2
    push_scale_start: float   = 0.1
    min_iterations: int       = 5000
