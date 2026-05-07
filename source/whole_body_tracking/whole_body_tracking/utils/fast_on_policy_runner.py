"""FastMotionOnPolicyRunner: curriculum scheduling + plateau early stopping.

Design (MimicKit-inspired):
  - Only log() and learn() are overridden; rollout and PPO update are
    inherited from MotionOnPolicyRunner unchanged.
  - Zero boolean feature flags.  All behavior is controlled by hyperparameter
    sentinel values supplied via train_cfg:
      noise_scale_start = 1.0   → linear schedule is flat → curriculum off
      push_scale_start  = 1.0   → same
      min_iterations = int(1e9) → plateau check never triggers → early stop off
  - This keeps the class free of if-branches; config drives behavior,
    exactly as MimicKit's agent hierarchy does.

Keys read from train_cfg (all optional, sentinels default to disabled):
    noise_warmup_iters   int    default 5000
    noise_scale_start    float  default 1.0   (sentinel: 1.0 = disabled)
    push_warmup_iters    int    default 8000
    push_scale_start     float  default 1.0   (sentinel: 1.0 = disabled)
    plateau_window       int    default 1000
    plateau_threshold    float  default 5e-5
    min_iterations       int    default int(1e9)  (sentinel: large = disabled)
"""
from __future__ import annotations

import statistics

import wandb
from rsl_rl.env import VecEnv

from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner


class _EarlyStop(Exception):
    """Raised inside log() to signal plateau early stop to learn()."""

    def __init__(self, it: int):
        self.it = it


def _linear_schedule(it: int, start: float, end: float, warmup: int) -> float:
    """Linearly ramp from start → end over warmup iterations, then hold end.

    When start == end (sentinel), the schedule is always end — no-op.
    """
    if warmup <= 0 or start == end:
        return end
    return min(end, start + (end - start) * it / warmup)


class FastMotionOnPolicyRunner(MotionOnPolicyRunner):
    """Extends MotionOnPolicyRunner with curriculum scheduling + early stopping.

    Features activate/deactivate purely through hyperparameter values —
    no boolean flags, no branching in the runner class itself.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device,
                         registry_name=registry_name)

        self.noise_warmup_iters = int(train_cfg.get("noise_warmup_iters",   5000))
        self.noise_scale_start  = float(train_cfg.get("noise_scale_start",  1.0))
        self.push_warmup_iters  = int(train_cfg.get("push_warmup_iters",    8000))
        self.push_scale_start   = float(train_cfg.get("push_scale_start",   1.0))
        self.plateau_window     = int(train_cfg.get("plateau_window",       1000))
        self.plateau_threshold  = float(train_cfg.get("plateau_threshold",  5e-5))
        self.min_iterations     = int(train_cfg.get("min_iterations",       int(1e9)))

        self._reward_history: list[float] = []

    # ------------------------------------------------------------------
    # Curriculum helpers (no branches — sentinels make schedule a no-op)
    # ------------------------------------------------------------------

    def _update_curriculum(self, it: int) -> None:
        """Update runtime curriculum scales on env and observation noise cfg."""
        env = self.env.unwrapped
        obs_noise_scale = _linear_schedule(
            it, self.noise_scale_start, 1.0, self.noise_warmup_iters)
        push_velocity_scale = _linear_schedule(
            it, self.push_scale_start, 1.0, self.push_warmup_iters)

        env.obs_noise_scale = obs_noise_scale
        env.push_velocity_scale = push_velocity_scale

        # Keep noise cfg in sync with runtime scale for ScaledUniformNoiseCfg.
        obs_cfg = getattr(getattr(env, "cfg", None), "observations", None)
        policy_cfg = getattr(obs_cfg, "policy", None) if obs_cfg is not None else None
        if policy_cfg is not None:
            for term_name in (
                "motion_anchor_pos_b",
                "motion_anchor_ori_b",
                "base_lin_vel",
                "base_ang_vel",
                "joint_pos",
                "joint_vel",
            ):
                term = getattr(policy_cfg, term_name, None)
                noise = getattr(term, "noise", None) if term is not None else None
                if noise is not None and hasattr(noise, "scale"):
                    noise.scale = obs_noise_scale

    def _check_plateau(self, mean_reward: float, it: int) -> bool:
        """Return True when reward has plateaued and min_iterations passed.

        With min_iterations=int(1e9) this always returns False (disabled).
        """
        self._reward_history.append(mean_reward)
        if len(self._reward_history) > self.plateau_window:
            self._reward_history.pop(0)
        if it < self.min_iterations or len(self._reward_history) < self.plateau_window:
            return False
        n = len(self._reward_history)
        x_mean = (n - 1) / 2.0
        y_mean = sum(self._reward_history) / n
        num = sum((i - x_mean) * (r - y_mean)
                  for i, r in enumerate(self._reward_history))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 1e-12 else 0.0
        return abs(slope) < self.plateau_threshold

    # ------------------------------------------------------------------
    # Hooks — only log() and learn() are overridden
    # ------------------------------------------------------------------

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Called each iteration by OnPolicyRunner.learn(); inject curriculum."""
        super().log(locs, width, pad)

        it = locs.get("it", 0)
        self._update_curriculum(it)

        env = self.env.unwrapped
        if self.logger_type == "wandb" and wandb.run is not None:
            wandb.log({
                "Train/obs_noise_scale":     getattr(env, "obs_noise_scale",     1.0),
                "Train/push_velocity_scale": getattr(env, "push_velocity_scale", 1.0),
            }, commit=False)

        rewbuffer = locs.get("rewbuffer", [])
        mean_r = statistics.mean(rewbuffer) if rewbuffer else 0.0
        if self._check_plateau(mean_r, it):
            print(f"[FastRunner] Plateau at iter {it} "
                  f"(slope < {self.plateau_threshold}); saving checkpoint.")
            if self.log_dir:
                self.save(f"{self.log_dir}/model_{it}_early_stop.pt")
            raise _EarlyStop(it)

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Wrap parent learn() to catch _EarlyStop."""
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
        except _EarlyStop as e:
            print(f"[FastRunner] Training stopped early at iteration {e.it}.")
