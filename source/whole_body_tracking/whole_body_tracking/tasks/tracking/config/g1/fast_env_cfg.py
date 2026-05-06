"""G1FastEnvCfg: shared env for all fast-training tasks.

Replaces obs noise with ScaledUniformNoiseCfg (reads env.obs_noise_scale,
default 1.0 = identical to base) and push event with curriculum_push
(reads env.push_velocity_scale, default 1.0 = identical to base).

All fast-training tasks use this single env cfg; behavior differs only
by what the runner writes to obs_noise_scale / push_velocity_scale.
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.tasks.tracking.mdp.curriculum_noise import ScaledUniformNoiseCfg
from whole_body_tracking.tasks.tracking.mdp.curriculum_events import (
    curriculum_push_by_setting_velocity,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE


def _s(n_min: float, n_max: float) -> ScaledUniformNoiseCfg:
    return ScaledUniformNoiseCfg(n_min=n_min, n_max=n_max)


@configclass
class G1FastEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Replace obs noise with curriculum-aware version.
        # Amplitudes match G1FlatEnvCfg; scale=1.0 by default → no change.
        obs = self.observations.policy
        obs.motion_anchor_pos_b.noise = _s(-0.25,  0.25)
        obs.motion_anchor_ori_b.noise = _s(-0.05,  0.05)
        obs.base_lin_vel.noise        = _s(-0.5,   0.5)
        obs.base_ang_vel.noise        = _s(-0.2,   0.2)
        obs.joint_pos.noise           = _s(-0.01,  0.01)
        obs.joint_vel.noise           = _s(-0.5,   0.5)

        # Replace push_robot with curriculum-aware version.
        # velocity_range unchanged; scale=1.0 by default → no change.
        self.events.push_robot = EventTerm(
            func=curriculum_push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": VELOCITY_RANGE},
        )
