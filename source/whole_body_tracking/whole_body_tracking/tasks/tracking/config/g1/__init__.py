import gymnasium as gym

from . import agents, fast_env_cfg, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
    },
)

# Fast training: single task symmetric with Tracking-Flat-G1-v0 (one env + one runner cfg).
gym.register(
    id="Tracking-Fast-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fast_env_cfg.G1FastEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_fast_ppo_cfg:G1FastPPORunnerCfg",
    },
)
