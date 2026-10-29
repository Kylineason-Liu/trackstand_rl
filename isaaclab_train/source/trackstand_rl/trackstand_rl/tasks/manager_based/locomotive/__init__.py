"""Locomotive riding task: CycleRL cylinder paper recipe on the trackstand_rl plant.

Trained with skrl (scripts/skrl/train_locomotive.py), not RSL-RL. Independent of the
trackstand task package except for its robot cfg (same USD and joints).
"""

import gymnasium as gym

from .locomotive_env_cfg import LocomotiveEnvCfg

gym.register(
    id="Locomotive-Cylinder-Aligned-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LocomotiveEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
    },
)
