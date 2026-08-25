# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

# XUAN_BIKE model configs
from .xuan_bike.cyclerl_env_cfg import CycleRLEnvCfg
from .xuan_bike.cyclerl_flywheel_env_cfg import CycleRLFlywheelEnvCfg
from .xuan_bike.cyclerl_pid_flywheel_env_cfg import CycleRLPIDFlywheelEnvCfg

# THUBIKE model configs
from .thu_bike.cyclerl_env_cfg_thubike2 import THUBikeEnvCfg
from .thu_bike.thubike_balance_env_cfg import THUBikeBalanceEnvCfg
from .thu_bike.thubike_trackstand_env_cfg import TrackstandEnvCfg

# NEW_BIKE model configs
from .new_bike.cyclerl_env_cfg import NewBikeEnvCfg
from .new_bike.newbike_balance_env_cfg import NewBikeBalanceEnvCfg, NewBikeBalanceTorqueEnvCfg


# Register the environment with the gym registry
gym.register(
    id="CycleRL-Bicycle-v0",
    # Point to the generic RL environment class
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # This key is recognized by the IsaacLab-standard environment 
        # when initialized via the gym factory
        "cfg": CycleRLEnvCfg(), 
        "env_cfg_entry_point": CycleRLEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="CycleRL-Flywheel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Standard Isaac Lab 3.0 kwarg for passing the config type
        "env_cfg_entry_point": CycleRLFlywheelEnvCfg,
        # Re-using your existing SKRL yaml configuration
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml", 
    },
)

# Register the task so you can run it via the command line
gym.register(
    id="CycleRL-PIDFlywheel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": CycleRLPIDFlywheelEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="CycleRL-THUBike-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": THUBikeEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="CycleRL-NewBike-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NewBikeEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="THUBike-Balance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": THUBikeBalanceEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="NewBike-Balance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NewBikeBalanceEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_small_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NewBikeBalancePPORunnerCfg",
    },
)

gym.register(
    id="NewBike-Balance-Torque-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NewBikeBalanceTorqueEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_small_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NewBikeBalancePPORunnerCfg",
    },
)

gym.register(
    id="THUBike-Trackstand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TrackstandEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
    },
)