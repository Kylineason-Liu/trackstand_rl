"""Trackstand balance tasks: v0 Gaussian, v1 Beta torque and v2 Beta rear velocity.

Mirrors the CycleRL cyclerl package layout: agents/, assets/, bicycle_cfg.py,
mdp/, and a per-bike subpackage (trackstand_bike/) holding the env class and
the env config.
"""

import gymnasium as gym

from .trackstand_bike.trackstand_rl_env import TrackstandRLEnv
from .trackstand_bike.trackstand_rl_env_cfg import TrackstandRLEnvCfg
from .trackstand_bike.trackstand_rl_env_cfg_v1 import TrackstandRLEnvCfgV1
from .trackstand_bike.trackstand_rl_env_cfg_v2 import TrackstandRLEnvCfgV2

gym.register(
    id="Trackstand-Balance-v0",
    entry_point=(
        "source.trackstand_rl.trackstand_rl.tasks.manager_based."
        "trackstand.trackstand_bike.trackstand_rl_env:TrackstandRLEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TrackstandRLEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_small_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:TrackstandRLPPORunnerCfg",
    },
)

gym.register(
    id="Trackstand-Balance-v1",
    entry_point=(
        "source.trackstand_rl.trackstand_rl.tasks.manager_based."
        "trackstand.trackstand_bike.trackstand_rl_env:TrackstandRLEnvV1"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TrackstandRLEnvCfgV1,
        # The existing skrl YAML is Gaussian-only; v1 uses the RSL-RL Beta runner.
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_beta_cfg:TrackstandRLBetaPPORunnerCfg",
    },
)

# Independent v2 registration; the existing v0/v1 registrations are unchanged.
gym.register(
    id="Trackstand-Balance-v2",
    entry_point=(
        "source.trackstand_rl.trackstand_rl.tasks.manager_based."
        "trackstand.trackstand_bike.trackstand_rl_env_v2:TrackstandRLEnvV2"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TrackstandRLEnvCfgV2,
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_ppo_beta_v2_cfg:TrackstandRLVelocityBetaPPORunnerCfg"
        ),
    },
)
