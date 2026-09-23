"""Trackstand-Balance-v0 task family: SDF-wheel trackstand balance.

Mirrors the CycleRL cyclerl package layout: agents/, assets/, bicycle_cfg.py,
mdp/, and a per-bike subpackage (trackstand_bike/) holding the env class and
the env config.
"""

import gymnasium as gym

from .trackstand_bike.trackstand_rl_env import TrackstandRLEnv
from .trackstand_bike.trackstand_rl_env_cfg import TrackstandRLEnvCfg

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
