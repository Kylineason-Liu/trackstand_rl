"""Locomotive riding task: CycleRL cylinder paper recipe on the trackstand_rl plant.

Trained with skrl (scripts/skrl/train_locomotive.py), not RSL-RL. Independent of the
trackstand task package except for its robot cfg (same USD and joints).
"""

import gymnasium as gym

from .locomotive_env_cfg import LocomotiveEnvCfg
from .locomotive_env_cfg_ablation import ABLATION_CFGS

gym.register(
    id="Locomotive-Cylinder-Aligned-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LocomotiveEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
    },
)

# Port ablation (locomotive_env_cfg_ablation.py): one task per CycleRL-vs-port factor set,
# same skrl agent YAML.
for _name, _cfg in ABLATION_CFGS.items():
    gym.register(
        id=f"Locomotive-Abl-{_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _cfg,
            "skrl_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
        },
    )
