"""NewBike task configs (locomotion and balance)."""

from .cyclerl_env_cfg import NewBikeEnvCfg
from .newbike_balance_env_cfg import NewBikeBalanceEnvCfg, NewBikeBalanceTorqueEnvCfg

__all__ = ["NewBikeEnvCfg", "NewBikeBalanceEnvCfg", "NewBikeBalanceTorqueEnvCfg"]
