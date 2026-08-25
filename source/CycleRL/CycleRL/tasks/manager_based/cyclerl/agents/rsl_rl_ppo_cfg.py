# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl PPO config for the NewBike balance tasks.

Uses the current Isaac Lab rsl_rl API (``obs_groups`` + ``RslRlMLPModelCfg``,
mirroring the shipped AnymalD config), based on lmbike_rl's hyperparameters:
tiny [32, 32] actor/critic, KL-adaptive schedule (desired_kl 0.01), no obs
normalization. Entropy (0.005 -> 0.003) and base LR (1e-3 -> 5e-4) were lowered
from lmbike's values after a run where action std inflated off good policies and
reward regressed — gentler updates + less entropy pressure to stabilize.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class NewBikeBalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 15000
    save_interval = 50
    experiment_name = "newbike_balance_rsl"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,   # between lmbike's 0.005 and 0.001 (0.001 was too little exploration)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,  # lowered from 1e-3: gentler updates so one bad step can't blow away a good policy
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
