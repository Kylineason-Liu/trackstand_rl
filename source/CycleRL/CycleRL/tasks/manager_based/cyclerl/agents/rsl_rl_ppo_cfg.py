# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl PPO config for the NewBike balance tasks.

Uses the current Isaac Lab rsl_rl API (``obs_groups`` + ``RslRlMLPModelCfg``,
mirroring the shipped AnymalD config), based on lmbike_rl's hyperparameters:
tiny [32, 32] actor/critic, KL-adaptive schedule (desired_kl 0.01), no obs
normalization.

Entropy was once lowered 0.005 -> 0.003 (and the docstring claimed LR 1e-3 ->
5e-4, which was never applied) to stop action std inflating off good policies.
That over-corrected into the opposite failure: at 0.003 the std collapses to
~0.006, the policy goes deterministic, and it never learns to balance. Both are
back at lmbike's values. If std inflates late in a run, take an earlier
checkpoint rather than cutting entropy again.
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
        entropy_coef=0.005,   # lmbike's value. 0.003 COLLAPSES: std -> 0.006 by iter 400,
                              # policy goes deterministic, bike falls in ~1 s and never recovers
                              # (measured: best ep_len 36 @0.003 vs 375 @0.005, same seed/config).
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,  # lmbike's value; this is the INITIAL lr only (schedule="adaptive"
                               # re-tunes it from KL every minibatch). The docstring's "-> 5e-4"
                               # was never applied.
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
