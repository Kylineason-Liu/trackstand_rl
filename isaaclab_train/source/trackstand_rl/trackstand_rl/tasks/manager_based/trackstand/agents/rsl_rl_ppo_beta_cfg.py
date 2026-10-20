"""Trackstand-Balance-v1: the v0 PPO settings with a bounded Beta actor."""

from importlib.metadata import version

from packaging.version import Version

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .rsl_rl_ppo_cfg import TrackstandRLPPORunnerCfg


@configclass
class TrackstandBetaDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    class_name: str = f"{__package__}.beta_distribution:BetaDistribution"


@configclass
class TrackstandRLBetaPPORunnerCfg(TrackstandRLPPORunnerCfg):
    """Keep v0's network, optimizer, rewards and timing; change only the actor family.

    A new policy must be trained: Gaussian checkpoints are not Beta checkpoints.
    The initial Gaussian std is replaced by the Beta head's symmetric initial
    distribution; all PPO hyperparameters, including entropy_coef, are inherited.
    """

    actor = TrackstandRLPPORunnerCfg().actor.replace(distribution_cfg=TrackstandBetaDistributionCfg())

    def __post_init__(self):
        # Older RSL-RL config migration can discard distribution_cfg and fall
        # back to a Gaussian. Refuse that instead of silently violating v1.
        if Version(version("rsl-rl-lib")) < Version("5.0.0"):
            raise RuntimeError("Trackstand-Balance-v1 Beta policies require RSL-RL >= 5.0.0.")
