"""V3 reuses v1 Beta/PPO settings with a separate heading-reward log directory."""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_beta_cfg import TrackstandRLBetaPPORunnerCfg


@configclass
class TrackstandRLHeadingBetaPPORunnerCfg(TrackstandRLBetaPPORunnerCfg):
    """Same PPO parameters as v1; v3 runs go to their own folder (different reward/terminations)."""

    experiment_name = "trackstand_rl_v3_rsl"
