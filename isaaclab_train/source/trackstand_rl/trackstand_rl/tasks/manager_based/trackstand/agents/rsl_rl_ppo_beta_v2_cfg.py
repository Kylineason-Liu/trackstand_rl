"""V2 reuses v1 Beta/PPO settings with a separate velocity-task log directory."""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_beta_cfg import TrackstandRLBetaPPORunnerCfg


@configclass
class TrackstandRLVelocityBetaPPORunnerCfg(TrackstandRLBetaPPORunnerCfg):
    """Separate incompatible rear-action checkpoints; retain all PPO parameters."""

    experiment_name = "trackstand_rl_v2_rsl"
