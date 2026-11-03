"""V3: v1 with the yaw_deviation termination replaced by a dense heading penalty.

Why: in the 2026-11-03 DR ablation (summaries/2026-11-03_2332_trackstand_yaw_claim_check.md)
yaw > 30 deg ended 23-78 % of Isaac DR-on episodes. The v1 reward has no heading
term, so heading was penalised only by the 30 deg termination (lost survival reward,
no gradient below the wall). Earlier campaign notes (progress_record.md, iter9/10)
also found that recovery turns swing past 30 deg, squeezing the policy between the
yaw wall and the fall wall.

Change vs v1 (everything else, including obs/actions/actuators/DR/events, inherited):
- terminations.yaw_deviation -> None (removed).
- rewards.yaw_penalty (NEW): balance_yaw_excess_penalty =
  -max(|heading - spawn heading| - band, 0) / (pi - band): 0 within band (0.1 rad),
  falling linearly to -1 at |yaw| = pi. A pure penalty (no constant offset), so the
  alive bonus stays v1's (survival 5 + drift_gauss up to 2) and v3 vs v1 changes only
  the heading signal. It is balance_yaw_band - 1: unclamped, constant gradient out to
  a full reversal, no zero-gradient parking spot (the Gaussian and clamped variants
  failed for that reason, see balance_yaw_band). Reads the raw root quaternion and
  the env's per-episode spawn_yaw (TrackstandRLEnvV1 bookkeeping), the audited path.
  Worst case -2.0/step at weight 2.0 is below survival's +5.0, so ending an episode
  early is never better than continuing.

Orbiting is still bounded by the drifted (0.5 m) and wheel_net (7.2 rad)
terminations. The 11-obs / 2-action tensor contract is identical to v1, so v1
MuJoCo and deploy code runs v3 policies; only evaluation terminations differ (the
MuJoCo evaluators apply the 30 deg yaw termination unless --no_yaw_termination).

Weight 2.0 matches drift_gauss: -0.28/step at the old 30 deg wall, -2.0/step facing
backwards. Tune with ``env.rewards.yaw_penalty.weight=<w>`` or
``env.rewards.yaw_penalty.params.band=<rad>``.

MuJoCo eval of v3 policies: pass ``--no_yaw_termination`` to run_deploy_eval.py /
run_deploy_viewer.py (the MuJoCo env reports yaw but not this reward).
"""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils.configclass import configclass

from ..mdp.stationary.rewards import balance_yaw_excess_penalty
from .trackstand_rl_env_cfg_v1 import TrackstandRLEnvCfgV1, TrackstandRLRewardsCfgV1

# Penalty-free half-width (rad, ~5.7 deg) and slope 1/(pi - band): value -1 at |yaw| = pi.
YAW_BAND_RAD = 0.1
YAW_BAND_TAIL_SLOPE = 1.0 / (3.141592653589793 - YAW_BAND_RAD)


@configclass
class TrackstandRLRewardsCfgV3(TrackstandRLRewardsCfgV1):
    """V1 rewards (survival 5, drift_gauss 2, rate 0.1) plus a dense heading penalty."""

    # The function returns values in [-1, 0]; the weight stays POSITIVE.
    yaw_penalty = RewardTermCfg(
        func=balance_yaw_excess_penalty,
        weight=2.0,
        params={"band": YAW_BAND_RAD, "tail_slope": YAW_BAND_TAIL_SLOPE},
    )


@configclass
class TrackstandRLEnvCfgV3(TrackstandRLEnvCfgV1):
    """V1 plant and contract; heading is shaped by reward instead of terminated."""

    rewards = TrackstandRLRewardsCfgV3()

    def __post_init__(self):
        super().__post_init__()
        # Replaced by rewards.yaw_penalty; the term manager skips None entries.
        self.terminations.yaw_deviation = None
