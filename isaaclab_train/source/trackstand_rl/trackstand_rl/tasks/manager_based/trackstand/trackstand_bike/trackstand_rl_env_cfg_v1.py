"""V1: 11 observations, bounded/slewed steering and explicitly clipped motor torques.

Order: gravity_y/z, body angular velocity xyz, steering/rear position (rad),
steering/rear velocity (rad/s), sin(steering), cos(steering). No body linear
velocities. Both the observation function and the matching noise configuration
are independent of v0's 13-channel observation term. The steering action clips
targets to +/-1.57 rad and slews at 23.9 rad/s, once per 50-Hz policy update.
Steering IdealPDActuator clips PD torque to +/-5.6 N m at each 400-Hz physics step.
The rear IdealPDActuator clips feedforward torque to +/-6.28 N m (Kp = Kd = 0).
Rear passive moving friction is 0.106 N m; the equal static threshold is provisional.
Steering passive losses use the 2026-10-20 C1 sweep fit, separate from controller Kd.
V1 also fixes timeout bookkeeping and weights normalized action changes equally.
Termination conditions, reset events, timing and DR settings otherwise inherit v0.
"""

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.utils.configclass import configclass

from ..mdp.stationary.actions import SlewLimitedJointPositionActionCfg
from ..mdp.stationary.noise import PerChannelGaussianNoiseCfg
from ..mdp.stationary.observations import balance_imu_state_v1
from ..mdp.stationary.rewards import balance_action_rate_penalty
from .trackstand_rl_env_cfg import TrackstandRLActionsCfg, TrackstandRLEnvCfg, TrackstandRLRewardsCfg


@configclass
class TrackstandRLActionsCfgV1(TrackstandRLActionsCfg):
    steering = SlewLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=["steering_joint"],
        scale=1.57,
        clip={"steering_joint": (-1.57, 1.57)},
        max_target_velocity_rad_s=23.9,
    )


@configclass
class TrackstandRLObservationsCfgV1(ObservationGroupCfg):
    enable_corruption = True

    imu = ObservationTermCfg(
        func=balance_imu_state_v1,
        params={"steer_joint_name": "steering_joint", "drive_joint_name": "rear_wheel_joint"},
        noise=PerChannelGaussianNoiseCfg(
            std=(0.001, 0.001, 0.0012, 0.0012, 0.0012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class TrackstandRLRewardsCfgV1(TrackstandRLRewardsCfg):
    """Equal smoothing weights for the two already-normalized raw actions."""

    # ActionManager stores raw [-1, 1] actions, not rad / N m commands.
    # Do not divide their differences by physical motor scales a second time.
    rate_penalty = RewardTermCfg(func=balance_action_rate_penalty, weight=0.1)


@configclass
class TrackstandRLEnvCfgV1(TrackstandRLEnvCfg):
    """V0 plant with a Beta runner, 11 observations and constrained motor commands."""

    actions = TrackstandRLActionsCfgV1()
    observations = {"policy": TrackstandRLObservationsCfgV1()}
    rewards = TrackstandRLRewardsCfgV1()

    def __post_init__(self):
        super().__post_init__()
        # V1-only explicit PD: clip AFTER controller Kd, before passive joint losses.
        # The backend zeros native PD gains, avoiding a second controller in PhysX.
        self.scene.robot.actuators["steering"] = IdealPDActuatorCfg(
            joint_names_expr=["steering_joint"],
            stiffness=10.0,
            damping=0.2,  # Controller Kd, not the assembly's passive viscous friction.
            # Unloaded assembly fit: 20261020_180044 C1 sweep; fitted bias is not applied.
            friction=0.046036056104,  # N m: provisional static == dynamic; breakaway not measured.
            dynamic_friction=0.046036056104,  # N m: fitted moving Coulomb-like friction.
            viscous_friction=0.000910713492,  # N m s/rad: fitted passive resistance.
            effort_limit=5.6,  # Explicit post-PD motor torque clamp, N m.
            effort_limit_sim=5.6,
            velocity_limit=23.9,  # Stored parameter; IdealPD has no torque-speed envelope.
            velocity_limit_sim=23.9,  # Existing solver speed constraint, rad/s.
        )
        # V1-only replacement: the scene is an independent config instance.
        # With Kp = Kd = 0, compute at each physics step:
        # clip(6.28 * drive_action, -6.28, 6.28).
        # Passive joint friction is separate from the clipped actuator command.
        self.scene.robot.actuators["drive"] = IdealPDActuatorCfg(
            joint_names_expr=["rear_wheel_joint"],
            stiffness=0.0,
            damping=0.0,  # Pure feedforward torque control; no controller velocity damping.
            friction=0.106,  # N m: provisional static threshold (must be >= dynamic); not measured.
            dynamic_friction=0.106,  # N m: effective unloaded moving drag, 2026-10-20 speed sweep.
            effort_limit=6.28,  # Explicit actuator torque clamp, N m.
            effort_limit_sim=6.28,  # Retain the existing solver setting as well.
            velocity_limit=61.6,  # Explicit model parameter, matching the solver speed limit.
            velocity_limit_sim=61.6,  # Existing rear-wheel speed constraint, rad/s.
        )
