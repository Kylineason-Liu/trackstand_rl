"""V1: 11 observations, bounded/slewed steering and explicitly clipped drive torque.

Order: gravity_y/z, body angular velocity xyz, steering/rear position (rad),
steering/rear velocity (rad/s), sin(steering), cos(steering). No body linear
velocities. Both the observation function and the matching noise configuration
are independent of v0's 13-channel observation term. The steering action clips
targets to +/-1.57 rad and slews at 23.9 rad/s, once per 50-Hz policy update.
The rear IdealPDActuator clips feedforward plus damping to +/-6.28 N m.
No reward, termination, reset-event, timing, or DR settings change.
"""

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils.configclass import configclass

from ..mdp.stationary.actions import SlewLimitedJointPositionActionCfg
from ..mdp.stationary.noise import PerChannelGaussianNoiseCfg
from ..mdp.stationary.observations import balance_imu_state_v1
from .trackstand_rl_env_cfg import TrackstandRLActionsCfg, TrackstandRLEnvCfg


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
class TrackstandRLEnvCfgV1(TrackstandRLEnvCfg):
    """V0 plant with a Beta runner, 11 observations and constrained motor commands."""

    actions = TrackstandRLActionsCfgV1()
    observations = {"policy": TrackstandRLObservationsCfgV1()}

    def __post_init__(self):
        super().__post_init__()
        # State resolved limits explicitly in v1 without changing the shared v0 config.
        self.scene.robot.actuators["steering"].effort_limit = 5.6
        self.scene.robot.actuators["steering"].velocity_limit = 23.9
        # V1-only replacement: the scene is an independent config instance.
        # With the inherited zero velocity target, compute at each physics step:
        # clip(6.28 * drive_action - damping * rear_speed, -6.28, 6.28).
        # The explicit actuator path disables native PD gains (no double damping).
        self.scene.robot.actuators["drive"] = IdealPDActuatorCfg(
            joint_names_expr=["rear_wheel_joint"],
            stiffness=0.0,
            damping=0.05,  # PLACEHOLDER controller Kd, N m s/rad; not measured passive friction.
            effort_limit=6.28,  # Explicit clamp AFTER adding damping, N m.
            effort_limit_sim=6.28,  # Retain the existing solver setting as well.
            velocity_limit=61.6,  # Explicit model parameter, matching the solver speed limit.
            velocity_limit_sim=61.6,  # Existing rear-wheel speed constraint, rad/s.
        )
