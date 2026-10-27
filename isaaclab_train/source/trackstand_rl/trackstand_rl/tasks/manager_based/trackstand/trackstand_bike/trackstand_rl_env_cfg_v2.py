"""V2: v1 with a rear-wheel velocity action and the nominal MIT velocity law.

Rear Beta action [-1, 1] * 61.6 rad/s -> velocity target held at 50 Hz.
IdealPD computes clip(0.2 * (target_speed - measured_speed), +/-6.28 N m)
at 400 Hz, with Kp=0 and feedforward torque=0. Passive joint friction remains
separate. This models the requested MIT equation; it does not send CAN or
identify the physical driver's internal timing/current dynamics.
Controller gains stay fixed even with DR enabled. Other task settings inherit
v1, including 11 observations, steering, and the remaining DR terms.
"""

from isaaclab.envs.mdp.actions import JointVelocityActionCfg
from isaaclab.utils.configclass import configclass

from .trackstand_rl_env_cfg_v1 import TrackstandRLActionsCfgV1, TrackstandRLEnvCfgV1


# Match v1's rear-wheel speed limit. The individual GIM3510-8 datasheet lists
# 588 output rpm = 61.5752 rad/s, rounded to 61.6 in bicycle_cfg.py.
DRIVE_VELOCITY_SCALE_RAD_S = 61.6
DRIVE_MIT_KD_NM_S_PER_RAD = 0.2


@configclass
class TrackstandRLActionsCfgV2(TrackstandRLActionsCfgV1):
    """Keep v1 steering; change only the rear action from effort to velocity."""

    drive = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["rear_wheel_joint"],
        scale=DRIVE_VELOCITY_SCALE_RAD_S,
        clip={"rear_wheel_joint": (-DRIVE_VELOCITY_SCALE_RAD_S, DRIVE_VELOCITY_SCALE_RAD_S)},
        use_default_offset=False,
    )


@configclass
class TrackstandRLEnvCfgV2(TrackstandRLEnvCfgV1):
    """V1 plant/task with a torque-limited rear proportional velocity servo."""

    actions = TrackstandRLActionsCfgV2()

    def set_domain_randomization(self, enabled: bool) -> None:
        """Keep v1's other DR terms, but never randomize v2 controller gains."""
        super().set_domain_randomization(enabled)
        # Apply after the parent switch, which restores this event on --dr.
        # Covers construction and repeated pre-construction CLI/config switches.
        self.events.randomize_actuator_gains = None

    def __post_init__(self):
        super().__post_init__()
        # Retain all v1 torque/speed limits and separately configured passive losses.
        # IdealPD clips AFTER Kd * velocity error. Its native PhysX PD stays zero.
        self.scene.robot.actuators["drive"] = self.scene.robot.actuators["drive"].replace(
            stiffness=0.0,
            damping=DRIVE_MIT_KD_NM_S_PER_RAD,
        )
