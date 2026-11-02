"""V2: v1 with a rear-wheel velocity action and external integral torque.

Rear Beta action [-1, 1] * 61.6 rad/s -> velocity target held at 50 Hz.
External I integrates velocity error at 200 Hz with Ki=3.3 Nm/rad; IdealPD
computes clip(0.30 * (target_speed - measured_speed) + I, +/-6.28 Nm) at each
1200-Hz physics step. Position Kp=0. Episode reset clears I; target changes retain it.
Passive joint friction remains separate. This models the tested MIT P/host-I
law, not CAN quantization, async feedback or identified firmware dynamics.
Controller gains stay fixed even with DR enabled. Other task settings inherit
v1, including 11 observations, steering, and the remaining DR terms. V2 also
contains invalid pose/joint/PI state and non-finite terminal drift rewards.
"""

from isaaclab.utils.configclass import configclass

from ..mdp.stationary.numerics import finite_balance_drift_gaussian, nonfinite_or_extreme_state_v2
from ..mdp.stationary.velocity_pi import IntegralJointVelocityActionCfg
from .trackstand_rl_env_cfg_v1 import TrackstandRLActionsCfgV1, TrackstandRLEnvCfgV1


# Match v1's rear-wheel speed limit. The individual GIM3510-8 datasheet lists
# 588 output rpm = 61.5752 rad/s, rounded to 61.6 in bicycle_cfg.py.
DRIVE_VELOCITY_SCALE_RAD_S = 61.6
DRIVE_MIT_KD_NM_S_PER_RAD = 0.30
DRIVE_VELOCITY_KI_NM_PER_RAD = 3.3
DRIVE_INTEGRAL_RATE_HZ = 200.0
DRIVE_PI_PHYSICS_RATE_HZ = 1200.0


@configclass
class TrackstandRLActionsCfgV2(TrackstandRLActionsCfgV1):
    """Keep v1 steering; change only the rear action from effort to velocity."""

    drive = IntegralJointVelocityActionCfg(
        asset_name="robot",
        joint_names=["rear_wheel_joint"],
        scale=DRIVE_VELOCITY_SCALE_RAD_S,
        clip={"rear_wheel_joint": (-DRIVE_VELOCITY_SCALE_RAD_S, DRIVE_VELOCITY_SCALE_RAD_S)},
        use_default_offset=False,
        integral_gain_nm_per_rad=DRIVE_VELOCITY_KI_NM_PER_RAD,
        integral_rate_hz=DRIVE_INTEGRAL_RATE_HZ,
    )


@configclass
class TrackstandRLEnvCfgV2(TrackstandRLEnvCfgV1):
    """V1 plant/task with a torque-limited rear PI velocity servo."""

    actions = TrackstandRLActionsCfgV2()

    def set_domain_randomization(self, enabled: bool) -> None:
        """Keep v1's other DR terms, but never randomize v2 controller gains."""
        super().set_domain_randomization(enabled)
        # Apply after the parent switch, which restores this event on --dr.
        # Covers construction and repeated pre-construction CLI/config switches.
        self.events.randomize_actuator_gains = None

    def __post_init__(self):
        super().__post_init__()
        # The unloaded P=0.30 loop is numerically unstable at the inherited
        # 400 Hz. A 1200-Hz step passes the measured-inertia/DR live probe;
        # policy/steering target updates remain 50 Hz and external I 200 Hz.
        policy_dt = self.sim.dt * self.decimation
        self.sim.dt = 1.0 / DRIVE_PI_PHYSICS_RATE_HZ
        self.decimation = round(policy_dt / self.sim.dt)
        self.sim.render_interval = self.decimation
        # Retain all v1 torque/speed limits and separately configured passive losses.
        # IdealPD clips AFTER Kd * velocity error. Its native PhysX PD stays zero.
        self.scene.robot.actuators["drive"] = self.scene.robot.actuators["drive"].replace(
            stiffness=0.0,
            damping=DRIVE_MIT_KD_NM_S_PER_RAD,
        )
        # V2-only containment: pose/joint/PI NaNs must reset, and a dying env's
        # invalid position must not poison rewards before that reset occurs.
        self.terminations.diverged.func = nonfinite_or_extreme_state_v2
        self.rewards.drift_gauss.func = finite_balance_drift_gaussian
