"""Locomotive riding task: CycleRL cylinder paper recipe on trackstand_rl's motors and physics.

Flattened from ~/bike_test/CycleRL ``CycleRL-Trackstand-Cylinder-Paper-Aligned-v0``
(cyclerl_env_cfg_paper.py -> _paper_cylinder.py -> _paper_cylinder_aligned.py) with
the rank-1 ``1900_drive03`` recipe from scripts/skrl/train_cylinder_top5_rerun.sh
baked in as defaults. Other top-5 recipes are hydra ``env.*`` overrides on this cfg.

Tensor contract (unchanged from CycleRL): 8 obs [gravity_y, ang_vel_x/y/z, speed,
steer, vel_cmd, yaw_rate_cmd], 2 raw actions (steer position x20 deg, rear velocity
x3/0.07 rad/s), 50 Hz.

Robot, joints and physics are trackstand_rl's (``trackstand/bicycle_cfg.py``,
``trackstand_rl_env_cfg.py``, ``trackstand_rl_env_cfg_v1/v2.py``): same USD (tire
material 0.65/min authored), solver 4/2, on-ground spawn, 400 Hz physics with external
forces every iteration, v1 steering IdealPD and v2 rear IdealPD (Kd 0.2) with measured
losses. CycleRL's riding DR, rewards, commands and terminations are kept.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import DomeLightCfg, PhysxRigidBodyMaterialCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from ..trackstand.bicycle_cfg import TRACKSTAND_ROBOT_CFG
from .mdp import events as loco_events
from .mdp import observations as loco_obs
from .mdp import rewards as loco_rewards
from .mdp import terminations as loco_terminations


_WHEEL_RADIUS_M = 0.07  # Cylinder tire; drive scale = 3 m/s / r.
_STEER_SCALE_RAD = math.radians(20.0)
_DRIVE_SCALE_RAD_S = 3.0 / _WHEEL_RADIUS_M
_DRIVE_CLIP_RAD_S = 45.0
_HEADING_RANGE_RAD = (-math.radians(10.0), math.radians(10.0))
_ROLL_INIT_RAD = (-math.radians(10.0), math.radians(10.0))
_PHYSICS_DT_S = 0.0025
_DECIMATION = 8
_DRIVE_MIT_KD_NM_S_PER_RAD = 0.2  # trackstand_rl v2 rear velocity gain.

# trackstand_rl_env_cfg_v1.py steering, verbatim.
_STEERING_ACTUATOR = IdealPDActuatorCfg(
    joint_names_expr=["steering_joint"],
    stiffness=10.0,
    damping=0.2,  # Controller Kd, not the assembly's passive viscous friction.
    friction=0.046036056104,  # N m: provisional static == dynamic; breakaway not measured.
    dynamic_friction=0.046036056104,
    viscous_friction=0.000910713492,  # N m s/rad, 20261020_180044 C1 sweep fit.
    effort_limit=5.6,  # Explicit post-PD motor torque clamp, N m.
    effort_limit_sim=5.6,
    velocity_limit=23.9,
    velocity_limit_sim=23.9,
)

# trackstand_rl v1 rear plant with v2's velocity gain: tau = clip(0.2 * (w* - w), +/-6.28).
_DRIVE_ACTUATOR = IdealPDActuatorCfg(
    joint_names_expr=["rear_wheel_joint"],
    stiffness=0.0,
    damping=_DRIVE_MIT_KD_NM_S_PER_RAD,  # Velocity-error gain on the JointVelocityAction target.
    friction=0.106,  # N m: provisional static threshold (>= dynamic); not measured.
    dynamic_friction=0.106,  # N m: unloaded moving drag, 2026-10-20 speed sweep.
    viscous_friction=0.0,
    effort_limit=6.28,  # Explicit actuator torque clamp, N m (GIM3510-8 stall).
    effort_limit_sim=6.28,
    velocity_limit=61.6,  # 588 rpm output.
    velocity_limit_sim=61.6,
)


@configclass
class LocomotiveSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )
    # Trackstand robot: same USD/joints, solver 4/2, z = -0.016; only the actuators differ.
    robot = TRACKSTAND_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={"steering": _STEERING_ACTUATOR, "drive": _DRIVE_ACTUATOR},
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class LocomotiveActionsCfg:
    """Steer position (+/-20 deg) and rear-hub joint velocity (+/-45 rad/s)."""

    steering = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["steering_joint"],
        scale=_STEER_SCALE_RAD,
        clip={"steering_joint": (-_STEER_SCALE_RAD, _STEER_SCALE_RAD)},
    )
    drive = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["rear_wheel_joint"],
        scale=_DRIVE_SCALE_RAD_S,
        clip={"rear_wheel_joint": (-_DRIVE_CLIP_RAD_S, _DRIVE_CLIP_RAD_S)},
    )


@configclass
class LocomotiveObservationsCfg(ObservationGroupCfg):
    """8-D policy input: 6-D IMU/encoder + 2-D commands (noise std set per reset)."""

    enable_corruption = True
    imu_state = ObservationTermCfg(
        func=loco_obs.imu_orientation_and_velocity,
        params={"steer_joint_name": "steering_joint"},
        noise=GaussianNoiseCfg(mean=0.0, std=0.0),
    )
    commands = ObservationTermCfg(
        func=loco_obs.command_tracking_error,
        params={"use_vel_error": False},
        noise=GaussianNoiseCfg(mean=0.0, std=0.0),
    )


@configclass
class LocomotiveRewardsCfg:
    """Paper lambda = (1, 3, 5, 1, 2); rank-1 1900_drive03 velocity/action weights."""

    survival = RewardTermCfg(func=loco_rewards.survival_reward, weight=1.0)
    velocity = RewardTermCfg(
        func=loco_rewards.velocity_tracking_reward,
        weight=3.0,
        params={"alpha": 4.0, "squared": True, "tolerance": 0.0},
    )
    steering = RewardTermCfg(
        func=loco_rewards.heading_tracking_reward,
        weight=5.0,
        params={"command_name": "steering", "beta": 0.1},
    )
    act_mag = RewardTermCfg(
        func=loco_rewards.weighted_action_magnitude_penalty, weight=1.0, params={"weights": [1.0, 0.3]},
    )
    act_rate = RewardTermCfg(
        func=loco_rewards.weighted_action_rate_penalty, weight=2.0, params={"weights": [1.0, 1.0]},
    )


@configclass
class LocomotiveTerminationsCfg:
    """64 s timeout; crash when |roll| > 45 deg."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_crash = TerminationTermCfg(
        func=loco_terminations.bicycle_lean_termination,
        params={"limit_angle": math.radians(45.0)},
        time_out=False,
    )


@configclass
class LocomotiveCommandsCfg:
    """v* ~ U(0.5, 2.5) m/s; heading* ~ U(+/-10 deg) via P-control (k 0.5, clip 0.5 rad/s)."""

    velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,  # CycleRL had True: viz markers only, no effect on training.
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.5, 2.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
    )
    steering = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=_HEADING_RANGE_RAD,
        ),
    )


@configclass
class LocomotiveEventsCfg:
    """CycleRL Table I riding DR as trained by the cylinder top-5 (zero_lateral_com off)."""

    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    randomize_ground_friction = EventTermCfg(
        func=loco_events.randomize_ground_plane_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("terrain"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.0),
            "make_consistent": True,
        },
    )
    randomize_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "operation": "scale", "mass_distribution_params": (0.6, 1.8)},
    )
    randomize_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (-0.015, 0.015)},
        },
    )
    randomize_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    randomize_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    randomize_imu_noise = EventTermCfg(
        func=loco_events.randomize_imu_noise_fraction,
        mode="reset",
        params={"rho_range": (0.01, 0.20), "imu_smax": 0.02},
    )
    initial_state = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                           "roll": _ROLL_INIT_RAD, "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (1.0, 2.5), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_joint_state = EventTermCfg(
        func=loco_events.reset_paper_joint_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "steer_joint_name": "steering_joint",
            "drive_joint_name": "rear_wheel_joint",
            "steer_deg_range": (-20.0, 20.0),
            "drive_vel_range": (0.0, 3.0),
            "couple_drive_to_root_vx": False,
        },
    )


@configclass
class LocomotiveEnvCfg(ManagerBasedRLEnvCfg):
    """CycleRL riding recipe (rank-1 1900_drive03) on trackstand_rl's plant."""

    decimation = _DECIMATION
    scene = LocomotiveSceneCfg(num_envs=4096, env_spacing=2.5)
    actions = LocomotiveActionsCfg()
    observations = {"policy": LocomotiveObservationsCfg()}
    rewards = LocomotiveRewardsCfg()
    terminations = LocomotiveTerminationsCfg()
    commands = LocomotiveCommandsCfg()
    events = LocomotiveEventsCfg()
    curriculum = None

    def __post_init__(self):
        self.viewer.eye = [5.0, 5.0, 2.0]
        # Set sim.dt + decimation; env.step_dt is derived (0.0025 * 8 = 0.02 s).
        self.sim.dt = _PHYSICS_DT_S
        self.sim.render_interval = self.decimation
        self.episode_length_s = 64.0  # 3,200 steps @ 50 Hz
        # trackstand_rl_env_cfg.py: robot shapes without a USD material use 0.65/min;
        # the ground keeps its own bound material (randomized at startup above).
        self.sim.physics_material = PhysxRigidBodyMaterialCfg(
            static_friction=0.65, dynamic_friction=0.65, restitution=0.0, friction_combine_mode="min",
        )
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

        self.sim.physics = PhysxCfg(enable_external_forces_every_iteration=True)
        h0, h1 = self.commands.steering.ranges.heading
        assert h0 <= 0.0 <= h1, f"heading range must include 0, got {(h0, h1)}"
