"""CycleRL locomotion environment for NewBike."""

import math

import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import DomeLightCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from ..bicycle_cfg import NEW_BIKE_STANDARD_CFG
from ..config.bicycle_terrain_stairs import BICYCLE_STAIRS_TERRAINS_CFG
from ..config.triangle_mesh_terrain_importer import TriangleMeshTerrainImporter
from ..mdp.dynamic import curriculums as custom_curriculums
from ..mdp.dynamic import events as custom_events
from ..mdp.dynamic import observations as custom_obs
from ..mdp.dynamic import rewards as custom_rewards
from ..mdp.dynamic import terminations as custom_terminations


def _setup_heading_visualization(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Draw frame markers at each robot, rotated to the heading command."""
    if hasattr(env, "_heading_vis"):
        return

    from isaaclab.markers.visualization_markers import VisualizationMarkers

    vis_cfg = FRAME_MARKER_CFG.copy()
    vis_cfg.prim_path = "/Visuals/Command/heading_target"
    vis_cfg.markers["frame"].scale = (0.3, 0.3, 0.3)
    env._heading_vis = VisualizationMarkers(vis_cfg)

    def _heading_debug_vis_callback(_event):
        steer_term = env.command_manager._terms["steering"]
        heading_target = steer_term.heading_target
        robot = env.scene["robot"]
        positions = robot.data.root_pos_w[:, :3].clone()
        positions[:, 2] += 0.5
        zeros = torch.zeros_like(heading_target)
        quats = math_utils.quat_from_euler_xyz(zeros, zeros, heading_target)
        env._heading_vis.visualize(positions, quats)

    # ponytail: cosmetic heading arrow; skip if this Isaac Lab build lacks the
    # registry API (the command manager's own debug_vis arrows still show).
    if hasattr(env.sim, "vis_marker_registry"):
        env.sim.vis_marker_registry.add_callback("heading_target", _heading_debug_vis_callback)


@configclass
class NewBikeSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        class_type=TriangleMeshTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=BICYCLE_STAIRS_TERRAINS_CFG,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        max_init_terrain_level=0,
        debug_vis=True,
    )
    robot = NEW_BIKE_STANDARD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class NewBikeActionsCfg:
    steering = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["steering_joint"],
        scale=math.radians(45.0),
    )
    drive = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["rearwheel_joint"],
        scale=45.0,
        clip={"rearwheel_joint": (-45.0, 45.0)},
    )


@configclass
class NewBikeObservationsCfg(ObservationGroupCfg):
    enable_corruption = True
    imu_state = ObservationTermCfg(
        func=custom_obs.imu_orientation_and_velocity,
        params={"steer_joint_name": "steering_joint"},
        noise=GaussianNoiseCfg(mean=0.0, std=0.02),
    )
    commands = ObservationTermCfg(func=custom_obs.command_tracking_error)


@configclass
class NewBikeRewardsCfg:
    survival = RewardTermCfg(func=custom_rewards.survival_reward, weight=1.0)
    velocity = RewardTermCfg(func=custom_rewards.velocity_tracking_reward, weight=5.0)
    steering = RewardTermCfg(
        func=custom_rewards.steering_tracking_reward,
        weight=5.0,
        params={"steer_joint_name": "steering_joint"},
    )
    act_mag = RewardTermCfg(func=custom_rewards.action_magnitude_penalty, weight=2.0)
    act_rate = RewardTermCfg(func=custom_rewards.action_rate_penalty, weight=2.0)


@configclass
class NewBikeTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_crash = TerminationTermCfg(
        func=custom_terminations.bicycle_lean_termination,
        params={"limit_angle": 0.785},
        time_out=False,
    )
    velocity_goal_timeout = TerminationTermCfg(
        func=custom_terminations.velocity_goal_timeout,
        params={"timeout_s": 5.0, "error_threshold": 0.5, "command_name": "velocity"},
        time_out=False,
    )


@configclass
class NewBikeCommandsCfg:
    velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.8, 2.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )
    steering = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ExternalForceRandCfg:
    """Random world-frame force pulses scaled to the bike's current weight ``m g``.

    ``force_weight_frac_range`` is the pulse magnitude as a fraction of weight
    (NewBike ≈ 2.2 kg → 21.6 N; 0.10–0.35 is about 2–8 N). Set ``enabled=False``
    or both force fractions to 0 to disable.
    """

    enabled: bool = True
    force_weight_frac_range: tuple[float, float] = (0.10, 0.35)
    torque_weight_frac_range: tuple[float, float] = (0.0, 0.0)
    duration_range_s: tuple[float, float] = (0.05, 0.20)
    interval_range_s: tuple[float, float] = (3.0, 8.0)
    elevation_range: tuple[float, float] = (-0.25, 0.35)
    torque_lever_m: float = 0.10
    gravity: float = 9.81
    is_global: bool = True

    def event_params(self) -> dict:
        return {
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "enabled": self.enabled,
            "force_weight_frac_range": self.force_weight_frac_range,
            "torque_weight_frac_range": self.torque_weight_frac_range,
            "duration_range_s": self.duration_range_s,
            "interval_range_s": self.interval_range_s,
            "elevation_range": self.elevation_range,
            "torque_lever_m": self.torque_lever_m,
            "gravity": self.gravity,
            "is_global": self.is_global,
        }


@configclass
class NewBikeEventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    setup_heading_vis = EventTermCfg(func=_setup_heading_visualization, mode="startup")
    randomize_ground_friction = EventTermCfg(
        func=custom_events.randomize_ground_plane_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("terrain"),
            "static_friction_range": (0.5, 1.1),
            "dynamic_friction_range": (0.5, 1.1),
            "restitution_range": (0.0, 0.0),
            "make_consistent": True,
        },
    )
    randomize_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "mass_distribution_params": (0.5, 1.5),
        },
    )
    randomize_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
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
    initial_state = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.139, 0.139),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.2, 3.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    wrap_root_at_terrain_border = EventTermCfg(
        func=custom_events.wrap_root_at_terrain_border,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={"asset_cfg": SceneEntityCfg("robot"), "inset": 0.4},
    )
    # # --- External forces / kickstand (disabled for now) ---
    # reset_kickstand_lift = EventTermCfg(
    #     func=custom_events.reset_kickstand_lift_schedule,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #         "deploy_speed": 1.2,
    #         "retract_speed": 1.5,
    #         "delay_range_s": (0.2, 1.0),
    #         "torque_range": (0.05, 0.12),
    #     },
    # )
    # kickstand_lift_wrench = EventTermCfg(
    #     func=custom_events.apply_kickstand_lift_wrench,
    #     mode="interval",
    #     interval_range_s=(0.02, 0.02),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #         "deploy_speed": 1.2,
    #         "retract_speed": 1.5,
    #         "lift_duration_s": 0.3,
    #         "torque_range": (0.05, 0.12),
    #         "swing_rad": 0.4363,
    #     },
    # )
    # reset_random_external_force = EventTermCfg(
    #     func=custom_events.reset_random_external_force,
    #     mode="reset",
    #     params=ExternalForceRandCfg().event_params(),
    # )
    # random_external_force = EventTermCfg(
    #     func=custom_events.apply_random_external_force,
    #     mode="interval",
    #     interval_range_s=(0.02, 0.02),
    #     params=ExternalForceRandCfg().event_params(),
    # )


@configclass
class NewBikeCurriculumsCfg:
    """Spawn speed, spawn yaw, heading, IMU noise, and stairs difficulty."""

    # expand_spawn_vel = CurriculumTermCfg(
    #     func=custom_curriculums.expand_curriculum_range,
    #     params={
    #         "step_thresholds": [0, 10000, 20000],
    #         "max_values": [0.5, 2.5, 3.0],
    #         "target_param": "spawn_velocity_x",
    #     },
    # )
    # expand_spawn_yaw = CurriculumTermCfg(
    #     func=custom_curriculums.expand_curriculum_range,
    #     params={
    #         "step_thresholds": [0, 10000, 30000],
    #         "max_values": [0.0, 0.5, 0.7854],
    #         "target_param": "spawn_yaw",
    #     },
    # )
    expand_cmd_heading = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 10000, 20000],
            "max_values": [0.174, 0.7854, 1.04],
            "target_param": "command_heading",
        },
    )
    expand_imu_noise = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 10000, 50000],
            "max_values": [0.0, 0.01, 0.02],
            "target_param": "imu_noise_std",
        },
    )
    terrain_levels = CurriculumTermCfg(func=custom_curriculums.terrain_levels_vel)


@configclass
class NewBikeEnvCfg(ManagerBasedRLEnvCfg):
    decimation = 2
    scene = NewBikeSceneCfg(num_envs=16384, env_spacing=2.5)
    actions = NewBikeActionsCfg()
    observations = {"policy": NewBikeObservationsCfg()}
    rewards = NewBikeRewardsCfg()
    terminations = NewBikeTerminationsCfg()
    commands = NewBikeCommandsCfg()
    events = NewBikeEventsCfg()
    curriculum = NewBikeCurriculumsCfg()
    # external_force: ExternalForceRandCfg = ExternalForceRandCfg()

    def __post_init__(self):
        self.viewer.eye = [5.0, 5.0, 2.0]
        self.step_dt = 0.02
        self.episode_length_s = 64.0
        # inherit terrain physics material (high-friction for bicycle tyres)
        self.sim.physics_material = self.scene.terrain.physics_material
        # enable terrain difficulty curriculum
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        # Start heading / spawn at the first curriculum values (expand from here).
        self.commands.steering.ranges.heading = (-0.174, 0.174)
        self.events.initial_state.params["velocity_range"]["x"] = (0.5, 0.5)
        self.events.initial_state.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.observations["policy"].imu_state.noise.std = 0.0
        # force_params = self.external_force.event_params()
        # self.events.reset_random_external_force.params = force_params
        # self.events.random_external_force.params = force_params
        # self.events.random_external_force.interval_range_s = (self.step_dt, self.step_dt)
        self.events.wrap_root_at_terrain_border.interval_range_s = (self.step_dt, self.step_dt)
