import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    ActionTermCfg,
    TerminationTermCfg,
    EventTermCfg,
    SceneEntityCfg,
    CommandTermCfg,
    CurriculumTermCfg
)
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import DomeLightCfg, RigidBodyMaterialCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from ..bicycle_cfg import THUBIKE_STANDARD_CFG
from ..config.bicycle_terrain_complex import BICYCLE_TERRAINS_CFG
from ..config.triangle_mesh_terrain_importer import TriangleMeshTerrainImporter
from ..mdp.dynamic import observations as custom_obs
from ..mdp.dynamic import rewards as custom_rewards
from ..mdp.dynamic import terminations as custom_terminations
from ..mdp.dynamic import events as custom_events
from ..mdp.dynamic import curriculums as custom_curriculums


# ---------------------------------------------------------------------------
# Heading target visualization (startup event)
# ---------------------------------------------------------------------------

def _setup_heading_visualization(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """Register a render callback that draws frame markers at each robot's
    position rotated to the steering command's heading target.

    A red/green/blue XYZ triad appears above each bike.  The X-axis (red)
    points in the *target* heading direction.  The bike should turn to align
    its forward vector with that axis.
    """
    # ── guard against double-registration ──────────────────────────────
    if hasattr(env, "_heading_vis"):
        return

    # ── create the frame markers ───────────────────────────────────────
    from isaaclab.markers.visualization_markers import VisualizationMarkers

    vis_cfg = FRAME_MARKER_CFG.copy()
    vis_cfg.prim_path = "/Visuals/Command/heading_target"
    # Scale frames up slightly for a bicycle-sized robot
    vis_cfg.markers["frame"].scale = (0.3, 0.3, 0.3)
    env._heading_vis = VisualizationMarkers(vis_cfg)

    # ── define the per-render-tick callback ─────────────────────────────
    def _heading_debug_vis_callback(_event):
        # read the raw heading target buffer from the steering command term
        steer_term = env.command_manager._terms["steering"]
        heading_target = steer_term.heading_target  # (num_envs,) radians, world frame
        robot = env.scene["robot"]
        # place markers 0.5 m above each robot's root
        positions = robot.data.root_pos_w[:, :3].clone()
        positions[:, 2] += 0.5
        # convert heading (yaw) to quaternion
        zeros = torch.zeros_like(heading_target)
        quats = math_utils.quat_from_euler_xyz(zeros, zeros, heading_target)
        env._heading_vis.visualize(positions, quats)

    # ── register with the simulation's vis-marker registry ──────────────
    # ponytail: cosmetic heading arrow; skip if this Isaac Lab build lacks the
    # registry API (the command manager's own debug_vis arrows still show).
    if hasattr(env.sim, "vis_marker_registry"):
        env.sim.vis_marker_registry.add_callback(
            "heading_target", _heading_debug_vis_callback
        )


@configclass
class THUBikeSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        class_type=TriangleMeshTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=BICYCLE_TERRAINS_CFG,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        max_init_terrain_level=4,  # start at hardest row
        debug_vis=True,
    )
    robot = THUBIKE_STANDARD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

@configclass
class THUBikeActionsCfg:
    steering = JointPositionActionCfg(asset_name="robot", joint_names=["Revolute_9"], scale=0.785)  # ±45°
    drive = JointVelocityActionCfg(asset_name="robot", joint_names=["Revolute_8"], scale=20.0)

@configclass
class THUBikeObservationsCfg(ObservationGroupCfg):
    enable_corruption = True
    imu_state = ObservationTermCfg(
        func=custom_obs.imu_orientation_and_velocity,
        noise=GaussianNoiseCfg(mean=0.0, std=0.02),
    )
    commands = ObservationTermCfg(func=custom_obs.command_tracking_error)

@configclass
class THUBikeRewardsCfg:
    survival = RewardTermCfg(func=custom_rewards.survival_reward, weight=1.0)
    velocity = RewardTermCfg(func=custom_rewards.velocity_tracking_reward, weight=5.0)
    steering = RewardTermCfg(func=custom_rewards.steering_tracking_reward, weight=5.0)
    act_mag = RewardTermCfg(func=custom_rewards.action_magnitude_penalty, weight=0.1)
    act_rate = RewardTermCfg(func=custom_rewards.action_rate_penalty, weight=2.0)
    trackstand_drift = RewardTermCfg(
        func=custom_rewards.trackstand_stationary_penalty,
        weight=-0.5,
        params={"command_name": "velocity", "vel_threshold": 0.01},
    )

@configclass
class THUBikeTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_crash = TerminationTermCfg(
        func=custom_terminations.bicycle_lean_termination,
        params={"limit_angle": 0.785},
        time_out=False
        )

@configclass
class THUBikeCommandsCfg:
    # 1. Forward Velocity Command (X-axis only)
    velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.8, 3.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )

    # 2. Randomized heading: sample a world-frame yaw target and convert it
    #    to a yaw-rate command ω_z = clip(k * wrap_to_pi(ψ* − ψ), ±0.5).
    #    Resampled every 3–5 s. ±π covers left/right/U-turn; ω_z cap keeps
    #    bicycle lean rates reasonable (~R ≥ v/0.5 at 1–3 m/s).
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
class THUBikeEventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

    # Heading target frame-marker visualization (once at startup)
    setup_heading_vis = EventTermCfg(func=_setup_heading_visualization, mode="startup")

    # 2. ADD YOUR CUSTOM GROUND RANDOMIZATION HERE
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
            "com_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
            },
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
                "x": (0.2, 1.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    # Wrap on the flat apron so PhysX never applies wall contact impulses.
    wrap_root_at_terrain_border = EventTermCfg(
        func=custom_events.wrap_root_at_terrain_border,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "inset": 0.4,
        },
    )

# ---------------------------------------------------------
# ADDED CURRICULUMS CONFIG CLASS HERE
# ---------------------------------------------------------
@configclass
class THUBikeCurriculumsCfg:

    # Gradually decrease how many envs trackstand (command = 0) vs. drive
    fade_trackstand = CurriculumTermCfg(
        func=custom_curriculums.fade_trackstand_ratio,
        params={
            "step_thresholds": [0, 5000, 10000, 20000],
            "ratios": [1.0, 0.75, 0.25, 0.0]
        }
    )

    # Gradually expand the initial spawn velocity range
    expand_spawn_vel = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 10000, 20000],
            "max_values": [0.5, 1.5, 2.5],
            "target_param": "spawn_velocity_x"
        }
    )

    # Gradually expand the initial spawn yaw randomization
    expand_spawn_yaw = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 10000, 20000],
            "max_values": [0.0, 0.5, 3.14],
            "target_param": "spawn_yaw"
        }
    )

    # Gradually expand the range of commanded steering/heading angles
    expand_cmd_heading = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 10000, 20000],
            "max_values": [0.174, 1.0, 3.14],
            "target_param": "command_heading"
        }
    )

    # Gradually increase policy IMU Gaussian corruption
    expand_imu_noise = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 10000, 20000],
            "max_values": [0.0, 0.02, 0.05],
            "target_param": "imu_noise_std",
        },
    )


@configclass
class THUBikeEnvCfg(ManagerBasedRLEnvCfg):
    decimation = 2
    scene = THUBikeSceneCfg(num_envs=16384, env_spacing=2.5)
    actions = THUBikeActionsCfg()
    observations = {"policy": THUBikeObservationsCfg()}
    rewards = THUBikeRewardsCfg()
    terminations = THUBikeTerminationsCfg()
    commands = THUBikeCommandsCfg()
    events = THUBikeEventsCfg()
    # curriculum = THUBikeCurriculumsCfg()  # disabled: learn balancing while moving first

    def __post_init__(self):
        self.viewer.eye = [5.0, 5.0, 2.0]
        self.step_dt = 0.02
        self.episode_length_s = 64.0
        # inherit terrain physics material (high-friction for bicycle tyres)
        self.sim.physics_material = self.scene.terrain.physics_material
        # enable terrain difficulty curriculum
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        self.events.wrap_root_at_terrain_border.interval_range_s = (self.step_dt, self.step_dt)

