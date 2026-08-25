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
    CurriculumTermCfg  # <-- Added CurriculumTermCfg
)
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import DomeLightCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from ..bicycle_cfg import XUAN_BIKE_STANDARD_CFG
from ..mdp.dynamic import observations as custom_obs
from ..mdp.dynamic import rewards as custom_rewards
from ..mdp.dynamic import terminations as custom_terminations
from ..mdp.dynamic import events as custom_events
from ..mdp.dynamic import curriculums as custom_curriculums


@configclass
class CycleRLSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = XUAN_BIKE_STANDARD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

@configclass
class CycleRLActionsCfg:
    steering = JointPositionActionCfg(asset_name="robot", joint_names=["Revolute_7"], scale=1.0)
    drive = JointVelocityActionCfg(asset_name="robot", joint_names=["Revolute_9"], scale=20.0)

@configclass
class CycleRLObservationsCfg(ObservationGroupCfg):
    enable_corruption = True
    imu_state = ObservationTermCfg(
        func=custom_obs.imu_orientation_and_velocity,
        noise=GaussianNoiseCfg(mean=0.0, std=0.02),
    )
    commands = ObservationTermCfg(func=custom_obs.command_tracking_error)

@configclass
class CycleRLRewardsCfg:
    survival = RewardTermCfg(func=custom_rewards.survival_reward, weight=1.0)
    velocity = RewardTermCfg(func=custom_rewards.velocity_tracking_reward, weight=5.0)
    steering = RewardTermCfg(func=custom_rewards.steering_tracking_reward, weight=5.0)
    act_mag = RewardTermCfg(func=custom_rewards.action_magnitude_penalty, weight=1.0)
    act_rate = RewardTermCfg(func=custom_rewards.action_rate_penalty, weight=2.0)

@configclass
class CycleRLTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_crash = TerminationTermCfg(
        func=custom_terminations.bicycle_lean_termination, 
        params={"limit_angle": 0.785}, 
        time_out=False
        )

@configclass
class CycleRLCommandsCfg:
    # 1. Forward Velocity Command (X-axis only)
    velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,  
        debug_vis=True,         
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 5.0),   
            lin_vel_y=(0.0, 0.0), 
            ang_vel_z=(0.0, 0.0),   
        ),
    )

    # 2. Randomized heading: sample a world-frame yaw target and convert it
    #    to a yaw-rate command ω_z = clip(k * wrap_to_pi(ψ* − ψ), ±0.5).
    steering = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,       
        heading_control_stiffness=0.5,
        debug_vis=False,            
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),   
            lin_vel_y=(0.0, 0.0), 
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )
    
@configclass
class CycleRLEventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    
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
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (-0.15, 0.15),
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
                "roll": (-0.17453292519943295, 0.17453292519943295),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (1.0, 2.5),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

# ---------------------------------------------------------
# ADDED CURRICULUMS CONFIG CLASS HERE
# ---------------------------------------------------------
@configclass
class CycleRLCurriculumsCfg:
    
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
            "step_thresholds": [0, 5000, 10000], 
            "max_values": [0.5, 1.5, 2.5],
            "target_param": "spawn_velocity_x"
        }
    )

    # Gradually expand the initial spawn yaw randomization
    expand_spawn_yaw = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 5000, 10000],  
            "max_values": [0.0, 0.5, 3.14],
            "target_param": "spawn_yaw"
        }
    )

    # Gradually expand the range of commanded steering/heading angles
    expand_cmd_heading = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 5000, 10000], 
            "max_values": [0.174, 1.0, 3.14],   
            "target_param": "command_heading"
        }
    )

    expand_imu_noise = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 5000, 10000],
            "max_values": [0.0, 0.02, 0.05],
            "target_param": "imu_noise_std",
        },
    )


@configclass
class CycleRLEnvCfg(ManagerBasedRLEnvCfg):
    decimation = 2  
    scene = CycleRLSceneCfg(num_envs=16384, env_spacing=2.5)
    actions = CycleRLActionsCfg()
    observations = {"policy": CycleRLObservationsCfg()}
    rewards = CycleRLRewardsCfg()
    terminations = CycleRLTerminationsCfg()
    commands = CycleRLCommandsCfg()
    events = CycleRLEventsCfg()
    curriculum = CycleRLCurriculumsCfg() # <-- Instantiated your curriculums class here

    def __post_init__(self):
        self.viewer.eye = [5.0, 5.0, 2.0]
        self.step_dt = 0.02 
        self.episode_length_s = 64.0