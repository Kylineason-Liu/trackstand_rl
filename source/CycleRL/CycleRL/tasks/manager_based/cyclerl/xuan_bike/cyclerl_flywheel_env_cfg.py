import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg, JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
    EventTermCfg,
    CurriculumTermCfg
)
from isaaclab.utils.configclass import configclass

# Import your base components
from .cyclerl_env_cfg import (
    CycleRLEnvCfg,
    CycleRLActionsCfg,
    CycleRLObservationsCfg,
    CycleRLCommandsCfg,
    CycleRLRewardsCfg,
    CycleRLTerminationsCfg,
    CycleRLSceneCfg,
    CycleRLEventsCfg
)
from ..bicycle_cfg import XUAN_BIKE_FLYWHEEL_CFG

# Import your MDP module functions!
from ..mdp.dynamic import rewards as custom_rewards
from ..mdp.dynamic import terminations as custom_terminations
from ..mdp.dynamic import curriculums as custom_curriculums

JOINT_STEER = "Revolute_7"
JOINT_DRIVE = "Revolute_9"
JOINT_FLYWHEEL = "Revolute_11"

# =========================================================================
# Environment Configuration Sub-Classes
# =========================================================================

@configclass
class FlywheelSceneCfg(CycleRLSceneCfg):
    """Override the base scene to spawn the Flywheel-actuated robot."""
    robot = XUAN_BIKE_FLYWHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class FlywheelActionsCfg(CycleRLActionsCfg):
    # Inherits `steering` and `drive` automatically!
    steering = JointPositionActionCfg(asset_name="robot", joint_names=[JOINT_STEER], scale=1.0)
    drive = JointVelocityActionCfg(asset_name="robot", joint_names=[JOINT_DRIVE], scale=0.0)

    flywheel = JointEffortActionCfg(
        asset_name="robot",
        joint_names=[JOINT_FLYWHEEL], 
        scale=5.0 
    )
    
@configclass
class FlywheelCommandsCfg(CycleRLCommandsCfg):
    # 1. Forward Velocity Command
    velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        # 20% of environments will be forced to 0.0 m/s
        rel_standing_envs=0.2, 
        rel_heading_envs=1.0,
        heading_command=False,  
        debug_vis=True,         
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 3.0),   # Range updated to include 0.0
            lin_vel_y=(0.0, 0.0), 
            ang_vel_z=(0.0, 0.0),   
        ),
    )

    # 2. Steering Command
    steering = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        # CRITICAL: We MUST also apply rel_standing_envs=0.2 here so the 
        # steering target also drops to 0.0 during the trackstand!
        rel_standing_envs=0.8,
        rel_heading_envs=1.0,
        heading_command=True,       
        heading_control_stiffness=0.5,
        debug_vis=False,            
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),   
            lin_vel_y=(0.0, 0.0), 
            # Expanded the turning range so it learns how to lean properly!
            ang_vel_z=(-1.0, 1.0), 
            heading=(-1.0, 1.0),
        ),
    )

@configclass
class FlywheelEventsCfg(CycleRLEventsCfg):
    """Override the base events to spawn the robot stationary."""
    initial_state = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                # Keep the +/- 10 degree roll! The agent needs to spawn slightly 
                # off-balance so it learns to "catch" itself immediately.
                "roll": (-0.0872665, 0.0872665),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                # Spawns anywhere from perfectly stationary (0.0) up to 2.5 m/s
                "x": (0.0, 2.5), 
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

@configclass
class FlywheelPolicyGroupCfg(CycleRLObservationsCfg):
    flywheel_joint_vel = ObservationTermCfg(
        func=mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[JOINT_FLYWHEEL])}
    )

@configclass
class FlywheelObservationsCfg:
    policy: FlywheelPolicyGroupCfg = FlywheelPolicyGroupCfg()

@configclass
class FlywheelRewardsCfg(CycleRLRewardsCfg):
    # Inherits `velocity_tracking`, `steering_tracking` automatically
    survival = RewardTermCfg(func=custom_rewards.survival_reward, weight=0.5)
    velocity = RewardTermCfg(func=custom_rewards.velocity_tracking_reward, weight=5.0)
    # steering = RewardTermCfg(func=custom_rewards.steering_tracking_reward, weight=5.0)

    upright = RewardTermCfg(
        func=custom_rewards.upright_reward,
        weight=1.0, # Increased to match velocity/steering tracking
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    # 2. FLYWHEEL SATURATION
    flywheel_saturation_penalty = RewardTermCfg(
        func=custom_rewards.flywheel_saturation_penalty,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[JOINT_FLYWHEEL])}
    )
    
    # 3. OVERRIDDEN WEIGHTED PENALTIES
    act_mag = RewardTermCfg(
        func=custom_rewards.weighted_action_magnitude_penalty,
        weight=1.0, 
        params={
            # Steering takes 100%, Drive takes 100%, Flywheel takes 10%
            "weights": [1.0, 1.0, 0.01] 
        }
    )
    
    act_rate = RewardTermCfg(
        func=custom_rewards.weighted_action_rate_penalty,
        weight=2.0, 
        params={
            # Steering twitches take 100%, Drive takes 100%, Flywheel takes 10%
            "weights": [1.0, 1.0, 0.01] 
        }
    )

    anti_jump = RewardTermCfg(
        func=custom_rewards.no_jumping_penalty_local, # Using the new Body Frame function!
        weight=-5.0, # Keep the penalty high to bankrupt the bunny hop
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # flywheel_vibration = RewardTermCfg(
    #     func=custom_rewards.flywheel_oscillation_penalty,
    #     # A weight of -1.0 or -2.0 is usually enough because the squaring math gets huge very fast
    #     weight=-2.0, 
    #     params={
    #         "flywheel_idx": 2 # Tell the function exactly which motor to target
    #     }
    # )

    # trackstand_flywheel_bonus = RewardTermCfg(
    #     func=custom_rewards.flywheel_trackstand_bonus,
    #     # Keep this VERY small (e.g., +0.05). It is just a hint, not a primary objective!
    #     weight=0.1, 
    #     params={
    #         "flywheel_idx": 2, # Assuming [steer, drive, flywheel]
    #         "command_name": "velocity", # Check your command manager! Often "velocity_b" or "velocity_command"
    #         "vel_threshold": 0.1
    #     }
    # )

    strict_trackstand_penalty = RewardTermCfg(
        func=custom_rewards.strict_trackstand_motor_penalty,
        # Make this penalty massive so the agent is terrified to use these motors at 0 m/s
        weight=-0.5, 
        params={
            "command_name": "velocity",
            "vel_threshold": 0.1,
            # Assuming action order is [steer, drive, flywheel], we penalize indices 0 and 1
            "motor_indices": [0, 1] 
        }
    )

@configclass
class FlywheelCurriculumCfg:
    """Automated curriculum to shift from Trackstand priority to Driving priority."""
    
    trackstand_fade = CurriculumTermCfg(
        func=custom_curriculums.fade_trackstand_ratio,
        params={
            # The env.step() thresholds:
            # 0 steps: Start at 100% trackstand
            # 10,000 steps: Drop to 70% trackstand
            # 20,000 steps: Drop to 30% trackstand
            "step_thresholds": [0, 1000000, 2000000],
            "ratios": [1.0, 0.5, 0.2]
        }
    )

    spawn_velocity_fade = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 1000000, 2000000],
            "max_values": [0.0, 1.0, 2.5],
            "target_param": "spawn_velocity_x"
        }
    )

    spawn_heading_fade = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 1000000, 2000000],
            "max_values": [0.0, 0.26, 0.78], 
            "target_param": "spawn_yaw" 
        }
    )

    command_heading_fade = CurriculumTermCfg(
        func=custom_curriculums.expand_curriculum_range,
        params={
            "step_thresholds": [0, 2000000, 4000000],
            "max_values": [0.0, 0.5, 1.0], 
            "target_param": "command_heading"
        }
    )
@configclass
class FlywheelTerminationsCfg(CycleRLTerminationsCfg):
    # Inherits your base timeout, but we override the fall_over with the 45-degree check
    fall_over = TerminationTermCfg(
        func=custom_terminations.roll_exceeds_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 45.0
        }
    )

# =========================================================================
# Main Environment Configuration
# =========================================================================

@configclass
class CycleRLFlywheelEnvCfg(CycleRLEnvCfg):
    """Config 1: Flywheel-Only Trackstand."""

    scene: FlywheelSceneCfg = FlywheelSceneCfg(num_envs=16384, env_spacing=2.5)
    events: FlywheelEventsCfg = FlywheelEventsCfg()
    commands: FlywheelCommandsCfg = FlywheelCommandsCfg()
    
    actions: FlywheelActionsCfg = FlywheelActionsCfg()
    observations: FlywheelObservationsCfg = FlywheelObservationsCfg()
    rewards: FlywheelRewardsCfg = FlywheelRewardsCfg()
    terminations: FlywheelTerminationsCfg = FlywheelTerminationsCfg()
    curriculum: FlywheelCurriculumCfg = FlywheelCurriculumCfg()