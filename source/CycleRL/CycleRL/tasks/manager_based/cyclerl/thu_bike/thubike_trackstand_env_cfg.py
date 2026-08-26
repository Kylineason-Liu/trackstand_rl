"""Trackstand environment for THUBike — successfully discovers stationary balancing.

Goal: keep the bicycle upright at its spawn position using bidirectional velocity
commands, steering, and drive. Exponential position penalty forces the bike to
stay near spawn; rate-based action penalties prevent motor twitching without
punishing sustained drive output.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
    SceneEntityCfg,
)
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import DomeLightCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from ..bicycle_cfg import THUBIKE_STANDARD_CFG
from ..mdp.stationary import events as custom_events
from ..mdp.stationary.observations import balance_command_obs, balance_imu_state
from ..mdp.stationary.rewards import (
    balance_action_rate_penalty,
    balance_joint_acc_penalty,
    balance_stability_reward,
    balance_upright_reward,
    balance_velocity_tracking_reward,
    trackstand_position_penalty,
)
from ..mdp.stationary.terminations import thubike_fallen_termination


# ===========================================================================
# Config classes
# ===========================================================================


@configclass
class TrackstandSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = THUBIKE_STANDARD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class TrackstandActionsCfg:
    steering = JointPositionActionCfg(
        asset_name="robot", joint_names=["Revolute_9"], scale=1.57  # ±90°
    )
    drive = JointVelocityActionCfg(
        asset_name="robot", joint_names=["Revolute_8"], scale=10.0
    )


@configclass
class TrackstandObservationsCfg(ObservationGroupCfg):
    imu = ObservationTermCfg(
        func=balance_imu_state,
        params={"steer_joint_name": "Revolute_9", "drive_joint_name": "Revolute_8"},
        noise=GaussianNoiseCfg(mean=0.0, std=0.01),
    )
    commands = ObservationTermCfg(func=balance_command_obs)


@configclass
class TrackstandCommandsCfg:
    """Minimal bidirectional velocity commands.

    40% of envs get zero velocity (pure stationary balancing).
    Commands resample every 5-10 s so the policy has time to settle.
    """

    velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.4,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),  # bidirectional: backward to forward
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class TrackstandRewardsCfg:
    upright = RewardTermCfg(func=balance_upright_reward, weight=1.0)
    stability = RewardTermCfg(func=balance_stability_reward, weight=3.0)
    position = RewardTermCfg(func=trackstand_position_penalty, weight=0.1)
    vel_track = RewardTermCfg(func=balance_velocity_tracking_reward, weight=0.5)
    # Rate-based penalties only — magnitude penalties fight the drive motor
    act_rate = RewardTermCfg(func=balance_action_rate_penalty, weight=0.03)
    joint_acc = RewardTermCfg(func=balance_joint_acc_penalty, weight=0.003)


@configclass
class TrackstandTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    fallen = TerminationTermCfg(
        func=thubike_fallen_termination,
        params={"limit_angle": 0.785},  # 45°
        time_out=False,
    )


@configclass
class TrackstandEventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

    # --- Ground friction (startup) ---
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

    # --- Robot dynamics (reset) ---
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
                "z": (-0.05, 0.05),
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
                "roll": (-0.15, 0.15),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class TrackstandEnvCfg(ManagerBasedRLEnvCfg):
    decimation = 2
    scene = TrackstandSceneCfg(num_envs=4096, env_spacing=2.5)
    actions = TrackstandActionsCfg()
    observations = {"policy": TrackstandObservationsCfg()}
    rewards = TrackstandRewardsCfg()
    terminations = TrackstandTerminationsCfg()
    commands = TrackstandCommandsCfg()
    events = TrackstandEventsCfg()

    def __post_init__(self):
        self.viewer.eye = [5.0, 5.0, 2.0]
        # 50 Hz control. Assigning `self.step_dt` does NOT work — it is a
        # read-only property on the env (= sim.dt * decimation), not a cfg
        # field, so it silently left control at 30 Hz. Set sim.dt instead.
        self.sim.dt = 0.01  # * decimation(2) -> step_dt = 0.02 s = 50 Hz
        self.episode_length_s = 30.0  # -> 1500 steps
