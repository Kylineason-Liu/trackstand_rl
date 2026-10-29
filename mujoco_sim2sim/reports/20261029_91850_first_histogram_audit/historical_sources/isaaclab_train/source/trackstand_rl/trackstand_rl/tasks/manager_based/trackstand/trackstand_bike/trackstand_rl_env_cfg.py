"""Stationary balancing environment for the SDF-wheel trackstand bike.

Goal: keep the bicycle upright at its spawn position using steering and drive,
with a strong position-keeping penalty to prevent drift.

THE single env config (Trackstand-Balance-v0). Control contract (decided
2026-09-22): steering = joint POSITION servo (kp 10 / kd 0.2), drive = joint
EFFORT / torque (stiffness 0 / damping 0.05) — see bicycle_cfg.py's
TRACKSTAND_ROBOT_CFG. No other control-mode variants exist.

Fully standalone: the robot cfg (../bicycle_cfg), env class
(trackstand_rl_env), agent cfg (../agents/rsl_rl_ppo_cfg) and the MDP term
library (../mdp/stationary) all live inside this task package — no imports
from CycleRL or the NewBike task.
"""

from copy import deepcopy

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg,
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
from isaaclab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils.configclass import configclass

from ..mdp.stationary.noise import PerChannelGaussianNoiseCfg

from ..bicycle_cfg import TRACKSTAND_ROBOT_CFG
from ..mdp.stationary import curriculums as custom_curriculums
from ..mdp.stationary import events as custom_events
from ..mdp.stationary.observations import balance_imu_state
from ..mdp.stationary.rewards import (
    balance_action_rate_penalty_norm,
    balance_drift_gaussian,
    balance_survival_reward_binary,
)
from ..mdp.stationary.terminations import (
    balance_drift_termination,
    balance_wheel_net_termination,
    balance_yaw_deviation_termination,
    nonfinite_or_extreme_state,
    thubike_fallen_termination,
)


# ===========================================================================
# Config classes
# ===========================================================================


@configclass
class TrackstandRLSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = TRACKSTAND_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class TrackstandRLActionsCfg:
    """steering = joint POSITION (servo), drive = joint EFFORT (torque).

    Action vector is 2-dim: [steer_target, drive_torque], each in [-1, 1] before scaling.
    """

    steering = JointPositionActionCfg(
        asset_name="robot", joint_names=["steering_joint"], scale=1.57  # ±90°
    )
    drive = JointEffortActionCfg(
        asset_name="robot", joint_names=["rear_wheel_joint"], scale=6.28  # ±stall N·m
    )


@configclass
class TrackstandRLObservationsCfg(ObservationGroupCfg):
    enable_corruption = True  # turns ON the per-step gaussian obs noise (term #8 white noise)

    imu = ObservationTermCfg(
        func=balance_imu_state,
        params={"steer_joint_name": "steering_joint", "drive_joint_name": "rear_wheel_joint"},
        # per-channel std for the 13-dim obs [gravity_y, gravity_z, angvel*3, linvel*2,
        # steering_pos, drive_pos, steering_vel, drive_vel, sin, cos]:
        #   gravity: 0.001 (g units; accel noise 0.001 g RMS from the IMU manual)
        #   ang_vel: 0.0012 (rad/s; gyro noise 0.07 deg/s RMS from the IMU manual)
        #   lin_vel / joints / sin-cos: fusion- and encoder-derived, not in the manual -> 0
        noise=PerChannelGaussianNoiseCfg(std=(0.001, 0.001, 0.0012, 0.0012, 0.0012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    )


@configclass
class TrackstandRLRewardsCfg:
    """The active v1 reward set (the 09090950 recipe, kept through all runs).

    - survival (5.0): binary +1 while |roll|<30° — flat alive bonus, no shaping.
      Dominates everything else so balance stays primary.
    - drift_gauss (2.0): Gaussian drift reward, σ=0.25 m — position-keeping
      without fighting the necessary micro-motion.
    - rate_penalty (0.1): per-joint-normalised -‖Δa/scale‖₂ — damps jitter.
      The func returns NEGATIVE values, so the weight stays POSITIVE to remain
      a penalty (reward manager: value * weight; the sign was wrong in iter1).
    """

    survival = RewardTermCfg(func=balance_survival_reward_binary, weight=5.0)
    drift_gauss = RewardTermCfg(
        func=balance_drift_gaussian,
        weight=2.0,  # peak per-step contribution
        params={"sigma": 0.25},  # m
    )
    rate_penalty = RewardTermCfg(
        func=balance_action_rate_penalty_norm,
        weight=0.1,
        params={"action_scales": (1.57, 6.28)},  # steer rad, drive N-m
    )


@configclass
class TrackstandRLTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    fallen = TerminationTermCfg(
        func=thubike_fallen_termination,
        params={"limit_angle": 0.5236},  # 30°
        time_out=False,
    )
    # Containment: reset a diverging env before a NaN can poison the shared policy.
    diverged = TerminationTermCfg(
        func=nonfinite_or_extreme_state,
        # max_joint_vel: root-only checks miss a runaway wheel under torque control
        # (see terminations.py). Loose on purpose — 5x the drive motor's limit, so it
        # only fires on real divergence, not on ordinary velocity-limit overshoot.
        params={"max_lin_vel": 20.0, "max_ang_vel": 50.0, "max_joint_vel": 300.0},
        time_out=False,
    )
    # trackstand envelope:
    drifted = TerminationTermCfg(
        func=balance_drift_termination,
        params={"max_dist": 0.5},  # m; wheel radius ~0.07 m, so orbiting stays reset
        time_out=False,
    )
    yaw_deviation = TerminationTermCfg(
        func=balance_yaw_deviation_termination,
        params={"limit_angle": 0.5236},  # 30 deg vs spawn heading — a circling bike gets reset
        time_out=False,
    )
    wheel_net = TerminationTermCfg(
        func=balance_wheel_net_termination,
        params={"limit_angle": 7.2},  # rad of net rotation since reset; micro-motion oscillates ~0
        time_out=False,
    )


@configclass
class TrackstandRLEventsCfg:
    """Full DR event templates; TrackstandRLEnvCfg disables DR unless opted in."""

    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

    # --- Ground friction (startup): pinned to a fixed nominal, so it's DR-free.
    # The shared ground can't be ramped reliably mid-sim (USD material writes), so
    # friction DR is carried entirely by the per-env robot tire
    # (randomize_material below), which updates reliably at reset. ---
    randomize_ground_friction = EventTermCfg(
        func=custom_events.randomize_ground_plane_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("terrain"),
            "static_friction_range": (1.0, 1.0),   # pinned nominal; tune if needed
            "dynamic_friction_range": (1.0, 1.0),
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
            "mass_distribution_params": (0.8, 1.2),
        },
    )

    randomize_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (-0.005, 0.005),
            },
        },
    )

    randomize_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # The tire material carries frictionCombineMode="min", and the ground
            # is pinned at 1.0 -> the PAIR friction equals the tire value itself
            # (min wins over the ground's average per the PhysX combine-mode
            # precedence). So these ranges ARE the ground-wheel relative
            # friction, centered on the nominal pair 0.65 (the measured
            # real-environment value) — sim-to-real targeted.
            "static_friction_range": (0.5, 0.8),
            "dynamic_friction_range": (0.5, 0.8),
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
                "roll": (-0.10, 0.10),  # gentler starts — recovery must fit inside the 30-deg heading cap
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),  # stationary balancing: no initial forward momentum
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.25, 0.25),  # 0.5 swung heading past 30 deg during recovery (41% yaw deaths)
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # --- IMU turn-on residuals: per-episode post-calibration bias residual (reset) ---
    #     Turn-on residual is #1 error mentioned in the records. It can be largely
    #     eliminated by calibration, but the post-calibration residual is non-zero
    #     and un-negligible. (The 20-60 ms sensor/loop latency params were REMOVED
    #     on 2026-09-08.)
    reset_imu_turnon_residuals = EventTermCfg(
        func=custom_events.reset_imu_turnon_residuals,
        mode="reset",
        params={
            "gyro_res_bias_std": 0.0002,  # PLACEHOLDER 2e-4 rad/s (= calibration floor noise/sqrt(N) ~1e-4 + thermal allowance) — REPLACE with the MEASURED per-boot residual spread
            "accel_res_bias_std": 0.002,  # PLACEHOLDER 0.002 g (~0.11 deg, stand-leveling dominated) — REPLACE with the MEASURED accel residual
        },
    )


@configclass
class TrackstandRLCurriculumsCfg:
    # Diagnostics only: logs motor speeds to TensorBoard (Curriculum/...).
    motor_speeds = CurriculumTermCfg(func=custom_curriculums.log_motor_speeds)
    # DR is controlled by the environment's opt-in switch, never a ramp.


@configclass
class TrackstandRLEnvCfg(ManagerBasedRLEnvCfg):
    """THE trackstand env config: hybrid control (position steering + torque drive).

    No other trackstand env config exists — the control mode was decided
    2026-09-22. 400 Hz physics / 50 Hz control. DR is OFF by default;
    --dr enables the existing full-strength, per-environment/per-episode DR.
    """
    # decimation = 2
    decimation = 8
    scene = TrackstandRLSceneCfg(num_envs=4096, env_spacing=2.5)
    actions = TrackstandRLActionsCfg()
    observations = {"policy": TrackstandRLObservationsCfg()}
    rewards = TrackstandRLRewardsCfg()
    terminations = TrackstandRLTerminationsCfg()
    events = TrackstandRLEventsCfg()
    curriculum = TrackstandRLCurriculumsCfg()
    domain_randomization: bool = False
    # No commands — pure stationary balancing with position-keeping

    def set_domain_randomization(self, enabled: bool) -> None:
        """Select nominal or full-DR reset events before constructing the env.

        Keep the fixed shared-ground setup, reset_scene, observation white noise,
        rewards, actuators, and termination envelope unchanged. Enabling restores
        missing DR templates without replacing existing customized DR terms.
        """
        defaults = TrackstandRLEventsCfg()
        for name in (
            "randomize_mass",
            "randomize_com",
            "randomize_material",
            "randomize_actuator_gains",
            "reset_imu_turnon_residuals",
        ):
            if not enabled:
                setattr(self.events, name, None)
            elif getattr(self.events, name, None) is None:
                setattr(self.events, name, deepcopy(getattr(defaults, name)))

        # Nominal reset: upright with zero roll rate. Full DR restores the
        # existing roll/roll-rate ranges; other root-reset settings are retained.
        initial_state = self.events.initial_state
        if initial_state is not None:
            for key in ("pose_range", "velocity_range"):
                initial_state.params[key]["roll"] = (
                    defaults.initial_state.params[key]["roll"] if enabled else (0.0, 0.0)
                )
        self.domain_randomization = bool(enabled)

    def __post_init__(self):
        self.viewer.eye = [5.0, 5.0, 2.0]
        # 50 Hz control. step_dt is a READ-ONLY property on the env
        # (= sim.dt * decimation), NOT a cfg field — assigning `self.step_dt`
        # silently does nothing and left the policy running at 30 Hz
        # (sim.dt default 1/60 * decimation 2 = 33.3 ms). Set sim.dt instead.
        # self.sim.dt = 0.01
        self.sim.dt = 0.0025
        self.episode_length_s = 30.0  # 30 s of balancing per episode -> 1500 steps
        self.sim.render_interval = self.decimation
        # scene-default physics material (inherited by the robot shapes that
        # carry no USD-authored material): nominal pair friction 0.65 (the
        # measured real-environment value) with MIN-combine — pair = min(tire,
        # ground 1.0) = tire, so the tire value IS the ground-wheel relative
        # friction (see randomize_material).
        self.sim.physics_material.static_friction = 0.65
        self.sim.physics_material.dynamic_friction = 0.65
        self.sim.physics_material.friction_combine_mode = "min"
        # apply actuator/external forces EVERY TGS solver iteration: with the
        # flag off the drive torque is injected once per step and the velocity
        # sweeps can't propagate it (the x24.5 sluggish-drive root cause,
        # confirmed by the in-air probe 2026-09-13; see cfg_record.txt).
        # The default None physics cfg uses the engine defaults -- assigning an
        # explicit PhysxCfg with only this flag keeps everything else default.
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
        self.sim.physics = PhysxCfg(enable_external_forces_every_iteration=True)
        self.set_domain_randomization(self.domain_randomization)
