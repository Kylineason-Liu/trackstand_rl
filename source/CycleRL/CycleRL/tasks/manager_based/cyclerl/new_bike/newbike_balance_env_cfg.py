"""Stationary balancing environment for NewBike — independent of CycleRL.

Goal: keep the bicycle upright at its spawn position using steering and drive,
with a strong position-keeping penalty to prevent drift.
"""

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
from isaaclab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg, JointVelocityActionCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils.configclass import configclass

from ..mdp.stationary.noise import PerChannelGaussianNoiseCfg

from ..bicycle_cfg import NEW_BIKE_HYBRID_CFG, NEW_BIKE_STANDARD_CFG, NEW_BIKE_TORQUE_CFG
from ..mdp.stationary import curriculums as custom_curriculums
from ..mdp.stationary import events as custom_events
from ..mdp.stationary.observations import balance_imu_state
from ..mdp.stationary.rewards import (
    balance_action_rate_penalty,
    balance_action_rate_penalty_norm,
    balance_displacement_band,
    balance_drive_reward,
    balance_drift_gaussian,
    balance_joint_acc_penalty,
    balance_joint_torque_penalty,
    balance_micro_motion_band,
    balance_position_penalty,
    # balance_stability_reward,  # disabled below (kept in rewards.py)
    balance_roll_shaping,
    balance_steer_reward,
    balance_survival_reward,
    balance_survival_reward_binary,
    balance_termination_penalty,
    balance_wheel_net_angle_band,
    balance_wheel_net_gaussian,
    balance_yaw_band,
    balance_drift_penalty_quadratic,
    balance_wheel_net_angle_penalty,
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
class NewBikeBalanceSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    robot = NEW_BIKE_STANDARD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class NewBikeBalanceActionsCfg:
    steering = JointPositionActionCfg(
        asset_name="robot", joint_names=["steering_joint"], scale=1.57  # ±90°
    )
    drive = JointVelocityActionCfg(
        asset_name="robot", joint_names=["rearwheel_joint"], scale=10.0,
        # ponytail: cap the velocity target so the Gaussian policy can't command a
        # blow-up wheel speed (clip == scale, per this repo's convention). Tune.
        clip={"rearwheel_joint": (-10.0, 10.0)},
    )


@configclass
class NewBikeBalanceObservationsCfg(ObservationGroupCfg):
    enable_corruption = True  # turns ON the per-step gaussian obs noise (term #8 white noise)

    imu = ObservationTermCfg(
        func=balance_imu_state,
        params={"steer_joint_name": "steering_joint", "drive_joint_name": "rearwheel_joint"},
        # per-channel std matching the 9-dim obs [gravity_y, gravity_z, angvel*3, linvel*2, sin, cos]:
        #   gravity: 0.001 (g units; accel noise 0.001 g RMS from the IMU manual)
        #   ang_vel: 0.0012 (rad/s; gyro noise 0.07 deg/s RMS from the IMU manual)
        #   lin_vel / sin-cos: fusion- and encoder-derived, not in the manual -> 0 until measured
        noise=PerChannelGaussianNoiseCfg(std=(0.001, 0.001, 0.0012, 0.0012, 0.0012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ) #                                       0.001, 0.001, 0.0012, 0.0012, 0.0012, 0.0, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Shared reward, termination, and event configs (bike-agnostic)
# ---------------------------------------------------------------------------


@configclass
class BalanceRewardsCfg:
    # exponential survival (commented — flip back by swapping which line is active):
    #survival = RewardTermCfg(func=balance_survival_reward, weight=5.0)  # exp reward peaks at 1.0 (vertical), so weight == peak survival = 5.0.
    survival = RewardTermCfg(func=balance_survival_reward_binary, weight=5.0)  # binary +1 while |roll|<30°, else 0 (flat alive bonus, no shaping). Dominates drive(1.0)/steer(0.25).
    # action-rate (smoothness) penalty: -||a_t - a_{t-1}||. weight 0.0 = OFF by default
    # (contributes nothing, so no-flag runs are unchanged); set per-run with --arp_w
    # (e.g. 0.1) to damp motor jitter/chaos. Unbounded -> keep the weight small.
    action_rate = RewardTermCfg(func=balance_action_rate_penalty, weight=0.0)
    #termination_pen = RewardTermCfg(func=balance_termination_penalty, weight=10.0)  # one-time -1 on failure termination (fallen/diverged)
    # stability disabled: it penalises angular velocity, which fights the active
    # corrective motion the bike needs. Function kept in rewards.py.
    # stability = RewardTermCfg(func=balance_stability_reward, weight=1.0)
    #position = RewardTermCfg(func=balance_position_penalty, weight=0.01)
    # Quadratic (-v^2) velocity penalties; weights lowered so the motors aren't
    # frozen. Unbounded but polynomial (no overflow) — see rewards.py.
    drive_reward = RewardTermCfg(
        func=balance_drive_reward,
        weight=12.0,  # bounded reward peaks ~1.0; set to desired peak. Keep < survival (5) so balance stays primary.
        params={"drive_joint_name": "rearwheel_joint"},
    )

    '''
    
    '''
    steer_reward = RewardTermCfg(
            func=balance_steer_reward,
            weight=5.0,  # peaks ~1.0; keep SMALL — steering is the main balancing actuator, over-rewarding stillness suppresses corrective steering.
            params={"steer_joint_name": "steering_joint"},
    )
    '''
    
    '''
    
    #joint_acc = RewardTermCfg(func=balance_joint_acc_penalty, weight=0.01)


@configclass
class BalanceTerminationsCfg:
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


@configclass
class BalanceEventsCfg:
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
    '''
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
                "x": (-0.0, 0.0), # 0.02
                "y": (-0.01, 0.01), # 0.02
                "z": (-0.0, 0.0), # 0.02
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
                "roll": (-0.10, 0.10),  # iter10: gentler starts — recovery must fit inside the 30-deg heading cap
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),  # stationary balancing: no initial forward momentum
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.25, 0.25),  # iter10: 0.5 swung heading past 30 deg during recovery (41% yaw deaths)
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    
    # --- IMU turn-on residuals: per-episode post-calibration bias residual (reset) ---
    #     Turn-on residual is #1 error mentioned in cfg_record.txt. It can be largely eliminated
    #     by calibration, but the post-calibration residual is non-zero and un-negligible.
    #     (The 20-60 ms sensor/loop latency params were REMOVED on 2026-09-08.)
    reset_imu_turnon_residuals = EventTermCfg(
        func=custom_events.reset_imu_turnon_residuals,
        mode="reset",
        params={
            "gyro_res_bias_std": 0.0000,  # 0.03 original | PLACEHOLDER 2e-4 rad/s (= calibration floor noise/sqrt(N) ~1e-4 + thermal allowance) — REPLACE with the MEASURED per-boot residual spread
            "accel_res_bias_std": 0.002,  # 0.02 original | PLACEHOLDER 0.002 g (~0.11 deg, stand-leveling dominated) — REPLACE with the MEASURED accel residual
        },
    )
    
    '''


@configclass
class BalanceCurriculumsCfg:
    # Diagnostics only: logs motor speeds to TensorBoard (Curriculum/...).
    motor_speeds = CurriculumTermCfg(func=custom_curriculums.log_motor_speeds)
    # Gradual DR ramp: OFF at step 0 -> full authored ranges by step 50k.
    # alpha logged as Curriculum/dr_ramp. (Ground friction is pinned/DR-free;
    # the per-env robot tire material carries the friction DR.)
    # no dr_ramp for evaluating different DR terms' effects on policy learning curve
    '''
    dr_ramp = CurriculumTermCfg(
        func=custom_curriculums.ramp_domain_randomization,
        params={"start_step": 0, "end_step": 50000},
    )
    '''
    


@configclass
class NewBikeBalanceEnvCfg(ManagerBasedRLEnvCfg):
    decimation = 2
    scene = NewBikeBalanceSceneCfg(num_envs=4096, env_spacing=2.5)
    actions = NewBikeBalanceActionsCfg()
    observations = {"policy": NewBikeBalanceObservationsCfg()}
    rewards = BalanceRewardsCfg()
    terminations = BalanceTerminationsCfg()
    events = BalanceEventsCfg()
    curriculum = BalanceCurriculumsCfg()
    # No commands — pure stationary balancing with position-keeping

    def __post_init__(self):
        self.viewer.eye = [5.0, 5.0, 2.0]
        # 50 Hz control. step_dt is a READ-ONLY property on the env
        # (= sim.dt * decimation), NOT a cfg field — assigning `self.step_dt`
        # silently does nothing and left the policy running at 30 Hz
        # (sim.dt default 1/60 * decimation 2 = 33.3 ms). Set sim.dt instead.
        self.sim.dt = 0.01  # * decimation(2) -> step_dt = 0.02 s = 50 Hz
        self.episode_length_s = 30.0  # 30 s of balancing per episode -> 1500 steps


# ===========================================================================
# Torque-controlled variant (A/B against the position/velocity setup above)
# ===========================================================================


@configclass
class NewBikeBalanceTorqueActionsCfg:
    """Direct torque control: the policy outputs a torque per motor (like the
    well-balanced lmbike_rl setup and the GM4310/GIM3510 CAN torque interface).
    Scale = the motor stall torque; the actuator effort_limit clips beyond it."""

    steering = JointEffortActionCfg(
        asset_name="robot", joint_names=["steering_joint"], scale=5.6  # ±stall N·m
    )
    drive = JointEffortActionCfg(
        asset_name="robot", joint_names=["rearwheel_joint"], scale=6.28  # ±stall N·m
    )


@configclass
class NewBikeBalanceHybridActionsCfg:
    """Hybrid control: steering = joint POSITION (servo), drive = joint EFFORT (torque).

    Action vector is 2-dim: [steer_target, drive_torque], each in [-1, 1] before scaling.
    """

    steering = JointPositionActionCfg(
        asset_name="robot", joint_names=["steering_joint"], scale=1.57  # ±90°
    )
    drive = JointEffortActionCfg(
        asset_name="robot", joint_names=["rearwheel_joint"], scale=6.28  # ±stall N·m
    )


@configclass
class NewBikeBalanceHybridRewardsCfg:
    """Literature-based reward for the hybrid (torque drive + servo steering) track stand.

    Reworked per the static-balance survey: the old drive_reward/steer_reward paid
    for a *still* wheel, which contradicts the trackstand mechanism (Brockett: no
    static feedback law — micro-motion is necessary; Wang 2024: balance torque
    comes from torque-level rear-wheel micro-motion). Term-by-term:

    - survival (w=2.0): flat alive bonus while |roll|<30° — just enough to make
      falling strictly worse than wobbling; no shaping so it can't paper over
      bad uprightness.
    - roll_shaping (w=8.0): exp(-roll²/2σ²), σ=12° — the main gradient term,
      pushes upright inside the survival band (survival alone pays 29° == 0°).
    - drift_band (w=5.0): tolerance band ±0.2 m around spawn, Gaussian decay
      beyond — position-keeping WITHOUT fighting the necessary micro-motion.
    - yaw_band (w=4.0): ANTI-ORBITAL plateau+linear tail on |yaw| vs spawn
      heading — a circling policy earns nothing for holding a circle.
    - micro_band (w=4.0): plateau up to |ω_drive|=1.5 rad/s, linear tail beyond —
      rewards micro-motion, with CONSTANT gradient against runaway spin.
    - act_penalty (w=0.1): -‖a‖₂ on the 2-dim action — light effort cost.
      (Func returns negative; weight stays positive so it remains a penalty.)
    - rate_penalty (w=0.1): per-joint-normalised -‖Δa/scale‖₂ — damps jitter
      without over-penalising the necessary oscillation frequency.
    """

    survival = RewardTermCfg(func=balance_survival_reward_binary, weight=5.0)  # iter10: episodes die in <1s; alive-time gradient must dominate

    
    '''
    roll_shaping = RewardTermCfg(
        func=balance_roll_shaping,
        weight=5.0,
        params={"sigma": 0.2094},  # 12 deg: iter3's 15 deg flattened precision (roll RMS 4.94->6.51)
    )
    drift_band = RewardTermCfg( # Can we delete this, because yaw_band, wheel_net and micro_band already penalize drift?
        func=balance_displacement_band,
        weight=1.5,  # iter2 2.0: drift deaths (max 2.0 m, term firing). Position-keeping = first-class objective.
        params={"band": 0.2, "sigma": 0.3},
    )
    yaw_band = RewardTermCfg(
        func=balance_yaw_band,
        weight=2.5,  # ANTI-ORBITAL. iter3 Gaussian inert; iter4 CLAMPED tail -> policy parked AT the pi floor
        params={"band": 0.1, "tail_slope": 0.66},  # NO clamp: value ~ -1 at pi, constant gradient everywhere
    )
    wheel_net = RewardTermCfg(
        func=balance_wheel_net_angle_band,
        weight=1.0,  # anti-orbit/anti-roll backstop in wheel space (net rotation since reset)
        params={"band": 1.0ate_penalty = RewardTermCfg(
        func=balance_action_rate_penalty_norm,
        weight=0.1,  # iter1 -0.05 SIGN WRONG (rewarded slamming); 0.1 damps servo slam (steer max 151 rad/s)
        params={"action_scales": (1.57, 6.28)},  # steer rad, drive N-m
    ), "tail_slope": 0.19, "penalty_floor": -1.0},  # floor-capped: no suicide pressure
    )
    micro_band = RewardTermCfg(
        func=balance_micro_motion_band,
        weight=2.5,
        # iter3: drive still 6.1 rad/s vs target 1.5 -> tail gradient (0.03) too weak. Steepen to 0.05.
        params={"target": 1.5, "tail_slope": 0.05, "drive_joint_name": "rearwheel_joint"},
    )
    drift_quad = RewardTermCfg(
            func=balance_drift_penalty_quadratic,
            weight=1.0
        )
    wheel_net_quad = RewardTermCfg(
            func=balance_wheel_net_angle_penalty,
            weight=1.0,
        )

    # --- gaussian reward variants (EXPERIMENTAL — NOT active; import
    #     balance_drift_gaussian, balance_wheel_net_gaussian from ..mdp.stationary.rewards) ---
    # Gaussian wheel-net: positive, gradient everywhere incl. the band's dead zone;
    # wide sigma keeps the tail alive out to the 7.2 rad termination cap.
    
    # Gaussian drift: d bounded by the 0.5 m drifted termination -> gradient over
    # the whole range; ~0.83 at the 0.15 m operating point (barely perturbs the optimum).
    wheel_net_gauss = RewardTermCfg(
        func=balance_wheel_net_gaussian,
        weight=2.0,  # peak per-step contribution
        params={"sigma": 3.0},  # rad
    )
    '''
    
    drift_gauss = RewardTermCfg(
        func=balance_drift_gaussian,
        weight=2.0,  # peak per-step contribution
        params={"sigma": 0.25},  # m
    )

    # NOTE: both penalty funcs RETURN NEGATIVE values (see rewards.py), so the
    # weights must be POSITIVE to stay penalties — negative weight would flip
    # them into rewards (reward manager: value * weight). Iter1 had the sign
    # wrong (TB confirmed net-positive contributions); fixed in iter2.
    '''
    
    act_penalty = RewardTermCfg(func=balance_joint_torque_penalty, weight=0.1)

    '''
    rate_penalty = RewardTermCfg(
        func=balance_action_rate_penalty_norm,
        weight=0.1,  # iter1 -0.05 SIGN WRONG (rewarded slamming); 0.1 damps servo slam (steer max 151 rad/s)
        params={"action_scales": (1.57, 6.28)},  # steer rad, drive N-m
    )
    

@configclass
class NewBikeBalanceHybridTerminationsCfg(BalanceTerminationsCfg):
    """v1 terminations: parent's (time_out / fallen 30° / diverged) plus three
    trackstand-envelope terms:

    drifted (0.5 m): position envelope around spawn. 2 m let the bike orbit
    inside the limit with a ~0.07 m wheel radius; 0.5 m keeps the stand local.
    yaw_deviation (60°): heading envelope vs spawn heading — a circling bike
    accumulates yaw and gets reset.
    wheel_net (7.2 rad): net rear-wheel rotation envelope — constant rolling
    or sustained in-place spin accumulates it; micro-motion oscillates ~0.
    """

    drifted = TerminationTermCfg(
        func=balance_drift_termination,
        params={"max_dist": 0.5},  # m; wheel radius ~0.07 m, so 2.0 m allowed orbiting inside
        time_out=False,
    )
    yaw_deviation = TerminationTermCfg(
        func=balance_yaw_deviation_termination,
        params={"limit_angle": 0.5236},  # 30 deg
        time_out=False,
    )
    wheel_net = TerminationTermCfg(
        func=balance_wheel_net_termination,
        params={"limit_angle": 7.2},  # rad of net rotation since reset
        time_out=False,
    )


@configclass
class NewBikeBalanceHybridEnvCfg(NewBikeBalanceEnvCfg):
    """NewBike-Balance-v1: same task as v0, hybrid control.

    Drive = direct torque (like the Torque variant); steering = position servo
    (like v0). Rewards/terminations are the literature-based v1 set; events,
    curriculum and observations are inherited unchanged.
    """

    actions = NewBikeBalanceHybridActionsCfg()
    rewards = NewBikeBalanceHybridRewardsCfg()
    terminations = NewBikeBalanceHybridTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # steering servo gains + torque-mode drive (stiffness=0)
        self.scene.robot = NEW_BIKE_HYBRID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class NewBikeBalanceTorqueEnvCfg(NewBikeBalanceEnvCfg):
    """Torque-controlled A/B variant of the NewBike balance task.

    Both motors are driven by direct effort/torque instead of the parent's
    position (steering) + velocity (drive). Everything else — rewards,
    terminations, events, curriculum, observations — is inherited unchanged, so
    the only difference under test is the control mode.
    """

    actions = NewBikeBalanceTorqueActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        # swap in the torque actuators (stiffness=0 so the effort action isn't
        # fought by a PD servo)
        self.scene.robot = NEW_BIKE_TORQUE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
