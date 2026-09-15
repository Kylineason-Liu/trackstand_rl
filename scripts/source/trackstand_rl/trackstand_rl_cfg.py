"""Robot ArticulationCfg for the SDF-wheel trackstand bike — standalone.

The trackstand_rl_0914 import whose wheels carry SDF collision
(physics:approximation = "sdf" + PhysxSDFMeshCollisionAPI, sdfResolution 256,
sdfMargin 0.01). Joints: steering_joint (revolute, limited), rear_wheel_joint
(continuous, drive), front_wheel_joint (continuous, unactuated).

Actuator contract mirrors the real motors (config/motor_contract.yaml):
steering = GIM4310-10 (stall 5.6 Nm, 228 RPM), drive = GIM3510-8 (stall
6.28 Nm, 588 RPM). Independent of the NewBike cfg module.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TRACKSTAND_USD_PATH = os.path.join(CURRENT_DIR, "assets", "trackstand_rl.usda")

TRACKSTAND_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRACKSTAND_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # the probe-verified TGS fix from the engine audit (2026-09-13):
            # 16/8 delivers the effort-actuated drive rigidly (see cfg_record.txt)
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.02),
        rot=(0.0, 0.0, 0.0, 1.0),  # (x, y, z, w)
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={},
)

# position steering + velocity drive (the base-scene placeholder, swapped
# for the hybrid cfg in TrackstandRLHybridEnvCfg.__post_init__)
TRACKSTAND_STANDARD_CFG = TRACKSTAND_BASE_CFG.replace(
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["steering_joint"],
            effort_limit_sim=5.6,   # GIM4310-10 V1.1 stall (N·m)
            velocity_limit_sim=23.9,    # 228 RPM
            stiffness=10.0,         # same SI PD as MIT kp
            damping=0.2,            # same SI PD as MIT kd
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["rear_wheel_joint"],
            effort_limit_sim=6.28,  # GIM3510-8 stall (N·m)
            velocity_limit_sim=61.6,    # 588 RPM
            stiffness=0.0,          # velocity mode: τ = Kd*(ω*−ω)
            damping=5.0,
        ),
    }
)

# hybrid: position steering + torque drive — the Trackstand-rl-v0 robot cfg
TRACKSTAND_HYBRID_CFG = TRACKSTAND_BASE_CFG.replace(
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["steering_joint"],
            effort_limit_sim=5.6,   # GIM4310-10 V1.1 stall (N·m)
            velocity_limit_sim=23.9,    # 228 RPM
            stiffness=10.0,         # position servo: same SI PD as MIT kp
            damping=0.2,            # same SI PD as MIT kd
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["rear_wheel_joint"],
            effort_limit_sim=6.28,  # GIM3510-8 stall (N·m)
            velocity_limit_sim=61.6,    # 588 RPM
            stiffness=0.0,          # torque control: no position servo
            damping=0.05,           # light bearing friction; tune
        ),
    }
)

# direct torque on both motors (A/B variant; unused until a torque task exists)
TRACKSTAND_TORQUE_CFG = TRACKSTAND_BASE_CFG.replace(
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["steering_joint"],
            effort_limit_sim=5.6,       # GIM4310-10 V1.1 stall (N·m)
            velocity_limit_sim=23.9,    # 228 RPM
            stiffness=0.0,              # torque control: no position servo
            damping=0.2,                # light bearing friction; tune to real motor
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["rear_wheel_joint"],
            effort_limit_sim=6.28,      # GIM3510-8 stall (N·m)
            velocity_limit_sim=61.6,    # 588 RPM
            stiffness=0.0,              # torque control: no position servo
            damping=0.05,               # wheel spins near-freely; tune
        ),
    }
)
