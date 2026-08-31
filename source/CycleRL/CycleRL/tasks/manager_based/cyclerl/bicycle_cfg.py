import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
BIKE_USD_PATH = os.path.join(CURRENT_DIR, "assets", "xuan_bike_2kg5", "xuan_bike_2kg5.usda")
THUBIKE_USD_PATH = os.path.join(CURRENT_DIR, "assets", "thubike2", "thubike2.usda")
NEW_BIKE_USD_PATH = os.path.join(CURRENT_DIR, "assets", "new_bike2", "new_bike2.usda")

# =========================================================================
# 1. Common Base Configuration (Shared physics, limits, and USD)
# =========================================================================
XUAN_BIKE_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BIKE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), 
        rot=(0.0, 0.0, 0.0, 1.0), # (x, y, z, w)
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    # Actuators left empty to be defined by task-specific configs
    actuators={}, 
)

# =========================================================================
# 2. Standard CycleRL Configuration (Steering + Rear Drive)
# =========================================================================
XUAN_BIKE_STANDARD_CFG = XUAN_BIKE_BASE_CFG.replace(
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_7"],
            effort_limit=6.0,      # The physical 60kg-cm limit
            velocity_limit=7.0,    # The physical servo speed limit
            stiffness=100.0,       # High stiffness mimics the servo holding its position
            damping=10.0,
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_9"], 
            effort_limit=25.0,       # 10:1 ratio from 5Nm
            velocity_limit=20.0,
            stiffness=0.0,
            damping=5.0, # Velocity control
        ),
    }
)

# =========================================================================
# 3. Flywheel Configuration
# =========================================================================
XUAN_BIKE_FLYWHEEL_CFG = XUAN_BIKE_BASE_CFG.replace(
    actuators={
        "flywheel": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_11"], 
            # 5.2 Nm raw motor torque * 4 (gear ratio)
            effort_limit=23.0,    
            # 400 rad/s raw motor speed / 4 (gear ratio)
            velocity_limit=200.0,
            stiffness=0.0,       # 0.0 for pure effort (torque) control
            damping=0.063,         # 20.8/100
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_7"],
            effort_limit=6.0,      # The physical 60kg-cm limit
            velocity_limit=7.0,    # The physical servo speed limit
            stiffness=100.0,       # High stiffness mimics the servo holding its position
            damping=10.0,
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_9"], 
            effort_limit=25.0,       # 5:1 ratio from 5Nm
            velocity_limit=100.0,
            stiffness=0.0,
            damping=5.0, # Velocity control
        ),
    }
)

# =========================================================================
# 4. PID Flywheel Configuration
# =========================================================================
XUAN_BIKE_PID_FLYWHEEL_CFG = XUAN_BIKE_BASE_CFG.replace(
    actuators={
        "flywheel": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_11"], 
            # 5.2 Nm raw motor torque * 4 (gear ratio)
            effort_limit=23.0,    
            # 400 rad/s raw motor speed / 4 (gear ratio)
            velocity_limit=200.0,
            stiffness=0.0,       # 0.0 for pure effort (torque) control
            damping=0.0,         # 20.8/100
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_7"],
            effort_limit=6.0,      # The physical 60kg-cm limit
            velocity_limit=7.0,    # The physical servo speed limit
            stiffness=100.0,       # High stiffness mimics the servo holding its position
            damping=10.0,
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_9"],
            effort_limit=32.5,       # 5:1 ratio from 5Nm
            velocity_limit=20.0,
            stiffness=0.0,
            damping=5.0, # Velocity control
        ),
    }
)

# =========================================================================
# 5. Thubike2 Base Configuration
# =========================================================================
THUBIKE_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=THUBIKE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.02),
        rot=(0.0, 0.0, 0.0, 1.0), # (x, y, z, w)
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={},
)

# =========================================================================
# 6. Thubike2 Standard Configuration (Steering + Rear Drive)
# =========================================================================
THUBIKE_STANDARD_CFG = THUBIKE_BASE_CFG.replace(
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_9"],
            effort_limit=5.6,       # GIM4310-10 V1.1 stall (N·m)
            velocity_limit=23.9,    # 228 RPM
            stiffness=25.0,         # same SI PD as MIT kp
            damping=2.0,            # same SI PD as MIT kd
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_8"],
            effort_limit=6.28,      # GIM3510-8 stall (N·m)
            velocity_limit=61.6,    # 588 RPM
            stiffness=0.0,          # velocity mode: τ = Kd*(ω*−ω)
            damping=8.0,
        ),
    }
)

# =========================================================================
# 7. New Bike Base Configuration
# =========================================================================
NEW_BIKE_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=NEW_BIKE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,  # 0 gave inaccurate velocities → NaN blow-ups
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

# =========================================================================
# 8. New Bike Standard Configuration (Steering + Rear Drive)
#     PD and limits: config/motor_contract.yaml (GIM4310-10 V1.1 / GIM3510-8)
# =========================================================================
NEW_BIKE_STANDARD_CFG = NEW_BIKE_BASE_CFG.replace(
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["steering_joint"],
            # *_sim: the bare effort_limit/velocity_limit are dropped for IMPLICIT
            # actuators on Isaac Lab 7.2.0 (see NEW_BIKE_TORQUE_CFG below). Under
            # position (steering) / velocity (drive) control the action clips bound
            # the joints anyway, so enabling these nominal limits changes nothing
            # physically — it just uses the fields this version honours.
            effort_limit_sim=5.6,   # GIM4310-10 V1.1 stall (N·m)
            velocity_limit_sim=23.9,    # 228 RPM
            stiffness=10.0,         # same SI PD as MIT kp
            damping=0.2,            # same SI PD as MIT kd
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["rearwheel_joint"],
            effort_limit_sim=6.28,  # GIM3510-8 stall (N·m)
            velocity_limit_sim=61.6,    # 588 RPM
            stiffness=0.0,          # velocity mode: τ = Kd*(ω*−ω)
            damping=5.0,            # match Xuan bike; 8.0 saturated at ~0.79 rad/s error
        ),
    }
)

# =========================================================================
# 8b. New Bike Hybrid Configuration (position steering + torque drive)
#     Steering keeps the STANDARD position servo (stiffness/damping PD), the
#     drive motor runs in TORQUE mode (stiffness 0), like NEW_BIKE_TORQUE_CFG.
# =========================================================================
NEW_BIKE_HYBRID_CFG = NEW_BIKE_BASE_CFG.replace(
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["steering_joint"],
            effort_limit_sim=5.6,   # GIM4310-10 V1.1 stall (N·m)
            velocity_limit_sim=23.9,    # 228 RPM
            stiffness=10.0,         # position servo: same SI PD as MIT kp
            damping=0.2,            # same SI PD as MIT kd
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["rearwheel_joint"],
            effort_limit_sim=6.28,  # GIM3510-8 stall (N·m)
            velocity_limit_sim=61.6,    # 588 RPM
            stiffness=0.0,          # torque control: no position servo
            damping=0.05,           # light bearing friction; tune
        ),
    }
)

# =========================================================================
# 9. New Bike Torque Configuration (direct effort/torque control)
#     Both motors driven by torque: stiffness=0 so a JointEffortAction is not
#     fought by a PD servo. Matches GM4310 / GIM3510 CAN torque mode. Effort
#     limits = real stall torques; damping = light bearing/back-EMF friction.
# =========================================================================
NEW_BIKE_TORQUE_CFG = NEW_BIKE_BASE_CFG.replace(
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["steering_joint"],
            effort_limit_sim=5.6,       # GIM4310-10 V1.1 stall (N·m)
            velocity_limit_sim=23.9,    # 228 RPM
            stiffness=0.0,              # torque control: no position servo
            damping=0.2,                # light bearing friction; tune to real motor
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["rearwheel_joint"],
            # *_sim, NOT the bare effort_limit/velocity_limit: for IMPLICIT actuators
            # Isaac Lab ignores the bare velocity_limit (it warns, then drops it), so
            # under pure torque control nothing bounded the wheel — measured 2420 rad/s
            # (39x this limit), which fed a 2420-magnitude feature into the unnormalised
            # critic, blew the value loss to 1e4/NaN, and NaN'd every policy weight.
            # Only *_sim reaches PhysX and is actually enforced.
            effort_limit_sim=6.28,      # GIM3510-8 stall (N·m)
            velocity_limit_sim=61.6,    # 588 RPM
            stiffness=0.0,              # torque control: no position servo
            damping=0.05,               # wheel spins near-freely; tune
        ),
    }
)