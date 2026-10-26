"""Trackstand-Balance-v1 contract, transcribed from the Isaac task on 2026-10-22.

Keep a lightweight, Isaac-free runtime. The parity tests compare these values
and equations against the actual training sources; do not import Isaac Sim on Pi.
"""

import math
from pathlib import Path

TASK = "Trackstand-Balance-v1"
OBSERVATIONS = (
    "gravity_y", "gravity_z", "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "steering_pos", "drive_pos", "steering_vel", "drive_vel", "sin_steer", "cos_steer",
)
PHYSICS_DT = 0.0025
CONTROL_DT = 0.02
DECIMATION = 8
MAX_STEPS = 1500
SPAWN_Z = -0.016
STEER_SCALE = STEER_POSITION_LIMIT = 1.57
DRIVE_SCALE = DRIVE_EFFORT_LIMIT = 6.28
STEER_EFFORT_LIMIT = 5.6
STEER_SPEED_LIMIT = STEER_SLEW_RATE = 23.9
DRIVE_SPEED_LIMIT = 61.6
STEER_KP, STEER_KD = 10.0, 0.2
DRIVE_KP, DRIVE_KD = 0.0, 0.0
STEER_FRICTION = 0.046036056104
STEER_VISCOUS_FRICTION = 0.000910713492
DRIVE_FRICTION = 0.106
DRIVE_VISCOUS_FRICTION = 0.0  # Resolved simulator default, not proof physical B is exactly zero.
OBS_NOISE_STD = (0.001, 0.001, 0.0012, 0.0012, 0.0012, 0., 0., 0., 0., 0., 0.)
FALL_ANGLE = YAW_LIMIT = 0.5236
DRIFT_LIMIT = 0.5
WHEEL_NET_LIMIT = 7.2
MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED, MAX_JOINT_SPEED = 20.0, 50.0, 300.0
TERMINATIONS = ("time_out", "fallen", "diverged", "drifted", "yaw_deviation", "wheel_net")
DR_MASS = (0.8, 1.2)
DR_COM = 0.005
DR_FRICTION = (0.5, 0.8)
GROUND_FRICTION, NOMINAL_FRICTION = 1.0, 0.65
DR_FRICTION_BUCKETS = 32
DR_GAINS = (0.9, 1.1)
DR_INIT_ROLL, DR_INIT_ROLL_RATE = (-0.10, 0.10), (-0.25, 0.25)
DR_GYRO_RES, DR_ACCEL_RES = 0.0002, 0.002

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MJCF = REPO_ROOT / "mujoco_sim2sim/models/trackstand_rl_0916_bike/trackstand_rl_0916.xml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_00-41-38/model_75300.pt"
)


def physics_timing(physics_hz=None):
    """Resolve physics/PD timing without changing the fixed 50-Hz policy clock.

    Require an integer number of physics steps per policy interval rather than
    rounding the requested rate and silently changing elapsed simulation time.
    None retains the default (including existing process-local diagnostics).
    """
    hz = 1.0 / PHYSICS_DT if physics_hz is None else float(physics_hz)
    steps = hz * CONTROL_DT
    if not math.isfinite(steps) or steps < 1 or not math.isclose(steps, round(steps), rel_tol=0., abs_tol=1e-9):
        raise ValueError("--physics_hz must be a positive multiple of 50 Hz (for example 400, 800, 1600)")
    decimation = round(steps)
    return CONTROL_DT / decimation, decimation
