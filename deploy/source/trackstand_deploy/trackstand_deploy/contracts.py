"""Pure deployment contracts shared by ROS nodes and unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


OBSERVATION_NAMES = (
    "gravity_y", "gravity_z", "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "steering_pos", "drive_pos", "steering_vel", "drive_vel", "sin_steer", "cos_steer",
)
OBSERVATION_SIZE = len(OBSERVATION_NAMES)
ACTION_SIZE = 2
POLICY_RATE_HZ = 50.0
POLICY_DT_S = 1.0 / POLICY_RATE_HZ
STEERING_ACTION_SCALE_RAD = 1.57
DRIVE_ACTION_SCALE_NM = 6.28
STEERING_KP_NM_PER_RAD = 10.0
STEERING_KD_NM_S_PER_RAD = 0.2
DRIVE_DAMPING_NM_S_PER_RAD = 0.0
STEERING_EFFORT_LIMIT_NM = 5.6
DRIVE_EFFORT_LIMIT_NM = 6.28
STEERING_VELOCITY_LIMIT_RAD_S = 23.9
DRIVE_VELOCITY_LIMIT_RAD_S = 61.6
FALL_ANGLE_RAD = 0.5236


def physx_revolute_angle(angle_rad: float) -> float:
    """Match the measured v1 PhysX coordinate branch: period 4pi, range +/-2pi.

    This is a policy adapter, not an encoder or physical steering travel limit.
    See mujoco_sim2sim/reports/20261022_v1_alignment/isaac_angle_probe.json.
    """
    if not math.isfinite(angle_rad):
        return math.nan  # v1 observation non-finite repair below converts this to zero.
    return 2.0 * math.atan2(math.sin(angle_rad / 2.0), math.cos(angle_rad / 2.0))


@dataclass
class SteeringTargetLimiter:
    """Advance once per policy action, then hold during all motor transmissions."""

    target_rad: float = 0.0

    def reset(self, measured_position_rad: float) -> None:
        if not math.isfinite(measured_position_rad):
            raise ValueError("steering reset position must be finite")
        self.target_rad = float(np.clip(measured_position_rad, -STEERING_ACTION_SCALE_RAD, STEERING_ACTION_SCALE_RAD))

    def update(self, requested_position_rad: float) -> float:
        if not math.isfinite(requested_position_rad):
            raise ValueError("steering target must be finite")
        bounded = float(np.clip(requested_position_rad, -STEERING_ACTION_SCALE_RAD, STEERING_ACTION_SCALE_RAD))
        max_step = STEERING_VELOCITY_LIMIT_RAD_S * POLICY_DT_S
        self.target_rad += float(np.clip(bounded - self.target_rad, -max_step, max_step))
        return self.target_rad


def motor_fault_clear_allowed(
    requested_enable: bool, steering_position_rad: float, clear_limit_rad: float
) -> bool:
    """Allow clearing a motor fault only while disabled and near steering zero."""
    return not requested_enable and abs(steering_position_rad) < clear_limit_rad


def wrap_to_pi(angle_rad: float) -> float:
    """Generic single-turn utility; not used by the v1 observation contract."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be four finite xyzw values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError("quaternion norm is zero")
    return quaternion / norm


def quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw quaternions."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def quaternion_inverse_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = normalize_quaternion_xyzw(quaternion)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def rotate_vector_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a three-vector by an xyzw quaternion."""
    q = normalize_quaternion_xyzw(quaternion)
    v = np.asarray(vector, dtype=np.float64)
    qv = q[:3]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q[3] * v)


def rpy_degrees_to_quaternion_xyzw(rpy_deg: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw degrees to the repository's intrinsic ZYX rotation."""
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return normalize_quaternion_xyzw(
        np.array(
            [
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ]
        )
    )


def correct_imu_sample(
    world_from_sensor_xyzw: np.ndarray,
    gyro_sensor_rad_s: np.ndarray,
    sensor_to_bike_xyzw: np.ndarray,
    gyro_bias_sensor_rad_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world-from-bike quaternion and bike-frame angular velocity."""
    world_from_sensor = normalize_quaternion_xyzw(world_from_sensor_xyzw)
    sensor_to_bike = normalize_quaternion_xyzw(sensor_to_bike_xyzw)
    world_from_bike = quaternion_multiply_xyzw(
        world_from_sensor, quaternion_inverse_xyzw(sensor_to_bike)
    )
    body_rate = rotate_vector_xyzw(
        sensor_to_bike,
        np.asarray(gyro_sensor_rad_s, dtype=np.float64)
        - np.asarray(gyro_bias_sensor_rad_s, dtype=np.float64),
    )
    return normalize_quaternion_xyzw(world_from_bike), body_rate


def build_observation(
    world_from_bike_xyzw: np.ndarray,
    body_rate_rad_s: np.ndarray,
    steering_position_rad: float,
    rear_wheel_position_rad: float,
    steering_velocity_rad_s: float,
    rear_wheel_velocity_rad_s: float,
) -> np.ndarray:
    """Build v1's 11 channels from calibrated IMU and signed encoder measurements.

    Rear position must be continuous displacement from this balance episode's
    zero. Adapt it to the observed native PhysX branch only here. Steering keeps
    its mechanically straight reference; there are no inferred linear velocities.
    """
    world_from_bike = normalize_quaternion_xyzw(world_from_bike_xyzw)
    gravity_body = rotate_vector_xyzw(
        quaternion_inverse_xyzw(world_from_bike), np.array([0.0, 0.0, -1.0])
    )
    rates = np.asarray(body_rate_rad_s, dtype=np.float64)
    if rates.shape != (3,):
        raise ValueError("body_rate_rad_s must have three values")
    steering_position_rad = float(steering_position_rad)
    sin_steer, cos_steer = ((math.sin(steering_position_rad), math.cos(steering_position_rad))
                            if math.isfinite(steering_position_rad) else (math.nan, math.nan))
    rear_wheel_position_rad = physx_revolute_angle(rear_wheel_position_rad)
    rear_wheel_velocity_rad_s = float(rear_wheel_velocity_rad_s)
    observation = np.array(
        [
            gravity_body[1],
            gravity_body[2],
            rates[0],
            rates[1],
            rates[2],
            steering_position_rad,
            rear_wheel_position_rad,
            float(steering_velocity_rad_s),
            rear_wheel_velocity_rad_s,
            sin_steer,
            cos_steer,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)


def scale_policy_action(
    raw_action: np.ndarray, max_steering_target_rad: float = STEERING_ACTION_SCALE_RAD,
) -> tuple[float, float]:
    """Validate inherently bounded Beta outputs, scale once, then clip position.

    The caller advances SteeringTargetLimiter exactly once per 50-Hz inference.
    An out-of-contract actor faults instead of being silently converted by clipping.
    """
    action = np.asarray(raw_action, dtype=np.float32)
    if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
        raise ValueError("policy action must contain two finite values")
    if not math.isfinite(max_steering_target_rad) or max_steering_target_rad <= 0.0:
        raise ValueError("steering target limit must be positive and finite")
    if max_steering_target_rad != STEERING_ACTION_SCALE_RAD:
        raise ValueError("v1 steering target limit must equal 1.57 rad")
    if np.any(np.abs(action) > 1.0):
        raise ValueError("v1 Beta policy action must be within [-1, 1]")
    steering = float(np.clip(float(action[0]) * STEERING_ACTION_SCALE_RAD,
                             -max_steering_target_rad, max_steering_target_rad))
    drive = float(action[1]) * DRIVE_ACTION_SCALE_NM
    return steering, drive


class BalanceMode(str, Enum):
    DOWN = "down"
    BALANCING = "balancing"
    FAULT_LATCHED = "fault_latched"


@dataclass
class BalanceStateMachine:
    """RC-authorized, fail-closed kickstand/balance state machine."""

    mode: BalanceMode = BalanceMode.DOWN
    rc_up: bool = False

    def set_rc(self, up: bool, ready: bool) -> BalanceMode:
        self.rc_up = bool(up)
        if not self.rc_up:
            self.mode = BalanceMode.DOWN
        elif self.mode == BalanceMode.DOWN and ready:
            self.mode = BalanceMode.BALANCING
        return self.mode

    def update_ready(self, ready: bool) -> BalanceMode:
        if self.mode == BalanceMode.BALANCING and not ready:
            self.mode = BalanceMode.FAULT_LATCHED
        elif self.mode == BalanceMode.DOWN and self.rc_up and ready:
            self.mode = BalanceMode.BALANCING
        return self.mode

    @property
    def enabled(self) -> bool:
        return self.mode == BalanceMode.BALANCING
