"""Pure locomotive deployment contract: CycleRL riding policies on this bike's hardware.

The trained tensor contract is kept exactly (CycleRL-Trackstand-Cylinder-Paper-v0):
8 observations, 2 raw Gaussian-mean actions, 50 Hz. Only the mapping from this
bike's sensors/actuators onto that tensor lives here; per-model scales/ranges
come from models/locomotive_policy_manifest.json, generated from each run's
saved env.yaml by isaaclab_train/scripts/deployment/export_cyclerl_locomotive_onnx.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math

import numpy as np

from .contracts import quaternion_inverse_xyzw, normalize_quaternion_xyzw, rotate_vector_xyzw


OBSERVATION_NAMES = (
    "gravity_y", "ang_vel_x", "ang_vel_y", "ang_vel_z", "speed", "steer", "vel_cmd", "yaw_rate_cmd",
)
OBSERVATION_SIZE = len(OBSERVATION_NAMES)
ACTION_NAMES = ("steering_position", "rear_wheel_velocity")
ACTION_SIZE = len(ACTION_NAMES)
POLICY_RATE_HZ = 50.0
POLICY_DT_S = 1.0 / POLICY_RATE_HZ
# Trained JointVelocityAction clip; also the C1 backend's hard bound (motor limit 61.6).
DRIVE_VELOCITY_LIMIT_RAD_S = 45.0
MANIFEST_FILENAME = "locomotive_policy_manifest.json"


@dataclass(frozen=True)
class LocomotiveModelSpec:
    onnx: str
    onnx_sha256: str
    rank: int
    recipe: str
    run: str
    steering_scale_rad: float
    steering_clip_rad: float
    drive_scale_rad_s: float
    drive_clip_rad_s: float
    wheel_radius_m: float
    vel_cmd_min_m_s: float
    vel_cmd_max_m_s: float
    yaw_rate_cmd_max_rad_s: float
    initial_speed_min_m_s: float
    lean_termination_rad: float
    reference_observations: tuple
    reference_raw_actions: tuple
    reference_tolerance: float


def _symmetric(bounds, name: str) -> float:
    low, high = (float(value) for value in bounds)
    if not (math.isfinite(low) and math.isfinite(high) and high > 0 and math.isclose(low, -high)):
        raise ValueError(f"{name} must be a symmetric finite range")
    return high


def load_model_spec(manifest_path: str, onnx_filename: str) -> LocomotiveModelSpec:
    """Return one validated manifest entry; reject anything outside the supported contract."""
    with open(manifest_path) as stream:
        manifest = json.load(stream)
    entries = [entry for entry in manifest["models"] if entry["onnx"] == onnx_filename]
    if len(entries) != 1:
        raise ValueError(f"{onnx_filename} is not exactly one entry in {manifest_path}")
    entry = entries[0]
    contract = entry["contract"]
    if (tuple(contract["observation_names"]) != OBSERVATION_NAMES
            or tuple(contract["action_names"]) != ACTION_NAMES
            or entry["observation_size"] != OBSERVATION_SIZE or entry["action_size"] != ACTION_SIZE
            or not entry.get("weights_embedded") or not math.isclose(contract["policy_rate_hz"], POLICY_RATE_HZ)):
        raise ValueError(f"{onnx_filename}: manifest does not describe the 8-in/2-out 50 Hz contract")
    steering_clip = _symmetric(contract["steering_action_clip_rad"], "steering clip")
    drive_clip = _symmetric(contract["drive_action_clip_rad_s"], "drive clip")
    yaw_max = _symmetric(contract["yaw_rate_cmd_range_rad_s"], "yaw-rate command range")
    vel_min, vel_max = (float(value) for value in contract["vel_cmd_range_m_s"])
    spec = LocomotiveModelSpec(
        onnx=entry["onnx"], onnx_sha256=entry["onnx_sha256"], rank=int(entry["rank"]),
        recipe=entry["recipe"], run=entry["run"],
        steering_scale_rad=float(contract["steering_action_scale_rad"]), steering_clip_rad=steering_clip,
        drive_scale_rad_s=float(contract["drive_action_scale_rad_s"]), drive_clip_rad_s=drive_clip,
        wheel_radius_m=float(contract["wheel_radius_m"]),
        vel_cmd_min_m_s=vel_min, vel_cmd_max_m_s=vel_max, yaw_rate_cmd_max_rad_s=yaw_max,
        initial_speed_min_m_s=float(contract["initial_root_velocity_x_m_s"][0]),
        lean_termination_rad=float(contract["lean_termination_rad"]),
        reference_observations=tuple(map(tuple, entry["reference_io"]["observation"])),
        reference_raw_actions=tuple(map(tuple, entry["reference_io"]["raw_action"])),
        reference_tolerance=float(entry["reference_io"]["tolerance"]),
    )
    values = (spec.steering_scale_rad, spec.drive_scale_rad_s, spec.wheel_radius_m, vel_min, vel_max,
              spec.initial_speed_min_m_s, spec.lean_termination_rad)
    if (not all(math.isfinite(value) and value > 0 for value in values) or not vel_min < vel_max
            or spec.steering_clip_rad > math.radians(90.0) or spec.drive_clip_rad_s > DRIVE_VELOCITY_LIMIT_RAD_S):
        raise ValueError(f"{onnx_filename}: manifest scales/ranges are outside deployment limits")
    return spec


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_observation(
    world_from_bike_xyzw: np.ndarray,
    body_rate_rad_s: np.ndarray,
    steering_position_rad: float,
    rear_wheel_velocity_rad_s: float,
    vel_cmd_m_s: float,
    yaw_rate_cmd_rad_s: float,
    wheel_radius_m: float,
) -> np.ndarray:
    """Map calibrated IMU, signed encoders and RC commands onto CycleRL's 8 channels.

    gravity_y is projected_gravity_b[1] (= -sin(roll) for pure roll); gyro is the
    bike-frame body rate (root_ang_vel_b); speed is the no-slip rear-wheel estimate
    of root_lin_vel_b[0]; steer is wrapped to [-pi, pi] as in training. Non-finite
    inputs fail closed instead of being silently zeroed.
    """
    world_from_bike = normalize_quaternion_xyzw(world_from_bike_xyzw)
    gravity_body = rotate_vector_xyzw(quaternion_inverse_xyzw(world_from_bike), np.array([0.0, 0.0, -1.0]))
    rates = np.asarray(body_rate_rad_s, dtype=np.float64)
    if rates.shape != (3,):
        raise ValueError("body_rate_rad_s must have three values")
    steer = float(steering_position_rad)
    observation = np.array(
        [
            gravity_body[1], rates[0], rates[1], rates[2],
            float(wheel_radius_m) * float(rear_wheel_velocity_rad_s),
            math.atan2(math.sin(steer), math.cos(steer)),
            float(vel_cmd_m_s), float(yaw_rate_cmd_rad_s),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(observation)):
        raise ValueError("locomotive observation contains non-finite values")
    return observation


def scale_policy_action(raw_action: np.ndarray, spec: LocomotiveModelSpec) -> tuple[float, float]:
    """Apply the trained JointPosition/JointVelocity scale, then clip, exactly once."""
    action = np.asarray(raw_action, dtype=np.float32)
    if action.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action)):
        raise ValueError("policy action must contain two finite values")
    steering = float(np.clip(float(action[0]) * spec.steering_scale_rad, -spec.steering_clip_rad,
                             spec.steering_clip_rad))
    rear = float(np.clip(float(action[1]) * spec.drive_scale_rad_s, -spec.drive_clip_rad_s,
                         spec.drive_clip_rad_s))
    return steering, rear


def normalized_stick(value: float, deadband: float) -> float:
    """Clip an RC axis to [-1, 1] and remove a symmetric centre deadband."""
    value = max(-1.0, min(1.0, float(value)))
    if abs(value) <= deadband:
        return 0.0
    return math.copysign((abs(value) - deadband) / (1.0 - deadband), value)


@dataclass
class CommandShaper:
    """RC sticks -> (v*, yaw-rate*) inside the model's trained command ranges.

    Throttle full-down maps to the trained minimum speed and full-up to
    min(trained maximum, max_speed_m_s). v* is rate limited; yaw-rate is direct.
    """

    vel_min_m_s: float
    vel_max_m_s: float
    yaw_rate_max_rad_s: float
    accel_m_s2: float
    deadband: float = 0.05
    yaw_stick_sign: float = -1.0  # Standard RC: stick right (high pulse) is a right (negative z) turn.
    vel_cmd_m_s: float | None = None

    def __post_init__(self):
        if (not 0.0 < self.vel_min_m_s < self.vel_max_m_s or self.yaw_rate_max_rad_s <= 0.0
                or self.accel_m_s2 <= 0.0 or not 0.0 <= self.deadband < 1.0
                or self.yaw_stick_sign not in (-1.0, 1.0)):
            raise ValueError("invalid locomotive command shaping configuration")

    def reset(self, start_m_s: float) -> None:
        self.vel_cmd_m_s = max(self.vel_min_m_s, min(self.vel_max_m_s, float(start_m_s)))

    def target_speed(self, throttle: float) -> float:
        """Unslewed stick speed target inside [vel_min, vel_max]."""
        if not math.isfinite(throttle):
            raise ValueError("RC throttle must be finite")
        fraction = (max(-1.0, min(1.0, float(throttle))) + 1.0) / 2.0
        return self.vel_min_m_s + fraction * (self.vel_max_m_s - self.vel_min_m_s)

    def update(self, throttle: float, yaw: float, dt_s: float = POLICY_DT_S) -> tuple[float, float]:
        if not all(map(math.isfinite, (throttle, yaw, dt_s))):
            raise ValueError("RC sticks must be finite")
        wanted = self.target_speed(throttle)
        if self.vel_cmd_m_s is None:
            self.reset(wanted)
        step = self.accel_m_s2 * dt_s
        self.vel_cmd_m_s += max(-step, min(step, wanted - self.vel_cmd_m_s))
        yaw_rate = self.yaw_stick_sign * normalized_stick(yaw, self.deadband) * self.yaw_rate_max_rad_s
        return self.vel_cmd_m_s, yaw_rate


class LocomotiveMode(str, Enum):
    DOWN = "down"
    LAUNCH = "launch"          # Stands down, steering straight, rear speed ramps up.
    RIDING = "riding"          # Stands up, policy steers and drives.
    FAULT_LATCHED = "fault_latched"


@dataclass
class LocomotiveStateMachine:
    """RC-authorized, fail-closed launch -> ride sequence.

    Training episodes start rolling (>= initial_speed_min), so the policy takes
    over only once the measured speed reaches wake_speed. Leaving the riding
    envelope (too slow) or losing readiness latches a fault until RC down.
    """

    wake_speed_m_s: float
    min_riding_speed_m_s: float
    mode: LocomotiveMode = LocomotiveMode.DOWN
    rc_up: bool = False
    fault_reasons: tuple = ()

    def __post_init__(self):
        if not 0.0 < self.min_riding_speed_m_s < self.wake_speed_m_s:
            raise ValueError("require 0 < min_riding_speed < wake_speed")

    def set_rc(self, up: bool, ready: bool) -> LocomotiveMode:
        self.rc_up = bool(up)
        if not self.rc_up:
            self.mode, self.fault_reasons = LocomotiveMode.DOWN, ()
        elif self.mode == LocomotiveMode.DOWN and ready:
            self.mode = LocomotiveMode.LAUNCH
        return self.mode

    def latch(self, *reasons: str) -> None:
        if self.mode in (LocomotiveMode.LAUNCH, LocomotiveMode.RIDING):
            self.mode, self.fault_reasons = LocomotiveMode.FAULT_LATCHED, tuple(reasons)

    def update(self, ready: bool, reasons, speed_m_s: float) -> LocomotiveMode:
        if self.enabled and not ready:
            self.latch(*reasons)
        elif self.mode == LocomotiveMode.DOWN and self.rc_up and ready:
            self.mode = LocomotiveMode.LAUNCH
        elif self.mode == LocomotiveMode.LAUNCH and speed_m_s >= self.wake_speed_m_s:
            self.mode = LocomotiveMode.RIDING
        elif self.mode == LocomotiveMode.RIDING and speed_m_s < self.min_riding_speed_m_s:
            self.latch("speed_below_riding_envelope")
        return self.mode

    @property
    def enabled(self) -> bool:
        return self.mode in (LocomotiveMode.LAUNCH, LocomotiveMode.RIDING)

    @property
    def riding(self) -> bool:
        return self.mode == LocomotiveMode.RIDING
