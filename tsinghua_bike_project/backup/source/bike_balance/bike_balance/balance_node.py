#!/usr/bin/env python3
"""
Self-balancing bicycle control node — 50 Hz.

Three operating modes (parameter ``mode``):

  ``sign_check``  — Publish corrected IMU state only.  No motor commands.
                    Verify axis signs manually (tutorial 06 §7 step 2).

  ``inner_loop``  — Test the **motor's built-in speed loop** (C1 speed mode).
                    Subscribe to /bike_balance/steer_rate_ref for the target
                    steer rate (rad/s).  The motor firmware closes the speed
                    loop internally — verified in DEPLOYMENT.md step 7.

  ``full`` (default) — Complete balance control:
       Outer:  state feedback (pole or LQR) → steer_rate_ref (rad/s)
       Steer:  → C1 speed command (motor firmware speed loop)
       Rear:   speed profile → C1 speed command (motor firmware speed loop)

IMU coordinate correction (tutorial 06):
  - R_BS rotation from IMU sensor frame to bike body frame
  - Gyro bias subtraction (in sensor frame before rotation)
  - Roll rate from body rates + Euler kinematics

Motor protocol (GIM CAN V3.09b0):
  - C1 = SPEED_CTRL: target speed in rpm × 0.01 (little-endian int32)
  - CF = MOTOR_DISABLE: free state / emergency stop

Usage
-----
  ros2 run bike_balance balance_node --ros-args -p mode:=sign_check
  ros2 run bike_balance balance_node --ros-args -p mode:=inner_loop
  ros2 run bike_balance balance_node --ros-args -p mode:=full -p method:=pole
  ros2 run bike_balance balance_node --ros-args -p method:=lqr
  ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"
"""

from __future__ import annotations

import json
import math
import re
import struct
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray, String

from .bicycle_model import BicycleParameters, LinearBicycleModel
from .controller import NominalBikeController

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
_RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)       # rad/s → rpm
_C1_SCALE = 100.0                               # rpm → C1 units (rpm / 0.01)


def _rad_s_to_c1(rad_s: float) -> int:
    """Convert rad/s to C1 speed-command integer (rpm × 100, le int32)."""
    return int(round(rad_s * _RPM_PER_RAD_S * _C1_SCALE))


def _c1_to_rad_s(c1: int) -> float:
    """Convert C1 speed-command integer back to rad/s (for display)."""
    return c1 / _C1_SCALE / _RPM_PER_RAD_S


# ---------------------------------------------------------------------------
# Low-pass filter
# ---------------------------------------------------------------------------
class LowPass:
    """First-order low-pass:  y_k = alpha·x_k + (1-alpha)·y_{k-1}."""

    def __init__(self, alpha: float = 0.3, initial: float = 0.0) -> None:
        self.alpha = float(alpha)
        self.y = float(initial)

    def update(self, x: float) -> float:
        self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y

    @property
    def value(self) -> float:
        return self.y


# ---------------------------------------------------------------------------
# R_BS construction, state from IMU (tutorial 06 §5)
# ---------------------------------------------------------------------------
def _rotation_from_rpy_deg(rpy_deg):
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=float))
    return Rotation.from_euler("ZYX", [yaw, pitch, roll])


def build_bicycle_state(
    steering_angle: float,
    imu_quat_xyzw: np.ndarray,
    gyro_s: np.ndarray,
    sensor_to_bike: Rotation,
    gyro_bias_s: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Construct [steer, roll, roll_rate] from raw IMU data (tutorial 06 §5)."""
    world_from_sensor = Rotation.from_quat(np.asarray(imu_quat_xyzw, dtype=float))
    world_from_bike = world_from_sensor * sensor_to_bike.inv()
    yaw, pitch, roll = world_from_bike.as_euler("ZYX")

    p, q, r = sensor_to_bike.apply(
        np.asarray(gyro_s, dtype=float) - np.asarray(gyro_bias_s, dtype=float)
    )
    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) < 1e-6:
        raise ValueError("Pitch too close to ±90° — Euler singularity.")
    roll_rate = p + np.sin(roll) * np.tan(pitch) * q + np.cos(roll) * np.tan(pitch) * r

    state = np.array([[steering_angle], [roll], [roll_rate]])
    diag = {
        "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
        "body_rates": np.array([p, q, r]),
    }
    return state, diag


# ======================================================================
# Main node
# ======================================================================
class BalanceNode(Node):
    """50 Hz self-balancing controller — motor firmware closes speed loops."""

    def __init__(self) -> None:
        super().__init__("bike_balance")

        # ──────────────── Declare parameters ────────────────

        self.declare_parameter("mode", "full")          # sign_check | inner_loop | full
        self.declare_parameter("method", "pole")        # pole | lqr

        # CAN (GIM V3.09b0: host→motor = 1XX, motor→host = 0XX)
        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("steer_can_id", "101")
        self.declare_parameter("drive_can_id", "102")
        self.declare_parameter("steer_feedback_id", "001")
        self.declare_parameter("drive_feedback_id", "002")
        self.declare_parameter("encoder_resolution", 16384)
        self.declare_parameter("steer_gear_ratio", 1.0)
        self.declare_parameter("drive_gear_ratio", 1.0)

        # Speed profile
        self.declare_parameter("wheel_radius", 0.07)    # m
        self.declare_parameter("target_speed", 2.7)     # m/s
        self.declare_parameter("acceleration_time", 1.5) # s

        # Control tuning
        self.declare_parameter("control_dt", 0.02)        # 50 Hz
        self.declare_parameter("min_scheduling_speed", 1.0)
        self.declare_parameter("max_steer_velocity", 10.0) # rad/s outer clamp

        # Pole placement
        self.declare_parameter("pole_wc", -5.0)

        # LQR
        self.declare_parameter("lqr_Q_diag", [4.0, 100.0, 2.0])
        self.declare_parameter("lqr_R_diag", [10.0])

        # IMU (tutorial 06)
        self.declare_parameter("sensor_to_bike_rpy_deg", [0.0, 0.0, 0.0])
        self.declare_parameter("gyro_bias_dps", [0.0, 0.0, 0.0])

        # Bicycle physical model
        self.declare_parameter("bike_mass", 5.0)
        self.declare_parameter("bike_wheelbase", 0.271)
        self.declare_parameter("bike_rear_to_com", 0.117)
        self.declare_parameter("bike_com_height", 0.292)
        self.declare_parameter("bike_trail", 0.0)
        self.declare_parameter("bike_steering_axis_angle", 0.0)
        self.declare_parameter("bike_roll_inertia", -1.0)

        # Safety
        self.declare_parameter("max_roll_angle_deg", 30.0)
        self.declare_parameter("steer_angle_limit_deg", 35.0)
        self.declare_parameter("enable_on_start", False)
        self.declare_parameter("balance_start_delay", 0.1)

        # ──────────────── Read parameters ────────────────

        self._mode: str = self.get_parameter("mode").value
        if self._mode not in ("sign_check", "inner_loop", "full"):
            self.get_logger().error(f"Unknown mode '{self._mode}', falling back to 'full'")
            self._mode = "full"

        self._method: str = self.get_parameter("method").value
        self._can_iface: str = self.get_parameter("can_interface").value
        self._steer_can_id: str = self.get_parameter("steer_can_id").value
        self._drive_can_id: str = self.get_parameter("drive_can_id").value
        self._steer_fb_id: str = self.get_parameter("steer_feedback_id").value
        self._drive_fb_id: str = self.get_parameter("drive_feedback_id").value
        self._enc_res: int = self.get_parameter("encoder_resolution").value
        self._steer_gear: float = self.get_parameter("steer_gear_ratio").value
        self._drive_gear: float = self.get_parameter("drive_gear_ratio").value
        self._wheel_radius: float = self.get_parameter("wheel_radius").value
        self._target_speed: float = self.get_parameter("target_speed").value
        self._accel_time: float = self.get_parameter("acceleration_time").value
        self._control_dt: float = self.get_parameter("control_dt").value
        self._min_speed: float = self.get_parameter("min_scheduling_speed").value
        self._max_steer_vel: float = self.get_parameter("max_steer_velocity").value
        self._pole_wc: float = self.get_parameter("pole_wc").value

        self._sensor_to_bike = _rotation_from_rpy_deg(
            self.get_parameter("sensor_to_bike_rpy_deg").value)
        self._gyro_bias_s = np.deg2rad(
            np.asarray(self.get_parameter("gyro_bias_dps").value, dtype=float))

        self._max_roll_rad: float = np.deg2rad(
            self.get_parameter("max_roll_angle_deg").value)
        self._steer_angle_limit_deg: float = self.get_parameter("steer_angle_limit_deg").value
        self._enable_on_start: bool = self.get_parameter("enable_on_start").value
        self._balance_delay: float = self.get_parameter("balance_start_delay").value

        # ──────────────── Bicycle model ────────────────

        roll_iner = self.get_parameter("bike_roll_inertia").value
        bike_params = BicycleParameters(
            rear_contact_to_com=self.get_parameter("bike_rear_to_com").value,
            wheelbase=self.get_parameter("bike_wheelbase").value,
            trail=self.get_parameter("bike_trail").value,
            com_height=self.get_parameter("bike_com_height").value,
            mass=self.get_parameter("bike_mass").value,
            roll_inertia=None if roll_iner < 0 else roll_iner,
            steering_axis_angle=self.get_parameter("bike_steering_axis_angle").value,
        )
        self._bike = LinearBicycleModel(
            dt=self._control_dt, parameters=bike_params,
            track_roll=False, min_forward_speed=self._min_speed,
        )
        self._build_controller()

        # ──────────────── Runtime state ────────────────

        # IMU
        self._imu_quat_xyzw: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
        self._imu_gyro: np.ndarray = np.array([0.0, 0.0, 0.0])
        self._imu_roll: float = 0.0
        self._roll_rate: float = 0.0   # corrected
        self._body_rates: np.ndarray = np.array([0.0, 0.0, 0.0])

        # Encoders
        self._steer_angle: float = 0.0
        self._steer_rate: float = 0.0
        self._last_steer_angle: float = 0.0
        self._steer_zero_offset: int = 0
        self._steer_zero_locked: bool = False
        self._prev_steer_ts: float = 0.0
        self._rear_rate: float = 0.0
        self._last_rear_angle: float = 0.0
        self._prev_rear_ts: float = 0.0
        self._forward_speed: float = 0.0

        # Control
        self._enabled: bool = self._enable_on_start
        self._start_time: float | None = None
        self._prev_steer_rate_ref: float = 0.0
        self._prev_steer_c1: int = 0
        self._prev_drive_c1: int = 0
        self._sat_outer: bool = False

        # Ext steer_rate_ref for inner_loop mode
        self._ext_steer_rate_ref: float = 0.0

        # Jitter
        self._last_tick: float = 0.0
        self._jitter_samples: list[float] = []

        # Filters
        self._steer_rate_lp = LowPass(alpha=0.5, initial=0.0)
        self._rear_rate_lp = LowPass(alpha=0.3, initial=0.0)

        # CAN
        self._can_lock = threading.Lock()
        self._latest_steer_enc: int | None = None
        self._latest_drive_enc: int | None = None
        self._steer_enc_ts: float = 0.0
        self._drive_enc_ts: float = 0.0
        self._candump_proc: subprocess.Popen | None = None

        # ──────────────── Subscriptions ────────────────

        self.create_subscription(Float32MultiArray, "imu/euler", self._euler_cb, 10)
        self.create_subscription(Imu, "imu/data_raw", self._imu_cb, 10)
        self.create_subscription(Bool, "bike_balance/enable", self._enable_cb, 10)

        if self._mode == "inner_loop":
            self.create_subscription(
                Float32MultiArray, "bike_balance/steer_rate_ref",
                self._steer_ref_cb, 10)

        # ──────────────── Publishers ────────────────

        self._status_pub = self.create_publisher(String, "bike_balance/status", 10)
        self._debug_pub = self.create_publisher(Float32MultiArray, "bike_balance/debug", 10)

        # ──────────────── Start ────────────────

        self._start_can_reader()
        self._timer = self.create_timer(self._control_dt, self._tick)
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._last_tick = time.time()

        self.get_logger().info(
            f"BikeBalance | mode={self._mode} method={self._method} dt={self._control_dt}s | "
            f"CAN={self._can_iface} steer={self._steer_can_id} drive={self._drive_can_id} | "
            f"R_BS={self.get_parameter('sensor_to_bike_rpy_deg').value}° "
            f"bias={self.get_parameter('gyro_bias_dps').value} dps | "
            f"target_v={self._target_speed}m/s wheel_r={self._wheel_radius}m | "
            f"enabled={self._enabled}"
        )

    # ==================================================================
    # Controller
    # ==================================================================
    def _build_controller(self) -> None:
        common = dict(u_min=-self._max_steer_vel, u_max=self._max_steer_vel)
        if self._method == "pole":
            self._ctrl = NominalBikeController(
                self._bike.sys, method="place_multiple_poles",
                wc=self._pole_wc, **common)
        elif self._method == "lqr":
            Qc = np.diag(self.get_parameter("lqr_Q_diag").value)
            Rc = np.diag(self.get_parameter("lqr_R_diag").value)
            self._ctrl = NominalBikeController(
                self._bike.sys, method="lqr", Qc=Qc, Rc=Rc, **common)
        else:
            raise ValueError(f"Unknown method: {self._method}")
        self.get_logger().info(
            f"Controller: {self._method} K={np.array2string(self._ctrl.K.flatten(), precision=4)}")

    # ==================================================================
    # IMU callbacks
    # ==================================================================
    def _euler_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 1:
            self._imu_roll = float(msg.data[0])

    def _imu_cb(self, msg: Imu) -> None:
        self._imu_gyro = np.array([
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        ])
        self._imu_quat_xyzw = np.array([
            float(msg.orientation.x), float(msg.orientation.y),
            float(msg.orientation.z), float(msg.orientation.w),
        ])

    # ==================================================================
    # Enable / steer_ref
    # ==================================================================
    def _enable_cb(self, msg: Bool) -> None:
        was = self._enabled
        self._enabled = bool(msg.data)
        if self._enabled and not was:
            self._steer_rate_lp = LowPass(alpha=0.5, initial=0.0)
            self._rear_rate_lp = LowPass(alpha=0.3, initial=0.0)
            self._start_time = None
            self._sat_outer = False
            self.get_logger().info("BALANCE ENABLED")
        elif not self._enabled and was:
            self._stop_motors()
            self.get_logger().info("BALANCE DISABLED — motors stopped")

    def _steer_ref_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 1:
            self._ext_steer_rate_ref = float(msg.data[0])

    # ==================================================================
    # CAN reader  (candump subprocess + parser thread)
    # ==================================================================
    def _start_can_reader(self) -> None:
        self._candump_proc = subprocess.Popen(
            f"candump {self._can_iface}", shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._can_read_loop, daemon=True).start()
        self.get_logger().info(f"CAN reader started: candump {self._can_iface}")

    def _can_read_loop(self) -> None:
        # candump adds timestamps when stdout is a pipe (non-TTY), producing:
        #   (1234567890.123456) can0  001   [5]  C1 0D 00 00 00
        # From a TTY (no pipe) the output has no timestamp prefix:
        #   can0  001   [5]  C1 0D 00 00 00
        # The regex handles both formats by making the optional (timestamp) group.
        pattern = re.compile(
            r"(?:\([^)]+\)\s+)?\S+\s+(\S+)\s+\[\d+\]\s+(.*)")
        assert self._candump_proc is not None
        for line in self._candump_proc.stdout:
            m = pattern.match(line)
            if not m:
                continue
            can_id, hex_str = m.group(1), m.group(2).strip().replace(" ", "")
            if len(hex_str) < 8:
                continue
            try:
                # Feedback frame is: <1-byte-tag> <4-byte-LE-int32-position>.
                # Skip the tag byte (first 2 hex chars) to get the position.
                raw = struct.unpack("<i", bytes.fromhex(hex_str[2:10])[:4])[0]
            except (ValueError, struct.error):
                continue
            now = time.time()
            with self._can_lock:
                if can_id == self._steer_fb_id:
                    self._latest_steer_enc = raw
                    self._steer_enc_ts = now
                    if not self._steer_zero_locked:
                        self._steer_zero_offset = raw
                        self._steer_zero_locked = True
                        self.get_logger().info(f"Steer zero: enc={raw}")
                elif can_id == self._drive_fb_id:
                    self._latest_drive_enc = raw
                    self._drive_enc_ts = now

    # ==================================================================
    # CAN writing  (C1 = SPEED_CTRL, CF = MOTOR_DISABLE)
    # ==================================================================
    def _send_speed(self, can_id: str, rad_s: float) -> None:
        """Send C1 speed command: value = rpm × 100, little-endian int32."""
        c1 = _rad_s_to_c1(rad_s)
        data_hex = struct.pack("<i", c1).hex()
        subprocess.run(
            f"cansend {self._can_iface} {can_id}#C1{data_hex}",
            shell=True, capture_output=True)

    def _stop_motors(self) -> None:
        for cid in (self._steer_can_id, self._drive_can_id):
            self._send_speed(cid, 0.0)
            subprocess.run(
                f"cansend {self._can_iface} {cid}#CF",
                shell=True, capture_output=True)

    # ==================================================================
    # Main tick dispatcher
    # ==================================================================
    def _tick(self) -> None:
        now = time.time()
        if self._last_tick > 0:
            measured_dt = now - self._last_tick
            self._jitter_samples.append(measured_dt)
            if len(self._jitter_samples) > 250:
                self._jitter_samples.pop(0)
        self._last_tick = now
        if self._start_time is None:
            self._start_time = now

        if self._mode == "sign_check":
            self._tick_sign_check(now)
        elif self._mode == "inner_loop":
            self._tick_inner_loop(now)
        else:
            self._tick_full(now)

    # ==================================================================
    # tick: sign_check  — read sensors, correct, publish, no motors
    # ==================================================================
    def _tick_sign_check(self, now: float) -> None:
        self._update_encoders(now)
        try:
            state, diag = build_bicycle_state(
                self._steer_angle, self._imu_quat_xyzw, self._imu_gyro,
                self._sensor_to_bike, self._gyro_bias_s)
        except ValueError:
            return
        self._roll_rate = float(state[2, 0])
        self._body_rates = diag["body_rates"]
        self.get_logger().info(
            f"[SIGN_CHECK] roll={np.rad2deg(diag['roll']):.1f}° "
            f"pitch={np.rad2deg(diag['pitch']):.1f}° "
            f"yaw={np.rad2deg(diag['yaw']):.1f}° "
            f"p={np.rad2deg(self._body_rates[0]):.1f} "
            f"q={np.rad2deg(self._body_rates[1]):.1f} "
            f"r={np.rad2deg(self._body_rates[2]):.1f} dps "
            f"roll_rate={np.rad2deg(self._roll_rate):.1f} dps | "
            f"steer={np.rad2deg(self._steer_angle):.1f}°",
            throttle_duration_sec=2.0)
        corrected = Float32MultiArray()
        corrected.data = [
            float(diag["roll"]), float(diag["pitch"]), float(diag["yaw"]),
            float(self._body_rates[0]), float(self._body_rates[1]),
            float(self._body_rates[2]), float(self._steer_angle)]
        self._debug_pub.publish(corrected)

    # ==================================================================
    # tick: inner_loop  — steer C1 speed tracking with angle limit
    # ==================================================================
    def _tick_inner_loop(self, now: float) -> None:
        self._update_encoders(now)

        if not self._enabled:
            self._send_speed(self._steer_can_id, 0.0)
            self._send_speed(self._drive_can_id, 0.0)
            return

        if abs(self._imu_roll) > self._max_roll_rad:
            self.get_logger().error(
                f"INNER STOP: |roll|={np.rad2deg(self._imu_roll):.1f}° > "
                f"{np.rad2deg(self._max_roll_rad):.1f}°")
            self._enabled = False
            self._stop_motors()
            return

        ref = self._ext_steer_rate_ref

        # ==== hard steer angle limit (active force-back) ====
        # If past limit: actively command steer back toward center.
        # If within 15% of limit: block further outward motion.
        # This protects mechanical parts from over-travel — the motor
        # firmware's speed loop will resist any external force that
        # tries to push the steering past the limit.
        _limit = self._steer_angle_limit_deg                 # degrees
        _return_speed = 3.0                                  # rad/s  (gentle but firm)
        steer_deg = np.rad2deg(self._steer_angle)
        if steer_deg > _limit:
            ref = -_return_speed
            self.get_logger().warn(
                f"STEER LIMIT+: {steer_deg:.1f}° > +{_limit}° — forcing back",
                throttle_duration_sec=1.0)
        elif steer_deg < -_limit:
            ref = _return_speed
            self.get_logger().warn(
                f"STEER LIMIT-: {steer_deg:.1f}° < -{_limit}° — forcing back",
                throttle_duration_sec=1.0)
        elif steer_deg > _limit * 0.85 and ref > 0:
            ref = 0.0        # approaching +limit → block outward
        elif steer_deg < -_limit * 0.85 and ref < 0:
            ref = 0.0        # approaching -limit → block outward

        self._send_speed(self._steer_can_id, ref)
        self._send_speed(self._drive_can_id, 0.0)
        self._prev_steer_rate_ref = ref
        self._prev_steer_c1 = _rad_s_to_c1(ref)

    # ==================================================================
    # tick: full  — balance outer loop → C1 speed commands
    # ==================================================================
    def _tick_full(self, now: float) -> None:
        self._update_encoders(now)
        elapsed = now - self._start_time if self._start_time else 0.0

        # ---- stale CAN check ----
        with self._can_lock:
            steer_enc_now = self._latest_steer_enc
            steer_ts = self._steer_enc_ts
        if steer_enc_now is None or now - steer_ts > 0.5:
            if self._enabled:
                self.get_logger().warn("Steer CAN stale — disabling", throttle_duration_sec=2.0)
                self._enabled = False
                self._stop_motors()
            return

        # ---- forward speed ----
        self._forward_speed = self._wheel_radius * abs(self._rear_rate)

        # ---- speed profile ramp ----
        if self._accel_time > 0:
            speed_ref = min(elapsed / self._accel_time * self._target_speed, self._target_speed)
        else:
            speed_ref = self._target_speed
        rear_rate_ref = speed_ref / self._wheel_radius  # rad/s

        # ---- build corrected state (tutorial 06) ----
        try:
            state, diag = build_bicycle_state(
                self._steer_angle, self._imu_quat_xyzw, self._imu_gyro,
                self._sensor_to_bike, self._gyro_bias_s)
        except ValueError:
            return
        self._roll_rate = float(state[2, 0])
        self._body_rates = diag["body_rates"]
        corrected_roll = float(state[1, 0])

        # ---- gain scheduling ----
        sch_v = max(self._forward_speed, self._min_speed)
        self._bike.update_system_parameters(sch_v, min_forward_speed=self._min_speed)
        self._ctrl.update_gain()

        # ==== outer loop: state feedback → steer_rate_ref ====
        if elapsed < self._balance_delay or not self._enabled:
            steer_rate_ref = 0.0
            self._sat_outer = False
        else:
            u_raw = float(self._ctrl.step(np.array([[0.0]]), state)[0, 0])
            steer_rate_ref = float(np.clip(u_raw, -self._max_steer_vel, self._max_steer_vel))
            self._sat_outer = abs(u_raw - steer_rate_ref) > 1e-6

        # ==== hard steer angle limit (active force-back) ====
        # Same logic as inner_loop: actively force steer back when past limit,
        # and block outward motion when approaching the limit.
        _limit = self._steer_angle_limit_deg                 # degrees
        _return_speed = 3.0                                  # rad/s
        steer_deg = np.rad2deg(self._steer_angle)
        if steer_deg > _limit:
            steer_rate_ref = -_return_speed
            self.get_logger().warn(
                f"STEER LIMIT+: {steer_deg:.1f}° > +{_limit}° — forcing back",
                throttle_duration_sec=1.0)
        elif steer_deg < -_limit:
            steer_rate_ref = _return_speed
            self.get_logger().warn(
                f"STEER LIMIT-: {steer_deg:.1f}° < -{_limit}° — forcing back",
                throttle_duration_sec=1.0)
        elif steer_deg > _limit * 0.85 and steer_rate_ref > 0:
            steer_rate_ref = 0.0
        elif steer_deg < -_limit * 0.85 and steer_rate_ref < 0:
            steer_rate_ref = 0.0

        # ==== safety: emergency roll stop ====
        if abs(corrected_roll) > self._max_roll_rad:
            if self._enabled:
                self.get_logger().error(
                    f"EMERGENCY: |roll|={np.rad2deg(corrected_roll):.1f}°")
                self._enabled = False
                self._stop_motors()
            return

        # ==== send C1 speed commands (motor firmware closes speed loop) ====
        if self._enabled:
            self._send_speed(self._steer_can_id, steer_rate_ref)
            self._send_speed(self._drive_can_id, rear_rate_ref)
        else:
            self._send_speed(self._steer_can_id, 0.0)
            self._send_speed(self._drive_can_id, 0.0)

        self._prev_steer_rate_ref = steer_rate_ref
        self._prev_steer_c1 = _rad_s_to_c1(steer_rate_ref)
        self._prev_drive_c1 = _rad_s_to_c1(rear_rate_ref)

    # ==================================================================
    # Shared: encoder → angles & rates
    # ==================================================================
    def _update_encoders(self, now: float) -> None:
        with self._can_lock:
            se, de = self._latest_steer_enc, self._latest_drive_enc
            sts, dts = self._steer_enc_ts, self._drive_enc_ts

        if se is not None and self._steer_zero_locked:
            motor_rad = (se - self._steer_zero_offset) * 2.0 * math.pi / self._enc_res
            sa = motor_rad / self._steer_gear
            dt_s = max(sts - self._prev_steer_ts, 1e-6)
            if 0.0 < dt_s < 0.5:
                self._steer_rate = self._steer_rate_lp.update((sa - self._last_steer_angle) / dt_s)
            self._last_steer_angle = sa
            self._steer_angle = sa
            self._prev_steer_ts = sts

        if de is not None:
            wa = de * 2.0 * math.pi / self._enc_res / self._drive_gear
            dt_r = max(dts - self._prev_rear_ts, 1e-6)
            if 0.0 < dt_r < 0.5:
                self._rear_rate = self._rear_rate_lp.update((wa - self._last_rear_angle) / dt_r)
            self._last_rear_angle = wa
            self._prev_rear_ts = dts

    # ==================================================================
    # Status (2 Hz)
    # ==================================================================
    def _publish_status(self) -> None:
        jitter = {}
        if len(self._jitter_samples) >= 10:
            s = np.array(self._jitter_samples)
            jitter = {
                "dt_mean_ms": round(float(s.mean()) * 1000, 2),
                "dt_std_ms": round(float(s.std()) * 1000, 2),
                "dt_min_ms": round(float(s.min()) * 1000, 2),
                "dt_max_ms": round(float(s.max()) * 1000, 2)}

        status = {
            "mode": self._mode,
            "enabled": self._enabled,
            "method": self._method if self._mode == "full" else "n/a",
            "roll_deg": round(np.rad2deg(self._imu_roll), 2),
            "roll_rate_dps": round(np.rad2deg(self._roll_rate), 2),
            "body_p_dps": round(np.rad2deg(float(self._body_rates[0])), 2),
            "body_q_dps": round(np.rad2deg(float(self._body_rates[1])), 2),
            "body_r_dps": round(np.rad2deg(float(self._body_rates[2])), 2),
            "steer_angle_deg": round(np.rad2deg(self._steer_angle), 2),
            "steer_rate_dps": round(np.rad2deg(self._steer_rate), 2),
            "speed_m_s": round(self._forward_speed, 3),
            "steer_rate_ref_rad_s": round(self._prev_steer_rate_ref, 3),
            "steer_c1": self._prev_steer_c1,
            "drive_c1": self._prev_drive_c1,
            "sat_outer": self._sat_outer,
            "K": [round(float(k), 4) for k in self._ctrl.K.flatten()],
            "steer_zero_locked": self._steer_zero_locked,
            "jitter": jitter}
        msg = String()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)

        debug = Float32MultiArray()
        debug.data = [
            self._imu_roll, self._roll_rate,
            self._steer_angle, self._steer_rate,
            self._forward_speed, self._prev_steer_rate_ref,
            float(self._sat_outer),
            float(1.0 if self._enabled else 0.0),
            float(self._ctrl.K[0, 0]),
            float(self._ctrl.K[0, 1]),
            float(self._ctrl.K[0, 2])]
        self._debug_pub.publish(debug)

    # ==================================================================
    # Cleanup
    # ==================================================================
    def destroy_node(self) -> None:
        self.get_logger().info("Shutting down")
        self._stop_motors()
        if self._candump_proc:
            self._candump_proc.terminate()
            try:
                self._candump_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._candump_proc.kill()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BalanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
