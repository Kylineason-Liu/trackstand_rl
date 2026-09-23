#!/usr/bin/env python3
"""
Cascaded-PID self-balancing bicycle control node — 50 Hz.

Reads pre-processed sensor state from three topic publishers and outputs
CAN Q-axis current commands (C0) to the steering motor and CAN speed
commands (C1) to the drive motor via python-can.

Architecture (what runs where)
------------------------------
Hardware                Node (ROS 2)           Topic            Consumer
────────────────────────────────────────────────────────────────────────────
YahBoom IMU (/dev/ttyUSB0) → roll_publisher  → /roll_info  ─┐
GIM steer motor (CAN 101/001) ↔ steer_publisher → /steer_info ─┼─ balance_node
GIM drive motor (CAN 102/002) ↔ rear_publisher  → /rear_info  ─┘
SBUS RC receiver (/dev/ttyUSB1) → rc_publisher → /rc_info  ───┘  (vctrl="rc")
                                              balance_node → CAN 101 (C0 steer torque)
                                              balance_node → CAN 102 (C1 drive speed)

Subscribes
----------
  /roll_info  (Float64MultiArray)  — [roll, pitch, yaw, roll_rate, p, q, r]
  /steer_info (Float64MultiArray)  — [steer_angle (rad), steer_rate (rad/s)]
  /rear_info  (Float64MultiArray)  — [rear_rate (rad/s)]
  /rc_info    (Float64MultiArray)  — [drive_rpm]           (only when vctrl="rc")
  /bike_balance/enable (Bool)      — arm / disarm the controller
  /new_balance/steer_rate_ref (Float64MultiArray) — steer rate ref (inner_loop only)

Publishes
---------
  /new_balance/status (String)            JSON status at 2 Hz
  /new_balance/debug (Float64MultiArray)  Debug floats at 50 Hz
  /steer_cmd (Float64MultiArray)          [q_current_a] at 50 Hz

Parameters (also see config/cascaded_params.yaml for full list)
---------------------------------------------------------------
  mode (str)           : sign_check | inner_loop | full   (default: full)
  method (str)         : pole | lqr                      (default: pole)
  vctrl (str)          : "fixed" | "rc"                    (default: fixed)
  can_interface (str)  : SocketCAN interface              (default: can0)
  steer_can_id (str)   : Host → steer motor CAN ID         (default: 101)
  drive_can_id (str)   : Host → drive motor CAN ID         (default: 102)

  PID parameters (all may be overridden; defaults are extracted from K):
    outer_kp, outer_ki, outer_kd, outer_max_roll_rad
    mid_kp,   mid_ki,   mid_kd,   mid_max_rate_rad_s
    inner_kp, inner_ki, inner_kd, inner_max_q_current_a


╔══════════════════════════════════════════════════════════════════════════╗
║                      CASCADED PID CONTROL ARCHITECTURE                   ║
╚══════════════════════════════════════════════════════════════════════════╝

  Outer loop  (steer velocity → roll angle):
      e_outer = 0 − δ̇_measured
      φ_ref   = PID_outer(e_outer)

  Middle loop (roll angle → roll rate):
      e_mid   = φ_ref − φ_measured
      φ̇_ref   = PID_mid(e_mid)

  Inner loop  (roll rate → Q-axis current):
      e_inner = φ̇_ref − φ̇_measured
      I_q     = PID_inner(e_inner)   →  sent as C0 Q-axis current

  PID gains are extracted from the LQR/pole-placement K vector:

      K = [k₁, k₂, k₃]  where  u = k₁·e_δ + k₂·e_φ + k₃·e_φ̇

  Heuristic mapping (configurable overrides):
      inner_kp  = |k₃| · steer_torque_constant    (roll rate → Q-axis current, A per rad/s)
      mid_kp    = |k₂ / k₃|   (clamped)            (roll angle → roll rate)
      outer_kp  = |k₁ / k₂|   (clamped)            (steer vel → roll angle)

  Rear motor control is IDENTICAL to main_balance.py (C1 speed commands).


╔══════════════════════════════════════════════════════════════════════════╗
║                           MODE 1 — sign_check                           ║
╚══════════════════════════════════════════════════════════════════════════╝

  (Identical to main_balance.py — see that file for full documentation.)


╔══════════════════════════════════════════════════════════════════════════╗
║                         MODE 2 — inner_loop                             ║
╚══════════════════════════════════════════════════════════════════════════╝

  (Identical to main_balance.py — direct steer C1 speed control.)


╔══════════════════════════════════════════════════════════════════════════╗
║                     MODE 3a / 3b — full (cascaded PID)                  ║
╚══════════════════════════════════════════════════════════════════════════╝

  Purpose: Self-balancing via cascaded PID.  The steer motor receives
  C0 (Q-axis current / torque) commands; the drive motor receives C1
  (speed) commands — identical to main_balance.py.

  Step-by-step
  ────────────
  1-3. Same as main_balance.py (start publishers for IMU, steer, rear).

  4. Terminal 4: balance node (full mode)
         ros2 run new_balance cascaded_balance_node --ros-args -p mode:=full -p vctrl:=fixed

     The node builds a pole-placement (or LQR) controller internally to
     extract K = [k₁,k₂,k₃] from the linearised bike model, then derives
     the three cascaded PID proportional gains from that vector.

  5. Terminal 5: enable balancing
         ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"

  6. Monitor:
         ros2 topic echo /new_balance/status

     Key additions vs main_balance.py:
       "cascaded"          — true
       "outer_pid"         — {kp, ki, kd, e, ref, out}
       "mid_pid"           — {kp, ki, kd, e, ref, out}
       "inner_pid"         — {kp, ki, kd, e, ref, out}
"""

from __future__ import annotations

import json
import math
import struct
import time

import can
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray, String

from nominal_bike_control.bicycle_model import BicycleParameters, LinearBicycleModel
from nominal_bike_control.controller import NominalBikeController

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
_RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)
_C1_SCALE = 100.0
_STEER_KT = 1.0  # Nm/A  — torque constant of the GIM3510 steer motor (verified via B0)


def _rad_s_to_c1(rad_s: float) -> int:
    """Convert rad/s to C1 speed units (rpm × 100)."""
    return int(round(rad_s * _RPM_PER_RAD_S * _C1_SCALE))


# ======================================================================
# PID controller with anti-windup
# ======================================================================
class PID:
    """Simple discrete PID with back-calculation anti-windup.

    Parameters
    ----------
    kp, ki, kd : float
        Proportional, integral, derivative gains.
    output_min, output_max : float
        Output clamping limits.
    """

    def __init__(
        self,
        kp: float = 0.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_min: float = -float("inf"),
        output_max: float = float("inf"),
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.reset()

    def reset(self) -> None:
        """Clear integrator and previous-error state."""
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._last_output: float = 0.0

    def step(self, error: float, dt: float) -> float:
        """Evaluate one PID step.

        Returns the clamped output.  The integrator uses back-calculation
        (conditional integration) to prevent wind-up.
        """
        # Proportional
        p = self.kp * error

        # Integral (trapezoidal, with back-calculation)
        if dt > 0:
            self._integral += error * dt
        i = self.ki * self._integral

        # Derivative (filtered by dt, clamped to avoid spikes on NaN)
        if dt > 1e-9:
            d = self.kd * (error - self._prev_error) / dt
        else:
            d = 0.0
        self._prev_error = error

        output = p + i + d

        # Clamp and clear integral on saturation
        if output > self.output_max:
            self._integral = 0.0
            output = float(self.output_max)
        elif output < self.output_min:
            self._integral = 0.0
            output = float(self.output_min)

        self._last_output = output
        return output

    @property
    def integral(self) -> float:
        return self._integral

    @property
    def last_output(self) -> float:
        return self._last_output


# ======================================================================
# Main node
# ======================================================================
class BalanceNode(Node):
    """50 Hz cascaded-PID self-balancing controller — reads topics, writes CAN."""

    def __init__(self) -> None:
        super().__init__("cascaded_balance_node")

        # ════════════════ Parameters ════════════════
        self.declare_parameter("mode", "full")
        self.declare_parameter("method", "pole")
        self.declare_parameter("vctrl", "fixed")

        # CAN
        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("steer_can_id", "101")
        self.declare_parameter("drive_can_id", "102")

        # Speed profile
        self.declare_parameter("wheel_radius", 0.07)
        self.declare_parameter("target_speed", 1.0)
        self.declare_parameter("acceleration_time", 1.5)
        self.declare_parameter("drive_direction_invert", True)
        self.declare_parameter("steer_direction_invert", True)
        self.declare_parameter("rc_max_rpm",
                             2.7 / (2.0 * math.pi * 0.07) * 60.0)

        # Control
        self.declare_parameter("control_dt", 0.01)
        self.declare_parameter("min_scheduling_speed", 1.0)
        self.declare_parameter("max_steer_velocity", 10.0)

        # Pole
        self.declare_parameter("pole_wc", -14.3)

        # LQR
        self.declare_parameter("lqr_Q_diag", [4.0, 100.0, 2.0])
        self.declare_parameter("lqr_R_diag", [10.0])

        # Bike model
        self.declare_parameter("bike_mass", 2.2370)
        self.declare_parameter("bike_wheelbase", 0.2900)
        self.declare_parameter("bike_rear_to_com", 0.1350)
        self.declare_parameter("bike_com_height", 0.1400)
        self.declare_parameter("bike_trail", 0.0100)
        self.declare_parameter("bike_steering_axis_angle", 0.1745)
        self.declare_parameter("bike_roll_inertia", 0.0438)

        # Safety
        self.declare_parameter("max_roll_angle_deg", 30.0)
        self.declare_parameter("steer_angle_limit_deg", 30.0)
        self.declare_parameter("enable_on_start", False)
        self.declare_parameter("balance_start_delay", 0.1)

        # ── Cascaded PID parameters (overridable; defaults from K extraction) ──
        self.declare_parameter("outer_kp", -1.0)   # <0 → compute from K
        self.declare_parameter("outer_ki", 0.0)
        self.declare_parameter("outer_kd", 0.0)
        self.declare_parameter("outer_max_roll_rad", 0.3)   # ~17°

        self.declare_parameter("mid_kp", -1.0)
        self.declare_parameter("mid_ki", 0.0)
        self.declare_parameter("mid_kd", 0.0)
        self.declare_parameter("mid_max_rate_rad_s", 5.0)    # rad/s

        self.declare_parameter("inner_kp", -1.0)
        self.declare_parameter("inner_ki", 0.0)
        self.declare_parameter("inner_kd", 0.0)
        self.declare_parameter("inner_max_q_current_a", 2.0)  # A → ≈ 2 Nm

        # ── Gain-scheduling toggle ──
        self.declare_parameter("pid_gain_scheduling", False)
        self.declare_parameter("steer_torque_constant", _STEER_KT)

        # ════════════════ Read parameters ════════════════
        self._mode: str = self.get_parameter("mode").value
        if self._mode not in ("sign_check", "inner_loop", "full"):
            self.get_logger().error(f"Unknown mode '{self._mode}', falling back to 'full'")
            self._mode = "full"

        self._method: str = self.get_parameter("method").value
        self._vctrl: str = self.get_parameter("vctrl").value
        self._can_iface: str = self.get_parameter("can_interface").value
        self._steer_cid: int = int(self.get_parameter("steer_can_id").value, 16)
        self._drive_cid: int = int(self.get_parameter("drive_can_id").value, 16)
        self._wheel_radius: float = self.get_parameter("wheel_radius").value
        self._target_speed: float = self.get_parameter("target_speed").value
        self._accel_time: float = self.get_parameter("acceleration_time").value
        self._drive_invert: bool = self.get_parameter("drive_direction_invert").value
        self._steer_invert: bool = self.get_parameter("steer_direction_invert").value
        self._rc_max_rpm: float = self.get_parameter("rc_max_rpm").value
        self._control_dt: float = self.get_parameter("control_dt").value
        self._min_speed: float = self.get_parameter("min_scheduling_speed").value
        self._max_steer_vel: float = self.get_parameter("max_steer_velocity").value
        self._pole_wc: float = self.get_parameter("pole_wc").value
        self._max_roll_rad: float = np.deg2rad(
            self.get_parameter("max_roll_angle_deg").value)
        self._steer_angle_limit_deg: float = self.get_parameter("steer_angle_limit_deg").value
        self._enable_on_start: bool = self.get_parameter("enable_on_start").value
        self._balance_delay: float = self.get_parameter("balance_start_delay").value
        self._pid_scheduling: bool = self.get_parameter("pid_gain_scheduling").value
        self._steer_kt: float = self.get_parameter("steer_torque_constant").value

        # PID overrides
        self._outer_kp_override: float = self.get_parameter("outer_kp").value
        self._outer_ki: float = self.get_parameter("outer_ki").value
        self._outer_kd: float = self.get_parameter("outer_kd").value
        self._outer_max_roll: float = self.get_parameter("outer_max_roll_rad").value
        self._mid_kp_override: float = self.get_parameter("mid_kp").value
        self._mid_ki: float = self.get_parameter("mid_ki").value
        self._mid_kd: float = self.get_parameter("mid_kd").value
        self._mid_max_rate: float = self.get_parameter("mid_max_rate_rad_s").value
        self._inner_kp_override: float = self.get_parameter("inner_kp").value
        self._inner_ki: float = self.get_parameter("inner_ki").value
        self._inner_kd: float = self.get_parameter("inner_kd").value
        self._inner_max_q: float = self.get_parameter("inner_max_q_current_a").value

        # ════════════════ Open CAN bus ════════════════
        self.get_logger().info(f"Opening CAN bus {self._can_iface} …")
        self._bus = can.interface.Bus(
            channel=self._can_iface, bustype="socketcan", receive_own_messages=False)

        # ════════════════ Bicycle model (for K extraction) ════════════════
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
        # Build LQR controller, extract K, initialise cascaded PIDs
        self._build_cascaded_pids()

        # ════════════════ Runtime state ════════════════
        self._roll_angle: float = 0.0
        self._roll_rate: float = 0.0
        self._body_rates: np.ndarray = np.array([0.0, 0.0, 0.0])
        self._steer_angle: float = 0.0
        self._steer_rate: float = 0.0
        self._rear_rate: float = 0.0

        self._roll_ts: float = 0.0
        self._steer_ts: float = 0.0
        self._rear_ts: float = 0.0

        self._enabled: bool = self._enable_on_start
        self._start_time: float | None = None
        self._prev_steer_c0: int = 0  # steer Q-axis current in 0.001 A units
        self._prev_drive_c1: int = 0
        self._sat_outer: bool = False
        self._steer_fault: bool = False
        self._forward_speed: float = 0.0
        self._rc_drive_rpm: float = 0.0
        self._ext_steer_rate_ref: float = 0.0

        # PID diagnostic signals
        self._pid_outer_out: float = 0.0
        self._pid_mid_out: float = 0.0
        self._pid_inner_out: float = 0.0

        self._last_tick: float = 0.0
        self._jitter_samples: list[float] = []

        # ════════════════ Subscriptions ════════════════
        self.create_subscription(Float64MultiArray, "roll_info", self._roll_cb, 10)
        self.create_subscription(Float64MultiArray, "steer_info", self._steer_cb, 10)
        self.create_subscription(Float64MultiArray, "rear_info", self._rear_cb, 10)
        self.create_subscription(Bool, "bike_balance/enable", self._enable_cb, 10)

        if self._vctrl == "rc":
            self.create_subscription(
                Float64MultiArray, "rc_info", self._rc_cb, 10)

        if self._mode == "inner_loop":
            self.create_subscription(
                Float64MultiArray, "new_balance/steer_rate_ref",
                self._steer_ref_cb, 10)

        # ════════════════ Publishers ════════════════
        self._status_pub = self.create_publisher(String, "new_balance/status", 10)
        self._debug_pub = self.create_publisher(Float64MultiArray, "new_balance/debug", 10)
        self._steer_cmd_pub = self.create_publisher(Float64MultiArray, "steer_cmd", 10)

        # ════════════════ Timers ════════════════
        self._timer = self.create_timer(self._control_dt, self._tick)
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._last_tick = time.time()

        self.get_logger().info(
            f"CascadedBalance | mode={self._mode} method={self._method} dt={self._control_dt}s | "
            f"CAN={self._can_iface} steer=0x{self._steer_cid:03X} drive=0x{self._drive_cid:03X} | "
            f"target_v={self._target_speed}m/s vctrl={self._vctrl} "
            f"rc_max_rpm={self._rc_max_rpm} enabled={self._enabled} | "
            f"PIDs: outer=({self._pid_outer.kp:.3f},{self._pid_outer.ki:.3f},{self._pid_outer.kd:.3f}) "
            f"mid=({self._pid_mid.kp:.3f},{self._pid_mid.ki:.3f},{self._pid_mid.kd:.3f}) "
            f"inner=({self._pid_inner.kp:.3f},{self._pid_inner.ki:.3f},{self._pid_inner.kd:.3f})")

    # ==================================================================
    # Build LQR controller, extract K, initialise cascaded PIDs
    # ==================================================================
    def _build_cascaded_pids(self) -> None:
        """Build the LQR/pole-placement controller, extract K = [k₁,k₂,k₃],
        and derive initial PID gains from K.

        Formula (heuristic):
            inner_kp  = |k₃| · Kt                  (roll rate → torque)
            mid_kp    = |k₂ / k₃|    clamped        (roll angle → roll rate)
            outer_kp  = |k₁ / k₂|    clamped        (steer vel → roll angle)

        Any explicitly-set parameter (> 0 for kp) overrides the extracted value.
        """
        # ---- Build LQR / pole-placement controller ----
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

        K = self._ctrl.K  # shape (1, 3)
        k1, k2, k3 = float(K[0, 0]), float(K[0, 1]), float(K[0, 2])
        self.get_logger().info(
            f"Extracted K = [{k1:.4f}, {k2:.4f}, {k3:.4f}]")
        self._K = [k1, k2, k3]

        # ---- Compute default PID gains from K ----
        k1a, k2a, k3a = abs(k1), abs(k2), abs(k3)

        # Inner: roll rate (rad/s) → torque (Nm ≈ A for Kt=1.0)
        inner_kp_default = k3a * self._steer_kt
        # Middle: roll angle (rad) → roll rate (rad/s)
        mid_kp_default = float(np.clip(k2a / max(k3a, 0.01), 0.5, 50.0))
        # Outer: steer velocity (rad/s) → roll angle (rad)
        outer_kp_default = float(np.clip(k1a / max(k2a, 0.01), 0.05, 10.0))

        self.get_logger().info(
            f"PID defaults from K — outer_kp={outer_kp_default:.3f} "
            f"mid_kp={mid_kp_default:.3f} inner_kp={inner_kp_default:.3f}")

        # ---- Apply overrides if set (> 0) ----
        outer_kp = outer_kp_default if self._outer_kp_override < 0 else self._outer_kp_override
        mid_kp   = mid_kp_default   if self._mid_kp_override   < 0 else self._mid_kp_override
        inner_kp = inner_kp_default if self._inner_kp_override < 0 else self._inner_kp_override

        # ---- Build three PID instances ----
        self._pid_outer = PID(
            kp=outer_kp, ki=self._outer_ki, kd=self._outer_kd,
            output_min=-self._outer_max_roll, output_max=self._outer_max_roll,
        )
        self._pid_mid = PID(
            kp=mid_kp, ki=self._mid_ki, kd=self._mid_kd,
            output_min=-self._mid_max_rate, output_max=self._mid_max_rate,
        )
        self._pid_inner = PID(
            kp=inner_kp, ki=self._inner_ki, kd=self._inner_kd,
            output_min=-self._inner_max_q, output_max=self._inner_max_q,
        )

    def _update_cascaded_pids(self) -> None:
        """Re-extract K from the current model state (gain scheduling)."""
        if not self._pid_scheduling:
            return
        self._ctrl.update_gain(wc=self._pole_wc)
        K = self._ctrl.K
        k1, k2, k3 = float(K[0, 0]), float(K[0, 1]), float(K[0, 2])
        self._K = [k1, k2, k3]

        k1a, k2a, k3a = abs(k1), abs(k2), abs(k3)

        # Update only if not overridden
        if self._inner_kp_override < 0:
            self._pid_inner.kp = k3a * self._steer_kt
        if self._mid_kp_override < 0:
            self._pid_mid.kp = float(np.clip(k2a / max(k3a, 0.01), 0.5, 50.0))
        if self._outer_kp_override < 0:
            self._pid_outer.kp = float(np.clip(k1a / max(k2a, 0.01), 0.05, 10.0))

    def _reset_pids(self) -> None:
        """Clear all PID integrators and previous-error states."""
        self._pid_outer.reset()
        self._pid_mid.reset()
        self._pid_inner.reset()

    # ==================================================================
    # Topic callbacks
    # ==================================================================
    def _roll_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 7:
            self._roll_angle = float(msg.data[0])
            self._roll_rate = float(msg.data[3])
            self._body_rates = np.array([
                float(msg.data[4]), float(msg.data[5]), float(msg.data[6]),
            ])
            self._roll_ts = time.time()

    def _steer_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 2:
            self._steer_angle = float(msg.data[0])
            self._steer_rate = float(msg.data[1])
            self._steer_ts = time.time()

    def _rear_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 1:
            self._rear_rate = float(msg.data[0])
            self._rear_ts = time.time()

    def _enable_cb(self, msg: Bool) -> None:
        was = self._enabled
        self._enabled = bool(msg.data)
        if self._enabled and not was:
            self._start_time = None
            self._sat_outer = False
            self._steer_fault = False
            self._reset_pids()
            self.get_logger().info("BALANCE ENABLED  —  PIDs reset")
        elif not self._enabled and was:
            self._stop_motors()
            self.get_logger().info("BALANCE DISABLED — motors stopped")

    def _steer_ref_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 1:
            self._ext_steer_rate_ref = float(msg.data[0])

    def _rc_cb(self, msg: Float64MultiArray) -> None:
        """Store RC drive RPM from /rc_info.  Clamped to ±rc_max_rpm."""
        if len(msg.data) >= 1:
            raw = float(msg.data[0])
            self._rc_drive_rpm = max(-self._rc_max_rpm, min(raw, self._rc_max_rpm))

    # ==================================================================
    # CAN writing
    # ==================================================================
    def _send_speed(self, can_id: int, rad_s: float) -> None:
        """Send C1 speed command (same as main_balance.py)."""
        c1 = _rad_s_to_c1(rad_s)
        data = bytes([0xC1]) + struct.pack("<i", c1)
        try:
            self._bus.send(can.Message(
                arbitration_id=can_id, data=data, is_extended_id=False))
        except can.CanError as exc:
            self.get_logger().warn(
                f"CAN send failed: 0x{can_id:03X} C1={c1}: {exc}",
                throttle_duration_sec=2.0)

    def _send_q_current(self, can_id: int, current_a: float) -> None:
        """Send C0 Q-axis current command.

        Frame:  <0xC0> <4-byte LE int32  current in 0.001 A>
        The motor firmware converts:  τ = Kt × I_q  internally.
        """
        raw = int(round(current_a * 1000.0))  # A → 0.001 A
        data = bytes([0xC0]) + struct.pack("<i", raw)
        try:
            self._bus.send(can.Message(
                arbitration_id=can_id, data=data, is_extended_id=False))
        except can.CanError as exc:
            self.get_logger().warn(
                f"CAN send failed: 0x{can_id:03X} C0={raw}: {exc}",
                throttle_duration_sec=2.0)

    def _stop_motors(self) -> None:
        """Send C1=0 + CF to both motors."""
        for cid in (self._steer_cid, self._drive_cid):
            self._send_speed(cid, 0.0)
            self._send_cf(cid)

    def _send_cf(self, can_id: int) -> None:
        """Send CF (MOTOR_DISABLE)."""
        try:
            self._bus.send(can.Message(
                arbitration_id=can_id, data=bytes([0xCF]),
                is_extended_id=False))
        except can.CanError:
            pass

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
    # tick: sign_check  (identical to main_balance.py)
    # ==================================================================
    def _tick_sign_check(self, now: float) -> None:
        if self._steer_ts > 0 and now - self._steer_ts > 1.0:
            self.get_logger().warn("Steer data stale", throttle_duration_sec=2.0)
        if self._roll_ts > 0 and now - self._roll_ts > 1.0:
            self.get_logger().warn("Roll data stale", throttle_duration_sec=2.0)
        if self._rear_ts > 0 and now - self._rear_ts > 1.0:
            self.get_logger().warn("Rear data stale", throttle_duration_sec=2.0)

        self.get_logger().info(
            f"[SIGN_CHECK] roll={np.rad2deg(self._roll_angle):.1f}° "
            f"p={np.rad2deg(self._body_rates[0]):.1f} "
            f"q={np.rad2deg(self._body_rates[1]):.1f} "
            f"r={np.rad2deg(self._body_rates[2]):.1f} dps "
            f"roll_rate={np.rad2deg(self._roll_rate):.1f} dps | "
            f"steer={np.rad2deg(self._steer_angle):.1f}° "
            f"rear={self._rear_rate:.2f} rad/s",
            throttle_duration_sec=2.0)

        debug = Float64MultiArray()
        debug.data = [
            self._roll_angle, self._roll_rate,
            float(self._body_rates[0]), float(self._body_rates[1]),
            float(self._body_rates[2]),
            self._steer_angle, self._rear_rate,
        ]
        self._debug_pub.publish(debug)

    # ==================================================================
    # tick: inner_loop  (identical to main_balance.py)
    # ==================================================================
    def _tick_inner_loop(self, now: float) -> None:
        if self._steer_ts > 0 and now - self._steer_ts > 1.0:
            self.get_logger().warn("Steer data stale", throttle_duration_sec=2.0)
        if self._roll_ts > 0 and now - self._roll_ts > 1.0:
            self.get_logger().warn("Roll data stale", throttle_duration_sec=2.0)

        if not self._enabled:
            self._send_speed(self._steer_cid, 0.0)
            self._send_speed(self._drive_cid, 0.0)
            return

        if self._roll_ts > 0 and abs(self._roll_angle) > self._max_roll_rad:
            self.get_logger().error(
                f"INNER STOP: |roll|={np.rad2deg(self._roll_angle):.1f}° > "
                f"{np.rad2deg(self._max_roll_rad):.1f}°")
            self._enabled = False
            self._stop_motors()
            return

        ref = self._ext_steer_rate_ref

        # Steer angle limit  (C1 speed mode — same as main_balance.py)
        steer_deg = np.rad2deg(self._steer_angle)
        if abs(steer_deg) > self._steer_angle_limit_deg:
            if not self._steer_fault:
                self._steer_fault = True
                self._send_cf(self._steer_cid)
                self.get_logger().error(
                    f"STEER FAULT: |{steer_deg:.1f}|° > {self._steer_angle_limit_deg}° "
                    f"— steer motor DISABLED (CF). Manually re-center to re-enable.",
                    throttle_duration_sec=1.0)
            self._send_speed(self._drive_cid, 0.0)
            self._prev_steer_c1 = 0
            msg = Float64MultiArray()
            msg.data = [0.0]
            self._steer_cmd_pub.publish(msg)
            return

        if self._steer_fault and abs(steer_deg) < self._steer_angle_limit_deg * 0.5:
            self._steer_fault = False
            self.get_logger().info(
                f"STEER RECOVERED: |{steer_deg:.1f}|° within safe range — re-enabled")

        if self._steer_fault:
            self._send_speed(self._drive_cid, 0.0)
            msg = Float64MultiArray()
            msg.data = [0.0]
            self._steer_cmd_pub.publish(msg)
            return

        msg = Float64MultiArray()
        msg.data = [float(ref)]
        self._steer_cmd_pub.publish(msg)

        self._send_speed(self._steer_cid, -ref if self._steer_invert else ref)
        self._send_speed(self._drive_cid, 0.0)
        self._prev_steer_c1 = _rad_s_to_c1(ref)

    # ==================================================================
    # tick: full  — cascaded PID outer/mid/inner → C0 torque to steer
    # ==================================================================
    def _tick_full(self, now: float) -> None:
        elapsed = now - self._start_time if self._start_time else 0.0

        # ---- stale data checks (identical to main_balance.py) ----
        if self._steer_ts == 0.0 or now - self._steer_ts > 0.5:
            if self._enabled:
                self.get_logger().warn("Steer data stale — disabling", throttle_duration_sec=2.0)
                self._enabled = False
                self._stop_motors()
            return
        if self._roll_ts == 0.0 or now - self._roll_ts > 0.5:
            if self._enabled:
                self.get_logger().warn("Roll data stale — disabling", throttle_duration_sec=2.0)
                self._enabled = False
                self._stop_motors()
            return
        if self._rear_ts > 0 and now - self._rear_ts > 0.5:
            self.get_logger().warn("Rear data stale", throttle_duration_sec=2.0)

        # ---- forward speed (same as main_balance.py) ----
        self._forward_speed = self._wheel_radius * abs(self._rear_rate)

        # ---- speed profile for rear (drive) motor (same as main_balance.py) ----
        if self._vctrl == "rc":
            drive_rpm = max(-self._rc_max_rpm,
                            min(self._rc_max_rpm, self._rc_drive_rpm))
            rear_rate_ref = drive_rpm * (2.0 * math.pi / 60.0)
        else:
            if self._accel_time > 0:
                speed_ref = min(elapsed / self._accel_time * self._target_speed, self._target_speed)
            else:
                speed_ref = self._target_speed
            rear_rate_ref = speed_ref / self._wheel_radius

        # ---- optional PID gain scheduling (re-extract K, update PID gains) ----
        sch_v = max(self._forward_speed, self._min_speed)
        self._bike.update_system_parameters(sch_v, min_forward_speed=self._min_speed)
        self._update_cascaded_pids()

        # ════════════════════════════════════════════════════════════════
        # Cascaded PID
        # ════════════════════════════════════════════════════════════════
        if elapsed < self._balance_delay or not self._enabled:
            steer_current_a = 0.0
            self._sat_outer = False
            self._pid_outer_out = 0.0
            self._pid_mid_out = 0.0
            self._pid_inner_out = 0.0
        else:
            # ---- Outer PID: steer velocity → roll angle reference ----
            e_outer = 0.0 - self._steer_rate  # target: zero steer velocity
            self._pid_outer_out = self._pid_outer.step(e_outer, self._control_dt)
            roll_ref = float(self._pid_outer_out)

            # ---- Middle PID: roll angle → roll rate reference ----
            e_mid = roll_ref - self._roll_angle
            self._pid_mid_out = self._pid_mid.step(e_mid, self._control_dt)
            roll_rate_ref = float(self._pid_mid_out)

            # ---- Inner PID: roll rate → Q-axis current (A) ----
            e_inner = roll_rate_ref - self._roll_rate
            self._pid_inner_out = self._pid_inner.step(e_inner, self._control_dt)
            steer_current_a = float(self._pid_inner_out)

            self._sat_outer = (
                abs(self._pid_outer.last_output) >= self._outer_max_roll * 0.99
            )

        # ---- steer angle limit (same fault logic as main_balance.py) ----
        steer_deg = np.rad2deg(self._steer_angle)
        if abs(steer_deg) > self._steer_angle_limit_deg:
            if not self._steer_fault:
                self._steer_fault = True
                self._send_cf(self._steer_cid)
                self.get_logger().error(
                    f"STEER FAULT: |{steer_deg:.1f}|° > {self._steer_angle_limit_deg}° "
                    f"— steer motor DISABLED (CF). Manually re-center to re-enable.",
                    throttle_duration_sec=1.0)
            # Keep sending drive motor
            if self._enabled:
                self._send_speed(self._drive_cid,
                                 -rear_rate_ref if self._drive_invert else rear_rate_ref)
            else:
                self._send_speed(self._drive_cid, 0.0)
            steer_current_a = 0.0
            self._prev_drive_c1 = _rad_s_to_c1(rear_rate_ref)
            msg = Float64MultiArray()
            msg.data = [0.0]
            self._steer_cmd_pub.publish(msg)
            self._reset_pids()
            return

        # ---- auto-recovery: angle returns within 5% of limit ----
        if self._steer_fault and abs(steer_deg) < self._steer_angle_limit_deg * 0.05:
            self._steer_fault = False
            self._reset_pids()
            self.get_logger().info(
                f"STEER RECOVERED: |{steer_deg:.1f}|° within safe range — re-enabled")

        if self._steer_fault:
            if self._enabled:
                self._send_speed(self._drive_cid,
                                 -rear_rate_ref if self._drive_invert else rear_rate_ref)
            else:
                self._send_speed(self._drive_cid, 0.0)
            steer_current_a = 0.0
            self._prev_drive_c1 = _rad_s_to_c1(rear_rate_ref)
            msg = Float64MultiArray()
            msg.data = [0.0]
            self._steer_cmd_pub.publish(msg)
            return

        # ---- emergency roll stop (same as main_balance.py) ----
        if abs(self._roll_angle) > self._max_roll_rad:
            if self._enabled:
                self.get_logger().error(
                    f"EMERGENCY: |roll|={np.rad2deg(self._roll_angle):.1f}°")
                self._enabled = False
                self._stop_motors()
            return

        # Publish steer_cmd (Q-axis current in A, before inversion)
        msg = Float64MultiArray()
        msg.data = [float(steer_current_a)]
        self._steer_cmd_pub.publish(msg)

        # ════════════════════════════════════════════════════════════════
        # Send CAN commands
        #   Steer:  C0 (Q-axis current)
        #   Drive:  C1 (speed) — identical to main_balance.py
        # ════════════════════════════════════════════════════════════════
        if self._enabled:
            # Steer: C0 Q-axis current (apply inversion)
            current_steer = -steer_current_a if self._steer_invert else steer_current_a
            self._send_q_current(self._steer_cid, current_steer)
            # Drive: C1 speed
            self._send_speed(self._drive_cid,
                             -rear_rate_ref if self._drive_invert else rear_rate_ref)
        else:
            self._send_speed(self._steer_cid, 0.0)
            self._send_speed(self._drive_cid, 0.0)

        self._prev_steer_c0 = int(round(steer_current_a * 1000.0))  # 0.001 A units, for logging
        self._prev_drive_c1 = _rad_s_to_c1(rear_rate_ref)

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
                "dt_max_ms": round(float(s.max()) * 1000, 2),
            }

        status = {
            "mode": self._mode,
            "cascaded": True,
            "enabled": self._enabled,
            "method": self._method if self._mode == "full" else "n/a",
            "roll_deg": round(np.rad2deg(self._roll_angle), 2),
            "roll_rate_dps": round(np.rad2deg(self._roll_rate), 2),
            "body_p_dps": round(np.rad2deg(float(self._body_rates[0])), 2),
            "body_q_dps": round(np.rad2deg(float(self._body_rates[1])), 2),
            "body_r_dps": round(np.rad2deg(float(self._body_rates[2])), 2),
            "steer_angle_deg": round(np.rad2deg(self._steer_angle), 2),
            "steer_rate_dps": round(np.rad2deg(self._steer_rate), 2),
            "speed_m_s": round(self._forward_speed, 3),
            "steer_current_a": round(self._pid_inner_out, 4),
            "steer_c0": self._prev_steer_c0,
            "sat_outer": self._sat_outer,
            "steer_fault": self._steer_fault,
            "K": [round(float(k), 4) for k in self._K],
            "outer_pid": {
                "kp": round(self._pid_outer.kp, 4),
                "ki": round(self._pid_outer.ki, 4),
                "kd": round(self._pid_outer.kd, 4),
                "integral": round(self._pid_outer.integral, 4),
                "output": round(self._pid_outer_out, 4),
            },
            "mid_pid": {
                "kp": round(self._pid_mid.kp, 4),
                "ki": round(self._pid_mid.ki, 4),
                "kd": round(self._pid_mid.kd, 4),
                "integral": round(self._pid_mid.integral, 4),
                "output": round(self._pid_mid_out, 4),
            },
            "inner_pid": {
                "kp": round(self._pid_inner.kp, 4),
                "ki": round(self._pid_inner.ki, 4),
                "kd": round(self._pid_inner.kd, 4),
                "integral": round(self._pid_inner.integral, 4),
                "output": round(self._pid_inner_out, 4),
            },
            "jitter": jitter,
        }
        msg = String()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)

        debug = Float64MultiArray()
        debug.data = [
            self._roll_angle, self._roll_rate,
            self._steer_angle, self._steer_rate,
            self._forward_speed, self._pid_inner_out,
            float(self._sat_outer),
            float(1.0 if self._enabled else 0.0),
            self._pid_outer_out,
            self._pid_mid_out,
            self._pid_inner_out,
        ]
        self._debug_pub.publish(debug)

    # ==================================================================
    # Cleanup
    # ==================================================================
    def destroy_node(self) -> None:
        self.get_logger().info("Shutting down")
        self._stop_motors()
        if hasattr(self, "_bus") and self._bus is not None:
            self._bus.shutdown()
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