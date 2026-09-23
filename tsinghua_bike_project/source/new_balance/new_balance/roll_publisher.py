#!/usr/bin/env python3
"""
Reads the YahBoom IMU directly via USB serial (/dev/ttyUSB0) and publishes
corrected roll/pitch/yaw angles and body angular rates.

This node does NOT subscribe to any pre-existing topic.  It opens the IMU
serial port directly using the vendor ``YbImuSerial`` library, applies the
R_BS sensor-to-bike rotation and gyro bias compensation, then publishes
the corrected state on ``/roll_info`` at 200 Hz.

Publishes
---------
  /roll_info (std_msgs/Float64MultiArray)  — 9 floats:
      Indices 0–6:  [roll, pitch, yaw, roll_rate, body_p, body_q, body_r]
      Index   7:     publish_ts  — time.time() for latency measurement
      Index   8:     droll_dt    — discrete derivative of roll (rad/s)

      Angles in rad, rates in rad/s.  All corrected to bike body frame.
      roll_rate is the Euler-kinematic roll rate (from gyro).
      droll_dt is the finite-difference derivative (roll − prev_roll) / dt.

      roll (index 0) is passed through a first-order low-pass filter
      (new_balance.lpf.FirstOrderLPF): y[n] = α·x[n] + (1−α)·y[n−1],
      with α = roll_lp_alpha ∈ (0, 1].  α = 1 bypasses the filter (raw).
      droll_dt (index 8) is the derivative of the FILTERED roll, so the
      angle and its derivative stay consistent.  All other entries are raw.

Parameters
----------
  serial_port             — IMU serial device       (default: /dev/ttyUSB1)
  sensor_to_bike_rpy_deg  — R_BS [roll, pitch, yaw] degrees, ZYX intrinsic
  gyro_bias_dps           — Gyro bias [x, y, z] deg/s in IMU sensor frame
  publish_rate_hz         — ROS timer rate          (default: 200.0)
  roll_lp_alpha           — roll LPF coefficient α ∈ (0, 1] (default: 0.8)
                            Can be changed at runtime via set_parameters
                            (used by alpha_sweep).
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64MultiArray
import sys
from pathlib import Path

from new_balance.lpf import FirstOrderLPF

YBIMU_LIB_DIR = Path('/home/ubuntu/assign_2_Hardwares/assign_2_Hardwares/1.IMU/Driver/YbImuLib')
if str(YBIMU_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(YBIMU_LIB_DIR))

from YbImuLib import YbImuSerial


# ---------------------------------------------------------------------------
# Coordinate correction (same math as the original balance_node)
# ---------------------------------------------------------------------------
def _rotation_from_rpy_deg(rpy_deg):
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=float))
    return Rotation.from_euler("ZYX", [yaw, pitch, roll])


def _correct_imu(quat_xyzw, gyro_s, sensor_to_bike, gyro_bias_s):
    """Apply R_BS + gyro bias to raw IMU data.

    Returns
    -------
    roll, pitch, yaw  : float   corrected Euler angles (rad)
    roll_rate         : float   corrected roll rate via Euler kinematics (rad/s)
    body_rates        : (3,) ndarray  [p, q, r] in bike body frame (rad/s)
    """
    world_from_sensor = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float))
    world_from_bike = world_from_sensor * sensor_to_bike.inv()
    yaw, pitch, roll = world_from_bike.as_euler("ZYX")

    body_rates = sensor_to_bike.apply(
        np.asarray(gyro_s, dtype=float) - np.asarray(gyro_bias_s, dtype=float))
    p, q, r = body_rates

    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) < 1e-6:
        raise ValueError("Pitch too close to ±90° — Euler singularity.")
    roll_rate = p + np.sin(roll) * np.tan(pitch) * q + np.cos(roll) * np.tan(pitch) * r

    return float(roll), float(pitch), float(yaw), float(roll_rate), body_rates


# ======================================================================
# Main node
# ======================================================================
class RollInfoPublisher(Node):
    """YahBoom IMU serial → corrected roll state → /roll_info."""

    def __init__(self) -> None:
        super().__init__("roll_info_publisher")

        # ── Parameters ──
        self.declare_parameter("serial_port", "/dev/ttyUSB1")
        self.declare_parameter("sensor_to_bike_rpy_deg", [180.0, 180.0, 0.0])
        self.declare_parameter("gyro_bias_dps", [0.0113, 0.0, 0.0])
        self.declare_parameter("publish_rate_hz", 200.0)
        self.declare_parameter("roll_lp_alpha", 0.5)

        port: str = self.get_parameter("serial_port").value
        rpy_val = self.get_parameter("sensor_to_bike_rpy_deg").value
        bias_dps = self.get_parameter("gyro_bias_dps").value
        pub_hz: float = self.get_parameter("publish_rate_hz").value
        lpf_alpha: float = self.get_parameter("roll_lp_alpha").value

        self._sensor_to_bike = _rotation_from_rpy_deg(rpy_val)
        self._gyro_bias_s = np.deg2rad(np.asarray(bias_dps, dtype=float))

        # ── Roll low-pass filter (shared module, runtime-retunable) ──
        self._roll_lpf = FirstOrderLPF(lpf_alpha)
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── Discrete-time derivative state ──
        self._prev_roll: float = 0.0
        self._prev_droll_dt: float = 0.0
        self._prev_tick_ts: float = 0.0

        # ── Open IMU directly via vendor serial library ──
        self.get_logger().info(f"Opening IMU on {port} …")
        try:
            self._imu = YbImuSerial(port)
        except Exception as exc:
            self.get_logger().error(f"Failed to open IMU serial port {port}: {exc}")
            raise

        self._imu.create_receive_threading()
        self._imu.set_algo_type(6)   # 9-axis fusion for stable yaw
        self._imu.set_report_rate(100)  # IMU outputs at 100 Hz internally

        # ── Publisher ──
        self._pub = self.create_publisher(Float64MultiArray, "roll_info", 10)

        # ── Timer ──
        self._timer = self.create_timer(1.0 / pub_hz, self._tick)

        self.get_logger().info(
            f"RollInfoPublisher | port={port} pub={pub_hz} Hz | "
            f"R_BS={rpy_val}° bias={bias_dps} dps | roll LPF α={lpf_alpha}")

    def _on_param_change(self, params):
        """Accept runtime roll_lp_alpha updates (used by alpha_sweep)."""
        for p in params:
            if p.name == "roll_lp_alpha" and p.type_ == Parameter.Type.DOUBLE:
                if not 0.0 < p.value <= 1.0:
                    return SetParametersResult(
                        successful=False,
                        reason="roll_lp_alpha must be in (0, 1]")
                self._roll_lpf.alpha = p.value
                self.get_logger().info(f"roll LPF α set to {p.value:.3f}")
        return SetParametersResult(successful=True)

    def _tick(self) -> None:
        """Read IMU, apply coordinate correction and roll LPF, publish."""
        now = time.time()

        try:
            qw, qx, qy, qz = self._imu.get_imu_quaternion_data()
            quat_xyzw = np.array([qx, qy, qz, qw], dtype=float)
            gx, gy, gz = self._imu.get_gyroscope_data()
            gyro_s = np.array([gx, gy, gz], dtype=float)
        except Exception:
            return

        try:
            roll, pitch, yaw, roll_rate, body = _correct_imu(
                quat_xyzw, gyro_s, self._sensor_to_bike, self._gyro_bias_s)
        except ValueError:
            return

        raw_p, raw_q, raw_r = float(body[0]), float(body[1]), float(body[2])

        # First-order low-pass filter on roll (α = 1 ⇒ raw passthrough)
        roll_filt = self._roll_lpf.update(roll)
        
        if abs(roll_filt - self._prev_roll) >= 0.5: # avoid sudden imu error spike readings
            roll_filt = self._prev_roll
            droll_dt = 0.0
        else:
            # Discrete-time derivative of FILTERED roll angle (rad/s)
            if self._prev_tick_ts > 0:
                dt = now - self._prev_tick_ts
                droll_dt = (roll_filt - self._prev_roll) / dt if dt > 1e-9 else 0.0
            else:
                droll_dt = 0.0

        self._prev_roll = roll_filt
        self._prev_droll_dt = droll_dt
        self._prev_tick_ts = now

        msg = Float64MultiArray()
        msg.data = [roll_filt, pitch, yaw, roll_rate, raw_p, raw_q, raw_r, now, roll, droll_dt]
        self._pub.publish(msg)

    def destroy_node(self) -> None:
        if hasattr(self, "_imu") and self._imu is not None:
            try:
                self._imu._dev.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RollInfoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
