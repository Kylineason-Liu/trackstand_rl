#!/usr/bin/env python3
"""
IMU axis sign verification node — tutorial 06 §7 step 2.

Subscribes to raw IMU data, applies the R_BS sensor-to-bike rotation and
gyro bias correction, then logs the corrected roll / pitch / yaw angles
and body angular rates so you can verify axis signs by hand.

Usage
-----
  1. Keep the bike stationary and level.
  2. Run this node (optionally with your R_BS and bias parameters):
       ros2 run new_balance imu_sign_check --ros-args -p sensor_to_bike_rpy_deg:="[0.0,0.0,0.0]" -p gyro_bias_dps:="[0.0,0.0,0.0]"
  3. Manually tilt / rotate the bike and check that the logged values
     match the physical motion:

     ── Roll ──
       Tilt bike LEFT  → roll should become more NEGATIVE (or positive;
                          the sign depends on your R_BS convention).
       Tilt bike RIGHT → roll should have the opposite sign.
       Verify that the *direction* is consistent with your convention.

     ── Pitch ──
       Tilt bike FORWARD (nose down) → pitch should change with a
       consistent sign.

     ── Yaw ──
       Rotate bike LEFT (counter-clockwise from above) → yaw rate (r)
       should show a consistent sign.

     ── Body rates ──
       p = roll rate  (around bike x-axis, forward)
       q = pitch rate (around bike y-axis, leftward)
       r = yaw rate   (around bike z-axis, upward)

  4. If any sign is wrong, adjust sensor_to_bike_rpy_deg (R_BS) until
     the reported directions match physical motion.

Publishes
---------
  /imu/sign_check (std_msgs/Float64MultiArray)  — 7 floats:
      [roll, pitch, yaw, roll_rate, body_p, body_q, body_r]
      Angles in rad, rates in rad/s.  All corrected to bike body frame.

Parameters
----------
  sensor_to_bike_rpy_deg  — R_BS rotation [roll, pitch, yaw] in degrees (ZYX intrinsic)
  gyro_bias_dps           — Gyro bias [x, y, z] in deg/s (IMU sensor frame)
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray


# ---------------------------------------------------------------------------
# IMU correction — same logic as roll_publisher.py
# ---------------------------------------------------------------------------
def _rotation_from_rpy_deg(rpy_deg):
    """Build a Rotation from [roll, pitch, yaw] in degrees (ZYX intrinsic)."""
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=float))
    return Rotation.from_euler("ZYX", [yaw, pitch, roll])


def _correct_imu(quat_xyzw, gyro_s, sensor_to_bike, gyro_bias_s):
    """Apply R_BS + gyro bias to raw IMU data.

    Returns
    -------
    roll, pitch, yaw  : float   corrected Euler angles (rad)
    roll_rate         : float   corrected roll rate via Euler kinematics (rad/s)
    body_rates        : (3,) ndarray  corrected body angular rates [p, q, r] (rad/s)
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
class ImuSignCheckNode(Node):
    """Log corrected IMU state for axis sign verification."""

    def __init__(self) -> None:
        super().__init__("imu_sign_check")

        self.declare_parameter("sensor_to_bike_rpy_deg", [0.0, 0.0, 0.0])
        self.declare_parameter("gyro_bias_dps", [0.0, 0.0, 0.0])

        rpy_val = self.get_parameter("sensor_to_bike_rpy_deg").value
        bias_dps = self.get_parameter("gyro_bias_dps").value

        self._sensor_to_bike = _rotation_from_rpy_deg(rpy_val)
        self._gyro_bias_s = np.deg2rad(np.asarray(bias_dps, dtype=float))

        # Latest IMU data
        self._quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        self._gyro_s = np.array([0.0, 0.0, 0.0])

        self._sub = self.create_subscription(Imu, "imu/data_raw", self._imu_cb, 10)
        self._pub = self.create_publisher(Float64MultiArray, "imu/sign_check", 10)

        self.get_logger().info(
            "=" * 60)
        self.get_logger().info(
            "IMU SIGN CHECK — verify axis directions by hand")
        self.get_logger().info(
            f"  R_BS  = {rpy_val}  (roll/pitch/yaw deg, ZYX intrinsic)")
        self.get_logger().info(
            f"  bias  = {bias_dps}  dps (gyro bias, sensor frame)")
        self.get_logger().info(
            "")
        self.get_logger().info(
            "  Tilt bike LEFT  → watch 'roll'  sign & magnitude.")
        self.get_logger().info(
            "  Tilt bike FWD   → watch 'pitch' sign & magnitude.")
        self.get_logger().info(
            "  Rotate bike CCW → watch 'r'     sign & magnitude.")
        self.get_logger().info(
            "  If signs are wrong, adjust sensor_to_bike_rpy_deg.")
        self.get_logger().info(
            "=" * 60)

    def _imu_cb(self, msg: Imu) -> None:
        """IMU callback: correct and log state at throttled rate."""
        self._quat_xyzw = np.array([
            float(msg.orientation.x), float(msg.orientation.y),
            float(msg.orientation.z), float(msg.orientation.w),
        ])
        self._gyro_s = np.array([
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        ])

        try:
            roll, pitch, yaw, roll_rate, body = _correct_imu(
                self._quat_xyzw, self._gyro_s,
                self._sensor_to_bike, self._gyro_bias_s)
        except ValueError:
            return  # pitch singularity — wait for next sample

        p, q, r = float(body[0]), float(body[1]), float(body[2])

        # Log at throttled rate (every ~2 s) so the terminal is readable
        self.get_logger().info(
            f"roll={np.rad2deg(roll):+.1f}°  "
            f"pitch={np.rad2deg(pitch):+.1f}°  "
            f"yaw={np.rad2deg(yaw):+.1f}°  |  "
            f"p={np.rad2deg(p):+.1f}  q={np.rad2deg(q):+.1f}  "
            f"r={np.rad2deg(r):+.1f} dps  |  "
            f"roll_rate={np.rad2deg(roll_rate):+.1f} dps",
            throttle_duration_sec=2.0,
        )

        # Publish corrected state
        out = Float64MultiArray()
        out.data = [
            roll, pitch, yaw, roll_rate,
            p, q, r,
        ]
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuSignCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
