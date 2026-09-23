#!/usr/bin/env python3
"""
IMU gyro zero-bias calibration node — tutorial 06 §7 step 3.

Collects raw gyro samples from /imu/data_raw while the bike is held
perfectly stationary, then computes the mean (bias) and standard
deviation (noise) for each gyro axis.

The estimated bias values should be copied into your balance parameter
file (e.g. config/balance_params.yaml) under ``gyro_bias_dps``.

Usage
-----
  1. Place the bike on a level surface so it cannot move.
  2. Run this node for 10–20 seconds:
       ros2 run new_balance imu_calibrate --ros-args -p duration_s:=15.0
  3. Read the estimated bias from the terminal output.
  4. Copy the ``gyro_bias_dps`` line into your parameter file.

Publishes
---------
  /imu/calibration_result (std_msgs/String)  — JSON result with bias & noise
      Fields: n_samples, duration_s, gyro_bias_rad_s, gyro_bias_dps,
              gyro_noise_std_dps

Parameters
----------
  duration_s  — sampling duration in seconds (default: 15.0)
"""

from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String


class ImuCalibrateNode(Node):
    """Collect gyro samples → estimate zero bias."""

    def __init__(self) -> None:
        super().__init__("imu_calibrate")

        self.declare_parameter("duration_s", 15.0)

        self._duration: float = self.get_parameter("duration_s").value
        self._samples: list[np.ndarray] = []
        self._start_time: float | None = None
        self._done: bool = False

        self._sub = self.create_subscription(Imu, "imu/data_raw", self._cb, 100)
        self._pub = self.create_publisher(String, "imu/calibration_result", 10)

        self.get_logger().info(
            f"Calibrating for {self._duration:.1f} s — KEEP THE BIKE STATIONARY"
        )

    def _cb(self, msg: Imu) -> None:
        if self._done:
            return

        now = time.time()
        if self._start_time is None:
            self._start_time = now

        self._samples.append(np.array([
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        ]))

        if now - self._start_time >= self._duration:
            self._finish()

    def _finish(self) -> None:
        """Compute and report bias statistics."""
        self._done = True
        data = np.array(self._samples)          # (N, 3)
        bias = data.mean(axis=0)                # rad/s
        std = data.std(axis=0)                  # rad/s

        bias_dps = np.rad2deg(bias)
        std_dps = np.rad2deg(std)

        result = {
            "n_samples": len(self._samples),
            "duration_s": round(self._duration, 1),
            "gyro_bias_rad_s": [round(float(v), 8) for v in bias],
            "gyro_bias_dps": [round(float(v), 6) for v in bias_dps],
            "gyro_noise_std_dps": [round(float(v), 6) for v in std_dps],
        }

        self.get_logger().info("=" * 60)
        self.get_logger().info("CALIBRATION RESULT")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Samples       : {result['n_samples']}")
        self.get_logger().info(
            f"  Gyro bias     : bx={bias_dps[0]:.4f}, by={bias_dps[1]:.4f}, "
            f"bz={bias_dps[2]:.4f}  deg/s"
        )
        self.get_logger().info(
            f"  Noise std     : sx={std_dps[0]:.4f}, sy={std_dps[1]:.4f}, "
            f"sz={std_dps[2]:.4f}  deg/s"
        )
        self.get_logger().info("")
        self.get_logger().info(
            "Copy into your balance params YAML:"
        )
        self.get_logger().info(
            f"  gyro_bias_dps: [{bias_dps[0]:.4f}, {bias_dps[1]:.4f}, "
            f"{bias_dps[2]:.4f}]"
        )
        self.get_logger().info("=" * 60)

        msg = String()
        msg.data = json.dumps(result)
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuCalibrateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
