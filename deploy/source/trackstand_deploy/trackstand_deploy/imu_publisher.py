"""YahBoom serial IMU to corrected ROS 2 Imu samples."""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

from .contracts import correct_imu_sample, rpy_degrees_to_quaternion_xyzw
from .imu_serial import YbImuSerial


class ImuPublisher(Node):
    def __init__(self) -> None:
        super().__init__("trackstand_imu_publisher")
        self.declare_parameter(
            "serial_port", "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
        )
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("sensor_to_bike_rpy_deg", [1.42, 0.0, 0.0])
        self.declare_parameter("gyro_bias_dps", [0.0, 0.0, 0.0])
        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("source_timeout_s", 0.25)

        port = str(self.get_parameter("serial_port").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        rpy_deg = np.asarray(self.get_parameter("sensor_to_bike_rpy_deg").value, dtype=float)
        bias_dps = np.asarray(self.get_parameter("gyro_bias_dps").value, dtype=float)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._source_timeout_s = float(self.get_parameter("source_timeout_s").value)
        if (
            rpy_deg.shape != (3,)
            or bias_dps.shape != (3,)
            or publish_rate_hz <= 0.0
            or self._source_timeout_s <= 0.0
        ):
            raise ValueError("IMU RPY/bias need three values; publish rate and source timeout must be positive")

        self._sensor_to_bike = rpy_degrees_to_quaternion_xyzw(rpy_deg)
        self._gyro_bias_rad_s = np.deg2rad(bias_dps)
        self.get_logger().info(f"Opening YahBoom IMU on {port}")
        self._imu = YbImuSerial(port)
        self._imu.create_receive_threading()
        self._imu.set_report_rate(min(100, max(10, round(publish_rate_hz))))
        self._last_source_time = 0.0

        self._publisher = self.create_publisher(Imu, "/trackstand/imu", 10)
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._publish)

    def _publish(self) -> None:
        source_time = float(self._imu.last_valid_frame_monotonic)
        if (
            source_time <= self._last_source_time
            or source_time <= 0.0
            or time.monotonic() - source_time > self._source_timeout_s
        ):
            return
        try:
            qw, qx, qy, qz = self._imu.get_imu_quaternion_data()
            gyro_sensor = np.asarray(self._imu.get_gyroscope_data(), dtype=float)
            quaternion, body_rate = correct_imu_sample(
                np.array([qx, qy, qz, qw], dtype=float),
                gyro_sensor,
                self._sensor_to_bike,
                self._gyro_bias_rad_s,
            )
        except (ValueError, TypeError) as error:
            self.get_logger().warning(f"Invalid IMU sample: {error}", throttle_duration_sec=2.0)
            return

        message = Imu()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        message.orientation.x = float(quaternion[0])
        message.orientation.y = float(quaternion[1])
        message.orientation.z = float(quaternion[2])
        message.orientation.w = float(quaternion[3])
        message.angular_velocity.x = float(body_rate[0])
        message.angular_velocity.y = float(body_rate[1])
        message.angular_velocity.z = float(body_rate[2])
        message.linear_acceleration_covariance[0] = -1.0
        self._publisher.publish(message)
        self._last_source_time = source_time

    def destroy_node(self) -> None:
        if hasattr(self, "_imu"):
            try:
                self._imu._dev.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
