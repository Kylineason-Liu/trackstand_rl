"""YahBoom serial IMU to corrected ROS 2 Imu samples."""

from __future__ import annotations

import math
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
            or not np.all(np.isfinite(rpy_deg))
            or not np.all(np.isfinite(bias_dps))
            or not math.isfinite(publish_rate_hz)
            or not math.isfinite(self._source_timeout_s)
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
        sample = self._imu.get_timed_sample()
        if sample is None:
            return
        source_time, quaternion_wxyz, gyro_sensor = sample
        source_age = time.monotonic() - source_time
        if (
            source_time <= self._last_source_time
            or source_time <= 0.0
            or not 0.0 <= source_age <= self._source_timeout_s
        ):
            return
        try:
            qw, qx, qy, qz = quaternion_wxyz
            quaternion, body_rate = correct_imu_sample(
                np.array([qx, qy, qz, qw], dtype=float),
                gyro_sensor,
                self._sensor_to_bike,
                self._gyro_bias_rad_s,
            )
            if not np.all(np.isfinite(body_rate)):
                raise ValueError("gyro values must be finite")
        except (ValueError, TypeError) as error:
            self.get_logger().warning(f"Invalid IMU sample: {error}", throttle_duration_sec=2.0)
            return

        message = Imu()
        # Preserve the older serial channel's parse age through ROS delivery;
        # UART buffering and hardware acquisition time remain unmeasured.
        now = self.get_clock().now()
        stamp_ns = now.nanoseconds - round((time.monotonic() - source_time) * 1e9)
        message.header.stamp = now.to_msg()
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
        message.header.frame_id = self._frame_id
        for axis, value in zip("xyzw", quaternion):
            setattr(message.orientation, axis, float(value))
        for axis, value in zip("xyz", body_rate):
            setattr(message.angular_velocity, axis, float(value))
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
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = ImuPublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
