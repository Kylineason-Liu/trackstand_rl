"""FEETECH dual-servo driver for the bicycle kickstands."""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .servo_control import SMS_STS


class KickstandDriver(Node):
    def __init__(self) -> None:
        super().__init__("trackstand_kickstand_driver")
        self.declare_parameter("serial_port", "/dev/ttyAMA2")
        self.declare_parameter("baud_rate", 1000000)
        self.declare_parameter("servo_ids", [1, 2])
        self.declare_parameter("servo_up_positions", [1648, 2454])
        self.declare_parameter("servo_down_positions", [1972, 2085])
        self.declare_parameter("status_rate_hz", 0.0)
        self.declare_parameter("disable_torque_on_shutdown", False)

        serial_port = str(self.get_parameter("serial_port").value)
        baud_rate = int(self.get_parameter("baud_rate").value)
        self._servo_ids = [int(value) for value in self.get_parameter("servo_ids").value]
        self._up_positions = [
            int(value) for value in self.get_parameter("servo_up_positions").value
        ]
        self._down_positions = [
            int(value) for value in self.get_parameter("servo_down_positions").value
        ]
        status_rate_hz = float(self.get_parameter("status_rate_hz").value)
        self._disable_on_shutdown = bool(self.get_parameter("disable_torque_on_shutdown").value)
        positions = self._up_positions + self._down_positions
        if (
            self._servo_ids != [1, 2]
            or len(self._up_positions) != 2
            or len(self._down_positions) != 2
            or any(position < 0 or position > 4095 for position in positions)
            or status_rate_hz < 0.0
        ):
            raise ValueError(
                "kickstands require servo_ids [1, 2], two 0..4095 targets per pose, "
                "and a nonnegative status rate"
            )

        self._servos = SMS_STS()
        if not self._servos.begin(baud_rate, serial_port):
            raise RuntimeError(f"failed to open kickstand serial port {serial_port}")
        self._status_publisher = self.create_publisher(String, "/trackstand/kickstands_status", 10)
        command_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._command_subscription = self.create_subscription(
            String, "/trackstand/kickstands_command", self._command_callback, command_qos
        )
        self._timer = None
        if status_rate_hz > 0.0:
            self._timer = self.create_timer(1.0 / status_rate_hz, self._publish_status)
        self._command_down()
        self.get_logger().info(f"Kickstand driver ready on {serial_port}; startup position is down")

    def _move(self, servo_id: int, position: int, speed: int = 3000, acceleration: int = 255) -> None:
        if not self._servos.write_pos_ex(servo_id, position, speed, acceleration):
            self.get_logger().warning(f"No acknowledgement from kickstand servo {servo_id}")

    def _command_positions(self, positions) -> None:
        # Enable BOTH servos before moving either, preserving the original order.
        for servo_id in self._servo_ids:
            self._servos.enable_torque(servo_id, 1)
        for servo_id, position in zip(self._servo_ids, positions):
            self._move(servo_id, position)

    def _command_down(self) -> None:
        self._command_positions(self._down_positions)

    def _command_up(self) -> None:
        self._command_positions(self._up_positions)

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == "up":
            self._command_up()
            self.get_logger().info("Kickstands commanded up")
        elif command == "down":
            self._command_down()
            self.get_logger().info("Kickstands commanded down")
        else:
            self.get_logger().warning(f"Unknown kickstand command: {command!r}")

    def _publish_status(self) -> None:
        for servo_id in self._servo_ids:
            if self._servos.feedback(servo_id) == -1:
                self.get_logger().warning(
                    f"Kickstand servo {servo_id} feedback failed", throttle_duration_sec=5.0
                )
                continue
            message = String()
            message.data = json.dumps(
                {
                    "id": servo_id,
                    "position": self._servos.read_pos(-1),
                    "speed": self._servos.read_speed(-1),
                    "load": self._servos.read_load(-1),
                    "voltage": self._servos.read_voltage(-1),
                    "temperature": self._servos.read_temper(-1),
                    "moving": self._servos.read_move(-1),
                    "current": self._servos.read_current(-1),
                },
                sort_keys=True,
            )
            self._status_publisher.publish(message)

    def destroy_node(self) -> None:
        if hasattr(self, "_servos"):
            try:
                self._command_down()
                if self._disable_on_shutdown:
                    for servo_id in self._servo_ids:
                        self._servos.enable_torque(servo_id, 0)
                self._servos.end()
            except Exception:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KickstandDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
