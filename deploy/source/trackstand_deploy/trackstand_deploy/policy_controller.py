"""50 Hz Trackstand-Balance-v0 policy and RC/kickstand state machine."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray, String

from .contracts import (
    BalanceMode,
    BalanceStateMachine,
    build_observation,
    rotate_vector_xyzw,
    quaternion_inverse_xyzw,
    scale_policy_action,
)
from .policy import OnnxPolicy


MODEL_FILENAME = "trackstand_rl_rsl_2026-09-17_19-46-37_model_41050.onnx"
MODEL_SHA256 = "4de70e7cb3d183f265b3063298f41c0c82545e59bc23c071b0eaa54788e929b2"
MODEL_DATA_SHA256 = "207b77ba1454021dcbbdfb890cae14d305008800cd054e1e9ea107cae4c27d26"


class PolicyController(Node):
    def __init__(self) -> None:
        super().__init__("trackstand_policy_controller")
        default_model = os.path.join(
            get_package_share_directory("trackstand_deploy"), "models", MODEL_FILENAME
        )
        self.declare_parameter("model_path", default_model)
        self.declare_parameter("policy_rate_hz", 50.0)
        self.declare_parameter("imu_timeout_s", 0.06)
        self.declare_parameter("motor_timeout_s", 0.06)
        self.declare_parameter("rc_timeout_s", 0.15)
        self.declare_parameter("wheel_radius_m", 0.07)
        self.declare_parameter("max_roll_rad", math.radians(30.0))
        self.declare_parameter("max_steering_target_rad", math.radians(30.0))

        model_path = str(self.get_parameter("model_path").value)
        policy_rate_hz = float(self.get_parameter("policy_rate_hz").value)
        self._imu_timeout_s = float(self.get_parameter("imu_timeout_s").value)
        self._motor_timeout_s = float(self.get_parameter("motor_timeout_s").value)
        self._rc_timeout_s = float(self.get_parameter("rc_timeout_s").value)
        self._wheel_radius_m = float(self.get_parameter("wheel_radius_m").value)
        self._max_roll_rad = float(self.get_parameter("max_roll_rad").value)
        self._max_steering_target_rad = float(
            self.get_parameter("max_steering_target_rad").value
        )
        if policy_rate_hz != 50.0:
            raise ValueError("Trackstand-Balance-v0 policy_rate_hz must remain exactly 50.0")
        if (
            min(self._imu_timeout_s, self._motor_timeout_s, self._rc_timeout_s) <= 0.0
            or self._wheel_radius_m <= 0.0
            or min(self._max_roll_rad, self._max_steering_target_rad) <= 0.0
        ):
            raise ValueError("policy timeouts, wheel radius, and safety limits must be positive")
        model_data_path = model_path + ".data"
        if self._sha256(model_path) != MODEL_SHA256:
            raise ValueError(f"model checksum mismatch for {model_path}")
        if self._sha256(model_data_path) != MODEL_DATA_SHA256:
            raise ValueError(f"model external-data checksum mismatch for {model_data_path}")
        self._policy = OnnxPolicy(model_path)

        self._world_from_bike: np.ndarray | None = None
        self._body_rate_rad_s: np.ndarray | None = None
        self._motor_state: np.ndarray | None = None
        self._imu_monotonic = 0.0
        self._motor_monotonic = 0.0
        self._rc_monotonic = 0.0
        self._motor_fault = False
        self._state_machine = BalanceStateMachine()
        self._last_kickstand_command: str | None = None
        self._last_observation = np.zeros(13, dtype=np.float32)
        self._last_raw_action = np.zeros(2, dtype=np.float32)

        self._motor_publisher = self.create_publisher(
            Float64MultiArray, "/trackstand/motor_command", 10
        )
        kickstand_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._kickstand_publisher = self.create_publisher(
            String, "/trackstand/kickstands_command", kickstand_qos
        )
        self._observation_publisher = self.create_publisher(
            Float32MultiArray, "/trackstand/observation", 10
        )
        self._action_publisher = self.create_publisher(
            Float32MultiArray, "/trackstand/raw_action", 10
        )
        self._status_publisher = self.create_publisher(String, "/trackstand/status", 10)
        self.create_subscription(Imu, "/trackstand/imu", self._imu_callback, 10)
        self.create_subscription(
            Float64MultiArray, "/trackstand/motor_state", self._motor_callback, 10
        )
        self.create_subscription(Bool, "/trackstand/kickstands_up", self._rc_callback, 10)
        self.create_subscription(Bool, "/trackstand/motor_fault", self._motor_fault_callback, 10)
        self._policy_timer = self.create_timer(1.0 / policy_rate_hz, self._policy_tick)
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._command_kickstands(False)
        self._publish_motor_command(False, 0.0, 0.0)
        self.get_logger().info(
            f"Loaded {MODEL_FILENAME} ({MODEL_SHA256[:12]}...), policy=50 Hz; starts disabled"
        )

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _imu_callback(self, message: Imu) -> None:
        quaternion = np.array(
            [message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w],
            dtype=float,
        )
        body_rate = np.array(
            [message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z],
            dtype=float,
        )
        if np.all(np.isfinite(quaternion)) and np.linalg.norm(quaternion) > 0.5 and np.all(np.isfinite(body_rate)):
            self._world_from_bike = quaternion / np.linalg.norm(quaternion)
            self._body_rate_rad_s = body_rate
            self._imu_monotonic = time.monotonic()

    def _motor_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) < 4:
            return
        state = np.asarray(message.data[:4], dtype=float)
        if np.all(np.isfinite(state)):
            self._motor_state = state
            self._motor_monotonic = time.monotonic()

    def _rc_callback(self, message: Bool) -> None:
        self._rc_monotonic = time.monotonic()
        up = bool(message.data)
        if not up:
            self._state_machine.set_rc(False, self._ready(time.monotonic()))
            self._publish_motor_command(False, 0.0, 0.0)
            self._command_kickstands(False)
            return
        mode = self._state_machine.set_rc(True, self._ready(time.monotonic()))
        if mode == BalanceMode.BALANCING:
            self._command_kickstands(True)
        elif mode == BalanceMode.DOWN:
            self._command_kickstands(False)

    def _motor_fault_callback(self, message: Bool) -> None:
        self._motor_fault = bool(message.data)

    def _ready(self, now: float) -> bool:
        if (
            self._world_from_bike is None
            or self._body_rate_rad_s is None
            or self._motor_state is None
            or self._rc_monotonic <= 0.0
        ):
            return False
        if (
            now - self._imu_monotonic > self._imu_timeout_s
            or now - self._motor_monotonic > self._motor_timeout_s
            or now - self._rc_monotonic > self._rc_timeout_s
            or self._motor_fault
        ):
            return False
        gravity_body = rotate_vector_xyzw(
            quaternion_inverse_xyzw(self._world_from_bike), np.array([0.0, 0.0, -1.0])
        )
        fallen = abs(float(gravity_body[1])) > math.sin(self._max_roll_rad) or gravity_body[2] > 0.0
        steering_over_limit = abs(float(self._motor_state[0])) > self._max_steering_target_rad
        return not fallen and not steering_over_limit

    def _policy_tick(self) -> None:
        now = time.monotonic()
        self._state_machine.update_ready(self._ready(now))
        if not self._state_machine.enabled:
            self._publish_motor_command(False, 0.0, 0.0)
            return
        self._command_kickstands(True)
        assert self._world_from_bike is not None
        assert self._body_rate_rad_s is not None
        assert self._motor_state is not None
        try:
            observation = build_observation(
                self._world_from_bike,
                self._body_rate_rad_s,
                steering_position_rad=self._motor_state[0],
                rear_wheel_position_rad=self._motor_state[1],
                steering_velocity_rad_s=self._motor_state[2],
                rear_wheel_velocity_rad_s=self._motor_state[3],
                wheel_radius_m=self._wheel_radius_m,
            )
            raw_action = self._policy.act(observation)
            steering_target, drive_torque = scale_policy_action(
                raw_action, self._max_steering_target_rad
            )
        except Exception as error:
            self.get_logger().error(f"Policy inference fault: {error}")
            self._state_machine.mode = BalanceMode.FAULT_LATCHED
            self._publish_motor_command(False, 0.0, 0.0)
            return
        self._last_observation = observation
        self._last_raw_action = raw_action
        self._observation_publisher.publish(Float32MultiArray(data=observation.tolist()))
        self._action_publisher.publish(Float32MultiArray(data=raw_action.tolist()))
        self._publish_motor_command(True, steering_target, drive_torque)

    def _publish_motor_command(self, enable: bool, steering_target: float, drive_torque: float) -> None:
        message = Float64MultiArray()
        message.data = [1.0 if enable else 0.0, float(steering_target), float(drive_torque)]
        self._motor_publisher.publish(message)

    def _command_kickstands(self, up: bool) -> None:
        command = "up" if up else "down"
        if command == self._last_kickstand_command:
            return
        self._kickstand_publisher.publish(String(data=command))
        self._last_kickstand_command = command
        self.get_logger().info(f"Kickstands commanded {command}")

    def _publish_status(self) -> None:
        now = time.monotonic()
        message = String()
        message.data = json.dumps(
            {
                "mode": self._state_machine.mode.value,
                "policy_enabled": self._state_machine.enabled,
                "kickstands_requested_up": self._state_machine.rc_up,
                "ready": self._ready(now),
                "imu_age_ms": self._age_ms(now, self._imu_monotonic),
                "motor_age_ms": self._age_ms(now, self._motor_monotonic),
                "rc_age_ms": self._age_ms(now, self._rc_monotonic),
                "motor_fault": self._motor_fault,
                "model": MODEL_FILENAME,
            },
            sort_keys=True,
        )
        self._status_publisher.publish(message)

    @staticmethod
    def _age_ms(now: float, sample_time: float) -> float | None:
        return None if sample_time <= 0.0 else round((now - sample_time) * 1000.0, 2)

    def destroy_node(self) -> None:
        self._publish_motor_command(False, 0.0, 0.0)
        self._command_kickstands(False)
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
