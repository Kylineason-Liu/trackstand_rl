"""50 Hz Trackstand-Balance-v1 Beta policy and RC/kickstand state machine."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray, MultiArrayDimension, String

from .contracts import (
    BalanceMode,
    BalanceStateMachine,
    FALL_ANGLE_RAD,
    OBSERVATION_NAMES,
    OBSERVATION_SIZE,
    POLICY_RATE_HZ,
    STEERING_ACTION_SCALE_RAD,
    SteeringTargetLimiter,
    build_observation,
    rotate_vector_xyzw,
    quaternion_inverse_xyzw,
    scale_policy_action,
)
from .policy import OnnxPolicy
from .telemetry import MOTOR_STATE_FIELDS, MOTOR_STATE_LABEL, POLICY_TIMING_FIELDS, age_ms, telemetry_values


# This export embeds its weights; the ONNX checksum covers the complete policy.
MODEL_FILENAME = "trackstand_rl_rsl_2026-10-21_11-18-29_model_2500.onnx"
MODEL_SHA256 = "d9701945b48e62fec14ea3258cf4255b0c7541d530eb8ec7789a770e9a5a63b2"


class PolicyController(Node):
    def __init__(self) -> None:
        super().__init__("trackstand_policy_controller")
        default_model = os.path.join(
            get_package_share_directory("trackstand_deploy"), "models", MODEL_FILENAME
        )
        defaults = {
            "model_path": default_model, "model_sha256": MODEL_SHA256, "policy_rate_hz": POLICY_RATE_HZ,
            "imu_timeout_s": 0.06, "motor_timeout_s": 0.06, "rc_timeout_s": 0.15,
            "max_roll_rad": FALL_ANGLE_RAD, "max_steering_target_rad": STEERING_ACTION_SCALE_RAD,
            "max_steering_start_rad": math.radians(5.0),
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default, ParameterDescriptor(read_only=True))

        model_path = str(self.get_parameter("model_path").value)
        policy_rate_hz = float(self.get_parameter("policy_rate_hz").value)
        self._imu_timeout_s = float(self.get_parameter("imu_timeout_s").value)
        self._motor_timeout_s = float(self.get_parameter("motor_timeout_s").value)
        self._rc_timeout_s = float(self.get_parameter("rc_timeout_s").value)
        self._max_roll_rad = float(self.get_parameter("max_roll_rad").value)
        self._max_steering_start_rad = float(self.get_parameter("max_steering_start_rad").value)
        self._max_steering_target_rad = float(
            self.get_parameter("max_steering_target_rad").value
        )
        if policy_rate_hz != POLICY_RATE_HZ:
            raise ValueError("Trackstand-Balance-v1 policy_rate_hz must remain exactly 50.0")
        if (
            not all(math.isfinite(value) for value in (
                self._imu_timeout_s, self._motor_timeout_s, self._rc_timeout_s,
                self._max_roll_rad, self._max_steering_target_rad, self._max_steering_start_rad,
            ))
            or min(self._imu_timeout_s, self._motor_timeout_s, self._rc_timeout_s) <= 0.0
            or not 0.0 < self._max_roll_rad < math.pi / 2
            or not 0.0 < self._max_steering_start_rad < STEERING_ACTION_SCALE_RAD
            or self._max_steering_target_rad != STEERING_ACTION_SCALE_RAD
        ):
            raise ValueError("invalid policy timeouts/safety limits; v1 steering target limit must be 1.57 rad")
        self._model_path = model_path
        self._model_sha256 = str(self.get_parameter("model_sha256").value)
        if self._sha256(model_path) != self._model_sha256:
            raise ValueError(f"model checksum mismatch for {model_path}")
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
        self._last_observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        self._last_raw_action = np.zeros(2, dtype=np.float32)
        self._steering_limiter = SteeringTargetLimiter()
        self._rear_episode_zero_rad: float | None = None
        self._episode_index = 0
        self._policy_sequence = 0
        self._last_policy_monotonic = 0.0
        self._last_policy_event_state = None
        self._policy_fault_reasons: list[str] = []

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
        self._timing_publisher = self.create_publisher(
            Float64MultiArray, "/trackstand/policy_timing",
            QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._policy_event_publisher = self.create_publisher(String, "/trackstand/policy_event", 100)
        self._timing_dimension = MultiArrayDimension(
            label="policy_timing_v1:" + ",".join(POLICY_TIMING_FIELDS),
            size=len(POLICY_TIMING_FIELDS), stride=len(POLICY_TIMING_FIELDS),
        )
        self.create_subscription(Imu, "/trackstand/imu", self._imu_callback, 1)
        self.create_subscription(
            Float64MultiArray, "/trackstand/motor_state", self._motor_callback, 1
        )
        self.create_subscription(Bool, "/trackstand/kickstands_up", self._rc_callback, 10)
        self.create_subscription(Bool, "/trackstand/motor_fault", self._motor_fault_callback, 10)
        self._policy_timer = self.create_timer(1.0 / policy_rate_hz, self._policy_tick)
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._command_kickstands(False)
        self._publish_motor_command(False, 0.0, 0.0)
        self.get_logger().info(
            f"Loaded v1 {os.path.basename(model_path)} ({self._model_sha256[:12]}...), "
            "11 observations, Beta means, policy/target slew=50 Hz; starts disabled"
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
        if (len(message.data) != len(MOTOR_STATE_FIELDS) or not message.layout.dim
                or message.layout.dim[0].label != MOTOR_STATE_LABEL):
            # Old nodes wrap A3 at +/-pi. Never silently combine that v0 state
            # contract with episode-relative v1 observation construction.
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
            self._policy_fault_reasons = []
            self._rear_episode_zero_rad = None
            self._publish_motor_command(False, 0.0, 0.0)
            self._command_kickstands(False)
            return
        mode = self._state_machine.set_rc(True, self._ready(time.monotonic()))
        if mode == BalanceMode.BALANCING:
            if self._rear_episode_zero_rad is None:
                self._begin_episode()
            self._command_kickstands(True)
        elif mode == BalanceMode.DOWN:
            self._command_kickstands(False)

    def _motor_fault_callback(self, message: Bool) -> None:
        self._motor_fault = bool(message.data)

    def _begin_episode(self) -> None:
        """Reset software episode state without changing the straight steering origin.

        Motor state is continuous since node startup; only rear displacement is
        rebased. Never relabel a physically turned front wheel as straight.
        """
        assert self._motor_state is not None
        self._rear_episode_zero_rad = float(self._motor_state[1])
        self._steering_limiter.reset(float(self._motor_state[0]))
        self._episode_index += 1

    def _ready(self, now: float) -> bool:
        return not self._readiness_reasons(now)

    def _readiness_reasons(self, now: float) -> list[str]:
        reasons = []
        if (
            self._world_from_bike is None
            or self._body_rate_rad_s is None
            or self._motor_state is None
            or self._rc_monotonic <= 0.0
        ):
            if self._world_from_bike is None or self._body_rate_rad_s is None:
                reasons.append("imu_not_received")
            if self._motor_state is None:
                reasons.append("motor_state_not_received")
            if self._rc_monotonic <= 0.0:
                reasons.append("rc_not_received")
            return reasons
        # A stopped motor I/O node cannot publish a new motor_fault message.
        if now - self._imu_monotonic > self._imu_timeout_s:
            reasons.append("imu_timeout")
        if now - self._motor_monotonic > self._motor_timeout_s:
            reasons.append("motor_state_timeout")
        if now - self._rc_monotonic > self._rc_timeout_s:
            reasons.append("rc_timeout")
        if self._motor_fault:
            reasons.append("motor_fault")
        if reasons:
            return reasons
        gravity_body = rotate_vector_xyzw(
            quaternion_inverse_xyzw(self._world_from_bike), np.array([0.0, 0.0, -1.0])
        )
        if abs(gravity_body[1]) > math.sin(self._max_roll_rad) or gravity_body[2] > 0.0:
            return ["roll_or_inversion_safety"]
        if (self._state_machine.mode == BalanceMode.DOWN
                and abs(self._motor_state[0]) > self._max_steering_start_rad):
            return ["steering_not_centered_for_start"]
        return []

    def _policy_tick(self) -> None:
        now = time.monotonic()
        self._policy_sequence += 1
        interval_ms = age_ms(now, self._last_policy_monotonic)
        self._last_policy_monotonic = now
        reasons = self._readiness_reasons(now)
        ready = not reasons
        was_balancing = self._state_machine.enabled
        observation_ms = inference_ms = processing_ms = math.nan
        inference_success = False
        try:
            self._state_machine.update_ready(ready)
            if was_balancing and not self._state_machine.enabled:
                self._policy_fault_reasons = reasons.copy()
            if not self._state_machine.enabled:
                self._rear_episode_zero_rad = None
                self._publish_motor_command(False, 0.0, 0.0)
                return
            if self._rear_episode_zero_rad is None:
                self._begin_episode()
            self._command_kickstands(True)
            assert self._world_from_bike is not None
            assert self._body_rate_rad_s is not None
            assert self._motor_state is not None
            try:
                started = time.monotonic()
                try:
                    observation = build_observation(
                        self._world_from_bike, self._body_rate_rad_s,
                        steering_position_rad=self._motor_state[0],
                        rear_wheel_position_rad=self._motor_state[1] - self._rear_episode_zero_rad,
                        steering_velocity_rad_s=self._motor_state[2], rear_wheel_velocity_rad_s=self._motor_state[3],
                    )
                finally:
                    observation_ms = (time.monotonic() - started) * 1000.0
                started = time.monotonic()
                try:
                    raw_action = self._policy.act(observation)
                finally:
                    inference_ms = (time.monotonic() - started) * 1000.0
                started = time.monotonic()
                try:
                    steering_target, drive_torque = scale_policy_action(raw_action, self._max_steering_target_rad)
                    steering_target = self._steering_limiter.update(steering_target)
                    # Do not timestamp an old inference as a fresh motor command
                    # if a stalled CPU made its source observations expire.
                    late_reasons = self._readiness_reasons(time.monotonic())
                    if late_reasons:
                        raise RuntimeError("inputs expired during inference: " + ", ".join(late_reasons))
                finally:
                    processing_ms = (time.monotonic() - started) * 1000.0
            except Exception as error:
                self.get_logger().error(f"Policy inference fault: {error}")
                self._state_machine.mode = BalanceMode.FAULT_LATCHED
                self._policy_fault_reasons = ["policy_inference_error"]
                self._publish_motor_command(False, 0.0, 0.0)
                return
            inference_success = True
            self._last_observation = observation
            self._last_raw_action = raw_action
            message = Float32MultiArray(data=observation.tolist())
            message.layout.dim = [MultiArrayDimension(
                label="trackstand_v1:" + ",".join(OBSERVATION_NAMES), size=OBSERVATION_SIZE, stride=OBSERVATION_SIZE,
            )]
            self._observation_publisher.publish(message)
            self._action_publisher.publish(Float32MultiArray(data=raw_action.tolist()))
            self._publish_motor_command(True, steering_target, drive_torque)
        finally:
            timing = Float64MultiArray()
            timing.layout.dim = [self._timing_dimension]
            ros_time_s = self.get_clock().now().nanoseconds * 1e-9
            timing.data = telemetry_values(POLICY_TIMING_FIELDS, {
                "sequence": self._policy_sequence, "ros_time_s": ros_time_s, "monotonic_time_s": now,
                "cycle_interval_ms": interval_ms, "work_duration_ms": (time.monotonic() - now) * 1000.0,
                "observation_build_ms": observation_ms, "inference_ms": inference_ms,
                "action_processing_ms": processing_ms, "inference_success": inference_success,
                "ready": ready, "policy_enabled": self._state_machine.enabled,
                "imu_age_ms": age_ms(now, self._imu_monotonic), "motor_age_ms": age_ms(now, self._motor_monotonic),
                "rc_age_ms": age_ms(now, self._rc_monotonic), "motor_fault": self._motor_fault,
            })
            self._timing_publisher.publish(timing)
            event_state = (self._state_machine.mode, ready, tuple(reasons), tuple(self._policy_fault_reasons),
                           self._episode_index)
            if event_state != self._last_policy_event_state:
                event = String()
                event.data = json.dumps({
                    "sequence": self._policy_sequence, "ros_time_s": ros_time_s, "monotonic_time_s": now,
                    "mode": self._state_machine.mode.value, "ready": ready,
                    "readiness_reasons": reasons, "latched_fault_reasons": self._policy_fault_reasons,
                    "task": "Trackstand-Balance-v1", "episode": self._episode_index,
                    "rear_episode_zero_rad": self._rear_episode_zero_rad,
                }, sort_keys=True)
                self._policy_event_publisher.publish(event)
                self._last_policy_event_state = event_state

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
                "task": "Trackstand-Balance-v1",
                "model": os.path.basename(self._model_path),
                "model_sha256": self._model_sha256,
                "observation_names": OBSERVATION_NAMES,
                "episode": self._episode_index,
                "rear_episode_zero_rad": self._rear_episode_zero_rad,
                "held_steering_target_rad": self._steering_limiter.target_rad,
            },
            sort_keys=True,
        )
        self._status_publisher.publish(message)

    @staticmethod
    def _age_ms(now: float, sample_time: float) -> float | None:
        return None if sample_time <= 0.0 else round((now - sample_time) * 1000.0, 2)

    def destroy_node(self) -> None:
        if self.context.ok():
            self._publish_motor_command(False, 0.0, 0.0)
            self._command_kickstands(False)
        super().destroy_node()


def main(args=None) -> None:
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = PolicyController()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
