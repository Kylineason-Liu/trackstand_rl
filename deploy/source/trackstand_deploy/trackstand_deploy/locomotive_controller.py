"""50 Hz CycleRL riding policy, RC speed/yaw commands and launch -> ride state machine.

Original 8-in/2-out tensor contract (see locomotive_contracts). Hardware mapping:
calibrated /trackstand/imu, signed /trackstand/motor_state, CRSF channel-7
authority (/trackstand/kickstands_up) and proportional sticks (/locomotive/rc_sticks).
Output goes to /locomotive/motor_command = [enable, steering_target_rad,
rear_velocity_rad_s], consumed only by motor_mit_velocity_io (C1 rear speed).
"""

from __future__ import annotations

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

from .contracts import normalize_quaternion_xyzw, quaternion_inverse_xyzw, rotate_vector_xyzw
from .locomotive_contracts import (
    MANIFEST_FILENAME, OBSERVATION_NAMES, OBSERVATION_SIZE, POLICY_DT_S, POLICY_RATE_HZ,
    CommandShaper, LocomotiveMode, LocomotiveStateMachine, build_observation, load_model_spec,
    scale_policy_action, sha256_file,
)
from .locomotive_policy import LocomotiveOnnxPolicy
from .telemetry import MOTOR_STATE_FIELDS, MOTOR_STATE_LABEL, POLICY_TIMING_FIELDS, age_ms, telemetry_values


# Rank 1 of the CycleRL cylinder top-5 (Isaac 100% / MuJoCo 100% flat BSR).
MODEL_FILENAME = "locomotive_1900_drive03_2026-10-19_16-30-17.onnx"
MODEL_SHA256 = "cbd76d1789b903966ba84f4330224f88efa05940d97e0213bbf178ead3ab1c9c"
TASK = "CycleRL-Trackstand-Cylinder-Paper-v0"
# The commanded launch speed overshoots wake slightly so the measured speed can reach it.
LAUNCH_MARGIN_M_S = 0.1


class LocomotiveController(Node):
    def __init__(self) -> None:
        super().__init__("locomotive_policy_controller")
        models = os.path.join(get_package_share_directory("trackstand_deploy"), "models")
        defaults = {
            "model_filename": MODEL_FILENAME, "model_sha256": MODEL_SHA256, "models_dir": models,
            "policy_rate_hz": POLICY_RATE_HZ, "imu_timeout_s": 0.06, "motor_timeout_s": 0.06,
            "rc_timeout_s": 0.15, "max_roll_rad": 0.5236, "max_steering_start_rad": math.radians(5.0),
            "wake_speed_m_s": 1.0, "min_riding_speed_m_s": 0.3, "launch_accel_m_s2": 0.5,
            "command_accel_m_s2": 1.0, "max_speed_cmd_m_s": 1.5, "max_yaw_rate_cmd_rad_s": 0.5,
            "stick_deadband": 0.05, "yaw_stick_sign": -1.0, "use_policy_drive": True,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default, ParameterDescriptor(read_only=True))
        parameter = lambda name: self.get_parameter(name).value

        model_filename = str(parameter("model_filename"))
        models_dir = str(parameter("models_dir"))
        self._model_path = os.path.join(models_dir, model_filename)
        self._model_sha256 = str(parameter("model_sha256"))
        spec = load_model_spec(os.path.join(models_dir, MANIFEST_FILENAME), model_filename)
        if self._model_sha256 != spec.onnx_sha256 or sha256_file(self._model_path) != self._model_sha256:
            raise ValueError(f"model checksum mismatch for {self._model_path}")
        self._spec = spec
        self._policy = LocomotiveOnnxPolicy(self._model_path, spec)

        if float(parameter("policy_rate_hz")) != POLICY_RATE_HZ:
            raise ValueError("locomotive policy_rate_hz must remain exactly 50.0 (trained 0.01 s x 2)")
        self._imu_timeout_s = float(parameter("imu_timeout_s"))
        self._motor_timeout_s = float(parameter("motor_timeout_s"))
        self._rc_timeout_s = float(parameter("rc_timeout_s"))
        self._max_roll_rad = float(parameter("max_roll_rad"))
        self._max_steering_start_rad = float(parameter("max_steering_start_rad"))
        self._launch_accel_m_s2 = float(parameter("launch_accel_m_s2"))
        self._use_policy_drive = bool(parameter("use_policy_drive"))
        wake = float(parameter("wake_speed_m_s"))
        max_speed = min(float(parameter("max_speed_cmd_m_s")), spec.vel_cmd_max_m_s)
        max_yaw = float(parameter("max_yaw_rate_cmd_rad_s"))
        scalars = (self._imu_timeout_s, self._motor_timeout_s, self._rc_timeout_s, self._max_roll_rad,
                   self._max_steering_start_rad, self._launch_accel_m_s2, wake, max_speed, max_yaw)
        if (not all(map(math.isfinite, scalars))
                or min(self._imu_timeout_s, self._motor_timeout_s, self._rc_timeout_s) <= 0.0
                or not 0.0 < self._max_roll_rad < spec.lean_termination_rad
                or not 0.0 < self._max_steering_start_rad < spec.steering_clip_rad
                or self._launch_accel_m_s2 <= 0.0
                or not spec.vel_cmd_min_m_s <= wake <= max_speed
                or not 0.0 < max_yaw <= spec.yaw_rate_cmd_max_rad_s):
            raise ValueError("invalid locomotive timeouts/limits or commands outside the trained ranges")
        self._shaper = CommandShaper(
            spec.vel_cmd_min_m_s, max_speed, max_yaw, float(parameter("command_accel_m_s2")),
            float(parameter("stick_deadband")), float(parameter("yaw_stick_sign")),
        )
        self._state_machine = LocomotiveStateMachine(wake, float(parameter("min_riding_speed_m_s")))

        self._world_from_bike: np.ndarray | None = None
        self._body_rate_rad_s: np.ndarray | None = None
        self._motor_state: np.ndarray | None = None
        self._sticks: tuple[float, float] | None = None
        self._imu_monotonic = self._motor_monotonic = self._rc_monotonic = self._sticks_monotonic = 0.0
        self._motor_fault = False
        self._launch_speed_m_s = 0.0
        self._last_kickstand_command: str | None = None
        self._last_observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        self._last_raw_action = np.zeros(2, dtype=np.float32)
        self._last_command = (0.0, 0.0, 0.0)
        self._episode_index = self._policy_sequence = 0
        self._last_policy_monotonic = 0.0
        self._last_event_state = None

        self._motor_publisher = self.create_publisher(Float64MultiArray, "/locomotive/motor_command", 10)
        kickstand_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                   durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._kickstand_publisher = self.create_publisher(String, "/trackstand/kickstands_command", kickstand_qos)
        self._observation_publisher = self.create_publisher(Float32MultiArray, "/locomotive/observation", 10)
        self._action_publisher = self.create_publisher(Float32MultiArray, "/locomotive/raw_action", 10)
        self._status_publisher = self.create_publisher(String, "/locomotive/status", 10)
        self._timing_publisher = self.create_publisher(
            Float64MultiArray, "/locomotive/policy_timing",
            QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
        self._event_publisher = self.create_publisher(String, "/locomotive/policy_event", 100)
        self._timing_dimension = MultiArrayDimension(
            label="locomotive_policy_timing_v1:" + ",".join(POLICY_TIMING_FIELDS),
            size=len(POLICY_TIMING_FIELDS), stride=len(POLICY_TIMING_FIELDS))
        self.create_subscription(Imu, "/trackstand/imu", self._imu_callback, 1)
        self.create_subscription(Float64MultiArray, "/trackstand/motor_state", self._motor_callback, 1)
        self.create_subscription(Bool, "/trackstand/kickstands_up", self._rc_callback, 10)
        self.create_subscription(Float32MultiArray, "/locomotive/rc_sticks", self._sticks_callback, 10)
        self.create_subscription(Bool, "/trackstand/motor_fault", self._motor_fault_callback, 10)
        self._policy_timer = self.create_timer(POLICY_DT_S, self._policy_tick)
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._command_kickstands(False)
        self._publish_motor_command(False, 0.0, 0.0)
        self.get_logger().info(
            f"Loaded {model_filename} (rank {spec.rank}, {spec.recipe}, {self._model_sha256[:12]}...): "
            f"8 obs, steer x{spec.steering_scale_rad:.4f} clip {spec.steering_clip_rad:.4f} rad, "
            f"rear x{spec.drive_scale_rad_s:.3f} clip {spec.drive_clip_rad_s:g} rad/s, "
            f"v* {spec.vel_cmd_min_m_s:g}-{max_speed:g} m/s, wake {wake:g} m/s; starts disabled"
        )

    # ---- inputs (same freshness rules as policy_controller) -------------------------------
    def _imu_callback(self, message: Imu) -> None:
        received = time.monotonic()
        stamp = message.header.stamp
        age = (self.get_clock().now().nanoseconds - stamp.sec * 1_000_000_000 - stamp.nanosec) * 1e-9
        if not 0.0 <= age <= self._imu_timeout_s:
            return
        quaternion = np.array([message.orientation.x, message.orientation.y, message.orientation.z,
                               message.orientation.w], dtype=float)
        body_rate = np.array([message.angular_velocity.x, message.angular_velocity.y,
                              message.angular_velocity.z], dtype=float)
        if np.all(np.isfinite(quaternion)) and math.hypot(*quaternion) > 0.5 and np.all(np.isfinite(body_rate)):
            try:
                quaternion = normalize_quaternion_xyzw(quaternion)
            except ValueError:
                return
            self._world_from_bike, self._body_rate_rad_s = quaternion, body_rate
            self._imu_monotonic = received - age

    def _motor_callback(self, message: Float64MultiArray) -> None:
        received = time.monotonic()
        if (len(message.data) != len(MOTOR_STATE_FIELDS) or not message.layout.dim
                or message.layout.dim[0].label != MOTOR_STATE_LABEL):
            return
        state = np.asarray(message.data[:4], dtype=float)
        age = time.time() - float(message.data[4])
        if np.all(np.isfinite(state)) and 0.0 <= age <= self._motor_timeout_s:
            self._motor_state, self._motor_monotonic = state, received - age

    def _sticks_callback(self, message: Float32MultiArray) -> None:
        values = tuple(float(value) for value in message.data)
        if len(values) == 2 and all(math.isfinite(value) and abs(value) <= 1.0 for value in values):
            self._sticks, self._sticks_monotonic = values, time.monotonic()

    def _rc_callback(self, message: Bool) -> None:
        self._rc_monotonic = time.monotonic()
        if not message.data:
            self._state_machine.set_rc(False, False)
            try:
                self._publish_motor_command(False, 0.0, 0.0)
            finally:
                self._command_kickstands(False)
            return
        if self._state_machine.set_rc(True, self._ready(time.monotonic())) == LocomotiveMode.LAUNCH:
            self._begin_launch()

    def _motor_fault_callback(self, message: Bool) -> None:
        self._motor_fault = bool(message.data)
        if self._motor_fault and self._state_machine.enabled:
            self._state_machine.latch("motor_fault")
            self._publish_motor_command(False, 0.0, 0.0)

    # ---- state ----------------------------------------------------------------------------
    def _speed_m_s(self) -> float:
        return 0.0 if self._motor_state is None else self._spec.wheel_radius_m * float(self._motor_state[3])

    def _begin_launch(self) -> None:
        self._launch_speed_m_s = max(0.0, self._speed_m_s())
        self._episode_index += 1

    def _ready(self, now: float) -> bool:
        return not self._readiness_reasons(now)

    def _readiness_reasons(self, now: float) -> list[str]:
        missing = []
        if self._world_from_bike is None or self._body_rate_rad_s is None:
            missing.append("imu_not_received")
        if self._motor_state is None:
            missing.append("motor_state_not_received")
        if self._rc_monotonic <= 0.0:
            missing.append("rc_not_received")
        if self._sticks is None:
            missing.append("rc_sticks_not_received")
        if missing:
            return missing
        reasons = []
        if now - self._imu_monotonic > self._imu_timeout_s:
            reasons.append("imu_timeout")
        if now - self._motor_monotonic > self._motor_timeout_s:
            reasons.append("motor_state_timeout")
        if now - self._rc_monotonic > self._rc_timeout_s:
            reasons.append("rc_timeout")
        if now - self._sticks_monotonic > self._rc_timeout_s:
            reasons.append("rc_sticks_timeout")
        if self._motor_fault:
            reasons.append("motor_fault")
        if reasons:
            return reasons
        gravity_body = rotate_vector_xyzw(quaternion_inverse_xyzw(self._world_from_bike), np.array([0.0, 0.0, -1.0]))
        if abs(gravity_body[1]) > math.sin(self._max_roll_rad) or gravity_body[2] > 0.0:
            return ["roll_or_inversion_safety"]
        if (self._state_machine.mode == LocomotiveMode.DOWN
                and abs(self._motor_state[0]) > self._max_steering_start_rad):
            return ["steering_not_centered_for_start"]
        if self._state_machine.mode == LocomotiveMode.DOWN and self._speed_m_s() < -0.05:
            return ["rolling_backward_at_start"]
        return []

    # ---- 50 Hz tick -----------------------------------------------------------------------
    def _policy_tick(self) -> None:
        now = time.monotonic()
        self._policy_sequence += 1
        interval_ms = age_ms(now, self._last_policy_monotonic)
        self._last_policy_monotonic = now
        reasons = self._readiness_reasons(now)
        ready = not reasons
        observation_ms = inference_ms = processing_ms = math.nan
        inference_success = False
        try:
            previous = self._state_machine.mode
            mode = self._state_machine.update(ready, reasons, self._speed_m_s())
            if mode == LocomotiveMode.LAUNCH and previous != LocomotiveMode.LAUNCH:
                self._begin_launch()
            if not self._state_machine.enabled:
                self._publish_motor_command(False, 0.0, 0.0)
                return
            throttle, yaw = self._sticks
            if mode == LocomotiveMode.LAUNCH:
                # Stands down, fork straight, rear speed ramped: the policy never trained at rest.
                self._command_kickstands(False)
                wake = self._state_machine.wake_speed_m_s
                target = max(self._shaper.target_speed(throttle), wake) + LAUNCH_MARGIN_M_S
                self._launch_speed_m_s = min(target, self._launch_speed_m_s + self._launch_accel_m_s2 * POLICY_DT_S)
                self._publish_motor_command(True, 0.0, self._launch_speed_m_s / self._spec.wheel_radius_m)
                return
            if previous != LocomotiveMode.RIDING:
                self._shaper.reset(self._speed_m_s())  # Start v* at the actual hand-over speed.
            self._command_kickstands(True)
            try:
                started = time.monotonic()
                try:
                    vel_cmd, yaw_rate_cmd = self._shaper.update(throttle, yaw)
                    observation = build_observation(
                        self._world_from_bike, self._body_rate_rad_s, self._motor_state[0], self._motor_state[3],
                        vel_cmd, yaw_rate_cmd, self._spec.wheel_radius_m)
                finally:
                    observation_ms = (time.monotonic() - started) * 1000.0
                started = time.monotonic()
                try:
                    raw_action = self._policy.act(observation)
                finally:
                    inference_ms = (time.monotonic() - started) * 1000.0
                started = time.monotonic()
                try:
                    steering_target, rear_velocity = scale_policy_action(raw_action, self._spec)
                    if not self._use_policy_drive:
                        # CycleRL bike option: hub follows v* directly; a1 is still logged.
                        rear_velocity = max(-self._spec.drive_clip_rad_s, min(
                            self._spec.drive_clip_rad_s, vel_cmd / self._spec.wheel_radius_m))
                    late_reasons = self._readiness_reasons(time.monotonic())
                    if late_reasons:
                        raise RuntimeError("inputs expired during inference: " + ", ".join(late_reasons))
                finally:
                    processing_ms = (time.monotonic() - started) * 1000.0
            except Exception as error:
                self.get_logger().error(f"Locomotive policy fault: {error}")
                self._state_machine.latch("policy_inference_error")
                self._publish_motor_command(False, 0.0, 0.0)
                return
            inference_success = True
            self._last_observation, self._last_raw_action = observation, raw_action
            self._publish_motor_command(True, steering_target, rear_velocity)
            message = Float32MultiArray(data=observation.tolist())
            message.layout.dim = [MultiArrayDimension(
                label="locomotive_v1:" + ",".join(OBSERVATION_NAMES), size=OBSERVATION_SIZE, stride=OBSERVATION_SIZE)]
            self._observation_publisher.publish(message)
            self._action_publisher.publish(Float32MultiArray(data=raw_action.tolist()))
        finally:
            self._publish_timing(now, interval_ms, observation_ms, inference_ms, processing_ms,
                                 inference_success, ready)
            self._publish_event(now, ready, reasons)

    # ---- outputs --------------------------------------------------------------------------
    def _publish_timing(self, now, interval_ms, observation_ms, inference_ms, processing_ms, success, ready):
        timing = Float64MultiArray()
        timing.layout.dim = [self._timing_dimension]
        timing.data = telemetry_values(POLICY_TIMING_FIELDS, {
            "sequence": self._policy_sequence, "ros_time_s": self.get_clock().now().nanoseconds * 1e-9,
            "monotonic_time_s": now, "cycle_interval_ms": interval_ms,
            "work_duration_ms": (time.monotonic() - now) * 1000.0, "observation_build_ms": observation_ms,
            "inference_ms": inference_ms, "action_processing_ms": processing_ms, "inference_success": success,
            "ready": ready, "policy_enabled": self._state_machine.riding,
            "imu_age_ms": age_ms(now, self._imu_monotonic), "motor_age_ms": age_ms(now, self._motor_monotonic),
            "rc_age_ms": max(age_ms(now, self._rc_monotonic), age_ms(now, self._sticks_monotonic)),
            "motor_fault": self._motor_fault,
        })
        self._timing_publisher.publish(timing)

    def _publish_event(self, now, ready, reasons) -> None:
        state = (self._state_machine.mode, ready, tuple(reasons), self._state_machine.fault_reasons,
                 self._episode_index)
        if state == self._last_event_state:
            return
        self._event_publisher.publish(String(data=json.dumps({
            "sequence": self._policy_sequence, "monotonic_time_s": now, "mode": self._state_machine.mode.value,
            "ready": ready, "readiness_reasons": reasons,
            "latched_fault_reasons": list(self._state_machine.fault_reasons),
            "task": TASK, "model": self._spec.onnx, "episode": self._episode_index, "speed_m_s": self._speed_m_s(),
        }, sort_keys=True)))
        self._last_event_state = state

    def _publish_motor_command(self, enable: bool, steering_target: float, rear_velocity: float) -> None:
        self._last_command = (1.0 if enable else 0.0, float(steering_target), float(rear_velocity))
        self._motor_publisher.publish(Float64MultiArray(data=list(self._last_command)))

    def _command_kickstands(self, up: bool) -> None:
        command = "up" if up else "down"
        if command == self._last_kickstand_command:
            return
        self._kickstand_publisher.publish(String(data=command))
        self._last_kickstand_command = command
        self.get_logger().info(f"Kickstands commanded {command}")

    def _publish_status(self) -> None:
        now = time.monotonic()
        self._status_publisher.publish(String(data=json.dumps({
            "mode": self._state_machine.mode.value, "policy_enabled": self._state_machine.riding,
            "kickstands_requested_up": self._state_machine.rc_up, "ready": self._ready(now),
            "imu_age_ms": age_ms(now, self._imu_monotonic), "motor_age_ms": age_ms(now, self._motor_monotonic),
            "rc_age_ms": age_ms(now, self._rc_monotonic), "rc_sticks_age_ms": age_ms(now, self._sticks_monotonic),
            "motor_fault": self._motor_fault, "task": TASK, "model": self._spec.onnx, "model_rank": self._spec.rank,
            "model_sha256": self._model_sha256, "observation_names": OBSERVATION_NAMES,
            "episode": self._episode_index, "speed_m_s": self._speed_m_s(),
            "launch_speed_m_s": self._launch_speed_m_s, "vel_cmd_m_s": self._shaper.vel_cmd_m_s,
            "use_policy_drive": self._use_policy_drive, "last_motor_command": list(self._last_command),
            "latched_fault_reasons": list(self._state_machine.fault_reasons),
        }, sort_keys=True)))

    def destroy_node(self) -> None:
        try:
            if self.context.ok():
                try:
                    self._publish_motor_command(False, 0.0, 0.0)
                finally:
                    self._command_kickstands(False)
        finally:
            super().destroy_node()


def main(args=None) -> None:
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = LocomotiveController()
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
