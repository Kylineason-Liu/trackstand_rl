"""200 Hz GIM encoder polling and actuator-law execution over SocketCAN."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import threading
import time

import can
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray, String

from .can_protocol import (
    parse_a2_velocity,
    parse_a3_multi_turn_count,
    parse_a3_position,
    parse_b0_torque_constant,
    q_current_payload,
    query,
)
from .contracts import (
    DRIVE_DAMPING_NM_S_PER_RAD,
    DRIVE_EFFORT_LIMIT_NM,
    DRIVE_VELOCITY_LIMIT_RAD_S,
    STEERING_EFFORT_LIMIT_NM,
    STEERING_KD_NM_S_PER_RAD,
    STEERING_KP_NM_PER_RAD,
    STEERING_VELOCITY_LIMIT_RAD_S,
)


@dataclass
class ChannelState:
    name: str
    host_id: int
    feedback_id: int
    gear_ratio: float
    encoder_sign: float
    torque_sign: float
    zero_count: int | None = None
    position_rad: float = 0.0
    velocity_rad_s: float = 0.0
    torque_constant_nm_per_a: float | None = None
    sample_monotonic: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class MotorIo(Node):
    def __init__(self) -> None:
        super().__init__("trackstand_motor_io")
        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("steering_host_id", "102")
        self.declare_parameter("steering_feedback_id", "002")
        self.declare_parameter("drive_host_id", "101")
        self.declare_parameter("drive_feedback_id", "001")
        self.declare_parameter("steering_gear_ratio", 1.0)
        self.declare_parameter("drive_gear_ratio", 1.0)
        self.declare_parameter("steering_encoder_sign", 1.0)
        self.declare_parameter("drive_encoder_sign", -1.0)
        self.declare_parameter("steering_torque_sign", 1.0)
        self.declare_parameter("drive_torque_sign", -1.0)
        self.declare_parameter("poll_rate_hz", 200.0)
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("command_timeout_s", 0.08)
        self.declare_parameter("feedback_timeout_s", 0.05)
        self.declare_parameter("steering_limit_rad", math.radians(30.0))
        self.declare_parameter("dry_run", False)

        self._can_interface = str(self.get_parameter("can_interface").value)
        self._poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        self._control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self._command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self._feedback_timeout_s = float(self.get_parameter("feedback_timeout_s").value)
        self._steering_limit_rad = float(self.get_parameter("steering_limit_rad").value)
        self._dry_run = bool(self.get_parameter("dry_run").value)
        if (
            min(self._poll_rate_hz, self._control_rate_hz) <= 0.0
            or min(self._command_timeout_s, self._feedback_timeout_s) <= 0.0
            or self._steering_limit_rad <= 0.0
        ):
            raise ValueError("motor rates, timeouts, and steering limit must be positive")

        def channel(prefix: str, name: str) -> ChannelState:
            return ChannelState(
                name=name,
                host_id=int(str(self.get_parameter(f"{prefix}_host_id").value), 16),
                feedback_id=int(str(self.get_parameter(f"{prefix}_feedback_id").value), 16),
                gear_ratio=float(self.get_parameter(f"{prefix}_gear_ratio").value),
                encoder_sign=float(self.get_parameter(f"{prefix}_encoder_sign").value),
                torque_sign=float(self.get_parameter(f"{prefix}_torque_sign").value),
            )

        self._steering = channel("steering", "steering")
        self._drive = channel("drive", "drive")
        for motor in (self._steering, self._drive):
            if motor.gear_ratio <= 0.0 or motor.encoder_sign not in (-1.0, 1.0):
                raise ValueError(f"invalid encoder configuration for {motor.name}")
            if motor.torque_sign not in (-1.0, 1.0):
                raise ValueError(f"invalid torque sign for {motor.name}")

        self._running = True
        self._requested_enable = False
        self._steering_target_rad = 0.0
        self._slewed_steering_target_rad = 0.0
        self._drive_target_nm = 0.0
        self._command_monotonic = 0.0
        self._fault_latched = False
        self._outputs_active = False
        self._command_lock = threading.Lock()
        self._last_published_sample_times = (0.0, 0.0)

        self._tx_bus = can.interface.Bus(
            channel=self._can_interface, interface="socketcan", receive_own_messages=False
        )
        self._state_publisher = self.create_publisher(
            Float64MultiArray, "/trackstand/motor_state", 10
        )
        self._fault_publisher = self.create_publisher(Bool, "/trackstand/motor_fault", 10)
        self._status_publisher = self.create_publisher(String, "/trackstand/motor_status", 10)
        self._command_subscription = self.create_subscription(
            Float64MultiArray, "/trackstand/motor_command", self._command_callback, 10
        )

        self._threads = [
            threading.Thread(target=self._poll_motor, args=(motor,), daemon=True)
            for motor in (self._steering, self._drive)
        ]
        for thread in self._threads:
            thread.start()
        self._control_timer = self.create_timer(1.0 / self._control_rate_hz, self._control_tick)
        self._publish_timer = self.create_timer(1.0 / self._poll_rate_hz, self._publish_state)
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._disable_outputs(force=True)
        self.get_logger().info(
            f"Motor I/O on {self._can_interface}: poll={self._poll_rate_hz:.0f} Hz, "
            f"control={self._control_rate_hz:.0f} Hz, dry_run={self._dry_run}"
        )

    def _poll_motor(self, motor: ChannelState) -> None:
        bus = can.interface.Bus(
            channel=self._can_interface,
            interface="socketcan",
            receive_own_messages=False,
            can_filters=[{"can_id": motor.feedback_id, "can_mask": 0x7FF, "extended": False}],
        )
        request_b0 = can.Message(
            arbitration_id=motor.host_id, data=bytes([0xB0]), is_extended_id=False
        )
        request_a2 = can.Message(
            arbitration_id=motor.host_id, data=bytes([0xA2]), is_extended_id=False
        )
        request_a3 = can.Message(
            arbitration_id=motor.host_id, data=bytes([0xA3]), is_extended_id=False
        )
        period_s = 1.0 / self._poll_rate_hz
        deadline = time.monotonic()
        retry_b0_at = 0.0
        try:
            while self._running:
                now = time.monotonic()
                if motor.torque_constant_nm_per_a is None and now >= retry_b0_at:
                    payload = query(bus, request_b0, motor.feedback_id, max(0.002, period_s * 0.8))
                    if payload is not None:
                        try:
                            torque_constant = parse_b0_torque_constant(payload)
                            with motor.lock:
                                motor.torque_constant_nm_per_a = torque_constant
                            self.get_logger().info(
                                f"{motor.name} B0 torque constant={torque_constant:.6g} N m/A"
                            )
                        except ValueError as error:
                            self.get_logger().error(str(error), throttle_duration_sec=2.0)
                    retry_b0_at = now + 1.0

                position_payload = query(
                    bus, request_a3, motor.feedback_id, max(0.001, period_s * 0.35)
                )
                velocity_payload = query(
                    bus, request_a2, motor.feedback_id, max(0.001, period_s * 0.35)
                )
                if position_payload is not None and velocity_payload is not None:
                    try:
                        encoder_count = parse_a3_multi_turn_count(position_payload)
                        with motor.lock:
                            if motor.zero_count is None:
                                motor.zero_count = encoder_count
                                self.get_logger().info(
                                    f"{motor.name} encoder zero={encoder_count} counts"
                                )
                            motor.position_rad = parse_a3_position(
                                position_payload,
                                motor.zero_count,
                                motor.gear_ratio,
                                motor.encoder_sign,
                            )
                            motor.velocity_rad_s = parse_a2_velocity(
                                velocity_payload, motor.gear_ratio, motor.encoder_sign
                            )
                            motor.sample_monotonic = time.monotonic()
                    except (ValueError, IndexError) as error:
                        self.get_logger().warning(
                            f"Invalid {motor.name} A2/A3 response: {error}",
                            throttle_duration_sec=2.0,
                        )

                deadline += period_s
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    deadline = time.monotonic()
        finally:
            bus.shutdown()

    def _command_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) != 3:
            self.get_logger().warning("motor_command must be [enable, steering_target_rad, drive_torque_nm]")
            return
        values = np.asarray(message.data, dtype=float)
        if not np.all(np.isfinite(values)):
            self.get_logger().error("Rejected non-finite motor command")
            return
        with self._command_lock:
            self._requested_enable = bool(values[0] > 0.5)
            self._steering_target_rad = float(
                np.clip(values[1], -self._steering_limit_rad, self._steering_limit_rad)
            )
            self._drive_target_nm = float(
                np.clip(values[2], -DRIVE_EFFORT_LIMIT_NM, DRIVE_EFFORT_LIMIT_NM)
            )
            self._command_monotonic = time.monotonic()

    def _snapshot(self, motor: ChannelState) -> tuple[float, float, float, float | None]:
        with motor.lock:
            return (
                motor.position_rad,
                motor.velocity_rad_s,
                motor.sample_monotonic,
                motor.torque_constant_nm_per_a,
            )

    def _control_tick(self) -> None:
        now = time.monotonic()
        steering = self._snapshot(self._steering)
        drive = self._snapshot(self._drive)
        with self._command_lock:
            requested_enable = self._requested_enable
            command_age = now - self._command_monotonic
            steering_target = self._steering_target_rad
            drive_target_nm = self._drive_target_nm

        feedback_ready = (
            steering[3] is not None
            and drive[3] is not None
            and now - steering[2] <= self._feedback_timeout_s
            and now - drive[2] <= self._feedback_timeout_s
        )
        command_ready = self._command_monotonic > 0.0 and command_age <= self._command_timeout_s
        over_limit = (
            abs(steering[0]) > self._steering_limit_rad
            or abs(steering[1]) > STEERING_VELOCITY_LIMIT_RAD_S * 1.1
            or abs(drive[1]) > DRIVE_VELOCITY_LIMIT_RAD_S * 1.1
        )
        if over_limit:
            self._fault_latched = True
        if not requested_enable and abs(steering[0]) < self._steering_limit_rad * 0.5:
            self._fault_latched = False

        enabled = requested_enable and command_ready and feedback_ready and not self._fault_latched
        if not enabled:
            self._slewed_steering_target_rad = steering[0]
            self._disable_outputs()
            fault = self._fault_latched or (requested_enable and not feedback_ready)
            self._fault_publisher.publish(Bool(data=bool(fault)))
            return

        max_step = STEERING_VELOCITY_LIMIT_RAD_S / self._control_rate_hz
        target_error = steering_target - self._slewed_steering_target_rad
        self._slewed_steering_target_rad += float(np.clip(target_error, -max_step, max_step))
        steering_torque_nm = STEERING_KP_NM_PER_RAD * (
            self._slewed_steering_target_rad - steering[0]
        ) - STEERING_KD_NM_S_PER_RAD * steering[1]
        steering_torque_nm = float(
            np.clip(steering_torque_nm, -STEERING_EFFORT_LIMIT_NM, STEERING_EFFORT_LIMIT_NM)
        )
        drive_torque_nm = drive_target_nm - DRIVE_DAMPING_NM_S_PER_RAD * drive[1]
        drive_torque_nm = float(
            np.clip(drive_torque_nm, -DRIVE_EFFORT_LIMIT_NM, DRIVE_EFFORT_LIMIT_NM)
        )
        self._send_torque(self._steering, steering_torque_nm, float(steering[3]))
        self._send_torque(self._drive, drive_torque_nm, float(drive[3]))
        self._outputs_active = True
        self._fault_publisher.publish(Bool(data=False))

    def _drain_tx(self) -> None:
        for _ in range(16):
            try:
                if self._tx_bus.recv(timeout=0.0) is None:
                    break
            except can.CanError:
                break

    def _send_torque(self, motor: ChannelState, torque_nm: float, torque_constant: float) -> None:
        if self._dry_run:
            return
        current_a = motor.torque_sign * torque_nm / torque_constant
        try:
            self._drain_tx()
            self._tx_bus.send(
                can.Message(
                    arbitration_id=motor.host_id,
                    data=q_current_payload(current_a),
                    is_extended_id=False,
                )
            )
        except can.CanError as error:
            self._fault_latched = True
            self.get_logger().error(f"{motor.name} C0 send failed: {error}", throttle_duration_sec=1.0)

    def _disable_outputs(self, force: bool = False) -> None:
        if not force and not self._outputs_active:
            return
        if not self._dry_run:
            for motor in (self._steering, self._drive):
                try:
                    self._tx_bus.send(
                        can.Message(
                            arbitration_id=motor.host_id,
                            data=q_current_payload(0.0),
                            is_extended_id=False,
                        )
                    )
                    self._tx_bus.send(
                        can.Message(
                            arbitration_id=motor.host_id,
                            data=bytes([0xCF]),
                            is_extended_id=False,
                        )
                    )
                except can.CanError:
                    pass
        self._outputs_active = False

    def _publish_state(self) -> None:
        steering = self._snapshot(self._steering)
        drive = self._snapshot(self._drive)
        if steering[2] <= 0.0 or drive[2] <= 0.0:
            return
        now = time.monotonic()
        if (
            now - steering[2] > self._feedback_timeout_s
            or now - drive[2] > self._feedback_timeout_s
        ):
            return
        sample_times = (steering[2], drive[2])
        if (
            sample_times[0] <= self._last_published_sample_times[0]
            or sample_times[1] <= self._last_published_sample_times[1]
        ):
            return
        message = Float64MultiArray()
        message.data = [steering[0], drive[0], steering[1], drive[1], time.time()]
        self._state_publisher.publish(message)
        self._last_published_sample_times = sample_times

    def _publish_status(self) -> None:
        steering = self._snapshot(self._steering)
        drive = self._snapshot(self._drive)
        message = String()
        message.data = json.dumps(
            {
                "outputs_active": self._outputs_active,
                "fault_latched": self._fault_latched,
                "dry_run": self._dry_run,
                "steering_torque_constant_nm_per_a": steering[3],
                "drive_torque_constant_nm_per_a": drive[3],
            },
            sort_keys=True,
        )
        self._status_publisher.publish(message)

    def destroy_node(self) -> None:
        self._running = False
        self._disable_outputs(force=True)
        for thread in getattr(self, "_threads", []):
            thread.join(timeout=0.2)
        if hasattr(self, "_tx_bus"):
            self._tx_bus.shutdown()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MotorIo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
