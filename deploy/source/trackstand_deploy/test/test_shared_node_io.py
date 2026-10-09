"""Regression checks for shared ROS interfaces, servo ordering and IMU fields."""

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import sys
import struct

import numpy as np
import pytest

from trackstand_deploy.telemetry import MOTOR_STATE_FIELDS


@pytest.mark.parametrize("filename,classname", [
    ("motor_io", "MotorIo"), ("motor_io_position", "MotorIoPosition"), ("motor_mit_io", "MotorMitIo"),
])
def test_common_motor_interfaces_keep_each_versioned_schema(node_module, filename, classname):
    module = node_module(filename)
    node = object.__new__(getattr(module, classname))
    node.create_publisher = Mock()
    node.create_subscription = Mock()
    node._create_interfaces()
    assert [call.args[1] for call in node.create_publisher.call_args_list] == [
        "/trackstand/motor_state", "/trackstand/motor_control",
        "/trackstand/motor_event", "/trackstand/motor_fault", "/trackstand/motor_status",
    ]
    assert node.create_subscription.call_args.args[1:] == (
        "/trackstand/motor_command", node._command_callback, 10,
    )
    assert node._control_dimension.label == node._control_label + ":" + ",".join(module.MOTOR_CONTROL_FIELDS)
    assert node._control_dimension.size == node._control_dimension.stride == len(module.MOTOR_CONTROL_FIELDS)
    assert node._state_dimension.label == "motor_state_v2:" + ",".join(MOTOR_STATE_FIELDS)
    assert node._state_dimension.size == node._state_dimension.stride == 7
    assert node.create_publisher.call_args_list[0].args[2] == 10  # Keep state/observation QoS unchanged.
    assert node.create_publisher.call_args_list[1].args[2].depth == 50


@pytest.mark.parametrize("up", [False, True])
def test_kickstand_pose_enables_both_servos_before_any_move(node_module, monkeypatch, up):
    servo_module = ModuleType("trackstand_deploy.servo_control")
    servo_module.SMS_STS = Mock()
    monkeypatch.setitem(sys.modules, servo_module.__name__, servo_module)
    module = node_module("kickstand_driver")
    node = object.__new__(module.KickstandDriver)
    node._servo_ids = [1, 2]
    node._up_positions, node._down_positions = [1648, 2454], [1972, 2085]
    node._servos = Mock()
    parent = Mock()
    parent.attach_mock(node._servos.enable_torque, "enable")
    parent.attach_mock(node._servos.write_pos_ex, "move")
    (node._command_up if up else node._command_down)()
    positions = node._up_positions if up else node._down_positions
    assert [(call[0], call[1]) for call in parent.mock_calls] == [
        ("enable", (1, 1)), ("enable", (2, 1)),
        ("move", (1, positions[0], 3000, 255)), ("move", (2, positions[1], 3000, 255)),
    ]


def test_imu_component_assignment_preserves_units_order_and_freshness(node_module, monkeypatch):
    serial_module = ModuleType("trackstand_deploy.imu_serial")
    serial_module.YbImuSerial = Mock()
    monkeypatch.setitem(sys.modules, serial_module.__name__, serial_module)
    module = node_module("imu_publisher")
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(module, "Imu", lambda: SimpleNamespace(
        header=SimpleNamespace(), orientation=SimpleNamespace(), angular_velocity=SimpleNamespace(),
        linear_acceleration_covariance=[0.0] * 9,
    ))
    node = object.__new__(module.ImuPublisher)
    node._imu = SimpleNamespace(
        last_valid_frame_monotonic=99.99,
        get_imu_quaternion_data=lambda: [1.0, 0.0, 0.0, 0.0],
        get_gyroscope_data=lambda: [0.2, -0.3, 0.4],
    )
    node._last_source_time, node._source_timeout_s = 0.0, 0.25
    node._frame_id = "imu_link"
    node._sensor_to_bike = np.array([0.0, 0.0, 0.0, 1.0])
    node._gyro_bias_rad_s = np.array([0.1, 0.1, 0.1])
    node._publisher = Mock()
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))
    node._publish()
    message = node._publisher.publish.call_args.args[0]
    assert [getattr(message.orientation, axis) for axis in "xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert [getattr(message.angular_velocity, axis) for axis in "xyz"] == pytest.approx([0.1, -0.4, 0.3])
    assert message.header.stamp == "stamp" and message.header.frame_id == "imu_link"
    assert message.linear_acceleration_covariance[0] == -1.0
    node._publish()  # Repeated data must not be published as a fresh sample.
    assert node._publisher.publish.call_count == 1


@pytest.mark.parametrize("filename,classname", [
    ("motor_io", "MotorIo"), ("motor_io_position", "MotorIoPosition"),
])
def test_shared_encoder_poller_preserves_startup_zero_and_c2_multi_turn_count(
    node_module, monkeypatch, filename, classname,
):
    module = node_module(filename)
    node = object.__new__(getattr(module, classname))
    base = sys.modules[module.MotorIo.__module__]
    clock, poll_bus = SimpleNamespace(value=100.0), Mock()
    motor = module.ChannelState("steering", 0x102, 2, 1.0, -1.0, -1.0)
    node._can_interface, node._poll_rate_hz, node._running = "can0", 200.0, True
    node.get_logger = Mock()
    monkeypatch.setattr(base.can, "interface", SimpleNamespace(Bus=lambda **kwargs: poll_bus), raising=False)
    monkeypatch.setattr(base, "time", SimpleNamespace(
        monotonic=lambda: clock.value, sleep=lambda delay: setattr(clock, "value", clock.value + delay),
    ))
    counts = iter([1234, 1490])
    velocities = []

    def query(bus, request, feedback_id, timeout):
        assert bus is poll_bus and feedback_id == 2 and timeout > 0
        clock.value += 0.0005
        if request.data[0] == 0xB0:
            return b"\xb0\x00" + struct.pack("<f", 0.78) + b"\x00"
        if request.data[0] == 0xA3:
            return b"\xa3" + struct.pack("<Hi", 0, next(counts))
        velocities.append(True)
        node._running = len(velocities) < 2
        return b"\xa2" + struct.pack("<i", 6000)

    monkeypatch.setattr(base, "query", query)
    node._poll_motor(motor)
    assert motor.zero_count == 1234
    assert motor.position_rad == pytest.approx(-256 * 2 * np.pi / 16384)
    assert motor.velocity_rad_s == pytest.approx(-2 * np.pi)
    assert motor.position_sample_monotonic < motor.velocity_sample_monotonic == motor.sample_monotonic
    assert motor.position_query_failures == motor.velocity_query_failures == 0
    assert len(node._snapshot(motor)) == (12 if filename == "motor_io_position" else 10)
    if filename == "motor_io_position":
        assert node._snapshot(motor)[-2:] == (1490, 1234)
    poll_bus.shutdown.assert_called_once()
