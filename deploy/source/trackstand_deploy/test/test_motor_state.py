"""Unified position/velocity/measured-torque state contract; no ROS/CAN hardware."""

import math
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from trackstand_deploy.telemetry import MOTOR_STATE_FIELDS, MOTOR_STATE_LABEL


@pytest.fixture(params=[
    ("motor_io", "MotorIo"), ("motor_io_position", "MotorIoPosition"), ("motor_mit_io", "MotorMitIo"),
])
def state_node(node_module, monkeypatch, request):
    filename, classname = request.param
    module = node_module(filename)
    clock = SimpleNamespace(value=100.0)
    task_time = SimpleNamespace(monotonic=lambda: clock.value, time=lambda: 123456.0)
    monkeypatch.setattr(module, "time", task_time)
    if filename == "motor_io_position":
        monkeypatch.setattr(sys.modules[module.MotorIo.__module__], "time", task_time)
    node = object.__new__(getattr(module, classname))
    node._initialize_command_state()
    node._steering = module.ChannelState("steering", 0x102, 2, 1.0, -1.0, -1.0)
    node._drive = module.ChannelState("drive", 0x101, 1, 1.0, 1.0, -1.0)
    node._steering.position_rad, node._drive.position_rad = 0.1, -0.2
    node._steering.velocity_rad_s, node._drive.velocity_rad_s = 0.3, -0.4
    for motor, constant in ((node._steering, 0.78), (node._drive, 0.3)):
        motor.sample_monotonic = 99.99
        motor.torque_constant_nm_per_a = constant
    node._feedback_timeout_s = 0.05
    node._pending = None
    node.create_publisher = Mock(side_effect=lambda *args: Mock())
    node.create_subscription = Mock()
    node.get_logger = Mock()
    node._create_interfaces()
    return node, module, clock


def receive_current(node, module, motor, opcode, milliamps):
    payload = bytes([opcode]) + struct.pack("<i", milliamps)
    if hasattr(node, "_consume"):
        frame = module.CanReply(motor.feedback_id, payload, False, 0, module.time.monotonic())
        node._consume(frame, module.time.monotonic())
    else:
        frame = SimpleNamespace(
            arbitration_id=motor.feedback_id, data=payload, is_extended_id=False,
            is_remote_frame=False, is_error_frame=False,
        )
        node._update_torque_feedback(frame)


def test_unified_state_includes_measured_unclipped_torque_and_preserves_old_prefix(state_node):
    node, module, _ = state_node
    receive_current(node, module, node._steering, 0xA1, 20000)
    receive_current(node, module, node._drive, 0xC0, -1000)
    node._publish_state()
    message = node._state_publisher.publish.call_args.args[0]
    assert message.data == pytest.approx([0.1, -0.2, 0.3, -0.4, 123456.0, -15.6, 0.3])
    assert message.layout.dim[0].label == MOTOR_STATE_LABEL
    assert message.layout.dim[0].size == message.layout.dim[0].stride == len(MOTOR_STATE_FIELDS) == 7
    assert not hasattr(node, "_torque_publisher")


@pytest.mark.parametrize("steering_at,drive_at,expected", [
    (0.0, 0.0, (math.nan, math.nan)),
    (99.99, 0.0, (1.2, math.nan)),
    (0.0, 99.99, (math.nan, -0.7)),
    (99.9, 99.99, (math.nan, -0.7)),
    (99.99, 99.9, (1.2, math.nan)),
    (99.99, 99.98, (1.2, -0.7)),
    (100.1, 99.99, (math.nan, -0.7)),
])
def test_missing_and_stale_torque_are_nan_without_blocking_encoder_state(
    state_node, steering_at, drive_at, expected,
):
    node, _, _ = state_node
    node._steering.torque_nm, node._drive.torque_nm = 1.2, -0.7
    node._steering.torque_sample_monotonic, node._drive.torque_sample_monotonic = steering_at, drive_at
    node._publish_state()
    message = node._state_publisher.publish.call_args.args[0]
    assert message.data[:5] == [0.1, -0.2, 0.3, -0.4, 123456.0]
    for actual, wanted in zip(message.data[5:], expected):
        assert math.isnan(actual) if math.isnan(wanted) else actual == pytest.approx(wanted)


def test_fresh_zero_torque_is_not_missing_feedback(state_node):
    node, module, _ = state_node
    for motor in (node._steering, node._drive):
        receive_current(node, module, motor, 0xA1, 0)
    node._publish_state()
    assert node._state_publisher.publish.call_args.args[0].data[5:] == [0.0, 0.0]


def test_missing_b0_does_not_substitute_commanded_torque(state_node):
    node, module, _ = state_node
    node._steering.torque_constant_nm_per_a = None
    node._drive_target_nm = 6.28
    receive_current(node, module, node._steering, 0xA1, 1000)
    node._publish_state()
    assert all(math.isnan(value) for value in node._state_publisher.publish.call_args.args[0].data[5:])


def test_torque_cache_ages_out_even_when_encoders_continue(state_node):
    node, module, clock = state_node
    for motor in (node._steering, node._drive):
        receive_current(node, module, motor, 0xA1, -100)
    node._publish_state()
    assert all(math.isfinite(value) for value in node._state_publisher.publish.call_args.args[0].data[5:])
    clock.value += 0.06
    for motor in (node._steering, node._drive):
        motor.sample_monotonic = clock.value
    node._publish_state()
    assert node._state_publisher.publish.call_count == 2
    assert all(math.isnan(value) for value in node._state_publisher.publish.call_args.args[0].data[5:])


@pytest.mark.parametrize("steering_at,drive_at,previous", [
    (0.0, 99.99, (0.0, 0.0)), (99.99, 0.0, (0.0, 0.0)),
    (99.9, 99.99, (0.0, 0.0)), (99.99, 99.9, (0.0, 0.0)),
    (99.99, 99.99, (99.99, 99.99)), (100.0, 99.99, (99.99, 99.99)),
    (100.1, 99.99, (0., 0.)), (99.99, 100.1, (0., 0.)),
])
def test_torque_replies_do_not_make_stale_or_repeated_encoder_pairs_fresh(
    state_node, steering_at, drive_at, previous,
):
    node, module, _ = state_node
    node._steering.sample_monotonic, node._drive.sample_monotonic = steering_at, drive_at
    node._last_published_sample_times = previous
    for motor in (node._steering, node._drive):
        receive_current(node, module, motor, 0xA1, -100)
    node._publish_state()
    node._state_publisher.publish.assert_not_called()


@pytest.mark.parametrize("torques", [(math.nan, math.nan), (-0.1, 0.2)])
def test_policy_still_uses_only_kinematics_from_extended_state(node_module, torques):
    module = node_module("policy_controller")
    module.time = SimpleNamespace(monotonic=lambda: 100.0, time=lambda: 123456.0)
    node = object.__new__(module.PolicyController)
    node._motor_state = None
    node._motor_monotonic = 0.0
    node._motor_timeout_s = .06
    message = module.Float64MultiArray(data=[0.1, -0.2, 0.3, -0.4, 123456.0, *torques])
    message.layout.dim = [SimpleNamespace(label=MOTOR_STATE_LABEL)]
    node._motor_callback(message)
    np.testing.assert_array_equal(node._motor_state, [0.1, -0.2, 0.3, -0.4])
    assert node._motor_monotonic == 100.0
