"""Exercise motor-I/O measured-torque caching without ROS or a physical CAN bus."""

import math
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def motor_node(node_module):
    module = node_module("motor_io")
    module.time = SimpleNamespace(monotonic=lambda: 100.0)
    node = object.__new__(module.MotorIo)
    node._steering = module.ChannelState("steering", 0x102, 0x002, 1.0, -1.0, -1.0)
    node._drive = module.ChannelState("drive", 0x101, 0x001, 1.0, 1.0, -1.0)
    node._steering.torque_constant_nm_per_a = 1.25
    node._drive.torque_constant_nm_per_a = 2.0
    node.get_logger = Mock(return_value=Mock())
    return node


def feedback_frame(motor_id, milliamps=-1234, command=0xC0, **flags):
    return SimpleNamespace(
        arbitration_id=motor_id,
        data=bytes([command]) + struct.pack("<i", milliamps),
        is_extended_id=flags.get("is_extended_id", False),
        is_remote_frame=flags.get("is_remote_frame", False),
        is_error_frame=flags.get("is_error_frame", False),
    )


@pytest.mark.parametrize(
    "motor_id,motor_name,constant",
    [(0x002, "steering", 1.25), (0x001, "drive", 2.0)],
)
@pytest.mark.parametrize("command", [0xA1, 0xC0])
def test_feedback_caches_each_motor_torque_and_receipt_time(motor_node, motor_id, motor_name, constant, command):
    motor_node._update_torque_feedback(feedback_frame(motor_id, command=command))
    motor = getattr(motor_node, f"_{motor_name}")
    assert motor.torque_nm == pytest.approx(1.234 * constant)
    assert motor.torque_sample_monotonic == 100.0
    other = motor_node._drive if motor_name == "steering" else motor_node._steering
    assert math.isnan(other.torque_nm) and other.torque_sample_monotonic == 0.0


def test_unavailable_torque_constant_does_not_publish_fabricated_torque(motor_node):
    motor_node._drive.torque_constant_nm_per_a = None
    motor_node._update_torque_feedback(feedback_frame(0x001))
    assert math.isnan(motor_node._drive.torque_nm)
    assert motor_node._drive.torque_sample_monotonic == 0.0


@pytest.mark.parametrize("motor_id,command", [(0x101, 0xC0), (0x102, 0xC0), (0x003, 0xC0), (0x001, 0xA2)])
def test_commands_unknown_ids_and_other_replies_are_not_torque(motor_node, motor_id, command):
    motor_node._update_torque_feedback(feedback_frame(motor_id, command=command))
    assert math.isnan(motor_node._steering.torque_nm) and math.isnan(motor_node._drive.torque_nm)


@pytest.mark.parametrize("flag", ["is_extended_id", "is_remote_frame", "is_error_frame"])
def test_nonstandard_data_frames_are_ignored(motor_node, flag):
    motor_node._update_torque_feedback(feedback_frame(0x001, **{flag: True}))
    assert math.isnan(motor_node._drive.torque_nm)


def test_bad_reply_is_ignored_without_modifying_control_state(motor_node):
    frame = feedback_frame(0x001)
    frame.data = b"\xc0\x01"
    motor_node._update_torque_feedback(frame)
    assert math.isnan(motor_node._drive.torque_nm)
    assert motor_node._drive.torque_sample_monotonic == 0.0
    motor_node.get_logger().warning.assert_called_once()


def test_feedback_is_not_clipped_or_replaced_by_command(motor_node):
    motor_node._drive_target_nm = 0.0
    motor_node._update_torque_feedback(feedback_frame(0x001, milliamps=-10000))
    assert motor_node._drive.torque_nm == 20.0


def test_listener_is_receive_only(motor_node):
    motor_node._running = True
    motor_node._torque_bus = Mock()

    def receive(timeout):
        assert timeout == 0.05
        motor_node._running = False
        return feedback_frame(0x001)

    motor_node._torque_bus.recv.side_effect = receive
    motor_node._receive_torque_feedback()
    motor_node._torque_bus.send.assert_not_called()
    assert motor_node._drive.torque_nm == pytest.approx(2.468)
    assert motor_node._drive.torque_sample_monotonic == 100.0
