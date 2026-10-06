"""Exercise motor-I/O torque telemetry without ROS or a physical CAN bus."""

import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def motor_node(node_module):
    module = node_module("motor_io")
    node = object.__new__(module.MotorIo)
    node._steering = module.ChannelState("steering", 0x102, 0x002, 1.0, -1.0, -1.0)
    node._drive = module.ChannelState("drive", 0x101, 0x001, 1.0, 1.0, -1.0)
    node._steering.torque_constant_nm_per_a = 1.25
    node._drive.torque_constant_nm_per_a = 2.0
    node._torque_publisher = Mock()
    node._stamp = object()
    node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: node._stamp)))
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
    "motor_id,joint_name,constant",
    [(0x002, "steering_joint", 1.25), (0x001, "rear_wheel_joint", 2.0)],
)
@pytest.mark.parametrize("command", [0xA1, 0xC0])
def test_feedback_publishes_named_timestamped_motor_torque(motor_node, motor_id, joint_name, constant, command):
    motor_node._publish_torque_feedback(feedback_frame(motor_id, command=command))
    sample = motor_node._torque_publisher.publish.call_args.args[0]
    assert sample.header.stamp is motor_node._stamp
    assert sample.name == [joint_name]
    assert sample.effort == pytest.approx([1.234 * constant])
    assert sample.position == []
    assert sample.velocity == []
    # Independent samples: no need for the other motor to have replied.
    motor_node._torque_publisher.publish.assert_called_once()


def test_unavailable_torque_constant_does_not_publish_fabricated_torque(motor_node):
    motor_node._drive.torque_constant_nm_per_a = None
    motor_node._publish_torque_feedback(feedback_frame(0x001))
    motor_node._torque_publisher.publish.assert_not_called()
    assert motor_node._drive.torque_sample_monotonic == 0.0


@pytest.mark.parametrize("motor_id,command", [(0x101, 0xC0), (0x102, 0xC0), (0x003, 0xC0), (0x001, 0xA2)])
def test_commands_unknown_ids_and_other_replies_are_not_torque(motor_node, motor_id, command):
    motor_node._publish_torque_feedback(feedback_frame(motor_id, command=command))
    motor_node._torque_publisher.publish.assert_not_called()


@pytest.mark.parametrize("flag", ["is_extended_id", "is_remote_frame", "is_error_frame"])
def test_nonstandard_data_frames_are_ignored(motor_node, flag):
    motor_node._publish_torque_feedback(feedback_frame(0x001, **{flag: True}))
    motor_node._torque_publisher.publish.assert_not_called()


def test_bad_reply_is_ignored_without_modifying_control_state(motor_node):
    frame = feedback_frame(0x001)
    frame.data = b"\xc0\x01"
    motor_node._publish_torque_feedback(frame)
    motor_node._torque_publisher.publish.assert_not_called()
    assert motor_node._drive.torque_sample_monotonic == 0.0
    motor_node.get_logger().warning.assert_called_once()


def test_feedback_is_not_clipped_or_replaced_by_command(motor_node):
    motor_node._drive_target_nm = 0.0
    motor_node._publish_torque_feedback(feedback_frame(0x001, milliamps=-10000))
    assert motor_node._torque_publisher.publish.call_args.args[0].effort == [20.0]


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
    motor_node._torque_publisher.publish.assert_called_once()
