"""Shutdown and servo-error regressions; mock transports only."""

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import sys

import pytest


@pytest.mark.parametrize("filename,classname", [
    ("imu_publisher", "ImuPublisher"), ("rc_kickstand", "RcKickstand"),
    ("kickstand_driver", "KickstandDriver"), ("motor_io", "MotorIo"),
    ("motor_io_position", "MotorIoPosition"), ("motor_mit_io", "MotorMitIo"),
    ("policy_controller", "PolicyController"),
])
@pytest.mark.parametrize("failure", ["external_shutdown", "constructor", "cleanup"])
def test_all_node_entrypoints_shutdown_once_even_after_signal_or_cleanup_error(
    node_module, monkeypatch, filename, classname, failure,
):
    adapter = ModuleType("trackstand_deploy.imu_serial")
    adapter.YbImuSerial = Mock()
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    executors = ModuleType("rclpy.executors")
    executors.ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})
    monkeypatch.setitem(sys.modules, executors.__name__, executors)
    module = node_module(filename)
    node = Mock()
    constructor = Mock(return_value=node)
    monkeypatch.setattr(module, classname, constructor)
    module.rclpy = SimpleNamespace(init=Mock(), try_shutdown=Mock(), spin=Mock())
    if failure == "constructor":
        constructor.side_effect = RuntimeError("injected constructor error")
    elif failure == "cleanup":
        node.destroy_node.side_effect = RuntimeError("injected cleanup error")
    else:
        module.rclpy.spin.side_effect = executors.ExternalShutdownException()
    if failure == "external_shutdown":
        module.main()
    else:
        with pytest.raises(RuntimeError):
            module.main()
    module.rclpy.try_shutdown.assert_called_once()
    assert node.destroy_node.call_count == (0 if failure == "constructor" else 1)


@pytest.mark.parametrize("error", [0, 1, 4, 32])
def test_servo_write_acknowledgement_with_error_is_not_success(error):
    from trackstand_deploy.servo_control.sms_sts import SMS_STS
    servo = SMS_STS()
    header = bytes([1, 2, error])
    servo._check_head = Mock(return_value=True)
    servo._read_exactly = Mock(return_value=header + bytes([servo._checksum(header)]))
    assert servo._ack(1) is (error == 0)


def test_servo_read_error_cannot_be_published_as_valid_position():
    from trackstand_deploy.servo_control.sms_sts import SMS_STS
    servo = SMS_STS()
    header, payload = bytes([1, 4, 4]), b"\x00\x08"
    servo._send_packet = Mock()
    servo._check_head = Mock(return_value=True)
    servo._read_exactly = Mock(side_effect=[header, payload, bytes([servo._checksum(header + payload)])])
    assert servo.read(1, 56, 2) is None


def test_failed_first_kickstand_write_still_attempts_second_stand(node_module):
    module = node_module("kickstand_driver")
    node = object.__new__(module.KickstandDriver)
    node._servo_ids = [1, 2]
    node._down_positions = [1972, 2085]
    node._servos = Mock()
    node._servos.write_pos_ex.side_effect = [OSError("disconnected"), True]
    node.get_logger = Mock(return_value=Mock())
    node._command_down()
    assert [call.args[0] for call in node._servos.write_pos_ex.call_args_list] == [1, 2]
    node.get_logger().error.assert_called_once()
