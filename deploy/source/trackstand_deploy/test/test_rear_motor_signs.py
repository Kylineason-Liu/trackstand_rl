"""Regression checks using physically labelled rear-wheel measurements (2026-10-12).

ROS/CAN are stand-ins; no node initialization, socket or real motor writes.
"""

import math
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from trackstand_deploy.can_protocol import (
    current_feedback_to_torque_nm,
    parse_a2_velocity,
    parse_a3_multi_turn_count,
    parse_a3_position,
    parse_current_feedback,
)
from trackstand_deploy.contracts import DRIVE_DAMPING_NM_S_PER_RAD, build_observation


@pytest.fixture(params=("motor_io", "motor_io_position", "motor_mit_io"))
def backend(node_module, request):
    module = node_module(request.param)
    cls = getattr(module, {
        "motor_io": "MotorIo", "motor_io_position": "MotorIoPosition", "motor_mit_io": "MotorMitIo",
    }[request.param])
    return module, cls


def test_all_backends_and_yaml_use_verified_rear_signs(backend):
    _, cls = backend
    config = Path(__file__).parents[1] / "config" / "params.yaml"
    parameters = yaml.safe_load(config.read_text())["trackstand_motor_io"]["ros__parameters"]
    for key in ("drive_encoder_sign", "drive_torque_sign", "steering_encoder_sign", "steering_torque_sign"):
        assert cls._parameter_defaults[key] == parameters[key] == -1.0
    assert cls._parameter_defaults["drive_host_id"] == parameters["drive_host_id"] == "101"
    assert cls._parameter_defaults["drive_feedback_id"] == parameters["drive_feedback_id"] == "001"


@pytest.mark.parametrize("speed_hex,native_rpm", [
    ("02D7FFFF", -104.94), ("64FAFFFF", -14.36), ("88F6FFFF", -24.24),
    ("87F0FFFF", -39.61), ("F4E2FFFF", -74.36),
])
def test_physically_forward_a2_samples_publish_positive_speed(backend, speed_hex, native_rpm):
    _, cls = backend
    defaults = cls._parameter_defaults
    payload = bytes.fromhex("A2" + speed_hex)
    velocity = parse_a2_velocity(payload, defaults["drive_gear_ratio"], defaults["drive_encoder_sign"])
    assert velocity == pytest.approx(-native_rpm * 2 * math.pi / 60)
    assert velocity > 0
    assert DRIVE_DAMPING_NM_S_PER_RAD == 0.0  # v1 has no rear controller damping.


def test_physically_forward_a3_decrease_becomes_positive_position(backend):
    _, cls = backend
    defaults = cls._parameter_defaults
    before = bytes.fromhex("A3232C236C5200")
    after = bytes.fromhex("A3630C634C5200")
    zero = parse_a3_multi_turn_count(before)
    assert zero == 5401635
    assert parse_a3_multi_turn_count(after) == 5393507
    position = parse_a3_position(after, zero, defaults["drive_gear_ratio"], defaults["drive_encoder_sign"])
    assert position == pytest.approx(8128 * 2 * math.pi / 16384)
    assert position > 0


def test_corrected_feedback_reaches_policy_with_positive_forward_velocity(backend):
    _, cls = backend
    defaults = cls._parameter_defaults
    velocity = parse_a2_velocity(bytes.fromhex("A264FAFFFF"), 1, defaults["drive_encoder_sign"])
    position = parse_a3_position(bytes.fromhex("A3630C634C5200"), 5401635, 1, defaults["drive_encoder_sign"])
    observation = build_observation([0, 0, 0, 1], [0, 0, 0], 0, position, 0, velocity)
    assert observation[6] == pytest.approx(position)
    assert observation[8] == pytest.approx(velocity)
    assert observation[6] > 0 and observation[8] > 0


def test_rear_c0_command_direction_and_a1_feedback_are_unchanged(backend):
    module, cls = backend
    defaults = cls._parameter_defaults
    channel = module.ChannelState("drive", 0x101, 1, 1, defaults["drive_encoder_sign"], defaults["drive_torque_sign"])
    node = object.__new__(cls)
    node._dry_run = False
    node._tx_bus = Mock()
    node._tx_bus.recv.return_value = None
    node.get_logger = Mock(return_value=Mock())

    # Positive native current was physically backward: signed policy torque is negative.
    result = node._send_torque(channel, -.3 * 8.192, .3)
    assert result.attempted and result.success
    assert result.encoded_current_a == pytest.approx(8.192)
    assert result.encoded_torque_nm < 0
    args = node._tx_bus.send.call_args.args
    payload = args[1] if len(args) == 2 else args[0].data
    assert payload == bytes.fromhex("C000200000")
    native_measured_a = parse_current_feedback(bytes.fromhex("A196020000"))
    assert native_measured_a == pytest.approx(.662)
    assert current_feedback_to_torque_nm(native_measured_a, .3, channel.torque_sign) == pytest.approx(-.1986)

    # A positive policy torque still encodes negative native current (forward).
    result = node._send_torque(channel, .05, .3)
    assert result.success and result.encoded_current_a < 0 and result.encoded_torque_nm > 0
