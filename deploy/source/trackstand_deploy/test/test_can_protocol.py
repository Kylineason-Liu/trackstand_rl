import math
import struct

import pytest

from trackstand_deploy.can_protocol import (
    current_feedback_to_torque_nm,
    parse_a2_velocity,
    parse_a3_multi_turn_count,
    parse_a3_position,
    parse_b0_torque_constant,
    parse_current_feedback,
    q_current_payload,
)


def test_a2_preserves_full_drive_speed_and_a3_preserves_multiturn_position():
    speed_payload = bytes([0xA2]) + struct.pack("<i", 58800)
    assert parse_a2_velocity(speed_payload, 1.0, -1.0) == pytest.approx(-61.575216, rel=1e-6)
    position_payload = bytes([0xA3]) + struct.pack("<H", 0) + struct.pack("<i", 20000)
    assert parse_a3_multi_turn_count(position_payload) == 20000
    expected = -((20000 - 100) * 2.0 * math.pi / 16384)
    assert parse_a3_position(position_payload, 100, 1.0, -1.0) == pytest.approx(expected)


def test_parse_b0_and_encode_c0():
    payload = bytes([0xB0, 0]) + struct.pack("<f", 1.25)
    assert parse_b0_torque_constant(payload) == pytest.approx(1.25)
    assert q_current_payload(-1.234) == bytes([0xC0]) + struct.pack("<i", -1234)


def test_invalid_torque_constant_is_rejected():
    with pytest.raises(ValueError):
        parse_b0_torque_constant(bytes([0xB0, 0]) + struct.pack("<f", 0.0))


@pytest.mark.parametrize("command", [0xA1, 0xC0])
@pytest.mark.parametrize("milliamps", [-1234, 0, 1234])
def test_measured_current_reply_units_and_sign(command, milliamps):
    payload = bytes([command]) + struct.pack("<i", milliamps)
    current_a = parse_current_feedback(payload)
    assert current_a == pytest.approx(milliamps * 0.001)
    assert current_feedback_to_torque_nm(current_a, 1.25, -1.0) == pytest.approx(-milliamps * 0.001 * 1.25)


@pytest.mark.parametrize("payload", [b"", b"\xc0", bytes([0xA2]) + bytes(4), bytes([0xC0]) + bytes(5)])
def test_invalid_current_replies_are_rejected(payload):
    with pytest.raises(ValueError):
        parse_current_feedback(payload)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_feedback_torque_inverts_command_conversion_without_clipping(sign):
    torque_nm, torque_constant = 10.0, 1.25
    command = q_current_payload(sign * torque_nm / torque_constant)
    measured_current = parse_current_feedback(command)
    assert current_feedback_to_torque_nm(measured_current, torque_constant, sign) == pytest.approx(torque_nm)


@pytest.mark.parametrize(
    "current,constant,sign",
    [(math.nan, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, math.inf, 1.0), (1.0, 1.0, 0.0)],
)
def test_invalid_feedback_torque_conversion_is_rejected(current, constant, sign):
    with pytest.raises(ValueError):
        current_feedback_to_torque_nm(current, constant, sign)
