import math
import struct

import pytest

from trackstand_deploy.can_protocol import (
    parse_a2_velocity,
    parse_a3_multi_turn_count,
    parse_a3_position,
    parse_b0_torque_constant,
    q_current_payload,
)


def test_a2_preserves_full_drive_speed_and_a3_wraps_position():
    speed_payload = bytes([0xA2]) + struct.pack("<i", 58800)
    assert parse_a2_velocity(speed_payload, 1.0, -1.0) == pytest.approx(-61.575216, rel=1e-6)
    position_payload = bytes([0xA3]) + struct.pack("<H", 0) + struct.pack("<i", 20000)
    assert parse_a3_multi_turn_count(position_payload) == 20000
    expected = -((20000 - 100) * 2.0 * math.pi / 16384)
    expected = (expected + math.pi) % (2.0 * math.pi) - math.pi
    assert parse_a3_position(position_payload, 100, 1.0, -1.0) == pytest.approx(expected)


def test_parse_b0_and_encode_c0():
    payload = bytes([0xB0, 0]) + struct.pack("<f", 1.25)
    assert parse_b0_torque_constant(payload) == pytest.approx(1.25)
    assert q_current_payload(-1.234) == bytes([0xC0]) + struct.pack("<i", -1234)


def test_invalid_torque_constant_is_rejected():
    with pytest.raises(ValueError):
        parse_b0_torque_constant(bytes([0xB0, 0]) + struct.pack("<f", 0.0))
