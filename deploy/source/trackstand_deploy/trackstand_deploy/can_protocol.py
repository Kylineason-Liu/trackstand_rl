"""GIM V3.09b0 CAN helpers used by the deployment motor node."""

from __future__ import annotations

import math
import struct
import time


COUNTS_PER_REVOLUTION = 16384
RADIANS_PER_COUNT = 2.0 * math.pi / COUNTS_PER_REVOLUTION


def parse_current_feedback(payload: bytes) -> float:
    """Decode measured Q-axis current from an A1 or C0 reply, in amperes."""
    if len(payload) != 5 or payload[0] not in (0xA1, 0xC0):
        raise ValueError("invalid A1/C0 current response")
    return struct.unpack("<i", payload[1:5])[0] * 0.001


def current_feedback_to_torque_nm(current_a: float, torque_constant: float, torque_sign: float) -> float:
    """Estimate motor torque from measured current in the policy's sign convention."""
    if not math.isfinite(current_a) or not math.isfinite(torque_constant) or torque_constant <= 0.0:
        raise ValueError("current must be finite and torque constant must be positive and finite")
    if torque_sign not in (-1.0, 1.0):
        raise ValueError("torque_sign must be +/-1")
    # Inverse of the command conversion; do not clip measured feedback.
    return torque_sign * torque_constant * current_a


def parse_a2_velocity(payload: bytes, gear_ratio: float, sign: float) -> float:
    """Parse A2's int32 speed field and return joint rad/s."""
    if len(payload) < 5 or payload[0] != 0xA2:
        raise ValueError("invalid A2 response")
    if gear_ratio <= 0.0 or sign not in (-1.0, 1.0):
        raise ValueError("gear_ratio must be positive and sign must be +/-1")
    motor_rpm = struct.unpack("<i", payload[1:5])[0] * 0.01
    return sign * motor_rpm * (2.0 * math.pi / 60.0) / gear_ratio


def parse_a3_position(payload: bytes, zero_count: int, gear_ratio: float, sign: float) -> float:
    """Return signed continuous joint radians relative to the fixed A3 zero.

    Keep motor-state telemetry continuous. The policy applies its own per-episode
    reference and the training backend's angle branch, not a single-turn wrap.
    """
    if len(payload) < 7 or payload[0] != 0xA3:
        raise ValueError("invalid A3 response")
    if gear_ratio <= 0.0 or sign not in (-1.0, 1.0):
        raise ValueError("gear_ratio must be positive and sign must be +/-1")
    multi_turn_count = struct.unpack("<i", payload[3:7])[0]
    position = sign * (multi_turn_count - int(zero_count)) * RADIANS_PER_COUNT / gear_ratio
    return position


def parse_a3_multi_turn_count(payload: bytes) -> int:
    """Extract the cumulative encoder count used to latch the startup zero."""
    if len(payload) < 7 or payload[0] != 0xA3:
        raise ValueError("invalid A3 response")
    return int(struct.unpack("<i", payload[3:7])[0])


def parse_b0_torque_constant(payload: bytes) -> float:
    """Parse B0 torque constant in N m/A from bytes 2..5."""
    if len(payload) < 6 or payload[0] != 0xB0:
        raise ValueError("invalid B0 response")
    value = float(struct.unpack("<f", payload[2:6])[0])
    if not math.isfinite(value) or not 0.01 <= value <= 10.0:
        raise ValueError(f"implausible B0 torque constant: {value}")
    return value


def q_current_payload(current_a: float) -> bytes:
    """Encode C0 Q-axis current in 0.001 A signed units."""
    if not math.isfinite(current_a):
        raise ValueError("current must be finite")
    raw = int(round(float(current_a) * 1000.0))
    if not -(2**31) <= raw < 2**31:
        raise OverflowError("current does not fit the C0 protocol")
    return bytes([0xC0]) + struct.pack("<i", raw)


def c1_speed_payload(velocity_rad_s: float, gear_ratio: float, sign: float) -> bytes:
    """Encode a joint rad/s target as C1 signed int32 0.01 rpm in the A2 native frame.

    C1 regulates the same native speed that A2 reports, so the A2 encoder sign
    converts in both directions (deploy/experiments/drive_speed_sweep.py).
    """
    if not math.isfinite(velocity_rad_s):
        raise ValueError("speed must be finite")
    if gear_ratio <= 0.0 or sign not in (-1.0, 1.0):
        raise ValueError("gear_ratio must be positive and sign must be +/-1")
    raw = int(round(sign * gear_ratio * float(velocity_rad_s) * 60.0 / (2.0 * math.pi) * 100.0))
    if not -(2**31) <= raw < 2**31:
        raise OverflowError("speed does not fit the C1 protocol")
    return bytes([0xC1]) + struct.pack("<i", raw)


def parse_c1_speed(payload: bytes, gear_ratio: float, sign: float) -> float:
    """Decode a C1 command or reply speed back to joint rad/s."""
    if len(payload) != 5 or payload[0] != 0xC1:
        raise ValueError("invalid C1 payload")
    return sign * struct.unpack("<i", payload[1:5])[0] * 0.01 * (2.0 * math.pi / 60.0) / gear_ratio


def b3_current_limit_payload(current_a: float | None = None) -> bytes:
    """B3 read (no value) or write of the volatile speed/position-loop Q-current ceiling."""
    if current_a is None:
        return bytes([0xB3])
    if not math.isfinite(current_a) or current_a <= 0.0:
        raise ValueError("B3 current ceiling must be positive and finite")
    raw = int(math.floor(float(current_a) * 1000.0))  # Never round a ceiling upward.
    if not 1 <= raw < 2**32:
        raise OverflowError("current ceiling does not fit the B3 protocol")
    return bytes([0xB3]) + struct.pack("<I", raw)


def parse_b3_current_limit(payload: bytes) -> float:
    if len(payload) != 5 or payload[0] != 0xB3:
        raise ValueError("invalid B3 response")
    return struct.unpack("<I", payload[1:5])[0] * 0.001


def await_response(bus, feedback_id: int, command: int, timeout_s: float) -> bytes | None:
    """Receive until a feedback frame echoes the requested command."""
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None
        try:
            message = bus.recv(timeout=remaining)
        except Exception:
            return None
        if message is None:
            return None
        if message.arbitration_id != feedback_id or not message.data:
            continue
        if message.data[0] == command:
            return bytes(message.data)


def query(bus, request, feedback_id: int, timeout_s: float) -> bytes | None:
    """Drain old feedback, send one query, and match its command echo."""
    for _ in range(64):
        try:
            if bus.recv(timeout=0.0) is None:
                break
        except Exception:
            break
    try:
        bus.send(request)
    except Exception:
        return None
    return await_response(bus, feedback_id, request.data[0], timeout_s)
