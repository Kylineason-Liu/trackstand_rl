"""GIM V3.09b0 MIT wire encoding; all positions/velocities are native motor units."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class MitRanges:
    position_rad: float
    velocity_rad_s: float
    torque_nm: float

    def __post_init__(self):
        values = (self.position_rad, self.velocity_rad_s, self.torque_nm)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("MIT encoding ranges must be positive and finite")


@dataclass(frozen=True)
class MitFeedback:
    position_rad: float
    velocity_rad_s: float
    torque_nm: float
    mode_active: bool
    system_fault: bool
    status_raw: int


@dataclass(frozen=True)
class MitCommand:
    position_rad: float
    velocity_rad_s: float
    kp_nm_per_rad: float
    kd_nm_s_per_rad: float
    feedforward_torque_nm: float


def decode_symmetric(code: int, maximum: float, bits: int) -> float:
    return (2.0 * code / ((1 << bits) - 1) - 1.0) * maximum


def encode(value: float, lower: float, upper: float, bits: int) -> int:
    if not all(map(math.isfinite, (value, lower, upper))) or lower >= upper or not lower <= value <= upper:
        raise ValueError(f"MIT value {value} outside [{lower}, {upper}]")
    return round((value - lower) * ((1 << bits) - 1) / (upper - lower))


def parse_f0(payload: bytes) -> MitRanges:
    if len(payload) != 7 or payload[0] != 0xF0:
        raise ValueError("invalid F0 response")
    raw = struct.unpack("<HHH", payload[1:])
    if any(value == 0 for value in raw):
        raise ValueError("F0 encoding ranges must be nonzero")
    return MitRanges(raw[0] * 0.1, raw[1] * 0.01, raw[2] * 0.01)


def f0_with_position(payload: bytes, position_max_rad: float) -> bytes:
    parse_f0(payload)
    if not math.isfinite(position_max_rad):
        raise ValueError("MIT position maximum must be finite")
    raw = round(position_max_rad / 0.1)
    if not 1 <= raw <= 65535:
        raise ValueError("MIT position maximum cannot be encoded in F0")
    return bytes([0xF0]) + struct.pack("<H", raw) + payload[3:]


def parse_f1(payload: bytes, ranges: MitRanges) -> MitFeedback:
    if len(payload) != 7 or payload[0] != 0xF1:
        raise ValueError("invalid F1 response")
    p = (payload[1] << 8) | payload[2]
    v = (payload[3] << 4) | (payload[4] >> 4)
    t = ((payload[4] & 15) << 8) | payload[5]
    return MitFeedback(
        decode_symmetric(p, ranges.position_rad, 16),
        decode_symmetric(v, ranges.velocity_rad_s, 12),
        decode_symmetric(t, ranges.torque_nm, 12),
        bool(payload[6] & 1), bool(payload[6] & 2), payload[6],
    )


def mit_motion_payload(
    position_rad: float, ranges: MitRanges, kp_nm_per_rad: float, kd_nm_s_per_rad: float,
) -> bytes:
    """Position absolute; velocity/feedforward torque zero; gains use unsigned ranges."""
    p = encode(position_rad, -ranges.position_rad, ranges.position_rad, 16)
    v = encode(0.0, -ranges.velocity_rad_s, ranges.velocity_rad_s, 12)
    kp = encode(kp_nm_per_rad, 0.0, 500.0, 12)
    kd = encode(kd_nm_s_per_rad, 0.0, 5.0, 12)
    t = encode(0.0, -ranges.torque_nm, ranges.torque_nm, 12)
    return bytes((p >> 8, p & 255, v >> 4, ((v & 15) << 4) | (kp >> 8), kp & 255,
                  kd >> 4, ((kd & 15) << 4) | (t >> 8), t & 255))


def parse_mit_motion(payload: bytes, ranges: MitRanges) -> MitCommand:
    """Decode a command for diagnostics, including unavoidable zero quantization."""
    if len(payload) != 8:
        raise ValueError("invalid MIT motion payload")
    p = (payload[0] << 8) | payload[1]
    v = (payload[2] << 4) | (payload[3] >> 4)
    kp = ((payload[3] & 15) << 8) | payload[4]
    kd = (payload[5] << 4) | (payload[6] >> 4)
    t = ((payload[6] & 15) << 8) | payload[7]
    return MitCommand(decode_symmetric(p, ranges.position_rad, 16),
                      decode_symmetric(v, ranges.velocity_rad_s, 12), kp * 500 / 4095, kd * 5 / 4095,
                      decode_symmetric(t, ranges.torque_nm, 12))
