"""Pure CRSF parsing and kickstand-switch helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


CRSF_SYNC_BYTES = (0xC8, 0xEA, 0xEC, 0xEE)
CRSF_RC_CHANNELS_PACKED = 0x16
CRSF_CHANNEL_ABS_MAX = 1984
CRC_MODES = ("strict", "soft", "off")


def crc8_dvb_s2(data: Iterable[int]) -> int:
    """Return the CRC-8/DVB-S2 checksum used by CRSF."""
    crc = 0
    for value in data:
        crc ^= int(value) & 0xFF
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def unpack_rc_channels(payload: bytes) -> tuple[int, ...] | None:
    """Decode the 22-byte CRSF payload into sixteen 11-bit channels."""
    if len(payload) < 22:
        return None
    channels: list[int] = []
    bits = 0
    bits_available = 0
    byte_index = 0
    for _ in range(16):
        while bits_available < 11:
            if byte_index >= len(payload):
                return None
            bits |= payload[byte_index] << bits_available
            bits_available += 8
            byte_index += 1
        channels.append(bits & 0x07FF)
        bits >>= 11
        bits_available -= 11
    return tuple(channels)


def channels_look_like_crsf(channels: Sequence[int]) -> bool:
    """Apply the same soft-CRC plausibility test as ros2_ws."""
    if len(channels) < 8 or any(
        value < 0 or value > CRSF_CHANNEL_ABS_MAX for value in channels
    ):
        return False
    return sum(172 <= value <= 1811 for value in channels[:8]) >= 4


def decode_crsf_frame(frame: bytes, crc_mode: str = "soft") -> tuple[int, ...] | None:
    """Decode one complete CRSF packed-channel frame."""
    if crc_mode not in CRC_MODES:
        raise ValueError(f"crc_mode must be one of {CRC_MODES}")
    if len(frame) < 4 or frame[0] not in CRSF_SYNC_BYTES:
        return None
    length = frame[1]
    if length < 2 or length > 62 or len(frame) < length + 2:
        return None
    body = frame[2 : 2 + length]
    if body[0] != CRSF_RC_CHANNELS_PACKED:
        return None
    channels = unpack_rc_channels(body[1:-1])
    if channels is None:
        return None
    crc_ok = crc8_dvb_s2(body[:-1]) == body[-1]
    if not crc_ok and (
        crc_mode == "strict"
        or (crc_mode == "soft" and not channels_look_like_crsf(channels))
    ):
        return None
    return channels


class CrsfStreamParser:
    """Incrementally extract CRSF frames from a UART byte stream."""

    def __init__(self, crc_mode: str = "soft") -> None:
        if crc_mode not in CRC_MODES:
            raise ValueError(f"crc_mode must be one of {CRC_MODES}")
        self._crc_mode = crc_mode
        self._buffer = bytearray()

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes) -> list[tuple[int, ...]]:
        self._buffer.extend(data)
        frames: list[tuple[int, ...]] = []
        while len(self._buffer) >= 4:
            start = next(
                (
                    index
                    for index, value in enumerate(self._buffer)
                    if value in CRSF_SYNC_BYTES
                ),
                -1,
            )
            if start < 0:
                self._buffer.clear()
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 2:
                break
            length = self._buffer[1]
            if length < 2 or length > 62:
                del self._buffer[0]
                continue
            total = length + 2
            if len(self._buffer) < total:
                break
            channels = decode_crsf_frame(bytes(self._buffer[:total]), self._crc_mode)
            del self._buffer[:total]
            if channels is not None:
                frames.append(channels)
        return frames


def decode_three_position_switch(
    channels: Sequence[int],
    input_min: int,
    input_center: int,
    input_max: int,
    channel: int,
    high_is_up: bool,
    band_fraction: float,
) -> bool | None:
    """Return up/down at the ends and None in the hold-middle region."""
    value = int(channels[channel - 1])
    fraction = max(0.05, min(0.49, band_fraction))
    high_enter = input_center + round(
        (input_max - input_center) * (1.0 - fraction)
    )
    low_enter = input_center - round((input_center - input_min) * (1.0 - fraction))
    if value >= high_enter:
        return high_is_up
    if value <= low_enter:
        return not high_is_up
    return None


@dataclass
class BoolDebouncer:
    """Confirm a switch end after the configured number of equal frames."""

    required_frames: int
    confirmed: bool | None = None
    pending: bool | None = None
    pending_count: int = 0

    def observe(self, state: bool | None) -> bool | None:
        if state is None:
            self.pending = None
            self.pending_count = 0
            return self.confirmed
        if state == self.confirmed:
            self.pending = None
            self.pending_count = 0
            return self.confirmed
        if state == self.pending:
            self.pending_count += 1
        else:
            self.pending = state
            self.pending_count = 1
        if self.pending_count >= self.required_frames:
            self.confirmed = state
            self.pending = None
            self.pending_count = 0
        return self.confirmed
