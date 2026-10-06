#!/usr/bin/env python3
"""Send an absolute-position MIT steering sine using only SocketCAN/stdlib.

GIM CAN V3.09b0, pp.20–22: position is in native motor radians, not encoder
counts or an incremental displacement. Disable steering with CF, set the
current position as the origin with B1 and wait for its acknowledgement on
every execution. Configure F0 Pos_Max to 3 rad, preserving its actual velocity
and torque ranges, then verify F0, A3 and F1 before any motion.
B1's origin and F0's ranges persist across power cycles; preview opens no CAN.
Target = startup_position + sign * gear_ratio * amplitude * sin(2*pi*t/period).
Requested velocity/torque are always zero; gains are fixed at Kp=10, Kd=0.2.
These SI gains apply to motor coordinates, including when --gear-ratio != 1.

Defaults match steering_sine_position.py: can0, address 2, sign -1, gear 1,
nominal 200 Hz, +/-1 joint rad, 10 s period. A positive sine is rightward in
the existing steering feedback convention. No ROS nodes/build are required.
Stop other motor command writers before execution. CF disables this motor
on normal completion, Ctrl-C/SIGTERM, or an exception after a motion attempt.
There is no continuous feedback/watchdog or write to firmware gains/current limits.
MIT torque conversion uses the motor's already configured torque constant.

Symmetric 12-bit fields cannot encode exact zero: nearest code 2048 is used.
F0 ranges determine the residual. The nonnegative Kp/Kd encoding is distinct
from position/velocity/torque encoding; the supplied vendor helper gets this
wrong and is deliberately not imported. Preview uses labelled assumptions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import signal
import socket
import struct
import sys
import time


RATE_HZ = 200.0
KP_NM_PER_RAD = 10.0
KD_NM_S_PER_RAD = 0.2
CAN_FRAME = struct.Struct("=IB3x8s")
REPLY_TIMEOUT_S = 1.0
POSITION_MAX_RAD = 3.0
MAX_REPLY_DRAIN = 256
ENCODER_COUNT_RAD = 2.0 * math.pi / 16384
# Small startup drift is permitted; these checks do not establish physical alignment.
ZERO_TOLERANCE_COUNTS = 2
STARTUP_SPEED_TOL_RAD_S = 0.1
# CAN_ERR_FLAG in a filter mask subscribes to error frames instead of data.
CAN_FILTER_MASK = 0xC00007FF


@dataclass(frozen=True)
class MITRanges:
    position_rad: float
    velocity_rad_s: float
    torque_nm: float

    def __post_init__(self):
        if not all(math.isfinite(value) and value > 0.0 for value in vars(self).values()):
            raise ValueError("F0 position/velocity/torque encoding ranges must be positive and finite")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Send MIT commands (also the default).")
    mode.add_argument("--preview", action="store_true", help="Print frames only; do not open CAN.")
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--motor-id", type=lambda value: int(value, 0), default=2, help="Device address (default: 2).")
    parser.add_argument("--gear-ratio", type=finite_float, default=1.0, help="Motor radians/joint radian (default: 1).")
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--amplitude", type=finite_float, default=1.0, help="Peak joint radians (default: 1).")
    parser.add_argument("--period", type=finite_float, default=10.0, help="Seconds per cycle (default: 10).")
    parser.add_argument("--duration", type=finite_float, default=0.0, help="Seconds; 0 runs until Ctrl-C.")
    args = parser.parse_args(argv)
    if not 1 <= args.motor_id <= 254:
        parser.error("--motor-id must be 1..254; broadcast addresses are not supported")
    if args.amplitude <= 0.0 or args.gear_ratio <= 0.0:
        parser.error("--amplitude and --gear-ratio must be positive")
    if args.period <= 2.0 / RATE_HZ:
        parser.error("--period must exceed 0.01 s (sine frequency must be below 100 Hz)")
    if args.duration < 0.0:
        parser.error("--duration must be nonnegative")
    return args


def encode_unsigned(value, lower, upper, bits):
    """Nearest unsigned linear code; reject invalid/out-of-range inputs."""
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"value {value} outside encoding range [{lower}, {upper}]")
    return round((value - lower) / (upper - lower) * ((1 << bits) - 1))


def decode_symmetric(code, maximum, bits):
    return (2.0 * code / ((1 << bits) - 1) - 1.0) * maximum


def mit_payload(position_rad, ranges):
    """Eight high-bit-first bytes, no opcode; gains use nonnegative ranges."""
    position = encode_unsigned(position_rad, -ranges.position_rad, ranges.position_rad, 16)
    velocity = encode_unsigned(0.0, -ranges.velocity_rad_s, ranges.velocity_rad_s, 12)
    kp = encode_unsigned(KP_NM_PER_RAD, 0.0, 500.0, 12)
    kd = encode_unsigned(KD_NM_S_PER_RAD, 0.0, 5.0, 12)
    torque = encode_unsigned(0.0, -ranges.torque_nm, ranges.torque_nm, 12)
    return bytes([
        position >> 8, position & 0xFF,
        velocity >> 4, ((velocity & 0xF) << 4) | (kp >> 8), kp & 0xFF,
        kd >> 4, ((kd & 0xF) << 4) | (torque >> 8), torque & 0xFF,
    ])


def sine_position(args, elapsed_s, zero_rad):
    offset_rad = args.amplitude * math.sin(2.0 * math.pi * (elapsed_s % args.period) / args.period)
    # The native absolute origin must not be sign-inverted or gear-scaled.
    return zero_rad + args.sign * args.gear_ratio * offset_rad


def validate_trajectory(args, zero_rad, ranges):
    peak_motor_rad = args.amplitude * args.gear_ratio
    if not math.isfinite(peak_motor_rad) or abs(zero_rad) + peak_motor_rad > ranges.position_rad:
        raise ValueError("startup position +/- sine amplitude exceeds F0 Pos_Max; no MIT motion sent")


def send_frame(bus, can_id, payload):
    frame = CAN_FRAME.pack(can_id, len(payload), payload)
    if bus.send(frame) != len(frame):
        raise OSError("incomplete SocketCAN frame send")


def read_reply(bus, motor_id, opcode, reply_length=7, request_payload=None, expected_payload=None):
    """Discard queued replies, send a request, then await its matching response.

    CAN replies have no request IDs; other motor command writers must be stopped.
    """
    try:
        bus.settimeout(0.0)
        for _ in range(MAX_REPLY_DRAIN):
            try:
                bus.recv(CAN_FRAME.size)
            except BlockingIOError:
                break
        else:
            raise RuntimeError(f"CAN reply queue did not clear before {opcode:02X}; no MIT motion sent")
        bus.settimeout(REPLY_TIMEOUT_S)
        send_frame(bus, 0x100 | motor_id, request_payload if request_payload is not None else bytes([opcode]))
        deadline = time.monotonic() + REPLY_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"no valid {opcode:02X} startup reply; no MIT motion sent")
            bus.settimeout(remaining)
            try:
                frame = bus.recv(CAN_FRAME.size)
            except TimeoutError as error:
                raise TimeoutError(f"no valid {opcode:02X} startup reply; no MIT motion sent") from error
            if len(frame) != CAN_FRAME.size:
                continue
            can_id, length, data = CAN_FRAME.unpack(frame)
            if can_id == motor_id and length == reply_length and data[0] == opcode:
                payload = data[:length]
                if expected_payload is None or payload == expected_payload:
                    return payload
    finally:
        bus.settimeout(REPLY_TIMEOUT_S)


def read_startup(bus, motor_id):
    disabled = read_reply(bus, motor_id, 0xCF, reply_length=8)
    if disabled[6] != 0 or disabled[7] != 0:
        raise RuntimeError("CF did not confirm disabled/fault-free motor state; no MIT motion sent")
    read_reply(bus, motor_id, 0xB1, reply_length=3)
    print("B1 acknowledged: current steering position set as origin.", flush=True)
    config = read_reply(bus, motor_id, 0xF0)
    position, velocity, torque = struct.unpack("<HHH", config[1:7])
    MITRanges(position * 0.1, velocity * 0.01, torque * 0.01)  # Reject invalid ranges before writing.
    requested = b"\xf0" + struct.pack("<H", round(POSITION_MAX_RAD / 0.1)) + config[3:7]
    read_reply(bus, motor_id, 0xF0, request_payload=requested, expected_payload=requested)
    verified = read_reply(bus, motor_id, 0xF0)
    if verified != requested:
        raise RuntimeError("F0 read-back does not match requested ranges; no MIT motion sent")
    position, velocity, torque = struct.unpack("<HHH", verified[1:7])
    ranges = MITRanges(position * 0.1, velocity * 0.01, torque * 0.01)
    print(f"F0 verified: +/-{ranges.position_rad:g} rad; velocity/torque ranges preserved.", flush=True)
    encoder = read_reply(bus, motor_id, 0xA3)
    single_count, multi_count = struct.unpack("<Hi", encoder[1:7])
    single_delta = (single_count + 8192) % 16384 - 8192
    if abs(single_delta) > ZERO_TOLERANCE_COUNTS or abs(multi_count) > ZERO_TOLERANCE_COUNTS:
        raise RuntimeError(f"A3 not near B1 zero: single={single_count}, multi={multi_count}; no MIT motion sent")
    feedback = read_reply(bus, motor_id, 0xF1)
    if feedback[6] & 0x02:
        raise RuntimeError("F1 reports a motor system fault; no MIT motion sent")
    if feedback[6] & 0x01:
        raise RuntimeError("F1 reports MIT active after disable; stop other command writers; no MIT motion sent")
    zero_code = int.from_bytes(feedback[1:3], "big")
    zero_rad = decode_symmetric(zero_code, ranges.position_rad, 16)
    velocity_code = (feedback[3] << 4) | (feedback[4] >> 4)
    velocity_rad_s = decode_symmetric(velocity_code, ranges.velocity_rad_s, 12)
    torque_code = ((feedback[4] & 0xF) << 8) | feedback[5]
    torque_nm = decode_symmetric(torque_code, ranges.torque_nm, 12)
    # Allow one F1 increment in addition to the stated physical tolerance.
    position_tolerance_rad = ZERO_TOLERANCE_COUNTS * ENCODER_COUNT_RAD + 2 * ranges.position_rad / 65535
    speed_tolerance_rad_s = STARTUP_SPEED_TOL_RAD_S + 2 * ranges.velocity_rad_s / 4095
    if abs(zero_rad) > position_tolerance_rad or abs(velocity_rad_s) > speed_tolerance_rad_s:
        raise RuntimeError(f"F1 not near stationary zero: p={zero_rad:.6g} rad, v={velocity_rad_s:.6g} rad/s; "
                           "no MIT motion sent")
    print(f"Startup verified: A3 single={single_count}, multi={multi_count}; "
          f"F1 p={zero_rad:.9f} rad, v={velocity_rad_s:.6g} rad/s, torque={torque_nm:.6g} Nm, "
          f"status=0x{feedback[6]:02X}.", flush=True)
    return ranges, zero_rad


def print_encoding(ranges, zero_rad):
    kp_code = encode_unsigned(KP_NM_PER_RAD, 0.0, 500.0, 12)
    kd_code = encode_unsigned(KD_NM_S_PER_RAD, 0.0, 5.0, 12)
    print(
        f"Centre={zero_rad:.9f} native motor rad; F0 ranges=+/-{ranges.position_rad:g} rad, "
        f"+/-{ranges.velocity_rad_s:g} rad/s, +/-{ranges.torque_nm:g} Nm.\n"
        f"Requested v=0, torque_ff=0; nearest encoded values: "
        f"v={decode_symmetric(2048, ranges.velocity_rad_s, 12):.9f} rad/s, "
        f"torque_ff={decode_symmetric(2048, ranges.torque_nm, 12):.9f} Nm.\n"
        f"Requested Kp={KP_NM_PER_RAD:g}, Kd={KD_NM_S_PER_RAD:g}; "
        f"encoded gains={kp_code * 500 / 4095:.9f}, {kd_code * 5 / 4095:.9f}.",
        flush=True,
    )


def preview(args):
    ranges = MITRanges(POSITION_MAX_RAD, 45.0, 18.0)
    zero_rad = 0.0
    validate_trajectory(args, zero_rad, ranges)
    print("PREVIEW ONLY: Pos_Max=3 rad, assumed velocity/torque defaults and centre=0. "
          "Execution sets B1/F0 and verifies CF/F0/A3/F1. No CAN opened or settings written.")
    print_encoding(ranges, zero_rad)
    for index in range(math.ceil((args.duration or args.period) * RATE_HZ)):
        elapsed_s = index / RATE_HZ
        target_rad = sine_position(args, elapsed_s, zero_rad)
        payload = mit_payload(target_rad, ranges)
        print(f"t={elapsed_s:.3f}s p_abs={target_rad:.9f} rad {0x500 | args.motor_id:03X}#{payload.hex().upper()}")


def execute(args):
    with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, struct.pack("=II", args.motor_id, CAN_FILTER_MASK))
        bus.settimeout(REPLY_TIMEOUT_S)
        bus.bind((args.interface,))
        ranges, zero_rad = read_startup(bus, args.motor_id)
        validate_trajectory(args, zero_rad, ranges)
        print_encoding(ranges, zero_rad)
        print(
            f"MIT TX {0x500 | args.motor_id:03X}: +/-{args.amplitude:g} joint rad, "
            f"period={args.period:g} s, nominal {RATE_HZ:g} Hz; CF on exit.", flush=True,
        )
        # No feedback is used during the waveform; avoid accumulating unused replies.
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b"")
        start_s = time.monotonic()
        motion_attempted = False
        try:
            while True:
                tick_start = time.monotonic()
                elapsed_s = tick_start - start_s
                if args.duration and elapsed_s >= args.duration:
                    break
                payload = mit_payload(sine_position(args, elapsed_s, zero_rad), ranges)
                motion_attempted = True
                send_frame(bus, 0x500 | args.motor_id, payload)
                time.sleep(max(0.0, 1.0 / RATE_HZ - (time.monotonic() - tick_start)))
        finally:
            if motion_attempted:
                try:
                    send_frame(bus, 0x100 | args.motor_id, b"\xcf")
                    print("Sent CF to disable steering and exit MIT mode.", flush=True)
                except OSError as error:
                    print(f"CF send failed: {error}", file=sys.stderr)
                    raise


def main(argv=None):
    args = parse_args(argv)
    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        if args.preview:
            preview(args)
        else:
            execute(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    sys.exit(main())
