#!/usr/bin/env python3
"""Read-only MIT experiment recorder, using SocketCAN and Python's stdlib.

Keep the original A3/A2/A1/B0 measurements and their units/timestamps. A second
receive-only socket captures MIT commands, F0 ranges, F1 state and CF requests.
Only one-byte A1/A2/A3/B0/F0/F1 reads are transmitted; shutdown sends no commands.
No default MIT ranges are assumed. Raw frames/codes remain available when the
actual ranges are unknown. All times are host receipt times, not motor clocks.
For the current sine sender, queries pause from startup CF/B1 through the first
MIT target. F0 writes/confirmations and the sender's A3/F1 startup zeros are
captured passively; the recorder never writes the sender's +/-3 rad range.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import math
import queue
import signal
import socket
import struct
import sys
import threading
import time

CAN_FRAME = struct.Struct("=IB3x8s")
# CAN_ERR_FLAG in a filter mask selects Linux's error-frame receive list.
# Standard data filters include EFF/RTR flags, but MUST NOT include ERR.
CAN_MASK = 0xC00007FF
RADIANS_PER_COUNT = 2.0 * math.pi / 16384
READ_LENGTHS = {0xA1: 5, 0xA2: 5, 0xA3: 7, 0xB0: 7, 0xF0: 7, 0xF1: 7}
GROUP_IDS = (0x00, 0xFF, 0x400, 0x4FF)
# Linux UAPI asm-generic/socket.h and socket(7); not exported by all Python builds.
SO_RXQ_OVFL = getattr(socket, "SO_RXQ_OVFL", 40)
RX_BUFFER_BYTES = 1024 * 1024
MAX_QUERY_DRAIN = 128
MAX_WRITE_BATCH = 1000
MAX_PENDING_ROWS = 20000
SHUTDOWN_DRAIN_S = 0.25
FIELDS = (
    "record_type", "time_unix_s", "elapsed_s", "motor", "motor_id", "encoder_count", "zero_count",
    "position_rad", "position_unwrapped_rad", "velocity_rad_s", "current_a",
    "torque_nm", "torque_constant_nm_per_a", "position_elapsed_s", "velocity_elapsed_s", "current_elapsed_s",
    "gear_ratio", "encoder_sign", "torque_sign", "mit_can_id", "mit_payload_hex",
    "mit_position_code", "mit_velocity_code", "mit_kp_code", "mit_kd_code", "mit_torque_code",
    "mit_position_max_rad", "mit_velocity_max_rad_s", "mit_torque_max_nm", "mit_config_elapsed_s",
    "mit_config_source", "mit_decode_ready", "mit_zero_position_motor_rad",
    "mit_target_position_motor_rad", "mit_target_position_rad",
    "mit_target_velocity_motor_rad_s", "mit_target_velocity_rad_s",
    "mit_kp_nm_per_motor_rad", "mit_kd_nm_s_per_motor_rad",
    "mit_feedforward_torque_motor_nm", "mit_feedforward_torque_nm",
    "mit_feedback_position_motor_rad", "mit_feedback_position_rad",
    "mit_feedback_velocity_motor_rad_s", "mit_feedback_velocity_rad_s",
    "mit_feedback_torque_motor_nm", "mit_feedback_torque_nm", "mit_feedback_elapsed_s",
    "mit_status_raw", "mit_is_active", "mit_has_fault",
    "can_direction", "can_opcode", "can_length", "can_note", "socket_dropped_frames",
    "bus_voltage_v", "bus_current_a", "bus_current_raw", "motor_temperature_c", "motor_mode_raw", "motor_fault_code",
    "a1_reply_received", "a2_reply_received", "a3_reply_received", "b0_reply_received",
    "f0_reply_received", "f1_reply_received",
    "reference_epoch", "origin_pending",
    "startup_pending", "measurement_epoch", "mit_config_write_pending", "encoder_single_turn_count",
    "mit_requested_position_max_rad", "mit_requested_velocity_max_rad_s", "mit_requested_torque_max_nm",
)


@dataclass
class Motor:
    name: str
    device_id: int
    gear_ratio: float
    encoder_sign: int
    torque_sign: int
    zero_count: int | None = None
    torque_constant: float = math.nan
    retry_b0_at: float = 0.0
    mit_ranges: tuple[float, float, float] | None = None
    mit_config_elapsed_s: float = math.nan
    mit_zero_position_motor_rad: float = math.nan
    retry_mit_at: float = 0.0
    reference_invalid: bool = False
    reference_epoch: int = 0
    startup_pending: bool = False
    mit_motion_active: bool = False  # An observed target, not proof of motor enable.
    measurement_epoch: int = 0
    mit_config_requested: bytes | None = None
    mit_config_write_pending: bool = False
    lock: object = field(default_factory=threading.Lock, repr=False, compare=False)


@dataclass(frozen=True)
class Packet:
    frame: bytes
    received_s: float
    wall_time_s: float
    local_tx: bool
    dropped_frames: int = 0


@dataclass(frozen=True)
class Reply:
    data: bytes
    received_s: float
    wall_time_s: float


def receive_packet(bus):
    frame, ancillary, flags, _ = bus.recvmsg(CAN_FRAME.size, socket.CMSG_SPACE(4))
    received_s, wall_time_s = time.monotonic(), time.time()
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        raise RuntimeError("truncated CAN frame/receive metadata; recording is incomplete")
    dropped = 0
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == SO_RXQ_OVFL:
            if len(data) != 4:
                raise RuntimeError("invalid SO_RXQ_OVFL receive metadata")
            dropped = struct.unpack("=I", data)[0]
    return Packet(frame, received_s, wall_time_s, bool(flags & socket.MSG_DONTROUTE), dropped)


def check_overflow(packet, channel):
    if packet.dropped_frames:
        raise RuntimeError(f"{channel} socket reports {packet.dropped_frames} dropped CAN frames; recording incomplete")


def unpack_standard(frame):
    if len(frame) != CAN_FRAME.size:
        return None
    can_id, length, data = CAN_FRAME.unpack(frame)
    if can_id & ~0x7FF or length > 8:
        return None
    return can_id, length, data[:length]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Record A3/A2/A1 feedback and passively capture MIT targets/F1 state to CSV, without ROS.",
        epilog="Start before steering_sine_mit.py. Unknown measurements/ranges are NaN, not guessed. "
               "The nominal query rate is not a timing guarantee.",
    )
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--motors", nargs="+", choices=("steering", "drive"), default=["steering", "drive"])
    parser.add_argument("--rate", type=float, default=200.0, help="Target cycles/s per motor (default: 200).")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 means until Ctrl-C.")
    parser.add_argument("--timeout", type=float, default=0.002, help="Maximum wait per query (default: .002 s).")
    parser.add_argument("--output", default=f"motor_feedback_mit_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
    for name, device_id, encoder_sign in (("steering", 2, -1), ("drive", 1, 1)):
        parser.add_argument(f"--{name}-id", type=lambda value: int(value, 0), default=device_id)
        parser.add_argument(f"--{name}-gear-ratio", type=float, default=1.0)
        parser.add_argument(f"--{name}-encoder-sign", type=int, choices=(-1, 1), default=encoder_sign)
        parser.add_argument(f"--{name}-torque-sign", type=int, choices=(-1, 1), default=-1)
    args = parser.parse_args(argv)
    values = (args.rate, args.duration, args.timeout, args.steering_gear_ratio, args.drive_gear_ratio)
    if (not all(math.isfinite(value) for value in values) or args.rate <= 0 or args.duration < 0
            or args.timeout <= 0 or min(args.steering_gear_ratio, args.drive_gear_ratio) <= 0):
        parser.error("rate, timeout and gear ratios must be positive and finite; duration must be nonnegative")
    if not all(1 <= value <= 254 for value in (args.steering_id, args.drive_id)):
        parser.error("device IDs must be 1..254")
    if len(set(args.motors)) != len(args.motors):
        parser.error("each motor may be selected only once")
    if len(args.motors) == 2 and args.steering_id == args.drive_id:
        parser.error("steering and drive must have different IDs")
    return args


def snapshot(motor):
    with motor.lock:
        return (motor.mit_ranges, motor.mit_config_elapsed_s, motor.mit_zero_position_motor_rad,
                motor.reference_epoch, motor.reference_invalid, motor.startup_pending,
                motor.measurement_epoch, motor.mit_config_requested, motor.mit_config_write_pending)


def new_row(motor, record_type, elapsed_s, wall_time):
    row = dict.fromkeys(FIELDS, math.nan)
    for name in ("encoder_count", "encoder_single_turn_count", "zero_count", "mit_can_id",
                 "mit_position_code", "mit_velocity_code",
                 "mit_kp_code", "mit_kd_code", "mit_torque_code", "mit_status_raw", "mit_is_active", "mit_has_fault",
                 "can_opcode", "can_length", "motor_mode_raw", "motor_fault_code",
                 "a1_reply_received", "a2_reply_received", "a3_reply_received", "b0_reply_received",
                 "f0_reply_received", "f1_reply_received"):
        row[name] = None
    row.update(record_type=record_type, elapsed_s=elapsed_s, time_unix_s=wall_time,
               motor=motor.name, motor_id=motor.device_id, gear_ratio=motor.gear_ratio,
               encoder_sign=motor.encoder_sign, torque_sign=motor.torque_sign,
               mit_payload_hex="", mit_config_source="", can_direction="", can_note="", socket_dropped_frames=0)
    (ranges, configured_at, zero_rad, epoch, origin_pending, startup_pending,
     measurement_epoch, requested, config_pending) = snapshot(motor)
    row.update(mit_decode_ready=ranges is not None, mit_config_elapsed_s=configured_at,
               mit_zero_position_motor_rad=zero_rad, reference_epoch=epoch, origin_pending=origin_pending,
               startup_pending=startup_pending, measurement_epoch=measurement_epoch,
               mit_config_write_pending=config_pending)
    if ranges is not None:
        row.update(mit_position_max_rad=ranges[0], mit_velocity_max_rad_s=ranges[1], mit_torque_max_nm=ranges[2])
    if requested is not None:
        p, v, t = struct.unpack("<HHH", requested[1:7])
        row.update(mit_requested_position_max_rad=p * 0.1,
                   mit_requested_velocity_max_rad_s=v * 0.01, mit_requested_torque_max_nm=t * 0.01)
    return row


def decode_symmetric(code, maximum, bits):
    return (2.0 * code / ((1 << bits) - 1) - 1.0) * maximum


def process_frame(frame, by_id, start, received, wall_time, local_tx=False):
    """Observer-only decoding/state updates; preserve one row per relevant frame."""
    unpacked = unpack_standard(frame)
    if unpacked is None:
        return None
    can_id, length, data = unpacked
    motor = by_id.get(can_id)
    if motor is None:
        return None
    elapsed_s = received - start
    host_id = 0x100 | motor.device_id
    opcode = data[0] if data else None
    host_request = can_id in (host_id, 0x00, 0xFF) or (can_id == motor.device_id and local_tx)
    motor_reply = can_id == motor.device_id and not local_tx
    is_target = can_id in (0x400 | motor.device_id, 0x500 | motor.device_id, 0x400, 0x4FF) and length == 8
    is_config_reply = motor_reply and length == 7 and opcode == 0xF0
    is_config_write = host_request and length == 7 and opcode == 0xF0
    is_feedback = motor_reply and length == 7 and opcode == 0xF1
    is_disable = (host_request or can_id == motor.device_id) and data == b"\xcf"
    is_origin_request = (host_request or can_id == motor.device_id) and data == b"\xb1"
    is_origin_ack = motor_reply and length == 3 and opcode == 0xB1
    is_reboot = (host_request or can_id == motor.device_id) and data == b"\x00\xff\x00\xff\x00\xff\x00\xff"
    if is_config_reply or is_config_write:
        # A write is not confirmation of the active ranges: invalidate until a reply.
        values = struct.unpack("<HHH", data[1:7])
        ranges = (values[0] * 0.1, values[1] * 0.01, values[2] * 0.01)
        note = ""
        with motor.lock:
            previous = motor.mit_ranges
            if is_config_write:
                motor.mit_config_requested = data
                motor.mit_config_write_pending = True
                motor.startup_pending = True
                motor.measurement_epoch += 1
                motor.mit_zero_position_motor_rad = math.nan
                motor.mit_ranges = None
                note = "F0 write observed; awaiting exact range confirmation and fresh F1 zero"
            elif motor.reference_invalid:
                motor.mit_ranges = None
                note = "F0 reply during pending origin reset; not used for decoding"
            elif motor.mit_config_requested is not None and data != motor.mit_config_requested:
                motor.mit_config_write_pending = True
                motor.mit_ranges = None
                note = "F0 reply differs from latest observed write; decoding remains unavailable"
            else:
                motor.mit_ranges = ranges if all(value > 0 for value in ranges) else None
                motor.mit_config_write_pending = motor.mit_config_requested is not None and motor.mit_ranges is None
            if previous is not None and previous != motor.mit_ranges:
                motor.mit_zero_position_motor_rad = math.nan
                if not is_config_write:
                    motor.measurement_epoch += 1
            motor.mit_config_elapsed_s = elapsed_s
        row = new_row(motor, "mit_config", elapsed_s, wall_time)
        row["mit_config_source"] = "reply" if is_config_reply else "write_request"
        row["can_note"] = note
    elif is_target:
        with motor.lock:
            motor.mit_motion_active = True
            if (motor.startup_pending and not motor.reference_invalid
                    and motor.mit_ranges is not None and not motor.mit_config_write_pending):
                motor.startup_pending = False
                motor.measurement_epoch += 1
        row = new_row(motor, "mit_target", elapsed_s, wall_time)
        position = int.from_bytes(data[:2], "big")
        velocity = (data[2] << 4) | (data[3] >> 4)
        kp = ((data[3] & 15) << 8) | data[4]
        kd = (data[5] << 4) | (data[6] >> 4)
        torque = ((data[6] & 15) << 8) | data[7]
        row.update(mit_position_code=position, mit_velocity_code=velocity, mit_kp_code=kp,
                   mit_kd_code=kd, mit_torque_code=torque,
                   mit_kp_nm_per_motor_rad=kp * 500.0 / 4095, mit_kd_nm_s_per_motor_rad=kd * 5.0 / 4095)
        if row["mit_decode_ready"]:
            p = decode_symmetric(position, row["mit_position_max_rad"], 16)
            v = decode_symmetric(velocity, row["mit_velocity_max_rad_s"], 12)
            t = decode_symmetric(torque, row["mit_torque_max_nm"], 12)
            row.update(mit_target_position_motor_rad=p,
                       mit_target_position_rad=motor.encoder_sign * (p - row["mit_zero_position_motor_rad"])
                       / motor.gear_ratio,
                       mit_target_velocity_motor_rad_s=v, mit_target_velocity_rad_s=motor.encoder_sign * v
                       / motor.gear_ratio, mit_feedforward_torque_motor_nm=t,
                       mit_feedforward_torque_nm=motor.torque_sign * t)
    elif is_feedback:
        row = new_row(motor, "mit_feedback", elapsed_s, wall_time)
        position = int.from_bytes(data[1:3], "big")
        velocity = (data[3] << 4) | (data[4] >> 4)
        torque = ((data[4] & 15) << 8) | data[5]
        row.update(mit_position_code=position, mit_velocity_code=velocity, mit_torque_code=torque,
                   mit_status_raw=data[6], mit_is_active=bool(data[6] & 1), mit_has_fault=bool(data[6] & 2),
                   mit_feedback_elapsed_s=elapsed_s)
        if row["mit_decode_ready"]:
            p = decode_symmetric(position, row["mit_position_max_rad"], 16)
            v = decode_symmetric(velocity, row["mit_velocity_max_rad_s"], 12)
            t = decode_symmetric(torque, row["mit_torque_max_nm"], 12)
            with motor.lock:
                if not motor.reference_invalid and math.isnan(motor.mit_zero_position_motor_rad):
                    motor.mit_zero_position_motor_rad = p
                zero_rad = motor.mit_zero_position_motor_rad
            row.update(mit_zero_position_motor_rad=zero_rad, mit_feedback_position_motor_rad=p,
                       mit_feedback_position_rad=motor.encoder_sign * (p - zero_rad) / motor.gear_ratio,
                       mit_feedback_velocity_motor_rad_s=v, mit_feedback_velocity_rad_s=motor.encoder_sign * v
                       / motor.gear_ratio, mit_feedback_torque_motor_nm=t,
                       mit_feedback_torque_nm=motor.torque_sign * t)
    elif is_disable:
        with motor.lock:
            # CF before motion is the sender's startup; CF after motion is exit.
            # Do not resume reads on B1/F0 acknowledgement during preflight.
            startup_pending = not motor.mit_motion_active
            if startup_pending != motor.startup_pending:
                motor.measurement_epoch += 1
            motor.startup_pending = startup_pending
            motor.mit_motion_active = False
        row = new_row(motor, "mit_disable", elapsed_s, wall_time)
        row["can_note"] = ("startup CF observed; queries paused until MIT motion" if startup_pending else
                           "exit CF observed; read-only measurements continue")
    elif motor_reply and length == 8 and opcode in (0xAE, 0xCF):
        row = new_row(motor, "motor_status", elapsed_s, wall_time)
        voltage, current = struct.unpack("<HH", data[1:5])
        row.update(bus_voltage_v=voltage * 0.01, bus_current_a=current * 0.01, bus_current_raw=current,
                   motor_temperature_c=data[5], motor_mode_raw=data[6], motor_fault_code=data[7])
    elif is_origin_request or is_origin_ack or is_reboot:
        with motor.lock:
            # A request begins an epoch. An unsolicited ack also begins one;
            # an ack of the observed request completes that same epoch.
            if not is_origin_ack or not motor.reference_invalid:
                motor.reference_epoch += 1
            motor.reference_invalid = not is_origin_ack
            motor.startup_pending = True
            motor.mit_motion_active = False
            motor.measurement_epoch += 1
            motor.zero_count = None
            motor.mit_zero_position_motor_rad = math.nan
            motor.torque_constant = math.nan
            motor.mit_ranges = None
            motor.mit_config_requested = None
            motor.mit_config_write_pending = False
            motor.mit_config_elapsed_s = math.nan
            motor.retry_mit_at = motor.retry_b0_at = 0.0
        row = new_row(motor, "origin_reset", elapsed_s, wall_time)
        row["can_note"] = ("reboot observed; restart recording" if is_reboot else
                           "B1 acknowledged; observing F0/A3/F1 preflight before resuming queries" if is_origin_ack else
                           "B1 requested; waiting for acknowledgement; measurements paused")
    elif motor_reply and length == READ_LENGTHS[0xA3] and opcode == 0xA3:
        single_count, count = struct.unpack("<Hi", data[1:7])
        with motor.lock:
            preflight_configured = (motor.mit_config_requested is not None and motor.mit_ranges is not None
                                    and not motor.mit_config_write_pending)
            if (motor.zero_count is None and not motor.reference_invalid
                    and (not motor.startup_pending or preflight_configured)):
                # The sender reads A3 before motion; keep this zero rather than
                # the first recorder query, which may arrive after motion starts.
                motor.zero_count = count
            zero_count = motor.zero_count
        row = new_row(motor, "motor_reply", elapsed_s, wall_time)
        row.update(encoder_single_turn_count=single_count, encoder_count=count, zero_count=zero_count,
                   position_elapsed_s=elapsed_s)
        if zero_count is not None and not row["origin_pending"]:
            position = motor.encoder_sign * (count - zero_count) * RADIANS_PER_COUNT / motor.gear_ratio
            row.update(position_unwrapped_rad=position, position_rad=(position + math.pi) % (2 * math.pi) - math.pi)
    elif host_request and length == 1 and opcode in READ_LENGTHS:
        row = new_row(motor, "read_query", elapsed_s, wall_time)
    elif motor_reply and opcode in READ_LENGTHS and length == READ_LENGTHS[opcode]:
        row = new_row(motor, "motor_reply", elapsed_s, wall_time)
    else:
        # Retain other selected-motor traffic and malformed commands for diagnosis.
        row = new_row(motor, "can_frame", elapsed_s, wall_time)
        row["can_note"] = "unclassified opcode/length; raw bytes preserved"
    row.update(mit_can_id=can_id, mit_payload_hex=data.hex().upper(), can_opcode=None if is_target else opcode,
               can_length=length, can_direction="local_tx" if local_tx else "bus_rx")
    return row


def receive_mit(bus, motors, start, stopped, pending):
    routing = {can_id: [motor] for motor in motors for can_id in (
        motor.device_id, 0x100 | motor.device_id, 0x400 | motor.device_id, 0x500 | motor.device_id,
    )}
    routing.update({can_id: motors for can_id in GROUP_IDS})
    drain_deadline = None
    try:
        while True:
            if stopped.is_set():
                if drain_deadline is None:
                    drain_deadline = time.monotonic() + SHUTDOWN_DRAIN_S
                    bus.setblocking(False)
                elif time.monotonic() >= drain_deadline:
                    raise RuntimeError("capture shutdown drain timed out; recording may be incomplete")
            try:
                packet = receive_packet(bus)
            except BlockingIOError:
                if drain_deadline is not None:
                    break
                raise
            except TimeoutError:
                if drain_deadline is not None:
                    break
                continue
            unpacked = unpack_standard(packet.frame)
            reboot_observed = False
            if unpacked is not None:
                can_id = unpacked[0]
                for motor in routing.get(can_id, ()):
                    row = process_frame(packet.frame, {can_id: motor}, start, packet.received_s,
                                        packet.wall_time_s, packet.local_tx)
                    if row is not None:
                        row["socket_dropped_frames"] = packet.dropped_frames
                        if pending.qsize() >= MAX_PENDING_ROWS:
                            raise RuntimeError("CSV writer backlog exceeded limit; recording incomplete")
                        pending.put(row)
                        reboot_observed |= row["record_type"] == "origin_reset" and row["can_note"].startswith("reboot")
            check_overflow(packet, "observer")
            if reboot_observed:
                raise RuntimeError("motor reboot observed; restart recording with new startup zeros")
    except Exception as error:
        pending.put(error)  # Capture failures must terminate recording, not silently lose targets.


def read_query(bus, motor_id, opcode, timeout):
    """Whitelisted read; return only a matching external, standard-frame reply."""
    if opcode not in READ_LENGTHS:
        raise ValueError("only A1/A2/A3/B0/F0/F1 reads are permitted")
    if not 1 <= motor_id <= 254 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("read needs a unicast motor address and a positive finite timeout")
    deadline = time.monotonic() + timeout
    bus.setblocking(False)
    try:
        for _ in range(MAX_QUERY_DRAIN):
            if time.monotonic() >= deadline:
                return None
            check_overflow(receive_packet(bus), "query")
    except BlockingIOError:
        pass
    else:
        # Do not send a new request while old replies still fill the query socket.
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    bus.settimeout(remaining)
    frame = CAN_FRAME.pack(0x100 | motor_id, 1, bytes([opcode]))
    try:
        sent = bus.send(frame)
    except TimeoutError as error:
        raise OSError(f"CAN read query {opcode:02X} transmission timed out") from error
    if sent != CAN_FRAME.size:
        raise OSError("incomplete SocketCAN query send")
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            bus.settimeout(remaining)
            packet = receive_packet(bus)
            check_overflow(packet, "query")
            if packet.received_s > deadline:
                return None
            unpacked = unpack_standard(packet.frame)
            if unpacked is None or packet.local_tx:
                continue
            can_id, length, data = unpacked
            if can_id == motor_id and length == READ_LENGTHS[opcode] and data[0] == opcode:
                return Reply(data, packet.received_s, packet.wall_time_s)
    except TimeoutError:
        return None


def query_mit_read(bus, motor_id, opcode, timeout):
    if opcode not in (0xF0, 0xF1):
        raise ValueError("only one-byte F0/F1 reads are permitted")
    return read_query(bus, motor_id, opcode, timeout)


def collect_motor(bus, motor, start, timeout):
    row = new_row(motor, "feedback", time.monotonic() - start, time.time())
    epoch = row["reference_epoch"]
    measurement_epoch = row["measurement_epoch"]
    if row["origin_pending"] or row["startup_pending"]:
        row["can_note"] = ("B1 acknowledgement pending; no measurement queries sent" if row["origin_pending"] else
                           "sender preflight pending; passive capture only until first MIT target")
        return row

    def read(opcode):
        with motor.lock:
            allowed = (not motor.reference_invalid and not motor.startup_pending
                       and motor.measurement_epoch == measurement_epoch)
        if not allowed:
            return None
        reply = read_query(bus, motor.device_id, opcode, timeout)
        row[f"{opcode:02x}_reply_received"] = reply is not None
        return reply

    ranges, _, zero_rad, *_ = snapshot(motor)
    now = time.monotonic()
    if (ranges is None or math.isnan(zero_rad)) and now >= motor.retry_mit_at:
        motor.retry_mit_at = now + 1.0
        if ranges is None:
            read(0xF0)
        if math.isnan(zero_rad):
            read(0xF1)
    now = time.monotonic()
    if math.isnan(motor.torque_constant) and now >= motor.retry_b0_at:
        motor.retry_b0_at = now + 1.0
        reply = read(0xB0)
        if reply is not None:
            constant = struct.unpack("<f", reply.data[2:6])[0]
            if math.isfinite(constant) and 0.01 <= constant <= 10.0:
                with motor.lock:
                    if (not motor.reference_invalid and not motor.startup_pending
                            and motor.measurement_epoch == measurement_epoch):
                        motor.torque_constant = constant
                print(f"{motor.name}: B0={constant:g} Nm/A", file=sys.stderr)
    for opcode, timestamp_field in ((0xA3, "position_elapsed_s"), (0xA2, "velocity_elapsed_s"),
                                    (0xA1, "current_elapsed_s")):
        with motor.lock:
            if motor.reference_invalid or motor.startup_pending or motor.measurement_epoch != measurement_epoch:
                break
        reply = read(opcode)
        if reply is None:
            continue
        row[timestamp_field] = reply.received_s - start
        if opcode == 0xA3:
            count = struct.unpack("<i", reply.data[3:7])[0]
            with motor.lock:
                if (motor.zero_count is None and not motor.reference_invalid and not motor.startup_pending
                        and motor.measurement_epoch == measurement_epoch):
                    motor.zero_count = count
                    print(f"{motor.name}: startup A3 zero={count} counts", file=sys.stderr)
                zero_count = motor.zero_count
            row.update(encoder_count=count, encoder_single_turn_count=struct.unpack("<H", reply.data[1:3])[0],
                       zero_count=zero_count)
            if zero_count is not None:
                position = motor.encoder_sign * (count - zero_count) * RADIANS_PER_COUNT / motor.gear_ratio
                row.update(position_unwrapped_rad=position, position_rad=(position + math.pi) % (2 * math.pi) - math.pi)
        elif opcode == 0xA2:
            rpm = struct.unpack("<i", reply.data[1:5])[0] * 0.01
            row["velocity_rad_s"] = motor.encoder_sign * rpm * 2 * math.pi / 60 / motor.gear_ratio
        else:
            row["current_a"] = struct.unpack("<i", reply.data[1:5])[0] * 0.001
            row["torque_nm"] = motor.torque_sign * row["current_a"] * motor.torque_constant
    # Refresh metadata only; failed query measurements remain NaN, not cached.
    metadata = new_row(motor, "feedback", time.monotonic() - start, time.time())
    for name in ("elapsed_s", "time_unix_s", "mit_decode_ready", "mit_config_elapsed_s",
                 "mit_zero_position_motor_rad", "mit_position_max_rad", "mit_velocity_max_rad_s", "mit_torque_max_nm",
                 "mit_config_write_pending", "mit_requested_position_max_rad", "mit_requested_velocity_max_rad_s",
                 "mit_requested_torque_max_nm"):
        row[name] = metadata[name]
    with motor.lock:
        changed = (motor.reference_invalid or motor.startup_pending or motor.reference_epoch != epoch
                   or motor.measurement_epoch != measurement_epoch)
        if not changed:
            row.update(zero_count=motor.zero_count, torque_constant_nm_per_a=motor.torque_constant)
    if changed:
        row = new_row(motor, "feedback", time.monotonic() - start, time.time())
        row["can_note"] = "origin/configuration/startup changed during cycle; mixed-state measurements discarded"
    return row


def write_pending(writer, pending, limit=MAX_WRITE_BATCH):
    # A continuously producing receiver must not starve the measurement loop.
    for _ in range(min(limit, pending.qsize())):
        try:
            row = pending.get_nowait()
        except queue.Empty:
            return
        if isinstance(row, Exception):
            raise row
        writer.writerow(row)


def run(args):
    motors = [Motor(name, getattr(args, f"{name}_id"), getattr(args, f"{name}_gear_ratio"),
                    getattr(args, f"{name}_encoder_sign"), getattr(args, f"{name}_torque_sign"))
              for name in args.motors]
    with open(args.output, "x", newline="") as log, \
            socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus, \
            socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as observer:
        for channel in (bus, observer):
            channel.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RX_BUFFER_BYTES)
            channel.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                       b"".join(struct.pack("=II", motor.device_id, CAN_MASK) for motor in motors))
        bus.bind((args.interface,))
        ids = [can_id for motor in motors for can_id in (
            motor.device_id, 0x100 | motor.device_id, 0x400 | motor.device_id, 0x500 | motor.device_id,
        )] + list(GROUP_IDS)
        observer.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                            b"".join(struct.pack("=II", can_id, CAN_MASK) for can_id in ids))
        observer.settimeout(0.05)
        observer.bind((args.interface,))
        writer = csv.DictWriter(log, fieldnames=FIELDS)
        writer.writeheader()
        log.flush()
        print(f"READ QUERIES ONLY on {args.interface}; target {args.rate:g} Hz per motor; "
              f"passive MIT command/F1 capture; CSV: {args.output}", flush=True)
        start = deadline = time.monotonic()
        next_report = start
        stopped, pending = threading.Event(), queue.Queue()
        receiver = threading.Thread(target=receive_mit, args=(observer, motors, start, stopped, pending), daemon=True)
        receiver.start()
        try:
            while not args.duration or time.monotonic() - start < args.duration:
                rows = [collect_motor(bus, motor, start, args.timeout) for motor in motors]
                writer.writerows(rows)
                write_pending(writer, pending)
                now = time.monotonic()
                if now >= next_report:
                    statuses = []
                    for row in rows:
                        ready = row["mit_decode_ready"] and math.isfinite(row["mit_zero_position_motor_rad"])
                        statuses.append(f"{row['motor']}: pos={row['position_unwrapped_rad']:.4f} rad, "
                                        f"vel={row['velocity_rad_s']:.4f} rad/s, torque={row['torque_nm']:.4f} Nm; "
                                        f"MIT ready={ready}; startup_pending={row['startup_pending']}, "
                                        f"origin_pending={row['origin_pending']}")
                    print(" | ".join(statuses), flush=True)
                    log.flush()
                    next_report = now + 1.0
                deadline += 1.0 / args.rate
                if deadline > now:
                    time.sleep(deadline - now)
                else:
                    deadline = now
        finally:
            stopped.set()
            receiver.join(timeout=1.0)
            if receiver.is_alive():
                raise RuntimeError("MIT capture thread did not stop")
            write_pending(writer, pending, limit=pending.qsize())


def main(argv=None):
    args = parse_args(argv)
    def request_stop(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, request_stop)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped recording; no motor control/disable commands sent.")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    sys.exit(main())
