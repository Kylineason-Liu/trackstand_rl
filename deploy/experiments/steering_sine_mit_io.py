#!/usr/bin/env python3
"""50 Hz MIT steering sine with integrated SocketCAN feedback/CSV recording.

Standalone Python stdlib only. One thread owns CAN, commands and motor state;
only CSV writing runs in another thread. The existing two scripts are not used.
Startup preserves the proven CF -> B1 -> F0 read/write/read-back -> A3/F1 checks.
F0 Pos_Max is set to 3 native motor rad; actual velocity/torque ranges are kept.
Position is absolute: verified F1 zero + sign * gear * joint sine. Kp=10,
Kd=.2, requested target velocity/feedforward torque=0 (quantization is logged).
The fixed 50 Hz is the host setpoint rate, not a claim about firmware PD rate.

Default: can0, steering ID 2, sign -1, gear 1, amplitude 1 rad, period 10 s;
F1 replies plus nominal 200 Hz A3/A2/A1 read cycles. --feedback mit omits these
extra motion-time reads. --include-drive adds read-only ID 1 collection; it
never commands, disables, or resets that motor. Stop all other selected-motor
CAN writers, including the old sender/recorder and deployment motor nodes.

Execution automatically creates an exclusive CSV in the current directory.
--preview opens neither CAN nor CSV. B1/F0 change persistent motor settings.
Once CAN is configured, exits attempt CF even after startup failure; no hard real-time or
physical disable guarantee is made. RX drops, writer failure/backlog, observed
interference, steering fault/unexpected MIT exit, or missing F1 beyond the
--feedback-timeout stop motion.
Times are host send/receipt times, not motor acquisition or CAN wire timestamps.
Protocol: local GIM CAN V3.09b0 pp.2, 4-8, 16, 20-22.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass
from datetime import datetime
import math
import queue
import select
import signal
import socket
import struct
import sys
import threading
import time

COMMAND_RATE_HZ = 50.0
COMMAND_DT_S = 1.0 / COMMAND_RATE_HZ
KP_NM_PER_RAD, KD_NM_S_PER_RAD = 10.0, 0.2
POSITION_MAX_RAD = 3.0
CAN_FRAME = struct.Struct("=IB3x8s")
CAN_MASK = 0xC00007FF  # EFF/RTR masked; ERR must not select the error receive list.
SO_RXQ_OVFL = getattr(socket, "SO_RXQ_OVFL", 40)
RADIANS_PER_COUNT = 2 * math.pi / 16384
READ_LENGTHS = {0xA1: 5, 0xA2: 5, 0xA3: 7, 0xB0: 7, 0xF0: 7, 0xF1: 7}
STARTUP_TIMEOUT_S, SHUTDOWN_TIMEOUT_S = 1.0, 0.25
MAX_RX_BATCH, MAX_LOG_ROWS = 64, 20000
FIELDS = (
    "record_type", "time_unix_s", "elapsed_s", "phase", "motor", "motor_id",
    "gear_ratio", "encoder_sign", "torque_sign", "reference_epoch", "zero_count",
    "mit_zero_position_motor_rad", "mit_position_max_rad", "mit_velocity_max_rad_s", "mit_torque_max_nm",
    "mit_can_id", "can_direction", "can_opcode", "can_length", "mit_payload_hex", "socket_dropped_frames",
    "command_index", "requested_position_rad", "command_interval_s", "command_lateness_s", "skipped_command_ticks",
    "mit_position_code", "mit_velocity_code", "mit_kp_code", "mit_kd_code", "mit_torque_code",
    "mit_target_position_motor_rad", "mit_target_position_rad", "mit_target_velocity_motor_rad_s",
    "mit_target_velocity_rad_s", "mit_kp_nm_per_motor_rad", "mit_kd_nm_s_per_motor_rad",
    "mit_feedforward_torque_motor_nm", "mit_feedforward_torque_nm",
    "encoder_count", "encoder_single_turn_count", "position_rad", "position_unwrapped_rad",
    "velocity_rad_s", "current_a", "torque_nm", "torque_constant_nm_per_a",
    "mit_feedback_position_motor_rad", "mit_feedback_position_rad", "mit_feedback_velocity_motor_rad_s",
    "mit_feedback_velocity_rad_s", "mit_feedback_torque_motor_nm", "mit_feedback_torque_nm",
    "mit_status_raw", "mit_is_active", "mit_has_fault", "bus_voltage_v", "bus_current_a", "bus_current_raw",
    "motor_temperature_c", "motor_mode_raw", "motor_fault_code",
    "query_opcode", "query_status", "query_elapsed_s", "query_round_trip_s", "can_note",
)


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Execute (also the default).")
    mode.add_argument("--preview", action="store_true", help="Print 50 Hz targets; no CAN or CSV.")
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--motor-id", type=lambda value: int(value, 0), default=2)
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--torque-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--gear-ratio", type=finite_float, default=1.0)
    parser.add_argument("--amplitude", type=finite_float, default=1.0, help="Peak joint radians, not peak-to-peak.")
    parser.add_argument("--period", type=finite_float, default=1.0, help="Sine period in seconds.")
    parser.add_argument("--duration", type=finite_float, default=0.0, help="Motion seconds; 0 until Ctrl-C.")
    parser.add_argument("--output", default=f"steering_sine_mit_io_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
    parser.add_argument("--feedback", choices=("mit", "extended"), default="extended")
    parser.add_argument("--feedback-rate", type=finite_float, default=200.0, help="Extended read cycles/s per motor.")
    parser.add_argument("--query-timeout", type=finite_float, default=0.002, help="Motion-time read deadline, seconds.")
    parser.add_argument("--feedback-timeout", type=finite_float, default=0.25,
                        help="Stop if steering F1 absent this long.")
    parser.add_argument("--include-drive", action="store_true", help="Read-only drive collection, extended mode only.")
    parser.add_argument("--drive-id", type=lambda value: int(value, 0), default=1)
    parser.add_argument("--drive-gear-ratio", type=finite_float, default=1.0)
    parser.add_argument("--drive-encoder-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--drive-torque-sign", type=int, choices=(-1, 1), default=-1)
    args = parser.parse_args(argv)
    if not all(1 <= mid <= 254 for mid in (args.motor_id, args.drive_id)):
        parser.error("device IDs must be 1..254")
    if min(args.amplitude, args.gear_ratio, args.drive_gear_ratio, args.feedback_rate,
           args.query_timeout, args.feedback_timeout) <= 0 or args.duration < 0:
        parser.error("amplitude, gears, rates and timeouts must be positive; duration must be nonnegative")
    if args.period <= 2 * COMMAND_DT_S:
        parser.error("--period must exceed .04 s (sine frequency below 25 Hz at 50 Hz command rate)")
    if args.amplitude * args.gear_ratio > POSITION_MAX_RAD:
        parser.error("amplitude * gear ratio exceeds the 3-rad native position encoding range")
    if args.include_drive and (args.feedback != "extended" or args.drive_id == args.motor_id):
        parser.error("--include-drive needs extended feedback and a distinct drive ID")
    return args


def decode_symmetric(code, maximum, bits):
    return (2 * code / ((1 << bits) - 1) - 1) * maximum


def encode(value, lower, upper, bits):
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"value {value} outside [{lower}, {upper}]")
    return round((value - lower) / (upper - lower) * ((1 << bits) - 1))


def mit_payload(position_rad, ranges):
    p = encode(position_rad, -ranges[0], ranges[0], 16)
    v = encode(0, -ranges[1], ranges[1], 12)
    kp, kd = encode(KP_NM_PER_RAD, 0, 500, 12), encode(KD_NM_S_PER_RAD, 0, 5, 12)
    t = encode(0, -ranges[2], ranges[2], 12)
    return bytes([p >> 8, p & 255, v >> 4, ((v & 15) << 4) | (kp >> 8), kp & 255,
                  kd >> 4, ((kd & 15) << 4) | (t >> 8), t & 255])


def sine_position(args, elapsed_s, zero_rad):
    return zero_rad + args.sign * args.gear_ratio * args.amplitude * math.sin(
        2 * math.pi * (elapsed_s % args.period) / args.period)


def target_fields(motor, payload):
    pcode = int.from_bytes(payload[:2], "big")
    vcode, kpcode = (payload[2] << 4) | (payload[3] >> 4), ((payload[3] & 15) << 8) | payload[4]
    kdcode, tcode = (payload[5] << 4) | (payload[6] >> 4), ((payload[6] & 15) << 8) | payload[7]
    p, v, t = (decode_symmetric(code, maximum, bits) for code, maximum, bits
               in zip((pcode, vcode, tcode), motor.ranges, (16, 12, 12)))
    return dict(mit_position_code=pcode, mit_velocity_code=vcode, mit_kp_code=kpcode,
                mit_kd_code=kdcode, mit_torque_code=tcode, mit_target_position_motor_rad=p,
                mit_target_position_rad=motor.encoder_sign * (p - motor.zero_rad) / motor.gear_ratio,
                mit_target_velocity_motor_rad_s=v, mit_target_velocity_rad_s=motor.encoder_sign * v / motor.gear_ratio,
                mit_kp_nm_per_motor_rad=kpcode * 500 / 4095, mit_kd_nm_s_per_motor_rad=kdcode * 5 / 4095,
                mit_feedforward_torque_motor_nm=t, mit_feedforward_torque_nm=motor.torque_sign * t)


@dataclass
class Motor:
    name: str
    device_id: int
    gear_ratio: float
    encoder_sign: int
    torque_sign: int
    zero_count: int | None = None
    zero_rad: float = math.nan
    torque_constant: float = math.nan
    ranges: tuple[float, float, float] | None = None
    requested_config: bytes | None = None
    reference_epoch: int = 0
    retry_at: float = 0.0


@dataclass
class PendingRead:
    motor: Motor
    opcode: int
    sent_at: float
    deadline: float


class CsvLog:
    """Bounded asynchronous writer; no CAN or Motor objects enter this thread."""

    def __init__(self, path):
        self.path, self.pending = path, queue.Queue(maxsize=MAX_LOG_ROWS)
        self.stopped, self.error = threading.Event(), None

    def __enter__(self):
        self.file = open(self.path, "x", newline="")  # Refuse overwrite before opening CAN.
        try:
            self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
            self.writer.writeheader()
            self.file.flush()
            self.thread = threading.Thread(target=self._write, daemon=True)
            self.thread.start()
        except BaseException:
            self.file.close()
            raise
        return self

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"CSV writer failed; recording incomplete: {self.error}") from self.error

    def emit(self, row):
        self.check()
        try:
            self.pending.put_nowait(row)
        except queue.Full as error:
            raise RuntimeError("CSV backlog exceeded limit; recording incomplete") from error

    def _write(self):
        next_flush = time.monotonic() + 1.0
        try:
            while not self.stopped.is_set() or not self.pending.empty():
                try:
                    self.writer.writerow(self.pending.get(timeout=0.05))
                except queue.Empty:
                    pass
                if time.monotonic() >= next_flush:
                    self.file.flush()
                    next_flush = time.monotonic() + 1.0
            self.file.flush()
        except Exception as error:
            self.error = error
        finally:
            try:
                self.file.close()
            except Exception as error:
                self.error = self.error or error

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError("CSV writer did not stop; recording may be incomplete")
        self.check()


class CanIO:
    """Sole CAN/state owner, used for both synchronous preflight and async motion."""

    def __init__(self, bus, log, motors):
        self.bus, self.log, self.motors = bus, log, motors
        self.start = time.monotonic()
        self.phase, self.last_f1_at = "startup", math.nan
        self.pending = None
        self.routing = {cid: motor for motor in motors for cid in (
            motor.device_id, 0x100 | motor.device_id, 0x400 | motor.device_id, 0x500 | motor.device_id)}

    def row(self, motor, kind, stamp=None, **values):
        stamp = time.monotonic() if stamp is None else stamp
        row = dict(record_type=kind, time_unix_s=time.time(), elapsed_s=stamp - self.start, phase=self.phase,
                   motor=motor.name, motor_id=motor.device_id, gear_ratio=motor.gear_ratio,
                   encoder_sign=motor.encoder_sign, torque_sign=motor.torque_sign,
                   reference_epoch=motor.reference_epoch, zero_count=motor.zero_count,
                   mit_zero_position_motor_rad=motor.zero_rad, torque_constant_nm_per_a=motor.torque_constant)
        if motor.ranges is not None:
            row.update(zip(("mit_position_max_rad", "mit_velocity_max_rad_s", "mit_torque_max_nm"), motor.ranges))
        row.update(values)
        return row

    def send(self, motor, payload, can_id=None, command_previous_s=None, command_deadline_s=None, **values):
        can_id = (0x100 | motor.device_id) if can_id is None else can_id
        stamp = time.monotonic()
        frame = CAN_FRAME.pack(can_id, len(payload), payload)
        # Send first so a failed logger cannot prevent the shutdown CF attempt.
        if self.bus.send(frame) != CAN_FRAME.size:
            raise OSError("incomplete SocketCAN send")
        if command_deadline_s is not None:
            lateness = max(0.0, stamp - command_deadline_s)
            interval = stamp - command_previous_s if command_previous_s is not None else math.nan
            values.update(command_interval_s=interval, command_lateness_s=lateness,
                          skipped_command_ticks=int(lateness / COMMAND_DT_S))
        self.log.emit(self.row(motor, "mit_target" if can_id & 0x400 else "can_tx", stamp,
                               mit_can_id=can_id, can_direction="local_tx", can_length=len(payload),
                               can_opcode=None if can_id & 0x400 else payload[0],
                               mit_payload_hex=payload.hex().upper(), **values))
        return stamp

    def receive(self):
        frame, ancillary, flags, _ = self.bus.recvmsg(CAN_FRAME.size, socket.CMSG_SPACE(4))
        stamp, wall_time = time.monotonic(), time.time()
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise RuntimeError("truncated CAN frame/metadata; recording incomplete")
        dropped = 0
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == SO_RXQ_OVFL:
                if len(data) != 4:
                    raise RuntimeError("invalid CAN overflow metadata")
                dropped = struct.unpack("=I", data)[0]
        if len(frame) != CAN_FRAME.size:
            raise RuntimeError("invalid CAN frame size")
        cid, length, padded = CAN_FRAME.unpack(frame)
        if cid & ~0x7FF or length > 8:
            if dropped:
                raise RuntimeError(f"CAN socket reports {dropped} dropped frames; recording incomplete")
            return None
        data = padded[:length]
        motors = self.motors if cid in (0, 0xFF, 0x400, 0x4FF) else [self.routing.get(cid)]
        local_tx, result = bool(flags & socket.MSG_DONTROUTE), None
        for motor in motors:
            if motor is None:
                continue
            external_command = (cid != motor.device_id or local_tx or data == b"\xcf"
                                or data == bytes.fromhex("00FF00FF00FF00FF"))
            row = self.row(motor, "can_rx", stamp, time_unix_s=wall_time, mit_can_id=cid,
                           can_direction="local_tx" if local_tx else "bus_rx", can_length=length,
                           can_opcode=None if cid & 0x400 else data[0] if data else None,
                           mit_payload_hex=data.hex().upper(), socket_dropped_frames=dropped)
            fatal = "another selected-motor CAN writer observed; stop other writers" if external_command else ""
            if not external_command and data:
                fatal = self.decode_reply(motor, data, stamp, row)
                result = motor, data, stamp
            self.log.emit(row)
            if dropped:
                raise RuntimeError(f"CAN socket reports {dropped} dropped frames; recording incomplete")
            if fatal:
                raise RuntimeError(fatal)
        if dropped:
            raise RuntimeError(f"CAN socket reports {dropped} dropped frames; recording incomplete")
        return result

    def decode_reply(self, motor, data, stamp, row):
        op = data[0]
        if op == 0xF0 and len(data) == 7:
            ranges = tuple(v * scale for v, scale in zip(struct.unpack("<HHH", data[1:]), (.1, .01, .01)))
            if motor.requested_config is not None and data != motor.requested_config:
                row["can_note"] = "F0 differs from requested bytes; not used for decoding"
                return "F0 changed during motion" if self.phase == "motion" else ""
            if not all(v > 0 for v in ranges):
                return "invalid F0 encoding ranges"
            if self.phase == "motion" and motor.ranges is not None and ranges != motor.ranges:
                return "F0 changed during motion"
            if motor.ranges != ranges:
                motor.zero_rad = math.nan
            motor.ranges = ranges
            row.update(mit_position_max_rad=ranges[0], mit_velocity_max_rad_s=ranges[1], mit_torque_max_nm=ranges[2])
        elif op == 0xF1 and len(data) == 7:
            p, v, t = int.from_bytes(data[1:3], "big"), (data[3] << 4) | (data[4] >> 4), ((data[4] & 15) << 8) | data[5]
            row.update(record_type="mit_feedback", mit_position_code=p, mit_velocity_code=v, mit_torque_code=t,
                       mit_status_raw=data[6], mit_is_active=bool(data[6] & 1), mit_has_fault=bool(data[6] & 2))
            if motor.ranges is not None:
                p, v, t = (decode_symmetric(code, maximum, bits) for code, maximum, bits
                           in zip((p, v, t), motor.ranges, (16, 12, 12)))
                if motor is not self.motors[0] and math.isnan(motor.zero_rad):
                    motor.zero_rad = p
                row.update(mit_zero_position_motor_rad=motor.zero_rad,
                           mit_feedback_position_motor_rad=p,
                           mit_feedback_position_rad=motor.encoder_sign * (p - motor.zero_rad) / motor.gear_ratio,
                           mit_feedback_velocity_motor_rad_s=v, mit_feedback_velocity_rad_s=motor.encoder_sign * v
                           / motor.gear_ratio, mit_feedback_torque_motor_nm=t,
                           mit_feedback_torque_nm=motor.torque_sign * t)
            if motor is self.motors[0] and self.phase == "motion":
                self.last_f1_at = stamp
                if data[6] & 2:
                    return "steering F1 reports a motor system fault"
                if not data[6] & 1:
                    return "steering F1 reports unexpected exit from MIT mode"
        elif op == 0xA3 and len(data) == 7:
            single, count = struct.unpack("<Hi", data[1:])
            if motor is not self.motors[0] and motor.zero_count is None:
                motor.zero_count = count
            p = (motor.encoder_sign * (count - motor.zero_count) * RADIANS_PER_COUNT / motor.gear_ratio
                 if motor.zero_count is not None else math.nan)
            row.update(record_type="feedback", encoder_count=count, encoder_single_turn_count=single,
                       zero_count=motor.zero_count, position_unwrapped_rad=p,
                       position_rad=(p + math.pi) % (2 * math.pi) - math.pi)
        elif op == 0xA2 and len(data) == 5:
            row.update(record_type="feedback", velocity_rad_s=motor.encoder_sign * struct.unpack("<i", data[1:])[0]
                       * .01 * 2 * math.pi / 60 / motor.gear_ratio)
        elif op == 0xA1 and len(data) == 5:
            current = struct.unpack("<i", data[1:])[0] * .001
            row.update(record_type="feedback", current_a=current,
                       torque_nm=motor.torque_sign * current * motor.torque_constant)
        elif op == 0xB0 and len(data) == 7:
            constant = struct.unpack("<f", data[2:6])[0]
            if math.isfinite(constant) and .01 <= constant <= 10:
                motor.torque_constant = constant
            row["torque_constant_nm_per_a"] = motor.torque_constant
        elif op in (0xCF, 0xAE) and len(data) == 8:
            voltage, current = struct.unpack("<HH", data[1:5])
            row.update(record_type="motor_status", bus_voltage_v=voltage * .01, bus_current_a=current * .01,
                       bus_current_raw=current, motor_temperature_c=data[5], motor_mode_raw=data[6],
                       motor_fault_code=data[7])
            if self.phase == "motion" and motor is self.motors[0] and data[7]:
                return "steering status reports a motor fault"
        elif op == 0xB1 and self.phase == "motion":
            return "unexpected origin acknowledgement during motion"
        return ""

    def wait(self, timeout):
        return bool(select.select([self.bus], [], [], max(0.0, timeout))[0])

    def drain(self):
        """Bound work so continuous traffic cannot starve the 50 Hz deadline."""
        for _ in range(MAX_RX_BATCH):
            try:
                result = self.receive()
            except BlockingIOError:
                return True
            if result is not None and self.pending is not None:
                motor, data, stamp = result
                pending = self.pending
                if motor is pending.motor and data[0] == pending.opcode and len(data) == READ_LENGTHS[pending.opcode]:
                    self.finish_query("received" if stamp <= pending.deadline else "late_reply", stamp)
            if self.phase == "motion" and time.monotonic() >= self.next_command:
                return False
        return False

    def request(self, motor, opcode, length=None, payload=None, expected=None, required=True):
        if not self.drain():
            raise RuntimeError("startup receive backlog did not clear")
        sent = self.send(motor, bytes([opcode]) if payload is None else payload)
        deadline = sent + STARTUP_TIMEOUT_S
        length = READ_LENGTHS.get(opcode, length) if length is None else length
        while time.monotonic() < deadline:
            self.log.check()
            if self.wait(min(.05, deadline - time.monotonic())):
                result = self.receive()
                if result is not None:
                    got_motor, data, stamp = result
                    if stamp <= deadline and got_motor is motor and len(data) == length and data[0] == opcode:
                        if expected is None or data == expected:
                            self.log.emit(self.row(motor, "query_result", stamp, query_opcode=opcode,
                                                   query_status="received", query_elapsed_s=sent - self.start,
                                                   query_round_trip_s=stamp - sent))
                            return data
        self.log.emit(self.row(motor, "query_result", query_opcode=opcode, query_status="timed_out",
                               query_elapsed_s=sent - self.start))
        if required:
            raise TimeoutError(f"no valid {opcode:02X} startup reply; no MIT motion sent")
        return None

    def finish_query(self, status, stamp=None):
        pending, self.pending = self.pending, None
        stamp = time.monotonic() if stamp is None else stamp
        self.log.emit(self.row(pending.motor, "query_result", stamp, query_opcode=pending.opcode, query_status=status,
                               query_elapsed_s=pending.sent_at - self.start,
                               query_round_trip_s=stamp - pending.sent_at))

    def shutdown(self):
        self.phase = "shutdown"
        motor = self.motors[0]
        sent = self.send(motor, b"\xcf")
        if self.pending is not None:
            self.finish_query("cancelled")
        deadline, confirmed = sent + SHUTDOWN_TIMEOUT_S, False
        while time.monotonic() < deadline:
            if self.wait(min(.02, deadline - time.monotonic())):
                result = self.receive()
                if result is not None:
                    got_motor, data, _ = result
                    if got_motor is motor and len(data) == 8 and data[0] == 0xCF:
                        confirmed = data[6] == 0
        self.log.emit(self.row(motor, "event", can_note="CF disabled mode acknowledged" if confirmed else
                               "CF sent but disabled mode not confirmed before shutdown deadline"))
        if not confirmed:
            raise RuntimeError("shutdown CF disabled mode not confirmed; check motor state physically")


def startup(io, args):
    motor = io.motors[0]
    disabled = io.request(motor, 0xCF, length=8)
    if disabled[6] or disabled[7]:
        raise RuntimeError("CF did not confirm disabled/fault-free steering; no MIT motion sent")
    motor.reference_epoch += 1
    motor.zero_count, motor.zero_rad, motor.ranges, motor.requested_config = None, math.nan, None, None
    motor.torque_constant = math.nan
    io.request(motor, 0xB1, length=3)
    config = io.request(motor, 0xF0)
    requested = b"\xf0" + struct.pack("<H", round(POSITION_MAX_RAD / .1)) + config[3:]
    motor.requested_config, motor.ranges, motor.zero_rad = requested, None, math.nan
    io.request(motor, 0xF0, payload=requested, expected=requested)
    if io.request(motor, 0xF0) != requested:
        raise RuntimeError("F0 read-back differs from requested ranges; no MIT motion sent")
    encoder = io.request(motor, 0xA3)
    single, count = struct.unpack("<Hi", encoder[1:])
    if abs((single + 8192) % 16384 - 8192) > 2 or abs(count) > 2:
        raise RuntimeError("A3 not near B1 zero; no MIT motion sent")
    motor.zero_count = count  # Use this verified reply, not an earlier unsolicited A3.
    feedback = io.request(motor, 0xF1)
    p = decode_symmetric(int.from_bytes(feedback[1:3], "big"), motor.ranges[0], 16)
    v = decode_symmetric((feedback[3] << 4) | (feedback[4] >> 4), motor.ranges[1], 12)
    if feedback[6] & 3 or abs(p) > 2 * RADIANS_PER_COUNT + 2 * motor.ranges[0] / 65535:
        raise RuntimeError("F1 not inactive/fault-free/near B1 zero; no MIT motion sent")
    if abs(v) > .1 + 2 * motor.ranges[1] / 4095:
        raise RuntimeError("F1 not stationary; no MIT motion sent")
    motor.zero_rad = p  # Same verified reply defines both the sine centre and F1 relative feedback.
    if abs(p) + args.amplitude * args.gear_ratio > motor.ranges[0]:
        raise ValueError("startup position +/- sine exceeds confirmed Pos_Max; no MIT motion sent")
    io.log.emit(io.row(motor, "event", can_note="B1/F0/A3/F1 verified; startup references fixed"))
    if args.feedback == "extended":
        for selected in io.motors:
            io.request(selected, 0xB0, required=False)
            if selected is not motor:
                for op in (0xA3, 0xF0, 0xF1):
                    io.request(selected, op, required=False)


def motion(io, args):
    motor = io.motors[0]
    io.phase = "motion"
    start = io.next_command = next_poll = io.last_f1_at = time.monotonic()
    end = start + args.duration if args.duration else math.inf
    jobs, previous_send, command_index = deque(), math.nan, 0
    while time.monotonic() < end:
        io.log.check()
        empty = io.drain()
        now = time.monotonic()
        if now >= end:
            break
        if now - io.last_f1_at >= args.feedback_timeout:
            raise RuntimeError("steering F1 feedback timeout; stopping MIT motion")
        if now >= io.next_command:
            p = sine_position(args, now - start, motor.zero_rad)
            payload = mit_payload(p, motor.ranges)
            sent = io.send(motor, payload, 0x500 | motor.device_id, command_index=command_index,
                           requested_position_rad=motor.encoder_sign * (p - motor.zero_rad) / motor.gear_ratio,
                           command_previous_s=previous_send, command_deadline_s=io.next_command,
                           **target_fields(motor, payload))
            previous_send, io.next_command = sent, sent + COMMAND_DT_S  # No catch-up command bursts.
            command_index += 1
        if io.pending is not None and now >= io.pending.deadline:
            io.finish_query("timed_out")
        if args.feedback == "extended" and not jobs and io.pending is None and now >= next_poll:
            for selected in io.motors:
                if now >= selected.retry_at:
                    selected.retry_at = now + 1
                    if math.isnan(selected.torque_constant):
                        jobs.append((selected, 0xB0))
                    if selected.ranges is None:
                        jobs.append((selected, 0xF0))
                    if math.isnan(selected.zero_rad):
                        jobs.append((selected, 0xF1))
                jobs.extend((selected, op) for op in (0xA3, 0xA2, 0xA1))
            next_poll = now + 1 / args.feedback_rate
        if empty and io.pending is None and jobs:
            selected, op = jobs.popleft()
            sent = io.send(selected, bytes([op]))
            io.pending = PendingRead(selected, op, sent, sent + args.query_timeout)
        wake = min(io.next_command, end, io.last_f1_at + args.feedback_timeout)
        if io.pending is not None:
            wake = min(wake, io.pending.deadline)
        elif args.feedback == "extended":
            wake = min(wake, next_poll)
        io.wait(0 if not empty or jobs and io.pending is None else wake - time.monotonic())


def execute(args):
    motors = [Motor("steering", args.motor_id, args.gear_ratio, args.sign, args.torque_sign)]
    if args.include_drive:
        motors.append(Motor("drive", args.drive_id, args.drive_gear_ratio,
                            args.drive_encoder_sign, args.drive_torque_sign))
    with CsvLog(args.output) as log, socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
        ids = [cid for motor in motors for cid in (motor.device_id, 0x100 | motor.device_id,
                                                  0x400 | motor.device_id, 0x500 | motor.device_id)]
        ids.extend((0, 0xFF, 0x400, 0x4FF))
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                       b"".join(struct.pack("=II", cid, CAN_MASK) for cid in ids))
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
        bus.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        bus.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
        bus.setblocking(False)
        bus.bind((args.interface,))
        io = CanIO(bus, log, motors)
        print(f"MIT commands: fixed 50 Hz; feedback={args.feedback}; CSV: {args.output}", flush=True)
        try:
            startup(io, args)
            print(f"Startup verified: F0={motors[0].ranges}, native F1 zero={motors[0].zero_rad:.9f} rad", flush=True)
            motion(io, args)
        except BaseException as error:
            try:
                log.emit(io.row(motors[0], "event", can_note=f"Stopping: {type(error).__name__}: {error}"))
            except RuntimeError:
                pass  # The original error remains visible; CF must still be attempted.
            raise
        finally:
            io.shutdown()


def preview(args):
    print("PREVIEW ONLY: 50 Hz; centre=0 and velocity/torque ranges=(45,18) are assumptions. No CAN/CSV.")
    for index in range(math.ceil((args.duration or args.period) * COMMAND_RATE_HZ)):
        t = index * COMMAND_DT_S
        p = sine_position(args, t, 0)
        print(f"t={t:.3f}s p_abs={p:.9f} rad {0x500 | args.motor_id:03X}#{mit_payload(p, (3,45,18)).hex().upper()}")


def main(argv=None):
    args = parse_args(argv)
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        preview(args) if args.preview else execute(args)
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
