#!/usr/bin/env python3
"""Replay Isaac v2 held steering-position and rear-velocity targets, with MIT/CAN logging.

Preview is default; only --execute opens CAN. Requires Python stdlib only.
Reference targets update at 50 Hz, rear host I at 200 Hz, packets at 50/200/400 Hz.
This is an independently supported motor-response test, not a balancing controller.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import select
import signal
import socket
import struct
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
DT, I_DT, KI = .02, .005, 3.3
FRAME = struct.Struct("=IB3x8s")
RAD_PER_COUNT = 2 * math.pi / 16384
READ_LENGTHS = {0xA1: 5, 0xA2: 5, 0xA3: 7, 0xAE: 8, 0xB0: 7, 0xB1: 3,
                0xB3: 5, 0xCD: 5, 0xCF: 8, 0xF0: 7, 0xF1: 7}
TARGET_FIELDS = ("time_s", "raw_steering_action", "raw_drive_action", "scaled_steering_position_rad",
                 "held_steering_position_target_rad", "drive_velocity_target_rad_s")
FIELDS = ("record_type", "time_unix_s", "host_monotonic_s", "elapsed_s", "phase", "motor", "motor_id",
          "can_id", "opcode", "payload_hex", "target_index", "source_time_s", "replay_elapsed_s",
          "raw_steering_action", "raw_drive_action", "scaled_steering_position_rad",
          "held_steering_position_target_rad", "drive_velocity_target_rad_s", "packet_index",
          "packet_interval_s", "lateness_s", "skipped_source_targets", "skipped_packet_slots", "skipped_integral_slots",
          "zero_count", "native_mit_zero_rad", "multi_turn_count",
          "position_rad", "velocity_rad_s", "velocity_age_s", "current_a", "current_torque_nm", "current_age_s",
          "mit_position_native_rad", "mit_velocity_rad_s", "mit_torque_nm", "mit_status", "f1_position_code",
          "encoded_position_rad", "encoded_velocity_rad_s", "effective_velocity_target_rad_s",
          "encoded_kp", "encoded_kd", "encoded_ff_native_nm", "estimated_total_torque_nm",
          "integral_error_rad", "integral_torque_nm", "integral_dt_s", "integral_error_rad_s",
          "encoder_sign", "torque_sign", "torque_constant_nm_per_a", "bus_voltage_v", "bus_current_a",
          "temperature_c", "driver_mode", "fault_code", "note")


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def grid_ticks(seconds):
    count = round(seconds / DT)
    if seconds <= 0 or not math.isclose(count * DT, seconds, rel_tol=0., abs_tol=1e-9):
        raise ValueError("duration must be positive and on the 50 Hz / 0.02 s grid")
    return count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Actually open CAN, command selected motors and record")
    parser.add_argument("--motors", choices=("steering", "drive", "both"), default="both")
    parser.add_argument("--targets", type=Path,
                        default=HERE / "replay_targets/v2_26600_no_dr/policy_targets_50hz.csv")
    parser.add_argument("--duration", type=finite, default=10., help="Replay source prefix from t=0 (seconds)")
    parser.add_argument("--packet-rate", type=int, choices=(50, 200, 400), default=200)
    parser.add_argument("--query-rate", type=finite, default=100., help="A1/A2/A3 queries per second each motor")
    parser.add_argument("--feedback-timeout", type=finite, default=.5)
    parser.add_argument("--status-timeout", type=finite, default=2.)
    parser.add_argument("--limit-tolerance", type=finite, default=1.20,
                        help="Measured speed/current-derived effort abort factor over nominal motor maximum")
    parser.add_argument("--lateness-warning-ms", type=finite, default=20.,
                        help="Record late events; not a timing abort")
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--fixture", choices=("airborne", "supported-loaded"), default="airborne")
    parser.add_argument("--note", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        grid_ticks(args.duration)
    except ValueError as error:
        parser.error(str(error))
    if (not 0 < args.query_rate <= 400 or args.feedback_timeout <= 1 / args.query_rate
            or args.status_timeout <= .2 or args.limit_tolerance < 1 or args.lateness_warning_ms <= 0):
        parser.error("invalid feedback rates/timeouts or measured-limit tolerance")
    if args.output is None:
        args.output = HERE / "logs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.packet_rate}hz_v2_motor_replay.csv"
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv")
    return args


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f32(value):
    return struct.unpack("=f", struct.pack("=f", value))[0]


def load_reference(path, duration):
    """Validate one genuine, contiguous episode; no resampling, ramp or second slew."""
    content = path.read_bytes()
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest.get("task") != "Trackstand-Balance-v2":
        raise ValueError("reference manifest does not declare Trackstand-Balance-v2")
    controller = manifest.get("controller", {})
    expected_controller = {"steering_position_kp_nm_per_rad": 10., "steering_velocity_kd_nm_s_per_rad": .2,
                           "drive_position_kp_nm_per_rad": 0., "drive_velocity_p_nm_s_per_rad": .3,
                           "drive_velocity_ki_nm_per_rad": KI, "integral_rate_hz": 200.}
    if manifest.get("target_rate_hz") != 50 or any(
            controller.get(key) != value for key, value in expected_controller.items()):
        raise ValueError("reference controller gains/rates differ from current v2 replay")
    expected = manifest.get("files", {}).get(path.name)
    expected = expected.get("sha256") if isinstance(expected, dict) else expected
    if expected != hashlib.sha256(content).hexdigest():
        raise ValueError("reference CSV hash does not match its manifest")
    metadata_path = path.parent / "reference_metadata.json"
    metadata_entry = manifest.get("files", {}).get(metadata_path.name, {})
    if metadata_entry.get("sha256") != sha(metadata_path):
        raise ValueError("reference initial-state metadata hash differs from manifest")
    metadata = json.loads(metadata_path.read_text())
    initial = [metadata.get("initial_integral_torque_nm")]
    for key in ("initial_native_joint_position_rad", "initial_joint_velocity_rad_s"):
        initial.extend(metadata.get(key, {}).get(name) for name in ("steering", "drive"))
    if any(value is None or not math.isclose(float(value), 0., abs_tol=1e-9) for value in initial):
        raise ValueError("reference must begin from verified zero joint positions/velocities and zero I")
    reader = csv.DictReader(io.StringIO(content.decode()))
    if not set(TARGET_FIELDS) <= set(reader.fieldnames or ()):
        raise ValueError("missing v2 held-position/velocity reference columns")
    rows = [{key: float(row[key]) for key in TARGET_FIELDS} for row in reader]
    if not rows or any(not math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError("empty or non-finite reference")
    previous = 0.
    for index, row in enumerate(rows):
        if not math.isclose(row["time_s"], index * DT, rel_tol=0., abs_tol=1e-8):
            raise ValueError("reference must start at 0 and remain contiguous on the 50 Hz grid")
        raw_s, raw_d = row["raw_steering_action"], row["raw_drive_action"]
        held = row["held_steering_position_target_rad"]
        desired = max(-f32(1.57), min(f32(1.57), f32(f32(raw_s)*f32(1.57))))
        change = max(-f32(.478), min(f32(.478), f32(desired-previous)))
        expected_held = f32(previous+change)
        if (max(abs(raw_s), abs(raw_d)) > 1 + 1e-7
                or abs(row["scaled_steering_position_rad"] - raw_s * 1.57) > 1e-5
                or abs(row["drive_velocity_target_rad_s"] - raw_d * 61.6) > 1e-5
                or abs(held) > 1.57 + 1e-6 or abs(held-expected_held) > 1e-6):
            raise ValueError("reference violates bounded v2 scaling, steering clip or 50 Hz slew")
        previous = held
    count = grid_ticks(duration)
    if count > len(rows):
        raise ValueError("requested prefix exceeds the recorded episode; no padding or extrapolation")
    return rows[:count], manifest


def encode(value, maximum, bits, unsigned=False):
    lower = 0. if unsigned else -maximum
    if not math.isfinite(value) or not lower <= value <= maximum:
        raise ValueError(f"MIT value {value:g} outside encoding range [{lower:g},{maximum:g}]")
    return round((value - lower) / (maximum - lower) * ((1 << bits) - 1))


def decode(code, maximum, bits):
    return (2 * code / ((1 << bits) - 1) - 1) * maximum


def packet(p, v, kp, kd, ff, ranges):
    pc, vc, tc = (encode(value, maximum, bits) for value, maximum, bits in
                  zip((p, v, ff), ranges, (16, 12, 12)))
    kc, dc = encode(kp, 500., 12, True), encode(kd, 5., 12, True)
    data = bytes((pc >> 8, pc & 255, vc >> 4, ((vc & 15) << 4) | (kc >> 8), kc & 255,
                  dc >> 4, ((dc & 15) << 4) | (tc >> 8), tc & 255))
    return data, (decode(pc, ranges[0], 16), decode(vc, ranges[1], 12), kc * 500 / 4095,
                  dc * 5 / 4095, decode(tc, ranges[2], 12))


class Integral:
    """Same trapezoid/anti-windup structure as v2; real elapsed time and cached A2."""
    def __init__(self):
        self.error_rad = self.torque_nm = self.previous_error = 0.
        self.previous_time = None
        self.limited_updates = 0

    def update(self, error, now, kd, limit):
        dt = now - self.previous_time if self.previous_time is not None else 0.
        increment = .5 * (error + self.previous_error) * dt if self.previous_time is not None else 0.
        proposed = KI * (self.error_rad + increment)
        candidate = max(-limit, min(limit, proposed))
        delta, total = candidate - self.torque_nm, kd * error + candidate
        if (total > limit and delta > 0) or (total < -limit and delta < 0):
            candidate = self.torque_nm
        self.limited_updates += int(candidate != proposed)
        self.torque_nm, self.error_rad = candidate, candidate / KI
        self.previous_error, self.previous_time = error, now
        return dt


@dataclass
class Motor:
    name: str
    device_id: int
    speed_limit: float
    torque_limit: float
    sign: int = -1
    zero_count: int | None = None
    last_count: int | None = None
    travel_count: int = 0
    zero_rad: float = math.nan
    position: float = math.nan
    velocity: float = math.nan
    current: float = math.nan
    kt: float = math.nan
    velocity_at: float = -math.inf
    current_at: float = -math.inf
    status_at: float = -math.inf
    f1_at: float = -math.inf
    first_packet_at: float | None = None
    f1_status: int = 0
    ranges: tuple | None = None
    settings: dict = field(default_factory=dict)
    applied_settings: dict = field(default_factory=dict)
    restore: list = field(default_factory=list)
    origin: dict = field(default_factory=dict)
    disabled_confirmed: bool = False
    packets: int = 0
    last_packet_at: float | None = None


class CsvLog:
    def __init__(self, path):
        self.file = path.open("x", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.queue, self.error = queue.Queue(maxsize=20000), None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    self.writer.writerow(self.queue.get(timeout=.05))
                except queue.Empty:
                    pass
            self.file.flush()
        except Exception as error:
            self.error = error
        finally:
            self.file.close()

    def check(self):
        if self.error:
            raise RuntimeError(f"CSV recording failed: {self.error}")

    def emit(self, row):
        self.check()
        try:
            self.queue.put_nowait(row)
        except queue.Full as error:
            raise RuntimeError("CSV recording backlog; log incomplete") from error

    def close(self):
        self.stopping.set()
        self.thread.join(3.)
        if self.thread.is_alive():
            raise RuntimeError("CSV writer did not stop")
        self.check()


class ReplayIO:
    def __init__(self, args, bus, log, motors, stopped=lambda: False, clock=time):
        self.args, self.bus, self.log, self.motors, self.clock, self.stopped = args, bus, log, motors, clock, stopped
        self.started = clock.monotonic()
        self.phase, self.cleaning, self.motion_start = "startup", False, None
        self.target, self.target_index = {}, -1
        self.integral = Integral()
        self.next_control = math.inf
        self.late_events = 0
        self.skipped_targets = self.skipped_packets = self.skipped_integrals = 0

    def emit(self, motor, kind, now=None, **fields):
        now = self.clock.monotonic() if now is None else now
        row = dict(record_type=kind, time_unix_s=self.clock.time(), host_monotonic_s=now,
                   elapsed_s=now-self.started, phase=self.phase, motor=motor.name, motor_id=motor.device_id,
                   zero_count=motor.zero_count, native_mit_zero_rad=motor.zero_rad, position_rad=motor.position,
                   velocity_rad_s=motor.velocity, velocity_age_s=now-motor.velocity_at, current_a=motor.current,
                   current_age_s=now-motor.current_at, current_torque_nm=motor.sign*motor.current*motor.kt,
                   torque_constant_nm_per_a=motor.kt, encoder_sign=motor.sign, torque_sign=motor.sign,
                   integral_error_rad=self.integral.error_rad, integral_torque_nm=self.integral.torque_nm)
        if self.motion_start is not None:
            row.update(target_index=self.target_index, source_time_s=self.target.get("time_s"),
                       replay_elapsed_s=now-self.motion_start,
                       **{key: value for key, value in self.target.items() if key != "time_s"})
        row.update(fields)
        try:
            self.log.emit(row)
        except Exception:
            if not self.cleaning:
                raise

    def send(self, motor, payload, mit=False, **fields):
        can_id = (0x500 if mit else 0x100) | motor.device_id
        deadline = self.clock.monotonic()+.020
        while True:
            try:
                if self.bus.send(FRAME.pack(can_id, len(payload), payload)) != FRAME.size:
                    raise OSError("incomplete SocketCAN send")
                break
            except OSError as error:
                if error.errno not in (errno.ENOBUFS, errno.EAGAIN) or self.clock.monotonic() >= deadline:
                    raise
                self.clock.sleep(.001)
        stamp = self.clock.monotonic()
        if mit:
            motor.disabled_confirmed = False
            interval = stamp-motor.last_packet_at if motor.last_packet_at is not None else math.nan
            fields.update(packet_index=motor.packets, packet_interval_s=interval)
            motor.packets += 1
            motor.last_packet_at = stamp
            if motor.first_packet_at is None:
                motor.first_packet_at = stamp
        self.emit(motor, "command" if mit else "tx", stamp, can_id=f"{can_id:03X}",
                  opcode="MIT" if mit else f"{payload[0]:02X}", payload_hex=payload.hex().upper(), **fields)
        return stamp

    def receive(self):
        frame, ancillary, flags, _ = self.bus.recvmsg(FRAME.size, socket.CMSG_SPACE(4))
        now = self.clock.monotonic()
        if len(frame) != FRAME.size or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise RuntimeError("truncated CAN frame/metadata")
        for level, kind, value in ancillary:
            if level == socket.SOL_SOCKET and kind == getattr(socket, "SO_RXQ_OVFL", 40):
                if len(value) != 4 or struct.unpack("=I", value)[0]:
                    raise RuntimeError("CAN RX overflow; recording incomplete")
        cid, length, padded = FRAME.unpack(frame)
        data = padded[:length]
        motor = next((item for item in self.motors if item.device_id == cid), None)
        if cid & ~0x7ff or length > 8:
            raise RuntimeError(f"CAN error/nonstandard frame {cid:08X}")
        if motor is None or flags & socket.MSG_DONTROUTE:
            raise RuntimeError("another selected-motor or broadcast CAN writer observed")
        if not data or READ_LENGTHS.get(data[0]) != length:
            raise RuntimeError(f"unexpected motor reply {data.hex()}")
        values, fault, op = {}, None, data[0]
        if op == 0xA1:
            motor.current = struct.unpack("<i", data[1:])[0]*.001
            motor.current_at = now
            if math.isfinite(motor.kt) and abs(motor.current*motor.kt) > motor.torque_limit*self.args.limit_tolerance:
                fault = "measured current-derived torque exceeds motor limit plus tolerance"
        elif op == 0xA2:
            motor.velocity = motor.sign*struct.unpack("<i", data[1:])[0]*.01*2*math.pi/60
            motor.velocity_at = now
            if abs(motor.velocity) > motor.speed_limit*self.args.limit_tolerance:
                fault = "measured motor speed exceeds limit plus tolerance"
        elif op == 0xA3:
            single, count = struct.unpack("<Hi", data[1:])
            if motor.last_count is None:
                motor.zero_count, motor.travel_count = count, 0
            else:
                motor.travel_count += (count-motor.last_count+2**31) % 2**32-2**31
            motor.last_count = count
            motor.position = motor.sign*motor.travel_count*RAD_PER_COUNT
            values["multi_turn_count"] = count
        elif op == 0xB0:
            motor.kt = struct.unpack("<f", data[2:6])[0]
            if not math.isfinite(motor.kt) or motor.kt <= 0:
                fault = "invalid queried B0 torque constant"
        elif op == 0xF0:
            ranges = tuple(value*scale for value, scale in zip(struct.unpack("<HHH", data[1:]), (.1, .01, .01)))
            if min(ranges) <= 0:
                fault = "invalid F0 encoding ranges"
            elif self.phase == "motion" and ranges != motor.ranges:
                fault = "F0 encoding ranges changed during replay"
            else:
                motor.ranges = ranges
        elif op == 0xF1 and motor.ranges is not None:
            pc, vc, tc = (data[1] << 8) | data[2], (data[3] << 4) | (data[4] >> 4), ((data[4] & 15) << 8) | data[5]
            motor.f1_at, motor.f1_status = now, data[6]
            values.update(mit_position_native_rad=decode(pc, motor.ranges[0], 16),
                          mit_velocity_rad_s=motor.sign*decode(vc, motor.ranges[1], 12),
                          mit_torque_nm=motor.sign*decode(tc, motor.ranges[2], 12),
                          mit_status=data[6], f1_position_code=pc)
            if data[6] & 2:
                fault = "F1 reports system fault"
            elif abs(values["mit_velocity_rad_s"]) > motor.speed_limit*self.args.limit_tolerance:
                fault = "MIT speed exceeds motor limit plus tolerance"
            elif abs(values["mit_torque_nm"]) > motor.torque_limit*self.args.limit_tolerance:
                fault = "MIT torque exceeds motor limit plus tolerance"
            elif (self.phase == "motion" and motor.first_packet_at is not None
                  and now-motor.first_packet_at > .1 and not data[6] & 1):
                fault = "motor unexpectedly exited MIT mode"
        elif op in (0xAE, 0xCF):
            volts, amps = struct.unpack("<HH", data[1:5])
            motor.status_at = now
            values.update(bus_voltage_v=volts*.01, bus_current_a=amps*.01,
                          temperature_c=data[5], driver_mode=data[6], fault_code=data[7])
            if data[7]:
                fault = f"motor driver fault code {data[7]}"
        elif op == 0xB1 and self.phase == "motion":
            fault = "unexpected origin acknowledgement during replay"
        self.emit(motor, "rx", now, can_id=f"{cid:03X}", opcode=f"{op:02X}", payload_hex=data.hex().upper(),
                  note=fault or "", **values)
        if fault and not self.cleaning:
            raise RuntimeError(fault)
        return motor, data

    def drain(self):
        for _ in range(64):
            if self.phase == "motion" and self.clock.monotonic() >= self.next_control:
                break
            try:
                self.receive()
            except BlockingIOError:
                break

    def request(self, motor, opcode, payload=None, timeout=1.):
        self.drain()
        self.send(motor, bytes([opcode]) if payload is None else payload)
        deadline = self.clock.monotonic()+timeout
        while self.clock.monotonic() < deadline:
            if self.stopped() and not self.cleaning:
                raise KeyboardInterrupt
            try:
                selected, data = self.receive()
                if selected is motor and data[0] == opcode:
                    return data
            except BlockingIOError:
                select.select([self.bus], [], [], max(0., min(.02, deadline-self.clock.monotonic())))
        raise TimeoutError(f"{motor.name} no {opcode:02X} reply within {timeout:g} s")

    def disable(self, motor):
        motor.disabled_confirmed = False
        deadline = self.clock.monotonic()+1.
        while self.clock.monotonic() < deadline:
            try:
                reply = self.request(motor, 0xCF, timeout=min(.2, deadline-self.clock.monotonic()))
                if reply[6] == 0:
                    motor.disabled_confirmed = True
                    return
                if reply[6] == 5 and reply[7] == 0:
                    status = self.request(motor, 0xF1, timeout=min(.2, max(.01, deadline-self.clock.monotonic())))
                    if status[6] & 3 == 0:
                        motor.disabled_confirmed = True
                        return
            except TimeoutError:
                pass
        raise RuntimeError(f"{motor.name} disable not confirmed")

    def setting(self, motor, wanted):
        if not motor.disabled_confirmed:
            raise RuntimeError("configuration write requires confirmed disabled motor")
        answer = self.request(motor, wanted[0], wanted)
        reread = self.request(motor, wanted[0])
        if answer != wanted or reread != wanted:
            raise RuntimeError(f"{motor.name} {wanted[0]:02X} readback mismatch")
        motor.applied_settings[f"{wanted[0]:02X}"] = wanted.hex().upper()

    def feedback_guard(self):
        now = self.clock.monotonic()
        for motor in self.motors:
            if now-min(motor.velocity_at, motor.current_at) > self.args.feedback_timeout:
                raise RuntimeError(f"{motor.name} A1/A2 feedback unavailable/stale")
            if now-motor.status_at > self.args.status_timeout:
                raise RuntimeError(f"{motor.name} driver status unavailable/stale")
            if (motor.first_packet_at is not None and now-motor.first_packet_at > self.args.feedback_timeout
                    and now-motor.f1_at > self.args.feedback_timeout):
                raise RuntimeError(f"{motor.name} MIT feedback unavailable/stale")


def steering_packet(motor, target):
    data, decoded = packet(motor.zero_rad + motor.sign*target, 0., 10., .2, 0., motor.ranges)
    p, v, kp, kd, ff = decoded
    return data, dict(encoded_position_rad=motor.sign*(p-motor.zero_rad),
                      encoded_velocity_rad_s=motor.sign*v, encoded_kp=kp, encoded_kd=kd,
                      encoded_ff_native_nm=ff)


def drive_packet(motor, target, integral):
    kd = round(.3*4095/5)*5/4095
    ff = decode(encode(motor.sign*integral.torque_nm, motor.ranges[2], 12), motor.ranges[2], 12)
    residual = ff-motor.sign*integral.torque_nm
    data, decoded = packet(0., motor.sign*target-residual/kd, 0., .3, motor.sign*integral.torque_nm, motor.ranges)
    p, v, kp, kd, ff = decoded
    effective = motor.sign*v+motor.sign*residual/kd
    estimate = kd*(effective-motor.velocity)+integral.torque_nm
    return data, dict(encoded_position_rad=p, encoded_velocity_rad_s=motor.sign*v,
                      effective_velocity_target_rad_s=effective, encoded_kp=kp, encoded_kd=kd,
                      encoded_ff_native_nm=ff, estimated_total_torque_nm=estimate)


def setup(io, rows, persist):
    for motor in io.motors:
        io.disable(motor)
        for op in (0xF0, 0xB0, 0xB3, 0xCD):
            motor.settings[f"{op:02X}"] = io.request(motor, op).hex().upper()
    persist()  # Durable originals before ANY persistent write, including B1.
    for motor in io.motors:
        for op in (0xA3, 0xA2, 0xA1, 0xAE):
            io.request(motor, op)
        if abs(motor.velocity) > .5:
            raise RuntimeError(f"{motor.name} must be stationary for initial references")
        if motor.name == "steering":
            motor.origin.update(method="B1 exactly once; A3/F1 bracketed reference", attempted=True)
            persist()
            ack = io.request(motor, 0xB1)
            motor.origin.update(acknowledged=True, acknowledgement_hex=ack.hex().upper())
            motor.last_count = motor.zero_count = None
            motor.travel_count = 0
            a3 = io.request(motor, 0xA3)
            count = struct.unpack("<i", a3[3:7])[0]
            single = struct.unpack("<H", a3[1:3])[0]
            if abs(count) > 2 or single >= 16384 or abs((single+8192) % 16384-8192) > 2:
                raise RuntimeError("steering A3 not near new B1 origin")
            status = io.request(motor, 0xF1)
            code = int.from_bytes(status[1:3], "big")
            native = decode(code, motor.ranges[0], 16)
            tolerance = 2*RAD_PER_COUNT+2*motor.ranges[0]/65535
            if code in (0, 65535) or status[6] & 3 or abs(native-count*RAD_PER_COUNT) > tolerance:
                raise RuntimeError("steering F1 clipped/active/faulted or inconsistent with fresh A3 origin")
            after = io.request(motor, 0xA3)
            if abs(struct.unpack("<i", after[3:7])[0]-count) > 2:
                raise RuntimeError("steering moved while capturing A3/F1 zero")
            motor.zero_rad = native
            motor.origin.update(verified=True, zero_count=motor.zero_count, native_mit_zero_rad=native)
            io.emit(motor, "origin", note="B1 acknowledged once; bracketed A3/F1 fixed steering zero verified")
            for row in rows:
                steering_packet(motor, row["held_steering_position_target_rad"])
        else:
            original = bytes.fromhex(motor.settings["F0"])
            peak = max(abs(row["drive_velocity_target_rad_s"]) for row in rows)
            kd = round(.3*4095/5)*5/4095
            velocity_max = math.ceil(max(3., 1.3*peak, peak+motor.ranges[2]/4095/kd+.01)*100)/100
            wanted = original[:3]+struct.pack("<H", round(velocity_max*100))+original[5:]
            if wanted != original:
                motor.restore.append(original)
                persist()
                io.setting(motor, wanted)
            motor.last_count = motor.zero_count = None
            motor.travel_count = 0
            io.request(motor, 0xA3)
            motor.origin.update(method="first fresh A3 software multi-turn zero; NO B1",
                                verified=True, zero_count=motor.zero_count)
            io.emit(motor, "origin", note="Rear software A3 zero captured after configuration; no B1 sent")
            io.request(motor, 0xF1)
        for op in (0xA2, 0xA1, 0xAE):
            io.request(motor, op)
    # Watchdogs last: no fsync/storage wait between enabling and motion.
    wanted_watchdogs = []
    for motor in io.motors:
        original = bytes.fromhex(motor.settings["CD"])
        if original[1] not in (0, 1):
            raise ValueError("invalid original watchdog setting")
        wanted = b"\xcd\x01"+struct.pack("<H", 500)+original[4:]
        if wanted != original:
            motor.restore.append(original)
            wanted_watchdogs.append((motor, wanted))
    persist()
    # Refresh all feedback after potentially slow durable writes. Motion starts
    # immediately after watchdog setup, without an intervening filesystem wait.
    for motor in io.motors:
        for op in (0xA2, 0xA1, 0xAE):
            io.request(motor, op)
    for motor, wanted in wanted_watchdogs:
        io.setting(motor, wanted)


def slot_index(start, period, now):
    # Subtracting a large monotonic epoch can put an exact 20 ms boundary a few
    # ulps below its grid point. Correct only floating-point representation
    # error (about 8 ns at an epoch of 1e7 s), not scheduler jitter or lateness.
    tolerance = max(1e-9, 4*max(math.ulp(now), math.ulp(start))/period)
    return math.floor((now-start)/period + tolerance)


def advance_slot(start, period, expected_slot, now):
    """Return the next anchored slot and count misses; never request a burst."""
    current = max(expected_slot, slot_index(start, period, now))
    return current+1, current-expected_slot


def motion(io, rows):
    io.phase = "motion"
    io.motion_start = io.clock.monotonic()
    start, end = io.motion_start, io.motion_start+len(rows)*DT
    packet_dt = 1/io.args.packet_rate
    integral_slot = packet_slot = 0
    next_integral = next_packet = start
    due = {(m.device_id, op): start for m in io.motors for op in (0xA1, 0xA2, 0xA3, 0xAE, 0xF0)}
    periods = {0xA1: 1/io.args.query_rate, 0xA2: 1/io.args.query_rate, 0xA3: 1/io.args.query_rate,
               0xAE: .2, 0xF0: 1.}
    drive = next((m for m in io.motors if m.name == "drive"), None)
    while io.clock.monotonic() < end:
        if io.stopped():
            raise KeyboardInterrupt
        io.log.check()
        now = io.clock.monotonic()
        index = min(len(rows)-1, slot_index(start, DT, now))
        skipped_targets = max(0, index-io.target_index-1)
        io.skipped_targets += skipped_targets
        io.target_index, io.target = index, rows[index]
        io.feedback_guard()
        if now >= next_integral:
            integral_slot, skipped_i = advance_slot(start, I_DT, integral_slot, now)
            io.skipped_integrals += skipped_i
            if drive is not None:
                kd = round(.3*4095/5)*5/4095
                limit = min(drive.torque_limit-kd*drive.ranges[1]/4095, drive.ranges[2])
                error = io.target["drive_velocity_target_rad_s"]-drive.velocity
                dt = io.integral.update(error, now, kd, limit)
                io.emit(drive, "integral_update", now, integral_dt_s=dt, integral_error_rad_s=error,
                        skipped_integral_slots=skipped_i, skipped_source_targets=skipped_targets)
            next_integral = start+integral_slot*I_DT
        if now >= next_packet:
            lateness = max(0., now-next_packet)
            packet_slot, skipped_p = advance_slot(start, packet_dt, packet_slot, now)
            io.skipped_packets += skipped_p
            if lateness > io.args.lateness_warning_ms*.001:
                io.late_events += 1
            for motor in io.motors:
                data, fields = (steering_packet(motor, io.target["held_steering_position_target_rad"])
                                if motor.name == "steering" else
                                drive_packet(motor, io.target["drive_velocity_target_rad_s"], io.integral))
                io.send(motor, data, True, lateness_s=lateness, skipped_packet_slots=skipped_p,
                        skipped_source_targets=skipped_targets, **fields)
            next_packet = start+packet_slot*packet_dt
        io.next_control = min(next_integral, next_packet, end)
        io.drain()
        jobs = [(due[motor.device_id, op], motor, op) for motor in io.motors for op in periods]
        query_due, motor, op = min(jobs, key=lambda item: item[0])
        now = io.clock.monotonic()
        if query_due <= now < io.next_control:
            io.send(motor, bytes([op]))  # One query, then receive again before the next.
            slot = slot_index(start, periods[op], io.clock.monotonic())+1
            due[motor.device_id, op] = start+slot*periods[op]
        deadline = min(io.next_control, *due.values())
        select.select([io.bus], [], [], max(0., min(.005, deadline-io.clock.monotonic())))


def cleanup(io):
    io.cleaning, io.phase = True, "shutdown"
    failures = []
    # Request both motors off before waiting for either acknowledgement or
    # restoring persistent settings. A failure on one must not skip the other.
    for motor in io.motors:
        try:
            io.send(motor, b"\xcf")
        except Exception as error:
            failures.append(f"{motor.name} initial CF: {error}")
    for motor in io.motors:
        try:
            io.disable(motor)
        except Exception as error:
            failures.append(f"{motor.name} disable: {error}")
            continue
    for motor in io.motors:
        if not motor.disabled_confirmed:
            continue
        for original in reversed(motor.restore.copy()):
            try:
                io.setting(motor, original)
                motor.restore.remove(original)
            except Exception as error:
                failures.append(f"{motor.name} restore {original[0]:02X}: {error}")
    if failures:
        raise RuntimeError("; ".join(failures))


def open_bus(interface, motors):
    bus = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        ids = [cid for motor in motors for cid in
               (motor.device_id, 0x100 | motor.device_id, 0x400 | motor.device_id, 0x500 | motor.device_id)]
        ids.extend((0, 0xff, 0x400, 0x4ff))
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                       b"".join(struct.pack("=II", cid, 0xC00007FF) for cid in ids))
        bus.setsockopt(socket.SOL_CAN_RAW, getattr(socket, "CAN_RAW_ERR_FILTER", 2), struct.pack("=I", 0x1fffffff))
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
        bus.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
        bus.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_RXQ_OVFL", 40), 1)
        bus.setblocking(False)
        bus.bind((interface,))
        return bus
    except BaseException:
        bus.close()
        raise


def json_ready(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex().upper()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def execute(args, rows, manifest):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(".json")
    reference_path = args.output.with_name(args.output.stem+"_reference.csv")
    temporary_path = metadata_path.with_suffix(".json.tmp")
    if any(path.exists() for path in (args.output, metadata_path, reference_path, temporary_path)):
        raise FileExistsError("experiment CSV/JSON/reference already exists")
    with reference_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TARGET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    motors = []
    if args.motors in ("steering", "both"):
        motors.append(Motor("steering", 2, 23.9, 5.6))
    if args.motors in ("drive", "both"):
        motors.append(Motor("drive", 1, 61.6, 6.28))
    metadata = dict(schema="v2_motor_replay_v1", completed=False, disabled_and_restored=False,
                    arguments=json_ready(vars(args)), source_manifest=manifest, targets_sha256=sha(args.targets),
                    script_sha256=sha(__file__), target_rate_hz=50, integral_rate_hz=200,
                    packet_rate_hz=args.packet_rate, desired_velocity_p_nm_s_per_rad=.3,
                    integral_gain_nm_per_rad=KI, motion_duration_s=None,
                    reference_semantics="held steering POSITION and rear VELOCITY targets, source prefix from reset",
                    timestamp_semantics="host send-call/receipt monotonic timestamps; not wire/acquisition timestamps",
                    torque_semantics="A1 Q-axis current times B0 Kt or encoded F1; not independent shaft torque",
                    timing="anchored 50 Hz source / 200 Hz I / packet slots; missed slots logged, no catch-up bursts",
                    post_pd_torque_clamp="Native MIT post-PD clamp not established; "
                                        "measured guard is not IdealPD clipping")
    io_owner = bus = log = None
    stopped = threading.Event()
    old_handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}

    def persist():
        metadata["motors"] = {motor.name: dict(device_id=motor.device_id, sign=motor.sign, gear_ratio=1.,
                                             settings=motor.settings, restore_payloads=motor.restore,
                                             applied_settings=motor.applied_settings,
                                             ranges=motor.ranges, origin=motor.origin,
                                             initial_multi_turn_count=motor.zero_count,
                                             native_mit_zero_rad=motor.zero_rad,
                                             torque_constant_nm_per_a=motor.kt,
                                             packets=motor.packets, disabled_confirmed=motor.disabled_confirmed)
                              for motor in motors}
        with temporary_path.open("w") as stream:
            json.dump(json_ready(metadata), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, metadata_path)
        directory = os.open(metadata_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    try:
        persist()
        log = CsvLog(args.output)
        bus = open_bus(args.interface, motors)
        io_owner = ReplayIO(args, bus, log, motors, stopped.is_set)
        try:
            setup(io_owner, rows, persist)
            motion(io_owner, rows)
            metadata["completed"] = True
        except BaseException as error:
            metadata["execution_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            if io_owner.motion_start is not None:
                metadata["motion_duration_s"] = time.monotonic()-io_owner.motion_start
                metadata["lateness_warning_events"] = io_owner.late_events
                metadata["final_integral_torque_nm"] = io_owner.integral.torque_nm
                metadata["integral_limited_updates"] = io_owner.integral.limited_updates
                metadata["skipped_source_targets"] = io_owner.skipped_targets
                metadata["skipped_packet_slots"] = io_owner.skipped_packets
                metadata["skipped_integral_slots"] = io_owner.skipped_integrals
            try:
                cleanup(io_owner)
                metadata["disabled_and_restored"] = True
            except BaseException as error:
                metadata["cleanup_error"] = f"{type(error).__name__}: {error}"
                if "execution_error" not in metadata:
                    raise
    except BaseException as error:
        metadata.update(completed=False, error=f"{type(error).__name__}: {error}")
        print(f"STOPPED: {metadata['error']}; partial logs retained", file=sys.stderr)
    finally:
        if bus is not None:
            try:
                bus.close()
            except Exception as error:
                metadata.update(completed=False, close_error=str(error))
        if log is not None:
            try:
                log.close()
            except Exception as error:
                metadata.update(completed=False, log_error=str(error))
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        persist()
    print(f"CSV: {args.output}; completed={metadata['completed']}; "
          f"disable/restore={metadata['disabled_and_restored']}")
    return 0 if metadata["completed"] and metadata["disabled_and_restored"] else 1


def main(argv=None):
    args = parse_args(argv)
    rows, manifest = load_reference(args.targets, args.duration)
    print(f"V2 reference: {len(rows)} held 50 Hz targets, source t = 0..{args.duration:g} s; {args.motors}")
    print(f"MIT packets {args.packet_rate} Hz; rear I 200 Hz independently; "
          f"queried A1/A2/A3 {args.query_rate:g} Hz each")
    for name, key, unit in (("steering", "held_steering_position_target_rad", "rad"),
                            ("rear", "drive_velocity_target_rad_s", "rad/s")):
        print(f"{name} target [{min(row[key] for row in rows):.9g}, {max(row[key] for row in rows):.9g}] {unit}")
    if not args.execute:
        print("PREVIEW: no CAN or output files; add --execute for the independently supported motor test")
        return 0
    print("Independent frame support, tested wheels free, steering initially STRAIGHT; all other CAN writers stopped.")
    print("Selected steering uses persistent B1 once; rear uses software A3 origin only. F0/CD originals restored.")
    return execute(args, rows, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
