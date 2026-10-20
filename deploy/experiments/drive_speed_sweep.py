#!/usr/bin/env python3
"""Standalone GIM rear-wheel C1 speed sweep and feedback CSV (Python stdlib).

PREVIEW is the default: add --execute to touch CAN and create a CSV. Secure the
bike with the rear wheel airborne; stop ALL other selected-motor CAN writers,
including deployment nodes. No ROS, python-can, or repository imports needed.

Each speed has ramp-up -> settle -> hold -> ramp-to-zero -> zero rest phases.
Default joint speeds: +/-1,2,4,6,8 rad/s; one repeat; hold 10 s, settle 5 s.
The hold label means a scheduled plateau, NOT proof that the wheel is steady.
Feedback/at_target columns let analysis reject poorly tracked plateaus.
The zero-speed rest phases still use enabled C1 speed control, not free coasting.

Defaults follow deploy configuration: device 1, host 0x101, feedback 0x001,
encoder/command/torque signs -1, gear ratio 1 (already output-side feedback).
Verify physical C1 direction at LOW speed first. No extra 8:1 scaling is done.
C1 uses signed int32 0.01 rpm; A2 same speed units; A1 signed 0.001 A;
B0 float32 torque constant; AE unsigned voltage/current, temperature/mode/fault.
Torque is an estimate from A1 * B0, NOT a shaft torque sensor or bus current.

Before motion: CF disable confirmation; read B0/B3/CD; temporarily LOWER B3
current ceiling to at most --current-limit-a (never raise the existing limit),
enable CD communication watchdog, verify both by readback. These two settings
are volatile per local CAN protocol 3.09b0 pp.9,14-15. No origins/gains/faults
are reset. B3 acknowledgement AND independent readback may be at most 1 mA
below the requested value, never above it or zero. Actual readback is logged.
CD remains byte-exact. The same B3 tolerance applies when restoring originals.
Original settings are restored ONLY after CF confirms disabled.
On crash/power loss restoration is not guaranteed; the reduced current limit
and enabled watchdog may remain. Watchdog is communication, not independently
verified command-age protection. Other CAN traffic may affect it.

Ctrl-C/SIGTERM, query failure, feedback age, fault, overspeed, overcurrent,
temperature, RX overflow, conflicting writers and logging errors stop the test
and attempt CF disable. CF removes actuation; it does NOT mechanically stop a
spinning wheel. Software/firmware checks are not an emergency-stop guarantee.
No hard-real-time claim. Commands nominally 50 Hz, A2/A1 each 100 Hz, AE 5 Hz,
B0 1 Hz. CSV records each TX/RX, exact payloads, query results and cached state
with individual ages. Timestamps are HOST send/receipt times, not motor sample
times. Samples are sequential, not simultaneous; missing values are NaN.

Examples:
  python3 drive_speed_sweep.py --preview
  python3 drive_speed_sweep.py --execute --speeds 1 -1 --hold 5
  python3 drive_speed_sweep.py --execute --repeats 3
"""

from __future__ import annotations

import argparse
from contextlib import closing
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import select
import signal
import socket
import struct
import sys
import time

CAN_FRAME = struct.Struct("=IB3x8s")
CAN_MASK = 0xC00007FF
SO_RXQ_OVFL = getattr(socket, "SO_RXQ_OVFL", 40)
B3_DOWNWARD_TOLERANCE_MA = 1  # One protocol count; observed 1000 mA request -> 999 mA reply.
LENGTHS = {0xA1: 5, 0xA2: 5, 0xB0: 7, 0xAE: 8, 0xB3: 5, 0xCD: 5, 0xCF: 8, 0xC1: 5}
FIELDS = (
    "record_type", "time_unix_s", "elapsed_s", "repeat", "step", "phase", "phase_elapsed_s",
    "motor_id", "gear_ratio", "command_sign", "encoder_sign", "torque_sign",
    "plateau_speed_rad_s", "command_speed_rad_s", "encoded_command_rad_s", "command_native_rpm",
    "command_interval_s", "command_lateness_s", "can_id_hex", "opcode_hex", "payload_hex",
    "query_status", "query_round_trip_s", "reply_native_rpm", "reply_velocity_rad_s",
    "velocity_rad_s", "current_a", "torque_estimate_nm", "torque_constant_nm_per_a",
    "velocity_sample_elapsed_s", "current_sample_elapsed_s", "status_sample_elapsed_s", "b0_sample_elapsed_s",
    "velocity_age_s", "current_age_s", "status_age_s", "b0_age_s", "at_target",
    "bus_voltage_v", "bus_current_a", "bus_current_raw", "motor_temperature_c", "motor_mode", "fault_code",
    "current_limit_a", "watchdog_ms", "watchdog_enabled", "note",
)


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Open CAN, run motor, save CSV.")
    mode.add_argument("--preview", action="store_true", help="Preview only (also the default).")
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--motor-id", type=lambda s: int(s, 0), default=1)
    parser.add_argument("--speeds", nargs="+", type=finite, default=[1, 2, 4, 6, 8, -1, -2, -4, -6, -8],
                        help="Ordered wheel/output speeds in rad/s; negative numbers accepted.")
    parser.add_argument("--repeats", type=int, default=1)
    for flag, default, help_text in (
        ("hold", 10.0, "Seconds at each constant target, after settling."),
        ("settle", 5.0, "Seconds at target before the hold phase; inspect actual tracking."),
        ("rest", 2.0, "Seconds at zero before/between speeds and at the end."),
        ("acceleration", 1.0, "Host target ramp rate, rad/s^2; firmware may ramp more slowly."),
        ("command-rate", 50.0, "Nominal C1 messages/s."),
        ("query-rate", 100.0, "Nominal A2 and A1 queries/s EACH."),
        ("status-rate", 5.0, "Nominal AE queries/s."),
        ("b0-rate", 1.0, "Nominal B0 queries/s."),
        ("query-timeout", 0.02, "Seconds per query; any timeout aborts, no stale retry substitution."),
        ("feedback-timeout", 0.25, "Maximum A2/A1 age, seconds."),
        ("current-limit-a", 1.0, "Temporary B3 current ceiling in A (never raises existing ceiling)."),
        ("speed-limit", 10.0, "Abort above this measured absolute output speed, rad/s."),
        ("max-temperature", 60.0, "Conservative abort temperature in degrees C, not a motor rating."),
        ("gear-ratio", 1.0, "Protocol-side speed / joint speed; do not double-count reduction."),
    ):
        parser.add_argument(f"--{flag}", type=finite, default=default, help=help_text)
    parser.add_argument("--watchdog-ms", type=int, default=500,
                        help="Temporary CD timeout; keeps a shorter enabled one.")
    for name in ("command", "encoder", "torque"):
        parser.add_argument(f"--{name}-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--output", type=Path, help="Exclusive CSV path; default: logs/ beside this script.")
    args = parser.parse_args(argv)
    numeric = (args.hold, args.settle, args.rest, args.acceleration, args.command_rate, args.query_rate,
               args.status_rate, args.b0_rate, args.query_timeout, args.feedback_timeout,
               args.current_limit_a, args.speed_limit, args.max_temperature, args.gear_ratio)
    if min(numeric) <= 0 or not 1 <= args.motor_id <= 254 or args.repeats < 1:
        parser.error("rates, durations, limits and gear must be positive; motor ID 1..254; repeats >= 1")
    if max(abs(v) for v in args.speeds) > args.speed_limit or not any(args.speeds):
        parser.error("speeds must include a nonzero value and stay within --speed-limit")
    if not 100 <= args.watchdog_ms <= 1000 or not 0.001 <= args.current_limit_a <= 4294967.295:
        parser.error("watchdog must be 100..1000 ms; current ceiling must fit uint32 milliamps and be >=0.001 A")
    if args.query_timeout >= args.feedback_timeout or max(args.command_rate, args.query_rate) > 500:
        parser.error("query timeout must be below feedback timeout; command/query rates at most 500 Hz")
    if 1/args.query_rate >= args.feedback_timeout:
        parser.error("feedback timeout must exceed the nominal A1/A2 query period")
    if max(args.status_rate, args.b0_rate) > args.query_rate:
        parser.error("status/B0 rates must not exceed the A1/A2 query rate")
    if 3000/args.command_rate > args.watchdog_ms:
        parser.error("watchdog duration must cover at least three nominal command periods")
    if args.output is None:
        name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_drive_speed_sweep.csv"
        args.output = Path(__file__).resolve().parent / "logs" / name
    try:
        for speed in args.speeds:  # Check encoded command range before any CAN access.
            speed_payload(args, speed)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    return args


def speed_payload(args, speed):
    native_rpm = args.command_sign * args.gear_ratio * speed * 60 / (2 * math.pi)
    code = round(native_rpm * 100)
    if not -(2**31) <= code < 2**31:
        raise ValueError("C1 target does not fit int32 0.01-rpm units")
    encoded = args.command_sign * code * 0.01 * 2 * math.pi / 60 / args.gear_ratio
    return b"\xc1" + struct.pack("<i", code), encoded, code * 0.01


def setting_matches(requested, reported):
    """Only B3 permits one-count downward quantization; never accept a higher ceiling."""
    if not requested or not reported or requested[0] != reported[0]:
        return False
    if requested[0] != 0xB3:
        return reported == requested
    if len(requested) != 5 or len(reported) != 5:
        return False
    wanted_ma = struct.unpack("<I", requested[1:])[0]
    actual_ma = struct.unpack("<I", reported[1:])[0]
    return actual_ma > 0 and 0 <= wanted_ma-actual_ma <= B3_DOWNWARD_TOLERANCE_MA


@dataclass(frozen=True)
class Phase:
    start: float
    duration: float
    name: str
    initial: float
    final: float
    plateau: float
    repeat: int
    step: int

    def target(self, elapsed):
        fraction = min(1.0, max(0.0, (elapsed - self.start) / self.duration))
        return self.initial + fraction * (self.final - self.initial)


def make_plan(args):
    phases, start = [], 0.0
    def add(name, duration, initial, final, plateau, repeat, step):
        nonlocal start
        if duration > 0:
            phases.append(Phase(start, duration, name, initial, final, plateau, repeat, step))
            start += duration
    add("rest", args.rest, 0, 0, 0, 0, 0)
    for repeat in range(1, args.repeats + 1):
        for step, speed in enumerate(args.speeds, 1):
            ramp = abs(speed) / args.acceleration
            add("ramp_up", ramp, 0, speed, speed, repeat, step)
            add("settle", args.settle, speed, speed, speed, repeat, step)
            add("hold", args.hold, speed, speed, speed, repeat, step)
            add("ramp_down", ramp, speed, 0, speed, repeat, step)
            add("rest", args.rest, 0, 0, speed, repeat, step)
    if not math.isfinite(start):
        raise ValueError("sweep duration overflow; use smaller speeds/repeats or a faster ramp")
    return phases


class CsvLog:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", newline="")  # Never overwrite previous measurements.
        try:
            self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
            self.writer.writeheader()
            self.file.flush()
        except BaseException:
            self.file.close()
            raise

    def emit(self, row):
        self.writer.writerow({key: row.get(key, math.nan) for key in FIELDS})

    def close(self):
        self.file.close()


def open_bus(args):
    bus = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        ids = (args.motor_id, 0x100 | args.motor_id, args.motor_id | 0x400, args.motor_id | 0x500,
               0, 0xFF, 0x400, 0x4FF)
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                       b"".join(struct.pack("=II", cid, CAN_MASK) for cid in ids))
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
        # Linux UAPI <linux/can/raw.h>: CAN_RAW_ERR_FILTER=2. Some Pi Python
        # builds omit the name even though the kernel supports the option.
        error_filter = getattr(socket, "CAN_RAW_ERR_FILTER", 2)
        bus.setsockopt(socket.SOL_CAN_RAW, error_filter, struct.pack("=I", 0x1FFFFFFF))
        bus.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
        bus.setblocking(False)
        bus.bind((args.interface,))
        return bus
    except BaseException:
        bus.close()
        raise


class Sweep:
    def __init__(self, args, bus, log, stop_requested=lambda: False, clock=time):
        self.args, self.bus, self.log, self.stop_requested, self.clock = args, bus, log, stop_requested, clock
        self.start = clock.monotonic()
        self.motion, self.stopping = False, False
        self.phase, self.phase_started, self.plateau, self.repeat, self.step = "startup", self.start, 0, 0, 0
        self.velocity = self.current = self.constant = math.nan
        self.times = dict.fromkeys(("velocity", "current", "status", "b0"), math.nan)
        self.status = {}
        self.command_speed = self.encoded_speed = self.command_rpm = 0.0
        self.last_command, self.next_command = math.nan, math.inf
        self.current_limit, self.watchdog_ms = math.nan, math.nan
        self.watchdog_enabled, self.log_error = None, None
        self.saved, self.last_direction_error = [], None

    def emit(self, kind, stamp=None, **values):
        now = self.clock.monotonic() if stamp is None else stamp
        args = self.args
        row = dict(record_type=kind, time_unix_s=self.clock.time(), elapsed_s=now-self.start,
                   repeat=self.repeat, step=self.step, phase=self.phase, phase_elapsed_s=now-self.phase_started,
                   motor_id=args.motor_id, gear_ratio=args.gear_ratio, command_sign=args.command_sign,
                   encoder_sign=args.encoder_sign, torque_sign=args.torque_sign, plateau_speed_rad_s=self.plateau,
                   command_speed_rad_s=self.command_speed, encoded_command_rad_s=self.encoded_speed,
                   command_native_rpm=self.command_rpm, velocity_rad_s=self.velocity, current_a=self.current,
                   torque_constant_nm_per_a=self.constant,
                   torque_estimate_nm=args.torque_sign*self.current*self.constant,
                   current_limit_a=self.current_limit, watchdog_ms=self.watchdog_ms,
                   watchdog_enabled=self.watchdog_enabled, **self.status)
        for name, stamp_value in self.times.items():
            row[f"{name}_sample_elapsed_s"] = stamp_value-self.start
            row[f"{name}_age_s"] = now-stamp_value
        row["at_target"] = abs(self.velocity-self.encoded_speed) <= max(0.05, 0.05*abs(self.encoded_speed))
        row.update(values)
        try:
            self.log.emit(row)
        except Exception as error:
            self.log_error = error
            if not self.stopping:
                raise
            # Disk failure must not prevent disable/restore CAN operations.

    def send(self, payload, **values):
        now = self.clock.monotonic()
        if self.bus.send(CAN_FRAME.pack(0x100 | self.args.motor_id, len(payload), payload)) != CAN_FRAME.size:
            raise OSError("incomplete CAN send")
        self.emit("command" if payload[0] == 0xC1 else "tx", now,
                  can_id_hex=f"{0x100 | self.args.motor_id:03X}", opcode_hex=f"{payload[0]:02X}",
                  payload_hex=payload.hex().upper(), **values)
        return now

    def receive(self):
        frame, ancillary, flags, _ = self.bus.recvmsg(CAN_FRAME.size, socket.CMSG_SPACE(4))
        now = self.clock.monotonic()
        if len(frame) != CAN_FRAME.size or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise RuntimeError("truncated CAN frame/metadata")
        for level, kind, value in ancillary:
            if level == socket.SOL_SOCKET and kind == SO_RXQ_OVFL:
                if len(value) != 4 or struct.unpack("=I", value)[0]:
                    raise RuntimeError("CAN receive overflow; recording incomplete")
        cid, length, padded = CAN_FRAME.unpack(frame)
        if cid & 0x20000000:
            raise RuntimeError(f"CAN error frame: {cid:08X} {padded.hex()}")
        payload = padded[:length]
        details = dict(can_id_hex=f"{cid:03X}", opcode_hex=f"{payload[0]:02X}" if payload else "",
                       payload_hex=payload.hex().upper())
        try:
            if cid != self.args.motor_id or flags & socket.MSG_DONTROUTE:
                raise RuntimeError("another selected-motor/broadcast CAN writer observed; stop all other writers")
            if length > 8 or not payload or LENGTHS.get(payload[0]) != length:
                raise RuntimeError("unexpected/malformed selected-motor reply")
            details.update(self.decode(payload, now))
        except (RuntimeError, ValueError) as error:
            self.emit("rx_error", now, note=str(error), **details)
            raise
        self.emit("rx", now, **details)
        return payload, now

    def decode(self, data, now):
        op, extra = data[0], {}
        if op in (0xA2, 0xC1):
            rpm = struct.unpack("<i", data[1:])[0]*0.01
            speed = self.args.encoder_sign*rpm*2*math.pi/60/self.args.gear_ratio
            extra.update(reply_native_rpm=rpm, reply_velocity_rad_s=speed)
            if not self.stopping and abs(speed) > self.args.speed_limit:
                raise RuntimeError(f"overspeed: {speed:.3f} rad/s")
            if op == 0xA2:
                self.velocity, self.times["velocity"] = speed, now
        elif op == 0xA1:
            self.current = struct.unpack("<i", data[1:])[0]*0.001
            self.times["current"] = now
            if (not self.stopping and math.isfinite(self.current_limit)
                    and abs(self.current) > 1.25*self.current_limit+0.02):
                raise RuntimeError(f"measured current {self.current:.3f} A exceeds test guard")
        elif op == 0xB0:
            constant = struct.unpack("<f", data[2:6])[0]
            if not math.isfinite(constant) or not 0.01 <= constant <= 10:
                raise RuntimeError("invalid B0 torque constant")
            if self.motion and not math.isclose(constant, self.constant, rel_tol=1e-5):
                raise RuntimeError("B0 torque constant changed during sweep")
            self.constant, self.times["b0"] = constant, now
        elif op == 0xB3:
            self.current_limit = struct.unpack("<I", data[1:])[0]*0.001
        elif op == 0xCD:
            self.watchdog_enabled = data[1]
            self.watchdog_ms = struct.unpack("<H", data[2:4])[0]
        elif op in (0xAE, 0xCF):
            voltage, bus_current = struct.unpack("<HH", data[1:5])
            self.status = dict(bus_voltage_v=voltage*0.01, bus_current_a=bus_current*0.01,
                               bus_current_raw=bus_current, motor_temperature_c=data[5], motor_mode=data[6],
                               fault_code=data[7])
            self.times["status"] = now
            if not self.stopping:
                if data[7] or data[5] >= self.args.max_temperature or voltage == 0:
                    raise RuntimeError(f"unsafe motor status: {self.status}")
                if self.motion and data[6] != 3:
                    raise RuntimeError(f"motor left C1 speed mode: mode={data[6]}")
        return extra

    def wait(self, duration):
        select.select([self.bus], [], [], max(0.0, duration))

    def service(self):
        if self.stopping:
            return
        if self.stop_requested():
            raise KeyboardInterrupt
        if not self.motion:
            return
        now = self.clock.monotonic()
        for name in ("velocity", "current"):
            if now-self.times[name] > self.args.feedback_timeout:
                raise RuntimeError(f"stale {name} feedback")
        if now-self.times["status"] > max(3/self.args.status_rate, self.args.feedback_timeout):
            raise RuntimeError("stale AE motor status")
        max_gap = min(max(0.1, 3/self.args.command_rate), self.watchdog_ms/2000)
        if now-self.last_command > max_gap:
            raise RuntimeError("C1 command scheduling gap exceeded safety deadline")
        if now >= self.next_command:
            elapsed = min(now-self.motion_start, self.plan[-1].start+self.plan[-1].duration)
            while self.phase_index+1 < len(self.plan) and elapsed >= self.plan[self.phase_index+1].start:
                self.phase_index += 1
                phase = self.plan[self.phase_index]
                self.phase, self.phase_started = phase.name, self.motion_start+phase.start
                self.repeat, self.step, self.plateau = phase.repeat, phase.step, phase.plateau
                self.emit("event", note="phase started; hold is a target plateau, not verified steady motion")
                if self.phase == "ramp_up" and abs(self.velocity) > 0.2:
                    raise RuntimeError("wheel did not stop before the next speed; increase --rest/adjust firmware")
                if self.phase == "hold":
                    print(f"Repeat {self.repeat}, step {self.step}: hold target {self.plateau:+g} rad/s; "
                          f"measured {self.velocity:+.3f} rad/s", flush=True)
            self.command_speed = self.plan[self.phase_index].target(elapsed)
            payload, self.encoded_speed, self.command_rpm = speed_payload(self.args, self.command_speed)
            interval, late = now-self.last_command, max(0.0, now-self.next_command)
            self.last_command = self.send(payload, command_interval_s=interval, command_lateness_s=late)
            self.next_command = self.last_command+1/self.args.command_rate  # Never burst missed ticks.
        wrong_direction = self.encoded_speed*self.velocity < 0 and abs(self.velocity) > 0.3
        if wrong_direction:
            if self.last_direction_error is None:
                self.last_direction_error = now
            elif now-self.last_direction_error > 0.2:
                raise RuntimeError("C1 command/encoder direction mismatch; verify signs at low speed")
        else:
            self.last_direction_error = None

    def request(self, opcode, payload=None, timeout=None):
        # Clear queued replies BEFORE sending; no reuse/retry after a failed motion query.
        # CF must be sent BEFORE inspecting a potentially unsafe/backlogged bus.
        if opcode != 0xCF:
            for _ in range(64):
                try:
                    self.receive()
                except BlockingIOError:
                    break
            else:
                raise RuntimeError("CAN receive backlog; cannot correlate the next query")
        self.service()
        sent = self.send(bytes([opcode]) if payload is None else payload)
        deadline = sent+(self.args.query_timeout if timeout is None else timeout)
        while True:
            now = self.clock.monotonic()
            if now >= deadline:
                self.emit("query", opcode_hex=f"{opcode:02X}", query_status="timeout")
                raise TimeoutError(f"no timely {opcode:02X} reply; stopping, not substituting cached data")
            self.service()
            try:
                data, received = self.receive()
            except BlockingIOError:
                self.wait(min(deadline-now, self.next_command-now) if self.motion else deadline-now)
                continue
            if data[0] == opcode and received <= deadline:
                self.emit("query", received, opcode_hex=f"{opcode:02X}", query_status="received",
                          query_round_trip_s=received-sent)
                return data

    def verify_setting(self, requested, reported, stage):
        if not setting_matches(requested, reported):
            raise RuntimeError(f"{stage} mismatch: requested={requested.hex().upper()}, "
                               f"reported={reported.hex().upper()}")
        if requested != reported:
            wanted_ma = struct.unpack("<I", requested[1:])[0]
            actual_ma = struct.unpack("<I", reported[1:])[0]
            note = (f"{stage}: accepted B3 {actual_ma} mA for {wanted_ma} mA request "
                    f"({wanted_ma-actual_ma} mA downward quantization)")
            self.emit("event", note=note)
            if stage.endswith("readback"):
                print(note, flush=True)

    def startup(self):
        self.emit("event", note=json.dumps({key: str(value) if isinstance(value, Path) else value
                                           for key, value in vars(self.args).items()}, sort_keys=True))
        if self.request(0xCF, timeout=0.25)[6] != 0:
            raise RuntimeError("CF did not confirm disabled motor")
        self.request(0xB0, timeout=0.25)
        original_b3 = self.request(0xB3, timeout=0.25)
        original_cd = self.request(0xCD, timeout=0.25)
        old_limit = struct.unpack("<I", original_b3[1:])[0]
        if old_limit == 0 or original_cd[1] not in (0, 1):
            raise RuntimeError("invalid/zero B3 ceiling or invalid CD watchdog configuration")
        ceiling = min(old_limit, math.floor(self.args.current_limit_a*1000))
        self.current_limit = ceiling*0.001
        old_ms = struct.unpack("<H", original_cd[2:4])[0]
        self.watchdog_ms = min(old_ms, self.args.watchdog_ms) if original_cd[1] else self.args.watchdog_ms
        if self.watchdog_ms < 3*1000/self.args.command_rate:
            raise RuntimeError("existing watchdog too short for command rate; no settings changed")
        configs = ((original_b3, b"\xb3"+struct.pack("<I", ceiling)),
                   (original_cd, b"\xcd\x01"+struct.pack("<H", self.watchdog_ms)+original_cd[4:]))
        for original, requested in configs:
            if original != requested:
                self.saved.append(original)  # Register restoration BEFORE a write/possibly lost acknowledgement.
                self.verify_setting(requested, self.request(requested[0], requested, timeout=0.25),
                                    "temporary safety setting acknowledgement")
            self.verify_setting(requested, self.request(requested[0], timeout=0.25),
                                "temporary safety setting readback")
        # Refresh measurements AFTER setup, without taking a motor-origin reference.
        self.request(0xAE, timeout=0.25)
        if self.status["motor_mode"] != 0:
            raise RuntimeError("motor is not disabled after setup")
        self.request(0xA1, timeout=0.25)
        self.request(0xA2, timeout=0.25)
        if abs(self.velocity) > 0.2:
            raise RuntimeError("rear wheel is not stationary at startup")
        print(f"Verified ID {self.args.motor_id}: Kt={self.constant:g} Nm/A, "
              f"current ceiling={self.current_limit:g} A, watchdog={self.watchdog_ms} ms", flush=True)

    def run(self):
        self.plan = make_plan(self.args)
        self.motion_start = self.next_command = self.last_command = self.clock.monotonic()
        self.phase_index, self.motion = 0, True
        self.phase, self.phase_started = "rest", self.motion_start
        periods = {0xA2: 1/self.args.query_rate, 0xA1: 1/self.args.query_rate,
                   0xAE: 1/self.args.status_rate, 0xB0: 1/self.args.b0_rate}
        due = dict.fromkeys(periods, self.motion_start)
        end = self.motion_start+self.plan[-1].start+self.plan[-1].duration
        next_flush = self.motion_start+1
        while self.clock.monotonic() < end:
            self.service()
            now = self.clock.monotonic()
            opcode = min(due, key=due.get)
            if now >= due[opcode]:
                self.request(opcode)
                due[opcode] = self.clock.monotonic()+periods[opcode]
            else:
                try:
                    self.receive()
                except BlockingIOError:
                    self.wait(min(due[opcode], self.next_command, end)-now)
            if self.clock.monotonic() >= next_flush:
                self.log.file.flush()
                next_flush = self.clock.monotonic()+1
        self.motion = False
        self.emit("event", note="sweep completed; disabling motor next")

    def shutdown(self):
        self.motion, self.stopping, self.phase = False, True, "shutdown"
        confirmed = False
        for _ in range(3):
            try:
                # A fresh AE read after draining queued frames also checks the mode;
                # do not restore settings on the strength of a queued old CF alone.
                if self.request(0xCF, timeout=0.15)[6] == 0 and self.request(0xAE, timeout=0.15)[6] == 0:
                    confirmed = True
                    break
            except (OSError, RuntimeError, ValueError):
                pass
        if not confirmed:
            print("ERROR: CF disable NOT confirmed. Safety settings retained. Check motor physically!", file=sys.stderr)
            return False
        try:
            for original in self.saved:  # Restore B3 then CD, only while confirmed disabled.
                self.verify_setting(original, self.request(original[0], original, timeout=0.15),
                                    "restoration acknowledgement")
                self.verify_setting(original, self.request(original[0], timeout=0.15), "restoration readback")
            if self.request(0xCF, timeout=0.15)[6] != 0:
                raise RuntimeError("final disable not confirmed")
            self.emit("event", note="CF disabled confirmed; original B3/CD verified, B3 may be at most 1 mA lower")
            print("Motor disabled; B3/CD restored and verified (B3 allows up to 1 mA lower). "
                  "Wheel may still coast.", flush=True)
            if self.log_error is not None:
                print(f"ERROR: CSV recording incomplete: {self.log_error}", file=sys.stderr)
                return False
            return True
        except (OSError, RuntimeError, ValueError) as error:
            # A failed logger or parameter restore must never prevent a final disable attempt.
            try:
                self.bus.send(CAN_FRAME.pack(0x100 | self.args.motor_id, 1, b"\xcf"))
            except OSError:
                pass
            print(f"ERROR: shutdown/restore incomplete: {error}; inspect motor/settings physically", file=sys.stderr)
            return False


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    print(f"Device {args.motor_id}: host {0x100 | args.motor_id:03X}, feedback {args.motor_id:03X}; "
          f"speeds {args.speeds} rad/s; {args.repeats} repeat(s); "
          f"planned duration {plan[-1].start+plan[-1].duration:.1f} s")
    if not args.execute:
        print("PREVIEW ONLY: no CAN, CSV or directories opened. Add --execute to run.")
        for phase in plan:
            print(f"{phase.start:7.1f}s {phase.duration:6.1f}s repeat={phase.repeat} step={phase.step} "
                  f"{phase.name:10} {phase.initial:+g} -> {phase.final:+g} rad/s")
        return 0
    stop = [False]
    def interrupt(signum, frame):
        stop[0] = True  # Cleanup is not interrupted by a second Ctrl-C/SIGTERM.
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with closing(CsvLog(args.output)) as log, closing(open_bus(args)) as bus:
            sweep = Sweep(args, bus, log, lambda: stop[0])
            print(f"CSV: {args.output}\nRear wheel MUST be airborne; all other CAN writers stopped.", flush=True)
            try:
                sweep.startup()
                sweep.run()
            finally:
                if not sweep.shutdown() and sys.exc_info()[0] is None:
                    raise RuntimeError("motor shutdown/restoration was not fully confirmed")
    except KeyboardInterrupt:
        print("Interrupted; partial CSV retained.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}; partial CSV retained if it was created.", file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
