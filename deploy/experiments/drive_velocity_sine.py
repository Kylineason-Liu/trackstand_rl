#!/usr/bin/env python3
"""Rear ID 1 velocity sine: C1, MIT+host integral, or C1 then MIT.

Python stdlib only; preview by default. --execute opens can0 and logs CSV/JSON.
Each selected method runs 10 wall-clock seconds, nominally 200 Hz. Default sine
is +/-4 rad/s at 0.5 Hz (five cycles). Rear wheel airborne/free, frame supported,
steering independently disabled, all other CAN writers stopped.

B8/B9 are read, never written. The local protocol does NOT document their gain
units or integral time convention. MIT therefore needs explicit --gain-speed-unit,
--gain-current-unit and --ki-timebase. These declare a basis, NOT a verified fact.
Kt from B0 converts current PI gains to torque gains; MIT Kp=0, Kd=converted B8,
feedforward=converted B9 * integral(target_velocity - A2_velocity) dt.

No angle guard, extra speed ramp, tracking-error abort, tight packet deadline or
silent gain/integral clipping. Late slots are logged/skipped, not burst or stretched.
Speed >61.6 rad/s, measured/estimated torque >6.28 Nm, stale feedback, driver/bus
faults and invalid packets stop motion. B3 is read/logged only: no current-limit
writes or matching checks. CD is enabled/shortened to at most 500 ms and restored
after confirmed CF disable. B1/F0/B3/B5/gains are never changed.
Host limits cannot guarantee a physical torque cap between feedback samples.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import queue
import select
import signal
import socket
import struct
import sys
import threading
import time


RATE_HZ, DURATION_S = 200.0, 10.0
SPEED_LIMIT_RAD_S, TORQUE_LIMIT_NM = 61.6, 6.28
SIGN = -1.0  # Native positive is backward; output-side feedback, gear ratio 1.
CAN_FRAME = struct.Struct("=IB3x8s")
RX_OVERFLOW = getattr(socket, "SO_RXQ_OVFL", 40)
LENGTHS = {0xA1: 5, 0xA2: 5, 0xA3: 7, 0xAE: 8, 0xB0: 7, 0xB3: 5, 0xB5: 5,
           0xB8: 5, 0xB9: 5, 0xC1: 5, 0xCD: 5, 0xCF: 8, 0xF0: 7, 0xF1: 7}
SPEED_UNITS_PER_RAD_S = {"rad_s": 1.0, "rpm": 60 / (2 * math.pi),
                         "centirpm": 6000 / (2 * math.pi), "turn_s": 1 / (2 * math.pi)}
FIELDS = (
    "record_type", "time_unix_s", "host_monotonic_s", "elapsed_s", "method", "phase_elapsed_s",
    "can_id", "payload_hex", "opcode", "command_index", "command_interval_s", "lateness_s", "skipped_slots",
    "target_velocity_rad_s", "target_sample_time_s", "encoded_target_velocity_rad_s", "velocity_rad_s",
    "velocity_age_s",
    "current_a", "current_age_s", "current_torque_nm", "position_rad", "multi_turn_count",
    "integral_error_rad", "integral_torque_nm", "estimated_mit_total_torque_nm",
    "mit_kp", "mit_kd", "mit_feedforward_native_nm", "mit_position_native_rad",
    "mit_velocity_rad_s", "mit_torque_nm", "mit_status", "bus_voltage_v", "bus_current_a",
    "temperature_c", "driver_mode", "fault_code", "note",
)


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("c1", "mit", "both"), default="both")
    parser.add_argument("--execute", action="store_true", help="Run motor; otherwise preview without CAN/files")
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--amplitude", type=finite, default=4.0, help="Peak velocity in rad/s, not peak-to-peak")
    parser.add_argument("--frequency", type=finite, default=.5, help="Sine frequency in Hz")
    parser.add_argument("--gain-speed-unit", choices=tuple(SPEED_UNITS_PER_RAD_S))
    parser.add_argument("--gain-current-unit", choices=("a", "ma"))
    parser.add_argument("--ki-timebase", choices=("second", "tick"))
    parser.add_argument("--ki-period-s", type=finite, help="Firmware PI period, required for per-tick Ki")
    parser.add_argument("--output", type=Path, help="Exclusive CSV; default logs/<timestamp>_drive_velocity_sine.csv")
    args = parser.parse_args(argv)
    if not 0 < args.amplitude <= SPEED_LIMIT_RAD_S or not 0 < args.frequency < RATE_HZ / 2:
        parser.error("amplitude must be in (0,61.6] rad/s and frequency in (0,100) Hz")
    if args.ki_timebase == "tick" and (args.ki_period_s is None or args.ki_period_s <= 0):
        parser.error("per-tick Ki requires the actual positive --ki-period-s, not the host packet period")
    if args.ki_period_s is not None and args.ki_timebase != "tick":
        parser.error("--ki-period-s applies only with --ki-timebase tick")
    if (args.execute and args.method != "c1"
            and not all((args.gain_speed_unit, args.gain_current_unit, args.ki_timebase))):
        parser.error("B8/B9 gain units are undocumented: MIT requires explicit --gain-speed-unit, "
                     "--gain-current-unit and --ki-timebase; no unit basis is assumed")
    if args.output is None:
        args.output = Path(__file__).resolve().parent / "logs" / (
            f"{datetime.now():%Y%m%d_%H%M%S_%f}_drive_velocity_sine.csv")
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must have .csv extension")
    return args


def converted_gains(kp, ki, kt, speed_unit, current_unit, timebase, firmware_period=None):
    if not all(math.isfinite(v) and v >= 0 for v in (kp, ki)) or not math.isfinite(kt) or kt <= 0:
        raise ValueError("invalid B8/B9/B0 values")
    factor = kt * {"a": 1.0, "ma": .001}[current_unit] * SPEED_UNITS_PER_RAD_S[speed_unit]
    if timebase not in ("second", "tick"):
        raise ValueError("unknown Ki timebase")
    if timebase == "tick" and (firmware_period is None or not math.isfinite(firmware_period) or firmware_period <= 0):
        raise ValueError("unknown firmware integral timestep")
    return kp * factor, ki * factor / (firmware_period if timebase == "tick" else 1.0)


def encode(value, maximum, bits):
    if not math.isfinite(value) or maximum <= 0 or abs(value) > maximum:
        raise ValueError(f"MIT value {value:g} outside encoding range +/-{maximum:g}")
    return round((value / maximum + 1) * ((1 << bits) - 1) / 2)


def decode(code, maximum, bits):
    return (2 * code / ((1 << bits) - 1) - 1) * maximum


def c1_payload(target_rad_s):
    code = round(SIGN * target_rad_s * 60 / (2 * math.pi) * 100)
    return b"\xc1" + struct.pack("<i", code), SIGN * code * .01 * 2 * math.pi / 60


def mit_payload(target_rad_s, integral_nm, measured_rad_s, kd, ranges):
    pmax, vmax, tmax = ranges
    if not 0 <= kd <= 5:
        raise ValueError(f"converted MIT Kd {kd:g} outside protocol range [0,5]; gain is not clipped")
    p, v, d, t = encode(0., pmax, 16), encode(SIGN * target_rad_s, vmax, 12), round(kd * 4095 / 5), encode(
        SIGN * integral_nm, tmax, 12)
    actual_v, actual_ff, actual_kd = decode(v, vmax, 12), decode(t, tmax, 12), d * 5 / 4095
    estimated = SIGN * (actual_kd * (actual_v - SIGN * measured_rad_s) + actual_ff)
    if abs(estimated) > TORQUE_LIMIT_NM:
        raise RuntimeError(f"estimated total MIT torque {estimated:g} Nm exceeds {TORQUE_LIMIT_NM:g}; stopping")
    payload = bytes((p >> 8, p & 255, v >> 4, (v & 15) << 4, 0,
                     d >> 4, ((d & 15) << 4) | (t >> 8), t & 255))
    return payload, dict(encoded_target_velocity_rad_s=SIGN * actual_v, mit_kp=0., mit_kd=actual_kd,
                         mit_feedforward_native_nm=actual_ff, estimated_mit_total_torque_nm=estimated)


@dataclass
class Integral:
    error_rad: float = 0.0
    previous_error: float | None = None
    previous_time: float | None = None

    def update(self, error, stamp):
        if self.previous_time is not None:
            dt = stamp - self.previous_time
            if dt < 0:
                raise RuntimeError("non-monotonic integral clock")
            self.error_rad += .5 * (error + self.previous_error) * dt
        self.previous_error, self.previous_time = error, stamp
        return self.error_rad


class CsvLog:
    """Keep file writes outside the 200-Hz CAN owner."""

    def __init__(self, path):
        self.path, self.rows = path, queue.Queue(maxsize=20000)
        self.stop, self.error = threading.Event(), None

    def __enter__(self):
        self.file = self.path.open("x", newline="")
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
        if self.error:
            raise RuntimeError(f"CSV write failed: {self.error}") from self.error

    def emit(self, row):
        self.check()
        try:
            self.rows.put_nowait(row)
        except queue.Full as error:
            raise RuntimeError("CSV queue full; recording cannot keep up") from error

    def _write(self):
        flush_at = time.monotonic() + 1
        try:
            while not self.stop.is_set() or not self.rows.empty():
                try:
                    self.writer.writerow(self.rows.get(timeout=.05))
                except queue.Empty:
                    pass
                if time.monotonic() >= flush_at:
                    self.file.flush()
                    flush_at = time.monotonic() + 1
            self.file.flush()
        except BaseException as error:
            self.error = error
        finally:
            self.file.close()

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("CSV writer did not finish")
        self.check()


def open_bus(interface):
    bus = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        ids = (1, 0x101, 0x501, 0, 0xff, 0x400, 0x4ff)
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                       b"".join(struct.pack("=II", cid, 0xC00007FF) for cid in ids))
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
        bus.setsockopt(socket.SOL_CAN_RAW, getattr(socket, "CAN_RAW_ERR_FILTER", 2), struct.pack("=I", 0x1FFFFFFF))
        bus.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        bus.setsockopt(socket.SOL_SOCKET, RX_OVERFLOW, 1)
        bus.bind((interface,))
        bus.setblocking(False)
        return bus
    except BaseException:
        bus.close()
        raise


class VelocityTest:
    def __init__(self, args, bus, log, stop_requested=lambda: False, clock=time):
        self.args, self.bus, self.log, self.clock, self.stop_requested = args, bus, log, clock, stop_requested
        self.started = clock.monotonic()
        self.phase_start, self.method, self.cleaning, self.active = self.started, "startup", False, False
        self.velocity = self.current = self.kt = math.nan
        self.velocity_at = self.current_at = self.status_at = self.f1_at = -math.inf
        self.raw_gains, self.saved, self.settings, self.phases = {}, [], {}, []
        self.applied_settings = {}
        self.ranges, self.f1_status = None, 0
        self.last_count, self.zero_count, self.travel_counts = None, None, 0
        self.position_rad, self.integral_nm = math.nan, 0.
        self.kd, self.ki = math.nan, math.nan
        self.next_command, self.last_command = math.inf, math.nan
        self.driver_status = {}

    def emit(self, kind, stamp=None, **extra):
        now = self.clock.monotonic() if stamp is None else stamp
        row = dict(record_type=kind, time_unix_s=self.clock.time(), host_monotonic_s=now,
                   elapsed_s=now - self.started, method=self.method, phase_elapsed_s=now - self.phase_start,
                   velocity_rad_s=self.velocity, velocity_age_s=now - self.velocity_at,
                   current_a=self.current, current_age_s=now - self.current_at,
                   current_torque_nm=SIGN * self.current * self.kt, position_rad=self.position_rad)
        row.update(extra)
        try:
            self.log.emit(row)
        except Exception:
            if not self.cleaning:
                raise  # During cleanup, logging must not prevent disable/restore.

    def send(self, payload, can_id=0x101, scheduled_at=None, **fields):
        stamp = self.clock.monotonic()
        if self.bus.send(CAN_FRAME.pack(can_id, len(payload), payload)) != CAN_FRAME.size:
            raise OSError("incomplete SocketCAN send")
        is_command = can_id == 0x501 or payload[0] == 0xC1
        if is_command:
            fields["command_interval_s"] = stamp - self.last_command if math.isfinite(self.last_command) else math.nan
            self.last_command = stamp
        if scheduled_at is not None:
            fields["lateness_s"] = max(0., stamp - scheduled_at)
        self.emit("command" if is_command else "tx", stamp,
                  can_id=f"{can_id:03X}", opcode="MIT" if can_id == 0x501 else f"{payload[0]:02X}",
                  payload_hex=payload.hex().upper(), **fields)
        return stamp

    def receive(self):
        frame, ancillary, flags, _ = self.bus.recvmsg(CAN_FRAME.size, socket.CMSG_SPACE(4))
        now = self.clock.monotonic()
        if len(frame) != CAN_FRAME.size or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise RuntimeError("truncated CAN frame")
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == RX_OVERFLOW and struct.unpack("=I", data)[0]:
                raise RuntimeError("CAN receive queue overflow; log is incomplete")
        cid, length, padded = CAN_FRAME.unpack(frame)
        data = padded[:length]
        if cid & ~0x7ff or length > 8:
            raise RuntimeError(f"CAN error/nonstandard frame {cid:08X}")
        if cid != 1 or flags & socket.MSG_DONTROUTE:
            raise RuntimeError("another rear/broadcast CAN writer observed; stop other writers")
        if not data or LENGTHS.get(data[0]) != length:
            raise RuntimeError(f"unexpected motor response: {data.hex()}")
        extra, fault = {}, None
        op = data[0]
        if op in (0xA2, 0xC1):
            self.velocity = SIGN * struct.unpack("<i", data[1:])[0] * .01 * 2 * math.pi / 60
            self.velocity_at = now
            if abs(self.velocity) > SPEED_LIMIT_RAD_S:
                fault = "motor speed exceeds maximum allowed speed"
        elif op == 0xA1:
            self.current, self.current_at = struct.unpack("<i", data[1:])[0] * .001, now
            if math.isfinite(self.kt) and abs(self.current * self.kt) > TORQUE_LIMIT_NM:
                fault = "measured current-derived torque exceeds maximum allowed torque"
        elif op == 0xA3:
            count = struct.unpack("<i", data[3:7])[0]
            if self.last_count is None:
                self.zero_count = count
            else:
                self.travel_counts += (count - self.last_count + 2**31) % 2**32 - 2**31
            self.last_count, self.position_rad = count, SIGN * self.travel_counts * 2 * math.pi / 16384
            extra["multi_turn_count"] = count
        elif op == 0xB0:
            self.kt = struct.unpack("<f", data[2:6])[0]
            if not math.isfinite(self.kt) or self.kt <= 0:
                fault = "invalid B0 torque constant"
        elif op in (0xB8, 0xB9):
            self.raw_gains[f"{op:02X}"] = struct.unpack("<f", data[1:])[0]
        elif op == 0xF0:
            p, v, t = struct.unpack("<HHH", data[1:])
            self.ranges = (p * .1, v * .01, t * .01)
            if min(self.ranges) <= 0:
                fault = "invalid F0 encoding ranges"
        elif op == 0xF1 and self.ranges is not None:
            p = (data[1] << 8) | data[2]
            v = (data[3] << 4) | (data[4] >> 4)
            t = ((data[4] & 15) << 8) | data[5]
            self.f1_at, self.f1_status = now, data[6]
            extra = dict(mit_position_native_rad=decode(p, self.ranges[0], 16),
                         mit_velocity_rad_s=SIGN * decode(v, self.ranges[1], 12),
                         mit_torque_nm=SIGN * decode(t, self.ranges[2], 12), mit_status=self.f1_status)
            if self.f1_status & 2:
                fault = "MIT driver system fault"
            elif abs(extra["mit_torque_nm"]) > TORQUE_LIMIT_NM:
                fault = "MIT torque feedback exceeds maximum allowed torque"
        elif op in (0xAE, 0xCF):
            volts, amps = struct.unpack("<HH", data[1:5])
            self.status_at = now
            self.driver_status = dict(bus_voltage_v=volts * .01, bus_current_a=amps * .01,
                                      temperature_c=data[5], driver_mode=data[6], fault_code=data[7])
            extra.update(self.driver_status)
            if data[7]:
                fault = f"motor driver fault code {data[7]}"
        self.emit("rx", now, can_id=f"{cid:03X}", payload_hex=data.hex().upper(), opcode=f"{op:02X}",
                  note=fault or "", **extra)
        if fault and not self.cleaning:
            raise RuntimeError(fault)
        return data

    def drain(self, maximum=64):
        for _ in range(maximum):
            if self.active and self.clock.monotonic() >= self.next_command:
                break
            try:
                self.receive()
            except BlockingIOError:
                break

    def request(self, opcode, payload=None, timeout=1.):
        if opcode != 0xCF:
            self.drain()
        self.send(bytes([opcode]) if payload is None else payload)
        deadline = self.clock.monotonic() + timeout
        while self.clock.monotonic() < deadline:
            if self.stop_requested() and not self.cleaning:
                raise KeyboardInterrupt
            try:
                data = self.receive()
                if data[0] == opcode:
                    return data
            except BlockingIOError:
                select.select([self.bus], [], [], max(0., min(.02, deadline - self.clock.monotonic())))
        raise TimeoutError(f"no {opcode:02X} response within {timeout:g} s")

    def disable(self):
        self.active = False
        deadline = self.clock.monotonic() + 1.
        while self.clock.monotonic() < deadline:
            try:
                response = self.request(0xCF, timeout=min(.1, deadline - self.clock.monotonic()))
                if response[6] == 0:
                    return
            except TimeoutError:
                pass
            # A first CF reply can still contain the previous active mode.
            # Allow the real transition instead of rejecting that one reply.
            retry_at = min(deadline, self.clock.monotonic() + .05)
            while self.clock.monotonic() < retry_at:
                self.drain()
                select.select([self.bus], [], [], max(0., retry_at - self.clock.monotonic()))
        raise RuntimeError("CF did not confirm disabled motor within 1 s")

    def configure_watchdog(self, wanted):
        reply = self.request(wanted[0], wanted)
        reread = self.request(wanted[0])
        if wanted != reply or wanted != reread:
            raise RuntimeError("CD watchdog readback mismatch")
        return reread

    def startup(self):
        self.disable()
        for op in (0xB0, 0xB8, 0xB9, 0xB3, 0xB5, 0xCD):
            self.settings[f"{op:02X}"] = self.request(op).hex().upper()
        if self.args.method != "c1":
            self.settings["F0"] = self.request(0xF0).hex().upper()
            self.kd, self.ki = converted_gains(self.raw_gains["B8"], self.raw_gains["B9"], self.kt,
                                              self.args.gain_speed_unit, self.args.gain_current_unit,
                                              self.args.ki_timebase, self.args.ki_period_s)
            if not 0 <= self.kd <= 5 or self.args.amplitude > self.ranges[1]:
                raise ValueError("converted Kd or sine velocity outside MIT encoding range; no gain/target clipping")
        cd = bytes.fromhex(self.settings["CD"])
        if cd[1] not in (0, 1):
            raise ValueError("invalid watchdog setting")
        timeout_ms = min(struct.unpack("<H", cd[2:4])[0], 500) if cd[1] else 500
        if timeout_ms <= 0:
            raise ValueError("invalid enabled watchdog timeout")
        wanted = b"\xcd\x01" + struct.pack("<H", timeout_ms) + cd[4:]
        if cd != wanted:
            self.saved.append(cd)
            readback = self.configure_watchdog(wanted)
        else:
            readback = cd
        self.applied_settings["CD"] = readback.hex().upper()
        for op in (0xA3, 0xA2, 0xA1, 0xAE):
            self.request(op)
        print(f"B8 Kp={self.raw_gains['B8']:.9g}, B9 Ki={self.raw_gains['B9']:.9g}, Kt={self.kt:.9g} Nm/A")
        if self.args.method != "c1":
            print(f"Declared-unit conversion: MIT Kp=0, Kd={self.kd:.9g} Nm/(rad/s), "
                  f"host Ki={self.ki:.9g} Nm/rad; basis is user-selected, not manufacturer-verified")

    def check_feedback(self, now):
        if now - min(self.velocity_at, self.current_at) > .25 or now - self.status_at > .5:
            raise RuntimeError("motor feedback unavailable/stale; integral cannot safely use old velocity")
        if self.method == "mit" and now - self.phase_start > .25:
            if now - self.f1_at > .25 or not self.f1_status & 1:
                raise RuntimeError("no fresh active-MIT feedback")

    def run_phase(self, method):
        self.disable()  # Explicit mode boundary, without assuming an internal PI reset.
        self.drain()
        self.method = method
        for op in (0xA2, 0xA1, 0xAE):
            self.request(op)
        integral = Integral()
        self.integral_nm = 0.
        self.phase_start = self.next_command = self.clock.monotonic()
        self.last_command = math.nan
        end = self.phase_start + DURATION_S
        self.active = True
        count, skipped_total, intervals, previous_send, next_slot = 0, 0, [], None, 0
        total_slots = round(DURATION_S * RATE_HZ)
        periods = {0xA2: 1 / 200., 0xA1: 1 / 100., 0xA3: 1 / 50., 0xAE: 1 / 10.}
        query_due = dict.fromkeys(periods, self.phase_start)
        print(f"{method.upper()}: +/-{self.args.amplitude:g} rad/s, {self.args.frequency:g} Hz, 10 s; sends 200 Hz")
        try:
            while self.clock.monotonic() < end:
                if self.stop_requested():
                    raise KeyboardInterrupt
                self.log.check()
                now = self.clock.monotonic()
                self.check_feedback(now)
                if next_slot < total_slots and now >= self.next_command:
                    phase_time = now - self.phase_start
                    target = self.args.amplitude * math.sin(2 * math.pi * self.args.frequency * phase_time)
                    extra = {}
                    if method == "mit":
                        error_integral = integral.update(target - self.velocity, now)
                        self.integral_nm = self.ki * error_integral
                        payload, extra = mit_payload(target, self.integral_nm, self.velocity, self.kd, self.ranges)
                        extra.update(integral_error_rad=error_integral, integral_torque_nm=self.integral_nm)
                    else:
                        payload, encoded = c1_payload(target)
                        extra["encoded_target_velocity_rad_s"] = encoded
                    skipped = max(0, int(phase_time * RATE_HZ) - next_slot)
                    last = self.send(payload, 0x501 if method == "mit" else 0x101, command_index=count,
                                     target_velocity_rad_s=target, target_sample_time_s=phase_time,
                                     scheduled_at=self.next_command, skipped_slots=skipped,
                                     **extra)
                    if previous_send is not None:
                        intervals.append(last - previous_send)
                    previous_send, count = last, count + 1
                    skipped_total += skipped
                    next_slot += skipped + 1
                    self.next_command = min(end, self.phase_start + next_slot / RATE_HZ)
                for op, period in periods.items():
                    now = self.clock.monotonic()
                    if now >= query_due[op] and now < min(self.next_command, end):
                        self.send(bytes([op]))
                        query_due[op] = now + period
                self.drain()
                deadline = min(end, self.next_command, *query_due.values())
                select.select([self.bus], [], [], max(0., min(.01, deadline - self.clock.monotonic())))
        finally:
            self.active = False
            self.phases.append(dict(method=method, duration_s=self.clock.monotonic() - self.phase_start,
                                    commands=count, skipped_slots=skipped_total,
                                    achieved_rate_hz=1 / (sum(intervals) / len(intervals)) if intervals else None,
                                    max_packet_interval_s=max(intervals) if intervals else None,
                                    final_integral_torque_nm=self.integral_nm))
        self.disable()

    def cleanup(self):
        self.cleaning, self.active = True, False
        try:
            self.disable()
        except Exception:
            # Retry an actual CF even if the first write/acknowledgement failed.
            self.disable()
        failures = []
        for original in reversed(self.saved):
            try:
                self.configure_watchdog(original)
            except Exception as error:
                failures.append(f"{original[0]:02X}: {error}")
        if failures:
            raise RuntimeError("disabled, but settings restoration not confirmed: " + "; ".join(failures))
        self.emit("event", note="CF disabled confirmed; original CD restored; B3 read-only")


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main(argv=None):
    args = parse_args(argv)
    methods = ("c1", "mit") if args.method == "both" else (args.method,)
    print(f"Rear ID 1: {', then '.join(methods)}; 10 s each, 200 Hz; "
          f"sine +/-{args.amplitude:g} rad/s at {args.frequency:g} Hz")
    if not args.execute:
        print("PREVIEW: no CAN/files. Add --execute. MIT needs explicit gain-unit/timebase options.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite an existing CSV/JSON")
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    report = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  gain_basis="user-selected; not verified by local protocol", source_sha256=hashlib.sha256(
                      Path(__file__).read_bytes()).hexdigest(), motor_id=1, encoder_sign=SIGN, torque_sign=SIGN,
                  packet_rate_hz=RATE_HZ, duration_per_method_s=DURATION_S, completed=False,
                  disabled_and_restored=False, error=None)
    io = bus = None
    try:
        with CsvLog(args.output) as log:
            print(f"CSV: {args.output}\nRear wheel airborne/free; frame supported; all other writers stopped.")
            try:
                bus = open_bus(args.interface)
                io = VelocityTest(args, bus, log, stopped.is_set)
                io.startup()
                for method in methods:
                    io.run_phase(method)
                report["completed"] = True
            except BaseException as error:
                report["execution_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                if io is not None:
                    try:
                        io.cleanup()
                        report["disabled_and_restored"] = True
                    except BaseException as error:
                        report["cleanup_error"] = f"{type(error).__name__}: {error}"
                        raise
    except BaseException as error:
        report.update(completed=False, error=f"{type(error).__name__}: {error}")
        print(f"STOPPED: {report['error']}; partial CSV retained", file=sys.stderr)
    finally:
        if bus is not None:
            bus.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if io is not None:
            report.update(raw_gains=io.raw_gains, torque_constant_nm_per_a=io.kt,
                          mit_kd_nm_s_per_rad=io.kd, integral_ki_nm_per_rad=io.ki,
                          original_settings=io.settings, applied_settings=io.applied_settings,
                          mit_encoding_ranges=io.ranges, phases=io.phases, zero_count=io.zero_count)
        with summary_path.open("x") as stream:
            json.dump(json_ready(report), stream, indent=2, allow_nan=False)
            stream.write("\n")
    if report["completed"]:
        print("Completed; CF disabled confirmed, original CD restored; B3 unchanged. Wheel may coast.")
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
