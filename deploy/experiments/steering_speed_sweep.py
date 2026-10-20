#!/usr/bin/env python3
"""Standalone bounded steering C1 speed-step sweep and raw feedback CSV.

PREVIEW by default; --execute opens CAN and creates an exclusive CSV. Support
the bike securely, FRONT WHEEL AIRBORNE, steering initially physically straight
and free of cables/stops throughout the chosen travel. Stop all other CAN
writers/recorders for this motor. Python stdlib only; transfer this file alone.

Default device 2 (host 102, feedback 002), command/encoder/torque signs -1,
gear 1, matching deploy's joint convention. C1 direction itself must be checked
at low speed. No motor-origin, gain, or firmware acceleration writes are made.
Startup A3 multi-turn count is the software zero; angle is NEVER pi-wrapped.

At each positive speed magnitude, run --cycles out-and-back traversals between
+/- --sweep-angle. First position at the negative endpoint; finish near zero.
EVERY speed command is a STEP (including zero stops), with NO host ramp. C1's
firmware B5 acceleration may still limit physical response: read/log, not change
it. The --settle period labels the beginning of each traversal for later
exclusion; there is no long fixed-speed hold that would wind up the steering.

Defaults: speeds 1,2,4,6,8,10,12 rad/s; endpoints +/-0.6 rad; hard abort +/-1.2 rad;
central analysis window +/-0.4 rad; three cycles/speed; one repeat. Position
checks, a velocity/age lookahead, stopped rests, leg timeouts and direction
checks supplement (NOT guarantee) safe travel. CF disables torque, not a brake.
Unmeasured stopping distance, scheduling stalls, cable loads and firmware
behavior still require physical supervision and an independent emergency stop.
High-speed strokes may contain only transients, not a steady friction plateau.
The 15-ms lookahead is a reaction guard, NOT a measured stopping-time allowance.
Long Pi stalls cannot be tolerated safely at these speeds; do not increase
timeouts just to suppress aborts without reassessing travel/stopping margins.

Temporary B3 ceiling: at most --current-limit-a (default 1 A), NEVER raises an
existing lower ceiling. Measured-current abort is more tolerant than the rear
script: 1.5 * ACTUAL B3 readback + .05 A (both configurable). This is a software
transient guard, NOT permission to exceed motor ratings. B3 write/readback
verification still allows only 1 mA DOWNWARD quantization; CD remains exact.
Save/verify B3/CD; restore only after fresh CF and AE confirm disabled. Crashes
can leave reduced B3/enabled CD. CD is a communication watchdog; query traffic
may keep it alive even without new C1 commands. Never rely on it alone.

C1 nominal 50 Hz plus immediate zero at a turnaround; A3/A2/A1 each 200 Hz,
AE 5 Hz, B0 1 Hz. Actual rates are logged, not guaranteed. Each TX/RX is a row
with raw bytes and cached values' individual ages. A1 * B0 estimates torque,
not a torque sensor. analysis_window marks central post-settle travel, NOT
verified steady motion; retain ripple, and inspect actual speed/at_target.

Examples:
  python3 steering_speed_sweep.py
  python3 steering_speed_sweep.py --execute --speeds .2 --cycles 1
  python3 steering_speed_sweep.py --execute --repeats 3
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
RADIANS_PER_COUNT = 2 * math.pi / 16384
LENGTHS = {0xA1: 5, 0xA2: 5, 0xA3: 7, 0xB0: 7, 0xB3: 5, 0xB5: 5, 0xAE: 8, 0xCD: 5, 0xCF: 8, 0xC1: 5}
FIELDS = (
    "record_type", "time_unix_s", "elapsed_s", "repeat", "step", "cycle", "leg", "phase", "phase_elapsed_s",
    "motor_id", "gear_ratio", "command_sign", "encoder_sign", "torque_sign", "target_position_rad",
    "plateau_speed_rad_s", "command_speed_rad_s", "encoded_command_rad_s", "command_native_rpm",
    "command_interval_s", "command_lateness_s", "can_id_hex", "opcode_hex", "payload_hex",
    "query_status", "query_round_trip_s", "reply_native_rpm", "reply_velocity_rad_s",
    "encoder_count", "encoder_single_turn_count", "zero_count", "position_rad", "velocity_rad_s",
    "current_a", "torque_estimate_nm", "torque_constant_nm_per_a", "firmware_acceleration_rad_s2",
    "position_sample_elapsed_s", "velocity_sample_elapsed_s", "current_sample_elapsed_s",
    "status_sample_elapsed_s", "b0_sample_elapsed_s", "position_age_s", "velocity_age_s",
    "current_age_s", "status_age_s", "b0_age_s", "at_target", "analysis_window",
    "bus_voltage_v", "bus_current_a", "bus_current_raw", "motor_temperature_c", "motor_mode", "fault_code",
    "current_limit_a", "current_guard_a", "watchdog_ms", "watchdog_enabled", "note",
)


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def speed_payload(args, speed):
    code = round(args.command_sign * args.gear_ratio * speed * 60 / (2 * math.pi) * 100)
    if not -(2**31) <= code < 2**31:
        raise ValueError("C1 target does not fit int32 0.01-rpm units")
    encoded = args.command_sign * code * .01 * 2 * math.pi / 60 / args.gear_ratio
    return b"\xc1" + struct.pack("<i", code), encoded, code * .01


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Open CAN and run; default is preview only.")
    mode.add_argument("--preview", action="store_true")
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--motor-id", type=lambda value: int(value, 0), default=2)
    parser.add_argument("--speeds", nargs="+", type=finite, default=[1., 2., 4., 6., 8., 10., 12.],
                        help="POSITIVE magnitudes in joint rad/s; both directions are automatic.")
    parser.add_argument("--cycles", type=int, default=3, help="Full out-and-back cycles per speed.")
    parser.add_argument("--repeats", type=int, default=1)
    options = (
        ("sweep-angle", .6, "Turnaround magnitude relative to physically straight startup, rad."),
        ("analysis-angle", .4, "Central post-settle CSV analysis-window magnitude, rad."),
        ("angle-limit", 1.2, "Hard angle abort, rad; must remain <=1.4, inside bike's +/-90 deg."),
        ("settle", .02, "Scheduled exclusion at start of each leg; NOT verified settling, NO command ramp."),
        ("rest", .3, "Continuous near-zero-speed rest at every endpoint, seconds."),
        ("stop-timeout", 2., "Maximum time for a zero command to stop and rest, seconds."),
        ("stopped-speed", .05, "Measured speed threshold for a stopped rest, rad/s."),
        ("guard-horizon", .015, "Reaction lookahead before hard boundary, seconds; NOT measured braking time."),
        ("command-rate", 50., "Nominal C1 Hz, plus immediate endpoint zero commands."),
        ("query-rate", 200., "Nominal A3/A2/A1 queries/s EACH."),
        ("status-rate", 5., "AE queries/s."), ("b0-rate", 1., "B0 queries/s."),
        ("query-timeout", .01, "Motion query deadline, seconds; failure stops test, no stale substitution."),
        ("feedback-timeout", .02, "Maximum A3/A2/A1 age during motion, seconds."),
        ("current-limit-a", 1., "Temporary B3 ceiling, A; never raises an existing lower limit."),
        ("current-guard-factor", 1.5, "Abort if abs(A1) exceeds factor * actual B3 + margin."),
        ("current-guard-margin-a", .05, "Extra measured-current guard allowance, A."),
        ("speed-limit", 16., "Measured overspeed abort, joint rad/s; must not exceed configured steering 23.9."),
        ("max-temperature", 60., "Conservative temperature abort, C, not a motor rating."),
        ("gear-ratio", 1., "Protocol/joint ratio; verified deployment uses 1, not another 10:1."),
    )
    for name, default, help_text in options:
        parser.add_argument(f"--{name}", type=finite, default=default, help=help_text)
    parser.add_argument("--watchdog-ms", type=int, default=500)
    for name in ("command", "encoder", "torque"):
        parser.add_argument(f"--{name}-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--output", type=Path, help="Exclusive CSV; default logs/<timestamp>_steering_speed_sweep.csv.")
    args = parser.parse_args(argv)
    if any(getattr(args, name.replace("-", "_")) <= 0 for name, _, _ in options
           if name not in ("settle", "current-guard-margin-a")):
        parser.error("rates, durations, angles, limits and gear ratio must be positive")
    if args.settle < 0 or args.current_guard_margin_a < 0 or args.current_guard_factor < 1:
        parser.error("settle/margin must be nonnegative; current guard factor must be >=1")
    if not 1 <= args.motor_id <= 254 or min(args.cycles, args.repeats) < 1:
        parser.error("motor ID must be 1..254; cycles/repeats must be >=1")
    if not all(args.stopped_speed < value <= args.speed_limit for value in args.speeds):
        parser.error("speeds must be positive magnitudes, above stopped-speed and <=speed-limit")
    if args.speed_limit > 23.9:
        parser.error("speed-limit must not exceed the bike's configured steering speed limit of 23.9 rad/s")
    if not 0 < args.analysis_angle < args.sweep_angle < args.angle_limit <= 1.4:
        parser.error("require 0 < analysis-angle < sweep-angle < angle-limit <=1.4 rad")
    # Endpoint zero bypasses the periodic C1 tick, so use the position-poll period.
    # Also bound travel during the allowed feedback age, not just nominal polling.
    reaction_budget = max(args.query_timeout+1/args.query_rate, args.feedback_timeout)
    lookahead = args.speed_limit*(reaction_budget+args.guard_horizon)
    if args.sweep_angle + lookahead >= args.angle_limit:
        parser.error("insufficient endpoint margin for speed/feedback/query/lookahead; reduce speed or travel")
    if args.query_timeout >= args.feedback_timeout or 1/args.query_rate >= args.feedback_timeout:
        parser.error("feedback-timeout must exceed query-timeout and nominal feedback period")
    if max(args.command_rate, args.query_rate) > 500 or max(args.status_rate, args.b0_rate) > args.query_rate:
        parser.error("command/query rates <=500 Hz; AE/B0 rates <=query-rate")
    if not 100 <= args.watchdog_ms <= 1000 or 3000/args.command_rate > args.watchdog_ms:
        parser.error("watchdog must be 100..1000 ms and cover at least three nominal C1 periods")
    if not .001 <= args.current_limit_a <= 4294967.295 or args.stop_timeout <= args.rest:
        parser.error("current ceiling must fit uint32 mA; stop-timeout must exceed rest")
    if not math.isfinite(args.current_guard_factor*args.current_limit_a+args.current_guard_margin_a):
        parser.error("current guard must remain finite")
    try:
        for speed in args.speeds:
            for direction in (-1, 1):
                _, encoded, _ = speed_payload(args, direction*speed)
                if encoded == 0 or abs(encoded) > args.speed_limit:
                    raise ValueError("quantized C1 command must be nonzero and within speed-limit; adjust gear/speed")
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    if args.output is None:
        args.output = Path(__file__).resolve().parent / "logs" / (
            f"{datetime.now():%Y%m%d_%H%M%S_%f}_steering_speed_sweep.csv"
        )
    return args


@dataclass(frozen=True)
class Leg:
    kind: str
    speed: float
    destination: float
    repeat: int = 0
    step: int = 0
    cycle: int = 0


def make_plan(args):
    positioning_speed = min(.4, min(args.speeds))
    plan = [Leg("positioning", -positioning_speed, -args.sweep_angle)]
    for repeat in range(1, args.repeats + 1):
        for step, speed in enumerate(args.speeds, 1):
            for cycle in range(1, args.cycles + 1):
                plan.extend((Leg("traverse", speed, args.sweep_angle, repeat, step, cycle),
                             Leg("traverse", -speed, -args.sweep_angle, repeat, step, cycle)))
    plan.append(Leg("return", positioning_speed, 0.))
    return plan


def setting_matches(requested, reported):
    if not requested or not reported or requested[0] != reported[0]:
        return False
    if requested[0] != 0xB3:
        return requested == reported
    if len(requested) != 5 or len(reported) != 5:
        return False
    wanted, actual = struct.unpack("<I", requested[1:])[0], struct.unpack("<I", reported[1:])[0]
    return actual > 0 and 0 <= wanted - actual <= 1


class CsvLog:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", newline="")
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
        ids = (args.motor_id, 0x100 | args.motor_id, 0x400 | args.motor_id, 0x500 | args.motor_id,
               0, 0xFF, 0x400, 0x4FF)
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                       b"".join(struct.pack("=II", cid, CAN_MASK) for cid in ids))
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
        bus.setsockopt(socket.SOL_CAN_RAW, getattr(socket, "CAN_RAW_ERR_FILTER", 2),
                       struct.pack("=I", 0x1FFFFFFF))
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
        self.motion = self.stopping = self.complete = False
        self.phase, self.phase_started, self.leg_index = "startup", self.start, -1
        self.leg, self.plan, self.saved = None, make_plan(args), []
        self.position = self.velocity = self.current = self.constant = self.acceleration = math.nan
        self.count = self.single_count = self.zero_count = None
        self.times = dict.fromkeys(("position", "velocity", "current", "status", "b0"), math.nan)
        self.status = {}
        self.command_speed = self.encoded_speed = self.command_rpm = 0.
        self.last_command, self.next_command = math.nan, math.inf
        self.current_limit = self.watchdog_ms = math.nan
        self.watchdog_enabled = self.log_error = self.direction_error = self.stationary_since = None

    @property
    def current_guard(self):
        return self.args.current_guard_factor * self.current_limit + self.args.current_guard_margin_a

    def emit(self, kind, stamp=None, **values):
        now = self.clock.monotonic() if stamp is None else stamp
        args, leg = self.args, self.leg
        row = dict(record_type=kind, time_unix_s=self.clock.time(), elapsed_s=now-self.start,
                   repeat=leg.repeat if leg else 0, step=leg.step if leg else 0, cycle=leg.cycle if leg else 0,
                   leg=self.leg_index, phase=self.phase, phase_elapsed_s=now-self.phase_started,
                   motor_id=args.motor_id, gear_ratio=args.gear_ratio, command_sign=args.command_sign,
                   encoder_sign=args.encoder_sign, torque_sign=args.torque_sign,
                   target_position_rad=leg.destination if leg else math.nan,
                   plateau_speed_rad_s=leg.speed if leg else 0., command_speed_rad_s=self.command_speed,
                   encoded_command_rad_s=self.encoded_speed, command_native_rpm=self.command_rpm,
                   encoder_count=self.count, encoder_single_turn_count=self.single_count, zero_count=self.zero_count,
                   position_rad=self.position, velocity_rad_s=self.velocity, current_a=self.current,
                   torque_estimate_nm=args.torque_sign*self.current*self.constant,
                   torque_constant_nm_per_a=self.constant, firmware_acceleration_rad_s2=self.acceleration,
                   current_limit_a=self.current_limit, current_guard_a=self.current_guard,
                   watchdog_ms=self.watchdog_ms, watchdog_enabled=self.watchdog_enabled, **self.status)
        for name, sample_time in self.times.items():
            row[f"{name}_sample_elapsed_s"] = sample_time-self.start
            row[f"{name}_age_s"] = now-sample_time
        row["at_target"] = abs(self.velocity-self.encoded_speed) <= max(.05, .05*abs(self.encoded_speed))
        row["analysis_window"] = self.phase == "hold" and abs(self.position) <= args.analysis_angle
        row.update(values)
        try:
            self.log.emit(row)
        except Exception as error:
            self.log_error = error
            if not self.stopping:
                raise

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
                    raise RuntimeError("CAN receive overflow")
        cid, length, padded = CAN_FRAME.unpack(frame)
        if cid & 0x20000000:
            raise RuntimeError(f"CAN error frame {cid:08X}")
        data = padded[:length]
        details = dict(can_id_hex=f"{cid:03X}", opcode_hex=f"{data[0]:02X}" if data else "",
                       payload_hex=data.hex().upper())
        try:
            if cid != self.args.motor_id or flags & socket.MSG_DONTROUTE:
                raise RuntimeError("other selected-motor/broadcast CAN writer observed; stop all other writers")
            if length > 8 or not data or LENGTHS.get(data[0]) != length:
                raise RuntimeError("unexpected/malformed selected-motor reply")
            details.update(self.decode(data, now))
        except (RuntimeError, ValueError) as error:
            self.emit("rx_error", now, note=str(error), **details)
            raise
        self.emit("rx", now, **details)
        return data, now

    def decode(self, data, now):
        op, extra, args = data[0], {}, self.args
        if op == 0xA3:
            self.single_count, self.count = struct.unpack("<Hi", data[1:7])
            if self.zero_count is None:
                self.zero_count = self.count
            difference = (self.count-self.zero_count+2**31) % 2**32 - 2**31
            self.position = args.encoder_sign*difference*RADIANS_PER_COUNT/args.gear_ratio
            self.times["position"] = now
            if self.motion and abs(self.position) >= args.angle_limit:
                raise RuntimeError(f"hard steering position limit: {self.position:+.4f} rad")
        elif op in (0xA2, 0xC1):
            rpm = struct.unpack("<i", data[1:])[0]*.01
            speed = args.encoder_sign*rpm*2*math.pi/60/args.gear_ratio
            extra.update(reply_native_rpm=rpm, reply_velocity_rad_s=speed)
            if not self.stopping and abs(speed) > args.speed_limit:
                raise RuntimeError(f"overspeed: {speed:+.3f} rad/s")
            if op == 0xA2:
                self.velocity, self.times["velocity"] = speed, now
        elif op == 0xA1:
            self.current, self.times["current"] = struct.unpack("<i", data[1:])[0]*.001, now
            if not self.stopping and math.isfinite(self.current_limit) and abs(self.current) > self.current_guard:
                raise RuntimeError(f"measured current {self.current:+.3f} A exceeds guard {self.current_guard:.3f} A")
        elif op == 0xB0:
            constant = struct.unpack("<f", data[2:6])[0]
            if not math.isfinite(constant) or not .01 <= constant <= 10:
                raise RuntimeError("invalid B0 torque constant")
            if self.motion and not math.isclose(constant, self.constant, rel_tol=1e-5):
                raise RuntimeError("B0 torque constant changed")
            self.constant, self.times["b0"] = constant, now
        elif op == 0xB3:
            self.current_limit = struct.unpack("<I", data[1:])[0]*.001
        elif op == 0xB5:
            self.acceleration = struct.unpack("<I", data[1:])[0]*.01*2*math.pi/60/args.gear_ratio
        elif op == 0xCD:
            self.watchdog_enabled, self.watchdog_ms = data[1], struct.unpack("<H", data[2:4])[0]
        elif op in (0xAE, 0xCF):
            voltage, bus_current = struct.unpack("<HH", data[1:5])
            self.status = dict(bus_voltage_v=voltage*.01, bus_current_a=bus_current*.01,
                               bus_current_raw=bus_current, motor_temperature_c=data[5],
                               motor_mode=data[6], fault_code=data[7])
            self.times["status"] = now
            if not self.stopping:
                if data[7] or data[5] >= args.max_temperature or voltage == 0:
                    raise RuntimeError(f"unsafe motor status: {self.status}")
                if self.motion and data[6] != 3:
                    raise RuntimeError(f"motor left C1 speed mode: {data[6]}")
        return extra

    def phase_change(self, phase, now, note):
        self.phase, self.phase_started = phase, now
        self.emit("event", now, note=note)

    def start_leg(self, now):
        self.leg_index += 1
        if self.leg_index == len(self.plan):
            self.complete = True
            return
        self.leg = self.plan[self.leg_index]
        self.leg_started = now
        distance = abs(self.leg.destination-self.position)
        self.leg_deadline = now+max(2., 3*distance/abs(self.leg.speed)+1.)
        self.stationary_since = self.direction_error = None
        self.command_speed = self.leg.speed
        self.next_command = now
        phase = "settle" if self.leg.kind == "traverse" else self.leg.kind
        self.phase_change(phase, now, "direct C1 speed step; no host acceleration ramp")

    def service(self):
        if self.stopping:
            return
        if self.stop_requested():
            raise KeyboardInterrupt
        if not self.motion or self.complete:
            return
        now, args = self.clock.monotonic(), self.args
        for name in ("position", "velocity", "current"):
            if not math.isfinite(self.times[name]) or now-self.times[name] > args.feedback_timeout:
                raise RuntimeError(f"stale {name} feedback")
        if now-self.times["status"] > max(3/args.status_rate, args.feedback_timeout):
            raise RuntimeError("stale AE status")
        if now-self.last_command > min(max(.1, 3/args.command_rate), self.watchdog_ms/2000):
            raise RuntimeError("C1 command scheduling gap exceeded safety deadline")
        projected = self.position+self.velocity*(now-self.times["position"]+args.guard_horizon)
        if abs(self.position) >= args.angle_limit or abs(projected) >= args.angle_limit:
            raise RuntimeError("steering hard/predictive position guard; disabling, not reversing")
        if self.phase in ("stopping", "rest"):
            if now-self.stop_started > args.stop_timeout:
                raise RuntimeError("steering did not stop/rest before deadline")
            if abs(self.velocity) <= args.stopped_speed:
                if self.stationary_since is None:
                    self.stationary_since = now
                    self.phase_change("rest", now, "C1 zero held; waiting for continuous stopped rest")
                if now-self.stationary_since >= args.rest:
                    self.start_leg(now)
                    if self.complete:
                        return
            else:
                self.stationary_since = None
                if self.phase == "rest":
                    self.phase_change("stopping", now, "speed left stopped band; restarting rest")
        else:
            if now >= self.leg_deadline:
                raise RuntimeError("steering leg timeout; inspect tracking/current ceiling/B5")
            direction = math.copysign(1., self.leg.speed)
            reached = direction*(self.position-self.leg.destination) >= 0
            if reached:
                self.command_speed, self.next_command, self.stop_started = 0., now, now
                self.stationary_since = self.direction_error = None
                self.phase_change("stopping", now, "endpoint reached; immediate C1 zero STEP")
            elif self.phase == "settle" and now-self.leg_started >= args.settle:
                self.phase_change("hold", now, "post-settle travel, NOT guaranteed steady motion")
        if now >= self.next_command:
            data, self.encoded_speed, self.command_rpm = speed_payload(args, self.command_speed)
            self.last_command = self.send(data, command_interval_s=now-self.last_command,
                                          command_lateness_s=max(0., now-self.next_command))
            self.next_command = self.last_command+1/args.command_rate
        wrong_direction = self.encoded_speed*self.velocity < 0 and abs(self.velocity) > args.stopped_speed
        if wrong_direction:
            if self.direction_error is None:
                self.direction_error = now
            elif now-self.direction_error > .15:
                raise RuntimeError("C1/encoder direction mismatch; verify signs at low speed")
        else:
            self.direction_error = None

    def wait(self, duration):
        select.select([self.bus], [], [], max(0., duration))

    def request(self, opcode, payload=None, timeout=None):
        # Disable BEFORE draining unsafe/backlogged replies during shutdown.
        if opcode != 0xCF:
            for _ in range(64):
                try:
                    self.receive()
                except BlockingIOError:
                    break
            else:
                raise RuntimeError("CAN receive backlog; query cannot be correlated")
        self.service()
        sent = self.send(bytes([opcode]) if payload is None else payload)
        deadline = sent+(self.args.query_timeout if timeout is None else timeout)
        while True:
            now = self.clock.monotonic()
            if now >= deadline:
                self.emit("query", opcode_hex=f"{opcode:02X}", query_status="timeout")
                raise TimeoutError(f"no timely {opcode:02X} reply; no stale substitution")
            self.service()
            try:
                data, received = self.receive()
            except BlockingIOError:
                wait_to = min(deadline, self.next_command) if self.motion and not self.complete else deadline
                self.wait(wait_to-now)
                continue
            if data[0] == opcode and received <= deadline:
                self.emit("query", received, opcode_hex=f"{opcode:02X}", query_status="received",
                          query_round_trip_s=received-sent)
                return data

    def verify_setting(self, requested, reported, stage):
        if not setting_matches(requested, reported):
            raise RuntimeError(f"{stage} mismatch: requested={requested.hex().upper()} "
                               f"reported={reported.hex().upper()}")
        if requested != reported:
            self.emit("event", note=f"{stage}: accepted <=1 mA downward B3 quantization")

    def startup(self):
        self.emit("event", note=json.dumps(vars(self.args), default=str, sort_keys=True))
        if self.request(0xCF, timeout=.25)[6] != 0:
            raise RuntimeError("CF did not confirm disabled steering")
        for opcode in (0xB0, 0xB5):
            self.request(opcode, timeout=.25)
        original_b3, original_cd = self.request(0xB3, timeout=.25), self.request(0xCD, timeout=.25)
        old_limit = struct.unpack("<I", original_b3[1:])[0]
        if old_limit == 0 or original_cd[1] not in (0, 1):
            raise RuntimeError("invalid B3/CD configuration")
        ceiling = min(old_limit, math.floor(self.args.current_limit_a*1000))
        old_ms = struct.unpack("<H", original_cd[2:4])[0]
        self.watchdog_ms = min(old_ms, self.args.watchdog_ms) if original_cd[1] else self.args.watchdog_ms
        if self.watchdog_ms < 3000/self.args.command_rate:
            raise RuntimeError("existing watchdog too short; no settings changed")
        settings = ((original_b3, b"\xb3"+struct.pack("<I", ceiling)),
                    (original_cd, b"\xcd\x01"+struct.pack("<H", self.watchdog_ms)+original_cd[4:]))
        for original, requested in settings:
            if original != requested:
                self.saved.append(original)
                self.verify_setting(requested, self.request(requested[0], requested, timeout=.25), "setting ack")
            self.verify_setting(requested, self.request(requested[0], timeout=.25), "setting readback")
        for opcode in (0xAE, 0xA1, 0xA2, 0xA3):
            self.request(opcode, timeout=.25)
        if self.status["motor_mode"] != 0 or abs(self.velocity) > self.args.stopped_speed:
            raise RuntimeError("steering not disabled and stationary at startup")
        print(f"Verified steering {self.args.motor_id}: zero={self.zero_count} counts, Kt={self.constant:g} Nm/A, "
              f"B3={self.current_limit:g} A, current guard={self.current_guard:g} A, "
              f"B5={self.acceleration:g} rad/s^2 (unchanged)", flush=True)
        if self.acceleration > 0:
            # Conditional ideal constant-acceleration calculation, not an identified motor model.
            limited = [v for v in self.args.speeds if v*v/(2*self.acceleration) > 2*self.args.sweep_angle]
            if limited:
                print(f"WARNING: if firmware follows the read B5 acceleration, targets {limited} rad/s "
                      "need more than the full stroke to reach from rest. Inspect actual feedback; "
                      "do not treat these legs as steady-speed friction measurements.", flush=True)

    def run(self):
        now = self.clock.monotonic()
        self.motion, self.last_command = True, now
        self.start_leg(now)
        periods = {0xA3: 1/self.args.query_rate, 0xA2: 1/self.args.query_rate,
                   0xA1: 1/self.args.query_rate, 0xAE: 1/self.args.status_rate, 0xB0: 1/self.args.b0_rate}
        due, next_flush = dict.fromkeys(periods, now), now+1
        while not self.complete:
            self.service()
            if self.complete:
                break
            now = self.clock.monotonic()
            opcode = min(due, key=due.get)
            if now >= due[opcode]:
                self.request(opcode)
                due[opcode] = self.clock.monotonic()+periods[opcode]
            else:
                try:
                    self.receive()
                except BlockingIOError:
                    self.wait(min(due[opcode], self.next_command)-now)
            if self.clock.monotonic() >= next_flush:
                self.log.file.flush()
                next_flush = self.clock.monotonic()+1
        self.motion = False
        self.phase_change("complete", self.clock.monotonic(), "sweep complete; CF disable next")

    def shutdown(self):
        self.motion, self.stopping, self.phase = False, True, "shutdown"
        confirmed = False
        for _ in range(3):
            try:
                if self.request(0xCF, timeout=.15)[6] == 0 and self.request(0xAE, timeout=.15)[6] == 0:
                    confirmed = True
                    break
            except (OSError, RuntimeError, ValueError):
                pass
        if not confirmed:
            print("ERROR: CF disable NOT confirmed; safety settings retained. Check motor!", file=sys.stderr)
            return False
        try:
            for original in self.saved:
                self.verify_setting(original, self.request(original[0], original, timeout=.15), "restore ack")
                self.verify_setting(original, self.request(original[0], timeout=.15), "restore readback")
            if self.request(0xCF, timeout=.15)[6] != 0:
                raise RuntimeError("final disable not confirmed")
            print("Steering disabled; B3/CD restored/verified (B3 may be <=1 mA lower). Steering may coast.",
                  flush=True)
            self.emit("event", note="CF disabled; original B3/CD restored/verified")
            if self.log_error is not None:
                print(f"ERROR: CSV incomplete: {self.log_error}", file=sys.stderr)
                return False
            return True
        except (OSError, RuntimeError, ValueError) as error:
            try:
                self.bus.send(CAN_FRAME.pack(0x100 | self.args.motor_id, 1, b"\xcf"))
            except OSError:
                pass
            print(f"ERROR: shutdown/restore incomplete: {error}; inspect motor/settings", file=sys.stderr)
            return False


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    print(f"Steering ID {args.motor_id}: host {0x100 | args.motor_id:03X}, feedback {args.motor_id:03X}; "
          f"speeds +/-{args.speeds} rad/s; {args.cycles} cycles x {args.repeats} repeats; "
          f"endpoints +/-{args.sweep_angle:g}, hard limit +/-{args.angle_limit:g} rad")
    if not args.execute:
        print("PREVIEW ONLY: no CAN, CSV or directories opened. Add --execute to run.")
        for index, leg in enumerate(plan):
            print(f"leg {index:3}: {leg.kind:11} repeat={leg.repeat} step={leg.step} cycle={leg.cycle} "
                  f"STEP speed {leg.speed:+g} rad/s -> angle {leg.destination:+g} rad -> zero/rest")
        print("No host ramp; B5 remains unchanged. Start physically straight; front wheel airborne.")
        return 0
    stop = [False]

    def interrupt(signum, frame):
        stop[0] = True

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with closing(CsvLog(args.output)) as log, closing(open_bus(args)) as bus:
            runner = Sweep(args, bus, log, lambda: stop[0])
            print(f"CSV: {args.output}\nFRONT WHEEL AIRBORNE, straight startup; all other motor writers stopped.",
                  flush=True)
            try:
                runner.startup()
                runner.run()
            finally:
                if not runner.shutdown() and sys.exc_info()[0] is None:
                    raise RuntimeError("shutdown/restoration not fully confirmed")
    except KeyboardInterrupt:
        print("Interrupted; partial CSV retained.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}; partial CSV retained if created.", file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
