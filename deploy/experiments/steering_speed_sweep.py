#!/usr/bin/env python3
"""Standalone steering C1 timed speed steps and feedback CSV; Python stdlib only.

PREVIEW is the default. --execute opens CAN and creates an exclusive CSV.
Device 2, host 102/reply 002, signs -1/-1/-1 and gear 1 match deployment.
Speeds are SIGNED joint rad/s, in the exact order supplied (like the rear
sweep). Default: 1,2,4,6,8,10,12 followed by -1,-2,-4,-6,-8,-10,-12.

Rear-sweep timing without ramps: initial zero rest, then for every speed:
STEP to target -> timed settle -> timed hold -> STEP to zero -> timed rest.
Defaults match rear timing: settle 5 s, hold 10 s, rest 2 s. Neither settle nor
rest waits for feedback conditions. There is NO positioning leg or return to
an angle. No software angle/speed/current/temperature/fault/freshness guard,
target clipping, or acceleration ramp is implemented. Hardware firmware limits
remain as configured: no B3/CD/B5/gain/origin writes, and no watchdog is enabled.
Only wire-format and CLI validity are checked. CAN/CSV I/O errors still end
the program; cleanup attempts CF disable on completion, Ctrl-C and SIGTERM.

WARNING: unbounded steering rotation can hit stops or twist wiring. At 12 rad/s,
the default 15 s nonzero interval requests 180 rad (28.6 turns) if followed.
Front wheel airborne is NOT proof that multi-turn steering is mechanically
safe. Choose hold/settle durations appropriate to your setup, inspect preview,
stop other CAN writers and keep an independent emergency stop available.
CF removes torque; it is not a mechanical brake. A crash/SIGKILL/power failure
may leave the last speed active, depending on existing firmware settings.

C1 nominal 50 Hz; A3/A2/A1 each 100 Hz, AE 5 Hz, B0 1 Hz; B5 read once at
startup. Replies are collected asynchronously: absent/delayed replies do not
abort or delay the timed command sequence. No retry/response-deadline guard.
CSV records each TX/RX and separate cached-value ages; NaN means unavailable.
Use record_type=rx and opcode A1/A2/A3 for independent fresh measurements.
C1 replies expose their own speed as reply_velocity_rad_s, not a fresh A2 row.
First A3 multi-turn count is the software zero; position is NOT pi-wrapped.
A1*B0 torque is a signed current-based estimate, not a shaft torque sensor.
Firmware acceleration is read/logged, not changed; speed steps do not guarantee
instantaneous physical speed or steady motion during the scheduled hold.

Examples:
  python3 steering_speed_sweep.py --preview
  python3 steering_speed_sweep.py --execute --speeds 1 -1 --settle 0 --hold .2 --rest .5
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
SO_RXQ_OVFL = getattr(socket, "SO_RXQ_OVFL", 40)
RADIANS_PER_COUNT = 2*math.pi/16384
LENGTHS = {0xA1: 5, 0xA2: 5, 0xA3: 7, 0xB0: 7, 0xB5: 5, 0xAE: 8, 0xCF: 8, 0xC1: 5}
FIELDS = (
    "record_type", "time_unix_s", "elapsed_s", "repeat", "step", "phase", "phase_elapsed_s",
    "motor_id", "gear_ratio", "command_sign", "encoder_sign", "torque_sign", "plateau_speed_rad_s",
    "command_speed_rad_s", "encoded_command_rad_s", "command_native_rpm", "command_interval_s",
    "command_lateness_s", "can_id_hex", "opcode_hex", "payload_hex", "reply_native_rpm", "reply_velocity_rad_s",
    "encoder_count", "encoder_single_turn_count", "zero_count", "position_rad", "velocity_rad_s", "current_a",
    "torque_estimate_nm", "torque_constant_nm_per_a", "firmware_acceleration_rad_s2",
    "position_sample_elapsed_s", "velocity_sample_elapsed_s", "current_sample_elapsed_s",
    "status_sample_elapsed_s", "b0_sample_elapsed_s", "position_age_s", "velocity_age_s", "current_age_s",
    "status_age_s", "b0_age_s", "bus_voltage_v", "bus_current_a", "bus_current_raw", "motor_temperature_c",
    "motor_mode", "fault_code", "note",
)


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def speed_payload(args, speed):
    native_rpm = args.command_sign*args.gear_ratio*speed*60/(2*math.pi)
    if not math.isfinite(native_rpm):
        raise ValueError("C1 conversion is not finite")
    code = round(native_rpm*100)
    if not -(2**31) <= code < 2**31:
        raise ValueError("C1 target does not fit signed int32 0.01-rpm wire format; not clipped")
    encoded = args.command_sign*code*.01*2*math.pi/60/args.gear_ratio
    return b"\xc1"+struct.pack("<i", code), encoded, code*.01


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Open CAN and run; default is preview only.")
    mode.add_argument("--preview", action="store_true")
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--motor-id", type=lambda value: int(value, 0), default=2)
    parser.add_argument("--speeds", nargs="+", type=finite,
                        default=[1., 2., 4., 6., 8., 10., 12., -1., -2., -4., -6., -8., -10., -12.],
                        help="SIGNED joint rad/s, sent in supplied order. No automatic sign expansion or clipping.")
    parser.add_argument("--repeats", type=int, default=1)
    for name, default, help_text in (
        ("settle", 5., "Timed interval at target before hold; does not wait for feedback, seconds."),
        ("hold", 10., "Timed target hold, seconds."),
        ("rest", 2., "Timed C1 zero before/between speeds; no stopped-state check, seconds."),
        ("command-rate", 50., "Nominal C1 Hz, plus immediate steps at phase boundaries."),
        ("query-rate", 100., "Nominal A3/A2/A1 requests/s EACH."),
        ("status-rate", 5., "Nominal AE requests/s."),
        ("b0-rate", 1., "Nominal B0 requests/s."),
        ("gear-ratio", 1., "Protocol/joint encoder and speed ratio; deployment uses 1."),
    ):
        parser.add_argument(f"--{name}", type=finite, default=default, help=help_text)
    for name in ("command", "encoder", "torque"):
        parser.add_argument(f"--{name}-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--output", type=Path, help="Exclusive CSV; default logs/<timestamp>_steering_speed_sweep.csv.")
    args = parser.parse_args(argv)
    if args.settle < 0 or args.rest < 0 or args.hold <= 0 or args.repeats < 1:
        parser.error("settle/rest must be nonnegative; hold and repeats must be positive")
    if min(args.command_rate, args.query_rate, args.status_rate, args.b0_rate, args.gear_ratio) <= 0:
        parser.error("rates and gear ratio must be positive")
    if not 1 <= args.motor_id <= 254:
        parser.error("motor ID must be 1..254")
    try:
        for speed in args.speeds:
            speed_payload(args, speed)
        make_plan(args)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    if args.output is None:
        args.output = Path(__file__).resolve().parent/"logs"/(
            f"{datetime.now():%Y%m%d_%H%M%S_%f}_steering_speed_sweep.csv"
        )
    return args


@dataclass(frozen=True)
class Phase:
    start: float
    duration: float
    name: str
    speed: float
    plateau: float
    repeat: int
    step: int


def make_plan(args):
    phases, start = [], 0.

    def add(name, duration, speed, plateau, repeat, step):
        nonlocal start
        if duration > 0:
            phases.append(Phase(start, duration, name, speed, plateau, repeat, step))
            start += duration

    add("rest", args.rest, 0., 0., 0, 0)
    for repeat in range(1, args.repeats+1):
        for step, speed in enumerate(args.speeds, 1):
            add("settle", args.settle, speed, speed, repeat, step)
            add("hold", args.hold, speed, speed, repeat, step)
            add("rest", args.rest, 0., speed, repeat, step)
    if not math.isfinite(start):
        raise ValueError("planned duration overflow")
    return phases


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
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, struct.pack("=II", args.motor_id, 0xC00007FF))
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
        self.plan, self.phase_index = make_plan(args), -1
        self.phase, self.phase_started, self.repeat, self.step = "startup", self.start, 0, 0
        self.plateau = self.command_speed = self.encoded_speed = self.command_rpm = 0.
        self.position = self.velocity = self.current = self.constant = self.acceleration = math.nan
        self.count = self.single_count = self.zero_count = None
        self.times = dict.fromkeys(("position", "velocity", "current", "status", "b0"), math.nan)
        self.status = {}
        self.last_command, self.last_requested_speed = math.nan, None
        self.shutting_down, self.disable_ack, self.log_error = False, False, None

    def emit(self, kind, stamp=None, **values):
        now = self.clock.monotonic() if stamp is None else stamp
        args = self.args
        row = dict(record_type=kind, time_unix_s=self.clock.time(), elapsed_s=now-self.start,
                   repeat=self.repeat, step=self.step, phase=self.phase, phase_elapsed_s=now-self.phase_started,
                   motor_id=args.motor_id, gear_ratio=args.gear_ratio, command_sign=args.command_sign,
                   encoder_sign=args.encoder_sign, torque_sign=args.torque_sign, plateau_speed_rad_s=self.plateau,
                   command_speed_rad_s=self.command_speed, encoded_command_rad_s=self.encoded_speed,
                   command_native_rpm=self.command_rpm, encoder_count=self.count,
                   encoder_single_turn_count=self.single_count, zero_count=self.zero_count,
                   position_rad=self.position, velocity_rad_s=self.velocity, current_a=self.current,
                   torque_estimate_nm=args.torque_sign*self.current*self.constant,
                   torque_constant_nm_per_a=self.constant, firmware_acceleration_rad_s2=self.acceleration,
                   **self.status)
        for name, sample_time in self.times.items():
            row[f"{name}_sample_elapsed_s"] = sample_time-self.start
            row[f"{name}_age_s"] = now-sample_time
        row.update(values)
        try:
            self.log.emit(row)
        except Exception as error:
            self.log_error = error
            if not self.shutting_down:
                raise

    def send(self, payload, **values):
        stamp = self.clock.monotonic()
        if self.bus.send(CAN_FRAME.pack(0x100 | self.args.motor_id, len(payload), payload)) != CAN_FRAME.size:
            raise OSError("incomplete CAN send")
        self.emit("command" if payload[0] == 0xC1 else "tx", stamp,
                  can_id_hex=f"{0x100 | self.args.motor_id:03X}", opcode_hex=f"{payload[0]:02X}",
                  payload_hex=payload.hex().upper(), **values)
        return stamp

    def decode(self, data, now):
        op, args, extra = data[0], self.args, {}
        if LENGTHS.get(op) != len(data):
            return {"note": "unsupported opcode/DLC; raw reply retained, cached state not updated"}
        if op == 0xA3:
            self.single_count, self.count = struct.unpack("<Hi", data[1:])
            if self.zero_count is None:
                self.zero_count = self.count
            difference = (self.count-self.zero_count+2**31) % 2**32-2**31
            self.position = args.encoder_sign*difference*RADIANS_PER_COUNT/args.gear_ratio
            self.times["position"] = now
        elif op in (0xA2, 0xC1):
            rpm = struct.unpack("<i", data[1:])[0]*.01
            velocity = args.encoder_sign*rpm*2*math.pi/60/args.gear_ratio
            extra.update(reply_native_rpm=rpm, reply_velocity_rad_s=velocity)
            if op == 0xA2:
                self.velocity, self.times["velocity"] = velocity, now
        elif op == 0xA1:
            self.current, self.times["current"] = struct.unpack("<i", data[1:])[0]*.001, now
        elif op == 0xB0:
            value = struct.unpack("<f", data[2:6])[0]
            self.constant = value if math.isfinite(value) and value > 0 else math.nan
            self.times["b0"] = now
        elif op == 0xB5:
            self.acceleration = struct.unpack("<I", data[1:])[0]*.01*2*math.pi/60/args.gear_ratio
        elif op in (0xAE, 0xCF):
            voltage, current = struct.unpack("<HH", data[1:5])
            self.status = dict(bus_voltage_v=voltage*.01, bus_current_a=current*.01, bus_current_raw=current,
                               motor_temperature_c=data[5], motor_mode=data[6], fault_code=data[7])
            self.times["status"] = now
        return extra

    def receive(self):
        frame, ancillary, flags, _ = self.bus.recvmsg(CAN_FRAME.size, socket.CMSG_SPACE(4))
        now = self.clock.monotonic()
        if len(frame) != CAN_FRAME.size:
            raise OSError("invalid SocketCAN frame size")
        cid, length, padded = CAN_FRAME.unpack(frame)
        data = padded[:min(length, 8)]
        details = dict(can_id_hex=f"{cid:03X}", opcode_hex=f"{data[0]:02X}" if data else "",
                       payload_hex=data.hex().upper())
        for level, kind, value in ancillary:
            if level == socket.SOL_SOCKET and kind == SO_RXQ_OVFL and len(value) == 4:
                details["note"] = f"kernel cumulative RX drops={struct.unpack('=I', value)[0]}"
        if cid & 0x20000000:
            self.emit("can_error", now, **details)
        elif flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or length > 8 or not data:
            self.emit("rx_error", now, note="truncated/empty CAN reply", **{k: v for k, v in details.items()
                                                                          if k != "note"})
        elif cid == self.args.motor_id:
            details.update(self.decode(data, now))
            self.emit("rx", now, **details)
        else:
            self.emit("observed_can", now, **details)
        return cid, data, now

    def wait(self, duration):
        select.select([self.bus], [], [], max(0., duration))

    def startup(self):
        self.emit("event", note=json.dumps(vars(self.args), default=str, sort_keys=True))
        self.send(b"\xcf")  # Deterministic initial disable; no feedback gate before timed commands.
        self.send(b"\xb5")  # Read only; acceleration remains unchanged.

    def run(self):
        start = self.clock.monotonic()
        end = start+(self.plan[-1].start+self.plan[-1].duration)
        next_command, next_flush = start, start+1
        periods = {0xA3: 1/self.args.query_rate, 0xA2: 1/self.args.query_rate,
                   0xA1: 1/self.args.query_rate, 0xAE: 1/self.args.status_rate, 0xB0: 1/self.args.b0_rate}
        due = dict.fromkeys(periods, start)
        while self.clock.monotonic() < end:
            if self.stop_requested():
                raise KeyboardInterrupt
            now = self.clock.monotonic()
            while self.phase_index+1 < len(self.plan) and now >= start+self.plan[self.phase_index+1].start:
                self.phase_index += 1
                phase = self.plan[self.phase_index]
                self.phase, self.phase_started = phase.name, start+phase.start
                self.repeat, self.step, self.plateau = phase.repeat, phase.step, phase.plateau
                self.command_speed = phase.speed
                self.emit("event", note="scheduled phase; no feedback condition or target ramp")
                if self.phase == "hold":
                    print(f"Repeat {self.repeat}, step {self.step}: target {self.plateau:+g} rad/s, "
                          f"timed hold {phase.duration:g} s", flush=True)
            if now >= next_command or self.command_speed != self.last_requested_speed:
                payload, self.encoded_speed, self.command_rpm = speed_payload(self.args, self.command_speed)
                self.last_command = self.send(payload, command_interval_s=now-self.last_command,
                                              command_lateness_s=max(0., now-next_command))
                self.last_requested_speed = self.command_speed
                next_command = self.last_command+1/self.args.command_rate
            opcode = min(due, key=due.get)
            if now >= due[opcode]:
                stamp = self.send(bytes([opcode]))
                due[opcode] = stamp+periods[opcode]  # No bursts of missed requests.
            for _ in range(32):  # Fairness budget, not a feedback/state threshold or abort condition.
                try:
                    self.receive()
                except BlockingIOError:
                    break
            now = self.clock.monotonic()
            if now >= next_flush:
                self.log.file.flush()
                next_flush = now+1
            boundary = start+(self.plan[self.phase_index].start+self.plan[self.phase_index].duration)
            self.wait(min(next_command, min(due.values()), boundary, end)-now)
        self.command_speed = self.encoded_speed = self.command_rpm = 0.
        self.send(speed_payload(self.args, 0.)[0], note="timed sweep completed; zero target, then CF disable")
        self.emit("event", note="sweep completed")

    def shutdown(self):
        self.shutting_down, self.phase = True, "shutdown"
        self.phase_started, self.disable_ack = self.clock.monotonic(), False

        def reply(opcode):
            # Cleanup acknowledgement wait ONLY; never used to gate a sweep phase.
            deadline = self.clock.monotonic()+.15
            while self.clock.monotonic() < deadline:
                try:
                    cid, data, _ = self.receive()
                    if cid == self.args.motor_id and len(data) == LENGTHS[opcode] and data[0] == opcode:
                        return data
                except BlockingIOError:
                    self.wait(deadline-self.clock.monotonic())
            return None

        for _ in range(3):
            try:
                self.send(b"\xcf")  # Always attempt disable, even after logging/CAN faults.
                cf = reply(0xCF)
                if cf is None or cf[6] != 0:
                    continue
                # A queued old CF alone must not authorize a successful shutdown report.
                for _ in range(64):
                    try:
                        self.receive()
                    except BlockingIOError:
                        break
                self.send(b"\xae")
                status = reply(0xAE)
                if status is not None and status[6] == 0:
                    self.disable_ack = True
                    break
            except (OSError, RuntimeError, ValueError):
                continue
        self.emit("event", note=f"CF disable acknowledged={self.disable_ack}; no firmware settings changed")
        print(f"CF disable acknowledged={self.disable_ack}; no firmware settings changed. Steering may coast.",
              flush=True)
        if self.log_error is not None:
            print(f"ERROR: CSV incomplete: {self.log_error}", file=sys.stderr)
        return self.disable_ack and self.log_error is None


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    duration = plan[-1].start+plan[-1].duration
    print(f"Steering ID {args.motor_id}: host {0x100 | args.motor_id:03X}, feedback {args.motor_id:03X}; "
          f"SIGNED speeds {args.speeds} rad/s; {args.repeats} repeat(s), planned duration {duration:g} s")
    print("WARNING: no angle/current/speed/stop-state guards. Multi-turn rotation can hit stops or twist wiring.")
    rotation = max(abs(speed) for speed in args.speeds)*(args.settle+args.hold)
    print(f"Largest per-step commanded travel if tracked: {rotation:g} rad ({rotation/(2*math.pi):.2f} turns).")
    if not args.execute:
        print("PREVIEW ONLY: no CAN, CSV or directories opened. Add --execute to run.")
        for phase in plan:
            print(f"{phase.start:8.3f}s {phase.duration:7.3f}s repeat={phase.repeat} step={phase.step} "
                  f"{phase.name:6} C1 target {phase.speed:+g} rad/s")
        return 0
    stop = [False]

    def interrupt(signum, frame):
        stop[0] = True

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with closing(CsvLog(args.output)) as log, closing(open_bus(args)) as bus:
            runner = Sweep(args, bus, log, lambda: stop[0])
            print(f"CSV: {args.output}\nFront wheel airborne; verify multi-turn clearance and stop other CAN writers.",
                  flush=True)
            try:
                runner.startup()
                runner.run()
            finally:
                if not runner.shutdown() and sys.exc_info()[0] is None:
                    raise RuntimeError("CF disable acknowledgement/CSV completion not confirmed")
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
