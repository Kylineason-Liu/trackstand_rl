#!/usr/bin/env python3
"""Replay recorded policy steering targets with MIT and integrated feedback logging.

Python stdlib + adjacent steering_sine_mit_io.py; no ROS/NumPy required on Pi.
Preview is the default. Only --execute opens CAN. Steering ID 2; no rear writes.
Targets update at 50 Hz; 200 Hz transmission repeats each encoded target four times.
Startup confirms stationary/disabled, sends B1 ONCE (persistent steering origin),
then verifies fresh A3/F1 near zero. Never writes F0, gains, B3 or CD settings.
MIT gains are carried in each packet, with requested velocity/feedforward torque zero.
F0 ranges are encoding ranges, NOT an established total-current/torque protection.
Use independent support, clear steering travel, verified protection and emergency stop.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime
import hashlib
import io as text_io
import json
import math
from pathlib import Path
import signal
import socket
import struct
import sys
import time

import steering_sine_mit_io as mit


HERE = Path(__file__).resolve().parent
DT = .02
KERNEL_TIMESTAMP = 64  # Linux SO_TIMESTAMPNS_NEW, 64-bit timespec also on 32-bit Pi.
TIMESPEC = struct.Struct("=qq")
LENGTHS = mit.READ_LENGTHS | {0xAE: 8, 0xCF: 8, 0xB3: 5, 0xCD: 5}
EXTRA_FIELDS = (
    "input_level", "v1_conditioning", "packet_rate_hz", "replay_time_s", "packet_index", "target_index",
    "target_time_s", "source_time_s", "raw_steering", "scaled_steering_rad", "conditioned_steering_rad",
    "selected_input_rad", "source_scaled_drive_nm", "source_submitted_drive_nm", "kernel_time_unix_s",
    "receive_queue_delay_s", "configured_current_limit_a", "configured_watchdog_hex", "note",
)
PLAN_FIELDS = ("target_index", "target_time_s", "phase", "source_time_s", "raw_steering",
               "scaled_steering_rad", "conditioned_steering_rad", "selected_input_rad", "target_rad",
               "source_scaled_drive_nm", "source_submitted_drive_nm")


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def f32(value):
    return struct.unpack("=f", struct.pack("=f", value))[0]


def ticks(seconds):
    count = round(seconds / DT)
    if seconds < 0 or not math.isclose(count * DT, seconds, abs_tol=1e-9):
        raise ValueError("times must be nonnegative multiples of 0.02 s")
    return count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Enable real steering motion and automatic CSV recording")
    mode.add_argument("--preview", action="store_true", help="Preview only (default): no CAN or output files")
    parser.add_argument("--targets", type=Path, default=HERE / "replay_targets/2500_isaac400_env0_episode1.csv")
    parser.add_argument("--level", choices=("scaled", "conditioned"), default="scaled")
    parser.add_argument("--v1-conditioning", action="store_true",
                        help="Scaled input only: apply +/-1.57-rad clip and 23.9-rad/s target slew at 50 Hz")
    parser.add_argument("--start", type=finite, default=12.5, help="Start timestamp in source episode, seconds")
    parser.add_argument("--duration", type=finite, default=5., help="Core replay duration in seconds")
    parser.add_argument("--ramp", type=finite, default=1., help="Preparation ramp seconds, sampled at 50 Hz")
    parser.add_argument("--pre-hold", type=finite, default=1.)
    parser.add_argument("--post-hold", type=finite, default=1.)
    parser.add_argument("--packet-rate", type=int, choices=(50, 200), default=50)
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--motor-id", type=int, default=2)
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--torque-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--kp", type=finite, default=10., help="Native MIT Kp, N m/rad; this bike has gear ratio 1")
    parser.add_argument("--kd", type=finite, default=.2, help="Native MIT Kd, N m s/rad")
    parser.add_argument("--query-rate", type=finite, default=100., help="A3/A2/A1 queries per second EACH")
    parser.add_argument("--query-timeout", type=finite, default=.02)
    parser.add_argument("--feedback-timeout", type=finite, default=.1)
    parser.add_argument("--max-lateness-ms", type=finite, default=3.,
                        help="Abort late sends; never skip target samples")
    parser.add_argument("--current-abort-a", type=finite, default=3.,
                        help="Measured-current abort, NOT a current clamp")
    parser.add_argument("--fixture", choices=("airborne", "supported-loaded"), default="airborne",
                        help="Record the physical test setup; does not alter commands")
    parser.add_argument("--note", default="", help="Record supply, firmware settings or other test conditions")
    parser.add_argument("--output", type=Path, help="Exclusive CSV path; default: logs/<timestamp>_policy_replay.csv")
    args = parser.parse_args(argv)
    if args.v1_conditioning and args.level != "scaled":
        parser.error("--v1-conditioning requires --level scaled; never condition an already-conditioned target twice")
    try:
        for value in (args.start, args.duration, args.ramp, args.pre_hold, args.post_hold):
            ticks(value)
    except ValueError as error:
        parser.error(str(error))
    if (args.duration <= 0 or not 1 <= args.motor_id <= 254 or not 0 <= args.kp <= 500
            or not 0 <= args.kd <= 5 or not 0 < args.query_rate <= 200
            or not 0 < args.query_timeout < args.feedback_timeout
            or 1 / args.query_rate >= args.feedback_timeout
            or not 0 < args.max_lateness_ms / 1000 < 1 / args.packet_rate
            or args.current_abort_a <= 0):
        parser.error("invalid gains, motor ID, rates, deadlines or abort thresholds")
    if args.output is None:
        name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{args.level}_{args.packet_rate}hz_policy_replay.csv"
        args.output = HERE / "logs" / name
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must have a .csv suffix")
    return args


def load_target_rows(path):
    """Read either motor's actual CSV levels with common timing/provenance validation."""
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    manifest = path.parent / "manifest.json"
    if manifest.exists():
        expected = json.loads(manifest.read_text()).get("files", {}).get(path.name)
        if expected is not None and expected["sha256"] != digest:
            raise ValueError("target CSV hash disagrees with its export manifest")
    reader = csv.DictReader(text_io.StringIO(content.decode("utf-8")))
    required = {"sample_index", "source_time_s", "episode", "raw_steering", "raw_drive",
                "scaled_steering_rad", "conditioned_steering_rad", "scaled_drive_nm", "submitted_drive_nm"}
    if not required <= set(reader.fieldnames or ()):
        raise ValueError(f"target CSV needs columns: {sorted(required)}")
    try:
        rows = [{key: float(row[key]) for key in required} for row in reader]
    except (TypeError, ValueError, csv.Error) as error:
        raise ValueError(f"invalid target CSV cell: {error}") from error
    if not rows or any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("target CSV is empty or contains non-finite samples")
    for index, row in enumerate(rows):
        if (row["sample_index"] != index or row["episode"] != rows[0]["episode"]
                or not math.isclose(row["source_time_s"], rows[0]["source_time_s"] + index * DT, abs_tol=1e-9)):
            raise ValueError("target CSV must be one uninterrupted episode on the 50 Hz grid")
        if (max(abs(row["raw_steering"]), abs(row["raw_drive"])) > 1
                or abs(row["scaled_steering_rad"] - f32(row["raw_steering"] * f32(1.57))) > 1e-6
                or abs(row["scaled_drive_nm"] - f32(row["raw_drive"] * f32(6.28))) > 1e-6):
            raise ValueError("target CSV violates the v1 bounded/scaled action contract")
    return rows, digest


def select_window(rows, start, duration):
    first = ticks(start - rows[0]["source_time_s"])
    stop = first + ticks(duration)
    if stop > len(rows):
        raise ValueError("requested interval exceeds the recorded episode; no padding or resampling")
    return rows[first:stop]


def load_plan(args):
    rows, digest = load_target_rows(args.targets)
    core = select_window(rows, args.start, args.duration)
    key = "scaled_steering_rad" if args.level == "scaled" else "conditioned_steering_rad"
    first_target, last_target = f32(core[0][key]), f32(core[-1][key])
    plan = []

    def add(phase, target, source=None):
        item = dict(target_index=len(plan), target_time_s=len(plan) * DT, phase=phase,
                    source_time_s=None, raw_steering=None, scaled_steering_rad=None,
                    conditioned_steering_rad=None, source_scaled_drive_nm=None, source_submitted_drive_nm=None,
                    selected_input_rad=f32(target), target_rad=f32(target))
        if source is not None:
            item.update({name: source[name] for name in ("source_time_s", "raw_steering",
                                                        "scaled_steering_rad", "conditioned_steering_rad")})
            item.update(source_scaled_drive_nm=source["scaled_drive_nm"],
                        source_submitted_drive_nm=source["submitted_drive_nm"])
        plan.append(item)

    n = ticks(args.ramp)
    for index in range(n):
        # Match the existing replay's NumPy float32 linspace, including its exact final endpoint.
        add("ramp", first_target if index == n - 1 else f32(f32(first_target / n) * (index + 1)))
    for _ in range(ticks(args.pre_hold)):
        add("pre_hold", first_target)
    for row in core:
        add("core", row[key], row)
    for _ in range(ticks(args.post_hold)):
        add("post_hold", last_target)
    if args.v1_conditioning:
        previous, step = 0., f32(23.9 * DT)
        for row in plan:
            desired = max(-f32(1.57), min(f32(1.57), row["selected_input_rad"]))
            previous = f32(previous + max(-step, min(step, f32(desired - previous))))
            row["target_rad"] = previous
    if max(abs(row["target_rad"]) for row in plan) > f32(1.57):
        raise ValueError("requested target exceeds +/-1.57 rad; input is rejected, never clipped")
    return plan, digest


class CsvLog(mit.CsvLog):
    fields = mit.FIELDS + EXTRA_FIELDS

    def __enter__(self):
        # Reuse the existing bounded background writer without changing its module globals.
        self.file = open(self.path, "x", newline="")
        self.stopping, self.failed = False, None
        try:
            self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
            self.writer.writeheader()
            self.file.flush()
            self.thread = mit.threading.Thread(target=self._write, daemon=True)
            self.thread.start()
        except BaseException:
            self.file.close()
            raise
        return self

    def check(self):
        if not self.stopping:
            super().check()

    def emit(self, row):
        try:
            super().emit(row)
        except Exception as error:
            self.failed = error
            if not self.stopping:
                raise

    def __exit__(self, *args):
        super().__exit__(*args)
        if self.failed or self.error:
            raise RuntimeError(f"CSV recording incomplete: {self.failed or self.error}")


class ReplayIO(mit.CanIO):
    reply_lengths = LENGTHS | {0xB1: 3}

    def __init__(self, bus, log, motor, args):
        super().__init__(bus, log, [motor])
        self.args, self.motion_start, self.next_command = args, None, math.inf
        self.offset = None
        self.sample_times = {}
        self.current = self.position = self.velocity = math.nan
        self.current_limit, self.watchdog = None, None

    def row(self, motor, kind, stamp=None, **values):
        now = time.monotonic() if stamp is None else stamp
        base = dict(input_level=self.args.level, v1_conditioning=self.args.v1_conditioning,
                    packet_rate_hz=self.args.packet_rate,
                    replay_time_s=now - self.motion_start if self.motion_start is not None else math.nan)
        base.update(values)
        return super().row(motor, kind, now, **base)

    def send(self, motor, payload, can_id=None, **values):
        deadline = values.get("command_deadline_s")
        if deadline is not None and time.monotonic() - deadline > self.args.max_lateness_ms / 1000:
            raise RuntimeError("missed packet deadline immediately before send")
        return super().send(motor, payload, can_id, **values)

    def drain(self):
        # receive() correlates queries using kernel arrival >= query-send time.
        # The inherited helper also matches queries, without that lower bound.
        for _ in range(mit.MAX_RX_BATCH):
            try:
                self.receive()
            except BlockingIOError:
                return True
            if self.phase == "motion" and time.monotonic() >= self.next_command:
                return False
        return False

    def receive(self):
        frame, ancillary, flags, _ = self.bus.recvmsg(16, socket.CMSG_SPACE(16) + socket.CMSG_SPACE(4))
        if len(frame) != 16 or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise RuntimeError("truncated CAN frame or timestamp")
        wall = None
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == mit.SO_RXQ_OVFL:
                if len(data) != 4 or struct.unpack("=I", data)[0]:
                    raise RuntimeError("CAN RX overflow; recording incomplete")
            if level == socket.SOL_SOCKET and kind == KERNEL_TIMESTAMP:
                if len(data) != 16 or wall is not None:
                    raise RuntimeError("invalid kernel timestamp")
                sec, ns = TIMESPEC.unpack(data)
                if sec < 0 or not 0 <= ns < 1_000_000_000:
                    raise RuntimeError("invalid kernel timestamp value")
                wall = sec + ns * 1e-9
        before, host_wall = time.monotonic(), time.time()
        after = time.monotonic()
        offset = host_wall - before
        if wall is None or after - before > .001 or self.offset is not None and abs(offset - self.offset) > .001:
            raise RuntimeError("missing timestamp or unstable system clock")
        self.offset = offset
        stamp = wall - offset
        if stamp > after + 1e-6:
            raise RuntimeError("future CAN timestamp")
        cid, length, padded = mit.CAN_FRAME.unpack(frame)
        motor = self.motors[0]
        if cid != motor.device_id or flags & socket.MSG_DONTROUTE:
            raise RuntimeError(f"CAN error/nonstandard frame or another selected-motor writer: "
                               f"ID={cid:08X}, data={padded[:length].hex()}")
        data = padded[:length]
        if not data or length > 8 or self.reply_lengths.get(data[0]) != length:
            raise RuntimeError("malformed/unexpected motor reply")
        row = self.row(motor, "can_rx", stamp, kernel_time_unix_s=wall,
                       receive_queue_delay_s=max(0., after - stamp), time_unix_s=wall,
                       mit_can_id=cid, can_direction="bus_rx", can_length=length, can_opcode=data[0],
                       mit_payload_hex=data.hex().upper())
        fatal = self.decode_reply(motor, data, stamp, row)
        self.log.emit(row)
        if fatal:
            raise RuntimeError(fatal)
        if self.pending is not None and data[0] == self.pending.opcode and stamp >= self.pending.sent_at:
            self.finish_query("received" if stamp <= self.pending.deadline else "late_reply", stamp)
        return motor, data, stamp

    def decode_reply(self, motor, data, stamp, row):
        if data[0] == 0xB0:
            constant = struct.unpack("<f", data[2:6])[0]
            if not math.isfinite(constant) or not .01 <= constant <= 10:
                return "invalid B0 torque constant"
            if self.phase == "motion" and constant != motor.torque_constant:
                return "B0 torque constant changed during replay"
        fatal = super().decode_reply(motor, data, stamp, row)
        op = data[0]
        if op in (0xA3, 0xA2, 0xA1, 0xAE):
            self.sample_times[op] = stamp
        if op == 0xA3:
            self.position = row["position_unwrapped_rad"]
        elif op == 0xA2:
            self.velocity = row["velocity_rad_s"]
        elif op == 0xA1:
            self.current = row["current_a"]
        elif op == 0xB3:
            self.current_limit = struct.unpack("<I", data[1:])[0] * .001
            row["configured_current_limit_a"] = self.current_limit
        elif op == 0xCD:
            self.watchdog = data.hex().upper()
            row["configured_watchdog_hex"] = self.watchdog
        if self.phase == "motion":
            if op == 0xA1 and abs(self.current) > self.args.current_abort_a:
                return "measured current exceeds experiment abort threshold"
        return fatal

    def request(self, motor, opcode, length=None, payload=None, expected=None, required=True):
        if not self.drain():
            raise RuntimeError("receive backlog before query")
        sent = self.send(motor, bytes([opcode]) if payload is None else payload)
        deadline = sent + .5
        while time.monotonic() < deadline:
            if self.wait(min(.02, max(0., deadline - time.monotonic()))):
                result = self.receive()
                if result is not None:
                    got, data, stamp = result
                    if got is motor and data[0] == opcode and sent <= stamp <= deadline:
                        return data
        raise TimeoutError(f"no fresh {opcode:02X} startup/shutdown reply")

    def guard(self):
        now = time.monotonic()
        self.log.check()
        for opcode in (0xA3, 0xA2, 0xA1):
            if now - self.sample_times.get(opcode, -math.inf) > self.args.feedback_timeout:
                raise RuntimeError(f"stale {opcode:02X} feedback")
        if now - self.last_f1_at > self.args.feedback_timeout:
            raise RuntimeError("stale F1 feedback")
        if abs(self.current) > self.args.current_abort_a:
            raise RuntimeError("measured current exceeds experiment abort threshold")


def open_bus(args):
    bus = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    try:
        ids = (args.motor_id, 0x100 | args.motor_id, 0x400 | args.motor_id, 0x500 | args.motor_id,
               0, 0xFF, 0x400, 0x4FF)
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                       b"".join(struct.pack("=II", cid, mit.CAN_MASK) for cid in ids))
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
        bus.setsockopt(socket.SOL_CAN_RAW, getattr(socket, "CAN_RAW_ERR_FILTER", 2), struct.pack("=I", 0x1FFFFFFF))
        bus.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        bus.setsockopt(socket.SOL_SOCKET, mit.SO_RXQ_OVFL, 1)
        bus.setsockopt(socket.SOL_SOCKET, KERNEL_TIMESTAMP, 1)
        bus.setblocking(False)
        bus.bind((args.interface,))
        return bus
    except BaseException:
        bus.close()
        raise


def startup(io, plan):
    motor = io.motors[0]
    io.origin_setup = dict(method="B1 persistent steering origin, once per execution",
                           attempted=False, acknowledged=False, verified=False)
    disabled = io.request(motor, 0xCF)
    if disabled[6] or disabled[7]:
        raise RuntimeError("motor not confirmed disabled and fault-free")
    motor.requested_config = io.request(motor, 0xF0)
    io.request(motor, 0xB0)
    if not math.isfinite(motor.torque_constant):
        raise RuntimeError("valid measured B0 torque constant required")
    io.request(motor, 0xB3)
    io.request(motor, 0xCD)
    if io.current_limit <= 0:
        raise RuntimeError("invalid configured B3 current limit")
    for row in plan:
        mit.encode(motor.encoder_sign * row["target_rad"], -motor.ranges[0], motor.ranges[0], 16)
    before = io.request(motor, 0xA3)
    io.origin_setup["encoder_count_before_reset"] = struct.unpack("<i", before[3:7])[0]
    io.request(motor, 0xA2)
    io.request(motor, 0xA1)
    status = io.request(motor, 0xAE)
    if status[6] or status[7] or abs(io.velocity) > .2 or abs(io.current) > io.args.current_abort_a:
        raise RuntimeError("steering must be disabled, stationary and below current threshold BEFORE B1")

    # B1 changes the saved origin; F1 may otherwise be clipped by the existing F0 range.
    # Do not repeat B1 on a lost reply or attempt an undocumented origin restoration.
    io.origin_setup["attempted"] = True
    ack = io.request(motor, 0xB1)
    io.origin_setup.update(acknowledged=True, acknowledgement_hex=ack.hex().upper())
    encoder = io.request(motor, 0xA3)
    single, count = struct.unpack("<Hi", encoder[1:])
    if single >= 16384 or abs((single + 8192) % 16384 - 8192) > 2 or abs(count) > 2:
        raise RuntimeError("A3 not near the new B1 origin; no MIT motion sent")
    feedback = io.request(motor, 0xF1)
    pcode = int.from_bytes(feedback[1:3], "big")
    p = mit.decode_symmetric(pcode, motor.ranges[0], 16)
    v = mit.decode_symmetric((feedback[3] << 4) | (feedback[4] >> 4), motor.ranges[1], 12)
    io.origin_setup.update(F1_position_code=pcode, F1_native_zero_rad=p, A3_zero_count=count)
    if pcode in (0, 65535):
        raise RuntimeError("F1 startup position is clipped at its encoding endpoint; no MIT motion sent")
    tolerance = 2 * mit.RADIANS_PER_COUNT + 2 * motor.ranges[0] / 65535
    if feedback[6] & 3 or abs(p - count * mit.RADIANS_PER_COUNT) > tolerance:
        raise RuntimeError("F1 must be inactive, fault-free and consistent with the new A3 origin")
    if abs(v) > .1 + 2 * motor.ranges[1] / 4095:
        raise RuntimeError("F1 reports moving steering after B1; no MIT motion sent")
    motor.zero_count, motor.zero_rad = count, p
    motor.reference_epoch = 1
    # Bracket F1 with fresh A3 samples so changing position cannot define mismatched zeros.
    after = io.request(motor, 0xA3)
    if abs(struct.unpack("<i", after[3:7])[0] - count) > 2:
        raise RuntimeError("steering moved while establishing A3/F1 references; no MIT motion sent")
    io.request(motor, 0xA2)
    io.request(motor, 0xA1)
    status = io.request(motor, 0xAE)
    if status[6] or status[7] or abs(io.velocity) > .2 or abs(io.current) > io.args.current_abort_a:
        raise RuntimeError("steering status/motion/current changed during initialization")
    if io.request(motor, 0xF0) != motor.requested_config:
        raise RuntimeError("F0 encoding ranges changed during initialization")
    for row in plan:
        mit.encode(motor.zero_rad + motor.encoder_sign * row["target_rad"], -motor.ranges[0], motor.ranges[0], 16)
    io.origin_setup["verified"] = True
    io.log.emit(io.row(motor, "event", can_note="B1 acknowledged; fresh stationary A3/F1 origins verified"))
    print(f"Steering origin verified after B1: A3={count} counts, F1={p:.9f} rad; "
          f"original A3={io.origin_setup['encoder_count_before_reset']} counts", flush=True)


def payload(target_rad, motor, args):
    p = mit.encode(motor.zero_rad + motor.encoder_sign * target_rad, -motor.ranges[0], motor.ranges[0], 16)
    v = mit.encode(0., -motor.ranges[1], motor.ranges[1], 12)
    t = mit.encode(0., -motor.ranges[2], motor.ranges[2], 12)
    kp, kd = mit.encode(args.kp, 0., 500., 12), mit.encode(args.kd, 0., 5., 12)
    return bytes([p >> 8, p & 255, v >> 4, ((v & 15) << 4) | (kp >> 8), kp & 255,
                  kd >> 4, ((kd & 15) << 4) | (t >> 8), t & 255])


def motion(io, plan):
    motor, args = io.motors[0], io.args
    packets = [payload(row["target_rad"], motor, args) for row in plan]
    io.phase, io.motion_start = "motion", time.monotonic()
    io.next_command = io.last_f1_at = io.motion_start
    repeats, packet_index, previous = args.packet_rate // 50, 0, None
    periods = {0xA3: 1 / args.query_rate, 0xA2: 1 / args.query_rate, 0xA1: 1 / args.query_rate,
               0xAE: .2, 0xB0: 1., 0xF0: 1.}
    due, jobs = dict.fromkeys(periods, io.motion_start), deque()
    end = io.motion_start + len(plan) * DT
    while time.monotonic() < end:
        empty = io.drain()
        io.guard()
        now = time.monotonic()
        if packet_index < len(plan) * repeats and now >= io.next_command:
            if now - io.next_command > args.max_lateness_ms / 1000:
                raise RuntimeError("missed packet deadline; replay aborted without skipping or bursting targets")
            index = packet_index // repeats
            sample = plan[index]
            previous = io.send(motor, packets[index], 0x500 | motor.device_id, command_previous_s=previous,
                               command_deadline_s=io.next_command, packet_index=packet_index, target_index=index,
                               target_time_s=sample["target_time_s"], phase=sample["phase"],
                               requested_position_rad=sample["target_rad"],
                               **{key: sample[key] for key in PLAN_FIELDS if key not in (
                                   "target_index", "target_time_s", "phase", "target_rad")},
                               **mit.target_fields(motor, packets[index]))
            packet_index += 1
            io.next_command = io.motion_start + packet_index / args.packet_rate
        if io.pending is not None and now >= io.pending.deadline:
            io.finish_query("timed_out")
        if not jobs and io.pending is None:
            for opcode, period in periods.items():
                if now >= due[opcode]:
                    jobs.append(opcode)
                    due[opcode] = now + period
        if empty and io.pending is None and jobs:
            opcode = jobs.popleft()
            sent = io.send(motor, bytes([opcode]))
            io.pending = mit.PendingRead(motor, opcode, sent, sent + args.query_timeout)
        wake = min(end, io.next_command, io.pending.deadline if io.pending else min(due.values()))
        io.wait(0 if not empty or jobs and io.pending is None else max(0., wake - time.monotonic()))
    if packet_index != len(plan) * repeats:
        raise RuntimeError("replay ended with missing packets")


def shutdown(io):
    io.phase, io.log.stopping = "shutdown", True
    io.pending = None
    for _ in range(3):
        try:
            # Disable is sent before draining so malformed traffic/logger failure cannot prevent it.
            io.bus.send(mit.CAN_FRAME.pack(0x100 | io.motors[0].device_id, 1, b"\xcf"))
            cf = io.request(io.motors[0], 0xCF)
            ae = io.request(io.motors[0], 0xAE)
            if cf[6] == 0 and ae[6] == 0:
                io.final_fault_code = ae[7] or cf[7]
                return True
        except (OSError, RuntimeError, ValueError):
            continue
    return False


def execute(args, plan, input_sha):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path, plan_path = args.output.with_suffix(".json"), args.output.with_name(args.output.stem + "_plan.csv")
    if any(path.exists() for path in (args.output, meta_path, plan_path)):
        raise FileExistsError("output CSV/JSON/plan already exists; recordings are never overwritten")
    metadata = dict(completed=False, disable_confirmed=False, targets_sha256=input_sha,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    transport_sha256=hashlib.sha256(Path(mit.__file__).read_bytes()).hexdigest(),
                    arguments={key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()},
                    target_hz=50, packet_hz=args.packet_rate, rear_motor_commanded=False,
                    torque_meaning="Signed A1 Q-current times measured B0 Kt, not independent shaft torque",
                    timestamps="Host send-call and kernel host RX times; not motor acquisition or wire TX times")
    with plan_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(plan)
    with meta_path.open("x+") as meta:
        json.dump(metadata, meta, indent=2)
        meta.flush()
        try:
            with CsvLog(args.output) as log, open_bus(args) as bus:
                motor = mit.Motor("steering", args.motor_id, 1., args.sign, args.torque_sign)
                io = ReplayIO(bus, log, motor, args)
                try:
                    startup(io, plan)
                    metadata.update(mit_ranges=motor.ranges, native_zero_rad=motor.zero_rad,
                                    encoder_zero_count=motor.zero_count, torque_constant_nm_per_a=motor.torque_constant,
                                    configured_B3_current_limit_a=io.current_limit, configured_CD_hex=io.watchdog)
                    motion(io, plan)
                    metadata["completed"] = True
                finally:
                    metadata["steering_origin_setup"] = getattr(io, "origin_setup", None)
                    metadata["disable_confirmed"] = shutdown(io)
                    metadata["final_fault_code"] = getattr(io, "final_fault_code", None)
                    if not metadata["disable_confirmed"]:
                        raise RuntimeError("CF/AE disable NOT confirmed; inspect motor physically")
                    if metadata["final_fault_code"]:
                        raise RuntimeError("motor disabled but reports a fault at shutdown")
                    if log.failed or log.error:
                        raise RuntimeError("recording failed; motor disable was attempted")
        except BaseException as error:
            metadata.update(completed=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            meta.seek(0)
            json.dump(metadata, meta, indent=2, allow_nan=False)
            meta.truncate()
            meta.flush()
    print(f"Replay finished; disable confirmed. CSV: {args.output}")


def main(argv=None):
    args = parse_args(argv)
    def stop(*_):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        plan, digest = load_plan(args)
        core = [row for row in plan if row["phase"] == "core"]
        print(f"{'EXECUTE' if args.execute else 'PREVIEW'}: {args.targets.name}; {args.level}; "
              f"targets=50 Hz, packets={args.packet_rate} Hz; {len(plan) * DT:g} s total")
        print(f"Source core [{core[0]['source_time_s']:g}, {core[-1]['source_time_s'] + DT:g}) s; "
              f"position [{min(r['target_rad'] for r in plan):.6g}, {max(r['target_rad'] for r in plan):.6g}] rad")
        if args.execute:
            print("Independent support, straight startup, free steering and verified protection required. "
                  "B1 will set the persistent steering origin once; rear motor is not commanded.", flush=True)
            execute(args, plan, digest)
        else:
            print("No CAN opened; no CSV created. Add --execute for hardware motion.")
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, OverflowError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
