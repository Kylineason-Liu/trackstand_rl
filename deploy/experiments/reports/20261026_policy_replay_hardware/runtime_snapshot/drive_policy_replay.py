#!/usr/bin/env python3
"""Rear-only C0 replay of recorded 50-Hz policy torques, with integrated feedback.

Preview is default; --execute is required for CAN. Default device ID 1, signs -1.
Reuses adjacent steering_policy_replay.py and steering_sine_mit_io.py transport
helpers; neither steering motion nor policy inference runs. No ROS/NumPy on Pi.
Torque -> signed current using measured B0 -> C0 int32 little-endian milliamps.
No added damping, friction compensation, interpolation or silent torque clipping.
Motor B3 limits position/speed modes, NOT C0: read it but do not assume protection.
Read B4 current slew without changing it. Temporarily enable/shorten CD watchdog;
restore it only after fresh CF/AE confirm disabled. Support bike, rear airborne,
other motor independently disabled, all other CAN writers stopped, emergency stop ready.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import sys
import time

import steering_policy_replay as shared


HERE = Path(__file__).resolve().parent
mit, DT = shared.mit, shared.DT
PLAN_FIELDS = ("target_index", "target_time_s", "phase", "source_time_s", "raw_drive",
               "scaled_drive_nm", "submitted_drive_nm", "requested_torque_nm")
DRIVE_FIELDS = ("raw_drive", "scaled_drive_nm", "submitted_drive_nm", "requested_torque_nm",
                "command_current_a", "encoded_current_a", "encoded_command_torque_nm", "current_code_ma",
                "current_slew_raw", "current_slew_a_s", "position_speed_current_limit_a",
                "can_id_hex", "payload_hex")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run the rear motor and automatically record CSV")
    mode.add_argument("--preview", action="store_true", help="Preview only (default); no CAN or output files")
    parser.add_argument("--targets", type=Path, default=HERE / "replay_targets/2500_isaac400_env0_episode1.csv")
    parser.add_argument("--level", choices=("scaled", "submitted"), default="scaled")
    parser.add_argument("--start", type=shared.finite, default=12.5, help="Source episode timestamp, seconds")
    parser.add_argument("--duration", type=shared.finite, default=5.)
    parser.add_argument("--pre-zero", type=shared.finite, default=2., help="Initial zero-torque seconds")
    parser.add_argument("--post-hold", type=shared.finite, default=0.,
                        help="Extra seconds at the LAST torque (default 0; this can keep accelerating the wheel)")
    parser.add_argument("--packet-rate", type=int, choices=(50, 200), default=200)
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--motor-id", type=int, default=1)
    parser.add_argument("--encoder-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--torque-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--query-rate", type=shared.finite, default=100., help="A3/A2/A1 queries/s EACH")
    parser.add_argument("--query-timeout", type=shared.finite, default=.02)
    parser.add_argument("--feedback-timeout", type=shared.finite, default=.1)
    parser.add_argument("--max-lateness-ms", type=shared.finite, default=3.)
    parser.add_argument("--command-current-limit-a", type=shared.finite, default=1.5,
                        help="Reject the ENTIRE plan if any encoded command exceeds this; never clip")
    parser.add_argument("--current-abort-a", type=shared.finite, default=2., help="Measured current abort threshold")
    parser.add_argument("--speed-abort-rad-s", type=shared.finite, default=10., help="Measured speed abort threshold")
    parser.add_argument("--temperature-abort-c", type=shared.finite, default=60.)
    parser.add_argument("--watchdog-ms", type=int, default=500,
                        help="Temporary CD timeout; never lengthens an enabled one")
    parser.add_argument("--note", default="", help="Record supply, physical fixture and other actual conditions")
    parser.add_argument("--output", type=Path,
                        help="Exclusive CSV path; default: logs/<timestamp>_drive_policy_replay.csv")
    args = parser.parse_args(argv)
    try:
        for value in (args.start, args.duration, args.pre_zero, args.post_hold):
            shared.ticks(value)
    except ValueError as error:
        parser.error(str(error))
    if (args.duration <= 0 or not 1 <= args.motor_id <= 254 or not 0 < args.query_rate <= 200
            or not 0 < args.query_timeout < args.feedback_timeout
            or 1 / args.query_rate >= args.feedback_timeout
            or not 0 < args.max_lateness_ms / 1000 < 1 / args.packet_rate
            or not 0 < args.command_current_limit_a < args.current_abort_a
            or min(args.speed_abort_rad_s, args.temperature_abort_c) <= 0
            or not 100 <= args.watchdog_ms <= 1000):
        parser.error("invalid timing, motor ID, watchdog or abort thresholds")
    if args.output is None:
        args.output = HERE / "logs" / (f"{datetime.now():%Y%m%d_%H%M%S_%f}_"
                                      f"{args.level}_{args.packet_rate}hz_drive_policy_replay.csv")
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must have a .csv suffix")
    return args


def load_plan(args):
    rows, digest = shared.load_target_rows(args.targets)
    core = shared.select_window(rows, args.start, args.duration)
    key = "scaled_drive_nm" if args.level == "scaled" else "submitted_drive_nm"
    plan = []

    def add(phase, target, source=None):
        item = dict(target_index=len(plan), target_time_s=len(plan) * DT, phase=phase,
                    source_time_s=None, raw_drive=None, scaled_drive_nm=None, submitted_drive_nm=None,
                    requested_torque_nm=shared.f32(target))
        if source is not None:
            item.update({name: source[name] for name in ("source_time_s", "raw_drive",
                                                        "scaled_drive_nm", "submitted_drive_nm")})
        plan.append(item)

    for _ in range(shared.ticks(args.pre_zero)):
        add("pre_zero", 0.)
    for row in core:
        add("core", row[key], row)
    for _ in range(shared.ticks(args.post_hold)):
        add("post_hold", core[-1][key])
    if max(abs(row["requested_torque_nm"]) for row in plan) > shared.f32(6.28):
        raise ValueError("requested torque exceeds the v1 +/-6.28 Nm contract; no clipping applied")
    return plan, digest


def current_payload(torque_nm, constant, sign):
    if not math.isfinite(constant) or not .01 <= constant <= 10 or sign not in (-1, 1):
        raise ValueError("valid measured B0 Kt and torque sign required")
    requested = sign * torque_nm / constant
    if not math.isfinite(requested):
        raise ValueError("current must be finite")
    code = round(requested * 1000)
    if not -(2**31) <= code < 2**31:
        raise OverflowError("C0 current does not fit signed int32 milliamps")
    return b"\xc0" + struct.pack("<i", code), dict(
        command_current_a=requested, encoded_current_a=code * .001, current_code_ma=code,
        encoded_command_torque_nm=sign * code * .001 * constant)


class CsvLog(shared.CsvLog):
    fields = shared.CsvLog.fields + DRIVE_FIELDS


class DriveIO(shared.ReplayIO):
    reply_lengths = {op: size for op, size in shared.LENGTHS.items() if op not in (0xF0, 0xF1)} | {0xC0: 5, 0xB4: 5}

    def __init__(self, bus, log, motor, args):
        super().__init__(bus, log, motor, args)
        self.current_slew_raw = self.original_cd = self.restore_cd = None
        self.watchdog_restored = False
        self.expected_cd, self.restore_error = None, None

    def row(self, motor, kind, stamp=None, **values):
        now = time.monotonic() if stamp is None else stamp
        extra = dict(input_level=self.args.level, packet_rate_hz=self.args.packet_rate,
                     replay_time_s=now - self.motion_start if self.motion_start is not None else math.nan)
        extra.update(values)
        if "mit_can_id" in extra:
            extra["can_id_hex"] = f"{extra['mit_can_id']:03X}"
        if "mit_payload_hex" in extra:
            extra["payload_hex"] = extra["mit_payload_hex"]
        return mit.CanIO.row(self, motor, kind, now, **extra)

    def decode_reply(self, motor, data, stamp, row):
        op = data[0]
        if op == 0xC0:
            current = struct.unpack("<i", data[1:])[0] * .001
            row.update(record_type="current_feedback", current_a=current,
                       torque_nm=motor.torque_sign * current * motor.torque_constant)
        else:
            # Generic A1/A2/A3/B0/AE decoder; do not call the MIT-mode or steering-angle guards.
            if op == 0xB0:
                value = struct.unpack("<f", data[2:6])[0]
                if not math.isfinite(value) or not .01 <= value <= 10:
                    return "invalid B0 torque constant"
                if self.phase == "motion" and value != motor.torque_constant:
                    return "B0 torque constant changed during replay"
            fatal = mit.CanIO.decode_reply(self, motor, data, stamp, row)
            if fatal:
                return fatal.replace("steering", "rear")
        if op in (0xA1, 0xA2, 0xA3, 0xAE, 0xC0):
            self.sample_times[op] = stamp
        if op in (0xA1, 0xC0):
            self.current = row["current_a"]
        elif op == 0xA2:
            self.velocity = row["velocity_rad_s"]
        elif op == 0xA3:
            self.position = row["position_unwrapped_rad"]
        elif op == 0xB3:
            self.current_limit = struct.unpack("<I", data[1:])[0] * .001
            row["position_speed_current_limit_a"] = self.current_limit
        elif op == 0xB4:
            value = struct.unpack("<I", data[1:])[0]
            if self.phase == "motion" and value != self.current_slew_raw:
                return "B4 current slew changed during replay"
            self.current_slew_raw = value
            row.update(current_slew_raw=value, current_slew_a_s=value * .001)
        elif op == 0xCD:
            self.watchdog = data.hex().upper()
            row["configured_watchdog_hex"] = self.watchdog
            if self.phase == "motion" and data != self.expected_cd:
                return "CD watchdog changed during rear replay"
        if self.phase == "motion":
            if op in (0xA1, 0xC0) and abs(self.current) > self.args.current_abort_a:
                return f"measured current {self.current:g} A exceeds rear abort {self.args.current_abort_a:g} A"
            if op == 0xA2 and abs(self.velocity) > self.args.speed_abort_rad_s:
                return f"rear speed {self.velocity:g} rad/s exceeds abort {self.args.speed_abort_rad_s:g} rad/s"
            if op in (0xAE, 0xCF):
                if data[6] != 2:
                    return "rear motor left C0 current mode"
                if data[7] or data[5] >= self.args.temperature_abort_c:
                    return "rear motor fault or excessive temperature"
        return ""

    def guard(self):
        self.log.check()
        now = time.monotonic()
        for op in (0xA1, 0xA2, 0xA3, 0xC0):
            if now - self.sample_times.get(op, self.motion_start if op == 0xC0 else -math.inf) \
                    > self.args.feedback_timeout:
                raise RuntimeError(f"stale rear {op:02X} feedback")
        if now - self.sample_times.get(0xAE, -math.inf) > max(.6, self.args.feedback_timeout):
            raise RuntimeError("stale rear motor status")
        if abs(self.current) > self.args.current_abort_a or abs(self.velocity) > self.args.speed_abort_rad_s:
            raise RuntimeError("rear current/speed exceeds experiment abort threshold")

    def verify_cd(self, desired):
        motor = self.motors[0]
        if self.request(motor, 0xCD, payload=desired) != desired or self.request(motor, 0xCD) != desired:
            raise RuntimeError("CD watchdog acknowledgement/readback mismatch")


def startup(io, plan):
    motor, args = io.motors[0], io.args
    disabled = io.request(motor, 0xCF)
    if disabled[6] or disabled[7]:
        raise RuntimeError("rear motor not confirmed disabled and fault-free")
    io.request(motor, 0xB0)
    if not math.isfinite(motor.torque_constant):
        raise RuntimeError("valid measured B0 torque constant required")
    io.request(motor, 0xB3)  # Informational: this is NOT a C0 current limit.
    io.request(motor, 0xB4)
    original = io.original_cd = io.request(motor, 0xCD)
    if original[1] not in (0, 1):
        raise RuntimeError("invalid CD watchdog enable byte")
    packets = [current_payload(row["requested_torque_nm"], motor.torque_constant, motor.torque_sign) for row in plan]
    peak_current = max(abs(fields["encoded_current_a"]) for _, fields in packets)
    if peak_current > args.command_current_limit_a:
        raise ValueError(f"plan needs {peak_current:g} A, above --command-current-limit-a "
                         f"{args.command_current_limit_a:g}; no C0 sent, no targets clipped")
    timeout = min(struct.unpack("<H", original[2:4])[0], args.watchdog_ms) if original[1] else args.watchdog_ms
    if timeout < 3 * 1000 / args.packet_rate:
        raise RuntimeError("existing enabled watchdog too short; no watchdog settings changed")
    requested = b"\xcd\x01" + struct.pack("<H", timeout) + original[4:]
    if original != requested:
        io.restore_cd = original  # Register rollback BEFORE write/possibly lost acknowledgement.
        io.verify_cd(requested)
    io.expected_cd = requested
    io.watchdog_restored = io.restore_cd is None
    encoder = io.request(motor, 0xA3)
    motor.zero_count, motor.reference_epoch = struct.unpack("<i", encoder[3:])[0], 1
    io.request(motor, 0xA3)
    io.request(motor, 0xA2)
    io.request(motor, 0xA1)
    status = io.request(motor, 0xAE)
    if (status[6] or status[7] or abs(io.velocity) > .2 or abs(io.current) > args.current_abort_a
            or status[5] >= args.temperature_abort_c):
        raise RuntimeError("rear motor must be stationary, disabled, fault-free and below test thresholds")
    return packets


def motion(io, plan, packets):
    motor, args = io.motors[0], io.args
    io.phase, io.motion_start = "motion", time.monotonic()
    io.next_command = io.motion_start
    repeats, packet_index, previous = args.packet_rate // 50, 0, None
    periods = {0xA3: 1 / args.query_rate, 0xA2: 1 / args.query_rate, 0xA1: 1 / args.query_rate,
               0xAE: .2, 0xB0: 1., 0xB4: 1., 0xCD: 1.}
    due, jobs = dict.fromkeys(periods, io.motion_start), deque()
    end = io.motion_start + len(plan) * DT
    while time.monotonic() < end:
        empty = io.drain()
        io.guard()
        now = time.monotonic()
        if packet_index < len(plan) * repeats and now >= io.next_command:
            index = packet_index // repeats
            sample, (data, fields) = plan[index], packets[index]
            previous = io.send(motor, data, command_previous_s=previous, command_deadline_s=io.next_command,
                               record_type="drive_target", packet_index=packet_index,
                               **sample, **fields)
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
        raise RuntimeError("rear replay ended with missing packets")


def shutdown(io):
    confirmed = shared.shutdown(io)
    if confirmed:
        try:
            if io.restore_cd is not None:
                io.verify_cd(io.restore_cd)
            io.watchdog_restored = True
        except (OSError, ValueError, RuntimeError) as error:
            io.restore_error = str(error)
        finally:
            # A failed restore or logger must not prevent a further disable attempt.
            confirmed = shared.shutdown(io) and confirmed
    return confirmed


def execute(args, plan, digest):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(".json")
    plan_path = args.output.with_name(args.output.stem + "_plan.csv")
    if any(path.exists() for path in (args.output, meta_path, plan_path)):
        raise FileExistsError("output CSV/JSON/plan exists; refuse overwrite")
    metadata = dict(completed=False, disable_confirmed=False, watchdog_restored=False,
                    target_hz=50, packet_hz=args.packet_rate, steering_motor_commanded=False,
                    targets_sha256=digest,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                        Path(__file__), Path(shared.__file__), Path(mit.__file__))},
                    arguments={key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
                    host_damping_nm_s_per_rad=0., friction_compensation_nm=0.,
                    torque_meaning="signed A1/C0 measured current times B0 Kt, not independent shaft torque",
                    timestamps="host send-call and kernel host RX times, not motor acquisition or on-wire TX",
                    B3_meaning="position/speed-mode current limit; not a C0 current ceiling")
    with plan_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(plan)
    with meta_path.open("x+") as stream:
        json.dump(metadata, stream, indent=2)
        stream.flush()
        try:
            with CsvLog(args.output) as log, shared.open_bus(args) as bus:
                motor = mit.Motor("drive", args.motor_id, 1., args.encoder_sign, args.torque_sign)
                io = DriveIO(bus, log, motor, args)
                try:
                    packets = startup(io, plan)
                    metadata.update(torque_constant_nm_per_a=motor.torque_constant, encoder_zero_count=motor.zero_count)
                    motion(io, plan, packets)
                    metadata["completed"] = True
                finally:
                    try:
                        metadata["disable_confirmed"] = shutdown(io)
                    finally:
                        metadata.update(watchdog_restored=io.watchdog_restored,
                                        CD_restore_error=io.restore_error,
                                        original_CD_hex=io.original_cd.hex() if io.original_cd else None,
                                        latest_CD_hex=io.watchdog, B3_position_speed_limit_a=io.current_limit,
                                        B4_current_slew_raw=io.current_slew_raw,
                                        B4_current_slew_a_s=(io.current_slew_raw * .001
                                                            if io.current_slew_raw is not None else None),
                                        torque_constant_nm_per_a=(motor.torque_constant
                                                                 if math.isfinite(motor.torque_constant) else None),
                                        encoder_zero_count=motor.zero_count,
                                        final_fault_code=getattr(io, "final_fault_code", None))
                    if not metadata["disable_confirmed"] or not metadata["watchdog_restored"]:
                        raise RuntimeError("rear disable or CD restore NOT confirmed; "
                                           "inspect motor/settings physically")
                    if metadata["final_fault_code"] or log.failed or log.error:
                        raise RuntimeError("rear motor fault or incomplete recording; disable attempted")
        except BaseException as error:
            metadata.update(completed=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            stream.seek(0)
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.truncate()
            stream.flush()
    print(f"Rear replay complete; disabled and CD restored. CSV: {args.output}")


def main(argv=None):
    args = parse_args(argv)
    def stop(*_):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        plan, digest = load_plan(args)
        low, high = min(r["requested_torque_nm"] for r in plan), max(r["requested_torque_nm"] for r in plan)
        print(f"{'EXECUTE' if args.execute else 'PREVIEW'} rear ID {args.motor_id}: {args.targets.name}; "
              f"{args.level}; targets 50 Hz, C0 {args.packet_rate} Hz; "
              f"{len(plan) * DT:g} s; torque [{low:g}, {high:g}] Nm")
        print(f"Core source [{args.start:g}, {args.start + args.duration:g}) s; "
              f"{args.pre_zero:g} s zero-torque preparation; {args.post_hold:g} s final-torque hold")
        if args.execute:
            print("Rear wheel airborne; independent support; steering independently disabled; "
                  "all other writers stopped.",
                  flush=True)
            execute(args, plan, digest)
        else:
            print("No CAN/output files opened. Current conversion requires measured B0 during --execute.")
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
