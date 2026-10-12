#!/usr/bin/env python3
"""Standalone, read-query-only GIM feedback recorder; no ROS or third-party modules.

Conversions follow motor_io.py/can_protocol.py. Only A1, A2, A3 and B0 reads
are transmitted. Never enables/disables motors, resets origins, changes gains,
or sends current/position commands. Each SocketCAN socket receives its own copy
of replies, so this recorder does not consume the sine script's A3 response.
Another receive-only socket logs outgoing C2 targets through SocketCAN loopback.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import math
import queue
import signal
import socket
import struct
import sys
import threading
import time


RADIANS_PER_COUNT = 2 * math.pi / 16384
READ_LENGTHS = {0xA1: (5,), 0xA2: (5,), 0xA3: (7,), 0xB0: (6, 7, 8)}
FIELDS = (
    "record_type", "time_unix_s", "elapsed_s", "motor", "motor_id", "encoder_count", "zero_count",
    "position_rad", "position_unwrapped_rad", "velocity_rad_s", "current_a",
    "torque_nm", "torque_constant_nm_per_a", "position_elapsed_s",
    "velocity_elapsed_s", "current_elapsed_s", "c2_target_count",
)


@dataclass
class Motor:
    name: str
    device_id: int
    gear_ratio: float
    encoder_sign: int
    torque_sign: int
    zero_count: int = None
    torque_constant: float = math.nan
    retry_b0_at: float = 0.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Record motor feedback and observed outgoing C2 target counts/timestamps to CSV, without ROS.",
        epilog="Run before the sine script to capture a stationary startup zero. NaN means unavailable, not zero. "
               "The query rate is a target, not a timing guarantee. A1/A2/A3 samples are sequential, not simultaneous.",
    )
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--motors", nargs="+", choices=("steering", "drive"), default=["steering", "drive"])
    parser.add_argument("--rate", type=float, default=200.0, help="Target sample cycles/s per motor (default: 200).")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 means until Ctrl-C.")
    parser.add_argument("--timeout", type=float, default=0.002, help="Maximum wait per query, seconds (default: .002).")
    parser.add_argument("--output", default=f"motor_feedback_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
    # Defaults are from deploy/source/trackstand_deploy/config/params.yaml.
    for name, device_id, encoder_sign in (("steering", 2, -1), ("drive", 1, -1)):
        parser.add_argument(f"--{name}-id", type=lambda text: int(text, 0), default=device_id)
        parser.add_argument(f"--{name}-gear-ratio", type=float, default=1.0)
        parser.add_argument(f"--{name}-encoder-sign", type=int, choices=(-1, 1), default=encoder_sign)
        parser.add_argument(f"--{name}-torque-sign", type=int, choices=(-1, 1), default=-1)
    args = parser.parse_args(argv)
    values = (args.rate, args.duration, args.timeout, args.steering_gear_ratio, args.drive_gear_ratio)
    if (not all(math.isfinite(value) for value in values) or args.rate <= 0 or args.duration < 0
            or args.timeout <= 0 or min(args.steering_gear_ratio, args.drive_gear_ratio) <= 0):
        parser.error("rate, timeout and gear ratios must be positive and finite; duration must be nonnegative")
    if not all(1 <= device_id <= 254 for device_id in (args.steering_id, args.drive_id)):
        parser.error("device IDs must be between 1 and 254")
    if len(set(args.motors)) != len(args.motors):
        parser.error("each motor may be selected only once")
    if len(args.motors) == 2 and args.steering_id == args.drive_id:
        parser.error("steering and drive must have different device IDs")
    return args


def query(bus, device_id, command, timeout):
    """Drain old replies, issue a one-byte read, then match device, echo and DLC."""
    lengths = READ_LENGTHS[command]  # Whitelist excludes all control/write commands.
    bus.setblocking(False)
    try:
        for _ in range(64):
            bus.recv(16)
    except BlockingIOError:
        pass
    bus.settimeout(timeout)
    try:
        bus.send(struct.pack("=IB3x8s", 0x100 | device_id, 1, bytes([command])))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            bus.settimeout(remaining)
            feedback_id, length, data = struct.unpack("=IB3x8s", bus.recv(16))
            if feedback_id == device_id and length in lengths and data[0] == command:
                return data[:length]
    except (TimeoutError, socket.timeout):
        return None


def collect_motor(bus, motor, start, timeout):
    """One row of fresh measurements; never fill a failed query with cached data."""
    now = time.monotonic()
    if math.isnan(motor.torque_constant) and now >= motor.retry_b0_at:
        motor.retry_b0_at = now + 1.0
        data = query(bus, motor.device_id, 0xB0, timeout)
        if data is not None:
            constant = struct.unpack("<f", data[2:6])[0]
            if math.isfinite(constant) and 0.01 <= constant <= 10.0:
                motor.torque_constant = constant
                print(f"{motor.name}: B0={constant:g} N m/A", file=sys.stderr)

    row = dict.fromkeys(FIELDS, math.nan)
    row.update(record_type="feedback", motor=motor.name, motor_id=motor.device_id, encoder_count=None,
               zero_count=motor.zero_count, c2_target_count=None,
               torque_constant_nm_per_a=motor.torque_constant)
    for command, timestamp_field in ((0xA3, "position_elapsed_s"), (0xA2, "velocity_elapsed_s"),
                                     (0xA1, "current_elapsed_s")):
        data = query(bus, motor.device_id, command, timeout)
        if data is None:
            continue
        row[timestamp_field] = time.monotonic() - start  # Host receipt time, not motor acquisition time.
        if command == 0xA3:
            count = struct.unpack("<i", data[3:7])[0]
            if motor.zero_count is None:
                motor.zero_count = count
                print(f"{motor.name}: startup A3 zero={count} counts", file=sys.stderr)
            position = motor.encoder_sign * (count - motor.zero_count) * RADIANS_PER_COUNT / motor.gear_ratio
            row.update(encoder_count=count, zero_count=motor.zero_count, position_unwrapped_rad=position,
                       position_rad=(position + math.pi) % (2 * math.pi) - math.pi)
        elif command == 0xA2:
            rpm = struct.unpack("<i", data[1:5])[0] * 0.01
            row["velocity_rad_s"] = motor.encoder_sign * rpm * (2 * math.pi / 60) / motor.gear_ratio
        else:
            row["current_a"] = struct.unpack("<i", data[1:5])[0] * 0.001
            row["torque_nm"] = motor.torque_sign * row["current_a"] * motor.torque_constant
    row.update(time_unix_s=time.time(), elapsed_s=time.monotonic() - start)
    return row


def receive_c2_targets(bus, motors, start, stopped, pending):
    """Capture each observed host-ID C2 command, separately from the query socket."""
    by_host_id = {0x100 | motor.device_id: motor for motor in motors}
    try:
        while not stopped.is_set():
            try:
                frame = bus.recv(16)
            except (TimeoutError, socket.timeout):
                continue
            received = time.monotonic()
            wall_time = time.time()
            host_id, length, data = struct.unpack("=IB3x8s", frame)
            motor = by_host_id.get(host_id)
            # Excludes motor-side C2 replies, which contain position feedback.
            if motor is None or length != 5 or data[0] != 0xC2:
                continue
            row = dict.fromkeys(FIELDS, math.nan)
            row.update(record_type="c2_target", time_unix_s=wall_time, elapsed_s=received - start,
                       motor=motor.name, motor_id=motor.device_id, encoder_count=None, zero_count=None,
                       c2_target_count=struct.unpack("<i", data[1:5])[0])
            pending.put(row)
    except OSError as error:
        pending.put(error)  # Surface a broken capture socket rather than silently losing command logging.


def write_c2_targets(writer, pending):
    while True:
        try:
            row = pending.get_nowait()
        except queue.Empty:
            return
        if isinstance(row, OSError):
            raise row
        writer.writerow(row)


def run(args):
    motors = [Motor(name, getattr(args, f"{name}_id"), getattr(args, f"{name}_gear_ratio"),
                    getattr(args, f"{name}_encoder_sign"), getattr(args, f"{name}_torque_sign"))
              for name in args.motors]
    with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus, \
            socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as target_bus:
        filters = b"".join(struct.pack("=II", motor.device_id, 0xC00007FF) for motor in motors)
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
        bus.bind((args.interface,))
        target_filters = b"".join(struct.pack("=II", 0x100 | motor.device_id, 0xC00007FF) for motor in motors)
        target_bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, target_filters)
        target_bus.settimeout(0.05)
        target_bus.bind((args.interface,))
        with open(args.output, "x", newline="") as log:  # Refuse to overwrite an existing recording.
            writer = csv.DictWriter(log, fieldnames=FIELDS)
            writer.writeheader()
            print(f"READ QUERIES ONLY on {args.interface}; target {args.rate:g} Hz per motor; "
                  f"passively capturing outgoing C2; CSV: {args.output}")
            start = deadline = time.monotonic()
            next_report = start
            period = 1 / args.rate
            stopped, pending = threading.Event(), queue.Queue()
            receiver = threading.Thread(target=receive_c2_targets,
                                        args=(target_bus, motors, start, stopped, pending), daemon=True)
            receiver.start()
            try:
                while not args.duration or time.monotonic() - start < args.duration:
                    rows = [collect_motor(bus, motor, start, args.timeout) for motor in motors]
                    writer.writerows(rows)
                    write_c2_targets(writer, pending)
                    now = time.monotonic()
                    if now >= next_report:
                        print(" | ".join(f"{row['motor']}: pos={row['position_unwrapped_rad']:.4f} rad, "
                                         f"vel={row['velocity_rad_s']:.4f} rad/s, torque={row['torque_nm']:.4f} N m"
                                         for row in rows), flush=True)
                        log.flush()
                        next_report = now + 1.0
                    deadline += period
                    if deadline > now:
                        time.sleep(deadline - now)
                    else:
                        deadline = now  # No catch-up burst when queries exceed the target period.
            finally:
                stopped.set()
                receiver.join()
                write_c2_targets(writer, pending)


def request_stop(signum, frame):
    raise KeyboardInterrupt


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped recording; no motor-control or disable commands sent.")
    except OSError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
