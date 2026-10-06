#!/usr/bin/env python3
# Raw C2 sine at 200 Hz. C2 targets absolute motor position (16384 counts/revolution).
# Defaults match deployment: can0, steering TX 0x102, gear ratio 1, sign -1.
# Native positive counts turn left from the rider's view; sign -1 reverses the sine.
# Read A3 once to centre the sine on startup position; no motor-origin reset.
# No continuous feedback, limit configuration, watchdog, or disable-on-exit.
import argparse
import math
import socket
import struct
import time

RATE_HZ = 200.0


def parse_args():
    p = argparse.ArgumentParser(prog="steering_sine_position.py", description="Send raw C2 sine commands at 200 Hz.")
    p.add_argument("--execute", action="store_true", help="Send commands (also the default).")
    p.add_argument("--preview", action="store_true", help="Print frames only; do not open CAN.")
    p.add_argument("--interface", default="can0")
    p.add_argument("--motor-id", type=lambda value: int(value, 0), default=2)
    p.add_argument("--gear-ratio", type=float, default=1.0)
    p.add_argument("--sign", type=int, default=-1)
    p.add_argument("--amplitude", type=float, default=1.0, help="Peak joint radians (default: 1).")
    p.add_argument("--period", type=float, default=10.0, help="Seconds per cycle (default: 10).")
    p.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 runs until Ctrl-C.")
    return p.parse_args()


def c2_payload(a, elapsed, zero_count=0):
    angle_rad = a.amplitude * math.sin(2 * math.pi * (elapsed % a.period) / a.period)
    count = round(a.sign * angle_rad * a.gear_ratio * 16384 / (2 * math.pi))
    return b"\xc2" + struct.pack("<i", zero_count + count)


def read_startup_count(bus, motor_id):
    """Read A3's signed multi-turn count, bytes 3..6, before sending any C2."""
    bus.send(struct.pack("=IB3x8s", 0x100 | motor_id, 1, b"\xa3"))
    deadline = time.monotonic() + 1.0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("No A3 startup position received; no C2 commands sent.")
            bus.settimeout(remaining)
            feedback_id, length, data = struct.unpack("=IB3x8s", bus.recv(16))
            if feedback_id == motor_id and length == 7 and data[0] == 0xA3:
                return struct.unpack("<i", data[3:7])[0]
    finally:
        bus.settimeout(None)


def execute(a):
    with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
        bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, struct.pack("=II", a.motor_id, 0xC00007FF))
        bus.bind((a.interface,))
        zero_count = read_startup_count(bus, a.motor_id)
        print(f"Startup A3 centre: {zero_count} counts; commanding the sine at nominal {RATE_HZ:g} Hz.", flush=True)
        start = time.monotonic()
        while True:
            tick_start = time.monotonic()
            elapsed = tick_start - start
            if a.duration and elapsed >= a.duration:
                break
            bus.send(struct.pack("=IB3x8s", 0x100 | a.motor_id, 5, c2_payload(a, elapsed, zero_count)))
            time.sleep(max(0, 1 / RATE_HZ - (time.monotonic() - tick_start)))


if __name__ == "__main__":
    args = parse_args()
    if args.preview:
        print("PREVIEW ONLY: startup count assumed 0; execution reads A3 once.")
        for i in range(math.ceil((args.duration or args.period) * RATE_HZ)):
            print(f"t={i / RATE_HZ:.3f}s {0x100 | args.motor_id:03X}#{c2_payload(args, i / RATE_HZ).hex().upper()}")
    else:
        execute(args)
