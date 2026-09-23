#!/usr/bin/env python3
"""
Measure steering motor (ID 1) speed-command latency.

For each of 10 loops:
  1. Enable:  C1=0  → ``101#C100000000``
  2. Send three speed commands in sequence (0.2 s apart):
       a. ``101#C110200000``  (≈ 82 rpm)
       b. ``101#C110800000``  (≈ 328 rpm)
       c. ``101#C110500000``  (≈ 205 rpm)
     For each, record the send timestamp, then poll A2 until the
     motor speed is within ``--threshold`` (rpm×100) of the target.
     Record the arrival timestamp; delay = arrival − send.
  3. Disable: C1=0 → ``101#C100000000``

After all loops, save a time-series line plot of the 30 delays to
``/home/ubuntu/source/new_balance/plots/`` with a mean-line overlay.

Usage
-----
    python compute_motor_delay.py [--loops N] [--interface can0]
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from datetime import datetime

import can
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── CAN IDs ──────────────────────────────────────────────────────────
HOST_STEER_ID = 0x101        # host → steer motor
STEER_FB_ID   = 0x001        # steer motor → host

# ── Speed commands (C1 value = rpm × 100) ────────────────────────────
_SPEED_TARGETS_C1 = [
    0x00001A10,   # 6672 → 66.72 rpm
    0x00001010,   # 4112 → 41.12 rpm
    0x00002510,   # 9488 → 94.88 rpm
]
_SPEED_LABELS = ["67 rpm", "41 rpm", "95 rpm"]


# ── Helpers ───────────────────────────────────────────────────────────

def _make_c1_data(c1_value: int) -> bytes:
    """Build a C1 speed-command CAN payload."""
    return bytes([0xC1]) + struct.pack("<i", c1_value)


def _send_a2_query(bus: can.BusABC) -> None:
    """Send an A2 (READ_SPEED) query to the steer motor."""
    bus.send(can.Message(
        arbitration_id=HOST_STEER_ID,
        data=bytes([0xA2]),
        is_extended_id=False,
    ))


def _parse_a2_raw(data: bytes) -> int | None:
    """Extract raw speed (rpm×100) from an A2 response, or None."""
    if len(data) < 5 or data[0] != 0xA2:
        return None
    try:
        return struct.unpack("<i", data[1:5])[0]
    except struct.error:
        return None


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Motor speed-command latency")
    parser.add_argument("--loops", type=int, default=50,
                        help="Number of loops (default: 50)")
    parser.add_argument("--threshold", type=int, default=200,
                        help="Speed tolerance in rpm×100 (default: 200 = 2 rpm)")
    parser.add_argument("--interface", type=str, default="can0",
                        help="SocketCAN interface (default: can0)")
    parser.add_argument("--timeout", type=float, default=0.195,
                        help="Max seconds to wait for speed settle (default: 0.195s)")
    args = parser.parse_args()

    bus = can.interface.Bus(
        channel=args.interface, bustype="socketcan", receive_own_messages=False)

    # ── Per-command delay buckets ──
    delays: list[list[float]] = [[] for _ in _SPEED_TARGETS_C1]
    # Flat list for combined mean
    all_delays: list[float] = []

    n_total = args.loops * len(_SPEED_TARGETS_C1)
    n_done = 0
    n_timeout = 0

    print(f"Motor speed delay: {args.loops} loops × {len(_SPEED_TARGETS_C1)} speeds "
          f"= {n_total} samples on {args.interface}")
    print("Press Ctrl-C to stop early.\n")

    for _ in range(args.loops):
        # ── 1. Enable: C1=0 ──
        bus.send(can.Message(
            arbitration_id=HOST_STEER_ID,
            data=_make_c1_data(0),
            is_extended_id=False,
        ))
        time.sleep(0.02)

        # ── 2. Three speed commands ──
        for i_target, target_c1 in enumerate(_SPEED_TARGETS_C1):
            # Send speed command
            send_ts = time.perf_counter()
            bus.send(can.Message(
                arbitration_id=HOST_STEER_ID,
                data=_make_c1_data(target_c1),
                is_extended_id=False,
            ))

            # Poll A2 until speed is within threshold
            settled = False
            deadline = time.perf_counter() + args.timeout
            delay_ms = args.timeout * 1000  # default = timeout

            while time.perf_counter() < deadline:
                _send_a2_query(bus)

                # Wait for response (tight poll)
                resp_deadline = time.perf_counter() + 0.005
                while time.perf_counter() < resp_deadline:
                    try:
                        rx = bus.recv(timeout=0.0005)
                    except can.CanError:
                        break
                    if rx is None or rx.arbitration_id != STEER_FB_ID:
                        continue
                    actual = _parse_a2_raw(rx.data)
                    if actual is None:
                        continue

                    if abs(actual - target_c1) <= args.threshold:
                        arrival_ts = time.perf_counter()
                        delay_ms = (arrival_ts - send_ts) * 1000.0
                        settled = True
                        break

                if settled:
                    break

            delays[i_target].append(delay_ms)
            all_delays.append(delay_ms)
            n_done += 1
            if not settled:
                n_timeout += 1

            status = "✓" if settled else "✗ TIMEOUT"
            pct = n_done / n_total * 100
            print(f"  [{n_done:3d}/{n_total}] {pct:5.1f}%  {_SPEED_LABELS[i_target]:>8s}  "
                  f"delay={delay_ms:.3f} ms  {status}")

            # 0.995 s between commands
            time.sleep(args.timeout - (delay_ms / 1000.0))

        # ── 3. Disable: C1=0 ──
        bus.send(can.Message(
            arbitration_id=HOST_STEER_ID,
            data=_make_c1_data(0),
            is_extended_id=False,
        ))
        time.sleep(0.05)

    bus.shutdown()

    # ── Summary ──
    print(f"\n--- Results ({n_done} samples, {n_timeout} timeouts) ---")
    for i, (label, ds) in enumerate(zip(_SPEED_LABELS, delays)):
        a = np.array(ds)
        print(f"  {label}: mean={a.mean():.3f}  std={a.std():.3f}  "
              f"min={a.min():.3f}  max={a.max():.3f} ms")
    a_all = np.array(all_delays)
    print(f"  ALL:   mean={a_all.mean():.3f}  std={a_all.std():.3f}  "
          f"min={a_all.min():.3f}  max={a_all.max():.3f} ms")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Motor Speed-Command Delay — steer ID 1 ({args.interface})", fontsize=13)

    # Single-color line for all samples
    ax.plot(all_delays, color="#1f77b4", linewidth=0.6, label="delay")

    # Overall mean line
    mean_all = a_all.mean()
    ax.axhline(y=mean_all, color="red", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.text(
        x=len(all_delays) * 0.02, y=mean_all,
        s=f"Overall mean = {mean_all:.3f} ms",
        color="red", fontsize=10, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8),
    )

    ax.set_xlabel("Sample #")
    ax.set_ylabel("Delay (ms)")
    ax.set_xlim(0, len(all_delays))
    ax.legend()

    plt.tight_layout()

    # ── Save ──
    out_dir = "/home/ubuntu/source/new_balance/plots"
    os.makedirs(out_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(out_dir, f"motor_delay_{args.interface}_{ts_str}.png")
    fig.savefig(fname, dpi=150)
    print(f"Plot saved → {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
