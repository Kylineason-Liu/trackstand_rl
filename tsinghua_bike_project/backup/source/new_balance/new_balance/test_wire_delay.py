#!/usr/bin/env python3
"""
Measure CAN bus physical round-trip delay.

Sends a broadcast command ``0FF#BA`` on ``can0``, records the send timestamp,
then measures the time until each motor responds on its feedback CAN ID.

  Steer motor:  host → 0x101 (or broadcast 0x0FF), motor responds on 0x001
  Drive motor:  host → 0x102 (or broadcast 0x0FF), motor responds on 0x002

Usage
-----
    python compute_wire_delay.py [--count N] [--interval S] [--interface can0]

    --count     Number of pings to send (default: 500)
    --interval  Seconds between pings (default: 0.02, i.e. 50 Hz)
    --interface SocketCAN interface (default: can0)

Output
------
    Console prints mean / std / min / max for each motor on exit.
    Plots are saved to ``/home/ubuntu/source/new_balance/plots/``
    with a timestamped filename (directory created if missing).
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import can
import matplotlib
matplotlib.use("Agg")  # headless — no GUI window needed
import matplotlib.pyplot as plt
import numpy as np

# ── CAN IDs ──────────────────────────────────────────────────────────
BROADCAST_ID = 0x0FF
STEER_FB_ID  = 0x001   # steer motor → host
DRIVE_FB_ID  = 0x002   # drive motor → host

BROADCAST_CMD = bytes([0xBA])


def main() -> None:
    parser = argparse.ArgumentParser(description="CAN wire delay measurement")
    parser.add_argument("--count", type=int, default=500,
                        help="Number of broadcast pings (default: 500)")
    parser.add_argument("--interval", type=float, default=0.02,
                        help="Seconds between pings (default: 0.02, 50 Hz)")
    parser.add_argument("--interface", type=str, default="can0",
                        help="SocketCAN interface (default: can0)")
    args = parser.parse_args()

    # ── Open CAN bus ──
    bus = can.interface.Bus(
        channel=args.interface, bustype="socketcan", receive_own_messages=False)

    steer_delays: list[float] = []
    drive_delays: list[float] = []

    print(f"Sending {args.count} broadcasts 0FF#BA at "
          f"{1.0 / args.interval:.0f} Hz on {args.interface} …")
    print("Press Ctrl-C to stop early.\n")

    for _ in range(args.count):
        # ── Send broadcast ──
        send_ts = time.perf_counter()

        msg = can.Message(
            arbitration_id=BROADCAST_ID,
            data=BROADCAST_CMD,
            is_extended_id=False,
        )
        try:
            bus.send(msg)
        except can.CanError:
            time.sleep(args.interval)
            continue

        # ── Collect responses ──
        steer_rcvd = False
        drive_rcvd = False
        deadline = time.perf_counter() + 0.01  # 10 ms timeout

        while time.perf_counter() < deadline and not (steer_rcvd and drive_rcvd):
            try:
                rx = bus.recv(timeout=0.0005)
            except can.CanError:
                break
            if rx is None:
                continue

            rx_ts = time.perf_counter()
            delay_ms = (rx_ts - send_ts) * 1000.0

            if rx.arbitration_id == STEER_FB_ID and not steer_rcvd:
                steer_delays.append(delay_ms)
                steer_rcvd = True
            elif rx.arbitration_id == DRIVE_FB_ID and not drive_rcvd:
                drive_delays.append(delay_ms)
                drive_rcvd = True

        # ── Rate-limit ──
        elapsed = time.perf_counter() - send_ts
        sleep_s = args.interval - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)

    bus.shutdown()

    # ── Summary ──
    print(f"\nResults ({len(steer_delays)} steer, {len(drive_delays)} drive samples):")
    for name, delays in (("Steer (0x001)", steer_delays), ("Drive (0x002)", drive_delays)):
        if delays:
            a = np.array(delays)
            print(f"  {name}:  mean={a.mean():.3f}  std={a.std():.3f}  "
                  f"min={a.min():.3f}  max={a.max():.3f} ms")
        else:
            print(f"  {name}:  NO RESPONSES")

    # ── Plot ──
    fig, ax_ts = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"CAN Wire Delay — 0FF#BA broadcast ({args.interface})", fontsize=13)

    colors = {"Steer (0x001)": "#1f77b4", "Drive (0x002)": "#ff7f0e"}

    # -- Time series (line plot, 0–1.0 ms) --
    for name, delays, c in (("Steer (0x001)", steer_delays, colors["Steer (0x001)"]),
                              ("Drive (0x002)", drive_delays, colors["Drive (0x002)"])):
        if delays:
            ax_ts.plot(delays, color=c, linewidth=0.6, label=name)
            mean_val = np.mean(delays)
            ax_ts.axhline(y=mean_val, color=c, linestyle="--", linewidth=1.0, alpha=0.8)
            ax_ts.text(
                x=len(delays) * 0.02, y=mean_val,
                s=f"  {name} mean = {mean_val:.3f} ms",
                color=c, fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c, alpha=0.7),
            )
    ax_ts.set_xlabel("Sample #")
    ax_ts.set_ylabel("Delay (ms)")
    ax_ts.set_ylim(0, 1.0)
    ax_ts.legend()

    plt.tight_layout()

    # ── Save to file ──
    out_dir = "/home/ubuntu/source/new_balance/plots"
    os.makedirs(out_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(out_dir, f"can_delay_{args.interface}_{ts_str}.png")
    fig.savefig(fname, dpi=150)
    print(f"Plot saved → {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
