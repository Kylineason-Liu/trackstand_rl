#!/usr/bin/env python3
"""
Actively polls the steering motor via CAN for position, then publishes the
steering joint angle.

This node queries the motor with ``READ_ANGLE`` (A3) at a configurable
rate.  A3 returns a **multi-turn cumulative angle** from the motor
firmware — no Python unwrapping is needed.  The motor itself tracks how
many full revolutions have occurred.

Motor *speed* is deliberately not polled.  ``main_balance`` commands the
steer motor with C1 speed frames and lets the motor's own speed loop close
them, so nothing downstream consumes a measured steer rate; dropping the
A4 query halves this node's CAN traffic and its per-cycle latency.

Angle limit enforcement is the responsibility of ``main_balance`` —
this node is pure hardware I/O.

No low-pass filter is applied — the angle is published raw.

Publishes
---------
  /steer_info (std_msgs/Float64MultiArray)  — 2 floats:
      [steer_angle (rad), publish_ts]

Parameters
----------
  can_interface        — SocketCAN interface name     (default: can0)
  steer_can_id         — Host → motor CAN ID           (default: 101)
  steer_feedback_id    — Motor → host CAN ID           (default: 001)
  steer_gear_ratio     — Motor rev → steering joint rev (default: 1.0)
  poll_rate_hz          — How often to query the motor  (default: 200.0)
  zero_on_start         — Latch current angle as zero   (default: true)
"""

from __future__ import annotations

import math
import struct
import threading
import time

import can
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from new_balance.can_io import apply_feedback_filter, query

_COUNTS_PER_REV = 16384
_RAD_PER_COUNT = 2.0 * math.pi / _COUNTS_PER_REV


def _parse_a3(data: bytes) -> int | None:
    """Parse an A3 (READ_ANGLE) motor response.

    Response layout (8 bytes):
      byte 0:    cmd echo (0xA3)
      bytes 1-2: single-turn angle (uint16 LE, 0–16383 counts)
      bytes 3-6: multi-turn angle  (int32  LE, cumulative counts)

    Returns the **multi-turn cumulative** encoder count, or None on failure.
    This value grows monotonically across revolutions — the motor firmware
    handles unwrapping internally.
    """
    if len(data) < 7 or data[0] != 0xA3:
        return None
    try:
        multi_raw = struct.unpack("<i", data[3:7])[0]
    except struct.error:
        return None
    return int(multi_raw)


class SteerInfoPublisher(Node):
    """CAN poll (A3) → cumulative steer angle → /steer_info."""

    def __init__(self) -> None:
        super().__init__("steer_info_publisher")

        # ── Parameters ──
        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("steer_can_id", "101")         # host → motor
        self.declare_parameter("steer_feedback_id", "001")    # motor → host
        self.declare_parameter("steer_gear_ratio", 1.0)
        self.declare_parameter("poll_rate_hz", 200.0)
        self.declare_parameter("zero_on_start", True)

        can_iface: str = self.get_parameter("can_interface").value
        self._host_id: int = int(self.get_parameter("steer_can_id").value, 16)
        self._fb_id: int = int(self.get_parameter("steer_feedback_id").value, 16)
        self._gear: float = self.get_parameter("steer_gear_ratio").value
        self._poll_dt: float = 1.0 / self.get_parameter("poll_rate_hz").value
        self._zero_on_start: bool = self.get_parameter("zero_on_start").value

        # ── State (protected by _lock) ──
        self._angle: float = 0.0     #  steer angle [rad] with respect to zero point
        self._lock = threading.Lock()
        self._running: bool = True

        # ── Open CAN bus ──
        self.get_logger().info(f"Opening CAN bus {can_iface} …")
        self._bus = can.interface.Bus(
            channel=can_iface, bustype="socketcan", receive_own_messages=False)

        apply_feedback_filter(self._bus, self._fb_id)

        # ── Publisher ──
        self._pub = self.create_publisher(Float64MultiArray, "steer_info", 10)

        # ── Polling thread ──
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        # ── ROS timer for publishing ──
        self._pub_timer = self.create_timer(self._poll_dt, self._publish_tick)

        self.get_logger().info(
            f"SteerInfoPublisher | can={can_iface} host=0x{self._host_id:03X} "
            f"fb=0x{self._fb_id:03X} gear={self._gear} "
            f"poll={1/self._poll_dt:.0f} Hz zero_on_start={self._zero_on_start}")

    # ─────────────────────────────────────────────────────────────
    # CAN polling loop (daemon thread)
    # ─────────────────────────────────────────────────────────────
    def _poll_loop(self) -> None:
        """Query A3 and track the cumulative steer angle."""
        query_A3 = can.Message( # read steer angle (multi-turn accumulated)
            arbitration_id=self._host_id,
            data=bytes([0xA3]),
            is_extended_id=False,
        )
        deadline = time.monotonic()

        # Response budget, leaving headroom inside one poll cycle.  With A4
        # gone there is a single query per cycle, so half the period is a
        # safe budget (same as rear_publisher).
        resp_timeout = max(self._poll_dt * 0.5, 0.001)

        # Per-cycle state (no locking needed — only this thread writes)
        zero_locked: bool = False
        zero_offset_angle_count: int = 0       # multi-turn counts at zero-lock instant

        while self._running:
            angle_count_parsed: int | None = None

            resp = query(self._bus, query_A3, self._fb_id, resp_timeout)
            if resp is not None:
                # multi-turn cumulative angle counts [converted to rad below]
                angle_count_parsed = _parse_a3(resp)

            if angle_count_parsed is None:
                self.get_logger().warn(
                    "No A3 response from steer motor", throttle_duration_sec=5.0)

            elif not zero_locked:
                # ── Latch zero offset on first good frame ──
                if self._zero_on_start:
                    zero_offset_angle_count = angle_count_parsed
                zero_locked = True
                self.get_logger().info(
                    f"Steer zero locked: multi-turn offset={zero_offset_angle_count} counts "
                    f"({zero_offset_angle_count * 360.0 / _COUNTS_PER_REV:.1f}°)")
                with self._lock:
                    self._angle = 0.0

            else:
                # ── Cumulative angle (motor firmware unwraps for us) ──
                net_angle_count = angle_count_parsed - zero_offset_angle_count
                net_angle = net_angle_count * _RAD_PER_COUNT / self._gear # [rad]

                with self._lock:
                    self._angle = net_angle

            # Sleep until next poll cycle
            deadline += self._poll_dt
            sleep_s = deadline - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                deadline = time.monotonic()

    # ─────────────────────────────────────────────────────────────
    # ROS publish tick
    # ─────────────────────────────────────────────────────────────
    def _publish_tick(self) -> None:
        with self._lock:
            angle = -self._angle

        msg = Float64MultiArray()
        msg.data = [float(angle), time.time()]
        self._pub.publish(msg)

    def destroy_node(self) -> None:
        self._running = False
        if hasattr(self, "_bus") and self._bus is not None:
            self._bus.shutdown()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SteerInfoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
