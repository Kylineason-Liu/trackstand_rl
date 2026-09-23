#!/usr/bin/env python3
"""
Actively polls the drive (rear) motor via CAN for speed, then publishes
the rear wheel angular rate.

Unlike the old passive ``candump`` approach, this node **actively queries**
the motor with ``READ_SPEED`` (A2) at a configurable rate.  The GIM motor
protocol is query-response — the motor only sends data when the host asks
for it.

Publishes
---------
  /rear_info (std_msgs/Float64MultiArray)  — 2 floats:
      [rear_rate (rad/s), publish_ts]

      No low-pass filter is applied — wheel rate is published raw.

Parameters
----------
  can_interface        — SocketCAN interface name    (default: can0)
  drive_can_id         — Host → motor CAN ID          (default: 102 → motor addr 0x02)
  drive_feedback_id    — Motor → host CAN ID          (default: 002)
  drive_gear_ratio     — Motor rev → wheel rev        (default: 1.0)
  poll_rate_hz          — How often to query the motor (default: 200.0)
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


def _parse_a2(data: bytes) -> float | None:
    """Parse an A2 (READ_SPEED) motor response.

    Response layout (5 bytes):
      byte 0:    cmd echo (0xA2)
      bytes 1-4: speed (int32 LE, × 0.01 rpm)

    Returns speed in rad/s, or None on parse failure.

    The command echo is checked: frames carrying this motor's feedback ID
    are not all A2 replies (``main_balance``'s C1 echoes share the ID),
    and decoding one of those as a speed yields garbage.
    """
    if len(data) < 5 or data[0] != 0xA2:
        return None
    try:
        raw = struct.unpack("<i", data[1:5])[0]
    except struct.error:
        return None
    rpm = raw * 0.01
    return float(rpm * (2.0 * math.pi / 60.0))


class RearInfoPublisher(Node):
    """CAN poll → rear wheel angular rate → /rear_info."""

    def __init__(self) -> None:
        super().__init__("rear_info_publisher")

        # ── Parameters ──
        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("drive_can_id", "102")          # host → motor
        self.declare_parameter("drive_feedback_id", "002")     # motor → host
        self.declare_parameter("drive_gear_ratio", 1.0)
        self.declare_parameter("poll_rate_hz", 200.0)

        can_iface: str = self.get_parameter("can_interface").value
        self._host_id: int = int(self.get_parameter("drive_can_id").value, 16)
        self._fb_id: int = int(self.get_parameter("drive_feedback_id").value, 16)
        self._gear: float = self.get_parameter("drive_gear_ratio").value
        self._poll_dt: float = 1.0 / self.get_parameter("poll_rate_hz").value

        # ── State ──
        self._rate: float = 0.0
        self._lock = threading.Lock()
        self._running: bool = True

        # ── Open CAN bus ──
        self.get_logger().info(f"Opening CAN bus {can_iface} …")
        self._bus = can.interface.Bus(
            channel=can_iface, bustype="socketcan", receive_own_messages=False)

        apply_feedback_filter(self._bus, self._fb_id)

        # ── Publisher ──
        self._pub = self.create_publisher(Float64MultiArray, "rear_info", 10)

        # ── Polling thread ──
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        # ── ROS timer for publishing ──
        self._pub_timer = self.create_timer(self._poll_dt, self._publish_tick)

        self.get_logger().info(
            f"RearInfoPublisher | can={can_iface} host=0x{self._host_id:03X} "
            f"fb=0x{self._fb_id:03X} gear={self._gear} "
            f"poll={1/self._poll_dt:.0f} Hz")

    # ─────────────────────────────────────────────────────────────
    # CAN polling loop (daemon thread)
    # ─────────────────────────────────────────────────────────────
    def _poll_loop(self) -> None:
        """Continuously query A2 and parse motor responses."""
        query_A2 = can.Message(
            arbitration_id=self._host_id,
            data=bytes([0xA2]),
            is_extended_id=False,
        )
        deadline = time.monotonic()

        # Response budget, leaving headroom inside one poll cycle.
        resp_timeout = max(self._poll_dt * 0.5, 0.001)

        while self._running:
            resp = query(self._bus, query_A2, self._fb_id, resp_timeout)

            if resp is None:
                self.get_logger().warn(
                    "No A2 response from drive motor", throttle_duration_sec=5.0)
            else:
                speed_rad_s = _parse_a2(resp)
                if speed_rad_s is not None:
                    # Apply gear ratio: motor speed → wheel speed
                    with self._lock:
                        self._rate = speed_rad_s / self._gear

            # Sleep until next poll
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
            rate = self._rate
        msg = Float64MultiArray()
        msg.data = [float(rate), time.time()]
        self._pub.publish(msg)

    def destroy_node(self) -> None:
        self._running = False
        if hasattr(self, "_bus") and self._bus is not None:
            self._bus.shutdown()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RearInfoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
