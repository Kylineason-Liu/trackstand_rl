#!/usr/bin/env python3
"""
Reads SBUS remote controller frames via USB serial and publishes
target steer angle and drive RPM on /rc_info.

Publishes
---------
  /rc_info (std_msgs/Float64MultiArray)  — 2 floats:
      [drive_rpm, steer_rad]   target drive RPM + target steer angle (rad)

Parameters
----------
  port         — SBUS receiver serial device (default: /dev/ttyUSB1)
  baudrate     — serial baud rate            (default: 115200)
  frame_len    — SBUS frame length: 25 or 35 (default: 35)
  steer_ch     — SBUS channel index for steer stick   (default: 0)
  throttle_ch  — SBUS channel index for throttle stick (default: 2)
  center_value — raw channel value at stick centre       (default: 992)
"""

from __future__ import annotations
import math
import time

import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class RcPublisher(Node):
    """SBUS serial → steer + throttle channels → [drive_rpm, steer_rad] → /rc_info."""

    def __init__(self) -> None:
        super().__init__("rc_publisher")

        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("frame_len", 35)
        self.declare_parameter("steer_ch", 0)
        self.declare_parameter("throttle_ch", 2)
        self.declare_parameter("servo_ch", 6)
        self.declare_parameter("center_value", 992)

        port: str = self.get_parameter("port").value
        baudrate: int = self.get_parameter("baudrate").value
        self._frame_len: int = self.get_parameter("frame_len").value
        self._steer_ch: int = self.get_parameter("steer_ch").value
        self._throttle_ch: int = self.get_parameter("throttle_ch").value
        self._servo_ch: int = self.get_parameter("servo_ch").value
        self._center: int = self.get_parameter("center_value").value

        self._steer_rad: float = 0.0
        self._drive_rpm: float = 0.0
        self._servo_up: float = 0.0

        # ── Open serial ──
        self.get_logger().info(f"Opening SBUS receiver on {port} …")
        self._ser = serial.Serial(port, baudrate, timeout=0.1)
        self._buffer = bytearray()

        # ── Publisher ──
        self._pub = self.create_publisher(Float64MultiArray, "rc_info", 10)

        # ── Timers ──
        self._read_timer = self.create_timer(0.02, self._read_data)
        self._pub_timer = self.create_timer(0.02, self._publish)

        self.get_logger().info(
            f"RcPublisher | port={port} frame_len={self._frame_len} "
            f"throttle_ch={self._throttle_ch} steer_ch={self._steer_ch} servo_ch={self._servo_ch} centre={self._center}")

    # ─────────────────────────────────────────────────────────────
    # SBUS frame reading (same as sbus_reader.py)
    # ─────────────────────────────────────────────────────────────
    def _read_data(self) -> None:
        if self._ser.in_waiting > 0:
            raw = self._ser.read(self._ser.in_waiting)
            self._buffer.extend(raw)

        # Search for the last complete frame (skip partial at buffer end)
        max_pos = len(self._buffer) - self._frame_len
        if max_pos < 0:
            return
        pos = self._buffer.rfind(b"\x0F", 0, max_pos + 1)
        if pos == -1:
            return

        frame = self._buffer[pos:pos + self._frame_len]

        # Trim consumed bytes to keep buffer small
        self._buffer = self._buffer[pos + self._frame_len:]

        if self._frame_len == 25:
            channels = self._decode_25(frame)
        elif self._frame_len == 35:
            channels = self._decode_35(frame)
        else:
            return

        if channels is None or len(channels) <= max(self._steer_ch, self._throttle_ch):
            return

        steer_raw = channels[self._steer_ch]
        throttle_raw = channels[self._throttle_ch]
        servo_raw = channels[self._servo_ch]
        # -- Convert raw steer angle (with respect to initial steer zero point) command to radians --
        # The steer angle command is limited at +-25 degrees, corresponding to the mechanical limit of the steering mechanism.
        #                        -800 ~ +800         -25 ~ +25 degrees    -25 ~ +25 degrees (converted to radians)                
        self._steer_rad = ((steer_raw - self._center) / 40.0) * math.pi / 180.0

        # ── Same formula as joy_to_cmd.py ──
        #    linear.x = (axes[3] - 992) / 8.0  →  drive_rpm
        self._drive_rpm = ((throttle_raw - self._center) / 16.0) * 2.0444

        # -- Convert raw servo command to normalized value --
        #self._servo_cmd = (servo_raw - self._center) / 1024.0
        self._servo_up = 1.0 if servo_raw < 200 else 0.0 # servo_raw is either 192 (means servo up) or 1792 (down)

    def _decode_25(self, f: bytes) -> list[int]:
        """Standard 25-byte SBUS (11-bit channels, 0–2047)."""
        if f[0] != 0x0F or f[24] != 0x00:
            return
        ch = []
        for i in range(16):
            b = 1 + i * 11 // 8
            bit = i * 11 % 8
            v = (f[b] >> bit) | (f[b + 1] << (8 - bit))
            if bit > 5:
                v |= f[b + 2] << (16 - bit)
            ch.append(v & 0x07FF)
        return ch

    def _decode_35(self, f: bytes) -> list[int]:
        """35-byte decoding (each channel = 2 bytes, high byte first)."""
        if f[0] != 0x0F:
            return
        ch = []
        for i in range(16):
            high = f[1 + i * 2]
            low = f[2 + i * 2]
            ch.append((high << 8) | low)
        return ch

    # ─────────────────────────────────────────────────────────────
    # ROS publish
    # ─────────────────────────────────────────────────────────────
    def _publish(self) -> None:
        msg = Float64MultiArray()
        msg.data = [float(self._drive_rpm), float(self._steer_rad), float(self._servo_up), time.time()]
        self._pub.publish(msg)

    def destroy_node(self) -> None:
        if hasattr(self, "_ser") and self._ser.is_open:
            self._ser.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RcPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
