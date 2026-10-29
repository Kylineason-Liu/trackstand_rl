"""Read the bicycle's CRSF/ExpressLRS receiver and publish kickstand authority."""

from __future__ import annotations

import os
import math
import time
from typing import Sequence

import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Bool, Float32MultiArray

from .crsf import (
    CRC_MODES,
    CRSF_SYNC_BYTES,
    BoolDebouncer,
    CrsfStreamParser,
    decode_three_position_switch,
    normalize_axis,
)

IMU_SYNC = b"\x7e\x23"


def serial_port_candidates(
    preferred: str,
    extra: Sequence[str],
    excluded: Sequence[str],
) -> list[str]:
    """Return existing UART paths in ros2_ws discovery order."""
    excluded_real = {
        os.path.realpath(path) for path in excluded if path and os.path.exists(path)
    }
    ordered = [preferred, *extra]
    by_path = "/dev/serial/by-path"
    if os.path.isdir(by_path):
        ordered.extend(
            os.path.join(by_path, name)
            for name in sorted(os.listdir(by_path))
            if "port" in name and "usbv2" not in name
        )
    ordered.extend(f"/dev/ttyUSB{index}" for index in range(8))
    result: list[str] = []
    seen: set[str] = set()
    for path in ordered:
        if not path or not os.path.exists(path):
            continue
        real = os.path.realpath(path)
        if real in excluded_real or real in seen:
            continue
        seen.add(real)
        result.append(path)
    return result


def score_crsf_port(port: str, baudrate: int, probe_s: float, crc_mode: str) -> int:
    """Score a serial device by the valid CRSF frames it emits."""
    try:
        with serial.Serial(port, baudrate, timeout=0.05) as receiver:
            receiver.reset_input_buffer()
            data = bytearray()
            deadline = time.monotonic() + max(0.1, probe_s)
            while time.monotonic() < deadline:
                chunk = receiver.read(256)
                if chunk:
                    data.extend(chunk)
                elif data:
                    break
    except (OSError, serial.SerialException):
        return -1
    if IMU_SYNC in data:
        return -1
    frames = CrsfStreamParser(crc_mode).feed(bytes(data))
    if frames:
        return 1000 + len(frames) * 10
    return sum(data.count(value) for value in CRSF_SYNC_BYTES)


def discover_crsf_port(
    preferred: str,
    extra: Sequence[str],
    excluded: Sequence[str],
    baudrate: int,
    crc_mode: str,
) -> str | None:
    """Select the non-IMU UART that produces CRSF traffic."""
    best_port = None
    best_score = 0
    for port in serial_port_candidates(preferred, extra, excluded):
        score = score_crsf_port(port, baudrate, 0.35, crc_mode)
        if score > best_score:
            best_port = port
            best_score = score
    return best_port


class RcKickstand(Node):
    def __init__(self) -> None:
        super().__init__("trackstand_rc_kickstand")
        self.declare_parameter("input_protocol", "crsf")
        self.declare_parameter(
            "port",
            "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
        )
        self.declare_parameter("auto_detect_port", True)
        self.declare_parameter(
            "port_candidates",
            [
                "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
                "/dev/serial/by-path/platform-xhci-hcd.1-usb-0:1.2:1.0-port0",
                "/dev/ttyUSB1",
                "/dev/serial/by-path/platform-xhci-hcd.0-usb-0:1:1.0-port0",
            ],
        )
        self.declare_parameter(
            "exclude_port_candidates",
            [
                "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
                "/dev/serial/by-path/platform-xhci-hcd.1-usb-0:1.1:1.0-port0",
                "/dev/ttyUSB0",
            ],
        )
        self.declare_parameter("baudrate", 420000)
        self.declare_parameter("auto_detect_baudrate", True)
        self.declare_parameter("baudrate_candidates", [420000, 416666, 400000, 921600])
        self.declare_parameter("baudrate_probe_sec", 2.0)
        self.declare_parameter("crsf_crc_mode", "soft")
        self.declare_parameter("input_min", 1000)
        self.declare_parameter("input_center", 1500)
        self.declare_parameter("input_max", 2000)
        self.declare_parameter("auto_detect_channel_range", True)
        self.declare_parameter("servo_pair_switch_channel", 7)
        self.declare_parameter("servo_pair_high_is_up", True)
        self.declare_parameter("servo_pair_end_band_fraction", 0.35)
        self.declare_parameter("switch_debounce_frames", 8)
        self.declare_parameter("read_rate_hz", 100.0)
        self.declare_parameter("reconnect_interval_s", 1.0)
        # Locomotive only: proportional sticks for v* / yaw-rate*. 0 disables publication.
        self.declare_parameter("throttle_channel", 0)
        self.declare_parameter("yaw_channel", 0)
        self.declare_parameter("throttle_reversed", False)
        self.declare_parameter("yaw_reversed", False)

        if str(self.get_parameter("input_protocol").value).lower() != "crsf":
            raise ValueError("the bike receiver input_protocol must be crsf")
        self._port = str(self.get_parameter("port").value)
        self._auto_detect_port = bool(self.get_parameter("auto_detect_port").value)
        self._port_candidates = [
            str(value) for value in self.get_parameter("port_candidates").value
        ]
        self._excluded_ports = [
            str(value) for value in self.get_parameter("exclude_port_candidates").value
        ]
        configured_baudrate = int(self.get_parameter("baudrate").value)
        self._auto_detect_baudrate = bool(self.get_parameter("auto_detect_baudrate").value)
        candidates = [int(value) for value in self.get_parameter("baudrate_candidates").value]
        self._baudrates = list(dict.fromkeys([configured_baudrate, *candidates]))
        self._baud_index = 0
        self._baud_probe_s = float(self.get_parameter("baudrate_probe_sec").value)
        self._crc_mode = str(self.get_parameter("crsf_crc_mode").value).lower()
        self._input_min = int(self.get_parameter("input_min").value)
        self._input_center = int(self.get_parameter("input_center").value)
        self._input_max = int(self.get_parameter("input_max").value)
        self._auto_detect_channel_range = bool(
            self.get_parameter("auto_detect_channel_range").value
        )
        self._channel = int(self.get_parameter("servo_pair_switch_channel").value)
        self._high_is_up = bool(self.get_parameter("servo_pair_high_is_up").value)
        self._band_fraction = float(
            self.get_parameter("servo_pair_end_band_fraction").value
        )
        debounce_frames = int(self.get_parameter("switch_debounce_frames").value)
        read_rate_hz = float(self.get_parameter("read_rate_hz").value)
        self._reconnect_interval_s = float(self.get_parameter("reconnect_interval_s").value)
        self._throttle_channel = int(self.get_parameter("throttle_channel").value)
        self._yaw_channel = int(self.get_parameter("yaw_channel").value)
        self._throttle_reversed = bool(self.get_parameter("throttle_reversed").value)
        self._yaw_reversed = bool(self.get_parameter("yaw_reversed").value)
        stick_channels = (self._throttle_channel, self._yaw_channel)
        self._publish_sticks = all(stick_channels)
        if (any(stick_channels) and not self._publish_sticks) or (self._publish_sticks and (
                not all(1 <= value <= 16 for value in stick_channels)
                or len({self._channel, *stick_channels}) != 3)):
            raise ValueError("throttle/yaw channels must both be 0, or distinct CRSF channels 1..16")
        if (
            self._crc_mode not in CRC_MODES
            or not all(map(math.isfinite, (read_rate_hz, self._baud_probe_s, self._reconnect_interval_s,
                                            self._band_fraction)))
            or not 1 <= self._channel <= 16
            or not self._input_min < self._input_center < self._input_max
            or debounce_frames <= 0
            or read_rate_hz <= 0.0
            or self._baud_probe_s <= 0.0
            or self._reconnect_interval_s <= 0.0
            or not self._baudrates
            or any(value <= 0 for value in self._baudrates)
        ):
            raise ValueError("invalid CRSF receiver, channel, debounce, or rate configuration")

        self._serial: serial.Serial | None = None
        self._parser = CrsfStreamParser(self._crc_mode)
        self._debouncer = BoolDebouncer(debounce_frames)
        self._last_connect_attempt = 0.0
        self._probe_started = 0.0
        self._baudrate_locked = not self._auto_detect_baudrate
        self._publisher = self.create_publisher(Bool, "/trackstand/kickstands_up", 10)
        self._stick_publisher = (
            self.create_publisher(Float32MultiArray, "/locomotive/rc_sticks", 10) if self._publish_sticks else None
        )
        self._timer = self.create_timer(1.0 / read_rate_hz, self._read)
        self._connect()

    def _connect(self) -> None:
        now = time.monotonic()
        self._last_connect_attempt = now
        baudrate = self._baudrates[self._baud_index]
        port = self._port
        if self._auto_detect_port:
            discovered = discover_crsf_port(
                self._port,
                self._port_candidates,
                self._excluded_ports,
                baudrate,
                self._crc_mode,
            )
            if discovered is None:
                self.get_logger().warning(
                    f"Cannot find CRSF receiver at {baudrate} baud; retrying",
                    throttle_duration_sec=2.0,
                )
                if self._auto_detect_baudrate:
                    self._baud_index = (self._baud_index + 1) % len(self._baudrates)
                return
            port = discovered
            self._port = discovered
        elif not os.path.exists(port):
            self.get_logger().warning(f"CRSF receiver {port} does not exist")
            return
        try:
            self._serial = serial.Serial(
                port,
                baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.0,
            )
            self._serial.reset_input_buffer()
            self._parser.reset()
            self._probe_started = now
            self.get_logger().info(
                f"CRSF kickstand switch: {port}, {baudrate} baud, CH{self._channel}, "
                f"high={'up' if self._high_is_up else 'down'}"
            )
        except (OSError, serial.SerialException) as error:
            self._serial = None
            self.get_logger().warning(f"Cannot open CRSF receiver {port}: {error}")

    def _close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except (OSError, serial.SerialException):
                pass
        self._serial = None
        self._parser.reset()
        # Do not reuse a pre-disconnection UP/DOWN decision or partial debounce.
        self._debouncer = BoolDebouncer(self._debouncer.required_frames)

    def _advance_baudrate(self) -> None:
        if self._baudrate_locked or len(self._baudrates) <= 1:
            return
        self._close()
        self._baud_index = (self._baud_index + 1) % len(self._baudrates)
        self._connect()

    def _read(self) -> None:
        now = time.monotonic()
        if self._serial is None:
            if now - self._last_connect_attempt >= self._reconnect_interval_s:
                self._connect()
            return
        try:
            waiting = self._serial.in_waiting
            data = self._serial.read(waiting) if waiting > 0 else b""
        except (OSError, serial.SerialException) as error:
            self.get_logger().warning(f"CRSF receiver disconnected: {error}")
            self._close()
            return
        frames = self._parser.feed(data)
        if frames:
            if self._auto_detect_channel_range:
                self._input_min = 172
                self._input_center = 992
                self._input_max = 1811
                self._auto_detect_channel_range = False
                self.get_logger().info("Auto-selected channel range: CRSF 172/992/1811")
            self._baudrate_locked = True
        elif not self._baudrate_locked and now - self._probe_started >= self._baud_probe_s:
            self._advance_baudrate()
            return
        for channels in frames:
            switch_state = decode_three_position_switch(
                channels,
                self._input_min,
                self._input_center,
                self._input_max,
                self._channel,
                self._high_is_up,
                self._band_fraction,
            )
            confirmed = self._debouncer.observe(switch_state)
            if confirmed is not None:
                self._publisher.publish(Bool(data=confirmed))
            if self._stick_publisher is not None:
                # [throttle, yaw], each in [-1, 1] with +1 at the high-pulse end; one per frame.
                self._stick_publisher.publish(Float32MultiArray(data=[
                    normalize_axis(channels, self._input_min, self._input_center, self._input_max,
                                   self._throttle_channel, self._throttle_reversed),
                    normalize_axis(channels, self._input_min, self._input_center, self._input_max,
                                   self._yaw_channel, self._yaw_reversed),
                ]))

    def destroy_node(self) -> None:
        self._close()
        super().destroy_node()


def main(args=None) -> None:
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = RcKickstand()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
