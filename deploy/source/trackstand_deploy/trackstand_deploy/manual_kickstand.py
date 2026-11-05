"""Terminal replacement for RC kickstand authority; same Bool topic and policy gates."""

from __future__ import annotations

import os
import select
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


PUBLISH_RATE_HZ = 50.0
TOPIC = "/trackstand/kickstands_up"


class ManualKickstand(Node):
    def __init__(self, input_stream=None) -> None:
        super().__init__("trackstand_manual_kickstand")
        self._input_fd = (sys.stdin if input_stream is None else input_stream).fileno()
        self._buffer = b""
        self._up = False
        self.finished = False
        self._publisher = self.create_publisher(Bool, TOPIC, 10)
        self._timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._tick)
        self._publish()
        self.get_logger().info(
            "Manual kickstand control: starts DOWN. Type up or down, then Enter; quit exits with DOWN. "
            "Run instead of rc_kickstand. Policy readiness/fault checks still apply."
        )

    def _publish(self) -> None:
        self._publisher.publish(Bool(data=self._up))

    def stop(self, reason="Manual controller stopping") -> None:
        self._up, self.finished = False, True
        if self.context.ok():
            self._publish()
        self.get_logger().info(f"{reason}; requesting DOWN")

    def _command(self, line: bytes) -> None:
        command = line.decode("utf-8", errors="replace").strip().lower()
        if command in ("up", "down"):
            self._up = command == "up"
            self._publish()  # Preserve every complete down/up transition in arrival order.
            self.get_logger().info(f"Kickstands requested {command.upper()}")
        elif command in ("quit", "exit"):
            self.stop()
        elif command:
            self.get_logger().warning("Unknown command; type up, down or quit, then Enter")

    def _tick(self) -> None:
        if self.finished:
            return
        try:
            readable, _, _ = select.select([self._input_fd], [], [], 0)
            if readable:
                # readline()/input() can block on a partial pipe line and starve
                # the authority heartbeat. Read only currently available bytes.
                chunk = os.read(self._input_fd, 4096)
                if not chunk:
                    self.stop("Terminal input closed")
                    return
                lines = (self._buffer + chunk).split(b"\n")
                self._buffer = lines.pop()
                for line in lines:
                    self._command(line)
                    if self.finished:
                        return
                if len(self._buffer) > 4096:
                    self.stop("Terminal input line too long")
                    return
        except (BlockingIOError, InterruptedError):
            pass  # Temporary lack of input is not loss of authority.
        except (OSError, ValueError) as error:
            self.stop(f"Terminal input unavailable: {error}")
            return
        self._publish()  # Held switch state, even while waiting for the next line.


def main(args=None) -> None:
    from rclpy.executors import ExternalShutdownException
    from rclpy.signals import SignalHandlerOptions

    node = None
    stopping = False
    previous_handlers = {}

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    try:
        # Keep ROS alive long enough to publish DOWN on Ctrl-C/TERM/HUP.
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[number] = signal.signal(number, request_stop)
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        node = ManualKickstand()
        while rclpy.ok() and not stopping and not node.finished:
            rclpy.spin_once(node, timeout_sec=1.0 / PUBLISH_RATE_HZ)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                try:
                    node.stop()
                    # Best-effort reliable delivery; never claim physical servo acknowledgement.
                    deadline = time.monotonic() + .1
                    while node.context.ok() and time.monotonic() < deadline:
                        node._publish()
                        rclpy.spin_once(node, timeout_sec=1.0 / PUBLISH_RATE_HZ)
                except ExternalShutdownException:
                    pass
                finally:
                    node.destroy_node()
        finally:
            try:
                rclpy.try_shutdown()
            finally:
                for number, handler in previous_handlers.items():
                    signal.signal(number, handler)


if __name__ == "__main__":
    main()
