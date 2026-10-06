#!/usr/bin/env python3
# Bench-test steering targets; no package build is needed.
# A literal `ros2 topic pub --rate 50 ... '{data: [...]}'` repeats one fixed
# value. Keep one rclpy publisher alive instead of restarting the CLI per sample.
import argparse
import math
import os
import signal
import sys
import time

RATE_HZ = 50.0
TOPIC = "/trackstand/motor_command"
MESSAGE_TYPE = "std_msgs/msg/Float64MultiArray"


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be a finite number")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="steering_sine.py",
        description="Publish [1.0, amplitude*sin(2*pi*t/period), 0.0] at nominal 50 Hz.",
        epilog=(
            "Source ROS 2 first. Stop policy_controller and all other motor-command publishers. "
            "Independently restrain the bike, lift the rear wheel, verify steering clearance, "
            "and keep physical power isolation available. This enables BOTH motors; RC-down "
            "does not stop these manual commands. The existing motor_io controller is unchanged. "
            "Preview first, then use motor_io with dry_run:=true before a powered test. "
            "The default +/-1 rad (+/-57.3 deg) has NOT been established safe by prior tests."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true", help="Print samples only; do not initialize ROS or publish.")
    mode.add_argument("--execute", action="store_true", help="Acknowledge the bench-test hazards and publish commands.")
    parser.add_argument(
        "--period", type=finite_float, default=10.0, metavar="SECONDS", help="Sine period (default: 10 s).",
    )
    parser.add_argument(
        "--amplitude", type=finite_float, default=1.0, metavar="RAD",
        help="Peak steering target in radians, 0 < amplitude <= 1 (default: 1). Use smaller bench targets first.",
    )
    parser.add_argument(
        "--duration", type=finite_float, default=0.0, metavar="SECONDS",
        help="Stop after this duration; 0 means until Ctrl-C for execution, or one period for preview (default: 0).",
    )
    args = parser.parse_args(argv)
    if args.period <= 2.0 / RATE_HZ:
        parser.error("--period must exceed 0.04 s (sine frequency must be below the 25 Hz sampling Nyquist limit)")
    if not 0.0 < args.amplitude <= 1.0:
        parser.error("--amplitude must be greater than 0 and no greater than 1 rad")
    if args.duration < 0.0:
        parser.error("--duration must be nonnegative")
    return args


def steering_target(elapsed_s, amplitude_rad, period_s):
    return amplitude_rad * math.sin(2.0 * math.pi * (elapsed_s % period_s) / period_s)


def preview(args):
    duration_s = args.duration or args.period
    print("PREVIEW ONLY: no ROS publisher. Samples below are spaced at 0.02 s; output is not paced.")
    for index in range(math.ceil(duration_s * RATE_HZ)):
        elapsed_s = index / RATE_HZ
        target_rad = steering_target(elapsed_s, args.amplitude, args.period)
        print(f"t={elapsed_s:.3f}s {{data: [1.0, {target_rad:.9f}, 0.0]}}")


def execute(args):
    try:
        import rclpy
        from rclpy.clock import Clock, ClockType
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from std_msgs.msg import Float64MultiArray
    except ImportError as error:
        raise RuntimeError("ROS Python modules are unavailable. Source your ROS 2 installation first.") from error

    print(
        f"BENCH COMMANDS: +/-{args.amplitude:g} rad ({math.degrees(args.amplitude):.1f} deg), "
        f"period={args.period:g} s, nominal {RATE_HZ:g} Hz. BOTH motors will be enabled.\n"
        "RC-down is not an interlock here. Stop immediately if vibration starts; have physical power isolation ready.",
        flush=True,
    )
    stop_requested = False
    started = False
    timer = None
    node = None
    sent_count = 0

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        # Keep the context alive on Ctrl-C/SIGTERM so the same publisher can send disable.
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node(f"trackstand_steering_sine_{os.getpid()}", namespace="/")
        publisher = node.create_publisher(
            Float64MultiArray, TOPIC,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE),
        )

        def check_command_route():
            endpoints = node.get_publishers_info_by_topic(TOPIC)
            if len(endpoints) != 1 or endpoints[0].node_name != node.get_name():
                raise RuntimeError(
                    "Another motor-command publisher exists (or discovery is incomplete). Stop it first."
                )
            motor_subscriptions = [
                endpoint for endpoint in node.get_subscriptions_info_by_topic(TOPIC)
                if endpoint.node_name == "trackstand_motor_io" and endpoint.node_namespace == "/"
                and endpoint.topic_type == MESSAGE_TYPE
            ]
            if len(motor_subscriptions) != 1 or publisher.get_subscription_count() < 1:
                raise RuntimeError("Expected exactly one /trackstand_motor_io subscribing to the motor-command topic.")

        print("Checking the ROS graph for 2 seconds; no commands are sent yet.", flush=True)
        discovery_deadline = time.monotonic() + 2.0
        while not stop_requested and time.monotonic() < discovery_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if stop_requested:
            return
        check_command_route()
        start_s = time.monotonic()
        message = Float64MultiArray()

        def publish_target():
            nonlocal stop_requested, sent_count
            if stop_requested:
                return
            elapsed_s = time.monotonic() - start_s
            if args.duration and elapsed_s >= args.duration:
                stop_requested = True
                return
            message.data = [1.0, steering_target(elapsed_s, args.amplitude, args.period), 0.0]
            publisher.publish(message)
            sent_count += 1

        # Start at zero. Phase thereafter follows elapsed monotonic time, not a
        # sample counter; missed deadlines do not cause a burst of old targets.
        started = True
        message.data = [1.0, 0.0, 0.0]
        publisher.publish(message)
        sent_count += 1
        timer = node.create_timer(1.0 / RATE_HZ, publish_target, clock=Clock(clock_type=ClockType.STEADY_TIME))
        next_graph_check_s = start_s + 1.0
        while not stop_requested and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1.0 / RATE_HZ)
            if time.monotonic() >= next_graph_check_s:
                check_command_route()
                next_graph_check_s = time.monotonic() + 1.0
    finally:
        stop_requested = True
        if timer is not None:
            timer.cancel()
        if started and rclpy.ok():
            # Best effort, not proof of physical disable. The existing 80 ms
            # motor_io command watchdog is a fallback if this process is killed.
            try:
                disable = Float64MultiArray(data=[0.0, 0.0, 0.0])
                for _ in range(10):
                    publisher.publish(disable)
                    time.sleep(1.0 / RATE_HZ)
                print(f"Stopped after {sent_count} enabled messages; sent [0.0, 0.0, 0.0] ten times.", flush=True)
            except Exception as error:
                print(f"WARNING: disable publication failed: {error}. Isolate power if necessary.", file=sys.stderr)
        elif started:
            print("WARNING: ROS context is unavailable; explicit disable could not be published.", file=sys.stderr)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
        for number, previous in previous_handlers.items():
            signal.signal(number, previous)


def main():
    args = parse_args()
    try:
        if args.preview:
            preview(args)
        else:
            execute(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
