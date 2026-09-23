#!/usr/bin/env python3
"""
ROS2 rear steer motor driver for GIM3510-8 (ZE300 series).

A single-file ROS2 node that provides:
  - Topic subscription (~/command) for motor control via MIT mode
  - Topic publication (~/status) for real-time motor feedback

Communicates with the motor over CAN bus using the motor_driver_python library.

Dependencies
------------
    pip install python-can
    # Then install the motor_driver_python library (bundled with this project):
    # cd GIM_Motor_Driver/motor_driver_python && pip install -e .

Usage
-----
    # 1. Bring up the CAN interface (1 Mbps):
    sudo ip link set can0 up type can bitrate 1000000

    # 2. Run the node:
    python3 rear_steer_motor_driver.py

    # 3. Send a command (steer to 15 degrees):
    ros2 topic pub /rear_steer_motor/command std_msgs/msg/Float32MultiArray \
        "data: [15.0, 0.0, 30.0, 1.0, 0.0]"

    # 4. Monitor status:
    ros2 topic echo /rear_steer_motor/status

Topics
------
Subscribed:
  ~/command (std_msgs/Float32MultiArray)
      [position_deg, velocity_dps, kp, kd, torque_nm]
      - position_deg  : Target steering angle in degrees (±95.5°)
      - velocity_dps  : Feed-forward velocity in deg/s (typically 0)
      - kp            : Position gain (0~500, typical steering: 10~50)
      - kd            : Velocity damping gain (0~5, typical: 0.5~2.0)
      - torque_nm     : Feed-forward torque in Nm (typically 0)

Published:
  ~/status (std_msgs/Float32MultiArray)
      [position_deg, velocity_dps, torque_nm, temperature_c,
       bus_voltage_v, bus_current_a, fault_code, run_mode]
"""

import os
import sys
import threading

# ---------------------------------------------------------------------------
# Import the motor_driver_python library
# ---------------------------------------------------------------------------
# Try pip-installed first, then fall back to the relative path.
_MOTOR_DRIVER_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    'GIM_Motor_Driver', 'motor_driver_python'
))

_motor_driver_loaded = False
for _attempt in ('installed', 'local'):
    try:
        from motor_driver_python import CANInterface, Motor
        from motor_driver_python import CANCommunicationError, CANTimeoutError
        _motor_driver_loaded = True
        break
    except ImportError as _e:
        if _attempt == 'installed':
            # Not pip-installed — try from the local GIM_Motor_Driver directory.
            if os.path.isdir(_MOTOR_DRIVER_DIR):
                sys.path.insert(0, os.path.dirname(_MOTOR_DRIVER_DIR))
            else:
                raise ImportError(
                    "motor_driver_python library not found.\n"
                    "Run: cd GIM_Motor_Driver/motor_driver_python && pip install -e .\n"
                    f"(checked: {_MOTOR_DRIVER_DIR})"
                ) from _e
        else:
            # Local import failed too — likely missing 'python-can' dependency.
            _name = getattr(_e, 'name', str(_e))
            raise ImportError(
                f"Cannot load motor_driver_python: {_e}\n\n"
                "Make sure python-can is installed:\n"
                "  pip install python-can"
            ) from _e

# ---------------------------------------------------------------------------
# ROS2 imports
# ---------------------------------------------------------------------------
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
RPM_TO_DEG_S = 360.0 / 60.0  # RPM → degrees/second


# =============================================================================
# ROS2 Node
# =============================================================================

class RearSteerMotorDriver(Node):
    """
    ROS2 node: subscribe to commands → control motor via CAN → publish status.

    One node handles both sides because the command and status paths share
    the same CAN bus connection and Motor object.
    """

    def __init__(self):
        super().__init__('rear_steer_motor_driver')

        # --- Parameters ---
        self.declare_parameter('can_interface', 'can0')
        self.declare_parameter('motor_id', 1)
        self.declare_parameter('status_rate', 50.0)
        self.declare_parameter('command_timeout', 0.1)

        can_iface         = self.get_parameter('can_interface').value
        motor_id          = self.get_parameter('motor_id').value
        status_hz         = self.get_parameter('status_rate').value
        self._cmd_timeout = self.get_parameter('command_timeout').value

        # --- Open CAN bus ---
        self.get_logger().info(f"Opening CAN '{can_iface}' for motor 0x{motor_id:02X} ...")
        self._can = CANInterface(channel=can_iface, timeout=self._cmd_timeout)
        try:
            self._can.open()
        except CANCommunicationError as e:
            self.get_logger().error(
                f"Failed to open CAN bus '{can_iface}': {e}\n"
                "Is the interface up?\n"
                f"  sudo ip link set {can_iface} up type can bitrate 1000000"
            )
            raise

        self._motor = Motor(self._can, dev_addr=motor_id, timeout=self._cmd_timeout)
        self.get_logger().info(f"Motor 0x{motor_id:02X} ready on {can_iface}")

        # --- MIT mode limits (defaults; refreshed from motor config below) ---
        self._mit_pos_max = 95.5   # degrees (±)
        self._mit_vel_max = 45.0   # deg/s (±)
        self._mit_t_max   = 18.0   # Nm (±)
        self._kp_max      = 500.0
        self._kd_max      = 5.0
        self._load_mit_config()

        # --- QoS ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Publisher & subscriber ---
        self._status_pub = self.create_publisher(Float32MultiArray, '~/status', qos)
        self._cmd_sub    = self.create_subscription(Float32MultiArray, '~/command',
                                                     self._command_callback, qos)
        self._timer = self.create_timer(1.0 / status_hz, self._publish_status)

        # --- Thread safety for shared CAN access ---
        self._cmd_lock = threading.Lock()

        self.get_logger().info(
            f"Running: ~/command → motor, ~/status @ {status_hz} Hz"
        )

    # ------------------------------------------------------------------
    # MIT config
    # ------------------------------------------------------------------

    def _load_mit_config(self):
        """
        Read MIT limits from the motor's non-volatile config.

        NOTE: the motor_driver_python library labels these as 'rad' /
        'rad/s', but comparing with the manufacturer's C reference shows
        the values are actually degrees / deg/s.  The default 95.5 is
        the ±95.5° mechanical steering range of the GIM3510-8 servo.
        We treat them as degrees everywhere.
        """
        try:
            cfg = self._motor.mit_read_config()
            self._mit_pos_max = cfg.pos_max
            self._mit_vel_max = cfg.vel_max
            self._mit_t_max   = cfg.t_max
            self.get_logger().info(
                f"MIT limits: pos=±{self._mit_pos_max:.1f}°, "
                f"vel=±{self._mit_vel_max:.1f}°/s, torque=±{self._mit_t_max:.1f} Nm"
            )
        except (CANCommunicationError, CANTimeoutError):
            self.get_logger().warning("Could not read MIT config — using defaults")

    # ------------------------------------------------------------------
    # Command callback
    # ------------------------------------------------------------------

    def _command_callback(self, msg):
        """
        Receive a ~/command and forward it to the motor in MIT mode.

        msg.data: [position_deg, velocity_dps?, kp?, kd?, torque_nm?]
        Missing elements default to 0.
        """
        data = msg.data
        if not data:
            self.get_logger().warning("Empty command received")
            return

        pos = float(data[0])
        vel = float(data[1]) if len(data) > 1 else 0.0
        kp  = float(data[2]) if len(data) > 2 else 0.0
        kd  = float(data[3]) if len(data) > 3 else 0.0
        tor = float(data[4]) if len(data) > 4 else 0.0

        # Clamp to limits
        pos = max(-self._mit_pos_max, min(self._mit_pos_max, pos))
        vel = max(-self._mit_vel_max, min(self._mit_vel_max, vel))
        tor = max(-self._mit_t_max,   min(self._mit_t_max,   tor))
        kp  = max(0.0,                min(self._kp_max,      kp))
        kd  = max(0.0,                min(self._kd_max,      kd))

        with self._cmd_lock:
            try:
                self._motor.mit_control(
                    position_rad=pos,       # actually degrees
                    velocity_rad_s=vel,     # actually deg/s
                    kp=kp, kd=kd, torque_nm=tor,
                )
            except CANTimeoutError:
                self.get_logger().warning("Motor did not respond (timeout)")
            except CANCommunicationError as e:
                self.get_logger().error(f"CAN error: {e}")

    # ------------------------------------------------------------------
    # Status publisher
    # ------------------------------------------------------------------

    def _publish_status(self):
        """
        Timer callback: read motor state and publish on ~/status.

        Uses read_status_all() for a fast 1-round-trip read (temp, current,
        speed, angle).  Attempts read_system_status() for voltage/fault data
        with a short timeout so failures don't block the fast path.
        """
        msg = Float32MultiArray()

        with self._cmd_lock:
            try:
                s = self._motor.read_status_all()

                pos_deg     = s.angle               # protocol returns degrees
                vel_dps     = s.speed * RPM_TO_DEG_S
                temp_c      = float(s.temperature)
                torque_nm   = s.q_current * (self._motor.torque_constant or 0.0)

                # Attempt extended status (voltage, current, fault) with
                # a short timeout — won't stall the timer if it fails.
                bus_v = bus_a = float('nan')
                fault_code = run_mode = 0
                try:
                    self._motor._timeout = 0.03
                    st = self._motor.read_system_status()
                    self._motor._timeout = self._cmd_timeout
                    bus_v = st.bus_voltage
                    bus_a = st.bus_current
                    fault_code = st.fault_code
                    run_mode   = int(st.run_mode)
                except (CANTimeoutError, CANCommunicationError, ValueError):
                    # ValueError catches unknown RunMode values from firmware
                    pass

                msg.data = [
                    pos_deg, vel_dps, torque_nm, temp_c,
                    bus_v, bus_a, float(fault_code), float(run_mode),
                ]
                self._status_pub.publish(msg)

            except CANTimeoutError:
                self.get_logger().warning(
                    "Motor not responding — check power & CAN wiring",
                    throttle_duration_sec=5.0,
                )
            except CANCommunicationError as e:
                self.get_logger().error(str(e), throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        self.get_logger().info("Shutting down …")
        try:
            if self._can.is_open:
                self._can.close()
        except Exception as e:
            self.get_logger().warning(f"Error closing CAN: {e}")
        super().destroy_node()


# =============================================================================
# Entry point
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RearSteerMotorDriver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal: {e}", file=sys.stderr)
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
