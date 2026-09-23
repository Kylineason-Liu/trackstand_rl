"""
Multi-motor group for synchronized control.

Provides coordination of multiple motors on the same CAN bus,
useful for robots with two or more motors (e.g., differential drive).

Usage::

    from motor_driver import Motor, MotorGroup, CANInterface

    can_if = CANInterface(channel='can0')
    can_if.open()

    left_motor = Motor(can_if, dev_addr=0x01)
    right_motor = Motor(can_if, dev_addr=0x02)
    motors = MotorGroup(left=left_motor, right=right_motor)

    # Synchronized control
    motors.set_speed(left=100.0, right=-100.0)

    # Broadcast commands to all motors
    motors.broadcast_disable()

    can_if.close()
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable

from .motor import Motor
from .protocol import (
    Command, SystemStatus, VersionInfo, MotorParams, StatusAll, AngleInfo, MITStatus,
    BROADCAST_ADDR,
    pack_1u, pack_2u, pack_4s, pack_4f,
    unpack_1u, unpack_2u, unpack_4s, unpack_4f,
    parse_fault_code,
)

logger = logging.getLogger(__name__)


class MotorGroup:
    """
    Manage and synchronize multiple motors on the same CAN bus.

    Motors can be accessed by name (attribute-style) for direct single-motor
    operations, or controlled together via group methods.

    Args:
        **motors: Named Motor instances (e.g., left=Motor(...), right=Motor(...))
    """

    def __init__(self, **motors: Motor):
        self._motors: Dict[str, Motor] = {}
        for name, motor in motors.items():
            self._add_motor(name, motor)

    def _add_motor(self, name: str, motor: Motor) -> None:
        """Add a motor to the group."""
        if not isinstance(motor, Motor):
            raise TypeError(f"Expected Motor instance, got {type(motor)}")
        self._motors[name] = motor
        # Allow attribute-style access (but not if it shadows existing attrs)
        if not hasattr(self.__class__, name):
            setattr(self, name, motor)

    def add(self, name: str, motor: Motor) -> None:
        """Add a motor to the group after initialization."""
        self._add_motor(name, motor)
        logger.info(f"Motor '{name}' (0x{motor.dev_addr:02X}) added to group")

    @property
    def motors(self) -> Dict[str, Motor]:
        """Dictionary of all motors in the group."""
        return self._motors

    @property
    def motor_names(self) -> List[str]:
        """List of motor names."""
        return list(self._motors.keys())

    def get(self, name: str) -> Motor:
        """Get a motor by name."""
        if name not in self._motors:
            raise KeyError(f"No motor named '{name}'. Available: {self.motor_names}")
        return self._motors[name]

    def __getitem__(self, name: str) -> Motor:
        return self.get(name)

    def __len__(self) -> int:
        return len(self._motors)

    def __iter__(self):
        return iter(self._motors.items())

    def __contains__(self, name: str) -> bool:
        return name in self._motors

    # =========================================================================
    # Per-motor operations (execute on each motor, collect results)
    # =========================================================================

    def _each(self, method_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a method on each motor with motor-specific or common arguments.

        Args:
            method_name: Name of the Motor method to call
            **kwargs: Either per-motor args (e.g., left=100, right=-100)
                      or a single 'all' keyword for common value.

        Returns:
            Dict mapping motor name to result.
        """
        results = {}
        for name, motor in self._motors.items():
            if name in kwargs:
                # Per-motor argument
                arg = kwargs[name]
                if isinstance(arg, (list, tuple)):
                    result = getattr(motor, method_name)(*arg)
                elif isinstance(arg, dict):
                    result = getattr(motor, method_name)(**arg)
                else:
                    result = getattr(motor, method_name)(arg)
            elif 'all' in kwargs:
                # Common argument for all
                arg = kwargs['all']
                if isinstance(arg, (list, tuple)):
                    result = getattr(motor, method_name)(*arg)
                elif isinstance(arg, dict):
                    result = getattr(motor, method_name)(**arg)
                else:
                    result = getattr(motor, method_name)(arg)
            else:
                # No arguments
                result = getattr(motor, method_name)()
            results[name] = result
        return results

    # =========================================================================
    # Group System Commands
    # =========================================================================

    def reboot_all(self) -> None:
        """
        Reboot all motors.

        .. warning::
            Motors will not respond after this command. Wait for them
            to restart before sending further commands.
        """
        for name, motor in self._motors.items():
            try:
                motor.reboot()
            except Exception as e:
                logger.warning(f"Failed to reboot motor '{name}': {e}")

    def read_all_versions(self) -> Dict[str, VersionInfo]:
        """Read version info from all motors."""
        return self._each('read_version')

    def read_all_system_status(self) -> Dict[str, SystemStatus]:
        """Read system status from all motors."""
        return self._each('read_system_status')

    def clear_all_faults(self) -> Dict[str, int]:
        """Clear faults on all motors."""
        return self._each('clear_fault')

    def disable_all(self) -> Dict[str, SystemStatus]:
        """Disable all motors (free state)."""
        return self._each('disable')

    def enable_all(self) -> None:
        """Clear faults on all motors to prepare for control."""
        for name, motor in self._motors.items():
            try:
                motor.enable()
            except Exception as e:
                logger.warning(f"Failed to enable motor '{name}': {e}")

    def emergency_stop_all(self) -> Dict[str, SystemStatus]:
        """Emergency stop all motors immediately."""
        return self._each('emergency_stop')

    # =========================================================================
    # Group Control Commands
    # =========================================================================

    def set_speed(self, **speeds_rpm: float) -> Dict[str, float]:
        """
        Set speed for each motor.

        Usage::

            motors.set_speed(left=100.0, right=-100.0)

        Args:
            **speeds_rpm: Motor name -> target speed in Rpm

        Returns:
            Dict mapping motor name to actual speed.
        """
        return self._each('set_speed', **speeds_rpm)

    def set_q_current(self, **currents_a: float) -> Dict[str, float]:
        """
        Set Q-axis current for each motor.

        Usage::

            motors.set_q_current(left=0.5, right=0.5)

        Args:
            **currents_a: Motor name -> target current in Amperes

        Returns:
            Dict mapping motor name to actual current.
        """
        return self._each('set_q_current', **currents_a)

    def set_absolute_position(self, **positions_counts: int) -> Dict[str, AngleInfo]:
        """
        Set absolute position for each motor in encoder counts.

        Usage::

            motors.set_absolute_position(left=8192, right=8192)
        """
        return self._each('set_absolute_position', **positions_counts)

    def set_absolute_position_degrees(self, **positions_deg: float) -> Dict[str, AngleInfo]:
        """
        Set absolute position for each motor in degrees.
        """
        return self._each('set_absolute_position_degrees', **positions_deg)

    def set_relative_position(self, **deltas_counts: int) -> Dict[str, AngleInfo]:
        """
        Set relative position for each motor in encoder counts.
        """
        return self._each('set_relative_position', **deltas_counts)

    # =========================================================================
    # Group MIT Mode
    # =========================================================================

    def mit_control(
        self,
        positions: Dict[str, float],
        velocities: Optional[Dict[str, float]] = None,
        kps: Optional[Dict[str, float]] = None,
        kds: Optional[Dict[str, float]] = None,
        torques: Optional[Dict[str, float]] = None,
    ) -> Dict[str, MITStatus]:
        """
        Send MIT mode control commands to multiple motors.

        Usage::

            motors.mit_control(
                positions={'left': 1.57, 'right': 1.57},
                kps={'left': 100.0, 'right': 100.0},
                kds={'left': 1.0, 'right': 1.0},
            )

        Args:
            positions: Motor name -> target position (rad), required
            velocities: Motor name -> feed-forward velocity (rad/s)
            kps: Motor name -> position gain (0-500)
            kds: Motor name -> velocity gain (0-5)
            torques: Motor name -> feed-forward torque (Nm)

        Returns:
            Dict mapping motor name to MITStatus.
        """
        results = {}
        velocities = velocities or {}
        kps = kps or {}
        kds = kds or {}
        torques = torques or {}

        for name, motor in self._motors.items():
            if name not in positions:
                continue
            try:
                result = motor.mit_control(
                    position_rad=positions[name],
                    velocity_rad_s=velocities.get(name, 0.0),
                    kp=kps.get(name, 0.0),
                    kd=kds.get(name, 0.0),
                    torque_nm=torques.get(name, 0.0),
                )
                results[name] = result
            except Exception as e:
                logger.warning(f"MIT control failed for motor '{name}': {e}")
                results[name] = None
        return results

    # =========================================================================
    # Synchronized Control (send all commands, then collect all responses)
    # =========================================================================

    def synced_speed(self, **speeds_rpm: float) -> Dict[str, float]:
        """
        Set speed on all motors with minimal delay between commands.

        This sends all commands first, then waits for all responses,
        providing better synchronization than per-motor calls.

        Usage::

            motors.synced_speed(left=100.0, right=-100.0)

        Returns:
            Dict mapping motor name to actual speed.
        """
        # Phase 1: Send all commands
        for name, speed in speeds_rpm.items():
            if name not in self._motors:
                logger.warning(f"Unknown motor '{name}' in synced command")
                continue
            motor = self._motors[name]
            raw = int(speed / 0.01)
            cmd_byte = pack_1u(Command.SPEED_CTRL)
            motor._can.send_command(motor._dev_addr, cmd_byte + pack_4s(raw))

        # Phase 2: Collect all responses
        results = {}
        for name, speed in speeds_rpm.items():
            if name not in self._motors:
                continue
            try:
                motor = self._motors[name]
                msg = motor._can.receive_response(motor._dev_addr, timeout=motor._timeout)
                actual_raw = unpack_4s(msg.data, 1)
                results[name] = actual_raw * 0.01
            except Exception as e:
                logger.warning(f"Failed to get response from '{name}': {e}")
                results[name] = None
        return results

    # =========================================================================
    # Broadcast Commands (using address 0x00 - no response)
    # =========================================================================

    def broadcast_disable(self) -> None:
        """
        Broadcast motor disable to ALL motors on the bus.

        Uses broadcast address 0x00 - all motors execute but none respond.
        """
        from .protocol import Command
        self._send_broadcast(pack_1u(Command.MOTOR_DISABLE))
        logger.info("Broadcast: all motors disabled")

    def broadcast_reboot(self) -> None:
        """
        Broadcast reboot to ALL motors on the bus.

        .. warning::
            All motors will reboot immediately without responding.
        """
        data = pack_1u(Command.REBOOT) + bytes([0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF])
        self._send_broadcast(data)
        logger.info("Broadcast: all motors rebooting")

    def broadcast_clear_fault(self) -> None:
        """Broadcast fault clear to ALL motors on the bus."""
        self._send_broadcast(pack_1u(Command.CLEAR_FAULT))
        logger.info("Broadcast: fault clear sent to all motors")

    def _send_broadcast(self, data: bytes) -> None:
        """Send a CAN message to the broadcast address (0x00)."""
        motor = next(iter(self._motors.values()))
        motor._can.send_command(BROADCAST_ADDR, data, use_host_prefix=False)

    # =========================================================================
    # Group Status
    # =========================================================================

    def get_all_fault_summary(self) -> Dict[str, dict]:
        """
        Get fault status for all motors.

        Returns:
            Dict mapping motor name to parsed fault dictionary.
        """
        results = {}
        for name, motor in self._motors.items():
            try:
                results[name] = motor.get_fault_info()
            except Exception as e:
                logger.warning(f"Failed to get faults for '{name}': {e}")
                results[name] = None
        return results

    def any_fault(self) -> bool:
        """Check if any motor has an active fault."""
        for name, motor in self._motors.items():
            try:
                status = motor.read_system_status()
                if status.fault_code != 0:
                    return True
            except Exception:
                pass
        return False

    def all_healthy(self) -> bool:
        """Check if all motors are fault-free."""
        return not self.any_fault()

    def wait_all_faults_clear(self, timeout: float = 5.0) -> bool:
        """
        Wait for all motor faults to clear.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if all motors are fault-free, False if timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.any_fault():
                return True
            time.sleep(0.1)
        return False

    # =========================================================================
    # Convenience
    # =========================================================================

    def configure_all_pid(self, **pid_values: float) -> Dict[str, dict]:
        """
        Configure PID gains for all motors with the same values.

        Usage::

            motors.configure_all_pid(pos_kp=10.0, pos_ki=0.5, speed_kp=5.0, speed_ki=0.1)
        """
        return self._each('configure_pid', all=pid_values)

    def get_all_full_status(self) -> Dict[str, dict]:
        """Get comprehensive status for all motors."""
        return self._each('get_full_status')

    def __repr__(self) -> str:
        motor_list = ', '.join(
            f"{name}=0x{motor.dev_addr:02X}"
            for name, motor in self._motors.items()
        )
        return f"MotorGroup({motor_list})"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


