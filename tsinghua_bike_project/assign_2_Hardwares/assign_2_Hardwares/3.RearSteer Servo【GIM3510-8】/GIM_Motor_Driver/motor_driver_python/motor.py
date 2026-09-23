"""
Single motor driver class.

Provides a high-level interface for controlling and configuring a single
motor via the CAN bus protocol.

Usage::

    from motor_driver import Motor, CANInterface

    can_if = CANInterface(channel='can0')
    can_if.open()

    motor = Motor(can_if, dev_addr=0x01)

    # Read motor status
    status = motor.read_system_status()
    print(f"Temperature: {status.temperature}°C")

    # Control motor
    motor.enable()
    motor.set_speed(100.0)  # 100 Rpm

    # Clean up
    motor.disable()
    can_if.close()
"""

import struct
import time
import logging
from typing import Optional, Tuple

from .protocol import (
    Command, RunMode, FaultBit, PositionType, ParamAccess, BrakeAction,
    VersionInfo, SystemStatus, MotorParams, StatusAll, AngleInfo,
    MITStatus, MITConfig, CANTimeoutConfig,
    pack_1u, pack_2u, pack_4u, pack_4s, pack_4f,
    unpack_1u, unpack_1s, unpack_2u, unpack_2s, unpack_4u, unpack_4s, unpack_4f,
    counts_to_degrees, degrees_to_counts,
    counts_to_radians, radians_to_counts,
    parse_fault_code,
    COUNTS_PER_REVOLUTION,
)
from .can_interface import (
    CANInterface,
    CANCommunicationError,
    CANTimeoutError,
    MotorResponseError,
)

logger = logging.getLogger(__name__)


class Motor:
    """
    Driver for a single motor communicating via CAN bus.

    Each motor is identified by its device address (1-254). The driver
    provides methods for:

    - Reading motor state and parameters
    - Configuring control parameters (PID gains, limits, etc.)
    - Sending control commands (torque, speed, position)
    - MIT mode operation
    - Fault management

    Args:
        can_interface: An opened CANInterface instance
        dev_addr: Motor device address (1-254, default 0x01)
        timeout: Default timeout for command-response cycles (seconds)
    """

    def __init__(
        self,
        can_interface: CANInterface,
        dev_addr: int = 0x01,
        timeout: float = 0.5,
    ):
        if dev_addr < 1 or dev_addr > 254:
            raise ValueError(f"Device address must be 1-254, got {dev_addr}")

        self._can = can_interface
        self._dev_addr = dev_addr
        self._timeout = timeout
        self._torque_constant: Optional[float] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def dev_addr(self) -> int:
        """Motor device address."""
        return self._dev_addr

    @property
    def timeout(self) -> float:
        """Default command timeout."""
        return self._timeout

    @timeout.setter
    def timeout(self, value: float) -> None:
        self._timeout = value

    @property
    def torque_constant(self) -> Optional[float]:
        """Motor torque constant (Nm/A), lazily loaded."""
        if self._torque_constant is None:
            try:
                params = self.read_motor_params()
                self._torque_constant = params.torque_constant
            except Exception:
                pass
        return self._torque_constant

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _cmd(self, command: Command, data: bytes = b'') -> bytes:
        """
        Send a command and receive the response data.

        Args:
            command: Command code
            data: Additional data bytes (after command byte)

        Returns:
            Response data bytes (including command byte)

        Raises:
            CANTimeoutError: If motor doesn't respond
            MotorResponseError: If motor reports a fault
        """
        cmd_byte = pack_1u(command)
        msg = self._can.send_and_receive(
            self._dev_addr,
            cmd_byte + data,
            timeout=self._timeout,
        )
        return msg.data

    def _cmd_no_response(self, command: Command, data: bytes = b'') -> None:
        """Send a command that expects no response (e.g., reboot)."""
        cmd_byte = pack_1u(command)
        self._can.send_command(self._dev_addr, cmd_byte + data)

    def _read_param_4u(self, command: Command) -> int:
        """Read a 4-byte unsigned parameter."""
        data = self._cmd(command, pack_1u(ParamAccess.READ))
        return unpack_4u(data, 1)

    def _write_param_4u(self, command: Command, value: int) -> int:
        """Write a 4-byte unsigned parameter, return the confirmed value."""
        data = self._cmd(command, pack_1u(ParamAccess.WRITE_4BYTE) + pack_4u(value))
        return unpack_4u(data, 1)

    def _read_param_4f(self, command: Command) -> float:
        """Read a 4-byte float parameter."""
        data = self._cmd(command, pack_1u(ParamAccess.READ))
        return unpack_4f(data, 1)

    def _write_param_4f(self, command: Command, value: float) -> float:
        """Write a 4-byte float parameter, return the confirmed value."""
        data = self._cmd(command, pack_1u(ParamAccess.WRITE_4BYTE) + pack_4f(value))
        return unpack_4f(data, 1)

    # =========================================================================
    # System Commands
    # =========================================================================

    def reboot(self) -> None:
        """
        Reboot the motor controller.

        .. warning::
            The motor will NOT respond to this command. After rebooting,
            the motor will be in free state (disabled).
        """
        data = pack_1u(Command.REBOOT) + bytes([0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF])
        self._can.send_command(self._dev_addr, data)
        logger.info(f"Motor 0x{self._dev_addr:02X}: reboot command sent")

    def read_version(self) -> VersionInfo:
        """
        Read firmware version information.

        Returns:
            VersionInfo with boot, app, hardware, and CAN protocol versions.
        """
        data = self._cmd(Command.READ_VERSION)
        return VersionInfo(
            boot_version=unpack_2u(data, 1),
            app_version=unpack_2u(data, 3),
            hardware_version=unpack_2u(data, 5),
            can_protocol_version=unpack_1u(data, 7),
        )

    def read_q_current(self) -> float:
        """
        Read real-time Q-axis current.

        Returns:
            Q-axis current in Amperes (A).
            Torque = torque_constant * Q_current.
        """
        data = self._cmd(Command.READ_Q_CURRENT)
        raw = unpack_4s(data, 1)
        return raw * 0.001  # Unit: 0.001A

    def read_speed(self) -> float:
        """
        Read real-time rotation speed.

        Returns:
            Rotation speed in Rpm.
        """
        data = self._cmd(Command.READ_SPEED)
        raw = unpack_4s(data, 1)
        return raw * 0.01  # Unit: 0.01Rpm

    def read_angle(self) -> AngleInfo:
        """
        Read real-time absolute angle (single-turn and multi-turn).

        Returns:
            AngleInfo with single_turn_angle and multi_turn_angle in degrees.
        """
        data = self._cmd(Command.READ_ANGLE)
        single_raw = unpack_2u(data, 1)
        multi_raw = unpack_4s(data, 3)
        return AngleInfo(
            single_turn_angle=counts_to_degrees(single_raw),
            multi_turn_angle=counts_to_degrees(multi_raw),
        )

    def read_status_all(self) -> StatusAll:
        """
        Read temperature, Q-axis current, speed, and angle in one command.

        Returns:
            StatusAll with temperature, q_current, speed, and angle.
        """
        data = self._cmd(Command.READ_STATUS_ALL)
        return StatusAll(
            temperature=unpack_1u(data, 1),
            q_current=unpack_2s(data, 2) * 0.001,
            speed=unpack_2s(data, 4) * 0.01,
            angle=counts_to_degrees(unpack_2u(data, 6)),
        )

    def read_system_status(self) -> SystemStatus:
        """
        Read bus voltage, bus current, temperature, run mode, and fault status.

        Note:
            When the motor detects a fault (other than communication fault),
            it will autonomously report status every 200ms.

        Returns:
            SystemStatus with full system state.
        """
        data = self._cmd(Command.READ_SYSTEM_STATUS)
        return SystemStatus(
            bus_voltage=unpack_2u(data, 1) * 0.01,
            bus_current=unpack_2u(data, 3) * 0.01,
            temperature=unpack_1u(data, 5),
            run_mode=RunMode(unpack_1u(data, 6)),
            fault_code=unpack_1u(data, 7),
        )

    def clear_fault(self) -> int:
        """
        Clear motor fault status.

        Returns:
            Current fault code after clearing (0 = no fault).
        """
        data = self._cmd(Command.CLEAR_FAULT)
        fault_code = unpack_1u(data, 1)
        if fault_code == 0:
            logger.info(f"Motor 0x{self._dev_addr:02X}: faults cleared")
        else:
            faults = parse_fault_code(fault_code)
            active = [name for name, active in faults.items() if active]
            logger.warning(
                f"Motor 0x{self._dev_addr:02X}: faults not cleared - {active}"
            )
        return fault_code

    def get_fault_info(self) -> dict:
        """
        Get parsed fault information.

        Returns:
            Dictionary mapping fault names to boolean values.
        """
        status = self.read_system_status()
        return parse_fault_code(status.fault_code)

    # =========================================================================
    # Parameter Commands
    # =========================================================================

    def read_motor_params(self) -> MotorParams:
        """
        Read motor parameters: pole pairs, torque constant.

        Returns:
            MotorParams with torque_constant (pole_pairs is always 0 in response).
        """
        data = self._cmd(Command.READ_MOTOR_PARAMS)
        pole_pairs = unpack_1u(data, 1) if len(data) > 1 else 0
        torque_constant = unpack_4f(data, 2)
        self._torque_constant = torque_constant
        return MotorParams(
            pole_pairs=pole_pairs,
            torque_constant=torque_constant,
            reduction_ratio=0.0,  # Protocol returns 0 for this field
        )

    def set_origin(self) -> int:
        """
        Set the current position as the origin (zero position).

        The single-turn absolute origin is saved to the driver board
        and persists across power cycles.

        .. note::
            If a second encoder is enabled, this operation takes ~35ms.

        Returns:
            Mechanical angle offset value.
        """
        data = self._cmd(Command.SET_ORIGIN)
        offset = unpack_2u(data, 1)
        logger.info(f"Motor 0x{self._dev_addr:02X}: origin set, offset={offset}")
        return offset

    # ---- Position mode max speed (0xB2) ----

    @property
    def position_max_speed(self) -> int:
        """Position mode maximum speed, unit 0.01Rpm."""
        return self._read_param_4u(Command.POS_MODE_MAX_SPEED)

    @position_max_speed.setter
    def position_max_speed(self, value_rpm_001: int) -> None:
        self._write_param_4u(Command.POS_MODE_MAX_SPEED, value_rpm_001)

    def set_position_max_speed_rpm(self, rpm: float) -> None:
        """Set position mode maximum speed in Rpm."""
        raw = int(rpm / 0.01)
        self._write_param_4u(Command.POS_MODE_MAX_SPEED, raw)

    # ---- Position/Speed mode max Q-current (0xB3) ----

    @property
    def max_q_current(self) -> int:
        """Position/Speed mode maximum Q-axis current, unit 0.001A."""
        return self._read_param_4u(Command.POS_SPEED_MAX_CURRENT)

    @max_q_current.setter
    def max_q_current(self, value_ma: int) -> None:
        self._write_param_4u(Command.POS_SPEED_MAX_CURRENT, value_ma)

    def set_max_q_current_amps(self, amps: float) -> None:
        """Set position/speed mode maximum Q-axis current in Amperes."""
        raw = int(amps / 0.001)
        self._write_param_4u(Command.POS_SPEED_MAX_CURRENT, raw)

    # ---- Q-axis current slope (0xB4) ----

    @property
    def q_current_slope(self) -> int:
        """Q-axis current slope, unit 0.001A/s."""
        return self._read_param_4u(Command.Q_CURRENT_SLOPE)

    @q_current_slope.setter
    def q_current_slope(self, value: int) -> None:
        self._write_param_4u(Command.Q_CURRENT_SLOPE, value)

    # ---- Speed acceleration (0xB5) ----

    @property
    def speed_acceleration(self) -> int:
        """Speed mode acceleration, unit 0.01Rpm/s."""
        return self._read_param_4u(Command.SPEED_ACCELERATION)

    @speed_acceleration.setter
    def speed_acceleration(self, value: int) -> None:
        self._write_param_4u(Command.SPEED_ACCELERATION, value)

    def set_speed_acceleration_rpm_s(self, rpm_per_s: float) -> None:
        """Set speed mode acceleration in Rpm/s."""
        raw = int(rpm_per_s / 0.01)
        self._write_param_4u(Command.SPEED_ACCELERATION, raw)

    # ---- Position loop Kp (0xB6) ----

    @property
    def pos_loop_kp(self) -> float:
        """Position control loop Kp (float)."""
        return self._read_param_4f(Command.POS_LOOP_KP)

    @pos_loop_kp.setter
    def pos_loop_kp(self, value: float) -> None:
        self._write_param_4f(Command.POS_LOOP_KP, value)

    # ---- Position loop Ki (0xB7) ----

    @property
    def pos_loop_ki(self) -> float:
        """Position control loop Ki (float)."""
        return self._read_param_4f(Command.POS_LOOP_KI)

    @pos_loop_ki.setter
    def pos_loop_ki(self, value: float) -> None:
        self._write_param_4f(Command.POS_LOOP_KI, value)

    # ---- Speed loop Kp (0xB8) ----

    @property
    def speed_loop_kp(self) -> float:
        """Speed control loop Kp (float)."""
        return self._read_param_4f(Command.SPEED_LOOP_KP)

    @speed_loop_kp.setter
    def speed_loop_kp(self, value: float) -> None:
        self._write_param_4f(Command.SPEED_LOOP_KP, value)

    # ---- Speed loop Ki (0xB9) ----

    @property
    def speed_loop_ki(self) -> float:
        """Speed control loop Ki (float)."""
        return self._read_param_4f(Command.SPEED_LOOP_KI)

    @speed_loop_ki.setter
    def speed_loop_ki(self, value: float) -> None:
        self._write_param_4f(Command.SPEED_LOOP_KI, value)

    # ---- Device address (0xBA) ----

    def read_device_addr(self) -> int:
        """Read the current device address."""
        data = self._cmd(Command.DEVICE_ADDR, pack_1u(ParamAccess.READ))
        return unpack_1u(data, 1)

    def set_device_addr(self, new_addr: int) -> None:
        """
        Set a new device address.

        .. warning::
            The new address takes effect only after power cycle or reboot.
            This setting is saved to non-volatile memory.
        """
        if new_addr < 1 or new_addr > 254:
            raise ValueError(f"Device address must be 1-254, got {new_addr}")
        self._cmd(Command.DEVICE_ADDR, pack_1u(ParamAccess.WRITE_4BYTE) + pack_2u(new_addr))
        logger.info(
            f"Motor 0x{self._dev_addr:02X}: device address changed to 0x{new_addr:02X}. "
            f"Reboot or power cycle required."
        )

    # =========================================================================
    # Control Commands
    # =========================================================================

    def set_q_current(self, current_a: float) -> float:
        """
        Q-axis current (torque) control mode.

        The motor must be in the correct mode for this to take effect.
        Torque = torque_constant * Q-axis current.

        Args:
            current_a: Target Q-axis current in Amperes.

        Returns:
            Actual Q-axis current reading in Amperes.
        """
        raw = int(current_a / 0.001)  # Unit: 0.001A
        data = self._cmd(Command.Q_CURRENT_CTRL, pack_4s(raw))
        actual_raw = unpack_4s(data, 1)
        return actual_raw * 0.001

    def set_speed(self, speed_rpm: float) -> float:
        """
        Speed control mode.

        Args:
            speed_rpm: Target speed in Rpm.

        Returns:
            Actual speed reading in Rpm.
        """
        raw = int(speed_rpm / 0.01)  # Unit: 0.01Rpm
        data = self._cmd(Command.SPEED_CTRL, pack_4s(raw))
        actual_raw = unpack_4s(data, 1)
        return actual_raw * 0.01

    def set_absolute_position(self, position_counts: int) -> AngleInfo:
        """
        Absolute position control.

        Args:
            position_counts: Target position in encoder counts
                            (16384 counts per revolution).

        Returns:
            AngleInfo with current actual position.
        """
        data = self._cmd(Command.ABS_POS_CTRL, pack_4s(position_counts))
        return AngleInfo(
            single_turn_angle=counts_to_degrees(unpack_2u(data, 1)),
            multi_turn_angle=counts_to_degrees(unpack_4s(data, 3)),
        )

    def set_absolute_position_degrees(self, degrees: float) -> AngleInfo:
        """Absolute position control using degrees."""
        return self.set_absolute_position(degrees_to_counts(degrees))

    def set_relative_position(self, delta_counts: int) -> AngleInfo:
        """
        Relative position control.

        Args:
            delta_counts: Position delta in encoder counts.

        Returns:
            AngleInfo with current actual position.
        """
        data = self._cmd(Command.REL_POS_CTRL, pack_4s(delta_counts))
        return AngleInfo(
            single_turn_angle=counts_to_degrees(unpack_2u(data, 1)),
            multi_turn_angle=counts_to_degrees(unpack_4s(data, 3)),
        )

    def set_relative_position_degrees(self, delta_degrees: float) -> AngleInfo:
        """Relative position control using degrees."""
        return self.set_relative_position(degrees_to_counts(delta_degrees))

    def shortest_path_to_origin(self) -> AngleInfo:
        """
        Move motor to origin via the shortest path.
        Rotation angle will not exceed 180 degrees.

        Returns:
            AngleInfo with final position.
        """
        data = self._cmd(Command.SHORTEST_ORIGIN)
        return AngleInfo(
            single_turn_angle=counts_to_degrees(unpack_2u(data, 1)),
            multi_turn_angle=counts_to_degrees(unpack_4s(data, 3)),
        )

    def disable(self) -> SystemStatus:
        """
        Disable motor output. Motor enters free (uncontrolled) state.

        This is the default state after power-on.

        Returns:
            SystemStatus with current system state.
        """
        data = self._cmd(Command.MOTOR_DISABLE)
        return SystemStatus(
            bus_voltage=unpack_2u(data, 1) * 0.01,
            bus_current=unpack_2u(data, 3) * 0.01,
            temperature=unpack_1u(data, 5),
            run_mode=RunMode(unpack_1u(data, 6)),
            fault_code=unpack_1u(data, 7),
        )

    def enable(self) -> None:
        """
        Enable the motor.

        This is a convenience method that clears faults.
        The motor is enabled by sending any control command (C0-C4, DA, DC, MIT).
        """
        self.clear_fault()
        logger.info(f"Motor 0x{self._dev_addr:02X}: faults cleared, ready for control")

    def emergency_stop(self) -> SystemStatus:
        """
        Emergency stop - immediately disable motor output.

        Returns:
            SystemStatus after disabling.
        """
        return self.disable()

    # ---- CAN Timeout Configuration (0xCD) ----

    def configure_can_timeout(
        self,
        enabled: bool,
        timeout_ms: int = 5000,
        no_software_clear: bool = False,
        brake_disconnect: bool = False,
    ) -> CANTimeoutConfig:
        """
        Configure CAN communication timeout behavior.

        When timeout is enabled and no CAN command is received within
        the specified duration, the motor reports a communication fault
        and disables.

        Args:
            enabled: Enable/disable timeout detection
            timeout_ms: Timeout duration in milliseconds (default 5000)
            no_software_clear: If True, fault cannot be cleared by software
            brake_disconnect: If True, disconnect brake on timeout

        Returns:
            CANTimeoutConfig with the confirmed configuration.
        """
        action = 0
        if no_software_clear:
            action |= (1 << 0)
        if brake_disconnect:
            action |= (1 << 1)

        data = self._cmd(
            Command.CAN_TIMEOUT_CFG,
            pack_1u(ParamAccess.WRITE_4BYTE) +
            pack_1u(0x01 if enabled else 0x00) +
            pack_2u(timeout_ms) +
            pack_1u(action)
        )
        return CANTimeoutConfig(
            enabled=(unpack_1u(data, 1) == 0x01),
            timeout_ms=unpack_2u(data, 2),
            no_software_clear=bool(unpack_1u(data, 4) & (1 << 0)),
            brake_disconnect=bool(unpack_1u(data, 4) & (1 << 1)),
        )

    def read_can_timeout_config(self) -> CANTimeoutConfig:
        """Read the current CAN timeout configuration."""
        data = self._cmd(Command.CAN_TIMEOUT_CFG, pack_1u(ParamAccess.READ))
        return CANTimeoutConfig(
            enabled=(unpack_1u(data, 1) == 0x01),
            timeout_ms=unpack_2u(data, 2),
            no_software_clear=bool(unpack_1u(data, 4) & (1 << 0)),
            brake_disconnect=bool(unpack_1u(data, 4) & (1 << 1)),
        )

    # ---- Brake Control (0xCE) ----

    def brake_connect(self) -> bool:
        """Connect (close) the brake switch."""
        data = self._cmd(Command.BRAKE_CTRL, pack_1u(BrakeAction.CONNECT))
        return unpack_1u(data, 1) == 0x01

    def brake_disconnect(self) -> bool:
        """Disconnect (open) the brake switch."""
        data = self._cmd(Command.BRAKE_CTRL, pack_1u(BrakeAction.DISCONNECT))
        return unpack_1u(data, 1) == 0x00

    def brake_read_status(self) -> bool:
        """Read brake switch status. Returns True if connected."""
        data = self._cmd(Command.BRAKE_CTRL, pack_1u(BrakeAction.READ_STATUS))
        return unpack_1u(data, 1) == 0x01

    # =========================================================================
    # Advanced Control Commands
    # =========================================================================

    # ---- Trapezoid Acceleration (0xD0) ----

    @property
    def trapezoid_acceleration(self) -> int:
        """Trapezoid curve acceleration, unit 0.01Rpm/s. Default: 1000 (10Rpm/s)."""
        return self._read_param_4u(Command.TRAPEZOID_ACC)

    @trapezoid_acceleration.setter
    def trapezoid_acceleration(self, value: int) -> None:
        self._write_param_4u(Command.TRAPEZOID_ACC, value)

    # ---- Trapezoid Deceleration (0xD1) ----

    @property
    def trapezoid_deceleration(self) -> int:
        """Trapezoid curve deceleration, unit 0.01Rpm/s. Default: 1000 (10Rpm/s)."""
        return self._read_param_4u(Command.TRAPEZOID_DEC)

    @trapezoid_deceleration.setter
    def trapezoid_deceleration(self, value: int) -> None:
        self._write_param_4u(Command.TRAPEZOID_DEC, value)

    # ---- Position Filter Bandwidth (0xD5) ----

    def get_position_filter_bandwidth(self) -> int:
        """Get position filter bandwidth in Hz. Default: 50Hz."""
        data = self._cmd(Command.POS_FILTER_BW, pack_1u(ParamAccess.READ))
        return unpack_2u(data, 1)

    def set_position_filter_bandwidth(self, hz: int) -> int:
        """Set position filter bandwidth in Hz."""
        data = self._cmd(Command.POS_FILTER_BW, pack_1u(ParamAccess.WRITE_2BYTE) + pack_2u(hz))
        return unpack_2u(data, 1)

    # ---- Position Filter Inertia (0xD6) ----

    def get_position_filter_inertia(self) -> float:
        """Get position filter inertia. Unit: Nm/(turn/s²). Default: 0.001."""
        return self._read_param_4f(Command.POS_FILTER_INERTIA)

    def set_position_filter_inertia(self, inertia: float) -> float:
        """Set position filter inertia. 0 disables current feedforward."""
        return self._write_param_4f(Command.POS_FILTER_INERTIA, inertia)

    # ---- Position Filter Feedforward Current Limit (0xD7) ----

    def get_position_filter_ff_limit(self) -> int:
        """Get position filter max feedforward current, unit 0.001A. Default: 1A."""
        return self._read_param_4u(Command.POS_FILTER_FF_LIMIT)

    def set_position_filter_ff_limit(self, current_ma: int) -> int:
        """Set position filter max feedforward current, unit 0.001A."""
        return self._write_param_4u(Command.POS_FILTER_FF_LIMIT, current_ma)

    # ---- Trapezoid Position Control (0xDA) ----

    def trapezoid_position(
        self,
        position_counts: int,
        position_type: PositionType = PositionType.ABSOLUTE,
    ) -> AngleInfo:
        """
        Trapezoid curve position control with configurable acc/dec.

        Args:
            position_counts: Target position in encoder counts
            position_type: Absolute or relative

        Returns:
            AngleInfo with current actual position.
        """
        data = self._cmd(
            Command.TRAPEZOID_POS_CTRL,
            pack_1u(position_type) + pack_4s(position_counts)
        )
        return AngleInfo(
            single_turn_angle=counts_to_degrees(unpack_2u(data, 1)),
            multi_turn_angle=counts_to_degrees(unpack_4s(data, 3)),
        )

    def trapezoid_position_degrees(
        self,
        degrees: float,
        absolute: bool = True,
    ) -> AngleInfo:
        """Trapezoid position control using degrees."""
        ptype = PositionType.ABSOLUTE if absolute else PositionType.RELATIVE
        return self.trapezoid_position(degrees_to_counts(degrees), ptype)

    # ---- Position Filter Control (0xDC) ----

    def position_filter_control(
        self,
        position_counts: int,
        position_type: PositionType = PositionType.ABSOLUTE,
    ) -> AngleInfo:
        """
        Position filter control with configurable bandwidth, inertia, and FF limit.

        Args:
            position_counts: Target position in encoder counts
            position_type: Absolute or relative

        Returns:
            AngleInfo with current actual position.
        """
        data = self._cmd(
            Command.POS_FILTER_CTRL,
            pack_1u(position_type) + pack_4s(position_counts)
        )
        return AngleInfo(
            single_turn_angle=counts_to_degrees(unpack_2u(data, 1)),
            multi_turn_angle=counts_to_degrees(unpack_4s(data, 3)),
        )

    def position_filter_control_degrees(
        self,
        degrees: float,
        absolute: bool = True,
    ) -> AngleInfo:
        """Position filter control using degrees."""
        ptype = PositionType.ABSOLUTE if absolute else PositionType.RELATIVE
        return self.position_filter_control(degrees_to_counts(degrees), ptype)

    # =========================================================================
    # MIT Mode Commands
    # =========================================================================

    def mit_configure(
        self,
        pos_max_rad: float,
        vel_max_rad_s: float,
        t_max_nm: float,
    ) -> MITConfig:
        """
        Configure MIT mode limits.

        These settings are saved to non-volatile memory.

        Args:
            pos_max_rad: Maximum position in rad. Unit: 0.1rad. Default: 95.5rad.
            vel_max_rad_s: Maximum velocity in rad/s. Unit: 0.01rad/s. Default: 45.0rad/s.
            t_max_nm: Maximum torque in Nm. Unit: 0.01Nm. Default: 18.0Nm.

        Returns:
            MITConfig with confirmed values.
        """
        raw_pos = int(pos_max_rad / 0.1)
        raw_vel = int(vel_max_rad_s / 0.01)
        raw_t = int(t_max_nm / 0.01)

        data = self._cmd(
            Command.MIT_CONFIG,
            pack_1u(ParamAccess.WRITE_6BYTE) +
            pack_2u(raw_pos) +
            pack_2u(raw_vel) +
            pack_2u(raw_t)
        )
        return MITConfig(
            pos_max=unpack_2u(data, 1) * 0.1,
            vel_max=unpack_2u(data, 3) * 0.01,
            t_max=unpack_2u(data, 5) * 0.01,
        )

    def mit_read_config(self) -> MITConfig:
        """Read MIT mode configuration."""
        data = self._cmd(Command.MIT_CONFIG, pack_1u(ParamAccess.READ))
        return MITConfig(
            pos_max=unpack_2u(data, 1) * 0.1,
            vel_max=unpack_2u(data, 3) * 0.01,
            t_max=unpack_2u(data, 5) * 0.01,
        )

    def mit_read_status(self) -> MITStatus:
        """
        Read MIT mode real-time status.

        Returns:
            MITStatus with position (rad), velocity (rad/s), torque (Nm).
        """
        data = self._cmd(Command.MIT_STATUS)

        # Position: 16 bits [0..65535] -> (-Pos_Max ~ Pos_Max)
        pos_max = self._read_mit_pos_max()
        pos_raw = (data[1] << 8) | data[2]
        position = self._mit_decode(pos_raw, 65535, pos_max)

        # Velocity: 12 bits [0..4095] -> (-Vel_Max ~ Vel_Max)
        vel_max = self._read_mit_vel_max()
        vel_raw = (data[3] << 4) | (data[4] >> 4)
        velocity = self._mit_decode(vel_raw, 4095, vel_max)

        # Torque: 12 bits [0..4095] -> (-T_Max ~ T_Max)
        t_max = self._read_mit_t_max()
        torque_raw = ((data[4] & 0x0F) << 8) | data[5]
        torque = self._mit_decode(torque_raw, 4095, t_max)

        status_byte = data[6]
        return MITStatus(
            position=position,
            velocity=velocity,
            torque=torque,
            is_mit_mode=bool(status_byte & 0x01),
            has_fault=bool(status_byte & 0x02),
        )

    def mit_control(
        self,
        position_rad: float,
        velocity_rad_s: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        torque_nm: float = 0.0,
    ) -> MITStatus:
        """
        Send MIT mode control command.

        This is a low-latency control command that sets the motor to MIT mode
        and sends target position, velocity, torque, and gains in one frame.

        The StdID has Bit[10] set to 1 to identify this as an MIT control frame.

        Args:
            position_rad: Target position in radians
            velocity_rad_s: Feed-forward velocity in rad/s
            kp: Position gain (0-500)
            kd: Velocity gain (0-5)
            torque_nm: Feed-forward torque in Nm

        Returns:
            MITStatus with current state.
        """
        # Get limits for encoding
        pos_max = self._read_mit_pos_max()
        vel_max = self._read_mit_vel_max()
        t_max = self._read_mit_t_max()

        # Encode values
        pos_enc = self._mit_encode(position_rad, pos_max, 65535)
        vel_enc = self._mit_encode(velocity_rad_s, vel_max, 4095)
        t_enc = self._mit_encode(torque_nm, t_max, 4095)
        kp_enc = self._mit_encode(kp, 500, 4095)
        kd_enc = self._mit_encode(kd, 5, 4095)

        # Pack data
        data = bytes([
            (pos_enc >> 8) & 0xFF,   # [0]: Position high 8 bits
            pos_enc & 0xFF,           # [1]: Position low 8 bits
            (vel_enc >> 4) & 0xFF,    # [2]: Velocity high 8 bits
            ((vel_enc & 0x0F) << 4) | ((kp_enc >> 8) & 0x0F),  # [3]: Vel low 4 + Kp high 4
            kp_enc & 0xFF,            # [4]: Kp low 8 bits
            (kd_enc >> 4) & 0xFF,     # [5]: Kd high 8 bits
            ((kd_enc & 0x0F) << 4) | ((t_enc >> 8) & 0x0F),  # [6]: Kd low 4 + Torque high 4
            t_enc & 0xFF,             # [7]: Torque low 8 bits
        ])

        self._can.send_mit_command(self._dev_addr, data)

        # Read response (same format as MIT_STATUS)
        return self.mit_read_status()

    # -------------------------------------------------------------------------
    # MIT helpers
    # -------------------------------------------------------------------------

    def _read_mit_pos_max(self) -> float:
        """Internal: read MIT Pos_Max in rad."""
        try:
            cfg = self.mit_read_config()
            return cfg.pos_max
        except Exception:
            return 95.5  # Default

    def _read_mit_vel_max(self) -> float:
        """Internal: read MIT Vel_Max in rad/s."""
        try:
            cfg = self.mit_read_config()
            return cfg.vel_max
        except Exception:
            return 45.0  # Default

    def _read_mit_t_max(self) -> float:
        """Internal: read MIT T_Max in Nm."""
        try:
            cfg = self.mit_read_config()
            return cfg.t_max
        except Exception:
            return 18.0  # Default

    @staticmethod
    def _mit_encode(value: float, max_val: float, max_enc: int) -> int:
        """Encode a value to MIT 12/16-bit format."""
        ratio = value / max_val
        ratio = max(-1.0, min(1.0, ratio))  # Clamp
        encoded = int((ratio + 1.0) / 2.0 * max_enc)
        return max(0, min(max_enc, encoded))

    @staticmethod
    def _mit_decode(encoded: int, max_enc: int, max_val: float) -> float:
        """Decode a value from MIT 12/16-bit format."""
        ratio = (encoded / max_enc) * 2.0 - 1.0
        return ratio * max_val

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def configure_pid(
        self,
        pos_kp: Optional[float] = None,
        pos_ki: Optional[float] = None,
        speed_kp: Optional[float] = None,
        speed_ki: Optional[float] = None,
    ) -> dict:
        """
        Configure PID gains in one call.

        Args:
            pos_kp: Position loop Kp (None = don't change)
            pos_ki: Position loop Ki (None = don't change)
            speed_kp: Speed loop Kp (None = don't change)
            speed_ki: Speed loop Ki (None = don't change)

        Returns:
            Dictionary with the confirmed PID values.
        """
        result = {}
        if pos_kp is not None:
            self.pos_loop_kp = pos_kp
            result['pos_kp'] = pos_kp
        if pos_ki is not None:
            self.pos_loop_ki = pos_ki
            result['pos_ki'] = pos_ki
        if speed_kp is not None:
            self.speed_loop_kp = speed_kp
            result['speed_kp'] = speed_kp
        if speed_ki is not None:
            self.speed_loop_ki = speed_ki
            result['speed_ki'] = speed_ki
        return result

    def get_full_status(self) -> dict:
        """
        Get comprehensive motor status by combining multiple read commands.

        Returns:
            Dictionary with version, system status, motor params, and real-time data.
        """
        try:
            version = self.read_version()
        except Exception:
            version = None

        try:
            sys_status = self.read_system_status()
        except Exception:
            sys_status = None

        try:
            params = self.read_motor_params()
        except Exception:
            params = None

        try:
            status_all = self.read_status_all()
        except Exception:
            status_all = None

        return {
            'version': version,
            'system_status': sys_status,
            'motor_params': params,
            'status_all': status_all,
        }

    def wait_for_fault_clear(self, timeout: float = 5.0) -> bool:
        """
        Wait for motor faults to clear.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if no faults remain, False if timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.read_system_status()
            if status.fault_code == 0:
                return True
            time.sleep(0.1)

        return False

    def __repr__(self) -> str:
        return f"Motor(dev_addr=0x{self._dev_addr:02X}, timeout={self._timeout}s)"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't auto-disable on exit - user should explicitly control this
        return False
