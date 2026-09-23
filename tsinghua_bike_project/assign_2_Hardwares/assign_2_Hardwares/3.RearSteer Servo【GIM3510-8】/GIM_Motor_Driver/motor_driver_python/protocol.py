"""
CAN communication protocol definitions for the motor driver.

Protocol version: V3.09b0
- Standard CAN frame format (11-bit StdID)
- Little-endian byte order
- Default baud rate: 1MHz
- Default device address: 0x01
"""

import struct
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple

# =============================================================================
# CAN Communication Constants
# =============================================================================

# Special addresses
BROADCAST_ADDR = 0x00       # All slaves execute, no response
PUBLIC_ADDR = 0xFF          # All slaves respond
DEFAULT_DEV_ADDR = 0x01     # Default device address
HOST_PREFIX = 0x100          # Host sends on (0x100 | dev_addr)
MIT_MODE_PREFIX = 0x400      # MIT mode: StdID Bit[10] = 1

# Baud rates (bit/s)
BAUD_RATES = {
    100000: "100Kbps",
    125000: "125Kbps",
    250000: "250Kbps",
    500000: "500Kbps",
    1000000: "1Mbps",
}

# Conversion constants
COUNTS_PER_REVOLUTION = 16384
DEGREES_PER_COUNT = 360.0 / COUNTS_PER_REVOLUTION
RADIANS_PER_COUNT = 2 * 3.141592653589793 / COUNTS_PER_REVOLUTION


# =============================================================================
# Command Codes
# =============================================================================

class Command(IntEnum):
    """CAN command codes for motor control."""
    # System commands
    REBOOT = 0x00              # Reboot slave (no response)
    READ_VERSION = 0xA0        # Read boot/software/hardware/CAN version
    READ_Q_CURRENT = 0xA1      # Read real-time Q-axis current
    READ_SPEED = 0xA2          # Read real-time rotation speed
    READ_ANGLE = 0xA3          # Read single-turn and multi-turn absolute angle
    READ_STATUS_ALL = 0xA4     # Read temperature, Q-current, speed, angle
    READ_SYSTEM_STATUS = 0xAE  # Read voltage, current, temp, run mode, fault
    CLEAR_FAULT = 0xAF         # Clear fault status

    # Parameter commands
    READ_MOTOR_PARAMS = 0xB0   # Read pole pairs, torque constant
    SET_ORIGIN = 0xB1          # Set current position as origin
    POS_MODE_MAX_SPEED = 0xB2  # Read/Set position mode max speed
    POS_SPEED_MAX_CURRENT = 0xB3  # Read/Set position/speed mode max Q-current
    Q_CURRENT_SLOPE = 0xB4     # Read/Set Q-axis current slope
    SPEED_ACCELERATION = 0xB5  # Read/Set speed mode acceleration
    POS_LOOP_KP = 0xB6         # Read/Set position loop Kp
    POS_LOOP_KI = 0xB7         # Read/Set position loop Ki
    SPEED_LOOP_KP = 0xB8       # Read/Set speed loop Kp
    SPEED_LOOP_KI = 0xB9       # Read/Set speed loop Ki
    DEVICE_ADDR = 0xBA         # Read/Set device address (saved on power-off)

    # Control commands
    Q_CURRENT_CTRL = 0xC0      # Q-axis current (torque) control
    SPEED_CTRL = 0xC1          # Speed control
    ABS_POS_CTRL = 0xC2        # Absolute position control
    REL_POS_CTRL = 0xC3        # Relative position control
    SHORTEST_ORIGIN = 0xC4     # Shortest path to origin
    CAN_TIMEOUT_CFG = 0xCD     # CAN timeout configuration
    BRAKE_CTRL = 0xCE          # Brake output control
    MOTOR_DISABLE = 0xCF       # Disable motor (free state)

    # Advanced control commands
    TRAPEZOID_ACC = 0xD0       # Read/Set trapezoid acceleration
    TRAPEZOID_DEC = 0xD1       # Read/Set trapezoid deceleration
    POS_FILTER_BW = 0xD5       # Read/Set position filter bandwidth
    POS_FILTER_INERTIA = 0xD6  # Read/Set position filter inertia
    POS_FILTER_FF_LIMIT = 0xD7  # Read/Set position filter current feedforward limit
    TRAPEZOID_POS_CTRL = 0xDA  # Trapezoid position control
    POS_FILTER_CTRL = 0xDC     # Position filter control

    # MIT mode commands
    MIT_CONFIG = 0xF0          # MIT mode Pos_Max, Vel_Max, T_Max config
    MIT_STATUS = 0xF1          # Read MIT mode status


# =============================================================================
# Enums for various modes and states
# =============================================================================

class RunMode(IntEnum):
    """Motor run mode states."""
    OFF = 0
    VOLTAGE_CONTROL = 1
    Q_CURRENT_CONTROL = 2
    SPEED_CONTROL = 3
    POSITION_CONTROL = 4


class FaultBit(IntEnum):
    """Fault code bit positions."""
    VOLTAGE = 0
    CURRENT = 1
    TEMPERATURE = 2
    ENCODER = 3
    RESERVED = 4
    COMMUNICATION = 5
    HARDWARE = 6
    SOFTWARE = 7


class PositionType(IntEnum):
    """Position control type."""
    ABSOLUTE = 0x00
    RELATIVE = 0x01


class ParamAccess(IntEnum):
    """Parameter read/write access codes."""
    READ = 0x01
    WRITE_2BYTE = 0x03   # 2-byte parameter write
    WRITE_4BYTE = 0x05   # 4-byte parameter write
    WRITE_6BYTE = 0x07   # 6-byte parameter write (MIT config)


class BrakeAction(IntEnum):
    """Brake control actions."""
    DISCONNECT = 0x00
    CONNECT = 0x01
    READ_STATUS = 0xFF


# =============================================================================
# Data Transfer Objects
# =============================================================================

@dataclass
class VersionInfo:
    """Motor firmware version information."""
    boot_version: int       # Boot software version
    app_version: int        # Application software version
    hardware_version: int   # Hardware version
    can_protocol_version: int  # CAN custom protocol version


@dataclass
class SystemStatus:
    """Real-time system status information."""
    bus_voltage: float       # Bus voltage (V)
    bus_current: float       # Bus current (A)
    temperature: int         # Operating temperature (Celsius)
    run_mode: RunMode        # Current run mode
    fault_code: int          # Fault bit flags


@dataclass
class MotorParams:
    """Motor parameters."""
    pole_pairs: int          # Motor pole pairs (always 0 in response)
    torque_constant: float   # Torque constant (N/A)
    reduction_ratio: float   # Reduction ratio (always 0 in response)


@dataclass
class StatusAll:
    """Combined status: temperature, Q-current, speed, angle."""
    temperature: int        # Celsius
    q_current: float        # A
    speed: float            # Rpm
    angle: float            # Degrees (single-turn absolute)


@dataclass
class AngleInfo:
    """Absolute angle information."""
    single_turn_angle: float   # Degrees (0-360)
    multi_turn_angle: float    # Degrees (cumulative)


@dataclass
class MITStatus:
    """MIT mode real-time status."""
    position: float     # rad (mapped from -Pos_Max ~ Pos_Max)
    velocity: float     # rad/s (mapped from -Vel_Max ~ Vel_Max)
    torque: float       # Nm (mapped from -T_Max ~ T_Max)
    is_mit_mode: bool   # True if in MIT mode
    has_fault: bool     # True if system has fault


@dataclass
class MITConfig:
    """MIT mode configuration."""
    pos_max: float      # Position max (rad), unit 0.1rad
    vel_max: float      # Velocity max (rad/s), unit 0.01rad/s
    t_max: float        # Torque max (Nm), unit 0.01Nm


@dataclass
class CANTimeoutConfig:
    """CAN communication timeout configuration."""
    enabled: bool           # Timeout enabled
    timeout_ms: int         # Timeout duration (ms)
    no_software_clear: bool # Fault cannot be cleared by software
    brake_disconnect: bool  # Disconnect brake on timeout


# =============================================================================
# Data Packing/Unpacking Utilities
# =============================================================================

def pack_1u(value: int) -> bytes:
    """Pack 1 unsigned byte."""
    return struct.pack('<B', value & 0xFF)


def pack_1s(value: int) -> bytes:
    """Pack 1 signed byte."""
    return struct.pack('<b', value)


def pack_2u(value: int) -> bytes:
    """Pack 2 unsigned bytes (little-endian)."""
    return struct.pack('<H', value & 0xFFFF)


def pack_2s(value: int) -> bytes:
    """Pack 2 signed bytes (little-endian)."""
    return struct.pack('<h', value)


def pack_4u(value: int) -> bytes:
    """Pack 4 unsigned bytes (little-endian)."""
    return struct.pack('<I', value & 0xFFFFFFFF)


def pack_4s(value: int) -> bytes:
    """Pack 4 signed bytes (little-endian)."""
    return struct.pack('<i', value)


def pack_4f(value: float) -> bytes:
    """Pack single-precision float (little-endian)."""
    return struct.pack('<f', value)


def unpack_1u(data: bytes, offset: int = 0) -> int:
    """Unpack 1 unsigned byte."""
    return data[offset]


def unpack_1s(data: bytes, offset: int = 0) -> int:
    """Unpack 1 signed byte."""
    return struct.unpack('<b', data[offset:offset+1])[0]


def unpack_2u(data: bytes, offset: int = 0) -> int:
    """Unpack 2 unsigned bytes (little-endian)."""
    return struct.unpack('<H', data[offset:offset+2])[0]


def unpack_2s(data: bytes, offset: int = 0) -> int:
    """Unpack 2 signed bytes (little-endian)."""
    return struct.unpack('<h', data[offset:offset+2])[0]


def unpack_4u(data: bytes, offset: int = 0) -> int:
    """Unpack 4 unsigned bytes (little-endian)."""
    return struct.unpack('<I', data[offset:offset+4])[0]


def unpack_4s(data: bytes, offset: int = 0) -> int:
    """Unpack 4 signed bytes (little-endian)."""
    return struct.unpack('<i', data[offset:offset+4])[0]


def unpack_4f(data: bytes, offset: int = 0) -> float:
    """Unpack single-precision float (little-endian)."""
    return struct.unpack('<f', data[offset:offset+4])[0]


def build_can_id(dev_addr: int, use_host_prefix: bool = True) -> int:
    """
    Build the CAN StdID for sending to a motor.

    Args:
        dev_addr: Device address (1-254)
        use_host_prefix: If True, use (0x100 | dev_addr) so slave can
                        distinguish host->slave from slave->host

    Returns:
        11-bit CAN StdID
    """
    if use_host_prefix:
        return HOST_PREFIX | dev_addr
    return dev_addr


def build_mit_can_id(dev_addr: int, use_host_prefix: bool = True) -> int:
    """
    Build the CAN StdID for MIT mode control.
    MIT mode requires Bit[10] = 1.

    Args:
        dev_addr: Device address (1-254)
        use_host_prefix: If True, use (0x400 | 0x100 | dev_addr)

    Returns:
        11-bit CAN StdID with Bit[10] set
    """
    base = dev_addr
    if use_host_prefix:
        base = HOST_PREFIX | dev_addr
    return MIT_MODE_PREFIX | base


def counts_to_degrees(counts: int) -> float:
    """Convert encoder counts to degrees."""
    return counts * DEGREES_PER_COUNT


def degrees_to_counts(degrees: float) -> int:
    """Convert degrees to encoder counts."""
    return int(degrees / DEGREES_PER_COUNT)


def counts_to_radians(counts: int) -> float:
    """Convert encoder counts to radians."""
    return counts * RADIANS_PER_COUNT


def radians_to_counts(radians: float) -> int:
    """Convert radians to encoder counts."""
    return int(radians / RADIANS_PER_COUNT)


def parse_fault_code(fault_code: int) -> dict:
    """
    Parse fault code into individual fault flags.

    Returns:
        Dictionary mapping fault name to bool
    """
    return {
        "voltage_fault": bool(fault_code & (1 << FaultBit.VOLTAGE)),
        "current_fault": bool(fault_code & (1 << FaultBit.CURRENT)),
        "temperature_fault": bool(fault_code & (1 << FaultBit.TEMPERATURE)),
        "encoder_fault": bool(fault_code & (1 << FaultBit.ENCODER)),
        "communication_fault": bool(fault_code & (1 << FaultBit.COMMUNICATION)),
        "hardware_fault": bool(fault_code & (1 << FaultBit.HARDWARE)),
        "software_fault": bool(fault_code & (1 << FaultBit.SOFTWARE)),
    }
