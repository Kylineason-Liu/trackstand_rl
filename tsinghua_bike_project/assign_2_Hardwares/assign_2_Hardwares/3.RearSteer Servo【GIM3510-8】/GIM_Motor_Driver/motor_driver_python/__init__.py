"""
Motor CAN Driver Library
========================

A Python library for controlling brushless DC motors via a custom CAN
communication protocol (V3.09b0).

Provides intuitive high-level interfaces for single-motor control,
multi-motor synchronization, and MIT mode operation.

Quick Start::

    from motor_driver import Motor, MotorGroup, CANInterface

    # Open CAN bus
    can_if = CANInterface(channel='can0')
    can_if.open()

    # Create motor instances
    left_motor = Motor(can_if, dev_addr=0x01)
    right_motor = Motor(can_if, dev_addr=0x02)
    motors = MotorGroup(left=left_motor, right=right_motor)

    # Read motor status
    status = left_motor.read_system_status()
    print(f"Temperature: {status.temperature}°C")

    # Control motors
    motors.enable_all()
    motors.set_speed(left=100.0, right=-100.0)

    # Clean up
    motors.disable_all()
    can_if.close()
"""

from .can_interface import (
    CANInterface,
    CANCommunicationError,
    CANTimeoutError,
    MotorResponseError,
)

from .protocol import (
    Command,
    RunMode,
    FaultBit,
    PositionType,
    ParamAccess,
    BrakeAction,
    VersionInfo,
    SystemStatus,
    MotorParams,
    StatusAll,
    AngleInfo,
    MITStatus,
    MITConfig,
    CANTimeoutConfig,
    BROADCAST_ADDR,
    PUBLIC_ADDR,
    DEFAULT_DEV_ADDR,
    COUNTS_PER_REVOLUTION,
    DEGREES_PER_COUNT,
    RADIANS_PER_COUNT,
    BAUD_RATES,
    counts_to_degrees,
    degrees_to_counts,
    counts_to_radians,
    radians_to_counts,
    parse_fault_code,
)

from .motor import Motor
from .motor_group import MotorGroup

__version__ = "1.0.0"
__all__ = [
    # Core classes
    'CANInterface',
    'Motor',
    'MotorGroup',
    # Exceptions
    'CANCommunicationError',
    'CANTimeoutError',
    'MotorResponseError',
    # Enums
    'Command',
    'RunMode',
    'FaultBit',
    'PositionType',
    'ParamAccess',
    'BrakeAction',
    # Data classes
    'VersionInfo',
    'SystemStatus',
    'MotorParams',
    'StatusAll',
    'AngleInfo',
    'MITStatus',
    'MITConfig',
    'CANTimeoutConfig',
    # Constants
    'BROADCAST_ADDR',
    'PUBLIC_ADDR',
    'DEFAULT_DEV_ADDR',
    'COUNTS_PER_REVOLUTION',
    'DEGREES_PER_COUNT',
    'RADIANS_PER_COUNT',
    'BAUD_RATES',
    # Conversion helpers
    'counts_to_degrees',
    'degrees_to_counts',
    'counts_to_radians',
    'radians_to_counts',
    'parse_fault_code',
]
