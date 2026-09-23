"""
CAN bus communication interface.

Provides a clean abstraction over python-can for sending and receiving
CAN messages to/from motors.
"""

import time
import logging
from typing import Optional, List

import can

from .protocol import (
    BAUD_RATES,
    HOST_PREFIX,
    MIT_MODE_PREFIX,
    build_can_id,
)

logger = logging.getLogger(__name__)


class CANInterface:
    """
    Manages the CAN bus connection for motor communication.

    Usage::

        # Create and open
        can_if = CANInterface(channel='can0', baudrate=1000000)
        can_if.open()

        # Send a command to motor at address 0x01
        can_if.send_command(0x01, bytes([0xA1]))

        # Receive response (with timeout)
        response = can_if.receive_response(0x01, timeout=0.5)

        # Close when done
        can_if.close()

    Or use as context manager::

        with CANInterface('can0') as can_if:
            response = can_if.send_and_receive(0x01, bytes([0xAE]))
    """

    def __init__(
        self,
        channel: str = 'can0',
        baudrate: int = 1000000,
        receive_own_messages: bool = False,
        timeout: float = 0.5,
    ):
        """
        Initialize CAN interface.

        Args:
            channel: CAN interface name (e.g., 'can0', 'vcan0')
            baudrate: CAN bus baud rate in bits/s (default 1MHz)
            receive_own_messages: Whether to receive our own sent messages
            timeout: Default receive timeout in seconds
        """
        self.channel = channel
        self.baudrate = baudrate
        self.timeout = timeout
        self._bus: Optional[can.BusABC] = None
        self._receive_own_messages = receive_own_messages

    def open(self) -> None:
        """Open the CAN bus connection."""
        if self._bus is not None:
            logger.warning(f"CAN bus {self.channel} already open")
            return

        try:
            self._bus = can.interface.Bus(
                channel=self.channel,
                bustype='socketcan',
                receive_own_messages=self._receive_own_messages,
            )
            logger.info(f"CAN bus {self.channel} opened at {BAUD_RATES.get(self.baudrate, str(self.baudrate))}")
        except Exception as e:
            raise CANCommunicationError(f"Failed to open CAN bus {self.channel}: {e}") from e

    def close(self) -> None:
        """Close the CAN bus connection."""
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down CAN bus: {e}")
            finally:
                self._bus = None
                logger.info(f"CAN bus {self.channel} closed")

    @property
    def is_open(self) -> bool:
        """Check if the CAN bus is open."""
        return self._bus is not None

    @property
    def bus(self) -> can.BusABC:
        """Get the underlying CAN bus object."""
        if self._bus is None:
            raise CANCommunicationError("CAN bus is not open")
        return self._bus

    def send(self, arbitration_id: int, data: bytes, is_extended_id: bool = False) -> None:
        """
        Send a CAN message.

        Args:
            arbitration_id: 11-bit (or 29-bit extended) CAN ID
            data: Data bytes (0-8 bytes)
            is_extended_id: Whether to use extended frame format
        """
        if self._bus is None:
            raise CANCommunicationError("CAN bus is not open")

        msg = can.Message(
            arbitration_id=arbitration_id,
            data=data,
            is_extended_id=is_extended_id,
        )

        try:
            self._bus.send(msg)
        except can.CanError as e:
            raise CANCommunicationError(f"Failed to send CAN message: {e}") from e

    def receive(self, timeout: Optional[float] = None) -> can.Message:
        """
        Receive a single CAN message.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Received CAN message

        Raises:
            CANTimeoutError: If no message received within timeout
        """
        if self._bus is None:
            raise CANCommunicationError("CAN bus is not open")

        if timeout is None:
            timeout = self.timeout

        msg = self._bus.recv(timeout=timeout)
        if msg is None:
            raise CANTimeoutError(f"No CAN message received within {timeout}s")
        return msg

    def send_command(
        self,
        dev_addr: int,
        data: bytes,
        use_host_prefix: bool = True,
    ) -> None:
        """
        Send a command to a specific motor.

        Args:
            dev_addr: Motor device address (1-254)
            data: Command data bytes (command code + parameters)
            use_host_prefix: If True, send on (0x100 | dev_addr)
        """
        can_id = build_can_id(dev_addr, use_host_prefix=use_host_prefix)
        self.send(can_id, data)

    def send_mit_command(
        self,
        dev_addr: int,
        data: bytes,
        use_host_prefix: bool = True,
    ) -> None:
        """
        Send an MIT mode control command.
        MIT mode requires StdID Bit[10] = 1.

        Args:
            dev_addr: Motor device address
            data: 8-byte MIT control data
            use_host_prefix: If True, send on (0x400 | 0x100 | dev_addr)
        """
        can_id = dev_addr
        if use_host_prefix:
            can_id = HOST_PREFIX | dev_addr
        can_id = MIT_MODE_PREFIX | can_id
        self.send(can_id, data)

    def receive_response(
        self,
        dev_addr: int,
        timeout: Optional[float] = None,
    ) -> can.Message:
        """
        Receive a response from a specific motor.

        The motor responds with its own dev_addr as the CAN ID.

        Args:
            dev_addr: Expected motor device address
            timeout: Maximum wait time

        Returns:
            Received CAN message with arbitration_id == dev_addr
        """
        deadline = time.monotonic() + (timeout or self.timeout)

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            msg = self._bus.recv(timeout=min(remaining, 0.1))
            if msg is None:
                continue

            # Motor responds on its own dev_addr
            if msg.arbitration_id == dev_addr:
                return msg

        raise CANTimeoutError(
            f"No response from motor 0x{dev_addr:02X} "
            f"within {timeout or self.timeout}s"
        )

    def send_and_receive(
        self,
        dev_addr: int,
        data: bytes,
        timeout: Optional[float] = None,
        use_host_prefix: bool = True,
    ) -> can.Message:
        """
        Send a command and wait for the motor's response.

        Args:
            dev_addr: Motor device address
            data: Command data bytes
            timeout: Receive timeout
            use_host_prefix: If True, send on (0x100 | dev_addr)

        Returns:
            Motor's response message
        """
        self.send_command(dev_addr, data, use_host_prefix=use_host_prefix)
        return self.receive_response(dev_addr, timeout=timeout)

    def flush(self) -> None:
        """Flush all pending received messages from the buffer."""
        if self._bus is None:
            return
        try:
            while self._bus.recv(timeout=0.001) is not None:
                pass
        except Exception:
            pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        return f"CANInterface(channel={self.channel}, state={state})"


class CANCommunicationError(Exception):
    """Raised when a CAN communication error occurs."""
    pass


class CANTimeoutError(CANCommunicationError):
    """Raised when a CAN message receive times out."""
    pass


class MotorResponseError(CANCommunicationError):
    """
    Raised when a motor's response indicates an error.

    Attributes:
        dev_addr: Motor device address
        command: Command code that was sent
        response_data: Raw response data
        fault_code: Parsed fault code (if available)
    """
    def __init__(
        self,
        message: str,
        dev_addr: int = 0,
        command: int = 0,
        response_data: bytes = b'',
        fault_code: int = 0,
    ):
        super().__init__(message)
        self.dev_addr = dev_addr
        self.command = command
        self.response_data = response_data
        self.fault_code = fault_code
