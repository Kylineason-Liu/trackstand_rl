#!/usr/bin/env python3
"""
Example usage of the Motor CAN Driver Library.

This script demonstrates:
1. Setting up the CAN interface
2. Creating motor instances for two motors
3. Using MotorGroup for synchronized control
4. Reading motor status and parameters
5. Different control modes (torque, speed, position)
6. MIT mode operation
7. Fault handling
8. Using the context manager pattern

Prerequisites:
    - CAN interface (can0) must be up: sudo ip link set can0 up type can bitrate 1000000
    - Motors connected and powered
"""

import time
import sys
sys.path.insert(0, '/home/ubuntu')

from motor_driver import (
    CANInterface, Motor, MotorGroup,
    CANTimeoutError, MotorResponseError,
    RunMode, PositionType,
    parse_fault_code,
    counts_to_degrees, degrees_to_counts,
)


def setup_can_bus(channel='can0', bitrate=1000000):
    """
    Set up the CAN bus interface.

    NOTE: This uses the 'ip' command, which requires sudo.
    You can also set up the interface manually before running this script:
        sudo ip link set can0 up type can bitrate 1000000
    """
    import subprocess
    try:
        subprocess.run(
            ['sudo', 'ip', 'link', 'set', channel, 'up', 'type', 'can', 'bitrate', str(bitrate)],
            check=True, capture_output=True, text=True
        )
        print(f"CAN interface {channel} configured at {bitrate}bps")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not configure {channel}: {e.stderr}")
        print(f"Please run: sudo ip link set {channel} up type can bitrate {bitrate}")


def print_separator(title: str):
    """Print a formatted section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def demo_basic_info(motor: Motor, name: str):
    """Demonstrate reading basic motor information."""
    print_separator(f"Basic Info - {name}")

    try:
        version = motor.read_version()
        print(f"  Boot version:    {version.boot_version}")
        print(f"  App version:     {version.app_version}")
        print(f"  Hardware version:{version.hardware_version}")
        print(f"  CAN proto ver:   {version.can_protocol_version}")
    except Exception as e:
        print(f"  Failed to read version: {e}")

    try:
        params = motor.read_motor_params()
        print(f"  Torque constant: {params.torque_constant:.4f} N/A")
    except Exception as e:
        print(f"  Failed to read motor params: {e}")


def demo_system_status(motor: Motor, name: str):
    """Demonstrate reading system status."""
    print_separator(f"System Status - {name}")

    try:
        status = motor.read_system_status()
        print(f"  Bus voltage:    {status.bus_voltage:.2f} V")
        print(f"  Bus current:    {status.bus_current:.2f} A")
        print(f"  Temperature:    {status.temperature} °C")
        print(f"  Run mode:       {status.run_mode.name}")
        print(f"  Fault code:     0x{status.fault_code:02X}")

        if status.fault_code != 0:
            faults = parse_fault_code(status.fault_code)
            active = [name for name, active in faults.items() if active]
            print(f"  ⚠ Active faults: {', '.join(active)}")
        else:
            print(f"  ✓ No faults")
    except Exception as e:
        print(f"  Failed to read status: {e}")


def demo_realtime_data(motor: Motor, name: str):
    """Demonstrate reading real-time sensor data."""
    print_separator(f"Real-time Data - {name}")

    try:
        status_all = motor.read_status_all()
        print(f"  Temperature:  {status_all.temperature} °C")
        print(f"  Q-axis curr:  {status_all.q_current:.3f} A")
        print(f"  Speed:        {status_all.speed:.2f} Rpm")
        print(f"  Angle:        {status_all.angle:.2f}°")
    except Exception as e:
        print(f"  Failed to read real-time data: {e}")

    try:
        angle = motor.read_angle()
        print(f"  Single-turn:  {angle.single_turn_angle:.2f}°")
        print(f"  Multi-turn:   {angle.multi_turn_angle:.2f}°")
    except Exception as e:
        print(f"  Failed to read angle: {e}")


def demo_pid_config(motor: Motor, name: str):
    """Demonstrate PID parameter configuration."""
    print_separator(f"PID Configuration - {name}")

    try:
        print("  Current PID values:")
        print(f"    Position Kp: {motor.pos_loop_kp:.3f}")
        print(f"    Position Ki: {motor.pos_loop_ki:.3f}")
        print(f"    Speed Kp:    {motor.speed_loop_kp:.3f}")
        print(f"    Speed Ki:    {motor.speed_loop_ki:.3f}")
    except Exception as e:
        print(f"  Failed to read PID values: {e}")


def demo_speed_control(motors: MotorGroup):
    """Demonstrate speed control mode."""
    print_separator("Speed Control Mode")

    try:
        # Configure speed acceleration for smooth ramping
        for _, motor in motors:
            motor.speed_acceleration = 500  # 5 Rpm/s

        # Set speed for both motors (e.g., differential drive)
        print("  Starting motors: left=50Rpm, right=-50Rpm")
        motors.set_speed(left=50.0, right=-50.0)
        time.sleep(1.0)

        # Read back actual speed
        for name, motor in motors:
            actual_speed = motor.read_speed()
            print(f"  {name} actual speed: {actual_speed:.2f} Rpm")

        # Stop motors
        print("  Stopping motors...")
        motors.set_speed(left=0.0, right=0.0)

    except Exception as e:
        print(f"  Speed control failed: {e}")


def demo_position_control(motor: Motor, name: str):
    """Demonstrate position control (read-only demo)."""
    print_separator(f"Position Control - {name}")

    try:
        # Configure position mode parameters
        print("  Configuring position mode...")
        motor.position_max_speed = 1000  # 10 Rpm
        motor.max_q_current = 3000       # 3 A

        # Read current position
        angle = motor.read_angle()
        print(f"  Current single-turn angle: {angle.single_turn_angle:.2f}°")
        print(f"  Current multi-turn angle:  {angle.multi_turn_angle:.2f}°")

        # Move to absolute position (90 degrees)
        target_counts = degrees_to_counts(90.0)
        print(f"  Moving to 90° ({target_counts} counts)...")
        result = motor.set_absolute_position(target_counts)
        print(f"  After move - single-turn: {result.single_turn_angle:.2f}°")
        time.sleep(0.5)

    except Exception as e:
        print(f"  Position control failed: {e}")


def demo_torque_control(motor: Motor, name: str):
    """Demonstrate Q-axis current (torque) control."""
    print_separator(f"Torque Control - {name}")

    try:
        tc = motor.torque_constant
        if tc:
            print(f"  Torque constant: {tc:.4f} Nm/A")

        # Apply small torque (0.5A)
        print("  Applying 0.5A Q-axis current...")
        actual = motor.set_q_current(0.5)
        print(f"  Actual Q-axis current: {actual:.3f} A")
        if tc:
            print(f"  Estimated torque: {actual * tc:.4f} Nm")

        time.sleep(0.5)

        # Stop torque
        print("  Stopping torque...")
        motor.set_q_current(0.0)

    except Exception as e:
        print(f"  Torque control failed: {e}")


def demo_mit_mode(motor: Motor, name: str):
    """Demonstrate MIT mode configuration and status reading."""
    print_separator(f"MIT Mode - {name}")

    try:
        # Read MIT config
        cfg = motor.mit_read_config()
        print(f"  Pos_Max: {cfg.pos_max:.2f} rad")
        print(f"  Vel_Max: {cfg.vel_max:.2f} rad/s")
        print(f"  T_Max:   {cfg.t_max:.2f} Nm")

        # Read MIT status
        status = motor.mit_read_status()
        print(f"  Position: {status.position:.3f} rad")
        print(f"  Velocity: {status.velocity:.3f} rad/s")
        print(f"  Torque:   {status.torque:.3f} Nm")
        print(f"  MIT mode: {status.is_mit_mode}")
        print(f"  Has fault: {status.has_fault}")

    except Exception as e:
        print(f"  MIT mode read failed: {e}")


def demo_fault_handling(motors: MotorGroup):
    """Demonstrate fault detection and clearing."""
    print_separator("Fault Handling")

    # Check all motors for faults
    summary = motors.get_all_fault_summary()
    for name, faults in summary.items():
        if faults:
            active = [f for f, v in faults.items() if v]
            if active:
                print(f"  ⚠ {name}: {', '.join(active)}")
            else:
                print(f"  ✓ {name}: No faults")

    # Clear faults on all motors
    if motors.any_fault():
        print("  Clearing faults...")
        result = motors.clear_all_faults()
        for name, code in result.items():
            print(f"  {name} fault code after clear: 0x{code:02X}")
    else:
        print("  All motors healthy, no faults to clear")


def demo_synchronized_control(motors: MotorGroup):
    """Demonstrate synchronized multi-motor control."""
    print_separator("Synchronized Control")

    try:
        # Synced speed control - all commands sent first, then responses collected
        print("  Synced speed: left=30Rpm, right=-30Rpm")
        results = motors.synced_speed(left=30.0, right=-30.0)
        for name, speed in results.items():
            print(f"  {name}: {speed:.2f} Rpm")
        time.sleep(0.5)

        # Stop
        motors.synced_speed(left=0.0, right=0.0)

    except Exception as e:
        print(f"  Synchronized control failed: {e}")


def main():
    """Main demo function."""
    # Configuration
    CAN_CHANNEL = 'can0'
    CAN_BITRATE = 1000000
    LEFT_MOTOR_ADDR = 0x01
    RIGHT_MOTOR_ADDR = 0x02

    print("Motor CAN Driver - Example Script")
    print(f"CAN: {CAN_CHANNEL} @ {CAN_BITRATE}bps")

    # Step 1: Set up CAN bus
    setup_can_bus(CAN_CHANNEL, CAN_BITRATE)

    # Step 2: Open CAN interface and create motors
    # Using context manager for clean resource handling
    with CANInterface(channel=CAN_CHANNEL) as can_if:
        # Create individual motors
        left_motor = Motor(can_if, dev_addr=LEFT_MOTOR_ADDR, timeout=0.5)
        right_motor = Motor(can_if, dev_addr=RIGHT_MOTOR_ADDR, timeout=0.5)

        # Group them for synchronized control
        motors = MotorGroup(left=left_motor, right=right_motor)
        print(f"Motors: {motors}")

        # --- Demo: Basic System Info ---
        demo_basic_info(left_motor, "Left Motor")
        demo_basic_info(right_motor, "Right Motor")

        # --- Demo: System Status ---
        demo_system_status(left_motor, "Left Motor")
        demo_system_status(right_motor, "Right Motor")

        # --- Demo: Real-time Data ---
        demo_realtime_data(left_motor, "Left Motor")

        # --- Demo: Fault Handling ---
        demo_fault_handling(motors)

        # --- Demo: PID Configuration ---
        demo_pid_config(left_motor, "Left Motor")

        # --- Demo: Control Modes ---
        # Enable motors (clear faults)
        motors.enable_all()

        try:
            # Torque control demo
            demo_torque_control(left_motor, "Left Motor")

            # Speed control demo
            demo_speed_control(motors)

            # Position control demo
            demo_position_control(left_motor, "Left Motor")

            # MIT mode demo
            demo_mit_mode(left_motor, "Left Motor")

            # Synchronized control
            demo_synchronized_control(motors)

        finally:
            # Always disable motors before exit
            print_separator("Cleanup")
            print("  Disabling all motors...")
            motors.emergency_stop_all()
            print("  Done.")

    print("\n✓ Demo completed.")


if __name__ == '__main__':
    main()
