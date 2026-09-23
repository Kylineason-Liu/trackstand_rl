#!/usr/bin/env python3
"""
Quick start: Operate two motors with minimal code.

Before running:
    sudo ip link set can0 up type can bitrate 1000000
"""

import sys
import time
sys.path.insert(0, '/home/ubuntu')

from motor_driver import CANInterface, Motor, MotorGroup

# 1. Open CAN bus
can_if = CANInterface(channel='can0')
can_if.open()

# 2. Create two motors (addresses 0x01 and 0x02)
steer = Motor(can_if, dev_addr=0x02)
rear = Motor(can_if, dev_addr=0x01)
motors = MotorGroup(rear=rear, steer=steer)

# 3. Check status
print(f"Steer motor temp: {steer.read_status_all().temperature}°C")
print(f"Steer motor fault: 0x{steer.read_system_status().fault_code:02X}")
print(f"Rear motor temp: {rear.read_status_all().temperature}°C")
print(f"Rear motor fault: 0x{rear.read_system_status().fault_code:02X}")

# 4. Clear faults and enable
motors.clear_all_faults()

# 5. Control - differential drive
motors.set_speed(steer=10.0,rear=10.0)   # Rotate in place

time.sleep(2.0)  # Run for 2 seconds

# 6. Stop and disable
motors.set_speed(steer=0.0,rear=0.0)
motors.disable_all()
can_if.close()
