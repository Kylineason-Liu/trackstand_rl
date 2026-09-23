"""Modularized self-balancing bicycle control package.

Architecture
------------
  roll_publisher   — YahBoom IMU (/dev/ttyUSB0) → corrected roll state → /roll_info
  steer_publisher  — CAN poll (A3) → steer angle                       → /steer_info
  rear_publisher   — CAN poll (A2) → rear wheel rate                   → /rear_info
  rc_publisher     — SBUS receiver → drive RPM + steer angle           → /rc_info
  main_balance     — reads those topics, runs the balance controller,
                     outputs C1 speed commands to both motors via python-can

Every node runs at 200 Hz.  The sensor nodes own their hardware ports
directly — no external topics are required as input.
"""
