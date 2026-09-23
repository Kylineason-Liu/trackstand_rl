"""Modularized self-balancing bicycle control package.

Architecture
------------
  roll_publisher   — YahBoom IMU (/dev/ttyUSB0) → corrected roll state → /roll_info
  steer_publisher  — CAN poll (A4) → steer angle + rate              → /steer_info
  rear_publisher   — CAN poll (A2) → rear wheel rate                  → /rear_info
  main_balance     — reads three topics, runs balance controller,
                     outputs C1 speed commands via python-can

The sensor nodes own their hardware ports directly — no external topics
are required as input.
"""
