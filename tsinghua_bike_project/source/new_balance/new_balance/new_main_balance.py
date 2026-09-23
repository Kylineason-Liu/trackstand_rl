#!/usr/bin/env python3
# todo: add compensator to automatically adjust reference roll angle just like manually manipulated by rc controller
"""
Modularized self-balancing bicycle control node — single 50 Hz loop.

Differs from main_balance.py in two ways: the control tick runs at 50 Hz
rather than 200 Hz, and gain rescheduling is decoupled from the control rate.
See "Two rates, not one" below.  Everything else — topics, parameters, modes,
safety — is unchanged.

Reads pre-processed sensor state from topic publishers and outputs CAN
speed commands (C1) to both the steering motor and the drive motor via
python-can.

The state-feedback law produces a steering *angular velocity*, which is
exactly what a C1 speed command expresses, so the controller output goes
straight onto the bus and the motor's built-in speed loop closes it.  There
is no user-space inner loop and no torque (C0) command: everything — state
feedback, safety, CAN output — happens in one ROS timer callback at
``control_hz``.

A ``dry_run`` parameter allows running the full control pipeline without
sending any CAN commands — useful for latency measurement and offline
testing.

Architecture (what runs where)
------------------------------
Hardware                Node (ROS 2)           Topic            Consumer
────────────────────────────────────────────────────────────────────────────
YahBoom IMU (/dev/ttyUSB0) → roll_publisher  → /roll_info  ─┐
GIM steer motor (CAN 101/001) ↔ steer_publisher → /steer_info ─┼─ balance_node
GIM drive motor (CAN 102/002) ↔ rear_publisher  → /rear_info  ─┘
SBUS RC receiver (/dev/ttyUSB1) → rc_publisher → /rc_info  ───┘  ([drive_rpm, steer_rad])
                                              balance_node → CAN 101 (C1 steer speed)
                                              balance_node → CAN 102 (C1 drive speed)

The publisher nodes run at 200 Hz and so does this node's control tick, so
every tick reads a fresh sample.  Only the gain reschedule is decimated, to
``gain_update_hz``; see the gain-scheduling section below.

Subscribes
----------
  /roll_info  (Float64MultiArray)  — [roll, pitch, yaw, roll_rate, p, q, r,
                                      publish_ts, droll_dt, roll_raw]
                                     roll (index 0) is low-pass filtered by
                                     roll_publisher (roll_lp_alpha); this node
                                     feeds it straight into the state vector.
  /steer_info (Float64MultiArray)  — [steer_angle (rad), publish_ts]
  /rear_info  (Float64MultiArray)  — [rear_rate (rad/s), publish_ts]
  /rc_info    (Float64MultiArray)  — [drive_rpm, steer_rad, servo_up, publish_ts]
                                      (only when vctrl="rc").  data[1] keeps
                                      rc_publisher's "steer_rad" name but is
                                      consumed here as the desired ROLL angle;
                                      see the model's track_roll setting.
  /bike_balance/enable (Bool)      — arm / disarm the controller
  /new_balance/steer_rate_ref (Float64MultiArray) — steer rate ref (steer_test only)

  The trailing ``publish_ts`` in each topic is a ``time.time()`` float
  appended by the publisher.  It is optional — callbacks use it only when
  the message array is long enough; otherwise the timing fields in status
  show -1.0.

Publishes
---------
  /new_balance/status (String)            JSON status at 2 Hz
  /new_balance/debug (Float64MultiArray)  Debug floats (19 fields) at control_hz
  /steer_cmd (Float64MultiArray)          [steer_rate_cmd_rad_s, steer_c1_raw]
                                          at control_hz

  Status JSON includes a ``timing`` object with per-topic pipeline latencies
  (see the Timing section below).

  /new_balance/debug layout:
    [0] roll (rad)          [1] roll_rate (rad/s)   [2] steer_angle (rad)
    [3] forward_speed (m/s) [4] steer_rate_ref (rad/s)
    [5] steer_c1 (raw, rpm×100 as sent)             [6] sat_outer (0/1)
    [7] enabled (0/1)       [8..10] K[0..2]
    [11] roll_age_ms        [12] steer_age_ms       [13] compute_ms
    [14] rc_roll_ref (rad)  — /rc_info.data[1] exactly as received
    [15] roll_ref (rad)     — what solve_equilibrium was actually given;
                              0 when disarmed, inside balance_start_delay,
                              in steer_test, or when vctrl != "rc".  Compare
                              against [14] to see whether the stick is
                              reaching the control law at all.
    [16..18] x_eq[0..2]     — the equilibrium state the law steers toward:
                              [-(a4/a2)·roll_ref, roll_ref, 0].  [17] should
                              track [15] exactly; [16] is the steering angle
                              that sustains the lean and is the one to watch
                              against steer_angle_limit_deg.

  Fields 0-13 are frozen: bags and bag_tune.py index them positionally, so
  new fields are appended, never inserted.

Parameters (also see config/params.yaml for full list)
------------------------------------------------------
  mode (str)           : sign_check | steer_test | full   (default: full)
  method (str)         : pole | pole_distinct | lqr       (default: lqr)
  vctrl (str)          : "fixed" | "rc"                   (default: rc)
  control_hz (double)     : control-law / safety / CAN rate  (default: 200.0)
  gain_update_hz (double) : gain rescheduling rate           (default: 50.0)
  can_interface (str)  : SocketCAN interface              (default: can0)
  steer_can_id (str)   : Host → steer motor CAN ID        (default: 101)
  drive_can_id (str)   : Host → drive motor CAN ID        (default: 102)

  poles_ctr (double[]) : method=pole_distinct only.  3 values [p1, p2, p3]
                         for real poles, or 6 values [re1, im1, re2, im2,
                         re3, im3] for a complex conjugate pair.  Solve them
                         from a gain with k_to_poles.py.  See _decode_poles.

  Dry-run / diagnostics:
    dry_run (bool)        — skip all CAN sends; compute + publish only    (default: false)


╔══════════════════════════════════════════════════════════════════════════╗
║                        CONTROL ARCHITECTURE                              ║
╚══════════════════════════════════════════════════════════════════════════╝

  State feedback (LQR / pole-placement), one tick at control_hz:
      x = [steer_angle, roll_angle, roll_rate]^T
      u = steer_rate_ref  ←  K · (x_eq − x) + u_eq
      → clipped to ±max_steer_velocity
      → CAN C1 to the steer motor:  <0xC1> <4-byte LE int32  rpm × 100>

  The drive motor gets its own C1 from the speed profile (fixed ramp) or
  from the RC throttle stick.

  Two rates, not one
  ------------------
  A, Bu are re-scheduled from the measured forward speed and K re-solved
  from them once every N control ticks, not every tick.  The control law,
  every safety check and both CAN sends run at the full control rate against
  fresh sensor data.

  N = round(control_hz / gain_update_hz), at least 1.  It is a tick count,
  not a deadline, so the ratio is exact by construction: at 200/50 exactly
  4 control ticks pass per reschedule, forever, and the first tick after
  start-up always reschedules.  The achieved rate is control_hz / N and is
  reported as ``gain_update_hz`` in the status JSON alongside
  ``gain_every_n_ticks``.

  Why: re-solving the gain is by far the most expensive thing in the tick —
  measured at ~810 us for lqr against ~350 us for pole, and lqr's jitter is
  worse still.  rclpy.spin runs a *single-threaded* executor, so the timer
  shares one thread with roughly 800 incoming messages a second, and a timer
  that eats its budget starves the subscription callbacks.  Observed on the
  bike at control_hz = 200 (a 5 ms budget): with method=lqr the steer angle
  stopped updating, so a hand-centered front wheel never re-entered the fault
  recovery band and the motor could not be re-enabled; with method=pole the
  same procedure worked.  The RC throttle going dead had the same cause.

  Nothing is lost by throttling it: forward speed cannot change meaningfully
  between reschedules, so a slightly older gain describes the same operating
  point.  ``gain_sched_v`` in the status JSON reports the speed the live K
  actually belongs to.

  The defaults are 200.0 / 50.0, i.e. N = 4.  That is the configuration the
  starvation above rules out for an undecimated tick: 200 Hz control against
  50 Hz rescheduling drops the average tick cost ~68% versus rescheduling
  every tick, which is what puts a 5 ms budget back within reach.  Setting
  gain_update_hz = control_hz restores the old every-tick behaviour and the
  failure that came with it.


╔══════════════════════════════════════════════════════════════════════════╗
║                        TIMING INSTRUMENTATION                            ║
║              Software pipeline latency measurement                       ║
╚══════════════════════════════════════════════════════════════════════════╝

  Each publisher appends a ``time.time()`` float to its message.  BalanceNode
  records a timestamp in each subscriber callback and computes the difference
  at the start of every control tick.  Results appear in the status JSON
  ``timing`` object:

    Key                    Unit  Meaning
    ────────────────────── ────  ─────────────────────────────────────────
    roll_pub_to_cb_ms       ms   /roll_info  publish → balance_node callback
    roll_cb_to_tick_ms      ms   balance_node callback → tick start
    roll_age_ms             ms   /roll_info  publish → tick start (total)
    steer_pub_to_cb_ms      ms   /steer_info publish → callback
    steer_cb_to_tick_ms     ms   callback → tick start
    steer_age_ms            ms   /steer_info publish → tick start (total)
    rear_pub_to_cb_ms       ms   /rear_info  publish → callback
    rear_age_ms             ms   /rear_info  publish → tick start
    rc_pub_to_cb_ms         ms   /rc_info    publish → callback (vctrl=rc only)
    rc_age_ms               ms   /rc_info    publish → tick start
    compute_us              µs   control computation

  Values of −1.0 indicate missing timestamps (e.g. publisher running old code).


╔══════════════════════════════════════════════════════════════════════════╗
║                           MODE 1 — sign_check                           ║
║                   Verify IMU axis signs by hand                         ║
╚══════════════════════════════════════════════════════════════════════════╝

  Purpose: Verify that the corrected roll/pitch/yaw angles and body rates
  have the correct signs before engaging any motor.  No CAN commands are
  sent — this is a pure read-only diagnostic mode.

  Prerequisites: CAN bus must be up (``sudo ip link set can0 up type can
  bitrate 1000000``), IMU must be connected to /dev/ttyUSB0.

  Step-by-step
  ────────────
  1.  Terminal 1: start the IMU publisher
          ros2 run new_balance roll_publisher

      Wait for "YbImu Serial Opened!" and "RollInfoPublisher …" messages.

  2.  Terminal 2: start the balance node in sign_check mode
          ros2 run new_balance balance_node --ros-args -p mode:=sign_check

  3.  Manually tilt / rotate the bike and check the terminal-2 log output:

      Tilt bike LEFT  → "roll" sign should be consistent (e.g. negative).
      Tilt bike RIGHT → "roll" should have the opposite sign.
      Tilt nose DOWN  → "pitch" sign should be consistent.
      Rotate bike CCW → "r" (body yaw rate) should be positive.

      If any sign is wrong, adjust sensor_to_bike_rpy_deg in
      config/params.yaml (or on the CLI via --ros-args), Ctrl-C and restart.

  4.  Monitor /roll_info directly (optional):
          ros2 topic echo /roll_info

  NOTE: steer_publisher and rear_publisher are NOT needed for this mode.


╔══════════════════════════════════════════════════════════════════════════╗
║                          MODE 2 — steer_test                            ║
║          Bench test: command the steer motor speed by topic             ║
╚══════════════════════════════════════════════════════════════════════════╝

  Purpose: Test the steering motor by publishing a target angular velocity
  (rad/s) that is sent straight to the motor as a C1 speed command, with no
  state feedback in the way.  Enforces the steer angle limit (±30° by
  default) and applies an emergency roll stop (>30°).  The drive motor is
  kept stopped (C1=0).

  Prerequisites
  ─────────────
      CAN bus up:   sudo ip link set can0 up type can bitrate 1000000
      python-can:   pip install python-can
      YbImuLib:     installed from assign_2_Hardwares/1.IMU/Driver/YbImuLib/

  Step-by-step
  ────────────
  1.  Terminal 1: IMU → /roll_info
          ros2 run new_balance roll_publisher

  2.  Terminal 2: steer CAN → /steer_info
          ros2 run new_balance steer_publisher

      Wait for "Steer zero locked: …" message.  This confirms the motor is
      responding to A3 queries.

  3.  Terminal 3: balance node (steer_test mode)
          ros2 run new_balance balance_node --ros-args -p mode:=steer_test

  4.  Terminal 4: enable the controller
          ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"

  5.  Terminal 4: command a steer rate (rad/s).  Positive = one direction,
      negative = the opposite direction — verify which is which physically.

          # Gentle test first
          ros2 topic pub /new_balance/steer_rate_ref std_msgs/Float64MultiArray "data: [1.0]"

          # Reverse
          ros2 topic pub /new_balance/steer_rate_ref std_msgs/Float64MultiArray "data: [-1.0]"

          # Stop
          ros2 topic pub /new_balance/steer_rate_ref std_msgs/Float64MultiArray "data: [0.0]"

  6.  Monitor status (optional, any terminal):
          ros2 topic echo /new_balance/status

      Watch "steer_angle_deg" — when it reaches ±30°, the node stops the
      steer motor (you will see a "STEER FAULT" error and the steer motor
      will be CF-disabled until manually re-centered).

  7.  To stop everything, Ctrl-C all terminals in reverse order.

  NOTE: rear_publisher is NOT needed for this mode (drive motor is idle).


╔══════════════════════════════════════════════════════════════════════════╗
║                         MODE 3a — full (fixed speed)                     ║
║      Complete state-feedback balancing, fixed target speed              ║
╚══════════════════════════════════════════════════════════════════════════╝

  Purpose: Run the full balance controller with a fixed target speed.
  The rear wheel ramps up to target_speed over acceleration_time and
  holds constant.

  Prerequisites: same as steer_test.

  Step-by-step
  ────────────
  1.  Terminal 1: IMU → /roll_info
          ros2 run new_balance roll_publisher

  2.  Terminal 2: steer CAN → /steer_info
          ros2 run new_balance steer_publisher

      Wait for "Steer zero locked".

  3.  Terminal 3: drive CAN → /rear_info
          ros2 run new_balance rear_publisher

  4.  Terminal 4: balance node (full mode, fixed speed)
          ros2 run new_balance balance_node --ros-args -p mode:=full -p vctrl:=fixed -p method:=pole

      Or with distinct poles (3 reals):
          ros2 run new_balance balance_node --ros-args -p mode:=full -p vctrl:=fixed -p method:=pole_distinct -p poles_ctr:=[-5.0,-10.0,-20.0]

      Or with a complex conjugate pair — 6 values, [re,im] per pole,
      i.e. -9.1504±3.3347j and -6.2280 below (see k_to_poles.py):
          ros2 run new_balance balance_node --ros-args -p mode:=full -p vctrl:=fixed -p method:=pole_distinct \
              -p poles_ctr:=[-9.150423,-3.334703,-9.150423,3.334703,-6.227983,0.0]

      Or with LQR:
          ros2 run new_balance balance_node --ros-args -p mode:=full -p vctrl:=fixed -p method:=lqr

  5.  Terminal 5: enable balancing
          ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"

      The bike will attempt to self-balance.  The rear wheel will ramp up
      to target_speed (1.0 m/s) over acceleration_time (1.5 s).

  6.  Monitor (any terminal):
          ros2 topic echo /new_balance/status

      Key fields in status JSON:
        "roll_deg"          — current roll angle
        "steer_angle_deg"   — steering angle
        "speed_m_s"         — forward speed estimate
        "enabled"           — controller armed
        "sat_outer"         — true if the control output is being clamped
        "steer_fault"       — true if steer angle exceeded limit (motor CF-disabled)
        "steer_c1"          — raw C1 value sent to the steer motor
        "drive_c1"          — raw C1 value sent to the drive motor
        "K"                 — current feedback gains (gain-scheduled)
        "timing"            — {roll_pub_to_cb_ms, roll_age_ms, compute_us, …}  see Timing section

  7.  Emergency stop (any terminal):
          ros2 topic pub /bike_balance/enable std_msgs/Bool "data: false"

      The controller auto-disarms if:
        - |roll| > max_roll_angle_deg (30°)
        - steer or roll topic goes stale (>0.5 s)
        - |steer_angle| > steer_angle_limit_deg (30°) → steer motor CF-disabled
          and then left alone (no C1 goes to it while the fault holds);
          auto-recovers when the angle returns inside
          steer_recovery_ratio × the limit (default 0.01 → 0.3° at a 30° limit)

  8.  To stop everything, Ctrl-C all terminals in reverse order.

  Dry-run (hardware-free) alternative:
      Add ``-p dry_run:=true`` to skip all CAN output.  This lets you run the
      full control pipeline and inspect computed outputs + timing without any
      risk of motor movement.  CAN bus must still be ``up`` so the
      ``can.interface.Bus`` constructor succeeds, but no frames are sent.


╔══════════════════════════════════════════════════════════════════════════╗
║                         MODE 3b — full (RC speed + steer)               ║
║      Complete state-feedback balancing, RC speed & steer                ║
╚══════════════════════════════════════════════════════════════════════════╝

  Purpose: Same as 3a, but the target forward speed AND the target lean
  are both controlled in real time by the RC transmitter sticks.
  /rc_info now carries two floats: [drive_rpm, steer_rad], the second of
  which this node reads as the desired roll angle.

  Additional prerequisite: RC transmitter + SBUS receiver connected to
  /dev/ttyUSB1 via USB2SBUS module.

  Step-by-step
  ────────────
  1.  Terminal 1: IMU → /roll_info
          ros2 run new_balance roll_publisher

  2.  Terminal 2: steer CAN → /steer_info
          ros2 run new_balance steer_publisher

  3.  Terminal 3: drive CAN → /rear_info
          ros2 run new_balance rear_publisher

  4.  Terminal 4: RC receiver → /rc_info
          ros2 run new_balance rc_publisher

      Verify it's receiving frames:
          ros2 topic echo /rc_info
      Should show [rpm, steer_rad, servo_up, publish_ts] — channel 2 → drive
      RPM (centre ≈ 0), channel 0 → target steer angle in radians (±40°).

  5.  Terminal 5: balance node (full mode, RC speed + steer)
          ros2 run new_balance balance_node --ros-args -p mode:=full -p vctrl:=rc -p method:=pole

  6.  Terminal 6: enable balancing
          ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"

      The bike will attempt to self-balance.  Speed is controlled by the
      throttle stick via drive RPM: centre = 0 RPM, full forward ≈
      +rc_max_rpm (500 RPM), full back ≈ −rc_max_rpm.
      Lean is controlled by the steer stick: /rc_info.data[1] is used as the
      desired ROLL angle — solve_equilibrium(roll_ref) produces
      [steer_eq, roll_ref, 0] as the target state for the
      step_with_state_reference control law, i.e. it works out the steering
      angle that holds that lean.  Centre stick = upright.

  7.  Monitor and emergency stop: same as MODE 3a.

  8.  To stop everything, Ctrl-C all terminals in reverse order.


╔══════════════════════════════════════════════════════════════════════════╗
║                      Launching with params.yaml                         ║
╚══════════════════════════════════════════════════════════════════════════╝

      # Build first
      cd source/new_balance
      colcon build
      source install/setup.bash

      # Fixed-speed mode (all four nodes)
      ros2 run new_balance roll_publisher   --ros-args --params-file install/new_balance/share/new_balance/config/params.yaml &
      ros2 run new_balance steer_publisher  --ros-args --params-file install/new_balance/share/new_balance/config/params.yaml &
      ros2 run new_balance rear_publisher   --ros-args --params-file install/new_balance/share/new_balance/config/params.yaml &
      ros2 run new_balance balance_node     --ros-args --params-file install/new_balance/share/new_balance/config/params.yaml

      # RC mode (all five nodes — note vctrl:=rc override)
      ros2 run new_balance roll_publisher   --ros-args --params-file install/new_balance/share/new_balance/config/params.yaml &
      ros2 run new_balance steer_publisher  --ros-args --params-file install/new_balance/share/new_balance/config/params.yaml &
      ros2 run new_balance rear_publisher   --ros-args --params-file install/new_balance/share/new_balance/config/params.yaml &
      ros2 run new_balance rc_publisher     --ros-args --params-file install/new_balance/share/new_balance/config/params.yaml &
      ros2 run new_balance balance_node     --ros-args --params-file install/new_balance/share/new_balance/config/params.yaml -p vctrl:=rc

      # Dry-run (no CAN output) — all five nodes, no physical motors
      ros2 run new_balance balance_node     --ros-args --params-file install/new_balance/share/new_balance/config/params.yaml -p dry_run:=true
"""

from __future__ import annotations

import json
import math
import struct
import time

import can
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray, String

from nominal_bike_control.bicycle_model import BicycleParameters, LinearBicycleModel
from nominal_bike_control.controller import NominalBikeController

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
_RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)
_C1_SCALE = 100.0

# Steer-fault recovery band for steer_test (bench) mode, as a fraction of
# steer_angle_limit_deg.  full mode uses the steer_recovery_ratio parameter.
_BENCH_RECOVERY_RATIO = 0.5


def _rad_s_to_c1(rad_s: float) -> int:
    return int(round(rad_s * _RPM_PER_RAD_S * _C1_SCALE))


def _decode_poles(values) -> np.ndarray:
    """Decode the ``poles_ctr`` parameter into three complex poles.

    ROS 2 has no complex parameter type, so ``poles_ctr`` is a double[] in one
    of two forms:

      3 values  [p1, p2, p3]                      — all-real poles, as before
      6 values  [re1, im1, re2, im2, re3, im3]     — interleaved real/imag

    A gain that is well tuned at one speed usually inverts to one real pole
    plus a complex conjugate pair (see k_to_poles.py), which the 3-value form
    cannot express.  ``scipy.signal.place_poles`` handles a conjugate pair
    directly, so the 6-value form needs no other change downstream.

    The multiset must be closed under conjugation, otherwise the characteristic
    polynomial is complex and no real gain realises it.  ``place_poles`` also
    rejects a pole repeated more than rank(Bu) = 1 times, so exact duplicates
    are reported here — with a clear message — rather than deep inside scipy.
    """
    raw = [float(v) for v in (values or ())]

    if len(raw) == 3:
        poles = np.array(raw, dtype=complex)
    elif len(raw) == 6:
        poles = np.array([complex(raw[0], raw[1]),
                          complex(raw[2], raw[3]),
                          complex(raw[4], raw[5])], dtype=complex)
    else:
        raise ValueError(
            f"poles_ctr needs 3 values [p1, p2, p3] or 6 values "
            f"[re1, im1, re2, im2, re3, im3]; got {len(raw)}")

    # Closed under conjugation <=> the characteristic polynomial is real.
    if float(np.max(np.abs(np.poly(poles).imag))) > 1e-9:
        raise ValueError(
            f"poles_ctr is not closed under conjugation: {poles.tolist()}. "
            "Complex poles must come as a conjugate pair.")

    if float(np.max(poles.real)) >= 0.0:
        raise ValueError(
            f"poles_ctr must all have negative real parts; got {poles.tolist()}")

    for i in range(3):
        for j in range(i + 1, 3):
            if abs(poles[i] - poles[j]) < 1e-9:
                raise ValueError(
                    f"poles_ctr repeats {poles[i]}; place_poles allows a pole "
                    "at most rank(Bu) = 1 times. Separate them slightly, or "
                    "use method:=pole (wc) for a genuine triple pole.")
    return poles


def _format_poles(poles: np.ndarray) -> str:
    """Compact log form, e.g. ``-9.1504-3.3347j, -9.1504+3.3347j, -6.2280``."""
    parts = []
    for z in np.asarray(poles, dtype=complex).ravel():
        if abs(z.imag) > 1e-9:
            parts.append(f"{z.real:.4f}{z.imag:+.4f}j")
        else:
            parts.append(f"{z.real:.4f}")
    return ", ".join(parts)


# ======================================================================
# Main node
# ======================================================================
class BalanceNode(Node):
    """Self-balancing controller — reads topics, writes CAN C1 speed commands.

    Everything runs in one ROS timer callback at ``control_hz``: state
    feedback, safety latches, and both motors' C1 output.
    """

    def __init__(self) -> None:
        super().__init__("new_balance_node")

        # ════════════════ Parameters ════════════════
        self.declare_parameter("mode", "full")
        self.declare_parameter("method", "lqr")
        self.declare_parameter("vctrl", "rc")      # "fixed" | "rc"

        # CAN
        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("steer_can_id", "101")
        self.declare_parameter("drive_can_id", "102")

        # Speed profile
        self.declare_parameter("wheel_radius", 0.07)
        self.declare_parameter("target_speed", 1.0)
        self.declare_parameter("acceleration_time", 1.5)
        self.declare_parameter("drive_direction_invert", True)
        self.declare_parameter("steer_direction_invert", True)
        self.declare_parameter("rc_max_rpm",
                             2.7 / (2.0 * math.pi * 0.07) * 60.0)  # = 2.7 m/s → rpm

        # Control — one rate for the whole node
        self.declare_parameter("control_hz", 200.0)
        # Gain rescheduling runs slower than the control law; see _tick_control.
        self.declare_parameter("gain_update_hz", 50.0)
        self.declare_parameter("min_scheduling_speed", 1.0)
        self.declare_parameter("max_steer_velocity", 15.0)

        # Pole
        self.declare_parameter("pole_wc", -15.0)
        # 3 values [p1, p2, p3] for real poles, or 6 values
        # [re1, im1, re2, im2, re3, im3] to carry a complex conjugate pair.
        # See _decode_poles, and k_to_poles.py for solving these from a gain.
        #
        # Every set below is solved at v = 1 m/s against the bike_* defaults
        # further down; re-solve them all with k_to_poles.py if any bike_*
        # value changes.  Swap by moving the value into declare_parameter, or
        # override on the CLI / in params.yaml without touching this file.
        #
        #   [A] K = [3.97165, 45.47517, 3.13077]   wn 9.76 rad/s, zeta 0.315
        #       real -7.9865, pair -3.0713 +/- 9.2655j
        #       Measured on the bike as marginal — roll converged on some runs
        #       and diverged on others, hence [B].
        #       [-7.986503, 0.0, -3.071325, -9.265501, -3.071325, 9.265501]
        #
        #   [B] K = [3.80000, 47.00000, 3.13000]   wn 10.08 rad/s, zeta 0.297
        #       real -7.9726, pair -2.9912 +/- 9.6243j   <-- active
        #       More roll authority (K_phi 45.5 -> 47.0) at slightly lower
        #       damping; K_phidot unchanged.
        #
        # Active: [B]
        self.declare_parameter(
            "poles_ctr",
            [-7.972585, 0.0, -2.991210, -9.624308, -2.991210, 9.624308])

        # LQR
        self.declare_parameter("lqr_Q_diag", [1.2961771265077047, 840.8586455916102, 4.692318728430152])
        self.declare_parameter("lqr_R_diag", [1.1828982163197082])

        # Bike model (tutorial defaults, matched to params.yaml)
        self.declare_parameter("bike_mass", 2.44)
        self.declare_parameter("bike_wheelbase", 0.2900)
        self.declare_parameter("bike_rear_to_com", 0.1225)
        self.declare_parameter("bike_com_height", 0.1400)
        self.declare_parameter("bike_trail", 0.0135)
        self.declare_parameter("bike_steering_axis_angle", 0.1745)
        self.declare_parameter("bike_roll_inertia", 0.0438)

        # Safety
        self.declare_parameter("max_roll_angle_deg", 30.0)
        self.declare_parameter("steer_angle_limit_deg", 60.0)
        self.declare_parameter("steer_recovery_ratio", 0.01)
        # How long /steer_info or /roll_info may go quiet before the
        # controller disarms.  See the stale checks in _tick_control.
        self.declare_parameter("stale_timeout_s", 0.15)
        self.declare_parameter("enable_on_start", False)
        self.declare_parameter("balance_start_delay", 0.1)

        # Dry run: compute + publish without CAN output
        self.declare_parameter("dry_run", False)

        # ════════════════ Read parameters ════════════════
        self._mode: str = self.get_parameter("mode").value
        if self._mode not in ("sign_check", "steer_test", "full"):
            self.get_logger().error(f"Unknown mode '{self._mode}', falling back to 'full'")
            self._mode = "full"

        self._method: str = self.get_parameter("method").value
        self._vctrl: str = self.get_parameter("vctrl").value
        self._can_iface: str = self.get_parameter("can_interface").value
        self._steer_cid: int = int(self.get_parameter("steer_can_id").value, 16)
        self._drive_cid: int = int(self.get_parameter("drive_can_id").value, 16)
        self._wheel_radius: float = self.get_parameter("wheel_radius").value
        self._target_speed: float = self.get_parameter("target_speed").value
        self._accel_time: float = self.get_parameter("acceleration_time").value
        self._drive_invert: bool = self.get_parameter("drive_direction_invert").value
        self._steer_invert: bool = self.get_parameter("steer_direction_invert").value
        self._rc_max_rpm: float = self.get_parameter("rc_max_rpm").value
        self._control_hz: float = self.get_parameter("control_hz").value
        self._control_dt: float = 1.0 / max(self._control_hz, 1.0)
        self._gain_hz: float = self.get_parameter("gain_update_hz").value
        # Turn the requested rate into an integer tick decimation, so the
        # control:reschedule ratio is exact and no float comparison can drift.
        # 200 / 50 -> every 4th tick.  Rates that do not divide evenly are
        # rounded, and the achieved rate is control_hz / _gain_every — logged
        # below and republished in the status JSON so it is never a guess.
        self._gain_every: int = max(1, int(round(
            self._control_hz / max(self._gain_hz, 1.0))))
        self._min_speed: float = self.get_parameter("min_scheduling_speed").value
        self._max_steer_vel: float = self.get_parameter("max_steer_velocity").value
        self._pole_wc: float = self.get_parameter("pole_wc").value
        self._max_roll_rad: float = np.deg2rad(
            self.get_parameter("max_roll_angle_deg").value)
        self._steer_angle_limit_deg: float = self.get_parameter("steer_angle_limit_deg").value
        self._steer_recovery_ratio: float = self.get_parameter("steer_recovery_ratio").value
        self._stale_timeout: float = self.get_parameter("stale_timeout_s").value
        self._enable_on_start: bool = self.get_parameter("enable_on_start").value
        self._balance_delay: float = self.get_parameter("balance_start_delay").value
        self._dry_run: bool = self.get_parameter("dry_run").value

        # ════════════════ Open CAN bus ════════════════
        self.get_logger().info(f"Opening CAN bus {self._can_iface} …")
        self._bus = can.interface.Bus(
            channel=self._can_iface, bustype="socketcan", receive_own_messages=False)

        # ════════════════ Bicycle model + controller ════════════════
        roll_iner = self.get_parameter("bike_roll_inertia").value
        bike_params = BicycleParameters(
            rear_contact_to_com=self.get_parameter("bike_rear_to_com").value,
            wheelbase=self.get_parameter("bike_wheelbase").value,
            trail=self.get_parameter("bike_trail").value,
            com_height=self.get_parameter("bike_com_height").value,
            mass=self.get_parameter("bike_mass").value,
            roll_inertia=None if roll_iner < 0 else roll_iner,
            steering_axis_angle=self.get_parameter("bike_steering_axis_angle").value,
        )
        # track_roll selects the controlled output Co that solve_equilibrium
        # inverts.  True  -> Co = [0, 1, 0], the reference is the ROLL angle.
        # False -> Co = [1, 0, 0], the reference is the steering angle.
        # Co appears nowhere in the gain computation, so this changes only what
        # the RC reference means, never K.
        self._bike = LinearBicycleModel(
            dt=self._control_dt, parameters=bike_params,
            track_roll=True, min_forward_speed=self._min_speed,
        )
        self._build_controller()

        # ════════════════ Runtime state ════════════════
        self._roll_angle: float = 0.0
        self._roll_rate: float = 0.0
        self._body_rates: np.ndarray = np.array([0.0, 0.0, 0.0])
        self._steer_angle: float = 0.0
        self._rear_rate: float = 0.0

        self._roll_ts: float = 0.0
        self._steer_ts: float = 0.0
        self._rear_ts: float = 0.0

        self._enabled: bool = self._enable_on_start
        self._start_time: float | None = None
        self._steer_rate_ref: float = 0.0    # last commanded steer rate (rad/s)
        self._prev_steer_c1: int = 0
        self._prev_drive_c1: int = 0
        self._sat_outer: bool = False
        # Reference actually fed to solve_equilibrium this tick, and the
        # equilibrium it returned.  Kept as state purely so /new_balance/debug
        # can carry them; the control law reads the locals.  Zero whenever no
        # equilibrium was solved (disarmed, start-up delay, steer_test), so the
        # topic shows "no reference in force" rather than a stale one.
        self._roll_ref: float = 0.0
        self._x_eq: np.ndarray = np.zeros(3)
        self._steer_fault: bool = False     # 转向角超限 → CF 失能，需手动归零后自动恢复
        self._forward_speed: float = 0.0
        # Gain-rescheduling decimation counter.  Starting at 0 makes the very
        # first control tick reschedule, then every _gain_every-th one after.
        self._gain_tick: int = 0
        self._gain_sched_v: float = float("nan")   # speed the live K belongs to
        self._rc_drive_rpm: float = 0.0
        # /rc_info.data[1].  rc_publisher still calls this field "steer_rad" and
        # still scales the stick as an angle in radians, but this node feeds it
        # to solve_equilibrium as the ROLL reference — see track_roll above.
        self._rc_roll_ref_rad: float = 0.0
        self._rc_servo_flag: float = 0.0        # /rc_info.data[2] — 0 or 1
        self._rc_servo_flag_prev: float = 0.0   # last value seen, for edge detection
        self._servo_init_done: bool = False
        self._ext_steer_rate_ref: float = 0.0

        self._last_tick: float = 0.0
        self._jitter_samples: list[float] = []

        # ── Timing instrumentation ──
        # Published-timestamp extracted from each topic (float seconds since epoch)
        self._roll_pub_ts: float = 0.0
        self._steer_pub_ts: float = 0.0
        self._rear_pub_ts: float = 0.0
        self._rc_pub_ts: float = 0.0
        # Timestamp when each subscriber callback last fired
        self._roll_cb_ts: float = 0.0
        self._steer_cb_ts: float = 0.0
        self._rear_cb_ts: float = 0.0
        self._rc_cb_ts: float = 0.0
        # Aggregated timing (populated each tick, published in status)
        self._timing: dict[str, float] = {}

        # ════════════════ Subscriptions ════════════════
        self.create_subscription(Float64MultiArray, "roll_info", self._roll_cb, 10)
        self.create_subscription(Float64MultiArray, "steer_info", self._steer_cb, 10)
        self.create_subscription(Float64MultiArray, "rear_info", self._rear_cb, 10)
        self.create_subscription(Bool, "bike_balance/enable", self._enable_cb, 10)

        if self._vctrl == "rc":
            self.create_subscription(
                Float64MultiArray, "rc_info", self._rc_cb, 10)

        if self._mode == "steer_test":
            self.create_subscription(
                Float64MultiArray, "new_balance/steer_rate_ref",
                self._steer_ref_cb, 10)

        # ════════════════ Publishers ════════════════
        self._status_pub = self.create_publisher(String, "new_balance/status", 10)
        self._debug_pub = self.create_publisher(Float64MultiArray, "new_balance/debug", 10)
        self._steer_cmd_pub = self.create_publisher(Float64MultiArray, "steer_cmd", 10)
        self._servo_cmd_pub = self.create_publisher(String, 'servo_cmd', 10)

        # ════════════════ Timers ════════════════
        # One control timer for every mode; status stays a 2 Hz human-readable
        # digest while the per-tick numbers go out on /new_balance/debug.
        self._timer = self.create_timer(self._control_dt, self._tick)
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._last_tick = time.time()

        self.get_logger().info(
            f"NewBalance | mode={self._mode} method={self._method} | "
            f"control={self._control_hz:.0f}Hz (dt={self._control_dt*1000:.1f}ms) "
            f"gain_resched=every {self._gain_every} ticks "
            f"({self._control_hz/self._gain_every:.1f}Hz) | "
            f"CAN={self._can_iface} steer=0x{self._steer_cid:03X} drive=0x{self._drive_cid:03X} | "
            f"target_v={self._target_speed}m/s vctrl={self._vctrl} "
            f"rc_max_rpm={self._rc_max_rpm} enabled={self._enabled}")

    # ==================================================================
    # Controller
    # ==================================================================
    def _build_controller(self) -> None:
        common = dict(u_min=-self._max_steer_vel, u_max=self._max_steer_vel)
        if self._method == "pole":
            self._ctrl = NominalBikeController(
                self._bike.sys, method="place_multiple_poles",
                wc=self._pole_wc, **common)
        elif self._method == "pole_distinct":
            # Complex pairs arrive interleaved; keep the decoded array so every
            # gain-scheduling reschedule re-places the same poles.
            poles_ctr = _decode_poles(self.get_parameter("poles_ctr").value)
            self._ctrl = NominalBikeController(
                self._bike.sys, method="place_distinct_poles",
                poles_ctr=poles_ctr, **common)
        elif self._method == "lqr":
            Qc = np.diag(self.get_parameter("lqr_Q_diag").value)
            Rc = np.diag(self.get_parameter("lqr_R_diag").value)
            self._ctrl = NominalBikeController(
                self._bike.sys, method="lqr", Qc=Qc, Rc=Rc, **common)
        else:
            raise ValueError(f"Unknown method: {self._method}")
        detail = (f" poles=[{_format_poles(self._ctrl.poles)}]"
                  if self._method == "pole_distinct" else "")
        self.get_logger().info(
            f"Controller: {self._method}{detail} "
            f"K={np.array2string(self._ctrl.K.flatten(), precision=4)}")

    # ==================================================================
    # Topic callbacks
    # ==================================================================
    def _roll_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 7:
            self._roll_cb_ts = time.time()
            self._roll_angle = float(msg.data[0])
            self._roll_rate = float(msg.data[3])
            self._body_rates = np.array([
                float(msg.data[4]), float(msg.data[5]), float(msg.data[6]),
            ])
            self._roll_ts = self._roll_cb_ts
            if len(msg.data) >= 8:
                self._roll_pub_ts = float(msg.data[7])

    def _steer_cb(self, msg: Float64MultiArray) -> None:
        """[steer_angle (rad), publish_ts].

        The motor's speed loop is closed in firmware now, so the steer rate
        is no longer polled or fed back — only the angle, which the state
        vector and the soft angle limit both need.
        """
        if len(msg.data) >= 1:
            self._steer_cb_ts = time.time()
            self._steer_angle = float(msg.data[0])
            self._steer_ts = self._steer_cb_ts
            if len(msg.data) >= 2:
                self._steer_pub_ts = float(msg.data[1])

    def _rear_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 1:
            self._rear_cb_ts = time.time()
            self._rear_rate = float(msg.data[0])
            self._rear_ts = self._rear_cb_ts
            if len(msg.data) >= 2:
                self._rear_pub_ts = float(msg.data[1])

    def _enable_cb(self, msg: Bool) -> None:
        was = self._enabled
        self._enabled = bool(msg.data)
        if self._enabled and not was:
            self._start_time = None
            self._sat_outer = False
            self._steer_fault = False        # 重新使能时清除转向故障；若角度仍超限，
                                             # 下一个 tick 会重新锁定
            self.get_logger().info("BALANCE ENABLED")
        elif not self._enabled and was:
            self._stop_motors()
            self.get_logger().info("BALANCE DISABLED — motors stopped")

    def _steer_ref_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 1:
            self._ext_steer_rate_ref = float(msg.data[0])

    def _rc_cb(self, msg: Float64MultiArray) -> None:
        """Store the RC roll reference, drive RPM, and servo-up flag.

        Layout: [drive_rpm, steer_rad, servo_up, publish_ts].  RPM clamped
        to ±rc_max_rpm.

        data[1] keeps rc_publisher's name and its radians-of-angle scaling,
        but is consumed here as the desired ROLL angle, not the desired
        steering angle: the model is built with track_roll=True, so
        solve_equilibrium inverts it into the steering angle needed to hold
        that lean.  It is stored raw — no clamp against max_roll_angle_deg.
        """
        self._rc_cb_ts = time.time()
        if len(msg.data) >= 2:
            raw_rpm = float(msg.data[0])
            self._rc_drive_rpm = max(-self._rc_max_rpm, min(raw_rpm, self._rc_max_rpm))
            self._rc_roll_ref_rad = float(msg.data[1]) * 0.5 # +-40 degrees (in rad) -> +-20 degrees (in rad)
            if len(msg.data) >= 3:
                self._rc_servo_flag = float(msg.data[2])
            if len(msg.data) >= 4:
                self._rc_pub_ts = float(msg.data[3])
        elif len(msg.data) >= 1:
            raw_rpm = float(msg.data[0])
            self._rc_drive_rpm = max(-self._rc_max_rpm, min(raw_rpm, self._rc_max_rpm))

    def _servo_up_cb(self) -> None:
        """Triggered when /rc_info.data[2] (_servo_up) transitions to 1."""
        msg1 = String()
        msg1.data = 'm 1 1424 3000 255'

        msg2 = String()
        msg2.data = 'm 2 624 3000 255'

        self._servo_cmd_pub.publish(msg1)
        self._servo_cmd_pub.publish(msg2)

    def _servo_down_cb(self) -> None:
        """Triggered once at start of the control tick and whenever /rc_info.data[2] (_servo_up) transitions to 0."""
        msg1 = String()
        msg1.data = 't 1 1' # enable servo 1

        msg2 = String()
        msg2.data = 't 2 1' # enable servo 2

        msg3 = String()
        msg3.data = 'm 1 944 3000 255' # command servo 1 to 954 (pos), with speed 3000 and accel 255

        msg4 = String()
        msg4.data = 'm 2 1104 3000 255' # command servo 2 to 1094 (pos), with speed 3000 and accel 255

        self._servo_cmd_pub.publish(msg1)
        self._servo_cmd_pub.publish(msg2)
        self._servo_cmd_pub.publish(msg3)
        self._servo_cmd_pub.publish(msg4)


    # ==================================================================
    # CAN writing (python-can — no subprocess)
    # ==================================================================
    def _send_speed(self, can_id: int, rad_s: float) -> int:
        """Send C1 speed command via python-can.  Returns the raw C1 value.

        C1 frame: <0xC1> <4-byte LE int32 speed in rpm×100>

        The raw value is returned (and still computed under ``dry_run``) so
        callers can log exactly what was, or would have been, put on the bus.
        """
        c1 = _rad_s_to_c1(rad_s)
        if self._dry_run:
            return c1
        data = bytes([0xC1]) + struct.pack("<i", c1)
        try:
            self._bus.send(can.Message(
                arbitration_id=can_id, data=data, is_extended_id=False))
        except can.CanError as exc:
            self.get_logger().warn(
                f"CAN send failed: 0x{can_id:03X} C1={c1}: {exc}",
                throttle_duration_sec=2.0)
        return c1

    def _stop_motors(self) -> None:
        """Send C1=0 + CF (motor disable) to both motors."""
        self._prev_steer_c1 = 0
        self._prev_drive_c1 = 0
        if self._dry_run:
            return
        for cid in (self._steer_cid, self._drive_cid):
            self._send_speed(cid, 0.0)
            self._send_cf(cid)

    def _send_cf(self, can_id: int) -> None:
        """Send CF (MOTOR_DISABLE) to a single motor — same as ``cansend can0 <id>#CF``."""
        if self._dry_run:
            return
        try:
            self._bus.send(can.Message(
                arbitration_id=can_id, data=bytes([0xCF]),
                is_extended_id=False))
        except can.CanError:
            pass

    # ==================================================================
    # Timing helper — compute latencies from captured timestamps
    # ==================================================================
    def _update_timing(self, tick_start: float) -> None:
        """Populate self._timing with per-topic latencies in milliseconds.

        Must be called from within the control tick after all subscriber
        callbacks for this cycle are complete.
        """
        def _ms(dt: float) -> float:
            return round(max(0.0, dt) * 1000.0, 3)

        t = self._timing  # shorthand

        # Roll: pub → cb (ROS2 transport) and cb → tick (timer alignment)
        if self._roll_pub_ts > 0 and self._roll_cb_ts > 0:
            t["roll_pub_to_cb_ms"] = _ms(self._roll_cb_ts - self._roll_pub_ts)
            t["roll_cb_to_tick_ms"] = _ms(tick_start - self._roll_cb_ts)
            t["roll_age_ms"] = _ms(tick_start - self._roll_pub_ts)
        else:
            t["roll_pub_to_cb_ms"] = -1.0
            t["roll_cb_to_tick_ms"] = -1.0
            t["roll_age_ms"] = -1.0

        # Steer
        if self._steer_pub_ts > 0 and self._steer_cb_ts > 0:
            t["steer_pub_to_cb_ms"] = _ms(self._steer_cb_ts - self._steer_pub_ts)
            t["steer_cb_to_tick_ms"] = _ms(tick_start - self._steer_cb_ts)
            t["steer_age_ms"] = _ms(tick_start - self._steer_pub_ts)
        else:
            t["steer_pub_to_cb_ms"] = -1.0
            t["steer_cb_to_tick_ms"] = -1.0
            t["steer_age_ms"] = -1.0

        # Rear
        if self._rear_pub_ts > 0 and self._rear_cb_ts > 0:
            t["rear_pub_to_cb_ms"] = _ms(self._rear_cb_ts - self._rear_pub_ts)
            t["rear_age_ms"] = _ms(tick_start - self._rear_pub_ts)
        else:
            t["rear_pub_to_cb_ms"] = -1.0
            t["rear_age_ms"] = -1.0

        # RC (only when vctrl=rc)
        if self._vctrl == "rc" and self._rc_pub_ts > 0 and self._rc_cb_ts > 0:
            t["rc_pub_to_cb_ms"] = _ms(self._rc_cb_ts - self._rc_pub_ts)
            t["rc_age_ms"] = _ms(tick_start - self._rc_pub_ts)
        else:
            t["rc_pub_to_cb_ms"] = -1.0
            t["rc_age_ms"] = -1.0

    # ==================================================================
    # Main tick dispatcher — one timer at control_hz for every mode
    # ==================================================================
    def _tick(self) -> None:
        now = time.time()
        if self._last_tick > 0:
            measured_dt = now - self._last_tick
            self._jitter_samples.append(measured_dt)
            if len(self._jitter_samples) > 250:
                self._jitter_samples.pop(0)
        self._last_tick = now
        if self._start_time is None:
            self._start_time = now

        if self._mode == "sign_check":
            self._tick_sign_check(now)
        else:
            self._tick_control(now)

    # ==================================================================
    # tick: sign_check  — log corrected sensor state, no motors
    # ==================================================================
    def _tick_sign_check(self, now: float) -> None:
        self._update_timing(now)
        if self._steer_ts > 0 and now - self._steer_ts > 1.0:
            self.get_logger().warn("Steer data stale", throttle_duration_sec=2.0)
        if self._roll_ts > 0 and now - self._roll_ts > 1.0:
            self.get_logger().warn("Roll data stale", throttle_duration_sec=2.0)
        if self._rear_ts > 0 and now - self._rear_ts > 1.0:
            self.get_logger().warn("Rear data stale", throttle_duration_sec=2.0)

        self.get_logger().info(
            f"[SIGN_CHECK] roll={np.rad2deg(self._roll_angle):.1f}° "
            f"p={np.rad2deg(self._body_rates[0]):.1f} "
            f"q={np.rad2deg(self._body_rates[1]):.1f} "
            f"r={np.rad2deg(self._body_rates[2]):.1f} dps "
            f"roll_rate={np.rad2deg(self._roll_rate):.1f} dps | "
            f"steer={np.rad2deg(self._steer_angle):.1f}° "
            f"rear={self._rear_rate:.2f} rad/s",
            throttle_duration_sec=2.0)

        debug = Float64MultiArray()
        debug.data = [
            self._roll_angle, self._roll_rate,
            float(self._body_rates[0]), float(self._body_rates[1]),
            float(self._body_rates[2]),
            self._steer_angle, self._rear_rate,
        ]
        self._debug_pub.publish(debug)

    # ==================================================================
    # Control tick (control_hz) — full and steer_test modes
    # ==================================================================
    def _tick_control(self, now: float) -> None:
        """State feedback + gain scheduling + safety + both motors' C1.

        Runs on a ROS2 timer at ``control_hz``.  There is no second thread:
        the steer command produced here goes straight onto the bus as a C1
        speed command and the motor's own speed loop closes it.
        """
        elapsed = now - self._start_time if self._start_time else 0.0

        self._update_timing(now)

        if (not self._servo_init_done) and (self._rc_servo_flag == 0.0):
            self._servo_down_cb()
            self._servo_init_done = True

        # ---- servo edge trigger from RC switch ----
        if self._rc_servo_flag != self._rc_servo_flag_prev:
            if self._rc_servo_flag == 1.0:
                self._servo_up_cb()
            else:
                self._servo_down_cb()
            self._rc_servo_flag_prev = self._rc_servo_flag

        # ---- stale data checks ----
        # stale_timeout_s is sized against the plant, not against the link.
        # The open-loop capsize mode sits at +sqrt(m g h / I) = 8.75 rad/s,
        # a 114 ms time constant: a 1 deg disturbance reaches the 30 deg
        # max_roll_angle_deg in 389 ms.  The old hard-coded 0.5 s therefore
        # fired *after* the bike was already down — it stopped the motors but
        # protected nothing.  0.15 s is 30 missed messages out of a 200 Hz
        # stream, far beyond anything scheduling jitter produces (measured
        # worst case: 3 consecutive), and 1.3 capsize time constants.
        #
        # Both branches log.  The disarmed case used to return in total
        # silence, which is how "every node is up but the rear wheel does
        # nothing" presented on the bike: no command, no error, no clue.
        if self._steer_ts == 0.0 or now - self._steer_ts > self._stale_timeout:
            if self._enabled:
                self.get_logger().warn("Steer data stale — disabling",
                                       throttle_duration_sec=2.0)
                self._enabled = False
                self._stop_motors()
            else:
                self.get_logger().warn(
                    "No /steer_info — idle, no motor commands are being sent",
                    throttle_duration_sec=5.0)
            return
        if self._roll_ts == 0.0 or now - self._roll_ts > self._stale_timeout:
            if self._enabled:
                self.get_logger().warn("Roll data stale — disabling",
                                       throttle_duration_sec=2.0)
                self._enabled = False
                self._stop_motors()
            else:
                self.get_logger().warn(
                    "No /roll_info — idle, no motor commands are being sent",
                    throttle_duration_sec=5.0)
            return
        # Rear speed only feeds gain scheduling, so a gap warns and the tick
        # carries on.  "Never arrived" counts as stale here too — the old
        # ``_rear_ts > 0`` guard meant a rear_publisher that never started was
        # silently tolerated forever, freezing _forward_speed at 0 and with it
        # the whole gain schedule at min_scheduling_speed.  steer_test does not
        # use the drive motor at all, so it is exempt.
        if self._mode == "full" and (self._rear_ts == 0.0
                                     or now - self._rear_ts > self._stale_timeout):
            self.get_logger().warn(
                "Rear data stale — gains are scheduled on a frozen speed",
                throttle_duration_sec=2.0)

        # ---- forward speed ----
        self._forward_speed = self._wheel_radius * abs(self._rear_rate)

        # ---- speed profile ----
        if self._mode != "full":
            rear_rate_ref = 0.0          # steer_test: the drive motor stays put
        elif self._vctrl == "rc":
            drive_rpm = max(-self._rc_max_rpm,
                            min(self._rc_max_rpm, self._rc_drive_rpm))
            rear_rate_ref = drive_rpm * (2.0 * math.pi / 60.0)
        else:
            if self._accel_time > 0:
                speed_ref = min(elapsed / self._accel_time * self._target_speed, self._target_speed)
            else:
                speed_ref = self._target_speed
            rear_rate_ref = speed_ref / self._wheel_radius

        # ══════════════════════════════════════════════════════════════════
        # FAULTS FIRST — every abort path below returns before the control
        # block, so a faulted tick never pays for the gain solve or the
        # control law.  That ordering is deliberate and load-bearing, not
        # tidiness: a latched steer fault is exactly the state in which the
        # operator is hand-centering the wheel, i.e. the state in which
        # /steer_info's callback *must* get executor time for the recovery
        # band to be reachable at all.  With the control block running first,
        # a faulted tick still cost ~900 us of lqr Riccati solve, the single
        # -threaded executor starved that callback, _steer_angle froze at the
        # over-limit value, and the fault could never clear — the bike showed
        # this as "cannot re-enable under lqr, but pole is fine".  Skipping
        # the work drops a faulted tick to a few tens of microseconds.
        # ══════════════════════════════════════════════════════════════════

        # ---- steer angle limit ----
        # Latch / recovery.  (_enable_cb also clears it on arm — if the angle
        # is still out of range the next tick re-faults.)
        self._update_steer_fault()

        # ---- emergency roll stop ----
        if abs(self._roll_angle) > self._max_roll_rad:
            if self._enabled:
                self.get_logger().error(
                    f"EMERGENCY: |roll|={np.rad2deg(self._roll_angle):.1f}°")
                self._enabled = False
                self._stop_motors()
            self._gain_tick = 0          # reschedule on the tick control resumes
            self._timing["compute_us"] = round((time.time() - now) * 1e6, 0)
            return

        # ---- steer fault ----
        # The drive motor keeps following its reference so the bike rolls to
        # a stop instead of stopping dead; the steer motor gets nothing.
        #
        # "Nothing" is the point: C1 is a motion command, and per the GIM
        # protocol CF puts the motor in the free state (0xCF: 关闭电机输出,
        # 电机进入自由态) while any later C1 — C1=0 included — takes it back
        # out of that state and servos to the commanded speed.  The previous
        # version sent C0=0 here, which was harmless only because zero
        # Q-axis current is zero torque.  C1 has no such idle value, so the
        # CF sent at the latch is left to stand on its own.
        if self._steer_fault:
            if self._enabled:
                self._prev_drive_c1 = self._send_speed(
                    self._drive_cid,
                    -rear_rate_ref if self._drive_invert else rear_rate_ref)
            else:
                self._prev_drive_c1 = self._send_speed(self._drive_cid, 0.0)
            self._steer_rate_ref = 0.0
            self._prev_steer_c1 = 0
            self._gain_tick = 0          # reschedule on the tick control resumes
            self._timing["compute_us"] = round((time.time() - now) * 1e6, 0)
            return

        if self._mode == "full":
            # ---- state ----
            state = np.array([[self._steer_angle], [self._roll_angle], [self._roll_rate]])

            # ---- gain scheduling (throttled to gain_update_hz) ----
            # Rescheduling is by far the most expensive thing in this tick —
            # solve_continuous_are for lqr, Ackermann for pole — and it is the
            # only part that does not need the control rate.  Forward speed
            # cannot change meaningfully between reschedules, so re-solving K
            # every tick buys nothing and starves the subscription callbacks:
            # rclpy.spin runs a single-threaded executor, and the timer shares
            # its thread with ~800 incoming messages a second.
            #
            # Running it every gain_every-th tick instead cuts the average tick
            # cost by that factor while the control law itself keeps evaluating
            # at full rate against fresh sensor data.
            #
            # The decimation is a *tick count*, not a deadline.  Counting ticks
            # makes the ratio exact by construction — 4 control ticks per
            # reschedule, always, first tick included.  The earlier deadline
            # form (now - last >= gain_dt) could not promise that: now is a
            # time.time() epoch near 1.79e9 where a double resolves to ~400 ns,
            # so the tick landing exactly on the deadline compared just under
            # it and slipped a whole period.  200/50 ran at 40 Hz rather than
            # 50, and once timer jitter was added it alternated 4- and 5-tick
            # gaps at random.
            #
            # solve_equilibrium below reads the same A, Bu that were pushed in
            # here, so the equilibrium and the gain always describe the same
            # scheduled operating point — never a mix of two.
            if self._gain_tick == 0:
                sch_v = max(self._forward_speed, self._min_speed)
                self._bike.update_system_parameters(
                    sch_v, min_forward_speed=self._min_speed)
                # Push the rescheduled matrices into the controller.
                # update_gain() alone would recompute K from the copies taken
                # at construction time, freezing both K and the equilibrium
                # point at the model's default 2 m/s.
                self._ctrl.update_system_and_gain(self._bike.sys)
                self._gain_sched_v = sch_v
            self._gain_tick = (self._gain_tick + 1) % self._gain_every

            # ---- state feedback ----
            if elapsed < self._balance_delay or not self._enabled:
                steer_rate_ref = 0.0
                self._sat_outer = False
                self._roll_ref = 0.0
                self._x_eq = np.zeros(3)
            else:
                # The reference is a ROLL angle (track_roll=True).  With
                # Co = [0, 1, 0], solve_equilibrium returns
                #     x_eq = [-(a4/a2) roll_ref, roll_ref, 0],  u_eq = 0
                # i.e. it works out the steering angle that sustains the
                # requested lean.  a4/a2 falls as 1/v^2, so the same lean asks
                # for a much larger steering angle at low speed.  Passed
                # through unclamped by design.
                roll_ref = -self._rc_roll_ref_rad if self._vctrl == "rc" else 0.0
                eq = self._ctrl.solve_equilibrium(roll_ref)
                x_eq = eq.state
                u_raw = float(self._ctrl.step_with_state_reference(x_eq, state)[0, 0])
                steer_rate_ref = float(np.clip(u_raw, -self._max_steer_vel, self._max_steer_vel))
                self._sat_outer = abs(u_raw - steer_rate_ref) > 1e-6
                self._roll_ref = roll_ref
                self._x_eq = np.asarray(x_eq, dtype=float).ravel()
        else:
            # steer_test: the topic reference is the command.  rear_rate_ref
            # was already zeroed with the speed profile, above the fault
            # checks, so a faulted steer_test tick still commands the drive
            # motor to 0 the way it always did.
            steer_rate_ref = float(np.clip(self._ext_steer_rate_ref,
                                           -self._max_steer_vel, self._max_steer_vel))
            self._sat_outer = abs(self._ext_steer_rate_ref - steer_rate_ref) > 1e-6
            self._roll_ref = 0.0          # no equilibrium is solved in steer_test
            self._x_eq = np.zeros(3)

        # ---- send C1 to both motors ----
        # Disarmed, the drive motor is held at C1=0 exactly as before, but the
        # steer motor gets nothing: _stop_motors already CF'd it, and a C1=0
        # would take it right back out of the free state.  The previous
        # version sent C0=0 in this spot, i.e. zero torque, so a disarmed
        # steer wheel was loose — sending no motion command is what preserves
        # that.  Same reason as the steer-fault branch above.
        rear_cmd = rear_rate_ref if self._enabled else 0.0
        drive_out = -rear_cmd if self._drive_invert else rear_cmd
        self._prev_drive_c1 = self._send_speed(self._drive_cid, drive_out)

        if self._enabled:
            steer_out = -steer_rate_ref if self._steer_invert else steer_rate_ref
            if self._rc_servo_flag == 1.0:
                self._prev_steer_c1 = self._send_speed(self._steer_cid, steer_out) # steer only when servo is up
            else:
                self._prev_steer_c1 = self._send_speed(self._steer_cid, 0.0)
        else:
            steer_rate_ref = 0.0
            self._prev_steer_c1 = 0

        self._finish_tick(now, steer_rate_ref)

    def _update_steer_fault(self) -> None:
        """Latch / clear the steer soft-limit fault from the current angle."""
        steer_deg = np.rad2deg(self._steer_angle)
        if abs(steer_deg) > self._steer_angle_limit_deg:
            if not self._steer_fault:
                self._steer_fault = True
                self._send_cf(self._steer_cid)
                self.get_logger().error(
                    f"STEER FAULT: |{steer_deg:.1f}|° > {self._steer_angle_limit_deg}° "
                    f"— steer motor DISABLED (CF). Manually re-center to re-enable.",
                    throttle_duration_sec=1.0)
        elif self._steer_fault:
            # Recovery band, as a fraction of the limit.  The default
            # 0.01 (0.3° at a 30° limit) is deliberate, NOT a mistyped
            # 0.1: the operating procedure centers the front wheel before
            # the nodes start, so the latched zero is the true mechanical
            # center, and recovery is meant to require the operator to
            # hand-center it again just as precisely.  Verified reachable
            # on the bike — 0.3° is ~14 encoder counts at 16384 counts/rev.
            # steer_test is a bench mode and keeps a wide, forgiving band.
            recovery_threshold = self._steer_angle_limit_deg * (
                self._steer_recovery_ratio if self._mode == "full"
                else _BENCH_RECOVERY_RATIO)
            if abs(steer_deg) < recovery_threshold:
                self._steer_fault = False
                self.get_logger().info(
                    f"STEER RECOVERED: |{steer_deg:.1f}|° within safe range "
                    f"— re-enabled")

    def _finish_tick(self, now: float, steer_rate_ref: float) -> None:
        """Publish /steer_cmd + /new_balance/debug and close out the timing.

        compute_us is stamped last, as before, so the debug array carries the
        previous tick's value.
        """
        self._steer_rate_ref = steer_rate_ref

        msg = Float64MultiArray()
        msg.data = [float(steer_rate_ref), float(self._prev_steer_c1)]
        self._steer_cmd_pub.publish(msg)

        debug = Float64MultiArray()
        debug.data = [
            self._roll_angle, self._roll_rate,
            self._steer_angle, self._forward_speed,
            steer_rate_ref, float(self._prev_steer_c1),
            float(self._sat_outer),
            float(1.0 if self._enabled else 0.0),
            float(self._ctrl.K[0, 0]),
            float(self._ctrl.K[0, 1]),
            float(self._ctrl.K[0, 2]),
            self._timing.get("roll_age_ms", -1.0),
            self._timing.get("steer_age_ms", -1.0),
            self._timing.get("compute_us", -1.0) / 1000.0,
            # Appended, never inserted: bag_tune.py and every recorded bag
            # address fields 0-13 positionally, so renumbering them would
            # silently mis-read old logs.
            float(self._rc_roll_ref_rad),   # /rc_info.data[1], raw as received
            float(self._roll_ref),          # what solve_equilibrium was given
            float(self._x_eq[0]),           # steer angle that sustains the lean
            float(self._x_eq[1]),           # = roll_ref when an equilibrium was solved
            float(self._x_eq[2]),           # always 0 for this model
        ]
        self._debug_pub.publish(debug)

        self._timing["compute_us"] = round((time.time() - now) * 1e6, 0)

    # ==================================================================
    # Status (2 Hz)
    # ==================================================================
    def _publish_status(self) -> None:
        jitter = {}
        if len(self._jitter_samples) >= 10:
            s = np.array(self._jitter_samples)
            jitter = {
                "dt_mean_ms": round(float(s.mean()) * 1000, 2),
                "dt_std_ms": round(float(s.std()) * 1000, 2),
                "dt_min_ms": round(float(s.min()) * 1000, 2),
                "dt_max_ms": round(float(s.max()) * 1000, 2),
            }

        status = {
            "mode": self._mode,
            "enabled": self._enabled,
            "method": self._method if self._mode == "full" else "n/a",
            "control_hz": self._control_hz,
            "gain_every_n_ticks": self._gain_every,
            "gain_update_hz": round(self._control_hz / self._gain_every, 2),
            "gain_sched_v": round(self._gain_sched_v, 3),
            "roll_deg": round(np.rad2deg(self._roll_angle), 2),
            "roll_rate_dps": round(np.rad2deg(self._roll_rate), 2),
            "body_p_dps": round(np.rad2deg(float(self._body_rates[0])), 2),
            "body_q_dps": round(np.rad2deg(float(self._body_rates[1])), 2),
            "body_r_dps": round(np.rad2deg(float(self._body_rates[2])), 2),
            "steer_angle_deg": round(np.rad2deg(self._steer_angle), 2),
            "speed_m_s": round(self._forward_speed, 3),
            # rc_roll_ref_deg is the stick as received; roll_ref_deg is what
            # reached solve_equilibrium.  They diverge whenever no equilibrium
            # was solved, which is the quick way to tell "the stick is dead"
            # from "the stick is fine but the controller is not using it".
            "rc_roll_ref_deg": round(np.rad2deg(self._rc_roll_ref_rad), 2),
            "roll_ref_deg": round(np.rad2deg(self._roll_ref), 2),
            "x_eq_deg": [round(np.rad2deg(float(self._x_eq[0])), 2),
                         round(np.rad2deg(float(self._x_eq[1])), 2),
                         round(np.rad2deg(float(self._x_eq[2])), 2)],
            "steer_rate_ref_rad_s": round(self._steer_rate_ref, 3),
            "steer_c1": self._prev_steer_c1,
            "drive_c1": self._prev_drive_c1,
            "sat_outer": self._sat_outer,
            "steer_fault": self._steer_fault,
            "K": [round(float(k), 4) for k in self._ctrl.K.flatten()],
            "jitter": jitter,
            "timing": self._timing,
        }
        msg = String()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)

    # ==================================================================
    # Cleanup
    # ==================================================================
    def destroy_node(self) -> None:
        self.get_logger().info("Shutting down")
        self._stop_motors()
        if hasattr(self, "_bus") and self._bus is not None:
            self._bus.shutdown()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BalanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
