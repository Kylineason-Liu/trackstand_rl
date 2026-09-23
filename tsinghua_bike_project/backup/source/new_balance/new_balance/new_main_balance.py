#!/usr/bin/env python3
"""
Modularized self-balancing bicycle control node — 50 Hz outer / 500 Hz inner.

Reads pre-processed sensor state from topic publishers and outputs CAN
Q-axis current commands (C0) to the steering motor and CAN speed commands
(C1) to the drive motor via python-can.

This version adds a manual inner-loop PD controller (steer velocity → Q-axis
current) on top of the LQR/pole-placement outer loop, replacing the motor's
built-in speed loop to reduce actuator latency (~60 ms → ~10-20 ms).

A ``dry_run`` parameter allows running the full control pipeline without
sending any CAN commands — useful for latency measurement and offline testing.

Architecture (what runs where)
------------------------------
Hardware                Node (ROS 2)           Topic            Consumer
────────────────────────────────────────────────────────────────────────────
YahBoom IMU (/dev/ttyUSB0) → roll_publisher  → /roll_info  ─┐
GIM steer motor (CAN 101/001) ↔ steer_publisher → /steer_info ─┼─ balance_node
GIM drive motor (CAN 102/002) ↔ rear_publisher  → /rear_info  ─┘
SBUS RC receiver (/dev/ttyUSB1) → rc_publisher → /rc_info  ───┘  ([drive_rpm, steer_rad])
                                              balance_node → CAN 101 (C0 steer torque)
                                              balance_node → CAN 102 (C1 drive speed)

Subscribes
----------
  /roll_info  (Float64MultiArray)  — [roll, pitch, yaw, roll_rate, p, q, r, publish_ts]
  /steer_info (Float64MultiArray)  — [steer_angle (rad), steer_rate_raw (rad/s), steer_rate_filtered (rad/s), publish_ts]
  /rear_info  (Float64MultiArray)  — [rear_rate (rad/s), publish_ts]
  /rc_info    (Float64MultiArray)  — [drive_rpm, steer_rad, publish_ts]  (only when vctrl="rc")
  /bike_balance/enable (Bool)      — arm / disarm the controller
  /new_balance/steer_rate_ref (Float64MultiArray) — steer rate ref (inner_loop only)

  The trailing ``publish_ts`` in each topic is a ``time.time()`` float
  appended by the publisher.  It is optional — callbacks use it only when
  the message array is long enough; otherwise the timing fields in status
  show -1.0.

Publishes
---------
  /new_balance/status (String)            JSON status at 2 Hz
  /new_balance/debug (Float64MultiArray)  Debug floats (14 fields) at outer_hz
  /steer_cmd (Float64MultiArray)          [q_current_a, steer_rate_ref_rad_s] at outer_hz

  Status JSON includes a ``timing`` object with per-topic pipeline latencies
  (see the Timing section below).  The debug array appends ``roll_age_ms``,
  ``steer_age_ms``, and ``compute_us`` to the original 11 fields.

Parameters (also see config/params.yaml for full list)
------------------------------------------------------
  mode (str)           : sign_check | inner_loop | full   (default: full)
  method (str)         : pole | pole_distinct | lqr         (default: lqr)
  vctrl (str)          : "fixed" | "rc"                    (default: rc)
  can_interface (str)  : SocketCAN interface              (default: can0)
  steer_can_id (str)   : Host → steer motor CAN ID         (default: 101)
  drive_can_id (str)   : Host → drive motor CAN ID         (default: 102)

  poles_ctr (double[]) : method=pole_distinct only.  3 values [p1, p2, p3]
                         for real poles, or 6 values [re1, im1, re2, im2,
                         re3, im3] for a complex conjugate pair.  Solve them
                         from a gain with k_to_poles.py.  See _decode_poles.

  Dry-run / diagnostics:
    dry_run (bool)        — skip all CAN sends; compute + publish only    (default: false)

  Inner PD parameters:
    inner_kp              — P gain for steer velocity→current (default: 0.0125)
    inner_kd              — D gain, acts on the measured rate  (default: 0.0001)
                            only (derivative on measurement), so a setpoint
                            change never produces a derivative kick
    inner_max_q_current_a — output clamp (A)                   (default: 1.5)


╔══════════════════════════════════════════════════════════════════════════╗
║                  CONTROL ARCHITECTURE — OUTER + INNER PD                 ║
╚══════════════════════════════════════════════════════════════════════════╝

  Outer loop (LQR / pole-placement):
      x = [steer_angle, roll_angle, roll_rate]^T
      u = steer_rate_ref  ←  K · (x_eq − x) + u_eq
      → clipped to ±max_steer_velocity

  Inner PD loop (steer velocity → Q-axis current):
      e      = steer_rate_ref − steer_rate
      I_q    = inner_kp · e − inner_kd · d(steer_rate)/dt
      I_q    = clamp(I_q, ±inner_max_q_current_a)

      The derivative acts on the measured rate, not on e: the setpoint is a
      staircase refreshed at outer_hz, and differentiating it would inject a
      one-tick spike on every step.  See PD.step.
      CAN C0: <0xC0> <4-byte LE int32  I_q in 0.001 A>

  This replaces the motor's built-in speed loop (C1 commands) with a
  user-controlled torque loop (C0 commands).  The PD gains are deliberately
  conservative to avoid overheating or burning the steer motor.


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
    compute_us              µs   control computation (LQR + PD)

  Values of −1.0 indicate missing timestamps (e.g. publisher running old code).

  Debug array (/new_balance/debug) high-frequency extras (indices 11–13):
    [11] roll_age_ms, [12] steer_age_ms, [13] compute_us.


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
║                         MODE 2 — inner_loop                             ║
║          Direct steer-motor Q-axis current control                      ║
╚══════════════════════════════════════════════════════════════════════════╝

  Purpose: Test the steering motor by commanding a target angular velocity
  (rad/s) via the inner PD loop → Q-axis current (C0).  Enforces the steer
  angle limit (±30° by default), and applies an emergency roll stop (>30°).
  The drive motor is kept stopped (C1=0).

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

      Wait for "Steer zero locked: offset=…" message.  This confirms the
      motor is responding to A3 queries.

  3.  Terminal 3: balance node (inner_loop mode)
          ros2 run new_balance balance_node --ros-args -p mode:=inner_loop

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

      Watch "steer_angle_deg" — when it reaches ±30°, the node should force
      it back toward center (you will see a "STEER FAULT" error and the
      steer motor will be CF-disabled until manually re-centered).

  7.  To stop everything, Ctrl-C all terminals in reverse order.

  NOTE: rear_publisher is NOT needed for this mode (drive motor is idle).


╔══════════════════════════════════════════════════════════════════════════╗
║                         MODE 3a — full (fixed speed)                     ║
║  Complete dual-loop self-balancing, inner PD, fixed target speed        ║
╚══════════════════════════════════════════════════════════════════════════╝

  Purpose: Run the full balance controller with a fixed target speed.
  The rear wheel ramps up to target_speed over acceleration_time and
  holds constant.

  Prerequisites: same as inner_loop.

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
        "sat_outer"         — true if outer-loop output is being clamped
        "steer_fault"       — true if steer angle exceeded limit (motor CF-disabled)
        "inner_pd"          — {kp, kd, output_min, output_max, output}  inner PD diagnostics
        "steer_q_current_a" — Q-axis current sent to steer motor (A)
        "K"                 — current feedback gains (gain-scheduled)
        "timing"            — {roll_pub_to_cb_ms, roll_age_ms, compute_us, …}  see Timing section

  7.  Emergency stop (any terminal):
          ros2 topic pub /bike_balance/enable std_msgs/Bool "data: false"

      The controller auto-disarms if:
        - |roll| > max_roll_angle_deg (30°)
        - steer or roll topic goes stale (>0.5 s)
        - |steer_angle| > steer_angle_limit_deg (30°) → steer motor CF-disabled,
          auto-recovers when angle returns below 5% of the limit

  8.  To stop everything, Ctrl-C all terminals in reverse order.

  Dry-run (hardware-free) alternative:
      Add ``-p dry_run:=true`` to skip all CAN output.  This lets you run the
      full control pipeline and inspect computed outputs + timing without any
      risk of motor movement.  CAN bus must still be ``up`` so the
      ``can.interface.Bus`` constructor succeeds, but no frames are sent.


╔══════════════════════════════════════════════════════════════════════════╗
║                         MODE 3b — full (RC speed + steer)               ║
║  Complete dual-loop self-balancing, inner PD, RC speed & steer         ║
╚══════════════════════════════════════════════════════════════════════════╝

  Purpose: Same as 3a, but the target forward speed AND target steering
  angle are both controlled in real time by the RC transmitter sticks.
  /rc_info now carries two floats: [drive_rpm, steer_rad].

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
      Should show [rpm, steer_rad, publish_ts] — channel 2 → drive RPM
      (centre ≈ 0), channel 0 → target steer angle in radians (±30° range).

  5.  Terminal 5: balance node (full mode, RC speed + steer)
          ros2 run new_balance balance_node --ros-args -p mode:=full -p vctrl:=rc -p method:=pole

  6.  Terminal 6: enable balancing
          ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"

      The bike will attempt to self-balance.  Speed is controlled by the
      throttle stick via drive RPM: centre = 0 RPM, full forward ≈
      +rc_max_rpm (500 RPM), full back ≈ −rc_max_rpm.
      Steering angle is controlled by the steer stick: the RC steer_rad
      is used as the equilibrium reference — solve_equilibrium(steer_rad)
      produces [steer_eq, roll_eq, 0] as the target state for the
      step_with_state_reference control law.

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

import threading

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

# Steer-fault recovery band for inner_loop (bench) mode, as a fraction of
# steer_angle_limit_deg.  full mode uses the steer_recovery_ratio parameter.
_BENCH_RECOVERY_RATIO = 0.5

# Inner-loop pacing: sleep until this close to the deadline, then spin.
# Wide enough to absorb nanosleep wake-up jitter, narrow enough that the
# thread is not pinned to a core.
_SPIN_GUARD_S = 0.0003


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
# Real-time clock thread — isolated from ROS2 executor jitter
# ======================================================================
class TimeKeeper:
    """Background thread that updates a monotonic timestamp at ~1 kHz.

    All ``BalanceNode`` ``time.time()`` calls read from ``TimeKeeper.now``
    instead of calling ``time.time()`` directly.  This avoids the ROS2
    multi-threaded executor's scheduling latency from contaminating
    per-tick timing measurements.
    """

    def __init__(self) -> None:
        self._now: float = time.time()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def now(self) -> float:
        with self._lock:
            return self._now

    def _run(self) -> None:
        while self._running:
            self._now = time.time()
            time.sleep(0.0001)  # ~10 kHz update rate

    def stop(self) -> None:
        self._running = False


# ======================================================================
# Simple PD controller (no integral — safe for motor torque mode)
# ======================================================================
class PD:
    """Discrete PD controller with output clamping.

    No integral term — avoids windup and overheating of the steer motor
    when operating in Q-axis current (C0) torque mode.
    """

    def __init__(
        self,
        kp: float = 0.0,
        kd: float = 0.0,
        output_min: float = -float("inf"),
        output_max: float = float("inf"),
    ) -> None:
        self.kp = float(kp)
        self.kd = float(kd)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.reset()

    def reset(self) -> None:
        """Clear derivative state.

        The next ``step`` emits no derivative term and seeds its history
        from that call's own arguments.  reset() runs at arming, at fault
        latch and at fault recovery — seeding the history with 0.0 instead
        would make the following tick differentiate against a fictitious
        zero and fire a one-tick current impulse at exactly those moments.
        """
        self._prev_meas: float = 0.0
        self._primed: bool = False
        self._last_output: float = 0.0

    def step(self, setpoint: float, measurement: float, dt: float) -> float:
        """Evaluate one PD step.  Returns clamped output.

        The derivative acts on ``measurement`` alone, never on the error.
        The outer loop refreshes the setpoint at ``outer_hz`` while this
        runs at ``inner_hz``, so the setpoint reaching it is a staircase;
        differentiating the error would turn each step of that staircase
        into a one-tick spike of ``kd·Δsetpoint/dt`` that reflects nothing
        the plant did.  For a constant setpoint the two forms are
        identical.  Here ``measurement`` is steer angular velocity, so its
        derivative is angular acceleration — damping in the physical sense.

        Taking the setpoint rather than a pre-computed error keeps the two
        arguments from ever disagreeing about which sample they describe.
        """
        # Proportional
        p = self.kp * (setpoint - measurement)

        # Derivative on measurement (negated: error = setpoint − measurement)
        if self._primed and dt > 1e-9:
            d = -self.kd * (measurement - self._prev_meas) / dt
        else:
            d = 0.0
        self._prev_meas = measurement
        self._primed = True

        output = p + d

        # Clamp
        if output > self.output_max:
            output = float(self.output_max)
        elif output < self.output_min:
            output = float(self.output_min)

        self._last_output = output
        return output

    @property
    def last_output(self) -> float:
        return self._last_output


# ======================================================================
# Main node
# ======================================================================
class BalanceNode(Node):
    """Self-balancing controller with inner PD — reads topics, writes CAN.

    Outer state-feedback loop runs on a ROS timer at ``outer_hz``; the inner
    steer-velocity PD runs in its own thread at ``inner_hz``.
    """

    def __init__(self) -> None:
        super().__init__("new_balance_node")

        self._tk = TimeKeeper()

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

        # Control
        self.declare_parameter("control_dt", 0.005)
        self.declare_parameter("min_scheduling_speed", 1.0)
        self.declare_parameter("max_steer_velocity", 10.0)

        # Frequency decoupling: outer LQR vs inner PD
        self.declare_parameter("outer_hz", 50.0)
        self.declare_parameter("inner_hz", 500.0)

        # Pole
        self.declare_parameter("pole_wc", -15.0)
        # 3 values [p1, p2, p3] for real poles, or 6 values
        # [re1, im1, re2, im2, re3, im3] to carry a complex conjugate pair.
        # See _decode_poles, and k_to_poles.py for solving these from a gain.
        #
        # Solved from K = [3.97165, 45.47517, 3.13077] at v = 1 m/s with the
        # bike_* defaults below: one real pole at -7.9865 plus the pair
        # -3.0713 +/- 9.2655j (wn 9.76 rad/s, zeta 0.315).  Re-solve with
        # k_to_poles.py if any bike_* value changes.
        self.declare_parameter(
            "poles_ctr",
            [-7.986503, 0.0, -3.071325, -9.265501, -3.071325, 9.265501])

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
        self.declare_parameter("enable_on_start", False)
        self.declare_parameter("balance_start_delay", 0.1)

        # Dry run: compute + publish without CAN output
        self.declare_parameter("dry_run", False)

        # Inner PD (steer velocity → Q-axis current)
        self.declare_parameter("inner_kp", 0.30)
        self.declare_parameter("inner_kd", 0.02)
        self.declare_parameter("inner_max_q_current_a", 1.5)

        # ════════════════ Read parameters ════════════════
        self._mode: str = self.get_parameter("mode").value
        if self._mode not in ("sign_check", "inner_loop", "full"):
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
        self._control_dt: float = self.get_parameter("control_dt").value
        self._outer_hz: float = self.get_parameter("outer_hz").value
        self._inner_hz: float = self.get_parameter("inner_hz").value
        self._outer_dt: float = 1.0 / max(self._outer_hz, 1.0)
        self._inner_dt: float = 1.0 / max(self._inner_hz, 1.0)
        self._min_speed: float = self.get_parameter("min_scheduling_speed").value
        self._max_steer_vel: float = self.get_parameter("max_steer_velocity").value
        self._pole_wc: float = self.get_parameter("pole_wc").value
        self._max_roll_rad: float = np.deg2rad(
            self.get_parameter("max_roll_angle_deg").value)
        self._steer_angle_limit_deg: float = self.get_parameter("steer_angle_limit_deg").value
        self._steer_recovery_ratio: float = self.get_parameter("steer_recovery_ratio").value
        self._enable_on_start: bool = self.get_parameter("enable_on_start").value
        self._balance_delay: float = self.get_parameter("balance_start_delay").value
        self._inner_kp: float = self.get_parameter("inner_kp").value
        self._inner_kd: float = self.get_parameter("inner_kd").value
        self._inner_max_q: float = self.get_parameter("inner_max_q_current_a").value
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
        self._bike = LinearBicycleModel(
            dt=self._control_dt, parameters=bike_params,
            track_roll=False, min_forward_speed=self._min_speed,
        )
        self._build_controller()

        # ════════════════ Inner PD controller ════════════════
        self._pd_inner = PD(
            kp=self._inner_kp,
            kd=self._inner_kd,
            output_min=-self._inner_max_q,
            output_max=self._inner_max_q,
        )

        # ════════════════ Runtime state ════════════════
        self._roll_angle: float = 0.0
        self._roll_rate: float = 0.0
        self._body_rates: np.ndarray = np.array([0.0, 0.0, 0.0])
        self._steer_angle: float = 0.0
        self._steer_rate: float = 0.0
        self._steer_rate_filt: float = 0.0
        self._rear_rate: float = 0.0

        self._roll_ts: float = 0.0
        self._steer_ts: float = 0.0
        self._rear_ts: float = 0.0

        self._enabled: bool = self._enable_on_start
        self._start_time: float | None = None
        self._steer_rate_ref: float = 0.0    # current target, written by _tick_outer, read by _inner_loop
        self._prev_steer_rate_ref: float = 0.0
        self._prev_q_current: float = 0.0
        self._prev_drive_c1: int = 0
        self._sat_outer: bool = False
        self._steer_fault: bool = False     # 转向角超限 → CF 失能，需手动归零后自动恢复
        self._pd_needs_reset: bool = False  # set by _enable_cb, consumed by _inner_loop
        self._forward_speed: float = 0.0
        self._rc_drive_rpm: float = 0.0
        self._rc_steer_rad: float = 0.0
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

        if self._mode == "inner_loop":
            self.create_subscription(
                Float64MultiArray, "new_balance/steer_rate_ref",
                self._steer_ref_cb, 10)

        # ════════════════ Publishers ════════════════
        self._status_pub = self.create_publisher(String, "new_balance/status", 10)
        self._debug_pub = self.create_publisher(Float64MultiArray, "new_balance/debug", 10)
        self._steer_cmd_pub = self.create_publisher(Float64MultiArray, "steer_cmd", 10)
        self._servo_cmd_pub = self.create_publisher(String, 'servo_cmd', 10)

        # ════════════════ Timers ════════════════
        if self._mode == "sign_check":
            self._timer = self.create_timer(self._control_dt, self._tick)
        else:
            self._outer_timer = self.create_timer(self._outer_dt, self._tick_outer)
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._last_tick = self._tk.now

        # ════════════════ Inner loop thread ════════════════
        self._inner_running: bool = False
        self._inner_thread: threading.Thread | None = None
        if self._mode != "sign_check":
            self._inner_running = True
            self._inner_thread = threading.Thread(target=self._inner_loop, daemon=True)
            self._inner_thread.start()

        self.get_logger().info(
            f"NewBalance (inner PD) | mode={self._mode} method={self._method} dt={self._control_dt}s | "
            f"outer={self._outer_hz:.0f}Hz inner={self._inner_hz:.0f}Hz | "
            f"CAN={self._can_iface} steer=0x{self._steer_cid:03X} drive=0x{self._drive_cid:03X} | "
            f"target_v={self._target_speed}m/s vctrl={self._vctrl} "
            f"rc_max_rpm={self._rc_max_rpm} enabled={self._enabled} | "
            f"inner_pd: kp={self._inner_kp} kd={self._inner_kd} max_q={self._inner_max_q}A")

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
            self._roll_cb_ts = self._tk.now
            self._roll_angle = float(msg.data[0])
            self._roll_rate = float(msg.data[3])
            self._body_rates = np.array([
                float(msg.data[4]), float(msg.data[5]), float(msg.data[6]),
            ])
            self._roll_ts = self._roll_cb_ts
            if len(msg.data) >= 8:
                self._roll_pub_ts = float(msg.data[7])

    def _steer_cb(self, msg: Float64MultiArray) -> None:
        """[angle, rate_raw, rate_filtered, publish_ts].

        Both control modes feed back rate_filtered; rate_raw is kept for
        diagnostics only (status JSON).
        """
        if len(msg.data) >= 2:
            self._steer_cb_ts = self._tk.now
            self._steer_angle = float(msg.data[0])
            self._steer_rate = float(msg.data[1])
            self._steer_ts = self._steer_cb_ts
            if len(msg.data) >= 3:
                self._steer_rate_filt = float(msg.data[2])
            if len(msg.data) >= 4:
                self._steer_pub_ts = float(msg.data[3])

    def _rear_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 1:
            self._rear_cb_ts = self._tk.now
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
            self._steer_fault = False        # 重新使能时清除转向故障
            # 只置标志，由内环线程自己 reset —— 本回调跑在 executor 线程，
            # 直接调 reset() 会和内环线程的 step() 撞在一起。
            self._pd_needs_reset = True
            self.get_logger().info("BALANCE ENABLED")
        elif not self._enabled and was:
            self._stop_motors()
            self.get_logger().info("BALANCE DISABLED — motors stopped")

    def _steer_ref_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 1:
            self._ext_steer_rate_ref = float(msg.data[0])

    def _rc_cb(self, msg: Float64MultiArray) -> None:
        """Store RC steer angle, drive RPM, and servo-up flag from /rc_info.

        Layout: [drive_rpm, steer_rad, servo_up, publish_ts].  RPM clamped
        to ±rc_max_rpm.
        """
        self._rc_cb_ts = self._tk.now
        if len(msg.data) >= 2:
            raw_rpm = float(msg.data[0])
            self._rc_drive_rpm = max(-self._rc_max_rpm, min(raw_rpm, self._rc_max_rpm))
            self._rc_steer_rad = float(msg.data[1])
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
        """Triggered once at start of tick_full and whenever /rc_info.data[2] (_servo_up) transitions to 0."""
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
    def _send_speed(self, can_id: int, rad_s: float) -> None:
        """Send C1 speed command via python-can.

        C1 frame: <0xC1> <4-byte LE int32 speed in rpm×100>
        """
        if self._dry_run:
            return
        c1 = _rad_s_to_c1(rad_s)
        data = bytes([0xC1]) + struct.pack("<i", c1)
        try:
            self._bus.send(can.Message(
                arbitration_id=can_id, data=data, is_extended_id=False))
        except can.CanError as exc:
            self.get_logger().warn(
                f"CAN send failed: 0x{can_id:03X} C1={c1}: {exc}",
                throttle_duration_sec=2.0)

    def _send_q_current(self, can_id: int, current_a: float) -> None:
        """Send C0 Q-axis current command.

        C0 frame: <0xC0> <4-byte LE int32 current in 0.001 A>
        The motor firmware converts: τ = Kt × I_q internally.
        """
        if self._dry_run:
            return
        raw = int(round(current_a * 1000.0))  # A → 0.001 A
        data = bytes([0xC0]) + struct.pack("<i", raw)
        try:
            self._bus.send(can.Message(
                arbitration_id=can_id, data=data, is_extended_id=False))
        except can.CanError as exc:
            self.get_logger().warn(
                f"CAN send failed: 0x{can_id:03X} C0={raw}: {exc}",
                throttle_duration_sec=2.0)

    def _stop_motors(self) -> None:
        """Send C1=0 + CF (motor disable) to both motors.
        Also sends C0=0 to steer motor for a clean stop.
        """
        if self._dry_run:
            return
        self._send_q_current(self._steer_cid, 0.0)
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

        Must be called from within _tick_outer / _tick_sign_check after all
        subscriber callbacks for this cycle are complete.
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
    # Main tick dispatcher — sign_check mode only.
    # full and inner_loop modes use _tick_outer (timer) + _inner_loop (thread).
    # ==================================================================
    def _tick(self) -> None:
        now = self._tk.now
        if self._last_tick > 0:
            measured_dt = now - self._last_tick
            self._jitter_samples.append(measured_dt)
            if len(self._jitter_samples) > 250:
                self._jitter_samples.pop(0)
        self._last_tick = now
        if self._start_time is None:
            self._start_time = now
        self._tick_sign_check(now)

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
    # Outer loop timer (50 Hz default) — full mode
    # ==================================================================
    def _tick_outer(self) -> None:
        """LQR outer loop + gain scheduling + safety + drive C1.

        Runs on a ROS2 timer at ``outer_hz``.  The inner PD loop runs
        independently in ``_inner_loop`` at ``inner_hz``, reading
        ``self._steer_rate_ref`` and publishing ``steer_cmd``.
        """
        now = self._tk.now
        if self._last_tick > 0:
            measured_dt = now - self._last_tick
            self._jitter_samples.append(measured_dt)
            if len(self._jitter_samples) > 250:
                self._jitter_samples.pop(0)
        self._last_tick = now
        if self._start_time is None:
            self._start_time = now

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
        if self._steer_ts == 0.0 or now - self._steer_ts > 0.5:
            if self._enabled:
                self.get_logger().warn("Steer data stale — disabling", throttle_duration_sec=2.0)
                self._enabled = False
                self._stop_motors()
            return
        if self._roll_ts == 0.0 or now - self._roll_ts > 0.5:
            if self._enabled:
                self.get_logger().warn("Roll data stale — disabling", throttle_duration_sec=2.0)
                self._enabled = False
                self._stop_motors()
            return
        if self._rear_ts > 0 and now - self._rear_ts > 0.5:
            self.get_logger().warn("Rear data stale", throttle_duration_sec=2.0)

        # ---- forward speed ----
        self._forward_speed = self._wheel_radius * abs(self._rear_rate)

        # ---- speed profile ----
        if self._vctrl == "rc":
            drive_rpm = max(-self._rc_max_rpm,
                            min(self._rc_max_rpm, self._rc_drive_rpm))
            rear_rate_ref = drive_rpm * (2.0 * math.pi / 60.0)
        else:
            if self._accel_time > 0:
                speed_ref = min(elapsed / self._accel_time * self._target_speed, self._target_speed)
            else:
                speed_ref = self._target_speed
            rear_rate_ref = speed_ref / self._wheel_radius

        if self._mode == "full":
            # ---- state ----
            state = np.array([[self._steer_angle], [self._roll_angle], [self._roll_rate]])
            corrected_roll = self._roll_angle

            # ---- gain scheduling ----
            sch_v = max(self._forward_speed, self._min_speed)
            self._bike.update_system_parameters(sch_v, min_forward_speed=self._min_speed)
            # Push the rescheduled matrices into the controller.  update_gain()
            # alone would recompute K from the copies taken at construction
            # time, freezing both K and the equilibrium point at the model's
            # default 2 m/s.
            self._ctrl.update_system_and_gain(self._bike.sys)

            # ---- outer loop (LQR) ----
            if elapsed < self._balance_delay or not self._enabled:
                steer_rate_ref = 0.0
                self._sat_outer = False
            else:
                steer_ref = -self._rc_steer_rad if self._vctrl == "rc" else 0.0
                eq = self._ctrl.solve_equilibrium(steer_ref)
                x_eq = eq.state
                u_raw = float(self._ctrl.step_with_state_reference(x_eq, state)[0, 0])
                steer_rate_ref = float(np.clip(u_raw, -self._max_steer_vel, self._max_steer_vel))
                self._sat_outer = abs(u_raw - steer_rate_ref) > 1e-6
        else:
            # inner_loop mode: no outer LQR, just safety checks below
            steer_rate_ref = 0.0
            rear_rate_ref = 0.0
            corrected_roll = self._roll_angle
            self._sat_outer = False

        # ---- steer angle limit ----
        # Detection and recovery live in _inner_loop (all modes), which runs
        # at inner_hz — one owner, and one inner period of latency instead of
        # one outer period.  This tick only reacts to the latched flag.
        if self._steer_fault:
            if self._enabled:
                self._send_speed(self._drive_cid,
                                 -rear_rate_ref if self._drive_invert else rear_rate_ref)
            else:
                self._send_speed(self._drive_cid, 0.0)
            steer_rate_ref = 0.0
            self._steer_rate_ref = 0.0
            self._prev_steer_rate_ref = 0.0
            self._prev_q_current = 0.0
            self._prev_drive_c1 = _rad_s_to_c1(rear_rate_ref)
            self._timing["compute_us"] = round((self._tk.now - now) * 1e6, 0)
            return

        # ---- emergency roll stop ----
        if abs(corrected_roll) > self._max_roll_rad:
            if self._enabled:
                self.get_logger().error(
                    f"EMERGENCY: |roll|={np.rad2deg(corrected_roll):.1f}°")
                self._enabled = False
                self._stop_motors()
            self._timing["compute_us"] = round((self._tk.now - now) * 1e6, 0)
            return

        # ---- drive motor C1 ----
        if self._enabled:
            self._send_speed(self._drive_cid,
                             -rear_rate_ref if self._drive_invert else rear_rate_ref)
        else:
            self._send_speed(self._drive_cid, 0.0)

        # ---- publish steer_cmd (reads _inner_loop output) ----
        msg = Float64MultiArray()
        msg.data = [float(self._prev_q_current), float(steer_rate_ref)]
        self._steer_cmd_pub.publish(msg)

        # ---- publish debug ----
        debug = Float64MultiArray()
        debug.data = [
            self._roll_angle, self._roll_rate,
            self._steer_angle, self._steer_rate_filt,
            self._forward_speed, steer_rate_ref,
            self._prev_q_current,
            float(self._sat_outer),
            float(1.0 if self._enabled else 0.0),
            float(self._ctrl.K[0, 0]) if self._mode == "full" else 0.0,
            float(self._ctrl.K[0, 1]) if self._mode == "full" else 0.0,
            float(self._ctrl.K[0, 2]) if self._mode == "full" else 0.0,
            self._timing.get("roll_age_ms", -1),
            self._timing.get("steer_age_ms", -1),
            self._timing.get("compute_us", -1)/1000.0,
        ]
        self._debug_pub.publish(debug)

        self._steer_rate_ref = steer_rate_ref
        self._prev_steer_rate_ref = steer_rate_ref
        self._prev_drive_c1 = _rad_s_to_c1(rear_rate_ref)
        self._timing["compute_us"] = round((self._tk.now - now) * 1e6, 0)

    # ==================================================================
    # Inner loop thread (500 Hz default)
    # ==================================================================
    def _inner_loop(self) -> None:
        """Inner PD loop at ``inner_hz`` — dedicated thread with busy-wait timing.

        Reads ``_steer_rate_ref`` (set by ``_tick_outer``) in full mode,
        or ``_ext_steer_rate_ref`` (set by ``/new_balance/steer_rate_ref``
        subscriber) in inner_loop mode.  Sends C0 Q-axis current to the
        steer motor.
        """
        period = self._inner_dt
        next_tick = time.perf_counter()
        overruns = 0

        while self._inner_running:
            # ── Consume reset request from _tick_outer ──
            if self._pd_needs_reset:
                self._pd_inner.reset()
                self._pd_needs_reset = False

            # ── Safety: steer angle limit (all modes) ──
            # This thread is the only place that latches and clears the fault
            # on an angle threshold, so the two mode-dependent recovery bands
            # can never race; _tick_outer only reads the flag.  (_enable_cb
            # also clears it on arm — if the angle is still out of range this
            # loop re-faults on the next tick.)
            #
            # Detecting here rather than in the outer tick bounds the travel
            # past the limit by one inner period instead of one outer period
            # — at max_steer_velocity that is degrees, not tens of degrees.
            steer_deg = np.rad2deg(self._steer_angle)
            if abs(steer_deg) > self._steer_angle_limit_deg:
                if not self._steer_fault:
                    self._steer_fault = True
                    self._pd_inner.reset()
                    self._send_cf(self._steer_cid)
                    self.get_logger().error(
                        f"STEER FAULT: |{steer_deg:.1f}|° > {self._steer_angle_limit_deg}° "
                        f"— steer motor DISABLED (CF). Manually re-center to re-enable.",
                        throttle_duration_sec=1.0)
                    self._prev_q_current = 0.0
            elif self._steer_fault:
                # Recovery band, as a fraction of the limit.  The default
                # 0.01 (0.3° at a 30° limit) is deliberate, NOT a mistyped
                # 0.1: the operating procedure centers the front wheel before
                # the nodes start, so the latched zero is the true mechanical
                # center, and recovery is meant to require the operator to
                # hand-center it again just as precisely.  Verified reachable
                # on the bike — 0.3° is ~14 encoder counts at 16384 counts/rev.
                # inner_loop is a bench mode and keeps a wide, forgiving band.
                recovery_threshold = self._steer_angle_limit_deg * (
                    self._steer_recovery_ratio if self._mode == "full"
                    else _BENCH_RECOVERY_RATIO)
                if abs(steer_deg) < recovery_threshold:
                    self._steer_fault = False
                    self._pd_inner.reset()
                    self.get_logger().info(
                        f"STEER RECOVERED: |{steer_deg:.1f}|° within safe range "
                        f"— re-enabled")

            # ── Safety: roll emergency (all modes) ──
            # _tick_outer keeps its own roll check as a backstop; both do the
            # same idempotent thing, so whichever notices first wins.
            if self._roll_ts > 0 and abs(self._roll_angle) > self._max_roll_rad:
                if self._enabled:
                    self.get_logger().error(
                        f"INNER STOP: |roll|={np.rad2deg(self._roll_angle):.1f}°")
                    self._enabled = False
                    self._stop_motors()

            # ── Inner PD ──
            if not self._enabled or self._steer_fault:
                self._send_q_current(self._steer_cid, 0.0)
            elif self._mode == "full":
                q = self._pd_inner.step(self._steer_rate_ref,
                                        self._steer_rate_filt, self._inner_dt)
                current = -q if self._steer_invert else q
                self._send_q_current(self._steer_cid, current)
                self._prev_q_current = q
            elif self._mode == "inner_loop":
                q = self._pd_inner.step(self._ext_steer_rate_ref,
                                        self._steer_rate_filt, self._inner_dt)
                current = -q if self._steer_invert else q
                self._send_q_current(self._steer_cid, current)
                self._prev_q_current = q

            # ── Precise timing ──
            # Sleep for the bulk of the period, spin only the last
            # _SPIN_GUARD_S.  A fixed 1 ms sleep threshold could never be
            # met at the default inner_hz=1000 — ``remain`` is at most one
            # period — so the sleep never ran and this pinned a core,
            # holding the GIL against the executor thread between sends.
            next_tick += period
            remain = next_tick - time.perf_counter()
            if remain <= 0.0:
                # Overran the period: resync rather than free-run trying to
                # catch up, which would burst C0 frames onto the bus.
                overruns += 1
                self.get_logger().warn(
                    f"Inner loop overrun ({overruns} total) — "
                    f"inner_hz={self._inner_hz:.0f} may be too high",
                    throttle_duration_sec=5.0)
                next_tick = time.perf_counter()
                continue
            if remain > _SPIN_GUARD_S:
                time.sleep(remain - _SPIN_GUARD_S)
            while time.perf_counter() < next_tick:
                pass

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
            "roll_deg": round(np.rad2deg(self._roll_angle), 2),
            "roll_rate_dps": round(np.rad2deg(self._roll_rate), 2),
            "body_p_dps": round(np.rad2deg(float(self._body_rates[0])), 2),
            "body_q_dps": round(np.rad2deg(float(self._body_rates[1])), 2),
            "body_r_dps": round(np.rad2deg(float(self._body_rates[2])), 2),
            "steer_angle_deg": round(np.rad2deg(self._steer_angle), 2),
            "steer_rate_dps": round(np.rad2deg(self._steer_rate), 2),
            "speed_m_s": round(self._forward_speed, 3),
            "steer_rate_ref_rad_s": round(self._prev_steer_rate_ref, 3),
            "steer_q_current_a": round(self._prev_q_current, 4),
            "drive_c1": self._prev_drive_c1,
            "sat_outer": self._sat_outer,
            "steer_fault": self._steer_fault,
            "inner_pd": {
                "kp": self._pd_inner.kp,
                "kd": self._pd_inner.kd,
                "output_min": self._pd_inner.output_min,
                "output_max": self._pd_inner.output_max,
                "output": round(self._prev_q_current, 4),
            },
            "K": [round(float(k), 4) for k in self._ctrl.K.flatten()],
            "jitter": jitter,
            "timing": self._timing,
        }
        msg = String()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)

        debug = Float64MultiArray()
        debug.data = [
            self._roll_angle, self._roll_rate,
            self._steer_angle, self._steer_rate_filt,
            self._forward_speed, self._prev_steer_rate_ref,
            self._prev_q_current,
            float(self._sat_outer),
            float(1.0 if self._enabled else 0.0),
            float(self._ctrl.K[0, 0]),
            float(self._ctrl.K[0, 1]),
            float(self._ctrl.K[0, 2]),
            self._timing.get("roll_age_ms", -1),
            self._timing.get("steer_age_ms", -1),
            self._timing.get("compute_us", -1)/1000.0,
        ]
        self._debug_pub.publish(debug)

    # ==================================================================
    # Cleanup
    # ==================================================================
    def destroy_node(self) -> None:
        self.get_logger().info("Shutting down")
        self._inner_running = False
        if self._inner_thread is not None:
            self._inner_thread.join(timeout=2.0)
        self._stop_motors()
        if hasattr(self, "_bus") and self._bus is not None:
            self._bus.shutdown()
        self._tk.stop()
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
