# MIT steering sine experiment

`steering_sine_mit.py` uses standard-library SocketCAN directly. Run it from
`deploy/experiments/`; no ROS environment or package build is needed.

```bash
python3 steering_sine_mit.py --preview --duration 0.05
python3 steering_sine_mit.py --execute --amplitude 1 --period 10 --duration 30
# Example: +/-0.1 rad at 2 Hz for 30 seconds:
python3 steering_sine_mit.py --execute --amplitude 0.1 --period 0.5 --duration 30
```

Defaults are `can0`, steering address `2` (MIT TX `0x502`), joint-to-motor
gear ratio `1`, sign `-1`, peak amplitude `1 rad`, period `10 s`, and nominal
command rate `200 Hz`. Duration `0` means run until Ctrl-C. The sine and its
direction match `steering_sine_position.py`. `--help` lists all options.
Every execution sets the persistent MIT position encoding range to **+/-3 rad**,
preserving the motor's current velocity and torque encoding ranges.
Run it as the only motor command writer; passive feedback recording can run
alongside it. Execution is also the default when neither mode flag is supplied.

## Absolute position and units

MIT commands specify a position in the motor's native coordinate system.
Position steering at the desired sine centre and keep it stationary before
starting; the script cannot identify physical straight-ahead alignment.
Every execution performs this startup sequence before sending any MIT motion:

```text
CF -> verify disabled/fault-free acknowledgement
   -> B1 -> wait for three-byte acknowledgement
   -> read F0 -> write Pos_Max=3 rad, preserving velocity/torque bytes
   -> verify write acknowledgement -> independently read back all F0 fields
   -> verify A3 zero -> verify F1 near stationary zero and fault-free/inactive status
   -> validate full waveform -> MIT sine at nominal 200 Hz
```

With velocity/torque ranges of +/-45 rad/s and +/-18 Nm, the F0 write is
`102#F01E0094110807`. Other actual velocity/torque settings are preserved,
not replaced with these defaults. Both B1's single-turn origin and F0's ranges
persist across power cycles. Startup failures stop before MIT motion; settings
already acknowledged are not rolled back. Preview opens no CAN socket and
does not reset the origin or write F0. Every target is computed independently:

```text
absolute_motor_target_rad = startup_motor_rad
    + sign * gear_ratio * amplitude_rad * sin(2*pi*elapsed_s/period_s)
```

The startup origin is neither sign-inverted nor gear-scaled. A3 independently
checks that the single-turn count (allowing wraparound) and signed multi-turn
count are each within two counts of zero. F1 position must be within two A3
count increments plus one F1 position increment of zero; F1 speed must be
within 0.1 rad/s plus one F1 velocity increment of zero. F1's fault and MIT-active
bits must both be clear. Reported torque is printed, not used as a startup
pass/fail criterion. These are bench preflight tolerances, not proof of calibration
or healthy mechanics. Only after these checks is the first F1 position latched
as `startup_motor_rad`. F1 uses the same position encoding as MIT targets.
With +/-3 rad the position increment is approximately 0.00009155 rad (0.005246 deg).
The complete waveform, including its latched offset and gear ratio, must fit
inside +/-3 rad; an excessive amplitude fails before motion rather than clipping.

Only position changes. Requested target velocity is always `0 rad/s`, requested
feed-forward torque is always `0 Nm`, Kp is `10 Nm/rad`, and Kd is
`0.2 Nm*s/rad` in **motor coordinates**. Gains are fixed constants near the top
of the script; gear-ratio overrides do not rescale them.

## Encoding and exit behavior

MIT packs position into 16 bits and velocity, Kp, Kd, and torque into 12 bits
each. It uses high-bit-first fields, has no opcode byte, and uses DLC 8.
Gains use nonnegative encoding; the erroneous vendor gain encoder is not used.
Actual F0 values and exact read-back verification are required; missing/invalid
replies never fall back to defaults. Only the F0 position range is changed;
B0, firmware gains, speed/current limits and the velocity/torque encoding ranges
are not changed. The motor's configured torque constant is used internally by MIT.
Queued replies are discarded before each startup request, with a bounded drain
that aborts on a continuously full queue. A matching F0 write acknowledgement is
required. The protocol has no request IDs, so in-flight replies from concurrent
querying processes still cannot be uniquely attributed; avoid manual queries
during startup and stop all other motor command writers.

Exact zero is unavailable in the symmetric 12-bit encoding. Requested zero
velocity/torque use nearest code 2048. With the documented F0 defaults, they
decode to approximately `+0.010989 rad/s` and `+0.004396 Nm`; Kp/Kd decode to
`10.012210` and `0.200244`. Startup output shows these values using the motor's
actual ranges. Preview uses +/-3 rad position, labels its assumed default
velocity/torque ranges, and uses a zero origin.

The first motion frame switches the motor into MIT mode. Normal completion,
Ctrl-C, SIGTERM, and exceptions after a motion attempt send one CF disable
command to steering address `0x102`; shutdown does not hold the final target.
The script has no continuous feedback watchdog. CF transmission is reported,
but no shutdown acknowledgement is awaited (startup does await CF's reply).
A CF send failure returns an error.

Use [`motor_feedback_mit.py`](motor_feedback_mit.py) to record measurements,
MIT targets, gains and F1 state. Start it before the sender and stop it after
the sender. It records B1 request/acknowledgement transitions in separate
reference epochs and reacquires its zeros without mixing measurements across
the reset. Recorder queries remain
paused through F0/A3/F1 preflight until the first MIT target. It observes the
sender's +/-3 rad F0 write and pre-motion A3/F1 zeros; after motion it resumes
read-only polling and continues after the exit CF.
See [MIT recording instructions](MOTOR_FEEDBACK_MIT.md) for commands and CSV fields.
The original `motor_feedback.py` captures C2 targets only. The current C2-specific
plot/analysis script does not yet accept the new MIT recording format.

## Verification

Run the offline transport/encoding tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s deploy/experiments/tests -p test_steering_sine_mit.py -v
```

Protocol source: local GIM CAN V3.09b0, pp.5–6 (A3), p.7 (status fields),
p.8 (B1), pp.20–22 (F0, F1, MIT command and control diagram), and p.16 (CF).
Tests simulate CAN; they do not establish
physical MIT direction, response bandwidth, or motor firmware loop frequency.
