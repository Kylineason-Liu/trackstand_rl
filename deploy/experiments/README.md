# Standalone motor feedback

`motor_feedback.py` uses Python's standard library and SocketCAN. No ROS nodes,
ROS environment, package build, or `python-can` installation is needed. The CAN
interface must already be configured/up and the motor(s) powered.

From `deploy/`, start the collector **before** the sine test so its first A3
sample captures the stationary starting position:

```bash
python3 scripts/motor_feedback.py --output motor_feedback.csv
# In another terminal:
python3 scripts/steering_sine_position.py --execute
```

The default collector records both motors at a **target 200 Hz per motor**.
To record only steering, use `--motors steering`. `--duration 30` records for
30 seconds; otherwise Ctrl-C stops recording and closes the CSV. The output
file is never overwritten. Console readings update approximately once a second.

Only one-byte **A3 position, A2 velocity, A1 current, and B0 torque-constant read
queries** are sent. No C0/C2/C3/CF, gain/limit writes, or origin reset is sent,
including at shutdown. Run this alongside the sine script, **not** a normal
`motor_io`/`motor_io_position` controller that can overwrite or disable its commands.
Queries add CAN traffic, so inspect the recorded timestamps for actual rates
and gaps; 200 Hz and the sine script's 5 ms spacing are not hard-real-time guarantees.

Defaults mirror `source/trackstand_deploy/config/params.yaml`: steering address
2, encoder/torque signs -1/-1; rear drive address 1, signs -1/-1; both gear ratios
1. These are standalone CLI defaults, not values dynamically loaded from YAML.
Override them using `--steering-id`, `--drive-id`, `--*-gear-ratio`,
`--*-encoder-sign`, and `--*-torque-sign` if the hardware configuration differs.

The rear encoder default was corrected on 2026-10-12 after forward rotation
was measured as negative native A2 velocity/decreasing A3 counts. The same
default applies to `motor_feedback_mit.py` and read-only rear collection in
`steering_sine_mit_io.py`. Rear torque sign remains -1; old CSVs retain the
signs used when they were recorded and must not be reinterpreted as new data.

The same CSV also records outgoing **C2 target counts and timestamps** through
a separate receive-only SocketCAN socket. Local SocketCAN loopback lets this
socket observe the sine script's commands without changing the sine script.
Start the recorder first: it cannot recover commands sent before it started.
Capture is limited to the motors selected with `--motors` and their configured IDs.

Use the `record_type` column to distinguish rows:

- `feedback`: one motor's position, velocity, current and torque measurements.
- `c2_target`: one observed outgoing C2 frame; `c2_target_count` is its signed
  absolute target count, including the sine script's startup offset.

Both row types share `time_unix_s` and `elapsed_s` (seconds since recording
start). For command rows these are **host receipt timestamps**, not the sine
script's send-call timestamps, CAN hardware timestamps, or confirmed motor
execution times. Command rows leave feedback fields unavailable; feedback rows
leave `c2_target_count` blank. No cached target is repeated on feedback rows.
Rows from the two asynchronous sources may not be ordered by timestamp; sort
by time when plotting. Compare `c2_target_count` with `encoder_count` directly,
or subtract the collector's `zero_count` and apply the same encoder sign and
gear conversion to both counts for target-versus-position plots. C2 replies
from the motor are **not** interpreted as target commands. Capture cannot
attribute a command to a particular process; keep the sine script as the only
motor-command writer. Observing a CAN frame is not proof the motor executed it.

Feedback rows contain one motor's fresh position, velocity, current and torque,
plus raw signed multi-turn encoder count, latched startup count, B0 constant
and separate host-receipt elapsed timestamps. Position is startup-relative:
`position_rad` wraps to [-pi, pi), as in `motor_io`; `position_unwrapped_rad`
preserves full revolutions. Positive steering position means right; positive
rear velocity means forward. The collector's startup zero is independent of
the sine script's separately sampled zero; compare raw counts if started at
different times. A1/A2/A3 measurements are sequential, **not simultaneous**.

Torque is `torque_sign * measured_Q_current * B0_constant` in N m. It is a
motor-current-based estimate, not an independent shaft-torque measurement.
B0 is retried once a second until valid; no guessed constant is substituted.
Failed measurement queries produce NaN (raw missing counts are blank), not
cached readings or fabricated zeros. Restart the collector after motor power
resets, since startup counts and torque constants are latched per recording.
