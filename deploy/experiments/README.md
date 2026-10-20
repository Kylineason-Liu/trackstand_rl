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

## Bounded steering speed-step sweep

`steering_speed_sweep.py` is a standalone stdlib/SocketCAN controller and recorder;
transfer this file alone to the Pi. It uses steering ID **2** (host `102`, reply
`002`), not rear drive ID 1. No ROS nodes, separate feedback collector or package
build is needed. Stop other motor writers/query recorders; secure the bike with
the **front wheel airborne**, steering physically straight at startup and free
throughout the selected travel. Keep an independent emergency stop available.

From the repository root:

```bash
# Preview: no CAN access or CSV creation.
python3 deploy/experiments/steering_speed_sweep.py
# First check: low-speed out-and-back, then return toward startup zero.
python3 deploy/experiments/steering_speed_sweep.py --execute --speeds .2 --cycles 1
# Full default speed set, three repeats (only after the first check is satisfactory).
python3 deploy/experiments/steering_speed_sweep.py --execute --repeats 3
```

Default speed magnitudes are **`1 2 4 6 8 10 12` rad/s**, with three out-and-back
cycles per speed. C1 commands jump directly to the signed speed, then to zero
at each endpoint: **no host ramp**. Endpoint zero commands can occur between
nominal 50-Hz ticks. Firmware **B5 acceleration is read/logged, not changed**;
actual speed can therefore still rise/fall gradually. Commands do not guarantee
instantaneous mechanical speed or steady motion.

The measured overspeed abort is **16 rad/s**, below the bike's configured
steering limit of 23.9 rad/s. Startup positioning and the final return toward
zero still use at most 0.4 rad/s; the measurement traversals use the full speed
steps above. No firmware current ceiling or acceleration is increased by this
higher-speed revision.
Startup warns when a constant-acceleration estimate using the read B5 value
would require more than the full stroke to reach a requested speed from rest.
That estimate is conditional on firmware enforcing B5, not a calibrated motor
response. The warning does not change B5 or suppress the requested speed step.

Turnarounds are at approximately **±0.6 rad** from the software startup zero;
the hard angle abort is **±1.2 rad**, with an additional velocity/age lookahead.
These are test settings, not measured stopping-distance guarantees. There are
stopped rests before reversals, leg/stop timeouts, freshness, direction, speed,
temperature, fault, bus-overflow and conflicting-writer checks. CF disables
torque but cannot guarantee a mechanical stop. No hardware origin or gains are
changed. Only the selected motor is commanded: ID 1 is untouched with the
default steering ID 2.

High-speed defaults use **200 Hz queries**, a **10 ms motion-query deadline**,
**20 ms feedback-age limit**, and **15 ms position lookahead**. The lookahead
is a reaction guard, **not an identified braking time**. Argument validation
checks endpoint margin against speed, feedback age, query period/deadline and
lookahead. Endpoint zero bypasses the periodic C1 tick. Long Pi scheduling
stalls remain dangerous; do not loosen timeouts merely to suppress aborts.

The default temporary **B3 ceiling remains 1 A**, never raising an existing
lower limit. `--current-limit-a` chooses a different test ceiling. The measured
current guard is more tolerant than the rear sweep: **1.5 × actual B3 + 0.05 A**
(about 1.55 A if B3 reads exactly 1 A), configurable with
`--current-guard-factor` and `--current-guard-margin-a`. This allowance is not a
motor rating. B3 acknowledgements/readbacks still permit only 1 mA downward
rounding; more current-spike tolerance does not weaken parameter verification.
Original B3/CD settings are restored only after CF and a fresh AE confirm
disabled; crashes/power loss can prevent restoration.

The CSV saves automatically to `logs/<timestamp>_steering_speed_sweep.csv` beside
the script; `--output` selects an exclusive alternative path. It records raw
TX/RX bytes, unwrapped startup-relative **A3 position**, A2 speed, A1 current,
B0 torque constant, AE status, B5 acceleration, phase/leg/cycle metadata, and
individual feedback ages. Nominal A3/A2/A1 rates are **200 Hz each**, AE 5 Hz,
B0 1 Hz. Torque is the signed A1 × B0 estimate, not an independent measurement.

For analysis, select fresh `record_type=rx` rows by opcode. Other rows repeat
cached state and are not independent measurements. `analysis_window` marks
post-settle travel within ±0.4 rad after a nominal **20 ms** exclusion,
**not verified steady motion**. At 12 rad/s an ideal full stroke lasts only
0.1 s; actual strokes may contain no steady-speed plateau. Inspect actual
speed, acceleration and angle dependence; do not fit friction to reversals or
assume every requested plateau was achieved. Compare directions at matching
angles to help separate friction from gravitational/angle-dependent loads.

Simulated-CAN tests (no real motor):

```bash
python3 -B -m unittest discover -s deploy/experiments/tests -p 'test_*speed_sweep.py' -v
```
