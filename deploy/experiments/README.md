# Standalone motor feedback

For controlled **rear C0 repeatability, starting-angle, breakaway and pulse/reversal
experiments**, see [DRIVE_REPEATABILITY.md](DRIVE_REPEATABILITY.md). The new
`drive_repeatability.py` provides preview, balanced rates, verified manual angle
placement, automatic feedback logs and resumable sessions.

For **both motors' separate recorded-target tests** and Raspberry Pi startup
steps, see [POLICY_REPLAY.md](POLICY_REPLAY.md). Rear C0 torque replay is provided
by `drive_policy_replay.py`; steering MIT replay remains separate. Never run the
two tests together or alongside deployment motor nodes.

For the model_2500 **recorded-policy steering replay**, both command levels,
50/200 Hz packet comparison and integrated feedback recording, see
[STEERING_POLICY_REPLAY.md](STEERING_POLICY_REPLAY.md). The new
`steering_policy_replay.py` defaults to preview-only and uses the portable CSVs
under `replay_targets/`; it does not run a policy or command the rear motor.

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

## Timed steering speed-step sweep

`steering_speed_sweep.py` is a standalone stdlib/SocketCAN controller and recorder;
transfer this file alone to the Pi. Default steering ID is **2** (host `102`,
reply `002`), with command/encoder/torque signs -1 and gear 1. No ROS nodes,
separate recorder or package build is needed. Stop other CAN writers.

**There is no steering travel protection.** Secure the bike with its front wheel
airborne, but also verify that all requested rotation is mechanically possible:
airborne does not prevent cable twisting or hitting steering stops. Keep an
independent emergency stop available. At 12 rad/s the default 15 s nonzero
interval requests **180 rad, or 28.6 revolutions**, if followed.

The flow now follows the rear sweep's timed pattern **without ramps or feedback
gates**: initial zero rest, then for each speed **step to target → settle → hold
→ step to zero → rest**. Defaults match rear timing: `--settle 5 --hold 10
--rest 2`. All phases end by elapsed time, regardless of actual motor state.
There is no stopped-speed test, positioning leg, angle reversal or return to zero
position. Zero-speed rest is enabled C1 control, not free coasting.

Speeds are **signed**, sent in the exact supplied order. Defaults are
`1 2 4 6 8 10 12 -1 -2 -4 -6 -8 -10 -12` rad/s. Positive-only `--speeds` does
not automatically add the negative directions. No host current/speed/angle,
temperature, fault, freshness or query-deadline stop is implemented. Old guard
flags and `--cycles` were removed; use `--repeats` for the whole timed sequence.

From the repository root:

```bash
# Preview only: no CAN access or CSV creation.
python3 deploy/experiments/steering_speed_sweep.py
# Example of choosing explicit short intervals; still verify physical clearance.
python3 deploy/experiments/steering_speed_sweep.py --execute \
  --speeds 1 -1 --settle 0 --hold .2 --rest .5
# Rear-style full timing, only if continuous multi-turn steering is possible:
python3 deploy/experiments/steering_speed_sweep.py --execute --repeats 3
```

**No B3/CD/B5/gain/origin settings are written.** The old temporary 1 A ceiling
is gone; firmware current/acceleration/protection settings stay as they already
are, rather than being raised or disabled. B5 is read once for the CSV. Actual
speed may still ramp under the motor's firmware and physical dynamics.

C1 targets are nominally **50 Hz**, with immediate changes at speed-step
boundaries. A3/A2/A1 requests are nominally **100 Hz each**, AE 5 Hz, B0 1 Hz.
Replies are collected asynchronously: missing/delayed measurements do not stall
or abort timed commands. Rates are not hard-real-time guarantees. CAN and CSV
I/O errors still terminate the script. Normal completion, Ctrl-C and SIGTERM
attempt CF disable; the cleanup acknowledgement wait is not a motion-phase
timeout. CF removes torque, not guaranteed mechanical motion. A crash/SIGKILL
may leave the last speed active: the script no longer enables a watchdog.

CSV saves automatically beside the script under
`logs/<timestamp>_steering_speed_sweep.csv`; `--output` selects an exclusive
alternative path. It retains raw TX/RX bytes, phase/repeat/step metadata, A3
unwrapped startup-relative position, A2 speed, A1 current, B0 torque constant,
AE status, B5 acceleration and separate value ages. First A3 count is a software
reference only. A1 × B0 is a current-based torque estimate, not a torque sensor.

Analyse fresh `record_type=rx` rows by opcode; other rows repeat cached state.
Missing data stays NaN until received, then its age exposes any later gaps.
There are no `query` completion rows, cycle/leg/guard fields or `analysis_window`
in the new schema. Old CSVs are unchanged. A scheduled `hold` is not proof of
steady speed: inspect actual velocity before fitting friction/damping.

Simulated-CAN tests (no hardware):

```bash
python3 -B -m unittest discover -s deploy/experiments/tests -p 'test_*speed_sweep.py' -v
```
