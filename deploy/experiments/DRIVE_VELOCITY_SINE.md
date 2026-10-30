# Rear velocity sine: C1 versus MIT with an integral term

`drive_velocity_sine.py` is a standalone Python 3.10+ experiment for rear motor
**ID 1**, host **0x101**, MIT host **0x501**, feedback **0x001**. It needs no ROS
build or third-party Python packages. Keep the rear wheel airborne/free, support
the frame, independently disable steering, and stop all other motor CAN writers.

## Commands and timing

Run from the repository root:

```bash
# Preview only; no CAN or output files.
python3 deploy/experiments/drive_velocity_sine.py --method both

# C1 only: works without knowing the gain units; reads/records B8 and B9.
python3 deploy/experiments/drive_velocity_sine.py --method c1 --execute
```

`--method c1` or `mit` runs that method for **10 wall-clock seconds**.
`--method both` runs **C1 for 10 s, disables/confirms the mode transition, then
MIT for 10 s**. Startup, mode transition and shutdown are additional time; there
is no added two-second rest or stop-speed threshold. No firmware-integrator reset
is assumed from CF, and the host integral starts at zero for the MIT phase.

Both modes send setpoints at nominal **200 Hz**. The default sine is **±4 rad/s
(±38.2 rpm), 0.5 Hz**, giving five cycles per phase and peak requested acceleration
**12.57 rad/s²**. This is a moderate initial profile based on the previously tested
rear speed range, not a measured optimum. Set `--amplitude` and `--frequency` to
change it. Actual monotonic elapsed time defines the waveform. Late slots are
recorded/skipped rather than sent in bursts or extending the ten-second window.

## Gain units are not established: do not silently copy raw B8 into MIT Kd

The local **GIM CAN protocol 3.09b0**, pages **11–12**, defines B8/B9 as little-endian
float32 velocity-loop Kp/Ki. It **does not specify their internal speed units,
current units or integral timestep convention**. Page **24** shows that C1's PI
output is Q-axis **current**. Page **20** shows that MIT combines proportional,
derivative and feedforward **torque** before converting to current. C1 packet
speed being `0.01 rpm` does **not** prove that its internal gain uses that unit.
The vendor Python/C++ wrappers only read/write the floats; they do not resolve
the units. The user also confirmed the units are unknown.

Therefore MIT execution requires these **explicit unit declarations**:

| Flag | Meaning / choices |
|---|---|
| `--gain-speed-unit` | Numeric speed-error unit used by B8/B9: `rad_s`, `rpm`, `centirpm` (0.01 rpm counts), or `turn_s` |
| `--gain-current-unit` | PI-output numeric unit: `a` or `ma` |
| `--ki-timebase` | `second` if Ki multiplies a time integral; `tick` if it multiplies error once per firmware update |
| `--ki-period-s` | Required only for `tick`: the **firmware PI period**, not automatically the 5 ms packet period |

Selecting flags records a **user-declared basis**, not manufacturer confirmation.
The script refuses MIT motion without a basis; it does not guess one from gain
magnitudes or from the C1 packet format. Confirm the unit convention with the
manufacturer before claiming a gain-matched comparison.

For example, **only if** gains are confirmed to use amperes and rad/s, with
continuous-time Ki:

```bash
python3 deploy/experiments/drive_velocity_sine.py --method both --execute \
  --gain-speed-unit rad_s --gain-current-unit a --ki-timebase second
```

Use `--method mit` with those same verified conversion flags for MIT alone.
If Ki is instead a per-firmware-update coefficient, use `--ki-timebase tick`
and the confirmed `--ki-period-s`; do not insert an assumed 1 kHz rate.

Let `s` be speed-unit counts per rad/s and `a` amperes per PI-output unit.
The script reads the actual Kt using **B0**, then computes:

```text
MIT position Kp = 0
MIT velocity Kd = B8_Kp × s × a × Kt               [Nm/(rad/s)]
host Ki         = B9_Ki × s × a × Kt              [Nm/rad]    (continuous Ki)
host Ki         = B9_Ki × s × a × Kt / firmware_dt            (per-update Ki)
integral        = ∫(desired_speed − measured_speed) dt         [rad]
MIT feedforward = host Ki × integral                          [Nm]
```

Integration is trapezoidal using **actual monotonic command intervals** and the
latest A2 velocity during MIT operation. Native velocity and torque signs are
**−1**, with output-side gear ratio **1**, matching the existing rear configuration.
C1 speed is signed int32 little-endian **0.01 rpm**; MIT uses **rad/s and Nm**,
with the motor's read-only F0 ranges and protocol bit packing. MIT Kd is quantized
within `[0,5]`, Kp is encoded exactly zero, and position is zero but has no control
effect. **No B1 reset or absolute-position origin is required with Kp=0**. A3's
first multi-turn count is retained as a software displacement reference for logs.

Even with confirmed gain units, this is not a proof of identical controllers:
C1 has an internal PI loop, while MIT uses internal velocity P plus a **200 Hz
host integral using delayed feedback**. Quantization, C1's B3 current ceiling,
B5 acceleration setting, internal filtering/feedforward, integrator initialization
and firmware timing can differ. Original B3/B5/gains are recorded; B5 and gains
are not modified. Compare actual recorded target timing and response.

## Limits and cleanup

There is **no angle/travel guard, artificial target slew, tracking-error abort,
small experiment-current threshold or packet-lateness abort**. The integral and
gain are not silently clipped. The experiment stops on:

- Measured speed beyond **61.6 rad/s** or current-derived/F1 torque beyond
  **6.28 Nm**, following the existing rear motor limits.
- Estimated **total** MIT torque (encoded Kd velocity term **plus** encoded
  integral feedforward) beyond **6.28 Nm**. This is checked before sending; it is
  not merely a feedforward-only limit or a guarantee between feedback samples.
- An unencodable target/feedforward or converted Kd outside the protocol range.
- Missing velocity/current feedback for **0.25 s**, driver status for **0.5 s**,
  or fresh active-MIT feedback not established within **0.25 s**.
- Driver/system faults, CAN errors/overflow, malformed data, a detected competing
  rear/broadcast writer, CSV failure/backlog, or Ctrl-C/SIGTERM.

These guards terminate the test; there is no continuing saturated integrator and
no hidden anti-windup formula replacing the requested integral law. C1's internal
integrator/anti-windup is not exposed by this protocol.

Before motion, B3 is **only lowered if it exceeds 6.28/Kt A**, never raised. Its
acknowledgement/readback now accepts a downward difference of **0.1 A or 1% of
the requested current, whichever is larger**, replacing the overly tight 1 mA
tolerance. Accepted differences are printed and logged, with actual applied
readback retained in JSON. A zero value, a value above the requested ceiling,
or a larger downward discrepancy still fails with requested/returned values
in the error. This is a software acceptance tolerance, not a measured firmware
quantization specification. The same tolerance applies to restoration, so the
restored ceiling can be slightly below its original value.
B3 protects C1 position/speed-mode current, **not proven MIT total torque**. CD is
enabled/shortened to at most **500 ms**, preserving any shorter enabled timeout
and existing action bits. Only after CF confirms mode zero are originals restored.
CF confirmation permits the first reply to retain its previous active mode,
retrying within **one second**. Logging failure cannot block the CF attempt; a
failed restoration does not prevent attempts for other settings. No B1/F0/B5/
gain/fault-clear writes occur. Firmware-watchdog behavior across these modes has
not been physically validated by this implementation; CF also does not stop coast.

## Files and feedback

Execution automatically saves exclusive timestamped **CSV and JSON** under
`deploy/experiments/logs/`; no logging flag is needed. `--output path.csv` and
`--interface can0` are optional. Files are never overwritten.

The CSV records every command/query/reply, native CAN ID and payload, desired and
encoded target, actual command interval/lateness/skipped slots, signed velocity,
A1 current and current-derived torque, continuous multi-turn A3 displacement,
F1 feedback/status, AE voltage/current/temperature/mode/fault, integral error and
torque, and estimated total MIT torque. Nominal reads are **A2 200 Hz, A1 100 Hz,
A3 50 Hz, AE 10 Hz**; C1/F1 automatic replies provide additional feedback. Data
are asynchronous; timestamps are **host send/receipt times**, not motor acquisition
or independent on-wire timestamps. Current-derived torque is not shaft torque.

JSON retains requested mode/settings, raw B8/B9, B0 Kt, selected unit basis,
converted gains, original/applied configuration, F0 ranges, per-phase counts/
achieved rates/gaps, first A3 count, completion/disable/restore/error results and
script hash. Missing numerical values are JSON null; partial CSVs remain on errors.

Validation: **33 simulated-CAN tests passed**, covering byte layout/signs, all unit
conversions, real-dt integration, two 2,000-command phases, delays without timing
abort, feedback loss, logger failure, CF transitions, restoration and main-program
success/failure JSON output. This is
software verification, not a completed bike test or evidence one method is better.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s deploy/experiments/tests -p test_drive_velocity_sine.py -v
```
