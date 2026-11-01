# Rear MIT velocity-control tests

`drive_mit_velocity_test.py` repeats the **same target procedures as
`drive_c1_kp_identify.py`**, using rear motor **ID 1**, MIT host **0x501**, feedback
**0x001**. It reuses the existing plan and phase scheduler. The scheduler's C1
packet is replaced before bus transmission: **no C1 motion command is sent**.
This experiment needs Python stdlib and the adjacent `drive_c1_kp_identify.py`
and `drive_velocity_sine.py`. No ROS build or separate recorder is needed.

## Controller and manually adjustable P/I gains

```text
MIT position Kp = 0
MIT velocity Kd = --velocity-kp       [Nm/(rad/s)]
host integral Ki = --velocity-ki     [Nm/rad; default 0]
integral torque = Ki × integral(target_velocity − A2_velocity) dt
native feedforward = SIGN × integral torque
intended user torque = Kd × (target_velocity − measured_velocity) + integral torque
```

`--kd` is an alias of `--velocity-kp`. Default **0.30 Nm/(rad/s)** is a provisional
starting value from the real C1 identification, not a verified rear MIT setting.
`--velocity-ki 0` preserves the previous P-only packets and plan labels. A positive
Ki enables host PI and labels segments `mit_velocity_pi`. Ki is explicitly in SI
units; it is **not raw B9**. The previous C1 review suggested approximately
**3.3 Nm/rad** as a provisional effective I estimate, not a manufacturer-defined
register conversion or an already validated MIT setting.

The integral updates at the host packet rate (default 200 Hz), using actual
monotonic elapsed time and trapezoidal error integration. Steady A2 is queried
at 100 Hz, so intervening updates use the latest available speed. It resets at
**each complete segment's start and motor disable**, but stays continuous through
the baseline/pulse/return within a step and through a sine's settling/measurement.
The first packet after reset has zero integral. An active-zero segment also starts
with zero integral; it may then accumulate braking torque if the wheel coasts.
The motor still applies P internally; the host's I update is not its firmware PI loop.

Kd and feedforward are sent in each packet, never written through B8/B9. Protocol Kd range is
**0–5** with 12-bit resolution; this test requires a positive encodable value.
For example, requested 0.30 encodes as **0.3003663**. Position gain encodes exactly
zero; position target is irrelevant. No B1 zero write is needed. The first A3
received during capture initialization after configuration defines the software
displacement origin; earlier initialization replies do not choose it. Its count
and monotonic receipt timestamp are recorded in JSON and a CSV zero-reference
event. Signed rollover and each segment's initial position/speed/age are retained.
The measured velocity is never offset to zero; a coasting wheel remains visible
to the velocity P controller.

Native velocity/torque signs use **−1**, positive user velocity means forward,
matching the established rear configuration. Prior P-only tests confirmed the
expected current response but showed poor near-zero tracking; this new host-PI
mode still needs hardware validation. First check the short signed pilot airborne.
Do not infer actual tracking from successful packet transmission alone.

## Finer velocity range

The script reads **actual F0**, preserves its position and torque ranges, and
changes **only Vel_Max**, with write acknowledgement and independent readback.
Automatic range is the larger of **3 rad/s**, **1.3 × the selected plan's peak**,
and the space needed for the zero-encoding correction below. It rounds upward
to F0's 0.01-rad/s configuration unit.

| Selected plan | Automatic velocity range | Velocity encoding step |
|---|---:|---:|
| Full C1-matching plan, requests through ±10 rad/s | **±13 rad/s** | **0.0063492 rad/s** |
| Small zero-baseline plan with peak ≤about 2.3 rad/s | **±3 rad/s** | **0.0014652 rad/s** |
| Protocol default, for comparison | ±45 rad/s | 0.0219780 rad/s |

The full plan cannot use ±3 rad/s without losing its ±10-rad/s cases. The ±13
default covers the previous C1 run's measured 12.687846-rad/s peak. The prepared
v1 policy startup/settling reference peaks at 2.483943 rad/s, and its real 400-Hz
C1 replay peaked at 2.955 rad/s; ±3 covers those recorded cases, not every future
motion. This script generates the identification holds/steps/sines, not a replay
of that policy episode.

`--velocity-max <rad/s>` overrides the automatic range. A range that cannot encode
all selected targets plus the codec correction is rejected before any F0 write;
targets are never silently clipped. F0 is an **encoding range**, not a proven
physical motor speed/torque limit. A1/A2/A3 remain the primary high-resolution
measurements; raw F1 is also logged. A speed beyond the F1 range can make F1
unrepresentative without invalidating A2's independent field conversion.

**Feedforward encoding detail:** signed 12-bit velocity/torque fields have 4,096 codes,
so neither field has an exact zero under the documented symmetric mapping. With
torque range ±18 Nm, nearest-zero feedforward decodes to +0.0043956 Nm. Sending
that uncorrected would introduce a fixed near-zero torque bias. The script
therefore encodes native velocity as
`SIGN*desired_velocity − (wire_ff − SIGN*integral_torque)/actual_Kd`.
This cancels **only the feedforward encoding error**, preserving the intended
integral torque. At Ki=0 it is exactly the previous zero-code correction.
There is no friction compensation, target smoothing or velocity error correction
in this encoding step. The actual torque range is read, not assumed to equal 18 Nm.

CSV `encoded_target_velocity_rad_s` is the **actual wire velocity target** in
forward-positive coordinates. Its torque-equivalent target is:

```text
effective_P_target = encoded_target_velocity_rad_s
                   − (mit_feedforward_native_nm + integral_torque_nm) / mit_kd
total_user_torque = mit_kd × (effective_P_target − measured_velocity)
                  + integral_torque_nm
```

That equivalent target matches `target_velocity_rad_s` within **Vel_Max/4095**.
This difference is explicit, not a change to the test waveform. The original
torque range stays unchanged. `integral_error_rad` records the retained integral
state after anti-windup, and `integral_torque_nm = Ki × integral_error_rad`.
Both remain zero at Ki=0. `mit_feedforward_native_nm` is the actual encoded native
feedforward, including its quantization residual; it varies when I is enabled.

## Procedure and frequencies

Defaults match the C1 campaign exactly: seed **42**, three shuffled repetitions,
**342 segments /1,194 s (19 min 54 s)** excluding startup/shutdown:

- **54 steady holds:** ±0.05/0.1/0.2/0.4/0.8/1.5/3/6/10 rad/s; 2-s settling +3-s measurement.
- **120 steps:** baselines −6/−2/0/2/6; perturbations ±0.1/0.25/0.5/1 rad/s;
  0.6-s baseline +0.4-s pulse +0.6-s return.
- **108 sines:** baselines −3/0/3, amplitudes 0.25/1 rad/s, frequencies
  0.5/1/2/5/10/20 Hz; 1-s settling then at least four cycles and four seconds.
- **60 zero contexts:** active zero-velocity P/PI control for 2 s, not disabled rest
  or proof that the wheel stopped or returned to its physical starting angle.

MIT packets default **200 Hz**, configurable with `--packet-rate 50|200|400`.
A1/A2 queries are **100 Hz each** for steady/zero, **400 Hz each** for dynamic
segments; A3 **50 Hz**, AE **10 Hz**, B8/B9 **1 Hz each**. F1 replies accompany MIT
packets. Actual rates/timestamps/skipped slots are recorded; packet frequency is
not the driver's internal controller frequency.

## Run on the Pi

Rear wheel airborne/free, frame supported, steering independently disabled and
all other motor writers stopped. ROS/RC cannot stop this standalone program.
From the repository root:

```bash
# Preview: opens neither CAN nor output files.
python3 deploy/experiments/drive_mit_velocity_test.py

# Short signed pilot: 7.2 s, +/-1-rad/s steps, automatic +/-3-rad/s encoding.
python3 deploy/experiments/drive_mit_velocity_test.py --execute \
  --suite steps --repeats 1 --step-bases 0 --step-sizes 1

# Full matching campaign: automatic +/-13-rad/s encoding.
python3 deploy/experiments/drive_mit_velocity_test.py --execute --velocity-kp 0.30

# Short PI pilot, then the full same-plan P-versus-PI comparison.
# Ki=3.3 Nm/rad is provisional, based on the earlier C1 effective-gain estimate.
python3 deploy/experiments/drive_mit_velocity_test.py --execute \
  --velocity-kp 0.30 --velocity-ki 3.3 \
  --suite steps --repeats 1 --step-bases 0 --step-sizes 1
python3 deploy/experiments/drive_mit_velocity_test.py --execute \
  --velocity-kp 0.30 --velocity-ki 3.3
```

For a higher-gain **near-zero** comparison, this example selects smaller targets
and uses ±3-rad/s encoding. Repeat it with different gains and unchanged physical
conditions to compare gain effects; it is a different case subset from the full
C1 campaign:

```bash
python3 deploy/experiments/drive_mit_velocity_test.py --execute --velocity-kp 1.0 \
  --levels .05 .1 .2 .4 .8 1.5 \
  --step-bases 0 --step-sizes .1 .25 .5 1 \
  --sine-bases 0 --sine-amplitudes .05 .1 .25 .5 1
```

All C1 plan/timing/rate flags are available: suites, levels, bases, amplitudes,
frequencies, holds, repeats, seed and feedback rates. `--output` selects an
exclusive CSV path; default files are timestamped under `deploy/experiments/logs/`.

## Necessary guards, settings and cleanup

There is no tight packet-lateness, angle, tracking-error, small-current or
rest-speed abort. Missed slots are logged/skipped without bursts or time stretching.
Existing **61.6 rad/s** measured speed, **6.28 Nm** measured/current-derived and
estimated total MIT torque guards remain. A larger gain with a large velocity
step can legitimately trigger the torque guard; for example, gain 1 and an
initial 10-rad/s error requests 10 Nm. Use smaller test cases for higher gains.
Targets and P gain are never clipped. **Integral anti-windup is explicit:** I
torque is bounded by the smaller of the actual F0 torque range and physical
6.28-Nm allowance, reserving `actual_Kd × Vel_Max/4095` for velocity quantization.
Integration increments are frozen if they push P+I further beyond that allowance;
unwinding remains possible and the integration timestamp still advances.
The stored integral itself is corrected, preventing hidden state accumulation.
This adds no new abort threshold. The existing total-torque guard still stops
an excessive P request or an excessive P+I request after an abrupt error change.

Missing A1/A2/F1 beyond **1 s**, AE beyond **2 s**, driver faults, competing CAN
writers, bus/logging errors and unencodable commands stop execution. Invalid F0
readback prevents motion because command and feedback decoding would be wrong.
These host checks do not certify between-query torque or firmware protection.
B3/B5/B8/B9/origins are read only; B3 is not assumed to cap MIT total torque.

Startup confirms CF disable, reads settings, **durably saves original F0/CD in
JSON before any write**, configures/verifies F0 and a CD watchdog ≤500 ms, then
reads initial feedback and starts the timed plan. Completion, Ctrl-C or error
attempts CF disable, verifies it, then restores original CD/F0 with readback.
Mode 0 confirms disable; if CF retains MIT mode 5, a subsequent F1 must confirm
inactive/fault-free status. Retained mode 5 alone is never accepted as disabled.

**F0 persists across power loss.** SIGKILL, power loss or failed cleanup can leave
the narrowed range behind. JSON retains original bytes and pending restoration
payloads; the terminal prints original F0 before configuration. After independently
disabling the motor, original settings can be restored using those exact saved
payloads via `cansend can0 101#<original_F0_hex>` and, if needed, original CD.
Do not invent a fixed restoration range or reset the motor's origin.
If only part of restoration fails, `pending_restore_payloads` contains only the
unverified settings; a subsequent cleanup retry does not rewrite successful ones.

## Recorded data and validation

CSV includes all raw TX/RX, desired and encoded targets, encoded Kd/Kp/feedforward,
estimated total torque, fresh A1/A2/A3/F1/AE and state ages. JSON and `_plan.csv`
retain the exact plan, source/helper hashes, original/applied settings, execution
versus final readback ranges, initial multi-turn reference, completion/cleanup
outcomes and phase rates/skips. Collection is automatic, including partial runs.
Use fresh opcode-specific RX rows, not repeated cached state in other rows.
`mit_configuration` in JSON summarizes requested/encoded Kd, raw gain code,
position Kp=0, chosen Ki, integration/reset/anti-windup rules, zero-integral codec
reference, all F0 ranges and velocity resolution. Each phase includes final
integral state/torque and the number of anti-windup-limited updates. The new JSON
schema is `drive_mit_velocity_test_v2`; the CSV columns remain compatible.
The CAN filter monitors both legal rear MIT IDs
0x401 and 0x501 so a competing writer on either ID is detected.

Copy all three files back for MIT/C1 comparison. The existing C1 PI analyzer
intentionally accepts its own C1 schema; do not pass these MIT logs to it as C1
measurements. No physical improvement is claimed before examining the new data.
The previous P-only MIT/C1 report analyzer also assumes I=0; PI analysis must use
the logged integral torque in the torque equation above.

Virtual CAN/time checks, never actual hardware:

```bash
python3 -B -m unittest discover -s deploy/experiments/tests -p 'test_drive_mit_velocity_test.py' -v
```

Protocol basis: local **GIM CAN 3.09b0, pages 20–22**, including persistent F0,
MIT SI units, unsigned gain ranges, packed command/reply fields and F1 status.

## Configuration audit

The audit checks gain validation **before rounding**, independently calculated
packet bytes and native torque signs, non-default F0 ranges and exact restoration,
C1-register independence, first-capture zeroing versus earlier unsolicited A3,
signed multi-turn rollover, retained-mode disable verification, lost write replies,
partial restoration retries and alternate MIT writer detection. No new tracking,
angle, rest-speed or tight timing abort was added.

For reference, requested velocity P=0.30 with F0 ranges ±95.5 rad/±13 rad/s/±18 Nm
and zero desired speed encodes as `80007FD0000F6800`: position Kp code **0**, velocity
Kd code **246** (0.3003663 Nm/(rad/s)), and the documented zero-code correction.
The actual original position/torque ranges are read from the motor, not fixed to
those example numbers. Packet correctness does not establish rear MIT hardware
tracking performance; run the supported signed pilot and inspect its data.
