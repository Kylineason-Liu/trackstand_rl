# V2 motor replay: offline deployment audit

Audited on 2026-11-04. This audit concerns the new
[`v2_motor_replay.py`](../../v2_motor_replay.py), not a live bike run.
**No CAN interface, ROS node, serial device, GPU or motor was opened/started.**
The user selected steering **position** plus rear **velocity**, preserving the
current v2 action meanings.

## Result

The reviewed script and its actual reference bundle pass the independent
protocol/lifecycle checks and the 16 focused tests. No critical code finding
remains from this audit. Real hardware execution, loaded response, firmware
protection and measured packet cadence remain unverified.

- [Independent mock result](independent_audit.json), also repeated with monotonic
  origins [1,000,000 s](independent_audit_1000000.json) and
  [10,000,000 s](independent_audit_10000000.json).
- [Focused test log](focused_tests.log),
  [audit source](audit_replay.py), [final verification](verification.json).
- [Hardware instructions](../../V2_MOTOR_REPLAY.md) and
  [actual source manifest](../../replay_targets/v2_26600_no_dr/manifest.json).

## Checks completed

The independent two-device mock uses the documented reply bytes and an
independent MIT bit decoder, with 150 microseconds of artificial submission
overhead per frame. At each of the three clock origins, a 10-second source
prefix produces **2,000 steering packets, 2,000 rear packets and 2,000 host-I
updates**, without extending the waveform. These are **mock counts**, not
measured Pi/CAN rates. The test also checks:

- Preview returns with socket creation prohibited; no output files are created
  by the preview path.
- The 500 target rows cover source time 0–9.98 s and hold through 10 s; the CSV
  is byte-identical to the actual Isaac capture, not an invented waveform.
- Steering receives one B1 on ID 2 only, then fresh A3/F1 references; rear uses
  a software multi-turn origin without B1. A signed-int32 rear rollover produces
  the correct continuous displacement.
- Native MIT command IDs are 0x502 steering and 0x501 rear; feedback IDs are
  0x002 and 0x001. Encoded signs, fields and physical units match the current
  verified source mappings, including zero-position offsets.
- **3,000 random codec cases per mock run** match an independent decoder to
  half an encoding LSB. Steering gains encode approximately 10.01221/0.200244;
  rear position Kp is zero and velocity P encodes approximately 0.3003663.
- Rear velocity encoding cancels feedforward-torque quantization residual;
  the effective velocity target remains within half a velocity LSB of the
  requested target. Wire velocity and effective target are distinct log fields.
- Both motors receive CF before any configuration rollback. Retained mode 5
  requires an inactive F1, rather than treating mode 5 itself as disabled.
  Original F0/CD settings are restored, including a mocked original 20-ms
  watchdog. No B3 or firmware C1 gain is written.

The focused suite additionally exercises corrupt reference hashes, a bounded
but incorrect steering processing result, source-prefix restrictions, PI memory
and anti-windup, fault/overspeed detection, primary versus cleanup failures,
50/200/400-Hz packet choices with a separate 200-Hz integrator, and a 37-ms
scheduling delay without a catch-up burst.

## Findings addressed before the final check

The initial implementation used `now + period` deadlines, which could accumulate
dispatch/wakeup delay into waveform stretch. Anchored deadlines now keep source
duration fixed and select the current 50-Hz source row; missed source, packet
and I slots are logged rather than replayed in bursts.

An independent mock then exposed a floating-point boundary error: subtracting
an absolute monotonic origin can round an exact target boundary just below its
integer slot. The final slot calculation includes an ULP-aware arithmetic
tolerance and passes all 1,500 boundaries at ordinary and large clock origins.
The initial failed audit log is retained as `audit_replay.log`; it is not a
failed hardware run. Arithmetic examples are in
[`timing_boundary_examples.json`](timing_boundary_examples.json).

Other addressed findings include activation invalidating the disabled flag,
an explicit temporary 500-ms watchdog instead of retaining an unusably short
timeout, fresh feedback after slow journal writes, both-motor CF dispatch before
waiting for either acknowledgement, accurate pending-restore state, preservation
of the primary execution error when cleanup also fails, and file plus parent-
directory fsync before firmware configuration writes. These changes were made by
the implementer; this analyst wrote no production code.

## Protocol and interpretation boundaries

The local manufacturer PDF `自定义CAN通信协议_3.09b0.pdf`, pages 20–22, explicitly
defines MIT position in rad, velocity in rad/s and torque in N·m; F0 fields are
little-endian 0.1 rad / 0.01 rad/s / 0.01 N·m units. Commands on 0x400/0x500 IDs
enter MIT mode and return the F1 layout. CF exits and B1 sets the current origin.
The PDF was read directly because `tsinghua_bike_project/can_protocol.txt` ends
at page 20. A1/A2/A3/B0 layouts and physical sign mappings were cross-checked
against the existing protocol modules, proven steering replay and MIT rear test.

Steering native MIT receives an absolute position relative to the newly verified
zero. Rear MIT receives a velocity target, position Kp=0 and host integral as
feedforward torque. The I state starts once at zero and is retained across target
changes. No second steering slew, passive-friction compensation or source-time
resampling is added. The default replay starts at episode time zero; selecting a
later slice with a fresh I would not recreate source controller history.

The nominal 200-Hz host-I/packet schedule is not the motor's internal loop rate.
At 50-Hz packets the host still computes I at 200 Hz, but new feedforward reaches
the motor only at 50 Hz. A2 is asynchronous feedback and hardware I uses actual
elapsed time, unlike fixed-5-ms integration of ideal fresh simulation velocity.
Host timestamps identify send-call completion and receipt, not motor acquisition
or actual wire time. Actual encoded gains and values are logged.

**F0 ranges are encoding ranges, not proof of a post-PD torque clamp.** The script
monitors measured/current-derived effort and speed with the documented tolerance;
that abort mechanism is not Isaac's instantaneous post-PD effort clipping.
A1 times B0 is a current-derived estimate, not independent shaft torque. The
program is a supported command/response experiment; it does not balance the bike,
operate kickstands, or establish hardware equivalence to Isaac/MuJoCo physics.

## Reproduction

From this directory, all offline:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B audit_replay.py
AUDIT_CLOCK_ORIGIN=1000000 PYTHONDONTWRITEBYTECODE=1 python3 -B audit_replay.py
AUDIT_CLOCK_ORIGIN=10000000 PYTHONDONTWRITEBYTECODE=1 python3 -B audit_replay.py
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s ~/trackstand_rl/deploy/experiments/tests -p test_v2_motor_replay.py -v
```

The helper substitutes a Python mock bus and never calls `open_bus`. Do not add
`--execute` to an unaudited test command; follow the dedicated replay guide when
the user performs the independently supported bike experiment.
