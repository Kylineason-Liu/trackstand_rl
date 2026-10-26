# Recorded rear-motor replay: implementation and audit

Implemented on 2026-10-26. Verification used recorded simulator inputs and virtual
CAN/clock responses. **No real motor or CAN interface was operated.**

See [the Raspberry Pi procedure](../../POLICY_REPLAY.md),
[rear replay script](../../drive_policy_replay.py), and
[target provenance](../../replay_targets/manifest.json).

## Command contract

The rear-only tool selects the actual `scaled_drive_nm` or `submitted_drive_nm`
column, updates its reference at 50 Hz, and sends C0 at 200 Hz by default. Four
successive packets repeat each reference. `--packet-rate 50` is available for
comparison. It converts torque to signed current using measured B0 Kt, then
encodes signed int32 little-endian milliamps. Encoder/torque signs default to
-1, matching the verified rear configuration.

No host damping, friction compensation, interpolation, torque slew or silent
clipping is added. The normal deployment node's existing rear damping remains
unchanged; this standalone test bypasses that ROS command pipeline. Current
feedback times B0 estimates electromagnetic torque, not independent shaft torque.

The default 350 samples comprise two seconds at zero torque followed by the
recorded 12.50–17.50 s core. They exactly equal the first seven seconds of the
previous matched simulator replay for all three sources and both rear levels.
`--post-hold 1` explicitly reproduces all 400 original samples; it is not default
because the last nonzero torque can continue accelerating an airborne wheel.
Both rear levels happen to be equal throughout these three real episodes.

## Protocol and cleanup

The repository's CAN protocol Rev.3.09b0 specifies C0/A1 current in 0.001 A,
A2 speed in 0.01 rpm, and A3 single/multiturn position. B3 limits current in
**position/speed modes**; it is logged, never treated as C0 protection. B4 is
current-mode current slew in 0.001 A/s, read without changing it. Zero's special
meaning is not assumed.

CD is a volatile communication watchdog. The tool temporarily enables/shortens
it to at most 500 ms, preserving action bits and tighter enabled timeouts. It
registers rollback before writing, checks acknowledgement and independent
readback, and restores the original only after fresh CF/AE confirm disabled.
Disable and restoration results are recorded separately. Other traffic may
refresh this communication watchdog; command-age protection is not established.
No B1, F0, B3, B4, gains or fault-clear writes are made.

The script controls only ID 1. It neither commands nor disables steering.
Independent support, an airborne rear wheel, other writers stopped and the
other motor independently disabled are required. RC/kickstand commands do not
stop standalone replay. Ctrl-C requests cleanup; CF and zero torque are not
mechanical brakes. Process crashes can leave a temporary watchdog configured.

The default 1.5 A command bound rejects the entire plan before C0, without
clipping. Measured-current, speed, temperature, freshness and scheduling aborts
stop incomplete tests. The 10 rad/s speed threshold is an initial experimental
bound near the earlier 8 rad/s sweep, not the simulated 61.6 rad/s limit. A free
wheel may reach it before completing the sequence even with good torque tracking.

## Verification

**70 tests passed:** 20 rear tests, 21 steering regression tests and 29 existing
shared transport tests. They cover actual simulator-input equality, sign/units,
quantization, packet repetition, absence of velocity-dependent host damping,
multiturn feedback, one-motor ownership, measured excursions, independent C0/A1
freshness, missed deadlines without catch-up bursts, lost watchdog write replies,
restore failures, confirmed-disable requirements and recording failures.

```bash
python3 -B -m unittest discover -s deploy/experiments/tests -p test_drive_policy_replay.py -v
python3 -B -m unittest discover -s deploy/experiments/tests -p test_steering_policy_replay.py -v
python3 -B -m unittest discover -s deploy/experiments/tests -p test_steering_sine_mit_io.py -v
```

Fresh-copy previews of both scripts succeeded from an unrelated working directory
with `python3 -S -B`, without site packages or new log directories. Runtime needs
only Python >=3.10 and the three adjacent scripts plus target CSVs/manifest.
Syntax, Pyflakes, 120-column and whitespace checks passed.

The only existing files changed are the experiment README, steering guide and
steering replay's shared-helper extraction. All other steering functions retain
their original syntax trees; steering plans and behavior pass regression tests.
**167 protected existing files remained unchanged**, including production code,
models/captures, target exports, the original sine helper, both progress ledgers
and Git index. New files are the rear script/test, combined guide and this audit.

See [verification.json](verification.json), [rear tests](rear_tests.log),
[steering regression](steering_regression.log), and
[shared transport tests](shared_transport_tests.log). Actual Pi scheduling,
firmware response, tracking accuracy and physical protection still require
hardware measurement; virtual CAN tests do not establish motor performance.
