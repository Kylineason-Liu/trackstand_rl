# Steering initial-transient replay entry point

This report records the initial implementation and its saved validation. The
later [relaxed replay timing](../20261027_replay_relaxed_timing/README.md) supersedes
the original 3 ms deadline below with 20 ms and a send-relative schedule. Targets,
initialization and other trips are unchanged; durations/rates are now nominal.

Implemented 2026-10-27. The new
[steering_initial_transient_replay.py](../../steering_initial_transient_replay.py)
reuses the unchanged steering_policy_replay.py backend. Only steering ID 2 is
addressed; no rear replay or production motor node is modified.

## Recorded input and timing

The default is the existing model_2500 Isaac400 env-0 episode-1 CSV, SHA256
`c799e80fbc87af307871e35dce7ad409f3e636ff662e99ed498c3800a4cf3ad6`.
It selects **source 0–5 s**, 250 recorded targets, then adds a **one-second final
hold**: 300 target intervals /6 seconds total. There is no preparation ramp or
pre-hold. The first recorded command is applied directly from the verified
physically straight reference. Source and plan timestamps remain separately logged.

For this source the first target is **0.3573741 rad /20.476 degrees**; the
five-second range is **0.3573741–1.3396667 rad /20.476–76.757 degrees**.
References update at **50 Hz**; --packet-rate 200 sends each held target four
times. Defaults are scaled input, Kp=10, Kd=0.2 and requested velocity/feedforward
zero. --level conditioned uses recorded conditioned targets without another slew.
Both levels happen to coincide in the provided source files.

## Runtime and protection boundary

The backend retains verified B1/A3/F1 zero establishment, measured B0 torque
constant, A3/A2/A1 queries at 100 Hz each, kernel RX timestamps, current feedback,
fault/freshness checks and CF/AE-confirmed cleanup. B1 stores a persistent origin;
physically straighten before every execution. Rear must be independently disabled,
front wheel airborne/free and bike independently supported, with other writers stopped.

Existing **3 A measured-current /3 ms send-lateness trips** remain unchanged.
Nominal stationary Kp times the first target is **3.573741 Nm**, so the initial
input is stronger than the previous ramped mid-episode test. Actual current peaks
are not predicted by this simple calculation; a hardware overcurrent trip may
produce an incomplete test. Keep partial files rather than silently softening
the original sequence or assuming the trip is a firmware torque clamp.

The new entry point cannot select another motor, a nonzero source start, or add
preparation ramp/pre-hold. It exposes duration, final hold, source, level, packet
rate, gains, current/deadline trip values, interface, note and output path.
Feedback/trip internals remain in the existing tested backend.

## Raspberry Pi commands

Sync the new file with the adjacent steering_policy_replay.py,
steering_sine_mit_io.py and replay_targets/. Python standard library only; no
ROS build or ROS nodes are needed.

```bash
cd ~/trackstand_rl/deploy/experiments
python3 steering_initial_transient_replay.py --preview
python3 steering_initial_transient_replay.py --execute --packet-rate 50
# With steering disabled, physically straighten again before the next run.
python3 steering_initial_transient_replay.py --execute --packet-rate 200
```

Each execution automatically saves measurement CSV, plan CSV and JSON under
logs/, named `<timestamp>_<level>_<rate>hz_steering_initial_transient.*`.
The unchanged backend records script/transport hashes; extra JSON arguments
entry_point and entry_point_sha256 identify this wrapper. The default preview
does not open CAN or create files. The six-second duration excludes preflight
and shutdown. RC/kickstand commands do not stop standalone replay.

## Validation

[Six new test methods](../../tests/test_steering_initial_transient_replay.py)
exercise exact recorded startup values, absence of preparation, an optional full
30-second source plan, invalid options, no-CAN previews, both complete default
50/200-Hz virtual-CAN runs, B1 reset from the old clipped-origin case, payload
repetition/equality, steering-only addresses, wrapper provenance and failure cleanup.
This fake motor validates protocol/control flow, not physical tracking or current peaks.

Run the new and existing steering/rear replay regressions:

```bash
python3 -B -m unittest discover -s deploy/experiments/tests -p 'test_*replay.py' -v
python3 -S -B deploy/experiments/steering_initial_transient_replay.py --preview
```

[Saved regression log](unit_tests.log) and [verification](verification.json)
record results, source integrity and the unchanged target/recording/backend boundary.
No hardware, motor/CAN interface or simulation evaluation was run during implementation.
