# Rear C1 replay of a real simulation episode

This experiment checks whether C1 can reproduce the **actual simulated rear-wheel
velocity** of v1 **model_2500**, including startup and settling. The v1 policy
outputs torque, not velocity: this deliberately uses the resulting motion as a
new C1 velocity reference. It does not validate a v2 policy or establish that a
simulated P controller matches C1 internally.

## Prepared reference

`replay_targets/2500_isaac_velocity_startup.csv` contains simulation seconds
**0–10**, env 0, first episode, seed 42, **DR enabled**. There is no episode reset
or stitching in this window. Times and velocities are in **seconds and rad/s**;
positive rear velocity means **forward**. Native C1 and encoder signs are −1.

- Reference CSV: 501 actual state samples at 50 Hz, including the final endpoint.
  The endpoint defines the duration; it is not an extra held command interval.
- `_simulation.csv`: all 4,001 state samples, including t=0, at 400 Hz.
- `.json`: checkpoint/source hashes, timing, units, selected environment and DR.
- `_env.yaml`: effective configuration of the fresh simulation recording.

[Preview startup and settling](replay_targets/2500_isaac_velocity_startup_preview.png).
The native trace peaks at 2.484 rad/s; final 8–10 s RMS speed is 0.0513 rad/s.
Fresh velocities exactly match the independent archived evaluation for the same
model/seed/env. [Verification](replay_targets/2500_isaac_velocity_startup_verification.json)
records source/data hashes, 44 passing CPU tests and end-to-end virtual CAN runs
at both packet rates. No real motor run has been performed by this implementation.

The command at each 20 ms boundary is held until the next boundary. At 200 Hz it
is resent four times; there is no interpolation, new sine generation, amplitude
scaling, friction compensation or hidden clipping. Sampling a 400-Hz response at
50 Hz discards some within-interval changes; the full-rate curve is retained so
that this difference can be inspected separately from hardware tracking error.

## Run on the Raspberry Pi

Support the frame, lift the rear wheel clear of the ground, leave it free to
rotate, independently disable steering, and stop all other motor CAN writers.
Start with this airborne test; normal ground load is a separate experiment.

Pull/copy the reference CSV and its JSON, this script, and the adjacent
`drive_velocity_sine.py` helper. No ROS node, separate recorder, package build,
NumPy or other third-party package is needed on the Pi.

```bash
cd ~/trackstand_rl

# Inspect the reference and timing; no CAN or output files.
python3 deploy/experiments/drive_velocity_replay.py

# Run and automatically record all feedback.
python3 deploy/experiments/drive_velocity_replay.py --execute
```

The sequence is **C1 zero speed for 1 s → 10 s reference → C1 zero speed for
1 s → CF disable**. Preparation/shutdown are labelled separately from the replay.
Zero-speed C1 is active braking, not disabled coasting; no tight rest-speed
criterion extends the test. The first A3 multi-turn reading is the software
position origin. The core-start position/speed snapshot and A3 age are also
recorded; velocity is never zero-offset corrected. No B1/encoder reset occurs.

Optional flags: `--packet-rate 50`, `--targets file.csv`, `--pre-zero 1`,
`--post-zero 1`, `--interface can0`, `--note 'actual fixture and supply'`, and
`--output new.csv`. Default packets are **200 Hz**; targets remain **50 Hz**.
Different starting wheel angles are allowed and their A3 origins are recorded.

## Logging and retained checks

Exclusive timestamped files under `logs/` are saved automatically:

- `.csv`: every command/query/reply, actual host timestamps, held/encoded C1
  target, A2 velocity, A1 current and current-derived torque, A3 displacement,
  AE supply/temperature/mode/fault, packet gaps/lateness and skipped slots.
- `.json`: completion/error/disable status, phase origins, original settings,
  checkpoint/reference/helper hashes, source provenance and initial A3 count.
- `_reference.csv`: exact copy of the reference used for that run.

A2/A1/A3/AE queries are nominally **200/100/50/10 Hz**. Feedback is asynchronous:
host receipt time is not a known motor acquisition time. Torque is **A1 × B0 Kt**,
not an independent shaft-torque measurement. B8/B9/B3/B5 are read/logged only.

There is **no angle/travel, tracking-error, small-current, startup-rest or
packet-lateness abort**. Late packet slots are logged/skipped rather than sent
in catch-up bursts or stretching the reference. Defaults permit **1 s** without
fresh velocity/current and **2 s** without status, configurable using
`--feedback-timeout`/`--status-timeout`. The existing helper still stops on speed
above **61.6 rad/s**, current-derived torque above **6.28 Nm**, driver faults,
CAN/log failures, malformed feedback or competing rear/broadcast writers.
These are host checks, not guaranteed physical caps between samples. The CD
watchdog is temporarily enabled/shortened to at most 500 ms and restored only
after CF confirms disable. B3 is never written or compared against a requested
current ceiling. Ctrl-C also attempts disable/restore and retains partial data.

## Compare after the hardware run

Copy the run's three files back into `deploy/experiments/logs/` on this PC:

```bash
python3 deploy/experiments/drive_velocity_replay.py \
  --plot deploy/experiments/logs/<timestamp>_200hz_drive_velocity_replay.csv
```

This creates `_comparison.png`, `_comparison.svg` and `_comparison.json` beside
the log. Panels show velocity (simulation/C1 target/real A2), actual current-derived
torque, and displacement. All share the core-start clock; there is no fitted
time shift, smoothing or amplitude adjustment. The plot uses the full 400-Hz
simulation CSV when its local hash matches the manifest, otherwise the copied
50-Hz state samples. Metrics state which source was used and compare fresh A2
both to the last transmitted target and to the simulation curve. Only the latter
comparison interpolates saved simulation states at A2 receipt times; CAN targets
are never interpolated. Partial runs remain labelled partial.

Position is diagnostic: replay commands velocity, not position, and changes the
loading from rolling-bike simulation to an airborne wheel. Do not interpret every
displacement difference as a velocity-control fault. Core-start A3 zero is a
timestamped pre-command snapshot rather than an invented simultaneous reading.

## Make another simulation recording on the PC

The prepared reference already exists. To create a new file with the unchanged
v1 environment, rather than overwrite it:

```bash
~/IsaacLab/_isaac_sim/python.sh deploy/experiments/record_drive_velocity.py \
  --dr --duration 10 --num-envs 64 --env-index 0 --seed 42 \
  --output deploy/experiments/replay_targets/another_velocity_episode.csv
```

The headless recorder hooks post-physics state updates **before automatic
resets**, includes the true initial state at t=0 and stops at the first selected
environment termination. Joint position is unwrapped at the physics rate. It
does not change v0/v1/v2, actuator gains/friction, physics rate, checkpoint or
termination settings. `--checkpoint` selects a compatible v1 checkpoint;
`--duration`, `--env-index`, `--seed`, `--device` and opt-in `--dr` are configurable.

It can also export a previously audited capture without starting Isaac Sim:

```bash
python3 deploy/experiments/record_drive_velocity.py \
  --from-capture isaaclab_train/reports/20261025_2500_dr_comparison/isaac400/motor_samples.npz \
  --output deploy/experiments/replay_targets/archived_velocity_episode.csv
```

Archive mode retains the archive's own DR, seed and timing, verifies its
checkpoint hash, and ends before a reset can contaminate the episode.
