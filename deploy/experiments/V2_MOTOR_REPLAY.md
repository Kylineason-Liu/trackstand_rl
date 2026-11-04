# V2 recorded motor-target replay

`v2_motor_replay.py` replays one real IsaacLab **Trackstand-Balance-v2** episode
with integrated feedback logging. It uses **steering position control and rear
velocity PI control**, matching v2. It does not run policy inference, ROS or a
balancing controller. No existing deployment or experiment script was modified.

## Reference and what gets replayed

The verified bundle is `replay_targets/v2_26600_no_dr/`, captured from
`model_26600.pt`, no DR, seed 42, environment 0, episode 1. It contains:

- `policy_targets_50hz.csv`: 1500 held targets from t = 0 to 29.98 s, covering 30 s.
- `joint_velocity_reference_1200hz.csv`: 36001 simulated actual joint position
  and velocity samples, including the initial state and t = 30 s.
- `reference_metadata.json` and `manifest.json`: source/configuration, initial
  zeros, gains, rates and hashes. The loader checks these and target provenance.

The authoritative commands are `held_steering_position_target_rad` (after the
training clip/slew) and `drive_velocity_target_rad_s`. No second steering slew,
filter, ramp or torque scaling is applied. Raw actions and scaled steering
requests remain in the reference and outgoing-command logs for diagnosis.
Simulated actual velocities are comparison curves, **not motor commands**.

Default is the first 10 s: the capture shows the initial transient in 0–1 s and
smaller ongoing balancing motion in 5–10 s. Both motors' simulated initial positions
and velocities and rear I are zero. `--duration 30` selects the full episode.
There is deliberately no `--start`: skipping the prefix would omit physical and
integral controller history. Each run starts from zero I only once; target
changes do not clear it.

## Run on the Pi

Pull/copy the new script **and its reference directory**. Python 3 stdlib is
sufficient; no ROS build or NumPy installation is needed. First preview:

```bash
cd ~/trackstand_rl
python3 deploy/experiments/v2_motor_replay.py
```

Preview opens no CAN socket and creates no files. For the physical test,
independently support the frame, make the selected wheel/steering free to move,
physically straighten steering before **every** steering run, and stop all other
CAN writers. An unselected motor must already be independently disabled; the
script does not send it commands. Start with the airborne tests:

```bash
python3 deploy/experiments/v2_motor_replay.py --execute --motors steering
python3 deploy/experiments/v2_motor_replay.py --execute --motors drive
python3 deploy/experiments/v2_motor_replay.py --execute --motors both
```

These are separate runs; let motion stop and re-establish the straight startup
pose before the next. Keep independent support when testing both. The script
does not command kickstands. Ctrl-C requests stop, then disables selected motors
and restores temporary settings. A disabled wheel may continue coasting.

Options include:

```bash
python3 deploy/experiments/v2_motor_replay.py --execute --motors both \
  --packet-rate 200 --duration 30 --fixture airborne --note 'supply/fixture details'
```

`--packet-rate` accepts 50, 200, 400; default 200 matches the tested MIT host-I
architecture. `--query-rate` defaults to 100 Hz for each of A1/A2/A3 per selected
motor. `--interface`, `--output`, `--fixture` and `--note` record/setup the test.
`--fixture supported-loaded` only labels the fixture; it does not supply support
or alter control. Loaded tests require an appropriate independent support setup.

## Control law and clocks

Targets follow the recorded 50 Hz source timeline. Packet, target and I schedules
are anchored to the replay start. A delayed loop selects the current source row,
logs skipped target/packet/I slots and never catches up with a packet burst or
silently stretches a 10 s waveform. A 20 ms lateness threshold is diagnostic only;
there is no small timing-error abort.

Steering ID 2 receives MIT absolute position
`verified_native_zero + (-1) * held_position_rad`, requested Kp = 10 N·m/rad,
Kd = 0.2 N·m·s/rad, target velocity 0 and feedforward torque 0. It performs PD inside
the motor. Symmetric fields cannot encode exact zero; actual encoded values and
gains are logged, rather than assumed exact.

Rear ID 1 receives MIT **position Kp = 0**, requested velocity P/MIT Kd = 0.30 N·m·s/rad.
A host trapezoidal integral updates independently at nominal 200 Hz with
**Ki = 3.3 N·m/rad**, using actual elapsed monotonic time and latest A2 speed.
The first update primes previous error without an I impulse. I persists across
all target changes. Conditional anti-windup rejects increments pushing P+I
farther into saturation and corrects stored integral after clipping; previous
error advances even on a rejected update. Integral torque is feedforward.

Rear FF quantization residual is cancelled in velocity encoding, preserving
intended I. The recorded effective P target includes that correction. As in the
proven MIT velocity test, a small codec allowance bounds I below the 6.28 N·m
reference limit and the actual F0 torque range. Encoded Kd is about
0.3003663 rather than exactly 0.30; queries and packets are asynchronous.

At 50 Hz packets, I still updates at 200 Hz but the motor receives its new feedforward
only at 50 Hz; at 400 Hz packets some I values repeat. These are deliberate rate
comparisons, not identical wire-level controllers. Actual update times/ages and
missed slots are recorded. MuJoCo/Isaac physics rate 1200 Hz is not a requested Pi
CAN send rate; this script does not establish the motor's internal loop rate.

## Signs, origins and persistent settings

The fixed hardware mapping follows current verified deployment settings: steering
ID 2, rear ID 1, gear ratios 1, encoder signs −1 and torque signs −1. A2 is signed
0.01 rpm converted to rad/s; A1 is signed 0.001 A multiplied by queried B0 Kt and
sign. A3 uses signed **multi-turn** counts with 16384 counts/rev, including int32
rollover. Position logs stay continuous relative to each run's fixed origin.
No F1/single-turn angle is substituted for rear displacement.

Selected motors are confirmed disabled before setup. Steering uses the proven
initialization fix: B1 exactly once, fresh near-zero A3, unsaturated inactive F1
consistent with A3, then another A3 bracketing the F1 reference. This prevents
using an initially clipped F1 as the origin. **B1 persists in firmware and is
not restored**; it records the physically straight starting position. A lost B1
reply is not retried. Rear gets a fresh software A3 zero after configuration;
**no rear B1 is sent**. Both original counts and reference details are logged.

Original F0/CD/B3 and B0 are read. The rear F0 velocity encoding range is selected
to cover this prefix with quantization headroom, then verified. CD temporarily
uses a verified 500 ms watchdog. Original settings are journaled with file and
parent-directory fsync **before** persistent writes. No B3 current limit,
C1 gains or acceleration setting is written. Final cleanup sends CF to all
selected motors before waiting/restoring, confirms disable (retained mode 5 needs
inactive F1), then restores and verifies changed F0/CD settings. Unrestored
payloads remain in JSON if cleanup fails; check the reported state physically.

## Measurements, guards and limitations

Each executed run saves three exclusive files in `deploy/experiments/logs/`:
`<timestamp>_<rate>hz_v2_motor_replay.csv`, matching `.json`, and `_reference.csv`.
The CSV includes raw CAN frames, both motor IDs, host timestamps, actual source
index/commands, wire values, gains, current-derived torque, A2 velocity, A3
continuous position, feedback ages, I state/dt/error and missed-slot diagnostics.
The JSON retains provenance, as-found settings, origins, pending restoration,
completion and primary/cleanup errors. Existing captures are never overwritten.

Guards cover driver/CAN faults, malformed or dropped receive data, another writer,
changed encoding ranges, stale feedback and measured speed/effort. By default,
measured speed/current-derived torque is stopped above 120% of the configured
23.9/61.6 rad/s and 5.6/6.28 N·m reference maxima. This explicit 20% measurement-guard
tolerance is not a motor-rating change or a larger commanded limit. A1/A2/F1
feedback age defaults to 0.5 s; driver status to 2 s. `--limit-tolerance`,
`--feedback-timeout` and `--status-timeout` expose these experimental criteria.
There is **no measured steering-angle, target-tracking, tiny rest-speed or
3 ms deadline abort** during motion. An ENOBUFS/EAGAIN send is retried for up to
20 ms before being treated as a bus failure; no CAN interface configuration is
changed automatically.

MIT's F0 ranges are encoding ranges, not an established post-PD torque clamp.
The script does not pretend to reproduce IdealPD's exact 5.6/6.28 N·m instantaneous
torque saturation inside the real motor. Its measured-effort guard is an abort,
not that clamp. Source simulation uses fresh ideal speed and fixed 5 ms I intervals;
hardware uses asynchronous feedback, real elapsed intervals and quantized gains.
Airborne versus loaded motion, friction/cogging and firmware dynamics can differ.
Host timestamps are not synchronized motor acquisition or wire transmission times.
This test measures command/response fidelity; it does not prove successful balance.

Protocol evidence: local GIM CAN Rev 3.09b0 PDF pp. 5–6 (A1/A2/A3), 8 (B1),
16 (CF), 20–22 (F0/F1/MIT). The existing successful steering replay origin fix and
`drive_mit_velocity_test.py` provide hardware-tested initialization/sign/codec
precedents; this new two-motor replay itself has only offline validation so far.

Offline tests (all CAN/socket operations mocked):

```bash
cd /tmp
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s ~/trackstand_rl/deploy/experiments/tests -p test_v2_motor_replay.py -v
```
