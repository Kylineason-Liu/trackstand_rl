# Recorded-policy steering replay

The separate rear C0 replay is now available. See [POLICY_REPLAY.md](POLICY_REPLAY.md)
for the procedure for both motors; this steering script still commands only ID 2.

`steering_policy_replay.py` sends a fixed, recorded steering reference through
MIT and collects feedback in the same process. This is a supported actuator
experiment, not policy inference or a balancing attempt. Only the selected
steering motor (default ID **2**, native/joint sign **-1**, gear **1**) is commanded.
The rear motor is neither commanded nor disabled by this script.

## Actual target files

`replay_targets/` contains three CSVs exported from the October 25 model_2500 DR
captures, env 0, **first completed episode**:

- `2500_isaac400_env0_episode1.csv`
- `2500_mujoco400_env0_episode1.csv`
- `2500_mujoco800_env0_episode1.csv`

Each contains **1,500 real samples / 30 seconds**: raw steering/rear actions,
steering action × 1.57 rad, rear action × 6.28 Nm, the held steering target entering
PD, and submitted rear motor torque. Friction is not baked into the position
reference. `manifest.json` records source/CSV hashes and provenance; known CSVs
with mismatching hashes are rejected. Full source precision is retained.

For these three first episodes, scaled and conditioned steering columns are
**identical throughout**, not merely in the selected detail window. They still
represent separate pipeline stages. Selecting a level does not invent a
difference, and identical results between these levels cannot establish that
the limiter would never matter on another trajectory.

Default replay selects source **12.50–17.50 s**, with a **1 s preparation ramp,
1 s pre-hold, 5 s unchanged core and 1 s final hold**. References are canonicalized
to float32 exactly as in the completed simulator replay. Tests verify the full
400-sample plan against all three existing eight-second simulator inputs.
Preparation/hold rows are labeled separately from real episode rows. There is
no new sine generation, inference, core interpolation, amplitude scaling, or
response shifting. Default Isaac reference spans approximately **0.0253–1.3254 rad**
including preparation; it is not a small-angle test around zero.

## What each mode tests

| Selection | Input to the motor packet encoder |
| --- | --- |
| `--level scaled` (default) | Scaled raw position, directly |
| `--level conditioned` | Recorded position after the source software's clip/slew, directly |
| `--level scaled --v1-conditioning` | Scaled position through ±1.57-rad clip and 23.9-rad/s target slew, once per 20 ms |

Conditioning twice is rejected. `--v1-conditioning` starts its software limiter
at zero for this new test, including preparation; it does not reconstruct hidden
prior limiter state for an arbitrary mid-episode interval. For an exact recorded
motor-reference comparison, use `conditioned` without another limiter pass.
Direct replay rejects targets outside ±1.57 rad; it never silently clamps them.
Changing the preparation, selected window, gains or processing requires the
same changed test in simulation before claiming a matched comparison.

## Raspberry Pi procedure

Copy/update **both Python files** (`steering_policy_replay.py` and the adjacent
`steering_sine_mit_io.py`) plus the **`replay_targets/` directory** on the Pi.
Execution needs Python >=3.10 and an operational Linux SocketCAN `can0`; no ROS
nodes, ROS package build, NumPy, python-can, or separate feedback collector.
The existing sine file supplies shared protocol/CSV helpers; it is imported,
never launched, and its sine function is not called.

Independently support the bike, initially with the front wheel airborne, steering
physically straight and free over the full requested travel. Stop all other
motor writers/query tools. Independently verify rear disabled, motor current
protection and emergency stopping. Airborne steering can still hit stops or
tighten cables. This test should follow a small-angle hardware check. Do not
raise abort settings merely to complete a trace without verifying protection.

From repository root:

```bash
# Preview only: validates the real file, no CAN access or output files.
python3 deploy/experiments/steering_policy_replay.py

# Actual policy sequence: one new reference /20 ms, one packet /20 ms.
python3 deploy/experiments/steering_policy_replay.py --execute \
  --level scaled --packet-rate 50

# Physically straighten again with steering disabled before restarting.
# SAME reference /20 ms, resent unchanged four times /20 ms.
python3 deploy/experiments/steering_policy_replay.py --execute \
  --level scaled --packet-rate 200
```

Keep `--level` and every other condition fixed for the packet-rate comparison.
Use `--level conditioned` to select the other recorded level. To compare the
fast MuJoCo reference after validating the initial test, select:

```bash
python3 deploy/experiments/steering_policy_replay.py --preview \
  --targets deploy/experiments/replay_targets/2500_mujoco800_env0_episode1.csv \
  --level conditioned --packet-rate 200
```

`--start`, `--duration`, `--ramp`, `--pre-hold`, `--post-hold` are seconds on the
20 ms grid. Files cannot cross episode boundaries or be padded after their end.
`--fixture supported-loaded` and `--note '...'` record actual test conditions;
they do not alter motor commands. Repeat supported ground-loaded tests separately.
Do not mix loaded and airborne traces as identical fixtures.

## Commands, feedback and stopping

- Policy-reference updates stay **50 Hz**. Packet rates are **50 or 200 Hz**;
  at 200 Hz each four-packet group has identical payload bytes. No intermediate
  targets are generated, and optional slew does not advance at packet rate.
- Nominal native MIT gains are **Kp 10 Nm/rad, Kd 0.2 Nm s/rad**, configurable with
  `--kp/--kd`. Requested velocity/feedforward torque are zero. Actual encoded
  gains, target and near-zero quantized velocity/torque are logged. Compare these
  decoded values when refining simulation, rather than assuming exact zeros/gains.
- These are DR-origin commands tested with nominal gains, matching the earlier
  simulator replay. Original source episodes had their own sampled DR gains.
- Startup: **CF** confirms disabled/fault-free; read **F0/B0/B3/CD**, **A3/A2/A1/F1/AE**;
  record A3 and F1 software references; verify stationary and range compatibility.
  There are **no B1 origin resets or F0/B3/CD configuration writes**. Native MIT
  position is `first_F1_position + sign * joint_target`. If that absolute path
  exceeds the existing F0 range, refuse motion without modifying settings.
- A3/A2/A1 queries target **100 Hz each** at either packet rate. AE targets 5 Hz;
  B0/F0 target 1 Hz. Replies are asynchronous/sequential, not simultaneous.
  Query deadlines and actual timestamps are logged; inspect achieved rates.
- Kernel **SO_TIMESTAMPNS_NEW** receive times preserve queue age. Missing/bad
  timestamps, clock steps, RX overflow, malformed replies, observed conflicting
  writers, F1 faults/unexpected exit and stale feedback stop replay.
- Default feedback timeout **100 ms**, packet lateness threshold **3 ms**.
  Missed schedules abort rather than dropping source targets or bursting to catch
  up. These are software checks, not hard-real-time guarantees.
- Default **measured-current abort 3 A** and **measured-angle abort 1.7 rad** are
  conservative experiment stop thresholds, not motor ratings or torque clamps.
  Every received sample is checked, including transient excursions. Change with
  `--current-abort-a/--angle-abort-rad` only against independently verified limits.
  F0 and the read B3 value do not establish MIT's actual total torque protection.
- Completion, faults, Ctrl-C and SIGTERM attempt CF and check fresh CF/AE disabled
  replies. An unconfirmed stop fails the run. Disable removes actuation; it is
  not a mechanical brake. Crash/SIGKILL/power loss cannot guarantee software cleanup.
  Existing watchdog settings are read, not enabled or changed by this tool.

## Saved data

Every executed run automatically creates exclusive files under `logs/`:

- `<timestamp>_<level>_<rate>hz_policy_replay.csv`: sparse, individually timestamped
  TX/RX, exact bytes, source/plan indices, selected and decoded commands, A3
  unwrapped position (rad), A2 velocity (rad/s), A1 current (A), current-derived
  torque (Nm), F1 fields, B0, status, and query timing. Missing values remain blank/NaN.
- Same stem `_plan.csv`: exact 50 Hz reference used, including labeled preparation
  and held phases. Use this file/recorded encoded commands for simulator comparison.
- Same stem `.json`: file/code hashes, settings, software zeros, F0/B3/CD values,
  fixture note, completion/error and disable-confirmation state.

Target rows identify `ramp`, `pre_hold`, `core` and `post_hold`. Feedback/query
rows use phase `motion`; select their core interval by `replay_time_s` (default
2–7 s), rather than filtering feedback rows by `phase=core`.

Torque is **signed A1 current × measured B0 Kt**, not an independent shaft sensor.
Compare it with simulated motor actuator output; Isaac solver joint-axis effort
is a different measurement. Times are host send-call and kernel host-receive
times, not motor acquisition or on-wire transmit timestamps. No hardware run or
physical motor performance is certified by the offline tests.

PC verification and export to a **new** folder:

```bash
python3 -B -m unittest discover -s deploy/experiments/tests -p test_steering_policy_replay.py -v
python3 deploy/experiments/export_policy_replay.py --output-dir /tmp/new_policy_replay_targets
```
