# Identical-target tests on the real bike

Run the motors **separately**, with the bike independently supported. These
standalone tests replay recorded model_2500 commands; they do not run inference
or balance the bike. No ROS node or package build is needed.

| Script | Motor | Reference | Default packet rate | Other motor |
| --- | --- | --- | --- | --- |
| `steering_policy_replay.py` | ID 2, MIT | Position in rad | 50 Hz | Neither commanded nor disabled |
| `steering_initial_transient_replay.py` | ID 2, MIT | Position from source time zero | 50 Hz | Neither commanded nor disabled |
| `drive_policy_replay.py` | ID 1, C0 | Torque in Nm, converted to current with measured B0 | 200 Hz | Neither commanded nor disabled |

All nominally update references at **50 Hz**. Selecting `--packet-rate 200` repeats the
same encoded reference four times per 20 ms; it does not generate new targets.
The actual send and feedback timestamps are logged. Default mode is preview;
only `--execute` opens CAN and permits motor actuation.

Replay scheduling tolerates **20 ms per-send lateness by default**, rather than
the old 3 ms. Each next packet is scheduled one nominal packet interval after the
previous send completes. All recorded targets and repeated packets are preserved;
there are no catch-up bursts. Host delays therefore stretch the actual replay
and can reduce its achieved rate. Compare feedback against actual logged send
times; target_time_s remains the nominal source-plan clock. JSON records both
nominal_motion_duration_s and actual_motion_duration_s. Motor fault, current,
speed, temperature, feedback-age and initialization checks are unchanged.

Use **--max-lateness-ms 50** to allow more jitter, or **--max-lateness-ms 3** for
the old tighter trip. The value may exceed a packet interval, including at
200 Hz. This limit does not replace the independent stale-feedback checks.

## Files to transfer

Keep these three files adjacent under `deploy/experiments/` on the Pi:

- `drive_policy_replay.py`
- `steering_policy_replay.py`
- `steering_sine_mit_io.py`

Also copy `replay_targets/`, including its manifest. The rear script imports
shared CSV/SocketCAN helpers from the other two files; importing them starts
neither their motion loop nor any ROS node. Python >=3.10, standard library,
Linux SocketCAN and the existing working `can0` configuration are required.
NumPy is needed only for the PC exporter/tests, not the Pi runtime.

The three provided CSVs are actual env-0 first episodes from Isaac400,
MuJoCo400 and MuJoCo800 DR evaluations. Each has 1,500 samples / 30 seconds and
both motor-command levels. Source and CSV hashes are preserved; altered known
files are rejected. See [the manifest](replay_targets/manifest.json).

## Preparation on the Pi

1. Stop all motor nodes, policy/launch processes, other experiment scripts and
   querying CAN tools. Do not run motor replay scripts simultaneously.
2. Secure the bike independently. Rear testing requires the rear wheel airborne,
   unobstructed and initially stopped. Steering testing requires straight-ahead
   startup and free travel over the requested range, initially with the front
   wheel airborne. Verify physical current protection and emergency stopping.
   Steering `--execute` now sends **B1 once**, storing the straight position as
   its persistent encoder origin, then verifies fresh A3/F1 near zero before
   replay. It preserves F0 ranges. Rear replay does not reset either origin.
3. Disable the other motor independently. For example, with other writers
   stopped, `cansend can0 102#CF` requests steering disabled before rear testing;
   `cansend can0 101#CF` requests rear disabled before steering testing. Verify
   the stopped/disabled condition; sending a frame alone is not confirmation.
4. **RC/kickstand commands do not stop these standalone scripts.** Normal ROS
   arming/fault/kickstand logic is not running. Ctrl-C requests software cleanup;
   keep physical emergency stopping available. Zero torque and CF are not brakes.

## Commands

From the Pi's repository root:

```bash
cd ~/trackstand_rl

# Check files and intervals; these two commands do not access hardware.
python3 deploy/experiments/drive_policy_replay.py --preview
python3 deploy/experiments/steering_policy_replay.py --preview

# Rear first: short recorded interval, 50 Hz-held references /200 Hz C0.
python3 deploy/experiments/drive_policy_replay.py --execute \
  --level scaled --packet-rate 200 --duration 0.20

# After inspecting the short test and confirming the wheel is stopped:
# default recorded five-second core, with two initial zero-torque seconds.
python3 deploy/experiments/drive_policy_replay.py --execute \
  --level scaled --packet-rate 200

# Optional packet-rate comparison: SAME source, level and physical conditions.
python3 deploy/experiments/drive_policy_replay.py --execute \
  --level scaled --packet-rate 50

# Separately: disable rear, physically straighten steering, verify free travel.
python3 deploy/experiments/steering_policy_replay.py --execute \
  --level scaled --packet-rate 50

# Physically straighten again with steering disabled before this new run.
python3 deploy/experiments/steering_policy_replay.py --execute \
  --level scaled --packet-rate 200
```

Do not repeat an aborted run without reading its error/JSON and checking the
physical cause. A speed/current threshold abort is an incomplete experiment,
not automatically a motor fault or proof of poor torque tracking. The wheel
can accelerate rapidly even if current tracks correctly. Do not increase abort
thresholds just to finish a trace; use independently validated hardware limits.

To use the other recorded command level, change rear to `--level submitted`,
or steering to `--level conditioned`. Keep every other condition fixed. In these
three provided episodes, both rear levels are identical throughout, as are both
steering levels; equal results between levels are therefore expected.

To select a different source:

```bash
python3 deploy/experiments/drive_policy_replay.py --preview \
  --targets deploy/experiments/replay_targets/2500_mujoco800_env0_episode1.csv \
  --level submitted --packet-rate 200
```

Both scripts accept `--start` and `--duration` in seconds, on the 20 ms grid.
The default source core is **12.50–17.50 s**. Changing sources/windows may change
peak torque/current substantially; preview first. Record actual supply/firmware
and fixture conditions with `--note '...'`. Steering additionally has
`--fixture airborne|supported-loaded`; the rear test is intended airborne.

## Steering initial-transient replay

Use [steering_initial_transient_replay.py](steering_initial_transient_replay.py)
to replay the **first five seconds from source time zero**, followed by a one-second
final hold. It uses the same verified B1/A3/F1 startup, MIT gains 10/0.2, 100-Hz
A3/A2/A1 queries and integrated CSV/JSON/plan logging as steering_policy_replay.py.
There is **no preparatory ramp or pre-hold**; the first recorded target is applied
directly from straight. Only steering ID 2 is addressed; the rear must already be
disabled independently. This does not change the old scripts or their 12.5–17.5 s
default core interval.

Keep the new script adjacent to steering_policy_replay.py and
steering_sine_mit_io.py, with replay_targets/ present. No ROS build is needed.
The same supported, front-wheel-airborne/free, physically straight setup and
stopped-writers requirements above apply before **each** execution.

```bash
cd ~/trackstand_rl/deploy/experiments
python3 steering_initial_transient_replay.py --preview
python3 steering_initial_transient_replay.py --execute --packet-rate 50
# Disable/straighten again before the separate second run:
python3 steering_initial_transient_replay.py --execute --packet-rate 200
```

The default Isaac400 file begins at **0.357374 rad (20.476 degrees)** and spans
**0.357374–1.339667 rad** over the selected five seconds. Nominal Kp times the
initial stationary position error is approximately **3.574 Nm**, before real
motor dynamics/current control. This is a stronger input than the previous
artificial ramp. The **3 A measured-current trip** remains; the packet-lateness
trip is now **20 ms** by default. A run may still abort; keep partial data rather
than interpreting an incomplete trace as proof of failed tracking. These trips
are not firmware current clamps. No new measured-angle abort is added.

Use --duration to change how many recorded seconds are played from zero, and
--post-hold 0 to omit the final hold. Other exposed options include --targets,
--level, --kp/--kd, --current-abort-a, --max-lateness-ms, --interface, --note and
--output. Unexposed backend options cannot redirect this entry point to the rear
motor or insert a preparation ramp. At 200 Hz targets still change only at 50 Hz.

Three files are saved automatically under logs/ with a
`<timestamp>_<level>_<rate>hz_steering_initial_transient` stem. JSON arguments
include entry_point and entry_point_sha256 for this wrapper; script_sha256 and
transport_sha256 continue to identify the unchanged backend/helper. Front-wheel
replay only: no rear initial-transient implementation or motor command is added.

## Rear command and feedback details

The rear flow is:

```text
recorded scaled_drive_nm OR submitted_drive_nm, held for 20 ms
→ I_requested = torque_sign * torque_Nm / measured_B0_Kt
→ signed int32, round(I_requested * 1000), little endian
→ 101#C0<4 current bytes>
```

Encoder and torque signs default to **-1**, gearing to **1**, following the
verified bike configuration. Requested/encoded currents and resulting quantized
torques are logged. There is **no host damping, extra scaling, friction
compensation, torque slew or silent clipping**. Plans exceeding ±6.28 Nm or the
configured command-current bound are rejected in full before any C0 is sent.

This matches the v1 rear feedforward contract. This standalone replay does not
execute ROS motor code or modify deployment configuration. It diagnoses motor
response rather than certifying the whole current ROS command pipeline.

Rear preflight confirms CF disabled/fault-free, reads **B0/B3/B4/CD**, establishes
an A3 software zero, and reads **A3/A2/A1/AE**. B0 is required; no guessed Kt is
substituted. Initial wheel speed must be within 0.2 rad/s. No encoder origin,
gains, current-slew or position/speed limit settings are changed.

Protocol Rev.3.09b0 establishes:

- **C0/A1:** signed current in 0.001 A; C0's reply is measured current, not
  an acknowledgement interpreted as the desired current.
- **B3:** maximum current in **position/speed modes**. It is read/logged as
  `position_speed_current_limit_a`, not used as an assumed C0 current cap.
- **B4:** current-mode current slew in 0.001 A/s. It is read/logged and left
  unchanged; the protocol excerpt does not establish the meaning of zero.
- **CD:** volatile communication watchdog. The rear tool enables/shortens it
  to at most `--watchdog-ms` (default 500 ms), retaining existing action bits
  and never lengthening an already enabled shorter timeout. It verifies write
  acknowledgement and independent readback. This is a communication watchdog;
  other traffic may refresh it. It is not proven command-age protection.

The original CD setting is registered before a write and restored only after
fresh CF/AE responses confirm disabled. Failed/lost setup acknowledgements also
trigger rollback after disable. If disable cannot be confirmed, the changed
watchdog is retained. Crash/SIGKILL can leave it changed. Restoration errors are
reported separately from disable confirmation. No AF fault-clear command is sent.

## Rates, aborts and intervals

Rear A3/A2/A1 queries target **100 Hz each**, AE 5 Hz, B0/B4/CD 1 Hz, independent of
the chosen C0 packet rate. C0 replies and A1 queries remain separate freshness
checks. Query/receipt times are not simultaneous motor acquisition times.

Rear defaults, explicitly experimental rather than motor ratings:

| Setting | Default | Behavior |
| --- | ---: | --- |
| `--command-current-limit-a` | 1.5 A | Reject a plan if any encoded C0 current exceeds it; never clips |
| `--current-abort-a` | 2 A | Stop on any measured current excursion |
| `--speed-abort-rad-s` | 10 rad/s | Stop on a measured speed excursion; does not regulate speed |
| `--temperature-abort-c` | 60 °C | Stop on an AE temperature excursion |
| `--feedback-timeout` | 0.1 s | Stop on stale A3/A2/A1/C0 feedback |
| `--max-lateness-ms` | 20 ms | Abort beyond this send lateness; stretch replay without skipping/bursts |

The 10 rad/s speed abort is a conservative initial bench threshold near the
previous rear sweep's tested range (up to 8 rad/s), not the simulator's 61.6 rad/s
speed constraint. Free-wheel replay may stop early on this threshold while
torque tracking is healthy. Keep partial recordings for analysis.

Rear nominal default length is **7 seconds**: 2 s zero torque + 5 s core, then CF.
These **350 target samples exactly match** the first seven seconds of the saved
simulator replay for all three sources and both rear levels. Rear stops here
because holding a nonzero final torque can continue accelerating a free wheel.
`--post-hold 1` explicitly adds the original last-torque hold and reproduces all
400 samples of the prior eight-second input, but is not the initial hardware test.
Steering retains its nominal eight-second ramp/hold/core/hold plan. Actual wall
duration may increase under host jitter. Comparisons
must use the same declared input interval and fixture, not conflate cleanup phases.

## Output to return for analysis

Each executed run automatically creates three exclusive files under
`deploy/experiments/logs/`: a measurement `.csv`, a `_plan.csv` and a `.json`.
Return all three, including failed/partial runs. Check JSON `completed`,
`disable_confirmed`, `watchdog_restored`, `error`, B4 and the saved settings.
Use `--output path.csv` for another unique output path.

Rear CSV includes source index/time, raw/scaled/submitted/requested torque,
encoded C0 current/torque, exact bytes, C0/A1 measured current and signed
**current × B0** torque, unwrapped A3 position, A2 speed, B4, status, query timing
and kernel receive queue age. Generic legacy transport columns are retained;
`can_id_hex` and `payload_hex` provide clear C0 frame fields.

Rear core feedback is selected by `replay_time_s` **2–7 s** for the default run;
target rows carry `phase=core`, while RX rows use `phase=motion`. Do not substitute
cached/interpolated samples for fresh measurements. Current-derived torque is
not independent shaft torque. Compare it with simulated **motor actuator torque**,
not Isaac solver joint-axis effort. Do not subtract 0.106 Nm friction from A1 × Kt
again: mechanical losses affect wheel motion separately.

See [steering details](STEERING_POLICY_REPLAY.md) and
[rear implementation verification](reports/20261026_drive_policy_replay/README.md).
