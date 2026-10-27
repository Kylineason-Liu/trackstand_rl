# Replay timing relaxed for Raspberry Pi

Implemented 2026-10-27 for the three live recorded-target entry points:

- [steering_policy_replay.py](../../steering_policy_replay.py)
- [drive_policy_replay.py](../../drive_policy_replay.py)
- [steering_initial_transient_replay.py](../../steering_initial_transient_replay.py)

## Behavior

Default **--max-lateness-ms is now 20**, previously 3. Twenty milliseconds is
one nominal 50-Hz target interval, chosen as a bench-replay tolerance, not a
motor specification or a guarantee about Pi scheduling. Any positive finite
override is accepted, even if larger than a 5 ms packet period at 200 Hz.
Use --max-lateness-ms 50 for more tolerance, or 3 for the tighter lateness trip.

Increasing only the threshold on the old fixed clock could send overdue packets
back-to-back and compress target intervals. Both motor loops now schedule the
next packet one nominal packet period **after the preceding send call and its
logging return**. This also leaves time for state queries if a send/logging call
itself runs slowly. Every recorded target and every repeated packet is preserved,
with no interpolation, sample skipping or catch-up bursts. The final target is
held for at least one full packet period; completion depends on the packet count
and final hold rather than the old fixed wall-clock deadline.

Actual duration can increase and achieved rates can fall below nominal 50/200 Hz.
At 200 Hz each target still has four consecutive copies; the actual time between
different targets may exceed 20 ms. This changes replay timing, not target values.
Use actual command timestamps for tracking analysis; compare exact-rate simulation
with that limitation or replay the measured hardware schedule in simulation.

CSV command_lateness_s measures lateness relative to the current send-relative
deadline; command_interval_s and replay_time_s retain actual host timestamps.
target_time_s and source_time_s retain nominal plan/source coordinates.
skipped_command_ticks is explicitly zero for recorded commands, since no target
is skipped even when a user allows a longer stall. JSON adds:

- command_schedule = send_relative_no_catchup
- rate_labels_are_nominal = true
- nominal_motion_duration_s and actual_motion_duration_s, including partial runs

Deadline errors include measured lateness and the configured limit. The last
field above ends before shutdown; it excludes initialization and cleanup.

## Unchanged

Actual target CSVs, scale/clip/conditioning, packet payloads, signs/units/gains,
B1/A3/F1 zero validation, current/speed/temperature/fault trips, feedback freshness,
exclusive output files and disable/restore cleanup are unchanged. Stale feedback
can still stop the replay during a larger host stall. No motor or deployment-node
configuration was changed, and no real CAN/motor operation was run here.

The sine generator is not a recorded-target replay and already spaces commands
without this 3 ms abort; its runtime is unchanged. Exporters and archived runtime
copies are not live replay entry points and remain unchanged.

## Pi usage

Sync all three live scripts and keep steering_sine_mit_io.py adjacent. The same
commands work with the new 20 ms default; no ROS build is required:

```bash
cd ~/trackstand_rl/deploy/experiments
python3 steering_initial_transient_replay.py --execute --packet-rate 50
# Disable/physically straighten steering again before a separate run.
python3 steering_initial_transient_replay.py --execute --packet-rate 200
```

For example, add --max-lateness-ms 50 to either command for a wider lateness
window. Front wheel remains airborne/free, bike independently supported, rear
disabled and all other writers stopped. RC/kickstand controls do not stop these
standalone tests. Keep partial CSV/plan/JSON files if another condition aborts.

## Verification and historical provenance

**56 recorded-replay regression methods passed**, including six new methods.
Injected scheduling delays exceeding 3 ms now complete at both packet rates for
steering and rear. Tests verify complete source order, repeated payloads, no
inter-packet burst, the full final hold, extended duration and rear watchdog
restoration. Slow send work exceeding one 200-Hz period also completes without
starving queries or dropping the end of the sequence. Over-limit/explicit 3 ms
trips still disable motors, and other existing fault/current/staleness tests pass.

The October 27 successful steering hardware records used the old strict runtime.
Its exact backend/helper were saved to that report's
[runtime_snapshot/](../20261027_steering_replay_after_zero_fix/runtime_snapshot/).
The analyzer now validates those archived hashes, rather than the changed live
script. Raw files, statistics and plots are unchanged. Earlier October 26 reports
already used their own archived runtime and are untouched. The initial-transient
implementation report is marked historical where it describes the old 3 ms trip.

See [regression log](unit_tests.log), [cold previews](previews.log),
[historical-data validation](historical_validation.json) and
[preservation checks](verification.json). Reproduce the software checks from root:

```bash
python3 -B -m unittest discover -s deploy/experiments/tests -p 'test_*replay.py' -v
python3 -S -B deploy/experiments/steering_initial_transient_replay.py --preview --packet-rate 200
```
