# Deployment Telemetry Contract

Record `/trackstand/motor_state`, `/trackstand/motor_control`, `/trackstand/motor_event`, `/trackstand/policy_timing`, and `/trackstand/policy_event` alongside the existing policy, sensor, and command topics. Position, velocity, and torque feedback share `motor_state`; there is no separate `motor_torque` publisher. This page describes the signals emitted by the current deployment. Schema constants live in `trackstand_deploy/telemetry.py`.

## Unified motor feedback

All three motor nodes publish `/trackstand/motor_state` as `std_msgs/msg/Float64MultiArray` with `layout.dim[0].label = motor_state_v2:<comma-separated MOTOR_STATE_FIELDS>`. Its seven fields are:

```text
[steering_position_rad, drive_position_rad,
 steering_velocity_rad_s, drive_velocity_rad_s,
 unix_time_s, steering_torque_nm, drive_torque_nm]
```

The old five-field prefix is unchanged; the policy consumes only indices 0–3. Torque fields are signed N·m estimates from `configured_torque_sign × B0_torque_constant × measured_A1/C0_Q_axis_current`, not commanded torque or an independent shaft-torque sensor. F1's separate native MIT torque remains in `motor_control`. Torque is not clipped to command limits.

State publication uses a nominal **100 Hz timer for MIT**, **200 Hz for the legacy C0/C2 nodes**, and requires new, fresh encoder pairs from both motors. Each torque field carries that motor's latest valid current-derived estimate while its independent receipt age is at most `feedback_timeout_s` (default 50 ms); missing/stale torque is `NaN`, not zero. Cached torque may repeat between current replies. Torque availability does not gate position/velocity publication or policy observations. `data[4]` is Unix publication time, not acquisition time; positions, velocities, and torques are asynchronously sampled snapshots. `motor_status` retains per-motor torque receipt ages at **100 Hz for MIT**, **2 Hz for legacy C0/C2**, using host receipt time for C0/C2 and kernel CAN arrival time for MIT. MIT `/motor_fault` is published on that same 100 Hz timer; detection and CAN disable remain in the 200 Hz safety loop.

## Numeric telemetry

`/trackstand/motor_control` and `/trackstand/policy_timing` are `std_msgs/msg/Float64MultiArray`. The ordered field names appear in `layout.dim[0].label` as `motor_control_v1`, `motor_control_position_v2`, or `motor_control_mit_v1` followed by `:<comma-separated names>`, or as `policy_timing_v1:<comma-separated names>`. Match those names to `data` by index. Fields ending `_ms`, `_s`, `_rad`, `_rad_s`, `_nm`, and `_a` use milliseconds, seconds, radians, radians per second, newton metres, and amperes respectively. Flags are 0/1. NaN means unavailable or not applicable; it does not mean zero. Both messages carry `sequence`, a ROS tick timestamp in seconds, and a process `time.monotonic()` tick timestamp. These timestamps are from the host, not from the motor firmware. Monotonic time can be compared across the two nodes on the same Raspberry Pi boot.

The motor control record is sampled at nominal **200 Hz for MIT**, **50 Hz for legacy C0/C2**, including while disabled. The backend-neutral gate accepts a selected rate; legacy callers retain its 50 Hz default. It limits numeric-message construction and publication only, not CAN commands or transition detection. The first tick publishes immediately, then a fixed 5 ms (MIT) or 20 ms (legacy) phase selects the first available control tick at each deadline. Missed slots are skipped, never replayed in a catch-up burst. An overloaded host can reduce the actual publication rate. This gate does not synchronize with policy inference; MIT dispatch uses the common nominal 200 Hz actuation loop.

`sequence` still counts **every control tick** so records correlate with event sequences. At a regular 200 Hz loop, legacy records use `1, 5, 9, ...`, while MIT normally publishes every tick. `cycle_interval_ms` measures consecutive control starts (normally about 5 ms), not a topic-specific sampling period. Use successive tick timestamps to measure the topic rate. Records are sampled ticks, not averages or aggregations: skipped/unsampled ticks are not reconstructed. Fault/event detection remains unthrottled; MIT fault publication itself is 100 Hz. Historical analyses must account for the backend's sampling rate before equating sequence gaps with message loss. The ordered fields and schema labels are unchanged. The record's groups are:

| Group | Fields and interpretation |
| --- | --- |
| Gate and freshness | `requested_enable`, `gate_enabled`, `outputs_active`, `fault_latched`, `command_age_ms`, `command_ready`, per-motor `feedback_ready`, paired and individual position/velocity ages. `gate_enabled` is the software decision for this tick; `outputs_active` is software bookkeeping, not a firmware confirmation. |
| Received targets | `requested_*_target` is the last ROS command before motor-I/O clamps. `bounded_*_target` is the stored target after those clamps; `*_target_clipped` reports changes. |
| Steering loop | `slewed_steering_target_rad`, `steering_slew_limited`, `steering_error_rad`, `steering_p_torque_nm`, `steering_d_torque_nm`, and `steering_unclipped_torque_nm` expose the control calculation. `steering_final_torque_nm` is its clipped result. |
| Rear loop | `drive_damping_torque_nm` is `−0.05 × measured rear speed`; `drive_unclipped_torque_nm` includes the bounded feed-forward target; `drive_final_torque_nm` is clipped to ±6.28 N·m. |
| CAN submission | `*_current_command_a` precedes C0's 0.001 A quantization; `*_encoded_current_a` and `*_encoded_torque_nm` decode the outgoing payload. `*_send_attempted`, `*_send_success`, and `*_send_duration_ms` describe local socket writes. Socket acceptance does not establish that the motor executed the frame. All three command/current fields are NaN when that motor's send was skipped. In `dry_run`, a would-be payload can be calculated, but `send_attempted` and `send_success` are 0. |
| Disabling | `disable_attempted` and separate zero-current/CF success flags for both motors. Those flags are NaN if no frame was attempted. A failed disable is retried on subsequent disabled ticks and reported in `motor_event`. |
| Scheduling/polling | `cycle_interval_ms` is the interval between motor loop starts; `work_duration_ms` is work through CAN submission before publishing the diagnostic record. `*_query_ms` is host query duration. `*_query_failures` is a cumulative count of queries returning no reply, including send errors; it is not exclusively a timeout count. |

`/trackstand/policy_timing` is emitted on every nominal 50 Hz policy tick. It includes tick interval, work duration before timing publication, observation-build time, ONNX inference time, action-processing time, whether inference succeeded, software readiness/enabled flags, and IMU/motor-state/RC callback ages. Inference fields are NaN when inference was not attempted. A long cycle interval reveals scheduler delays even if reported work duration is short.

## Events and current feedback

`/trackstand/motor_event` is a JSON `String` on each change in motor gate, output state, fault state, or disable reasons. It records `sequence`, timestamps, `disable_reasons`, `fault_reason`, and any failed zero-current/CF writes. Legacy command timeouts can appear here while `/trackstand/motor_fault` remains false; MIT active-episode command/feedback watchdog failures now latch that fault until a safe RC-down reset. MIT events also track per-motor disable confirmation and emit a terminal `mit_io_stopped` transition after worker cleanup, without requiring another control tick. `/trackstand/policy_event` records mode/readiness transitions, the current readiness reasons, and the reasons remembered when policy balancing latched off. These topics record *transitions*, not 200 Hz/50 Hz streams.

MIT control-record timestamps identify tick start; command/feedback age and readiness fields are refreshed at the final safety checks before motor submissions. A stalled tick can therefore contain larger ages than those at tick start; `work_duration_ms` exposes that elapsed work. Dry-run logical slew timing follows the same elapsed-time rule but never fabricates actual sends, send intervals, or F1 receipts.

The diagnostic numeric topics publish with best-effort QoS; unified `motor_state` keeps reliable depth-10 QoS for its policy subscriber. Inspect timestamp gaps with the intended sampling rate in mind, and verify the rosbag actually contains the expected topics. Consult [BIKE_DIAGNOSTIC_TESTS.md](BIKE_DIAGNOSTIC_TESTS.md) for the recording command and staged tests.
