# Deployment Telemetry Contract

Record `/trackstand/motor_control`, `/trackstand/motor_event`, `/trackstand/policy_timing`, `/trackstand/policy_event`, and `/trackstand/motor_torque` alongside the existing policy, sensor, and command topics. This page describes the signals emitted by the current deployment. Schema constants live in `trackstand_deploy/telemetry.py`.

## Numeric telemetry

`/trackstand/motor_control` and `/trackstand/policy_timing` are `std_msgs/msg/Float64MultiArray`. The ordered field names appear in `layout.dim[0].label` as `motor_control_v1:<comma-separated names>` or `policy_timing_v1:<comma-separated names>`. Match those names to `data` by index. Fields ending `_ms`, `_s`, `_rad`, `_rad_s`, `_nm`, and `_a` use milliseconds, seconds, radians, radians per second, newton metres, and amperes respectively. Flags are 0/1. NaN means unavailable or not applicable; it does not mean zero. Both messages carry `sequence`, ROS publication time in seconds, and a process `time.monotonic()` timestamp. These timestamps are from the host, not from the motor firmware. Monotonic time can be compared across the two nodes on the same Raspberry Pi boot.

The motor control record is emitted on every 200 Hz control tick, including disabled ticks. Its groups are:

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

`/trackstand/motor_event` is a JSON `String` on each change in motor gate, output state, fault state, or disable reasons. It records `sequence`, timestamps, `disable_reasons`, `fault_reason`, and any failed zero-current/CF writes. A command timeout appears here even if `/trackstand/motor_fault` remains false. `/trackstand/policy_event` records mode/readiness transitions, the current readiness reasons, and the reasons remembered when policy balancing latched off. These topics record *transitions*, not 200 Hz/50 Hz streams.

`/trackstand/motor_torque` is a `sensor_msgs/msg/JointState` with one joint name and one `effort` sample per reply. Split by `name[0]`: `steering_joint` or `rear_wheel_joint`. `effort[0] = configured_torque_sign × B0_torque_constant × measured_Q_axis_current` in N·m. Its ROS header time is listener receipt. This current-derived motor-torque estimate is not a shaft torque sensor, and its sample is not synchronized to either motor's control tick. Missing replies remain gaps. `/trackstand/motor_status` adds the age of each motor's last torque sample at its 2 Hz status rate.

The high-rate numeric topics and torque feedback publish with best-effort QoS. Check the sequence for dropped records, and verify the rosbag actually contains these topics. Consult [BIKE_DIAGNOSTIC_TESTS.md](BIKE_DIAGNOSTIC_TESTS.md) for the recording command and staged tests.
