# MIT motor I/O alternative

`motor_mit_io` is an alternative to `motor_io` and `motor_io_position`; run exactly
one of them. It keeps `/trackstand/motor_command` as
`[enable, steering_target_rad, drive_torque_nm]` and publishes the same state,
measured torque, fault, status, control, and event topics. The control array
uses the `motor_control_mit_v1` label: existing fields retain their order;
MIT fields are appended. Steering's host PD/current-command fields are `NaN`
because MIT PD runs in the motor. A1 current times the measured B0 torque
constant remains the source of `/trackstand/motor_torque`; F1's separately
decoded motor torque is in `/trackstand/motor_control`.
The appended MIT fields also report the last encoded target, gains,
zero-velocity/feedforward quantization residuals, and actual send interval.

The policy still supplies targets at 50 Hz. This node sends MIT steering
position/zero-velocity/zero-feedforward packets at 50 Hz using Kp=10 and
Kd=0.2 in **native motor units**. A 200 Hz safety/C0 loop applies the
unchanged rear-wheel damping law and polls both A3/A2 encoder pairs. Steering
A1 is requested nominally at 50 Hz; rear C0 replies provide current feedback.
Actual rates and ages are visible in recorded telemetry and may fall below
nominal rates if CAN or the host is delayed. Firmware PD loop frequency is
undocumented.
`motor_control` is published on safety/C0 ticks and MIT dispatch ticks;
interleaving can produce more than 200 records/s. Send flags identify new
CAN commands; MIT interval fields describe actual sends.

At startup, with kickstands down and steering straight, the powered node sends C0 zero/CF to both
motors, B1 once to steering, configures F0 Pos_Max to 3 native rad while
retaining the motor's actual Vel_Max/T_Max encoding ranges, then verifies F0,
A3, F1, and B0. B1 and F0 persist beyond this process; no B1 is sent on
later RC transitions. The node blocks enable until fresh paired encoder
feedback and B0 constants are available. It checks unwrapped steering angle
against ±90°, steering/rear speed, F1 mode/fault status, missing F1, command
age, feedback age, and **both motors' AE/CF hardware fault reports**.
RC down sends steering CF and drive C0 zero/CF. Disable remains pending
until each motor replies to CF with mode=0; new active commands are blocked
during this transition. A missing/late acknowledgement beyond 250 ms
latches `disable_ack_timeout` and requires restart. Failed sends are retried
at 20 ms intervals; successfully sent CF commands are not repeated while
waiting for their initial acknowledgement. After timeout, unconfirmed
motors receive further disable retries. Shutdown also waits up to 250 ms
for confirmation before closing CAN; an unconfirmed disable is reported,
never treated as proof that the motor is off.
Hard MIT, transport, and origin faults require a process restart; ordinary
latched limits clear only with RC down, confirmed disable, fresh feedback,
no currently exceeded angle/speed limits, and steering within ±5°.
`mit_restart_required` distinguishes these cases in control/status.
The first target step of every enable cycle is bounded to 23.9/50=0.478 rad;
time spent disabled cannot enlarge it. The first hard-fault reason is
retained through subsequent disable/shutdown failures.

Linux kernel receive timestamps (`SO_TIMESTAMPNS_NEW`, 64-bit timespec)
preserve queue age. A3/A2 replies must arrive after the matching query was
sent and within its deadline, including during startup. Queued faults are
drained before actuation, and the verified startup A3 reply defines steering
zero. Missing/invalid timestamps, excessive backlog, or uncertain
realtime-to-monotonic clock conversion stop the node rather than silently
refreshing stale feedback. Settle system clock synchronization before a
powered test. These are **host kernel arrival times, not motor acquisition
timestamps**; the protocol has no transaction ID to identify an arbitrarily
delayed reply from an older identical query.

Appended control fields report `disable_pending`, `disable_confirmed`,
`steering_disable_confirmed`, and `drive_disable_confirmed`. Status also
lists `motors_awaiting_disable`; events include the transition state.
`outputs_active=true` during pending disable means output may still be
active, not that disable was confirmed. Existing `*_disable_send_success`
fields still describe socket transmission only. During dry-run,
`disable_confirmed` is a simulated gate, not a hardware acknowledgement.

The local protocol establishes F0 as MIT *encoding ranges*; it does not
establish that B3's position/speed current limit applies to MIT, or that F0
T_Max limits the total internal PD torque. This node therefore cannot claim
the old host-side ±5.6 N·m steering torque clamp. Confirm the motor's actual
MIT current/torque limiting independently; current feedback and F1 torque
are exposed for this comparison. `dry_run:=true`
never sends CF, B1, F0 writes, MIT, or C0; it still reads hardware.

From `deploy/`, after building and sourcing the package:

```bash
ros2 run trackstand_deploy motor_mit_io --ros-args \
  --params-file source/trackstand_deploy/config/params.yaml -p dry_run:=true
```

For powered operation, replace `dry_run:=true` with `dry_run:=false`. Start the
other deployment nodes exactly as for `motor_io`, but do not start another
motor I/O process or standalone CAN writer on the selected motors. The
default launch file continues to select `motor_io`.

Optional ROS parameters `mit_kp_nm_per_motor_rad` and
`mit_kd_nm_s_per_motor_rad` override the default gains. `control_rate_hz` and
`poll_rate_hz` keep their existing defaults of 200; MIT command rate is fixed
at 50 Hz. `feedback_timeout_s` must exceed its 20 ms reply interval.

Offline validation covers packet units/gains, motor signs, startup and
persistent references, 50/200 Hz scheduling, paired feedback ages, disabled
operation, faults, receive overflow, and ROS shutdown. The package also
builds in an isolated colcon workspace. These checks do not measure the
Raspberry Pi's execution rate or the bike's firmware response.

Post-audit validation on 2026-10-05 includes regression cases for driver
faults, queued replies, CF confirmation/timeout/partial failures, startup zero,
fault recovery, re-enable slew, and powered/dry-run 50/200 Hz scheduling.
The real Linux socket timestamp test uses a local Unix socket pair, not CAN.
Results: **289 applicable deployment tests passed**; the kernel socket test
was skipped under sandbox restrictions and **passed separately outside the
sandbox**. Pyflakes is clean, and an isolated colcon build succeeded with
installed source matching the audited source.
The full deployment suite retains its
pre-existing 3 failures/51 errors in three script test modules that still
refer to missing `deploy/scripts/` paths; those failures were reproduced
before this implementation.
