# MIT motor I/O alternative

`motor_mit_io` is an alternative to `motor_io` and `motor_io_position`; run exactly
one of them. It keeps `/trackstand/motor_command` as
`[enable, steering_target_rad, drive_torque_nm]` and publishes the same unified
state, fault, status, control, and event topics. The control array
uses the `motor_control_mit_v1` label: existing fields retain their order;
MIT fields are appended. Steering's host PD/current-command fields are `NaN`
because MIT PD runs in the motor. A1 current times the measured B0 torque
constant remains the source of the torque fields in `/trackstand/motor_state`; F1's separately
decoded motor torque is in `/trackstand/motor_control`.
The appended MIT fields also report the last encoded target, gains,
zero-velocity/feedforward quantization residuals, and actual send interval.

`motor_state` uses the shared seven-field `motor_state_v2` format: the existing
position/velocity/time prefix followed by steering and rear torque in N·m.
Missing/stale torque is `NaN`; the policy still uses only the four kinematic
fields. There is no separate `motor_torque` publisher. See
[TELEMETRY.md](TELEMETRY.md) for field order and asynchronous-snapshot semantics.

The implementation is independent of `motor_io.py` and `motor_io_position.py`:
`MotorMitIo` inherits directly from ROS `Node` and owns its channel state,
21 parameter defaults, command handling, snapshots, ROS interfaces, state
publication and control/event telemetry. It shares only backend-neutral
helpers in `can_protocol.py`, `mit_protocol.py`, `contracts.py` and `telemetry.py`;
it does not import either legacy node or `python-can`.

## Three frequency divisions

| Division | Responsibility |
| --- | --- |
| 50 Hz (`COMMAND_RATE_HZ`) | Receive policy targets on `/trackstand/motor_command`. The policy owns this rate; callbacks are not throttled, so an incoming disable is never discarded. |
| 100 Hz (`STATE_RATE_HZ`) | Request A3 position, A2 velocity and A1 current from **each** motor once per query round; publish `/trackstand/motor_state`, `/trackstand/motor_status` and `/trackstand/motor_fault`. |
| 200 Hz (`MIT_RATE_HZ`) | One CAN-owner actuation loop sends MIT steering and C0 rear torque, checks safety, and publishes `/trackstand/motor_control`. |

Steering requests zero velocity/zero feedforward torque, Kp=10 and Kd=0.2 in
**native motor units**. Rear damping, gains, signs, limits and the default
80 ms command / 50 ms feedback watchdogs are unchanged. The current policy
target is held between its 50 Hz updates; 200 Hz CAN sends do not refresh
the ROS-command watchdog. Actual rates may be lower under host/CAN load.
MIT dispatch is aligned with the common actuation tick while respecting its
last-send deadline, avoiding extra MIT-only control ticks or catch-up bursts.
Firmware PD loop frequency is still undocumented.

The 100 Hz query rate is **per opcode per motor**, not 100 CAN frames/s;
six live-state queries are serialized each round. F0 encoding-range
verification and missing-B0 calibration retries retain their 1 s interval
inside the state-query scheduler; startup/preflight queries are one-shot.
CAN receive is continuously serviced: automatic MIT F1 and rear C0 replies
can arrive at 200 Hz and are not discarded to enforce the ROS state rate.
State publication still requires new, fresh encoder pairs from both motors,
so its actual rate is **up to** 100 Hz. Status and fault use the same 10 ms
ROS timer even if state is unavailable or the CAN worker has stopped.

Fault detection and CAN disabling do **not** wait for the 100 Hz fault
publisher. `/trackstand/motor_event` remains transition-only, evaluated on
every actuation tick. Control diagnostics use a separate 200 Hz publication
gate; its sequence counts all control ticks and send flags describe that
tick, not an aggregate. See [TELEMETRY.md](TELEMETRY.md).

At startup, with kickstands down and steering straight, the powered node sends C0 zero/CF to both
motors, B1 once to steering, configures F0 Pos_Max to 3 native rad while
retaining the motor's actual Vel_Max/T_Max encoding ranges, then verifies F0,
A3, F1, and B0. B1 and F0 persist beyond this process; no B1 is sent on
later RC transitions. The node blocks enable until fresh paired encoder
feedback and B0 constants are available. It checks F1 mode/fault status, missing F1, command age, feedback age,
and **both motors' AE/CF hardware fault reports**. Measured steering angle
and steering/rear speed remain diagnostic data and do not trigger stop faults.
RC down sends steering CF and drive C0 zero/CF. Disable remains pending
until each motor returns a fault-free CF acknowledgement. A mode-0 reply confirms
the transition directly; a retained mode (1–5) additionally requires a fresh,
fault-free F1 reply with MIT inactive. The firmware on this bike retains mode 2
after zero-current C0/CF and mode 5 after MIT/CF; the selected-mode byte alone
does not establish whether CF stopped output. New active commands are blocked
during confirmation. A missing/late acknowledgement beyond 250 ms
latches `disable_ack_timeout` and requires restart. Failed sends are retried
at 20 ms intervals; successfully sent CF commands are not repeated while
waiting for their initial acknowledgement. After timeout, unconfirmed
motors receive further disable retries. Shutdown also waits up to 250 ms
for confirmation before closing CAN; an unconfirmed disable is reported,
never treated as proof that the motor is off.
Hard MIT, transport, and origin faults require a process restart. During an
active episode, command timeout or lost encoder/B0 readiness now **latches**
the motor fault: a brief recovery cannot automatically restart balancing or
disappear between 100 Hz fault publications. Incomplete startup feedback
still gates enable without latching an episode fault. Ordinary
latched watchdog/send faults clear only with RC down, confirmed disable,
fresh feedback, and steering within ±5°.
`mit_restart_required` distinguishes these cases in control/status.
The first target step of every enable cycle is bounded to 23.9/200=0.1195 rad;
a delayed step is capped at two actuation intervals (0.239 rad). Time spent
disabled cannot enlarge it. The first hard-fault reason is
retained through subsequent disable/shutdown failures.

Both motor sends revalidate safety **after payload encoding**, without ROS
work between that check and socket submission. They use the original
packet's command timestamp; a newer policy message cannot refresh an old
packet. Ages are sampled after snapshot-lock waits, and a disable-generation
counter remembers RC down even if up follows before a motor tick or during
the tick. A rapid down/up cancellation latches `enable_withdrawn_during_control`
until a safe down/reset. If steering was already submitted when the gate
closes, rear motion is withheld and both motors enter confirmed-disable
handling. Dry-run now uses separate logical slew timestamps to match
jittered powered scheduling, without fabricating actual MIT sends or F1 feedback.

Control timestamps still identify tick start; age/readiness fields reflect
the latest safety check within that tick. Events additionally expose each
motor's disable confirmation, including partial acknowledgements. A worker
failure/shutdown emits a terminal `mit_io_stopped` event after cleanup,
without depending on another control tick. ROS reporting failure is logged
but cannot skip the preceding CAN disable/close attempts.

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

The selected CAN interface must already be UP with the bitrate configured to
match both motors. The node checks this before sending motor preflight commands;
it does not configure the interface or choose a bitrate. A send failure reports
the interface, CAN ID, payload, and original system error number. An I/O failure
is latched and requires a node restart after correcting the transport.

When ROS has already shut down (for example, after Ctrl+C), CAN disable/close
handling still runs, but terminal ROS event publication is skipped.

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
`mit_kd_nm_s_per_motor_rad` override the default gains. MIT-specific rate
parameters are `command_rate_hz=50.0`, `state_rate_hz=100.0`, and
`actuation_rate_hz=200.0`; startup rejects values outside this fixed profile.
Legacy `poll_rate_hz` / `control_rate_hz` in the shared `params.yaml` are
**not used by MIT**, so that file remains compatible with all three backends
without changing legacy control rates. Inspect the MIT-specific parameters
or the effective rate fields in `motor_status`. `command_timeout_s` must
exceed 20 ms; `feedback_timeout_s` must exceed the 10 ms state-query interval.
Neither default watchdog was rescaled. Rebuild/source the package after
deploying the modified Python files.

All 21 MIT configuration parameters are **startup-only/read-only**. Launch,
parameter files and `--ros-args -p ...` overrides still work at startup;
runtime changes are rejected instead of reporting values that the running
worker has not adopted. In particular, `ros2 param set ... dry_run true` is
**not a stop command**. Restart with the intended configuration while supported.

The increased control/status telemetry adds CPU and ROS traffic compared
with the earlier 50 Hz control / 2 Hz status configuration. Check actual
topic rates and `mit_actual_send_interval_ms`, policy timing and feedback
ages while recording on the Raspberry Pi, with the bike supported first.

2026-10-08 post-fix validation: **515 deployment tests passed, 1 skipped** (the
sandbox-restricted local socket test). Tests cover 50 Hz command updates,
100 Hz queries and state/status/fault publishing, 200 Hz MIT/C0/control,
nonzero simulated CAN-send work, no query/send catch-up bursts, watchdog
independence, prompt disabling between state publications and all existing
safety regressions, transient watchdog latches, pauses before/between sends
and inside snapshot acquisition, rapid down/up, terminal worker/close faults,
and dry-run jitter. A separate **1,000 healthy powered/dry-run mock ticks**
match the pre-fix CAN packets, slew, state/fault outputs, defaults and schemas.
Real local ROS 2 Humble parameter methods confirm startup overrides still
work and live read-only mutations fail, without initializing DDS or CAN.
The isolated colcon build and installed import with real ROS classes pass;
the installed MIT source matches the audited file. All **130 experiment
tests** also pass. These are offline tests, not measured bike timings.

## 2026-10-08 live CF confirmation fix

The live-mode failure was reproduced with a steering CF reply whose selected-mode
byte remained 2 and fault byte was 0; the subsequent F1 status was 0 (MIT inactive,
fault-free). Startup and runtime/shutdown now use the fresh F1 confirmation path
for retained modes, while rejecting faults, active MIT, malformed status, stale
replies and missing confirmations. Dry-run does not exercise CF confirmation.

Validation: **210 motor MIT tests passed**. The deployment suite had **538 passed
and 1 failed**; the failure was the policy ONNX test because this Python environment
lacks `onnxruntime`. The ROS 2 Jazzy colcon build succeeded and installed motor
source matched the edited source. A live, non-dry-run probe on both motors ran
only CF/F1 and drive C0 zero during shutdown: both disable confirmations completed,
with no pending confirmations or hard fault. It did not send B1, F0 writes or MIT
motion targets, and did not validate active balancing.

## Remaining hardware safety checks

Disable confirmation is based on the CF acknowledgement plus the F1 MIT-state
readback when the firmware retains its selected mode. These replies are protocol
evidence, not an independent measurement of power-stage output: F1 reports MIT
activity and faults, not a separate output-enable flag for every control mode.
Physical free-state verification remains necessary with the bike supported.
Runtime AE reports are consumed, **not actively polled at 100 Hz**; the 100 Hz
fault/status topics are software publications, not 100 Hz hardware fault samples.
The node does not configure/read back the motor's CD communication watchdog.
The local protocol documents that watchdog as nonpersistent and disabled
by default; the bike's live setting and its behavior in MIT mode are unknown.
No host check can act during an arbitrary process/OS freeze, power loss or
broken CAN connection. Verify motor-side deadman behavior and actual MIT
current/torque protection before unsupported balancing. No undocumented
firmware setting was changed by this fix.

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
At that date, the full deployment suite retained its
pre-existing 3 failures/51 errors in three script test modules that
refer to missing `deploy/scripts/` paths; those failures were reproduced
before this implementation.

The relocated-script tests were fixed in the subsequent 2026-10-06 refactor.
After MIT separation, validation is **386 deployment tests passed, 1 skipped**
(sandbox-restricted local socket test), plus **117 experiment tests** and
**5 historical tracking-analysis tests** passed. A blocked-import regression
checks independence from both legacy nodes and `python-can`. The isolated
ROS 2 Humble build and an installed import with real ROS classes also pass.
An immediate pre-separation/current comparison matches **1,000 synthetic
MIT control ticks** across seeds 0–9, including command clipping, CAN sends,
six ROS publications, stale inputs, overspeed, failed writes and CF disable
confirmations. All 20 defaults and telemetry field order match; 16 existing
MIT methods, including startup, worker, control, transport and disable logic,
are AST-identical. These are offline checks, not new real-bike measurements.
