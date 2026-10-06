# C2 steering motor I/O alternative

`motor_io_position.py` is an alternative to `motor_io.py`, not an additional node to run alongside it. Both use ROS node name `/trackstand_motor_io`, the same parameter-file section, and the same topic names. The default launch still starts the original C0 node; that file, launch, and hardware configuration are unchanged.

## Command conversion

`/trackstand/motor_command` still contains `[enable, steering_target_rad, drive_target_nm]`. The steering target is an **absolute angle relative to the startup encoder zero**, not a relative motion request. The existing ±90° clamp and 23.9 rad/s target slew remain.

At nominal 200 Hz, the new node computes:

```text
offset_count = round(encoder_sign × gear_ratio × slewed_target_rad × 16384 / 2π)
C2_count     = startup_zero_count + offset_count
payload      = C2 + signed_int32_little_endian(C2_count)
```

The startup zero is the first valid A3 **multi-turn signed encoder count** (bytes 3–6), not a wrapped angle or integer revolution count. Hold steering physically straight while starting the node. The zero is retained across disable/re-enable and kickstand cycles; restarting the node acquires a new zero. No persistent encoder origin is written.

The user verified that naturally increasing steering position turns **left from the rider's view**. Policy-positive steering turns right, so the existing `steering_encoder_sign=-1` subtracts the converted angle from the native startup zero. With gear ratio 1, a +0.30 rad target requests approximately `startup_zero_count − 782`. Repeating a held target repeats the same C2 payload, irrespective of current measured position; it does not accumulate increments. Verify a small C2 motion on the supported bike before relying on physical tracking.

The final encoded target is also clamped to the integer-count representation of the steering-angle limit, so rounding with a noninteger gear ratio cannot request a target outside ±90°.

The raw policy action is not an angle: `policy_controller.py` first applies the existing 1.57 rad steering scale and hardware clamp. This node consumes the resulting angle and preserves its target slew limit; it does not scale the action again.

Live position is **not used to calculate the C2 target**. It is still required for policy observations and safety: missing/stale feedback or an out-of-range unwrapped steering position prevents powered operation. Signed int32 absolute-target overflow latches a fault rather than wrapping. C2 removes C3's relative-reference ambiguity and feedback-dependent conversion error, not communication delay, zero-calibration error, or mechanical tracking error. The obsolete `c3_current_reference_verified` parameter is removed; do not pass it.

## Preserved behavior and position-mode differences

- Policy remains 50 Hz; command execution and A2/A3 polling remain nominal 200 Hz. The firmware's internal loop frequency is unknown.
- Rear C0 torque control, its damping/clamps, motor IDs/signs, command watchdog, freshness gates, angle/speed fault checks, fault clearing, and zero-current/CF disables are retained.
- Before powered C2, steering B2 speed and B3 current limits are written and independently read back. They match the existing 23.9 rad/s and 5.6 N·m caps, rounded **down** to protocol units. B3 uses the measured B0 torque constant. Missing/mismatched confirmation gates off both motors; setup retries at most once per second. Initial setup can lengthen a control tick; feedback is re-snapshotted and freshness rechecked afterward.
- The host no longer computes steering PD/current commands. Firmware B6–B9 gains are **not changed** or assumed equivalent to the host's `10/0.2` gains. Suitable firmware tuning still needs hardware validation.
- C2 replies contain position, not current. An extra nonblocking steering A1 query follows each active command cycle so `/trackstand/motor_torque` continues to publish actual-current-derived estimates. This adds CAN traffic; verify timing and feedback rates while recording.

## Start and verify

Build once to install the new executable:

```bash
cd ~/trackstand_rl/deploy
colcon build --symlink-install --packages-select trackstand_deploy
source install/setup.bash
ros2 run trackstand_deploy motor_io_position --ros-args \
  --params-file source/trackstand_deploy/config/params.yaml -p dry_run:=true
```

Stop the original `motor_io` first. No policy or other motor-command publisher should run during manual sine tests. Verify `dry_run` with `ros2 param get /trackstand_motor_io dry_run`. The ROS-topic sine publisher `deploy/scripts/steering_sine.sh` works unchanged; use small, independently restrained bench tests first, not unsupported balancing. The separate raw-CAN `steering_sine_position.py` does **not** exercise this node or its safety gates. Its current C2 waveform is centered on native encoder zero, not this node's recorded startup zero; do not substitute it for the ROS-topic test or run it alongside either motor I/O node.

After checking the straight-steering startup zero and dry-run telemetry, stop the dry-run node and start a supported, supervised powered test with:

```bash
ros2 run trackstand_deploy motor_io_position --ros-args \
  --params-file source/trackstand_deploy/config/params.yaml \
  -p dry_run:=false
```

B2/B3 must still confirm before actuation. Position mode uses firmware gains and can produce substantial torque despite a small angle request. Both motors are enabled by one command; RC-down does not override manual commands. Keep physical power isolation available. Restart this node with steering straight after motor power/reset or origin changes: its startup zero and volatile B2/B3 confirmation cannot be reused across a reset.

## Telemetry and protocol evidence

All existing topic names are retained. `/trackstand/motor_state` and `/trackstand/motor_torque` retain their formats. `/trackstand/motor_control` now uses `motor_control_position_v2:<fields>`: original fields come first, followed by the native C2 startup zero, absolute target count, target offset count, quantized target radians, B2/B3 confirmation and raw limits, A1 socket-send results, and `gate_monotonic_time_s`. This replaces the old C3-specific fields; match recorded layout labels rather than hardcoding offsets. Steering host-PD/current/torque-command fields are NaN because they are not applicable; rear command fields remain valid. Timing ages are evaluated at the recorded gate time. `/trackstand/motor_status` identifies `C2_absolute_position` mode and reports the native startup zero and limit confirmation. Socket-send success is not motor-execution confirmation.

Disk sources: the [CAN protocol Rev.3.09b0](<../../tsinghua_bike_project/assign_2_Hardwares/assign_2_Hardwares/3.RearSteer Servo【GIM3510-8】/CAN Protocol Doc/自定义CAN通信协议_3.09b0.pdf>), pp. 4–6 (A1/A2/A3), pp. 8–9 (B2/B3), p. 14 (C2), p. 24 (absolute position-loop diagram); `ros2_ws/src/motor_can_control/motor_can_control/can_protocol.py:set_position_angle`; and the original deployment `motor_io.py`/`contracts.py`. The normative C2 reply definition refers to A3; C2 replies are not used as torque feedback.

## Implementation validation

The 2026-10-03 full deployment suite passed **208 tests**, including **106 C2-node tests**. Run `PYTHONPATH=deploy/source/trackstand_deploy python3 -m pytest -q deploy/source/trackstand_deploy/test` from the repository root. C2 tests cover startup-zero offsets, both signs, multi-turn native targets, identical held targets despite changing feedback, quantization, int32 overflow, limit set/readback failures, unchanged rear laws, dry-run, watchdogs, CAN submission failures, constructor topic/timer registration, zero retention across disable/re-enable, status, and shutdown. AST comparisons confirm that the copied command callback, rear torque sender, disable handling, state/torque publication, and transition telemetry methods retain their original logic.

The audit retains post-setup age reporting/freshness checks, count-level angle bounds, unwrapped-position safety, and idempotent ROS shutdown. Python compilation and an isolated `colcon` build under `/tmp` passed; the build registered `motor_io_position`. Byte comparisons confirmed the original `motor_io.py`, `params.yaml`, default launch, and setup entry points were unchanged during this conversion. The separately edited raw-CAN sine helper and its tests were left intact. Tests use ROS/CAN stand-ins, not firmware or physical tracking validation. No hardware was commanded; actual C2 response and firmware gain tuning still need supported-bike validation.
