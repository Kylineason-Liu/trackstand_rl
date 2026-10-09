# Deployment Audit

Audit date: 2026-09-23

## Naming and interfaces

- ROS node names use the `trackstand_*` prefix; all runtime topics use `/trackstand/*`.
- Variables and parameters carry unit suffixes (`_rad`, `_rad_s`, `_nm`, `_hz`, `_s`, `_m`).
- The paired supports are consistently called `kickstands`; steering and rear drive remain distinct.
- Motor messages have fixed documented layouts: state is `[steer_pos_rad, drive_pos_rad, steer_vel_rad_s, drive_vel_rad_s, publish_time_s]`; command is `[enable, steer_target_rad, drive_torque_nm]`.

## Logic and safety

- Startup lowers the stands and sends zero-current plus motor-disable frames.
- Only a fresh RC-up state plus fresh IMU and both encoder streams enables policy output.
- RC down disables motors before commanding the stands down. Sensor loss, invalid policy output, excessive roll/speed, and steering-limit violations fail closed.
- Faults latch until an RC down/up cycle. CAN control commands have an independent 80 ms watchdog.
- B0 torque constants are required; no default motor constant is substituted. A2 int32 speed is used instead of range-limited A4 speed.
- IMU and encoder nodes publish only when a new hardware sample arrives, preventing cached data from defeating freshness checks.

## Unit and model consistency

- The 13 policy channels match `balance_imu_state` order and use radians, rad/s, m/s, and normalized gravity.
- At the original 2026-09-23 audit, raw actions mapped to `1.57 rad` steering and `6.28 N·m` drive scales, and deployment steering was further limited to ±30°. The steering limit was superseded on 2026-09-28 as recorded below.
- The 200 Hz motor loop implements steering `10 N·m/rad` proportional and `0.2 N·m·s/rad` derivative gains plus rear damping `0.05 N·m·s/rad`.
- ONNX and external-data SHA-256 values are checked at startup. ONNX versus checkpoint inference differed by at most `9.537e-07` over 128 seeded random inputs.

## Verification and remaining physical checks

Pure tests pass for observation construction, quaternion correction, state transitions, A2/A3/B0/C0 protocol units, action scaling, and ONNX determinism. The ROS package built successfully into temporary build/install directories.

No bicycle hardware is attached, so live serial/CAN timing and physical signs were not claimed as tested. The deployment now matches the active `ros2_ws` hardware configuration: CRSF/ELRS on the CP2102 UART at 420000 baud, CH7 paired kickstand control, steering motor ID 2, drive motor ID 1, and the CH340 IMU. Its current `params.yaml` applies a 1.42-degree roll correction rather than an identity transform. Complete `HARDWARE_CHECKLIST.md` before lifting the stands. Rear encoder velocity supplies a no-slip longitudinal estimate, and lateral velocity is constrained to zero; these are the only observation approximations.

## 2026-09-28 steering-range update

The requested controller target and motor-I/O steering limits are now ±90° (`1.5707963268 rad`), matching the simulated joint range. The policy action scale stays at the training value `1.57 rad`, so its maximum commanded target is about ±89.95°. The motor-I/O measured-angle fault threshold also changes from ±30° to ±90°, and its disabled-state fault-clear threshold from ±15° to ±45°. The roll safety limit remains ±30°. These are software limits; physical clearance through the expanded range has not been verified on the bike. Check it with the stands down and the frame secured before enabling motorized steering.

The subsequent motor fault-clear setting is independent of the ±90° command limit: `steering_fault_clear_rad` is now ±5° (`0.0872664626 rad`) while the motors are disabled. This replaces the ±45° clearing threshold above. The controller's own fault latch still requires an RC-down cycle before another RC-up request can start balancing.

## 2026-09-28 motor-sign update

The bike's reported raw checks supersede the earlier `ros2_ws` motor-sign assumptions: positive C0 current turns steering left and the rear wheel backward; positive A2/A3 feedback occurs when steering left and the rear wheel forward. Deployment signs are therefore steering encoder/current `-1/-1` and rear encoder/current `+1/-1`. The report was not independently measured by this repository audit; verify the installed parameters and low-power motion with the bike secured before lifting the stands.

## 2026-10-01 raw-action limit update

Removed the controller's `[-1, 1]` raw-action clipping to match Trackstand-Balance-v0's unclipped RSL-RL action configuration. Scaling remains `1.57 rad` for steering and `6.28 N·m` for drive. The controller's independent ±90° steering-target clamp remains, superseding the earlier ±89.95° reachable-target statement. Drive targets published by the controller can now exceed ±6.28 N·m, but motor I/O still clamps them to that range before damping and final torque saturation. Motor safety, fault detection, and kickstand logic are unchanged.

## 2026-10-01 motor-current torque telemetry

Historical interface: the separate torque topic described below was superseded on 2026-10-06 by the unified seven-field `motor_state_v2` message. See [TELEMETRY.md](TELEMETRY.md) for the current contract; the original validation record is retained here.

Added a separate receive-only CAN listener for measured A1/C0 Q-axis current replies. `/trackstand/motor_torque` publishes individual timestamped `JointState` samples with the motor joint name and `effort` in N·m, calculated as `torque_sign × Kt × Iq`; no requested current or saturation is used for feedback. The existing motor-state layout, encoder polling, control laws, safety logic, and command rate are unchanged, and no additional CAN requests are sent. Torque-constant availability is required; missing replies remain gaps. The ROS timestamp denotes listener receipt, not hardware acquisition. This is an electrical torque estimate, not a calibrated shaft measurement. Physical firmware reply behavior and timing still require a secured hardware check.

Verification: 51 pure tests passed, including reply decoding, torque signs, invalid/missing feedback, independent named samples, and a receive-only listener. ROS 2 Humble package build succeeded in temporary build/install directories. Real ROS `JointState` serialization/deserialization passed for both motor samples with synthetic CAN frames; no physical CAN or motor response was exercised.

## 2026-10-01 high-rate telemetry and full same-day audit

Added `telemetry.py`, `test_telemetry.py`, `test_motor_control.py`, `test_policy_timing.py`, and isolated ROS/CAN test fixtures. Motor I/O emits 200 Hz versioned numeric records for final steering/drive commands, steering slew/P/D/saturation, per-channel feedback ages/query failures, send results, and cycle timings. Policy emits 50 Hz inference and input-age timing. JSON transition events report explicit motor and policy disable reasons. The schema and staged hardware steps are in `TELEMETRY.md` and `BIKE_DIAGNOSTIC_TESTS.md`. High-rate diagnostics and torque feedback use best-effort QoS; bag sequence gaps must be checked.

Audit of all same-day deployment edits: the default bundled self-contained `09221945/model_91850` ONNX SHA-256 matches the controller constant; ONNX checker passed and all six embedded actor weight/bias tensors exactly matched the original checkpoint. Raw policy outputs remain unclipped in the controller; the ±90° steering target and motor-I/O effort limits remain independent. Invalid configured steering target limits and nonfinite motor/policy timing limits now fail startup checks. The torque-feedback listener still adds no CAN queries. A discovered CAN error path was corrected: a failed steering C0 write now skips the drive C0 write, attempts immediate zero-current/CF for both motors, and reports a fault in that same control tick. Failed disable writes are reported and retried while disabled. Success of a host CAN socket write is never represented as confirmed motor execution. No automatic kickstand or normal motor-control equation change was introduced.

Verification: **80 tests passed**, including 500 seeded normal-control targets against the prior steering/drive equations, command timeout without `motor_fault`, stale feedback, torque sign/units, dry-run honesty, and injected C0/CF failures. Real ROS 2 message serialization passed for 101 motor control ticks (active and disabled), both torque replies, and policy timing/events/actions using synthetic CAN frames. The package built in isolated temporary directories. None of these checks establishes actual motor firmware timing or torque response on the bike.

The same-day `isaaclab_train/reports/motor_limits_09221945_91850_2026-10-01` audit artifacts were also checked read-only: saved run configuration differences remained restricted to the documented allowances; original/copied checkpoint hashes match; array shapes are 3,000 × 8 × 64 × 2 for physics torque and 3,000 × 64 × 2 for policy targets; 129 completed episodes and 1,265 steering / 171 drive target exceedances were independently reproduced from the saved arrays. The wrapper's in-process `OnPolicyRunner` reference is now restored in its `finally` block. Historical numeric outputs and graphs were not regenerated. Its README now dates the old deployment clipping statement and records the subsequent unclipped-action change.

Remaining boundary: telemetry must be checked on the Raspberry Pi for real reply availability, bag drop rate, and any observer overhead. Current-derived torque remains an estimate; firmware run mode, voltage, and hardware acquisition times are not yet published. Avoid claiming a measured shaft torque or a guaranteed motor disable from host socket success alone.
