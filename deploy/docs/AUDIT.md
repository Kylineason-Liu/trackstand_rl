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
- Raw actions map to `1.57 rad` steering and `6.28 N·m` drive scales. Hardware steering is further limited to the bike's established 30-degree safe envelope.
- The 200 Hz motor loop implements steering `10 N·m/rad` proportional and `0.2 N·m·s/rad` derivative gains plus rear damping `0.05 N·m·s/rad`.
- ONNX and external-data SHA-256 values are checked at startup. ONNX versus checkpoint inference differed by at most `9.537e-07` over 128 seeded random inputs.

## Verification and remaining physical checks

Pure tests pass for observation construction, quaternion correction, state transitions, A2/A3/B0/C0 protocol units, action scaling, and ONNX determinism. The ROS package built successfully into temporary build/install directories.

No bicycle hardware is attached, so live serial/CAN timing and physical signs were not claimed as tested. The deployment now matches the active `ros2_ws` hardware configuration: CRSF/ELRS on the CP2102 UART at 420000 baud, CH7 paired kickstand control, steering motor ID 2, drive motor ID 1, and the CH340 IMU with an identity mounting transform. Complete `HARDWARE_CHECKLIST.md` before lifting the stands. Rear encoder velocity supplies a no-slip longitudinal estimate, and lateral velocity is constrained to zero; these are the only observation approximations.
