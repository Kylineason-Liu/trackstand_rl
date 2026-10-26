# Trackstand Bicycle Deployment

This ROS 2 workspace deploys `Trackstand-Balance-v1` on the Tsinghua bicycle. It is a stationary-balancing controller: the RC throttle and steering sticks are intentionally ignored. The RC kickstand switch is the operating authority.

## Runtime flow

1. Balancing starts disabled, with the RC kickstand switch down.
2. Powered MIT startup verifies disable and records the physically straight steering origin using B1/F0/A3/F1. Keep steering straight and stationary during startup.
3. RC up starts an episode only with fresh IMU, v3 motor state and RC data, no motor fault, acceptable lean, and steering within 5 degrees of its fixed straight reference.
4. At episode start, record a new software rear-angle reference and seed the steering target limiter from measured steering. Steering's physical straight reference stays fixed.
5. Every 20 ms: assemble 11 observations, infer two Beta mean actions, scale, clip steering to ±1.57 rad, slew by at most 0.478 rad, and publish the held steering target and rear torque.
6. MIT sends the held steering target and gains at 200 Hz; rear C0 sends torque at 200 Hz. Queries and state/status/fault publication are nominally 100 Hz. These rates do not identify firmware's internal rate.
7. RC down disables policy and requests stands down immediately. Stale inputs, motor faults, inference errors, |gravity_y| > sin(0.5236), or inversion latch balancing off. Policy faults do not automatically lower stands; RC down does.

The default is **Trackstand-Balance-v1**, DR-trained run **2026-10-21_11-18-29 / model_2500**, packaged as one self-contained ONNX file with a checked SHA-256. The export computes the deterministic Beta mean, already bounded within [-1,1]; four internal shape logits are never treated as motor actions. There is no observation normalization or added simulated sensor noise on the Pi.

Steering flow: bounded action ×1.57 rad → explicit ±1.57 clip → 23.9 rad/s target slew, updated **once per 50-Hz policy inference** → held motor target. Motor nodes do not apply another target ramp. Rear flow: bounded action ×6.28 N m → ±6.28 N m torque cap → signed C0 current using measured B0. Rear controller damping is now **0**, matching v1. Measured passive friction is already part of the real assembly; it is not subtracted or compensated again in software.

V1's simulated steering IdealPD clips total PD torque to ±5.6 N m. The physical MIT protocol does not establish an equivalent total-PD/current cap: F0 only establishes encoding ranges. C0's host PD backend enforces its own ±5.6 cap but runs at 200 Hz with sampled feedback; C2 firmware gains/dynamics are also not identified. These remain actuator differences, not fixed by observation alignment. Simulator speed constraints, contact/friction laws, and electrical dynamics are not guaranteed motor-firmware protections.

Hardware IDs, signs, IMU correction, CRSF receiver, kickstand servo poses and RC authority are unchanged. Steering ID 2 and rear ID 1 use encoder/current signs -1/-1: policy-positive steering means right from the rider; positive rear rotation/torque means forward. Ordinary motor fault recovery still requires enable off and steering strictly inside ±5 degrees; MIT additionally needs fresh readiness and confirmed disable. Hard MIT faults require restart.

## Build and run

```bash
cd deploy
python3 -m pip install -r requirements.txt
colcon build --symlink-install
source install/setup.bash
sudo ip link set can0 up type can bitrate 1000000
ros2 launch trackstand_deploy trackstand.launch.py
```

The checked-in hardware paths match `ros2_ws`: the CH340 by-id path is the IMU, the CP2102 by-id path is the CRSF receiver, and `/dev/ttyAMA2` is the servo bus. Edit `source/trackstand_deploy/config/params.yaml` only if the hardware identities change. ROS dependencies include `rclpy`, `rcl_interfaces`, `sensor_msgs`, and `std_msgs`. The default launch selects `motor_mit_io` (MIT steering + rear C0). The checked-in dry_run is false; use individual nodes with dry_run:=true for an inference-only check.

An optional `motor_io_position` executable uses firmware **C2 absolute position control** for steering while retaining the rear-wheel C0 controller and existing safety gates. Run it **instead of**, never alongside, `motor_io`; the default launch selects MIT. Targets use the latched raw A3 startup zero plus the converted steering angle, not the current measured position. Native increasing steering position turns left; `steering_encoder_sign=-1` maps policy-positive steering into decreasing native counts. Start with steering straight and in dry-run. Powered C2 requires B2/B3 limit readback; the former C3 reference-verification flag is removed. See [C2 node instructions and telemetry](docs/MOTOR_IO_POSITION.md) before using it on the bike.

The separate `motor_mit_io` executable has three rate divisions: **50 Hz policy commands**, **100 Hz motor-state queries and state/status/fault topics**, and **200 Hz steering MIT, rear C0, safety checks and control telemetry**. Steering uses native Kp=10, Kd=0.2, requested velocity=0, and feedforward torque=0. It inherits directly from ROS `Node` and does not import or inherit either legacy motor backend; only backend-neutral protocol/contract/telemetry helpers are shared. Its startup follows the verified MIT experiment's CF/B1/F0/A3/F1 checks. MIT cannot reproduce the host-side ±5.6 N·m steering torque clamp from `motor_io`; the protocol does not establish that the C2 current limit applies to total MIT PD torque. See [MIT node instructions, rate parameters and limits](docs/MOTOR_MIT_IO.md).

For individual-node commissioning, run **only** this motor backend, first in dry-run:

```bash
ros2 pkg executables trackstand_deploy
ros2 run trackstand_deploy motor_mit_io --ros-args \
  --params-file source/trackstand_deploy/config/params.yaml -p dry_run:=true
```

If `ros2 run` says `No executable found`, rebuild the synchronized package and source its installation; sourcing alone does not create the entry point. Check `ros2 pkg prefix trackstand_deploy` for an unintended overlay. MIT's 21 parameters are startup-only/read-only; changing `dry_run` at runtime is not a stop mechanism. Powered MIT requires CAN already UP, steering straight and stationary, and kickstands down during persistent-origin setup. Confirm CF/F1 disable handling as described in the MIT guide; neither send success nor an inactive-MIT reply alone proves a physical free state.

## Observation contract

The exact order is:

```text
[gravity_y, gravity_z, gyro_x, gyro_y, gyro_z,
 steering_position, rear_position,
 steering_velocity, rear_velocity,
 sin(steering_position), cos(steering_position)]
```

Gravity is the normalized world-down vector rotated by the corrected IMU quaternion into the bike frame. Gyro is bike-frame rad/s. Positions are radians; motor speeds are rad/s. There are no linear velocity estimates and no finite ±100 clamp. Non-finite repair remains in the pure observation builder; invalid received sensor samples cannot refresh controller readiness.

All motor backends publish continuous, signed, startup-relative A3 position as **motor_state_v3**. The policy subtracts the current episode's rear reference, then uses 2*atan2(sin(q/2),cos(q/2)) to match the measured v1 PhysX branch: period 4π, approximately ±2π, not single-turn ±π and not indefinitely cumulative. Steering retains its fixed straight coordinate. The controller rejects the old wrapped v2 motor-state label.

This matches the tested simulator observation calculation. Physical IMU fusion accuracy, acquisition latency and cross-sensor synchronization still need hardware validation. Source comparisons, tests and export provenance are in the [v1 migration report](experiments/reports/20261026_v1_deployment_alignment/README.md).

## Before first unsupported test

Run with the stands down and inspect `/trackstand/observation`, `/trackstand/status`, and `/trackstand/motor_state`. Confirm the configured IMU transform (currently `[1.42, 0.0, 0.0]` degrees in `params.yaml`), all four motor sign parameters, steering zero, kickstand positions, B0 torque constants, and CRSF channel values. Use `dry_run: true` for the first data-flow check. Set `status_rate_hz` above zero only during kickstand bench checks; it is disabled during balancing so serial feedback cannot delay commands. Do not lift the stands until the sign checklist in `docs/HARDWARE_CHECKLIST.md` passes.

## Record motor feedback

All three motor nodes (`motor_io`, `motor_io_position`, and `motor_mit_io`) publish position, velocity, and measured torque together on `/trackstand/motor_state` (`std_msgs/msg/Float64MultiArray`). The versioned layout label is `motor_state_v3:<fields>`, with data ordered as:

```text
[steering_position_rad, drive_position_rad,
 steering_velocity_rad_s, drive_velocity_rad_s,
 unix_time_s, steering_torque_nm, drive_torque_nm]
```

Field indices are unchanged, but the v3 position fields are continuous. The policy requires the v3 layout and reads only the first four kinematic fields. There is no separate `/trackstand/motor_torque` publisher. Positive torque means steering right or rear-wheel forward, following the configured torque signs.

The estimate is `torque_sign × B0 torque constant × measured Q-axis current`. Legacy nodes use a receive-only listener for C0/C2/A1 replies; MIT's sole CAN owner explicitly queries both motors' A1 current at 100 Hz and also consumes rear C0 replies. It never substitutes the commanded current or clamps measured torque. This is **motor-current-based feedback, not an independent shaft-torque measurement**; torque-constant accuracy and mechanical losses are not calibrated by this code.

During active C0 control, expect roughly one reply per motor per 200 Hz command cycle, subject to motor replies, CAN delivery, and scheduling. Combining feedback into one state topic adds no queries; MIT's explicit A1 queries are a separate backend feature. Each state message includes the latest independently received torque for each motor; a cached value can repeat while it is fresh. A torque field is `NaN` until valid current/B0 feedback exists or when its age exceeds `feedback_timeout_s` (default 50 ms). Missing torque does not suppress fresh position/velocity or replace feedback with a commanded torque. MIT can collect A1 current in disabled/dry-run operation; it does not fabricate powered torque. Disabled legacy motors and `dry_run` do not provide a continuous C0 feedback stream. `motor_status` includes `steering_torque_age_ms` and `drive_torque_age_ms` (`null` until the first valid torque reply).

`data[4]` is Unix wall-clock time at state publication, **not** motor acquisition time. Encoder and torque replies, and the two motors, are sampled independently; this is a latest-feedback snapshot, not a hardware-synchronized measurement. State publication uses a nominal 100 Hz timer for MIT, 200 Hz for legacy C0/C2, and requires new, fresh encoder pairs from both motors.

After rebuilding and sourcing the package, record:

```bash
ros2 bag record -o jitter_test \
  /trackstand/raw_action /trackstand/motor_command \
  /trackstand/motor_state \
  /trackstand/motor_control /trackstand/motor_event \
  /trackstand/policy_timing /trackstand/policy_event \
  /trackstand/observation /trackstand/imu \
  /trackstand/motor_fault /trackstand/motor_status \
  /trackstand/status /trackstand/kickstands_up \
  /trackstand/kickstands_command /rosout
```

Before an unsupported test, verify that `motor_state.data[5]` and `[6]` contain finite steering/rear torque feedback in `ros2 topic echo /trackstand/motor_state` during a secured, powered motor test. If feedback is absent, inspect the applicable A1/C0 replies on CAN; `NaN` is not evidence of zero torque.

`motor_control` samples all four essential motor diagnostic groups at nominal 200 Hz for MIT and 50 Hz for legacy C0/C2. MIT state/status/fault publication is 100 Hz; safety checks and CAN disable stay in its 200 Hz actuation loop. `policy_timing` reports 50 Hz policy timing, including disabled/faulted ticks. Their `Float64MultiArray.layout.dim[0].label` contains a versioned, ordered field list. Flags are 0/1 and unavailable values are NaN. Numeric diagnostic topics use best-effort QoS to avoid reliable diagnostic backpressure; `motor_state` retains its reliable depth-10 QoS for policy observations. Motor-control `sequence` counts control ticks; use timestamp spacing and the applicable backend's rate to interpret sequence gaps. Transition-only JSON `motor_event` and `policy_event` explain disable/latch causes without telemetry throttling. Send success means the local CAN socket accepted a frame, not confirmed motor execution. `outputs_active` is software state (also used to keep failed disables pending), not firmware run-mode feedback.

See [TELEMETRY.md](docs/TELEMETRY.md) for field semantics and [BIKE_DIAGNOSTIC_TESTS.md](docs/BIKE_DIAGNOSTIC_TESTS.md) for the staged real-bike procedure. Diagnostic publishing adds CPU work; verify 5 ms motor/20 ms policy cycle intervals on the Raspberry Pi while recording before relying on the timing.

The [2026-10-06 refactoring audit](docs/REFACTOR_AUDIT_2026-10-06.md) records shared-code changes, behavior-equivalence checks, unchanged hardware settings and remaining safety limitations.
