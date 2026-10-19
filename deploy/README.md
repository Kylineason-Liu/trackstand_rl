# Trackstand Bicycle Deployment

This ROS 2 workspace deploys `Trackstand-Balance-v0` on the Tsinghua bicycle. It is a stationary-balancing controller: the RC throttle and steering sticks are intentionally ignored. The RC kickstand switch is the operating authority.

## Runtime flow

1. The policy starts disabled and commands both kickstands down; the selected motor backend performs its own startup checks.
2. CRSF channel 7 high requests the known up positions; low requests down and the middle position holds the last state. A change is accepted after eight matching frames.
3. With fresh IMU and encoder data, the 50 Hz ONNX policy starts immediately while the stands lift.
4. RC low immediately publishes motor-enable off and commands the stands down. Motor I/O then attempts CAN disabling; the policy does not wait for motor acknowledgements before lowering the stands.
5. Stale sensors, stale RC, motor faults, inference errors, or inversion disable and latch balancing. Cycle the RC switch down before trying again.

The default `motor_io` backend polls both motors' A3 multi-turn position and A2 full-range speed frames and closes the nominal actuator laws at 200 Hz. MIT uses the separate rate profile below. Both require each motor's B0 torque constant before enable. The default policy is the Raspberry-Pi-friendly, self-contained ONNX export of run `2026-09-22_19-45-06`, checkpoint `model_91850.pt`; its SHA-256 is checked at startup and action scaling is applied in code. The host-side explicit servo is an approximation, not an exact reproduction of Isaac's implicit actuator.

Raw policy actions are not clipped to `[-1, 1]`, matching Trackstand-Balance-v0's RSL-RL configuration. Steering is scaled by `1.57 rad` and drive torque by `6.28 N·m`.

**Experimental change (2026-10-13):** the shared policy's steering target clamp and MIT backend's ±90° target clamp, 23.9 rad/s target slew, and both ±6.28 N·m rear torque clamps are **commented out by request**, not deleted. MIT now submits the scaled steering target directly and rear torque `6.28*raw_action - 0.05*rear_speed`, subject to unchanged encoding/gate checks. These requests can greatly exceed motor capability: 91850 requests about **105 N·m at an ideal upright start**. Do not run this version powered on an unsupported bike; validate independent motor current protection and emergency stopping first. Legacy `motor_io`/`motor_io_position` retain their own clamps/slew/protections. MIT still has no proven total-PD ±5.6 N·m clamp. See [the MIT guide](docs/MOTOR_MIT_IO.md) for remaining protocol bounds and telemetry semantics.

The current policy has **no 30° roll cutoff**. With fresh inputs it checks inversion as `gravity_body[2] > 0`, plus sensor/RC timeouts and motor faults. The leftover `max_roll_rad` entry in `params.yaml` is not declared or read and cannot restore the removed cutoff. A policy fault disables balancing but does not automatically lower the kickstands; RC down does that. Motor safety is backend-specific:

| Motor backend | Measured steering-angle stop | Measured steering/rear overspeed stops |
|---|---|---|
| `motor_io` / `motor_io_position` | Retained at ±90° | Retained above 26.29 / 67.76 rad/s |
| `motor_mit_io` | Removed | Removed |

Legacy backends' ±90° target clamps are not measured-overtravel stops; MIT's target clamp is currently disabled. `steering_fault_clear_rad` still requires disabled steering **strictly inside ±5°** of startup zero to clear an ordinary motor latch; MIT also requires confirmed disable and fresh encoder/B0 readiness. Hard MIT faults require restart. Verify physical clearance, emergency stopping and firmware protection with independent support before powered tests.

Motor signs follow the bike's physical C0/A2/A3 checks: steering encoder/current `-1/-1`, rear encoder/current `-1/-1`. The 2026-10-12 rear check supersedes the earlier `+1` encoder assumption: forward rotation produces negative native A2 velocity and decreasing A3 counts, while positive native C0 current drives backward. Positive policy steering targets and measured positions mean right; positive policy rear torque, position change and wheel speed mean forward. Rebuild/source and restart the selected motor node after this correction, then confirm published signs with the bike independently supported before balancing. Frequencies, policy selection, steering signs and torque-command signs are unchanged.

## Build and run

```bash
cd deploy
python3 -m pip install -r requirements.txt
colcon build --symlink-install
source install/setup.bash
sudo ip link set can0 up type can bitrate 1000000
ros2 launch trackstand_deploy trackstand.launch.py
```

The checked-in hardware paths match `ros2_ws`: the CH340 by-id path is the IMU, the CP2102 by-id path is the CRSF receiver, and `/dev/ttyAMA2` is the servo bus. Edit `source/trackstand_deploy/config/params.yaml` only if the hardware identities change. ROS dependencies include `rclpy`, `rcl_interfaces`, `sensor_msgs`, and `std_msgs`. The default launch selects `motor_io`, not MIT.

An optional `motor_io_position` executable uses firmware **C2 absolute position control** for steering while retaining the rear-wheel C0 controller and existing safety gates. Run it **instead of**, never alongside, `motor_io`; the default launch remains unchanged. Targets use the latched raw A3 startup zero plus the converted steering angle, not the current measured position. Native increasing steering position turns left; `steering_encoder_sign=-1` maps policy-positive steering into decreasing native counts. Start with steering straight and in dry-run. Powered C2 requires B2/B3 limit readback; the former C3 reference-verification flag is removed. See [C2 node instructions and telemetry](docs/MOTOR_IO_POSITION.md) before using it on the bike.

The separate `motor_mit_io` executable has three rate divisions: **50 Hz policy commands**, **100 Hz motor-state queries and state/status/fault topics**, and **200 Hz steering MIT, rear C0, safety checks and control telemetry**. Steering uses native Kp=10, Kd=0.2, requested velocity=0, and feedforward torque=0. It inherits directly from ROS `Node` and does not import or inherit either legacy motor backend; only backend-neutral protocol/contract/telemetry helpers are shared. Its startup follows the verified MIT experiment's CF/B1/F0/A3/F1 checks. MIT cannot reproduce the host-side ±5.6 N·m steering torque clamp from `motor_io`; the protocol does not establish that the C2 current limit applies to total MIT PD torque. See [MIT node instructions, rate parameters and limits](docs/MOTOR_MIT_IO.md).

To select MIT, start nodes individually with **only** this motor backend, first in dry-run:

```bash
ros2 pkg executables trackstand_deploy
ros2 run trackstand_deploy motor_mit_io --ros-args \
  --params-file source/trackstand_deploy/config/params.yaml -p dry_run:=true
```

If `ros2 run` says `No executable found`, rebuild the synchronized package and source its installation; sourcing alone does not create the entry point. Check `ros2 pkg prefix trackstand_deploy` for an unintended overlay. MIT's 21 parameters are startup-only/read-only; changing `dry_run` at runtime is not a stop mechanism. Powered MIT requires CAN already UP, steering straight and stationary, and kickstands down during persistent-origin setup. Confirm CF/F1 disable handling as described in the MIT guide; neither send success nor an inactive-MIT reply alone proves a physical free state.

## Observation contract

The policy input order is:

```text
[gravity_y, gravity_z, gyro_x, gyro_y, gyro_z,
 linear_velocity_x, linear_velocity_y,
 steering_position, rear_wheel_position,
 steering_velocity, rear_wheel_velocity,
 sin(steering_position), cos(steering_position)]
```

All angles are radians, rates are rad/s, velocity is m/s, and gravity is unitless. The IMU supplies gravity and angular rate. Encoders supply the four joint channels. With no external odometry allowed, longitudinal velocity is the no-slip estimate `wheel_radius * rear_wheel_velocity`; lateral velocity is zero under the bicycle nonholonomic constraint. These are the only two approximated channels; no synthetic observation noise is added.

## Before first unsupported test

Run with the stands down and inspect `/trackstand/observation`, `/trackstand/status`, and `/trackstand/motor_state`. Confirm the configured IMU transform (currently `[1.42, 0.0, 0.0]` degrees in `params.yaml`), all four motor sign parameters, steering zero, kickstand positions, B0 torque constants, and CRSF channel values. Use `dry_run: true` for the first data-flow check. Set `status_rate_hz` above zero only during kickstand bench checks; it is disabled during balancing so serial feedback cannot delay commands. Do not lift the stands until the sign checklist in `docs/HARDWARE_CHECKLIST.md` passes.

## Record motor feedback

All three motor nodes (`motor_io`, `motor_io_position`, and `motor_mit_io`) publish position, velocity, and measured torque together on `/trackstand/motor_state` (`std_msgs/msg/Float64MultiArray`). The versioned layout label is `motor_state_v2:<fields>`, with data ordered as:

```text
[steering_position_rad, drive_position_rad,
 steering_velocity_rad_s, drive_velocity_rad_s,
 unix_time_s, steering_torque_nm, drive_torque_nm]
```

The first five fields retain their previous indices, and the policy still reads only the first four. There is no separate `/trackstand/motor_torque` publisher. Positive torque means steering right or rear-wheel forward, following the configured torque signs.

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
