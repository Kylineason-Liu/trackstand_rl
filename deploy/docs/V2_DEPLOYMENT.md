# Trackstand-Balance-v2 deployment

V2 has a parallel ROS path: `policy_controller_v2` →
`/trackstand/motor_command_v2` → `motor_mit_io_v2`. The old launch and v1
model/backend remain the defaults. Run only one policy and one CAN owner.
Do not run replay scripts or other motor writers alongside deployment.

## Policy and command contract

The packaged policy is **model_50100**, DR-trained run **2026-11-02_22-40-01**.
ONNX SHA256: `a43583ec06d3c441d74d93057378fc25d8d79255a4ad312aeca6f59ad1c25c33`.
The adjacent `.contract.json` is required; startup checks its task, action
meaning, ordered observations, rear gains/scales and model checksum. An
11-input v1 torque policy is incompatible even though its shape matches.

Observations remain the same eleven quantities as v2 training:
`gravity_y, gravity_z, ang_vel_x/y/z, steering_pos, drive_pos,
steering_vel, drive_vel, sin_steer, cos_steer`. Gravity is quaternion-derived;
gyro and encoders use the existing bike transforms/signs. Positions are rad,
speeds rad/s. Rear displacement is rebased per RC-enabled episode, then
adapted to the measured PhysX 4π angle branch. Steering keeps its straight
startup reference. No linear velocity, integral-state input, or ±100 clamp
is added.

| Stage | Nominal frequency | Operation |
| --- | --- | --- |
| IMU publication | 100 Hz | Existing quaternion/gyro node and configuration |
| Policy and ROS targets | 50 Hz | Eleven inputs → deterministic bounded Beta means |
| Steering conditioning | 50 Hz | ×1.57 rad → ±1.57 clip → at most 0.478 rad change per target |
| Rear conditioning | 50 Hz | ×61.6 rad/s → ±61.6 velocity clip; hold between targets |
| CAN state queries | 100 Hz per query/motor | A3, A2, A1, AE; serialized by one CAN owner |
| Host rear integral | 200 Hz | Trapezoidal velocity-error integration, Ki=3.3 N·m/rad |
| MIT dispatch and safety | 200 Hz | Steering position MIT and rear velocity MIT |
| State/status/fault topics | Up to 100 / 100 / 100 Hz | State requires fresh encoder pairs from both motors |
| Control diagnostics | Nominal 200 Hz | Per-tick requests, wire fields, feedback and PI state |

Steering MIT requests position, velocity=0, Kp=10, Kd=0.2, feedforward=0.
Rear MIT requests position=0, Kp=0, velocity=target, Kd=0.30, feedforward=host I.
MIT Kd is the rear velocity controller's P gain, in N·m/(rad/s).
Its encoded value is approximately 0.3003663. The host compensates the
feedforward quantization residual through the velocity field; remaining
velocity quantization is reported. Native feedback/current signs remain −1
for both motors: policy-positive steering is right, rear rotation is forward.
No rear C0/C1 command is sent. Passive friction is physical, not an added
software torque subtraction.

The integral primes on first enable, retains history across target changes,
uses actual elapsed time with fresh cached A2, and resets on disable/fault.
Conditional anti-windup matches the tested PI structure, with a small
encoding reserve below ±6.28 N·m for its feedforward bound. Missed actuation
slots are skipped without catch-up bursts. Real scheduling is not guaranteed;
inspect measured intervals while recording. Isaac's 1200-Hz physics/PD rate
does **not** become a 1200-Hz Pi CAN requirement.

**Remaining actuator difference:** F0 is an encoding range, not a proven
firmware limit on total MIT P+I torque. Simulation clips the total rear torque
to ±6.28 N·m and steering PD to ±5.6 N·m. These equivalent internal hardware
clamps are unverified. Bounding host I does not bound total internal P+I.
The node reports the estimated total and clipped simulation reference
separately; it does not falsely label the latter as the sent/actual torque,
or silently change the velocity request to enforce it. Current-derived A1
torque is feedback, not an independent shaft-torque measurement.

## Build and start on the Pi

Start physically straight and stationary, kickstands down. Existing IMU,
CRSF receiver, motor IDs/signs and servo poses are copied unchanged into
`params_v2.yaml`. Source the Pi's installed ROS distribution first, then:

```bash
cd ~/trackstand_rl/deploy
colcon build --symlink-install --packages-select trackstand_deploy
source install/setup.bash
ros2 pkg executables trackstand_deploy
```

With CAN already configured at 1 Mbit/s, start the following in separate
terminals, each sourced and in `~/trackstand_rl/deploy`:

```bash
ros2 run trackstand_deploy kickstand_driver --ros-args --params-file source/trackstand_deploy/config/params_v2.yaml
ros2 run trackstand_deploy imu_publisher --ros-args --params-file source/trackstand_deploy/config/params_v2.yaml
ros2 run trackstand_deploy motor_mit_io_v2 --ros-args --params-file source/trackstand_deploy/config/params_v2.yaml
ros2 run trackstand_deploy rc_kickstand --ros-args --params-file source/trackstand_deploy/config/params_v2.yaml
ros2 run trackstand_deploy policy_controller_v2 --ros-args --params-file source/trackstand_deploy/config/params_v2.yaml
```

Alternatively, the complete v2 launch is:

```bash
ros2 launch trackstand_deploy trackstand_v2.launch.py
```

Use `-p dry_run:=true` on the **motor node** for read-only CAN/inference checks.
It still queries real feedback, but sends no motion, B1, F0 or CD writes.
For these checks, leave `kickstand_driver` stopped and support the bike;
RC-up still requests stands up even when motors are in dry-run. Dry-run
encoding uses planned rear ranges; F1 decoding uses the actual read ranges.
Do not change dry-run on a live node; parameters are startup-only.

The v2 policy additionally requires fresh motor status identifying the v2
velocity backend and matching gains/rates. It cannot arm from v1 motor state
alone. The existing RC flow remains: down disables; a ready up transition
starts balancing; down requests support immediately. Existing freshness,
lean/inversion, fault latch, five-degree arming/recovery and hard-fault
restart requirements remain. Simulator timeouts/yaw/drift scoring are not
added to hardware operation.

## References and persistent settings

Powered startup confirms both motors disabled, records original F0/CD values,
sends steering B1 once, verifies steering A3/F1 zero, and records a fresh
software rear A3 reference. **No rear B1** is sent. RC cycles do not rewrite
either motor's persistent origin. Rear A3 telemetry also handles int32 rollover.

Before persistent writes, originals are fsynced to
`~/.ros/trackstand_deploy/v2_mit_settings.json`. Rear F0 covers ±61.6 rad/s
plus codec headroom; both CD watchdogs are set to 500 ms. Normal shutdown
confirms both CF disables (using F1 if mode 5 is retained), restores and
readback-verifies temporary F0/CD values, then removes the journal. Steering
B1 is persistent and is not restored. Unconfirmed disable/restoration leaves
the recovery record and blocks the next powered start.

After a crash, with every other CAN writer stopped and the bike supported,
run this recovery-only command, then restart normally:

```bash
ros2 run trackstand_deploy motor_mit_io_v2 --ros-args \
  --params-file source/trackstand_deploy/config/params_v2.yaml \
  -p recover_pending_settings:=true
```

It disables both motors, restores the recorded originals and exits without
motion or B1. Never delete a pending journal to bypass recovery. A malformed
journal or failed readback requires inspection; restoration is not assumed.

## Telemetry and commissioning

`/trackstand/motor_command_v2` is a `Float64MultiArray` of
`[enable, steering_target_rad, drive_target_rad_s]`, with required label
`motor_command_velocity_v2:enable,steering_target_rad,drive_target_rad_s`.
The old torque topic is not subscribed to.

Other topic names are retained: `/trackstand/raw_action`, `/observation`,
`/policy_timing`, `/policy_event`, `/status`, `/motor_state`, `/motor_status`,
`/motor_fault`, `/motor_control`, `/motor_event` (all with `/trackstand` prefix).
The seven-field `motor_state_v3` remains unchanged, including A1×B0 torque.
`motor_control_mit_velocity_v2` replaces rear Nm-request/C0-law fields with
rad/s requests, A2 age/error, P torque, integral state/dt, estimated total,
clipped simulation reference, rear F0/F1 and encoded MIT fields. Field names
and units are embedded in its layout label. C0 current-command fields are
unavailable (`NaN`), not fabricated. Prepared packet fields need per-tick send
flags to establish socket submission; submission is not motor execution.
`mit_total_torque_clamp_verified` is explicitly false.

Record, for example:

```bash
ros2 bag record -o v2_deployment \
  /trackstand/imu /trackstand/kickstands_up /trackstand/kickstands_command \
  /trackstand/observation /trackstand/raw_action /trackstand/motor_command_v2 \
  /trackstand/motor_state /trackstand/motor_control /trackstand/motor_status \
  /trackstand/motor_fault /trackstand/motor_event /trackstand/status \
  /trackstand/policy_timing /trackstand/policy_event /rosout
```

Offline tests use ROS/CAN stand-ins and real ONNX inference. They do not
establish physical response or successful balance. Before an unsupported
run, verify read-only observations/arming, supported low-speed MIT signs,
PI reset on RC-down, actual A1 feedback and timing; then run the existing
[v2 identical-target replay](../experiments/V2_MOTOR_REPLAY.md) with independent
support and matching initial state. The known sim2sim gap and firmware
total-torque limits remain subjects for validation.
