# Trackstand Bicycle Deployment

This ROS 2 workspace deploys `Trackstand-Balance-v0` on the Tsinghua bicycle. It is a stationary-balancing controller: the RC throttle and steering sticks are intentionally ignored. The RC kickstand switch is the operating authority.

## Runtime flow

1. Startup commands both kickstands down and keeps both GIM motors disabled.
2. CRSF channel 7 high requests the known up positions; low requests down and the middle position holds the last state. A change is accepted after eight matching frames.
3. With fresh IMU and encoder data, the 50 Hz ONNX policy starts immediately while the stands lift.
4. RC high immediately disables/freewheels both motors and commands the stands down.
5. A stale sensor, stale RC stream, excessive roll, or steering-limit violation disables and latches balancing. Cycle the RC switch down before trying again.

The motor I/O node polls both motors' A3 multi-turn position and A2 full-range speed frames and closes the simulated actuator laws at 200 Hz. It reads each motor's B0 torque constant before allowing C0 current commands. The policy is the Raspberry-Pi-friendly ONNX export of run `2026-09-17_19-46-37`, checkpoint `model_41050.pt`; action scaling is applied in code.

## Build and run

```bash
cd deploy
python3 -m pip install -r requirements.txt
colcon build --symlink-install
source install/setup.bash
sudo ip link set can0 up type can bitrate 1000000
ros2 launch trackstand_deploy trackstand.launch.py
```

The checked-in hardware paths match `ros2_ws`: the CH340 by-id path is the IMU, the CP2102 by-id path is the CRSF receiver, and `/dev/ttyAMA2` is the servo bus. Edit `source/trackstand_deploy/config/params.yaml` only if the hardware identities change. Build dependencies include ROS 2 `rclpy`, `sensor_msgs`, and `std_msgs`.

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

Run with the stands down and inspect `/trackstand/observation`, `/trackstand/status`, and `/trackstand/motor_state`. Confirm the identity IMU transform, all four motor sign parameters, steering zero, kickstand positions, B0 torque constants, and CRSF channel values. Use `dry_run: true` for the first data-flow check. Set `status_rate_hz` above zero only during kickstand bench checks; it is disabled during balancing so serial feedback cannot delay commands. Do not lift the stands until the sign checklist in `docs/HARDWARE_CHECKLIST.md` passes.
