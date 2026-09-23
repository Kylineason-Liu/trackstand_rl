# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **self-balancing bicycle/motorcycle project** from Tsinghua University. The goal is to build a physical self-balancing bike using a Raspberry Pi 5 running Ubuntu 24.04 with ROS2 (Jazzy) and a real-time kernel, and to study the control theory behind it via MuJoCo simulation.

## Repository Structure

```
├── 整体顺序.md                        # Overall task sequence (Chinese)
├── assign_1_Raspberry Pi 5/           # Raspberry Pi setup guide + system image
├── assign_2_Hardwares/                # Hardware datasheets, drivers, and vendor code
│   ├── 1.IMU/                         #   YahBoom IMU (USB serial)
│   ├── 2.Servo【ST3250C001】/         #   FEETECH serial bus servos (UART)
│   ├── 3.RearSteer Servo【GIM3510-8】/ #   Rear/steer motors (CAN bus)
│   └── 4.Remote Controller/           #   Hot RC 10A + SBUS receiver (USB serial)
├── assign_3_Assemble/                 # Physical assembly instructions
├── assign4&5_bike_mujocosim_with_control/  # Simulation & control theory
│   ├── model/orange_bike/             #   MuJoCo XML multi-body model
│   ├── nominal_bike_control/          #   Python package: 3-state linear model + LQR/pole-placement
│   ├── tutorials/                     #   7 Jupyter notebooks (modeling → closed-loop sim)
│   └── tests/                         #   Unit tests for nominal_bike_control
└── source/                            # ROS2 nodes deployed on the Raspberry Pi
    ├── imu_publish.py                 #   IMU → ROS2 topics (imu/data_raw, imu/mag, imu/euler)
    ├── servo_control/                 #   Servo driver + keyboard control node
    └── remote_control/                #   SBUS reader, motor controller (CAN), servo RC controller, joystick bridge
```

## Build / Run Commands

### Simulation (assign4&5)

```bash
# Run unit tests for the control package
python -m unittest -v tests.test_nominal_bike_control

# Run Jupyter tutorials (from project root or tutorials/ directory)
jupyter notebook tutorials/
```

Key dependencies for simulation: `numpy`, `scipy`, `matplotlib`, `mujoco`, `mediapy`. Python 3.12.

### Raspberry Pi (source/)

All `source/` nodes are ROS2 Python packages installed via `setup.py` (or colcon). Key packages:

#### remote_control — SBUS, motor, servo hardware nodes

```bash
# Build a single package (run on Raspberry Pi)
cd source/remote_control/remote_controller
pip install -e .

# Or use colcon (from source/remote_control/)
colcon build --packages-select remote_controller controller_bridge
source install/setup.bash

# Run nodes
ros2 run remote_controller sbus_reader        # SBUS → /joy topic
ros2 run remote_controller motor_controller   # /cmd_vel → CAN motor commands
ros2 run remote_controller servo_controller   # /joy → /servo_command (RC control of servos)
ros2 run controller_bridge joy_to_cmd         # /joy → /cmd_vel
ros2 run servo_control servo_driver           # UART servo hardware driver
ros2 run servo_control keyboard_control       # Keyboard → /servo_command
python3 source/imu_publish.py                 # IMU → ROS2 topics
```

Hardware setup on the Pi before running nodes:
```bash
# Enable CAN (once per boot)
sudo ip link set can0 up type can bitrate 1000000

# Install CAN tools (once)
sudo apt install can-utils

# Enable UART2 (add "dtoverlay=uart2-pi5" to /boot/firmware/config.txt, then reboot)
```

#### bike_balance — 50 Hz self-balancing controller (tutorial 07 → real bike)

```bash
# Build
cd source/bike_balance
pip install -e .

# Run with pole-placement (default)
ros2 run bike_balance balance_node

# Run with LQR
ros2 run bike_balance balance_node --ros-args -p method:=lqr

# Run with params file
ros2 run bike_balance balance_node --ros-args --params-file config/balance_params.yaml

# Enable balancing (required unless enable_on_start:=true)
ros2 topic pub /bike_balance/enable std_msgs/Bool "data: true"

# Monitor status
ros2 topic echo /bike_balance/status
```

Control architecture (mirrors tutorial 07 notebook):
- **Outer loop (50 Hz)**: IMU → [steer_angle, roll, roll_rate] → pole/LQR gain-scheduled feedback → desired steering angular velocity
- **Inner loop (50 Hz)**: (steer_rate_ref − measured steer_rate) → PI → steer motor speed cmd (CAN `#C1`)
- **Rear loop (50 Hz)**: (speed_ref/wheel_radius − measured rear_rate) → P → drive motor speed cmd (CAN `#C1`)
- **Speed profile**: linear ramp to `target_speed` over `acceleration_time`
- **Gain scheduling**: A, Bu recomputed every tick from current forward speed (min clamped at `min_scheduling_speed`)
- **Safety**: emergency stop if |roll| > `max_roll_angle_deg`; auto-disable if CAN feedback stale > 0.5 s

Key parameters (see `config/balance_params.yaml` for full list):
| Parameter | Pole default | LQR default | Description |
|---|---|---|---|
| `pole_wc` | −5.0 | — | All 3 closed-loop poles |
| `lqr_Q_diag` | — | [4, 100, 2] | State cost (δ, φ, φ̇) |
| `lqr_R_diag` | — | [10] | Control cost |
| `steer_kp` / `steer_ki` | 0.01 / 0.01 | same | Inner PI gains |
| `rear_kp` | 0.1 | same | Rear wheel P gain |
| `max_steer_velocity` | 10.0 | same | Outer-loop output clamp (rad/s) |
| `steer_speed_scale` | 1.0 | same | rad/s → motor cmd units |
| `drive_speed_scale` | 1.0 | same | rad/s → motor cmd units |

## Architecture: ROS2 Topic Graph

```
Hardware                    ROS2 Node                 ROS2 Topic           Consumer
─────────────────────────────────────────────────────────────────────────────────
IMU (USB)         →    imu_publish.py          →  /imu/data_raw
                                                  /imu/mag
                                                  /imu/euler

SBUS receiver     →    sbus_reader             →  /joy (sensor_msgs/Joy)  →  joy_to_cmd → /cmd_vel
(USB serial)                                                              →  servo_controller → /servo_command

Servo (UART2)     ↔    servo_driver            ↔  /servo_command (String)
  GPIO4/GPIO5                                      /servo_status  (String JSON)

Motors (CAN bus)  ←    motor_controller         ←  /cmd_vel (geometry_msgs/Twist)
  GIM3510/GIM4305    (reads CAN feedback)

                    ────── bike_balance (alternative motor control) ──────

IMU (USB)         →    balance_node             →  steer motor (CAN #C1)
  /imu/euler                                          drive motor  (CAN #C1)
  /imu/data_raw                          publishes  /bike_balance/status
                                                     /bike_balance/debug
                    ←  CAN encoder feedback (steer angle + rear speed)
```

- `/joy` carries 16-channel SBUS raw values (axes field), published at 50 Hz.
- `/cmd_vel` convention: `linear.x` = drive motor RPM, `linear.y` = steer motor speed.
- `/servo_command` is text-based: `m <id> <pos> <speed> <acc>` for position, `t <id> <0|1>` for torque.
- `/bike_balance/status` publishes JSON with all control states (roll, steer, speed, gains) at 2 Hz.
- `/bike_balance/enable` (Bool) arms/disarms the balance controller.

## Key Technical Details

### nomimal_bike_control package

The 3-state linear bicycle model used in simulation:

```
State:  x = [steering_angle, roll_angle, roll_rate]^T  (rad, rad, rad/s)
Input:  u = steering_angular_velocity                   (rad/s)
Model:  x_dot = A(v) x + Bu(v) u
```

- `LinearBicycleModel` / `ScaleBikeModel` — discrete-time model with speed-dependent A, Bu matrices. Default min forward speed clamped to 1 m/s to avoid model degeneracy at low speeds.
- `NominalBikeController` — three gain methods: `place_multiple_poles`, `place_distinct_poles`, `lqr`. Control law: `u = u_eq + K (x_eq - x)`.
- Equilibrium points are solved from `A x_eq + Bu u_eq = 0` subject to `Co x_eq = reference`. Non-zero steering angle requires non-zero roll angle.
- MuJoCo XML uses torque-driven actuators at 0.5 ms physics step; control runs at 50 Hz (0.02 s). An inner PI loop converts steering angular velocity commands to steering motor torque.
- Default model parameters differ significantly from the XML — the nominal model is a baseline, not a linearization of the XML. Use `BicycleParameters` explicitly for better alignment.

### CAN Motor Protocol

- CAN bitrate: 1,000,000
- Default motor CAN ID: `001` (drive), `002` (steer, after reconfiguration)
- Command format: `cansend can0 <ID>#C1<4-byte-little-endian-hex>`
- Feedback: motors send position/speed via CAN frames; parsed from `candump` output
- Encoder resolution: 16384 counts/rev

### Servo Protocol (FEETECH SMS_STS)

- UART at 1,000,000 baud on `/dev/ttyAMA2` (GPIO4=TX, GPIO5=RX)
- IDs: 1 and 2 (set via FD.exe Windows tool)
- Register-based protocol (read/write memory addresses 40–70)
- Position range: 0–4095 (0–360°)

### SBUS Protocol

- 25-byte standard frames or 35-byte extended frames at 115200 baud
- 16 proportional channels, decoded to published as `Joy.axes`
- Frame start marker: `0x0F`

## Notes for Development

- All angles in the simulation are in radians internally; plots convert to degrees. The real bike uses degrees for servo positions.
- The nominal model assumes small angles and constant forward speed — it is a pedagogical baseline, not an accurate model of the physical bike.
- Before running real-bike experiments, IMU coordinate alignment and gyro bias calibration must be done (see `tutorials/06_IMU安装误差_坐标标定与状态构造.ipynb`).
- ROS2 packages under `source/` use `rclpy` and standard message types (`sensor_msgs`, `geometry_msgs`, `std_msgs`). Parameters are declared via `declare_parameter` for runtime configuration.
