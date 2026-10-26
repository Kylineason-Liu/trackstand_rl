# Real-Bike Diagnostic Test Procedure

These steps collect evidence about the observed vibration before another unsupported balancing attempt. They assume this repository is `~/trackstand_rl`, ROS 2 Humble is installed on the Raspberry Pi, and the existing hardware checklist has been completed. Adjust only the repository and ROS installation paths if yours differ. Keep the bicycle independently restrained, with room for steering and rear-wheel motion, and have physical power isolation available. RC-down controls the motors **only when `policy_controller` is running**; a manually published `/trackstand/motor_command` bypasses that RC interlock.

## 1. Install and identify what is running

Stop old deployment nodes before rebuilding. In a Pi terminal:

```bash
cd ~/trackstand_rl/deploy
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select trackstand_deploy
source install/setup.bash
ros2 pkg prefix trackstand_deploy
ros2 pkg executables trackstand_deploy
```

Keep CH7 in its **down** position before starting nodes. Check running nodes and parameters after they start, as directed below. Verify exactly one `/trackstand_motor_io` node and at most one `/trackstand_policy_controller`; multiple publishers to `/trackstand/motor_command` invalidate the test. `motor_io` alone does not read RC commands. Save parameter dumps and startup console logs with the bag. The policy node checks the ONNX SHA-256 at startup. Note the B0 torque constants and steering encoder zero logged by `motor_io`. Do not restart `motor_io` mid-test without recording that its encoder zeros were re-established.

### Selecting a backend for these diagnostic examples

The Stage A–C commands below use `motor_io`. To test MIT, substitute
`motor_mit_io` in **each motor-node command**, retaining the parameter file
and startup `--ros-args -p dry_run:=true/false` override. Never start both.
The v1 all-node launch selects MIT; these staged examples explicitly select their backend. `No executable found` means the selected
installation lacks its wrapper: synchronize `setup.py`, rebuild/source,
and inspect the package prefix/executable list before investigating CAN.

| Backend | Policy command Hz | Actuation Hz | State/status/fault | Control diagnostics Hz |
|---|---:|---:|---|---:|
| C0 (`motor_io`) | 50 | 200 | State up to 200 Hz; status 2 Hz; fault each control tick | 50 |
| MIT (`motor_mit_io`) | 50 | 200 | State up to 100 Hz; status/fault 100 Hz | 200 |

MIT polls A3/A2/A1 for each motor at nominal 100 Hz and receives automatic
F1/C0 replies continuously. In Stage B, inspect the MIT encoded target/gains,
not steering host-PD torque fields: those are NaN. Prepared targets/current
may exist without transmission; `*_send_attempted`/`*_send_success` must be 0
in dry-run. MIT diagnostics normally have ~5 ms spacing, not C0's ~20 ms.
A1 current can be recorded even in MIT dry-run; it is not commanded torque.

Keep steering straight/stationary and enable off during powered MIT startup:
the node verifies CAN UP, CF/F1, writes persistent B1/F0 once, and checks
origin/ranges. Its parameters are startup-only/read-only; restarting with
different values is required. `ros2 param set ... dry_run true` cannot stop it.
RC down/up does not rewrite the startup origin.

Current v1 policy checks |gravity_y| > sin(0.5236) or gravity_z > 0, alongside input/motor/inference gates. It starts only within 5 degrees of the fixed straight steering reference. Steering target clip/slew is ±1.57 rad and 23.9 rad/s, advanced once per 50-Hz inference; the motor nodes hold the result. MIT retains its lack of measured steering-angle/overspeed stops and verified total-PD torque cap; C0/C2 retain their measured-limit checks. Do not use these software
gates as the test's physical restraint or emergency stop.

For MIT Stage C, a publisher-end watchdog timeout now latches the ordinary
motor fault rather than automatically resuming. After every command trial,
publish enable off, confirm `disable_pending=false`, both per-motor
confirmations, fresh encoder/B0 readiness, steering strictly inside ±5°
and a cleared motor fault **before** the next enable. With policy absent,
the RC switch cannot perform this reset; use the explicit disabled command
shown below. `mit_restart_required=true` requires supported restart, not
another enable. CF mode 0 confirms directly; retained modes 1–5 need fresh
inactive/fault-free F1. Verify physical free state independently. A lack of
terminal ROS event after Ctrl-C does not settle whether CAN cleanup succeeded.

## 2. Start a bag for each test

Run this from `~/trackstand_rl/deploy` in a separate terminal after sourcing `install/setup.bash`. Use a different, previously nonexistent output directory for each stage, for example `../rosbags/diag_a_disabled` and `../rosbags/diag_c_motor_bench`. Start recording before changing RC or motor commands, and stop the recorder cleanly with Ctrl-C after the stage:

```bash
mkdir -p ../rosbags
ros2 bag record -o ../rosbags/diag_a_disabled \
  /trackstand/imu /trackstand/motor_state \
  /trackstand/motor_control /trackstand/motor_event \
  /trackstand/observation /trackstand/raw_action /trackstand/motor_command \
  /trackstand/policy_timing /trackstand/policy_event \
  /trackstand/motor_fault /trackstand/motor_status /trackstand/status \
  /trackstand/kickstands_up /trackstand/kickstands_command /rosout
```

Change `-o` for each stage. Recorder discovery must list the active high-rate topics. Verify recorded topic names and counts afterward using `ros2 bag info ../rosbags/diag_a_disabled`; a successful bag file alone does not prove the best-effort telemetry was captured. Torque fields `motor_state.data[5]` and `[6]` can be `NaN` with motors disabled or `dry_run=true`; missing/stale current feedback is not zero torque. Event topics may be quiet until a state change. The names and units of numeric telemetry fields are documented in [TELEMETRY.md](TELEMETRY.md).

## 3. Stage A: disabled sensors, RC, and steering zero

With CH7 down, start these nodes in separate terminals, each sourced with `install/setup.bash` and run from `~/trackstand_rl/deploy`:

```bash
ros2 run trackstand_deploy kickstand_driver --ros-args --params-file source/trackstand_deploy/config/params.yaml
ros2 run trackstand_deploy imu_publisher --ros-args --params-file source/trackstand_deploy/config/params.yaml
ros2 run trackstand_deploy rc_kickstand --ros-args --params-file source/trackstand_deploy/config/params.yaml
ros2 run trackstand_deploy motor_io --ros-args --params-file source/trackstand_deploy/config/params.yaml -p dry_run:=true
```

Leave the policy node stopped. Record about 20–30 seconds with the bike motionless. Verify kickstands stay down, steering and rear speeds are near zero, IMU orientation/rates are stable, both encoder streams update, `gate_enabled=0`, and no C0 control sends occur. Slowly tilt right/upright/left/upright while the bike is supported; note the exact sequence and approximate times. Move steering right and left by hand, then return it to the start; note which direction makes `motor_state.data[0]` increase. If safe in your fixture, rotate the raised rear wheel by hand and check the sign of `motor_state.data[3]`. Record another motionless interval. Check for increasing per-motor query-failure counters, stale-feedback events, or large 5 ms motor-cycle gaps. `motor_state.data[4]` is state publication time, not encoder acquisition time.

After these nodes start, run the checks in another sourced terminal. Confirm `dry_run: true`, the expected motor IDs/signs and IMU transform, and no unexpected command publisher:

```bash
ros2 node list
ros2 param dump /trackstand_motor_io
ros2 topic info /trackstand/motor_command --verbose
```

## 4. Stage B: policy and telemetry without motor current

For this stage, keep the bike independently restrained. Stop `kickstand_driver` while CH7 is down so a policy kickstand-up request cannot move the physical stands; verify the driver node is absent and the stands are down. Start `policy_controller` and leave `motor_io` in `dry_run:=true`:

```bash
ros2 run trackstand_deploy policy_controller --ros-args --params-file source/trackstand_deploy/config/params.yaml
```

Once this node has started, capture its effective policy path and limits:

```bash
ros2 param dump /trackstand_policy_controller
ros2 param get /trackstand_policy_controller model_path
```

Record a new bag. With CH7 down, confirm disabled timing messages but no raw actions. Briefly request CH7 up and then down while the bike remains restrained. Confirm the mode and events transition and the 50 Hz observation/raw-action/command stream during enable. With the explicitly selected C0 backend, the 50 Hz motor record shows computed steering/drive torque with `send_attempted=0`; MIT uses the fields/rate described above. Disabled motor commands continue outside enable. Inspect policy timing, target clipping/slew, applicable torque calculations, command gaps and feedback ages. `cycle_interval_ms` reflects the nominal 200 Hz motor loop (~5 ms); C0 record timestamps are ~20 ms apart, MIT ~5 ms. Sequence gaps depend on sampling and scheduling. Events/safety are not diagnostic-rate throttled. If kickstands actually move or a motor receives an actuation command in dry-run, stop and resolve the mismatch.

## 5. Stage C: small powered motor commands, policy stopped

Only after reviewing Stages A and B, return CH7 down, stop `policy_controller`, and ensure the physical kickstands are down. Stop the dry-run `motor_io` process and restart it in its terminal with:

```bash
ros2 run trackstand_deploy motor_io --ros-args --params-file source/trackstand_deploy/config/params.yaml -p dry_run:=false
```

Secure the bike independently; allow the rear wheel and steering to move freely in a controlled range. Start a fresh bag and verify there is no other publisher on `/trackstand/motor_command`. The following examples publish for about 2 seconds at 50 Hz. End each publisher before starting another; the 80 ms watchdog will then send zero-current and disable frames. In the first command, verify finite steering/rear torque in `motor_state.data[5]` and `[6]`, fresh torque ages in `motor_status`, and `*_send_success=1`. These commands send to **both** motors; v1 zero rear torque requests no software damping, while the real assembly retains its physical friction.

```bash
ros2 topic pub --rate 50 --times 100 --qos-durability volatile \
  /trackstand/motor_command std_msgs/msg/Float64MultiArray '{data: [1.0, 0.0, 0.0]}'
ros2 topic pub --rate 50 --times 100 --qos-durability volatile \
  /trackstand/motor_command std_msgs/msg/Float64MultiArray '{data: [1.0, 0.02, 0.0]}'
ros2 topic pub --rate 50 --times 100 --qos-durability volatile \
  /trackstand/motor_command std_msgs/msg/Float64MultiArray '{data: [1.0, -0.02, 0.0]}'
```

If the steering tests were smooth and correctly signed, perform brief rear tests with the wheel clear of the ground and free to spin:

```bash
ros2 topic pub --rate 50 --times 50 --qos-durability volatile \
  /trackstand/motor_command std_msgs/msg/Float64MultiArray '{data: [1.0, 0.0, 0.05]}'
ros2 topic pub --rate 50 --times 50 --qos-durability volatile \
  /trackstand/motor_command std_msgs/msg/Float64MultiArray '{data: [1.0, 0.0, -0.05]}'
ros2 topic pub --once --qos-durability volatile \
  /trackstand/motor_command std_msgs/msg/Float64MultiArray '{data: [0.0, 0.0, 0.0]}'
```

The values are diagnostic examples, not certified safe hardware limits. Stop immediately if signs are wrong, torque/current feedback is absent or implausible, a joint approaches its physical limit, or vibration begins. Record which command and time caused it. Do not hold a joint against a fixture or rely on the kickstands to restrain torque. RC-down does **not** override these manually published commands when the policy node is absent; stop the publisher and isolate motor power if necessary. The final disable message is explicit; use it after **each** trial when testing MIT and wait for its safe fault reset before the next enable. The watchdog also requests disable after command publication stops. Check `/trackstand/motor_event` for `command_timeout` at each intended gap. Socket-send success is not proof of physical disable; verify mechanically before touching the bike.

## 6. Hand off the evidence

Send the stage bags, ROS node parameter dumps, policy model filename/hash, and a short note listing the RC changes and each command's start/end time. Include any console/CAN errors and whether vibration occurred with sensors only, with policy dry run, or with small powered commands. If `candump` is installed, a simultaneous passive `candump can0` capture during Stage C helps compare outgoing C0/CF traffic with motor replies; it should not send CAN commands. Record video/audio with a visible time cue if practical.

Do not start an unsupported policy-balancing run until these stages show correct signs, fresh feedback, and stable command timing. If Stages A–C already reproduce the vibration, their bags can locate it without another lift of the kickstands.
