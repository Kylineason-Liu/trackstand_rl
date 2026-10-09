# Hardware Commissioning Checklist

Keep the kickstands down, independently support the bike, and start exactly one
motor backend with `--ros-args -p dry_run:=true` for the first pass. MIT parameters
are startup-only; a runtime parameter change is not an emergency stop.

1. Derive normalized gravity from `/trackstand/imu`'s quaternion: upright should give approximately `gravity_y=0`, `gravity_z=-1`. These are also the first two policy-observation fields when inference runs; they are not named fields in `sensor_msgs/Imu`. A right lean must match the Isaac/MuJoCo observation sign.
2. Turn the handlebar by hand. `/trackstand/motor_state[0]` must increase when steering right; index 2 must be positive while turning right. Steering is motor ID 2, with `steering_encoder_sign=-1` and `steering_torque_sign=-1` based on the bike's C0/A2/A3 checks.
3. Rotate the rear wheel forward by hand. State indices 1 and 3 must increase; positive wheel velocity supplies positive estimated body-x velocity. Drive is motor ID 1, with `drive_encoder_sign=+1` and `drive_torque_sign=-1`. Confirm all four signs again after rebuilding on the bike.
4. Verify the startup log reports plausible, positive B0 torque constants for both motors. Missing or invalid values must leave output inhibited.
5. Verify the CP2102 receiver produces CRSF at 420000 baud. CH7 high must publish `true`, low must publish `false`, and middle must hold the last state after the eight-frame debounce.
6. Verify servo positions: up `(ID1=1648, ID2=2454)`, down `(ID1=1972, ID2=2085)`.
7. With `dry_run: false` and stands down, command no lift. Both motors must remain free/disabled.
8. Secure the frame in a rig before the first stand-up test. Validate steering correction direction and rear-wheel torque direction at limited current before an unsupported run.
9. Before using the new ±90° steering command limit, verify the full steering travel, wiring clearance, and absence of interference on the supported bike with steering motors disabled. The previous software fault limit was ±30°; a software limit does not prove the mechanism is safe at ±90°.
10. Do not expect a 30° roll stop: current policy checks inversion only (`gravity_body[2] > 0`), besides readiness/faults; YAML `max_roll_rad` is unused. MIT has no measured-angle or overspeed stop, unlike C0/C2. Confirm physical power isolation and actual firmware current/torque protection before powered MIT tests.
11. Before powered MIT startup, verify CAN is UP at the motor bitrate and steering is straight/stationary; B1/F0 modify persistent motor settings. With the policy running, RC down requests disable and stands down without waiting for acknowledgements. MIT must report both motor confirmations and no pending disable; retained CF modes 1–5 need a fresh inactive/fault-free F1. Check physical free state separately.
12. To recover an ordinary MIT latch, keep enable off until disable is confirmed, encoder/B0 feedback is fresh, and unwrapped steering is strictly inside ±5°. A hard fault (`mit_restart_required=true`) requires restart. A quick down/up can latch another fault; do not assume any switch cycle guarantees recovery.

The deployment cannot validate physical mounting, wiring, or motor direction without the bicycle. A wrong sign can make the learned controller destabilizing; do not treat software unit tests as a substitute for this checklist.
