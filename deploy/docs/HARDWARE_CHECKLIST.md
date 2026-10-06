# Hardware Commissioning Checklist

Keep the kickstands down and set `motor_io.dry_run: true` for the first pass.

1. Confirm `/trackstand/imu` reads approximately `gravity_y=0`, `gravity_z=-1` when upright. A right lean must have the same `gravity_y` sign as the Isaac/MuJoCo observation.
2. Turn the handlebar by hand. `/trackstand/motor_state[0]` must increase when steering right; index 2 must be positive while turning right. Steering is motor ID 2, with `steering_encoder_sign=-1` and `steering_torque_sign=-1` based on the bike's C0/A2/A3 checks.
3. Rotate the rear wheel forward by hand. State indices 1 and 3 must increase; positive wheel velocity supplies positive estimated body-x velocity. Drive is motor ID 1, with `drive_encoder_sign=+1` and `drive_torque_sign=-1`. Confirm all four signs again after rebuilding on the bike.
4. Verify the startup log reports plausible, positive B0 torque constants for both motors. Missing or invalid values must leave output inhibited.
5. Verify the CP2102 receiver produces CRSF at 420000 baud. CH7 high must publish `true`, low must publish `false`, and middle must hold the last state after the eight-frame debounce.
6. Verify servo positions: up `(ID1=1648, ID2=2454)`, down `(ID1=1972, ID2=2085)`.
7. With `dry_run: false` and stands down, command no lift. Both motors must remain free/disabled.
8. Secure the frame in a rig before the first stand-up test. Validate steering correction direction and rear-wheel torque direction at limited current before an unsupported run.
9. Before using the new ±90° steering command limit, verify the full steering travel, wiring clearance, and absence of interference on the supported bike with steering motors disabled. The previous software fault limit was ±30°; a software limit does not prove the mechanism is safe at ±90°.

The deployment cannot validate physical mounting, wiring, or motor direction without the bicycle. A wrong sign can make the learned controller destabilizing; do not treat software unit tests as a substitute for this checklist.
