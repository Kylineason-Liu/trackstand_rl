# Hardware Commissioning Checklist

Keep the kickstands down and set `motor_io.dry_run: true` for the first pass.

1. Confirm `/trackstand/imu` reads approximately `gravity_y=0`, `gravity_z=-1` when upright. A right lean must have the same `gravity_y` sign as the Isaac/MuJoCo observation.
2. Turn the handlebar by hand. `/trackstand/motor_state[0]` must be steering radians and index 2 its derivative in rad/s. Steering is motor ID 2 and uses positive encoder and command signs, matching `ros2_ws`.
3. Rotate the rear wheel by hand. State indices 1 and 3 must have matching signs. Positive wheel velocity is defined as positive estimated body-x velocity. Confirm all four encoder/torque sign parameters together.
4. Verify the startup log reports plausible, positive B0 torque constants for both motors. Missing or invalid values must leave output inhibited.
5. Verify the CP2102 receiver produces CRSF at 420000 baud. CH7 high must publish `true`, low must publish `false`, and middle must hold the last state after the eight-frame debounce.
6. Verify servo positions: up `(ID1=1648, ID2=2454)`, down `(ID1=1972, ID2=2085)`.
7. With `dry_run: false` and stands down, command no lift. Both motors must remain free/disabled.
8. Secure the frame in a rig before the first stand-up test. Validate steering correction direction and rear-wheel torque direction at limited current before an unsupported run.

The deployment cannot validate physical mounting, wiring, or motor direction without the bicycle. A wrong sign can make the learned controller destabilizing; do not treat software unit tests as a substitute for this checklist.
