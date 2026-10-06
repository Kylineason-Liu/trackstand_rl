# Stage D: distinguish settling, enable delay, and control-rate sensitivity

## Safety and scope

Keep the bicycle independently restrained with kickstands down, the rear wheel clear of the ground, and steering unobstructed. Use physical motor-power isolation and an observer. Do not touch rotating parts. Stop all policy and other motor-command nodes; verify one intended command publisher. Without the policy node, RC-down is **not** an emergency motor stop for manually published motor commands. No unsupported balancing is part of this test.

**Do not run the powered tests below with current production `motor_io.py`.** It commands both motors, has hardcoded steering gains/±5.6 N·m torque bounds, and lacks the tight tracking/oscillation aborts this experiment needs. A dedicated guarded steering-only diagnostic mode must be implemented and tested first. C2/C3 would require a separate guarded implementation and changed plant/gains; they are not a clean way to vary the external C0 rate.

The proposed current/angle bounds below are conservative diagnostic starting choices, **not certified safe limits or guaranteed adequate excitation**. Confirm the fixture and physical clearance first. Abort results and saturation are useful data; never bypass a guard to finish a trial.

## D0: scheduling and fresh feedback, motors not actuated

This part is runnable now. Start `imu_publisher` and `rc_kickstand` with the normal parameters to reproduce sensor load. Keep policy and kickstand driver stopped; confirm stands remain down independently of software. Verify motor outputs are physically disabled before starting; `dry_run=true` does not send a hardware disable. Use the Pi's actual installed ROS distribution (the supplied traceback used Jazzy), source `deploy/install/setup.bash`, and run from `~/trackstand_rl/deploy`.

Start a new recorder using the existing [A/B/C topic list](../../docs/BIKE_DIAGNOSTIC_TESTS.md#2-start-a-bag-for-each-test), with a separate output such as `../rosbags/d0_100hz`. Also save a passive CAN capture and effective parameters. In another terminal start:

```bash
ros2 run trackstand_deploy motor_io --ros-args \
  --params-file source/trackstand_deploy/config/params.yaml \
  -p dry_run:=true -p poll_rate_hz:=200.0 -p control_rate_hz:=100.0
```

Verify before publishing:

```bash
ros2 param get /trackstand_motor_io dry_run
ros2 param dump /trackstand_motor_io
ros2 topic info /trackstand/motor_command --verbose
```

`dry_run` must print true. Once subscriber discovery is complete, run one continuous publisher for about ten seconds and then stop it:

```bash
ros2 topic pub --rate 50 --qos-durability volatile \
  /trackstand/motor_command std_msgs/msg/Float64MultiArray '{data: [1.0, 0.0, 0.0]}'
```

Both motor feedback streams must be fresh so computed control actually runs; powered-but-disabled motors may still supply encoder replies. Confirm C0 send attempts remain zero and raw CAN contains no C0/CF from this dry-run node. Repeat with new bags and restarted `motor_io` at **200.0** and **400.0** Hz, keeping polling at 200 Hz. Do not use a mid-run `ros2 param set` to change rates: this node constructs its timers only at startup.

Compare achieved callback intervals, callback work, fresh A3/A2 update intervals, feedback age, query failures, command reception gaps and CPU load. This tests achievable scheduling/feedback, **not physical rate sufficiency**.

## Required guarded mode for powered D1–D3

The mode must preserve the calibrated encoder/torque signs and use only the steering host `0x102` / feedback `0x002`. It must leave the rear motor disabled (never send rear C0), retain normal freshness/watchdog checks, publish the existing telemetry, and have one continuously discovered target publisher. Use an explicitly documented startup reference with steering physically centered; do not rewrite persistent firmware zero with B1.

Add enforced steering torque/current bounds, a small target envelope, tight actual-angle/speed aborts, and an explicit operator stop with zero-current/CF and physical confirmation. Proposed starting limits are **0.1 N·m** (approximately **0.128 A** with the recorded 0.78 N·m/A constant), targets **±0.25°**, abort actual displacement beyond **±3°** from the marked center or speed above **2 rad/s**. The torque constant must be read on the bike, not assumed. A higher cap/target should require review of the first recording, not automatic escalation. Log cap/slew/safety activation; clipped trials cannot establish unbounded actuator bandwidth. Preserve disabled-state physical support.

## D1: already-enabled fixed-target response

At 200 Hz control / 200 Hz polling, first enable at zero and hold until the measured current, mode and angle are stable. Then one continuous publisher issues `0 → +0.25° → 0 → −0.25° → 0`, each held two seconds, without disable/re-enable between targets. Repeat five times. The guarded mode should execute this sequence; do not approximate it by repeatedly starting CLI publishers. Record small equal steps from the same initial center. Only after a reviewed stable result consider ±0.5° with an appropriate reviewed cap.

Measure first motion, first 90% crossing, overshoot, steady error, and settling within explicitly stated ±0.1° and ±0.3° bands; count successful C0 sends and fresh feedback within each first 20 ms. These test bands are not assumed policy requirements. If targets are too small to overcome deadband, record that rather than increasing gain blindly.

## D2: disable/re-enable current delivery

With the same guarded mode, record five separate enable trials starting near the same marked center. Compare first enable after startup, re-enable after an 80 ms-watchdog disable, and re-enable after an explicit disabled command. Apply the same small position offset and cap in each case, and allow motor current/angle to become inactive before the next trial. Tight guards remain armed throughout.

Capture raw CAN C0/CF, current replies, A2/A3, and motor AE mode/fault/bus voltage; query AE at about 10–20 Hz through the diagnostic harness. This tests whether the observed ~154 ms current-delivery discrepancy repeats, when the firmware enters an active mode, and whether the host requested current, native CAN bytes, and returned current agree. The loop must not interpret its own `gate_enabled` flag as firmware confirmation.

## D3: matched rate and moving-target comparison

Only after D1 is stable, repeat the same fixed-target sequence at **100, 200 and 400 Hz**, keeping gains, polling, sign mapping, current cap, fixture, starting angle and command schedule identical. Repeat each five times in interleaved order to reduce warmup effects. Record actual rates, not merely parameter values. Stop a rate immediately on an abort/oscillation; do not disable guards or increase force to complete the sweep.

Then at a stable configuration use a small smooth sinusoidal position target, still published at **50 Hz**, at **0.5, 1, 2 and 5 Hz** (review higher frequencies only afterward). Measure tracking amplitude, phase lag, RMS/peak error, saturation and freshness. Failure to settle to individual 20 ms samples is not itself failure; stable tracking of the commanded trajectory is the criterion. There is no recorded powered policy bandwidth requirement yet, so this is characterization, not a certification of balancing.

## Required handoff and interpretation

Send separate bags, passive `candump` logs, synchronized video, exact effective parameters, console logs and firmware version (`A0`). While disabled, read and preserve native responses for B4 current slope, B6/B7 position gains and B8/B9 speed gains; do not change them for this experiment. Firmware/internal-loop frequency remains unknown unless specified by the manufacturer.

Use the entire A/B/C ROS topic list plus diagnostic target/abort metadata and firmware status/voltage. Exact command schedule and response timestamps are essential. Note power cycles, motor-node restarts, manual centering, battery voltage, fixture geometry, wheel contact, and any movement by hand.

- If stable response improves consistently at a higher rate with comparable freshness and no clipping, that is evidence of rate sensitivity—not yet a universal minimum rate.
- If fixed-target instability persists at all rates, rate alone is insufficient; identify/tune gains, feedback/current delay and actuator nonlinearities.
- If firmware current stays near zero after enable, resolve that transition before assessing 20 ms tracking.
- If smooth 50 Hz target tracking is stable with finite lag, “not finishing every 20 ms” is not automatically a defect. Compare the measured dynamics with the policy's simulator before a guarded balancing test.
