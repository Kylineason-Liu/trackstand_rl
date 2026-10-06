# Real-bike Stage A/B/C diagnosis — 2026-10-02

## Conclusion

**Fix and characterize the deployed actuator-control layer before changing the policy.** Stage C reproduces severe steering oscillation with a constant target and **no policy running**. This establishes a low-level steering-control problem independently of model_41050. It does not establish that the bicycle needs replacement or that the policy is ready for unsupported balancing.

The user confirmed that Stage C's rear wheel was lifted clear of the ground, steering was free, and neither wheel/motor was touched or restrained. The rear wheel did not visibly vibrate; the front wheel visibly turned. This report uses the recorded motor signals to quantify that movement.

These recordings are diagnostic bench tests, **not a recorded unsupported balancing/fall episode**. They expose a convincing failure mechanism but cannot reconstruct every cause of the earlier deployment failure or assign meaningful numerical probabilities.

## Inputs and method

| Stage | Bag directory under `rosbags/` | Duration | Decoded messages | Observed mode |
| --- | --- | ---: | ---: | --- |
| A | `rosbag2_2026_10_01-16_11_49` | 126.68 s | 150,000 | Motors gated off; `dry_run=false` |
| B | `rosbag2_2026_10_01-17_18_11` | 436.35 s | 549,816 | Policy enabled intermittently; `dry_run=true` |
| C | `rosbag2_2026_10_01-17_32_00` | 122.01 s | 143,283 | Fixed commands; real C0 sends; no policy topics |

All MCAP messages were decoded; numeric telemetry was interpreted using the versioned field names stored in each message's layout, rather than assuming offsets. Bag/model SHA-256 hashes, topic counts, events, and statistics are in [summary.json](summary.json). Original bags, policies, training files, and deployment runtime/configuration files were not changed.

Control/inference timing uses the nodes' monotonic-clock diagnostics. IMU comparisons use header publication times. Bag-record time is not acquisition time: for example, Stage B has a 338.5 ms recorded motor-topic gap while consecutive diagnostics advance only one sequence and approximately 4.25 ms. That is recording/transport delay, not a 338.5 ms motor-control stall. Likewise, the policy stream ends at about 95 s; dividing its sample count by the entire 436 s bag duration would give a misleading frequency.

## Stage A: sensors mostly coherent; damaged recording interval

No enabled motor-control ticks, C0 control sends, or motor faults were recorded. `dry_run` was **false**, not the planned true setting. The absence of enable commands kept this captured interval inactive; this does not make an incorrectly set dry-run flag safe for subsequent policy tests.

Outside the damaged interval, quaternions have unit norm. Roll spans −35.96° to +31.48°. Quaternion-derived body-x rotation rate and gyro-x correlate at **0.978**, with measured/derived gain **0.965** using 40 ms quaternion increments. A2 speed versus a 20 ms central difference of unwrapped A3 position correlates at **0.991 steering / 0.992 rear**, with gains **0.971 / 0.980**. These observations argue against a gross roll-rate sign inversion, degree/radian conversion mistake, or A2/A3 disagreement. They do not independently certify the absolute physical mounting, encoder-to-joint ratio, or torque direction.

Pitch/yaw gyro/quaternion agreement is weaker (correlations about 0.70, gains 0.50–0.59); Stage C's vibration also worsens some IMU comparisons. Sensor fusion/filtering and asynchronous quaternion/raw-gyro packets are possible explanations, not proven diagnoses. A separate slow, marked excitation of each body axis is needed for complete validation.

**Recording integrity:** metadata claims 150,953 messages, but the MCAP physically decodes to 150,000, independently confirmed using indexed and streaming readers. At **106.31774 s**, one IMU payload contains unrelated readable ROS-log text in numeric fields, producing impossible quaternion/gyro values. Approximately 0.8 s of multiple-topic records is missing, including 159 motor-control sequence values. The malformed IMU record and missing interval were excluded from sensor comparisons. The normal publisher normalizes/rejects invalid quaternions, so this payload is not evidence of a genuine physical gyro excursion. Verify the original Pi file and this copy by checksum, then re-copy/re-record if necessary; the corruption's origin is unknown.

See [Stage A sensor plot](stage_A_sensors.png).

## Stage B: model_41050 inference and scaling work

Both the recorded policy status and an independent ONNX replay confirm **09171946/model_41050**, not model_91850. Replaying all **1,718** recorded observation vectors through the bundled 41050 ONNX yields maximum absolute action difference **4.66×10⁻⁸**. Reconstructed action scaling matches all enabled motor commands exactly.

There are nine RC-enabled intervals, totaling 1,718 inference steps, approximately 34.36 s. During enabled operation:

| Measurement | Median | 99th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Policy interval | 19.98 ms | 22.41 ms | 24.01 ms |
| ONNX inference | 0.289 ms | 0.805 ms | 1.593 ms |
| Policy IMU callback age | 2.79 ms | 12.33 ms | 14.58 ms |
| Policy motor-state callback age | 4.07 ms | 15.48 ms | 33.94 ms |

Raw steering action ranges **−0.03563 to −0.03438**; scaled target ranges **−3.205° to −3.093°**. Requested rear torque ranges **−0.00522 to +0.00357 N·m**. Neither target clipping nor torque saturation is recorded. Steering torque computed in dry run is approximately **−0.575 to −0.533 N·m**, but **no C0 sends occur**, and there are no torque-feedback samples. Computed torque must not be mistaken for measured motor torque.

All recorded policy gyro channels are zero, while roll implied by projected gravity stays about **−1.36°**. This tests essentially one stationary, supported state—not dynamic recovery. A −1.36° reading alone cannot distinguish an actual stand-supported lean from residual calibration bias. There is no policy fault-latched transition; enabled intervals end on RC-down. One brief motor-feedback-stale event occurs while already disabled.

The fast inference and successful replay make export corruption, incorrect selected checkpoint, and slow ONNX inference **unlikely explanations of the observed bench instability**. They do not validate closed-loop policy robustness.

See [Stage B policy/timing plot](stage_B_policy.png).

## Stage C: low-level steering is unstable under fixed commands

Times below are elapsed bag times at recorded motor-control ticks; enabled-run ends are approximate to one control interval. Speeds/angles are signed, converted encoder feedback. Torques are current-derived estimates, not shaft torque-sensor measurements.

| Enabled interval | Steering target | Rear target | Steering position range | Peak absolute steering speed | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 51.286–51.496 s | 0 | 0 | +0.088° | 0.052 rad/s | Essentially stationary; only 0.21 s enabled |
| 60.677–62.736 s | +0.020 rad (+1.146°) | 0 | +0.088° to +1.187° | 1.142 rad/s | Settles near +0.901°, with ≈0.245° steady error |
| 71.637–71.902 s | −0.020 rad (−1.146°) | 0 | −13.689° to +17.556° | 21.387 rad/s | Large oscillation begins before watchdog disable |
| 72.173–74.001 s | −0.020 rad (−1.146°) | 0 | −18.918° to +19.556° | 22.092 rad/s | Sustained oscillation; dominant frequency ≈14.25 Hz |
| 85.907–86.972 s | 0 | +0.050 N·m | −21.951° to +22.017° | 22.309 rad/s | Steering oscillates; this is not an isolated rear-only test |
| 102.341–103.408 s | 0 | −0.050 N·m | −22.983° to +15.820° | 21.956 rad/s | Steering oscillates again |

The first negative steering step exceeds 5° absolute position at **71.760 s**, before the **71.899 s** watchdog disable. Oscillation then persists during a continuous 1.83 s enabled interval. Therefore neither the later command gap nor policy-output variation is required to initiate/sustain this instability.

All **1,289** active steering sends and all 1,289 active rear sends report socket-write success. Exactly **1,295 current-feedback samples per motor** are recorded: 1,289 active sends plus six zero-current disable sends. B0 constants stay approximately **0.78 N·m/A steering / 0.30 N·m/A rear**. The protocol specifies C0 replies as measured Q-axis current, not target echoes.

For all 1,289 active ticks, recomputing the recorded steering and rear torque equations gives **zero discrepancy**:

```text
steering = clip(10 × steering_error − 0.2 × steering_speed, ±5.6 N·m)
rear     = clip(bounded_requested_torque − 0.05 × rear_speed, ±6.28 N·m)
```

The fixed position target is not oscillating, but measured motion feeds those equations and creates alternating torque. Across active operation, the P term reaches **4.01 N·m** and D term **4.46 N·m**. Steering torque hits its limit on **17/1,289 ticks (1.32%)**; current-derived feedback reaches **5.237 N·m absolute**. Consecutive active torque-command changes reach **8.72 N·m**, with 95th percentile **3.52 N·m**. This is a measured explanation for the perceived discontinuous torque, rather than speculation about raw policy jitter.

No recorded `motor_fault` or motor latch becomes true. The motion stays within the configured ±90° position fault bound and 26.29 rad/s steering speed fault bound. Those checks do **not** detect oscillation or persistent tracking error. No firmware diagnostic-status frames, supply-voltage trace, or calibrated mechanical torque measurements were recorded, so zero software faults does not certify healthy firmware/power/mechanics.

### Timing and actuator mismatch

| Stage C active measurement | Median | 99th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Control interval | 5.01 ms | 7.79 ms | 12.00 ms |
| Control callback work | 1.73 ms | 5.30 ms | 8.66 ms |
| A3 steering position receipt age | 5.64 ms | 18.44 ms | 26.23 ms |
| A2 steering speed receipt age | 3.76 ms | 15.78 ms | 24.83 ms |

Across the C bag, query-failure counters increase by **355 steering A3 / 143 steering A2 / 43 rear A3 / 544 rear A2**. A complete A2/A3 pair is required before either stored measurement updates. When one query fails, both old values persist. Position-derived versus A2 steering speed in active motion still correlates at 0.925 with gain 1.006, arguing against a gross speed-unit error, but this does not remove the age/jitter problem.

Matching current replies to C0 sends in per-motor order gives median tick-start-to-feedback-receipt **3.62 ms steering / 4.64 ms rear**. Steering feedback correlates better with the preceding command (0.938) than the same-order command (0.879). This supports delayed/filtered current response; it is **not** an identified firmware time constant, because reply timestamps are host receipt times and replies do not contain sequence IDs.

The saved 41050 training configuration uses **100 Hz physics / 50 Hz policy**, steering `ImplicitActuator` with stiffness 10 and damping 0.2, and rear damping 0.05. Isaac Lab's local actuator implementation states that implicit PD is integrated inside the physics engine and directly returns position targets. Deployment instead implements explicit sampled PD using delayed CAN feedback and C0 current commands. **Matching numerical gains does not make these two closed-loop actuator responses equivalent.** These recordings strongly implicate that implementation/plant interaction, but do not uniquely distinguish gain tuning, firmware current-loop behavior, drivetrain nonlinearities, power issues, or scheduling delay.

Six disable events use `command_timeout`; most are expected after a finite publisher stops. The negative-steering sequence also has a recorded **352 ms internal command gap**, causing an unintended disable/re-enable. Improve test-command delivery, but do not treat this as the sole explanation: oscillation occurs before that disable and during uninterrupted subsequent operation. Stage C also has IMU publication gaps of approximately **336 ms and 218 ms**, before the respective motor-enable intervals. Their cause is unknown; during an actual policy run they would be incompatible with the 60 ms IMU freshness requirement if observed by the controller. Stage B's active IMU ages do not show such an outage.

See [steering overview](stage_C_steering_overview.png) and [steering close-up](stage_C_steering_zoom.png). The close-up shows a constant target, alternating actual-current-derived torque, oscillating encoder position/speed, and an enabled gate with no fault latch.

### Rear wheel: tiny motion and a sign warning

| Rear requested torque | Mean current-derived torque | Encoder displacement during enabled interval |
| --- | ---: | ---: |
| +0.050 N·m | +0.05067 N·m | **−0.009587 rad (−0.549°)** |
| −0.050 N·m | −0.04737 N·m | **+0.008053 rad (+0.461°)** |

Current is delivered, so “no rear commands reached the motor” is not supported. But wheel movement is tiny, and its sign is opposite the commanded policy-coordinate torque in **both** tests. Native encoded command currents average approximately **−0.169 A** for the positive target and **+0.167 A** for the negative target, consistent with `drive_torque_sign=-1` in the local configuration.

This contradicts the *expected published-coordinate response* and deserves a fresh sign/calibration test. The bags do not record raw CAN A2/A3 bytes, a physical front/back motion label, or complete effective Pi parameter dumps. Moreover steering is oscillating during both rear tests. Do **not** blindly flip an encoder/torque sign from these small, coupled tests. Repeat an isolated, supervised rear test with known configuration and video/raw CAN, using a carefully controlled torque sufficient to distinguish breakaway from encoder noise. Check firmware direction, cogging/stiction/deadband, and current limiting/ramping. The present data cannot prove which is responsible or establish a breakaway torque.

See [rear torque/position/speed plot](stage_C_drive.png).

## Likelihood ranking and improvement order

The following are qualitative evidence rankings, not measured probabilities.

| Candidate | Evidence-based assessment | Priority |
| --- | --- | --- |
| Unstable deployed steering-control/actuator response | **Established in C**, independently of the policy; sampled-PD gain/delay/nonlinearity interaction is the leading mechanism hypothesis | First |
| CAN feedback age, query failures, and host-control timing | **Measured; likely contributor** to steering-loop instability, not a unique isolated cause | First, with steering tuning |
| Rear torque/encoder direction and low-torque behavior | **Sign warning and poor movement established**; exact direction mapping/breakaway mechanism and contribution to the earlier fall remain unresolved | Resolve before balancing |
| Policy sim-to-real robustness | **Plausible secondary issue**, not tested by these supported/dry-run bags; 41050 lacks actuator/physical DR in its saved event configuration | After actuator calibration |
| IMU calibration/data freshness | Roll-rate coherence is encouraging; complete-axis validation, publication gaps, and bag integrity still need checking | Before unsupported testing |
| Wrong export or slow policy inference | **Low likelihood** for these tests: model identification, replay/scaling, and active 50 Hz timing succeed | Not the first fix |
| Motor damage, power fault, or mechanical defect | **Not established** by these data; software control instability and delivered current do not prove component failure, and firmware/power diagnostics are missing | Investigate with discriminating tests |

Recommended next work:

1. **Keep the bike independently supported; do not repeat unsupported balancing yet.** Establish a stable steering bench loop with continuous targets and conservative, supervised current/torque limits. Tune against measured response rather than copying the implicit simulation gains. Do not increase damping blindly: a delayed/noisy speed signal can make a larger D gain worse. Improving feedback timing and/or using a properly tuned firmware position loop (C2/C3) are candidate solutions, not verified fixes. Preserve the required 50 Hz policy interface.
2. Isolate each motor for characterization. Current Stage C enables both; a new explicit per-motor bench mode or supervised hardware procedure is needed for clean tests. Verify physical direction versus native CAN and signed ROS coordinates, then identify position step response, small-torque breakaway, current response, delay, and repeatability. Do not tune a balancing policy around an unresolved sign discrepancy.
3. Capture effective parameter dumps, raw passive CAN, motor diagnostic status/supply voltage, synchronized video, and all existing telemetry. Use one continuous test-command publisher after DDS discovery, rather than repeatedly starting finite CLI publishers. Retain the watchdog and verify gaps are recording-only versus actual delivery outages.
4. Confirm IMU roll zero using a physical upright reference and repeat slow marked body-axis rotations. Track quaternion/raw-gyro freshness separately if packet-age testing shows a problem. Re-copy/checksum Stage A and verify sensor/RC delivery under the normal deployed node load.
5. **Then improve training if needed.** Keep the verified geometry. Match the identified actuator dynamics, delays/jitter, friction/deadband, limits, and encoder-speed noise; randomize around measured values. 41050's saved observation noise has zero noise on encoder channels, while Stage B's stationary steering-speed observation has standard deviation ≈0.0212 rad/s. Also match available-sensor observation semantics: training reads ground-truth body linear velocity; deployment uses rear-wheel speed × radius for body-x and zero body-y. These bags cannot validate that approximation under real slipping/balancing motion. Switching to a DR policy alone will not repair the demonstrated bench instability.

## Source evidence

- [Motor control and freshness/fault handling](../../source/trackstand_deploy/trackstand_deploy/motor_io.py): `_poll_motor`, `_control_tick`, `_send_torque`, `_disable_outputs`.
- [Current/A2/A3 units and conversions](../../source/trackstand_deploy/trackstand_deploy/can_protocol.py) and [local hardware configuration](../../source/trackstand_deploy/config/params.yaml).
- [Policy readiness and observation construction](../../source/trackstand_deploy/trackstand_deploy/policy_controller.py), [observation/action contracts](../../source/trackstand_deploy/trackstand_deploy/contracts.py), and [IMU publisher](../../source/trackstand_deploy/trackstand_deploy/imu_publisher.py).
- [Saved 41050 environment](../../../isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-09-17_19-46-37/params/env.yaml): physics step line 20; decimation line 98; actuator settings lines 231–255; noise lines 310–339; actions/events from line 340. Its saved events have reset and fixed ground-material setup, not randomized actuator parameters. [Saved runner settings](../../../isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-09-17_19-46-37/params/agent.yaml) leave action clipping null.
- Local Isaac Lab implementation read from `/home/lmbike/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py`, `ImplicitActuator` (lines 37–53 and 119–143): internal physics PD, approximate effort reporting, direct return of position targets.
- Motor protocol PDF read from `tsinghua_bike_project/assign_2_Hardwares/assign_2_Hardwares/3.RearSteer Servo【GIM3510-8】/Supplementary Materials/2.自定义CAN通信协议/自定义CAN通信协议_3.09b0.pdf`: A1/A2/A3/B0/C0 layouts, B4 current ramp, and AE diagnostic status. The PDF specifies C0 response contents as A1 measured current.

## Reproduction

Use a Python environment with ROS standard-message serializers, `mcap`, `numpy`, `PyYAML`, `onnxruntime`, and `matplotlib`. ROS Humble serializers decoded the standard Jazzy bag messages; no rosbag2 MCAP storage plugin was needed. From the repository root:

```bash
source /opt/ros/humble/setup.bash  # use your installed ROS distribution
TASK_BAG_CACHE=$(mktemp -d /tmp/trackstand_bag_cache.XXXXXX)
python3 deploy/reports/rosbag_ABC_2026-10-02/decode_bags.py --cache-dir "$TASK_BAG_CACHE"
python3 deploy/reports/rosbag_ABC_2026-10-02/analyze_bags.py --cache-dir "$TASK_BAG_CACHE"
```

The decoder writes only diagnostic cache files; the analyzer writes `summary.json` and PNG plots in this report directory. ONNX replay uses the named bundled 41050 file and its external-data sidecar. It does not run a simulator or send hardware commands.
