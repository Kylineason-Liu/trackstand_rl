# Real Trackstand deployment diagnosis — 2026-10-12

## Conclusion

The observed rear-wheel runaway and near-90° steering are commanded behavior, not evidence that MIT randomly selected those positions. The strongest suspected sustained-runaway cause is **inconsistent rear torque/encoder/forward-motion signs**. Independently, **model_91850 produces an excessive startup action even for an ideal upright, motionless observation**, and its training actuator path differs from the explicitly bounded hardware path. Both need attention before another unsupported trial. Missing containment and intermittent feedback/host failures compound the problem.

This does not establish defective bicycle geometry or a need to replace the motors/bike. The XML/USD geometry is not changed or challenged. The user confirms both wheels grounded, no holding/lifting/restraint during all four RC-up attempts, straight steering at every motor-node start/restart, unchanged settings, and an up-to-date Pi repository. The user's visual observation is forward wheel spin/travel; the bag has no independent translational-velocity/video measurement. No numerical probabilities can responsibly be assigned to the suspected causes.

## Recording, provenance and method

Input: `deploy/experiments/logs/rosbag2_2026_10_12-12_06_32/rosbag2_2026_10_12-12_06_32_0.mcap`.

- 85,540,864 bytes; SHA-256 `262c9510421a9efd477b3171ec018c71cc17ca08ba894edf6a3eb2df1fc3c06f`.
- No `metadata.yaml`, completed footer, or terminal MCAP magic. Streaming CRC-checked decoding recovers **227,892 messages**, spanning **155.296 s**, before EOF in the unfinished tail. No decoded-message errors occurred. Final completeness/expected counts and the truncation's origin cannot be established. The input was not repaired or changed.
- 249 observations, 249 raw actions, 249 enabled ROS commands and 249 successful-inference timing records; 839 successful rear C0-send ticks. Numeric layouts are interpreted from their recorded field-name labels, not guessed offsets.
- Status names **09221945/model_91850**, `dry_run=false`, MIT steering/C0 rear, nominal 50/100/200 Hz. Encoded MIT gains are **10.0122100122 / 0.2002442002**; B0 constants approximately **0.78 / 0.30 N·m/A**. No parameter-event payload was recorded, so repository state/user confirmation is not a full effective-parameter dump.
- Replaying all observations through the bundled 91850 ONNX gives maximum raw-action error **2.384×10⁻⁷**. Independently reconstructing the original checkpoint's ELU actor gives **1.526×10⁻⁵** maximum error, consistent with float32 engine differences. Action scaling/clamping and the rear damping/final-clipping equation reconstruct with **zero error**. This strongly argues against wrong checkpoint selection, corrupted export, missing normalization or an accidental extra action scale.
- Times locating episodes/events are elapsed record times. Response intervals and scheduling statistics use node monotonic diagnostics; these are host timestamps, not firmware acquisition times. Physical kickstand-release time is unknown: no actual servo-position/status messages were recorded.
- Torque feedback is signed **measured Q-axis current × B0 Kt**, not an independent shaft-torque sensor. Finite outgoing torque and successful socket writes do not prove that torque was physically delivered.

Artifacts: [HTML plot index](index.html), [full statistics/events/hashes](summary.json), [249 policy samples](policy_samples.csv), [839 active rear commands](active_motor_commands.csv).

## What happened

Initial demands below are the raw action multiplied by `(1.57 rad, 6.28 N·m)`, **before** deployment limits. “Peak rear speed” is signed-policy-coordinate encoder speed magnitude, not measured bicycle translation.

| Attempt | First inference, bag s | Inferences | Initial steer / rear demand | Peak rear speed | Actual steering range while powered | End of control |
|---|---:|---:|---|---:|---|---|
| 1 | 7.489 | 76 | +220.78° / +142.37 N·m | 61.72 rad/s | +2.20°…+90.18° | Rear feedback becomes stale; motor fault, then policy fault |
| 2 | 48.008 | 109 | +153.38° / +107.91 N·m | 62.54 rad/s | −35.51°…+95.38° | RC down |
| 3 | 79.590 | 2 | +368.74° / +222.08 N·m | No powered rear sends | Initial reported steering ≈178.95° | MIT encoding rejected; motor fault |
| 4 | 117.810 | 62 | +154.09° / +108.83 N·m | 62.77 rad/s | 0°…+89.87° | Inversion detection, then motor disable |

Across all inference samples, **188/249 (75.5%)** steering requests exceed ±90° and **200/249 (80.3%)** rear requests exceed ±6.28 N·m. Both are vastly more frequent than the archived successful Isaac evaluation's **0.659% / 0.0891%**; the datasets have different trajectories and durations, so these are descriptive comparisons, not matched failure probabilities.

All three powered attempts reach rear speed magnitude >50 rad/s **54.74 / 61.74 / 56.64 ms** after the first successful rear send. Steering reaches >85° after **188.19 / 102.11 / 180.29 ms**. Rear final commands are within 0.001 N·m of the ±6.28 bound on **90.4% / 70.4% / 100%** of their powered ticks. Attempt 4 sends approximately **+6.28 N·m throughout**, while rear speed is predominantly **−60 rad/s**.

Near-90° steering alone is not proof of a faulty policy or motor: it is inside the trained steering range and can be part of a trackstand strategy. The recorded failure is sustained wheel/net-motion runaway and loss of balance, not merely a large steering angle.

See [four-attempt plots](attempts.png), [unclipped requests](policy_requests.png), and [first-350-ms detail](startup.png).

## Suspected reasons, ranked

### 1. Rear command/feedback direction mismatch — very high likelihood for sustained runaway

**Measured:** positive signed rear torque commands, encoded native C0 current near **−20.933 A**, produce strongly **negative** A2/A3 motion. Positive-command/negative-speed combinations occupy **91.6% / 78.9% / 98.1%** of powered ticks. Attempt 4's encoder net rotation is about **−70.62 rad**. Current-derived torque is generally positive, including **+6.274 N·m** during initial acceleration; the sign of this estimate is the software conversion, not an independent mechanical direction measurement.

**Physical inconsistency:** the user reports forward spin/travel, but deployment feeds the policy negative rear speed and negative estimated body-x speed. Current configuration expects positive feedback for forward rotation. Read-only kinematic evaluation of the existing MuJoCo model confirms that positive rear joint rotation rolls toward model-root +x/front: world rear axis approximately `[-0.013263, +0.999911, -0.001485]`; rear-to-front direction `[+0.998736, -0.050249, +0.001474]`. This identifies the model's sign without questioning its geometry.

The rear control term `−0.05 × speed` therefore **adds** approximately **+3 N·m** at the recorded −60 rad/s, rather than subtracting torque from visually forward motion. The policy also sees `estimated_vx = 0.07 × speed ≈ −4.2 m/s`. Under a wrong torque-to-feedback pairing, the nominal damping becomes reinforcing rather than dissipative in the physical coordinate.

**Offline sensitivity evidence:** reversing rear position, rear speed and their derived vx in the recorded observation vectors reduces raw rear actions beyond ±1 from **200/249 to 40/249**. For attempt 4 the median rear action changes **+4.479 → −0.138**; the startup burst remains. This is a fixed-record counterfactual, **not proof that a corrected closed loop balances**: all other recorded states are unchanged.

**Verify:** with policy stopped and motors confirmed disabled, support the bicycle and physically label forward/backward rolling. Capture raw A2/A3 CAN bytes and published state. Then, with rear wheel raised and frame independently secured, test small positive/negative torque pulses within verified safe limits; capture raw C0/A1/B0 and acceleration. Use one CAN writer only; a passive `candump` listener is appropriate, competing query/control scripts are not. Confirm a positive policy torque produces positive policy-coordinate acceleration, and zero requested torque damps both rotation directions.

**Potential correction:** derive encoder and current signs independently from these tests. If forward movement gives negative native encoder feedback while negative native C0 current drives forward, the candidate is `drive_encoder_sign=-1` with `drive_torque_sign=-1` retained. This is **conditional**, not an instruction to flip both or blindly edit a sign. A2/A3 position, velocity, wrapped angle and vx must remain mutually consistent. Earlier manual sign reports conflict with this new behavior; resolve that conflict empirically.

### 2. Excessive learned startup demand and actuator saturation mismatch — confirmed exposure; high-likelihood contributor

An ideal upright, zero-rate, zero-wheel-motion, straight-steering vector yields original 91850 raw actions **`[1.599545, 16.720318]`**, i.e. approximately **+143.886° / +105.004 N·m** before limits. Thus residual ~1° IMU tilt is **not required** for the burst. Removing gyro inputs or flipping signs cannot eliminate this initial demand when the wheel is stationary.

The archived Isaac motor-limit dataset independently shows **all 171 episode starts** requesting **63.89–132.00 N·m**, versus a non-startup maximum **2.765 N·m**. After 20 ms, rear velocity is positive in **167/171** starts (median **+60.31 rad/s**). Hardware instead takes several policy steps to develop motion and reports the opposite sign. The training PhysX feed-forward submission buffer retained the excessive requests; the plotted ±6.28 “applied torque” was an implicit-actuator estimate, not proof of a capped total solver torque. Actual reliance on the large initial impulse remains to be established by a strictly bounded simulation rerun.

**Verify:** first perform a supported, genuine `dry_run=true` 91850 inference test and confirm the large initial request without motion. Then compare headless evaluations using identical initial conditions/seeds with physically enforced rear feed-forward/total torque limits and the deployment steering clamp/slew/damping. Inspect actual submitted/applied forces, first 200 ms and completion metrics—not only the diagnostic torque estimate. Models stay unchanged.

**Potential correction:** train/evaluate with the actual bounded action/actuator path, measured delay, torque-speed/current limitations and supported-to-free release conditions. A deployed torque-rate limit or ramp can reduce risk but changes the policy's plant; validate it in simulation and retrain if necessary. Blindly scaling down a saturated policy is not an established balancing solution. 41050's near-zero stationary action differs, but its earlier failed C0 deployment means swapping policies is not a proven fix.

### 3. Body-velocity observation mismatch/slip — medium likelihood; sign error is one part

Isaac/MuJoCo supply actual body-frame linear x/y velocities. Deployment substitutes wheel speed ×0.07 for x and **always zero** for y. In this bag the substituted x reaches **−4.39 m/s**, but there is no odometry proving the bicycle translated at that speed. At near-90° steering, turning, lateral motion, wheel slip and changing attitude can invalidate the simple no-slip longitudinal proxy. Both wheels being grounded does not prove pure rolling. Gyro/encoder feedback is real; this proxy is not measured root velocity.

**Verify:** use temporary overhead/side video with distance markers or external tracking to measure bike translation and compare it with the wheel proxy, starting under restraint at low speed. Replay both observation constructions and compare policy sensitivity; then evaluate the same estimator in both simulators.

**Potential correction:** train with the estimator/observations actually available on the bike, including slip/delay, rather than exposing privileged true body velocity and replacing it at deployment. An IMU/encoder estimator may improve consistency; extra sensors are not assumed available or authorized by this diagnosis.

### 4. Ground-loaded steering response differs from training — plausible, not the main explanation for 90°

In this recording the wheel travels close to the **requested, clipped** +90° target, so unexplained MIT steering drift is not supported. Nevertheless, it cannot follow instantaneous large reversals in 20 ms, and attempt 2 overshoots to **95.38°**. Prior unloaded MIT sine reports establish attenuation/lag at 5/10 Hz; they do not establish ground-loaded performance. Equal Kp/Kd does not equate motor firmware PD with implicit simulator PD. Recorded MIT gains agree with the intended 10/0.2 encoding; firmware-loop frequency and total PD current limit remain unknown.

**Verify:** supported, independently secured tests with ground loading, small-amplitude held targets at the actual 50 Hz policy cadence, MIT resends at 200 Hz, both motion directions, A3/A2/A1/F1, supply voltage/current and consistent origin. Use realistic policy target sequences, not only fresh 200 Hz sine targets.

**Potential correction:** model the measured response and verified limits in training; retune gains only with matched supported tests. Do not treat higher command frequency alone as a cure or replace the motor based on this bag.

### 5. Feedback/host scheduling failures — confirmed secondary failures, lower likelihood for initiating runaway

Inference is fast: interval median/p99/max **19.99/21.76/22.44 ms**, ONNX inference **0.192/0.561/1.558 ms**. Motor MIT send intervals median/p99/max are **5.42/9.42/17.55 ms**, mean **5.83 ms** (~172 Hz for those intervals), not a guaranteed 200 Hz.

Attempt 1 stops at **8.990 s** on `active_feedback_unavailable`, with rear pair age **51.88 ms**. This interrupts balancing but occurs long after the first speed/steering excursion. At **16.014 / 123.522 s**, worker errors report `system clock sampling delayed; CAN feedback age is uncertain`; both occur after the preceding powered attempt ended. They require repair/validation, but cannot explain its initial motion. This is clock-sampling uncertainty, **not evidence of a clock jump**.

RC publishes ~**494 Hz**, despite a nominal 100 Hz read timer: the parser publishes each received RC frame. While down, policy callbacks emit an off command for each frame, producing **77,091 disabled motor commands**. This is a measured unnecessary-traffic candidate, not a measured CPU-overload cause. Active policy commands still number exactly 249 at 50 Hz. Source traces: `rc_kickstand._read`, `policy_controller._rc_callback`.

**Verify/correct:** inspect Pi CPU, thermal throttling, scheduling, CAN RTT/loss and bus load while recording. Reduce duplicate RC/down messages while retaining prompt transitions and freshness heartbeats; measure query timeout adequacy; improve timestamp-uncertainty handling without accepting stale feedback or weakening disable protection. Re-test supported enable/disable cycles and watchdogs.

## Confirmed safety/arming gaps, separate from root-cause ranking

- Deployment checks inversion, not the training 30° fallen criterion. Attempt 4 crosses `abs(gravity_y)>0.5` at **118.522 s**, inversion at **119.040 s**, and policy cutoff at **119.047 s**: approximately **0.52 s** of continued inference after the first training-fall-threshold crossing. The gravity criterion need not stay true throughout that interval. Attempt 2 crosses the same threshold at **48.380 s** but keeps running until RC-down near **50.18 s**. These are projected-gravity thresholds, not a claim that pure Euler roll alone explains the fall.
- All three powered attempts exceed **7.2 rad net rear rotation after ~162–166 ms**, versus the training `wheel_net` episode termination. Deployment has no equivalent runaway containment. Encoder net rotation is not travelled distance when slipping.
- Attempt 3 starts with reported steering **3.123 rad (~179°)**. The host slew proposes native **−3.004114 rad**, outside configured ±3, and rejects it before either motion send. Nevertheless, policy already issued kickstands-up. This exposes a readiness/arming gap; it does not establish encoder-origin corruption. Straightness was confirmed at motor-node restarts, not each subsequent RC-up.
- Motor disable is issued/confirmed in protocol telemetry at the relevant stops; policy faults do not automatically lower the stands. Acknowledgement flags do not independently establish physical zero torque/support.

Restore an explicit, validated fall/runaway envelope and require a centered valid steering reference plus full motor readiness before releasing support. Establish a physical emergency stop and firmware protection. Safety guards reduce consequences; they do not repair the learned controller or sign contract. **Do not repeat unsupported balancing until the sign and startup issues are resolved.**

## Recommended order of work

1. Confirm real forward/encoder/current directions with supported, policy-off tests; archive raw CAN, video direction labels and `ros2 param dump` for every node.
2. Supported 91850 dry-run: verify initial observations/actions and correct policy/sign/reference selection; inspect finite sensors, centering, motor readiness and disable acknowledgements.
3. Matched headless simulation with hardware limits/estimator/release behavior; determine whether 91850 still works or requires retraining. Keep verified geometry unchanged.
4. Supported low-amplitude loaded actuator and watchdog/arming tests; resolve host/timestamp/RC load problems.
5. Only after those pass, restrained/tethered closed-loop trials with strict limits, quick physical isolation, synchronized video and the full topic bag. No unrestricted trial is justified by a passing export/unit test.

Capture `/trackstand/imu`, `motor_state`, `motor_control`, `motor_status`, `motor_fault`, `motor_event`, `observation`, `raw_action`, `motor_command`, `policy_timing`, `policy_event`, `status`, `kickstands_up`, `kickstands_command`, `/rosout`, `/parameter_events` plus available servo status, passive raw CAN and video. Save exact launch commands, selected policy/hash, effective parameters and firmware settings. Current bag has no load-voltage/current/protection-register trace; power sag/current ramp/stiction cannot be uniquely diagnosed or ruled out.

## Reproduction and sources

Uses the local MCAP 1.5.0 parser, ROS standard message serializers (Humble can decode these recorded Jazzy standard types), NumPy, PyTorch, ONNX Runtime, MuJoCo and Matplotlib. Set the ROS distro/interpreter to your installed compatible environment; install MCAP in that environment if absent. This analysis never initializes ROS or opens CAN. Cache can remain outside the repository:

```bash
source /opt/ros/humble/setup.bash
repo=/home/lmbike/trackstand_rl
analysis_cache=$(mktemp -d /tmp/trackstand_deployment_cache.XXXXXX)
python3 "$repo/deploy/experiments/reports/deployment_20261012_120632/decode_mcap.py" \
  --bag "$repo/deploy/experiments/logs/rosbag2_2026_10_12-12_06_32/rosbag2_2026_10_12-12_06_32_0.mcap" \
  --output "$analysis_cache"
python3 "$repo/deploy/experiments/reports/deployment_20261012_120632/analyze_deployment.py" \
  --cache "$analysis_cache"
```

Implementation evidence: [observation/action contract](../../../source/trackstand_deploy/trackstand_deploy/contracts.py), [policy/readiness](../../../source/trackstand_deploy/trackstand_deploy/policy_controller.py), [MIT/C0 backend](../../../source/trackstand_deploy/trackstand_deploy/motor_mit_io.py), [CAN units/sign conversions](../../../source/trackstand_deploy/trackstand_deploy/can_protocol.py), [telemetry semantics](../../../docs/TELEMETRY.md), [Isaac observation construction](../../../../isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/observations.py), [archived Isaac motor-limit evidence](../../../../isaaclab_train/reports/motor_limits_09221945_91850_2026-10-01/README.md), [previous 200 Hz MIT experiments](../mit_rate_comparison_20261008/README.md). Simulator root/joint axes were checked with an in-memory model and `mj_forward`; no dynamic simulation or model-file changes were made.

Only new offline analysis/report files are added. Original recording, policies/checkpoints, deployment runtime/configuration and training/simulation sources remain unchanged. No hardware test, new training or dynamic simulator evaluation was performed by this diagnosis.

Validation: independently reran the packaged streaming decoder and obtained byte-for-byte-equivalent decoded arrays (including NaNs) and identical details/counts. Verified original bag, training dataset and eight runtime-source hashes; both analysis scripts parse; all **249/839** CSV rows have the correct columns; all **four PNG / four SVG** plots are structurally valid and the startup plot was visually inspected; all **27** local report/HTML links resolve. Replay/sample alignment and exact action/rear-equation reconstruction assertions pass. `git diff --check` is clean. These validate the offline analysis, not bike safety or successful balancing.
