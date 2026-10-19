# Corrected-sign deployment diagnosis — 2026-10-12 18:06

## Main conclusion

The rear direction correction is consistent with this run: the wheel now moves in both signed directions, and positive commanded torque initially produces positive reported motion. There is no evidence here to justify another sign flip. **The remaining controller requests are incompatible with the deployed actuator trajectory**, and observation/timing differences remain. The motor largely follows the **slew-limited MIT target**, not the rapidly changing policy target. This is stronger evidence for a policy–deployment contract mismatch than for a defective steering motor or incorrect ONNX export.

The user confirms both wheels grounded, no manual disturbance, straight steering initially, unchanged configuration except the rear encoder sign, and that a kickstand caught the bike **less than one second after lifting**. A sustained lean therefore is **not successful balancing**. Exact support-loss/contact times and actual bike translation are unrecorded. The first-second statistics below are an **early interval**, not a certified unsupported interval; later stable-lean data are treated as support-contaminated. One-second plot markers are relative to first inference, not measured servo release/contact times. No numerical causal probabilities are assigned.

[Open graphs](index.html) · [Full statistics](summary.json) · [Policy samples](policy_samples.csv) · [Actual motor submissions](active_motor_commands.csv)

## Recording and verification

Input: `deploy/experiments/logs/rosbag2_2026_10_12-18_06_07/rosbag2_2026_10_12-18_06_07_0.mcap`.

- **39,914,531 bytes**, SHA-256 `24be09c76b25d338c58ca6ae3d95bd468107a4afb4fa5b38139a1d8e568c3e78`.
- CRC-checked streaming decoded **112,856 messages / 82.416237545 seconds**, with no decode/stream errors. Every topic count, start timestamp and duration matches `metadata.yaml`. This establishes readable recorded contents, not that every sensor acquisition or ROS publication was captured.
- **894 observations/actions/enabled commands/successful inferences**, **3,022 successful MIT and rear C0 submission ticks**. Status selects **09221945/model_91850**, `dry_run=false`, nominal 50/100/200 Hz.
- All observation sensor channels exactly reconstruct from matching recorded IMU/motor samples at float32 precision: maximum error **0**. ONNX replay error **3.8147e-6**; independently reconstructed original checkpoint error **1.5259e-5**. Scaling and the rear clipping/damping equation both reconstruct with maximum error **0**. Different replay engines can differ slightly at float32 precision.
- Matching sensor values uses a ±90 ms cross-topic recording window, allowing recorder ordering. Exact matching does **not** prove acquisition synchronization/freshness; policy and CAN ages are analyzed separately.
- Torque means **signed measured Q-axis current × B0**, not commanded torque or calibrated shaft torque. B0 remains approximately **0.78 N·m/A steering / 0.30 rear**. The bag contains neither raw CAN frames nor an effective parameter dump, servo-position trace, supply-voltage/current-limit trace or external translation measurement.

| Attempt | First inference / last inference, bag s | Samples | Initial unclipped steer / rear request | First training `abs(gravity_y)>0.5`, after inference starts | End of control |
|---|---|---:|---|---:|---|
| 1 | 6.162 / 7.203 | 53 | +154.22° / +108.57 N·m | 0.426 s | Steering feedback stale; policy state timeout and motor latch |
| 2 | 13.841 / 23.180 | 468 | +138.16° / +101.41 N·m | 0.608 s | RC down |
| 3 | 32.641 / 40.080 | 373 | +123.71° / +95.12 N·m | 0.377 s | CAN worker timestamp-uncertainty failure |

The recorded Euler roll first exceeds 20° magnitude after **0.356/0.448/0.347 s**. These are measured attitude thresholds, not a measurement of when the kickstand touched. [Early traces](startup.png) show the large initial control activity and rapid lean before the later software stops.

## 1. Observation units and directions

Channel numbers are one-based. Unit correctness is separate from whether a proxy actually measures the quantity supplied during training.

For a pure roll angle φ, projected gravity is `g_y=-sin(φ), g_z=-cos(φ)`; opposite signs of roll and gravity-y are therefore expected, not evidence of an IMU inversion.

| Channels | Meaning / expected unit | Evidence and verdict |
|---|---|---|
| 1–2 | Projected gravity y/z, dimensionless | Computed as inverse quaternion rotation of `[0,0,-1]`; quaternion norm is 1 and observed values reproduce exactly. Near upright, z≈−1; y has the mathematical sign expected from the quaternion. No g-versus-m/s² or quaternion-order error appears. Physical bike-axis alignment is not independently proved by this unlabelled motion. |
| 3–5 | Bike-frame angular rates, rad/s | Vendor raw conversion explicitly includes π/180; publisher does not convert it twice. Published gyro channels equal policy channels exactly. Quaternion/gyro motion has consistent overall signs, but dynamic coherence is imperfect; see below. |
| 6 | Longitudinal speed, m/s | Numerically `0.07*rear_velocity`, maximum residual **3.24e-7 m/s**, range **−4.229…+4.266 m/s**. This is a wheel-based proxy, not the simulator's measured body-root velocity. Slip, yaw/turning, root-to-axle motion and stand contact can break equivalence. |
| 7 | Lateral speed, m/s | Always zero, by design. It is **not a measurement** of real body lateral velocity, unlike Isaac/MuJoCo. |
| 8, 10 | Steering position/speed, rad / rad/s | Configured positive direction is right; native encoder/absolute MIT position increases left, so sign −1 is applied consistently. Sign-corrected F1 position versus A3 correlation **0.998/0.996/0.994**, slope **1.002/0.992/0.990** across attempts. Same-interval A3 motion/A2 speed correlation **0.989**, slope **0.950**: no gross angle/speed unit or direction mismatch. |
| 9, 11 | Rear position/speed, rad / rad/s | Native forward A2/A3 was independently shown negative in the previous manual CAN tests; corrected encoder sign −1 makes forward positive. Initial positive C0 policy torque now produces positive feedback. Same-interval A3/A2 correlation **0.956**, slope **0.846**, supports consistent scale/direction but not perfect dynamic accuracy. Position wrapping/reference parity needs attention below. |
| 12–13 | sin/cos steering, dimensionless | Maximum reconstruction errors **5.10e-8 / 7.86e-8**. Correctly derived from steering radians. |

Encoder consistency compares **angle change and interval-averaged A2 velocity over the same ~40 ms window**, excluding gaps. Comparing an averaged derivative with an instantaneous velocity during these rapid oscillations would create misleading gain errors. Acquisition skew/filtering and sampling remain; this is not independent gear-ratio metrology.

### IMU dynamic coherence is not fully established

Stationary pre-start gyro is exactly zero; pre-start Euler roll medians are **−0.998/−1.564/−0.851°**. These may include actual stand-supported lean or calibration residual; without an external upright reference, do not automatically subtract them as sensor bias. The policy also has an excessive ideal-zero startup action, so this offset is not required for that burst.

Quaternion-increment versus gyro x/y/z correlations at the same reported time are **0.502/0.596/0.721**. Shifting the gyro sample time by **+18/+12/+1 ms** raises correlations to **0.787/0.701/0.721**. These are whole-record descriptive alignments over ~40 ms rotations, not identified firmware filter delays or proof of a single scale error. Quaternion and gyro arrive in separate vendor packets, and the publisher's freshness stamp tracks **any valid packet**, not each field separately. Filtering, acceleration-sensitive orientation fusion, vibration and per-field packet age need an isolated test. No additional axis reversal is justified by this bag.

### Rear position: earlier exact-parity claim is not secure

Deployment explicitly wraps the startup-relative A3 angle to **[−π,π)** and retains its process-start reference across RC cycles. Attempt starts have rear observations **0 / −1.801 / −2.082 rad**. Training resets joint position to zero each episode; `balance_imu_state` reads `joint_pos` directly, with **no explicit ±π wrap**. The saved 91850 evaluation has **four policy-boundary joint-position samples above π**, maximum **3.521434 rad**, minimum −0.081578. This contradicts an unconditional claim that the interfaces always share exactly the same ±π representation. It does **not** prove the backend's unlimited multi-turn behavior; an older termination comment claiming ±π wrapping is not enough to resolve the discrepancy.

There are **69 observed ±π branch jumps** in this bag. In attempt 2 at **14.139745 s**, rear position changes **+2.793→−2.743 rad** and raw rear action changes **−0.150→+56.856** (other inputs also change). Keeping that same recorded state but choosing the equivalent positive angle branch changes the rear raw action to **+7.694**—large sensitivity, but still beyond the torque cap. Overall this branch-only replay reduces rear request exceedances **341→263**, not to zero. Blind per-attempt unwrapping increases exceedances to **713** on the same recorded trajectory. These are **counterfactual input replays, not alternative physical runs or a validated correction**. Verify the exact simulator branch/reference contract before changing hardware behavior. [Phase plot](rear_phase.png), [replay CSV](rear_phase_sensitivity.csv).

**Observation verdict:** conversion/order/sign implementation is largely consistent with the sensor messages and prior physical direction checks. The **whole observation space is not experimentally certified as training-equivalent**: vx/vy are approximations, gyro/attitude timing needs validation, and rear phase/reference parity is unresolved. The bag cannot independently certify every physical IMU axis without labelled motion/video.

## 2. Are the raw policy outputs reasonable?

They are **the correct model's outputs**, not an export/normalization/scaling error. They are not generally reasonable as physically achievable instantaneous targets on this deployed plant.

- Whole-record raw steering spans **−6.036…+25.030** and rear **−2.508…+106.251**. Scaled maxima are approximately **+2251.54° / +667.26 N·m**, before deployment limits.
- **594/894 (66.4%)** steering requests exceed ±90°; **341/894 (38.1%)** rear requests exceed ±6.28 N·m. Even in the first second of each attempt, exceedances are **32/51, 36/51, 34/51** steering and **18/51, 25/51, 23/51** rear. Those early intervals can already include stand contact.
- After clipping, **520/891** consecutive steering changes imply a position-change rate above **23.9 rad/s**; **115** are full +90°↔−90° reversals between policy frames. At the configured slew limit, a full reversal needs at least **π/23.9≈131 ms**, not 20 ms. This is a trajectory-rate comparison, not a rule that every target must settle before the next inference.
- The original 91850 ideal upright/zero-motion observation gives raw `[1.599545,16.720318]`, approximately **+143.886° / +105.004 N·m**. Thus the startup issue survives the encoder correction and does not require tilt/gyro bias. The archived Isaac dataset has **171 startup requests of 63.887–132.002 N·m**; its feed-forward buffer retained unbounded requests despite a clipped diagnostic torque estimate. This does not establish how much total solver torque was delivered, but it invalidates assuming the training path is identical to the bounded C0 path.
- Rear final torque strongly reverses **312** times (both adjacent magnitudes ≥0.5 N·m). Only **12** occur while the exact policy rear request is unchanged; those are explained by changing sampled damping. Most large request reversals therefore accompany policy changes, rather than being random motor commands or solely the host damping loop.

Positive steering command maps to right; positive rear torque maps to forward native negative C0 current. These mappings are consistent with the verified direction contract. A multivariable balance policy need not command torque with the same sign as velocity—it can brake—and a snapshot cannot prove that a particular steering sign is dynamically stabilizing. **Do not flip signs merely because the bike leaned while steering the opposite way.** [Unclipped request plots](policy_requests.png), [rear control decomposition](rear_control.png).

## 3. Did the motors precisely execute the outputs? What delay?

The motors do not receive the raw actions literally:

`raw steer ×1.57 → ±90° clamp → 23.9 rad/s target slew → signed absolute MIT packing → motor PD`

`raw rear ×6.28 → ±6.28 request clamp → −0.05*reported rear speed → ±6.28 final clamp → signed C0 current`

Software conversion is precise: submitted MIT target differs from the slewed target by at most **4.58e-5 rad (0.00262°)**; C0 current quantization contributes at most **0.000150 N·m**. Both steering/rear submission flags succeed on 3,022 recorded active ticks, and measured motion/current confirms actual actuation. Host write success still is not an execution acknowledgement for every frame.

### Steering tracking

| Attempt | RMS actual versus current clipped policy target | RMS actual versus current submitted MIT target | Best submitted-target→published A3 alignment | RMS after that time alignment, without gain/bias correction |
|---|---:|---:|---:|---:|
| 1 | 80.16° | 22.26° | 24 ms; correlation 0.995 | 4.32° |
| 2 | 69.57° | 17.59° | 24 ms; correlation 0.981 | 4.99° |
| 3 | 76.45° | 19.63° | 25 ms; correlation 0.985 | 5.58° |

Using estimated kernel reception times of direct F1 feedback yields approximately **22/21/20 ms** instead of 24/24/25. Reception is estimated from a diagnostic age sampled after loop entry, not firmware acquisition time. Comparing the clipped policy target to A3 instead gives weaker correlations and **49/50/39 ms** alignment, because the slew limiter changes the trajectory, not merely its timestamp. First-second submitted-target/A3 alignments remain **24/23/25 ms**.

Therefore MIT steering functions and largely tracks its **actual input with lag**, but does not precisely reproduce the rapidly reversing **policy target at the same instant**. Even time-aligned error is not zero. These fitted closed-loop alignments are **not universal motor dead times, firmware-loop frequency measurements or proof that the motor is faulty**. Current-derived steering torque ranges **−3.143…+3.625 N·m**, with no recorded samples above 5.6; this does not certify firmware protection or rule out missed peaks. [Alignment scans](tracking_alignment.png).

### Rear actuation

Rear speed ranges approximately **−60…+62 rad/s**, with **26/226/164** >5-rad/s direction reversals. Median reversal spacing is about **39/39/40 ms** (a descriptive crossing statistic, not a unique oscillation frequency). Estimated net rotations are **+10.85/+137.77/+97.07 rad**, despite back-and-forth motion; the training 7.2-rad envelope is crossed after approximately **447/409/437 ms** from the first rear send. These are unwrapped reported-angle estimates, not measured travelled distance, and require adequately sampled feedback.

Final C0 target versus published current-derived torque best aligns at **5/5/4 ms**, correlations **0.935/0.919/0.889**. After alignment, uncompensated torque-error RMS is **1.52/1.66/1.92 N·m**. For aligned requests ≥3 N·m magnitude, measured magnitude is below half the request on **6/40, 108/372, 86/282** samples. This shows imperfect recorded current tracking, but asynchronous cached A1/C0 feedback, speed-dependent capability, filtering, current ramps and power supply are not separated. It is **not an independent shaft-torque or internal-current-loop latency measurement**.

### Host timing

- Policy period median/p99/max **20.00/22.19/26.28 ms**; inference **0.210/0.614/4.177 ms**. Inference generally fits comfortably in 20 ms.
- Policy tick→first matching successful motor-dispatch tick median **4.37/4.03/4.36 ms**. This includes observation/inference/transport/scheduling; actual write occurs within that tick's work, so it is not an exact wire-level latency. Two final policy samples have no corresponding recorded successful dispatch before stopping.
- Within active episodes, MIT-send interval median/p99/max **5.48/10.40/22.92 ms**, mean **5.90 ms (~169.5 Hz)**. First sends after RC-down are excluded so multi-second gaps do not masquerade as control latency.
- Policy IMU/motor callback age medians **8.94/3.95 ms**, maxima **17.39/36.75 ms**. Active CAN steering/rear pair ages median **6.62/6.22 ms**, maxima **42.66/48.02 ms**. Callback ages exclude upstream sensor/firmware age and should not simply be added to every other reported delay.
- Attempt 1's motor stale-feedback latch occurs around **7.25 bag s** (~1.07 s after first inference); policy records motor-state timeout. Event monotonic times, not cross-topic recorder order, establish the order. Attempt 3's worker fails around **40.08 s**, long after the early lean, with `system clock sampling delayed; CAN feedback age is uncertain`. Both stop control, but neither explains the preceding first 0.4–0.6 s loss by itself.
- RC averages **493.9 Hz**, generating **32,239 disabled commands**. Motor-state publication averages about **68.6 Hz while its worker is alive**, not a guaranteed 100 Hz. No CPU/temperature/bus-load trace identifies the cause. [Timing plot](timing.png).

## Suspected causes, ranked by likelihood

1. **Policy–actuator dynamics/limit mismatch — very high.** The excessive ideal startup request, frequent clamping, impossible requested steering changes and large policy-target tracking errors are confirmed. The successful Isaac actuator path does not establish the same bounded/delayed physical path. Whether saturation, startup impulse dependence or tracking phase is dominant needs matched experiments. Evaluate the frozen policy with genuinely enforced hardware torque limits, the exact clamp/slew/damping, sample holds and measured delay before selecting/retraining a policy. Do not merely plot an implicit clipped estimate and call the simulation hardware-bounded.
2. **Body-velocity estimation and IMU timing/coherence mismatch — high-to-medium.** Correct units do not make wheel speed equal root velocity, especially during near-90° steering/slip/stand support. The gyro/quaternion alignment discrepancy is measured; a physical mounting inversion is not. Use synchronized video/markers to validate translation and labelled body-axis tilts to validate signs/scale/field timing. Train with the actual IMU/encoder estimator (or an observation set available on the bike), rather than privileged velocities substituted only at deployment. Zeroing vx is not a demonstrated fix: the offline zero-vx replay worsens rear exceedances **341→604**.
3. **Rear phase/reference contract discrepancy — medium, not a demonstrated sole cause.** Direct training joint positions, explicit hardware wrapping and per-process versus per-episode zeroing are not securely matched. A nonzero start reference cannot explain attempt 1, which starts at zero. Branch jumps strongly affect outputs, but branch/unwrap replay does not eliminate aggressive commands and can worsen them. Probe simulator observation output through known ±π crossings and resets, record raw hardware A3 counts and effective parameters, and define one explicit common representation/reference before changing implementation or retraining.
4. **Sampling/transport/host robustness — medium as a deployment problem, lower for the initial lean.** Stale-feedback and timestamp uncertainty demonstrably stop balancing. Actual update rates and occasional tens-of-ms gaps are below/above nominal expectations, respectively. Profile CPU/thermal throttling, ROS traffic, CAN RTT and clock sampling while supported. Reduce duplicate RC/down traffic, consider isolated single-thread inference/recording tests, and improve timestamp/query handling without weakening stale-data or disable protection. Fresh F1 versus delayed paired queries warrants investigation, not blind watchdog extension.
5. **Rear current/damping and ground-loaded actuator mismatch — medium-to-lower as a distinct hardware cause.** Current-derived torque does not exactly track requested torque. Host damping is sampled/delayed rather than the simulator's implicit damping; it explains some same-request reversals, but only 12/312 strong torque flips, so damping alone is not established as the main oscillator. Perform guarded rear-only pulse/zero-request damping tests and loaded steering target replay. Measure voltage/current/firmware limits before blaming stiction, power supply or motor failure. The good submitted-MIT/position correlation argues against replacing the steering motor first.

Kickstand contact explains why lean persists and why later data are not free-balance performance. It does not establish that the initial controller failed because a servo malfunctioned: servo position/contact timing is absent. **No evidence from this recording requires changing the verified XML/USD geometry or replacing the bicycle.** Interface validation and realistic policy training are the priority, alongside the confirmed timing robustness issues.

## Recommended tests and safety boundary

1. **Supported sensor/reference check, policy disabled:** level the frame independently; record all three labelled angular axes, steering left/right and rear forward/backward, raw passive CAN, IMU, state/control/status and effective parameter dumps. Measure a known full wheel revolution and marked fork angles to independently verify the configured gear ratio/count-to-joint scale, not merely direction. Cross-check integrated gyro/quaternion and A2/A3 at slow motions where temporal averaging cannot hide scale errors. Validate simulator rear observation at +2.8/+3.5/−3.5 rad and at RC/episode resets. This is a proposed probe, not a new simulation performed here.
2. **Rear-only characterization, no policy:** independently secure the frame with rear wheel clear. Use one guarded CAN owner with short low-current pulses and an independently enforced low total torque/speed cap; log C0/A1/B0/A2/A3. Compare zero-request/small-pulse response under damping 0, a lower coefficient and 0.05, with identical initial conditions. Current deployment does **not** provide CLI parameters for varying this damping/total effort cap; prepare an audited test harness first. Do not run the unrestricted ±6.28-N·m backend and assume a small ROS request also limits its damping contribution. Stop immediately on growing oscillation or unsafe speed.
3. **Matched steering replay under support:** first use slow/small targets, then a safely scaled logged 50-Hz policy sequence with the current 200-Hz MIT dispatch, Kp/Kd and load. Compare clipped target, submitted target, A3/F1 and current. Distinguish intentional slew modification, transport delay and motor response. Do not replay raw thousands-of-degree requests or remove the limiter to improve a tracking plot.
4. **Headless matched evaluation:** preserve verified geometry, run the same checkpoint/seed/reset settings against progressively added hardware bounds, estimator, field/transport delays and release conditions; log true submitted forces, early 200 ms, saturation, roll, wheel net and completion metrics. Reconcile rear representation first. Retrain only after establishing the actual controllable interface; validate several seeds/DR settings and retain effective configurations.
5. **Supported/tethered closed loop only after those checks:** stop promptly at loss of balance/contact and command RC down; do not collect seconds of active stand-supported oscillation as a balancing episode. Add a validated fall/net-motion/centering envelope and physical emergency stop before an unsupported retry. Current policy has no 30° roll cutoff and MIT has no measured-angle/overspeed stops; target clipping is not containment. Hard-fault recovery still requires restart.

Capture exact launch commands, package/source and model hashes, `ros2 param dump`, firmware settings, synchronized video showing stand release/contact, passive raw CAN and `/trackstand/{imu,motor_state,motor_control,motor_command,observation,raw_action,policy_timing,policy_event,status,motor_fault,motor_status,motor_event,kickstands_up,kickstands_command}`, `/rosout`, `/parameter_events`, plus servo status when available. Do not run a second selected-motor query/control script alongside `motor_mit_io`; its interference guard will stop it. A passive listener is different from a querying recorder.

## Reproduction, source evidence and validation

This report writes only new offline analysis artifacts. Original recordings, previous report, runtime/configuration, policy/checkpoint, training/simulation sources and root progress records are unchanged. No hardware command, ROS executor, new training or dynamic simulator run was performed.

```bash
source /opt/ros/humble/setup.bash  # use your compatible installed ROS environment
analysis_cache=$(mktemp -d /tmp/trackstand_corrected_cache.XXXXXX)
python3 deploy/experiments/reports/deployment_20261012_120632/decode_mcap.py \
  --bag deploy/experiments/logs/rosbag2_2026_10_12-18_06_07/rosbag2_2026_10_12-18_06_07_0.mcap \
  --output "$analysis_cache"
MPLCONFIGDIR=/tmp/trackstand_corrected_mpl python3 \
  deploy/experiments/reports/deployment_20261012_180607/analyze_corrected.py --cache "$analysis_cache"
```

The reused decoder requires `mcap` with compression dependencies and ROS standard serializers; the analyzer requires NumPy, PyTorch, ONNX Runtime, PyYAML and Matplotlib. Original 91850 checkpoint and archived Isaac dataset must be present locally. Decoder Humble/Jazzy standard-message compatibility is verified by the actual successful decoding/count comparisons, not assumed from the distribution name.

Evidence: [observation/action construction](../../../source/trackstand_deploy/trackstand_deploy/contracts.py), [policy timing/readiness](../../../source/trackstand_deploy/trackstand_deploy/policy_controller.py), [MIT/C0 implementation](../../../source/trackstand_deploy/trackstand_deploy/motor_mit_io.py), [CAN units](../../../source/trackstand_deploy/trackstand_deploy/can_protocol.py), [MIT encoding](../../../source/trackstand_deploy/trackstand_deploy/mit_protocol.py), [IMU publisher](../../../source/trackstand_deploy/trackstand_deploy/imu_publisher.py), [vendor gyro/quaternion conversion](../../../source/trackstand_deploy/trackstand_deploy/YbImuLib/YbImuSerialLib.py), [training observation](../../../../isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/observations.py), [saved actuator-limit study](../../../../isaaclab_train/reports/motor_limits_09221945_91850_2026-10-01/README.md), [pre-correction failure report](../deployment_20261012_120632/README.md), and [sign-correction audit](../../../docs/AUDIT.md).

Machine-readable statistics, sample CSVs and all seven PNG/SVG pairs are in this directory. **Eight synthetic helper tests pass** (synthetic data are not included in the experimental outputs); Pyflakes and parse checks pass. All **42 local report/HTML links** resolve, seven PNG/SVG pairs validate, and CSVs contain the expected **894 policy/phase rows and 3,022 motor-submission rows**. Original bag, model/checkpoint, archived Isaac data and runtime/configuration hashes are unchanged. Exact metadata counts/time coverage, sensor/action reconstruction and independent checkpoint replay also pass. Validation supports the recorded analysis, not a claimed successful balance or physically certified actuator response.
