# Question 3 — Can observation differences explain the excessive policy outputs?

Investigation: 2026-10-16. Policy throughout: **09221945 / model_91850**. No model, runtime source, motor settings or verified USD/XML geometry was changed. The observation ablations described below change two channels **in experiment-local process memory**, not deployment/training source files.

## Answer

**Yes: deployment's body-velocity observations are materially different, and substituting them is sufficient to cause catastrophic failure and much larger outputs inside the otherwise unchanged Isaac simulation.** This is stronger evidence than simply noticing different observation distributions.

**No: I did not find a general quaternion-order, angular-velocity-frame, radians/degrees, channel-order or actor-export bug that explains the Isaac–MuJoCo gap.** Their observation mathematics agrees to floating-point accuracy for matched states in the common rear-angle range. MuJoCo reaches substantially different physical states under its different actuation/contact/control behavior, which then produces different observations and outputs. Rear-angle representation is a real additional discrepancy, but the tested representation change did not consistently improve balancing.

This does not prove that velocity substitution is the only real-bike cause. The bag lacks independently measured bike translational velocity and exact stand-contact timing; IMU field timing and actuator discrepancies remain relevant.

## 1. Actual observation contracts

Indices below are **one-based**; code uses zero-based indices.

| Channels | Isaac Lab | Current MuJoCo viewer environment | Deployment |
|---|---|---|---|
| 1–2: gravity y/z | Inverse-rotate world `[0,0,-1]` using root quaternion | Same operation using free-root quaternion | Same operation using corrected IMU quaternion |
| 3–5: body angular velocity, rad/s | Rotate root-link world angular velocity into body frame | Free-joint angular qvel, already body-frame | Corrected IMU gyro, already converted to rad/s |
| **6: body vx, m/s** | **Actual root-link body velocity** | **Actual free-root body velocity** | **`0.07 * rear_encoder_velocity`** |
| **7: body vy, m/s** | **Actual root-link body velocity** | **Actual free-root body velocity** | **Always zero** |
| 8: steering position, rad | Direct joint position; reset reference zero | Direct hinge position; reset reference zero | Signed A3 count relative to motor-node startup zero, wrapped by decoder |
| **9: rear position, rad** | **Direct backend joint position; no explicit ±π wrap** | **Explicit `[-π,π)` wrap** | **Explicit `[-π,π)` wrap; startup zero persists across RC cycles** |
| 10–11: steering/rear speed, rad/s | Joint velocity | Joint velocity | Signed A2 speed, converted from 0.01 RPM units |
| 12–13: sin/cos steering | sin/cos of channel 8 | Same | Same |

Isaac's executable source:

```python
q = asset.data.root_quat_w.torch  # xyzw in the installed stack
ang_vel = math_utils.quat_apply_inverse(q, asset.data.root_link_ang_vel_w.torch)
lin_vel_xy = math_utils.quat_apply_inverse(q, asset.data.root_link_lin_vel_w.torch)[:, :2]
drive_pos = asset.data.joint_pos[:, drive_idx : drive_idx + 1]
```

MuJoCo's executable source:

```python
ang_vel = np.asarray(d.qvel[adr + 3:adr + 6])
vel_b = R.T @ np.asarray(d.qvel[adr:adr + 3])
drive_pos = (drive_raw + np.pi) % (2 * np.pi) - np.pi
```

Deployment's executable observation construction includes:

```python
float(wheel_radius_m) * rear_wheel_velocity_rad_s,  # vx
0.0,                                             # vy
```

Sources: [Isaac observation](../../../../isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/observations.py), [Isaac configuration](../../../../isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_cfg.py), [MuJoCo observation](../../../../mujoco_sim2sim/src/newbike_mujoco/env.py), [deployment construction](../../../source/trackstand_deploy/trackstand_deploy/contracts.py), [policy sampling](../../../source/trackstand_deploy/trackstand_deploy/policy_controller.py), [encoder decoder](../../../source/trackstand_deploy/trackstand_deploy/can_protocol.py), [motor-state publication](../../../source/trackstand_deploy/trackstand_deploy/motor_mit_io.py).

### Other differences and non-differences

- **Quaternion conventions are handled correctly:** Isaac/deployment use xyzw; MuJoCo's helper uses wxyz. They encode the same rotation when reordered correctly. This was tested, not inferred from docstrings.
- **Noise:** both simulations add gravity σ=0.001 and gyro σ=0.0012 rad/s white noise even without DR. DR additionally adds per-episode gravity σ=0.002 and gyro σ=0.0002 residual biases. Deployment adds no synthetic noise; it has the real sensor's errors instead. Removing white noise in fixed-state replay barely changes the output statistics below.
- **Observation bounds:** Isaac sanitizes/clamps its term to ±100 before manager white noise; deployment sanitizes/clamps its final vector to ±100. MuJoCo has no matching final finite-value/±100 guard. The captured inputs are finite and below these bounds, so this discrepancy does not explain these recordings.
- **Normalization/history:** the saved 91850 actor has an identity observation normalizer, no actor observation history and no extra observation scaling. The dormant Isaac latency helper is not called. Deployment consumes the latest asynchronous IMU and motor messages rather than a synchronized physics snapshot.
- **Real units:** the vendor gyro decoder multiplies by `π/180`; deployment does not convert the already-radian gyro again. A2 uses `0.01 RPM * 2π/60 / gear_ratio`; A3 uses `2π/16384 / gear_ratio`. Configured steering/rear encoder signs are both −1, consistent with the recorded physical direction checks. This code audit is not an independent physical calibration of every IMU axis or gear ratio.

IMU evidence: [publisher](../../../source/trackstand_deploy/trackstand_deploy/imu_publisher.py), [freshness adapter](../../../source/trackstand_deploy/trackstand_deploy/imu_serial.py), [vendor conversion](../../../source/trackstand_deploy/trackstand_deploy/YbImuLib/YbImuSerialLib.py), [parameters](../../../source/trackstand_deploy/config/params.yaml).

## 2. Matched-state and same-input tests

The [analysis script](question_3_observations/analyze.py) executes the **actual Isaac observation function** extracted from disk against supplied CPU tensors, the actual `BikeBalanceEnv.observe()`, and actual deployment `build_observation()`.

For **1,000 synthetic normalized orientations/world velocities/joint states**, with rear angle inside the common ±π branch and noise/bias disabled:

- Isaac versus MuJoCo: maximum discrepancy in any observation channel **9.54×10⁻⁷**.
- Deployment versus Isaac, excluding its intentionally different vx/vy: maximum discrepancy **9.54×10⁻⁷**.
- MuJoCo free-joint body angular velocity independently matches `mj_objectVelocity(..., local=1)` to **3.56×10⁻¹⁵ rad/s**.
- The small float32 observation roundoff changes raw outputs by at most **1.53×10⁻⁵ steering / 3.82×10⁻⁵ rear** in these synthetic tests—not factors of ten or a hundred.

All baseline simulator/bike recorded inputs were also replayed through 91850 ONNX and a checkpoint-reconstructed PyTorch actor. Across those baseline datasets, recorded-action reproduction error is at most **5.73×10⁻⁶**, and PyTorch–ONNX disagreement at most **1.53×10⁻⁵**. Thus the large bike/MuJoCo values are genuine responses of the same deterministic actor, not a different model or hidden normalization/export gain.

Synthetic parity probes are not physical bike measurements or a new balancing result. They deliberately do not claim rear-angle equivalence outside the shared branch.

## 3. Fresh baseline captures: outputs and actual input distributions

Fresh Isaac nominal/DR evaluations: **64 environments × 3,000 actions**, seed 42, 50 Hz policy / 400 Hz physics, standard terminations, unchanged checkpoint/actions/physics. Actual actor-input vectors are captured without additional observation-manager calls or RNG draws. Raw actions and reset flags are **bitwise identical** to the preceding measured-actuation evaluations in both modes.

Fresh MuJoCo nominal/DR captures: **64 independent seeds 42–105 × 3,000 actions**, using the current viewer's augmented model and actual environment, including automatic resets. MuJoCo retains its own time-out/fallen-only termination set. These are not identical episode populations or a controlled simulator-performance benchmark.

Bike evidence: **894 policy frames in three attempts** from the corrected-sign 2026-10-12 18:06:07 deployment. Both wheels were grounded, nobody restrained/disturbed the bike, and the user reports stand contact within <1 s. Much of the bag is consequently stand-supported, not successful free balancing. These are pre-limit-disabling data, not evidence of the current unrestricted backend's behavior.

Here “OOB” means `abs(raw_steer*1.57)>π/2` or `abs(raw_rear*6.28)>6.28`, with 10⁻⁵ tolerance in native units. It concerns **demand**, not actual torque or actual travel.

| Capture | Policy samples | Steering demand OOB | Rear demand OOB | Max abs raw steering / rear |
|---|---:|---:|---:|---:|
| Isaac, nominal | 192,000 | 0.067% | 0.067% | 1.63 / 16.92 |
| Isaac, DR | 192,000 | 0.659% | 0.089% | 2.64 / 21.02 |
| MuJoCo, nominal | 192,000 | 9.145% | 0.841% | 12.88 / 38.03 |
| MuJoCo, DR | 192,000 | 8.941% | 1.248% | 15.22 / 48.92 |
| Bike | 894 | 66.443% | 38.143% | 25.03 / 106.25 |

After excluding each episode's first 0.5 s, Isaac DR still has 0.532% steering / **0% rear** OOB; MuJoCo DR has 7.084% / 0.386%; bike has 65.809% / 37.132%. Thus resetting more often contributes, but does not explain the entire gap.

Actual actor inputs are dramatically different. The following are **95th percentiles of absolute value**, not sensor noise specifications:

| Input | Isaac DR | MuJoCo DR | Bike |
|---|---:|---:|---:|
| gravity y, normalized | 0.0462 | 0.0893 | 0.4691 |
| body roll rate, rad/s | 0.0370 | 0.8689 | 1.9185 |
| body yaw rate, rad/s | 0.0470 | 1.2258 | 2.6056 |
| supplied vx, m/s | 0.0083 | 0.5169 | 3.8687 |
| supplied vy, m/s | 0.0026 | 0.0930 | 0 |
| steering speed, rad/s | 1.3247 | 10.5961 | 20.2852 |
| rear speed, rad/s | 0.1233 | 21.9412 | 55.2673 |

For the two simulations, these are largely **different physical motions**, not different unit conversion. On the bike, vx is a different estimator as well. Being outside this successful-evaluation distribution is not proof that an input never occurred during PPO training; actual training input trajectories were not logged here.

Evidence: [all-feature ECDFs](question_3_observations/observation_distributions.png), [numeric distributions](question_3_observations/feature_distributions.csv), [rear-phase/output operating regions](question_3_observations/rear_phase_operating_regions.png), [machine-readable summary](question_3_observations/summary.json).

## 4. Velocity substitution: fixed-state evidence

Wheel circumferential speed is **not generally root-link translational velocity**. Slip, startup transients, lean/pitch and the root origin's location matter. Even ideal rigid-frame kinematics give `v_root = v_rear_axle + omega_frame × (r_root - r_rear_axle)`. A no-lateral-slip constraint at a contact point therefore does not imply zero lateral velocity at every body point; wheel rolling/slip also determines whether circumferential speed estimates axle translation in the first place.

This discrepancy exists even in the successful Isaac simulation. Comparing `0.07*rear_speed` with its true vx gives maximum errors **4.13 m/s nominal / 4.54 m/s DR**, concentrated in transients. DR 95th-percentile absolute error is only 0.00254 m/s across the whole run: the important large early errors are rare, not constant.

For a concrete example, **Isaac DR env 24, action index 3 (60 ms)**:

| Quantity | Original actual input | Deployment-style replacement |
|---|---:|---:|
| vx, m/s | +0.180148 | −0.425714 |
| vy, m/s | −0.219561 | 0 |
| Raw steering | −0.329742 | +9.122833 |
| Raw rear | +0.179491 | +40.185425 |

All other eleven observations are held identical. Rear scaled demand changes approximately **1.13→252.36 N·m**. This is an **offline same-state replay**, not torque actually delivered or a different physical run.

Across all 192,000 Isaac DR inputs, the combined substitution raises steering/rear OOB **0.659/0.089%→0.996/0.278%**, and raw maxima **2.64/21.02→11.56/40.19**. Single-channel replays and interactions are saved separately; the joint result should not be attributed solely to vx.

Evidence: [same-state transient example](question_3_observations/isaac_velocity_mismatch_example.png), [all-simulation fixed-state comparison](question_3_observations/velocity_substitution_effect.png), [counterfactual actions](question_3_observations/counterfactual_actions.npz).

## 5. Velocity substitution: controlled closed-loop evidence

Fixed-state replay keeps the original recovering trajectory, so it cannot capture the feedback loop's subsequent divergence. I therefore ran **actual closed-loop ablations**, not just replay.

### Isaac: two observation channels only

The experiment-local capture wrapper replaces only zero-based channels 5/6 with `0.07*rear_velocity` and `0`, before the same observation-manager white noise. Checkpoint, seed, state-reset settings, DR settings, physics, action scaling, actuator limits and termination functions stay unchanged. Base configurations and source-integrity results are saved. **The YAML records the original configuration, not the observation manager's process-local override: read the `deploy_velocity_ablation` metadata flag, invocation and capture script together.** Initial physical states and the other eleven observation channels match the baseline; under DR the root's initial lateral velocity can already be nonzero, so initial raw actions need not match after the intended substitution. Under nominal conditions the initial full observation and initial action are bitwise identical.

| Isaac condition | Mean completed episode | Reached 30 s / completed | Steering demand OOB | Rear demand OOB |
|---|---:|---:|---:|---:|
| Nominal, original observations | 30.000 s | 128/128 | 0.067% | 0.067% |
| Nominal, deployment vx/vy | **0.220 s** | **0/17,408** | **72.733%** | **54.533%** |
| DR, original observations | 27.379 s | 76/129 | 0.659% | 0.089% |
| DR, deployment vx/vy | **0.263 s** | **3/14,326** | **66.830%** | **48.509%** |

In the nominal substitution test **every completed episode lasts exactly 11 actions / 0.22 s and fires `fallen`**. In DR, only **3/14,326** reach the 30 s horizon; the evaluator's three-decimal timeout rate rounds to 0.000. About **85.3%** fire `fallen` and **19.1%** fire `yaw_deviation`; terms can overlap. Do not interpret the printed aggregate `fall_rate` as the fraction firing only `fallen`.

Steering/rear demand OOB rises **0.659/0.089%→66.830/48.509%**. Raw maxima become **29.95 / 90.41**, resembling the magnitude regime of the failed bike recording despite using the Isaac plant, not the bike's motor timing or MuJoCo physics.

Evidence: [Isaac closed-loop graph](question_3_observations/isaac_closed_loop_observation_ablation.png), [original nominal evaluation](question_3_observations/isaac_no_dr_gpu/eval.txt), [substituted nominal evaluation](question_3_observations/isaac_no_dr_estimator_gpu/eval.txt), [original DR evaluation](question_3_observations/isaac_dr_gpu/eval.txt), [velocity-substituted DR evaluation](question_3_observations/isaac_dr_estimator_gpu/eval.txt).

### MuJoCo: independently reproduced with eight matched seeds

Seeds 42–49, 3,000 actions/environment, same original actuator/controller settings and only vx/vy substituted. Each mean is over completed episodes; final unfinished episodes are excluded. Consequently the number of completed episodes differs.

| Mode | Original mean episode | Deployment vx/vy mean episode | Original steering / rear OOB | Deployment vx/vy steering / rear OOB |
|---|---:|---:|---:|---:|
| No DR | 7.856 s | 0.439 s | 9.23% / 0.81% | 83.56% / 48.39% |
| DR | 4.627 s | 0.326 s | 9.38% / 1.15% | 75.93% / 43.73% |

Original nominal has **2/52** timeouts; the estimator version has **0/1,090**. Original DR has **8/68** timeouts; estimator **0/1,467**. These small-seed diagnostics are not a universal benchmark of the policy or a claim that no MuJoCo run can balance.

Evidence: [MuJoCo closed-loop graph](question_3_observations/closed_loop_observation_ablation.png), [captured ablation trajectories](question_3_observations/mujoco_estimator_ablation.npz).

**Inference:** the observation substitution alone can cause severe instability and greatly increase OOB actions in both simulators. This elevates observation-contract alignment from a plausible concern to a high-priority experimentally supported cause of transfer failure. It does not establish that correcting those channels alone will fix the real bike.

## 6. Rear-angle discrepancy: confirmed, but not a demonstrated sole explanation

The MuJoCo comment claiming PhysX always wraps joint position to ±π is **incorrect for the installed, tested task**. Fresh Isaac DR inputs contain **four rear-angle samples above π**, maximum **3.521434 rad**.

After each baseline evaluation, a separately labelled synthetic backend probe writes known rear angles and advances one physics step. Requested **+3.5 rad remains about +3.499 rad**, and −3.5 remains about −3.5; the actual observation term returns those values, not the equivalent ∓2.783 branch. Requested ±7.1 produces approximately ∓5.466, consistent with a 4π branch in these probes, not proof of an unlimited multi-turn representation.

The network cares about the scalar branch. Holding an ideal upright/zero-motion observation otherwise unchanged:

| Supplied rear angle | Raw steering | Raw rear |
|---|---:|---:|
| +3.5 rad | −0.462824 | −0.054442 |
| 3.5−2π = −2.783185 rad | +1.212733 | +14.293247 |

Both represent the same physical wheel orientation but **not the same network input**.

However, replacing MuJoCo's rear observation with a **candidate ±2π / 4π-period branch**, all else unchanged, does not consistently improve the eight-seed closed-loop results: nominal mean **7.856→6.745 s**, DR **4.627→4.486 s**. This candidate is not a fully validated general PhysX branch emulator. It demonstrates why blindly changing wrapping is not an established fix.

Deployment also zeros A3 once per motor-node startup rather than every RC release. The recorded three attempt starts have rear angles **0 / −1.801 / −2.082 rad**. The earlier bag replay found strong branch sensitivity, but blind unwrapping worsened rear OOB counts; no code change is justified solely by this analysis.

Evidence: [DR backend probe](question_3_observations/isaac_dr_gpu/rear_angle_probe.json), [nominal backend probe](question_3_observations/isaac_no_dr_gpu/rear_angle_probe.json), [candidate branch trajectories](question_3_observations/mujoco_phase_ablation.npz), [earlier bike phase analysis](../deployment_20261012_180607/README.md).

## 7. Real-sensor timing: additional unresolved observation mismatch

The recorded observation vector faithfully reproduces its sensor messages, but that does not prove synchronous or physically correct state estimation. Previously measured policy callback ages are median **8.94 ms IMU / 3.95 ms motor**, with maxima **17.39/36.75 ms**; those exclude sensor firmware age.

The IMU driver separately parses raw gyro and quaternion packets. Its adapter refreshes `last_valid_frame_monotonic` on any parsed frame, and the publisher reads latest cached quaternion and gyro without their individual acquisition timestamps. The same ROS header therefore does **not** establish simultaneous measurements. Motors likewise publish paired queried values with acquisition-time differences, and the policy does not resample all sensors to a common time.

The earlier quaternion-increment/gyro comparison gives roll/pitch best correlations about **0.787/0.701** after approximately **18/12 ms** relative shifts. Those are descriptive fits, not proven firmware latency, scale calibration or a demonstrated axis inversion. Nine of the 894 recorded policy frames have all three gyro values exactly zero; zeros alone are not proof of a faulty sensor.

There is not enough independent on-bike motion ground truth here to determine every physical sign/scale, actual body vx/vy, slip ratio or exact sensor delay. The source conversions and the bag's internal consistency should not be overstated as physical certification.

## 8. What to change or verify next

1. **Highest priority: make observations deployable by construction.** Train/evaluate with the same IMU/encoder-derived estimator available on the bike, or design a new observation set without privileged body vx/vy. Keep the true simulated velocity for rewards/diagnostics if useful. Removing channels changes network input dimension and requires new training/export. An estimator change alone is sufficient to break frozen 91850; do not silently swap inputs in the real controller and expect recovery.
2. **Independently validate the estimator without feeding forbidden sensors into policy.** With policy disabled and the bike supported, record marked-wheel motion plus synchronized video/known-distance frame motion, lean/yaw and IMU/encoders. Include rear spin with little frame translation, slow rolling, steering changes and stand-release transients. This distinguishes body velocity from wheel circumferential speed. External motion measurement can be validation ground truth while policy inputs remain IMU/encoders only.
3. **Specify one rear-angle representation and reset reference.** Probe controlled ±π/±2π crossings and multiple revolutions in both engines, then match the same deliberate mapping in training, MuJoCo and deployment. Validate RC/startup referencing before selecting a correction. The present six-state backend probes do not certify all multi-turn behavior.
4. **Measure IMU field freshness, not just node receipt times.** Log separate raw-gyro/quaternion packet timestamps/counters and passive CAN acquisition times, with labelled single-axis motion. Add measured delays/sample-holds to a new training/evaluation setup; do not disable stale-data protection to hide the discrepancy.
5. **Keep actuation alignment as a separate controlled factor.** Question 1/2 already establish motor-control/limit differences. Once the observation contract is fixed for a new policy, repeat matched evaluations with actual steering/rear control, saturation and delay. The current MuJoCo baseline gap persists even with true root velocity, so velocity estimation cannot explain every Isaac–MuJoCo difference.

Do not start an unsupported balance retry merely from these findings. The experiments do not restore the commented deployment motor target/slew/torque bounds or prove an improved hardware safety envelope.

## Artifacts, reproduction and limits

- [Experiment scripts/data/graphs](question_3_observations/README.md), [summary](question_3_observations/summary.json), [independent checks](question_3_observations/verification.json).
- Same model checkpoint SHA256: `f77aadf7c021a1d20603a636daf757601dca2fe4faf264d5bba99c2dcf2b5c5e`; ONNX `9232a0723df8806e8b5f6fb4e39a7a11ccb5af8bf058d9026eddb57dab8fd806`.
- No new hardware experiment, CAN command, ROS executor, model training or runtime source/configuration edit was performed. Only new report-local scripts/artifacts were added. Existing reports, root progress records, recordings and checkpoint files were preserved.
- Isaac's GPU was unavailable inside the sandbox; successful captures used explicitly approved GPU access. The initial failed sandbox launch is not an experimental result. Existing simulation warnings remain in the saved logs; actual baseline trajectories match the preceding evaluation bitwise.
- The repository's geometry remains untouched. This investigation concerns observation semantics, measured state distributions and feedback behavior, not a claim that the verified bike models are wrong.
