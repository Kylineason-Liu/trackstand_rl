# Deployment Failure Analysis — Question 1: Motor Model Comparison

Recorded: 2026-10-15.

Question: Is the motor model—including torque/speed limits and how commands are sent—identical between Isaac Lab, MuJoCo, and deployment?

This document preserves the complete preceding answer, including its code excerpts, numerical evidence, qualifications, and conclusions. Repository links are relative to this report; links into the installed Isaac Lab library are machine-local. The analysis reflects the source snapshot inspected on 2026-10-15, not necessarily future revisions or the installed Raspberry Pi package.

## Answer

No. The three systems share the same intended action interface and several nominal constants, but their effective actuator models are not identical.

The differences include command clipping, torque saturation, speed-limit enforcement, feedback timing, and where the steering control loop runs. Some hardware limits are still unverified.

The audit checked both progress records, the current sources, model_91850's saved configuration, installed Isaac Lab implementation, CAN protocol, and existing experiment reports. It also independently rechecked the archived Isaac arrays and compiled/probed the MuJoCo actuators in memory. The audit changed no files; saving this report adds only this document.

## 1. Scope: which deployment implementation?

The main comparison below concerns:

- Isaac Lab: `Trackstand-Balance-v0`, including **09221945/model_91850**.
- MuJoCo: `run_deploy_viewer.py --task deploy_sim` with the trackstand XML.
- Deployment: **`motor_mit_io.py`**, meaning MIT steering + C0 rear control.

This matters because deployment also contains two other motor backends, discussed below.

Also, **the deployment source inspected on 2026-10-15 differs from the source used in the latest analyzed bike bag**: the steering target clamp, steering slew limiter, and rear torque clamps are now commented out. The recorded bag used those restrictions before they were disabled. This distinction is documented in [progress_record.md, §§65–72](../../../../progress_record.md) and the corresponding [text ledger](../../../../progress_record.txt).

## 2. What actually matches?

The nominal interface is:

| Item | Steering | Rear drive |
|---|---:|---:|
| Policy action meaning | Absolute joint-position target | Feed-forward joint torque |
| Raw-action scale | ×1.57 rad | ×6.28 N·m |
| Nominal stiffness | 10 N·m/rad | 0 |
| Nominal damping | 0.2 N·m·s/rad | 0.05 N·m·s/rad |
| Simulator effort setting | 5.6 N·m | 6.28 N·m |
| Simulator speed setting | 23.9 rad/s | 61.6 rad/s |

Isaac's [bicycle_cfg.py](../../../../isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/bicycle_cfg.py#L49) contains:

```python
"steering": ImplicitActuatorCfg(
    joint_names_expr=["steering_joint"],
    effort_limit_sim=5.6,
    velocity_limit_sim=23.9,
    stiffness=10.0,
    damping=0.2,
),
"drive": ImplicitActuatorCfg(
    joint_names_expr=["rear_wheel_joint"],
    effort_limit_sim=6.28,
    velocity_limit_sim=61.6,
    stiffness=0.0,
    damping=0.05,
),
```

These values also appear in [91850's saved env.yaml](../../../../isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-09-22_19-45-06/params/env.yaml#L230).

However, **having the same constants does not mean that all three implementations enforce the same behavior**.

With `--dr`, both simulators additionally randomize stiffness/damping by factors in `[0.9, 1.1]`. Deployment uses fixed requested MIT gains unless explicitly overridden.

## 3. Isaac Lab: unbounded targets, implicit servo, and an important torque caveat

### Commands are scaled but not clipped

The task config registers position and effort actions:

```python
steering = JointPositionActionCfg(
    asset_name="robot", joint_names=["steering_joint"], scale=1.57
)
drive = JointEffortActionCfg(
    asset_name="robot", joint_names=["rear_wheel_joint"], scale=6.28
)
```

Source: [trackstand_rl_env_cfg.py](../../../../isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_cfg.py#L79).

The saved configuration has `clip: null` for both terms and `clip_actions: null` in the [agent configuration](../../../../isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-09-22_19-45-06/params/agent.yaml#L11). Therefore:

```text
steering target = 1.57 × raw steering action
rear effort     = 6.28 × raw rear action
```

There is **no explicit ±90° target clamp, steering target slew limiter, or rear feed-forward torque clamp in this action-processing path**. The task's comment saying actions are in `[-1,1]` is not an enforced bound.

### Steering PD is computed inside the simulator

The installed `ImplicitActuator` documents that the simulator handles the PD integration. Its Python torque calculation is approximate diagnostic information:

```python
self.computed_effort = (
    self.stiffness * error_pos
    + self.damping * error_vel
    + control_action.joint_efforts
)
self.applied_effort = self._clip_effort(self.computed_effort)
return control_action
```

Source: installed [actuator_pd.py](/home/lmbike/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:136).

The last line is crucial: **it returns the original commands, not the clipped diagnostic torque**.

The articulation then submits:

```python
self.root_view.set_dof_actuation_forces(
    self._joint_effort_target_sim, self._ALL_INDICES
)
```

Source: installed [PhysX articulation.py](/home/lmbike/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:290).

Thus, a plot of `applied_torque` bounded at 6.28 N·m does **not** establish that the rear motor's total solver torque was bounded identically to a hardware current command.

### Archived experimental evidence

The audit independently rechecked the original **64 environments × 3,000 steps, seed 42, full-DR** arrays:

| Quantity | Observed maximum absolute value |
|---|---:|
| Rear feed-forward effort submitted to PhysX | **132.00243 N·m** |
| Rear clipped torque diagnostic | **6.28000 N·m** |
| Actual simulated steering speed | **24.27422 rad/s** |
| Actual simulated rear speed | **92.51144 rad/s** |

The submitted rear effort matched the unbounded processed target with **zero numerical difference**. There were **171** rear policy samples above 6.28 N·m, all at episode starts.

These observations establish two things:

- Isaac's speed constraints permit transient overshoot; they are not equivalent to directly clipping velocity state.
- Its clipped diagnostic is not proof of a hardware-equivalent total torque cap.

They do **not** establish that the motor actually delivered 132 N·m of total solver torque. That quantity was not measured.

Evidence: [motor-limit experiment report](../../../../isaaclab_train/reports/motor_limits_09221945_91850_2026-10-01/README.md) and [original motor-limit arrays](../../../../isaaclab_train/reports/motor_limits_09221945_91850_2026-10-01/motor_limit_samples.npz).

## 4. MuJoCo: additional target processing and explicit saturation

### Steering target clipping and slew are active

The actual environment executes:

```python
desired = float(np.clip(action[0] * self.STEER_SCALE, -1.5708, 1.5708))
self._steer_cmd += float(np.clip(
    desired - self._steer_cmd, -23.9 * 0.02, 23.9 * 0.02
))
d.ctrl[0] = self._steer_cmd
d.ctrl[1] = float(action[1]) * self.DRIVE_SCALE
```

Source: [env.py](../../../../mujoco_sim2sim/src/newbike_mujoco/env.py#L394).

Therefore steering target changes are limited to **0.478 rad per 20 ms update**. Isaac's action-processing path does not do this.

A target slew limit also differs from a physical speed limit: it modifies the requested trajectory before the motor responds.

### Control and force limits are both active

The viewer creates these actuator settings:

```python
# Steering
gainprm=[10.0, ...]
biasprm=[0.0, -10.0, -0.2, ...]
ctrllimited=True, ctrlrange=[-1.5708, 1.5708]
forcelimited=True, forcerange=[-5.6, 5.6]

# Rear
gainprm=[1.0, ...]
biasprm=[0.0, 0.0, -0.05, ...]
ctrllimited=True, ctrlrange=[-6.28, 6.28]
forcelimited=True, forcerange=[-6.28, 6.28]
```

Excerpt from [run_deploy_viewer.py](../../../../mujoco_sim2sim/scripts/run_deploy_viewer.py#L274); `...` denotes omitted zero coefficients.

At nominal gains, this means:

```text
steering torque =
    clip(10 × (bounded/slewed target − position) − 0.2 × speed, ±5.6)

rear torque =
    clip(clip(6.28 × raw rear action, ±6.28) − 0.05 × speed, ±6.28)
```

The audit confirmed these compiled settings and repeated a static engine probe:

| Rear stored `data.ctrl` | Rear speed | Resulting actuator torque |
|---:|---:|---:|
| 400 N·m | 0 rad/s | 6.28 N·m |
| 400 N·m | +60 rad/s | 3.28 N·m |
| 400 N·m | −60 rad/s | 6.28 N·m |

This is direct evidence that **stored `data.ctrl` is not necessarily the force actually applied**.

### Speed enforcement differs from Isaac

After every physics step, MuJoCo explicitly changes velocity state:

```python
if abs(d.qvel[self._steer_dof]) > 23.9:
    d.qvel[self._steer_dof] = np.clip(
        d.qvel[self._steer_dof], -23.9, 23.9
    )
if abs(d.qvel[self._rear_dof]) > 61.6:
    d.qvel[self._rear_dof] = np.clip(
        d.qvel[self._rear_dof], -61.6, 61.6
    )
```

Source: [env.py](../../../../mujoco_sim2sim/src/newbike_mujoco/env.py#L412).

That is **not identical to Isaac's solver constraint**, as the archived overshoot demonstrates.

## 5. Current MIT deployment: different limits and a real firmware control loop

### The three host restrictions are currently disabled

Policy scaling now does:

```python
steering = float(action[0]) * STEERING_ACTION_SCALE_RAD
# Disabled by request: steering target clamp.
...
drive = float(action[1]) * DRIVE_ACTION_SCALE_NM
```

Source: [contracts.py](../../../source/trackstand_deploy/trackstand_deploy/contracts.py#L159). Here `...` denotes the intervening commented-out clamp.

The MIT motor callback stores the requests directly:

```python
self._steering_target_rad, self._drive_target_nm = values[1:]

# self._steering_target_rad = max(-limit, min(limit, values[1]))
# self._drive_target_nm = max(-DRIVE_EFFORT_LIMIT_NM,
#                             min(DRIVE_EFFORT_LIMIT_NM, values[2]))
```

The active statements and commented originals are in [motor_mit_io.py](../../../source/trackstand_deploy/trackstand_deploy/motor_mit_io.py#L380). The excerpt abbreviates `self._steering_limit_rad` as `limit` in the commented expression.

The steering control path likewise uses:

```python
proposed = target_rad
# Disabled by request: steering target slew.
```

And rear control uses:

```python
drive_damping = -DRIVE_DAMPING_NM_S_PER_RAD * diag["drive_velocity_rad_s"]
drive_unclipped = target_drive_nm + drive_damping
drive_torque = drive_unclipped
# Disabled by request: final rear torque clamp.
```

Source: [motor_mit_io.py](../../../source/trackstand_deploy/trackstand_deploy/motor_mit_io.py#L990).

Consequently, **current MIT deployment does not implement MuJoCo's bounded/slewed command path**.

### Steering is an absolute MIT position command

The command conversion is:

```python
native = (
    self._mit_zero_rad
    + self._steering.encoder_sign * self._steering.gear_ratio * target_rad
)
payload = mit_motion_payload(native, self._mit_ranges, self._mit_kp, self._mit_kd)
self._send(0x400 | self._steering.host_id, payload)
```

Source: [motor_mit_io.py](../../../source/trackstand_deploy/trackstand_deploy/motor_mit_io.py#L771).

With steering host ID `0x102`, motion frames go to **`0x502`**. The payload requests:

- Absolute native position.
- Zero target velocity.
- Kp = 10 and Kd = 0.2 by default.
- Zero feed-forward torque.

The motor's firmware performs the PD/current-control work; Python does not calculate steering torque in this backend.

Source: [mit_protocol.py](../../../source/trackstand_deploy/trackstand_deploy/mit_protocol.py#L84).

These are the same **intended PD coefficients**, but not proof of identical dynamics. For example, quantization makes the encoded gains approximately **10.01221 and 0.200244**, rather than exactly 10 and 0.2.

### MIT encoding bounds are not physical protection

Powered startup sets **F0 Pos_Max = 3 native rad**, while preserving the motor's reported velocity/torque encoding ranges.

A requested native position outside `[-3,3]` causes encoding failure and the fault/disable path—not clipping to ±90° and continued control.

Furthermore:

- `motor_mit_io.py` does not establish a verified **±5.6 N·m total steering PD torque cap**.
- It does not configure a verified physical **23.9 rad/s steering speed cap**.
- F0 velocity/torque ranges describe protocol encoding; they do not prove those total-output protections.
- The CAN document's B2/B3 limits are described for position/speed modes; their applicability to MIT is not established.

Evidence: [startup implementation](../../../source/trackstand_deploy/trackstand_deploy/motor_mit_io.py#L595), [encoder rejection](../../../source/trackstand_deploy/trackstand_deploy/mit_protocol.py#L45), and [documented firmware-limit uncertainty](../../../docs/MOTOR_MIT_IO.md#L188).

### Rear control is current control, not a direct physical torque interface

The host converts its torque request into C0 current:

```python
current_a = motor.torque_sign * torque_nm / torque_constant
payload = q_current_payload(current_a)
self._send(motor.host_id, payload)
```

Source: [motor_mit_io.py](../../../source/trackstand_deploy/trackstand_deploy/motor_mit_io.py#L747).

C0 sends signed integer milliamps to **`0x101`**. With the host clamps disabled, neither ±6.28 N·m nor 61.6 rad/s is enforced by this command calculation.

The firmware's actual current limits and response still matter. The protocol also exposes **B4 current slew**, but this backend does not read or configure it. Its real setting therefore remains unknown.

Protocol evidence: [CAN Rev.3.09b0 PDF](<../../../../tsinghua_bike_project/assign_2_Hardwares/assign_2_Hardwares/3.RearSteer Servo【GIM3510-8】/CAN Protocol Doc/自定义CAN通信协议_3.09b0.pdf>), printed pp. 9–10 (B2/B3/B4), 13 (C0), and 20–22 (MIT).

## 6. Effective-limit comparison

| Property | Isaac Lab | MuJoCo viewer | Current MIT deployment |
|---|---|---|---|
| Steering target clamp | None in action processing | ±1.5708 rad | ±90° clamp commented out |
| Steering target slew | None | 23.9 rad/s at 50 Hz | Commented out |
| Steering effort | Configured 5.6; implicit solver, clipped diagnostic | Explicit actuator-force limit ±5.6 N·m | Total firmware limit unverified |
| Rear feed-forward clamp | None before PhysX submission | ±6.28 before damping | Commented out |
| Rear final torque clamp | Diagnostic clipping is not proof of total solver cap | Explicit ±6.28 after damping | Commented out |
| Steering/rear speed | Solver limits; measured overshoot | Explicit post-step state clipping | Matching physical limits not established |
| Steering travel | Joint limit approximately ±90° | Joint limit approximately ±90° | Physical stops not certified by these sources; MIT encoding bound is separate |

**Neither the current deployment nor the historical capped deployment is an exact copy of both simulators.**

## 7. Frequencies and feedback are also different

| Layer | Isaac Lab / 91850 | MuJoCo | MIT deployment |
|---|---|---|---|
| Fresh policy target | 50 Hz | 50 Hz simulated time | Nominal 50 Hz wall time |
| Target submission | Held targets written each 400 Hz physics step | `data.ctrl` written at 50 Hz, held across substeps | MIT steering/C0 rear nominally 200 Hz |
| Inner actuation | PhysX implicit integration | Actuator evaluated during 400 Hz physics | Firmware loop frequency unknown |
| Feedback used by host | Simulator state updated each physics step | Simulator state | Nominal 100 Hz serialized CAN query rounds |

Evidence: [Isaac stepping loop](/home/lmbike/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:193), [MuJoCo substeps](../../../../mujoco_sim2sim/src/newbike_mujoco/env.py#L404), and [MIT frequency divisions](../../../source/trackstand_deploy/trackstand_deploy/motor_mit_io.py#L52).

An important consequence concerns rear damping:

- Simulators calculate damping using their evolving simulated velocity.
- Deployment repeatedly calculates damping using the latest **sampled, delayed CAN velocity**.

Even with the same coefficient `0.05`, these are not identical feedback loops.

Also, **200 Hz transmission does not mean a 200 Hz firmware PD loop**, nor four fresh position targets per policy interval. It mainly resends the latest 50 Hz target.

## 8. Existing hardware tests confirm that a real response model is still missing

The free-steering MIT tests with **Kp=10, Kd=0.2**, matching deployment's requested gains, measured:

| Sine frequency | Position fundamental amplitude | Gain versus held command | Phase-equivalent tracking lag |
|---|---:|---:|---:|
| 2 Hz | 0.9892 rad | 99.2% | 21.41 ms |
| 5 Hz | 0.7790 rad | 79.2% | 30.63 ms |
| 10 Hz | 0.3600 rad | 38.5% | 29.04 ms |

At high frequency, speed plateaued around **20.6 rad/s**, and the 10 Hz test reached a **9.85 N·m current-derived torque estimate**, above the simulator's 5.6 reference.

These are unloaded tests, not loaded balancing results. The lag is frequency-specific phase lag—not a universal pure delay—and torque is measured current × B0 constant, not calibrated shaft torque. Nevertheless, they prevent a claim of proven actuator equivalence.

Evidence: [MIT/C2 experimental comparison](../mit_sine_sweep_20261007/README.md).

Neither simulator currently adds a separately identified firmware/transport response model reproducing those measurements. Their actuators are idealized PD/effort models, not complete electrical motor models with measured voltage-dependent torque–speed curves and thermal limits. The retained datasheets also distinguish **stall torque from nominal continuous torque**; the simulator uses stall values as fixed effort settings.

Datasheet evidence: [GIM4310-10 steering motor](<../../../../tsinghua_bike_project/assign_2_Hardwares/assign_2_Hardwares/3.RearSteer Servo【GIM3510-8】/GIM4310-10_motor_datasheet.png>) and [GIM3510-8 rear motor](<../../../../tsinghua_bike_project/assign_2_Hardwares/assign_2_Hardwares/3.RearSteer Servo【GIM3510-8】/GIM3510-8_motor_datasheet.png>).

## 9. The other deployment backends are different again

| Backend | Steering command | Steering loop location |
|---|---|---|
| `motor_io.py` | C0 current | Host PD at nominal 200 Hz, explicitly torque-clipped |
| `motor_io_position.py` | C2 absolute encoder count | Firmware position/speed loops; B2/B3 configured/read back |
| `motor_mit_io.py` | MIT absolute position + gains | Firmware MIT PD; matching total-output limits unverified |

Sources: [C0 host PD](../../../source/trackstand_deploy/trackstand_deploy/motor_io.py#L450), [C2 conversion and limits](../../../source/trackstand_deploy/trackstand_deploy/motor_io_position.py#L57).

The packaged [launch file](../../../source/trackstand_deploy/launch/trackstand.launch.py#L28) still selects **`motor_io`**, not MIT. Therefore the executable actually started must be recorded when comparing experiments.

## Conclusion

**Same intended interface and nominal constants: yes. Identical motor model, enforced limits, and command delivery: no.**

The most consequential confirmed mismatches are:

1. **Different saturation behavior**, particularly the unbounded Isaac rear effort submission versus MuJoCo's explicit clamps and current deployment's disabled host clamps.
2. **Different steering target processing**: MuJoCo clips/slews; Isaac does not; current MIT deployment no longer does.
3. **Different speed enforcement**, with Isaac overshoot, MuJoCo state clipping, and unverified matching hardware limits.
4. **Different low-level feedback and dynamics**, including sampled rear damping, CAN timing, and undocumented firmware loops.

These establish an actuator-interface mismatch that must be accounted for when analyzing deployment failure. They do **not**, by themselves, prove which mismatch caused the bike to fall or that the motors need replacing.
