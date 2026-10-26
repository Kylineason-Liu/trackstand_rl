# Trackstand-Balance-v1 deployment alignment — 2026-10-26

Deployment now uses the current v1 observation and bounded-action contract,
with the DR-trained **2026-10-21_11-18-29 / model_2500** actor. This replaces
the v0/91850 deployment contract. Rear output remains **torque**, as in the
current v1 source; the separate MIT rear-velocity proposal was not implemented.
This is a software migration, not a new balancing or hardware validation result.

## Comparison and fixes

| Area | Before this migration | Current deployment |
| --- | --- | --- |
| Policy | 13-input v0 Gaussian 91850 | 11-input v1 Beta 2500; correct deterministic mean mapping |
| Linear velocity | Rear-speed-derived X estimate and constant Y | Both removed, matching v1 |
| Observation bound | Finite values clipped to ±100 | No finite clipping; non-finite repair retained |
| Encoder state | Startup-relative A3 wrapped at ±π | Continuous signed A3 displacement in motor_state_v3 |
| Rear policy reference | Process-start zero reused across RC cycles | New software zero for each enabled episode |
| Rear policy angle | Single-turn ±π representation | Measured PhysX 4π branch after episode rebasing |
| Steering target clip | Disabled in policy/MIT; legacy bounds used ±π/2 | Explicit ±1.57 rad clip, matching v1 action configuration |
| Steering slew | Disabled in MIT; legacy nodes ramped at 200 Hz | 23.9 rad/s, updated once per 50-Hz policy inference; ≤0.478 rad/update |
| Motor target delivery | Backend-dependent target ramps/bypasses | Hold the conditioned steering target between policy actions |
| Rear controller damping | -0.05 × measured speed | Zero, matching current v1 IdealPD gains |
| Rear torque cap | Disabled in MIT | Active ±6.28 N m cap; current conversion uses measured B0 |
| Fallen check | Inversion only; YAML max_roll_rad unused | v1 abs(gravity_y) > sin(0.5236), or gravity_z > 0 |
| Startup pose | No explicit policy arming check for fork alignment | Wait until steering is within 5° of fixed straight reference |
| Launch | Host-PD/C0 motor_io | motor_mit_io: steering MIT, rear C0 |
| Diagnostics | Old contract/model identity | v1 observation labels; episode/rear reference/model/checksum in status/events |

The old 5° ordinary motor-fault clearing requirement remains. The new 5° policy
arming tolerance reuses that commissioning scale; it is a deployment readiness
guard, **not a training observation clamp or a limit on steering while balancing**.
Steering retains the straight reference established at motor startup. The code
never declares a physically turned fork to be the new straight position.

## Exact current input and command paths

The 11 observations are, in order:

```text
gravity_y, gravity_z, gyro_x, gyro_y, gyro_z,
steering_position, rear_position, steering_velocity, rear_velocity,
sin(steering_position), cos(steering_position)
```

Gravity is normalized world-down rotated into the calibrated bike frame by the
IMU quaternion. Gyro is bike-frame rad/s; angles are rad; motor speeds are rad/s.
The established physical signs are unchanged: policy-positive steering is right
from the rider; rear rotation and torque are forward. Both motors' encoder and
torque signs remain -1, gear ratio 1. No simulated noise, bias draw, physical
friction compensation, or observation normalization is added to Pi inference.

Motor state contains continuous signed displacement from each node-start A3
reference. For each RC-enabled episode the policy subtracts the latest rear
displacement, then applies `2*atan2(sin(q/2),cos(q/2))`. This matches the native
branch observed in the saved [Isaac probe](../../../../mujoco_sim2sim/reports/20261022_v1_alignment/isaac_angle_probe.json).
It preserves ±3.5 rad and changes branch near ±2π; it is **not** ±π wrapping or
an indefinitely cumulative simulator observation. The original live-probe
qualification for extreme injected states remains; no new native-engine probe
was performed here.

Steering:

```text
11 observations -> deterministic Beta mean in [-1,1]
-> steering action * 1.57 rad
-> explicit +/-1.57 rad position clip
-> slew <=23.9 * 0.02 = 0.478 rad, once per policy action
-> /trackstand/motor_command [enable, held_position_rad, rear_torque_nm]
-> 200-Hz MIT packets: native_zero + encoder_sign * held_position_rad
   Kp=10, Kd=0.2, requested velocity/feedforward=0
```

Rear:

```text
Beta drive mean in [-1,1] -> *6.28 N m -> +/-6.28 N m torque cap
-> native current = torque_sign * requested_torque / measured_B0
-> signed int32 milliamp C0 packet at 200 Hz
```

Wire encoding is quantized. Current-derived feedback is still measured A1/C0
Q-current times B0, without clipping; it is not an independent shaft-torque
measurement. Passive friction is part of the real assembly and is not subtracted
from or compensated in the outgoing command. Legacy C0/C2 nodes share the new
continuous-state and rear-command contract, and do not apply a second target
slew. C2 firmware's proprietary position loop remains a different actuator.

## Policy artifact and initialization

The packaged self-contained export and its
[manifest](../../../source/trackstand_deploy/models/v1_policy_manifest.json)
record the original checkpoint hash, embedded ONNX hash, distribution and units.
The checkpoint is unchanged. The exporter retains v1's four shape logits and
Softplus+1 alpha/beta transformation, then exports two bounded means. The ONNX
does not bake in physical action scales. Startup validates checksum and the
float32 [batch,11] -> [batch,2] contract; unbounded/non-finite outputs fault.
Historical bundled 41050/91850 files remain unchanged and are rejected by this
loader. An alternate compatible v1 model needs both model_path and model_sha256.

ONNX Runtime uses one intra-op and one inter-op CPU thread. Policy parameters
are startup-only/read-only. IMU and motor subscriptions retain only the latest
sample. Old motor_state_v2 messages cannot refresh v1 readiness. Input freshness
is checked again after inference so slow computation cannot refresh an expired
observation by publishing a newly timestamped command.

RC down resets the episode reference and disables policy. A new episode records
rear zero and seeds the target limiter from measured steering; it does not write
another B1/F0. RC and kickstand movement semantics, CAN IDs, all motor signs,
IMU paths/calibration, CRSF receiver/channel/debounce and servo targets are unchanged.

## Validation

- Deployment suite: **584 passed, 1 skipped**; see [unit_tests.log](unit_tests.log).
  Baseline had 545 passed, 5 failed, 1 skipped: four assertions expected previously
  removed MIT guards/old CF handling, and one expected the removed roll cutoff.
  Tests now assert current v1 behavior and the existing CF/F1 confirmation handshake;
  no removed MIT travel/speed guard was silently restored.
- Actual installed Isaac observation function on **1,000 matched states**:
  maximum channel error **1.66893e-6**, within atol=rtol=2e-6. This includes finite
  speeds above 100 to check that no hidden observation cap remains.
- Actual v1 steering-action class on **1,000 actions with periodic resets**:
  maximum target difference **1.56045e-7 rad**. Actual installed IdealPD equation
  and effort clipping agree exactly for the tested float32 steering/rear tensors.
  See [isaac_parity.json](isaac_parity.json) and [verify_isaac.py](verify_isaac.py).
- Packaged ONNX versus the actual Isaac BetaMean reference on **4,096 inputs**:
  maximum action error **1.04905e-5**; first 1,000 kinematically consistent states
  **1.66893e-6**; 99th percentile **3.57628e-7**. Remaining vectors deliberately
  include impossible Gaussian gravity/sine values. The 2e-5 tolerance accommodates
  float32 batched-PyTorch versus single-row ONNX arithmetic; this is not bitwise
  equality. The matched-state set includes speed stresses beyond motor limits;
  it is a function check, not proof those states are physically achievable.
  See [onnx_parity.json](onnx_parity.json).
- Regression coverage includes disabled startup, model checksum/shape rejection,
  bounded actions, rear RC-cycle rebasing, fixed steering reference, four identical
  MIT payloads per 50-Hz-held target, current units/signs, freshness/disable gates,
  CF/F1 confirmation, torque telemetry, RC and servo behavior.

Reproduce from the repository root:

```bash
PYTHONPATH=deploy/source/trackstand_deploy python3 -m pytest -q deploy/source/trackstand_deploy/test
~/IsaacLab/_isaac_sim/python.sh -B deploy/experiments/reports/20261026_v1_deployment_alignment/verify_isaac.py
python3 deploy/experiments/reports/20261026_v1_deployment_alignment/verify_onnx.py
```

These checks execute CPU task functions and simulated node/transport stand-ins.
No simulator physics, ROS installation/build, CAN interface, Pi test or hardware
actuation was started. A Warp initialization message reports unavailable CUDA;
the CPU function checks complete successfully without a GPU.

## Remaining differences requiring physical evidence

1. **MIT total steering torque cap:** v1 clips total PD effort to ±5.6 N m.
   The local protocol does not establish the corresponding physical MIT cap.
   F0 ranges and C2 B3 current limits cannot be assumed to implement it. No
   speculative firmware configuration was written to conceal this gap.
2. **Dynamics and clocks:** v1 computes PD at 400-Hz physics; deployment sends
   position/gains at 200 Hz for the driver's internal control, whose actual rate
   and current/voltage dynamics remain unidentified. Policy/target slew stay 50 Hz;
   queries/state/status/fault stay 100 Hz. No claim of identical response is made.
3. **Physical constraints:** simulator joint travel/speed constraints, static versus
   moving friction, tyre contact, loaded steering and torque-speed envelope are not
   reproduced by a ROS command cap. MIT still has no measured travel/speed trips;
   legacy backends retain their existing trips. Real friction is not a Pi parameter
   to add again. The software cannot respawn the bike or reset its actual velocity.
4. **Sensors and episodes:** quaternion fusion, noise, source timestamps, CAN/IMU
   asynchrony and Pi scheduling need real recordings. Hardware RC episodes continue
   until RC down or a fault; simulation's 30-s horizon and drift/yaw/wheel-net scoring
   are not automatic hardware resets. No wheel-encoder-only drift estimate is
   substituted for unavailable body translation.

Rebuild/source the complete updated package on the Pi, rather than mixing old
motor state with the new policy. Start physically straight with RC down, verify
the v1/model_2500 startup message and v3 state, then perform supported dry-run and
powered commissioning before interpreting a new real-bike balancing result.
