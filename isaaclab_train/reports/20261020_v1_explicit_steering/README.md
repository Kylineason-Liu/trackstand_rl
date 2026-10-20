# V1 explicit steering PD and measured passive losses

Date: 2026-10-20. Only `Trackstand-Balance-v1` changes from implicit steering
to the installed `IdealPDActuator`. The rear actuator remains IdealPD with
Kp=Kd=0. This supersedes earlier descriptions of v1's steering actuator class,
not their historical measurements.

## Configuration and command contract

Configuration: [trackstand_rl_env_cfg_v1.py](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_cfg_v1.py).

```text
Beta steering action [-1, 1], 50 Hz
  -> scale 1.57 rad
  -> position-target clip [-1.57, 1.57] rad
  -> target slew <=23.9 rad/s, updated once per policy step (<=0.478 rad)
  -> hold target through eight 400-Hz physics steps
  -> each physics step: torque = Kp*(target-position) - Kd*velocity
  -> clip motor torque to [-5.6, 5.6] N m
  -> submit clipped effort to PhysX
```

Nominal controller Kp=10 N m/rad, Kd=0.2 N m s/rad; velocity target and
feedforward torque remain zero. Existing gain DR scales Kp/Kd by U(0.9,1.1)
per environment/reset. Explicit gains are updated before torque computation;
native PhysX stiffness/damping remain zero, so no second controller is added.
The software and solver effort limits are both5.6 N m; velocity limit fields
both23.9 rad/s. IdealPD's torque clamp is speed-independent: its stored
`velocity_limit` does not create an electrical torque-speed envelope.

Passive steering losses, preserved from the immediately preceding update:

- `viscous_friction=0.000910713492` N m s/rad.
- `dynamic_friction=0.046036056104` N m.
- Static `friction=0.046036056104` N m is provisional, not a measured breakaway value.

These come from the [unloaded steering sweep](../../../deploy/experiments/reports/steering_damping_20261020_180044/README.md).
The fitted offset is not applied. Passive losses act separately in PhysX,
outside the motor-generated torque clamp, not as another controller Kd term.
No aggregate external/contact/constraint torque cap is implied.

Rear flow is unchanged: held50-Hz action -> scale6.28 N m -> IdealPD at400 Hz
with Kp=Kd=0 -> clip±6.28 N m -> joint, with passive moving friction0.106 N m
and equal provisional static threshold. Rear solver speed limit remains61.6 rad/s.

## Verification

[34 CPU tests passed](unit_tests.log), covering Beta/observations/export/PPO,
50-Hz clip/slew/reset behavior, both actuator configs, post-PD saturation,
nonzero feedforward/velocity targets in isolated actuator tests, per-environment
gain updates and v0/config-instance isolation.

The real headless environment was checked using [verify_steering.py](verify_steering.py):
nominal and full DR, seed42, each4 environments x32 policy steps. Synthetic
commands plus injected steering velocities exercise saturation and unsaturated
behavior. This is not a learned-policy survival or hardware-performance test.

| Observed quantity | No DR | Full DR |
| --- | ---: | ---: |
| Physics-step steering samples | 1,024 | 1,024 |
| Samples including reset submissions | 1,036 | 1,036 |
| Explicit resets, including initial | 4 | 4 |
| Steering action processing calls | 32 | 32 |
| Maximum absolute pre-clamp torque, N m | 15.70000076 | 16.70362663 |
| Maximum absolute submitted torque, N m | 5.59999990 | 5.59999990 |
| Samples requiring post-PD clipping, including resets | 11 | 14 |
| Controller Kp range | 10.0 | 9.06181622–10.89394474 |
| Controller Kd range | 0.200000003 | 0.182042465–0.219837576 |

Every submitted steering torque equals the clipped actuator effort exactly.
There are eight steering computations but one target-slew update per policy
step. Native steering/rear PD gains are zero after startup, resets and steps.
Native friction readback, rear torque cap and both speed limits pass throughout;
observations remain finite with shape(4,11). Reset submissions are counted
separately from the1,024 physics samples. Completed episodes under synthetic
stress commands are not balance-performance measurements.

Artifacts: [nominal summary](nominal/summary.json), [DR summary](dr/summary.json),
[nominal samples](nominal/steering_samples.npz), [DR samples](dr/steering_samples.npz),
[nominal run log](nominal_run.log), [DR run log](dr_run.log).
NPZ arrays retain target, measured position/speed, actual Kp/Kd, unclipped/clipped
torque, submitted steering/rear torque and native steering gains.

The [prior implicit-friction summary](prior_implicit_friction_summary.json)
is an unchanged copy of the preceding turn's successful nominal/DR readback
probe, originally `/tmp/trackstand_steering_friction.LNbjTn/summary.json`.
It verified the initial passive-loss installation with implicit steering,
4env x32steps/seed42, before this class change; it is not a rerun of that old
configuration. Its nonzero native PD gains are expected historical behavior.

## Scope and reproduction

No v0, shared bicycle asset, observation/action implementation, Beta runner,
reward, termination, DR-event definition, reset logic, physics timing, MuJoCo,
Pi code, motor firmware, checkpoint or verified USD/XML is changed. Explicit
PD integration differs from implicit solver PD; numerical correctness here
does not establish equivalent balancing or identify motor internal-loop timing.
No DCMotor envelope is introduced or calibrated by these friction sweeps.

From `isaaclab_train/`, use fresh output directories:

```bash
~/IsaacLab/_isaac_sim/python.sh -B -m unittest discover \
  -s tests -p 'test_trackstand_*.py' -v

~/IsaacLab/_isaac_sim/python.sh -B \
  reports/20261020_v1_explicit_steering/verify_steering.py \
  --output /tmp/v1_steering_nominal_fresh --headless --device cuda:0

~/IsaacLab/_isaac_sim/python.sh -B \
  reports/20261020_v1_explicit_steering/verify_steering.py \
  --output /tmp/v1_steering_dr_fresh --dr --headless --device cuda:0
```

Existing deployment still explicitly applies rear controller damping0.05 and
retains its separately recorded disabled command clamps; this task does not
synchronize deployment with v1. An approximately zero measured passive B does
not automatically remove a software damping command.
