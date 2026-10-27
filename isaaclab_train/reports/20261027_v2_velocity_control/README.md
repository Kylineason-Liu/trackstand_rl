# Trackstand-Balance-v2: rear MIT velocity control

V2 is a new Isaac task derived from v1. The only plant/action change is rear
velocity control instead of feedforward torque. V2 also disables controller-gain
randomization for both motors, including with --dr. The existing v0/v1 task files,
registrations and resolved configurations remain unchanged; deployment and
MuJoCo remain on their existing v1 contract.

## Control contract

| Term | V1 | V2 |
| --- | --- | --- |
| Raw actor outputs | Two bounded Beta actions in [-1,1] | Same |
| Observations | 11 channels, no body linear velocity | Same function/noise/order |
| Steering action | ×1.57 rad, ±1.57 clip, 0.478 rad/50-Hz-step slew | Same |
| Rear action | ×6.28 N m torque | ×61.6 rad/s target velocity; explicit ±61.6 target clip |
| Rear controller | IdealPD Kp=0, Kd=0, scaled action as feedforward | IdealPD Kp=0, nominal Kd=0.2, feedforward zero |
| Rear torque cap | ±6.28 N m | Same, after velocity-error calculation |
| Rear speed limit | 61.6 rad/s solver constraint | Same |
| Rear passive losses | Moving friction 0.106 N m, provisional equal static threshold, viscous 0 | Same |
| Timing | 50-Hz policy/observation, 400-Hz physics/explicit PD | Same |
| Rewards, terminations, reset bookkeeping | V1 | Inherited unchanged |
| DR | V1 | Same except controller-gain randomization is disabled |

The nominal rear command path is:

```text
Beta raw drive action in [-1,1]
  -> scale by 61.6 rad/s
  -> clip velocity target to [-61.6, 61.6] rad/s
  -> hold for eight 400-Hz physics steps
  -> compute 0.2 * (target_velocity - measured_velocity)
  -> clip motor torque to [-6.28, 6.28] N m
  -> submit motor effort; PhysX applies passive losses separately
```

This is the requested **MIT velocity equation**, with Kp=0 and feedforward zero,
implemented through Isaac's explicit IdealPDActuator. Position does not contribute
to rear torque. There is no C1 firmware PI controller or added integral term.
Kd has units N m s/rad and is controller feedback, not passive viscous friction.
Zero drive action requests zero speed and produces braking torque when moving.
At nominal Kd, speed error above 31.4 rad/s requires effort clipping. The measured
moving-drag estimate implies about 0.53 rad/s steady error in a simple unsaturated
moving P-loop model; this is not measured MIT tracking accuracy or breakaway speed.

The scale is based on the existing v1 speed constraint and the saved individual
[GIM3510-8 datasheet](<../../../tsinghua_bike_project/assign_2_Hardwares/assign_2_Hardwares/3.RearSteer Servo【GIM3510-8】/GIM3510-8_motor_datasheet.png>),
which lists 588 output rpm (61.5752 rad/s) and 6.28 N m stall torque. The catalog
PDF also on disk gives different GIM3510-8 numbers (564.6 rpm/6.02 N m); it was
not substituted for the individual sheet that matches the established v1 values.
This target scale does not model the physical torque-speed envelope or establish
speed/current authority under every load.

With or without --dr, steering Kp/Kd remain 10/0.2 and rear Kp/Kd remain 0/0.2.
V2 removes the inherited randomize_actuator_gains event after every DR switch,
including constructor and CLI switches. Other DR terms and their sampling are
unchanged. Fixed gains do not establish perfect physical tracking accuracy.

## Implementation files

- New task config: [trackstand_rl_env_cfg_v2.py](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_cfg_v2.py).
  It inherits v1, overrides the rear action with JointVelocityActionCfg, and
  replaces the rear actuator's controller damping with 0.2, and overrides the DR
  switch to disable gain randomization for both motors.
- New environment entry point: [trackstand_rl_env_v2.py](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_v2.py).
  It inherits v1's failure/timeout bookkeeping without overriding behavior.
- New runner: [rsl_rl_ppo_beta_v2_cfg.py](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/agents/rsl_rl_ppo_beta_v2_cfg.py).
  It retains all Beta/PPO parameters and uses a separate trackstand_rl_v2_rsl log
  namespace to distinguish incompatible rear-action checkpoints.
- Shared registration only: [task __init__.py](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/__init__.py)
  gains an independent v2 entry; the v0/v1 registration objects are unchanged.
- New tests: [test_trackstand_velocity_v2.py](../../tests/test_trackstand_velocity_v2.py).
  Existing tests, source functions, v0/v1 files, hardware code and models were not edited.
- The training README gains an appended v2 section; its prior content is preserved.

## Original validation results (before gain DR was disabled)

The results below and the original no_dr/ and dr/ artifacts are preserved historical
evidence from the initial implementation. The current verify_v2.py instead requires
fixed gains for both motors in every mode; use a fresh output directory to rerun it.

**13 new CPU tests passed**, using actual installed Isaac action/actuator classes.
They check complete v1/v2 configuration equality after reverting only the rear
action/controller difference; instance isolation; Beta/PPO configuration equality
apart from log name; velocity scaling/clip/offset; eight held target applications;
post-Kd torque clipping, zero-command braking, unchanged passive losses and gains.

Headless GPU functional evaluation ran the real task with seed 42, 16 environments,
both nominal and DR. Each mode executed 64 synthetic action steps, 32 steps for
two diagnostic PPO updates, and one forced failure/timeout reset step. No trained
v2 balance policy was evaluated or saved.

| Recorded quantity | No DR | DR |
| --- | ---: | ---: |
| Policy/observation/action-processing calls | 97 | 97 |
| Rear controller/effort-write calls | 776 | 776 |
| Scalar rear actuator samples | 12,416 | 12,416 |
| Sampled rear Kd | 0.200000003 | 0.180121–0.219383 |
| Maximum target speed | 61.599998 rad/s | 61.599998 rad/s |
| Maximum measured speed | 61.600033 rad/s | 63.547943 rad/s |
| Maximum computed torque before clip | 20.659220 N m | 25.684324 N m |
| Maximum submitted torque | 6.2800002 N m | 6.2800002 N m |
| Samples requiring torque clipping | 345 | 1,733 |

Every actual submitted rear torque equals the clipped velocity-error calculation
exactly in the recorded float32 buffers. Feedforward is always zero. Native PhysX
PD gains are zero for both motors, preventing a second controller. Joint friction
and configured speed constraints remain unchanged. Forced timeout/failure resets
restore joint position/speed, wheel-net bookkeeping and steering target as checked.
Both diagnostic PPO runs produced finite losses/weights and bounded Beta actions.

The DR speed overshoot is real recorded simulation output: a solver speed limit
is not a strict post-step velocity projection. No extra hard state clamp was added.
The high computed torques prove the effort cap is active; they do not imply those
unclipped demands reached the joint. These synthetic/untrained runs establish the
implementation's data flow, not successful balance or a physical motor response.

Files: [nominal summary](no_dr/summary.json), [DR summary](dr/summary.json), each
mode's capture.npz, [unit log](v2_unit_tests.log), [full suite log](unit_tests.log),
[baseline test log](baseline_beta_tests.log), and [preservation comparison](v0_v1_configuration_comparison.json).

The full legacy suite ran 56 test methods and reported two failing DR/no-DR
subcases of the pre-existing v0/v1 comparison test. It incorrectly treats v0's
viscous_friction=None and v1's explicit 0.0 as identical. The same two subcase
failures reproduce with the original pre-v2 registry in an isolated baseline.
That test and both task definitions were left untouched. These are not new v2
failures and are not represented here as a fully passing legacy suite.

## Fixed controller gains validation (2026-10-27)

After disabling v2 gain DR, all **13 updated v2 CPU tests passed**. They cover
both constructor modes and repeated nominal/DR switches through the real CLI
helper. Complete configuration comparisons establish that only the intended
rear action/controller differences and the missing gain-DR event differ from v1.
V0/v1 gain DR remains enabled when requested; 71 other Python files were unchanged.

The current headless audit passed with **seed 42, 16 environments, --dr**:
97 policy steps, 776 physics/controller updates, and 12,416 rear actuator samples,
including two diagnostic PPO updates and forced timeout/failure resets. Steering
Kp/Kd remained 10/0.2 and rear Kp/Kd remained 0/0.2 across resets. The runtime
reset-event list retained mass, COM, contact material, initial-state and IMU-bias
randomization, while excluding randomize_actuator_gains.

Every submitted rear effort still equaled the clipped velocity-error PD exactly;
maximum absolute submitted effort was 6.2800002 N m (float32). The synthetic run
recorded a maximum rear speed of 71.135254 rad/s despite the unchanged 61.6 rad/s
soft solver speed constraint. This is a functional check, not a balance benchmark
or proof of physical tracking precision. No trained checkpoint was changed.

Current artifacts: [summary](fixed_gains_dr/summary.json),
[capture](fixed_gains_dr/capture.npz), [headless log](fixed_gains_dr/headless.log),
[unit-test log](fixed_gains_dr/unit_tests.log), and
[preservation checks](fixed_gains_dr/verification.json).

## Reproduce and train

From the repository root, CPU tests:

```bash
~/IsaacLab/_isaac_sim/python.sh -B -m unittest discover \
  -s isaaclab_train/tests -p test_trackstand_velocity_v2.py -v
```

The saved [verify_v2.py](verify_v2.py) command is in each mode's summary.json.
It requires GPU access and a new --output directory; add --dr for randomized
validation. Its two PPO updates are diagnostics, not a full training campaign.

For new training from isaaclab_train/:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Trackstand-Balance-v2 --num_envs 4096 --seed 42 --headless --dr
```

Omit --dr for no DR. Checkpoints go to logs/rsl_rl/trackstand_rl_v2_rsl/.
Do not resume v1 torque-action checkpoints or feed a v2 policy to the existing
v1 MuJoCo/Pi consumers simply because tensor sizes match. A future hardware
velocity backend must send MIT target velocity in rad/s with Kp=0, nominal
Kd=0.2 and feedforward zero, using verified rear signs, F0 velocity encoding
range and total torque/current protection. None of those rear MIT hardware
properties was identified by this Isaac implementation, and no CAN was sent.
