# V1 explicit rear drive torque clamp

Date: 2026-10-19. This supersedes the **rear actuator configuration** described
in the earlier [steering/damping report](../20261019_v1_steering_drive_damping/README.md).
It does not change that report's historical measurements.

## Change and contract

Only `Trackstand-Balance-v1` replaces its rear `ImplicitActuator` with the
installed built-in `IdealPDActuator`. Configuration is in
[`TrackstandRLEnvCfgV1.__post_init__()`](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_cfg_v1.py).

```text
Beta drive action [-1, 1], updated at 50 Hz
  -> JointEffortAction scale 6.28 N m, held between policy updates
  -> IdealPDActuator at 400 Hz:
       unclipped = feedforward - Kd * current_rear_speed
       submitted = clip(unclipped, -6.28, 6.28) N m
  -> simulation effort buffer
```

Settings: stiffness=0, nominal damping **Kd=0.05 N m s/rad**, target velocity=0,
explicit `effort_limit=6.28`, existing `effort_limit_sim=6.28` and
`velocity_limit_sim=61.6` rad/s. The framework zeros the native rear-drive PD
gains for the explicit actuator, so damping is not applied twice. The clamp
affects the submitted torque, not merely diagnostic `applied_effort`.

**0.05 remains an unmeasured placeholder for controller Kd.** No passive-friction
coefficient was identified or configured. A future measurement of passive
losses must be distinguished from a choice of active controller damping.

The v0 configuration, steering actuator, 50-Hz steering clip/slew, Beta actor,
observations, rewards, terminations, resets, timing and DR event definitions
were not changed. Existing gain DR correctly updates the explicit actuator's
Kd; nominal 0.05 is scaled by U(0.9,1.1) when DR is enabled.

## Verification

**29 CPU tests passed:** existing Beta/observation/PPO/export tests, steering
tests, and seven new drive tests. New tests cover saturation after damping,
zero-action damping, zero-Kd torque control, per-environment gain updates and
configuration isolation. The v0/v1 comparison checks that every other
configuration field remains equal after excluding their intended differences.

The real GPU environment was checked twice using [verify_drive.py](verify_drive.py):
nominal and full DR, **4 environments x 64 policy steps**, seed 42, 400-Hz
physics, 50-Hz policy. Eight explicit resets exercise gain randomization.
Synthetic commands and velocity writes intentionally force saturation; this
is not a policy-balance benchmark or hardware friction measurement.

| Observed quantity | No DR | Full DR |
| --- | ---: | ---: |
| Physics-step rear samples | 2,048 | 2,048 |
| Total submissions, including reset writes | 2,080 | 2,080 |
| Maximum absolute pre-clamp torque | 7.280000 N m | 8.162170 N m |
| Maximum absolute submitted torque | 6.28000021 N m | 6.28000021 N m |
| Submissions requiring post-damping clipping | 16 | 19 |
| Explicit Kd range | 0.05000000 | 0.04533492–0.05417423 |

Every submitted effort matched the explicit clipped effort **exactly**. All
eight calculations per policy step were observed. Native rear stiffness and
damping were zero after startup and each explicit reset; the rear solver speed
limit remained 61.6 rad/s. The tiny excess over decimal 6.28 is float32
representation, not a controller-limit violation.

Artifacts: [unit test log](unit_tests.log),
[nominal summary](nominal/summary.json), [DR summary](dr/summary.json),
[nominal samples](nominal/drive_samples.npz), [DR samples](dr/drive_samples.npz).
Logs are retained beside each summary. The NPZs include feedforward torque,
speed, actual Kd, pre-clamp torque, clipped/submitted torque and native gains.

This verifies the **motor effort command** cap. Contact, passive-friction and
joint-constraint forces are not themselves clamped by the actuator. No claim
about trained-policy survival or real-bike performance follows from this test.
MuJoCo and deployment were not modified.

## Reproduction

From `isaaclab_train/`, using the installed Isaac Lab Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/IsaacLab/_isaac_sim/python.sh \
  -m unittest discover -s tests -p 'test_trackstand_*.py' -v

PYTHONPATH="$PWD" ~/IsaacLab/_isaac_sim/python.sh \
  reports/20261019_v1_explicit_drive/verify_drive.py \
  --headless --device cuda:0 --output /tmp/v1_idealpd_nominal_fresh

PYTHONPATH="$PWD" ~/IsaacLab/_isaac_sim/python.sh \
  reports/20261019_v1_explicit_drive/verify_drive.py \
  --headless --device cuda:0 --dr --output /tmp/v1_idealpd_dr_fresh
```

Probe output directories must not already exist. The synthetic probe has no
checkpoint dependency. SDK warnings about two actuators for three joints refer
to the intentionally passive front wheel.
