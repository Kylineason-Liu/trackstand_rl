# V1 steering constraints and rear damping comparison

Historical snapshot: the subsequent [v1 IdealPD rear-actuator change](../20261019_v1_explicit_drive/README.md)
adds the post-damping torque clamp. The bicycle results below describe the
preceding implicit-rear-actuator configuration, not that later change.

Date: 2026-10-19. Task: `Trackstand-Balance-v1`. No v0, bicycle geometry,
MuJoCo or deployment configuration was changed. Rear damping remains **0.05**
by default; the zero-damping bicycle run used a temporary command-line override.

## Implemented steering flow

```text
Beta steering action in [-1, 1]
  -> multiply by 1.57 rad (default steering offset is zero)
  -> explicitly clip requested position to [-1.57, 1.57] rad
  -> limit change from the previous target to +/-0.478 rad per policy step
  -> hold this target for all eight 400-Hz physics steps
  -> existing steering ImplicitActuator / joint constraints
```

The slew state advances in `process_actions()`, **once at 50 Hz**, not in
`apply_actions()` at 400 Hz. The discrete target-step bound is
`23.9 rad/s * 0.02 s = 0.478 rad`. This bounds changes between held position
requests; it does not guarantee measured steering speed or target tracking.
The existing actuator remains Kp=10, Kd=0.2, effort limit 5.6 N m and joint
velocity limit 23.9 rad/s.

Each environment's reset initializes its target from its current, post-reset
steering position, clipped to the target range. Partial resets leave other
environments' slew state unchanged. The original action is preserved in
`raw_actions`; clipping is on the scaled position, before slew limiting.

Implementation: [action term](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/actions.py)
and [v1 configuration](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_cfg_v1.py).
Drive actions, rewards, resets, terminations, DR and physics timing remain inherited.

## Steering verification

- **22 unit tests passed**: 14 Beta/configuration tests and 8 steering tests.
  Tests cover scale/clip/slew ordering, reversals, eight identical physics
  applications, partial/full/empty resets and invalid configuration.
- The actual GPU environment also passed a synthetic-action check:
  **4 environments x 32 policy steps**, seed 42, no DR. Every target written
  in all eight physics applications matched its held processed target exactly.
- Maximum absolute target: **1.570000052 rad**; maximum target change per
  policy update: **0.478000045 rad** (float32 rounding).
- The headless check intentionally includes raw requests outside [-1,1] to
  exercise clipping. Those are test inputs, not Beta policy outputs.

Artifacts: [unit test log](unit_tests.log), [integration summary](steering_integration/summary.json),
[per-physics-application samples](steering_integration/steering_samples.npz).

## What rear damping zero changes

With zero desired velocity and zero stiffness, the intended unsaturated
implicit-controller equation is:

```text
Kd=0.05: torque approximately equals 6.28 * drive_action - 0.05 * wheel_speed
Kd=0.00: torque approximately equals 6.28 * drive_action
```

`damping` is controller Kd here, not the separately configured passive
`viscous_friction`. Setting Kd=0 does **not** disable feedforward torque control.
It removes active speed-proportional braking: 1 N m at 20 rad/s, or 3.08 N m
at 61.6 rad/s. Contact, passive friction, the wheel's inertia and the configured
joint velocity constraint are not removed. More coasting, stronger same-direction
acceleration and weaker opposite-direction braking are expected at nonzero speed.
At rest there is no damping contribution to remove.

### Controlled actuator experiment

[Probe source](probe_drive_damping.py) constructs **synthetic fixed-base,
unit-inertia rotors (J=1 kg m^2)**, not a bicycle or an identified real motor.
Seven scenarios are run with both Kd values: 14 rotors, dt=0.0025 s, 5 s,
seed 42, PhysX/CUDA. Gravity, contacts, passive friction and rigid-body damping
are absent. Effort limit is 6.28 N m; the speed limit is deliberately raised
to 10000 rad/s so it cannot confound damping measurements.

| Condition | Kd=0.05 | Kd=0.0 |
| --- | ---: | ---: |
| Zero torque, initial speed 10 rad/s: speed after 5 s | 7.7882 rad/s | 10.0000 rad/s |
| +6.28 N m from rest: speed after 5 s | 27.7825 rad/s | 31.3797 rad/s |
| +0.50 N m, initial speed 10 rad/s: speed after 5 s | 10.0000 rad/s | 12.5024 rad/s |
| -6.28 N m, initial speed +61.6 rad/s: first measured net joint effort | -9.3594 N m | -6.2800 N m |

The last row confirms an important existing distinction: this implicit
configuration does **not** apply an explicit +/-6.28 N m clamp to feedforward
plus damping. The native effort sensor was used, cross-checked against
`J * delta_speed / dt`; this is not the actuator's clipped diagnostic
`applied_effort` estimate. An actual bicycle at its speed constraint or in
contact can also experience constraint/contact forces; this isolated test
does not establish a universal bound on those forces.

All trajectories agree with `J * speed_dot = feedforward - Kd * speed`
within **0.0885 rad/s** over 5 s. Effort versus acceleration discrepancy is
at most **0.0243 N m**. The probe explicitly saves validation tolerances of
0.1 rad/s and 0.05 N m; these are numerical test tolerances, not claims about
real-motor accuracy. An initial stricter 0.03-rad/s absolute tolerance was not
met in the high-speed zero-damping braking case; it was not silently treated
as a pass. The retained validated rerun records actual errors and pass status.

Artifacts: [graph](rotor_validated/drive_damping_comparison.png),
[all scenario results](rotor_validated/summary.json),
[conditions and validation](rotor_validated/metadata.json),
[full samples](rotor_validated/rotor_samples.npz).

### Bicycle runtime smoke comparison, not a trained-policy benchmark

The only clearly identified v1 checkpoint available for this check is
`model_2.pt` after **three PPO iterations**. The older trained 41050/91850
policies have a different Gaussian/13-observation contract and are not v1
checkpoints. Identical copies of the v1 smoke checkpoint were evaluated with
Kd=0.05 and Kd=0: **32 environments x 128 policy steps**, seed 42, no DR,
normal observation noise and standard terminations.

Checkpoint SHA-256:
`ad08a89530d2640931b9c6334b9318f51444ef554f58a909c970c6078f260de8`.

| Smoke result | Kd=0.05 | Kd=0.0 |
| --- | ---: | ---: |
| Completed episodes | 96 | 96 |
| Mean episode length | 0.80 s | 0.80 s |
| Fallen / timeout | 100% / 0% | 100% / 0% |
| Rear speed, mean absolute / maximum | 0.79 / 1.10 rad/s | 1.52 / 2.69 rad/s |
| Mean / maximum position drift | 0.018 / 0.046 m | 0.031 / 0.088 m |

Sources: [Kd=0.05 summary](smoke_kd_005/eval.txt), [Kd=0 summary](smoke_kd_000/eval.txt).
Both runs prove runtime compatibility and continued torque actuation, not
that either damping is better for a trained controller. A proper learning
comparison needs separate v1 training runs with matched seeds/budgets and
held-out evaluations. No trained v1 balance-performance conclusion is supported.

## Recommendation and cross-platform implications

If the intended rear motor command is **pure requested torque**, Kd=0 is a
coherent controller choice. If 0.05 was intended to model passive motor drag,
identify the real passive friction before configuring a plant-friction value;
do not claim 0.05 is measured or automatically transfer it to viscous friction.
If deliberate virtual damping is desired, retaining Kd is reasonable, but an
explicit post-damping torque clamp is still a separate design task.

MuJoCo currently uses `DRIVE_KV=0.05`; deployment uses
`DRIVE_DAMPING_NM_S_PER_RAD=0.05`. Setting only Isaac Kd to zero would create a
new control mismatch. These files were not modified. Train/evaluate the final
selected contract consistently across platforms before unsupported deployment.

## Reproduction

Run from `isaaclab_train/` using the Isaac Lab Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/IsaacLab/_isaac_sim/python.sh \
  -m unittest discover -s tests -p 'test_trackstand_*.py' -v

PYTHONPATH="$PWD" ~/IsaacLab/_isaac_sim/python.sh \
  reports/20261019_v1_steering_drive_damping/verify_steering.py \
  --headless --device cuda:0 --output /tmp/v1_steering_check_fresh

~/IsaacLab/_isaac_sim/python.sh \
  reports/20261019_v1_steering_drive_damping/probe_drive_damping.py \
  --headless --device cuda:0 --seed 42 --output /tmp/v1_rotor_check_fresh
```

Both probe output directories must not already exist. The bicycle smoke command
below was run twice with separate checkpoint copies/output directories. Set
`SMOKE_DIR` to a fresh directory containing the matching v1 `model_2.pt`:

```bash
PYTHONPATH="$PWD" ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/play_eval.py \
  --task Trackstand-Balance-v1 --checkpoint "$SMOKE_DIR/model_2.pt" \
  --num_envs 32 --num_steps 128 --seed 42 --headless --device cuda:0 \
  agent.experiment_name="$SMOKE_DIR"
```

Append `env.scene.robot.actuators.drive.damping=0.0` for the zero-damping run.
No source edit is required for this override. Full captured logs are retained
beside each summary. SDK warnings about the unactuated third joint refer to
the passive front wheel; the two commanded joints are steering and rear drive.
