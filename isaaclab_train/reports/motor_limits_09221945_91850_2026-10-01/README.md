# Motor-limit audit: 09221945 / model_91850

Evaluated on 2026-10-01. One completed evaluation of the repository's unchanged
`scripts/rsl_rl/play_eval.py` used 64 environments, 3,000 policy steps, seed 42,
full domain randomization (`--dr`), 50 Hz control, and 400 Hz physics. This gives
192,000 policy samples and 1,536,000 physics samples per motor. These are empirical
frequencies for this run, not probabilities guaranteed for other seeds or hardware.

## Reproduction and integrity

From `isaaclab_train/`:

```bash
/home/lmbike/IsaacLab/_isaac_sim/python.sh \
  reports/motor_limits_09221945_91850_2026-10-01/evaluate_motor_limits.py \
  --task Trackstand-Balance-v0 \
  --checkpoint reports/motor_limits_09221945_91850_2026-10-01/model_91850.pt \
  --num_envs 64 --num_steps 3000 --seed 42 --dr --headless --device cuda:0 \
  agent.max_iterations=500000
```

The instrumentation executes the original evaluator and observes its existing
action, scene-write, and scene-update calls. It records motor state before and
after each physics step, before automatic episode resets. No extra simulation
steps or policy clipping are introduced. `configuration_audit.json` reports no
unexpected differences from the saved `params/env.yaml` and `agent.yaml`.
Allowed differences are environment count, output/checkpoint paths, and the
new explicit DR-enable configuration field; relocated USD/noise paths are normalized.
`source_integrity.json` verifies 638 existing source/configuration/model files
were unchanged. The original checkpoint was copied to this directory so the
evaluator's fixed output filenames could not overwrite previous evaluations.

Two recorder preflight attempts were stopped before a full evaluation; their
logs are retained separately. `run.log` is the single completed 3,000-step run.

## Limits and control laws in Isaac Lab

| Item | Steering | Rear drive |
| --- | --- | --- |
| Action meaning | Position target, raw action times 1.57 rad | Feed-forward effort target, raw action times 6.28 N m |
| Raw action clipping | None: runner `clip_actions: null`, action `clip: null` | None: same settings |
| Position range | Live joint bounds +/-1.5707960129 rad, approximately +/-90 degrees | Continuous; no finite joint-position limit |
| Configured joint velocity limit | +/-23.9 rad/s | +/-61.6 rad/s |
| Configured actuator effort limit / estimated-torque clip | +/-5.6 N m | +/-6.28 N m |
| Target slew limiter | None in Isaac action processing | No torque slew limiter in the Isaac action term |
| Nominal actuator law | 10*(target-position) - 0.2*velocity | feed-forward target - 0.05*velocity |

Full DR scales stiffness and damping independently by U(0.9,1.1). This run
observed steering stiffness 9.0009--10.9951 N m/rad, steering damping
0.18046--0.21972 N m s/rad, and rear damping 0.04530--0.05494 N m s/rad.
Rear stiffness remained zero. Position, velocity, and effort bounds were not randomized.

There are also episode containment limits involving motors: any joint speed
above 300 rad/s or non-finite state causes `diverged`; accumulated rear-wheel
net rotation above +/-7.2 rad causes `wheel_net`. These reset episodes, rather
than clipping commands. The rear position observation wrapping to +/-pi is
an observation representation, not a physical rotation stop. The USD's
100 N m authored steering-drive value and 100 rad/s authored steering velocity
value are superseded by the live actuator values listed above.

These are the training configuration's motor bounds. The bicycle config labels
5.6/6.28 N m as stall values and 23.9/61.6 rad/s as 228/588 RPM. This audit does
not establish the real bike's measured torque-speed envelope, physical steering
stops, firmware current limits, or thermal/continuous ratings.

## Measured exceedances

An exceedance uses a 1e-5 tolerance in the quantity's native units. Target rows
use 192,000 policy samples; torque and speed rows use 1,536,000 physics samples.

| Quantity | Count / samples | Fraction | Largest absolute value |
| --- | ---: | ---: | ---: |
| Steering position target outside joint range | 1,265 / 192,000 | 0.65885% | 237.68182 degrees |
| Rear feed-forward torque target above 6.28 N m | 171 / 192,000 | 0.08906% | 132.00243 N m |
| Steering computed torque estimate above 5.6 N m | 4,335 / 1,536,000 | 0.28223% | 48.27404 N m |
| Rear computed torque estimate above 6.28 N m | 1,368 / 1,536,000 | 0.08906% | 132.00243 N m |
| Steering clipped applied-torque estimate above 5.6 N m | 0 / 1,536,000 | 0% | 5.6 N m |
| Rear clipped applied-torque estimate above 6.28 N m | 0 / 1,536,000 | 0% | 6.28 N m |
| Steering speed above 23.9 rad/s | 671 / 1,536,000 | 0.04368% | 24.27422 rad/s |
| Rear speed above 61.6 rad/s | 608 / 1,536,000 | 0.03958% | 92.51144 rad/s |

All 64 environments had at least one excessive target for each motor.
Steering actual position exceeded the mathematical joint bound in 284 physics
samples, with a maximum of 90.00555 degrees: a small solver overshoot, unlike
the 237.68-degree requested target. Speed constraints also permit transient
overshoot. At 50 Hz the original evaluator reports maxima 24.09/65.47 rad/s;
capturing all 400 Hz substeps exposes larger intervening peaks.

There were 171 observed episode beginnings: the initial 64 and 107 resets
followed by another policy action. **Every one of these 171 first actions
exceeded both the steering position range and rear effort bound.** Every
excessive rear target occurred on an episode's first policy step. Rear requests
at these starts ranged from 63.88702 to 132.00243 N m; after the first step,
the maximum absolute rear request was only 2.76509 N m. Steering also had 1,094
excessive targets after the first episode step. These observations establish
when the bursts occur; they do not diagnose why the policy produces them.

Steering target changes exceeded 23.9 rad/s in 1,385/191,829 within-episode
transitions (0.72200%), peaking at 236.44310 rad/s. This is the rate required
to follow target changes exactly, not a velocity command: a position controller
may lag rather than reach every target within 20 ms.

## Torque interpretation and feasibility

The recorded torque estimates exactly reconstruct the PD/damping laws and their
clipping: maximum reconstruction and clipping errors were both zero at stored
float32 precision. However, `ImplicitActuator.compute()` explicitly computes
**approximate** torques and returns the original position/velocity/effort action
unchanged. The library then submits the feed-forward effort buffer to PhysX.

Experimentally, the rear feed-forward effort buffer equaled the unbounded
scaled policy output at all captured substeps: maximum difference zero. It
exceeded 6.28 N m on 1,368 substeps and reached 132.00243 N m, even while the
diagnostic `applied_torque` estimate was clipped to 6.28 N m. Consequently,
the standard evaluator's torque plot is not evidence that the actual implicit
drive's total solver torque is capped at 6.28 N m. This experiment did not
measure that total solver torque or physical motor current/torque.

For the feasibility question:

- A requested steering position outside an assumed +/-90-degree physical
  steering range cannot be attained as specified. The simulation may still
  apply an attainable saturated correction toward that target. The simulated
  +/-90-degree joint range does not prove the real mechanism has identical stops.
- Under a genuine +/-6.28 N m maximum total torque, a demand beyond it cannot
  be delivered unchanged. In this dataset all 171 excessive rear requests
  remained above that bound after damping, on all eight substeps; damping did
  not make any of those requests fit the bound.
- Exceeding a speed-based target-change rate means exact immediate tracking
  is infeasible under that speed bound; it does not mean the in-range position
  is unreachable eventually.
- A simulation using ideal implicit actuators cannot prove a statement of
  "100% physically impossible" on the real motors. That requires verified
  steering travel and motor torque/current/speed ratings at the actual voltage
  and load. No measured motor torque-speed curve is supplied by this evaluation.

At the time of this evaluation, deployment clipped raw actions to +/-1, capped steering targets at
1.57 rad via the action scale and rear targets at 6.28 N m, slews steering
targets at 23.9 rad/s in its 200 Hz loop, and explicitly clips final calculated
torque before converting it to C0 current. MuJoCo also clamps steering targets
and clips actuator control/force. Those pipelines therefore handle the observed
out-of-range requests differently from Isaac's unbounded action targets.

Later on 2026-10-01, deployment's raw-action clipping was removed at the user's
request. Its separate +/-90-degree steering-target clamp and motor-I/O
feed-forward/final torque limits remain. This subsequent deployment change does
not alter the recorded Isaac evaluation data or statistics above.

## Evaluation result and artifacts

The completed run reproduces the previous seed-42 evaluation: 129 completed
episodes; mean length 1,369.0 steps; timeout 58.9%; yaw termination 41.1%; actual
fall termination 0%; roll RMS 1.43 degrees. The evaluator's `fall_rate=0.411`
means all non-timeout terminations, not physical falls.

- [Target and torque distributions](limit_distributions.png): all recorded samples.
- [Full motor traces](motor_traces.png): environments 9 and 44, selected for the largest absolute steering/rear target.
- [Startup detail](startup_detail.png): the first 0.6 seconds of those same traces.
- [Original evaluator torque plot](torque_profile.png): three environments, clipped torque estimates at 50 Hz.
- [Raw motor samples](motor_limit_samples.npz): every policy target and physics substep; motor order `[steering, drive]`.
- [Statistics](motor_limit_statistics.json), [exceedance CSV](exceedances.csv), [evaluation summary](eval.txt).
- [Configuration audit](configuration_audit.json), [source integrity](source_integrity.json), [execution log](run.log).

Source evidence: task action and actuator configurations in
`source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/`, saved run
`params/env.yaml` and `params/agent.yaml`, local Isaac Lab `joint_actions.py`,
`actuator_pd.py:ImplicitActuator.compute`, and PhysX articulation
`write_data_to_sim` (which submits the original effort target buffer).
