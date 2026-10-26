# Trackstand-Balance-v1 → MuJoCo audit

Baseline: current on-disk v1, not v0/41050/91850. The table was established
before the runtime changes. Implementation/validation details are in README.md.
No verified MJCF/USD geometry, inertia tensor, mesh or joint axis is retuned.

| Aspect | Isaac v1 standard | MuJoCo before this change | Required correction |
|---|---|---|---|
| Task/default model | Trackstand v1 / Beta / 75300 no-DR reference | NewBike names, model_500; stale absolute 41050 path for trackstand | v1 defaults and relocatable paths |
| Observation size/order | 11: gravity YZ, body omega XYZ, steering/rear q, steering/rear qdot, sin/cos steering | 13 with privileged linear XY velocity | Remove XY; validate policy shape |
| Units/signs/frames | normalized gravity; rad; rad/s; root link axes | Same math inside common angle branch | Keep axes, test matched states |
| Rear angle | Native PhysX coordinate; NOT explicit ±π | Explicit ±π | Probe native boundary, then align observed branch |
| Finite observation values | No ±100 clamp | No finite clamp | Retain no clamp |
| Nonfinite repair | NaN/±Inf→0 before manager noise | None | Repair before white noise |
| White noise | gravity .001, gyro .0012 at 50 Hz, even no-DR | Same first five stds | Preserve, extend explicit 11-channel tuple |
| Episode IMU bias | gyro SD .0002; gravity SD .002; only DR | Same intended | Preserve and verify reset order |
| History/latency/normalization | None | None on ONNX; PT silently discarded normalizer | Strict checkpoint handling |
| Actor | 4 shape logits → softplus+1 → 2 Beta means in [-1,1] | PT returned four logits as if actions | Implement deterministic Beta mean, reject Gaussian artifacts |
| Policy / physics | 50 / 400 Hz, exactly 8 steps | Model-dependent rounded substep count | Enforce .0025/.02 and decimation8 |
| Steering scale/target clamp | 1.57; ±1.57 rad | 1.57; ±1.5708 | Exact 1.57 |
| Steering slew | 23.9 rad/s once per 20 ms, reset from fresh q | Same rate, hardcoded period, zero reset | Contract dt and reset priming |
| Steering torque | Explicit sample PD at each 400-Hz boundary, Kp10/Kd.2, then ±5.6 | Native affine servo inside integrator | Explicit PD→torque-only actuator |
| Rear torque | Pure feedforward×6.28, Kp=Kd0, post-calculation clip±6.28 | Additional −.05*qdot | Remove software damping |
| Passive steering losses | friction .046036056104; viscous .000910713492 | Missing | Native frictionloss + joint damping, outside motor cap |
| Passive rear losses | friction .106; viscous resolves0 | Missing / controller damping confused with plant | Native frictionloss .106; damping0 |
| Static breakaway | Equal moving friction, explicitly provisional | Not represented | Same provisional threshold via frictionloss |
| Physical travel | USD steering ±90°; rear continuous | Export steering ±1.570796; rear continuous | Preserve verified model; target clamp is separate |
| Speed constraints | PhysX solver23.9/61.6; can overshoot | Hard qvel projection at each physics step | Unavoidable backend mismatch: expose/label approximation |
| Effort/speed envelope | Constant torque cap; NO DC/thermal model | Constant torque cap, no electrical model | Do not invent an envelope |
| Spawn | root(0,0,-.016), identity, all joint q/qdot0 | z−.0156 | Exact nominal root state |
| Initial DR | roll±.1; world roll rate±.25, zero translation/pitch/yaw | Same intended | Preserve |
| Initial linear velocity | ROOT-COM world velocity0, even with initial roll-rate | Free-joint ORIGIN velocity0 | Convert v_origin=−omega×(COM−origin) at reset |
| Horizon | 1500 policy steps /30s | Same in env; stale CLI --steps unused | Honor evaluator horizon |
| Terminations | All six terms; simultaneous flags retained | Only timeout/fallen, short-circuit | Restore six; retain overlapping flags |
| Fallen | abs(gy)>sin(.5236) or gz>0 | sin(exact30°) | Exact configured threshold |
| Diverged | Nonfinite or ROOT-COM norm v>20, omega>50, any joint speed>300 | Removed | Restore velocity-only definition and COM semantics, not an invented pose guard |
| Drift | XY distance from env origin>.5m | Removed | Restore |
| Yaw | wrapped deviation from current episode heading>.5236 | Removed | Restore |
| Wheel net | previous policy-step endpoint-speed accumulator>7.2, one-step lag | Removed | Restore exact ordering, not raw rear q |
| Reset bookkeeping | physical reset on failure OR timeout; yaw/wheel/actions/slew cleared | Partial bookkeeping only | Clear all per-episode state on both |
| Rewards | dt*(5*alive +2*exp(-d²/(2*.25²)) -.1*norm(raw−previous_raw)) | Not calculated | Add exact three-term reward and return diagnostics |
| Mass DR | independent physical-body×U(.8,1.2); inertia same factor | Included massless synthetic wrapper | Only massive robot bodies; refresh MuJoCo constants |
| CoM DR | one shared LOCAL-body offset ±.005 per axis, no inertia shift | Same raw offset applied to rotated CAD frames; missing derived-constant refresh | Convert offset per body frame, call mj_setConst |
| Material DR | shared32 buckets, independent static/dynamic, dynamic≤static; per shape reset draw | Separate bucket bank per env; only static used | Shared bank; retain both sampled coefficients; document native friction limitation |
| Gain DR | independent Kp/Kd multipliers(.9,1.1), defaults each reset | Native servo parameters; rear .05 damping randomized | Explicit steering gains; rear zeros stay zero |
| DR reset | defaults→draw, never compound; off by default | Mostly same; cached model constants stale | Restore baseline, update derived model constants |
| Self collisions | Disabled on articulation | Nonadjacent robot collisions possible | Exclude robot-body pairs in runtime augmentation |
| Root rigid-body defaults | Live USD: angularDamping .05, linearDamping0, maxDepenetrationVelocity1; child APIs absent | No equivalent explicitly configured | Engine-level residual, not an excuse to add .05 to motor Kd |
| Contact/solver | PhysX TGS, SDF wheel collisions, 4/2 articulation iterations | MuJoCo contact solver/mesh collision | Engine difference, NOT geometry defect; preserve models |
| Force telemetry | Submitted actuator effort ≠ projected incoming joint wrench | Called all forces “actual torque” | Label actuator torque; separate passive/constraint forces |
| Numerical precision / RNG | PhysX/Torch float32 and Torch RNG | MuJoCo double dynamics / NumPy RNG | Toleranced matched-state tests, not claimed bitwise trajectories |
| Terminal observation draw | Auto-reset, then one noisy next-episode input | Terminal draw followed by another reset draw | Suppress unused terminal draw in auto-reset callers |
| Viewer/evaluator | Same task/contract | Different model/policy/reset paths | Shared runtime loader/policy contract |
| Output integrity | Preserve old experiment evidence | Loader wrote hidden XML into models/; torque x-axis restarted each episode | In-memory augmentation; cumulative trace axis; non-overwriting reports |

## Primary source map

- `isaaclab_train/.../trackstand_bike/trackstand_rl_env_cfg_v1.py` plus inherited
  `trackstand_rl_env_cfg.py` / `bicycle_cfg.py`.
- `.../mdp/stationary/{observations,noise,actions,rewards,terminations,events}.py`.
- `.../trackstand_bike/trackstand_rl_env.py`: exact wheel integration/reset order.
- `.../agents/beta_distribution.py` and `scripts/deployment/export_rsl_rl_onnx.py`.
- Installed Isaac `actuators/actuator_pd.py`, `envs/manager_based_rl_env.py`,
  `envs/mdp/events.py`: IdealPD clipping and actual DR sampling semantics.
- Root progress_record.md §§94–107, 139–158 and progress_record.txt: historical
  wrapping correction, measured friction, current actuator and evaluation evidence.
- Official MuJoCo [joint attributes](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint),
  [friction loss](https://mujoco.readthedocs.io/en/stable/computation.html#friction-loss),
  [simulation loop](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#simulation-loop).

This is contract alignment, not a claim that different rigid-body engines will
produce identical trajectories. A v0 checkpoint is not a valid v1 policy.
