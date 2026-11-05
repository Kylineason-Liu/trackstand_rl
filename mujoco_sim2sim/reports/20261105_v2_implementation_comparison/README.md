# Current Isaac Lab / MuJoCo v2 implementation comparison

Source audit on 2026-11-05, performed by one agent as requested. Compared the
active `Trackstand-Balance-v2` inheritance chain, task MDP terms, USD layers,
installed Isaac Lab/PhysX implementation, the independent
`newbike_mujoco_v2` package, MJCF, policy loaders and standard evaluation scripts.
MuJoCo 3.12.0 was used to compile/inspect the current model and call
`mj_forward` at its reference pose. No physics rollout, training, inference,
GPU/ROS/CAN or hardware experiment was run. No subagent was used.

## Matched default contract

- Eleven observations, their order and additive noise distributions; deterministic
  Beta mean inference in evaluation; no observation normalization.
- Policy/held targets 50 Hz, physics/explicit PD 1200 Hz, rear I 200 Hz.
- Steering position action scale/bound 1.57 rad, target slew 23.9 rad/s,
  explicit PD 10/0.2 and torque cap 5.6 Nm.
- Rear velocity scale/bound 61.6 rad/s, P 0.30, Ki 3.3 Nm/rad, trapezoidal
  integration, first-sample priming, conditional anti-windup, retained target-change
  memory, episode reset, combined P+I effort cap 6.28 Nm.
- Passive-loss numerical coefficients, fixed controller gains with/without DR,
  nominal gravity/spawn settings, six termination thresholds and 30-s horizon.
- Default three reward equations/weights and 20-ms integration. Controller
  integral/held steering-target memory remains absent from observations in both.

These matches describe default equations/parameters. They do not establish
identical physics trajectories, random draws or evaluation statistics.

## Substantive differences

| Area | Isaac Lab v2 | MuJoCo v2 |
| --- | --- | --- |
| Framework and execution | Manager/action/event architecture; normally GPU-vectorized physics and policy | Independent model/data per bike; CPU MuJoCo and Python per-bike stepping |
| Numeric precision | PhysX/Warp/Torch state and PI buffers use float32 | MuJoCo state and NumPy/Python PI arithmetic use float64; policy inputs/actions remain float32 |
| Solver/integration | PhysX TGS, articulation position/velocity iterations 4/2, external forces every solver position iteration | Euler integrator, Newton solver, 100 iterations and 50 line-search iterations in the current compiled model |
| Wheel collision geometry | Analytic cylinders, radius 0.07 m and height 0.001 m | CAD `collision00.stl` mesh geoms, including rim/hub geometry; not the same primitive colliders |
| Inertial assets | USD-authored masses, principal moments and principal axes | CAD MJCF full inertias compiled into principal frames; small numeric moment differences remain, in addition to rounding/frame changes |
| Contact response | PhysX contact/friction-patch machinery, max depenetration velocity 1 m/s | Explicit `solref=[0.005,1.5]`, `solimp=[0.9,0.95,0.001,0.5,2]`, default pyramidal cone/3-D contacts and zero geom margin; these are not a TGS parameter mapping |
| Contact friction | Separate static/dynamic material coefficients; ground/tire MIN combine | One sliding coefficient; tire priority selects the pair value. DR dynamic-friction draw is logged but not applied |
| Passive joint loss implementation | PhysX static, dynamic and viscous joint-friction properties | MuJoCo `frictionloss` constraint plus `dof_damping`; equal configured magnitudes do not equate the solvers |
| Joint speed limits | Native soft solver constraints | Default `project` directly clips post-step steering/rear qvel to 23.9/61.6 rad/s; diagnostic `none` removes that approximation |
| Actuator plumbing | `IdealPDActuator` and external action-term I feedforward; native PhysX PD zeroed | NumPy computes/clips torque and writes unit-gear torque actuators; no native velocity servo. Default equations are matched |
| Native coordinates | xyzw quaternion, world velocities converted to actor axes, native rear-angle 4pi branch | wxyz quaternion and native free-joint body angular velocity; cumulative hinge angle adapted to the 4pi observation. Wheel-net uses cumulative increments rather than branch unwrapping |
| Random-number streams | Torch CPU/GPU material/reset/noise sampling, vectorized across environments | NumPy generator per bike (`seed+i`) plus a shared material bank. Same seed/ranges do not produce matched individual initial states, DR or noise |
| Configuration | Inherited task config, manager terms and Hydra/CLI overrides | Duplicated constants and hard-coded default reward/termination equations. Policy metadata validates selected rear-PI fields; it does not load/apply the entire saved environment config |
| Policy support/admission | PPO training can sample Beta actions; evaluation uses runner mean inference | Inference only, CPU Torch or ONNX Runtime. Reconstructs supported Linear/ELU actor; rejects incompatible metadata/normalizers and nonfinite/out-of-range policy outputs |
| Environment/reset API | Gym-style vector API, separate terminated/truncated flags, automatic selected-env reset inside step | Raw API returns `(obs,done)`, keeps terminal state and exposes reward/flags via attributes; evaluator/group caller resets individual bikes |
| Evaluation sampling | Standard `play_eval.py` runs a fixed number of policy steps for every respawning stream | Also supports exact balanced per-environment completion quotas; quota-complete streams stop stepping; budget shortfall is reported |
| Terminal metrics | Standard evaluator reads pose/speeds after step has auto-reset done environments; terminal extrema/turn increments can be omitted | Per-episode statistics are collected before caller reset and include terminal state |
| Metric populations | State averages/maxima/RMS include all evaluated steps, including unfinished episodes | Summary state statistics are aggregated only over completed episodes; unfinished trajectories are saved separately |
| Trace/output scope | Standard evaluator samples final-substep applied torque at policy rate for three bikes and overwrites checkpoint-adjacent eval/torque files | Saves all-env physics-substep controller/torque/I traces and metadata, with unique run prefixes under `logs/v2` by default |
| Default checkpoint resolution | Explicit checkpoint or runner's latest matching run/checkpoint | Fixed no-DR `model_26600.pt` default unless explicitly overridden; presence of the 50100 export does not change that default |

The shared API and parameter choices should not be confused with exact engine
equivalence. This audit identifies implementation differences; it does not assign
their contributions to the previously measured policy-transfer gap.

## Fresh static asset measurements

The saved `compiled_model_comparison.json` captures the inspected MuJoCo
options, collision mesh envelopes, transformed CoMs and principal inertia values.
Principal eigenvalues can be compared without conflating CAD-local versus
USD-local body axes. Maximum absolute relative principal-moment difference
`100 * (MuJoCo / USD - 1)` per corresponding link:

| Link | Maximum magnitude |
| --- | ---: |
| Base | 0.0111253% |
| Rear wheel | 0.2078655% |
| Steering | 0.5467670% |
| Front wheel | 0.0508447% |

These compare compiled MJCF moments with authored USD values, not a fresh live
PhysX measurement. The mesh-envelope widths in JSON include hub geometry and
must not be presented as measured tire contact-patch width. Native coordinate
adapters and reset COM/origin-velocity correction are already implemented;
their different storage conventions are not, by themselves, observation bugs.

## Primary sources

- [Isaac v2 config](../../../isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_cfg_v2.py), inherited v1/v0 config and bicycle config in the same task tree.
- [Isaac PI](../../../isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/velocity_pi.py), observation/reward/noise/termination/numerics terms and v2 environment.
- [USD wheel colliders](../../../isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/assets/payloads/base.usda) and [authored inertias](../../../isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/assets/payloads/Physics/physics.usda).
- [MuJoCo contract](../../src/newbike_mujoco_v2/contract.py), [environment](../../src/newbike_mujoco_v2/env.py), [model loader](../../src/newbike_mujoco_v2/model.py), controller, policy/contract loader, group and reporting modules.
- [MJCF](../../models/trackstand_rl_0916_bike/trackstand_rl_0916.xml).
- [Isaac evaluator](../../../isaaclab_train/scripts/rsl_rl/play_eval.py) versus [MuJoCo evaluator](../../scripts/run_deploy_eval_v2.py).
- Installed Isaac Lab sources under `/home/lmbike/IsaacLab/source/`: PhysX manager cfg/articulation, manager-based environment, MDP events/terminations/actions and IdealPDActuator.

`source_sha256.json` binds current comparison inputs. Temporary recording backups
and checks live under `/tmp/trackstand_isaac_mujoco_v2_20261105_lf_86ver/`.
