# Trackstand-Balance-v1 in MuJoCo

This component validates v1's **11-observation, bounded Beta-action contract**
using the verified Trackstand0916 bicycle. It now shares one environment and
policy loader between the viewer and headless evaluator. Historical v0
41050/91850 and NewBike policies are **incompatible**, not silently adapted.

## Quick start

```bash
cd ~/trackstand_rl/mujoco_sim2sim
python3 -m pip install -r requirements.txt
python3 scripts/run_deploy_viewer.py --task deploy_sim --num_envs 64 --seed 42
python3 scripts/run_deploy_eval.py --num_envs 64 --num_steps 3000 --seed 42
```

Both commands default to the repository-relative verified
`models/trackstand_rl_0916_bike/trackstand_rl_0916.xml` and
`../isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_00-41-38/model_75300.pt`.
The checkpoint is not assumed to be distributed by Git: supply
`--pt /path/to/v1_model.pt` if it is absent. No Isaac installation is needed for
MuJoCo/PT inference. Use `--onnx /path/to/v1_export.onnx` with onnxruntime for an
exported **11-input/2-action Beta mean**, including the optional companion
`.onnx.data`. Scale is applied by the environment, not the network loader.

Add `--dr` for full reset randomization; it is off by default, while IMU white
noise remains on. **The default75300 was trained without DR.** Testing it with
`--dr` is an out-of-distribution robustness test, not a DR-trained policy.

Both entry points accept `--physics_hz 800` to run physics and explicit motor
PD at 800 Hz (16 substeps per policy step). The default remains **400 Hz**
(8 substeps). Observations, inference, target slew, rewards and terminations
remain **50 Hz**, with the same 30-second horizon. Select a positive multiple
of 50 Hz; invalid rates are rejected rather than rounded. `--speed` affects
viewer pacing only. Non-default physics/PD rates differ from Isaac v1's
400-Hz reference and are warned at startup and recorded in saved summaries.

For the DR-trained model_2500 at 800 Hz:

```bash
python3 scripts/run_deploy_viewer.py --task deploy_sim \
  --pt ../isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_11-18-29/model_2500.pt \
  --physics_hz 800 --dr --num_envs 64 --seed 42 --pure_survival
```

Remove `--pure_survival` for all six termination conditions. The headless
equivalent uses `scripts/run_deploy_eval.py --physics_hz 800` with the same
policy/DR options and `--num_steps 3000` for a fixed evaluation budget.

## Exact task-side contract

| Item | Current value |
|---|---|
| Policy / observation / slew | 50Hz,20ms |
| Physics / explicit torque calculation | Default 400 Hz, 2.5 ms, 8 substeps; configurable via `--physics_hz` |
| Observation order | gravityYZ, body angularXYZ, steering/rear angle, steering/rear speed, sin/cos steering |
| Observation units | normalized gravity; rad; rad/s; no linear velocity/history/normalization/finite±100 cap |
| Rear-angle adapter | Observed native PhysX4π branch, jumps near±2π; physical MuJoCo hinge remains cumulative |
| Steering action | Beta[-1,1]→×1.57rad→clip±1.57→slew23.9rad/s once/20ms→hold |
| Steering motor | Every physics step: clip(10*(target−q)−.2*qdot,±5.6Nm) |
| Rear motor | clip(6.28*BetaAction,±6.28Nm); controllerKp=Kd0 |
| Separate passive steering losses | frictionloss.046036056104Nm; damping.000910713492Nms/rad |
| Separate passive rear losses | frictionloss.106Nm; damping0 |
| Nominal reset | root(0,0,−.016)m; identity quaternion; all joint angles/speeds0; fresh slew/action/yaw/wheel state |
| Standard horizon | 1500policy steps=30s |
| Standard terminations | timeout; abs(gy)>sin(.5236) or inversion; velocity divergence; drift>.5m; relativeyaw>.5236rad; previous wheel-net>7.2rad |
| Reward | .02*(5*alive+2*exp(−drift²/(2*.25²))−.1*norm(raw−previousRaw)) |

Steering target-velocity limiting is distinct from a measured speed constraint.
Passive losses act in mechanical dynamics **outside** the motor command clamp.
Equal static/moving joint friction remains provisional, as in v1. Neither
platform implements an identified torque–speed/thermal/electrical envelope.

All six termination flags are evaluated independently; simultaneous activations
are retained. The wheel-net counter deliberately reproduces v1's one-step-lag
endpoint-speed integration. The caller resets completed environments; failure
and timeout both clear episode state. Only one noisy observation is sampled
for each new policy input, including automatic resets.

With randomized initial roll-rate, the ROOT-COM linear velocity is zeroed, as
in Isaac. The free-joint origin gets the corresponding `−omega×COM_offset`
linear velocity. The divergence check likewise uses COM, not origin, speed.

## Domain randomization

Per environment/episode: each physical body's mass×U(.8,1.2), inertia scaled by
the same factor; one shared Isaac-local CoM XYZ offset±.005m, converted into each
CAD body's coordinate frame; steeringKp/Kd×independentU(.9,1.1); roll±.1rad,
roll-rate±.25rad/s; gyro residualSD.0002rad/s and gravity residualSD.002.
RearKp/Kd stay zero. Ground/tire nominal pair friction is.65.

A **shared32-bucket material bank** is sampled at group initialization; each
collider draws a bucket at reset. Static/dynamic draws are both U(.5,.8), with
dynamic=min(static,dynamic). See the limitation below: MuJoCo's applied contact
coefficient is the sampled static value; both draws are retained in runtime
`env.dr`, not saved as a per-episode parameter history in these captures.
No draws compound; derived model constants are refreshed after mass/CoM edits.
NumPy and Torch random streams are not bitwise-identical across engines.

## Important remaining engine differences

This is **not an exact PhysX emulator**, nor a promise that75300 balances here:

- PhysX TGS articulation/contact constraints and SDF contacts differ from
  MuJoCo's solver and existing mesh collisions. The verified MJCF/USD geometry,
  masses, inertias and joint axes are not retuned to hide that difference.
- PhysX's23.9/61.6rad/s solver speed limits can overshoot. MuJoCo has no matching
  native joint-velocity-limit attribute. Default `--velocity_limit_mode project`
  retains the earlier hard per-step velocity projection, explicitly **stricter**
  and non-physical as an exact PhysX replacement. `none` omits that guard for
  diagnostics; it is **also not equivalent**. The choice is warned and recorded.
- MuJoCo contact friction has one sliding coefficient; PhysX randomizes separate
  static/dynamic coefficients. Nominal.65/.65 matches the numerical setting,
  but DR's static→dynamic transition is not exactly reproduced. No uncalibrated
  switching threshold or substitute electrical model is invented.
- Native passive joint friction and integration are solver-dependent even when
  coefficients match. Extreme manually injected rear angles are not certified
  by the ordinary continuous-rotation branch probe.
- Engine rigid-body defaults are not fully emulated: the live USD root has
  angularDamping.05, linearDamping0 and maxDepenetrationVelocity1. The child
  links have no authored PhysxRigidBodyAPI. Their resolved native damping and
  the exact TGS recovery/drag implementation were not established by this
  audit, so no guessed drag is added to motor Kd or passive joint friction.

See the [complete comparison and validation report](reports/20261022_v1_alignment/README.md).
Interface/conformance tests and learned-policy performance are separate results.

## Options and outputs

Viewer: `--num_envs`(64), `--seed`(42), `--mjcf`/`--scene`, `--pt`/`--onnx`,
`--dr`, `--physics_hz`(400), `--pure_survival`, `--velocity_limit_mode project|none`, `--speed`.
`--task zero_agent` disables motor torque; `free_fall` also removes passive
losses and auto-reset. `--lock_steer` is zero_agent-only. These demonstrations
are not standard v1 evaluations. `--base_inertia` is retained as a warned,
non-v1 diagnostic override; keep1 for ordinary comparisons.

Headless: same model/policy/DR/termination options, plus `--num_steps`(3000 per
environment), `--episodes N`(alternative completed-episode budget),
`--steps`(episode horizon1500), `--output_dir`(logs). A non-default horizon is a
diagnostic, not the standard v1 task. All bikes are independent CPU model/data
pairs, with seeds seed+i; the viewer grid offsets are rendering-only.

Reports share `YYYYMMDDHHMM_<model>_<dr|no_dr>`; existing names receive a suffix:

- `_epi_len_histogram.png/.csv`: ten3s bins[0,3),…,[27,30], completed episodes only.
- `_torque.png/.npz`: actuator effort, **not** incoming joint wrench/net acceleration torque.
- `_summary.log/.json`: completed-episode returns, durations, step-weighted
  motion statistics and overlapping termination counts.
- Headless also saves `_evaluation.npz`: actual11 inputs, bounded actions,
  held steering targets, post-step motor q/qdot, all configured substep actuator
  torques, rewards, terminations and reset events. Motor order is steering/rear.
  `_metadata.json` records invocation, timing, policy hash, limitations and episodes.

No runtime model augmentation writes XML into models/. Closing the viewer
saves available outputs. A headless run with no completed episode writes
null aggregate statistics and an empty histogram, not a fabricated survival.

## Layout and tests

`src/newbike_mujoco/` retains its historical package name:
`contract.py`(audited constants), `model.py`(read-only runtime augmentation),
`env.py`(task), `policy.py`(Beta inference), `viewer_envs.py`(independent bikes),
`reporting.py`(shared reports). Diagnostics in other viewer scripts intentionally
remain separate from the policy task. Old policy/model artifacts are preserved.

```bash
python3 -B -m unittest discover -s tests -v
~/IsaacLab/_isaac_sim/python.sh -B -m unittest discover -s tests -v
```

The second command additionally executes actual Isaac v1 observation, reward,
termination, resolved-config and Beta-mean parity tests. It does not launch a
simulator. GPU angle probes and headless MuJoCo evaluations are separate,
documented experiments. The previous component README is preserved as
[historical NewBike/v0 notes](docs/LEGACY_NEWBIKE.md); it is not today's contract.
