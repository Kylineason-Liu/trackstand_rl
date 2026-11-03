# Trackstand-Balance-v2 in MuJoCo

This is a separate implementation of the current IsaacLab v2 policy contract.
The existing `newbike_mujoco/` package, v1 scripts, configuration, models and
policies remain unchanged. The new runtime imports only `newbike_mujoco_v2/`.

## Files and startup

- `src/newbike_mujoco_v2/`: independent environment, PI controller, model loader,
  Beta inference, viewer support and reporting.
- `scripts/run_deploy_viewer_v2.py`: interactive, respawning multi-bike viewer.
- `scripts/run_deploy_eval_v2.py`: headless evaluation and full controller traces.
- `tests/test_v2_contract.py`: controller, timing, reset and task-contract checks.

Use the same Python environment as the existing MuJoCo implementation. If needed,
install `python3 -m pip install -r requirements_v2.txt` from this directory.

```bash
cd ~/trackstand_rl/mujoco_sim2sim

# Viewer; nominal/no DR, 64 bikes by default.
python3 -B scripts/run_deploy_viewer_v2.py --num_envs 64 --seed 42

# Headless; one completed episode per environment, all six terminations active.
python3 -B scripts/run_deploy_eval_v2.py \
  --num_envs 64 --episodes 64 --seed 42
```

Both default to v2 **model_26600.pt**, from the no-DR run
`isaaclab_train/logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_21-49-32/`, and the
verified `models/trackstand_rl_0916_bike/trackstand_rl_0916.xml` geometry.
The model is configured in memory; source XML and meshes are never rewritten.

Common flags: `--pt`, `--onnx`, `--policy_config`, `--mjcf`, `--num_envs`, `--seed`,
`--dr`, `--physics_hz`, `--pure_survival`, `--velocity_limit_mode`, `--output_dir`.
Use `--help` for the complete lists. `--dr` is opt-in and never randomizes either
motor's controller gains. Ordinary per-observation IMU noise remains enabled
without DR. `--pure_survival` is a diagnostic that disables four task terminations.

Physics/PD defaults to **1200 Hz**, policy and target updates to **50 Hz**, and
the rear integral to **200 Hz**. `--physics_hz` must be a positive multiple of
200 to preserve both slower schedules; changing it is a numerical diagnostic.

## Motor command flow

The eleven observations and bounded deterministic Beta means use v2 ordering.
No additional normalization or action squashing is introduced at inference.

**Steering:** raw action × 1.57 → position clip ±1.57 rad → target slew limited
to 23.9 rad/s, updated once per 20-ms policy interval. The held target drives
`10 * (target_position - position) - 0.2 * velocity` at every physics step,
clipped to ±5.6 N·m. Passive friction remains separate.

**Rear drive:** raw action × 61.6 → velocity clip ±61.6 rad/s → hold for 20 ms.
Let `e = target_velocity - measured_joint_velocity`, in rad/s. At each physics
step the explicit controller sends:

```text
rear_torque = clip(0.30 * e + I, -6.28, +6.28) N·m
```

`I` updates every 5 ms using the trapezoidal error integral with
`Ki = 3.3 N·m/rad`. The first update after reset only records the previous error.
I is limited to ±6.28 N·m; conditional integration rejects changes that push
the total P+I farther into saturation while permitting unwinding. Target changes
retain I and error history. Episode reset clears I, previous error, held motor
targets and wheel-angle references. There are four I ticks and 24 physics/PD
steps per policy interval at the default rate.

MuJoCo's rear actuator is a unit-gear torque actuator receiving this computed
PI torque. Its native actuator gain is 1, with no hidden velocity controller;
position Kp is zero. Rear joint passive damping is 0 and `frictionloss` is
0.106 N·m. Passive losses act in the solver; they are not subtracted from the
command again. The torque limit applies **after P+I**, before passive losses.

## Policy compatibility

V1 and v2 have the same eleven inputs and two Beta actions, but v1's rear action
means torque. Shape checks alone cannot distinguish them. The new loader verifies
adjacent `params/env.yaml` and rejects the v1 torque-action configuration.
For a relocated checkpoint/export, pass the original v2 saved YAML explicitly:

```bash
python3 -B scripts/run_deploy_viewer_v2.py \
  --onnx /path/to/v2_policy.onnx \
  --policy_config ../isaaclab_train/logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_21-49-32/params/env.yaml
```

Alternatively, place `<policy-stem>.contract.json` beside the policy with keys
`task: "Trackstand-Balance-v2"`, `drive_action: "velocity"`, `observations` equal
to the ordered `contract.OBSERVATIONS`, `drive_velocity_scale_rad_s: 61.6`,
`drive_velocity_p_nm_s_per_rad: 0.30`, `drive_integral_gain_nm_per_rad: 3.3`,
`drive_integral_rate_hz: 200`, and `drive_effort_limit_nm: 6.28`.
Optional `policy_sha256` binds the metadata to the actual file. A user-supplied
sidecar declares provenance; it cannot independently prove an export's origin.
PT inference uses CPU Torch; ONNX uses ONNX Runtime's CPU provider.

## Evaluation, outputs and limits

Headless default mode runs 3000 policy steps per environment. `--episodes N`
instead assigns fixed per-environment quotas, differing by at most one episode;
completed streams stop stepping. `--max_policy_steps` bounds the global loop.
A quota/budget shortfall saves partial results and returns exit code 2. Completed
episodes alone enter the histogram and summary; unfinished lengths and active
masks are retained separately. Individual overlapping termination flags are kept.

Default outputs live in **`logs/v2/`**, with prefix
`YYYYMMDDHHMM_<model>_<dr|no_dr>` and a collision suffix when needed. They include
the ten-bin episode histogram, torque graph/raw data, summary JSON/log and, for
headless runs, metadata and an evaluation NPZ. That archive retains raw actions,
held targets, observations, actual positions/speeds, physics-rate pre-clip and
applied torques, rear I/error samples and reset events. Motor arrays use
`[steering, rear]`; torque-plot arrays explicitly use `[drive, steering]`.
Inactive quota streams are NaN-masked and identified by `active_mask`.
Each CLI isolates native `MUJOCO_LOG.TXT` in a new `<prefix>_runtime/` directory
and restores the caller's working directory on exit. Resolve relative input paths
from the directory where the command is launched.

Wheel-net accounting uses actual physics-step rear-angle displacement before
termination checks. The rear observation separately reproduces the installed
PhysX 4π position branch. V2 invalid-state containment and finite terminal drift
reward are included. All motor/angle/controller state resets per episode.

Matching this interface does not make the physics engines identical. MuJoCo
`frictionloss` approximates equal static/moving joint friction; contact has one
sliding coefficient. Default `--velocity_limit_mode project` clips joint speed
at 23.9/61.6 rad/s and is stricter than PhysX's soft solver constraint; `none`
omits that approximation. Contact solvers/integration and torque-speed/thermal
dynamics remain distinct. No real-bike command or firmware loop is simulated.

Validation evidence is under
[`reports/20261103_v2_parallel_backend/`](reports/20261103_v2_parallel_backend/README.md).
