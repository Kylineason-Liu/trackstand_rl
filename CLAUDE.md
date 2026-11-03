# CLAUDE.md

Guidance for AI coding agents working in this repository.

## What this is

One bicycle robot, **two policy families**, each with its own Isaac task, MuJoCo
check and Pi ROS 2 nodes:

- **Trackstand** (stationary balance): keep the bike upright with minimal planar
  drift. Active contract **`Trackstand-Balance-v1`** (11 obs, Beta policy,
  RSL-RL). Outputs steering **position** and rear-wheel **torque**.
- **Locomotive** (riding): ride at a commanded speed and heading. CycleRL's
  cylinder-wheel paper recipe, ported onto this repo's plant as
  **`Locomotive-Cylinder-Aligned-v0`** (8 obs, Gaussian policy, **skrl**).
  Outputs steering **position** and rear-wheel **velocity**.

Branch **`locomotive`** carries both stacks; `main` is trackstand-only. The two
share the bike USD, robot cfg, hardware params and some deploy nodes (IMU, RC,
kickstand), but their tensor contracts, trainers, motor backends, topics and
launch files are separate. **Never run trackstand and locomotive nodes together.**

Pipeline (both): **Isaac Lab train → ONNX export → MuJoCo sim-to-sim → Raspberry Pi ROS 2**.
This is a research codebase, not a certified autonomous or road-ready system.

CycleRL as a separate package was removed on 2026-09-23; ignore stale `CycleRL` /
`NewBike-Balance` paths in old docs and script headers. The locomotive stack is a
deliberate, self-contained port of CycleRL's riding recipe under
`tasks/manager_based/locomotive/` (MDP files copied verbatim, no CycleRL imports).
The original CycleRL repo lives at `~/bike_test/CycleRL` (read-only reference).

### At a glance

| | Trackstand (v1) | Locomotive (Aligned-v0) |
|---|---|---|
| Isaac task | `Trackstand-Balance-v1` | `Locomotive-Cylinder-Aligned-v0` |
| Package | `tasks/manager_based/trackstand/` | `tasks/manager_based/locomotive/` |
| Trainer | RSL-RL (`scripts/rsl_rl/train_deterministic.py`) | skrl 2.1 PPO (`scripts/skrl/train_locomotive.py`) |
| Network | MLP [32,32], Beta (mean deployed), no obs norm | [512,256,128,64] ELU, Gaussian mean, obs+value `RunningStandardScaler` (±5σ, embedded in ONNX) |
| Obs / actions | 11 / 2 | 8 / 2 |
| Steering action | ×1.57 rad → slew 23.9 rad/s → position target | ×20° (clip ±20°) → position target |
| Rear action | ×6.28 N·m torque | ×42.857 rad/s (clip ±45) velocity target, τ = clip(0.2·(ω*−ω), ±6.28) |
| Rates | 400 Hz physics / 50 Hz policy | same |
| Episode | 30 s (1500 steps) | 64 s (3200 steps) |
| DR | Off by default; opt in with `--dr` | Always on (CycleRL riding DR in the cfg; no flag) |
| Logs | `logs/rsl_rl/<experiment>/<timestamp>/` | `logs/skrl/locomotive/<task>/<timestamp>_ppo_torch_locomotive_ppo/` |
| MuJoCo | `mujoco_sim2sim/scripts/run_deploy_{viewer,eval}.py` | `mujoco_sim2sim/scripts/locomotive/eval_paper_suite.py` |
| Pi nodes | `policy_controller` + `motor_mit_io` | `locomotive_controller` + `motor_mit_velocity_io` |
| Launch | `trackstand.launch.py` | `locomotive.launch.py` (`params.yaml` + `config/locomotive.yaml`; records each run) |
| Deploy doc | `deploy/README.md`, `deploy/docs/MOTOR_MIT_IO.md` | `deploy/docs/LOCOMOTIVE.md` |

Both launch files are **live** (`dry_run: false`).

---

## Trackstand

### Task versions

| Task | Status | Obs | Policy dist | Rear action | Runner / log dir |
|------|--------|-----|-------------|-------------|------------------|
| `Trackstand-Balance-v0` | Historical | 13 | Gaussian | torque ×6.28 N·m | `trackstand_rl_rsl` |
| `Trackstand-Balance-v1` | **Active everywhere** | 11 | Beta (mean deployed) | torque ×6.28 N·m | `trackstand_rl_rsl` |
| `Trackstand-Balance-v2` | Experimental, Isaac only | 11 | Beta | velocity ×61.6 rad/s | `trackstand_rl_v2_rsl` |

- **v1** = v0 plant + Beta runner, 11 obs (no body linear velocity), steering
  target clipped ±1.57 rad and slewed at 23.9 rad/s (one step per 50 Hz action),
  explicit `IdealPDActuator` steering (Kp 10 / Kd 0.2, clipped ±5.6 N·m) and pure
  feedforward rear torque (clipped ±6.28 N·m), measured passive friction
  (steer 0.046 N·m + viscous 0.00091; rear 0.106 N·m), fixed timeout bookkeeping.
- **v2** = v1 with rear velocity target and MIT-equivalent law
  `clip(0.2·(v_target − v), ±6.28)` (Kp 0, no FF); actuator-gain DR is always
  disabled; wheel-net tracked at physics rate. Has an unresolved **400 Hz rear
  oscillation**; no Pi backend or trained/qualified policy. A PI velocity loop is
  only a proposal. (The locomotive port uses this same rear law.)

Note: the v0 cfg docstrings still say "THE single env config / no other variants"
— that is stale.

### Control contract (v1; shared across Isaac / MuJoCo / deploy)

| Item | Value |
|------|-------|
| Observation dim | **11** |
| Action dim | **2** raw in [−1, 1] (Beta mean `2α/(α+β)−1`); scaled in env/deploy, not in ONNX |
| Steering | ×**1.57 rad** → clip ±1.57 → slew 0.478 rad per 20 ms → position target |
| Drive | ×**6.28 N·m** effort, clipped ±6.28 |
| Policy / control rate | **50 Hz** (`sim.dt=0.0025`, `decimation=8` → 400 Hz physics) |
| Episode | **30 s** → 1500 steps |
| Obs noise | Always on (gravity 0.001, gyro 0.0012); independent of DR |
| Obs normalization | Off in RSL-RL |

Observation order (`mdp/stationary/observations.py::balance_imu_state_v1` /
`deploy/.../contracts.py::OBSERVATION_NAMES`):

```text
[gravity_y, gravity_z,
 ang_vel_x, ang_vel_y, ang_vel_z,
 steering_pos, drive_pos,
 steering_vel, drive_vel,
 sin(steering_pos), cos(steering_pos)]
```

On the bike: IMU supplies gravity + gyro; encoders supply joint channels. Rear
position is rebased per episode and mapped to the PhysX 4π branch
(`physx_revolute_angle`); steering keeps its fixed straight zero. Historical
13-input exports (41050 / 91850, v0 with `lin_vel_x/y`) are **rejected** by
MuJoCo and deploy.

### Rewards and terminations

- `survival` **5.0** (binary while |roll| < 30°), `drift_gauss` **2.0**
  (Gaussian σ = 0.25 m), `rate_penalty` **0.1** (v1: −‖Δa‖ on raw actions).
- Terminations (all versions): `time_out`, `fallen` (|roll| > 30° or inverted),
  `diverged`, `drifted` (> 0.5 m), `yaw_deviation` (> 30°), `wheel_net`
  (> 7.2 rad accumulated rear angle). MuJoCo v1 viewer/evaluator enable all six by
  default; `--pure_survival` restricts to `time_out` + `fallen`.

### Domain randomization

- **Default off** (`domain_randomization=False`). Opt in with **`--dr` only**
  (no `--no_dr`, no ramp). **All new trackstand training uses `--dr`.**
- With `--dr`: mass ×U(0.8,1.2), COM ±5 mm, tire friction U(0.5,0.8) (nominal
  pair 0.65 via min-combine), actuator gains ×U(0.9,1.1) (not in v2), initial
  roll/rate U(±0.10)/U(±0.25), IMU turn-on residuals.
- Match DR setting across train / eval / MuJoCo when comparing. Check saved
  `params/env.yaml` — a checkpoint path does not restore historical cfg.

### Reference v1 checkpoints

| Run | DR | Model | Use |
|-----|----|-------|-----|
| `2026-10-21_00-41-38` | off | `model_75300.pt` | MuJoCo default `--pt` (`newbike_mujoco/contract.py`) |
| `2026-10-21_11-18-29` | on | `model_2500.pt` | **Pi default** ONNX |

### Commands (from `isaaclab_train/`)

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_deterministic.py \
  --task Trackstand-Balance-v1 --num_envs 4096 \
  --max_iterations 60000 --seed 42 --headless --dr

~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_eval.py \
  --task Trackstand-Balance-v1 --checkpoint /abs/path/model.pt \
  --num_envs 64 --num_steps 3000 --seed 42 --headless --dr      # match --dr to the run

~/IsaacLab/isaaclab.sh -p scripts/deployment/export_rsl_rl_onnx.py \
  --checkpoint /abs/path/model.pt \
  --output scripts/deployment/trackstand_policy.onnx           # raw actions; opset 18
```

Last checkpoint ≠ best. `play_eval.py` overwrites `eval.txt` / torque profiles
beside the checkpoint.

---

## Locomotive

Package `isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/locomotive/`:
`locomotive_env_cfg.py` (task), `locomotive_env_cfg_ablation.py` (port-ablation
variants), `agents/skrl_ppo_cfg.yaml`, `mdp/` (verbatim CycleRL MDP copies;
`eval_events.py` is eval-only), `eval/` (ported CycleRL eval stack,
`paper_metrics.py`), `config/` (terrain presets/generators, used only by eval
terrain modes).

### Task: `Locomotive-Cylinder-Aligned-v0`

CycleRL's rank-1 cylinder recipe (`1900_drive03`) baked in as defaults, on
trackstand's robot cfg (`trackstand/bicycle_cfg.py`, same USD/joints, solver 4/2,
tire 0.65/min) at 400 Hz physics. Other CycleRL top-5 recipes are Hydra `env.*`
overrides (see `scripts/skrl/train_locomotive.py` header).

| Item | Value |
|------|-------|
| Observations (8) | `[gravity_y, ang_vel_x, ang_vel_y, ang_vel_z, speed (body vx), steer, vel_cmd, yaw_rate_cmd]`; per-reset IMU noise fraction ρ ~ U(0.01, 0.20) |
| Actions (2) | steering ×20° clip ±20° → position; rear ×(3/0.07) rad/s clip ±45 → velocity |
| Steering actuator | v1 IdealPD Kp 10 / Kd 0.2, ±5.6 N·m, measured friction |
| Rear actuator | IdealPD Kp 0 / Kd 0.2, ±6.28 N·m, 0.106 N·m friction (= v2 law) |
| Commands | v* ~ U(0.5, 2.5) m/s; heading* ~ U(±10°) via P-control (k 0.5, clip 0.5 rad/s); resample U(3,5) s |
| Rewards | survival 1, velocity 3, heading 5, action magnitude 1 (w [1, 0.3]), action rate 2 |
| Terminations | 64 s timeout; \|roll\| > 45° |
| DR (always on) | ground/tire friction U(0.5,1.2), mass ×U(0.6,1.8), COM z ±15 mm, actuator gains ×U(0.9,1.1), roll U(±10°), spawn vx U(1.0,2.5), steer U(±20°) |
| PPO | rollouts 16, 4 mini-batches, 5 epochs, lr 1e-4 cosine (T_max = updates), entropy 0.01; 7,500 updates (120k timesteps); saves `best_agent.pt` |

The policy never acts at rest (episodes start rolling at 1.0–2.5 m/s).

### Port ablation tasks: `Locomotive-Abl-<name>-v0`

Registered from `ABLATION_CFGS` in `locomotive_env_cfg_ablation.py` (same agent
YAML): `CycleRL` (replica of CycleRL's original plant), the 2×2×2
`{100Hz,400Hz}-{Implicit,IdealPD}-{Kd0.2,Kd5}` factorial, and one-factor reverts
`Port-{CycleRLSolver,Friction1.0,NoPassiveFriction,SpawnHigh}`. Results:
`isaaclab_train/logs/locomotive_ablation/summary.md`. Main finding: **rear Kd 5
(CycleRL) vs 0.2 (port) is what explains the transfer gap** (MuJoCo flat STE
12.3° → ~1–2°, gravel BSR ~0% → 85–93%); physics rate and minor factors matter
little; the two 100 Hz IdealPD runs did not train (not valid evidence).

### Commands

```bash
cd isaaclab_train && export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Train (skrl; --task defaults to Locomotive-Cylinder-Aligned-v0; Hydra env./agent. overrides)
~/IsaacLab/isaaclab.sh -p scripts/skrl/train_locomotive.py --headless --seed 42

# Isaac paper metrics, one mode (Isaac python)
~/IsaacLab/isaaclab.sh -p scripts/skrl/play_locomotive_metrics.py \
  --checkpoint <run>/checkpoints/best_agent.pt --eval_mode paper_flat --headless
# Full suite (system python; one Kit process per mode) and scorer
python3 scripts/eval_locomotive/eval_all_metrics.py --checkpoint <best_agent.pt> --headless
python3 scripts/eval_locomotive/score_paper_table.py ...

# MuJoCo paper suite (system python; exports npz via Isaac python; rear Kd read from env.yaml)
cd ../mujoco_sim2sim && python3 scripts/locomotive/eval_paper_suite.py --checkpoint <run|.pt> --modes paper_flat

# ONNX + manifest for the Pi (Isaac python; deterministic hashes)
cd ../isaaclab_train && ~/IsaacLab/_isaac_sim/python.sh scripts/deployment/export_cyclerl_locomotive_onnx.py \
  --output-dir ../deploy/source/trackstand_deploy/models
```

Paper metrics: BSR (balance success, |roll| < 0.5 rad all episode), STE (heading
RMSE, deg), VTE (velocity error), plus BRT/CAT/MNT/MSS. Compare transfer with
MuJoCo flat STE and gravel BSR; gaps under ~4° STE are seed noise.
MuJoCo locomotive has **no passive joint friction**; terrain rows (rough/slope/steps)
are Isaac-only. `scripts/skrl/train.py` and `play_with_metrics.py` are stale CycleRL
copies — use `train_locomotive.py` / `play_locomotive_metrics.py`.

### Models

Default **rank 0**: `locomotive_aligned_1900_drive03_2026-10-29_17-06-47.onnx`
(`Locomotive-Cylinder-Aligned-v0`, SHA `ec3ccae9…`; flat BSR Isaac 99.9% / MuJoCo
100%, flat STE 3.51° / 2.38°, MuJoCo gravel 93.2%). Ranks 1–5 are the CycleRL
originals (trained on implicit rear Kd 5, which no bike mode reproduces). All in
`deploy/source/trackstand_deploy/models/locomotive_*` with
`locomotive_policy_manifest.json` (hashes, contract, 16 reference I/O pairs checked
at startup).

---

## MuJoCo sim-to-sim (`mujoco_sim2sim/`)

| Path | Role |
|------|------|
| Trackstand v1 viewer | `scripts/run_deploy_viewer.py --task deploy_sim\|zero_agent\|free_fall` |
| Trackstand v1 headless eval | `scripts/run_deploy_eval.py` (same env/policy as viewer) |
| Trackstand contract mirror | `src/newbike_mujoco/contract.py`, `env.py`, `policy.py`, `viewer_envs.py` |
| Locomotive | `scripts/locomotive/` (`eval_paper_suite.py`, `compare_isaac_mujoco.py`, `eval_command_sweeps.py`, `control_gui.py`, `trackstand_plant.py`, `policy_numpy.py`) |
| MJCF | `models/trackstand_rl_0916_bike/trackstand_rl_0916.xml` |
| Reports | `mujoco_sim2sim/reports/<date>_<topic>/` |

```bash
cd mujoco_sim2sim
python scripts/run_deploy_viewer.py --task deploy_sim \
  --mjcf models/trackstand_rl_0916_bike/trackstand_rl_0916.xml \
  --num_envs 64 --seed 42            # default --pt = no-DR model_75300
python scripts/run_deploy_eval.py --num_envs 64 --num_steps 3000 --seed 42
```

Trackstand: use `--pt` (v1 checkpoint) or `--onnx` (v1 export); files in
`policies/` are historical (13-input / NewBike) and incompatible with v1.
`--physics_hz 800` is a diagnostic (Isaac reference 400 Hz).
`--velocity_limit_mode project` (default) is a hard speed projection, not
PhysX-equivalent. The viewer augments the Fusion MJCF (actuators, joint names,
timestep, tire/floor friction); per-geom friction/solref on the export overrides
class defaults — augmentation must touch geoms explicitly. Viewer exit saves metrics
and torque traces under `mujoco_sim2sim/logs/`.

---

## Isaac Lab notes (`isaaclab_train/`)

| Piece | Location (under `source/trackstand_rl/trackstand_rl/tasks/manager_based/`) |
|-------|----------|
| Trackstand registration | `trackstand/__init__.py` → v0 / v1 / v2 |
| Trackstand env classes | `trackstand/trackstand_bike/trackstand_rl_env.py`, `trackstand_rl_env_v2.py` |
| Trackstand env cfgs | `trackstand/trackstand_bike/trackstand_rl_env_cfg.py` (v0 base), `_cfg_v1.py`, `_cfg_v2.py` |
| Robot / actuators | `trackstand/bicycle_cfg.py` (`TRACKSTAND_ROBOT_CFG`, shared with locomotive); v1/v2 and locomotive override actuators |
| Trackstand MDP | `trackstand/mdp/stationary/` (active); `mdp/dynamic/` is legacy |
| Trackstand PPO | `trackstand/agents/rsl_rl_ppo_cfg.py` (MLP [32,32], entropy 0.005); Beta: `rsl_rl_ppo_beta_cfg.py`, `rsl_rl_ppo_beta_v2_cfg.py`, `beta_distribution.py` |
| Locomotive | `locomotive/` (see above) |
| Reports | `isaaclab_train/reports/<date>_<topic>/` |

- Import trigger: `import source.trackstand_rl.trackstand_rl.tasks` with
  `PYTHONPATH=isaaclab_train`.
- Hydra overrides use the `env.` node (e.g. `env.sim.dt=0.01`; skrl also `agent.`).
  **Do not** assign `env.step_dt` — set `sim.dt` and `decimation`.
- Logs are gitignored.
- Lint: `ruff check source scripts && ruff format --check source scripts`.
- Subtree `isaaclab_train/CLAUDE.md` is stale (v0-era, external
  `~/newbike_mujoco_deploy/`); prefer this file.

---

## Raspberry Pi deploy (`deploy/`)

Package `source/trackstand_deploy/`. Shared I/O contract (trackstand):
`trackstand_deploy/contracts.py`.

| Executable | Stack | Role |
|------------|-------|------|
| `imu_publisher` | both | YahBoom IMU → `/trackstand/imu` (header backdated to parse time) |
| `rc_kickstand` | both | CRSF CH7 → enable/kickstand authority; locomotive overlay adds `/locomotive/rc_sticks` (throttle, yaw) |
| `kickstand_driver` | both | STS3250 UART `/dev/ttyAMA2` |
| `policy_controller` | trackstand | ONNX @ 50 Hz; SHA-checks model |
| `motor_mit_io` | trackstand | Firmware MIT steering (Kp 10, Kd 0.2) + C0 rear torque (**default launch**) |
| `motor_io` | trackstand | Host PD steering + C0 rear |
| `motor_io_position` | trackstand | Firmware C2 steering + C0 rear |
| `locomotive_controller` | locomotive | ONNX @ 50 Hz, launch → ride state machine, → `/locomotive/motor_command` `[enable, steer_rad, rear_rad_s]` |
| `motor_mit_velocity_io` | locomotive | MIT steering + rear `rear_control_mode`: **`mit`** (Kp 0 / Kd 0.2, host-bounded 6.28 N·m; drive F0 Vel_Max widened to 50 rad/s and restored on clean exit) or `c1` (firmware speed PI); B3 lowered to 6.28 N·m/Kt |

Run **exactly one** motor backend (never alongside another backend or experiment
CAN writers). Only the MIT backends have the single CAN owner, kernel timestamps,
AE driver-status freshness gate and acknowledged-disable state machine.

Model selection (both stacks): filename **and** SHA-256 must change together.
- Trackstand default `trackstand_rl_rsl_2026-10-21_11-18-29_model_2500.onnx`
  (SHA `d9701945b48e62fec14ea3258cf4255b0c7541d530eb8ec7789a770e9a5a63b2`),
  `MODEL_FILENAME` / `MODEL_SHA256` in `policy_controller.py`; provenance
  `models/v1_policy_manifest.json`.
- Locomotive: `model_filename` / `model_sha256` in `config/locomotive.yaml`, and
  `rear_control_mode` must match the model's trained rear law (`mit` for the
  aligned model).

```bash
cd deploy
source /opt/ros/jazzy/setup.bash   # or humble
colcon build --symlink-install --packages-select trackstand_deploy
source install/setup.bash
sudo ip link set can0 up type can bitrate 1000000
# trackstand: ros2 launch trackstand_deploy trackstand.launch.py   (LIVE)
# locomotive: ros2 launch trackstand_deploy locomotive.launch.py   (LIVE)
```

`locomotive.launch.py` records every run (rosbag `-a`, node logs, config,
`session.json` with git commit / model SHA / dry_run) to
`~/trackstand_logs/locomotive_<timestamp>/`; `record:=false` disables it,
`log_dir:=` moves it.

For first bring-up start nodes individually with `-p dry_run:=true` on the motor
node (suppresses motor CAN writes; does **not** disable kickstand servos or
already-enabled motors; locomotive dry-run only reads B3/F0). Motor parameters are
read-only after startup — restart to change them. Locomotive commissioning order:
`deploy/docs/LOCOMOTIVE.md`.

### Deploy safety (current source)

- Trackstand readiness gate: `max_roll_rad` (0.5236) roll / inversion check,
  IMU/motor/RC freshness, motor faults, steering centered for start. RC down
  disables balancing and requests stands down (published in `finally`). Policy
  faults latch off but do **not** lower stands. Hard driver faults need restart.
- Locomotive: DOWN → LAUNCH (stands down, steer straight, rear ramps 0.5 m/s²) →
  RIDING at ≥ `wake_speed_m_s` 1.0 → FAULT_LATCHED on any readiness loss or
  speed < 0.3 m/s (stands not moved; only RC down clears). `max_speed_cmd_m_s` 1.5.
- MIT: total steering PD torque cap (±5.6 N·m) is **not established** in firmware;
  motor-side CD watchdog not configured. Read `deploy/docs/MOTOR_MIT_IO.md`,
  `AUDIT.md`, `AUDIT_2026-10-29.md`.
- Known unresolved: rear sim-vs-real response mismatch, IMU latency, real plant
  fidelity. Locomotive: rear MIT Kd 0.2 not run on hardware (only Kd 0.30
  airborne), whether B3 bounds MIT current, stick channels (CH3/CH4) and rolling
  with stands down unverified. Passing tests are not a hardware green light.

Hardware params (serial by-id paths, signs, servo poses) are bike-specific and
single-sourced in `config/params.yaml`; `config/locomotive.yaml` holds only
locomotive keys. Checklist: `deploy/docs/HARDWARE_CHECKLIST.md`, staged procedure:
`deploy/docs/BIKE_DIAGNOSTIC_TESTS.md`, telemetry: `deploy/docs/TELEMETRY.md`.

### Motor experiments (`deploy/experiments/`)

Standalone CAN scripts (speed sweeps, sine tests, policy / initial-transient
replays, `drive_repeatability.py`) with docs (`*.md`) and dated analysis under
`reports/<date>_<topic>/`. They write CAN directly — never run with a deploy
motor backend.

---

## Reference / teaching (`tsinghua_bike_project/`)

Not a deployment path. Useful for hardware docs, CAN/servo/IMU vendor materials
(`can_protocol.txt`), assembly CAD, and the 3-state linear model + LQR teaching
package under `assign4&5_bike_mujocosim_with_control/`. Its nominal controller is a
**mobile** balance baseline (speed-dependent A(v)) — closer to locomotive than to
trackstand, but not either RL contract. See `tsinghua_bike_project/CLAUDE.md`.

## Summary log (always)

**Every results summary you produce** (eval or ablation results, run reports, findings given in chat)
must also be written as a dated Markdown file in `summaries/`, named
`YYYY-MM-DD_HHMM_<topic>.md` (HKT, time written). Include the dates and times that matter (run
start/finish, eval finish, summary written), the source files, setup, results tables, findings,
caveats and next steps. Add a row at the top of `summaries/README.md`. Never rewrite earlier entries;
a correction is a new file that links the old one. This is in addition to `progress_record.md`
(see the "Experiment record" row below), not a replacement for it.

## Where to change what

| Change | Put it here |
|--------|-------------|
| Trackstand reward / obs / event / termination | `trackstand/mdp/stationary/` + register in the versioned env cfg |
| New trackstand task variant | New `trackstand_rl_env_cfg_vN.py` + `gym.register` in `trackstand/__init__.py`; leave older versions unchanged |
| Locomotive recipe / plant | `locomotive/locomotive_env_cfg.py` (keep `mdp/` verbatim CycleRL copies; add new terms beside them) |
| Locomotive ablation variant | `locomotive_env_cfg_ablation.py` `ABLATION_CFGS` (auto-registered as `Locomotive-Abl-<name>-v0`) |
| Actuator gains / joint limits | Versioned cfg `__post_init__` / locomotive actuator cfgs (and `bicycle_cfg.py` for v0) + keep MuJoCo/deploy in sync |
| Trackstand action scales / obs layout | Isaac cfg **and** `deploy/.../contracts.py` **and** `mujoco_sim2sim/src/newbike_mujoco/contract.py` |
| Locomotive tensor contract | Isaac cfg **and** `export_cyclerl_locomotive_onnx.py` / manifest **and** `locomotive_controller.py` **and** `mujoco_sim2sim/scripts/locomotive/` |
| Policy selection on bike | Trackstand: `policy_controller.py` constants; locomotive: `config/locomotive.yaml`; file under `models/` + manifest |
| Motor protocol / rates | Selected `motor_*.py` + `params.yaml` (locomotive keys in `locomotive.yaml`); never run two backends |
| Hardware signs / serial paths | `deploy/.../config/params.yaml` only |
| Experiment record | Append to `progress_record.md` (§N) and `progress_record.txt`; never rewrite earlier entries |
| Results summary | New dated `summaries/YYYY-MM-DD_HHMM_<topic>.md` + row in `summaries/README.md` (always) |

## Tests

```bash
# Deploy package, both stacks (incl. test_locomotive.py; 2026-10-29 audit: 649 passed / 1 skipped)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/source/trackstand_deploy \
  python3 -m pytest -q -p no:cacheprovider deploy/source/trackstand_deploy/test

# Deploy experiments + MuJoCo (trackstand)
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s deploy/experiments/tests
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s mujoco_sim2sim/tests

# Isaac (trackstand only; needs Isaac Lab interpreter; two known legacy v0/v1 viscous None-vs-0 subcase failures)
cd isaaclab_train && ~/IsaacLab/isaaclab.sh -p -m unittest discover -s tests

# Nominal teaching controller
cd tsinghua_bike_project/assign4\&5_bike_mujocosim_with_control/assign4\&5_bike_mujocosim_with_control \
  && python -m unittest -v tests.test_nominal_bike_control
```

There are no Isaac or MuJoCo unit tests for locomotive; validate with the paper
suites (Isaac + MuJoCo, seeded). For any simulation behavior change, run a seeded
headless eval and report metrics; a viewer run alone is insufficient (`AGENTS.md`).

## Common pitfalls

1. **Wrong stack** — trackstand and locomotive differ in obs, actions, trainer, motor backend, topics and launch; never mix nodes, ONNX files or checkpoints across them.
2. **Wrong trackstand version** — use `Trackstand-Balance-v1` unless deliberately working on v0 history or v2.
3. **13 vs 11 obs** — v0 exports (41050, 91850, `policies/*.onnx`) do not run in v1 MuJoCo/deploy.
4. **RSL-RL vs skrl** — locomotive trains with `scripts/skrl/train_locomotive.py`; `scripts/skrl/train.py` / `play_with_metrics.py` are stale CycleRL copies.
5. **Rear law mismatch (locomotive)** — `rear_control_mode` must match the model (aligned = MIT Kd 0.2). Rear Kd is the dominant sim-to-sim factor.
6. **Stale CycleRL paths** — use `source.trackstand_rl...` + `PYTHONPATH=isaaclab_train`; `~/bike_test/CycleRL` is reference only, not imported.
7. **Wrong Python** — Isaac / MuJoCo / ROS are separate; locomotive suite drivers use system python and spawn Isaac python; never build Pi nodes with Isaac Python.
8. **`--dr` mismatch (trackstand)** — default is off; compare only like-for-like. Locomotive has no DR switch.
9. **Export scales** — trackstand ONNX emits raw actions (deploy ×1.57 / ×6.28); locomotive ONNX embeds the obs scaler and emits raw Gaussian means (controller scales ×20° / ×42.857).
10. **Launch is live motors** — both launch files; use explicit `dry_run:=true` for data-flow checks.
11. **Old ONNX sidecars** — some exports need companion `.onnx.data`; current v1 and locomotive exports embed weights.
12. **`ros2_ws/` empty** — runtime is `deploy/`; reference code also in `tsinghua_bike_project/source/`.
13. **Subtree CLAUDE / docstring drift** — prefer this file + `README.MD` + source + latest `progress_record.md` entries.

## Further reading

- Root `README.MD` — trackstand command walkthrough
- `deploy/docs/LOCOMOTIVE.md` — locomotive models, contract, motor modes, commissioning, open items
- `AGENTS.md` — style, modular MDP registration, commit/PR norms
- `deploy/README.md`, `deploy/docs/*.md` — hardware, MIT/C2, telemetry, audits
- `deploy/experiments/README.md` — motor replay / repeatability experiments
- `mujoco_sim2sim/README.md` — v1 alignment, contact/velocity approximations
- `isaaclab_train/logs/locomotive_ablation/summary.md` — why the port transfers better than CycleRL
- `~/bike_test/trackstand_rl_test_layer/isaaclab_train/logs/ablation_eval_dr/summary.md` — trackstand training-factor ablation, all `--dr` (2026-11-03); no-DR batch in `ablation_eval/`
- `progress_record.md` — evidence-backed experiment history
