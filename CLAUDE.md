# CLAUDE.md

Guidance for AI coding agents working in this repository.

## What this is

**Trackstand RL** trains and deploys a **stationary bicycle balancing** policy
(trackstand): keep the bike upright with minimal planar drift. The **active
contract is `Trackstand-Balance-v1`** (11 observations, Beta policy) across
Isaac, MuJoCo and the Pi. A policy outputs steering **position** and rear-wheel
**torque**; RC sticks are not policy inputs — the RC switch only authorizes
balancing and commands paired kickstands.

Pipeline: **Isaac Lab train → ONNX export → MuJoCo sim-to-sim → Raspberry Pi ROS 2**.
This is a research codebase, not a certified autonomous or road-ready system.

CycleRL was removed on 2026-09-23. Prefer current paths under
`isaaclab_train/source/trackstand_rl/` and ignore stale `CycleRL` /
`NewBike-Balance` references in old docs or script headers.

## Task versions

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
  only a proposal.

Note: the v0 cfg docstrings still say "THE single env config / no other variants"
— that is stale.

## Control contract (v1; shared across Isaac / MuJoCo / deploy)

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

### Isaac rewards

- `survival` weight **5.0** — binary while \|roll\| < 30°
- `drift_gauss` weight **2.0** — Gaussian σ = 0.25 m
- `rate_penalty` weight **0.1** — v1: −‖Δa‖ on raw actions (equal weights; v0
  divided by physical scales)

### Isaac terminations (all versions)

`time_out`, `fallen` (\|roll\| > 30° or inverted), `diverged`, `drifted` (> 0.5 m),
`yaw_deviation` (> 30°), `wheel_net` (> 7.2 rad accumulated rear angle).

MuJoCo v1 viewer/evaluator now enable **all six** by default; `--pure_survival`
restricts to `time_out` + `fallen`.

### Domain randomization

- **Default off** (`domain_randomization=False`). Opt in with **`--dr` only**
  (no `--no_dr`, no ramp).
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

## Isaac Lab (`isaaclab_train/`)

Active package:
`source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/`

| Piece | Location |
|-------|----------|
| Gym registration | `__init__.py` → v0 / v1 / v2 |
| Env classes | `trackstand_bike/trackstand_rl_env.py` (`TrackstandRLEnv`, `TrackstandRLEnvV1`), `trackstand_rl_env_v2.py` |
| Env cfgs | `trackstand_bike/trackstand_rl_env_cfg.py` (v0 base: rewards, terminations, events, DR switch), `_cfg_v1.py`, `_cfg_v2.py` |
| Robot / actuators | `bicycle_cfg.py` (`TRACKSTAND_ROBOT_CFG`); v1/v2 override actuators in their cfg `__post_init__` |
| MDP terms | `mdp/stationary/` (active, incl. `actions.py` slew limiter, `wheel_net.py`); `mdp/dynamic/` is legacy |
| PPO cfg | `agents/rsl_rl_ppo_cfg.py` (MLP **[32,32]**, entropy 0.005); Beta: `rsl_rl_ppo_beta_cfg.py`, `rsl_rl_ppo_beta_v2_cfg.py`, `beta_distribution.py` |
| Reports | `isaaclab_train/reports/<date>_<topic>/` |

Import trigger used by scripts:
`import source.trackstand_rl.trackstand_rl.tasks`

From `isaaclab_train/`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Train (deterministic variant preferred for reproducibility)
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_deterministic.py \
  --task Trackstand-Balance-v1 --num_envs 4096 \
  --max_iterations 60000 --seed 42 --headless --dr

# Headless eval (match --dr to the run)
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_eval.py \
  --task Trackstand-Balance-v1 --checkpoint /abs/path/model.pt \
  --num_envs 64 --num_steps 3000 --seed 42 --headless --dr

# ONNX export (raw actions; opset 18)
~/IsaacLab/isaaclab.sh -p scripts/deployment/export_rsl_rl_onnx.py \
  --checkpoint /abs/path/model.pt \
  --output scripts/deployment/trackstand_policy.onnx
```

Logs: `logs/rsl_rl/<experiment_name>/<timestamp>/` (gitignored). Last checkpoint
≠ best. `play_eval.py` overwrites `eval.txt` / torque profiles beside the
checkpoint. Hydra overrides use the `env.` node (e.g. `env.sim.dt=0.01`).

**Do not** assign `env.step_dt` on the cfg — set `sim.dt` and `decimation`.

Lint: `ruff check source scripts && ruff format --check source scripts`

Subtree `isaaclab_train/CLAUDE.md` is stale (v0-era, external
`~/newbike_mujoco_deploy/`); the in-repo MuJoCo mirror is `mujoco_sim2sim/`.

## MuJoCo sim-to-sim (`mujoco_sim2sim/`)

| Path | Role |
|------|------|
| v1 viewer | `scripts/run_deploy_viewer.py --task deploy_sim\|zero_agent\|free_fall` |
| v1 headless eval | `scripts/run_deploy_eval.py` (same env/policy as viewer) |
| Contract mirror | `src/newbike_mujoco/contract.py`, `env.py`, `policy.py`, `viewer_envs.py` |
| Trackstand MJCF | `models/trackstand_rl_0916_bike/trackstand_rl_0916.xml` |
| Reports | `mujoco_sim2sim/reports/<date>_<topic>/` |

```bash
cd mujoco_sim2sim
python scripts/run_deploy_viewer.py --task deploy_sim \
  --mjcf models/trackstand_rl_0916_bike/trackstand_rl_0916.xml \
  --num_envs 64 --seed 42            # default --pt = no-DR model_75300
python scripts/run_deploy_eval.py --num_envs 64 --num_steps 3000 --seed 42
```

Use `--pt` (v1 checkpoint) or `--onnx` (v1 export) for another policy; files in
`policies/` are historical (13-input / NewBike) and incompatible with v1.
`--physics_hz 800` is a diagnostic (Isaac reference 400 Hz).
`--velocity_limit_mode project` (default) is a hard speed projection, not
PhysX-equivalent. Viewer augments Fusion MJCF (actuators, joint names, timestep,
tire/floor friction); per-geom friction/solref on the export overrides class
defaults — augmentation must touch geoms explicitly. Viewer exit saves metrics
and torque traces under `mujoco_sim2sim/logs/`.

## Raspberry Pi deploy (`deploy/`)

Package: `source/trackstand_deploy/`. Shared I/O contract:
`trackstand_deploy/contracts.py`.

| Executable | Role |
|------------|------|
| `imu_publisher` | YahBoom IMU → `/trackstand/imu` (header backdated to parse time) |
| `rc_kickstand` | CRSF channel 7 → kickstand authority |
| `kickstand_driver` | STS3250 UART `/dev/ttyAMA2` |
| `policy_controller` | ONNX @ 50 Hz; SHA-checks model |
| `motor_mit_io` | Firmware MIT steering (Kp=10, Kd=0.2) + C0 rear (**default launch**) |
| `motor_io` | Host PD steering + C0 rear |
| `motor_io_position` | Firmware C2 steering + C0 rear |

Run **exactly one** motor backend (never alongside experiment CAN writers).
Only MIT has the single CAN owner, kernel timestamps, AE driver-status freshness
gate and acknowledged-disable state machine.

Default bundled policy: `trackstand_rl_rsl_2026-10-21_11-18-29_model_2500.onnx`
(SHA-256 `d9701945b48e62fec14ea3258cf4255b0c7541d530eb8ec7789a770e9a5a63b2`,
`MODEL_FILENAME` / `MODEL_SHA256` in `policy_controller.py`; provenance in
`models/v1_policy_manifest.json`). Switching models requires updating filename +
hash (or passing `model_path` + `model_sha256` params); copying an ONNX alone does
nothing.

```bash
cd deploy
source /opt/ros/jazzy/setup.bash   # or humble
colcon build --symlink-install --packages-select trackstand_deploy
source install/setup.bash
sudo ip link set can0 up type can bitrate 1000000
```

`ros2 launch trackstand_deploy trackstand.launch.py` uses **`motor_mit_io`** and
checked-in `dry_run: false` — **not** a dry-run shortcut. For first bring-up,
start nodes individually with `-p dry_run:=true` on the motor node (suppresses
motor CAN writes; does **not** disable kickstand servos or already-enabled motors).
Motor parameters are read-only after startup — restart to change them.

### Deploy safety (current source)

- Readiness gate includes `max_roll_rad` (0.5236, read by the controller) roll /
  inversion check, IMU/motor/RC freshness, motor faults, and steering centered for start.
- RC down disables balancing and requests stands down (published in `finally`).
  Policy faults latch off but do **not** lower stands. Hard driver faults need restart.
- MIT: total steering PD torque cap (±5.6 N·m) is **not established** in firmware;
  motor-side CD watchdog is not configured. Read `deploy/docs/MOTOR_MIT_IO.md`,
  `AUDIT.md`, `AUDIT_2026-10-29.md`.
- Known unresolved: rear sim-vs-real response mismatch, IMU latency, real plant
  fidelity. Passing tests are not a hardware green light.

Hardware params (serial by-id paths, signs, servo poses) are bike-specific —
edit `config/params.yaml` only when hardware identities change. Checklist:
`deploy/docs/HARDWARE_CHECKLIST.md`, staged procedure:
`deploy/docs/BIKE_DIAGNOSTIC_TESTS.md`, telemetry: `deploy/docs/TELEMETRY.md`.

### Locomotive stack (branch `locomotive`)

CycleRL riding policies (`CycleRL-Trackstand-Cylinder-Paper-v0`, same cylinder USD)
with their **original tensor contract** kept: 8 obs `[gravity_y, ang_vel_x/y/z, speed,
steer, vel_cmd, yaw_rate_cmd]`, skrl `[512,256,128,64]` ELU with the ±5σ input
normaliser embedded in the ONNX, raw Gaussian means, 50 Hz. Top-5 ONNX + manifest
(`models/locomotive_*`), exporter `isaaclab_train/scripts/deployment/export_cyclerl_locomotive_onnx.py`.
Training in this repo: `Locomotive-Cylinder-Aligned-v0`
(`isaaclab_train/.../tasks/manager_based/locomotive/`) — CycleRL's cylinder paper
recipe (rank-1 `1900_drive03` defaults, riding DR/rewards/commands, verbatim MDP copies
in `locomotive/mdp/`) on trackstand's robot cfg, 400 Hz physics, v1 steering IdealPD and
v2 rear IdealPD Kd 0.2. **skrl**, not RSL-RL:
`~/IsaacLab/isaaclab.sh -p scripts/skrl/train_locomotive.py --headless` (logs
`logs/skrl/locomotive/<task>/`). `scripts/skrl/train.py` is a stale CycleRL copy.
Eval (CycleRL's stack, ported): Isaac `scripts/skrl/play_locomotive_metrics.py --eval_mode
paper_flat` (package `locomotive/eval/`, eval-only events in `mdp/eval_events.py`), suite
driver `scripts/eval_locomotive/eval_all_metrics.py` (system python), scorer
`score_paper_table.py`; MuJoCo `mujoco_sim2sim/scripts/locomotive/eval_paper_suite.py
--checkpoint <run|.pt> --modes paper_flat` (system python; exports npz via Isaac python,
rear Kd read from the run's env.yaml; no passive joint friction). The old
`scripts/skrl/play_with_metrics.py` is a stale CycleRL copy.
Separate nodes/topics: `locomotive_controller` → `/locomotive/motor_command`
`[enable, steer_rad, rear_rad_s]` → `motor_mit_velocity_io` (MIT steering + rear
**C1 speed**, B3 lowered to 6.28 N·m/Kt). Launch `locomotive.launch.py` loads
`params.yaml` then `config/locomotive.yaml`. Never mix with trackstand nodes.
Details and open items: `deploy/docs/LOCOMOTIVE.md`.

### Motor experiments (`deploy/experiments/`)

Standalone CAN scripts (speed sweeps, sine tests, policy / initial-transient
replays, `drive_repeatability.py`) with docs (`*.md`) and dated analysis under
`reports/<date>_<topic>/`. They write CAN directly — never run with a deploy
motor backend.

## Reference / teaching (`tsinghua_bike_project/`)

Not the Trackstand RL deployment path. Useful for hardware docs, CAN/servo/IMU
vendor materials (`can_protocol.txt`), assembly CAD, and the 3-state linear model
+ LQR teaching package under `assign4&5_bike_mujocosim_with_control/`. Nominal
controller is a **mobile** balance pedagogical baseline (speed-dependent A(v)),
not the stationary RL contract. See `tsinghua_bike_project/CLAUDE.md`.

## Where to change what

| Change | Put it here |
|--------|-------------|
| Reward / obs / event / termination | `isaaclab_train/.../mdp/stationary/` + register in the versioned env cfg |
| New task variant | New `trackstand_rl_env_cfg_vN.py` + `gym.register` in `__init__.py`; leave older versions unchanged |
| Actuator gains / joint limits | v1/v2 cfg `__post_init__` (and `bicycle_cfg.py` for v0) + keep MuJoCo/deploy in sync |
| Action scales / obs layout | Isaac cfg **and** `deploy/.../contracts.py` **and** `mujoco_sim2sim/src/newbike_mujoco/contract.py` |
| Policy selection on bike | `policy_controller.py` `MODEL_FILENAME` / `MODEL_SHA256` + file under `models/` + manifest |
| Motor protocol / rates | Selected `motor_*.py` + `params.yaml`; never run two backends |
| Hardware signs / serial paths | `deploy/.../config/params.yaml` only |
| Experiment record | Append to `progress_record.md` (§N) and `progress_record.txt`; never rewrite earlier entries |

## Tests

```bash
# Deploy package (2026-10-29 audit: 649 passed / 1 skipped)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/source/trackstand_deploy \
  python3 -m pytest -q -p no:cacheprovider deploy/source/trackstand_deploy/test

# Deploy experiments + MuJoCo
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s deploy/experiments/tests
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s mujoco_sim2sim/tests

# Isaac (needs Isaac Lab interpreter; two known legacy v0/v1 viscous None-vs-0 subcase failures)
cd isaaclab_train && ~/IsaacLab/isaaclab.sh -p -m unittest discover -s tests

# Nominal teaching controller
cd tsinghua_bike_project/assign4\&5_bike_mujocosim_with_control/assign4\&5_bike_mujocosim_with_control \
  && python -m unittest -v tests.test_nominal_bike_control
```

For simulation behavior changes: run headless eval with seed and report metrics;
a viewer run alone is insufficient (`AGENTS.md`).

## Common pitfalls

1. **Wrong task version** — use `Trackstand-Balance-v1` unless deliberately working on v0 history or v2.
2. **13 vs 11 obs** — v0 exports (41050, 91850, `policies/*.onnx`) do not run in v1 MuJoCo/deploy.
3. **Stale CycleRL paths** — use `source.trackstand_rl...` + `PYTHONPATH=isaaclab_train`.
4. **Wrong Python** — Isaac / MuJoCo / ROS are separate; never build Pi nodes with Isaac Python.
5. **`--dr` mismatch** — default is off; compare only like-for-like.
6. **Export scales** — ONNX emits raw actions; deploy multiplies by 1.57 / 6.28.
7. **Launch is live motors** — use explicit `dry_run:=true` for data-flow checks.
8. **Old ONNX sidecars** — some exports need companion `.onnx.data`; current v1 exports embed weights.
9. **`ros2_ws/` empty** — Trackstand runtime is `deploy/`; reference code also in `tsinghua_bike_project/source/`.
10. **Subtree CLAUDE / docstring drift** — prefer this file + `README.MD` + source + latest `progress_record.md` entries.

## Further reading

- Root `README.MD` — full command walkthrough
- `AGENTS.md` — style, modular MDP registration, commit/PR norms
- `deploy/README.md`, `deploy/docs/*.md` — hardware, MIT/C2, telemetry, audits
- `deploy/experiments/README.md` — motor replay / repeatability experiments
- `mujoco_sim2sim/README.md` — v1 alignment, contact/velocity approximations
- `progress_record.md` — evidence-backed experiment history (latest §244, 2026-10-29)
