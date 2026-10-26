# Trackstand RL

Reinforcement-learning-based **stationary bicycle balancing**, from Isaac Lab training to MuJoCo sim-to-sim validation and Raspberry Pi deployment with ROS 2.

Training, sim-to-sim and Raspberry Pi deployment now use the `Trackstand-Balance-v1` contract: 11 observations and bounded Beta actions. V0 remains available for historical experiments. A policy commands steering **position** and rear-wheel **torque** for stationary balance. RC controls the kickstands and authorizes balancing, not steering/throttle policy inputs. Matching software contracts does not establish identical physical actuator response or successful hardware balance.

The workflow is **train → evaluate → export ONNX → validate in MuJoCo → commission on the supported bike**. Simulation results and unloaded motor experiments are recorded, but this is a research codebase, not a certified autonomous or road-ready system.

## Repository layout

| Directory / file | Purpose |
|---|---|
| [`isaaclab_train/`](isaaclab_train/) | Isaac Lab environments, RSL-RL PPO training/evaluation, policy export, USD/URDF assets and simulation reports. Active task code: `source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/`; executable scripts: `scripts/rsl_rl/` and `scripts/deployment/`. |
| [`mujoco_sim2sim/`](mujoco_sim2sim/) | Independent MuJoCo policy validation. `src/newbike_mujoco/` contains the environment/inference code; `scripts/` contains viewers/evaluation; `models/` contains MJCF/meshes; `policies/` contains exports; `tests/` contains regressions. |
| [`deploy/`](deploy/) | ROS 2 workspace for the Raspberry Pi. `source/trackstand_deploy/` contains nodes, launch/configuration and bundled policies; `docs/` contains commissioning/telemetry guides; `experiments/` contains standalone motor tests and analysis reports. |
| [`ros2_ws/`](ros2_ws/) | Separate reference/on-bike ROS 2 workspace with hardware drivers and nominal controllers. It is not the Trackstand deployment package; its settings are not interchangeable with `deploy/`. Currently stored as a Git link and may need obtaining separately if empty after cloning. |
| [`tsinghua_bike_project/`](tsinghua_bike_project/) | Reference bicycle code, hardware/CAN documentation, assembly/CAD materials, nominal-control teaching code and historical records. |
| [`rosbags/`](rosbags/) | Local sensor/deployment/test recordings. MCAP files are ignored by Git and are not guaranteed to be present in a clone. |
| [`progress_record.md`](progress_record.md) / [`progress_record.txt`](progress_record.txt) | Detailed evidence-backed history / concise chronological ledger. Consult dated entries before comparing experiments. |
| [`AGENTS.md`](AGENTS.md) | Contributor conventions and repository-specific development guidance. |

Training checkpoints, TensorBoard events and generated logs belong in component `logs/` directories, which are ignored by Git. Historical scripts/comments can refer to removed `CycleRL` packages or old checkout paths; the explicit commands below target the current layout.

## Requirements and checkout

```bash
git clone https://github.com/Kylineason-Liu/trackstand_rl.git
cd trackstand_rl
```

Use separate environments for the three workflows:

- **Training:** a compatible Isaac Lab/Isaac Sim installation, its Python environment, RSL-RL and an NVIDIA GPU suitable for the selected number of environments. Follow the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) and check this project's [environment metadata](isaaclab_train/pyproject.toml); do not assume a generic system Python or the MuJoCo environment can run Isaac Sim.
- **MuJoCo:** Python with the component requirements, `onnxruntime` for exported inference, and Matplotlib for viewer-generated plots. The viewer needs a desktop/display; physics runs on the CPU.
- **Bike:** Raspberry Pi with Linux SocketCAN, ROS 2, colcon/ament, appropriate serial/CAN permissions, and the IMU, two motors, CRSF receiver and kickstand servos. Install the Python dependencies in [requirements.txt](deploy/requirements.txt) plus `smbus2`, using the **same Python environment used by ROS/colcon**. ROS dependencies are declared in [package.xml](deploy/source/trackstand_deploy/package.xml). Respect your OS's managed-Python rules rather than forcing packages into an incompatible interpreter.

Examples assume the checkout is `~/trackstand_rl` and Isaac Lab is `~/IsaacLab`; adjust those paths. The ROS examples use **Jazzy**; substitute your installed distribution, such as `humble`. Never use the training environment to build/run Raspberry Pi nodes.

## 1. Isaac Lab training and evaluation

Run from `isaaclab_train/`. The current task is imported directly as `source.trackstand_rl.trackstand_rl.tasks`; the old `pip install -e source/CycleRL` template instruction does not apply. Add this component directory to Python's import path:

```bash
cd ~/trackstand_rl/isaaclab_train
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_deterministic.py \
  --task Trackstand-Balance-v1 --num_envs 4096 \
  --max_iterations 60000 --seed 42 --headless --dr
```

`train_deterministic.py` enables deterministic training settings; `train.py` is the regular variant. Adjust the training budget and environment count to your hardware. Runs are saved beneath `logs/rsl_rl/trackstand_rl_rsl/<timestamp>/`, including effective configuration, TensorBoard events and `model_<iteration>.pt` checkpoints. The last checkpoint is not necessarily the best.

**DR is off by default.** Keep `--dr` for full per-environment/per-episode domain randomization; omit it for nominal physics/reset conditions. There is no DR ramp and no `--no_dr` flag. Observation white noise remains enabled in nominal mode.

Evaluate a checkpoint with matched task/DR settings; replace the placeholder path with an actual checkpoint. Selecting a checkpoint does not automatically restore its historical task configuration: compare the saved `params/env.yaml` before reproducing older experiments.

```bash
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_eval.py \
  --task Trackstand-Balance-v1 --checkpoint /absolute/path/to/v1_model.pt \
  --num_envs 64 --num_steps 3000 --seed 42 --headless --dr
```

This reports episode lengths, termination rates and motion metrics; it is preferable to judging a policy only by its training reward or a viewer run. Use `scripts/rsl_rl/play.py` for Isaac playback. To export the deterministic actor, with the export script's PyTorch/ONNX dependencies installed:

```bash
~/IsaacLab/isaaclab.sh -p scripts/deployment/export_rsl_rl_onnx.py \
  --checkpoint /absolute/path/to/model.pt \
  --output scripts/deployment/trackstand_policy.onnx
```

The export produces raw actions; deployment applies the physical scales. Some older ONNX exports require their companion `.onnx.data` file. Preserve both files when transferring those models.

## 2. MuJoCo sim-to-sim

Set up an independent Python environment:

```bash
cd ~/trackstand_rl/mujoco_sim2sim
python3 -m venv ~/.venvs/trackstand-mujoco
source ~/.venvs/trackstand-mujoco/bin/activate
python -m pip install -r requirements.txt matplotlib onnxruntime
```

Run the v1 Trackstand model (default **20261021 / model_75300**, if the checkpoint is present locally):

```bash
python scripts/run_deploy_viewer.py --task deploy_sim \
  --mjcf models/trackstand_rl_0916_bike/trackstand_rl_0916.xml \
  --num_envs 64 --seed 42
```

Use `--pt /path/to/v1_model.pt` or `--onnx /path/to/v1_export.onnx` for another v1 policy. Historical 13-input 41050/91850 exports are incompatible and rejected. Default paths are repository-relative; exporting a policy does not switch another component's model.

`--num_envs` defaults to **64**; choose `1` for a single-bike view. `--dr` enables randomized resets; omit it for nominal runs. `--speed` controls playback pacing, not policy frequency. Inspect all options with `python scripts/run_deploy_viewer.py --help`.

Both viewer and headless evaluator accept `--physics_hz 800` for 800 Hz physics
and motor PD, while policy inference and target slew stay at 50 Hz. The default
remains 400 Hz; rates must be positive multiples of 50 Hz. Non-default rates
are diagnostic departures from Isaac v1's 400 Hz reference.

On normal viewer exit, available episode metrics and steering/rear torque traces are saved under `mujoco_sim2sim/logs/`. The grid is render-only: environments have independent physics/randomization and do not collide with one another. CPU load may make playback slower than real time.

Headless evaluation now uses the **same v1 environment and policy** as the viewer:

```bash
python scripts/run_deploy_eval.py --num_envs 64 --num_steps 3000 --seed 42
```

All six v1 terminations are enabled by default; `--pure_survival` selects only timeout/fallen. See the [MuJoCo guide](mujoco_sim2sim/README.md) for the remaining contact-friction and velocity-constraint approximations. Contract alignment does not guarantee Isaac-equivalent survival.

## 3. Raspberry Pi ROS 2 deployment

Read the [hardware checklist](deploy/docs/HARDWARE_CHECKLIST.md) and [staged diagnostic procedure](deploy/docs/BIKE_DIAGNOSTIC_TESTS.md) first. Independently support the bike, keep the RC kickstand switch down, start with steering straight/stationary, and have physical power isolation available.

After installing the ROS/Python dependencies, build on the Pi:

```bash
cd ~/trackstand_rl/deploy
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select trackstand_deploy
source install/setup.bash
ros2 pkg executables trackstand_deploy
ip -details link show can0
```

The configured motor bus is `can0` at **1 Mbit/s**. If the interface exists but is not configured/up, bring it up **with all motor writers stopped**:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

Inspect [params.yaml](deploy/source/trackstand_deploy/config/params.yaml) for IMU/receiver serial identities, signs, steering zero assumptions and servo poses. These are specific to this bike, not universal defaults.

### Start nodes individually: MIT steering, dry-run first

Run **one command per terminal**, each from `~/trackstand_rl/deploy` with ROS and `install/setup.bash` sourced. Start the policy last:

```bash
ros2 run trackstand_deploy kickstand_driver --ros-args --params-file source/trackstand_deploy/config/params.yaml
ros2 run trackstand_deploy imu_publisher --ros-args --params-file source/trackstand_deploy/config/params.yaml
ros2 run trackstand_deploy rc_kickstand --ros-args --params-file source/trackstand_deploy/config/params.yaml
ros2 run trackstand_deploy motor_mit_io --ros-args --params-file source/trackstand_deploy/config/params.yaml -p dry_run:=true
ros2 run trackstand_deploy policy_controller --ros-args --params-file source/trackstand_deploy/config/params.yaml
```

`dry_run` suppresses **motor actuation/configuration writes**, not servo commands, and does not disable a previously active motor. Keep the switch down during initial data-flow checks. Follow the staged procedure to prevent physical stand lifting during inference-only testing.

For a supported powered test **after commissioning**, stop the dry-run motor node and restart only that node with `-p dry_run:=false`. MIT parameters are startup-only; a runtime parameter change is not an emergency stop.

| Motor executable | Steering control | Rear control |
|---|---|---|
| `motor_io` | Host PD → C0 current | C0 current, v1 pure torque feedforward |
| `motor_io_position` | Firmware C2 absolute position | Same C0 rear path |
| `motor_mit_io` | Firmware MIT position-PD | Same C0 rear path |

Run **exactly one** backend, never alongside standalone selected-motor CAN writers/recorders. The all-node launch is `ros2 launch trackstand_deploy trackstand.launch.py`: it now selects **`motor_mit_io`**, and uses the checked-in `dry_run: false` setting. It is not a dry-run shortcut.

The bundled deployment default is **2026-10-21_11-18-29 / model_2500**, the DR-trained v1 Beta actor, in a self-contained ONNX file checked by SHA-256. See [policy_controller.py](deploy/source/trackstand_deploy/trackstand_deploy/policy_controller.py) and the [policy manifest](deploy/source/trackstand_deploy/models/v1_policy_manifest.json). Another compatible v1 export requires both `model_path` and matching `model_sha256` startup parameters. Old 13-input 41050/91850 actors are rejected.

### Operating contract and important limits

Policy inference is **50 Hz**. Isaac physics is **400 Hz**; MuJoCo physics/PD defaults to **400 Hz**, configurable with `--physics_hz`. MIT deployment sends steering MIT/rear C0 at nominal **200 Hz** and queries/publishes motor feedback at **100 Hz** (state only when new/fresh). These rates do not establish the motor firmware's undocumented internal-loop frequency.

The 11 observations are projected gravity YZ, body gyro XYZ, steering/rear position, steering/rear speed, and steering sine/cosine; there are no linear-velocity estimates or finite ±100 clamp. Motor state v3 contains continuous signed encoder displacement. Policy rear displacement is re-zeroed per RC-enabled episode, then adapted to the measured PhysX 4π angle branch. Steering keeps its physically straight reference. Beta outputs are inherently bounded in [-1,1]. Steering follows ×1.57 → ±1.57 clip → at most 0.478 rad per 50-Hz update → hold. Rear follows ×6.28 → ±6.28 N m torque cap → C0 current, with controller damping **0** and no duplicate passive-friction compensation.

Startup holds balancing disabled. RC up authorizes inference when data are ready; RC down requests motor disable and stands down. This does not wait for physical motor-stop confirmation. Fault recovery depends on the selected backend; hard MIT faults require restart.

The policy now uses v1's approximately 30° fallen/inversion check and requires steering within 5° of its fixed straight origin before starting an episode. The total physical MIT PD-torque/current cap is still unverified; F0 encoding ranges do not reproduce Isaac's ±5.6 N m steering clamp. MIT also retains its lack of measured overtravel/overspeed stops, unlike the legacy backends. Firmware timing, motor speed constraints, sensor synchronization and physical contact remain validation gaps. See the [v1 migration report](deploy/experiments/reports/20261026_v1_deployment_alignment/README.md) and [MIT guide](deploy/docs/MOTOR_MIT_IO.md); dated old audits describe historical snapshots.

Record raw actions, commands, IMU, motor state/control/status/fault and policy timing/events using the commands in the [telemetry guide](deploy/docs/TELEMETRY.md). Motor torque feedback is current × measured torque constant, not a calibrated shaft-torque measurement. The motor node publishes telemetry but does not automatically save a rosbag.

## Tests, reproducibility and contributions

From the repository root, with the corresponding Python dependencies installed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/source/trackstand_deploy \
  python3 -m pytest -q -p no:cacheprovider deploy/source/trackstand_deploy/test
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s deploy/experiments/tests
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s mujoco_sim2sim/tests
```

These are offline tests, not permission to enable hardware. The current deployment suite has known fixture/safety-expectation failures documented in the audit; historical passing counts do not certify the current source. Isaac-specific tests require the Isaac Lab interpreter, for example `~/IsaacLab/isaaclab.sh -p -m unittest discover -s tests` from `isaaclab_train/` with its import path configured.

Follow [AGENTS.md](AGENTS.md). For a behavior change, include the affected component, checkpoint/hash, effective configuration, DR setting, seed, commands and quantitative results. Submit a GitHub issue or focused pull request; include logs/plots where appropriate, but do not unintentionally commit secrets, raw bags, checkpoints or generated build trees. Preserve third-party material and its notices.

## Further documentation and licensing

- [Detailed progress and evidence](progress_record.md) and [concise timeline](progress_record.txt).
- [Deployment guide](deploy/README.md), [MIT backend](deploy/docs/MOTOR_MIT_IO.md), [C2 backend](deploy/docs/MOTOR_IO_POSITION.md) and [diagnostic tests](deploy/docs/BIKE_DIAGNOSTIC_TESTS.md).
- [MuJoCo guide](mujoco_sim2sim/README.md) and [Isaac component guide](isaaclab_train/README.md); their dated/template sections are historical, so use current source and this root quick start for active paths.

The root [LICENSE](LICENSE) contains the **GNU General Public License v3**. Included third-party code, SDKs, hardware documents and assets may have their own license notices. This project builds on Isaac Lab, RSL-RL, MuJoCo, ROS 2, ONNX Runtime and the reference Tsinghua bicycle materials; retain the applicable component notices when reusing them.
