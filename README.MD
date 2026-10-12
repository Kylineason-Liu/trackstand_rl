# Trackstand RL

Reinforcement-learning-based **stationary bicycle balancing**, from Isaac Lab training to MuJoCo sim-to-sim validation and Raspberry Pi deployment with ROS 2.

The active task is `Trackstand-Balance-v0`. A policy commands steering **position** and rear-wheel **torque** to keep the bicycle upright with minimal displacement. This differs from the reference bike's remote-controlled mobile-balancing objective: RC steering/throttle sticks are not policy inputs; the RC switch controls the paired kickstands and authorizes balancing.

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
  --task Trackstand-Balance-v0 --num_envs 4096 \
  --max_iterations 60000 --seed 42 --headless --dr
```

`train_deterministic.py` enables deterministic training settings; `train.py` is the regular variant. Adjust the training budget and environment count to your hardware. Runs are saved beneath `logs/rsl_rl/trackstand_rl_rsl/<timestamp>/`, including effective configuration, TensorBoard events and `model_<iteration>.pt` checkpoints. The last checkpoint is not necessarily the best.

**DR is off by default.** Keep `--dr` for full per-environment/per-episode domain randomization; omit it for nominal physics/reset conditions. There is no DR ramp and no `--no_dr` flag. Observation white noise remains enabled in nominal mode.

Evaluate a checkpoint with matched task/DR settings; replace the placeholder path with an actual checkpoint. Selecting a checkpoint does not automatically restore its historical task configuration: compare the saved `params/env.yaml` before reproducing older experiments.

```bash
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_eval.py \
  --task Trackstand-Balance-v0 --checkpoint /absolute/path/to/model.pt \
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

Run the Trackstand model with the included **09221945 / model_91850** export:

```bash
python scripts/run_deploy_viewer.py --task deploy_sim \
  --mjcf models/trackstand_rl_0916_bike/trackstand_rl_0916.xml \
  --onnx policies/trackstand_rl_rsl_2026-09-22_19-45-06_model_91850.onnx \
  --num_envs 64 --seed 42 --dr
```

Always pass `--onnx` explicitly: the Trackstand fallback paths in the viewer still reference an old checkout location. Use your own export path to validate a different policy; an ONNX export does not automatically update any other component's selected model.

`--num_envs` defaults to **64**; choose `1` for a single-bike view. `--dr` enables randomized resets; omit it for nominal runs. `--speed` controls playback pacing, not policy frequency. Inspect all options with `python scripts/run_deploy_viewer.py --help`.

On normal viewer exit, available episode metrics and steering/rear torque traces are saved under `mujoco_sim2sim/logs/`. The grid is render-only: environments have independent physics/randomization and do not collide with one another. CPU load may make playback slower than real time.

The separate headless command below exercises the **legacy NewBike baseline**, not the Trackstand MJCF viewer configuration:

```bash
python scripts/run_deploy_eval.py --episodes 64 --seed 42
```

See the [MuJoCo component guide](mujoco_sim2sim/README.md) and the progress records for the distinction between legacy and Trackstand experiments.

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
| `motor_io` | Host PD → C0 current | C0 current with host torque conversion/damping |
| `motor_io_position` | Firmware C2 absolute position | Same C0 rear path |
| `motor_mit_io` | Firmware MIT position-PD | Same C0 rear path |

Run **exactly one** backend, never alongside standalone selected-motor CAN writers/recorders. The existing all-node launch is `ros2 launch trackstand_deploy trackstand.launch.py`: it selects **C0 `motor_io`, not MIT**, and uses the checked-in `dry_run: false` setting. It is not a dry-run shortcut.

The bundled deployment default is **09221945 / model_91850 ONNX**, checked by SHA-256 in [policy_controller.py](deploy/source/trackstand_deploy/trackstand_deploy/policy_controller.py). Two commented constants retain the 09171946 / model_41050 alternative. A different model must be deliberately selected with a matching checksum; merely exporting or copying it does not switch the policy.

### Operating contract and important limits

Policy inference is **50 Hz**. Current Isaac/MuJoCo physics is **400 Hz**; MIT deployment sends steering MIT/rear C0 at nominal **200 Hz** and queries/publishes motor feedback at **100 Hz** (state only when new/fresh). These rates do not establish the motor firmware's undocumented internal-loop frequency.

The 13 policy observations use IMU orientation/rates and motor encoders. Longitudinal speed is estimated from rear-wheel speed; lateral speed is assumed zero. Raw actions are scaled by **1.57 rad / 6.28 N·m** without a `[-1,1]` clamp; motor I/O separately bounds requested targets.

Startup holds balancing disabled. RC up authorizes inference when data are ready; RC down requests motor disable and stands down. This does not wait for physical motor-stop confirmation. Fault recovery depends on the selected backend; hard MIT faults require restart.

**Current safety caveats:** the policy has no 30° roll cutoff, only inversion/freshness/fault gates. MIT has no measured steering-overtravel or overspeed stop, and its total internal PD torque/current cap is not established by F0 encoding ranges. Its target clamp does not guarantee physical travel limits. Do not attempt unsupported balancing based only on passing simulation, a false fault flag, or successful CAN writes. Read the [MIT guide](deploy/docs/MOTOR_MIT_IO.md) and [audit](deploy/docs/AUDIT.md).

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
