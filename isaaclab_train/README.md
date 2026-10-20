# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Trackstand domain randomization

`Trackstand-Balance-v0` uses nominal physics, upright/zero-roll-rate resets, and
zero IMU residual biases by default. Observation white noise remains enabled.
Add `--dr` to RSL-RL training, deterministic training, play, or evaluation to
enable the existing full per-environment, per-episode randomization ranges:

```bash
python scripts/rsl_rl/train.py --task Trackstand-Balance-v0
python scripts/rsl_rl/train.py --task Trackstand-Balance-v0 --dr
python scripts/rsl_rl/play_eval.py --task Trackstand-Balance-v0 \
  --checkpoint /path/to/model.pt --num_envs 64 --num_steps 3000 --seed 42 --dr
```

There is no DR ramp. The effective setting is saved as `domain_randomization`
in the environment configuration. This switch does not change rewards,
termination limits, observation white noise, or the 400 Hz physics/50 Hz policy
timing. Omit `--dr` for nominal evaluation; append it when comparing with prior
full-DR training/evaluation records.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/CycleRL

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/CycleRL/CycleRL/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Trackstand-Balance-v1: bounded Beta policy

`Trackstand-Balance-v1` uses an RSL-RL Beta actor. Both sampled training actions
and deterministic evaluation/export actions lie in `[-1, 1]`. The four internal
MLP outputs parameterize two Beta distributions; they are **not** motor commands.
The two resulting actions retain v0's scales: steering position x `1.57` rad,
rear-wheel torque x `6.28` N m.

V1 steering additionally clips the scaled position request to `[-1.57, 1.57]`
rad, then limits target change to `23.9 * 0.02 = 0.478` rad per **50-Hz policy
update**. The target is held for all eight 400-Hz physics substeps. The limiter
resets per environment from measured steering position, clipped to the target
range. V0's steering action is unchanged.

V1's rear drive uses `IdealPDActuator` with zero stiffness and zero target
velocity. At each **400-Hz physics step**, it computes
`clip(6.28 * drive_action - Kd * rear_speed, -6.28, 6.28)` N m from the held
50-Hz action. Nominal `Kd=0.05` N m s/rad remains a **placeholder**, not an
identified passive-friction coefficient. Set it in v1's `__post_init__()` in
`trackstand_bike/trackstand_rl_env_cfg_v1.py`. Gain DR still scales this Kd by
`U(0.9, 1.1)` when enabled. `effort_limit=6.28` provides the explicit
post-damping clamp; the existing `effort_limit_sim=6.28` and
`velocity_limit_sim=61.6` rad/s are retained. The framework disables native rear
PD gains, so damping is not added twice. V0 retains its implicit rear actuator.

Both observation functions live in `mdp/stationary/observations.py`. V1 uses
the independent `balance_imu_state_v1()` function with 11 channels: projected
gravity Y/Z, body angular velocity XYZ, steering/rear positions, steering/rear
velocities, and steering sine/cosine.
It never reads body linear velocity and has its own matching noise configuration.
V0 retains its original 13-channel `balance_imu_state()` function in the same file,
including body linear velocity X/Y, with its original 13-entry noise tuple and
Gaussian policy. Editing v0's observation layout does not change v1's layout.
Apart from these observation and motor-command changes, environment settings
and PPO hyperparameters inherit v0, including
400 Hz physics, 50 Hz policy, unchanged noise on retained channels and opt-in
`--dr`. V1 intentionally has no skrl entry: the existing skrl configuration is
Gaussian-only.

Run from `isaaclab_train/` using the documented Isaac Lab Python environment:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Trackstand-Balance-v1 --num_envs 4096 --seed 42 --headless
# Append --dr for randomized training. Train a NEW policy; do not resume a v0 checkpoint.
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_eval.py \
  --task Trackstand-Balance-v1 --checkpoint /path/to/new_beta_model.pt \
  --num_envs 64 --num_steps 3000 --seed 42 --headless
~/IsaacLab/isaaclab.sh -p scripts/deployment/export_rsl_rl_onnx.py \
  --checkpoint /path/to/new_beta_model.pt --output /path/to/beta_policy.onnx
```

V1 requires RSL-RL >= 5.0 (the current installation is 5.0.1); its Beta
distribution is project-local, so no RSL-RL upgrade or vendor edit is needed.
Run the CPU contract/PPO/export tests with:

```bash
~/IsaacLab/isaaclab.sh -p -m unittest discover -s tests -p 'test_trackstand_*.py' -v
```

The standalone exporter detects v1 from its checkpoint marker and includes the
bounded mean transform. Native RSL-RL JIT/ONNX exports also include it. This does
not update MuJoCo or Raspberry Pi's current 13-input observation interfaces:
those must be matched to the new 11-input policy before transfer.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/CycleRL"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
