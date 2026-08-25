# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CycleRL is an **Isaac Lab (3.x) manager-based RL extension** that trains bicycle
self-balancing and locomotion policies (skrl PPO), then exports them to ONNX/TFLite
for embedded deployment. It is developed as a standalone extension that runs against a
separate Isaac Lab source checkout at `~/IsaacLab`.

The extension package lives in `source/CycleRL/CycleRL/`. The `pyproject.toml` at the
repo root is Isaac Lab's own dev pyproject (vendored config), **not** this project's —
the project is installed with `pip install -e source/CycleRL`.

The two PDFs and `README.md` are background/reference. The README is the stock Isaac Lab
template and is **stale** (it references a `Template-` task prefix and bare `python
scripts/...`); use the commands below instead.

## Running things

Everything runs through Isaac Lab's launcher, which supplies the Isaac Sim Python
interpreter. `-p` runs a script with that interpreter:

```bash
# list registered CycleRL envs
~/IsaacLab/isaaclab.sh -p scripts/list_envs.py

# train (headless is much faster; scale envs to your GPU)
~/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task CycleRL-NewBike-v0 \
    --headless --num_envs 4096 --terrain stairs --max_iterations 5000

# play + full evaluation metrics (checkpoint is a skrl best_agent.pt)
~/IsaacLab/isaaclab.sh -p scripts/skrl/play_with_metrics.py --task CycleRL-NewBike-v0 \
    --checkpoint logs/skrl/cyclerl_logs/CycleRL-NewBike-v0/<run>/checkpoints/best_agent.pt \
    --eval_suite nominal,kick,mass,friction,training --num_envs 64 --headless

# sanity-check an env with dummy policies
~/IsaacLab/isaaclab.sh -p scripts/zero_agent.py   --task CycleRL-NewBike-v0
~/IsaacLab/isaaclab.sh -p scripts/random_agent.py --task CycleRL-NewBike-v0

# export a trained policy for embedded use
python scripts/deployment/export_onnx.py --checkpoint <best_agent.pt> --obs-dim 13 --action-dim 2
```

Training logs/checkpoints go to `logs/skrl/cyclerl_logs/<task>/<run_name>/checkpoints/`.

`--terrain` (`flat` | `complex` | `stairs`) is a preset selector shared by train/play;
`flat` is the default when omitted. `scripts/skrl/train.py` and the `play*.py` scripts
carry a `DeprecationWarning` but are the working entry points.

Lint/format (ruff, line length 120):

```bash
pre-commit run --all-files
```

## Architecture

### Task registration
`source/CycleRL/CycleRL/tasks/manager_based/cyclerl/__init__.py` registers every gym env
(`CycleRL-NewBike-v0`, `NewBike-Balance-v0`, `CycleRL-THUBike-v0`, `THUBike-Balance-v0`,
`THUBike-Trackstand-v0`, `CycleRL-Flywheel-v0`, …). All point at the stock
`isaaclab.envs:ManagerBasedRLEnv` with a project `env_cfg_entry_point` and the single
shared agent config `agents/skrl_ppo_cfg.yaml` (Gaussian MLP `[512,256,128,64]` ELU).

### Two task families — this is the main structural axis
- **Dynamic / locomotion** (`*/cyclerl_env_cfg.py`): the bike rides forward and tracks a
  velocity + heading command. Has a `commands` manager, terrain curriculum, spawn velocity.
- **Stationary / balance** (`*/*balance_env_cfg.py`, `thubike_trackstand_env_cfg.py`): keep
  the bike upright in place, no commands, strong position-keeping penalty.

The MDP terms mirror this split in `mdp/`:
- `mdp/dynamic/` and `mdp/stationary/` each hold their own `events / rewards / observations
  / terminations / curriculums`.
- `mdp/__init__.py` re-exports all stock Isaac Lab mdp terms plus the few genuinely shared
  helpers (`randomize_ground_plane_material`, `action_rate_penalty`). When adding an MDP
  term, put it in the matching family package, not the shared top level.

### Multiple bike hardware, one MDP codebase
`bicycle_cfg.py` defines the robot `ArticulationCfg`s for each physical bike (Xuan bike
variants, THU bike, New bike) — USD path, actuator PD gains, and effort/velocity limits
tuned to real motors (e.g. GIM4310-10, GIM3510-8). Per-bike env configs live in sibling
subpackages (`new_bike/`, `thu_bike/`, `xuan_bike/`). The bike-specific surface is just the
`ArticulationCfg` + USD + joint names; reward/observation logic is shared across bikes.
USD assets are under `.../cyclerl/assets/<bike>/` (payload-based `.usda`).

### Sim-to-real contract
Getting the same policy to run on hardware drives several design choices:
- `config/motor_contract.yaml` records real motor params; `eval_contract.py`
  (`BikeEvalContract`) resolves joint names, action slices, scales, clips, and effort
  limits **from the live env** rather than hardcoding USD joint names — so eval analyzers
  work across bikes. Prefer this resolution path over string-matching joints.
- Observations are aligned between the real motor and sim (see git history). A NewBike
  policy is `obs-dim 13`, `action-dim 2` (steering = joint-position action, drive =
  joint-velocity action).

### Terrain
`config/` holds custom terrain generators (`bicycle_terrain_stairs.py`,
`bicycle_terrain_complex.py`, `blended_wave_terrain.py`) plus a custom
`TriangleMeshTerrainImporter` and `terrain_presets.py` (maps `--terrain` → generator, and
`complex_L0..L4` / `stairs_L0..L2` eval stages that pin every env to one difficulty row).
Curriculum terms progressively expand spawn velocity, command-heading range, IMU noise, and
terrain difficulty over training steps.

### Evaluation framework
`eval_*.py` in the cyclerl package is a substantial, bike-agnostic eval harness driven by
`play_with_metrics.py`: `eval_logger.py` (`BicycleMetricsLogger`), `eval_scenarios.py`
(named perturbation suites — kick, mass, friction, …), `eval_failure.py`, `eval_contract.py`,
and `eval_csv_plots.py` / `scripts/analyze_eval_csv.py` for post-hoc CSV analysis.

### `sim_pid_control/`
Standalone Isaac Sim scripts (`cascade_2pid.py`, `cascade_3pid.py`, `pid_statespace.py`) —
classical PID/state-space balance baselines. These launch Isaac Sim directly (not through
the RL env stack) and are independent of the gym tasks.
