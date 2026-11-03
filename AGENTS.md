# Repository Guidelines

## Project Structure & Module Organization

This repository develops and deploys a self-balancing bicycle controller across three areas:

- `isaaclab_train/` contains Isaac Lab reinforcement-learning environments. The active extension is `source/trackstand_rl/trackstand_rl/`; task configuration and MDP terms live beneath `tasks/manager_based/trackstand/`. Training, evaluation, and export entry points are in `scripts/`.
- `mujoco_sim2sim/` validates exported policies in MuJoCo. Keep simulation code in `src/newbike_mujoco/`, executable helpers in `scripts/`, XML and meshes in `models/`, and policy artifacts in `policies/`.
- `tsinghua_bike_project/` holds hardware documentation, CAD, ROS2 materials, and the nominal-control teaching package. Its Python unit tests are under `assign4&5_bike_mujocosim_with_control/.../tests/`.

Do not edit generated build directories, training logs, checkpoints, or vendor archives unless the task explicitly requires it.

## Build, Test, and Development Commands

Run commands from their component directory:

```bash
cd isaaclab_train && python scripts/list_envs.py
cd isaaclab_train && python scripts/rsl_rl/train.py --task <TASK_NAME>
cd mujoco_sim2sim && pip install -r requirements.txt
cd mujoco_sim2sim && python scripts/run_deploy_eval.py --episodes 64 --seed 42
cd tsinghua_bike_project/assign4\&5_bike_mujocosim_with_control/assign4\&5_bike_mujocosim_with_control && python -m unittest -v tests.test_nominal_bike_control
```

Isaac Lab requires its documented Python/Isaac Sim environment. Use the current `source/trackstand_rl` path rather than the stale `CycleRL` README reference.

## Coding Style & Naming Conventions

Write Python with four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and explicit units in names or comments where ambiguity is possible (for example, `roll_rad` or `control_dt`). Isaac Lab code uses Ruff with a 120-character limit and Pyright basic checking; run:

```bash
cd isaaclab_train && ruff check source scripts && ruff format --check source scripts
```

Keep environment behavior modular: place a new reward, observation, event, or termination in the corresponding `mdp/` module and register it from configuration rather than embedding it in a training script.

## Testing Guidelines

Add focused tests alongside the nominal-controller tests for deterministic control logic. Name test files `test_*.py` and test methods `test_*`. For simulation changes, run an appropriate headless evaluation and report the command, seed, and key outcome; do not treat a visual viewer run as sufficient verification.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, often naming the affected model or fix (for example, `Fix quaternion-order bug in yaw terms`). Keep commits scoped and explain configuration or physics changes in the body. Pull requests should describe the affected simulator/hardware path, list validation results, link related issues or experiments, and include plots or screenshots for behavior changes. Never commit secrets, rosbags, or generated checkpoints unintentionally.

## Codex Multi-Agent Workflow

The root Codex session acts as `main_coordinator`. For substantial implementation,
evaluation, or cross-component audits, delegate applicable independent work to
these project roles; small questions and simple edits may stay in the main session.
Read `docs/codex_multi_agent.md` for handoffs, evidence and resource ownership.

| Role | Responsibility |
| --- | --- |
| `main_coordinator` | Scope, assignments, evidence review, shared-resource scheduling and final response |
| `implementer` | Assigned production-code changes and focused tests |
| `isaaclab_evaluation_analyst` | Isaac Lab evaluation and simulation evidence |
| `mujoco_evaluation_analyst` | MuJoCo evaluation, survival statistics and sim-to-sim analysis |
| `deploy_evaluation_analyst` | Deployment/bench logs, CAN/ROS telemetry and motor response |
| `summarizer` | Final evidence synthesis and factual handoff for recording |
| `recorder` | Automatically record material unrecorded findings, progress and conclusions in both ledgers |

Role definitions live in `.codex/agents/*.toml`; settings live in `.codex/config.toml`.
Do not spawn a redundant coordinator or all roles for every task. Specialists
report to the main coordinator; delegate further only when explicitly assigned.
Give each assignment a bounded objective, evidence inputs, owned output paths,
validation scope and resource allocation. Parallel work must have disjoint writes.
Analysts own new report artifacts/helpers; runtime fixes go to the implementer.
The recorder is the sole writer of both progress ledgers; summarizer does not edit
them. Before concluding completed substantive work, the coordinator automatically
delegates to recorder to check and append material unrecorded findings, progress
and conclusions in both existing formats, then reviews its diff and preservation
checks. No separate recording request is needed. Record completed milestones
during long tasks when useful. If nothing material is new, leave both files
unchanged; do not record trivial exchanges or repeated checks. Explicit user
instructions against recording or file changes take precedence. Recorder runs
within the task workflow, not as an independent background watcher.

Read current source and matching raw metadata before applying dated progress
conclusions. Return evidence paths, exact commands/conditions, completed sample
counts, verification and limitations. Preserve raw inputs and prior reports.
The coordinator schedules shared GPU/CPU jobs and any already authorized live
hardware operations; deployment analysis is offline by default. Worktrees do not
isolate GPUs or CAN. Role instructions inherit session permissions and do not
authorize additional hardware operations.
