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

The root Codex session acts as `main_coordinator`. Read
`docs/codex_multi_agent.md` for the assignment/report contract and loop protocol.
Classify substantial user work and route only applicable tasks:

| Task | Agent |
| --- | --- |
| Code/configuration/protocol/methodology audits | `inspector` |
| Implementation and focused tests | `implementer` |
| Isaac Lab evaluation or evaluation analysis | `isaaclab_evaluation_analyst` |
| MuJoCo evaluation or evaluation analysis | `mujoco_evaluation_analyst` |
| Deployment evaluation or analysis | `deploy_evaluation_analyst` |
| Collect worker reports, synthesize criteria and final report | `summarizer` |
| Automatically record accepted material progress in both ledgers | `recorder` |

Before delegation, coordinator identifies `single_pass` or `iterative` intent,
defines the target and required acceptance criteria, and gives summarizer the
expected task/report set and iteration/candidate identity. Workers parse their
assignments, do their owned work and send complete evidence/criteria reports to
the designated summarizer, with coordinator delivery receipts. When peer messaging
or slots are unavailable, coordinator explicitly relays full reports and verifies
delivery. Missing reports and unrun/stale checks cannot silently become passes.

In iterative work, summarizer returns a criterion table, conclusions and a
continue/achieved/blocked recommendation; coordinator judges whether the target
is achieved. If unmet, coordinator assigns the next focused improvement and
revalidation. Reuse threads/iterations rather than recursively spawning new
coordinators. Never weaken criteria or equate stopping/blocked work with success.
In single-pass work, summarizer authors the final user-ready report. The root
relays it after recording because only the root delivers the CLI user-facing
final; do not produce a competing synthesis. Small questions can stay in root.

There are eight configured roles: one coordinator and seven subagent roles.
Role definitions live in `.codex/agents/*.toml`; settings live in
`.codex/config.toml`. Do not spawn redundant coordinators or all roles for every
task. Give bounded objectives, evidence inputs, owned paths, validation scope,
resources and dependencies. Parallel writes must be disjoint. Analysts own new
report artifacts/helpers; implementer owns assigned production fixes. Inspector
owns audits and assigned new audit reports, not production code changes.

Recorder alone writes both progress ledgers; summarizer never edits them. Before
concluding substantive completed work, coordinator automatically passes accepted
summaries/decisions to recorder, then reviews its diff and preservation checks.
No separate recording request is needed. Record useful completed milestones;
no-op if nothing material is new, and do not recursively record recording itself.
Explicit user no-record/read-only instructions take precedence. Recorder has no
loop-control authority and is task-bound, not an independent background watcher.

Read current source and matching metadata before applying dated conclusions.
Return exact commands/conditions, evidence, completed sample counts, criterion
data/status and limitations. Preserve raw inputs, prior reports and user changes.
Coordinator schedules shared GPU/CPU and already authorized live hardware work;
deployment analysis is offline by default. Worktrees do not isolate GPU/CAN.
All roles inherit session permissions and do not grant hardware authorization.
