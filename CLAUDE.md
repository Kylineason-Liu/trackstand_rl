# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CycleRL was removed on 2026-09-23. The repo now centers on **Trackstand-Balance-v0**,
a standalone Isaac Lab (3.x) manager-based RL extension that trains bicycle
trackstand self-balancing policies (rsl_rl PPO), then exports them to ONNX for the
MuJoCo deployment viewer (`~/newbike_mujoco_deploy/`, a separate project that mirrors
the task's deploy contract).

The task package lives in `source/trackstand_rl/` and runs against the separate Isaac
Lab source checkout at `~/IsaacLab`. It has ZERO imports from the removed CycleRL
package (verified). The root `pyproject.toml` is Isaac Lab's own dev pyproject
(vendored config), not this project's.

The `sim_pid_control/` directory holds standalone classical-control baselines
(Isaac Sim scripts, independent of the RL stack). `progress_record.md` at the repo
root is the consolidated progress log — new entries are appended there.

## Running things

Everything runs through Isaac Lab's launcher from the repo root:

```bash
# train (headless; deterministic seed 42)
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_deterministic.py \
    --task Trackstand-Balance-v0 --max_iterations 500000

# deterministic eval of a checkpoint (full DR from step 0)
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_eval.py --task Trackstand-Balance-v0 \
    --checkpoint logs/rsl_rl/trackstand_rl_rsl/<run>/model_<N>.pt \
    --num_envs 64 --num_steps 3000 --seed 42

# export an rsl_rl checkpoint for the MuJoCo deploy viewer
~/IsaacLab/isaaclab.sh -p scripts/deployment/export_rsl_rl_onnx.py \
    --checkpoint logs/rsl_rl/trackstand_rl_rsl/<run>/model_<N>.pt
```

Training logs/checkpoints go to `logs/rsl_rl/trackstand_rl_rsl/<run>/`; the saved
`params/env.yaml` + `params/agent.yaml` and the `git/` diff snapshot record exactly
what each run launched with. Hydra overrides use the `env.` node (e.g.
`env.sim.dt=0.01 env.decimation=2`).

## Architecture

### Task registration
`source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/__init__.py`
registers `Trackstand-Balance-v0` (env class `TrackstandRLEnv`, cfg class
`TrackstandRLEnvCfg`). Registration is triggered by
`import source.trackstand_rl.trackstand_rl.tasks` — the training/eval scripts carry
that import directly (it used to piggyback on CycleRL's aggregation import).

### The single env config (no variants exist)
`trackstand_bike/trackstand_rl_env_cfg.py` — `TrackstandRLEnvCfg`:
- **Control contract (decided)**: steering = joint POSITION servo (kp 10 / kd 0.2,
  action scale 1.57), drive = joint EFFORT torque (stiffness 0 / damping 0.05,
  action scale 6.28). Robot cfg: `../bicycle_cfg.py` `TRACKSTAND_ROBOT_CFG`.
- 400 Hz physics (dt 0.0025, decimation 8, render_interval 8), 50 Hz control,
  30 s episodes, TGS solver 4/2 + `enable_external_forces_every_iteration=True`.
- 13-dim obs with always-on per-channel white noise (`enable_corruption`);
  rewards: survival 5.0 / drift_gauss 2.0 (σ 0.25) / rate_penalty 0.1.
- Full per-episode DR: mass ×U(0.8,1.2), COM ±5 mm, actuator gains ×U(0.9,1.1),
  initial state roll U(±0.10)/roll-rate U(±0.25), IMU turn-on residuals,
  and **friction**: tire material carries `frictionCombineMode="min"` with the
  ground pinned at 1.0, so the tire value IS the ground-wheel pair — nominal
  **0.65** (measured real-environment value), DR U(**0.5, 0.8**), 32 buckets.
- The mdp library lives in `mdp/` (stationary + dynamic — the mdp copy must stay
  complete: `mdp/events.py`'s module-level `__getattr__` is probed with `__path__`
  by the import machinery).

### MuJoCo deploy mirror
`~/newbike_mujoco_deploy/` (separate project): `src/newbike_mujoco/env.py` mirrors
the contract (including the DR terms behind `--dr`), and
`scripts/run_deploy_viewer.py` augments the Fusion2Mujoco export (joint renames,
actuators, timestep, floor 0.1 pass-through + tire 0.65, stiff solref 0.005/1.5).
CAUTION: the export carries explicit per-geom friction/solref that override
default-class edits — the augmentation must loop over the geoms explicitly.

### Deploy policies
Exported ONNX files live in `scripts/deployment/` (naming:
`trackstand_rl_rsl_<run>_model_<N>.onnx`). `export_rsl_rl_onnx.py` defaults to
opset 18 (opset 13 fails under torch 2.10 dynamo).

### Lint/format
```bash
pre-commit run --all-files
```
