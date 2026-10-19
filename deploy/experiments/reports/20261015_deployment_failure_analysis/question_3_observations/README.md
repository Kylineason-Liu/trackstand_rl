# Observation-contract investigation — 91850

See the [complete Question 3 report](../03_observation_contract_comparison.md) and [figure gallery](index.html).

The key controlled result is that replacing only true body vx/vy with deployment's wheel-speed estimate/zero can break the same policy inside Isaac and MuJoCo. These experiments do not change runtime source, geometry or hardware.

## Files

- `capture_isaac.py`: instruments unchanged `play_eval.py` to record exact actor inputs, raw actions, corresponding true root/joint state, bias and reset flags. `--deploy-velocity` opts into the two-channel, in-memory closed-loop ablation. Synthetic rear-angle backend probes run **after** the saved evaluation.
- `isaac_{no_dr,dr}_gpu/`: unchanged baseline captures. `isaac_{no_dr,dr}_estimator_gpu/`: observation-substitution captures. Each has `observations.npz`, effective configuration, experiment metadata, evaluator summary, source-integrity result and rear-angle probe. Copied checkpoints isolate evaluator outputs from training logs. Legacy `torque_profile` files are the evaluator's diagnostic estimates, **not independently measured motor torque**.
- `analyze.py`: matched-state source-function parity tests; actual MuJoCo rollouts; fixed-input model replays; statistical analysis and figures. It never opens CAN or starts ROS.
- `mujoco_observations.npz`: 64 seeds × 3,000 actions, nominal and DR. `mujoco_estimator_ablation.npz` and `mujoco_phase_ablation.npz`: eight matched seeds × 3,000 actions/mode. The latter tests a candidate 4π-period rear-angle observation, not a proven general PhysX emulator.
- `summary.json`, `feature_distributions.csv`, `counterfactual_actions.npz`: numerical evidence; `verification.json`: independent replay/count/capture checks. Baseline source/input hashes and report validation are in `validation.json`.
- Graphs are available in PNG/SVG/PDF: `observation_distributions`, `velocity_substitution_effect`, `isaac_velocity_mismatch_example`, `rear_phase_operating_regions`, `closed_loop_observation_ablation`, `isaac_closed_loop_observation_ablation`.

## Reproduction

Run from the repository root, with the compatible Isaac Sim environment and the original checkpoint present. Use a **new output directory** for each capture; existing captures are never overwritten.

```bash
cd isaaclab_train
~/IsaacLab/_isaac_sim/python.sh \
  ../deploy/experiments/reports/20261015_deployment_failure_analysis/question_3_observations/capture_isaac.py \
  --capture-dir /tmp/new_91850_observation_capture \
  --task Trackstand-Balance-v0 --num_envs 64 --num_steps 3000 \
  --seed 42 --dr --headless --device cuda:0 agent.max_iterations=500000
```

Omit `--dr` for nominal. Add `--deploy-velocity` for the observation-only closed-loop ablation. This changes the experimental observation calculation, not source files; do not confuse it with an unchanged baseline evaluation. The YAML records the base configuration; the process-local observation-manager override is recorded by the `deploy_velocity_ablation` metadata flag and invocation, and implemented in the capture script. The six-state post-evaluation backend probe needs at least six environments.

For analysis, Python needs NumPy, Torch, ONNX Runtime, MuJoCo, Matplotlib and Pillow. The actor export accepts dynamic batches. Isaac capture needs the installed Isaac Lab/Sim stack and GPU access; CPU parity probes load only the installed quaternion helper and the actual task observation function, not an Isaac application.

```bash
MPLCONFIGDIR=/tmp/trackstand_observation_mpl python3 \
  deploy/experiments/reports/20261015_deployment_failure_analysis/question_3_observations/analyze.py --collect-mujoco
MPLCONFIGDIR=/tmp/trackstand_observation_mpl python3 \
  deploy/experiments/reports/20261015_deployment_failure_analysis/question_3_observations/analyze.py --collect-phase
MPLCONFIGDIR=/tmp/trackstand_observation_mpl python3 \
  deploy/experiments/reports/20261015_deployment_failure_analysis/question_3_observations/analyze.py --analyze
MPLCONFIGDIR=/tmp/trackstand_observation_mpl python3 \
  deploy/experiments/reports/20261015_deployment_failure_analysis/question_3_observations/verify.py
```

Collection refuses existing NPZ outputs; use a fresh copy/location of this experiment bundle for a new collection. The analyzer expects the four Isaac directories named as listed above. `--refresh-report` explicitly regenerates this script's own summary/figures/counterfactual arrays, never raw captures. `verify.py` writes only its verification result.

No-DR still includes the task's white observation noise. Different platform termination rules and the bike's stand contact mean that cross-platform input distributions are **not matched physical trajectories**. Fixed-input replays are not alternate physical runs; the separately labelled closed-loop ablations are actual simulation runs. There is no independent real-bike body-velocity ground truth in the input bag.
