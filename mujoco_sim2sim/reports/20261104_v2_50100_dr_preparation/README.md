# Model 50100: v2 DR sim2sim preparation

Prepared 2026-11-04 directly, without subagents. The existing v1/v2 runtimes,
default model_26600, verified bike XML/meshes and training artifacts are unchanged.

## Policy and configuration

- Source: `isaaclab_train/logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_22-40-01/model_50100.pt`.
- Checkpoint SHA256: `32e43762691fcd25a8e72182339557f0f71f43c14cde2106bb71118eb5019af1`.
- Export: [`trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx`](../../policies/trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx).
- ONNX SHA256: `a43583ec06d3c441d74d93057378fc25d8d79255a4ad312aeca6f59ad1c25c33`.
- Self-contained ONNX opset 18, dynamic `(batch,11)` → `(batch,2)`, deterministic
  Beta means in `[-1,1]`, no observation normalization or baked action scales.
- Adjacent `.contract.json` binds the export hash, source checkpoint/config hashes
  and v2 rear-velocity PI contract. The loader discovers it automatically.

The saved DR run matches the v2 backend: policy/observations/target slew **50 Hz**,
physics/explicit PD **1200 Hz**, rear integral **200 Hz**. Steering uses scale
1.57 rad, slew 23.9 rad/s, PD gains 10/0.2 and torque cap 5.6 Nm. Rear uses
velocity scale 61.6 rad/s, velocity P=0.30, Ki=3.3 and final torque cap 6.28 Nm.
The export helper's generic final log message describes the older torque-action
tasks; its graph contains no scaling. For v2 the rear scale means **velocity**.

## Start the viewer with DR

```bash
cd ~/trackstand_rl/mujoco_sim2sim
python3 -u scripts/run_deploy_viewer_v2.py \
  --task deploy_sim \
  --onnx policies/trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx \
  --num_envs 64 --seed 42 --physics_hz 1200 --dr
```

This uses `models/trackstand_rl_0916_bike/trackstand_rl_0916.xml`, all six normal
termination terms and independent respawns. Close the viewer to save completed
episode statistics, the episode-length histogram and torque plots in `logs/v2/`.

## Reproducible headless evaluation

```bash
cd ~/trackstand_rl/mujoco_sim2sim
python3 -u scripts/run_deploy_eval_v2.py \
  --onnx policies/trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx \
  --num_envs 64 --episodes 64 --seed 42 --physics_hz 1200 --dr
```

This requests one completed episode per environment with a 30-second horizon.
It saves full traces, per-episode DR draws, summary JSON/log, ten-bin episode
histogram and torque graph under `logs/v2/YYYYMMDDHHMM_50100_dr_*`.
Omit `--dr` only for a nominal-environment evaluation of this same DR-trained
policy. Do not add `--pure_survival` for the standard task comparison.

To use PT directly, replace `--onnx ...` with
`--pt ../isaaclab_train/logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_22-40-01/model_50100.pt`.
Its original adjacent saved YAML is discovered automatically. This option was
also checked against the ONNX inference path.

## Completed preparation checks

- Saved action/actuator/observation, timing, DR-range and fixed-gain checks pass.
- ONNX checker passes; CPU ONNX Runtime agrees with CPU PT on **1,505 inputs**
  including batches 1/4/64/257, random input scales and real DR observations.
  Maximum absolute action difference: **6.556510925292969e-6**, tolerance `1e-5`.
- Inference integration: four DR environments ×50 policy steps, **200 environment
  transitions /4,800 physics steps**, two resets; finite bounded actions.
- Native CLI smoke: four DR environments ×50 policy steps, seed 42, exit 0;
  expected metadata, full trace, histogram, torque and summary files generated
  under [`cli_smoke/`](cli_smoke/).
- The native short smoke completed two episodes, at **0.36 s (fallen)** and
  **0.86 s (yaw_deviation)**; four trajectories remained unfinished at the end
  of its finite step budget. This is an integration check with incomplete
  trajectories, **not a full DR performance evaluation**. No GUI was opened.
- All **67 protected existing files** retain their hashes, including checkpoint,
  configs, prior policies, simulator code and geometry. No deployment/CAN work.

The existing MuJoCo approximations still apply: one sliding-friction coefficient,
different contact/integration semantics and default hard speed projection versus
PhysX's soft velocity constraint. Preparation does not remove those differences.

## Evidence and reproduction

[`verification.json`](verification.json), [`inference_comparison.npz`](inference_comparison.npz),
[`verification.log`](verification.log), [`export.log`](export.log),
[`cli_smoke.log`](cli_smoke.log), and protected before/after SHA manifests retain
the checks. [`commands.json`](commands.json) records the commands and conditions.

From the repository root, reproduce the export using the existing exporter:

```bash
~/IsaacLab/_isaac_sim/python.sh -B isaaclab_train/scripts/deployment/export_rsl_rl_onnx.py \
  --checkpoint isaaclab_train/logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_22-40-01/model_50100.pt \
  --output mujoco_sim2sim/policies/trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx
~/IsaacLab/_isaac_sim/python.sh -B mujoco_sim2sim/reports/20261104_v2_50100_dr_preparation/pack_export.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B mujoco_sim2sim/reports/20261104_v2_50100_dr_preparation/verify_preparation.py
```

The final check refreshes the hash-bound policy sidecar and inference evidence.
The protected baseline covers pre-existing inputs, not the newly generated export.
