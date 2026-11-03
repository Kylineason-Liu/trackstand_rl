# Trackstand-Balance-v2: best checkpoint and no-DR evaluation

## Checkpoint selection

Run: `2026-11-02_21-49-32` under `logs/rsl_rl/trackstand_rl_v2_rsl/`.

**Selected checkpoint: `model_26600.pt`.** The exact maximum raw logged
`Train/mean_episode_length` is **1500 policy steps (30 s)**. There are **345 saved
checkpoints tied at this maximum**; iteration 26600 is the latest. The newest
saved checkpoint, 26650, instead logs 1485.969970703125 steps. A total of 26,681
scalar events were inspected without downsampling (iterations 0 through 26680),
and 534 saved checkpoints were eligible. The global logged maximum is also
1500, last seen at unsaved iteration 26668; it is not a larger unattainable peak.

The installed RSL-RL logger computes this scalar from a buffer of the last 100
completed training episodes. No additional TensorBoard smoothing was applied.
This selection criterion does not establish robustness beyond the 30-s horizon
or with domain randomization.

- [Selection metadata and every tied saved iteration](checkpoint_selection.json)
- [All raw scalar events](raw_mean_episode_length.csv)
- [Hash and checkpoint architecture inspection](checkpoint_inspection.json)

The source checkpoint is copied here before evaluation. Its SHA-256 is
`5733e489490c6688c45085970fef46b62d1fe96a90fb7e5035c3b0de5699af7b`.
The original run already contains evaluation artifacts, which are preserved.

## Evaluation result

The requested no-DR evaluation completed: **64 environments × 3000 policy
steps**, seed 42, full task terminations, deterministic Beta mean.

| Metric | Result |
| --- | ---: |
| Completed episodes | 128 (two per environment) |
| Episode length, mean/min/max | 1500 steps / 30.00 s |
| Episode-length standard deviation | 0 steps |
| Timeouts | 128 / 128 (100%) |
| Fallen / diverged / drifted / yaw / wheel-net terminations | 0 each |
| Overlapping termination flags | 0 episodes |
| Unfinished final trajectories with positive duration | 0 |
| Mean episodic return | 207.254 |
| Mean / RMS / maximum absolute roll | 0.65° / 0.66° / 1.58° |
| Drive speed, mean / maximum absolute | 0.30 / 2.06 rad/s |
| Steering speed, mean / maximum absolute | 0.17 / 7.75 rad/s |
| Drift from spawn, mean / maximum | 0.075 / 0.091 m |
| Absolute heading deviation, mean / maximum | 9.90° / 13.61° |

- [Unchanged evaluator summary](eval.txt)
- [All completed episodes and individual termination flags](completed_episodes.csv)
- [Episode counts, final lengths and censoring](episode_capture_summary.json)
- [Applied-actuator torque traces for three sampled environments](torque_profile.png)
  ([raw arrays](torque_profile.npz)); these are sampled once per 50-Hz policy
  step, not complete 1200-Hz torque or solver joint-reaction records.
- [Independent result and preservation checks](verification.json)

All 64 first episodes and all 64 respawned second episodes reached the horizon.
The 64 new, zero-duration trajectories automatically spawned at the last step
are not counted as completed or positive-duration censored episodes. The native
summary's generic `fall_rate` means non-timeout endings; here every individual
non-timeout flag is independently zero.

## Compatibility and evaluation conditions

The saved training source revision is
`a941eebf7f7e198a6af269b925ccf90c7c449a67`; the inspected current revision is
`cf63e966b98fcf2ce42a747e90fe2f7b6b50ebfb`. The Git diff between them for
`isaaclab_train/source` and `isaaclab_train/scripts` is empty. Saved configuration:

- `Trackstand-Balance-v2`, 11 observations, bounded Beta actor, no observation normalization.
- Physics and explicit PD: 1200 Hz; observations, policy and targets: 50 Hz;
  rear external integral: 200 Hz, P = 0.30 N·m·s/rad, I gain = 3.3 N·m/rad.
- No domain randomization; existing per-step IMU white noise remains enabled.
- Horizon 30 s; all six configured termination terms remain active:
  `time_out`, `fallen`, `diverged`, `drifted`, `yaw_deviation`, `wheel_net`.

The completed evaluation used 64 environments × 3000 policy steps, seed 42,
headless CUDA device 0, deterministic Beta mean. `--dr` and `--pure_survival`
are omitted. Saved training `agent.max_iterations=100000` is retained as
metadata; it does not mean training reached 100000 iterations.

`evaluate_with_capture.py` executes the production `play_eval.py` unchanged. It
adds observation-only configuration and episode-result capture, verifies the
saved/resolved configurations before stepping, and preserves original step
results. Allowed configuration differences are the evaluation environment count,
output location and copied checkpoint path. It records each completed episode,
overlapping termination flags, and any unfinished final trajectories.

The post-construction environment comparison found only `log_dir`,
`scene.num_envs` (4096 → 64), and `scene.terrain.num_envs` (4096 → 64) differences.
The agent comparison found only the copied checkpoint path. All numerical task,
observation, action, actuator, reward and termination configuration matched the
saved training YAML. Full comparisons are in `env_config_comparison.json` and
`agent_config_comparison.json`; resolved configurations are retained alongside.

**Evaluation status: completed, exit code 0; independent verification passed.**

## Reproducibility and preservation

`select_checkpoint.py` reproduces checkpoint selection. `preflight_provenance.json`
contains the source revision, dirty state, source hashes and task/script diff.
`original_run_manifest_before.json` hashes every original run file, including
checkpoints, TensorBoard events, saved configuration and pre-existing artifacts.
Before/after hashes verified all **542 original run files** unchanged, including
the existing evaluation artifacts, and all **45 production Python inputs**
unchanged. The staged checkpoint hash also matches. No training process was
stopped, restarted or modified. A separate ongoing DR training job remained
present during the evaluation; this limits interpretation of wall-clock speed,
not the requested simulated step counts.

CPU checks completed: report helpers compile, wrapper CLI help exits successfully,
and saved YAML tuple/slice decoding, JSON normalization and the allowed-difference
guard were tested. The completed episode CSV and native evaluator summary agree
on counts, lengths and every termination flag. `execution_status.json` and
`startup_attempts.json` record completion and the earlier report-helper startup
failure: comparing pre-construction defaults to post-construction saved YAML
incorrectly rejected generated prim paths/terrain/camera metadata. That startup
produced zero trajectories; the helper was corrected to compare at the matching
lifecycle stage. Its log and pre-construction differences are preserved with the
`startup1_` prefix. Production behavior was unchanged.

Runtime: Python 3.12.13, Isaac Lab 7.2.0, RSL-RL 5.0.1, Torch 2.10.0+cu128,
CUDA 12.8, NVIDIA GeForce RTX 4090. Kit build identifiers and existing startup
warnings are retained in `run.log`; package versions are in `versions.json`.

Exact capture command, from `isaaclab_train/` (also in `command.json`):

```bash
/home/lmbike/IsaacLab/_isaac_sim/python.sh -B \
  reports/20261103_v2_202611022149_no_dr_eval/evaluate_with_capture.py \
  --task Trackstand-Balance-v2 \
  --checkpoint /home/lmbike/trackstand_rl/isaaclab_train/reports/20261103_v2_202611022149_no_dr_eval/model_26600.pt \
  --num_envs 64 --num_steps 3000 --seed 42 --headless --device cuda:0 \
  agent.max_iterations=100000
```

For the same production evaluator without extra episode/config capture, replace
the report helper path with `scripts/rsl_rl/play_eval.py`. Continue using a copied
checkpoint to protect existing training/evaluation artifacts.

## Limitations

This establishes nominal, no-DR performance over two 30-s episodes per
environment for one seed. It does not establish DR transfer, hardware balance,
performance under disturbances, or survival beyond 30 s. Observation white noise
remained enabled. All environments use the nominal fixed reset state.

The native roll, velocity, drift and heading metrics sample after each policy
step and automatic reset. They are not full physics-rate extrema, and terminal
poses can be replaced by reset poses. The torque graph shows applied actuator
torque, not measured hardware torque or solver net joint-axis effort. Startup
warnings about attempted articulation lookup and the passive third joint are
preserved in the log; the requested evaluation itself completed without a
traceback or a non-timeout termination.
