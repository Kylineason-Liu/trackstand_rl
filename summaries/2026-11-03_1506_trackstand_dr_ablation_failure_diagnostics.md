# Trackstand DR ablation: why policies succeed or fail (Isaac vs MuJoCo)

- **Summary written:** 2026-11-03 15:06 HKT
- **Data:** the DR ablation batch (trained 2026-11-02 16:48 → 2026-11-03 03:10; evals done 03:54). See
  [2026-11-03_1501_trackstand_dr_training_ablation.md](2026-11-03_1501_trackstand_dr_training_ablation.md).
- **Report script (new):**
  `~/bike_test/trackstand_rl_test_layer/isaaclab_train/logs/ablation_queue_dr/failure_report.py`
  (system python3, ~5 s)
- **Full output:**
  `~/bike_test/trackstand_rl_test_layer/isaaclab_train/logs/ablation_eval_dr/failure_report.txt`
- **TensorBoard folder (named symlinks to each run):** `.../logs/ablation_eval_dr/tb/`

## Commands

```bash
T=~/bike_test/trackstand_rl_test_layer/isaaclab_train; E=$T/logs/ablation_eval_dr

# TensorBoard: all 10 DR runs by name, then open http://localhost:6006
tensorboard --logdir $E/tb --port 6006
#   useful tag filter (RSL-RL): Train/mean_episode_length|Episode_Termination|Policy/mean_std|Loss/learning_rate
#   CycleRL (skrl) uses other tags: "Episode / Total timesteps (mean)", "Policy / Standard deviation"

# Diagnostics for every policy (or a subset); the no-DR batch works the same way
python3 $T/logs/ablation_queue_dr/failure_report.py
python3 $T/logs/ablation_queue_dr/failure_report.py --runs 5L-Norm,2L-Base
python3 $T/logs/ablation_queue_dr/failure_report.py $T/logs/ablation_eval

# Raw per-run outputs
cat $E/summary.md
cat $E/<run>/isaac_nodr/eval.txt              # also isaac_dr/
cat $E/<run>/mujoco_nodr/*_summary.log        # also mujoco_dr/
xdg-open $E/<run>/isaac_nodr/torque_profile.png
xdg-open $E/<run>/mujoco_nodr/*_epi_len_histogram.png

# Watch a policy in Isaac (GUI). play_eval overwrites eval.txt beside the checkpoint.
cd $T && export PYTHONPATH=$PWD && ~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Trackstand-Balance-v1-Abl-5L-Norm --num_envs 16 \
  --checkpoint $T/logs/rsl_rl/trackstand_rl_abl_5l_norm/2026-11-02_16-52-23_dr/model_7499.pt

# Watch the same policy in MuJoCo (add --dr for DR). RSL-RL ONNX only:
# the viewer has no CycleRL 8-obs backend.
cd $T/../mujoco_sim2sim && ~/IsaacLab/_isaac_sim/python.sh scripts/run_deploy_viewer.py \
  --task deploy_sim --onnx $E/5L-Norm/policy.onnx --num_envs 16 --seed 42
```

Task names and checkpoints for each run are listed in `$E/runs.txt`.

## Key indicators (no-DR evals)

| Run | Training: yaw-termination share | Isaac: len s / main term | MuJoCo: len s / main term | MuJoCo: mean steer action / mean steer angle | MuJoCo: steer jitter mean \|Δa\| | Rear \|ω\| rad/s, Isaac → MuJoCo | Steer τ RMS N·m, Isaac → MuJoCo |
|---|---|---|---|---|---|---|---|
| 5L-Norm | 0.24 | 30.0 / — | **14.5** / yaw 98% | +0.80 / 72° | 0.085 | 0.04 → 0.19 | 0.04 → 0.49 |
| 2L-Base | 0.35 | 30.0 / — | **13.1** / drift 71% | +0.14 / 14° | 0.031 | 1.51 → 1.93 | 0.04 → 0.16 |
| 5L-Base-DriftQuad | 0.29 | 13.7 / yaw 100% | **12.2** / yaw 78% | +0.74 / 66° | 0.068 | 0.09 → 0.47 | 0.04 → 0.42 |
| 5L-Base | 0.30 | 18.9 / yaw 100% | 6.2 / yaw 64% | +0.62 / 62° | 0.189 | 0.12 → 2.49 | 0.04 → 0.65 |
| 5L-Base-WheelBand | 0.29 | 11.9 / yaw 100% | 3.9 / drift 37%, fall 36% | +0.37 / 46° | 0.437 | 0.15 → 4.21 | 0.04 → 0.83 |
| 5L-NormCos | 0.22 | 28.4 / yaw 23% | 1.8 / yaw 85% | +0.09 / 34° | 0.548 | 0.12 → 6.17 | 0.32 → 0.95 |
| 5L-NormCosEnt | 0.27 | 30.0 / — | 1.1 / wheel 59% | +0.12 / 28° | 0.678 | 0.12 → 14.2 | 0.22 → 1.08 |
| CycleRL-Recipe | n/a (skrl) | 8.2 / yaw 100% | 1.1 / wheel 67% | −0.31 / 23° (87% of steps at the action limit) | 6.10 | 0.32 → 16.5 | 0.15 → 1.16 |
| 2L-NormCosEnt | 0.46 (did not learn) | 3.0 / yaw 100% | 1.4 / yaw 37%, fall 34% | +0.12 / 28° | 0.745 | 2.44 → 15.7 | 0.11 → 1.05 |
| 5L-Cos | 0.60 (did not learn) | 11.9 / yaw 100% | 0.3 / fall 100% | +0.45 / 43° | 0.758 | 0.96 → 29.2 | 0.14 → 1.27 |

Column notes:

- **Training yaw-termination share:** mean of `Episode_Termination/yaw_deviation` over the last 100 updates.
- **Rear wheel |ω|:** time means. The 61.6 rad/s maxima in MuJoCo are reset transients for the good
  runs (0–0.14% of steps above 30 rad/s); for CycleRL it is 22% of steps.
- **Steer jitter:** mean |Δa| of the raw steering action between consecutive policy steps.

## Findings

1. **Steering jitter separates success from failure in MuJoCo.**
   - The three runs that last more than 12 s have steering jitter ≤ 0.09 and a quiet rear wheel
     (≤ 2 rad/s mean).
   - Every run that lasts under 2 s has jitter 0.55–0.76 (CycleRL: 6.1), and its rear wheel spins at
     6–29 rad/s. That spin causes the wheel_net terminations and falls.
   - This is a correlation; it has not been tested as a cause.
2. **The heading-drift failure starts in training, not in the Isaac-to-MuJoCo gap.**
   - In the good runs, 21–35% of training episodes end on yaw deviation.
   - With no DR in Isaac, 5L-Base, DriftQuad and WheelBand still end every episode on yaw.
   - The v1 reward has no yaw term (survival + drift_gauss + rate penalty), yet a yaw change over 30°
     ends the episode.
3. **The policies use two balancing strategies with different failure modes.**
   - *Bar turned:* 5L-Norm, DriftQuad and 5L-Base hold a steering action of +0.6–0.8, i.e. the bar at
     62–72° in MuJoCo. These balance tightly (roll RMS 1.3–1.6°) but rotate in place
     (5L-Norm: 160° turned per episode in MuJoCo vs 39° in Isaac), and end on yaw.
   - *Bar near straight:* 2L-Base keeps the bar at about 14° and rocks forward and back instead. It
     drifts 0.36 m even in Isaac and ends on drift in MuJoCo.
4. **Sim-to-sim gap signal.** The same policy uses about 4–15× more steering torque (RMS) in MuJoCo
   than in Isaac (0.04 → 0.16–0.65 N·m for the RSL-RL runs).
   - Isaac traces only 3 envs, sampled at the policy step.
   - Isaac's `applied_torque` may exclude joint friction, so the two numbers may not measure the same
     thing. Treat this as a lead, not a result.
5. **Training failures.**
   - 5L-Cos and 2L-NormCosEnt: the lr 1e-4 cosine schedule had decayed to 1.5e-8 by the end while
     training episodes were still only 408 and 140 steps long.
   - CycleRL recipe: policy std is still 0.65. Its Gaussian means reach about ±16, so it effectively
     switches steering between full left and full right.

## Possible next steps (not started)

- Add a yaw or yaw-rate penalty to the trackstand reward, then retrain the top three with `--dr`.
- Check whether the steering-torque gap is real: log the Isaac steering torque with friction, for all
  envs, at physics rate.
