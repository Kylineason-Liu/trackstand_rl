# TensorBoard: why the RSL-RL curves look noisier than the CycleRL (skrl) curve

- **Summary written:** 2026-11-03 15:24 HKT
- **Data:** the DR ablation batch, TensorBoard folder
  `~/bike_test/trackstand_rl_test_layer/isaaclab_train/logs/ablation_eval_dr/tb/`
  (training 2026-11-02 16:48 → 2026-11-03 03:10)
- **Related:** [2026-11-03_1506_trackstand_dr_ablation_failure_diagnostics.md](2026-11-03_1506_trackstand_dr_ablation_failure_diagnostics.md)

## How each library logs

| | RSL-RL (9 runs) | skrl (CycleRL recipe) |
|---|---|---|
| Logged | every PPO update | every `write_interval: 1000` timesteps (= 62.5 updates) |
| Points per run | 7,500 | 120 |
| Episode-length value | mean of the last 100 finished episodes (`deque(maxlen=100)` in `rsl_rl/utils/logger.py`) | mean over all episodes in the 1,000-timestep window |
| X axis ("Step") | update number 0–7,500 | timestep 0–120,000 (16× wider for the same training) |

## Measured noise

Noise is the mean |point-to-point change| over the second half of training, divided by the mean
episode length. The RSL-RL curves are block-averaged to 120 points to match skrl's resolution.

| Run | Mean episode length (2nd half, steps) | Relative noise at 120 points |
|---|---|---|
| CycleRL-Recipe | 331 | **0.090** |
| 2L-Base | 923 | 0.082 |
| 5L-Base | 1,213 | 0.044 |
| 5L-Base-WheelBand | 1,239 | 0.039 |
| 5L-Norm | 1,266 | 0.033 |
| 5L-Base-DriftQuad | 1,230 | 0.031 |
| 5L-Cos | 353 | 0.022 |
| 5L-NormCosEnt | 1,187 | 0.021 |
| 5L-NormCos | 1,344 | 0.020 |
| 2L-NormCosEnt | 134 | 0.007 |

## Findings

1. At the same resolution, every RSL-RL run is **less** noisy than the CycleRL run (0.007–0.082 vs
   0.090). The CycleRL curve only looks smooth because it has 60× fewer points, each averaging
   62.5 updates.
2. When the runs are overlaid on TensorBoard's "Step" axis, the RSL-RL curves end at step 7,500 while
   CycleRL runs to 120,000. The RSL-RL curves are squeezed into the first 1/16 of the axis, so they
   look like a dense band of noise.
3. One run has a real wander: 2L-Base's training episode length dropped from 1,263 steps at update
   2,500 to 642 at update 5,000, then recovered to 1,162. It uses an adaptive learning rate that ended
   at 2.7e-3, the highest of the batch.

## How to view them fairly

- Show one library at a time with the TensorBoard run filter (e.g. `^(?!CycleRL)`), or compare with
  the horizontal axis set to "Relative" (wall time).
- Set the smoothing slider to 0.95–0.99 for the RSL-RL runs.
