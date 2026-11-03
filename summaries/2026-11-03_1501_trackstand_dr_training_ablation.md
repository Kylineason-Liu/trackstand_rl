# Trackstand training-factor ablation, DR re-run: results

- **Summary written:** 2026-11-03 15:01 HKT
- **Training:** 2026-11-02 16:48 → 2026-11-03 03:10 HKT (queue `ablation_queue_dr/`, max 4 concurrent)
- **Evals finished:** 2026-11-03 03:54 HKT
- **Worktree:** `~/bike_test/trackstand_rl_test_layer` (branch `test_layer`, uncommitted)
- **Full tables:** `~/bike_test/trackstand_rl_test_layer/isaaclab_train/logs/ablation_eval_dr/summary.md`
- **Earlier no-DR batch (comparison):** `.../logs/ablation_eval/summary.md`

## Setup

- Plant: `Trackstand-Balance-v1`; every run trained with `--dr` (checked in each run's `params/env.yaml`).
- Seed 42, 4096 envs, 7,500 PPO updates (CycleRL budget: 120k skrl timesteps).
- Evaluated checkpoint: final (`model_7499.pt` / `agent_120000.pt`).
- Eval: 64 envs × 3,000 steps (~2 episodes per env), seed 42, Isaac and MuJoCo, each with DR off and on.
  Episodes cap at 30 s.
- All 10 jobs exited 0, with no watchdog retries. The CycleRL "best" checkpoint has the same policy
  weights as its final checkpoint, so it is reported once.

| Run | Changes vs 5-layer baseline |
|---|---|
| 5L-Base | 5-layer Beta / RSL-RL, lr 1e-3 adaptive, entropy 0.005, no obs norm |
| 5L-Norm | + obs normalization |
| 5L-Cos | + lr 1e-4 cosine |
| 5L-NormCos | + norm + cosine |
| 5L-NormCosEnt | + norm + cosine + entropy 0.01 |
| 2L-Base | [32,32] trackstand recipe |
| 2L-NormCosEnt | [32,32] + norm + cosine + entropy 0.01 |
| 5L-Base-DriftQuad | 5L-Base with drift-quad reward |
| 5L-Base-WheelBand | 5L-Base with wheel-band reward |
| CycleRL-Recipe | CycleRL 8-D obs + CycleRL skrl YAML (Gaussian, obs + value scaler) |

## Results

Mean episode length in seconds (cap 30). Brackets give the same run from the earlier no-DR training batch.

| Run | Isaac, DR on | MuJoCo, no DR | MuJoCo, DR on | Main MuJoCo failure (no DR) |
|---|---|---|---|---|
| 5L-Norm | 17.2 (2.0) | **14.5** (0.6) | **5.4** (0.3) | yaw > 30°: 98% |
| 2L-Base | **25.4** (3.6) | 13.1 (30.0) | **5.4** (2.4) | drifted: 71% |
| 5L-Base-DriftQuad | 20.4 (2.5) | 12.2 (0.3) | 4.1 (5.2) | yaw > 30°: 78% |
| 5L-Base | 21.5 (1.0) | 6.2 (1.4) | 4.7 (1.2) | yaw > 30°: 64% |
| 5L-Base-WheelBand | 14.7 (0.7) | 3.9 (30.0) | 3.2 (0.5) | mixed |
| 5L-NormCos | 19.9 (1.3) | 1.8 (0.3) | 2.1 (0.4) | yaw > 30°: 85% |
| 5L-NormCosEnt | 24.6 (2.0) | 1.1 (0.4) | 1.1 (0.5) | wheel_net: 59% |
| CycleRL-Recipe | 14.7 (0.6) | 1.1 (0.2) | 0.7 (0.2) | wheel_net: 67% |
| 2L-NormCosEnt | 2.7 (2.0) | 1.4 (1.4) | 0.9 (0.9) | did not learn |
| 5L-Cos | 8.6 (3.8) | 0.3 (30.0) | 0.5 (6.2) | did not learn |

Mean training episode length at the end of training (steps, max 1,500):

| Run | Steps |
|---|---|
| 5L-Base | 1,201 |
| DriftQuad | 1,322 |
| WheelBand | 1,273 |
| 5L-Norm | 1,192 |
| 5L-NormCos | 1,364 |
| 5L-NormCosEnt | 1,270 |
| 2L-Base | 1,162 |
| 5L-Cos | **429** |
| 2L-NormCosEnt | **137** |

Most runs had levelled off by update 5,000.

## Findings

1. **DR training worked in Isaac.** Under DR, Isaac episodes went from 0.6–3.8 s to 8.6–25.4 s.
   2L-Base reached a 76% timeout rate.
2. **None of the runs transfers to MuJoCo.** The best MuJoCo timeout rate is 12.6% with DR off
   (DriftQuad) and 7.7% with DR on (2L-Base).
3. **The main failure is now heading drift (yaw > 30°), not falling.** In MuJoCo with DR off,
   5L-Norm keeps roll at 1.3° RMS but ends 98% of episodes on yaw. Several runs also end every Isaac
   no-DR episode on yaw; 5L-Base, for example, lasts 18.9 s there with a 0% timeout rate.
4. **Factor effects** (one seed, so indicative only):
   - Obs normalization helps: MuJoCo no-DR episode length goes from 6.2 to 14.5 s.
   - The lr 1e-4 cosine schedule hurts. 5L-Cos and 2L-NormCosEnt never learned, and adding the
     schedule drops 5L-Norm from 14.5 to 1.8 s in MuJoCo.
   - The CycleRL recipe is near the bottom.
   - Depth: [32,32] is about as good as 5-layer, but it drifts more.
5. **Comparison with the no-DR batch.** Three no-DR-trained runs (2L-Base, 5L-Cos, WheelBand) lasted
   the full 30 s in MuJoCo with DR off, but their DR-trained versions don't. Those no-DR policies
   collapse once MuJoCo DR is on (≤ 6.2 s). Results also change sharply between the two batches
   (5L-Cos: 30 → 0.3 s), which suggests large run-to-run variance. Single-factor gaps need more seeds.

## Suggested next steps (not started)

- Find the cause of the yaw failure in 5L-Norm and 2L-Base: does it come from the policy or from the
  plant?
- Train the top three (5L-Norm, 2L-Base, DriftQuad) with 1–2 more seeds, all `--dr`.

## Housekeeping done with this summary

- Fixed the stale header in `ablation_queue_dr/summarize.py`, which said "no-DR training" and claimed
  old 5L-Base checkpoints, and regenerated `summary.md`. The tables did not change.
- Removed the finished active-runs section from `CLAUDE.md` and added a link to this summary.
