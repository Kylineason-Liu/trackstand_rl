# Model_2500: 6,000-episode MuJoCo survival evaluation

This experiment collects **6,000 completed episodes** for the DR-trained
`2026-10-21_11-18-29/model_2500.pt`, checkpoint SHA256
`5389effbd6a0d3e78c9ea9caeac309ceb5082622c8ee125bb701c63d1511fe52`.
The evaluation is complete. The definitive results and figure are under `run_6000/`.

[Histogram PNG](run_6000/model_2500_mujoco_800hz_6000_survival_histogram.png),
[PDF](run_6000/model_2500_mujoco_800hz_6000_survival_histogram.pdf),
[SVG](run_6000/model_2500_mujoco_800hz_6000_survival_histogram.svg).

| Result | Value |
| --- | ---: |
| Completed episodes | 6,000 |
| Mean duration | 24.21382 s |
| 30-second timeouts | 4,704 / 6,000 = 78.4% |
| Falls | 1,296 / 6,000 = 21.6% |
| Total simulated time | 145,282.92 s = 40.35637 hours |
| Maximum worker evaluation time, excluding startup | 398.50 s |

The last bin contains 4,727 episodes: 4,704 timeouts and 23 falls between 27
and 30 seconds. Bin counts are **1,097, 6, 5, 19, 25, 38, 40, 26, 17, 4,727**.

## Evaluation conditions

- MuJoCo physics: **800 Hz**, timestep 0.00125 seconds.
- Motor torque calculation/application: **800 Hz**, 16 substeps per policy step.
- Deterministic Beta-mean policy and held position-target updates: **50 Hz**.
- Domain randomization: **ON**, with fresh episode draws on each reset.
- Enabled terminations: **timeout and fallen only**; horizon 1,500 policy steps
  / 30 seconds. Drift, yaw, divergence and wheel-net terminations are disabled.
- Production motor gains, losses, limits, observation noise, reset behavior,
  model geometry and `project` velocity-limit mode are retained.

`evaluate.py` calls the production `DeployedPolicy`, `ViewerEnvGroup` and
`BikeBalanceEnv.step`; it does not implement a second physics/controller loop.
Each policy call retains the original 64-row batch shape. Independent row
streams are distributed among 12 CPU subprocesses, each using one Torch thread.
Unused MLP input rows do not affect active rows. This is an evaluation helper;
production controller and model files are not edited.

## Sampling

The 64 original seed streams use seeds **42–105**. Each receives a fixed completed
episode quota: streams 0–47 contribute 94 episodes each; streams 48–63 contribute
93 each. This sums to exactly 6,000 without selecting a larger fraction from
streams that happen to fail faster. Every counted episode finishes; no partial
final episode is included or substituted for a completed one.

The runs share the same seed-42 32-bucket contact-friction bank as the earlier
evaluation. Per-episode mass, CoM, gain, contact-bucket, initial-state and IMU-bias
draws are retained in the raw episode records. This is 6,000 episodes across 64
random streams with one fixed material bank, not 6,000 separately trained
policies or 6,000 independently sampled material banks. The earlier 139-episode
plot used a fixed 60-second budget per environment; this experiment uses fixed
completed-episode quotas, so it is not simply that old sample replicated.

## Artifacts and validation

- `run_6000/model_2500_mujoco_800hz_6000_survival_histogram.png`, plus PDF/SVG:
  blue bars with the original ten 3-second bins and exact count labels.
- `run_6000/completed_episodes.csv`: all 6,000 episode lengths and outcomes.
- `run_6000/summary.json`: aggregate results and per-stream sample counts.
- `run_6000/histogram_counts.csv`: the plotted bin counts.
- `run_6000/worker_*/episodes.jsonl`: complete per-episode scalar statistics,
  sampled DR values, initial generalized positions/velocities and controller gains.
- `run_6000/experiment.json`: invocation, versions, configuration and source,
  model and checkpoint hashes; `verification.json`: completion/integrity checks.

Before continuing each stream, its first episode must reproduce the saved
October 23 survival-only reference's length, timeout outcome and squared-roll
sum (within numerical tolerance). All 64 streams must pass. Further checks cover
exact per-stream quotas, every episode's reset-to-end step count, simulation
duration, finite terminal state, torque bounds, enabled termination channels,
histogram totals, and unchanged protected source/model/checkpoint hashes.

All checks passed. An independent read of the exported CSV confirmed exactly
6,000 unique `(environment, episode)` pairs, the 93/94-episode quotas, every bin
count, mean duration and timeout/fall totals. Raw episode-file hashes also match
the verification manifest.

The evaluation workers finished successfully. The first reporting pass then
failed because its worker-directory glob also matched `worker_XX.log` files.
Only that one reporting line was corrected. `evaluate_as_run.py` preserves the
exact evaluator used for simulation; `finish_report.py` verifies that the fix is
the sole difference, checks unchanged runtime/model/checkpoint inputs, and
generates the figure and summary from the completed records. See
`run_6000/postprocessing_provenance.json`. No episodes were rerun or replaced to
correct the reporting step.

The JSONL episode files retain all statistics needed for the plot and summary.
Full per-physics-step trajectories are not collected by this histogram-specific
runner. `run/` and `run.log` belong to an interrupted initial attempt and are
excluded from every reported result.

## Reproduction

From the repository root, choose a new output directory:

```bash
python3 -B mujoco_sim2sim/reports/20261026_2500_800hz_6000_survival/evaluate.py \
  --episodes 6000 --workers 12 \
  --output-dir mujoco_sim2sim/reports/20261026_2500_800hz_6000_survival/reproduction
```

The runner refuses to overwrite an existing output directory. It requires
NumPy, MuJoCo, Torch and Matplotlib; exact versions are recorded in the run.
The completed run used MuJoCo 3.12.0, Torch 2.12.1, NumPy 2.2.6 and
Matplotlib 3.10.9. To regenerate this run's report without physics evaluation:

```bash
python3 -B mujoco_sim2sim/reports/20261026_2500_800hz_6000_survival/finish_report.py
```
