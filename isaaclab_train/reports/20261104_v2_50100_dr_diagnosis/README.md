# Frozen v2 model_50100: IsaacLab DR diagnostic

Unchanged `scripts/rsl_rl/play_eval.py`, with observation-only config/episode
capture, completed 2026-11-04. The original model was hash-verified and copied
into this new report directory before running, so native eval.txt and torque
outputs cannot overwrite training records or earlier reports.

64 environments, seed 42, 1500 policy steps, DR on, pure-survival mode;
policy 50 Hz /physics-PD 1200 Hz /rear integral 200 Hz. Deterministic Beta means,
11 observations, no observation normalization. Saved-vs-resolved environment
and agent comparisons allow only environment count, isolated output/checkpoint
paths and the four user-requested disabled terminations. All other fields match.

All 64 first episodes completed: **58 timeouts (90.625%) /6 falls (9.375%)**.
Mean **1364.640625 steps =27.2928125 s**. There were no additional completed
episodes; six later respawn trajectories remain censored and are not counted.
96,000 total environment transitions. The DR ranges match MuJoCo's configuration,
but the engines use independent random generators/draws, not identical sampled
physical plants. This establishes a substantial frozen-policy transfer gap;
it is not a per-draw causal engine comparison or a multi-seed robustness result.

[Native summary](eval.txt), [completed episodes](completed_episodes.csv),
[capture summary](episode_capture_summary.json), [command](command.json),
[environment comparison](env_config_comparison.json),
[agent comparison](agent_config_comparison.json), [versions](versions.json).
Native pose metrics are sampled after auto-reset; terminal orientations can
be replaced by spawn state. Episode durations/flags use the pre-reset manager
bookkeeping and do not have that limitation.

The first sandbox startup failed before evaluation because CUDA was unavailable;
`startup_sandbox.log` is preserved. An authorized run outside that sandbox
completed successfully, exit 0. No training, source/config edit or hardware work.
The [main comparative analysis](../../../mujoco_sim2sim/reports/20261104_v1_v2_transfer_diagnosis/README.md)
contains matched MuJoCo v1/v2 and component-ablation evidence.
