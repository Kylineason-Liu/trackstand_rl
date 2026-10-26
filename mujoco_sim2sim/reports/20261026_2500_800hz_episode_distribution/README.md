# Model_2500 MuJoCo episode length distribution, 800 Hz

## Survival-only version

[Survival-only PNG](model_2500_mujoco_800hz_episode_length_distribution_survival_only.png),
[PDF](model_2500_mujoco_800hz_episode_length_distribution_survival_only.pdf),
[SVG](model_2500_mujoco_800hz_episode_length_distribution_survival_only.svg).

The saved **October 23** `diagnostic800_survival` evaluation has only timeout and
fallen terminations enabled. DR is ON, physics and motor torque application are
800 Hz, and policy targets update at 50 Hz. There are 64 environments, seeds
42–105, and 60 simulated seconds per environment. The **139 completed episodes**
have mean duration **23.3236 seconds**; **104 / 139 (74.82014%)** reach 30 seconds.
The last histogram bin contains 105 episodes: 104 timeouts and one fall in the
27–30-second interval. It must not be read as 105 successful timeouts.

This is the original survival-only simulation, with its own reset history.
No new evaluation was needed. Its records, timing and checkpoint hash are checked
by the same drawing script. All disabled termination channels are checked to be
false throughout the capture. See
[verification_survival_only.json](verification_survival_only.json) and
[completed_episodes_survival_only.csv](completed_episodes_survival_only.csv).

```bash
python3 mujoco_sim2sim/reports/20261026_2500_800hz_episode_distribution/draw_histogram.py --survival-only
```

## Full-task version

[PNG histogram](model_2500_mujoco_800hz_episode_length_distribution.png),
[PDF](model_2500_mujoco_800hz_episode_length_distribution.pdf),
[SVG](model_2500_mujoco_800hz_episode_length_distribution.svg).

This figure uses the saved **October 25** model_2500 evaluation, with domain
randomization enabled, 64 environments, seed 42 (per-environment seeds 42–105),
and 3,000 policy steps / 60 simulated seconds per environment. No new evaluation
was needed. The 149 completed episodes have mean duration **19.4554 seconds**;
**92 / 149 (61.74497%)** reach the 30-second timeout.

Physics and motor torque application both run at **800 Hz** (0.00125 seconds),
with 16 physics substeps per **50 Hz** policy update. The steering position target
is updated/slewed at 50 Hz and held across those substeps; steering PD torque is
recomputed and assigned to MuJoCo's motor controls on every substep. Rear
feedforward torque is also assigned every substep. There are no CAN messages
in this simulation. See `BikeBalanceEnv.step` in `src/newbike_mujoco/env.py`.

The plot follows the October 15 histogram's blue count bars, 3-second bins,
0–30-second X axis, and exact counts above every bar. Timeouts are included;
unfinished final episodes are excluded. All **six current v1 termination
conditions** are active. The October 15 v0 evaluation used different termination
criteria and evaluation budget, so identical plotting style does not make these
two runs a controlled comparison. The 149 completed episodes are repeated
episodes from 64 environments, not 149 independent randomized environments.

Source records are under
`isaaclab_train/reports/20261025_2500_dr_comparison/mujoco800/`, prefix
`202610252053_2500_dr`. The source metadata names the DR-trained
`2026-10-21_11-18-29/model_2500.pt` checkpoint, SHA256
`5389effbd6a0d3e78c9ea9caeac309ceb5082622c8ee125bb701c63d1511fe52`.

`draw_histogram.py` reconstructs every completed episode from the original reset
events, checks the termination flags and metadata episode lengths, and compares
all bin counts, mean duration and timeout count with the saved source results.
It also checks the saved timing, checkpoint hash, and recorded torque-array
substep dimension. Source hashes and check results are in
[verification.json](verification.json); episode-level data are in
[completed_episodes.csv](completed_episodes.csv).

Reproduce from the repository root with system Python, NumPy and Matplotlib:

```bash
python3 mujoco_sim2sim/reports/20261026_2500_800hz_episode_distribution/draw_histogram.py
```

Only files in this report directory are written. Existing evaluation captures,
figures, controller source and checkpoints are preserved.
