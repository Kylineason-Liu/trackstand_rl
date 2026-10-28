# Loaded steering replay versus Isaac Lab

The **ground-loaded** hardware recording is compared with its matching original
Isaac Lab evaluation segment. Each figure has three panels and two curves per
panel: steering torque, angular velocity, and position. Blue is Isaac; orange is
loaded hardware. [Open the gallery](index.html).

- [First five elapsed seconds](steering_loaded_sim_vs_replay_200hz.png) —
  [SVG](steering_loaded_sim_vs_replay_200hz.svg), [PDF](steering_loaded_sim_vs_replay_200hz.pdf).
- [Startup, 0–0.5 s](steering_loaded_sim_vs_replay_startup.png) —
  [SVG](steering_loaded_sim_vs_replay_startup.svg), [PDF](steering_loaded_sim_vs_replay_startup.pdf).
- [Later motion, 1–5 s](steering_loaded_sim_vs_replay_detail.png) —
  [SVG](steering_loaded_sim_vs_replay_detail.svg), [PDF](steering_loaded_sim_vs_replay_detail.pdf).

## Recordings and matching

Hardware is `20261027_190254_424315_scaled_200hz_policy_replay`, nominal MIT
packet rate **200 Hz**, target rate **50 Hz**. The user confirmed normal front-tyre
loading on the ground, frame supported upright, and steering free to turn.
Only this loaded rate is available. The rear motor was not commanded in this test.

This run replays **source seconds 0–5**, with no preparation ramp. The previous
airborne comparison used 12.5–17.5 s and therefore cannot supply the matching
simulation curves. Here the simulation source is
`isaaclab_train/reports/20261025_2500_dr_comparison/isaac400/motor_samples.npz`,
environment 0, episode 1, model_2500, DR enabled, physics 400 Hz.
**All 250 requested steering targets match exactly**, including every held
physics substep. No reset occurs in this interval.

## Clocks, measurements, and limits of interpretation

All figures retain native elapsed timestamps: **no smoothing, fitted delay,
time warping, or bias subtraction**. Hardware takes **5.2246 s** to reach the
end of the nominal five-second source sequence; its effective source-target
rate is **47.85 Hz**. The main graph shows the first **five elapsed seconds**,
so the hardware has not yet finished every source target at the right edge.
Accumulating separation of later curves includes this playback-rate difference;
it is not a measured motor latency.

Torque is simulation **submitted actuator effort before passive losses** versus
hardware **signed A1 Q-current × B0 Kt**, not net torque after friction. Hardware
A1/A2/A3 arrive around 100 Hz, versus 400 Hz simulation. Narrow hardware peaks
can be missed: the original loaded report contains an F1 startup torque peak
of **3.046 Nm**, while sampled A1 here peaks at **1.427 Nm**. Different channel
peaks alone do not establish torque saturation or bandwidth.

Position retains each fixed straight reference; hardware B1/A3/F1 origin checks
pass. Raw A1/A2/A3 decoding and MIT position bytes are independently checked.
The full core's unshifted A3 target-tracking RMS is **2.62162°**, including startup.

**This is not a matched-plant causal experiment.** The original simulated bike
balances freely on the ground with sampled DR, while the hardware frame is
supported upright. Simulated steering Kp/Kd are **10.6517/0.209184**; hardware
requests **10/0.2**. Full sampled simulation DR values are retained in
[statistics.json](statistics.json). No new simulation or hardware test was run.

## Reproduce

From the repository root:

```bash
python3 -B deploy/experiments/reports/20261029_loaded_steering_sim_vs_replay/plot_comparison.py
```

[Plotted samples](plotted_samples.csv), [statistics and source hashes](statistics.json),
[verification](verification.json), and the
[original loaded analysis](../20261027_loaded_steering_replay/README.md)
provide the underlying data and checks. Original plots, captures, source code,
and progress ledgers are preserved.
