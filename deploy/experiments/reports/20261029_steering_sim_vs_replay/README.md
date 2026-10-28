# Steering simulation versus corrected-zero hardware replay

Each figure has **three panels and two curves per panel**: actuator torque (Nm),
steering velocity (rad/s), and steering position (rad). Blue is saved Isaac Lab
400 Hz simulation; orange is the corrected-zero airborne hardware replay.

- **[50 Hz comparison](steering_sim_vs_replay_50hz.png)** — [SVG](steering_sim_vs_replay_50hz.svg), [PDF](steering_sim_vs_replay_50hz.pdf).
- **[200 Hz comparison](steering_sim_vs_replay_200hz.png)** — [SVG](steering_sim_vs_replay_200hz.svg), [PDF](steering_sim_vs_replay_200hz.pdf).
- [Gallery](index.html), [native plotted samples](plotted_samples.csv), [statistics/input hashes](statistics.json).

This follows the [rear comparison format](../20261029_rear_sim_vs_replay/README.md).
It uses the corrected October 27 steering captures, not the older recordings with
the 10–12-degree origin error. Both verified B1 once, A3 zero count 0 and valid F1
zero. Their requested 400-sample position plans exactly equal the saved simulation
case `isaac400_dr_env0_12p5_17p5_friction_on`, index 4, including preparation.
MIT position encoding introduces its documented small quantization error.

The selected middle-episode source window is **12.5–17.5 s** from model_2500's
Isaac-recorded DR episode. In both hardware and saved nominal fixed-base simulation,
it follows a 1-second ramp and 1-second hold. Figures show **replay time 2–7 s**,
relabeled 0–5 s. Targets update at 50 Hz; hardware sends MIT packets at 50/200 Hz.
No time shift, smoothing, fitted bias or core-boundary position reset is applied.

Torque is **submitted simulation actuator effort before passive friction** versus
**hardware signed A1 Q current × measured B0 Kt**. It is not solver joint-axis
effort or independently measured net torque. Position uses the fixed spawn zero
in simulation and verified straight multi-turn A3 reference on hardware. Positive
steering follows the existing policy coordinate (rightward on this bike).

Raw A1/A2/A3 decode is checked against saved columns and MIT target bytes against
their recorded position. Recomputed unshifted target-tracking RMS is **0.701129°
at 50 Hz /0.787812° at 200 Hz**, reproducing the existing corrected-zero report.
This tracking error is distinct from hardware-versus-simulation curve difference.

Simulation supplies 400 Hz samples and hardware approximately 100 Hz per A1/A2/A3
channel. Sharper or larger simulated speed/torque peaks cannot alone establish a
real bandwidth failure; asynchronous sampling and unmodeled dynamics both matter.
Support, nominal versus physical parameters and MIT quantization are not fully
matched. These two unloaded middle-window tests do not establish a universal rate
ranking, startup performance or full balancing capability.

No new motor/CAN operation, policy inference or simulation was performed. Inputs
are unchanged. Reproduce from the repository root:

```bash
python3 -B deploy/experiments/reports/20261029_steering_sim_vs_replay/plot_comparison.py
```
