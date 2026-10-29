# Seven rear-wheel simulation versus replay comparisons

Each figure follows `20261029_rear_sim_vs_replay`: three vertically aligned
panels, two curves per panel (torque, velocity and cumulative position). Blue is
the same saved Isaac Lab 400 Hz actuator replay; orange is fresh hardware feedback.
All seven are **nominal 50 Hz C0** replays, not a 50-versus-200 Hz experiment.

[HTML gallery](index.html) · [Seven-page PDF](all_replays.pdf) ·
[Summary CSV](summary.csv) · [Native plotted samples](plotted_samples.csv) ·
[Statistics and input hashes](statistics.json) · [Verification](verification.json).

| Replay / timestamp | Last position (rad; PNG) | Peak speed (rad/s) | Core duration (s) | Packet rate (Hz) |
| --- | ---: | ---: | ---: | ---: |
| 1 / 152721_598360 | [-0.825282](20261029_152721_598360_rear_sim_vs_replay_50hz.png) | 4.107109 | 5.042994 | 49.573 |
| 2 / 152744_269780 | [-0.204019](20261029_152744_269780_rear_sim_vs_replay_50hz.png) | 2.123717 | 5.040167 | 49.601 |
| 3 / 152804_029946 | [-0.241985](20261029_152804_029946_rear_sim_vs_replay_50hz.png) | 3.008599 | 5.040724 | 49.596 |
| 4 / 152822_687404 | [-0.002301](20261029_152822_687404_rear_sim_vs_replay_50hz.png) | 0.083776 | 5.040669 | 49.596 |
| 5 / 152846_268393 | [-0.000383](20261029_152846_268393_rear_sim_vs_replay_50hz.png) | 0.061785 | 5.049553 | 49.510 |
| 6 / 152930_048554 | [-2.967486](20261029_152930_048554_rear_sim_vs_replay_50hz.png) | 4.892507 | 5.040342 | 49.599 |
| 7 / 152944_019062 | [-16.267483](20261029_152944_019062_rear_sim_vs_replay_50hz.png) | 9.542064 | 5.043597 | 49.570 |

Each plan has 100 zero-torque preparation commands and 250 core commands from
model_2500's Isaac-recorded episode window **12.5–17.5 s**. All 350 values exactly
match the saved simulator's targets. All seven logs report completion, confirmed
disable, restored watchdog and final fault code zero; every logged query completed.
The script independently verifies transmitted C0 payloads and A1/A2/A3 units/signs.

Time zero is the **actual first core command send** in each hardware log, and the
nominal first core command in simulation. This uses known command events, not a
fitted response delay. Hardware intervals retain their actual durations; no
smoothing, interpolation of the plotted curves, time rescaling or bias fitting is
applied. Send-relative scheduling stretches playback slightly. The full actual
hardware core is shown; the simulator curve stops at its nominal five-second
endpoint. Consequently, targets match in order and value, but their timing is not
identical. End-position samples are asynchronous and are not all at exactly 5.000 s.

Torque compares simulation **submitted actuator effort before passive friction**
with hardware **signed A1 current × measured B0 Kt**, not net wheel torque or an
independent torque sensor. Solver-measured joint-axis effort is not plotted.
Hardware position uses signed A3 multi-turn count minus that run's startup count.
Simulation sums native per-physics-step position increments, resolving its 4π
branch. Neither uses velocity integration, ±π wrapping or a new core-start zero.

The user confirmed all seven had the rear wheel airborne and free, without hand
contact/restraint, and unchanged firmware and supply settings. **The wheel was not
returned to a common physical starting angle.** A relative encoder zero therefore
does not make motor electrical/mechanical starting phase identical. Different
motion is visible, but these plots alone cannot identify its cause or prove
intrinsic motor randomness.

The simulation reference is the original nominal fixed-base case
`isaac400_dr_env0_12p5_17p5_friction_on` (case 4), not the original on-ground policy
episode. The source policy was DR-trained, but this actuator replay has no fresh
DR sampling or inference. Support and actuator dynamics are not fully matched.
No new simulation or hardware was run to make these plots. The original report,
recordings, target file and control code are unchanged.

Reproduce from the repository root:

```bash
python3 -B deploy/experiments/reports/20261029_rear_replay_repeats/plot_comparison.py
```
