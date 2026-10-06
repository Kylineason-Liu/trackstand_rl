# C2 steering sine analysis: 5.00 Hz, ±0.5 rad

| Hz | Amplitude rad | Actual amplitude rad | Gain % | Phase lag deg | Lag ms | Error RMS rad | Peak speed rad/s | Peak torque Nm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.00 | 0.5 | 0.1676 | 33.6 | 71.1 | 39.50 | 0.3349 | 5.45 | 0.760 |

Measured window: 6.43676–21.90996 s.
Gain is the ratio of fitted A3 amplitude to the fundamental of the recorded held C2 targets.
Lag is frequency-specific phase divided by 2πf, **not pure motor/CAN delay**.
Peak speed/torque and RMS error use the same steady window; torque is A1 Q-current × B0 Kt, not a shaft sensor.

- [Five-window four-panel HTML](steering_sine_report.html)
- [Full-record PNG](steering_sine_report.png) / [PDF](steering_sine_report.pdf)
- [Middle-second PNG](steering_sine_zoom.png) / [three-cycle overlay](tracking_overlay.png)
- [Plot provenance](summary.json) / [tracking calculations](analysis.json)

Method: Discard first max(1 s,3 periods), last max(.25 s,1 period); sin/cos/DC fits at each signal's own receipt times; recorded ZOH command on .1 ms grid; error against latest target; no time shifting.

Timestamp caveat: Host receipt times only; C2 loopback is not motor acceptance; A3/A2/A1 are sequential.
Subwindow ranges reflect within-record variability, not a statistical confidence interval.
No firmware gains/limits/status/voltage readbacks are present. Neither this analysis nor the raw sine helper configures firmware limits.
Raw helper exit does not disable the motor. Grey plot shading does not mean disabled.
Fixture: User: steering free to move, motor conditions the same across tests; exact firmware readbacks not logged.

Source SHA-256: `7b6f12ee39cc82ee75dd37901cbd16583832675060780fcf74609ed6688fe785`.
Feedback gaps >50 ms (recorder seconds): `{"A3": [], "A2": [], "A1": []}`.
The plotting helper breaks feedback lines across these gaps and marks them red; no missing measurements are fabricated.
Final-hold statistics in analysis.json are finite-record settling after commands stop changing, not a controlled step experiment.
