# C2 steering sine analysis: 2.00 Hz, ±2 rad

| Hz | Amplitude rad | Actual amplitude rad | Gain % | Phase lag deg | Lag ms | Error RMS rad | Peak speed rad/s | Peak torque Nm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.00 | 2 | 1.3239 | 66.2 | 48.5 | 67.30 | 1.0593 | 16.95 | 0.948 |

Measured window: 5.99857–24.03460 s.
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

Source SHA-256: `89e762817ee74b96fab25d39483cee404ef1955ab71eb86928f5ffdb1a168f77`.
Feedback gaps >50 ms (recorder seconds): `{"A3": [[0.5556458840001142, 0.8973866579999594], [5.652610894999725, 5.758634764999897]], "A2": [[0.5561171599997579, 0.8979291609998654], [5.6531046529999, 5.759154284000033]], "A1": [[0.5566226969999661, 0.8984845720001431], [5.653598114000033, 5.759677247000127]]}`.
The plotting helper breaks feedback lines across these gaps and marks them red; no missing measurements are fabricated.
Final-hold statistics in analysis.json are finite-record settling after commands stop changing, not a controlled step experiment.
