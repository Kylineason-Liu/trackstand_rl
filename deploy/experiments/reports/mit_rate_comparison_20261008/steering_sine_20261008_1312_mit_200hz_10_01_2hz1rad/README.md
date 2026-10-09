# MIT steering sine: 2 Hz, ±1 rad, Kp=10, Kd=0.1

| Mode | Hz | Kd | A3 amp rad | Gain % | Phase ° | Lag ms | RMS error rad | Peak torque Nm | Position R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MIT / nominal 200 Hz | 2 | 0.1 | 1.0198 | 102.0 | 8.4 | 11.65 | 0.1146 | 0.575 | 1.0000 |

Gain = fitted A3 fundamental / recorded **encoded held-target** fundamental.
The A3 amplitude column is the actual sinusoidal component, not a full-record peak or fitted gain.
Steady window: 1.504738–19.807837 s. Method: Exclude first max(1 s, 3 periods) and last max(.25 s, 1 period); fit sin/cos/DC at each trace's own timestamp; held encoded input on .1 ms grid; RMS error against latest encoded target, no alignment shift or smoothing.
Phase-equivalent lag is **not pure motor delay**. Position R² measures sinusoidal fit quality, not controller stability.
Subwindow ranges in analysis.json are within-record variability, not repeat-trial confidence intervals.

## Recording verification

- 19,951 target/A3/A2/A1/F1 rows decoded again from raw payloads; conversions and references checked.
- Encoded settings: `{"mit_kp_nm_per_motor_rad": 10.012210012210012, "mit_kd_nm_s_per_motor_rad": 0.10012210012210013, "mit_target_velocity_motor_rad_s": 0.01098901098901095, "mit_feedforward_torque_motor_nm": 0.00439560439560438}`. Quantization makes requested zero velocity/feedforward slightly nonzero.
- Target rate 196.325 Hz; A3 196.634 Hz; maximum command lateness 2.889 ms; skipped ticks 0.
- Read outcomes: `{"0xA1/received": 3992, "0xA2/late_reply": 2, "0xA2/received": 3991, "0xA3/late_reply": 2, "0xA3/received": 3989, "0xA3/timed_out": 2}`. Late/unmatched replies can still appear as received measurements; host receipt is not acquisition time.
- Socket-reported drops maximum 0; F1 motion fault rows 0; gaps: `{"MIT encoded position target": [], "A3 position feedback": [], "A2 velocity feedback": [], "A1 current-derived torque": []}`.
- F1 position cross-check: gain 102.02%, phase 8.03°. It is another motor report, not an independent physical sensor.
- A2 peak 13.105 rad/s; ideal continuous sine demands 12.566 rad/s.
- A1 samples above training's 5.6 Nm reference: 0.000%. This is a sample fraction, not an exact time fraction or verified mechanical overload.

## Important shutdown discrepancy

**The log does not confirm disabled mode after CF.** Events: `[{"phase": "startup", "elapsed_s": 0.004087862995220348, "note": "B1/F0/A3/F1 verified; startup references fixed"}, {"phase": "motion", "elapsed_s": 20.308805250999285, "note": "Stopping: KeyboardInterrupt: "}, {"phase": "shutdown", "elapsed_s": 20.55897019999975, "note": "CF sent but disabled mode not confirmed before shutdown deadline"}]`.
Status/raw bytes: `[{"phase": "startup", "elapsed_s": 0.0004884549998678267, "payload_hex": "CFE4070000200000", "mode": 0, "fault": 0, "voltage_v": 20.2, "temperature_c": 32}, {"phase": "shutdown", "elapsed_s": 20.309649201997672, "payload_hex": "CFE4070100260500", "mode": 5, "fault": 0, "voltage_v": 20.2, "temperature_c": 38}]`.
The script expects CF byte 6 = 0; these recordings end with byte 6 = 5, fault byte = 0.
The local protocol documents mode 0 as disabled, but does not define mode 5 in the AE table.
This could be firmware reply/transition semantics or failure to disable; these logs do not distinguish them.
Do not treat a successfully sent CF or a fault-free reply as physical disable. Check subsequent AE/F1 and physical free state with independent support before powered deployment.
No final-hold/settling result is inferred: the experiment sends CF rather than holding a final position.

## Artifacts and limitations

- [Four-panel selectable HTML](steering_sine_report.html)
- [Full PNG](steering_sine_report.png) / [PDF](steering_sine_report.pdf)
- [Middle-second PNG](steering_sine_zoom.png) / [three-cycle overlay](tracking_overlay.png)
- [Plot summary](summary.json) / [complete tracking analysis](analysis.json)
- [Source CSV](../../../logs/202610081312_200Hz_10_01_2Hz.csv)

Offline free-steering sine comparison, not a balancing trial. Yesterday steering was confirmed free; user confirms the same free steering, power supply and firmware speed/current limits today. Host send/receipt clocks, not motor acquisition timestamps. Phase-equivalent lag is frequency-specific, not pure firmware delay. A1 torque is current times B0 Kt, not calibrated shaft torque. F0 encoding ranges do not verify total PD limits. Different days/temperatures and one recording per condition limit causality. 200 Hz sends contain fresh sine samples, not repeated held 50 Hz policy targets.
No B2/B3 or position/speed PI gain readback is present. F0 ranges (3 rad,45 rad/s,18 Nm) are encoding metadata, not verified total-PD safety limits.
Standalone sine does not reproduce deployment's target slew, safety limits, rear motor, policy or ground-loaded bicycle.

Source SHA-256: `1d4bc1cef8ddbc0abd8dca8180f2258643761c65b6ea5c5d570ba5e2b957c1f4`.
