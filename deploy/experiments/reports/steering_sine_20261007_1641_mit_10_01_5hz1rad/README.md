# MIT steering sine: 5 Hz, ±1 rad, Kp=10, Kd=0.1

| Mode | Hz | Kd | A3 amp rad | Gain % | Phase ° | Lag ms | RMS error rad | Peak torque Nm | Position R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MIT | 5 | 0.1 | 0.8047 | 81.8 | 62.7 | 34.85 | 0.6806 | 6.359 | 0.9918 |

Gain = fitted A3 fundamental / recorded **encoded held-target** fundamental.
The A3 amplitude column is the actual sinusoidal component, not a full-record peak or fitted gain.
Steady window: 1.004577–26.502123 s. Method: Exclude first max(1 s, 3 periods) and last max(.25 s, 1 period); fit sin/cos/DC at each trace's own timestamp; held encoded input on .1 ms grid; RMS error against latest encoded target, no alignment shift or smoothing.
Phase-equivalent lag is **not pure motor delay**. Position R² measures sinusoidal fit quality, not controller stability.
Subwindow ranges in analysis.json are within-record variability, not repeat-trial confidence intervals.

## Recording verification

- 18,467 target/A3/A2/A1/F1 rows decoded again from raw payloads; conversions and references checked.
- Encoded settings: `{"mit_kp_nm_per_motor_rad": 10.012210012210012, "mit_kd_nm_s_per_motor_rad": 0.10012210012210013, "mit_target_velocity_motor_rad_s": 0.01098901098901095, "mit_feedforward_torque_motor_nm": 0.00439560439560438}`. Quantization makes requested zero velocity/feedforward slightly nonzero.
- Target rate 49.724 Hz; A3 196.874 Hz; maximum command lateness 0.278 ms; skipped ticks 0.
- Read outcomes: `{"0xA1/received": 5268, "0xA2/late_reply": 1, "0xA2/received": 5267, "0xA3/received": 5268, "0xA3/timed_out": 1}`. Late/unmatched replies can still appear as received measurements; host receipt is not acquisition time.
- Socket-reported drops maximum 0; F1 motion fault rows 0; gaps: `{"MIT encoded position target": [], "A3 position feedback": [], "A2 velocity feedback": [], "A1 current-derived torque": []}`.
- F1 position cross-check: gain 81.76%, phase 61.77°. It is another motor report, not an independent physical sensor.
- A2 peak 20.573 rad/s; ideal continuous sine demands 31.416 rad/s.
- A1 samples above training's 5.6 Nm reference: 2.072%. This is a sample fraction, not an exact time fraction or verified mechanical overload.

## Important shutdown discrepancy

**The log does not confirm disabled mode after CF.** Events: `[{"phase": "startup", "elapsed_s": 0.003958256000260008, "note": "B1/F0/A3/F1 verified; startup references fixed"}, {"phase": "motion", "elapsed_s": 26.764363958000104, "note": "Stopping: KeyboardInterrupt: "}, {"phase": "shutdown", "elapsed_s": 27.014478030000646, "note": "CF sent but disabled mode not confirmed before shutdown deadline"}]`.
Status/raw bytes: `[{"phase": "startup", "elapsed_s": 0.0004785670007549925, "payload_hex": "CFE4070000240000", "mode": 0, "fault": 0, "voltage_v": 20.2, "temperature_c": 36.0}, {"phase": "shutdown", "elapsed_s": 26.765098335999937, "payload_hex": "CFE3070B002C0500", "mode": 5, "fault": 0, "voltage_v": 20.19, "temperature_c": 44.0}]`.
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
- [Source CSV](../../logs/202610071641_mit_10_01_5Hz.csv)

Free steering/front wheel confirmed by user; not ground-loaded balancing. Host send/receive times, not wire, acquisition or execution times. Phase-equivalent lag is frequency-specific, not identified pure delay. A1 torque is Q-current times read B0 Kt, not an independent shaft sensor. C2 ran approximately 197 Hz and MIT approximately 50 Hz; different days and unrecorded C2 firmware gains/limits prevent a strictly controlled mode-only comparison. No identified C3 experimental record: C3 performance cannot be quantified here.
No B2/B3 or position/speed PI gain readback is present. F0 ranges (3 rad,45 rad/s,18 Nm) are encoding metadata, not verified total-PD safety limits.
Standalone sine does not reproduce deployment's target slew, safety limits, rear motor, policy or ground-loaded bicycle.

Source SHA-256: `4b5c448e6236578e418895e0000d1e8cd4068f136b38764f3ab311d8e07fb6b6`.
