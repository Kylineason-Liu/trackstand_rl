# C2 steering sine sweep (2026-10-03 / 04)

## Result

All 12 recorded commands match their filename frequencies and amplitudes (`01rad` = ±0.1 rad; `05rad` = ±0.5 rad).
The user confirmed free steering and unchanged motor conditions. The C2 response is repeatable but substantially attenuated and phase-lagged,
including small demands. These data support limited effective position-tracking bandwidth, not a single fixed motor delay.

| Hz | Amplitude rad | Actual amplitude rad | Gain % | Phase lag deg | Lag ms | Error RMS rad | Peak speed rad/s | Peak torque Nm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.00 | 0.1 | 0.0635 | 63.5 | 50.5 | 70.12 | 0.0546 | 1.09 | 0.325 |
| 2.00 | 0.5 | 0.3311 | 66.2 | 49.0 | 67.99 | 0.2675 | 4.48 | 0.593 |
| 2.00 | 1 | 0.6617 | 66.2 | 48.6 | 67.51 | 0.5314 | 8.48 | 0.619 |
| 2.00 | 2 | 1.3239 | 66.2 | 48.5 | 67.30 | 1.0593 | 16.95 | 0.948 |
| 5.00 | 0.1 | 0.0324 | 32.5 | 72.5 | 40.29 | 0.0675 | 1.25 | 0.355 |
| 5.00 | 0.5 | 0.1676 | 33.6 | 71.1 | 39.50 | 0.3349 | 5.45 | 0.760 |
| 5.00 | 1 | 0.3363 | 33.7 | 71.0 | 39.42 | 0.6694 | 10.74 | 1.531 |
| 5.00 | 2 | 0.6745 | 33.8 | 71.1 | 39.48 | 1.3398 | 21.93 | 2.480 |
| 10.00 | 0.1 | 0.0168 | 16.9 | 80.3 | 22.31 | 0.0698 | 1.30 | 0.370 |
| 10.00 | 0.5 | 0.0889 | 17.9 | 81.5 | 22.63 | 0.3497 | 5.86 | 1.417 |
| 10.00 | 1 | 0.1786 | 17.9 | 81.2 | 22.55 | 0.6996 | 12.07 | 2.566 |
| 10.00 | 2 | 0.3543 | 17.8 | 81.5 | 22.65 | 1.3998 | 22.06 | 5.789 |

Gain uses the fundamental of the actual held C2 targets, not peak-to-peak extrema that include startup/final hold.
Every row uses a documented steady window in comparison.json: exclude first max(1 s,3 periods), last max(.25 s,1 period).
Lag is phase divided by 2πf. Peaks/error use that same window, not the full record. Torque is current-derived, not a shaft measurement.

## Conclusions and limits

1. **Attenuation is not merely the old ±2 rad/10 Hz speed demand.** At ±0.1 rad, demands at 2/5/10 Hz are only approximately 1.26/3.14/6.28 rad/s,
   all below deployment's 23.9 rad/s reference, yet the gain is still about 0.64/0.33/0.17. Larger amplitudes yield similar gain/phase at the same frequency.
   Thus limited closed-loop tracking response is strongly supported. Exact active speed/current limits and firmware gains remain unmeasured.
2. **A phase-equivalent lag is not fixed latency.** It varies approximately 67–70 ms at 2 Hz, 39–40 ms at 5 Hz, and 22–23 ms at 10 Hz,
   while phase lag increases. The earlier approximately 23 ms estimate remains specific to 10 Hz, not a universal actuator delay.
3. **A simple low-pass model summarizes the larger-amplitude records.** Equal-weight complex-response fit of ±0.5/1/2 rad gives
   `H(jw)=K/(1+jw*tau)`, K=1.0053, tau=90.35 ms, model-implied corner 1.762 Hz;
   complex-response RMS residual 0.00457, maximum phase discrepancy 1.53°.
   This is a descriptive fit, **not a uniquely identified firmware/physical model**. Its corner is below the lowest tested 2 Hz and is not directly measured.
   The 0.1 rad points are not used to fit it; compare their small additional attenuation/phase lag rather than inventing a deadband or stiction threshold.
4. **C2 bench behavior is promising, not deployment certification.** Position traces are largely sinusoidal rather than growing autonomous oscillations,
   Final recorded one-second hold windows have median target errors at most **0.000383495 rad (one encoder count, about 0.022°)**,
   and maximum instantaneous error **0.001533981 rad (four counts, about 0.088°)** across these recordings.
   These are observations after the sine stops changing, not a controlled step or motor-disable test. Current-derived torque still fluctuates.
   There is no matched C0/C2 test, 50 Hz target-interface test, ground-loaded steering, or balancing policy experiment here.
   A 50 Hz policy does not mean it demands 50 Hz physical steering motion; actuator compatibility depends on the spectrum of its actual commands.
5. **Improve/characterize the actuator interface before judging the policy.** Read back B2/B3, B6–B9 and AE/status/voltage; do not copy host PD gains into firmware PI registers.
   Extend to 0.2/0.5/1 Hz and supervised small steps/enable transitions, then repeat small sine commands at the actual 50 Hz policy interface.
   Check loaded behavior while independently supporting the bike. Do not blindly increase gains, disable protections, or retry unsupported balance based on this sweep.

## Recording integrity

All source hashes and count-to-angle/current-to-torque conversions were checked; every steering query row contains its separate timestamp/value.
The 2 Hz/±2 rad record has approximately 342 ms and 106 ms feedback gaps before the selected steady window;
there are no >50 ms A3/A2/A1 gaps inside any selected steady window. Full plots mark feedback outages rather than connect through them.
They establish receipt/recording gaps, not the cause or the motor's physical behavior inside them.

## Artifacts

- [Comparison PNG](c2_sweep_comparison.png) / [PDF](c2_sweep_comparison.pdf) / [SVG](c2_sweep_comparison.svg)
- [Small-amplitude tracking overlays](small_amplitude_tracking.png)
- [Comparison CSV](comparison.csv) / [complete JSON and SHA-256 provenance](comparison.json)
- [Offline HTML index](index.html)
- [Validation and reproducible numerical tests](VALIDATION.md)

- [2 Hz / ±0.1 rad](../steering_sine_20261004_1321_2hz01rad/steering_sine_report.html)
- [2 Hz / ±0.5 rad](../steering_sine_20261004_1319_2hz05rad/steering_sine_report.html)
- [2 Hz / ±1 rad](../steering_sine_20261004_1318_2hz1rad/steering_sine_report.html)
- [2 Hz / ±2 rad](../steering_sine_20261004_1317_2hz2rad/steering_sine_report.html)
- [5 Hz / ±0.1 rad](../steering_sine_20261004_1325_5hz01rad/steering_sine_report.html)
- [5 Hz / ±0.5 rad](../steering_sine_20261004_1324_5hz05rad/steering_sine_report.html)
- [5 Hz / ±1 rad](../steering_sine_20261004_1323_5hz1rad/steering_sine_report.html)
- [5 Hz / ±2 rad](../steering_sine_20261004_1322_5hz2rad/steering_sine_report.html)
- [10 Hz / ±0.1 rad](../steering_sine_20261004_1313_10hz01rad/steering_sine_report.html)
- [10 Hz / ±0.5 rad](../steering_sine_20261004_1310_10hz05rad/steering_sine_report.html)
- [10 Hz / ±1 rad](../steering_sine_20261004_1315_10hz1rad/steering_sine_report.html)
- [10 Hz / ±2 rad](../steering_sine_20261003_1439/steering_sine_report.html)

Each new condition includes the existing workflow's PNG/PDF, five SVG windows and selectable four-panel HTML,
plus README.md, analysis.json, and a three-cycle target/feedback overlay. The original 10 Hz/±2 rad report is preserved byte-for-byte;
its new comparison statistics and relocated source link are in this sweep, not written over the old report.

## Measurement caveats / reproduction

Recorded C2 is observer SocketCAN-loopback receipt, not exact motor acceptance. A3/A2/A1 are sequential host-receipt times.
Plots retain raw values and clocks with no smoothing or phase shifts. Fits/correlation use documented numerical sampling separately.
Subwindow ranges are within-record variability, not repeat-trial confidence intervals. No firmware sampling rate can be inferred.
Last-C2 settling assumes no later control command outside this capture; the raw sine script does not disable on exit.

From the repository root: `MPLCONFIGDIR=/tmp/trackstand_c2_mpl python3 deploy/experiments/analyze_c2_sine.py --reports /tmp/new_c2_sweep_reports`.
Use a fresh output directory: this helper refuses to overwrite existing reports. It never opens CAN or modifies source CSVs/runtime/configurations.
