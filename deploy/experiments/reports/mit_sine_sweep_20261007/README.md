# MIT versus C2 free-steering sine sweep (2026-10-07)

## Results

All eight new MIT recordings have verified ±1 rad sine requests, Kp≈10, and encoded Kd≈0.1 or 0.2.
Front wheel/steering fully unconstrained, per user. Recomputed all three comparable ±1 rad C2 baselines from original CSVs;
their hashes and statistics match the previous sweep. No measured C3 baseline was identified.

| Mode | Hz | Kd | A3 amp rad | Gain % | Phase ° | Lag ms | RMS error rad | Peak torque Nm | Position R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MIT | 1 | 0.1 | 1.0035 | 100.4 | 4.3 | 11.90 | 0.0588 | 0.634 | 1.0000 |
| MIT | 1 | 0.2 | 0.9969 | 99.8 | 7.8 | 21.70 | 0.0988 | 0.797 | 1.0000 |
| MIT | 2 | 0.1 | 1.0098 | 101.2 | 8.7 | 12.04 | 0.1200 | 1.602 | 0.9996 |
| MIT | 2 | 0.2 | 0.9892 | 99.2 | 15.4 | 21.41 | 0.1960 | 1.575 | 0.9999 |
| MIT | 5 | 0.1 | 0.8047 | 81.8 | 62.7 | 34.85 | 0.6806 | 6.359 | 0.9918 |
| MIT | 5 | 0.2 | 0.7790 | 79.2 | 55.1 | 30.63 | 0.6058 | 5.404 | 0.9951 |
| MIT | 10 | 0.1 | 0.3663 | 39.2 | 111.6 | 30.99 | 0.8457 | 9.728 | 0.7700 |
| MIT | 10 | 0.2 | 0.3600 | 38.5 | 104.5 | 29.04 | 0.8124 | 9.846 | 0.8555 |
| C2 | 2 | unknown PI | 0.6617 | 66.2 | 48.6 | 67.51 | 0.5314 | 0.619 | 1.0000 |
| C2 | 5 | unknown PI | 0.3363 | 33.7 | 71.0 | 39.42 | 0.6694 | 1.531 | 0.9998 |
| C2 | 10 | unknown PI | 0.1786 | 17.9 | 81.2 | 22.55 | 0.6996 | 2.566 | 0.9988 |

Gain uses the **held encoded command's fundamental**, not the ±1 rad ideal continuous wave.
At 10 Hz / ~50 Hz MIT setpoints, this fundamental is approximately 0.935 rad; A3 fundamental is only 0.36–0.37 rad.
No 1 Hz C2 recording exists here. Kd is not a transferable C2 PI register.

## Conclusions supported by the recordings

1. **MIT substantially improves low-frequency tracking over recorded C2.** At 2 Hz, MIT gains are 99.2–101.2% versus C2 66.2%;
   phase 8.7–15.4° versus 48.6°; RMS error 0.120–0.196 rad versus 0.531 rad (approximately 63–77% reduction).
   At 1 Hz, MIT achieves approximately unity gain and 0.059–0.099 rad RMS error; no measured C2 comparison at that frequency.
2. **At 5 Hz MIT has higher amplitude, but tradeoffs remain.** Gain 79.2–81.8% versus C2 33.7%.
   Kd=.2 gives phase 55.1° / RMS 0.606 rad versus C2 71.0° / 0.669 rad.
   Kd=.1 gives phase 62.7° / RMS 0.681 rad: higher amplitude does not automatically mean lower total tracking error.
3. **At 10 Hz MIT is not a tracking fix.** Gain 38.5–39.2% versus C2 17.9%, but MIT phase is 104.5–111.6° versus C2 81.2°,
   RMS error 0.812–0.846 rad versus C2 0.700 rad, and A3 sine-fit R² only 0.770–0.855.
   Thus higher fundamental amplitude coexists with worse overall error and substantial nonsinusoidal behavior.
   The finite traces do not prove an autonomous instability or diagnose its exact mechanism.
4. **High-frequency response is consistent with a speed plateau, not setpoint starvation.** 5/10 Hz demand ideal peaks of
   31.4/62.8 rad/s, while A2 peaks remain approximately 20.6 rad/s. At 5 Hz, |speed|>20 rad/s for 46–61% of samples.
   All eight MIT target streams run 49.718–49.728 Hz, maximum logged lateness below 0.336 ms, no skipped command ticks;
   A3/A2/A1 receipt streams approximately 196.8–196.9 Hz, no >50 ms gaps, query RTT p99 below 0.89 ms.
   This makes host scheduling stalls a weak explanation for these particular free-wheel results. It does not identify the
   active speed/current limit, voltage/back-EMF mechanism, firmware sampling frequency, or ground-loaded behavior.
5. **Torque limits are not matched to training.** Steady A1-derived peaks at 10 Hz are 9.73–9.85 Nm, above the training 5.6 Nm reference;
   roughly 7.8–8.7% of samples exceed 5.6 Nm. Kd=.1 at 5 Hz reaches 6.36 Nm (2.1% of samples above 5.6), versus .2 at 5 Hz 5.40 Nm.
   F0 T_Max=18 Nm is encoding metadata, not proof of a safe total-PD clamp. These are current/Kt estimates, not calibrated shaft torque
   or proof of mechanical damage. Read back/calibrate the actual firmware current/torque protection before balancing.
6. **Lower damping is not uniformly better.** Kd=.1 reduces low-frequency lag to approximately 12 ms versus .2 approximately 21 ms,
   but has worse 5/10 Hz RMS error, larger 5 Hz torque and slightly more than unity 1/2 Hz gain.
   The default .2 matches the training's requested damping; do not silently switch the deployment based on one unloaded trial per condition.
   These are frequency-specific phase-equivalent times, not pure transport/motor delays or per-target settling times.
7. **Shutdown verification is unresolved in all eight tests.** Each log ends on KeyboardInterrupt, sends CF, receives CF byte 6=5/fault=0,
   and records `CF sent but disabled mode not confirmed before shutdown deadline`. Startup CF had mode=0.
   Protocol AE mode table defines 0 as disabled but does not define 5; do not invent its semantics or infer permanent enable.
   With independent physical support and a means of cutting power, record subsequent AE/F1 and confirm free state after CF;
   resolve firmware reply/transition semantics before trusting the deployment's acknowledged-disable transitions.

## Scope and next experiment

This supports **MIT as a better low-frequency actuator interface than the recorded C2 settings**, not a repaired/damaged-motor verdict,
universal MIT superiority, or proof that a policy will balance. The 50 Hz host interface can clearly track 1–2 Hz physical motion here;
it does not establish the firmware loop rate or the adequacy of the previous external 200 Hz C0 PD loop.
The absolute MIT target uses fixed F1 zero + signed sine offset; A3 uses its independent fixed encoder zero.
Wire re-decoding and F1 gain/phase cross-checks support the conversion; do not attribute attenuation to an invented sign/unit fault.

Next: first resolve disabled-state verification and firmware total-torque limits. Then repeat **matched 50 Hz C2/MIT** tests at ±0.1/0.2 rad,
0.5/1/2/5 Hz, with gains/limits, voltage, temperature and status recorded, followed by independently supported **ground-loaded** steering.
Small amplitudes avoid the present high-speed demands and better resemble balancing corrections. Inspect actual policy command spectrum,
model the measured tracking/limits in simulation, and only then decide whether gain tuning or policy adaptation is needed.
Changing to C3 is not an evidenced cure: it changes reference semantics, and its performance was not measured by these CSVs.

## Methods, integrity and artifacts

Exclude first max(1 s, 3 periods) and last max(.25 s, 1 period); fit sin/cos/DC at each trace's own timestamp; held encoded input on .1 ms grid; RMS error against latest encoded target, no alignment shift or smoothing. Three subwindows record variability, not independent-repeat confidence intervals.
Each applicable MIT target/A3/A2/A1/F1 payload was re-decoded; sparse rows use **their own elapsed_s**, never cached/imputed measurements.
Independent held-input correlation agrees within 0.5 ms with A3 phase-equivalent lag in these recordings;
F1 fitted gain/phase agrees closely, but shares the motor and receipt-clock limitations, so is not independent physical calibration.
Query timeouts/late replies are retained and counted, not silently called successful or synthesized into missing rows.
F1 reports active and fault-free during all recorded motion; socket overflow metadata reports zero dropped frames.
Fault-free F1 is not evidence of compliance with the 5.6 Nm training cap. No final-hold/step-settling statistics are invented after CF.

Free steering/front wheel confirmed by user; not ground-loaded balancing. Host send/receive times, not wire, acquisition or execution times. Phase-equivalent lag is frequency-specific, not identified pure delay. A1 torque is Q-current times read B0 Kt, not an independent shaft sensor. C2 ran approximately 197 Hz and MIT approximately 50 Hz; different days and unrecorded C2 firmware gains/limits prevent a strictly controlled mode-only comparison. No identified C3 experimental record: C3 performance cannot be quantified here.

- [Offline HTML index](index.html)
- [Comparison PNG](mit_c2_comparison.png) / [PDF](mit_c2_comparison.pdf) / [SVG](mit_c2_comparison.svg)
- [Three-cycle tracking grid](tracking_comparison.png) / [PDF](tracking_comparison.pdf)
- [Comparison CSV](comparison.csv) / [complete JSON with source SHA-256s](comparison.json)

- [1 Hz / Kp 10 / Kd 0.1](../steering_sine_20261007_1637_mit_10_01_1hz1rad/steering_sine_report.html)
- [1 Hz / Kp 10 / Kd 0.2](../steering_sine_20261007_1628_mit_10_02_1hz1rad/steering_sine_report.html)
- [2 Hz / Kp 10 / Kd 0.1](../steering_sine_20261007_1639_mit_10_01_2hz1rad/steering_sine_report.html)
- [2 Hz / Kp 10 / Kd 0.2](../steering_sine_20261007_1632_mit_10_02_2hz1rad/steering_sine_report.html)
- [5 Hz / Kp 10 / Kd 0.1](../steering_sine_20261007_1641_mit_10_01_5hz1rad/steering_sine_report.html)
- [5 Hz / Kp 10 / Kd 0.2](../steering_sine_20261007_1634_mit_10_02_5hz1rad/steering_sine_report.html)
- [10 Hz / Kp 10 / Kd 0.1](../steering_sine_20261007_1642_mit_10_01_10hz1rad/steering_sine_report.html)
- [10 Hz / Kp 10 / Kd 0.2](../steering_sine_20261007_1643_mit_10_02_10hz1rad/steering_sine_report.html)

Each MIT condition has the prior workflow's five selectable four-panel SVG views, full PNG/PDF, zoom PNG,
tracking overlay, README, summary.json and analysis.json. All earlier reports and raw CSVs are preserved.
No runtime, policy, firmware setting, simulation or progress record is modified; no hardware command is sent by this analysis.

Reproduce in a fresh directory: `MPLCONFIGDIR=/tmp/trackstand_mit_mpl PYTHONDONTWRITEBYTECODE=1 python3 deploy/experiments/analyze_mit_sine.py --reports /tmp/new_mit_reports`.
