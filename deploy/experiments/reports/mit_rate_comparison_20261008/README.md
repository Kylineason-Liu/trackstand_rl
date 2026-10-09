# MIT steering: 200 Hz versus 50 Hz fresh sine targets (2026-10-08)

## Verdict

**A partial improvement, not a faster motor bandwidth or a complete tracking fix.**
With requested Kp=10, Kd=0.1 and ±1 rad sine commands, nominal 200 Hz improves continuous-sine tracking lag by **7.26–7.90 ms**, reduces peak current-derived torque, and makes the 10 Hz position response much more sinusoidal.
But lag measured against the actual held commands changes by less than **0.40 ms**, 5 Hz fitted position amplitude is effectively unchanged, and neither rate achieves full ±1 rad steady stroke at 5/10 Hz.
RMS error against the targets actually sent is slightly worse at 5/10 Hz. At 10 Hz, lower peak torque does **not** mean uniformly lower torque demand.

The user confirms steering/front wheel fully free, the same power supply, and unchanged firmware speed/current limits. These are separate days and one recording per condition; recorded temperatures differ. This is not ground-loaded balancing or replay of a real policy.

## Tracking data

Matched against yesterday's **Kd=0.1**, not Kd=0.2. Requested gains decode to exactly **Kp=10.0122100122 / Kd=0.1001221001** in all six recordings.
Raw payloads verify ±1 rad requests, motor ID 2, gear=1, encoder/torque signs −1, F0 3 rad / 45 rad/s / 18 Nm, and B0 Kt≈0.77999997139 Nm/A.
A3 and F1 each retain their own fixed startup zero; the small differing native F1 zero in yesterday's 2 Hz test is accounted for, not silently merged.

| Sine Hz | Nominal send Hz (measured) | A3 fundamental rad | Gain vs held input % | Lag vs held targets ms | Lag vs continuous sine ms | RMS vs held targets rad | RMS vs continuous sine rad | Position R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 50 (49.73) | 1.0098 | 101.25 | 12.04 | 22.09 | 0.1200 | 0.1974 | 0.9996 |
| 2 | 200 (196.32) | 1.0198 | 102.00 | 11.65 | 14.20 | 0.1146 | 0.1282 | 1.0000 |
| 5 | 50 (49.72) | 0.8047 | 81.83 | 34.85 | 44.90 | 0.6806 | 0.8356 | 0.9918 |
| 5 | 200 (196.28) | 0.8047 | 80.56 | 35.00 | 37.55 | 0.6968 | 0.7214 | 0.9914 |
| 10 | 50 (49.72) | 0.3663 | 39.19 | 30.99 | 41.05 | 0.8457 | 0.9475 | 0.7700 |
| 10 | 200 (196.06) | 0.3860 | 38.76 | 31.24 | 33.79 | 0.8566 | 0.8823 | 0.9947 |

- **2 Hz:** held-target RMS improves **4.55%**; ideal-continuous RMS improves **35.09%**. Fundamental amplitude rises **1.0098 → 1.0198 rad**, so slightly more than unity amplitude is not proof of universally better gain.
- **5 Hz:** A3 fundamental **0.804722 → 0.804732 rad**, essentially identical; held-target RMS worsens **2.37%**, ideal-sine RMS improves **13.66%**. The host is delivering the desired smooth wave sooner; it has not recovered missing motor amplitude.
- **10 Hz:** fundamental amplitude increases **0.3663 → 0.3860 rad**, only about **5.4%**. Held-input gain actually changes **39.19 → 38.76%**, because the input itself has a stronger fundamental at the higher send rate. Position R² rises **0.7700 → 0.9947**, a major improvement in waveform shape, but fundamental phase lag is still **111.58 → 112.46°**, and held-target RMS worsens **1.28%**. Ideal-sine RMS improves **6.88%**. A clean, small, late sine is still poor full-amplitude tracking.

### Why the two lag definitions differ

Phase-equivalent lag is phase divided by 360×frequency, expressed in ms.
Compared with the ideal requested continuous sine, the held input introduces approximately **10.05 ms** at ~49.72 Hz versus **2.55 ms** at ~196 Hz.
This accounts for almost all the observed **7–8 ms** improvement in ideal-sine tracking lag.
Against the input actually sent, the motor response remains approximately **12 / 35 / 31 ms** at 2/5/10 Hz.

These are host-send/receipt phase-equivalent times, not identified pure transport latency, firmware-loop frequency or per-setpoint settling time.
Independent held-input correlation agrees within **0.25 ms** here. At yesterday's distorted 10 Hz waveform, a single lag/fit alone is inadequate.

**Do not extrapolate this directly to deployment:** over **99.94%** of consecutive new encoded targets change, so these logs supply a freshly evaluated sine every ~5.1 ms. A 50 Hz policy produces a new target every 20 ms. Resending its last target four times is not the experiment performed here and does not make that policy a 200 Hz policy. Interpolation/prediction is another controller change, not automatically equivalent either.

## Speed and current-derived torque

| Sine Hz | Nominal send Hz | Peak A2 speed rad/s | Peak torque Nm | Torque RMS Nm | A1 samples above 5.6 Nm % | Steady A3 min/max rad |
|---|---:|---:|---:|---:|---:|---|
| 2 | 50 | 14.850 | 1.602 | 0.536 | 0.000 | -1.0155 / 1.0166 |
| 2 | 200 | 13.105 | 0.575 | 0.245 | 0.000 | -1.0178 / 1.0178 |
| 5 | 50 | 20.573 | 6.359 | 1.689 | 2.072 | -0.9288 / 0.9215 |
| 5 | 200 | 20.528 | 4.728 | 1.624 | 0.000 | -0.9541 / 0.8989 |
| 10 | 50 | 20.580 | 9.728 | 2.764 | 7.805 | -0.6040 / 0.5956 |
| 10 | 200 | 20.538 | 8.034 | 2.911 | 11.319 | -0.4349 / 0.4184 |

A2 still peaks near **20.5–20.6 rad/s** at 5/10 Hz; ideal ±1 rad sine would demand **31.4 / 62.8 rad/s**.
Increasing send frequency has not removed this effective speed plateau; the exact mechanism is unmeasured.
At 5 Hz, |speed|>20 rad/s on **61.41% → 57.99%** of samples. At 10 Hz it is **23.18% → 37.00%**.

Peak torque decreases approximately **64.12 / 25.64 / 17.42%** at 2/5/10 Hz, consistent with reduced command-step ripple, but one test per condition does not isolate its cause.
At 10 Hz **torque RMS increases 5.32%**, and the fraction above the training 5.6 Nm reference rises **7.805 → 11.319%** despite a lower peak.
The higher send rate is therefore **not a verified torque-protection solution**.
A1 current×B0 Kt is a current-derived estimate, not calibrated shaft torque; F0 T_Max is encoding metadata, not proof of total internal-PD protection.

## Timing, conditions and safety

| Sine Hz | Nominal send Hz | Startup/end voltage V | Startup/end temperature °C | Max logged command lateness ms | Query RTT p99 ms |
|---|---:|---|---|---:|---:|
| 2 | 50 | 20.19 / 20.22 | 36 / 42 | 0.335 | 0.882 |
| 2 | 200 | 20.20 / 20.20 | 32 / 38 | 2.889 | 1.085 |
| 5 | 50 | 20.20 / 20.19 | 36 / 44 | 0.278 | 0.872 |
| 5 | 200 | 20.20 / 20.14 | 32 / 38 | 2.909 | 1.081 |
| 10 | 50 | 20.20 / 20.11 | 38 / 44 | 0.252 | 0.880 |
| 10 | 200 | 20.20 / 20.17 | 34 / 44 | 2.476 | 1.072 |

Measured new send rates are **196.06–196.32 Hz**, not exactly 200; yesterday's matched streams **49.72–49.73 Hz**.
A3/A2/A1 receipts remain approximately **196.6–196.9 Hz**, with no >50 ms gaps, no reported socket drops and no logged skipped command ticks.
New maximum command intervals are **7.48–7.91 ms**; successful operation is not proof of lower Pi load (CPU was not measured).
Some queries timed out or arrived late; raw outcomes are retained in provenance, not treated as perfect reception.
Voltage/temperature above are only startup/end samples, not an under-load voltage trace. Current/protection/PI registers were not reread by these logs.

**All three new shutdowns retain the unresolved CF issue:** startup mode 0/fault 0; shutdown mode **5**, fault **0**, and “CF sent but disabled mode not confirmed before shutdown deadline.”
Shutdown payloads at 2/5/10 Hz: **CFE4070100260500 / CFDE071000260500 / CFE1070B002C0500**.
Mode 5 semantics and physical disabling are not established by fault-free status. Do not loosen the deployment acknowledgement checks based on these traces.

## Deployment implication and next experiment

Keep the policy rate at **50 Hz**; this experiment does not justify changing it or claiming the internal PD loop became four times faster.
To decide whether MIT dispatch should be 200 Hz in deployment, compare **50 Hz sending versus 200 Hz resending identical 50 Hz-held targets**, with the same gains, origin, supply and firmware limits.
Use small supported ±0.1/0.2 rad commands first, and record actual send intervals, A3/A2/A1, F1, status and disable acknowledgements. Replaying the same supported-policy target sequence would be more representative than evaluating the sine anew at each send.
Separately verify current/torque protection and CF semantics before unsupported balancing; additional trials should remain independently supported.

## Artifacts, methods and validation

- [Offline HTML index](index.html)
- [Rate comparison PNG](rate_comparison.png) / [PDF](rate_comparison.pdf) / [SVG](rate_comparison.svg)
- [Three-cycle raw tracking PNG](tracking_comparison.png) / [PDF](tracking_comparison.pdf) / [SVG](tracking_comparison.svg)
- [Full-precision CSV](comparison.csv) / [JSON, source hashes and provenance](comparison.json)
- [2 Hz sine / nominal 200 Hz MIT](steering_sine_20261008_1312_mit_200hz_10_01_2hz1rad/steering_sine_report.html) — source `202610081312_200Hz_10_01_2Hz.csv`
- [5 Hz sine / nominal 200 Hz MIT](steering_sine_20261008_1313_mit_200hz_10_01_5hz1rad/steering_sine_report.html) — source `202610081313_200Hz_10_01_5Hz.csv`
- [10 Hz sine / nominal 200 Hz MIT](steering_sine_20261008_1314_mit_200hz_10_01_10hz1rad/steering_sine_report.html) — source `202610081314_200Hz_10_01_10Hz.csv`

Uses the existing `analyze_mit_sine` decoding and numerical functions unchanged; only the new filename label is adapted in memory (`_200Hz_` → `_mit_`).
Recomputed all three 50 Hz baselines from raw CSVs and checked them against yesterday's saved report metrics/hashes.
Read **244,987 rows** across the six CSVs, re-decoding **100,161 motion target/A3/A2/A1/F1 rows**, including **52,469** in today's recordings; F0/B0/status raw bytes also checked.
Exclude first max(1 s, three periods), last max(0.25 s, one period); fit sin/cos/DC on each trace's own host timestamps; held input on a 0.1 ms grid; no phase-shifted errors or smoothed plots.
Ideal-sine RMS uses the fitted requested sine evaluated at the actual A3 timestamps. Three subwindows describe within-record variability, not confidence intervals or independent repeated trials.
Comparison windows align the **requested sine phase only**: each run's target and feedback are translated together, never shifted relative to one another.
Original CSVs, reports, runtime/configuration/policy and `progress_record.md` were not modified; no hardware/ROS/simulator was started by this offline task.

Validation: **29 numerical-analysis tests passed** (`test_mit_sine_analysis.py` and `test_c2_sine_analysis.py`); all three recomputed 50 Hz baselines exactly match prior gain/phase/lag/RMS metrics and source hashes. New F1 gains agree with A3 within **0.001**, correlation lag within **0.25 ms**. Validated **17 SVG / 11 PNG / 5 PDF** artifacts, **36 local HTML links**, all six comparison CSV rows and new per-run JSON metrics. Visually inspected both aggregate plots. Fingerprinted **381 pre-existing files** (including all CSVs, earlier reports, deployment Python/configuration, experiment sources/tests and progress record) before/after and verified every hash unchanged. The temporary plot helper passes Pyflakes. The analysis tests emit only the pre-existing Matplotlib 3-D-projection warning; outputs here are 2-D.

Numerical reproduction from repository root (the existing batch filename parser does not accept the new naming convention directly):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("deploy/experiments").resolve()))
import analyze_mit_sine as m
original_labels = m.labels
m.labels = lambda p: original_labels(Path(p.name.replace("_200Hz_", "_mit_")))
result, traces = m.analyze(Path("deploy/experiments/logs/202610081312_200Hz_10_01_2Hz.csv"))
print(result["steady"])
```

The report generator used only a temporary offline helper; it does not change the repository analyzers or deployment command frequency.
