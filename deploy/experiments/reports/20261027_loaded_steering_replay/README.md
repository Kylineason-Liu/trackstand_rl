# Ground-loaded steering replay compared with airborne tests

Analysis date: 2026-10-27. [Figure gallery](index.html), [summary CSV](summary.csv),
[exact statistics](statistics.json), [startup comparison](startup_comparison.png),
[holding-load comparison](holding_load.png).

**Result:** the new ground-loaded replay completed and closely reproduced the two
airborne responses. Tracking after source second 1 remains approximately **0.60°
RMS**, with roughly **20 ms descriptive waveform lag**. Ground contact produces a
small sustained holding offset and torque, but no large tracking collapse in this
particular supported, upright experiment. Startup lag remains substantial briefly.

## Recording and comparison validity

The three new files are:

- `20261027_190254_424315_scaled_200hz_policy_replay.csv`
- `20261027_190254_424315_scaled_200hz_policy_replay.json`
- `20261027_190254_424315_scaled_200hz_policy_replay_plan.csv`

They are compared with the two initial-transient airborne runs at **18:12:29**
and **18:13:49**, not the older middle-episode replay with different preparation.
The user confirms the front tyre carried its normal weight on the ground, the
frame was supported upright, and steering remained free. Exact normal force and
the frame's freedom to translate were not measured in these recordings.

All three plans are byte-identical. Every run sends the same **1,200 MIT packet
payloads**, including every repeated target. The source is model_2500 Isaac400
DR env-0 episode-1, source seconds **0–5**, followed by a **1-second final hold**;
there is no preparation ramp or pre-hold. Source SHA256:
`c799e80fbc87af307871e35dce7ad409f3e636ff662e99ed498c3800a4cf3ad6`.
The wrapper used airborne and the generic backend used loaded construct the same
actual target plan; the different executable name does not change this input.

The recorded backend/transport hashes match the preserved
[runtime snapshots](../20261027_steering_initial_transient_hardware/runtime_snapshot/).
Requested gains are **Kp=10 Nm/rad, Kd=0.2 Nm s/rad**, velocity and feedforward
zero, nominal packet rate 200 Hz, target rate 50 Hz, query rate 100 Hz. Payload
quantization is identical across runs. Encoder and torque signs are both −1.
B1 was sent once in each run; the confirmed A3 zero is 0 and F1 startup position
code 32767 is unsaturated. These checks establish the firmware reference, not an
independent visual measurement of physical straightness.

The analyzer reconstructs the plans and all payloads and independently decodes
raw A3 position, A2 speed, A1 current and F1 position/torque. Each run completes,
reports no AE fault, and confirms final motor disable. Only steering ID 2 is
addressed; the rear motor is not commanded. All 1,903 loaded motion queries are
recorded as received. Loaded supply is **20.19–20.22 V**, temperature **32–34°C**,
queried B3 current setting **15 A**, and watchdog setting `CD00881300`.
Unqueried firmware settings are not established by the CSV.

## Tracking and timing

Errors use fresh feedback minus the latest encoded target already sent, with no
time shifting, smoothing or mean-error subtraction. Source-window boundaries use
the actual timestamps when the corresponding targets were sent.

| Measurement | Airborne 1 | Airborne 2 | Ground-loaded |
| --- | ---: | ---: | ---: |
| Sent/planned packets | 1,200/1,200 | 1,200/1,200 | 1,200/1,200 |
| A3 RMS, complete source 0–5 s | 2.578° | 2.585° | **2.622°** |
| A3 RMS, source 0–1 s | 5.630° | 5.647° | **5.722°** |
| A3 RMS, source 1–5 s | 0.584° | 0.593° | **0.600°** |
| A3 95th-percentile absolute error, source 1–5 s | 1.150° | 1.175° | **1.168°** |
| Largest sampled F1 startup error | 32.869° | 33.132° | **33.116°** |
| Startup waveform alignment, actual 0.1–0.6 s | 19.6 ms | 20.0 ms | **20.2 ms** |
| Later waveform alignment, source 1–5 s | 19.4 ms | 16.8 ms | **19.6 ms** |
| Peak sampled A2 speed | 18.081 rad/s | 17.895 rad/s | **18.175 rad/s** |
| Actual motion duration, nominal 6 s | 6.314 s | 6.289 s | **6.263 s** |
| Actual packet rate | 190.04 Hz | 190.82 Hz | **191.61 Hz** |
| Actual source target-update rate | 47.40 Hz | 47.62 Hz | **47.85 Hz** |

Loaded later RMS increases only **0.007–0.016°** relative to these airborne runs.
Its 95th-percentile error lies between the two airborne values. This is a small
observed difference, not a statistical equivalence claim from one loaded trial.

The first target steps directly from rest to **0.357374 rad /20.476°**, then rises
quickly. Thus the briefly large startup error remains: denser F1 samples expose
about **33.1°** while sparse A3 exposes **25.5°** in the loaded run. First detectable
loaded F1 movement, using a two-A3-count threshold, is bracketed **11.23–16.42 ms**
after the first host send, close to the airborne **11.04–16.23 ms** brackets.
The first second's aggregate RMS should not be confused with a persistent error.

The lag values minimize demeaned waveform error over shifts 0–100 ms in 0.1 ms
steps, using actual send times and zero-order-held targets. They describe tracking
alignment, **not pure communication delay or internal controller frequency**.
The loaded startup fit's within-1%-of-minimum-loss range is **17.0–20.8 ms**;
the 0.2–0.6 ms difference from airborne best fits is not a demonstrated delay
increase. Kernel RX and host send-call times are not motor acquisition/wire times.

The send-relative scheduler retains every target but stretches playback. Loaded
maximum packet gap is **9.288 ms** and maximum lateness is **3.557 ms**, below the
configured 20 ms experiment deadline. Actual source timing differs slightly
between runs; an exact simulator comparison must reproduce these timestamps.

## A real effect of ground loading: holding torque and offset

Use the final **0.5 actual seconds**, after the same constant final target has
settled. Values are means, with signed error defined as actual minus sent target.

| Final holding measurement | Airborne 1 | Airborne 2 | Ground-loaded |
| --- | ---: | ---: | ---: |
| A3 position error | −0.040° | −0.025° | **−0.538°** |
| F1 position error | −0.032° | −0.022° | **−0.532°** |
| A1 current × Kt torque | +0.0015 Nm | −0.0067 Nm | **+0.0911 Nm** |
| F1 firmware torque | +0.0040 Nm | +0.0011 Nm | **+0.0892 Nm** |

Both position channels show the loaded motor approximately **0.00939 rad** short
of target. With Kp≈10, the position-error term alone is about **0.094 Nm**,
consistent in scale with the observed holding torque. Quantized feedforward,
velocity fields and feedback contribute additional small terms. A PD controller
without integral action can require a nonzero position error to supply sustained
torque against an external load. This behavior is consistent with ground/support
loading, rather than the old reference-reset bug or a severe tracking failure.
This one holding condition does not identify a universal tyre-friction parameter.

## Current sampling still limits interpretation

Loaded peak **A1 current magnitude is 1.830 A**. Peak **F1 torque is 3.0462 Nm**
at 16.56 ms, equivalent to **3.905 A** using measured Kt=0.77999997 Nm/A.
The airborne F1 peaks correspond to 3.714/3.860 A. These are similar narrow startup
features, not evidence of a large loaded-only torque increase.

The experiment trip checks A1 only. A1 is sampled at approximately **98.87 Hz**,
whereas F1 arrives with MIT replies near the packet rate. Consequently, a completed
run does not establish that peak current remained below 3 A. F1 torque/Kt is a
firmware-based equivalent current, not an independent A1 current measurement or
shaft-torque sensor. The channels are asynchronous and their internal filtering
is unknown. No current threshold or controller parameter was changed here.

## Conclusions and next action

1. **Steering passes this particular loaded replay well after startup.** The
   evidence does not support stationary ground contact causing a dramatic loss of
   target tracking. The motor can reproduce this selected trajectory under the
   tested upright load and nominal 200 Hz resend schedule.
2. **Startup response still needs faithful simulation.** Brief ~33° error and
   ~20 ms descriptive lag are present loaded and airborne. An actuator model
   that follows startup targets much faster may still mislead the policy.
3. **The small holding error is a measurable load effect, not proof the gains
   should be changed.** Do not increase Kp solely to eliminate a 0.54° deficit
   without checking current, dynamic response and simulation agreement.
4. **Next compare this exact initial trajectory and recorded timing with matched
   simulated actuator tests**, and resolve the previously observed rear-motor
   response discrepancy. Verify full-loop observations and timing before using
   steering replay success as grounds for unsupported balance.

This is one upright supported trial, with positive steering targets approximately
20.5–76.8° and small later movements. It does not establish behavior at all angles,
left/right loads, lean angles, wheel speeds or tyre conditions; 50 Hz packet-rate
behavior remains unmeasured for this loaded setup. No policy ran closed-loop on
the bike during this replay. No new simulator run, motor operation, inference or
training was performed for this analysis, and no deployment code or progress
ledger was modified.

## Artifacts and reproduction

- Five plot sets in PNG/SVG/PDF: [tracking](tracking_comparison.png),
  [startup](startup_comparison.png), [holding load](holding_load.png),
  [host timing](timing.png), [source-aligned diagnostic](source_aligned_comparison.png).
- [Summary CSV](summary.csv), [statistics JSON](statistics.json), fresh-A3 aligned
  CSVs for all three recordings, and [verification](verification.json).
- [Analyzer](analyze.py) reuses the existing pure reader and metric functions.
  Main traces/metrics retain original timestamps. Only the explicitly labelled
  source-aligned diagnostic remaps time to nominal target-plan coordinates.

```bash
python3 -B deploy/experiments/reports/20261027_loaded_steering_replay/analyze.py
```
