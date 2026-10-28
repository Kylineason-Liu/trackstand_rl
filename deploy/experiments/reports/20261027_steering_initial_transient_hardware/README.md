# Initial-transient steering replay: two nominal 200 Hz runs

Analysis date: 2026-10-27. Open the [figure gallery](index.html),
[startup tracking](startup_detail.png), [current sampling](current_sampling.png),
[summary CSV](summary.csv) or [exact statistics](statistics.json).

**Result:** both runs complete with a valid zero and closely repeated responses.
The motor has substantial brief startup lag, then sub-degree RMS tracking over
source seconds 1–5. Completion at 200 Hz does not establish that current stayed
below 3 A: F1 contains a short torque peak equivalent to about 3.7–3.9 A, while
the A1-only abort input misses it. Failed 50 Hz recordings are unavailable, so
the reason those runs trip cannot be conclusively attributed to packet rate.

## Inputs, setup and validation

The six new files under [logs/](../../logs/) are the CSV, JSON and plan for:

- **Run 1:** 20261027_181229_866490_scaled_200hz_steering_initial_transient
- **Run 2:** 20261027_181349_675883_scaled_200hz_steering_initial_transient

User confirms the front wheel was airborne/free, steering physically straight,
no hand contact or restraints, and no settings changes. The user also confirms
that no failed 50 Hz logs were collected and neither current-abort nor lateness
flag was changed. No failed 50 Hz files were provided here; automatic partial
recordings may still exist on the Pi. Both successful-run metadata files record
**3 A /20 ms**, respectively.

Both replay the same model_2500 Isaac400 DR env-0 episode-1 source **0–5 s**,
followed by a one-second final hold, with no preparatory ramp. Source SHA256
is `c799e80fbc87af307871e35dce7ad409f3e636ff662e99ed498c3800a4cf3ad6`.
The two plans are identical, with **300 nominal 20 ms intervals**; every target
has four identical MIT packets, giving **1,200/1,200 sent packets per run**.
Every raw payload was checked against the source and reconstructed runtime plan.

Runtime hashes match the local scripts, preserved in [runtime_snapshot/](runtime_snapshot/).
B1 was sent once; both A3 zero counts are **0**, F1 startup code **32767** is
unsaturated, and every motion F1 status is 1. No AE fault was recorded and the
steering motor was confirmed disabled after both runs. Only ID 2 was addressed.
Gains remain requested Kp=10/Kd=0.2, requested velocity/feedforward zero; encoded
values have the same finite quantization as the previous tests.

## Tracking: distinguish startup from later motion

Errors below use fresh A3 against the latest encoded target already sent, without
time shifting, smoothing or subtracting mean error. Source-window boundaries are
mapped to **actual command send timestamps**, since the replay is stretched.

| Measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| Full recorded source 0–5 s, A3 RMS | 2.578 degrees | 2.585 degrees |
| Source 0–1 s, A3 RMS | 5.630 degrees | 5.647 degrees |
| Source 1–5 s, A3 RMS | **0.584 degrees** | **0.593 degrees** |
| Source 1–5 s, mean signed A3 error | +0.024 degrees | +0.047 degrees |
| Source 1–5 s, 95th-percentile absolute A3 error | 1.150 degrees | 1.175 degrees |
| Largest sampled A3 error during startup | 25.397 degrees | 25.397 degrees |
| Largest sampled F1 error during startup | **32.869 degrees** | **33.132 degrees** |
| Peak sampled A2 speed | 18.081 rad/s | 17.895 rad/s |

The first target is **0.357374 rad /20.476 degrees** from rest; the next two are
0.623081 and 0.888993 rad, nominally 20 ms apart. The motor cannot instantly reach
each step. The largest F1 errors occur at about **21.77/21.50 ms**, immediately
after the second target. Denser F1 exposes a larger peak than approximately
99 Hz A3 sampling. This is brief dynamic lag, not the old fixed 10–12 degree
initial-reference defect. Both position channels agree closely on the response
shape. The [startup graph](startup_detail.png) shows the first 300 ms directly.

Using a two-A3-count motion-detection threshold on F1 position, the last quiet
and first detected-motion samples bracket approximately **11–16 ms after the
first host send** in both runs. A waveform-alignment scan over actual 0.1–0.6 s
fits **19.6/20.0 ms**; later fits are **19.4/16.8 ms**. These are descriptive
response timings, not precise transport latency or the motor's internal PD rate.
The target changes during startup, so classic fixed-step settling/rise times
cannot be directly inferred. Mean removal is used only in the labeled lag fit;
no physical plot or ordinary error table shifts/corrects the response.

Final-half-second F1 mean errors are **−0.032/−0.022 degrees**. The aggregate
startup and subsequent tracking are repeatable in this pair, but two repetitions
do not establish all-condition reliability or ground-loaded performance.

## Actual timing after relaxing the deadline

| Timing / communication | Run 1 | Run 2 |
| --- | ---: | ---: |
| Actual motion duration, nominal 6 s | **6.3144 s** | **6.2885 s** |
| Actual packet rate | 190.0435 Hz | 190.8227 Hz |
| Actual new-target rate over the recorded core | 47.3974 Hz | 47.6202 Hz |
| Packet interval mean /maximum | 5.2620 /8.2288 ms | 5.2405 /9.5502 ms |
| Maximum lateness relative to next deadline | 2.7411 ms | 3.9687 ms |
| A1 feedback rate | 98.8472 Hz | 98.6107 Hz |
| Completed motion queries, all received | 1,920 | 1,908 |

The relaxed scheduler retained every target without catch-up bursts. It also
made the input approximately 5% slower than an exact 50 Hz source playback.
This must be accounted for in an exact simulator comparison or rate experiment;
the default packet-rate label alone does not define identical target timing.
The successful second run includes a lateness sample above the former 3 ms
threshold. Recorded supply stays **20.19–20.21 V**, temperatures **32–34 C**.
These successful runs show no logged fault/feedback timeout, not proof of zero
wire loss or a universal real-time guarantee.

## Why a 50 Hz current trip does not establish worse motor control

The abort checks **A1 current only**: its decoder updates ReplayIO.current,
then decode_reply/guard stop if abs(current)>3. F1 torque is decoded and logged
but does not update that current variable. See
[A1 guard](../../steering_policy_replay.py) and
[F1 decoder](../../steering_sine_mit_io.py).

| Startup observation | Run 1 | Run 2 |
| --- | ---: | ---: |
| Largest A1 current magnitude | **1.737 A** | **2.303 A** |
| F1 reported torque peak | **2.8967 Nm** | **3.0110 Nm** |
| That torque divided by measured B0 Kt | **3.7137 A** | **3.8602 A** |
| Time of that F1 peak | 16.266 ms | 16.321 ms |
| Nearby A1 query-reply times | 12.016 /21.978 ms | 12.070 /22.438 ms |

Measured B0 Kt is **0.77999997 Nm/A**. Thus the 3 A trip corresponds to about
**2.34 Nm**. Each run has one F1 torque sample above that equivalent threshold,
between the A1 samples. Raw examples:

- Run 1, **F17FC27B96B601**: low 12 torque bits are 1718, decoded at ±18 Nm to
  native −2.896703 Nm, policy sign −1 gives +2.896703 Nm.
- Run 2, **F17FC77B66A901**: torque code 1705 similarly gives +3.010989 Nm.

The [CAN protocol PDF](<../../../../tsinghua_bike_project/assign_2_Hardwares/assign_2_Hardwares/3.RearSteer Servo【GIM3510-8】/CAN Protocol Doc/自定义CAN通信协议_3.09b0.pdf>)
pages 21–22 document F1 real-time torque, its 12-bit packing, and command replies
with the same format. F1 torque/Kt is an **equivalent current inferred from a
firmware torque report**, not a second A1 reading or independent shaft sensor.
Internal filtering/acquisition timing is unestablished; the two channels are
not simultaneous. The result nonetheless shows that A1-only completion cannot
be treated as proof that the physical peak stayed below 3 A.

The strongest supported explanations for the reported 50 Hz stops are:

1. **The experimental 3 A trip intersects an ordinary startup transient.**
   Even the initial nominal stationary Kp*error is 3.573741 Nm, equivalent to
   roughly 4.58 A before actual motor/current dynamics. This calculation is a
   demand estimate, not a measured peak. The 3 A value is a chosen experiment
   trip, not the motor's documented fault or continuous-current rating.
2. **A1 sampling phase may catch the narrow peak at 50 Hz and miss it at 200 Hz.**
   The new data directly shows asynchronous channels sampling different maxima.
   This is a plausible explanation, not a proven reconstruction of unrecorded
   50 Hz failures.
3. **Actual target timing or firmware response may differ with packet rate.**
   Four repeated packets and accumulated scheduling overhead stretch source
   timing at 200 Hz. Hardware MIT refresh behavior is not independently known.
   These possibilities need measured 50 Hz traces before ranking them conclusively.

There is no evidence here that CAN congestion caused a current-abort error, that
200 Hz inherently requires less current, or that 50 Hz steering is unsuitable.
Healthy sampled motor status is consistent with a host experiment trip rather
than a reported motor fault.

## Next verification and limits

Record a failed 50 Hz attempt under the same supported/free-wheel conditions.
The script already writes CSV, JSON and plan even when it aborts; retain all
three. Compare the first 100 ms, including raw A1/F1, latest target, position,
velocity and actual send/query times. A short failed trace is useful evidence.
Do not change gains/target waveform at the same time as investigating rate.

Decide a hardware-justified current trip and apply it consistently to any
channels intended to protect against peak current. The present A1-only 3 A
check is not a continuous motor-current clamp. Automatically increasing it or
averaging away startup peaks would change the experiment/protection behavior;
**no such code change was made in this analysis**.

Previous middle-episode simulated replay errors are not a matched benchmark for
this new startup sequence. Establish simulator agreement by replaying this same
startup and measured timing with matched support/gains, rather than comparing
unrelated RMS figures. These results concern an airborne motor, not unsupported
balance, loaded tire response, the full actuator envelope or rear-motor control.

## Artifacts and reproduction

- Five PNG/SVG/PDF sets: [tracking](tracking.png), [startup](startup_detail.png),
  [current sampling](current_sampling.png), [host timing](host_timing.png),
  [repeatability](repeatability.png). Main plots use actual timestamps; only the
  repeatability diagnostic remaps/interpolates to nominal plan coordinates.
- [Exact statistics](statistics.json), [summary CSV](summary.csv),
  [raw startup current samples](startup_current_samples.csv), two fresh-A3 aligned
  CSVs, [runtime snapshots](runtime_snapshot/) and [verification](verification.json).
- Source/plan/payload/raw decoding and an independent standard-library numerical
  check are documented in verification.json. Original recordings, production
  scripts, old figures/statistics and progress ledgers are preserved.

```bash
python3 -B deploy/experiments/reports/20261027_steering_initial_transient_hardware/analyze.py
```

No real CAN, motor operation, new simulator run or policy inference was performed.
