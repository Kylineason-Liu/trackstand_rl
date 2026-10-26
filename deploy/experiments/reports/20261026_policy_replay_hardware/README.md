# Real identical-target replay: steering and rear drive, 50 versus 200 Hz

Analysis date: 2026-10-26. All 12 supplied files were found: four measurement CSVs,
four plans and four metadata JSONs. No new motor/CAN operation or simulator run
was performed. Open [the figure gallery](index.html) or [exact statistics](statistics.json).

These recordings predate the steering B1 initialization fix. The matching
[runtime snapshots](runtime_snapshot/) preserve their exact source versions;
hash validation uses those snapshots rather than today's updated replay code.

## Findings

1. **A replay initialization bug biases physical steering by about 10–12 degrees.**
   Both initial MIT F1 readings are at the +3-rad encoding endpoint, although A3
   puts the straight-start shaft above +3 rad. The script accepted this endpoint
   as a valid zero. F1-relative tracking therefore looks good while actual
   startup-relative A3 displacement is too far right. This is a defect in the
   replay I wrote, not proof the steering motor itself has a 10-degree tracking error.
2. **Rear current-derived torque tracks reasonably closely at both rates.**
   A1 RMS error is 0.0158 / 0.0165 Nm at
   50 / 200 Hz. No clear torque-accuracy improvement is established at 200 Hz.
   Current times Kt is an electromagnetic torque estimate, not a shaft sensor.
3. **Rear movement differs greatly despite identical target payloads.** Peak
   backward speed is 8.049 versus
   2.833 rad/s; final displacement is
   -5.282 versus -0.335 rad.
   Near-zero-speed behavior is not adequately established by the old moving-drag fit.
   One run per rate, different initial rotor angles and undersampled current
   transients prevent assigning this difference uniquely to packet rate.
4. **The 200-Hz steering run is incomplete because of a host deadline.** It
   sent 1,074/1,600 planned packets, then disabled successfully. Rear 200 Hz
   completed. This does not establish a motor/CAN 200-Hz limit or a firmware rate.

## Provenance, fixture and comparison boundary

The user confirms both tested wheels airborne/free, no restraint or hand
contact, straight steering startup, actual steering right, and no configuration
or fixture changes between rates. Only one motor was tested at a time.

All four runs use the exact same IsaacLab-source CSV, model **2500**, training
run **2026-10-21_11-18-29**, DR evaluation seed **42**, env **0**, episode **1**.
Source file: `replay_targets/2500_isaac400_env0_episode1.csv`, SHA-256
`c799e80fbc87af307871e35dce7ad409f3e636ff662e99ed498c3800a4cf3ad6`. The raw CSV/JSON bytes, captured runtime hashes,
exact CAN encoding and every transmitted target were checked. Both rate pairs
have byte-identical plans; encoded command payloads match for every common
50-Hz interval. At 200 Hz each reference is repeated four times, not recomputed.

Steering has 1 s ramp + 1 s hold + core 2–7 s + 1 s final hold. Rear has 2 s
zero torque + core 2–7 s, then CF. Core source timestamps are 12.50–17.50 s.
The aborted steering comparison uses common complete intervals **2–5.36 s**;
the remaining successful 50-Hz trace is separately reported. There are no
invented episode resets, interpolated target ramps or motion samples after abort.

| Run | Completed | Packets sent/planned | Achieved host TX Hz | p99 interval ms |
| --- | --- | ---: | ---: | ---: |
| Steering MIT 50 Hz | Yes | 400/400 | 50.000 | 20.153 |
| Steering MIT 200 Hz | No: deadline at ~5.373 s | 1074/1600 | 199.955 | 5.188 |
| Rear C0 50 Hz | Yes | 350/350 | 50.000 | 21.094 |
| Rear C0 200 Hz | Yes | 1400/1400 | 199.999 | 5.163 |

All four JSONs confirm disabled, fault 0. Both rear runs confirm CD restored.
All recorded active steering AE modes are 5 and F1 statuses 1; rear modes are 2.
All finished motion query records say received; this is not proof of zero wire
loss or known motor acquisition age. Supply stays about 20.2 V, with no recorded
large sag; the CSVs do not establish the firmware's internal control frequency.

## Steering: actual position versus a clipped reference

| Run | Available-core A3 RMS error ° | Mean error ° | Common-window RMS ° | Predicted zero bias ° | Demeaned error RMS ° |
| --- | ---: | ---: | ---: | ---: | ---: |
| Steering MIT 50 Hz | 9.905 | +9.876 | 9.962 | 10.090 | 0.754 |
| Steering MIT 200 Hz | 12.119 | +12.094 | 12.119 | 12.244 | 0.773 |

Errors above use fresh A3 multi-turn displacement from the real startup count
against the latest actually sent encoded target, with no shifting or bias removal.
Positive error means too far right. The last column removes only the mean error
to describe shape variation; it does **not** make physical steering correct.

| Rate | Startup A3 count | A3 native rad | Recorded F1 zero rad | F1 code |
| --- | ---: | ---: | ---: | ---: |
| 50 Hz | 8282 | 3.176107221 | 3.000000000 | 65535 |
| 200 Hz | 8380 | 3.213689751 | 3.000000000 | 65535 |

F0 position range is ±3 rad in both records. The script's absolute MIT command
is `recorded_F1_zero - requested_right_angle`, with requested velocity and
feedforward torque zero. When the actual initial native angle exceeds the
recorded F1 value, the requested displacement includes that missing offset.
The initial differences are **0.176107 rad**
and **0.213690 rad**. In the core,
interpolating A3 only to cross-check asynchronous F1 gives mean A3-relative minus
F1-relative differences **0.176202**
and **0.213781 rad**.
The close agreement, plus endpoint plateaus during startup, strongly supports
F1 clipping as the cause. No firmware source or independent angle transducer is
available, so internal saturation implementation is not claimed independently.

F1-based core RMS errors are only
0.893° and
0.808°: using that channel
alone hides the origin error. Native A3 counts decrease and signed A3 displacement
increases while the user observes right steering; this supports the configured
position sign, rather than a sign inversion. Reported torque uses sign −1 too.

The corrected-origin *diagnostic* tracking errors and descriptive lag scans are
saved in JSON. They are not silently substituted into any physical-response plot.
Tracking dynamics appear much closer than the 10–12° absolute errors suggest,
but clean repeat tests require a valid, unsaturated startup reference first.
This replay defect does not establish the cause of earlier policy deployment
failures: normal `motor_mit_io` has a different startup/origin procedure.

## Rear: current tracks, motion is not repeatable in this pair

| Run | A1 torque RMS error Nm | Mean error Nm | p95 absolute error Nm | Minimum speed rad/s | Last A3 displacement rad |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rear C0 50 Hz | 0.015848 | -0.000389 | 0.033000 | -8.0488 | -5.282263 |
| Rear C0 200 Hz | 0.016496 | +0.000458 | 0.033180 | -2.8327 | -0.334791 |

The shared recorded request spans −0.233069 to +0.009307 Nm; encoded C0 spans
approximately −0.233100 to +0.009300 Nm. The negative request means backward
torque under the verified signs; negative measured wheel rotation agrees.
The trace has little positive demand and cannot independently validate strong
forward-torque behavior. Commands are `I_native=-torque/B0`, not a velocity target.

Quantization is at most half a milliamp times B0 (~0.00015 Nm). Error metrics use
the independent A1 query stream (~99 Hz) for both rates, not differently sampled
C0 replies. C0 replies are also plotted as actual measured-current reports.
During initial zero commands, A1 RMS is
0.0119/0.0092 Nm.
Thus current-report/current-loop noise already accounts for a substantial scale
of the core error; the CSV cannot separate true torque ripple from reporting noise.

Wheel speed is within ±0.1 rad/s for
71.7% / 94.3%
of core A1 samples using the latest A2 reading. At these near-zero speeds,
95th-percentile absolute current-derived torque is
0.1576/
0.1698 Nm, above the prior 0.106 Nm
**moving** drag fit. The old speed sweep explicitly did not identify static
breakaway friction. This supports investigating static/angle-dependent drag,
cogging and small-command current dynamics; it does not measure a new universal
friction coefficient or justify adding an arbitrary viscous term.

The starting raw A3 counts differ by 13774
counts (5.282263 rad, 302.65°), so wheel/rotor
phase was not controlled even though firmware settings were unchanged. The mean
torque errors are small, but sub-query transients can differ and are not fully
observed. Rate changes, initial angle and stick-slip effects are confounded.
There is no defensible conclusion that smaller wheel motion means better
200-Hz torque control, or that the motor cannot produce the requested current.

## Timing and delay

| Run | Median interval ms | Maximum interval ms | Maximum successful-TX lateness ms | Max motion RX queue age ms | A1 Hz |
| --- | ---: | ---: | ---: | ---: | ---: |
| Steering MIT 50 Hz | 19.999 | 22.297 | 2.420 | 2.936 | 98.88 |
| Steering MIT 200 Hz | 4.999 | 7.548 | 2.997 | 4.213 | 98.96 |
| Rear C0 50 Hz | 19.999 | 22.017 | 2.114 | 3.467 | 98.90 |
| Rear C0 200 Hz | 4.999 | 6.842 | 2.161 | 4.020 | 99.01 |

The aborted 200-Hz steering command was scheduled at **5.370 s**; the first
logged shutdown TX is at **5.373267 s**. It had
exceeded the 3-ms host lateness threshold. Earlier successful intervals can be
shorter/longer than 5 ms as the scheduler returns toward its absolute grid.
The rejected packet has no successful-TX row. Kernel queue delays demonstrate
host-processing stalls, but do not identify Linux scheduling versus Python
logging contention, CPU load or another specific cause. No profiler was recorded.

Best time-shift fits are descriptive. A 0–100 ms scan compares measured values
to the actual zero-order-held sent targets; steering fits remove a constant
bias, rear fits do not. In the common steering window, best shifts are
20.6/
22.5 ms. Rear best shifts over
the full core are 1.4/
2.0 ms. Interpret these as approximately
**tens of milliseconds for steering response** and **a few milliseconds for
sampled current response**, not fixed pure latency. A1/A3 sampling is about
10 ms, waveforms are not broadband identification inputs, acquisition time is
unknown and timing/noise biases remain. The scan grid is not measurement precision;
the saved near-minimum intervals are not statistical confidence intervals.

## Comparison with the saved simulator replay

The previous supported nominal replay uses exactly the same 50-Hz targets
(Isaac-source case 4, steering losses ON) in Isaac400, MuJoCo400 and MuJoCo800.
No new simulation was run. Steering true-startup A3 is offset from the simulated
response because the hardware reference is biased. Simulated rear peak backward
speeds in the core are **21.40**, **21.76**
and **21.76 rad/s**,
well above the hardware pair's 8.05/2.83 rad/s. Motor torque submissions in those
simulators are ideal; current dynamics/static drag/initial phase are not faithfully
identified. Real fixture versus fixed-root support, nominal parameters and the
steering-origin defect also limit a strict physical-equivalence claim.

## Next steps, in priority order

1. **Fix steering startup-reference validation**, then repeat. Reject endpoint
   F1 zeros and verify A3/F1 consistency. Establish an unsaturated native MIT
   reference with sufficient range for the straight position and target path;
   verify any protocol/range adjustment before actuation. Do not hide the error
   by subtracting 10° from the plotted physical position or change the sign.
2. **Make the timing experiment finish and record the missed deadline itself.**
   Measure host workload and queue age; reduce hot-loop/logging contention or
   revise the explicitly documented abort tolerance from measured Pi jitter.
   Keep actual send timestamps and avoid dropped/burst targets. The present pair
   supports neither a universal 200-Hz failure nor a need for 800-Hz Pi commands.
3. **Identify rear near-zero/breakaway behavior.** Repeat both rates in alternating
   order from the same marked rear-wheel angle, at least several times, with the
   same supply/temperature. Add controlled small-torque plateaus/ramp tests and
   current logging fast enough to resolve transients. Measure static threshold
   versus angle separately from moving drag; independently calibrate torque if
   a mechanical accuracy claim is needed. A 0.106-Nm moving fit is not that test.
4. **Replay faster MuJoCo target sequences only after these checks.** The present
   Isaac core has small steering variation around a large rightward offset and
   small rear torque. It does not validate 10–25 Hz steering reversal tracking,
   loaded tyre-ground behavior, full motor envelopes or unsupported balance.

## Reproduction and artifacts

From repository root:

```bash
python3 -B deploy/experiments/reports/20261026_policy_replay_hardware/analyze.py
python3 -B -m unittest discover -s deploy/experiments/reports/20261026_policy_replay_hardware -v
```

Figures: [steering](steering_tracking.png), [zero-reference defect](steering_zero_reference.png),
[rear torque/motion](rear_tracking.png), [host timing](packet_timing.png),
[saved simulator comparison](simulator_comparison.png). Each also has SVG/PDF.
Legends are at bottom right; physical min/max annotations exclude vertical markers.
Plots show raw, unshifted samples. Only the explicitly labelled cross-check/lag
calculations use interpolation or a shifted reference; no plotted response is altered.
Four `*_aligned.csv` files expose fresh A3 (rad) or A1 (Nm) against the latest
actually sent encoded reference. [statistics.json](statistics.json) includes
full metrics/provenance, and [verification.json](verification.json) records tests
and input/production preservation. Original logs, runtime code, models, training
data, earlier reports and progress ledgers remain unchanged.
