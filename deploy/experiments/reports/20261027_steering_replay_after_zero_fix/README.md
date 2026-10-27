# Corrected-zero steering replay: 50 versus 200 Hz

Analysis date: 2026-10-27. All **six new files** were found and analyzed: two
measurement CSVs, two plans and two metadata JSONs. Open the [figure gallery](index.html),
[tracking graph](steering_tracking.png), [summary CSV](summary.csv) or
[exact statistics](statistics.json). No motor/CAN operation or new simulation was run.

## Conclusions

1. **The initialization correction worked in both physical tests.** B1 was sent
   exactly once per run, A3 zero count is **0**, and F1 zero is code **32767**,
   corresponding to **−0.0000457771 rad**, rather than a clipped endpoint.
   The old 10–12 degree tracking bias is absent. New five-second core mean
   errors are **+0.020 degrees at 50 Hz /−0.131 degrees at 200 Hz**.
2. **Steering follows this recorded target with about 0.7–0.8 degree RMS error.**
   It does not follow every 20 ms target change perfectly: 95th-percentile
   absolute errors are **1.44/1.53 degrees**, maximum **2.54/2.61 degrees**.
   All error metrics use unshifted, startup-relative A3 feedback against the
   latest encoded target already sent, without removing bias.
3. **200 Hz does not improve tracking in this pair.** Both runs completed and
   disabled successfully. RMS error is slightly larger at 200 Hz, but there is
   only one run per rate, so this is not evidence that 200 Hz causes worse tracking.
   The same 50 Hz reference sequence is sent once or four times per target.
4. **The descriptive response lag is approximately 16–20 ms.** Best core
   waveform-alignment shifts are **16.0/19.5 ms**. This is not a precise pure
   delay measurement, internal firmware rate, or proof of a meaningful 3.5 ms
   difference: A3 is queried at about 100 Hz and motor acquisition/wire-send
   timestamps are unknown.
5. **Hardware response is of the same error scale as the saved identical-input
   simulator replays, with some remaining discrepancy.** Saved nominal RMS
   errors are **0.572 degrees Isaac400 /0.509 MuJoCo400 /0.582 MuJoCo800**, versus
   **0.701/0.788 hardware**. Support, loading and motor/current dynamics are not
   fully matched; this does not certify the simulated plant or unsupported balancing.

## Inputs, fixture and provenance

| Rate | Measurement stem | Rows | Sent/planned MIT packets | Completed/disabled |
| --- | --- | ---: | ---: | --- |
| 50 Hz | 20261027_144059_652887_scaled_50hz_policy_replay | 8,137 | 400/400 | Yes/yes |
| 200 Hz | 20261027_144436_005178_scaled_200hz_policy_replay | 10,550 | 1,600/1,600 | Yes/yes |

Each stem has `.csv`, `.json` and `_plan.csv` under
[experiments/logs](../../logs/). The user confirms **front wheel airborne,
steering free, no restraints or hand contact, physically straight before each
run, and no settings changes**. Measured temperatures are **30–32 C /32–34 C**;
therefore temperature was not identical even though settings were unchanged.

Both replay model **2500**, run **2026-10-21_11-18-29**, actual Isaac400 DR env 0
episode 1 commands. Source CSV: `2500_isaac400_env0_episode1.csv`, SHA256
`c799e80fbc87af307871e35dce7ad409f3e636ff662e99ed498c3800a4cf3ad6`.
Core source interval is **12.50–17.50 s**. The full eight-second plan contains
1 s preparation ramp, 1 s hold, core **2–7 s**, and 1 s final hold.

The two plan files are **byte-identical**, match an independently regenerated
in-memory plan from the current runtime/source, and all 400 target payloads
match between rates. At 200 Hz each is repeated four times, with no new target
or second position-processing pass. Runtime SHA256
`d14b274ef49b77467fd57aecb739d0523e933b63cba3497ec22447942e91e40c` and transport
SHA256 `e1ca97c2b86ebf126ed975b474fefadac55bf2454f0e8364cd63ed0845822754`
match the local fixed scripts exactly.

## Position tracking and the repaired zero

Core metrics use **495 fresh A3 query replies per run**, not denser F1 replies
or repeated cached motor states. Reference is a zero-order hold on actual
host send-call timestamps: no future command is used. Positive steering is
the existing rightward policy coordinate, obtained with encoder sign −1.
Feedback is decoded from signed multi-turn A3 counts and rebased to verified
post-B1 zero. No extra angle wrap, time shift or constant correction is applied.

| Core measurement, 2–7 s | 50 Hz packets | 200 Hz packets |
| --- | ---: | ---: |
| RMS error | 0.701 degrees | 0.788 degrees |
| Mean signed error | +0.020 degrees | −0.131 degrees |
| Mean absolute error | 0.543 degrees | 0.630 degrees |
| 95th-percentile absolute error | 1.440 degrees | 1.528 degrees |
| Maximum absolute error | 2.538 degrees | 2.609 degrees |
| Sampled speed range | −1.264 to +1.373 rad/s | −1.686 to +1.406 rad/s |
| Sampled current-derived torque range | −0.347 to +0.343 Nm | −0.488 to +0.369 Nm |

Encoded core targets span **1.225360–1.325429 rad**, approximately **70.21–75.94
degrees**. This is a modest varying signal around a large steering offset,
not full-range rapid reversals. The preparation ramp separately moves from
straight toward that offset. It is not an experiment requiring a motor to
settle completely at every new 20 ms target.

The last-half-second pre-core hold errors are **−0.0148/−0.4991 degrees**; final
hold errors are **+0.3646/+0.0530 degrees**. These direction/history-dependent
residuals show that held position is not guaranteed exact under proportional
position control and real static losses. Two holds cannot identify a new
friction coefficient or demonstrate a systematic packet-rate advantage.

For a fair comparison with the old, aborted 200 Hz recording, use the same
common **2.00–5.36 s** window in all before/after traces:

| Rate | Old RMS /mean error | New RMS /mean error |
| --- | --- | --- |
| 50 Hz | 9.962 /+9.932 degrees | 0.709 /−0.012 degrees |
| 200 Hz | 12.119 /+12.094 degrees | 0.765 /−0.072 degrees |

The raw reference validation plus disappearance of the large offset supports
the initialization diagnosis. It is not proof of a comparably large improvement
in motor dynamics: the old de-biased shape errors were already much smaller.
The [before/after graph](before_after_zero_fix.png) never subtracts those old biases.

Interpolating asynchronous F1 only as a coordinate cross-check gives A3-minus-F1
core mean **−0.000184/−0.000142 rad**, with RMS **0.001074/0.000311 rad**.
F1 arrives at the packet rate while A3 is queried around 100 Hz, so the smaller
cross-channel discrepancy at 200 Hz is not an independent improvement in motor
accuracy. No motion F1 position code is at either encoding endpoint.

## Timing, gains and measured torque

| Recorded timing | 50 Hz packets | 200 Hz packets |
| --- | ---: | ---: |
| Achieved host send rate | 49.9994 Hz | 199.9993 Hz |
| Median packet interval | 19.9996 ms | 4.9992 ms |
| Maximum packet interval | 22.1050 ms | 6.2104 ms |
| Maximum successful-send lateness | 2.2537 ms | 1.3303 ms |
| Maximum kernel RX queue age | 2.4130 ms | 2.3687 ms |
| A3 feedback rate | 98.8928 Hz | 99.0947 Hz |
| Completed motion queries, all received | 2,430 | 2,435 |
| Best descriptive alignment shift | 16.0 ms | 19.5 ms |

The existing 3 ms send-deadline guard did not abort either run. This demonstrates
successful 200 Hz host transmission on this occasion, rather than a universal
timing guarantee or proof that the earlier scheduling problem cannot recur.
AE fault codes are zero, active mode is 5, and motion F1 status is 1. Recorded
supply remains **20.19–20.21 V** in both runs; no large sampled sag is present.
Successful-send records and received-query counts do not independently establish
zero wire loss or exact motor acquisition age.

Lag scanning uses actual held targets, fixed fresh A3 samples and **0–100 ms**
shifts at 0.1 ms spacing. Mean error is removed only for this explicitly labeled
shape-alignment diagnostic. The within-1%-of-minimum-loss bands are **15.5–18.4
ms /16.2–21.8 ms**, which overlap; they are not statistical confidence intervals.
All physical-response plots and ordinary error tables remain unshifted.

Requested MIT gains are Kp **10**, Kd **0.2**, velocity and feedforward **0**.
With the recorded F0 ranges **±3 rad /±45 rad/s /±18 Nm**, payload decoding gives:

- Kp **10.012210 Nm/rad**, Kd **0.2002442 Nm s/rad**.
- Native target velocity **+0.010989 rad/s**, feedforward **+0.0043956 Nm**,
  due to quantization of zero in symmetric 12-bit fields; both rates use the
  same bytes. These small effective inputs may contribute a constant torque
  bias; they are not an intentional change between rates.
- Position quantization error relative to request is at most **0.00004578 rad**,
  about **0.00262 degrees**, much smaller than measured tracking error.

Reported torque is **signed A1 Q current × measured B0 Kt=0.77999997 Nm/A**,
not independently measured output-shaft torque. Core RMS current-derived
torque is **0.08860/0.10802 Nm**. Different feedback states can produce different
PD torque even with identical position payloads. B3 readback is **15 A** in both
runs, but the protocol does not establish that B3 caps total MIT torque/current.
The sampled small currents and healthy status do not validate full-range MIT
protection or unobserved sub-query torque peaks.

## Comparison with saved simulations and next tests

The saved nominal fixed-root, airborne replays receive the same complete
400-sample target plan. Their five-second core RMS position errors are:

| Replay | RMS error |
| --- | ---: |
| Isaac400 | 0.5722 degrees |
| MuJoCo400 | 0.5089 degrees |
| MuJoCo800 | 0.5815 degrees |
| New hardware, 50 Hz packets | 0.7011 degrees |
| New hardware, 200 Hz packets | 0.7878 degrees |

Simulation errors use every post-physics state against that interval's target;
hardware uses approximately 100 Hz A3 replies against actual sent targets.
This sampling difference, fixed-root versus real support, nominal versus
physical parameters and unmodeled current dynamics limit precision of the
comparison. No simulation was rerun or retuned. The graph uses raw traces from
the existing [identical-target experiment](../../../../isaaclab_train/reports/20261025_matched_target_replay/README.md).

**50 Hz packets are sufficient for tracking this particular tested sequence
at the measured accuracy.** There is no evidence here that 200 Hz is needed,
nor evidence that 800 Hz host transmission is necessary. This does not establish
tracking authority for faster MuJoCo sequences, high-load tire-ground steering,
simultaneous rear actuation, full torque/speed limits or successful balancing.

Useful next steps:

1. Repeat both rates in alternating order to measure repeatability; keep the
   physical straight reference, support, supply, settings and temperature range
   comparable. Preserve partial runs if timing errors recur.
2. Replay the faster MuJoCo-recorded sequence, first airborne, with identical
   held targets and the same gains; compare against its saved simulated response.
   Inspect the preview and expected range before execution.
3. Validate a separately supported, ground-loaded steering condition only after
   the airborne comparison. Do not infer loaded balance performance from this test.

## Artifacts, validation and reproduction

- Five figure sets, each PNG/SVG/PDF: [tracking](steering_tracking.png),
  [error/lag](tracking_error_and_delay.png), [zero/timing](initialization_and_timing.png),
  [simulators](simulator_comparison.png), [before/after](before_after_zero_fix.png).
- Legends are lower-right; tracking axes show observed extrema and useful zoom.
  Only start and the preparation/core/hold boundaries exist; no episode respawn
  occurred in these standalone runs.
- Two `*_aligned.csv` files expose fresh A3, requested/encoded targets, and raw
  errors. [statistics.json](statistics.json) includes exact values and raw-file
  hashes; [summary.csv](summary.csv) is suitable for a presentation table.
- The analyzer independently regenerates each plan and every transmitted
  payload from the unchanged runtime, checks all motion A3/A2/A1/F1 signs/units,
  B1 count/verification, mode/fault status, and equality of both rate inputs.
- [verification.json](verification.json) records the independent numerical
  cross-check, source/old-report/raw-data preservation and artifact checks.

From the repository root:

```bash
python3 -B deploy/experiments/reports/20261027_steering_replay_after_zero_fix/analyze.py
```

The analyzer imports the previous report's numerical reader/helpers and the
current runtime's pure plan/payload functions; neither main hardware loop runs.
Raw recordings, production scripts, old reports, simulator captures and both
progress ledgers remain unchanged. This report is newly generated evidence;
recording it in the progress ledgers is a separate task.
