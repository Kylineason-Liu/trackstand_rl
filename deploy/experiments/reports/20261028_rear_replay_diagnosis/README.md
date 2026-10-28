# Rear-wheel replay discrepancy: script audit and diagnosis

Analysis date: 2026-10-28. [Figures](index.html), [exact statistics](statistics.json),
[summary CSV](summary.csv), [full response](rear_response.png),
[first torque burst](first_200ms.png).

**Finding:** no command-encoding, reference-subtraction or replay-completion error
was found that explains these recordings. The measured motion differs genuinely.
The strongest lead is near-zero, state-dependent motor/drivetrain resistance,
combined with different physical initial angles. Firmware/current dynamics remain
possible. Two trials cannot distinguish these causes conclusively.

## 1. Script audit and 20-ms tolerance

The live [drive_policy_replay.py](../../drive_policy_replay.py) already defaults to
**20 ms send lateness**, via steering_policy_replay.DEFAULT_MAX_LATENESS_MS.
Its query timeout is independently 20 ms; feedback freshness is 100 ms. These
are different deadlines. A cold preview confirms the default and explicit flag.
No new runtime edit was necessary for the requested tolerance.

Audited source and shared helper flow:

1. `startup`, lines 234–269: confirm disabled/fault-free, read B0/B3/B4/CD,
   validate the whole current plan, verify temporary watchdog, obtain a fresh A3
   multi-turn count, then query A3/A2/A1/AE and check stationary startup.
2. Lines 261–262 record `motor.zero_count` from signed A3 bytes 3–6. The shared
   A3 decoder calculates `-(count - zero_count) * 2*pi/16384`. The unwrapped
   channel was used for all reported displacement. A separate wrapped display
   channel exists but is not used here.
3. `current_payload`, lines 125–136: `native_current = -requested_torque / B0`,
   rounded to signed int32 little-endian milliamps, command C0 to CAN ID 0x101.
   No angle, velocity feedback, extra scale, compensation or damping enters C0.
4. Motion sends all references in order, once at 50 Hz or four identical copies
   at 200 Hz. Protection rejects/aborts rather than silently clipping this plan.
   Cleanup confirms disable and watchdog restoration.

**B1 is unnecessary for this C0 test.** Software subtraction already gives a
valid displacement zero. Changing the stored encoder origin would not physically
return the wheel to the same angle or correct torque tracking. The previous
steering defect involved an absolute MIT F1 position range; rear C0 does not use
F1, an absolute position setpoint or its encoding range.

**The experiment did have a missing control:** physical wheel angle was not
standardized between runs. The earlier procedure should have required a marked
wheel starting angle for a packet-rate comparison. Re-zeroing coordinates cannot
replace this requirement.

The live relaxed scheduler waits a full packet period after each send/log call
returns. Therefore actual replay can stretch differently at 50/200 Hz. This is
another limitation for future rate comparisons; inspect timestamps or reproduce
them in simulation. It **did not cause these old runs**, whose metadata and
archived runtime confirm the earlier fixed schedule with a 3-ms trip.

Fresh verification: **56 replay regression tests passed**, including multi-turn
feedback, sign/units, absence of host damping, repetition, delayed sends and
cleanup. [Saved test output](replay_tests.log). This is software verification,
not proof of undocumented firmware behavior or absence of every possible bug.

## 2. Independent decoding of the two original rear runs

Inputs in experiments/logs, each CSV/JSON/plan:

- `20261026_163432_217938_scaled_50hz_drive_policy_replay`
- `20261026_163545_994664_scaled_200hz_drive_policy_replay`

The user previously confirmed airborne/free wheel, no contact/restraint and no
configuration changes. The analyzer independently decodes raw C0/A1/A2/A3/AE,
checks every transmitted byte against the original source, verifies archived
runtime hashes, and recomputes statistics. There is no policy inference here.

| Quantity | 50 Hz | 200 Hz |
| --- | ---: | ---: |
| Completed C0 packets | 350/350 | 1,400/1,400 |
| A1 torque-tracking RMS | 0.015848 Nm | 0.016496 Nm |
| Signed command integral over replay | −0.499017 Nm s | −0.498972 Nm s |
| Final motion-phase A3 displacement | −5.282263 rad | −0.334791 rad |
| Integral of fresh A2 speed | −5.203803 rad | −0.330827 rad |
| Most negative A2 speed | −8.048760 rad/s | −2.832669 rad/s |
| Initial A3 count | 324,554 | 338,328 |
| Native initial angle modulo one turn | 291.3135° | 233.9648° |
| First displacement exceeding 0.002 rad after core starts | 23.6 ms | 1,470.1 ms |

The movement threshold is a descriptive onset criterion, **not measured motor
latency**. Encoder and velocity channels share firmware, so their agreement is a
cross-check rather than independent metrology. Nevertheless the approximately
15.78-fold displacement difference is not explained by wrapping or a missing
encoder zero: speed integration independently reproduces its scale.

Further evidence:

- The first run's last A3 count is **338,328**, exactly the second run's initial
  count. Their multi-turn difference is 302.65°, equivalent to **−57.35° modulo
  one turn**. This is a physical starting-phase difference, not a reporting bias.
- Both two-second zero-command preparations remain within **one encoder count
  (0.0003835 rad)**, returning to zero before the core. Residual initial rolling
  does not explain the observed difference.
- All corresponding 50-Hz reference payloads are identical. Command integrals
  differ by only **0.00895%**; maximum corresponding target-onset discrepancy is
  **2.025 ms**. This is not an exact match of continuous waveforms, but no missing
  targets, fourfold torque scaling, long interruption or early rear abort exists.
- During actual time **2.00–3.44 s**, the 200-Hz run stays within **0.000767 rad**
  of zero despite A1 current-derived torque reaching **−0.206100 Nm**. The 50-Hz
  wheel has already moved about **−0.344 rad** under the same target sequence.
- Voltage ranges are **20.17–20.21 /20.19–20.21 V**, temperature **30–32°C** both.
  Same firmware Kt=0.3000000119 Nm/A, B4 raw=0, B3=82.498 A; B3 is a speed/position
  limit, not documented C0 protection. B4 zero's firmware semantics are unknown.
  Active AE mode is 2/fault 0; every logged completed query is received; both
  runs confirm disable/watchdog restoration. This does not prove zero wire loss
  or exclude supply/current transients between samples.

### C0 reply sampling can create a misleading rate comparison

Comparing all C0 replies gives RMS errors **0.043257/0.023888 Nm**, apparently
favoring 200 Hz. But taking only the **first reply after each new 50-Hz target**
gives **0.043257/0.043573 Nm**. The other three repeated-copy groups at 200 Hz give
0.011336/0.010160/0.012338 Nm. Thus its aggregate mixes more replies after the
target has been held. The comparable independent A1 streams give similar errors.
Do not equate an immediate C0 response with the settled current for its new
command, or infer improved torque control from these unequal sampling groups.

## 3. Ranked explanations

1. **State-dependent breakaway/resistance and different starting angle — strongest
   lead, not an identified mechanism.** The response consists of stationary
   intervals and motion bursts; substantial current is reported while nearly
   stationary. The existing 0.106-Nm fit used steady C1 motion at 1–8 rad/s and
   explicitly excluded static breakaway. Static friction, gear/bearing resistance,
   magnetic cogging or wheel imbalance can change the torque needed to start and
   sustain motion. Near a threshold, small differences can select sticking versus
   moving, then accumulate into a large displacement. The dataset cannot separate
   those mechanisms or justify fitting a universal 0.206-Nm static coefficient.
2. **Current/commutation dynamics, history and packet-refresh effects — plausible.**
   A1 is ~100 Hz and C0 replies are firmware current reports, not continuous output
   shaft torque. Inter-sample ripple, current-to-shaft conversion error, internal
   controller state and repeat-command handling are unmeasured. Similar A1 RMS
   does not guarantee identical mechanical torque. Fixed-angle replicated runs
   are needed before attributing the difference to packet rate.
3. **Gross host timing, supply sag, thermal change or faults — less supported.**
   Both runs complete with nearly identical input timing/integral and similar
   sampled supply/temperature. Small timing/current differences could still be
   amplified by a threshold; a categorical exclusion would overstate the data.

The balance equation is `J*domega/dt = motor_torque - losses - external_load`.
Open-loop torque replay has no speed/position correction. It is sensitive to
unmodeled losses and initial conditions; this is not proof of intrinsic random
torque behavior. Grounded balancing introduces additional contact dynamics not
measured in these airborne trials.

## 4. Applicable remedies and their limits

### Identify the missing dynamics, then use targeted DR

Current v1 drive config sets **static friction=dynamic friction=0.106 Nm,
viscous friction=0, IdealPD Kp=Kd=0**. Existing DR covers mass, CoM, contact
material, initial root state, IMU residuals and gains; it does not randomize rear
joint breakaway, motor torque conversion, cogging or actuator delay. Multiplying
zero rear gains by random factors does not randomize feedforward motor strength.
V2 also retains these passive losses and disables controller-gain DR.

After measurement, add per-environment/episode ranges for motor joint static and
moving losses separately, torque gain/offset and delay; add angle-dependent
resistance/phase or hysteresis only if demonstrated. Use history/adaptation if
unobserved actuator state matters. This requires a new policy. DR can improve
closed-loop robustness; it cannot make identical open-loop torques guarantee
identical displacement or cure faulty hardware.

Relevant precedent: [Hu et al., Static Friction in Sim2Real](https://arxiv.org/html/2503.01255)
report improved Saturn Lite transfer with joint-friction-aware DR after ordinary
DR was inadequate. Their policy is position-action RL and their simplified model
equates static and moving friction; it is evidence to include relevant joint
losses, not a measured breakaway law for our motor. [TOCABI](https://arxiv.org/html/2304.09434)
randomizes motor constant, joint friction/damping and delay for torque-action RL.

The installed Isaac Lab `envs/mdp/events.py` has randomize_joint_parameters, but
its single friction range is applied to static/dynamic/viscous channels; separate
measured ranges require appropriate custom handling. No DR edits were made here.

### Calibrated compensation or a learned actuator model

If repeatable angle-dependent resistance is established, identify a phase map
and bounded feedforward correction. If a near-zero dead zone is established,
identify direction/history-dependent response rather than simply adding a
constant at every sign change. Maintain the same compensated behavior in training
and deployment: the v1 policy already sees 0.106-Nm simulated drag, so adding
0.106 Nm indiscriminately changes its motor contract.

[ODrive's official anticogging guide](https://docs.odriverobotics.com/v/latest/guides/anticogging.html)
uses a measured position-dependent torque map and requires a repeatable absolute
reference. It also warns that feedback delay/inaccuracy can reduce stability.
This illustrates a method, not an available SteadyWin feature or proof of cogging
here. If applying a phase map, preserve the native phase separately from the
episode-relative policy angle; repeatedly setting an arbitrary B1 origin would
destroy that alignment without retained offsets.

[PACE](https://arxiv.org/html/2509.06342) identifies and validates actuator/current
and mechanical behavior before full-robot RL deployment. [ANYmal](https://arxiv.org/html/1901.08652)
instead learns actuator response from hardware histories. Either motivates
fitting a validated response model rather than guessing a larger friction value;
two seven-second replays are insufficient training/validation data for a general
learned motor model. Current-derived torque is not net joint torque ground truth.

### Feedback, disturbance observers and MPC

A wheel-speed/position feedback controller can adjust torque when motion differs
from the desired motion. MIT Kp=0/Kd=0.2 is only proportional speed feedback; it
does not guarantee exact velocity under static/moving losses. PI with suitable
anti-windup or an identified disturbance observer can address sustained error,
subject to bandwidth, noise and available torque. The known v2 numerical issue
must still be resolved before comparing trained policies.

Full balancing MPC is another architecture, not a drop-in correction to raw
torque without a motion objective. [Wang's rear-driven trackstand](https://arxiv.org/html/2410.19615)
demonstrates MPC plus disturbance/equilibrium estimation on Pi4B, with rear torque
and 100/200-Hz optimization/feedback. Its theory assumes bounded slow-varying
disturbances; fast cogging or stick-slip is not automatically covered. Pi execution
time and state estimation must be validated for our model. An observer or
compensator would also need identified inertia/losses and filtered velocity
information, not noisy direct differentiation treated as exact acceleration.

## 5. Most informative next hardware experiment

1. Retain the same source, support, settings and torque mode. Mark the wheel's
   physical starting angle and record native A3; restore the mark **while disabled**
   before every trial, then let the wheel rest. Do not use B1 to imitate physical
   restoration. The script records a new reporting zero automatically.
2. At that angle, collect at least **three runs per packet rate**, alternating
   order. Repeat at a second marked angle: **12 runs total**. Keep all three files
   per run. Compare actual timestamps as well as payloads; the relaxed scheduler
   can stretch rate groups differently.
3. Angle effect at a fixed rate supports position-dependent resistance; rate
   effect with matched initial conditions supports refresh/current/timing effects.
   Large variation within the same condition calls for more data on controller
   history, temperature or unresolved transients. These are discriminating tests,
   not proofs of one mechanism by themselves.
4. Independently measure signed breakaway by increasing small current commands
   from rest at several angles, ending each trial when motion begins; measure
   subsequent moving drag separately. Existing replay is not a threshold-sweep
   implementation. Use those results to select calibration/DR/compensation, not
   guessed numbers from the present two trials.

Current replay commands, from deploy/experiments (wheel airborne/free, frame
supported, steering independently disabled, other motor writers stopped):

```bash
python3 drive_policy_replay.py --execute --packet-rate 50 --max-lateness-ms 20 --note "angle A repeat 1"
# After disable: restore the same physical wheel mark and allow rest.
python3 drive_policy_replay.py --execute --packet-rate 200 --max-lateness-ms 20 --note "angle A repeat 1"
```

## Reproduction and scope

```bash
python3 -B deploy/experiments/reports/20261028_rear_replay_diagnosis/analyze.py
python3 -B -m unittest discover -s deploy/experiments/tests -p 'test_*replay.py' -v
```

Created only this report directory: analyzer, two PNG/SVG/PDF plot sets, HTML,
statistics, CSV and audit logs. No production/runtime, original data/report,
progress-ledger, controller or firmware changes were made; no hardware or
simulation was operated. Existing 20-ms tolerance is verified rather than
rewritten. The root cause remains conditional on the additional experiments.
