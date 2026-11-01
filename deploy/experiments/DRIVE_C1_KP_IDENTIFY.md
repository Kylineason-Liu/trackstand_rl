# Identify C1 velocity-loop gains without changing firmware gains

`drive_c1_kp_identify.py` records the rear motor's response across speeds,
directions, perturbation sizes and frequencies. **B8/Kp and B9/Ki are strictly
read only**, including a sender check that rejects gain writes. This follows
the user's finding that changing those parameters does not work on this motor.
The registered values are logged; the estimator does not assume their units or
that they describe the effective firmware gains correctly.

The PC analyzer `analyze_drive_c1_kp.py` estimates effective P **and I together**.
It cannot isolate P by setting Ki to zero, and does not pretend integral action
disappears during a short step or a steady hold.

## Measurement principle and units

Let `e = actually_sent_velocity − measured_velocity`, in rad/s. The approximation
being tested is:

```text
user_positive_Iq(t) ≈ P_current * e(t) + I_current * integral(e dt) + trial_offset
torque_estimate(t)  = measured_B0_Kt * user_positive_Iq(t)
```

- `P_current`: **A/(rad/s)**; `I_current`: **A/rad**.
- `P_torque = B0_Kt × P_current`: **Nm/(rad/s)**, equivalent to Nm·s/rad.
- `I_torque = B0_Kt × I_current`: **Nm/rad**.

`P_torque` is the starting coefficient for a velocity-P law in simulation:
`torque = P_torque × (target_velocity − velocity)`. For an IdealPDActuator this
corresponds to `stiffness=0`, `damping=P_torque`; MIT calls it **Kd** with Kp=0.
It is **controller feedback**, not passive motor damping. A full C1 replica also
needs integral action, torque limits and any identified delay/filtering. Existing
friction/inertia parameters remain separate; this experiment changes no simulator.

The current-derived torque is electromagnetic torque estimated using the motor's
B0 constant, not an independent shaft-torque measurement. Calibration error is
not removed by repeating tests. Firmware gain units and internal loop frequency
remain unknown even if the effective SI model fits well. See the local
[CAN protocol](../../tsinghua_bike_project/can_protocol.txt) for A1/A2/A3/B0/B8/B9,
and [MathWorks' PI definition](https://www.mathworks.com/help/control/ref/pid.html)
for the distinction between P, I and the discrete integrator convention.

## Test conditions and commands on the Pi

Rear wheel fully airborne and free, frame independently supported, steering
disabled, other motor CAN writers stopped. Do not restrain the wheel by hand.
The standalone script does not respond to ROS/RC stop commands; Ctrl-C attempts
CF disable and watchdog restoration. Start airborne; a loaded comparison needs
an appropriate supported roller/dynamometer fixture, not a freely moving bike.

Copy/pull `drive_c1_kp_identify.py` and its adjacent `drive_velocity_sine.py` helper.
The Pi uses Python's standard library; no ROS build, NumPy or separate recorder.

```bash
cd ~/trackstand_rl

# Preview the full plan: no CAN, no files.
python3 deploy/experiments/drive_c1_kp_identify.py

# Short data-path check: two +/-0.25-rad/s pulses around zero, about 7.2 s.
# This is not enough data for a comprehensive Kp estimate.
python3 deploy/experiments/drive_c1_kp_identify.py --execute \
  --suite steps --repeats 1 --step-bases 0 --step-sizes 0.25

# Full experiment, about 19 min 54 s plus startup/shutdown overhead.
python3 deploy/experiments/drive_c1_kp_identify.py --execute \
  --note 'airborne/free; supply and firmware unchanged'
```

## Default coverage

Cases are reproducibly shuffled within each repetition, with seed 42, to reduce
simple ordering/temperature confounds. Physical rotor phase is recorded, not
automatically matched. All targets mean forward-positive rad/s.

| Group | Conditions | Trials | Time per trial |
| --- | --- | ---: | --- |
| Steady speed | Both signs of 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 6, 10 rad/s; 3 repeats | 54 | 2 s settling + 3 s measurement |
| Small steps | Baselines −6, −2, 0, 2, 6; steps ±0.1, ±0.25, ±0.5, ±1 rad/s; 3 repeats | 120 | 0.6 s baseline → 0.4 s pulse → 0.6 s return |
| Sine response | Baselines −3, 0, 3; amplitudes 0.25 and 1 rad/s; 0.5, 1, 2, 5, 10, 20 Hz; 3 repeats | 108 | 1 s settling + ≥4 cycles and ≥4 s measurement |
| Zero-speed context | Before every steady hold and at repeat boundaries | 60 | 2 s active C1 zero speed |

Maximum default target magnitude is **10 rad/s**. Zero speed is active velocity
control, not CF-disabled coasting, and does not guarantee a stopped wheel or a
reset integrator. There is no hidden target ramp. The first A3 multi-turn reading
defines a software displacement origin; B1 never resets the physical encoder.

Default C1 sending is **200 Hz**. A1/A2 queries are **100 Hz each** during steady
and zero holds, **400 Hz each** during steps and sines. A3 is 50 Hz, AE 10 Hz,
B8/B9 1 Hz each. Replies are asynchronous; actual host timing is recorded.

Useful flags: `--suite all|steady|steps|sine`, `--repeats`, `--seed`, `--levels`,
`--step-bases`, `--step-sizes`, `--sine-bases`, `--sine-amplitudes`,
`--frequencies`, `--packet-rate 50|200|400`, `--steady-feedback-rate`,
`--dynamic-feedback-rate`, `--interface`, `--note`, `--output`. Duration flags
are listed by `--help`. A steady-only run provides drag/tracking context; it
usually cannot identify P separately from I.

To check sending-rate dependence afterward, repeat a shorter sine suite with
`--packet-rate 50`, `200` or `400`, keeping the other settings fixed. The analyzer
uses each run's exact held input, since changing packet rate also changes the
sampled command waveform. Separate repeated runs are needed for that comparison.

## Logs and retained checks

Three timestamped files are automatically saved under `deploy/experiments/logs/`:

- `.csv`: every command/query/reply with raw bytes, sent targets, fresh A1/A2/A3,
  current-derived torque, supply/status/faults, actual timestamps and skipped slots.
- `.json`: full plan, original/applied settings, B0 Kt, source/helper hashes,
  software origin, per-trial timing/state, completion/error and cleanup status.
- `_plan.csv`: target conditions and analysis windows for all planned trials.

The full experiment can produce hundreds of MB of data. Individual trials and
repetitions remain identifiable; cached values in unrelated rows are not treated
as additional fresh measurements.

There is no angle, tracking-error, small-current, stop-speed or packet-lateness
abort. Late slots are logged/skipped without catch-up bursts or stretching the
waveform. Existing stops remain: measured speed above **61.6 rad/s**, measured
current-derived torque above **6.28 Nm**, driver/CAN/log faults, competing motor
writers, **1 s** without velocity/current or **2 s** without status. Timeout flags
remain configurable. These checks do not guarantee physical caps between samples.

B3/B5/B8/B9 and the encoder origin are never written. The shared helper temporarily
enables/shortens CD to at most 500 ms, retaining a tighter existing setting, and
restores it after CF confirms disabled. Faults are not automatically cleared.
A hard kill or broken communication can prevent cleanup; inspect the saved
completion/error and disable/restoration flags before treating a run as complete.

## Analyze on the PC

Copy the three run files back into this repository, then:

```bash
python3 deploy/experiments/analyze_drive_c1_kp.py \
  deploy/experiments/logs/<timestamp>_drive_c1_kp_identify.csv
```

NumPy and Matplotlib are required on the PC. The default new report directory is
`deploy/experiments/reports/<run_stem>_analysis/`; existing reports are not
overwritten. `--output-dir`, `--bootstrap` (default 500) and `--seed` are available.
Outputs include Markdown/HTML reports, PNG/SVG/PDF plots, exact statistics,
condition-specific estimates, harmonic measurements and steady-condition CSVs.

Two estimation paths cross-check the answer:

1. **Time domain:** fit P and I jointly to fresh A1 samples. Error uses the actual
   held C1 target and A2; integrated error uses the command integral minus A3
   displacement. A nuisance offset per trial accounts for unknown prior integral
   state. Scan −10…20 ms apparent delay; inspect P by direction, base speed,
   perturbation size, frequency and repeat.
2. **Frequency domain:** estimate target, velocity and current harmonics at each
   excitation frequency. Target harmonics integrate the exact held, quantized C1
   packets; A1/A2 use their own timestamps. Fit complex `current/error` to
   `(P + I/(jω)) × exp(−jω delay)/(1 + jω lag)`. The optional current/measurement
   lag is tested at 0/1/2.5/5/10/20 ms. This avoids equating high-frequency
   attenuation directly with a smaller P.

With the default three repetitions, repeat 1 fits gains, repeat 2 selects
delay/lag, and repeat 3 is untouched test data. Final estimates then refit all
trials; uncertainty bootstraps **whole trials**, not correlated individual CAN
samples. One-repeat runs split complete trials instead; two-repeat runs have no
untouched third repeat. Confidence intervals hold the selected delay/lag fixed
and do not cover systematic sensor/firmware/model error.

Partial segments, missing interpolation brackets beyond 100 ms, weak harmonic
excitation, and possible clipping are screened **during analysis only**. A trial
with >1% of A1 samples within 2% of the smaller documented B3/current-derived
6.28-Nm ceiling is excluded as possibly clipped. No such screen changes commands.
Harmonic error must exceed 0.001 rad/s and three approximate coefficient standard
errors; correlated noise can still make that confidence optimistic.

The provisional simulation coefficient is withheld when methods disagree by
>30%, frequency validation/test relative error exceeds 35%, P is nonpositive or
its conditional interval crosses zero, regressors are poorly separated, or the
delay/lag optimum touches the search boundary. These are explicit evidence
heuristics, not guarantees of accuracy or motor abort criteria. Even a passing
initial estimate should be checked with a separate target replay before training
or deployment. No result is claimed until real measurements are collected.
Each frequency-domain training/validation/test partition must contain at least
three usable excitation frequencies before an initial coefficient is suggested.

Nonzero B5 acceleration settings or changing reported B8/B9 values also prevent
an automatic initial-coefficient suggestion. Internal target ramping must be
resolved before interpreting an input/output fit as controller P. A zero B5
readback does not by itself establish all firmware target/filter semantics.

## Offline validation

The new test suite has **14 passing tests**, plus **47 passing existing velocity
experiment tests**. Checks include blocking gain/settings writes, sign and C1
units, query/sending schedules, delayed slots, fault/interrupt/logger cleanup,
three-file recording, weak-excitation rejection, independent test repetitions,
exact held-target harmonics, and full CSV-to-report generation.

A **synthetic mathematical PI fixture**, with true torque-P **0.300 Nm/(rad/s)**,
was recovered as **0.3024** by the time-domain method and **0.3029** by the
frequency-domain method. This tests the estimator under its modeled assumptions;
these numbers are **not measurements or recommended settings for the real bike**.

```bash
MPLCONFIGDIR=/tmp/c1_kp_identification_mpl python3 -m unittest discover \
  -s deploy/experiments/tests -p 'test_drive_c1_kp_identify.py' -v
python3 -m unittest discover -s deploy/experiments/tests -p 'test_drive_velocity_*.py' -v
```
