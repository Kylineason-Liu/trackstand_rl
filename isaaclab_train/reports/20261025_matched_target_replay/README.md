# Identical-target replay: model 2500 command sequences

Experiment started 2026-10-25; report completed 2026-10-26.
Open the [figure gallery](index.html).

**Result:** when given the same rapidly alternating targets, Isaac also has
large tracking error, similar to MuJoCo. Disabling passive steering friction
does not recover tracking of those fast signals. There is nevertheless a
separate, experimentally visible difference in static-friction behavior.

## Controlled experiment

- One [immutable float32 input file](inputs.npz), with [source hashes and protocol](inputs.json).
- New targets every **20 ms**, held unchanged between updates. No interpolation,
  fresh policy inference, observation noise, DR resampling, or automatic resets.
- Steering inputs are already in **rad**, after the recorded scale/clip/slew.
  Rear inputs are recorded requested **Nm**. No second scaling or slew pass.
- Same nominal v1 configuration: steering Kp=10, Kd=0.2, torque cap 5.6 Nm;
  rear Kp=Kd=0, cap 6.28 Nm. Steering passive dry friction 0.046036056104 Nm,
  B=0.000910713492 Nm s/rad; rear friction 0.106 Nm and B=0.
- **Both engines at 400 and 800 Hz physics/PD**, with target updates always 50 Hz.
  Isaac 800 is a timing diagnostic, not the model's original training rate.
- Current verified USD/MJCF geometry, inertias and axes were preserved.
  Source mass values match to floating-point precision. Nominal initial world
  CoMs agree closely; the existing rear CoM differs by about 0.066 mm between
  assets. Neither asset was retuned to force identical trajectories.

This uses DR-recorded **commands** with matched nominal plants. It is an actuator
response diagnostic, **not** another DR balancing evaluation or a hardware test.
It removes independent closed-loop target generation and differing DR draws.

## Supported motor test

The bicycle base is fixed in place with both wheels airborne, gravity retained.
MuJoCo reports zero contacts throughout. In Isaac the fixed root is verified,
self-collision remains disabled, and the bike is 2 m above the plane, with other
fixtures separated by 2 m. Only the support boundary differs from the normal
floating-bike task; no motor coordinate is imposed by the fixture.

Five inputs are each tested with passive steering friction ON and OFF:

1. 1 Hz sine, amplitude 0.1 rad, sampled/held at 50 Hz.
2. Alternating +0.1/−0.1 rad targets every 20 ms: a 25 Hz square wave.
3. Actual Isaac 400 DR env 0 targets and rear requests from 12.50–17.50 s.
4. Actual MuJoCo 400 DR env 0 targets and rear requests from 12.50–17.50 s.
5. Actual MuJoCo 800 DR env 0 targets and rear requests from 12.50–17.50 s.

Each eight-second replay has a 1 s preparation ramp and 1 s hold, the unchanged
five-second sequence at 2–7 s, then a 1 s hold of its final target. OFF removes
**only steering static/dynamic/viscous losses**; controller Kd remains 0.2 and
rear losses remain 0.106 Nm/B=0. All cases begin with zero joint positions/speeds.
Initial floating poses/velocities are explicitly restored after Isaac startup.

### Tracking under identical inputs

Five-second core RMS position error, degrees, steering friction ON:

| Shared input sequence | Isaac 400 | MuJoCo 400 | MuJoCo 800 | Isaac 800 |
| --- | ---: | ---: | ---: | ---: |
| 1 Hz sine, 0.1 rad | 0.6862 | 0.5774 | 0.7116 | 0.7181 |
| 25 Hz alternating, 0.1 rad | 5.1429 | 4.8329 | 5.3504 | 5.5274 |
| Recorded Isaac DR sequence | 0.5722 | 0.5089 | 0.5815 | 0.6001 |
| Recorded MuJoCo 400 DR sequence | 3.3106 | 3.0938 | 3.4288 | 3.5507 |
| Recorded MuJoCo 800 DR sequence | 3.6178 | 3.3841 | 3.7495 | 3.8832 |

The same fast MuJoCo-recorded targets produce much larger errors in **both**
engines. The original 0.57-degree versus 3.1–3.8-degree independent-run difference
therefore cannot be treated as a motor-response comparison under the same input.
Here the broad difference follows the input sequence, while residual engine and
time-step differences remain. This does not identify why the original MuJoCo
closed loop generated rapid target alternation in the first place.
Similar position error does not establish identical torque/velocity ripple or
physical fidelity; the separate torque traces retain those differences. This
experiment does not resolve the previously observed balance sensitivity to
400 versus 800 Hz.

Every supported core has **zero steering torque-cap hits**, and maximum native
steering speed remains below 23.9 rad/s (largest 18.6068 rad/s). The observed fast
tracking error persists without active torque/speed saturation in this fixture.

### What friction removal changes

For both recorded fast MuJoCo sequences, turning steering losses OFF changes
core RMS error by **less than 0.7%** in every engine/rate. For example:

| MuJoCo 800-recorded input | Friction ON RMS | OFF RMS |
| --- | ---: | ---: |
| Isaac 400 | 3.6178 degrees | 3.6008 degrees |
| MuJoCo 400 | 3.3841 degrees | 3.3838 degrees |
| MuJoCo 800 | 3.7495 degrees | 3.7399 degrees |

This controlled ablation argues against passive steering losses being the
dominant direct cause of the large fast-input tracking error. It does not prove
friction is irrelevant to the closed-loop policy or balance stability. In the
slow 1 Hz test, removing losses reduces RMS error by roughly 19–33%.

### A genuine remaining static-friction difference

After holding the final position, the average last-half-second error is
approximately **0.0046 rad (0.264 degrees) in Isaac**, with friction ON. This is
the expected scale 0.046036/10 for static resistance balanced by proportional
controller torque. MuJoCo's error converges below 3e−8 rad in the same holds,
despite its nonzero configured frictionloss. Friction OFF makes both engines
converge close to target; exact values are in [statistics.json](statistics.json).

Thus identical numeric friction parameters do **not** yield identical behavior
near rest. The native MuJoCo bounded/regularized friction constraint allows
creep here, while PhysX stops within a static deadband. This test establishes
the response difference; it does not certify either engine's static threshold
against hardware. Our static threshold was provisional in the original fit.

The 1 Hz fitted response gain with friction ON is 0.9769–0.9846; phase-equivalent
lag relative to the actual held reference is 22.32–27.77 ms across the four runs.
The 25 Hz square wave's **fundamental** gain is only 0.298–0.311. These are frequency-
dependent fit results, not motor firmware rates or universal transport delays.
The fits use 3–7 s, with no response shifting. Full stored samples are unsmoothed.

## Grounded replay and limits of interpretation

Two additional normal floating-bike cases replay the first eight seconds of
the original Isaac and MuJoCo 800 DR command streams, from nominal upright
spawn with all joint/root velocities zero. Both engines receive exactly the
same steering/rear requests. No fresh feedback policy or automatic reset runs.

All open-loop grounded cases cross 30-degree roll after 0.285–0.3725 s. The plot
marks each crossing; subsequent traces describe a falling bike. Grounded RMS
comparison uses only the common **0–0.25 s** pre-crossing interval. These crossing
times are **not** survival evaluations: commands copied from a different DR
trajectory cannot be expected to stabilize a new nominal trajectory. They show
that full contact/base dynamics still differ; the supported test isolates motor
tracking more cleanly. Do not claim backend equivalence or solved sim2real.

## Files and reproduction

- [Five-second matched positions](matched_position_core.png).
- [Half-second matched position detail](matched_position_detail.png).
- [Full preparation/core/hold](matched_position_full.png).
- [Final hold/static error](matched_position_final_hold.png).
- [Matched motor torque detail](matched_torque_detail.png).
- [Grounded first half-second](ground_replay_first_half_second.png).
- Each figure also has SVG/PDF. Fitted axes show all samples in the declared
  window; min/max values are labeled. Red vertical lines mark the only spawn
  at 0 s. Each case has one continuous replay and no subsequent spawns.
- `isaac{400,800}/` and `mujoco{400,800}/`: complete NPZ substep traces and
  metadata. Motor order is [steering,rear]. Commands are at interval start,
  positions at interval end. Rear submitted commands coincide; steering PD
  torque may differ because measured feedback states differ.
- [Exact statistics](statistics.json), [input preparation](prepare.py),
  [Isaac replay](replay_isaac.py), [MuJoCo replay](replay_mujoco.py),
  [analysis](analyze.py), [verification tests](test_replay.py).

From repository root, the completed inputs must be reused. Capture programs
refuse to overwrite their output directories. To capture again, create a fresh
folder under `isaaclab_train/reports/`, copy only the Python scripts,
`inputs.npz`, `inputs.json` and `protected_before.json` there, and substitute
that folder's name in the capture commands below. Keep the same relative depth;
do not copy the completed output directories into the fresh folder.

```bash
~/IsaacLab/_isaac_sim/python.sh -B -u \
  isaaclab_train/reports/20261025_matched_target_replay/replay_isaac.py \
  --physics_hz 400 --headless --device cuda:0
python3 -B -u isaaclab_train/reports/20261025_matched_target_replay/replay_mujoco.py --physics_hz 400
# Repeat both with --physics_hz 800.
python3 -B isaaclab_train/reports/20261025_matched_target_replay/analyze.py
python3 -B -m unittest discover -s isaaclab_train/reports/20261025_matched_target_replay -v
```

All output archives require `metadata.json: completed=true`; process exit 0
alone is insufficient because Isaac app shutdown can mask earlier exceptions.
Initial setup attempts are retained under clearly named incomplete/before-reset
directories/logs. They are excluded from analysis. The final root path is
explicit for the fixed-base fixture, and the final floating state is reset
after the startup warm-up. No saved geometry, task/controller source, policy,
original evaluation artifact or progress ledger was modified. There was no
training, hardware/CAN test, or deployment change.

Final verification: **10 tests passed in 0.945 s**. Checks cover exact shared
inputs, actual PD/torque reconstruction, effective friction pairs and gains,
explicit initial states, zero MuJoCo bench contacts, valid grounded comparison
windows, phase fitting, physical extrema excluding marker artists, all figure
formats and links. **243 protected preexisting files are unchanged**, including
sources, original report/captures, model assets, ledgers and Git index. Syntax,
Pyflakes, 120-column checks and `git diff --check` passed. See
[verification.json](verification.json) and [test log](unit_tests.log).
