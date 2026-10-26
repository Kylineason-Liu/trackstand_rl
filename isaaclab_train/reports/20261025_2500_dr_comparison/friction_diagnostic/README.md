# Model 2500: why the Isaac and MuJoCo demand/response plots differ

Analysis date: 2026-10-25. Original plots/data remain unchanged. Runtime sources,
verified geometry/inertias/axes, checkpoint, and progress ledgers were not edited.

## Findings

**The rear-torque difference primarily comes from plotting different quantities.
The steering difference is real, but these runs also produce very different
steering target sequences. The evidence does not establish incorrect numeric
joint-friction settings in MuJoCo.** Its friction solver is nevertheless not
identical to PhysX, particularly around rest and reversals.

### 1. The two green rear curves are different measurements

Original plotting code: [`../analyze.py`](../analyze.py), `response_field()` and
`panel_series()`. Isaac's recorder reads
`robot.root_view.get_dof_projected_joint_forces()` after each physics step.
MuJoCo's recorder reads `data.actuator_force`. Isaac therefore shows a projected
incoming joint-axis effort, while MuJoCo shows isolated actuator output before
joint losses. Neither is automatically total torque accelerating a grounded wheel.

The rear IdealPD controller has Kp=Kd=0 in both implementations. Submitted torque
follows the scaled action. MuJoCo's ideal motor with gear=1 outputs that same
torque, while separate joint forces can resist or assist its instantaneous sign.
It is incorrect to expect `actuator_force` itself to subtract joint friction.

In the original Isaac capture, among 158,780 ordinary moving/motoring samples
(both start/end speed magnitudes >0.1 rad/s, unchanged sign, command aligned with
speed and larger than 0.106 Nm), median `|command|-|solver effort|` is
**0.1059938222 Nm**. This is not universal: on all 494,061 ordinary moving samples,
`solver effort ≈ command - 0.106 sign(speed)` is within 0.0001 Nm only **75.6652%**
of the time. Sticking, reversals, solver impulses and constraints matter.
Friction opposes motion, not necessarily the commanded torque. During braking,
it can increase the magnitude of the combined resisting effort.

### 2. MuJoCo really applies the configured losses

`newbike_mujoco/contract.py` defines the numbers; `env.py:120` writes them into
`model.dof_damping` and `model.dof_frictionloss`, ordered [steering, rear].

| Parameter | Isaac v1 | MuJoCo runtime |
| --- | ---: | ---: |
| Steering moving dry friction | 0.046036056104 Nm | frictionloss 0.046036056104 Nm |
| Steering passive viscous coefficient | 0.000910713492 Nm s/rad | dof_damping 0.000910713492 |
| Rear moving dry friction | 0.106 Nm | frictionloss 0.106 Nm |
| Rear passive viscous coefficient | zero model baseline | dof_damping 0 |
| Steering controller Kp/Kd | 10 / 0.2, randomized independently | 10 / 0.2, randomized independently |
| Rear controller Kp/Kd | 0 / 0 | 0 / 0 |

Static friction is provisionally set equal to moving friction in both plants;
breakaway was not identified by the original speed sweeps. The saved Isaac
training YAML leaves rear viscous friction null, and the original capture retained
that setting with its documented runtime override. This is not evidence that
physical rear damping is mathematically exactly zero.

I repeated **both** production MuJoCo DR evaluations with read-only force logging:
64 environments, 3000 policy steps, seed 42, same model_2500. All **13 original
production archive arrays are byte-identical** to their respective original
400/800 Hz captures. Thus the extra measurements describe the plotted trajectories.
Each force capture contains 1,536,000 / 3,072,000 physics samples per motor.

The logger isolates joint dry-friction constraint rows, then computes their
generalized force using `mj_mulJacTVec`. It separately records `qfrc_passive`,
`qfrc_actuator`, and the complete `qfrc_constraint`. This avoids incorrectly
treating contact forces as joint friction or expecting dry friction inside
`qfrc_passive`. Every sample verifies passive viscous torque = `-B * speed_before`;
maximum error is **zero** in both repeats. Friction is enabled in all 64 models,
and isolated rear dry-friction force reaches **both -0.106 and +0.106 Nm**.

Concrete measured examples, with positive rear speed and opposing dry friction:

| Run / env / command time | Motor request | Actual actuator output | Dry friction | Motor + joint losses |
| --- | ---: | ---: | ---: | ---: |
| 400 Hz /5 /25.09250 s | +0.2410512865 | +0.2410512865 | -0.106 | +0.1350512865 |
| 800 Hz /24 /34.11375 s | +0.1113765836 | +0.1113765836 | -0.106 | +0.0053765836 |

All torque columns are Nm. The last column excludes contacts, inertia and bias;
it is **not** an exact equivalent of Isaac's projected joint-force sensor.

**There is a friction-law approximation, not missing friction.** MuJoCo dry
friction is a bounded solver constraint, rather than an unconditional explicit
`-0.106 sign(speed)` subtraction. Its compiled default joint-friction solver
settings are solref `[0.02,1]`, solimp `[0.9,0.95,0.001,0.5,2]`. Among samples with
unchanged sign and both speeds >0.1 rad/s, median absolute rear dry force is
0.09275835 Nm at 400 Hz / 0.07487029 Nm at 800 Hz. Above 0.5 rad/s, the median is
0.106 Nm in both, with **94.9322% /96.4875%** exactly at that bound. Matching
numeric loss parameters does not certify identical stiction/reversal behavior.

### 3. MuJoCo's steering requests are faster and harder to follow

For env 0, the same displayed interval **12.50–17.50 s**:

| Measurement | Isaac 400 | MuJoCo 400 | MuJoCo 800 |
| --- | ---: | ---: | ---: |
| Held-target peak-to-peak, rad | 0.100061 | 0.259231 | 0.224583 |
| Actual steering peak-to-peak, rad | 0.075106 | 0.145148 | 0.139520 |
| Tracking RMS error, degrees | 0.5713 | 3.1209 | 3.7812 |
| 95th-percentile target step per 20 ms, rad | 0.035602 | 0.168156 | 0.172852 |
| Target variation energy in 10–25 Hz | 25.2678% | 95.1116% | 94.3300% |
| Steering torque samples at 5.6 Nm cap | 0% | 0% | 0% |

Spectrum: full discrete Fourier transform of 250 demeaned held targets at 50 Hz,
using a rectangular five-second window; both positive/negative frequencies are
counted. These are recorded closed-loop demands, not a matched-input actuator test.
Different observations, DR draws and engine dynamics lead to different actions.
MuJoCo's target repeatedly reverses at nearly the policy Nyquist frequency.
A finite-bandwidth PD/inertial system does not instantly attain every rapidly
alternating position target. Isaac also attenuates some target variation.

The MuJoCo steering gains in this episode are Kp=9.46787897, Kd=0.19125536;
Isaac's are 10.65173340 and 0.20918396. At rest, 0.04603606 Nm dry friction alone
corresponds to a position-error scale of only **0.2786 degrees** for this MuJoCo
Kp, versus observed 3.12–3.78 degree RMS dynamic tracking errors. This is a
quasi-static scale comparison, not a causal decomposition of the closed loop.

MuJoCo's 95th-percentile steering **controller** damping torque is 1.83965 / 0.90599 Nm
at 400/800 Hz. Passive **viscous** torque is only 0.008760 / 0.004314 Nm at the same
percentile. B is 0.476% of the episode's controller Kd. Passive damping is present
and much smaller than the explicitly intended controller damping.

Across all 64 environments/all 60 s, steering position RMS error is 0.0299553 rad
in Isaac, 0.0596396 at MuJoCo 400, 0.0593096 at MuJoCo 800. These are descriptive errors
under different demands, not isolated bandwidth rankings. Previously established
physics/PD-step sensitivity, hard speed projection versus PhysX's soft solver
constraint, contact and dry-friction solver differences remain relevant;
this experiment does not isolate their individual causal contributions.

## Graphs and evidence

- [Half-second force/tracking detail](friction_and_tracking_detail.png)
- [Five-second force/tracking view](friction_and_tracking_5s.png)
- [Exact statistics](statistics.json), [env 0 source traces](env0_traces.npz)
- [400 Hz verification](mujoco400/verification.json), [800 Hz verification](mujoco800/verification.json)
- [Read-only recorder](record_forces.py), [analysis](analyze_forces.py), [analysis log](analysis.log)

The new purple rear curve explicitly adds motor force, passive damping and
isolated dry friction. It is not total net generalized torque and not claimed
identical to Isaac's green sensor. Torque components are indexed by physics
interval start; actual steering positions are sampled at interval end. Curves
are unsmoothed, axes fitted independently, min/max labeled. This interval has
no spawns; the plotting helper retains red markers if a spawn is in view.

## Recommended next verification

Do not subtract 0.106 again from the MuJoCo motor command: joint friction is
already applied, so that would double-count it. First compare submitted motor
torque with submitted motor torque, and display passive forces separately.
For steering, use the previously proposed identical 50 Hz-held target replay
under comparable support/contact conditions, then compare PD torque, q, qdot,
overshoot and delay. A separate matched-state/input comparison with friction
enabled/disabled could isolate sensitivity to the friction solver; neither the
existing plots nor this logging-only repeat provides that causal A/B result.
No hardware test or production-control modification was made here.

Verification: both repeats check all 13 production arrays byte-for-byte and
all 64 runtime configurations. Passive viscous torque and dry-friction bounds
are checked at every step (4,608,000 new samples per motor across both runs).
Protected source/checkpoint/model/ledger hashes are unchanged. The original
report's **13 tests passed in 6.690 s**; syntax, Pyflakes, line length, image and
local-link checks also pass. See [verification.json](verification.json) and
[test log](original_report_tests.log).

## Primary references

- [MuJoCo 3.12 force decomposition and friction-loss constraints](https://mujoco.readthedocs.io/en/3.12.0/computation/index.html#friction-loss).
- [MuJoCo 3.12 joint damping and frictionloss](https://mujoco.readthedocs.io/en/3.12.0/XMLreference.html#body-joint-damping).
- [NVIDIA articulation joint-friction model](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/rigid_bodies_articulations/articulations.html#articulation-joint-friction).
