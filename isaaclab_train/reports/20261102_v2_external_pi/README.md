# V2 external PI velocity control and hardware comparison — 2026-11-02

V2 now uses **IdealPDActuator for velocity P plus an external, per-environment
integrator supplied through joint-effort feedforward**. It implements a PI law;
it does not identify the complete motor/driver dynamics. V0, v1, MuJoCo and the
Pi scripts/backend are unchanged.

The comparison is against
[`drive_mit_velocity_test.py`](../../../deploy/experiments/drive_mit_velocity_test.py)
and its shared transport/integrator in
[`drive_velocity_sine.py`](../../../deploy/experiments/drive_velocity_sine.py),
plus the plan/scheduler in
[`drive_c1_kp_identify.py`](../../../deploy/experiments/drive_c1_kp_identify.py).
Distinguish the script's **default Ki=0** from the completed hardware run using
**`--velocity-kp 0.30 --velocity-ki 3.3 --packet-rate 200`**. The latter is the
reference for v2's gains, not undocumented firmware B8/B9 unit guesses.

## Implemented command flow

```text
Beta rear action inherently in [-1, 1], produced at 50 Hz
  -> multiply by 61.6 rad/s
  -> explicitly clip target to [-61.6, 61.6] rad/s
  -> hold this target for 20 ms
  -> at 200 Hz: sample simulated speed; trapezoidally integrate target - speed
       first post-reset sample primes previous error and leaves I=0
       Ki=3.3 Nm/rad; conditional integration and I-state clamping
  -> write I torque into joint_effort_target (held between I updates)
  -> at 1600 Hz: IdealPD computes 0.30*(held target - current speed) + held I
  -> clip combined P+I to [-6.28, 6.28] Nm
  -> submit generalized rear-joint torque to PhysX
  -> passive joint friction/contact/inertia affect the resulting motion
```

The implementation is in
[`velocity_pi.py`](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/velocity_pi.py)
and
[`trackstand_rl_env_cfg_v2.py`](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_cfg_v2.py).
The installed Isaac Lab `IdealPDActuator.compute()` adds `joint_efforts` to its
position/velocity PD expression **before** `_clip_effort()`; no actuator subclass
or library modification is needed.

At each scheduled I update, with `e = target - speed`:

```text
proposed_I = Ki * (stored_integrated_error + 0.5*(e + previous_e)*0.005)
candidate_I = clip(proposed_I, -effort_limit, effort_limit)
candidate_total = actuator.damping * e + candidate_I
freeze if candidate_total is above/below its limit AND delta_I pushes farther out
accepted_I = previous_I when frozen, otherwise candidate_I
stored_integrated_error = accepted_I / Ki
previous_e = e                         # advance even when integration freezes
```

The first sample omits the trapezoidal increment. Ki=0 is an explicitly supported
P-only mode. The action reads live actuator damping/effort-limit tensors for
anti-windup, avoiding a second independent P-gain/effort-cap configuration.

## Controller-law and memory comparison

| Aspect | MIT experiment | Current v2 |
|---|---|---|
| Position controller | MIT position Kp exactly 0 | IdealPD stiffness exactly 0 |
| Velocity P | Requested 0.30 by default; 12-bit Kd code 246 gives **0.3003663004 Nm/(rad/s)**, about 0.1221% above the request | IdealPD damping **0.30 Nm/(rad/s)**; no gain quantization |
| Velocity I | `--velocity-ki` default **0**; tested PI run explicitly uses **3.3 Nm/rad** | `integral_gain_nm_per_rad=3.3` by default |
| Meaning of damping | MIT Kd is velocity-error gain | IdealPD damping is velocity-error gain, not passive resistance |
| Integral location | Host Python process on Pi | Action term outside IdealPDActuator, with tensor state for each environment/joint |
| Feedback used by I | Most recent received A2, converted to forward-positive rad/s | Current simulator joint velocity sampled on I-update ticks |
| Integral rule | Trapezoidal error integral using actual monotonic elapsed time | Same rule, deterministic simulated elapsed time: 5 ms |
| First sample | Sets previous error/time; I remains 0 | Sets previous error/initialized flag; I remains 0 |
| Ordinary target change | Retains I and previous error, including step baseline/pulse/return | Retains I and previous error across policy actions |
| Zero target/error | Zero target can accumulate braking I; zero error preserves existing I | Same; reaching a target does not force torque to zero |
| Reset boundary | Each complete test segment and motor disable, not each waveform phase | Each environment episode reset, including failure/timeout/manual partial reset |
| Reset contents | Stored integral, previous error/time and limited-update count | Same controller state, plus selected environments' processed velocity and backend velocity/effort targets |
| Multiple instances | One rear motor; one integral state | Separate state for every environment; resetting one never clears another |
| I bound | `min(F0_torque_range, 6.28 - encoded_Kd*F0_velocity_range/4095)` | Actuator effort limit: **6.28 Nm** |
| Codec headroom | Full-plan ±13-rad/s range reserves about **0.000953544 Nm** for velocity quantization; allowance about **6.279046456 Nm** | No wire codec, so no quantization reserve is needed |
| Anti-windup | Clamp I; freeze increments pushing estimated P+I farther into saturation; allow unwinding | Same conditional-integration structure, using the exact simulated P and cap |
| Hidden windup prevention | Correct the stored integrated error after clamping/freezing; advance previous error/time | Same; no hidden unbounded integral behind a clipped output |
| P+I request beyond 6.28 Nm | **Abort the experiment** when the estimated encoded request exceeds the guard; not a host final-torque clamp | IdealPD **clips the combined request** and continues the episode |
| Integral observation | Logged to CSV/JSON; no RL policy in this test | Available as `env.action_manager.get_term("drive").integral.torque_nm`; **not added** to the 11 actor observations |

The conditional-integration algorithms match structurally, not bit for bit at
their saturation boundaries, because hardware has encoded gains/codec headroom.
The unit test compares the **actual hardware script's integrator** with the new
tensor controller over 300 unsaturated samples, including target changes.

## Frequencies, target generation and measurement

| Aspect | MIT experiment | Current v2 |
|---|---|---|
| Source of targets | Generated holds, steps and sines, evaluated when a packet is due | Beta policy; a target is produced/processed once every 20 ms |
| Target range | Full default plan steady through ±10 rad/s; step/sine ranges come from the plan | Raw [-1,1] scaled/clipped to **±61.6 rad/s**, unchanged from earlier v2 |
| Source-target update rate | Typically 200 Hz in the recorded experiment; **not a 50-Hz-held policy sequence** | 50 Hz |
| MIT packet send rate | Default 200 Hz; selectable 50/200/400 | No CAN packets |
| External I update rate | At each attempted motion send, default 200 Hz | Configurable `integral_rate_hz`, default 200 Hz, eight physics steps per update |
| P-loop execution | Inside motor; exact firmware frequency/delay not established by the files | Every physics step, **1600 Hz** |
| I output between updates | Last FF packet held by the motor until replaced | Last I effort retained while IdealPD continues updating P |
| Duplicate position/velocity controller | Firmware MIT forms the P-plus-FF request before its current loop | Native PhysX joint PD is zero; the explicit IdealPD computes torque once per physics step |
| A1/A2 query rates | Each 100 Hz steady/zero, 400 Hz dynamic by default; selectable 100/200/400 | No queries; exact model speed available each physics step, subsampled for I |
| Other telemetry | A3 50 Hz, AE 10 Hz, B8/B9 1 Hz; F1 responses to MIT packets | Native joint state/actuator buffers; observations remain 50 Hz |
| Missing/delayed packets | Monotonic dt, stale cached speed possible, skipped late slots, no deliberate catch-up burst | Fixed simulation clock; no modeled CAN drops/jitter/acquisition delay |
| Feedback noise/filtering | Actual encoder/firmware estimation/filtering and asynchronous arrival | No encoder noise/filter/delay added to the controller input; IMU observation noise remains separate |
| Decimation requirement | Driven by host wall clock | Physics must be an integer multiple of I rate; action must execute each physics step; unsupported settings fail explicitly |
| Rate meaning | Host timestamps do not identify internal motor-loop frequency | Simulated integration rates do not imply any Pi packet-rate requirement |

The ±61.6-rad/s action range exceeds the experiment's ±13-rad/s F0 codec range.
The simulator preserves the existing v2 policy scale; this report does not
silently claim the experiment already validates those larger targets. An eventual
v2 hardware consumer must deliberately configure its wire range/action contract.

## Units, signs, encoder origins and CAN encoding

| Aspect | MIT experiment | Current v2 |
|---|---|---|
| Velocity/torque coordinates | User velocity rad/s, torque Nm; native rear velocity/torque signs **−1** | Model joint coordinates, rad/s and Nm; no CAN sign multiplication |
| Motor addressing | Rear ID 1, MIT motion **0x501**, feedback **0x001**; watches competing legal IDs including 0x401 | Joint name `rear_wheel_joint`, actuator group `drive` |
| Speed conversion | A2 signed int32 ×0.01 rpm ×2π/60 ×SIGN | Simulator-native joint speed; no rpm or gear conversion |
| Position reference | First post-configuration A3 defines logged multi-turn displacement; rollover handled; **no B1** | Reset events restore rear angle/speed to zero; wheel-net tracker refreshes per episode |
| Does rear position enter PI? | No: position Kp=0; A3 is diagnostic | No: stiffness=0; native rear angle remains an observation/bookkeeping input |
| Gain encoding | Kd range [0,5], 12 bits | Floating-point gain |
| Velocity encoding | F0-defined signed range, 12 bits; full-plan ±13 gives step 0.006349206 rad/s | Floating-point target/speed; no simulated MIT quantization |
| FF torque encoding | F0-defined range, 12 bits; observed ±18 Nm gives step 0.008791209 Nm and no exact-zero code | Integral Nm inserted directly into the effort buffer |
| Quantization compensation | Adjusts encoded velocity to cancel **FF quantization residual only**, preserving intended I | Unnecessary with ideal continuous effort/velocity inputs |
| Torque constant | B0 readback used for A1 current-derived torque diagnostics; not used to reinterpret raw B9 as Ki | Generalized torque applied directly; no B0/current conversion model |
| Zero-speed calibration | Does not subtract initial A2 or invent a velocity bias zero | Uses native joint velocity with `use_default_offset=False` |

Neither encoder coordinates nor this implementation establishes new physical
calibration. The earlier ~1.56–1.58% A2/A3 discrepancy remains uncorrected in the
hardware tools. Current-derived torque is not an independent shaft-torque sensor.

## Physical motor, limits and runtime behavior

| Aspect | MIT experiment | Current v2 |
|---|---|---|
| Measured speed limit | Abort if queried absolute speed exceeds 61.6 rad/s | Velocity target bounded at ±61.6; `velocity_limit` and `velocity_limit_sim` both 61.6. IdealPD does not itself clip measured speed; solver constraint is not a measured overspeed abort |
| Effort limit | Estimated request, A1×Kt and F1 torque checks against 6.28 Nm | `effort_limit=effort_limit_sim=6.28`, final P+I clamp; solver reaction effort need not equal electromagnetic command |
| Physical current limit | Existing B3 read/logged, no gain/current-limit tuning; observed 82.498 A | No separate electrical current limiter or current PI model |
| Friction | Actual assembly's losses; no fixed friction compensation added to command | Passive static **0.106 Nm** (provisional), dynamic **0.106 Nm**, viscous **0**, inherited from v1 |
| Physical I compensation | I can supply torque to overcome actual resistance | I can compensate simulated losses; friction remains in the plant and is not subtracted twice |
| Inertia | Real wheel/motor/assembly, not replaced by a fitted ideal inertia in the script | Asset/body inertias; no artificial armature added to hide instability |
| Torque-speed envelope, electrical/thermal dynamics | Whatever the real supply/motor/firmware produce, incompletely identified | Still not modeled by IdealPD; constant torque and solver speed limits are not a calibrated envelope |
| Fixture | Tested rear wheel airborne/free; steering independently disabled | Ground/contact bicycle task; live diagnostic briefly lifts the model solely for an isolated test |
| Other motor | Test controls rear only | Steering remains v1 position PD 10/0.2 with ±1.57 target, 23.9-rad/s target slew and ±5.6-Nm torque cap; its explicit PD now also runs at 1600 Hz |
| DR | No synthetic DR; hardware conditions are recorded | Controller P/I fixed, gain DR disabled; inherited mass/inertia, COM, materials, initial pose/rate and IMU DR remain opt-in |
| Watchdog/startup | CF disable, original F0/CD saved, ranges/watchdog read back; restore after confirmed disable | No CAN watchdog/arming/persistent register state; task reset initializes controller state |
| Other aborts | Driver faults, stale feedback, invalid packets, competing writers, bus/log failures; no tight tracking/angle/rest/lateness abort | Existing episode time/fall/divergence/drift/yaw/wheel-net terminations; reset clears I |
| CAN failure recovery | Existing ENOBUFS weakness remains; this change does not modify the Pi script | No network interface or transmit queue |
| Logging | Raw sent/received packets, state ages, I/error, P+I estimate, settings and cleanup metadata | Action I/error/limited-update tensors, actuator computed/applied effort and native state; this report adds diagnostic captures |

**Controller memory is represented but remains hidden from the actor.** The
11-observation contract is intentionally preserved. Adding I as an observation
or using a recurrent/history policy is a separate decision requiring matching
training/export/deployment changes.

The production Pi deployment remains the v1 policy and rear C0 torque path.
This task implements v2 simulation PI; it does not switch that backend to velocity
or deploy an existing torque-action checkpoint as a velocity-action policy.

## Why v2 physics changed from 400 to 1600 Hz

The first actual PI probe at the inherited 400 Hz was severely oscillatory,
despite an unchanged 1-rad/s target. Keeping that timestep would leave a known
numerical defect in the new controller. The new default is **dt=0.000625 s,
decimation=32**, preserving 20-ms policy steps and 200-Hz external I.

All probes use seed 42 and 64 actual v2 environments. For the isolated 0.4-s
response, all bodies are placed at z=3 m with zero velocities, retaining asset
geometry/inertia/friction and the selected DR; the model remains above 2 m.
Metrics use the last 0.2 s and are spans during the remaining settling response,
not a claim of exactly constant steady-state ripple.

| Metric | 400 Hz DR | 1600 Hz DR | 1600 Hz nominal |
|---|---:|---:|---:|
| Maximum late speed peak-to-peak across envs, rad/s | 107.929459 | 0.031131 | 0.031069 |
| Median late speed peak-to-peak, rad/s | 89.223549 | 0.031086 | 0.031069 |
| Mean late absolute speed error, rad/s | 43.905045 | 0.013027 | 0.013100 |
| Maximum applied torque, Nm | 6.28 | 0.30 | 0.30 |
| Maximum absolute I, Nm | 0.223883 | 0.105010 | 0.105014 |
| Grounded random-action automatic resets, 100 policy steps | 316 | 320 | 231 |
| Command equation/clamp, finite observations/rewards, partial/timeout resets | Pass | Pass | Pass |

The near-0.106-Nm I value is consistent with compensation for the modeled moving
friction. This is a controller/physics diagnostic, **not trained balancing
performance or an independently validated real-motor model**. The earlier
[P-only audit](../20261027_v2_pretraining_audit/README.md) already demonstrated
timestep sensitivity at P=0.2; the higher tested P here makes a smaller timestep
necessary. The experiment does not identify the real motor's internal P rate.

## Validation and reproduction

- Focused [PI tests](pi_tests.log): **21 passed**, including the real hardware
  integrator comparison, 200-Hz scheduling at multiple physics rates, retained
  memory, partial/full reset, final combined clamp, anti-windup/unwinding and
  corrected stored state without hidden windup.
- [Broader suite](full_tests.log): 76 test methods before the final added clamp
  test; two failing subcases of the existing v0/v1 comparison omit the known
  `viscous_friction=None` versus `0.0` difference. Those tests/sources were not
  altered. The failure was already recorded in the October 27 audit. This suite
  is **not represented as all passing**.
- Completed headless captures and full logs:
  [400-Hz DR](400_dr/summary.json), [1600-Hz DR](1600_dr/summary.json),
  [1600-Hz nominal](1600_nominal/summary.json). Each checks actual backend
  feedforward/velocity targets, all 100 grounded random steps and every automatic
  reset, plus an explicit partial reset and forced timeout. Native NPZ files
  preserve speed, I and applied torque for all 64 environments.
- Python AST/120-column and Git whitespace checks run. Ruff/Pyflakes are absent
  from the available Python environments; no claim of a successful Ruff run.
- No hardware/CAN operation, firmware change, trained-policy evaluation or new
  training occurred. A failed preliminary validation fixture passed int64 IDs
  to the installed material-reset API requiring int32; corrected in the harness.
  Earlier partial captures are not counted as completed checks.

From repository root, with an unused output directory:

```bash
~/IsaacLab/_isaac_sim/python.sh -B -m unittest discover \
  -s isaaclab_train/tests -p test_trackstand_velocity_v2.py -v
~/IsaacLab/_isaac_sim/python.sh -B \
  isaaclab_train/reports/20261102_v2_external_pi/validate_live.py \
  --headless --device cuda:0 --physics-hz 1600 --dr --output /tmp/v2_pi_new_dr
```

Omit `--dr` for nominal diagnostics. `--physics-hz 400` reproduces the adverse
comparison without editing task defaults. These commands do not evaluate a
trained policy. Before hardware use, validate policy-like **50-Hz-held** targets
through the chosen 200-Hz host-I/MIT path and then under defined supported load.
