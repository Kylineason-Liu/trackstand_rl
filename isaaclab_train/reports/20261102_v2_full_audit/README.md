# Trackstand-Balance-v2 full audit and fixes — 2026-11-02

**Confirmed numerical-containment and training-CLI bugs were fixed. The complete
104-test suite passes. Nominal and DR headless audits pass at the existing
50-Hz policy / 200-Hz integral / 1200-Hz physics defaults, including two actual
PPO updates per mode. An additional real training-entry smoke test passes.**

This is a functional/numerical audit of the current v2 training task, not an
evaluation of a trained balancing policy or a real-motor validation. No hardware
or CAN operation occurred. No old checkpoint or recorded experiment was edited.

## Confirmed issues and fixes

| Issue | Evidence before the change | Fix |
|---|---|---|
| Incomplete divergence detection | A NaN root position or joint position with finite velocities returns `diverged=False`; sanitizing the observation does not repair physical state. [Reproduction](before_numerical_gaps.json). | V2 additionally checks root position/quaternion, raw world velocities, joint positions, wheel-net angle and PI memory for non-finite values; a zero quaternion also terminates. Existing speed thresholds remain 20/50/300. |
| Non-finite terminal reward | A NaN root X makes the Gaussian drift reward NaN. RewardManager computes and accumulates rewards after terminations but **before** resetting the scene. | V2 maps only non-finite drift-reward results to zero. Every finite drift reward, its sigma and its weight stay identical. |
| `train.py --arp_w` has no effect on Trackstand | The CLI looked up `action_rate`, while v0/v1/v2 declare `rate_penalty`. | Preserve the existing alias and fall back to `rate_penalty`. Actual CLI training with `--arp_w 0.17` saves and uses weight **0.17**. Zero still disables the resolved term. |
| Existing regression suite was unreliable | Two baseline subcases failed because the comparison omitted v1's already-explicit `viscous_friction=0.0`. The DR config-only tests unnecessarily launched Kit; closing it ended discovery before subsequent classes ran. | Assert the intended None-to-zero difference explicitly; remove the unnecessary simulator lifecycle from configuration-only tests. All 104 tests now finish with `OK`. |

Numerical changes are isolated to the new
[numerics.py](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/numerics.py)
and v2's function overrides in
[trackstand_rl_env_cfg_v2.py](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_cfg_v2.py).
V0/v1 task implementations, physics, rewards and termination functions are
unchanged. The CLI alias fix is shared and therefore also makes the existing
flag work for v0/v1. README and regression tests document the intended behavior.

## Audited command and observation contracts

- Registry: distinct v2 environment class/config and bounded-Beta runner;
  separate `trackstand_rl_v2_rsl` checkpoint directory. The Gaussian and v1
  registrations remain unchanged.
- Observation: **11 channels**, gravity Y/Z, body angular velocity XYZ,
  steering/rear angle, steering/rear speed, steering sine/cosine. Raw xyzw
  quaternion math and current joint buffers agree with independently computed
  rotation-matrix observations. No linear velocities or finite +/-100 clamp.
  Observation processing occurs once per normal **50-Hz** policy step.
- Steering: Beta action in [-1,1] -> scale 1.57 -> clip +/-1.57 rad -> slew at
  **23.9 rad/s at 50 Hz** (0.478 rad/update) -> hold for 24 physics steps ->
  IdealPD `10*(q_target-q)-0.2*omega` at **1200 Hz** -> clip +/-5.6 Nm.
- Rear: Beta action in [-1,1] -> scale/clip +/-61.6 rad/s at **50 Hz** ->
  external trapezoidal integral, Ki **3.3 Nm/rad at 200 Hz**, with retained
  target-change memory and conditional anti-windup -> IdealPD
  `0.30*(v_target-v)+I` at **1200 Hz** -> clip combined torque +/-6.28 Nm.
  Position stiffness is zero. Integral state/previous error and target buffers
  clear only for selected resetting environments.
- Physics: native PhysX joint PD gains are zero for both explicit actuators;
  no duplicate native PD. Joint-friction properties match configuration:
  steering static/dynamic **0.046036056104 Nm**, viscous
  **0.000910713492 Nm·s/rad**; rear static/dynamic **0.106 Nm**, viscous **0**.
  Passive losses act in the plant, separately from actuator torque clipping.
  Backend actuator/submission code was also inspected.
- Rewards: finite-state values match
  `0.02*(5*alive + 2*exp(-drift^2/(2*0.25^2)) - 0.1*norm(raw-prev_raw))`.
  Only invalid terminal drift values acquire the finite fallback.
- Terminations: timeout 30 s, fallen about 30 degrees, divergence, drift 0.5 m,
  yaw deviation about 30 degrees and wheel-net 7.2 rad. All six stimuli are
  exercised. Wheel-net accumulates native 4pi-branch increments at physics rate
  before terminations; crossing the threshold terminates on that same step.
- Reset: failure, timeout, manual full/partial reset restore the selected joint
  positions/speeds, steering slew state, rear integral/targets, wheel-net
  reference and spawn heading. Other environments keep their memory. Ordinary
  timeout bootstrapping remains enabled; termination/truncation signals are not
  rewritten by these fixes.
- DR: opt-in, per-environment reset randomization retained. Controller-gain DR
  remains disabled; steering gains 10/0.2 and rear gains 0/0.30 stay fixed.

## Completed experimental checks

Both successful live runs use **64 environments, seed 42, default 1200-Hz physics**.
Each contains **256 bounded synthetic-action stress steps + three diagnostic
steps + 32 stochastic Beta-policy steps for two PPO updates = 291 policy steps**.

| Metric | Nominal | DR |
|---|---:|---:|
| Observation / action-process / reward calls | 291 each | 291 each |
| IdealPD evaluations per motor | 6,984 | 6,984 |
| Integral updates | 1,164 | 1,164 |
| Maximum independent pre-noise observation error | 9.54e-7 | 9.54e-7 |
| Stress-phase maximum steering actuator torque, Nm | 5.55553 | 5.60000 |
| Stress-phase maximum rear actuator torque, Nm | 6.28000 | 6.28000 |
| Stress-phase maximum rear pre-clip request, Nm | 36.64307 | 42.81634 |
| Stress-phase automatic resets | 620 | 874 |
| PPO value losses, updates 1 / 2 | 1.43177 / 0.84492 | 0.89750 / 0.54697 |

Reset counts are stress-test outcomes, not trained-policy survival scores.
All actor actions remain within [-1,1], losses/weights are finite, and the effort
equations and final clips are checked at every instrumented controller call.
Noise statistics reproduce the configured 0.001/0.0012 channel spreads.

The live recovery check injects invalid **cached** root/joint/PI/wheel-net state
after physics, before the actual termination/reward/reset managers run. All four
affected environments terminate; returned observations, rewards and episode sums
remain finite and state resets cleanly. It intentionally does not send NaNs into
native PhysX: recovery from a native solver/GPU fatal error is not established.

Successful captures:
[nominal summary](no_dr_pass/summary.json) / [raw NPZ](no_dr_pass/capture.npz) /
[log](no_dr_pass.log),
[DR summary](dr_pass/summary.json) / [raw NPZ](dr_pass/capture.npz) / [log](dr_pass.log).
The [104-test log](unit_tests.log) includes targeted invalid-state, reset,
PI/anti-windup, Beta, wheel-net, steering and configuration regressions.

The real `train.py` entry point additionally completed **two PPO iterations,
2,048 transitions**, with `--dr --arp_w 0.17`. Its
[resolved-config verification](cli_verification.json) confirms Beta, 11-channel
task configuration, 1200/50/200-Hz timing, Ki3.3, no gain DR and actual rate weight
0.17. [Training log](cli_smoke/train.log). The tiny model_0/model_1 checkpoints
in this isolated audit folder are smoke-test artifacts, not useful balancing
policies. Production training defaults/old checkpoints were not changed.

## Remaining limitations and deliberate non-changes

1. The policy still observes 11 channels, without the retained integral or held
   steering target. Those hidden controller states can limit a memoryless
   policy's observability. Adding controller state/history would change the
   requested observation contract and was not silently done.
2. The current passive-friction model is still an approximation. The recent
   [C1 friction recheck](../../../deploy/experiments/reports/20261102_rear_friction_recheck/README.md)
   supports about 0.100 Nm moderate-speed moving drag and unresolved positive B,
   with greater effective near-zero torque. Static thresholds, loaded behavior,
   torque-speed/thermal envelopes and hardware delay remain uncalibrated.
   The audit does not replace measured settings with a new guess.
3. Solver speed limits are soft constraints. Bounded targets and clipped actuator
   torque do not prove that instantaneous measured speeds or solver reaction
   efforts never exceed those nominal values.
4. The simulator's PI law does not identify the exact C1 firmware loop or real
   current dynamics. Ideal feedback, no CAN quantization/delay and different
   integral lifecycle details remain explicit modeling limits.
5. `--pure_survival` intentionally disables divergence containment, along with
   other task-envelope terms. These recovery claims concern the normal active
   divergence configuration. V1 torque-action policies and existing v1 hardware/
   MuJoCo consumers remain semantically incompatible with v2 velocity actions.

## Reproduce and preservation

From repository root, use unused output directories for live runs:

```bash
~/IsaacLab/_isaac_sim/python.sh -B -m unittest discover \
  -s isaaclab_train/tests -p 'test_*.py' -v
~/IsaacLab/_isaac_sim/python.sh -B \
  isaaclab_train/reports/20261102_v2_full_audit/validate_live.py \
  --headless --device cuda:0 --output /tmp/v2_audit_nominal_new
~/IsaacLab/_isaac_sim/python.sh -B \
  isaaclab_train/reports/20261102_v2_full_audit/validate_live.py \
  --headless --device cuda:0 --dr --output /tmp/v2_audit_dr_new
```

The CLI smoke ran from `cli_smoke/`, with `PYTHONPATH` set to the absolute
`isaaclab_train` directory and the absolute training script path, so its new
logs/checkpoints stay inside this report. The exact invocation and working directory are saved in
[cli_command.json](cli_command.json).

Pre-fix test failures are retained in `baseline_tests.log`. `after_tests.log`
records the initial all-tests attempt's sandbox GPU failure; `full_tests.log`
records the old config test ending discovery when it closed Kit. They are not
counted as full-suite passes. Three preliminary live harness attempts are kept
separately: a CPU/GPU expected-tensor mismatch, an incorrect `dt` hook keyword,
and an incomplete artificial PI-memory fixture. These were harness errors,
corrected before the successful nominal/DR runs; their failure JSONs are retained.

Python AST, Pyflakes, 120-column checks for new Python files and Git whitespace
checks pass. Ruff is unavailable. [Preservation snapshot](preservation_before.json),
[final verification](verification.json) and [artifact manifest](manifest.json)
record the exact changed/protected files and outcomes. Root progress records,
Git index and all other pre-existing task/training files remain unchanged apart
from the explicitly listed implementation, documentation and test edits.
