# Trackstand-Balance-v1 final pre-training audit

Date: 2026-10-21. **Recommendation: fix the confirmed timeout-bookkeeping bug
before a long training run.** Observation math, Beta wiring, control rates,
reward evaluation and short PPO updates passed the checks below. No runtime
source/configuration or progress record was changed by this audit.

## Findings, highest priority first

### 1. Confirmed: timeout resets leave old per-episode wheel bookkeeping

[`TrackstandRLEnv.step()`](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env.py)
clears `wheel_net_angle` and refreshes `spawn_yaw` only under
`self.reset_terminated`. The installed ManagerBasedRLEnv distinguishes
`reset_terminated` from `reset_time_outs`; both kinds reset the physical scene,
but timeout-only episodes do not enter the custom bookkeeping-clear mask.
V0 and v1 share this environment class: this is an inherited bug, not a new
observation/actuator change.

**Reproduced in actual headless physics, both no-DR and full-DR:** seed two old
episode accumulators with2rad and heading baselines0.123rad; force env0 to
timeout and env1 to fail drift containment. These are controlled diagnostic
sentinels, not a naturally observed2rad/0.123rad episode.

| State after automatic reset | Timeout-only env0 | Failure env1 | Expected |
| --- | ---: | ---: | ---: |
| All joint positions | 0 | 0 | 0 |
| All joint velocities | 0 | 0 | 0 |
| Steering slew target | 0 | 0 | 0 |
| Four encoder observation channels | 0 | 0 | 0 |
| `wheel_net_angle` | **2rad retained** | 0 | 0 |
| `spawn_yaw` | **0.123rad retained** | 0 | New spawn heading0 |

The normal configured spawn yaw is fixed0, so a heading-baseline leak may be
numerically invisible in ordinary runs. The wheel accumulator leak is real:
the next episode's wheel-net termination can depend on motion from an earlier
episode despite the encoder resetting to0. Explicit full `env.reset()` does
clear the fields. Repair should cover **all resetting environments**, including
timeouts, and have a targeted timeout regression test; no repair is applied here.

The wheel-net termination also uses the **previous policy step's accumulator**:
integration happens after `super().step()`. Integration is an endpoint-speed
sum at50Hz, not a400Hz sum or an exact native-angle difference. The20ms lag
is0.03rad at1.5rad/s but1.232rad at61.6rad/s, not universally negligible.

### 2. Confirmed: action-rate penalty is not symmetric normalized-action smoothing

[`balance_action_rate_penalty_norm()`](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/rewards.py)
reads ActionManager's **raw normalized** actions, then divides their changes
again by physical scales1.57/6.28. CPU checks with equal unit changes give:

| Change | Unweighted penalty | Actual weighted contribution at50Hz |
| --- | ---: | ---: |
| Steering only, Δa=1 | −0.6369426250 | −0.00127388514 |
| Rear only, Δa=1 | −0.1592356861 | −0.00031847137 |

Thus steering has approximately4x the penalty for equal normalized changes.
This is not a NaN/sign bug, and training can execute with it, but it is not the
unit-normalization claimed by the comments. Decide whether this relative
weighting is intended before starting a new campaign. It also penalizes raw
requests, not the post-slew steering target or applied motor torque.

### 3. Synthetic containment gap: position NaNs are not fully covered

The `diverged` term tests body linear/angular velocities and all joint speeds,
not root positions or joint positions. An isolated CPU test with non-finite
root position and otherwise finite velocities gives `diverged=False`,
`drifted=False`, and **NaN drift reward**. A non-finite joint position alone
also does not trigger this guard. No NaN occurred in the actual physics/PPO
checks; this is a reproduced edge-case vulnerability, not evidence of an
ongoing numerical failure.

Hardening should check appropriate full-state finiteness **and ensure rewards
remain finite on a bad terminal step**. Adding a termination check alone is
insufficient: the engine computes rewards before automatic reset. V1's
observation `nan_to_num` does not sanitize reward terms, and the default RSL-RL
`check_for_nan=True` would stop a run on a NaN reward.

### 4. Design boundaries, not newly discovered wiring failures

- `fallen` is a **lateral projected-gravity** threshold plus inversion test,
  not a30-degree total-tilt or general Euler-roll/pitch limit. Actual-function
  CPU cases: roll29° survives; roll31° terminates; pure pitch60° does **not**
  trigger `fallen`; inversion does. Keep this only if that is the intended definition.
- The policy is memoryless. Removed linear velocities, unknown DR parameters,
  and the unobserved previous slew target make the11-channel state partially
  observable. This does not prove learning failure or require undoing the
  chosen observation design, but a clean smoke test cannot establish sufficiency.
- Static friction equal to moving friction remains provisional; no calibrated
  electrical torque-speed/delay/thermal envelope is introduced by IdealPD.

## 1. Observation contract and calculation

The registered v1 term is
[`balance_imu_state_v1()`](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/observations.py),
with exactly this concatenation order:

| Channels,1-based | Value | Frame/unit |
| --- | --- | --- |
| 1–2 | gravity_y, gravity_z | `R(q)^T * [0,0,-1]`, body frame, dimensionless |
| 3–5 | ang_vel_x/y/z | `R(q)^T * root_link_ang_vel_w`, rad/s |
| 6 | steering_pos | Native steering joint coordinate, rad |
| 7 | drive_pos | Native rear joint coordinate, rad |
| 8 | steering_vel | Native steering joint speed, rad/s |
| 9 | drive_vel | Native rear joint speed, rad/s |
| 10–11 | sin(steering_pos), cos(steering_pos) | Dimensionless |

The raw quaternion is **xyzw**. There is no new manual wrapping or spawn-angle
subtraction; joint zero comes from the model/configured reset. No body linear
velocity is read. True gravity and angular velocity are transformed before
sensor corruption; gyro rates are body angular velocity, not Euler-angle rates.

With DR, per-episode tilt-channel bias std0.002 and gyro bias std0.0002rad/s
are added. They are explicitly marked placeholders in config, not newly measured
post-calibration constants. The term then maps NaN/±Inf to0, with **no finite±100
clamp**. The manager adds white Gaussian noise with std0.001/0.001 for gravity,
0.0012rad/s for each gyro axis, zero for the remaining six channels. White noise
remains enabled without DR. No observation latency/history/normalization is active.

Independent3x3 rotation-matrix reconstruction matched every monitored raw
observation: maximum absolute numerical difference4.77e−7 nominal /9.54e−7 DR.
Measured white-noise standard deviations were approximately0.001 on gravity and
0.0012 on gyro, with exactly zero noise on channels6–11.

## 2. Frequencies and bounded policy output

| Runtime operation | Simulated-time frequency | Observed calls over32 policy steps |
| --- | ---: | ---: |
| Observation computation for policy | 50Hz | 32 |
| Fresh raw policy action / action processing | 50Hz | 32 |
| Steering target-slew update | 50Hz | Once per action processing |
| Steering explicit torque computation | 400Hz | 256 |
| Rear explicit torque computation | 400Hz | 256 |
| Reward / termination evaluation | 50Hz | 32 each |

Physics dt0.0025s and decimation8 give policy dt0.02s. Startup/configuration and
explicit `get_observations()` calls also compute observations; they are not
additional periodic control samples. Terminations/rewards are evaluated after
the eight physics substeps, then resetting envs get fresh spawn observations.
These are simulated-time rates, not guarantees about wall-clock throughput.

Registration resolves the v1 config and RSL-RL Beta runner, not the v0 Gaussian.
The MLP produces four shape parameters for two Beta distributions:
`alpha=softplus(alpha_raw)+1`, `beta=softplus(beta_raw)+1`.
The two actual motor actions are `2*Beta.sample()-1` during rollout, or
`2*alpha/(alpha+beta)-1` for deterministic inference/export. These are inherently
bounded[-1,1] for valid finite network parameters. Internal four MLP outputs
are not the motor actions. No wrapper action clamp is configured; actor/critic
observation normalization is off. V0 checkpoints are not v1 Beta checkpoints.

Observed sampled ranges: nominal[-0.99578923,0.99841630],
DR[-0.99356538,0.99201787]. Each mode completed two real PPO updates,16 rollout
steps/update; all losses/parameters remained finite. Optimization evaluates
stored batches multiple times and is **not** another motor-command frequency.
This is a diagnostic of execution, not evidence of learned balancing ability.

## 3. Active rewards and terminations

Only these three reward terms are active. RewardManager multiplies every weight
by dt0.02, so configuration weights are not literal per-step reward amounts:

| Term | Unweighted expression | Weight | Per-policy-step contribution |
| --- | --- | ---: | --- |
| survival | 1 if asin(abs(gravity_y))<30°, else0 | 5 | 0.1 when alive |
| drift_gauss | exp(−d_xy²/(2×0.25²)) | 2 | Up to0.04 |
| rate_penalty | −sqrt(sum((Δraw_action/[1.57,6.28])²)+1e−8) | 0.1 | −0.002 times the norm |

There is no active roll-shaping, motor-effort/speed penalty or separate terminal
failure cost. The negative rate term has a positive weight, so its sign is
correct. Survival is flat inside its lean gate, not “more upright is better.”
Independent total-reward reconstruction matched actual RewardManager output
within2.98e−8 in both modes.

| Termination | Actual condition |
| --- | --- |
| time_out | episode counter>=1500, nominal30s |
| fallen | abs(projected gravity_y)>sin(0.5236), or gravity_z>0 |
| diverged | Non-finite body linear/angular velocity or joint speed; or norms>20m/s /50rad/s; or any joint speed magnitude>300rad/s |
| drifted | XY distance from environment origin>0.5m |
| yaw_deviation | Wrapped abs(yaw−spawn_yaw)>0.5236rad |
| wheel_net | abs(stored accumulated rear angle)>7.2rad, with the lag/reset issues above |

All six are active during ordinary training. `--pure_survival` is an evaluation
option, not the ordinary training configuration. The root-position drift
reference equals the spawn XY because reset XY offsets are fixed0. Tilt tests
use physical projected gravity, not noisy/bias-corrupted policy observations.

## 4. Reset and startup state

The scene reset restores **all three joint positions and speeds to0**, including
the passive front wheel. The steering action resets its previous target from
the freshly reset joint coordinate, clipped to±1.57rad; action histories clear
to0. This is restoration to a fixed configured joint zero, not measurement of
a new motor zero at each spawn. Both forced reset paths passed these physical/
action-state checks. The custom accumulator/heading fields are the exception
documented in finding1.

Root defaults: identity xyzw quaternion, position env_origin+(0,0,−0.016)m.
No-DR starts with zero roll/rates. `--dr` adds rollU(−0.1,0.1)rad and roll-rate
U(−0.25,0.25)rad/s, resamples configured dynamics/gains/IMU residuals per
environment/reset; pitch/yaw/other initial velocities remain0. Gravity_z and
cos(steering) are naturally−1/+1 at upright zero, not zero observations.

The training scripts call `learn(..., init_at_random_ep_len=True)`: initial
episode counters are randomized0–1499 for staggered timeouts. This does not
randomize motor zero, but first episodes can be shorter than30s and the timeout
bookkeeping bug can therefore appear early. DR is off by default; `--dr` enables
full strength, with no ramp. Start fresh, not `--resume` from a v0 checkpoint.

## Evidence, reproduction and scope

- [35 focused tests passed](unit_tests.log),2.597s, including Beta math/PPO/export,
  v0 isolation,11-observation units/layout/sanitization/no finite cap, actuator
  contracts and50Hz steering clip/slew/reset tests.
- [GPU integration summary](integration_summary.json) / [successful run log](integration_run.log):
  seed42,64env,32policy steps and two PPO updates **per nominal/DR mode**;
  [nominal](nominal_rollout.npz) / [DR](dr_rollout.npz) action/observation arrays.
- [CPU edge-case results](term_checks.json) / [log](term_checks.log) reproduce
  penalty weighting, fallen/yaw/wheel boundaries and non-finite-position gap.
- [Source integrity](source_integrity.json):71 protected source/script/test/
  README/progress-record files unchanged. Only new audit artifacts were added.
- [audit_v1.py](audit_v1.py) / [check_terms.py](check_terms.py). The initial
  [instrumentation setup failure](instrumentation_setup_failure.log) was the
  diagnostic wrapper's `**kwargs` signature being rejected by the manager;
  correcting the wrapper resolved it without changing task code.

From `isaaclab_train/`, using the installed Isaac Python:

```bash
~/IsaacLab/_isaac_sim/python.sh -B -m unittest discover -s tests -p 'test_trackstand_*.py' -v
~/IsaacLab/_isaac_sim/python.sh -B reports/20261021_v1_pretraining_audit/check_terms.py
~/IsaacLab/_isaac_sim/python.sh -B reports/20261021_v1_pretraining_audit/audit_v1.py \
  --audit_num_envs 64 --headless --device cuda:0
```

This audit does not repair the identified issues, start a training campaign,
write a policy checkpoint, validate long-run convergence or real-bike transfer,
or change verified bike geometry, MuJoCo, deployment or firmware settings.
