# Parallel Trackstand-Balance-v2 MuJoCo implementation

Implemented on 2026-11-03 alongside the existing v1 backend. **All 568
pre-existing MuJoCo files remain byte-identical** (502,647,216 bytes checked),
including scripts, source, models, policies, tests, old reports/logs and caches.
The new backend does not import or monkey-patch the v1 runtime. No training or
deployment source was changed. The coordinator completed the remaining CLI,
documentation and evaluation work after specialist turns hit their usage limit.

## New files and operating contract

- `src/newbike_mujoco_v2/`: independent contract, controller, environment, model
  loader, policy compatibility/inference, reporting, viewer group and CLI helpers.
- `scripts/run_deploy_viewer_v2.py` and `scripts/run_deploy_eval_v2.py`.
- `README_v2.md`, `requirements_v2.txt`, `tests/test_v2_contract.py` and
  `tests/test_v2_runtime.py`.

Default policy: no-DR **model_26600.pt**, run `2026-11-02_21-49-32`.
Policy SHA256: `5733e489490c6688c45085970fef46b62d1fe96a90fb7e5035c3b0de5699af7b`.
The verified 0916 MJCF geometry is loaded/configured in memory; no XML or mesh
file is rewritten. Initial repository revision and dirty state are in
[`baseline.json`](baseline.json).

| Stage | Rate | Behavior |
| --- | ---: | --- |
| Eleven observations / deterministic Beta means / targets | 50 Hz | Same ordering as v2; actions inherently bounded |
| Steering target | 50 Hz | ×1.57, clip ±1.57 rad, slew ±23.9 rad/s |
| Rear velocity target | 50 Hz | ×61.6, clip ±61.6 rad/s, hold 20 ms |
| Rear integral | 200 Hz | Trapezoidal Ki=3.3 N·m/rad; first tick primes history |
| Steering explicit PD | 1200 Hz | Kp=10, Kd=0.2; final clip ±5.6 N·m |
| Rear explicit velocity PI | 1200 Hz | clip(0.30×velocity_error + I, ±6.28 N·m) |
| Physics / passive losses | 1200 Hz | Rear frictionloss 0.106 N·m, passive damping 0 |

Integral changes that worsen total P+I saturation are frozen; unwinding remains
allowed. I is itself bounded and its stored error integral is corrected after
clipping. Previous-error history advances even after a rejected increment.
Ordinary target changes preserve memory. Episode reset clears targets and I,
primes fresh encoder references, and resets wheel-net accounting. Physics-step
rear position deltas update wheel-net before the current termination check.
Controller gains remain fixed under DR; inherited mass, CoM, tire, reset-state
and IMU randomization remain available. V2 invalid-state containment and finite
terminal drift reward are included.

The policy loader checks saved YAML or an explicit sidecar, since v1 torque and
v2 velocity actors have the same input/output shapes. Native MuJoCo logs are
isolated in new output runtime directories. Headless episode quotas are assigned
per environment, with inactive streams masked rather than sampled repeatedly.

## Completed validation

**22 focused tests pass**: see [`validation/v2_all_tests.log`](validation/v2_all_tests.log).
The earlier 20-test log retains one fixture assertion failure: float32 61.6 was
compared to Python double 61.6. Correcting that expected value required no motor
law change. AST syntax checks and Pyflakes passed for new production/tests and
the final verification helpers; Ruff is unavailable. Git whitespace checks pass.

Independent checks prepared by the analyst:

- [`independent_pi_check.json`](validation/independent_pi_check.json): exact
  AST-loaded Isaac `VelocityIntegral` versus the new scalar NumPy controller,
  **36,000 updates**, seven resets per case, five state fields, maximum error
  **0** using CPU float64. Includes current PI, I-bound and optional P-only cases.
- [`independent_policy_check.json`](validation/independent_policy_check.json):
  **1,024 observations**, CPU inference exactly matches direct checkpoint
  weights plus Isaac BetaMean; action error **0**. Same-shaped v1 metadata,
  nonfinite observations and wrong input dimensions are rejected.
- [`independent_runtime_check.json`](validation/independent_runtime_check.json):
  **192 actual physics steps / 32 I ticks**, submitted equations/timing, reset,
  eleven-observation math, wheel-net crossing, timeout, eight invalid-state
  cases and **20 DR resets** with fixed gains.

[`cli_validation.json`](validation/cli_validation.json) additionally verifies
both CLI help paths, relative input paths, restoration of caller CWD, exact
per-environment quotas (five diagnostic episodes over two environments), a
partial budget returning exit 2 with unfinished episodes excluded, and
fixed-step mode with no completions. A two-bike viewer executes three real
physics/scene-assembly intervals with a mocked passive window; **no real GUI
window was opened**.

## Learned-policy evaluation

All runs are CPU-only, seed **42**, all six standard terminations, default
**1200/50/200 Hz** physics/policy/I, unchanged reference geometry and controller
constants, default `project` speed approximation. One episode per seed stream
prevents a first-N-completions sampling bias.

| Test | Completed episodes | Mean length | Timeouts | Other endings |
| --- | ---: | ---: | ---: | --- |
| No DR, model_26600 | 64 | **30.00 s** | **64/64** | 0 |
| DR functional smoke, same no-DR-trained model | 4 | 1.075 s | 0/4 | 3 fallen, 1 yaw_deviation |

Nominal measured roll RMS **0.460756°**, maximum roll **1.274051°**, maximum
drift **0.144485 m**. This is one 30-second episode per environment, not the
earlier Isaac evaluation's two per environment. The four DR cases establish
execution of randomized configuration and controller wiring; they demonstrate
failure of this particular no-DR policy under those draws, not a robust DR
performance estimate or a reason to tune the port's constants.

See native logs [`no_dr_run.log`](validation/no_dr_run.log),
[`dr_smoke_run.log`](validation/dr_smoke_run.log), and their output directories
[`no_dr/`](validation/no_dr/) and [`dr_smoke/`](validation/dr_smoke/). These contain
episode metadata with sampled DR values, native summary JSON/log, histogram,
torque graph/arrays and the full physics-rate evaluation archive.

[`verification.json`](verification.json) reconstructs every recorded I tick
independently, checks scaling, held steering slew, measured velocity errors,
P+I and submitted effort clipping, and reconciles counts/quotas/histograms.
All maximum discrepancies are **0** in these captures. No-DR captures cover
**96,000 environment transitions / 2,304,000 physics steps / 384,000 I ticks**.
The DR smoke covers **215 / 5,160 / 860**, respectively. The same verifier
rehashes all pre-existing files against [`preservation_before.json`](preservation_before.json);
the matching snapshot is [`preservation_after.json`](preservation_after.json).

## Reproduction

Exact commands and exit status are retained in [`commands.json`](commands.json).
From `mujoco_sim2sim/`, the practical equivalent is:

```bash
python3 -B scripts/run_deploy_viewer_v2.py --num_envs 64 --seed 42
python3 -B scripts/run_deploy_eval_v2.py --num_envs 64 --episodes 64 --seed 42
```

Default outputs go to new `logs/v2/` paths. The validation commands instead used
this report's new output directories. Runtime: Python 3.10.12, MuJoCo 3.12.0,
NumPy 2.2.6, Torch 2.12.1+cu130 (CPU only). No GPU, training job, ROS node or
hardware command was started; existing hardware submodule changes were preserved.

## Interpretation limits

Matched policy/actuator interfaces do not imply identical simulator trajectories.
MuJoCo and PhysX contact, integration and passive-friction solvers differ. MuJoCo
has one sliding contact coefficient; the sampled dynamic coefficient is retained
as metadata. Default hard speed projection is stricter than PhysX's soft solver
constraint and can be disabled diagnostically. Native actuator force is commanded
electromagnetic effort before passive losses/contact, not solver net joint-axis
effort or independently measured shaft torque. No torque-speed/thermal envelope,
CAN delay, quantization or motor firmware loop was identified by this work.
