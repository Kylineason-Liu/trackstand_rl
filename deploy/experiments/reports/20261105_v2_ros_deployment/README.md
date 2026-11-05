# Parallel v2 ROS deployment implementation — 2026-11-05

Implemented directly, without subagents or hardware access. User target:
deploy Trackstand-Balance-v2 alongside the existing v1 path. Status:
**implemented and tested offline; Pi build/live actuation not tested here**.

## Delivered behavior

- `policy_controller_v2.py`: packaged DR model_50100 from
  `2026-11-02_22-40-01`, checksum and adjacent contract validation, eleven
  existing v2 observations and deterministic Beta means. Steering
  ×1.57/clip/slew remains 50 Hz; rear ×61.6 now means rad/s.
- `motor_mit_io_v2.py`: steering MIT position control; rear MIT Kp=0,
  Kd=0.30 with a Ki=3.3 N·m/rad host trapezoidal integral. Command/state/
  actuation rates are 50/100/200 Hz; I updates at 200 Hz with measured elapsed
  time and fresh cached A2. Target changes retain I; disable/fault resets it.
  Conditional anti-windup, wire quantization compensation and bounded I
  follow the previously tested v2 replay controller.
- Versioned `/trackstand/motor_command_v2` and mandatory velocity layout;
  v2 policy requires fresh backend status matching task/action/gains/rates.
  Existing v1 motor-state messages alone cannot authorize v2 arming.
- Same IMU/CRSF/servo configuration, CAN identities and signs, RC support
  authority, eleven-input observation construction and fault state machine.
  No rear C0/C1 motion commands or duplicate passive-friction subtraction.
- One CAN owner handles both motors. Steering B1 is verified once/startup;
  rear uses a fresh software A3 zero without B1. Rear A3 int32 rollover is
  handled continuously. Rear F1 position saturation is not used as an
  observation or position guard because position Kp is zero.
- Before temporary F0/CD writes, fsynced originals are retained in an
  exclusive journal. Rear velocity encoding covers the full ±61.6-rad/s
  contract with codec headroom; CD watchdogs are 500 ms. Both CF disables
  must be confirmed before restoration/readback; pending recovery blocks
  normal restart. Recovery-only mode disables/restores/exits without B1
  or motion. Persistent steering B1 is not restored.
- V2-specific control telemetry reports velocity requests/error, I/P,
  decoded rear MIT fields/F1, settings state and torque estimates. Existing
  seven-field motor_state_v3 and A1×B0 measured-current torque remain.
- `params_v2.yaml`, `trackstand_v2.launch.py`, console entries, self-contained
  ONNX/contract and [Pi instructions](../../../docs/V2_DEPLOYMENT.md).

The v2 nodes subclass the existing policy/MIT nodes. Backward-compatible
hooks reuse transport, query, feedback, fault and telemetry machinery.
They do not import the legacy C0/C2 node classes. V1 defaults remain intact;
the existing MIT base has intentionally been edited to provide these hooks.

## Acceptance evidence

| Criterion | Result | Evidence / limit |
| --- | --- | --- |
| Correct v2 model/action/observation contract | Pass offline | Real ONNX inference, hash/manifest checks, actual policy tick and unit tests |
| Rear MIT PI and lifecycle | Pass offline | Actual Isaac integral class compared across 1,000 updates, target changes, saturation and reset; codec/sign/range checks |
| Disable, re-arm, stale feedback and withdrawal | Pass offline | Actual nodes with deterministic ROS/CAN stand-ins; faults from either motor; RC-down between sends; rapid down/up |
| Persistent settings and shutdown | Pass offline | Real query/consume paths with firmware stand-in; full worker lifecycle, retained-mode CF/F1, restoration while loop stopped, crash recovery, journal failure/ownership |
| Packaging and hardware settings | Pass static/offline | Entry points/assets/config-node names; shared hardware YAML equality |
| V1 regression | Pass offline | Full package suite: **695 passed, 1 skipped**, including **46 new tests** |
| Source preservation | Pass | Of 62 captured pre-existing deployment files, 55 unchanged; seven intentional changes listed in verification.json; old launch/config byte-identical |
| ROS install / Pi timing / physical response | Not run | ROS 2 and bike hardware unavailable here |
| Kernel timestamp socket option test | Skipped | Sandbox prevents the socket option; retained as an explicit coverage limit |

Command from repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/source/trackstand_deploy \
  python3 -B -m pytest -q -rs -p no:cacheprovider \
  --junitxml=deploy/experiments/reports/20261105_v2_ros_deployment/pytest.xml \
  deploy/source/trackstand_deploy/test
```

See [pytest.log](pytest.log), [pytest.xml](pytest.xml),
[verification.json](verification.json) and [baseline hashes](before_sha256.json).
All 52 package Python sources parsed; `git diff --check` passed.
Ruff is unavailable, so no Ruff pass is claimed. No training/evaluation,
live CAN, ROS launch or physical balancing was performed.

## Audit fixes and remaining boundaries

The audit fixed a shutdown dependency: configuration queries previously
stopped receiving when the motion-loop flag went false. A restoration-only
receive allowance now works without re-enabling motion, and v2 permits up
to four seconds for bounded disable/restoration cleanup. Journal ownership
is committed only after successful exclusive creation/fsync; failed
atomic updates retain in-memory pending originals. Dry-run uses actual F0
for F1 decoding and separately planned F0 for command encoding.

**Not resolved by this implementation:** F0 is an encoding range, not a
verified firmware clamp on total internal MIT torque. Hardware rear P+I
and steering PD are not proven equivalent to simulated ±6.28/±5.6 N·m
clamps. V2 records estimated total versus the clipped simulation reference;
it does not claim that reference was applied. The host I bound alone
cannot cap the internal P term. Requested gains and wire-quantized gains
also differ slightly. Firmware PD timing, physical speed protections,
passive losses/contact, IMU fusion and host scheduling remain validation
boundaries. Prior poor v2 sim2sim survival is not fixed or re-evaluated here.

Next hardware work: build/source on the Pi, read-only observation and
backend-readiness checks, supported low-speed sign/PI-reset/timing checks,
then the existing identical-target v2 replay with independent support.
See the [guide](../../../docs/V2_DEPLOYMENT.md) before a powered launch.
