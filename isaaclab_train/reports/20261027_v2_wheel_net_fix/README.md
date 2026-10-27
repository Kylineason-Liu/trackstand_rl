# V2 wheel-net bookkeeping fix — 2026-10-27

**Fixed for Trackstand-Balance-v2 only.** Wheel-net rotation now follows actual
joint-angle increments at physics rate, and the existing termination checks the
current accumulated value at the end of the same policy step. V0/v1 retain their
previous implementation. Motor gains, action/observation contracts, limits, reward
weights, termination thresholds, DR settings, and simulation frequencies were not
changed. The separate 400 Hz rear-loop stability finding remains unresolved.

## Implementation

- [wheel_net.py](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/mdp/stationary/wheel_net.py)
  adds WheelNetAngleTracker. It sums signed changes in native rear position,
  resolving the installed PhysX 4pi branch. It does not estimate displacement
  from the final speed. Its reference is copied, so mutable simulator buffers
  cannot silently move the reference.
- [trackstand_rl_env_v2.py](../../source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/trackstand_bike/trackstand_rl_env_v2.py)
  registers a supported POST_PHYSICS_STEP callback, reads fresh native positions
  directly from the physics view, and updates the tracker after every physics
  step. V2 calls ManagerBasedRLEnv.step directly, bypassing the inherited delayed
  speed integral. The ordinary manager retains all physics, actions, rewards,
  observations, recorder calls, termination checks and automatic resets.
- V2 _reset_idx resets only the selected environments' wheel-net/reference and
  spawn-yaw buffers after scene reset. The explicit reset delegates directly to
  the standard manager, supporting full and partial reset. Failure and timeout
  resets use the same path. A scene-reset teleport contributes no wheel motion.
- V2 close deregisters its callback before destroying the scene. The PhysX event
  import is deferred until environment construction, preserving CPU-only config
  and unit-test imports.

Current sequence:

```text
50 Hz policy action processing
  -> eight physics steps at 400 Hz
       -> after each step: unwrap actual rear-angle increment and accumulate it
  -> check abs(wheel_net_angle) > 7.2 rad using the current accumulated angle
  -> reset terminated/timed-out environments and refresh only their references
  -> return the next 50 Hz observation
```

The threshold is still checked at policy rate. There is no extra one-policy-step
delay, but this is not a new physics-rate termination or an excursion latch.
Tracking is signed net rotation, not total absolute distance traveled. The
native 4pi branch is resolved only for bookkeeping; the policy observation's
native rear-angle term is unchanged.

The unwrapping assumes less than 2pi motion per physics sample. At 400 Hz that
corresponds to approximately 2,513 rad/s, far above the configured motor speed
and 300 rad/s divergence threshold. No new speed cap or numerical-state guard
was added as part of this fix.

## Validation

**26 focused v2 tests passed**: 13 existing velocity/config tests and 13 new
[wheel-net tests](../../tests/test_trackstand_wheel_net_v2.py). They cover signed
motion, changing substep speeds, back-and-forth cancellation, branch crossings,
multiple turns, mutable input isolation, partial/full/empty reset, bypassing the
old post-step integral, yaw reset isolation, and callback cleanup.

An initial runtime probe independently compared post-physics callback positions
against native positions read immediately after sim.step returned. All **40**
samples agreed exactly (maximum difference **0 rad**), with nonzero actual motion.
The probe source/log/summary are retained here for reproducibility.

The new [headless regression](verify_wheel_net.py) ran the real task with seed
**42**, **64 environments**, both with and without DR. Each mode ran **256 bounded
synthetic action steps**, plus signed threshold, timeout, reset and wrap fixtures.
Per mode there were **260 policy steps / 2,080 physics tracking updates** and one
observation per normal step; four additional observations came from explicit
resets. There was no additional per-step observation computation.

At each termination check, an independent native policy-end angle difference
was compared with the physics-rate accumulated angle, before any automatic reset.
The old endpoint-speed increment was also computed on these same trajectories
as a comparison; it was not applied to the environment.

| Measurement | No DR | DR |
| --- | ---: | ---: |
| New accumulated-angle maximum error | 1.4305e-6 rad | 1.4305e-6 rad |
| New accumulated-angle RMS error | 1.5675e-7 rad | 1.5052e-7 rad |
| Old endpoint-speed increment maximum error on these trajectories | 0.434113 rad | 1.814674 rad |
| Old endpoint-speed increment RMS error | 0.082393 rad | 0.536585 rad |
| Automatic resets in synthetic action phase | 618 | 862 |

The previous audit's trajectories are not reused as a paired balance benchmark:
correct earlier wheel-net resets change later states and DR samples. The table
compares bookkeeping calculations against actual angle changes, not survival.

Both +7.2 and -7.2 rad boundaries terminated **on the first crossing policy step**:

| Fixture: start at +/-7.19 rad, initial/target rear speed +/-20 rad/s | No DR | DR |
| --- | ---: | ---: |
| Positive pre-reset net angle | +7.507138 rad | +7.478361 rad |
| Negative pre-reset net angle | -7.531284 rad | -7.508286 rad |
| Wheel-net termination on this step | Both true | Both true |
| Post-reset net angle and joint position/speed | Zero | Zero |

A third environment timed out in that same policy step; its wheel-net/reference
was also cleared. Explicit partial resets preserved the continuing environments'
angle, native reference and yaw buffers exactly.

Actual native branch crossings were tested with the wheels away from contact:
starting at +/-6.2 rad produced the following measured endpoints and net motion.

| Quantity | No DR, positive / negative | DR, positive / negative |
| --- | --- | --- |
| Native endpoint | -5.976995 / +5.976995 rad | -5.974689 / +5.988100 rad |
| Tracked physical increment | +0.389375 / -0.389377 rad | +0.391683 / -0.378272 rad |

Full resets excluded the fixture teleport, and close removed the callback from
the manager's registered callbacks. All normal observations/rewards were finite.
These are functional tests, not a trained-policy balance or hardware evaluation.

The full Trackstand suite ran **69 methods** and retained the two known failing
subcases of one unchanged legacy v0/v1 viscous-friction comparison. No new test
failure was introduced; the full suite is not claimed to be entirely passing.

## Artifacts and reproduction

- [No-DR summary](no_dr/summary.json), [DR summary](dr/summary.json), each with
  capture.npz and run.log.
- [Focused unit tests](unit_tests.log), [full suite](full_suite.log).
- [Freshness probe](callback_probe.py), [probe result](callback_probe.json),
  callback_probe.log.
- [Preservation and source hashes](verification.json).

From repository root, use new output directories:

```bash
~/IsaacLab/_isaac_sim/python.sh -B -m unittest discover \
  -s isaaclab_train/tests -p 'test_trackstand*_*v2.py' -v
~/IsaacLab/_isaac_sim/python.sh -B isaaclab_train/reports/20261027_v2_wheel_net_fix/verify_wheel_net.py \
  --headless --device cuda:0 --output /tmp/v2_wheel_net_nominal
~/IsaacLab/_isaac_sim/python.sh -B isaaclab_train/reports/20261027_v2_wheel_net_fix/verify_wheel_net.py \
  --headless --device cuda:0 --dr --output /tmp/v2_wheel_net_dr
```

Production changes are limited to the new tracker and v2 environment. The v2
registration test and training README were updated, with new focused tests and
this report. **72 protected existing source/script/test/ledger files remained
byte-identical**, including v0/v1, shared terminations and both progress ledgers.
The v2 actuator/config file was also verified unchanged. No trained checkpoint,
model geometry, MuJoCo or Raspberry Pi code was modified.
