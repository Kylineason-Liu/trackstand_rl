# IsaacLab v2 model_26600: startup and settling replay reference

Captured on 2026-11-03; report completed on 2026-11-04. Open the
[HTML gallery](index.html) or [first 10 seconds](capture_replay_first10s.png).

This is a new no-DR rollout of **Trackstand-Balance-v2**, using
`2026-11-02_21-49-32/model_26600.pt`, seed **42**, **64 environments**, all six
normal termination terms and the saved training configuration. All **64 first
episodes reached 1500 steps / 30 seconds**. Environment 0's first complete
episode supplies this reference. Ordinary observation white noise remains on.
The numerical configuration matches saved training; only evaluation environment
counts, output directory and copied checkpoint path differ.

Policy/targets: **50 Hz**. Physics and explicit PD: **1200 Hz**. External rear
integral: **200 Hz**. Initial motor positions, velocities and I state are all
zero. Motor arrays are ordered **[steering, rear]**.

## Commands versus measured motion

V2 has a **steering position target** and a **rear velocity target**. There is
no steering velocity command in the policy contract. As confirmed by the user,
hardware replay retains those meanings. Both actual joint velocities are also
captured for comparison; they are response data, not substituted commands.

| File | Contents | Timing |
| --- | --- | --- |
| [policy_targets_50hz.csv](policy_targets_50hz.csv) | Raw actions, scaled steering demand, final held steering position and rear velocity targets | 1500 rows, t=0 through 29.98 s; held through t=30 s |
| [joint_velocity_reference_1200hz.csv](joint_velocity_reference_1200hz.csv) | Actual motor positions and velocities, including the initial state | 36,001 rows, t=0 through 30 s |
| [motor_physics_1200hz.csv](motor_physics_1200hz.csv) | Pre/post-step joint state, targets, computed/applied/submitted torque, solver-axis diagnostic effort and rear I | 36,000 intervals; separate command/response timestamps |
| [motor_samples.npz](motor_samples.npz) | Canonical full capture, policy inputs, manager events and physics diagnostics | Full source arrays |
| [reference_metadata.json](reference_metadata.json) | Units, origin, target meanings, windows and file hashes | Portable source provenance |

The first targets are **−0.3866762817 rad steering** and **+0.44905698 rad/s
rear velocity**. These signs are Isaac joint coordinates; physical direction
requires the separately verified encoder/MIT mapping. No direction is inferred
from a graph or an assumed viewing orientation.

Replay uses `held_steering_position_target_rad` **after** Isaac clipping and
slew, plus `drive_velocity_target_rad_s`. Applying another slew filter would
change the test. Replaying recorded I/torque values would also change the
experiment: the real controller must compute its response from real feedback.

## Startup and later balancing

The recommended replay is the contiguous **0–10 s prefix**, preserving initial
conditions and PI memory. The complete 30 seconds remains available. Do not
skip directly to a later slice with zero I and a straight front wheel and call
it the same controller-state experiment.

| Source window | Steering velocity RMS | Rear velocity RMS |
| --- | ---: | ---: |
| 0–1 s, initial transient | 2.06256 rad/s | 1.17826 rad/s |
| 5–10 s, smaller ongoing balancing | 0.44863 rad/s | 0.39736 rad/s |

The measured steering peak is **17.30060 rad/s** at physics rate; the rear peak
is **1.93123 rad/s**. Policy-boundary-only sampling can miss the steering's
brief initial peak. “Settling” means lower-amplitude balancing motion here,
not a zero-motion equilibrium.

Figures are available as PNG/SVG/PDF: [full episode](capture_full_episode.png),
[startup](capture_startup.png), [replay prefix](capture_replay_first10s.png), and
[later balancing](capture_later_balancing.png). Red lines mark spawn; extrema
and SI units are labeled. Plots retain actual physics-rate response samples.

## Verification and preservation

[verification.json](verification.json) passes exact 24-step target holds,
motor equations and effort clipping/submission, six-step I holds, initial
state and CSV float32 round trips. All **542 original run files** and **86
training source/script/asset inputs** remain unchanged. The checkpoint copy is
SHA256 `5733e489490c6688c45085970fef46b62d1fe96a90fb7e5035c3b0de5699af7b`.
The canonical archive hash is
`3695e6331b646c1d2da8640c3dbb7e69564fe1dd04b41ba56e21d9c2c43bfbb9`.
Source/after manifests, resolved configurations and the successful run log are
retained alongside. [experiment.json](experiment.json) records versions and
the exact capture arguments. An initial sandbox startup failed before physics
because CUDA/cache access was unavailable; its log is preserved separately.
The authorized retry completed without changing production code or stopping
the separate user training process.

From `isaaclab_train/`, the completed capture command was:

```bash
/home/lmbike/IsaacLab/_isaac_sim/python.sh -B \
  reports/20261103_v2_26600_replay_capture/record_capture.py \
  --num-envs 64 --seed 42 --duration 30 --device cuda:0
```

This captures simulated balancing, not a real-bike trial. The hardware replay
code and portable copies are under `deploy/experiments/`; no CAN, ROS, serial
or real motor command was issued while preparing this reference. Airborne and
loaded motor replay conditions are different from a freely balancing bike.
