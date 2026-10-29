# Locomotive deployment (CycleRL riding policies)

Branch `locomotive` runs CycleRL's **riding** policies on this bike. The trained
tensor contract is kept exactly as trained; only the deployment motor contract and
hardware mapping are realigned to this repository's audited deploy stack. This is
a different task from Trackstand-Balance (stationary). Nothing here changes the
trackstand launch, nodes or default model.

## Models

Source task `CycleRL-Trackstand-Cylinder-Paper-v0`: the same `trackstand_rl_0916`
cylinder-wheel USD as this repo (geometry/mass/joint payloads hash-identical; this
repo's `base.usda` additionally authors tire friction 0.65/min). Ranked by
flat-ground MuJoCo balance success among runs with 100% Isaac success
(CycleRL `logs/skrl/reward_sweeps/cylinder_top5_{rerun,limitfix}/comparison.md`):

| Rank | ONNX (`models/`) | Recipe | Steer | v* range | MuJoCo BSR |
|---|---|---|---|---|---|
| 1 (default) | `locomotive_1900_drive03_2026-10-19_16-30-17.onnx` | 1900_drive03 | ±20° | 0.5–2.5 m/s | 100% |
| 2 | `locomotive_1900_2026-10-19_19-04-32.onnx` | 1900 | ±20° | 0.5–2.5 | 88.2% |
| 3 | `locomotive_1900_drive03_2026-10-20_03-18-29.onnx` | 1900_drive03 | ±20° | 0.5–2.5 | 82.5% |
| 4 | `locomotive_deploy_10deg_2026-10-20_04-32-20.onnx` | deploy_10deg | ±10° | 0.5–2.5 | 68.7% |
| 5 | `locomotive_act_refine_lo_2026-10-19_15-30-18.onnx` | act_refine_lo | ±20° | 0.3–2.5 | 64.7% |

These are paper-suite results (riding: balance success, steering/velocity tracking),
not trackstand metrics. Each file is self-contained (weights embedded) and listed in
`models/locomotive_policy_manifest.json` with checkpoint/ONNX SHA-256, its contract
(read from the run's saved `params/env.yaml`) and 16 reference input/output pairs.
The controller reproduces those pairs at startup and refuses a mismatching file.

Regenerate (Isaac Lab interpreter; deterministic, same hashes):

```bash
cd isaaclab_train
~/IsaacLab/_isaac_sim/python.sh scripts/deployment/export_cyclerl_locomotive_onnx.py \
  --output-dir ../deploy/source/trackstand_deploy/models
```

The exporter embeds skrl 2.1.0's `RunningStandardScaler` exactly
(`clamp((x − mean)/(sqrt(var) + 1e-8), ±5)`), checked against skrl's own class.
CycleRL's older `export_onnx.py` used `1/sqrt(var + eps)` with no clip, which
diverges beyond ±5σ; do not use its exports for these runs.

## Tensor contract (unchanged)

`[512, 256, 128, 64]` ELU MLP, Gaussian **mean** (raw, unbounded), 50 Hz
(trained `dt = 0.01`, decimation 2).

| # | Observation | This bike |
|---|---|---|
| 0 | `gravity_y` = projected_gravity_b[1] (−sin roll) | IMU quaternion (bike frame, as v1) |
| 1–3 | `ang_vel_x/y/z` (root_ang_vel_b) | IMU bike-frame gyro |
| 4 | `speed` (root_lin_vel_b x) | 0.07 m × rear A2 velocity (no-slip estimate) |
| 5 | `steer`, wrapped to [−π, π] | steering A3 position, fixed straight zero |
| 6 | `vel_cmd` v* | throttle stick → [v_min, min(v_max, `max_speed_cmd_m_s`)], slewed 1 m/s² |
| 7 | `yaw_rate_cmd` ωz* | yaw stick × 0.5 rad/s (stick right → negative/right turn) |

Actions: steering `a0 × scale` clipped to ±scale (20° or 10° per model) → MIT
position target; rear `a1 × 42.857` clipped ±45 rad/s → C1 speed target.
No observation LPF, speed floor or host steering slew (none were in training).

## Motor / hardware contract

| | Training (Isaac) | Bike (`motor_mit_velocity_io`) |
|---|---|---|
| Steering | ImplicitActuator Kp 10, Kd 0.2, 5.6 N·m | firmware MIT Kp 10, Kd 0.2 (unchanged backend) |
| Rear | ImplicitActuator τ = 5.0·(ω* − ω), clip 6.28 N·m | **C1 firmware speed PI**, B3 current ceiling = 6.28 N·m / B0 Kt |
| Rates | physics 100 Hz, policy 50 Hz | policy 50 Hz, MIT/C1 200 Hz, feedback 100 Hz |

- **Rear mismatch (unresolved):** firmware C1 PI gains (B8/B9) and acceleration
  (B5) are untouched and not identified as equal to the trained Kd = 5 P-law.
  Both are torque-capped stiff speed servos; transient response will differ.
- **B3:** startup reads B3 (observed default ≈ 82.5 A), writes
  min(existing, 6.28 / Kt) (≈ 20.9 A at Kt 0.30), and accepts acknowledgement and
  readback only ≤ request and at most 1 mA below. Never raised; volatile until
  power cycle; not restored on exit. `dry_run` only reads it.
- **Signs/IDs:** all from `params.yaml` (rear encoder sign −1). Both trainings use
  the same USD joints, so this repo's validated steering/rear signs apply unchanged;
  CycleRL's own bike config (SBUS, flipped IMU, `ros_positive_steer: left`) was for a
  different stack and is not used. C1 uses the A2 native frame, so the encoder sign
  converts both ways (as `drive_speed_sweep.py`).
- **Disable:** inherited unchanged — steering CF; rear C0 0 A then CF (coasts).
- **Topics:** commands on `/locomotive/motor_command`
  `[enable, steering_target_rad, rear_velocity_rad_s]`. The trackstand
  `/trackstand/motor_command` (torque) is never read by this backend. Control
  telemetry label `motor_control_mit_c1_v1`: its inherited
  `requested/bounded_drive_target_nm` fields carry rad/s here.
- **Sticks:** `rc_kickstand` publishes `/locomotive/rc_sticks` = `[throttle, yaw]`
  in [−1, 1] only when `throttle_channel`/`yaw_channel` are set (overlay: CH3/CH4,
  AETR — **unverified on this transmitter**).

## Launch → ride sequence

Training episodes start rolling at 1.0–2.5 m/s, so the policy never acts at rest.

1. **DOWN** — RC CH7 down: motors disabled, stands down.
2. **LAUNCH** — RC up and ready (IMU/motor/RC/sticks fresh, |roll| < 30°, fork
   within 5°, not rolling backward): stands stay down, steering held straight,
   rear speed ramps at 0.5 m/s² to max(stick v*, wake) + 0.1 m/s.
3. **RIDING** — measured speed ≥ `wake_speed_m_s` (1.0): stands up, policy steers
   and drives; v* starts at the hand-over speed.
4. **FAULT_LATCHED** — any readiness loss, inference error, motor fault, or speed
   < `min_riding_speed_m_s` (0.3): motors disabled; stands are **not** moved
   (same as trackstand). Only RC down clears it and lowers the stands.

Whether the bike can roll with its stands down during LAUNCH is **not verified**.

## Commissioning (each step needs the previous one)

```bash
cd ~/trackstand_rl/deploy && colcon build --symlink-install --packages-select trackstand_deploy
source install/setup.bash
P="--params-file source/trackstand_deploy/config/params.yaml --params-file source/trackstand_deploy/config/locomotive.yaml"
```

1. Sticks: `ros2 run trackstand_deploy rc_kickstand --ros-args $P`, then
   `ros2 topic echo /locomotive/rc_sticks`; throttle up → +1, yaw right → +1.
2. Data flow with `-p dry_run:=true` on `motor_mit_velocity_io`, plus IMU, RC,
   kickstand driver and `locomotive_controller`; check `/locomotive/status`,
   `/locomotive/policy_timing` (inference_ms) and the dry-run B3 log line.
3. Rear wheel **airborne**: live `motor_mit_velocity_io`, publish low
   `/locomotive/motor_command` speeds by hand; confirm forward rotation, the B3
   log, `drive_c1_*` telemetry and A2 tracking.
4. Supported rolling launch below wake speed, then a supported hand-over.

Never run another motor backend, `motor_mit_io` included, alongside it.
`ros2 launch trackstand_deploy locomotive.launch.py` is **live** (`dry_run: false`).

Switch models by setting `model_filename` **and** its manifest `model_sha256`.

## Open items

Rear C1-vs-trained-P transient mismatch; physical MIT steering torque cap
(see `MOTOR_MIT_IO.md`); unverified stick channels and rolling with stands
down; CycleRL's Isaac→MuJoCo drop (0–35% BSR) means sim success is not a
hardware green light; no MuJoCo locomotive mirror in this repo yet.
